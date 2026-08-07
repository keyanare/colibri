/* DeepSeek-V4-Flash inference engine — 284B MoE (13B active), pure C.
 *
 * Sibling engine in the one-file-per-family pattern (colibri.c = GLM-5.2,
 * kimi_k3.c, inkling.c, olmoe.c). Shares st.h, json.h, tok.h, quant.h and the
 * two headers written for this architecture: dsv4.h (compute primitives, unit
 * tested) and dsv4_model.h (config + tensor manifest, unit tested).
 *
 *   make deepseek_v4
 *   ./deepseek_v4 <model_dir> "prompt" [--ngen N]
 *
 * ============================ STATUS ======================================
 * UNVERIFIED AGAINST THE MODEL. This was written from the published reference
 * implementation (`inference/model.py`, `inference/kernel.py`) and config.json
 * of deepseek-ai/DeepSeek-V4-Flash, without ever loading the 160 GB
 * checkpoint. The primitives in dsv4.h and the shapes in dsv4_model.h have
 * unit tests; everything in THIS file -- the loader, the caches, the layer
 * assembly, the decode loop -- has none, because a meaningful test of it needs
 * the weights. Expect bugs. The repo's merge gate is a token-exact oracle and
 * this does not have one.
 *
 * Known gaps, deliberate and named rather than papered over:
 *   - TOKENIZER. The HF repo ships no tokenizer.json (only an `encoding/`
 *     directory). Without one this engine cannot turn text into ids: it
 *     refuses at startup with an explicit message rather than inventing a
 *     vocabulary. A generator in the shape of tools/k3_tokenizer.py is the
 *     missing piece.
 *   - SPECULATIVE HEAD. compress_ratios carries trailing entries for it: one
 *     on the preview (num_nextn_predict_layers=1), three on -0731 (DSpark,
 *     mtp.0/1/2). The reference does not build it and its tensor names are
 *     unknown, so the head is not loaded and speculation is off.
 *   - PREFILL is a loop over single tokens. Correct, and slow; batching the
 *     prompt is the first optimization and needs no design change.
 *   - Batch size is 1 throughout, like every other engine in this repo.
 *
 * ============================ ARCHITECTURE ================================
 * Four things here are unlike anything else in the repo:
 *
 *  1. FOUR RESIDUAL STREAMS (mHC). The hidden state is [hc_mult][dim], not
 *     [dim]. Each sublayer is entered through dsv4_hc_pre (collapse to one)
 *     and left through dsv4_hc_post (expand back), mixed by a doubly
 *     stochastic matrix from Sinkhorn normalization.
 *
 *  2. K AND V ARE THE SAME VECTOR. One head_dim=512 latent per position
 *     serves all 64 heads as both key and value. That is why the KV cache is
 *     small enough to matter, and why the attention OUTPUT must be
 *     de-rotated: the value side carries RoPE.
 *
 *  3. HYBRID SPARSE ATTENTION. Every layer attends to a 128-token sliding
 *     window; layers with compress_ratio>0 additionally attend to compressed
 *     positions (one latent per `ratio` tokens). At ratio 4 (CSA) a learned
 *     indexer picks the top-512 of those; at 128 (HCA) all of them are used.
 *
 *  4. HASH ROUTING on the first 3 layers: experts are chosen by a table
 *     lookup on the INPUT TOKEN ID, not by the router. The working set is
 *     therefore known before the forward pass -- exact prefetch, and the one
 *     place in this repo where expert placement needs no prediction at all.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "st.h"
#include "tok.h"
#include "quant.h"
#include "dsv4.h"
#include "dsv4_model.h"
#include "omp_tune.h"
#include "route_trace.h"

/* ------------------------------------------------------------------ knobs */
static int   g_ngen        = 128;
/* Prefill chunk: how many prompt tokens share one pass of expert reads.
 * 32 matches kimi_k3's K3_CHUNK default. 1 restores the token-at-a-time path,
 * which is the fallback if batching is ever suspected -- results are identical
 * either way, only the read pattern differs. Bounded because the staging
 * buffers scale with it (chunk * topk * dim floats). */
static int   g_chunk       = 32;
static float g_temp        = 0.0f;    /* 0 = greedy */
static float g_expert_gb   = 2.0f;    /* GLOBAL streamed-expert cache budget.
                                       * Deliberately a byte budget shared by
                                       * every layer, not a per-layer slot
                                       * count: routed experts all have the
                                       * same footprint here, so one pool lets
                                       * whichever layers actually repeat their
                                       * routing keep their working set, and
                                       * -- unlike a per-layer count -- the
                                       * number the user sets is the number of
                                       * bytes they get. A per-layer 64 was the
                                       * first cut and meant 64 x n_layers
                                       * residents (~37 GB on this model),
                                       * which is not what "64" reads as. */
static int   g_max_seq     = 8192;    /* cache sizing; config's 1M would need
                                       * ~GBs of KV per layer and is not a
                                       * useful default for a first run */
static int   g_verbose     = 1;
static int   g_preflight   = 0;      /* --preflight: check the checkpoint, load nothing */
static int   g_ids_mode    = 0;      /* --ids: token ids given directly, no tokenizer */
static const char *g_dump  = NULL;   /* --dump-logits: oracle output for the fixture */
/* --direct: route expert reads through the twin O_DIRECT / F_NOCACHE fd
 * (st_direct_fd), bypassing the page cache. OFF by default: it is drive-
 * dependent (README: "measure DIRECT=1") and the coalesced read ALIGNS its
 * window so the Linux/Windows direct contract is honoured, but with no way to
 * A/B it here it stays a measured lever, not a promise. When OFF the WILLNEED
 * prefetch (overlap) is active; with --direct there is no page cache to warm,
 * exactly like kimi_k3.c's K3_DIRECT. */
static int   g_direct      = 0;

static double now_s(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}
static void *xalloc(size_t n, const char *what){
    void *p = n ? malloc(n) : NULL;
    if(n && !p){ fprintf(stderr,"OOM allocating %zu bytes for %s\n",n,what); exit(1); }
    return p;
}
static void *xzalloc(size_t n, const char *what){
    void *p = n ? calloc(1,n) : NULL;
    if(n && !p){ fprintf(stderr,"OOM allocating %zu bytes for %s\n",n,what); exit(1); }
    return p;
}

/* ------------------------------------------------------------------ weights
 * A weight is one of four storage classes, matching dsv4_model.h's roles. The
 * class decides which quant.h kernel runs; nothing else in the engine looks at
 * it. `s` is always f32 in memory: a ue8m0 sidecar is expanded once at load
 * (lossless -- see quant.h's ue8m0_decode), so the fp8 kernel needs no ue8m0
 * branch. FP4 is the exception and keeps its raw u8 exponents, because
 * matmul_mxfp4 decodes them inline with a bit trick. */
typedef struct {
    int role; int64_t O, I;
    uint8_t *b;      /* FP8: O*I e4m3 | FP4: O*ceil(I/2) nibbles */
    float   *s;      /* FP8: f32 block scales (expanded)          */
    uint8_t *e8;     /* FP4: O*ceil(I/32) ue8m0 exponents         */
    float   *f;      /* PLAIN: O*I floats                          */
    uint16_t *h;     /* PLAIN_BF16: O*I bf16, decoded in w_matmul  */
    int32_t *i32;    /* I32: O*I table entries                     */
} W;

static int64_t w_bytes(const W *w){
    switch(w->role){
        case DSV4_T_FP8: return w->O*w->I + dsv4_nblk128(w->O)*dsv4_nblk128(w->I)*4;
        case DSV4_T_FP4: return w->O*((w->I+1)/2) + w->O*((w->I+31)/32);
        case DSV4_T_I32: return w->O*w->I*4;
        case DSV4_T_PLAIN_BF16: return w->O*w->I*2;
        default:         return w->O*w->I*4;
    }
}

/* y[O] = W @ x[I]  (single row; S>1 handled by the kernels themselves) */
static void w_matmul(const W *w, float *y, const float *x, int S){
    switch(w->role){
        case DSV4_T_FP8:
            matmul_fp8(y,x,w->b,w->s,S,(int)w->I,(int)w->O); break;
        case DSV4_T_FP4:
            matmul_mxfp4(y,x,w->b,w->e8,S,(int)w->I,(int)w->O); break;
        case DSV4_T_PLAIN_BF16: {
            /* Kept bf16 in RAM, decoded on the fly. bf16->f32 is EXACT, so each
             * row decodes to the same f32 the PLAIN path would have materialized
             * at load -- the logits are bit-identical, and the resident set is
             * half the size. Deliberately scalar, like the PLAIN branch: the
             * dense matmuls here (head, compressor wkv/wgate) already accumulate
             * in double and are not the expert hot loop. */
            int O=(int)w->O, I=(int)w->I;
            const uint16_t *h=w->h;
            #pragma omp parallel for schedule(static)
            for(int o=0;o<O;o++){
                const uint16_t *r=h+(int64_t)o*I;
                for(int s=0;s<S;s++){
                    const float *xs=x+(int64_t)s*I; double a=0;
                    for(int i=0;i<I;i++) a+=(double)bf16_to_f32(r[i])*xs[i];
                    y[(int64_t)s*O+o]=(float)a;
                }
            }
            break;
        }
        default: {
            int O=(int)w->O, I=(int)w->I;
            #pragma omp parallel for schedule(static)
            for(int o=0;o<O;o++){
                const float *r=w->f+(int64_t)o*I;
                for(int s=0;s<S;s++){
                    const float *xs=x+(int64_t)s*I; double a=0;
                    for(int i=0;i<I;i++) a+=(double)r[i]*xs[i];
                    y[(int64_t)s*O+o]=(float)a;
                }
            }
        }
    }
}

static void w_free(W *w){
    free(w->b); free(w->s); free(w->e8); free(w->f); free(w->h); free(w->i32);
    memset(w,0,sizeof *w);
}

/* Load one tensor named by the manifest. The block-scale sidecar is named by
 * dsv4_scale_name -- `<stem>.scale`, the checkpoint's own convention, NOT
 * colibri.c's ".qs" and NOT the full name with ".scale" appended. */
static int w_load(shards *S, const DSV4Tensor *t, W *w, int required){
    memset(w,0,sizeof *w);
    if(!st_has(S,t->name)){
        if(required){ fprintf(stderr,"missing tensor: %s\n",t->name); exit(1); }
        return 0;
    }
    w->role=t->role; w->O=t->d0; w->I=t->d1?t->d1:1;
    char sn[192]; dsv4_scale_name(sn,sizeof sn,t->name);

    if(t->role==DSV4_T_FP8){
        int64_t nb=st_nbytes(S,t->name), want=w->O*w->I;
        if(nb!=want){ fprintf(stderr,"%s: %lld weight bytes, expected %lld for [%lld,%lld]\n",
            t->name,(long long)nb,(long long)want,(long long)w->O,(long long)w->I); exit(1); }
        w->b=(uint8_t*)xalloc((size_t)want,"fp8 weights");
        st_read_raw(S,t->name,w->b,0);
        int64_t nblk=dsv4_nblk128(w->O)*dsv4_nblk128(w->I);
        w->s=(float*)xalloc((size_t)nblk*sizeof(float),"fp8 block scales");
        if(!st_has(S,sn)){ fprintf(stderr,"%s: fp8 tensor has no .scale sidecar\n",t->name); exit(1); }
        int64_t ns=st_nbytes(S,sn);
        if(ns==nblk){                       /* ue8m0: one byte per block */
            uint8_t *raw=(uint8_t*)xalloc((size_t)nblk,"ue8m0 staging");
            st_read_raw(S,sn,raw,0);
            fp8_ue8m0_expand(w->s,raw,nblk);
            free(raw);
        } else if(ns==nblk*4){              /* f32 block scales */
            st_read_f32_cap(S,sn,w->s,nblk,0);
        } else {
            fprintf(stderr,"%s: scale sidecar is %lld bytes, expected %lld (ue8m0) or %lld (f32)\n",
                    sn,(long long)ns,(long long)nblk,(long long)(nblk*4)); exit(1);
        }
    } else if(t->role==DSV4_T_FP4){
        int64_t nb=w->O*((w->I+1)/2), ng=w->O*((w->I+31)/32);
        int64_t got=st_nbytes(S,t->name);
        if(got!=nb){ fprintf(stderr,"%s: %lld nibble bytes, expected %lld for [%lld,%lld]\n",
            t->name,(long long)got,(long long)nb,(long long)w->O,(long long)w->I); exit(1); }
        w->b=(uint8_t*)xalloc((size_t)nb,"mxfp4 nibbles");
        st_read_raw(S,t->name,w->b,0);
        if(!st_has(S,sn)){ fprintf(stderr,"%s: fp4 tensor has no .scale sidecar\n",t->name); exit(1); }
        int64_t ns=st_nbytes(S,sn);
        if(ns!=ng){ fprintf(stderr,"%s: %lld scale bytes, expected %lld (ue8m0 per 32)\n",
            sn,(long long)ns,(long long)ng); exit(1); }
        w->e8=(uint8_t*)xalloc((size_t)ng,"mxfp4 scales");
        st_read_raw(S,sn,w->e8,0);
    } else if(t->role==DSV4_T_I32){
        /* The hash table is an INDEX tensor, and torch's default index dtype is
         * int64 -- so it arrives 8 bytes wide unless the exporter narrowed it.
         * st_read_raw copies the tensor's own byte span into this buffer, which
         * is sized from the config, so the width must be settled BEFORE the
         * read: at 8 bytes/entry an unchecked read overruns by exactly 2x.
         * Both widths are accepted and narrowed here; every other span is a
         * refusal, because a table half-read is a routing table that silently
         * sends tokens to the wrong experts. */
        int64_t n=w->O*w->I, nb=st_nbytes(S,t->name);
        w->i32=(int32_t*)xalloc((size_t)n*4,"hash table");
        if(nb==n*4){
            st_read_raw(S,t->name,w->i32,0);
        } else if(nb==n*8){
            int64_t *raw=(int64_t*)xalloc((size_t)n*8,"i64 hash staging");
            st_read_raw(S,t->name,raw,0);
            for(int64_t i=0;i<n;i++){
                int64_t v=raw[i];
                if(v<INT32_MIN || v>INT32_MAX){
                    fprintf(stderr,"%s: entry %lld is %lld, not representable as int32\n",
                            t->name,(long long)i,(long long)v); exit(1); }
                w->i32[i]=(int32_t)v;
            }
            free(raw);
        } else {
            fprintf(stderr,"%s: %lld bytes for %lld entries — expected %lld (int32) or %lld (int64)\n",
                    t->name,(long long)nb,(long long)n,(long long)(n*4),(long long)(n*8)); exit(1);
        }
    } else if(t->role==DSV4_T_PLAIN_BF16){
        /* Unquantized bf16, kept bf16 in RAM (2 bytes/element); w_matmul decodes
         * on the fly. Must be actually-BF16: F16 is also 2 bytes/element but a
         * bf16->f32 decode of F16 bits is a silent misread, so the dtype is
         * refused, not guessed. */
        int64_t n=w->O*w->I, nb=st_nbytes(S,t->name), dt=st_dtype(S,t->name);
        if(nb!=n*2){ fprintf(stderr,"%s: %lld bytes, expected %lld for [%lld,%lld] bf16\n",
            t->name,(long long)nb,(long long)(n*2),(long long)w->O,(long long)w->I); exit(1); }
        if(dt!=0){  /* 0 = BF16; 1 = F16, 2 = F32 */
            fprintf(stderr,"%s: dtype code %lld, expected 0 (BF16)\n",
                    t->name,(long long)dt); exit(1);
        }
        w->h=(uint16_t*)xalloc((size_t)n*2,"bf16 weights");
        st_read_raw(S,t->name,w->h,0);
    } else {
        int64_t n=w->O*w->I;
        w->f=(float*)xalloc((size_t)n*sizeof(float),"plain weights");
        st_read_f32_cap(S,t->name,w->f,n,0);
    }
    return 1;
}

/* ------------------------------------------------------------------ model */

/* A Compressor's weights plus its rolling state. The state machine is the
 * subtle part: groups of `ratio` tokens are accumulated in kv_state/score_state
 * and collapse into one cached latent when the group closes. At ratio 4 the
 * group OVERLAPS the previous one, which is why the projections are twice as
 * wide (coff=2) and why the state holds 2*ratio slots. */
typedef struct {
    int ratio, coff, hd, active;
    W ape, wkv, wgate, norm;
    float *kv_state, *score_state;   /* [coff*ratio][coff*hd] */
    float *cache;                    /* [max_seq/ratio][hd] compressed latents */
    int    n_cached;
    float *freqs;                    /* rope table for the compressed stream */
    /* Per-step scratch, allocated once. Deliberately NOT alloca: this runs per
     * token per layer, and alloca is a portability trap across the toolchains
     * this repo builds on (MSVC/mingw spell it differently and it interacts
     * badly with the OpenMP regions above). */
    float *s_kv, *s_score, *s_pool_kv, *s_pool_sc, *s_out;
} Compressor;

typedef struct {
    int active;
    W wq_b, weights_proj;
    Compressor comp;                 /* the indexer's OWN compressor, at index_head_dim */
} Indexer;

typedef struct {
    /* mHC */
    W hc_attn_fn, hc_attn_base, hc_attn_scale;
    W hc_ffn_fn,  hc_ffn_base,  hc_ffn_scale;
    W attn_norm, ffn_norm;
    /* attention */
    W attn_sink, wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b;
    int ratio;
    Compressor comp;
    Indexer    idx;
    float *kv_cache;                 /* [window + max_seq/ratio][head_dim] */
    int64_t kv_cache_rows;
    float *freqs;                    /* this layer's rope table */
    /* MoE */
    W gate_w, gate_bias, tid2eid;
    int hash;
    W sh_w1, sh_w3, sh_w2;
} Layer;

/* One slot of the global routed-expert cache. Keyed by (layer, eid): every
 * routed expert in this model has identical dimensions, so a single pool can
 * hold experts from any layer without fragmentation. `base` is a 4K-aligned
 * slab that holds the expert's six byte segments (w1/w3/w2 weights + their
 * ue8m0 scales); the W views point into it, so a cache-miss is a pread that
 * overwrites the slab -- no malloc/free on the hot path. */
typedef struct {
    int layer, eid;
    W w1,w3,w2;
    uint64_t used;
    uint8_t *base; size_t base_cap;
} ESlot;

/* One routed expert's six on-disk byte spans, resolved ONCE at load so the
 * hot path does zero st_find and no byte-count validation. Order is the file
 * order: w1.weight, w1.scale, w3.weight, w3.scale, w2.weight, w2.scale.
 * The checkpoint (and the tiny fixture) packs them back-to-back per expert,
 * so `contig` collapses the whole expert to a SINGLE pread -- the batched
 * expert I/O this engine was doing six preads per expert before. */
typedef struct { int fd[6]; int64_t off[6]; int contig; } ExRef;

typedef struct {
    DSV4Cfg c;
    shards S;
    Tok tok;
    int have_tok;
    W embed, norm, head;
    W hc_head_fn, hc_head_base, hc_head_scale;
    Layer *L;
    ESlot *eslot; int n_eslot; int64_t expert_bytes;
    /* Per-expert on-disk geometry, built once at load (see ExRef). */
    ExRef *xref;
    int64_t e_w1p, e_w1s, e_w2p, e_w2s, e_slot;
    uint64_t clock;
    uint64_t expert_hits, expert_miss;
    double expert_load_s;
    float *embrow;   /* one embedding row, read per token instead of resident */
    /* scratch, sized once */
    float *x;        /* [hc_mult][dim] residual streams */
    float *xres;     /* [hc_mult][dim] */
    float *h;        /* [dim] sublayer input/output */
    float *hn;       /* [dim] normalized */
    float *q;        /* [n_heads][head_dim] */
    float *attn_out; /* [n_heads][head_dim] */
    float *kv;       /* [head_dim] */
    float *ffn_g, *ffn_u, *ffn_o;
    float *logits;
    int32_t *idxbuf; int idxcap;
    float *iscore;   /* indexer scores over compressed positions */
    int   *itop;
    /* Exact hash-layer prefetch scratch (see hash_prefetch): the union of
     * experts a batch of token ids will route to on one hash layer, collected
     * without any forward compute. */
    int *hpf_eid; unsigned char *hpf_seen;
} Model;

/* ------------------------------------------------------------------ math */

static void rmsnorm(float *out, const float *in, const float *w, int n, float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)in[i]*in[i];
    float r=1.0f/sqrtf((float)(ss/(double)n)+eps);
    for(int i=0;i<n;i++) out[i]=in[i]*r*(w?w[i]:1.0f);
}

/* ------------------------------------------------------------------ load */

static void compressor_init(Model *m, Compressor *C, shards *S, const char *prefix,
                            int ratio, int hd, int with_rope_table, float theta,
                            int original_seq_len){
    C->ratio=ratio; C->coff=dsv4_coff(ratio); C->hd=hd; C->active=1;
    DSV4Tensor t;
    #define LOADC(field,role_,d0_,d1_,suffix) do{ \
        memset(&t,0,sizeof t); t.role=(role_); t.d0=(d0_); t.d1=(d1_); \
        snprintf(t.name,sizeof t.name,"%s." suffix,prefix); \
        w_load(S,&t,&C->field,1); }while(0)
    LOADC(ape,   DSV4_T_PLAIN, ratio, (int64_t)C->coff*hd, "ape");
    /* PLAIN_BF16, not FP8 and not PLAIN: the compressor projections ship
     * unquantized bf16 with no .scale sidecar -- see dsv4_compressor_tensors.
     * They are consumed ONLY through w_matmul, so keeping them bf16 in RAM
     * halves the resident set (decoded on the fly, bit-exact). The manifest
     * and preflight already require exact bf16; loading as PLAIN would expand
     * to f32 and silently contradict them. */
    LOADC(wkv,   DSV4_T_PLAIN_BF16, (int64_t)C->coff*hd, m->c.dim, "wkv.weight");
    LOADC(wgate, DSV4_T_PLAIN_BF16, (int64_t)C->coff*hd, m->c.dim, "wgate.weight");
    LOADC(norm,  DSV4_T_PLAIN, hd, 1, "norm.weight");
    #undef LOADC
    size_t st_n=(size_t)C->coff*ratio*(size_t)C->coff*hd;
    C->kv_state   =(float*)xzalloc(st_n*sizeof(float),"compressor kv_state");
    C->score_state=(float*)xalloc(st_n*sizeof(float),"compressor score_state");
    for(size_t i=0;i<st_n;i++) C->score_state[i]=-INFINITY;
    int rows=g_max_seq/ratio+1;
    C->cache=(float*)xzalloc((size_t)rows*hd*sizeof(float),"compressed kv cache");
    C->n_cached=0;
    int wide=C->coff*hd, prows=C->coff*ratio;
    C->s_kv     =(float*)xalloc((size_t)wide*sizeof(float),"compressor scratch kv");
    C->s_score  =(float*)xalloc((size_t)wide*sizeof(float),"compressor scratch score");
    C->s_pool_kv=(float*)xalloc((size_t)prows*hd*sizeof(float),"compressor pool kv");
    C->s_pool_sc=(float*)xalloc((size_t)prows*hd*sizeof(float),"compressor pool score");
    C->s_out    =(float*)xalloc((size_t)hd*sizeof(float),"compressor out");
    if(with_rope_table){
        C->freqs=(float*)xalloc((size_t)(m->c.rope_head_dim/2)*sizeof(float),"compressor rope");
        dsv4_rope_freqs(C->freqs,m->c.rope_head_dim,theta,original_seq_len,
                        m->c.rope_factor,m->c.beta_fast,m->c.beta_slow);
    }
}

/* ---------------------------------------------------------------- stop ids
 * The generation loop ran to --ngen regardless of what the model said. That is
 * invisible on a base-style completion and wrong the moment a chat template is
 * used: the model closes its turn and then keeps going, inventing the next one.
 *
 * TWO SOURCES, because they disagree by design. config.json's eos_token_id is
 * the pretraining end-of-sequence (1 here); a DeepSeek chat turn ends with
 * <|EOT|>, which appears only in generation_config.json -- and there the key
 * may be a LIST. Collect from both and stop on any of them. */
#define DSV4_MAX_STOP 8
static int g_stop[DSV4_MAX_STOP], g_nstop;

static void stop_add(int id){
    if(id<0) return;
    for(int i=0;i<g_nstop;i++) if(g_stop[i]==id) return;
    if(g_nstop<DSV4_MAX_STOP) g_stop[g_nstop++]=id;
}
static void stop_from(jval *r, const char *key){
    jval *v=json_get(r,key);
    if(!v) return;
    if(v->t==J_NUM) stop_add((int)v->num);
    else if(v->t==J_ARR)
        for(int i=0;i<v->len;i++) if(v->kids[i]->t==J_NUM) stop_add((int)v->kids[i]->num);
}
static int is_stop(int id){
    for(int i=0;i<g_nstop;i++) if(g_stop[i]==id) return 1;
    return 0;
}

static void model_load(Model *m, const char *dir){
    double t0=now_s();
    st_init(&m->S,dir);

    /* config */
    char cp[2100]; snprintf(cp,sizeof cp,"%s/config.json",dir);
    FILE *f=fopen(cp,"rb");
    if(!f){ fprintf(stderr,"cannot open %s\n",cp); exit(1); }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<=0||n>(64L<<20)){ fprintf(stderr,"%s: implausible size %ld\n",cp,n); exit(1); }
    char *buf=(char*)xalloc((size_t)n+1,"config");
    size_t got=fread(buf,1,(size_t)n,f); buf[got]=0; fclose(f);
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    if(!root||dsv4_cfg_from_json(root,&m->c)!=0){ fprintf(stderr,"config.json rejected\n"); exit(1); }
    stop_from(root,"eos_token_id");
    /* generation_config.json is optional and, where it exists, authoritative
     * about how a TURN ends rather than how a document ends. */
    { char gp[2100]; snprintf(gp,sizeof gp,"%s/generation_config.json",dir);
      FILE *gf=fopen(gp,"rb");
      if(gf){
          fseek(gf,0,SEEK_END); long gn=ftell(gf); fseek(gf,0,SEEK_SET);
          if(gn>0 && gn<(1L<<20)){
              char *gb=(char*)xalloc((size_t)gn+1,"generation_config");
              size_t gg=fread(gb,1,(size_t)gn,gf); gb[gg]=0;
              char *ga=NULL; jval *groot=json_parse(gb,&ga);
              if(groot) stop_from(groot,"eos_token_id");
              free(ga); free(gb);
          }
          fclose(gf);
      } }

    /* tokenizer: refuse loudly rather than invent a vocabulary. Skipped
     * entirely in --ids mode, which is how the synthetic-fixture oracle drives
     * this engine (tools/make_tiny_dsv4.py) -- the same shape as inkling's
     * tiny-model gate, and the reason that gate does not need a vocabulary. */
    char tp[2100]; snprintf(tp,sizeof tp,"%s/tokenizer.json",dir);
    FILE *tf=fopen(tp,"rb");
    if(tf){ fclose(tf); tok_load(&m->tok,tp); m->have_tok=1; }
    else if(g_ids_mode){ m->have_tok=0; }
    else {
        fprintf(stderr,
          "%s/tokenizer.json is missing.\n"
          "  DeepSeek-V4-Flash ships a raw `encoding/` directory, not a HF tokenizer.json.\n"
          "  A generator (in the shape of c/tools/k3_tokenizer.py) has to write one first;\n"
          "  this engine will not guess a vocabulary.\n", dir);
        exit(1);
    }

    const DSV4Cfg *c=&m->c;
    if(g_verbose)
        fprintf(stderr,"  config: %d layers, dim %d, %d heads x %d, %d experts (top-%d), %d hash layers\n",
                c->n_layers,c->dim,c->n_heads,c->head_dim,c->n_routed_experts,
                c->n_activated_experts,c->n_hash_layers);

    /* globals */
    DSV4Tensor g[16]; DSV4List GL={g,0,16,0};
    dsv4_global_tensors(&GL,c);
    if(GL.overflow){ fprintf(stderr,"internal: global tensor list overflowed\n"); exit(1); }
    for(int i=0;i<GL.n;i++){
        /* embed.weight is NOT made resident: exactly one row is used per token,
         * so a 2.1 GB f32 expansion of [vocab,dim] buys nothing. It is read as
         * a slice at forward time instead (st_read_slice_f32, one ~16 KB pread
         * per token, which is noise next to the expert traffic). */
        if(!strcmp(g[i].name,"embed.weight"))         continue;
        else if(!strcmp(g[i].name,"norm.weight"))     w_load(&m->S,&g[i],&m->norm,1);
        else if(!strcmp(g[i].name,"head.weight"))     w_load(&m->S,&g[i],&m->head,1);
        else if(!strcmp(g[i].name,"hc_head_fn"))      w_load(&m->S,&g[i],&m->hc_head_fn,1);
        else if(!strcmp(g[i].name,"hc_head_base"))    w_load(&m->S,&g[i],&m->hc_head_base,1);
        else if(!strcmp(g[i].name,"hc_head_scale"))   w_load(&m->S,&g[i],&m->hc_head_scale,1);
    }

    m->L=(Layer*)xzalloc((size_t)c->n_layers*sizeof(Layer),"layers");
    int hs=c->n_heads*c->head_dim;
    int hcmix=(2+c->hc_mult)*c->hc_mult;

    for(int l=0;l<c->n_layers;l++){
        Layer *L=&m->L[l];
        L->ratio=c->compress_ratios[l];
        L->hash = l < c->n_hash_layers;
        DSV4Tensor t;
        #define LOADL(field,role_,d0_,d1_,fmt_) do{ \
            memset(&t,0,sizeof t); t.role=(role_); t.d0=(d0_); t.d1=(d1_); \
            snprintf(t.name,sizeof t.name,fmt_,l); \
            w_load(&m->S,&t,&L->field,1); }while(0)

        LOADL(hc_attn_fn,   DSV4_T_PLAIN, hcmix, (int64_t)c->hc_mult*c->dim, "layers.%d.hc_attn_fn");
        LOADL(hc_attn_base, DSV4_T_PLAIN, hcmix, 1, "layers.%d.hc_attn_base");
        LOADL(hc_attn_scale,DSV4_T_PLAIN, 3, 1,     "layers.%d.hc_attn_scale");
        LOADL(hc_ffn_fn,    DSV4_T_PLAIN, hcmix, (int64_t)c->hc_mult*c->dim, "layers.%d.hc_ffn_fn");
        LOADL(hc_ffn_base,  DSV4_T_PLAIN, hcmix, 1, "layers.%d.hc_ffn_base");
        LOADL(hc_ffn_scale, DSV4_T_PLAIN, 3, 1,     "layers.%d.hc_ffn_scale");
        LOADL(attn_norm,    DSV4_T_PLAIN, c->dim, 1,"layers.%d.attn_norm.weight");
        LOADL(ffn_norm,     DSV4_T_PLAIN, c->dim, 1,"layers.%d.ffn_norm.weight");

        LOADL(attn_sink, DSV4_T_PLAIN, c->n_heads, 1,          "layers.%d.attn.attn_sink");
        LOADL(wq_a,      DSV4_T_FP8,   c->q_lora_rank, c->dim, "layers.%d.attn.wq_a.weight");
        LOADL(q_norm,    DSV4_T_PLAIN, c->q_lora_rank, 1,      "layers.%d.attn.q_norm.weight");
        LOADL(wq_b,      DSV4_T_FP8,   hs, c->q_lora_rank,     "layers.%d.attn.wq_b.weight");
        LOADL(wkv,       DSV4_T_FP8,   c->head_dim, c->dim,    "layers.%d.attn.wkv.weight");
        LOADL(kv_norm,   DSV4_T_PLAIN, c->head_dim, 1,         "layers.%d.attn.kv_norm.weight");
        LOADL(wo_a,      DSV4_T_FP8,   (int64_t)c->o_groups*c->o_lora_rank, hs/c->o_groups,
                                                               "layers.%d.attn.wo_a.weight");
        LOADL(wo_b,      DSV4_T_FP8,   c->dim, (int64_t)c->o_groups*c->o_lora_rank,
                                                               "layers.%d.attn.wo_b.weight");

        /* Two rope bases: compressed layers use compress_rope_theta WITH YaRN,
         * window-only layers use rope_theta with YaRN disabled. */
        L->freqs=(float*)xalloc((size_t)(c->rope_head_dim/2)*sizeof(float),"layer rope");
        if(L->ratio) dsv4_rope_freqs(L->freqs,c->rope_head_dim,c->compress_rope_theta,
                                     c->original_seq_len,c->rope_factor,c->beta_fast,c->beta_slow);
        else         dsv4_rope_freqs(L->freqs,c->rope_head_dim,c->rope_theta,
                                     0,c->rope_factor,c->beta_fast,c->beta_slow);

        L->kv_cache_rows = c->window_size + (L->ratio ? g_max_seq/L->ratio+1 : 0);
        L->kv_cache=(float*)xzalloc((size_t)L->kv_cache_rows*c->head_dim*sizeof(float),"kv cache");

        if(L->ratio){
            char pfx[96]; snprintf(pfx,sizeof pfx,"layers.%d.attn.compressor",l);
            compressor_init(m,&L->comp,&m->S,pfx,L->ratio,c->head_dim,1,
                            c->compress_rope_theta,c->original_seq_len);
            if(dsv4_has_indexer(L->ratio)){
                L->idx.active=1;
                memset(&t,0,sizeof t); t.role=DSV4_T_FP8;
                t.d0=(int64_t)c->index_n_heads*c->index_head_dim; t.d1=c->q_lora_rank;
                snprintf(t.name,sizeof t.name,"layers.%d.attn.indexer.wq_b.weight",l);
                w_load(&m->S,&t,&L->idx.wq_b,1);
                memset(&t,0,sizeof t); t.role=DSV4_T_PLAIN;
                t.d0=c->index_n_heads; t.d1=c->dim;
                snprintf(t.name,sizeof t.name,"layers.%d.attn.indexer.weights_proj.weight",l);
                w_load(&m->S,&t,&L->idx.weights_proj,1);
                char ip[128]; snprintf(ip,sizeof ip,"layers.%d.attn.indexer.compressor",l);
                compressor_init(m,&L->idx.comp,&m->S,ip,4,c->index_head_dim,1,
                                c->compress_rope_theta,c->original_seq_len);
            }
        }

        memset(&t,0,sizeof t); t.role=DSV4_T_PLAIN; t.d0=c->n_routed_experts; t.d1=c->dim;
        snprintf(t.name,sizeof t.name,"layers.%d.ffn.gate.weight",l);
        w_load(&m->S,&t,&L->gate_w,1);
        if(L->hash){
            memset(&t,0,sizeof t); t.role=DSV4_T_I32;
            t.d0=c->vocab_size; t.d1=c->n_activated_experts;
            snprintf(t.name,sizeof t.name,"layers.%d.ffn.gate.tid2eid",l);
            w_load(&m->S,&t,&L->tid2eid,1);
            /* w_load settles the WIDTH; the expert grid is only known here. An
             * out-of-range id indexes the expert array on the hot path, where
             * it would be an OOB read rather than a wrong answer. */
            for(int64_t i=0,ne=(int64_t)c->vocab_size*c->n_activated_experts;i<ne;i++){
                int e=L->tid2eid.i32[i];
                if(e<0 || e>=c->n_routed_experts){
                    fprintf(stderr,"layers.%d.ffn.gate.tid2eid[%lld]=%d, outside [0,%d)\n",
                            l,(long long)i,e,c->n_routed_experts); exit(1); }
            }
        } else {
            memset(&t,0,sizeof t); t.role=DSV4_T_PLAIN; t.d0=c->n_routed_experts; t.d1=1;
            snprintf(t.name,sizeof t.name,"layers.%d.ffn.gate.bias",l);
            w_load(&m->S,&t,&L->gate_bias,1);
        }
        if(c->n_shared_experts){
            memset(&t,0,sizeof t); t.role=DSV4_T_FP8; t.d0=c->moe_inter_dim; t.d1=c->dim;
            snprintf(t.name,sizeof t.name,"layers.%d.ffn.shared_experts.w1.weight",l);
            w_load(&m->S,&t,&L->sh_w1,1);
            snprintf(t.name,sizeof t.name,"layers.%d.ffn.shared_experts.w3.weight",l);
            w_load(&m->S,&t,&L->sh_w3,1);
            memset(&t,0,sizeof t); t.role=DSV4_T_FP8; t.d0=c->dim; t.d1=c->moe_inter_dim;
            snprintf(t.name,sizeof t.name,"layers.%d.ffn.shared_experts.w2.weight",l);
            w_load(&m->S,&t,&L->sh_w2,1);
        }
        #undef LOADL

        if(g_verbose && (l%8==0 || l==c->n_layers-1))
            fprintf(stderr,"\r  loading resident weights: layer %d/%d",l+1,c->n_layers);
    }
    if(g_verbose) fprintf(stderr,"\r  resident weights loaded in %.1fs%20s\n",now_s()-t0,"");

    /* Global expert cache. One routed expert is w1+w3+w2 with their ue8m0
     * sidecars; the budget in GB divides by that to give the slot count, so
     * the number the user sets is the number of bytes they actually get. */
    {
        int64_t MI=c->moe_inter_dim, D=c->dim;
        m->expert_bytes = 2*(MI*((D+1)/2) + MI*((D+31)/32))
                        +   (D*((MI+1)/2) + D*((MI+31)/32));
        int64_t budget=(int64_t)(g_expert_gb*1e9);
        m->n_eslot=(int)(budget/m->expert_bytes);
        /* Floor of 1 per layer: below that a single token's routing evicts
         * entries it is still going to need in the same forward pass, and the
         * cache stops being a cache. */
        if(m->n_eslot < c->n_layers) m->n_eslot = c->n_layers;
        m->eslot=(ESlot*)xzalloc((size_t)m->n_eslot*sizeof(ESlot),"expert cache");
        for(int s=0;s<m->n_eslot;s++){ m->eslot[s].layer=-1; m->eslot[s].eid=-1; }
        int per_token = c->n_activated_experts*c->n_layers;
        if(g_verbose)
            fprintf(stderr,"  expert cache: %d slots x %.1f MB = %.1f GB "
                    "(a token routes %d experts = %.1f GB)\n",
                    m->n_eslot,(double)m->expert_bytes/1e6,
                    (double)m->n_eslot*m->expert_bytes/1e9,
                    per_token,(double)per_token*m->expert_bytes/1e9);
        if(m->n_eslot < per_token && g_verbose)
            fprintf(stderr,"  note: the cache is smaller than one token's working set — "
                    "every layer will miss. Raise --expert-gb to at least %.1f to hold it.\n",
                    (double)per_token*m->expert_bytes/1e9);

        /* Per-expert on-disk geometry: resolve the six byte spans (weights +
         * ue8m0 scales) and their contiguity ONCE, so the hot path does zero
         * st_find and the byte-count refusal happens here, at load, instead
         * of mid-inference. Same shape as kimi_k3.c's ERef table. */
        m->e_w1p=MI*((D+1)/2); m->e_w1s=MI*((D+31)/32);   /* w1/w3: [MI,D] fp4 */
        m->e_w2p=D*((MI+1)/2); m->e_w2s=D*((MI+31)/32);   /* w2:    [D,MI] fp4 */
        m->e_slot=2*(m->e_w1p+m->e_w1s)+m->e_w2p+m->e_w2s;
        m->xref=(ExRef*)xzalloc((size_t)c->n_layers*c->n_routed_experts*sizeof(ExRef),
                                "expert on-disk table");
        int64_t want[6]={m->e_w1p,m->e_w1s,m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s};
        const char *mat[3]={"w1","w3","w2"};
        int missing=0;
        for(int l=0;l<c->n_layers;l++){
            for(int e=0;e<c->n_routed_experts;e++){
                ExRef *er=&m->xref[(int64_t)l*c->n_routed_experts+e];
                er->fd[0]=-1;
                for(int k=0;k<6;k++){
                    char nm[192], sn0[192];
                    snprintf(nm,sizeof nm,"layers.%d.ffn.experts.%d.%s.weight",l,e,mat[k/2]);
                    const char *key=(k&1) ? (dsv4_scale_name(sn0,sizeof sn0,nm), sn0) : nm;
                    st_tensor *t=st_find(&m->S,key);
                    if(!t){ missing++; er->fd[k]=-1; continue; }
                    if(t->nbytes!=want[k]){
                        fprintf(stderr,"%s: %lld bytes, expected %lld — refusing (untrusted container)\n",
                                key,(long long)t->nbytes,(long long)want[k]); exit(1); }
                    er->fd[k]=t->fd; er->off[k]=t->off;
                    if(l==0 && e==0 && getenv("DSV4_DEBUG"))
                        fprintf(stderr,"[XREF] L0E0 k=%d key=%s off=%lld nbytes=%lld\n",k,key,
                                (long long)t->off,(long long)t->nbytes);
                }
                /* HF shards pack an expert's six tensors back-to-back — collapse
                 * the load to ONE pread when true. */
                er->contig=1;
                for(int k=0;k<5;k++)
                    if(er->fd[k]!=er->fd[k+1]||er->off[k]+want[k]!=er->off[k+1]) er->contig=0;
                if(getenv("DSV4_FALLBACK")) er->contig=0;
            }
        }
        if(missing && g_verbose)
            fprintf(stderr,"  note: %d routed-expert tensors missing (incomplete download?) — touching one aborts\n",
                    missing);
    }

    /* scratch */
    int D=c->dim, HC=c->hc_mult;
    m->embrow  =(float*)xalloc((size_t)D*sizeof(float),"embed row");
    m->x       =(float*)xzalloc((size_t)HC*D*sizeof(float),"x");
    m->xres    =(float*)xzalloc((size_t)HC*D*sizeof(float),"xres");
    m->h       =(float*)xalloc((size_t)D*sizeof(float),"h");
    m->hn      =(float*)xalloc((size_t)D*sizeof(float),"hn");
    m->q       =(float*)xalloc((size_t)hs*sizeof(float),"q");
    m->attn_out=(float*)xalloc((size_t)hs*sizeof(float),"attn_out");
    m->kv      =(float*)xalloc((size_t)c->head_dim*sizeof(float),"kv");
    m->ffn_g   =(float*)xalloc((size_t)c->moe_inter_dim*sizeof(float),"ffn_g");
    m->ffn_u   =(float*)xalloc((size_t)c->moe_inter_dim*sizeof(float),"ffn_u");
    m->ffn_o   =(float*)xalloc((size_t)D*sizeof(float),"ffn_o");
    m->logits  =(float*)xalloc((size_t)c->vocab_size*sizeof(float),"logits");
    m->idxcap  = c->window_size + g_max_seq/4 + 8;
    m->idxbuf  =(int32_t*)xalloc((size_t)m->idxcap*sizeof(int32_t),"index buffer");
    m->iscore  =(float*)xalloc((size_t)(g_max_seq/4+8)*sizeof(float),"indexer scores");
    m->itop    =(int*)xalloc((size_t)(g_max_seq/4+8)*sizeof(int),"indexer topk");
    m->hpf_eid =(int*)xalloc((size_t)c->n_routed_experts*sizeof(int),"hash prefetch eid");
    m->hpf_seen=(unsigned char*)xzalloc((size_t)c->n_routed_experts,"hash prefetch seen");
}

/* --------------------------------------------------------- expert streaming
 * Per-layer LRU over the routed experts (global slot pool, keyed by
 * layer/eid; every routed expert has one shape so the pool is unfragmented).
 * The interesting placement work (routing heat, learned pins, one-layer-ahead
 * prefetch -- and, on the hash layers, exact prefetch from the token id)
 * belongs on top of a loader that is known to read the right bytes.
 *
 * BATCHED EXPERT I/O: a cache-miss used to be SIX st_read_raw preads (w1,w3,w2
 * + their three scales), each with its own st_find + byte validation and a
 * malloc/free of its buffers. Now the six spans were resolved ONCE into xref
 * at load; the slot owns one 4K-aligned slab; and a miss reads the whole
 * expert with a SINGLE pread (contiguous path, the checkpoint's own layout)
 * or, when the tensors straddle a shard boundary, six per-piece preads packed
 * into that same slab. The W views point into the slab -- zero copies, zero
 * hot-path allocation. With --direct the contiguous read goes through the
 * twin O_DIRECT/F_NOCACHE fd; otherwise the WILLNEED prefetch warms the page
 * cache ahead of the demand reads (overlap with compute). */

/* Issue async readahead (WILLNEED) for a list of experts. Hints only — never
 * changes what is read. Skipped under --direct (no page cache to warm). */
static void expert_prefetch(Model *m, int layer, const int *eids, int n){
    if(g_direct) return;
    int64_t sizes[6]={m->e_w1p,m->e_w1s,m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s};
    for(int j=0;j<n;j++){
        if(eids[j]<0) continue;
        ExRef *er=&m->xref[(int64_t)layer*m->c.n_routed_experts+eids[j]];
        if(er->fd[0]<0) continue;
        if(er->contig){ if(er->fd[0]>=0) posix_fadvise(er->fd[0],er->off[0],m->e_slot,POSIX_FADV_WILLNEED); }
        else for(int k=0;k<6;k++) if(er->fd[k]>=0) posix_fadvise(er->fd[k],er->off[k],sizes[k],POSIX_FADV_WILLNEED);
    }
}

/* EXACT HASH-LAYER PREFETCH. On layers l < n_hash_layers the router is a
 * lookup, not a prediction: tid2eid maps the input token id straight to the
 * K experts that token will use, and dsv4_route_hash picks exactly that set
 * (weights come from the gate logits, the SET comes from the table -- see
 * dsv4_route_hash). So the working set is knowable with ZERO forward compute:
 * the one place in this repository where expert placement needs no routing
 * heat, no learned pin, no one-layer-ahead lookahead.
 *
 * Collect the union over the given token ids and WILLNEED it NOW, so those
 * reads overlap the attention sublayer and the dense routing matmul that run
 * before the FFN instead of starting cold after them. Exact by construction:
 * every prefetched expert sits on a row some token will route through, and no
 * row entry is missed. A hint only -- it never changes what is read, so the
 * logits are untouched, and --direct skips it like every other prefetch. */
static void hash_prefetch(Model *m, Layer *L, const int *ids, int C){
    if(!L->hash) return;
    const DSV4Cfg *c=&m->c;
    int K=c->n_activated_experts;
    int nu=0;
    for(int t=0;t<C;t++){
        const int32_t *row=L->tid2eid.i32+(size_t)ids[t]*K;
        for(int k=0;k<K;k++){
            int e=row[k];
            /* entries were validated into [0,E) at load (see model_load), and
             * K<=64 <= E here; the seen map dedups repeated row entries. */
            if(m->hpf_seen[e]) continue;
            m->hpf_seen[e]=1; m->hpf_eid[nu++]=e;
        }
    }
    for(int j=0;j<nu;j++) m->hpf_seen[m->hpf_eid[j]]=0;
    if(getenv("DSV4_DEBUG"))
        fprintf(stderr,"[HASH] exact prefetch L%d: %d unique experts for %d token ids\n",
                (int)(L-m->L),nu,C);
    expert_prefetch(m,(int)(L-m->L),m->hpf_eid,nu);
}

/* Load expert (layer,eid) into slot v's slab and point the three W views at
 * it. The W geometry is fixed for all routed experts, so the slab is
 * allocated once per slot (lazily). Mirrors kimi_k3.c's expert_read. */
static void expert_load_slot(Model *m, ESlot *v, int layer, int eid){
    if(!v->base){
        if(posix_memalign((void**)&v->base,4096,(size_t)m->e_slot+8192)){
            fprintf(stderr,"OOM expert slot\n"); exit(1); }
        v->base_cap=(size_t)m->e_slot+8192;
    }
    ExRef *er=&m->xref[(int64_t)layer*m->c.n_routed_experts+eid];
    if(er->fd[0]<0){ fprintf(stderr,"[DSV4] expert L%d E%d missing on disk\n",layer,eid); exit(1); }
    if(getenv("DSV4_DEBUG")) fprintf(stderr,"[DSV4] load L%d E%d contig=%d off[0..5]=%lld,%lld,%lld,%lld,%lld,%lld\n",
        layer,eid,er->contig,(long long)er->off[0],(long long)er->off[1],(long long)er->off[2],
        (long long)er->off[3],(long long)er->off[4],(long long)er->off[5]);

    W *w1=&v->w1,*w3=&v->w3,*w2=&v->w2;
    w1->role=w3->role=DSV4_T_FP4; w1->O=w3->O=m->c.moe_inter_dim; w1->I=w3->I=m->c.dim;
    w2->role=DSV4_T_FP4; w2->O=m->c.dim; w2->I=m->c.moe_inter_dim;

    uint8_t *base=v->base;
    if(er->contig){
        int dfd=g_direct?st_direct_fd(&m->S,er->fd[0]):-1;
        if(dfd>=0){
            /* Aligned window read; sub-4K head slack and the tail past the
             * last aligned block are fetched with a tiny buffered pread —
             * O_DIRECT/F_NOCACHE wants aligned lengths. */
            int64_t a0=er->off[0]&~4095LL, pad=er->off[0]-a0;
            int64_t want=pad+m->e_slot;
            struct stat sb;
            int64_t dlen=(want+4095)&~4095LL;
            if(fstat(dfd,&sb)==0 && a0+dlen>sb.st_size) dlen=(sb.st_size-a0)&~4095LL;
            if(dlen>0) st_pread_full(dfd,v->base,dlen,a0,"pread expert direct");
            if(dlen<want) st_pread_full(er->fd[0],v->base+dlen,want-dlen,a0+dlen,"pread expert tail");
            base=v->base+pad;
        } else {
            /* One pread for the whole expert (the batched-I/O fast path). */
            st_pread_full(er->fd[0],v->base,m->e_slot,er->off[0],"pread expert");
            if(getenv("DSV4_DEBUG")){
                /* self-check: the single coalesced pread must be byte-identical
                 * to the per-piece packing of the same six spans. */
                static uint8_t *scr; static size_t scr_cap;
                size_t need=(size_t)m->e_slot+8192;
                if(!scr || scr_cap<need){ scr=realloc(scr,need); scr_cap=need; }
                uint8_t *b=scr; size_t o=0;
                int64_t ss[6]={m->e_w1p,m->e_w1s,m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s};
                for(int k=0;k<6;k++){ st_pread_full(er->fd[k],b+o,ss[k],er->off[k],"chk"); o+=(size_t)ss[k]; }
                if(memcmp(v->base,b,m->e_slot)){
                    fprintf(stderr,"[DSV4] FAST-PATH MISMATCH L%d E%d\n",layer,eid);
                } else {
                    fprintf(stderr,"[DSV4] fast-path ok L%d E%d\n",layer,eid);
                }
            }
        }
    } else {
        /* Rare: an expert's own tensors straddle a shard boundary. Pack them
         * back-to-back in the slab with per-piece preads. */
        int64_t sizes[6]={m->e_w1p,m->e_w1s,m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s};
        uint8_t *dst=v->base;
        for(int k=0;k<6;k++){
            if(er->fd[k]<0){ fprintf(stderr,"[DSV4] expert L%d E%d tensor %d missing on disk\n",layer,eid,k); exit(1); }
            st_pread_full(er->fd[k],dst,sizes[k],er->off[k],"pread expert");
            dst+=sizes[k];
        }
    }
    /* W views into the slab, in file order: w1p w1s w3p w3s w2p w2s. */
    w1->b=base;                          w1->e8=base+m->e_w1p;
    w3->b=base+m->e_w1p+m->e_w1s;        w3->e8=base+m->e_w1p+m->e_w1s+m->e_w1p;
    w2->b=base+m->e_w1p+m->e_w1s+m->e_w1p+m->e_w1s;      w2->e8=base+m->e_w1p+m->e_w1s+m->e_w1p+m->e_w1s+m->e_w2p;

    v->layer=layer; v->eid=eid; v->used=++m->clock; m->expert_miss++;
    if(layer==0 && eid==0 && getenv("DSV4_DEBUG"))
        fprintf(stderr,"[E0] contig=%d w1.b[0..3]=%u,%u,%u,%u w1.e8[0]=%u w2.b[0..3]=%u,%u,%u,%u w2.e8[0]=%u\n",
                er->contig,w1->b[0],w1->b[1],w1->b[2],w1->b[3],w1->e8[0],
                w2->b[0],w2->b[1],w2->b[2],w2->b[3],w2->e8[0]);
}

static void expert_get(Model *m, int layer, int eid, W **w1, W **w3, W **w2){
    for(int s=0;s<m->n_eslot;s++) if(m->eslot[s].layer==layer && m->eslot[s].eid==eid){
        m->eslot[s].used=++m->clock; m->expert_hits++;
        *w1=&m->eslot[s].w1; *w3=&m->eslot[s].w3; *w2=&m->eslot[s].w2; return;
    }
    int victim=0;
    for(int s=0;s<m->n_eslot;s++){
        if(m->eslot[s].eid<0){ victim=s; break; }
        if(m->eslot[s].used<m->eslot[victim].used) victim=s;
    }
    ESlot *v=&m->eslot[victim];
    double t0=now_s();
    expert_load_slot(m,v,layer,eid);
    m->expert_load_s += now_s()-t0;
    *w1=&v->w1; *w3=&v->w3; *w2=&v->w2;
}

/* ------------------------------------------------------------ compressor */

/* Feed one token. Returns 1 if a group closed and a new compressed latent was
 * appended to C->cache. Mirrors Compressor.forward's decode branch: the group
 * closes when (pos+1) % ratio == 0. */
static int compressor_step(Model *m, Compressor *C, const float *x, int pos){
    const DSV4Cfg *c=&m->c;
    int ratio=C->ratio, coff=C->coff, hd=C->hd, wide=coff*hd;
    float *kv=C->s_kv, *score=C->s_score;
    w_matmul(&C->wkv,  kv,   x,1);
    w_matmul(&C->wgate,score,x,1);
    int slot_in_group = pos % ratio;
    const float *ape = C->ape.f + (size_t)slot_in_group*wide;
    for(int i=0;i<wide;i++) score[i]+=ape[i];

    if(coff==2){
        /* Overlapped (ratio 4): the live half of the window sits at
         * ratio + (pos%ratio); the previous group's tail occupies [0,ratio). */
        memcpy(C->kv_state   +(size_t)(ratio+slot_in_group)*wide, kv,   (size_t)wide*sizeof(float));
        memcpy(C->score_state+(size_t)(ratio+slot_in_group)*wide, score,(size_t)wide*sizeof(float));
    } else {
        memcpy(C->kv_state   +(size_t)slot_in_group*wide, kv,   (size_t)wide*sizeof(float));
        memcpy(C->score_state+(size_t)slot_in_group*wide, score,(size_t)wide*sizeof(float));
    }
    if((pos+1)%ratio) return 0;

    /* Group closed: gated softmax pooling over the group axis, per dimension. */
    int rows = coff*ratio;
    float *out=C->s_out;
    if(coff==2){
        /* The reference splices the low half of the OLD slots with the high
         * half of the NEW ones so the pooled window straddles the boundary. */
        float *kvs=C->s_pool_kv, *scs=C->s_pool_sc;
        for(int r=0;r<ratio;r++){
            memcpy(kvs+(size_t)r*hd, C->kv_state   +(size_t)r*wide,        (size_t)hd*sizeof(float));
            memcpy(scs+(size_t)r*hd, C->score_state+(size_t)r*wide,        (size_t)hd*sizeof(float));
        }
        for(int r=0;r<ratio;r++){
            memcpy(kvs+(size_t)(ratio+r)*hd, C->kv_state   +(size_t)(ratio+r)*wide+hd,(size_t)hd*sizeof(float));
            memcpy(scs+(size_t)(ratio+r)*hd, C->score_state+(size_t)(ratio+r)*wide+hd,(size_t)hd*sizeof(float));
        }
        dsv4_compress_pool(kvs,scs,NULL,rows,hd,out);
        /* roll: the new group's slots become the previous tail */
        memcpy(C->kv_state,    C->kv_state   +(size_t)ratio*wide,(size_t)ratio*wide*sizeof(float));
        memcpy(C->score_state, C->score_state+(size_t)ratio*wide,(size_t)ratio*wide*sizeof(float));
    } else {
        dsv4_compress_pool(C->kv_state,C->score_state,NULL,ratio,hd,out);
    }

    rmsnorm(out,out,C->norm.f,hd,c->norm_eps);
    dsv4_rope_apply(out+hd-c->rope_head_dim,c->rope_head_dim,pos+1-ratio,C->freqs,0);
    int rows_cap=g_max_seq/ratio+1;
    if(C->n_cached<rows_cap){
        memcpy(C->cache+(size_t)C->n_cached*hd,out,(size_t)hd*sizeof(float));
        C->n_cached++;
    }
    return 1;
}

/* --------------------------------------------------------------- indexer */

/* Score every compressed position and keep the top index_topk. Returns the
 * count written into `out` (compressed-space indices, caller offsets them). */
static int indexer_select(Model *m, Layer *L, const float *qr, const float *x,
                          int pos, int32_t *out){
    const DSV4Cfg *c=&m->c;
    Indexer *X=&L->idx;
    int H=c->index_n_heads, D=c->index_head_dim;
    int n_ctx=X->comp.n_cached;
    if(n_ctx<=0) return 0;

    float *q=(float*)xalloc((size_t)H*D*sizeof(float),"indexer q");
    w_matmul(&X->wq_b,q,qr,1);
    for(int h=0;h<H;h++)
        dsv4_rope_apply(q+(size_t)h*D+D-c->rope_head_dim,c->rope_head_dim,pos,X->comp.freqs,0);
    /* The reference additionally applies a Hadamard rotation and FP4 QAT
     * simulation to q and the indexer's kv here. Both are norm-preserving
     * quantization-aware steps; skipped, which perturbs WHICH positions win
     * ties but not the mechanism. Flagged rather than silently omitted. */

    float *wts=(float*)xalloc((size_t)H*sizeof(float),"indexer weights");
    w_matmul(&X->weights_proj,wts,x,1);
    float k=(1.0f/sqrtf((float)D))*(1.0f/sqrtf((float)H));
    for(int h=0;h<H;h++) wts[h]*=k;

    dsv4_index_score(q,X->comp.cache,wts,H,D,n_ctx,m->iscore);
    free(q); free(wts);

    int topk = c->index_topk < n_ctx ? c->index_topk : n_ctx;
    /* partial selection: topk is 512 and n_ctx grows to g_max_seq/4, so a
     * repeated max-scan beats a full sort at realistic context lengths */
    for(int i=0;i<n_ctx;i++) m->itop[i]=0;
    int nsel=0;
    for(int i=0;i<topk;i++){
        int best=-1; float bs=-INFINITY;
        for(int t=0;t<n_ctx;t++){
            if(m->itop[t]) continue;
            if(best<0||m->iscore[t]>bs){ best=t; bs=m->iscore[t]; }
        }
        if(best<0) break;
        m->itop[best]=1; out[nsel++]=(int32_t)best;
    }
    return nsel;
}

/* ------------------------------------------------------------- attention */

static void attention(Model *m, Layer *L, const float *xin, int pos, float *out){
    const DSV4Cfg *c=&m->c;
    int H=c->n_heads, HD=c->head_dim, rd=c->rope_head_dim, hs=H*HD;

    /* Q: LoRA down, norm, up, then an UNWEIGHTED per-head RMS, then RoPE on
     * the trailing rd dims. The per-head normalize has no learned parameter --
     * it is not q_norm, which already ran on the rank-1024 vector. */
    float *qr=(float*)xalloc((size_t)c->q_lora_rank*sizeof(float),"qr");
    w_matmul(&L->wq_a,qr,xin,1);
    rmsnorm(qr,qr,L->q_norm.f,c->q_lora_rank,c->norm_eps);
    w_matmul(&L->wq_b,m->q,qr,1);
    for(int h=0;h<H;h++){
        float *qh=m->q+(size_t)h*HD;
        double ss=0; for(int i=0;i<HD;i++) ss+=(double)qh[i]*qh[i];
        float r=1.0f/sqrtf((float)(ss/(double)HD)+c->norm_eps);
        for(int i=0;i<HD;i++) qh[i]*=r;
        dsv4_rope_apply(qh+HD-rd,rd,pos,L->freqs,0);
    }

    /* KV: ONE head_dim vector, shared by every head as both key and value. */
    w_matmul(&L->wkv,m->kv,xin,1);
    rmsnorm(m->kv,m->kv,L->kv_norm.f,HD,c->norm_eps);
    dsv4_rope_apply(m->kv+HD-rd,rd,pos,L->freqs,0);
    memcpy(L->kv_cache+(size_t)(pos % c->window_size)*HD,m->kv,(size_t)HD*sizeof(float));

    /* index set: sliding window, then compressed positions */
    int n_idx=0;
    int win_lo = pos - c->window_size + 1; if(win_lo<0) win_lo=0;
    for(int p=win_lo;p<=pos;p++) m->idxbuf[n_idx++]=(int32_t)(p % c->window_size);

    if(L->ratio){
        compressor_step(m,&L->comp,xin,pos);
        if(L->idx.active) compressor_step(m,&L->idx.comp,xin,pos);
        int base=c->window_size;
        if(L->idx.active){
            int32_t *sel=m->idxbuf+n_idx;
            int ns=indexer_select(m,L,qr,xin,pos,sel);
            for(int i=0;i<ns;i++) sel[i]+=base;
            n_idx+=ns;
        } else {
            for(int t=0;t<L->comp.n_cached && n_idx<m->idxcap;t++)
                m->idxbuf[n_idx++]=(int32_t)(base+t);
        }
        /* Publish the compressed latents into the layer's own kv cache so the
         * attention kernel sees one contiguous position space. */
        for(int t=0;t<L->comp.n_cached;t++){
            int64_t row=c->window_size+t;
            if(row<L->kv_cache_rows)
                memcpy(L->kv_cache+(size_t)row*HD,L->comp.cache+(size_t)t*HD,(size_t)HD*sizeof(float));
        }
    }
    free(qr);

    dsv4_sparse_attn(m->q,L->kv_cache,m->idxbuf,L->attn_sink.f,
                     H,HD,(int)L->kv_cache_rows,n_idx,
                     1.0f/sqrtf((float)HD),m->attn_out);

    /* De-rotate: K and V are the same tensor, so the value side carried RoPE. */
    for(int h=0;h<H;h++)
        dsv4_rope_apply(m->attn_out+(size_t)h*HD+HD-rd,rd,pos,L->freqs,1);

    /* Grouped output LoRA: wo_a applies PER GROUP over hs/o_groups slices,
     * then wo_b maps the concatenated ranks back to dim. */
    int G=c->o_groups, gin=hs/G, R=c->o_lora_rank;
    float *proj=(float*)xalloc((size_t)G*R*sizeof(float),"o proj");
    for(int g=0;g<G;g++){
        const float *src=m->attn_out+(size_t)g*gin;
        const uint8_t *wb=L->wo_a.b+(size_t)g*R*gin;
        /* wo_a is stored [G*R, gin]; group g owns rows [g*R, (g+1)*R). The
         * block-scale array is indexed by absolute row, so slice it by block
         * row rather than by group. */
        int64_t nblkI=dsv4_nblk128(gin);
        for(int r=0;r<R;r++){
            int64_t orow=(int64_t)g*R+r;
            const uint8_t *w=wb+(size_t)r*gin;
            const float *scl=L->wo_a.s+(orow/128)*nblkI;
            double a=0;
            for(int64_t bi=0; bi*128<gin; bi++){
                int b0=(int)(bi*128), bl=128; if(b0+bl>gin) bl=gin-b0;
                float acc=0;
                for(int i=b0;i<b0+bl;i++) acc+=e4m3_decode(w[i])*src[i];
                a+=(double)acc*scl[bi];
            }
            proj[(size_t)g*R+r]=(float)a;
        }
    }
    w_matmul(&L->wo_b,out,proj,1);
    free(proj);
}

/* -------------------------------------------------------------------- moe */

static void moe(Model *m, Layer *L, const float *xin, int token_id, float *out){
    const DSV4Cfg *c=&m->c;
    int E=c->n_routed_experts, K=c->n_activated_experts, MI=c->moe_inter_dim, D=c->dim;
    float *logits=(float*)xalloc((size_t)E*sizeof(float),"router logits");
    w_matmul(&L->gate_w,logits,xin,1);

    int idx[64]; float wt[64];
    int n;
    if(L->hash){
        const int32_t *row=L->tid2eid.i32+(size_t)token_id*K;
        n=dsv4_route_hash(logits,row,K,c->norm_topk,c->route_scale,idx,wt);
    } else {
        n=dsv4_route(logits,L->gate_bias.f,E,K,c->norm_topk,c->route_scale,idx,wt);
    }
    free(logits);
    /* Prefetch this token's experts before applying any: the router already
     * picked them, so the WILLNEED lands while the first matmuls run (overlap
     * with compute on the buffered path). */
    expert_prefetch(m,(int)(L-m->L),idx,n);

    memset(out,0,(size_t)D*sizeof(float));
    for(int i=0;i<n;i++){
        W *w1,*w3,*w2;
        expert_get(m,(int)(L-m->L),idx[i],&w1,&w3,&w2);
        w_matmul(w1,m->ffn_g,xin,1);
        w_matmul(w3,m->ffn_u,xin,1);
        dsv4_swiglu(m->ffn_g,m->ffn_g,m->ffn_u,MI,c->swiglu_limit);
        for(int j=0;j<MI;j++) m->ffn_g[j]*=wt[i];
        w_matmul(w2,m->ffn_o,m->ffn_g,1);
        for(int d=0;d<D;d++) out[d]+=m->ffn_o[d];
    }
    if(c->n_shared_experts){
        w_matmul(&L->sh_w1,m->ffn_g,xin,1);
        w_matmul(&L->sh_w3,m->ffn_u,xin,1);
        dsv4_swiglu(m->ffn_g,m->ffn_g,m->ffn_u,MI,c->swiglu_limit);
        w_matmul(&L->sh_w2,m->ffn_o,m->ffn_g,1);
        for(int d=0;d<D;d++) out[d]+=m->ffn_o[d];
    }
}

/* ------------------------------------------------------------------ block */

static void block(Model *m, int l, int pos, int token_id){
    const DSV4Cfg *c=&m->c;
    Layer *L=&m->L[l];
    int HC=c->hc_mult, D=c->dim;
    float pre[DSV4_HC_MAX], post[DSV4_HC_MAX], comb[DSV4_HC_MAX*DSV4_HC_MAX];

    memcpy(m->xres,m->x,(size_t)HC*D*sizeof(float));
    dsv4_hc_pre(m->x,HC,D,L->hc_attn_fn.f,L->hc_attn_scale.f,L->hc_attn_base.f,
                c->hc_sinkhorn_iters,c->hc_eps,c->norm_eps,m->h,pre,post,comb);
    rmsnorm(m->hn,m->h,L->attn_norm.f,D,c->norm_eps);
    attention(m,L,m->hn,pos,m->h);
    dsv4_hc_post(m->h,m->xres,post,comb,HC,D,m->x);

    memcpy(m->xres,m->x,(size_t)HC*D*sizeof(float));
    dsv4_hc_pre(m->x,HC,D,L->hc_ffn_fn.f,L->hc_ffn_scale.f,L->hc_ffn_base.f,
                c->hc_sinkhorn_iters,c->hc_eps,c->norm_eps,m->h,pre,post,comb);
    rmsnorm(m->hn,m->h,L->ffn_norm.f,D,c->norm_eps);
    moe(m,L,m->hn,token_id,m->h);
    dsv4_hc_post(m->h,m->xres,post,comb,HC,D,m->x);
}

/* Run one token through the stack; leaves logits in m->logits. */
static void forward(Model *m, int token_id, int pos){
    const DSV4Cfg *c=&m->c;
    int HC=c->hc_mult, D=c->dim;
    if(token_id<0||token_id>=c->vocab_size){
        fprintf(stderr,"token id %d outside [0,%d)\n",token_id,c->vocab_size); exit(1); }
    st_read_slice_f32(&m->S,"embed.weight",(int64_t)token_id*D,D,m->embrow,0);
    /* All hc_mult streams start as copies of the embedding (the reference's
     * `h.unsqueeze(2).repeat(1,1,hc_mult,1)`): hc_pre's gates are what
     * differentiates them from the first layer onward. */
    for(int j=0;j<HC;j++) memcpy(m->x+(size_t)j*D,m->embrow,(size_t)D*sizeof(float));
    /* The hash layers route by table, so this token's whole expert working
     * set is known before the first layer runs. WILLNEED it all now: the
     * reads overlap the dense work of the entire stack instead of starting
     * per-layer after each routing. */
    if(c->n_hash_layers)
        for(int l=0;l<c->n_hash_layers;l++) hash_prefetch(m,&m->L[l],&token_id,1);
    for(int l=0;l<c->n_layers;l++) block(m,l,pos,token_id);
    /* Collapse the four streams through the head's OWN hc gates -- not a plain
     * sum. The Transformer carries hc_head_fn/base/scale for exactly this. */
    dsv4_hc_collapse(m->x,HC,D,m->hc_head_fn.f,m->hc_head_scale.f[0],
                     m->hc_head_base.f,c->hc_eps,c->norm_eps,m->h);
    rmsnorm(m->hn,m->h,m->norm.f,D,c->norm_eps);
    w_matmul(&m->head,m->logits,m->hn,1);
}

/* ----------------------------------------------------------------- prefill
 * A CHUNK of C prompt tokens through the stack, LAYER-MAJOR: the MoE loads each
 * unique expert once and applies it to every token in the chunk that routed to
 * it. Same shape as kimi_k3.c's step_chunk.
 *
 * WHY. forward() walks one token through all 43 layers, so a token's six
 * experts per layer are read for that token alone: ~3.4 GB per PROMPT token
 * (measured prefill is 4-6 s/token on short prompts, most of it here). Every
 * token of the prompt passes through the same weights, so inverting the loops
 * lets one read serve every token that wanted it. The ceiling is 256 reads per
 * layer per chunk (the whole expert grid) no matter how long the chunk is.
 *
 * BIT-IDENTICAL to the token-at-a-time path, deliberately:
 *   - the sequential state (kv ring, the compressor's rolling window, the
 *     indexer's cache) still advances ONE TOKEN AT A TIME inside each layer,
 *     which is exactly the original order;
 *   - the dense matmuls still run at S=1 rather than batching over the chunk;
 *   - each token's expert contributions are STAGED per slot and summed in route
 *     order, not in the expert-major order they were computed in.
 * The last one costs C*K*dim floats and buys the property that this is a pure
 * I/O reordering -- which makes the numpy oracle a real regression check on it
 * instead of an approximate one. Batching the dense matmuls is the next step
 * and is NOT free in that sense: it changes reduction order.
 *
 * The head is skipped. Prefill needs no logits, and head.weight is the largest
 * dense matmul in the model. */
static void prefill_chunk(Model *m, const int *ids, int pos0, int C,
                          int total, double t0){
    const DSV4Cfg *c=&m->c;
    int HC=c->hc_mult, D=c->dim, K=c->n_activated_experts;
    int E=c->n_routed_experts, MI=c->moe_inter_dim;
    size_t hcD=(size_t)HC*D;

    float *X   =(float*)xalloc((size_t)C*hcD*sizeof(float),"prefill streams");
    float *XRES=(float*)xalloc((size_t)C*hcD*sizeof(float),"prefill residual");
    float *HN  =(float*)xalloc((size_t)C*D*sizeof(float),  "prefill normed");
    float *POST=(float*)xalloc((size_t)C*DSV4_HC_MAX*sizeof(float),"prefill post");
    float *COMB=(float*)xalloc((size_t)C*DSV4_HC_MAX*DSV4_HC_MAX*sizeof(float),"prefill comb");
    float *ESTG=(float*)xalloc((size_t)C*K*D*sizeof(float),"prefill expert staging");
    int   *IDX =(int*)  xalloc((size_t)C*K*sizeof(int),  "prefill route idx");
    float *WT  =(float*)xalloc((size_t)C*K*sizeof(float),"prefill route wt");
    int   *NSEL=(int*)  xalloc((size_t)C*sizeof(int),    "prefill route n");
    float *logits=(float*)xalloc((size_t)E*sizeof(float),"router logits");
    /* counting sort over (token,slot) -> expert */
    int *map    =(int*)xalloc((size_t)E*sizeof(int),    "expert map");
    int *uid    =(int*)xalloc((size_t)C*K*sizeof(int),  "unique experts");
    int *pcnt   =(int*)xalloc((size_t)C*K*sizeof(int),  "expert counts");
    int *pfirst =(int*)xalloc((size_t)C*K*sizeof(int),  "expert heads");
    int *cur    =(int*)xalloc((size_t)C*K*sizeof(int),  "expert cursors");
    int *poslist=(int*)xalloc((size_t)C*K*sizeof(int),  "expert token list");
    int *slotlist=(int*)xalloc((size_t)C*K*sizeof(int), "expert slot list");

    for(int t=0;t<C;t++){
        if(ids[t]<0||ids[t]>=c->vocab_size){
            fprintf(stderr,"token id %d outside [0,%d)\n",ids[t],c->vocab_size); exit(1); }
        st_read_slice_f32(&m->S,"embed.weight",(int64_t)ids[t]*D,D,m->embrow,0);
        for(int j=0;j<HC;j++) memcpy(X+(size_t)t*hcD+(size_t)j*D,m->embrow,(size_t)D*sizeof(float));
    }

    for(int l=0;l<c->n_layers;l++){
        Layer *L=&m->L[l];
        float pre[DSV4_HC_MAX];
        /* Per LAYER, not per chunk: a chunk of 128 is ~5 minutes of silence
         * otherwise, which reads as a hang -- and was reported as one. */
        if(g_verbose){
            double frac=((double)pos0+(double)C*l/c->n_layers)/(double)total;
            double el=now_s()-t0;
            fprintf(stderr,"\r  prefill %d/%d tokens · layer %d/%d · %.0fs elapsed, ~%.0fs left    ",
                    pos0,total,l+1,c->n_layers,el, frac>0 ? el/frac-el : 0.0);
        }

        /* Hash layers: the chunk's expert set is a pure function of the token
         * ids, so WILLNEED it BEFORE the attention sublayer -- those reads then
         * overlap this layer's attention and routing compute. Phase 2's prefetch
         * (after routing) becomes a redundant hint on the same region. */
        if(L->hash) hash_prefetch(m,L,ids,C);

        /* attention sublayer -- token order, because every piece of state here
         * (the kv ring, the compressor window, the indexer) is a recurrence. */
        for(int t=0;t<C;t++){
            float *x=X+(size_t)t*hcD;
            float post[DSV4_HC_MAX], comb[DSV4_HC_MAX*DSV4_HC_MAX];
            memcpy(m->xres,x,hcD*sizeof(float));
            dsv4_hc_pre(x,HC,D,L->hc_attn_fn.f,L->hc_attn_scale.f,L->hc_attn_base.f,
                        c->hc_sinkhorn_iters,c->hc_eps,c->norm_eps,m->h,pre,post,comb);
            rmsnorm(m->hn,m->h,L->attn_norm.f,D,c->norm_eps);
            attention(m,L,m->hn,pos0+t,m->h);
            dsv4_hc_post(m->h,m->xres,post,comb,HC,D,x);
        }

        /* FFN sublayer, phase 1: normalize and ROUTE every token, so the expert
         * working set of the whole chunk is known before a single read. */
        for(int t=0;t<C;t++){
            float *x=X+(size_t)t*hcD;
            memcpy(XRES+(size_t)t*hcD,x,hcD*sizeof(float));
            dsv4_hc_pre(x,HC,D,L->hc_ffn_fn.f,L->hc_ffn_scale.f,L->hc_ffn_base.f,
                        c->hc_sinkhorn_iters,c->hc_eps,c->norm_eps,m->h,pre,
                        POST+(size_t)t*DSV4_HC_MAX,
                        COMB+(size_t)t*DSV4_HC_MAX*DSV4_HC_MAX);
            rmsnorm(HN+(size_t)t*D,m->h,L->ffn_norm.f,D,c->norm_eps);
            w_matmul(&L->gate_w,logits,HN+(size_t)t*D,1);
            if(L->hash){
                const int32_t *row=L->tid2eid.i32+(size_t)ids[t]*K;
                NSEL[t]=dsv4_route_hash(logits,row,K,c->norm_topk,c->route_scale,
                                        IDX+(size_t)t*K,WT+(size_t)t*K);
            } else {
                NSEL[t]=dsv4_route(logits,L->gate_bias.f,E,K,c->norm_topk,c->route_scale,
                                   IDX+(size_t)t*K,WT+(size_t)t*K);
            }
        }

        /* phase 2: bucket (token,slot) pairs by expert, then walk expert-major. */
        for(int e=0;e<E;e++) map[e]=-1;
        int nu=0;
        for(int t=0;t<C;t++) for(int k=0;k<NSEL[t];k++){
            int e=IDX[(size_t)t*K+k];
            if(map[e]<0){ map[e]=nu; uid[nu]=e; pcnt[nu]=0; nu++; }
            pcnt[map[e]]++;
        }
        int acc=0;
        for(int j=0;j<nu;j++){ pfirst[j]=acc; cur[j]=acc; acc+=pcnt[j]; }
        for(int t=0;t<C;t++) for(int k=0;k<NSEL[t];k++){
            int j=map[IDX[(size_t)t*K+k]];
            poslist[cur[j]]=t; slotlist[cur[j]]=k; cur[j]++;
        }
        /* Loads in DISK-OFFSET order (the checkpoint is not expert-id-ordered
         * inside a layer), and WILLNEED the whole working set before the first
         * demand read so kernel readahead overlaps the matmuls that follow. */
        for(int a2=0;a2<nu-1;a2++) for(int b2=a2+1;b2<nu;b2++){
            ExRef *ea=&m->xref[(int64_t)l*E+uid[a2]], *eb=&m->xref[(int64_t)l*E+uid[b2]];
            if(eb->fd[0]<ea->fd[0]||(eb->fd[0]==ea->fd[0]&&eb->off[0]<ea->off[0])){
                int tt=uid[a2];uid[a2]=uid[b2];uid[b2]=tt;
                tt=pcnt[a2];pcnt[a2]=pcnt[b2];pcnt[b2]=tt;
                tt=pfirst[a2];pfirst[a2]=pfirst[b2];pfirst[b2]=tt;
            }
        }
        expert_prefetch(m,l,uid,nu);
        for(int j=0;j<nu;j++){
            W *w1,*w3,*w2;
            expert_get(m,l,uid[j],&w1,&w3,&w2);   /* ONE read for pcnt[j] tokens */
            for(int q=pfirst[j];q<pfirst[j]+pcnt[j];q++){
                int t=poslist[q], k=slotlist[q];
                const float *xin=HN+(size_t)t*D;
                w_matmul(w1,m->ffn_g,xin,1);
                w_matmul(w3,m->ffn_u,xin,1);
                dsv4_swiglu(m->ffn_g,m->ffn_g,m->ffn_u,MI,c->swiglu_limit);
                float wt=WT[(size_t)t*K+k];
                for(int i=0;i<MI;i++) m->ffn_g[i]*=wt;
                w_matmul(w2,ESTG+((size_t)t*K+k)*D,m->ffn_g,1);
            }
        }

        /* phase 3: sum in ROUTE order (see the bit-identical note), add the
         * shared expert -- resident, so no ordering pressure -- and expand. */
        for(int t=0;t<C;t++){
            float *out=m->h;
            memset(out,0,(size_t)D*sizeof(float));
            for(int k=0;k<NSEL[t];k++){
                const float *e=ESTG+((size_t)t*K+k)*D;
                for(int d=0;d<D;d++) out[d]+=e[d];
            }
            if(c->n_shared_experts){
                const float *xin=HN+(size_t)t*D;
                w_matmul(&L->sh_w1,m->ffn_g,xin,1);
                w_matmul(&L->sh_w3,m->ffn_u,xin,1);
                dsv4_swiglu(m->ffn_g,m->ffn_g,m->ffn_u,MI,c->swiglu_limit);
                w_matmul(&L->sh_w2,m->ffn_o,m->ffn_g,1);
                for(int d=0;d<D;d++) out[d]+=m->ffn_o[d];
            }
            dsv4_hc_post(out,XRES+(size_t)t*hcD,
                         POST+(size_t)t*DSV4_HC_MAX,
                         COMB+(size_t)t*DSV4_HC_MAX*DSV4_HC_MAX,HC,D,X+(size_t)t*hcD);
        }
    }

    free(X); free(XRES); free(HN); free(POST); free(COMB); free(ESTG);
    free(IDX); free(WT); free(NSEL); free(logits);
    free(map); free(uid); free(pcnt); free(pfirst); free(cur);
    free(poslist); free(slotlist);
}

/* ---------------------------------------------------------------- sampling */

static int sample(Model *m){
    int V=m->c.vocab_size;
    int best=0;
    for(int i=1;i<V;i++) if(m->logits[i]>m->logits[best]) best=i;
    if(g_temp<=0.0f) return best;
    double sum=0;
    for(int i=0;i<V;i++){ m->logits[i]=expf((m->logits[i]-m->logits[best])/g_temp); sum+=m->logits[i]; }
    double r=((double)rand()/((double)RAND_MAX+1.0))*sum, acc=0;
    for(int i=0;i<V;i++){ acc+=m->logits[i]; if(acc>=r) return i; }
    return best;
}

/* -------------------------------------------------------------------- main */


/* ---------------------------------------------------------------- preflight
 * Walk the manifest against a real checkpoint WITHOUT reading a single weight
 * byte: st_init parses only the safetensors headers, so this is seconds on a
 * 160 GB container. It answers the question that otherwise costs a download
 * plus a crash -- does this checkpoint contain what the engine expects, with
 * the shapes it expects?
 *
 * It deliberately uses dsv4_manifest, the same derivation the loader consumes.
 * A separate re-implementation of the expected tensor set would drift from the
 * loader precisely when it mattered.
 *
 * UNMATCHED CONTAINER TENSORS ARE NOT ERRORS. The manifest covers the main
 * stack only: the speculative head is real (one MTP layer on the preview,
 * three DSpark modules on -0731, each with a trailing compress_ratios entry)
 * but its names are unknown, so anything left over is reported as
 * expected-unknown. Treating it as corruption would make a correct checkpoint
 * look broken. */
static int64_t pf_expect_weight(const DSV4Tensor *t){
    int64_t O=t->d0, I=t->d1?t->d1:1;
    switch(t->role){
        case DSV4_T_FP8: return O*I;                 /* raw e4m3 */
        case DSV4_T_FP4: return O*((I+1)/2);         /* packed nibbles */
        case DSV4_T_I32: return O*I*4;
        case DSV4_T_PLAIN_BF16: return O*I*2;        /* bf16, must be exact */
        default:         return -1;                  /* PLAIN: bf16 or f32 */
    }
}

static int preflight(const char *dir){
    double t0=now_s();
    fprintf(stderr,"  preflight: %s\n",dir);

    char cp[2100]; snprintf(cp,sizeof cp,"%s/config.json",dir);
    FILE *f=fopen(cp,"rb");
    if(!f){ fprintf(stderr,"  FAIL  cannot open %s\n",cp); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<=0||n>(64L<<20)){ fprintf(stderr,"  FAIL  %s: implausible size %ld\n",cp,n); fclose(f); return 1; }
    char *buf=(char*)xalloc((size_t)n+1,"config");
    size_t got=fread(buf,1,(size_t)n,f); buf[got]=0; fclose(f);
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    DSV4Cfg c;
    if(!root || dsv4_cfg_from_json(root,&c)!=0){ fprintf(stderr,"  FAIL  config.json rejected\n"); return 1; }
    fprintf(stderr,"  config: %d layers, dim %d, %d experts/layer, top-%d, vocab %d\n",
            c.n_layers,c.dim,c.n_routed_experts,c.n_activated_experts,c.vocab_size);
    if(c.n_spec)
        fprintf(stderr,"  config: %d speculative module(s) after the main stack — not loaded\n",
                c.n_spec);

    static shards S; memset(&S,0,sizeof S);
    st_init(&S,dir);
    fprintf(stderr,"  container: %d tensors indexed (headers only, no weights read)\n",S.n);

    int nman=dsv4_manifest_count(&c);
    DSV4Tensor *man=(DSV4Tensor*)xalloc((size_t)nman*sizeof(DSV4Tensor),"manifest");
    if(dsv4_manifest(&c,man,nman)!=nman){ fprintf(stderr,"  FAIL  internal: manifest count disagrees with the walk\n"); return 1; }
    fprintf(stderr,"  manifest: %d tensors expected\n\n",nman);

    unsigned char *seen=(unsigned char*)xzalloc((size_t)S.n,"seen map");
    int missing=0, badsize=0, badscale=0, ok=0, shown=0;
    #define PF_SHOW(...) do{ if(shown<20){ fprintf(stderr,__VA_ARGS__); shown++; } \
                             else if(shown==20){ fprintf(stderr,"  ... (further problems suppressed)\n"); shown++; } }while(0)

    for(int i=0;i<nman;i++){
        DSV4Tensor *t=&man[i];
        st_tensor *st=st_find(&S,t->name);
        if(!st){
            if(!t->optional){ missing++; PF_SHOW("  MISSING   %s  [%lld,%lld]\n",
                t->name,(long long)t->d0,(long long)t->d1); }
            continue;
        }
        seen[st - S.t]=1;
        /* Mark the sidecar as accounted for BEFORE validating the weight: on a
         * size mismatch we bail out below, and an unmarked sidecar would then
         * surface in the "not covered by the manifest" list, which reads as a
         * second, unrelated problem. */
        st_tensor *ss=NULL;
        char sn0[192]; dsv4_scale_name(sn0,sizeof sn0,t->name);
        if(t->role==DSV4_T_FP8 || t->role==DSV4_T_FP4){
            ss=st_find(&S,sn0);
            if(ss) seen[ss - S.t]=1;
        }
        int64_t want=pf_expect_weight(t);
        /* An index table is int32 or int64 -- torch's default is the latter and
         * w_load narrows it -- so both spans are correct here. */
        if(t->role==DSV4_T_I32 && st->nbytes==want*2) want=st->nbytes;
        if(want>=0 && st->nbytes!=want){
            badsize++; PF_SHOW("  SIZE      %s  %lld bytes, expected %lld for [%lld,%lld]\n",
                t->name,(long long)st->nbytes,(long long)want,(long long)t->d0,(long long)t->d1);
            continue;
        }
        if(want<0){   /* PLAIN: accept bf16/f16 (2 bytes) or f32 (4) */
            int64_t nel=t->d1?t->d0*t->d1:t->d0;
            if(st->nbytes!=nel*2 && st->nbytes!=nel*4){
                badsize++; PF_SHOW("  SIZE      %s  %lld bytes, expected %lld (bf16) or %lld (f32)\n",
                    t->name,(long long)st->nbytes,(long long)(nel*2),(long long)(nel*4));
                continue;
            }
        }
        /* scale sidecar, where the role has one */
        if(t->role==DSV4_T_FP8 || t->role==DSV4_T_FP4){
            if(!ss){
                badscale++; PF_SHOW("  NO SCALE  %s is absent\n",sn0);
                continue;
            }
            int64_t I=t->d1?t->d1:1;
            int64_t nblk = (t->role==DSV4_T_FP8)
                         ? dsv4_nblk128(t->d0)*dsv4_nblk128(I)
                         : t->d0*((I+31)/32);
            int fp8_f32 = (t->role==DSV4_T_FP8 && ss->nbytes==nblk*4);
            if(ss->nbytes!=nblk && !fp8_f32){
                badscale++; PF_SHOW("  SCALE     %s is %lld bytes, expected %lld (ue8m0)%s\n",
                    sn0,(long long)ss->nbytes,(long long)nblk,
                    t->role==DSV4_T_FP8 ? " or 4x that (f32)" : "");
                continue;
            }
        }
        ok++;
    }

    int extra=0;
    for(int i=0;i<S.n;i++) if(!seen[i]) extra++;
    if(extra){
        fprintf(stderr,"\n  %d container tensor(s) the manifest does not cover — "
                       "EXPECTED, not an error:\n",extra);
        int e=0;
        for(int i=0;i<S.n && e<6;i++) if(!seen[i]){ fprintf(stderr,"    %s\n",S.t[i].name); e++; }
        if(extra>6) fprintf(stderr,"    ... and %d more\n",extra-6);
        fprintf(stderr,"  (the speculative head is the known case: %d module(s) exist in this\n"
                       "   checkpoint, the reference does not build them, so their names are\n"
                       "   unknown here)\n", c.n_spec);
    }

    char tp[2100]; snprintf(tp,sizeof tp,"%s/tokenizer.json",dir);
    FILE *tf=fopen(tp,"rb"); int have_tok = tf!=NULL; if(tf) fclose(tf);

    int bad = missing+badsize+badscale;
    fprintf(stderr,"\n  %d/%d matched", ok, nman);
    if(missing)  fprintf(stderr,", %d missing",missing);
    if(badsize)  fprintf(stderr,", %d wrong size",badsize);
    if(badscale) fprintf(stderr,", %d scale problems",badscale);
    fprintf(stderr,"  (%.1fs)\n",now_s()-t0);
    if(!have_tok)
        fprintf(stderr,"  tokenizer.json absent — install one before running:\n"
                       "    python3 tools/dsv4_tokenizer.py %s --from <tokenizer.json>\n",dir);
    fprintf(stderr,"\n  %s\n", bad ? "PREFLIGHT FAILED — the checkpoint does not match what this engine expects."
                                   : have_tok ? "PREFLIGHT OK — shapes and names check out."
                                              : "PREFLIGHT OK on weights; the tokenizer is still missing.");
    free(seen); free(man); free(buf); free(arena);
    #undef PF_SHOW
    return bad ? 1 : 0;
}

static void usage(const char *p){
    fprintf(stderr,
      "usage: %s <model_dir> \"prompt\" [--ngen N] [--temp T] [--expert-gb G] [--max-seq N]\n"
      "       %s <model_dir> \"3,7,1\" --ids [--dump-logits out.json]   (oracle mode, no tokenizer)\n"
      "       %s <model_dir> --preflight     (check names and shapes, read no weights)\n"
      "  --chunk N: prefill tokens per pass of expert reads (default 32, 1 = per-token)\n"
      "  --direct: expert reads via O_DIRECT / F_NOCACHE, bypassing the page cache\\n"
      "           (drive-dependent, measure it; --direct disables the WILLNEED prefetch)\\n"
      "  DeepSeek-V4-Flash (284B MoE, 13B active). UNVERIFIED -- see the header comment.\n",p,p,p);
}

int main(int argc, char **argv){
    if(argc<2){ usage(argv[0]); return 1; }
    const char *dir=argv[1];
    const char *prompt = (argc>2 && argv[2][0]!='-') ? argv[2] : "";
    for(int i=(argc>2 && argv[2][0]!='-')?3:2; i<argc; i++){
        if(!strcmp(argv[i],"--ngen")   && i+1<argc) g_ngen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--temp")  && i+1<argc) g_temp=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--expert-gb")&&i+1<argc) g_expert_gb=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--max-seq")&&i+1<argc) g_max_seq=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--chunk")  &&i+1<argc) g_chunk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--direct")) g_direct=1;
        else if(!strcmp(argv[i],"--quiet")) g_verbose=0;
        else if(!strcmp(argv[i],"--ids")) g_ids_mode=1;
        else if(!strcmp(argv[i],"--preflight")) g_preflight=1;
        else if(!strcmp(argv[i],"--dump-logits") && i+1<argc) g_dump=argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if(g_expert_gb<=0.0f) g_expert_gb=0.25f;
    if(g_max_seq<64) g_max_seq=64;
    if(g_chunk<1) g_chunk=1;
    if(g_chunk>512) g_chunk=512;
#ifdef _OPENMP
    coli_omp_tune_threads("deepseek_v4");   /* physical-core team, no spin-wait — see omp_tune.h */
#endif

    fprintf(stderr,"  deepseek_v4 — 284B MoE, 13B active (UNVERIFIED build)\n");
    if(g_preflight) return preflight(dir);

    static Model m; memset(&m,0,sizeof m);
    model_load(&m,dir);

    int *ids=(int*)xalloc((size_t)(g_max_seq+8)*sizeof(int),"prompt ids");
    int n_ids=0;
    if(g_ids_mode){
        const char *p=prompt;
        while(*p && n_ids<g_max_seq){
            char *end; long v=strtol(p,&end,10);
            if(end==p) break;
            ids[n_ids++]=(int)v;
            p=end; while(*p==','||*p==' ') p++;
        }
    } else {
        n_ids=tok_encode(&m.tok,prompt,(int)strlen(prompt),ids,g_max_seq);
    }
    if(n_ids<=0){ fprintf(stderr,"no input tokens\n"); return 1; }

    FILE *dump=NULL;
    if(g_dump){
        dump=fopen(g_dump,"wb");
        if(!dump){ fprintf(stderr,"cannot write %s\n",g_dump); return 1; }
        fprintf(dump,"{\"steps\":[");
    }
    int dumped=0;

    double t0=now_s();
    int pos=0, tok=ids[0];
    /* Prefill, one token at a time (see the header's known gaps). Each prompt
     * token costs a FULL forward pass -- the same ~3.4 GB of expert reads as a
     * generated one -- so a few hundred tokens of prompt is tens of minutes
     * before the first output character. Silence there is indistinguishable
     * from a hang, and was reported as one, hence the progress line. */
    if(g_verbose && n_ids>1)
        fprintf(stderr,"  prompt: %d tokens, prefill in chunks of %d\n",n_ids,g_chunk);
    while(pos<n_ids-1){
        int cc=n_ids-1-pos; if(cc>g_chunk) cc=g_chunk;
        prefill_chunk(&m,ids+pos,pos,cc,n_ids-1,t0);
        pos+=cc;
    }
    if(g_verbose && n_ids>1){
        double el=now_s()-t0;
        fprintf(stderr,"\r  prefill %d/%d tokens · done in %.0fs%20s",n_ids-1,n_ids-1,el,"");
    }
    tok=ids[n_ids-1];
    if(g_verbose && n_ids>1) fprintf(stderr,"\n");
    /* Prefill and decode are different regimes -- one is N forward passes with
     * no output, the other is one per token -- and averaging them reported
     * 0.03 tok/s for a run that decoded at 0.25. Time them apart. */
    double t_pre=now_s();

    char piece[512];
    int gen=0, hit_stop=0;
    for(int g=0;g<g_ngen;g++){
        forward(&m,tok,pos++);
        if(dump){
            fprintf(dump,"%s{\"pos\":%d,\"in\":%d,\"logits\":[",dumped?",":"",pos-1,tok);
            for(int v=0;v<m.c.vocab_size;v++)
                fprintf(dump,"%s%.9g",v?",":"",(double)m.logits[v]);
            fprintf(dump,"]}");
            dumped++;
        }
        tok=sample(&m);
        /* --ids is the oracle path: the fixture compares a FIXED number of
         * steps against numpy, so a stop there would turn a random sample into
         * a spurious test failure. Text generation is where a user wants it. */
        if(!g_ids_mode && is_stop(tok)){ hit_stop=1; break; }
        if(m.have_tok){
            int nb=tok_decode(&m.tok,&tok,1,piece,(int)sizeof piece);
            if(nb>0){ fwrite(piece,1,(size_t)nb,stdout); fflush(stdout); }
        } else {
            printf("%s%d",gen?",":"",tok); fflush(stdout);
        }
        gen++;
        if(pos>=g_max_seq-1) break;
    }
    if(dump){ fprintf(dump,"]}\n"); fclose(dump); }
    double dt=now_s()-t_pre, pre=t_pre-t0;
    /* `gen`, not g_ngen: the loop can end early on a stop id or on max-seq, and
     * reporting the request instead of the result overstates tok/s. */
    fprintf(stderr,"\n");
    if(n_ids>1)
        fprintf(stderr,"  prefill: %d tokens in %.1fs (%.1f s/token)\n",
                n_ids-1,pre,pre/(n_ids-1));
    fprintf(stderr,"  decode:  %d tokens in %.1fs (%.2f tok/s)%s\n",
            gen,dt,dt>0?gen/dt:0.0, hit_stop?" [stopped at eos]":"");
    fprintf(stderr,"  experts: %llu hit / %llu miss, %.1fs loading\n",
            (unsigned long long)m.expert_hits,(unsigned long long)m.expert_miss,m.expert_load_s);
    free(ids);
    return 0;
}
