#ifndef COLIBRI_DSV4_MODEL_H
#define COLIBRI_DSV4_MODEL_H
/* DeepSeek-V4-Flash config + tensor manifest.
 *
 * dsv4.h is pure compute; this header is the layer above it: it reads
 * config.json and derives, from the config ALONE, the complete set of tensors
 * the checkpoint must contain -- every name, every shape, every role. Nothing
 * here touches a file or a Model struct, so tests can build a config in memory
 * and check the whole manifest without a 160 GB download.
 *
 * WHY A MANIFEST RATHER THAN INLINE qt_load CALLS. A loader that names tensors
 * inline (as colibri.c does) discovers a wrong shape at the tensor that
 * happens to load first, with whatever error the safetensors layer produces.
 * A manifest can be walked BEFORE any bytes are read, so a checkpoint can be
 * verified in one pass with precise per-tensor messages -- the `coli doctor
 * --deep` shape of preflight. It also means the shape derivation itself is
 * unit-testable, which matters here because it was derived by reading the
 * published reference implementation, not by loading the model.
 *
 * SHAPES: derived from `inference/model.py` (Attention / Compressor / Indexer /
 * Block / MoE / Expert constructors) in deepseek-ai/DeepSeek-V4-Flash, cross-
 * checked against the checkpoint's own tensor-name set. UNVERIFIED against the
 * real weights -- see tests/test_dsv4_model.c, whose parameter-count assertion
 * is the strongest external check available without the checkpoint. */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "json.h"
#include "dsv4.h"

/* 128x128 block count for the fp8 scale sidecar. Deliberately NOT pulled in
 * from quant.h: this header is about shapes, and dragging the whole kernel
 * library in to reuse one ceil-divide would make every manifest consumer
 * depend on the SIMD stack. Kept in sync by tests/test_dsv4_model.c's on-disk
 * size assertion, which would move if this diverged from quant.h's FP8_BLOCK. */
static inline int64_t dsv4_nblk128(int64_t n){ return (n+127)/128; }

#define DSV4_MAX_LAYERS 128

typedef struct {
    int dim, n_layers, n_heads, head_dim, rope_head_dim;
    int q_lora_rank, o_lora_rank, o_groups;
    int n_routed_experts, n_activated_experts, moe_inter_dim, n_shared_experts;
    int vocab_size, n_hash_layers;
    int index_n_heads, index_head_dim, index_topk;
    int hc_mult, hc_sinkhorn_iters;
    int window_size, original_seq_len, max_seq_len;
    int norm_topk;      /* norm_topk_prob: renormalize the selected expert
                         * weights to sum 1 before route_scale */
    int n_nextn;        /* num_nextn_predict_layers, as published */
    int n_spec;         /* speculative-decoding modules appended after the main
                         * stack = how many trailing compress_ratios entries are
                         * NOT main-stack layers. Derived, because the two
                         * published checkpoints disagree with each other about
                         * which key states it -- see the note below. */
    float hc_eps, norm_eps, route_scale, swiglu_limit;
    float rope_theta, compress_rope_theta, rope_factor, beta_fast, beta_slow;
    int compress_ratios[DSV4_MAX_LAYERS];
    int n_compress_ratios;
} DSV4Cfg;

/* Tensor roles. The role decides how the bytes are read, and each has a
 * different sidecar convention -- getting this wrong is a silent misread, so
 * the manifest carries it explicitly rather than letting the loader guess from
 * the name. */
enum {
    DSV4_T_PLAIN = 0,   /* bf16/f32, no sidecar: norms, ape, attn_sink, hc gates */
    DSV4_T_FP8,         /* fp8-e4m3 + `<name>.scale`, 128x128 blocks, ue8m0 (config's
                         * quantization_config) -- the format colibri.c's fmt=8 reads */
    DSV4_T_FP4,         /* mxfp4 routed expert: `<name>` is [O,I/2] nibbles,
                         * `<name>.scale` is [O,I/32] ue8m0 -- quant.h's matmul_mxfp4 */
    DSV4_T_I32          /* tid2eid hash table, int32, no sidecar */
};

typedef struct {
    char name[160];
    int64_t d0, d1;     /* logical [out, in]; d1==0 means a 1-D tensor of d0 */
    int role;
    int optional;       /* absent on some layers by design (bias / tid2eid / MTP) */
} DSV4Tensor;

/* ---------------- config ---------------- */

static int dsv4_gi(jval *r, const char *k, int dflt){
    jval *v=json_get(r,k); return (v&&v->t==J_NUM)?(int)v->num:dflt;
}
static float dsv4_gf(jval *r, const char *k, float dflt){
    jval *v=json_get(r,k); return (v&&v->t==J_NUM)?(float)v->num:dflt;
}

/* Returns 0 on success, -1 with a message on stderr otherwise. Deliberately
 * strict about the keys that have no safe default: a missing compress_ratios
 * or head_dim is a different model, not a variation. */
static int dsv4_cfg_from_json(jval *r, DSV4Cfg *c){
    memset(c,0,sizeof *c);
    c->dim              = dsv4_gi(r,"hidden_size",0);
    c->n_layers         = dsv4_gi(r,"num_hidden_layers",0);
    c->n_heads          = dsv4_gi(r,"num_attention_heads",0);
    c->head_dim         = dsv4_gi(r,"head_dim",0);
    c->rope_head_dim    = dsv4_gi(r,"qk_rope_head_dim",0);
    c->q_lora_rank      = dsv4_gi(r,"q_lora_rank",0);
    c->o_lora_rank      = dsv4_gi(r,"o_lora_rank",0);
    c->o_groups         = dsv4_gi(r,"o_groups",0);
    c->n_routed_experts = dsv4_gi(r,"n_routed_experts",0);
    c->n_activated_experts = dsv4_gi(r,"num_experts_per_tok",0);
    c->moe_inter_dim    = dsv4_gi(r,"moe_intermediate_size",0);
    c->n_shared_experts = dsv4_gi(r,"n_shared_experts",0);
    c->vocab_size       = dsv4_gi(r,"vocab_size",0);
    c->n_hash_layers    = dsv4_gi(r,"num_hash_layers",0);
    c->index_n_heads    = dsv4_gi(r,"index_n_heads",0);
    c->index_head_dim   = dsv4_gi(r,"index_head_dim",0);
    c->index_topk       = dsv4_gi(r,"index_topk",0);
    c->hc_mult          = dsv4_gi(r,"hc_mult",0);
    c->hc_sinkhorn_iters= dsv4_gi(r,"hc_sinkhorn_iters",0);
    c->window_size      = dsv4_gi(r,"sliding_window",0);
    c->max_seq_len      = dsv4_gi(r,"max_position_embeddings",0);
    c->n_nextn          = dsv4_gi(r,"num_nextn_predict_layers",0);
    { jval *nt=json_get(r,"norm_topk_prob");
      c->norm_topk = (nt&&nt->t==J_BOOL) ? nt->boolean : 1; }
    c->hc_eps           = dsv4_gf(r,"hc_eps",1e-6f);
    c->norm_eps         = dsv4_gf(r,"rms_norm_eps",1e-6f);
    c->route_scale      = dsv4_gf(r,"routed_scaling_factor",1.0f);
    c->swiglu_limit     = dsv4_gf(r,"swiglu_limit",0.0f);
    /* Two RoPE bases: compressed layers use compress_rope_theta WITH YaRN,
     * sliding-window-only layers use rope_theta with YaRN disabled. */
    c->rope_theta          = dsv4_gf(r,"rope_theta",10000.f);
    c->compress_rope_theta = dsv4_gf(r,"compress_rope_theta",c->rope_theta);
    { jval *rs=json_get(r,"rope_scaling");
      c->rope_factor      = rs?dsv4_gf(rs,"factor",1.f):1.f;
      c->beta_fast        = rs?dsv4_gf(rs,"beta_fast",32.f):32.f;
      c->beta_slow        = rs?dsv4_gf(rs,"beta_slow",1.f):1.f;
      c->original_seq_len = rs?dsv4_gi(rs,"original_max_position_embeddings",0):0; }
    { jval *cr=json_get(r,"compress_ratios");
      if(!cr||cr->t!=J_ARR){ fprintf(stderr,"config: compress_ratios missing or not an array\n"); return -1; }
      if(cr->len>DSV4_MAX_LAYERS){ fprintf(stderr,"config: compress_ratios has %d entries, max %d\n",cr->len,DSV4_MAX_LAYERS); return -1; }
      c->n_compress_ratios=cr->len;
      for(int i=0;i<cr->len;i++) c->compress_ratios[i]=(int)cr->kids[i]->num; }
    /* DSpark (the -0731 release) states its speculative-module count only by
     * the length of dspark_target_layer_ids -- one module per fused layer,
     * mtp.0..mtp.N-1 -- while leaving num_nextn_predict_layers at the preview's
     * 1. Neither key alone explains that checkpoint's trailing entries. */
    int n_dspark=0;
    { jval *dt=json_get(r,"dspark_target_layer_ids");
      if(dt&&dt->t==J_ARR) n_dspark=dt->len; }

    #define REQ(field,name,lo,hi) if(c->field<(lo)||c->field>(hi)){ \
        fprintf(stderr,"config: %s=%d outside [%d,%d]\n",name,(int)c->field,(int)(lo),(int)(hi)); return -1; }
    REQ(dim,"hidden_size",1,1<<20)                 REQ(n_layers,"num_hidden_layers",1,DSV4_MAX_LAYERS)
    REQ(n_heads,"num_attention_heads",1,4096)      REQ(head_dim,"head_dim",1,1<<16)
    REQ(rope_head_dim,"qk_rope_head_dim",2,c->head_dim)
    REQ(q_lora_rank,"q_lora_rank",1,1<<20)         REQ(o_lora_rank,"o_lora_rank",1,1<<20)
    REQ(o_groups,"o_groups",1,1024)                REQ(n_routed_experts,"n_routed_experts",1,1<<16)
    REQ(n_activated_experts,"num_experts_per_tok",1,64)
    REQ(moe_inter_dim,"moe_intermediate_size",1,1<<20)
    REQ(n_shared_experts,"n_shared_experts",0,64)  REQ(vocab_size,"vocab_size",1,1<<24)
    REQ(n_hash_layers,"num_hash_layers",0,c->n_layers)
    REQ(index_n_heads,"index_n_heads",1,4096)      REQ(index_head_dim,"index_head_dim",2,1<<16)
    REQ(hc_mult,"hc_mult",1,DSV4_HC_MAX)           REQ(hc_sinkhorn_iters,"hc_sinkhorn_iters",1,1024)
    REQ(window_size,"sliding_window",1,1<<20)
    #undef REQ
    /* compress_ratios covers the main stack AND the speculative head(s), and
     * how many of the latter there are differs between the two published
     * checkpoints:
     *   preview      44 entries, num_hidden_layers=43, one MTP layer
     *   -0731 DSpark 46 entries, num_hidden_layers=43, three modules
     *                (mtp.0/1/2, one per dspark_target_layer_ids entry) while
     *                num_nextn_predict_layers still reads 1
     * Accept exactly the lengths a key in THIS config accounts for -- no head,
     * n_nextn heads, or n_dspark modules -- and refuse anything else rather
     * than silently indexing past the main stack. The trailing entries are only
     * ever read by a head this engine does not build; what they must not do is
     * shift which ratio a main-stack layer gets. */
    if(c->n_compress_ratios == c->n_layers)                        c->n_spec=0;
    else if(c->n_compress_ratios == c->n_layers + c->n_nextn)      c->n_spec=c->n_nextn;
    else if(n_dspark && c->n_compress_ratios == c->n_layers + n_dspark) c->n_spec=n_dspark;
    else {
        fprintf(stderr,"config: compress_ratios has %d entries, expected %d "
                "(num_hidden_layers) or %d (+ num_nextn_predict_layers=%d)",
                c->n_compress_ratios,c->n_layers,c->n_layers+c->n_nextn,c->n_nextn);
        if(n_dspark) fprintf(stderr," or %d (+ %d dspark_target_layer_ids)",
                             c->n_layers+n_dspark,n_dspark);
        fprintf(stderr,"\n");
        return -1; }
    for(int i=0;i<c->n_compress_ratios;i++){
        int rt=c->compress_ratios[i];
        /* Only 0 (sliding window only), 4 (CSA, overlapped) and 128 (HCA) are
         * meaningful: dsv4_coff and dsv4_has_indexer branch on exactly these,
         * so an unexpected value would silently take the HCA path. */
        if(rt!=0 && rt!=4 && rt!=128){
            fprintf(stderr,"config: compress_ratios[%d]=%d — expected 0, 4 or 128\n",i,rt);
            return -1; }
    }
    if(c->rope_head_dim & 1){ fprintf(stderr,"config: qk_rope_head_dim=%d must be even\n",c->rope_head_dim); return -1; }
    if((c->n_heads*c->head_dim) % c->o_groups){
        fprintf(stderr,"config: n_heads*head_dim=%d not divisible by o_groups=%d\n",
                c->n_heads*c->head_dim,c->o_groups); return -1; }
    if(c->index_head_dim < c->rope_head_dim){
        fprintf(stderr,"config: index_head_dim=%d < qk_rope_head_dim=%d\n",c->index_head_dim,c->rope_head_dim); return -1; }
    return 0;
}

/* A compressor with ratio 4 overlaps its group with the previous one, which
 * doubles the width of its kv/gate projections and its positional embedding.
 * `coff` (1 + overlap) is that factor; it is the single place the CSA-vs-HCA
 * shape difference comes from. */
static inline int dsv4_coff(int ratio){ return ratio==4 ? 2 : 1; }
static inline int dsv4_has_indexer(int ratio){ return ratio==4; }

/* ---------------- manifest ---------------- */

typedef struct { DSV4Tensor *v; int n, cap; int overflow; } DSV4List;

static void dsv4_push(DSV4List *L, int role, int optional, int64_t d0, int64_t d1,
                      const char *fmt, ...){
    if(L->n>=L->cap){ L->overflow=1; return; }
    DSV4Tensor *t=&L->v[L->n++];
    va_list ap; va_start(ap,fmt);
    vsnprintf(t->name,sizeof t->name,fmt,ap);
    va_end(ap);
    t->d0=d0; t->d1=d1; t->role=role; t->optional=optional;
}

/* Every tensor a Compressor owns, under `prefix`. `hd` is its head dim: the
 * attention compressor uses head_dim (512), the indexer's own compressor uses
 * index_head_dim (128). */
static void dsv4_compressor_tensors(DSV4List *L, const DSV4Cfg *c, const char *prefix,
                                    int ratio, int hd){
    int coff=dsv4_coff(ratio);
    /* wkv/wgate are NOT quantized. Every other projection in this model is fp8
     * with a `.scale` sidecar, and these were classified with them by analogy;
     * the checkpoint has no `<prefix>.wkv.scale` at all, and the tensors are
     * bf16. The evidence closed exactly: 41 attention compressors + 21 indexer
     * compressors, two tensors each, is 124 -- precisely the gap between the
     * 490 fp8 tensors this manifest expected and the 390 the checkpoint holds.
     * Small enough to be plausible either way, which is why the sidecar's
     * absence, not the byte count, is what settles it. */
    dsv4_push(L,DSV4_T_PLAIN,0,ratio,(int64_t)coff*hd, "%s.ape",   prefix);
    dsv4_push(L,DSV4_T_PLAIN,0,(int64_t)coff*hd,c->dim, "%s.wkv.weight",   prefix);
    dsv4_push(L,DSV4_T_PLAIN,0,(int64_t)coff*hd,c->dim, "%s.wgate.weight", prefix);
    dsv4_push(L,DSV4_T_PLAIN,0,hd,0,                    "%s.norm.weight",  prefix);
}

/* One transformer layer. Layer index decides three things: the compression
 * class (compress_ratios[l]), whether an indexer exists (ratio==4), and
 * whether routing is by hash table (l < n_hash_layers). */
static void dsv4_layer_tensors(DSV4List *L, const DSV4Cfg *c, int l){
    int ratio = c->compress_ratios[l];
    int hs    = c->n_heads*c->head_dim;              /* 32768 */
    int hcmix = (2+c->hc_mult)*c->hc_mult;           /* 24 */

    /* mHC: two independent gate sets per layer, one per sublayer boundary. */
    dsv4_push(L,DSV4_T_PLAIN,0,hcmix,(int64_t)c->hc_mult*c->dim, "layers.%d.hc_attn_fn",l);
    dsv4_push(L,DSV4_T_PLAIN,0,hcmix,0,                          "layers.%d.hc_attn_base",l);
    dsv4_push(L,DSV4_T_PLAIN,0,3,0,                              "layers.%d.hc_attn_scale",l);
    dsv4_push(L,DSV4_T_PLAIN,0,hcmix,(int64_t)c->hc_mult*c->dim, "layers.%d.hc_ffn_fn",l);
    dsv4_push(L,DSV4_T_PLAIN,0,hcmix,0,                          "layers.%d.hc_ffn_base",l);
    dsv4_push(L,DSV4_T_PLAIN,0,3,0,                              "layers.%d.hc_ffn_scale",l);
    dsv4_push(L,DSV4_T_PLAIN,0,c->dim,0,                         "layers.%d.attn_norm.weight",l);
    dsv4_push(L,DSV4_T_PLAIN,0,c->dim,0,                         "layers.%d.ffn_norm.weight",l);

    /* Attention. Q is LoRA-factored (wq_a -> q_norm -> wq_b); KV collapses to
     * ONE head_dim vector shared by every head as both key and value; O is a
     * GROUPED LoRA (wo_a is [o_groups*o_lora_rank, hs/o_groups], applied per
     * group, then wo_b maps the concatenation back to dim). */
    dsv4_push(L,DSV4_T_PLAIN,0,c->n_heads,0,                     "layers.%d.attn.attn_sink",l);
    dsv4_push(L,DSV4_T_FP8,  0,c->q_lora_rank,c->dim,            "layers.%d.attn.wq_a.weight",l);
    dsv4_push(L,DSV4_T_PLAIN,0,c->q_lora_rank,0,                 "layers.%d.attn.q_norm.weight",l);
    dsv4_push(L,DSV4_T_FP8,  0,hs,c->q_lora_rank,                "layers.%d.attn.wq_b.weight",l);
    dsv4_push(L,DSV4_T_FP8,  0,c->head_dim,c->dim,               "layers.%d.attn.wkv.weight",l);
    dsv4_push(L,DSV4_T_PLAIN,0,c->head_dim,0,                    "layers.%d.attn.kv_norm.weight",l);
    dsv4_push(L,DSV4_T_FP8,  0,(int64_t)c->o_groups*c->o_lora_rank, hs/c->o_groups,
                                                                 "layers.%d.attn.wo_a.weight",l);
    dsv4_push(L,DSV4_T_FP8,  0,c->dim,(int64_t)c->o_groups*c->o_lora_rank,
                                                                 "layers.%d.attn.wo_b.weight",l);

    if(ratio>0){
        char pfx[96]; snprintf(pfx,sizeof pfx,"layers.%d.attn.compressor",l);
        dsv4_compressor_tensors(L,c,pfx,ratio,c->head_dim);
        if(dsv4_has_indexer(ratio)){
            dsv4_push(L,DSV4_T_FP8,0,(int64_t)c->index_n_heads*c->index_head_dim,c->q_lora_rank,
                      "layers.%d.attn.indexer.wq_b.weight",l);
            dsv4_push(L,DSV4_T_PLAIN,0,c->index_n_heads,c->dim,
                      "layers.%d.attn.indexer.weights_proj.weight",l);
            char ip[128]; snprintf(ip,sizeof ip,"layers.%d.attn.indexer.compressor",l);
            /* The indexer carries its OWN compressor, at index_head_dim and
             * always ratio 4 (it only exists on ratio-4 layers). */
            dsv4_compressor_tensors(L,c,ip,4,c->index_head_dim);
        }
    }

    /* MoE. The router weight is present on every layer; the noaux_tc bias only
     * on score-routed layers, and the tid2eid table only on hash layers --
     * the reference makes them mutually exclusive in Gate.__init__. */
    int hash = l < c->n_hash_layers;
    dsv4_push(L,DSV4_T_PLAIN,0,c->n_routed_experts,c->dim, "layers.%d.ffn.gate.weight",l);
    if(hash) dsv4_push(L,DSV4_T_I32,  0,c->vocab_size,c->n_activated_experts,
                                                          "layers.%d.ffn.gate.tid2eid",l);
    else     dsv4_push(L,DSV4_T_PLAIN,0,c->n_routed_experts,0,
                                                          "layers.%d.ffn.gate.bias",l);

    for(int e=0;e<c->n_routed_experts;e++){
        dsv4_push(L,DSV4_T_FP4,0,c->moe_inter_dim,c->dim, "layers.%d.ffn.experts.%d.w1.weight",l,e);
        dsv4_push(L,DSV4_T_FP4,0,c->moe_inter_dim,c->dim, "layers.%d.ffn.experts.%d.w3.weight",l,e);
        dsv4_push(L,DSV4_T_FP4,0,c->dim,c->moe_inter_dim, "layers.%d.ffn.experts.%d.w2.weight",l,e);
    }
    /* The shared expert is built WITHOUT the fp4 dtype in the reference, so it
     * is not a routed-expert-format tensor. */
    for(int s=0;s<c->n_shared_experts;s++){
        dsv4_push(L,DSV4_T_FP8,0,c->moe_inter_dim,c->dim, "layers.%d.ffn.shared_experts.w1.weight",l);
        dsv4_push(L,DSV4_T_FP8,0,c->moe_inter_dim,c->dim, "layers.%d.ffn.shared_experts.w3.weight",l);
        dsv4_push(L,DSV4_T_FP8,0,c->dim,c->moe_inter_dim, "layers.%d.ffn.shared_experts.w2.weight",l);
    }
}

static void dsv4_global_tensors(DSV4List *L, const DSV4Cfg *c){
    dsv4_push(L,DSV4_T_PLAIN,0,c->vocab_size,c->dim, "embed.weight");
    dsv4_push(L,DSV4_T_PLAIN,0,c->dim,0,             "norm.weight");
    dsv4_push(L,DSV4_T_FP8,  0,c->vocab_size,c->dim, "head.weight");
    /* The head's OWN hyper-connection gates, which collapse the hc_mult
     * residual streams to one before the output norm. Note the row count is
     * hc_mult, not (2+hc)*hc: only the `pre` family exists here, because
     * nothing is expanded again afterwards (see dsv4_hc_collapse). */
    dsv4_push(L,DSV4_T_PLAIN,0,c->hc_mult,(int64_t)c->hc_mult*c->dim, "hc_head_fn");
    dsv4_push(L,DSV4_T_PLAIN,0,c->hc_mult,0,                          "hc_head_base");
    dsv4_push(L,DSV4_T_PLAIN,0,1,0,                                   "hc_head_scale");
}

/* Fill `out` with the manifest for the MAIN STACK. Returns the count, or -1 if
 * `cap` was too small (the caller sizes with dsv4_manifest_count).
 *
 * DELIBERATE GAP: the speculative head -- one MTP layer on the preview, three
 * DSpark modules on -0731, whose compression classes are the trailing
 * n_spec compress_ratios entries -- is NOT emitted. Its tensor NAMES are not
 * known here (the published reference inference code does not build it, so
 * there is nothing to transcribe) and inventing names would produce a manifest
 * that fails against the real checkpoint for a reason that looks like a shape
 * bug. A verifier walking this manifest should therefore treat unmatched
 * `layers.43.*` / `mtp.*` tensors in a container as EXPECTED-UNKNOWN, not as
 * corruption. Filling this in needs one look at the checkpoint's own tensor
 * index. */
static int dsv4_manifest(const DSV4Cfg *c, DSV4Tensor *out, int cap){
    DSV4List L={out,0,cap,0};
    dsv4_global_tensors(&L,c);
    for(int l=0;l<c->n_layers;l++) dsv4_layer_tensors(&L,c,l);
    return L.overflow ? -1 : L.n;
}

/* Exact manifest size without materializing it -- lets a caller allocate once.
 * Kept as arithmetic rather than a dry-run so a mismatch between this and
 * dsv4_manifest is a test failure, not a silent short buffer. */
static int dsv4_manifest_count(const DSV4Cfg *c){
    int n=6;                                   /* embed, norm, head + the head's three hc gates */
    for(int l=0;l<c->n_layers;l++){
        n += 8;                                /* mHC (6) + attn_norm + ffn_norm */
        n += 8;                                /* attention proper */
        int ratio=c->compress_ratios[l];
        if(ratio>0){
            n += 4;                            /* attention compressor */
            if(dsv4_has_indexer(ratio)) n += 2 + 4;   /* indexer + its compressor */
        }
        n += 1 + 1;                            /* gate.weight + (bias | tid2eid) */
        n += 3*c->n_routed_experts;
        n += 3*(c->n_shared_experts?1:0);
    }
    return n;
}

/* ---------------- sizes ---------------- */

/* Logical element count (parameters), independent of storage format. This is
 * what the published "284B total / 13B active" figures count, and comparing
 * against them is the one external check on this whole derivation available
 * without the checkpoint. */
static int64_t dsv4_tensor_numel(const DSV4Tensor *t){
    return t->d1 ? t->d0*t->d1 : t->d0;
}

/* On-disk bytes for a tensor INCLUDING its scale sidecar, per role:
 *   FP8: O*I raw e4m3 + ceil(O/128)*ceil(I/128) ue8m0 scale bytes
 *   FP4: O*ceil(I/2) nibbles + O*ceil(I/32) ue8m0 scale bytes  (mxfp4)
 *   I32: 4 bytes/element;  PLAIN: bf16, 2 bytes/element */
static int64_t dsv4_tensor_bytes(const DSV4Tensor *t){
    int64_t O=t->d0, I=t->d1?t->d1:1, n=O*I;
    switch(t->role){
        case DSV4_T_FP8: return n + dsv4_nblk128(O)*dsv4_nblk128(I);
        case DSV4_T_FP4: return O*((I+1)/2) + O*((I+31)/32);
        case DSV4_T_I32: return n*4;
        default:         return n*2;
    }
}

/* Parameters resident in one forward pass for one token: everything except the
 * routed experts that are NOT selected. */
static int64_t dsv4_active_params(const DSV4Cfg *c, const DSV4Tensor *m, int n){
    int64_t act=0;
    for(int i=0;i<n;i++){
        const DSV4Tensor *t=&m[i];
        if(t->role==DSV4_T_FP4){            /* routed expert: only topk of them fire */
            act += dsv4_tensor_numel(t)*c->n_activated_experts/c->n_routed_experts;
        } else if(t->role==DSV4_T_I32){
            continue;                        /* a lookup table, not compute */
        } else if(!strcmp(t->name,"embed.weight")){
            continue;                        /* one row read, not a matmul */
        } else act += dsv4_tensor_numel(t);
    }
    return act;
}

#endif /* COLIBRI_DSV4_MODEL_H */
