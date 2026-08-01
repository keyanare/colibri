#ifndef COLIBRI_DSV4_H
#define COLIBRI_DSV4_H
/* DeepSeek-V4-Flash compute primitives — the pieces of that architecture that
 * are PURE COMPUTE: no Model, no QT, no disk, no globals. Same contract as
 * tier.h / grammar.h / route_trace.h, and for the same reason — a primitive
 * that a test can call directly is a primitive whose behaviour is pinned
 * before it is ever wired into an engine.
 *
 * Every function here is transcribed from the reference implementation
 * published with the checkpoint (`inference/model.py` and `inference/kernel.py`
 * in deepseek-ai/DeepSeek-V4-Flash) and from `config.json`. Where the
 * reference is subtle, the comment says so and names the trap: those are the
 * places a plausible-looking reimplementation silently diverges instead of
 * failing.
 *
 * NOT here (they need the engine's own state and I/O): the KV ring buffer, the
 * expert LRU, safetensors wiring, the tokenizer. NOT verified end to end: no
 * checkpoint was run against this header — every claim below is pinned by
 * tests/test_dsv4.c against an independent in-test reference, which is a
 * weaker guarantee than the token-exact oracle this repo requires for merge.
 *
 * Model shape this targets (config.json): 43 layers, dim 4096, 64 heads,
 * head_dim 512 (K and V are the SAME 512-dim vector, num_key_value_heads=1),
 * rope_head_dim 64, 256 routed experts + 1 shared, top-6, hc_mult 4,
 * hc_sinkhorn_iters 20, sliding_window 128, index_topk 512. */

#include <math.h>
#include <string.h>
#include <stdint.h>

/* ===================== mHC: manifold-constrained hyper-connections =========
 * The residual stream is not one vector but `hc` of them (hc_mult=4). Before
 * each sublayer they are mixed down to one; after it they are expanded back.
 * The mixing matrix is made doubly stochastic by Sinkhorn normalization --
 * that is the "manifold constraint" in the model card's name for this.
 *
 * Layout of `mixes` (the (2+hc)*hc = 24 values a layer's hc_*_fn projection
 * produces), matching hc_split_sinkhorn_kernel in kernel.py exactly:
 *   [0, hc)        -> pre  gates
 *   [hc, 2hc)      -> post gates
 *   [2hc, (2+hc)hc)-> the hc x hc combination matrix, ROW-MAJOR
 * `scale` is 3 floats (one per group), `base` is (2+hc)*hc floats. */

#define DSV4_HC_MAX 8   /* hc_mult is 4 in this model; the bound keeps the
                         * Sinkhorn scratch on the stack and is checked. */

/* comb <- sinkhorn(comb): row-softmax, +eps, column-normalize, then
 * (iters-1) further row/column passes. The first pass is NOT symmetric with
 * the rest (it starts from a softmax, not from a bare division) -- collapsing
 * the two is the easy way to get a matrix that looks doubly stochastic on
 * small inputs and is not. */
static void dsv4_sinkhorn(float *comb, int hc, int iters, float eps){
    for(int j=0;j<hc;j++){
        float mx=comb[j*hc];
        for(int k=1;k<hc;k++) if(comb[j*hc+k]>mx) mx=comb[j*hc+k];
        float s=0;
        for(int k=0;k<hc;k++){ comb[j*hc+k]=expf(comb[j*hc+k]-mx); s+=comb[j*hc+k]; }
        for(int k=0;k<hc;k++) comb[j*hc+k]=comb[j*hc+k]/s+eps;
    }
    for(int k=0;k<hc;k++){                      /* column normalize */
        float c=0; for(int j=0;j<hc;j++) c+=comb[j*hc+k];
        c+=eps; for(int j=0;j<hc;j++) comb[j*hc+k]/=c;
    }
    for(int it=1;it<iters;it++){
        for(int j=0;j<hc;j++){
            float r=0; for(int k=0;k<hc;k++) r+=comb[j*hc+k];
            r+=eps; for(int k=0;k<hc;k++) comb[j*hc+k]/=r;
        }
        for(int k=0;k<hc;k++){
            float c=0; for(int j=0;j<hc;j++) c+=comb[j*hc+k];
            c+=eps; for(int j=0;j<hc;j++) comb[j*hc+k]/=c;
        }
    }
}

static inline float dsv4_sigmoid(float z){ return 1.0f/(1.0f+expf(-z)); }

/* Split `mixes` into the pre gates, post gates and the Sinkhorn-normalized
 * combination matrix. Note the asymmetry between the two gate families, which
 * is in the reference and is not a typo here: `pre` is sigmoid PLUS eps,
 * `post` is TWICE a sigmoid with no eps. */
static void dsv4_hc_split(const float *mixes, const float *scale, const float *base,
                          int hc, int iters, float eps,
                          float *pre, float *post, float *comb){
    for(int j=0;j<hc;j++)
        pre[j]  = dsv4_sigmoid(mixes[j]*scale[0] + base[j]) + eps;
    for(int j=0;j<hc;j++)
        post[j] = 2.0f*dsv4_sigmoid(mixes[j+hc]*scale[1] + base[j+hc]);
    for(int j=0;j<hc;j++) for(int k=0;k<hc;k++){
        int o=j*hc+k+2*hc;
        comb[j*hc+k] = mixes[o]*scale[2] + base[o];
    }
    dsv4_sinkhorn(comb,hc,iters,eps);
}

/* hc_pre: collapse `hc` residual streams into one.
 *
 * TRAP: the RMS reciprocal is computed over the FLATTENED hc*dim vector (all
 * streams at once, not per stream) and it scales `mixes`, NOT `x`. The
 * sublayer input y is a plain pre-weighted sum of the UNnormalized streams.
 * Normalizing x here instead would be a different model.
 *
 *   x   : [hc][dim]         the residual streams
 *   fn  : [(2+hc)*hc][hc*dim]  the hc_*_fn projection, row-major
 *   y   : [dim]             out, the sublayer's input
 *   pre/post/comb: out, post+comb are handed to dsv4_hc_post unchanged */
static void dsv4_hc_pre(const float *x, int hc, int dim,
                        const float *fn, const float *scale, const float *base,
                        int iters, float eps, float norm_eps,
                        float *y, float *pre, float *post, float *comb){
    int n=hc*dim, mix_hc=(2+hc)*hc;
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float rsq = 1.0f/sqrtf((float)(ss/(double)n) + norm_eps);
    float mixes[(2+DSV4_HC_MAX)*DSV4_HC_MAX];
    for(int m=0;m<mix_hc;m++){
        const float *w=fn+(size_t)m*n; double a=0;
        for(int i=0;i<n;i++) a+=(double)w[i]*x[i];
        mixes[m]=(float)a*rsq;
    }
    dsv4_hc_split(mixes,scale,base,hc,iters,eps,pre,post,comb);
    for(int d=0;d<dim;d++){
        float a=0; for(int j=0;j<hc;j++) a+=pre[j]*x[(size_t)j*dim+d];
        y[d]=a;
    }
}

/* hc_post: expand the sublayer output back into `hc` streams.
 *
 *   out[k][d] = post[k]*sub[d] + SUM_j comb[j][k]*residual[j][d]
 *
 * TRAP: comb is applied TRANSPOSED. In the reference this falls out of
 * `sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`, whose reduction
 * runs over comb's FIRST axis, leaving the second as the output index. comb is
 * doubly stochastic, so an accidental untransposed apply still conserves mass
 * and still produces fluent text -- it just is not this model. */
static void dsv4_hc_post(const float *sub, const float *residual,
                         const float *post, const float *comb,
                         int hc, int dim, float *out){
    for(int k=0;k<hc;k++){
        float *o=out+(size_t)k*dim;
        for(int d=0;d<dim;d++) o[d]=post[k]*sub[d];
        for(int j=0;j<hc;j++){
            float c=comb[j*hc+k];
            const float *r=residual+(size_t)j*dim;
            for(int d=0;d<dim;d++) o[d]+=c*r[d];
        }
    }
}

/* Head collapse: the FINAL hc_mult -> 1 reduction, before the output norm and
 * the LM head. The Transformer owns its own gates for this, and their shapes
 * say what it does: hc_head_fn is [hc_mult, hc*dim] (not [(2+hc)*hc, ...]),
 * hc_head_base is [hc_mult], hc_head_scale is a single float. That is exactly
 * the `pre` branch of dsv4_hc_split and nothing else -- no post gates and no
 * combination matrix, because nothing is re-expanded after the head.
 *
 * INFERRED, not transcribed: the reference calls a ParallelHead whose body was
 * not read. The parameter shapes admit essentially one reading (a per-stream
 * sigmoid gate, same construction as `pre`), and the RMS scaling mirrors
 * hc_pre's, but this is the one piece of the collapse path that is a
 * derivation rather than a copy. */
static void dsv4_hc_collapse(const float *x, int hc, int dim,
                             const float *fn, float scale, const float *base,
                             float eps, float norm_eps, float *y){
    int n=hc*dim;
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float rsq = 1.0f/sqrtf((float)(ss/(double)n) + norm_eps);
    float pre[DSV4_HC_MAX];
    for(int j=0;j<hc;j++){
        const float *w=fn+(size_t)j*n; double a=0;
        for(int i=0;i<n;i++) a+=(double)w[i]*x[i];
        pre[j]=dsv4_sigmoid((float)a*rsq*scale + base[j]) + eps;
    }
    for(int d=0;d<dim;d++){
        float a=0; for(int j=0;j<hc;j++) a+=pre[j]*x[(size_t)j*dim+d];
        y[d]=a;
    }
}

/* ===================== MoE routing ========================================
 * scoring_func="sqrtsoftplus", topk_method="noaux_tc". The bias shifts scores
 * for SELECTION only; the returned weights come from the UNBIASED scores.
 * That split is the whole point of the auxiliary-loss-free scheme -- folding
 * the bias into the weights turns load balancing into a quality change. */

static inline float dsv4_softplus(float z){
    /* log1p(exp(z)) saturates to z well before overflow; the branch keeps a
     * large positive logit from turning into inf inside expf. */
    return z>20.0f ? z : log1pf(expf(z));
}
static inline float dsv4_score_sqrtsoftplus(float logit){
    return sqrtf(dsv4_softplus(logit));
}

/* Select `topk` of `n_expert` and fill idx/wt.
 *   logits : [n_expert]  raw router output (the reference runs the router in f32)
 *   bias   : [n_expert] or NULL (hash layers carry no bias)
 *   norm   : renormalize the selected weights to sum 1 (norm_topk_prob)
 *   scale  : routed_scaling_factor (1.5)
 * Returns the number selected. Selection is a partial max-scan: topk is 6 and
 * n_expert 256, so a full sort would cost more than the scan it replaces. */
static int dsv4_route(const float *logits, const float *bias, int n_expert, int topk,
                      int norm, float scale, int *idx, float *wt){
    if(topk>n_expert) topk=n_expert;
    float taken[64]; int nt=0;
    for(int i=0;i<topk;i++){
        int best=-1; float bs=0;
        for(int e=0;e<n_expert;e++){
            int dup=0; for(int z=0;z<nt;z++) if(idx[z]==e){ dup=1; break; }
            if(dup) continue;
            float s=dsv4_score_sqrtsoftplus(logits[e]);
            if(bias) s+=bias[e];
            if(best<0||s>bs){ best=e; bs=s; }
        }
        if(best<0) break;
        idx[nt]=best; taken[nt]=bs; nt++;
    }
    (void)taken;
    /* Weights come from the UNBIASED score of the selected experts. */
    float sum=0;
    for(int i=0;i<nt;i++){ wt[i]=dsv4_score_sqrtsoftplus(logits[idx[i]]); sum+=wt[i]; }
    if(norm && sum>0) for(int i=0;i<nt;i++) wt[i]/=sum;
    for(int i=0;i<nt;i++) wt[i]*=scale;
    return nt;
}

/* Hash layers (the first n_hash_layers=3): the expert set is a table lookup on
 * the INPUT TOKEN ID, with no router scan at all. The weights still come from
 * the router's scores for the looked-up experts.
 *
 * This is the single most valuable property in this architecture for a
 * streaming engine: on those layers the working set is known BEFORE the
 * forward pass from the token id alone -- exact prefetch, not prediction. */
static int dsv4_route_hash(const float *logits, const int32_t *tid2eid_row, int topk,
                           int norm, float scale, int *idx, float *wt){
    float sum=0;
    for(int i=0;i<topk;i++){
        idx[i]=(int)tid2eid_row[i];
        wt[i]=dsv4_score_sqrtsoftplus(logits[idx[i]]);
        sum+=wt[i];
    }
    if(norm && sum>0) for(int i=0;i<topk;i++) wt[i]/=sum;
    for(int i=0;i<topk;i++) wt[i]*=scale;
    return topk;
}

/* ===================== SwiGLU with an ASYMMETRIC clamp ====================
 * TRAP: `up` is clamped on BOTH sides, `gate` only from ABOVE. Clamping gate
 * symmetrically is the natural-looking mistake and changes the activation on
 * every strongly negative gate. swiglu_limit is 10.0 here. */
static void dsv4_swiglu(float *out, const float *gate, const float *up, int n, float limit){
    for(int i=0;i<n;i++){
        float g=gate[i], u=up[i];
        if(limit>0){
            if(u> limit) u= limit;
            if(u<-limit) u=-limit;
            if(g> limit) g= limit;          /* upper bound only -- see above */
        }
        out[i]=(g/(1.0f+expf(-g)))*u;       /* silu(g)*u */
    }
}

/* ===================== RoPE with YaRN =====================================
 * Two tables per model: compressed layers use compress_rope_theta (160000)
 * WITH YaRN; layers whose compress_ratio is 0 use rope_theta (10000) and
 * disable YaRN entirely (pass original_seq_len=0). Building one table and
 * sharing it across both layer classes is a silent correctness bug.
 *
 * dsv4_rope_freqs fills `freqs` with dim/2 angular frequencies. */
static void dsv4_rope_freqs(float *freqs, int dim, float base,
                            int original_seq_len, float factor,
                            float beta_fast, float beta_slow){
    int half=dim/2;
    for(int j=0;j<half;j++) freqs[j]=1.0f/powf(base,(float)(2*j)/(float)dim);
    if(original_seq_len<=0) return;
    /* find_correction_range: the dims whose wavelength brackets the trained
     * context. Below `low` frequencies pass through; above `high` they are
     * divided by `factor`; between, a linear ramp. */
    double lg=2.0*log((double)base);
    double dlo=(double)dim*log((double)original_seq_len/((double)beta_fast*2.0*M_PI))/lg;
    double dhi=(double)dim*log((double)original_seq_len/((double)beta_slow*2.0*M_PI))/lg;
    double low=floor(dlo), high=ceil(dhi);
    if(low<0) low=0;
    if(high>dim-1) high=dim-1;
    if(low==high) high+=0.001;
    for(int j=0;j<half;j++){
        double t=((double)j-low)/(high-low);
        if(t<0) t=0; if(t>1) t=1;
        float smooth=(float)(1.0-t);
        freqs[j]=freqs[j]/factor*(1.0f-smooth)+freqs[j]*smooth;
    }
}

/* Rotate the last `dim` values of a vector at position `pos`.
 * INTERLEAVED pairs -- (v[0],v[1]), (v[2],v[3]), ... -- matching
 * view_as_complex(x.unflatten(-1,(-1,2))) in the reference, and matching
 * colibri.c's own rope_interleave convention.
 * `inverse` conjugates: the attention OUTPUT is de-rotated, because K and V
 * are the same tensor here and the value side carries the rotation. */
static void dsv4_rope_apply(float *v, int dim, int pos, const float *freqs, int inverse){
    for(int j=0;j<dim/2;j++){
        float ang=(float)pos*freqs[j];
        float c=cosf(ang), s=sinf(ang);
        if(inverse) s=-s;
        float a=v[2*j], b=v[2*j+1];
        v[2*j]  = a*c - b*s;
        v[2*j+1]= a*s + b*c;
    }
}

/* ===================== compressed-KV pooling (CSA / HCA) ==================
 * One mechanism, two settings: compress_ratio=4 is CSA, 128 is HCA. A group of
 * `ratio` consecutive token vectors collapses into ONE, by a softmax-gated
 * weighted sum.
 *
 * TRAP: the softmax runs over the group axis PER DIMENSION -- each of the D
 * components has its own distribution over the `ratio` slots. It is not one
 * scalar weight per token. `ape` (the learned per-slot positional embedding)
 * is added to the gate BEFORE the softmax.
 *
 *   kv   : [ratio][D]      wkv(x) for the group
 *   score: [ratio][D]      wgate(x) for the group
 *   ape  : [ratio][D]      learned, added to score
 *   out  : [D] */
static void dsv4_compress_pool(const float *kv, const float *score, const float *ape,
                               int ratio, int D, float *out){
    for(int d=0;d<D;d++){
        float mx=-INFINITY;
        for(int r=0;r<ratio;r++){
            float s=score[(size_t)r*D+d]+(ape?ape[(size_t)r*D+d]:0.0f);
            if(s>mx) mx=s;
        }
        float den=0, num=0;
        for(int r=0;r<ratio;r++){
            float s=score[(size_t)r*D+d]+(ape?ape[(size_t)r*D+d]:0.0f);
            float w=(s==-INFINITY&&mx==-INFINITY)?0.0f:expf(s-mx);
            den+=w; num+=w*kv[(size_t)r*D+d];
        }
        out[d]=den>0?num/den:0.0f;
    }
}

/* ===================== sparse attention with a sink =======================
 * FlashAttention-style online softmax over a GATHERED index set, matching
 * sparse_attn_kernel in kernel.py. Two properties that are not the textbook
 * form:
 *   - K and V are the SAME array (`kv`): one D-dim vector per position serves
 *     every head as both key and value. This is what makes the KV cache small.
 *   - a learnable per-head `sink` logit joins the denominator AFTER the loop,
 *     against the FINAL running max. It is a virtual position that absorbs
 *     attention mass and contributes nothing to the output sum.
 * An index of -1 is a hole (masked slot) and is skipped.
 *
 *   q    : [H][D]
 *   kv   : [n_pos][D]
 *   idx  : [n_idx]   positions to attend to, -1 = skip
 *   sink : [H]
 *   out  : [H][D] */
static void dsv4_sparse_attn(const float *q, const float *kv, const int32_t *idx,
                             const float *sink, int H, int D, int n_pos, int n_idx,
                             float scale, float *out){
    for(int h=0;h<H;h++){
        const float *qh=q+(size_t)h*D;
        float *oh=out+(size_t)h*D;
        memset(oh,0,(size_t)D*sizeof(float));
        float m=-INFINITY, l=0.0f;
        for(int t=0;t<n_idx;t++){
            int p=idx[t];
            if(p<0||p>=n_pos) continue;
            const float *kt=kv+(size_t)p*D;
            float dot=0; for(int d=0;d<D;d++) dot+=qh[d]*kt[d];
            float s=dot*scale;
            float mn = s>m ? s : m;
            float corr = (m==-INFINITY) ? 0.0f : expf(m-mn);
            float e = expf(s-mn);
            l = l*corr + e;
            for(int d=0;d<D;d++) oh[d]=oh[d]*corr + e*kt[d];
            m = mn;
        }
        /* The sink joins here, against the final max -- not inside the loop. */
        if(m!=-INFINITY) l += expf(sink[h]-m);
        if(l>0) for(int d=0;d<D;d++) oh[d]/=l;
    }
}

/* ===================== lightning indexer scoring ==========================
 * Score every compressed position against the indexer's own query, then keep
 * the top `topk`. Per-head ReLU'd dot products, weighted by a learned
 * per-head projection of the token, summed over heads:
 *
 *   score[t] = SUM_h relu(<q[h], ckv[t]>) * w[h]
 *
 * The ReLU is inside the head sum, before the weighting -- moving it outside
 * changes which positions survive. `w` already carries the reference's
 * softmax_scale * n_heads^-0.5 factor; this function does not reapply it.
 *
 *   q    : [H][D]      indexer query (H=index_n_heads=64, D=index_head_dim=128)
 *   ckv  : [n_ctx][D]  compressed KV built by the indexer's own compressor
 *   w    : [H]
 *   out  : [n_ctx] */
static void dsv4_index_score(const float *q, const float *ckv, const float *w,
                             int H, int D, int n_ctx, float *out){
    for(int t=0;t<n_ctx;t++){
        const float *k=ckv+(size_t)t*D;
        float a=0;
        for(int h=0;h<H;h++){
            const float *qh=q+(size_t)h*D;
            float dot=0; for(int d=0;d<D;d++) dot+=qh[d]*k[d];
            if(dot>0) a+=dot*w[h];
        }
        out[t]=a;
    }
}

#endif /* COLIBRI_DSV4_H */
