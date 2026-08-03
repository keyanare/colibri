/* dsv4.h primitive tests — DeepSeek-V4-Flash compute core.
 *
 * Each block below pins one primitive against an INDEPENDENT reference
 * written inside this file (a different formulation, not a copy of the header's
 * loop), or against an invariant the reference implementation guarantees
 * structurally. Where a formulation is subtle, the test targets the specific
 * way a plausible reimplementation goes wrong -- an untransposed comb, a
 * symmetric swiglu clamp, a sink folded into the loop, a per-token instead of
 * per-dimension pooling softmax. Those are the assertions that carry weight;
 * "it produces finite numbers" is not.
 *
 * SCOPE. This is unit coverage of pure compute. No checkpoint was run: the
 * engine-level guarantee this repo requires for merge is a token-exact oracle,
 * which needs weights and a tokenizer neither of which exist in-tree yet. Read
 * every "ok" below as "matches the published reference implementation as
 * transcribed", not as "matches the model". */
#include "../dsv4.h"
#include "../quant.h"      /* matmul_mxfp4 -- the routed-expert kernel (NEON test below) */
#include <stdio.h>
#include <stdlib.h>

static int fails=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n",__FILE__,__LINE__,#c); fails++; } }while(0)
#define CLOSE(a,b,tol) do{ float _d=fabsf((a)-(b)); if(!(_d<=(tol))){ \
    printf("FAIL %s:%d: |%g - %g| = %g > %g\n",__FILE__,__LINE__,(double)(a),(double)(b),(double)_d,(double)(tol)); fails++; } }while(0)

static uint64_t rng=0xD54C0DE1234ull;
static float rndf(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
    return ((int64_t)(rng&0xFFFFF)-0x80000)/(float)0x80000; }

/* ---------------- mHC / Sinkhorn ---------------- */

/* Sinkhorn drives the matrix toward double stochasticity, but the reference's
 * loop ENDS ON A COLUMN PASS. So the guarantee is asymmetric and this test
 * pins it that way rather than asserting a symmetry the algorithm does not
 * deliver: columns are exact (they were just normalized), rows are close and
 * get closer with iterations. Asserting a tight row sum here would be
 * asserting a bug -- it would only pass for an implementation that added a
 * trailing row pass the reference does not have. */
static void test_sinkhorn_doubly_stochastic(void){
    for(int trial=0;trial<8;trial++){
        int hc=4; float comb[16], seed[16];
        for(int i=0;i<16;i++) seed[i]=comb[i]=rndf()*4.0f;
        dsv4_sinkhorn(comb,hc,20,1e-6f);
        for(int k=0;k<hc;k++){                       /* exact: last pass */
            float c=0; for(int j=0;j<hc;j++) c+=comb[j*hc+k];
            CLOSE(c,1.0f,1e-5f);
        }
        float row_err20=0;
        for(int j=0;j<hc;j++){
            float r=0; for(int k=0;k<hc;k++) r+=comb[j*hc+k];
            row_err20+=fabsf(r-1.0f);
        }
        CHECK(row_err20<0.15f);                      /* close, not exact */
        /* Convergence is real: 20 iterations must beat 1 on the row residual. */
        float one[16]; memcpy(one,seed,sizeof one);
        dsv4_sinkhorn(one,hc,1,1e-6f);
        float row_err1=0;
        for(int j=0;j<hc;j++){
            float r=0; for(int k=0;k<hc;k++) r+=one[j*hc+k];
            row_err1+=fabsf(r-1.0f);
        }
        CHECK(row_err20<=row_err1);
        for(int i=0;i<16;i++) CHECK(comb[i]>0.0f);   /* softmax + eps, never zero */
    }
}

/* One Sinkhorn iteration must NOT be enough to reach double stochasticity from
 * a skewed start -- if it were, the 20-iteration config value would be
 * unobservable and this suite could not tell a 1-iteration bug from a correct
 * implementation. Pins that the iteration count is load-bearing. */
static void test_sinkhorn_iters_matter(void){
    int hc=4; float a[16], b[16];
    for(int i=0;i<16;i++){ a[i]=b[i]=(float)((i*7)%5)*3.0f; }
    dsv4_sinkhorn(a,hc,1,1e-6f);
    dsv4_sinkhorn(b,hc,20,1e-6f);
    float diff=0; for(int i=0;i<16;i++) diff+=fabsf(a[i]-b[i]);
    CHECK(diff>1e-3f);
    float worst=0;
    for(int k=0;k<hc;k++){ float c=0; for(int j=0;j<hc;j++) c+=a[j*hc+k];
        if(fabsf(c-1.0f)>worst) worst=fabsf(c-1.0f); }
    CHECK(worst<2e-3f);   /* 1 iter already ends on a column pass... */
    float wr=0;
    for(int j=0;j<hc;j++){ float r=0; for(int k=0;k<hc;k++) r+=a[j*hc+k];
        if(fabsf(r-1.0f)>wr) wr=fabsf(r-1.0f); }
    CHECK(wr>1e-4f);      /* ...but the ROWS are not converged yet. */
}

/* Gate families are deliberately asymmetric in the reference: pre = sigmoid+eps
 * (so strictly in (eps, 1+eps)), post = 2*sigmoid (so in (0,2), NO eps). */
static void test_hc_split_gate_ranges(void){
    int hc=4, mix=(2+4)*4;
    float mixes[24], base[24], scale[3]={1.3f,-0.7f,2.0f};
    float pre[4], post[4], comb[16];
    for(int i=0;i<mix;i++){ mixes[i]=rndf()*6.0f; base[i]=rndf(); }
    dsv4_hc_split(mixes,scale,base,hc,20,1e-6f,pre,post,comb);
    for(int j=0;j<hc;j++){
        CHECK(pre[j]>1e-6f && pre[j]<1.0f+1e-5f);
        float ref=1.0f/(1.0f+expf(-(mixes[j]*scale[0]+base[j])))+1e-6f;
        CLOSE(pre[j],ref,1e-6f);
        CHECK(post[j]>0.0f && post[j]<2.0f);
        float refp=2.0f/(1.0f+expf(-(mixes[j+hc]*scale[1]+base[j+hc])));
        CLOSE(post[j],refp,1e-6f);
    }
}

/* hc_pre's weighted sum, checked against a direct reference. Also pins the
 * trap: the RMS reciprocal scales `mixes`, not `x`, so y is a sum of the
 * UNnormalized streams and scaling x by c must scale y by exactly c. */
static void test_hc_pre_sum(void){
    enum { HC=4, DIM=6, MIX=(2+HC)*HC, N=HC*DIM };
    float x[N], fn[MIX*N], base[MIX], scale[3]={0.9f,1.1f,0.8f};
    float y[DIM], pre[HC], post[HC], comb[HC*HC];
    for(int i=0;i<N;i++) x[i]=rndf();
    for(int i=0;i<MIX*N;i++) fn[i]=rndf()*0.3f;
    for(int i=0;i<MIX;i++) base[i]=rndf();
    dsv4_hc_pre(x,HC,DIM,fn,scale,base,20,1e-6f,1e-6f,y,pre,post,comb);
    for(int d=0;d<DIM;d++){
        float ref=0; for(int j=0;j<HC;j++) ref+=pre[j]*x[j*DIM+d];
        CLOSE(y[d],ref,1e-5f);
    }
    /* Homogeneity: y(c*x) == c*y(x) would FAIL if x were RMS-normalized
     * before the weighted sum (normalization is scale-invariant). The gates
     * do shift, so compare against a re-run rather than assuming equality. */
    float x2[N], y2[DIM], pre2[HC], post2[HC], comb2[HC*HC];
    for(int i=0;i<N;i++) x2[i]=x[i]*3.0f;
    dsv4_hc_pre(x2,HC,DIM,fn,scale,base,20,1e-6f,1e-6f,y2,pre2,post2,comb2);
    for(int d=0;d<DIM;d++){
        float ref=0; for(int j=0;j<HC;j++) ref+=pre2[j]*x2[j*DIM+d];
        CLOSE(y2[d],ref,3e-5f);
    }
    /* And the mixes are RMS-scaled, so tripling x must NOT triple them: the
     * gates change only a little. If x were used raw, pre would saturate. */
    float moved=0; for(int j=0;j<HC;j++) moved+=fabsf(pre2[j]-pre[j]);
    CHECK(moved<0.5f);
}

/* hc_post: comb is applied TRANSPOSED. Built with a deliberately asymmetric
 * comb so an untransposed implementation cannot pass. */
static void test_hc_post_transpose(void){
    enum { HC=3, DIM=4 };
    float sub[DIM], resid[HC*DIM], post[HC], comb[HC*HC], out[HC*DIM];
    for(int d=0;d<DIM;d++) sub[d]=rndf();
    for(int i=0;i<HC*DIM;i++) resid[i]=rndf();
    for(int j=0;j<HC;j++) post[j]=0.5f+0.3f*(float)j;
    /* asymmetric on purpose: comb[j][k] != comb[k][j] everywhere off-diagonal */
    float raw[HC*HC]={0.10f,0.70f,0.20f,
                      0.60f,0.15f,0.25f,
                      0.30f,0.15f,0.55f};
    memcpy(comb,raw,sizeof raw);
    dsv4_hc_post(sub,resid,post,comb,HC,DIM,out);
    for(int k=0;k<HC;k++) for(int d=0;d<DIM;d++){
        float ref=post[k]*sub[d];
        for(int j=0;j<HC;j++) ref+=comb[j*HC+k]*resid[j*DIM+d];
        CLOSE(out[k*DIM+d],ref,1e-6f);
    }
    /* The untransposed variant must give a DIFFERENT answer -- otherwise this
     * test proves nothing about orientation. */
    int differs=0;
    for(int k=0;k<HC && !differs;k++) for(int d=0;d<DIM;d++){
        float wrong=post[k]*sub[d];
        for(int j=0;j<HC;j++) wrong+=comb[k*HC+j]*resid[j*DIM+d];
        if(fabsf(wrong-out[k*DIM+d])>1e-4f){ differs=1; break; }
    }
    CHECK(differs);
}

/* The head's collapse is the `pre` branch alone: hc_mult gates, one shared
 * scale, no post and no comb. Checked against a direct reference, and pinned
 * to be DIFFERENT from a plain sum of the streams -- a plain sum is the
 * natural-looking shortcut, and it is what this replaced. */
static void test_hc_collapse(void){
    enum { HC=4, DIM=5, N=HC*DIM };
    float x[N], fn[HC*N], base[HC], y[DIM];
    float scale=1.7f, eps=1e-6f, neps=1e-6f;
    for(int i=0;i<N;i++) x[i]=rndf()*2.0f;
    for(int i=0;i<HC*N;i++) fn[i]=rndf()*0.4f;
    for(int j=0;j<HC;j++) base[j]=rndf();
    dsv4_hc_collapse(x,HC,DIM,fn,scale,base,eps,neps,y);

    double ss=0; for(int i=0;i<N;i++) ss+=(double)x[i]*x[i];
    float rsq=1.0f/sqrtf((float)(ss/(double)N)+neps);
    float pre[HC];
    for(int j=0;j<HC;j++){
        double a=0; for(int i=0;i<N;i++) a+=(double)fn[j*N+i]*x[i];
        pre[j]=1.0f/(1.0f+expf(-((float)a*rsq*scale+base[j])))+eps;
        CHECK(pre[j]>eps && pre[j]<1.0f+eps);      /* a gate, not a free weight */
    }
    for(int d=0;d<DIM;d++){
        float ref=0; for(int j=0;j<HC;j++) ref+=pre[j]*x[j*DIM+d];
        CLOSE(y[d],ref,1e-5f);
    }
    int differs=0;
    for(int d=0;d<DIM;d++){
        float plain=0; for(int j=0;j<HC;j++) plain+=x[j*DIM+d];
        if(fabsf(plain-y[d])>1e-3f) differs=1;
    }
    CHECK(differs);
}

/* ---------------- routing ---------------- */

static void test_sqrtsoftplus(void){
    /* softplus(0)=ln2 -> sqrt(ln2) */
    CLOSE(dsv4_score_sqrtsoftplus(0.0f), sqrtf(logf(2.0f)), 1e-6f);
    /* large positive saturates to sqrt(z) */
    CLOSE(dsv4_score_sqrtsoftplus(100.0f), 10.0f, 1e-4f);
    /* large negative -> softplus ~ exp(z) -> sqrt(exp(z)), and must stay finite
     * and non-negative rather than producing NaN from sqrt of a negative. */
    float s=dsv4_score_sqrtsoftplus(-40.0f);
    CHECK(s>=0.0f && s<1e-6f && !isnan(s));
    /* strictly increasing */
    float prev=-1;
    for(float z=-10;z<=10;z+=0.5f){ float v=dsv4_score_sqrtsoftplus(z); CHECK(v>prev); prev=v; }
}

/* The noaux_tc split: bias steers SELECTION, never the returned weights. */
static void test_route_bias_selection_only(void){
    enum { E=8, K=3 };
    float logits[E], bias[E]={0,0,0,0,0,0,0,0};
    int idx[K], idx2[K]; float wt[K], wt2[K];
    for(int e=0;e<E;e++) logits[e]=(float)e*0.25f;   /* expert 7 strongest */
    int n=dsv4_route(logits,bias,E,K,1,1.5f,idx,wt);
    CHECK(n==K);
    CHECK(idx[0]==7 && idx[1]==6 && idx[2]==5);
    /* Now bias expert 0 up hard: selection must change, and the weight the
     * newly-selected expert receives must be its UNBIASED score share. */
    bias[0]=100.0f;
    int n2=dsv4_route(logits,bias,E,K,1,1.5f,idx2,wt2);
    CHECK(n2==K && idx2[0]==0);
    float s0=dsv4_score_sqrtsoftplus(logits[0]);
    float sum=0; for(int i=0;i<K;i++) sum+=dsv4_score_sqrtsoftplus(logits[idx2[i]]);
    CLOSE(wt2[0], s0/sum*1.5f, 1e-5f);
    CHECK(wt2[0] < 1.5f);   /* a 100.0 bias must not leak into the weight */
    /* renormalized weights sum to route_scale */
    float t=0; for(int i=0;i<K;i++) t+=wt[i];
    CLOSE(t,1.5f,1e-5f);
    float t2=0; for(int i=0;i<K;i++) t2+=wt2[i];
    CLOSE(t2,1.5f,1e-5f);
    /* no duplicate selections */
    CHECK(idx[0]!=idx[1] && idx[1]!=idx[2] && idx[0]!=idx[2]);
}

/* Hash layers: indices come from the table, weights from the router scores. */
static void test_route_hash(void){
    enum { E=16, K=4 };
    float logits[E]; int idx[K]; float wt[K];
    for(int e=0;e<E;e++) logits[e]=rndf()*2.0f;
    int32_t row[K]={11,2,15,7};
    int n=dsv4_route_hash(logits,row,K,1,1.5f,idx,wt);
    CHECK(n==K);
    for(int i=0;i<K;i++) CHECK(idx[i]==(int)row[i]);   /* table wins, not the scores */
    float sum=0; for(int i=0;i<K;i++) sum+=dsv4_score_sqrtsoftplus(logits[row[i]]);
    for(int i=0;i<K;i++) CLOSE(wt[i], dsv4_score_sqrtsoftplus(logits[row[i]])/sum*1.5f, 1e-5f);
    float t=0; for(int i=0;i<K;i++) t+=wt[i];
    CLOSE(t,1.5f,1e-5f);
}

/* ---------------- swiglu ---------------- */

static void test_swiglu_asymmetric_clamp(void){
    float limit=10.0f;
    float gate[4]={ 50.0f, -50.0f, 1.0f, -1.0f };
    float up[4]  ={ 50.0f, -50.0f, 2.0f, -2.0f };
    float out[4];
    dsv4_swiglu(out,gate,up,4,limit);
    /* up clamped BOTH ways */
    CLOSE(out[0], (10.0f/(1.0f+expf(-10.0f)))*10.0f, 1e-3f);
    /* gate NOT clamped from below: -50 stays -50, silu(-50) ~ 0 */
    float g1=-50.0f, u1=-10.0f;
    CLOSE(out[1], (g1/(1.0f+expf(-g1)))*u1, 1e-3f);
    CHECK(fabsf(out[1])<1e-3f);
    /* A symmetric clamp would have produced silu(-10)*-10 -- materially
     * different. Pin that the two disagree so the test has teeth. */
    float gsym=-10.0f;
    CHECK(fabsf((gsym/(1.0f+expf(-gsym)))*u1 - out[1]) > 1e-3f);
    /* below the limit, plain silu*up */
    CLOSE(out[2], (1.0f/(1.0f+expf(-1.0f)))*2.0f, 1e-6f);
    /* limit<=0 disables clamping entirely */
    float raw[1]={50.0f}, rup[1]={3.0f}, rout[1];
    dsv4_swiglu(rout,raw,rup,1,0.0f);
    CLOSE(rout[0], 50.0f*3.0f, 1e-2f);
}

/* ---------------- RoPE / YaRN ---------------- */

static void test_rope_freqs_and_yarn(void){
    enum { RD=64 };
    float f[RD/2], y[RD/2];
    dsv4_rope_freqs(f,RD,10000.0f,0,16.0f,32.0f,1.0f);      /* YaRN off */
    for(int j=0;j<RD/2;j++) CLOSE(f[j], 1.0f/powf(10000.0f,(float)(2*j)/(float)RD), 1e-7f);
    CLOSE(f[0],1.0f,1e-7f);
    /* YaRN on: low-index (high-frequency) dims must be untouched, high-index
     * (low-frequency) dims divided by `factor`. Between them, a ramp. */
    dsv4_rope_freqs(y,RD,160000.0f,65536,16.0f,32.0f,1.0f);
    float base[RD/2];
    dsv4_rope_freqs(base,RD,160000.0f,0,16.0f,32.0f,1.0f);
    CLOSE(y[0],base[0],1e-7f);                              /* untouched */
    CLOSE(y[RD/2-1], base[RD/2-1]/16.0f, 1e-9f);            /* fully scaled */
    int monotone=1;
    for(int j=1;j<RD/2;j++){
        float rj=y[j]/base[j], rp=y[j-1]/base[j-1];
        if(rj>rp+1e-6f) monotone=0;                          /* ratio never rises */
    }
    CHECK(monotone);
    for(int j=0;j<RD/2;j++) CHECK(y[j]<=base[j]+1e-9f && y[j]>=base[j]/16.0f-1e-9f);
}

static void test_rope_apply_inverse(void){
    enum { RD=8 };
    float f[RD/2]; dsv4_rope_freqs(f,RD,10000.0f,0,16.0f,32.0f,1.0f);
    float v[RD], orig[RD];
    for(int i=0;i<RD;i++) v[i]=orig[i]=rndf();
    dsv4_rope_apply(v,RD,37,f,0);
    int moved=0; for(int i=0;i<RD;i++) if(fabsf(v[i]-orig[i])>1e-4f) moved=1;
    CHECK(moved);
    /* Norm is preserved pairwise (it is a rotation). */
    for(int j=0;j<RD/2;j++){
        float n0=orig[2*j]*orig[2*j]+orig[2*j+1]*orig[2*j+1];
        float n1=v[2*j]*v[2*j]+v[2*j+1]*v[2*j+1];
        CLOSE(n0,n1,1e-5f);
    }
    /* De-rotation is exact -- this is what the attention output relies on. */
    dsv4_rope_apply(v,RD,37,f,1);
    for(int i=0;i<RD;i++) CLOSE(v[i],orig[i],1e-5f);
    /* position 0 is the identity */
    float w[RD]; memcpy(w,orig,sizeof w);
    dsv4_rope_apply(w,RD,0,f,0);
    for(int i=0;i<RD;i++) CLOSE(w[i],orig[i],1e-7f);
}

/* ---------------- compressed-KV pooling ---------------- */

/* The softmax is PER DIMENSION over the group. Constructed so a per-token
 * (scalar-weight) implementation gives a visibly different answer: dimension 0
 * favours slot 0, dimension 1 favours slot 1. */
static void test_compress_pool_per_dimension(void){
    enum { R=4, D=2 };
    float kv[R*D], score[R*D], out[D];
    for(int r=0;r<R;r++){ kv[r*D+0]=(float)r; kv[r*D+1]=(float)(10*r); }
    for(int r=0;r<R;r++){ score[r*D+0]=0.0f; score[r*D+1]=0.0f; }
    score[0*D+0]=20.0f;      /* dim 0 -> slot 0 */
    score[1*D+1]=20.0f;      /* dim 1 -> slot 1 */
    dsv4_compress_pool(kv,score,NULL,R,D,out);
    CLOSE(out[0],0.0f,1e-3f);        /* kv[0][0] == 0 */
    CLOSE(out[1],10.0f,1e-3f);       /* kv[1][1] == 10 */
    /* Uniform gates -> plain mean, per dimension. */
    for(int i=0;i<R*D;i++) score[i]=1.234f;
    dsv4_compress_pool(kv,score,NULL,R,D,out);
    CLOSE(out[0],(0+1+2+3)/4.0f,1e-5f);
    CLOSE(out[1],(0+10+20+30)/4.0f,1e-4f);
    /* ape is added BEFORE the softmax: a large ape on one slot must dominate. */
    float ape[R*D]; memset(ape,0,sizeof ape);
    ape[3*D+0]=30.0f;
    dsv4_compress_pool(kv,score,ape,R,D,out);
    CLOSE(out[0],3.0f,1e-3f);
    CLOSE(out[1],(0+10+20+30)/4.0f,1e-4f);   /* dim 1 unaffected */
}

/* ---------------- sparse attention ---------------- */

/* Online softmax must equal a direct (offline) softmax over the same gathered
 * set. The reference here is deliberately the naive two-pass formulation. */
static void test_sparse_attn_vs_offline(void){
    enum { H=3, D=5, NPOS=12, NIDX=7 };
    float q[H*D], kv[NPOS*D], sink[H], out[H*D];
    int32_t idx[NIDX]={9,0,4,-1,7,2,11};      /* includes a hole */
    for(int i=0;i<H*D;i++) q[i]=rndf();
    for(int i=0;i<NPOS*D;i++) kv[i]=rndf();
    for(int h=0;h<H;h++) sink[h]=rndf();
    float scale=1.0f/sqrtf((float)D);
    dsv4_sparse_attn(q,kv,idx,sink,H,D,NPOS,NIDX,scale,out);

    for(int h=0;h<H;h++){
        float s[NIDX]; int n=0; int pos[NIDX];
        for(int t=0;t<NIDX;t++){
            if(idx[t]<0) continue;
            float dot=0; for(int d=0;d<D;d++) dot+=q[h*D+d]*kv[idx[t]*D+d];
            s[n]=dot*scale; pos[n]=idx[t]; n++;
        }
        float mx=s[0]; for(int i=1;i<n;i++) if(s[i]>mx) mx=s[i];
        float den=0; for(int i=0;i<n;i++) den+=expf(s[i]-mx);
        den+=expf(sink[h]-mx);                  /* the sink, against the max */
        for(int d=0;d<D;d++){
            float num=0;
            for(int i=0;i<n;i++) num+=expf(s[i]-mx)*kv[pos[i]*D+d];
            CLOSE(out[h*D+d], num/den, 2e-5f);
        }
    }
}

/* The sink is a real term: raising it must shrink every output component,
 * because it adds mass to the denominator and nothing to the numerator. */
static void test_sparse_attn_sink_effect(void){
    enum { H=1, D=4, NPOS=6, NIDX=3 };
    float q[H*D], kv[NPOS*D], out_lo[H*D], out_hi[H*D];
    int32_t idx[NIDX]={0,3,5};
    for(int i=0;i<H*D;i++) q[i]=rndf();
    for(int i=0;i<NPOS*D;i++) kv[i]=rndf()+1.0f;   /* keep signs stable */
    float scale=1.0f/sqrtf((float)D);
    float lo[H]={-30.0f}, hi[H]={30.0f};
    dsv4_sparse_attn(q,kv,idx,lo,H,D,NPOS,NIDX,scale,out_lo);
    dsv4_sparse_attn(q,kv,idx,hi,H,D,NPOS,NIDX,scale,out_hi);
    for(int d=0;d<D;d++) CHECK(fabsf(out_hi[d]) < fabsf(out_lo[d]));
    /* A very negative sink is effectively absent: the result must match a
     * plain softmax over the gathered set. */
    float num[D]; memset(num,0,sizeof num); float den=0;
    float s[NIDX]; float mx=-INFINITY;
    for(int t=0;t<NIDX;t++){
        float dot=0; for(int d=0;d<D;d++) dot+=q[d]*kv[idx[t]*D+d];
        s[t]=dot*scale; if(s[t]>mx) mx=s[t];
    }
    for(int t=0;t<NIDX;t++){ float e=expf(s[t]-mx); den+=e;
        for(int d=0;d<D;d++) num[d]+=e*kv[idx[t]*D+d]; }
    for(int d=0;d<D;d++) CLOSE(out_lo[d], num[d]/den, 1e-4f);
}

/* All-masked index set must not divide by zero or emit NaN. */
static void test_sparse_attn_all_masked(void){
    enum { H=2, D=3, NPOS=4, NIDX=3 };
    float q[H*D], kv[NPOS*D], sink[H]={0.5f,-0.5f}, out[H*D];
    int32_t idx[NIDX]={-1,-1,-1};
    for(int i=0;i<H*D;i++) q[i]=rndf();
    for(int i=0;i<NPOS*D;i++) kv[i]=rndf();
    dsv4_sparse_attn(q,kv,idx,sink,H,D,NPOS,NIDX,0.5f,out);
    for(int i=0;i<H*D;i++){ CHECK(!isnan(out[i])); CLOSE(out[i],0.0f,0.0f); }
    /* Out-of-range indices are treated as holes, not as reads past the end. */
    int32_t bad[NIDX]={NPOS,NPOS+100,-5};
    dsv4_sparse_attn(q,kv,bad,sink,H,D,NPOS,NIDX,0.5f,out);
    for(int i=0;i<H*D;i++) CHECK(!isnan(out[i]));
}

/* ---------------- indexer scoring ---------------- */

/* ReLU sits INSIDE the head sum, before the per-head weighting. A position
 * whose per-head dots are large but negative must score 0, not a large
 * negative -- that is what decides which positions survive the top-k. */
static void test_index_score_relu_inside(void){
    enum { H=3, D=4, NCTX=5 };
    float q[H*D], ckv[NCTX*D], w[H], out[NCTX];
    for(int i=0;i<H*D;i++) q[i]=rndf();
    for(int i=0;i<NCTX*D;i++) ckv[i]=rndf();
    for(int h=0;h<H;h++) w[h]=0.3f+0.2f*(float)h;
    dsv4_index_score(q,ckv,w,H,D,NCTX,out);
    for(int t=0;t<NCTX;t++){
        float ref=0;
        for(int h=0;h<H;h++){
            float dot=0; for(int d=0;d<D;d++) dot+=q[h*D+d]*ckv[t*D+d];
            ref += (dot>0?dot:0)*w[h];
        }
        CLOSE(out[t],ref,1e-5f);
        CHECK(out[t]>=0.0f);          /* non-negative weights -> non-negative score */
    }
    /* Flip the query: every dot flips sign, so every score must collapse to 0.
     * A ReLU applied AFTER the weighted sum would not do this. */
    float qn[H*D]; for(int i=0;i<H*D;i++) qn[i]=-q[i];
    float outn[NCTX];
    dsv4_index_score(qn,ckv,w,H,D,NCTX,outn);
    int any_positive_before=0;
    for(int t=0;t<NCTX;t++) if(out[t]>1e-4f) any_positive_before=1;
    CHECK(any_positive_before);
    for(int t=0;t<NCTX;t++){
        float ref=0;
        for(int h=0;h<H;h++){
            float dot=0; for(int d=0;d<D;d++) dot+=qn[h*D+d]*ckv[t*D+d];
            ref += (dot>0?dot:0)*w[h];
        }
        CLOSE(outn[t],ref,1e-5f);
    }
}

/* ---------------- MXFP4 routed-expert matmul (quant.h matmul_mxfp4) ------- */
/* On ARM the kernel takes the NEON path for I%32==0 (I=dim / I=moe_inter_dim
 * for every DSV4 expert); on x86 the AVX2 path. Whatever the SIMD branch, this
 * pins the byte layout and the e2m1 * 2^(s-127) semantics against an
 * INDEPENDENT scalar reference written here with the UNDOUBLED e2m1 LUT -- so a
 * sign-bit flip, a low/high nibble swap, or a doubled-value-without-half-scale
 * mistake in the SIMD decode each fail here. */
static void test_mxfp4_neon(void){
    enum { S=2, I=64, O=5 };
    static const float true_lut[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                     -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    uint8_t q4[O*I/2], e8[O*I/32]; float x[S*I], y[O*S], ref[O*S];
    for(int i=0;i<O*I/2;i++) q4[i]=(uint8_t)(rng&0xFF);
    for(int i=0;i<O*I/32;i++) e8[i]=(uint8_t)(100+rng%28);        /* exact exponent */
    for(int i=0;i<S*I;i++) x[i]=rndf();
    matmul_mxfp4(y,x,q4,e8,S,I,O);
    for(int o=0;o<O;o++) for(int s=0;s<S;s++){
        float a=0; const float *xs=x+s*I;
        for(int g=0;g<I/32;g++){
            float sc; { union{uint32_t u;float f;}b; b.u=(uint32_t)e8[o*I/32+g]<<23; sc=b.f; }
            float ga=0;
            for(int c=0;c<32;c++){
                uint8_t byte=q4[o*I/2+(g*32+c)/2];
                uint8_t nib=(c&1)?(byte>>4):(byte&0xF);           /* low=even, high=odd */
                ga+=xs[g*32+c]*true_lut[nib];
            }
            a+=ga*sc;
        }
        ref[s*O+o]=a;
    }
    double num=0,den=0;
    for(int i=0;i<O*S;i++){ double d=(double)y[i]-ref[i]; num+=d*d; den+=ref[i]*ref[i]; }
    float rel=(float)(sqrt(num)/(sqrt(den)+1e-30));
    CHECK(rel<1e-5f);
    printf("  mxfp4: rel_l2 %.2e (S=%d I=%d O=%d, %s)\n",(double)rel,S,I,O,
#ifdef __ARM_NEON
           "NEON");
#elif defined(__AVX2__)
           "AVX2");
#else
           "scalar");
#endif
}

int main(void){
    test_sinkhorn_doubly_stochastic();
    test_sinkhorn_iters_matter();
    test_hc_split_gate_ranges();
    test_hc_pre_sum();
    test_hc_post_transpose();
    test_hc_collapse();
    test_sqrtsoftplus();
    test_route_bias_selection_only();
    test_route_hash();
    test_swiglu_asymmetric_clamp();
    test_rope_freqs_and_yarn();
    test_rope_apply_inverse();
    test_compress_pool_per_dimension();
    test_sparse_attn_vs_offline();
    test_sparse_attn_sink_effect();
    test_sparse_attn_all_masked();
    test_index_score_relu_inside();
    test_mxfp4_neon();
    if(fails){ printf("dsv4 primitive tests: %d FAILED\n",fails); return 1; }
    printf("dsv4 primitive tests: ok\n");
    return 0;
}
