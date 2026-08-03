/* Microbenchmark: matmul_fp8 -- current scalar kernel vs a hand-written NEON
 * decode+gather version (quant.h). NOT a unit test -- test_fp8_passthrough.c
 * proves correctness against a double-precision reference. This measures whether
 * writing NEON by hand beats what clang already does to the scalar loop under
 * -mcpu=native (the LUT gather is what usually stops auto-vectorization), and is
 * the gate for whether a NEON matmul_fp8 is worth wiring into quant.h at all.
 *
 * Methodology (honest):
 *   - baseline = a TRUE scalar reference (copy of the pre-NEON loop with clang's
 *     auto-vectorizer disabled), so the ratio is an upper bound on the win the way
 *     bench_mxfp4 does it -- a real pre-NEON -mcpu=native build may have
 *     auto-vectorized parts of the old loop;
 *   - "in-tree" = the real matmul_fp8, compiled as it runs (on ARM64 that means
 *     the NEON path: per-128-block, 4 independent float32x4 accumulators break
 *     the serial scalar acc+= chain, decode still through the exact E4M3_LUT so
 *     NaN propagation is unchanged);
 *   - fp8 weight bytes per call (= O*I) reported as GB/s so the number is
 *     comparable; the S=1 dense cases are the decode path that matters.
 *
 * Run:  make -C c tests/bench_fp8 ARCH=native && ./c/tests/bench_fp8
 *       (not in TEST_BINS -- not a gate)
 */
#include "../quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- TRUE scalar reference: the pre-NEON matmul_fp8 inner loop, verbatim
 * shape, with clang's auto-vectorizer disabled so it measures actual scalar
 * throughput rather than whatever -mcpu=native happens to vectorize. The kernel
 * under test is the in-tree matmul_fp8, which on ARM now takes the NEON path. */
__attribute__((noinline))
static void matmul_fp8_scalar(float *y, const float *x, const uint8_t *q8,
                              const float *bscale, int S, int I, int O){
    int64_t nblkI = fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w = q8 + (int64_t)o*I;
        int64_t blkO = o / FP8_BLOCK;
        const float *scl = bscale + blkO*nblkI;
        for(int s=0;s<S;s++){
            const float *xs = x + (int64_t)s*I;
            double a=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                float sc=scl[bi]; float acc=0;
                #pragma clang loop vectorize(disable)
                for(int i=base;i<base+blen;i++) acc += e4m3_decode(w[i])*xs[i];
                a += (double)acc*sc;
            }
            y[(int64_t)s*O+o]=(float)a;
        }
    }
}


static uint32_t rs=0x1234ABCDu;
static uint32_t xr(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }
static double now_ns(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec*1e9+(double)ts.tv_nsec;
}
static int cmp_dbl(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b;
    return x<y?-1:x>y?1:0; }
static double median(double *v,int n){ qsort(v,n,sizeof(double),cmp_dbl); return v[n/2]; }
typedef struct { int S,I,O; const char *name; } Case;

static double bench(void (*kern)(float*,const float*,const uint8_t*,const float*,int,int,int),
                    float *y,const float*x,const uint8_t*q,const float*sc,int S,int I,int O){
    for(int w=0;w<3;w++) kern(y,x,q,sc,S,I,O);
    enum { CAP=500000 };
    double *t=(double*)malloc(sizeof(double)*CAP);
    double wall=0; int n=0;
    while(n<CAP && wall<3.5e8){
        double a=now_ns(); kern(y,x,q,sc,S,I,O); double d=now_ns()-a;
        t[n++]=d; wall+=d;
    }
    double m=median(t,n); free(t); return m;
}

int main(void){
    static const Case cases[]={
        {1,4096,4096,"dense square  S=1  I=4096 O=4096"},
        {1,2048,4096,"dense proj    S=1  I=2048 O=4096"},
        {8,4096,4096,"dense prefill S=8  I=4096 O=4096"},
        {1, 257, 513,"odd tail      S=1  I=257  O=513"},
    };
    const int ncases=(int)(sizeof(cases)/sizeof(cases[0]));
    const char *path =
#if defined(__ARM_NEON)
        "NEON";
#else
        "scalar (no ARM SIMD for matmul_fp8)";
#endif
    printf("matmul_fp8 microbench -- dense FP8 kernel (in-tree path: %s)\n", path);
    for(int c=0;c<ncases;c++){
        int S=cases[c].S,I=cases[c].I,O=cases[c].O;
        int64_t nblk=fp8_nblk(O)*fp8_nblk(I);
        double bytes=(double)((int64_t)O*I);          /* fp8 weight bytes/call */
        uint8_t *q=(uint8_t*)malloc((size_t)O*I);
        float *sc=(float*)malloc((size_t)nblk*sizeof(float));
        float *x=(float*)malloc((size_t)S*I*sizeof(float));
        float *yb=(float*)malloc((size_t)S*O*sizeof(float));
        float *yn=(float*)malloc((size_t)S*O*sizeof(float));
        if(!q||!sc||!x||!yb||!yn){ fprintf(stderr,"OOM\n"); return 1; }
        for(int64_t i=0;i<O*I;i++) q[i]=(uint8_t)xr();   /* may include NaN codes 0x7F/FF -- good */
        for(int64_t i=0;i<nblk;i++) sc[i]=(float)((int)(xr()%2001)-1000)/500.f;
        for(int i=0;i<S*I;i++) x[i]=(float)((int)(xr()&0x3FFFF)-0x20000)/0x20000;

        double ts=bench(matmul_fp8_scalar, yb,x,q,sc,S,I,O);   /* true scalar, old path */
        double tn=bench(matmul_fp8,        yn,x,q,sc,S,I,O);   /* in-tree (NEON on ARM) */

        double num=0,den=0; int nanbad=0;
        for(int i=0;i<S*O;i++){
            int bn=isnan(yb[i]), nn=isnan(yn[i]);
            if(bn!=nn){ nanbad=1; continue; }
            if(!bn){ double d=(double)yb[i]-yn[i]; num+=d*d; den+=yb[i]*yb[i]; }
        }
        if(nanbad){ printf("  FAIL %s: NaN mismatch\n",cases[c].name); return 1; }
        double rel=sqrt(num)/(sqrt(den)+1e-30);
        if(rel>1e-5){ printf("  FAIL %s: rel_l2 %.2e > 1e-5\n",cases[c].name,rel); return 1; }
        printf("  %-34s %7.1f ns/call | %6.2f GB/s (scalar)  vs  %7.1f ns/call | %6.2f GB/s (in-tree)  ->  %.2fx\n",
               cases[c].name, ts, bytes/ts, tn, bytes/tn, ts/tn);
        free(q); free(sc); free(x); free(yb); free(yn);
    }
    printf("done\n");
    return 0;
}
