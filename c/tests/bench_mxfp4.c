/* Microbenchmark: matmul_mxfp4 NEON path vs a true-scalar reference (quant.h).
 * NOT a unit test -- test_dsv4.c::test_mxfp4_neon proves correctness against an
 * independent reference. This measures the compute ceiling of the routed-expert
 * MXFP4 kernel on this machine: how fast the NEON decode+fma loop consumes the
 * weight bytes, and how that compares to a scalar loop at the same geometry --
 * the before/after for the NEON matmul_mxfp4 work.
 *
 * Methodology (chosen to be honest, not to flatter the change):
 *   - I%32==0 shapes only, because that is every DSV4 expert (dim and
 *     moe_inter_dim are both divisible by 32); the NEON branch is the one the
 *     engine actually takes. w1/w3 and w2 are different reduction widths, so
 *     both directions are swept, plus a small prefill batch.
 *   - The scalar reference disables clang's auto-vectorizer on its inner loop so
 *     it is a TRUE scalar baseline. The real pre-NEON build was compiled with
 *     -mcpu=native too, so clang may have auto-vectorized pieces of the old loop;
 *     treat the reported ratio as an upper bound on the win, not the exact old
 *     number. Both kernels are OpenMP-parallel over O, same as the real kernel.
 *   - Warm cache, N repeats, MEDIAN reported (robust to scheduler noise), plus
 *     GB/s-of-weights (nibbles + scale bytes per call) so the number is
 *     comparable across machines and kernels.
 *
 * Run:  make -C c tests/bench_mxfp4 ARCH=native && ./c/tests/bench_mxfp4
 *       (not in TEST_BINS -- not a gate)
 */
#include "../quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- TRUE scalar reference: the pre-NEON matmul_mxfp4 inner loop, verbatim
 * shape, with clang's auto-vectorizer disabled so it measures actual scalar
 * throughput rather than whatever -mcpu=native happens to vectorize. ---- */
__attribute__((noinline))
static void matmul_mxfp4_scalar(float *y, const float *x, const uint8_t *q4,
                                const uint8_t *e8s, int S, int I, int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const uint8_t *scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0;g<ng;g++){
                int base=g*32, glen=32; if(base+glen>I) glen=I-base;
                float sc=mx4_scale(scl[g]), ga=0;
                #pragma clang loop vectorize(disable) interleave(disable)
                for(int i=base;i<base+glen;i+=2){
                    uint8_t byte=w[i>>1];
                    ga+=xs[i]*mx4_lut[byte&0xF];
                    if(i+1<base+glen) ga+=xs[i+1]*mx4_lut[byte>>4];
                }
                a+=ga*sc;
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

static uint32_t rs=0x9E3779B9u;
static uint32_t xr(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }

static double now_ns(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec*1e9+(double)ts.tv_nsec;
}
static int cmp_dbl(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b;
    return x<y?-1:x>y?1:0; }
static double median(double *v,int n){ qsort(v,n,sizeof(double),cmp_dbl); return v[n/2]; }

typedef struct { int S,I,O; const char *name; } Case;

/* Time-budget based: warm the cache, then run until ~0.35s of kernel time has
 * accumulated, return the MEDIAN ns/call. Budgeting (not a fixed repeat count)
 * makes the bench correct on any machine/size without overshooting wall time. */
static double bench(void (*kern)(float*,const float*,const uint8_t*,const uint8_t*,int,int,int),
                    float *y,const float*x,const uint8_t*q,const uint8_t*e,int S,int I,int O){
    for(int w=0;w<3;w++) kern(y,x,q,e,S,I,O);          /* warm caches */
    enum { CAP=500000 };
    double *t=(double*)malloc(sizeof(double)*CAP);
    double wall=0; int n=0;
    while(n<CAP && wall<3.5e8){
        double a=now_ns(); kern(y,x,q,e,S,I,O); double d=now_ns()-a;
        t[n++]=d; wall+=d;
    }
    double m=median(t,n); free(t); return m;
}

int main(void){
    static const Case cases[]={
        {1,4096,2048,"w1/w3 decode   I=dim=4096  O=2048"},
        {1,2048,4096,"w2    decode   I=moe=2048  O=4096"},
        {8,4096,2048,"w1/w3 prefill  I=4096 O=2048 S=8"},
        {1, 64,  32, "tiny fixture   I=64   O=32"},
    };
    const int ncases=(int)(sizeof(cases)/sizeof(cases[0]));
    const char *path =
#if defined(__ARM_NEON)
        "NEON";
#elif defined(__AVX2__)
        "AVX2";
#else
        "scalar (no SIMD)";
#endif
    printf("matmul_mxfp4 microbench -- kernel SIMD path: %s\n", path);

    double tot_neon=0,tot_scalar=0;
    for(int c=0;c<ncases;c++){
        int S=cases[c].S,I=cases[c].I,O=cases[c].O;
        int64_t rb=(int64_t)((I+1)/2)*O, ns=(int64_t)((I+31)/32)*O;
        double bytes=(double)(rb+ns);               /* weight bytes consumed per call */
        uint8_t *q=(uint8_t*)malloc(rb), *e=(uint8_t*)malloc(ns);
        float *x=(float*)malloc((size_t)S*I*sizeof(float));
        float *yn=(float*)malloc((size_t)S*O*sizeof(float));
        float *ys=(float*)malloc((size_t)S*O*sizeof(float));
        if(!q||!e||!x||!yn||!ys){ fprintf(stderr,"OOM\n"); return 1; }
        for(int64_t i=0;i<rb;i++) q[i]=(uint8_t)xr();
        for(int64_t i=0;i<ns;i++) e[i]=(uint8_t)(100+xr()%28);   /* exact exponents */
        for(int i=0;i<S*I;i++) x[i]=(float)((int)(xr()&0x3FFFF)-0x20000)/0x20000;

        double tn=bench(matmul_mxfp4,       yn,x,q,e,S,I,O);
        double ts=bench(matmul_mxfp4_scalar,ys,x,q,e,S,I,O);

        /* correctness gate: NEON must match the scalar reference */
        double num=0,den=0;
        for(int i=0;i<S*O;i++){ double d=(double)yn[i]-ys[i]; num+=d*d; den+=ys[i]*ys[i]; }
        double rel=sqrt(num)/(sqrt(den)+1e-30);
        if(rel>1e-5){ printf("  FAIL %s: rel_l2 %.2e > 1e-5\n",cases[c].name,rel); return 1; }

        printf("  %-34s %7.1f ns/call | %6.2f GB/s (NEW)  vs  %7.1f ns/call | %6.2f GB/s (scalar)  ->  %.1fx\n",
               cases[c].name, tn, bytes/tn, ts, bytes/ts, ts/tn);
        tot_neon+=tn; tot_scalar+=ts;
        free(q); free(e); free(x); free(yn); free(ys);
    }
    printf("  overall: scalar median %.1f ns  SIMD median %.1f ns  ->  %.2fx\n",
           tot_scalar,tot_neon,tot_scalar/tot_neon);
    printf("done\n");
    return 0;
}

