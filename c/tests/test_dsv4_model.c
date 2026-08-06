/* dsv4_model.h tests — config parsing and the tensor manifest.
 *
 * THE LOAD-BEARING TEST is test_param_counts(): summing every tensor in the
 * derived manifest must reproduce DeepSeek's own published figures for this
 * checkpoint (284B total, 13B active per token). Those numbers were not used
 * to build the manifest -- they come from the model card -- so agreement is a
 * genuine external check on a derivation that was otherwise made by reading
 * `inference/model.py` without ever loading the weights. A single wrong shape
 * anywhere in the attention stack or the expert grid moves the total by more
 * than the tolerance.
 *
 * The config below is verbatim from the published config.json. */
#include "../dsv4_model.h"
#include <stdio.h>
#include <stdlib.h>

static int fails=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n",__FILE__,__LINE__,#c); fails++; } }while(0)

/* compress_ratios below has 44 entries for num_hidden_layers=43 -- the extra
 * one belongs to the MTP head (num_nextn_predict_layers=1), which is why
 * dsv4_cfg_from_json accepts n_layers or n_layers+n_nextn.
 *
 * The INTERIOR of the array was long unverified here -- it reached this file
 * through a summarizing fetch that gave inconsistent answers about the tail --
 * so what is asserted below is structural rather than literal: the element
 * count, the leading zeros, the trailing zero, and that every value is one of
 * {0,4,128} (the only three dsv4_coff / dsv4_has_indexer distinguish). The
 * published -0731 config has since been read directly and DOES have this
 * interleave, with 41 compressor and 21 indexer layers to match. The
 * assertions stay structural anyway: the manifest reads the array from the
 * config at runtime, so nothing downstream depends on this literal. */
static const char *CFG_JSON =
"{\"hidden_size\":4096,\"num_hidden_layers\":43,\"num_attention_heads\":64,"
"\"head_dim\":512,\"qk_rope_head_dim\":64,\"q_lora_rank\":1024,\"o_lora_rank\":1024,"
"\"o_groups\":8,\"n_routed_experts\":256,\"num_experts_per_tok\":6,"
"\"moe_intermediate_size\":2048,\"n_shared_experts\":1,\"vocab_size\":129280,"
"\"num_hash_layers\":3,\"index_n_heads\":64,\"index_head_dim\":128,\"index_topk\":512,"
"\"hc_mult\":4,\"hc_sinkhorn_iters\":20,\"hc_eps\":1e-06,\"sliding_window\":128,"
"\"max_position_embeddings\":1048576,\"rms_norm_eps\":1e-06,"
"\"num_nextn_predict_layers\":1,"
"\"routed_scaling_factor\":1.5,\"swiglu_limit\":10.0,\"rope_theta\":10000,"
"\"compress_rope_theta\":160000,"
"\"rope_scaling\":{\"beta_fast\":32,\"beta_slow\":1,\"factor\":16,"
"\"original_max_position_embeddings\":65536,\"type\":\"yarn\"},"
"\"compress_ratios\":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,"
"4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0]}";

static jval *parse_cfg(char **arena){
    char *buf=strdup(CFG_JSON);
    jval *r=json_parse(buf,arena);
    /* json_parse keeps pointers into buf; leak it deliberately, the test is
     * short-lived and freeing it would dangle every string in the tree. */
    return r;
}

static DSV4Cfg CFG;

static void test_cfg_parse(void){
    char *arena=NULL; jval *r=parse_cfg(&arena);
    CHECK(r!=NULL);
    CHECK(dsv4_cfg_from_json(r,&CFG)==0);
    CHECK(CFG.dim==4096);
    CHECK(CFG.n_layers==43);
    CHECK(CFG.n_heads==64);
    CHECK(CFG.head_dim==512);
    CHECK(CFG.rope_head_dim==64);
    CHECK(CFG.q_lora_rank==1024 && CFG.o_lora_rank==1024 && CFG.o_groups==8);
    CHECK(CFG.n_routed_experts==256 && CFG.n_activated_experts==6);
    CHECK(CFG.moe_inter_dim==2048 && CFG.n_shared_experts==1);
    CHECK(CFG.vocab_size==129280 && CFG.n_hash_layers==3);
    CHECK(CFG.index_n_heads==64 && CFG.index_head_dim==128 && CFG.index_topk==512);
    CHECK(CFG.hc_mult==4 && CFG.hc_sinkhorn_iters==20);
    CHECK(CFG.window_size==128);
    /* 44 entries for 43 layers: the trailing one is the MTP head's. */
    CHECK(CFG.n_nextn==1);
    CHECK(CFG.n_compress_ratios==CFG.n_layers+CFG.n_nextn);
    CHECK(CFG.n_spec==1);
    /* Two RoPE bases, and YaRN parameters read from the nested object -- a
     * flat lookup would silently give the defaults here. */
    CHECK(CFG.rope_theta==10000.f);
    CHECK(CFG.compress_rope_theta==160000.f);
    CHECK(CFG.rope_factor==16.f);
    CHECK(CFG.beta_fast==32.f && CFG.beta_slow==1.f);
    CHECK(CFG.original_seq_len==65536);
    CHECK(CFG.route_scale==1.5f && CFG.swiglu_limit==10.0f);
    /* Structural properties only -- see the note above CFG_JSON for why the
     * interior interleave is deliberately not asserted. */
    CHECK(CFG.compress_ratios[0]==0 && CFG.compress_ratios[1]==0);
    CHECK(CFG.compress_ratios[CFG.n_compress_ratios-1]==0);   /* MTP: sliding window only */
    for(int i=0;i<CFG.n_compress_ratios;i++){
        int r2=CFG.compress_ratios[i];
        CHECK(r2==0||r2==4||r2==128);
    }
}

/* An unrecognized compression class must be refused, not silently routed down
 * the HCA path by dsv4_coff's `ratio==4 ? 2 : 1`. */
static void test_cfg_refuses_unknown_ratio(void){
    char *buf=strdup("{\"hidden_size\":4096,\"num_hidden_layers\":2,\"num_attention_heads\":64,"
        "\"head_dim\":512,\"qk_rope_head_dim\":64,\"q_lora_rank\":1024,\"o_lora_rank\":1024,"
        "\"o_groups\":8,\"n_routed_experts\":256,\"num_experts_per_tok\":6,"
        "\"moe_intermediate_size\":2048,\"n_shared_experts\":1,\"vocab_size\":129280,"
        "\"num_hash_layers\":0,\"index_n_heads\":64,\"index_head_dim\":128,\"index_topk\":512,"
        "\"hc_mult\":4,\"hc_sinkhorn_iters\":20,\"sliding_window\":128,"
        "\"compress_ratios\":[0,64]}");
    char *a=NULL; jval *r=json_parse(buf,&a);
    DSV4Cfg bad;
    CHECK(dsv4_cfg_from_json(r,&bad)==-1);
}

/* A config whose compress_ratios length disagrees with num_hidden_layers is a
 * different model; it must be refused, not padded. */
static void test_cfg_refuses_mismatch(void){
    char *buf=strdup("{\"hidden_size\":4096,\"num_hidden_layers\":43,\"num_attention_heads\":64,"
        "\"head_dim\":512,\"qk_rope_head_dim\":64,\"q_lora_rank\":1024,\"o_lora_rank\":1024,"
        "\"o_groups\":8,\"n_routed_experts\":256,\"num_experts_per_tok\":6,"
        "\"moe_intermediate_size\":2048,\"n_shared_experts\":1,\"vocab_size\":129280,"
        "\"num_hash_layers\":3,\"index_n_heads\":64,\"index_head_dim\":128,\"index_topk\":512,"
        "\"hc_mult\":4,\"hc_sinkhorn_iters\":20,\"sliding_window\":128,"
        "\"compress_ratios\":[0,0,4]}");
    char *arena=NULL; jval *r=json_parse(buf,&arena);
    DSV4Cfg bad;
    CHECK(dsv4_cfg_from_json(r,&bad)==-1);
    /* o_groups that does not divide n_heads*head_dim is also a refusal. */
    char *buf2=strdup("{\"hidden_size\":4096,\"num_hidden_layers\":1,\"num_attention_heads\":64,"
        "\"head_dim\":512,\"qk_rope_head_dim\":64,\"q_lora_rank\":1024,\"o_lora_rank\":1024,"
        "\"o_groups\":7,\"n_routed_experts\":256,\"num_experts_per_tok\":6,"
        "\"moe_intermediate_size\":2048,\"n_shared_experts\":1,\"vocab_size\":129280,"
        "\"num_hash_layers\":0,\"index_n_heads\":64,\"index_head_dim\":128,\"index_topk\":512,"
        "\"hc_mult\":4,\"hc_sinkhorn_iters\":20,\"sliding_window\":128,"
        "\"compress_ratios\":[0]}");
    char *a2=NULL; jval *r2=json_parse(buf2,&a2);
    DSV4Cfg bad2;
    CHECK(dsv4_cfg_from_json(r2,&bad2)==-1);
}

/* The -0731 (DSpark) checkpoint appends THREE speculative modules -- mtp.0/1/2,
 * one per dspark_target_layer_ids entry -- while leaving
 * num_nextn_predict_layers at the preview's 1. So 46 entries for 43 layers is
 * correct there, and the n_layers+n_nextn rule alone refuses a good checkpoint.
 * What must stay refused is the same 46 entries with nothing in the config that
 * accounts for them: the length is only ever unlocked by a key that states it.
 *
 * Everything below the head is the main stack, unchanged between the two
 * releases; only the tail differs. */
static const char *CFG46_HEAD =
"{\"hidden_size\":4096,\"num_hidden_layers\":43,\"num_attention_heads\":64,"
"\"head_dim\":512,\"qk_rope_head_dim\":64,\"q_lora_rank\":1024,\"o_lora_rank\":1024,"
"\"o_groups\":8,\"n_routed_experts\":256,\"num_experts_per_tok\":6,"
"\"moe_intermediate_size\":2048,\"n_shared_experts\":1,\"vocab_size\":129280,"
"\"num_hash_layers\":3,\"index_n_heads\":64,\"index_head_dim\":128,\"index_topk\":512,"
"\"hc_mult\":4,\"hc_sinkhorn_iters\":20,\"sliding_window\":128,"
"\"num_nextn_predict_layers\":1,"
"\"compress_ratios\":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,"
"4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0]";

static void test_cfg_accepts_dspark_head(void){
    char b[2048];
    snprintf(b,sizeof b,"%s,\"dspark_target_layer_ids\":[40,41,42],"
                        "\"dspark_block_size\":5,\"dspark_markov_rank\":256}",CFG46_HEAD);
    char *buf=strdup(b); char *a=NULL; jval *r=json_parse(buf,&a);
    DSV4Cfg c;
    CHECK(dsv4_cfg_from_json(r,&c)==0);
    CHECK(c.n_layers==43);
    CHECK(c.n_compress_ratios==46);
    CHECK(c.n_nextn==1);
    CHECK(c.n_spec==3);                 /* from dspark_target_layer_ids, not n_nextn */
    /* The point of the whole exercise: the extra entries must not shift the
     * main stack. Layer 42 is a ratio-4 (CSA) layer in both releases. */
    CHECK(c.compress_ratios[42]==4);

    /* Same 46 entries, no key that accounts for them -> still a refusal. */
    char b2[2048];
    snprintf(b2,sizeof b2,"%s}",CFG46_HEAD);
    char *buf2=strdup(b2); char *a2=NULL; jval *r2=json_parse(buf2,&a2);
    DSV4Cfg c2;
    CHECK(dsv4_cfg_from_json(r2,&c2)==-1);
}

/* The block-scale sidecar REPLACES the `.weight` suffix. Appending `.scale` to
 * the full name instead produces `<stem>.weight.scale`, which exists nowhere in
 * the real container: every quantized tensor then reads as "sidecar absent"
 * while the real sidecars pile up in the not-covered list. That is what a first
 * run against the published checkpoint looked like, so pin the convention. */
static void test_scale_names(void){
    char b[192];
    dsv4_scale_name(b,sizeof b,"layers.0.attn.wkv.weight");
    CHECK(!strcmp(b,"layers.0.attn.wkv.scale"));
    dsv4_scale_name(b,sizeof b,"layers.7.ffn.experts.100.w1.weight");
    CHECK(!strcmp(b,"layers.7.ffn.experts.100.w1.scale"));
    dsv4_scale_name(b,sizeof b,"layers.2.attn.indexer.wq_b.weight");
    CHECK(!strcmp(b,"layers.2.attn.indexer.wq_b.scale"));
    /* A name without the suffix gets `.scale` appended rather than losing
     * seven characters of itself. */
    dsv4_scale_name(b,sizeof b,"head");
    CHECK(!strcmp(b,"head.scale"));
    /* ".weight" alone is exactly the suffix length: must not underflow to "". */
    dsv4_scale_name(b,sizeof b,".weight");
    CHECK(!strcmp(b,".weight.scale"));
}

static DSV4Tensor *MAN; static int NMAN;

static void test_manifest_count_matches(void){
    NMAN=dsv4_manifest_count(&CFG);
    CHECK(NMAN>0);
    MAN=(DSV4Tensor*)malloc((size_t)NMAN*sizeof(DSV4Tensor));
    CHECK(MAN!=NULL);
    int got=dsv4_manifest(&CFG,MAN,NMAN);
    /* The closed-form count and the actual walk must agree exactly -- if they
     * drift, a caller sizing from the former overruns or truncates. */
    CHECK(got==NMAN);
    /* One-short buffer must report overflow rather than write past the end. */
    DSV4Tensor *small=(DSV4Tensor*)malloc((size_t)(NMAN-1)*sizeof(DSV4Tensor));
    CHECK(dsv4_manifest(&CFG,small,NMAN-1)==-1);
    free(small);
}

/* Layer-class structure: which layers own a compressor, and which own an
 * indexer on top of it. */
static void test_layer_classes(void){
    int with_compressor=0, with_indexer=0;
    for(int i=0;i<NMAN;i++){
        if(strstr(MAN[i].name,".attn.compressor.ape")) with_compressor++;
        if(strstr(MAN[i].name,".indexer.wq_b.weight")) with_indexer++;
    }
    int expect_c=0, expect_i=0;
    for(int l=0;l<CFG.n_layers;l++){
        if(CFG.compress_ratios[l]>0) expect_c++;
        if(CFG.compress_ratios[l]==4) expect_i++;
    }
    CHECK(with_compressor==expect_c);
    CHECK(with_indexer==expect_i);
    /* Bounds, not exact counts, so this test does not depend on the literal
     * above (see the CFG_JSON note): every indexer layer is a compressor layer,
     * both are a strict subset of the stack, and the two leading ratio-0 layers
     * plus at least one more mean the compressor count cannot reach n_layers.
     * On the real config these come out 41 and 21. */
    CHECK(expect_i>0 && expect_i<=expect_c);
    CHECK(expect_c>0 && expect_c<CFG.n_layers);

    /* Hash layers carry tid2eid and NO bias; score layers the reverse. The two
     * are mutually exclusive in the reference's Gate.__init__. */
    int tid=0, bias=0;
    for(int i=0;i<NMAN;i++){
        if(strstr(MAN[i].name,".gate.tid2eid")) tid++;
        if(strstr(MAN[i].name,".gate.bias"))    bias++;
    }
    CHECK(tid==CFG.n_hash_layers);
    CHECK(bias==CFG.n_layers-CFG.n_hash_layers);
    CHECK(tid+bias==CFG.n_layers);
}

/* The CSA/HCA shape difference comes from `coff` alone. Layer 2 (ratio 4)
 * must have a doubly-wide compressor; layer 3 (ratio 128) a single-width one. */
static void test_compressor_shapes(void){
    const DSV4Tensor *c4=NULL,*c128=NULL,*a4=NULL,*a128=NULL;
    for(int i=0;i<NMAN;i++){
        if(!strcmp(MAN[i].name,"layers.2.attn.compressor.wkv.weight")) c4=&MAN[i];
        if(!strcmp(MAN[i].name,"layers.3.attn.compressor.wkv.weight")) c128=&MAN[i];
        if(!strcmp(MAN[i].name,"layers.2.attn.compressor.ape"))        a4=&MAN[i];
        if(!strcmp(MAN[i].name,"layers.3.attn.compressor.ape"))        a128=&MAN[i];
    }
    CHECK(c4&&c128&&a4&&a128);
    if(c4&&c128){
        CHECK(c4->d0==2*CFG.head_dim && c4->d1==CFG.dim);      /* overlap -> coff 2 */
        CHECK(c128->d0==CFG.head_dim && c128->d1==CFG.dim);    /* no overlap -> coff 1 */
    }
    if(a4&&a128){
        CHECK(a4->d0==4   && a4->d1==2*CFG.head_dim);          /* [ratio, coff*hd] */
        CHECK(a128->d0==128 && a128->d1==CFG.head_dim);
    }
    /* The indexer's own compressor runs at index_head_dim, always ratio 4. */
    for(int i=0;i<NMAN;i++)
        if(!strcmp(MAN[i].name,"layers.2.attn.indexer.compressor.wkv.weight")){
            CHECK(MAN[i].d0==2*CFG.index_head_dim && MAN[i].d1==CFG.dim);
        }
}

/* Attention shapes, including the grouped output LoRA whose factorization is
 * easy to get backwards. */
static void test_attention_shapes(void){
    struct { const char *n; int64_t d0,d1; } want[] = {
        {"layers.5.attn.wq_a.weight", 1024, 4096},
        {"layers.5.attn.wq_b.weight", 64*512, 1024},
        {"layers.5.attn.wkv.weight",  512, 4096},          /* ONE kv vector, all heads */
        {"layers.5.attn.wo_a.weight", 8*1024, 64*512/8},   /* [groups*rank, hs/groups] */
        {"layers.5.attn.wo_b.weight", 4096, 8*1024},
        {"layers.5.attn.attn_sink",   64, 0},
        {"layers.5.attn.q_norm.weight", 1024, 0},
        {"layers.5.attn.kv_norm.weight", 512, 0},
        {"layers.5.hc_attn_fn", 24, 4*4096},
        {"layers.5.hc_attn_base", 24, 0},
        {"layers.5.hc_attn_scale", 3, 0},
    };
    for(size_t w=0;w<sizeof want/sizeof *want;w++){
        int found=0;
        for(int i=0;i<NMAN;i++) if(!strcmp(MAN[i].name,want[w].n)){
            found=1;
            if(MAN[i].d0!=want[w].d0 || MAN[i].d1!=want[w].d1)
                printf("FAIL shape %s: got [%lld,%lld] want [%lld,%lld]\n",want[w].n,
                       (long long)MAN[i].d0,(long long)MAN[i].d1,
                       (long long)want[w].d0,(long long)want[w].d1), fails++;
            break;
        }
        if(!found){ printf("FAIL missing tensor %s\n",want[w].n); fails++; }
    }
}

/* Expert grid: 256 routed (fp4) + 1 shared (not fp4) per layer, three matrices
 * each, with w2 transposed relative to w1/w3. */
static void test_expert_grid(void){
    int routed=0, shared=0;
    for(int i=0;i<NMAN;i++){
        if(strstr(MAN[i].name,".ffn.experts.")) routed++;
        if(strstr(MAN[i].name,".ffn.shared_experts.")) shared++;
    }
    CHECK(routed==3*CFG.n_routed_experts*CFG.n_layers);
    CHECK(shared==3*CFG.n_layers);
    for(int i=0;i<NMAN;i++){
        if(!strcmp(MAN[i].name,"layers.7.ffn.experts.100.w1.weight")){
            CHECK(MAN[i].d0==CFG.moe_inter_dim && MAN[i].d1==CFG.dim);
            CHECK(MAN[i].role==DSV4_T_FP4);
        }
        if(!strcmp(MAN[i].name,"layers.7.ffn.experts.100.w2.weight")){
            CHECK(MAN[i].d0==CFG.dim && MAN[i].d1==CFG.moe_inter_dim);
            CHECK(MAN[i].role==DSV4_T_FP4);
        }
        /* the shared expert is NOT fp4 in the reference -- it is fp8, which the
         * published checkpoint confirms: [2048,4096] at exactly O*I bytes with
         * a .scale sidecar next to it. */
        if(!strcmp(MAN[i].name,"layers.7.ffn.shared_experts.w1.weight"))
            CHECK(MAN[i].role==DSV4_T_FP8);
    }
}

/* ---- the external check ---- */
static void test_param_counts(void){
    int64_t total=0;
    for(int i=0;i<NMAN;i++) total+=dsv4_tensor_numel(&MAN[i]);
    double tb=(double)total/1e9;
    printf("  manifest: %d tensors, %.1fB parameters total\n",NMAN,tb);
    /* Published: 284B total. Tolerance covers the small tensors a manifest
     * derived from the reference may legitimately differ on (an unlisted MTP
     * head, a norm we have not accounted for) without letting a genuine shape
     * error through -- one wrong expert dimension moves this by tens of B. */
    CHECK(tb>280.0 && tb<288.0);

    int64_t act=dsv4_active_params(&CFG,MAN,NMAN);
    double ab=(double)act/1e9;
    printf("  active per token: %.1fB\n",ab);
    /* Published: 13B activated. */
    CHECK(ab>12.0 && ab<14.5);

    /* The experts must dominate: they are the reason this engine streams at
     * all. If they were not ~97% of parameters, the whole tiering premise
     * would not apply to this model. */
    int64_t exp_params=0;
    for(int i=0;i<NMAN;i++) if(MAN[i].role==DSV4_T_FP4) exp_params+=dsv4_tensor_numel(&MAN[i]);
    double share=(double)exp_params/(double)total;
    printf("  routed experts: %.1f%% of parameters\n",share*100.0);
    CHECK(share>0.95 && share<0.99);

    /* On-disk bytes: experts at 4.25 bits/weight, everything else at fp8 or
     * bf16, must land near the published 160 GB of safetensors. */
    int64_t bytes=0;
    for(int i=0;i<NMAN;i++) bytes+=dsv4_tensor_bytes(&MAN[i]);
    double gb=(double)bytes/1e9;
    printf("  on-disk: %.1f GB\n",gb);
    CHECK(gb>140.0 && gb<185.0);
}

/* Roles decide how bytes are read; a tensor landing in the wrong role is a
 * silent misread rather than a load failure, so pin the classification. */
static void test_roles(void){
    for(int i=0;i<NMAN;i++){
        const DSV4Tensor *t=&MAN[i];
        if(strstr(t->name,".ffn.experts.")) CHECK(t->role==DSV4_T_FP4);
        if(strstr(t->name,".gate.tid2eid")) CHECK(t->role==DSV4_T_I32);
        if(strstr(t->name,"_norm.weight")||strstr(t->name,".ape")
           ||strstr(t->name,"attn_sink")||strstr(t->name,"hc_"))
            CHECK(t->role==DSV4_T_PLAIN);
        /* The compressor projections are unquantized -- and ONLY those. The
         * same leaf names under .attn (wkv) are fp8, so match the prefix, not
         * the leaf: pinning this is what keeps a future edit from "restoring"
         * the symmetry the rest of the model has. They ship bf16 and are kept
         * bf16 in RAM (decoded on the fly in w_matmul), so PLAIN_BF16. */
        if(strstr(t->name,".compressor.wkv.weight")
           ||strstr(t->name,".compressor.wgate.weight"))
            CHECK(t->role==DSV4_T_PLAIN_BF16);
        if(strstr(t->name,".attn.wkv.weight")) CHECK(t->role==DSV4_T_FP8);
        /* The two biggest dense tensors are unquantized in the checkpoint;
         * head.weight ships bf16 and is the largest single dense tensor, so it
         * stays bf16 in RAM (PLAIN_BF16) instead of expanding to f32. */
        if(!strcmp(t->name,"head.weight"))  CHECK(t->role==DSV4_T_PLAIN_BF16);
        if(!strcmp(t->name,"embed.weight")) CHECK(t->role==DSV4_T_PLAIN);
        CHECK(t->d0>0);
        CHECK(t->d1>=0);
        CHECK(t->name[0]!=0);
    }
    /* No duplicate names anywhere in the manifest -- a duplicate would mean two
     * roles claim the same bytes. Checked on a sample grid rather than O(n^2)
     * over ~33k entries: the expert names are the only generated family. */
    for(int l=0;l<3;l++) for(int e=0;e<4;e++){
        char nm[160]; snprintf(nm,sizeof nm,"layers.%d.ffn.experts.%d.w1.weight",l,e);
        int hits=0;
        for(int i=0;i<NMAN;i++) if(!strcmp(MAN[i].name,nm)) hits++;
        CHECK(hits==1);
    }
}

int main(void){
    test_cfg_parse();
    if(fails){ printf("dsv4 model tests: %d FAILED (config)\n",fails); return 1; }
    test_cfg_refuses_mismatch();
    test_cfg_refuses_unknown_ratio();
    test_cfg_accepts_dspark_head();
    test_scale_names();
    test_manifest_count_matches();
    if(!MAN){ printf("dsv4 model tests: manifest alloc failed\n"); return 1; }
    test_layer_classes();
    test_compressor_shapes();
    test_attention_shapes();
    test_expert_grid();
    test_roles();
    test_param_counts();
    if(fails){ printf("dsv4 model tests: %d FAILED\n",fails); return 1; }
    printf("dsv4 model tests: ok\n");
    return 0;
}
