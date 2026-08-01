/* tok.h cross-check harness for the DeepSeek-V4 vocabulary.
 *
 *   tests/test_tok_dsv4 <tokenizer.json> <cases.txt>
 *
 * Driven by tools/dsv4_tokenizer.py --ctest, which writes the case file with
 * expected ids produced by HF `tokenizers` on the SAME vocabulary. Each line is
 *   <text with \n and \r escaped>\t<comma-separated ids>
 * Same contract as tools/k3_tokenizer.py --ctest and tests/test_tok_o200k.c;
 * unlike those, this one takes the vocabulary as an argument because the
 * DeepSeek-V4 checkpoint ships none and the file has to come from wherever the
 * user obtained it.
 *
 * Encoding equality is the gate. Round-tripping (decode(encode(x)) == x) is
 * checked too but reported separately: a byte-level BPE can legitimately
 * re-encode differently after a lossy normalizer, and conflating the two hides
 * which of the pair actually broke. */
#define _GNU_SOURCE
#include "../tok.h"

static int unescape(char *s){
    char *r=s, *w=s;
    while(*r){
        if(*r=='\\' && r[1]){
            r++;
            if(*r=='n') *w++='\n';
            else if(*r=='r') *w++='\r';
            else if(*r=='t') *w++='\t';
            else if(*r=='\\') *w++='\\';
            else *w++=*r;
            r++;
        } else *w++=*r++;
    }
    *w=0;
    return (int)(w-s);
}

int main(int argc, char **argv){
    if(argc<3){
        fprintf(stderr,"usage: %s <tokenizer.json> <cases.txt>\n",argv[0]);
        return 2;
    }
    Tok T;
    tok_load(&T,argv[1]);
    fprintf(stderr,"  tok.h family: %s\n",
            T.dsv4 ? "dsv4" : T.kimi ? "kimi" : T.o200k ? "o200k" : "cl100k");
    fprintf(stderr,"  rank-BPE mode: %s\n", T.rankbpe ? "yes (no merges list)" : "no");

    FILE *f=fopen(argv[2],"rb");
    if(!f){ perror(argv[2]); return 2; }

    char line[8192];
    int tot=0, pass=0, dpass=0;
    while(fgets(line,sizeof line,f)){
        size_t n=strlen(line);
        while(n>0 && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n]=0;
        if(n==0) continue;
        char *tab=strchr(line,'\t');
        if(!tab) continue;
        *tab=0;
        char *text=line; const char *idstr=tab+1;
        int tlen=unescape(text);

        int got[4096];
        int ng=tok_encode(&T,text,tlen,got,(int)(sizeof got/sizeof *got));

        int want[4096], nw=0;
        for(const char *p=idstr; *p && nw<4096; ){
            char *end; long v=strtol(p,&end,10);
            if(end==p) break;
            want[nw++]=(int)v;
            p=end; while(*p==',') p++;
        }

        tot++;
        int ok = (ng==nw);
        if(ok) for(int i=0;i<ng;i++) if(got[i]!=want[i]){ ok=0; break; }
        if(ok) pass++;
        else {
            fprintf(stderr,"  MISMATCH on %.60s\n    tok.h(%d):",text,ng);
            for(int i=0;i<ng && i<24;i++) fprintf(stderr," %d",got[i]);
            fprintf(stderr,"\n    ref  (%d):",nw);
            for(int i=0;i<nw && i<24;i++) fprintf(stderr," %d",want[i]);
            fprintf(stderr,"\n");
        }

        char back[8192];
        int nb=tok_decode(&T,got,ng,back,(int)sizeof back);
        if(nb==tlen && !memcmp(back,text,(size_t)tlen)) dpass++;
    }
    fclose(f);

    fprintf(stderr,"  encode: %d/%d cases match the reference\n",pass,tot);
    fprintf(stderr,"  decode: %d/%d cases round-trip\n",dpass,tot);
    if(tot==0){ fprintf(stderr,"  no cases parsed\n"); return 2; }
    return pass==tot ? 0 : 1;
}
