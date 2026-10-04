/* Developer-only passive family capture. Reader authority: native d73 readdef.c
 * GetInfoOpt:2101-2113; GetInfoOptOrbitalParalell:2084-2098. No int read for
 * unwritten flags. Unsupported DH/BF/OptTrans fail rather than guessed masks. */
#include "family.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
int native13_family_masks(const Native13Layout *layout,uint8_t *mask,int length) {
    if(!layout || !mask || layout->npara<=0 || layout->npara>INT_MAX/2 || length!=2*layout->npara ||
       layout->nproj<0 || layout->nrbm<0 || layout->nslater<0 || layout->nopt<0 ||
       layout->ap_count<0 || layout->parallel_count<0 ||
       (layout->flag_rbm!=0 && layout->flag_rbm!=1) ||
       (layout->orbital_general!=0 && layout->orbital_general!=1) ||
       (!layout->flag_rbm && layout->nrbm!=0) ||
       (layout->orbital_general && layout->parallel_count==0) ||
       (!layout->orbital_general && layout->parallel_count!=0) ||
       (int64_t)layout->nslater!=(int64_t)layout->ap_count+2*(int64_t)layout->parallel_count ||
       layout->nprojbf!=0 || layout->nopt!=0 || layout->nqpopt<0 ||
       (int64_t)layout->npara!=(int64_t)layout->nproj+(int64_t)layout->flag_rbm*layout->nrbm+layout->nslater ||
       layout->segment_count<=0 || !layout->segments) return -1;
    memset(mask,0,(size_t)length);
    int next=0;
    for(int i=0;i<layout->segment_count;i++) {
        const Native13Segment *s=&layout->segments[i];
        if(!s->name || s->start!=next || s->length<0 || s->length>layout->npara-next ||
           s->complex_flag<0 || (s->reader!=NATIVE13_GETINFOOPT && s->reader!=NATIVE13_PARALLEL) ||
           (s->reader==NATIVE13_PARALLEL && s->length%2)) return -2;
        for(int p=s->start;p<s->start+s->length;p++) {
            mask[2*p]=1;
            /* Parallel writes imaginary values even for ComplexType0. */
            mask[2*p+1]=(s->reader==NATIVE13_PARALLEL || s->complex_flag>0);
        }
        next+=s->length;
    }
    return next==layout->npara?0:-3;
}
int native13_family_checkpoint(const char *stage,int rank,int group,int seed,
    int all_complex,int captured,const double complex *para,const int *flags,
    const Native13Layout *layout) {
    if(rank!=0)return 0;
    const char *root=getenv("MVMC_NATIVE13_INIT_OBSERVER_ROOT");
    if(!root || !*root || !stage || !layout || captured<0 ||
       (captured!=0 && captured!=layout->npara) || (captured && (!para || !flags)) ||
       (layout->nqpopt && !layout->qpopt))return -1;
    uint8_t *mask=malloc((size_t)2*layout->npara);
    if(!mask)return -2;
    if(native13_family_masks(layout,mask,2*layout->npara)){free(mask);return -3;}
    uint32_t raw[624],next[624],again[624],peek[624];
    int cursor,cursor2;uint64_t count,count2;
    if(native13_snapshot(raw,&cursor,&count,next) || native13_snapshot(again,&cursor2,&count2,peek) ||
       memcmp(raw,again,sizeof(raw)) || memcmp(next,peek,sizeof(next)) || cursor!=cursor2 || count!=count2) {free(mask);return -4;}
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s.txt",root,stage);
    if(n<0 || (size_t)n>=sizeof(path)){free(mask);return -5;}
    FILE *out=fopen(path,"wx");if(!out){free(mask);return -6;}
    int bad=fprintf(out,"rank=%d group=%d seed=%d AllComplexFlag=%d CapturedNPara=%d cursor=%d observed_gen_rand32=%llu\n",
        rank,group,seed,all_complex,captured,cursor,(unsigned long long)count)<0;
    bad|=fprintf(out,"NPara=%d NProj=%d NRBM=%d FlagRBM=%d NSlater=%d NOptTrans=%d NQPOptTrans=%d NProjBF=%d APCount=%d ParallelHeaderCount=%d OrbitalGeneral=%d\n",
        layout->npara,layout->nproj,layout->nrbm,layout->flag_rbm,layout->nslater,layout->nopt,layout->nqpopt,layout->nprojbf,layout->ap_count,layout->parallel_count,layout->orbital_general)<0;
    for(int i=0;i<layout->segment_count;i++) {
        const Native13Segment *s=&layout->segments[i];
        bad|=fprintf(out,"segment %s start=%d count=%d ComplexFlag=%d reader=%d\n",s->name,s->start,s->length,s->complex_flag,s->reader)<0;
    }
    for(int i=0;i<layout->nqpopt;i++)bad|=fprintf(out,"ParaQPOptTrans %d %.17g\n",i,layout->qpopt[i])<0;
    for(int i=0;i<624;i++)bad|=fprintf(out,"%u%c",raw[i],i==623?'\n':' ')<0;
    for(int i=0;i<624;i++)bad|=fprintf(out,"%u%c",next[i],i==623?'\n':' ')<0;
    for(int i=0;i<captured;i++) {
        bad|=fprintf(out,"parameter %d %.17g %.17g\n",i,creal(para[i]),cimag(para[i]))<0;
        for(int axis=2*i;axis<2*i+2;axis++) {
            if(mask[axis])bad|=fprintf(out,"flag %d mask=1 value=%d\n",axis,flags[axis])<0;
            else bad|=fprintf(out,"flag %d mask=0 NOT_DEFINED\n",axis)<0;
        }
    }
    free(mask);bad|=ferror(out)!=0;bad|=fclose(out)!=0;return bad?-7:0;
}
