// Standalone C-authority probe. See README.md; never built by Cargo.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int Nsite=4, Ne=1, LocSpn[4]={0};
static uint32_t tape[1024];
static int size, cursor;
static uint32_t gen_rand32(void) { assert(cursor<size); return tape[cursor++]; }
static double genrand_real2(void) { return gen_rand32()*(1.0/4294967296.0); }
#include "candidate_upstream.inc"
static void proposal(int spin,int occupied,int repeats,int destination,uint32_t accept,int valid) {
  tape[size++]=123; tape[size++]=spin ? UINT32_C(0x80000000) : 0;
  for(int i=0;i<repeats;i++) tape[size++]=occupied;
  tape[size++]=destination;
  if(valid) tape[size++]=accept;
}
static void emit(const char* name,int hops) {
  int ele[2]={0,0}, cfg[8], accepted=0, aw=0;
  for(int i=0;i<8;i++)cfg[i]=-1;
  cfg[0]=cfg[4]=0; cursor=0;
  printf("case %s %d\nwords",name,hops);
  for(int i=0;i<size;i++)printf(" %u",tape[i]);
  puts("");
  for(int hop=0;hop<hops;hop++) {
    int mi,ri,rj,spin,reject,accept=0;
    makeCandidate_hopping(&mi,&ri,&rj,&spin,&reject,ele,cfg);
    if(!reject) {
      // Analytic n=2 overlap: S[up,down+Nsite]=2^(up+down).
      const double logIpOld=log(ldexp(1.0,ele[0]+ele[1]));
      const double logIpNew=log(ldexp(1.0,ele[1-spin]+rj));
      const double x=0.0;
      double w=exp(2.0*(x+(logIpNew-logIpOld)));
      if(!isfinite(w))w=-1.0;
      if(w>genrand_real2()) {
        cfg[ri+spin*Nsite]=-1; cfg[rj+spin*Nsite]=mi;ele[mi+spin*Ne]=rj;
        accept=1;accepted++;aw++;
      }
    }
    // C's accepted window is reset on > Nsite, not >=.
    if(aw>Nsite)aw=0;
    printf("step %d %d %d %d %d %d %d %d %d %d %d\n",cursor,mi,spin,ri,rj,reject,accept,ele[0],ele[1],accepted,aw);
  }
  assert(cursor==size);
  puts("end");
}
int main(void) {
  size=0;proposal(0,0,0,1,0,1);proposal(1,0,0,1,UINT32_MAX,1);
  emit("complete_before_peers",2);
  for(int prefix=17;prefix<=20;prefix++) {
    size=0;
    proposal(0,0,prefix-4,1,0,1);
    proposal(1,0,0,1,UINT32_MAX,1);
    char name[32];snprintf(name,sizeof(name),"boundary_%d",prefix);
    emit(name,2);
  }
  size=0;
  proposal(0,0,4,1,0,1);
  proposal(1,0,0,1,UINT32_MAX,1);
  proposal(0,1,0,0,UINT32_MAX,1);
  proposal(0,1,17,2,0,0); // cap: final empty destination still rejected
  proposal(0,1,16,3,0,1); // final permitted destination
  proposal(1,1,15,0,0,1);
  proposal(0,3,0,2,UINT32_MAX,1);
  proposal(0,3,0,2,0,1);
  proposal(1,0,6,3,0,1);
  proposal(1,3,0,2,UINT32_MAX,1);
  proposal(0,2,0,0,0,1);
  proposal(1,3,0,1,0,1);
  emit("mixed_rebuild",12);
  return 0;
}
