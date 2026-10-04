/* Optional first-operation diagnostic; original InitParameter included, not
 * rewritten. Build with the independently extracted parameter-stage include. */
#define main reviewed_parameter_acquisition_main
#include "reviewed_parameter_c_audit.c"
#undef main
static void scalar(const char *name,double value) {
    uint64_t bits; memcpy(&bits,&value,sizeof(bits));
    printf("%s %.18e %016llx\n",name,value,(unsigned long long)bits);
}
int main(void) {
    NProj=6; NRBM=86; NSlater=10; NPara=102; FlagRBM=1;
    AllComplexFlag=1; Nneuron=40; NOptTrans=0;
    for(int i=0;i<2*NPara;i++) OptFlag[i]=1;
    init_gen_rand(12395);
    double r1=genrand_real2(),r2=genrand_real2();
    double radius=1e-2*r1;
    double complex phase=2.0*I*M_PI*r2;
    double complex unit=cexp(phase);
    double complex decomposed=radius*unit;
    scalar("r1",r1); scalar("r2",r2); scalar("radius",radius);
    scalar("phase-real",creal(phase)); scalar("phase-imag",cimag(phase));
    scalar("unit-real",creal(unit)); scalar("unit-imag",cimag(unit));
    scalar("decomposed-real",creal(decomposed)); scalar("decomposed-imag",cimag(decomposed));
    init_gen_rand(12395); draw_count=0;
    InitParameter(); /* Original unchanged C expression, same first active slot. */
    scalar("actual-init-real",creal(RBM[0])); scalar("actual-init-imag",cimag(RBM[0]));
    printf("actual-init-primitive-count %lu\n",draw_count);
}
