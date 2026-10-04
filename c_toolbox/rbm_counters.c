/* Optional actual-C reader/counter probe. Cargo consumes checked-in data. */
#include <complex.h>
#ifndef CMPLX
/* glibc does not expose CMPLX with Clang's GNU compatibility version.
 * Construct the input without arithmetic so signed zero is preserved.
 * https://clang.llvm.org/docs/LanguageExtensions.html#initializer-lists-for-complex-numbers-in-c */
#define CMPLX(re, im) __builtin_complex(re, im)
#endif
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define D_FileNameMax 256
int Nsite, Nsite2, Nneuron, NneuronCharge, NneuronSpin, NneuronGeneral;
int NRBM_PhysLayerIdx, NRBM_HiddenLayerIdx;
int NChargeRBM_PhysLayerIdx, NSpinRBM_PhysLayerIdx, NGeneralRBM_PhysLayerIdx;
int NChargeRBM_HiddenLayerIdx, NSpinRBM_HiddenLayerIdx, NGeneralRBM_HiddenLayerIdx;
int NChargeRBM_PhysHiddenIdx, NSpinRBM_PhysHiddenIdx, NGeneralRBM_PhysHiddenIdx;
int *ChargeRBM_PhysLayerIdx, *SpinRBM_PhysLayerIdx, *GeneralRBM_PhysLayerIdx;
int *ChargeRBM_HiddenLayerIdx, *SpinRBM_HiddenLayerIdx, *GeneralRBM_HiddenLayerIdx;
int **ChargeRBM_PhysHiddenIdx, **SpinRBM_PhysHiddenIdx, **GeneralRBM_PhysHiddenIdx;
double complex *RBM;
#include "rbm_contracts_upstream.inc"
#include "rbm_counters_upstream.inc"

static void print_bits(const double complex *values, int count) {
  for (int i = 0; i < count; i++) {
    double re = creal(values[i]), im = cimag(values[i]);
    uint64_t rb, ib;
    memcpy(&rb, &re, sizeof(rb)); memcpy(&ib, &im, sizeof(ib));
    printf("%016" PRIx64 " %016" PRIx64 "%s", rb, ib, i+1 == count ? "\n" : " ");
  }
}

int main(int argc, char **argv) {
  if (argc != 12) return 2;
  Nsite = atoi(argv[1]); Nsite2 = 2*Nsite;
  NneuronCharge = NneuronSpin = NneuronGeneral = atoi(argv[2]);
  Nneuron = NneuronCharge+NneuronSpin+NneuronGeneral;
  if (Nsite < 1 || Nsite > 6 || NneuronCharge < 1 || NneuronCharge > 4) return 2;
  int *widths[] = {&NChargeRBM_PhysLayerIdx, &NSpinRBM_PhysLayerIdx, &NGeneralRBM_PhysLayerIdx,
    &NChargeRBM_HiddenLayerIdx, &NSpinRBM_HiddenLayerIdx, &NGeneralRBM_HiddenLayerIdx,
    &NChargeRBM_PhysHiddenIdx, &NSpinRBM_PhysHiddenIdx, &NGeneralRBM_PhysHiddenIdx};
  int layer[6][12] = {{0}}, matrix[3][12][4] = {{{0}}}, *rows[3][12];
  for (int s = 0; s < 3; s++) for (int r = 0; r < 12; r++) rows[s][r] = matrix[s][r];
  ChargeRBM_PhysLayerIdx=layer[0]; SpinRBM_PhysLayerIdx=layer[1]; GeneralRBM_PhysLayerIdx=layer[2];
  ChargeRBM_HiddenLayerIdx=layer[3]; SpinRBM_HiddenLayerIdx=layer[4]; GeneralRBM_HiddenLayerIdx=layer[5];
  ChargeRBM_PhysHiddenIdx=rows[0]; SpinRBM_PhysHiddenIdx=rows[1]; GeneralRBM_PhysHiddenIdx=rows[2];
  int flags[256], total = 0;
  for (int s = 0; s < 9; s++) {
    if (!strcmp(argv[s+3], "-")) continue;
    FILE *fp = fopen(argv[s+3], "r");
    int complex_flag, count = 0, status;
    char line[256];
    if (!fp || !ReadBuffIntCmpFlg(fp, widths[s], &complex_flag)) return 2;
    if (!fgets(line,sizeof(line),fp) || !fgets(line,sizeof(line),fp)) return 2;
    if (*widths[s] < 1 || *widths[s] > 97) return 2;
    if (s == 2) status=GetInfoGeneralRBM_Layer(fp, layer[s], flags, complex_flag, &count, 0, Nsite, *widths[s], "RBM");
    else if (s < 6) status=GetInfoRBM_Layer(fp, layer[s], flags, complex_flag, &count, 0, s < 3 ? Nsite : NneuronCharge, *widths[s], "RBM");
    else if (s == 8) status=GetInfoGeneralRBM_PhysHidden(fp, rows[s-6], flags, complex_flag, &count, 0, Nsite, NneuronCharge, *widths[s], "RBM");
    else status=GetInfoRBM_PhysHidden(fp, rows[s-6], flags, complex_flag, &count, 0, Nsite, NneuronCharge, *widths[s], "RBM");
    fclose(fp);
    if (status) return 2;
    total += *widths[s];
  }
  NRBM_PhysLayerIdx=NChargeRBM_PhysLayerIdx+NSpinRBM_PhysLayerIdx+NGeneralRBM_PhysLayerIdx;
  NRBM_HiddenLayerIdx=NChargeRBM_HiddenLayerIdx+NSpinRBM_HiddenLayerIdx+NGeneralRBM_HiddenLayerIdx;
  RBM=calloc((size_t)(total+1),sizeof(*RBM));
  for (int i = 0; i < total; i++) {
    uint64_t rb, ib; double re, im;
    if (scanf("%" SCNx64 " %" SCNx64, &rb, &ib) != 2) return 2;
    memcpy(&re,&rb,sizeof(re)); memcpy(&im,&ib,sizeof(im));
    RBM[i]=CMPLX(re,im);
  }
  int mask, ri, rj, spin, count=NRBM_PhysLayerIdx+Nneuron;
  double complex old[256], updated[256], inplace[256]; int occupation[12];
  while (scanf("%d %d %d %d", &mask, &ri, &rj, &spin) == 4) {
    for (int i = 0; i < Nsite2; i++) occupation[i]=(mask >> i)&1;
    MakeRBMCnt(old,occupation);
    UpdateRBMCnt(ri,rj,spin,updated,old,occupation);
    memcpy(inplace,old,(size_t)count*sizeof(*old));
    UpdateRBMCnt(ri,rj,spin,inplace,inplace,occupation);
    print_bits(old,count); print_bits(updated,count); print_bits(inplace,count);
  }
  free(RBM);
  return 0;
}
