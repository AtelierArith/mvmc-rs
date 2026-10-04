/* Optional developer oracle. Actual avevar.c bodies, no Rust dependency. */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define D_FileNameMax 4096
int NSROptItrSmp, NPara, NGutzwillerIdx, NJastrowIdx;
int NDoublonHolon2siteIdx, NDoublonHolon4siteIdx;
int NChargeRBM_PhysLayerIdx, NSpinRBM_PhysLayerIdx, NGeneralRBM_PhysLayerIdx;
int NChargeRBM_HiddenLayerIdx, NSpinRBM_HiddenLayerIdx, NGeneralRBM_HiddenLayerIdx;
int NChargeRBM_PhysHiddenIdx, NSpinRBM_PhysHiddenIdx, NGeneralRBM_PhysHiddenIdx;
int NSlater, NOptTrans, iFlgOrbitalGeneral, iNOrbitalAntiParallel, iNOrbitalParallel;
char CParaFileHead[D_FileNameMax];
double complex Etot, Etot2, *Para, *SROptData;
#include "ctest_opt_window_upstream.inc"

int main(int argc, char **argv) {
    if ((argc != 3 && argc != 6) || strlen(argv[2]) + 64 >= sizeof(CParaFileHead)) return 2;
    if (argc == 6 &&
        (sscanf(argv[3], "%d", &iFlgOrbitalGeneral) != 1 ||
         sscanf(argv[4], "%d", &iNOrbitalAntiParallel) != 1 ||
         sscanf(argv[5], "%d", &iNOrbitalParallel) != 1)) return 10;
    FILE *input = fopen(argv[1], "r");
    if (!input) return 3;
    if (fscanf(input, "%d %d", &NSROptItrSmp, &NPara) != 2 ||
        NSROptItrSmp < 1 || NSROptItrSmp > 10000 || NPara < 0 || NPara > 10000) return 4;
    int *widths[] = {&NGutzwillerIdx, &NJastrowIdx,
        &NDoublonHolon2siteIdx, &NDoublonHolon4siteIdx,
        &NChargeRBM_PhysLayerIdx, &NSpinRBM_PhysLayerIdx, &NGeneralRBM_PhysLayerIdx,
        &NChargeRBM_HiddenLayerIdx, &NSpinRBM_HiddenLayerIdx, &NGeneralRBM_HiddenLayerIdx,
        &NChargeRBM_PhysHiddenIdx, &NSpinRBM_PhysHiddenIdx, &NGeneralRBM_PhysHiddenIdx,
        &NSlater, &NOptTrans};
    int total = 0;
    for (int i = 0; i < 15; ++i) {
        if (fscanf(input, "%d", widths[i]) != 1 || *widths[i] < 0) return 5;
        total += *widths[i] * (i == 2 ? 6 : i == 3 ? 10 : 1);
    }
    if (total != NPara) return 6;
    Para = calloc((size_t)NPara + 1, sizeof(*Para));
    SROptData = calloc((size_t)NSROptItrSmp * (NPara + 2), sizeof(*SROptData));
    if (!Para || !SROptData) return 7;
    for (int sample = 0; sample < NSROptItrSmp; ++sample) {
        for (int i = 0; i < NPara + 2; ++i) {
            double re, im;
            if (fscanf(input, "%lf %lf", &re, &im) != 2 || !isfinite(re) || !isfinite(im)) return 8;
            double complex value = re + I * im;
            if (i == 0) Etot = value;
            else if (i == 1) Etot2 = value;
            else Para[i - 2] = value;
        }
        StoreOptData(sample);
    }
    if (fscanf(input, " %*s") != EOF) return 9;
    fclose(input);
    strcpy(CParaFileHead, argv[2]);
    /* FSZ changes auxiliary filenames only, not the contiguous main stream. */
    OutputOptData();
    free(SROptData);
    free(Para);
    return 0;
}
