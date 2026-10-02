#include <stdio.h>
#include <stdlib.h>
#define D_FileNameMax 256
#include "jastrow_contracts_upstream.inc"

int Nsite, NProj, NGutzwillerIdx = 2, NJastrowIdx;
int NDoublonHolon2siteIdx = 0, NDoublonHolon4siteIdx = 0;
int *GutzwillerIdx, **JastrowIdx, **DoublonHolon2siteIdx, **DoublonHolon4siteIdx;
#include "jastrow_projection_upstream.inc"

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  Nsite = atoi(argv[1]);
  FILE *fp = fopen(argv[2], "r");
  if (!fp || Nsite < 2 || Nsite > 8) return 2;
  int complex_flag = 0;
  if (!ReadBuffIntCmpFlg(fp, &NJastrowIdx, &complex_flag)) return 2;
  char line[256];
  if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) return 2;
  if (NJastrowIdx < 1 || NJastrowIdx > 32) return 2;
  int matrix[8][8], *idx[8], gutz[8], flags[256] = {0}, count = 0;
  for (int i = 0; i < 8; i++) {
    idx[i] = matrix[i]; gutz[i] = i%2;
    for (int j = 0; j < 8; j++) matrix[i][j] = -1;
  }
  if (GetInfoJastrow(fp, idx, flags, complex_flag, &count,
                     2, Nsite, NJastrowIdx, "Jastrow")) return 2;
  // Run kernels only for fully assigned, bounded off-diagonal tables.
  for (int i = 0; i < Nsite; i++)
    for (int j = 0; j < Nsite; j++)
      if (i != j && (matrix[i][j] < 0 || matrix[i][j] >= NJastrowIdx)) return 2;
  JastrowIdx = idx; GutzwillerIdx = gutz; NProj = 2 + NJastrowIdx;
  for (int pattern = 0; pattern < 8; pattern++) {
    int before[16], after[16], old[64], next[64];
    for (int spin = 0; spin < 2; spin++)
      for (int i = 0; i < Nsite; i++) {
        int value = pattern < 4 ? ((pattern >> spin)&1)
          : pattern == 4 ? (spin == 0 && i%2 == 0)
          : pattern == 5 ? (spin == 1 && i%2 == 0)
          : pattern == 6 ? (i%2 == spin)
          : (spin == 1 || i%2 == 0);
        before[spin*Nsite+i] = value;
      }
    MakeProjCnt(old, before);
    // One no-hop record plus every legal directed single-spin hop.
    for (int spin = -1; spin < 2; spin++)
      for (int ri = 0; ri < (spin < 0 ? 1 : Nsite); ri++)
        for (int rj = 0; rj < (spin < 0 ? 1 : Nsite); rj++) {
          if (spin >= 0 && (ri == rj || !before[spin*Nsite+ri] || before[spin*Nsite+rj])) continue;
          for (int k = 0; k < 2*Nsite; k++) after[k] = before[k];
          if (spin >= 0) { after[spin*Nsite+ri]--; after[spin*Nsite+rj]++; }
          if (spin < 0) for (int k = 0; k < NProj; k++) next[k] = old[k];
          else UpdateProjCnt(ri, rj, spin, next, old, after);
          printf("%d %d %d %d ", pattern, ri, rj, spin);
          for (int k = 0; k < 2*Nsite; k++) printf("%d ", before[k]);
          for (int k = 0; k < NProj; k++) printf("%d ", old[k]);
          for (int k = 0; k < NProj; k++) printf("%d ", next[k]);
          puts("");
        }
  }
  fclose(fp);
  return 0;
}
