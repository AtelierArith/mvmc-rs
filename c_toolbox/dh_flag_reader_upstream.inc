/* Generated verbatim from authoritative mVMC-1.4.0: readdef.c sha256=9e1f2c635bffb2acdf3210613ab7330c5b28967fae928b303f3b151dfa1b7ef3.
 * Regenerate with the corresponding scripts/check_*_c_parity.py --write.
 * This optional C oracle is not a Rust test/build dependency. */
/*
mVMC - A numerical solver package for a wide range of quantum lattice models based on many-variable Variational Monte Carlo method
Copyright (C) 2016 The University of Tokyo, All rights reserved.

This program is developed based on the mVMC-mini program
(https://github.com/fiber-miniapp/mVMC-mini)
which follows "The BSD 3-Clause License".

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details. 

You should have received a copy of the GNU General Public License 
along with this program. If not, see http://www.gnu.org/licenses/. 
*/
/*-------------------------------------------------------------
 * Variational Monte Carlo
 * Read Definition Files
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/

int CheckQuadSite(
    const int iSite1,
    const int iSite2,
    const int iSite3,
    const int iSite4,
    const int iMaxNum
) {
  if (CheckPairSite(iSite1, iSite2, iMaxNum) != 0) {
    return (-1);
  }
  if (CheckPairSite(iSite3, iSite4, iMaxNum) != 0) {
    return (-1);
  }
  return 0;
}
int
GetInfoDH2(FILE *fp, int **ArrayIdx, int *ArrayOpt, int iComplxFlag, int *iOptCount, int _fidx, int Nsite, int NArray,
           char *defname) {
  int idx0 = 0, idx1 = 0, info = 0;
  int i = 0, x0 = 0, x1 = 0, n = 0;
  int fidx = _fidx;
  if (NArray > 0) {
    idx0 = idx1 = 0;
    while (fscanf(fp, "%d %d %d %d\n", &i, &(x0), &(x1), &n) != EOF) {
      ArrayIdx[n][2 * i] = x0;
      ArrayIdx[n][2 * i + 1] = x1;
      if (CheckSite(i, Nsite) != 0 || CheckPairSite(x0, x1, Nsite) != 0) {
        fprintf(stderr, "Error: Site index is incorrect. \n");
        info = 1;
        break;
      }
      idx0++;
      if (idx0 == Nsite * NArray) break;
    }
    idx1 = GetInfoOpt(fp, ArrayOpt, iComplxFlag, iOptCount, fidx);
    if (idx0 != Nsite * NArray || idx1 != 2 * 3 * NArray) {
      info = ReadDefFileError(defname);
    }
  }
  return info;
}
int
GetInfoDH4(FILE *fp, int **ArrayIdx, int *ArrayOpt, int iComplxFlag, int *iOptCount, int _fidx, int Nsite, int NArray,
           char *defname) {
  int idx0 = 0, idx1 = 0, info = 0;
  int i = 0, x0 = 0, x1 = 0, x2 = 0, x3 = 0, n = 0;
  int fidx = _fidx;
  if (NArray > 0) {
    idx0 = idx1 = 0;
    while (fscanf(fp, "%d %d %d %d %d %d\n",
                  &i, &(x0), &(x1), &(x2), &(x3), &n) != EOF) {
      ArrayIdx[n][4 * i] = x0;
      ArrayIdx[n][4 * i + 1] = x1;
      ArrayIdx[n][4 * i + 2] = x2;
      ArrayIdx[n][4 * i + 3] = x3;
      if (CheckSite(i, Nsite) != 0 || CheckQuadSite(x0, x1, x2, x3, Nsite) != 0) {
        fprintf(stderr, "Error: Site index is incorrect. \n");
        info = 1;
        break;
      }
      idx0++;
      if (idx0 == Nsite * NArray) break;
    }
    idx1 = GetInfoOpt(fp, ArrayOpt, iComplxFlag, iOptCount, fidx);
    if (idx0 != Nsite * NArray || idx1 != 2 * 5 * NArray) {
      info = ReadDefFileError(defname);
    }
  }
  return info;
}
