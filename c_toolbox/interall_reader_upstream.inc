/* Generated verbatim from authoritative mVMC-1.3.0: readdef.c sha256=6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9.
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

char *ReadBuffInt(FILE *fp, int *iNbuf) {
  char *cerr;
  char ctmp[D_FileNameMax];
  char ctmp2[D_FileNameMax];
  cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp);
  if (cerr != NULL) {
    cerr = fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp);
    sscanf(ctmp2, "%s %d\n", ctmp, iNbuf); //2
  }
  return cerr;
}
int ReadDefFileError(const char *defname) {
  fprintf(stderr, "error: %s (Broken file or Not exist)\n", defname);
  return 1;
}
int CheckSite(
    const int iSite,
    const int iMaxNum
) {
  if (iSite >= iMaxNum || iSite < 0) return (-1);
  return 0;
}
int CheckPairSite(
    const int iSite1,
    const int iSite2,
    const int iMaxNum
) {
  if (CheckSite(iSite1, iMaxNum) != 0) {
    return (-1);
  }
  if (CheckSite(iSite2, iMaxNum) != 0) {
    return (-1);
  }
  return 0;
}
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
int GetInfoInterAll(FILE *fp, int **ArrayIdx, double complex *ArrayValue,
                    int Nsite, int NArray, char *defname) {
  char ctmp2[256];
  int idx = 0, info = 0;
  int x0 = 0, x1 = 0, x2 = 0, x3 = 0;
  int x4 = 0, x5 = 0, x6 = 0, x7 = 0;
  double dReValue = 0, dImValue = 0;

  if (NArray == 0) return 0;
  while (fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp) != NULL) {
    sscanf(ctmp2, "%d %d %d %d %d %d %d %d %lf %lf\n",
           &x0, &x1, &x2, &x3, &x4, &x5, &x6, &x7,
           &dReValue, &dImValue);

    ArrayIdx[idx][0] = x0;
    ArrayIdx[idx][1] = x1;
    ArrayIdx[idx][2] = x2;
    ArrayIdx[idx][3] = x3;
    ArrayIdx[idx][4] = x4;
    ArrayIdx[idx][5] = x5;
    ArrayIdx[idx][6] = x6;
    ArrayIdx[idx][7] = x7;

    if (CheckQuadSite(x0, x2, x4, x6, Nsite) != 0) {
      fprintf(stderr, "Error: Site index is incorrect. \n");
      info = 1;
      break;
    }

    ArrayValue[idx] = dReValue + I * dImValue;

    if (TwoSz != -1 && !(x1 == x3 && x5 == x7)) {
      fprintf(stderr, "  Error:  Sz non-conserved system is not yet supported for InterAll.\n");
      info = ReadDefFileError(defname);
      break;
    }
    idx++;
  }
  if (idx != NArray) info = ReadDefFileError(defname);
  return info;
}
