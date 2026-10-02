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

int GetInfoOrbitalGeneral(FILE *fp, int **Array, int *ArrayOpt, int **ArraySgn, int *iOptCount,
                          int _fidx, int _iComplexFlag, int _iFlagOrbitalGeneral, int _APFlag, int Nsite, int NArray,
                          char *defname) {
  char ctmp2[256];
  int i, j;
  int idx0 = 0, idx1 = 0;
  int itmp = 0, info = 0;
  int spn_i, spn_j;
  int all_i, all_j;
  int fij = 0, fijSign = 1;
  int fidx = _fidx;

  if (NArray == 0) return 0;
  while (fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp) != NULL) {
    sscanf(ctmp2, "%d %d %d %d %d %d\n", &i, &spn_i, &j, &spn_j, &fij, &fijSign);
    all_i = i + spn_i * Nsite; //fsz
    all_j = j + spn_j * Nsite; //fsz
    if (CheckPairSite(i, j, Nsite) != 0) {
      fprintf(stderr, "Error: Site index is incorrect. \n");
      return -1;
    }
    if (all_i >= all_j) itmp = 1;
    idx0++;
    Array[all_i][all_j] = fij;
    ArraySgn[all_i][all_j] = fijSign;
    // Note F_{IJ}=-F_{JI}
    Array[all_j][all_i] = fij;
    ArraySgn[all_j][all_i] = -fijSign;
    if (idx0 == (2 * Nsite * Nsite - Nsite)) break; //2N*(2N-1)/2
  }

  if (_APFlag == 0) {
    for (i = 0; i < 2 * Nsite; i++) {
      for (j = i + 1; j < 2 * Nsite; j++) {
        ArraySgn[i][j] = 1;
        ArraySgn[j][i] = -1;
      }
    }
  }

  idx1 = GetInfoOpt(fp, ArrayOpt, _iComplexFlag, iOptCount, fidx);
  if (idx0 != (2 * Nsite * Nsite - Nsite) || idx1 != NArray || itmp == 1) {
    info = ReadDefFileError(defname);
  }

  return info;
}
