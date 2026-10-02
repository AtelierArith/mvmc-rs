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
int GetInfoOpt(FILE *fp, int *ArrayOpt, int iComplxFlag, int *iTotalOptCount, int fidx) {
  int i;
  int iLocalOptCount = 0;
  while (fscanf(fp, "%d ", &i) != EOF) {
    fscanf(fp, "%d\n", &(ArrayOpt[2 * fidx])); // TBC real
    if(iComplxFlag>0){
      ArrayOpt[2 * fidx + 1] = ArrayOpt[2 * fidx]; //  TBC imaginary
    }
    fidx++;
    (iLocalOptCount)++;
    (*iTotalOptCount)++;
  }
  return (iLocalOptCount);
}
char *ReadBuffIntCmpFlg(FILE *fp, int *iNbuf, int *iComplexFlag) {
  char *cerr;
  char ctmp[D_FileNameMax];
  char ctmp2[D_FileNameMax];
  cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp);
  if (cerr != NULL) {
    cerr = fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp);
    sscanf(ctmp2, "%s %d\n", ctmp, iNbuf); //2
    if (*iNbuf == 0) {
      cerr = NULL;
      fprintf(stderr, "error: Number of components defined in the 2nd line is set as 0 or illegal header.\n");
      fprintf(stderr, "error: 2nd line %s", ctmp2);
    } else {
      cerr = fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp);
      sscanf(ctmp2, "%s %d\n", ctmp, iComplexFlag);
    }
  }
  return cerr;
}
int GetInfoRBM_Layer(FILE *fp, int *ArrayIdx, int *ArrayOpt, int iComplxFlag, int *iOptCount, int _fidx, int Nlayer, int NArray,
                      char *defname) {
  int idx0 = 0, idx1 = 0, info = 0;
  int i = 0;
  int fidx = _fidx;

  if (NArray > 0) {
    idx0 = idx1 = 0;
    while (fscanf(fp, "%d ", &i) != EOF) {
      fscanf(fp, "%d\n", &(ArrayIdx[i]));
      if (CheckSite(i, Nlayer) != 0) {
        fprintf(stderr, "Error: Layer index is incorrect. (i=%d, Nlayer=%d) \n",i,Nlayer);
        info = 1;
        break;
      }
      idx0++;
      if (idx0 == Nlayer) break;
    }
    
    idx1 = GetInfoOpt(fp, ArrayOpt, iComplxFlag, iOptCount, fidx);
    if (idx0 != Nlayer || idx1 != NArray) {
      info = ReadDefFileError(defname);
    }
  }
  return info;
}
int GetInfoOptTrans(FILE *fp, int **Array, double *ArrayPara, int *ArrayOpt, int **ArraySgn,
                    int _iFlagOptTrans, int *iOptCount, int _fidx, int _APFlag, int Nsite, int NArray, char *defname) {
  char ctmp2[256];
  int idx = 0, info = 0;
  int i = 0, j = 0;
  int itmp = 0, itmpsgn = 0;
  int fidx = _fidx;
  double dReValue;
  if (_iFlagOptTrans > 0) {
    for (i = 0; i < NArray; i++) {
      itmp = 0;
      dReValue = 0;
      fgets(ctmp2, D_CharTmpReadDef, fp);
      sscanf(ctmp2, "%d %lf \n", &itmp, &dReValue);
      ArrayPara[itmp] = dReValue;
      ArrayOpt[fidx] = 1;
      fidx++;
      (*iOptCount)++;
    }
    idx = 0;
    while (fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp) != NULL) {
      sscanf(ctmp2, "%d %d %d %d\n", &i, &j, &itmp, &itmpsgn);
      Array[i][j] = itmp;
      ArraySgn[i][j] = itmpsgn;
      idx++;
    }
    if (_APFlag == 0) {
      for (i = 0; i < NArray; i++) {
        for (j = 0; j < Nsite; j++) ArraySgn[i][j] = 1;
      }
    }
  }
  return info;
}
int GetInfoOptOrbitalParalell(FILE *fp, int *ArrayOpt, int iComplxFlag, int *iTotalOptCount, int fidx) {
  int i;
  int iLocalOptCount = 0;
  int idxOptFlag = 0;
  while (fscanf(fp, "%d ", &i) != EOF) {
    idxOptFlag = 2 * (fidx + 2 * iLocalOptCount);
    fscanf(fp, "%d\n", &(ArrayOpt[idxOptFlag])); // up-up real
    ArrayOpt[idxOptFlag + 1] = iComplxFlag; //  up-up imag
    ArrayOpt[idxOptFlag + 2] = ArrayOpt[idxOptFlag]; //  up-up imag
    ArrayOpt[idxOptFlag + 3] = iComplxFlag; //  up-up imag
    (iLocalOptCount)++;
    (*iTotalOptCount) += 2;
  }
  return (iLocalOptCount);
}
