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
int GetInfoGeneralRBM_Layer(FILE *fp, int *ArrayIdx, int *ArrayOpt, int iComplxFlag, int *iOptCount, int _fidx, int Nlayer, int NArray,
                      char *defname) {
  int idx0 = 0, idx1 = 0, info = 0;
  int i = 0, s = 0;
  int fidx = _fidx;

  if (NArray > 0) {
    idx0 = idx1 = 0;
    while (fscanf(fp, "%d %d ", &i, &s) != EOF) {
      fscanf(fp, "%d\n", &(ArrayIdx[i + s*Nsite]));
      if (CheckSite(i, Nlayer) != 0 || CheckSite(s, 2) != 0) {
        fprintf(stderr, "Error: Layer index is incorrect. (i=%d, Nlayer=%d) \n",i,Nlayer);
        info = 1;
        break;
      }
      idx0++;
      if (idx0 == 2*Nlayer) break;
    }
    
    idx1 = GetInfoOpt(fp, ArrayOpt, iComplxFlag, iOptCount, fidx);
    if (idx0 != 2*Nlayer || idx1 != NArray) {
      info = ReadDefFileError(defname);
    }
  }
  return info;
}
int GetInfoRBM_PhysHidden(FILE *fp, int **ArrayIdx, int *ArrayOpt, int iComplxFlag, int *iOptCount, int _fidx, int Nlayer, int Nlayer2, int NArray,
                      char *defname) {
  int idx0 = 0, idx1 = 0, info = 0;
  int i = 0, j = 0;
  int fidx = _fidx;
  if (NArray > 0) {
    while (fscanf(fp, "%d %d ", &i, &j) != EOF) {
      if (CheckSite(i, Nlayer) != 0 || CheckSite(j, Nlayer2)) {
        fprintf(stderr, "Error: Site index is incorrect. \n");
        info = 1;
        break;
      }

      fscanf(fp, "%d\n", &(ArrayIdx[i][j]));
      idx0++;
      if (idx0 == Nlayer * Nlayer2) break;
    }
    idx1 = GetInfoOpt(fp, ArrayOpt, iComplxFlag, iOptCount, fidx);
    if (idx0 != Nlayer * Nlayer2 || idx1 != NArray) {
      info = ReadDefFileError(defname);
    }
  }
  return info;

}
int GetInfoGeneralRBM_PhysHidden(FILE *fp, int **ArrayIdx, int *ArrayOpt, int iComplxFlag, int *iOptCount, int _fidx, int Nlayer, int Nlayer2, int NArray,
                      char *defname) {
  int idx0 = 0, idx1 = 0, info = 0;
  int i = 0, j = 0, s = 0;
  int fidx = _fidx;
  if (NArray > 0) {
    while (fscanf(fp, "%d %d %d ", &i, &s, &j) != EOF) {
      if (CheckSite(i, Nlayer) != 0 || CheckSite(s, 2) != 0 || CheckSite(j, Nlayer2) != 0) {
        fprintf(stderr, "Error: Site index is incorrect. \n");
        info = 1;
        break;
      }

      fscanf(fp, "%d\n", &(ArrayIdx[i+s*Nsite][j]) );
      idx0++;
      if (idx0 == 2*Nlayer * Nlayer2) break;
    }
    idx1 = GetInfoOpt(fp, ArrayOpt, iComplxFlag, iOptCount, fidx);
    if (idx0 != 2*Nlayer * Nlayer2 || idx1 != NArray) {
      info = ReadDefFileError(defname);
    }
  }
  return info;

}
