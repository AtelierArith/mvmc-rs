/* Generated verbatim from authoritative mVMC-1.3.0: readdef.c sha256=9e1f2c635bffb2acdf3210613ab7330c5b28967fae928b303f3b151dfa1b7ef3.
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
int GetInfoOpt(FILE *fp, int *ArrayOpt, int iComplxFlag, int *iTotalOptCount, int fidx) {
  int i;
  int iLocalOptCount = 0;
  while (fscanf(fp, "%d ", &i) != EOF) {
    fscanf(fp, "%d\n", &(ArrayOpt[2 * fidx])); // TBC real
    if(iComplxFlag>0){
      ArrayOpt[2 * fidx + 1] = ArrayOpt[2 * fidx]; //  TBC imaginary
    }else{
      ArrayOpt[2 * fidx + 1] = 0;
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
int GetInfoJastrow(FILE *fp, int **ArrayIdx, int *ArrayOpt, int iComplxFlag, int *iOptCount, int _fidx, int Nsite,
                   int NArray, char *defname) {
  int idx0 = 0, idx1 = 0, info = 0;
  int i = 0, j = 0, n = 0;
  int a = 0, b = 0;
  int line = 0;
  int idxExpected = Nsite * (Nsite - 1);
  int *pairClass = NULL;
  int fidx = _fidx;
  if (NArray > 0) {
    pairClass = (int *)malloc(sizeof(int) * Nsite * Nsite);
    if (pairClass == NULL) {
      fprintf(stderr, "Error: memory allocation failed in GetInfoJastrow.\n");
      return 1;
    }
    for (i = 0; i < Nsite; i++) {
      for (j = 0; j < Nsite; j++) {
        if (i == j) {
          ArrayIdx[i][j] = -1;
          pairClass[i * Nsite + j] = -1;
        } else {
          ArrayIdx[i][j] = -2;
          pairClass[i * Nsite + j] = -2;
        }
      }
    }

    while (idx0 < idxExpected) {
      if (fscanf(fp, "%d %d %d", &i, &j, &n) != 3) {
        fprintf(stderr, "Error in %s: failed to read Jastrow row %d.\n", defname, line + 1);
        info = 1;
        break;
      }
      line++;
      if (i == j) {
        fprintf(stderr, "Error in %s: [Condition] i neq j at row %d.\n", defname, line);
        info = 1;
        break;
      }
      if (CheckPairSite(i, j, Nsite) != 0) {
        fprintf(stderr, "Error in %s: Site index is incorrect at row %d.\n", defname, line);
        info = 1;
        break;
      }
      if (n < 0 || n >= NArray) {
        fprintf(stderr, "Error in %s: class index %d out of range [0, %d) at row %d.\n",
                defname, n, NArray, line);
        info = 1;
        break;
      }
      if (ArrayIdx[i][j] != -2) {
        fprintf(stderr, "Error in %s: duplicated pair (%d,%d) at row %d.\n", defname, i, j, line);
        info = 1;
        break;
      }
      ArrayIdx[i][j] = n;

      if (i < j) {
        a = i;
        b = j;
      } else {
        a = j;
        b = i;
      }
      if (pairClass[a * Nsite + b] == -2) {
        pairClass[a * Nsite + b] = n;
      } else if (pairClass[a * Nsite + b] != n) {
        fprintf(stderr, "Error in %s: inconsistent class for pair (%d,%d) at row %d.\n",
                defname, a, b, line);
        info = 1;
        break;
      }
      idx0++;
    }
    if (info == 0) {
      for (i = 0; i < Nsite; i++) {
        for (j = 0; j < Nsite; j++) {
          if (i == j) continue;
          if (ArrayIdx[i][j] == -2) {
            fprintf(stderr, "Error in %s: missing pair (%d,%d).\n", defname, i, j);
            info = 1;
            break;
          }
        }
        if (info != 0) break;
      }
    }
    if (info == 0) {
      for (i = 0; i < Nsite; i++) {
        for (j = i + 1; j < Nsite; j++) {
          if (pairClass[i * Nsite + j] == -2) {
            fprintf(stderr, "Error in %s: missing undirected pair (%d,%d).\n", defname, i, j);
            info = 1;
            break;
          }
          ArrayIdx[i][j] = pairClass[i * Nsite + j];
          ArrayIdx[j][i] = pairClass[i * Nsite + j];
        }
        if (info != 0) break;
      }
    }
    if (info == 0) {
      idx1 = GetInfoOpt(fp, ArrayOpt, iComplxFlag, iOptCount, fidx);
    }
    if (info != 0 || idx0 != idxExpected || idx1 != NArray) {
      info = ReadDefFileError(defname);
    }
    free(pairClass);
  }
  return info;
}
