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
int CheckWords(
    const char *ctmp,
    const char *cKeyWord
) {

  int i = 0;

  char ctmp_small[256] = {0};
  char cKW_small[256] = {0};
  int n;
  n = strlen(cKeyWord);
  strncpy(cKW_small, cKeyWord, n);

  for (i = 0; i < n; i++) {
    cKW_small[i] = tolower(cKW_small[i]);
  }

  n = strlen(ctmp);
  strncpy(ctmp_small, ctmp, n);
  for (i = 0; i < n; i++) {
    ctmp_small[i] = tolower(ctmp_small[i]);
  }
  if (n < strlen(cKW_small)) n = strlen(cKW_small);
  return (strncmp(ctmp_small, cKW_small, n));
}
int CheckKW(
    const char *cKW,
    char cKWList[][D_CharTmpReadDef],
    int iSizeOfKWidx,
    int *iKWidx
) {
  *iKWidx = -1;
  int itmpKWidx;
  int iret = -1;
  for (itmpKWidx = 0; itmpKWidx < iSizeOfKWidx; itmpKWidx++) {
    if (strcmp(cKW, "") == 0) {
      break;
    } else if (CheckWords(cKW, cKWList[itmpKWidx]) == 0) {
      iret = 0;
      *iKWidx = itmpKWidx;
    }
  }
  return iret;
}
int GetFileName(
    const char *cFileListNameFile,
    char cFileNameList[][D_CharTmpReadDef]
) {
  FILE *fplist;
  int itmpKWidx = -1;
  char ctmpFileName[D_FileNameMaxReadDef];
  char ctmpKW[D_CharTmpReadDef], ctmp2[256];
  int i;
  for (i = 0; i < KWIdxInt_end; i++) {
    strcpy(cFileNameList[i], "");
  }

  fplist = fopen(cFileListNameFile, "r");
  if (fplist == NULL) return ReadDefFileError(cFileListNameFile);

  while (fgets(ctmp2, 256, fplist) != NULL) {
    sscanf(ctmp2, "%s %s\n", ctmpKW, ctmpFileName);
    if (strncmp(ctmpKW, "#", 1) == 0 || *ctmp2 == '\n' || (strcmp(ctmpKW, "") && strcmp(ctmpFileName, "")) == 0) {
      continue;
    } else if (strcmp(ctmpKW, "") * strcmp(ctmpFileName, "") == 0) {
      fprintf(stderr, "Error: keyword and filename must be set as a pair in %s.\n", cFileListNameFile);
      fclose(fplist);
      return (-1);
    }
    /*!< Check KW */
    if (CheckKW(ctmpKW, cKWListOfFileNameList, KWIdxInt_end, &itmpKWidx) != 0) {

      fprintf(stderr, "Error: Wrong keywords '%s' in %s.\n", ctmpKW, cFileListNameFile);
      fprintf(stderr, "%s", "Choose Keywords as follows: \n");
      for (i = 0; i < KWIdxInt_end; i++) {
        fprintf(stderr, "%s \n", cKWListOfFileNameList[i]);
      }
      fclose(fplist);
      return (-1);
    }
    /*!< Check cFileNameList to prevent from double registering the file name */
    if (strcmp(cFileNameList[itmpKWidx], "") != 0) {
      fprintf(stderr, "Error: Same keywords exist in %s.\n", cFileListNameFile);
      fclose(fplist);
      return (-1);
    }
    /*!< Copy FileName */
    strcpy(cFileNameList[itmpKWidx], ctmpFileName);
  }
  fclose(fplist);
  return 0;
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
    }
    fidx++;
    (iLocalOptCount)++;
    (*iTotalOptCount)++;
  }
  return (iLocalOptCount);
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
int GetInfoOrbitalAntiParallel(FILE *fp, int **Array, int *ArrayOpt, int **ArraySgn, int *iOptCount,
                               int _fidx, int _iComplexFlag, int _iFlagOrbitalGeneral, int _APFlag, int Nsite,
                               int NArray, char *defname) {
  char ctmp2[256];
  int i, j;
  int idx0 = 0, idx1 = 0;
  int itmp = 0, info = 0;
  int spn_i, spn_j;
  int all_i, all_j;
  int fij = 0, fijSign = 1;
  int fidx = _fidx;

  if (NArray == 0) return 0;

  if (_iFlagOrbitalGeneral == 0) {
    while (fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp) != NULL) {
      sscanf(ctmp2, "%d %d %d %d \n", &i, &j, &fij, &fijSign);
      if (CheckPairSite(i, j, Nsite) != 0) {
        fprintf(stderr, "Error: Site index is incorrect. \n");
        return -1;
      }
      idx0++;
      Array[i][j] = fij;
      ArraySgn[i][j] = fijSign;
      if (idx0 == Nsite * Nsite) break;
    }

    if (_APFlag == 0) {
      for (i = 0; i < Nsite; i++) {
        for (j = 0; j < Nsite; j++) {
          ArraySgn[i][j] = 1;
        }
      }
    }
  } else { //_iFlagOrbitalGeneral == 0
    while (fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp) != NULL) {
      sscanf(ctmp2, "%d %d %d %d \n", &i, &j, &fij, &fijSign);
      spn_i = 0;
      spn_j = 1;
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
      if (idx0 == (Nsite * Nsite)) break;
    }

    if (_APFlag == 0) {
      for (i = 0; i < Nsite; i++) {
        for (j = Nsite; j < 2 * Nsite; j++) {
          ArraySgn[i][j] = 1;
          ArraySgn[j][i] = -1;
        }
      }
    }
  }

  idx1 = GetInfoOpt(fp, ArrayOpt, _iComplexFlag, iOptCount, fidx);
  if (idx0 != Nsite * Nsite || idx1 != NArray || itmp == 1) {
    info = ReadDefFileError(defname);
  }

  return info;
}
int GetInfoOrbitalParallel(FILE *fp, int **Array, int *ArrayOpt, int **ArraySgn, int *iOptCount,
                           int _fidx, int _iComplexFlag, int _iFlagOrbitalGeneral, int _APFlag, int Nsite, int NArray,
                           int NArrayAP, char *defname) {
  char ctmp2[256];
  int i, j;
  int idx0 = 0, idx1 = 0;
  int itmp = 0, info = 0;
  int spn_i;
  int all_i, all_j;
  int fij = 0, fijSign = 1, fij_org;
  int fidx = _fidx;

  if (NArray == 0) return 0;
  while (fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp) != NULL) {
    sscanf(ctmp2, "%d %d %d %d\n", &i, &j, &fij_org, &fijSign);

    if (CheckPairSite(i, j, Nsite) != 0) {
      fprintf(stderr, "Error: Site index is incorrect. \n");
      return -1;
    }

    for (spn_i = 0; spn_i < 2; spn_i++) {
      all_i = i + spn_i * Nsite; //fsz
      all_j = j + spn_i * Nsite; //fsz
      if (all_i >= all_j) itmp = 1;
      idx0++;
      fij = NArrayAP + 2 * fij_org + spn_i;

      Array[all_i][all_j] = fij;
      ArraySgn[all_i][all_j] = fijSign;
      // Note F_{IJ}=-F_{JI}
      Array[all_j][all_i] = fij;
      ArraySgn[all_j][all_i] = -fijSign;
    }
    if (idx0 == (Nsite * (Nsite - 1))) break;
  }

  if (_APFlag == 0) {
    for (spn_i = 0; spn_i < 2; spn_i++) {
      for (i = 0; i < Nsite; i++) {
        for (j = i + 1; j < Nsite; j++) {
          all_i = i + spn_i * Nsite; //fsz
          all_j = j + spn_i * Nsite; //fsz
          ArraySgn[all_i][all_j] = 1;
          ArraySgn[all_j][all_i] = -1;
        }
      }
    }
  }


  idx1 = GetInfoOptOrbitalParalell(fp, ArrayOpt, _iComplexFlag, iOptCount, fidx);
  if (idx0 != (Nsite * (Nsite - 1)) || idx1 != NArray || itmp == 1) {
    info = ReadDefFileError(defname);
  }

  return info;
}
