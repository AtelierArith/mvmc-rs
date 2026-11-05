/* Generated verbatim from authoritative mVMC-1.3.0: readdef.c sha256=9e1f2c635bffb2acdf3210613ab7330c5b28967fae928b303f3b151dfa1b7ef3; readdef.h sha256=02c6e668b13925bf1b504b8d4e6f6dc437fbdf3fa6dc5a4425f210d841fc8409; global.h sha256=247e935bceed9e13e4585daabd91575b34fb1b54697772ad2a70e16b7072f821.
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
int ReadDefFileError(const char *defname) {
  fprintf(stderr, "error: %s (Broken file or Not exist)\n", defname);
  return 1;
}
void SetDefaultValuesModPara(int *bufInt, double *bufDouble) {
  bufInt[IdxVMCCalcMode] = 0;
  bufInt[IdxLanczosMode] = 0;
  bufInt[IdxDataIdxStart] = 0;
  bufInt[IdxDataQtySmp] = 1;
  bufInt[IdxNsite] = 16;
  bufInt[IdxNe] = 8;
  bufInt[IdxSPGaussLeg] = 1;
  bufInt[IdxSPStot] = 0;
  bufInt[IdxMPTrans] = 0;
  bufInt[IdxSROptItrStep] = 1000;
  bufInt[IdxSROptItrSmp] = bufInt[IdxSROptItrStep] / 10;
  bufInt[IdxSROptFixSmp] = 1;
  bufInt[IdxVMCWarmUp] = 10;
  bufInt[IdxVMCInterval] = 1;
  bufInt[IdxVMCSample] = 10;
  bufInt[IdxExUpdatePath] = 0;
  bufInt[IdxRndSeed] = 11272;
  bufInt[IdxSplitSize] = 1;
  bufInt[IdxNLocSpin] = 0;
  bufInt[IdxNTrans] = 0;
  bufInt[IdxNCoulombIntra] = 0;
  bufInt[IdxNCoulombInter] = 0;
  bufInt[IdxNHund] = 0;
  bufInt[IdxNPairHop] = 0;
  bufInt[IdxNExchange] = 0;
  bufInt[IdxNGutz] = 0;
  bufInt[IdxNJast] = 0;
  bufInt[IdxNSpinJast] = 0;
  bufInt[IdxNDH2] = 0;
  bufInt[IdxNDH4] = 0;
  bufInt[IdxNOrbit] = 0;
  bufInt[IdxNQPTrans] = 0;
  bufInt[IdxNOneBodyG] = 0;
  bufInt[IdxNTwoBodyG] = 0;
  bufInt[IdxNTwoBodyGEx] = 0;
  bufInt[IdxNInterAll] = 0;
  bufInt[IdxNQPOptTrans] = 1;
  bufInt[IdxSROptCGMaxIter] = 0;
  bufInt[IdxNBF] = 0;
  bufInt[IdxNrange] = 0;
  bufInt[IdxNNz] = 0;
  bufInt[Idx2Sz] = -1;// -1: sz is not fixed :fsz
  bufInt[IdxNCond] = -1;

//RBM 
  bufInt[IdxNneuron] = 0;
  bufInt[IdxNneuronGeneral] = 0;
  bufInt[IdxNneuronCharge] = 0;
  bufInt[IdxNneuronSpin] = 0;
  bufInt[IdxNChargeRBM_HiddenLayer] = 0;
  bufInt[IdxNChargeRBM_PhysLayer]   = 0;
  bufInt[IdxNChargeRBM_PhysHidden]  = 0;
  bufInt[IdxNSpinRBM_HiddenLayer] = 0;
  bufInt[IdxNSpinRBM_PhysLayer]   = 0;
  bufInt[IdxNSpinRBM_PhysHidden]  = 0;
  bufInt[IdxNGeneralRBM_HiddenLayer] = 0;
  bufInt[IdxNGeneralRBM_PhysLayer]   = 0;
  bufInt[IdxNGeneralRBM_PhysHidden]  = 0;
  bufInt[IdxNBlockSize_RBMRatio]  = 200;
//RBM

  bufDouble[IdxSROptRedCut] = 0.001;
  bufDouble[IdxSROptStaDel] = 0.02;
  bufDouble[IdxSROptStepDt] = 0.02;
  bufDouble[IdxSROptCGTol] = 1.0e-10;
  NStoreO = 1;
  NSRCG = 0;
  NSRCGFallback = 0;
  NSRCGAbortOnFail = 1;
  RescaleSmat  = 0;
  useDiagScale = 0;
  reweight = 0;

  bufInt[IdxNx] = 1;
  bufInt[IdxNy] = 1;
  bufInt[IdxNz] = 1;
  bufInt[IdxNorb] = 1;
  bufInt[IdxNTwist] = 0;
}
int GetInfoFromModPara(int *bufInt, double *bufDouble) {

  FILE *fp;
  char defname[D_FileNameMax];
  char ctmp[D_FileNameMax];
  char ctmp2[D_FileNameMax];

  int itmp;
  char *cerr;

  int iKWidx = 0;
  int iret = 0;
  fprintf(stdout, "Start: Read ModPara File .\n");
  for (iKWidx = 0; iKWidx < KWIdxInt_end; iKWidx++) {
    strcpy(defname, cFileNameListFile[iKWidx]);
    if (strcmp(defname, "") == 0) continue;
    fp = fopen(defname, "r");
    if (fp == NULL) {
      iret=ReadDefFileError(defname);
      fclose(fp);
      break;
    } else {
      switch (iKWidx) {
        case KWModPara:
          /* Read modpara.def---------------------------------------*/
          cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp);
          if(cerr == NULL) return -1;
          cerr = fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp);
          if(cerr == NULL) return -1;
          sscanf(ctmp2, "%s %d\n", ctmp, &itmp); //2
          cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp); //3
          if(cerr == NULL) return -1;
          cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp); //4
          if(cerr == NULL) return -1;
          cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp); //5
          if(cerr == NULL) return -1;
          cerr = fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp);
          if(cerr == NULL) return -1;
          sscanf(ctmp2, "%s %s\n", ctmp, CDataFileHead); //6
          cerr = fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp);
          if(cerr == NULL) return -1;
          sprintf(ctmp, "output/%s", CDataFileHead);
          strcpy(CDataFileHead, ctmp);
          sscanf(ctmp2, "%s %s\n", ctmp, CParaFileHead); //7
          sprintf(ctmp, "output/%s", CParaFileHead);
          strcpy(CParaFileHead, ctmp);
          cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp);   //8
          if(cerr == NULL) return -1;

          iret = system("mkdir -p output");

          double dtmp;
          while (fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp) != NULL) {
            if (*ctmp2 == '\n' || ctmp2[0] == '-') continue;
            sscanf(ctmp2, "%s %lf\n", ctmp, &dtmp);
            if (CheckWords(ctmp, "NVMCCalMode") == 0) {
              bufInt[IdxVMCCalcMode] = (int) dtmp;
            } else if (CheckWords(ctmp, "NLanczosMode") == 0) {
              bufInt[IdxLanczosMode] = (int) dtmp;
            } else if (CheckWords(ctmp, "NDataIdxStart") == 0) {
              bufInt[IdxDataIdxStart] = (int) dtmp;
            } else if (CheckWords(ctmp, "NDataQtySmp") == 0) {
              bufInt[IdxDataQtySmp] = (int) dtmp;
            } else if (CheckWords(ctmp, "Nsite") == 0) {
              bufInt[IdxNsite] = (int) dtmp;
            } else if (CheckWords(ctmp, "Ne") == 0 || CheckWords(ctmp, "Nelectron") == 0) {
              bufInt[IdxNe] = (int) dtmp;
            } else if (CheckWords(ctmp, "Ncond") == 0) {
              bufInt[IdxNCond] = (int) dtmp;
            } else if (CheckWords(ctmp, "2Sz") == 0) {
              bufInt[Idx2Sz] = (int) dtmp;
              if (bufInt[Idx2Sz] == -1) {
                fprintf(stdout, "Error: 2Sz must be even number.\n");
                return (-1);
              }
            } else if (CheckWords(ctmp, "NSPGaussLeg") == 0) {
              bufInt[IdxSPGaussLeg] = (int) dtmp;
            } else if (CheckWords(ctmp, "NSPStot") == 0) {
              bufInt[IdxSPStot] = (int) dtmp;
            } else if (CheckWords(ctmp, "NMPTrans") == 0) {
              bufInt[IdxMPTrans] = (int) dtmp;
            } else if (CheckWords(ctmp, "NSROptItrStep") == 0) {
              bufInt[IdxSROptItrStep] = (int) dtmp;
            } else if (CheckWords(ctmp, "NSROptItrSmp") == 0) {
              bufInt[IdxSROptItrSmp] = (int) dtmp;
            } else if (CheckWords(ctmp, "DSROptRedCut") == 0) {
              bufDouble[IdxSROptRedCut] = (double) dtmp;
            } else if (CheckWords(ctmp, "DSROptStaDel") == 0) {
              bufDouble[IdxSROptStaDel] = (double) dtmp;
            } else if (CheckWords(ctmp, "DSROptStepDt") == 0) {
              bufDouble[IdxSROptStepDt] = (double) dtmp;
            } else if (CheckWords(ctmp, "NSROptCGMaxIter") == 0) {
              bufInt[IdxSROptCGMaxIter] = (int) dtmp;
            } else if (CheckWords(ctmp, "DSROptCGTol") == 0) {
              bufDouble[IdxSROptCGTol] = (double) dtmp;
            } else if (CheckWords(ctmp, "NVMCWarmUp") == 0) {
              bufInt[IdxVMCWarmUp] = (int) dtmp;
            } else if (CheckWords(ctmp, "NVMCInterval") == 0) {
              bufInt[IdxVMCInterval] = (int) dtmp;
            } else if (CheckWords(ctmp, "NVMCSample") == 0) {
              bufInt[IdxVMCSample] = (int) dtmp;
            } else if (CheckWords(ctmp, "NExUpdatePath") == 0) {
              bufInt[IdxExUpdatePath] = (int) dtmp;
            } else if (CheckWords(ctmp, "RndSeed") == 0) {
              bufInt[IdxRndSeed] = (int) dtmp;
            } else if (CheckWords(ctmp, "NSplitSize") == 0) {
              bufInt[IdxSplitSize] = (int) dtmp;
            } else if (CheckWords(ctmp, "NStore") == 0) {
              NStoreO = (int) dtmp;
            } else if (CheckWords(ctmp, "NSRCG") == 0) {
              NSRCG = (int) dtmp;
            } else if (CheckWords(ctmp, "NSRCGFallback") == 0) {
              NSRCGFallback = (int) dtmp;
            } else if (CheckWords(ctmp, "NSRCGAbortOnFail") == 0) {
              NSRCGAbortOnFail = (int) dtmp;
            } else if (CheckWords(ctmp, "RescaleSmat") == 0) {
              RescaleSmat = (int) dtmp;
            } else if (CheckWords(ctmp, "useDiagScale") == 0) {
              useDiagScale = (int) dtmp;
            } else if (CheckWords(ctmp, "reweight") == 0) {
              reweight = (int) dtmp;
//RBM
            } else if (CheckWords(ctmp, "Nneuron") == 0) {
              bufInt[IdxNneuron] = (int) dtmp;
            } else if (CheckWords(ctmp, "NneuronCharge") == 0) {
              bufInt[IdxNneuronCharge] = (int) dtmp;
            } else if (CheckWords(ctmp, "NneuronSpin") == 0) {
              bufInt[IdxNneuronSpin] = (int) dtmp;
            } else if (CheckWords(ctmp, "NneuronGeneral") == 0) {
              bufInt[IdxNneuronGeneral] = (int) dtmp;
            } else if (CheckWords(ctmp, "NBlockSize_RBMRatio") == 0) {
              bufInt[IdxNBlockSize_RBMRatio] = (int) dtmp;
//RBM
            } else {
              fprintf(stderr, "  Error: keyword \" %s \" is incorrect. \n", ctmp);
              iret = ReadDefFileError(defname);
              return iret;
            }
          }
          if (bufInt[IdxRndSeed] < 0) {
            bufInt[IdxRndSeed] = (int) time(NULL);
            fprintf(stdout, "  remark: Seed = %d\n", bufInt[IdxRndSeed]);
          }
          break;//modpara file
        default:
          break;
      }
      fclose(fp);
    }
  }
  //fclose(fp);
  fprintf(stdout, "End: Read ModPara File .\n");
  return iret;
}
/* Wrapper added by the toolbox: the block below is verbatim from ReadDefFileNInt. */
static void ApplyNBlockSizeRBMRatioAdjustment(int *bufInt) {
  int itmp;
  if((bufInt[IdxNBlockSize_RBMRatio]%8)>0) {
      itmp=bufInt[IdxNBlockSize_RBMRatio]/8;
      if(itmp==0) itmp=1;
      bufInt[IdxNBlockSize_RBMRatio]=itmp*8;
      printf("\n --- Notice --- \n");
      printf("NBlockSize_RBMRatio Adjust to %d\n\n",bufInt[IdxNBlockSize_RBMRatio]);
    }
}
/* Wrapper added by the toolbox: the block below is verbatim from ReadDefFileNInt. */
static void ApplySrCgNormalization(int rank) {
  if (NSRCG == 2){
    useDiagScale = 1;
    NSRCG = 1;
  //  if (rank == 0) printf("remark: use preconditioned CG (Diag Scale)\n");
  } //else {
  //  useDiagScale = 0;
  //}
  NSRCGFallback = (NSRCGFallback != 0);
  NSRCGAbortOnFail = (NSRCGAbortOnFail != 0);

  if (useDiagScale){
    if(NSRCG == 1){
      if (rank == 0) printf("remark: use preconditioned CG (Diag Scale)\n");
    }else{
      if (rank == 0) printf("remark: not use preconditioned CG (Diag Scale) because NSRCG=%d != 1. Use direct method instead.\n",NSRCG);
      useDiagScale = 0;
    }
  }

  if (RescaleSmat){
    if (NSRCG != 1) {
      if (rank == 0) {
        fprintf(stderr,
                "error: invalid RescaleSmat configuration. "
                "RescaleSmat=1 requires NSRCG==1, but got NSRCG=%d.\n",
                NSRCG);
      }
      MPI_Abort(comm, EXIT_FAILURE);
    }
    if (rank == 0) printf("remark: rescale S matrix \n");
  }

  if (DSROptStepDt < 0) {
    SRFlag = 1; /* diagonalization */
    if (rank == 0) fprintf(stderr, "remark: Diagonalization Mode\n");
    DSROptStepDt *= -1;
  } else {
    SRFlag = 0;
  }

  if (SRFlag != 0 && NSRCG != 0) {
    if (rank == 0) {
      fprintf(stderr,
              "error: SR-CG (NSRCG=%d) cannot be combined with diagonalization mode "
              "(DSROptStepDt < 0). Use either DSROptStepDt > 0 with SR-CG, "
              "or DSROptStepDt < 0 with NSRCG=0.\n",
              NSRCG);
    }
    MPI_Abort(comm, EXIT_FAILURE);
  }
}
