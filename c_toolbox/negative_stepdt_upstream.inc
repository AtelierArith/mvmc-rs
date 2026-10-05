/* Generated verbatim from authoritative mVMC-1.3.0: readdef.c sha256=6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9; initfile.c sha256=8a7b20d54ab495cfac4df42646a30102aebaa228311ca3b17c1e29af5d0581f8; stcopt_dposv.c sha256=2bd48d880dcbd95ea1b1b92931178c07fd08f981b8ebf04e7db1709e909e57ca; global.h sha256=a8834e9457e5b8345496087222e5f724c26896928f86fe1bf4ac7e8190b409e1.
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

/* Wrapper added by the toolbox: the block is verbatim from ReadDefFileNInt. */
static void NormalizeStepDt(int rank) {
  if (DSROptStepDt < 0) {
    SRFlag = 1; /* diagonalization */
    if (rank == 0) fprintf(stderr, "remark: Diagonalization Mode\n");
    DSROptStepDt *= -1;
  } else {
    SRFlag = 0;
  }
}
/* Verbatim initfile.c SRinfo header selection (printed by the toolbox probe). */
#if 0
if(SRFlag == 0){
      fprintf(FileSRinfo,
            "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax\n");
    }else{
      fprintf(FileSRinfo,
            "#Npara Msize optCut diagCut sEigenMax  sEigenMin    absRmax       imax\n");
    }
#endif
void stcOptInit(double *const S, double *const g, const int nSmat, const int *const smatToParaIdx) {
  const double ratioDiag = 1.0 + DSROptStaDel;
  int si,sj,pi,pj,idx,offset;
  double tmp;
  
  /* calculate the overlap matrix S */
  /* S[i][j] = OO[i+1][j+1] - OO[0][i+1] * OO[0][j+1]; */
  for(si=0;si<nSmat;++si) {
    pi = smatToParaIdx[si];
    //offset = (pi+1)*SROptSize;
    offset = (pi+2)*(2*SROptSize);
    tmp = creal(SROptOO[pi+2]);

    for(sj=0;sj<nSmat;++sj) {
      pj = smatToParaIdx[sj];
      idx = si + nSmat*sj; /* column major */
      S[idx] = creal(SROptOO[offset+(pj+2)]) - tmp * creal(SROptOO[pj+2]);
    }

    /* modify diagonal elements */
    idx = si + nSmat*si;
    S[idx] *= ratioDiag;
  }

  /* calculate the energy gradient * (-dt) */
  /* energy gradient = 2.0*( HO[i+1] - HO[0] * OO[i+1]) */
  for(si=0;si<nSmat;++si) {
    pi = smatToParaIdx[si];
    g[si] = -DSROptStepDt*2.0*(creal(SROptHO[pi+2]) - creal(SROptHO[0]) * creal(SROptOO[pi+2]));
  }

  return;
}
