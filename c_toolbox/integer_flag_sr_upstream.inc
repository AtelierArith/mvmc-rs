/* Generated verbatim from authoritative mVMC-1.3.0: stcopt.c sha256=43ed8790cff2715284849f0f8906e4645179dcbb100b51f819918b279d6a36f2.
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
 * Stochastic Reconfiguration method
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/

// #define _DEBUG_STCOPT

  si = 0;
  for(pi=0;pi<2*nPara;pi++) {
    //printf("DEBUG: nPara=%d pi=%d OptFlag=%d r=%lf\n",nPara,pi,OptFlag[pi],r[pi]);
    if(OptFlag[pi]!=1) { /* fixed by OptFlag */
      optNum++;
      continue; //skip sDiag
    }
// s:this part will be skipped if OptFlag[pi]!=1
    sDiag = r[pi];
    if(sDiag < diagCutThreshold) { /* fixed by diagCut */
      cutNum++;
    } else { /* optimized */
      smatToParaIdx[si] = pi; // si -> restricted parameters , pi -> full paramer 0 <-> 2*NPara
      si += 1;
    }
// e
  }
  nSmat = si;
  for(si=nSmat;si<2*nPara;si++) {
    smatToParaIdx[si] = -1; // parameters that will not be optimized
  }
