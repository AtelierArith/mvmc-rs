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

  if (rank == 0) {
    AllComplexFlag = iComplexFlgGutzwiller + iComplexFlgJastrow + iComplexFlgSpinJastrow + iComplexFlgDH2; //TBC
    AllComplexFlag += iComplexFlgDH4 + iComplexFlgOrbital;//TBC
    //AllComplexFlag  = 1;//DEBUG
    // AllComplexFlag= 0 -> All real, !=0 -> complex
    //if(AllComplexFlag == 0 && iFlgOrbitalGeneral == 1){
    //    fprintf(stderr, "Error: Variational parameters should be complex when orbital is general in this version.\n");
    //    info = 1;
    //}
    if(iComplexFlgOrbital > 0){
      iComplexFlgOrbital = 1;
      fprintf(stderr, "Warning: All the pairings are treated as complex variational parameters.\n");
    }
    if(FlagRBM == 1 && AllComplexFlag == 0){
      fprintf(stderr, "Error: RBM requires complex variational parameters (AllComplexFlag != 0).\n");
      info = 1;
    }
  }
