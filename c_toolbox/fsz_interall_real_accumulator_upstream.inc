/* Generated verbatim from authoritative mVMC-1.3.0: locgrn_fsz_real.c sha256=be6a1028502e1b6b37806a2442f5e77f160b63306e7b3132e1c62697b4bb88a0; calham_fsz_real.c sha256=8e0d24a4a4b9f431dec235871923fb853876c52dcf27489b8f6c258fe3358684.
 * Regenerate with the corresponding scripts/check_*_c_parity.py --write.
 * This optional C oracle is not a Rust test/build dependency. */
/*
mVMC - A numerical solver package for a wide range of quantum lattice models based on many-variable Variational Monte Carlo method
Copyright (C) 2016 The University of Tokyo, All rights reserved.

his program is developed based on the mVMC-mini program
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
 * local Green Functions
 *-------------------------------------------------------------
 *-------------------------------------------------------------*/
for(idx=0;idx<NInterAll;idx++) {
      ri = InterAll[idx][0];
      s  = InterAll[idx][1];
      rj = InterAll[idx][2];
      t  = InterAll[idx][3];
      rk = InterAll[idx][4];
      u  = InterAll[idx][5];
      rl = InterAll[idx][6];
      v  = InterAll[idx][7];
      
      if(s==t && u==v){
        myEnergy += ParaInterAll[idx]
          * GreenFunc2_fsz_real(ri,rj,rk,rl,s,u,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myEleSpn,myProjCntNew,myBuffer);
      }else{
        myEnergy += ParaInterAll[idx]
          * GreenFunc2_fsz2_real(ri,rj,rk,rl,s,t,u,v,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myEleSpn,myProjCntNew,myBuffer);
      } 
    }
