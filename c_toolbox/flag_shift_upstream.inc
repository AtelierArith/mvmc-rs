/* Generated verbatim from authoritative mVMC-1.3.0: parameter.c sha256=46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0.
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
 * Variational parameters
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/
void SetFlagShift() {
  int i,start,end;

  /* Gutzwiller */
  FlagShiftGJ=0;
  FlagShiftDH2=0;
  FlagShiftDH4=0;
  if(NGutzwillerIdx==0) return; // no Gutz -> do nothing
  start = 0;
  end = start + NGutzwillerIdx;
  for(i=start;i<end;i++) {
    if(OptFlag[2*i]!=1) return;   // fixed Gutx -> do nothing
  }
  
  /* Jastrow */
  if(NJastrowIdx>0) {
    start = end;
    end   = start + NJastrowIdx;
    FlagShiftGJ=1; // unfixed J -> FlagShift on
    for(i=start;i<end;i++) {
      if(OptFlag[2*i]!=1) {       // fixed jast -> do nothing
        FlagShiftGJ=0;
        break;
      }
    }
  }
  
  /* 2-site Doublon-Holon */
  if(NDoublonHolon2siteIdx>0) {
    start = end;
    end   = start + 6*NDoublonHolon2siteIdx;
    FlagShiftDH2=1;         // unfixed D-H -> FlagShift on
    for(i=start;i<end;i++) {
      if(OptFlag[2*i]!=1) { // fixed D-H -> do nothing
        FlagShiftDH2=0;
        break;
      }
    }
  }
  
  /* 4-site Doublon-Holon */
  if(NDoublonHolon4siteIdx>0) {
    start = end;
    end   = start + 10*NDoublonHolon4siteIdx;
    FlagShiftDH4=1;         // unfixed D-H -> FlagShift on
    for(i=start;i<end;i++) { 
      if(OptFlag[2*i]!=1) { // fixed D-H -> do nothing
        FlagShiftDH4=0;
        break;
      }
    }
  }
  
  return;
}
