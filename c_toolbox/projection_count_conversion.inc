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

  NMPTrans = bufInt[IdxMPTrans];
  if (NMPTrans < 0) {
    APFlag = 1; /* anti-periodic boundary */
    NMPTrans *= -1;
  } else {
    APFlag = 0;
  }

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

  NQPFix = NSPGaussLeg * NMPTrans;
  NQPFull = NQPFix * NQPOptTrans;
