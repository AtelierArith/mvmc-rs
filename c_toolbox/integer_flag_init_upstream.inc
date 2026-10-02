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
void InitParameter() {
  int i;
  //printf("AllComplexFlag=%d \n",AllComplexFlag);
  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<NProj;i++) Proj[i] = 0.0;

  if (FlagRBM) {
    if(AllComplexFlag==0){
      for(i=0;i<NRBM;i++){
        if(OptFlag[2*i+2*NProj] > 0){ //TBC
          RBM[i]  =   0.01*(genrand_real2() -0.5)/(double)Nneuron; /* uniform distribution [0,1) */
        } else {
          RBM[i] = 0.0;
        }
      }
    }else{
      for(i=0;i<NRBM;i++){
        if(OptFlag[2*i+2*NProj] > 0){ //TBC
          RBM[i]  =   1e-2*genrand_real2()*cexp(2.0*I*M_PI*genrand_real2());
          //RBM[i]  =   1e-2*genrand_real2()*cexp(2.0*I*M_PI*genrand_real2())/(double)Nneuron;
        } else {
          RBM[i] = 0.0;
        }
      }
    }
  }

  if(AllComplexFlag==0){
    for(i=0;i<NSlater;i++){
      if(OptFlag[2*i+2*NProj + 2*FlagRBM*NRBM] > 0){ //TBC
        Slater[i] =  2*(genrand_real2()-0.5); /* uniform distribution [-1,1) */
        //Slater[i] =  1*genrand_real2(); /* uniform distribution [0,1) */
        //Slater[i] += 1*I*genrand_real2(); /* uniform distribution [0,1) */
        //printf("DEBUG: i=%d slater=%lf %lf \n",i,creal(Slater[i]),cimag(Slater[i]));
      } else {
        Slater[i] = 0.0;
      }
    }
  }
  else{
    for(i=0;i<NSlater;i++){
      if(OptFlag[2*i+2*NProj + 2*FlagRBM*NRBM] > 0){ //TBC
        Slater[i] =  2*(genrand_real2()-0.5); /* uniform distribution [-1,1) */
        Slater[i] += 2*I*(genrand_real2()-0.5); /* uniform distribution [-1,1) */
        Slater[i] /=sqrt(2.0);
        //printf("i=%d: %lf %lf \n",i,creal(Slater[i]),cimag(Slater[i]));
      } else {
        Slater[i] = 0.0;
      }
    }
  }

  for(i=0;i<NOptTrans;i++){
    OptTrans[i] = ParaQPOptTrans[i];
  }

  return;
}
