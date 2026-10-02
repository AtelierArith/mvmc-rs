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
int ReadInitParameter(char *initFile) {
  FILE *fp;
  int i, xi;
  double xtmp;
  double tmp_real,tmp_comp;


  fp = fopen(initFile, "r");
  if(fp!=NULL){
    while(fscanf(fp, "%lf ", &xtmp)!=EOF){
      //for(i=1;i<4;i++) fscanf(fp, "%lf ", &xtmp);
      for (i = 1; i < 6; i++) fscanf(fp, "%lf ", &xtmp);
      for(xi=0;xi<NProj;xi++) {
        fscanf(fp, "%lf %lf %lf ", &tmp_real, &tmp_comp, &xtmp);
        Proj[xi] = tmp_real+tmp_comp*I; 
      }
      if (FlagRBM) {
        for(xi=0;xi<NRBM;xi++) {
          fscanf(fp, "%lf %lf %lf ", &tmp_real, &tmp_comp, &xtmp);
//        printf("RBM[%d] = %.3e %.3e\n",xi,tmp_real, tmp_comp);
          RBM[xi] = tmp_real+tmp_comp*I; 
        }
      }
      for(xi=0;xi<NSlater;xi++) {
        fscanf(fp, "%lf %lf %lf ", &tmp_real,&tmp_comp, &xtmp);
        Slater[xi] = tmp_real+tmp_comp*I; 
      }
      for(xi=0;xi<NOptTrans;xi++) {
        fscanf(fp, "%lf %lf %lf ", &tmp_real,&tmp_comp, &xtmp);
        OptTrans[xi] = tmp_real+tmp_comp*I; 
      }
    }
    fclose(fp);
  } else { fprintf(stderr, "Error: %s does not exist.\n",initFile); }
  
  return 0;
}
