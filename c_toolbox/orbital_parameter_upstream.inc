/* Generated verbatim from authoritative mVMC-1.3.0: parameter.c sha256=46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0; slater.c sha256=cba5ddb9b8f42894dfa808aa867b05b358caab17175fb39931caf4625037db70.
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
void SyncModifiedParameter(MPI_Comm comm) {
  double gShift=0.0;
  double xmax,ratio;
  int i;

#ifdef _mpi_use
  int size;
  MPI_Comm_size(comm, &size);
  if(size>1) MPI_Bcast(Para, NPara, MPI_DOUBLE_COMPLEX, 0, comm);
#endif /* _mpi_use */

  /***** shift correlation factors *****/
  /* shift the DH correlation factors */
  if(FlagShiftDH2==1) gShift += shiftDH2();
  if(FlagShiftDH4==1) gShift += shiftDH4();
  /* shift the Gutzwiller factors */
  for(i=0;i<NGutzwillerIdx;i++) Proj[i] += gShift;
  /* shift the Gutzwiller-Jastrow factors */
  if(FlagShiftGJ==1) shiftGJ();

  /***** rescale Slater *****/
  xmax = cabs(Slater[0]);
  for(i=1;i<NSlater;i++){
    if(xmax < cabs(Slater[i])) xmax = cabs(Slater[i]);
  }
  ratio = D_AmpMax/xmax;
  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<NSlater;i++) Slater[i] *= ratio;

  /***** normalize OptTrans *****/
  if(FlagOptTrans>0){
    xmax = cabs(OptTrans[0]);
    for(i=1;i<NOptTrans;++i) {
      if(xmax < cabs(OptTrans[i])) xmax = cabs(OptTrans[i]);
    }
    //ratio = 1.0/copysign(xmax, OptTrans[0]);
    ratio = 1.0/xmax; //TBC
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<NOptTrans;++i) {
      OptTrans[i] *= ratio;
    }
  }

  return;
}
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
void UpdateSlaterElm_fcmp() {
  int ri,ori,tri,sgni,rsi0,rsi1;
  int rj,orj,trj,sgnj,rsj0,rsj1;
  int qpidx,mpidx,spidx,optidx;
  double cs,cc,ss;
  double complex slt_ij,slt_ji;
  int *xqp, *xqpSgn, *xqpOpt, *xqpOptSgn;
  double complex *sltE,*sltE_i0,*sltE_i1;

  #pragma omp parallel for default(shared)        \
    private(qpidx,optidx,mpidx,spidx,                      \
            xqpOpt,xqpOptSgn,xqp,xqpSgn,cs,cc,ss,sltE,     \
            ri,ori,tri,sgni,rsi0,rsi1,sltE_i0,sltE_i1,      \
            rj,orj,trj,sgnj,rsj0,rsj1,slt_ij,slt_ji)
  #pragma loop noalias
  for(qpidx=0;qpidx<NQPFull;qpidx++) {
    // qpidx  = optidx*NQPFix+NSPGaussLeg*mpidx+spidx 
    // NQPFix = NSPGaussLeg*NMPTrans
    optidx    = qpidx / NQPFix;                // optidx -> optrans projection (will not be used ?)
    mpidx     = (qpidx%NQPFix) / NSPGaussLeg;  // mpidx  -> momentum projection
    spidx     = qpidx % NSPGaussLeg;           // spidx  -> spin     projection

    xqpOpt    = QPOptTrans[optidx];    //
    xqpOptSgn = QPOptTransSgn[optidx]; //
    xqp       = QPTrans[mpidx];        //QPTrans[i][j]: i # of trans. op.,j origin of trans. op. 
    xqpSgn    = QPTransSgn[mpidx];
    cs        = SPGLCosSin[spidx];
    cc        = SPGLCosCos[spidx];
    ss        = SPGLSinSin[spidx];
    
    sltE      = SlaterElm + qpidx*Nsite2*Nsite2;
    
    for(ri=0;ri<Nsite;ri++) {
      ori     = xqpOpt[ri];           // ri (OptTrans) -> ori (Trans)-> tri
      tri     = xqp[ori];
      sgni    = xqpSgn[ori]*xqpOptSgn[ri];
      rsi0    = ri;
      rsi1    = ri+Nsite;
      sltE_i0 = sltE + rsi0*Nsite2;
      sltE_i1 = sltE + rsi1*Nsite2;
      
      for(rj=0;rj<Nsite;rj++) {
        orj = xqpOpt[rj];
        trj = xqp[orj];
        sgnj = xqpSgn[orj]*xqpOptSgn[rj];
        rsj0 = rj;
        rsj1 = rj+Nsite;
        
        slt_ij = Slater[ OrbitalIdx[tri][trj] ] * (double)(OrbitalSgn[tri][trj]*sgni*sgnj);
        slt_ji = Slater[ OrbitalIdx[trj][tri] ] * (double)(OrbitalSgn[trj][tri]*sgni*sgnj);
        
        sltE_i0[rsj0] = -(slt_ij - slt_ji)*cs;   // up   - up
        sltE_i0[rsj1] =   slt_ij*cc + slt_ji*ss; // up   - down
        sltE_i1[rsj0] = -slt_ij*ss - slt_ji*cc;  // down - up
        sltE_i1[rsj1] =  (slt_ij - slt_ji)*cs;   // down - down 
      }
    }
  }

  return;
}
