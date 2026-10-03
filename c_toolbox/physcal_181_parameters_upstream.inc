/* Verbatim parameter.c InitParameter/ReadInitParameter/SyncModifiedParameter; readdef.c ReadInputParameters.
 * parameter_sha256=46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0
 * readdef_sha256=6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9 */
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

/* read initial vaules of variational parameters from initFile */
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

/* sync and modify variational parameters */
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

/* shift Gutzwiller-Jastrow factor */
int ReadInputParameters(char *xNameListFile, MPI_Comm comm) {
  FILE *fp;
  char defname[D_FileNameMax];
  char ctmp[D_FileNameMax], ctmp2[256];
  int iKWidx = 0;
  int i, idx;
  int rank;
  int count = 0;
  int info = 0;
  int iNTotalIdx;
  double tmp_real, tmp_comp;
  MPI_Comm_rank(comm, &rank);
  char *cerr;
  if (rank == 0) {
    for (iKWidx = KWInGutzwiller; iKWidx < KWIdxInt_end; iKWidx++) {
      strcpy(defname, cFileNameListFile[iKWidx]);
      if (strcmp(defname, "") == 0) continue;
      fp = fopen(defname, "r");
      if (fp == NULL) {
        info = ReadDefFileError(defname);
        fclose(fp);
        continue;
      }
      /*=======================================================================*/
      cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp);//1
      if(cerr == NULL) return -1;
      cerr = fgets(ctmp2, sizeof(ctmp2) / sizeof(char), fp);//2
      if(cerr == NULL) return -1;
      sscanf(ctmp2, "%s %d\n", ctmp, &idx);
      cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp);//3
      if(cerr == NULL) return -1;
      cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp);//4
      if(cerr == NULL) return -1;
      cerr = fgets(ctmp, sizeof(ctmp) / sizeof(char), fp);//5
      if(cerr == NULL) return -1;

      switch (iKWidx) {
        //get idx

        case KWInGutzwiller:
          fprintf(stdout, "Read InGutzwiller File. \n");
          if (idx != NGutzwillerIdx) {
            info = 1;
            continue;
          }
          count = 0;
          for (i = 0; i < NGutzwillerIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            Proj[idx+count] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInJastrow:
          fprintf(stdout, "Read InJastrow File. \n");
          if (idx != NJastrowIdx) {
            info = 1;
            continue;
          }
          count = NGutzwillerIdx;
          for (i = count; i < count + NJastrowIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            Proj[idx+count] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInDH2:
          fprintf(stdout, "Read InDH2 File. \n");
          if (idx != NDoublonHolon2siteIdx) {
            info = 1;
            continue;
          }
          count = NGutzwillerIdx + NJastrowIdx;
          for (i = count; i < count + 2 * 3 * NDoublonHolon2siteIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            Proj[idx+count] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInDH4:
          fprintf(stdout, "Read InDH4 File. \n");
          if (idx != NDoublonHolon4siteIdx) {
            info = 1;
            continue;
          }
          count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          for (i = count; i < count + 2 * 5 * NDoublonHolon4siteIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            Proj[idx+count] = tmp_real + I * tmp_comp;
          }
          break;

//RBM
        case KWInChargeRBM_PhysLayer:
          fprintf(stdout, "Read InChargeRBM_PhysLayer File. \n");
          if (idx != NChargeRBM_PhysLayerIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = 0;
          for (i = count; i < count + NChargeRBM_PhysLayerIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;


        case KWInSpinRBM_PhysLayer:
          fprintf(stdout, "Read InSpinRBM_PhysLayer File. \n");
          if (idx != NSpinRBM_PhysLayerIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = NChargeRBM_PhysLayerIdx;
          for (i = count; i < count + NSpinRBM_PhysLayerIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInGeneralRBM_PhysLayer:
          fprintf(stdout, "Read InGeneralRBM_PhysLayer File. \n");
          if (idx != NGeneralRBM_PhysLayerIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = NChargeRBM_PhysLayerIdx + NSpinRBM_PhysLayerIdx;
          for (i = count; i < count + NGeneralRBM_PhysLayerIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;



        case KWInChargeRBM_HiddenLayer:
          fprintf(stdout, "Read InChargeRBM_HiddenLayer File. \n");
          if (idx != NChargeRBM_HiddenLayerIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = NRBM_PhysLayerIdx;
          for (i = count; i < count + NChargeRBM_HiddenLayerIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInSpinRBM_HiddenLayer:
          fprintf(stdout, "Read InSpinRBM_HiddenLayer File. \n");
          if (idx != NSpinRBM_HiddenLayerIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = NRBM_PhysLayerIdx+NChargeRBM_HiddenLayerIdx;
          for (i = count; i < count + NSpinRBM_HiddenLayerIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInGeneralRBM_HiddenLayer:
          fprintf(stdout, "Read InGeneralRBM_HiddenLayer File. \n");
          if (idx != NGeneralRBM_HiddenLayerIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = NRBM_PhysLayerIdx+NChargeRBM_HiddenLayerIdx+NSpinRBM_HiddenLayerIdx;
          for (i = count; i < count + NGeneralRBM_HiddenLayerIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;


        case KWInChargeRBM_PhysHidden:
          fprintf(stdout, "Read InChargeRBM_PhysHidden File. \n");
          if (idx != NChargeRBM_PhysHiddenIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = NRBM_PhysLayerIdx + NRBM_HiddenLayerIdx;
          for (i = count; i < count + NChargeRBM_PhysHiddenIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInSpinRBM_PhysHidden:
          fprintf(stdout, "Read InSpinRBM_PhysHidden File. \n");
          if (idx != NSpinRBM_PhysHiddenIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = NRBM_PhysLayerIdx + NRBM_HiddenLayerIdx + NChargeRBM_PhysHiddenIdx;
          for (i = count; i < count + NSpinRBM_PhysHiddenIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInGeneralRBM_PhysHidden:
          fprintf(stdout, "Read InGeneralRBM_PhysHidden File. \n");
          if (idx != NGeneralRBM_PhysHiddenIdx) {
            info = 1;
            continue;
          }
          //count = NGutzwillerIdx + NJastrowIdx + 2 * 3 * NDoublonHolon2siteIdx;
          count = NRBM_PhysLayerIdx + NRBM_HiddenLayerIdx + NChargeRBM_PhysHiddenIdx + NSpinRBM_PhysHiddenIdx;
          for (i = count; i < count + NGeneralRBM_PhysHiddenIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            RBM[idx + count] = tmp_real + I * tmp_comp;
          }
          break;
//RBM

        case KWInOrbital:
        case KWInOrbitalAntiParallel:
          iNTotalIdx = (iFlgOrbitalGeneral > 0) ? iNOrbitalAntiParallel : NOrbitalIdx;
          fprintf(stdout, "Read InOrbital/InOrbitalAntiParallel File. \n");
          if (idx != iNTotalIdx) {
            info = 1;
            continue;
          }
          for (i = 0; i < iNTotalIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            Slater[idx] = tmp_real + I * tmp_comp;
          }
          break;

        case KWInOrbitalParallel:
          fprintf(stdout, "Read iNOrbitalParallel File. \n");
          //printf("MDEBUG: %d %d \n",idx,iNOrbitalParallel);
          if ((idx / 2) != iNOrbitalParallel) {
            info = 1;
            continue;
          }
          for (i = 0; i < 2 * iNOrbitalParallel; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            Slater[iNOrbitalAntiParallel + idx] = tmp_real + I * tmp_comp; //up-up
            //printf("MDEBUG: %d %d %d %lf\n",i,idx,iNOrbitalParallel,tmp_real,tmp_comp);
            //ierr = fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            //Slater[iNOrbitalAntiParallel+idx+1] = tmp_real + I * tmp_comp;//down-down
          }
          break;

        case KWInorbitalGeneral:
          fprintf(stdout, "Read InOrbitalGeneral File. \n");
          if (idx != NOrbitalIdx) {
            info = 1;
            continue;
          }
          for (i = 0; i < NOrbitalIdx; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            Slater[idx] = tmp_real + I * tmp_comp; //up-up
          }
          break;

        case KWInOptTrans:
          fprintf(stdout, "Read InOptTrans File. \n");
          if (idx != NOptTrans) {
            info = 1;
            continue;
          }
          for (i = 0; i < NOptTrans; i++) {
            fscanf(fp, "%d %lf %lf ", &idx, &tmp_real, &tmp_comp);
            OptTrans[idx] = tmp_real + I * tmp_comp;
          }
          break;

        default:
          info = 0;
          break;
      }
      if (info != 0) {
        fprintf(stderr, "error: Indices of %s file is incomplete.\n", defname);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      fclose(fp);
    }
  } /* if(rank==0) */

  if (info != 0) {
    if (rank == 0) {
      fprintf(stderr, "error: Indices of %s file is incomplete.\n", defname);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  return info;
}



