/* Verbatim lines from authoritative mVMC-1.3.0 (GPL-3.0-or-later, Copyright (C) 2016 The University of Tokyo).
 * Regenerate with scripts/check_time_srinfo_c_parity.py --write. Optional oracle, not a Rust build/test dependency. */
/* initfile.c:33-70 sha256=8a7b20d54ab495cfac4df42646a30102aebaa228311ca3b17c1e29af5d0581f8 */
void InitFile(char *xNameListFile, int rank) {
  char fileName[D_FileNameMax];

  if(rank!=0) return;

  //sprintf(fileName, "%s_cfg_%03d.dat", CDataFileHead, NDataIdxStart);
  //writeConfig(xNameListFile, fileName);

  sprintf(fileName, "%s_time_%03d.dat", CDataFileHead, NDataIdxStart);
  FileTime = fopen(fileName, "w");

  if(NVMCCalMode==0) {
    sprintf(fileName, "%s_SRinfo.dat", CDataFileHead);
    FileSRinfo = fopen(fileName, "w");
    if(SRFlag == 0){
      fprintf(FileSRinfo,
            "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax\n");
    }else{
      fprintf(FileSRinfo,
            "#Npara Msize optCut diagCut sEigenMax  sEigenMin    absRmax       imax\n");
    }

    sprintf(fileName, "%s_out_%03d.dat", CDataFileHead, NDataIdxStart);
    FileOut = fopen(fileName, "w");

    if(FlagBinary==0) {
      sprintf(fileName, "%s_var_%03d.dat", CDataFileHead, NDataIdxStart);
      FileVar = fopen(fileName, "w");
    } else {
      sprintf(fileName, "%s_varbin_%03d.dat", CDataFileHead, NDataIdxStart);
      FileVar = fopen(fileName, "wb");
      fwrite(&NPara,sizeof(int),1,FileVar);
      fwrite(&NSROptItrStep,sizeof(int),1,FileVar);
    }
  }

  return;
}
/* initfile.c:144-156 sha256=8a7b20d54ab495cfac4df42646a30102aebaa228311ca3b17c1e29af5d0581f8 */
void CloseFile(int rank) {
  if(rank!=0) return;

  fclose(FileTime);

  if(NVMCCalMode==0) {
    fclose(FileSRinfo);
    fclose(FileOut);
    fclose(FileVar);
  }

  return;
}
/* initfile.c:192-204 sha256=8a7b20d54ab495cfac4df42646a30102aebaa228311ca3b17c1e29af5d0581f8 */
void FlushFile(int step, int rank) {
  if(rank!=0) return;

  if(step%NFileFlushInterval==0) {
    fflush(FileTime);
    if(NVMCCalMode==0) {
      fflush(FileSRinfo);
      fflush(FileOut);
      fflush(FileVar);
    }
  }
  return;
}
/* vmcclock.c:34-48 sha256=38fe6b4080df1bada7c1e8c4fa200f902d18eecef6b976e8e1cec486247df99c */
void OutputTime(int step) {
  time_t tx;
  double pHop,pEx,pLSF;

  tx = time(NULL);
  if(step==0) {
    fprintf(FileTime, "%05d  acc_hop acc_ex  acc_lsf n_hop    n_ex      n_lsf   : %s", step, ctime(&tx));
  } else {
    pHop = (Counter[0] == 0) ? 0.0 : (double)Counter[1] / (double)Counter[0];
    pEx  = (Counter[2] == 0) ? 0.0 : (double)Counter[3] / (double)Counter[2];
    pLSF = (Counter[4] == 0) ? 0.0 : (double)Counter[5] / (double)Counter[4];
    fprintf(FileTime, "%05d  %.5lf %.5lf %.5lf %-8d %-8d  %-8d: %s", step, pHop,pEx,pLSF,
            Counter[0], Counter[2],Counter[4], ctime(&tx));
  }
}
