/* Verbatim extern/mVMC-1.3.0/src/mVMC/vmccal.c:
 * calculateOO_Store_real, through its closing brace before calculateOO_Store.
 * SHA256 c2db5fd32c5c83be189ffd9fbb89684f0696bab7fff5f7d36abaa370274d9d2f.
 * Upstream GPLv3-or-later; see authoritative source for copyright/license.
 */
void calculateOO_Store_real(double *srOptOO_real, double *srOptHO_real, double *srOptO_Store_real,
                 const double w, const double e, int srOptSize, int sampleSize) {

  int i,j;
  char jobz, uplo;
  double alpha,beta,o;
  
  alpha = 1.0;
  beta  = 0.0;
  
  jobz = 'N';
  uplo = 'T';
  if(NSRCG==0){
    M_DGEMM(&jobz,&uplo,&srOptSize,&srOptSize,&sampleSize,&alpha,srOptO_Store_real,&srOptSize,srOptO_Store_real,&srOptSize,&beta,srOptOO_real,&srOptSize);
  }else{
#pragma omp parallel for default(shared) private(i)
#pragma loop noalias
    for(i=0; i<srOptSize; ++i){
      srOptOO_real[i] = 0.0;
      srOptOO_real[i+srOptSize] = 0.0;
    }
    for(j=0; j<sampleSize; ++j){
#pragma omp parallel for default(shared) private(i,o)
#pragma loop noalias
    for(i=0; i<srOptSize; ++i){
      o = srOptO_Store_real[i+j*srOptSize];
      srOptOO_real[i] += o;
      srOptOO_real[i+srOptSize] += o*o;
    }}
  }

  return;
}
