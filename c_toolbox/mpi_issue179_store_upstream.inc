/* Verbatim extern/mVMC-1.4.0/src/mVMC/vmccal.c (v1.4.0, lines 669-717):
 * calculateOO_Store_real, through its closing brace before calculateOO_Store.
 * SHA256 c06d5eef360ee3e442d1843ec218f0beb34fc7813e5c6de46768b4d6999055d6.
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
    if(reweight==1){
    for(j=0; j<sampleSize; ++j){
      /* Store holds sqrt(w_j)*O; row 0 (O_0 = 1) gives sqrt(w_j),
         so multiplying once more by it makes <O> w-weighted,
         consistent with the dposv/GEMM paths. With reweight off
         (w_j = 1) the original loop below is kept bit-identical. */
      const double sqrtw = srOptO_Store_real[j*srOptSize];
#pragma omp parallel for default(shared) private(i,o)
#pragma loop noalias
    for(i=0; i<srOptSize; ++i){
      o = srOptO_Store_real[i+j*srOptSize];
      srOptOO_real[i] += sqrtw*o;
      srOptOO_real[i+srOptSize] += o*o;
    }}
    }else{
    for(j=0; j<sampleSize; ++j){
#pragma omp parallel for default(shared) private(i,o)
#pragma loop noalias
    for(i=0; i<srOptSize; ++i){
      o = srOptO_Store_real[i+j*srOptSize];
      srOptOO_real[i] += o;
      srOptOO_real[i+srOptSize] += o*o;
    }}
    }
  }

  return;
}
