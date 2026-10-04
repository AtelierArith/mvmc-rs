/* Generated verbatim from authoritative mVMC-1.3.0: projection.c sha256=2c01cc16ec75f6d5f3c9e7927bd68f96d1db329c124702b06b771ccd314522c1; pfupdate_real.c sha256=61b5977b31be7d1f3301a473b434ac47016472c42b63286577519f297145c111; pfupdate_two_real.c sha256=e30feeb3d1710be4db69a792fbd73408bbb0e95438af0268745acf49556c1efe; qp_real.c sha256=eec104362f5540c368aed7e42403a79eeefc21a509da6c57c668b5c4fe887b71; locgrn_real.c sha256=4fb1e9104ffe5961ab632365b042154134728c8ccc6b5ba51231930930f0de8e.
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
 * Gutzwiller-Jastrow Projection and DH correlation factors
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/
inline double ProjRatio(const int *projCntNew, const int *projCntOld) {
  int idx;
  double z=0;
  for(idx=0;idx<NProj;idx++) {
    z += creal(Proj[idx]) * (double)(projCntNew[idx]-projCntOld[idx]); // TVC
  }
  return exp(z);
}
void UpdateProjCnt(const int ri, const int rj, const int s,
                   int *projCntNew, const int *projCntOld,
                   const int *eleNum) {
  const int *n0=eleNum;
  const int *n1=eleNum+Nsite;
  int idx,offset;
  int rk;
  int xi,xn,xm;
  int r0,r1,r2,r3;
  const int *dh;
  /* optimization for Kei */
  const int nProj=NProj;
  const int nSite=Nsite;

  if(projCntNew!=projCntOld) {
    for(idx=0;idx<nProj;idx++) projCntNew[idx] = projCntOld[idx];
  }
  if(ri==rj) return;

  if(NGutzwillerIdx>0){
    idx = GutzwillerIdx[ri];
    projCntNew[idx] -= n0[ri]+n1[ri];
    idx = GutzwillerIdx[rj];
    projCntNew[idx] += n0[rj]*n1[rj];
  }

  if(NJastrowIdx>0){
    offset = NGutzwillerIdx; 
    /* update [ri][rj] */
    if(ri<rj) idx = offset + JastrowIdx[ri][rj];
    else  idx = offset + JastrowIdx[rj][ri];
    projCntNew[idx] += n0[ri]+n1[ri]-n0[rj]-n1[rj]+1;
    /* update [ri][rk] (rk != ri, rj) */
    for(rk=0;rk<nSite;rk++) {
      if(rk==rj) continue;
      if(rk==ri) continue;
      if(rk>ri) idx = offset + JastrowIdx[ri][rk];
      else idx = offset + JastrowIdx[rk][ri]; 
      projCntNew[idx] -= n0[rk]+n1[rk]-1;
    }
    /* update [rj][rk] (rk != ri, rj) */
    for(rk=0;rk<nSite;rk++) {
      if(rk==ri) continue;
      if(rk==rj) continue;
      if(rk>rj) idx = offset + JastrowIdx[rj][rk];
      else idx = offset + JastrowIdx[rk][rj]; 
      projCntNew[idx] += n0[rk]+n1[rk]-1;
    }
  }

  if(NDoublonHolon2siteIdx==0 && NDoublonHolon4siteIdx==0) return;

  offset = NGutzwillerIdx + NJastrowIdx;
  for(idx=offset;idx<nProj;idx++) projCntNew[idx] = 0;

  /* 2-site doublon-holon correlation factor */
  offset = NGutzwillerIdx + NJastrowIdx;
  #pragma omp parallel for default(shared) private(xn,dh,rk,xi,r0,r1,xm,idx)
  for(xn=0;xn<NDoublonHolon2siteIdx;xn++) {
    dh=DoublonHolon2siteIdx[xn];
    for(rk=0;rk<Nsite;rk++) {
      xi = n0[rk]+n1[rk];
      if(xi==1) continue;
      xi = xi/2; /* xi=0: holon, xi=1: doublon */
      
      r0 = dh[2*rk];
      r1 = dh[2*rk+1];
      if(xi==0) {
        xm = n0[r0]*n1[r0] + n0[r1]*n1[r1]; /* count doublons on r0 and r1 */
      } else { /* doublon */
        xm = (1-n0[r0])*(1-n1[r0]) 
          + (1-n0[r1])*(1-n1[r1]); /* count holons on r0 and r1 */
      }
      
      idx = offset + xn + (xi+2*xm)*NDoublonHolon2siteIdx;
      projCntNew[idx] += 1;
      
    }
  }

  /* 4-site doublon-holon correlation factor */
  offset = NGutzwillerIdx + NJastrowIdx + 6*NDoublonHolon2siteIdx;
  #pragma omp parallel for default(shared) private(xn,dh,rk,xi,r0,r1,r2,r3,xm,idx)
  for(xn=0;xn<NDoublonHolon4siteIdx;xn++) {
    dh=DoublonHolon4siteIdx[xn];
    for(rk=0;rk<Nsite;rk++) {
      xi = n0[rk]+n1[rk];
      if(xi==1) continue;
      xi = xi/2; /* xi=0: holon, xi=1: doublon */
      
      r0 = dh[4*rk];
      r1 = dh[4*rk+1];
      r2 = dh[4*rk+2];
      r3 = dh[4*rk+3];
      if(xi==0) {
        /* count doublons on r0, r1, r2, r3 */
        xm= n0[r0]*n1[r0] + n0[r1]*n1[r1]
          + n0[r2]*n1[r2] + n0[r3]*n1[r3];
      } else { /* doublon */
        /* count holons on r0, r1, r2, r3 */
        xm= (1-n0[r0])*(1-n1[r0]) + (1-n0[r1])*(1-n1[r1])
          + (1-n0[r2])*(1-n1[r2]) + (1-n0[r3])*(1-n1[r3]);
      }
      
      idx = offset + xn + (xi+2*xm)*NDoublonHolon4siteIdx;
      projCntNew[idx] += 1;
    }
  }
  
  return;

}
void CalculateNewPfM_real(const int ma, const int s, double *pfMNew_real, const int *eleIdx,
                     const int qpStart, const int qpEnd) {
  #pragma procedure serial
  const int qpNum = qpEnd-qpStart;
  const int msa = ma+s*Ne;
  const int rsa = eleIdx[msa] + s*Nsite;

  int qpidx;
  int msj,rsj;
  const double *sltE_a; /* update elements of msa-th row */
  const double *invM_a;
  double ratio;

  /* optimization for Kei */
  const int nsize = Nsize;
  const int ne = Ne;

  #pragma loop noalias
  for(qpidx=0;qpidx<qpNum;qpidx++) {
    sltE_a = SlaterElm_real + (qpidx+qpStart)*Nsite2*Nsite2 + rsa*Nsite2;
    invM_a = InvM_real + qpidx*Nsize*Nsize + msa*Nsize;

    ratio = 0.0;
    for(msj=0;msj<ne;msj++) {
      rsj = eleIdx[msj];
      ratio += invM_a[msj] * sltE_a[rsj];
    }
    for(msj=ne;msj<nsize;msj++) {
      rsj = eleIdx[msj] + Nsite;
      ratio += invM_a[msj] * sltE_a[rsj];
    }

    pfMNew_real[qpidx] = -ratio*PfM_real[qpidx];
  }

  return;
}
void CalculateNewPfMTwo_real(const int ma, const int s, const int mb, const int t,
                        double *pfMNew_real, const int *eleIdx,
                        const int qpStart, const int qpEnd, double *buffer) {
  #pragma procedure serial
  const int qpNum = qpEnd-qpStart;
  const int msa = ma+s*Ne;
  const int msb = mb+t*Ne;
  int qpidx;
  double *vec_a = buffer;
  double *vec_b = buffer + Nsize;

  if(msa==msb) {
    CalculateNewPfM_real(mb, t, pfMNew_real, eleIdx, qpStart, qpEnd);
    return;
  }

  for(qpidx=0;qpidx<qpNum;qpidx++) {
    calculateNewPfMTwo_child_real(ma, s, mb, t, pfMNew_real, eleIdx,
                             qpStart, qpEnd, qpidx, vec_a, vec_b);
  }

  return;
}
void calculateNewPfMTwo_child_real(const int ma, const int s, const int mb, const int t,
                              double *pfMNew_real, const int *eleIdx,
                              const int qpStart, const int qpEnd, const int qpidx,
                              double *vec_a, double *vec_b) {
  #pragma procedure serial
  /* const int qpNum = qpEnd-qpStart; */
  const int msa = ma+s*Ne;
  const int msb = mb+t*Ne;
  const int rsa = eleIdx[msa] + s*Nsite;
  const int rsb = eleIdx[msb] + t*Nsite;
  const int nsize = Nsize;

  int msi,msj;
  int rsi;

  double p_a,p_b,q_a,q_b,bMa;
  double ratio,tmp;

  const double *sltE;
  const double *sltE_a; /* update elements of msa-th row */
  const double *sltE_b; /* update elements of msb-th row */

  double *invM;
  const double *invM_a, *invM_b, *invM_i;
  double invM_ab,invM_ai,invM_bi;

  double vec_ba,vec_ai,vec_bi;

  sltE = SlaterElm_real + (qpidx+qpStart)*Nsite2*Nsite2;//TBC
  sltE_a = sltE + rsa*Nsite2;
  sltE_b = sltE + rsb*Nsite2;

  #pragma loop noalias
  for(msi=0;msi<nsize;msi++) {
    rsi = eleIdx[msi] + (msi/Ne)*Nsite;
    vec_a[msi] = sltE_a[rsi];
    vec_b[msi] = sltE_b[rsi];
  }
  vec_ba = vec_b[msa];

  invM = InvM_real + qpidx*Nsize*Nsize; //TBC
  invM_a = invM + msa*Nsize;
  invM_b = invM + msb*Nsize;
  invM_ab = invM_a[msb];

  p_a = p_b = q_a = q_b = bMa = 0.0;

  #pragma loop noalias
  for(msi=0;msi<nsize;msi++) {
    vec_ai = vec_a[msi];
    vec_bi = vec_b[msi];
    invM_ai = invM_a[msi];
    invM_bi = invM_b[msi];

    p_a += invM_ai * vec_ai;
    p_b += invM_bi * vec_ai;
    q_a += invM_ai * vec_bi;
    q_b += invM_bi * vec_bi;
  }

  #pragma loop noalias
  for(msi=0;msi<nsize;msi++) {
    invM_i = invM + msi*Nsize;
    tmp = 0.0;
    for(msj=0;msj<nsize;msj++) {
      tmp += invM_i[msj] * vec_a[msj];
    }
    bMa += vec_b[msi] * tmp;
  }

  /* Calculate ratio PfMNew/PfMOld */
  ratio = invM_ab*vec_ba + invM_ab*bMa + p_a*q_b - p_b*q_a;

  /* Update pfMNew_real */
  pfMNew_real[qpidx] = ratio*PfM_real[qpidx];

  return;
}
double  CalculateIP_real(double* const pfM, const int qpStart, const int qpEnd, MPI_Comm comm) {
  const int qpNum = qpEnd-qpStart;
  double ip=0.0+0.0*I, ip2;
  int qpidx;
  int size;
  MPI_Comm_size(comm,&size);

  #pragma loop noalias
  for(qpidx=0;qpidx<qpNum;qpidx++) {
    ip += creal(QPFullWeight[qpidx+qpStart]) * pfM[qpidx];
    //printf("DEBUG: qpidx =%d: %lf %lf   \n",qpidx,creal( QPFullWeight[qpidx+qpStart] ),creal(pfM[qpidx]));
  }
  if(size>1) {
    MPI_Allreduce(&ip, &ip2, 1, MPI_DOUBLE, MPI_SUM, comm);
    ip = ip2;
  }
  return ip;
}
double  GreenFunc1_real(const int ri, const int rj, const int s, const double ip,
                  int *eleIdx, const int *eleCfg, int *eleNum, const int *eleProjCnt,
                  int *projCntNew, double *buffer) {
  double  z;
  int mj,msj,rsi,rsj;
  double  *pfMNew_real = buffer; /* NQPFull */

  if(ri==rj) return eleNum[ri+s*Nsite];
  if(eleNum[ri+s*Nsite]==1 || eleNum[rj+s*Nsite]==0) return 0.0;

  mj = eleCfg[rj+s*Nsite];
  msj = mj + s*Ne;
  rsi = ri + s*Nsite;
  rsj = rj + s*Nsite;

  /* hopping */
  eleIdx[msj] = ri;
  eleNum[rsj] = 0;
  eleNum[rsi] = 1;
  UpdateProjCnt(rj, ri, s, projCntNew, eleProjCnt, eleNum);
  z = ProjRatio(projCntNew,eleProjCnt);

  /* calculate Pfaffian */
  CalculateNewPfM_real(mj, s, pfMNew_real, eleIdx, 0, NQPFull);
  z *= CalculateIP_real(pfMNew_real, 0, NQPFull, MPI_COMM_SELF);

  /* revert hopping */
  eleIdx[msj] = rj;
  eleNum[rsj] = 1;
  eleNum[rsi] = 0;

  return z/ip;//TBC
}
double GreenFunc2_real(const int ri, const int rj, const int rk, const int rl,
                  const int s, const int t, const double ip,
                  int *eleIdx, const int *eleCfg, int *eleNum, const int *eleProjCnt,
                  int *projCntNew, double *buffer) {
  double z;
  int mj,msj,ml,mtl;
  int rsi,rsj,rtk,rtl;
  double *pfMNew_real = buffer; /* [NQPFull] */
  double *bufV   = buffer+NQPFull; /* 2*Nsize */

  rsi = ri + s*Nsite;
  rsj = rj + s*Nsite;
  rtk = rk + t*Nsite;
  rtl = rl + t*Nsite;

  if(s==t) {
    if(rk==rl) { /* CisAjsNks */
      if(eleNum[rtk]==0) return 0.0;
      else return GreenFunc1_real(ri,rj,s,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,buffer); /* CisAjs */
    }else if(rj==rl) {
      return 0.0; /* CisAjsCksAjs (j!=k) */
    }else if(ri==rl) { /* AjsCksNis */
      if(eleNum[rsi]==0) return 0.0;
      else if(rj==rk) return 1.0-eleNum[rsj];
      else return -GreenFunc1_real(rk,rj,s,ip,eleIdx,eleCfg,eleNum,
                              eleProjCnt,projCntNew,buffer); /* -CksAjs */
    }else if(rj==rk) { /* CisAls(1-Njs) */
      if(eleNum[rsj]==1) return 0.0;
      else if(ri==rl) return eleNum[rsi];
      else return GreenFunc1_real(ri,rl,s,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,buffer); /* CisAls */
    }else if(ri==rk) {
      return 0.0; /* CisAjsCisAls (i!=j) */
    }else if(ri==rj) { /* NisCksAls (i!=k,l) */
      if(eleNum[rsi]==0) return 0.0;
      else return GreenFunc1_real(rk,rl,s,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,buffer); /* CksAls */
    }
  }else{
    if(rk==rl) { /* CisAjsNkt */
      if(eleNum[rtk]==0) return 0.0;
      else if(ri==rj) return eleNum[rsi];
      else return GreenFunc1_real(ri,rj,s,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,buffer); /* CisAjs */
    }else if(ri==rj) { /* NisCktAlt */
      if(eleNum[rsi]==0) return 0.0;
      else return GreenFunc1_real(rk,rl,t,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,buffer); /* CktAlt */
    }
  }

  if(eleNum[rsi]==1 || eleNum[rsj]==0 || eleNum[rtk]==1 || eleNum[rtl]==0) return 0.0;

  mj = eleCfg[rj+s*Nsite];
  ml = eleCfg[rl+t*Nsite];
  msj = mj + s*Ne;
  mtl = ml + t*Ne;

  /* hopping */
  eleIdx[mtl] = rk;
  eleNum[rtl] = 0;
  eleNum[rtk] = 1;
  UpdateProjCnt(rl, rk, t, projCntNew, eleProjCnt, eleNum);
  eleIdx[msj] = ri;
  eleNum[rsj] = 0;
  eleNum[rsi] = 1;
  UpdateProjCnt(rj, ri, s, projCntNew, projCntNew, eleNum);

  z = ProjRatio(projCntNew,eleProjCnt);

  /* calculate Pfaffian */
  CalculateNewPfMTwo_real(ml, t, mj, s, pfMNew_real, eleIdx, 0, NQPFull, bufV);
  z *= CalculateIP_real(pfMNew_real, 0, NQPFull, MPI_COMM_SELF);

  /* revert hopping */
  eleIdx[mtl] = rl;
  eleNum[rtl] = 1;
  eleNum[rtk] = 0;
  eleIdx[msj] = rj;
  eleNum[rsj] = 1;
  eleNum[rsi] = 0;

  return z/ip;//TBC
}
