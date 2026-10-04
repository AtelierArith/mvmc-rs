/* Generated verbatim from authoritative mVMC-1.3.0: projection.c sha256=2c01cc16ec75f6d5f3c9e7927bd68f96d1db329c124702b06b771ccd314522c1; pfupdate.c sha256=78013065bbadfd6cfc6f6bc8995c14984b173bdc97df4a9db0cc8ebeccc68f79; pfupdate_two_fcmp.c sha256=2fdcb8227bcd4ff52ce6bdc103a028397cddbd1eca592f5373bb3998f4091a4e; qp.c sha256=ab53f8d929130bee4dbc88ba1b834d182eb65a6d70a3e5d2c51894020faf8e6f; locgrn.c sha256=14cdec8555efda3d1132969dc31696fcda6021b03dc91fd8b83b1c1d91eeb8b6.
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
void CalculateNewPfM(const int ma, const int s, double complex *pfMNew, const int *eleIdx,
                     const int qpStart, const int qpEnd) {
  #pragma procedure serial
  const int qpNum = qpEnd-qpStart;
  const int msa = ma+s*Ne;
  const int rsa = eleIdx[msa] + s*Nsite;

  int qpidx;
  int msj,rsj;
  const double complex *sltE_a; /* update elements of msa-th row */
  const double complex *invM_a;
  double complex ratio;

  /* optimization for Kei */
  const int nsize = Nsize;
  const int ne = Ne;

  #pragma loop noalias
  for(qpidx=0;qpidx<qpNum;qpidx++) {
    sltE_a = SlaterElm + (qpidx+qpStart)*Nsite2*Nsite2 + rsa*Nsite2;
    invM_a = InvM + qpidx*Nsize*Nsize + msa*Nsize;

    ratio = 0.0;
    for(msj=0;msj<ne;msj++) {
      rsj = eleIdx[msj];
      ratio += invM_a[msj] * sltE_a[rsj];
    }
    for(msj=ne;msj<nsize;msj++) {
      rsj = eleIdx[msj] + Nsite;
      ratio += invM_a[msj] * sltE_a[rsj];
    }

    pfMNew[qpidx] = -ratio*PfM[qpidx];
  }

  return;
}
void CalculateNewPfMTwo_fcmp(const int ma, const int s, const int mb, const int t,
                        double complex *pfMNew, const int *eleIdx,
                        const int qpStart, const int qpEnd, double complex *buffer) {
  #pragma procedure serial
  const int qpNum = qpEnd-qpStart;
  const int msa = ma+s*Ne;
  const int msb = mb+t*Ne;
  int qpidx;
  double complex *vec_a = buffer;
  double complex *vec_b = buffer + Nsize;

  if(msa==msb) {
    CalculateNewPfM(mb, t, pfMNew, eleIdx, qpStart, qpEnd);
    return;
  }

  for(qpidx=0;qpidx<qpNum;qpidx++) {
    calculateNewPfMTwo_child_fcmp(ma, s, mb, t, pfMNew, eleIdx,
                             qpStart, qpEnd, qpidx, vec_a, vec_b);
  }

  return;
}
void calculateNewPfMTwo_child_fcmp(const int ma, const int s, const int mb, const int t,
                              double complex *pfMNew, const int *eleIdx,
                              const int qpStart, const int qpEnd, const int qpidx,
                              double complex *vec_a, double complex *vec_b) {
  #pragma procedure serial
  /* const int qpNum = qpEnd-qpStart; */
  const int msa = ma+s*Ne;
  const int msb = mb+t*Ne;
  const int rsa = eleIdx[msa] + s*Nsite;
  const int rsb = eleIdx[msb] + t*Nsite;
  const int nsize = Nsize;

  int msi,msj;
  int rsi;

  double complex p_a,p_b,q_a,q_b,bMa;
  double complex ratio,tmp;

  const double complex *sltE;
  const double complex *sltE_a; /* update elements of msa-th row */
  const double complex *sltE_b; /* update elements of msb-th row */

  double complex *invM;
  const double complex *invM_a, *invM_b, *invM_i;
  double complex invM_ab,invM_ai,invM_bi;

  double complex vec_ba,vec_ai,vec_bi;

  sltE = SlaterElm + (qpidx+qpStart)*Nsite2*Nsite2;
  sltE_a = sltE + rsa*Nsite2;
  sltE_b = sltE + rsb*Nsite2;

  #pragma loop noalias
  for(msi=0;msi<nsize;msi++) {
    rsi = eleIdx[msi] + (msi/Ne)*Nsite;
    vec_a[msi] = sltE_a[rsi];
    vec_b[msi] = sltE_b[rsi];
  }
  vec_ba = vec_b[msa];

  invM = InvM + qpidx*Nsize*Nsize;
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

  /* Update pfMNew */
  pfMNew[qpidx] = ratio*PfM[qpidx];

  return;
}
double complex CalculateIP_fcmp(double complex * const pfM, const int qpStart, const int qpEnd, MPI_Comm comm) {
  const int qpNum = qpEnd-qpStart;
  double complex ip=0.0+0.0*I, ip2;
  int qpidx;
  int size;
  MPI_Comm_size(comm,&size);

  #pragma loop noalias
  for(qpidx=0;qpidx<qpNum;qpidx++) {
    ip += QPFullWeight[qpidx+qpStart] * pfM[qpidx];
    //printf("DEBUG: qpidx =%d: %lf %lf   \n",qpidx,creal( QPFullWeight[qpidx+qpStart] ),creal(pfM[qpidx]));
  }
  if(size>1) {
    MPI_Allreduce(&ip, &ip2, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    ip = ip2;
  }
  return ip;
}
double complex GreenFunc1(const int ri, const int rj, const int s, const double complex  ip,
                  int *eleIdx, const int *eleCfg, int *eleNum, const int *eleProjCnt,
                  int *projCntNew, const double complex *rbmCnt, double complex *rbmCntNew, double complex *buffer) {
  double complex z;
  int mj,msj,rsi,rsj;
  double complex *pfMNew = buffer; /* NQPFull */

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

  if (FlagRBM) {
  UpdateRBMCnt(rj, ri, s, rbmCntNew, rbmCnt, eleNum);
  z = ProjRatio(projCntNew,eleProjCnt);
  z *= RBMRatio(rbmCntNew,rbmCnt);
  }
  else {
  UpdateProjCnt(rj, ri, s, projCntNew, eleProjCnt, eleNum);
  z = ProjRatio(projCntNew,eleProjCnt);
  }

  /* calculate Pfaffian */
  CalculateNewPfM(mj, s, pfMNew, eleIdx, 0, NQPFull);
  z *= CalculateIP_fcmp(pfMNew, 0, NQPFull, MPI_COMM_SELF);

  /* revert hopping */
  eleIdx[msj] = rj;
  eleNum[rsj] = 1;
  eleNum[rsi] = 0;

  return conj(z/ip);//TBC
}
double complex GreenFunc2(const int ri, const int rj, const int rk, const int rl,
                  const int s, const int t, const double complex ip,
                  int *eleIdx, const int *eleCfg, int *eleNum, const int *eleProjCnt,
                  int *projCntNew, const double complex *rbmCnt, double complex *rbmCntNew, double complex *buffer) {
  double complex z;
  int mj,msj,ml,mtl;
  int rsi,rsj,rtk,rtl;
  double complex *pfMNew = buffer; /* [NQPFull] */
  double complex *bufV   = buffer+NQPFull; /* 2*Nsize */

  rsi = ri + s*Nsite;
  rsj = rj + s*Nsite;
  rtk = rk + t*Nsite;
  rtl = rl + t*Nsite;

  if(s==t) {
    if(rk==rl) { /* CisAjsNks */
      if(eleNum[rtk]==0) return 0.0;
      else return GreenFunc1(ri,rj,s,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,rbmCnt,rbmCntNew,buffer); /* CisAjs */
    }else if(rj==rl) {
      return 0.0; /* CisAjsCksAjs (j!=k) */
    }else if(ri==rl) { /* AjsCksNis */
      if(eleNum[rsi]==0) return 0.0;
      else if(rj==rk) return 1.0-eleNum[rsj];
      else return -GreenFunc1(rk,rj,s,ip,eleIdx,eleCfg,eleNum,
                              eleProjCnt,projCntNew,rbmCnt,rbmCntNew,buffer); /* -CksAjs */
    }else if(rj==rk) { /* CisAls(1-Njs) */
      if(eleNum[rsj]==1) return 0.0;
      else if(ri==rl) return eleNum[rsi];
      else return GreenFunc1(ri,rl,s,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,rbmCnt,rbmCntNew,buffer); /* CisAls */
    }else if(ri==rk) {
      return 0.0; /* CisAjsCisAls (i!=j) */
    }else if(ri==rj) { /* NisCksAls (i!=k,l) */
      if(eleNum[rsi]==0) return 0.0;
      else return GreenFunc1(rk,rl,s,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,rbmCnt,rbmCntNew,buffer); /* CksAls */
    }
  }else{
    if(rk==rl) { /* CisAjsNkt */
      if(eleNum[rtk]==0) return 0.0;
      else if(ri==rj) return eleNum[rsi];
      else return GreenFunc1(ri,rj,s,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,rbmCnt,rbmCntNew,buffer); /* CisAjs */
    }else if(ri==rj) { /* NisCktAlt */
      if(eleNum[rsi]==0) return 0.0;
      else return GreenFunc1(rk,rl,t,ip,eleIdx,eleCfg,eleNum,
                             eleProjCnt,projCntNew,rbmCnt,rbmCntNew,buffer); /* CktAlt */
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
  if (FlagRBM) {
    UpdateRBMCnt(rl, rk, t, rbmCntNew, rbmCnt, eleNum);
  }
  eleIdx[msj] = ri;
  eleNum[rsj] = 0;
  eleNum[rsi] = 1;
  UpdateProjCnt(rj, ri, s, projCntNew, projCntNew, eleNum);
  if (FlagRBM) {
    UpdateRBMCnt(rj, ri, s, rbmCntNew, rbmCntNew, eleNum);
  }

  z = ProjRatio(projCntNew,eleProjCnt);
  if (FlagRBM) {
    z *= RBMRatio(rbmCntNew,rbmCnt);
  }

  /* calculate Pfaffian */
  CalculateNewPfMTwo_fcmp(ml, t, mj, s, pfMNew, eleIdx, 0, NQPFull, bufV);
  z *= CalculateIP_fcmp(pfMNew, 0, NQPFull, MPI_COMM_SELF);

  /* revert hopping */
  eleIdx[mtl] = rl;
  eleNum[rtl] = 1;
  eleNum[rtk] = 0;
  eleIdx[msj] = rj;
  eleNum[rsj] = 1;
  eleNum[rsi] = 0;

  return conj(z/ip);//TBC
}
