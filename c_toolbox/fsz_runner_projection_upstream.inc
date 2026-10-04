/* Generated verbatim from authoritative mVMC-1.3.0: projection.c sha256=2c01cc16ec75f6d5f3c9e7927bd68f96d1db329c124702b06b771ccd314522c1.
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
void MakeProjCnt(int *projCnt, const int *eleNum) {
  const int *n0=eleNum; //up-spin
  const int *n1=eleNum+Nsite; //down-spin
  int idx,offset;
  int ri,rj;
  int xi,xj,xn,xm;
  int r0,r1,r2,r3;
  const int *dh;
  /* optimization for Kei */
  const int nProj=NProj;
  const int nSite=Nsite;

  /* initialization */
  for(idx=0;idx<nProj;idx++) projCnt[idx] = 0;

  /* Gutzwiller factor */
  if(NGutzwillerIdx>0) {
    for(ri=0;ri<nSite;ri++) {
      projCnt[ GutzwillerIdx[ri] ] += n0[ri]*n1[ri];
    }
  }

  /* Jastrow factor exp(sum {v_ij * (ni-1) * (nj-1)}) */
  if(NJastrowIdx>0) {
    offset = NGutzwillerIdx;
    for(ri=0;ri<nSite;ri++) {
      xi = n0[ri]+n1[ri]-1;
      if(xi==0) continue;
      for(rj=ri+1;rj<nSite;rj++) {
        xj = n0[rj]+n1[rj]-1;
        idx = offset + JastrowIdx[ri][rj];
        projCnt[idx] += xi*xj;
      }
    }
  }

  /* 2-site doublon-holon correlation factor */
  if(NDoublonHolon2siteIdx>0) {
    offset = NGutzwillerIdx + NJastrowIdx;
    #pragma omp parallel for default(shared) private(xn,dh,ri,xi,r0,r1,xm,idx)
    for(xn=0;xn<NDoublonHolon2siteIdx;xn++) {
      dh=DoublonHolon2siteIdx[xn];
      for(ri=0;ri<nSite;ri++) {
        xi = n0[ri]+n1[ri];
        if(xi==1) continue;
        xi = xi/2; /* xi=0: holon, xi=1: doublon */

        r0 = dh[2*ri];
        r1 = dh[2*ri+1];
        if(xi==0) {
          xm = n0[r0]*n1[r0] + n0[r1]*n1[r1]; /* count doublons on r0 and r1 */
        } else { /* doublon */
          xm = (1-n0[r0])*(1-n1[r0]) 
            + (1-n0[r1])*(1-n1[r1]); /* count holons on r0 and r1 */
        }

        idx = offset + xn + (xi+2*xm)*NDoublonHolon2siteIdx;
        projCnt[idx] += 1;
      }
    }
  }

  /* 4-site doublon-holon correlation factor */
  if(NDoublonHolon4siteIdx>0) {
    offset = NGutzwillerIdx + NJastrowIdx + 6*NDoublonHolon2siteIdx;
    #pragma omp parallel for default(shared) private(xn,dh,ri,xi,r0,r1,r2,r3,xm,idx)
    for(xn=0;xn<NDoublonHolon4siteIdx;xn++) {
      dh=DoublonHolon4siteIdx[xn];
      for(ri=0;ri<nSite;ri++) {
        xi = n0[ri]+n1[ri];
        if(xi==1) continue;
        xi = xi/2; /* xi=0: holon, xi=1: doublon */

        r0 = dh[4*ri];
        r1 = dh[4*ri+1];
        r2 = dh[4*ri+2];
        r3 = dh[4*ri+3];
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
        projCnt[idx] += 1;
      }
    }
  }

  return;
}
