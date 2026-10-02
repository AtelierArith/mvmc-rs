/* Generated verbatim from authoritative mVMC-1.3.0: rbm.c sha256=e35af053ce06e127309b1636dd2776650b4f1b2abd1ea0eaca10cd5257c3d742.
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
 * RBM factors
 *-------------------------------------------------------------*/
void MakeRBMCnt(double complex *rbmCnt, const int *eleNum) {
  const int *n=eleNum;
  const int *n0=eleNum; //up-spin
  const int *n1=eleNum+Nsite; //down-spin
  int idx;
  int ri,hi,hidx,xi;
  double complex ctmp;
  /* optimization for Kei */
  const int nRBM=NRBM_PhysLayerIdx + Nneuron;
  //const int nRBM=NRBM;
  const int nSite=Nsite;
  const int nSite2=Nsite2;
  const int nNeuronGeneral=NneuronGeneral;
  const int nNeuronCharge=NneuronCharge;
  const int nNeuronSpin=NneuronSpin;
  const double complex *RBM_Hidden     = RBM+NRBM_PhysLayerIdx;
  const double complex *RBM_PhysHidden = RBM+NRBM_PhysLayerIdx+NRBM_HiddenLayerIdx;
  int offset,offset2;

  /* initialization */
  for(idx=0;idx<nRBM;idx++) rbmCnt[idx] = 0.0;

  /* Potential on Physical Layer */
  if(NChargeRBM_PhysLayerIdx>0) {
    for(ri=0;ri<nSite;ri++) {
      rbmCnt[ ChargeRBM_PhysLayerIdx[ri] ] += n0[ri]+n1[ri]-1;
    }
  }
  if(NSpinRBM_PhysLayerIdx>0) {
    offset = NChargeRBM_PhysLayerIdx;
    for(ri=0;ri<nSite;ri++) {
      rbmCnt[ SpinRBM_PhysLayerIdx[ri] + offset] += n0[ri]-n1[ri];
    }
  }
  if(NGeneralRBM_PhysLayerIdx>0) {
    offset = NChargeRBM_PhysLayerIdx + NSpinRBM_PhysLayerIdx;
    for(ri=0;ri<nSite2;ri++) {
      rbmCnt[ GeneralRBM_PhysLayerIdx[ri] + offset] += 2*n0[ri]-1;
    }
  }

  /* Potential on Hidden Layer */
  if(NChargeRBM_HiddenLayerIdx>0) {
    for(hi=0;hi<nNeuronCharge;hi++) {
      hidx = ChargeRBM_HiddenLayerIdx[hi];
      rbmCnt[hi+NRBM_PhysLayerIdx] += RBM_Hidden[hidx];
    }
  }
  if(NSpinRBM_HiddenLayerIdx>0) {
    for(hi=0;hi<nNeuronSpin;hi++) {
      hidx = SpinRBM_HiddenLayerIdx[hi];
      rbmCnt[hi+NRBM_PhysLayerIdx + NneuronCharge] += RBM_Hidden[hidx + NChargeRBM_HiddenLayerIdx];
    }
  }
  if(NGeneralRBM_HiddenLayerIdx>0) {
    offset  = NRBM_PhysLayerIdx + NneuronCharge + NneuronSpin;
    offset2 = NChargeRBM_HiddenLayerIdx + NSpinRBM_HiddenLayerIdx;
    for(hi=0;hi<nNeuronGeneral;hi++) {
      hidx = GeneralRBM_HiddenLayerIdx[hi];
      rbmCnt[hi+offset] += RBM_Hidden[hidx + offset2];
    }
  }

  /* Coupling between Phys-Hidden Layers */
  if(NChargeRBM_PhysHiddenIdx>0) {
    for(hi=0;hi<nNeuronCharge;hi++) {
      ctmp = 0.0;
      for(ri=0;ri<nSite;ri++) {
        xi = n0[ri]+n1[ri]-1;
        hidx = ChargeRBM_PhysHiddenIdx[ri][hi];
        ctmp += RBM_PhysHidden[hidx]*xi;
      }
      rbmCnt[hi+NRBM_PhysLayerIdx] += ctmp;
    }
  }
  if(NSpinRBM_PhysHiddenIdx>0) {
    for(hi=0;hi<nNeuronSpin;hi++) {
      ctmp = 0.0;
      for(ri=0;ri<nSite;ri++) {
        xi = n0[ri]-n1[ri];
        hidx = SpinRBM_PhysHiddenIdx[ri][hi];
        ctmp += RBM_PhysHidden[hidx + NChargeRBM_PhysHiddenIdx]*xi;
      }
      rbmCnt[hi+NRBM_PhysLayerIdx + NneuronCharge] += ctmp;
    }
  }
  if(NGeneralRBM_PhysHiddenIdx>0) {
    offset  = NRBM_PhysLayerIdx + NneuronCharge + NneuronSpin;
    offset2 = NChargeRBM_PhysHiddenIdx+NSpinRBM_PhysHiddenIdx;
    for(hi=0;hi<nNeuronGeneral;hi++) {
      ctmp = 0.0;
      for(ri=0;ri<nSite2;ri++) {
        xi = 2*n[ri]-1;
        hidx = GeneralRBM_PhysHiddenIdx[ri][hi];
        ctmp += RBM_PhysHidden[hidx + offset2]*xi;
      }
      rbmCnt[hi+offset] += ctmp;
    }
  }
 
  return;
}
void UpdateRBMCnt(const int ri, const int rj, const int s,
                   double complex *rbmCntNew, const double complex *rbmCntOld, const int *eleNum) {
//  const int *n0=eleNum; //up-spin
//  const int *n1=eleNum+Nsite; //down-spin
  int idx,offset;
  int hi,hidx,xi;
  double complex ctmp;
  /* optimization for Kei */
  const int nRBM=NRBM_PhysLayerIdx + Nneuron;
  const int nSite=Nsite;
  const int nSite2=Nsite2;
  const int nNeuronGeneral=NneuronGeneral;
  const int nNeuronCharge=NneuronCharge;
  const int nNeuronSpin=NneuronSpin;
  const double complex *RBM_PhysHidden = RBM+NRBM_PhysLayerIdx+NRBM_HiddenLayerIdx;
  const int rsi=ri+s*Nsite;
  const int rsj=rj+s*Nsite;
  const int t = 1-s;
  const int nit = eleNum[ri + t*Nsite];
  const int njt = eleNum[rj + t*Nsite];
  
  if(rbmCntNew != rbmCntOld) {
    for(idx=0;idx<nRBM;idx++) rbmCntNew[idx] = rbmCntOld[idx];
  }
  if(ri==rj) return;

  /* Potential on Physical Layer */
  if(NChargeRBM_PhysLayerIdx>0) {
    rbmCntNew[ ChargeRBM_PhysLayerIdx[ri] ] += -1;
    rbmCntNew[ ChargeRBM_PhysLayerIdx[rj] ] +=  1;
  }
  if(NSpinRBM_PhysLayerIdx>0) {
    rbmCntNew[ SpinRBM_PhysLayerIdx[ri] + NChargeRBM_PhysLayerIdx ] += 2*s-1;
    rbmCntNew[ SpinRBM_PhysLayerIdx[rj] + NChargeRBM_PhysLayerIdx ] += 1-2*s;
  }
  if(NGeneralRBM_PhysLayerIdx>0) {
    rbmCntNew[ GeneralRBM_PhysLayerIdx[rsi] + NChargeRBM_PhysLayerIdx + NSpinRBM_PhysLayerIdx] += -2;
    rbmCntNew[ GeneralRBM_PhysLayerIdx[rsj] + NChargeRBM_PhysLayerIdx + NSpinRBM_PhysLayerIdx] +=  2;
  }

  /* Potential on Hidden Layer */
  // No Change

  /* Coupling between Phys-Hidden Layers */
  if(NChargeRBM_PhysHiddenIdx>0) {
    for(hi=0;hi<nNeuronCharge;hi++) {
      hidx = ChargeRBM_PhysHiddenIdx[ri][hi];
      rbmCntNew[hi+NRBM_PhysLayerIdx] -= RBM_PhysHidden[hidx];
    }
    
    for(hi=0;hi<nNeuronCharge;hi++) {
      hidx = ChargeRBM_PhysHiddenIdx[rj][hi];
      rbmCntNew[hi+NRBM_PhysLayerIdx] += RBM_PhysHidden[hidx];
    }
  }
  if(NSpinRBM_PhysHiddenIdx>0) {
    for(hi=0;hi<nNeuronSpin;hi++) {
      hidx = SpinRBM_PhysHiddenIdx[ri][hi];
      rbmCntNew[hi+NRBM_PhysLayerIdx+nNeuronCharge] -= (1-2*s)*RBM_PhysHidden[hidx+NChargeRBM_PhysHiddenIdx];
    }
    
    for(hi=0;hi<nNeuronSpin;hi++) {
      hidx = SpinRBM_PhysHiddenIdx[rj][hi];
      rbmCntNew[hi+NRBM_PhysLayerIdx+nNeuronCharge] += (1-2*s)*RBM_PhysHidden[hidx+NChargeRBM_PhysHiddenIdx];
    }
  }
  if(NGeneralRBM_PhysHiddenIdx>0) {
    for(hi=0;hi<nNeuronGeneral;hi++) {
      hidx = GeneralRBM_PhysHiddenIdx[rsi][hi];
      rbmCntNew[hi+NRBM_PhysLayerIdx+nNeuronCharge+nNeuronSpin] -= 2*RBM_PhysHidden[hidx+NChargeRBM_PhysHiddenIdx+NSpinRBM_PhysHiddenIdx];
    }
    
    for(hi=0;hi<nNeuronGeneral;hi++) {
      hidx = GeneralRBM_PhysHiddenIdx[rsj][hi];
      rbmCntNew[hi+NRBM_PhysLayerIdx+nNeuronCharge+nNeuronSpin] += 2*RBM_PhysHidden[hidx+NChargeRBM_PhysHiddenIdx+NSpinRBM_PhysHiddenIdx];
    }
  }



  return;
}
