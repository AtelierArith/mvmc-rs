#include "family.h"
/* Included AFTER the pinned vmcmain.h defines native globals/readers. */
static int native13_capture_family(const char *stage,int rank,int group,int seed,int initialized_para) {
    if(NDoublonHolon2siteIdx || NDoublonHolon4siteIdx || NProjBF || NOptTrans ||
       (iFlgOrbitalGeneral && !iNOrbitalParallel))return -20;
    const int rbm_base=NProj;
    const int hidden=rbm_base+NRBM_PhysLayerIdx;
    const int ph=hidden+NRBM_HiddenLayerIdx;
    const int orbital=NProj+FlagRBM*NRBM;
    const Native13Segment segments[]={
        {"Gutz",0,NGutzwillerIdx,iComplexFlgGutzwiller,NATIVE13_GETINFOOPT},
        {"Jast",NGutzwillerIdx,NJastrowIdx,iComplexFlgJastrow,NATIVE13_GETINFOOPT},
        {"ChargeRBM_PhysLayer",rbm_base,NChargeRBM_PhysLayerIdx,iComplexFlgChargeRBM_PhysLayer,NATIVE13_GETINFOOPT},
        {"SpinRBM_PhysLayer",rbm_base+NChargeRBM_PhysLayerIdx,NSpinRBM_PhysLayerIdx,iComplexFlgSpinRBM_PhysLayer,NATIVE13_GETINFOOPT},
        {"GeneralRBM_PhysLayer",rbm_base+NChargeRBM_PhysLayerIdx+NSpinRBM_PhysLayerIdx,NGeneralRBM_PhysLayerIdx,iComplexFlgGeneralRBM_PhysLayer,NATIVE13_GETINFOOPT},
        {"ChargeRBM_HiddenLayer",hidden,NChargeRBM_HiddenLayerIdx,iComplexFlgChargeRBM_HiddenLayer,NATIVE13_GETINFOOPT},
        {"SpinRBM_HiddenLayer",hidden+NChargeRBM_HiddenLayerIdx,NSpinRBM_HiddenLayerIdx,iComplexFlgSpinRBM_HiddenLayer,NATIVE13_GETINFOOPT},
        {"GeneralRBM_HiddenLayer",hidden+NChargeRBM_HiddenLayerIdx+NSpinRBM_HiddenLayerIdx,NGeneralRBM_HiddenLayerIdx,iComplexFlgGeneralRBM_HiddenLayer,NATIVE13_GETINFOOPT},
        {"ChargeRBM_PhysHidden",ph,NChargeRBM_PhysHiddenIdx,iComplexFlgChargeRBM_PhysHidden,NATIVE13_GETINFOOPT},
        {"SpinRBM_PhysHidden",ph+NChargeRBM_PhysHiddenIdx,NSpinRBM_PhysHiddenIdx,iComplexFlgSpinRBM_PhysHidden,NATIVE13_GETINFOOPT},
        {"GeneralRBM_PhysHidden",ph+NChargeRBM_PhysHiddenIdx+NSpinRBM_PhysHiddenIdx,NGeneralRBM_PhysHiddenIdx,iComplexFlgGeneralRBM_PhysHidden,NATIVE13_GETINFOOPT},
        {"OrbitalAP",orbital,iNOrbitalAntiParallel,iComplexFlgOrbital,NATIVE13_GETINFOOPT},
        {"OrbitalParallel",orbital+iNOrbitalAntiParallel,2*iNOrbitalParallel,iComplexFlgOrbital,NATIVE13_PARALLEL}
    };
    const Native13Layout layout={NPara,NProj,NRBM,NSlater,NOptTrans,NQPOptTrans,NProjBF,
        iNOrbitalAntiParallel,iNOrbitalParallel,iFlgOrbitalGeneral,FlagRBM,
        ParaQPOptTrans,segments,(int)(sizeof(segments)/sizeof(*segments))};
    return native13_family_checkpoint(stage,rank,group,seed,AllComplexFlag,
        initialized_para?NPara:0,initialized_para?Para:NULL,initialized_para?OptFlag:NULL,&layout);
}
