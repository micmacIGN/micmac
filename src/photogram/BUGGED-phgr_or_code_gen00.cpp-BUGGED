/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/


#include "StdAfx.h"

#if (ELISE_INSERT_CODE_GEN)



//===============================

//===============================


#endif // ELISE_INSERT_CODE_GEN


#include "../../CodeGenere/photogram/cCylindre_EqRat_CodGen.h"

#include "../../CodeGenere/photogram/cSetVar.h"
#include "../../CodeGenere/photogram/cSetValsEq.h"
#include "../../CodeGenere/photogram/cSetNormEuclid3.h"
#include "../../CodeGenere/photogram/cSetNormEuclidVect3.h"
#include "../../CodeGenere/photogram/cSetScal3.h"
#include "../../CodeGenere/photogram/cRegD1.h"
#include "../../CodeGenere/photogram/cRegD2.h"

#include "../../CodeGenere/photogram/cEqCorLI_Single_5.h"
#include "../../CodeGenere/photogram/cEqCorLI_Single_9.h"
#include "../../CodeGenere/photogram/cEqCorLI_Multi_5.h"
#include "../../CodeGenere/photogram/cEqCorLI_Multi_9.h"


#include "../../CodeGenere/photogram/cEqCorLI_Multi_1_DRPts0_NonNorm.h"
#include "../../CodeGenere/photogram/cEqCorLI_Multi_1_DRPts1_NonNorm.h"
#include "../../CodeGenere/photogram/cEqCorLI_Multi_1_DRPts2_NonNorm.h"
#include "../../CodeGenere/photogram/cEqCorLI_Multi_1_DRPts3_NonNorm.h"


//=======================  GUIMBAL LOCK ========================
#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CNoDist.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CNoDist.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CDRad5.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CDRad5APFraser.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CDRad5APFraser.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CEbner.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CEbner.h"


#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CDCBrown.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CDCBrown.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CFishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CFishEye_10_5_5.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CDRad_PPaEqPPs.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CFraser_PPaEqPPs.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CFraser_PPaEqPPs.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CDRad_PPaEqPPs.h"



#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn2.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CPolyn2.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn3.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CPolyn3.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn4.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CPolyn4.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn5.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CPolyn5.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn6.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CPolyn6.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn7.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__PProjInc_M2CPolyn7.h"

//=======================   NO DIST =============================

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CNoDist.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CNoDist.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CNoDist.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CNoDist.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CDRad5.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CDRad5APFraser.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CDRad5APFraser.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CDRad5APFraser.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CDRad5APFraser.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CEbner.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CEbner.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CEbner.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CEbner.h"





#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CDCBrown.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CDCBrown.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CDCBrown.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CDCBrown.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CFishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CFishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CFishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CFishEye_10_5_5.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CDRad_PPaEqPPs.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CFraser_PPaEqPPs.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CDRad_PPaEqPPs.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CFraser_PPaEqPPs.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CDRad_PPaEqPPs.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CFraser_PPaEqPPs.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CDRad_PPaEqPPs.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CFraser_PPaEqPPs.h"





#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CPolyn2.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CPolyn2.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CPolyn2.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CPolyn2.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CPolyn3.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CPolyn3.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CPolyn3.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CPolyn3.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CPolyn4.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CPolyn4.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CPolyn4.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CPolyn4.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CPolyn5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CPolyn5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CPolyn5.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CPolyn5.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CPolyn6.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CPolyn6.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CPolyn6.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CPolyn6.h"

#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CPolyn7.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CPolyn7.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CPolyn7.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CPolyn7.h"


// #include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn2.h"
// #include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn3.h"
// #include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn4.h"
// #include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn5.h"
// #include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn6.h"
// #include "../../CodeGenere/photogram/cEqAppui_GL__PTInc_M2CPolyn7.h"


//===============================================================


#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CDCBrown.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_C2MDCBrown.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CDCBrown.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_C2MDCBrown.h"

#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CDRad5APFraser.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_C2MDRad5APFraser.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CDRad5APFraser.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_C2MDRad5APFraser.h"




#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_C2MDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_C2MDRad5.h"

#include "../../CodeGenere/photogram/cEqAppui_AFocal_TerFix_M2CDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_AFocal_GL__TerFix_M2CDRad5.h" 
#include "../../CodeGenere/photogram/cEqAppui_AFocal_GL__PProjInc_M2CDRad5.h" 
#include "../../CodeGenere/photogram/cEqAppui_AFocal_PTInc_M2CDRad5.h"
#include "../../CodeGenere/photogram/cEqAppui_AFocal_GL__PTInc_M2CDRad5.h"    
#include "../../CodeGenere/photogram/cEqAppui_AFocal_PProjInc_M2CDRad5.h"   



#define AFOC_ADD_ENTRY(aDist)\
AddEntry("cEqAppui_AFocal_TerFix_M2C"#aDist,cEqAppui_AFocal_TerFix_M2C##aDist::Alloc);\
AddEntry("cEqAppui_AFocal_GL__TerFix_M2C"#aDist,cEqAppui_AFocal_GL__TerFix_M2C##aDist::Alloc);\
AddEntry("cEqAppui_AFocal_GL__PProjInc_M2C"#aDist,cEqAppui_AFocal_GL__PProjInc_M2C##aDist::Alloc);\
AddEntry("cEqAppui_AFocal_PTInc_M2C"#aDist,cEqAppui_AFocal_PTInc_M2C##aDist::Alloc);\
AddEntry("cEqAppui_AFocal_GL__PTInc_M2C"#aDist,cEqAppui_AFocal_GL__PTInc_M2C##aDist::Alloc);\
AddEntry("cEqAppui_AFocal_PProjInc_M2C"#aDist,cEqAppui_AFocal_PProjInc_M2C##aDist::Alloc);\




#include "../../CodeGenere/photogram/cEqAppui_PTInc_C2MEbner.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CEbner.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_C2MEbner.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CEbner.h"



// HERE


#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CPolyn2.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CPolyn3.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CPolyn4.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CPolyn5.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CPolyn6.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CPolyn7.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CFishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CDRad_PPaEqPPs.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CFraser_PPaEqPPs.h"


#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CPolyn2.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CPolyn3.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CPolyn4.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CPolyn5.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CPolyn6.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CPolyn7.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CFishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CDRad_PPaEqPPs.h"
#include "../../CodeGenere/photogram/cEqAppui_PProjInc_M2CFraser_PPaEqPPs.h"





#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CDPol3.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_C2MDPol3.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CDPol5.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_C2MDPol5.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CDPol7.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_C2MDPol7.h"


#include "../../CodeGenere/photogram/cEqAppui_PTInc_M2CNoDist.h"
#include "../../CodeGenere/photogram/cEqAppui_PTInc_C2MNoDist.h"

#include "../../CodeGenere/photogram/cEqAppuiXDHom.h"
#include "../../CodeGenere/photogram/cEqAppuiYDHom.h"

#include "../../CodeGenere/photogram/cEqCibleEllipse7.h"
#include "../../CodeGenere/photogram/cEqCibleEllipse6.h"
#include "../../CodeGenere/photogram/cEqCibleEllipse5.h"
#include "../../CodeGenere/photogram/cEqCibleEllipse1.h"

#include "../../CodeGenere/photogram/cEqResiduIm1DRad5Id.h"
#include "../../CodeGenere/photogram/cEqResiduIm2DRad5Id.h"
#include "../../CodeGenere/photogram/cEqCoplanDRad5Id.h"



#include "../../CodeGenere/photogram/cEqResiduIm1DRad5APFraserId.h"
#include "../../CodeGenere/photogram/cEqResiduIm2DRad5APFraserId.h"
#include "../../CodeGenere/photogram/cEqCoplanDRad5APFraserId.h"




#include "../../CodeGenere/photogram/cEqCoplanGrid.h"
#include "../../CodeGenere/photogram/cEqResiduIm1Grid.h"
#include "../../CodeGenere/photogram/cEqResiduIm2Grid.h"
#include "../../CodeGenere/photogram/cEqAppuiGrid.h"

#include "../../CodeGenere/photogram/cEqResiduIm1NoDistId.h"
#include "../../CodeGenere/photogram/cEqResiduIm2NoDistId.h"
#include "../../CodeGenere/photogram/cEqCoplanNoDistId.h"




#include "../../CodeGenere/photogram/cEqAppuiXDPol3.h"
#include "../../CodeGenere/photogram/cEqAppuiYDPol3.h"
#include "../../CodeGenere/photogram/cEqAppuiXDPol5.h"
#include "../../CodeGenere/photogram/cEqAppuiYDPol5.h"
#include "../../CodeGenere/photogram/cEqAppuiXDPol7.h"
#include "../../CodeGenere/photogram/cEqAppuiYDPol7.h"
#include "../../CodeGenere/photogram/EqObsBaclt.h"




#include "../../CodeGenere/photogram/cEqCoplanDPol3Id.h"
#include "../../CodeGenere/photogram/cEqCoplanDPol5Id.h"
#include "../../CodeGenere/photogram/cEqCoplanDPol7Id.h"

#include "../../CodeGenere/photogram/cEqHomogrXDeg5.h"
#include "../../CodeGenere/photogram/cEqHomogrYDeg5.h"

#include "../../CodeGenere/photogram/cEqLin_1.h"
#include "../../CodeGenere/photogram/cEqLin_2.h"
#include "../../CodeGenere/photogram/cEqLin_3.h"

#include "../../CodeGenere/photogram/cEqCorrelGrid_9_Im2Var.h"
// #include "../../CodeGenere/photogram/cEqCorrelGrid_25_Im2Var.h"

#include "../../CodeGenere/photogram/cEqObsRotVect_CodGen.h"

#include "../../CodeGenere/photogram/cEqCalibCroisee_NoDist_CodGenC2M.h"
#include "../../CodeGenere/photogram/cEqCalibCroisee_DRad5_CodGenC2M.h"
#include "../../CodeGenere/photogram/cEqCalibCroisee_DRad5APFraser_CodGenC2M.h"
#include "../../CodeGenere/photogram/cEqCalibCroisee_NoDist_CodGenM2C.h"
#include "../../CodeGenere/photogram/cEqCalibCroisee_DRad5_CodGenM2C.h"
#include "../../CodeGenere/photogram/cEqCalibCroisee_DRad5APFraser_CodGenM2C.h"

#include "../../CodeGenere/photogram/cEqDirectDistDRad5Reformat_CodGen.h"
#include "../../CodeGenere/photogram/cEqDirectDistDRad5Interp_CodGen.h"
#include "../../CodeGenere/photogram/cEqDirectDistDRad5Bayer_CodGen.h"



#include "../../CodeGenere/photogram/cEqCoplanEbnerId.h"
#include "../../CodeGenere/photogram/cEqCoplanDCBrownId.h"
#include "../../CodeGenere/photogram/cEqResiduIm1EbnerId.h"
#include "../../CodeGenere/photogram/cEqResiduIm1DCBrownId.h"
#include "../../CodeGenere/photogram/cEqResiduIm2EbnerId.h"
#include "../../CodeGenere/photogram/cEqResiduIm2DCBrownId.h"



#include "../../CodeGenere/photogram/cCodeGenEqPlanInconnuFormel.h"


#define NEW_ADD_ENTRY(aDist)\
AddEntry("cEqAppui_TerFix_M2C"#aDist,cEqAppui_TerFix_M2C##aDist::Alloc);\
AddEntry("cEqAppui_GL__TerFix_M2C"#aDist,cEqAppui_GL__TerFix_M2C##aDist::Alloc);\
AddEntry("cEqAppui_GL__PTInc_M2C"#aDist,cEqAppui_GL__PTInc_M2C##aDist::Alloc);\
AddEntry("cEqAppui_GL__PProjInc_M2C"#aDist,cEqAppui_GL__PProjInc_M2C##aDist::Alloc);\
AddEntry("cEqAppui_NoDist__GL__PTInc_M2C"#aDist,cEqAppui_NoDist__GL__PTInc_M2C##aDist::Alloc);\
AddEntry("cEqAppui_NoDist__GL__PProjInc_M2C"#aDist,cEqAppui_NoDist__GL__PProjInc_M2C##aDist::Alloc);\
AddEntry("cEqAppui_NoDist__PTInc_M2C"#aDist,cEqAppui_NoDist__PTInc_M2C##aDist::Alloc);\
AddEntry("cEqAppui_NoDist__PProjInc_M2C"#aDist,cEqAppui_NoDist__PProjInc_M2C##aDist::Alloc);


#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PProjInc_M2CNoVar.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__GL__PTInc_M2CNoVar.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PProjInc_M2CNoVar.h"
#include "../../CodeGenere/photogram/cEqAppui_NoDist__PTInc_M2CNoVar.h"



//   NOUVEAUX APPUIS 

#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CNoVar.h"          
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CNoDist.h"         

#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CDRad5.h"                    
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CDRad5APFraser.h"            

#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CEbner.h"                    
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CDCBrown.h"                  




#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CPolyn2.h"         
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CPolyn3.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CPolyn4.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CPolyn5.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CPolyn6.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CPolyn7.h"

#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CFishEye_10_5_5.h" 
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CDRad_PPaEqPPs.h"                    
#include "../../CodeGenere/photogram/cEqAppui_GL__TerFix_M2CFraser_PPaEqPPs.h"                    


#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CNoVar.h"          
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CNoDist.h"         

#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CDRad5.h"                    
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CDRad5APFraser.h"            

#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CEbner.h"                    
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CDCBrown.h"                  


#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CPolyn2.h"         
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CPolyn3.h"
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CPolyn4.h"
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CPolyn5.h"
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CPolyn6.h"
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CPolyn7.h"

#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CFishEye_10_5_5.h" 
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CEquiSolid_FishEye_10_5_5.h"
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CDRad_PPaEqPPs.h"                    
#include "../../CodeGenere/photogram/cEqAppui_TerFix_M2CFraser_PPaEqPPs.h"                    








void cElCompiledFonc::InitEntries()
{
     static bool First = true;
     if (! First)
	return;

     First = false;

     AFOC_ADD_ENTRY(DRad5)

     NEW_ADD_ENTRY(NoDist)
     NEW_ADD_ENTRY(DRad5)
     NEW_ADD_ENTRY(DRad5APFraser)
     NEW_ADD_ENTRY(Ebner)
     NEW_ADD_ENTRY(DCBrown)
     NEW_ADD_ENTRY(FishEye_10_5_5)
     NEW_ADD_ENTRY(EquiSolid_FishEye_10_5_5)
     NEW_ADD_ENTRY(DRad_PPaEqPPs)
     NEW_ADD_ENTRY(Fraser_PPaEqPPs)
     NEW_ADD_ENTRY(Polyn2)
     NEW_ADD_ENTRY(Polyn3)
     NEW_ADD_ENTRY(Polyn4)
     NEW_ADD_ENTRY(Polyn5)
     NEW_ADD_ENTRY(Polyn6)
     NEW_ADD_ENTRY(Polyn7)


     AddEntry("cEqAppui_NoDist__GL__PProjInc_M2CNoVar",cEqAppui_NoDist__GL__PProjInc_M2CNoVar::Alloc);    // EqHom
     AddEntry("cEqAppui_NoDist__GL__PTInc_M2CNoVar",cEqAppui_NoDist__GL__PTInc_M2CNoVar::Alloc);    // EqHom
     AddEntry("cEqAppui_NoDist__PProjInc_M2CNoVar",cEqAppui_NoDist__PProjInc_M2CNoVar::Alloc);    // EqHom
     AddEntry("cEqAppui_NoDist__PTInc_M2CNoVar",cEqAppui_NoDist__PTInc_M2CNoVar::Alloc);    // EqHom



     AddEntry("cCylindre_EqRat_CodGen",cCylindre_EqRat_CodGen::Alloc);

     AddEntry("cEqDirectDistDRad5Reformat_CodGen",cEqDirectDistDRad5Reformat_CodGen::Alloc);    // EqHom
     AddEntry("cEqDirectDistDRad5Interp_CodGen",cEqDirectDistDRad5Interp_CodGen::Alloc);    // EqHom
     AddEntry("cEqDirectDistDRad5Bayer_CodGen",cEqDirectDistDRad5Bayer_CodGen::Alloc);    // EqHom

     AddEntry("cEqCalibCroisee_NoDist_CodGenC2M",cEqCalibCroisee_NoDist_CodGenC2M::Alloc);    // EqHom
     AddEntry("cEqCalibCroisee_DRad5_CodGenC2M",cEqCalibCroisee_DRad5_CodGenC2M::Alloc);    // EqHom
     AddEntry("cEqCalibCroisee_DRad5APFraser_CodGenC2M",cEqCalibCroisee_DRad5APFraser_CodGenC2M::Alloc);    // EqHom

     AddEntry("cEqCalibCroisee_NoDist_CodGenM2C",cEqCalibCroisee_NoDist_CodGenM2C::Alloc);    // EqHom
     AddEntry("cEqCalibCroisee_DRad5_CodGenM2C",cEqCalibCroisee_DRad5_CodGenM2C::Alloc);    // EqHom
     AddEntry("cEqCalibCroisee_DRad5APFraser_CodGenM2C",cEqCalibCroisee_DRad5APFraser_CodGenM2C::Alloc);    // EqHom


     AddEntry("cEqObsRotVect_CodGen",cEqObsRotVect_CodGen::Alloc);    // EqHom

     AddEntry("cEqHomogrXDeg5",cEqHomogrXDeg5::Alloc);    // EqHom
     AddEntry("cEqHomogrYDeg5",cEqHomogrYDeg5::Alloc);
     AddEntry("cEqCoplanDPol3Id",cEqCoplanDPol3Id::Alloc);  // DPol
     AddEntry("cEqCoplanDPol5Id",cEqCoplanDPol5Id::Alloc);
     AddEntry("cEqCoplanDPol7Id",cEqCoplanDPol7Id::Alloc);
     AddEntry("cEqResiduIm1NoDistId",cEqResiduIm1NoDistId::Alloc);  // NoDist
     AddEntry("cEqResiduIm2NoDistId",cEqResiduIm2NoDistId::Alloc);
     AddEntry("cEqCoplanNoDistId",cEqCoplanNoDistId::Alloc);


     AddEntry("cEqCoplanGrid",cEqCoplanGrid::Alloc);          // Grid
     AddEntry("cEqAppuiGrid",cEqAppuiGrid::Alloc);          // Grid
     AddEntry("cEqResiduIm1Grid",cEqResiduIm1Grid::Alloc);
     AddEntry("cEqResiduIm2Grid",cEqResiduIm2Grid::Alloc);
     AddEntry("cSetVar",cSetVar::Alloc);                        // Uti
     AddEntry("cSetValsEq",cSetValsEq::Alloc);                        // Uti
     AddEntry("cRegD1",cRegD1::Alloc);
     AddEntry("cRegD2",cRegD2::Alloc);
     AddEntry("cSetNormEuclid3",cSetNormEuclid3::Alloc);
     AddEntry("cSetNormEuclidVect3",cSetNormEuclidVect3::Alloc);
     AddEntry("cSetScal3",cSetScal3::Alloc);


     AddEntry("cEqCorLI_Single_5",cEqCorLI_Single_5::Alloc);    // LaserImage
     AddEntry("cEqCorLI_Single_9",cEqCorLI_Single_9::Alloc);
     AddEntry("cEqCorLI_Multi_5",cEqCorLI_Multi_5::Alloc);     // LaserImage

     AddEntry("cEqCorLI_Multi_1_DRPts0_NonNorm",cEqCorLI_Multi_1_DRPts0_NonNorm::Alloc);     // LaserImage
     // AddEntry("cEqCorLI_Multi_1_DRPts1_NonNorm",cEqCorLI_Multi_1_DRPts1_NonNorm::Alloc);     // LaserImage
     // AddEntry("cEqCorLI_Multi_1_DRPts2_NonNorm",cEqCorLI_Multi_1_DRPts2_NonNorm::Alloc);     // LaserImage
     // AddEntry("cEqCorLI_Multi_1_DRPts3_NonNorm",cEqCorLI_Multi_1_DRPts3_NonNorm::Alloc);     // LaserImage

     AddEntry("cEqCorLI_Multi_9",cEqCorLI_Multi_9::Alloc);
     AddEntry("cEqAppuiXDHom",cEqAppuiXDHom::Alloc);              // DHom
     AddEntry("cEqAppuiYDHom",cEqAppuiYDHom::Alloc);
     AddEntry("cEqResiduIm1DRad5Id",cEqResiduIm1DRad5Id::Alloc);   // DRad
     AddEntry("cEqResiduIm2DRad5Id",cEqResiduIm2DRad5Id::Alloc);
     AddEntry("cEqCoplanDRad5Id",cEqCoplanDRad5Id::Alloc);



     AddEntry("cEqCibleEllipse7",cEqCibleEllipse7::Alloc);     // Ellipse
     AddEntry("cEqCibleEllipse6",cEqCibleEllipse6::Alloc);     // Ellipse
     AddEntry("cEqCibleEllipse5",cEqCibleEllipse5::Alloc);
     AddEntry("cEqCibleEllipse1",cEqCibleEllipse1::Alloc);
     AddEntry("cEqLin_1",cEqLin_1::Alloc);  // cEqLin_
     AddEntry("cEqLin_2",cEqLin_2::Alloc);  
     AddEntry("cEqLin_3",cEqLin_3::Alloc);  

     AddEntry("cEqCoplanDRad5APFraserId",cEqCoplanDRad5APFraserId::Alloc);
     AddEntry("cEqResiduIm1DRad5APFraserId",cEqResiduIm1DRad5APFraserId::Alloc);
     AddEntry("cEqResiduIm2DRad5APFraserId",cEqResiduIm2DRad5APFraserId::Alloc);

     AddEntry("cEqAppuiXDPol3",cEqAppuiXDPol3::Alloc);
     AddEntry("cEqAppuiYDPol3",cEqAppuiYDPol3::Alloc);
     AddEntry("cEqAppuiXDPol5",cEqAppuiXDPol5::Alloc);
     AddEntry("cEqAppuiYDPol5",cEqAppuiYDPol5::Alloc);
     AddEntry("cEqAppuiXDPol7",cEqAppuiXDPol7::Alloc);
     AddEntry("cEqAppuiYDPol7",cEqAppuiYDPol7::Alloc);
     AddEntry("EqObsBaclt",EqObsBaclt::Alloc);

     AddEntry("cEqCorrelGrid_9_Im2Var",cEqCorrelGrid_9_Im2Var::Alloc);   //Grid Cor
     // AddEntry("cEqCorrelGrid_25_Im2Var",cEqCorrelGrid_25_Im2Var::Alloc);  



     AddEntry("cEqCoplanEbnerId",cEqCoplanEbnerId::Alloc);
     AddEntry("cEqCoplanDCBrownId",cEqCoplanDCBrownId::Alloc);
     AddEntry("cEqResiduIm1EbnerId",cEqResiduIm1EbnerId::Alloc);
     AddEntry("cEqResiduIm1DCBrownId",cEqResiduIm1DCBrownId::Alloc);
     AddEntry("cEqResiduIm2EbnerId",cEqResiduIm2EbnerId::Alloc);
     AddEntry("cEqResiduIm2DCBrownId",cEqResiduIm2DCBrownId::Alloc);

     AddEntry("cCodeGenEqPlanInconnuFormel",cCodeGenEqPlanInconnuFormel::Alloc);

     AddEntry("cEqAppui_PTInc_M2CDCBrown",cEqAppui_PTInc_M2CDCBrown::Alloc);
     AddEntry("cEqAppui_PTInc_C2MDCBrown",cEqAppui_PTInc_C2MDCBrown::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CDCBrown",cEqAppui_PProjInc_M2CDCBrown::Alloc);
     AddEntry("cEqAppui_PProjInc_C2MDCBrown",cEqAppui_PProjInc_C2MDCBrown::Alloc);

     AddEntry("cEqAppui_PTInc_M2CDRad5APFraser",cEqAppui_PTInc_M2CDRad5APFraser::Alloc);
     AddEntry("cEqAppui_PTInc_C2MDRad5APFraser",cEqAppui_PTInc_C2MDRad5APFraser::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CDRad5APFraser",cEqAppui_PProjInc_M2CDRad5APFraser::Alloc);
     AddEntry("cEqAppui_PProjInc_C2MDRad5APFraser",cEqAppui_PProjInc_C2MDRad5APFraser::Alloc);


     AddEntry("cEqAppui_PTInc_M2CDRad5",cEqAppui_PTInc_M2CDRad5::Alloc);
     AddEntry("cEqAppui_PTInc_C2MDRad5",cEqAppui_PTInc_C2MDRad5::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CDRad5",cEqAppui_PProjInc_M2CDRad5::Alloc);
     AddEntry("cEqAppui_PProjInc_C2MDRad5",cEqAppui_PProjInc_C2MDRad5::Alloc);


     AddEntry("cEqAppui_PTInc_M2CEbner",cEqAppui_PTInc_M2CEbner::Alloc);
     AddEntry("cEqAppui_PTInc_C2MEbner",cEqAppui_PTInc_C2MEbner::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CEbner",cEqAppui_PProjInc_M2CEbner::Alloc);
     AddEntry("cEqAppui_PProjInc_C2MEbner",cEqAppui_PProjInc_C2MEbner::Alloc);

     AddEntry("cEqAppui_PTInc_M2CPolyn2",cEqAppui_PTInc_M2CPolyn2::Alloc);
     AddEntry("cEqAppui_PTInc_M2CPolyn3",cEqAppui_PTInc_M2CPolyn3::Alloc);
     AddEntry("cEqAppui_PTInc_M2CPolyn4",cEqAppui_PTInc_M2CPolyn4::Alloc);
     AddEntry("cEqAppui_PTInc_M2CPolyn5",cEqAppui_PTInc_M2CPolyn5::Alloc);
     AddEntry("cEqAppui_PTInc_M2CPolyn6",cEqAppui_PTInc_M2CPolyn6::Alloc);
     AddEntry("cEqAppui_PTInc_M2CPolyn7",cEqAppui_PTInc_M2CPolyn7::Alloc);
     AddEntry("cEqAppui_PTInc_M2CFishEye_10_5_5",cEqAppui_PTInc_M2CFishEye_10_5_5::Alloc);
     AddEntry("cEqAppui_PTInc_M2CEquiSolid_FishEye_10_5_5",cEqAppui_PTInc_M2CEquiSolid_FishEye_10_5_5::Alloc);
     AddEntry("cEqAppui_PTInc_M2CDRad_PPaEqPPs",cEqAppui_PTInc_M2CDRad_PPaEqPPs::Alloc);
     AddEntry("cEqAppui_PTInc_M2CFraser_PPaEqPPs",cEqAppui_PTInc_M2CFraser_PPaEqPPs::Alloc);


     AddEntry("cEqAppui_PProjInc_M2CPolyn2",cEqAppui_PProjInc_M2CPolyn2::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CPolyn3",cEqAppui_PProjInc_M2CPolyn3::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CPolyn4",cEqAppui_PProjInc_M2CPolyn4::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CPolyn5",cEqAppui_PProjInc_M2CPolyn5::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CPolyn6",cEqAppui_PProjInc_M2CPolyn6::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CPolyn7",cEqAppui_PProjInc_M2CPolyn7::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CFishEye_10_5_5",cEqAppui_PProjInc_M2CFishEye_10_5_5::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CEquiSolid_FishEye_10_5_5",cEqAppui_PProjInc_M2CEquiSolid_FishEye_10_5_5::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CDRad_PPaEqPPs",cEqAppui_PProjInc_M2CDRad_PPaEqPPs::Alloc);
     AddEntry("cEqAppui_PProjInc_M2CFraser_PPaEqPPs",cEqAppui_PProjInc_M2CFraser_PPaEqPPs::Alloc);





     AddEntry("cEqAppui_PTInc_M2CDPol3",cEqAppui_PTInc_M2CDPol3::Alloc);
     AddEntry("cEqAppui_PTInc_C2MDPol3",cEqAppui_PTInc_C2MDPol3::Alloc);

     AddEntry("cEqAppui_PTInc_M2CDPol5",cEqAppui_PTInc_M2CDPol5::Alloc);
     AddEntry("cEqAppui_PTInc_C2MDPol5",cEqAppui_PTInc_C2MDPol5::Alloc);

     AddEntry("cEqAppui_PTInc_M2CDPol7",cEqAppui_PTInc_M2CDPol7::Alloc);
     AddEntry("cEqAppui_PTInc_C2MDPol7",cEqAppui_PTInc_C2MDPol7::Alloc);




     AddEntry("cEqAppui_PTInc_M2CNoDist",cEqAppui_PTInc_M2CNoDist::Alloc);
     AddEntry("cEqAppui_PTInc_C2MNoDist",cEqAppui_PTInc_C2MNoDist::Alloc);


// ====  GUIMBAL LOCK ============

#if (0)
     AddEntry("cEqAppui_GL__PTInc_M2CNoDist",cEqAppui_GL__PTInc_M2CNoDist::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CNoDist",cEqAppui_GL__PProjInc_M2CNoDist::Alloc);

     AddEntry("cEqAppui_GL__PTInc_M2CDRad5",cEqAppui_GL__PTInc_M2CDRad5::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CDRad5",cEqAppui_GL__PProjInc_M2CDRad5::Alloc);

     AddEntry("cEqAppui_GL__PTInc_M2CDRad5APFraser",cEqAppui_GL__PTInc_M2CDRad5APFraser::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CDRad5APFraser",cEqAppui_GL__PProjInc_M2CDRad5APFraser::Alloc);

     AddEntry("cEqAppui_GL__PTInc_M2CEbner",cEqAppui_GL__PTInc_M2CEbner::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CEbner",cEqAppui_GL__PProjInc_M2CEbner::Alloc);

     AddEntry("cEqAppui_GL__PTInc_M2CDCBrown",cEqAppui_GL__PTInc_M2CDCBrown::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CDCBrown",cEqAppui_GL__PProjInc_M2CDCBrown::Alloc);


     AddEntry("cEqAppui_GL__PTInc_M2CFishEye_10_5_5",cEqAppui_GL__PTInc_M2CFishEye_10_5_5::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CFishEye_10_5_5",cEqAppui_GL__PProjInc_M2CFishEye_10_5_5::Alloc);

     AddEntry("cEqAppui_GL__PTInc_M2CEquiSolid_FishEye_10_5_5",cEqAppui_GL__PTInc_M2CEquiSolid_FishEye_10_5_5::Alloc);
     AddEntry("cEqAppui_GL__PTInc_M2CDRad_PPaEqPPs",           cEqAppui_GL__PTInc_M2CDRad_PPaEqPPs::Alloc);
     AddEntry("cEqAppui_GL__PTInc_M2CFraser_PPaEqPPs",          cEqAppui_GL__PTInc_M2CFraser_PPaEqPPs::Alloc);

     AddEntry("cEqAppui_GL__PProjInc_M2CEquiSolid_FishEye_10_5_5",cEqAppui_GL__PProjInc_M2CEquiSolid_FishEye_10_5_5::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CDRad_PPaEqPPs",           cEqAppui_GL__PProjInc_M2CDRad_PPaEqPPs::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CFraser_PPaEqPPs",         cEqAppui_GL__PProjInc_M2CFraser_PPaEqPPs::Alloc);





     AddEntry("cEqAppui_GL__PTInc_M2CPolyn2",cEqAppui_GL__PTInc_M2CPolyn2::Alloc);
     AddEntry("cEqAppui_GL__PProjInc_M2CPolyn2",cEqAppui_GL__PProjInc_M2CPolyn2::Alloc);
#endif


/*
     AddEntry("cEqAppui_GL__PTInc_M2CPolyn2",cEqAppui_GL__PTInc_M2CPolyn2::Alloc);
     AddEntry("cEqAppui_GL__PTInc_M2CPolyn3",cEqAppui_GL__PTInc_M2CPolyn3::Alloc);
     AddEntry("cEqAppui_GL__PTInc_M2CPolyn4",cEqAppui_GL__PTInc_M2CPolyn4::Alloc);
     AddEntry("cEqAppui_GL__PTInc_M2CPolyn5",cEqAppui_GL__PTInc_M2CPolyn5::Alloc);
     AddEntry("cEqAppui_GL__PTInc_M2CPolyn6",cEqAppui_GL__PTInc_M2CPolyn6::Alloc);
     AddEntry("cEqAppui_GL__PTInc_M2CPolyn7",cEqAppui_GL__PTInc_M2CPolyn7::Alloc);

*/
}

/*
#include "../../CodeGenere/photogram/cEqCoplanEbnerId.h"
#include "../../CodeGenere/photogram/cEqCoplanDCBrownId.h"
#include "../../CodeGenere/photogram/cEqResiduIm1EbnerId.h"
#include "../../CodeGenere/photogram/cEqResiduIm1DCBrownId.h"
#include "../../CodeGenere/photogram/cEqResiduIm2EbnerId.h"
#include "../../CodeGenere/photogram/cEqResiduIm2DCBrownId.h"
*/


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
