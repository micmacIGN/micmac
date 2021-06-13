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
#include "general/all.h"
#include "private/all.h"


//================== ELLIPSE =====================================

void GenCodeEllipse(const cMirePolygonEtal & aMire)
{
   cSetEqFormelles aSet1;
   aSet1.NewEqElIm(aMire,true);
}

void GenCodeEllipse()
{
/*
    GenCodeEllipse(cMirePolygonEtal::SofianeMire2());
    GenCodeEllipse(cMirePolygonEtal::IGNMire5());
    GenCodeEllipse(cMirePolygonEtal::IGNMire7());
    GenCodeEllipse(cMirePolygonEtal::MtdMire9());
    GenCodeEllipse(cMirePolygonEtal::MtdMire9());
    */
}


//==================== LASER ======================================

void GenCodeLaserImage(bool Normalize,bool Multi,INT aNb,INT aDegre)
{
     tParamAFocal aNOPAF;
	cout << "GenCodeLaserImage\n";
     CamStenopeIdeale aCam(true,1.0,Pt2dr(0,0),aNOPAF);
     cSetEqFormelles aSet;
     Im2D_REAL4 aI(1,1);

     cLIParam_Image  * aP1 =  aSet.NewLIParamImage(aI,1.0,aCam,cNameSpaceEqF::eRotFigee);
     cLIParam_Image  * aP2 =  aSet.NewLIParamImage(aI,1.0,aCam,cNameSpaceEqF::eRotFigee);

     cRotationFormelle * aRotPts = 0;
     if (aDegre >=0)
        aRotPts = aSet.NewRotationEvol(ElRotation3D(Pt3dr(0,0,0),0,0,0),aDegre);
     aSet.NewLIEqVueLaserIm(aRotPts,Multi,Normalize,aNb,*aP1,*aP2,true);
}

void GenCodeLaserImage()
{
     GenCodeLaserImage(false,true,1,0);
     GenCodeLaserImage(false,true,1,1);
     GenCodeLaserImage(false,true,1,2);
     GenCodeLaserImage(false,true,1,3);
     // GenCodeLaserImage(false,true,5,3);
     // GenCodeLaserImage(false,true,5,0);

     // GenCodeLaserImage(true,true,5,0);
     // GenCodeLaserImage(true,true,5,1);
     // GenCodeLaserImage(true,true,5,2);
/*
     GenCodeLaserImage(true,false,5,-1);
     GenCodeLaserImage(true,true,5,-1);
     GenCodeLaserImage(true,false,9,-1);
     GenCodeLaserImage(true,true,9,-1);
     GenCodeLaserImage(true,true,25,-1);
     GenCodeLaserImage(true,true,49,-1);
*/
}

//==================== UTI pour Appui/Liaison ==================

template <class Type> Type * CamSetSize(Type * aCam)
{
    aCam->SetSz(Pt2di(2000,3000));
    return aCam;
}

static  CamStenopeIdeale * CamIdeale(bool C2M,const tParamAFocal  & aPAF)
{
   return CamSetSize(new CamStenopeIdeale(C2M,1.0,Pt2dr(0,0),aPAF));
}


static cCamStenopeDistRadPol * CamDRad5(bool C2M,const tParamAFocal  & aPAF)
{
   ElDistRadiale_PolynImpair aDist =
	                   ElDistRadiale_PolynImpair::DistId(1.0,Pt2dr(0,0),5);
   return  CamSetSize(new cCamStenopeDistRadPol(C2M,1.0,Pt2dr(0,0),aDist,aPAF));
}

static cCamStenopeModStdPhpgr * CamPhgrStd(bool C2M,const tParamAFocal  & aPAF)
{
   ElDistRadiale_PolynImpair aDrad =
	                   ElDistRadiale_PolynImpair::DistId(1.0,Pt2dr(0,0),5);
   cDistModStdPhpgr aDStd(aDrad);
   return  CamSetSize(new cCamStenopeModStdPhpgr(C2M,1.0,Pt2dr(0,0),aDStd,aPAF));
}
static cCamStenopeDistHomogr * CamHom(bool C2M,const tParamAFocal  & aPAF)
{
   return CamSetSize(new cCamStenopeDistHomogr(C2M,1.0,Pt2dr(0,0),cElHomographie::Id(),aPAF));
}


static cCamStenopeDistPolyn * CamPolXY_3(bool C2M,const tParamAFocal  & aPAF)
{
       return  CamSetSize(new cCamStenopeDistPolyn(C2M,1,Pt2dr(0,0),ElDistortionPolynomiale::DistId(3,1.0),aPAF));
}

static cCamStenopeDistPolyn * CamPolXY_5(bool C2M,const tParamAFocal  & aPAF)
{
       return  CamSetSize(new cCamStenopeDistPolyn(C2M,1,Pt2dr(0,0),ElDistortionPolynomiale::DistId(5,1.0),aPAF));
}

static cCamStenopeDistPolyn * CamPolXY_7(bool C2M,const tParamAFocal  & aPAF)
{
       return  CamSetSize(new cCamStenopeDistPolyn(C2M,1,Pt2dr(0,0),ElDistortionPolynomiale::DistId(7,1.0),aPAF));
}

static cCam_Ebner * CamEbner(bool C2M,const tParamAFocal  & aPAF) 
{
    return  new cCam_Ebner(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF);
}

static cCam_DCBrown * CamDCBrown(bool C2M,const tParamAFocal  & aPAF) 
{
    return  new cCam_DCBrown(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF);
}

//CamEquiSolFishEye_10_5_5

static cCam_DRad_PPaEqPPs * CamDrPPaPPs(bool C2M,const tParamAFocal  & aPAF) { return  new cCam_DRad_PPaEqPPs(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }
static cCam_Fraser_PPaEqPPs * CamFraPPaPPs(bool C2M,const tParamAFocal  & aPAF) { return  new cCam_Fraser_PPaEqPPs(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }

static cCam_Polyn2 * CamPolyn2(bool C2M,const tParamAFocal  & aPAF) { return  new cCam_Polyn2(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }
static cCam_Polyn3 * CamPolyn3(bool C2M,const tParamAFocal  & aPAF) { return  new cCam_Polyn3(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }
static cCam_Polyn4 * CamPolyn4(bool C2M,const tParamAFocal  & aPAF) { return  new cCam_Polyn4(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }
static cCam_Polyn5 * CamPolyn5(bool C2M,const tParamAFocal  & aPAF) { return  new cCam_Polyn5(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }
static cCam_Polyn6 * CamPolyn6(bool C2M,const tParamAFocal  & aPAF) { return  new cCam_Polyn6(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }
static cCam_Polyn7 * CamPolyn7(bool C2M,const tParamAFocal  & aPAF) { return  new cCam_Polyn7(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }
static cCamLin_FishEye_10_5_5 * CamLinFishEye_10_5_5(bool C2M,const tParamAFocal  & aPAF) { return  new cCamLin_FishEye_10_5_5(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }
static cCamEquiSol_FishEye_10_5_5 * CamEquiSolFishEye_10_5_5(bool C2M,const tParamAFocal  & aPAF) { return  new cCamEquiSol_FishEye_10_5_5(C2M,1,Pt2dr(0,0),Pt2dr(2000,3000),aPAF); }



cParamIntrinsequeFormel * PIF_For_GC
                          (
			       bool                 C2M,
                               const std::string &  aType,
			       cSetEqFormelles &    aSet,
                               const tParamAFocal & aPAF
			  )
{
    if (aType == "NoVar")
    {
       return aSet.NewParamIntrNoDist(C2M,CamIdeale(C2M,aPAF),false);
    }
    if (aType == "NoDist")
    {
       return aSet.NewParamIntrNoDist(C2M,CamIdeale(C2M,aPAF));
    }
    if (aType == "DRad5")
    {
       return  aSet.NewIntrDistRad(C2M,CamDRad5(C2M,aPAF),3);
    }
    if (aType == "PhgrStd")
    {
       return  aSet.NewIntrDistStdPhgr(C2M,CamPhgrStd(C2M,aPAF),3);
    }
    if (aType == "Homogr")
    {
       return aSet.NewDistHomF(C2M,CamHom(C2M,aPAF),cNameSpaceEqF::eHomLibre);
       // return aSet.NewDistHomF(new cCamStenopeDistHomogr(1.0,Pt2dr(0,0),cElHomographie::Id()),cNameSpaceEqF::eHomLibre);
    }

    if (aType == "PolXY3")
    {
       return aSet.NewIntrPolyn(C2M,CamPolXY_3(C2M,aPAF));
    }
    if (aType == "PolXY5")
    {
       
       return aSet.NewIntrPolyn(C2M,CamPolXY_5(C2M,aPAF));
       // cCamStenopeDistPolyn *  aCam5 = new cCamStenopeDistPolyn(1,Pt2dr(0,0),ElDistortionPolynomiale::DistId(5,1.0));
       // return aSet.NewIntrPolyn(aCam5);
    }
    if (aType == "PolXY7")
    {
       return aSet.NewIntrPolyn(C2M,CamPolXY_7(C2M,aPAF));
    }

    if (aType == "Ebner")
    {
	// cCam_Ebner * aCamEbner = new cCam_Ebner(1,Pt2dr(0,0),Pt2di(2000,3000));
        // return cPIF_Ebner::Alloc(aSet,1,Pt2dr(0,0),aDist);
        return cPIF_Ebner::Alloc(C2M,CamEbner(C2M,aPAF),aSet);
    }

    if (aType == "DCBrown")
    {
	// cDist_DCBrown aDist(Pt2dr(3000,2000));
        return cPIF_DCBrown::Alloc(C2M,CamDCBrown(C2M,aPAF),aSet);
    }

    if (aType == "Polyn2") { return cPIF_Polyn2::Alloc(C2M,CamPolyn2(C2M,aPAF),aSet); }
    if (aType == "Polyn3") { return cPIF_Polyn3::Alloc(C2M,CamPolyn3(C2M,aPAF),aSet); }
    if (aType == "Polyn4") { return cPIF_Polyn4::Alloc(C2M,CamPolyn4(C2M,aPAF),aSet); }
    if (aType == "Polyn5") { return cPIF_Polyn5::Alloc(C2M,CamPolyn5(C2M,aPAF),aSet); }
    if (aType == "Polyn6") { return cPIF_Polyn6::Alloc(C2M,CamPolyn6(C2M,aPAF),aSet); }
    if (aType == "Polyn7") { return cPIF_Polyn7::Alloc(C2M,CamPolyn7(C2M,aPAF),aSet); }
    if (aType == "FishEye_10_5_5") { return cPIFLin_FishEye_10_5_5::Alloc(C2M,CamLinFishEye_10_5_5(C2M,aPAF),aSet); }
    if (aType == "EquiSolFishEye_10_5_5") { return cPIFEquiSol_FishEye_10_5_5::Alloc(C2M,CamEquiSolFishEye_10_5_5(C2M,aPAF),aSet); }


    if (aType ==  "RadPPaEqPPs") {return cPIF_DRad_PPaEqPPs::Alloc(C2M,CamDrPPaPPs(C2M,aPAF),aSet);}
    if (aType ==  "FraserPPaEqPPs") {return cPIF_Fraser_PPaEqPPs::Alloc(C2M,CamFraPPaPPs(C2M,aPAF),aSet);}


    ELISE_ASSERT(false,"Unknown Type in Param Intr Form (PIF_For_GC)");
    return 0;
}

//==================== APPUI ======================================

void GenCodeAppui(bool C2M,bool isFixe,bool isGL,bool isAFocal,bool wDist,const std::string & aType)
{
        std::vector<double>  aPAF;
        if (isAFocal)
        {
            aPAF.push_back(0.0);
            aPAF.push_back(0.0);
        }


std::cout << "Type Appui = " << aType << "\n";
	cSetEqFormelles aSet;
	cParamIntrinsequeFormel * aPIF = PIF_For_GC(C2M,aType,aSet,aPAF);
	ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);

        // Genere auto les appuis fixe en X et Y
         if (isFixe)
	 {
	    aPIF->NewCam(cNameSpaceEqF::eRotFigee,aRot,0,"toto",true,true);
         }
	 else
	{
	   cCameraFormelle * aCam =  aPIF->NewCam(cNameSpaceEqF::eRotFigee,aRot,0,"toto",false,false);
           if (isGL)
              aCam->SetGL(true);
	   aCam->AddForUseFctrEqAppuisInc(true,false,wDist);
	   aCam->AddForUseFctrEqAppuisInc(true,true,wDist);
	}
}

void GenCodeAppui(bool C2M,bool isFixe,bool isGL,bool isAFocal)
{
/*
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"DRad5");


    GenCodeAppui(C2M,isFixe,isGL,isAFocal,false,"NoVar");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"NoDist");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"PhgrStd");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"Ebner");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"DCBrown");


    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"Polyn2");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"Polyn3");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"Polyn4");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"Polyn5");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"Polyn6");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"Polyn7");
*/

/*
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"FishEye_10_5_5");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"EquiSolFishEye_10_5_5");
    GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"EquiSolFishEye_10_5_5");
*/

    // GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"RadPPaEqPPs");
    // GenCodeAppui(C2M,isFixe,isGL,isAFocal,true,"FraserPPaEqPPs");

    // GenCodeAppui(C2M,isFixe,isGL,isAFocal,false,"RadPPaEqPPs");
    // GenCodeAppui(C2M,isFixe,isGL,isAFocal,false,"FraserPPaEqPPs");

}

void GenCodeAppui()
{
/*  
*/
  // SANS AFOCAL
    GenCodeAppui(false,true,true,false);
    GenCodeAppui(false,true,false,false);
    GenCodeAppui(false,false,true,false);
    GenCodeAppui(false,false,false,false);

  // AVEC AFOCAL
/*
    GenCodeAppui(false,true,true,true);
    GenCodeAppui(false,true,false,true);
    GenCodeAppui(false,false,true,true);
    GenCodeAppui(false,false,false,true);
*/


}

//==================== LIASION ======================================

void GenCodeLiaison(const std::string & aType,cNameSpaceEqF::eModeResidu aMode)
{
     std::vector<double> aNoPAF;
std::cout << "TYPE liason = " <<  aType << "\n";
     cSetEqFormelles aSet;
     ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);
     cParamIntrinsequeFormel * pPIF = PIF_For_GC(true,aType,aSet,aNoPAF); // Liaisons : tjs C2M
     cCameraFormelle * pCam1 = pPIF->NewCam(cNameSpaceEqF::eRotFigee,aRot);
     cCameraFormelle * pCam2 = pPIF->NewCam(cNameSpaceEqF::eRotBaseU,aRot);

     aSet.NewCpleCam(*pCam1,*pCam2,aMode,true);
}

void GenCodeLiaison(const std::string & aType)
{
      GenCodeLiaison(aType,cNameSpaceEqF::eResiduCoplan);
      GenCodeLiaison(aType,cNameSpaceEqF::eResiduIm1);
      GenCodeLiaison(aType,cNameSpaceEqF::eResiduIm2);
}


void GenCodeLiaison()
{
	GenCodeLiaison("Ebner");
	GenCodeLiaison("DCBrown");
/*
	GenCodeLiaison("PolXY3");
	GenCodeLiaison("PolXY5");
	GenCodeLiaison("PolXY7");
*/

	GenCodeLiaison("NoDist");
	GenCodeLiaison("DRad5");
	GenCodeLiaison("PhgrStd");

}

//==================== GRILLE ========================================


void GenEqPlanInc()
{
     cSetEqFormelles aSet;
     cTriangulFormelle * aTr1 = aSet.NewTriangulFormelleUnitaire(1);
     Pt2dr aP = aTr1->APointInTri();

     std::cout << aP << "\n";

     cTFI_Triangle & aTri = aTr1->GetTriFromP(aP);

     aSet.NewEqPlanIF(&aTri,true);
}


void GenCodeGrid (cNameSpaceEqF::eModeResidu aMode)
{

      cSetEqFormelles aSet;
      ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);

      cTriangulFormelle * aTr1 = aSet.NewTriangulFormelleUnitaire(2);
      cTriangulFormelle * aTr2 = aSet.NewTriangulFormelleUnitaire(2);

      cRotationFormelle * aRF1 = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);
      cRotationFormelle * aRF2 = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);

      aSet.NewCpleGridEq(*aTr1,*aRF1,*aTr2,*aRF2,aMode,true);
}

void GenCodeAppuiGrid()
{
     cSetEqFormelles aSet;
     ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);
                                                                                             
     cTriangulFormelle * aTri = aSet.NewTriangulFormelleUnitaire(2);
     cRotationFormelle * aRF = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);

     aSet.NewEqAppuiGrid(*aTri,*aRF,true);

}

void GenCodeGrid()
{
      // GenCodeAppuiGrid();
      // GenCodeGrid(cNameSpaceEqF::eResiduCoplan);
      // GenCodeGrid(cNameSpaceEqF::eResiduIm1);
      // GenCodeGrid(cNameSpaceEqF::eResiduIm2);
}

//==================== HOMOGRAPHIE ========================================

void GenCodeEqHom(bool WithDrf,bool InSpaceInit)
{
      cSetEqFormelles aSet;
      cHomogFormelle * aH1 = aSet.NewHomF(cElHomographie::Id(),
                                          cNameSpaceEqF::eHomLibre);
      cHomogFormelle * aH2 = aSet.NewHomF(cElHomographie::Id(),
                                          cNameSpaceEqF::eHomLibre);

      ElDistRadiale_PolynImpair aDist =
                ElDistRadiale_PolynImpair::DistId(1.0,Pt2dr(0,0),5);
      cDistRadialeFormelle * aDRF = aSet.NewDistF(true,true,5,aDist);
      aSet.NewEqHomog(InSpaceInit,*aH1,*aH2,WithDrf ?aDRF : 0,true);
}

void GenCodeEqHom()
{
    GenCodeEqHom(true,false);
    GenCodeEqHom(false,false);
    GenCodeEqHom(false,true);
}
//===========================================================================

void GenCodeCorrelGrid(INT aNbPix,bool Im2MoyVar)
{
      cSetEqFormelles aSet;
      aSet.NewEqCorrelGrid(aNbPix,Im2MoyVar,true);
}

void GenCodeCorrelGrid()
{
      GenCodeCorrelGrid(9,true);
      GenCodeCorrelGrid(9,false);
      GenCodeCorrelGrid(25,true);
      GenCodeCorrelGrid(25,false);
      GenCodeCorrelGrid(49,true);
      GenCodeCorrelGrid(49,false);
     // GenCodeCorrelGrid(81,true);
     // GenCodeCorrelGrid(81,false);
}



//===========================================================================
void GenCodeDiv()
{
    cElCompiledFonc::FoncSetValsEq(0,1,true);
    cElCompiledFonc::FoncSetVar(0,true);
    cElCompiledFonc::FoncFixeNormEucl(0,3,1.0,true);
    cElCompiledFonc::FoncFixeNormEuclVect(0,3,3,1.0,true);

    cElCompiledFonc::RegulD1(0,true);
    cElCompiledFonc::RegulD2(0,true);

    for (INT aK=1 ; aK<4 ; aK++)
    {
        cSetEqFormelles aSet;
	aSet.NewEqLin(aK,aK+12,true); // +12 ou n'importe quoi pourGenCode
    }
}

void GenCodeDiv2()
{
    cElCompiledFonc::FoncFixedScal(0,3,3,0.0,true);
}
//===========================================================================

void GencEqObsRotVect()
{
   cSetEqFormelles aSet;
   aSet.NewEqObsRotVect(0,true);
}

void  GencqCalibCroisee(bool C2M)
{
   
   tParamAFocal aNOPAF;
   {
      cSetEqFormelles aSet;
      cParamIntrinsequeFormel * aPar0 = aSet.NewParamIntrNoDist(C2M,CamIdeale(C2M,aNOPAF));
      aSet.NewEqCalibCroisee(C2M,*aPar0,0,true);
   }

   {
      cSetEqFormelles aSet;
      //  ElDistRadiale_PolynImpair aDRad = ElDistRadiale_PolynImpair::DistId(1e4,Pt2dr(0,0),5);
      cParamIFDistRadiale * aPar1 = aSet.NewIntrDistRad(C2M,CamDRad5(C2M,aNOPAF),3);
      aSet.NewEqCalibCroisee(C2M,*aPar1,0,true);
   }

   {
      cSetEqFormelles aSet;
      // ElDistRadiale_PolynImpair aDRad = ElDistRadiale_PolynImpair::DistId(1e4,Pt2dr(0,0),5);
      // cDistModStdPhpgr aDStd(aDRad);
      cParamIFDistStdPhgr * aPar2 =  aSet.NewIntrDistStdPhgr(C2M,CamPhgrStd(C2M,aNOPAF),3);
      aSet.NewEqCalibCroisee(C2M,*aPar2,0,true);
   }
}
void  GencqCalibCroisee()
{
    GencqCalibCroisee(true);
    GencqCalibCroisee(false);
}

void GenDirecteDistorsion(cNameSpaceEqF::eTypeEqDisDirecre   Usage)
{
   tParamAFocal aNOPAF;
   {
      cSetEqFormelles aSet;
      // ElDistRadiale_PolynImpair aDRad = ElDistRadiale_PolynImpair::DistId(1e4,Pt2dr(0,0),5);
      // A priori C2M comme avant, a voir ....
      cParamIFDistRadiale * aPar1 = aSet.NewIntrDistRad(true,CamDRad5(true,aNOPAF),3);
      aSet.NewEqDirecteDistorsion(*aPar1,Usage,true);
   }
}

void GenDirecteDistorsion()
{
    GenDirecteDistorsion(cNameSpaceEqF::eTEDD_Reformat);
    GenDirecteDistorsion(cNameSpaceEqF::eTEDD_Bayer);
    GenDirecteDistorsion(cNameSpaceEqF::eTEDD_Interp);
}


void   GenCodeSurf()
{
   if (1)
   {
        ElSeg3D aSeg(Pt3dr(0,0,0),Pt3dr(1,0,0));
        cCylindreRevolution aCyl(true,aSeg,Pt3dr(0,1,0));
        cSetEqFormelles  aSet;
        aSet.AllocCylindre(aCyl,true);
   }

   if (1) 
   {
        cSetEqFormelles  aSet; 
        Pt3dr aP(0,0,0);
        cSolBasculeRig aSBR(aP,aP,ElMatrix<double>::Rotation(0,0,0),1);
        aSet.NewEqObsBascult(aSBR,true);
   }
}

//===========================================================================

int main(int argc,char ** argv)
{
     // GenEqPlanInc();
     // GenDirecteDistorsion();
     //   GencqCalibCroisee();
     // GencEqObsRotVect();
      // GenCodeEllipse();
     // GenCodeLaserImage();
     //  GenCodeAppui();
     // GenCodeGrid(); 
      // GenCodeLiaison();
      GenCodeEqHom();
     // GenCodeCorrelGrid();
     // GenCodeDiv();
     // GenCodeDiv2();
      // GenCodeSurf();
}

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
