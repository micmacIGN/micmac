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


/*
*/

#include "all_etal.h"
#include <algorithm>

#include "XML_GEN/all.h"

using  namespace NS_ParamChantierPhotogram;


cCiblePointeScore::cCiblePointeScore
(
    cCamIncEtalonage * aCam,
    REAL aScore,
    INT anId
) :
   mScore (aScore),
   mId    (anId),
   mCam   (aCam)
{
}

bool operator < (const cCiblePointeScore & C1,const cCiblePointeScore & C2)
{
   return C1.mScore > C2.mScore;
}


/************************************************************/
/*                                                          */
/*                      cEtalonnage                         */
/*                                                          */
/************************************************************/


// ElRotation3D cEtalonnage::RotationFromAppui(std::string &) 
// std::list<Appar23> cEtalonnage::StdAppuis(std::string &) 
//
//             => 
//                     voir cCamIncEtalonage.cpp
//

const std::string cEtalonnage::TheNameDradInterm = "DRadInterm";
const std::string cEtalonnage::TheNameDradFinale = "DRadFinale";
const std::string cEtalonnage::TheNamePhgrStdInterm = "PhgrStdInterm";
const std::string cEtalonnage::TheNamePhgrStdFinale = "PhgrStdFinale";

const std::string cEtalonnage::TheNameRotInit = "RotInit";
const std::string cEtalonnage::TheNameRotFinale = "RotFinale";
const std::string cEtalonnage::TheNameRotCouplee = "RotCouplee";

tCStrPtr cEtalonnage::TheRotPoss[NbRotEstim] = 
         {
		 &cEtalonnage::TheNameRotCouplee,
		 &cEtalonnage::TheNameRotFinale,
		 &cEtalonnage::TheNameRotInit
	 };



ElDistRadiale_PolynImpair cEtalonnage::CurDist() const
{
     ElDistRadiale_PolynImpair aDist = PIFDR()->DistCur();

// std::cout << "SSSSSSSSSSSSSSSS "<< aDist.RMax() << "\n";
     Pt2dr CD = FromPN(aDist.Centre());
     aDist = aDist.MapingChScale(mFactNorm);
     aDist.Centre() = CD;

     aDist.ActuRMaxFromDist(Pt2di(mSzIm));

     return aDist;
}

ElDistRadiale_PolynImpair cEtalonnage::CurDistInv(INT aDelta) const
{
     ElDistRadiale_PolynImpair aDist = CurDist();
     return  aDist.DistRadialeInverse(mMaxRay,aDelta);
}

REAL cEtalonnage::CurFoc() const {return  PIFDR()->CurFocale() * mFactNorm;}
Pt2dr cEtalonnage::CurPP() const {return  FromPN(PIFDR()->CurPP());}

REAL  cEtalonnage::NormFoc0()  const {return mNormFoc0;}
Pt2dr cEtalonnage::NormPP0()  const {return mNormPP0;}


cCpleCamEtal * cEtalonnage::CpleMaxDist() const
{

    cCpleCamEtal * aRes = 0;
    REAL DMax = -1;
    for 
    (
        tContCpleCam::const_iterator itC = mCples.begin();
        itC != mCples.end();
        itC++
    )
    {
        REAL aD =  (*itC)->DCopt();
/*
	cout  << "[" <<  (*itC)->Cam1()->Name()  
              << "," <<  (*itC)->Cam2()->Name() << "]"
              << " aD = " << aD << "\n";
*/
        if (aD >DMax)
        {
           DMax = aD;
           aRes = *itC;
        }
    }
    return aRes;
}


cCamIncEtalonage * cEtalonnage::CamFromShrtName(const std::string & aShortName,bool SVP)
{
     for (tContCam::iterator itC=mCams.begin() ; itC!=mCams.end() ; itC++)
     {
	 if ((*itC)->Name() == aShortName)
            return *itC;
     }
     if (! SVP)
     {
        std::cout << "For name =" << aShortName << "\n";
        ELISE_ASSERT(false,"cEtalonnage::CamFromShrtName");
     }
     return 0;
}

cCamIncEtalonage * cEtalonnage::AddCam
		  (
                       const std::string & aNameTiff,
                       const std::string & aShortName,
		       bool                PointeCanBeVide,
                       const std::string & aNamePointesFull
	          )
{
    cCamIncEtalonage * aRes = CamFromShrtName(aShortName,true);
    if (aRes) 
        return aRes;

/*
     for (tContCam::iterator itC=mCams.begin() ; itC!=mCams.end() ; itC++)
     {
	 if ((*itC)->Name() == aShortName)
            return *itC;
     }
*/
     cCamIncEtalonage * pCam = new cCamIncEtalonage
	                          (
				      mCams.size(),
				      *this,
				      aNameTiff,
				      aShortName,
				      PointeCanBeVide,
				      aNamePointesFull
				   );

     pCam->InitOrient
     (
          mParamIFGen,
	  mCams.empty()  ? 0 : mCams.back()
     );

     if ((mParamIFHom == 0) && (mParamIFPol == 0))
     {
         for (tContCam::iterator itC=mCams.begin() ; itC!=mCams.end() ; itC++)
         {
              mCples.push_back
              (
	           new cCpleCamEtal(mParam.ModeC2M(),Set(),pCam,*itC)
              );
         }
     }
     else
	     cout << "PAS DE CPLE CAM EN MODE HOMOGRAPHIE ou POLXY\n";

     mCams.push_back(pCam);

     return pCam;
}

std::string cEtalonnage::NamePointeInit(const std::string & aName)
{
	// return mDir + "PointeInitIm." + aName;
	return mDir + "Pointe" + mParam.PointeInitIm() +    "." + aName;
}

std::string cEtalonnage::NameTiffIm(const std::string & aName)
{
	return  mParam.NameTiff(aName);

}

std::string cEtalonnage::NamePointeResult(const std::string & aName,bool Interm,bool Compl)
{
	return    mDir 
		+ std::string(Compl  ? "Detail" : "")
                + std::string("Pointe") 
		+ std::string(Interm ? "Interm" : "Final")
                + std::string(".") 
                + aName;
}



void cEtalonnage::InitNormCmaIdeale()
{
    delete mNorm;
    mNorm = cCoordNormalizer::NormCamId(mFactNorm,mDecNorm);
}

void cEtalonnage::InitNormIdentite()
{
    delete mNorm;
    mNorm = cCoordNormalizer::NormCamId(1.0,Pt2dr(0,0));
}




void cEtalonnage::InitNormGrid(const std::string & aName)
{
	mNorm = cCoordNormalizer::NormalizerGrid(mDir+aName);
}

void cEtalonnage::InitNormDrad(const std::string & aName)
{
     std::vector<double>   NoParAdd;
     std::string aNameFull = mDir + aName + ".dat";
     ELISE_fp aFileBin(aNameFull.c_str(),ELISE_fp::READ);

     REAL aFoc = aFileBin.read((REAL *)0);
     Pt2dr aPP =  aFileBin.read((Pt2dr *)0);
     ElDistRadiale_PolynImpair aDist= ElDistRadiale_PolynImpair::read(aFileBin);

     mNorm = cCoordNormalizer::NormCamDRad(mParam.ModeC2M(),aFoc,aPP,aDist);

     // ElDistRadiale_PolynImpair aDInv = aDist.DistRadialeInverse(mMaxRay,0);
     pCamDRad = new cCamStenopeDistRadPol(mParam.ModeC2M(),aFoc,aPP,aDist,NoParAdd);
}


void cEtalonnage::AddEqCam(cCamIncEtalonage & aCam,REAL anEc)
{
     cCameraFormelle & aCF = aCam.CF();
     tLPointes & aL = aCam.SetPointes().Pointes();

     if (aL.size() <= 6)
     {
	     cout << "For Cam [" << aCam.Name() 
                  <<  " insuficient number pointes : " << aL.size()
                  << "\n";
	     ELISE_ASSERT(false,"cEtalonnage::AddEqCam");
     }

     REAL SEc = 0;
     REAL SPds = 0;
     REAL DMax = 0.0;
     // int  IDMax = -1;
     for 
     (
          tLPointes::iterator iT = aL.begin();
	  iT != aL.end();
	  iT++
     )
     {
	  REAL aPds =  iT->Pds();
          Pt2dr Ec = aCF.ResiduAppui(iT->PosTer(),iT->PosIm());
          REAL aD = euclid(Ec);

          if ( aD > anEc)
          {
               cout << aCam.Name() << " " 
                    << iT->Cible().Ind()  << " "
                    << aD * mFactNorm << "\n";
          }
          else
          {
	         aCF.AddAppui(iT->PosTer(),iT->PosIm(),aPds);
	         AddErreur(mFactNorm*euclid(Ec),aPds);
                 mVCPS.push_back(cCiblePointeScore(&aCam,euclid(Ec),iT->Cible().Ind()));
          // Pt2dr Ec = aCF.ResiduAppui(iT->PosTer(),iT->PosIm());
	         if (WGlob())
	         {
	              // INT Ind = iT->Cible().Ind();
                      WGlob()->draw_seg
                      (
                         FromPN(iT->PosIm()),
                         FromPN(iT->PosIm()+Ec*500),
                         WGlob()->pdisc()(P8COL::red)
                      );
                      WGlob()->draw_circle_abs
                      (
		           FromPN(iT->PosIm()),
		           2.0,
		           iT->Cible().Ind() == 158 ? 
		               WGlob()->prgb()(255,0,0) :
		               WGlob()->prgb()(0,0,255)
                      );
	         }
		 double aDEuc = euclid(Ec);
		 if (aDEuc > DMax)
		 {
		     // IDMax = iT->Cible().Ind() ;
		     DMax = aDEuc;
                     // ElSetMax(DMax,euclid(Ec));
		 }
                 SEc +=  aDEuc;
                 SPds += 1;
       }
     }

     
/*
     cout << "CAM= "<<  aCam.Name() << " "
          << "NB = " << SPds 
	  << "DIST MOY = " << (SEc /SPds)  * mFactNorm 
          << " D MAX = " << DMax * mFactNorm
	  << " ; IDMax=" << IDMax
	  << "\n";
*/
}

void cEtalonnage::AddEqAllCam(bool Sauv,REAL anEc)
{
    if (WGlob()) 
	WGlob()->clear();
    Set().AddContrainte(mParamIFGen-> StdContraintes(),true);

    Set().SetPhaseEquation();

    for (tContCam::iterator iTC = mCams.begin() ; iTC != mCams.end() ; iTC++)
        AddEqCam(**iTC,anEc);

    Set().SolveResetUpdate();

    if (Sauv)
    {
        for 
        (
	     tContCam::iterator iTC = mCams.begin() ; 
	     iTC != mCams.end() ; 
	     iTC++
	)
             (*iTC)->SauvEOR();
    }
}


Pt2dr cEtalonnage::ToPN(Pt2dr aP) const
{
	ELISE_ASSERT(mNorm!=0,"cEtalonnage::ToPN");
	return mNorm->ToCoordNorm(aP);
}

Pt2dr cEtalonnage::FromPN(Pt2dr aP) const
{
	ELISE_ASSERT(mNorm!=0,"cEtalonnage::ToPN");
	return mNorm->ToCoordIm(aP);
}

Pt2dr cEtalonnage::SzIm()
{
   return mSzIm;
}

cEtalonnage::~cEtalonnage () 
{
    if (WGlob()) 
       WGlob()->clik_in();
}

cParamIFDistRadiale * cEtalonnage::PIFDR() const
{
     ELISE_ASSERT(mParamIFDR!=0,"No mParamIFDR");
     return mParamIFDR;
}

cEtalonnage::cEtalonnage 
(
    bool               isLastEtape,
    const cParamEtal & aParam,
    cBlockEtal *       aBlock,
    const std::string  &  aModeDist
) :
   mParam            (aParam),
   mIsLastEtape      (isLastEtape),
   mDoNormPt         (! mIsLastEtape),
   mDir              (mParam.Directory()),
   mSzIm             (mParam.SzIm()),
   mFocAPriori       (mParam.FocaleInit()),
   mMil              (mSzIm/2.0),
   mNorm             (0),
   mNormId           (0),
   mMaxRay           (euclid(mMil)),
   mNormFoc0         (mDoNormPt ? (1.0): mFocAPriori),
   mNormPP0          (mDoNormPt ? Pt2dr(0,0): mMil),
   mFactNorm         ( (!mDoNormPt)  ? (1.0): mFocAPriori),
   mDecNorm          ((!mDoNormPt) ? Pt2dr(0,0): mMil),
   mPol              (*cPolygoneEtal::FromName(mParam.NameCible3DPolygone(),&mParam)),
   pBlock            (aBlock ?  aBlock : new cBlockEtal(mIsLastEtape,aParam,false)),
   mBloquee          (false),
   mParamIFDR        (0),
   mParamIFHom       (0),
   mParamPhgrStd     (0),
   mParamIFPol       (0),
   mStats            (10000),
   mWGlob            (0),
   pEtalRattachement (0),
   pCamDRad          (0),
   mModeDist         (aModeDist)
{
   std::vector<double> NoParAdd;
   cTplValGesInit<std::string> aUseless;
   mICNM  =cInterfChantierNameManipulateur::StdAlloc(0,0,mDir,aUseless);



      bool aC2M =  mParam.ModeC2M();
      REAL RMax = mMaxRay / mFactNorm;


      if ((aModeDist == "Hom") || (aModeDist == "NoDist"))
      {
	      ElPackHomologue aPack;
	      // aPack.add(ElCplePtsHomologues(Pt2dr(0,0),Pt2dr(0.5,0.03)));
	      // cElHomographie aH (aPack,true);
	      cDistHomographie  aD(aPack,true);
              cCamStenopeDistHomogr * aCamH = new cCamStenopeDistHomogr
	                                         (
						     aC2M,
						     mNormFoc0,mNormPP0,
						     aD,
                                                     NoParAdd
						 );
	      mParamIFHom = Set().NewDistHomF(aC2M,aCamH,cNameSpaceEqF::eHomFigee);;
	      mParamIFGen = mParamIFHom;
              if (aModeDist == "NoDist")
                  mBloquee = true;
      }
      else if 
           (
                   (aModeDist == "Pol3") 
                || (aModeDist == "Pol4") 
                || (aModeDist == "Pol5") 
                || (aModeDist == "Pol6") 
                || (aModeDist == "Pol7")
           )
      {
	       ElPackHomologue aPack;
	      // aPack.add(ElCplePtsHomologues(Pt2dr(0,0),Pt2dr(0.5,0.03)));
	      // 0.119954  D5  0.135832 rouge  
	      // 0.112924  D7   
	      // 0.162113  D3
	      mDegPolXY = 5;
              if (aModeDist == "Pol3")
	          mDegPolXY = 3;
              if ((aModeDist == "Pol7") || (aModeDist == "Pol6"))
	          mDegPolXY = 7;
              ElDistortionPolynomiale aDistP = 
	              ElDistortionPolynomiale::DistId(mDegPolXY,1.0);
	      cCamStenopeDistPolyn * aCamP = new cCamStenopeDistPolyn
	                                         (
						    aC2M,
						    mNormFoc0,mNormPP0,
						    aDistP,
                                                    NoParAdd
						 );

	      mParamIFPol = Set().NewIntrPolyn(aC2M,aCamP);
              if (aModeDist == "Pol4")
	          mDegPolXY = 4;
              if (aModeDist == "Pol6")
	          mDegPolXY = 6;
	      mParamIFPol->SetFige(0);
	      mParamIFGen = mParamIFPol;
            mParamIFGen->SetPPFree(false);
            mParamIFGen->SetFocFree(false);
      }
      else if ((aModeDist == "") || (aModeDist== "DRad"))
      {
           cCamStenopeDistRadPol * aCamDRP = new cCamStenopeDistRadPol
	                                         (
						      aC2M,
						      mNormFoc0,mNormPP0,
						      ElDistRadiale_PolynImpair::DistId(1.1*RMax,mNormPP0,5),
                                                      NoParAdd
						 );
            aCamDRP->SetSz(Pt2di(mSzIm));
            mParamIFDR  = Set().NewIntrDistRad(aC2M,aCamDRP,0);
            mParamIFGen = mParamIFDR;
            mParamIFDR->SetLibertePPAndCDist(false,false);
            mParamIFDR->SetFocFree(false);
      }
      else if (aModeDist == "PhgrStd")
      {
            cDistModStdPhpgr aDist(ElDistRadiale_PolynImpair::DistId(1.1*RMax,mNormPP0,5));
            cCamStenopeModStdPhpgr * aCam = new cCamStenopeModStdPhpgr(aC2M,mNormFoc0,mNormPP0,aDist,NoParAdd);

            aCam->SetSz(Pt2di(mSzIm));
	    mParamPhgrStd =  Set().NewIntrDistStdPhgr(aC2M,aCam,0);
            mParamPhgrStd->SetParam_Aff_Fige();
            mParamPhgrStd->SetParam_Dec_Fige();

            mParamPhgrStd->SetLibertePPAndCDist(false,false);
            mParamPhgrStd->SetFocFree(false);
            mParamIFGen = mParamPhgrStd;
      }
      else
      {
	      ELISE_ASSERT(false,"Unknown ModeDist");
      }


      if (mParamIFDR)
      {
         PIFDR()->SetLibertePPAndCDist(false,false);

         PIFDR()->SetFocFree(false);
         PIFDR()->SetDRFDegreFige(0);
      }

      mNormId = cCoordNormalizer::NormCamId(mFactNorm,mDecNorm);

}

Video_Win *  cEtalonnage::WGlob()
{
   if (mWGlob) 
      return mWGlob;
   if (mParam.Zoom() <=0) 
      return 0;

   INT Sz = 600;
   mWGlob = Video_Win::PtrWStd(Pt2di(Sz,Sz));
   REAL Ratio = REAL(Sz) / REAL (dist8(mSzIm));
   mWGlob = mWGlob->PtrChc(Pt2dr(0,0),Pt2dr(Ratio,Ratio));
   return mWGlob;
}

REAL cEtalonnage::FocAPriori() const
{
   return mFocAPriori;
}

const cPolygoneEtal &  cEtalonnage::Polygone() const
{
  return mPol;
}

const cParamEtal & cEtalonnage::Param() const
{
      return mParam;
}

void cEtalonnage::CalculLiaison(bool ParamFree)
{

      PIFDR()->SetValInitOnValCur();
      PIFDR()->SetFocFree(ParamFree);
      PIFDR()->SetLibertePPAndCDist(ParamFree,ParamFree);
      PIFDR()->SetDRFDegreFige(ParamFree ? 3 : 0);

      // CA MARCHE PAS
      for (tContCam::iterator iTC = mCams.begin() ; iTC != mCams.end() ; iTC++)
          (*iTC)->CF().RF().SetValInitOnValCur();

      for (INT aK=0 ; aK< 40 ; aK++)
      {
	  ResetErreur();
          Set().AddContrainte(mParamIFGen->StdContraintes(),true);

          for (tContCam::iterator iTC = mCams.begin() ; iTC != mCams.end() ; iTC++)
              Set().AddContrainte((*iTC)->CF().RF().StdContraintes(),true);

          for 
          (
              tContCpleCam::iterator itC = mCples.begin();
              itC != mCples.end();
              itC++
          )
          {
		  (*itC)->AddLiaisons(*this);
          }
          Set().SolveResetUpdate();
	  cout << "ERR MOY " << mSomErr/mSomPds << "\n";
      cout << "FOC = " << PIFDR()->CurFocale() * mFactNorm << "\n";
      cout << "PP = " << FromPN(PIFDR()->CurPP()) << "\n";
      cout << "CDIST = " << FromPN(PIFDR()->DistCur().Centre()) << "\n";
      }
}

cEtalonnage *  cEtalonnage::EtalRattSvp()
{
    if (! pEtalRattachement)
    {
         if (! mParam.HasFileRat())
             return 0;
	 /*
         std::string  aNR = mParam.NameFileRatt();
	 char * (tab[2]);
	 tab[0] = 0;
	 tab[1] = const_cast<char *>(aNR.c_str());
	 cParamEtal  aParam(2,tab);
	 pEtalRattachement = new cEtalonnage(aParam);
	 */
	 pEtalRattachement = new cEtalonnage(mIsLastEtape,mParam.ParamRatt());
    }
    return pEtalRattachement;
}
cEtalonnage &  cEtalonnage::EtalRatt()
{
   cEtalonnage * pE = EtalRattSvp();
   ELISE_ASSERT(pE!=0,"cEtalonnage::EtalRatt");
   return *pE;
}

cSetEqFormelles & cEtalonnage::Set()
{
    return pBlock->Set();
}


std::string cEtalonnage::NameRot(const std::string & Etape,const std::string & Cam)
{
    return mParam.Directory() + Etape + "." + Cam;
}

void cEtalonnage::GenImageSaisie()
{
   if (! mParam.HasGenImageSaisie()) 
      return;
   cCamIncEtalonage * aCam=0;
   for (tContCam::iterator iTC = mCams.begin() ; iTC != mCams.end() ; iTC++)
   {
       /*
       std::string aNR = NameRot(aNameSauvRot,(*iTC)->Name());
       ElRotation3D aR = (*iTC)->CF().CurRot().inv();
       SauvFile(aR,aNR);
       */
       if ((*iTC)->Name() ==  mParam.ShortNameImageGenImageSaisie())
           aCam = *iTC;
   }
   ELISE_ASSERT(aCam!=0,"cEtalonnage::GenImageSaisie");
   CamStenope * aCS = aCam->CurCamGen();
   std::string aNamePol = mParam.NamePolygGenImageSaisie();
   cPolygoneEtal * aPol = cPolygoneEtal::FromName(aNamePol,&mParam);
  
   std::string aNameI = mParam.FullNameImageGenImageSaisie ();
   FILE * aFP = FopenNN(aNameI,"w","File Im in GenImageSaisie");
   for 
   (
       cPolygoneEtal::tContCible::const_iterator itC= aPol->ListeCible().begin();
       itC!= aPol->ListeCible().end();
       itC++
   )
   {
       Pt2dr aPIm = aCS->R3toF2((*itC)->Pos());
       aPIm = mNorm->ToCoordIm(aPIm)/mParam.ZoomGenImageSaisie();
       if ((aPIm.x>0) && (aPIm.y>0) && (aPIm.x<mSzIm.x) && (aPIm.y<mSzIm.y))
       {
           std::cout << (*itC)->Ind() << ":::" << aPIm << "\n";
           fprintf(aFP,"%d %f %f\n",(*itC)->Ind(),aPIm.x,aPIm.y);
       }
   }
   std::cout << aNameI << "\n";
   std::cout << "-- " << aCam->Name() <<  "  -----\n";getchar();
   ElFclose(aFP);
}

void cEtalonnage::SauvAppuisFlottant()
{
     const  cPolygoneEtal::tContCible  &  aLC = mPol.ListeCible(); 
     cDicoAppuisFlottant aDAF;

     for (cPolygoneEtal::tContCible::const_iterator itC=aLC.begin() ; itC!=aLC.end() ; itC++)
     {
          cOneAppuisDAF oneAF;

          oneAF.Pt() =  (*itC)->Pos();
          oneAF.NamePt() =  ToString((*itC)->Ind());
          oneAF.Incertitude() = Pt3dr(1,1,1);
          aDAF.OneAppuisDAF().push_back(oneAF);
     }
     MakeFileXML(aDAF,mICNM->Dir()+"DicoAppuisFlottants.xml");

     cSetOfMesureAppuisFlottants aSMAF;

     for (tContCam::iterator iTC = mCams.begin() ; iTC != mCams.end() ; iTC++)
     {
          cMesureAppuiFlottant1Im aMAF;
          aMAF.NameIm() =  mParam.NameTiffSsDir((*iTC)->Name());

          cListeAppuis1Im  aLIm = El2Xml((*iTC)->AppuisInit(),(*iTC)->IndAppuisInit());

          for (std::list<cMesureAppuis>::iterator itM=aLIm.Mesures().begin() ; itM!=aLIm.Mesures().end() ; itM++)
          {
                cOneMesureAF1I  aM;
                aM.PtIm() = itM->Im();
                aM.NamePt() = ToString(itM->Num().Val());
                aMAF.OneMesureAF1I().push_back(aM);
          }


          aSMAF.MesureAppuiFlottant1Im().push_back(aMAF);
     }
     MakeFileXML(aSMAF,mICNM->Dir()+"MesuresAppuisFlottants.xml");
}

   // anEt.CalculModeleRadiale(false,"",false,false,aParam.AllImagesCibles(),false);
void cEtalonnage::CalculModeleRadiale
                  (
		           bool Sauv,
		           const std::string & aNameSauvRot,
		           bool ModeleFige,
			   bool FreeRad,
                           const std::vector<string> & mVNames,
                           bool PtsInterm
                  )

{
      for 
      (
           std::vector<string>::const_iterator iTN = mVNames.begin();
           iTN != mVNames.end();
	   iTN++
      )
      {
	    const std::string & aName = *iTN;
            cCamIncEtalonage * pCam = AddCam
            (
                NameTiffIm(aName),
                aName,
                false,
                NamePointeResult(aName,PtsInterm,false)
            );

	    pCam->SetPointes().RemoveCibles(mParam.CiblesRejetees());
      }
      Set().SetClosed();

      {
         std::string aName  =    mParam.Directory() + std::string("GlobPointes")
	                      +  std::string(PtsInterm ? "Interm" : "Final")
			      +  std::string(".pk1");
     
         SauvPointes(aName);
      }

      if (mParam.ParamRechDRad().mUseCI==eUCI_Only)
      {
          ModeleFige = true;
      }

      INT NbStep = 30;
      for (INT aK=0 ; aK< NbStep ; aK++)
      {
          mVCPS.clear();
          REAL anEc = (aK==0) ? 1e3 :  (ECT() * mParam.SeuilCoupure()) ;
	  ResetErreur();
          AddEqAllCam((aK==(NbStep-1))&&Sauv&&(!PtsInterm),anEc);
          cout << "STEP = " << aK <<  " E2= " << ECT() * mFactNorm << "\n";
	  if (mParamIFDR)
	  {
             if (! ModeleFige)
             {
                  INT aD = mParam.DegDist();
	          if (aK >= 3)
                      PIFDR()->SetFocFree(true);

                  if (FreeRad)
		  {
	              if (aK >= 5)
	              {
                          // INT aD = mParam.DegDist();
                           PIFDR()->SetDRFDegreFige(ElMin(aD,(aK-5)/2));
                      }

	              if (aK >= 15)
                          PIFDR()->SetCDistPPLie();

	              if (aK >= 20)
		      {
                        if (mParam.CDistLibre(true))
                           PIFDR()->SetLibertePPAndCDist(true,true);
                      }
                  }
               

/*
	          if (aK >= 3)
	          {
                     if (FreeRad)
                         PIFDR()->SetDRFDegreFige(ElMin(aD,2));
                     PIFDR()->SetFocFree(true);
	          }
	          if (aK >= 6)
	          {
                     if (FreeRad)
                        PIFDR()->SetDRFDegreFige(ElMin(aD,3));
	          }
	          if (aK >= 9)
	          {
                     PIFDR()->SetCDistPPLie();
	          }
	          if (aK >= 12)
	          {

                     if (FreeRad)
                        PIFDR()->SetDRFDegreFige(ElMin(aD,5));
	          }
                  if (aK >=15)
                  {
                     if (FreeRad)
		     {
                        if (mParam.CDistLibre())
                           PIFDR()->SetLibertePPAndCDist(true,true);
		     }
	          }
*/
             }
	  }

          if (mParamPhgrStd)
          {
	       if (aK >= 4)
                  mParamPhgrStd->SetFocFree(true);

	       if (aK >= 5)
	       {
                   INT aD = mParam.DegDist();
                   mParamPhgrStd->SetDRFDegreFige(ElMin(aD,3));
               }

	       if (aK >= 7)
                  mParamPhgrStd->SetCDistPPLie();
               
               if (aK >= 10)
                   mParamPhgrStd->SetParam_Dec_Free();
               if (aK >= 12)
                   mParamPhgrStd->SetParam_Aff_Free();

	       if ((aK >= 15) && (mParam.CDistLibre(false)))
               {
                   mParamPhgrStd->SetLibertePPAndCDist(true,true);
               }
          }
	  if (mParamIFHom)
	  {
             if (aK > 6) 
             {
                if (! mBloquee)
                   mParamIFHom->SetStdBloqueRot();
             }
	  }
	  if (mParamIFPol)
	  {
             if (aK > 6) 
                mParamIFPol->SetFige(ElMin(mDegPolXY,aK-5));
	  }
      }
      if (mParamIFDR)
      {
          cout << "FOC = " << PIFDR()->CurFocale() * mFactNorm << "\n";
          cout << "PP = " << FromPN(PIFDR()->CurPP()) << "\n";
          cout << "CDIST = " << FromPN(PIFDR()->DistCur().Centre()) << "\n";
      }

      cCpleCamEtal * aCpleMax = CpleMaxDist();
      if (aCpleMax)
      {
          cout << "MAX DCOPT = " << aCpleMax->DCopt() << "\n";
      }

      if ( Sauv)
      {
         if ((! PtsInterm) && (mParam.ExportAppuisAsDico().IsInit()))
	 {
            ExportAsDico(mParam.ExportAppuisAsDico().Val());
         }
         SauvDRad
         (
            PtsInterm ? TheNameDradInterm : TheNameDradFinale,
            PtsInterm ? TheNamePhgrStdInterm : TheNamePhgrStdFinale
         );


         


         SauvAppuisFlottant();
         for (tContCam::iterator iTC = mCams.begin() ; iTC != mCams.end() ; iTC++)
	 {
             std::string aNR = NameRot(aNameSauvRot,(*iTC)->Name());
	     ElRotation3D aR = (*iTC)->CF().CurRot().inv();
	     SauvFile(aR,aNR);
	     // std::string aNameFile = mDir+"Orient_"+  mParam.NameCamera() + "_" + (*iTC)->Name()+".xml";
// std::cout << "NAME TIFF ==========" << mParam.NameTiff((*iTC)->Name()) << "\n";
             std::string  aNameFullIm = mParam.NameTiff((*iTC)->Name());
             std::string aDir,aNameIm;
             SplitDirAndFile(aDir,aNameIm,aNameFullIm);

             std::string aKey = mParam.KeyExportOri();
             std::string aNameOri = mICNM->Assoc1To1(aKey,aNameIm,true);
             // std::string aN2 = mICNM->Assoc1To1(aKey,aNameOri,false);
// std::cout << aNameIm << " " << aN2 << "\n";
              // ELISE_ASSERT(aN2==aNameIm,"Coherence pb with Key-Assoc-Im2AppuiOri-Polygone");

             std::string aNameFile = mDir+ aNameOri;

           // std::cout << aNameOri << "  // " << aN2 << "\n"; 


	     XML_SauvFile(aR.inv(),aNameFile,"CalibPolygone",true);
	     // std::list<Appar23>  aL32 = (*iTC)->StdAppuis(false);
	      // cListeAppuis1Im  aLAI  = El2Xml((*iTC)->StdAppuis(false));
	     //  AddFileXML( El2Xml((*iTC)->AppuisInit()),aNameFile);
	     AddFileXML( El2Xml((*iTC)->AppuisInit(),(*iTC)->IndAppuisInit()),aNameFile);

	     // R est la rotation qui transforme les coordonnees monde en
	     // coordonnees camera :  aR.IRecAff(Pt3dr(0,0,0))  est un centre optique
	     
	     /*
	     std::cout << "  " << aR.ImAff(Pt3dr(0,0,0)) 
	                 << " " <<  aR.IRecAff(Pt3dr(0,0,0))<< "AAAA\n"; getchar();
			 */
	 }
      }
      if (mModeDist != "")
         SauvXML(mModeDist);

      std::sort(mVCPS.begin(), mVCPS.end());
      INT aNb = ElMin(200,INT(mVCPS.size()));
      for (INT aK =aNb-1; aK >= 0 ; aK--)
	      cout << (mVCPS[aK].mScore * mFactNorm) 
		   << " " << (mVCPS[aK].mScore * mFactNorm) * (mVCPS.size()/REAL(mVCPS.size()-aK))
		   << " " << mVCPS[aK].mId
		   << " " << mVCPS[aK].mCam->Name()
		   << "\n";
     GenImageSaisie();

     UseParamCompl();
}

void cEtalonnage::AddErreur(REAL anEcart,REAL aPds)
{
     mSomErr +=  anEcart * aPds;
     mSomE2  +=  ElSquare(anEcart) * aPds;
     mSomPds +=  aPds;
     mStats.AddErreur(anEcart);
}
REAL cEtalonnage::ECT() const
{
   return sqrt(mSomE2/mSomPds) / mFactNorm;
}

void cEtalonnage::ResetErreur()
{
	mSomErr = 0;
	mSomE2 = 0;
	mSomPds = 0;
	mStats.Reset();
}

void cEtalonnage::Do8Bits(const std::vector<string> & mVNames)
{
    if (mParam.CalledByItsef() || (!mParam.DoSift()))
       return;

     System("make bin/to8Bits");
          
     cEl_GPAO aGPAO;

     cElTask &  aGlobTask = aGPAO.GetOrCreate("all","");

     for 
     (
           std::vector<string>::const_iterator iTN = mVNames.begin();
           iTN != mVNames.end();
	   iTN++
     )
     {
         if(mParam.DoSift(*iTN))
         {
               std::string  aCom  =
                                 "bin/to8Bits " 
                                 + mParam.NameTiff(*iTN)
                                 + " AdaptMinMax=1";

               std::string aTF =  StdPrefix(mParam.NameTiff(*iTN)) + "_8Bits.tif";
               // std::cout << aCom << "\n";

               cElTask & aTK = aGPAO.NewTask(aTF,aCom);
                aGlobTask.AddDep(aTK);
         }
      }

      std::string aNameMk = "MakeEtalRechCibles";

      aGPAO.GenerateMakeFile(aNameMk);

      std::string aComMake = "make all -f " + aNameMk + " -j" + ToString(ElMax(2,mParam.ByProcess()));
      System(aComMake);
}

void cEtalonnage::RechercheCibles
     (
          int argc,char ** argv,
          const std::vector<string> & mVNames,
	  const cParamRechCible & aPRC
     ) 
{ 

    Do8Bits(mVNames);

//
      bool ProcessMaitre  = false;
      if (mParam.ByProcess() && (!mParam.CalledByItsef()))
      {
          ProcessMaitre  = true;
          cEl_GPAO aGPAO;

          cElTask &  aGlobTask = aGPAO.GetOrCreate("all","");

          for 
          (
           std::vector<string>::const_iterator iTN = mVNames.begin();
           iTN != mVNames.end();
	   iTN++
          )
          {
               std::string aCom =    ToCommande(argc,argv)
                      +  " CalledByItsef=1"
                      +  " Im=" +*iTN;
               std::cout << aCom << "\n";

               cElTask & aTK = aGPAO.NewTask(*iTN,aCom);
                aGlobTask.AddDep(aTK);
          }

          std::string aNameMk = "MakeEtalRechCibles";

          aGPAO.GenerateMakeFile(aNameMk);

          std::string aComMake = "make all -f " + aNameMk + " -j" + ToString(mParam.ByProcess());
          System(aComMake);
          // return;
      }
      cCibleRechImage aCRI(*this,aPRC.mSzW,mParam.Zoom());

      for 
      (
           std::vector<string>::const_iterator iTN = mVNames.begin();
           iTN != mVNames.end();
	   iTN++
      )
      {
	  const std::string & aName = *iTN;
	  if ((mParam.ImDeTest() == "")|| (mParam.ImDeTest()== aName))
	  {
            cCamIncEtalonage * pCam =
                               AddCam
                               (
	                          NameTiffIm(aName),
				  aName,
				  true,
				  NamePointeInit(aName)
                               );
	     if (pCam && (!ProcessMaitre))
	     {
		 cout << "BEGIN  " << aName << "\n";
                 cSetHypDetectCible aSet(mParam,
				         mPol,*pCam,aPRC.mDistConf,
				         aPRC.mEllipseConf,
					 mParam.CibDirU(),
					 mParam.CibDirV()
					 );
                 const std::list<cHypDetectCible *> lH = aSet.Hyps();
		 cout << "DETECTION DANS IMAGE " << aName << "\n";
	         INT aK=0;
                 for 
                 (
                      std::list<cHypDetectCible *>::const_iterator iT = lH.begin();
	              iT != lH.end() ;
	              iT++
                 )
                 {
                    cHypDetectCible  & anHyp = **iT;
	            if (
		             (aPRC.mUseCI != eUCI_Only)
		          && (anHyp.OkForDetectInitiale(aPRC.mStepInit))
		       )
		    {
                       aCRI.RechercheImage(anHyp);
		       aK++;
		    }
                 }

		 if (mParam.CibleDeTest() == -1)
                 {
	             aSet.SauvFile
		     (
		         *this,
		         pCam->Pointes(),
		         aPRC,
		         NamePointeResult(aName,aPRC.mStepInit,false),
			 false
                     );
	             aSet.SauvFile
		     (
		           *this,
		           pCam->Pointes(),
		           aPRC,
		           NamePointeResult(aName,aPRC.mStepInit,true),
			   true
                     );
                 }
	     }

          }
      }
}

void cEtalonnage::XmlSauvPointe()
{
     for (tContCam::iterator iTC = mCams.begin() ; iTC != mCams.end() ; iTC++)
     {
             std::string  aNameFullIm = mParam.NameTiff((*iTC)->Name());
             std::string aDir,aNameIm;
             SplitDirAndFile(aDir,aNameIm,aNameFullIm);

             std::string aKey = "Key-Assoc-Appui-Init-Etalonage";
             std::string aNameOri = mICNM->Assoc1To1(aKey,aNameIm,true);

             std::string aNameFile = mDir+ aNameOri;


             MakeFileXML(El2Xml((*iTC)->AppuisInit(),(*iTC)->IndAppuisInit()),aNameFile);
      }
}

	     
void cEtalonnage::RechercheCiblesInit(int argc,char ** argv)
{
     cParamEtal   aParam(argc,argv);
     
     {
        cSauvegardeSetString  aSND;
        
        const std::vector<std::string>  &  aLN = aParam.AllImagesCibles();
        for 
        (
            std::vector<std::string>::const_iterator itS=aLN.begin();
            itS!=aLN.end();
            itS++
        )
        {
            aSND.Name().push_back(aParam.NameTiffSsDir(*itS) );
        }
        MakeFileXML(aSND,aParam.Directory()+"ListeNamesIm.xml");
     }



     cEtalonnage  anEt(false,aParam);
     anEt.InitNormCmaIdeale();
     anEt.RechercheCibles(argc,argv,aParam.ImagesInit(),aParam.ParamRechInit());
     if (! aParam.CalledByItsef())
         anEt.XmlSauvPointe();
}

void cEtalonnage::RechercheCiblesDRad(int argc,char ** argv)
{
     cParamEtal   aParam(argc,argv);
     cEtalonnage  anEt(false,aParam);
     anEt.InitNormDrad(TheNameDradInterm);
     anEt.RechercheCibles(argc,argv,aParam.AllImagesCibles(),aParam.ParamRechDRad());
}


void cEtalonnage::TestGrid(int argc,char ** argv)
{
   ELISE_ASSERT(argc==3,"Bad Arg Number in TestHomOnGrid");
   cParamEtal   aParam(2,argv);
   cEtalonnage  anEt(false,aParam,0,"");
   if (std::string(argv[2]) == std::string("NoGrid"))
     anEt.InitNormCmaIdeale();
   else
     anEt.InitNormGrid(argv[2]);
   anEt.CalculModeleRadiale(false,"",true,false,aParam.AllImagesCibles(),false);
   // anEt.CalculLiaison(false);
}

void cEtalonnage::TestHomOnGrid(int argc,char ** argv)
{
   cParamEtal   aParam(2,argv);

   std::string aKindIn;
   std::string aKindOut;
   INT XML = 1;
   ElInitArgMain
   (
        argc-1,argv+1,
        LArgMain()      << EAM(aKindIn)
                        << EAM(aKindOut) ,
        LArgMain() << EAM(XML,"XML",true)
   );

   bool NoGr = (aKindIn == std::string("NoGrid"));

   cEtalonnage  anEt(NoGr,aParam,0,aKindOut);
   // anEt.TestVisuFTM();

   if (NoGr)
     anEt.InitNormCmaIdeale();
   else
     anEt.InitNormGrid(aKindIn);
   anEt.CalculModeleRadiale(false,"",false,true,aParam.AllImagesCibles(),false);

   
   anEt.SauvGrid
   (
       aParam.StepGridXML(),//  10.0,
           std::string("GRID_")
         + aKindIn
         + std::string("_")
         + aKindOut,
         XML
    );
   // anEt.CalculLiaison(false);
}


/*
void cEtalonnage::TestHomOnDRad(int argc,char ** argv)
{
   cParamEtal   aParam(argc,argv);
   cEtalonnage  anEt(aParam,0,"Hom");
   anEt.InitNormDrad(TheNameDradFinale);
   anEt.CalculModeleRadiale(false,"",false,false,aParam.AllImagesCibles(),false);
   // anEt.CalculLiaison(false);
}

void cEtalonnage::TestPolOnDRad(int argc,char ** argv)
{
   cParamEtal   aParam(argc,argv);
   cEtalonnage  anEt(aParam,0,"Pol5");
   anEt.InitNormDrad(TheNameDradFinale);
   anEt.CalculModeleRadiale(false,"",false,false,aParam.AllImagesCibles(),false);

   anEt.SauvGrid(20.0,"GridPol5",false);
   // anEt.CalculLiaison(false);
}

*/

void cEtalonnage::CalculModeleRadialeInit(int argc,char ** argv)
{
   cParamEtal   aParam(argc,argv);
   cEtalonnage  anEt(false,aParam);
   anEt.InitNormCmaIdeale();
   // anEt.InitNormIdentite();
   anEt.CalculModeleRadiale(true,TheNameRotInit,false,true,aParam.ImagesInit(),true);
}

void cEtalonnage::CalculModeleRadialeFinal(int argc,char ** argv)
{
   cParamEtal   aParam(argc,argv);
   cEtalonnage  anEt(false,aParam);
   anEt.InitNormCmaIdeale();
   // anEt.InitNormIdentite();
   anEt.CalculModeleRadiale(true,TheNameRotFinale,false,true,aParam.AllImagesCibles(),false);
}

void cEtalonnage::TestLiaison(int argc,char ** argv)
{
   cParamEtal   aParam(argc,argv);
   cEtalonnage  anEt(false,aParam);
   anEt.InitNormCmaIdeale();
   anEt.CalculModeleRadiale(true,"",false,true,aParam.AllImagesCibles(),false);
   anEt.CalculLiaison(false);
}

INT cEtalonnage::CodeSuccess()
{
	return 54;
}


void cEtalonnage::DoCompensation(const std::string & aParamComp,const std::string & aPrefix,bool PhgrStd)
{
   std::string aNameCam =  mParam.NameCamera();

/*
std::cout << "LOEMI " << mParam.CalibSpecifLoemi() << "\n"; getchar();
   if (mParam.CalibSpecifLoemi())
   {
      std::string aNameAuto = "Syst([0-9]{1,3}_[0-9]{2,4}_[0-9]{1,2}_[0-9]{2,3}_[rvbilmno])";
      cElRegex anAutom(aNameAuto,10);
      if (! anAutom.Match(aNameCam))
      {
          std::cout << "Nom de camera " << aNameCam << "\n";
          std::cout << "Automate de specif : " << aNameAuto << "\n";
          ELISE_ASSERT(false," Nom de camera incorrecte");
      }
      aNameCam = MatchAndReplace(anAutom,aNameCam,"$1");
   }
*/

   std::string aLibCD = "eLib_PP_CD_11";
   std::string aTolC = "-1";
   if (PhgrStd)
   {
         
       aLibCD = "eLib_PP_CD_Lies";
       if (mParam.CDistLibre(true))
       {
          aTolC = "5000.0";
       }
       else
       {
          aTolC = "1e-5";
       }
   }
   else
   {
/*
       if (!mParam.CDistLibre(true))
       {
           aLibCD = "eLib_PP_CD_Lies";
       }
*/
   }

   // bool isCDL =  mParam.CDistLibre(true) || (! OptionFigeC);
   // std::string aLibCD = isCDL ? "eLib_PP_CD_11" : "eLib_PP_CD_Lies" ;

   std::string aCom =   MMDir() +  std::string("bin/Apero ")
                        + std::string(" ") + MMDir() + aParamComp
                        + std::string(" DirectoryChantier=") + mDir
                        + std::string(" \"+PatternIm=") + mParam.PatternGlob() +std::string("\"")
                        + std::string(" +NameCam=")     +  aNameCam
                        + std::string(" +PrefCal=") +  aPrefix
                        + std::string(" +KeySetOri=") +   mParam.KeySetOri()
                        + std::string(" +KeyAssocOri=") +  mParam.KeyExportOri()
                        + std::string(" +DoGrid=") +  ToString(mParam.DoGrid())
                        + std::string(" +FileParamEtal=") +  mParam.NameFile()


                        + std::string(" +SeuilRejet=") +  ToString(mParam.SeuilRejetAbs())
                        + std::string(" +SeuilPonder=") +  ToString(mParam.SeuilPonder())

                        + std::string(" +LiberteCentre=") +  aLibCD
                        + std::string(" +TolLiberteCentre=") +  aTolC  ; 

                      ;
   
   std::cout << "RUN COM=" << aCom << "\n";
   int aK = system(aCom.c_str());
   if (aK!=0)
   {
      std::cout << "COM=" << aCom << "\n";
      ELISE_ASSERT(false,"Erreur dans la commande de compensation");
   }
}

void cEtalonnage::DoCompensation(int argc,char **argv)
{
   cParamEtal   aParam(argc,argv);
   cEtalonnage  anEt(false,aParam);
   for ( const char * anOC = aParam.mOrderComp.c_str() ; *anOC ; anOC++)
   {
       if (*anOC=='X')
          anEt.DoCompensation("applis/XML-Pattron/EP_Compensation_Rectang_Param.xml","Rect",true);
       else if (*anOC=='P')
          anEt.DoCompensation("applis/XML-Pattron/EP_Compensation_PhgrStd_Param.xml","PhgrStd",true);
       else if (*anOC=='R')
          anEt.DoCompensation("applis/XML-Pattron/EP_Compensation_DRad_Param.xml","DRad",false);
       else if (*anOC=='5')
          anEt.DoCompensation("applis/XML-Pattron/EP_Compensation_Pol5_Param.xml","Pol5",true);
       else
       {
          ELISE_ASSERT(false,"Unknown compens mode");
       }
   }
}

/*
void cEtalonnage::VerifArgFile(int argc,char ** argv)
{
    cParamEtal   aParam(argc,argv);
    cEtalonnage  anEt;
}

*/


/*
void cEtalonnage::TestModeleRadiale(int argc,char ** argv,const std::string & aName)
{
   cParamEtal   aParam(argc,argv);
   cEtalonnage  anEt(aParam);
   anEt.InitNormDrad(TheNameDradInterm);
   CalculModeleRadiale(true,mParam.ImagesInit(),false);
}

void TestEtal(int argc,char ** argv)
{
   anEt.CalculModeleRadialeInit();
}
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
