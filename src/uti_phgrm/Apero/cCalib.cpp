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
#include "Apero.h"


Pt2dr aDebugPIm(915.982,2820.98);
Pt2dr aDebugPL3(-0.339882,0.170243);


void VerifTolNonPos(double aTol,eTypeContrainteCalibCamera aVal)
{
    if (aTol>0)
    {
        std::cout  << "------- Pour la contrainte :" << eToString(aVal) << "\n";
	ELISE_ASSERT
	(
	    false,
	    "Tolerance inutile pour la liberation de contrainte"
	);
    }
}

bool DRF_SetContrainte
     (
          double aTol,
          cParamIFDistRadiale &      aPIF,
          const eTypeContrainteCalibCamera & aCstr
     )
{
    switch(aCstr)
    {
            case eAllParamLibres:
	        aPIF.SetLibertePPAndCDist(true,true,aTol);
	        aPIF.SetDRFDegreFige(5,aTol);
            break;

            case eAllParamFiges:
	        aPIF.SetLibertePPAndCDist(false,false,aTol);
	        aPIF.SetDRFDegreFige(0,aTol);
            break;


            case eLib_PP_CD_00 :
	        aPIF.SetLibertePPAndCDist(false,false,aTol);
	    break;
            case eLib_PP_CD_10 :
	        aPIF.SetLibertePPAndCDist(true,false,aTol);
	    break;
            case eLib_PP_CD_01 :
	        aPIF.SetLibertePPAndCDist(false,true,aTol);
	    break;
            case eLib_PP_CD_11 :
	        VerifTolNonPos(aTol,aCstr);
	        aPIF.SetLibertePPAndCDist(true,true,aTol);
	    break;
            case eLib_PP_CD_Lies :
	       aPIF.SetCDistPPLie(aTol);
	    break;


            case eLiberte_DR0 :
	       aPIF.SetDRFDegreFige(0,aTol);
	    break;
            case eLiberte_DR1 :
	       aPIF.SetDRFDegreFige(1,aTol);
	    break;
            case eLiberte_DR2 :
	       aPIF.SetDRFDegreFige(2,aTol);
	    break;
            case eLiberte_DR3 :
	       aPIF.SetDRFDegreFige(3,aTol);
	    break;
            case eLiberte_DR4 :
	       aPIF.SetDRFDegreFige(4,aTol);
	    break;
            case eLiberte_DR5 :
	        VerifTolNonPos(aTol,aCstr);
	       aPIF.SetDRFDegreFige(5,aTol);
	    break;


	    default :
	         return false;
	    break;
    }
    return true;
}


/**************************************************/
/*                                                */
/*                 cCalibCam                      */
/*                                                */
/**************************************************/

static const double DefRMaxU2 = 1e20;

cCalibCam::cCalibCam
(
    const cCalibrationInternConique & aCIC,
    bool                          isFE,
    const                         std::string & aKeyId,
    cAppliApero &                 anAppli,
    const cCalibrationCameraInc & aCCI,
    cParamIntrinsequeFormel &     aPIF,
    CamStenope &                  aCamInit,
    Pt2di                         aSzIm
) :
    mIsFE    (isFE),
    mAppli   (anAppli),
    mKeyId   (aKeyId),
    mCCI     (aCCI),
    mPIF     (aPIF),
    mCamInit (aCamInit),
    mSzIm    (aSzIm),
    mMil     (Pt2dr(mSzIm)/2.0),
    mRMaxU2  (DefRMaxU2),
    mFiged   (false),
    mPropDiagU (aCCI.PropDiagUtile().Val()),
    mRay2Max  (10 * square_euclid(mMil)),
    mReducPReg (50.0),
    mSzPReg    (round_up(Pt2dr(mSzIm)/mReducPReg)),
    mImReg     (mSzPReg.x,mSzPReg.y,0.0),
    mTImReg    (mImReg)
{
   SetRMaxU(aCCI.RayMaxUtile().Val(),aCCI.RayIsRelatifDiag().Val(),aCCI.RayApplyOnlyFE().Val());

   CamStenope * aCS = mPIF.CurPIF();
   if (aCCI.RayMaxUtile().IsInit())
   {
   }
   if (aCS->HasRayonUtile() )
   {
      SetRMaxU(aCS->RayonUtile(),false,false);
   }

   if (aCIC.RayonUtile().IsInit())
   {
      SetRMaxU(aCIC.RayonUtile().Val(),false,false);
   }
}

cCalibCam::~cCalibCam() {}

void cCalibCam::ActiveContrainte(bool Stricte)
{
   mAppli.SetEq().AddContrainte(mPIF.StdContraintes(),Stricte);

   if (mAppli.Param().GridOptimKnownDist().Val())
      mPIF.UpdateCamGrid( true );


/*
   bool aLastFiged = mFiged;
   mFiged = mPIF.AllParamIsFiged();
   std::cout << "FIGED " <<  aLastFiged << " " << mFiged << "\n";
   if (mFiged && (! aLastFiged))
   {
      std::cout << " ------------ Transition --------------\n";
      
      CamStenope * aCS = mPIF.CurPIF();

      Pt2di aRab(20,20);
      Pt2di aStep(10,10);
      double aRabR = 0;
      Pt2di aSz = aCS->Sz();

      double aR = euclid(aSz)/2.0;
      if (aCS->HasRayonUtile())
         aR = aCS->RayonUtile();

      cCamStenopeGrid * aCSG = cCamStenopeGrid::Alloc(aR+aRabR,*aCS,aStep,true,false);
       
      double aTol = 1e-1;
      double aRMinTol = 1e20;
      



       for (int aKx=100 ; aKx< aCS->Sz().x ; aKx += 200)
       {
           for (int aKy=100 ; aKy< aCS->Sz().y ; aKy += 200)
           {
               Pt2dr aP(aKx,aKy);
               double aEps=1e-5;

                Pt3dr aRay = aCS->F2toDirRayonL3(aP);

                Pt3dr  aRayX = aRay + Pt3dr(aEps,0,0);
                Pt3dr  aRayY = aRay + Pt3dr(0,aEps,0);
                
                Pt2dr aP1 = aCS->L3toF2(aRay);
                Pt2dr aPG = aCSG->L3toF2(aRay);

                Pt2dr aDx = (aCSG->L3toF2(aRayX)-aPG)/aEps;
                Pt2dr aDy = (aCSG->L3toF2(aRayY)-aPG)/aEps;


                Pt2dr aDGX,aDGY;
                Pt2dr aPG2 = aCSG->L2toF2AndDer(Pt2dr(aRay.x,aRay.y),aDGX,aDGY);

                std::cout << aPG << aDx << aDy << "\n";
                std::cout <<  "    " << aPG2 << aDGX << aDGY << "\n";

                if (euclid(aP1,aPG) >aTol)
                   aRMinTol = ElMin(aRMinTol,euclid(aP,aSz/2.0));
           }
       }

       std::cout << "------------RTOl------------ " << aRMinTol << "\n"; // getchar();
   }
*/
}

    //   ACCESSEURS 
   

CamStenope & cCalibCam::CamInit() {return mCamInit;}
cParamIntrinsequeFormel & cCalibCam::PIF() {return mPIF;}
Pt2di  cCalibCam::SzIm() const {return mSzIm;}

/*
double cCalibCam::RMaxU() const
{
   return mRMaxU;
}
*/

bool cCalibCam::IsInZoneU(const Pt2dr & aP) const
{
   return square_euclid(aP-mMil) < mRMaxU2;
}

void cCalibCam::SetRMaxU(double  aR,bool IsRel,bool OnlyFE)
{
    if (IsRel) 
        aR = aR * euclid(mMil) * mPropDiagU;
    if ((!OnlyFE) || mIsFE)
    {
        double aR2 =  ElSquare(aR);
        if (aR2 < mRay2Max)
        {
           mRMaxU2 = aR2;
           CamStenope * aCS = mPIF.CurPIF();
           aCS->SetRayonUtile(sqrt(aR2),30);
        }
    }
}

bool  cCalibCam::HasRayonMax() const
{
   return mRMaxU2 < mRay2Max;
}
double cCalibCam::RayonMax() const
{
    ELISE_ASSERT(HasRayonMax(),"Invalid call to cCalibCam::HasRayonMax");
    return sqrt(mRMaxU2);
}


void cCalibCam::SetContrainte(const cContraintesCamerasInc & aCCI)
{

    for
    (
        std::list<eTypeContrainteCalibCamera>::const_iterator  itC=aCCI.Val().begin();
	itC!=aCCI.Val().end();
	itC++
    )
    {
        double aTol = aCCI.TolContrainte().Val();
	
        switch (*itC) 
	{
            case eAllParamFiges :
	        VerifTolNonPos(aTol,*itC);
	        mPIF.SetFocFigee(aTol);
	        InstSetContrainte(aTol,*itC);
            break;

            case eAllParamLibres :
	        VerifTolNonPos(aTol,*itC);
	        mPIF.SetFocFree(true);
	        InstSetContrainte(aTol,*itC);
            break;


            case eLiberteFocale_0 :
	        mPIF.SetFocFigee(aTol);
	    break;

            case eLiberteFocale_1 :
	        VerifTolNonPos(aTol,*itC);
	        mPIF.SetFocFree(true);
	    break;

      //  PARAMETRES AFOCAUX
            case eLiberte_AFocal0 :
                 mPIF.SetAF1Free(true);
            break;
            case eLiberte_AFocal1 :
                 mPIF.SetAF2Free(true);
            break;
            case eFige_AFocal0 :
                 mPIF.SetAF1Free(false);
            break;
            case eFige_AFocal1 :
                 mPIF.SetAF2Free(false);
            break;

            default :
	        bool OK = InstSetContrainte(aTol,*itC);
                if (mAppli.Param().GenereErreurOnContraineCam().Val())
                {
/// AJOUUTER CONTRIANT
                    std::cout << "FOR CONTRAINTE " << eToString(*itC) << "\n";
		    ELISE_ASSERT
		    (
		        OK,
		        "Cannot handle contrainte"
		    );
                }
	    break;

	}
    }
}

const cCalibrationCameraInc &   cCalibCam::CCI()
{
   return mCCI;
}


const std::string & cCalibCam::KeyId()
{
   return mKeyId;
}

void  cCalibCam::Inspect()
{
}
/**************************************************/
/*                                                */
/*          cCalibCam_ModeleRadial                */
/*                                                */
/**************************************************/


class cCalibCam_ModeleRadial : public  cCalibCam
{
     public :
         virtual ~cCalibCam_ModeleRadial() {}

         cCalibCam_ModeleRadial
	 (
              const         std::string & aKeyId,
	      cAppliApero & anAppli,
	      const cCalibrationCameraInc & aCCI,
	      const cCalibrationInternConique & aCIR,
	      cParamIFDistRadiale &              aPIF,
	      cCamStenopeDistRadPol &            aCamInit
	 );
	 bool InstSetContrainte
              (
                     double aTol,
                     const eTypeContrainteCalibCamera &
              );
     private :

        cParamIFDistRadiale &      mPIF;
	//cCamStenopeDistRadPol &    mCamInit;
};


cCalibCam_ModeleRadial::cCalibCam_ModeleRadial
(
       const         std::string & aKeyId,
       cAppliApero & anAppli,
       const cCalibrationCameraInc & aCCI,
       const cCalibrationInternConique & aCIC,
       cParamIFDistRadiale &              aPIF,
       cCamStenopeDistRadPol &            aCamInit
) :
   cCalibCam (aCIC,false,aKeyId,anAppli,aCCI,aPIF,aCamInit,aCIC.SzIm()),
   mPIF      (aPIF)//,
   //mCamInit  (aCamInit)
{
}


bool cCalibCam_ModeleRadial::InstSetContrainte
     (
         double aTol,
         const eTypeContrainteCalibCamera & aCstr
     )
{
   return DRF_SetContrainte(aTol,mPIF,aCstr);
}





/**************************************************/
/*                                                */
/*          cCalibCam_ModeleUnif                  */
/*                                                */
/**************************************************/

class cCalibCam_ModeleUnif : public  cCalibCam
{
     public :

         void Inspect();
         ~cCalibCam_ModeleUnif() {}

         cCalibCam_ModeleUnif
	 (
              bool          isFE,
              const         std::string & aKeyId,
	      cAppliApero & anAppli,
	      const cCalibrationCameraInc & aCCI,
	      const cCalibrationInternConique & aCIR,
	      cPIF_Unif_Gen &              aPIF,
	      cCamera_Param_Unif_Gen &            aCamInit
	 );
	 bool InstSetContrainte
              (
                     double aTol,
                     const eTypeContrainteCalibCamera &
              );
     private :

        cPIF_Unif_Gen &             mPIF;
	//cCamera_Param_Unif_Gen &    mCamInit;
};

void cCalibCam_ModeleUnif::Inspect()
{
    mPIF.Inspect();
}


cCalibCam_ModeleUnif::cCalibCam_ModeleUnif
(
       bool          isFE,
       const         std::string & aKeyId,
       cAppliApero & anAppli,
       const cCalibrationCameraInc & aCCI,
       const cCalibrationInternConique & aCIC,
       cPIF_Unif_Gen &              aPIF,
       cCamera_Param_Unif_Gen &            aCamInit
) :
   cCalibCam (aCIC,isFE,aKeyId,anAppli,aCCI,aPIF,aCamInit,aCIC.SzIm()),
   mPIF      (aPIF)//,
   //mCamInit  (aCamInit)
{
}


bool cCalibCam_ModeleUnif::InstSetContrainte
     (
         double aTol,
         const eTypeContrainteCalibCamera & aCstr
     )
{

   switch (aCstr)
   {

       case eAllParamLibres :
	     mPIF.SetPPFree(true);
             mPIF.FigeIfDegreSup(1,aTol,eModeContDGCDist);
             mPIF.FigeIfDegreSup(10,aTol,eModeContDGDRad);
             mPIF.FigeIfDegreSup(5,aTol,eModeContDGDCent);
             mPIF.FigeIfDegreSup(10,aTol,eModeContDGPol);
       break;
       case  eAllParamFiges :
	     mPIF.SetPPFree(false);
             mPIF.FigeIfDegreSup(0,aTol,eModeContDGCDist);
             mPIF.FigeIfDegreSup(0,aTol,eModeContDGDRad);
             mPIF.FigeIfDegreSup(0,aTol,eModeContDGDCent);
             mPIF.FigeIfDegreSup(0,aTol,eModeContDGPol);
       break;



       case eLiberteParamDeg_0 :
           mPIF.FigeIfDegreSup(0,aTol,eModeContDGPol);
       break;

       case eLiberteParamDeg_1 :
           mPIF.FigeIfDegreSup(1,aTol,eModeContDGPol);
       break;

       case eLiberteParamDeg_2 :
           mPIF.FigeIfDegreSup(2,aTol,eModeContDGPol);
       break;
       case eLiberteParamDeg_2_NoAff :
           mPIF.FigeD1_Ou_IfDegreSup(2,aTol);
       break;

       case eLiberteParamDeg_3 :
           mPIF.FigeIfDegreSup(3,aTol,eModeContDGPol);
       break;
       case eLiberteParamDeg_3_NoAff :
           mPIF.FigeD1_Ou_IfDegreSup(3,aTol);
       break;

       case eLiberteParamDeg_4 :
           mPIF.FigeIfDegreSup(4,aTol,eModeContDGPol);
       break;
       case eLiberteParamDeg_4_NoAff :
           mPIF.FigeD1_Ou_IfDegreSup(4,aTol);
       break;

       case eLiberteParamDeg_5 :
           mPIF.FigeIfDegreSup(5,aTol,eModeContDGPol);
       break;
       case eLiberteParamDeg_5_NoAff :
           mPIF.FigeD1_Ou_IfDegreSup(5,aTol);
       break;


       case eLiberteParamDeg_6 :
           mPIF.FigeIfDegreSup(6,aTol,eModeContDGPol);
       break;
       case eLiberteParamDeg_7 :
           mPIF.FigeIfDegreSup(7,aTol,eModeContDGPol);
       break;



       case eLib_PP_CD_00 :
	    mPIF.SetPPFFige(aTol);
            mPIF.FigeIfDegreSup(0,aTol,eModeContDGCDist);
       break;

       case eLib_PP_CD_10 :
	    mPIF.SetPPFree(true);
            mPIF.FigeIfDegreSup(0,aTol,eModeContDGCDist);
       break;

       case eLib_PP_CD_01 :
	    mPIF.SetPPFFige(aTol);
            mPIF.FigeIfDegreSup(1,aTol,eModeContDGCDist);
       break;


       case eLib_PP_CD_11 :
	     mPIF.SetPPFree(true);
             mPIF.FigeIfDegreSup(1,aTol,eModeContDGCDist);
             VerifTolNonPos(aTol,aCstr);
       break;


        case eLiberte_DR0 :
             mPIF.FigeIfDegreSup(0,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR1 :
             mPIF.FigeIfDegreSup(1,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR2 :
             mPIF.FigeIfDegreSup(2,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR3 :
             mPIF.FigeIfDegreSup(3,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR4 :
             mPIF.FigeIfDegreSup(4,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR5 :
             mPIF.FigeIfDegreSup(5,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR6 :
             mPIF.FigeIfDegreSup(6,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR7 :
             mPIF.FigeIfDegreSup(7,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR8 :
             mPIF.FigeIfDegreSup(8,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR9 :
             mPIF.FigeIfDegreSup(9,aTol,eModeContDGDRad);
	break;
        case eLiberte_DR10 :
             mPIF.FigeIfDegreSup(10,aTol,eModeContDGDRad);
	break;


        case eLiberte_Dec0 :
             mPIF.FigeIfDegreSup(0,aTol,eModeContDGDCent);
	break;
        case eLiberte_Dec1 :
             mPIF.FigeIfDegreSup(1,aTol,eModeContDGDCent);
	break;
        case eLiberte_Dec2 :
             mPIF.FigeIfDegreSup(2,aTol,eModeContDGDCent);
	break;
        case eLiberte_Dec3 :
             mPIF.FigeIfDegreSup(3,aTol,eModeContDGDCent);
	break;
        case eLiberte_Dec4 :
             mPIF.FigeIfDegreSup(4,aTol,eModeContDGDCent);
	break;
        case eLiberte_Dec5 :
             mPIF.FigeIfDegreSup(5,aTol,eModeContDGDCent);
	break;






       default :
           return false;
       break;

   }
   return true;
}


/**************************************************/
/*                                                */
/*          cCalibCam_BiLin                       */
/*                                                */
/**************************************************/

/*
class cCalibCam_BiLin : public  cCalibCam
{
     public :
         ~cCalibCam_BiLin() {}

         cCalibCam_BiLin
	 (
              const         std::string & aKeyId,
*/

class cCalibCam_BiLin : public  cCalibCam
{
     public :

//          void Inspect();
         ~cCalibCam_BiLin() {}

         cCalibCam_BiLin
	 (
              const         std::string & aKeyId,
	      cAppliApero & anAppli,
	      const cCalibrationCameraInc & aCCI,
	      const cCalibrationInternConique & aCIR,
	      cPIF_Bilin &              aPIF,
	      cCamStenopeBilin &            aCamInit
	 );
	 bool InstSetContrainte
              (
                     double aTol,
                     const eTypeContrainteCalibCamera &
              );
     private :

        cPIF_Bilin &             mPIF;
	//cCamera_Param_Unif_Gen &    mCamInit;
};

cCalibCam_BiLin::cCalibCam_BiLin
(
       const         std::string & aKeyId,
       cAppliApero & anAppli,
       const cCalibrationCameraInc & aCCI,
       const cCalibrationInternConique & aCIC,
       cPIF_Bilin &              aPIF,
       cCamStenopeBilin &            aCamInit
) :
   cCalibCam (aCIC,false,aKeyId,anAppli,aCCI,aPIF,aCamInit,aCIC.SzIm()),
   mPIF      (aPIF)//,
   //mCamInit  (aCamInit)
{
}

bool cCalibCam_BiLin::InstSetContrainte
     (
            double aTol,
            const eTypeContrainteCalibCamera & aCstr
     )
{
   switch (aCstr)
   {

       case eAllParamLibres :
	     mPIF.SetPPFree(true);
	     mPIF.SetDistFree(100);
       break;
       case  eAllParamFiges :
	     mPIF.SetPPFree(false);
	     mPIF.SetDistFigee();
       break;

       case eLiberteParamDeg_0 :
       break;

       case eLiberteParamDeg_1 :
           mPIF.SetDistFree(1);
       break;

       case eLiberteParamDeg_2 :
           mPIF.SetDistFree(2);
       break;

       case eLiberteParamDeg_3 :
           mPIF.SetDistFree(3);
       break;

       case eLiberteParamDeg_4 :
           mPIF.SetDistFree(4);
       break;

       case eLiberteParamDeg_5 :
           mPIF.SetDistFree(5);
       break;

       case eLiberteParamDeg_6 :
           mPIF.SetDistFree(6);
       break;
       case eLiberteParamDeg_7 :
           mPIF.SetDistFree(7);
       break;


       default :
           return false;
       break;

   }
   return true;
}

/**************************************************/
/*                                                */
/*          cCalibCam_PhgrStd                     */
/*                                                */
/**************************************************/

class cCalibCam_PhgrStd : public  cCalibCam
{
     public :
         ~cCalibCam_PhgrStd() {}

         cCalibCam_PhgrStd
	 (
              const         std::string & aKeyId,
	      cAppliApero & anAppli,
	      const cCalibrationCameraInc & aCCI,
	      const cCalibrationInternConique & aCIR,
	      cParamIFDistStdPhgr &              aPIF,
	      cCamStenopeModStdPhpgr &            aCamInit
	 );
	 bool InstSetContrainte
              (
                     double aTol,
                     const eTypeContrainteCalibCamera &
              );
     private :

        cParamIFDistStdPhgr &      mPIF;
	//cCamStenopeModStdPhpgr &    mCamInit;
};

cCalibCam_PhgrStd::cCalibCam_PhgrStd
(
       const         std::string & aKeyId,
       cAppliApero & anAppli,
       const cCalibrationCameraInc & aCCI,
       const cCalibrationInternConique & aCIC,
       cParamIFDistStdPhgr &              aPIF,
       cCamStenopeModStdPhpgr &            aCamInit
) :
   cCalibCam (aCIC,false,aKeyId,anAppli,aCCI,aPIF,aCamInit,aCIC.SzIm()),
   mPIF      (aPIF)//,
   //mCamInit  (aCamInit)
{
}

bool cCalibCam_PhgrStd::InstSetContrainte
     (
                     double aTol,
                     const eTypeContrainteCalibCamera & aCstr
     )
{
    switch (aCstr)
    {
        case  eAllParamLibres:
              mPIF.SetParam_Aff_Free();
              mPIF.SetParam_Dec_Free();
              DRF_SetContrainte(aTol,mPIF,aCstr);
        break;
        case eAllParamFiges:
              mPIF.SetParam_Aff_Fige();
              mPIF.SetParam_Dec_Fige();
              DRF_SetContrainte(aTol,mPIF,aCstr);
        break;

        case eLiberte_Phgr_Std_Aff :
             mPIF.SetParam_Aff_Free();
             VerifTolNonPos(aTol,aCstr);
	break;

        case eLiberte_Phgr_Std_Dec :
             mPIF.SetParam_Dec_Free();
             VerifTolNonPos(aTol,aCstr);
	break;

        case eFige_Phgr_Std_Aff :
             mPIF.SetParam_Aff_Fige(aTol);
	break;

        case eFige_Phgr_Std_Dec :
             mPIF.SetParam_Dec_Fige(aTol);
	break;

        default :
             return DRF_SetContrainte(aTol,mPIF,aCstr);
	break;
    }


    return true;
}

/**************************************************/
/*                                                */
/*                 CalibAutom                     */
/*                                                */
/**************************************************/

cCalibrationInternConique   CalibInternAutom
                            (
                                const cMetaDataPhoto & aMDP,
                                eTypeCalibAutom aType,
                                const std::string & aName,
                                double              aSeuilFE,
                                Pt2dr               aPPRel
                            )
{


  ELISE_ASSERT(aType!=eCalibAutomNone,"Internal Error eCalibAutomNone");

   cCalibrationInternConique aRes;
/*
std::cout << "AAAAAAAAAAAAAAAAAAAAAa\n";
aMDP.SzImTifOrXif();
std::cout << "aaaaaaaaaaaaaaaaaaaaaaa\n";

   Pt2di aSzIm = aMDP.TifSzIm();
std::cout << "BBBBBBBBBBBBBBBB\n";
*/
   // MPD Modif pour que ca passe avec des fichiers vides
   Pt2di aSzIm = aMDP.SzImTifOrXif();


   Pt2dr aMil = Pt2dr(aSzIm).mcbyc(aPPRel);


   double aF35 = aMDP.Foc35();

   if (aF35<aSeuilFE) 
      aType= eCalibAutomFishEyeLineaire;
/*
   if (aType==eCalibAutomNone)
   {
      else
         aType= eCalibAutomRadial;
   }
*/




   if (aF35<=0)
   {
       std::cout << "File = " << aName << "\n";
       ELISE_ASSERT(false,"Cannot get foc equiv 35 mm");
   }
    
   // double aF = aF35 * euclid(aSzIm) / euclid(aPRefFullFrame);
   double aF = aMDP.FocPix();
if (1)
{
    std::cout << " MdPppppF= " << aF << " SFE=" << aSeuilFE  << " FocMm" << aMDP.FocMm()  
              << " F35=" <<  aMDP.Foc35()  << " XSZ=" << aMDP.XifSzIm() << "\n";
}

   aRes.PP() = aMil;
   aRes.F() = aF;
   aRes.SzIm() = aSzIm;

   cCalibDistortion aDist = ElDistortion22_Gen::XmlDistNoVal();

// eCalibAutomPhgrStd
   if (
              (aType==eCalibAutomRadial)      || (aType==eCalibAutomPhgrStd)
           || (aType==eCalibAutomRadialBasic) || (aType==eCalibAutomPhgrStdBasic)
      )
   {
       cCalibrationInterneRadiale aDRad;
       aDRad.CDist() = aMil;
       aDRad.RatioDistInv().SetVal(1.3);
       if ((aType==eCalibAutomRadialBasic) || (aType==eCalibAutomPhgrStdBasic))
       {
          aDRad.PPaEqPPs().SetVal(true);
       }

       if ((aType==eCalibAutomRadial) || (aType==eCalibAutomRadialBasic))
       {
          aDist.ModRad().SetVal(aDRad);
       }
       else
       {
           cCalibrationInternePghrStd aCPS;
           aCPS.RadialePart() = aDRad;
           aCPS.P1().SetVal(0.0);
           aCPS.P2().SetVal(0.0);
           aCPS.b1().SetVal(0.0);
           aCPS.b2().SetVal(0.0);
           aDist.ModPhgrStd().SetVal(aCPS);
       }
   }
   else if(
                  (aType==eCalibAutomFishEyeLineaire)
              ||  (aType==eCalibAutomFishEyeEquiSolid)
              ||  (aType==eCalibAutomFishEyeStereographique)
          )
   {
       cCalibrationInterneUnif aCIU;

       if  (aType==eCalibAutomFishEyeLineaire)
           aCIU.TypeModele() = eModele_FishEye_10_5_5 ;
       else if  (aType==eCalibAutomFishEyeEquiSolid)
           aCIU.TypeModele() = eModele_EquiSolid_FishEye_10_5_5 ;
       else if  (aType==eCalibAutomFishEyeStereographique)
           aCIU.TypeModele() = eModele_Stereographik_FishEye_10_5_5 ;

       aCIU.Etats().push_back(aF);
       aCIU.Params().push_back(aMil.x);
       aCIU.Params().push_back(aMil.y);
        
       aDist.ModUnif().SetVal(aCIU);
   }
   else if (
                  (aType==eCalibAutomFour7x2)
              ||  (aType==eCalibAutomFour11x2)
              ||  (aType==eCalibAutomFour15x2)
              ||  (aType==eCalibAutomFour19x2)
           )
   {
       cCalibrationInterneUnif aCIU;
       if  (aType==eCalibAutomFour7x2)
           aCIU.TypeModele() = eModeleRadFour7x2;
       else if  (aType==eCalibAutomFour11x2)
           aCIU.TypeModele() = eModeleRadFour11x2;
       else if  (aType==eCalibAutomFour15x2)
           aCIU.TypeModele() = eModeleRadFour15x2;
       else if  (aType==eCalibAutomFour19x2)
           aCIU.TypeModele() = eModeleRadFour19x2;

       aDist.ModUnif().SetVal(aCIU);
   }
   else if (
                  (aType==eCalibAutomEbner)
              ||  (aType==eCalibAutomBrown)
           )
   {
       cCalibrationInterneUnif aCIU;
       if  (aType==eCalibAutomEbner)
       {
           aCIU.TypeModele() = eModeleEbner;
       }
       else if  (aType==eCalibAutomBrown)
       {
           aCIU.TypeModele() = eModeleDCBrown;
       }
       aDist.ModUnif().SetVal(aCIU);
   }
   else
   {
       ELISE_ASSERT(false,"Internal error unknown eTypeCalibAutom");
   }

   aRes.CalibDistortion().push_back(aDist);


   return aRes;
}

/**************************************************/
/*                                                */
/*                 cCalibCam                      */
/*                                                */
/**************************************************/


class cNameFileWithExistigChangeVersionNameCam
{
    public :
        cNameFileWithExistigChangeVersionNameCam
        (
            const std::string & aDirGlob,
            cInterfChantierNameManipulateur * anICNM,
            const std::string & aKey,
            const std::string & aNamePose
        ) :
          mDirG     (aDirGlob),
          mICNM     (anICNM),
          mKey      (aKey),
          mNamePose (aNamePose),
          mMMU      (const_cast<cMMUserEnvironment&> (MMUserEnv())),
          mCurNumC  (mMMU.VersionNameCam().Val())
       {
       }

       std::string GetName()
       {
            std::string aRes0 = TestOnFile(mCurNumC);

            if (ELISE_fp::exist_file(aRes0)) return aRes0;

            for (int aNum=0 ; aNum<2 ; aNum++)
            {
                if (aNum!=mCurNumC)
                {
                    std::string aRes = TestOnFile(aNum);
                    if (ELISE_fp::exist_file(aRes)) return aRes;
                }
            }

            return aRes0;
       }

    private :

          std::string TestOnFile(int aNum)
          {
               mMMU.VersionNameCam().SetVal(aNum);
               std::string aRes = mDirG + mICNM->Assoc1To1(mKey,mNamePose,true);
               mMMU.VersionNameCam().SetVal(mCurNumC);
               return aRes;
          }
          std::string mDirG;
          cInterfChantierNameManipulateur *mICNM;
          std::string mKey;
          std::string mNamePose;
          cMMUserEnvironment & mMMU;
          int                  mCurNumC;
};


std::string cInterfChantierNameManipulateur::StdNameCalib(const std::string & anOri,const std::string & aNameIm)
{
     std::string aKey = "NKS-Assoc-FromFocMm@Ori-"+ anOri + "/AutoCal@.xml";
     cNameFileWithExistigChangeVersionNameCam     aCNFW(Dir(),this,aKey,aNameIm);
     return aCNFW.GetName();
}



void cAppliApero::NormaliseScTr(CamStenope & aCam)
{
   aCam.StdNormalise(mParam.NormaliseEqSc().Val(),mParam.NormaliseEqTr().Val());
}


void cCalibCam::AddViscosite(const std::vector<double> & aTol)
{
    mPIF.AddRapViscosite(aTol[0]);
}


void cCalibCam::InitAvantCompens()
{
   mSomNbReg = 0.0;
   mSomPdsReg = 0.0;
   mImReg.raz();
}

void  cCalibCam::PostFinCompens()
{
   // On renormalise pour que ce soit equivalent a un nombre de point et prop a une ponderation
   ELISE_COPY(mImReg.all_pts(),mImReg.in() * (mSomNbReg/mSomPdsReg),mImReg.out());

   const cXmlPondRegDist * aPond = mAppli.CurXmlPondRegDist();

   if ((aPond==0) || (mSomNbReg==0)) return;
   
   Im2D_REAL4 aImPds(mSzPReg.x,mSzPReg.y);

   int aNbCase = aPond->NbCase();
   Pt2di aSzW = round_up(Pt2dr(mSzPReg) * (1/(1.0+2.0*aNbCase)));
   ELISE_COPY
   (
        mImReg.all_pts(),
        rect_som(mImReg.in(0),aSzW) /  rect_som(mImReg.inside(),aSzW),
        aImPds.out()
   );

   // Renormalise a som=mSomNbReg, et multiplie par une case pour que en moyenne, la valeur d'un pixel soit 
   // la somme sur une case
   double aSom;
   ELISE_COPY(aImPds.all_pts(),aImPds.in(),sigma(aSom));
   double aNbByCase = (1+2*aSzW.x)*(1+2*aSzW.y);
   ELISE_COPY(aImPds.all_pts(),aImPds.in() * ((aNbByCase*mSomNbReg)/aSom),aImPds.out());

   double aSeuil = aPond->SeuilNbPtsByCase();
   Fonc_Num aF = Square(aSeuil / (aSeuil + aImPds.in()));

   double aS1,aSP;
   ELISE_COPY(aImPds.all_pts(),Virgule(aF,1),Virgule(aImPds.out()|sigma(aSP),sigma(aS1)));

   double aMul = aSP /aS1;
   mPIF.AddCstrRegulGlob(1+2*aNbCase,aPond->Pds0()*aMul,aPond->Pds1()*aMul,aPond->Pds2()*aMul,&aImPds);

}

void cCalibCam::AddPds(const Pt2dr & aPt,const double & aPds)
{
   mTImReg.incr(aPt/mReducPReg,aPds);
   mSomNbReg++;
   mSomPdsReg += aPds;
}

void cCalibCam::Export(const std::string & aNameXml)
{
    std::string aNameTif = DirOfFile(aNameXml) + "Densite_" + StdPrefix(NameWithoutDir(aNameXml)) + ".tif";

    Tiff_Im::CreateFromIm(mImReg,aNameTif);
    // std::cout << "cCalibCam::Export :: " << aNameTif << "\n"; getchar();
}

extern std::string TheSpecMess;

cCalibCam *  cCalibCam::Alloc(const std::string & aKeyId,cAppliApero & anAppli,const cCalibrationCameraInc & aCCI,cPoseCam * aPC)
{

    cCalibrationInternConique aCIC;
    bool Done= false;
    std::string  aTestFullName = "";

    if (aCCI.CalFromValues().IsInit())
    {
       Done=true;
       aCIC = aCCI.CalFromValues().Val();
    }


    // bool Test= TheSpecMess=="PbMehdi";

    if ((!Done) && (aCCI.CalFromFileExtern().IsInit()))
    {

        std::string aDirAdd =  aCCI.Directory().Val();
        if ( isUsingSeparateDirectories() )
            aDirAdd = MMOutputDirectory() + aDirAdd;
        else if (aCCI.AddDirCur().Val())
            aDirAdd = anAppli.DC() + aDirAdd;


        cSpecExtractFromFile aSEF = aCCI.CalFromFileExtern().Val();
        if (aPC)
        {
            ELISE_ASSERT(aCCI.CalibPerPose().IsInit(),"Internal errour in cCalibCam::Alloc"); 
            cCalibPerPose aCPP = aCCI.CalibPerPose().Val();
            if (aCPP.KeyInitFromPose().IsInit())
            {
               cNameFileWithExistigChangeVersionNameCam aNWECV(aDirAdd,anAppli.ICNM(),aCPP.KeyInitFromPose().Val(),aPC->Name());


               aSEF.NameFile() = aNWECV.GetName();
               // aSEF.NameFile() = anAppli.ICNM()->Assoc1To1(aCPP.KeyInitFromPose().Val(),aPC->Name(),true);
            }
        }

        //std::string aFullName = aDirAdd + aSEF.NameFile(); 
        std::string aFullName = aSEF.NameFile();
        aTestFullName = aFullName;


        bool IsExistFile = ELISE_fp::exist_file(aFullName);


        if ((!IsExistFile) && ELISE_fp::exist_file(aDirAdd + aFullName))
        {
              aFullName = aDirAdd + aFullName;
              IsExistFile = true;
        }

        if (IsExistFile)
        {
            aCIC = StdGetObjFromFile<cCalibrationInternConique>
	           (
	               aFullName,
		       StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
		       aSEF.NameTag(),
		       "CalibrationInternConique"
	           );
             Done = true;
         }
         else
         {
              if (! aSEF.AutorizeNonExisting().Val())
              {
                  std::cout << "For file " << aFullName << "\n";
                  ELISE_ASSERT(false,"Required calibration file do not exist");
              }
         }
    }



    if ((!Done) && (aCCI.CalibFromMmBD().Val()))
    {
       
        if (aPC)
        {
            std::string aNameIm = anAppli.DC()+aPC->Name();
            std::string aNameCal = StdNameGeomCalib(aNameIm);

            if (aNameCal!="")
            {
                aCIC = StdGetObjFromFile<cCalibrationInternConique>
	               (
	                   aNameCal,
		           StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
		           "CalibrationInternConique",
		           "CalibrationInternConique"
	               );
                 Done = true;
                 std::cout << "BDCAL " << aNameCal << "\n";
            }

        }
        // std::string aName = StdNameGeomCalib();
    }

    if ((!Done) && (aCCI.CalibAutomNoDist().IsInit()))
    {
        cCalibAutomNoDist aCAD = aCCI.CalibAutomNoDist().Val();

        eTypeCalibAutom aTCA = aCAD.TypeDist();
        std::string aNameIm;
        if (aCCI.CalibPerPose().IsInit())
        {
             // ELISE_ASSERT(false,"Calib Per Pose Autom to implement");
             ELISE_ASSERT(aPC,"hh ; Internal errour in cCalibCam::Alloc")
             aNameIm = aPC->Name();
        }
        else
        {
            aNameIm = aCAD.NameIm().Val("NameIm of CalibAutomNoDist");
        }

        if (aTCA == eCalibAutomNone)
        {
            std::cout << "For name " << aNameIm << "\n";
            ELISE_ASSERT(false,"Cannot determine internal calibration");
        }

        cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(anAppli.DC()+aNameIm);
        aCIC   = CalibInternAutom
                 (
                      aMDP,
                      aTCA,
                      aNameIm,
                      anAppli.Param().SeuilAutomFE().Val(),
                      aCAD.PositionRelPP().Val()
                 );
if (0)
{
   std::cout << "aaaaAAaHHhhhhJuii  " << aCIC.CalibDistortion().size() << "\n";
   cCalibDistortion aCD = aCIC.CalibDistortion()[0];
   std::cout << "IsInit= " << aCD.ModUnif().IsInit() << "\n";
   const cCalibrationInterneUnif & aCIU = aCD.ModUnif().Val();
   std::cout << "ETAT= " <<  aCIU.Etats() << "\n";

   getchar();
}
        if (aCAD.KeyFileSauv().IsInit())
        {
            std::string aNameSauv = anAppli.ICNM()->Assoc1To1(aCAD.KeyFileSauv().Val(),aKeyId,true);
            MakeFileXML(aCIC,anAppli.DC()+aNameSauv);
        }
        Done=true;
        // aCCI.
    }


    if (!Done)
    {
       if (aPC) std::cout << "For name = " << aPC->Name() << "\n";
       if (aTestFullName !="" )
        std::cout << "CANNOT GET " << aTestFullName << "\n";
       ELISE_ASSERT(false,"Cannot determine   calibration");
    }

    if (aCCI.DistortionAddInc().IsInit())
    {
       aCIC.CalibDistortion().push_back(aCCI.DistortionAddInc().Val());
    }

    if (aCCI.AddParamAFocal().IsInit())
    {
        const tParamAFocal &  aPAF = aCCI.AddParamAFocal().Val().Coeffs();
        ELISE_ASSERT
        (
           aPAF.size() == NbParamAF,
           "Bad number in AFOCAL parameters \n"
        );
        aCIC.ParamAF() = aPAF;
    }

    eConventionsOrientation aConv = aCIC.KnownConv().ValWithDef(aCCI.ConvCal().Val());



    cCalibDistortion  aCD = aCIC.CalibDistortion().back();
    AdaptDist2PPaEqPPs(aCD);

    std::string aNameIdCam = anAppli.GetNewIdCalib(aTestFullName);


    if (aCD.ModRad().IsInit())
    {
	cCamStenopeDistRadPol * aCam = Std_Cal_DRad_C2M(aCIC,aConv);
        aCam->SetIdentCam(aNameIdCam );
        anAppli.NormaliseScTr(*aCam);
        cParamIFDistRadiale * aPIF = aCam->AllocDRadInc(aCam->DistIsC2M(),anAppli.SetEq());

        cCalibCam_ModeleRadial * aRes =  new cCalibCam_ModeleRadial(aKeyId,anAppli,aCCI,aCIC,*aPIF,*aCam);


        return  aRes;
        
    }
    else if (aCD.ModPhgrStd().IsInit())
    {
        cCamStenopeModStdPhpgr * aCam =  Std_Cal_PS_C2M(aCIC,aConv);
        aCam->SetIdentCam(aNameIdCam );

        anAppli.NormaliseScTr(*aCam);
        cParamIFDistStdPhgr *  aPIF = aCam->AllocPhgrStdInc(aCam->DistIsC2M(),anAppli.SetEq());

        return  new cCalibCam_PhgrStd(aKeyId,anAppli,aCCI,aCIC,*aPIF,*aCam);
    }
    else if (aCD.ModUnif().IsInit())
    {
       cCamera_Param_Unif_Gen * aCam = Std_Cal_Unif(aCIC,aConv);
// std::cout << "CCaaaam "<< aCam << "\n";
       anAppli.NormaliseScTr(*aCam);
       aCam->SetIdentCam(aNameIdCam );

       cPIF_Unif_Gen * aPIF = aCam->PIF_Gen(aCam->DistIsC2M(),anAppli.SetEq());

       return new cCalibCam_ModeleUnif(aCam->IsFE(),aKeyId,anAppli,aCCI,aCIC,*aPIF,*aCam);
    }
    else if (aCD.ModGridDef().IsInit())
    {
        cCamStenopeBilin * aCBL =  Std_Cal_Bilin(aCIC,aConv);
        aCBL->SetIdentCam(aNameIdCam );
        anAppli.NormaliseScTr(*aCBL);

        cPIF_Bilin *  aPIF = anAppli.SetEq().NewPIFBilin(aCBL);

        return new cCalibCam_BiLin(aKeyId,anAppli,aCCI,aCIC,*aPIF,*aCBL);
        // ELISE_ASSERT(false,"aCD.ModGridDef : to finish");
    }
    else
    {
        ELISE_ASSERT(false,"Use a (still) unsuported init of calibration");
    }
    return 0;
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
