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


class cAppliMMOnePair;  // Appli principale
class cMMOnePair;  // Cree pour gerer l'ordre d'initialisatrion (a faire avant cAppliWithSetImage), afin de transformer en Pattern les 2 images


std::string Pair2PattWithoutDir(std::string & aDirRes,const std::string & aName1,const std::string & aName2);
std::string Name2PattWithoutDir(std::string & aDirRes,const std::vector<std::string>  & aVName1);

static std::string aBlk =  " " ;

class cMMOnePair
{
    public :
      cMMOnePair(int argc,char ** argv);
    protected :

      void ExeCom(const std::string &);

      std::string NamePx(int aStep) {return "Px1_Num"+ ToString(aStep) + "_DeZoom"+ ToString(mVZoom[aStep]) + "_LeChantier.tif";}
      std::string NameAutoM(int aStep,std::string aPost = "" ) {return "AutoMask_LeChantier_Num_" + ToString(ElMin(aStep,mStepEnd-1)) + aPost +".tif";}

      bool             mExe;
      bool             mShow;
      bool             mMM1PInParal;
      int              mZoom0;
      int              mZoomF;
      int              mStepEnd;
      std::vector<int> mVZoom;
      bool             mModeEpip;
      bool             mCreateEpip;


      bool             mCMS;
      bool             mForceCreateE;
      std::string      mNameIm1Init;
      std::string      mNameIm2Init;
      std::string      mNameOriInit;

      std::string      mNameIm1;
      std::string      mNameIm2;
      std::string      mNameOri;
      cCpleEpip        *mCpleE;
      bool             mDoubleSens;
      bool             mNoOri;


      std::string      mDirP;
      std::string      mPatP;
      std::vector<char *>        mArgcAWS;  // AppliWithSetImage
      bool             mDoMR;
      double           mSigmaP;
      Box2di           mBoxIm;
      bool             mPurge;
      bool             mPurgeAtEnd;
      std::string      mBascMTD;
      bool             mRIE;
      std::string      mBascDEST;
      bool             mDoHom;
      int              mDegCorrEpip;
      eTypeQuality     mQualOr;
      std::string      mStrQualOr;
      bool             mDoPly;
      int              mScalePly;
      bool             mDoOnlyMF;
      bool             mDebugCreatE;
      int              mNbCommand;
      const std::string mNameMasqFinal;
      bool              mHasVeget;
      bool              mSkyBackgGound;
      std::string       mMM1PMasq3D;
      bool	        mUseGpu;
      double            mDefCor;
      double            mZReg;
      bool		mExpTxt;
      bool              mUseCensQ;
      std::string       mModeCensus;
      int               mNbS;
      int               mSzW0;
};

class cAppliMMOnePair : public cMMOnePair,
                        public cAppliWithSetImage
{
     public :
         cAppliMMOnePair(int argc,char ** argv);
     private :
         void PurgeOneFileEpi(const std::string & aName,const std::string & aPost);
         void PurgeFileEpi(const std::string & aName);
         void MatchTwoWay(int aStep0,int aStepF);
         void MatchOneWay(bool MasterIs1,int aStep0,int aStepF,bool ForMTD);
         void DoMasqReentrant(bool First,int aStep,bool Last);
         void SauvMasqReentrant(bool First,int aStep,bool Last);
         void SymetriseMasqReentrant();
         void UseReentrant(bool First,int aStep,bool Last);
         void GenerateMTDEpip(bool MasterIs1);
         void BasculeGround(bool MasterIs1);
         void BasculeEpip(bool MasterIs1);

         cImaMM * mIm1;
         cImaMM * mIm2;
};

/*****************************************************************/
/*                                                               */
/*             cMMOnePair                                        */
/*                                                               */
/*****************************************************************/


cMMOnePair::cMMOnePair(int argc,char ** argv) :
    mExe          (true),
    mShow         (false),
    mMM1PInParal  (true),
    mZoom0        (64),
    mZoomF        (1),
    mModeEpip     (true),
    mCreateEpip   (true),
    mCMS          (true),
    mForceCreateE (false),
    mCpleE        (0),
    mDoubleSens   (true),
    mNoOri        (false),
    mDoMR         (true),
    mSigmaP       (1.5),
    mPurge        (true),
    mPurgeAtEnd   (false),
    mRIE          (false),
    mBascDEST     ("Basculed-"),
    mDoHom        (false),
    mDegCorrEpip  (4),
    mQualOr       (eQual_High),
    mDoPly        (false),
    mScalePly     (4),
    mDoOnlyMF     (false),
    mDebugCreatE  (false),
    mNbCommand    (-1),
    mNameMasqFinal ("Masq_Etape_Last.tif"),
    mHasVeget       (false),
    mSkyBackgGound  (true),
    mUseGpu	    (false),
    mDefCor         (0.5),
    mZReg           (0.05),
    mExpTxt	    (false),
    mUseCensQ       (false),
    mModeCensus     ("eMCC_CensusCorrel"),
    mNbS            (3)
{
  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(mNameIm1Init,"Name Im1", eSAM_IsExistFile)
                    << EAMC(mNameIm2Init,"Name Im2", eSAM_IsExistFile)
                    << EAMC(mNameOriInit,"Orientation (if NONE, work directly on epipolar)", eSAM_IsExistDirOri),
        LArgMain()  << EAM(mExe,"Exe",true,"Execute Commands, else only print them (Def=true)", eSAM_IsBool)
                    << EAM(mShow,"Show",true,"Show Commande", eSAM_IsBool)
                    << EAM(mZoom0,"Zoom0",true,"Zoom Init (Def=64)",eSAM_IsPowerOf2)
                    << EAM(mZoomF,"ZoomF",true,"Zoom Final (Def=1)",eSAM_IsPowerOf2)
                    << EAM(mCreateEpip,"CreateE",true," Create Epipolar (def = true when appliable)", eSAM_IsBool)
                    << EAM(mDoubleSens,"2Way",true,"Match in 2 Way (Def=true)", eSAM_IsBool)
                    << EAM(mCMS,"CMS",true,"Multi Scale Correl (Def=ByEpip)")
                    << EAM(mDoMR,"DoMR",true,"Do re-entering masq (Def=true)", eSAM_IsBool)
                    << EAM(mSigmaP,"SigmaP",true,"Sigma Pixel for coherence (Def=1.5)")
                    << EAM(mBoxIm,"BoxIm",true,"Box of calc in Epip, tuning purpose (Def=All image)",eSAM_InternalUse)
                    << EAM(mPurge,"Purge",true,"Purge directory, tuning (Def=true)", eSAM_InternalUse)
                    << EAM(mPurgeAtEnd,"PurgeAtEnd",true,"Purge Final Resul", eSAM_IsBool)
                    << EAM(mMM1PInParal,"InParal",true,"Do it in parallel (Def=true)", eSAM_IsBool)
                    << EAM(mBascMTD,"BascMTD",true,"Metadata of file to bascule (Def No Basc)")
                    << EAM(mRIE,"RIE",true,",Inverse re-sampling from epipolar", eSAM_IsBool)
                    << EAM(mBascDEST,"BascMTD",true,"Res of Bascule (Def Basculed-)")
                    << EAM(mDoHom,"DoHom",true,"Do Hom in epolar (Def=false)", eSAM_IsBool)
                    << EAM(mDegCorrEpip,"DegCE",true,"Degree of epipolar correction when Qual Orient is not high (Def=4)")
                    << EAM(mStrQualOr,"QualOr",true,"Quality orient (in High, Average, Low, Def= Low)",eSAM_None,ListOfVal(eNbTypeQual,"eQual_"))
                    << EAM(mDoPly,"DoPly",true,"Generate Ply", eSAM_IsBool)
                    << EAM(mScalePly,"ScalePly",true,"Dowsize of generated Ply (Def=4)")
                    << EAM(mDoOnlyMF,"DoOMF",true,"Do Only Masq Final (tuning purpose)", eSAM_InternalUse)
                    << EAM(mDebugCreatE,"DCE",true,"Debug Create Etpi (tuning purpose)", eSAM_InternalUse)
                    << EAM(mHasVeget,"HasVeg",true,"Has vegetation, Def= false", eSAM_IsBool)
                    << EAM(mSkyBackgGound,"HasSBG",true,"Has Sky Background , Def= true", eSAM_IsBool)
                    << EAM(mMM1PMasq3D,"Masq3D",true,"Masq 3D to filter points", eSAM_IsBool)
                    << EAM(mUseGpu,"UseGpu",false,"Use cuda (Def=false)")
                    << EAM(mDefCor,"DefCor",false,"Def cor (Def=0.5)")
                    << EAM(mZReg,"ZReg",false,"Regularisation factor (Def=0.05)")
		    << EAM(mExpTxt,"ExpTxt",false,"Use txt tie points for generating epipolar geometry (Def false, e.g. use dat format)")
		    << EAM(mNbS,"NbS",false,"Nb Scale,def=3")
		    << EAM(mSzW0,"SzW0",false,"Sz first Windows, def depend of NbS (1 MS, 2 no MS)")
		    << EAM(mUseCensQ,"CensusQ",false,"Use Census Quantitaive")
  );

  if (mUseCensQ)
  {
      mModeCensus  = "eMCC_CensusQuantitatif"; 
      if (! EAMIsInit(&mZReg))   mZReg = 0.01;
      if (! EAMIsInit(&mDefCor)) mDefCor = 0.90;
  }
  if (! EAMIsInit(&mSzW0))
  {
     mSzW0 =  (mNbS==1) ? 2 : 1;
  }

  if (!mExe) mShow = true;

  if (MMVisualMode) return;

  mNoOri = (mNameOriInit=="NONE");
  if (mNoOri)
  {
       mQualOr = eQual_High;
       mCreateEpip = false;
       mModeEpip =   true;
  }
  else
  {
      mCreateEpip = mModeEpip;
  }

  if (mNameIm1Init > mNameIm2Init)
     ElSwap(mNameIm1Init,mNameIm2Init);

  if (EAMIsInit(&mStrQualOr))
     mQualOr = Str2eTypeQuality("eQual_"+mStrQualOr);


  mDoHom = mDoHom || (mQualOr!= eQual_High);
  if (mQualOr==eQual_High)
    mDegCorrEpip = -1;
  else
    mDegCorrEpip = ElMax(1,mDegCorrEpip);

  mDirP =DirOfFile(mNameIm1Init);

  if (! mNoOri)
      StdCorrecNameOrient(mNameOriInit,mDirP); ;

  if (!EAMIsInit(&mCMS))
     mCMS = mModeEpip;

   if (MPD_MM())
   {
       std::cout << "MODE-E=" << mModeEpip << " CMS=" << mCMS << "\n";
   }

  if (mQualOr==eQual_Low)
  {
      std::string aCom =        MMBinFile(MM3DStr)
                               + std::string(" MMHomCorOri ")
                               + " " + mNameIm1Init
                               + " " + mNameIm2Init
                               + " " + mNameOriInit
                               + " Match=true "
                        ;
      if (mZoomF!=1)
         aCom = aCom + " ZoomF=4 ";
      ExeCom(aCom);
  }

  if (mCreateEpip)
  {
       mCpleE = StdCpleEpip(mDirP,mNameOriInit,mNameIm1Init,mNameIm2Init);
       mNameIm1 =  mCpleE->LocNameImEpi(mNameIm1Init);
       mNameIm2 =  mCpleE->LocNameImEpi(mNameIm2Init);
       mNameOri =  "Epi";
       if (
               (! ELISE_fp::exist_file(mDirP+mNameIm1))
            || (! ELISE_fp::exist_file(mDirP+mNameIm2))
            || mForceCreateE
          )
       {
             std::string aCom =        MMBinFile(MM3DStr)
                               + std::string(" CreateEpip ")
                               + " " + mNameIm1Init
                               + " " + mNameIm2Init
                               + " " + mNameOriInit
                               + " InParal=" + ToString(mMM1PInParal)
			       + " ExpTxt=" + ToString(mExpTxt)
                              ;

             if (mDegCorrEpip >=0)
             {
                  aCom = aCom + " Degre=" + ToString(mDegCorrEpip) + " ";
                  aCom = aCom + " NameH=" +  ((mQualOr==eQual_Average)? " " : "-DenseM ");
             }
             else if (mDoHom)
                  aCom = aCom + " NameH= " ;
    // mDoHom        (false),
    // mDegCorrEpip  (-1)
// mm3d CreateEpip MVxxxx_MAP_7078.NEF MVxxxx_MAP_7079.NEF Step4  DoIm=true DoHom=true Degre=1



             // System(aCom);  //cMMOnePair Car sinon la non existence des epi, bloque le reste a cause a AppliWitSetImage
             ExeCom(aCom);
       }
  }
  else
  {
       mNameIm1 = mNameIm1Init;
       mNameIm2 = mNameIm2Init;
       mNameOri = mNameOriInit;
  }

  mPatP = Pair2PattWithoutDir(mDirP,mNameIm1,mNameIm2);
  mPatP = mDirP+mPatP;
  mArgcAWS.push_back(const_cast<char *>(mPatP.c_str()));
  mArgcAWS.push_back(const_cast<char *>(mNameOri.c_str()));
}

void cMMOnePair::ExeCom(const std::string & aCom)
{
    mNbCommand++;
    if (mShow)
    {
        std::cout << "================= COM " << mNbCommand << " ================\n";
        std::cout << aCom << "\n";
    }
    if (mExe)
    {
        // ExeCom(aCom);
        System(aCom);
        return;
    }
    if (mShow)
    {
            std::cout << " Done COM : " << mNbCommand << " \n";
    }
}


/*****************************************************************/
/*                                                               */
/*             cAppliMMOnePair                                   */
/*                                                               */
/*****************************************************************/

int FlagcAppliMMOnePair(int argc,char ** argv)
{
/*
   std::cout << "A0 " << argv[0] << "\n";
   std::cout << "A1 " << argv[1] << "\n";
   std::cout << "A2 " << argv[2] << "\n";
*/
   if ((argc>=4) && (std::string(argv[3])=="NONE"))
       return cAppliWithSetImage::TheFlagNoOri;

   return 0;
}


cAppliMMOnePair::cAppliMMOnePair(int argc,char ** argv) :
   cMMOnePair(argc,argv),
   cAppliWithSetImage(2,&(mArgcAWS[0]),FlagcAppliMMOnePair(argc,argv)),
   mIm1 (0),
   mIm2 (0)
{
    if (! EAMIsInit(&mZoom0))
    {
       mZoom0 = DeZoomOfSize(7e4);
    }
    mZoom0 = ElMax(mZoom0,mZoomF);
    // std::cout  << "ZZ " << mZoom0 << "\n"; getchar();

    mVZoom.push_back(-1);
    for (int aDZ = mZoom0 ; aDZ >= mZoomF ; aDZ /=2)
    {
         mVZoom.push_back(aDZ);
         if ((aDZ==mZoom0) || (aDZ==mZoomF))
             mVZoom.push_back(aDZ);
    }
    // mStepEnd = round_ni(log2(mZoom0/double(mZoomF))) + 3;
    mStepEnd = (int)(mVZoom.size() - 1);

    // std::cout << "STEP END = " << mStepEnd << " " << round_ni(log2(mZoom0/double(mZoomF))) + 3 << " :: " << mVZoom << "\n"; StdEXIT(0);

    int aK=0;
    for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
    {
        if (aK==0)
           mIm1 =  (*anITS).attr().mIma;
        if (aK==1)
           mIm2 =  (*anITS).attr().mIma;
        aK++;
    }
    ELISE_ASSERT(aK==2,"Expect exactly 2 images in cAppliMMOnePair");
    // mIm1 = mImages[0];
    // mIm2 = mImages[1];


   std::string aComPly =    MMBinFile(MM3DStr)
                          + " Nuage2Ply "
                          +  mEASF.mDir+LocDirMec2Im(mNameIm1,mNameIm2) + "NuageImProf_Chantier-Ori_Etape_Last.xml "
                          + " Attr=" + mNameIm1
                          + " RatioAttrCarte=" + ToString(mZoomF)
                          + " Scale=" + ToString(mScalePly)
                          + " Out=" +  LocDirMec2Im(mNameIm1,mNameIm2) +"PLY-" + mNameIm1 + "-"+ mNameIm1+".ply";
                        ;

   if (mDebugCreatE)
      return;


    if (false)
    {
       MatchTwoWay(1,mStepEnd+1);
    }
    else
    {
        for (int aStep=1 ; aStep<=mStepEnd ; aStep++)
        {
           if (! mDoOnlyMF)
           {
              MatchTwoWay(aStep,aStep+1);
           }
           if ((aStep==1) && mCreateEpip)  // Mis ici pour Nuage2Ply
           {
              GenerateMTDEpip(true);
              GenerateMTDEpip(false);
           }



           int aDeZoom = mVZoom[aStep];

           if (     mDoMR
                 && ((aDeZoom!= mZoomF) || (aStep==mStepEnd))
                 && (aDeZoom<=16)
                 && ((!mDoOnlyMF) || (aStep==mStepEnd))
              )
           {
              DoMasqReentrant(true,aStep,aStep==mStepEnd);
              DoMasqReentrant(false,aStep,aStep==mStepEnd);


              SauvMasqReentrant(true,aStep,aStep==mStepEnd);
              SauvMasqReentrant(false,aStep,aStep==mStepEnd);
           }
        }

        if (mDoPly)
        {
                ExeCom(aComPly);
        }

    }

    if (mRIE)
    {
       BasculeEpip(true);
       BasculeEpip(false);
    }


    if (EAMIsInit(&mBascMTD))
    {
       BasculeGround(true);
       BasculeGround(false);
    }

    if (mPurgeAtEnd)
    {
        if (mCreateEpip)
        {
           PurgeFileEpi(mNameIm1);
           PurgeFileEpi(mNameIm2);
        }
        ELISE_fp::PurgeDir(Dir() + LocDirMec2Im(mNameIm1,mNameIm2),true);
        ELISE_fp::PurgeDir(Dir() + LocDirMec2Im(mNameIm2,mNameIm1),true);
    }
}

void cAppliMMOnePair::PurgeOneFileEpi(const std::string & aName,const std::string & aPost)
{
     ELISE_fp::RmFile( Dir() + StdPrefix(aName) + aPost);
}

void cAppliMMOnePair::PurgeFileEpi(const std::string & aName)
{
    PurgeOneFileEpi(aName,".tif");
    PurgeOneFileEpi(aName,"_Masq.tif");
    PurgeOneFileEpi(aName,"_Masq.xml");
}




void cAppliMMOnePair::BasculeEpip(bool MasterIs1)
{
    std::string aNamInitA =  (MasterIs1 ? mNameIm1Init : mNameIm2Init);
    std::string aNamInitB =  (MasterIs1 ? mNameIm2Init : mNameIm1Init);

    std::string aCom =   MMBinFile(MM3DStr) + " TestLib RIE "
                                            + aBlk + aNamInitA
                                            + aBlk + aNamInitB
                                            + aBlk + mNameOriInit
                                            + aBlk + " Dir=" +  mEASF.mDir
                       ;
//  mm3d TestLib RIE MVxxxx_MAP_6937.NEF MVxxxx_MAP_6938.NEF Basc

    ExeCom(aCom);
}

void cAppliMMOnePair::BasculeGround(bool MasterIs1)
{
    std::string aNamA = MasterIs1 ? mNameIm1 : mNameIm2;
    std::string aNamB = MasterIs1 ? mNameIm2 : mNameIm1;
    std::string aNamInitA = MasterIs1 ? mNameIm1Init : mNameIm2Init;
    std::string aNamInitB = MasterIs1 ? mNameIm2Init : mNameIm1Init;

    std::string aDirMatch =  mEASF.mDir + LocDirMec2Im(aNamA,aNamB);
    std::string aNuageIn =  aDirMatch          + std::string("NuageImProf_Chantier-Ori_Etape_Last.xml");
    std::string aNuageGeom =     mEASF.mDir +  mBascMTD;
    std::string aNuageTarget =  mEASF.mDir +  DirOfFile(mBascMTD) + mBascDEST + aNamInitA + "-" + aNamInitB + ".xml";

    std::string aCom =   MMBinFile(MM3DStr) + " NuageBascule "
                           + aBlk + aNuageIn
                           + aBlk + aNuageGeom
                           + aBlk + aNuageTarget
                           + " SeuilE=500"
                           + aBlk + " Paral=" + ToString(mMM1PInParal)
                         ;
    ExeCom(aCom);

/*
         std::string aNuageGeom =    mDir +  std::string("MTD-Nuage/NuageImProf_LeChantier_Etape_1.xml");
         std::string aNuageTarget =  mDir +  std::string("MTD-Nuage/Basculed-")
                                          + ((aK==0) ? anI1.mNameIm : anI2.mNameIm )
                                          + "-" + ((aK==0) ? anI2.mNameIm :anI1.mNameIm) + ".xml";


         std::string aCom =   MMBinFile(MM3DStr) + " NuageBascule "
                           + aBlank + aNuageIn
                           + aBlank + aNuageGeom
                           + aBlank + aNuageTarget
                           + " SeuilE=500";
         std::cout << aCom << "\n";

*/
}


void cAppliMMOnePair::SymetriseMasqReentrant()
{
    std::string aDir1 =  mEASF.mDir+LocDirMec2Im(mNameIm1,mNameIm2);
    std::string aDir2 =  mEASF.mDir+LocDirMec2Im(mNameIm2,mNameIm1);
    std::string aNPx = NamePx(mStepEnd);
    std::string aNAM = NameAutoM(mStepEnd);

    int aSzY = Tiff_Im::StdConv(aDir1+aNPx).sz().y;

    std::string aCom =    MMBinFile(MM3DStr)
                        + std::string(" TestLib MMSMA ")
                        + aBlk + aDir1 + aNPx
                        + aBlk + aDir1 + aNAM
                        + aBlk + aDir2 + aNPx
                        + aBlk + aDir2 + aNAM
                        + aBlk + "0"
                        + aBlk + ToString(aSzY)
                        + aBlk + "0"
                        + aBlk + ToString(aSzY)
                        + aBlk + ToString(!mCpleE->IsLeft(true))
                        + " InParal=" + ToString(mMM1PInParal) ;

   ExeCom(aCom);

   for (int aK=0 ; aK <2 ; aK++)
   {
        std::string aDir = (aK==0) ? aDir1 : aDir2;

        ELISE_fp::MvFile(aDir+NameAutoM(mStepEnd,"_MSym"),aDir+aNAM);
   }
   std::cout << "cAppliMMOnePair::SymetriseMasqReentrant \n";
}




void cAppliMMOnePair::GenerateMTDEpip(bool MasterIs1)
{
    std::string aNamA = MasterIs1 ? mNameIm1 : mNameIm2;
    std::string aNamB = MasterIs1 ? mNameIm2 : mNameIm1;
    std::string aNameInitA = MasterIs1 ? mNameIm1Init : mNameIm2Init;

    MatchOneWay(MasterIs1,1,mStepEnd+1,true);

    double aMul = mCpleE->IsLeft(aNameInitA) ? -1 : 1;


    for (int aStep=1 ; aStep<=mStepEnd  ; aStep++)
    {
       bool IsLast = (aStep==mStepEnd);
       std::string aNameStep = IsLast ? "Last" : ToString(aStep);
       std::string aNameIn =   mEASF.mDir+LocDirMec2Im(aNamA,aNamB) + "NuageImProf_Chantier-Ori_Etape_"+ ToString(aStep) +".xml";
       std::string aNameOut =   mEASF.mDir+LocDirMec2Im(aNamA,aNamB) + "NuageImProf_Chantier-Ori_Etape_"+ aNameStep +".xml";

       cXML_ParamNuage3DMaille aNuage =  StdGetFromSI(aNameIn,XML_ParamNuage3DMaille);
       aNuage.Image_Profondeur().Val().Image() = NamePx(aStep);
       aNuage.Image_Profondeur().Val().Masq() =  NameAutoM(aStep);
       aNuage.Image_Profondeur().Val().ResolutionAlti() *= aMul;

       if (IsLast)
       {
            aNuage.Image_Profondeur().Val().Masq() =  mNameMasqFinal;
            aNuage.Image_Profondeur().Val().Correl().SetVal("Score-AR.tif");
       }
       else
            aNuage.Image_Profondeur().Val().Correl().SetNoInit();

       MakeFileXML(aNuage,aNameOut);
       if (IsLast)
       {
            MakeFileXML(aNuage,aNameIn);
       }

    }
}


void cAppliMMOnePair::DoMasqReentrant(bool MasterIs1,int aStep,bool aLast)
{
     std::string aNameInitA = MasterIs1 ? mNameIm1Init : mNameIm2Init;
     std::string aNameInitB = MasterIs1 ? mNameIm2Init : mNameIm1Init;
     std::string aPref = "AR"+ std::string(MasterIs1? "1" : "2") + "-" + aNameInitA + "-" + aNameInitB;

     int aZoom = mVZoom[aStep];
     std::string aName =  mEASF.mDir+LocDirMec2Im(mNameIm1,mNameIm2)+"Z_Num"+ToString(aStep)+"_DeZoom"+ToString(aZoom)+"_LeChantier.xml";
     cFileOriMnt aFOM = StdGetFromPCP(aName,FileOriMnt);
     double aResol = aFOM.ResolutionAlti() / double ( aFOM.ResolutionPlani().x);


     std::string aCom =     MMBinFile(MM3DStr)
                          + std::string(" CoherEpip ")
                          + aNameInitA + aBlk
                          + aNameInitB + aBlk
                          + mNameOriInit + aBlk
                          + " DoM=true"  // Pas utilise dans coher epip, et peu creer bug ...
                          + " ByE="      + ToString(mModeEpip)
                          + " NumPx="    + ToString(aStep)
                          + " NumMasq="  + ToString(aLast ? (aStep-1) : aStep)
                          + " Zoom="     + ToString(mVZoom[aStep])
                          + " Step="     + ToString(aResol)
                          + " SigP="     + ToString(mSigmaP * ((aZoom==1) ? 1.5 : 1.0))
                          + " Prefix="   + aPref
                          + " InParal="  + ToString(mMM1PInParal)
                          + " RegCh="  + ToString(! mHasVeget)
                          + " FBH="  + ToString( mSkyBackgGound)
                          + " Regul=0.5"
                      ;


     if (EAMIsInit(&mMM1PMasq3D)) aCom = aCom + " Masq3D=" +mMM1PMasq3D;

     aCom = aCom + " RedM=1.0 ";   // Avec la prog dyn, pas de raison de ne pas faire ts le temps a full resol
     if (aLast)
     {
        aCom = aCom + " ExpFin=true " ;
     }

     ExeCom(aCom);
}

void cAppliMMOnePair::SauvMasqReentrant(bool MasterIs1,int aStep,bool aLast)
{

     std::string aNamA = MasterIs1 ? mNameIm1 : mNameIm2;
     std::string aNamB = MasterIs1 ? mNameIm2 : mNameIm1;
     std::string aNameInitA = MasterIs1 ? mNameIm1Init : mNameIm2Init;
     std::string aNameInitB = MasterIs1 ? mNameIm2Init : mNameIm1Init;
     std::string aPref = "AR"+ std::string(MasterIs1? "1" : "2") + "-" + aNameInitA + "-" + aNameInitB;



     if (1) // (! mExe)
     {

           std::cout << "SauvMasqReentrant, M1=" << MasterIs1 << " S=" << aStep << " L=" << aLast << "\n";
           if (aLast)
           {
               std::string aName =  mEASF.mDir + aPref + "_Glob.tif";
               std::string aDest =  mEASF.mDir + LocDirMec2Im(aNamA,aNamB) + "Score-AR.tif";
               std::cout << "    MMVVV " << aName << " => " << aDest << "\n";
           }

           if (! mExe)
              return;
     }



     std::string aNameMasq =    aLast                                                             ?
                               //   ("AutoMask_LeChantier_Num_" + ToString(aStep-1)+".tif")           :
                               mNameMasqFinal                                                    :
                               ("Masq_LeChantier_DeZoom" + ToString(mVZoom[aStep+1]) +  ".tif")  ;
     aNameMasq =      mEASF.mDir +  LocDirMec2Im(aNamA,aNamB) + aNameMasq;
     std::string aNameNew = aPref + "_Masq1_Glob.tif";




/*
          std::string aNameNuage =   mEASF.mDir+LocDirMec2Im(aNamA,aNamB) + "NuageImProf_Chantier-Ori_Etape_"+ ToString(aStep) +".xml";
          std::string aCom =   MM3dBinFile("TestLib")
                           + " Masq3Dto2D "
                           + mMasq3D + std::string(" ")
                           + aNameNuage  + std::string(" ")
                           + aNameNew
                           + " MasqNuage=" + aNameNew;

          S-ystem(aCom);
     }
*/


     if (aLast)
     {
           ELISE_fp::CpFile(aNameNew,aNameMasq);
     }


     Tiff_Im aTifMasqCor(aNameMasq.c_str());
     Tiff_Im aFNew(aNameNew.c_str());

     Fonc_Num aFonc = aFNew.in(0);
     if (!aLast)
     {
        aFonc = dilat_32((close_32(aFonc,8)),4);

        int aZoomCur = mVZoom[aStep];
        int aZoomNext = mVZoom[aStep+1];
        double aRatio = aZoomNext / double(aZoomCur);

        aFonc = StdFoncChScale_Bilin(aFonc,Pt2dr(0,0),Pt2dr(aRatio,aRatio));
     }


     ELISE_COPY
     (
        aTifMasqCor.all_pts(),
        aTifMasqCor.in() && aFonc,
        aTifMasqCor.out()
     );

     if (aLast)
     {
           std::string aName =  mEASF.mDir + aPref + "_Glob.tif";
           std::string aDest =  mEASF.mDir + LocDirMec2Im(aNamA,aNamB) + "Score-AR.tif";


           ELISE_fp::MvFile(aName,aDest);

           aName =  mEASF.mDir + aPref + "_ImDistor_Glob.tif";
           aDest =  mEASF.mDir + LocDirMec2Im(aNamA,aNamB) + "Distorsion.tif";
           ELISE_fp::MvFile(aName,aDest);

           ELISE_fp::RmFile( mEASF.mDir + aPref + "*.tif");
     }
}


void cAppliMMOnePair::MatchTwoWay(int aStep0,int aStepF)
{
    for (int aK=0 ; aK<2 ; aK++)
    {
       bool First = (aK==0);
       if (mDoubleSens |First )
          MatchOneWay(First,aStep0,aStepF,false);
    }
}

void cAppliMMOnePair::MatchOneWay(bool MasterIs1,int aStep0,int aStepF,bool ForMTD)
{
     std::string aNamA = MasterIs1 ? mNameIm1 : mNameIm2;
     std::string aNamB = MasterIs1 ? mNameIm2 : mNameIm1;

     std::string aCom =     MMBinFile(MM3DStr)
                          + std::string(" MICMAC ")
                          +  XML_MM_File("MM-Epip.xml ")
                          + " WorkDir="  +  mEASF.mDir
                          + " +Im1="     + aNamA
                          + " +Im2="     + aNamB
                          + " +Zoom0="   + ToString(mZoom0)
                          + " +ZoomF="   + ToString(mZoomF)
                          + " FirstEtapeMEC=" + ToString(aStep0)
                          + " LastEtapeMEC=" + ToString(aStepF)
                          + " +Purge="   +  ToString(mPurge && (aStep0==1) && (!ForMTD))
                          + " +Ori="     + (ForMTD ? "Epi" :mNameOri)
                          + " +DoEpi="   + ToString((mModeEpip) && (!ForMTD))
                          + " +CMS="     + ToString(mCMS)
                          + " +DoOnlyXml="     + ToString(ForMTD)
                          + " +MMC="     + ToString(!ForMTD)
                          + " +NbProc=" + ToString(mMM1PInParal ? MMNbProc() : 1)
                          + " +UseGpu=" + ToString(mUseGpu)
                          + " +DefCor=" + ToString(mDefCor)
                          + " +ZReg="   + ToString(mZReg)
                          + " +ModeCensus=" + mModeCensus
                          + " +NbS=" + ToString(mNbS)
                          + " +SzW0=" + ToString(mSzW0)
                          + " "+  QUOTE( "+ExtImIn=("   + StdPostfix(mNameIm1) + "|" + StdPostfix(mNameIm2) + ")")
// FirstEtapeMEC=5 LastEtapeMEC=6
                      ;

     std::string aDyrPyram = mCreateEpip ? LocDirMec2Im(mNameIm1,mNameIm2) : "Pyram/";
     aCom = aCom+ " +DirPyram=" + aDyrPyram;

     if (mNoOri) aCom = aCom+ " +MasqImOptional=true";

     if (EAMIsInit(&mBoxIm))
     {
        aCom  = aCom + " +ClipCalc=true"
                     + std::string(" +X0Clip=") + ToString(mBoxIm._p0.x)
                     + std::string(" +Y0Clip=") + ToString(mBoxIm._p0.y)
                     + std::string(" +X1Clip=") + ToString(mBoxIm._p1.x)
                     + std::string(" +Y1Clip=") + ToString(mBoxIm._p1.y)
              ;
     }



/*
     bool AddPly = (!ForMTD) && ((aStepF-1)== mStepEnd)  && (mDoPly)  && (MasterIs1);
     if (AddPly)
     {
          aCom = aCom + " +DoPly=true " + " +ScalePly=" + ToString(mScalePly) +  " ";
     }
*/

     ExeCom(aCom);

}

/*****************************************************************/
/*                                                               */
/*                ::                                             */
/*                                                               */
/*****************************************************************/

std::string Name2PattWithoutDir(std::string & aDirRes,const std::vector<std::string>  & aVName)
{
   ELISE_ASSERT(aVName.size()!=0,"Name2Patt");
   std::string aRes = "(";

   for (int aK=0 ; aK<int(aVName.size()) ; aK++)
   {
        std::string aDir,aName;
        SplitDirAndFile(aDir,aName,aVName[aK]);
        if (aK==0)
        {
            aDirRes = aDir;
        }
        else
        {
            ELISE_ASSERT(aDirRes == aDir,"Variable dir in Name2PattWithoutDir");
            aRes += "|";
        }
        aRes += "("  + aName + ")";
   }
   return aRes + ")";
}

std::string Pair2PattWithoutDir(std::string & aDir,const std::string & aName1,const std::string & aName2)
{
    std::vector<std::string> aVName;
    aVName.push_back(aName1);
    aVName.push_back(aName2);
    return Name2PattWithoutDir(aDir,aVName);
}

int MMOnePair_main(int argc,char ** argv)
{
    // for (int aK=0 ; aK<argc ; aK++) std::cout <<  "**[" <<  argv[aK] << "]\n";


   cAppliMMOnePair aMOP(argc,argv);

    return 0;
}

static std::string NameMasqSym(const std::string &aNameMasq)
{
    return StdPrefix(aNameMasq) + "_MSym.tif";
}

static Tiff_Im MakeMaskSauv(const std::string &aNameMasq)
{
   Tiff_Im aTifIn(aNameMasq.c_str());
   std::string aNameRes = NameMasqSym(aNameMasq);

   return Tiff_Im
          (
                 aNameRes.c_str(),
                 aTifIn.sz(),
                 GenIm::bits1_msbf,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
          );
}

class cOneISMR
{
    public :
        cOneISMR
        (
            const std::string &  aNamePx,
            const std::string &  aNameMasq,
            int                  aY0In,
            int                  aY1In,
            int                  aY0Out,
            int                  aY1Out,
            bool                 isRight
        )  :
           mFPx       (Tiff_Im::StdConv(aNamePx)),
           mFMasq     (Tiff_Im::StdConv(aNameMasq)),
           mFileSauv  (Tiff_Im::StdConv(NameMasqSym(aNameMasq))),
           mSzX       (mFPx.sz().x),
           mSzY       (aY1In-aY0In),
           mSzIn      (mSzX,mSzY),
           mP0In      (0,aY0In),
           mImPX      (mSzIn.x,mSzIn.y),
           mTPx       (mImPX),
           mImMasqIn  (mSzIn.x,mSzIn.y),
           mTMIn      (mImMasqIn),
           mImMasqOut (mSzIn.x,mSzIn.y),
           mTMOut     (mImMasqOut),
           mRightIm   (isRight),
           mY0Out     (aY0Out),
           mY1Out     (aY1Out)
        {
            ELISE_COPY(mImPX.all_pts(),trans(mFPx.in(),mP0In),mImPX.out());
            ELISE_COPY(mImMasqIn.all_pts(),trans(mFMasq.in(),mP0In),mImMasqIn.out());
            for (int aY=0 ; aY<mSzY ; aY++)
            {
                mTMIn.oset(Pt2di(0,aY),0);
                mTMIn.oset(Pt2di(mSzX-1,aY),0);
            }
            ELISE_COPY(mImMasqIn.all_pts(),mImMasqIn.in(),mImMasqOut.out());
        }
        bool OkX(int anX) {return (anX>=0) && (anX<mSzX);}

        void DoOneSen(cOneISMR &);

   private :

        Tiff_Im      mFPx;
        Tiff_Im      mFMasq;
        Tiff_Im      mFileSauv;
        int          mSzX;
        int          mSzY;
        Pt2di        mSzIn;
        Pt2di        mP0In;
        Im2D_REAL4   mImPX;
        TIm2D<REAL4,REAL> mTPx;
        Im2D_Bits<1> mImMasqIn;
        TIm2DBits<1> mTMIn;
        Im2D_Bits<1> mImMasqOut;
        TIm2DBits<1> mTMOut;
        bool         mRightIm;
        int          mY0Out;
        int          mY1Out;
};

void cOneISMR::DoOneSen(cOneISMR & anI2)
{
   // int aSzX2 = anI2.mSzX;
   for (int aY=0 ; aY<mSzY ; aY++)
   {
/*
        Im1D_INT4 anImNum(aSzX2);
        INT4 * aDN = anImNum.data();
        int aNum = -1;
        bool Prev = false;
        for (aP.x=0; aP.x<aSzX2; aP.x++)
        {
            bool Next = anI2.mTMIn.get(aP);
            if (Next && (!Prev))
            {
               aNum++;
            }
            aDN[aP.x] = Next ? aNum : -1;
            Prev = Next;
        }
*/

        int aX0   = mRightIm ? 0      : mSzX-1;
        int aXEnd = mRightIm ? mSzX-1 :     0;
        int aStep = mRightIm ? 1      :     -1;

        Pt2di aP(0,aY);
        Pt2di aPNext(0,aY);
        for ( aP.x = aX0 ; aP.x!=aXEnd ; aP.x+=aStep)
        {
             // aPNext.x = aP.x+aStep;
             if (mTMIn.get(aP))
             {
                 int aX2 = aP.x + round_ni(mTPx.get(aP));
                 // int aX2N = aPNext.x + round_ni(mTPx.get(aPNext));
                 if (anI2.OkX(aX2) && (!anI2.mTPx.get(Pt2di(aX2,aY))))
                    mTMOut.oset(aP,0);
             }
        }
   }

   ELISE_COPY
   (
        rectangle(Pt2di(0,mY0Out),Pt2di(mSzX,mY1Out)),
        trans(mImMasqOut.in(),Pt2di(0,-mY0Out)),
        mFileSauv.out()
   );
}

int MMSymMasqAR_main(int argc,char ** argv)
{

  std::string aPx1,aPx2,aMasq1,aMasq2;
  int aY0In,aY1In,aY0Out,aY1Out;
  bool aFirsIsRight;
  double  aStep = 1.0;
  bool   CalleByP = false;
  bool   InParal = false;

  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(aPx1,"Name1")
                    << EAMC(aMasq1,"Masq1")
                    << EAMC(aPx2,"Name2")
                    << EAMC(aMasq2,"Masq2")
                    << EAMC(aY0In,"Y0In")
                    << EAMC(aY1In,"Y1In")
                    << EAMC(aY0Out,"Y0Out")
                    << EAMC(aY1Out,"Y1Out")
                    << EAMC(aFirsIsRight,"FR"),
        LArgMain()  << EAM(aStep,"Step",true,"Step of pax")
                    << EAM(CalleByP,"CalleByP","Internal use")
                    << EAM(InParal,"InParal","Internal use")
  );

  if (! CalleByP)
  {
     MakeMaskSauv (aMasq1);
     MakeMaskSauv (aMasq2);

     Pt2di  aSzXY = Tiff_Im::StdConv(aPx1).sz();
     int aSzX = aSzXY.x;
     int aSzY = aSzXY.y;

     cDecoupageInterv1D aDec = cDecoupageInterv1D
                               (
                                    cInterv1D<int>(0,aSzY),
                                    1e7 / aSzX,
                                    cInterv1D<int>(-10,10)
                               );
      std::list<std::string> aLCom;
      for (int aK=0 ; aK<aDec.NbInterv() ; aK++)
      {
           cInterv1D<int> aII =  aDec.KthIntervIn(aK);
           cInterv1D<int> aIO =  aDec.KthIntervIn(aK);
           std::string aCom =    MMBinFile(MM3DStr)
                                 + std::string(" TestLib MMSMA ")
                                 + aBlk + aPx1
                                 + aBlk + aMasq1
                                 + aBlk + aPx2
                                 + aBlk + aMasq2
                                 + aBlk + ToString(aII.V0())
                                 + aBlk + ToString(aII.V1())
                                 + aBlk + ToString(aIO.V0())
                                 + aBlk + ToString(aIO.V1())
                                 + aBlk + ToString(aFirsIsRight)
                                 + " Step=" + ToString(aStep)
                                 + " CalleByP=1";

             aLCom.push_back(aCom);
      }
      if (InParal)
         cEl_GPAO::DoComInParal(aLCom,"MakeSymMasq");
      else
         cEl_GPAO::DoComInSerie(aLCom);
  }
  else
  {

      cOneISMR  aI1(aPx1,aMasq1,aY0In,aY1In,aY0Out,aY1Out,aFirsIsRight);
      cOneISMR  aI2(aPx2,aMasq2,aY0In,aY1In,aY0Out,aY1Out,!aFirsIsRight);

      aI1.DoOneSen(aI2);
      aI2.DoOneSen(aI1);
  }
  // Im2D_REAL4 aImPX1(aSz1In);

  return 0;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
