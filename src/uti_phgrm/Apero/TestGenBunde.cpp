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



/***********************************************************************/
/*                                                                     */
/*                                                                     */
/*                                                                     */
/***********************************************************************/


class cCamTest_PBGC3M2DF;
class cTest_PBGC3M2DF;

class cCamTest_PBGC3M2DF
{
    public :
         cCamTest_PBGC3M2DF(cImaMM &,cTest_PBGC3M2DF&,int aK);
       
    // private :

       cTest_PBGC3M2DF *       mAppli; 
       cImaMM *                mIma;
       std::string             mNameIm;
       CamStenope *            mCS0;
       std::string             mNameSaveCS0;
       ElMatrix<double>        mMatPert;
       CamStenope *            mCSCur;
       cPolynomial_BGC3M2D     mPolCam;
       cPolynBGC3M2D_Formelle  mFPC;
       double                  mNbMesPts; 
       double                  mSomPdsMes; 
       int                     mK;

       void Show();
};

void cCamTest_PBGC3M2DF::Show()
{
   mFPC.TypedCamCur()->Show();
}

class cSetCTest_PBGC3M2DF
{
    public :
         std::vector<cCamTest_PBGC3M2DF *>   mVCams;
         ElPackHomologue        mPack12;     
         ElPackHomologue        mPack21;     
         std::vector<Pt2dr>     mVP[3];
         cSubstitueBlocIncTmp * mBufSub;

         const cBasicGeomCap3D * KCamCur(int aKC) const {return mVCams[aKC]->mFPC.GPF_CurBGCap3D();}

         const cBasicGeomCap3D * Cam0(int aKC) const {return mVCams[aKC]->mCS0;}
         
};


class cTest_PBGC3M2DF : public cAppliWithSetImage
{
    public :
       cTest_PBGC3M2DF(int argc,char ** argv);
    // private :
       cSetEqFormelles *                       mSet;
       cEqfP3dIncTmp *                         mEqP3I;
       std::string                             mPat;
       std::string                             mOri;
       std::vector<cCamTest_PBGC3M2DF *>       mVCT;
       std::vector<cSetCTest_PBGC3M2DF *>      mVCpleT;
       std::vector<cSetCTest_PBGC3M2DF *>      mVTriT;
       std::map<Pt2di,cSetCTest_PBGC3M2DF *>  mMapCpleT;
       int                                     mDeg;
       int                                     mNbSom;
       double                                  mPerturbAng;
       double                                  mPerturbPol;
       bool                                    mPerfectData;
       CamStenope *                            CamPerturb(CamStenope *, ElMatrix<double> &);

       bool HasArc(int aK1, int aK2)
       {
            if (aK1>aK2) ElSwap(aK1,aK2);
            return DicBoolFind(mMapCpleT,Pt2di(aK1,aK2));
       }
       double RandAngle() {return mPerturbAng * NRrandC();}

       void OneIterBundle();
       double AddBundle(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS,double anErr);
       void SetPerfectData(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS);
};

cCamTest_PBGC3M2DF::cCamTest_PBGC3M2DF(cImaMM & anIma,cTest_PBGC3M2DF& anAppli,int aK) :
   mAppli   (& anAppli),
   mIma     (& anIma),
   mNameIm  (mIma->mNameIm),
   mCS0     (mIma->CamSNN()),
   mMatPert (3,3),
   mCSCur   (mAppli->CamPerturb(mCS0,mMatPert)),
   mPolCam  (0,mCSCur,"Test","tutu",anAppli.mDeg,anAppli.mPerturbPol),  // TAGGG
   mFPC     (*(mAppli->mSet),mPolCam,false,false,false),
   mNbMesPts (0.0),
   mSomPdsMes (0.0),
   mK       (aK)
{
     std::cout << " cCamTest_PBGC3M2DF: " << mNameIm << "\n";
     {
         std::string aDest = "PertTestBundle";
         mNameSaveCS0 = mAppli->EASF().mICNM->Assoc1To1("NKS-Assoc-Im2UnCorMMOrient@-"+aDest,NameWithoutDir(mNameIm),true);
         MakeFileXML(mCSCur->StdExportCalibGlob(),mNameSaveCS0);
         
	 cPolynomial_BGC3M2D aPol(0,mCSCur,mNameSaveCS0,mNameIm,mAppli->mDeg); // TAGG
         aPol.Save2XmlStdMMName(0,"",aDest,ElAffin2D::Id());
     }
}


CamStenope * cTest_PBGC3M2DF::CamPerturb(CamStenope * aCS0,ElMatrix<double> & aMPert)
{
   CamStenope * aCS = aCS0->Dupl();
   ElRotation3D  aR =aCS->Orient().inv();

   // int i = 3 +  5* 4;

   aMPert  = ElMatrix<double>::Rotation(RandAngle(),RandAngle(),RandAngle());
   aR = ElRotation3D(aR.tr(),aR.Mat()*aMPert,true);
   aCS->SetOrientation(aR.inv());

   return aCS;
}



void cTest_PBGC3M2DF::SetPerfectData(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS)
{
    for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
    {
       cSetCTest_PBGC3M2DF * aSet = aVS[aKS];
       int aNbCam = (int)aSet->mVCams.size();
       int aNbP = (int)aSet->mVP[0].size();
       std::vector<Pt2dr>  aNewVP[3];
       for (int aKP=0 ; aKP<aNbP ; aKP++)
       {
          std::vector<Pt3dr> aVP0;
          std::vector<Pt3dr> aVP1;
          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              // ElSeg3D aSeg = aSet->mVCams[aKC]->mFPC.GPF_CurBGCap3D()->Capteur2RayTer(aSet->mVP[aKC][aKP]);
              ElSeg3D aSeg = aSet->Cam0(aKC)->Capteur2RayTer(aSet->mVP[aKC][aKP]);
              aVP0.push_back(aSeg.P0());
              aVP1.push_back(aSeg.P1());
          }
          bool Ok;
          Pt3dr  aPImTer = InterSeg(aVP0,aVP1,Ok);
          bool AllOk = true;

          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              if (!aSet->Cam0(aKC)->PIsVisibleInImage(aPImTer))
              {
                  AllOk = false;
              }
          }

          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              Pt2dr aProj  = aSet->Cam0(aKC)->Ter2Capteur(aPImTer);
              if (!aSet->Cam0(aKC)->CaptHasData(aProj))
              {
                  AllOk = false;
              }
          }

          if (AllOk)
          {
              for (int aKC=0 ; aKC<aNbCam ; aKC++)
              {
                  aNewVP[aKC].push_back(aSet->Cam0(aKC)->Ter2Capteur(aPImTer));
              }
          }
       }
       for (int aK=0 ; aK<3 ; aK++)
       {
          aSet->mVP[aK] = aNewVP[aK];
       }
    }
}


double cTest_PBGC3M2DF::AddBundle(const  std::vector<cSetCTest_PBGC3M2DF *> & aVS,double aErrStd)
{
    std::vector<double> aVEr;
    for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
    {
       cSetCTest_PBGC3M2DF * aSet = aVS[aKS];
       int aNbCam = (int)aSet->mVCams.size();
       int aNbP = (int)aSet->mVP[0].size();
       for (int aKP=0 ; aKP<aNbP ; aKP++)
       {
          std::vector<Pt3dr> aVP0;
          std::vector<Pt3dr> aVP1;
          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              // ElSeg3D aSeg = aSet->mVCams[aKC]->mFPC.GPF_CurBGCap3D()->Capteur2RayTer(aSet->mVP[aKC][aKP]);

              ElSeg3D aSeg = aSet->KCamCur(aKC)->Capteur2RayTer(aSet->mVP[aKC][aKP]);
              aVP0.push_back(aSeg.P0());
              aVP1.push_back(aSeg.P1());
          }
          bool Ok;
          Pt3dr  aPImTer = InterSeg(aVP0,aVP1,Ok);
          double anEr = 0;

          mEqP3I->InitEqP3iVal(aPImTer);

          for (int aKC=0 ; aKC<aNbCam ; aKC++)
          {
              Pt2dr aPIm = aSet->mVP[aKC][aKP];
              cParamPtProj aPPP(1.0,1.0,false,-1);
              aPPP.mTer = aPImTer;

              Pt2dr aPAp = aSet->mVCams[aKC]->mFPC.AddEqAppuisInc(aPIm,0,aPPP,false,NullPCVU);
              anEr += euclid(aPAp);
          }
          anEr /= aNbCam;
          aVEr.push_back(anEr);
          
          if ((aErrStd >0) && (anEr< (10 * aErrStd)))
          {
              double aPds = 1 /(1 + ElSquare(anEr/aErrStd));
              for (int aKC=0 ; aKC<aNbCam ; aKC++)
              {
                  Pt2dr aPIm = aSet->mVP[aKC][aKP];
                  cParamPtProj aPPP(1.0,1.0,false,-1);
                  aPPP.mTer = aPImTer;
                  aSet->mVCams[aKC]->mFPC.AddEqAppuisInc(aPIm,aPds,aPPP,false,NullPCVU);
              }
              aSet->mBufSub->DoSubstBloc(NullPCVU);
          }
       }
    }
    // Max en cas de donnees parfaite ...
    return ElMax(0.0001,KthValProp(aVEr,0.75));
}

void cTest_PBGC3M2DF::OneIterBundle()
{
   // mVCT[0]->Show();
   mSet->SetPhaseEquation();

   for (int aKC=0 ; aKC<int(mVCT.size()) ; aKC++)
   {
       cCamTest_PBGC3M2DF * aCT =  mVCT[aKC];
       cPolynBGC3M2D_Formelle & aCF = aCT->mFPC;

       double aSomD,aSomRot;
       // aCF.TestRot(Pt2di(0,0),aCF.SzCell(),aSomD,aSomRot,0);
       // Avec forcage, les resultat sont "bons"
       aCF.TestRot(Pt2di(0,0),aCF.SzCell(),aSomD,aSomRot,&(mVCT[aKC]->mMatPert));
       ElTimer aT;

       aCF.AddEqRot(Pt2di(0,0),aCF.SzCell(),aCT->mSomPdsMes* 100);
       aCF.AddEqAttachGlob(aCT->mSomPdsMes *1e-5,true,20,0);
       aCF.AddEqAttachGlob(aCT->mSomPdsMes * 1e-7,false,20,0);


       std::cout << "SOMD " << mVCT[aKC]->mIma->mNameIm << " " <<  aSomD << " " << aSomRot  << " T " << aT.uval() << " Pds " << aCT->mSomPdsMes << "\n";
   }

   double aErCple = AddBundle(mVCpleT,-1);
   std::cout << "ERCPLE " << aErCple << "\n";
   AddBundle(mVCpleT,aErCple);


   if (mVTriT.size())
   {
       double aErrTri = AddBundle(mVTriT,-1);
       AddBundle(mVTriT,aErrTri);
       std::cout << "ER TRI " << aErrTri  << "\n";
   }

// DebugBundleGen = true;
   std::cout << "\n";
   mSet->SolveResetUpdate();
}



extern bool ShowStatMatCond;

cTest_PBGC3M2DF::cTest_PBGC3M2DF(int argc,char ** argv)  :
    cAppliWithSetImage   (argc-1,argv+1,0),
    mSet                 (new cSetEqFormelles(cNameSpaceEqF::eSysPlein)),
    mDeg                 (2),
    mPerturbAng          (0.01),
    mPerturbPol          (0.0),
    mPerfectData         (false)
{
   ShowStatMatCond = false;
   //  cSubstitueBlocIncTmp::AddInc recouvrement / TMP
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mPat,"Full Name (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(mOri,"Orientation", eSAM_IsExistDirOri),
        LArgMain()
                    << EAM(mDeg,"Degre", true,"Degre of polynomial correction (Def=3)")
                    << EAM(mPerturbAng,"PertAng", true,"Angle Perturbation")
                    << EAM(mPerturbPol,"PertPol", true,"Polynomial Perturbation")
                    << EAM(mPerfectData,"PerfectData", true,"Set data with potentially perfect projection")
   );
   mNbSom = (int)mVSoms.size();

   for (int aK=0 ; aK<mNbSom ; aK++)
   {
       mVCT.push_back(new cCamTest_PBGC3M2DF(*mVSoms[aK]->attr().mIma,*this,aK));
   }

   std::cout << "DONE INIT \n"; 

   mEqP3I  =  mSet->Pt3dIncTmp();
   std::string aKey = "NKS-Assoc-CplIm2Hom@@dat";

   

   for (int aK1=0 ;  aK1 <mNbSom ; aK1++)
   {
       for (int aK2=aK1+1 ;  aK2 <mNbSom ; aK2++)
       {
            cCamTest_PBGC3M2DF * aC1 = mVCT[aK1];         
            cCamTest_PBGC3M2DF * aC2 = mVCT[aK2];         
            std::string aN12 =  mEASF.mICNM->Assoc1To2(aKey,aC1->mIma->mNameIm,aC2->mIma->mNameIm,true);
            std::string aN21 =  mEASF.mICNM->Assoc1To2(aKey,aC2->mIma->mNameIm,aC1->mIma->mNameIm,true);
            if (ELISE_fp::exist_file(aN12) && ELISE_fp::exist_file(aN21))
            {
                 cSetCTest_PBGC3M2DF * aCple = new cSetCTest_PBGC3M2DF;
                 aCple->mPack12 =  ElPackHomologue::FromFile(aN12);
                 aCple->mPack21 =  ElPackHomologue::FromFile(aN21);
                 aCple->mVCams.push_back(aC1);
                 aCple->mVCams.push_back(aC2);


                 Merge2Pack(aCple->mVP[0],aCple->mVP[1],1,aCple->mPack12,aCple->mPack21);

                 std::cout << aN12 << " " << aN21 
                          << " Sz0= " << aCple->mPack12.size() << " " << aCple->mPack21.size() 
                          << " SzM= " << aCple->mVP[0].size()  << "\n";
                 int aNbPts = (int)aCple->mVP[0].size();
                 if (aNbPts > 10)
                 {
                     cSubstitueBlocIncTmp * aBS = new cSubstitueBlocIncTmp(*mEqP3I);
                     aCple->mBufSub =  aBS;
                     aBS->AddInc(aC1->mFPC.IntervAppuisPtsInc());
                     aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                     aBS->Close();

// aBuf->AddInc(*(mVLInterv[aK]));

                     mVCpleT.push_back(aCple);
                     mMapCpleT[Pt2di(aK1,aK2)] = aCple;

                     aC1->mNbMesPts+= aNbPts;
                     aC2->mNbMesPts+= aNbPts;
                 }
            }
       }
   }


   for (int aK1=0 ;  aK1 <mNbSom ; aK1++)
   {
       for (int aK2=aK1+1 ;  aK2 <mNbSom ; aK2++)
       {
           for (int aK3=aK2+1 ;  aK3 <mNbSom ; aK3++)
           {
               if (HasArc(aK1,aK2) && HasArc(aK1,aK3) && HasArc(aK2,aK3))
               {
                   cSetCTest_PBGC3M2DF * aTri = new cSetCTest_PBGC3M2DF;
                   cCamTest_PBGC3M2DF * aC1 = mVCT[aK1];         
                   cCamTest_PBGC3M2DF * aC2 = mVCT[aK2];         
                   cCamTest_PBGC3M2DF * aC3 = mVCT[aK3];         
                   aTri->mVCams.push_back(aC1);
                   aTri->mVCams.push_back(aC2);
                   aTri->mVCams.push_back(aC3);

                   /*aTri->mCam1 = mVCT[aK1];         
                   aTri->mCam2 = mVCT[aK2];         
                   aTri->mCam3 = mVCT[aK3];          */
                   cSetCTest_PBGC3M2DF * aCp12 = mMapCpleT[Pt2di(aK1,aK2)];
                   cSetCTest_PBGC3M2DF * aCp13 = mMapCpleT[Pt2di(aK1,aK3)];
                   cSetCTest_PBGC3M2DF * aCp23 = mMapCpleT[Pt2di(aK2,aK3)];



                   Merge3Pack
                   (
                        aTri->mVP[0],aTri->mVP[1],aTri->mVP[2],
                        3,
                        aCp12->mVP[0], aCp12->mVP[1],
                        aCp13->mVP[0], aCp13->mVP[1],
                        aCp23->mVP[0], aCp23->mVP[1]
                   );
                   int aNbPts = (int)aTri->mVP[0].size();
                   if (aNbPts > 5)
                   {
                       cSubstitueBlocIncTmp * aBS = new cSubstitueBlocIncTmp(*mEqP3I);
                       aTri->mBufSub =  aBS;
                       aBS->AddInc(aC1->mFPC.IntervAppuisPtsInc());
                       aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                       aBS->AddInc(aC3->mFPC.IntervAppuisPtsInc());

                       // aBS->AddInc(aC2->mFPC.IntervAppuisPtsInc());
                       // aBS->AddInc(aC3->mFPC.IntervAppuisPtsInc());
                       aBS->Close();
  
                       aC1->mNbMesPts += aNbPts;
                       aC2->mNbMesPts += aNbPts;
                       aC3->mNbMesPts += aNbPts;


                       mVTriT.push_back(aTri);
                       std::cout  << aC1->mIma->mNameIm  << " "
                                  << aC2->mIma->mNameIm  << " "
                                  << aC3->mIma->mNameIm  << " "
                                  <<  aTri->mVP[0].size()  << " "
                                  <<  aTri->mVP[1].size()  << " "
                                  <<  aTri->mVP[2].size()  << "\n";
                   }
               }
           }
       }
   }

   for (int aK=0 ; aK<mNbSom ; aK++)
   {
       mVCT[aK]->mSomPdsMes = mVCT[aK]->mNbMesPts;
   }


   if (mPerfectData)
   {
      SetPerfectData(mVCpleT);
      SetPerfectData(mVTriT);
   }
/*
*/

   mSet->SetClosed();


   for (int aK=0 ; aK<100 ; aK++)
   {
      OneIterBundle();
   }
}

int CPP_TestBundleGen(int argc,char ** argv)  
{
    cTest_PBGC3M2DF anAppli(argc,argv);

    return EXIT_SUCCESS;
}


/***********************************************************************/
/*                                                                     */
/*           Convert 2 Gen Bundle                                      */
/*                                                                     */
/***********************************************************************/

class cApppliConvertBundleGen
{
    public :
        cApppliConvertBundleGen(int argc,char ** argv);

        void Export();
   private :
        double RandAngle() {return mPertubAng * NRrandC();}


        std::string mPatIm,mPatOrient,mDest;
        std::string mPostFix;
        cElemAppliSetFile mEASF;
        int mDegPol;
        std::string mNameOutInit;
        cBasicGeomCap3D * mCamGen;
        const	cSystemeCoord * mChSys;
        std::string       mNameType;
        eTypeImporGenBundle mType;
        double              mPertubAng;
};

cApppliConvertBundleGen::cApppliConvertBundleGen (int argc,char ** argv)   :
     mDegPol     (2),
     mCamGen     (0),
     mNameType   ("TIGB_Unknown"),
     mPertubAng  (0)
{

   std::string aNameType = "TIGB_Unknown";
   std::string aChSysStr = "";

   //  cSubstitueBlocIncTmp::AddInc recouvrement / TMP
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mPatIm,"Name of Image")
                    << EAMC(mPatOrient,"Name of input Orientation File")
                    << EAMC(mDest,"Directory of output Orientation (MyDir -> Oi-MyDir)"),
        LArgMain()
	            << EAM(aChSysStr,"ChSys", true, "Change coordinate file (MicMac XML convention)")
                    << EAM(mDegPol,"Degre", true,"Degre of polynomial correction (Def=2)")
                    << EAM(mNameType,"Type",true,"Type of sensor (see eTypeImporGenBundle)",eSAM_None,ListOfVal(eTT_NbVals,"eTT_"))
                    << EAM(mPertubAng,"PertubAng",true,"Type of sensor (see eTypeImporGenBundle)")
 
    );

    
    bool mModeHelp;

    StdReadEnum(mModeHelp,mType,mNameType,eTIGB_NbVals);
    mEASF.Init(mPatIm);
    if (mEASF.SetIm()->size()==0)
    {
        std::cout << "Cannot find " << mPatIm << "\n";
        ELISE_ASSERT(false,"Not any image");
    }

    if(aChSysStr=="") 
    {
      mChSys=0;
    }
    else
    {
        mChSys = new cSystemeCoord(StdGetObjFromFile<cSystemeCoord>
                     (
                         aChSysStr,
                         StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                         "SystemeCoord",
                         "SystemeCoord"
                     ));
    }


    cElRegex  anAutom(mPatIm.c_str(),10);

    for (size_t aKIm=0  ; aKIm< mEASF.SetIm()->size() ; aKIm++)
    {
        std::string aNameIm = (*mEASF.SetIm())[aKIm];
// std::string aNameOrient  =  mPatOrient;
std::string aNameOrient  =  MatchAndReplace(anAutom,aNameIm,mPatOrient);
        AutoDetermineTypeTIGB(mType,aNameOrient);
        mPostFix = IsPostfixed(aNameOrient) ?  StdPostfix(aNameOrient) : "";


       int aIntType = mType;
       mCamGen = cBasicGeomCap3D::StdGetFromFile(aNameOrient,aIntType,mChSys);
       mType = (eTypeImporGenBundle)  aIntType;
       CamStenope * aCS = mCamGen->DownCastCS();
       if (aCS)
       {
           if (mPertubAng != 0)
           {
               aCS = aCS->Dupl();
               ElRotation3D  aR =aCS->Orient().inv();
               ElMatrix<double> aMPert  = ElMatrix<double>::Rotation(RandAngle(),RandAngle(),RandAngle());
               aR = ElRotation3D(aR.tr(),aR.Mat()*aMPert,true);
               aCS->SetOrientation(aR.inv());
           }

            mNameOutInit = mEASF.mICNM->Assoc1To1("NKS-Assoc-Im2UnCorMMOrient@-"+mDest,NameWithoutDir(aNameIm),true);
         
            MakeFileXML(aCS->StdExportCalibGlob(),mNameOutInit);
         
       }
       else
       {
            mNameOutInit =  mEASF.mICNM->Assoc1To1("NKS-Assoc-Im2UnCorExternOrient@-"+mDest, eToString(mType) + "-" + NameWithoutDir(aNameOrient),true);
            ELISE_fp::CpFile(aNameOrient,mNameOutInit);
        
       }


       cPolynomial_BGC3M2D aPol(mChSys,mCamGen,mNameOutInit,aNameIm,mDegPol,0); // TAGG
       aPol.Save2XmlStdMMName(mEASF.mICNM,mDest,aNameIm,ElAffin2D::Id());
    }
    
}

int CPP_ConvertBundleGen(int argc,char ** argv)  
{
    cApppliConvertBundleGen anAppli(argc,argv);
    return EXIT_SUCCESS;
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
