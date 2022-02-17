#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"



namespace MMVII
{



class cAppliExtractLearnVecDM : public cAppliLearningMatch
{
     public :
        typedef cIm2D<tU_INT1>             tImMasq;
        typedef cIm2D<tREAL4>              tImPx;
        typedef cDataIm2D<tU_INT1>         tDataImMasq;
        typedef cDataIm2D<tREAL4>          tDataImPx;
        typedef cIm2D<tREAL4>              tImFiltred;
        typedef cDataIm2D<tREAL4>          tDataImF;
        typedef cGaussianPyramid<tREAL4>   tPyr;
        typedef std::shared_ptr<tPyr>      tSP_Pyr;

        cAppliExtractLearnVecDM(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

        // 0 Hom , 1 close , 2 Non Hom
        void AddLearn(cFileVecCaracMatch &,const cAimePCar & aAP1,const cAimePCar & aAP2,int aLevHom);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	std::vector<std::string>  Samples() const  override;

    
        const cBox2di &     CurBoxIn(bool IsIm1) const {return   IsIm1 ? mCurBoxIn1 : mCurBoxIn2 ;}
        const std::string & NameIm(bool IsIm1) const {return   IsIm1 ? mNameIm1 : mNameIm2 ;}


        void MakeOneBox(const cPt2di & anIndex,const cParseBoxInOut<2> &);
        std::string NameHom(int aNumHom) {return HomFromIm1(mNameIm1,aNumHom,Index(mNumIndex)+mExtSave);}

        void MakeCutHisto(int aY,const std::vector<bool> & Ok1,const std::vector<cAimePCar> & aVPC1,
                            const std::vector<bool> & Ok2,const std::vector<cAimePCar> & aVPC2);
              // MakeCut(aPixIm1.y(),aVOk1,aV1,aVOk2,aV2);   std::vector<cAimePCar> std::vector<bool> 
	cIm2D<tU_INT1> OneCutRadiom(cPyr1ImLearnMatch *,int aY,bool Im1);
        void MakeCutRadiom(int aY);

           // --- Mandatory ----
        std::string  mPatIm1;

           // --- Optionnal ----
        int          mSzTile;
        int          mOverlap;
        std::string  mExtSave;
        std::string  mPatShowCarac;
        int          mNb2Select;  ///< In case we want only a max number of points
        int          mFlagRand;

        std::string       mCutsNameH;
        std::vector<int>  mCutsParam;
        int               mCutsFreqPx;  // 1/ Steps of Px 
        double            mCutsExp;  // 1/ Steps of Px 
        std::string       mCutsPrefix;

           // --- Internal variables ----

        bool            mDoCuts;
        bool            mCutsHisto; // Do Cuts 4 Histo
        bool            mCutsRadiom; // Do Cut 4 Radiom
        bool            m4Learn;  // when 4Learn, load Px, Masq ...
        cHistoCarNDim   mCutHND;
        int             mCutPxMin;
        int             mCutPxMax;
        std::string     mNameIm1;
        std::string     mNameIm2;
        std::string     mNamePx1;
        std::string     mNameMasq1;



        cBox2di      mCurBoxIn1;
        cPt2di       mSzIm1;
        cBox2di      mCurBoxIn2;
        cBox2di      mCurBoxOut;
        int          mDeltaPx ; // To Add to Px1 to take into account difference of boxe
        tImMasq      mImMasq1;
        tImPx        mImPx1;

	cPyr1ImLearnMatch * mPyrL1;
	cPyr1ImLearnMatch * mPyrL2;

        bool         mSaveImFilter;
        bool         mShowCarac;
        tNameSelector mSelShowCarac;
        int           mNumIndex;

        cFilterPCar  mFPC;  ///< Used to compute Pts
};

cAppliExtractLearnVecDM::cAppliExtractLearnVecDM(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mSzTile       (3000),
   mOverlap      (200),
   mExtSave      ("Std"),
   mFlagRand     (0),
   mCutsFreqPx   (1),
   mCutsExp      (1.0),
   mCutsPrefix   ("Cut"),
   mDoCuts       (false),
   m4Learn       (true),
   // mCutHND        (nullptr),
   mCurBoxIn1    (cBox2di::Empty()),  // To have a default value
   mCurBoxIn2    (cBox2di::Empty()),  // To have a default value
   mCurBoxOut    (cBox2di::Empty()),  // To have a default value
   mImMasq1      (cPt2di(1,1)),       // To have a default value
   mImPx1        (cPt2di(1,1)),       // To have a default value
   mPyrL1         (nullptr),
   mPyrL2         (nullptr),
   mSaveImFilter (false),
   mShowCarac    (false),
   mSelShowCarac (AllocRegex(".*")),
   mFPC          (false)
{
    mFPC.FinishAC();
    mFPC.Check();
}


cCollecSpecArg2007 & cAppliExtractLearnVecDM::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mPatIm1,"Name of input(s) file(s), Im1",{{eTA2007::MPatFile,"0"}})
   ;
}

cCollecSpecArg2007 & cAppliExtractLearnVecDM::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mSzTile, "TileSz","Size of tile for spliting computation",{eTA2007::HDV})
          << AOpt2007(mOverlap,"TileOL","Overlap of tile to limit sides effects",{eTA2007::HDV})
          << AOpt2007(mPatShowCarac,"PSC","Pattern for Showing Caracteristics")
          << AOpt2007(mNb2Select,"Nb2S","Number of point to select, def=all in masq")
          << AOpt2007(mExtSave,"ExtOut","Ext for save file",{eTA2007::HDV})
          << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV,eTA2007::Tuning})
          << AOpt2007(mFlagRand,"FlagRand","Images to randomizes (1 or 2), bit of flag [0-3]",{eTA2007::HDV,eTA2007::Tuning})
          << AOpt2007(mCutsNameH,"CutNamH","Name of Histo to create similarity cuts (NONE if radiom)")
          << AOpt2007(mCutsParam,"CutParam","Interval Pax + Line of cuts[PxMin,PxMax,Y0,Y1,....]",{{eTA2007::ISizeV,"[3,10000]"}})
          << AOpt2007(mCutsFreqPx,"CutStep"," Step in px for the cuts",{eTA2007::HDV})
          << AOpt2007(mCutsPrefix,"CutPref"," Name to add to results",{eTA2007::HDV})
          << AOpt2007(mCutsExp,"CutExp","Exposant to set dynamic of view ",{eTA2007::HDV})
          << AOpt2007(mNameIm2,"Im2","Name Im2, Def .*Im1.tif => .*Im2.tif ")
          // << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV})
   ;
}

bool TESTPT(const cPt2dr & aPt,int aLine,const std::string& aFile)
{
    cPt2dr aPTEST(778.357,261);
    if (Norm2(aPTEST-aPt)>1e-2) return false;
    StdOut() << "TEST PT Line=" << aLine << " " << aFile << "\n";
    return true;
}


void cAppliExtractLearnVecDM::AddLearn(cFileVecCaracMatch & aFVCM,const cAimePCar & aAP1,const cAimePCar & aAP2,int aLevHom)
{

   cVecCaracMatch aVCM (*mPyrL1,*mPyrL2,aAP1,aAP2);
   aFVCM.AddCarac(aVCM);

   if (mShowCarac)
   {
       //SaveInFile(aVCM,"XXXXTest.xml");
       //SaveInFile(aVCM,"XXXXTest.dmp");
       aVCM.Show(mSelShowCarac);
       StdOut() << "LLLLLLlevvvv " << aLevHom << "PPPp==="  << aAP1.Pt() << " " << aAP2.Pt() << "\n";
   }
}

std::vector<std::string>  cAppliExtractLearnVecDM::Samples() const
{
    return std::vector<std::string>
           (
               {
                   "MMVII DM2CalcHistoCarac DMTrain_MDLB2014-Vintage-perfect_Box0Std_LDHAime0.dmp Test",
                   "MMVII DM2CalcHistoCarac DMTrain_MDLB2014.*LDHAime0.dmp AllMDLB2014"
               }
          );

}



int  cAppliExtractLearnVecDM::Exe()
{

   // If a multiple pattern, run in // by recall
   if (RunMultiSet(0,0))
      return ResultMultiSet();

   mDoCuts = IsInit(&mCutsNameH);
   if (mDoCuts)
   {
     m4Learn = false;
     if (mCutsNameH==MMVII_NONE)
     {
         mCutsRadiom = true;
         SetIfNotInit(mCutsHisto,false);
     }


     if ((mCutsNameH!=MMVII_NONE) || (mCutsHisto))
     {
	 mCutsHisto = true;
         SetIfNotInit(mCutsRadiom,false);
         ReadFromFile(mCutHND,mCutsNameH);
         MMVII_INTERNAL_ASSERT_strong(IsInit(&mCutsParam),"Px interv and line cut non initialezd in cut mode");
         mCutPxMin = mCutsParam.at(0);
         mCutPxMax = mCutsParam.at(1);
     }
   }
   else
   {
      mCutsHisto = false;
      mCutsRadiom = false;
   }


   mShowCarac = IsInit(&mPatShowCarac);
   if (mShowCarac)
      mSelShowCarac = AllocRegex(mPatShowCarac);

   // If we are here, single image, because user specified 1, or by recall of the pattern

      // ---- Compute name of images from Im1 -----
   mNameIm1 = mPatIm1;                   // To  homogenize naming 
   if (!IsInit(&mNameIm2))
       mNameIm2 = Im2FromIm1(mNameIm1);
   if (m4Learn)
   {
       mNamePx1  = Px1FromIm1(mNameIm1);
       mNameMasq1  = Masq1FromIm1(mNameIm1);
   }


   cDataFileIm2D aDFIm1 = cDataFileIm2D::Create(mNameIm1,false);

   StdOut() << "IM2=" << mNameIm2 << " " << mNamePx1 << " " << mNameMasq1 << "\n";
   StdOut() << "Sz=" <<  aDFIm1.Sz() << "\n";

   cParseBoxInOut<2> aPBI = cParseBoxInOut<2>::CreateFromSizeCste(aDFIm1,mSzTile);

   mNumIndex=0;
   for (const auto & anIndex : aPBI.BoxIndex())
   {
       // Store Boxes as members
       mCurBoxIn1  = aPBI.BoxIn(anIndex,mOverlap);
       mSzIm1 = mCurBoxIn1.Sz();
       mCurBoxOut = aPBI.BoxOut(anIndex);
       MakeOneBox(anIndex,aPBI);
       mNumIndex++;
   }

   return EXIT_SUCCESS;
}

// cIm2D<tU_INT1>
cIm2D<tU_INT1> cAppliExtractLearnVecDM::OneCutRadiom(cPyr1ImLearnMatch * aPyr,int aY,bool Im1)
{
    const tDataImF & aIm = aPyr-> ImInit() ;
    cIm2D<tU_INT1> aImCut(cPt2di(aIm.Sz().x(),256));

    for (int aX=0 ; aX<aIm.Sz().x() ; aX++)
    {
         int aRad = aIm.GetV(cPt2di(aX,aY));
	 for (int aY=0 ; aY<256 ; aY++)
	 {
             aImCut.DIm().SetV(cPt2di(aX,255-aY),255*(aY<aRad));
	 }
    }
    std::string aNameRes =    DirVisu() +  mCutsPrefix + "_RadL" + ToStr(aY) 
                          + "_" + MMVII::Prefix(NameIm(Im1)) + ".tif";
    aImCut.DIm().ToFile(aNameRes);
    return aImCut;
}
void cAppliExtractLearnVecDM::MakeCutRadiom(int aY)
{
    OneCutRadiom(mPyrL1,aY,true);
    OneCutRadiom(mPyrL2,aY,false);
}

void cAppliExtractLearnVecDM::MakeCutHisto
     (
          int aY,
          const std::vector<bool> & aVOk1,const std::vector<cAimePCar> & aVPC1,
          const std::vector<bool> & aVOk2,const std::vector<cAimePCar> & aVPC2
     )
{
   StdOut() << "LINECUT " << aY << "\n";
   int aNbPix = aVOk1.size();
   int aNbPax = mCutsFreqPx * (mCutPxMax-mCutPxMin);
   cIm2D<tU_INT1>  aResult(cPt2di(aNbPix,aNbPax),nullptr,eModeInitImage::eMIA_Null);

   for (int aX=0; aX<aNbPix ; aX++)
   {
       if (aVOk1.at(aX))
       {
           int aXMinTh = (aX+mCutPxMin)*mCutsFreqPx;
           int aXHom0 = std::max(0,aXMinTh);
           int aXHom1 = std::min(int(aVPC2.size()),(aX+mCutPxMax)*mCutsFreqPx);
           for(int aXH=aXHom0 ; aXH<aXHom1 ; aXH++)
           {
               if (aVOk2.at(aXH))
               {
                   cVecCaracMatch aVCM ( *mPyrL1,*mPyrL2, aVPC1.at(aX),aVPC2.at(aXH));
                   double aScore = mCutHND.HomologyLikelihood(aVCM,false);
                   int aPax = aXH-aXMinTh;
                   aResult.DIm().SetV(cPt2di(aX,aPax),1+round_ni((254.0)*(1-pow(1-aScore,mCutsExp))));
               }
           }
        }
   }
   std::string aNameRes =    DirVisu() +  mCutsPrefix + "_L" + ToStr(aY) 
                          + "_" + MMVII::Prefix(mNameIm1) + "_" +  mCutHND.Name() + ".tif";

   aResult.DIm().ToFile(aNameRes); //  Ok
}

void cAppliExtractLearnVecDM::MakeOneBox(const cPt2di & anIndex,const cParseBoxInOut<2> & aPBI)
{

   // Read Px & Masq
   int aNbInMasq = 0;
   tDataImMasq * aDIMasq1 = nullptr;
   const tDataImPx   * aDImPx1  = nullptr;
   if (m4Learn)
   {
      mImMasq1 = tImMasq::FromFile(mNameMasq1,mCurBoxIn1);
      mImPx1   =   tImPx::FromFile(  mNamePx1,mCurBoxIn1);
      aDIMasq1 = & mImMasq1.DIm();
      aDImPx1  = & mImPx1.DIm();

      for (const auto & aPix : *aDIMasq1)
          if (aDIMasq1->GetV(aPix)) 
             aNbInMasq++;


      if (IsInit(&mNb2Select))
      {
          int aNbLoc2Sel  =  mNb2Select / aPBI.BoxIndex().NbElem();
          int aKMasq = 0;
          int aNbInMasqInit = aNbInMasq;
          aNbInMasq = 0;
          for (const auto & aPix : *aDIMasq1)
          {
             if (aDIMasq1->GetV(aPix)) 
             {
                if (SelectQAmongN(aKMasq,aNbLoc2Sel,aNbInMasqInit))
                {
                    aNbInMasq ++;
                   //static int aCpt=0; aCpt++;
                   //StdOut() << "CCCCcc   " << aCpt << "\n";
                }
                else
                {
                    aDIMasq1->SetV(aPix,0) ;
                }
                aKMasq++;
             }
          }
      }


      {
         //  Compute  Px intervall min and max to compute   Box2
         tREAL4 aPxMin = 1e10;
         tREAL4 aPxMax = -1e10;
         for (const auto & aPix : *aDIMasq1)
         {
             if (aDIMasq1->GetV(aPix))
             {
                 UpdateMinMax(aPxMin,aPxMax,aDImPx1->GetV(aPix));
             }
         }
         cPt2di aP0 = mCurBoxIn1.P0();
         cPt2di aP1 = mCurBoxIn1.P1();
         // Compute box of image2 taking into account Px 
         mCurBoxIn2 = cBox2di
                      (
                          cPt2di(aP0.x()+round_down(aPxMin),aP0.y()),
                          cPt2di(aP1.x()+round_up(aPxMax)  ,aP1.y())
                      );

         cDataFileIm2D  aFile2 = cDataFileIm2D::Create(mNameIm2,true);
         mCurBoxIn2 = mCurBoxIn2.Inter(cBox2di(cPt2di(0,0),aFile2.Sz()));

         StdOut() << "INDEX " << anIndex << " PX=[" << aPxMin << " : "<< aPxMax << "] B=" <<  mCurBoxIn2 << "\n";
      }
   }
   else
   {
      mCurBoxIn2 = mCurBoxIn1;
      aNbInMasq = mCurBoxIn1.Sz().x() * (mCutPxMax - mCutPxMin);
   }
   // Now we have the boxes we can create the pyramid
   mPyrL1 = new cPyr1ImLearnMatch(mCurBoxIn1,mCurBoxOut,mNameIm1,*this,mFPC,(mFlagRand&1)!=0);
   mPyrL2 = new cPyr1ImLearnMatch(mCurBoxIn2,mCurBoxOut,mNameIm2,*this,mFPC,(mFlagRand&2)!=0);
   if (mSaveImFilter)
   {
      mPyrL1->cPyr1ImLearnMatch::SaveImFiltered();
      mPyrL2->cPyr1ImLearnMatch::SaveImFiltered();
   }
 
   // ex : X0=10; X1=0; homologous of P0=0  will be 10
   mDeltaPx =  mCurBoxIn1.P0().x() -  mCurBoxIn2.P0().x();


   cPt2di aPixIm1;
   double aSD0=0;
   double aSD1=0;
   double aSDK=0;
   int    aNbSD=0;
   cFileVecCaracMatch aFVC_Hom(mFPC,aNbInMasq);
   cFileVecCaracMatch aFVC_CloseHom(mFPC,aNbInMasq);
   cFileVecCaracMatch aFVC_NonHom(mFPC,aNbInMasq);
   for (aPixIm1.y()=0 ; aPixIm1.y()<mSzIm1.y()  ; aPixIm1.y()++)
   {
       std::vector<cAimePCar> aV1;
       std::vector<cAimePCar> aV2;
       if (mDoCuts)
       {
           bool DoThisCut = (std::find(mCutsParam.begin()+2, mCutsParam.end(), aPixIm1.y()) != mCutsParam.end());
           if (DoThisCut)
           {
              if (mCutsHisto)
              {
                 std::vector<bool> aVOk1;
                 std::vector<bool> aVOk2;
                 for (aPixIm1.x()=0 ; aPixIm1.x()<mSzIm1.x()  ; aPixIm1.x()++)
                 {
                     aVOk1.push_back(mPyrL1->CalculAimeDesc(ToR(aPixIm1))); 
                     aV1.push_back(mPyrL1->DupLPIm());

                     for (int aK=0 ; aK<mCutsFreqPx ; aK++)
                     {
                         cPt2dr aP2(aPixIm1.x()+ double(aK/mCutsFreqPx),aPixIm1.y());
                         aVOk2.push_back(mPyrL2->CalculAimeDesc(aP2));
                         aV2.push_back(mPyrL2->DupLPIm());
                     }
                 }
                 MakeCutHisto(aPixIm1.y(),aVOk1,aV1,aVOk2,aV2);
              }
              if (mCutsRadiom)
	      {
                  MakeCutRadiom(aPixIm1.y());
              }
           }
       }
       else
       {
           for (aPixIm1.x()=0 ; aPixIm1.x()<mSzIm1.x()  ; aPixIm1.x()++)
           {
              if (aDIMasq1->GetV(aPixIm1))
              {

                  double aPx = aDImPx1->GetV(aPixIm1)+mDeltaPx;
                  cPt2dr aP2(aPixIm1.x()+aPx,aPixIm1.y());
                  if (mPyrL1->CalculAimeDesc(ToR(aPixIm1)) && mPyrL2->CalculAimeDesc(aP2))
                  {
                      aV1.push_back(mPyrL1->DupLPIm());
                      aV2.push_back(mPyrL2->DupLPIm());
                  }
              }
           }

           int aNbDesc = aV1.size();
           for (int aK=0 ; aK<aNbDesc ; aK++)
           {
                const cAimePCar & aAP1 = aV1[aK];
                const cAimePCar & aHom = aV2[aK];
                const cAimePCar & aCloseHom = aV2.at(std::min(aK+1,aNbDesc-1));
                const cAimePCar & aNonHom = aV2.at((aK+aNbDesc/2)%aNbDesc);

                aSD0 += aAP1.L1Dist(aHom);
                aSD1 += aAP1.L1Dist(aCloseHom);
                aSDK += aAP1.L1Dist(aNonHom);

                AddLearn(aFVC_Hom      , aAP1,aHom     ,0);
                AddLearn(aFVC_CloseHom , aAP1,aCloseHom,1);
                AddLearn(aFVC_NonHom   , aAP1,aNonHom,2);
            // AddLearn(aAP1,aCloseHom,1);
            // AddLearn(aAP1,aNonHom,2);
                if (mShowCarac)
                   getchar();
                aNbSD++;
           }
           if (0&&aNbSD && (aPixIm1.y()%20==0))
           {
               StdOut() << "Y=== " << mSzIm1.y() - aPixIm1.y()  
                        << " " << aSD0/aNbSD 
                        << " " << aSD1/aNbSD 
                        << " " << aSDK/aNbSD<< "\n";
           }
      }
   }
   if (! mDoCuts)
   {
       SaveInFile(aFVC_Hom      ,  NameHom(0));
       SaveInFile(aFVC_CloseHom ,  NameHom(1));
       SaveInFile(aFVC_NonHom   ,  NameHom(2));
    }

   delete mPyrL1;
   delete mPyrL2;
}




/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_ExtractLearnVecDM(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliExtractLearnVecDM(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractLearnVecDM
(
     "DM1ExtractVecLearn",
      Alloc_ExtractLearnVecDM,
      "Extract ground truch vector to learn Dense Matching",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::FileSys},
      __FILE__
);



};
