#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{

class cBiPyramMatch
{
    public :
        typedef cIm2D<tU_INT1>             tImMasq;
        typedef cGaussianPyramid<tREAL4>   tPyr;
        typedef std::shared_ptr<tREAL4>    tSP_Pyr;

    private :
        std::vector<tSP_Pyr>   mVPyr;
};

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
        tImFiltred & ImF(bool IsIm1 ) {return IsIm1 ? mImF1 : mImF2;}
        tSP_Pyr CreatePyr(bool IsIm1);
        const cBox2di &     CurBoxIn(bool IsIm1) const {return   IsIm1 ? mCurBoxIn1 : mCurBoxIn2 ;}
        const std::string & NameIm(bool IsIm1) const {return   IsIm1 ? mNameIm1 : mNameIm2 ;}

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        void MakeOneBox(const cPt2di & anIndex,const cParseBoxInOut<2> &);
        std::string NameHom(int aNumHom) {return HomFromIm1(mNameIm1,aNumHom,Index(mNumIndex)+mExtSave);}

           // --- Mandatory ----
        std::string  mPatIm1;

           // --- Optionnal ----
        int          mSzTile;
        int          mOverlap;
        std::string  mExtSave;
        std::string  mPatShowCarac;
        int          mNb2Select;  ///< In case we want only a max number of points
        int          mFlagRand;

           // --- Internal variables ----
        std::string  mNameIm1;
        std::string  mNameIm2;
        std::string  mNamePx1;
        std::string  mNameMasq1;



        cBox2di      mCurBoxIn1;
        cPt2di       mSzIm1;
        cBox2di      mCurBoxIn2;
        cBox2di      mCurBoxOut;
        int          mDeltaPx ; // To Add to Px1 to take into account difference of boxe
        tImMasq      mImMasq1;
        tImPx        mImPx1;
        int          mNbOct;
        int          mNbLevByOct;
        int          mNbOverLapByO;
        tSP_Pyr      mPyr1;
        tSP_Pyr      mPyr2;
        tImFiltred   mImF1;
        tImFiltred   mImF2;
        bool         mSaveImFilter;
        bool         mShowCarac;
        tNameSelector mSelShowCarac;
        int           mNumIndex;


        cFilterPCar  mFPC;  ///< Used to compute Pts

        bool  CalculAimeDesc(bool Im1,const cPt2dr & aPt);

        cAimePCar    mPC1;
        cAimePCar    mPC2;
};

cAppliExtractLearnVecDM::cAppliExtractLearnVecDM(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mSzTile       (3000),
   mOverlap      (200),
   mExtSave      ("Std"),
   mFlagRand     (0),
   mCurBoxIn1    (cBox2di::Empty()),  // To have a default value
   mCurBoxIn2    (cBox2di::Empty()),  // To have a default value
   mCurBoxOut    (cBox2di::Empty()),  // To have a default value
   mImMasq1      (cPt2di(1,1)),       // To have a default value
   mImPx1        (cPt2di(1,1)),       // To have a default value
   mNbOct        (5),                 // 3 octave for window , maybe add 2 learning multiscale 
   mNbLevByOct   (2),                 // 
   mNbOverLapByO (1),                 // 1 overlap is required for junction at decimation
   mPyr1         (nullptr),
   mPyr2         (nullptr),
   mImF1         (cPt2di(1,1)),
   mImF2         (cPt2di(1,1)),
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
          // << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV})
   ;
}


bool  cAppliExtractLearnVecDM::CalculAimeDesc(bool Im1,const cPt2dr & aPt)
{
 
    cAimePCar & aAPC = Im1 ?  mPC1 : mPC2;
    tPyr * aPyr = Im1 ? mPyr1.get() : mPyr2.get();
    cProtoAimeTieP<tREAL4> aPAT(aPyr->GPImTop(),aPt);

    if (! aPAT.FillAPC(mFPC,aAPC,true))
       return false;

    aPAT.FillAPC(mFPC,aAPC,false);

    return true;
}

void cAppliExtractLearnVecDM::AddLearn(cFileVecCaracMatch & aFVCM,const cAimePCar & aAP1,const cAimePCar & aAP2,int aLevHom)
{
   tREAL4 aV1 = ImF(true).DIm().GetVBL(aAP1.Pt());
   tREAL4 aV2 = ImF(false).DIm().GetVBL(aAP2.Pt());

/*
{
// 73410
   static int aCpt=0; aCpt++;
   bool BUG = (aCpt==73410);
   if (BUG)
   {
       std::cout << "Cccc= " << aCpt << " Val=" << aV1 << " " << aV2  << " Pts=" << aAP1.Pt() << " " << aAP2.Pt() << "\n";
   }
}
*/

   cVecCaracMatch aVCM(mPyr1->MulScale(),aV1,aV2,aAP1,aAP2);
   aFVCM.AddCarac(aVCM);

   if (mShowCarac)
   {
       //SaveInFile(aVCM,"XXXXTest.xml");
       //SaveInFile(aVCM,"XXXXTest.dmp");
       aVCM.Show(mSelShowCarac);
       StdOut() << "LLLLLLlevvvv " << aLevHom << "PPPp==="  << aAP1.Pt() << " " << aAP2.Pt() << "\n";
   }
}


int  cAppliExtractLearnVecDM::Exe()
{
   // If a multiple pattern, run in // by recall
   if (RunMultiSet(0,0))
      return ResultMultiSet();

   mShowCarac = IsInit(&mPatShowCarac);
   if (mShowCarac)
      mSelShowCarac = AllocRegex(mPatShowCarac);

   // If we are here, single image, because user specified 1, or by recall of the pattern

      // ---- Compute name of images from Im1 -----
   mNameIm1 = mPatIm1;                   // To  homogenize naming 
   mNameIm2 = Im2FromIm1(mNameIm1);
   mNamePx1  = Px1FromIm1(mNameIm1);
   mNameMasq1  = Masq1FromIm1(mNameIm1);


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


cAppliExtractLearnVecDM::tSP_Pyr cAppliExtractLearnVecDM::CreatePyr(bool IsIm1)
{
    const cBox2di & aBox = CurBoxIn(IsIm1);
    const std::string & aName = NameIm(IsIm1);
    cGP_Params aGP(aBox.Sz(),mNbOct,mNbLevByOct,mNbOverLapByO,this,false);
    aGP.mFPC = mFPC;

    tSP_Pyr aPyr =  tPyr::Alloc(aGP,aName,CurBoxIn(IsIm1),mCurBoxOut);

    int aNumIm = (IsIm1 ? 1 : 2);
    if (mFlagRand & aNumIm)
    {
        aPyr->ImTop().DIm().InitRandom(0.0,100.0);
    }
    else 
    {
        aPyr->ImTop().Read(cDataFileIm2D::Create(aName,true),aBox.P0());
    }
    aPyr->ComputGaussianFilter();

    // Compute the filtered images used for having "invariant" gray level
    tImFiltred & aImF = ImF(IsIm1);
    aImF = aPyr->ImTop().Dup(); 
    float aFact = 50.0;
    ExpFilterOfStdDev(aImF.DIm(),5,aFact);

    tDataImF &aDIF = aImF.DIm();
    tDataImF &aDI0 =  aPyr->ImTop().DIm();
    for (const auto & aP : aDIF)
    {
        aDIF.SetV(aP,(1+NormalisedRatioPos(aDI0.GetV(aP),aDIF.GetV(aP))) / 2.0);
    }


    if (mSaveImFilter)
    {
        std::string  aName = "FILTRED-" + (IsIm1 ? mNameIm1  : mNameIm2);
        cIm2D<tU_INT1> aImS(aDIF.Sz());
        for (const auto & aP : aDIF)
        {
            int aVal = round_ni(aDIF.GetV(aP)*255.0);
            aImS.DIm().SetV(aP,aVal);
        }
        aImS.DIm().ToFile(aName); //  Ok
    }

    ///aIm

    return  aPyr;
}


void cAppliExtractLearnVecDM::MakeOneBox(const cPt2di & anIndex,const cParseBoxInOut<2> & aPBI)
{

   // Read Px & Masq
   mImMasq1 = tImMasq::FromFile(mNameMasq1,mCurBoxIn1);
   mImPx1   =   tImPx::FromFile(  mNamePx1,mCurBoxIn1);
   tDataImMasq & aDIMasq1 = mImMasq1.DIm();
   const tDataImPx   & aDImPx1  = mImPx1.DIm();

   int aNbInMasq = 0;
   for (const auto & aPix : aDIMasq1)
       if (aDIMasq1.GetV(aPix)) 
          aNbInMasq++;


   if (IsInit(&mNb2Select))
   {
       int aNbLoc2Sel  =  mNb2Select / aPBI.BoxIndex().NbElem();
       int aKMasq = 0;
       int aNbInMasqInit = aNbInMasq;
       aNbInMasq = 0;
       for (const auto & aPix : aDIMasq1)
       {
          if (aDIMasq1.GetV(aPix)) 
          {
             if (SelectQAmongN(aKMasq,aNbLoc2Sel,aNbInMasqInit))
             {
                 aNbInMasq ++;
                //static int aCpt=0; aCpt++;
                //StdOut() << "CCCCcc   " << aCpt << "\n";
             }
             else
             {
                 aDIMasq1.SetV(aPix,0) ;
             }
             aKMasq++;
          }
       }
   }


   {
      //  Compute  Px intervall min and max to compute   Box2
      tREAL4 aPxMin = 1e10;
      tREAL4 aPxMax = -1e10;
      for (const auto & aPix : aDIMasq1)
      {
          if (aDIMasq1.GetV(aPix))
          {
              UpdateMinMax(aPxMin,aPxMax,aDImPx1.GetV(aPix));
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
   // Now we have the boxes we can create the pyramid
   mPyr1 = CreatePyr(true);
   mPyr2 = CreatePyr(false);
 
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
       for (aPixIm1.x()=0 ; aPixIm1.x()<mSzIm1.x()  ; aPixIm1.x()++)
       {
          if (aDIMasq1.GetV(aPixIm1))
          {
              double aPx = aDImPx1.GetV(aPixIm1)+mDeltaPx;
              cPt2dr aP2(aPixIm1.x()+aPx,aPixIm1.y());
              if (CalculAimeDesc(true,ToR(aPixIm1)) && CalculAimeDesc(false,aP2))
              {
                  aV1.push_back(mPC1.DupLPIm());
                  aV2.push_back(mPC2.DupLPIm());
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
   SaveInFile(aFVC_Hom      ,  NameHom(0));
   SaveInFile(aFVC_CloseHom ,  NameHom(1));
   SaveInFile(aFVC_NonHom   ,  NameHom(2));
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
