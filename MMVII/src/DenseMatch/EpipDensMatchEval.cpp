#include "include/MMVII_all.h"
#include "include/V1VII.h"
#include "include/MMVII_Tpl_Images.h"

// #include "include/V1VII.h"


namespace MMVII
{



class cAppliEpipDMEval : public cMMVII_Appli,
                         public cAppliParseBoxIm<tREAL4>
{
     public :
        typedef cAppliParseBoxIm<tREAL4> tAPBI;
        typedef tAPBI::tIm               tImAPBI;
        typedef tAPBI::tDataIm           tDImAPBI;
        typedef cIm2D<tREAL4>            tImPx;
        typedef cIm2D<tU_INT1>           tImMasq;

       // typedef cIm2D

       // =========== Declaration ========
                 // --- Method to be a MMVII application
        cAppliEpipDMEval(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        std::vector<std::string>  Samples() const  override;

                 // --- Method to be a cAppliParseBoxIm<tREAL4>
        int ExeOnParsedBox() override; ///< Action to exec for each box, When the appli parse a big file


     private :

                 // --- Others 
        void MakeImRectified() ; ///< compute rectified image
        void MakeHidden();  ///< make hidden part
        void MakeCorrel();  ///< make image of correlation


        ///  Compute similiraty by normalised cross correlatio,
        double   SimilElemByCor(const cPt2di & aP0,const cPt2di & aP1,int aSzW) const;
        ///  Compute similiraty by censuss quantitif
        double   SimilElemByCQ(const cPt2di & aP0,const cPt2di & aP1,int aSzW) const; // Census quant
        ///  Compute similiraty by radiometry ratio
        double   SimilElemByRad(const cPt2di & aP0,const cPt2di & aP1) const; // Census quant
        /// Compute similarity based on the rank (compare paralax selected with others)
        double   ReliabilityByRank(const cPt2di & aP1) const;
        ///  Compute  simil elementary using one of the above functions
        double   SimilElem(const cPt2di & aP0,const cPt2di & aP1) const;


       // =========== Data ========
           // --- Mandatory arg ----

        std::string     mNameIm2;  ///< Name second image
        std::string     mNamePx1;  ///< Name 
        bool            mR2L;      ///< Is first image right's one

           // --- Optionnal ----
        std::string     mNameMasq1;    ///< Name first image
        std::string     mNameMasq2;    ///< Name first image
        double          mPropIntervPx; ///< 
           // --- Variable ----
        tImAPBI       mIm2Rect;   ///< Rectified im2, superposable to Im1
        tImPx         mImPx1;     ///<  Image of px im1 -> im2
        tImMasq       mImMasq1;   ///<  Image of masq of image 1
        tImMasq       mMasq2Redr; ///< masq of image rectified to image1
        int           mNbDecimPx; ///< Decimation for fast paralx statistic
        int           mSzW;       ///< Size of windows
        cBox2di       mBoxWOk;    ///< Box of pixels where score can be computed taken into account windows size
         
        std::string   mMasqHidden_Name; ///< Name for saving hidden pixel masq
        std::string   mImCorrel_Name;   ///< Name for saving correlation image

        // std::string   mImCorrel_Name;
        // tImIMasq       mImCorrel_Im;
};



cAppliEpipDMEval::cAppliEpipDMEval
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli               (aVArgs,aSpec) ,
   cAppliParseBoxIm<tREAL4>   (*this,true,cPt2di(2000,2000),cPt2di(150,50),true),
   mPropIntervPx              (0.1),
   mIm2Rect                   (cPt2di(1,1)),
   mImPx1                     (cPt2di(1,1)),
   mImMasq1                   (cPt2di(1,1)),
   mMasq2Redr                 (cPt2di(1,1)),
   mNbDecimPx                 (4),
   mSzW                       (2),
   mBoxWOk                    (cBox2di::Empty())
{
}


cCollecSpecArg2007 & cAppliEpipDMEval::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   
   return 
            APBI_ArgObl(anArgObl)
         << Arg2007(mNameIm2,"Name Input Image2",{eTA2007::FileImage})
         << Arg2007(mNamePx1,"Name Paralax Im1->Im2",{eTA2007::FileImage})
         << Arg2007(mR2L,"Is first image the  right ones ?")
   ;
}

std::vector<std::string> cAppliEpipDMEval::Samples() const 
{
   return {"MMVII DenseMatchEpipEval Im_R.tif Im_L.tif Px_R.tif true HiddenMask=HM_R.tif ImCorrel=Cor_R.tif Masq1=Im_R_Masq.tif Masq2=Im_L.tif"};

}


cCollecSpecArg2007 & cAppliEpipDMEval::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
           APBI_ArgOpt
           (
               anArgOpt
                   << AOpt2007(mNameMasq1, "Masq1","Masq of first image if any")
                   << AOpt2007(mNameMasq2, "Masq2","Masq of second image if any")
                   << AOpt2007(mSzW, "SzW","Size of window")
                   << AOpt2007(mMasqHidden_Name, "HiddenMask","Name for save and compute mask of hidden")
                   << AOpt2007(mImCorrel_Name, "ImCorrel","Name for save and compute image of correlation")
           )
   ;
}

double   cAppliEpipDMEval::SimilElemByCor(const cPt2di & aP1,const cPt2di & aP2,int aSzW) const
{
    if ( (!mBoxWOk.Inside(aP1)) ||  (!mBoxWOk.Inside(aP2)) )  
       return 1.0;
    cMatIner2Var<double>  aMat;

    const tDImAPBI & aI1 =  APBI_DIm();
    const tDImAPBI & aI2 =  mIm2Rect.DIm();
    for (const auto & aDP : cRect2::BoxWindow(aSzW))
    {
        aMat.Add(aI1.GetV(aP1+aDP),aI2.GetV(aP2+aDP));
    }
    return (1- aMat.Correl())/2.0;
}

double   cAppliEpipDMEval::SimilElemByCQ(const cPt2di & aP1,const cPt2di & aP2,int aSzW) const
{
    const tDImAPBI & aI1 =  APBI_DIm();
    const tDImAPBI & aI2 =  mIm2Rect.DIm();
    double aVC1 = aI1.GetV(aP1);
    double aVC2 = aI2.GetV(aP2);

    double aSomDif = 0.0;
    double aNb     = 0.0;
    for (const auto & aDP : cRect2::BoxWindow(aSzW))
    {
       double aV1 = aI1.GetV(aP1+aDP);
       double aV2 = aI2.GetV(aP2+aDP);
       double aR1 = NormalisedRatioPos(aV1,aVC1);
       double aR2 = NormalisedRatioPos(aV2,aVC2);
       
       aSomDif += std::abs(aR1-aR2);
       aNb++;
    }
    double aRes = std::min(1.0,aSomDif / aNb);
    aRes = pow(aRes,0.25);

    return aRes;
}

double   cAppliEpipDMEval::SimilElemByRad(const cPt2di & aP1,const cPt2di & aP2) const
{
    const tDImAPBI & aI1 =  APBI_DIm();
    const tDImAPBI & aI2 =  mIm2Rect.DIm();
    double aVC1 = aI1.GetV(aP1);
    double aVC2 = aI2.GetV(aP2);

    return std::abs(NormalisedRatioPos(aVC1,aVC2));
}


void   cAppliEpipDMEval::MakeCorrel()
{
   cIm2D<tU_INT1>  aImCorr(CurSzIn());
   for (const auto & aPix : aImCorr.DIm())
       aImCorr.DIm().SetVTrunc(aPix,255.0*(1-SimilElemByCor(aPix,aPix,mSzW)));
   APBI_WriteIm(mImCorrel_Name,aImCorr);
}


void   cAppliEpipDMEval::MakeHidden()
{
    tImMasq aMasqH(CurSzIn());
    int aSzY = CurSzIn().y();
    int aSzX = CurSzIn().x();

    for (int aY=0 ; aY<aSzY ; aY++)
    {
        tREAL4*  aLIn  =  mImPx1.DIm().ExtractRawData2D()[aY];
        tU_INT1 *  aLOut =  aMasqH.DIm().ExtractRawData2D()[aY];
        tU_INT1 *  aLM1 =  mImMasq1.DIm().ExtractRawData2D()[aY];
        tU_INT1 *  aLM2 =  mMasq2Redr.DIm().ExtractRawData2D()[aY];
        // tImMasq       mMmMasq2Redrasq2Redr;

        tREAL4 aPrevX = -1e6;
        for (int aX=0 ; aX<aSzX ; aX++)
        {
             bool InMasq =  (aLM1[aX]!=0) && (aLM2[aX]!=0) ;

             tREAL4 aVIn  = aLIn[aX] ;
             aPrevX = std::max(aVIn,aPrevX-1);
           
             aLOut[aX] = 255 * ((aPrevX == aVIn) && InMasq);
        }
    }
    APBI_WriteIm(mMasqHidden_Name,aMasqH);
}


double   cAppliEpipDMEval::SimilElem(const cPt2di & aP1,const cPt2di & aP2) const
{
    if ( (!mBoxWOk.Inside(aP1)) ||  (!mBoxWOk.Inside(aP2)) )  
       return 1.0;

    // return SimilElemByCor(aP1,aP2,mSzW);
    // return SimilElemByCQ(aP1,aP2,mSzW);
    return SimilElemByRad(aP1,aP2);
}


double   cAppliEpipDMEval::ReliabilityByRank(const cPt2di & aP1) const
{
      int aNb= 30;
      int aStep = 1;
    
      double  aScore0 = SimilElem(aP1,aP1);

      double aSOk     = 0;
      double aSNotOk  = 0;

      for (int aKPx = -aNb ; aKPx<= aNb ; aKPx++)
      {
          if (aKPx!=0)
          {
              double  aScoreK = SimilElem(aP1,aP1+cPt2di(aKPx*aStep,0));
              if (aScoreK>aScore0)
                 aSOk += 1.0;
              else if (aScoreK<aScore0)
                 aSNotOk += 1.0;
              else
              {
                 aSOk     += 0.5 ;
                 aSNotOk  += 0.5 ;
              }
          }
      }
      return aSNotOk / (aSOk+aSNotOk);
}



void cAppliEpipDMEval::MakeImRectified() 
{
   mIm2Rect.DIm().Resize(CurSzIn());
   mMasq2Redr.DIm().Resize(CurSzIn());

   // Compute interval of Px and then box of Im2
   int aPxMin,aPxMax;
   BornesFonc (aPxMin,aPxMax,mImPx1,&mImMasq1,mNbDecimPx,mPropIntervPx,1.0);
   cBox2di aBoxPx = DilateFromIntervPx(CurBoxIn(),aPxMin,aPxMax);
   aBoxPx = aBoxPx.Inter(DFI2d()); // Must be include in file im2

   tImAPBI  aIm2Init   = tImAPBI::FromFile(mNameIm2,aBoxPx);
   tImMasq  aMasq2Init = ReadMasqWithDef(aBoxPx,mNameMasq2);
   int  aDecPx = aBoxPx.P0().x() - CurBoxIn().P0().x();  // X2-X1  => X2 = X1 + aDecPx

   for (const auto & aPix1 : mIm2Rect.DIm())
   {
       double aX2 = aPix1.x() + mImPx1.DIm().GetV(aPix1) - aDecPx;
       cPt2dr aPix2(aX2 , aPix1.y());
       bool  Ok = aMasq2Init.DIm().DefGetVBL(aPix2,0) > 0.99;
       if (Ok)
       {
           mIm2Rect.DIm().SetV(aPix1,aIm2Init.DIm().GetVBL(aPix2));
       }
       mMasq2Redr.DIm().SetV(aPix1,Ok);
   }
}


int cAppliEpipDMEval::ExeOnParsedBox() 
{
   MMVII_INTERNAL_ASSERT_strong(mR2L,"mode L2R need data to test ");

   StdOut() << "======== ONEBOX =================\n";
   mImPx1 = APBI_ReadIm<tREAL4>(mNamePx1);

   mBoxWOk = CurBoxInLoc().Dilate(-mSzW); // Box of pix with window include => erosion of cur box
   mImMasq1 = ReadMasqWithDef(CurBoxIn(),mNameMasq1);

   MakeImRectified();

   if (IsInit(&mMasqHidden_Name))
      MakeHidden();

   if (IsInit(&mImCorrel_Name))
      MakeCorrel();


   return EXIT_SUCCESS;
}

int cAppliEpipDMEval::Exe()
{
/*
   if (RunMultiSet(0,0))
      return ResultMultiSet();
*/

   APBI_ExecAll();

   return EXIT_SUCCESS;
}


/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_EpipDMEval(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliEpipDMEval(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecEpipDenseMatchEval
(
     "DenseMatchEpipEval",
      Alloc_EpipDMEval,
      "Evaluation of dense matching",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);

};

