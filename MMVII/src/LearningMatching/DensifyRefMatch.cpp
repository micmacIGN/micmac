#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"



namespace MMVII
{



class cAppliDensifyRefMatch : public cAppliLearningMatch,
	                      public cAppliParseBoxIm<tREAL4>
{
     public :
        typedef cIm2D<tU_INT1>             tImMasq;
        typedef cIm2D<tREAL4>              tImPx;
        typedef cDataIm2D<tU_INT1>         tDataImMasq;
        typedef cDataIm2D<tREAL4>          tDataImPx;

        cAppliDensifyRefMatch(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	int ExeOnParsedBox() override;
	void MakeOneTri(const  cTriangle2DCompiled &);
	// std::vector<std::string>  Samples() const  override;


           // --- Mandatory ----
           // --- Optionnal ----
           // --- Internal ----

         cIm2D<tREAL4>       mIPx  ;
         cDataIm2D<tREAL4> * mDIPx ;
         cIm2D<tU_INT1>      mIMasqIn;
         cDataIm2D<tU_INT1>* mDIMasqIn;
         cIm2D<tREAL4>       mImInterp ;
         cDataIm2D<tREAL4>*  mDImInterp;
};

cAppliDensifyRefMatch::cAppliDensifyRefMatch(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch        (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>   (*this,true,cPt2di(2000,2000),cPt2di(50,50)),
   mIPx                       (cPt2di(1,1)),
   mDIPx                      (nullptr),
   mIMasqIn                   (cPt2di(1,1)),
   mDIMasqIn                  (nullptr),
   mImInterp                  (cPt2di(1,1)),
   mDImInterp                 (nullptr)
{
}


cCollecSpecArg2007 & cAppliDensifyRefMatch::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      APBI_ArgObl(anArgObl)

          // <<   Arg2007(mIm1,"Name of input(s) file(s), Im1",{{eTA2007::MPatFile,"0"}})
   ;
}

cCollecSpecArg2007 & cAppliDensifyRefMatch::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return APBI_ArgOpt
	   (
	       anArgOpt
           //   << AOpt2007(toto, "toto","ooooo",{eTA2007::HDV})
	   )
          // << AOpt2007(mSzTile, "TileSz","Size of tile for spliting computation",{eTA2007::HDV})
          // << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV,eTA2007::Tuning})
          // << AOpt2007(mCutsParam,"CutParam","Interval Pax + Line of cuts[PxMin,PxMax,Y0,Y1,....]",{{eTA2007::ISizeV,"[3,10000]"}})
          // << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV})
   ;
}

/*
std::vector<std::string>  cAppliDensifyRefMatch::Samples() const
{
    return std::vector<std::string>
           (
               {
                   "MMVII DM2CalcHistoCarac DMTrain_MDLB2014-Vintage-perfect_Box0Std_LDHAime0.dmp Test",
                   "MMVII DM2CalcHistoCarac DMTrain_MDLB2014.*LDHAime0.dmp AllMDLB2014"
               }
          );

}
*/

void cAppliDensifyRefMatch::MakeOneTri(const  cTriangle2DCompiled & aTri)
{
    double aNoisePx =  1.0;
    static std::vector<cPt2di> aVPixTri;
    cPt3dr aPPx;
    double aPxMin =  1e5;
    double aPxMax = -1e5;
    double aSomPx = 0;
    for (int aKp=0 ; aKp<3 ; aKp++)
    {
	aPPx[aKp] = mDIPx->GetV(ToI(aTri.Pt(aKp)));
	UpdateMinMax(aPxMin,aPxMax,aPPx[aKp]);
	aSomPx += aPPx[aKp];
    }
    double aMul = 1;
    double anEc = aPxMax-aPxMin;
    if (anEc!=0)
    {
        aMul = std::max(0.0,anEc-aNoisePx)/anEc;
    }


    aTri.PixelsInside(aVPixTri);

    cPt2dr aG  = aTri.GradientVI(aPPx)*aMul;
    double aNG = Norm2(aG);

    for (const auto & aPix : aVPixTri)
    {
        // mDImInterp->SetV(aPix,aTri.ValueInterpol(ToR(aPix),aPPx));
        mDImInterp->SetV(aPix,aNG);
    }
}

int  cAppliDensifyRefMatch::ExeOnParsedBox()
{
    mIPx    =  APBI_ReadIm<tREAL4> ( Px1FromIm1(APBI_NameIm()));
    mDIPx = &mIPx.DIm();
    mIMasqIn =  APBI_ReadIm<tU_INT1>( Masq1FromIm1(APBI_NameIm()));
    mDIMasqIn = &mIMasqIn.DIm();

    mImInterp = cIm2D<tREAL4>(mDIPx->Sz(),nullptr,eModeInitImage::eMIA_Null);
    mDImInterp = &(mImInterp.DIm());

    // cDataIm2D<tREAL4>          tDataImPx;
    StdOut() << "SZIM= " << APBI_DIm().Sz()  << mIPx.DIm().Sz() << "\n";

    std::vector<cPt2dr> aVPts;
    for (const auto & aPix : *mDIMasqIn)
    {
         if (mDIMasqIn->GetV(aPix))
            aVPts.push_back(ToR(aPix));
    }
    cTriangulation2D aTriangul(aVPts);
    StdOut() << "NbTri= " <<  aTriangul.NbTri() << "\n";


    for (int aKTri=0 ; aKTri<aTriangul.NbTri() ; aKTri++)
    {
         MakeOneTri(cTriangle2DCompiled(aTriangul.KthTri(aKTri)));
    }

     mDImInterp->ToFile("DDDDDDDDDDDD.tif");

    return EXIT_SUCCESS;
}


int  cAppliDensifyRefMatch::Exe()
{
   // If a multiple pattern, run in // by recall
   if (RunMultiSet(0,0))
      return ResultMultiSet();

   APBI_ExecAll();


   return EXIT_SUCCESS;
}



/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_DensifyRefMatch(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliDensifyRefMatch(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecDensifyRefMatch
(
     "DM01DensifyRefMatch",
      Alloc_DensifyRefMatch,
      "Create dense map using a sparse one (LIDAR) with or without images",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);



};
