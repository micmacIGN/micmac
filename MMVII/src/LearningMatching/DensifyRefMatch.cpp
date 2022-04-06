#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"



namespace MMVII
{

typedef tREAL8  tCoordDensify;
typedef cTriangle2DCompiled<tCoordDensify>  tTriangle2DCompiled;


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
	void MakeOneTri(const  tTriangle2DCompiled &);
	// std::vector<std::string>  Samples() const  override;


           // --- Mandatory ----
           // --- Optionnal ----
	 double              mThreshGrad;
	 double              mNoisePx;
	 int                 mMasq2Tri;

           // --- Internal ----

         cIm2D<tREAL4>       mIPx  ;
         cDataIm2D<tREAL4> * mDIPx ;
         cIm2D<tU_INT1>      mIMasqIn;
         cDataIm2D<tU_INT1>* mDIMasqIn;
         cIm2D<tREAL4>       mImInterp ;
         cDataIm2D<tREAL4>*  mDImInterp;
         cIm2D<tU_INT1>      mIMasqOut;
         cDataIm2D<tU_INT1>* mDIMasqOut;
};

cAppliDensifyRefMatch::cAppliDensifyRefMatch(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch        (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>   (*this,true,cPt2di(2000,2000),cPt2di(50,50)),
   mThreshGrad                (0.3),
   mNoisePx                   (1.0),
   mMasq2Tri                  (0),
   mIPx                       (cPt2di(1,1)),
   mDIPx                      (nullptr),
   mIMasqIn                   (cPt2di(1,1)),
   mDIMasqIn                  (nullptr),
   mImInterp                  (cPt2di(1,1)),
   mDImInterp                 (nullptr),
   mIMasqOut                  (cPt2di(1,1)),
   mDIMasqOut                 (nullptr)
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
                   << AOpt2007(mThreshGrad, "ThG","Threshold for gradient given occlusion",{eTA2007::HDV})
                   << AOpt2007(mNoisePx, "NoisePx","Value of noise in paralax",{eTA2007::HDV})
                   << AOpt2007(mMasq2Tri, "Masq2Tri","Value to set in masq for triangle with 2 vertices low",{eTA2007::HDV,eTA2007::Tuning})
	   )
    // double aNoisePx =  1.0;
    // double aThreshGrad =  0.4;
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

void cAppliDensifyRefMatch::MakeOneTri(const  tTriangle2DCompiled & aTri)
{
    bool   isHGrowPx=false;

    // Compute 3 value in a point
    cPt3dr aPPx;
    for (int aKp=0 ; aKp<3 ; aKp++)
    {
	aPPx[aKp] = mDIPx->GetV(ToI(aTri.Pt(aKp)));
    }
    //  Tricky for WMM, but if used aWMM() => generate warning
    cWhitchMinMax<int,double>  aWMM(0,aPPx[0]);
    for (int aKp=1 ; aKp<3 ; aKp++)
    {
        aWMM.Add(aKp,aPPx[aKp]);
    }

    // Compute Min,Max,Med
    int aKMin = aWMM.Min().IndexExtre();
    int aKMax = aWMM.Max().IndexExtre();
    int aKMed = 3-aKMin-aKMax;   // KMed is remaining index : 0,1,2 => sum equal 3

    double aPxMax = aPPx[aKMax];
    double aPxMin = aPPx[aKMin];
    double aPxMed = aPPx[aKMed];
    
    // Compute attenuation to take into account noise in gradident estimate , 
    double aMul = 1;
    double anEc = aPxMax - aPxMin;
    if (anEc!=0)
    {
        aMul = std::max(0.0,anEc-mNoisePx)/anEc;
    }

    // Compute occlusion on gradient threshold
    cPt2dr aG  = aTri.GradientVI(aPPx)*aMul;
    double aNG = Norm2(aG);
    bool isOcclusion = (aNG>mThreshGrad);
    int aValMasq = isOcclusion ? 0 : 255;

    int aKLow = isHGrowPx ? aKMin : aKMax;
    double  aValOcl = aPPx[aKLow];
    bool isTri2Low =  isOcclusion && (std::abs(aPxMed-aValOcl)<anEc/2.0);

    double aValTri;
    cPt2dr aVecTri;
    if (isTri2Low)  // Case where two vertices of the triangle are low
    {
        aValMasq = mMasq2Tri;
	cSegment aSeg(aTri.Pt(aKLow),aTri.Pt(aKMed));
	aSeg.CompileFoncLinear(aValTri,aVecTri,aPPx[aKLow],aPPx[aKMed]);
    }

    //  Now compute all the pixel and set the value

    static std::vector<cPt2di> aVPixTri;
    aTri.PixelsInside(aVPixTri);
    for (const auto & aPix : aVPixTri)
    {
        if (isOcclusion)
	{
            if (isTri2Low)  // 2 point low, interpol along segment
               mDImInterp->SetV(aPix,aValTri+Scal(aVecTri,ToR(aPix)));
            else
               mDImInterp->SetV(aPix,aValOcl);  // One point low, used lowest value
	}
	else   // Not occluded, use linear interpol
	{
            mDImInterp->SetV(aPix,aTri.ValueInterpol(ToR(aPix),aPPx));
	}
        mDIMasqOut->SetV(aPix,aValMasq);
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
    mIMasqOut  = cIm2D<tU_INT1>(mDIPx->Sz(),nullptr,eModeInitImage::eMIA_Null);
    mDIMasqOut = &mIMasqOut.DIm();
    // cDataIm2D<tREAL4>          tDataImPx;
    StdOut() << "SZIM= " << APBI_DIm().Sz()  << mIPx.DIm().Sz() << "\n";

    std::vector<cPt2dr> aVPts;
    for (const auto & aPix : *mDIMasqIn)
    {
         if (mDIMasqIn->GetV(aPix))
            aVPts.push_back(ToR(aPix));
    }
    cTriangulation2D<tCoordDensify> aTriangul(aVPts);
    aTriangul.MakeDelaunay();
    StdOut() << "NbFace= " <<  aTriangul.NbFace() << "\n";


    // Initiate image of interpolated value
    for (int aKTri=0 ; aKTri<aTriangul.NbFace() ; aKTri++)
    {
         MakeOneTri(cTriangle2DCompiled(aTriangul.KthTri(aKTri)));
    }

    mDImInterp->ToFile("DensifyPx_"+LastPrefix(APBI_NameIm()) + ".tif");
    mDIMasqOut->ToFile("DensifyMasq_"+LastPrefix(APBI_NameIm()) + ".tif");

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
