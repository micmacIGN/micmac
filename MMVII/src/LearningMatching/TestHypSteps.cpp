#include "include/MMVII_all.h"
//#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{


class cAppliTestHypStep : public cAppliLearningMatch
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

        cAppliTestHypStep(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	// -------------- Mandatory args -------------------
	std::string mNameIm;
	// -------------- Optionnal args -------------------

	// -------------- Internal variables -------------------

	bool mIsIm1;
        std::string mNamePx;
        std::string mNameMasq;
	tImMasq     mImMasq;
	tImPx       mImPax;
	tImPx       mImRes;
	cPt2di      mSz;
};

cAppliTestHypStep::cAppliTestHypStep(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mImMasq              (cPt2di(1,1)),
   mImPax               (cPt2di(1,1)),
   mImRes               (cPt2di(1,1))
{
}


cCollecSpecArg2007 & cAppliTestHypStep::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameIm,"Name of input image, Im1 or Im2",{{eTA2007::MPatFile,"0"}})
   ;
}

cCollecSpecArg2007 & cAppliTestHypStep::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          // << AOpt2007(mSzTile, "TileSz","Size of tile for spliting computation",{eTA2007::HDV})
   ;
}


int  cAppliTestHypStep::Exe()
{
   mIsIm1 = Im1OrIm2(mNameIm);
   mNamePx = PxFromIm(mNameIm);
   mNameMasq = MasqFromIm(mNameIm);
   mImMasq = tImMasq::FromFile(mNameMasq);
   mImPax  = tImPx::FromFile(mNamePx);
   const tDataImMasq & aDMasq = mImMasq.DIm();
   const tDataImPx  &  aDPax  = mImPax.DIm();

   mSz = aDMasq.Sz();
   mImRes = tImPx(mSz,nullptr,eModeInitImage::eMIA_Null);
   tDataImPx  &  aDRes  = mImRes.DIm();

        // typedef cDataIm2D<tREAL4>          tDataImPx;

   for (int aY=0 ; aY<mSz.y() ; aY++)
   {
       for (int aX0=1;aX0<mSz.x() ; aX0++)
       {
	       /*
	       if ((aDMasq.GetV(cPt2di(aX0-1,aY))!=0) )
	          std::cout<< "JJJJJJ  "  << aY << " " << aX0 << "\n";
		  */
            if ((aDMasq.GetV(cPt2di(aX0-1,aY))!=0) && (aDMasq.GetV(cPt2di(aX0,aY))==0))
	    {
                 int aX1 = aX0;
		 while ((aX1<mSz.x()) && (aDMasq.GetV(cPt2di(aX1,aY))==0))
                       aX1++;
		 if (aX1<mSz.x())
                 {
                     tREAL4 aPx0 = aDPax.GetV(cPt2di(aX0-1,aY));
                     tREAL4 aPx1 = aDPax.GetV(cPt2di(aX1,aY));
		     tREAL4 aSteep = (aPx1-aPx0) / (1+aX1-aX0);
		     for (int aX=aX0+1 ; aX<aX1 ; aX++)
                        aDRes.SetV(cPt2di(aX,aY),aSteep);
                 }
	    }
       }
   }

   std::string aNameOut = "Steep.tif";
   aDRes.ToFile(aNameOut);

   StdOut() << "IM1 " <<  mIsIm1 << "\n";
   StdOut() << "  Px= " <<  PxFromIm(mNameIm) << "\n";
   StdOut() << "  Masq= " <<  MasqFromIm(mNameIm) << "\n";
   return EXIT_SUCCESS;
}




/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_TestHypStep(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliTestHypStep(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecTestHypStep
(
     "DM0BisTestHypStep",
      Alloc_TestHypStep,
      "Compute statistic to fix the relation hiden-part/step-paralax",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::Console},
      __FILE__
);



};
