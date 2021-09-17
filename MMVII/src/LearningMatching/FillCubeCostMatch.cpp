#include "include/MMVII_all.h"
//#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{


class cAppliFillCubeCost : public cAppliLearningMatch
{
     public :
        typedef cIm2D<tINT2>               tImZ;
        typedef cDataIm2D<tINT2>           tDataImZ;
        typedef cIm2D<tREAL4>              tImRad;
        typedef cDataIm2D<tREAL4>          tDataImRad;

        typedef cIm2D<tREAL4>              tImPx;
        typedef cDataIm2D<tU_INT1>         tDataImMasq;
        typedef cDataIm2D<tREAL4>          tDataImPx;
        typedef cIm2D<tREAL4>              tImFiltred;
        typedef cDataIm2D<tREAL4>          tDataImF;
        typedef cGaussianPyramid<tREAL4>   tPyr;
        typedef std::shared_ptr<tPyr>      tSP_Pyr;

        cAppliFillCubeCost(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	double ComputCorrel(const cPt2di & aPI1,const cPt2dr & aPI2,int mSzW) const;
	void PushCost(double aCost);

	// -------------- Mandatory args -------------------
	std::string   mNameI1;
	std::string   mNameI2;
	std::string   mModele;
	cPt2di        mP0Z;  // Pt corresponding in Im1 to (0,0)
	cBox2di       mBoxI1;  // Box to Load, taking into account siwe effect
	cBox2di       mBoxI2;
	// cBox2di       mBoxI2;
	std::string   mNamePost;

	// -------------- Optionnal args -------------------
	tREAL8        mStepZ;

	// -------------- Internal variables -------------------
	
	std::string StdName(const std::string & aPre,const std::string & aPost);

        std::string mNameZMin;
        std::string mNameZMax;
        std::string mNameCube;
        cMMVII_Ofs* mFileCube;

	tImZ        mImZMin;
	tImZ        mImZMax;
	tImRad      mIm1;
	tDataImRad  *mDI1;
	tImRad      mIm2;
	tDataImRad  *mDI2;
};

cAppliFillCubeCost::cAppliFillCubeCost(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mBoxI1               (cBox2di::Empty()),
   mBoxI2               (cBox2di::Empty()),
   mStepZ               (1.0),
   mFileCube            (nullptr),
   mImZMin              (cPt2di(1,1)),
   mImZMax              (cPt2di(1,1)),
   mIm1                 (cPt2di(1,1)),
   mDI1                 (nullptr),
   mIm2                 (cPt2di(1,1)),
   mDI2                 (nullptr)
{
}


cCollecSpecArg2007 & cAppliFillCubeCost::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameI1,"Name of first image")
          <<   Arg2007(mNameI2,"Name of second image")
          <<   Arg2007(mModele,"Name for modele : .*dmp|Compare|MMVIICorrel")
          <<   Arg2007(mP0Z,"Origin in first image")
          <<   Arg2007(mBoxI1,"Box to read 4 Im1")
          <<   Arg2007(mBoxI2,"Box to read 4 Im2")
          <<   Arg2007(mNamePost,"Post fix for other names (ZMin,ZMax,Cube)")
   ;
}

cCollecSpecArg2007 & cAppliFillCubeCost::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mStepZ, "StepZ","Step for paralax",{eTA2007::HDV})
   ;
}

std::string cAppliFillCubeCost::StdName(const std::string & aPre,const std::string & aPost)
{
	return aPre + "_" + mNamePost + "." + aPost;
}

double cAppliFillCubeCost::ComputCorrel(const cPt2di & aPCI1,const cPt2dr & aPCI2,int aSzW) const
{
   cMatIner2Var<tREAL4> aMat;

   for (int aDx=-aSzW ; aDx<=aSzW  ; aDx++)
   {
       for (int aDy=-aSzW ; aDy<=aSzW  ; aDy++)
       {
            aMat.Add
            (
                mDI1->GetV  (aPCI1+cPt2di(aDx,aDy)),
                mDI2->GetVBL(aPCI2+cPt2dr(aDx,aDy))
            );
       }
   }

   return aMat.Correl();
}

void cAppliFillCubeCost::PushCost(double aCost)
{
   tU_INT2 aICost = round_ni(1e4*(std::max(0.0,std::min(1.0,aCost))));
   mFileCube->Write(aICost);
}

int  cAppliFillCubeCost::Exe()
{
   // Compute names
   mNameZMin = StdName("ZMin","tif");
   mNameZMax = StdName("ZMax","tif");
   mNameCube = StdName("MatchingCube","data");

   mFileCube = new cMMVII_Ofs(mNameCube,false);

   bool Correl = (mModele=="Compare") ||(mModele=="MMVIICorrel");

   //  Read images 
   mImZMin = tImZ::FromFile(mNameZMin);
   tDataImZ & aDZMin = mImZMin.DIm();
   mImZMax = tImZ::FromFile(mNameZMax);
   tDataImZ & aDZMax = mImZMax.DIm();

   mIm1 = tImRad::FromFile(mNameI1,mBoxI1);
   mDI1 = &(mIm1.DIm());
   mIm2 = tImRad::FromFile(mNameI2,mBoxI2);
   mDI2 = &(mIm2.DIm());

   cPt2di aSz = aDZMin.Sz();
   cPt2di aPix;

   int aSzW=3;
   cPt2di aPSzW(aSzW,aSzW);
   int aCpt=0;

   for (aPix.y()=0 ; aPix.y()<aSz.y() ; aPix.y()++)
   {
       for (aPix.x()=0 ; aPix.x()<aSz.x() ; aPix.x()++)
       {
            cPt2di aPAbs = aPix + mP0Z;
            cPt2di aPC1  = aPAbs-mBoxI1.P0();
            cPt2di aPC20 = aPAbs-mBoxI2.P0();
            for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
            {
               cPt2dr aPC2Z(aPC20.x()+aDz*mStepZ,aPC20.y());

	       if (Correl)
	       {
	           double aCorrel = 0.0;
                   if (WindInside4BL(*mDI1,aPC1,aPSzW) && WindInside4BL(*mDI2,aPC2Z,aPSzW))
 	           {
                       aCorrel = ComputCorrel(aPC1,aPC2Z,aSzW);
	           }
                   PushCost((1-aCorrel)/2.0);
	       }
               aCpt++;
            }
       }
   }

   delete mFileCube;

   return EXIT_SUCCESS;
}




/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_FillCubeCost(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliFillCubeCost(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecFillCubeCost
(
     "DM4FillCubeCost",
      Alloc_FillCubeCost,
      "Fill a cube with matching costs",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::ToDef},
      __FILE__
);



};
