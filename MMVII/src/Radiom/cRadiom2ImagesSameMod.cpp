/*
#include "MMVII_Ptxd.h"
#include "MMVII_Radiom.h"
#include "MMVII_Stringifier.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Bench.h"
*/

#include "MMVII_Sensor.h"
#include "MMVII_Radiom.h"
#include "cMMVII_Appli.h"


namespace MMVII
{

   /* =============================================== */
   /*                                                 */
   /*            cAppliRadiom2Image                   */
   /*                                                 */
   /* =============================================== */

class cAppliRadiom2ImageSameMod : public cMMVII_Appli
{
     public :

        cAppliRadiom2ImageSameMod(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        typedef tREAL8              tElSys;
        typedef cLeasSqtAA<tElSys>  tSys;

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        void MakeLinearCpleIm(size_t aK1,size_t aK2,tSys &);
        void MakeLinearModel();

     // ---  Mandatory args ------
    
	std::string     mNamePatternIm;
	std::string     mNameRadiom;

     // ---  Optionnal args ------
	bool    mShow;
	size_t  mNbMinByClpe;

     // --- constructed ---
        cPhotogrammetricProject            mPhProj;
	std::vector<cImageRadiomData *>    mVIRD;
	size_t                             mNbIm;
	tElSys                             mSomWLinear; // som of weight use in linear step, to fix the gauge constraint
        cDenseVect<tElSys>                 mSolLinear;  // Solution of first linear equation
};


cCollecSpecArg2007 & cAppliRadiom2ImageSameMod::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl
          <<   Arg2007(mNamePatternIm,"Name of image", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
          <<   mPhProj.RadiomInMand()

   ;
}

cCollecSpecArg2007 & cAppliRadiom2ImageSameMod::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mShow,"Show","Show messages",{eTA2007::HDV})
	   /*
           << AOpt2007(mResolZBuf,"ResZBuf","Resolution of ZBuffer", {eTA2007::HDV})
	   */
   ;
}



cAppliRadiom2ImageSameMod::cAppliRadiom2ImageSameMod(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli               (aVArgs,aSpec),
    mShow                      (true),
    mNbMinByClpe               (50),
    mPhProj                    (*this),
    mSolLinear                 (1)
{
}


void cAppliRadiom2ImageSameMod::MakeLinearCpleIm(size_t aK1,size_t aK2,tSys & aSys)
{
      std::vector<cPt2di> aVCpleI;
      mVIRD[aK1]->GetIndexCommon(aVCpleI,*mVIRD[aK2]);
      if (aVCpleI.size() < mNbMinByClpe)
         return;
      const cImageRadiomData::tVRadiom & aVRad1 = mVIRD[aK1]->VRadiom(0);
      const cImageRadiomData::tVRadiom & aVRad2 = mVIRD[aK2]->VRadiom(0);

      // compute the median of ration as a robust estimator
      std::vector<double>  aVRatio;
      for (const auto & aCpl : aVCpleI)
      {
          cImageRadiomData::tRadiom aR1 = aVRad1.at(aCpl.x());
          cImageRadiomData::tRadiom aR2 = aVRad2.at(aCpl.y());

	  double aRatio =  aR1/std::max(1.0,double(aR2));
	  aVRatio.push_back(aRatio);

      }
      double aRMed = KthVal(aVRatio,0.5);

      //  Use sqrt of ratio to have a more symetric equation
      //  aRatio = R1/R2   R1-R2 Ratio = 0
      //    R1/Sqrt(R) -R2*Sqrt(R) = 0
      double aSqrR = std::sqrt(aRMed);

      tElSys  aWeight = std::sqrt(aVCpleI.size()); // a bit (a lot ?) arbitrary
      mSomWLinear += aWeight;						   

      cSparseVect<tElSys>  aSV;
      aSV.AddIV(aK1,1.0/aSqrR);
      aSV.AddIV(aK2,-aSqrR);
      aSys.AddObservation(aWeight,aSV,0.0);


      if (mShow)
         StdOut() 
	      << " Low=" << KthVal(aVRatio,0.2) 
	      << " MED=" << aRMed
	      << " High=" << KthVal(aVRatio,0.8) 
	      << " Nb=" << aVCpleI.size()
	      << " " << VectMainSet(0).at(aK1)  
	      << " " << VectMainSet(0).at(aK2) 
	      << "\n"; 

}

void cAppliRadiom2ImageSameMod::MakeLinearModel()
{
    mSomWLinear = 0.0;
    tSys  aSys(mNbIm);
    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        StdOut()  << "Still " << mNbIm - aKIm1 << "\n";
        for (size_t aKIm2=aKIm1+1 ; aKIm2<mNbIm; aKIm2++)
        {
            MakeLinearCpleIm(aKIm1,aKIm2,aSys);
        }
    }

    // Add an equation that fix  Avg(Ratio) = 1
    cDenseVect<tElSys>  aVecAll1 = cDenseVect<tElSys>::Cste( mNbIm,1.0);
    aSys.AddObservation(mSomWLinear/1e2,aVecAll1,tElSys(mNbIm));

    mSolLinear = aSys.Solve();
    tElSys anAVg = mSolLinear.SumElem() /mNbIm;
    StdOut() << " 111 - AVG LINEAR= " << anAVg-1 << "\n";

    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
	mSolLinear(aKIm1) /= anAVg;
        StdOut()  << "LINEAR " << mSolLinear(aKIm1)  << " for : " << VectMainSet(0).at(aKIm1) << "\n";
    }
    anAVg = mSolLinear.SumElem() /mNbIm;
    StdOut() << "AVG LINEAR= " << anAVg-1 << "\n";
}


int cAppliRadiom2ImageSameMod::Exe()
{
    mPhProj.FinishInit();

    mNbIm = VectMainSet(0).size();

    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        if (mShow) 
           StdOut()  << "Reading, Still " << mNbIm - aKIm1 << "\n";
        std::string aNameIm = VectMainSet(0).at(aKIm1);
        mVIRD.push_back(mPhProj.AllocRadiom(aNameIm));
    }


    MakeLinearModel();


    DeleteAllAndClear(mVIRD);
    return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_Radiom2ImageSameMod(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliRadiom2ImageSameMod(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecRadiom2ImageSameMod
(
     "RadiomTest",
      Alloc_Radiom2ImageSameMod,
      "Estimate radiometric model (test 4 now)",
      {eApF::Radiometry},
      {eApDT::Radiom},
      {eApDT::Radiom},
      __FILE__
);

};
