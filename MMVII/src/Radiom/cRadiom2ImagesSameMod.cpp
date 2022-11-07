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
        typedef cLeasSqtAA<double>  tSys;

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
	   /*
           << AOpt2007(mNameCloud2DIn,"M2","Mesh 2D, dev of cloud 3D,to generate a visu of hiden part ",{eTA2007::FileCloud,eTA2007::Input})
           << AOpt2007(mResolZBuf,"ResZBuf","Resolution of ZBuffer", {eTA2007::HDV})
	   */
   ;
}



cAppliRadiom2ImageSameMod::cAppliRadiom2ImageSameMod(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli               (aVArgs,aSpec),
    mShow                      (false),
    mNbMinByClpe               (50),
    mPhProj                    (*this)
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

      std::vector<double>  aVRatio;
      for (const auto & aCpl : aVCpleI)
      {
          cImageRadiomData::tRadiom aR1 = aVRad1.at(aCpl.x());
          cImageRadiomData::tRadiom aR2 = aVRad2.at(aCpl.y());

	  double aRatio =  aR1/std::max(1.0,double(aR2));
	  aVRatio.push_back(aRatio);

      }
      //  aRatio = R1/R2   R1-R2 Ratio = 0
      //    R1/Sqrt(R) -R2/Sqrt(R) = 0
      double aRMed = KthVal(aVRatio,0.5);

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
    tSys  aSys(mNbIm);
    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        StdOut()  << "Still " << mNbIm - aKIm1 << "\n";
        for (size_t aKIm2=aKIm1+1 ; aKIm2<mNbIm; aKIm2++)
        {
            MakeLinearCpleIm(aKIm1,aKIm2,aSys);
        }
    }
}


int cAppliRadiom2ImageSameMod::Exe()
{
    mPhProj.FinishInit();

    mNbIm = VectMainSet(0).size();

    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
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
