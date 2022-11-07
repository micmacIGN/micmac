#include "cMMVII_Appli.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Radiom.h"



namespace MMVII
{

   /* =============================================== */
   /*                                                 */
   /*            cAppliRadiom2Image                   */
   /*                                                 */
   /* =============================================== */

class cRadOneIma;
class cRad_SIoSC;  // SetImOfSameCalib


class cRadOneIma
{
    public :
       cImageRadiomData*     mIRD;
       std::string           mName;
       cPerspCamIntrCalib*   mCalib;
       cRad_SIoSC *          mGRP;
};

class cRad_SIoSC   // SetImOfSameCalib
{
     public :
        std::list<cRadOneIma*>   mLIm;
};


class cAppliRadiom2ImageSameMod : public cMMVII_Appli
{
     public :

        cAppliRadiom2ImageSameMod(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        typedef tREAL8              tElSys;
        typedef cLeasSqtAA<tElSys>  tSys;
        typedef cWeightAv<tElSys>   tWAvg;

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        void MakeLinearCpleIm(size_t aK1,size_t aK2,tSys &);
        void MakeLinearModel();
        void MakeRadialModel(size_t aK1,size_t aK2,tSys & aSys,tWAvg &);
        void MakeRadialModel();

     // ---  Mandatory args ------
    
	std::string     mNamePatternIm;
	std::string     mNameRadiom;

     // ---  Optionnal args ------
	bool    mShow;
	size_t  mNbMinByClpe;
	size_t  mNbDegRad;

     // --- constructed ---
        cPhotogrammetricProject            mPhProj;
	std::vector<cRadOneIma>            mVRadIm;
	size_t                             mNbIm;
	tElSys                             mSomWLinear; // som of weight use in linear step, to fix the gauge constraint
        cDenseVect<tElSys>                 mSolLinear;  // Solution of first linear equation

	std::map<cPerspCamIntrCalib*,cRad_SIoSC>  mMapCal2Im;
};


cCollecSpecArg2007 & cAppliRadiom2ImageSameMod::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl
          <<   Arg2007(mNamePatternIm,"Name of image", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
          <<   mPhProj.RadiomInMand()
          <<   mPhProj.CalibInMand()

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
    mNbDegRad                  (4),
    mPhProj                    (*this),
    mSolLinear                 (1)
{
}


void cAppliRadiom2ImageSameMod::MakeLinearCpleIm(size_t aK1,size_t aK2,tSys & aSys)
{
      std::vector<cPt2di> aVCpleI;
      mVRadIm[aK1].mIRD->GetIndexCommon(aVCpleI,*mVRadIm[aK2].mIRD);
      if (aVCpleI.size() < mNbMinByClpe)
         return;
      const cImageRadiomData::tVRadiom & aVRad1 = mVRadIm[aK1].mIRD->VRadiom(0);
      const cImageRadiomData::tVRadiom & aVRad2 = mVRadIm[aK2].mIRD->VRadiom(0);

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

    mSolLinear = aSys.Solve().Dup();
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


void cAppliRadiom2ImageSameMod::MakeRadialModel(size_t aKIm1,size_t aKIm2,tSys & aSys,tWAvg & aWAvg)
{
      cRadOneIma & aRIm1 = mVRadIm[aKIm1];
      cImageRadiomData * aIRD1 = aRIm1.mIRD;
      const cImageRadiomData::tVRadiom & aVRad1 = aIRD1->VRadiom(0);
      tElSys aRatio1 = mSolLinear(aKIm1);
      const std::vector<cPt2di> & aVP1 = aIRD1->VPts();

      cRadOneIma & aRIm2 = mVRadIm[aKIm2];
      cImageRadiomData * aIRD2 = aRIm2.mIRD;
      const cImageRadiomData::tVRadiom & aVRad2 = aIRD2->VRadiom(0);
      tElSys aRatio2 = mSolLinear(aKIm2);
      const std::vector<cPt2di> & aVP2 = aIRD2->VPts();

      std::vector<cPt2di> aVCpleI;
      aIRD1->GetIndexCommon(aVCpleI,*aIRD2);

      tPt2di aC1 = ToI(aRIm1.mCalib->PP());
      tPt2di aC2 = ToI(aRIm2.mCalib->PP());

      size_t aK1=0;
      size_t aK2=0;

 
      for (const auto & aCpl : aVCpleI)
      {
          cImageRadiomData::tRadiom aRad1 = aVRad1.at(aCpl.x()) / aRatio1;
          cImageRadiomData::tRadiom aRad2 = aVRad2.at(aCpl.y()) / aRatio2;

          aWAvg.Add(1.0,std::abs(aRad1-aRad2));

          cDenseVect<tElSys>  aVec = cDenseVect<tElSys>::Cste(aSys.NbVar(),0.0);
	  tREAL8  aR2_1 = SqN2(aVP1.at(aCpl.x())-aC1);
	  tREAL8  aR2_2 = SqN2(aVP2.at(aCpl.y())-aC2);

	  tREAL8 aPowR1 = aR2_1;
	  tREAL8 aPowR2 = aR2_2;
	  for (size_t aDeg=0 ; aDeg<mNbDegRad ; aDeg++)
	  {
              /*    Rad1 (1 + K1 R1^2 + K2 R^4 +  ...) =  Rad2 (1 + K1 R1^2 + K2 R^4 +  ...)  */	  
              aVec(aK1+aDeg)  +=  aRad1 * aPowR1;
              aVec(aK2+aDeg)  -=  aRad2 * aPowR2;

	      aPowR1 *= aR2_1;
	      aPowR2 *= aR2_2;
	  }
	  aSys.AddObservation(1.0,aVec,aRad2-aRad1);
      }
}

void cAppliRadiom2ImageSameMod::MakeRadialModel()
{
    tWAvg anAvg;
    tSys  aSys(mNbDegRad);

    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        for (size_t aKIm2=aKIm1+1 ; aKIm2<mNbIm; aKIm2++)
        {
            MakeRadialModel(aKIm1,aKIm2,aSys,anAvg);
        }
    }
    StdOut()  <<  " AVGG " << anAvg.Average() << "\n";
}


int cAppliRadiom2ImageSameMod::Exe()
{
    mPhProj.FinishInit();

    mNbIm = VectMainSet(0).size();

    mVRadIm.resize(mNbIm);

    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        if (mShow) 
           StdOut()  << "Reading, Still " << mNbIm - aKIm1 << "\n";
        std::string aNameIm = VectMainSet(0).at(aKIm1);
	cRadOneIma* aRadI = & mVRadIm[aKIm1];
        cPerspCamIntrCalib* aCalib = mPhProj.AllocCalib(aNameIm);

	aRadI->mName = aNameIm; 
	aRadI->mIRD = mPhProj.AllocRadiom(aNameIm);
	aRadI->mCalib = aCalib;

	mMapCal2Im[aCalib].mLIm.push_back(aRadI);
        aRadI->mGRP = & mMapCal2Im[aCalib];
    }
    MMVII_INTERNAL_ASSERT_tiny(mMapCal2Im.size() ==1,"Dont handle multi calib");

    MakeLinearModel();
    MakeRadialModel();

    for (auto & aRadIm : mVRadIm)
    {
        delete aRadIm.mIRD;
    }

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
