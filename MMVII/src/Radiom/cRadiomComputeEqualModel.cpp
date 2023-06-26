#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Radiom.h"
#include "MMVII_Stringifier.h"
#include "MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{
class cComputeCalibRadSensor;  
class cComputeCalibRadIma ;


   /* =============================================== */
   /*                                                 */
   /*            cAppliRadiom2Image                   */
   /*                                                 */
   /* =============================================== */

typedef tREAL8              tElSys;


/** Class for computing the radiometric-correction of 1 image  */

class cComputeCalibRadIma : public cMemCheck
{
    public :
       cComputeCalibRadIma (cImageRadiomData* anIRD,cCalibRadiomIma * aCalRadIm) :
           mIRD (anIRD),
	   mCalRadIm (aCalRadIm)
       {
       }

       ~cComputeCalibRadIma()
       {
	       delete mIRD;
	       delete mCalRadIm;
       }
       cImageRadiomData&   IRD() {return *mIRD;}
       cCalibRadiomIma &   CalRadIm() {return *mCalRadIm;}
       const std::string & NameIm() const {return mCalRadIm->NameIm();}


    private :
       cComputeCalibRadIma(const cComputeCalibRadIma &) = delete;

       cImageRadiomData*         mIRD;            ///< Data containg radiomety => ID, point, 
       cCalibRadiomIma *         mCalRadIm;       ///< Radiometric model of the image
};



/* **************************************************************** */
/*                                                                  */
/*                     cAppliRadiom2ImageSameMod                    */
/*                                                                  */
/* **************************************************************** */

typedef std::pair<cPerspCamIntrCalib*,tREAL8>  tIdCalRad;

/**  Class for doing a basic equalization, we have :
 *
 *     - a multiplicative unknown/image
 *     - a radial unkonwown funcion/sensor
 *
 *
 *     We proceed in two step :
 *     - (1) compute image mult only, using a robust approach  
 *          * median of ratio for each pair
 *          * then least squaure with this median as observation
 *     - (2) global system, using previous solution as init, the obs are :
 *
 *         I1(p1)              I2(p2)
 *   ----------------- =   -------------------
 *     k1   Rad1(p1)         k2  Rad2(p2)
 *
 *         during the first iterations ki are fixed,  for a supplementary obs  is addeed Avg(ki) = 1
 */
class cAppliRadiom2ImageSameMod : public cMMVII_Appli
{
     public :

        cAppliRadiom2ImageSameMod(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        typedef cLeasSqtAA<tElSys>  tSys;
        typedef cWeightAv<tElSys>   tWAvg;

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // Compute robust ratio between pair, and add it as obversyion to global ratio
        void ComputeLinearModelCpleIm(size_t aNumIm1,size_t aNumIm2);
        // Make a first global linear model, using all ratio
        void ComputeLinearModel();

	void InitSys(bool FrozenRad,int aMaxDegIma);

        void   OneIterationGen(int aDegFroze);

        tREAL8   ComputeSigma(std::vector<tElSys> & aVSima);

     // ---  Mandatory args ------
    
	std::string     mNamePatternIm;

     // ---  Optionnal args ------
	bool    mShow;
	size_t  mNbMinByClpe;

     // --- constructed ---
        cPhotogrammetricProject            mPhProj;     ///< The Project, as usual
	std::vector<cComputeCalibRadIma*>  mVRadIm;     ///< set of all unknown ca
	std::set<cCalibRadiomSensor *>     mSetCalRS;   ///< set of sensor calib : use std::set to avoid duplicata
	size_t                             mNbIm;       ///< number of images : facility 
	tElSys                             mSomWLinear; ///< som of weight use in linear step, to fix the gauge constraint
        cDenseVect<tElSys>                 mSolRadial;  ///< Solution of radial system 

	// std::map<cPerspCamIntrCalib*,cComputeCalibRadSensor*>  mMapCal2Im;
	// std::map<tIdCalRad,cComputeCalibRadSensor*>  mMapCal2Im; ///<  Mapd GeomCalib+Aperture -->  Radial Calib
	cFusionIRDSEt                                mFusIndex;  ///< Fusion index for creating multiple points
	cSetInterUK_MultipeObj<tREAL8> *             mSetInterv;  ///< Handle unkwons
	cResolSysNonLinear<tREAL8> *                 mCurSys;
        int                                          mMaxDegree; ///< Max Degree used in image model
        cSparseVect<tElSys>                          mSVFixSum;  // sparse vector to fixx the sum
};


cCollecSpecArg2007 & cAppliRadiom2ImageSameMod::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl
          <<   Arg2007(mNamePatternIm,"Name of image", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
          <<   mPhProj.DPRadiomData().ArgDirInMand()
          <<   mPhProj.DPRadiomModel().ArgDirInMand()
          <<   mPhProj.DPRadiomModel().ArgDirOutMand()
          <<   mPhProj.DPOrient().ArgDirInMand("InputCalibration")

   ;
}

cCollecSpecArg2007 & cAppliRadiom2ImageSameMod::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mShow,"Show","Show messages",{eTA2007::HDV})
   ;
}



cAppliRadiom2ImageSameMod::cAppliRadiom2ImageSameMod(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli               (aVArgs,aSpec),
    mShow                      (true),
    mNbMinByClpe               (50),
    mPhProj                    (*this),
    mSolRadial                 (1),
    mFusIndex                  (1),
    mCurSys                    (nullptr),
    mMaxDegree                 (-1)
{
}

        //  =================================================
        //               Linear  correction 
        //  =================================================

void cAppliRadiom2ImageSameMod::ComputeLinearModelCpleIm(size_t aK1,size_t aK2)
{
      cComputeCalibRadIma & aRad1 = *mVRadIm.at(aK1);
      cComputeCalibRadIma & aRad2 = *mVRadIm.at(aK2);
      // Extract the index of pair common to  Rad1 & Rad2
      std::vector<cPt2di> aVCpleI;
      aRad1.IRD().GetIndexCommon(aVCpleI,aRad2.IRD());
      if (aVCpleI.size() < mNbMinByClpe)
         return;

      const cImageRadiomData::tVRadiom & aVRad1 = mVRadIm[aK1]->IRD().VRadiom(0);
      const cImageRadiomData::tVRadiom & aVRad2 = mVRadIm[aK2]->IRD().VRadiom(0);

      // 1 : estimate the ratio between Im1 and Im2,
      // use the median of ration as a robust estimator
      std::vector<double>  aVRatio;  // stack of ratio
      for (const auto & aCpl : aVCpleI)
      {
          cImageRadiomData::tRadiom aR1 = aVRad1.at(aCpl.x());
          cImageRadiomData::tRadiom aR2 = aVRad2.at(aCpl.y());

	  double aRatio =  aR1/std::max(1.0,double(aR2));
	  aVRatio.push_back(aRatio);

      }
      double aRMed = NC_KthVal(aVRatio,0.5); // estimate the median as propotion 0.5

      //  Use sqrt of ratio to have a more symetric equation
      //  R1 and R2 are the unkown multiplier , ratio is the observation computed
      //  aRatio = R1/R2   R1-R2 Ratio = 0
      //    to symetrize and linarize we write equation :
      //           R1/Sqrt(R) -R2*Sqrt(R) = 0  [EqRatio]
      double aSqrR = std::sqrt(aRMed);

      tElSys  aWeight = std::sqrt(aVCpleI.size()); // a bit (a lot ?) arbitrary
      mSomWLinear += aWeight;						   

      cSparseVect<tElSys>  aSV;  // sparse vector will be filled by eq  [EqRatio]

      aSV.AddIV(aRad1.CalRadIm().IndDegree(cPt2di(0,0)),1.0/aSqrR);
      aSV.AddIV(aRad2.CalRadIm().IndDegree(cPt2di(0,0)),-aSqrR);
      mCurSys->AddObservationLinear(aWeight,aSV,0.0);  // Weight/Linear/  0.0=Const


      if (false)
         StdOut() 
	      << " Low=" << NC_KthVal(aVRatio,0.2) 
	      << " MED=" << aRMed
	      << " High=" << NC_KthVal(aVRatio,0.8) 
	      << " Nb=" << aVCpleI.size()
	      << " " << mVRadIm.at(aK1)->NameIm()
	      << " " << mVRadIm.at(aK2)->NameIm()
	      << "\n"; 

}


void cAppliRadiom2ImageSameMod::ComputeLinearModel()
{
    InitSys(true,0);

    // Add all equations on ratio
    mSomWLinear = 0.0;
    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        if ((aKIm1%50)==0)
           StdOut()  << "      Linear Still " << mNbIm - aKIm1 << "\n";
        for (size_t aKIm2=aKIm1+1 ; aKIm2<mNbIm; aKIm2++)
        {
            ComputeLinearModelCpleIm(aKIm1,aKIm2);
        }
    }

    mCurSys->AddObservationLinear(mSomWLinear/1e2,mSVFixSum,tElSys(mNbIm));

    // we get a vector of ratio
    cDenseVect<tElSys> aSol = mCurSys->SolveUpdateReset();

    // aSol.SetAvg(1.0);  // we force Avg to 1
    mSetInterv->SetVUnKnowns(aSol); // we retransfer the value in object
}

        //  =================================================
        //               Mixt correction
        //  =================================================

tREAL8   cAppliRadiom2ImageSameMod::ComputeSigma(std::vector<tElSys> & aVSigma)
{
     aVSigma.clear();
     // 1- Compute sigma
     for (const auto & aVecIndMul : mFusIndex.VVIndexes())
     {
         cUB_ComputeStdDev<1>  aDev;  // compute deviation for one point
         for (const  auto & aImInd : aVecIndMul)   // aImInd : x=NumIm , y=NumPoint
         {
             cComputeCalibRadIma * aRIM = mVRadIm.at(aImInd.x());
	     int aIndPt = aImInd.y();

	     cPt2dr  aPt = ToR(aRIM->IRD().Pt(aIndPt)); // extract point to correct rad
	     tREAL8    aRadCor =   aRIM->CalRadIm().ImageCorrec(aRIM->IRD().Gray(aIndPt),aPt);

             aDev.Add(&aRadCor,1.0);  // accumulate population of corrected radiometry
         }
	 double aVar = sqrt(*(aDev.ComputeUnBiasedVar()));
         aVSigma.push_back(aVar);
     }

     // Extract robust estimator, important to use constant to maintain order as aVSIgma will be reused
     return Cst_KthVal(aVSigma,0.75);
}

void   cAppliRadiom2ImageSameMod::OneIterationGen(int aDegFroze)
{
     std::vector<tElSys> aVSigma;
     tElSys aSigma = ComputeSigma(aVSigma);
     // 1- Compute sigma

     InitSys(false,aDegFroze);  // froze the variables
     cWeightAv<tREAL8> aAvgAlb;  // compute average albeda (used for stat normalization)

     tElSys  aSomWeight = 0.0;
     // 2- Add equations
     for (size_t aKPMul=0; aKPMul<mFusIndex.VVIndexes().size(); aKPMul++)
     {
          const auto & aVecIndMul = mFusIndex.VVIndexes().at(aKPMul);
         // 2.1 - Compute "albedo" = average of corrected radiom
         tElSys aAlbedo = 0;
         for (const  auto & aImInd : aVecIndMul)
         {
              cComputeCalibRadIma * aRIM = mVRadIm.at(aImInd.x());
              int aIndPt = aImInd.y();
              cPt2dr aPt =  ToR(aRIM->IRD().Pt(aIndPt));
              aAlbedo +=  aRIM->CalRadIm().ImageCorrec(aRIM->IRD().Gray(aIndPt),aPt);
         }
         aAlbedo /= aVecIndMul.size();

	 tElSys  aW =  1.0 / sqrt(1.0 + Square(aVSigma[aKPMul]/aSigma));  // compute weight with micmac's "magical" formula
	 cResidualWeighter<tElSys> aRW(aW);  // weighter constant
	 aAvgAlb.Add(1.0,aAlbedo);  // accumulate for albedo averaging

         // 2.2 -  Add observation
         cSetIORSNL_SameTmp<tElSys>  aSetTmp({aAlbedo}); // structure for schur-substitution of albedo
         for (const  auto & aImInd : aVecIndMul)
         {
              // extract  Image+Sensor Calib, radiom data
             cComputeCalibRadIma * aRIM = mVRadIm.at(aImInd.x());
             cCalibRadiomIma & aCRI = aRIM->CalRadIm();
	     cCalibRadiomSensor & aCalS = aCRI.CalibSens();
             cImageRadiomData&   aIRD = aRIM->IRD();

	     // extract point & Radiometry of image
             int aIndPt = aImInd.y();
             cPt2dr  aPt =  ToR(aIRD.Pt(aIndPt));
             tREAL8  aRadIm = aIRD.Gray(aIndPt);

	     // compute observation an unknowns current value
             std::vector<tREAL8> aVObs = Append({aRadIm},aCalS.VObs(aPt)); // Radiom,P.x(),P.y(),C.x(),C.y(),Rho
             std::vector<int>  aVInd({-1});  // initialize with negative index to subst
             aCalS.PushIndexes(aVInd);       // Add indexes of sensor calibration
             aCRI.PushIndexes(aVInd);        // Add indexes of image calibration

	     //  Now accumulate observation
             mCurSys->AddEq2Subst(aSetTmp,aCRI.ImaEqual(),aVInd,aVObs,aRW);

	     aSomWeight += aW;
         }
	 // All equation for 1 albedo have been accumulated, we can make substitution
         mCurSys->AddObsWithTmpUK(aSetTmp); 
     }

     if (mShow)
     {
         // Normalized sigmais important for comparing different option, else we can artificially privelegiate formulas
	 // that make decrease globally the radiometry
         StdOut()  
	           << "* SigAbs=" << aSigma 
		   << " SigNorm= " << 100.0*(aSigma/aAvgAlb.Average()) 
	           << " AVGALB " << aAvgAlb.Average() 
		   << "\n";
         // for (auto & aPtrCalS : mSetCalRS) StdOut() << "CRADD " << aPtrCalS->CoeffRad() << "\n";
     }

    // 3/Add an equation that fix  Avg(Ratio) = 1
    // If not the system is undetermined, takes equation 
    //    {aRadIm - aAlbedo * aCorrecIm * aCorrecSens}; 
    // Albedo/L and aCorrecIm *L will lead to the same solution
    if ( aDegFroze>=0)  // Dont do it if everything is frozen
    {
       // Weight is a "small" weight, proportionnal to total weighting
       mCurSys->AddObservationLinear(aSomWeight/1e2,mSVFixSum,tElSys(mNbIm));
    }

    cDenseVect<tElSys> aSol = mCurSys->SolveUpdateReset();
    mSetInterv->SetVUnKnowns(aSol);
}

        //  =================================================
        //     Exe  ->   global call
        //  =================================================

void cAppliRadiom2ImageSameMod::InitSys(bool FrozenSens,int aMaxDegIma)
{
     mCurSys->UnfrozeAll();

     // If sensor are frozen, fix all their var
     if (FrozenSens)
     {
         for (auto & aPtrCalS : mSetCalRS)
             mCurSys->SetFrozenAllCurrentValues(*aPtrCalS);
     }

     // for all image, freeze all polynom with Degree > aMaxDegIma
     for (auto & aRadIm : mVRadIm)
     {
          auto & aCalRad = aRadIm->CalRadIm();  
          auto & aVParam = aCalRad.Params();   // extract vector of parameters 
          const auto & aVDesc = aCalRad.VDesc(); // extract the descriptor associated to it
          for (size_t aKP=0 ; aKP<aVParam.size() ; aKP++)
          {
              if (aVDesc.at(aKP).mDegTot >aMaxDegIma)
	      {
                 mCurSys->SetFrozenVarCurVal(aCalRad,aVParam[aKP]);
	      }
	      else
	      {
	      }
          }
     }
}

int cAppliRadiom2ImageSameMod::Exe()
{
    mPhProj.FinishInit();  //  necessary to use mPhProj

    //==================================================
    //     1  INITIALIZE / READ
    //==================================================
    mSetInterv	= new cSetInterUK_MultipeObj<tREAL8>;

    size_t aLimitIndex =0;  // will store max of max of index for dimensionning mFusIndex
    mNbIm = VectMainSet(0).size(); // facility

    //  For each image : read initial model, read data, update degree of polynom (Max), update
    //  max of index point used, 
    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        if ((aKIm1%50)==0)
           StdOut()  << "      Reading, Still " << mNbIm - aKIm1 << "\n";
        std::string aNameIm = VectMainSet(0).at(aKIm1);
        cImageRadiomData* aIRD = mPhProj.ReadRadiomData(aNameIm);   // read data radiom
        if (aIRD->VIndex().size() >10)  //if enough data
        {
            UpdateMax(aLimitIndex,aIRD->LimitIndex());  // need to know max inde
	    cCalibRadiomIma *  aCRI = mPhProj.ReadCalibRadiomIma(aNameIm); // read inital cam radiom model

	    UpdateMax(mMaxDegree,aCRI->MaxDegree());
	    mVRadIm.push_back(new cComputeCalibRadIma(aIRD,aCRI));

	    mSetCalRS.insert(&aCRI->CalibSens());
	    mSetInterv->AddOneObj(aCRI);
            mSVFixSum.AddIV(aCRI->IndCste(),1.0);  // Add index of cste polyn
        }
        else
           delete aIRD;  // free now, will no longer be referenced
    }
    mNbIm  = mVRadIm.size();  // update , not all image selected

    // add the sensor calibration 
    for (auto & aPtrCalS : mSetCalRS)
        mSetInterv->AddOneObj(aPtrCalS);

    // create the  non linear optimizer
    mCurSys = new  cResolSysNonLinear<tElSys>(eModeSSR::eSSR_LsqDense,mSetInterv->GetVUnKnowns());
    //  seems to work as well
    // mCurSys = new  cResolSysNonLinear<tElSys>(eModeSSR::eSSR_LsqNormSparse,mSetInterv->GetVUnKnowns());


    //  Tentative not to use normal equation, because schurr complemnt can be cost important; fast at the beginin
    //  but after become slow
    // mCurSys = new  cResolSysNonLinear<tElSys>(eModeSSR::eSSR_LsqSparseGC,mSetInterv->GetVUnKnowns());



    // Finalize indexing of multiple tie points
    mFusIndex.Resize(aLimitIndex);
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
        mFusIndex.AddIndex(aKIm,mVRadIm[aKIm]->IRD().VIndex());
    mFusIndex.FilterSzMin(2);
   
    //==================================================
    //     2  OPTIMIZE  MODEL 
    //==================================================

    std::vector<tElSys> aVSigma;
    StdOut()  << "* SigmaInit=" << ComputeSigma(aVSigma) << "\n";

    //   ==========  Compute initial model, begin by this because we can very
    //   ==========  high dynamic for it (for ex if different apperture time)
    ComputeLinearModel();

    //  ====  make adjustment =================
    for (int aDegree=-1 ; aDegree<=mMaxDegree ; aDegree++)
    {
         StdOut() << "=============== Begin degree :" << aDegree << " ======= \n";
         OneIterationGen(aDegree);
         OneIterationGen(aDegree);
    }
    OneIterationGen(mMaxDegree);  // one more iteration
    StdOut()  << "* SigmaFinal=" << ComputeSigma(aVSigma) << "\n";
 
    //==================================================
    //     3  SAVE AND FREE
    //==================================================

    //   =====  Save the results ===============
    for (auto & aRadIm : mVRadIm)
    {
        mPhProj.SaveCalibRad(aRadIm->CalRadIm());
    }

    //  ===   free memory allocated, required for checking ===================
    delete mSetInterv;
    delete mCurSys;
    DeleteAllAndClear(mVRadIm);

    // ====== Ok, all were fine =================
    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_Radiom2ImageSameMod(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliRadiom2ImageSameMod(aVArgs,aSpec));
   //return tMMVII_UnikPApli(nullptr);
}


cSpecMMVII_Appli  TheSpecRadiom2ImageSameMod
(
     "RadiomComputeEqual",
      Alloc_Radiom2ImageSameMod,
      "Estimate radiometric model for equalization",
      {eApF::Radiometry},
      {eApDT::Radiom},
      {eApDT::Radiom},
      __FILE__
);

};
