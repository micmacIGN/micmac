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

class cComputeCalibRadIma : public cMemCheck,
                            public cObjWithUnkowns<tElSys>
{
    public :
       cComputeCalibRadIma (const std::string& aName,cImageRadiomData*,cComputeCalibRadSensor* aGRP);
       void PutUknowsInSetInterval() override ; ///< overiding cObjWithUnkowns
       tREAL8   CorrectedRadiom(tREAL8,const cPt2df & aPt) const; ///< used to correct radiom (before weighting)

       ~cComputeCalibRadIma();
       tREAL8 DivIm() const; ///< Value to divide image (contained in mCalRadIm)


       const std::string & NameIm() const;            ///< Accessor
       cImageRadiomData&   IRD();                     ///< Accessor
       cComputeCalibRadSensor & ComputeCalSens();     ///< Accessor
       const cCalibRadiomIma & CalibRabIma() const;   ///< Accessor


    private :
       cComputeCalibRadIma(const cComputeCalibRadIma &) = delete;

       std::string               mNameIm;         ///< Name of image, always usefull
       cImageRadiomData*         mIRD;            ///< Data containg radiomety => ID, point, 
       cComputeCalibRadSensor *  mComputeCalSens; ///< data that will compute sensor calib
       cCalRadIm_Cst             mCalRadIm;       ///< Radiometric model of the image
};

/** class for computing the radiometric correction of one sensor */

class cComputeCalibRadSensor : public cMemCheck,  // SetImOfSameCalib
	           public cObjWithUnkowns<tElSys>
{
     public :
        cComputeCalibRadSensor(cPerspCamIntrCalib * aCalib,int aNbRad,const std::string& aNameCalRad);
        void PutUknowsInSetInterval() override ;

	tElSys  NormalizedRho2(const cPt2df & aPt) const {return mRCRS.NormalizedRho2(ToR(aPt)) ;}
	tElSys  FactCorrecRadiom(const cPt2df & aPt) const;
	cRadialCRS & RCRS();
     private :
	cComputeCalibRadSensor(const cComputeCalibRadSensor &) = delete;

	cRadialCRS    mRCRS;
};

    // ===================================================
    //             cComputeCalibRadIma   
    // ===================================================

cComputeCalibRadIma::cComputeCalibRadIma(const std::string& aNameIm,cImageRadiomData* aIRD,cComputeCalibRadSensor* aCompCalS) :
   mNameIm           (aNameIm),
   mIRD              (aIRD),
   mComputeCalSens   (aCompCalS),
   mCalRadIm         (&(aCompCalS->RCRS()),aNameIm)
{
}

const cCalibRadiomIma & cComputeCalibRadIma::CalibRabIma() const {return mCalRadIm;}

cComputeCalibRadIma::~cComputeCalibRadIma()
{
   delete mIRD;
}

void cComputeCalibRadIma::PutUknowsInSetInterval() 
{
	mSetInterv->AddOneInterv(mCalRadIm.DivIm());
}

/** mCalRadIm  : contain ImageFact + Ptr to CalibSensor, so its modifed by OnUpdate */
tElSys  cComputeCalibRadIma::CorrectedRadiom(tREAL8 aRadInit,const cPt2df & aPt) const
{
	return aRadInit / (mCalRadIm.ImageCorrec(ToR(aPt)));
}

const std::string & cComputeCalibRadIma::NameIm() const {return mNameIm;}
cImageRadiomData&   cComputeCalibRadIma::IRD() {return *mIRD;}

cComputeCalibRadSensor & cComputeCalibRadIma::ComputeCalSens() {return *mComputeCalSens;}
tREAL8 cComputeCalibRadIma::DivIm() const {return mCalRadIm.DivIm();}

    // ===================================================
    //             cComputeCalibRadSensor   
    // ===================================================

cComputeCalibRadSensor::cComputeCalibRadSensor(cPerspCamIntrCalib * aCalib,int aNbRad,const std::string& aNameCalRad)  :
       mRCRS (aCalib->PP(),aNbRad,aCalib->SzPix(),aNameCalRad)
{
   int TheCpt = 0; 
   TheCpt++;
   if (TheCpt!=1)
   {
      // A priori multiple calib has been validated
      if (NeverHappens())
      {
         MMVII_DEV_WARNING("cComputeCalibRadSensor test multi calib")
      }
   }

}

void cComputeCalibRadSensor::PutUknowsInSetInterval() 
{
     mSetInterv->AddOneInterv(mRCRS.CoeffRad());
}

tElSys  cComputeCalibRadSensor::FactCorrecRadiom(const cPt2df & aPt) const
{
     return mRCRS.FlatField(ToR(aPt));
}

cRadialCRS & cComputeCalibRadSensor::RCRS() {return mRCRS;}

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

        void MakeLinearCpleIm(size_t aK1,size_t aK2);
        void MakeLinearModel();

        void   MakeOneIterMixtModel(int aKIter);
        void   MakeMixtModel();

     // ---  Mandatory args ------
    
	std::string     mNamePatternIm;
	std::string     mNameRadiom;

     // ---  Optionnal args ------
	bool    mShow;
	size_t  mNbMinByClpe;
	size_t  mNbDegRad;

     // --- constructed ---
        cPhotogrammetricProject            mPhProj;     ///< The Project, as usual
	std::vector<cComputeCalibRadIma*>  mVRadIm;     ///< set of all unknown ca
	size_t                             mNbIm;       ///< number of images : facility 
	tElSys                             mSomWLinear; ///< som of weight use in linear step, to fix the gauge constraint
        cDenseVect<tElSys>                 mSolRadial;  ///< Solution of radial system 

	// std::map<cPerspCamIntrCalib*,cComputeCalibRadSensor*>  mMapCal2Im;
	std::map<tIdCalRad,cComputeCalibRadSensor*>  mMapCal2Im; ///<  Mapd GeomCalib+Aperture -->  Radial Calib
	cFusionIRDSEt                                mFusIndex;  ///< Fusion index for creating multiple points
	cCalculator<double> *                        mCalcEqRad; ///< calculator for equalizing radiometry
	cSetInterUK_MultipeObj<tElSys> *             mSetInterv;  ///< Handle unkwons
	cResolSysNonLinear<tElSys> *                 mCurSys;
};


cCollecSpecArg2007 & cAppliRadiom2ImageSameMod::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return anArgObl
          <<   Arg2007(mNamePatternIm,"Name of image", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
          <<   mPhProj.DPRadiom().ArgDirInMand()
          <<   mPhProj.DPOrient().ArgDirInMand("InputCalibration")

   ;
}

cCollecSpecArg2007 & cAppliRadiom2ImageSameMod::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mShow,"Show","Show messages",{eTA2007::HDV})
	   << mPhProj.DPRadiom().ArgDirOutOpt()
	   /*
           << AOpt2007(mResolZBuf,"ResZBuf","Resolution of ZBuffer", {eTA2007::HDV})
	   */
   ;
}



cAppliRadiom2ImageSameMod::cAppliRadiom2ImageSameMod(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli               (aVArgs,aSpec),
    mShow                      (true),
    mNbMinByClpe               (50),
    mNbDegRad                  (5),
    mPhProj                    (*this),
    mSolRadial                 (1),
    mFusIndex                  (1),
    mCalcEqRad                 (nullptr),
    mCurSys                    (nullptr)
{
}

        //  =================================================
        //               Linear  correction 
        //  =================================================

void cAppliRadiom2ImageSameMod::MakeLinearCpleIm(size_t aK1,size_t aK2)
{
      std::vector<cPt2di> aVCpleI;
      mVRadIm[aK1]->IRD().GetIndexCommon(aVCpleI,mVRadIm[aK2]->IRD());
      if (aVCpleI.size() < mNbMinByClpe)
         return;
      const cImageRadiomData::tVRadiom & aVRad1 = mVRadIm[aK1]->IRD().VRadiom(0);
      const cImageRadiomData::tVRadiom & aVRad2 = mVRadIm[aK2]->IRD().VRadiom(0);

      // compute the median of ration as a robust estimator
      std::vector<double>  aVRatio;
      for (const auto & aCpl : aVCpleI)
      {
          cImageRadiomData::tRadiom aR1 = aVRad1.at(aCpl.x());
          cImageRadiomData::tRadiom aR2 = aVRad2.at(aCpl.y());

	  double aRatio =  aR1/std::max(1.0,double(aR2));
	  aVRatio.push_back(aRatio);

      }
      double aRMed = NC_KthVal(aVRatio,0.5);

      //  Use sqrt of ratio to have a more symetric equation
      //  aRatio = R1/R2   R1-R2 Ratio = 0
      //    R1/Sqrt(R) -R2*Sqrt(R) = 0
      double aSqrR = std::sqrt(aRMed);

      tElSys  aWeight = std::sqrt(aVCpleI.size()); // a bit (a lot ?) arbitrary
      mSomWLinear += aWeight;						   

      cSparseVect<tElSys>  aSV;
      aSV.AddIV(aK1,1.0/aSqrR);
      aSV.AddIV(aK2,-aSqrR);
      mCurSys->AddObservationLinear(aWeight,aSV,0.0);


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

void cAppliRadiom2ImageSameMod::MakeLinearModel()
{
    mSetInterv = new cSetInterUK_MultipeObj<tElSys> ;
    //for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    for (auto & aPtrIm : mVRadIm)
	mSetInterv->AddOneObj(aPtrIm);

    mCurSys = new  cResolSysNonLinear<tElSys>(eModeSSR::eSSR_LsqDense,mSetInterv->GetVUnKnowns());

    mSomWLinear = 0.0;
    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        if ((aKIm1%20)==0)
           StdOut()  << "Linear Still " << mNbIm - aKIm1 << "\n";
        for (size_t aKIm2=aKIm1+1 ; aKIm2<mNbIm; aKIm2++)
        {
            MakeLinearCpleIm(aKIm1,aKIm2);
        }
    }

    // Add an equation that fix  Avg(Ratio) = 1
    cDenseVect<tElSys>  aVecAll1 = cDenseVect<tElSys>::Cste( mNbIm,1.0);
    mCurSys->AddObservationLinear(mSomWLinear/1e2,aVecAll1,tElSys(mNbIm));

    cDenseVect<tElSys> aSol = mCurSys->SolveUpdateReset();

    aSol.SetAvg(1.0);
    mSetInterv->SetVUnKnowns(aSol);

    StdOut() << " 111 - AVG LINEAR= " << aSol.AvgElem()-1 << "\n";

    delete mSetInterv;
    mSetInterv = nullptr;
    delete mCurSys;
    mCurSys = nullptr;
}

        //  =================================================
        //               Mixt correction
        //  =================================================

void   cAppliRadiom2ImageSameMod::MakeOneIterMixtModel(int aKIter)
{
     // 0-Eventually froze linear
     bool isFrozenLinear = (aKIter<2);
     if (isFrozenLinear)
     {
         for (auto & aPtrIm : mVRadIm)
             mCurSys->SetFrozenAllCurrentValues(*aPtrIm);
     }
     else
        mCurSys->UnfrozeAll();

     // 1- Compute sigma
     cWeightAv<tElSys>  aAvgDev;
     cWeightAv<tElSys>  aAvgCRS;  ///  Correction radial signed
     cWeightAv<tElSys>  aAvgCRA;  ///  Correction radial abs
     std::vector<tElSys> aVSigma;
     for (const auto & aVecIndMul : mFusIndex.VVIndexes())
     {
         cUB_ComputeStdDev<1>  aDev;
         for (const  auto & aImInd : aVecIndMul)
         {
             cComputeCalibRadIma * aRIM = mVRadIm.at(aImInd.x());
	     int aInd = aImInd.y();

	     const cPt2df & aPt = aRIM->IRD().Pt(aInd);
	     tREAL8    aRadCor =   aRIM->CorrectedRadiom(aRIM->IRD().Gray(aInd),aPt);

             aDev.Add(&aRadCor,1.0);

	     tElSys aCor =  aRIM->ComputeCalSens().FactCorrecRadiom(aPt);
	     aAvgCRS.Add(1.0,aCor-1.0);
	     aAvgCRA.Add(1.0,std::abs(aCor-1.0));
         }
	 double aVar = sqrt(*(aDev.ComputeUnBiasedVar()));
	 aAvgDev.Add(1.0,aVar);
         aVSigma.push_back(aVar);
     }

     tElSys aSigma = Cst_KthVal(aVSigma,0.75);

     tElSys  aSomWRad = 0.0;

     // 2- Add equations
     for (size_t aKPMul=0; aKPMul<mFusIndex.VVIndexes().size(); aKPMul++)
     {
          const auto & aVecIndMul = mFusIndex.VVIndexes().at(aKPMul);
         // 2.1 - Compute "albedo" = average of corrected radiom
         tElSys aAlbedo = 0;
         for (const  auto & aImInd : aVecIndMul)
         {
              cComputeCalibRadIma * aRIM = mVRadIm.at(aImInd.x());
              int aInd = aImInd.y();
              const cPt2df & aPt =  aRIM->IRD().Pt(aInd);
              aAlbedo +=  aRIM->CorrectedRadiom(aRIM->IRD().Gray(aInd),aPt);
         }
         aAlbedo /= aVecIndMul.size();

	 tElSys  aW =  1.0 / sqrt(1.0 + Square(aVSigma[aKPMul]/aSigma));
	 cResidualWeighter<tElSys> aRW(aW);

         // 2.2 -  Add observation
         cSetIORSNL_SameTmp<tElSys>  aSetTmp({aAlbedo});
         for (const  auto & aImInd : aVecIndMul)
         {
             cComputeCalibRadIma * aRIM = mVRadIm.at(aImInd.x());
             int aInd = aImInd.y();
             const cPt2df & aPt =  aRIM->IRD().Pt(aInd);
             tElSys aRho2 =  aRIM->ComputeCalSens().NormalizedRho2(aPt);
             tElSys aRadIm = aRIM->IRD().Gray(aInd);

             std::vector<tElSys> aVObs({aRadIm,aRho2});
             std::vector<int>  aVInd({-1});
             aRIM->PushIndexes(aVInd);
             aRIM->ComputeCalSens().PushIndexes(aVInd);

             mCurSys->AddEq2Subst(aSetTmp,mCalcEqRad,aVInd,aVObs,aRW);

	     aSomWRad += aW;
         }
         mCurSys->AddObsWithTmpUK(aSetTmp);
     }
    // 3/Add an equation that fix  Avg(Ratio) = 1
    if (! isFrozenLinear)
    {
       cDenseVect<tElSys>  aVecAll1 = cDenseVect<tElSys>::Cste(mCurSys->NbVar() ,0.0);
       for (auto & aPtrIm : mVRadIm)
       {
           for (int aKInd=aPtrIm->IndUk0() ; aKInd<aPtrIm->IndUk1() ; aKInd++)
               aVecAll1(aKInd) = 1.0;
       }
       mCurSys->AddObservationLinear(aSomWRad/1e2,aVecAll1,tElSys(mNbIm));
    }

    cDenseVect<tElSys> aSol = mCurSys->SolveUpdateReset();
    mSetInterv->SetVUnKnowns(aSol);

    tElSys  aSomDiv=0;
    for (auto & aPtrIm : mVRadIm)
        aSomDiv += aPtrIm->DivIm();

     tElSys aDev = aAvgDev.Average() ;
     StdOut() << " DEV=" <<  aDev  << " S[75%]=" << aSigma << " AvgDiv=" << (aSomDiv/mNbIm -1) << "\n";
     StdOut() << " Avg RadCor, signed=" << aAvgCRS.Average() << " Abs="  <<  aAvgCRA.Average() << "\n";

     for (auto & aPair : mMapCal2Im)
     {
         StdOut() <<  "COEFF=" << aPair.second->RCRS().CoeffRad() << "\n";

	 cPt2dr aMil = ToR(aPair.first.first->SzPix()/2);
	 for (int aK=0 ; aK<=10 ; aK++)
		 StdOut() << " " << (aPair.second->RCRS().FlatField(aMil*(1-aK/10.0)) -1.0) ;
         StdOut() << "\n";
     }

     StdOut() << "\n";
}

void   cAppliRadiom2ImageSameMod::MakeMixtModel()
{
    mSetInterv = new cSetInterUK_MultipeObj<tElSys> ;
    mCalcEqRad = EqRadiomVignettageLinear(5,true,1);

    for (auto & aPtrIm : mVRadIm)
	mSetInterv->AddOneObj(aPtrIm);

    for (auto & aPair : mMapCal2Im)
	mSetInterv->AddOneObj(aPair.second);

    mCurSys = new  cResolSysNonLinear<tElSys>(eModeSSR::eSSR_LsqDense,mSetInterv->GetVUnKnowns());

    for (int aK=0 ; aK<10 ; aK++)
    {
        MakeOneIterMixtModel(aK);
    }

    delete mSetInterv;
    mSetInterv = nullptr;
 
    delete mCurSys;
    mCurSys = nullptr;
}

        //  =================================================
        //     Exe  ->   global call
        //  =================================================


int cAppliRadiom2ImageSameMod::Exe()
{
    mPhProj.FinishInit();

    size_t aLimitIndex =0;
    mNbIm = VectMainSet(0).size();
    for (size_t aKIm1=0 ; aKIm1<mNbIm; aKIm1++)
    {
        if ((aKIm1%20)==0)
           StdOut()  << "Reading, Still " << mNbIm - aKIm1 << "\n";
        std::string aNameIm = VectMainSet(0).at(aKIm1);
        cImageRadiomData* aIRD = mPhProj.AllocRadiomData(aNameIm);
        if (aIRD->VIndex().size() >10)
        {
            cPerspCamIntrCalib* aCalib = mPhProj.AllocCalib(aNameIm);
            UpdateMax(aLimitIndex,aIRD->LimitIndex());

            cMetaDataImage aMetaData =  mPhProj.GetMetaData(aNameIm);


            int aDelta= 0;
            if (false) // activate this if, for test, we want to generate multi calib
            {
               static int aCpt=0; aCpt++;
               int aDelta = (aCpt%4);
               StdOut() << "DDDD "<< aDelta << "\n";
            }
	    std::string aNameCalRad = mPhProj.NameCalibRadiomSensor(*aCalib,aMetaData);


            tIdCalRad aIdCalRad{aCalib,aMetaData.Aperture()+aDelta};
	    if (mMapCal2Im[aIdCalRad]==nullptr)
            {
                mMapCal2Im[aIdCalRad] = new cComputeCalibRadSensor(aCalib,mNbDegRad,aNameCalRad);
            }

	    cComputeCalibRadIma* aRadI = new cComputeCalibRadIma(aNameIm,aIRD,mMapCal2Im[aIdCalRad]);
            // aRadI->mCalRad =  mMapCal2Im[aCalib];
	    mVRadIm.push_back(aRadI);
        }
        else
           delete aIRD;
    }

    mNbIm  = mVRadIm.size();

    mFusIndex.Resize(aLimitIndex);
    for (size_t aKIm=0 ; aKIm<mNbIm ; aKIm++)
        mFusIndex.AddIndex(aKIm,mVRadIm[aKIm]->IRD().VIndex());
    mFusIndex.FilterSzMin(2);
    // ----------------  Linear ----------------------

    MakeLinearModel();

    

    // ----------------  Mixte ----------------------

    MakeMixtModel();

    // ----------------  Save ----------------------

    for (auto & aRadIm : mVRadIm)
    {
        mPhProj.SaveCalibRad(aRadIm->CalibRabIma());
    }

    // ----------------  Delete ----------------------
    for (auto & aRadIm : mVRadIm)
    {
        delete aRadIm;
    }
    for (auto & aPair: mMapCal2Im)
       delete aPair.second;

    delete mCalcEqRad;
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
     "RadiomComputeEqual",
      Alloc_Radiom2ImageSameMod,
      "Estimate radiometric model (test 4 now)",
      {eApF::Radiometry},
      {eApDT::Radiom},
      {eApDT::Radiom},
      __FILE__
);

};
