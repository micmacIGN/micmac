#include "include/MMVII_all.h"
#include "IndexBinaire.h"
// #include "include/MMVII_2Include_Serial_Tpl.h"
// #include<map>

/** \file cCalcul_IndexBinaire.cpp
    \brief Command for computing the parameters of 

*/


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*           cAppli_ComputeParamIndexBinaire            */
/*                                                      */
/* ==================================================== */

/**
    Application for computing optimized binary indexes
*/




cCollecSpecArg2007 & cAppli_ComputeParamIndexBinaire::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
          <<   Arg2007(mDirGlob,"Directory where data are loaded",{})
   ;
}

cCollecSpecArg2007 & cAppli_ComputeParamIndexBinaire::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mPatPCar,"PatPCar","Pattern for P Carac for ex \"eLap.*\"",{})
         << AOpt2007(mPatInvRad,"PatIR","Pattern of radial invariants",{})
         << AOpt2007(mPropFile,"PropF","Prop of File selected (tuning, faster in test)",{})
   ;
}



cAppli_ComputeParamIndexBinaire::cAppli_ComputeParamIndexBinaire(int argc,char** argv,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (argc,argv,aSpec),
  mPatPCar     ("eTPR_LaplMax"),
  mPatInvRad   ("eTVIR_ACG."),
  mNbPixTot    (0.0),
  mPropFile    (1.0),
  mNbValByP    (0),
  mTmpVect     (1),           // need to initialize, but do not know the size
  mMoyVect     (1),           // need to initialize, but do not know the size
  mCovVect     (1,1)  // idem
{
}



int cAppli_ComputeParamIndexBinaire::Exe()
{
   // Transform pattern in of enum radial  invariant
   mVecTyIR = SubOfPat<eTyInvRad>(mPatInvRad,false);
  // Extract Pts carac corresponding to pattern
   mLDirPC = GetSubDirFromDir(mDirGlob,BoostAllocRegex(mPatPCar));


   // If only one p carac process it
   if (mLDirPC.size() ==1)
   {
       ProcessOneDir(mLDirPC[0]);
   }
   // If only one p carac process it
   else if (mLDirPC.size() > 1)
   {
       ExeMultiAutoRecallMMVII ("PatPCar",mLDirPC,cColStrAOpt(),true);
   }
   else
   {
       MMVII_UsersErrror(eTyUEr::eEmptyPattern,"No directory of Pt carac found");
   }

   return EXIT_SUCCESS;
}

const std::string & cAppli_ComputeParamIndexBinaire::DirCurPC() const  {return mDirCurPC;}
double  cAppli_ComputeParamIndexBinaire::PropFile() const {return mPropFile;}



void  cAppli_ComputeParamIndexBinaire::ProcessOneDir(const std::string & aDir)
{
    StdOut() << "================== " << mPatPCar << " ===========\n";
    mDirCurPC = mDirGlob + aDir + DirSeparator();

    cDataOneInvRad * aLast = nullptr;
    for (int aK=0 ; aK<int(mVecTyIR.size()) ; aK++)
    {
       cDataOneInvRad * aNew = new cDataOneInvRad (*this,aLast,mVecTyIR[aK]);
       // mVecDIR.push_back(std::unique_ptr<cDataOneInvRad> (new cDataOneInvRad (*this,mVecTyIR[aK])));
       mVecDIR.push_back(std::unique_ptr<cDataOneInvRad> (aNew));
       mNbPixTot += mVecDIR.back()->NbPixTot();
       mNbValByP += mVecDIR.back()->NbValByP();
       aLast = aNew;
    }

    // Check coher of any relatively to the first one
    for (int aK=1 ; aK<int(mVecDIR.size()) ; aK++)
    {
       mVecDIR[0]->CheckCoherence(*(mVecDIR[aK]));
    }

    // Allocate the vector for car point
    mNbPts = mVecDIR[0]->NbPatch();
    mVIR.reserve(mNbPts);
    for (int aK=0 ; aK<mNbPts ; aK++)
    {
       mVIR.push_back(std::unique_ptr<cVecInvRad>(new cVecInvRad(mNbValByP)));
    }


    StdOut() 
           << " NbValByP= " << mNbValByP 
           << " NbPts= " << mNbPts 
           << " NbPixTot= " << mNbPixTot 
           << "\n";

   // Fill the vect with data from files
   for (auto & aDIR : mVecDIR)
   {
       aDIR->AddPCar();
   }

    // Allocate data for stats
    mTmpVect = cDenseVect<tREAL4>(mNbValByP);
    mMoyVect = cDenseVect<tREAL8>(mNbValByP,eModeInitImage::eMIA_Null);
    mCovVect = cDenseMatrix<tREAL8>(mNbValByP,mNbValByP,eModeInitImage::eMIA_Null);


    // Fill the stats
    int aPerAff = 10000;
    for (int aK=0 ; aK< mNbPts ;aK++)
    {
        if ((aK+1)%aPerAff ==0)
        {
            StdOut() << "Fill cov, Remain " << (mNbPts-aK) << " on " << mNbPts << "\n";
        }
        mVIR.at(aK)->Add2Stat(mTmpVect,mMoyVect,mCovVect);
    }
    cVecInvRad::PostStat(mMoyVect,mCovVect,mNbPts);

    //  Eigen value
    cResulSymEigenValue<double> aRSEV = mCovVect.SymEigenValue();

    for (int aK=0 ; aK<mNbValByP; aK++)
        StdOut() << "EV =" << aRSEV.EigenValues()(aK) << "\n";

    // mMoyVectI
    // 

/*
   std::vector<std::unique_ptr<cVecInvRad> > aVIR;
   aVIR.reserve(mNbPts);
   for (int aK=0 ; aK<mNbPts; aK++)
      aVIR.push_back(std::unique_ptr<cVecInvRad>(new cVecInvRad));
*/
}

cVecInvRad*  cAppli_ComputeParamIndexBinaire::IR(int aK)
{
   return  mVIR.at(aK).get();
}



     // =====================

tMMVII_UnikPApli Alloc_ComputeParamIndexBinaire(int argc,char ** argv,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ComputeParamIndexBinaire(argc,argv,aSpec));
}

cSpecMMVII_Appli  TheSpec_ComputeParamIndexBinaire
(
     "CompPIB",
      Alloc_ComputeParamIndexBinaire,
      "This command is used compute Parameter of Binary Index",
      {eApF::TieP},
      {eApDT::FileSys},
      {eApDT::Xml,eApDT::Console},
      __FILE__
);

};

