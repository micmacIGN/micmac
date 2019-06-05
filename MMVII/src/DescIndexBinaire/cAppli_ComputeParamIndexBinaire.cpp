#include "include/MMVII_all.h"
#include "IndexBinaire.h"
#include "include/MMVII_Tpl_Images.h"

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
         << AOpt2007(mSaveFileSelVectIR,"SaveFSVIR","File to save selected radial inv",{})
   ;
}



cAppli_ComputeParamIndexBinaire::cAppli_ComputeParamIndexBinaire(int argc,char** argv,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (argc,argv,aSpec),
  mPatPCar     ("eTPR_LaplMax"),
  mPatInvRad   ("eTVIR_ACG."),
  mNbPixTot    (0.0),
  mPropFile    (1.0),
  mNbValByP    (0),
  mTmpVect     (1),    // need to initialize, but do not know the size
  mStat2       (1),     // idem
  mEigen       (nullptr),
  mSaveFileSelVectIR    ("")
{
}

const std::string & cAppli_ComputeParamIndexBinaire::DirCurPC() const  {return mDirCurPC;}
double  cAppli_ComputeParamIndexBinaire::PropFile() const {return mPropFile;}
cVecInvRad*  cAppli_ComputeParamIndexBinaire::IR(int aK) { return  mVIR.at(aK).get(); }
cDenseVect<tREAL8> &    cAppli_ComputeParamIndexBinaire::TmpVect() {return mTmpVect;}
const cStrStat2<tREAL8> & cAppli_ComputeParamIndexBinaire::Stat2() { return mStat2;}

const cResulSymEigenValue<tREAL8> &  cAppli_ComputeParamIndexBinaire::Eigen() const
{
   MMVII_INTERNAL_ASSERT_medium(mEigen!=nullptr,"cAppli_ComputeParamIndexBinaire::Eigen()");
   return *mEigen;
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




void  cAppli_ComputeParamIndexBinaire::ProcessOneDir(const std::string & aDir)
{
    StdOut() << "================== " << mPatPCar << " ===========\n";
    mDirCurPC = mDirGlob + aDir + DirSeparator();

    cDataOneInvRad * aLast = nullptr;
    // For all type of invariant create a structure from the folder
    for (int aK=0 ; aK<int(mVecTyIR.size()) ; aK++)
    {
       // Creation will reas the fils
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
       mVIR.push_back(tPtVIR(new cVecInvRad(mNbValByP)));
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

   for (int aKIter=0 ; aKIter<3; aKIter++)
       ComputeIndexBinaire();
}

void  cAppli_ComputeParamIndexBinaire::ComputeIndexBinaire()
{
    mVecTrueP.clear();
    mVecFalseP.clear();
    mVVBool.clear();

    for (int aK=0 ; aK<mNbPts ; aK++)
    {
        if (     ((aK%2)==0) 
              && (mVIR[aK]->mSelected)
           )
           mVecTrueP.push_back(cPt2di(aK,aK+1));

        mVecFalseP.push_back(cPt2di(aK,RandUnif_N(mNbPts)));
        mVecFalseP.push_back(cPt2di(aK,RandUnif_N(mNbPts)));
    }

    // Allocate data for stats
    mTmpVect = cDenseVect<tREAL8>(mNbValByP);
    mStat2 = cStrStat2<tREAL8>(mNbValByP);

    // Fill the stats
    int aPerAff = 10000;
    for (int aK=0 ; aK< mNbPts ;aK++)
    {
        if ((aK+1)%aPerAff ==0)
        {
            StdOut() << "Fill cov, Remain " << (mNbPts-aK) << " on " << mNbPts << "\n";
        }
        if ( mVIR[aK]->mSelected)
        {
            CopyIn(mTmpVect.DIm(),mVIR.at(aK)->mVec.DIm());
            mStat2.Add(mTmpVect);
        }
    }
    mStat2.Normalise();
    mEigen = &mStat2.DoEigen();


    int aNbVal = 100;
    for (int aK=mNbValByP-1 ; aK>=std::max(0,mNbValByP-aNbVal); aK--)
    {
        StdOut() << "EV["<< aK<< "]=" <<  mEigen->EigenValues()(aK) << "\n";
        mVVBool.push_back(tPtVBool ( new cVecBool (new cIB_LinearFoncBool(*this,aK,0), mVIR)));
    }

    TestRandom(500,80);

    int mTestSelBit = 3;
    int aNbOk =0;
    for (int aK=0 ; aK<mNbPts ; aK+=2)
    {
        if (NbbBitDif(mBestVB,cPt2di(aK,aK+1))<=mTestSelBit)
        {
            mVIR.at(aK  )->mSelected = false;
            mVIR.at(aK+1)->mSelected = false;
        }
        if (!mVIR.at(aK  )->mSelected)
           aNbOk += 2;
    }
    StdOut() << "Prop Deteteced " << aNbOk / double(mNbPts) << "\n";

    // TestNbBit();
    // SaveFileData();
}

void cAppli_ComputeParamIndexBinaire::SaveFileData()
{
}

std::vector<const cVecBool*> cAppli_ComputeParamIndexBinaire::IndexToVB(const std::vector<int>& aSet) const
{
   std::vector<const cVecBool*> aRes;
   for (const auto & aI : aSet)
       aRes.push_back(mVVBool.at(aI).get());
   return aRes;
}


std::vector<const cVecBool*> 
    cAppli_ComputeParamIndexBinaire::GenereVB(std::vector<int> & aSet,int aK,int aNb) const
{
   aSet = RandSet(aK,aNb);
   return IndexToVB(aSet);
}

void cAppli_ComputeParamIndexBinaire::TestNewSol
     (
           const  std::vector<int> & aVI,
           const  std::vector<const cVecBool*> &aNewVB
     )
{
   cStatDifBits aStTrue(mVecTrueP,aNewVB);
   cStatDifBits aStFalse(mVecFalseP,aNewVB);

   int aKMax;
   double aSc = aStTrue.Score(aStFalse,100.0,aKMax);
   if (aSc>mBestSc)
   {
       mBestSc = aSc;
       mBestVB    = aNewVB;
       mBestIndex = aVI;

       StdOut() << "========== Score : " << aSc  << "\n";
       aStTrue.Show(aStFalse,0,aKMax+3);
   }
}


void cAppli_ComputeParamIndexBinaire::TestRandom(int aNbTest,int aNb) 
{
   mBestSc=-1e10;
   aNb = std::min(aNb,int(mVVBool.size()));
   int aCpt=0;
   int aKB=24;
   int aNb0 = aNb;
   double aT0= SecFromT0();
   while (aCpt < aNbTest)
   {
        aNb0 = aKB + aCpt% (aNb-aKB);
        std::vector<int> aNewIndex;
        std::vector<const cVecBool*> aNewVB = GenereVB(aNewIndex,aKB,aNb0);

        TestNewSol(aNewIndex,aNewVB);
        aCpt++;
        if (aCpt%10==0) 
        {
           MMVII::StdOut() << "Random Init " << aCpt 
                           << " AverTim=" << (SecFromT0()-aT0)/aCpt 
                           << "\n";
        }
   }

   int aNbIter = 800;
   for (int aKC=0 ; aKC< aNbIter ; aKC++)
   {
        StdOut() << "Chnage cphase " << aNbIter - aKC  << "\n";

        int aPer = 7;
        int aDelta = mVVBool.size() -aKB;
        int aNbMax = aKB + (aDelta * Square(2+aKC%aPer)) / Square(aPer+1);
        
        for (int aNbC=1 ; aNbC<=4 ; aNbC++)
        {
             std::vector<int>  aNewIndex = RandNeighSet(aNbC,aNbMax,mBestIndex);
             std::vector<const cVecBool*>  aNewVB = IndexToVB(aNewIndex);
             TestNewSol(aNewIndex,aNewVB);
        }
   }
}


void cAppli_ComputeParamIndexBinaire::TestNbBit() const
{
    for (int aNb=16 ; aNb<=24 ; aNb+=2)
    {
       TestNbBit(aNb);
    }

    for (int aNb=30 ; aNb<=100 ; aNb+=10)
       TestNbBit(aNb);
}

void cAppli_ComputeParamIndexBinaire::TestNbBit(int aNb) const
{
    std::vector<int>             aIndexRes;
    std::vector<const cVecBool*> aVVB = GenereVB(aIndexRes,aNb,aNb);
/*
    std::vector<const cVecBool*> aVVB ;
    for (int aK=0 ; aK<aNb ; aK++)
    {
         aVVB.push_back(mVVBool.at(aK).get());
    }
*/
    cStatDifBits aStTrue(mVecTrueP,aVVB);
    cStatDifBits aStFalse(mVecFalseP,aVVB);

    MMVII::StdOut() << "================= " << aNb << "==============\n";
    for (int aK =0 ; aK <= aNb ; aK++)
        MMVII::StdOut() << aK << " " << aStTrue.mStatRCum[aK] << " " << aStFalse.mStatRCum[aK]<< "\n";
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

