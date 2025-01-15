#include "MMVII_PCSens.h"
#include "MMVII_Interpolators.h"

/**
   \file PerturbRandOrient.cpp

   \brief file for generating random permutation
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_PerturbRandomOri                    */
   /*                                                            */
   /* ********************************************************** */

class cAppli_PerturbRandomOri : public cMMVII_Appli
{
     public :
        typedef cIm2D<tU_INT1>  tIm;

        cAppli_PerturbRandomOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;

        void TestPly();
     private :

	cPhotogrammetricProject    mPhProj;
	std::string                mSpecIm;
        tREAL8                     mRandOri;
        tREAL8                     mRandC;
        std::string                mPlyTest;
        cTriangulation3D<tREAL4> * mTri;
        std::vector<cSensorImage *> mVSI;
        std::vector<tIm>            mVIm;

};

cAppli_PerturbRandomOri::cAppli_PerturbRandomOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mTri          (nullptr)
{
}

cCollecSpecArg2007 & cAppli_PerturbRandomOri::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mSpecIm ,"Name of Input File",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPOrient().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_PerturbRandomOri::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return      anArgObl
            << AOpt2007(mRandOri,"RandOri","Random perturbation on orientations")
            << AOpt2007(mRandC  ,"RandC"  ,"Random perturbation on center")
            << AOpt2007(mPlyTest  ,"PlyTest"  ,"Test ply (temporary)")
    ;
}

// template <class Type> void cTriangulation3D<Type>::PlyInit(const std::string & aNameFile)

void cAppli_PerturbRandomOri::TestPly()
{
    if (! mTri)
       return;

    if (mVIm.empty())
    {
        for (auto aSI : mVSI)
        {
             mVIm.push_back(tIm::FromFile(aSI->NameImage()));
        }
    }

    cStdStatRes aStatRes;
    cCubicInterpolator aInterpol(-0.5);

    for (size_t aKP=0 ; aKP<mTri->NbPts() ; aKP++)
    {
        cPt3df aPF  = mTri->KthPts(aKP);
        cPt3dr aPGround(aPF.x(),aPF.y(),aPF.z());

        cComputeStdDev<tREAL8>  aStdDev;
        int aNbOk=0;
        for (size_t aKIm=0 ; aKIm<mVSI.size() ; aKIm++)
        {
            if (mVSI[aKIm]->IsVisible(aPGround))
            {
                cDataIm2D<tU_INT1> & aDIm = mVIm[aKIm].DIm();
                cPt2dr aPIm = mVSI[aKIm]->Ground2Image(aPGround);
                if (aDIm.InsideInterpolator(aInterpol,aPIm))
                {
                    aStdDev.Add(aDIm.GetValueInterpol(aInterpol,aPIm));
                    aNbOk++;
                }
            }
        }
        if (aNbOk>=2)
        {
            tREAL8 aDev = aStdDev.StdDev(1e-5);
            aStatRes.Add( (aDev *aNbOk) / (aNbOk-1));
        }
    }
    StdOut()  << " Avg=" <<  aStatRes.Avg() 
              << " Med=" << aStatRes.ErrAtProp(0.5)   
              << " P80=" << aStatRes.ErrAtProp(0.8) 
              << " P95=" << aStatRes.ErrAtProp(0.95) 
              << "\n";
}

int cAppli_PerturbRandomOri::Exe()
{
    mPhProj.FinishInit();

    if (IsInit(&mPlyTest))
    {
         mTri = new  cTriangulation3D<tREAL4>(mPlyTest);
         StdOut()  << "TRIII, NbPts=" << mTri->NbPts() << " NbF=" << mTri->NbFace() << "\n";
    }


    for (const auto & aNameIm : VectMainSet(0))
    {
        cSensorImage* aSI = mPhProj.ReadSensor(aNameIm,true,false);
        mVSI.push_back(aSI);
    }

    TestPly();

    for (auto aSI : mVSI)
    {
        if (IsInit(&mRandOri))
        {
            cSensorCamPC * aCamPC = aSI->UserGetSensorCamPC();
            aCamPC->SetOrient( aCamPC->Orient() * tRotR::RandomRot(mRandOri) );
        }
        if (IsInit(&mRandC))
        {
            cSensorCamPC * aCamPC = aSI->UserGetSensorCamPC();
            aCamPC->SetCenter(aCamPC->Center() + cPt3dr::PRandC() * mRandC);
        }
        
        mPhProj.SaveSensor(*aSI);
    }
    TestPly();

    delete mTri;
    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_PerturbRandomOri::Samples() const
{
   return {"NO SAMPLES FOR NOW"};
}



tMMVII_UnikPApli Alloc_PerturbRandomOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_PerturbRandomOri(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_PerturbRandomOri
(
     "SimulOriPerturbRandom",
      Alloc_PerturbRandomOri,
      "Perturbate random de orientation (for simulations)",
      {eApF::Ori,eApF::Simul},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);

}; // MMVII

