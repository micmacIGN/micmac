#include "MMVII_TplHeap.h"

#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_DeclareAllCmd.h"

namespace MMVII
{

class cMemoryInterfImportHom : public cInterfImportHom
{
      public :
           void GetHom(cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2) const override;
           bool HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const  override;

           void Add(const cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2)     ;
      private :
             std::map<tSS,cSetHomogCpleIm> mMapN2Cple;
};


bool cMemoryInterfImportHom::HasHom(const std::string & aN1,const std::string & aN2) const
{
    //return BoolFind(mMapN2Cple,tSS(aN1,aN2));

    return mMapN2Cple.find(tSS(aN1,aN2)) != mMapN2Cple.end();
}


void cMemoryInterfImportHom::GetHom(cSetHomogCpleIm & aCple,const std::string & aN1,const std::string & aN2) const
{
    const auto & anIter = mMapN2Cple.find(tSS(aN1,aN2));
    aCple = anIter->second;
}

void cMemoryInterfImportHom::Add(const cSetHomogCpleIm & aCple,const std::string & aN1,const std::string & aN2)
{
     mMapN2Cple[tSS(aN1,aN2)] = aCple;
}


class cOriTriplets
{
   public :
            cOriTriplets(std::vector<std::string>, const cPhotogrammetricProject &);

            ~cOriTriplets();
   private :

            std::vector<std::string>           mVNames;
            const cPhotogrammetricProject&     mPhProj;
            std::vector<cPerspCamIntrCalib*>   mVCalibs;
            cComputeMergeMulTieP *             mTiepMul;


};

cOriTriplets::cOriTriplets(std::vector<std::string> aVNames, const cPhotogrammetricProject & aPhProj) :
    mVNames (aVNames),
    mPhProj (aPhProj)
{
    for (const auto & aName :  mVNames)
    {
        mVCalibs.push_back(mPhProj.InternalCalibFromImage(aName));
    }

    cMemoryInterfImportHom aMIIH;

    for (size_t aK1=0 ; aK1< mVNames.size() ; aK1++)
    {
        const std::string & aN1 = mVNames.at(aK1);
        for (size_t aK2=aK1+1 ; aK2<mVNames.size() ; aK2++)
        {
            const std::string & aN2 = mVNames.at(aK2);
            cSetHomogCpleIm aSet;
            int aNbInit;
            mPhProj.ReadHomolMultiSrce(aNbInit,aSet,aN1,aN2);
            aMIIH.Add(aSet,aN1,aN2);
        }
    }

    mTiepMul = new cComputeMergeMulTieP(aVNames,&aMIIH);



  //  mPhProj
  //  for ()
  //  void ReadHomolMultiSrce(int & aNbInit,cSetHomogCpleIm &,const std::string & aNI1,const std::string & aNI2);

}


cOriTriplets::~cOriTriplets()
{
    delete mTiepMul;
}

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriRel2Im                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_OriRelTripletsOfIm : public cMMVII_Appli
{
     public :

        static const int RESULT_NO_POSE = 2007;

        typedef cIsometry3D<tREAL8>  tPose;

        cAppli_OriRelTripletsOfIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,int aMode);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
         std::vector<std::string>  Samples() const override;

     private :


        int                       mModeCompute;
        cPhotogrammetricProject   mPhProj;
        std::string               mIm1;
        std::string               mIm2;
        std::string               mIm3;

        cPerspCamIntrCalib        * mCalib1;
        cPerspCamIntrCalib        * mCalib2;
        cPerspCamIntrCalib        * mCalib3;


};

cAppli_OriRelTripletsOfIm::cAppli_OriRelTripletsOfIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,int aMode) :
    cMMVII_Appli  (aVArgs,aSpec),
    mModeCompute  (aMode),
    mPhProj       (*this),
    mCalib1       (nullptr),
    mCalib2       (nullptr),
    mCalib3       (nullptr)
{
}

// mEstimatePose

cCollecSpecArg2007 & cAppli_OriRelTripletsOfIm::ArgObl(cCollecSpecArg2007 & anArgObl)
{
     if (mModeCompute==0)
     {
         anArgObl
              << Arg2007(mIm1,"name first image",{eTA2007::FileImage})
              << Arg2007(mIm2,"name second image",{eTA2007::FileImage})
              << Arg2007(mIm3,"name second image",{eTA2007::FileImage})

         ;
     }
     if (mModeCompute==1)
     {
         anArgObl << Arg2007(mIm1,"name first image",{eTA2007::FileImage})
                  << mPhProj.DPOriRel().ArgDirInMand()
          ;
     }
     if (mModeCompute==2)
     {
         anArgObl    << mPhProj.DPOriRel().ArgDirInMand() ;
     }

     anArgObl <<  mPhProj.DPOrient().ArgDirInMand("Input orientation for calibration")  ;


     return anArgObl;
}

cCollecSpecArg2007 & cAppli_OriRelTripletsOfIm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return       anArgOpt
            <<  mPhProj.DPTieP().ArgDirInOpt()
            <<  mPhProj.DPGndPt2D().ArgDirInOpt()
            <<  mPhProj.DPMulTieP().ArgDirInOpt()
           ;
}


int cAppli_OriRelTripletsOfIm::Exe()
{
   // mTimeSegm = mShow ? new cTimerSegm(this) : nullptr ;
    mPhProj.FinishInit();




  //  delete mTimeSegm;
    return EXIT_SUCCESS;
}




/*
void        cAppli_OriRelTripletsOfIm::EstimatePose2IM(const std::string& aIm1,const std::string& aIm2)
{
    mIm1 = aIm1;
    mIm2 = aIm2;
    mIm3 = aIm3;

}
*/


/* ====================================================== */
/*               OriPoseEstimRel2Im                       */
/* ====================================================== */

tMMVII_UnikPApli Alloc_OriRel3Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRelTripletsOfIm(aVArgs,aSpec,0));
}

cSpecMMVII_Appli  TheSpec_OriRel3Im
(
     "OriPoseEstimRel3Im",
      Alloc_OriRel3Im,
      "Estimate relative orientation of 3 images testing various different algorithms & configs",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);


}; // MMVII




