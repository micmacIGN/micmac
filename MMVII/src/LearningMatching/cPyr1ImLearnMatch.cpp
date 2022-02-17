#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"



namespace MMVII
{


cPyr1ImLearnMatch::cPyr1ImLearnMatch
(
      const cBox2di &       aBox,
      const cBox2di &       aBoxOut,
      const std::string &   aName,
      const cAppliLearningMatch & anAppli,
      const cFilterPCar&    aFPC,
      bool  initRand
) :
    mBox    (aBox),
    mNameIm (aName),
    mAppli  (anAppli),
    mGP     (mBox.Sz(),mAppli.NbOct(),mAppli.NbLevByOct(),mAppli.NbOverLapByO(),&mAppli,false),
    mPyr    (nullptr),
    mImF    (cPt2di(1,1))
{
    mGP.mFPC = aFPC;
    mPyr =  tPyr::Alloc(mGP,mNameIm,mBox,aBoxOut);
    if (initRand)
        mPyr->ImTop().DIm().InitRandom(0.0,100.0);
    else
        mPyr->ImTop().Read(cDataFileIm2D::Create(mNameIm,true),mBox.P0());

    mPyr->ComputGaussianFilter();

    // Compute the filtered images used for having "invariant" gray level
        // Filter to have a local average
    mImF =  mPyr->ImTop().Dup();
    float aFact = 50.0;
    ExpFilterOfStdDev(mImF.DIm(),5,aFact);

       //   make a ratio image
    {
        tDataImF &aDIF = mImF.DIm();
        tDataImF &aDI0 =  mPyr->ImTop().DIm();
        for (const auto & aP : aDIF)
        {
            aDIF.SetV(aP,(1+NormalisedRatioPos(aDI0.GetV(aP),aDIF.GetV(aP))) / 2.0);
        }
    }
}
void cPyr1ImLearnMatch::SaveImFiltered() const
{
   std::string  aName = "FILTRED-" + mNameIm;
   const tDataImF &aDIF = mImF.DIm();
   cIm2D<tU_INT1> aImS(aDIF.Sz());
   for (const auto & aP : aDIF)
   {
       int aVal = round_ni(aDIF.GetV(aP)*255.0);
       aImS.DIm().SetV(aP,aVal);
   }
   aImS.DIm().ToFile(aName); //  Ok
}

double cPyr1ImLearnMatch::MulScale() const  {return mPyr->MulScale();}
const cPyr1ImLearnMatch::tDataImF &  cPyr1ImLearnMatch::ImInit() const {return mPyr->ImTop().DIm();}
const cPyr1ImLearnMatch::tDataImF &  cPyr1ImLearnMatch::ImFiltered() const {return mImF.DIm();}
          // const tDataImF &  ImFiltered() const;


bool  cPyr1ImLearnMatch::CalculAimeDesc(const cPt2dr & aPt)
{
    {
       cPt2dr aSzV(mAppli.SzMaxStdNeigh(),mAppli.SzMaxStdNeigh());
       cPt2dr aP1 = aPt - aSzV;
       cPt2dr aP2 = aPt + aSzV;
       tDataImF & aImPyr = mPyr->ImTop().DIm();
       if (!(aImPyr.InsideBL(aP1) && aImPyr.InsideBL(aP2)))
          return false;
       tDataImF & aDImF = mImF.DIm();
       if (!(aDImF.InsideBL(aP1) && aDImF.InsideBL(aP2)))
          return false;
    }
    
    cProtoAimeTieP<tREAL4> aPAT(mPyr->GPImTop(),aPt);

    if (! aPAT.FillAPC(mGP.mFPC,mPC,true))
       return false;


    aPAT.FillAPC(mGP.mFPC,mPC,false);

    return true;
}
cAimePCar   cPyr1ImLearnMatch::DupLPIm() const { return mPC.DupLPIm(); }


};

