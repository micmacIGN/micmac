
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
#include "include/MMVII_Tpl_Images.h"
#include "include/MMVII_TplLayers3D.h"

/*
   C = (1-(1-L) ^2)

*/

namespace MMVII
{

namespace  cNS_FillCubeCost
{

class cAppliFillCubeCost;

struct cOneModele
{
    public :
        cOneModele
        (
            const std::string & aNameModele,
            cAppliFillCubeCost  & aAppliLearn
        );

	double ComputeCost(bool &Ok,const cPt2di & aPC1,const cPt2di & aPC2,int aDZ) const;
        void CalcCorrelExterneTerm(const cBox2di & aBoxInitIm1,int aPxMin,int aPxMax);
        void CalcCorrelExterneRecurs(const cBox2di & aBoxIm1);
        void CalcCorrelExterne();

	cAppliFillCubeCost  * mAppli;
	std::string           mNameModele;
        bool                  mWithIntCorr;
        bool                  mWithExtCorr;
	bool                  mWithStatModele;
        cHistoCarNDim         mModele;
        cPyr1ImLearnMatch *   mPyrL1;
        cPyr1ImLearnMatch *   mPyrL2;
        int                   mSzW;
        cPt2di                mPSzW;
};


static const std::string TheNameCorrel  = "MMVIICorrel";
static const std::string TheNameExtCorr = "ExternCorrel";


class cAppliFillCubeCost : public cAppliLearningMatch
{
     public :
        typedef tINT2                      tElemZ;
        typedef cIm2D<tElemZ>              tImZ;
        typedef cDataIm2D<tElemZ>          tDataImZ;
        typedef cIm2D<tREAL4>              tImRad;
        typedef cDataIm2D<tREAL4>          tDataImRad;
	typedef cLayer3D<float,tElemZ>     tLayerCor;
        typedef cIm2D<tREAL4>              tImPx;
        typedef cDataIm2D<tU_INT1>         tDataImMasq;
        typedef cDataIm2D<tREAL4>          tDataImPx;
        typedef cIm2D<tREAL4>              tImFiltred;
        typedef cDataIm2D<tREAL4>          tDataImF;
        typedef cGaussianPyramid<tREAL4>   tPyr;
        typedef std::shared_ptr<tPyr>      tSP_Pyr;
        typedef cPyr1ImLearnMatch *        tPtrPyr1ILM;

        cAppliFillCubeCost(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
	double ComputCorrel(const cPt2di & aPI1,const cPt2dr & aPI2,int mSzW) const;

	const tDataImRad & DI1() {return *mDI1;}
	const tDataImRad & DI2() {return *mDI2;}
        cPyr1ImLearnMatch * PyrL1 () {return PyrL(mPyrL1,mBoxGlob1,mNameI1);}
        cPyr1ImLearnMatch * PyrL2 () {return PyrL(mPyrL2,mBoxGlob2,mNameI2);}
	tREAL8     StepZ() const {return mStepZ;}
        bool Ok1(int aX) const {return Ok(aX,mVOk1);}
        bool Ok2(int aX) const {return Ok(aX,mVOk2);}
        const cAimePCar & PC1(int aX) const {return mVPC1.at(aX);}
        const cAimePCar & PC2(int aX) const {return mVPC2.at(aX);}
        const cBox2di  & BoxGlob1() const {return mBoxGlob1;}  ///< Accessor
        const cBox2di  & BoxGlob2() const {return mBoxGlob2;}  ///< Accessor
        const std::string   & NameI1() const {return mNameI1;}  ///< Accessor
        const std::string   & NameI2() const {return mNameI2;}  ///< Accessor

	cBox2di BoxFile1() const {return cDataFileIm2D::Create(mNameI1,false);}
	cBox2di BoxFile2() const {return cDataFileIm2D::Create(mNameI2,false);}

	int  SzW() const {return mSzW;}
	bool InterpolLearn() const {return mInterpolLearn;}
	double ExpLearn()    const {return mExpLearn; }
	double FactLearn()   const {return mFactLearn; }
        const cFilterPCar  & FPC() const {return mFPC;}  ///< Used to compute Pts

	const tImZ  & ImZMin() {return  mImZMin;}
	const tImZ  & ImZMax() {return  mImZMax;}
	void MakeNormalizedIm();
	// -------------- Internal variables -------------------
     private :

        cPyr1ImLearnMatch * PyrL (tPtrPyr1ILM & aPtrPyr,const cBox2di & aBoxI,const std::string & aNameIm)
	{
	   if (aPtrPyr==nullptr)
              aPtrPyr = new cPyr1ImLearnMatch(aBoxI,aBoxI,aNameIm,*this,mFPC,false);
	   return aPtrPyr;
	}



        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	void PushCost(double aCost);
        bool Ok(int aX,const std::vector<bool> &  aV) const
	{
            return (aX>=0) && (aX<int(aV.size())) && (aV.at(aX)) ;
	}

	void MakeLinePC(int aYLoc,bool Im1);
	// -------------- Mandatory args -------------------
	std::string   mNameI1;
	std::string   mNameI2;
	std::string   mNameModele;
	cPt2di        mP0Z;  // Pt corresponding in Im1 to (0,0)
	cBox2di       mBoxGlob1;  // Box to Load, taking into account siwe effect
	cBox2di       mBoxGlob2;
	std::string   mNamePost;

	// -------------- Optionnal args -------------------
	tREAL8        mStepZ;
	bool          mCmpCorLearn; //  Create a comparison between Correl & Learn
	bool          mInterpolLearn; // Use interpolation mode for learned cost
	double        mExpLearn; // Exposant to adapt learned cost
	double        mFactLearn; // Factor to adapt learned cost
	std::string   mNameCmpModele;
	int           mSzW;
	// -------------- Internal variables -------------------
	
	std::string StdName(const std::string & aPre,const std::string & aPost);

        int         mNbCmpCL;
	cIm2D<tREAL8>  mImCmp;
        std::string mNameZMin;
        std::string mNameZMax;
        std::string mNameCube;
        cMMVII_Ofs* mFileCube;

	tImZ        mImZMin;
	tImZ        mImZMax;
	tImRad      mIm1;
	tDataImRad  *mDI1;
	tImRad      mIm2;
	tDataImRad  *mDI2;

        // Normalized images, in radiometry, avearge 0, std dev 1.0
	tImRad      mImNorm1;
	tDataImRad  *mDINorm1;
	tImRad      mImNorm2;
	tDataImRad  *mDINorm2;
	tLayerCor   mLayerCor;

	double      ToCmpCost(double aCost) const;

        cPyr1ImLearnMatch * mPyrL1;
        cPyr1ImLearnMatch * mPyrL2;
        cFilterPCar  mFPC;  ///< Used to compute Pts
        std::vector<bool>         mVOk1;
        std::vector<cAimePCar>    mVPC1;
        std::vector<bool>         mVOk2;
        std::vector<cAimePCar>    mVPC2;
};


/* *************************************************** */
/*                                                     */
/*                   cOneModele                        */
/*                                                     */
/* *************************************************** */


cOneModele::cOneModele
(
    const std::string & aNameModele,
    cAppliFillCubeCost  & aAppliLearn
) :
   mAppli          (&aAppliLearn),
   mNameModele     (aNameModele),
   mWithIntCorr    (mNameModele==TheNameCorrel),
   mWithExtCorr    (mNameModele==TheNameExtCorr),
   mWithStatModele (! (mWithIntCorr || mWithExtCorr)),
   mPyrL1          (nullptr),
   mPyrL2          (nullptr),
   mSzW            (mAppli->SzW()),
   mPSzW           (mSzW,mSzW)
{
    if (mWithStatModele)
    {
       ReadFromFile(mModele,mNameModele);
       mPyrL1 = mAppli->PyrL1 ();
       mPyrL2 = mAppli->PyrL2 ();
    }
    else if (mWithExtCorr)
    {
         CalcCorrelExterne();
    }
}

void cOneModele::CalcCorrelExterneTerm(const cBox2di & aBoxInitIm1,int aPxMin,int aPxMax)
{
     // cBox2di aBoxDil = aBoxInitIm1.Inter
}

void cOneModele::CalcCorrelExterneRecurs(const cBox2di & aBoxIm1)
{
     cPt2di aP0Glob = mAppli->BoxGlob1().P0();
     int aMinPxMin = 1e6;
     int aMaxPxMax = -1e6;
     int aTotPx = 0;
     int aNbPx = 0;
     const cAppliFillCubeCost::tDataImZ & aDIZMin = mAppli->ImZMin().DIm();
     const cAppliFillCubeCost::tDataImZ & aDIZMax = mAppli->ImZMax().DIm();

     cRect2 aR2(aBoxIm1.P0(),aBoxIm1.P1());
     for (const auto & aP : aR2)
     {
          cPt2di aPLoc = aP-aP0Glob;
	  int aPxMin = aDIZMin.GetV(aPLoc);
	  int aPxMax = aDIZMax.GetV(aPLoc);
	  UpdateMin(aMinPxMin,aPxMin);
	  UpdateMax(aMaxPxMax,aPxMax);
	  aTotPx += aPxMax - aPxMin;
	  aNbPx++;
     }

     double aAvgPx = aTotPx / double(aNbPx);
     int   aIntervPx = (aMaxPxMax-aMinPxMin);

     bool   isTerminal = aIntervPx < 2 * aAvgPx;

     if ((aIntervPx*aIntervPx) > 1e8)
     {
        isTerminal = false;
     }

     {
        int aSzMin = MinAbsCoord(aBoxIm1.Sz());
	if (aSzMin<200)
           isTerminal =true;
     }


     if (isTerminal)
     {
         CalcCorrelExterneTerm(aBoxIm1,aMinPxMin,aMaxPxMax);
     }
     else
     {
        cPt2di aP0 = aBoxIm1.P0();
        cPt2di aP1 = aBoxIm1.P1();
        std::vector<int> aVx{aP0.x(),(aP0.x()+aP1.x())/2,aP1.x()};
        std::vector<int> aVy{aP0.y(),(aP0.y()+aP1.y())/2,aP1.y()};
        for (int aKx=0 ; aKx<2 ; aKx++)
        {
             for (int aKy=0 ; aKy<2 ; aKy++)
             {
                  cPt2di aQ0(aVx.at(aKx),aVy.at(aKy));
                  cPt2di aQ1(aVx.at(aKx+1),aVy.at(aKy+1));
                  CalcCorrelExterneRecurs(cBox2di(aQ0,aQ1));
             }
        }
     }
}

void cOneModele::CalcCorrelExterne()
{
   mAppli->MakeNormalizedIm();
   CalcCorrelExterneRecurs(mAppli->BoxGlob1());
}

double cOneModele::ComputeCost(bool & Ok,const cPt2di & aPC1,const cPt2di & aPC20,int aDZ) const
{
    Ok = false;
    double aCost= 1.0;
    if (mWithStatModele)
    {
       int aX1 =  aPC1.x();
       int aX2 =  aPC20.x() + aDZ;
       aCost = 0.5;
       if (mAppli->Ok1(aX1) && mAppli->Ok2(aX2))
       {
           cVecCaracMatch aVCM(*mPyrL1,*mPyrL2,mAppli->PC1(aX1),mAppli->PC2(aX2));
	   aCost = 1-mModele.HomologyLikelihood(aVCM,mAppli->InterpolLearn());
           aCost = mAppli->FactLearn() * pow(std::max(0.0,aCost),mAppli->ExpLearn());
           Ok = true;
       };
    }
    else if (mWithIntCorr)
    {
        cPt2dr aPC2Z(aPC20.x()+aDZ*mAppli->StepZ(),aPC20.y());
        double aCorrel = 0.0;

        if (WindInside4BL(mAppli->DI1(),aPC1,mPSzW) && WindInside4BL(mAppli->DI2(),aPC2Z,mPSzW))
        {
            aCorrel = mAppli->ComputCorrel(aPC1,aPC2Z,mSzW);
	    Ok = true;
	}
	aCost=(1-aCorrel)/2.0;
    }
    else if (mWithExtCorr)
    {
    }
    return aCost;
}

/* *************************************************** */
/*                                                     */
/*              cAppliFillCubeCost                     */
/*                                                     */
/* *************************************************** */

cAppliFillCubeCost::cAppliFillCubeCost(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mBoxGlob1            (cBox2di::Empty()),
   mBoxGlob2            (cBox2di::Empty()),
   mStepZ               (1.0),
   mCmpCorLearn         (true),
   mInterpolLearn       (true),
   mExpLearn            (0.5),
   mFactLearn           (0.33333),
   mSzW                 (3),
   mNbCmpCL             (200),
   mImCmp               (cPt2di(mNbCmpCL+2,mNbCmpCL+2),nullptr,eModeInitImage::eMIA_Null),
   mFileCube            (nullptr),
   mImZMin              (cPt2di(1,1)),
   mImZMax              (cPt2di(1,1)),
   mIm1                 (cPt2di(1,1)),
   mDI1                 (nullptr),
   mIm2                 (cPt2di(1,1)),
   mDI2                 (nullptr),
   mImNorm1             (cPt2di(1,1)),
   mDINorm1             (nullptr),
   mImNorm2             (cPt2di(1,1)),
   mDINorm2             (nullptr),
   mLayerCor            (tLayerCor::Empty()),
   mPyrL1               (nullptr),
   mPyrL2               (nullptr),
   mFPC                 (false)
{
    mFPC.FinishAC();
    mFPC.Check();
}

void cAppliFillCubeCost::MakeNormalizedIm()
{
    if (mDINorm1!= nullptr) return;

    mImNorm1 = NormalizedAvgDev(mIm1,1e-4);
    mDINorm1 = &(mImNorm1.DIm());

    mImNorm2 = NormalizedAvgDev(mIm2,1e-4);
    mDINorm2 = &(mImNorm2.DIm());

    mLayerCor  = tLayerCor(mImZMin,mImZMax);
}


double cAppliFillCubeCost::ToCmpCost(double aCost) const
{
   return mNbCmpCL * std::max(0.0,std::min(1.0,aCost));
}

cCollecSpecArg2007 & cAppliFillCubeCost::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameI1,"Name of first image")
          <<   Arg2007(mNameI2,"Name of second image")
          <<   Arg2007(mNameModele,"Name for modele : .*dmp|MMVIICorrel")
          <<   Arg2007(mP0Z,"Origin in first image")
          <<   Arg2007(mBoxGlob1,"Box to read 4 Im1")
          <<   Arg2007(mBoxGlob2,"Box to read 4 Im2")
          <<   Arg2007(mNamePost,"Post fix for other names (ZMin,ZMax,Cube)")
   ;
}

cCollecSpecArg2007 & cAppliFillCubeCost::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          // << AOpt2007(mStepZ, "StepZ","Step for paralax",{eTA2007::HDV})
          << AOpt2007(mNameCmpModele, "ModCmp","Modele for Comparison")
          << AOpt2007(mSzW, "SzW","Size for windows to match",{eTA2007::HDV})
   ;
}

std::string cAppliFillCubeCost::StdName(const std::string & aPre,const std::string & aPost)
{
	return aPre + "_" + mNamePost + "." + aPost;
}

double cAppliFillCubeCost::ComputCorrel(const cPt2di & aPCI1,const cPt2dr & aPCI2,int aSzW) const
{
   cMatIner2Var<tREAL4> aMat;

   for (int aDx=-aSzW ; aDx<=aSzW  ; aDx++)
   {
       for (int aDy=-aSzW ; aDy<=aSzW  ; aDy++)
       {
            aMat.Add
            (
                mDI1->GetV  (aPCI1+cPt2di(aDx,aDy)),
                mDI2->GetVBL(aPCI2+cPt2dr(aDx,aDy))
            );
       }
   }

   return aMat.Correl();
}

void cAppliFillCubeCost::PushCost(double aCost)
{
   tU_INT2 aICost = round_ni(1e4*(std::max(0.0,std::min(1.0,aCost))));
   mFileCube->Write(aICost);
}

void cAppliFillCubeCost::MakeLinePC(int aYLoc,bool Im1)
{
   if (mPyrL1==nullptr)
      return;
   MMVII_INTERNAL_ASSERT_strong(mStepZ==1.0,"For now do not handle StepZ!=1 with model");

   std::vector<bool>     & aVOK  = Im1 ? mVOk1      : mVOk2;
   std::vector<cAimePCar>& aVPC  = Im1 ? mVPC1      : mVPC2;
   const cBox2di         & aBox  = Im1 ? mBoxGlob1  : mBoxGlob2;
   cPyr1ImLearnMatch     & aPyrL = Im1 ? *mPyrL1    : *mPyrL2;

   aVOK.clear();
   aVPC.clear();

   for (int aX=aBox.P0().x() ; aX<=aBox.P1().x()  ; aX++)
   {
       cPt2di aPAbs (aX,aYLoc+mP0Z.y());
       cPt2di aPLoc = aPAbs - aBox.P0();
       aVOK.push_back(aPyrL.CalculAimeDesc(ToR(aPLoc)));
       aVPC.push_back(aPyrL.DupLPIm());
   }
}

int  cAppliFillCubeCost::Exe()
{

   // Compute names
   mNameZMin = StdName("ZMin","tif");
   mNameZMax = StdName("ZMax","tif");
   mNameCube = StdName("MatchingCube","data");
   
   //  Read images 
   mImZMin = tImZ::FromFile(mNameZMin);
   tDataImZ & aDZMin = mImZMin.DIm();
   mImZMax = tImZ::FromFile(mNameZMax);
   tDataImZ & aDZMax = mImZMax.DIm();

   mIm1 = tImRad::FromFile(mNameI1,mBoxGlob1);
   mDI1 = &(mIm1.DIm());
   mIm2 = tImRad::FromFile(mNameI2,mBoxGlob2);
   mDI2 = &(mIm2.DIm());

   mFileCube = new cMMVII_Ofs(mNameCube,false);

   mCmpCorLearn = IsInit(&mNameCmpModele);
   std::vector<cOneModele*> aVMods;
   aVMods.push_back(new cOneModele(mNameModele,*this));
   if (mCmpCorLearn)
       aVMods.push_back(new cOneModele(mNameCmpModele,*this));



   cPt2di aSz = aDZMin.Sz();
   cPt2di aPix;

   int aCpt=0;

   for (aPix.y()=0 ; aPix.y()<aSz.y() ; aPix.y()++)
   {
       StdOut() << "Line " << aPix.y() << " on " << aSz.y()  << "\n";
       MakeLinePC(aPix.y(),true );
       MakeLinePC(aPix.y(),false);
       for (aPix.x()=0 ; aPix.x()<aSz.x() ; aPix.x()++)
       {
            cPt2di aPAbs = aPix + mP0Z;
            cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
            cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
            for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
            {
               double aTabCost[2];
               bool   aTabOk[2];
	       for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                    aTabCost[aK] = aVMods[aK]->ComputeCost(aTabOk[aK],aPC1,aPC20,aDz);
               aCpt++;
               PushCost(aTabCost[0]);

	       if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
	       {
                  double aC0 = ToCmpCost(aTabCost[0]);
                  double aC1 = ToCmpCost(aTabCost[1]);
                  mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
	       }
            }
       }
   }


   if (mCmpCorLearn)
   {
       mImCmp.DIm().ToFile("CmpCorrLearn_"+ mNamePost + ".tif");
       //BREAK_POINT("Compare Corr/Learned made");
   }

   delete mFileCube;
   delete mPyrL1;
   delete mPyrL2;
   DeleteAllAndClear(aVMods);

   return EXIT_SUCCESS;
}



};

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
using namespace  cNS_FillCubeCost;

tMMVII_UnikPApli Alloc_FillCubeCost(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliFillCubeCost(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecFillCubeCost
(
     "DM4FillCubeCost",
      Alloc_FillCubeCost,
      "Fill a cube with matching costs",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::ToDef},
      __FILE__
);



};
