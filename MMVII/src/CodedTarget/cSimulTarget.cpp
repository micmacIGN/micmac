#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Sensor.h"
#include "MMVII_Random.h"



// Test git branch

namespace MMVII
{

/*  *********************************************************** */
/*                                                              */
/*                   cGeomSimDCT                                */
/*                                                              */
/*  *********************************************************** */

bool cGeomSimDCT::Intersect(const cGeomSimDCT &  aG2) const
{
     return  Norm2(mC-aG2.mC) < mR2+aG2.mR2 ;
}
cGeomSimDCT::cGeomSimDCT() :
    mResExtr (nullptr)
{
}

void cGeomSimDCT::Translate(const cPt2dr & aTr)
{
    mC += aTr;
    mCornEl1 += aTr;
    mCornEl2 += aTr;
}


cGeomSimDCT::cGeomSimDCT(const cOneEncoding & anEncod, const  cPt2dr& aC, const double& aR1, const double& aR2, const std::string &aName):
    mResExtr (nullptr),
    mEncod   (anEncod),
    //  mNum (aNum),
    mC   (aC),
    mR1  (aR1),
    mR2  (aR2),
    mName(aName)
{
}

void AddData(const  cAuxAr2007 & anAux,cGeomSimDCT & aGSD)
{
   aGSD.mEncod.AddData(cAuxAr2007("Encod",anAux));
   //  ========   MMVII::AddData(cAuxAr2007("Name",anAux),aGSD.mName);
   MMVII::AddData(cAuxAr2007("Center",anAux),aGSD.mC);
   MMVII::AddData(cAuxAr2007("CornEl1",anAux),aGSD.mCornEl1);
   MMVII::AddData(cAuxAr2007("CornEl2",anAux),aGSD.mCornEl2);
   MMVII::AddData(cAuxAr2007("R1",anAux),aGSD.mR1);
   MMVII::AddData(cAuxAr2007("R2",anAux),aGSD.mR2);
}

/*  *********************************************************** */
/*                                                              */
/*                      cResSimul                               */
/*                                                              */
/*  *********************************************************** */

cResSimul::cResSimul() :
    mRadiusMinMax (15.0,60.0),
    mBorder    (1.0),
    mRatioMinMax  (0.3,1.0)
{
}

double cResSimul::BorderGlob() const
{
    return mBorder * mRadiusMinMax.y();
}

void AddData(const  cAuxAr2007 & anAux,cResSimul & aRS)
{
   // Modif MPD, know that commande are quoted, they cannot be used as tag =>  "MMVII  "toto" "b=3" " => PB!!
   //  MMVII::AddData(cAuxAr2007("Com",anAux),aRS.mCom);
   anAux.Ar().AddComment(aRS.mCom);
   MMVII::AddData(cAuxAr2007("RadiusMinMax",anAux),aRS.mRadiusMinMax);
   MMVII::AddData(cAuxAr2007("RatioMax",anAux),aRS.mRatioMinMax);
   MMVII::AddData(cAuxAr2007("Geoms",anAux),aRS.mVG);
}

cResSimul  cResSimul::FromFile(const std::string& aName)
{
   cResSimul aRes;

   ReadFromFile(aRes,aName);
   return aRes;
}



/*  *********************************************************** */
/*                                                              */
/*              cAppliSimulCodeTarget                           */
/*                                                              */
/*  *********************************************************** */


class cAppliSimulCodeTarget : public cMMVII_Appli
{
     public :
        typedef tREAL4            tElem;
        typedef cIm2D<tElem>      tIm;
        typedef cDataIm2D<tElem>  tDIm;
        typedef cAffin2D<tREAL8>  tAffMap;


        cAppliSimulCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        // =========== overridding cMMVII_Appli::methods ============
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;


        // =========== other methods ============

	/// Find a new  position of target, not too close from existing ones, in mRS.mVG
        void  AddPosTarget(const cOneEncoding & );

	/// Put the target in the image
        void  IncrustTarget(cGeomSimDCT & aGSD);

        cPhotogrammetricProject     mPhProj;

        // =========== Mandatory args ============

	std::string mNameIm;       ///< Name of background image
	std::string mNameSpecif;   ///< Name of specification file

        // =========== Optionnal args ============
	cResSimul           mRS;        /// List of result
        double              mSzKernel;  /// Sz of interpolation kernel
	std::string         mPatternNames;

                //  --
	double              mDownScale;       ///< initial downscale of target
	cPt2dr              mAttenContrast;         ///< min/max amplitude of (random) gray attenuatio,
	cPt2dr              mAttenMul;         ///< min/max Multiplicative attenuation
	cPt2dr              mPropSysLin;      ///< min/max amplitude of (random) linear bias
	cPt2dr              mAmplWhiteNoise;  ///< min/max amplitude of random white noise

        // =========== Internal param ============
        tIm                        mImIn;        ///< Input global image
	cFullSpecifTarget *        mSpec;        ///< Specification of target creation
	std::string                mSuplPref;   ///< Supplementary prefix
	std::string                mPrefixOut;   ///< Prefix for generating image & ground truth
};


/* *************************************************** */
/*                                                     */
/*              cAppliSimulCodeTarget                  */
/*                                                     */
/* *************************************************** */

cAppliSimulCodeTarget::cAppliSimulCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mSzKernel        (2.0),
   mPatternNames    (".*"),
   mDownScale       (3.0),
   mAttenContrast   (0.,0.2),
   mAttenMul        (0.0,0.4),
   mPropSysLin      (0.,0.2),
   mAmplWhiteNoise  (0.,0.1),
   mImIn            (cPt2di(1,1)),
   mSpec            (nullptr),
   mSuplPref        ("")
{
}

cCollecSpecArg2007 & cAppliSimulCodeTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Pattern of files",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
             <<   Arg2007(mNameSpecif,"Name of target file specification",{{eTA2007::XmlOfTopTag,cFullSpecifTarget::TheMainTag}})
   ;
}


cCollecSpecArg2007 & cAppliSimulCodeTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
	        anArgOpt
             <<   mPhProj.DPGndPt2D().ArgDirOutOptWithDef("Simul")
             <<   AOpt2007(mRS.mRadiusMinMax,"Radius","Min/Max radius for gen target",{eTA2007::HDV})
             <<   AOpt2007(mRS.mRatioMinMax,"Ratio","Min/Max ratio between target ellipses axis (<=1)",{eTA2007::HDV})
             <<   AOpt2007(mPatternNames,"PatNames","Pattern for selection of names",{eTA2007::HDV})
             <<   AOpt2007(mSzKernel,"SzK","Sz of Kernel for interpol",{eTA2007::HDV})
             <<   AOpt2007(mRS.mBorder,"Border","Border w/o target, prop to R Max",{eTA2007::HDV})
             <<   AOpt2007(mAmplWhiteNoise,"NoiseAmpl","Amplitude White Noise",{eTA2007::HDV})
             <<   AOpt2007(mPropSysLin,"PropLinBias","Amplitude Linear Bias",{eTA2007::HDV})
             <<   AOpt2007(mAttenContrast,"ContrastAtten","Attenution of B/W contrast",{eTA2007::HDV})
             <<   AOpt2007(mAttenMul,"MulAtten","Attenution multiplicatives",{eTA2007::HDV})
             <<   AOpt2007(mSuplPref,"SuplPref","Suplementary prefix for outputs")
   ;
}

void   cAppliSimulCodeTarget::AddPosTarget(const cOneEncoding & anEncod)
{
     cBox2dr aBoxGenerate = mImIn.DIm().ToR().Dilate(-mRS.BorderGlob());
     // make a certain number of try for getting a target not intesecting already selected
     for (int aK=0 ; aK< 200 ; aK++)
     {
        cPt2dr  aC = aBoxGenerate.GeneratePointInside(); // generat a random point inside the box
        //  Compute two random radii in the given interval
        double  aRbig = RandInInterval(mRS.mRadiusMinMax);
        double  aRsmall = aRbig*RandInInterval(mRS.mRatioMinMax);

        // check if there is already a selected target overlaping
        cGeomSimDCT aGSD(anEncod,aC,aRsmall,aRbig,anEncod.Name());
	bool GotClose = false;
	for (const auto& aG2 : mRS.mVG)
            GotClose = GotClose || aG2.Intersect(aGSD);
        // if not :all fine, memorize and return
	if (! GotClose)
	{
            mRS.mVG.push_back(aGSD);
	    return;
	}
     }
}

void  cAppliSimulCodeTarget::IncrustTarget(cGeomSimDCT & aGSD)
{
    // We want that random is different for each image, but deterministic, independent of number of pixel noise drawn
    cRandGenerator::TheOne()->setSeed(HashValue(mNameIm+"*"+aGSD.mName,true));

    // [1] -- Load and scale image of target
    tIm aImT =  Convert((tElem*)nullptr,mSpec->OneImTarget(aGSD.mEncod).DIm());

/*
    static bool isFirst = false;
    if (isFirst && mShowFirst)
    {
       isFirst = true;
   }
*/
    aImT =  aImT.GaussDeZoom(mDownScale,5);
    

    // [2] -- Make a "noisy" version of image (white noise, affine biase, grey attenuation)
    tDIm & aDImT = aImT.DIm();
    cPt2dr aSz = ToR(aDImT.Sz());
    cPt2dr aC0 = mSpec->Center() /mDownScale;
    // cPt2dr aC0 = mPCT.mCenterF/mDownScale;

    cPt2dr aDirModif = FromPolar(1.0,M_PI*RandUnif_C());
    double aDiag = Norm2(aC0);
    double aAttenContrast = RandInInterval(mAttenContrast);
    double aAttenMul = (1-mAttenMul.y()) + RandInInterval(mAttenMul);
    double aAttenLin  = RandInInterval(mPropSysLin);
    for (const auto & aPix : aDImT)
    {
         double aVal = aDImT.GetV(aPix);
	 aVal =  128  + (aVal-128) * (1-aAttenContrast)   ;              //  attenuate, to have grey-level
         double aScal = Scal(ToR(aPix)-aC0,aDirModif) / aDiag;   // compute amplitude of linear bias
	 aVal =  128  + (aVal-128) * (1-aAttenLin)  + aAttenLin * aScal * 128;
	 aVal = aVal * aAttenMul;
	 //
	 aDImT.SetV(aPix,aVal);
    }

    // [3] -- Compute mapping  ImageTarget  <--> Simul Image


    tAffMap aMap0 =  tAffMap::Translation(-aC0);                           // Set center in 0,0
    tAffMap aMap1 =  tAffMap::Rotation(M_PI*RandUnif_C());                 // Apply a rotation
    tAffMap aMap2 =  tAffMap::HomotXY(aGSD.mR1/aSz.x(),aGSD.mR2/aSz.y());  // Apply a homotety on each axe
    tAffMap aMap3 =  tAffMap::Rotation(M_PI*RandUnif_C());                 //  Apply a rotation again
    tAffMap aMap4 =  tAffMap::Translation(aGSD.mC);                        // set 0,0 to center

    tAffMap aMapT2Im =  aMap4 * aMap3 * aMap2 * aMap1 * aMap0;             // compute composition, get TargetCoord -> Image Coord
    tAffMap aMapIm2T =  aMapT2Im.MapInverse();                             // need inverse for resample


    // [4] -- Do the incrustation of target in  image
    cBox2di aBoxIm = ImageOfBox(aMapT2Im,aDImT.ToR()).Dilate(mSzKernel+2).ToI();

    tDIm & aDImIn = mImIn.DIm();
    for (const auto & aPix : cRect2(aBoxIm)) // ressample image, parse image coordinates
    {
        if ( aDImIn.Inside(aPix))
        {
            // compute a weighted coordinate in target coordinates,
            cRessampleWeigth  aRW = cRessampleWeigth::GaussBiCub(ToR(aPix),aMapIm2T,mSzKernel);
            const std::vector<cPt2di>  & aVPts = aRW.mVPts;
            if (!aVPts.empty())
            {
                double aSomW = 0.0;  // sum of weight
                double aSomVW = 0.0;  //  weighted sum of vals
                for (int aK=0; aK<int(aVPts.size()) ; aK++)
                {
                    if (aDImT.Inside(aVPts[aK]))
                    {
                        double aW = aRW.mVWeight[aK];
                        aSomW  += aW;
                        aSomVW += aW * aDImT.GetV(aVPts[aK]);
                    }
                }
                // in cRessampleWeigth =>  Sum(W) is standartized and equals 1

                aSomVW = aSomVW * (1- mAmplWhiteNoise.y()) + RandInInterval_C(mAmplWhiteNoise) * 128 * aSomW;
                double aVal = aSomVW + (1-aSomW)*aDImIn.GetV(aPix);
                aDImIn.SetV(aPix,aVal);
            }
        }
    }
    // aGSD.mCornEl1 = aMapT2Im.Value(mPCT.mCornEl1/mDownScale);
    // aGSD.mCornEl2 = aMapT2Im.Value(mPCT.mCornEl2/mDownScale);
    aGSD.mCornEl1 = aMapT2Im.Value(mSpec->CornerlEl_BW()/mDownScale);
    aGSD.mCornEl2 = aMapT2Im.Value(mSpec->CornerlEl_WB()/mDownScale);

    StdOut() << "NNN= " << aGSD.mEncod.Name() << " C0=" << aC0 <<  aBoxIm.Sz() <<  " " << aGSD.mR2/aGSD.mR1 << std::endl;
}

const std::string ThePrefixSimulTarget = "SimulTarget_";
const std::string ThePostfixGTSimulTarget = "_GroundTruth.xml";

// make sure that interval.x<interval.y, check if all in [0,1]
bool orderAndAssertInterval01(cPt2dr & interval, const std::string & aIntervalName)
{
    if (interval.y()<interval.x())
        std::swap(interval.x(), interval.y());
    if ((interval.x()<0.)||(interval.y()>1.))
    {
         MMVII_USER_WARNING(aIntervalName+" must be in [0;1].");
         return false;
    }
    return true;
}

int  cAppliSimulCodeTarget::Exe()
{
   mPhProj.FinishInit();

   if (RunMultiSet(0,0))
   {
	   return ResultMultiSet();
   }

   // generally dont want to add target on target
   if (starts_with(FileOfPath(mNameIm),ThePrefixSimulTarget))
   {
        MMVII_USER_WARNING("File is already a simulation, dont process");
        return EXIT_SUCCESS;
   }

   // check that all intervals are in (0;1]
   if (!orderAndAssertInterval01(mRS.mRatioMinMax, "Ratio"))
       return EXIT_FAILURE;
   if (!orderAndAssertInterval01(mAmplWhiteNoise, "NoiseAmpl"))
       return EXIT_FAILURE;
   if (!orderAndAssertInterval01(mPropSysLin, "PropLinBias"))
       return EXIT_FAILURE;
   if (!orderAndAssertInterval01(mAttenContrast, "ContrastAtten"))
       return EXIT_FAILURE;
   if (!orderAndAssertInterval01(mAttenMul, "MulAtten"))
       return EXIT_FAILURE;


   // We want that random is different for each image, but deterministic for one given image
   cRandGenerator::TheOne()->setSeed(HashValue(mNameIm,true));

   mPrefixOut =  ThePrefixSimulTarget +  mSuplPref + LastPrefix(FileOfPath(mNameIm));
   mRS.mCom = CommandOfMain().Com();
   // mPCT.InitFromFile(mNameSpecif);
   mSpec =  cFullSpecifTarget::CreateFromFile(mNameSpecif);


   mImIn = tIm::FromFile(mNameIm);

   for (const auto & anEncod : mSpec->Encodings())
   {
        if (MatchRegex(anEncod.Name(),mPatternNames))
	{
            // We want that random is different for each image, but deterministic for one given image
            cRandGenerator::TheOne()->setSeed(HashValue(mNameIm+"/"+anEncod.Name(),true));
            AddPosTarget(anEncod);
            //StdOut() <<  "Target " << anEncod.Name() << " " << mRS.mVG.back().mC << std::endl;
	}
   }

   if (!mRS.mVG.empty())
       StdOut() <<  "1st target " << mRS.mVG[0].mName << " " << mRS.mVG[0].mC << std::endl;

   for (auto  & aG : mRS.mVG)
   {
       IncrustTarget(aG);
   }

   std::string aNameOut = mPrefixOut+".tif";
   {
        cSetMesPtOf1Im  aSetM(aNameOut);
        for (const auto & aGSim : mRS.mVG)
        {
            aSetM.AddMeasure(cMesIm1Pt(aGSim.mC,aGSim.mEncod.Name(),1.0));
        }
        mPhProj.SaveMeasureIm(aSetM);
   }

   SaveInFile(mRS,mPhProj.DPGndPt2D().FullDirOut() + mPrefixOut + ThePostfixGTSimulTarget);
   // StdOut() <<  "ooo--OOOOO=" << mPhProj.DPPointsMeasures().FullDirOut() + mPrefixOut + ThePostfixGTSimulTarget << "\n";

   mImIn.DIm().ToFile(aNameOut,eTyNums::eTN_U_INT1);


   delete mSpec;

   return EXIT_SUCCESS;
}


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_SimulCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliSimulCodeTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecSimulCodedTarget
(
     "CodedTargetSimul",
      Alloc_SimulCodedTarget,
      "Simulate images of coded targets, with ground truth",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


};
