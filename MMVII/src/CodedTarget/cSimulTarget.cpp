#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"


// Test git branch

namespace MMVII
{

namespace  cNS_CodedTarget
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


cGeomSimDCT::cGeomSimDCT(const cOneEncoding & anEncod,const  cPt2dr& aC,const double& aR1,const double& aR2):
    mResExtr (nullptr),
    mEncod   (anEncod),
    //  mNum (aNum),
    mC   (aC),
    mR1  (aR1),
    mR2  (aR2)
{
}

void AddData(const  cAuxAr2007 & anAux,cGeomSimDCT & aGSD)
{
   aGSD.mEncod.AddData(cAuxAr2007("Encod",anAux));
   MMVII::AddData(cAuxAr2007("Name",anAux),aGSD.name);
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
    mRayMinMax (15.0,60.0),
    mBorder    (1.0),
    mRatioMax  (3.0)
{
}

double cResSimul::BorderGlob() const
{
    return mBorder * mRayMinMax.y();
}

void AddData(const  cAuxAr2007 & anAux,cResSimul & aRS)
{
   MMVII::AddData(cAuxAr2007("Com",anAux),aRS.mCom);
   MMVII::AddData(cAuxAr2007("RayMinMax",anAux),aRS.mRayMinMax);
   MMVII::AddData(cAuxAr2007("RatioMax",anAux),aRS.mRatioMax);
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

	/// return a random value in the specified interval 
	double RandomRay() const;


        // =========== Mandatory args ============

	std::string mNameIm;       ///< Name of background image
	std::string mNameSpecif;   ///< Name of specification file

        // =========== Optionnal args ============
	cResSimul           mRS;        /// List of result
        double              mSzKernel;  /// Sz of interpolation kernel 

                //  --
	double              mDownScale;       ///< initial downscale of target
	double              mAttenBW;         ///< amplitude of (random) gray attenuatio,
	double              mPropSysLin;      ///< amplitude of (random) linear bias
	double              mAmplWhiteNoise;  ///< amplitud of random white noise

        // =========== Internal param ============
        tIm                        mImIn;        ///< Input global image
	cFullSpecifTarget *        mSpec;        ///< Specification of target creation
	std::string                mPrefixOut;   ///< Prefix for generating image & ground truth
};


/* *************************************************** */
/*                                                     */
/*              cAppliSimulCodeTarget                  */
/*                                                     */
/* *************************************************** */

cAppliSimulCodeTarget::cAppliSimulCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mSzKernel        (2.0),
   mDownScale       (3.0),
   mAttenBW         (0.2),
   mPropSysLin      (0.2),
   mAmplWhiteNoise  (0.1),
   mImIn            (cPt2di(1,1)),
   mSpec            (nullptr)
{
}

cCollecSpecArg2007 & cAppliSimulCodeTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Name of image (first one)",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
             <<   Arg2007(mNameSpecif,"Name of target file")
   ;
}


cCollecSpecArg2007 & cAppliSimulCodeTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
	        anArgOpt
             <<   AOpt2007(mRS.mRayMinMax,"Rays","Min/Max ray for gen target",{eTA2007::HDV})
             <<   AOpt2007(mSzKernel,"SzK","Sz of Kernel for interpol",{eTA2007::HDV})
             <<   AOpt2007(mRS.mBorder,"Border","Border w/o target, prop to R Max",{eTA2007::HDV})
             <<   AOpt2007(mAmplWhiteNoise,"NoiseAmpl","Amplitude White Noise",{eTA2007::HDV})
             <<   AOpt2007(mPropSysLin,"PropLinBias","Amplitude Linear Bias",{eTA2007::HDV})
             <<   AOpt2007(mAttenBW,"BWAttend","Attenution of B/W extrem vaues",{eTA2007::HDV})
   ;
}

double cAppliSimulCodeTarget::RandomRay() const { return RandInInterval(mRS.mRayMinMax.x(),mRS.mRayMinMax.y());}



void   cAppliSimulCodeTarget::AddPosTarget(const cOneEncoding & anEncod)
{
     cBox2dr aBoxGenerate = mImIn.DIm().ToR().Dilate(-mRS.BorderGlob());
     // make a certain number of try for getting a target not intesecting already selected
     for (int aK=0 ; aK< 200 ; aK++)
     {
        cPt2dr  aC = aBoxGenerate.GeneratePointInside(); // generat a random point inside the box
        //  Compute two random ray in the given interval
        double  aR1 = RandomRay() ;
        double  aR2 = RandomRay() ;
	OrderMinMax(aR1,aR2);  //   assure  aR1 <= aR2

        // assure that  R2/R1 <= RatioMax
        // if not "magic" formula to assure R1/R2 = RatioMax  R1R2 = R1Init R2Init
	if (aR2/aR1 > mRS.mRatioMax)  
	{
            double aR = sqrt(aR1*aR2);
	    aR1 = aR / sqrt(mRS.mRatioMax);
	    aR2 = aR * sqrt(mRS.mRatioMax);
	}
	// check if there is already a selected target overlaping
        cGeomSimDCT aGSD(anEncod,aC,aR1,aR2);
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
    // [1] -- Load and scale image of target
    tIm aImT =  Convert((tElem*)nullptr,mSpec->OneImTarget(aGSD.mEncod).DIm());
    aImT =  aImT.GaussDeZoom(mDownScale,5);
    

    // [2] -- Make a "noisy" version of image (white noise, affine biase, grey attenuation)
    tDIm & aDImT = aImT.DIm();
    cPt2dr aSz = ToR(aDImT.Sz());
    cPt2dr aC0 = mSpec->Center() /mDownScale;
    // cPt2dr aC0 = mPCT.mCenterF/mDownScale;

    cPt2dr aDirModif = FromPolar(1.0,M_PI*RandUnif_C());
    double aDiag = Norm2(aC0);
    double aAtten = mAttenBW * RandUnif_0_1();
    double aAttenLin  = mPropSysLin * RandUnif_0_1();
    for (const auto & aPix : aDImT)
    {
         double aVal = aDImT.GetV(aPix);
	 aVal =  128  + (aVal-128) * (1-aAtten)   ;              //  attenuate, to have grey-level
         double aScal = Scal(ToR(aPix)-aC0,aDirModif) / aDiag;   // compute amplitude of linear bias
	 aVal =  128  + (aVal-128) * (1-aAttenLin)  + aAttenLin * aScal * 128;
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
		aSomVW =  aSomVW * (1- mAmplWhiteNoise) +  mAmplWhiteNoise * 128 * RandUnif_C() * aSomW;
	        double aVal = aSomVW + (1-aSomW)*aDImIn.GetV(aPix);
	        aDImIn.SetV(aPix,aVal);
	    }
	}
    }
    // aGSD.mCornEl1 = aMapT2Im.Value(mPCT.mCornEl1/mDownScale);
    // aGSD.mCornEl2 = aMapT2Im.Value(mPCT.mCornEl2/mDownScale);
    aGSD.mCornEl1 = aMapT2Im.Value(mSpec->CornerlEl_BW()/mDownScale);
    aGSD.mCornEl2 = aMapT2Im.Value(mSpec->CornerlEl_WB()/mDownScale);

    StdOut() << "NNN= " << aGSD.mEncod.Name() << " C0=" << aC0 <<  aBoxIm.Sz() <<  " " << aGSD.mR2/aGSD.mR1 << "\n";
}

int  cAppliSimulCodeTarget::Exe()
{
   mPrefixOut =  "SimulTarget_" + LastPrefix(mNameIm);
   mRS.mCom = CommandOfMain();
   // mPCT.InitFromFile(mNameSpecif);
   mSpec =  cFullSpecifTarget::CreateFromFile(mNameSpecif);

   mImIn = tIm::FromFile(mNameIm);

   for (const auto & anEncod : mSpec->Encodings())
   {
        AddPosTarget(anEncod);
        StdOut() <<  "Ccc=" << mRS.mVG.back().mC << "\n";
   }

   for (auto  & aG : mRS.mVG)
   {
       IncrustTarget(aG);
   }
   SaveInFile(mRS,mPrefixOut +"_GroundTruth.xml");

   mImIn.DIm().ToFile(mPrefixOut+".tif",eTyNums::eTN_U_INT1);


   delete mSpec;

   return EXIT_SUCCESS;
}
};


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
using namespace  cNS_CodedTarget;

tMMVII_UnikPApli Alloc_SimulCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliSimulCodeTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecSimulCodedTarget
(
     "CodedTargetSimul",
      Alloc_SimulCodedTarget,
      "Extract coded target from images",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


};
