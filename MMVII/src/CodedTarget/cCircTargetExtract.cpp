#include "MMVII_Tpl_Images.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Sensor.h"
#include "MMVII_TplImage_PtsFromValue.h"

/*   Modularistion
 *   Code extern tel que ellipse
 *   Ellipse => avec centre
 *   Pas de continue
 */

namespace MMVII
{


struct cParamBWTarget;  ///< Store pararameters for Black & White target
struct cSeedBWTarget;   ///< Store data for seed point of circular extraction


// ==  enum class eEEBW_Lab;   ///< label used in ellipse extraction
class cExtract_BW_Target;  ///< class for ellipse extraction

/*  *********************************************************** */
/*                                                              */
/*                   cParamBWTarget                             */
/*                                                              */
/*  *********************************************************** */

struct cParamBWTarget
{
    public :
      cParamBWTarget();

      int NbMaxPtsCC() const; ///<Max number of point (computed from MaxDiam)
      int NbMinPtsCC() const; ///<Min number of point (computed from MinDiam)

      double    mFactDeriche;   ///< Factor for gradient with deriche-method
      int       mD0BW;          ///< distance to border
      double    mValMinW;       ///< Min Value for white
      double    mValMaxB;       ///< Max value for black
      double    mRatioMaxBW;    ///< Max Ratio   Black/White
      double    mMinDiam;       ///< Minimal diameter
      double    mMaxDiam;       ///< Maximal diameter
      double    mPropFr;        ///< Minima prop of point wher frontier extraction suceeded
      int       mNbMinFront;    ///< Minimal number of point
};




int cParamBWTarget::NbMaxPtsCC() const { return M_PI * Square(mMaxDiam/2.0); }
int cParamBWTarget::NbMinPtsCC() const { return M_PI * Square(mMinDiam/2.0); }


cParamBWTarget::cParamBWTarget() :
    mFactDeriche (2.0),
    mD0BW        (2),
    mValMinW     (20), 
    mValMaxB     (100),
    mRatioMaxBW  (1/1.5),
    mMinDiam     (7.0),
    mMaxDiam     (60.0),
    mPropFr      (0.95),
    mNbMinFront  (10)
{
}

/*  *********************************************************** */
/*                                                              */
/*               cSeedBWTarget                                  */
/*                                                              */
/*  *********************************************************** */

struct cSeedBWTarget
{
    public :
       cPt2di mPixW;
       cPt2di mPixTop;

       cPt2di mPInf;
       cPt2di mPSup;

       tREAL4 mBlack;
       tREAL4 mWhite;
       bool   mOk;
       bool   mMarked4Test;

       cSeedBWTarget(const cPt2di & aPixW,const cPt2di & aPixTop,  tREAL4 mBlack,tREAL4 mWhite);
};

cSeedBWTarget::cSeedBWTarget(const cPt2di & aPixW,const cPt2di & aPixTop,tREAL4 aBlack,tREAL4 aWhite):
   mPixW        (aPixW),
   mPixTop      (aPixTop),
   mBlack       (aBlack),
   mWhite       (aWhite),
   mOk          (true),
   mMarked4Test (false)
{
}


/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Target                             */
/*                                                              */
/*  *********************************************************** */

enum class eEEBW_Lab : tU_INT1
{
   eFree,
   eBorder,
   eTmp,
   eBadZ,
   eBadFr,
   eElNotOk,
   eBadEl,
   eAverEl,
   eBadTeta
};


class cExtract_BW_Target
{
   public :
        typedef tREAL4              tElemIm;
        typedef cDataIm2D<tElemIm>  tDataIm;
        typedef cIm2D<tElemIm>      tIm;
        typedef cImGrad<tElemIm>    tImGrad;

	typedef cIm2D<tU_INT1>      tImMarq;
	typedef cDataIm2D<tU_INT1>  tDImMarq;

        cExtract_BW_Target(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest);

        void ExtractAllSeed();
        const std::vector<cSeedBWTarget> & VSeeds() const;
	const tDImMarq&    DImMarq() const;
	const tDataIm &    DGx() const;
	const tDataIm &    DGy() const;

	void SetMarq(const cPt2di & aP,eEEBW_Lab aLab) {mDImMarq.SetV(aP,tU_INT1(aLab));}

	void CC_SetMarq(eEEBW_Lab aLab); ///< set marqer on all connected component


	eEEBW_Lab  GetMarq(const cPt2di & aP) {return eEEBW_Lab(mDImMarq.GetV(aP));}
	bool MarqEq(const cPt2di & aP,eEEBW_Lab aLab) const {return mDImMarq.GetV(aP) == tU_INT1(aLab);}
	bool MarqFree(const cPt2di & aP) const {return MarqEq(aP,eEEBW_Lab::eFree);}

        bool AnalyseOneConnectedComponents(cSeedBWTarget &);
        bool ComputeFrontier(cSeedBWTarget & aSeed);

   protected :

	/// Is the point a candidate for seed (+- local maxima)
        bool IsExtremalPoint(const cPt2di &) ;

	/// Update the data for connected component with a new point (centroid, bbox, heap...)
	void AddPtInCC(const cPt2di &);
	// Prolongat on the vertical, untill its a max or a min
        cPt2di Prolongate(cPt2di aPix,bool IsW,tElemIm & aMaxGy) const;

	/// Extract the accurate frontier point, essentially prepare data to call "cGetPts_ImInterp_FromValue"
        cPt2dr RefineFrontierPoint(const cSeedBWTarget & aSeed,const cPt2di & aP0,bool & Ok);

        tIm              mIm;      ///< Image to analyse
        tDataIm &        mDIm;     ///<  Data of Image
	cPt2di           mSz;      ///< Size of image
	tImMarq          mImMarq;    ///< Marqer used in cc exploration
	tDImMarq&        mDImMarq;   ///< Data of Marqer
        cParamBWTarget   mPBWT;      ///<  Copy of parameters
        tImGrad          mImGrad;    ///<  Structure for computing gradient
	tDataIm &        mDGx;       ///<  Access to x-grad
	tDataIm &        mDGy;       ///<  Access to y-grad

        std::vector<cSeedBWTarget> mVSeeds;

	std::vector<cPt2di>  mPtsCC;
	int                  mIndCurPts;  ///< index of point explored in connected component
	cPt2dr               mCentroid;   ///< Centroid of conected compoonent, used for direction & reduction of coordinates
        cIm2D<tU_INT1>       mMasqTest;   ///< Mask for "special" point where we want to make test (debug/visu ...)
        cDataIm2D<tU_INT1>&  mDMasqT;     ///< Data of Masq

	cPt2di               mPSup;  ///< For bounding box, Sup corner
	cPt2di               mPInf;  ///< For bounding box, Inf corner
        std::vector<cPt2dr>  mVFront;
};


cExtract_BW_Target::cExtract_BW_Target(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest) :
   mIm        (anIm),
   mDIm       (mIm.DIm()),
   mSz        (mDIm.Sz()),
   mImMarq    (mSz),
   mDImMarq   (mImMarq.DIm()),
   mPBWT       (aPBWT),
   mImGrad    (Deriche( mDIm,mPBWT.mFactDeriche)),
   mDGx       (mImGrad.mGx.DIm()),
   mDGy       (mImGrad.mGy.DIm()),
   mMasqTest  (aMasqTest),
   mDMasqT    (mMasqTest.DIm())
{
   mDImMarq.InitInteriorAndBorder(tU_INT1(eEEBW_Lab::eFree),tU_INT1(eEEBW_Lab::eBorder));
}

/*
        0 0 0 0 0
      L 0 0 1 0 0  R 
        0 1 1 1 0
     
       R => negative gradient on x
       L => positive gradient on x
*/

///cSeedBWTarget

cPt2di cExtract_BW_Target::Prolongate(cPt2di aPix,bool IsW,tElemIm & aMaxGy) const
{
    cPt2di aDir = cPt2di(0,IsW?1:-1);

    while
    ( 
            MarqFree(aPix+aDir) 
	&&  (  IsW == (mDIm.GetV(aPix+aDir) >mDIm.GetV(aPix)))
    )
    {
        aPix = aPix + aDir;
	UpdateMax(aMaxGy ,mImGrad.mGy.DIm().GetV(aPix));
    }

    return aPix;
}
         
bool cExtract_BW_Target::IsExtremalPoint(const cPt2di & aPix) 
{
   // is it a point where gradient cross vertical line
   if ( (mDGx.GetV(aPix)>0) ||  (mDGx.GetV(aPix+cPt2di(-1,0)) <=0) )
      return false;

   // const tDataIm & aDGy = mImGrad.mGy.DIm();

   tElemIm aGy =  mDGy.GetV(aPix);
   // At top , grady must be positive
   if (  aGy<=0) 
      return false;

   // Now test that grad is a local maxima

   if (    (aGy  <   mDGy.GetV(aPix+cPt2di(0, 2)))
        || (aGy  <   mDGy.GetV(aPix+cPt2di(0, 1)))
        || (aGy  <=  mDGy.GetV(aPix+cPt2di(0,-1)))
        || (aGy  <=  mDGy.GetV(aPix+cPt2di(0,-2)))
      )
      return false;

   // tElemIm aVBlack =  mDIm.GetV(aPix+cPt2di(0,-2));
   // tElemIm aVWhite =  mDIm.GetV(aPix+cPt2di(0,2));
   /*
   if ((aVBlack/double(aVWhite)) > mPBWT.mRatioP2)
      return false;
      */

   tElemIm aMaxGy = aGy;
   cPt2di aPixB =  Prolongate(aPix,false,aMaxGy);
   tElemIm aVBlack =  mDIm.GetV(aPixB);

   cPt2di aPixW =  Prolongate(aPix,true,aMaxGy);
   tElemIm aVWhite =  mDIm.GetV(aPixW);

   if (aMaxGy> aGy)
      return false;
   
   if (aVWhite < mPBWT.mValMinW)
      return false;

   if (aVBlack > mPBWT.mValMaxB)
      return false;

    if ((aVBlack/double(aVWhite)) > mPBWT.mRatioMaxBW)
      return false;

   mVSeeds.push_back(cSeedBWTarget(aPixW,aPix,aVBlack,aVWhite));

   return true;
}

void cExtract_BW_Target::ExtractAllSeed()
{
   const cBox2di &  aFullBox = mDIm;
   cRect2  aBoxInt (aFullBox.Dilate(-mPBWT.mD0BW));
   int aNb=0;
   int aNbOk=0;
   for (const auto & aPix : aBoxInt)
   {
       aNb++;
       if (IsExtremalPoint(aPix))
       {
           aNbOk++;
       }
   }
   std::sort
   (
      mVSeeds.begin(),
      mVSeeds.end(),
      [](const cSeedBWTarget &  aS1,const cSeedBWTarget &  aS2) {return aS1.mWhite>aS2.mWhite;}
   );
   StdOut() << " PPPP="  << aNbOk / double(aNb) << "\n";
}

const std::vector<cSeedBWTarget> &      cExtract_BW_Target::VSeeds() const { return mVSeeds; }
const cExtract_BW_Target::tDImMarq&    cExtract_BW_Target::DImMarq() const {return mDImMarq;}
const cExtract_BW_Target::tDataIm&    cExtract_BW_Target::DGx() const {return mDGx;}
const cExtract_BW_Target::tDataIm&    cExtract_BW_Target::DGy() const {return mDGy;}

void cExtract_BW_Target::AddPtInCC(const cPt2di & aPt)
{
     mDImMarq.SetV(aPt,tU_INT1(eEEBW_Lab::eTmp) );
     mPtsCC.push_back(aPt);
     mCentroid = mCentroid + ToR(aPt);

     SetInfEq(mPInf,aPt);
     SetSupEq(mPSup,aPt);
}


void cExtract_BW_Target::CC_SetMarq(eEEBW_Lab aLab)
{
    for (const auto & aP : mPtsCC)
        SetMarq(aP,aLab);
}



cPt2dr cExtract_BW_Target::RefineFrontierPoint(const cSeedBWTarget & aSeed,const cPt2di & aPt,bool & Ok)
{
    Ok = false;
    cPt2dr aP0 = ToR(aPt);

    double aDist =  Norm2(aP0-mCentroid);
    if (aDist==0) return aP0;
    cPt2dr aDir = (aP0-mCentroid) /aDist;
    tREAL8 aGrFr = (aSeed.mBlack+aSeed.mWhite)/2.0;

    cGetPts_ImInterp_FromValue<tREAL4> aGPV(mDIm,aGrFr,0.1,aP0,aDir);

    Ok = aGPV.Ok();
    if (Ok) return aGPV.PRes();

    return cPt2dr(-1e10,1e20);
}

bool  cExtract_BW_Target::AnalyseOneConnectedComponents(cSeedBWTarget & aSeed)
{
     cPt2di aPTest(-99999999,594);

     mCentroid = cPt2dr(0,0);
     mPtsCC.clear();
     cPt2di aP0 = aSeed.mPixW;
     mPSup = aP0;
     mPInf = aP0;

     // if the point has been explored or is in border 
     if (! MarqFree(aP0)) 
     {
        aSeed.mOk = false;
        return false;
     }

     mIndCurPts = 0;
     AddPtInCC(aP0);

     double aPdsW =  0.5;

     // Minimal value for being white, with P=0.5 => average
     tREAL4 aVMinW =  (1-aPdsW)* aSeed.mBlack +  aPdsW*aSeed.mWhite; 
     // Maximal value for being white, symetric formula
     tREAL4 aVMaxW =  (-aPdsW)* aSeed.mBlack +  (1+aPdsW)*aSeed.mWhite;  

     size_t aMaxNbPts = mPBWT.NbMaxPtsCC();  // if we want to avoid big area
     std::vector<cPt2di> aV4Neigh =  AllocNeighbourhood<2>(1); 

     bool touchOther = false;
     // Classical  CC loop + stop condition on number of points
     while (   (mIndCurPts!=int(mPtsCC.size())) && (mPtsCC.size()<aMaxNbPts)   )
     {
           cPt2di aPix = mPtsCC.at(mIndCurPts);  // extract last point in the heap
           for (const auto & aNeigh : aV4Neigh)   // explorate its neighboord
           {
               cPt2di aPN = aPix + aNeigh;
               if (MarqFree(aPN))  // they have not been met
               {
                   tElemIm aValIm = mDIm.GetV(aPN);
		   if ((aValIm>=aVMinW)  && (aValIm<=aVMaxW))  // if their value is in the good interval
                   {
                      if (mDMasqT.GetV(aPN))
                         aSeed.mMarked4Test = true;
                      AddPtInCC(aPN);
		      if (aPN== aPTest)
		      {
                         aSeed.mMarked4Test = true;
		      }
                   }
               }
	       else if (! MarqEq(aPN,eEEBW_Lab::eTmp))
                    touchOther = true;
           }
           mIndCurPts++;
     }

     if ((mPtsCC.size() >= aMaxNbPts) || touchOther  || (int(mPtsCC.size()) < mPBWT.NbMinPtsCC()))
     {            
        CC_SetMarq(eEEBW_Lab::eBadZ); 
        return false;
     }

     mCentroid = mCentroid / double(mIndCurPts);
     aSeed.mPInf = mPInf;
     aSeed.mPSup = mPSup;
     return true;
}

bool  cExtract_BW_Target::ComputeFrontier(cSeedBWTarget & aSeed)
{
     std::vector<cPt2di> aV8Neigh =  AllocNeighbourhood<2>(2);
     mVFront.clear();
     int aNbOk = 0;
     int aNbFront = 0;
     for (const auto & aPix : mPtsCC)
     {
          bool HasNeighFree=false;
	  for (const auto & aN : aV8Neigh)
              if (MarqFree(aPix+aN))
		 HasNeighFree = true;

	  if (HasNeighFree)
	  {
              aNbFront ++;
              bool Ok;
              cPt2dr aPFr = RefineFrontierPoint(aSeed,aPix,Ok);
	      if (Ok)
	      {
                  aNbOk++;
                  mVFront.push_back(aPFr);
	      }
	  }
     }

     if (aNbOk<mPBWT.mNbMinFront)
     {
        CC_SetMarq(eEEBW_Lab::eBadFr); 
	return false;
     }
     double aProp = aNbOk / double(aNbFront);
     if ( aProp < mPBWT.mPropFr)
     {
        CC_SetMarq(eEEBW_Lab::eBadFr); 
	return false;
     }

     return true;
}

/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Ellipse                            */
/*                                                              */
/*  *********************************************************** */

struct cExtracteEllipse
{
     public :
        cSeedBWTarget    mSeed;
	cEllipse         mEllipse;

	cExtracteEllipse(const cSeedBWTarget& aSeed,const cEllipse & anEllipse);

	tREAL8  mDist;
	tREAL8  mDistPond;
	tREAL8  mEcartAng;
	bool    mValidated;
};

cExtracteEllipse::cExtracteEllipse(const cSeedBWTarget& aSeed,const cEllipse & anEllipse) :
    mSeed      (aSeed),
    mEllipse   (anEllipse),
    mDist      (10.0),
    mDistPond  (10.0),
    mEcartAng  (10.0),
    mValidated (false)
{
}


/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Ellipse                            */
/*                                                              */
/*  *********************************************************** */

class cExtract_BW_Ellipse  : public cExtract_BW_Target
{
	public :
             cExtract_BW_Ellipse(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest);

             void AnalyseAllConnectedComponents(const std::string & aNameIm);
             bool AnalyseEllipse(cSeedBWTarget & aSeed,const std::string & aNameIm);
	private :

	     std::list<cExtracteEllipse> mListExtEl;
};

cExtract_BW_Ellipse::cExtract_BW_Ellipse(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest) :
	cExtract_BW_Target(anIm,aPBWT,aMasqTest)
{
}

void cExtract_BW_Ellipse::AnalyseAllConnectedComponents(const std::string & aNameIm)
{
    for (auto & aSeed : mVSeeds)
    {
        if (AnalyseOneConnectedComponents(aSeed))
        {
            if (ComputeFrontier(aSeed))
	    {
                AnalyseEllipse(aSeed,aNameIm);
	    }
        }
    }
}

bool  cExtract_BW_Ellipse::AnalyseEllipse(cSeedBWTarget & aSeed,const std::string & aNameIm)
{
     cEllipse_Estimate anEEst(mCentroid);
     for (const auto  & aPFr : mVFront)
         anEEst.AddPt(aPFr);
     cEllipse anEl = anEEst.Compute();
     if (! anEl.Ok())
     {
        CC_SetMarq(eEEBW_Lab::eElNotOk); 
        return false;
     }
     double aSomD = 0;
     double aSomRad = 0;
     tREAL8 aGrFr = (aSeed.mBlack+aSeed.mWhite)/2.0;
     for (const auto  & aPFr : mVFront)
     {
         aSomD += std::abs(anEl.ApproxSigneDist(aPFr));
	 aSomRad += std::abs(mDIm.GetVBL(aPFr)-aGrFr);
     }

     aSomD /= mVFront.size();

     tREAL8 aSomDPond =  aSomD / (1+anEl.RayMoy()/50.0);

     int aNbPts = round_ni(4*(anEl.LGa()+anEl.LSa()));
     tREAL8 aSomTeta = 0.0;
     for (int aK=0 ; aK<aNbPts ; aK++)
     {
            double aTeta = (aK * 2.0 * M_PI) / aNbPts;
	    cPt2dr aGradTh;
	    cPt2dr aPt = anEl.PtAndGradOfTeta(aTeta,aGradTh);

	    if (! mDGx.InsideBL(aPt))
	    {
                  CC_SetMarq(eEEBW_Lab::eElNotOk); 
                  return false;
	    }
            cPt2dr aGradIm (mDGx.GetVBL(aPt),mDGy.GetVBL(aPt));
	    aSomTeta += std::abs(ToPolar(aGradIm/-aGradTh).y());
     }
     aSomTeta /= aNbPts;

     if (aSomDPond>0.2)
     {
        CC_SetMarq(eEEBW_Lab::eBadEl); 
	return false;
     }
     

     cExtracteEllipse  anEE(aSeed,anEl);
     anEE.mDist      = aSomD;
     anEE.mDistPond  = aSomDPond;
     anEE.mEcartAng  = aSomTeta;

     if (aSomDPond>0.1)
     {
         CC_SetMarq(eEEBW_Lab::eAverEl); 
     }
     else
     {
         if (aSomTeta>0.05)
	 {
            CC_SetMarq(eEEBW_Lab::eBadTeta); 
	 }
	 else
	 {
            anEE.mValidated = true;
	 }
     }

     mListExtEl.push_back(anEE);
     return true;
}

void  ShowEllipse(const cExtracteEllipse & anEE,const std::string & aNameIm,const std::vector<cPt2dr> & aVFront)
{
    static int aCptIm = 0;
    aCptIm++;
    const cSeedBWTarget &  aSeed = anEE.mSeed;
    const cEllipse &       anEl  = anEE.mEllipse;

     StdOut() << "SOMDDd=" << anEE.mDist << " DP=" << anEE.mDistPond << " " << aSeed.mPixTop 
		 << " NORM =" << anEl.Norm()  << " RayM=" << anEl.RayMoy()
		 << "\n";

     /*
        std::vector<cPt3dr> aVF3;
        for (const auto  & aP : mVFront)
	{
             aVF3.push_back(cPt3dr(aP.x(),aP.y(),ToPolar(aP-mCentroid).y()));
	}
        std::sort
        (
	    aVF3.begin(),
	    aVF3.end(),
            [](const cPt3dr  aP1,const cPt3dr &  aP2) {return aP1.z() >aP2.z();}
        );
	*/

        StdOut() << " BOX " << aSeed.mPInf << " " << aSeed.mPSup << "\n";

	cPt2di  aPMargin(6,6);
	cBox2di aBox(aSeed.mPInf-aPMargin,aSeed.mPSup+aPMargin);

	int aZoom = 21;
	cRGBImage aRGBIm = cRGBImage::FromFile(aNameIm,aBox,aZoom);  ///< Allocate and init from file
	cPt2dr aPOfs = ToR(aBox.P0());
	/*
	int aNbPts = 5000;
	for (int aK=0 ; aK<aNbPts ; aK++)
	{
            double aTeta = (aK * 2.0 * M_PI) / aNbPts;
	    cPt2dr aP1 = anEl.PtOfTeta(aTeta);
	    tREAL8 aEps=1e-3;
	    cPt2dr aP0 = anEl.PtOfTeta(aTeta - aEps );
	    cPt2dr aP2 = anEl.PtOfTeta(aTeta + aEps );

	    cPt2dr aGrad = VUnit((aP2-aP0) / aEps)/cPt2dr(0,1);
            //  aRGBIm.SetRGBPoint(aPT-aPOfs,cRGBImage::Green);
	    cPt2dr aG3;
            cPt2dr aP3 =   anEl.PtAndGradOfTeta(aTeta,aG3);

	    // StdOut() << " Pt" << Norm2(aP1-aP3)  << " Gd "<< Norm2(aGrad-aG3)  << aGrad << aG3<< "\n";
	}
	*/

	cPt2dr aCenter = anEl.Center() - aPOfs;
	aCenter = aRGBIm.PointToRPix(aCenter);
	std::vector<cPt2di> aVPts;
	GetPts_Ellipse(aVPts,aCenter,anEl.LGa()*aZoom,anEl.LSa()*aZoom,anEl.TetaGa(),true);
	for (const auto & aPix : aVPts)
	{
            aRGBIm.RawSetPoint(aPix,cRGBImage::Blue);
	}

        for (const auto  & aPFr : aVFront)
	{
            aRGBIm.SetRGBPoint(aPFr-aPOfs,cRGBImage::Red);
	    //StdOut() <<  "DDDD " <<  anEl.ApproxSigneDist(aPFr) << "\n";
	}

	aRGBIm.ToFile("VisuDetectCircle_"+ ToStr(aCptIm) + ".tif");
#if (0)
#endif
}

/*  *********************************************************** */
/*                                                              */
/*             cAppliExtractCodeTarget                          */
/*                                                              */
/*  *********************************************************** */

class cAppliExtractCircTarget : public cMMVII_Appli,
	                        public cAppliParseBoxIm<tREAL4>
{
     public :
        typedef tREAL4              tElemIm;
        typedef cDataIm2D<tElemIm>  tDataIm;
        typedef cImGrad<tElemIm>    tImGrad;


        cAppliExtractCircTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        int ExeOnParsedBox() override;

        bool            mVisu;
        cExtract_BW_Ellipse * mExtrEll;
        cParamBWTarget  mPBWT;

        tImGrad         mImGrad;
        cRGBImage       mImVisu;
        cIm2D<tU_INT1>  mImMarq;

        std::vector<cSeedBWTarget> mVSeeds;
	cPhotogrammetricProject     mPhProj;

	bool                        mHasMask;
	std::string                 mNameMask;
};



cAppliExtractCircTarget::cAppliExtractCircTarget
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli  (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>(*this,true,cPt2di(10000,10000),cPt2di(300,300),false) ,
   mVisu    (true),
   mExtrEll (nullptr),
   mImGrad  (cPt2di(1,1)),
   mImVisu  (cPt2di(1,1)),
   mImMarq  (cPt2di(1,1)),
   mPhProj  (*this)

{
}

        // cExtract_BW_Target * 
cCollecSpecArg2007 & cAppliExtractCircTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   // Standard use, we put args of  cAppliParseBoxIm first
   return
         APBI_ArgObl(anArgObl)
                   //  << AOpt2007(mDiamMinD, "DMD","Diam min for detect",{eTA2007::HDV})
   ;
}

cCollecSpecArg2007 & cAppliExtractCircTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return APBI_ArgOpt
          (
                anArgOpt
             << mPhProj.DPMask().ArgDirInOpt("TestMask","Mask for selecting point used in detailed mesg/output")
             << AOpt2007(mPBWT.mMinDiam,"DiamMin","Minimum diameters for ellipse",{eTA2007::HDV})
             << AOpt2007(mPBWT.mMaxDiam,"DiamMax","Maximum diameters for ellipse",{eTA2007::HDV})
          );
}



int cAppliExtractCircTarget::ExeOnParsedBox()
{
   double aT0 = SecFromT0();

   mExtrEll = new cExtract_BW_Ellipse(APBI_Im(),mPBWT,mPhProj.MaskWithDef(mNameIm,CurBoxIn(),false));
   if (mVisu)
      mImVisu =  cRGBImage::FromFile(mNameIm,CurBoxIn());

   double aT1 = SecFromT0();
   mExtrEll->ExtractAllSeed();
   double aT2 = SecFromT0();
   mExtrEll->AnalyseAllConnectedComponents(mNameIm);
   double aT3 = SecFromT0();

   StdOut() << "TIME-INIT " << aT1-aT0 << "\n";
   StdOut() << "TIME-SEED " << aT2-aT1 << "\n";
   StdOut() << "TIME-CC   " << aT3-aT2 << "\n";


   if (mVisu)
   {
       const cExtract_BW_Target::tDImMarq&     aDMarq =  mExtrEll->DImMarq();
       for (const auto & aPix : aDMarq)
       {
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eTmp))
               mImVisu.SetRGBPix(aPix,cRGBImage::Green);

            if (     (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadZ))
                  || (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eElNotOk))
	       )
               mImVisu.SetRGBPix(aPix,cRGBImage::Blue);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadFr))
               mImVisu.SetRGBPix(aPix,cRGBImage::Cyan);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadEl))
               mImVisu.SetRGBPix(aPix,cRGBImage::Red);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eAverEl))
               mImVisu.SetRGBPix(aPix,cRGBImage::Orange);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadTeta))
               mImVisu.SetRGBPix(aPix,cRGBImage::Yellow);
	       /*
	       */

	       /*
	    cPt3di aRGB =  mImVisu.GetRGBPix(aPix);
	    if ( mExtrEll->DGx().GetV(aPix) >0)
	        aRGB[2] =    aRGB[2] * 0.9;
	    else
	        aRGB[2] =   255 - (255- aRGB[2]) * 0.9;
            mImVisu.SetRGBPix(aPix,aRGB);
	    */

	    // if (mGrad.mGy.DIm().GetV(aPix) == 0)
       }

       for (const auto & aSeed : mExtrEll->VSeeds())
       {
           if (aSeed.mOk)
	   {
              mImVisu.SetRGBPix(aSeed.mPixW,cRGBImage::Red);
              mImVisu.SetRGBPix(aSeed.mPixTop,cRGBImage::Yellow);
	   }
	   else
	   {
              mImVisu.SetRGBPix(aSeed.mPixW,cRGBImage::Yellow);
	   }
       }
   }


   delete mExtrEll;
   if (mVisu)
      mImVisu.ToFile("TTTT.tif");

   return EXIT_SUCCESS;
}



int  cAppliExtractCircTarget::Exe()
{
   mPhProj.FinishInit();

   mHasMask =  mPhProj.ImageHasMask(APBI_NameIm()) ;
   if (mHasMask)
      mNameMask =  mPhProj.NameMaskOfImage(APBI_NameIm());


   APBI_ExecAll();  // run the parse file  SIMPL

   StdOut() << "MAK=== " <<   mHasMask << " " << mNameMask  << "\n";

   return EXIT_SUCCESS;
}

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_ExtractCircTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliExtractCircTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractCircTarget
(
     "CodedTargetCircExtract",
      Alloc_ExtractCircTarget,
      "Extract coded target from images",
      {eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);


};

