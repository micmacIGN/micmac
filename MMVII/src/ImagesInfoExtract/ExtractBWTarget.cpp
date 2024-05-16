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


int cParamBWTarget::NbMaxPtsCC() const { return M_PI * Square(mMaxDiam/2.0); }
int cParamBWTarget::NbMinPtsCC() const { return M_PI * Square(mMinDiam/2.0); }


cParamBWTarget::cParamBWTarget() :
    mFactDeriche    (2.0),
    mD0BW           (2),
    mValMinW        (20), 
    mValMaxB        (100),
    mRatioMaxBW     (1/1.5),
    mMinDiam        (15.0),
    mMaxDiam        (100.0),
    mPropFr         (0.95),
    mNbMinFront     (10),
    mDistMinMaxLoc  (15.0),
    mInvGray        (false)
{
}

void cParamBWTarget::SetMax4Inv(tREAL4 aMaxGray)
{
   mInvGray = true;
   mMaxGray = aMaxGray;
}

tREAL8    cParamBWTarget::CorrecRad(tREAL8 aGray) const
{
	return mInvGray ? (mMaxGray-aGray) : aGray;
}

tREAL8    cParamBWTarget::RatioBW(tREAL8 aBlack,tREAL8 aWhite) const
{
    if (mInvGray)  return CorrecRad(aWhite) / std::max(1.0,CorrecRad(aBlack));

    return aBlack / std::max(1.0,aWhite);
}



/*  *********************************************************** */
/*                                                              */
/*               cSeedBWTarget                                  */
/*                                                              */
/*  *********************************************************** */


cSeedBWTarget::cSeedBWTarget(const cPt2di & aPixW,const cPt2di & aPixTop,tREAL4 aBlack,tREAL4 aWhite):
   mPixW        (aPixW),
   mPixTop      (aPixTop),
   mBlack       (aBlack),
   mWhite       (aWhite),
   mOk          (true),
   mMarked4Test (false)
{
}

tREAL4  cSeedBWTarget::Contrast() const {return  mWhite - mBlack;}
tREAL4  cSeedBWTarget::Avg()      const {return (mWhite + mBlack)/2.0;}


/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Target                             */
/*                                                              */
/*  *********************************************************** */




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

cPt2di cExtract_BW_Target::Prolongate(cPt2di aPix,bool IsW) const
{
    cPt2di aDir = cPt2di(0,IsW?1:-1);

    while
    ( 
            MarqFree(aPix+aDir) 
	&&  (  IsW == (mDIm.GetV(aPix+aDir) >mDIm.GetV(aPix)))
    )
    {
        aPix = aPix + aDir;
    }

    return aPix;
}
         
bool cExtract_BW_Target::IsExtremalPoint(const cPt2di & aPix) 
{
   cPt2di aPixB =  Prolongate(aPix,false);
   tElemIm aVBlack =  mDIm.GetV(aPixB);

   cPt2di aPixW =  Prolongate(aPix,true);
   tElemIm aVWhite =  mDIm.GetV(aPixW);

   tElemIm aBlackInit = mPBWT.mInvGray ?  mPBWT.CorrecRad(aVWhite) : aVBlack;
   tElemIm aWhiteInit = mPBWT.mInvGray ?  mPBWT.CorrecRad(aVBlack) : aVWhite;


   if (aWhiteInit < mPBWT.mValMinW)
      return false;

   if (aBlackInit > mPBWT.mValMaxB)
      return false;

    if (aBlackInit > (mPBWT.mRatioMaxBW * aWhiteInit))
      return false;

   mVSeeds.push_back(cSeedBWTarget(aPixW,aPix,aVBlack,aVWhite));

   return true;
}

void cExtract_BW_Target::ExtractAllSeed()
{
    /* 1 : Extract "pre-seed" point = a point where x-grad change sign
     *     and for which y-grad is maximal among them
     */
    cResultExtremum  aRExtr(false,true);
    {
       // 1.0 prepare data
       tIm       aIExtre(mSz,nullptr,eModeInitImage::eMIA_Null);
       tDataIm&  aDExtr (aIExtre.DIm());

       const cBox2di &  aFullBox = mDIm;
       cRect2  aBoxInt (aFullBox.Dilate(-mPBWT.mD0BW));

       // 1.1  make the image of y-grad for changing sign pixel
       for (const auto & aPix : aBoxInt)
       {
          if ( (mDGx.GetV(aPix)<=0) &&  (mDGx.GetV(aPix+cPt2di(-1,0)) >0) )
	  {
              tREAL4 aGy = mDGy.GetV(aPix);
	      if (aGy >0)  // we wan to point so gradient must be positive
	      {
		  aDExtr.SetV(aPix,aGy);
              }
	  }
       }
 
       // 1.2 extract extremum
       ExtractExtremum1(aDExtr,aRExtr,mPBWT.mDistMinMaxLoc);  
    }

    /*  2 extract Extremal point */

    for (const auto & aPix : aRExtr.mPtsMax)
    {
          if (IsExtremalPoint(aPix))  // by border-effect point are added in mVSeeds
          {
          }
    }

    /* 3 now sort the point, privilegiate the max contrast, the idea is that "real" target are pure white a black */ 
   SortOnCriteria(mVSeeds, [](const cSeedBWTarget &  aSeed) {return - aSeed.Contrast();});
}

const std::vector<cSeedBWTarget> &      cExtract_BW_Target::VSeeds() const { return mVSeeds; }
const cExtract_BW_Target::tDImMarq&    cExtract_BW_Target::DImMarq() const {return mDImMarq;}
const cExtract_BW_Target::tDataIm&    cExtract_BW_Target::DGx() const {return mDGx;}
const cExtract_BW_Target::tDataIm&    cExtract_BW_Target::DGy() const {return mDGy;}

void cExtract_BW_Target::AddPtInCC(const cPt2di & aPt,tREAL8 aW)
{
     SetMarq(aPt,eEEBW_Lab::eTmp);

     mPtsCC.push_back(aPt);
     mCentroid = mCentroid + ToR(aPt)*aW;
     mWCenter += aW;

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

    cGetPts_ImInterp_FromValue<tREAL4> aGPV(mDIm,aGrFr,0.1,aP0,aDir);  ///  (aSeed.mMarked4Test);

    Ok = aGPV.Ok();
    if (Ok) return aGPV.PRes();

    return cPt2dr(-1e10,1e20);
}

bool  cExtract_BW_Target::AnalyseOneConnectedComponents(cSeedBWTarget & aSeed)
{
     cPt2di aPTest(-99999999,594);

     mCentroid = cPt2dr(0,0);
     mWCenter  = 0.0;
     mPtsCC.clear();
     cPt2di aP0 = aSeed.mPixW;
     mPSup = aP0;
     mPInf = aP0;

     // if the point has been explored or is in border 
     if (! MarqFree(aP0)) 
     {
        if (aSeed.mMarked4Test)
           StdOut() << "###  For Marked point " << aSeed.mPixW << " Point has already been explored   ###" << std::endl;

        aSeed.mOk = false;
        return false;
     }

     mIndCurPts = 0;

     double aPdsW =  0.5;

     // Minimal value for being white, with P=0.5 => average
     tREAL4 aVMinW =  (1-aPdsW)* aSeed.mBlack +  aPdsW*aSeed.mWhite; 
     // Maximal value for being white, symetric formula
     tREAL4 aVMaxW =  (-aPdsW)* aSeed.mBlack +  (1+aPdsW)*aSeed.mWhite;  

     size_t aMaxNbPts = mPBWT.NbMaxPtsCC();  // if we want to avoid big area
     std::vector<cPt2di> aV4Neigh =  AllocNeighbourhood<2>(1); 

     AddPtInCC(aP0,mDIm.GetV(aP0)-aVMinW);
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
                      AddPtInCC(aPN,aValIm-aVMinW);
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
        if (aSeed.mMarked4Test)
	{
             if (mPtsCC.size() >= aMaxNbPts)
                 StdOut() << "====> ###  For Marked point " << aSeed.mPixW << " too many points:" << mPtsCC.size() << " ###" << std::endl;
             if (touchOther)
                 StdOut() << "====> ###  For Marked point " << aSeed.mPixW << " touchOther "  << " ###" << std::endl;
             if (int(mPtsCC.size()) < mPBWT.NbMinPtsCC())
                 StdOut() << "====> ###  For Marked point " << aSeed.mPixW << " not enough point:" << mPtsCC.size() << " ###" << std::endl;
	}

        CC_SetMarq(eEEBW_Lab::eBadZ); 
        return false;
     }

     mCentroid = mCentroid / mWCenter;
     aSeed.mPInf = mPInf;
     aSeed.mPSup = mPSup;
     return true;
}

bool  cExtract_BW_Target::ComputeFrontier(cSeedBWTarget & aSeed)
{
     std::vector<cPt2di> aV8Neigh =  AllocNeighbourhood<2>(2);
     std::vector<cPt2di> aV4Neigh =  AllocNeighbourhood<2>(1);


     // [A]  First we must compute frontier with a separation between exterior and other in
     // cas there is whole

     std::vector<cPt2di> aVecFrExt;  // point of exterior frontier
     cWhichMin<cPt2di,int>  aCompExtre; // for storing the extremal point, this one we are sure is on external frontier
     //  A1 marq all front point  with label eFront
     for (const auto & aPix : mPtsCC)
     {
              bool HasNeighFree=false;
	      for (const auto & aN : aV8Neigh)
                  if (MarqFree(aPix+aN))
		     HasNeighFree = true;
	      if (HasNeighFree)
	      {
                 SetMarq(aPix,eEEBW_Lab::eFront);
		 aCompExtre.Add(aPix,aPix.x());
	      }
     }
     cPt2di aPExtre = aCompExtre.IndexExtre();

     // a little check   StdOut() << int(eEEBW_Lab::eFront) << " " << int (GetMarq(aPExtre)) << std::endl;
     MMVII_INTERNAL_ASSERT_tiny(eEEBW_Lab::eFront==GetMarq(aPExtre),"Chekc in front ext");

     // A2 now search eFront point connected to seed 
     SetMarq(aPExtre,eEEBW_Lab::eFrontExt);
     aVecFrExt.push_back(aPExtre);


     //  now put in "aVecFrExt"  the point, marked frontier, connected to 
     size_t aIndCur = 0;
     while (aIndCur !=aVecFrExt.size())
     {
         cPt2di aPix = aVecFrExt.at(aIndCur);  // extract last point in the heap
         for (const auto & aNeigh : aV4Neigh)   // explorate its neighboord
         {
             cPt2di aPN = aPix + aNeigh;
             if (GetMarq(aPN) == eEEBW_Lab::eFront)  // they have not been met
             {
	        SetMarq(aPN,eEEBW_Lab::eFrontExt);
                aVecFrExt.push_back(aPN);
	     }
	 }
	 aIndCur++;
     }

     // [B]  now  refine the frontier to have a sub pixel estimation
     mVFront.clear();
     int aNbOk = 0;
     int aNbFront = 0;

     for (const auto & aPix: aVecFrExt)
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

     //  [C]  some test of validation 

     if (aNbOk<mPBWT.mNbMinFront)
     {
        CC_SetMarq(eEEBW_Lab::eBadFr); 
	if (aSeed.mMarked4Test)
	{
            StdOut() << "---### Not enough point for frontier : " << aNbOk << "### " << std::endl;
	}
	return false;
     }
     double aProp = aNbOk / double(aNbFront);
     if ( aProp < mPBWT.mPropFr)
     {
        CC_SetMarq(eEEBW_Lab::eBadFr); 
	if (aSeed.mMarked4Test)
	{
            StdOut() << "---### Proportion frontier-point ok to low : " << aProp << "### " << std::endl;
	}
	return false;
     }

     return true;
}


};

