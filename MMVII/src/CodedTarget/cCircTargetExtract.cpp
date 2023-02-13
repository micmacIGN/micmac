#include "MMVII_Tpl_Images.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Geom2D.h"

/*   Modularistion
 *   Code extern tel que ellipse
 *   Ellipse => avec centre
 *   Pas de continue
 */

namespace MMVII
{

static bool TEST = false;


/*  *********************************************************** */
/*                                                              */
/*                   cParamCircTarg                             */
/*                                                              */
/*  *********************************************************** */

struct cParamCircTarg
{
    public :
      cParamCircTarg();

      int NbMaxPtsCC() const;
      int NbMinPtsCC() const;

      double    mFactDeriche;
      int       mD0BW;
      double    mValMinW;
      double    mValMaxB;
      double    mRatioMaxBW;
      double    mMinDiam;
      double    mMaxDiam;
      double    mPropFr;
};


int cParamCircTarg::NbMaxPtsCC() const { return M_PI * Square(mMaxDiam/2.0); }
int cParamCircTarg::NbMinPtsCC() const { return M_PI * Square(mMinDiam/2.0); }


cParamCircTarg::cParamCircTarg() :
    mFactDeriche (2.0),
    mD0BW        (2),
    mValMinW     (20), 
    mValMaxB     (100),
    mRatioMaxBW  (1/1.5),
    mMinDiam     (7.0),
    mMaxDiam     (60.0),
    mPropFr      (0.95)
{
}

/*  *********************************************************** */
/*                                                              */
/*               cEllipseEstimate                               */
/*                                                              */
/*  *********************************************************** */

//    (X-C) ^2 /R2 = 1
//    R2 =  A^2 + 2B^2 + C^2
//
//   ( (X-C)^2/R2 -1) 

class cEllipse
{
     public :
       cEllipse(cDenseVect<tREAL8> aDV,const cPt2dr & aC0);
       double SignedD2(cPt2dr aP) const;
       double Dist(const cPt2dr & aP) const;
       double   Norm() const  {return std::sqrt(1/ mNorm);}

     private :
       cDenseVect<tREAL8> mV;
       double             mNorm;
       cPt2dr             mC0;
};

cEllipse::cEllipse(cDenseVect<tREAL8> aDV,const cPt2dr & aC0) :
    mV    (aDV.Dup()),
    mNorm (std::sqrt(Square(mV(0)) + 2 * Square(mV(1))  + Square(mV(2)))),
    mC0   (aC0)
{
}

double cEllipse::SignedD2(cPt2dr aP) const
{
     aP = aP-mC0;
     tREAL8 x = aP.x();
     tREAL8 y = aP.y();
     tREAL8 aRes =   mV(0)*x*x  + mV(1)*x*y + mV(2)*y*y + mV(3)*x+mV(4)*y -1;

     return aRes / mNorm;
}
double cEllipse::Dist(const cPt2dr & aP) const {return std::sqrt(std::abs(SignedD2(aP)));}


class cEllipse_Estimate
{
//  A X2 + BXY + C Y2 + DX + EY = 1
      public :
        cLeasSqtAA<tREAL8> & Sys() {return mSys;}

	// indicate a rough center, for better numerical accuracy
	cEllipse_Estimate(const cPt2dr & aC0);
	void AddPt(cPt2dr aP) ;

	cEllipse Compute() ;
      private :
         cLeasSqtAA<tREAL8> mSys;
	 cPt2dr             mC0;
};

cEllipse_Estimate::cEllipse_Estimate(const cPt2dr & aC0) :
    mSys  (5),
    mC0   (aC0)
{
}

void cEllipse_Estimate::AddPt(cPt2dr aP) 
{
     aP = aP-mC0;

     cDenseVect<tREAL8> aDV(5);
     aDV(0) = Square(aP.x());
     aDV(1) = aP.x() * aP.y();
     aDV(2) = Square(aP.y());
     aDV(3) = aP.x();
     aDV(4) = aP.y();

     mSys.AddObservation(1.0,aDV,1.0);
}

cEllipse cEllipse_Estimate::Compute() {return cEllipse(mSys.Solve(),mC0);}

/*  *********************************************************** */
/*                                                              */
/*               cSeedCircTarg                                  */
/*                                                              */
/*  *********************************************************** */

struct cSeedCircTarg
{
    public :
       cPt2di mPixW;
       cPt2di mPixTop;

       tREAL4 mBlack;
       tREAL4 mWhite;
       bool   mOk;

       cSeedCircTarg(const cPt2di & aPixW,const cPt2di & aPixTop,  tREAL4 mBlack,tREAL4 mWhite);
};

cSeedCircTarg::cSeedCircTarg(const cPt2di & aPixW,const cPt2di & aPixTop,tREAL4 aBlack,tREAL4 aWhite):
   mPixW    (aPixW),
   mPixTop  (aPixTop),
   mBlack   (aBlack),
   mWhite   (aWhite),
   mOk      (true)
{
}

/*  *********************************************************** */
/*                                                              */
/*               cGetPts_ImInterp_FromValue                     */
/*                                                              */
/*  *********************************************************** */

template <class Type> class cGetPts_ImInterp_FromValue
{
     public :
       cGetPts_ImInterp_FromValue (const cDataIm2D<Type> & aDIm,tREAL8 aVal,tREAL8 aTol,cPt2dr aP0,const cPt2dr & aDir):
	        mOk          (false),
		mDIm         (aDIm),
                mMaxIterInit (10),
                mMaxIterEnd  (20)
       {
          tREAL8 aV0 = GetV(aP0);
	  if (!CheckNoVal(aV0))  return; 
	  mP0IsSup = (aV0>=aVal);

          cPt2dr aP1 = aP0 + aDir;
          double aV1 = GetV(aP1);
	  if (!CheckNoVal(aV1))  return; 

	  int aNbIter=0;
          while ( (aV1>=aVal)==mP0IsSup )
          {
                aV0 = aV1;
		aP0 = aP1;
		aP1 += aDir;
                aV1 = GetV(aP1);
                if (!CheckNoVal(aV1))  return; 
		aNbIter++;
		if (aNbIter>mMaxIterInit) return;
          }

	  tREAL8 aTol0 = std::abs(aV0-aVal);
	  tREAL8 aTol1 = std::abs(aV1-aVal);
	  bool  InInterv = true;
	  aNbIter=0;
          while ((aTol0>aTol) && (aTol1>aTol) && InInterv  && (aNbIter<mMaxIterEnd))
          {
               aNbIter++;
               cPt2dr aNewP =  Centroid(aTol1,aP0,aTol0,aP1);
               tREAL8 aNewV = GetV(aNewP);
               if (!CheckNoVal(aNewV))  return; 
	       if (   ((aNewV<=aV0) != mP0IsSup) || ((aNewV>=aV1) != mP0IsSup) )
	       {
		    InInterv = false;
	       }
	       else
	       {
                    if (   (aNewV<aVal) == mP0IsSup)   //  V1  <  NewV  < Val  < V0  or  V0 < Val  aNewV  < V1
		    {
                        aV1 = aNewV;
			aTol1 = std::abs(aV1-aVal);
			aP1 = aNewP;
		    }
		    else
		    {
                        aV0 = aNewV;
			aTol0 = std::abs(aV0-aVal);
			aP0 = aNewP;
		    }
               }
          }
          mPRes = (aTol0<aTol1) ? aP0 : aP1;
	  mOk = true;
       }

       bool Ok() const {return mOk;}
       const cPt2dr & PRes() const 
       {
            MMVII_INTERNAL_ASSERT_tiny(mOk,"Try to get cGetPts_ImInterp_FromValue::PRes() w/o success");
            return mPRes;
       }

     private :

       inline tREAL8 GetV(const cPt2dr & aP) {return mDIm.DefGetVBL(aP,NoVal);}
       inline bool CheckNoVal(const tREAL8 aV)
       {
           if (aV==NoVal)
           {
	       return false;
           }
	   return true;
       }

       bool  mP0IsSup;
       static constexpr tREAL8 NoVal= -1e10;
       bool   mOk;
       const cDataIm2D<Type> & mDIm;
       int    mMaxIterInit;
       int    mMaxIterEnd;
       cPt2dr mPRes;
};

/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Ellipse                            */
/*                                                              */
/*  *********************************************************** */

enum class eEEBW_Lab : tU_INT1
{
   eFree,
   eBorder,
   eTmp,
   eBadZ,
   eBadFr,
   eBadEl,
   eAverEl
};


class cExtract_BW_Ellipse
{
   public :
        typedef tREAL4              tElemIm;
        typedef cDataIm2D<tElemIm>  tDataIm;
        typedef cIm2D<tElemIm>      tIm;
        typedef cImGrad<tElemIm>    tImGrad;

	typedef cIm2D<tU_INT1>      tImMarq;
	typedef cDataIm2D<tU_INT1>  tDImMarq;

        cExtract_BW_Ellipse(tIm anIm,const cParamCircTarg & aPCT);

        void ExtractAllSeed();
        void AnalyseAllConnectedComponents();
        const std::vector<cSeedCircTarg> & VSeeds() const;
	const tDImMarq&    DImMarq() const;
	const tDataIm &    DGx() const;
	const tDataIm &    DGy() const;

	void SetMarq(const cPt2di & aP,eEEBW_Lab aLab) {mDImMarq.SetV(aP,tU_INT1(aLab));}
	bool MarqEq(const cPt2di & aP,eEEBW_Lab aLab) const {return mDImMarq.GetV(aP) == tU_INT1(aLab);}
	bool MarqFree(const cPt2di & aP) const {return MarqEq(aP,eEEBW_Lab::eFree);}

   private :

        bool IsCandidateTopOfEllipse(const cPt2di &) ;
        void AnalyseOneConnectedComponents(cSeedCircTarg &);

	void AddPtInCC(const cPt2di &);
	// Prolongat on the vertical, untill its a max or a min
        cPt2di Prolongate(cPt2di aPix,bool IsW,tElemIm & aMaxGy) const;
        cPt2dr ExtractFrontier(const cSeedCircTarg & aSeed,const cPt2di & aP0,bool & Ok);

        tIm              mIm;
        tDataIm &        mDIm;
	cPt2di           mSz;
	tImMarq          mImMarq;
	tDImMarq&        mDImMarq;
        cParamCircTarg   mPCT;
        tImGrad          mImGrad;
	tDataIm &        mDGx;
	tDataIm &        mDGy;

        std::vector<cSeedCircTarg> mVSeeds;

	std::vector<cPt2di>  mPtsCC;
	int                  mCurPts;
	cPt2dr               mCDG;
};

cExtract_BW_Ellipse::cExtract_BW_Ellipse(tIm anIm,const cParamCircTarg & aPCT) :
   mIm      (anIm),
   mDIm     (mIm.DIm()),
   mSz      (mDIm.Sz()),
   mImMarq  (mSz),
   mDImMarq (mImMarq.DIm()),
   mPCT     (aPCT),
   mImGrad  (Deriche( mDIm,mPCT.mFactDeriche)),
   mDGx     (mImGrad.mGx.DIm()),
   mDGy     (mImGrad.mGy.DIm())
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


///cSeedCircTarg

cPt2di cExtract_BW_Ellipse::Prolongate(cPt2di aPix,bool IsW,tElemIm & aMaxGy) const
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
         
bool cExtract_BW_Ellipse::IsCandidateTopOfEllipse(const cPt2di & aPix) 
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
   if ((aVBlack/double(aVWhite)) > mPCT.mRatioP2)
      return false;
      */

   tElemIm aMaxGy = aGy;
   cPt2di aPixB =  Prolongate(aPix,false,aMaxGy);
   tElemIm aVBlack =  mDIm.GetV(aPixB);

   cPt2di aPixW =  Prolongate(aPix,true,aMaxGy);
   tElemIm aVWhite =  mDIm.GetV(aPixW);

   if (aMaxGy> aGy)
      return false;
   
   if (aVWhite < mPCT.mValMinW)
      return false;

   if (aVBlack > mPCT.mValMaxB)
      return false;

    if ((aVBlack/double(aVWhite)) > mPCT.mRatioMaxBW)
      return false;

   mVSeeds.push_back(cSeedCircTarg(aPixW,aPix,aVBlack,aVWhite));

   return true;
}

void cExtract_BW_Ellipse::ExtractAllSeed()
{
   const cBox2di &  aFullBox = mDIm;
   cRect2  aBoxInt (aFullBox.Dilate(-mPCT.mD0BW));
   int aNb=0;
   int aNbOk=0;
   for (const auto & aPix : aBoxInt)
   {
       aNb++;
       if (IsCandidateTopOfEllipse(aPix))
       {
           aNbOk++;
       }
   }
   std::sort
   (
      mVSeeds.begin(),
      mVSeeds.end(),
      [](const cSeedCircTarg &  aS1,const cSeedCircTarg &  aS2) {return aS1.mWhite>aS2.mWhite;}
   );
   StdOut() << " PPPP="  << aNbOk / double(aNb) << "\n";
}

const std::vector<cSeedCircTarg> &      cExtract_BW_Ellipse::VSeeds() const { return mVSeeds; }
const cExtract_BW_Ellipse::tDImMarq&    cExtract_BW_Ellipse::DImMarq() const {return mDImMarq;}
const cExtract_BW_Ellipse::tDataIm&    cExtract_BW_Ellipse::DGx() const {return mDGx;}
const cExtract_BW_Ellipse::tDataIm&    cExtract_BW_Ellipse::DGy() const {return mDGy;}

void cExtract_BW_Ellipse::AddPtInCC(const cPt2di & aP0)
{
     mDImMarq.SetV(aP0,tU_INT1(eEEBW_Lab::eTmp) );
     mPtsCC.push_back(aP0);
     mCDG = mCDG + ToR(aP0);
}

void cExtract_BW_Ellipse::AnalyseAllConnectedComponents()
{
    for (auto & aSeed : mVSeeds)
        AnalyseOneConnectedComponents(aSeed);
}






cPt2dr cExtract_BW_Ellipse::ExtractFrontier(const cSeedCircTarg & aSeed,const cPt2di & aPt,bool & Ok)
{
    Ok = false;
    cPt2dr aP0 = ToR(aPt);

    double aDist =  Norm2(aP0-mCDG);
    if (aDist==0) return aP0;
    cPt2dr aDir = (aP0-mCDG) /aDist;
    tREAL8 aGrFr = (aSeed.mBlack+aSeed.mWhite)/2.0;

    cGetPts_ImInterp_FromValue<tREAL4> aGPV(mDIm,aGrFr,0.1,aP0,aDir);

    Ok = aGPV.Ok();
    if (Ok) return aGPV.PRes();

    return cPt2dr(-1e10,1e20);
}

void  cExtract_BW_Ellipse::AnalyseOneConnectedComponents(cSeedCircTarg & aSeed)
{
TEST = (
             aSeed.mPixTop==(cPt2di(2666,586))   // 58
          || aSeed.mPixTop==(cPt2di(2669,597))   // 58
	  /*
             aSeed.mPixTop==(cPt2di(3510,3310))  // 44
          || aSeed.mPixTop==(cPt2di(1162,531))   // 15
          || aSeed.mPixTop==(cPt2di(2669,597))   // 58
          || aSeed.mPixTop==(cPt2di(2666,586))   // 58
          || aSeed.mPixTop==(cPt2di(3580,1906))  // 41
          || aSeed.mPixTop==(cPt2di(3581,1906))  // 41
	  */
       );

     mCDG = cPt2dr(0,0);
     mPtsCC.clear();
     cPt2di aP0 = aSeed.mPixW;

     if (! MarqFree(aP0)) 
     {
        aSeed.mOk = false;
        return ;
     }

     mCurPts = 0;
     AddPtInCC(aP0);

     double aPdsW =  0.5;

     tREAL4 aVMin =  (1-aPdsW)* aSeed.mBlack +  aPdsW*aSeed.mWhite;
     tREAL4 aVMax =  (-aPdsW)* aSeed.mBlack +  (1+aPdsW)*aSeed.mWhite;

     size_t aMaxNbPts = mPCT.NbMaxPtsCC();
     std::vector<cPt2di> aV4Neigh =  AllocNeighbourhood<2>(1);


     bool touchOther = false;
     while (   (mCurPts!=int(mPtsCC.size())) && (mPtsCC.size()<aMaxNbPts)   )
     {
           cPt2di aPix = mPtsCC.at(mCurPts);
           for (const auto & aNeigh : aV4Neigh)
           {
               cPt2di aPN = aPix + aNeigh;
               if (MarqFree(aPN))
               {
                   tElemIm aValIm = mDIm.GetV(aPN);
		   if ((aValIm>=aVMin)  && (aValIm<=aVMax))
                      AddPtInCC(aPN);
               }
	       else if (! MarqEq(aPN,eEEBW_Lab::eTmp))
                    touchOther = true;
           }
           mCurPts++;
     }

     if ((mPtsCC.size() >= aMaxNbPts) || touchOther  || (int(mPtsCC.size()) < mPCT.NbMinPtsCC()))
     {
        for (const auto & aP : mPtsCC)
            SetMarq(aP,eEEBW_Lab::eBadZ);
        return;
     }

     mCDG = mCDG / double(mCurPts);

     std::vector<cPt2di> aV8Neigh =  AllocNeighbourhood<2>(2);
     std::vector<cPt2dr> aVFront;
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
              cPt2dr aPFr = ExtractFrontier(aSeed,aPix,Ok);
	      if (Ok)
	      {
                  aNbOk++;
                  aVFront.push_back(aPFr);
	      }
	  }
     }

     double aProp = aNbOk / double(aNbFront);
     if ( aProp < mPCT.mPropFr)
     {
        // StdOut() << "PROP = " << aProp << "\n";
        for (const auto & aP : mPtsCC)
            SetMarq(aP,eEEBW_Lab::eBadFr);
	return ;
     }

     cEllipse_Estimate anEE(mCDG);
     for (const auto  & aPFr : aVFront)
         anEE.AddPt(aPFr);

     cEllipse anEl = anEE.Compute();
     double aSomD = 0;
     double aSomRad = 0;
     tREAL8 aGrFr = (aSeed.mBlack+aSeed.mWhite)/2.0;
     for (const auto  & aPFr : aVFront)
     {
         aSomD += anEl.Dist(aPFr);
	 aSomRad += std::abs(mDIm.GetVBL(aPFr)-aGrFr);
     }

     aSomD /= aVFront.size();

     aSomD /=  (1+anEl.Norm()/50.0);

     if (aSomD>1.5)
     {
        for (const auto & aP : mPtsCC)
            SetMarq(aP,eEEBW_Lab::eBadEl);
     }
     else if (aSomD>1.0)
     {
        for (const auto & aP : mPtsCC)
            SetMarq(aP,eEEBW_Lab::eAverEl);
     }




     if (TEST)
     {
         StdOut() << "SOMDDd=" << aSomD << " " << aSeed.mPixTop 
		 << " GRAY=" << aSomRad/ aVFront.size()
		 << " NORM =" << anEl.Norm()
		 << "\n";

        std::vector<cPt3dr> aVF3;
        for (const auto  & aP : aVFront)
	{
             aVF3.push_back(cPt3dr(aP.x(),aP.y(),ToPolar(aP-mCDG).y()));
	}
        std::sort
        (
	    aVF3.begin(),
	    aVF3.end(),
            [](const cPt3dr  aP1,const cPt3dr &  aP2) {return aP1.z() >aP2.z();}
        );

        for (const auto  & aP : aVF3)
        {
            cPt2dr aP2(aP.x(),aP.y());
            StdOut()  <<  "Teta " << aP.z()   << " S="<< anEl.SignedD2(aP2) << " " << mDIm.GetVBL(aP2)  << "\n";
	}
	//
        StdOut() << "PROP = " << aProp << "\n";
     }
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

        bool IsCandidateTopOfEllipse(const cPt2di &) ;

        bool            mVisu;
        cExtract_BW_Ellipse * mExtrEll;
        cParamCircTarg  mPCT;

        tImGrad         mImGrad;
        cRGBImage       mImVisu;
        cIm2D<tU_INT1>  mImMarq;

        std::vector<cSeedCircTarg> mVSeeds;

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
   mImMarq  (cPt2di(1,1))
{
}

        // cExtract_BW_Ellipse * 
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
                   //  << AOpt2007(mDiamMinD, "DMD","Diam min for detect",{eTA2007::HDV})
	  );
   ;
}






int cAppliExtractCircTarget::ExeOnParsedBox()
{
   double aT0 = SecFromT0();

   mExtrEll = new cExtract_BW_Ellipse(APBI_Im(),mPCT);
   if (mVisu)
      mImVisu =  cRGBImage::FromFile(mNameIm,CurBoxIn());

   double aT1 = SecFromT0();
   mExtrEll->ExtractAllSeed();
   double aT2 = SecFromT0();
   mExtrEll->AnalyseAllConnectedComponents();
   double aT3 = SecFromT0();

   StdOut() << "TIME-INIT " << aT1-aT0 << "\n";
   StdOut() << "TIME-SEED " << aT2-aT1 << "\n";
   StdOut() << "TIME-CC   " << aT3-aT2 << "\n";


   if (mVisu)
   {
       const cExtract_BW_Ellipse::tDImMarq&     aDMarq =  mExtrEll->DImMarq();
       for (const auto & aPix : aDMarq)
       {
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eTmp))
               mImVisu.SetRGBPix(aPix,cRGBImage::Green);

            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadZ))
               mImVisu.SetRGBPix(aPix,cRGBImage::Blue);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadFr))
               mImVisu.SetRGBPix(aPix,cRGBImage::Cyan);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadEl))
               mImVisu.SetRGBPix(aPix,cRGBImage::Red);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eAverEl))
               mImVisu.SetRGBPix(aPix,cRGBImage::Orange);
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
   APBI_ExecAll();  // run the parse file  SIMPL



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

