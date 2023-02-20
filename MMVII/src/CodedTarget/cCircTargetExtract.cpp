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

static bool TEST = false;

struct cParamCircTarg;  ///< Store pararameters for circular/elipse target
struct cSeedCircTarg;   ///< Store data for seed point of circular extraction


// ==  enum class eEEBW_Lab;   ///< label used in ellipse extraction
class cExtract_BW_Ellipse;  ///< class for ellipse extraction

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
   eElNotOk,
   eBadEl,
   eAverEl,
   eBadTeta
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

        cExtract_BW_Ellipse(tIm anIm,const cParamCircTarg & aPCT,cIm2D<tU_INT1> aMasqTest);

        void ExtractAllSeed();
        void AnalyseAllConnectedComponents(const std::string & aNameIm);
        const std::vector<cSeedCircTarg> & VSeeds() const;
	const tDImMarq&    DImMarq() const;
	const tDataIm &    DGx() const;
	const tDataIm &    DGy() const;

	void SetMarq(const cPt2di & aP,eEEBW_Lab aLab) {mDImMarq.SetV(aP,tU_INT1(aLab));}

	void CC_SetMarq(eEEBW_Lab aLab); ///< set marqer on all connected component


	eEEBW_Lab  GetMarq(const cPt2di & aP) {return eEEBW_Lab(mDImMarq.GetV(aP));}
	bool MarqEq(const cPt2di & aP,eEEBW_Lab aLab) const {return mDImMarq.GetV(aP) == tU_INT1(aLab);}
	bool MarqFree(const cPt2di & aP) const {return MarqEq(aP,eEEBW_Lab::eFree);}

   private :

        bool IsCandidateTopOfEllipse(const cPt2di &) ;
        void AnalyseOneConnectedComponents(cSeedCircTarg &,const std::string & aNameIm);

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
        cIm2D<tU_INT1>       mMasqTest;
        cDataIm2D<tU_INT1>&  mDMasqT;

	cPt2di               mPSup;
	cPt2di               mPInf;
};

cExtract_BW_Ellipse::cExtract_BW_Ellipse(tIm anIm,const cParamCircTarg & aPCT,cIm2D<tU_INT1> aMasqTest) :
   mIm        (anIm),
   mDIm       (mIm.DIm()),
   mSz        (mDIm.Sz()),
   mImMarq    (mSz),
   mDImMarq   (mImMarq.DIm()),
   mPCT       (aPCT),
   mImGrad    (Deriche( mDIm,mPCT.mFactDeriche)),
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

void cExtract_BW_Ellipse::AddPtInCC(const cPt2di & aPt)
{
     mDImMarq.SetV(aPt,tU_INT1(eEEBW_Lab::eTmp) );
     mPtsCC.push_back(aPt);
     mCDG = mCDG + ToR(aPt);

     SetInfEq(mPInf,aPt);
     SetSupEq(mPSup,aPt);
}

void cExtract_BW_Ellipse::AnalyseAllConnectedComponents(const std::string & aNameIm)
{
    for (auto & aSeed : mVSeeds)
        AnalyseOneConnectedComponents(aSeed,aNameIm);
}

void cExtract_BW_Ellipse::CC_SetMarq(eEEBW_Lab aLab)
{
    for (const auto & aP : mPtsCC)
        SetMarq(aP,aLab);
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

void  cExtract_BW_Ellipse::AnalyseOneConnectedComponents(cSeedCircTarg & aSeed,const std::string & aNameIm)
{
    TEST = false;
    cPt2di aPTest(-99999999,594);

     mCDG = cPt2dr(0,0);
     mPtsCC.clear();
     cPt2di aP0 = aSeed.mPixW;
     mPSup = aP0;
     mPInf = aP0;

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
                   {
                      if (mDMasqT.GetV(aPN))
                         TEST = true;
                      AddPtInCC(aPN);
		      if (aPN== aPTest)
		      {
			      StdOut() << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa\n";
			      TEST=true;
		      }
                   }
               }
	       else if (! MarqEq(aPN,eEEBW_Lab::eTmp))
                    touchOther = true;
           }
           mCurPts++;
     }

     if ((mPtsCC.size() >= aMaxNbPts) || touchOther  || (int(mPtsCC.size()) < mPCT.NbMinPtsCC()))
     {            
        CC_SetMarq(eEEBW_Lab::eBadZ); 
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
        CC_SetMarq(eEEBW_Lab::eBadFr); 
	return ;
     }

     cEllipse_Estimate anEE(mCDG);
     for (const auto  & aPFr : aVFront)
         anEE.AddPt(aPFr);

     cEllipse anEl = anEE.Compute();
     if (! anEl.Ok())
     {
        CC_SetMarq(eEEBW_Lab::eElNotOk); 
        return;
     }
     double aSomD = 0;
     double aSomRad = 0;
     tREAL8 aGrFr = (aSeed.mBlack+aSeed.mWhite)/2.0;
     for (const auto  & aPFr : aVFront)
     {
         aSomD += std::abs(anEl.ApproxSigneDist(aPFr));
	 aSomRad += std::abs(mDIm.GetVBL(aPFr)-aGrFr);
     }

     aSomD /= aVFront.size();

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
                  return;
	    }
            cPt2dr aGradIm (mDGx.GetVBL(aPt),mDGy.GetVBL(aPt));
	    aSomTeta += std::abs(ToPolar(aGradIm/-aGradTh).y());
     }
     aSomTeta /= aNbPts;

     if (aSomDPond>0.2)
     {
        CC_SetMarq(eEEBW_Lab::eBadEl); 
     }
     else if (aSomDPond>0.1)
     {
        CC_SetMarq(eEEBW_Lab::eAverEl); 
     }
     else
     {
         if (aSomTeta>0.05)
	 {
             CC_SetMarq(eEEBW_Lab::eBadTeta); 
	 }
	 /*
	     static int aCpt=0;aCpt++;
	     StdOut() << "SOMTETA " << aSomTeta << "\n";
	     StdOut() << " N=" << aCpt 
		      << " SOMD = " << aSomD 
                      << " AXES " << 2*anEl.LGa() << " " << 2*anEl.LSa() 
		      << "\n";
	*/
     }


     if (TEST)
     {
         static int aCptIm = 0;
	 aCptIm++;

         StdOut() << "SOMDDd=" << aSomD << " DP=" << aSomDPond << " " << aSeed.mPixTop 
		 << " GRAY=" << aSomRad/ aVFront.size()
		 << " NORM =" << anEl.Norm()  << " RayM=" << anEl.RayMoy()
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
            // StdOut()  <<  "Teta " << aP.z()   << " S="<< anEl.SignedD2(aP2) << " " << mDIm.GetVBL(aP2)  << "\n";
	}
	//
        StdOut() << "PROP = " << aProp << " BOX " << mPInf << " " << mPSup << "\n";

	cPt2di  aPMargin(6,6);
	cBox2di aBox(mPInf-aPMargin,mPSup+aPMargin);

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
             << mPhProj.DPMask().ArgDirInOpt("TestMask","Mask for selecting point used in detailed mesg/output")
             << AOpt2007(mPCT.mMinDiam,"DiamMin","Minimum diameters for ellipse",{eTA2007::HDV})
             << AOpt2007(mPCT.mMaxDiam,"DiamMax","Maximum diameters for ellipse",{eTA2007::HDV})
          );
}



int cAppliExtractCircTarget::ExeOnParsedBox()
{
   double aT0 = SecFromT0();

   mExtrEll = new cExtract_BW_Ellipse(APBI_Im(),mPCT,mPhProj.MaskWithDef(mNameIm,CurBoxIn(),false));
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
       const cExtract_BW_Ellipse::tDImMarq&     aDMarq =  mExtrEll->DImMarq();
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

