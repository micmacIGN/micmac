#include "MMVII_Geom2D.h"
#include "MMVII_SysSurR.h"


namespace MMVII
{

/* *********************************************************** */
/*                                                             */
/*                         cEllipse                            */
/*                                                             */
/* *********************************************************** */

cEllipse::cEllipse(cDenseVect<tREAL8> aDV,const cPt2dr & aC0) :
    mV    (aDV.Dup()),
    mNorm (std::sqrt(Square(mV(0)) + 2 * Square(mV(1))  + Square(mV(2)))),
    mC0   (aC0),
    mQF   (M2x2FromLines(cPt2dr(mV(0),mV(1)),cPt2dr(mV(1),mV(2))))
{
    cPt2dr aSol = SolveCol(mQF,cPt2dr(mV(3),mV(4)))/2.0;
    mCenter  = aC0-aSol;
    mCste = -1-QScal(aSol,mQF,aSol);

    cResulSymEigenValue<tREAL8>  aRSEV = mQF.SymEigenValue();

     mLGa = aRSEV.EigenValues()(0);
     mLSa = aRSEV.EigenValues()(1);

     mOk = (mLGa >0) && (mLSa>0) && (mCste<0) ;
     if (!mOk) return;

     mLGa = std::sqrt((-mCste)/mLGa);
     mLSa = std::sqrt((-mCste)/mLSa);
                     
     GetCol(mVGa,aRSEV.EigenVectors(),0);
     GetCol(mVSa,aRSEV.EigenVectors(),1);

     // There is no warantee on orientaion  from jacobi !!
     if ((mVGa ^ mVSa) < 0)
        mVSa = - mVSa;

     mRayMoy = std::sqrt(mLGa*mLSa);
     mSqRatio = std::sqrt(mLGa/mLSa);
}


bool   cEllipse::Ok() const   {return mOk;}
tREAL8 cEllipse::LGa() const  {return mLGa;}
tREAL8 cEllipse::LSa() const  {return mLSa;}
tREAL8 cEllipse::RayMoy() const  {return mRayMoy;}
const cPt2dr &  cEllipse::Center() const {return mCenter;}

cPt2dr  cEllipse::PtOfTeta(tREAL8 aTeta,tREAL8 aMulRho) const
{
    return  mCenter+ mVGa *(cos(aTeta)*mLGa*aMulRho) + mVSa *(sin(aTeta)*mLSa*aMulRho);
}


cPt2dr  cEllipse::PtAndGradOfTeta(tREAL8 aTeta,cPt2dr &aGrad,tREAL8 aMulRho) const
{
    // Tgt = DP/Dteta =  (-mLGa sin(aTeta)  ,  mLSa cos(teta)
    // Norm = Tgt * P(0,-1) =>    mLSa cos(teta) , mLGa sin(aTeta)

   tREAL8 aCos = cos(aTeta);
   tREAL8 aSin = sin(aTeta);

   aGrad = VUnit( mVGa *(mLSa * aCos)  +  mVSa*(mLGa*aSin));
   return mCenter+ mVGa *(aCos*mLGa*aMulRho) + mVSa *(aSin *mLSa*aMulRho);
}

double cEllipse::TetaGa() const { return ToPolar(mVGa).y(); }


double cEllipse::SignedD2(cPt2dr aP) const
{
     if (1)
     {
         cPt2dr aQ = aP-mCenter;
         tREAL8 aRes =   QScal(aQ,mQF,aQ)  + mCste;
         return aRes / mNorm;
     }

     aP = aP-mC0;
     tREAL8 x = aP.x();
     tREAL8 y = aP.y();
     tREAL8 aRes =   mV(0)*x*x  + 2*mV(1)*x*y + mV(2)*y*y + mV(3)*x+mV(4)*y -1;

     return aRes / mNorm;
}

double cEllipse::ApproxSigneDist(cPt2dr aP) const
{
    aP = (aP-mCenter) / mVGa;
    tREAL8 aRayMoy = std::sqrt(mLGa*mLSa);
    tREAL8 aRatio = std::sqrt(mLGa/mLSa);

    aP = cPt2dr(aP.x()/aRatio,aP.y()*aRatio);

    return Norm2(aP) - aRayMoy;
}


double cEllipse::Dist(const cPt2dr & aP) const {return std::sqrt(std::abs(SignedD2(aP)));}

/*  *********************************************************** */
/*                                                              */
/*               cEllipseEstimate                               */
/*                                                              */
/*  *********************************************************** */
cEllipse_Estimate::cEllipse_Estimate(const cPt2dr & aC0) :
    mSys  (new cLeasSqtAA<tREAL8> (5)),
    mC0   (aC0)
{
}

cLeasSqtAA<tREAL8> & cEllipse_Estimate::Sys() {return *mSys;}

cEllipse_Estimate::~cEllipse_Estimate()
{
    delete mSys;
}

void cEllipse_Estimate::AddPt(cPt2dr aP)
{
     aP = aP-mC0;

     cDenseVect<tREAL8> aDV(5);
     aDV(0) = Square(aP.x());
     aDV(1) = 2 * aP.x() * aP.y();
     aDV(2) = Square(aP.y());
     aDV(3) = aP.x();
     aDV(4) = aP.y();

     mSys->AddObservation(1.0,aDV,1.0);

     mVObs.push_back(aP);
}

cEllipse cEllipse_Estimate::Compute()
{
     auto  aSol = mSys->Solve();

     return cEllipse(aSol,mC0);
     /// return  aRes;
}

/*  *********************************************************** */
/*                                                              */
/*               cExtractedEllipse                              */
/*                                                              */
/*  *********************************************************** */


cExtractedEllipse::cExtractedEllipse(const cSeedBWTarget& aSeed,const cEllipse & anEllipse) :
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


     cExtractedEllipse  anEE(aSeed,anEl);
     anEE.mDist      = aSomD;
     anEE.mDistPond  = aSomDPond;
     anEE.mEcartAng  = aSomTeta;
     anEE.mVFront    = mVFront;

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

const std::list<cExtractedEllipse> & cExtract_BW_Ellipse::ListExtEl() const {return mListExtEl;}

void  cExtractedEllipse::ShowOnFile(const std::string & aNameIm,int aZoom,const std::string& aPrefName) const
{

    static int aCptIm = 0;
    aCptIm++;
    const cSeedBWTarget &  aSeed = mSeed;
    const cEllipse &       anEl  = mEllipse;


	cPt2di  aPMargin(6,6);
	cBox2di aBox(aSeed.mPInf-aPMargin,aSeed.mPSup+aPMargin);

	cRGBImage aRGBIm = cRGBImage::FromFile(aNameIm,aBox,aZoom);  ///< Allocate and init from file
	cPt2dr aPOfs = ToR(aBox.P0());

// MMVII_INTERNAL_ASSERT_tiny(false,"CHANGE TO DO IN ShowEllipse !!!");

        aRGBIm.DrawEllipse(cRGBImage::Blue,anEl.Center() - aPOfs,anEl.LGa(),anEl.LSa(),anEl.TetaGa());

        for (const auto  & aPFr : mVFront)
	{
            aRGBIm.SetRGBPoint(aPFr-aPOfs,cRGBImage::Red);
	    //StdOut() <<  "DDDD " <<  anEl.ApproxSigneDist(aPFr) << "\n";
	}

	aRGBIm.ToFile(aPrefName + "_Ellipses_" + ToStr(aCptIm) + ".tif");
}

};
