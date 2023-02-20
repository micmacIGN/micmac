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

     mRayMoy = std::sqrt(mLGa*mLSa);
     mSqRatio = std::sqrt(mLGa/mLSa);
}


bool   cEllipse::Ok() const   {return mOk;}
tREAL8 cEllipse::LGa() const  {return mLGa;}
tREAL8 cEllipse::LSa() const  {return mLSa;}
tREAL8 cEllipse::RayMoy() const  {return mRayMoy;}
const cPt2dr &  cEllipse::Center() const {return mCenter;}

cPt2dr  cEllipse::PtOfTeta(tREAL8 aTeta) const
{
        return  mCenter+ mVGa *(cos(aTeta)*mLGa) + mVSa *(sin(aTeta)*mLSa);
}


cPt2dr  cEllipse::PtAndGradOfTeta(tREAL8 aTeta,cPt2dr &aGrad) const
{
    // Tgt = DP/Dteta =  (-mLGa sin(aTeta)  ,  mLSa cos(teta)
    // Norm = Tgt * P(0,-1) =>    mLSa cos(teta) , mLGa sin(aTeta)

   tREAL8 aCos = cos(aTeta);
   tREAL8 aSin = sin(aTeta);

   aGrad = VUnit( mVGa *(mLSa * aCos)  +  mVSa*(mLGa*aSin));
   return mCenter+ mVGa *(aCos*mLGa) + mVSa *(aSin *mLSa);
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
}

cEllipse cEllipse_Estimate::Compute() {return cEllipse(mSys->Solve(),mC0);}


};
