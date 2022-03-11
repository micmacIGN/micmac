#include "include/MMVII_all.h"

namespace MMVII
{

/* ********************************************** */
/*                                                */
/*           cTriangle2D                          */
/*                                                */
/* ********************************************** */

cTriangle2D::cTriangle2D(const cPt2dr & aP0,const cPt2dr & aP1,const cPt2dr & aP2) 
{
   mPts[0] = aP0;
   mPts[1] = aP1;
   mPts[2] = aP2;
}

const cPt2dr & cTriangle2D::Pt(int aK) const
{
     MMVII_INTERNAL_ASSERT_tiny((aK>=0) && (aK<3),"cTriangle2D::Pt");
     return mPts[aK];
}

cPt2dr  cTriangle2D::FromCoordBarry(const cPt3dr & aP) const
{
     double aSP = aP.x()+aP.y()+aP.z();
     MMVII_INTERNAL_ASSERT_tiny(aSP!=0,"Sum weith null in barry");

     return (aP.x()*mPts[0]+aP.y()*mPts[1]+aP.z()*mPts[2]) / aSP;
}

double cTriangle2D::Regularity() const
{
    cPt2dr aV01 = mPts[1]-mPts[0];
    cPt2dr aV02 = mPts[2]-mPts[0];
    cPt2dr aV12 = mPts[2]-mPts[1];
    double aSomSqN2 = SqN2(aV01) + SqN2(aV02) + SqN2(aV12);
    if (aSomSqN2==0) return 0;
    return std::abs(aV01 ^ aV02) / aSomSqN2;
}

cPt2dr cTriangle2D::CenterInscribedCircle() const
{
   cLeasSqtAA<double>  aSys(2);

   for (int aKp=0 ; aKp<3 ; aKp++)
   {
       const cPt2dr & aP0 = mPts[aKp];
       const cPt2dr & aP1 = mPts[(aKp+1)%3];
       cPt2dr aMil = (aP0+aP1)/2.0;
       cPt2dr aV01 = aP1-aP0;

       aSys.AddObservation(1.0,aV01.ToVect(),Scal(aMil,aV01));
   }

   return cPt2dr::FromVect(aSys.Solve());
}
/*
*/

/* ********************************************** */
/*                                                */
/*           cTriangle2DCompiled                  */
/*                                                */
/* ********************************************** */

cTriangle2DCompiled::cTriangle2DCompiled(const cTriangle2D & aTri) :
    cTriangle2D(aTri)
{
    cPt2dr aV01 = mPts[1]-mPts[0];
    cPt2dr aV02 = mPts[2]-mPts[0];
    mDelta = aV01 ^ aV02;
    if (mDelta !=0)
    {
	      // L1 = (CX1  CY1)   (X1-X0   X2-X0)  =  (1  0)
	      // L2 = (CX2  CY20   (Y1-Y0   Y2-Y0)     (0  1)
        mL1  = cPt2dr( aV02.y() ,-aV02.x()) / mDelta;
        mL2  = cPt2dr(-aV01.y() , aV01.x()) / mDelta;
    }
}

cTriangle2DCompiled::cTriangle2DCompiled(const cPt2dr & aP0,const cPt2dr & aP1,const cPt2dr & aP2) :
     cTriangle2DCompiled(cTriangle2D(aP0,aP1,aP2))
{
}

bool cTriangle2DCompiled::Regular() const {return mDelta !=0;}

void  cTriangle2DCompiled::AssertRegular() const
{
     MMVII_INTERNAL_ASSERT_tiny(Regular(),"Non regular triangle");
}

cTriangle2DCompiled cTriangle2DCompiled::RandomRegularTri(double aSz,double aEps)
{
  cTriangle2DCompiled aTri(cPt2dr::PRandC()*aSz,cPt2dr::PRandC()*aSz,cPt2dr::PRandC()*aSz);
   if (std::abs(aTri.mDelta)> aSz*aEps)
      return aTri;
   return RandomRegularTri(aSz,aEps);
}

cPt3dr  cTriangle2DCompiled::CoordBarry(const     cPt2dr & aP) const
{
     cPt2dr aV0P = aP -mPts[0];
     double aC1 = Scal(mL1,aV0P);
     double aC2 = Scal(mL2,aV0P);

     return cPt3dr(1.0-aC1-aC2,aC1,aC2);
}

double cTriangle2DCompiled::Insideness(const cPt2dr & aPt) const
{
     cPt3dr aCBar = CoordBarry(aPt);

     return std::min(aCBar.x(),std::min(aCBar.y(),aCBar.z()));
}

bool   cTriangle2DCompiled::Insides(const cPt2dr & aPt,double aTol) const
{
    return Insideness(aPt) < aTol;
}

void BenchTri2D(double aSz)
{
    cTriangle2DCompiled  aTri = cTriangle2DCompiled::RandomRegularTri(aSz);
    cPt2dr aPB=cPt2dr::PCste(2*aSz);
    cBox2dr aBox(-aPB,aPB);

    for (int aK=0 ; aK<1000 ; aK++)
    {
        cPt2dr aP1 = aBox.GeneratePointInside();
        cPt3dr aCoBa = aTri.CoordBarry(aP1);
        cPt2dr aP2 = aTri.FromCoordBarry(aCoBa);

	double aDist = Norm2(aP1-aP2)/ aSz;
	MMVII_INTERNAL_ASSERT_bench (aDist<1e-5,"Barry incorrect in Tri2d");
    }
    

    cPt2dr  aC = aTri.CenterInscribedCircle();
    double aRay = Norm2(aC-aTri.Pt(0));
    for (int aK=1 ; aK<3 ; aK++)
    {
         double aRay2 = Norm2(aC-aTri.Pt(aK));
	 double aDif = std::abs((aRay-aRay2)/aSz);
	 MMVII_INTERNAL_ASSERT_bench (aDif<1e-5,"InscribedCircle");
    }
}

void BenchTri2D(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Tri")) return;

    for (int aK=1 ; aK<10 ; aK++)
        BenchTri2D(aK*aK);

    aParam.EndBench();

}


};
