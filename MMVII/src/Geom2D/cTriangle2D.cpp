#include "include/MMVII_all.h"

namespace MMVII
{


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

void cTriangle2DCompiled::PixelsInside(std::vector<cPt2di> & aRes,double aTol) const
{
   aRes.clear();
   cBox2di  aBPE = BoxPixEngl() ;

   cPt2di aPix;

   cPt3dr  aC0Y = CoordBarry(ToR(aBPE.P0()));
   cPt3dr  aDxC = CoordBarry(ToR(aBPE.P0()+cPt2di(1,0))) -aC0Y;
   cPt3dr  aDyC = CoordBarry(ToR(aBPE.P0()+cPt2di(0,1))) -aC0Y;

   for (aPix.y()=aBPE.P0().y() ; aPix.y()<=aBPE.P1().y() ; aPix.y()++)
   {
        cPt3dr  aCXY = aC0Y;
        for (aPix.x()=aBPE.P0().x() ; aPix.x()<=aBPE.P1().x() ; aPix.x()++)
	{
            if ((aCXY.x()>=aTol)&&(aCXY.y()>=aTol)&&(aCXY.z()>=aTol))
	    {
		    aRes.push_back(aPix);
	    }
	     aCXY += aDxC;
	}
	aC0Y += aDyC;
   }
}

double  cTriangle2DCompiled::ValueInterpol(const cPt2dr& aP,const cPt3dr & aValues) const
{
   return Scal(CoordBarry(aP),aValues);
}

cPt2dr  cTriangle2DCompiled::GradientVI(const cPt3dr & aValues) const
{
     double aV0 = ValueInterpol(mPts[0],aValues);
     double aV0x = ValueInterpol(mPts[0]+cPt2dr(1,0),aValues);
     double aV0y = ValueInterpol(mPts[0]+cPt2dr(0,1),aValues);

     return cPt2dr(aV0x-aV0,aV0y-aV0);
}


void BenchTri2D(double aSz)
{
    cTriangle2DCompiled  aTri = cTriangle2DCompiled::RandomRegularTri(aSz);
    cPt2dr aPB=cPt2dr::PCste(2*aSz);
    cBox2dr aBox(-aPB,aPB);
// StdOut() << "BbbBB: " << aBox << "\n";

    // FakeUseIt(aTri);
    FakeUseIt(aBox);
    for (int aK=0 ; aK<1000 ; aK++)
    {
        // cPt2dr aP1 = aBox.GeneratePointInside();
        cPt2dr aP1 (RandUnif_C()*2*aSz,RandUnif_C()*2*aSz);
        FakeUseIt(aP1);
        cPt3dr aCoBa = aTri.CoordBarry(aP1);
        cPt2dr aP2 = aTri.FromCoordBarry(aCoBa);

	double aDist = Norm2(aP1-aP2)/ aSz;
	MMVII_INTERNAL_ASSERT_bench (aDist<1e-5,"Barry incorrect in Tri2d");
    /*
    */
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
