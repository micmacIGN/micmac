
#include "MMVII_Sys.h"
#include "MMVII_Geom2D.h"

namespace MMVII
{

template <class Type> const std::vector<cTriangle<Type,2> > &  SplitPixIn2(bool DiagCrois)
{
  cPtxd<Type,2> aP00(0,0);
  cPtxd<Type,2> aP10(1,0);
  cPtxd<Type,2> aP11(1,1);
  cPtxd<Type,2> aP01(0,1);


  if (DiagCrois)
  {
      static std::vector<cTriangle<Type,2> > aRes;
      if (aRes.empty())
      {
          aRes.push_back(cTriangle<Type,2>(aP00,aP10,aP11));
          aRes.push_back(cTriangle<Type,2>(aP00,aP11,aP01));
      }
      return aRes;
  }

  static std::vector<cTriangle<Type,2> > aRes;
  if (aRes.empty())
  {
     aRes.push_back(cTriangle<Type,2>(aP00,aP10,aP01));
     aRes.push_back(cTriangle<Type,2>(aP10,aP11,aP01));
  }
  return aRes;
}

template  const std::vector<cTriangle<int,2> > &  SplitPixIn2<int>(bool DiagCrois);


/* ********************************************** */
/*                                                */
/*           cTriangle2DCompiled                  */
/*                                                */
/* ********************************************** */
	   
template <class Type>  cTriangle2DCompiled<Type>::cTriangle2DCompiled(const cTriangle<Type,2> & aTri) :
    cTriangle<Type,2>(aTri)
{
    tPt aV01 = this->mPts[1]-this->mPts[0];
    tPt aV02 = this->mPts[2]-this->mPts[0];
    mDelta = aV01 ^ aV02;
    if (mDelta !=0)
    {
	      // L1 = (CX1  CY1)   (X1-X0   X2-X0)  =  (1  0)
	      // L2 = (CX2  CY20   (Y1-Y0   Y2-Y0)     (0  1)
        mL1  = tPt( aV02.y() ,-aV02.x()) / mDelta;
        mL2  = tPt(-aV01.y() , aV01.x()) / mDelta;
    }
}

template <class Type> cTriangle2DCompiled<Type>::cTriangle2DCompiled
                       (const tPt & aP0,const tPt & aP1,const tPt & aP2) :
     cTriangle2DCompiled<Type>(cTriangle<Type,2>(aP0,aP1,aP2))
{
}

template <class Type> bool cTriangle2DCompiled<Type>::Regular() const {return mDelta !=0;}

template <class Type> void  cTriangle2DCompiled<Type>::AssertRegular() const
{
     MMVII_INTERNAL_ASSERT_tiny(Regular(),"Non regular triangle");
}

template <class Type> cTriangle2DCompiled<Type> 
                      cTriangle2DCompiled<Type>::RandomRegularTri(Type aSz,Type aEps)
{
  cTriangle2DCompiled<Type> aTri(tPt::PRandC()*aSz,tPt::PRandC()*aSz,tPt::PRandC()*aSz);
   if (std::abs(aTri.mDelta)> aSz*aEps)
      return aTri;
   return RandomRegularTri(aSz,aEps);
}

template <class Type> cPtxd<Type,3> 
        cTriangle2DCompiled<Type>::CoordBarry(const     tPt & aP) const
{
     tPt aV0P = aP - this->mPts[0];
     Type aC1 = Scal(mL1,aV0P);
     Type aC2 = Scal(mL2,aV0P);

     return t3Val(1.0-aC1-aC2,aC1,aC2);
}

template <class Type> Type cTriangle2DCompiled<Type>::Insideness(const tPt & aPt) const
{
     t3Val aCBar = CoordBarry(aPt);

     return std::min(aCBar.x(),std::min(aCBar.y(),aCBar.z()));
}

template <class Type>  bool   cTriangle2DCompiled<Type>::Insides(const tPt & aPt,Type aTol) const
{
    return Insideness(aPt) < aTol;
}

template <class Type> 
   void cTriangle2DCompiled<Type>::PixelsInside(std::vector<cPt2di> & aRes,double aTol,std::vector<t3Val> * aVW) const
{
   aRes.clear();
   cBox2di  aBPE = this->BoxPixEngl() ;

   cPt2di aPix;

   t3Val  aC0Y = CoordBarry(tPt::FromPtInt(aBPE.P0()));
   t3Val  aDxC = CoordBarry(tPt::FromPtInt(aBPE.P0()+cPt2di(1,0))) -aC0Y;
   t3Val  aDyC = CoordBarry(tPt::FromPtInt(aBPE.P0()+cPt2di(0,1))) -aC0Y;

   for (aPix.y()=aBPE.P0().y() ; aPix.y()<=aBPE.P1().y() ; aPix.y()++)
   {
        t3Val  aCXY = aC0Y;
        for (aPix.x()=aBPE.P0().x() ; aPix.x()<=aBPE.P1().x() ; aPix.x()++)
	{
            if ((aCXY.x()>=aTol)&&(aCXY.y()>=aTol)&&(aCXY.z()>=aTol))
	    {
                aRes.push_back(aPix);
		if (aVW)
                   aVW->push_back(aCXY);
	    }
	    aCXY += aDxC;
	}
	aC0Y += aDyC;
   }
}

template <class Type> Type  
    cTriangle2DCompiled<Type>::ValueInterpol(const tPt& aP,const t3Val & aValues) const
{
   return Scal(CoordBarry(aP),aValues);
}

template <class Type> cPtxd<Type,2>  cTriangle2DCompiled<Type>::GradientVI(const t3Val & aValues) const
{
     Type aV0  = ValueInterpol(this->mPts[0],aValues);
     Type aV0x = ValueInterpol(this->mPts[0]+tPt(1,0),aValues);
     Type aV0y = ValueInterpol(this->mPts[0]+tPt(0,1),aValues);

     return tPt(aV0x-aV0,aV0y-aV0);
}


template<class Type> void BenchTri2D(Type aSz)
{
    cTriangle2DCompiled<Type>  aTri = cTriangle2DCompiled<Type>::RandomRegularTri(aSz);
    cPtxd<Type,2>  aPB=cPtxd<Type,2>::PCste(2*aSz);
    cTplBox<Type,2> aBox(-aPB,aPB);

    // FakeUseIt(aTri);
    // FakeUseIt(aBox);
    for (int aK=0 ; aK<1000 ; aK++)
    {
        // cPt2dr aP1 = aBox.GeneratePointInside();
        cPtxd<Type,2> aP1 (RandUnif_C()*2*aSz,RandUnif_C()*2*aSz);
        FakeUseIt(aP1);
        cPtxd<Type,3> aCoBa = aTri.CoordBarry(aP1);
        cPtxd<Type,2> aP2 = aTri.FromCoordBarry(aCoBa);

	Type aDist = Norm2(aP1-aP2)/ aSz;
	Type aEps = tElemNumTrait<Type>::Accuracy();
	MMVII_INTERNAL_ASSERT_bench (aDist<aEps,"Barry incorrect in Tri2d");
    /*
    */
    }
    

    cPtxd<Type,2>   aC = aTri.CenterInscribedCircle();
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
    {
        BenchTri2D<tREAL4> (aK*aK);
        BenchTri2D<tREAL8> (aK*aK);
        BenchTri2D<tREAL16>(aK*aK);
    }

    aParam.EndBench();

}

template  class  cTriangle2DCompiled<tREAL4>;
template  class  cTriangle2DCompiled<tREAL8>;
template  class  cTriangle2DCompiled<tREAL16>;

};
