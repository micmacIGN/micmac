#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_SysSurR.h"

namespace MMVII
{


/* ========================== */
/*          cTrans2D          */
/* ========================== */

static constexpr int TrIndTrx   = 0;
static constexpr int TrIndTry   = 1;

template <class Type>  cTrans2D<Type> cTrans2D<Type>::FromParam(const cDenseVect<Type> & aVec) 
{
   return cTrans2D<Type> ( tPt(aVec(TrIndTrx),aVec(TrIndTry)));
}

template <class Type>  
      void cTrans2D<Type>::ToEqParamFromLinear(Type & aLHSOut,cDenseVect<Type>& aVecOut,const tPt &aPIn,const Type & aLHSIn,const tPt & aVec)
{
   //  aLHSIn = aVec . (aPIn+aTr)
   //  aLHSIn - aVec.aPIn = aVec.x * aTr.x + aVec.y * aTr.y
   aVecOut(TrIndTrx) =  aVec.x();
   aVecOut(TrIndTry) =  aVec.y();
   aLHSOut = aLHSIn - Scal(aVec,aPIn);
}

template <class Type>  
      void cTrans2D<Type>::ToEqParam(tPt& aRHS,cDenseVect<Type>& aVX,cDenseVect<Type> & aVY,const tPt & aPIn,const tPt & aPOut)
{
   //  param = trx try 
   //  XOut  =   1*trx + 0*try +   XIN 
   //  YOut  =   0*trx + 1*try +   YIN 
   aVX(TrIndTrx) = 1;
   aVX(TrIndTry) = 0;

   aVY(TrIndTrx) = 0;
   aVY(TrIndTry) = 1;

   aRHS = aPOut - aPIn;
}
template <class Type>  
      void cTrans2D<Type>::ToEqParam(tPt& aRHS,std::vector<cDenseVect<Type>>& aVXY,const tPt & aPIn,const tPt & aPOut)
{
    ToEqParam(aRHS,aVXY.at(0),aVXY.at(1),aPIn,aPOut);
}

template <class Type>  cTrans2D<Type> cTrans2D<Type>::FromMinimalSamples(const tTabMin& aTabIn,const tTabMin& aTabOut)
{
   return cTrans2D<Type>(aTabOut[0]-aTabIn[0]);
}

template <class Type>  cPtxd<Type,2> cTrans2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return Value(aPIn) - aPOut;
}

template <class Type>  Type cTrans2D<Type>::Divisor(const tPt & aPIn) const {return 1.0;}



/* ========================== */
/*          cHomot2D          */
/* ========================== */

static constexpr int HomotIndScale = TrIndTry+1;

template <class Type>  cHomot2D<Type> cHomot2D<Type>::RandomHomotInv(const Type & AmplTr,const Type & AmplSc,const Type & AmplMinSc)
{
    auto v1 = tPt::PRandC() * AmplTr;
    auto v2 = RandUnif_C_NotNull(AmplMinSc/AmplSc)*AmplSc;
    return cHomot2D<Type> ( v1, v2 );
}

template <class Type>  cHomot2D<Type> cHomot2D<Type>::FromParam(const cDenseVect<Type> & aVec) 
{
   return cHomot2D<Type> 
          (
              tPt(aVec(TrIndTrx),aVec(TrIndTry)),
              aVec(HomotIndScale)
          );
}

template <class Type>  
      void cHomot2D<Type>::ToEqParamFromLinear(Type & aLHSOut,cDenseVect<Type>& aVecOut,const tPt &aPIn,const Type & aLHSIn,const tPt & aVec)
{
   //  aLHSIn = aVec . (aScale* aPIn+aTr)
   //  aLHSIn  = aVec.x * aTr.x + aVec.y * aTr.y + aVec.aPIn * aScale

   aVecOut(TrIndTrx) =  aVec.x();
   aVecOut(TrIndTry) =  aVec.y();
   aVecOut(HomotIndScale) = Scal(aVec,aPIn);
   aLHSOut = aLHSIn;

}


template <class Type>  
      void cHomot2D<Type>::ToEqParam(tPt& aRHS,std::vector<cDenseVect<Type>>& aVXY,const tPt & aPIn,const tPt & aPOut)
{
    ToEqParam(aRHS,aVXY.at(0),aVXY.at(1),aPIn,aPOut);
}


template <class Type>  
      void cHomot2D<Type>::ToEqParam(tPt& aRHS,cDenseVect<Type>& aVX,cDenseVect<Type> & aVY,const tPt & aPIn,const tPt & aPOut)
{
   //  param = trx try scale
   //  XOut  =   1*trx + 0*try +  scale * XIN 
   //  YOut  =   0*trx + 1*try +  scale * YIN 
   aVX(TrIndTrx) = 1;
   aVX(TrIndTry) = 0;
   aVX(HomotIndScale) = aPIn.x();

   aVY(TrIndTrx) = 0;
   aVY(TrIndTry) = 1;
   aVY(HomotIndScale) = aPIn.y();

   aRHS = aPOut;
}


template <class Type>  cHomot2D<Type> cHomot2D<Type>::FromMinimalSamples(const tTabMin& aTabIn,const tTabMin& aTabOut)
{
  tPt aCdgIn  =  (aTabIn[0] + aTabIn[1] )/Type(2.0);
  tPt aVecIn =  aTabIn[0]-aTabIn[1];
  Type aDIn =  Norm2(aVecIn);

  // just to avoid degenerency ...
  if (aDIn==0) 
     return cHomot2D<Type>();

  tPt aCdgOut =  (aTabOut[0]+ aTabOut[1])/Type(2.0);
  tPt aVecOut =  aTabOut[0]-aTabOut[1];
  Type aDOut = Norm2(aVecOut);

  Type aScale = aDOut/aDIn;
  if (Scal(aVecIn,aVecOut) < 0) 
      aScale=-aScale;

  //  Tr +  aCdgIn   * S = aCdgOut

  return tTypeMap(aCdgOut-aCdgIn*aScale,aScale);
}


template <class Type>  cPtxd<Type,2> cHomot2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return Value(aPIn) - aPOut;
}

template <class Type>  Type cHomot2D<Type>::Divisor(const tPt & aPIn) const {return 1.0;}


/* ========================== */
/*          cSim2D            */
/* ========================== */


template <class Type>  cSim2D<Type> cSim2D<Type>::FromMinimalSamples(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  
{
    tPt aVIn = (aP1In-aP0In);
    // avoid degenerate case
    if (! IsNotNull(aVIn))
       return cSim2D<Type>();

    tPt aScale = (aP1Out-aP0Out)  /  aVIn;

    return cSim2D<Type>(aP0Out-aScale*aP0In,aScale);
}

template <class Type>  cSim2D<Type> cSim2D<Type>::FromMinimalSamples(const tTabMin& aTabIn,const tTabMin& aTabOut)
{
  return FromMinimalSamples(aTabIn[0],aTabIn[1],aTabOut[0],aTabOut[1]);
}

template <class Type>  cSim2D<Type> cSim2D<Type>::RandomSimInv(const Type & AmplTr,const Type & AmplSc,const Type & AmplMinSc)
{
    auto v1 = tPt::PRandC() * AmplTr;
    auto v2 = tPt::PRandUnitDiff(tPt(0,0),AmplMinSc/AmplSc)*AmplSc;
    return cSim2D<Type> ( v1, v2 );
}



template <class Type>  cSimilitud3D<Type> cSim2D<Type>::Ext3D() const
{
     Type aNSc = Norm2(mSc);
     cDenseMatrix<Type> aMRot2 = MatOfMul (mSc/aNSc);
     cDenseMatrix<Type> aMRot3 = aMRot2.ExtendSquareMatId(3);

     return cSimilitud3D<Type>
	    (
	         aNSc,
		 TP3z0(mTr),
                 cRotation3D<Type>(aMRot3,false)
	    );

}

static constexpr int SimIndTrx = 0;
static constexpr int SimIndTry = 1;
static constexpr int SimIndScx = 2;
static constexpr int SimIndScy = 3;


template <class Type>  cSim2D<Type> cSim2D<Type>::FromParam(const cDenseVect<Type> & aVec) 
{
   return cSim2D<Type> 
          (
              tPt(aVec(SimIndTrx),aVec(SimIndTry)),
              tPt(aVec(SimIndScx),aVec(SimIndScy))
          );
}

template <class Type>  
      void cSim2D<Type>::ToEqParam(tPt& aRHS,std::vector<cDenseVect<Type>>& aVXY,const tPt & aPIn,const tPt & aPOut)
{
    ToEqParam(aRHS,aVXY.at(0),aVXY.at(1),aPIn,aPOut);
}

template <class Type>  
      void cSim2D<Type>::ToEqParamFromLinear(Type & aLHSOut,cDenseVect<Type>& aVecOut,const tPt &aPIn,const Type & aLHSIn,const tPt & aVec)
{
   //  aLHSIn = aVec . (aScale* aPIn+aTr) 
   //          = aVec . ( aScale.x * aPIn.x - aScale.y*aPIn.y + aTr.x)
   //                   ( aScale.x * aPIn.y + aScale.y * aPIn.x + aTr.y)
   //
   //  aLHSIn  = aVec.x * aTr.x + aVec.y * aTr.y 
   //            + ( aVec.x.aPIn.x+ aVec.y.PIn.y) * aScale.x
   //            + (-aVec.x*aPIn.y + aVec.y*aPIn.x) * aScale.y
   aVecOut(SimIndTrx) =  aVec.x();
   aVecOut(SimIndTry) =  aVec.y();
   aVecOut(SimIndScx) =  aVec.x()*aPIn.x() + aVec.y() * aPIn.y();
   aVecOut(SimIndScy) =  -aVec.x()*aPIn.y() + aVec.y() * aPIn.x();
   //aVecOut(SimIndScx) =  aVec.y();
   aLHSOut = aLHSIn;
}

template <class Type>  void cSim2D<Type>::ToEqParam(tPt& aRHS,cDenseVect<Type>& aVX,cDenseVect<Type> & aVY,const tPt & aPIn,const tPt & aPOut)
{
   //  param = trx try scx scy
   //  XOut  =   1*trx + 0*try +  scx * XIN - scy YIN 
   //  YOut  =   0*trx + 1*try +  scx * YIN + scy XIN 
   aVX(SimIndTrx) = 1;
   aVX(SimIndTry) = 0;
   aVX(SimIndScx) = aPIn.x();
   aVX(SimIndScy) = -aPIn.y();

   aVY(SimIndTrx) = 0;
   aVY(SimIndTry) = 1;
   aVY(SimIndScx) = aPIn.y();
   aVY(SimIndScy) = aPIn.x();

   aRHS = aPOut;
}


template <class Type>  cPtxd<Type,2> cSim2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return Value(aPIn) - aPOut;
}

template <class Type>  Type cSim2D<Type>::Divisor(const tPt & aPIn) const {return 1.0;}


/* ========================== */
/*          cAffin2D          */
/* ========================== */

template <class Type>  cAffin2D<Type>::cAffin2D(const tPt & aTr,const tPt & aImX,const tPt aImY) :
    mTr     (aTr),
    mVX     (aImX),
    mVY     (aImY),
    mDelta  (mVX ^ mVY),
    mVInvX  (mDelta  ? tPt(mVY.y(),-mVX.y()) /mDelta : tPt(0,0)),
    mVInvY  (mDelta  ? tPt(-mVY.x(),mVX.x()) /mDelta : tPt(0,0))
{
}
template <class Type>  cAffin2D<Type>::cAffin2D() :
    cAffin2D<Type>(tPt(0,0),tPt(1,0),tPt(0,1))
{
}
// template <class Type>  const int cAffin2D<Type>::NbDOF() {return 6;}
template <class Type> cPtxd<Type,2>  cAffin2D<Type>::Value(const tPt & aP) const 
{
    return  mTr + mVX * aP.x() + mVY *aP.y();
}
template <class Type> cPtxd<Type,2>  cAffin2D<Type>::Inverse(const tPt & aP) const 
{
    return   mVInvX * (aP.x()-mTr.x()) + mVInvY * (aP.y()-mTr.y());
}
template <class Type>  cAffin2D<Type> cAffin2D<Type>::MapInverse() const 
{
	return tTypeMapInv ( VecInverse(-mTr), mVInvX, mVInvY);
}




template <class Type> cPtxd<Type,2>  cAffin2D<Type>::VecValue(const tPt & aP) const 
{
    return   mVX * aP.x() + mVY *aP.y();
}
template <class Type> cPtxd<Type,2>  cAffin2D<Type>::VecInverse(const tPt & aP) const 
{
    return   mVInvX * aP.x() + mVInvY *aP.y();
}


template <class Type>  cAffin2D<Type> cAffin2D<Type>::AllocRandom(const Type & aDeltaMin)
{
   tPt aP0(0,0);
   tTypeMap aRes(aP0,aP0,aP0);
   while (std::abs(aRes.mDelta)<aDeltaMin)
   {
       auto v1 = tPt::PRandC()*Type(10.0);
       auto v2 = tPt::PRandC()*Type(2.0);
       auto v3 = tPt::PRandC()*Type(2.0);
	   aRes =tTypeMap(v1,v2,v3);
   }
   return aRes;
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::operator *(const tTypeMap& aMap2) const
{
	return tTypeMap 
		( 
		    mTr + VecValue(aMap2.mTr), 
		    VecValue(aMap2.mVX), 
		    VecValue(aMap2.mVY)
		);
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::Translation(const tPt  & aTr)
{
	return tTypeMap ( aTr, tPt(1,0),tPt(0,1));
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::Rotation(const Type  & aTeta)
{
	tPt aImX =FromPolar<Type>(Type(1.0),aTeta);
	return tTypeMap (tPt(0,0),aImX,Rot90(aImX));
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::Homot(const Type & aScale)
{
	return tTypeMap (tPt(0,0), tPt(aScale,0),tPt(0,aScale));
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::HomotXY(const Type & aScaleX,const Type & aScaleY)
{
	return tTypeMap (tPt(0,0), tPt(aScaleX,0),tPt(0,aScaleY));
}



template <class Type>  const Type& cAffin2D<Type>::Delta() const {return mDelta;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::Tr() const {return mTr;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::VX() const {return mVX;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::VY() const {return mVY;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::VInvX() const {return mVInvX;}
template <class Type>  const cPtxd<Type,2> & cAffin2D<Type>::VInvY() const {return mVInvY;}


static constexpr int AffIndTrx  = 0;
static constexpr int AffIndTry  = 1;
static constexpr int AffIndXScx = 2;
static constexpr int AffIndXScy = 3;
static constexpr int AffIndYScx = 4;
static constexpr int AffIndYScy = 5;

template <class Type>  cAffin2D<Type> cAffin2D<Type>::FromParam(const cDenseVect<Type> & aVec) 
{
   return cAffin2D<Type> 
          (
              tPt(aVec(AffIndTrx)  , aVec(AffIndTry) ),
              tPt(aVec(AffIndXScx) , aVec(AffIndYScx)),
              tPt(aVec(AffIndXScy) , aVec(AffIndYScy))
          );
}

template <class Type>  
      void cAffin2D<Type>::ToEqParam(tPt& aRHS,std::vector<cDenseVect<Type>>& aVXY,const tPt & aPIn,const tPt & aPOut)
{
    ToEqParam(aRHS,aVXY.at(0),aVXY.at(1),aPIn,aPOut);
}

template <class Type>  void cAffin2D<Type>::ToEqParam(tPt& aRHS,cDenseVect<Type>& aVX,cDenseVect<Type> & aVY,const tPt & aPIn,const tPt & aPOut)
{
   //  param = trx try scx scy
   //  XOut  =   1*trx + 0*try +  Xscx * XIN + Xscy YIN  + YScx * 0 + Y Scy * 0
   //  YOut  =   0*trx + 1*try +  Xscx * 0 +   Xscy * 0 + YScx * XIN + YScy * YIN
   aVX(AffIndTrx) = 1;
   aVX(AffIndTry) = 0;
   aVX(AffIndXScx) = aPIn.x();
   aVX(AffIndXScy) = aPIn.y();
   aVX(AffIndYScx) = 0;
   aVX(AffIndYScy) = 0;

   aVY(AffIndTrx) = 0;
   aVY(AffIndTry) = 1;
   aVY(AffIndXScx) = 0;
   aVY(AffIndXScy) = 0;
   aVY(AffIndYScx) = aPIn.x();
   aVY(AffIndYScy) = aPIn.y();

   aRHS = aPOut;
}

/*
    aR0ToIn (1,0) = aTabIn[0] + VecX = aTabIn[1]

*/
template <class Type>   cAffin2D<Type>  cAffin2D<Type>::FromMinimalSamples(const tTabMin& aTabIn,const tTabMin& aTabOut)
{
    cAffin2D<Type> aR0ToIn  (aTabIn[0]  , aTabIn[1] -aTabIn[0]  , aTabIn[2] -aTabIn[0]  );
    // avoid degenerate case
    if (aR0ToIn.Delta()==0)
       return cAffin2D<Type>();

    cAffin2D<Type> aR0ToOut (aTabOut[0] , aTabOut[1]-aTabOut[0] , aTabOut[2]-aTabOut[0] );

    return aR0ToOut* aR0ToIn.MapInverse() ;
}

template <class Type>   cAffin2D<Type>  cAffin2D<Type>:: Tri2Tri(const tTri& aTriIn,const tTri& aTriOut)
{
    tTabMin aTabIn {aTriIn.Pt(0) ,aTriIn.Pt(1) ,aTriIn.Pt(2) };
    tTabMin aTabOut{aTriOut.Pt(0),aTriOut.Pt(1),aTriOut.Pt(2)};

    return FromMinimalSamples(aTabIn,aTabOut);
}

template <class Type> Type cAffin2D<Type>::MinResolution() const
{
    //  See documentation of mmv1, appendix E, for justification of the forlua
    
    Type aVX2 =   SqN2(mVX) ;
    Type aVY2 =   SqN2(mVY) ;

    Type aRadical = Square(aVX2-aVY2) +4*Square(Scal(mVX,mVY));

    Type aRes = (aVX2 + aVY2 - std::sqrt(aRadical)) / 2.0;

    return std::sqrt(std::max(Type(0.0),aRes));
}

template <class Type>  cPtxd<Type,2> cAffin2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return Value(aPIn) - aPOut;
}
template <class Type>  Type cAffin2D<Type>::Divisor(const tPt & aPIn) const {return 1.0;}


/* ========================== */
/*          cHomogr2D         */
/* ========================== */

static constexpr int iHXx = 0;
static constexpr int iHXy = 1;
static constexpr int iHX1 = 2;
static constexpr int iHYx = 3;
static constexpr int iHYy = 4;
static constexpr int iHY1 = 5;
static constexpr int iHZx = 6;
static constexpr int iHZy = 7;

template <class Type> cHomogr2D<Type>::cHomogr2D(const tElemH & aHX,const tElemH & aHY,const tElemH & aHZ) :
    mHX (aHX/aHZ.z()),
    mHY (aHY/aHZ.z()),
    mHZ (aHZ/aHZ.z())
{
   cDenseMatrix<Type> aMatI = Mat().Inverse();
   GetLine(mIHX,0,aMatI);
   GetLine(mIHY,1,aMatI);
   GetLine(mIHZ,2,aMatI);
}

template <class Type> cHomogr2D<Type>::cHomogr2D() :
    cHomogr2D<Type>(tElemH(1,0,0),tElemH(0,1,0),tElemH(0,0,1))
{
}

template <class Type> cHomogr2D<Type> cHomogr2D<Type>::FromParam(const cDenseVect<Type> & aV)
{
    return cHomogr2D<Type>
           (
               tElemH(aV(iHXx),aV(iHXy),aV(iHX1)),
               tElemH(aV(iHYx),aV(iHYy),aV(iHY1)),
               tElemH(aV(iHZx),aV(iHZy),1.0    )
           );
}

template <class Type> cHomogr2D<Type>  cHomogr2D<Type>::FromMinimalSamples(const tTabMin& aTabIn,const tTabMin&aTabOut)
{
   MMVII_INTERNAL_ASSERT_always((NbDOF%TheDim)==0,"Div assert in FromMinimalSamples");
   cDenseMatrix<Type> aMat(NbDOF);
   cDenseVect<Type>   aVect(NbDOF);

   cDenseVect<Type> aVX(NbDOF);
   cDenseVect<Type> aVY(NbDOF);
   tPt aRhs;
   for (int aK=0 ; aK<NbPtsMin ; aK++)
   {
       ToEqParam(aRhs,aVX,aVY,aTabIn[aK],aTabOut[aK]);
       
       int aLineX = 2 *aK;
       aMat.WriteLine(aLineX,aVX);
       aVect(aLineX) = aRhs.x();

       int aLineY = aLineX + 1;
       aMat.WriteLine(aLineY,aVY);
       aVect(aLineY) = aRhs.y();
   }
   // aMat.Show();

   cDenseVect<Type> aSol =  aMat.SolveColumn(aVect);
   // for (int aK=0 ; aK<8 ; aK++) StdOut() << "PPPP " << aSol(aK) << std::endl;
   return FromParam(aSol);
}

template <class Type>  
      void cHomogr2D<Type>::ToEqParam(tPt& aRHS,std::vector<cDenseVect<Type>>& aVXY,const tPt & aPIn,const tPt & aPOut)
{
    ToEqParam(aRHS,aVXY.at(0),aVXY.at(1),aPIn,aPOut);
}

template <class Type> void cHomogr2D<Type>::ToEqParam
                           (
                                 tPt & aRHS,
                                 cDenseVect<Type>& aVX,
                                 cDenseVect<Type> & aVY,
                                 const tPt &In,
                                 const tPt & Out
                           )
{
      //  I =  (xx Xin + yy Yin + zz) / (zx Xin + zy Yin + 1.0)
      //  I = xx Xin + yy Yin + zz  - zx (I Xin)   - zy (I Yin)
      Type I    = Out.x();

      aVX(iHXx) = In.x();
      aVX(iHXy) = In.y();
      aVX(iHX1) = 1.0;

      aVX(iHYx) = 0.0;
      aVX(iHYy) = 0.0;
      aVX(iHY1) = 0.0;

      aVX(iHZx) = - In.x() * I;
      aVX(iHZy) = - In.y() * I;

      Type J    = Out.y();
      
      aVY(iHXx) = 0.0;
      aVY(iHXy) = 0.0;
      aVY(iHX1) = 0.0;

      aVY(iHYx) = In.x();
      aVY(iHYy) = In.y();
      aVY(iHY1) = 1.0;

      aVY(iHZx) = - In.x() * J;
      aVY(iHZy) = - In.y() * J;

      aRHS = Out;
}


template <class Type> cDenseMatrix<Type> cHomogr2D<Type>::Mat() const
{
   return M3x3FromLines(mHX,mHY,mHZ);
}

template <class Type> cHomogr2D<Type> cHomogr2D<Type>::FromMat(const cDenseMatrix<Type> & aMat)
{
   tElemH aHx,aHy,aHz;
   GetLine(aHx,0,aMat);
   GetLine(aHy,1,aMat);
   GetLine(aHz,2,aMat);

   return cHomogr2D<Type>(aHx,aHy,aHz);
}

template <class Type> cHomogr2D<Type> cHomogr2D<Type>::operator *(const tTypeMap& aM2) const
{
    return FromMat(Mat() * aM2.Mat());
} 

template <class Type> cHomogr2D<Type>  cHomogr2D<Type>::MapInverse() const 
{
    return FromMat(Mat().Inverse());
}

template <class Type>  cPtxd<Type,2> cHomogr2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return tPt(S(mHX,aPIn),S(mHY,aPIn)) - aPOut * S(mHZ,aPIn);
}

template <class Type>  Type cHomogr2D<Type>::Divisor(const tPt & aPIn) const {return  S(mHZ,aPIn);}

template <class Type>   cHomogr2D<Type>  cHomogr2D<Type>::AllocRandom(const Type & aAmpl)
{
    auto aPair = RandomPtsHomgr(aAmpl);
    return StdGlobEstimate(aPair.first,aPair.second);
}

/*   Solving the paral homographic equation
 *
 *           ax + by +cz +d             ix + jy + kz +l
 *    xo=    ---------------      yo =  --------------- 
 *           ex + fy + gz + 1           ex + fy + gz + 1
 *
 *    
 *    xo = (Hx(P) + cz) / (Hz(P) + gz)     yo = (Hy(P) + kz) / (Hz(P) + gz)
 *
 *    We have a linear system with 3 unknown cz,kz,fd.  For each point where we know xi,yi,x,y we can write :
 *
 *    xo Hz(P) - Hx(P)  = cz - xo gz
 *    yo Hz(P) - Hy(P)  = kz - yo gz
 *
 *    We can solve it with 2 known correp
 */

template <class Type>   cHomogr2D<Type>  cHomogr2D<Type>::LeastSqParalPlaneShift(tCRVPts aVIn,tCRVPts aVOut) const
{

   MMVII_INTERNAL_ASSERT_medium(aVIn.size()==aVOut.size(),"Bad sizes in LeastSqParalPlaneShift");
   // MMVII_INTERNAL_ASSERT_medium(aVIn.size()==aVOut.size(),"Bad sizes in LeastSqParalPlaneShift");
   MMVII_INTERNAL_ASSERT_medium(aVIn.size()>=2,"Bad sizes in LeastSqParalPlaneShift");

   cLeasSqtAA<Type>  aSys(3);
   cDenseVect<Type>  aVec(3);

   for (size_t aKP=0 ; aKP<aVIn.size() ; aKP++)
   {
       const tPt & aPIn = aVIn[aKP];
       const tPt & aPOut = aVOut[aKP];
       Type aHx = S(mHX,aPIn);
       Type aHy = S(mHY,aPIn);
       Type aHz = S(mHZ,aPIn);
       Type xo = aPOut.x();
       Type yo = aPOut.y();

 // *    xo Hz(P) - Hx(P)  = cz - xo gz
       aVec(0) = 1;
       aVec(1) = 0;
       aVec(2) = -xo;
       aSys.PublicAddObservation(1.0,aVec,xo*aHz-aHx);

 // *    yo Hz(P) - Hy(P)  = kz - yi gz
       aVec(0) = 0;
       aVec(1) = 1;
       aVec(2) = -yo;
       aSys.PublicAddObservation(1.0,aVec,yo*aHz-aHy);
   }

   cDenseVect<Type>  aSol = aSys.PublicSolve();
   tTypeMap  aRes 
	     (
	         mHX + tElemH(0,0,aSol(0)),
	         mHY + tElemH(0,0,aSol(1)),
	         mHZ + tElemH(0,0,aSol(2))
	     );

   return aRes;
}

template <class Type>   cHomogr2D<Type>  cHomogr2D<Type>::RansacParalPlaneShift(tCRVPts aVIn,tCRVPts aVOut,int aNbMin,int aNbMax) const
{
   aNbMax =std::min(aNbMax,int(aVOut.size()));
   int aNbTest = 10000;

   MMVII_INTERNAL_ASSERT_medium(aVIn.size()==aVOut.size(),"Bad sizes in RansacParalPlaneShift");
   cWhichMin<tTypeMap,tTypeElem>  aWMin(tTypeMap(),1e30);


   for (int aNb=aNbMin ; aNb<=aNbMax ; aNb++)
   {
       std::vector<cSetIExtension> aVSubInd;
       GenRanQsubCardKAmongN(aVSubInd,aNbTest,aNb,aVIn.size());

       for (const auto & aSubInd : aVSubInd)
       {
           tVPts aSubIn = SubVector(aVIn,aSubInd.mElems);
           tVPts aSubOut = SubVector(aVOut,aSubInd.mElems);
	   tTypeMap  aHomTest = LeastSqParalPlaneShift(aSubIn,aSubOut);
	   tTypeElem aSc = aHomTest.AvgDistL1(aVIn,aVOut);

	   aWMin.Add(aHomTest,aSc);
       }
   }
   return aWMin.IndexExtre();
}



/* ========================== */
/*          cRot2D            */
/* ========================== */


template <class Type>  cRot2D<Type> cRot2D<Type>::RandomRot(const Type & AmplTr)
{
    auto v1 = tPt::PRandC() * AmplTr;
    auto v2 = RandUnif_C() * 10 * M_PI;
    return cRot2D<Type>( v1, v2 );
}


static constexpr int RotIndTrx =  0;
static constexpr int RotIndTry =  1;
static constexpr int RotIndTeta = 2;


template <class Type>  cRot2D<Type> cRot2D<Type>::FromParam(const cDenseVect<Type> & aVec) 
{
   return cRot2D<Type> 
          (
              tPt(aVec(RotIndTrx),aVec(RotIndTry)),
              aVec(RotIndTeta)
          );
}

template <class Type>  
      void cRot2D<Type>::ToEqParam(tPt& aRHS,std::vector<cDenseVect<Type>>& aVXY,const tPt & aPIn,const tPt & aPOut)
{
    ToEqParam(aRHS,aVXY.at(0),aVXY.at(1),aPIn,aPOut);
}

template <class Type>  void cRot2D<Type>::ToEqParam(tPt& aRHS,cDenseVect<Type>& aVX,cDenseVect<Type> & aVY,const tPt & aPIn,const tPt & aPOut)
{
   //  param = trx try scx scy
   //  XOut  =   1*trx + 0*try +   XIN - teta YIN 
   //  YOut  =   0*trx + 1*try +   YIN + teta XIN 
   aVX(RotIndTrx)  = 1;
   aVX(RotIndTry)  = 0;
   aVX(RotIndTeta) = -aPIn.y();

   aVY(RotIndTrx)  = 0;
   aVY(RotIndTry)  = 1;
   aVY(RotIndTeta) = aPIn.x();

   aRHS = aPOut -aPIn;
}


template <class Type>  cRot2D<Type> cRot2D<Type>::FromMinimalSamples(const tTabMin& aTabIn,const tTabMin& aTabOut)
{
  tPt aCdgIn  =  (aTabIn[0] + aTabIn[1] )/Type(2.0);
  tPt aVecIn  =  aTabIn[1]-aTabIn[0];
  
  if (IsNull (aVecIn))
     return  cRot2D<Type>();

  tPt aCdgOut =  (aTabOut[0]+ aTabOut[1])/Type(2.0);
  tPt aVecOut =  aTabOut[1]-aTabOut[0];

  tPt aRot = aVecOut/aVecIn;
  if (IsNotNull(aRot))
     aRot = VUnit(aRot);
  //  Tr +  aCdgIn   * S = aCdgOut
  // return tTypeMap(aCdgOut-aCdgIn*aScale,aScale);
  return  cRot2D<Type>(aCdgOut-aCdgIn*aRot,ToPolar(aRot,Type(0.0)).y());
   
}

template <class Type>  cPtxd<Type,2> cRot2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return Value(aPIn) - aPOut;
}
template <class Type>  Type cRot2D<Type>::Divisor(const tPt & aPIn) const {return 1.0;}

template <class Type>  cRot2D<Type> cRot2D<Type>::QuickEstimate(tCRVPts aVIn,tCRVPts aVOut)
{
     CheckSzInOut<tTypeMap>(aVIn,aVOut);

     cPtxd<Type,2>  aCdgI =   cPtxd<Type,2>::FromPtR(Centroid(aVIn));
     cPtxd<Type,2>  aCdgO =   cPtxd<Type,2>::FromPtR(Centroid(aVOut));

     // etimate rotation as weighted average  of  Rot * VIn = VOut
     cPtxd<Type,2> aVRot(0,0);
     for (size_t aK=0; aK<aVIn.size() ; aK++)
     {
            cPtxd<Type,2> aVecI =  aVIn[aK] - aCdgI;
            cPtxd<Type,2> aVecO =  aVOut[aK] - aCdgO;

            Type aW = Norm2(aVecI);
            if  (aW>0)
            {
                 aVRot +=  (aVecO/aVecI) * aW;
            }
     }

     Type aTeta = ToPolar(aVRot,Type(0.0)).y(); // vect rot to angle
     aVRot = FromPolar(Type(1.0),aTeta);

     cPtxd<Type,2>  aTr = aCdgO - aCdgI*aVRot;  // usign  Out = Tr +In* Rot

     return cRot2D<Type>(aTr,aTeta);

}


// ============================================================================
//           static void ToEqParamFromLinear(Type & aRHS,cDenseVect<Type>&,const tPt &aPIn,const Type & aLHSIn,const tPt & aScal);


template <class tMap>  void ToEqInSeg(tREAL8& aRHS,cDenseVect<tREAL8> & aVec,const cPt2dr &aPIn,const cSegment2DCompiled<tREAL8> & aSeg)
{
    // aSeg.Nom . Q = aSeg.Norm  aSeg.Mil

    tMap::ToEqParamFromLinear(aRHS,aVec,aPIn,Scal(aSeg.Normal(),aSeg.Middle()),aSeg.Normal());
}

template   void ToEqInSeg<cTrans2D<tREAL8>>(tREAL8& aRHS,cDenseVect<tREAL8> & aVec,const cPt2dr &aPIn,const cSegment2DCompiled<tREAL8> & aSeg);
template   void ToEqInSeg<cHomot2D<tREAL8>>(tREAL8& aRHS,cDenseVect<tREAL8> & aVec,const cPt2dr &aPIn,const cSegment2DCompiled<tREAL8> & aSeg);
template   void ToEqInSeg<cSim2D<tREAL8>>(tREAL8& aRHS,cDenseVect<tREAL8> & aVec,const cPt2dr &aPIn,const cSegment2DCompiled<tREAL8> & aSeg);

// ==============================================================================
// ==============================================================================
// ===  INSTANTIATION INSTANTIATION INSTANTIATION INSTANTIATION INSTANTIATION  ==
// ==============================================================================
// ==============================================================================



#define MACRO_INSTATIATE_ALL_MAP_GEOM2D_OF_TYPE(TYPE) \
template class cTrans2D<TYPE>;\
template class cRot2D<TYPE>;\
template class cHomogr2D<TYPE>;\
template class cAffin2D<TYPE>;\
template class cSim2D<TYPE>;\
template class cHomot2D<TYPE>;


MACRO_INSTATIATE_ALL_MAP_GEOM2D_OF_TYPE(tREAL4);
MACRO_INSTATIATE_ALL_MAP_GEOM2D_OF_TYPE(tREAL8);
MACRO_INSTATIATE_ALL_MAP_GEOM2D_OF_TYPE(tREAL16);

};
