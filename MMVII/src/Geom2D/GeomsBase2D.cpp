#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_SysSurR.h"

namespace MMVII
{

const  cPt2di FreemanV8[8]   {{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1}};
const  cPt2di FreemanV4[4]   {{1,0},{0,1},{-1,0},{0,-1}};
const  cPt2di FreemanV10[10] {{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1}};

const std::vector<cPt2di> & Alloc4Neighbourhood() { return AllocNeighbourhood<2>(1); }
const std::vector<cPt2di> & Alloc8Neighbourhood() { return AllocNeighbourhood<2>(2); }

template <class Type> Type DbleAreaPolygOriented(const std::vector<cPtxd<Type,2>>& aPolyg)
{
    Type aRes = 0;
    size_t aNb = aPolyg.size() ;
    if (aNb)
    {
        // substract origin, avoid overflow, better accuracy
        cPtxd<Type,2> aP0 = aPolyg.at(0);
        for (size_t aK=0 ; aK<aNb ; aK++)
            aRes += (aPolyg.at(aK) -aP0)  ^ (aPolyg.at((aK+1)%aNb) -aP0);
    }
    return aRes;
}

/* ========================== */
/*         cBox2di            */
/* ========================== */

cBox2di DilateFromIntervPx(const cBox2di & aBox,int aDPx0,int aDPx1)
{
   cPt2di aP0 = aBox.P0();
   cPt2di aP1 = aBox.P1();
   return  cBox2di
           (
                cPt2di(aP0.x()+aDPx0,aP0.y()),
                cPt2di(aP1.x()+aDPx1,aP1.y())
           );
}

cBox2di BoxAndCorresp(cHomot2D<tREAL8> & aHomIn2Image,const cBox2dr & aBox,int aSzIm,int aMarginImage)
{
   int anAmpl = aSzIm - 2*aMarginImage;
   double aScale  = double(anAmpl)/ NormInf(aBox.Sz());
   cPt2di aPMargin(aMarginImage,aMarginImage);

   cPt2di aSzBox = aPMargin*2 +  ToI(aBox.Sz()*aScale);


   aHomIn2Image = cHomot2D<tREAL8>(ToR(aPMargin) - aBox.P0()*aScale,aScale);  //  Tr+P0()*aScale = (aMargeImage
   
   return cBox2di(cPt2di(0,0),aSzBox);
}



/* ========================== */
/*    cSegment2DCompiled      */
/* ========================== */

template <class Type> cSegment2DCompiled<Type>::cSegment2DCompiled(const tPt& aP1,const tPt& aP2) :
    cSegmentCompiled<Type,2> (aP1,aP2),
    mNorm            (Rot90(this->mTgt))
{
}

template <class Type> cSegment2DCompiled<Type>::cSegment2DCompiled(const cSegment<Type,2>& aSeg) :
    cSegment2DCompiled<Type>(aSeg.P1(),aSeg.P2())
{
}

template <class Type> Type cSegment2DCompiled<Type>::SignedDist(const tPt& aPt) const
{
    return Scal(mNorm,aPt - this->mP1);
}
template <class Type> Type cSegment2DCompiled<Type>::Dist(const tPt& aPt) const
{
    return std::abs(SignedDist(aPt));
}


template <class Type> cPtxd<Type,2> cSegment2DCompiled<Type>::ToCoordLoc(const tPt& aPt) const
{
    tPt   aV1P = aPt - this->mP1;
    return tPt(Scal(this->mTgt,aV1P),Scal(mNorm,aV1P));
}

template <class Type> cPtxd<Type,2> cSegment2DCompiled<Type>::FromCoordLoc(const tPt& aPt) const
{
    return  this->mP1 + this->mTgt*aPt.x()  + mNorm*aPt.y();
}

template <class Type> Type cSegment2DCompiled<Type>::DistLine(const tPt& aPt) const
{
    return std::abs(Scal(mNorm,aPt - this->mP1));
}


template <class Type> Type cSegment2DCompiled<Type>::DistClosedSeg(const tPt& aPt) const
{
    tPt aPL = ToCoordLoc(aPt);

    if (aPL.x() < 0)
       return  Norm2(aPL);

    if (aPL.x() >  this->mN2)
      return  Norm2(aPt-this->mP2);

    return std::abs(aPL.y());
}

template <class Type> 
    cPtxd<Type,2>  cSegment2DCompiled<Type>::InterSeg
                   (
		        const cSegment2DCompiled<Type> & aSeg2,
			tREAL8 aMinAngle,
			bool *IsOk
                   )
{
      //  (aM1 + L1 T1) . N2 = M2 . N2
      //  L1 (T1.N2)  = N2. (aM2-aM1) 
      tPt aM1 = this->PMil();
      tPt aM2 = aSeg2.PMil();

      Type aScalT1N2 = Scal(this->mTgt,aSeg2.mNorm);

      if (std::abs(aScalT1N2) <= aMinAngle)
      {
          if (IsOk)
	  {
		  *IsOk = false;
		  return cPtxd<Type,2>::Dummy();
	  }
	  MMVII_INTERNAL_ERROR("Segment2DCompiled<Type>::InterSeg  : almost parallel");
      }

      if (IsOk) 
          *IsOk = true;

      Type aLambda = Scal(aM2-aM1,aSeg2.mNorm) / aScalT1N2 ;

      return aM1 + this->mTgt * aLambda;
}


/* ========================== */
/*         cClosedSeg2D       */
/* ========================== */



cClosedSeg2D::cClosedSeg2D(const cPt2dr & aP0,const cPt2dr & aP1) :
    mSeg (aP0,aP1)
{
}

bool  cClosedSeg2D::InfEqDist(const cPt2dr & aPt,tREAL8 aDist) const
{
      return mSeg.DistClosedSeg(aPt) < aDist;
}

cBox2dr cClosedSeg2D::GetBoxEnglob() const
{
   //  return cBox 2dr(mSeg.P1(),mSeg.P2(),true); => does not work  because P1,P2 are not "min-max-ed" 
   cTplBoxOfPts<tREAL8,2> aBox =   cTplBoxOfPts<tREAL8,2>::FromVect({mSeg.P1(),mSeg.P2()});
   return aBox.CurBox(true);
}

const cSegment2DCompiled<tREAL8> & cClosedSeg2D::Seg() const {return mSeg;}


/* ========================== */
/*         cMapEstimate       */
/* ========================== */

template <class TypeMap> class  cMapEstimate
{
    public :
           typedef  typename TypeMap::tPt  tPt;
           typedef  typename TypeMap::tTypeElem  tTypeElem;
           typedef  std::vector<tPt>   tVPts;
           typedef std::vector<tTypeElem> tVVals;
           typedef  const tVPts &   tCRVPts;
           typedef const tVVals * tCPVVals;


           static void CheckInOut(const  tVPts& aVIn,const tVPts & aVOut);
           /// Estimate the map M such that  M(aVIn[aK]) = aVOut[aK]
           static  TypeMap LeasSqEstimate(const  tVPts& aVIn,const tVPts & aVOut,tTypeElem * aRes2,const tVVals* aVW);
           /// Estimate the map M such that  M(aVIn[aK]) = aVOut[aK] using ransac
           static inline TypeMap RansacL1Estimate(const  tVPts& aVIn,const tVPts & aVOut,int aNbTest);

          /// For non linear case, make an iteration using current sol
          static TypeMap LeastSquareRefine(const TypeMap &,tCRVPts,tCRVPts,tTypeElem*,tCPVVals);

          /// For non linear make Ransac+linear refine
          static TypeMap LeastSquareNLEstimate(tCRVPts,tCRVPts,tTypeElem*,tCPVVals,const cParamCtrlOpt&);
    private :
};

template <class TypeMap>  
    void  cMapEstimate<TypeMap>::CheckInOut(const  tVPts& aVIn,const tVPts & aVOut)
{
   MMVII_INTERNAL_ASSERT_medium(aVIn.size()==aVOut.size(),"Bad sizes in cMapEstimate");
   MMVII_INTERNAL_ASSERT_medium(aVIn.size()>= TypeMap::NbPtsMin,"Not enough obs in cMapEstimate");
}

template <class TypeMap> 
   typename TypeMap::tTypeElem  AvgDistL1(const TypeMap & aMap,typename TypeMap::tCRVPts aVIn,typename TypeMap::tCRVPts aVOut)
{
    MMVII_INTERNAL_ASSERT_medium(aVIn.size()==aVOut.size(),"Bad sizes in AvgDistL1");
    typename TypeMap::tTypeElem  aRes= 0.0;
    for (size_t aKP=0 ; aKP< aVIn.size() ; aKP++)
        aRes +=  Norm1(aMap.Value(aVIn[aKP])-aVOut[aKP]);

    return aRes / aVIn.size();
}


template <class TypeMap>  
    TypeMap  cMapEstimate<TypeMap>::LeasSqEstimate
             (
                const  tVPts & aVIn,
                const tVPts & aVOut,
                tTypeElem * aRes2,
                const tVVals* aVW
             )
{
   CheckInOut(aVIn,aVOut);

   cLeasSqtAA<tTypeElem> aSys(TypeMap::NbDOF);
   cDenseVect<tTypeElem> aVX(TypeMap::NbDOF);
   cDenseVect<tTypeElem> aVY(TypeMap::NbDOF);

   for (int aK=0; aK<int(aVIn.size()) ; aK++)
   {
        tPt aRHS;
        TypeMap::ToEqParam(aRHS,aVX,aVY,aVIn[aK],aVOut[aK]);
        tTypeElem aWeight = aVW ? (aVW->at(aK)) : 1.0;
        aSys.PublicAddObservation(aWeight,aVX,aRHS.x());
        aSys.PublicAddObservation(aWeight,aVY,aRHS.y());
   }
   cDenseVect<tTypeElem> aSol =  aSys.PublicSolve();
   TypeMap aMap =  TypeMap::FromParam(aSol);

   if (aRes2)
   {
      *aRes2 = 0.0;
      for (int aK=0; aK<int(aVIn.size()) ; aK++)
      {
          *aRes2 += SqN2(aVOut[aK]-aMap.Value(aVIn[aK]));
      }
      ///StdOut() << "NOOrrrmSol= " << aSomR2 << std::endl;
   }

   return aMap;
}

template <class TypeMap,class TypePt>  void InspectMap2D(const TypeMap & aMap,const TypePt & aPt)
{
}

void InspectMap2D(const cHomogr2D<tREAL8> & aMap,const tPt2dr & aPt)
{
     StdOut() << " ----------- Pt=" << aPt << "-------------" << std::endl;
     StdOut() << "Hxx : " <<  aMap.Hx() << " S=" << aMap.S(aMap.Hx(),aPt) << std::endl;
     StdOut() << "Hyy : " <<  aMap.Hy() << " S=" << aMap.S(aMap.Hy(),aPt) << std::endl;
     StdOut() << "Hzz : " <<  aMap.Hz() << " S=" << aMap.S(aMap.Hz(),aPt) << std::endl;
}


template <class TypeMap>  
    TypeMap  cMapEstimate<TypeMap>::RansacL1Estimate(const  tVPts& aVAllIn,const tVPts & aVAllOut,int aNbTest)
{
// StdOut() << "L111 " << TypeMap::Name() << " " << tNumTrait<tTypeElem>::NameType() << std::endl;


    tTypeElem aSigma = 0.5 * (      cComputeCentroids<tVPts>::MedianSigma(aVAllIn)
                                 +  cComputeCentroids<tVPts>::MedianSigma(aVAllOut)
                             );


    CheckInOut(aVAllIn,aVAllOut);
    std::vector<cSetIExtension> aVSubInd;
    // generate NbTest subset
    GenRanQsubCardKAmongN(aVSubInd,aNbTest,TypeMap::NbPtsMin,aVAllIn.size());

    cWhichMin<TypeMap,tTypeElem>  aWMin(TypeMap(),1e30);
    typename TypeMap::tTabMin aVMinIn,aVMinOut;
    //  Parse all subset
    for (const auto & aSub : aVSubInd)
    {
         // Generate the minimal subset of points In&Out
         for(int aK=0 ; aK<TypeMap::NbPtsMin ; aK++)
         {
             aVMinIn[aK]  = aVAllIn [aSub.mElems[aK]];
             aVMinOut[aK] = aVAllOut[aSub.mElems[aK]];
         }
         // Compute the map
         TypeMap aMap = TypeMap::FromMinimalSamples(aVMinIn,aVMinOut);

         // Compute the residual for this map
         tTypeElem aSomDist = 0;

         for (int aKP=0 ; aKP<int(aVAllIn.size()) ; aKP++)
         {
             // tTypeElem aDist = Norm2(aVAllOut[aKP]-aMap.Value(aVAllIn[aKP]));
             tTypeElem aDist = Norm2(aMap.DiffInOut(aVAllIn[aKP],aVAllOut[aKP]));
             aDist = aDist * aSigma/ (aSigma+aDist);
             aSomDist += aDist;
         }
         // Update best map
         aWMin.Add(aMap,aSomDist);
    }

     return aWMin.IndexExtre();
}

template <class TypeMap>  
    TypeMap  cMapEstimate<TypeMap>::LeastSquareRefine
             (const TypeMap & aMap0,tCRVPts aVIn,tCRVPts aVOut,tTypeElem * aRes2,tCPVVals aVW)
{
    //  aMap (IN) = Out   
    // We could write
    //    =>   aMap0  * aDelta (In) = Out    =>  aDelta (In) = aMap0-1 Out
    // BUT  if maps are no isometric the min & dist would be measured input space an potentially biaised  

    // So we write rather :
    //   =>   aDelta * aMap0  (In) = Out    => aMap0  (In) = aDelta-1 Out
    //     return  Inverse(aDelta-1) * aMap0

    CheckInOut(aVIn,aVOut);

    tVPts aVMI;
    for (const auto & aPtIn : aVIn)
       aVMI.push_back(aMap0.Value(aPtIn));

    TypeMap  aDeltaM1 = LeasSqEstimate(aVOut,aVMI,aRes2,aVW);

/*
if (aRes2)
StdOut() << "===RRRR " << *aRes2 << std::endl;
*/

    return aDeltaM1.MapInverse() * aMap0;
}

template <class TypeMap>  
    TypeMap  cMapEstimate<TypeMap>::LeastSquareNLEstimate
             (
                  tCRVPts aVIn,
                  tCRVPts aVOut,
                  tTypeElem* aPtrRes,
                  tCPVVals   aVW,
                  const cParamCtrlOpt& aParam
             )
{
    TypeMap  aMap = RansacL1Estimate(aVIn,aVOut,aParam.ParamRS().NbTestOfErrAdm(TypeMap::NbPtsMin));
    cParamCtrNLsq  aPLSq= aParam.ParamLSQ();
 
    tTypeElem  aResidual;
    if (aPtrRes==nullptr)
       aPtrRes = &aResidual;

    while (true)
    {
         aMap =   LeastSquareRefine(aMap,aVIn,aVOut,aPtrRes,aVW);
         if (aPLSq.StabilityAfterNextError(*aPtrRes))
            return aMap;
    }

    return aMap;
}
/*
*/


/* ========================== */
/*          cHomot2D          */
/* ========================== */

static constexpr int HomIndTrx   = 0;
static constexpr int HomIndTry   = 1;
static constexpr int HomIndScale = 2;

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
              tPt(aVec(HomIndTrx),aVec(HomIndTry)),
              aVec(HomIndScale)
          );
}

template <class Type>  void cHomot2D<Type>::ToEqParam(tPt& aRHS,cDenseVect<Type>& aVX,cDenseVect<Type> & aVY,const tPt & aPIn,const tPt & aPOut)
{
   //  param = trx try scale
   //  XOut  =   1*trx + 0*try +  scale * XIN 
   //  YOut  =   0*trx + 1*try +  scale * YIN 
   aVX(HomIndTrx) = 1;
   aVX(HomIndTry) = 0;
   aVX(HomIndScale) = aPIn.x();

   aVY(HomIndTrx) = 0;
   aVY(HomIndTry) = 1;
   aVY(HomIndScale) = aPIn.y();

   aRHS = aPOut;
}



template <class Type>  
     cHomot2D<Type> cHomot2D<Type>::StdGlobEstimate
                        (tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVWeights)
{
    return cMapEstimate<cHomot2D<Type>>::LeasSqEstimate(aVIn,aVOut,aRes2,aVWeights);
}

template <class Type>  cHomot2D<Type> cHomot2D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
{
    return cMapEstimate<cHomot2D<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);
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

/*
*/

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



template <class Type>  cSim2D<Type> cSim2D<Type>::StdGlobEstimate
                        (tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVWeights)
{
    return cMapEstimate<cSim2D<Type>>::LeasSqEstimate(aVIn,aVOut,aRes2,aVWeights);
}

template <class Type>  cSim2D<Type> cSim2D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
{
    return cMapEstimate<cSim2D<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);
}

template <class Type>  cPtxd<Type,2> cSim2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return Value(aPIn) - aPOut;
}

template <class Type>  Type cSim2D<Type>::Divisor(const tPt & aPIn) const {return 1.0;}

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

template <class Type>  cRot2D<Type> cRot2D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
{
    return cMapEstimate<cRot2D<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);
}

template <class Type>  
         cRot2D<Type> cRot2D<Type>::LeastSquareRefine(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVW)const
{
    return cMapEstimate<cRot2D<Type>>::LeastSquareRefine(*this,aVIn,aVOut,aRes2,aVW);
}

template <class Type>  cRot2D<Type> cRot2D<Type>::QuickEstimate(tCRVPts aVIn,tCRVPts aVOut)
{
     cMapEstimate<tTypeMap>::CheckInOut(aVIn,aVOut);

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

template <class Type>  
         cRot2D<Type> cRot2D<Type>::StdGlobEstimate
                      (
                           tCRVPts aVIn,
                           tCRVPts aVOut,
                           tTypeElem* aRes,
                           tCPVVals   aVW,
                           cParamCtrlOpt aParam
                      )
{
    return  cMapEstimate<cRot2D<Type> >::LeastSquareNLEstimate(aVIn,aVOut,aRes,aVW,aParam);
}

template <class Type>  cPtxd<Type,2> cRot2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return Value(aPIn) - aPOut;
}
template <class Type>  Type cRot2D<Type>::Divisor(const tPt & aPIn) const {return 1.0;}

//  cRot2D<Type> cRot2D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
//  cRot2D<Type> cRot2D<Type>::LeastSquareRefine(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVW)const

// template  TMAP TMAP::LeastSquareEstimate(const std::vector<tPt>&,const std::vector<tPt>&,TYPE*,const std::vector<TYPE> *);

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
/*
*/

template <class Type>  
     cAffin2D<Type> cAffin2D<Type>::StdGlobEstimate
                        (tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVWeights)
{
    return cMapEstimate<cAffin2D<Type>>::LeasSqEstimate(aVIn,aVOut,aRes2,aVWeights);
}

template <class Type>  cAffin2D<Type> cAffin2D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
{
    return cMapEstimate<cAffin2D<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);
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


/*
template <class Type> const cPtxd<Type,3>& cHomogr2D<Type>::Hx() const {return mHX;}
template <class Type> const cPtxd<Type,3>& cHomogr2D<Type>::Hy() const {return mHY;}
template <class Type> const cPtxd<Type,3>& cHomogr2D<Type>::Hz() const {return mHZ;}

template <class Type> cPtxd<Type,3>& cHomogr2D<Type>::Hx() {return mHX;}
template <class Type> cPtxd<Type,3>& cHomogr2D<Type>::Hy() {return mHY;}
template <class Type> cPtxd<Type,3>& cHomogr2D<Type>::Hz() {return mHZ;}
*/

template <class Type>  cHomogr2D<Type> cHomogr2D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
{
    return cMapEstimate<cHomogr2D<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);
}

template <class Type>  
     cHomogr2D<Type> cHomogr2D<Type>::StdGlobEstimate
                        (tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVWeights)
{
    return cMapEstimate<cHomogr2D<Type>>::LeasSqEstimate(aVIn,aVOut,aRes2,aVWeights);
}

template <class Type>  cPtxd<Type,2> cHomogr2D<Type>::DiffInOut(const tPt & aPIn,const tPt & aPOut) const
{
   return tPt(S(mHX,aPIn),S(mHY,aPIn)) - aPOut * S(mHZ,aPIn);
}

template <class Type>  Type cHomogr2D<Type>::Divisor(const tPt & aPIn) const {return  S(mHZ,aPIn);}
/*
template <class Type>  
     cHomogr2D<Type> cHomogr2D<Type>::StdGlobEstimate
                        (tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVWeights)
{
    return cMapEstimate<cHomogr2D<Type>>::LeasSqEstimate(aVIn,aVOut,aRes2,aVWeights);
}
*/

template <class Type>  Type cHomogr2D<Type>::AvgDistL1(tCRVPts aVIn,tCRVPts aVOut)
{
	return MMVII::AvgDistL1(*this,aVIn,aVOut);
}


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

void BenchHomogr2D()
{
/*
   for (int aTest=0 ; aTest< 20 ; aTest++)
   {
       cPt2dr aTabIn[4];
       cPt2dr aTabOut[4];

       for (int aKP=0 ; aKP< 4 ; aKP++)
       {
           aTabIn [aKP] = cPt2dr::PRandC();
           aTabOut[aKP] = cPt2dr::PRandC();
       }

       cHomogr2D<tREAL8> aH   = cHomogr2D<tREAL8>::FromMinimalSamples(aTabIn,aTabOut);
       cHomogr2D<tREAL8> aHI  = aH.MapInverse();
       cHomogr2D<tREAL8> aID1 = aH * aHI;

       for (int aKP=0 ; aKP< 4 ; aKP++)
       {
           StdOut() << Norm2(aTabOut[aKP]  - aH.Value(aTabIn[aKP]))   << " " 
                    << Norm2(aHI.Value(aTabOut[aKP])  - aTabIn[aKP])  << " "
                    << Norm2(aH.Inverse(aTabOut[aKP])  - aTabIn[aKP]) << " "
                    << Norm2(aID1.Value(aTabIn[aKP])  - aTabIn[aKP])  << " "

                    << " IO " <<  Norm2(aH.DiffInOut(aTabIn[aKP],aTabOut[aKP]))
                    << "\n";
       }

       StdOut() << " ======================================== " << std::endl;
   }
*/
}



/* ========================== */
/*             ::             */
/* ========================== */

template <class Type> cDenseMatrix<Type> MatOfMul (const cPtxd<Type,2> & aP)
{
    cDenseMatrix<Type> aRes(2);

    SetCol(aRes,0,aP);         // P * (1,0)
    SetCol(aRes,1,Rot90(aP));  // P * (0,1)

    return aRes;
}

template <class Type> 
    std::vector<cPtxd<Type,2> > RandomPtsOnCircle(const std::vector<int> & aVInd,const cPtxd<Type,2>& aP0, double aRho )
{
  int aNbPts = aVInd.size();
  std::vector<cPtxd<Type,2> > aRes;
  // std::vector<int> aVInd =  RandPerm(aNbPts);
  double aTeta0 = RandUnif_0_1() * 2 * M_PI;
  double aEcartTeta =  ( 2 * M_PI)/aNbPts;
  // double aRho  = RandUnif_C_NotNull(0.1);
  // cPtxd<Type,2> aP0 = cPtxd<Type,2>::PRand();

  for (int aK=0 ; aK<aNbPts ; aK++)
  {
       double aTeta = aTeta0 +  aEcartTeta * (aVInd[aK] +0.2 * RandUnif_C());
       cPtxd<Type,2> aP =  aP0 + FromPolar(Type(aRho),Type(aTeta));
       aRes.push_back(aP);
  }

  return aRes;
}

template <class Type> std::vector<cPtxd<Type,2> > RandomPtsOnCircle(int aNbPts)
{
     return RandomPtsOnCircle(RandPerm(aNbPts),cPtxd<Type,2>::PRand(), RandUnif_C_NotNull(0.1));
}


template <class Type> 
      std::pair<std::vector<cPtxd<Type,2> >,std::vector<cPtxd<Type,2>>> RandomPtsHomgr(Type aR)
{
    std::pair<std::vector<cPtxd<Type,2> >,std::vector<cPtxd<Type,2>>>  aRes;
    std::vector<int> aVInd =  RandPerm(4);

    cPtxd<Type,2> aC0 = cPtxd<Type,2>::PRand() *Type(0.1);
    cPtxd<Type,2> aC1 = cPtxd<Type,2>::PRand() *Type(0.1);

     Type aR0 = RandInInterval(aR,2.0*aR);
     Type aR1 = RandInInterval(aR,2.0*aR);

     aRes.first  = RandomPtsOnCircle(aVInd,aC0,aR0);
     aRes.second = RandomPtsOnCircle(aVInd,aC1,aR1);
     

    return aRes;

}

template <class Type> 
      std::pair<std::vector<cPtxd<Type,2> >,std::vector<cPtxd<Type,2>>> RandomPtsId(int aNb,Type aEpsId)
{
    std::pair<std::vector<cPtxd<Type,2> >,std::vector<cPtxd<Type,2>>>  aRes;
    for (int aK=0 ; aK< aNb ; aK++)
    {
         cPtxd<Type,2> aP1 =  FromPolar(Type(1.0),Type( ( 2 * M_PI * aK)/aNb));
         aRes.first.push_back(aP1);
         aRes.second.push_back(aP1 + cPtxd<Type,2>::PRandC() * aEpsId);
    }
    return aRes;
}

template <class tMap>  tMap RandomMapId(typename tMap::tTypeElem aEpsId)
{
     auto aPair = RandomPtsId(tMap::NbPtsMin,aEpsId);

    return tMap::StdGlobEstimate(aPair.first,aPair.second);
}



template <class Type,class tMap>  cTriangle<Type,2>  ImageOfTri(const cTriangle<Type,2> & aTri,const tMap & aMap)
{
     return  cTriangle<Type,2>(aMap.Value(aTri.Pt(0)),aMap.Value(aTri.Pt(1)),aMap.Value(aTri.Pt(2)));
}

template <class T>  T LineAngles(const cPtxd<T,2> & aDir1,const cPtxd<T,2> & aDir2)
{
     T aRes = std::abs(Teta(aDir1* conj(aDir2)));  // Angle Dir1/Dir2
     return std::min(aRes,(T)M_PI - aRes);
}



/* ========================== */
/*       INSTANTIATION        */
/* ========================== */

#define INSTANTIATE_GEOM_REAL(TYPE)\
template std::vector<cPtxd<TYPE,2> > RandomPtsOnCircle<TYPE>(int aNbPts);\
template std::pair<std::vector<cPtxd<TYPE,2> >,std::vector<cPtxd<TYPE,2>>> RandomPtsHomgr(TYPE);\
template std::pair<std::vector<cPtxd<TYPE,2> >,std::vector<cPtxd<TYPE,2>>> RandomPtsId(int aNb,TYPE aEpsId);\
template class cSegment2DCompiled<TYPE>;\
template class  cAffin2D<TYPE>;\
template class  cHomogr2D<TYPE>;\
template   cHomogr2D<TYPE> RandomMapId(TYPE);\
template TYPE LineAngles(const cPtxd<TYPE,2> & aDir1,const cPtxd<TYPE,2> & aDir2);

INSTANTIATE_GEOM_REAL(tREAL4)
INSTANTIATE_GEOM_REAL(tREAL8)
INSTANTIATE_GEOM_REAL(tREAL16)

// MACRO_INSTATIATE_GEOM2D_MAPPING(TYPE,cAffin2D<TYPE>,2);


#define MACRO_INSTATIATE_GEOM2D_MAPPING(TYPE,TMAP,DIM)\
template   cTriangle<TYPE,2>  ImageOfTri(const cTriangle<TYPE,2> & aTri,const TMAP & aMap);\
template  TMAP TMAP::FromParam(const cDenseVect<TYPE> & aVec) ;\
template  void TMAP::ToEqParam(tPt&,cDenseVect<TYPE>&,cDenseVect<TYPE> &,const tPt &,const tPt &);\
template  TMAP TMAP::FromMinimalSamples(const tTabMin& ,const tTabMin& );\
template TMAP TMAP::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);\
template  TYPE TMAP::Divisor(const tPt & aPIn) const;

#define MACRO_INSTATIATE_LINEAR_GEOM2D_MAPPING(TYPE,TMAP,DIM)\
template  TMAP TMAP::StdGlobEstimate(tCRVPts,tCRVPts,TYPE*,tCPVVals);\
MACRO_INSTATIATE_GEOM2D_MAPPING(TYPE,TMAP,DIM)

#define MACRO_INSTATIATE_NON_LINEAR_GEOM2D_MAPPING(TYPE,TMAP,DIM)\
MACRO_INSTATIATE_GEOM2D_MAPPING(TYPE,TMAP,DIM); \
template TMAP TMAP::LeastSquareRefine(tCRVPts,tCRVPts,TYPE *,tCPVVals)const;\
template TMAP TMAP::StdGlobEstimate(tCRVPts,tCRVPts,tTypeElem*,tCPVVals,cParamCtrlOpt);\
template   TMAP RandomMapId(TYPE);

#define MACRO_INSTATIATE_GEOM2D(TYPE)\
template  cRot2D<TYPE>  cRot2D<TYPE>::QuickEstimate(tCRVPts aVIn,tCRVPts aVOut);\
MACRO_INSTATIATE_NON_LINEAR_GEOM2D_MAPPING(TYPE,cRot2D<TYPE>,2);\
template  cRot2D<TYPE> cRot2D<TYPE>::RandomRot(const TYPE & AmplTr);\
MACRO_INSTATIATE_LINEAR_GEOM2D_MAPPING(TYPE,cSim2D<TYPE>,2);\
MACRO_INSTATIATE_LINEAR_GEOM2D_MAPPING(TYPE,cHomot2D<TYPE>,2);\
template  cSim2D<TYPE> cSim2D<TYPE>::RandomSimInv(const TYPE & AmplTr,const TYPE & AmplSc,const TYPE & AmplMinSc);\
template  cHomot2D<TYPE> cHomot2D<TYPE>::RandomHomotInv(const TYPE &,const TYPE &,const TYPE &);\
template  cSim2D<TYPE> cSim2D<TYPE>::FromMinimalSamples(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  ;\
template  cSimilitud3D<TYPE> cSim2D<TYPE>::Ext3D() const;\
template  cDenseMatrix<TYPE> MatOfMul (const cPtxd<TYPE,2> & aP);\
template TYPE DbleAreaPolygOriented(const std::vector<cPtxd<TYPE,2>> & aPolyg);


//  MACRO_INSTATIATE_LINEAR_GEOM2D_MAPPING(TYPE,cHomogr2D<TYPE>,2);


MACRO_INSTATIATE_GEOM2D(tREAL4)
MACRO_INSTATIATE_GEOM2D(tREAL8)
MACRO_INSTATIATE_GEOM2D(tREAL16)

template int DbleAreaPolygOriented(const std::vector<cPtxd<int,2>> & aPolyg);


};
