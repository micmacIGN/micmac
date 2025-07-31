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
      tPt aM1 = this->Middle();
      tPt aM2 = aSeg2.Middle();

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

#if (0)

template <class TypeMap> class  cMapEstimate
{
    public :
           typedef  typename TypeMap::tPt  tPt;
           typedef  typename TypeMap::tTypeElem  tTypeElem;
           typedef  std::vector<tPt>   tVPts;
           typedef std::vector<tTypeElem> tVVals;
           typedef  const tVPts &   tCRVPts;
           typedef const tVVals * tCPVVals;


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
// StdOut() << "cMapEstimate<TypeMap>::LeasSqEstimatecMapEstimate<TypeMap>::LeasSqEstimatecMapEstimate<TypeMap>::LeasSqEstimate\n";
   CheckSzInOut<TypeMap>(aVIn,aVOut);

   cLeasSqtAA<tTypeElem> aSys(TypeMap::NbDOF);
   std::vector<cDenseVect<tTypeElem>>  aVV;
   for (int aDim=0; aDim<TypeMap::TheDim ; aDim++)
       aVV.push_back( cDenseVect<tTypeElem> (TypeMap::NbDOF));


   for (int aK=0; aK<int(aVIn.size()) ; aK++)
   {
        tPt aRHS;
        TypeMap::ToEqParam(aRHS,aVV,aVIn[aK],aVOut[aK]);
        tTypeElem aWeight = aVW ? (aVW->at(aK)) : 1.0;
        for (int aDim=0 ; aDim<TypeMap::TheDim ; aDim++)
            aSys.PublicAddObservation(aWeight,aVV.at(aDim),aRHS[aDim]);
   }

/*
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
*/
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


    CheckSzInOut<TypeMap>(aVAllIn,aVAllOut);
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

    CheckSzInOut<TypeMap>(aVIn,aVOut);

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


/* ========================== */
/*          cHomot2D          */
/* ========================== */

/*

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
*/


/* ========================== */
/*          cSim2D            */
/* ========================== */


/*
template <class Type>  cSim2D<Type> cSim2D<Type>::StdGlobEstimate
                        (tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVWeights)
{
    return cMapEstimate<cSim2D<Type>>::LeasSqEstimate(aVIn,aVOut,aRes2,aVWeights);
}

template <class Type>  cSim2D<Type> cSim2D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
{
    return cMapEstimate<cSim2D<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);
}
*/


/* ========================== */
/*          cRot2D            */
/* ========================== */


/*
template <class Type>  cRot2D<Type> cRot2D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
{
    return cMapEstimate<cRot2D<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);
}

template <class Type>  
         cRot2D<Type> cRot2D<Type>::LeastSquareRefine(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVW)const
{
    return cMapEstimate<cRot2D<Type>>::LeastSquareRefine(*this,aVIn,aVOut,aRes2,aVW);
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
*/


/* ========================== */
/*          cAffin2D          */
/* ========================== */


/*
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

*/


/* ========================== */
/*          cHomogr2D         */
/* ========================== */


/*
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

template <class Type>  Type cHomogr2D<Type>::AvgDistL1(tCRVPts aVIn,tCRVPts aVOut)
{
	return MMVII::AvgDistL1(*this,aVIn,aVOut);
}
*/

#endif



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
template  cDenseMatrix<TYPE> MatOfMul (const cPtxd<TYPE,2> & aP);\
template TYPE DbleAreaPolygOriented(const std::vector<cPtxd<TYPE,2>> & aPolyg);\
template std::vector<cPtxd<TYPE,2> > RandomPtsOnCircle<TYPE>(int aNbPts);\
template std::pair<std::vector<cPtxd<TYPE,2> >,std::vector<cPtxd<TYPE,2>>> RandomPtsHomgr(TYPE);\
template std::pair<std::vector<cPtxd<TYPE,2> >,std::vector<cPtxd<TYPE,2>>> RandomPtsId(int aNb,TYPE aEpsId);\
template class cSegment2DCompiled<TYPE>;\
template TYPE LineAngles(const cPtxd<TYPE,2> & aDir1,const cPtxd<TYPE,2> & aDir2);

INSTANTIATE_GEOM_REAL(tREAL4)
INSTANTIATE_GEOM_REAL(tREAL8)
INSTANTIATE_GEOM_REAL(tREAL16)




#define MACRO_INSTATIATE_GEOM2D_OF_MAP(TMAP,TYPE) \
template  TMAP RandomMapId<TMAP> (TYPE);\
template   cTriangle<TYPE,2>  ImageOfTri(const cTriangle<TYPE,2> & aTri,const TMAP & aMap);


#define MACRO_INSTATIATE_ALL_MAP_GEOM2D_OF_TYPE(TYPE) \
MACRO_INSTATIATE_GEOM2D_OF_MAP(cRot2D<TYPE>,TYPE);\
MACRO_INSTATIATE_GEOM2D_OF_MAP(cSim2D<TYPE>,TYPE);\
MACRO_INSTATIATE_GEOM2D_OF_MAP(cHomot2D<TYPE>,TYPE);\
MACRO_INSTATIATE_GEOM2D_OF_MAP(cHomogr2D<TYPE>,TYPE);\
MACRO_INSTATIATE_GEOM2D_OF_MAP(cAffin2D<TYPE>,TYPE);

MACRO_INSTATIATE_ALL_MAP_GEOM2D_OF_TYPE(tREAL4);
MACRO_INSTATIATE_ALL_MAP_GEOM2D_OF_TYPE(tREAL8);
MACRO_INSTATIATE_ALL_MAP_GEOM2D_OF_TYPE(tREAL16);

template int DbleAreaPolygOriented(const std::vector<cPtxd<int,2>> & aPolyg);

//=====================================================================================================================


};
