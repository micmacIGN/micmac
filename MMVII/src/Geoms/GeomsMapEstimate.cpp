#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_SysSurR.h"

namespace MMVII
{


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
/*          cHomogr2D         */
/* ========================== */
template <class Type>  Type cHomogr2D<Type>::AvgDistL1(tCRVPts aVIn,tCRVPts aVOut)
{
	return MMVII::AvgDistL1(*this,aVIn,aVOut);
}



/* ========================== */
/*       INSTANTIATION        */
/* ========================== */


#define  MACRO_DEFINE_MAP_ESTIMATE(TMAP)\
template <class Type>  TMAP<Type> TMAP<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)\
{\
    return cMapEstimate<TMAP<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);\
}\
template <class Type>  TMAP<Type> TMAP<Type>::StdGlobEstimate ( tCRVPts aVIn, tCRVPts aVOut, tTypeElem* aRes, tCPVVals   aVW, cParamCtrlOpt aParam)\
{\
    return  cMapEstimate<TMAP<Type> >::LeastSquareNLEstimate(aVIn,aVOut,aRes,aVW,aParam); \
}\
template <class Type> \
         TMAP<Type> TMAP<Type>::LeastSquareRefine(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVW)const \
{ \
    return cMapEstimate<TMAP<Type>>::LeastSquareRefine(*this,aVIn,aVOut,aRes2,aVW);\
}\
template class TMAP<tREAL4>;\
template class TMAP<tREAL8>;\
template class TMAP<tREAL16>;

MACRO_DEFINE_MAP_ESTIMATE(cIsometry3D)
MACRO_DEFINE_MAP_ESTIMATE(cRot2D)
MACRO_DEFINE_MAP_ESTIMATE(cSim2D)
MACRO_DEFINE_MAP_ESTIMATE(cHomot2D)
MACRO_DEFINE_MAP_ESTIMATE(cHomogr2D)
MACRO_DEFINE_MAP_ESTIMATE(cAffin2D)


//=====================================================================================================================

/*
template <class Type>  cIsometry3D<Type> cIsometry3D<Type>::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest)
{
    return cMapEstimate<cIsometry3D<Type>>::RansacL1Estimate(aVIn,aVOut,aNbTest);
}
template tPoseR tPoseR::RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);

template <class Type>  
         cIsometry3D<Type> cIsometry3D<Type>::LeastSquareRefine(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2,tCPVVals aVW)const
{
    return cMapEstimate<cIsometry3D<Type>>::LeastSquareRefine(*this,aVIn,aVOut,aRes2,aVW);
}
template tPoseR tPoseR::LeastSquareRefine(tCRVPts aVIn,tCRVPts aVOut,tTypeElem * aRes2,tCPVVals aVW) const;
*/

};
