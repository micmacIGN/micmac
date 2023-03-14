#include "MMVII_Mappings.h"
#include "MMVII_Triangles.h"

namespace MMVII
{

/* ============================================= */
/*                                               */
/*      cDataBoundedSet<Type>                    */
/*                                               */
/* ============================================= */

template <class Type,const int Dim>
    cDataBoundedSet<Type,Dim>::cDataBoundedSet(const tBox & aBox) :
       mBox(aBox)
{
}

template <class Type,const int Dim>
    cDataBoundedSet<Type,Dim>::~cDataBoundedSet()
{
}

template <class Type,const int Dim>
    bool cDataBoundedSet<Type,Dim>::InsideWithBox(const tPt & aPt) const
{
   return InsidenessWithBox(aPt) >= 0.0 ;
}

template <class Type,const int Dim>
    bool cDataBoundedSet<Type,Dim>::InsideWithBox(const cTriangle<Type,Dim> & aTri) const
{
   for (int aK=0 ; aK<3 ; aK++)
      if (! InsideWithBox(aTri.Pt(aK)))
         return false;
   return true;
}


template <class Type,const int Dim>
    bool cDataBoundedSet<Type,Dim>::Inside(const tPt & aPt) const
{
   return Insideness(aPt) >= 0.0 ;
}


template <class Type,const int Dim>
    tREAL8 cDataBoundedSet<Type,Dim>::InsidenessWithBox(const tPt & aP) const
{
    return std::min(mBox.Insideness(aP),Insideness(aP));
}

template <class Type,const int Dim>
    tREAL8 cDataBoundedSet<Type,Dim>::Insideness(const tPt &) const
{
    return 1e10;
}


template <class Type,const int Dim>
    const typename cDataBoundedSet<Type,Dim>::tBox & cDataBoundedSet<Type,Dim>::Box() const
{
   return mBox;
}

///  algo : generate random in bounding box, untill it is inside
template <class Type,const int Dim>
    typename cDataBoundedSet<Type,Dim>::tPt  cDataBoundedSet<Type,Dim>::GeneratePointInside() const
{
   tPt aRes = mBox.GeneratePointInside();
   while (! Inside(aRes))
         aRes = mBox.GeneratePointInside();
   return aRes;
}

/// algo : generate the regular grid in box, select points if inside
template <class Type,const int Dim>
    void cDataBoundedSet<Type,Dim>::GridPointInsideAtStep(tVecPt& aRes,Type aStepMoy) const
{
     cPtxd<int,Dim>  aNb =  Pt_round_up(mBox.Sz()/aStepMoy); 
     cPixBox<Dim>  aPixB (aNb+cPtxd<int,Dim>::PCste(1));
     tPt   aStep  = DivCByC(mBox.Sz(),ToR(aNb));

     for (const auto & aPix : aPixB)
     {
          tPt  aPt = mBox.P0() + MulCByC(ToR(aPix),aStep);
          if (Inside(aPt))
             aRes.push_back(aPt);
     }
}

template <class Type,const int Dim>
    void cDataBoundedSet<Type,Dim>::GridPointInsideOfNbPoints(tVecPt& aRes,int aNbPts) const
{
     Type aVolMoy = mBox.NbElem() / double(aNbPts);
     Type aStepMoy = pow(aVolMoy,1/double(Dim));

     GridPointInsideAtStep(aRes,aStepMoy);
}

/* ============================================= */
/*                                               */
/*      cSphereBoundedSet<Type,Dim>              */
/*                                               */
/* ============================================= */

template <class Type,const int Dim> 
     cSphereBoundedSet<Type,Dim>::cSphereBoundedSet(const tBox & aBox,const tPt& aC,const Type & aRadius) :
        cDataBoundedSet<Type,Dim> (aBox),
        mCenter  (aC),
        mRadius  (aRadius)
{
}

template <class Type,const int Dim> 
   tREAL8   cSphereBoundedSet<Type,Dim>::Insideness(const tPt & aPt) const
{
   return  mRadius -Norm2(aPt-mCenter) ;
}

/* ============================================= */
/*                                               */
/*      cDataMappedBoundedSet<Type,Dim>          */
/*                                               */
/* ============================================= */

template <class Type,const int Dim>
   cDataMappedBoundedSet<Type,Dim>::cDataMappedBoundedSet
   (
          const tDBSet * aSet,
          const tDIMap * aMap,
          bool AdoptSet,
          bool AdoptMap
   ) :
     cDataBoundedSet<Type,Dim>(aMap->BoxOfCorners(aSet->Box())),
     mSet       (aSet),
     mMap       (aMap),
     mAdoptSet  (AdoptSet),
     mAdoptMap  (AdoptMap)
{
}

template <class Type,const int Dim>
   cDataMappedBoundedSet<Type,Dim>::~cDataMappedBoundedSet ()
{
    if (mAdoptSet)  delete mSet;
    if (mAdoptMap)  delete mMap;
}

template <class Type,const int Dim> tREAL8 cDataMappedBoundedSet<Type,Dim>::Insideness (const tPt & aPt) const
{
        return mSet->Insideness(mMap->Inverse(aPt));
}

template class cDataMappedBoundedSet<tREAL8,2>;
                                                      

/* ============================================= */
/*                                               */
/*      cDataMapping<Type>                       */
/*                                               */
/* ============================================= */

     //  =========== Constructors =============

template <class Type,const int DimIn,const int DimOut> 
    cDataMapping<Type,DimIn,DimOut>::cDataMapping(const tPtIn & aEpsJac) :
       mEpsJac          (aEpsJac),
       mJacByFiniteDif  (mEpsJac.x()>0)
#if (! MAP_STATIC_BUF)
   ,  mBufIn1Val       ({tPtIn()})
#endif
{
    // BufIn1Val().clear();
    // BufIn1Val().push_back(tPtIn());
    // We need to at least be abble to put one point for finite diff
}

template <class Type,const int DimIn,const int DimOut> 
    cDataMapping<Type,DimIn,DimOut>::cDataMapping(const Type & aVal) :
    cDataMapping<Type,DimIn,DimOut>(tPtIn::PCste(aVal))
{
}


template <class Type,const int DimIn,const int DimOut> 
    cDataMapping<Type,DimIn,DimOut>::cDataMapping() :
    cDataMapping<Type,DimIn,DimOut>(Type(0.0))
{
}

template <class Type,const int DimIn,const int DimOut> 
    cDataMapping<Type,DimIn,DimOut>::~cDataMapping()
{
}



     //  =========== Compute values =============

    ///   Buffered mode by default calls unbeferred mode
template <class Type,const int DimIn,const int DimOut> 
    const typename cDataMapping<Type,DimIn,DimOut>::tVecOut & 
                   cDataMapping<Type,DimIn,DimOut>::Values(tVecOut & aBufOut,const tVecIn & aVIn) const
{
/**/MACRO_CHECK_RECURS_BEGIN;
    for (const auto  & aP : aVIn)
        aBufOut.push_back(Value(aP));
/**/MACRO_CHECK_RECURS_END;
    return aBufOut;
}

    ///   call previous method with local buffer
template <class Type,const int DimIn,const int DimOut> 
    const typename cDataMapping<Type,DimIn,DimOut>::tVecOut & 
                   cDataMapping<Type,DimIn,DimOut>::Values(const tVecIn & aVIn) const
{
   return Values(BufOutCleared(),aVIn);
}




    ///   Unbeferred mode  by default calls buferred mode
template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tPtOut  
             cDataMapping<Type,DimIn,DimOut>::Value(const tPtIn & aPt) const
{
   BufIn1Val()[0] = aPt;
   const tVecOut & aRes = Values(BufIn1Val());
   return aRes[0];
}

     //  =========== Compute jacobian and vals =============

template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tVecJac & 
             cDataMapping<Type,DimIn,DimOut>::BufJac(tU_INT4 aSz) 
#if (!MAP_STATIC_BUF)
                                                                     const
#endif
{
#if (MAP_STATIC_BUF)
   cMemManager::SetActiveMemoryCount(false);  // static vector of shared matrix will never be unallocated
   static tVecJac  mJacReserve;
   static tVecJac  mJacResult;
#endif
   while (mJacReserve.size()<aSz)
         mJacReserve.push_back(tJac(DimIn,DimOut));
   // If too small
   for (tU_INT4 aK=mJacResult.size() ; aK<aSz ; aK++)
       mJacResult.push_back(mJacReserve[aK]);
   // If too big
   while (mJacResult.size()>aSz)
        mJacResult.pop_back();
#if (MAP_STATIC_BUF)
   cMemManager::SetActiveMemoryCount(true);
#endif
   return mJacResult;
}



    ///   Buffered mode by default calls finit difference OR  unbeferred mode 
template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tCsteResVecJac
                   cDataMapping<Type,DimIn,DimOut>::Jacobian(tResVecJac aRes,const tVecIn & aVIn) const
{
/**/MACRO_CHECK_RECURS_BEGIN;

    tU_INT4 aNbIn = aVIn.size();
    tVecOut & aJBufOut = *(aRes.first);
    tVecJac & aBufJac =  *(aRes.second);
/*
    tResVecJac aRes(&aJBufOut,&aBufJac);
*/
    // tResVecJac aRes(nullptr,nullptr);
    if (mJacByFiniteDif)
    {
       // compute number in buf , be sure at least one
       tU_INT4 aNbInBuf = std::max(tU_INT4(1),tU_INT4(aNbIn/(1+2*DimIn)));
       for (tU_INT4 aKpt0=0 ; aKpt0<aNbIn ; aKpt0+=aNbInBuf)
       {
          tU_INT4 aKpt1 = std::min(aKpt0+aNbInBuf,aNbIn);
          tVecIn& aBufIn = JBufInCleared(); // buf in of jacobian, dif of buf in used in values
          for (tU_INT4 aKpt=aKpt0 ; aKpt<aKpt1 ; aKpt++) // put all points
          {
              tPtIn aPK = aVIn[aKpt];  // make a copy of input
              aBufIn.push_back(aPK);
              for (int aD=0 ; aD<DimIn ; aD++)  // parse all dims
              {
                  aPK[aD] -= mEpsJac[aD];
                  aBufIn.push_back(aPK);   // put xk -eps
                  aPK[aD] += 2.0*mEpsJac[aD];
                  aBufIn.push_back(aPK);  // put xk+eps
                  aPK[aD] -= mEpsJac[aD];  // restore initial value of xk
              }
          }
          const tVecOut & aResOut =   Values(aBufIn); // compute all val 
          int aInd = 0;  // will parse value in same order 
          for (tU_INT4 aKpt=aKpt0 ; aKpt<aKpt1 ; aKpt++)
          {
             aJBufOut.push_back(aResOut[aInd++]);  // first we have the point
             for (int aD=0 ; aD<DimIn ; aD++)
             {
                tPtOut aPm = aResOut[aInd++];  // get xk -eps
                tPtOut aGrad = (aResOut[aInd++]-aPm) / (2.0*mEpsJac[aD]);  // get xk+ eps, deduce finite difference

                SetCol(aBufJac[aKpt],aD,aGrad); // fill the colums of jacobian
             }
          }
       }
    }
    else
    {
        // else call elem function
        for (tU_INT4 aKpt=0 ; aKpt<aNbIn ; aKpt++)
        {
           auto [aPt,aJac] = Jacobian(aVIn[aKpt]);
           aJBufOut.push_back(aPt);
           aBufJac[aKpt] = aJac;
        }
    }
/**/MACRO_CHECK_RECURS_END;
      // StdOut() <<  "TO CHECK LINE " << __LINE__  << " of " << __FILE__ << "\n";
     return tCsteResVecJac(aRes.first,aRes.second);
     //  return aRes;  // dont understand why not that, maybe some type checking with some version of compilor ?
}

template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tCsteResVecJac
                   cDataMapping<Type,DimIn,DimOut>::Jacobian(const tVecIn & aVIn) const
{
/**/MACRO_CHECK_RECURS_BEGIN;
    tU_INT4 aNbIn = aVIn.size();
    tVecOut & aJBufOut = JBufOutCleared();
    tVecJac & aBufJac = BufJac(aNbIn);
    tResVecJac aRes(&aJBufOut,&aBufJac);
/**/MACRO_CHECK_RECURS_END;
    return Jacobian(aRes,aVIn);
}

template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tResJac
                   cDataMapping<Type,DimIn,DimOut>::Jacobian(const tPtIn & aPtIn) const
{
   BufIn1Val()[0] = aPtIn;
   tCsteResVecJac  aResVec = Jacobian(BufIn1Val());
   return tResJac(aResVec.first->at(0),aResVec.second->at(0));
}


template <class Type,const int DimIn,const int DimOut>
      cTplBox<Type,DimOut> cDataMapping<Type,DimIn,DimOut>::BoxOfCorners(const cTplBox<Type,DimIn>& aBoxIn) const
{
   typename cTplBox<Type,DimIn>::tCorner aCornersIn;
   aBoxIn.Corners(aCornersIn);
   std::vector<tPtIn> aVIn(&(aCornersIn[0]),&(aCornersIn[0]) + cTplBox<Type,DimIn>::NbCorners);
   const std::vector<tPtOut> & aVOut = Values(aVIn);

   cTplBoxOfPts<Type,DimOut> aBoxOfPts;
   for (const auto & aP : aVOut)
       aBoxOfPts.Add(aP);

   return  cTplBox<Type,DimOut>(aBoxOfPts.P0(),aBoxOfPts.P1());
   
}

template <class Type,const int DimIn,const int DimOut>
      cTriangle<Type,DimOut> cDataMapping<Type,DimIn,DimOut>::TriValue(const cTriangle<Type,DimIn> &aTriIn) const
{
	std::vector<tPtIn> aVPtIn;

	for (int aK=0 ; aK<3 ; aK++)
	{
             aVPtIn.push_back(aTriIn.Pt(aK));
	}
	std::vector<tPtOut> aVPtOut = Values(aVPtIn);

	return cTriangle<Type,DimOut> (aVPtOut[0],aVPtOut[1],aVPtOut[2]);
}

/* ============================================= */
/*                                               */
/*      cDataNxNMapping<Type>                    */
/*                                               */
/* ============================================= */

template <class Type,const int Dim>
      cBijAffMapElem<Type,Dim> cDataNxNMapping<Type,Dim>::Linearize(const tPt & aPtIn) const
{
    tResJac   aRJ = this->Jacobian(aPtIn);

    tPt& aImP = aRJ.first;
    tJac&  aMat = aRJ.second;

    return cBijAffMapElem<Type,Dim>(aMat.Dup(),aImP-aMat*aPtIn);
}

template <class Type,const int Dim>
      cDataNxNMapping<Type,Dim>::cDataNxNMapping() :
           cDataMapping<Type,Dim,Dim> ()
{
}

template <class Type,const int Dim>
      cDataNxNMapping<Type,Dim>::cDataNxNMapping(const tPt &  aEps) :
           cDataMapping<Type,Dim,Dim> (aEps)
{
}


/* ============================================= */
/*                                               */
/*                cMapping<Type>                 */
/*                                               */
/* ============================================= */

/*
template <class Type,const int DimIn,const int DimOut>  
       cMapping<Type,DimIn,DimOut>::cMapping(tDataMap * aDM) :
            mPtr    (aDM),
            mRawPtr (aDM)
{
}
*/


/* ============================================= */
/*                                               */
/*        cMappingIdentity<Type>                 */
/*                                               */
/* ============================================= */


template <class Type,const int Dim>
    const typename cMappingIdentity<Type,Dim>::tVecPt & 
                   cMappingIdentity<Type,Dim>::Values(tVecPt & aVRes,const tVecPt & aVIn) const
{
   // tVecPt & aBufOut = this->BufOut();
   // aBufOut = aVIn;
   aVRes = aVIn;
   return aVRes;
}


template <class Type,const int Dim>
    typename cMappingIdentity<Type,Dim>::tPt 
                   cMappingIdentity<Type,Dim>::Value(const tPt & aPt) const
{
   return aPt;
}

template <class Type,const int Dim>
         cMappingIdentity<Type,Dim>::cMappingIdentity() :
             cDataNxNMapping<Type,Dim> (tPt::PCste(Type(1)))
{
}

     /* ============================================= */
     /* ============================================= */
     /* ============================================= */

#define INSTANCE_TWO_DIM_MAPPING(DIM1,DIM2)\
template class cDataMapping<double,DIM1,DIM2>;

//template class cMapping<double,DIM1,DIM2>;

#define INSTANCE_ONE_DIM_MAPPING(DIM)\
template class cDataNxNMapping<double,DIM>;\
template class cSphereBoundedSet<double,DIM>;\
template class cDataBoundedSet<double,DIM>;\
template class cMappingIdentity<double,DIM>;\
INSTANCE_TWO_DIM_MAPPING(DIM,2);\
INSTANCE_TWO_DIM_MAPPING(DIM,3);


INSTANCE_ONE_DIM_MAPPING(2)
INSTANCE_ONE_DIM_MAPPING(3)


/* ============================================= */
/* ============================================= */
/* ====                                      === */       
/* ====            CHECK/BENCH               === */ 
/* ====                                      === */
/* ============================================= */
/* ============================================= */

/**   class used to make some basic bench on mappings,
      it define a smooth mapping (Image) R3->R2,
      and its hancrafter gradient
*/

class cTestMapp 
{
   public :
       static cPt3dLR Image(const cPt2dLR & aP) 
       {
          float x=  0.3*aP.x() + 0.9*aP.y() + 0.1*sin(aP.y());
          float y=  aP.x() + 0.1*sin(2*aP.x());
          float z=  1/(1+Square(aP.x())+Square(aP.y()));
          return cPt3dLR (x,y,z);
       }
       static cDenseMatrix<tREAL16> Grad(const cPt2dLR & aP) 
       {
          double aDiv = Square(1+Square(aP.x())+Square(aP.y()));

          cPt3dLR aGx(0.3,1.0+0.2*cos(2*aP.x()),(-2*aP.x())/aDiv); // handcrafted  dImage/dx
          cPt3dLR aGy(0.9+0.1*cos(aP.y()),0,(-2*aP.y())/aDiv);     // hancrafted   dImage/dy

          cDenseMatrix<tREAL16> aRes(2,3); // Gradient is a Matrix 2 colum, 3line
          SetCol(aRes,0,aGx);
          SetCol(aRes,1,aGy);
          return aRes;
       }
};

/**  Test a mapping that defines values as elementary ones & and no grad but an epsilon value*/
class cTestMap1 : public cDataMapping<tREAL16,2,3>
{
    public :
        cPt3dLR Value(const cPt2dLR & aP) const override {return cTestMapp::Image(aP);}
        cTestMap1() :
            cDataMapping<tREAL16,2,3>(cPt2dLR(1e-3,1e-3))
        {
        }
};

/**  Test a mapping that defines values in a buffer and  no grad */
class cTestMap2 : public cDataMapping<tREAL16,2,3>
{
    public :
        const std::vector<cPt3dLR> & Values
                   (
                       std::vector<cPt3dLR> & aBufOut,
                       const std::vector<cPt2dLR> & aVIn
                   ) const override 
        {
            // std::vector<cPt3dLR> & aBufOut = *aRes;
            for (const auto & aP : aVIn)
                aBufOut.push_back(cTestMapp::Image(aP));
            return aBufOut;
        }
        cTestMap2() :
            cDataMapping<tREAL16,2,3>(cPt2dLR(1e-3,1e-3))
        {
        }
};


template <class TypeMap> void OneBenchMapping(cParamExeBench & aParam)
{

    for (int aKTest=0 ; aKTest<1000; aKTest++)
    {
        tU_INT4 aSzV = RandUnif_N(50);
        std::vector<cPt2dLR>  aVIn;  // Vector of random input points
        std::vector<cPt3dLR>  aVOut;  // Image of this point by the "Ground Truth"
        std::vector<cDenseMatrix<tREAL16>>  aVDif; // Differiential of these point with the ground truth
        for (tU_INT4 aKP=0 ; aKP<aSzV ; aKP++)
        {
             cPt2dLR aP= cPt2dLR::PRandC();
             aVIn.push_back(aP);
             aVOut.push_back(cTestMapp::Image(aP));
             aVDif.push_back(cTestMapp::Grad(aP));
        }
        TypeMap aMap;
        cDataMapping<tREAL16,2,3> * aPM = &aMap; // use a pointer because virtuality
        // compute vector of input
        const auto & aVO2 = aPM->Values(aVIn);
        // check size
        MMVII_INTERNAL_ASSERT_bench(aVOut.size()==aSzV,"Sz in BenchMapping");
        MMVII_INTERNAL_ASSERT_bench(aVO2.size() ==aSzV,"Sz in BenchMapping");

        // check vector  of input with by buffer (VO2) and by Value elem
        for (tU_INT4 aKP=0 ; aKP<aSzV ; aKP++) 
        {
            MMVII_INTERNAL_ASSERT_bench(Norm2(aVOut[aKP] - aVO2[aKP])<1e-5,"Buf/UnBuf in mapping");
            MMVII_INTERNAL_ASSERT_bench(Norm2(aVOut[aKP] - aPM->Value(aVIn[aKP]) )<1e-5,"Buf/UnBuf in mapping");
        }

        // check jacobian
        auto [aVO3,aVG3] = aPM->Jacobian(aVIn);
        for (tU_INT4 aKP=0 ; aKP<aSzV ; aKP++)
        {
            MMVII_INTERNAL_ASSERT_bench(Norm2(aVOut[aKP]-(*aVO3)[aKP])<1e-5,"Val in Grad in mapping");
            MMVII_INTERNAL_ASSERT_bench(aVDif[aKP].L2Dist(aVG3->at(aKP))<1e-3,"Jacobian in mapping");
        }
    }
}

/** Check efficiency MACRO_CHECK_RECURS_BEGIN, this function define no elementary value, nor
    buffered value, so it will activate a fatal error
*/

class cBugRecMap : public cDataMapping<tREAL8,2,2>
{
    public :
};


void BenchMapping(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GenMapping")) return;

    
    if (aParam.GenerateBug("MapRecursion"))
    {
       cBugRecMap aMap;
       aMap.Value(cPt2dr(2,2));
    }

    BenchLeastSqMap(aParam);
    BenchInvertMapping(aParam);
    BenchSymDerMap(aParam);

    OneBenchMapping<cTestMap1>(aParam);
    OneBenchMapping<cTestMap2>(aParam);
    aParam.EndBench();
}

/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


};
