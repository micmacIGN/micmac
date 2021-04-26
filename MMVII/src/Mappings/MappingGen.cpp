#include "include/MMVII_all.h"

namespace MMVII
{

/* ============================================= */
/*      cDataMapping<Type>                       */
/* ============================================= */

     //  =========== Constructors =============

template <class Type,const int DimIn,const int DimOut> 
    cDataMapping<Type,DimIn,DimOut>::cDataMapping(const tPtIn & aEpsJac) :
       mEpsJac          (aEpsJac),
       mJacByFiniteDif  (mEpsJac.x()>0),
       mBufIn1Val       ({tPtIn()})
{
    // BufIn1Val().clear();
    // BufIn1Val().push_back(tPtIn());
    // We need to at least be abble to put one point for finite diff
}

template <class Type,const int DimIn,const int DimOut> 
    cDataMapping<Type,DimIn,DimOut>::cDataMapping() :
    cDataMapping<Type,DimIn,DimOut>(tPtIn::PCste(0.0))
{
}




     //  =========== Compute values =============

    ///   Buffered mode by default calls unbeferred mode
template <class Type,const int DimIn,const int DimOut> 
    const typename cDataMapping<Type,DimIn,DimOut>::tVecOut & 
                   cDataMapping<Type,DimIn,DimOut>::Direct(const tVecIn & aVIn) const
{
   tVecOut & aBufOut = BufOutCleared();
   for (const auto  & aP : aVIn)
      aBufOut.push_back(Direct(aP));
   return aBufOut;
}

    ///   Unbeferred mode  by default calls buferred mode
template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tPtOut  
             cDataMapping<Type,DimIn,DimOut>::Direct(const tPtIn & aPt) const
{
   BufIn1Val()[0] = aPt;
   const tVecOut & aRes = Direct(BufIn1Val());
   return aRes[0];
}

     //  =========== Compute jacobian and vals =============

template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tVecJac & 
             cDataMapping<Type,DimIn,DimOut>::BufJac(tU_INT4 aSz) const
{
   while (mJacReserve.size()<aSz)
         mJacReserve.push_back(tJac(DimIn,DimOut));
   // If too small
   for (tU_INT4 aK=mJacResult.size() ; aK<aSz ; aK++)
       mJacResult.push_back(mJacReserve[aK]);
   // If too big
   while (mJacResult.size()>aSz)
        mJacResult.pop_back();
   return mJacResult;
}



    ///   Buffered mode by default calls finit difference OR  unbeferred mode 
template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tResVecJac
                   cDataMapping<Type,DimIn,DimOut>::Jacobian(const tVecIn & aVIn) const
{
    tU_INT4 aNbIn = aVIn.size();
    tVecOut & aJBufOut = JBufOutCleared();
    tVecJac & aBufJac = BufJac(aNbIn);
    tResVecJac aRes(&aJBufOut,&aBufJac);
    // tResVecJac aRes(nullptr,nullptr);
    if (mJacByFiniteDif)
    {
       // tU_INT4 aNbPByJac = 1+2*DimIn;
       tU_INT4 aNbInBuf = std::max(tU_INT4(1),tU_INT4(aNbIn/(1+2*DimIn)));
       for (tU_INT4 aKpt0=0 ; aKpt0<aNbIn ; aKpt0+=aNbInBuf)
       {
          tU_INT4 aKpt1 = std::min(aKpt0+aNbInBuf,aNbIn);
          tVecIn& aBufIn = BufInCleared();
          for (tU_INT4 aKpt=aKpt0 ; aKpt<aKpt1 ; aKpt++)
          {
              tPtIn aPK = aVIn[aKpt];
              aBufIn.push_back(aPK);
              for (int aD=0 ; aD<DimIn ; aD++)
              {
                  aPK[aD] -= mEpsJac[aD];
                  aBufIn.push_back(aPK);
                  aPK[aD] += 2.0*mEpsJac[aD];
                  aBufIn.push_back(aPK);
                  aPK[aD] -= mEpsJac[aD];
              }
          }
          const tVecOut & aResOut =   Direct(aBufIn);
          int aInd = 0;
          for (tU_INT4 aKpt=aKpt0 ; aKpt<aKpt1 ; aKpt++)
          {
             aJBufOut.push_back(aResOut[aInd++]);
             for (int aD=0 ; aD<DimIn ; aD++)
             {
                tPtOut aPm = aResOut[aInd++];
                tPtOut aGrad = (aResOut[aInd++]-aPm) / (2.0*mEpsJac[aD]);

                SetCol(aBufJac[aKpt],aD,aGrad);
             }
          }
       }
    }
    else
    {
        for (tU_INT4 aKpt=0 ; aKpt<aNbIn ; aKpt++)
        {
           auto [aPt,aJac] = Jacobian(aVIn[aKpt]);
           aJBufOut.push_back(aPt);
           aBufJac[aKpt] = aJac;
        }
    }
    return aRes;
}

template <class Type,const int DimIn,const int DimOut> 
    typename cDataMapping<Type,DimIn,DimOut>::tResJac
                   cDataMapping<Type,DimIn,DimOut>::Jacobian(const tPtIn & aPtIn) const
{
   BufIn1Val()[0] = aPtIn;
   tResVecJac  aResVec = Jacobian(BufIn1Val());
   return tResJac(aResVec.first->at(0),aResVec.second->at(0));
}

/* ============================================= */
/*                cMapping<Type>                 */
/* ============================================= */

template <class Type,const int DimIn,const int DimOut>  
       cMapping<Type,DimIn,DimOut>::cMapping(tDataMap * aDM) :
            mPtr    (aDM),
            mRawPtr (aDM)
{
}

/* ============================================= */
/*            cInvertByIter<Type,Dim>            */
/*            - cStrPtInvDIM<Type,Dim>           */
/* ============================================= */

template <class Type,const int Dim>  struct cStrPtInvDIM
{
    public :
        tU_INT4          mNum;
        Type             mCurEr;
        cPtxd<Type,Dim>  mCurInv;
        cPtxd<Type,Dim>  mCurVal;
        cPtxd<Type,Dim>  mNewInv;
        cPtxd<Type,Dim>  mNewVal;
        cPtxd<Type,Dim>  mPTarget;
};

template <class Type,const int Dim> class cInvertDIMByIter
{
    public :
      typedef cDataInvertibleMapping<Type,Dim>     tDIM;
      typedef cStrPtInvDIM<Type,Dim>               tSPIDIM;
      typedef typename  tDIM::tPt        tPt;
      typedef typename  tDIM::tVecPt     tVecPt;
      typedef typename  tDIM::tResVecJac tResVecJac;


      cInvertDIMByIter(const tDIM & aDIM,const tVecPt & aVTarget);
    private :
      void OneIterInversion();

      const tDIM &           mDIM;
      std::vector<tSPIDIM>   mVInv;
      std::vector<tU_INT4>   mVSubSet;  // Subset of index to compute
      tSPIDIM & InvOfKSubS(tU_INT4 aInd) {return mVInv.at(mVSubSet.at(aInd));}
};

template <class Type,const int Dim> 
   void cInvertDIMByIter<Type,Dim>::OneIterInversion()
{
   // Put in aVCurInv the curent estimation of inverses
   tVecPt & aVCurInv =  mDIM.BufInCleared();
   for (tU_INT4 aKInd=0 ; aKInd<mVSubSet.size();  aKInd++)
   {
      aVCurInv.push_back(InvOfKSubS(aKInd).mCurInv);
   }

   //  Compute vals ans jacobian at current inverse
   tResVecJac  aVJ = mDIM.Jacobian(aVCurInv);

   tU_INT4 aNewInd=0;
   for (tU_INT4 aKInd=0 ; aKInd<mVSubSet.size();  aKInd++)
   {
       tSPIDIM & aSInv = InvOfKSubS(aKInd);
       aSInv.mCurVal =  aVJ.first->at(aKInd);
       tPt  aEr = aSInv.mPTarget-aSInv.mCurVal;
       aSInv.mCurEr = Norm2(aEr);
       if (aSInv.mCurEr< mDIM.mDTolInv)
       {
            // Nothing to do , mCurInv is a good inverse
       }
       else
       {
            // Use Jacobian to compute the correction giving the error
            tPt aCor =  SolveCol(aVJ.second->at(aKInd),aEr);
            aSInv.mNewInv = aSInv.mCurInv+ aCor; // Restimate inverse with correction
            aVCurInv.at(aNewInd) = aSInv.mNewInv; // Put new inverse in vect to evaluate
            mVSubSet.at(aNewInd) = aSInv.mNum; //this one is in the new subset
            aNewInd++;
       }
   }
   // We have filled only partially the new indexes
   mVSubSet.resize(aNewInd);
   aVCurInv.resize(aNewInd);

   const  tVecPt &  aVO =  mDIM.Direct(aVCurInv);

   for (tU_INT4 aKInd=0 ; aKInd<mVSubSet.size();  aKInd++)
   {
        // tPt aNewVal = 
   }
   
FakeUseIt(aVO);
}


template <class Type,const int Dim> 
     cInvertDIMByIter<Type,Dim>::cInvertDIMByIter(const tDIM & aDIM,const tVecPt & aVTarget) :
       mDIM (aDIM)

{
    mVSubSet.clear();
    mVInv.clear();
    const tVecPt & aVInit = mDIM.RoughInv()->Direct(aVTarget);
    for (tU_INT4 aKPt=0; aKPt<aVInit.size() ; aKPt++)
    {
       tSPIDIM aStr;
       aStr.mNum = aKPt;
       aStr.mPTarget = aVTarget[aKPt];
       aStr.mCurInv = aVInit[aKPt];
       mVSubSet.push_back(aKPt);
       mVInv.push_back(aStr);
    }

}


/* ============================================= */
/*      cDataInvertibleMapping<Type>             */
/* ============================================= */

template <class Type,const int Dim>
   cDataInvertibleMapping<Type,Dim>::cDataInvertibleMapping() :
       mRoughInv        (nullptr)
{
}

template <class Type,const int Dim> 
    void cDataInvertibleMapping<Type,Dim>::SetRoughInv(tMap aMap,const Type & aDistTol,int aNbIterMaxInv)
{
   mDTolInv  = aDistTol;
   mNbIterMaxInv = aNbIterMaxInv;
   mRoughInv = aMap;
}

template <class Type,const int Dim>
   const typename cDataInvertibleMapping<Type,Dim>::tDataMap * 
                  cDataInvertibleMapping<Type,Dim>::RoughInv() const
{
       return mRoughInv.DM();
}


#if (0)


#endif

template <class Type,const int Dim>
    const typename cDataInvertibleMapping<Type,Dim>::tVecPt & 
                   cDataInvertibleMapping<Type,Dim>::ComputeInverse(const tVecPt & aVTarget) const
{
    const tVecPt & aVInit = mRoughInv.DM()->Direct(aVTarget);
/*
    std::vector<tU_INT1>   aVSubSetToC;  // Subset of index to compute
    std::vector<cPtInvDIM> aVInv;

    for (tU_INT4 aKPt=0; aKPt<aVInit.size() ; aKPt++)
    {
       cPtInvDIM<Type,Dim> aPtID;
       aP.mNum = aKPt;
       aP.Target = aVTarget[aKPt];
       aP.mCurInv = aVInit[aKPt]
       aP.mLastD   = 1e10 * mDTolInv;
       aVSubSetToC.push_back(aKPt);
       aVInv.push_back(aP);
    }
*/

/*
    bool Cont = true;
    while (Cont)
    {
        tVecPt aVCurs;
        aBufIn.clear();
        for (tU_INT4 aKSel=0 ; aKSel<aSubSetToC.size() ; aKSel++)
        {
            aBufIn.push_back(aVInv[aVSubSetToC[aKSel]].mCurInv);
        }
    }
*/

    return aVInit;
}


/* ============================================= */
/*        cMappingIdentity<Type>                 */
/* ============================================= */


template <class Type,const int Dim>
    const typename cMappingIdentity<Type,Dim>::tVecPt & 
                   cMappingIdentity<Type,Dim>::Direct(const tVecPt & aVIn) const
{
   tVecPt & aBufOut = this->BufOut();
   aBufOut = aVIn;
   return aBufOut;
}


template <class Type,const int Dim>
    typename cMappingIdentity<Type,Dim>::tPt 
                   cMappingIdentity<Type,Dim>::Direct(const tPt & aPt) const
{
   return aPt;
}

     /* ============================================= */
     /* ============================================= */
     /* ============================================= */

/*
template class cDataInvertibleMapping<double,2>;
template class cMappingIdentity<double,2>;
template class cMappingIdentity<double,3>;
*/
template class cDataMapping<double,3,2>;
template class cDataMapping<double,3,3>;
template class cDataMapping<double,2,2>;
template class cDataMapping<double,2,3>;

#define INSTANCE_ONE_DIM_MAPPING(DIM)\
template class cMappingIdentity<double,DIM>;\
template class cDataInvertibleMapping<double,DIM>;\
template class cInvertDIMByIter<double,DIM>;\

INSTANCE_ONE_DIM_MAPPING(2)
INSTANCE_ONE_DIM_MAPPING(3)

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
          cPt3dLR aGx(0.3,1.0+0.2*cos(2*aP.x()),(-2*aP.x())/aDiv);
          cPt3dLR aGy(0.9+0.1*cos(aP.y()),0,(-2*aP.y())/aDiv);

          cDenseMatrix<tREAL16> aRes(2,3);
          SetCol(aRes,0,aGx);
          SetCol(aRes,1,aGy);
          return aRes;
       }
};

class cTestMap1 : public cDataMapping<tREAL16,2,3>
{
    public :
        cPt3dLR Direct(const cPt2dLR & aP) const override {return cTestMapp::Image(aP);}
        cTestMap1() :
            cDataMapping<tREAL16,2,3>(cPt2dLR(1e-3,1e-3))
        {
        }
};

class cTestMap2 : public cDataMapping<tREAL16,2,3>
{
    public :
        const std::vector<cPt3dLR> & Direct(const std::vector<cPt2dLR> & aVIn) const override 
        {
            std::vector<cPt3dLR> & aBufOut = BufOutCleared();
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
        std::vector<cPt2dLR>  aVIn;
        std::vector<cPt3dLR>  aVOut;
        std::vector<cDenseMatrix<tREAL16>>  aVDif;
        for (tU_INT4 aKP=0 ; aKP<aSzV ; aKP++)
        {
             cPt2dLR aP= cPt2dLR::PRandC();
             aVIn.push_back(aP);
             aVOut.push_back(cTestMapp::Image(aP));
             aVDif.push_back(cTestMapp::Grad(aP));
        }
        TypeMap aMap;
        cDataMapping<tREAL16,2,3> * aPM = &aMap;
        // compute vector of input
        const auto & aVO2 = aPM->Direct(aVIn);
        MMVII_INTERNAL_ASSERT_bench(aVOut.size()==aSzV,"Sz in BenchMapping");
        MMVII_INTERNAL_ASSERT_bench(aVO2.size() ==aSzV,"Sz in BenchMapping");

        // check vector  of input with grad + unbefferd
        for (tU_INT4 aKP=0 ; aKP<aSzV ; aKP++) 
        {
            MMVII_INTERNAL_ASSERT_bench(Norm2(aVOut[aKP] - aVO2[aKP])<1e-5,"Buf/UnBuf in mapping");
            MMVII_INTERNAL_ASSERT_bench(Norm2(aVOut[aKP] - aPM->Direct(aVIn[aKP]) )<1e-5,"Buf/UnBuf in mapping");
        }

        // check jacobian
        auto [aVO3,aVG3] = aPM->Jacobian(aVIn);
        for (tU_INT4 aKP=0 ; aKP<aSzV ; aKP++)
        {
            //std::cout<<aVOut[aKP]-aVO3->at(aKP)<< aVDif[aKP].DIm().L2Dist((*aVG3)[aKP].DIm())<< "\n";
            MMVII_INTERNAL_ASSERT_bench(Norm2(aVOut[aKP]-(*aVO3)[aKP])<1e-5,"Val in Grad in mapping");
            MMVII_INTERNAL_ASSERT_bench(aVDif[aKP].L2Dist(aVG3->at(aKP))<1e-3,"Jacobian in mapping");
            // std::cout << aVDif[aKP].L2Dist(aVG3->at(aKP)) << "\n";
        }
    }
}

void BenchMapping(cParamExeBench & aParam)
{
    if (! aParam.NewBench("GenMapping")) return;

    OneBenchMapping<cTestMap1>(aParam);
    OneBenchMapping<cTestMap2>(aParam);
    aParam.EndBench();
}

/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


};
