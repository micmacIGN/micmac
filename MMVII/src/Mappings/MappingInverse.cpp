#include "include/MMVII_all.h"

namespace MMVII
{

// extern bool BUGINVMAP;

/* ============================================= */
/*        cDataInvertibleMapping<Type,Dim>       */
/* ============================================= */

// const  tVecPt &  Inverses(tVecPt &,const tVecPt & ) const;

template <class Type,const int Dim>
    cDataInvertibleMapping<Type,Dim>::cDataInvertibleMapping(const tPt& aEps) :
        cDataMapping<Type,Dim,Dim> (aEps)
{
}

template <class Type,const int Dim>
    cDataInvertibleMapping<Type,Dim>::cDataInvertibleMapping() :
        cDataMapping<Type,Dim,Dim> (tPt::PCste(0.0))
{
}

template <class Type,const int Dim>
    const typename cDataInvertibleMapping<Type,Dim>::tVecPt & 
                   cDataInvertibleMapping<Type,Dim>::Inverses(tVecPt & aBufOut,const tVecPt & aVIn) const
{
/**/MACRO_CHECK_RECURS_BEGIN;
    for (const auto  & aP : aVIn)
        aBufOut.push_back(Inverse(aP));
/**/MACRO_CHECK_RECURS_END;
    return aBufOut;
}

template <class Type,const int Dim>
    const typename cDataInvertibleMapping<Type,Dim>::tVecPt &
                   cDataInvertibleMapping<Type,Dim>::Inverses(const tVecPt & aVImages) const
{
   return Inverses(BufInvOutCleared(),aVImages);
}


template <class Type,const int Dim>
    typename cDataInvertibleMapping<Type,Dim>::tPt  
          cDataInvertibleMapping<Type,Dim>::Inverse(const tPt & aPImage) const
{
   this->BufIn1Val()[0] = aPImage;
   const tVecPt & aRes = Inverses(this->BufIn1Val());
   return aRes[0];
}

/* ============================================= */
/*            cInvertByIter<Type,Dim>            */
/*            - cStrPtInvDIM<Type,Dim>           */
/* ============================================= */

template <class Type,const int Dim>  struct cStrPtInvDIM
{
    public :
        typedef cPtxd<Type,Dim> tPt;
        Type Score(const tPt & aP) {return Norm2(mPTarget-aP);}
        void UpdateNew(const Type & anErr,const tPt & aInv,const tPt & aVal)
        {
           if (anErr<mBestErr)
           {
              mBestErr = anErr;
              mBestInv = aInv;
              mBestVal = aVal;
           }
        }
        bool ImprovedEnouh() const {return mBestErr<mThreshNextEr;}

        tU_INT4          mNum;

        Type             mThreshNextEr; // Threshold for next error is accepted

        Type             mBestErr;
        cPtxd<Type,Dim>  mBestInv;
        cPtxd<Type,Dim>  mBestVal;

        // For dicotomy approach, inverse & val at extremities
        cPtxd<Type,Dim>  mInvDic0;
        cPtxd<Type,Dim>  mValDic0;
        cPtxd<Type,Dim>  mInvDic1;
        cPtxd<Type,Dim>  mValDic1;

        cPtxd<Type,Dim>  mPTarget;
};

template <class Type,const int Dim> class cInvertDIMByIter : public cMemCheck
{
    public :
      typedef cDataIterInvertMapping<Type,Dim>     tDIM;
      typedef cStrPtInvDIM<Type,Dim>               tSPIDIM;
      typedef typename  tDIM::tPt        tPt;
      typedef typename  tDIM::tVecPt     tVecPt;
      // typedef typename  tDIM::tResVecJac tResVecJac;
      typedef typename  tDIM::tCsteResVecJac tCsteResVecJac;


      cInvertDIMByIter(const tDIM & aDIM);
      void Init(const tVecPt & aVTarget);
      void OneIterInversion();
      const tU_INT4 Remaining() const {return mVSubSet.size();}

      void PushResult(tVecPt &) const;

    private :
      void OneIterDicot();
      bool Converged(const tSPIDIM & aSInv) const {return aSInv.mBestErr<mDIM.mDTolInv;}

      const tDIM &           mDIM;
      std::vector<tSPIDIM>   mVInv;
      std::vector<tU_INT4>   mVSubSet;   // Subset of index still to compute
      std::vector<tU_INT4>   mVSubDicot;  // Subset of point that dont improve (enough) with jacobian
      tSPIDIM & InvOfKSubS(tU_INT4 aInd) {return mVInv.at(mVSubSet.at(aInd));}
};

template <class Type,const int Dim> 
   void cInvertDIMByIter<Type,Dim>::OneIterDicot()
{
   // Push in aVSampleInv  the indermediar values (interpol inverse) between mInvDic0 and mInvDic1
   tVecPt & aVSampleInv =  mDIM.BufInCleared();
   int aNbInterm = 3;
   for (tU_INT4 aKInd = 0 ; aKInd<mVSubDicot.size() ; aKInd++)
   {
       tSPIDIM & aSInv  =  mVInv.at(mVSubDicot.at(aKInd));
       for (int aKPds=1 ; aKPds<= aNbInterm ; aKPds++)
       {
           Type aPds = aKPds /(1.0+aNbInterm);
           aVSampleInv.push_back(aSInv.mInvDic0*(1-aPds)+aSInv.mInvDic1*aPds);
       }
   }

   // Compute the value of interpolated
   const  tVecPt &  aVSampleVal =  mDIM.Values(aVSampleInv);

   tU_INT4  aIndEchInv=0; // Use to follow the aVSampleVal  and aVSampleInv
   tU_INT4  aIndSubDicot=0; // Use to follow the aVSampleVal  and aVSampleInv
   for (tU_INT4 aKInd = 0 ; aKInd<mVSubDicot.size() ; aKInd++)
   {
       tSPIDIM & aSInv  =  mVInv.at(mVSubDicot.at(aKInd));

       std::vector<tPt>   aVInv;
       std::vector<tPt>   aVVal;

       aVInv.push_back(aSInv.mInvDic0);  // Correponds to pds 0
       aVVal.push_back(aSInv.mValDic0);
       for (int aKPds=1 ; aKPds<= aNbInterm ; aKPds++)
       {
           aVVal.push_back(aVSampleInv.at(aIndEchInv));
           aVInv.push_back(aVSampleVal.at(aIndEchInv));
           aIndEchInv++;
       }
       aVInv.push_back(aSInv.mInvDic1); // Correpond to pds 1
       aVVal.push_back(aSInv.mValDic1);


       cWhitchMin<tU_INT4,Type> aWM(-1,1e30);
       for (tU_INT4 aKEch=0 ; aKEch<aVVal.size() ; aKEch++)
       {
           Type aScore = aSInv.Score(aVVal[aKEch]);
           aWM.Add(aKEch,aScore);
           aSInv.UpdateNew(aScore,aVInv[aKEch],aVVal[aKEch]); // Each tested value must serve
       }

   
       if (aSInv.ImprovedEnouh())
       {
           // Nothing to do, we have reached an acceptable decrease
       }
       else
       {
          // We restrict the the interval arround minimal value KMin, the new interval will
          // be in KMin-1, KMin+1, but truncated when KMin is a bound
          int aKMin = aWM.Index();
          int aK0 = std::max(0,aKMin-1);
          int aK1 = std::min(aKMin+1,int(aVVal.size()) -1);
          aSInv.mInvDic0 = aVInv.at(aK0);
          aSInv.mValDic0 = aVVal.at(aK0);
          aSInv.mInvDic1 = aVInv.at(aK1);
          aSInv.mValDic1 = aVVal.at(aK1);
          mVSubDicot.at(aIndSubDicot++) = aSInv.mNum;
       }
   }
   mVSubDicot.resize(aIndSubDicot);
}

template <class Type,const int Dim> 
   void cInvertDIMByIter<Type,Dim>::OneIterInversion()
{
   // Put in aVCurInv the curent estimation of inverses
   tVecPt & aVCurInv =  mDIM.BufInCleared();
   for (tU_INT4 aKInd=0 ; aKInd<mVSubSet.size();  aKInd++)
   {
      aVCurInv.push_back(InvOfKSubS(aKInd).mBestInv);
   }

   //  Compute vals ans jacobian at current inverse
   tCsteResVecJac  aVJ = mDIM.Jacobian(aVCurInv);


   tU_INT4 aNewInd=0;
   for (tU_INT4 aKInd=0 ; aKInd<mVSubSet.size();  aKInd++)
   {
       tSPIDIM & aSInv = InvOfKSubS(aKInd);
       aSInv.mBestVal =  aVJ.first->at(aKInd);
       aSInv.mBestErr = aSInv.Score(aSInv.mBestVal);
       aSInv.mThreshNextEr =  aSInv.mBestErr / 2.0;
       if (Converged(aSInv))
       {
            // Nothing to do , mBestInv is a good inverse
       }
       else
       {
            tPt  aEr = aSInv.mPTarget-aSInv.mBestVal;
            // Use Jacobian to compute the correction giving the error
            tPt aCor =  SolveCol(aVJ.second->at(aKInd),aEr);

            // aCor =  SolveLine(aEr,aVJ.second->at(aKInd)); it was to check that it does not work

            tPt aNewInv = aSInv.mBestInv+ aCor; // Restimate inverse with correction
            aVCurInv.at(aNewInd) = aNewInv; // Put new inverse in vect to evaluate
            mVSubSet.at(aNewInd) = aSInv.mNum; //this one is in the new subset
            aNewInd++;
       }
   }
   // We have filled only partially the new indexes
   mVSubSet.resize(aNewInd);
   aVCurInv.resize(aNewInd);

   const  tVecPt &  aVO =  mDIM.Values(aVCurInv);

   aNewInd=0;
   mVSubDicot.clear();
   for (tU_INT4 aKInd=0 ; aKInd<mVSubSet.size();  aKInd++)
   {
       tSPIDIM & aSInv = InvOfKSubS(aKInd);
       tPt aNewInv =  aVCurInv.at(aKInd);
       tPt aNewVal =  aVO.at(aKInd);
       Type aNewEr = aSInv.Score(aNewVal);

       aSInv.UpdateNew(aNewEr,aNewInv,aNewVal);

       // signifiant improvement, no dicothomy required
       if (aSInv.ImprovedEnouh())
       {
           // We have made a significtive improvement memorize this new sol
           if (Converged(aSInv))
           {
               // we are at convergence ok
           }
           else
           {
                // still to improve , however
                mVSubSet.at(aNewInd++) = aSInv.mNum; 
           }
       }
       else // Possible instability, will use dicotomy
       {
           aSInv.mInvDic0 = aSInv.mBestInv;
           aSInv.mValDic0 = aSInv.mBestVal;
           aSInv.mInvDic1 = aNewInv;
           aSInv.mValDic1 = aNewVal;

           mVSubDicot.push_back( aSInv.mNum); // This one is in the hard subset
           mVSubSet.at(aNewInd++) = aSInv.mNum; // it will be also in the new subset new subset
       }
   }
   mVSubSet.resize(aNewInd);

   for (int aKIter=0 ; (aKIter<10) && (!mVSubDicot.empty()) ; aKIter++)
   {
       OneIterDicot();
   }
}

template <class Type,const int Dim>
   void cInvertDIMByIter<Type,Dim>::PushResult(tVecPt & aVecPt) const
{
    for (const auto & aSInv : mVInv)
        aVecPt.push_back(aSInv.mBestInv);
}

template <class Type,const int Dim> 
     cInvertDIMByIter<Type,Dim>::cInvertDIMByIter(const tDIM & aDIM):
       mDIM (aDIM)
{
}


template <class Type,const int Dim> 
   void  cInvertDIMByIter<Type,Dim>::Init(const tVecPt & aVTarget) 
{
    mVSubSet.clear();
    mVInv.clear();
    const tVecPt & aVInit = mDIM.RoughInv()->Values(aVTarget);
    for (tU_INT4 aKPt=0; aKPt<aVInit.size() ; aKPt++)
    {
       tSPIDIM aStr;
       aStr.mNum = aKPt;
       aStr.mPTarget = aVTarget[aKPt];
       aStr.mBestInv = aVInit[aKPt];
       // aStr.mBestErr = 1e
       mVSubSet.push_back(aKPt);
       mVInv.push_back(aStr);
    }

}


/* ============================================= */
/*      cDataIterInvertMapping<Type>             */
/* ============================================= */

template <class Type,const int Dim>
   cDataIterInvertMapping<Type,Dim>::cDataIterInvertMapping
   (const tPt& aEps,tMap aRoughInv,const Type& aDistTol,int aNbIterMaxInv) :
       cDataInvertibleMapping<Type,Dim> (aEps),
       mStrInvertIter                   (nullptr),
       mRoughInv                        (aRoughInv),
       mDTolInv                         (aDistTol),
       mNbIterMaxInv                    (aNbIterMaxInv)
{
}

template <class Type,const int Dim>
   cDataIterInvertMapping<Type,Dim>::cDataIterInvertMapping(tMap aRoughInv,const Type& aDistTol,int aNbIterMaxInv) :
      cDataIterInvertMapping<Type,Dim>(tPt::PCste(0.0),aRoughInv,aDistTol,aNbIterMaxInv)
{
}

template <class Type,const int Dim>  
      typename cDataIterInvertMapping<Type,Dim>::tHelperInvertIter *  
               cDataIterInvertMapping<Type,Dim>::StrInvertIter() const
{
   if (mStrInvertIter.get()==nullptr)
   {
       // mStrInvertIter = std::shared_ptr<tHelperInvertIter>(new  tHelperInvertIter(*this));
       mStrInvertIter.reset(new  tHelperInvertIter(*this));
   }
   return mStrInvertIter.get();
}

template <class Type,const int Dim>
    const typename cDataIterInvertMapping<Type,Dim>::tDataMap * 
   // std::unique_ptr<const typename cDataIterInvertMapping<Type,Dim>::tDataMap> 
                  cDataIterInvertMapping<Type,Dim>::RoughInv() const
{
       return   mRoughInv.DM();
}

template <class Type,const int Dim>
   const Type & cDataIterInvertMapping<Type,Dim>::DTolInv() const
{
   return mDTolInv;
}


template <class Type,const int Dim>
    const typename cDataIterInvertMapping<Type,Dim>::tVecPt & 
                   cDataIterInvertMapping<Type,Dim>::Inverses(tVecPt& aVRes,const tVecPt & aVTarget) const
{
    
    // Default method, use iterative approach
    MMVII_INTERNAL_ASSERT_strong(RoughInv() != nullptr,"No rough inverse");

    tHelperInvertIter * aSInvIter = StrInvertIter();
    aSInvIter->Init(aVTarget);

    for (int aKIter=0 ; (aKIter<mNbIterMaxInv) && (aSInvIter->Remaining()!=0); aKIter++)
    {
       aSInvIter->OneIterInversion();
    }

    // tVecPt & aVRes = this->BufOutCleared();
    aVRes.clear();
    aSInvIter->PushResult(aVRes);

    return aVRes;
}

/* ============================================= */
/*          cDataIIMFromMap<Type>                */
/* ============================================= */

template <class Type,const int Dim> 
      cDataIIMFromMap<Type,Dim>::cDataIIMFromMap
           (tMap aMap,const tPt & aEps,tMap aRoughInv,const Type& aDistTol,int aNbIterMax) :
              tDataIIMap   (aEps,aRoughInv,aDistTol,aNbIterMax),
              mMap(aMap)
{
}

template <class Type,const int Dim> 
      cDataIIMFromMap<Type,Dim>::cDataIIMFromMap
           (tMap aMap,tMap aRoughInv,const Type& aDistTol,int aNbIterMax) :
              tDataIIMap   (aRoughInv,aDistTol,aNbIterMax),
              mMap         (aMap)
{
}





template <class Type,const int Dim> 
      const  typename cDataIIMFromMap<Type,Dim>::tVecPt &  
            cDataIIMFromMap<Type,Dim>::Values(tVecPt & aVecOut,const tVecPt & aVecIn) const
{
    return mMap.DM()->Values(aVecOut,aVecIn);
}

template <class Type,const int Dim> 
      typename cDataIIMFromMap<Type,Dim>::tCsteResVecJac 
            cDataIIMFromMap<Type,Dim>::Jacobian(tResVecJac aResJac,const tVecPt & aVecIn) const
{
    return mMap.DM()->Jacobian(aResJac,aVecIn);
}


/* ============================================= */
/*          Compute MapInverse                   */
/* ============================================= */

template <class Type,const int Dim> class  cComputeMapInverse
{
    public :
        typedef cLeastSqComputeMaps<Type,Dim,Dim> tLSQ;
        typedef cDataBoundedSet<Type,Dim>         tSet;
        typedef cPtxd<Type,Dim>                   tPtR;
        typedef cPtxd<int,Dim>                    tPtI;
       

        cComputeMapInverse();
        void  DoIt();
    private :
        tPtI ToPix(const tPtR&) const;
        tPtR FromPix(const tPtI&) const;

        tLSQ &        mLSQ;
        tSet &        mSet;
        
        cPixBox<Dim>  mBoxPix;
};


// template <class Type,const int Dim> void  cComputeMapInverse


/* ============================================= */
/*          INSTANTIATION                        */
/* ============================================= */

#define INSTANCE_INVERT_MAPPING(DIM)\
template class cComputeMapInverse<double,DIM>;\
template class cDataIIMFromMap<double,DIM>;\
template class cDataInvertibleMapping<double,DIM>;\
template class cDataIterInvertMapping<double,DIM>;\
template class cInvertDIMByIter<double,DIM>;

INSTANCE_INVERT_MAPPING(2)
INSTANCE_INVERT_MAPPING(3)




/* ============================================= */
/* ============================================= */
/* ====                                      === */ 
/* ====            CHECK/BENCH               === */ 
/* ====                                      === */ 
/* ============================================= */
/* ============================================= */


class cTestMapInv : public cDataIterInvertMapping<tREAL8,3>
{
    public :
/*  Initialisation a bit tricky, because class is its own rough invers and we must
   must avoid infinite recursion,  TO CHANGE LATER with a two step init ...
*/
       cTestMapInv(double aFx,double aFy,double aFz,double aFreqCos,double aMulCos,bool ForRoughInv=false) :
          cDataIterInvertMapping<tREAL8,3> 
          (
              cPt3dr::PCste(1e-3/std::max(1e-5,mFreqCos)),
              cMapping<tREAL8,3,3>(ForRoughInv?nullptr:new cTestMapInv(1.0/aFy,1.0/aFx,1.0/aFz,1.0,0.0,true)),
              1e-4,
              20
          ),
          mFx      (aFx),
          mFy      (aFy),
          mFz      (aFz),
          mFreqCos (aFreqCos),
          mMulCos  (aMulCos)
       {
       }

       cPt3dr Value(const cPt3dr & aP) const override 
       {
           return   cPt3dr
                    (
                        mFx * aP.y() + cos((aP.x()+aP.y())*mFreqCos)*mMulCos,
                        mFy * aP.x() + sin((aP.y()-aP.z())*mFreqCos)*mMulCos,
                        mFz * aP.z() + sin((1.0+aP.x()-aP.y()+aP.z())*mFreqCos)*mMulCos
                    );
       }
       double mFx;
       double mFy;
       double mFz;
       double mFreqCos;
       double mMulCos;
};


void BenchInvertMapping(cParamExeBench & aParam)
{
    // Check in pure linear case, the inverse is exact
    {
       cTestMapInv  aM1(0.3,4.1,2.2,1000.0,0.0);
       const cDataMapping<tREAL8,3,3> & aM2 = *(aM1.RoughInv());
       for (int aK=0; aK<1000 ; aK++)
       {
           cPt3dr aP1 = cPt3dr::PRandC()*100.0;
           cPt3dr aP12 = aM1.Value(aM2.Value(aP1));
           cPt3dr aP21 = aM2.Value(aM1.Value(aP1));
           MMVII_INTERNAL_ASSERT_bench(Norm2(aP1-aP12)<1e-5,"cTestMapInv rough inverse");
           MMVII_INTERNAL_ASSERT_bench(Norm2(aP1-aP21)<1e-5,"cTestMapInv rough inverse");
       }
    }

    for (int aKMap=0 ; aKMap<100 ; aKMap++)
    {
        double aFX = RandUnif_C_NotNull(1e-1) * 3.0;
        double aFY = RandUnif_C_NotNull(1e-1) * 3.0;
        double aFZ = RandUnif_C_NotNull(1e-1) * 3.0;
        double aFreq = RandUnif_C_NotNull((1e-1) * 3.0);
        double aFMin =std::min(std::abs(aFX),std::min(std::abs(aFY),std::abs(aFZ)));
        double aMulCos =  (aFMin / aFreq) * 0.2 * RandUnif_0_1();

       cTestMapInv  aM1(aFX,aFY,aFZ,aFreq,aMulCos);
/*
       cMapping     aMInv(new cTestMapInv(aM1.RoughInverse()));
       double aEpsInv = 1e-4;
       aM1.SetRoughInv(aMInv,aEpsInv,20);
*/
       cDataInvertibleMapping<tREAL8,3> * aPM1 = & aM1;


       tREAL8  aEpsInv = aM1.DTolInv();
       for (int aKP=0 ; aKP<100 ; aKP++)
       {
           cPt3dr aPt = cPt3dr::PRandC()*100.0;
           cPt3dr aPtD = aM1.Value(aPt);
           cPt3dr aPtDI = aM1.Inverse(aPtD);

           cPt3dr aPtI = aM1.Inverse(aPt);
           cPt3dr aPtID = aM1.Value(aPtI);

           MMVII_INTERNAL_ASSERT_bench(Norm2(aPt -aPtDI)<10*aEpsInv,"elem inverse");
           MMVII_INTERNAL_ASSERT_bench(Norm2(aPt -aPtID)<10*aEpsInv,"elem inverse");
       }

       for (int aKV=0 ; aKV<10 ; aKV++)
       {
           int aNb = RandUnif_N(100);
           std::vector<cPt3dr> aVIn;
           for (int aKP=0 ; aKP<aNb ; aKP++)
           {
              aVIn.push_back(cPt3dr::PRandC()*100.0);
           }
           std::vector<cPt3dr> aVOut =  aPM1->Values(aVIn);
           std::vector<cPt3dr> aVInv =  aPM1->Inverses(aVOut);
           for (int aKP=0 ; aKP<aNb ; aKP++)
           {
              double aD = Norm2(aVIn[aKP] - aVInv[aKP]);
              MMVII_INTERNAL_ASSERT_bench(aD<10*aEpsInv,"elem inverse");
           }
       }
    }
}



};
