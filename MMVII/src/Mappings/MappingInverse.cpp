#include "include/MMVII_all.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

// extern bool BUGINVMAP;

/* ============================================= */
/*        cDataInvertibleMapping<Type,Dim>       */
/* ============================================= */

// const  tVecPt &  Inverses(tVecPt &,const tVecPt & ) const;

template <class Type,const int Dim>
    cDataInvertibleMapping<Type,Dim>::cDataInvertibleMapping(const tPt& aEps) :
        cDataNxNMapping<Type,Dim> (aEps)
{
}

template <class Type,const int Dim>
    cDataInvertibleMapping<Type,Dim>::cDataInvertibleMapping() :
        cDataNxNMapping<Type,Dim> (tPt::PCste(0.0))
{
}

/**  buffered version, call unbuferred version */

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

/** buffer version, without furnish the buf, call previous one with standard buff */
template <class Type,const int Dim>
    const typename cDataInvertibleMapping<Type,Dim>::tVecPt &
                   cDataInvertibleMapping<Type,Dim>::Inverses(const tVecPt & aVImages) const
{
   return Inverses(BufInvOutCleared(),aVImages);
}


/**  unbuffered version, call buferred version with a buffer of size 1 */
template <class Type,const int Dim>
    typename cDataInvertibleMapping<Type,Dim>::tPt  
          cDataInvertibleMapping<Type,Dim>::Inverse(const tPt & aPImage) const
{
   this->BufIn1Val()[0] = aPImage;
   const tVecPt & aRes = Inverses(this->BufIn1Val());
   return aRes[0];
}

/* ============================================= */
/*        cDataInvertOfMapping<Type,Dim>         */
/* ============================================= */
template <class Type,const int Dim>
    cDataInvertOfMapping<Type,Dim>::cDataInvertOfMapping(const tIMap * aMapToInv,bool toAdopt) :
         mMapToInv(aMapToInv),
         mAdopted (toAdopt)
{
}

template <class Type,const int Dim>
    cDataInvertOfMapping<Type,Dim>::~cDataInvertOfMapping()
{
    if (mAdopted)
       delete mMapToInv;
}

template <class Type,const int Dim>
         const  std::vector<cPtxd<Type,Dim>>  &
                 cDataInvertOfMapping<Type,Dim>::Inverses(tVecPt & aVOut,const tVecPt & aVIn) const
{
    return  mMapToInv->Values(aVOut,aVIn);
}

template <class Type,const int Dim>
         const  std::vector<cPtxd<Type,Dim>>  &
                 cDataInvertOfMapping<Type,Dim>::Values(tVecPt & aVOut,const tVecPt & aVIn) const
{
    return  mMapToInv->Inverses(aVOut,aVIn);
}


/* ============================================= */
/*            cInvertByIter<Type,Dim>            */
/*            - cStrPtInvDIM<Type,Dim>           */
/* ============================================= */

/*  
   The method for inversion may seems quite complex, it aims to avoid one trap of MMV1, with the too
  current message "too many iteration in inversion by finite diferenece unstable ..;"

   (1)  void cInvertDIMByIter<Type,Dim>::OneIterInversion()

   (2) void cInvertDIMByIter<Type,Dim>::OneIterDicot()

    Method for inversion  , at each iteration (1) ,  make one refinement of computing
    the value and the gradient and solve the linear equation,  three possibility :

          * we have converged  -> bingo, just memorize  the solution for final export

          * we have made an improvment but not converger, ->  memorize the next iteration

          * we have make no significant improvement, or worst we have  increase the resisudual
            in this case we will make a dicotomic (2) resarch, on the line of gradient, but without
            differenciation
    

    Note :  at different point we have subset that increase then decrase, we use the same pattern :
         * create an index VIND of object,
         * when we are in creation phase : init by clear and pushback
         * when we are in reduction phase :
               * create an ind of end, INDEND =0
               * insert by VIND.at(INDEN)++ = Something (no overflow because its a subset)
               * at end VIND.resize(INDEND)
    Used for  mVSubSet and  mVSubDicot
*/

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

        tPt  mInvDic0;
        tPt  mValDic0;
        tPt  mInvDic1;
        tPt  mValDic1;
        tPt  mPTarget;
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

      void SetRatioGainDicot(const Type & aRGD) {mRatioGainDicot = aRGD;}
    private :
      /// Make one iteration by dicotomy, for point that could not improved by gradient
      void OneIterDicot();
      bool Converged(const tSPIDIM & aSInv) const {return aSInv.mBestErr<mDIM.mDTolInv;}

      const tDIM &           mDIM;
      Type                   mRatioGainDicot;  ///< Ratio for gain of error that will run dicothomy
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
   tU_INT4  aIndSubDicot=0; // Use to follow the index remaining in Dicot
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
          int aKMin = aWM.IndexExtre();
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
       aSInv.mThreshNextEr =  aSInv.mBestErr / mRatioGainDicot ;
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
       //StdOut() << "DICOT  " <<    mVSubDicot.size() << "\n"; //getchar();
       OneIterDicot();
   }
   // StdOut() << "cInvertDIMByIter " <<   aNewInd << " " << mVSubDicot.size() << "\n\n"; //getchar();
}

template <class Type,const int Dim>
   void cInvertDIMByIter<Type,Dim>::PushResult(tVecPt & aVecPt) const
{
    for (const auto & aSInv : mVInv)
        aVecPt.push_back(aSInv.mBestInv);
}

template <class Type,const int Dim> 
     cInvertDIMByIter<Type,Dim>::cInvertDIMByIter(const tDIM & aDIM):
       mDIM             (aDIM),
       mRatioGainDicot  (2.0)
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


enum class eLabelIm : tU_INT1
{
   eFree,      // Mode MicMac V1
   eReached,  // Mode PSMNet
   eInvalid,  // Mode PSMNet
   eBorder,    // Mode PSMNet
   eNbVals
};

/**  Helper for extending map invere near frontier */
template <class Type,const int Dim>  struct cPtsExtendCMI
{
     public :
         typedef cPtxd<Type,Dim> tPt;

         cPtsExtendCMI(const tPt & aCurP,const tPt & aDir) :
             mCurP (aCurP),
             mDir  (aDir)
         {
         }
        
         tPt  mCurP;
         tPt  mDir;
};


/**   Class for computing an inverse mapping from :
         * the direct mapping to invert  EIn => EOut
         * a set of base function that linerly code the invert
         * a validity domain on the output space EOut
         * a "seed" point in input space

      The criterion for validity if   || J(Seed)^-1 * J(p) -Id ||  <  Threshold   J=Jacobian

       This is adapted to distorsion where :
          * we know the output space -> sensor space + an optional validty (masq image, circle ...)
          * we jo

       The method make grow a space where the mapping can reasonnabily be expect to be invertible,
  the critrion being for this is to ensure that the jacobian is always sufficiently close to the jacobian 
  at the seed (pushed to the limit, when equals it means that function is linear).

       The growing is made on a grid by a connected component analysis starting from the seed.  

       At the end, due to the sampling we may have few or no point close to the border/frontier. This
    is no good as we know that extrapolation do not work well, so we have a step  were we add
    a prolongation to go nearer to the frontier
*/


template <class Type,const int Dim> class  cComputeMapInverse
{
    public :
	    //  aCMaxRel => define the zone relatively to the rho max
        friend void OneBench_CMI(double aCMaxRel);
        // using enum eLabelIm;
        typedef cLeastSqComputeMaps<Type,Dim,Dim> tLSQ;
        typedef cDataBoundedSet<Type,Dim>         tSet;
        typedef cDataNxNMapping<Type,Dim>         tMap;
        typedef cPtxd<Type,Dim>                   tPtR;
        typedef cPtxd<int,Dim>                    tPtI;
        typedef cTplBox<Type,Dim>                 tBoxR;
        typedef cPtsExtendCMI<Type,Dim>           tExtent;
        typedef typename tMap::tCsteResVecJac     tCsteResVecJac;
       

        /// Constructor, essentially memorize parameters
        cComputeMapInverse
        (
             const Type & aThreshJac, ///< Threshold on jacobian to ensure inversability
             const tPtR& aSeed,       ///< Seed point, in input space
             const int & aNbPtsIn,    ///< Approximate number of point (in the biggest size)
             tSet &,  ///< Set of validity, in output space
             tMap&,   ///< Maping to invert : InputSpace -> OutputSpace
             tLSQ&,  ///< Structure for computing the invert on base of function using least square   
             bool Test=false
        );
        void  DoAll(std::vector<Type> & aVSol);

        static int constexpr  TheNbIterByStep = 3;
        static Type constexpr TheStepFrontLim = 3e-2;
    private :
        cComputeMapInverse(const cComputeMapInverse<Type,Dim> &) = delete;
        /** Compute an approximation of Input box as reciproque of output box, use jacobian as
            we dont know inverse (else we would not be here ...) */
        tBoxR  BoxInByJacobian() const;
        /** From input real space to grid space */
        tPtI ToPix(const tPtR& aPR) const
        {
              return  Pt_round_ni<Type>((aPR-mBoxMaj.P0())/mStep);
        }
        /** From grid space to input real space*/
        tPtR FromPix(const tPtI& aPI) const
        {
              return  mBoxMaj.P0() + tPtR::FromPtInt(aPI)*mStep;
        }
        /// Is the jacobian sufficently close to its value on seed ?
        bool ValideJac(const cDenseMatrix<Type> & aMat) const;

        /// Add a Pixel in the queue if has not already be visited
        void Add1PixelTopo(const tPtI & aPix);
        /// Filters pixel geometrically OK (Jac+domain) and add them as obs for least square
        void FilterAndAddPixelsGeom();

        /// Make on iteration, at given step, to have point closer to the frontier
        void OneStepFront(const Type & aStepFront);

        /// Validate (POut/Jac) if in domain and jacobian is OK
        bool ValidateK(const tCsteResVecJac & aVecPJ,int aKp)
        {
            return mSet.InsideWithBox((*aVecPJ.first)[aKp]) && ValideJac((*aVecPJ.second)[aKp]);
        }
        /// Add one observtion for computing inverse, IsFront used for memo in test mode
        void AddObsMapDirect(const tPtR & aPIn,const tPtR & aPOut,bool IsFront);

         // Copy of parameters
        Type          mThresholdJac; 
        tPtR          mPSeed; //  seed point that is waranteed to be inside the domain
        tSet &        mSet;   // Definition set of Output space
        tMap &        mMap;   // Map to invert
        tLSQ &        mLSQ;   // systeme to compute the inverse as a linear composition of given base functions (using least square)
          // Created members 
        tBoxR         mBoxByJac; ///< Box computed assuming that Map is equal to its jacobian in PSeed
        tBoxR         mBoxMaj;  ///< Majoration of box, taking into account possible  unstability and jacobian threshold
        Type          mStep;    ///< Step on the grid
        cPixBox<Dim>              mBoxPix;  ///< Pixel box to make image processing stuff
        cDataTypedIm<tU_INT1,Dim> mMarker;  ///< Marker image to make growing
        std::vector<tPtI>         mNextGen; ///< Next generation of pixel in growing region
        cDenseMatrix<Type>        mJacInv0; ///< Matrix invert of Jacobian in PSeed
        cDenseMatrix<Type>        mMatId;   ///< Id Matrix, helper for computing Jacobian criteria
        const std::vector<tPtI> &       mNeigh; ///< Neighbourhood for image-morpho-operation
        std::vector<tExtent>      mVExt; ///< Vector of "extension" to the frontier
        bool                      mTest; ///< Are we in test mode ?
    public :
        Type                      mStepFrontLim; // TheStepFrontLim
        std::vector<tPtR>         mVPtsInt; ///< For test, memo point interior
        std::vector<tPtR>         mVPtsFr;  ///< For test, memo point frontier

};


template <class Type,const int Dim> 
    cTplBox<Type,Dim>   cComputeMapInverse<Type,Dim>::BoxInByJacobian() const
{
    cBijAffMapElem<Type,Dim>  aDif = mMap.Linearize (mPSeed); // Compute linear application at PSeed

    cInvertMappingFromElem<cBijAffMapElem<Type,Dim> > aMap(aDif.MapInverse()); // Compute inverse mapping

    cTplBox<Type,Dim> aRes=  aMap.BoxOfCorners(mSet.Box());  // compute recripoque image of box out
    return aRes;

}

template <class Type,const int Dim> 
   void  cComputeMapInverse<Type,Dim>::AddObsMapDirect(const tPtR & aPIn,const tPtR & aPOut,bool IsFront)
{
     /*  Carrefull we add aPIn-aPOut : 
            * in distortion (see  PProjToImNorm in cMMVIIUnivDist) the code part is the additional part,
	    * i.e the distortion identity is the code by {0,0,...} params
     */
     mLSQ.AddObs(aPOut,aPIn-aPOut); // put is as as sample  Out => Map-Identity  = PIn-POut
     if (mTest)
     {
        if (IsFront)
           mVPtsFr.push_back(aPOut);
        else
           mVPtsInt.push_back(aPOut);
     }
}


template <class Type,const int Dim> 
        void  cComputeMapInverse<Type,Dim>::OneStepFront(const Type & aStepFront)
{
    std::vector<int> aVSel;  // Selection of indexes still in used
    for (int aK=0 ; aK<int(mVExt.size()) ; aK++)
        aVSel.push_back(aK);  // Initially put all indexes

    // Fix a limit number of step, and stop when empty
    for (int aKStep=0 ; (aKStep<TheNbIterByStep) && (!aVSel.empty()) ; aKStep++)
    {
        // for points still OK, go a step further to the frontier
        std::vector<tPtR> aVPt;
        for (int aKSel=0 ; aKSel<int(aVSel.size()) ; aKSel++)
        {
           const tExtent & anExt = mVExt[aVSel[aKSel]];
           aVPt.push_back(anExt.mCurP + anExt.mDir*aStepFront);
        }
	// compute their coordinates and jacobians
        tCsteResVecJac  aVecPJ = mMap.Jacobian(aVPt);

	// select those who are valid
        std::vector<int> aNextVSel; // prepare for next iter
        for (int aKSel=0 ; aKSel<int(aVSel.size()) ; aKSel++)
        {
            if (ValidateK(aVecPJ,aKSel))  // inside and jacobian still ok
            {
                int aIndGlob = aVSel[aKSel];   // Ind in full point of frontier
                tExtent & anExt = mVExt[aIndGlob];
                anExt.mCurP = aVPt[aKSel];
                aNextVSel.push_back(aIndGlob);
            }
        }
        // aNextVSel = aVSel;
        aVSel = aNextVSel;
    }
}

template <class Type,const int Dim> 
   cComputeMapInverse<Type,Dim>::cComputeMapInverse
   (
        const Type& aThresholdJac,
        const tPtR& aPSeed,
        const int & aNbPts,
        tSet &      aSet,
        tMap&       aMap,
        tLSQ&       aLSQ,
        bool        aTest
    ) :
       mThresholdJac  (aThresholdJac),
       mPSeed         (aPSeed),
       mSet           (aSet),
       mMap           (aMap),
       mLSQ           (aLSQ),
       mBoxByJac      (BoxInByJacobian()),
       mBoxMaj        (mBoxByJac.ScaleCentered(1/(1-mThresholdJac))),
       mStep          (NormInf(mBoxByJac.Sz()) / aNbPts),
       mBoxPix        (ToPix(mBoxMaj.P0()), ToPix(mBoxMaj.P1())),
       mMarker        (mBoxPix.P0(),mBoxPix.P1()),
       mJacInv0       (1,1),
       mMatId         (Dim,Dim,eModeInitImage::eMIA_MatrixId),
       mNeigh         (AllocNeighbourhood<Dim>(1)),
       mTest          (aTest),
       mStepFrontLim  (TheStepFrontLim)
{
}

template <class Type,const int Dim> void  cComputeMapInverse<Type,Dim>::Add1PixelTopo(const tPtI& aPix) 
{
   if (mMarker.VI_GetV(aPix)!= tU_INT1(eLabelIm::eFree))  // Test not already visited
      return;

   mMarker.VI_SetV(aPix,tU_INT1(eLabelIm::eReached));  // Set visited
   mNextGen.push_back(aPix); // put it in next generation
}

template <class Type,const int Dim>   
   bool cComputeMapInverse<Type,Dim>::ValideJac(const cDenseMatrix<Type> & aMat) const
{
    cDenseMatrix<Type>  aJacRel = mJacInv0 * aMat;  // JR = J0-1 * Jac  => NOT SURE BEST SENSE TO MAKE THE MULT ?
    Type  L2Dist = aJacRel.DIm().L2Dist(mMatId.DIm());  // Is it close to identity
    return (L2Dist< mThresholdJac);
}


template <class Type,const int Dim> void  cComputeMapInverse<Type,Dim>::FilterAndAddPixelsGeom()
{
    // Make grid pixel, pixel of input space
    std::vector<tPtR>   aNextGenReal;
    for (auto aPix : mNextGen)
        aNextGenReal.push_back(FromPix(aPix));

    // Compute values and jacobian for these pixel using buffered mode
    // typename tMap::tCsteResVecJac  aVecPJ = mMap.Jacobian(aNextGenReal);
    tCsteResVecJac  aVecPJ = mMap.Jacobian(aNextGenReal);
    std::vector<tPtI>   aNexGenFiltered;  // Will contain geometrically filtered
    
    for (size_t aKp=0 ; aKp<mNextGen.size() ; aKp++)
    {
        if (ValidateK(aVecPJ,aKp))
        {
             aNexGenFiltered.push_back(mNextGen[aKp]);  // select it for next
             AddObsMapDirect(aNextGenReal[aKp],(*aVecPJ.first)[aKp],false); 
        }
        else
        {
            mMarker.VI_SetV(mNextGen[aKp],tU_INT1(eLabelIm::eInvalid));  // Mark it as invalid
        }
    }

    mNextGen = aNexGenFiltered;
}

        // void OneStepFront(const Type & Front);
template <class Type,const int Dim> void  
     cComputeMapInverse<Type,Dim>::DoAll(std::vector<Type> & aVSol)
{
     // Initialize label : Interior and border
     mMarker.InitInteriorAndBorder(Type(eLabelIm::eFree),Type(eLabelIm::eBorder));

     tPtI aPixSeed = ToPix(mPSeed);
     MMVII_INTERNAL_ASSERT_tiny( mMarker.Inside(aPixSeed),"Seed outside in Map Inverse");
     MMVII_INTERNAL_ASSERT_tiny( mMarker.VI_GetV(aPixSeed)==int(eLabelIm::eFree),"Seed bored Map Inverse");

     typename tMap::tResJac  aPJ = mMap.Jacobian(mPSeed);
     mJacInv0 = aPJ.second.Inverse();

     // 0-  Init with the seed
     Add1PixelTopo(aPixSeed);   // init the  heap struct (nexgtgen , marker etc ... with the seed
     FilterAndAddPixelsGeom();  // create the geometry of the seed

     MMVII_INTERNAL_ASSERT_tiny( mNextGen.size()==1,"Seed Geom pb");  // if filtering removed the seed, we are bad ...


     // 1-  now recursively add valide point/pixel connected to  new one and still not explorer
     while (! mNextGen.empty())
     {
        std::vector<tPtI>     aCurGen  = mNextGen; // memorize current
        mNextGen.clear(); // clear fornext gen
        for (auto const & aPix : aCurGen)  // par cur gen
        {
            for (auto const & aN : mNeigh) // parse neighbourhood
            {
                Add1PixelTopo(aPix+aN);  // tentative add  (if not already visited)
            }
        }
        FilterAndAddPixelsGeom(); // select those who are OK
     }

     // 2- Make the extension to have point close to the frontier 

         // 2-1 Compute in grid pixel frontier :  reached pixel neighbor of unreached
	 // at this step put in structure to have the benefit of paralleization
     for (const auto aPix : mMarker)  // parse all pixel of image
     {
         if (mMarker.VI_GetV(aPix)== tU_INT1(eLabelIm::eReached))
         {
            // compute it is a frontier pixel (one neighbour not reached)
            bool isFront = false;
            for (auto const & aN : mNeigh)
            {
                 if (mMarker.VI_GetV(aPix+aN) != tU_INT1(eLabelIm::eReached))
                 {
                     isFront = true;
                 }
            }
            // Now its frontier, make it a real point
            if (isFront)
            {
                tPtR aPR = FromPix(aPix);
                tPtR aDir = VUnit(aPR-mPSeed) * mStep;  // Direction * by step to be ~ to a pixel lenght
                mVExt.push_back(tExtent(aPR,aDir));
            }
         }
     }


         // 2-2 Make extension at degrowing step
	 // for each step, we will be able to parallelize on all points of the frontier
     for (double aStepFront=1.0 ; aStepFront>mStepFrontLim; aStepFront /= 2.0)
     {
         OneStepFront(aStepFront);
     }

         // 2-3 Compute valuses and add obs
     std::vector<tPtR>  aVFrontIn;
     for (int aKp=0 ; aKp<int(mVExt.size()) ; aKp++)
          aVFrontIn.push_back(mVExt[aKp].mCurP);
     std::vector<tPtR>  aVFrontOut = mMap.Values(aVFrontIn);

     for (int aKp=0 ; aKp<int(mVExt.size()) ; aKp++)
     {
         AddObsMapDirect(aVFrontIn[aKp],aVFrontOut[aKp],true); // put is as as sample  Out => In 
     }
     
      mLSQ.ComputeSol(aVSol);
}


void  OneBench_CMI(double aCMaxRel)
{
    cPt3di aDegMapDir(2,0,0);
    cPt3di aDegMapInv(5,1,1);

    double aRho = 0.1 + 2 * RandUnif_0_1();
    double aCMax = aRho * (aCMaxRel);
    cPt2dr aP0 = cPt2dr::PCste(-aCMax);
    cPt2dr aP1 = cPt2dr::PCste(aCMax);
    cBox2dr aBox(aP0,aP1); 

    cSphereBoundedSet  aSBS(aBox,cPt2dr(0,0),aRho);

    cRandInvertibleDist aRIDBasique(aDegMapDir,std::min(aRho,aCMax*sqrt(2.0)),1.0,0.2);
    cDataNxNMapCalcSymbDer<double,2> *  aTargetFunc = aRIDBasique.MapDerSymb();

    cCalculator<double> * anEqBase = EqBaseFuncDist(aDegMapInv,10+ RandUnif_0_1()*25);  // Calculator for base of func
    cLeastSqCompMapCalcSymb<double,2,2> aLsqSymb(anEqBase);

    cComputeMapInverse<double,2> aCMI
                                 (
                                    0.5,
                                    cPt2dr(0.0,0.0),
                                    20,
                                    aSBS,
                                    *aTargetFunc,
                                    aLsqSymb,
                                    true
                                  );

    aCMI.mStepFrontLim = 1e-3;
    std::vector<double> aVParam;
    aCMI.DoAll(aVParam);
    cDataNxNMapCalcSymbDer<double,2> *  aMapInv = NewMapOfDist(aDegMapInv,aVParam,100);

    bool CaseDiskInclude = (aCMaxRel>1.0);
    bool CaseDiskExclude = (aCMaxRel<sqrt(0.5));
    // Disk is include in Box, all point front should be on the disk
    double aPrecFr = aCMI.mStepFrontLim*aCMI.mStep;
    // Case disk include in 
    if (CaseDiskInclude || CaseDiskExclude)
    {
       // Check that all point of the frontier are almost on the circle/square
       for (const auto & aP : aCMI.mVPtsFr)
       {
          double aPrec =  CaseDiskInclude ? std::abs(Norm2(aP)  -aRho)  : std::abs(NormInf(aP)  - aCMax);
          if (aPrec>aPrecFr*4)
          {
             StdOut() << "FFRRRRr " << aPrec / aPrecFr << "\n";
             MMVII_INTERNAL_ASSERT_bench(false,"Frontier Precision in MapInverse");
          }
       }
    }


    if (true) // CaseDiskInclude || CaseDiskExclude)
    {
       // Check that the frontier is sufficiently dense, for this check that in any
       // direction we get a point on the frontier at this direction
       double aPrecDenseAtan = aCMI.mStep / aRho;
       for (int aK=0 ; aK< 100 ; aK++)
       {
            cPt2dr aPts = FromPolar(1.0,1000*RandUnif_C()); // Generate dir
            double aAtanMin =1e10;
            for (const auto & aPFr : aCMI.mVPtsFr)
            {
                cPt2dr aRatio = VUnit(aPFr/aPts);
                aAtanMin = std::min(aAtanMin,std::abs(aRatio.y())); // More or less diff in radian
            }
              
            if (aAtanMin > aPrecDenseAtan *1.2 )
            {
                 StdOut() << "ATMIN " << aAtanMin /  aPrecDenseAtan  << "\n";
                 MMVII_INTERNAL_ASSERT_bench(false,"Frontier densite in MapInveres");
            }
       }
    }

    double anEcartMax = 0.0;
    double anEcartMoy = 0.0;
    int aNbTest = 500;
    // Test that Map Map-1 = Identity
    for (int aCptPoint=0 ; aCptPoint<aNbTest ; aCptPoint++)
    {
        cPt2dr aPOut = aSBS.GeneratePointInside(); // P In Out Space      
        cPt2dr aPIn = aMapInv->Value(aPOut);       // Image By invert Maping
        cPt2dr aPOut2 = aTargetFunc->Value(aPIn);  // Bck to Out Space
        double anEcart = Norm2(aPOut-aPOut2);      // should be the same as it's Map (Map-1)
        anEcartMax = std::max(anEcart,anEcartMax);
        anEcartMoy += anEcart;

        double aDInt =  sqrt(aPIn.MinSqN2(aCMI.mVPtsInt)) / aCMI.mStep;
        double aDFr  =  sqrt(aPIn.MinSqN2(aCMI.mVPtsFr )) / aCMI.mStep;
        double aDTestDense = std::min(aDInt,aDFr);
        if (aDTestDense>1.42)
        {
            StdOut() << "aDIntaDIntDENSE " <<  aDTestDense << "\n";
            MMVII_INTERNAL_ASSERT_bench(false,"Mapping inverse : grid not dense");
        }
    }
    anEcartMoy /= aNbTest;
    if ((anEcartMax>2e-5*aRho) || (anEcartMoy>1e-5*aRho))
    {
        StdOut() << "ECMax " << (anEcartMax)/ aRho << " EcMoy" << anEcartMoy / aRho<< "\n";
        MMVII_INTERNAL_ASSERT_bench(false,"Mapping inverse : is not inverse ....");
    }

    delete anEqBase;
    delete aTargetFunc;
    delete aMapInv;
}




/* ============================================= */
/*          INSTANTIATION                        */
/* ============================================= */

#define INSTANCE_INVERT_MAPPING(DIM)\
template  class  cDataInvertOfMapping<tREAL8,DIM>;\
template class cComputeMapInverse<tREAL8,DIM>;\
template class cDataIIMFromMap<tREAL8,DIM>;\
template class cDataInvertibleMapping<tREAL8,DIM>;\
template class cDataIterInvertMapping<tREAL8,DIM>;\
template class cInvertDIMByIter<tREAL8,DIM>;

INSTANCE_INVERT_MAPPING(2)
INSTANCE_INVERT_MAPPING(3)




/* ============================================= */
/* ============================================= */
/* ====                                      === */ 
/* ====            CHECK/BENCH               === */ 
/* ====                                      === */ 
/* ============================================= */
/* ============================================= */

/**  A class for testing inversion, implement a mapping R3 -> R3 with the sum of
  
       * a "dominant" basic linear function 
       * some  cosinus/sinus to make it non  linear but smooth
*/

class cTestMapInv : public cDataIterInvertMapping<tREAL8,3>
{
    public :
/*  Initialisation a bit tricky, because class is its own rough inverse (with different parameters) and we must
   must avoid infinite recursion,  

     Called with   IsRoughInv=false => create the rough inverse  IsRoughInv=true, now do try to call again ..
*/
       cTestMapInv(double aFx,double aFy,double aFz,double aFreqCos,double aMulCos,bool IsRoughInv=false) :
          cDataIterInvertMapping<tREAL8,3> 
          (
              cPt3dr::PCste(1e-3/std::max(1e-5,mFreqCos)),
              cMapping<tREAL8,3,3>(IsRoughInv?nullptr:new cTestMapInv(1.0/aFy,1.0/aFx,1.0/aFz,1.0,0.0,true)),
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
    {
       OneBench_CMI(1.1);
       OneBench_CMI(0.7);
       for (int aK=0 ; aK<10 ; aK++)
           OneBench_CMI((1.0 + 0.5 *RandUnif_C()));
/*
       OneBenchMapInv(1.1);
       OneBenchMapInv(0.7);
       OneBenchMapInv((1.0 + 0.5 *RandUnif_C()));
*/

    }
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
       cDataIterInvertMapping<tREAL8,3> * aPMIter1 = & aM1;

       cDataInvertOfMapping<tREAL8,3> aInvM1 (&aM1,false);



       // StdOut()  << "JJJJ " << aPMIter1->StrInvertIter() << "\n";
       
       std::vector<double> aVRatio{1.0,5.0,25.0,125.0,625.0};
       double aRatio = aVRatio.at(aKMap%5);
       aPMIter1->StrInvertIter()->SetRatioGainDicot(aRatio);

       tREAL8  aEpsInv = aM1.DTolInv();
       for (int aKP=0 ; aKP<100 ; aKP++)
       {
           cPt3dr aPt = cPt3dr::PRandC()*100.0;
           cPt3dr aPtD = aM1.Value(aPt);
           cPt3dr aPtDI = aM1.Inverse(aPtD);

           cPt3dr aPtI = aM1.Inverse(aPt);
           MMVII_INTERNAL_ASSERT_bench(Norm2(aPtI -aInvM1.Value(aPt))<1e-10,"cDataInvertOfMapping::Value");
	   
           cPt3dr aPtID = aM1.Value(aPtI);
           MMVII_INTERNAL_ASSERT_bench(Norm2(aPtID -aInvM1.Inverse(aPtI))<1e-10,"cDataInvertOfMapping::Inverse");

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
