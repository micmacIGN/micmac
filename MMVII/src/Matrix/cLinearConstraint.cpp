#include "LinearConstraint.h"

using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{
static bool DEBUG=false;
/* ************************************************************ */
/*                                                              */
/*                cSetLinearConstraint                          */
/*                                                              */
/* ************************************************************ */

template <class Type> cSetLinearConstraint<Type>::cSetLinearConstraint(int aNbVar) :
   mNbVar (aNbVar),
   mBuf   (aNbVar)
{
}

template <class Type> void cSetLinearConstraint<Type>::Add1Constr(const t1Constr & aConstr,const  tDV * aCurSol)
{
      mVCstrInit.push_back(aConstr.Dup(aCurSol));
}

template <class Type> void cSetLinearConstraint<Type>::Add1Constr(const tSV& aSV,const Type & aCste,const tDV * aCurSol)
{
   cOneLinearConstraint aCstr(aSV,aCste, mVCstrInit.size());
   Add1Constr(aCstr,aCurSol);
}


template <class Type> void cSetLinearConstraint<Type>::Reset()
{
    mVCstrInit.clear();
    mVCstrReduced.clear();
}

template <class Type> void cSetLinearConstraint<Type>::Add1ConstrFrozenVar(int aKVar,const Type & aVal,const  tDV * aCurSol)
{
    cSparseVect<Type> aSV;
    aSV.AddIV(aKVar,1.0);
    Add1Constr(aSV,aVal,aCurSol);
}


template <class Type> void cSetLinearConstraint<Type>::Compile(bool ForBench)
{
    // make a copy of initial cstr : use dup because shared pointer on mLP ....
    mVCstrReduced.clear();
    for (const auto & aCstr : mVCstrInit)
        mVCstrReduced.push_back(aCstr.Dup(nullptr));

    size_t aNbReduced = 0;
    while (aNbReduced != mVCstrInit.size())
    {
          // extract the  "best" Cstr, i.e. with highest value
          cWhichMax<int,Type> aMax(-1,-1.0);
          for (int aKC=0; aKC<int(mVCstrReduced.size()) ; aKC++)
          {
              if (! mVCstrReduced.at(aKC).mReduced) 
                 aMax.Add(aKC,std::abs(mVCstrReduced.at(aKC).LinearMax()->mVal));
          }
          t1Constr& aBest = mVCstrReduced.at(aMax.IndexExtre());

          aBest.InitSubst();
          aBest.mOrder = aNbReduced;
          // substract the selected constraint to all
          for (t1Constr &  aCstr : mVCstrReduced)
          {
               if (! aCstr.mReduced)
               {
                    if (DEBUG)
                        StdOut()  <<  "SIOC, ISUBS " << aBest.mISubst << " N=" << aBest.mNum << " => " << aCstr.mNum << "\n";
                    aBest.SubstituteInOtherConstraint(aCstr,mBuf);
               }
          }

          // Show("CCCReduc:" + ToStr(aNbReduced));
	  if (ForBench) 
	     TestSameSpace();
          if (DEBUG) 
          {
                if (DEBUG) Show("Reduc:" + ToStr(aNbReduced));
          }
          aNbReduced++;
    }
    std::sort
    (
         mVCstrReduced.begin(),
         mVCstrReduced.end(),
         [](const auto & aC1,const auto & aC2){return aC1.mOrder<aC2.mOrder;}
     );
}

template <class Type> void cSetLinearConstraint<Type>::SubstituteInSparseLinearEquation(tSV & aA,Type &  aB) const
{
    for (const auto & aCstr : mVCstrReduced)
       aCstr.SubstituteInSparseLinearEquation(aA,aB,mBuf);
}

template <class Type> void cSetLinearConstraint<Type>::SubstituteInDenseLinearEquation(tDV & aA,Type &  aB) const
{
    for (const auto & aCstr : mVCstrReduced)
       aCstr.SubstituteInDenseLinearEquation(aA,aB);
}

template <class Type> void cSetLinearConstraint<Type>::SubstituteInOutRSNL(tIO_RSNL& aIO) const
{
    for (const auto & aCstr : mVCstrReduced)
       aCstr.SubstituteInOutRSNL(aIO,mBuf);
}



template <class Type> void cSetLinearConstraint<Type>::Show(const std::string & aMsg) const
{
     StdOut()  << "========  SHOWTSELC " << aMsg << " =====================" << std::endl;

     for (const auto & aCstr: mVCstrReduced)
        aCstr.Show();
}
template <class Type>  void cSetLinearConstraint<Type>::TestSameSpace()
{
    std::vector<cDenseVect<Type> > aV0;  //<  Dense representation of initial cstr
    std::vector<cDenseVect<Type> > aVR;  //<  Dense representation of reduced cstr

    // make a dense representation for constraints already reduced
    for (size_t aKC=0 ; aKC<mVCstrReduced.size() ; aKC++)
    {
        if (mVCstrReduced[aKC].mReduced)
        {
           aV0.push_back(cDenseVect<Type>(mVCstrInit[aKC].mLP,mNbVar));
           aVR.push_back(mVCstrReduced[aKC].DenseVectRestored(mNbVar));
        }
    }

    // check the subspace Init&Reduce are almost the same
    Type aD = cDenseVect<Type>::ApproxDistBetweenSubspace(aV0,aVR);
    MMVII_INTERNAL_ASSERT_bench(aD<1e-5,"cSetLinearConstraint<Type>:: TestSameSpace");
}

template <class Type> void cSetLinearConstraint<Type>::AddConstraint2Sys(tLinearSysSR & aSys)
{
    // A priori identic to use init or reduced, simpler with init as there is no reconstruction
    for (const auto & aCstr : mVCstrInit)
    {
        aSys.PublicAddObservation(1.0,aCstr.mLP,aCstr.mCste);
    }
}



/* ************************************************************ */
/*                                                              */
/*                     cDSVect                                  */
/*                                                              */
/* ************************************************************ */

template <class Type> cDSVec<Type>::cDSVec(size_t aNbVar) :
     mVec (aNbVar,eModeInitImage::eMIA_Null),
     mSet (aNbVar)
{
}

template <class Type> void cDSVec<Type>::AddValInd(const Type & aVal,int anInd)
{
     mVec(anInd) += aVal;
     mSet.AddIndFixe(anInd);
}

template <class Type> void cDSVec<Type>::Reset()
{
   for (const auto & anInd : mSet.mVIndOcc)
   {
       mVec(anInd) = 0.0;
       mSet.mOccupied.at(anInd) = false;
   }
   mSet.mVIndOcc.clear();
}


template <class Type> void cDSVec<Type>::TestEmpty()
{
     for (const auto & aV : mVec.ToStdVect()) 
        MMVII_INTERNAL_ASSERT_tiny(aV==0.0,"Vec Test Empty");
     MMVII_INTERNAL_ASSERT_tiny(mSet.mVIndOcc.empty(),"Occ Test Empty");
     for (const auto & aV : mSet.mOccupied)
        MMVII_INTERNAL_ASSERT_tiny(aV==false,"Vec Test Empty");
}

template <class Type> void cDSVec<Type>::Show()
{
     StdOut() << "cDSVeccDSVec ";
     for (const auto & aV : mSet.mOccupied)
         StdOut()  << " " << aV ;
     StdOut() << std::endl;
}

/* ************************************************************** */
/*                                                                */
/*                         cOneLinearConstraint                   */
/*                                                                */
/* ************************************************************** */

template <class Type> cOneLinearConstraint<Type>::cOneLinearConstraint(const tSV&aLP,const Type& aCste,int aNum) :
    mLP       (aLP),
    mISubst   (-1),
    mCste     (aCste),
    mNum      (aNum),
    mOrder    (-1),
    mReduced (false)
{
}

//  If aCUR SOL to exprimate the constraint
//  0=   A . X -C  = A. (X-X0) + A .X0 -C   :  C -= A .X0
template <class Type>  cOneLinearConstraint<Type> cOneLinearConstraint<Type>::Dup(const tDV * aCurSol) const
{
    cOneLinearConstraint<Type> aRes = *this;
    aRes.mLP = mLP.Dup();

    if (aCurSol)
    {
        for (const auto & aPair : aRes.mLP.IV())
            aRes.mCste -=  aPair.mVal * (*aCurSol)(aPair.mInd);
    }

    return aRes;
}

template <class Type>  cDenseVect<Type> cOneLinearConstraint<Type>::DenseVectRestored(int aNbVar) const
{
     cDenseVect<Type> aRes(mLP,aNbVar);
     if (mISubst>=0)
        aRes(mISubst)=1.0;
     return aRes;
}

template <class Type> void cOneLinearConstraint<Type>::InitSubst()
{
    mReduced = true;                        // memorize as reduced
    const tCplIV *  aCple =  LinearMax() ;  // Extract the coord with maximal amplitude
    mISubst  = aCple->mInd;                 // This is the variable that will be substituted
    Type aV0 = aCple->mVal;                 // Make a copy because Erase will lost reference ... !!! 

    mLP.EraseIndex(mISubst);  // supress the coeff substituted, it values implicitely one

    // Normalize the rest 
    for (auto & aPair : mLP.IV())  
        aPair.mVal /= aV0;
    mCste /= aV0;
}

template <class Type> const typename cOneLinearConstraint<Type>::tCplIV * cOneLinearConstraint<Type>::LinearMax() const
{
     // Extract the pair
     cWhichMax<const tCplIV*,Type> aMax(nullptr,-1.0);
     for (const auto & aPair : mLP)
        aMax.Add(&aPair,std::abs(aPair.mVal));
     const tCplIV  * aRes = aMax.IndexExtre();

     // Some check,if no pair is found, probably the system was degenerated
     MMVII_INTERNAL_ASSERT_tiny(aRes!=nullptr,"cOneLinearConstraint<Type>::LinearMax probably bad formed constrained");
     // to see later if we replace by |aRes->mVal| > Epsilon ?
     MMVII_INTERNAL_ASSERT_tiny(aRes->mVal!=0,"cOneLinearConstraint<Type>::LinearMax probably bad formed constrained");
     return aRes ;
}

template <class Type> void  GlobAddBuf(cDSVec<Type>  & aBuf,const cSparseVect<Type> & aLP,const Type & aMul)
{
    for (const auto & aPair : aLP.IV())
    {
        aBuf.AddValInd(aPair.mVal*aMul,aPair.mInd);
    }
}

template <class Type> void  cOneLinearConstraint<Type>::AddBuf(cDSVec<Type>  & aBuf,const Type & aMul) const
{
    GlobAddBuf(aBuf, mLP,aMul);
}

template <class Type> void cOneLinearConstraint<Type>::SubstituteInOtherConstraint(cOneLinearConstraint<Type> & aToSub,cDSVec<Type>  & aBuf)
{
    SubstituteInSparseLinearEquation(aToSub.mLP,aToSub.mCste,aBuf);
}

template <class Type> void cOneLinearConstraint<Type>::SubstituteInDenseLinearEquation(cDenseVect<Type> & aA,Type &  aB) const
{
	//  ?????????????
    //  A X -B =  A' X' + AiXi -B = A'(X-X0' + X0') +Ai(C -mLX0) - B
	
    //      (A'-Ai mL) X = B - Ai mC
    Type & aAi =  aA(mISubst);
    aB -= aAi *  mCste;

    for (const auto  & aPair : mLP)
        if (aPair.mInd != mISubst)
           aA(aPair.mInd) -=  aAi * aPair.mVal;

    aAi = 0.0;
}

template <class Type> void cOneLinearConstraint<Type>::SubstituteInSparseLinearEquation(tSV & aA,Type &  aB,cDSVec<Type>  & aBuf) const
{
    //      (A'-Ai mL) X = B - Ai mC
    tCplIV * aPairInA = aA.Find(mISubst) ;


   // current case, if the index is not present in equation nothing to do (in this case Ai=0 and A'=A)
    if (aPairInA == nullptr) return;
    Type aValAi = aPairInA->mVal;  // Save value because erase will supress the ref ...

    // substract constant
    aB -=  mCste * aValAi;
    // Substract  index 
    aA.EraseIndex(mISubst);
    // aPairInA->mVal = 0.0; same effect than erase 

    // other current case, if the equation is a single substition (like frozen var) no more things to do
    if (mLP.size()==0) return;

    GlobAddBuf(aBuf,aA, (Type)1.0);
    this ->AddBuf(aBuf,-aValAi);

    aA.Reset();
    for (const auto &  aInd : aBuf.mSet.mVIndOcc)
    {
         aA.AddIV(aInd,aBuf.mVec(aInd));
    }
    aBuf.Reset();
}


template <class Type> void  cOneLinearConstraint<Type>::SubstituteInOutRSNL(tIO_RSNL& aIO,cDSVec<Type>  & aBuf) const
{
    // [1]  Find the index of mISubst
    int aKSubst = -1;  // Indexe where mGlobVIn potentially equals mISubst

    for (size_t aKVar=0 ; aKVar<aIO.mGlobVInd.size() ; aKVar++)
    {
         int aIndGlob = aIO.mGlobVInd[aKVar];
         if ((! cSetIORSNL_SameTmp<Type>::IsIndTmp(aIndGlob))  && ( aIndGlob== mISubst))
            aKSubst = aKVar;
    }

    //  if index subst is not involved, nothing to do
    if (aKSubst<0)  return;

/*   Case everybody in diff (or abs)
 *
 *     F(X) =  F(X0) +  D (X-X0) =  D (X-X0)  + V0
 *            =   D' (X'-X0') +Di (Xi-X0i) + V0
 *  But constraint is writen :
 *      mLp.(X'-X0') + (Xi-Xi0) = C
 *
 *  Then :
 *     F(X) = D' (X'-X0')  + V0 + Di(C - mLp .(X'-X0')) = ( D'- Di mLp ) (X'-X0') + V0 +Di C
 */

    bool  FirstDerNN = true;  // Used to check that we  do only once the extension of indexe
    for (size_t aKEq=0 ; aKEq<aIO.mVals.size() ; aKEq++)
    {
        Type & aDerI = aIO.mDers.at(aKEq).at(aKSubst);
        if (aDerI !=0 )
        {
             aIO.mVals[aKEq]  +=  aDerI * mCste;
             if (mLP.size() != 1)
             {
                  // [A]  Compute the constant and put the linear part in buf (to be indexable)
                  for (const auto & aPair : mLP.IV())
                  {
                       MMVII_INTERNAL_ASSERT_tiny(aPair.mInd != mISubst,"Index error");
                       aBuf.AddValInd(aPair.mVal,aPair.mInd);  // We memorize indexe
                  }
                  // [B] modify the derivate using the index, also purge partially the buffer,
		  // after this only the update will be partial (only variable present in mGlobVInd will be incremented)
                  for (size_t aKVar=0 ; aKVar<aIO.mGlobVInd.size() ; aKVar++)
                  {
                      int aInd = aIO.mGlobVInd[aKVar];
		      // if it's not a temporay and it has been put in Buf, then update
                      if ( (! cSetIORSNL_SameTmp<Type>::IsIndTmp(aInd))  && aBuf.mSet.mOccupied.at(aInd) )
                      {
                         aIO.mDers.at(aKEq).at(aKVar) -= aBuf.mVec(aInd) * aDerI;  // -Di mL
                         aBuf.mSet.mOccupied.at(aInd) = false;  // purge occuo
                         aBuf.mVec(aInd) = 0;  // purge vector
                      }
                  }
                  // [C]  modify the derivate for the index, present in constraint but not in equation
                  for (const auto & aPair : mLP.IV())
                  {
                      if (aBuf.mSet.mOccupied.at(aPair.mInd)) // if 
                      {
                         aIO.mDers.at(aKEq).push_back(-aPair.mVal * aDerI); // - Di mL
                         if (FirstDerNN) //  (aKEq==0)
                            aIO.mGlobVInd.push_back(aPair.mInd);
                         aBuf.mSet.mOccupied.at(aPair.mInd) =false;  // purge occupied
                         aBuf.mVec(aPair.mInd) = 0.0;                // purge vector
                      }   
                  }
          
                  // [D]  finish purge
                  aBuf.mSet.mVIndOcc.clear();
             }
             aDerI = 0;      // now supress the derivate of substituted variable
             FirstDerNN= false;  // No longer first Non Null derivate
        }
    }
 
 /*  Case contraint in abs, F in diff
     F(X) =  F(X0) +  D (X-X0)   = D (X-X0)  + V0
          =   D' (X'-X0') +Di (Xi-X0i) + V0
          =   D' (X'-X0') + Di (mC- mL X' -X0i) + V0
          =   D' (X'-X0') -  Di mL (X' -X0' + X0') -  Di X0i     +  V0  +  Di mC
          =   (D' -Di mL) (X'-X0')  +  V0  +  Di (mC -X0i - mL X0' )
*/
    
    /*
    Type  aDelta = (mCste-aCurSol(mISubst));  // mC-X0i
    bool  FirstDerNN = true;  // Used to check that we  do only once the extension of indexe
    for (size_t aKEq=0 ; aKEq<aIO.mVals.size() ; aKEq++)
    {
        Type & aDerI = aIO.mDers.at(aKEq).at(aKSubst);

        if (aDerI !=0 )
        {
             aIO.mVals[aKEq]  +=  aDerI * aDelta;  // Add Di*(mC -X0i)
             if (mLP.size() != 1)
             {
                  // [A]  Compute the constant and put the linear part in buf (to be indexable)
                  for (const auto & aPair : mLP.IV())
                  {
                       MMVII_INTERNAL_ASSERT_tiny(aPair.mInd != mISubst,"Index error");
                       aIO.mVals[aKEq] -=   aPair.mVal *  aCurSol(aPair.mInd) *aDerI ;     // -Di mL X'0
                       aBuf.AddValInd(aPair.mVal,aPair.mInd);  // We memorize indexe
                  }
                  // [B] modify the derivate using the index, also purge partially the buffer,
		  // after this only the update will be partial (only variable present in mGlobVInd will be incremented)
                  for (size_t aKVar=0 ; aKVar<aIO.mGlobVInd.size() ; aKVar++)
                  {
                      int aInd = aIO.mGlobVInd[aKVar];
		      // if it's not a temporay and it has been put in Buf, then update
                      if ( (! cSetIORSNL_SameTmp<Type>::IsIndTmp(aInd))  && aBuf.mSet.mOccupied.at(aInd) )
                      {
                         aIO.mDers.at(aKEq).at(aKVar) -= aBuf.mVec(aInd) * aDerI;  // -Di mL
                         aBuf.mSet.mOccupied.at(aInd) = false;  // purge occuo
                         aBuf.mVec(aInd) = 0;  // purge vector
                      }
                  }
                  // [C]  modify the derivate for the index, present in constraint but not in equation
                  for (const auto & aPair : mLP.IV())
                  {
                      if (aBuf.mSet.mOccupied.at(aPair.mInd)) // if 
                      {
                         aIO.mDers.at(aKEq).push_back(-aPair.mVal * aDerI); // - Di mL
                         if (FirstDerNN) //  (aKEq==0)
                            aIO.mGlobVInd.push_back(aPair.mInd);
                         aBuf.mSet.mOccupied.at(aPair.mInd) =false;  // purge occupied
                         aBuf.mVec(aPair.mInd) = 0.0;                // purge vector
                      }   
                  }
          
                  // [D]  finish purge
                  aBuf.mSet.mVIndOcc.clear();
             }
             aDerI = 0;      // now supress the derivate of substituted variable
             FirstDerNN= false;  // No longer first Non Null derivate
        }
    }
*/
}

template <class Type> void cOneLinearConstraint<Type>::Show() const
{
    StdOut()  << "   * N=" << mNum << " O="<< mOrder << " S=" << mReduced  <<  " I=" << mISubst ;

    for (const auto & aPair:mLP.IV())
        StdOut() <<  " [" << aPair.mInd << " : " << aPair.mVal << "]";

    StdOut()  << std::endl;
}

/* ************************************************************** */
/*                                                                */
/*                         cBenchLinearConstr                     */
/*                                                                */
/* ************************************************************** */

class cBenchLinearConstr
{
     public :
         cBenchLinearConstr(int aNbVar,int aNbCstr);

         int  mNbVar;
         int  mNbCstr;
         std::vector<cSparseVect<tREAL8> >  mLSV; // list of sparse vect used by const
         std::vector<cDenseVect<tREAL8> >   mLDV; // list of dense vect, copy of sparse, for easiness

         // So that the constaint  L.V = C  is equiv to L(V-V0) = 0 
         cDenseVect<tREAL8>    mV0;
         cSetLinearConstraint<tREAL8> mSetC;
};


cBenchLinearConstr::cBenchLinearConstr(int aNbVar,int aNbCstr) :
    mNbVar  (aNbVar),
    mNbCstr (aNbCstr),
    mV0     (aNbVar,eModeInitImage::eMIA_RandCenter),
    mSetC   (aNbVar)
{
    static int aCpt=0; aCpt++;
    for (int aK=0 ; aK< mNbCstr ; aK++)
    {
          bool Ok = false;
          while (!Ok)
          {
               cSparseVect<tREAL8> aSV = cSparseVect<tREAL8>::RanGenerate(aNbVar,0.3,0.1,1);  // Generate random sparse vect
               cDenseVect<tREAL8>  aDV(aSV,mNbVar);  // make a dense copy 
               if (aDV.DistToSubspace(mLDV) > 0.2 * aDV.L2Norm())  // ensure the vect is enouh far from space defines by others
               {
                    Ok=true;  // done for this one
                    mLDV.push_back(aDV);  // add new sparse V
                    mLSV.push_back(aSV);  // add new dense V
                    cOneLinearConstraint<tREAL8>  aLC(aSV,mV0.DotProduct(aDV),aK);
                    mSetC.Add1Constr(aLC,nullptr);
               }
          }
    }
    // the paramater true will check that constraint reduced define the same space
    mSetC.Compile(true);

    /*  Check that the constraint are somewhat truangular sup, ie for any constraint
     *  the substiution variable is absent from the constraint after
     */
    for (int aK1=0 ; aK1< mNbCstr ; aK1++)
    {
        const cOneLinearConstraint<tREAL8> &  aC1= mSetC.mVCstrReduced.at(aK1);
        for (int aK2=0 ; aK2< mNbCstr ; aK2++)
        {
              const cOneLinearConstraint<tREAL8> &  aC2= mSetC.mVCstrReduced.at(aK2);
              
	      //  extract the posible term of C2 corresponding to substituate of C1
              auto * aPair = aC2.mLP.Find(aC1.mISubst);

              if (aK1<=aK2)
              {
		 // this term should not exist if C1 is before C2
                 MMVII_INTERNAL_ASSERT_bench((aPair==0) || (aPair->mVal==0.0) ,"Reduce in LinearCstr");
                 if (DEBUG)
                 {
                      StdOut()  << "PPP " << aPair 
                                << " N1=" << aC1.mNum  << "," << aC1.mOrder
                                << " N2=" << aC2.mNum  << "," << aC2.mOrder
                                << " Cpt=" << aCpt << "\n";
                 }
              }
        }
    }
}

void  BenchLinearConstr(cParamExeBench & aParam)
{
   //return;
   if (! aParam.NewBench("LinearConstr")) return;

   // std::vector<cPt2di>  aV{{2,3},{3,2}};

   for (int aK=0 ; aK<50 ; aK++)
   {
       cBenchLinearConstr(4,2);
       cBenchLinearConstr(10,2);
       cBenchLinearConstr(10,3);
       cBenchLinearConstr(20,5);
   }
   int aMul = std::min(4,1+aParam.Level());

   int aNb = std::max(1,int(100.0/pow(aMul,4)) );
   for (int aK=0 ; aK<aNb ; aK++)
   {
       int aNbVar  = 1 + aMul*50 * RandUnif_0_1();
       int aNbCstr =  (aNbVar>1) ? RandUnif_N(aNbVar-1) : 0 ;
       cBenchLinearConstr(aNbVar,aNbCstr);
   }

   for (int aK=0 ; aK<500 ; aK++)
   {
       int aNbVar  = 1 + 20 * RandUnif_0_1();
       int aNbCstr =  (aNbVar>1) ? RandUnif_N(aNbVar-1) : 0 ;
       aNbCstr = std::min(10,aNbCstr);
       cBenchLinearConstr(aNbVar,aNbCstr);
   }


   aParam.EndBench();
}


#define INSTANTIATE_LINEAER_CONSTR(TYPE)\
template class  cSetLinearConstraint<TYPE>;\
template class  cOneLinearConstraint<TYPE>;\
template class  cDSVec<TYPE>;

INSTANTIATE_LINEAER_CONSTR(tREAL16)
INSTANTIATE_LINEAER_CONSTR(tREAL8)
INSTANTIATE_LINEAER_CONSTR(tREAL4)

// template class  cOneLinearConstraint<tREAL16>;
// template class  cOneLinearConstraint<tREAL8>;
// template class  cOneLinearConstraint<tREAL4>;

};
