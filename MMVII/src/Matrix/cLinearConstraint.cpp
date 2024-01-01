
#include "MMVII_Tpl_Images.h"

#include "MMVII_SysSurR.h"

using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{
static bool DEBUG=false;
//static bool DEBUG2=false;

template <class Type>  class  cDSVec;   // Sparse/Dense vect
template <class Type>  class  cOneLinearConstraint;  // represent 1 constr
template <class Type>  class  cSetLinearConstraint;  // represent a set of constraint

class cBenchLinearConstr;

/**  Class for a "sparse" dense vector,  i.e a vector that is represented by a dense vector
 */

template <class Type> class cDSVec
{
    public :
       cDSVec(size_t aNbVar);

       void AddValInd(const Type &,int anInd);

       cDenseVect<Type>  mVec;
       cSetIntDyn        mSet;

       void Reset();
       void Show();
       void TestEmpty();
};

/*    Class for handling linear constraint in non linear optimization system.
 *    Note the constraint on a vector X as :
 *
 *         mL . X = mC      where mL is a non null vector
 *
 * (I)   The way it is done is by substitution  :
 *
 *       (1) We select an arbitray non null coord of L  Li!=0; (something like the biggest one)
 *       (2) We suppose Li=1.0  (in fact we have it by setting  mL = mL/Li  , mC = mC/Li)
 *       (3) Let note X' the vector X without Xi
 *       (4) we have Xi =  mC- mL X'
 *       (5) Each time we add a new obs in sytem :
 *            A.X = B
 *            A.X-B =  A' X' -B + Ai Xi =  A' X' -B + Ai (mC-mL X')  
 *            (A'-Ai mL) X = B - Ai mC
 *
 *  (II)   So far so good, but now supose we have the two contraint:
 *       C1:  x +2y=0   C2  2x + y = 0  
 *    And a form  L :x + y +z, obviouly  as the two constraint implie x=y=0, it mean that L shoul reduce to z
 *
 *     But suppose  we use C1 as x ->-2y  and C2 as C2A : y-> -2x  or C2B  x-> -y/2
 *         using C1 we have  L ->  -y+z  and 
 *              C2A ->  2x+z
 *              C2B ->  -y+z  (nothing to do, x is already substitued)
 * 
 *     So this does not lead to the good reduction 
 *
 *  (III)  So now, before using the constraint we make a preprocessing, more a less a triangulation :
 *
 *        C1 :  x + 2y=0  ~   x->-2y
 *        C'2 :   C2(x->-2y) : -y=0      
 *
 *      now if we use C1 and C'2  L will reduce to 0  
 *    This the principe used in MMVII for constrained optimization : make a substitution afer a preprocessing
 *   that triangulate the constraint
 */


template <class Type>  class cOneLinearConstraint
{
     public :
       friend class cSetLinearConstraint<Type>;
       friend class cBenchLinearConstr;

       typedef cSparseVect<Type>          tSV;
       typedef cDenseVect<Type>           tDV;
       typedef typename tSV::tCplIV       tCplIV;
       typedef cInputOutputRSNL<Type>     tIO_RSNL;
       typedef cSetIORSNL_SameTmp<Type>   tSetIO_ST;

       /**  In Cstr we can fix the index of subst, if it value -1 let the system select the best , fixing can be usefull in case
	* of equivalence
	*/
        cOneLinearConstraint(const tSV&aLP,const Type& aCste,int aNum);

        // Subsract into "aToSub" so as to annulate the coeff with mISubst
	void SubstituteInOtherConstraint(cOneLinearConstraint<Type> & aToSub,cDSVec<Type>  & aBuf);
	void SubstituteInDenseLinearEquation (tDV & aA,Type &  aB) const;
        void SubstituteInSparseLinearEquation(tSV & aA,Type &  aB,cDSVec<Type>  & aBuf) const;
        void SubstituteInOutRSNL(tIO_RSNL& aIO,cDSVec<Type>  & aBuf,const tDV & aCurSol) const;

        const tCplIV *  LinearMax() const;

        /// One the Index of substitution is chosen, transformat by divide all equation by Li and supress Li tha implicitely=1
        void InitSubst();

        /// 4 Debug purpose
        void Show() const;

     private :

	void  AddBuf(cDSVec<Type>  & aBuf,const Type & aMul,int aI2Avoid) const;

        tSV       mLP;       /// Linear part
        int       mISubst;   /// Indexe which is substituted
	Type      mCste;     /// Constant of the constrainte   
        int       mNum;      /// Identifier, used for debug at least
        int       mOrder;    /// Order of reduction, used to sort the constraint
        bool      mSelected; /// a marker to know if a constraint has already been reduced
};

template <class Type>  class  cSetLinearConstraint
{
    public :
          friend class cBenchLinearConstr;

          typedef cSparseVect<Type>          tSV;
          typedef cDenseVect<Type>           tDV;
          typedef typename tSV::tCplIV       tCplIV;
          typedef cOneLinearConstraint<Type>  t1Constr;

          void  Compile();
          cSetLinearConstraint(int aNbVar);
          void Add1Constr(const t1Constr &);

          void Show(const std::string & aMsg) const;
    private :
          std::vector<t1Constr>   mVCstrInit;     // Initial constraint, 
          std::vector<t1Constr>   mVCstrReduced;  // Constraint after reduction
          cDSVec<Type>            mBuf;           // Buffer for computation
};

/* ************************************************************ */
/*                                                              */
/*                cSetLinearConstraint                          */
/*                                                              */
/* ************************************************************ */

template <class Type> cSetLinearConstraint<Type>::cSetLinearConstraint(int aNbVar) :
   mBuf (aNbVar)
{
}

template <class Type> void cSetLinearConstraint<Type>::Add1Constr(const t1Constr & aConstr)
{
      mVCstrInit.push_back(aConstr);
}


template <class Type> void cSetLinearConstraint<Type>::Compile()
{
    mVCstrReduced  = mVCstrInit;

    if (DEBUG) 
       Show("Init");

    //  Set no selected for all
    for (auto &  aCstr : mVCstrReduced)
        aCstr.mSelected = false;

    size_t aNbReduced = 0;
    while (aNbReduced != mVCstrInit.size())
    {
          // extract the  "best" Cstr, i.e. with highest value
          cWhichMax<int,Type> aMax(-1,-1.0);
          for (int aKC=0; aKC<int(mVCstrReduced.size()) ; aKC++)
          {
              if (! mVCstrReduced.at(aKC).mSelected) 
                 aMax.Add(aKC,std::abs(mVCstrReduced.at(aKC).LinearMax()->mVal));
          }
          t1Constr& aBest = mVCstrReduced.at(aMax.IndexExtre());

          aBest.InitSubst();
          aBest.mOrder = aNbReduced;
          // substract the selected constraint to all
          for (t1Constr &  aCstr : mVCstrReduced)
          {
               if (! aCstr.mSelected)
               {
                    if (DEBUG)
                        StdOut()  <<  "SIOC, ISUBS " << aBest.mISubst << " N=" << aBest.mNum << " => " << aCstr.mNum << "\n";
                    aBest.SubstituteInOtherConstraint(aCstr,mBuf);
               }
          }
          if (DEBUG) 
          {
                if (DEBUG) Show("Reduc:" + ToStr(aNbReduced));
          }
          aNbReduced++;
    }
    if (DEBUG) 
       StdOut()  <<  "=======================================\n";
    std::sort
    (
         mVCstrReduced.begin(),
         mVCstrReduced.end(),
         [](const auto & aC1,const auto & aC2){return aC1.mOrder<aC2.mOrder;}
     );
}

template <class Type> void cSetLinearConstraint<Type>::Show(const std::string & aMsg) const
{
     StdOut()  << "========  SHOWTSELC " << aMsg << " =====================" << std::endl;

     for (const auto & aCstr: mVCstrReduced)
        aCstr.Show();
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
     static bool First = true;
     if (First)
        StdOut() << "TestEmptyTestEmptyTestEmptyTestEmptyTestEmptyTestEmptyTestEmptyTestEmpty\n";
     First = false;
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
        mSelected (false)
{
}


template <class Type> void cOneLinearConstraint<Type>::InitSubst()
{
    mSelected = true;
    const tCplIV *  aCple =  LinearMax() ;
    mISubst  = aCple->mInd;

    mLP.EraseIndex(mISubst);

    for (auto & aPair : mLP.IV())
        aPair.mVal /= aCple->mVal;
    mCste /= aCple->mVal;
}

template <class Type> const typename cOneLinearConstraint<Type>::tCplIV * cOneLinearConstraint<Type>::LinearMax() const
{
     cWhichMax<const tCplIV*,Type> aMax(nullptr,-1.0);

     for (const auto & aPair : mLP)
        aMax.Add(&aPair,std::abs(aPair.mVal));

     const tCplIV  * aRes = aMax.IndexExtre();
     MMVII_INTERNAL_ASSERT_tiny(aRes!=nullptr,"cOneLinearConstraint<Type>::LinearMax probably bad formed cosntrained");
     MMVII_INTERNAL_ASSERT_tiny(aRes->mVal!=0,"cOneLinearConstraint<Type>::LinearMax probably bad formed cosntrained");
     return aRes ;
}

template <class Type> void  GlobAddBuf(cDSVec<Type>  & aBuf,const cSparseVect<Type> & aLP,const Type & aMul,int aI2Avoid) 
{
    for (const auto & aPair : aLP.IV())
    {
         if (aPair.mInd != aI2Avoid)
	 {
              aBuf.AddValInd(aPair.mVal*aMul,aPair.mInd);
	 }
    }
}

template <class Type> void  cOneLinearConstraint<Type>::AddBuf(cDSVec<Type>  & aBuf,const Type & aMul,int aI2Avoid) const
{
    GlobAddBuf(aBuf, mLP,aMul,aI2Avoid);
}

template <class Type> void cOneLinearConstraint<Type>::SubstituteInOtherConstraint(cOneLinearConstraint<Type> & aToSub,cDSVec<Type>  & aBuf)
{
  static int aCpt=0; aCpt++;
  //DEBUG2 = (aCpt==3);
  //if (DEBUG) StdOut() << "INNNN " << aToSub.mLP.Find(mISubst) << " Cpt=" << aCpt << " B2 "<< DEBUG2 << "\n";

    SubstituteInSparseLinearEquation(aToSub.mLP,aToSub.mCste,aBuf);

  //if (DEBUG) StdOut() << "OUUT " << aToSub.mLP.Find(mISubst) << "\n";
}

template <class Type> void cOneLinearConstraint<Type>::SubstituteInDenseLinearEquation(cDenseVect<Type> & aA,Type &  aB) const
{
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
    const tCplIV * aPairInA = aA.Find(mISubst) ;

    aBuf.TestEmpty();
    /*if (DEBUG2)
    {
         StdOut()  << "PAIR " << aPair << " SZ=" << mLP.size() << "\n";
    }*/


   // current case, if the index is not present in equation nothing to do (in this case Ai=0 and A'=A)
    if (aPairInA == nullptr) return;
    Type aValAi = aPairInA->mVal;  // Save value because erase will supress the ref ...

    // substract constant
    aB -=  mCste * aValAi;
    // Substract 
    aA.EraseIndex(mISubst);
    // other current case, if the equation is a single substition (like frozen var) no more things to do
    if (mLP.size()==0) return;

//if (DEBUG2) aBuf.Show();
    // mIsSubst is send as  parameter because it must disapear in the buf
    GlobAddBuf(aBuf,aA, (Type)1.0,mISubst);
//if (DEBUG2) aBuf.Show();
    this ->AddBuf(aBuf,-aValAi,mISubst);
//if (DEBUG2) aBuf.Show();

    aA.Reset();
    for (const auto &  aInd : aBuf.mSet.mVIndOcc)
    {
         aA.AddIV(aInd,aBuf.mVec(aInd));
    }
    aBuf.Reset();
    aBuf.TestEmpty();
}


template <class Type> void  cOneLinearConstraint<Type>::SubstituteInOutRSNL(tIO_RSNL& aIO,cDSVec<Type>  & aBuf,const tDV & aCurSol) const
{
    // [1]  Find the index of mISubst
    int aKSubst = -1;  // Indexe wher mGlobVIn potentially equals mISubst

    for (size_t aKVar=0 ; aKVar<aIO.mGlobVInd.size() ; aKVar++)
    {
         int aIndGlob = aIO.mGlobVInd[aKVar];
         if ((! cSetIORSNL_SameTmp<Type>::IsIndTmp(aIndGlob))  && ( aIndGlob== mISubst))
            aKSubst = aKVar;
    }

    //  if index subst is not involved, nothing to do
    if (aKSubst<0)  return;
 
 /*  F(X) =  F(X0) +  D (X-X0)   = D (X-X0)  + V0
          =   D' (X'-X0') +Di (Xi-X0i) + V0
          =   D' (X'-X0') + Di (mC- mL X' -X0i) + V0
          =   D' (X'-X0') -  Di mL (X' -X'0 + X'0) -  Di X0i     +  V0  +  Di mC
          =   (D' -Di mL) (X'-X0')  +  V0  +  Di (mC -X0i - mL X'0 )
*/
    
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
                      if (aPair.mInd != mISubst)
                      {
                          aIO.mVals[aKEq] -=   aPair.mVal *  aCurSol(aPair.mInd) *aDerI ;     // -Di mL X'0
                          aBuf.AddValInd(aPair.mVal,aPair.mInd);  // We memorize indexe
                      }
                  }
                  // [B] modify the derivate using the index, also purge partially the buffer
                  for (size_t aKVar=0 ; aKVar<aIO.mGlobVInd.size() ; aKVar++)
                  {
                      int aInd = aIO.mGlobVInd[aKVar];
                      if ( (! cSetIORSNL_SameTmp<Type>::IsIndTmp(aInd))  && aBuf.mSet.mOccupied.at(aInd) )
                      {
                         aIO.mDers.at(aKEq).at(aKVar) -= aBuf.mVec(aInd) * aDerI;  // -Di mL
                         aBuf.mSet.mOccupied.at(aInd) = false;  // purge occuo
                         aBuf.mVec(aInd) = 0;  // purge 
                      }
                  }
                  // [C]  modify the derivate for the index, presnt in constraint but not in equation
                  for (const auto & aPair : mLP.IV())
                  {
                      if (aPair.mInd != mISubst)
                      {
                          if (aBuf.mSet.mOccupied.at(aPair.mInd))
                          {
                              aIO.mDers.at(aKEq).push_back(-aPair.mVal * aDerI);
                              if (FirstDerNN) //  (aKEq==0)
                                 aIO.mGlobVInd.push_back(aPair.mInd);
                              aBuf.mSet.mOccupied.at(aPair.mInd) =false;
                              aBuf.mVec(aPair.mInd) = 0.0;
                          }   
                      }
                  }
          
                  // [D]  finish purge
                  aBuf.mSet.mVIndOcc.clear();
             }
             aDerI = 0;
             FirstDerNN= false;
        }
    }
}

template <class Type> void cOneLinearConstraint<Type>::Show() const
{
    StdOut()  << "   * N=" << mNum << " O="<< mOrder << " S=" << mSelected  <<  " I=" << mISubst ;

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
// DEBUG=  (aCpt==6);
    StdOut() << "0-NNNNnBvar " << mNbVar << " NbC=" << mNbCstr   << " Cpt " << aCpt << "\n";

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
                    mSetC.Add1Constr(aLC);
               }
          }
    }
    StdOut() << "A-NNNNnBvar " << mNbVar << " NbC=" << mNbCstr   << " Cpt " << aCpt << "\n";
    // if (!DEBUG) return; 
    mSetC.Compile();
    StdOut() << "B-NNNNnBvar " << mNbVar << " NbC=" << mNbCstr   << " Cpt " << aCpt << "\n";

    for (int aK1=0 ; aK1< mNbCstr ; aK1++)
    {
        const cOneLinearConstraint<tREAL8> &  aC1= mSetC.mVCstrReduced.at(aK1);
        for (int aK2=0 ; aK2< mNbCstr ; aK2++)
        {
              const cOneLinearConstraint<tREAL8> &  aC2= mSetC.mVCstrReduced.at(aK2);
              
              auto * aPair = aC2.mLP.Find(aC1.mISubst);

              if (aK1<=aK2)
              {
                 if (DEBUG)
                 {
                      StdOut()  << "PPP " << aPair 
                                << " N1=" << aC1.mNum  << "," << aC1.mOrder
                                << " N2=" << aC2.mNum  << "," << aC2.mOrder
                                << " Cpt=" << aCpt << "\n";
                 }
                 MMVII_INTERNAL_ASSERT_bench(aPair==0,"Reduce in LinearCstr");
              }
        }
    }
    StdOut() << "C-NNNNnBvar " << mNbVar << " NbC=" << mNbCstr   << " Cpt " << aCpt << "\n";
}

void  BenchLinearConstr(cParamExeBench & aParam)
{
return;
   if (! aParam.NewBench("LinearConstr")) return;

   StdOut()  << "BenchLinearConstrBenchLinearConstr\n";
   // std::vector<cPt2di>  aV{{2,3},{3,2}};

   for (int aK=0 ; aK<100 ; aK++)
   {
       cBenchLinearConstr(4,2);
       cBenchLinearConstr(10,2);
       cBenchLinearConstr(10,3);
       cBenchLinearConstr(20,5);
   }

   for (int aK=0 ; aK<5000 ; aK++)
   {
       int aNbVar  = 1 + 100 * RandUnif_0_1();
       int aNbCstr =  (aNbVar>1) ? RandUnif_N(aNbVar-1) : 0 ;
       // int aNbCstr=4;
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
