#ifndef _LINEARCONSTRAINT_H_
#define _LINEARCONSTRAINT_H_

#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"

using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{
//  static bool DEBUG=false;
//static bool DEBUG2=false;

template <class Type>  class  cDSVec;   // Sparse/Dense vect
template <class Type>  class  cOneLinearConstraint;  // represent 1 constr
template <class Type>  class  cSetLinearConstraint;  // represent a set of constraint
class cBenchLinearConstr;

/**  Class for a "sparse" dense vector,  i.e a vector that is represented by a dense vector
 */

template <class Type> class cDSVec : public cMemCheck
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


template <class Type>  class cOneLinearConstraint : public cMemCheck
{
     public :
       friend class cSetLinearConstraint<Type>;
       friend class cBenchLinearConstr;

       typedef cSparseVect<Type>          tSV;
       typedef cDenseVect<Type>           tDV;
       typedef typename tSV::tCplIV       tCplIV;
       typedef cInputOutputRSNL<Type>     tIO_RSNL;

       /**  In Cstr we can fix the index of subst, if it value -1 let the system select the best , fixing can be usefull in case
	* of equivalence
	*/
        cOneLinearConstraint(const tSV&aLP,const Type& aCste,int aNum);
        cOneLinearConstraint Dup() const;

        // Subsract into "aToSub" so as to annulate the coeff with mISubst
        void SubstituteInOtherConstraint(cOneLinearConstraint<Type> & aToSub,cDSVec<Type>  & aBuf);
        void SubstituteInDenseLinearEquation (tDV & aA,Type &  aB) const;
        void SubstituteInSparseLinearEquation(tSV & aA,Type &  aB,cDSVec<Type>  & aBuf) const;
        void SubstituteInOutRSNL(tIO_RSNL& aIO,cDSVec<Type>  & aBuf) const;

	///  Extract pair with maximal amplitude (in abs)
        const tCplIV *  LinearMax() const;

        /// One the Index of substitution is chosen, transformat by divide all equation by Li and supress Li tha implicitely=1
        void InitSubst();

        /// 4 Debug purpose
        void Show() const;

     private :

        /// Return the vector with the X[mISubst] to 1 (for test)
         cDenseVect<Type> DenseVectRestored(int aNbVar) const;

	void  AddBuf(cDSVec<Type>  & aBuf,const Type & aMul) const;

        tSV       mLP;       /// Linear part
        int       mISubst;   /// Indexe which is substituted
	Type      mCste;     /// Constant of the constrainte   
        int       mNum;      /// Identifier, used for debug at least
        int       mOrder;    /// Order of reduction, used to sort the constraint
        bool      mReduced; /// a marker to know if a constraint has already been reduced
};

template <class Type>  class  cSetLinearConstraint : public cMemCheck
{
    public :
          friend class cBenchLinearConstr;

          typedef cInputOutputRSNL<Type>     tIO_RSNL;
          typedef cSparseVect<Type>          tSV;
          typedef cDenseVect<Type>           tDV;
          typedef typename tSV::tCplIV       tCplIV;
          typedef cOneLinearConstraint<Type> t1Constr;

	  /// Cstr : allow the buffer for computatio,
          cSetLinearConstraint(int aNbVar);
	  /// Transformate the set of constraint to allow a cascade os substitution
          void  Compile(bool ForBench);
	  /// Add a new constraint (just debug)
          void Add1Constr(const t1Constr &,const tDV *);

	  void Reset();
	  void Add1ConstrFrozenVar(int aKVar,const Type & aVal,const tDV *);

          void SubstituteInSparseLinearEquation(tSV & aA,Type &  aB) const;
          void SubstituteInDenseLinearEquation (tDV & aA,Type &  aB) const;
          void SubstituteInOutRSNL(tIO_RSNL& aIO) const;
    private :
	  /// Show all the detail
          void Show(const std::string & aMsg) const;
          ///  Test that the reduced contsraint define the same space than initial (a bit slow ...)
          void TestSameSpace();

          std::vector<t1Constr>   mVCstrInit;     // Initial constraint, 
          std::vector<t1Constr>   mVCstrReduced;  // Constraint after reduction
	  int                     mNbVar;
          mutable cDSVec<Type>            mBuf;           // Buffer for computation
};

};
#endif // _LINEARCONSTRAINT_H_


