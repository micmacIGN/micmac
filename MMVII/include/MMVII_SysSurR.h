#ifndef  _MMVII_SysSurR_H_
#define  _MMVII_SysSurR_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Matrix.h"

namespace MMVII
{
/** \file MMVII_SysSurR.h
    \brief Classes for linear redundant system
*/

template <class Type> class  cInputOutputRSNL;
template <class Type> class  cSetIORSNL_SameTmp;
template <class Type> class  cLinearOverCstrSys  ;
template <class Type> class  cLeasSq ;
template <class Type> class  cLeasSqtAA ;
template <class Type> class  cBufSchurSubst;
template <class Type> class  cSetIORSNL_SameTmp;
template <class Type> class cResidualWeighter;
// template <class Type> class cObjOfMultipleObjUk;
template <class Type> class cObjWithUnkowns;
template <class Type> class cSetInterUK_MultipeObj;
template <class Type>  class  cSetLinearConstraint; // defined in "src/Matrix"


/**  Class for weighting residuals : compute the vector of weight from a 
     vector of residual; default return {1.0,1.0,...}
 */
template <class Type> class cResidualWeighter
{
       public :
            typedef std::vector<Type>     tStdVect;

            cResidualWeighter(const Type & aVal=1.0);
            virtual tStdVect WeightOfResidual(const tStdVect &) const;
       private :
             Type mVal;

};

/**  Class for weighting residuals with explicit weight of each residual
 */
template <class Type> class cResidualWeighterExplicit: public cResidualWeighter<Type>
{
       public :
            typedef std::vector<Type>     tStdVect;

            cResidualWeighterExplicit(bool isSigmas, const tStdVect & aData);
            virtual tStdVect WeightOfResidual(const tStdVect &) const override;
            tStdVect & getSigmas() { return mSigmas; }
            tStdVect & geWeights() { return mWeights; }
            int size() const { return mWeights.size(); }
       private :
            tStdVect mSigmas;
            tStdVect mWeights;
};


template <class Type> class cREAL8_RWAdapt : public cResidualWeighter<Type>
{
       public :
            typedef std::vector<Type>     tStdVect;
            cREAL8_RWAdapt(const cResidualWeighter<tREAL8> * aRW) ;
            tStdVect WeightOfResidual(const tStdVect & aVIn) const override;
       private :
            const cResidualWeighter<tREAL8>* mRW;
};


/// Index to use in vector of index indicating a variable to substituate
static constexpr int RSL_INDEX_SUBST_TMP = -1;

/**    cREAL8_RSNL   :  For now  deprecated , 
 *
 *       The idea was to have a non template interface with REAL8 object for  cResolSysNonLinear
 *       while having object in herited class with 8 or 16 byte for higher accuracy
 *
 *       For each method we need an access with REAL8 object , we habe
 *
 *            Method(Type ...)  in cResolSysNonLinear
 *            R_Method(tREAL8 ...) declared in cREAL8_RSNL and implemanted in cResolSysNonLinear
 *            as far as possible R_Method just call Method, but the type conversion is not always easy ...
 *
 *
 *       At the end it seems difficult to avoid some code duplication,  so for now I (MPD) take rather
 *       the direction of maintaining a template bundle adj
 *
 *       BTW, I maintain the transformation that has already be done, because not sure I will change again my mind
 *
 *       Also maybe if will usefull even for template case ....
*/

class cREAL8_RSNL
{
	public :

          cREAL8_RSNL(int aNbVar);

          typedef cDenseVect<tREAL8>                              tDVect;
          typedef cSparseVect<tREAL8>                             tSVect;
          typedef std::vector<int>                                tVectInd;
          typedef std::vector<tREAL8>                             tStdVect;
          typedef cResidualWeighter<tREAL8>                       tResidualW;
          typedef NS_SymbolicDerivative::cCalculator<tREAL8>      tCalc;
          typedef cSetIORSNL_SameTmp<tREAL8>                      tSetIO_ST;
          typedef cObjWithUnkowns<tREAL8>                         tObjWUk;

          virtual  ~cREAL8_RSNL();
	 /// basic allocator, using a mode of matrix + a solution  init
          static  cREAL8_RSNL * Alloc(eModeSSR,const tDVect & aInitSol);

          /// Accessor
          virtual tDVect    R_CurGlobSol() const = 0;
   
          /// Accessor
          virtual int R_NbVar() const = 0;  
          /// Value of a given num var
          virtual tREAL8    R_CurSol(int aNumV) const = 0;
          /// Set value, usefull for ex in dev-mesh because variable are activated stepby step
          virtual void R_SetCurSol(int aNumV,const tREAL8&) =0 ;
	  /// 
          virtual  tDVect    R_SolveUpdateReset(const tREAL8 & aLVM=0.0) = 0 ;  // Levenberg markard

          virtual void   R_AddEqFixVar(const int & aNumV,const tREAL8 & aVal,const tREAL8& aWeight) =0;
          virtual void   R_AddEqFixCurVar(const int & aNumV,const tREAL8 & aWeight) =0;

          virtual void   R_CalcAndAddObs(tCalc *,const tVectInd &,const tStdVect& aVObs,const tResidualW & = tResidualW()) = 0;

          virtual void  R_AddEq2Subst (tSetIO_ST & aSetIO,tCalc *,const tVectInd &,
                                       const tStdVect& aVObs,const tResidualW & = tResidualW()) = 0;
          virtual void  R_AddObsWithTmpUK (const tSetIO_ST & aSetIO) =0;

	   virtual void  R_SetFrozenVar(int aK,const  tREAL8 &) = 0;  ///< seti var var frozen /unfrozen


	  void  SetUnFrozen(int aK);  ///< indicate it var must be frozen /unfrozen
	  void  UnfrozeAll() ;                       ///< indicate it var must be frozen /unfrozen
	  bool  VarIsFrozen(int aK) const;           ///< indicate it var must be frozen /unfrozen
	  void  AssertNotInEquation() const;         ///< verify that we are notin equation step (to allow froze modification)
          // To update with Shared
	  int   CountFreeVariables() const;          ///< number of free variables

          // ------------------ Handling shared unknowns --------------------
          void   SetShared(const std::vector<int> &  aVUk);
          void   SetUnShared(const std::vector<int> &  aVUk);
          void   SetAllUnShared();

          //  ===
	protected :
          static constexpr int  TheLabelFrozen  =-1;
          static constexpr int  TheLabelNoEquiv =-2;

          void SetPhaseEq();
	  /// Mut be defined in inherited class because maniupulate mLinearConstr which depend of type
	  virtual void InitConstraint() = 0;

	  int                  mNbVar;
	  bool                 mInPhaseAddEq;   ///< check that dont modify val fixed after adding  equations
	  std::vector<bool>    mVarIsFrozen;    ///< indicate for each var is it is frozen
          int                  mNbIter;         ///< Number of iteration made
          // int                  mNbUnkown;
          int                  mCurMaxEquiv;       ///< Used to label the 
	  std::vector<int>     mEquivNum;       ///< Equivalence numerotation, used for shared unknowns
};



/**  Class for solving non linear system of equations
 */
template <class Type> class cResolSysNonLinear : public cREAL8_RSNL
{
      public :
          typedef cREAL8_RSNL                                   tR_Up;
          typedef tREAL8                                        tNumCalc;
          typedef NS_SymbolicDerivative::cCalculator<tNumCalc>  tCalc;
          typedef std::vector<tNumCalc>                         tStdCalcVect;
          typedef cInputOutputRSNL<Type>                        tIO_RSNL;
          typedef cSetIORSNL_SameTmp<Type>                      tSetIO_ST;


          typedef cLinearOverCstrSys<Type>                      tLinearSysSR;
          typedef cDenseVect<Type>                              tDVect;
          typedef cSparseVect<Type>                             tSVect;
          typedef std::vector<Type>                             tStdVect;
          typedef std::vector<int>                              tVectInd;
          typedef cResolSysNonLinear<Type>                      tRSNL;
          typedef cResidualWeighter<Type>                       tResidualW;
          typedef cObjWithUnkowns<Type>                         tObjWUk;

	  /// basic constructor, using a mode of matrix + a solution  init
          cResolSysNonLinear(eModeSSR,const tDVect & aInitSol);
	  ///  constructor  using linear system, allow finer control
          cResolSysNonLinear(tLinearSysSR *,const tDVect & aInitSol);
	  /// destructor 
          ~cResolSysNonLinear();


          /// Accessor
          const tDVect  &    CurGlobSol() const;
	  cREAL8_RSNL::tDVect    R_CurGlobSol() const override;  ///<  tREAL8 Equivalent
   
          /// Accessor
          int NbVar() const;  
          int R_NbVar() const override;  ///< tREAL8 Equivalent

          /// Value of a given num var
          const Type  &    CurSol(int aNumV) const;
          tREAL8    R_CurSol(int aNumV) const override; ///< tREAL8 Equivalent
          /// Set value, usefull for ex in dev-mesh because variable are activated stepby step
          void SetCurSol(int aNumV,const Type&) ;
          void R_SetCurSol(int aNumV,const tREAL8&) override; ///< tREAL8 Equivalent

          tLinearSysSR *  SysLinear() ; ///< Accessor

          /// Solve solution,  update the current solution, Reset the least square system
          const tDVect  &    SolveUpdateReset(const Type & aLVM =0.0) ;
	  cREAL8_RSNL::tDVect      R_SolveUpdateReset(const tREAL8& = 0.0) override ;

          /// Add 1 equation fixing variable
          void   AddEqFixVar(const int & aNumV,const Type & aVal,const Type& aWeight);
          void   R_AddEqFixVar(const int & aNumV,const tREAL8 & aVal,const tREAL8& aWeight) override;
          /// Add equation to fix variable to current value
          void   AddEqFixCurVar(const int & aNumV,const Type& aWeight);
          void   R_AddEqFixCurVar(const int & aNumV,const tREAL8 & aWeight) override;

          void   AddEqFixCurVar(const tObjWUk & anObj,const  Type & aVal,const Type& aWeight);
          void   AddEqFixCurVar(const tObjWUk & anObj,const  Type * aVal,size_t aNb,const Type& aWeight);
          void   AddEqFixCurVar(const tObjWUk & anObj,const  cPtxd<Type,3> &,const Type& aWeight);


          void   AddEqFixNewVal(const tObjWUk & anObj,const  Type & aV2Fix,const  Type & aNewVal,const Type& aWeight);
          void   AddEqFixNewVal(const tObjWUk & anObj,const  Type * aVal,const  Type * aNewVal,size_t aNb,const Type& aWeight);
          void   AddEqFixNewVal(const tObjWUk & anObj,const  cPtxd<Type,3> &,const  cPtxd<Type,3> &,const Type& aWeight);


          void AddNonLinearConstr(tCalc * aCalcVal,const tVectInd & aVInd,const tStdVect& aVObs,bool  OnlyIfFirst=true);

          /// Basic Add 1 equation , no bufferistion, no schur complement
          void   CalcAndAddObs(tCalc *,const tVectInd &,const tStdVect& aVObs,const tResidualW & = tResidualW());
          void   R_CalcAndAddObs(tCalc *,const tVectInd &,const  tR_Up::tStdVect& aVObs,const tR_Up::tResidualW & ) override;

          /**  Add 1 equation in structure aSetIO ,  who will accumulate all equation of a given temporary set of unknowns
	       relatively basic 4 now because don't use parallelism of tCalc
	  */
          void  AddEq2Subst (tSetIO_ST & aSetIO,tCalc *,const tVectInd &,
                             const tStdVect& aVObs,const tResidualW & = tResidualW());

          void  R_AddEq2Subst (tR_Up::tSetIO_ST  & aSetIO,tCalc *,const tVectInd &,
                             const tR_Up::tStdVect& aVObs,const tR_Up::tResidualW &) override;

	  /** Once "aSetIO" has been filled by multiple calls to  "AddEq2Subst",  do it using for exemple schur complement
	   */
          void  AddObsWithTmpUK (const tSetIO_ST & aSetIO);
          void  R_AddObsWithTmpUK (const tR_Up::tSetIO_ST & aSetIO) override;

	       //    frozen  checking

	   void  SetFrozenVar(int aK,const  Type &);  ///< seti var var frozen /unfrozen
	   void  R_SetFrozenVar(int aK,const  tREAL8 &) override;  ///< seti var var frozen /unfrozen
	   void  SetFrozenVarCurVal(int aK);  ///< idem to current val
	       // frozen for  cObjWithUnkowns
	   void  SetFrozenVarCurVal(tObjWUk & anObj,const  Type & aVal);  ///< Froze the value aVal, that must belong to anObj
	   void  SetFrozenVarCurVal(tObjWUk & anObj,const  Type * Vals,size_t aNb);  ///< Froze Nb values aVal, that must belong to anObj
	   void  SetFrozenVarCurVal(tObjWUk & anObj,const tStdVect & aVect);  ///< Froze aVect, that must belong to anObj
	   void  SetFrozenVarCurVal(tObjWUk & anObj,const cPtxd<Type,3> & aPt);  ///< Froze aPt that must belong to anObj
	   void  SetFrozenVarCurVal(tObjWUk & anObj,const cPtxd<Type,2> & aPt);  ///< Froze aPt that must belong to anObj
	   void  SetFrozenAllCurrentValues(tObjWUk & anObj);  ///< Froze all the value beloning to an anObj

	   void  SetFrozenFromPat(tObjWUk & anObj,const std::string& , bool Frozen);  ///< Froze all the value beloning to an anObj

           void AddObservationLinear(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS)  ;
           void AddObservationLinear(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) ;


	   void  SetUnFrozenVar(tObjWUk & anObj,const  Type & aVal); ///< Unfreeze the value, that must belong to anObj

	   int   GetNbObs() const;                    ///< get number of observations (last iteration if after reset, or current number if after AddObs)

          void  AddConstr(const tSVect & aVect,const Type & aCste,bool OnlyIfFirstIter=true);
          void SupressAllConstr();
     private :
          cResolSysNonLinear(const tRSNL & ) = delete;

	  ///  Modify equations to take into account var is frozen
	  void  ModifyFrozenVar (tIO_RSNL&);

          /// Add observations as computed by CalcVal
          void   AddObs(const std::vector<tIO_RSNL>&);

	  void InitConstraint() override;
          /** Bases function of calculating derivatives, dont modify the system as is
              to avoid in case  of schur complement , if it is used for linearizeing constraint "ForConstr" the process is slightly diff*/
          void   CalcVal(tCalc *,std::vector<tIO_RSNL>&,const tStdVect & aVTmp,bool WithDer,const tResidualW &,bool ForConstr );

          tDVect     mCurGlobSol;  ///< Curent solution
          tLinearSysSR*    mSysLinear;         ///< Sys to solve equations, equation are concerning the differences with current solution

	  std::vector<Type>    mValueFrozenVar;    ///< indicate for each var the possible value where it is frozen
	  int lastNbObs;                           ///< number of observations of last solving
	  int currNbObs;                           ///< number of observations currently added

          /// handle the linear constraint : fix var, shared var, gauge ...
          cSetLinearConstraint<Type>* mLinearConstr;  

          std::vector<Type>     mVCstrCstePart;    /// Cste part of linear constraint that dont have specific struct (i.e vs Froze/Share)
          std::vector<tSVect>   mVCstrLinearPart;  /// Linerar Part of 
};


/**  class for communinication  input and ouptut of equations in 
   cResolSysNonLinear
 */
template <class Type> class cInputOutputRSNL
{
     public :
          typedef std::vector<Type>  tStdVect;
          typedef std::vector<int>   tVectInd;

	  /// Create Input data w/o temporay
	  cInputOutputRSNL(const tVectInd&,const tStdVect & aVObs);
	  /// Create Input data with temporary temporay
	  // cInputOutputRSNL(const tVectInd&,const tStdVect &aVTmp,const tStdVect & aVObs);

	  /// Give the weight, handle case where mWeights is empty (1.0) or size 1 (considered constant)
	  Type WeightOfKthResisual(int aK) const;
	  /// Check all the size are coherent
	  bool IsOk() const;
	  ///  Real unknowns + Temporary
	  size_t NbUkTot() const;

          tVectInd   mGlobVInd;    ///<  index of unknown in the system + TMP (with -1)   mVIndGlob
          tStdVect   mVObs;     ///< Observation (i.e constants)
          tStdVect                mWeights;  ///< Weights of eq, size can equal mVals or be 1 (cste) or 0 (all 1.0) 
          tStdVect                mVals;     ///< values of fctr, i.e. residuals
          std::vector<tStdVect>   mDers;     ///< derivate of fctr
	  size_t                  mNbTmpUk;

          // use a s converter from tREAL8, "Fake" is used to separate from copy construtcor when Type == tREAL8
	  cInputOutputRSNL(bool Fake,const cInputOutputRSNL<tREAL8> &);
     private :
	  // cInputOutputRSNL(const cInputOutputRSNL<Type> &) = delete;

};

/**  class for grouping all the observation relative to a temporary variable,
     this is necessary because all must be processed simultaneously in schur elimination

     Basically this only a set of "cInputOutputRSNL"
 */
template <class Type> class cSetIORSNL_SameTmp
{
	public :

            friend class cSetIORSNL_SameTmp<tREAL4>;
            friend class cSetIORSNL_SameTmp<tREAL8>;
            friend class cSetIORSNL_SameTmp<tREAL16>;


            typedef cInputOutputRSNL<Type>  tIO_OneEq;
	    typedef std::vector<tIO_OneEq>  tIO_AllEq;
            typedef std::vector<Type>  tStdVect;
            typedef std::vector<int>   tVectInd;

	    /** Constructor :  take value of tmp+ optional vector of fixed var (given wih negatives value like {-1,-2})
	        if aValFix is not given, default is current value
		It can be strange to fix value of tmp var, but it's usefull for example if we want to use the same
		equation with unknown variable and fix value. This is the case in bundle adj when the same colinearity 
		equation can be use with tie-point or with a known GCP.
	     */
	    cSetIORSNL_SameTmp(const tStdVect & aValTmpUk,const tVectInd & aVFix={},const tStdVect & aValFix ={});

            /// Force tmp to its current value,  aNum is the same index (negative) than in AddEq2Subst
            void  AddFixCurVarTmp (int aNum,const Type& aWeight);
            /// Force tmp to a different value, probably not usefull in real (if we have a better value, why not use it), execpt for test
            void  AddFixVarTmp (int aNum,const Type& aVal,const Type& aWeight);

	    /// Add and equation,  check the coherence
	    void AddOneEq(const tIO_OneEq &);
	    ///  Accesor to all equations
	    const tIO_AllEq & AllEq() const;
	    /// To be Ok must have at least 1 eq, and number of eq must be >= to unkwnonw
	    void  AssertOk() const;

	    ///  Number of temporary unkown
	    size_t  NbTmpUk() const;
	    const tStdVect & ValTmpUk() const;
	    Type  Val1TmpUk(int aInd) const; ///make the correction of negativeness

	    static size_t ToIndTmp(int ) ;
	    static bool   IsIndTmp(int ) ;

	    cSetIORSNL_SameTmp(bool Fake,const cSetIORSNL_SameTmp<tREAL8> &) ;

	    int  NbRedundacy() const;
	private :
	    cSetIORSNL_SameTmp(const cSetIORSNL_SameTmp&) = delete;

	    tIO_AllEq          mVEq;
	    tVectInd           mVFix;
	    tStdVect           mValFix;
	    bool               mOk;
	    size_t             mNbTmpUk;
	    tStdVect           mValTmpUk;
	    std::vector<bool>  mVarTmpIsFrozen;       ///< indicate for each temporary var if it is frozen
	    tStdVect           mValueFrozenVarTmp;    ///< value of frozen tmp
	    size_t             mNbEq;
            cSetIntDyn         mSetIndTmpUk;
};


/** Virtual base classe for solving an over resolved system of linear equation;
    Typical derived classes can be :
        * l1 sys using barodale methods
        * least square using covariance system
        * least square not using covariance ...
*/

template <class Type> class cLinearOverCstrSys  : public cMemCheck
{
    public :
       ///  Basic Cstr
       cLinearOverCstrSys(int aNbVar);
       ///  static allocator
       static cLinearOverCstrSys<Type> * AllocSSR(eModeSSR,int aNbVar);

       //  This two method are the public methods , they may add some auxiliary  processing like levenberg markard stuff
       //  before calling the specific "SpecificAddObservation" 
       //
       /// Add  aPds (  aCoeff .X = aRHS) 
       void PublicAddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) ;
       /// Add  aPds (  aCoeff .X = aRHS) , version sparse
       void PublicAddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) ;


       /// Virtual methods => virtaul ~X()
       virtual ~cLinearOverCstrSys();
       

       /**  This the method for adding observation with temporaray unknown, the class can have various answer
	     -  eliminate the temporay via schur complement
             - treat temporary as unknowns and increase the size of their unknowns
	     - refuse to process =>default is error ...
	*/
       virtual void AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>&);

       /// "Purge" all accumulated equations
       virtual void Reset() = 0;
       /// Compute a solution
       virtual cDenseVect<Type>  Solve() = 0;
       ///  May contain a specialization for sparse system, default use generik
       virtual cDenseVect<Type>  SparseSolve() ;

       /** Return for a given "solution" the weighted residual of a given observation
           Typically can be square, abs ....
           Usefull for bench at least (check that solution is minimum, or least < to neighboor)
        */
       
       virtual Type Residual(const cDenseVect<Type> & aVect,const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) const = 0;
       
       //  ============ Fix value of variable =============
            ///  Fix value of curent variable, 1 variable
       virtual void AddObsFixVar(const Type& aWeight,int aIndVal,const Type & aVal);
            ///  Fix value of curent variable, N variable
       virtual void AddObsFixVar(const Type& aWeight,const cSparseVect<Type> & aVVarVals);
            ///  Fix value of curent variable, All variable
       virtual void AddObsFixVar(const Type& aWeight,const cDenseVect<Type>  &  aVRHS); 

       /// Accessor
       int NbVar() const;

      /// Normal Matrix defined 4 now only in cLeasSqtAA, maybe later defined in other classe, else error
      virtual cDenseMatrix<Type>  V_tAA() const;
      /// Idem  "normal" vector
      virtual cDenseVect<Type>    V_tARhs() const;  
      /// Indicate if it gives acces to these normal "stuff"
      virtual bool   Acces2NormalEq() const;  


      virtual void   AddCov(const cDenseMatrix<Type> &,const cDenseVect<Type>& ,const std::vector<int> &aVInd);
      //
       /// Add  aPds (  aCoeff .X = aRHS) 
       virtual void SpecificAddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) = 0;
       /// Add  aPds (  aCoeff .X = aRHS) , version sparse
       virtual void SpecificAddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) = 0;

       Type LVMW(int aK) const;

    protected :
       int mNbVar;
       cDenseVect<Type>  mLVMW;  // The Levenberg markad weigthing
    // private :
};

template <class Type>  cLinearOverCstrSys<Type> *  AllocL1_Barrodale(size_t aNbVar);



/** Class for fine parametrisation  allocation of normal sparse system */

class cParamSparseNormalLstSq
{
      public :
          cParamSparseNormalLstSq();
          cParamSparseNormalLstSq(double aPerEmptyBuf,size_t aNbMaxRangeDense,size_t aNbBufDense);

	  /// Def=4,  equation are buffered "as is" and at some frequency put in normal matrix
	  double mPerEmptyBuf;
	  /** Def={} , even with sparse system, it can happen that a small subset of variable are better handled as dense one, 
	    typically it could be the intrinsic parameters in bundle */
	  std::vector<size_t> mVecIndDense;

	  /** Def=0 ; it is recommandend that mIndDense correpond to low index, in the case where they are in fact range [0,N]
	       mIndMaxRangeDense allow an easier parametrization 
	   */
	  size_t mIndMaxRangeDense;

	  /** Def=13 the systeme can maintain a certain number of non dense variable in temporary  dense mode, def is purely arbitrary ...*/
	  size_t mNbBufDense;
};

/** Not sure the many thing common to least square system, at least there is the
    residual (sum of square ...)
*/
template <class Type> class  cLeasSq  :  public cLinearOverCstrSys<Type>
{
    public :
       cLeasSq(int aNbVar);
       Type Residual(const cDenseVect<Type> & aVect,const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) const override;
       
       /// Dont use normal equation 
       static cLeasSq<Type>*  AllocSparseGCLstSq(int aNbVar);
       
       /// Adpated to"very sparse" system like in finite element, probably also ok for photogrammetry
       static cLeasSq<Type>*  AllocSparseNormalLstSq(int aNbVar,const cParamSparseNormalLstSq & aParam);
       
       /// Basic dense systems  => cLeasSqtAA
       static cLeasSq<Type>*  AllocDenseLstSq(int aNbVar);


};

/**  Implemant least by suming tA A ,  simple and efficient, by the way known to have
  a conditionning problem 
*/

template <class Type> class  cLeasSqtAA  :  public cLeasSq<Type>
{
    public :
       cLeasSqtAA(int aNbVar);
       cLeasSqtAA<Type> Dup() const;

       virtual ~cLeasSqtAA();
       void Reset() override;
       /// Compute a solution
       cDenseVect<Type>  Solve() override;
       /// Use  sparse cholesky , usefull for "sparse dense" system ...
       cDenseVect<Type>  SparseSolve() override ;

       void AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>&) override;

       //  ================  Accessor used in Schur elim ========  :
       
       const cDenseMatrix<Type> & tAA   () const;   ///< Accessor 
       const cDenseVect<Type>   & tARhs () const;   ///< Accessor 
       cDenseMatrix<Type> & tAA   () ;         ///< Accessor 
       cDenseVect<Type>   & tARhs () ;         ///< Accessor 

      /// access to tAA via virtual interface
      cDenseMatrix<Type>  V_tAA() const override;
      /// access to tARhs via virtual interface
      cDenseVect<Type>    V_tARhs() const override;  
      /// true because acces is given
      bool   Acces2NormalEq() const override;  

      void   AddCov(const cDenseMatrix<Type> &,const cDenseVect<Type>& ,const std::vector<int> &aVInd) override;
    private :
       void SpecificAddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) override;
       void SpecificAddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;

       cDenseMatrix<Type>  mtAA;    /// Som(W tA A)
       cDenseVect<Type>    mtARhs;  /// Som(W tA Rhs)
       cBufSchurSubst<Type> * mBSC;
      
};


/**  Class for compute elimination of temporay equation using schur complement.

     For vector and matrix, We use notation of MMV1 documentation   L= Lamda

     (L      B   .. )     (X)   (A )
     (tB     M11 .. )  *  (Y) = (C1)   =>     (M11- tB L-1 B    ...)  (Y) = C1 - tB L-1 A
       ...                (Z)    ..           (                 ...)  (Z)
*/


template <class Type> class  cBufSchurSubst
{
     public :
          typedef cSetIORSNL_SameTmp<Type>  tSetEq;

          /// constructor , just alloc the vector to compute subset of unknosn
          cBufSchurSubst(size_t aNbVar);
          /// Make the computation from the set of equations
          void CompileSubst(const tSetEq &);

          //  ==== 3 accessors to the result

          /// return normal matrix after schur substitution ie : M11- tB L-1 B
          const cDenseMatrix<Type> & tAASubst() const;
          ///  return normal vector after schur subst  ie : C1 - tB L-1 A
          const cDenseVect<Type> & tARhsSubst() const;
          ///  return list of indexes  used to put compress mat/vect in "big" matrixes/vect
          const std::vector<size_t> & VIndexUsed() const;
     private :

          size_t            mNbVar;
          cSetIntDyn        mSetInd;
          cLeasSqtAA<Type>  mSysRed;
          cSparseVect<Type> mSV;
          size_t            mNbTmp;
          size_t            mNbUk;
          size_t            mNbUkTot;

          cDenseMatrix<Type> mL;
          // cDenseMatrix<Type> mLInv;
          cDenseMatrix<Type> mtB;
          cDenseMatrix<Type> mtB_LInv;
          cDenseMatrix<Type> mLInv_B;
          cDenseMatrix<Type> mB;
          cDenseMatrix<Type> mtB_LInv_B;
          cDenseMatrix<Type> mM11;

          cDenseVect<Type>   mA;
          cDenseVect<Type>   mC1;
          cDenseVect<Type>   mtB_LInv_A;
};


    /*  =======================   classes used for facilitating numerotation of unkwons ================= 
     *
     *   cObjWithUnkowns     ->  base-class for object having  unknows 
     *
     *   cOneInteralvUnkown  ->  an interval of Type*  , simply a pair  Type* - int
     *   cSetIntervUK_OneObj -> all the interval of an object  + the object itself
     *   cSetInterUK_MultipeObj -> all the object interacting in one system
     *
     * =================================================================================================== */

template <class Type> class cOneInteralvUnkown;
template <class Type> class cSetIntervUK_OneObj;
template <class Type> class cSetInterUK_MultipeObj;
template <class Type> class cObjWithUnkowns;

/*  Typical scenario
 
     //  for object having unknowns, make them inherit of cObjWithUnkowns, describe behaviour with P-utUknowsInSetInterval
     class  cObj: public cObjWithUnkowns
     {
	  double mUK1[4];  // first set of unknowns
	  int    mToto;
	  double mUK2[7];  // secon set unk
       
	   ...  do stuff specific to cObj ...

          void P-utUknowsInSetInterval() override 
	  {
	       mSetInterv->AddOneInterv(mUK1,4);
	       mSetInterv->AddOneInterv(mUK2,7);
	  }
     };

     {
        cObj aO1,aO2;
        cSetInterUK_MultipeObj<Type>  aSet;    //  create the object
        aSet.AddOneObj(aO1); // in this call aSet will call O1->P-utUknowsInSetInterval()
        aSet.AddOneObj(aO2);

	// create a sys with the vector of all unkwnon
	cResolSysNonLinear<double> * aSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqDense,mSetInterv.GetVUnKnowns());


	const auto & aVectSol = mSys->SolveUpdateReset();
	// modify all unkowns with new solution, call the method OnUpdate in case object have something to do
        mSetInterv.SetVUnKnowns(aVectSol);
     }

*/



///  represent one interval of consecutive unkown
template <class Type> class cOneInteralvUnkown
{
     public :
        Type * mVUk;
        size_t mNb;
        cOneInteralvUnkown(Type * aVUk,size_t aNb)  : mVUk (aVUk) , mNb (aNb) {}
};

///  represent all the unkown interval of one object
template <class Type> class cSetIntervUK_OneObj
{
     public :
         cSetIntervUK_OneObj(cObjWithUnkowns<Type>   *anObj) : mObj (anObj) {}

         cObjWithUnkowns<Type>   *         mObj;
         std::vector<cOneInteralvUnkown<Type>>   mVInterv;

};

///  represent all the object with unknown of a given system of equation

template <class Type> class cSetInterUK_MultipeObj
{
        public :
           friend class cObjWithUnkowns<Type>;

           cSetInterUK_MultipeObj(); /// constructor, init mNbUk
           ~cSetInterUK_MultipeObj();  /// indicate to all object that they are no longer active

	   /// This method is used to add the unknowns of one object
           void  AddOneObj(cObjWithUnkowns<Type> *);
	   /// Test if object already added to avoid error
           void  AddOneObjIfRequired(cObjWithUnkowns<Type> *);

	   ///  return a DenseVect filled with all unknowns  as expected to create a cResolSysNonLinear
           cDenseVect<Type>  GetVUnKnowns() const;

	   ///  fills all unknown of object with a vector as created by cResolSysNonLinear::SolveUpdateReset()
           void  SetVUnKnowns(const cDenseVect<Type> &);

	        // different method for adding intervalls

           void AddOneInterv(Type * anAdr,size_t aSz) ; ///<  generall method
           void AddOneInterv(Type & anAdr) ;            ///<  call with a single value
           void AddOneInterv(std::vector<Type> & aV) ;  ///<  call previous with a vector
           void AddOneInterv(cPtxd<Type,2> &);          ///<  call previous wih a point
           void AddOneInterv(cPtxd<Type,3> &);          ///<  call previous wih a point

	   void SIUK_Reset();  /// Maybe private later, now used for tricky destruction order
        private :

	   size_t IndOfVal(const cObjWithUnkowns<Type>&,const Type *) const;

           cSetInterUK_MultipeObj(const cSetInterUK_MultipeObj<Type> &) = delete;

           void IO_UnKnowns(cDenseVect<Type> & aV,bool isSetUK);

           std::vector<cSetIntervUK_OneObj<Type> >  mVVInterv;
           size_t                                    mNbUk;
};

/** Some object can be made of several object with uknowns, like PC Cam that are made of Pose+IntrCal */

/*
template <class Type> class cObjOfMultipleObjUk
{
     public :
        typedef cObjWithUnkowns<Type> * tPtrOUK;

        virtual  std::vector<tPtrOUK>  GetAllUK() =0;
};
*/

template <class Type> class cGetAdrInfoParam
{
    public :
	typedef cObjWithUnkowns<Type> tObjWUK;

        //  cGetAdrInfoParam(const std::string & aPattern);
	cGetAdrInfoParam(const std::string & aPattern,tObjWUK & aObj);

	static void PatternSetToVal(const std::string & aPattern,tObjWUK & aObj,const Type & aVal);

        void TestParam(tObjWUK*,Type *,const std::string &);

	const std::vector<Type*>  &      VAdrs()  const;
	const std::vector<std::string> & VNames() const;
	const std::vector<tObjWUK*> &    VObjs() const;

	static void ShowAllParam(tObjWUK &);
     private :

	tNameSelector  mPattern;
	std::vector<tObjWUK*>      mVObjs;
	std::vector<Type*>         mVAdrs;
	std::vector<std::string>   mVNames;
};

template <class Type> class cObjWithUnkowns //  : public cObjOfMultipleObjUk<Type>
{
       public :
          friend class cSetInterUK_MultipeObj<Type>;
          typedef cObjWithUnkowns<Type> * tPtrOUK;

	  /// Un object may contain other object, defautl behavior return the vector containing itself
          virtual std::vector<tPtrOUK>  GetAllUK() ;
	  /// Rare case where there is a chain
          std::vector<tPtrOUK>  RecursGetAllUK() ;
	  /// defautl constructor, put non init in all vars
          cObjWithUnkowns();
	  ///  check that object is no longer referenced when destroyd
          virtual ~cObjWithUnkowns();
	  
          /// Fundamental methos :  the object put it sets on unknowns intervals  in the glob struct
          virtual void PutUknowsInSetInterval() = 0;

	  ///  Default generate error 4 now
	  virtual  void  GetAdrInfoParam(cGetAdrInfoParam<Type> &);


          /// This callbak method is called after update, used when modification of linear var is not enough (see cSensorCamPC)
          virtual void OnUpdate();

	  ///  Push in vector all the number of unknowns
          void PushIndexes(std::vector<int> &) const;
	  ///  Push in vector a single value    
          void PushIndexes(std::vector<int> &,const Type &) const;
	  ///  Push in vector aNbVal single value    
          void PushIndexes(std::vector<int> &,const Type *,size_t aNbVal) const;
	  ///  Push in vector the index of 3 coords
          void PushIndexes(std::vector<int> &,const cPtxd<Type,3> & ) const;

	  ///  indicate if the object has been initialized
          bool  UkIsInit() const;

	  /// recover the index from a value
	  size_t IndOfVal(const Type *) const;

          int   IndUk0() const;   ///< Accessor
          int   IndUk1() const;   ///< Accessor

	  // void GetAllValues(std::vector<Type> & aVRes);

       protected :
	  /// defautl constructor, put non init in all vars
          void OUK_Reset();
          cObjWithUnkowns(const cObjWithUnkowns<Type> &) = delete;
          void operator = (const cObjWithUnkowns<Type> &) = delete;


          cSetInterUK_MultipeObj<Type> *  mSetInterv;
          int   mNumObj;
          int   mIndUk0;
          int   mIndUk1;
};

template <class T1,class T2> void ConvertVWD(cInputOutputRSNL<T1> & aIO1 , const cInputOutputRSNL<T2> & aIO2);

/**   Class for representing a Pt of R3 in bundle adj, when it is considered as
 *   unknown.
 *      +  we have the exact value and uncertainty of the point is covariance is used
 *      -  it add (potentially many)  unknowns and then  it take more place in  memory & time
 */

template <const int Dim>  class cPtxdr_UK :  public cObjWithUnkowns<tREAL8>,
                                             public cMemCheck
{
   public :
      typedef cPtxd<tREAL8,Dim>  tPt;

      cPtxdr_UK(const tPt &);
      ~cPtxdr_UK();
      void PutUknowsInSetInterval() override;
      const tPt & Pt() const ;
   private :
      cPtxdr_UK(const cPtxdr_UK&) = delete;
      tPt mPt;
};

typedef cPtxdr_UK<2> cPt2dr_UK ;
typedef cPtxdr_UK<3> cPt3dr_UK ;




};

#endif  //  _MMVII_SysSurR_H_
