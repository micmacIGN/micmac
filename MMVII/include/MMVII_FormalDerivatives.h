#ifndef _MMVII_FormalDerivative_H_
#define _MMVII_FormalDerivative_H_

/** \file MMVII_FormalDerivate.h
    \brief File for generating formal derivate

   Classes for generated formal derivative.  All classes are single template classes. 
   The template parameter indicate the numerical type used for storage/computation 
   (float, double ...)

   This file is the only file to include. It contains :

      * declaration of operators
      * definition of "main" classes : cFormula , cCoordinatorF ,  cImplemF "  ;  
      * the 3 class for Atomic formula who will (probably) stay the same : Unkown, Observation, Constants

   This file include 2 files corresponding to type of formula :

      * classes for "unary" formulas in  "MMVII_FormDer_UnaryOp.h"
      * classes for "binary" formulas in "MMVII_FormDer_BinaryOp.h"

   These 2 files have "vocation" to be extended during the future.

             -------------------------------------------------

    *  cFormula<Type> : represent a mathematicall formula; as in math :
          - if F is a formula, exp(F), log(F) ....are formulas
          - if F1 and F2 are formulas, F1+F2 , F1*F2 ... are formulas
          - there exist some atomic formulas like constants, unknown and observations
          - if F is a formula  F->Derivate(k) is a formula corresponding to is derivate dF/dXk
        Formulas are a complete algebric type.


     * cCoordinatorF<Type> : is the "coordinator" class.
       This class has, between others, the responsability of :
           - creating the initial atomic formula corresponding  to unknowns and observation
           - maintain an inventory of existing formulas for efficiency purpose
        
     *  Using this library is mainly :
          - create a coordinator with a given number of unkown and observations
          - create a formula using atoms an operator, generally the user function creating a
            formula will be a template that can operate on any complete algebric type 
            (double, float, Formula , jets ...)
          - indicate to the coordinator the formula you want work on, with generally its derivate
          - evaluate the values of the formula for  given unknows and observations
             
       cFormula<Type> is no more than an encapsulation of the "concrete" class 

           
   
*/


#define WITH_MMVII true

#if (WITH_MMVII)
#include "include/MMVII_all.h"
#include "include/MMVII_Derivatives.h"
using namespace MMVII;
#else             //========================================================== WITH_MMVI
class cMemCheck
{
};
#include <memory>
#include <map>
#include <iostream> 

#endif            //========================================================== WITH_MMVI


namespace  NS_MMVII_FormalDerivative
{

/* *************************************************** */
/*                                                     */
/*  P0-Definition of global functions                  */
/*                                                     */
/* *************************************************** */

/*

*/

/// This function is required if we want to have same operation on numbers double and formulas
double square(const double & aV)  {return aV*aV;}
// static double square(const float & aV)  {return aV*aV;}

void Error(const std::string & aMes,const std::string & aExplanation)
{
    std::cout << "In MMVII_FormalDerivative a fatal error" << "\n";
    std::cout << "  Likely Source   ["<< aExplanation << "\n";
    std::cout << "  Message  ["<< aMes << "]\n";
    assert(false);
}

void InternalError(const std::string & aMes)
{
   Error(aMes,"Internal Error of the Library");
}
void UserSError(const std::string & aMes)
{
   Error(aMes,"Probable error on user's side due to unapropriate usage of the library");
}


/** This function computes derivates by finites difference
    It is used in the tests to check correction of  formal derivatives. Also used
    in didactic parts.
*/

template <class Type,class TypeFct>
std::vector<Type> NumericalDerivate
                  (
                      TypeFct                 & aFctr,  ///< Function
                      const std::vector<Type> & aVUk,   ///< Unknown
                      const std::vector<Type> & aVObs,  ///< Observations
                      int aNumVar,                      ///< Num of unknown we derivate by
                      const Type & aEpsilon             ///< "Small" number to compute variations
                  )

{

     std::vector<Type> aVPlus = aVUk;
     aVPlus.at(aNumVar) += aEpsilon;
     std::vector<Type> aResPlus = aFctr( aVPlus,aVObs);

     std::vector<Type> aVMinus = aVUk;
     aVMinus.at(aNumVar) -= aEpsilon;
     std::vector<Type> aResMinus = aFctr( aVMinus,aVObs);

     std::vector<Type> aDerivate;
     for (size_t aK=0 ; aK<aResPlus.size() ; aK++)
        aDerivate.push_back((aResPlus.at(aK)-aResMinus.at(aK)) / (2*aEpsilon));

     return aDerivate;
}

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *         Main user interace                      * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

   // ------------- The two classes visible by user are cFormula  and cCoordinatorF  ------

/** Abstraction of mathemicall formula, this the object manipulated by user, its
has all algerbric operation required. This object is just an encapsulation of 
a pointer on cImplemF.
*/
template <class TypeElem> class cFormula ; 

/** Class for managing the "context", i.e. coordinating all the  formula 
    and their derivative corresponding to a single use .
*/
template <class TypeElem> class cCoordinatorF;  




    // -------- Declaration all binary operators  ----------------
      
          // For each operator with have the 3 versions  "Formula x Formula"  ,
          //  "Number x Formula"    and "Formula x Number"

      //  Operator +
template <class TypeElem> cFormula <TypeElem>  
         operator +(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         operator +(const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         operator +(const cFormula <TypeElem> & aF1,const TypeElem & aV2);
      //  Operator *
template <class TypeElem> cFormula <TypeElem>  
         operator *(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         operator *(const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         operator *(const cFormula <TypeElem> & aF1,const TypeElem & aV2);
      //  Operator -
template <class TypeElem> cFormula <TypeElem>  
         operator -(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         operator -(const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         operator -(const cFormula <TypeElem> & aF1,const TypeElem & aV2);
      //  Operator /
template <class TypeElem> cFormula <TypeElem>  
         operator /(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         operator /(const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         operator /(const cFormula <TypeElem> & aF1,const TypeElem & aV2);
      //  pow
template <class TypeElem> cFormula <TypeElem>  
         pow (const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         pow (const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  
         pow (const cFormula <TypeElem> & aF1,const TypeElem & aV2);


    // -------- Declare all unary operators  ----------------
template <class TypeElem> cFormula <TypeElem>  
         square(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  
         exp(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  
         operator - (const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  
         log(const cFormula <TypeElem> & aF);

    // -------- Declaration of Coordinator class  ----------------

template <class TypeElem> class cCoordinatorF : public cMemCheck
{
    public :
 
        typedef cFormula <TypeElem>       tFormula;
        typedef std::vector<TypeElem>     tOneRes;

      // ---------------------------  Constructors / Destructor -------------------
        /// Constructor with explicit Id for Unknown/Observation. Used if we want to analyze the generated code
        inline cCoordinatorF(int SzBuf,const std::vector<std::string> & aVecUK,const std::vector<std::string> & aVecObs);
        /// Constructor with basic Id (used if we dont generate code, or dont want to analyse it by human)
        inline cCoordinatorF(int SzBuf,int aNbUnknown,int aNbObservation);
        /// Destructeur will free allocated formulas
        ~cCoordinatorF();
        /// Copies are not allowed on this kind of object.
        cCoordinatorF(const cCoordinatorF<TypeElem> &) = delete;  

      // ---------------------------  Accessors to Atomic Formulas -------------------
        const std::vector<cFormula <TypeElem> >& VUk()  const {return  mVFormUnknowns;}  ///< Unknowns
        const std::vector<cFormula <TypeElem> >& VObs() const {return  mVFormObservations;}      ///< Observations

      // ---------------------------  Manipulation  -------------------

         ///  Set the formulas that with be used for computation
         void  SetCurFormulas(const std::vector<tFormula> &);

         /**  SetCurFormulas + all its derivative , order of storage will be 
              VF0 dVF0/dX0  dVF0/dX1 .... VF1 dVF1/dX0 ... */
         void  SetCurFormulasWithDerivative(const std::vector<tFormula> & aVF);

         ///  Add a new set of vals (unknown + Obs) inside de evaluation "queue"
         void  PushNewEvals(const std::vector<TypeElem> & aVUK,const std::vector<TypeElem> & aVObs);
               
         bool BufIsFull() const {return mNbInBuf == mSzBuf;} ///< Can we push more value ?
         size_t    SzBuf() const  {return mSzBuf;}  ///< Number of value we can add

         /** Make the evaluation of current functions on pushe valuse . Let V be the result
             (*V[0]) is  the  vector of containing current formula for first SetCur
             !! => Warn the same memory space is recycled ...
         */
         const std::vector<tOneRes *> & EvalAndClear();
         /// Retur value computed taking into account order of storage
         const TypeElem &  ValComp(int aNumPush,int aKElem)
         {
             return  mBufRes.at(aNumPush)->at(mSzInterval*aKElem);
         }
         /// Retur value of derivate computed taking into account order of storage
         const TypeElem &  DerComp(int aNumPush,int aKElem,int aKVarDer)
         {
             if (! mWithDer)  UserSError("Acces to derivate wich were not computed");
             return  mBufRes.at(aNumPush)->at(mSzInterval*aKElem +1 + aKVarDer);
         }
        

    private :  // END-USER
         /*   =================================================================================
             ABOVE WAS THE REAL PUBLIC PART OF cCoordinatorF FOR USER OF LIBRARY.  THE REST
            IS PUBLIC FOR IMPLEMENTERS  BUT NOT NEEDED BY USER
         =====================================================================================*/
    public :

      // Result of several evaluation are stored in a buffer, Eigen vector are used 
      // as they implement efficiently arithmeticall operation
        // typedef Eigen::Array<TypeElem, 1, Eigen::Dynamic>  tBuf;
        typedef std::vector<TypeElem>  tBuf;


      // ---------------------------  Acces to function from names, values -------------------
        /// Indicate if the formula corresponding to a given string allreay exist
        inline bool  ExistFunc(const std::string & aName) const 
        {
              return (mDicoFunc.find(aName) != mDicoFunc.end());
        }
        /// Func of given name, Error if don't exist
        inline tFormula FuncOfName(const std::string & aName) const ;
        /// Add a function (put it in dico), Error if already exist
        inline void AddFormula(tFormula aPF)
        {
           if (ExistFunc(aPF->Name())) InternalError ("Multiple add of identic name :[" + aPF->Name() + "]");
           mDicoFunc[aPF->Name()] = aPF;
           mVAllFormula.push_back(aPF);
        }

        /// Func of given constant, create if don't exist
        inline tFormula CsteOfVal(const TypeElem & aCste) ;
        tFormula  Cste0() const  {return mCste0;}  ///< Acces to a current constant
        tFormula  Cste1() const  {return mCste1;}  ///< Another Acces to a current constant
        tFormula  Cste2() const  {return mCste2;}  ///< Yet another Acces to a current constant
        /// Tuning ---  Print the stack of function as a tree 
        inline void ShowStackFunc() const;


        size_t      NbCurFonc() const {return mVAllFormula.size();}
    private :

         /// Called by PushNewEvals to Set Unknown/Observations
         void  SetNewVals(std::vector<tFormula> & aVF,const std::string & aMes,const std::vector<TypeElem> & aVVals);

        /// Used to generate automatically Id for Unknown/Observatio, when we dont need to control them explicitely
        static std::vector<std::string>   MakeAutomId(const std::string & aPrefix,int aNb);

        size_t                         mSzBuf;       ///< Capacity of bufferirsation
        size_t                         mNbCste;      ///< Number Cste
        size_t                         mNbUK;        ///< Dim=number of unkown
        size_t                         mNbObs;       ///< Number of obserbation variable
        std::vector<tFormula>          mVFormUnknowns; ///< Vector of All Unknowns
        std::vector<tFormula>          mVFormObservations; ///< Vector of All Observations
        std::map<std::string,tFormula> mDicoFunc;    ///< Map Name => Func
        std::vector<tFormula>          mVAllFormula; ///< Vector of All Func, allow to parse them in creation order
        std::map<TypeElem,tFormula>    mDicoCste;    ///< Map  Value => Func Constant
        tFormula                       mCste0;       ///< Fonc constant null
        tFormula                       mCste1;       ///< Fonc constant 1
        tFormula                       mCste2;       ///< Fonc constant 1
        size_t                         mNbInBuf;     ///< Number of Unknown/Obs vect currenlty loaded in buf
        bool                           mWithDer;     ///< Done With Derivate
        int                            mSzInterval;  ///< Size between two val, depends if computation done with deriv
        std::vector<tFormula>          mVCurF;       ///< Current evaluted formulas
        std::vector<tFormula>          mVReachedF;   ///< Formula "reachable" i.e. necessary to comput mVCurF
        std::vector<tOneRes>           mBufLineRes;  ///< Reserve memory for each line
        std::vector<tOneRes*>          mBufRes;      ///< Reserve memory for result itself

};

/* ************************************************** 
 *                                                  * 
 *     Pre-Declaration of all classes               * 
 *   Not required  by compilation                   *
 *   (Except for cImplemF )but I like to have       * 
 *  a quick view of all existing classes            * 
 *                                                  * 
 * **************************************************/

/**   "Mother" Interface  class of all classes implementing the service , 
       abstract class with pure virtual method
*/

template <class TypeElem> class cImplemF  ;  

          // ---------------  "Atomic" function : Unknown, constant, observation-----------------
template <class TypeElem> class cAtomicF ;   ///< Mother Class of all atomic formulas 
              /// "Observations" corresponding to user constant (change for each evaluation)
template <class TypeElem> class cObservationF ;   
              /// "Constant" function
template <class TypeElem> class cConstantF ;   
              /// "Unknown"  for representing coordinates function X0,X1,X2 ....
template <class TypeElem> class cUnknownF; 

          // -----------------------------  Unary operator ------------------------------------
template <class TypeElem> class cUnaryF ;    ///< Mother Class of all unary operator
template <class TypeElem> class cSquareF ;   ///< Class for square operator
template <class TypeElem> class cExpF ;      ///< Class for exponential operator
template <class TypeElem> class cMin1F ;     ///< Class for Unary Minus
template <class TypeElem> class cLogF ;       ///< Class for neperien log

          // --------------------------------  Binary operator -------------------------------------
template <class TypeElem> class cBinaryF ;  ///< Mother class of binary operators
template <class TypeElem> class cSumF ;     ///< Class for sum of 2 functions
template <class TypeElem> class cMulF ;     ///< Class for multiplication of 2 functions
template <class TypeElem> class cSubF ;     ///< Class for substraction of 2 functions
template <class TypeElem> class cDivF ;     ///< Class for division of 2 functions
template <class TypeElem> class cPowF ;     ///< Class for division of 2 functions


/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        Definition of all classes                * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

            //  -------------------  2 "Main" Classes  -------------------------
            //      cFormula  / cImplemF 
            // ----------------------------------------------------------------

template <class TypeElem> class cImplemF  : public cMemCheck
{
    public :
      // See eigen documentation,  this macro is mandatory for alignment reason
       EIGEN_MAKE_ALIGNED_OPERATOR_NEW

       typedef TypeElem                        tElem;
       typedef cCoordinatorF<TypeElem>         tCoordF;
       typedef typename tCoordF::tBuf          tBuf;
       typedef typename tCoordF::tFormula      tFormula;

       //----------- For derivation and reduction--------------
       virtual bool  IsCste0() const {return false;} ///< To redefine in constant func, Used for simplification in "/ * + -"
       virtual bool  IsCste1() const {return false;} ///< To redefine in constant func, Used for simplification in "/ *"
       virtual tFormula Derivate(int aK) const  = 0;  ///< Compute the formula of it's derivative to Kth unknown

       // --------------  For Computation -------------------------
       /// Method that wil compute data inside  mBuf
       virtual void ComputeBuf(int aK0,int aK1) =0;

       /// Return "Sub"-formula referenced
       virtual std::vector<tFormula> Ref() const =0;

     // ---------- Accessors ---------------
       const std::string  & Name() const {return mName;}  ///< Standard accessor
       tCoordF *  CoordF() const {return mCoordF;}   ///< Standard accesor
       int  NumGlob() const {return mNumGlob;}  ///< Standard accessor
     // ---------- Acces to Buf data ---------------
       void SetBuf(size_t anIndex,const TypeElem & aVal) {mBuf.at(anIndex) = aVal;}
       const TypeElem &  GetBuf(size_t anIndex) {return mBuf.at(anIndex);}
       TypeElem * DataBuf() {return mDataBuf;}
     // ---------- Reached Flag ---------------
       bool Reached() const {return mReached;}  ///< Standard accessor
       void SetReached(bool IsReached) {mReached = IsReached;}  ///< Fix Reached
       /// Compute in the reference graphe and put formal explored in VReached
       void CalcRecursiveDepth(std::vector<tFormula> & VReached) ;
       int Depth() const {return mDepth;}  ///< Standard accessor
       void SetDepth(bool aDepth) {mDepth = aDepth;}  ///< Fix Reached


     // ---------- Tuning / Debugging / Analysing ---------------
       /// Used to print constant from generic formula
       virtual const TypeElem * ValCste() const  {return nullptr;}
       /// Infixed "Pretty" Print .  For tuning and checking (i.e correction of reduction, derivative, rewrite ...)
       virtual std::string  InfixPPrint() const =0; 
       /// Number of reference that would occur without reduction on identic formula (to test performance in paper)
       int RecursiveRec() const;

      // --------------------  Destructor / Constructor  --------------------------
       virtual ~cImplemF () {}   ///< Add a virtual ~X() when we have virtual methods, who knows ...
    protected :
       inline cImplemF (tCoordF * aCoordF,const std::string & aName) :
              mCoordF (aCoordF),
              mBuf     (mCoordF->SzBuf(),TypeElem(0.0)),
              mDataBuf (mBuf.data()),
              mName    (aName),
              mNumGlob (mCoordF->NbCurFonc()),
              mReached (false),
              mDepth   (-1)
       { 
       }

       tCoordF *             mCoordF;   ///< Coordinator that manage all the funcion cooperating
       tBuf                   mBuf;       ///< Buf to store values
       TypeElem *             mDataBuf;   ///< Raw pointer
       const std::string      mName;      ///< string represention of the formula as for ex : C2, X1, V0 , square F3,  F18/F3 ...
       int                    mNumGlob;   ///< Global number (!= Num in class)
       bool                   mReached;   ///< Flag to know if a formula is usefull for compute current
       int                    mDepth;     ///< Used for topological sort
    private  :

       cImplemF (const cImplemF<TypeElem>  &) = delete; ///< No Copy
};

template <class TypeElem> class cFormula 
{
    public :
       typedef cCoordinatorF<TypeElem>   tCoordF;
       typedef cImplemF<TypeElem>    tImplemF;
       typedef typename tCoordF::tFormula    tFormula;

       //  -------------------- constructor -------------------
             /// Construct from a pointer, standard 
       cFormula  (tImplemF  * aRawPtr) :
           mPtr  (aRawPtr)
       {
       }
             /// Default constructor, required by some code (vector ?)
       cFormula  ():
           cFormula  <TypeElem> (nullptr)
       {
       }
       // --------------- operator on pointer ---------------------

       // UNUSED 4 NOW  tImplemF & operator*() const  {return *mPtr;}  ///< Standard behaviour of a pointer
       tImplemF * operator->() const {return mPtr;}  ///< Standard behaviour of a pointer
       // DO NOT WORK  const std::unique_ptr<tImplemF>  operator->() const {return std::unique_ptr<mPtr>;} 
       bool IsNull() const {return mPtr==nullptr;} ///< Safer than giving acces to raw pointer

       // --------------- Naming  ---------------------

           /// Generate the unique indentifier of a binary expression
       std::string NameFormulaBin(const std::string & aNameOper,const tFormula & aF2) const
       {
           return "F"+ std::to_string((*this)->NumGlob()) + aNameOper + "F"  + std::to_string(aF2->NumGlob());
       }

           /// Generate the unique indentifier of a unary expression
       std::string NameFormulaUn(const std::string & aNameOper) const
       {
           return  aNameOper + " F"  + std::to_string((*this)->NumGlob());
       }
       /// To allow destruction without giving access to raw pointer
       void FreeMem() {delete mPtr; mPtr=nullptr;}

    private :
       tImplemF* mPtr;  ///< Faster than shared and deallocation is easy as object controlled by context
};

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        ATOMIC FORMULA                           * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

            /*  ----------------------------------------------------------
               Class for atomic formula
                  MOTHER CLASS : cAtomicF
                  DERIVED :   cUnknownF / cObservationF / cConstantF
             ----------------------------------------------------------------*/

template <class TypeElem> class cAtomicF : public cImplemF<TypeElem>
{
     public :
            typedef cImplemF<TypeElem>  tImplemF;
            typedef typename tImplemF::tCoordF  tCoordF;
            typedef typename tCoordF::tFormula  tFormula;

            /// Should work always
            std::string  InfixPPrint() const override {return tImplemF::Name();}
            /// Rule deriv=0 , work by default (constant and observations)
            tFormula Derivate(int aK) const override {return tImplemF::mCoordF->Cste0();}

            /// Generally nothing to do in atomic, their buffer has been filled witj adequate values
            void ComputeBuf(int aK0,int aK1) override  { }
            std::vector<tFormula> Ref() const override{return std::vector<tFormula>();}
     protected :
            inline cAtomicF(tCoordF * aCoordF,const std::string& aName) :
                tImplemF       (aCoordF,aName)
            { }
};

template <class TypeElem> class cUnknownF : public cAtomicF<TypeElem>
{
      public :
            typedef cAtomicF<TypeElem>  tAtom;
            typedef typename tAtom::tImplemF     tImplemF;
            typedef typename tImplemF::tCoordF  tCoordF;
            typedef typename tCoordF::tFormula  tFormula;

            std::string  InfixPPrint() const override {return tImplemF::Name();}
            ///  rule :  dXi/dXj = delta(i,j)
            tFormula Derivate(int aK) const override 
            {
                return (aK==mNumUnk) ? tImplemF::mCoordF->Cste1() :  tImplemF::mCoordF->Cste0();
            }

            friend tCoordF;
      private  :
            inline cUnknownF(tCoordF * aCoordF,const std::string& aName,int aNum) :
                tAtom   (aCoordF,aName),
                mNumUnk (aNum)
            { }

            int  mNumUnk; ///< Number of the Unknown; like  : 0 for X0,  1 for X1 ...
};

template <class TypeElem> class cObservationF : public cAtomicF<TypeElem>
{
      public :
            typedef cAtomicF<TypeElem>  tAtom;
            typedef typename tAtom::tImplemF     tImplemF;
            typedef typename tImplemF::tCoordF  tCoordF;
            typedef typename tCoordF::tFormula  tFormula;
            friend tCoordF;

      private  :
            inline cObservationF(tCoordF * aCoordF,const std::string & aName,int aNum) : 
                  tAtom  (aCoordF,aName),
                  mNum   (aNum)
            { }
            int     mNum; ///< Number of the Observation; like  : 0 for V0,  1 for V1 ...
};

template <class TypeElem> class cConstantF : public cAtomicF<TypeElem>
{
      public :
            typedef cAtomicF<TypeElem>  tAtom;
            typedef typename tAtom::tImplemF     tImplemF;
            typedef typename tImplemF::tCoordF  tCoordF;
            typedef typename tCoordF::tFormula  tFormula;
            typedef typename tCoordF::tBuf      tBuf;
            friend tCoordF;

            bool  IsCste0() const override {return mVal==0.0;} ///< Here we know if we are constant 0
            bool  IsCste1() const override {return mVal==1.0;} ///< Here we know if we are constant 1
            const TypeElem * ValCste() const override  {return &mVal;}

      protected  :
            inline cConstantF(tCoordF * aCoordF,const std::string & aName,int aNum,const TypeElem& aVal) : 
               tAtom   (aCoordF,aName),
               mNum    (aNum),
               mVal    (aVal)
            {
               for (auto & aV : tImplemF::mBuf) aV = aVal;  // Initialize buf  with const val
            }
            int     mNum;
            const TypeElem mVal;
};


/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *         cImplemF / cCoordinatorF                    * */
/* *        External Definition of methods           * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

      /* ---------------------- */
      /*        cImplemF        */
      /* ---------------------- */

template <class TypeElem> int cImplemF<TypeElem>::RecursiveRec() const
{
   int aRes = 1;
   for (const auto  & aF : Ref())
   {
      aRes += aF->RecursiveRec();
   }
   return aRes;
}

template <class TypeElem> void cImplemF<TypeElem>::CalcRecursiveDepth(std::vector<tFormula> & aVReached) 
{
   if (mDepth != -1) return; // if we were already here , nothing to do
   for (const auto  & aF : Ref())
   {
      aF->CalcRecursiveDepth(aVReached); // parse sub formula
      mDepth = std::max(mDepth,aF->mDepth); // Memo max depth
   }
   mDepth++; // my depth is 1 + max of depth of my referenced formulas
   aVReached.push_back(tFormula(this));
}
      /* ---------------------- */
      /*       cCoordinatorF        */
      /* ---------------------- */

template <class TypeElem> 
std::vector<std::string> cCoordinatorF<TypeElem>::MakeAutomId(const std::string & aPrefix,int aNb)
{
   std::vector<std::string> aRes;
   for (int aK=0 ; aK<aNb ; aK++)
       aRes.push_back(aPrefix+ std::to_string(aK));
   return aRes;
}

template <class TypeElem> 
cCoordinatorF<TypeElem>::cCoordinatorF
(
       int aSzBuf,
       const std::vector<std::string> & aVNameUK,
       const std::vector<std::string> & aVNameObs
) :
    mSzBuf      (aSzBuf),
    mNbCste     (0),
    mNbUK       (aVNameUK.size()),
    mNbObs      (aVNameObs.size()),
    mCste0      (CsteOfVal(0.0)),
    mCste1      (CsteOfVal(1.0)),
    mCste2      (CsteOfVal(2.0)),
    mNbInBuf    (0),
    mBufLineRes (mSzBuf),
    mBufRes     ()
    
{
    mBufRes.reserve(mSzBuf);
    // Generate all the function corresponding to unknown
    for (size_t aNumUK=0 ; aNumUK<mNbUK ; aNumUK++)
    {
        tFormula aFuncUK(new cUnknownF<TypeElem>(this,aVNameUK[aNumUK],aNumUK));  // Create it
        mVFormUnknowns.push_back(aFuncUK);   // Push it in vector of coordinat func
        AddFormula(aFuncUK);  // Add to all func
    }

    // Generate all the function corresponding to observations
    for (size_t aNumObs=0 ; aNumObs<mNbObs ; aNumObs++)
    {
        tFormula aFuncObs(new cObservationF<TypeElem>(this,aVNameObs[aNumObs],aNumObs));  // Create it
        mVFormObservations.push_back(aFuncObs);   // Push it in vector of coordinat func
        AddFormula(aFuncObs);  // Add to all func
    }
}

template <class TypeElem> 
cCoordinatorF<TypeElem>::cCoordinatorF(int aSzBuf,int aNbUK,int aNbObs) :
         cCoordinatorF<TypeElem>(aSzBuf,MakeAutomId("X",aNbUK),MakeAutomId("V",aNbObs))
{
}

template <class TypeElem> 
cCoordinatorF<TypeElem>::~cCoordinatorF()
{
    for (auto & aForm : mVAllFormula)
    {
       aForm.FreeMem();
    }
}

template <class TypeElem>
cFormula<TypeElem> cCoordinatorF<TypeElem>::CsteOfVal(const TypeElem & aCste) 
{
  tFormula & aRef = mDicoCste[aCste];
  if (aRef.IsNull())  // If it was not existing, the map contain now the def element
  {
     aRef=tFormula(new cConstantF<TypeElem>(this,"C"+std::to_string(mNbCste),mNbCste,aCste));
     mNbCste++;
     AddFormula(aRef);
  }

  return aRef;
}

template <class TypeElem>
cFormula <TypeElem> cCoordinatorF<TypeElem>::FuncOfName(const std::string & aName) const 
{
    const auto & anIt = mDicoFunc.find(aName);
    if (anIt == mDicoFunc.end()) InternalError ("Try to acces non existing name :[" + aName + "]");
    return anIt->second;
}

template <class TypeElem> 
void cCoordinatorF<TypeElem>::SetNewVals
     (
         std::vector<tFormula> & aVF,
         const std::string & aMes,
         const std::vector<TypeElem> & aVVals
     )
{
    if (aVF.size() != aVVals.size())  // Check size are coherents
    {
       UserSError("Bad size in " + aMes);
    }
    for (size_t aK=0 ; aK<aVF.size() ; aK++)  // Init Vals of formulas buffer
    {
        aVF[aK]->SetBuf(mNbInBuf,aVVals[aK]);
    }
}

template <class TypeElem> 
void cCoordinatorF<TypeElem>::PushNewEvals
     (
          const std::vector<TypeElem> & aVUK,
          const std::vector<TypeElem> & aVObs
     )
{

    if (mNbInBuf >= mSzBuf)
    {
       UserSError("Push exceed buffer capacity");
    }
    SetNewVals(mVFormUnknowns,"Unknowns",aVUK);
    SetNewVals(mVFormObservations,"Observations",aVObs);
    mNbInBuf++;
}


template <class TypeElem> 
void    cCoordinatorF<TypeElem>::SetCurFormulasWithDerivative(const std::vector<tFormula> & aVF)
{
   std::vector<tFormula> aVWDer;
   for (const auto & aF : aVF)
   {
       aVWDer.push_back(aF);
       for (size_t aUK=0 ; aUK<mNbUK ; aUK++)
       {
           aVWDer.push_back(aF->Derivate(aUK));
       }
   }
   SetCurFormulas(aVWDer);
   mWithDer    = true;
   mSzInterval = 1+mNbUK;
}

template <class TypeElem> 
void cCoordinatorF<TypeElem>::SetCurFormulas(const std::vector<tFormula> & aVF)
{
    mWithDer=false;
    mSzInterval = 1;
    mVCurF     = aVF;

    // Erase previous
    for (auto & aF : mVReachedF)  
        aF->SetDepth(-1);
    mVReachedF.clear(); 

    // Compute depth for topologicall sort
    for (auto & aF : mVCurF)  
    {
        aF->CalcRecursiveDepth(mVReachedF);
    } 

    // Use depth to have topological sort
    std::sort
    (
        mVReachedF.begin(),
        mVReachedF.end(),
        [](const tFormula & aF1,const tFormula &aF2) {return aF1->Depth() < aF2->Depth();}
    );

    
    // Make Buf of Res to have right size
    for (auto & aLine : mBufLineRes)
    {
        aLine.resize(mVCurF.size());
    }
}

template <class TypeElem> 
const std::vector<std::vector<TypeElem> *> & cCoordinatorF<TypeElem>::EvalAndClear()
{
    // Make the real hard stuff, compute the data, the depedancy ordering should make it coherent
    for (auto & aF : mVReachedF)
    {
       aF->ComputeBuf(0,mNbInBuf);
    }

    mBufRes.clear();
    for (size_t aKLine=0 ; aKLine<mNbInBuf ;  aKLine++)
    {
        std::vector<TypeElem> & aLine  = mBufLineRes[aKLine];
        mBufRes.push_back(&aLine);
        for (size_t aKFunc=0 ; aKFunc< mVCurF.size() ; aKFunc++)
            aLine[aKFunc] = mVCurF[aKFunc]->GetBuf(aKLine);
    }
    
    return mBufRes;
}

template <class TypeElem> 
void cCoordinatorF<TypeElem>::ShowStackFunc() const
{
    for (const auto & aForm : mVAllFormula)
    {
       if (aForm->Depth()==-1)
          std::cout <<   "---" ;
       else 
          std::cout <<  "-" << aForm->Depth() << "-";

       std::cout << " Form[" << aForm->NumGlob() << "] => " << aForm->Name();
       const TypeElem * aPV = aForm->ValCste();
       if (aPV)
           std::cout << " ; Val=" << *aPV;
       std::cout << "\n";
    }

    std::cout << "REACHED ";
    for (const auto & aForm : mVReachedF)
    {
       std::cout <<  aForm->NumGlob() << " ";
    }
    std::cout << "\n";

    std::cout << "CUR ";
    for (const auto & aForm : mVCurF)
    {
       std::cout <<  aForm->NumGlob() << " ";
    }
    std::cout << "\n";
}

}; //   NS_MMVII_FormalDerivative

#include "MMVII_FormDer_UnaryOp.h"
#include "MMVII_FormDer_BinaryOp.h"

/*
https://www.itl.nist.gov/div898/strd/nls/data/ratkowsky3.shtml
http://en.wikipedia.org/wiki/Automatic_differentiation
https://git.irc.umbc.edu/photorig/openMVG/blob/260584fda68dce095e279362efd24a2d7d7cf5d9/src/third_party/ceres-solver/include/ceres/jet.h

https://mc-stan.org/
http://www.met.reading.ac.uk/clouds/adept/array_features.html
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.7749&rep=rep1&type=pdf
http://www.autodiff.org/
*/

#endif // _MMVII_FormalDerivative_H_
