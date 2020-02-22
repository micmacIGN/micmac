#ifndef _MMVII_DynAndStatFormalDerivation_H_
#define _MMVII_DynAndStatFormalDerivation_H_


/** \file TestDynCompDerivatives.cpp
    \brief Proto of generated code

Prototype of generated code, do not know how it will evolve
   
*/

#define WITH_MMVII true

#if (WITH_MMVII)   //=========================================================
#include "include/MMVII_all.h"
#include "include/MMVII_Derivatives.h"
using namespace MMVII;
#else             //==========================================================
class cMemCheck
{
};
#include "ExternalInclude/Eigen/Dense"
#include <memory>
#include <map>
#include <iostream> 

#endif            // WITH_MMVI ==========================================================

namespace  NS_MMVII_FormalDerivative
{
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


/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *         Main user interace                      * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

   // ------------- The two classes visible by user are cFormula  and cContextF  ------

/** Abstraction of mathemicall formula, this the object manipulated by user, its
has all algerbric operation required. This object is just an encapsulation of 
a pointer on cImplemF.
*/
template <class TypeElem,const int TheSzBuf> class cFormula ; 

/** Class for managing the "context", i.e. coordinating all the  formula 
    and their derivative corresponding to a single use .
*/
template <class TypeElem,const int TheSzBuf> class cContextF;  




    // -------- Declaration all binary operators  ----------------
      
          // For each operator with have the 3 versions  "Formula x Formula"  ,
          //  "Number x Formula"    and "Formula x Number"

      //  Operator +
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator +(const cFormula <TypeElem,TheSzBuf> & aF1 ,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator +(const TypeElem & aV1,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator +(const cFormula <TypeElem,TheSzBuf> & aF1,const TypeElem & aV2);
      //  Operator *
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator *(const cFormula <TypeElem,TheSzBuf> & aF1 ,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator *(const TypeElem & aV1,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator *(const cFormula <TypeElem,TheSzBuf> & aF1,const TypeElem & aV2);
      //  Operator -
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator -(const cFormula <TypeElem,TheSzBuf> & aF1 ,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator -(const TypeElem & aV1,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator -(const cFormula <TypeElem,TheSzBuf> & aF1,const TypeElem & aV2);
      //  Operator /
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator /(const cFormula <TypeElem,TheSzBuf> & aF1 ,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator /(const TypeElem & aV1,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator /(const cFormula <TypeElem,TheSzBuf> & aF1,const TypeElem & aV2);
      //  pow
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         pow (const cFormula <TypeElem,TheSzBuf> & aF1 ,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         pow (const TypeElem & aV1,const cFormula <TypeElem,TheSzBuf> & aF2);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         pow (const cFormula <TypeElem,TheSzBuf> & aF1,const TypeElem & aV2);


    // -------- Declare all unary operators  ----------------
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         square(const cFormula <TypeElem,TheSzBuf> & aF);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         exp(const cFormula <TypeElem,TheSzBuf> & aF);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         operator - (const cFormula <TypeElem,TheSzBuf> & aF);
template <class TypeElem,const int TheSzBuf> cFormula <TypeElem,TheSzBuf>  
         log(const cFormula <TypeElem,TheSzBuf> & aF);

    // -------- Declaration of Context class  ----------------

template <class TypeElem,const int TheSzBuf> class cContextF : public cMemCheck
{
    public :
 
        typedef cFormula <TypeElem,TheSzBuf>       tFormula;

      // ---------------------------  Constructors / Destructor -------------------
        /// Constructor with explicit Id for Unknown/Observation. Used if we want to analyze the generated code
        inline cContextF(const std::vector<std::string> & aVecUK,const std::vector<std::string> & aVecObs);
        /// Constructor with basic Id (used if we dont generate code, or dont want to analyse it by human)
        inline cContextF(int aNbUnknown,int aNbObservation);
        /// Destructeur will free allocated formulas
        ~cContextF();
        /// Copies are not allowed
        cContextF(const cContextF<TypeElem,TheSzBuf> &) = delete;  

      // ---------------------------  Accessors -------------------
        const std::vector<cFormula <TypeElem,TheSzBuf> >& VUK()  const {return  mVFormUnknowns;}  ///< Unknowns
        const std::vector<cFormula <TypeElem,TheSzBuf> >& VObs() const {return  mVFormObservations;}      ///< Observations

      // ---------------------------  Manipulation  -------------------

         ///  Add a new set of vals (unknown + Obs) inside de evaluation "queue"
         void  PushNewEvals(const std::vector<TypeElem> & aVUK,const std::vector<TypeElem> & aVObs);
         ///  Set
         void  SetCurFormulas(const std::vector<tFormula> &);
        

    private :  // END-USER
         /*   =================================================================================
             ABOVE WAS THE REAL PUBLIC PART OF cContextF FOR USER OF LIBRARY.  THE REST
            IS PUBLIC FOR IMPLEMENTERS  BUT NOT NEEDED BY USER
         =====================================================================================*/
    public :

      // Result of several evaluation are stored in a buffer, Eigen vector are used 
      // as they implement efficiently arithmeticall operation
        typedef Eigen::Array<TypeElem, 1, TheSzBuf>  tBuf;


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
         void  SetNewEvals(std::vector<tFormula> & aVF,const std::string & aMes,const std::vector<TypeElem> & aVVals);

        /// Used to generate automatically Id for Unknown/Observatio, when we dont need to control them explicitely
        static std::vector<std::string>   MakeAutomId(const std::string & aPrefix,int aNb);

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
        int                            mNbInBuf;     ///< Number of Unknown/Obs vect currenlty loaded in buf
        bool                           mCurFormSet;  ///< Check if formula was set
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

template <class TypeElem,const int TheSzBuf> class cImplemF  ;  

          // ---------------  "Atomic" function : Unknown, constant, observation-----------------
template <class TypeElem,const int TheSzBuf> class cAtomicF ;   ///< Mother Class of all atomic formulas 
              /// "Observations" corresponding to user constant (change for each evaluation)
template <class TypeElem,const int TheSzBuf> class cObservationF ;   
              /// "Constant" function
template <class TypeElem,const int TheSzBuf> class cConstantF ;   
              /// "Unknown"  for representing coordinates function X0,X1,X2 ....
template <class TypeElem,const int TheSzBuf> class cUnknownF; 

          // -----------------------------  Unary operator ------------------------------------
template <class TypeElem,const int TheSzBuf> class cUnaryF ;    ///< Mother Class of all unary operator
template <class TypeElem,const int TheSzBuf> class cSquareF ;   ///< Class for square operator
template <class TypeElem,const int TheSzBuf> class cExpF ;      ///< Class for exponential operator
template <class TypeElem,const int TheSzBuf> class cMin1F ;     ///< Class for Unary Minus
template <class TypeElem,const int TheSzBuf> class cLogF ;       ///< Class for neperien log

          // --------------------------------  Binary operator -------------------------------------
template <class TypeElem,const int TheSzBuf> class cBinaryF ;  ///< Mother class of binary operators
template <class TypeElem,const int TheSzBuf> class cSumF ;     ///< Class for sum of 2 functions
template <class TypeElem,const int TheSzBuf> class cMulF ;     ///< Class for multiplication of 2 functions
template <class TypeElem,const int TheSzBuf> class cSubF ;     ///< Class for substraction of 2 functions
template <class TypeElem,const int TheSzBuf> class cDivF ;     ///< Class for division of 2 functions
template <class TypeElem,const int TheSzBuf> class cPowF ;     ///< Class for division of 2 functions


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


template <class TypeElem,const int TheSzBuf> class cImplemF  : public cMemCheck
{
    public :
      // See eigen documentation,  this macro is mandatory for alignment reason
       EIGEN_MAKE_ALIGNED_OPERATOR_NEW

       typedef cContextF<TypeElem,TheSzBuf>    tContext;
       typedef typename tContext::tBuf         tBuf;
       typedef typename tContext::tFormula     tFormula;

       //----------- For derivation and reduction--------------
       virtual bool  IsCste0() const {return false;} ///< To redefine in constant func, Used for simplification in "/ * + -"
       virtual bool  IsCste1() const {return false;} ///< To redefine in constant func, Used for simplification in "/ *"
       virtual tFormula Derivate(int aK) const  = 0;  ///< Compute the formula of it's derivative to Kth unknown

       // --------------  For Computation -------------------------
       /// Method that wil compute data inside  mBuf
       virtual void ComputeBuf() =0;

       /// Return "Sub"-formula referenced
       virtual std::vector<tFormula> Ref() const =0;

     // ---------- Accessors ---------------
       const std::string  & Name() const {return mName;}  ///< Standard accessor
       tContext *  Context() const {return mContext;}   ///< Standard accesor
       int  NumGlob() const {return mNumGlob;}  ///< Standard accessor
       tBuf &  Buf() {return mBuf;}       ///< Return buffer
     // ---------- Reached Flag ---------------
       bool Reached() const {return mReached;}  ///< Standard accessor
       void SetReached(bool IsReached) const {mReached = IsReached;}  ///< Fix Reached


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
       inline cImplemF (tContext * aContext,const std::string & aName) :
              mBuf     (tBuf::Random()), // Initialize however
              mContext (aContext),
              mName    (aName),
              mNumGlob (mContext->NbCurFonc()),
              mReached (false)
       { }

       tBuf                   mBuf;       ///< Buf to store values
       tContext *             mContext;   ///< Context that manage all the funcion cooperating
       const std::string      mName;      ///< string represention of the formula as for ex : C2, X1, V0 , square F3,  F18/F3 ...
       int                    mNumGlob;   ///< Global number (!= Num in class)
       bool                   mReached;   ///< Flag to know if a formula is usefull for compute current
    private  :
       cImplemF (const cImplemF<TypeElem,TheSzBuf>  &) = delete; ///< No Copy
};

template <class TypeElem,const int TheSzBuf> class cFormula 
{
    public :
       typedef cContextF<TypeElem,TheSzBuf>   tContext;
       typedef cImplemF<TypeElem,TheSzBuf>    tImplemF;
       typedef typename tContext::tFormula    tFormula;

       //  -------------------- constructor -------------------
             /// Construct from a pointer, standard 
       cFormula  (tImplemF  * aRawPtr) :
           mPtr  (aRawPtr)
       {
       }
             /// Default constructor, required by some code (vector ?)
       cFormula  ():
           cFormula  <TypeElem,TheSzBuf> (nullptr)
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

template <class TypeElem,const int TheSzBuf> class cAtomicF : public cImplemF<TypeElem,TheSzBuf>
{
     public :
            typedef cImplemF<TypeElem,TheSzBuf>  tImplemF;
            typedef typename tImplemF::tContext  tContext;
            typedef typename tContext::tFormula  tFormula;

            /// Should work always
            std::string  InfixPPrint() const override {return tImplemF::Name();}
            /// Rule deriv=0 , work by default (constant and observations)
            tFormula Derivate(int aK) const override {return tImplemF::mContext->Cste0();}

            /// Generally nothing to do in atomic, their buffer has been filled witj adequate values
            void ComputeBuf() override  { }
            std::vector<tFormula> Ref() const override{return std::vector<tFormula>();}
     protected :
            inline cAtomicF(tContext * aContext,const std::string& aName) :
                tImplemF       (aContext,aName)
            { }
};

template <class TypeElem,const int TheSzBuf> class cUnknownF : public cAtomicF<TypeElem,TheSzBuf>
{
      public :
            typedef cAtomicF<TypeElem,TheSzBuf>  tAtom;
            typedef typename tAtom::tImplemF     tImplemF;
            typedef typename tImplemF::tContext  tContext;
            typedef typename tContext::tFormula  tFormula;

            std::string  InfixPPrint() const override {return tImplemF::Name();}
            ///  rule :  dXi/dXj = delta(i,j)
            tFormula Derivate(int aK) const override 
            {
                return (aK==mNumUnk) ? tImplemF::mContext->Cste1() :  tImplemF::mContext->Cste0();
            }

            friend tContext;
      private  :
            inline cUnknownF(tContext * aContext,const std::string& aName,int aNum) :
                tAtom   (aContext,aName),
                mNumUnk (aNum)
            { }

            int  mNumUnk; ///< Number of the Unknown; like  : 0 for X0,  1 for X1 ...
};

template <class TypeElem,const int TheSzBuf> class cObservationF : public cAtomicF<TypeElem,TheSzBuf>
{
      public :
            typedef cAtomicF<TypeElem,TheSzBuf>  tAtom;
            typedef typename tAtom::tImplemF     tImplemF;
            typedef typename tImplemF::tContext  tContext;
            typedef typename tContext::tFormula  tFormula;
            friend tContext;

      private  :
            inline cObservationF(tContext * aContext,const std::string & aName,int aNum) : 
                  tAtom  (aContext,aName),
                  mNum   (aNum)
            { }
            int     mNum; ///< Number of the Observation; like  : 0 for V0,  1 for V1 ...
};

template <class TypeElem,const int TheSzBuf> class cConstantF : public cAtomicF<TypeElem,TheSzBuf>
{
      public :
            typedef cAtomicF<TypeElem,TheSzBuf>  tAtom;
            typedef typename tAtom::tImplemF     tImplemF;
            typedef typename tImplemF::tContext  tContext;
            typedef typename tContext::tFormula  tFormula;
            typedef typename tContext::tBuf      tBuf;
            friend tContext;

            bool  IsCste0() const override {return mVal==0.0;} ///< Here we know if we are constant 0
            bool  IsCste1() const override {return mVal==1.0;} ///< Here we know if we are constant 1
            const TypeElem * ValCste() const override  {return &mVal;}

      protected  :
            inline cConstantF(tContext * aContext,const std::string & aName,int aNum,const TypeElem& aVal) : 
               tAtom   (aContext,aName),
               mNum    (aNum),
               mVal    (aVal)
            {
              tImplemF::mBuf= tBuf::Constant(mVal); // We know the value of buf, it  will never change
            }
            int     mNum;
            const TypeElem mVal;
};


/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        UNARY FORMULA                            * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */


            /*  ----------------------------------------------------------
               Class implementing unary operation on formula
                  MOTHER CLASS : cUnaryF
                  DERIVED :   cSquareF / cExpF / cMin1F / cLogF
             ----------------------------------------------------------------*/

template <class TypeElem,const int TheSzBuf> class cUnaryF : public cImplemF<TypeElem,TheSzBuf>
{
      public :
            typedef cImplemF<TypeElem,TheSzBuf> tImplemF;
            typedef typename tImplemF::tContext tContext;
            typedef typename tImplemF::tFormula tFormula;
            typedef typename tImplemF::tBuf     tBuf;

            virtual std::string  NameOperator() const = 0;
            std::string  InfixPPrint() const override 
            {
               return NameOperator() + " "+  mF->InfixPPrint() ;
            }
      protected  :
            std::vector<tFormula> Ref() const override{return std::vector<tFormula>{mF};}
            inline cUnaryF(tFormula aF,const std::string & aName) :
                 tImplemF (aF->Context(),aName),
                 mF       (aF),
                 mFBuf    (mF->Buf())
            { }
            tFormula      mF;    ///< The formula argument of unary operator
            const tBuf &  mFBuf; ///< A fast access to its buffer of data
};


/*   Probably not more efficient than implementing Square as F*F, because derivation would give
     F'F + F'F  BUT would be reorder as F'F + F'F and unified ...
     By the way it was a test, if necessary replace Square by F*F */
template <class TypeElem,const int TheSzBuf> class cSquareF : public cUnaryF<TypeElem,TheSzBuf>
{
     public :
            typedef cImplemF  <TypeElem,TheSzBuf>     tImplemF;
            typedef typename tImplemF::tFormula       tFormula;
            typedef cUnaryF <TypeElem,TheSzBuf>       tUF;

            cSquareF (tFormula aF,const std::string & aName) :
                tUF (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "square";}
            void ComputeBuf() override  {tImplemF::mBuf =  tUF::mFBuf * tUF::mFBuf;}
            ///  rule : (FF)' =   2 F' F
            tFormula Derivate(int aK) const override 
            {
                return  2.0  * tUF::mF->Derivate(aK)  * tUF::mF;
            }
};


template <class TypeElem,const int TheSzBuf> class cExpF : public cUnaryF<TypeElem,TheSzBuf>
{
     public :
            typedef cImplemF<TypeElem,TheSzBuf>  tImplemF;
            typedef typename tImplemF::tFormula         tFormula;
            typedef cUnaryF<TypeElem,TheSzBuf> tUF;

            cExpF (tFormula aF,const std::string & aName) :
                tUF (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "exp";}
            void ComputeBuf() override  {tImplemF::mBuf =  exp(tUF::mFBuf);}
            ///  rule : (exp F)' =    F'  * exp F
            tFormula Derivate(int aK) const override 
            {
                return   tUF::mF->Derivate(aK)  * exp(tUF::mF);
            }
};

template <class TypeElem,const int TheSzBuf> class cMin1F : public cUnaryF<TypeElem,TheSzBuf>
{
     public :
            typedef cImplemF  <TypeElem,TheSzBuf>     tImplemF;
            typedef typename tImplemF::tFormula  tFormula;
            typedef cUnaryF <TypeElem,TheSzBuf>     tUF;

            cMin1F (tFormula aF,const std::string & aName) :
                tUF (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "-";}
            void ComputeBuf() override  {tImplemF::mBuf =  -tUF::mFBuf;}
            /// rule : (-F)'  =  - (F')
            tFormula Derivate(int aK) const override 
            {
                return   - tUF::mF->Derivate(aK)  ;
            }
};

template <class TypeElem,const int TheSzBuf> class cLogF : public cUnaryF<TypeElem,TheSzBuf>
{
     public :
            typedef cImplemF  <TypeElem,TheSzBuf>     tImplemF;
            typedef typename tImplemF::tFormula  tFormula;
            typedef cUnaryF <TypeElem,TheSzBuf>     tUF;

            cLogF (tFormula aF,const std::string & aName) :
                tUF (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "log";}
            void ComputeBuf() override  {tImplemF::mBuf =  log(tUF::mFBuf) ;}
            /// rule : (log F)'  =  F' / F
            tFormula Derivate(int aK) const override 
            {
                return   tUF::mF->Derivate(aK) / tUF::mF ;
            }
};



      /* ---------------------------------------*/
      /*           Global Functio on unary op   */
      /* ---------------------------------------*/

/**  A Helper class to avoid code duplication on the process :
       * compute name
       * test existence
       * eventualy create
     Template parameter is the class created
*/

template <class TypeCompiled>  class cGenOperatorUnaire
{
    public :
         typedef typename TypeCompiled::tContext     tContext;
         typedef typename TypeCompiled::tImplemF     tImplemF;
         typedef typename tImplemF::tFormula  tFormula;

         static tFormula   Generate(tFormula aF,const std::string & aNameOp)
         {
             tContext* aPCont = aF->Context();  // Get the context from the formula
             std::string aNameForm = aF.NameFormulaUn(aNameOp);  // Compute the name formula should have

             if (aPCont->ExistFunc(aNameForm))  // If it already exist 
               return aPCont->FuncOfName(aNameForm);  // Then return formula whih this name

             tFormula aResult (new TypeCompiled(aF,aNameForm)); // else create it
             aPCont->AddFormula(aResult); // indicate to the context to remember this new formula 
             return aResult;              // return it
         }
};
    
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf>  square(const cFormula<TypeElem,TheSzBuf> & aF)
{
    return cGenOperatorUnaire<cSquareF<TypeElem,TheSzBuf> >::Generate(aF,"square");
}

template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> exp(const cFormula<TypeElem,TheSzBuf> & aF)
{
    return cGenOperatorUnaire<cExpF<TypeElem,TheSzBuf> >::Generate(aF,"exp");
}
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf>  operator - (const cFormula<TypeElem,TheSzBuf> & aF)
{
    return cGenOperatorUnaire<cMin1F<TypeElem,TheSzBuf> >::Generate(aF,"-");
}
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf>  log (const cFormula<TypeElem,TheSzBuf> & aF)
{
    return cGenOperatorUnaire<cLogF<TypeElem,TheSzBuf> >::Generate(aF,"log");
}

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        BINARY  FORMULA                          * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

            /*  ----------------------------------------------------------
               Class implementing unary operation on formula
                  MOTHER CLASS : cBinaryF
                  DERIVED :   cSumF / cMulF / cSubF / cDivF / cPowF
             ----------------------------------------------------------------*/

template <class TypeElem,const int TheSzBuf> class cBinaryF : public cImplemF<TypeElem,TheSzBuf>
{
      public :
            typedef cImplemF<TypeElem,TheSzBuf>   tImplemF;
            typedef typename tImplemF::tContext   tContext;
            typedef typename tImplemF::tFormula   tFormula;
            typedef typename tImplemF::tBuf       tBuf;

            /// An operator must describe its name
            virtual std::string  NameOperator() const = 0;
            /// We can compute print of binary formula using operator name
            std::string  InfixPPrint() const override 
            {
               return "("+ NameOperator() + " "+ mF1->InfixPPrint() + " " + mF2->InfixPPrint() + ")";
            }
      protected  :
            std::vector<tFormula> Ref() const override{return std::vector<tFormula>{mF1,mF2};}
            inline cBinaryF(tFormula aF1,tFormula aF2,const std::string & aName):
                 tImplemF   (aF1->Context(),aName),
                 mF1    (aF1),
                 mBuf1  (aF1->Buf()),
                 mF2    (aF2),
                 mBuf2  (aF2->Buf())
            {
                // It doesn't work to mix formula from different context
                if (mF1->Context()!=mF2->Context())
                   UserSError("Mix formula from different context");
            }

            tFormula  mF1;   ///< First argument of operator
            const tBuf& mBuf1; ///< A fast access to its buffer of data
            tFormula  mF2;   ///< Second argument of operator
            const tBuf& mBuf2; ///< A fast access to its buffer of data
};


template <class TypeElem,const int TheSzBuf> class cSumF : public cBinaryF <TypeElem,TheSzBuf>
{
      public :
            typedef cImplemF<TypeElem,TheSzBuf>    tImplemF;
            typedef typename tImplemF::tFormula    tFormula;
            typedef cBinaryF<TypeElem,TheSzBuf>    tBF;

            inline cSumF(tFormula aF1,tFormula aF2,const std::string & aName) :
                   tBF (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "+";}
            void ComputeBuf() override  {tImplemF::mBuf =  tBF::mBuf1 + tBF::mBuf2;}
            /// Formula (F+G)' = F' + G'
            tFormula Derivate(int aK) const override {return  tBF::mF1->Derivate(aK) + tBF::mF2->Derivate(aK);}
};

template <class TypeElem,const int TheSzBuf> class cMulF : public cBinaryF<TypeElem,TheSzBuf>
{
      public :
            typedef cImplemF  <TypeElem,TheSzBuf>   tImplemF;
            typedef typename tImplemF::tFormula     tFormula;
            typedef cBinaryF <TypeElem,TheSzBuf>    tBF;

            inline cMulF(tFormula aF1,tFormula aF2,const std::string & aName) :
                   tBF (aF1,aF2,aName)
            { }
      private  :
            std::string  NameOperator() const override {return "*";}
            void ComputeBuf() override  {tImplemF::mBuf =  tBF::mBuf1 * tBF::mBuf2;}
            /// Formula (FG)' = F'G + G'F
            tFormula Derivate(int aK) const override 
            {
                return  tBF::mF2 * tBF::mF1->Derivate(aK)  +  tBF::mF1 * tBF::mF2->Derivate(aK);
            }
};

template <class TypeElem,const int TheSzBuf> class cSubF : public cBinaryF<TypeElem,TheSzBuf>
{
      public :
            typedef cImplemF<TypeElem,TheSzBuf>   tImplemF;
            typedef typename tImplemF::tFormula   tFormula;
            typedef cBinaryF<TypeElem,TheSzBuf>   tBF;

            inline cSubF(tFormula aF1,tFormula aF2,const std::string & aName) :
                   tBF (aF1,aF2,aName)
            { }
      private  :
            std::string  NameOperator() const override {return "-";}
            void ComputeBuf() override  {tImplemF::mBuf =  tBF::mBuf1 - tBF::mBuf2;}
            /// Formula (F-G)' = F' - G'
            tFormula Derivate(int aK) const override {return tBF::mF1->Derivate(aK) - tBF::mF2->Derivate(aK);}
};

template <class TypeElem,const int TheSzBuf> class cDivF : public cBinaryF<TypeElem,TheSzBuf>
{
      public :
            typedef cImplemF<TypeElem,TheSzBuf>   tImplemF;
            typedef typename tImplemF::tFormula   tFormula;
            typedef cBinaryF<TypeElem,TheSzBuf>   tBF;

            inline cDivF(tFormula aF1,tFormula aF2,const std::string & aName) :
                   tBF (aF1,aF2,aName)
            { }
      private  :
            std::string  NameOperator() const override {return "/";}
            /// Formula (F/G)' = (F'G - G'F) / G ^ 2
            void ComputeBuf() override  {tImplemF::mBuf =  tBF::mBuf1 / tBF::mBuf2;}
            tFormula Derivate(int aK) const override 
            {
               return (tBF::mF1->Derivate(aK)* tBF::mF2 - tBF::mF2->Derivate(aK) * tBF::mF1) / square(tBF::mF2);
            }
};

template <class TypeElem,const int TheSzBuf> class cPowF : public cBinaryF<TypeElem,TheSzBuf>
{
      public :
            typedef cImplemF<TypeElem,TheSzBuf>   tImplemF;
            typedef typename tImplemF::tFormula   tFormula;
            typedef cBinaryF<TypeElem,TheSzBuf>   tBF;
            using tBF::mF1;
            using tBF::mF2;

            inline cPowF(tFormula aF1,tFormula aF2,const std::string & aName) :
                   tBF (aF1,aF2,aName)
            { }
      private  :
            std::string  NameOperator() const override {return "^";}
            /// Formula F ^ G = exp(G log(F))
            /// (F^G) ' = (F^G) (G F'/F + G'log(F))
            void ComputeBuf() override  {tImplemF::mBuf =  pow(tBF::mBuf1,tBF::mBuf2);}
            tFormula Derivate(int aK) const override 
            {
               return   pow(mF1,mF2) * ( (mF1->Derivate(aK)/mF1)*mF2 + mF2->Derivate(aK)*log(mF1)) ;
            }
};


      /* ---------------------------------------*/
      /*           Global Functio on unary op   */
      /* ---------------------------------------*/
/**  A Helper class to avoid code duplication on the process , see detailed comment in  cGenOperatorUnaire (analogous) */

template <class TypeCompiled>  class cGenOperatorBinaire
{
    public :
         typedef typename TypeCompiled::tContext  tContext;
         typedef typename TypeCompiled::tImplemF  tImplemF;
         typedef typename tImplemF::tFormula      tFormula;

         static tFormula   Generate(tFormula aF1,tFormula aF2,const std::string & aNameOp)
         {
             // Extract context (take F1 ou F2, does not matter, they must be the same)
             tContext * aPCont = aF1->Context();  
             std::string aNameForm =  aF1.NameFormulaBin(aNameOp,aF2);

             if (aPCont->ExistFunc(aNameForm))
               return aPCont->FuncOfName(aNameForm);

             tFormula aResult (new TypeCompiled(aF1,aF2,aNameForm));
             aPCont->AddFormula(aResult);
             return aResult;
         }
};

       
template <class TypeElem,const int TheSzBuf>
cFormula<TypeElem,TheSzBuf> operator +
                              (
                                    const cFormula <TypeElem,TheSzBuf>& aF1,
                                    const cFormula<TypeElem,TheSzBuf> & aF2
                              ) 
{
     // Use the fact that 0 is neutral element to simplify
     if (aF1->IsCste0()) return aF2;
     if (aF2->IsCste0()) return aF1;

     // Use commutativity of + to have a unique representation
     if (aF1->Name() > aF2->Name()) 
        return aF2+aF1;

     return cGenOperatorBinaire<cSumF<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"+");
}

template <class TypeElem,const int TheSzBuf>
cFormula<TypeElem,TheSzBuf> operator -
                              (
                                    const cFormula <TypeElem,TheSzBuf>& aF1,
                                    const cFormula<TypeElem,TheSzBuf> & aF2
                              ) 
{
     // Use the fact that 0 is neutral element to simplify
     if (aF1->IsCste0()) return -aF2;
     if (aF2->IsCste0()) return aF1;

     return cGenOperatorBinaire<cSubF<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"-");
}

template <class TypeElem,const int TheSzBuf>
cFormula<TypeElem,TheSzBuf> operator *
                              (
                                    const cFormula <TypeElem,TheSzBuf>& aF1,
                                    const cFormula<TypeElem,TheSzBuf> & aF2
                              ) 
{
     // Use the fact that 1 is neutral element to simplify
     if (aF1->IsCste1()) return aF2;
     if (aF2->IsCste1()) return aF1;

     // Use the fact that 0 is absorbant element to simplify
     if (aF1->IsCste0()) return aF1;
     if (aF2->IsCste0()) return aF2;


     // Use commutativity of + to have a unique representation
     if (aF1->Name() > aF2->Name()) 
        return aF2 * aF1;

     return cGenOperatorBinaire<cMulF<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"*");
}

template <class TypeElem,const int TheSzBuf>
cFormula<TypeElem,TheSzBuf> operator /
                              (
                                    const cFormula <TypeElem,TheSzBuf>& aF1,
                                    const cFormula<TypeElem,TheSzBuf> & aF2
                              ) 
{
     if (aF1->IsCste0()) return aF1;  // 0/F2 -> 0
     if (aF2->IsCste1()) return aF1;  // F1/1 -> F1

     return cGenOperatorBinaire<cDivF<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"/");
}

template <class TypeElem,const int TheSzBuf>
cFormula<TypeElem,TheSzBuf>   pow
                              (
                                    const cFormula <TypeElem,TheSzBuf>& aF1,
                                    const cFormula<TypeElem,TheSzBuf> & aF2
                              ) 
{
     return cGenOperatorBinaire<cPowF<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"^");
}

      /* ----------------------------------------------------------*/
      /*  Binary   Operator  between Formula and  constants        */
      /* ----------------------------------------------------------*/

        // ++++++++++++++++++++++++
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> operator +(const TypeElem & aV1,const cFormula<TypeElem,TheSzBuf> & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) + aF2;
}
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> operator +(const cFormula<TypeElem,TheSzBuf> & aF1,const TypeElem & aV2)
{
    return aV2+aF1;
}
        // ************************
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> operator *(const TypeElem & aV1,const cFormula<TypeElem,TheSzBuf> & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) * aF2;
}
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> operator *(const cFormula<TypeElem,TheSzBuf> & aF1,const TypeElem & aV2)
{
    return aV2*aF1;
}
        // --------------------------
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> operator -(const TypeElem & aV1,const cFormula<TypeElem,TheSzBuf> & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) - aF2;
}
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> operator -(const cFormula<TypeElem,TheSzBuf> & aF1,const TypeElem & aV2)
{
  return aF1-aF1->Context()->CsteOfVal(aV2) ;
}
        // /////////////////////////
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> operator /(const TypeElem & aV1,const cFormula<TypeElem,TheSzBuf> & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) / aF2;
}
template <class TypeElem,const int TheSzBuf> 
inline cFormula<TypeElem,TheSzBuf> operator /(const cFormula<TypeElem,TheSzBuf> & aF1,const TypeElem & aV2)
{
  return aF1/aF1->Context()->CsteOfVal(aV2) ;
}


/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *         cImplemF / cContextF                    * */
/* *        External Definition of methods           * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

      /* ---------------------- */
      /*        cImplemF        */
      /* ---------------------- */

template <class TypeElem,const int TheSzBuf> int cImplemF<TypeElem,TheSzBuf>::RecursiveRec() const
{
   int aRes = 1;
   for (const auto  & aF : Ref())
   {
      aRes += aF->RecursiveRec();
   }
   return aRes;
}

      /* ---------------------- */
      /*       cContextF        */
      /* ---------------------- */

template <class TypeElem,const int TheSzBuf> 
std::vector<std::string> cContextF<TypeElem,TheSzBuf>::MakeAutomId(const std::string & aPrefix,int aNb)
{
   std::vector<std::string> aRes;
   for (int aK=0 ; aK<aNb ; aK++)
       aRes.push_back(aPrefix+ std::to_string(aK));
   return aRes;
}

template <class TypeElem,const int TheSzBuf> 
cContextF<TypeElem,TheSzBuf>::cContextF
(
       const std::vector<std::string> & aVNameUK,
       const std::vector<std::string> & aVNameObs
) :
    mNbCste     (0),
    mNbUK       (aVNameUK.size()),
    mNbObs      (aVNameObs.size()),
    mCste0      (CsteOfVal(0.0)),
    mCste1      (CsteOfVal(1.0)),
    mCste2      (CsteOfVal(2.0)),
    mNbInBuf    (0),
    mCurFormSet (false)
{
    // Generate all the function corresponding to unknown
    for (size_t aNumUK=0 ; aNumUK<mNbUK ; aNumUK++)
    {
        tFormula aFuncUK(new cUnknownF<TypeElem,TheSzBuf>(this,aVNameUK[aNumUK],aNumUK));  // Create it
        mVFormUnknowns.push_back(aFuncUK);   // Push it in vector of coordinat func
        AddFormula(aFuncUK);  // Add to all func
    }

    // Generate all the function corresponding to observations
    for (size_t aNumObs=0 ; aNumObs<mNbObs ; aNumObs++)
    {
        tFormula aFuncObs(new cObservationF<TypeElem,TheSzBuf>(this,aVNameObs[aNumObs],aNumObs));  // Create it
        mVFormObservations.push_back(aFuncObs);   // Push it in vector of coordinat func
        AddFormula(aFuncObs);  // Add to all func
    }
}

template <class TypeElem,const int TheSzBuf> 
cContextF<TypeElem,TheSzBuf>::cContextF(int aNbUK,int aNbObs) :
         cContextF<TypeElem,TheSzBuf>(MakeAutomId("X",aNbUK),MakeAutomId("V",aNbObs))
{
}

template <class TypeElem,const int TheSzBuf> 
cContextF<TypeElem,TheSzBuf>::~cContextF()
{
    for (auto & aForm : mVAllFormula)
    {
       aForm.FreeMem();
    }
}

template <class TypeElem,const int TheSzBuf>
cFormula<TypeElem,TheSzBuf> cContextF<TypeElem,TheSzBuf>::CsteOfVal(const TypeElem & aCste) 
{
  tFormula & aRef = mDicoCste[aCste];
  if (aRef.IsNull())  // If it was not existing, the map contain now the def element
  {
     aRef=tFormula(new cConstantF<TypeElem,TheSzBuf>(this,"C"+std::to_string(mNbCste),mNbCste,aCste));
     mNbCste++;
     AddFormula(aRef);
  }

  return aRef;
}

template <class TypeElem,const int TheSzBuf>
cFormula <TypeElem,TheSzBuf> cContextF<TypeElem,TheSzBuf>::FuncOfName(const std::string & aName) const 
{
    const auto & anIt = mDicoFunc.find(aName);
    if (anIt == mDicoFunc.end()) InternalError ("Try to acces non existing name :[" + aName + "]");
    return anIt->second;
}

template <class TypeElem,const int TheSzBuf> 
void cContextF<TypeElem,TheSzBuf>::PushNewEvals
     (
          const std::vector<TypeElem> & aVUK,
          const std::vector<TypeElem> & aVObs
     )
{
    if (mNbInBuf >= TheSzBuf)
    {
       UserSError("Push exceed buffer capacity");
    }
    SetNewVals(mVFormUnknowns,"Unknowns",aVUK);
    SetNewVals(mVFormObservations,"Observations",aVObs);
    mNbInBuf++;
}

template <class TypeElem,const int TheSzBuf> 
void cContextF<TypeElem,TheSzBuf>::SetNewEvals
     (
         std::vector<tFormula> & aVF,
         const std::string & aMes,
         const std::vector<TypeElem> & aVVals
     )
{
    if (aVF.size() != aVVals.size())
    {
       UserSError("Bad size in " + aMes);
    }
    for (size_t aK=0 ; aK<aVF.size() ; aK++)
    {
        aVF[aK]->Buf()[mNbInBuf] = aVVals[aK];
    }
}

template <class TypeElem,const int TheSzBuf> 
void cContextF<TypeElem,TheSzBuf>::ShowStackFunc() const
{
    for (const auto & aForm : mVAllFormula)
    {
       std::cout << "Form[" << aForm->NumGlob() << "] => " << aForm->Name();
       const TypeElem * aPV = aForm->ValCste();
       if (aPV)
           std::cout << " ; Val=" << *aPV;
       std::cout << "\n";
    }
}

}; //   NS_MMVII_FormalDerivative

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        TEST                                     * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

namespace  FD = NS_MMVII_FormalDerivative;

static const int SzBufTest = 8;
typedef  double TypeTest;
typedef  FD::cFormula <TypeTest,SzBufTest>  tFormulaTest;


void   BenchFormalDer()
{
    {
       FD::cContextF<TypeTest,SzBufTest>  aCFD(3,5);

       tFormulaTest  X0 = aCFD.VUK().at(0);
       if (0)
       {
          FD::cContextF<TypeTest,SzBufTest>  aCFD2(3,5);
          tFormulaTest  B0 = aCFD2.VUK().at(0);
          X0 + B0;
       }
       tFormulaTest  X1 = aCFD.VUK().at(1);
       tFormulaTest  X2 = aCFD.VUK().at(2);

       tFormulaTest  aF0 =  X0 ;
       for (int aK=0 ; aK<5 ; aK++)
       {
           std::cout << "K= " << aK << " R=" << aF0->RecursiveRec() << "\n";
           aF0 = aF0 + aF0;
       }
/*
       tFormulaTest  aF1 = aF0 + aF0;
       tFormulaTest  aF2 = aF1 + aF1;
       tFormulaTest  aF3 = aF2 + aF2;
       tFormulaTest  aF4 = aF3 + aF3;
       std::cout << "Re=" << aF->InfixPPrint() << "\n";
*/
   

       tFormulaTest  aF = (X0+X1) * (X0 +square(X2)) - exp(-square(X0))/X0;
       // tFormulaTest  aF = X0 * X0;
       tFormulaTest  aFd0 = aF->Derivate(0);

       std::cout << "F=" << aF->InfixPPrint() << "\n";
       std::cout << "Fd=" << aFd0->InfixPPrint() << "\n";

       aF->ComputeBuf();

       aCFD.ShowStackFunc();
/*
       aCFD.CsteOfVal(3.14);
       aCFD.CsteOfVal(3.14);
       tFormulaTest  aU0 = aCFD.VUK()[0];
       tFormulaTest  aU1 = aCFD.VUK()[1];
       tFormulaTest  aO0 = aCFD.VObs()[0];
       tFormulaTest  aO1 = aCFD.VObs()[1];
       tFormulaTest  aO2 = aCFD.VObs()[2];

       tFormulaTest  aSom00 = aU0 + aO0;
       tFormulaTest  aSomInv00 = aO0 + aU0;
       tFormulaTest  aSom11 = aO1 + aU1;

       tFormulaTest  aSom0 = aCFD.VUK()[0] + aCFD.Cste0();
       tFormulaTest  aSom1 = aCFD.VUK()[0] + aCFD.Cste1();

       tFormulaTest  aSom3 = aCFD.VUK()[0] + 3.14;
       tFormulaTest  aSom4 = 3.14 + aCFD.VUK()[0] ;
       std::cout << "TEST ADD CST " << aSom0->Name() << " " << aSom1->Name() << "\n";
       std::cout << "TEST ADD CST " << aSom3->Name() << " " << aSom4->Name() << "\n";

       aO0+aO1;
       aO1+aO2;
       aO0+(aO1+aO2);
       {
          tFormulaTest aS=(aO0+aO1)*(aO2+2.1);
          std::cout << "PP=" << aS->InfixPPrint() << "\n";
       }
*/

       // cFormula<TypeTest,SzBufTest> aPtr(nullptr);
       // aPtr->IsCste0();
       

       // std::shared_ptr<cFuncFormalDer <8,double> > aF1  =
       
    }
    // new cContextF<double,100> (3,5);

    int i=10;
    std::string aStr = "i="+ std::to_string(i);
    std::cout  << "BenchFormalDerBenchFormalDerBenchFormalDer " << aStr << "\n";

    getchar();
}


#endif // _MMVII_DynAndStatFormalDerivation_H_
