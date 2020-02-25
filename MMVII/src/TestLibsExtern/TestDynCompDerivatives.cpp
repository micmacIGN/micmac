#ifndef _MMVII_DynAndStatFormalDerivation_H_
#define _MMVII_DynAndStatFormalDerivation_H_


/** \file TestDynCompDerivatives.cpp
    \brief Proto of generated code

Prototype of generated code, do not know how it will evolve

http://en.wikipedia.org/wiki/Automatic_differentiation
https://git.irc.umbc.edu/photorig/openMVG/blob/260584fda68dce095e279362efd24a2d7d7cf5d9/src/third_party/ceres-solver/include/ceres/jet.h

https://mc-stan.org/
http://www.met.reading.ac.uk/clouds/adept/array_features.html
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.7749&rep=rep1&type=pdf
http://www.autodiff.org/
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
template <class TypeElem> class cFormula ; 

/** Class for managing the "context", i.e. coordinating all the  formula 
    and their derivative corresponding to a single use .
*/
template <class TypeElem> class cContextF;  




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

    // -------- Declaration of Context class  ----------------

template <class TypeElem> class cContextF : public cMemCheck
{
    public :
 
        typedef cFormula <TypeElem>       tFormula;
        typedef std::vector<TypeElem>     tOneRes;

      // ---------------------------  Constructors / Destructor -------------------
        /// Constructor with explicit Id for Unknown/Observation. Used if we want to analyze the generated code
        inline cContextF(int SzBuf,const std::vector<std::string> & aVecUK,const std::vector<std::string> & aVecObs);
        /// Constructor with basic Id (used if we dont generate code, or dont want to analyse it by human)
        inline cContextF(int SzBuf,int aNbUnknown,int aNbObservation);
        /// Destructeur will free allocated formulas
        ~cContextF();
        /// Copies are not allowed on this kind of object.
        cContextF(const cContextF<TypeElem> &) = delete;  

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
             ABOVE WAS THE REAL PUBLIC PART OF cContextF FOR USER OF LIBRARY.  THE REST
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
       typedef cContextF<TypeElem>             tContext;
       typedef typename tContext::tBuf         tBuf;
       typedef typename tContext::tFormula     tFormula;

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
       tContext *  Context() const {return mContext;}   ///< Standard accesor
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
       inline cImplemF (tContext * aContext,const std::string & aName) :
              mContext (aContext),
              mBuf     (mContext->SzBuf(),TypeElem(0.0)),
              mDataBuf (mBuf.data()),
              mName    (aName),
              mNumGlob (mContext->NbCurFonc()),
              mReached (false),
              mDepth   (-1)
       { 
       }

       tContext *             mContext;   ///< Context that manage all the funcion cooperating
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
       typedef cContextF<TypeElem>   tContext;
       typedef cImplemF<TypeElem>    tImplemF;
       typedef typename tContext::tFormula    tFormula;

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
            typedef typename tImplemF::tContext  tContext;
            typedef typename tContext::tFormula  tFormula;

            /// Should work always
            std::string  InfixPPrint() const override {return tImplemF::Name();}
            /// Rule deriv=0 , work by default (constant and observations)
            tFormula Derivate(int aK) const override {return tImplemF::mContext->Cste0();}

            /// Generally nothing to do in atomic, their buffer has been filled witj adequate values
            void ComputeBuf(int aK0,int aK1) override  { }
            std::vector<tFormula> Ref() const override{return std::vector<tFormula>();}
     protected :
            inline cAtomicF(tContext * aContext,const std::string& aName) :
                tImplemF       (aContext,aName)
            { }
};

template <class TypeElem> class cUnknownF : public cAtomicF<TypeElem>
{
      public :
            typedef cAtomicF<TypeElem>  tAtom;
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

template <class TypeElem> class cObservationF : public cAtomicF<TypeElem>
{
      public :
            typedef cAtomicF<TypeElem>  tAtom;
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

template <class TypeElem> class cConstantF : public cAtomicF<TypeElem>
{
      public :
            typedef cAtomicF<TypeElem>  tAtom;
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
               for (auto & aV : tImplemF::mBuf) aV = aVal;  // Initialize buf  with const val
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

template <class TypeElem> class cUnaryF : public cImplemF<TypeElem>
{
      public :
            typedef cImplemF<TypeElem> tImplemF;
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
                 mDataF   (mF->DataBuf())
            { }
            tFormula        mF;    ///< The formula argument of unary operator
            const TypeElem* mDataF; ///< Fast access to data of buf
};


/*   Probably not more efficient than implementing Square as F*F, because derivation would give
     F'F + F'F  BUT would be reorder as F'F + F'F and unified ...
     By the way it was a test, if necessary replace Square by F*F */
template <class TypeElem> class cSquareF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;
     
            cSquareF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "square";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] =  mDataF[aK] * mDataF[aK];
            }
            ///  rule : (FF)' =   2 F' F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  2.0  * mF->Derivate(aK)  * mF;
            }
};


template <class TypeElem> class cExpF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cExpF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "exp";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] = std::exp(mDataF[aK]);
            }
            ///  rule : (exp FF)' =   F' exp F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return   mF->Derivate(aK)  * exp(mF);
            }
};

template <class TypeElem> class cMin1F : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cMin1F (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "-";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] = - mDataF[aK];
            }
            /// rule : (-F)'  =  - (F')
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return   - mF->Derivate(aK)  ;
            }
};

template <class TypeElem> class cLogF : public cUnaryF<TypeElem>
{
     public :
            using cUnaryF<TypeElem>::mF;
            using cUnaryF<TypeElem>::mDataF;
            using cImplemF<TypeElem>::mDataBuf;

            cLogF (cFormula<TypeElem> aF,const std::string & aName) :
                cUnaryF <TypeElem> (aF,aName)
            { }
      private :
            std::string  NameOperator() const override {return "log";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                    mDataBuf[aK] = std::log(mDataF[aK]);
            }
            /// rule : (log F)'  =  F' / F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return   mF->Derivate(aK) / mF ;
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
    
template <class TypeElem> 
inline cFormula<TypeElem>  square(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cSquareF<TypeElem> >::Generate(aF,"square");
}

template <class TypeElem> 
inline cFormula<TypeElem> exp(const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cExpF<TypeElem> >::Generate(aF,"exp");
}
template <class TypeElem> 
inline cFormula<TypeElem>  operator - (const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cMin1F<TypeElem> >::Generate(aF,"-");
}
template <class TypeElem> 
inline cFormula<TypeElem>  log (const cFormula<TypeElem> & aF)
{
    return cGenOperatorUnaire<cLogF<TypeElem> >::Generate(aF,"log");
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

template <class TypeElem> class cBinaryF : public cImplemF<TypeElem>
{
      public :
            typedef cImplemF<TypeElem>   tImplemF;
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
                 tImplemF (aF1->Context(),aName),
                 mF1      (aF1),
                 mDataF1  (aF1->DataBuf()),
                 mF2      (aF2),
                 mDataF2  (aF2->DataBuf())
            {
                // It doesn't work to mix formula from different context
                if (mF1->Context()!=mF2->Context())
                   UserSError("Mix formula from different context");
            }

            tFormula  mF1;   ///< First argument of operator
            const TypeElem  *mDataF1; ///< Fast access to data of buf F1
            tFormula  mF2;   ///< Second argument of operator
            const TypeElem  *mDataF2; ///< Fast access to data of buf F2
};


template <class TypeElem> class cSumF : public cBinaryF <TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1+aV2;}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cSumF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "+";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] + mDataF2[aK];
            }
            cFormula<TypeElem> Derivate(int aK) const override {return  mF1->Derivate(aK) + mF2->Derivate(aK);}
};

template <class TypeElem> class cMulF : public cBinaryF<TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1*aV2;}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cMulF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "*";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] * mDataF2[aK];
            }
            /// Formula (FG)' = F'G + G'F
            cFormula<TypeElem> Derivate(int aK) const override 
            {
                return  mF2*mF1->Derivate(aK) + mF1*mF2->Derivate(aK);
            }
};

template <class TypeElem> class cSubF : public cBinaryF<TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1-aV2;}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cSubF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "-";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] - mDataF2[aK];
            }
            /// Formula (F-G)' = F' - G'
            cFormula<TypeElem> Derivate(int aK) const override {return mF1->Derivate(aK) - mF2->Derivate(aK);}
};

template <class TypeElem> class cDivF : public cBinaryF<TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return aV1/aV2;}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cDivF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "/";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  mDataF1[aK] / mDataF2[aK];
            }
            /// Formula (F/G)' = (F'G - G'F) / G ^ 2
            cFormula<TypeElem> Derivate(int aK) const override 
            {
               return (mF1->Derivate(aK)*mF2 - mF2->Derivate(aK)*mF1)/square(mF2);
            }
};

template <class TypeElem> class cPowF : public cBinaryF<TypeElem>
{
      public :
            /// Required by constant reduction in  cGenOperatorBinaire
            static TypeElem Operation(const TypeElem & aV1,const TypeElem & aV2) {return std::pow(aV1,aV2);}
            using cBinaryF<TypeElem>::mF1;
            using cBinaryF<TypeElem>::mF2;
            using cBinaryF<TypeElem>::mDataF1;
            using cBinaryF<TypeElem>::mDataF2;
            using cImplemF<TypeElem>::mDataBuf;

            inline cPowF(cFormula<TypeElem> aF1,cFormula<TypeElem> aF2,const std::string & aName) :
                   cBinaryF<TypeElem> (aF1,aF2,aName) 
            { }
      private  :
            std::string  NameOperator() const override {return "^";}
            void ComputeBuf(int aK0,int aK1) override  
            {
                for (int aK=aK0 ; aK<aK1 ; aK++)
                     mDataBuf[aK] =  std::pow(mDataF1[aK],mDataF2[aK]);
            }
            /// Formula F ^ G = exp(G log(F))
            /// (F^G) ' = (F^G) (G F'/F + G'log(F))
            cFormula<TypeElem> Derivate(int aK) const override 
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
         typedef typename TypeCompiled::tElem     tElem;
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

             // Maybe the two operand are constant ? Then we can reduce
             {
                 const tElem * aC1 = aF1->ValCste();
                 const tElem * aC2 = aF2->ValCste();
                 if (aC1 && aC2) 
                 {
                    tElem  aC12= TypeCompiled::Operation(*aC1,*aC2);
                    std::cout << "ZZZZZZ " << *aC1 << " " << *aC2 << " " << aC12<< "\n";
                 }
             }

             tFormula aResult (new TypeCompiled(aF1,aF2,aNameForm));
             aPCont->AddFormula(aResult);
             return aResult;
         }
};

       
template <class TypeElem>
cFormula<TypeElem> operator +
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     // Use the fact that 0 is neutral element to simplify
     if (aF1->IsCste0()) return aF2;
     if (aF2->IsCste0()) return aF1;

     // Use commutativity of + to have a unique representation
     if (aF1->Name() > aF2->Name()) 
        return aF2+aF1;

     return cGenOperatorBinaire<cSumF<TypeElem> >::Generate(aF1,aF2,"+");
}

template <class TypeElem>
cFormula<TypeElem> operator -
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     // Use the fact that 0 is neutral element to simplify
     if (aF1->IsCste0()) return -aF2;
     if (aF2->IsCste0()) return aF1;

     return cGenOperatorBinaire<cSubF<TypeElem> >::Generate(aF1,aF2,"-");
}

template <class TypeElem>
cFormula<TypeElem> operator *
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
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

     return cGenOperatorBinaire<cMulF<TypeElem> >::Generate(aF1,aF2,"*");
}

template <class TypeElem>
cFormula<TypeElem> operator /
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     if (aF1->IsCste0()) return aF1;  // 0/F2 -> 0
     if (aF2->IsCste1()) return aF1;  // F1/1 -> F1

     return cGenOperatorBinaire<cDivF<TypeElem> >::Generate(aF1,aF2,"/");
}

template <class TypeElem>
cFormula<TypeElem>   pow
                              (
                                    const cFormula <TypeElem>& aF1,
                                    const cFormula<TypeElem> & aF2
                              ) 
{
     return cGenOperatorBinaire<cPowF<TypeElem> >::Generate(aF1,aF2,"^");
}

      /* ----------------------------------------------------------*/
      /*  Binary   Operator  between Formula and  constants        */
      /* ----------------------------------------------------------*/

        // ++++++++++++++++++++++++
template <class TypeElem> 
inline cFormula<TypeElem> operator +(const TypeElem & aV1,const cFormula<TypeElem> & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) + aF2;
}
template <class TypeElem> 
inline cFormula<TypeElem> operator +(const cFormula<TypeElem> & aF1,const TypeElem & aV2)
{
    return aV2+aF1;
}
        // ************************
template <class TypeElem> 
inline cFormula<TypeElem> operator *(const TypeElem & aV1,const cFormula<TypeElem> & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) * aF2;
}
template <class TypeElem> 
inline cFormula<TypeElem> operator *(const cFormula<TypeElem> & aF1,const TypeElem & aV2)
{
    return aV2*aF1;
}
        // --------------------------
template <class TypeElem> 
inline cFormula<TypeElem> operator -(const TypeElem & aV1,const cFormula<TypeElem> & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) - aF2;
}
template <class TypeElem> 
inline cFormula<TypeElem> operator -(const cFormula<TypeElem> & aF1,const TypeElem & aV2)
{
  return aF1-aF1->Context()->CsteOfVal(aV2) ;
}
        // /////////////////////////
template <class TypeElem> 
inline cFormula<TypeElem> operator /(const TypeElem & aV1,const cFormula<TypeElem> & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) / aF2;
}
template <class TypeElem> 
inline cFormula<TypeElem> operator /(const cFormula<TypeElem> & aF1,const TypeElem & aV2)
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
      /*       cContextF        */
      /* ---------------------- */

template <class TypeElem> 
std::vector<std::string> cContextF<TypeElem>::MakeAutomId(const std::string & aPrefix,int aNb)
{
   std::vector<std::string> aRes;
   for (int aK=0 ; aK<aNb ; aK++)
       aRes.push_back(aPrefix+ std::to_string(aK));
   return aRes;
}

template <class TypeElem> 
cContextF<TypeElem>::cContextF
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
cContextF<TypeElem>::cContextF(int aSzBuf,int aNbUK,int aNbObs) :
         cContextF<TypeElem>(aSzBuf,MakeAutomId("X",aNbUK),MakeAutomId("V",aNbObs))
{
}

template <class TypeElem> 
cContextF<TypeElem>::~cContextF()
{
    for (auto & aForm : mVAllFormula)
    {
       aForm.FreeMem();
    }
}

template <class TypeElem>
cFormula<TypeElem> cContextF<TypeElem>::CsteOfVal(const TypeElem & aCste) 
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
cFormula <TypeElem> cContextF<TypeElem>::FuncOfName(const std::string & aName) const 
{
    const auto & anIt = mDicoFunc.find(aName);
    if (anIt == mDicoFunc.end()) InternalError ("Try to acces non existing name :[" + aName + "]");
    return anIt->second;
}

template <class TypeElem> 
void cContextF<TypeElem>::SetNewVals
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
void cContextF<TypeElem>::PushNewEvals
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
void    cContextF<TypeElem>::SetCurFormulasWithDerivative(const std::vector<tFormula> & aVF)
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
void cContextF<TypeElem>::SetCurFormulas(const std::vector<tFormula> & aVF)
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
const std::vector<std::vector<TypeElem> *> & cContextF<TypeElem>::EvalAndClear()
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
void cContextF<TypeElem>::ShowStackFunc() const
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

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        TEST                                     * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

namespace  FD = NS_MMVII_FormalDerivative;
using FD::square;

std::vector<double> VRand(unsigned aSeed,int aNb)
{
    std::srand(aSeed);
    std::vector<double> aVRes;
    for (int aK=0 ; aK<aNb ; aK++)
    {
       double aV =  std::rand()/((double) RAND_MAX );
       aVRes.push_back(aV);
    }

    return aVRes;
}

template <class Type> 
std::vector<Type> Residu
                  (
                     const std::vector<Type> & aVUk,
                     const std::vector<Type> & aVObs
                  )
{
    const Type & X0 = aVUk.at(0);
    const Type & X1 = aVUk.at(1);
    const Type & X2 = aVUk.at(2);

    const Type & V0 = aVObs.at(0);
    const Type & V1 = aVObs.at(1);

    Type aF0 =  2.0 *X0 + X0*X1*X2 + pow(square(V0)+square(X0-X2),(X1*V1)/X2);
    Type aF1 =  log(square(X0+X1+X2+V0+V1));
    Type aF2 =  -aF0 + aF1;
             
    return {aF0,aF1,aF2};
}


void TestDyn()
{
    int aNbUk  = 3;
    int aNbObs = 2;

    FD::cContextF<double>  aCFD(100,aNbUk,aNbObs);
    aCFD.SetCurFormulasWithDerivative(Residu(aCFD.VUk(),aCFD.VObs()));

    int aNbT = 2;
    unsigned aSeedUk=333, aSeedObs=222;
    for (int aKTest=0 ; aKTest<aNbT ; aKTest++)
    {
       std::vector<double> aVUk  = VRand(aKTest+aSeedUk ,aNbUk);
       std::vector<double> aVObs = VRand(aKTest+aSeedObs,aNbObs);
       aCFD.PushNewEvals(aVUk,aVObs);
    }
    aCFD.EvalAndClear();

    for (int aKTest=0 ; aKTest<aNbT ; aKTest++)
    {
       // const std::vector<double> & aLineDyn =  *(aVDyn[aKTest]);
       std::vector<double> aVUk  = VRand(aKTest+aSeedUk ,aNbUk);
       std::vector<double> aVObs = VRand(aKTest+aSeedObs,aNbObs);
       std::vector<double> aVRes = Residu(aVUk,aVObs);
       int aNbRes = aVRes.size();

       for (int aKx=0 ; aKx<aNbUk ; aKx++)
       {
           double aEps = 1e-5;
           std::vector<double> aVUkP  = aVUk;
           std::vector<double> aVUkM  = aVUk;
           aVUkP[aKx] += aEps;
           aVUkM[aKx] -= aEps;

           std::vector<double> aVResP = Residu(aVUkP,aVObs);
           std::vector<double> aVResM = Residu(aVUkM,aVObs);
           for (int aKRes=0 ; aKRes<aNbRes ; aKRes++)
           {
               double aDerNum  = (aVResP[aKRes]-aVResM[aKRes]) / (2*aEps);
               double aDerForm = aCFD.DerComp(aKTest,aKRes,aKx);
               double aDif = std::abs(aDerNum-aDerForm);
               assert(aDif<1e-4);
           }
       }
       for (int aKRes=0 ; aKRes<aNbRes ; aKRes++)
       {
           double aDif = std::abs(aCFD.ValComp(aKTest,aKRes) - aVRes[aKRes] );
           assert(aDif<1e-7);
       }

    }
    aCFD.ShowStackFunc();
    getchar();
}



typedef  double TypeTest;
typedef  FD::cFormula <TypeTest>  tFormulaTest;

// #include "include/MMVII_all.h"
// #include "include/MMVII_Derivatives.h"






#define SzTEigen 90
typedef float tTEigen;
typedef  Eigen::Array<tTEigen,1,Eigen::Dynamic>  tEigenSubArray;
typedef  Eigen::Map<tEigenSubArray > tEigenWrap;
void   BenchFormalDer()
{
    // TestDyn();
    if (1)
    {
        Eigen::Array<tTEigen, 1, SzTEigen>  aAFix = Eigen::Array<tTEigen, 1, SzTEigen>::Random();
        // Eigen::Array<tTEigen,Eigen::Dynamic,Eigen::Dynamic>   aADyn(1,SzTEigen);
        // Eigen::Array<tTEigen,Eigen::Dynamic,1>   aADyn(SzTEigen);
        Eigen::Array<tTEigen,1,Eigen::Dynamic>   aADyn(SzTEigen);
        Eigen::Array<tTEigen,Eigen::Dynamic,Eigen::Dynamic>   aADyn1(1,1);
        Eigen::Array<tTEigen,Eigen::Dynamic,Eigen::Dynamic>   aADyn2(1,SzTEigen);


        for (int aX=0 ; aX<SzTEigen ; aX++)
        {
            aAFix(0,aX)  = 10 + 2.0*aX;
            aAFix(0,aX)  = 1;
            aAFix(0,aX)  = 10 + 2.0*aX;
        }
        aAFix = 1;
        aADyn = aAFix;
       
        aADyn1(0,0) = 1.0;
         
#if (WITH_MMVII)
        int aNb=1e7;
        double aT0 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
             aAFix = aAFix + aAFix -10;
             aAFix = (aAFix + 10)/2;
        }
        double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
             aADyn = aADyn + aADyn -10;
             aADyn = (aADyn + 10)/2;
        }
        double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        if (0)
        {
           for (int aK=0 ; aK<aNb*SzTEigen ; aK++)
           {
               aADyn1 = aADyn1 + aADyn1 -10;
               aADyn1 = (aADyn1 + 10)/2;
           }
        }
        double aT3 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
             Eigen::Array<tTEigen,1,Eigen::Dynamic>   aBloc 
                // = aADyn.topLeftCorner(1,SzTEigen);
                // = aADyn.block(0,0,1,SzTEigen);
                = aADyn.head(SzTEigen-1);
             aBloc = aBloc + aBloc -10;
             aBloc = (aBloc + 10)/2;
             if (aK==0)
             {
                  std::cout << "AAAAADr  " << &(aBloc(0,0)) - &(aADyn(0,0)) << "\n";
                  std::cout << "AAAAADr  " << aBloc(0,0)   << " " << aADyn(0,0) << "\n";
             }
        }
        double aT4 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
            for (int aX=0 ; aX<SzTEigen ; aX++)
            {
                aADyn2(aX) = aADyn2(aX) + aADyn2(aX) -10;
                aADyn2(aX) = (aADyn2(aX) + 10)/2;
            }
        }
        double aT5 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
            tTEigen * aData = &  aADyn(0) ;
            for (int aX=0 ; aX<SzTEigen ; aX++)
            {
                aData[aX] =  aData[aX] + aData[aX] -10;
                aData[aX] = (aData[aX] + 10)/2;
            }
        }
        double aT6 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        for (int aK=0 ; aK<aNb ; aK++)
        {
             tEigenWrap aWrap(&aADyn(0),1,SzTEigen-1);
             // aWrap += aWrap ;
             // aWrap += 10;
             aWrap = aWrap + aWrap -10;
             aWrap = (aWrap + 10)/2;
        }
        double aT7 = cMMVII_Appli::CurrentAppli().SecFromT0(); 

        std::cout << " T01-EigenFix " << aT1-aT0 << " T12-EigenDyn " << aT2-aT1 
                  << " T23 " << aT3-aT2 << " T34-EigenBloc " << aT4-aT3  << "\n"
                  << " T45-EigenElem " << aT5-aT4 << " T56_RawData " << aT6-aT5 
                  << " T67-EigenWrap " << aT7-aT6 
                  << "\n";
        std::cout << "FIXSZ " << aAFix.rows() << " C:" <<  aAFix.cols() << "\n";
        std::cout << "DYNSZ " << aADyn.rows() << " C:" <<  aADyn.cols() << "\n";
#endif
    }


    {
       int aNbUk  = 3;
       int aNbObs = 5;
       FD::cContextF<TypeTest>  aCFD(100,aNbUk,aNbObs);

       std::vector<TypeTest> aVUk(aNbUk,0.0);
       std::vector<TypeTest> aVObs(aNbObs,0.0);
       aCFD.PushNewEvals(aVUk,aVObs);
       aCFD.EvalAndClear();

       tFormulaTest  X0 = aCFD.VUk().at(0);
       if (0)
       {
          FD::cContextF<TypeTest>  aCFD2(100,3,5);
          tFormulaTest  B0 = aCFD2.VUk().at(0);
          X0 + B0;
       }
       tFormulaTest  X1 = aCFD.VUk().at(1);
       tFormulaTest  X2 = aCFD.VUk().at(2);

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

       // aF->ComputeBuf();
       std::vector<tFormulaTest> aVF{aF0,aF0};
       aCFD.SetCurFormulas(aVF);
       aCFD.SetCurFormulasWithDerivative(aVF);

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

       // aPtr->IsCste0();
       

       // std::shared_ptr<cFuncFormalDer <8,double> > aF1  =
       
    }
    // new cContextF<double,100> (3,5);

    

    int i=10;
    std::string aStr = "i="+ std::to_string(i);
    std::cout  << "BenchFormalDerBenchFormalDerBenchFormalDer " << aStr << "\n";

    Eigen::MatrixXf m(10,20);
    Eigen::MatrixXf aM2 = m.topLeftCorner(8,15);


    Eigen::Array<double, 2, 25>  a;

    std::cout << "MMMM R:" << m.rows() << " C:" <<  m.cols() << "\n";
    std::cout << "MMMM R:" << aM2.rows() << " C:" <<  aM2.cols() << "\n";
    std::cout << "MMMM A:" << a.rows() << " C:" <<  a.cols() << "\n";

    getchar();
}


#endif // _MMVII_DynAndStatFormalDerivation_H_
