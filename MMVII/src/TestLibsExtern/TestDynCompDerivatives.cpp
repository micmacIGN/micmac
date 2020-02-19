#ifndef _MMVII_DynAndStatFormalDerivation_H_
#define _MMVII_DynAndStatFormalDerivation_H_


/** \file TestDynCompDerivatives.cpp
    \brief Proto of generated code

Prototype of generated code, do not know how it will evolve
   
*/

#define WITH_MMVII false

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

#endif            //==========================================================

#define TPLDecl <class TypeElem,const int TheSzBuf>
#define TPLInst <TypeElem,TheSzBuf>

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *    Pre-Declaration of all classes               * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

   // The two classes visible by user are cFormulaFD (mainly) and cContextFormalDer (rarely)

/** Encupsulation of share ptr on cImplemFormFD , */
template TPLDecl class cFormulaFD ; 
/** Class for managing the "context", i.e. coordinating all the  formula 
    and their derivative corresponding to a single use */
template TPLDecl class cContextFormalDer;  

     // Other classe are cImplemFormFD  and its derivatives
template TPLDecl class cImplemFormFD  ;   ///< Mother class of all FunFormal, abstract class with pure virtual method
          // ---  "Atomic" function : Unknown, constant, observation
template TPLDecl class cFObsFormalDer ;   ///< "Observations" corresponding to user constant (change for each evaluation)
template TPLDecl class cFCsteFormalDer ;   ///< Constant function
template TPLDecl class cFUnknownFormalDer; ///< "Unknown"  for representing coordinates function X0,X1,X2 ....
          // ---  Unary operator
template TPLDecl class cOpUnFormalDer ;     ///< Mother Class of all unary operator
template TPLDecl class cSquareFormalDer ;   ///< Class for square operator
template TPLDecl class cExpFormalDer ;      ///< Class for exponential operator
template TPLDecl class cMin1FormalDer ;     ///< Class for Unary Minus
          // ---  Binary operator
template TPLDecl class cOpBinFormalDer ;   ///< Mother class of binary operators
template TPLDecl class cSumFormalDer ;     ///< Class for sum of 2 functions
template TPLDecl class cMulFormalDer ;     ///< Class for multiplication of 2 functions
template TPLDecl class cSubFormalDer ;     ///< Class for substraction of 2 functions
template TPLDecl class cDivFormalDer ;     ///< Class for division of 2 functions

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *         Declare all operator                    * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

    // -------- Declare all binary operators  ----------------
      
          // For each operator with have the 3 versions  "Formula x Formula"  ,
          //  "Number x Formula"    and "Formula x Number"

      //  Operator +
template TPLDecl cFormulaFD TPLInst  operator +(const cFormulaFD TPLInst & aF1 ,const cFormulaFD TPLInst & aF2);
template TPLDecl cFormulaFD TPLInst  operator +(const TypeElem & aV1,const cFormulaFD TPLInst & aF2);
template TPLDecl cFormulaFD TPLInst  operator +(const cFormulaFD TPLInst & aF1,const TypeElem & aV2);
      //  Operator *
template TPLDecl cFormulaFD TPLInst  operator *(const cFormulaFD TPLInst & aF1 ,const cFormulaFD TPLInst & aF2);
template TPLDecl cFormulaFD TPLInst  operator *(const TypeElem & aV1,const cFormulaFD TPLInst & aF2);
template TPLDecl cFormulaFD TPLInst  operator *(const cFormulaFD TPLInst & aF1,const TypeElem & aV2);
      //  Operator -
template TPLDecl cFormulaFD TPLInst  operator -(const cFormulaFD TPLInst & aF1 ,const cFormulaFD TPLInst & aF2);
template TPLDecl cFormulaFD TPLInst  operator -(const TypeElem & aV1,const cFormulaFD TPLInst & aF2);
template TPLDecl cFormulaFD TPLInst  operator -(const cFormulaFD TPLInst & aF1,const TypeElem & aV2);
      //  Operator /
template TPLDecl cFormulaFD TPLInst  operator /(const cFormulaFD TPLInst & aF1 ,const cFormulaFD TPLInst & aF2);
template TPLDecl cFormulaFD TPLInst  operator /(const TypeElem & aV1,const cFormulaFD TPLInst & aF2);
template TPLDecl cFormulaFD TPLInst  operator /(const cFormulaFD TPLInst & aF1,const TypeElem & aV2);

    // -------- Declare all unary operators  ----------------
template TPLDecl cFormulaFD TPLInst  square(const cFormulaFD TPLInst & aF);
template TPLDecl cFormulaFD TPLInst  exp(const cFormulaFD TPLInst & aF);
template TPLDecl cFormulaFD TPLInst  operator - (const cFormulaFD TPLInst & aF);


/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        Definition of all classes                * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

            //  -------------------  3 "Main" Classes  -------------------------
            //     cContextFormalDer  / cFormulaFD  / cImplemFormFD 
            // ----------------------------------------------------------------

template TPLDecl class cContextFormalDer : public cMemCheck
{
    public :

      // Result of several evaluation are stored in a buffer, Eigen vector are used 
      // as they implement efficiently arithmeticall operation
        typedef Eigen::Matrix<TypeElem, 1, TheSzBuf> tBuf;
        typedef cImplemFormFD  TPLInst               tIFD;
        typedef cFormulaFD TPLInst                   tFormulaFD;
        typedef std::map<std::string,tFormulaFD>     tDicoFunc;
        typedef std::map<TypeElem,tFormulaFD>        tDicoCste;
        typedef std::vector<std::string>             tVecId;

        friend tFormulaFD;


      // ---------------------------  Constructors -------------------
        /// Constructor with explicit Id for Unknown/Observation if we want to analyze the generated code
        inline cContextFormalDer(const tVecId & aVecUK,const tVecId & aVecObs);
        /// Constructor with basic Id (if we dont generate code, or dont want to analyse it by human)
        inline cContextFormalDer(int aNbUnknown,int aNbObservation);

      // ---------------------------  Accessors -------------------
        const std::vector<tFormulaFD>& VUK()  const {return  mVFuncUnknown;}  ///< Unknowns
        const std::vector<tFormulaFD>& VObs() const {return  mVFuncObs;}      ///< Observations



      // ---------------------------  Acces to function from names, values -------------------
        /// Indicate if the formula corresponding to a given string allreay exist
        inline bool  ExistFunc(const std::string & aName) const 
        {
              return (mDicoFunc.find(aName) != mDicoFunc.end());
        }
        /// Func of given name, Error if don't exist
        inline tFormulaFD FuncOfName(const std::string & aName) const ;
        /// Add a function (put it in dico), Error if already exist
        inline void AddFormula(tFormulaFD aPF)
        {
           if (ExistFunc(aPF->Name())) throw ("Multiple add of identic name :[" + aPF->Name() + "]");
           mDicoFunc[aPF->Name()] = aPF;
           mVAllFormula.push_back(aPF);
        }

        /// Func of given constant, create if don't exist
        inline tFormulaFD CsteOfVal(const TypeElem & aCste) ;
        tFormulaFD  Cste0() const  {return mCste0;}  ///< Acces to a current constant
        tFormulaFD  Cste1() const  {return mCste1;}  ///< Another Acces to a current constant
        tFormulaFD  Cste2() const  {return mCste2;}  ///< Yet another Acces to a current constant
        /// Tuning ---  Print the stack of function as a tree 
        inline void ShowStackFunc() const;


        size_t      NbCurFonc() const {return mVAllFormula.size();}
    private :
        cContextFormalDer(const cContextFormalDer &) = delete;  ///< Don't allow copy

        /// Used to generate automatically Id for Unknown/Observatio, when we dont need to control them explicitely
        static std::vector<std::string>   MakeAutomId(const std::string & aPrefix,int aNb);

        size_t                    mNbCste;   ///< Number Cste
        size_t                    mNbUK;  ///< Dim=number of unkown
        size_t                    mNbObs;   ///< Number of obserbation variable
        std::vector<tFormulaFD>   mVFuncUnknown; ///< Vector of All Unknowns
        std::vector<tFormulaFD>   mVFuncObs; ///< Vector of All Observations
        tDicoFunc                 mDicoFunc;  ///< Map Name => Func
        std::vector<tFormulaFD>   mVAllFormula;  ///< Vector of All Func, allow to parse them in creation order
        tDicoCste                 mDicoCste;  ///< Map  Value => Func Constant
        tFormulaFD                mCste0;     ///< Fonc constant null
        tFormulaFD                mCste1;     ///< Fonc constant 1
        tFormulaFD                mCste2;     ///< Fonc constant 1
};

template TPLDecl class cFormulaFD 
{
    public :
       typedef cContextFormalDer TPLInst     tContext;
       typedef typename tContext::tIFD       tIFD;
       typedef typename tContext::tFormulaFD tFormulaFD;
       typedef std::shared_ptr<tIFD>         tSPtrFuncFD;

       //  -------------------- constructor -------------------
             /// Construct from a pointer, standard 
       cFormulaFD  (tIFD  * aRawPtr) :
           mSPtr  (aRawPtr)
       {
       }
             /// Default constructor, required by some code (vector ?)
       cFormulaFD  ():
           cFormulaFD  TPLInst (nullptr)
       {
       }
       // --------------- operator on pointer ---------------------

       tIFD & operator*() const  {return *mSPtr;}  ///< Standard behaviour of a pointer
       tIFD * operator->() const {return mSPtr.operator ->();}  ///< Standard behaviour of a pointer
       bool IsNull() const {return mSPtr.get()==nullptr;} ///< Safer than giving acces to raw pointer

       // --------------- Naming  ---------------------

           /// Generate the unique indentifier of a binary expression
       std::string NameOperBin(const std::string & aNameOper,const tFormulaFD & aF2) const
       {
           return "F"+ std::to_string((*this)->NumGlob()) + aNameOper + "F"  + std::to_string(aF2->NumGlob());
       }

           /// Generate the unique indentifier of a unary expression
       std::string NameOperUn(const std::string & aNameOper) const
       {
           return  aNameOper + " F"  + std::to_string((*this)->NumGlob());
       }

    private :
       tSPtrFuncFD mSPtr;  ///< Smart/shared point on object 
};


template TPLDecl class cImplemFormFD  : public cMemCheck
{
    public :
      // See eigen documentation,  this macro is mandatory for alignment reason
       EIGEN_MAKE_ALIGNED_OPERATOR_NEW

       typedef TypeElem  tElem;
       static const int  vSzBuf = TheSzBuf;

       typedef cContextFormalDer TPLInst    tContext;
       typedef typename tContext::tBuf      tBuf;
       typedef typename tContext::tFormulaFD  tFormulaFD;

       friend tContext;

       virtual bool  IsCste0() const {return false;} ///< To redefine in constant func, Used for simplification in "/ * + -"
       virtual bool  IsCste1() const {return false;} ///< To redefine in constant func, Used for simplification in "/ *"
       // virtual void ComputeBuf() = 0;
       virtual tFormulaFD Derivate(int aK) const  = 0;
       // Used to print constant from generic pointer
       virtual const TypeElem * ValCste() const  {return nullptr;}

      
       /// Infixed "Pretty" Print .  For tuning and checking (reduction, derivative)
       virtual std::string  InfixPPrint() const =0; 

     // ---------- Accessors ---------------
       const std::string  & Name() const {return mName;}  ///< Standard accessor
       tContext *  Context() const {return mContext;}   ///< Standard accesor
       int  NumGlob() const {return mNumGlob;}  ///< Standard accessor

       virtual ~cImplemFormFD () {}   ///< Add a virtual ~X() when we have virtual methods
    protected :
       inline cImplemFormFD (tContext * aContext,const std::string & aName) :
              mContext (aContext),
              mName    (aName),
              mNumGlob (mContext->NbCurFonc())
       { }

       tBuf                   mBuf;       ///< Buf to store values
       tContext *             mContext;   ///< Context that manage all the funcion cooperating
       const std::string      mName;      ///< string represention of the formula like  for example "+C2*S6cos3.6"
       int                    mNumGlob;   ///< Global number (!= Num in class)
    private  :
       cImplemFormFD (const cImplemFormFD  &) = delete;
};

            //  -------------------  ATOMIC FORMULA -------------------------
            //     cFUnknownFormalDer / cFUnknownFormalDer /  cFUnknownFormalDer
            // ----------------------------------------------------------------

template TPLDecl class cFUnknownFormalDer : public cImplemFormFD  TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tContext  tContext;
            typedef typename tContext::tFormulaFD  tFormulaFD;

            std::string  InfixPPrint() const override {return tIFD::Name();}
            tFormulaFD Derivate(int aK) const override 
            {
                return (aK==mNumUnk) ? tIFD::mContext->Cste1() :  tIFD::mContext->Cste0();
            }

            friend tContext;
      private  :
            inline cFUnknownFormalDer(tContext * aContext,const std::string& aName,int aNum) :
                tIFD       (aContext,aName),
                mNumUnk    (aNum)
            { }

            int  mNumUnk;
};

template TPLDecl class cFObsFormalDer : public cImplemFormFD  TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tContext  tContext;
            typedef typename tContext::tFormulaFD  tFormulaFD;
            friend tContext;

            std::string  InfixPPrint() const override {return tIFD::Name();}
            tFormulaFD Derivate(int aK) const override {return tIFD::mContext->Cste0();}

      protected  :
            inline cFObsFormalDer(tContext * aContext,const std::string & aName,int aNum) : 
                  tIFD   (aContext,aName),
                  mNum   (aNum)
            { }
            int     mNum;
};

template TPLDecl class cFCsteFormalDer : public cImplemFormFD TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tContext  tContext;
            typedef typename tContext::tBuf  tBuf;
            typedef typename tContext::tFormulaFD  tFormulaFD;
            friend tContext;

            bool  IsCste0() const override {return mVal==0.0;} ///< Here we know if we are constant 0
            bool  IsCste1() const override {return mVal==1.0;} ///< Here we know if we are constant 1
            //  std::string  InfixPPrint() const override {return "C" + std::to_string(mVal);}
            std::string  InfixPPrint() const override {return tIFD::Name();}
            tFormulaFD Derivate(int aK) const override {return tIFD::mContext->Cste0();}
            const TypeElem * ValCste() const override  {return &mVal;}

      protected  :
            inline cFCsteFormalDer(tContext * aContext,const std::string & aName,int aNum,const TypeElem& aVal) : 
               tIFD   (aContext,aName),
               mNum   (aNum),
               mVal   (aVal)
            {
              tIFD::mBuf= tBuf::Constant(mVal); // We know the value of buf that will never change
            }
            int     mNum;
            const TypeElem mVal;
};
            //  -------------------  FORMULA ON UNARY OPERATOR ---------------
            //          MOTHER CLASS : cOpUnFormalDer
            //  cSquareFormalDer / cExpFormalDer
            // ----------------------------------------------------------------

template TPLDecl class cOpUnFormalDer : public cImplemFormFD  TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tContext  tContext;
            typedef typename tIFD::tFormulaFD  tFormulaFD;

            virtual std::string  NameOperator() const = 0;
            std::string  InfixPPrint() const override 
            {
               return NameOperator() + " "+  mF->InfixPPrint() ;
            }
      protected  :
            inline cOpUnFormalDer(tFormulaFD aF,const std::string & aName) :
                 tIFD  (aF->Context(),aName),
                 mF (aF)
            { }
            tFormulaFD  mF;
};


/*   Probably not more efficient than implementing Square as F*F, because derivation would give
     F'F + F'F  BUT would be reorder as F'F + F'F and unified ...
     By the way it was a test, if necessary replace Square by F*F */
template TPLDecl class cSquareFormalDer : public cOpUnFormalDer  TPLInst
{
     public :
            typedef cImplemFormFD  TPLInst     tIFD;
            typedef typename tIFD::tFormulaFD  tFormulaFD;
            typedef cOpUnFormalDer TPLInst     tO1;

            std::string  NameOperator() const override {return "square";}
            cSquareFormalDer (tFormulaFD aF,const std::string & aName) :
                tO1 (aF,aName)
            { }
      private :
            tFormulaFD Derivate(int aK) const override 
            {
                return  2.0  * tO1::mF->Derivate(aK)  * tO1::mF;
            }
};


template TPLDecl class cExpFormalDer : public cOpUnFormalDer  TPLInst
{
     public :
            typedef cImplemFormFD  TPLInst     tIFD;
            typedef typename tIFD::tFormulaFD  tFormulaFD;
            typedef cOpUnFormalDer TPLInst     tO1;

            std::string  NameOperator() const override {return "exp";}
            cExpFormalDer (tFormulaFD aF,const std::string & aName) :
                tO1 (aF,aName)
            { }
      private :
            tFormulaFD Derivate(int aK) const override 
            {
                return   tO1::mF->Derivate(aK)  * exp(tO1::mF);
            }
};

template TPLDecl class cMin1FormalDer : public cOpUnFormalDer  TPLInst
{
     public :
            typedef cImplemFormFD  TPLInst     tIFD;
            typedef typename tIFD::tFormulaFD  tFormulaFD;
            typedef cOpUnFormalDer TPLInst     tO1;

            std::string  NameOperator() const override {return "-";}
            cMin1FormalDer (tFormulaFD aF,const std::string & aName) :
                tO1 (aF,aName)
            { }
      private :
            tFormulaFD Derivate(int aK) const override 
            {
                return   - tO1::mF->Derivate(aK)  ;
            }
};


            //  -------------------  FORMULA ON BINARY OPERATOR ---------------
            //          MOTHER CLASS : cOpBinFormalDer
            //  cSumFormalDer / cMulFormalDer / cSubFormalDer / cDivFormalDer
            // ----------------------------------------------------------------

template TPLDecl class cOpBinFormalDer : public cImplemFormFD  TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tContext  tContext;
            typedef typename tIFD::tFormulaFD  tFormulaFD;

            virtual std::string  NameOperator() const = 0;
            std::string  InfixPPrint() const override 
            {
               return NameOperator() + "("+  mF1->InfixPPrint() + " " + mF2->InfixPPrint() + ")";
            }
      protected  :
            inline cOpBinFormalDer(tFormulaFD aF1,tFormulaFD aF2,const std::string & aName):
                 tIFD  (aF1->Context(),aName),
                 mF1 (aF1),
                 mF2 (aF2)
            {
                // It doesn't work to mix formula from different context
                assert(mF1->Context()==mF2->Context());
            }


            tFormulaFD  mF1;
            tFormulaFD  mF2;
};


template TPLDecl class cSumFormalDer : public cOpBinFormalDer TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tFormulaFD  tFormulaFD;
            typedef cOpBinFormalDer TPLInst    tO2;

            inline cSumFormalDer(tFormulaFD aF1,tFormulaFD aF2,const std::string & aName) :
                   tO2 (aF1,aF2,aName) 
            { }
      private  :
            tFormulaFD Derivate(int aK) const override {return  tO2::mF1->Derivate(aK) + tO2::mF2->Derivate(aK);}
            std::string  NameOperator() const override {return "+";}
};

template TPLDecl class cMulFormalDer : public cOpBinFormalDer TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tFormulaFD  tFormulaFD;
            typedef cOpBinFormalDer TPLInst    tO2;

            inline cMulFormalDer(tFormulaFD aF1,tFormulaFD aF2,const std::string & aName) :
                   tO2 (aF1,aF2,aName)
            { }
      private  :
            tFormulaFD Derivate(int aK) const override 
            {
                return  tO2::mF2*tO2::mF1->Derivate(aK) + tO2::mF1*tO2::mF2->Derivate(aK);
            }
            std::string  NameOperator() const override {return "*";}
};

template TPLDecl class cSubFormalDer : public cOpBinFormalDer TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tFormulaFD  tFormulaFD;
            typedef cOpBinFormalDer TPLInst    tO2;

            inline cSubFormalDer(tFormulaFD aF1,tFormulaFD aF2,const std::string & aName) :
                   tO2 (aF1,aF2,aName)
            { }
      private  :
            tFormulaFD Derivate(int aK) const override {return tO2::mF1->Derivate(aK) - tO2::mF2->Derivate(aK);}
            std::string  NameOperator() const override {return "-";}
};

template TPLDecl class cDivFormalDer : public cOpBinFormalDer TPLInst
{
      public :
            typedef cImplemFormFD  TPLInst   tIFD;
            typedef typename tIFD::tFormulaFD  tFormulaFD;
            typedef cOpBinFormalDer TPLInst    tO2;

            inline cDivFormalDer(tFormulaFD aF1,tFormulaFD aF2,const std::string & aName) :
                   tO2 (aF1,aF2,aName)
            { }
      private  :
            tFormulaFD Derivate(int aK) const override 
            {
               return (tO2::mF1->Derivate(aK)* tO2::mF2 - tO2::mF2->Derivate(aK) * tO2::mF1) / square(tO2::mF2);
            }
            std::string  NameOperator() const override {return "/";}
};



// template TPLDecl class cDivFormalDer ;     ///< Class for division of 2 functions
/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        External Definition of methods           * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */





      /* ----------------------------------------------*/
      /*             cContextFormalDer                 */
      /* ----------------------------------------------*/

template TPLDecl std::vector<std::string> cContextFormalDer TPLInst::MakeAutomId(const std::string & aPrefix,int aNb)
{
      tVecId aRes;
      for (int aK=0 ; aK<aNb ; aK++)
          aRes.push_back(aPrefix+ std::to_string(aK));
      return aRes;
}

template TPLDecl cContextFormalDer TPLInst::cContextFormalDer(const tVecId & aVNameUK,const tVecId & aVNameObs) :
    mNbCste (0),
    mNbUK   (aVNameUK.size()),
    mNbObs  (aVNameObs.size()),
    mCste0  (CsteOfVal(0.0)),
    mCste1  (CsteOfVal(1.0)),
    mCste2  (CsteOfVal(2.0))
{
    // Generate all the function corresponding to unknown
    for (size_t aNumUK=0 ; aNumUK<mNbUK ; aNumUK++)
    {
        tFormulaFD aFuncUK(new cFUnknownFormalDer TPLInst(this,aVNameUK[aNumUK],aNumUK));  // Create it
        mVFuncUnknown.push_back(aFuncUK);   // Push it in vector of coordinat func
        AddFormula(aFuncUK);  // Add to all func
    }

    // Generate all the function corresponding to observations
    for (size_t aNumObs=0 ; aNumObs<mNbObs ; aNumObs++)
    {
        tFormulaFD aFuncObs(new cFObsFormalDer TPLInst(this,aVNameObs[aNumObs],aNumObs));  // Create it
        mVFuncObs.push_back(aFuncObs);   // Push it in vector of coordinat func
        AddFormula(aFuncObs);  // Add to all func
    }
}

template TPLDecl cContextFormalDer TPLInst::cContextFormalDer(int aNbUK,int aNbObs) :
     cContextFormalDer TPLInst(MakeAutomId("X",aNbUK),MakeAutomId("V",aNbObs))
{
}

template TPLDecl cFormulaFD TPLInst cContextFormalDer TPLInst::CsteOfVal(const TypeElem & aCste) 
{
  tFormulaFD & aRef = mDicoCste[aCste];
  if (aRef.IsNull())  // If it was not existing, the map contain now the def element
  {
     aRef=tFormulaFD(new cFCsteFormalDer TPLInst(this,"C"+std::to_string(mNbCste),mNbCste,aCste));
     mNbCste++;
     AddFormula(aRef);
  }

  return aRef;
}

template TPLDecl cFormulaFD TPLInst cContextFormalDer TPLInst::FuncOfName(const std::string & aName) const 
{
    const auto & anIt = mDicoFunc.find(aName);
    if (anIt == mDicoFunc.end()) throw ("Try to acces non existing name :[" + aName + "]");
    return anIt->second;
}

template TPLDecl void cContextFormalDer TPLInst::ShowStackFunc() const
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

/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *        Global function                          * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

      /* ----------------------------------------------*/
      /*     Binary operator between formulas          */
      /* ----------------------------------------------*/

template <class TypeCompiled>  class cGenOperatorBinaire
{
    public :
         typedef typename TypeCompiled::tIFD     tIFD;
         typedef typename tIFD::tFormulaFD  tFormulaFD;
         static tFormulaFD   Generate(tFormulaFD aF1,tFormulaFD aF2,const std::string & aNameOp)
         {
             // Extract context (take F1 ou F2, does not matter, they must be the same)
             auto aPCont = aF1->Context();  
             std::string aNameForm =  aF1.NameOperBin(aNameOp,aF2);

             if (aPCont->ExistFunc(aNameForm))
               return aPCont->FuncOfName(aNameForm);

             tFormulaFD aResult (new TypeCompiled(aF1,aF2,aNameForm));
             aPCont->AddFormula(aResult);
             return aResult;
         }
};

       
template TPLDecl cFormulaFD TPLInst  operator +(const cFormulaFD TPLInst & aF1,const cFormulaFD TPLInst & aF2) 
{
     // Use the fact that 0 is neutral element to simplify
     if (aF1->IsCste0()) return aF2;
     if (aF2->IsCste0()) return aF1;

     // Use commutativity of + to have a unique representation
     if (aF1->Name() > aF2->Name()) 
        return aF2+aF1;

     return cGenOperatorBinaire<cSumFormalDer<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"+");
}

template TPLDecl cFormulaFD TPLInst  operator -(const cFormulaFD TPLInst & aF1,const cFormulaFD TPLInst & aF2) 
{
     // Use the fact that 0 is neutral element to simplify
     if (aF1->IsCste0()) return -aF2;
     if (aF2->IsCste0()) return aF1;

     return cGenOperatorBinaire<cSubFormalDer<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"-");
}

template TPLDecl cFormulaFD TPLInst  operator *(const cFormulaFD TPLInst & aF1,const cFormulaFD TPLInst & aF2) 
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

     return cGenOperatorBinaire<cMulFormalDer<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"*");
}

template TPLDecl cFormulaFD TPLInst  operator /(const cFormulaFD TPLInst & aF1,const cFormulaFD TPLInst & aF2) 
{
     if (aF1->IsCste0()) return aF1;  // 0/F2 = 0
     if (aF2->IsCste1()) return aF1;  // F1/1 = F1

     return cGenOperatorBinaire<cDivFormalDer<TypeElem,TheSzBuf> >::Generate(aF1,aF2,"/");
}


      /* ----------------------------------------------------------*/
      /*  Binary   Operator  between Formula and  constants        */
      /* ----------------------------------------------------------*/

        // ++++++++++++++++++++++++
template TPLDecl inline cFormulaFD TPLInst  operator +(const TypeElem & aV1,const cFormulaFD TPLInst & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) + aF2;
}
template TPLDecl inline cFormulaFD TPLInst  operator +(const cFormulaFD TPLInst & aF1,const TypeElem & aV2)
{
    return aV2+aF1;
}
        // ************************
template TPLDecl inline cFormulaFD TPLInst  operator *(const TypeElem & aV1,const cFormulaFD TPLInst & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) * aF2;
}
template TPLDecl inline cFormulaFD TPLInst  operator *(const cFormulaFD TPLInst & aF1,const TypeElem & aV2)
{
    return aV2*aF1;
}
        // ************************
template TPLDecl inline cFormulaFD TPLInst  operator -(const TypeElem & aV1,const cFormulaFD TPLInst & aF2)
{
  return aF2->Context()->CsteOfVal(aV1) - aF2;
}
template TPLDecl inline cFormulaFD TPLInst  operator -(const cFormulaFD TPLInst & aF1,const TypeElem & aV2)
{
  return aF1-aF1->Context()->CsteOfVal(aV2) ;
}



      /* ---------------------------------------*/
      /*           Unary   Operator             */
      /* ---------------------------------------*/

template <class TypeCompiled>  class cGenOperatorUnaire
{
    public :
         typedef typename TypeCompiled::tIFD     tIFD;
         typedef typename tIFD::tFormulaFD  tFormulaFD;
         static tFormulaFD   Generate(tFormulaFD aF,const std::string & aNameOp)
         {
             auto aPCont = aF->Context();  
             std::string aNameForm = aF.NameOperUn(aNameOp);

             if (aPCont->ExistFunc(aNameForm))
               return aPCont->FuncOfName(aNameForm);

             tFormulaFD aResult (new TypeCompiled(aF,aNameForm));
             aPCont->AddFormula(aResult);
             return aResult;
         }
};
    
template TPLDecl inline cFormulaFD TPLInst  square(const cFormulaFD TPLInst & aF)
{
    return cGenOperatorUnaire<cSquareFormalDer<TypeElem,TheSzBuf> >::Generate(aF,"square");
}

template TPLDecl inline cFormulaFD TPLInst  exp(const cFormulaFD TPLInst & aF)
{
    return cGenOperatorUnaire<cExpFormalDer<TypeElem,TheSzBuf> >::Generate(aF,"exp");
}
template TPLDecl inline cFormulaFD TPLInst  operator - (const cFormulaFD TPLInst & aF)
{
    return cGenOperatorUnaire<cMin1FormalDer<TypeElem,TheSzBuf> >::Generate(aF,"-");
}

      /* ----------------------------------------------*/
      /*                     TEST                      */
      /* ----------------------------------------------*/
static const int SzBufTest = 8;
typedef  double TypeTest;
typedef  cFormulaFD <TypeTest,SzBufTest>  tFormulaTest;


void   BenchFormalDer()
{
    {
       cContextFormalDer<TypeTest,SzBufTest>  aCFD(3,5);

       tFormulaTest  X0 = aCFD.VUK().at(0);
       tFormulaTest  X1 = aCFD.VUK().at(1);
       tFormulaTest  X2 = aCFD.VUK().at(2);

       tFormulaTest  aF = (X0+X1) * (X0 +square(X2)) - exp(-square(X0));
       // tFormulaTest  aF = X0 * X0;
       tFormulaTest  aFd0 = aF->Derivate(0);

       std::cout << "F=" << aF->InfixPPrint() << "\n";
       std::cout << "Fd=" << aFd0->InfixPPrint() << "\n";

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

       // cFormulaFD<TypeTest,SzBufTest> aPtr(nullptr);
       // aPtr->IsCste0();
       

       // std::shared_ptr<cFuncFormalDer <8,double> > aF1  =
       
        // const std::vector<tFormulaFD>& VUK()  const {return  mVFuncUnknown;}  ///< Unknowns
        // const std::vector<tFormulaFD>& VObs() const {return  mVFuncObs;}      ///< Observations
    }
    // new cContextFormalDer<double,100> (3,5);

    int i=10;
    std::string aStr = "i="+ std::to_string(i);
    std::cout  << "BenchFormalDerBenchFormalDerBenchFormalDer " << aStr << "\n";

    getchar();
}


#endif // _MMVII_DynAndStatFormalDerivation_H_
