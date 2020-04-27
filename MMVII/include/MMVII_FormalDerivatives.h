#ifndef _MMVII_FormalDerivative_H_
#define _MMVII_FormalDerivative_H_
using namespace std;

#ifdef _OPENMP
#include <omp.h>
#endif


#define WITH_MMVII false
#define WITH_EIGEN false


#if WITH_EIGEN
#include "ExternalInclude/Eigen/Dense"  // TODO => replace with standard eigen file
#define EIGEN_ALLIGNMENT_IN_MMVII EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#else
#define EIGEN_ALLIGNMENT_IN_MMVII 
#endif
/*
*/

/** \file MMVII_FormalDerivate.h
    \brief File for generating formal derivate

   Classes for generated formal derivative.  All classes are single template classes. 
   The template parameter indicate the numerical type used for storage/computation 
   (float, double ...)

   This file is the only file to include. It contains :

      * declaration of operators
      * definition of "main" classes : cFormula , cCoordinatorF ,  cImplemF "  ;  
      * the 3 class for Atomic formula who will (probably) stay the same : Unkown, Observation, Constants

   This file include 2 files corresponding to following type of formula :

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
             
       cFormula<Type> is no more than an encapsulation of a pointer on the "concrete" class cImplemF.

      * cImplemF<Type> : is the mother class of all the formula. It's a pure abstract class, it contains
        several pure virtual methods. The two main methods are "Derivate"  and "ComputeBuf", this is
        the two methods the users will have to define when extension to the library with new
        operator is required.

          -  cFormula<Type> Derivate(int aK) return the formula of its derivate by Xk. Heres is
            two example extract from the code, one for multiplication, other from unknowns :

              o return  mF2*mF1->Derivate(aK) + mF1*mF2->Derivate(aK); // From cMulF : (FG)' = F'G + FG'
              o return (aK==mNumUnk) ? tImplemF::mCoordF->Cste1() :  tImplemF::mCoordF->Cste0(); // from cUnknownF


           -  void ComputeBuf(int aK0,int aK1) : update the buffer of its data, once it subformula has
             been updated, this is method that does the real job. Here an extract from  cExpF and cDivF :

              o  for (int aK=aK0 ; aK<aK1 ; aK++) mDataBuf[aK] = std::exp(mDataF[aK]);
              o  for (int aK=aK0 ; aK<aK1 ; aK++) mDataBuf[aK] =  mDataF1[aK] / mDataF2[aK];

*/



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
#include <cassert>
#include "memory.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <list>
#include <map>
#include <ctime>
#include <chrono>
#include <math.h>   
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#endif            //========================================================== WITH_MMVI

// REDUCTION RULES
// TODO => REPLACE BY METHOD ON COORDINATOR WHEN THEY IMPROVE THINGS ....
#define  DOREDUCE      false

#define  REDUCE_CSTE    true    //  Cste+Cste => cste
#define  REDUCE_MM      DOREDUCE    //  - - x => x  ;  a-(-b) => a+b
#define  REDUCE_ASSOCP  DOREDUCE       /*  B + (A + C) = > A + ( B + C),  
      more generally order the + operator, could be done with '*'  */
#define  REDUCE_DISTRIB DOREDUCE    //  A#B ~ A#C=> A#(B~C)  ; # in "*/" and   ~ in "+-"
#define  REDUCE_ApA     DOREDUCE    //  A+A => 2*A, not good by itself, but may creat other reduc
#define  REDUCE_DIST1   DOREDUCE    //  A + A*C => A *(1+C) si C est csteto have all constant close
static inline void SHOW_REDUCE(const std::string & aMes) {} // std::cout << "REDUCE " << aMes << "\n";}


namespace  NS_MMVII_FormalDerivative
{

/* *************************************************** */
/*                                                     */
/*  P0-Definition of global functions                  */
/*                                                     */
/* *************************************************** */


/// The CreateCste is required for formula, so we need it also on num type
template <class Type> inline Type CreateCste(const Type  & aV,const Type &) { return aV; }

/// because pow is defined in std and there is cast int->float that would make it unaccessible
template <class Type> inline Type pow(const Type & aV,const int & aExp)    
{
    return std::pow(aV,Type(aExp));
}

/* These functions are required if we want to have same operation on numbers double and formulas
   They are suposed to be optimized implementation of pow for integer low value
   of the exponent 
*/
template <class Type> inline Type square(const Type & aV)  {return aV*aV;}
template <class Type> inline Type cube(const Type & aV)    {return aV*aV*aV;}
template <class Type> inline Type pow4(const Type & aV)    {return square(square(aV));}
template <class Type> inline Type pow5(const Type & aV)    {return aV *pow4(aV);}
template <class Type> inline Type pow6(const Type & aV)    {return square(cube(aV));}
template <class Type> inline Type pow7(const Type & aV)    {return aV *pow6(aV);}
template <class Type> inline Type pow8(const Type & aV)    {return square(pow4(aV));}
template <class Type> inline Type pow9(const Type & aV)    {return aV *pow8(aV);}


      //============= BASIC  ERROR HANDLING ==============


static inline void Error(const std::string & aMes,const std::string & aExplanation)
{
    std::cout << "In MMVII_FormalDerivative a fatal error" << "\n";
    std::cout << "  Likely Source   ["<< aExplanation << "\n";
    std::cout << "  Message  ["<< aMes << "]\n";
    assert(false);
}
     ///    Error due probably to internal mistake
static inline void InternalError(const std::string & aMes)
{
   Error(aMes,"Internal Error of the Library");
}
     /// Error probably due to bas usage of the library (typically out limit vector access)
static inline void UserSError(const std::string & aMes)
{
   Error(aMes,"Probable error on user's side due to unapropriate usage of the library");
}

     /// Check equality in test, taking account numericall error
static inline void AssertAlmostEqual(const double & aV1,const double & aV2,const double & aEps)
{
   if ( (std::abs(aV1-aV2)> aEps*(std::abs(aV1)+std::abs(aV2))) )
      InternalError("Test equality failed");
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
          //  "Number x Formula"    and "Formula x Number" , the two last are rather 
          // syntactic suggar (i.e. make usage easier, but do not extend the library power)
          

      //  Operator +
template <class TypeElem> cFormula <TypeElem>  
         operator +(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  operator +(const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  operator +(const cFormula <TypeElem> & aF1,const TypeElem & aV2);
      //  Operator *
template <class TypeElem> cFormula <TypeElem>  
         operator *(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  operator *(const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  operator *(const cFormula <TypeElem> & aF1,const TypeElem & aV2);
      //  Operator -
template <class TypeElem> cFormula <TypeElem>  
         operator -(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  operator -(const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  operator -(const cFormula <TypeElem> & aF1,const TypeElem & aV2);
      //  Operator /
template <class TypeElem> cFormula <TypeElem>  
         operator /(const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  operator /(const TypeElem & aV1,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  operator /(const cFormula <TypeElem> & aF1,const TypeElem & aV2);
      //  pow
template <class TypeElem> cFormula <TypeElem>  
         pow (const cFormula <TypeElem> & aF1 ,const cFormula <TypeElem> & aF2);
template <class TypeElem> cFormula <TypeElem>  pow (const TypeElem & aV1,const cFormula <TypeElem> & aF2);
           /// This one defined in MMVII_FormDer_UnaryOp.h
template <class TypeElem> cFormula <TypeElem>  pow (const cFormula <TypeElem> & aF1,const TypeElem & aV2);
template <class TypeElem> cFormula <TypeElem>  pow (const cFormula <TypeElem> & aF1,const int & aV2);


    // -------- integer low power  ----------------
template <class TypeElem> cFormula <TypeElem>  square(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  cube(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  pow4(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  pow5(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  pow6(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  pow7(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  pow8(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  pow9(const cFormula <TypeElem> & aF);

    // --- other unary operator
template <class TypeElem> cFormula <TypeElem>  exp(const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  operator - (const cFormula <TypeElem> & aF);
template <class TypeElem> cFormula <TypeElem>  log(const cFormula <TypeElem> & aF);

     // ---- sometime we need a templetized way to create constants
template <class T>  cFormula<T> CreateCste(const T & aV,const cFormula<T> & aF);


     /// --- powI , return pow of integral exponent,

template <class Type> Type powI(const Type & aV,const int & aExp)
{
   switch (aExp)
   {
        // case 0 : return Type(1.0);
        case 0 : return CreateCste(1.0,aV);
        case 1 : return aV;
        case 2 : return square(aV);
        case 3 : return cube(aV);
        case 4 : return pow4(aV);
        case 5 : return pow5(aV);
        case 6 : return pow6(aV);
        case 7 : return pow7(aV);
        case 8 : return pow8(aV);
        case 9 : return pow9(aV);
   }
   // else use the classical pow
   return pow(aV,aExp);
}

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
        const std::vector<tFormula>& VUk()  const {return  mVFormUnknowns;}     ///< Unknowns
        const std::vector<tFormula>& VObs() const {return  mVFormObservations;} ///< Observations

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
         const TypeElem &  ValComp(int aNumPush,int aKElem) const
         {
             return  mBufRes.at(aNumPush)->at(mSzInterval*aKElem);
         }
         /// Retur value of derivate computed taking into account order of storage
         const TypeElem &  DerComp(int aNumPush,int aKElem,int aKVarDer) const
         {
             if (! mWithDer)  UserSError("Acces to derivate wich were not computed");
             return  mBufRes.at(aNumPush)->at(mSzInterval*aKElem +1 + aKVarDer);
         }
        
         // ---------- Code generator ---------------
         /** Generate code, class cName  , file cName.h, cName.cpp */
         std::vector<std::string> GenerateCode(const std::string & Name) const { return {genCodeNAddr(Name),genCodeDevel(Name)}; }
         std::string genCodeNAddr(const std::string &formulaName) const;
         std::string genCodeDevel(const std::string &formulaName) const;


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
           aPF->TryReducAssoc();
        }

        /// Func of given constant, create if don't exist
        inline tFormula CsteOfVal(const TypeElem & aCste) ;
        tFormula  Cste0() const  {return mCste0;}  ///< Acces to a current constant
        tFormula  Cste1() const  {return mCste1;}  ///< Another Acces to a current constant
        tFormula  Cste2() const  {return mCste2;}  ///< Yet another Acces to a current constant
        /// Tuning ---  Print the stack of function as a tree 
        inline void ShowStackFunc() const;
        /// Formula used for computation, 
        const std::vector<tFormula>& VReached() const {return  mVReachedF;} 


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
       //  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
       EIGEN_ALLIGNMENT_IN_MMVII 

       typedef TypeElem                        tElem;
       typedef cCoordinatorF<TypeElem>         tCoordF;
       typedef typename tCoordF::tBuf          tBuf;
       typedef typename tCoordF::tFormula      tFormula;

       //----------- For derivation and reduction--------------
       virtual bool  IsCste(const TypeElem &) const {return false;} ///< To redefine in constant func, Used for simplification in "/ * + -"
        virtual bool IsDistribInt() const {return false;} ///< To redefine in *,/ for distributivity

       virtual tFormula Derivate(int aK) const  = 0;  ///< Compute the formula of it's derivative to Kth unknown

        /**  In this functionwe try to make reduction using associativity (and maybe others),
             as we want to do it only on maximal chains of + (or *) this has to be run by the father of
             the chain
        */
        void TryReducAssoc();
        virtual cImplemF<TypeElem> * ReducAssoc() {return this;}
        virtual bool  IsMult()  const {return false;}
        virtual bool  IsSum()   const {return false;}
        bool  ReducAssocTried() const {return mReducAssocTried;}
        virtual cFormula<TypeElem> VOper2(const tFormula &,const tFormula &) const; ///< Use in distributive reducion to recal the operator binaire if suitable

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
       void SetDepth(int aDepth) {mDepth = aDepth;}  ///< Fix Reached

       // ---------- Code gen -----------------------
       virtual bool isAtomic() const { return false;}
       virtual std::string  genCodeFormName() const {return NameGlob();} // Name of formula, referenced value for Atomic
       virtual std::string  genCodeNAddr() const = 0;      // N-Addresses code generation
       virtual std::string  genCodeDef() const = 0;        // Formula definition generation
       virtual std::string  genCodeRef() const;            // Formula reference generation
       int usedCnt() const {return mUsedCnt;}  ///< Standard accessor

     // ---------- Tuning / Debugging / Analysing ---------------
       /// Used to print constant from generic formula
       virtual const TypeElem * ValCste() const  {return nullptr;}
       /// Infixed "Pretty" Print .  For tuning and checking (i.e correction of reduction, derivative, rewrite ...)
       virtual std::string  InfixPPrint() const =0; 
       /// Number of reference that would occur without reduction on identic formula (to test performance in paper)
       int RecursiveRec() const;

     // Every where a reference name is needed
       std::string NameGlob() const { return "F" + std::to_string(NumGlob());}

       /// Access at global level is 4 reducing, also it is used 4 implemant in Unary & Binary
       virtual const std::string &  NameOperator() const = 0;


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
              mDepth   (-1),
              mUsedCnt (0),
              mReducAssocTried (false)
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
       unsigned               mUsedCnt;
       bool  mReducAssocTried;
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
       tImplemF * RawPtr() const {return mPtr;}  ///< Explicit acces
       // DO NOT WORK  const std::unique_ptr<tImplemF>  operator->() const {return std::unique_ptr<mPtr>;} 
       bool IsNull() const {return mPtr==nullptr;} ///< Safer than giving acces to raw pointer

       // --------------- Naming  ---------------------

           /// Generate the unique indentifier of a binary expression
       std::string NameFormulaBin(const std::string & aNameOper,const tFormula & aF2) const
       {
           return (*this)->NameGlob() + aNameOper + aF2->NameGlob();
       }

           /// Generate the unique indentifier of a unary expression
       std::string NameFormulaUn(const std::string & aNameOper) const
       {
           return aNameOper + " " + (*this)->NameGlob();
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
            bool isAtomic() const override { return true;}
            std::string genCodeFormName() const override { return this->Name();}
            std::string genCodeNAddr() const override { return this->genCodeFormName();}
            std::string genCodeRef() const override { return this->genCodeFormName();}
            std::string genCodeDef() const override { return mCodeValue;}

            inline cAtomicF(tCoordF * aCoordF,const std::string& aName) :
                tImplemF       (aCoordF,aName)
            { }
            std::string mCodeValue;
};

template <class TypeElem> class cUnknownF : public cAtomicF<TypeElem>
{
      public :
            typedef cAtomicF<TypeElem>  tAtom;
            typedef typename tAtom::tImplemF     tImplemF;
            typedef typename tImplemF::tCoordF  tCoordF;
            typedef typename tCoordF::tFormula  tFormula;

            const std::string &  NameOperator() const override {static std::string s("UK"); return s;}

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
            {
                  this->mCodeValue =  "this->vvUk[aK][" + std::to_string(mNumUnk) + "]";
            }

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

            const std::string &  NameOperator() const override {static std::string s("Obs"); return s;}
      private  :
            inline cObservationF(tCoordF * aCoordF,const std::string & aName,int aNum) : 
                  tAtom  (aCoordF,aName),
                  mNum   (aNum)
            {
                  this->mCodeValue =  "this->vvObs[aK][" + std::to_string(mNum) + "]";
            }
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

            bool  IsCste(const TypeElem &K) const override {return mVal==K;} ///< Here we know if we are a constant of value K
            const TypeElem * ValCste() const override  {return &mVal;}
            const std::string &  NameOperator() const override {static std::string s("Cste"); return s;}
      protected  :
            inline cConstantF(tCoordF * aCoordF,const std::string & aName,int aNum,const TypeElem& aVal) : 
               tAtom   (aCoordF,aName),
               mNum    (aNum),
               mVal    (aVal)
            {
               for (auto & aV : tImplemF::mBuf) aV = aVal;  // Initialize buf  with const val
               std::stringstream ss;
              // Precision that ensures that Num0 -> ASCII -> Num1 => Num1 == Num0
              // May cause some odd but correct value for non exactly representable numbers
               ss << std::setprecision(std::numeric_limits<decltype(mVal)>::max_digits10) << mVal;
               this->mCodeValue =  ss.str();
            }
            std::string genCodeFormName() const override { return this->mCodeValue;}
            int     mNum;
            const TypeElem mVal;
};


/* *************************************************** */
/* *************************************************** */
/* *                                                 * */
/* *     cFormula / cImplemF / cCoordinatorF         * */
/* *        External Definition of methods           * */
/* *                                                 * */
/* *************************************************** */
/* *************************************************** */

      /* ---------------------- */
      /*        cFormula        */
      /* ---------------------- */

template <class T>  cFormula<T> CreateCste(const T & aV,const cFormula<T> & aF)
{
    return aF->CoordF()->CsteOfVal(aV);
}


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
   if (mDepth != -1)  {
       mUsedCnt++;
       return; // if we were already here , nothing to do
   }
   mUsedCnt = 1;
   for (const auto  & aF : Ref())
   {
      aF->CalcRecursiveDepth(aVReached); // parse sub formula
      mDepth = std::max(mDepth,aF->mDepth); // Memo max depth
   }
   mDepth++; // my depth is 1 + max of depth of my referenced formulas
   aVReached.push_back(tFormula(this));
}

template <class TypeElem> void cImplemF<TypeElem>::TryReducAssoc()
{
   for (auto  & aF : Ref())
   {
       // F will not belong to the terminal command that will have to reparsed
       // If we are in the config  (A+B) + .. maybe the chain will grow later
       if (aF->NameOperator() != NameOperator())
       {
          aF = aF->ReducAssoc();
       }
       aF->mReducAssocTried = true;
   }
}

template <class TypeElem> cFormula<TypeElem> cImplemF<TypeElem>::VOper2(const tFormula & aF1,const tFormula &) const
{
   InternalError("Uncorrect virtula binary operation");
   return aF1;
}


template <class TypeElem>
std::string  cImplemF<TypeElem>::genCodeRef() const
{
    if (usedCnt() == 1) {
        return genCodeDef();
    } else {
        return genCodeFormName();
    }
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
     // The ! is used to make constant first in alphab order, used for reduction ?
     aRef=tFormula(new cConstantF<TypeElem>(this,"_C"+std::to_string(mNbCste),mNbCste,aCste));
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
void cCoordinatorF<TypeElem>::SetCurFormulas(const std::vector<tFormula> & aVF0)
{
    std::vector<tFormula> aVF;
    for(auto aF : aVF0)
    {
       if (! aF->ReducAssocTried())
       {
          aF = tFormula(aF->ReducAssoc());
          // std::cout << "GGGGGGGG " << aF->Name() << " \n";
       }
       aVF.push_back(aF);
    }
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
    // In fact it is probably not necessary to make this sort, initial order of reaching order
    // should work;  by the way : no dammage ..
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
#ifdef _OPENMP
#pragma omp parallel
    {
        size_t thread_num = omp_get_thread_num();
        size_t num_threads = omp_get_num_threads();
        size_t start = thread_num * mNbInBuf / num_threads;
        size_t end = (thread_num + 1) * mNbInBuf / num_threads;

        if (end>start)
        {
            for (auto & aF : mVReachedF)
            {
                aF->ComputeBuf(start,end);
            }
        }
    }
#else
    for (auto & aF : mVReachedF)
    {
       aF->ComputeBuf(0,mNbInBuf);
    }
#endif

    mBufRes.clear();
    for (size_t aKLine=0 ; aKLine<mNbInBuf ;  aKLine++)
    {
        std::vector<TypeElem> & aLine  = mBufLineRes[aKLine];
        mBufRes.push_back(&aLine);
        for (size_t aKFunc=0 ; aKFunc< mVCurF.size() ; aKFunc++)
            aLine[aKFunc] = mVCurF[aKFunc]->GetBuf(aKLine);
    }
    mNbInBuf = 0;
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

       std::cout << aForm->usedCnt() << "- ";
       std::cout << aForm->NameGlob() << " => " << aForm->Name();

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

template <class TypeElem>
std::string cCoordinatorF<TypeElem>::genCodeNAddr(const std::string &formulaName) const
{
    std::string className  = formulaName + "NAddr";
    std::string fileName  = "CodeGen_" + className + ".h";
    std::ofstream os(fileName);

    std::string parentClass =
            "GenFuncTpl<TypeElem," +
            std::to_string(mNbUK) + "," +
            std::to_string(mNbObs) + "," +
            std::to_string(mVCurF.size()) +  "," +
            std::to_string(mSzInterval) + ">" ;

    os << "#include <vector>\n"
          "#ifdef _OPENMP\n"
          "#include <omp.h>\n"
          "#endif\n"
          "#include \"include/MMVII_FormDer_CGenTpl.h\"\n"
          "\n"
          "namespace CodeGen {\n\n"
          "template<typename TypeElem>\n"
          "class " << className << " : public " << parentClass << "\n"
          "{\n"
          "public:\n"
          "    " << className << "(size_t szBuf) : " << parentClass << "(szBuf) {}\n"
          "    static std::string FormulaName() { return \"" << formulaName << "\";}\n"
          "    void evalAndClear();\n"
          "};\n"
          "\n"
          "template<typename TypeElem>\n"
          "void " << className << "<TypeElem>::evalAndClear()\n"
          "{\n"
          "#ifdef _OPENMP\n"
          "#pragma omp parallel for\n"
          "#endif\n"
          "  for (size_t aK=0; aK < this->mInBuf; aK++) {\n"
          "// Declare local vars in loop to make them per thread\n";

    for (auto & aForm : mVFormUnknowns)
        os << "    TypeElem &" << aForm->genCodeFormName() << " = " << aForm->genCodeDef() << ";\n";
    for (const auto & aForm : mVFormObservations)
        os << "    TypeElem &" << aForm->genCodeFormName() << " = " << aForm->genCodeDef() << ";\n";

    for (const auto & aForm : mVReachedF) {
        if (!aForm->isAtomic())
            os << "    TypeElem " << aForm->genCodeFormName() << " = " << aForm->genCodeNAddr() << ";\n";
    }

    for (size_t i=0; i<mVCurF.size(); i++)
       os <<  "    this->vvRes[aK][" << i << "] = " << mVCurF[i]->genCodeFormName() << ";\n";
    os << "  }\n"
          "  this->mInBuf=0;\n"
          "}\n\n"
          "} // namespace CodeGen\n";
    return fileName;
}

template <class TypeElem>
std::string cCoordinatorF<TypeElem>::genCodeDevel(const std::string &formulaName) const
{
    std::string className  = formulaName;
    std::string fileName  = "CodeGen_" + className + ".h";
    std::ofstream os(fileName);

    std::string parentClass =
            "GenFuncTpl<TypeElem," +
            std::to_string(mNbUK) + "," +
            std::to_string(mNbObs) + "," +
            std::to_string(mVCurF.size()) + "," +
            std::to_string(mSzInterval) + ">" ;

    os << "#include <vector>\n"
          "#ifdef _OPENMP\n"
          "#include <omp.h>\n"
          "#endif\n"
          "#include \"include/MMVII_FormDer_CGenTpl.h\"\n"
          "\n"
          "namespace CodeGen {\n\n"
          "template<typename TypeElem>\n"
          "class " << className << " : public " << parentClass << "\n"
          "{\n"
          "public:\n"
          "    " << className << "(size_t szBuf) : " << parentClass << "(szBuf) {}\n"
          "    static std::string FormulaName() { return \"" << formulaName << "\";}\n"
          "    void evalAndClear();\n"
          "};\n"
          "\n"
          "template<typename TypeElem>\n"
          "void " << className << "<TypeElem>::evalAndClear()\n"
          "{\n"
          "#ifdef _OPENMP\n"
          "#pragma omp parallel for\n"
          "#endif\n"
          "  for (size_t aK=0; aK < this->mInBuf; aK++) {\n"
          "// Declare local vars in loop to make them per thread\n";


    for (auto & aForm : mVFormUnknowns)
        os << "    TypeElem &" << aForm->genCodeFormName() << " = " << aForm->genCodeDef() << ";\n";
    for (const auto & aForm : mVFormObservations)
        os << "    TypeElem &" << aForm->genCodeFormName() << " = " << aForm->genCodeDef() << ";\n";

    for (const auto & aForm : mVReachedF) {
        if (aForm->usedCnt() != 1 && !aForm->isAtomic()) {
            os << "    TypeElem " << aForm->genCodeFormName() << " = " << aForm->genCodeDef() << ";\n";
        }
    }
    for (size_t i=0; i<mVCurF.size(); i++)
       os <<  "    this->vvRes[aK][" << i << "] = " << mVCurF[i]->genCodeRef() << ";\n";
    os << "  }\n"
          "  this->mInBuf=0;\n"
          "}\n\n"
          "} // namespace CodeGen\n";
    return fileName;
}

} //   NS_MMVII_FormalDerivative

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
