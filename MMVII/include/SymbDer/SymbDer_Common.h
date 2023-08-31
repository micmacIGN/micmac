#ifndef _SymbDer_Common_H_
#define _SymbDer_Common_H_

#ifndef SYMBDER_WITH_MMVII
#define SYMBDER_WITH_MMVII true
#endif

namespace NS_SymbolicDerivative 
{
template<typename T> class cCalculator;
template<typename T> class cCompiledCalculator;
};


#if (SYMBDER_WITH_MMVII)
#include "MMVII_util.h"
#include "MMVII_Derivatives.h"
#define SYMBDER_cMemCheck MMVII::cMemCheck
#define SYMBDER_DelAtEnd  MMVII::cObj2DelAtEnd
#else             //========================================================== WITH_MMVI
class SYMBDER_cMemCheck
{
};
class SYMBDER_DelAtEnd
{
};
#endif



#include <iostream>
#include <assert.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
// ===================== MPD  error: call of overloaded ‘abs(const double&)’ is ambiguous ===============
#include <math.h>
#include <cmath>


namespace NS_SymbolicDerivative {

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


static inline void DefaultErrorHandler(const std::string & aMes,const std::string & aExplanation, const std::string& aContext)
{
    std::cerr << "In SymbolicDerivative a fatal error" << "\n";
    std::cerr << "  Likely Source   ["<< aExplanation << "]\n";
    if (aContext.size())
        std::cerr << "  For formula     ["<< aContext << "]\n";
    std::cerr << "  Message  ["<< aMes << "]\n";
    std::cerr << std::flush;
    abort();
}

class ErrorMgr {
public:
    typedef void (*ErrorHandler) (const std::string & aMes,const std::string & aExplanation, const std::string& aContext);

    static void Error(const std::string & aMes,const std::string & aExplanation, const std::string& aContext)
    {
        GetHandler()(aMes, aExplanation, aContext);
    }

    static ErrorHandler GetHandler()
    {
        return GetSetHandler(nullptr) ;
    }
    static ErrorHandler SetHandler(ErrorHandler handler)
    {
        if (handler == nullptr)
            handler = DefaultErrorHandler;
        return GetSetHandler(handler) ;
    }

private:
    ErrorMgr() = delete;
    ~ErrorMgr() = delete;
    static ErrorHandler GetSetHandler(ErrorHandler handler)
    {
        static ErrorHandler theErrorHandler = DefaultErrorHandler;

        if (handler != nullptr)
            theErrorHandler = handler;
        return theErrorHandler;
    }
};

     ///    Error due probably to internal mistake
static inline void InternalError(const std::string & aMes, const std::string& aContext)
{
    ErrorMgr::Error(aMes,"Internal Error of the Library",aContext);
}
     /// Error probably due to bad usage of the library (typically out limit vector access)
static inline void UserSError(const std::string & aMes, const std::string& aContext)
{
    ErrorMgr::Error(aMes,"Probable error on user's side due to unapropriate usage of the library",aContext);
}


     /// Check equality in test, taking account numericall error
static inline void AssertAlmostEqual(const double & aV1,const double & aV2,const double & aEps)
{
   if ( (std::abs(aV1-aV2)> aEps*(std::abs(aV1)+std::abs(aV2))) )
      InternalError("Test equality failed","");
}

class cConvStrRank 
{
   public :
     cConvStrRank (const std::vector<std::string> & aVName) :
       mRange2Name  (aVName)
     {
        for (size_t aRange=0 ; aRange<mRange2Name.size() ; aRange++)
            mName2Range[mRange2Name[aRange]] = aRange;
     }
     int RangeOfName(const std::string & aName,bool SVP) const
     {
         const auto & anIter = mName2Range.find(aName);
         if (anIter== mName2Range.end())
         {
              if (! SVP)  UserSError("Name dont exist",aName);
              return -1;
         }
         return anIter->second;
     }
     const std::string & NameOfRange(int aRange) const
     {
         return mRange2Name.at(aRange);
     }
     const std::vector<std::string> & Names() const {return mRange2Name;}
   private :
     std::vector<std::string>      mRange2Name;
     std::map<std::string,size_t>  mName2Range;
};

template<typename T>
class cCalculator  : public SYMBDER_cMemCheck,
	             public SYMBDER_DelAtEnd
{
public:
    typedef T                      TypeElem;
    typedef std::vector<TypeElem>  tOneRes;

    cCalculator() = delete;
    cCalculator(const cCalculator&) = delete;
    virtual ~cCalculator() {}

    const std::string& Name() const { return mName;}
    void SetName(const std::string& aName) { this->mName = aName;}

    bool BufIsFull() const {return mNbInBuf == mSzBuf;} ///< Can we push more value ?
    size_t SzBuf() const  {return mSzBuf;}  ///< Total Number of value we can push
    size_t FreeSzBuf() const  {return mSzBuf - mNbInBuf;}  ///< Number of value we can still add
    size_t NbInBuf() const  {return  mNbInBuf;}  ///< Number of value already in buf


    ///  Add a new set of vals (unknown + Obs) inside de evaluation "queue"
    void PushNewEvals(const std::vector<TypeElem> & aVUK,const std::vector<TypeElem> & aVObs);

    /** Make the evaluation of current functions on pushes values . Let V be the result
        (*V[0]) is  the  vector of containing current formula for first SetCur
        !! => Warn the same memory space is recycled ...
    */
    const std::vector<std::vector<T> *> & EvalAndClear();
    ///  Get one eval
    std::vector<T> & DoOneEval(const std::vector<TypeElem> & aVUK,const std::vector<TypeElem> & aVObs);

    /// Return value computed taking into account order of storage
    const TypeElem & ValComp(int aNumPush,int aKElem) const
    {
        return  mBufRes.at(aNumPush)->at(mSzInterval*aKElem);
    }
    /// Retur value of derivate computed taking into account order of storage
    const TypeElem & DerComp(int aNumPush,int aKElem,int aKVarDer) const
    {
        if (! mWithDer)  UserSError("Acces to derivate wich were not computed",Name());
        return  mBufRes.at(aNumPush)->at(mSzInterval*aKElem +1 + aKVarDer);
    }

    bool   WithDer() const { return mWithDer; }           // With derive ? Usable for checking
    size_t NbUk() const { return mNbUK; }                 // Nb of unknowns
    size_t NbObs() const { return mNbObs; }               // Nb of Observations
    size_t NbElem() const { return mNbElem; }             // Nb of primary values returned by formula (w/o counting derivatives)
    const std::vector<tOneRes*> & Result() const { return mBufRes; }
    
    int RangeOfUk(const std::string & aName,bool SVP=false) const
    {return mConvNamesUk.RangeOfName(aName,SVP);}
    const std::vector<std::string> & NamesUk() const {return mConvNamesUk.Names();}
    

    int RangeOfObs(const std::string & aName,bool SVP=false) const
    {return mConvNamesObs.RangeOfName(aName,SVP);}
    const std::vector<std::string> & NamesObs() const {return mConvNamesObs.Names();}

protected:
    cCalculator
    (
         const std::string& aName, 
         int aSzBuf, 
         const std::vector<std::string> & aVNUk,  // Variable, fix Dim In
         const std::vector<std::string> & aVNObs,
         bool aWithDer=false, 
         int aSzInterval=1
     ) :
        mName         (aName),
        mSzBuf        (aSzBuf),
        mNbUK         (aVNUk.size()),
        mConvNamesUk  (aVNUk),
        mNbObs        (aVNObs.size()),
        mConvNamesObs (aVNObs),
        mNbElem       (0),
        mNbInBuf      (0),
        mWithDer      (aWithDer),
        mSzInterval   (aSzInterval),
        mBufLineRes   (mSzBuf),
        mBufRes       ()
    {
        for (size_t i=0; i< aVNUk.size(); i++) {
            if (!IsValidCIdentifier(aVNUk[i])) {
                UserSError("Name for Unknown #" + std::to_string(i) +
                           " is not a valid C++ identifier ('"
                           + aVNUk[i] + "')",
                           mName);
            }
        }
        for (size_t i=0; i< aVNObs.size(); i++) {
            if (!IsValidCIdentifier(aVNObs[i])) {
                UserSError("Name for Observation #" + std::to_string(i) +
                           " is not a valid C++ identifier ('"
                           + aVNObs[i] + "')",
                           mName);
            }
        }
        mBufRes.reserve(mSzBuf);
    }

    // Do actual storage for Uks and Obs. Sizes have been checked
    virtual void SetNewUks(const std::vector<TypeElem> &aVUks) = 0;
    virtual void SetNewObs(const std::vector<TypeElem> &aVObs) = 0;

    // Do actual caluculus. Just store resulst in mBurLineRes. This class manages mBufRes
    virtual void DoEval() = 0;

    // Utility function that checks if string s is a valid C/C++ identifier
    bool IsValidCIdentifier(const std::string& s) const;

    std::string                    mName;
    size_t                         mSzBuf;       ///< Capacity of bufferirsation
    size_t                         mNbUK;        ///< DimIn=number of unkown
    cConvStrRank                   mConvNamesUk;    ///< Names of unknonw, used as a helper
    size_t                         mNbObs;       ///< Number of obserbation variable
    cConvStrRank                   mConvNamesObs;   ///< Names of observation, used as a helper
    size_t                         mNbElem;      ///< DimOut=Number of elements returned by the formula (w/o derivative)
    size_t                         mNbInBuf;     ///< Number of Unknown/Obs vect currenlty loaded in buf
    bool                           mWithDer;     ///< Done With Derivate
    int                            mSzInterval;  ///< Size between two val, depends if computation done with deriv
    std::vector<tOneRes>           mBufLineRes;  ///< Reserve memory for each line, make the allocation at init
    std::vector<tOneRes*>          mBufRes;      ///< Reserve memory for result itself, point on mBufLineRes to limit allocation


};

template<typename T>
void cCalculator<T>::PushNewEvals(const std::vector<T> &aVUK, const std::vector<T> &aVObs)
{

    if (this->mNbInBuf >= this->mSzBuf)
    {
        UserSError("Push exceed buffer capacity",Name());
    }
    if (aVUK.size() != NbUk())  // Check size are coherents
    {
        UserSError("Bad size in Unknowns",Name());
    }
    if (aVObs.size() != NbObs())  // Check size are coherents
    {
        UserSError("Bad size in Observations",Name());
    }

    this->SetNewUks(aVUK);
    this->SetNewObs(aVObs);
    this->mNbInBuf++;
}

template<typename T>
const std::vector<std::vector<T> *> & cCalculator<T>::EvalAndClear()
{
    this->mBufRes.clear();
    this->DoEval();
    for (size_t aK=0; aK < this->mNbInBuf; aK++)
        this->mBufRes.push_back(&this->mBufLineRes[aK]);
    this->mNbInBuf = 0;
    return mBufRes;
}

template<typename T>
std::vector<T> & cCalculator<T>::DoOneEval(const std::vector<TypeElem> & aVUK,const std::vector<TypeElem> & aVObs)
{
   if (this->mNbInBuf!=0)
   {
      UserSError("DoOneEval: buffer not empty","");
   }
   this->PushNewEvals(aVUK,aVObs);
   return *(this->EvalAndClear()[0]);
}

template<typename T>
bool cCalculator<T>::IsValidCIdentifier(const std::string& s) const
{
    if (s.size() == 0)
        return false;
    if (! (std::isalpha(s[0]) || s[0] == '_'))
        return false;
    return std::all_of(s.cbegin()+1, s.cend(),[](char c) { return std::isalnum(c) || c == '_';});
}



/** Specilisation for calculator opering generated code  (v.s dynamic just after formula)
*/
template<typename T> class cCompiledCalculator : public cCalculator<T>
{
   public :

      cCompiledCalculator
      (
         const std::string& aName, 
         size_t aSzBuf, 
         size_t aNbElem,   // Dim out
         size_t aSzLine,   // should be aNbElem * aSzInterval
         const std::vector<std::string> & aVNUk,  // Variable, fix Dim In
         const std::vector<std::string> & aVNObs,
         bool aWithDer,
         size_t aSzInterval
      ) :
         cCalculator<T> (aName,aSzBuf,aVNUk,aVNObs,aWithDer,aSzInterval) ,
         mVUk  (aSzBuf),
         mVObs (aSzBuf)
      {
         this->mNbElem = aNbElem;
         for (auto& line : this->mBufLineRes)
             line.resize(aSzLine); 
         for (auto& aUk : this->mVUk)
             aUk.resize(this->NbUk());
         for (auto& aObs : this->mVObs)
             aObs.resize(this->NbObs());

      }
   protected :
      virtual void SetNewUks(const std::vector<T> & aVUks) override
      {
          for (size_t i=0; i<this->NbUk(); i++)
            this->mVUk[this->mNbInBuf][i] = aVUks[i];
      }
      virtual void SetNewObs(const std::vector<T> & aVObs) override
      {
          for (size_t i=0; i<this->NbObs(); i++)
            this->mVObs[this->mNbInBuf][i] = aVObs[i];
      }
      std::vector<std::vector<T>> mVUk;
      std::vector<std::vector<T>> mVObs;
};


// This function must be defined somewhere and initialize Name/Alloc association
// by calling cName2Calc<>::Register(Name,&Alloc) multiple times.
// File "cName2CalcRegisterAll.cpp.tmpl" is a template file for such an empty function.
// This template must be compiled/linked in if this function is not defined elsewhere.
// cGenNameAlloc::generateRFile() can create such a file for auto generated code
void cName2CalcRegisterAll(void);


template <class Type>  class cName2Calc
{
  public :
    typedef  cCompiledCalculator<Type>  tCalc;
    typedef  tCalc * (* tAllocator) (int aSzBuf);


    // It's actually more a namespace than a class
    cName2Calc() = delete;

    /// That's what we want : alloc an object from its name
    static tCalc * CalcFromName(const std::string & aName,int aSzBuf,bool SVP=false)
    {
        tAllocator anAlloc =  Name2Alloc(aName,nullptr,SVP);
        if (anAlloc==nullptr)
        {
            if (SVP) return nullptr;
            UserSError("Cannot create Calculator,",aName);
        }
        return anAlloc(aSzBuf);
    }

    void static Register(const std::string & aName,tAllocator anAlloc)
    {
         Name2Alloc(aName,anAlloc,false);
    }

  private :
    /// if tAllocator=0 return Alloc associated to name, else store the association
    /// At first call, will try to initialize Name/Alloc associating map by calling InitMapAllocator()
    ///   (see bellow)
    static tAllocator  Name2Alloc(const std::string & aName,tAllocator anAlloc,bool SVP=false)
    {
       static std::map<std::string,tAllocator> TheMap;
       static bool firstCall = true;

       if (firstCall) {
           firstCall=false;
           cName2CalcRegisterAll();
       }
       auto anIter = TheMap.find(aName);

       // If no allocator, we are in the mode where we want to retrieve it
       if (anAlloc==nullptr)
       {
           if(anIter==TheMap.end()) // There must be something associated
           {
             if (SVP) return nullptr;
             UserSError("Cannot extract allocator. Check that this application was recompiled after code generation",aName);
             return nullptr;
           }
           return anIter->second;
       }

       if (anIter!=TheMap.end()) // We should not override
          UserSError("Allocator already exist,",aName);
       TheMap[aName] = anAlloc;

       return nullptr;
    }

};

} // namespace NS_SymbolicDerivative

#endif // _SymbDer_Common_H_
