#ifndef _SymbDer_Common_H_
#define _SymbDer_Common_H_

#include <iostream>
#include <assert.h>
#include <vector>
#include <string>
#include <map>
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

static inline void Error(const std::string & aMes,const std::string & aExplanation, const std::string& aContext)
{
    std::cout << "In SymbolicDerivative a fatal error" << "\n";
    std::cout << "  Likely Source   ["<< aExplanation << "]\n";
    if (aContext.size())
        std::cout << "  For formula     ["<< aContext << "]\n";
    std::cout << "  Message  ["<< aMes << "]\n";
    assert(false);
}
     ///    Error due probably to internal mistake
static inline void InternalError(const std::string & aMes, const std::string& aContext)
{
   Error(aMes,"Internal Error of the Library",aContext);
}
     /// Error probably due to bad usage of the library (typically out limit vector access)
static inline void UserSError(const std::string & aMes, const std::string& aContext)
{
   Error(aMes,"Probable error on user's side due to unapropriate usage of the library",aContext);
}


     /// Check equality in test, taking account numericall error
static inline void AssertAlmostEqual(const double & aV1,const double & aV2,const double & aEps)
{
   if ( (std::abs(aV1-aV2)> aEps*(std::abs(aV1)+std::abs(aV2))) )
      InternalError("Test equality failed","");
}


template<typename T>
class cCalculator
{
public:
    typedef T                      TypeElem;
    typedef std::vector<TypeElem>  tOneRes;

    cCalculator() = delete;
    cCalculator(const cCalculator&) = delete;
    virtual ~cCalculator() {}

    const std::string& Name() const { return mName;}
    void SetName(const std::string& aName) const { mName = aName;}

    bool BufIsFull() const {return mNbInBuf == mSzBuf;} ///< Can we push more value ?
    size_t SzBuf() const  {return mSzBuf;}  ///< Total Number of value we can push
    size_t FreeSzBuf() const  {return mSzBuf - mNbInBuf;}  ///< Number of value we can still add


    ///  Add a new set of vals (unknown + Obs) inside de evaluation "queue"
    void PushNewEvals(const std::vector<TypeElem> & aVUK,const std::vector<TypeElem> & aVObs);

    /** Make the evaluation of current functions on pushes values . Let V be the result
        (*V[0]) is  the  vector of containing current formula for first SetCur
        !! => Warn the same memory space is recycled ...
    */
    const std::vector<std::vector<T> *> & EvalAndClear();

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

    const size_t NbUk() const { return mNbUK; }                 // Nb of unknowns
    const size_t NbObs() const { return mNbObs; }               // Nb of Observations
    const size_t NbElem() const { return mNbElem; }             // Nb of primary values returned by formula (w/o counting derivatives)
    const std::vector<tOneRes*> & Result() const { return mBufRes; }

protected:
    cCalculator(const std::string& aName, int aSzBuf, size_t aNbUk, size_t aNbObs, bool aWithDer=false, int aSzInterval=1) :
    mName       (aName),
    mSzBuf      (aSzBuf),
    mNbUK       (aNbUk),
    mNbObs      (aNbObs),
    mNbElem     (0),
    mNbInBuf    (0),
    mWithDer    (aWithDer),
    mSzInterval (aSzInterval),
    mBufLineRes (mSzBuf),
    mBufRes     ()
    {
        mBufRes.reserve(mSzBuf);
    }

    // Do actual storage for Uks and Obs. Sizes have been checked
    virtual void SetNewUks(const std::vector<TypeElem> &aVUks) = 0;
    virtual void SetNewObs(const std::vector<TypeElem> &aVObs) = 0;

    // Do actual caluculus. Just store resulst in mBurLineRes. This class manages mBufRes
    virtual void DoEval() = 0;


    std::string                    mName;
    size_t                         mSzBuf;       ///< Capacity of bufferirsation
    size_t                         mNbUK;        ///< Dim=number of unkown
    size_t                         mNbObs;       ///< Number of obserbation variable
    size_t                         mNbElem;      ///< Number of elements returned by the formula (w/o derivative)
    size_t                         mNbInBuf;     ///< Number of Unknown/Obs vect currenlty loaded in buf
    bool                           mWithDer;     ///< Done With Derivate
    int                            mSzInterval;  ///< Size between two val, depends if computation done with deriv
    std::vector<tOneRes>           mBufLineRes;  ///< Reserve memory for each line
    std::vector<tOneRes*>          mBufRes;      ///< Reserve memory for result itself
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
        UserSError("Bad size in Onservations",Name());
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

template <class Type>  class cName2Calc
{
  public :
    typedef  cCalculator<double>  tCalc;
    typedef  tCalc * (* tAllocator) (int aSzBuf);

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

    /// The constructor is used to store the association that allow to create unknow object
    cName2Calc(const std::string & aName,tAllocator anAlloc)
    {
         Name2Alloc(aName,anAlloc,false);
    }
  private :
    /// if tAllocator=0 return Alloc associated to name, else store the association
    static tAllocator  Name2Alloc(const std::string & aName,tAllocator anAlloc,bool SVP=false)
    {
       static std::map<std::string,tAllocator> TheMap;
       auto anIter = TheMap.find(aName);

       // If no allocator, we are in the mode where we want to retrieve it
       if (anAlloc==nullptr)
       {
           if(anIter==TheMap.end()) // There must be something associated
           {
             if (SVP) return nullptr;
             UserSError("Cannot extract allocator,",aName);
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
