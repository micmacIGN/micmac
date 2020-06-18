#ifndef _SymbDer_Common_H_
#define _SymbDer_Common_H_

#include <iostream>
#include <assert.h>
#include <vector>
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

static inline void Error(const std::string & aMes,const std::string & aExplanation)
{
    std::cout << "In SymbolicDerivative a fatal error" << "\n";
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


template<typename T>
class cCalculator
{
public:
    typedef T                      TypeElem;
    typedef std::vector<TypeElem>  tOneRes;

    cCalculator() = delete;
    cCalculator(const cCalculator&) = delete;
    virtual ~cCalculator() {}

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
        if (! mWithDer)  UserSError("Acces to derivate wich were not computed");
        return  mBufRes.at(aNumPush)->at(mSzInterval*aKElem +1 + aKVarDer);
    }

    const size_t NbUk() const { return mNbUK; }
    const size_t NbObs() const { return mNbObs; }
    const std::vector<tOneRes*> & Result() const { return mBufRes; }

protected:
    cCalculator(int aSzBuf, size_t aNbUk, size_t aNbObs, bool aWithDer=false, int aSzInterval=1) :
    mSzBuf      (aSzBuf),
    mNbUK       (aNbUk),
    mNbObs      (aNbObs),
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

    size_t                         mSzBuf;       ///< Capacity of bufferirsation
    size_t                         mNbUK;        ///< Dim=number of unkown
    size_t                         mNbObs;       ///< Number of obserbation variable
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
        UserSError("Push exceed buffer capacity");
    }
    if (aVUK.size() != NbUk())  // Check size are coherents
    {
        UserSError("Bad size in Unknowns");
    }
    if (aVObs.size() != NbObs())  // Check size are coherents
    {
        UserSError("Bad size in Onservations");
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


} // namespace NS_SymbolicDerivative

#endif // _SymbDer_Common_H_
