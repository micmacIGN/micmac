#include "cPolyXY_N.h"
namespace MMVII
{

template<typename T>
cPolyXY_N<T>::cPolyXY_N(int degree)
: mDegree(degree)
, mVK((degree+1) * (degree+2) / 2)
{
}

template<typename T>
cPolyXY_N<T>::cPolyXY_N(std::initializer_list<T> aVK)
    : mDegree(static_cast<int>(std::sqrt(aVK.size()*2)) - 1)
    , mVK(aVK)
{
    MMVII_INTERNAL_ASSERT_medium((mDegree+1)*(mDegree+2)/2 == (int)aVK.size(),"Incorrect vector for cPolyXY_N coefficients");
}

template<typename T>
cPolyXY_N<T>::cPolyXY_N(const std::vector<T>& aVK)
    : mDegree(static_cast<int>(std::sqrt(aVK.size()*2)) - 1)
    , mVK(aVK)
{
    MMVII_INTERNAL_ASSERT_medium((mDegree+1)*(mDegree+2)/2 == (int)aVK.size(),"Incorrect vector for cPolyXY_N coefficients");
}

template<typename T>
T cPolyXY_N<T>::operator()(const T &x, const T &y)
{
    T X_n;
    T Y_n;
    T result;
    int n;

    X_n = 1;            // X^0
    n = 0;
    result = T{};
    for (int i=0; i<=mDegree; i++) {
        Y_n = 1;        // Y^0
        for (int j=0; j<=mDegree-i; j++) {
            result += mVK[n] * X_n * Y_n;
            Y_n *= y;   // Y^(j+1)
            n++;
        }
        X_n *= x;       // X^(i+1)
    }
    return result;
}

template<typename T>
T cPolyXY_N<T>::operator()(const cPtxd<T, 2>& aPt)
{
    return this->operator()(aPt.x(),aPt.y());
}

template<typename T>
const T &cPolyXY_N<T>::K(int i, int j) const
{
    return mVK[idx(i,j)];
}

template<typename T>
T& cPolyXY_N<T>::K(int i, int j)
{
    return mVK[idx(i,j)];
}

template<typename T>
inline const T &cPolyXY_N<T>::K(int n) const
{
    return mVK[n];
}

template<typename T>
inline T &cPolyXY_N<T>::K(int n)
{
    return mVK[n];
}

template<typename T>
const std::vector<T> &cPolyXY_N<T>::VK() const
{
    return mVK;
}

template<typename T>
int cPolyXY_N<T>::Degree() const
{
    return mDegree;
}

template<typename T>
int cPolyXY_N<T>::NbCoeffs() const
{
    return mVK.size();
}


template<typename T>
void cPolyXY_N<T>::ResetFit()
{
    mLeastSq.reset();
    mFixedK.clear();
}

template<typename T>
void cPolyXY_N<T>::AddFixedK(int i, int j, const T &k)
{
    MMVII_INTERNAL_ASSERT_medium(i>=0 && j>=0 && i+j<=mDegree,"Bad usage of cPolyXY_N::AddFixedK()");
    MMVII_INTERNAL_ASSERT_medium(! mLeastSq,"Can't add fixed K after obs in cPolyXY_N::AddFixedK()");
    mFixedK.insert_or_assign(std::make_pair(i,j),k);
}

template<typename T>
template <typename IT>
void cPolyXY_N<T>::SetVK(IT it)
{
    int n=0;
    for (int i=0; i<= mDegree; i++) {
        for (int j=0; j<=mDegree-i; j++) {
            auto itFixedK = mFixedK.find({i,j});
            if ( itFixedK == mFixedK.end()) {
                mVK[n] = *it++;
            } else {
                mVK[n] = itFixedK->second;
            }
            n++;
        }
    }
}


template<typename T>
template <typename IT>
T cPolyXY_N<T>::VarToCoeffs(const T &x, const T &y, IT CoeffIt,const T& aFactor)
{
    T aDiffObs = 0;
    T X_n = 1;
    for (int i=0; i<=mDegree; i++) {
        T Y_n = 1;
        for (int j=0; j<=mDegree-i; j++) {
            auto itFixedK = mFixedK.find({i,j});
            if ( itFixedK == mFixedK.end()) {
                *CoeffIt++ = X_n * Y_n * aFactor;
            } else {
                aDiffObs += itFixedK->second * X_n * Y_n * aFactor;
            }
            Y_n *= y;
        }
        X_n *= x;
    }
    return aDiffObs;
}

template<typename T>
template <typename IT>
T cPolyXY_N<T>::VarToCoeffs(const cPtxd<T, 2> &aPt, IT CoeffIt,const T& aFactor)
{
    return VarToCoeffs(aPt.x(),aPt.y(),CoeffIt,aFactor);
}

template<typename T>
void cPolyXY_N<T>::AddObs(const T &x, const T &y, const T &v, const T& aWeight)
{
    if (! mLeastSq)
        mLeastSq = std::make_unique<cLeasSqtAA<T>>(NbCoeffs() - mFixedK.size());
    cDenseVect<T> coeffs(NbCoeffs() - mFixedK.size());
    auto aDiffObs = VarToCoeffs(x,y,coeffs.RawData(),1);
    mLeastSq->PublicAddObservation(aWeight,coeffs, v - aDiffObs);
}

template<typename T>
void cPolyXY_N<T>::AddObs(const cPtxd<T, 2> aPt, const T &v, const T &aWeight)
{
    AddObs(aPt.x(),aPt.y(),v,aWeight);
}

template<typename T>
void cPolyXY_N<T>::Fit()
{
    auto aVK = mLeastSq->PublicSolve();
    SetVK(aVK.RawData());
}

template<typename T>
T cPolyXY_N<T>::VarCurSol() const
{
    return mLeastSq->VarCurSol();
}

template<typename T>
int cPolyXY_N<T>::idx(int i, int j) const
{
    MMVII_INTERNAL_ASSERT_medium(i>=0 && j>=0 && i+j<=mDegree,"Bad usage of cPolyXY_N");
    return j + ((mDegree + 1) * (mDegree + 2) - (mDegree - i + 1) * (mDegree - i + 2)) / 2;
}

template class cPolyXY_N<tREAL4>;
template class cPolyXY_N<tREAL8>;
template class cPolyXY_N<tREAL16>;

} // MMVII

