#ifndef MMVII_FORMDER_CGENTPL_H
#define MMVII_FORMDER_CGENTPL_H

#include <vector>
#include <array>

// TODO: specialized exceptions
namespace CodeGen {

template<typename TypeElem,size_t NB_UK, size_t NB_OBS, size_t NB_RES, unsigned Interval>
class GenFuncTpl
{
public:
    typedef TypeElem Type;
    typedef std::vector<Type> VType;
    typedef std::array<Type,NB_UK> UkType;
    typedef std::array<Type,NB_OBS> ObsType;
    typedef std::array<Type,NB_RES> ResType;

    constexpr static size_t NbUk() { return NB_UK;}
    constexpr static size_t NbObs() { return NB_OBS;}

    GenFuncTpl(size_t sizeBuf) :
        vvRes(sizeBuf),
        vvUk(sizeBuf),
        vvObs(sizeBuf),
        mSizeBuf(sizeBuf),
        mInBuf(0)
    {}
    void pushNewEvals(const UkType &aVUk,const ObsType &aVObs)
    {
        if (mInBuf >= mSizeBuf)
            throw std::range_error("Codegen buffer overflow");
        vvUk[mInBuf] = aVUk;
        vvObs[mInBuf] = aVObs;
        mInBuf++;
    }
    void pushNewEvals(const VType &aVUk,const VType &aVObs)
    {
        if (mInBuf >= mSizeBuf)
            throw std::range_error("Codegen buffer overflow");
        if (aVUk.size() != NB_UK)
            throw std::range_error("Codegen ukn size error");
        if (aVObs.size() != NB_OBS)
            throw std::range_error("Codegen obs size error");
        for (size_t i=0; i<NB_UK; i++)
            vvUk[mInBuf][i] = aVUk[i];
        for (size_t i=0; i<NB_OBS; i++)
            vvObs[mInBuf][i] = aVObs[i];
        mInBuf++;
    }
    size_t bufferFree() const {return mSizeBuf - mInBuf;}
    size_t bufferSize() const {return mSizeBuf;}

    void evalAndClear() { mInBuf=0; }
    Type val(size_t numBuf, size_t numVal) const
    {
        return vvRes[numBuf][numVal * Interval];
    }
    Type der(size_t numBuf, size_t numVal,size_t numDer) const
    {
        return vvRes[numBuf][numVal * Interval + numDer + 1];
    }
    const std::vector<ResType> &result() const
    {
        return vvRes;
    }
protected:
    std::vector<ResType> vvRes;
    std::vector<UkType> vvUk;
    std::vector<ObsType> vvObs;
    size_t mSizeBuf;
    size_t mInBuf;
};

} // namespace CodeGen

#endif // MMVII_FORMDER_CGENTPL_H
