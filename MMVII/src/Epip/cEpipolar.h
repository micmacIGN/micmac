#include "cPolyXY_N.h"


namespace MMVII
{

template<typename T>
class cEpipolarSingle
{
public:
    explicit cEpipolarSingle();

private:
    cPolyXY_N<T> mPolyV;
    cPolyXY_N<T> mPolyW;
    cPtxd<T,2> mCenter;
    cPtxd<T,2> mDir;
};

template<typename T>
class cEpipolarCouple
{
public:
    explicit cEpipolarCouple();
    
    cPtxd<T,2> FRommIm1ToIm2
    cEpipolarCouple FromSensors(const aSensor1* )

private:
    cEpipolarSingle<T> mEpipol1;
    cEpipolarSingle<T> mEpipol2;
};

} // MMVII

