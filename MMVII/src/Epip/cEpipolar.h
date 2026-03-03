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

    cPtxd<T,2> Im1ToEpip(const cPtxd<T,2>& aPt);
    static cEpipolarCouple<T> FromSensors(const cSensorImage* aSensor1, const cSensorImage* aSensor2, int aDegree);


private:
    explicit cEpipolarCouple();
    cEpipolarSingle<T> mEpipol1;
    cEpipolarSingle<T> mEpipol2;
};

} // MMVII

