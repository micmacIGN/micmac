#include "EsSimilitude.h"

cAppliEsSim::cAppliEsSim(cParamEsSim * aParam):
    mParam (aParam)
{
    cImgEsSim * aVgtX = new cImgEsSim(mParam->mImgX, this);
    cImgEsSim * aVgtY = new cImgEsSim(mParam->mImgY, this);

}
