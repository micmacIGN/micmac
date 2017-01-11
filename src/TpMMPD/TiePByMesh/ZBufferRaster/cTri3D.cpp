#include "ZBufferRaster.h"


cTri3D::cTri3D(Pt3dr P1, Pt3dr P2, Pt3dr P3):
    mP1 (P1),
    mP2 (P2),
    mP3 (P3),
    mCtr ((P1 + P2 + P3)/3),
    mIsLoaded (true),
    mHaveBasis(false)
{
}

cTri2D cTri3D::reprj(CamStenope * aCam)
{
    Pt2dr P1, P2, P3;
    if (mIsLoaded)
    {
        P1 = aCam->Ter2Capteur(mP1);
        P2 = aCam->Ter2Capteur(mP2);
        P3 = aCam->Ter2Capteur(mP3);

    }
    if      (
                 aCam->PIsVisibleInImage(mP1)
              && aCam->PIsVisibleInImage(mP1)
              && aCam->PIsVisibleInImage(mP1)
            )
    {
         cTri2D aTri2D(P1,P2,P3);
         return aTri2D;
    }
    else
        {
            return cTri2D::Default();
        }
}

double cTri3D::dist2Cam(CamStenope * aCam)
{
    return aCam->ProfondeurDeChamps(mCtr);
}

void cTri3D::calVBasis()
{
    mVec_21=mP2-mP1;
    mVec_31=mP3-mP1;
    mHaveBasis = true;
}





