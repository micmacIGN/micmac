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

cTri3D::cTri3D(Pt3dr P1, Pt3dr P2, Pt3dr P3, int ind):
    mP1 (P1),
    mP2 (P2),
    mP3 (P3),
    mCtr ((P1 + P2 + P3)/3),
    mIsLoaded (true),
    mHaveBasis(false),
    mInd (ind)
{
}


cTri2D cTri3D::reprj(cBasicGeomCap3D * aCam)
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

cTri2D cTri3D::reprj(cBasicGeomCap3D * aCam, bool & OK)
{
    Pt2dr P1, P2, P3;
    OK=false;
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
         OK=true;
         return aTri2D;
    }
    else
        {
            OK=false;
            cTri2D aTri2D(Pt2dr(-1,-1), Pt2dr(-1,-1), Pt2dr(-1,-1));
            return aTri2D;
            //return cTri2D::Default();
        }
}

// distance from triangle center to camera center
double cTri3D::dist2Cam(cBasicGeomCap3D * aCam)
{
    return aCam->ProfondeurDeChamps(mCtr);
}

void cTri3D::calVBasis()
{
    mVec_21=mP2-mP1;
    mVec_31=mP3-mP1;
    mHaveBasis = true;
}





