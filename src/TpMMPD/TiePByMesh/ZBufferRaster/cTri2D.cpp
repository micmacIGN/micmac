#include "ZBufferRaster.h"

cTri2D::cTri2D(Pt2dr P1, Pt2dr P2, Pt2dr P3):
    mP1 (P1),
    mP2 (P2),
    mP3 (P3),
    mIsInCam(true),
    mReech (1.0)
{
}

cTri2D::cTri2D():
    mP1 (Pt2dr(0.0,0.0)),
    mP2 (Pt2dr(0.0,0.0)),
    mP3 (Pt2dr(0.0,0.0)),
    mIsInCam(false)
{}

cTri2D cTri2D::Default()
{
   return cTri2D();
}

void cTri2D::SetReech(double & scale)
{
    mReech = scale;
    mP1 = mP1*mReech;
    mP2 = mP2*mReech;
    mP3 = mP3*mReech;
}

Pt3dr cTri2D::pt3DFromVBasis(Pt2dr & ptInTri2D, cTri3D & aTri3D)
{
    //comme le method ptsInTri2Dto3D in triangle.cpp
    Pt3dr vec_I=aTri3D.P2() - aTri3D.P1();
    Pt3dr vec_J=aTri3D.P3() - aTri3D.P1();

    Pt2dr vec_i = mP2 - mP1;
    Pt2dr vec_j = mP3 - mP1;

    vec_i = vec_i/mReech;
    vec_j = vec_j/mReech;
    Pt2dr aP = ptInTri2D - mP1/mReech;

    double alpha = (aP.x*vec_j.y-aP.y*vec_j.x)/(vec_i.x*vec_j.y-vec_j.x*vec_i.y);
    double beta = (aP.y-alpha*vec_i.y)/vec_j.y;

    Pt3dr pts3DInTri;
    pts3DInTri.x = alpha*vec_I.x + beta*vec_J.x;
    pts3DInTri.y = alpha*vec_I.y + beta*vec_J.y;
    pts3DInTri.z = alpha*vec_I.z + beta*vec_J.z;

    return( pts3DInTri + aTri3D.P1() );
}


double cTri2D::profOfPixelInTri(Pt2dr & ptInTri2D, cTri3D & aTri3D, CamStenope * aCam)
{
    Pt2dr ptInTri2DGlob(ptInTri2D.x/mReech, ptInTri2D.y/mReech);
    Pt3dr aPt = cTri2D::pt3DFromVBasis(ptInTri2DGlob, aTri3D);
    if (aCam->PIsVisibleInImage(aPt))
    {
        //Can I use this method ?
        return aCam->ProfondeurDeChamps(aPt);
        //return(euclid((aCam->VraiOpticalCenter()-aPt).AbsP() ));
    }
    else
        return TT_DEFAULT_PROF_NOVISIBLE;
}

bool cTri2D::orientToCam(CamStenope * aCam)
{
    if ( ((mP1-mP2) ^ (mP1-mP3)) > 0 )
        return false;
    else
        return true;
}

double cTri2D::surf()
{
    return ((mP1/mReech-mP2/mReech) ^ (mP1/mReech-mP3/mReech));
}
