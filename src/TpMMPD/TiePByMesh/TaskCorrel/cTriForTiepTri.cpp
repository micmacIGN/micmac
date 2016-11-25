#include "TaskCorrel.h"
cTriForTiepTri::cTriForTiepTri(cAppliTaskCorrel & aAppli, triangle * aTri3d):
    mNumImg (-1),
    mPt1    (Pt2dr(0.0,0.0)),
    mPt2    (Pt2dr(0.0,0.0)),
    mPt3    (Pt2dr(0.0,0.0)),
    mAppli  (aAppli),
    mTri3D  (aTri3d),
    mrprjOK (false)
{}

bool cTriForTiepTri::reprj(int aNumImg)
{
    cImgForTiepTri * aImg = mAppli.VImgs()[aNumImg];

    Pt3dr Pt1 = mTri3D->getSommet(0);
    Pt3dr Pt2 = mTri3D->getSommet(1);
    Pt3dr Pt3 = mTri3D->getSommet(2);

    mPt1 = aImg->Cam()->Ter2Capteur(Pt1);
    mPt2 = aImg->Cam()->Ter2Capteur(Pt2);
    mPt3 = aImg->Cam()->Ter2Capteur(Pt3);

    mNumImg = aNumImg;
    if      (     aImg->inside(mPt1)
              &&  aImg->inside(mPt2)
              &&  aImg->inside(mPt3)
              &&  aImg->Cam()->PIsVisibleInImage(Pt1)
              &&  aImg->Cam()->PIsVisibleInImage(Pt2)
              &&  aImg->Cam()->PIsVisibleInImage(Pt3)
            )
        return true;
    else
        return false;
}

double cTriForTiepTri::valElipse()
{
    if (mrprjOK || mNumImg == -1)
        return DBL_MIN;
    else
    {
        double aSurf =  (mPt1-mPt2) ^ (mPt1-mPt3);
        Pt3dr Pt1 = mTri3D->getSommet(0);
        Pt3dr Pt2 = mTri3D->getSommet(1);
        Pt3dr Pt3 = mTri3D->getSommet(2);
        if (-aSurf > TT_SEUIL_SURF_TRIANGLE && mrprjOK)
        {
            //creer plan 3D local contient triangle
            cElPlan3D * aPlanLoc = new cElPlan3D(Pt1, Pt2, Pt3);
            ElRotation3D aRot_PE = aPlanLoc->CoordPlan2Euclid();
            ElRotation3D aRot_EP = aRot_PE.inv();
            //calcul coordonne sommet triangle dans plan 3D local (devrait avoir meme Z)
            Pt3dr aPtP0 = aRot_EP.ImAff(Pt1); //sommet triangle on plan local
            Pt3dr aPtP1 = aRot_EP.ImAff(Pt2);
            Pt3dr aPtP2 = aRot_EP.ImAff(Pt3);
            //creer translation entre coordonne image global -> coordonne image local du triangle (plan image)
            ElAffin2D aAffImG2ImL(ElAffin2D::trans(mPt1));
            Pt2dr aPtPIm0 = aAffImG2ImL(mPt1);
            Pt2dr aPtPIm1 = aAffImG2ImL(mPt2);
            Pt2dr aPtPIm2 = aAffImG2ImL(mPt3);
            //calcul affine entre plan 3D local (elimine Z) et plan 2D local
            ElAffin2D aAffLc2Im;
            aAffLc2Im = aAffLc2Im.FromTri2Tri(  Pt2dr(aPtP0.x, aPtP0.y),
                                                Pt2dr(aPtP1.x, aPtP1.y),
                                                Pt2dr(aPtP2.x, aPtP2.y),
                                                aPtPIm0,aPtPIm1,aPtPIm2
                                             );
            //calcul vector max min pour choisir img master
            double vecA_cr =  aAffLc2Im.I10().x*aAffLc2Im.I10().x + aAffLc2Im.I10().y*aAffLc2Im.I10().y;
            double vecB_cr =  aAffLc2Im.I01().x*aAffLc2Im.I01().x + aAffLc2Im.I01().y*aAffLc2Im.I01().y;
            double AB_cr   =  pow(aAffLc2Im.I10().x*aAffLc2Im.I01().x,2) + pow(aAffLc2Im.I10().y*aAffLc2Im.I01().y,2);
            //double theta_max =  vecA_cr + vecB_cr +sqrt((vecA_cr - vecB_cr) + 4*AB_cr)*(0.5);
            return (vecA_cr + vecB_cr +sqrt((vecA_cr - vecB_cr) + 4*AB_cr)*(-0.5));
        }
        else
            return DBL_MIN;
    }
}


