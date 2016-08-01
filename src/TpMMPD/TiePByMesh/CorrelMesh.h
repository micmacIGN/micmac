#ifndef CORRELMESH_H
#define CORRELMESH_H

#include <stdio.h>
#include "StdAfx.h"
#include "../kugelhupf.h"
#include "InitOutil.h"
#include "Detector.h"
#include "Triangle.h"

typedef struct PtInterest {
    Pt2dr aP1;
    Pt2dr aP2;
    double scoreCorrel;
}PtInterest;

class CorrelMesh
{
public:
    CorrelMesh(InitOutil * aChain);
    void reloadTriandPic();
    void correlInTri(int indTri);
private:
    pic* chooseImgMaitre(bool assum1er);
    InitOutil * mChain;
    vector<pic*> mPtrListPic;
    vector<triangle*> mPtrListTri;
    pic * mPicMaitre;
    vector<pic*> mListPic2nd;
    int mSzW;
    Pt2dr correlPtsInteretInImaget(Pt2dr ptInt1,
                                   ImgetOfTri imgetMaitre, ImgetOfTri imget2nd,
                                   matAffine & affine,
                                   bool & foundMaxCorr,
                                   double seuil_corel);
};



#endif
