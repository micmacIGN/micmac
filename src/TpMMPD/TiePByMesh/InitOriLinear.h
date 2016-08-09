#ifndef INITORILINEAR_H
#define INITORILINEAR_H

#include <stdio.h>
#include "StdAfx.h"
#include "InitOutil.h"

class SerieCamLinear
{
    public:

    SerieCamLinear( string aPatImgREF, string aPatImgNEW, string aOri, string aOriOut, int index);
    void calPosRlt();
    void saveSystem(vector<SerieCamLinear*> aSystem){this->mSystem = aSystem;}
    Pt3dr calVecMouvement();
    void initSerie(Pt3dr vecMouvCam0 , vector<string> aVecPoseTurn, vector<double> aVecAngleTurn);
    void initSerieByRefSerie(SerieCamLinear* REFSerie);


	vector<string> mSetImgREF;
	vector<string> mSetImgNEW;
    vector<cOrientationConique> mSetOriREF;
    vector<cOrientationConique> mSetOriNEW;
	string mPatImgREF;
	string mPatImgNEW;
    string mOri;
    string mOriOut;
    cInterfChantierNameManipulateur * mICNM;
    int mIndexCam;
    vector<Pt3dr> posRltWithOtherCam;
    vector<SerieCamLinear*> mSystem;
    Pt3dr mVecMouvement;
};
#endif
