#ifndef INITORILINEAR_H
#define INITORILINEAR_H

#include "InitOutil.h"
#include <stdio.h>
#include "StdAfx.h"

class SerieCamLinear
{
    public:

    SerieCamLinear( string aPatImgREF, string aPatImgNEW,
                    string aOri, string aOriOut,
                    string aAxeOrient, vector<double>aMulF, int index);
    void calPosRlt();
    void saveSystem(vector<SerieCamLinear*> aSystem){this->mSystem = aSystem;}
    Pt3dr calVecMouvement();
    void initSerie(Pt3dr vecMouvCam0 , vector<string> aVecPoseTurn, vector<double> aVecAngleTurn);
    void initSerieByRefSerie(SerieCamLinear* REFSerie);
    void partageSection(vector<string> aVecPoseTurn,
                                        vector<double> aVecAngleTurn);
    void initSerieWithTurn( Pt3dr vecMouvCam0 ,
                            vector<string> aVecPoseTurn,
                            vector<double> aVecAngleTurn    );
    void calCodageMatrRot(cTypeCodageMatr ref, double angle,
                            cTypeCodageMatr &out, string axe);
    Pt3dr calVecMouvementTurn(Pt3dr vecRef, double angle, string axe);


    vector<string> mSetImgREF;
    vector<string> mSetImgNEW;
    vector<cOrientationConique> mSetOriREF;
    vector<cOrientationConique> mSetOriNEW;
    string mPatImgREF;
    string mPatImgNEW;
    string mOri;
    string mOriOut;
    string mAxeOrient;
    cInterfChantierNameManipulateur * mICNM;
    int mIndexCam;
    vector<Pt3dr> posRltWithOtherCam;
    vector<SerieCamLinear*> mSystem;
    Pt3dr mVecMouvement;
    vector< vector<string> >mSections;
    vector<double> mMulF; //multipli factor pour le deplacement entre les sections
};
#endif
