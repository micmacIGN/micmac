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
    void reloadPic();
    void reloadTri();
    void correlInTriWithViewAngle(int indTri, double angleF);
    void correlInTri(int indTri);
    void correlByCplExist(int indTri);
    void correlByCplExistWithViewAngle(int indTri, double angleF);
    void sortDescend(vector<Pt2dr> & input);
    vector<int> mTriHavePtInteret;
    vector<int> mTriCorrelSuper;
    double countPts;
private:
    pic* chooseImgMaitre(bool assum1er);
    vector<ElCplePtsHomologues> choosePtsHomoFinal(vector<Pt2dr> &scorePtsInTri, triangle* aTri,
                                     vector<ElCplePtsHomologues> &P1P2Correl);
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
                                   double & score,
                                   double seuil_corel);
    //static bool comparatorPt2dr (const Pt2dr &l, const Pt2dr &r);
};



#endif
