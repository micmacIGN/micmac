#ifndef CORRELMESH_H
#define CORRELMESH_H

#include "InitOutil.h"
#include <stdio.h>
#include "Detector.h"
#include "Triangle.h"
#include "../kugelhupf.h"


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
    void multiCorrel(int indTri, double angleF);
    void correlInTriWithViewAngle(int indTri, double angleF, bool debugByClick = false); //add visible condition
    void correlInTri(int indTri);                                   //basic
    void correlByCplExist(int indTri);
    void correlByCplExistWithViewAngle(int indTri, double angleF, bool debugByClick = false);
    void sortDescend(vector<Pt2dr> & input);
    void verifCplHomoByTriangulation(int indTri, double angleF);
    void homoCplSatisfiyTriangulation(int indTri, double angleF);
    vector<int> mTriHavePtInteret;
    vector<int> mTriCorrelSuper;
    double countPts;
    double countCplOrg;
    double countCplNew;
private:
    pic* chooseImgMaitre(int indTri, double & angleReturn, double angleVisibleLimit, bool assume1er=false);
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
    Pt2dr correlPtsInteretInImaget_NEW(Pt2dr ptInt1,
                                       ImgetOfTri imgetMaitre, ImgetOfTri imget2nd,
                                       matAffine & affine,
                                       bool & foundMaxCorr,
                                       double & scoreR, bool dbgByClick,
                                       double seuil_corel = 0.9);
    Pt2dr correlSubPixelPtsIntInImaget(Pt2dr ptInt1, ImgetOfTri imgetMaitre, ImgetOfTri imget2nd,
                                        matAffine & affine,
                                        bool & foundMaxCorr,
                                        double & scoreR,
                                        bool dbgByClick,
                                        double seuil_corel = 0.9);
    Pt2dr point_temp;
};



#endif
