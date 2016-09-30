#ifndef FAST_H
#define FAST_H

#include <stdio.h>
#include "StdAfx.h"

/* ==== FAST : ==========
 * class d'implementation detecteur de points d'interet
*/
class Fast
{
public:
    Fast (double threshold, double radius);    //utilise avec radius = 1 (imagette 3*3)
    void detect(Im2D<unsigned char, int> &pic, vector<Pt2dr> &resultCorner);
    void dispPtIntertFAST(Im2D<unsigned char, int> pic, vector<Pt2dr>  pts, int zoomF, string filename);
    void outElPackHomo(vector<Pt2dr> &packIn, ElPackHomologue & packOut);
private:
    void getVoisInteret(int radius);
    bool isCorner(Pt2di pxlCenter, Im2D<unsigned char, int> &pic, double & threshold);
    bool isConsecutive(vector<int> &pxlValidIndx);
    double threshold;
    int radius;
    vector<Pt2dr> lastDetectResult;
};

typedef double tPxl;

class FastNew
{
public:
    FastNew   (
                            const TIm2D<tPxl, tPxl> & anIm,
                            double threshold, double radius,
                            const TIm2DBits<1> & anMasq
              );
    void detect(const TIm2D<tPxl, tPxl> & anIm, const TIm2DBits<1> anMasq, vector<Pt2dr> & lstPt);
    const double & radius()    const {return mRad;}
    const double & threshold() const {return mThres;}
    const vector<Pt2dr> & lstPt() const {return mLstPt;}
private:
    void getVoisinInteret     (double radius);
    void sortCWfor_mVoisin    (vector<Pt2di> &);
    bool isContinue           (vector<int> & label , int typeExtreme);
    const double           mThres;
    const double           mRad;
    vector<Pt2di>          mVoisin;
    vector<Pt2dr>          mLstPt;
    Im2D  <tPxl, tPxl>     mImInit;
    TIm2D  <tPxl, tPxl>    mTImInit;
    Im2D_Bits  <1>         mImMasq;
    TIm2DBits  <1>         mTImMasq;
};


#endif
