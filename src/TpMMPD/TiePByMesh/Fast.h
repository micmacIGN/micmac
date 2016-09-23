#ifndef FAST_H
#define FAST_H

#include <stdio.h>
#include "StdAfx.h"

/* ==== FAST : ==========
 * class d'implementation detecteur de points d'interet
*/
const static Pt2di pxlPosition[17]=
{
/*
       11 12 13
     10 - -  - 14
    9 - - -  - - 15
    8 - - 16 - - 0
    7 - - -  - - 1
      6 - -  - 2
        5 4  3
*/
    Pt2di(0,3),	//0
    Pt2di(1,3),
    Pt2di(2,2),
    Pt2di(3,1),
    Pt2di(3,0),
    Pt2di(3,-1),
    Pt2di(2,-2),
    Pt2di(1,-3),
    Pt2di(0,-3),
    Pt2di(-1,-3),
    Pt2di(-2,-2),
    Pt2di(-3,-1),
    Pt2di(-3,0),
    Pt2di(-3,1),
    Pt2di(-2,2),
    Pt2di(-1,3),
    Pt2di(0,0)	//16
};
class Fast
{
public:
    Fast (double threshold, int radius);    //utilise avec radius = 1 (imagette 3*3)
    void detect(Im2D<unsigned char, int> &pic, vector<Pt2dr> &resultCorner);
    void dispPtIntertFAST(Im2D<unsigned char, int> pic, vector<Pt2dr>  pts, int zoomF, string filename);
    void outElPackHomo(vector<Pt2dr> &packIn, ElPackHomologue & packOut);
private:
    bool isCorner(Pt2di pxlCenter, Im2D<unsigned char, int> &pic, double & threshold);
    bool isConsecutive(vector<int> &pxlValidIndx);
    double threshold;
    int radius;
    vector<Pt2dr> lastDetectResult;
};


#endif
