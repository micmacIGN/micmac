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
