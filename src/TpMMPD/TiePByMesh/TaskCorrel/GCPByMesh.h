#include <stdio.h>
#include "StdAfx.h"


class cOneImgMesure;

extern void ReadXMLMesurePts(string aPath, vector<cOneImgMesure*> & aVImgMesure);

class cOneImgMesure
{
public:
    cOneImgMesure();
    string nameImg;
    vector<string> vNamePt;
    vector<Pt2dr> vMesure;
};






