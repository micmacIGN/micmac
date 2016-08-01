#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdio.h>
#include "StdAfx.h"
#include "Fast.h"
#include "Pic.h"
#include "InitOutil.h"
class Detector
{
    public:
        Detector( string typeDetector, vector<double> paramDetector,
                        string nameImg,
                        Im2D<unsigned char, int> * img,
                        InitOutil * aChain
                        );
        Detector( InitOutil * aChain , pic * pic1 , pic * pic2);   //from pack homo Init

        int detect();
        vector<Pt2dr> readResultDetectFromFileDigeo(string filename);
        vector<Pt2dr> importFromHomolInit(pic* pic2);
        void saveToPicTypeVector(pic* aPic);
        void saveToPicTypePackHomol(pic* aPic);
        void saveResultToDiskTypeDigeo(string aDir);
    private:
        string mNameImg;
        string mTypeDetector;           //FAST, DIGEO, HOMOLINIT
        vector<double> mParamDetector;
        vector<Pt2dr> mPtsInterest;
        Im2D<unsigned char, int> *mImg;
        InitOutil * mChain;
        pic* mPic2;
};
#endif
