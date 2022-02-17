#ifndef CTAPIOCA_IDR_H
#define CTAPIOCA_IDR_H
#include "StdAfx.h"

/*
 * When we combine two set of images, with different resolution, and we call tapioca
 * , the argument of image size is not appropriate --> tapioca/pastis think that all input images have the same
 * resolution
 *
 * This is annoying when we want to compute Tie Points between images set of 1000 pixel (imgSensor1) and images of 5000 pixels (imgSensor2)
 * Indeed, tapioca will determine that resolution is full if size parameter is 1000, and thus tie point are computed at 5000 pixels size --> very long
 *
 * solution: resize all the image prior to tapioca process, then scale the resulting tie points pack
 * */

class cTapioca_IDR
{
public:
    cTapioca_IDR(int argc, char** argv);
private:
    void purgeTmpFile();
    void resizeImg();
    void scaleTP();
    void mergeHomol();
    int runTapioca();
    std::string imCple2HomolFileName();

    cInterfChantierNameManipulateur * mICNM;
    std::string mImPat,mMode,mPatOrFile,mDetect;
    double mRatio;
    std::map<std::string,double> mImRatio; // for each images, the changing scale ratio
    std::list<std::string> mLFile;


    bool mIsSFS;
    int mNbNb,mLowRes,mExpTxt;
    bool mPurge;
    int mImLengthOut;
    std::string mTapiocaCom;
    std::string mSH_post; // Set of Homol postfix
    std::string mTmpDir, mDir;
    bool mMergeHomol;
    bool mDebug;
    std::string mHomolIDR,mHomolFormat;

};


class cResizeImg
{
public:
    cResizeImg(int argc, char** argv);
private:
    Pt2di readImSz();
    std::string mImNameIn, mImNameOut;
    int mImLengthOut;
    Pt2di mImSzOut, mImSzIn;
    std::string mDir;
    bool mF;
};


class cResizeHomol
{
public:
    cResizeHomol(int argc, char** argv);
private:

    std::string mHomolNameIn, mHomolNameOut;
    double mR1,mR2; // ratio for image 1 and 2
};













#endif // CTAPIOCA_IDR_H
