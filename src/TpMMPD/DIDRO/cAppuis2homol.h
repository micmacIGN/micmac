#ifndef CAPPUIS2HOMOL_H
#define CAPPUIS2HOMOL_H
#include "StdAfx.h"

// goal: convert 2D measure of appuis (from saisie appuis tools ) to homol format
class cAppuis2Homol
{
public:
    cAppuis2Homol(int argc, char** argv);
private:
    cInterfChantierNameManipulateur * mICNM;
    bool mDebug,mExpTxt;
    std::string mIm1,mIm2,mHomPackOut, mSH, m2DMesFileName;
};

#endif // CAPPUIS2HOMOL_H
