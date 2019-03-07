#ifndef CTHERMICTO8BITS_H
#define CTHERMICTO8BITS_H
#include "StdAfx.h"
#include "../imagesimpleprojection.h"

// transfo digital number to degree for optris and variocam thermal camera
double DN2Deg_Optris(int aVal);
double DN2Deg_Vario(int aVal);

int Deg2DN_Optris(double aVal);
int Deg2DN_Vario(double aVal);

// color palette for displaying thermal data in RGB
int Deg2R_pal1(double aDeg, Pt2dr aRange);
int Deg2G_pal1(double aDeg, Pt2dr aRange);
int Deg2B_pal1(double aDeg, Pt2dr aRange);

// retroengeneering to copy the rgb color palette of irbis: i compare radiometry of irb file with the jpg exported by irbis software
class cDeg2RGB;
class cMeasurePalDeg2RGB{
public:
    cMeasurePalDeg2RGB(int argc,char ** argv);
    void saveMes(std::string aFileName);
    std::string AssocTherm2JPG(std::string aName);
private:
    cInterfChantierNameManipulateur * mICNM;
    bool mDebug,mVario,mOptris;
    int mNbCarToRemove;
    std::string mDirT,mDirJPG,mOut,mPre,mSu,mExt;
    std::list<std::string> mLImTherm;
    std::list<std::string> mLImJPG;
    std::map<std::string,std::string> mImJPG;

    std::vector<cDeg2RGB*> mMes;
};

class cDeg2RGB{
public:
    cDeg2RGB(double aDeg,int aR, int aG,int aB);
    double deg(){return mDeg;}
    int r(){return mR;}
    int g(){return mG;}
    int b(){return mB;}
private:
    double mDeg;
    int mR,mG,mB;
};


// thermic camera as optris and variocam record images at 16 bits, we want to convert them to 8 bits. A range of temperature is provided in order to  stretch the radiometric value on this range

class cThermicTo8Bits
{
    public:
    std::string mDir;
    cThermicTo8Bits(int argc,char ** argv);
    private:
    std::string mFullDir;
    std::string mPat;
    std::string mPrefix;
    bool mOverwrite;
    Pt2dr mRangeT;// range of temperature in celcius degree
    bool mOptris; // operate on Optris camera frame
    bool mVario;  // operate on Variocam camera frame
    bool mRGB;    // convert in RGB 8 bits images with color palette, visual purspose
    bool mDebug;
};

#endif // CTHERMICTO8BITS_H
