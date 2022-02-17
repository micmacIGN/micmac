#ifndef IMAGESIMPLEPROJECTION_H
#define IMAGESIMPLEPROJECTION_H

#include "StdAfx.h"
#include "TpPPMD.h"

/*
    script par JL, avril 2015
    Motivations: in the context of wildlife census by means of unmanned aerial surveys, the operator has to handle lot of aerial images
    These aerial images are not devoted to mapping (no sufficient overlap for Structure form Motion, by exemple)
    Operator are willing to measure the size of animals, a georectification of the images is thus required.

   example :
    cd /media/jo/Data/Project_Photogrammetry/Exo_MM/Ex1_Felenne
       mm3d TestLib ImageRectification R00418.*.JPG Ori-sub-BL72/ Show=1

     to do: generate shp polygon with image footprint in shp, option for doing only this. choice RGB or blackandwhite.
*/

// List  of classes

// Class to store the application of Image Simple Rectification
class cISR_Appli;
// Contains the information about each image : Geometry (radiometric is handled by cISR_ColorImg)
class cISR_Ima;
// for each images, 2 colorImg instance; one for the image itself, one for the Rectified image
class cISR_ColorImg;
// class for a color point
class cISR_Color;

// classes declaration

class cISR_Appli
{
public :

    cISR_Appli(int argc, char** argv);
    const std::string & Dir() const {return mDir;}
    bool ShowArgs() const {return mShowArgs;}
    std::string NameIm2NameOri(const std::string &) const;
    cInterfChantierNameManipulateur * ICNM() const {return mICNM;}
    int mFlightAlti;
    int mDeZoom;
    std::string mOri;

    void Appli_InitGeomTerrain();
    void Appli_InitHomography();
    void Appli_ChangeGeomTerrain();
    void Appli_ApplyImProj(bool aShow);
    void Appli_ApplyImHomography(bool aShow);

private :
    cISR_Appli(const cISR_Appli &); // To avoid unwanted copies

    void DoShowArgs1();
    void DoShowArgs2(int aKIm);
    std::string mFullName;
    std::string mDir;
    std::string mPat;
    std::string mPrefixOut;
    std::list<std::string> mLFile;
    cInterfChantierNameManipulateur * mICNM;
    std::vector<cISR_Ima *>           mIms;
    bool                              mShowArgs;
    bool                              mByHomography;
    bool                              mQuickResampling;
};


class cISR_Ima
{
public:
    cISR_Ima(cISR_Appli & anAppli,const std::string & aName,int aAlti,int aDZ, std::string & aPrefix,bool aQuick);
    void InitGeomTerrain();
    void InitGeom();
    void ApplyImProj();
    void GenTFW();
    void GenTFW(double mGSD, Pt2dr offset, string aPrefix);
    void WriteImage(cISR_ColorImg & aImage);
    void WriteImage(cISR_ColorImg & aImage, string aPrefix);
    //void InitMemImProjHomography();
    void InitHomography();
    void ApplyImHomography();
    void ChangeGeomTerrain();
    Pt2di SzUV(){return mSzIm;}
    Pt2di SzXY(){return mSzImRect;}
    std::string Name(){return mName;}
    bool  DepthIsDefined(){return mCam->ProfIsDef();}
    void Estime4PtsProjectiveTransformation();
    void RectifyByProjectiveTransformation(vector<Pt2dr> aVp, vector<Pt3dr> aVP, vector<double> aParamProj);
    void RectifyByHomography();


    int 			   mAlti;
    int			   mZTerrain;
    int			   mDeZoom;
    int			   mBorder[4]; // the extent border of the rectified image
    double		   mIGSD; // Initial ground sample distance
    double		   mFGSD; // Final ground sample distance, after resample (goal= decrease the size of the resulting rectified image)
    double		   mLoopGSD; // The GSD used during the rectification in the loop, = IGSD if QuickResampling=0, =FGSD if QuickResampling=1
    bool           mQuickResampling;

private :
    cISR_Appli &    mAppli;
    std::string     mName;
    std::string     mNameTiff;
    Pt2di           mSzIm;
    Pt2di           mSzImRect;
    std::string     mNameOri;
    std::string	    mPrefix;
    CamStenope *    mCam;
    cElHomographie	mH;
};

//Color image
class cISR_ColorImg
{
public:
    cISR_ColorImg(std::string filename);
    cISR_ColorImg(Pt2di sz);
    ~cISR_ColorImg();
    cISR_Color get(Pt2di pt);
    cISR_Color getr(Pt2dr pt);
    void set(Pt2di pt, cISR_Color color);
    void write(std::string filename);
    cISR_ColorImg ResampleColorImg(double aFact);
    Pt2di sz(){return mImgSz;}
protected:
    std::string mImgName;
    Pt2di mImgSz;
    Im2D<U_INT1,INT4> *mImgR;
    Im2D<U_INT1,INT4> *mImgG;
    Im2D<U_INT1,INT4> *mImgB;
    TIm2D<U_INT1,INT4> *mImgRT;
    TIm2D<U_INT1,INT4> *mImgGT;
    TIm2D<U_INT1,INT4> *mImgBT;
};

//color value class. just one point
class cISR_Color
{
public:
    cISR_Color(U_INT1 r,U_INT1 g,U_INT1 b):mR(r),mG(g),mB(b){} // constructor
    void setR(U_INT1 r){mR=r;}
    void setG(U_INT1 g){mG=g;}
    void setB(U_INT1 b){mB=b;}
    U_INT1 r(){return mR;}
    U_INT1 g(){return mG;}
    U_INT1 b(){return mB;}
protected:
    U_INT1 mR;
    U_INT1 mG;
    U_INT1 mB;
};

#endif // IMAGESIMPLEPROJECTION_H

