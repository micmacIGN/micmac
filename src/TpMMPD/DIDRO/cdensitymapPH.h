#ifndef CDENSITYMAPPH_H
#define CDENSITYMAPPH_H
#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"
#include "../imagesimpleprojection.h"

extern int writeTFW(std::string aNameTiffFile, Pt2dr aGSD, Pt2dr aXminYmax);
extern std::string KeyAssocNameTif2TFW(std::string aOrtName);

Fonc_Num GaussBlur(Fonc_Num f);

// create a map of the entire area with very low resolution that depict the density of tie point (point homologue PH)

class cDensityMapPH
{
public:
    cDensityMapPH(int argc,char ** argv);

private:

    void determineGSD();
    void determineMosaicFootprint(); // based on orientation file
    void initDensityMap(); // initialise density map
    void loadPH(); // load Tie point 2D
    void populateDensityMap(); // compute 3D position for each TP and modify
    void populateDensityMap4Tests(); // Test purposes
    Pt2di XY2UV(Pt2dr aVal);

    cInterfChantierNameManipulateur * mICNM;

    bool mExpTxt,mDebug;
    std::string mSH,mDir,mOriPat,mOut,mFileSH;
    std::list<std::string> mOriFL;// OriFileList
    double mThreshResid;
    Box2dr mBoxTerrain;
    Pt2di mSz;
    int mWidth;
    double mGSD;
    cSetTiePMul * mTPM;
    Im2D_REAL4 mDM;
    std::vector<std::string> mImName;

    bool mSmoothing, mMultiplicity,mResid;
    std::map<int, CamStenope*> mCams;

};


// manipulate the new format of tie points (cSetTiePMul)
class cManipulate_NF_TP
{
public:
    cManipulate_NF_TP();
    // load tie point, orienation, list of images
    void init();
    // most of the time, no need to access radiometric value of image. But for versatility purpose, there is the option to load all image in a map
    const bool WithRadiom(){return mWithRadiometry;}
    void loadIm();
    LArgMain &  ArgCMNF()    {return (*mArgComp);}
    LArgMain &  ArgOMNF()    {return (*mArgOpt);}

    LArgMain                  *mArgComp,*mArgOpt;
    cInterfChantierNameManipulateur * mICNM;
    bool mDebug;
    bool mWithRadiometry;
    std::string mDir,mOriPat,mOut,mFileSH;
    std::list<std::string> mOriFL;// xml Orientation File List
    cSetTiePMul * mTPM;
    // vector of image names
    std::vector<std::string> mImName;
    // map indexed by ID of image containing the CamStenope of the image, which is the orientation of the camera (external and calibration)
    std::map<int, CamStenope*> mCams;
    // map indexed by ID of image containing 3 canal RGB images
    std::map<int, cISR_ColorImg*> mIms;
};

// aim: return 3D position for each tie point in 3D Appui micmac format
class cAppli_IntersectBundleHomol : public cManipulate_NF_TP
{
public:
    cAppli_IntersectBundleHomol(int argc,char ** argv);
private:
    std::string mStr0;

};

#endif // CDENSITYMAPPH_H
