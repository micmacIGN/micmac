#ifndef CDENSITYMAPPH_H
#define CDENSITYMAPPH_H
#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"
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

#endif // CDENSITYMAPPH_H
