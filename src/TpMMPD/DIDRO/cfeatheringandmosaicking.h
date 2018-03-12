#ifndef CFEATHERINGANDMOSAICKING_H
#define CFEATHERINGANDMOSAICKING_H
#include "cimgeo.h"


class  cFeatheringAndMosaicOrtho
{
    public:
    std::string mDir;
    cFeatheringAndMosaicOrtho(int argc,char ** argv);
    void ChamferNoBorder(Im2D<U_INT1,INT> i2d) const;
    template <class T,class TB> void SaveTiff(std::string & aName,  Im2D<T,TB> * aIm);


    void ChamferDist4AllOrt();
    void WeightingNbIm1and2();
    void WeightingNbIm3();
    void ComputeMosaic();

    private:
    std::map<int, cImGeo*> mIms; // key=label
    std::map<int, Im2D_REAL4>   mIm2Ds;
    std::map<int, Im2D_INT2>  mChamferDist;
    std::map<int, Im2D_REAL4>   mImWs;
    //std::map<int, Im2D_U_INT1>  mBlendingArea;
    //std::map<int, Im2D_U_INT1>  mMosaicArea;
    std::string mNameMosaicOut, mFullDir;
    int mDist;
    std::list<std::string> mLFile;
    cInterfChantierNameManipulateur * mICNM;
    Box2dr mBoxOverlapTerrain;
    double mLambda;
    // coin de l'ortho
    Pt2dr aCorner;
    Pt2di sz;
    bool mDebug;
    Im1D_REAL4 lut_w;

    Im2D_REAL4 mosaic;
    Im2D_REAL4 NbIm,PondInterne;
    // sum of chamfer distance , on outside the enveloppe (positive value) and on inside the enveloppe (negative value)
    Im2D_INT4 mSumDistInter;
    Im2D_INT4 mSumDistExt;
    // to check that sum weighting is equal to 1 everywhere
    Im2D_REAL4 mSumWeighting;
    Im2D_U_INT1 label;

};

#endif // CFEATHERINGANDMOSAICKING_H
