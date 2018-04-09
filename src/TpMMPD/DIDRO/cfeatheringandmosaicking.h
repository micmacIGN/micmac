#ifndef CFEATHERINGANDMOSAICKING_H
#define CFEATHERINGANDMOSAICKING_H
#include "cimgeo.h"

// member class that is used both by appli "mother" and software "box"
class  cMyICNM
{
public:
    cMyICNM(cInterfChantierNameManipulateur * aICNM, std::string aTmpDir);
    std::string nameNbIm(){return mNameNbIm;}
    std::string nameIW(){return mNameInternalW;}
    std::string nameSDE(){return mNameSumDistExt;}
    std::string nameSDI(){return mNameSumDistInt;}
    std::string nameSW(){return mNameSumWeighting;}
    std::string nameXmlImList(const int i){return mNameImList + ToString(i) + ".xml";}
    std::string KeyAssocNameOrt2PC(std::string aOrtName);
    std::string KeyAssocNameOrt2Incid(std::string aOrtName);
    std::string KeyAssocNameOrt2TFW(std::string aOrtName);
    std::vector<int> extractImID( Liste_Pts_INT2 * aListePt);
    int writeTFW(std::string aNameTiffFile, Pt2dr aGSD, Pt2dr aXminYmax);
    template <class T,class TB> void SaveTiff(std::string aName,  Im2D<T,TB> * aIm);
    template <class T,class TB> void SaveBoxInTiff(std::string aName,  Im2D<T,TB> * aIm,Box2di aBox);

private:
    cInterfChantierNameManipulateur * mICNM;
    // name of temporary file that have to be known by both software
    std::string mTmpDir,mNameNbIm,mNameInternalW,mNameSumDistExt,mNameSumDistInt,mNameSumWeighting;
    std::string mNameImList;
};


class  c_Appli_FeatheringAndMosaic
{
public:
    c_Appli_FeatheringAndMosaic(int argc,char ** argv);


private:
    std::string mDir,mTmpDir;
    std::string mNameMosaicOut, mFullDir, mLabel;
    Pt2di mSzTuile; // taille tuile/box
    int mDilat; //dilatation of each box, buffer
    int mNbBox;
    std::map<int, Box2di> mBoxes; // box in pixels
    std::map<int, Box2di> mDilatBoxes; // dilatation
    std::map<int, Pt2di>  mTrs; // translation between every single orthoimages and the corner of the mosaic
    int mDist;
    double mLambda;
    bool mDebug;
    std::list<std::string> mLFile;
    cInterfChantierNameManipulateur * mICNM;
    cMyICNM * mKA; // mKey assoc
    cFileOriMnt MTD;
    Pt2di sz;
    Pt2dr aCorner;
    Box2dr mBoxGlob;
    Im2D_U_INT1 label;

    void checkParam();
    void GenLabelTable();
    void GenerateTiff();
    void SplitInBoxes();
    void DoMosaicAndFeather(); // for each box
    void banniereFeathering();

};

class  cFeatheringAndMosaicOrtho
{
    public:
    int mNumBox;
    std::string mDir,mTmpDir,mListImFile;
    cFeatheringAndMosaicOrtho(int argc,char ** argv);
    void ChamferNoBorder(Im2D<U_INT1,INT> i2d) const;

    void ChamferDist4AllOrt();   // feathering along seam line is determined by the distance from the seamline, which is measured by Chamfer morphological filter
    void WeightingNbIm1and2();   // gaussian-distance weighting of 2 images in feathering area
    void WeightingNbIm3();       // if 3 images, should first blend 2 ortho and subsequently blend the result with the third ortho
    void ComputeMosaic();
    void LoadRadiomEgalLinMod(); // load radiometric egalisation models in order to apply them prior to mosaicking

    Box2di mBox; // box en pixel
    Box2di mDilat;  // dilatation buffer to remove prior result writing
    private:
    std::map<int, cImGeo*>      mIms; // key=label
    std::map<int, Im2D_REAL4>   mIm2Ds;
    std::map<int, Im2D_INT2>    mChamferDist;
    std::map<int, Im2D_REAL4>   mImWs;
    std::map<int, Pt2di>        mTrs; // translation from single ortho to mosaic corner

    std::string mNameMosaicOut, mFullDir, mLabel;
    int mDist;
    std::list<std::string> mLFile;
    cInterfChantierNameManipulateur * mICNM;
    cMyICNM * mKA; // mKey assoc
    Box2dr mBoxOverlapTerrain;
    double mLambda;
    // coin de la mosaique/box
    Pt2dr aCorner;
    Pt2di sz;
    cFileOriMnt MTD;
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
    std::string mProgressBar;

    void banniereFeatheringBox();


};



#endif // CFEATHERINGANDMOSAICKING_H
