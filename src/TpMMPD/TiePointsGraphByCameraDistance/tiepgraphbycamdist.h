#include "StdAfx.h"

/*
Info TiepGraphByCamDist :
- This tool is inspired by a function in Pix4D tie points extraction
- Generate an image pair for tie point search, constraint by distance between camera center (given by GPS or anything else)
- Given Ori folder or Navigation text file (format N_X_Y_Z_W_P_K as file for OriConvert tool)

  => User gives "Average distance" between consecutive images, tools will create a sphere with center is this image center and radius of Average distance,
 , and will generate match couples of this image with all other images included in the sphere.
*/

#define LINE_EMPTY -1


class cOneImg
{
    public:
        cOneImg(string aName, Pt3dr aP);
        cOneImg(string aName, Pt3dr aP, Pt3dr aOri);
        string mName;
        Pt3dr mPt;  // X Y Z
        Pt3dr mOri; // W P K
};

cOneImg::cOneImg(string aName, Pt3dr aP) :
    mName (aName),
    mPt (aP)
{}

cOneImg::cOneImg(string aName, Pt3dr aP, Pt3dr aOri) :
    mName (aName),
    mPt (aP),
    mOri (aOri)
{}

class cAppliTiepGraphByCamDist
{
    public :
        cAppliTiepGraphByCamDist();
        void ImportNavInfo(string & aName);
        void ComputeImagePair(double aD);
        void ExportGraphFile(string & aFName);
        int parseLine(string & aL, vector<string> & aVWord);
        std::map<string, cOneImg*> & Map_NameIm_NavInfo(){return mMap_NameIm_NavInfo;}
        std::vector<string> & VNameIm() {return mVNameIm;}
        cSauvegardeNamedRel & RelXML() {return mRelXML;}
        bool & Inv() {return mInv;}
        bool & OriMode() {return mOriMode;}
    private :
        bool ImportOri(string & aFName);
        void ImportMicMacOriFolder(string & aFolderName);
        std::map<string, cOneImg*> mMap_NameIm_NavInfo;
        std::vector<string> mVNameIm;
        cSauvegardeNamedRel mRelXML;
        bool mInv;
        bool mOriMode;
};

