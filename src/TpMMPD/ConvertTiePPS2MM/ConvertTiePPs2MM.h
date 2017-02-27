#include "StdAfx.h"

class cHomolPS;
class cAppliConvertTiePPs2MM;

class cHomolPS
{
public:
    cHomolPS(	std::string aImgName,
                int aId,
                Pt2dr aCoord,
                int aIdImg
             );
    std::string ImgName() {return mImgName;}
    int Id() {return mId;}
    Pt2dr Coord() {return mCoord;}
    int & IdImg() {return mIdImg;}
private:
    std::string mImgName;
    int mId;
    Pt2dr mCoord;
    int mIdImg;

};

class cOneImg
{
public:
    cOneImg(string aImgName);
    std::string ImgName() {return mImgName;}
    vector<bool> & VIsIdExist() {return mVIsIdExist;}
    vector <Pt2dr> & VCoor() {return mVCoor;}
    int & IdMax() {return mIdMax;}
    vector<ElPackHomologue*> & Pack() {return mPack;}

private:
    std::string mImgName;
    vector<bool> mVIsIdExist;
    vector <Pt2dr> mVCoor ;
    int mIdMax;
    vector<ElPackHomologue*> mPack;
};

class cAppliConvertTiePPs2MM
{
    public:
        cAppliConvertTiePPs2MM();
        bool readPSTxtFile(string aPSHomol, vector<cHomolPS*> & VRHomolPS);
        vector<cHomolPS*> & VHomolPS() {return mVHomolPS;}
        vector<cOneImg*> & ImgUnique() {return mImgUnique;}
        int getId (int aId, vector<cHomolPS*> &aVRItem);
        int & IdMaxGlobal() {return mIdMaxGlobal;}
        void initAllPackHomol(vector<cOneImg*> VImg);
        void addToHomol(vector<cHomolPS*> aVItemHaveSameId,  Pt2dr aCPS, Pt2dr aSizePS);
        void writeToDisk(string aOut, bool a2W, string mDir);
    private:
        vector<cHomolPS*> mVHomolPS;
        vector<cOneImg*> mImgUnique;
        int mIdMaxGlobal;
};



