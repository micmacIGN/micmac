#ifndef CIMGEO_H
#define CIMGEO_H
#include "StdAfx.h"

// pour manipuler des images géorérérencée avec .tfw file, des ortho individuelles générées avec Malt par ex.
class cImGeo
{
public:
    //constructeur
    cImGeo(std::string aName);
    cImGeo(cImGeo * imGeoTemplate,std::string aName);

    // accesseur
    Pt2di SzUV(){return mSzImPix;}
    Pt2dr SzXY(){return mSzImTer;}
    Pt2dr OriginePlani() {return mOrigine;}
    double Xmax() {return mXmax;}
    double Xmin() {return mXmin;}
    double Ymax() {return mYmax;}
    double Ymin() {return mYmin;}
    double GSD() {return mGSD;}
    std::string Name(){return mName;}
    Tiff_Im Im() {return  mIm;}

    // méthodes
    bool overlap(cImGeo * aIm2);
    Pt2di computeTrans(cImGeo * aIm2); // retourne la translation pixel e im1 et im2
    void applyTrans(Pt2di aTr);
    Pt2di overlapBox(cImGeo * aIm2);
    Tiff_Im clip(Pt2di aBox);
    std::vector<double> loadTFW(std::string aNameTFW);
    void Save(const std::string & aName);
    int transTFW(Pt2di aTrPix);
    int writeTFW();
    int writeTFW(std::string aName);
    Im2D_REAL4 toRAM();
    int updateTiffIm(Im2D_REAL4 * aIm);


private:
    std::string mName;
    double mGSD;
    Tiff_Im mIm;

    double mXmin, mXmax, mYmin, mYmax;
    Pt2dr mSzImTer,mOrigine;
    Pt2di mSzImPix;

};


#endif // CIMGEO_H
