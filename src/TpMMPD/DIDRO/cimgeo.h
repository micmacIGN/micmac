#ifndef CIMGEO_H
#define CIMGEO_H
#include "StdAfx.h"
#include "cransac_2dline.h"

// pour manipuler des images géorérérencée avec .tfw file, des ortho individuelles générées avec Malt par ex.
class cImGeo
{
public:
    //constructeur
    cImGeo(std::string aName, std::string aNameTFW);
    cImGeo(std::string aName);
    cImGeo(cImGeo * imGeoTemplate,std::string aName);
    cImGeo(std::string aName,Im2D_REAL4 * aIm,Box2dr aBoxTer);

    // accesseur
    int nbPix(){return mSzImPix.x*mSzImPix.y;}
    Pt2di SzUV(){return mSzImPix;}
    Pt2dr SzXY(){return mSzImTer;}
    Pt2dr OriginePlani() {return mOrigine;}
    double Xmax() {return mXmax;}
    double Xmin() {return mXmin;}
    double Ymax() {return mYmax;}
    double Ymin() {return mYmin;}
    Pt2dr center() {return mCentre;}
    Box2dr boxTer() {return mBoxTer;}
    double GSD() {return mGSD;}
    std::string Name(){return mName;}
    Tiff_Im Im() {return  mIm;}
    Im2D_REAL4 * Incid() {return  &mIncid;}

    // méthodes
    bool overlap(cImGeo * aIm2);
    bool overlap(cImGeo * aIm2,int aRec); // au minimum aRec (en pourcent) de recouvrement)
    int pixCommun(cImGeo * aIm2); // nombre de pixel en commun
    Pt2di computeTrans(cImGeo * aIm2); // retourne la translation pixel e im1 et im2
    Pt2di computeTrans(Pt2dr aPTer);
    void applyTrans(Pt2di aTr);
    //Pt2di overlapBox(cImGeo * aIm2);
    Tiff_Im clip(Pt2di aBox);
    void Save(const std::string & aName);
    int transTFW(Pt2di aTrPix);
    int writeTFW();
    int writeTFW(std::string aName);

    // je devrais pouvoir utiliser toutes ces fontions aussi bien pour l'incidence,le masque et la radiométrie.
    Im2D_REAL4 toRAM(); // copie l'image tiff dans la ram pour la manipuler avec elise
    Im2D_REAL4 clipImPix(Pt2di aMin,Pt2di aMax); // clip l'image avec une box pixel
    Im2D_REAL4 clipImTer(Pt2dr aMin,Pt2dr aMax); // clip l'image avec une box terrain
    Im2D_REAL4 clipImTer(Box2dr aBox); // clip l'image avec une box terrain
    Im2D_REAL4 clipIncidTer(Box2dr aBox);
    Im2D_REAL4 clipIncidPix(Pt2di aMin,Pt2di aMax);

    Box2dr overlapBox(cImGeo * aIm2); // renvoie la box terrain du recouvrement des 2 images
    bool containTer(Pt2dr pt);
    int updateTiffIm(Im2D_REAL4 * aIm);
    void display();
    //create empty image 2D from box and resolution
    Im2D_REAL4 box2Im(Box2dr aBox);
    // effectue la correction radiométrique au moyen d'un modele linéaire 2D
    Im2D_REAL4 applyRE(c2DLineModel aMod);
    Box2dr boxEnglob(cImGeo * aIm2);
    void loadIncid();
    Pt2di XY2UV(Pt2dr XY);
    Pt2dr UV2XY(Pt2di UV);
private:
    //Pt2di X2U(Pt2dr X); // je passe un point contenant xmin et xmax et il me retourne u min u max
    //Pt2di Y2V(Pt2dr Y);


    Pt2dr mCentre;
    Box2dr mBoxTer;
    std::string mName,mDir;
    double mGSD;
    Tiff_Im mIm;
    Im2D_REAL4 mIncid; // j'ai déjà fait une erreur en intégrant mIM comme Tiff_IM, ça aurait été plus maniable de l'avoir comme Im2D_REAL4. pour incid je fait comme cela

    double mXmin, mXmax, mYmin, mYmax;
    Pt2dr mSzImTer,mOrigine;
    Pt2di mSzImPix;
};

std::vector<double> loadTFW(std::string aNameTFW);
cFileOriMnt TFW2FileOriMnt(std::string aTFWName);

#endif // CIMGEO_H
