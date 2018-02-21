#ifndef TIEPTRIFAR_H
#define TIEPTRIFAR_H

#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../TaskCorrel/TaskCorrel.h"
#include <stack>
#include <iostream>

extern bool convexHull(vector<Pt2dr> points, stack<Pt2dr> S);

class cParamTiepTriFar;
class cAppliTiepTriFar;
class cImgTieTriFar;

class cAppliTiepTriFar
{
    public:
        cAppliTiepTriFar (cParamTiepTriFar & aParam,
                          cInterfChantierNameManipulateur * aICNM,
                          vector<cImgTieTriFar> & vImg,
                          string & aDir,
                          string & aOri
                         );

        cParamTiepTriFar & Param() {return mParam;}
        void LoadMesh(string & aMeshName);
        string & Dir() {return mDir;}
        string & Ori() {return mOri;}
        cInterfChantierNameManipulateur * ICNM() {return mICNM;}
        // FROM 3D MESH TO 2D MASK
        void loadMask2D();

    private:
        cParamTiepTriFar & mParam;
        vector<cImgTieTriFar> & mvImg;
        string & mDir;
        string & mOri;

        vector<cTri3D> mVTri3D;
        cInterfChantierNameManipulateur * mICNM;



};

class cImgTieTriFar
{
  public :
        cImgTieTriFar(cAppliTiepTriFar & aAppli, string & aName);

        Tiff_Im   Tif() {return mTif;}

        string NameIm() {return mNameIm;}

        Pt2di & SzIm() {return mSzIm;}


        cBasicGeomCap3D * CamGen() {return mCamGen;}
        CamStenope *      CamSten() {return mCamSten;}

        vector<Pt2dr> & SetVertices() {return mSetVertices;}

  private :
        cAppliTiepTriFar & mAppli;

        Tiff_Im   mTif;

        string    mNameIm;

        Pt2di     mSzIm;

        Im2D_Bits<1> mMasqIm;
        TIm2DBits<1> mTMasqIm;

        vector<Pt2dr> vPolyMask;

        cBasicGeomCap3D * mCamGen;
        CamStenope *      mCamSten;

        vector<Pt2dr> mSetVertices;

};







#endif // TIEPTRIFAR_H

