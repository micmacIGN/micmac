#ifndef MOSAICTFWIMG_H
#define MOSAICTFWIMG_H
#include "StdAfx.h"
#include "../TpPPMD.h"
#include "../imagesimpleprojection.h"


class cGeoImg;
class cAppliMosaicTFW;


class cGeoImg
{
    public:
        cGeoImg (const std::string & aName);
        string & Name() {return mName;}
        Tiff_Im & Tif() {return mTif;}
        Pt2dr & GSD() {return mGSD;}
        Pt2dr & Offset() {return mOffset;}
        Pt2di & SzPxl() {return mSzPxl;}
        Pt2dr & SzTer() {return mSzTer;}
        Pt2dr Pxl2Ter(Pt2dr aCoor);
        Pt2dr Ter2Pxl(Pt2dr aCoor);
        Pt2dr & SzTerInPxl() {return mSzTerInPxl;}
    private:
        string mName;
        Tiff_Im mTif;
        Pt2dr mGSD;
        Pt2dr mOffset;
        Pt2di mSzPxl;
        Pt2dr mSzTer;
        Pt2dr mSzTerInPxl;
};

class cAppliMosaicTFW
{
    public:
        cAppliMosaicTFW();
        cGeoImg * ImGSDMaxX() {return mImGSDMaxX;}
        cGeoImg * ImGSDMinX() {return mImGSDMinX;}
        cGeoImg * ImGSDMaxY() {return mImGSDMaxY;}
        cGeoImg * ImGSDMinY() {return mImGSDMinY;}
        cGeoImg * ImOffsetMaxX() {return mImOffsetMaxX;}
        cGeoImg * ImOffSetMinX() {return mImOffsetMinX;}
        cGeoImg * ImOffsetMaxY() {return mImOffsetMaxY;}
        cGeoImg * ImOffSetMinY() {return mImOffsetMinY;}
        vector<cGeoImg * > & VGeoImg() {return mVGeoImg;}
        void CalculParamMaxMin();
        Pt2dr & OffsetGlobal() {return mOffsetGlobal;}
        bool mDisp;
        int mDeZoom;
        void WriteImage(cISR_ColorImg & aImage, string aName);
    private:
        cGeoImg * mImGSDMaxX;
        cGeoImg * mImGSDMinX;
        cGeoImg * mImGSDMaxY;
        cGeoImg * mImGSDMinY;
        cGeoImg * mImOffsetMaxX;
        cGeoImg * mImOffsetMinX;
        cGeoImg * mImOffsetMaxY;
        cGeoImg * mImOffsetMinY;
        vector<cGeoImg * > mVGeoImg;
        Pt2dr mOffsetGlobal;
        Pt2dr mSzBoxTerrainMosaicPxl;
        Pt2dr mGSDMoy;

};




#endif // MOSAICTFWIMG_H
