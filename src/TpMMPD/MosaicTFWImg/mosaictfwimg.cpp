#include "mosaictfwimg.h"


void cAppliMosaicTFW::WriteImage(cISR_ColorImg & aImage, string aName)
{
    // performed a resample of the rectified image.
    if ((mDeZoom!=1))
    {
        cISR_ColorImg ImResampled = aImage.ResampleColorImg(mDeZoom);
        ImResampled.write(aName);
    }
    else aImage.write(aName);
}

Pt2dr cGeoImg::Pxl2Ter(Pt2dr aCoor)
{
    Pt2dr aTer;
    aTer.x = mGSD.x * aCoor.x + mOffset.x;
    aTer.y = mGSD.y * aCoor.y + mOffset.y;
    return aTer;
}

Pt2dr cGeoImg::Ter2Pxl(Pt2dr aCoor)
{
    Pt2dr aImg;
    aImg.x = (aCoor.x - mOffset.x)/mGSD.x;
    aImg.y = (aCoor.y - mOffset.y)/mGSD.y;
    return aImg;
}

cGeoImg::cGeoImg(const string & aName) :
    mName (aName),
    mTif  (Tiff_Im::UnivConvStd(aName))
{
    cout.precision(10);
    size_t lastindex = aName.find_last_of(".");
    string rawname = aName.substr(0, lastindex);

    std::string aNameTFW = rawname + ".tfw";

    std::ifstream aFp(aNameTFW);
    double a;
    int cnt = 0;
    while (aFp >> a)
    {
        cnt++;
        if (cnt == 1)
            mGSD.x = a;
        if (cnt == 4)
            mGSD.y = a;
        if (cnt == 5)
            mOffset.x = a;
        if (cnt == 6)
            mOffset.y = a;
    }
    aFp.close();
    mSzPxl = mTif.sz();
    cout<<"Import image : "<<mName<<endl;

    // Calcul coordonnées 4 coins images sur terrain
    Pt2dr aHGTer = Pxl2Ter(Pt2dr(0,0));
    Pt2dr aBGTer = Pxl2Ter(Pt2dr(0,mSzPxl.y));
    Pt2dr aHDTer = Pxl2Ter(Pt2dr(mSzPxl.x,0));
    Pt2dr aBDTer = Pxl2Ter(Pt2dr(mSzPxl));
    // tirer le minXYTer et max XYTer
    double maxX = ElMax4(aHGTer.x, aBGTer.x, aHDTer.x, aBDTer.x);
    double minY = ElMin4(aHGTer.y, aBGTer.y, aHDTer.y, aBDTer.y);
    double minX = ElMin4(aHGTer.x, aBGTer.x, aHDTer.x, aBDTer.x);
    double maxY = ElMax4(aHGTer.y, aBGTer.y, aHDTer.y, aBDTer.y);
    // calculer la taille box terrain
    mSzTer = Pt2dr(maxX, maxY) - Pt2dr(minX, minY);
    mSzTerInPxl = Pt2dr(abs(mSzTer.x / mGSD.x), abs(mSzTer.y / mGSD.y));
    cout<<" + SzImgRectifi(Pxl) "<<mSzPxl<<" + SzImTerrain(m) "<<mSzTer<<" + SzImTerrain(pxl) "<<mSzTerInPxl<<endl;

}

cAppliMosaicTFW::cAppliMosaicTFW():
    mSzBoxTerrainMosaicPxl(0,0),
    mGSDMoy (0,0)
{}



void cAppliMosaicTFW::CalculParamMaxMin()
{
    Pt2dr aPtMaxTerHD;

    vector<double>VGSD_x;
    vector<double>VGSD_y;
    vector<double>VOffset_x;
    vector<double>VOffset_y;

    vector<double>VBox_x;
    vector<double>VBox_y;

    vector<double>VBoxPxl_x;
    vector<double>VBoxPxl_y;

    for (uint aKIm = 0; aKIm<mVGeoImg.size(); aKIm++)
    {
        cGeoImg * aImg = mVGeoImg[aKIm];
        VGSD_x.push_back(aImg->GSD().x);
        VGSD_y.push_back(aImg->GSD().y);
        VOffset_x.push_back(aImg->Offset().x);
        VOffset_y.push_back(aImg->Offset().y);
        VBox_x.push_back(aImg->SzTer().x + aImg->Offset().x);
        VBox_y.push_back(aImg->SzTer().y + aImg->Offset().y);
        VBoxPxl_x.push_back(aImg->SzTerInPxl().x);
        VBoxPxl_y.push_back(aImg->SzTerInPxl().y);
        mGSDMoy = mGSDMoy + aImg->GSD().AbsP();
    }

    mGSDMoy = mGSDMoy/double(mVGeoImg.size());

    vector<double>::iterator it;

    it = std::max_element(VGSD_x.begin(), VGSD_x.end());
    mImGSDMaxX = mVGeoImg[int(it - VGSD_x.begin())];

    it = std::max_element(VGSD_y.begin(), VGSD_y.end());
    mImGSDMaxY = mVGeoImg[int(it - VGSD_y.begin())];

    it = std::min_element(VGSD_x.begin(), VGSD_x.end());
    mImGSDMinX = mVGeoImg[int(it - VGSD_x.begin())];

    it = std::min_element(VGSD_y.begin(), VGSD_y.end());
    mImGSDMinY = mVGeoImg[int(it - VGSD_y.begin())];

    it = std::max_element(VOffset_x.begin(), VOffset_x.end());
    mImOffsetMaxX = mVGeoImg[int(it - VOffset_x.begin())];

    it = std::max_element(VOffset_y.begin(), VOffset_y.end());
    mImOffsetMaxY = mVGeoImg[int(it - VOffset_y.begin())];

    it = std::min_element(VOffset_x.begin(), VOffset_x.end());
    mImOffsetMinX = mVGeoImg[int(it - VOffset_x.begin())];

    it = std::min_element(VOffset_y.begin(), VOffset_y.end());
    mImOffsetMinY = mVGeoImg[int(it - VOffset_y.begin())];

    it = std::max_element(VBox_x.begin(), VBox_x.end());
    aPtMaxTerHD.x = *it;

    it = std::max_element(VBox_y.begin(), VBox_y.end());
    aPtMaxTerHD.y = *it;

    it = std::max_element(VBoxPxl_x.begin(), VBoxPxl_x.end());
    mSzBoxTerrainMosaicPxl.x = *it;

    it = std::max_element(VBoxPxl_y.begin(), VBoxPxl_y.end());
    mSzBoxTerrainMosaicPxl.y = *it;

    mOffsetGlobal = Pt2dr(mImOffsetMinX->Offset().x, mImOffsetMinY->Offset().y);

    // rectification direct sens image-terrain
    // calcul taille d'image mosaic à partir de sz terrain
    Pt2di aSzMosaic(0,0);
    Pt2dr aSzMosaicTer = aPtMaxTerHD-mOffsetGlobal;
    cout<<endl<<"OffsetGlob : "<<mOffsetGlobal<<" -MaxTer : "<<aPtMaxTerHD<<" -SzTerBox "<<aSzMosaicTer<<endl;
    aSzMosaic = Pt2di(aSzMosaicTer.x/mGSDMoy.x, aSzMosaicTer.y/mGSDMoy.y);

    cout<<" + Sz mosaic "<<aSzMosaic<<endl;
    cout<<"Rectifi ... "<<endl<<endl;
    cISR_ColorImg ImColRect(aSzMosaic);
    for (uint aKIm = 0; aKIm<mVGeoImg.size(); aKIm++)
    {
        cGeoImg * aImg = mVGeoImg[aKIm];
        cout<<" +"<<aImg->Name()<<endl;
        Pt2di aPt(0,0);
        cISR_ColorImg ImCol(aImg->Name().c_str());
        for (aPt.x=0; aPt.x<aImg->SzPxl().x; aPt.x++)
        {
            for (aPt.y=0; aPt.y<aImg->SzPxl().y; aPt.y++)
            {
                // calcul pt3D
                Pt2dr aPtTer = aImg->Pxl2Ter(Pt2dr(aPt));
                // transfer from pt3d terrain to pt3d image
                //Pt2dr aPtRec = aPtTer-mOffsetGlobal;
                // transfer from pt3d image to pt3d image (pixel)
                Pt2dr aPtRecPxl(abs(aPtTer.x/aImg->GSD().x), abs(aPtTer.y/aImg->GSD().y));
                Pt2di aPtRecPxlInt(aPtRecPxl);
                if (aPtRecPxlInt.x < aSzMosaic.x && aPtRecPxlInt.y < aSzMosaic.y && aPtRecPxlInt.x > 0 && aPtRecPxlInt.y > 0)
                {
                    cISR_Color aCol=ImCol.get(aPt);
                    if (     static_cast<int>(aCol.b()) != 0
                          && static_cast<int>(aCol.g()) != 0
                          && static_cast<int>(aCol.r()) != 0
                       )
                    {
                        if (mDisp)
                            cout<<aPt<<aPtTer<<aPtRecPxlInt<<" -> Set couleur "<<endl;
                        ImColRect.set(Pt2di(aPtRecPxlInt),aCol);
                    }
                }
            }
        }
    }
    // write the rectified image in the working directory
    std::string aNameImMosaic = "OrthoMosaic.tif";
    ImColRect.write(aNameImMosaic);
    cout<<"done"<<endl;
}


int MosaicTFW(int argc, char** argv)
{
    cAppliMosaicTFW aAppli;
    aAppli.mDisp = 0;
    aAppli.mDeZoom = 4;
    cout.precision(10);
    string aDir, aPat, aFullName;
    bool aDoMosaic = false;
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pat)"),
                LArgMain()
                << EAM(aAppli.mDisp, "Disp",true,"Disp")
                << EAM(aDoMosaic, "Mosaic",true,"Do mosaic")
                << EAM(aAppli.mDeZoom, "ZoomF",true,"ZoomF")
                );
    // Initialize name manipulator & files
    SplitDirAndFile(aDir,aPat,aFullName);
    // define the "working directory" of this session
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    // create the list of images starting from the regular expression (Pattern)
    std::list<std::string> aLImg = aICNM->StdGetListOfFile(aPat);


    for  (
          std::list<std::string>::iterator itS=aLImg.begin();
          itS!=aLImg.end();
          itS++
          )
    {
        cGeoImg * aNewIm = new  cGeoImg(*itS);
        aAppli.VGeoImg().push_back(aNewIm);
    }
    aAppli.CalculParamMaxMin();

    return EXIT_SUCCESS;
}
