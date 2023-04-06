#include "TiePHistorical.h"

void GeoreferencedDepthMap(std::string aImg1, std::string aDir, std::string aDSMDir, std::string aDSMFile, std::string aOri1, std::string aPrefix)
{
    bool bMasq = false;
    std::string aOutDir = "";
    std::string aImName = "";
    std::string aMasqName = "";
    std::string aCorrelName = "";

    Pt2di aDSMSz = Pt2di(0,0);

    if(1){
        std::string aOutDir = aDSMDir;
        aDSMDir += '/';
        cout<<aDSMDir + aDSMFile<<endl;
        cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
        (
        aDSMDir + aDSMFile,
        StdGetFileXMLSpec("SuperposImage.xml"),
        "XML_ParamNuage3DMaille",
        "XML_ParamNuage3DMaille"
        );

        aDSMSz = aNuageIn.NbPixel();

        cImage_Profondeur aImDSM = aNuageIn.Image_Profondeur().Val();

        aImName = aDSMDir + aImDSM.Image();

        aMasqName = aDSMDir + aImDSM.Masq();

        aCorrelName = aDSMDir + aImDSM.Correl().Val();
    }

    cout<<"DSM file name: "<<aImName<<endl;
    cout<<"Masq file name: "<<aMasqName<<endl;
    cout<<"Correl file name: "<<aCorrelName<<endl;

    //load mask
    TIm2D<float,double> aTImMasq(aDSMSz);
        Tiff_Im aImMasqTif = Tiff_Im::StdConvGen((aMasqName).c_str(), -1, true ,true);
        ELISE_COPY
        (
        aTImMasq.all_pts(),
        aImMasqTif.in(),
        aTImMasq.out()
        );
        bMasq = true;

    //load Correlation score
    TIm2D<float,double> aTImCorrel(aDSMSz);
        Tiff_Im aImCorrelTif = Tiff_Im::StdConvGen((aCorrelName).c_str(), -1, true ,true);
        ELISE_COPY
        (
        aTImCorrel.all_pts(),
        aImCorrelTif.in(),
        aTImCorrel.out()
        );
        bMasq = true;

    //load DSM
    Tiff_Im aImDSMTif = Tiff_Im::StdConvGen((aImName).c_str(), -1, true ,true);
    TIm2D<float,double> aTImDSM(aDSMSz);
    ELISE_COPY
    (
    aTImDSM.all_pts(),
    aImDSMTif.in(),
    aTImDSM.out()
    );

    /*
    //load Correlation score
    Tiff_Im aImCorrelTif = Tiff_Im::StdConvGen((aCorrelName).c_str(), -1, true ,true);
    TIm2D<float,double> aTImDSM(aDSMSz);
    ELISE_COPY
    (
    aTImDSM.all_pts(),
    aImCorrelTif.in(),
    aTImDSM.out()
    );*/


    Tiff_Im aRGBIm1 = Tiff_Im::StdConvGen((aDir+aImg1).c_str(), -1, true ,true);
    Pt2di ImgSzL = aRGBIm1.sz();

    cout<<"DSM size: "<<ImgSzL.x<<", "<<ImgSzL.y<<endl;

    int aType = eTIGB_Unknown;
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    cBasicGeomCap3D * aCamL = cBasicGeomCap3D::StdGetFromFile(aIm1OriFile,aType);

    //double prof_d = aCamL->GetProfondeur();

    //std::string aOutIdx = aDSMDir + "/"+aPrefix+"_Idx.txt";
    std::string aOut2DPts = aDSMDir + "/"+aPrefix+"_2DPts.txt";
    std::string aOut3DPts = aDSMDir + "/"+aPrefix+"_3DPts.txt";
    std::string aOutCorrel = aDSMDir + "/"+aPrefix+"_Correl.txt";
    //FILE * fpOutIdx = fopen((aOutIdx).c_str(), "w");
    FILE * fpOut2DPts = fopen((aOut2DPts).c_str(), "w");
    FILE * fpOut3DPts = fopen((aOut3DPts).c_str(), "w");
    FILE * fpOutCorrel = fopen((aOutCorrel).c_str(), "w");

    //cout<<"aOutIdx file name: "<<aOutIdx<<endl;
    cout<<"aOut2DPts file name: "<<aOut2DPts<<endl;
    cout<<"aOut3DPts file name: "<<aOut3DPts<<endl;
    cout<<"aOutCorrel file name: "<<aOutCorrel<<endl;

    int i, j;
    int nIdx = 0;
    double aCorrelMax = -100000;
    double aCorrelMin = 100000;
    for(i=0; i<ImgSzL.x; i++){
        for(j=0; j<ImgSzL.y; j++){
            Pt2di aP1 = Pt2di(i, j);
            int nVal =  1;
            if(bMasq == true)
                nVal = aTImMasq.get(aP1);
            if(nVal > 0){
                double prof_d = aTImDSM.get(aP1);
                //Pt3dr aPTer1 = aCamL->ImEtProf2Terrain(Pt2dr(i, j), prof_d);
                Pt3dr aPTer1 = aCamL->ImEtZ2Terrain(Pt2dr(i, j), prof_d);

                if(i==131 && j==385)
                    printf("i, j, prof_d: %d, %d, %lf, aPTer1: [%lf %lf %lf]\n", i, j, prof_d, aPTer1.x, aPTer1.y, aPTer1.z);

                //fprintf(fpOutIdx, "%d\n", nIdx);
                fprintf(fpOut2DPts, "%d %d\n", i, j);
                double aCorrel = aTImCorrel.get(aP1);
                if(aCorrel > aCorrelMax)
                    aCorrelMax = aCorrel;
                if(aCorrel < aCorrelMin)
                    aCorrelMin = aCorrel;
                fprintf(fpOutCorrel, "%lf\n", aCorrel);
                fprintf(fpOut3DPts, "%lf %lf %lf\n", aPTer1.x, aPTer1.y, aPTer1.z);
                nIdx++;
            }
            //else
            //    fprintf(fpOutIdx, "%d\n", -1);  //-1 means the point is masked out
        }
    }
    cout<<"aCorrelMin: "<<aCorrelMin<<", aCorrelMax: "<<aCorrelMax<<endl;
    //fclose(fpOutIdx);
    fclose(fpOut2DPts);
    fclose(fpOut3DPts);
    fclose(fpOutCorrel);
}

int GeoreferencedDepthMap_main(int argc,char ** argv)
{
    /*
   cCommonAppliTiepHistorical aCAS3D;
*/
    std::string aOri1;
    std::string aImg1;

    std::string aDir = "./";

    std::string aDSMDir;
    std::string aDSMFile = "MMLastNuage.xml";

    std::string aPrefix = "";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()
               << EAMC(aDSMDir, "DSM directory")
               << EAMC(aImg1, "Img: Master image")
               << EAMC(aOri1,"Orientation of master image"),
        LArgMain()
                    //<< aCAS3D.ArgBasic()
               << EAM(aDir, "Dir", true, "Work directory, Def=./")
               << EAM(aPrefix, "Prefix", true, "Prefix of output file, Def=Img")
               << EAM(aDSMFile, "DSMFile", true, "The xml file that recorded the structure information of the DSM, Def=MMLastNuage.xml")

    );

   if(aPrefix.length() == 0)
       aPrefix = StdPrefix(aImg1);

   StdCorrecNameOrient(aOri1,"./",true);

   GeoreferencedDepthMap(aImg1, aDir, aDSMDir, aDSMFile, aOri1, aPrefix);

   return EXIT_SUCCESS;
}

