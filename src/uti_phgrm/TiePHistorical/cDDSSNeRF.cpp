#include "TiePHistorical.h"

void GeoreferencedDepthMap(std::string aImg1, std::string aDir, std::string aDSMDir, std::string aDSMFile, std::string aOri1, std::string aPrefix, int aScale, bool bPrint, std::string aOutDir, bool aMask)
{
    bool bMasq = false;
    std::string aImName = "";
    std::string aMasqName = "";
    std::string aCorrelName = "";

    Pt2di aDSMSz = Pt2di(0,0);

    if(1){
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

    cout<<"DSM size: "<<aDSMSz.x<<", "<<aDSMSz.y<<endl;
    cout<<"DSM file name: "<<aImName<<endl;
    cout<<"Masq file name: "<<aMasqName<<endl;
    cout<<"Correl file name: "<<aCorrelName<<endl;

    std::string aOutIm = aImName + "_tmp.tif";
    std::string aComScaleIm = MMBinFile(MM3DStr) + "ScaleIm " + aImName + " " + std::to_string(1.0/aScale) + " Out=" + aOutIm;
    cout<<aComScaleIm<<endl;
    System(aComScaleIm);
    Pt2di aDSMSclSz = Pt2di(aDSMSz.x*aScale, aDSMSz.y*aScale);
    cout<<"DSM size after scale: "<<aDSMSclSz.x<<", "<<aDSMSclSz.y<<endl;

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
    Tiff_Im aImDSMTif = Tiff_Im::StdConvGen((aOutIm).c_str(), -1, true ,true);
    TIm2D<float,double> aTImDSM(aDSMSclSz);
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

    cout<<"image size: "<<ImgSzL.x<<", "<<ImgSzL.y<<endl;

    int aType = eTIGB_Unknown;
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    cBasicGeomCap3D * aCamL = cBasicGeomCap3D::StdGetFromFile(aIm1OriFile,aType);

    //double prof_d = aCamL->GetProfondeur();
    if (ELISE_fp::exist_file(aOutDir) == false)
        ELISE_fp::MkDir(aOutDir);

    //std::string aOutIdx = aOutDir + "/"+aPrefix+"_Idx.txt";
    std::string aOut2DPts = aOutDir + "/"+aPrefix+"_2DPts.txt";
    std::string aOut3DPts = aOutDir + "/"+aPrefix+"_3DPts.txt";
    std::string aOutCorrel = aOutDir + "/"+aPrefix+"_Correl.txt";
    //FILE * fpOutIdx = fopen((aOutIdx).c_str(), "w");
    FILE * fpOut2DPts = fopen((aOut2DPts).c_str(), "w");
    FILE * fpOut3DPts = fopen((aOut3DPts).c_str(), "w");
    FILE * fpOutCorrel = fopen((aOutCorrel).c_str(), "w");

    cout<<"Files are saved here: "<<aOutDir<<endl;
    //cout<<"aOutIdx file name: "<<aOutIdx<<endl;
    cout<<"aOut2DPts file name: "<<aOut2DPts<<endl;
    cout<<"aOut3DPts file name: "<<aOut3DPts<<endl;
    cout<<"aOutCorrel file name: "<<aOutCorrel<<endl;

    int i, j;
    int nIdx = 0;
    int nOutOfBorder = 0;
    int nMasked = 0;
    int nLowCor = 0;
    int nCorTh = 5;
    double aCorrelMax = -100000;
    double aCorrelMin = 100000;
    double z_me = 0;
    double z_me_outlier = 0;
    double aZMax = -100000;
    double aZMin = 100000;
    for(j=0; j<ImgSzL.y; j++){      //first row first, the row/height is the y in MicMac
        for(i=0; i<ImgSzL.x; i++){
            Pt2di aP1_img = Pt2di(i, j);
            Pt2di aP1_dsm = Pt2di(int(i/aScale), int(j/aScale));
            int nVal = 0;
            double aCorrel = 0;
            if(bMasq == true){
                if(aP1_dsm.x < aDSMSz.x && aP1_dsm.y < aDSMSz.y){
                    nVal = aTImMasq.get(aP1_dsm);
                    aCorrel = aTImCorrel.get(aP1_dsm);
                    if(nVal == 0)
                        nMasked++;
                    else if(aCorrel < nCorTh)
                        nLowCor++;
                }
                else{
                    nOutOfBorder++;
                    //printf("Pt [%d, %d] in RGB correspond to Pt [%d, %d] in DSM, which is out of border of DSM, hence skipped\n", i, j, aP1_dsm.x, aP1_dsm.y);
                    continue;
                }
            }

            if(nVal > 0 || aMask==false)
            {
                //cout<<i<<", "<<j<<endl;
                double prof_d = aTImDSM.get(aP1_img);   //here is aP1_img because the DSM image is scaled
                Pt3dr aPTer1 = aCamL->ImEtZ2Terrain(Pt2dr(aP1_img.x, aP1_img.y), prof_d);
                //Pt3dr aPTer2 = aCamL->ImEtProf2Terrain(Pt2dr(aP1_img.x, aP1_img.y), prof_d);

                //double aCorrel = aTImCorrel.get(aP1_dsm);
                if(aCorrel < nCorTh){
                    z_me_outlier += aPTer1.z;
                    continue;
                }

                if(aPTer1.z > aZMax)
                    aZMax = aPTer1.z;
                if(aPTer1.z < aZMin)
                    aZMin = aPTer1.z;

                if(aCorrel > aCorrelMax)
                    aCorrelMax = aCorrel;
                if(aCorrel < aCorrelMin)
                    aCorrelMin = aCorrel;
                //fprintf(fpOutIdx, "%d\n", nIdx);
                //for(int p=0; p<aScale; p++){
                //    for(int q=0; q<aScale; q++){
                //fprintf(fpOut2DPts, "%d %d\n", i*aScale+p, j*aScale+q);
                fprintf(fpOut2DPts, "%d %d\n", i, j);   //here we don't divide by aScale because we want the grid with the same size as RGB images
                fprintf(fpOutCorrel, "%lf\n", aCorrel);
                fprintf(fpOut3DPts, "%lf %lf %lf\n", aPTer1.x, aPTer1.y, aPTer1.z);

                z_me += aPTer1.z;

                if(bPrint == true){
                    //if(i == 385 && j == 475)
                    printf("pt2d_img: [%d, %d], pt2d_dsm: [%d, %d], prof_d: %.2lf, Coorel: %.2lf, pt3d: [%.2lf, %.2lf, %.2lf]\n", aP1_img.x, aP1_img.y, aP1_dsm.x, aP1_dsm.y, prof_d, aCorrel, aPTer1.x, aPTer1.y, aPTer1.z);
                }
                //    }
                //}
                nIdx++;
            }
            //else
            //    fprintf(fpOutIdx, "%d\n", -1);  //-1 means the point is masked out
        }
    }
    cout<<"aCorrelMin: "<<aCorrelMin<<", aCorrelMax: "<<aCorrelMax<<" | aZMin: "<<aZMin<<", aZMax: "<<aZMax<<endl;
    printf("OutOfBorder pts number: %d, (%.2lf percent)\n", nOutOfBorder, nOutOfBorder*100.0/ImgSzL.x/ImgSzL.y);
    printf("nMasked pts number: %d, (%.2lf percent)\n", nMasked, nMasked*100.0/ImgSzL.x/ImgSzL.y);
    printf("nLowCor pts number in no masked area: %d, (%.2lf percent); z_mean: %.2lf\n", nLowCor, nLowCor*100.0/ImgSzL.x/ImgSzL.y, z_me_outlier/(nLowCor+1e-5));
    printf("Valid pts number: %d, (%.2lf percent); z_mean: %.2lf\n", nIdx, nIdx*100.0/ImgSzL.x/ImgSzL.y, z_me/(nIdx+1e-5));
    //cout<<z_me_outlier<<" "<<z_me<<endl;
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

    int aScale = 1;

    bool bPrint = false;

    std::string aOutDir = "";

    bool aMask = true;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()
               << EAMC(aDSMDir, "DSMDir: DSM directory")
               << EAMC(aImg1, "Img: Master image")
               << EAMC(aOri1,"Orientation of master image"),
        LArgMain()
                    //<< aCAS3D.ArgBasic()
               << EAM(aDir, "Dir", true, "Work directory, Def=./")
               << EAM(aPrefix, "Prefix", true, "Prefix of output file, Def=Img")
               << EAM(aScale, "Scale", true, "Scale up the resolution of the dense depth grid, Def=1")
               << EAM(aPrefix, "Prefix", true, "Prefix of output file, Def=Img")
               << EAM(bPrint, "Print", true, "Print debug information, Def=false")
               << EAM(aOutDir, "OutDir", true, "Output directory, Def=DSMDir")
               << EAM(aMask, "Mask", true, "Mask out the invalide regions indicated by the automatique mask file, Def=true")
    );

   if(aPrefix.length() == 0)
       aPrefix = StdPrefix(aImg1);

   if(aOutDir.length() == 0)
       aOutDir = aDSMDir;

   StdCorrecNameOrient(aOri1,"./",true);

   GeoreferencedDepthMap(aImg1, aDir, aDSMDir, aDSMFile, aOri1, aPrefix, aScale, bPrint, aOutDir, aMask);

   return EXIT_SUCCESS;
}

