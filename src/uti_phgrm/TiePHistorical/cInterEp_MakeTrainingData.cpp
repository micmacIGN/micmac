/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "TiePHistorical.h"
#include <ctime>
#include "cnpy.h"
//#include "cnpy.cpp"


void GetCorners(Pt2dr aOriPt, Pt2dr aSz, Pt2dr* aCorners)
{
    aCorners[0] = aOriPt;
    aCorners[1] = Pt2dr(aOriPt.x+aSz.x, aOriPt.y);
    aCorners[2] = Pt2dr(aOriPt.x+aSz.x, aOriPt.y+aSz.y);
    aCorners[3] = Pt2dr(aOriPt.x, aOriPt.y+aSz.y);
    /*
    for(int i=0; i<4; i++){
        aCorners[i].x *= dScale;
        aCorners[i].y *= dScale;
    }
    */
}

void SaveResampleTxt(std::string aNameSave, Pt2dr* aPtL, Pt2dr* aPtR, double dScaleL = 1, double dScaleR = 1)
{
    FILE * fpOutput = fopen((aNameSave).c_str(), "w");
    for(int i=0; i<4; i++)
    {
        fprintf(fpOutput, "%lf %lf %lf %lf\n", aPtL[i].x*dScaleL, aPtL[i].y*dScaleL, aPtR[i].x*dScaleR, aPtR[i].y*dScaleR);
    }
    fclose(fpOutput);
}

bool CheckInDSMBorder(Pt2dr* ImgCorner, Pt2dr PatchSz, cGet3Dcoor a3DL, cDSMInfo aDSMInfoL, bool bPrint)
//, std::vector<Pt2dr> & DSMCoor
{
    Pt2dr PatchCorner[4];
    GetCorners(Pt2dr(0, 0), PatchSz, PatchCorner);
    /*
    Pt2dr origin = Pt2dr(0, 0);
    PatchCorner[0] = origin;
    PatchCorner[1] = Pt2dr(origin.x+PatchSz.x, origin.y);
    PatchCorner[2] = Pt2dr(origin.x+PatchSz.x, origin.y+PatchSz.y);
    PatchCorner[3] = Pt2dr(origin.x, origin.y+PatchSz.y);
    */

    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = ImgCorner[i];
        Pt3dr aPTer1;
        bool bPreciseL;
        aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD(), true);

        if(bPreciseL == false)
            return false;
    }
    return true;
}

std::string PrepareDSM(std::string aDSMDir, std::string aDSMFile)
{
    Pt2di aDSMSz = cDSMInfo::GetDSMSz(aDSMFile, aDSMDir);
    cDSMInfo aDSMInfoL(aDSMSz, aDSMFile, aDSMDir);

    std::string aDSMName = aDSMInfoL.GetDSMName(aDSMFile, aDSMDir);

    return aDSMName;

    std::string aDSMImgNameSrc = StdPrefix(aDSMName) + "_gray.tif_sfs." + StdPostfix(aDSMName);
    std::string aDSMImgNameDes = aDSMDir + "_" + aDSMImgNameSrc;

    if (ELISE_fp::exist_file(aDSMDir + "/" + aDSMImgNameSrc) == false){
        cout<<"aDSMImgNameSrc: "<<aDSMImgNameSrc<<endl;
        std::string cmmd = MMBinFile(MM3DStr) + " TestLib DSM_Equalization "+aDSMDir+" DSMFile="+aDSMFile;
        cout<<cmmd<<endl;
        System(cmmd);
        cmmd = MMBinFile(MM3DStr) + " TestLib Wallis "+StdPrefix(aDSMName)+"_gray.tif Dir="+aDSMDir;
        cout<<cmmd<<endl;
        System(cmmd);
    }

    std::string cmmd = "cp " + aDSMDir + "/" + aDSMImgNameSrc + " " + aDSMImgNameDes;
    cout<<cmmd<<endl;
    System(cmmd);

    return aDSMImgNameDes;
}

void RemoveDSM(std::string aDSMImgName)
{
    if (ELISE_fp::exist_file(aDSMImgName) == true)
    {
        std::string cmmd = "rm " + aDSMImgName;
        cout<<cmmd<<endl;
        System(cmmd);
    }
}

//For no matches in GT tie point file, save point (-100,-100) for the convenience of look-up
void GetGTTiePts(Pt2dr aPtOriL, Pt2dr PatchSzL, Pt2dr PatchSzR, Pt2di ImgSzR, cTransform3DHelmert aTrans3DHL, cBasicGeomCap3D * aCamR, cGet3Dcoor a3DR, cDSMInfo aDSMInfoR, cBasicGeomCap3D * aCamL, cGet3Dcoor a3DL, cDSMInfo aDSMInfoL, bool bPrint, cElHomographie aHomo, std::string aOutDirImg, std::string aOutDirNpz, std::string aImg1, std::string aImg2, std::string SH, int nGap, double dThres, bool bSaveHomol)
{
    ElPackHomologue aPack;
    Pt2dr aPtLinImage, aPtLinPatch;
    Pt2dr aPtRinImage, aPtRinPatch;
    int i, j;
    //int nGap = 20;
    int nWid = int(PatchSzL.x/nGap);
    int nHei = int(PatchSzL.y/nGap);

    Pt3dr aPTer1, aPTer2;

    //double dThres = 3;
    cout<<"Homo for "<<aImg1<<endl;
    printf("1, 0, %.2lf\n0, 1, %.2lf\n0, 0, 1\n", aPtOriL.x, aPtOriL.y);
    cout<<"Homo for "<<aImg2<<endl;
    aHomo.Show();

    //create random data
    std::vector<std::complex<double>> data;

    for(i=0; i<nWid; i++){
        aPtLinPatch.x = i*nGap;
        aPtLinImage.x = aPtOriL.x + aPtLinPatch.x;
        for(j=0; j<nHei; j++){
            aPtLinPatch.y = j*nGap;
            aPtLinImage.y = aPtOriL.y + aPtLinPatch.y;

            //compute tie point in right image
            bool bPrecise;
            aPTer1 = a3DL.Get3Dcoor(aPtLinImage, aDSMInfoL, bPrecise, bPrint);//, a3DR.GetGSD(), true);
            aPTer1 = aTrans3DHL.Transform3Dcoor(aPTer1);
            aPtRinImage = aCamR->Ter2Capteur(aPTer1);

            //check out of border
            if(aPtRinImage.x < 0 || aPtRinImage.y < 0 || aPtRinImage.x >= ImgSzR.x || aPtRinImage.y >= ImgSzR.y)
            {
                data.push_back(std::complex<double>(-100,-100));
                /*if(bSaveHomol)
                    aPack.Cple_Add(ElCplePtsHomologues(aPtLinPatch, Pt2dr(-100,-100)));*/
                continue;
            }

            //check depth
            aPTer2 = a3DR.Get3Dcoor(aPtRinImage, aDSMInfoR, bPrecise, bPrint);
            double dDis = pow((pow((aPTer1.x-aPTer2.x), 2), pow((aPTer1.y-aPTer2.y), 2), pow((aPTer1.z-aPTer2.z), 2)), 0.5);
            /*if(true){
                aPtRinPatch = aHomo(aPtRinImage);
                printf("aPtLinPatch: [%.2lf, %.2lf], aPtRinPatch: [%.2lf, %.2lf], aPTer1: [%.2lf, %.2lf, %.2lf], aPTer2: [%.2lf, %.2lf, %.2lf]\nDiff: [%.2lf, %.2lf, %.2lf], dDis: %.2lf\n", aPtLinPatch.x, aPtLinPatch.y, aPtRinPatch.x, aPtRinPatch.y, aPTer1.x, aPTer1.y, aPTer1.z, aPTer2.x, aPTer2.y, aPTer2.z, aPTer1.x-aPTer2.x, aPTer1.y-aPTer2.y, aPTer1.z-aPTer2.z, dDis);
            }*/
            if(dDis > dThres)
            {
                data.push_back(std::complex<double>(-100,-100));
                continue;
            }

            //transform pt in right image to pt in right patch
            aPtRinPatch = aHomo(aPtRinImage);
            //check out of border
            if(aPtRinPatch.x < 0 || aPtRinPatch.y < 0 || aPtRinPatch.x >= PatchSzR.x || aPtRinPatch.y >= PatchSzR.y)
            {
                data.push_back(std::complex<double>(-100,-100));
                continue;
            }

            data.push_back(std::complex<double>(aPtRinPatch.x, aPtRinPatch.y));

            aPack.Cple_Add(ElCplePtsHomologues(aPtLinPatch, aPtRinPatch));
        }
    }

    //save to npz file
    if(false)
    {
        long unsigned int dataSz = data.size();
        cnpy::npz_save(aOutDirNpz+StdPrefix(aImg2)+".npz","homol",&data[0],{dataSz},"w");
    }
    else
    {
        FILE * fpTxt = fopen((aOutDirNpz+StdPrefix(aImg2)+".txt").c_str(), "w");
        for (int i=0; i<int(data.size()); i++)
        {
            fprintf(fpTxt, "%lf %lf\n", data[i].real(), data[i].imag());
        }
        fclose(fpTxt);
    }

    //save to MicMac format
    if(bSaveHomol)
    {      
        std::string aCom = "mm3d SEL" + BLANK + aOutDirImg + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+SH;
        printf("*************************\n%s\n", aCom.c_str());

        std::string aSHDir = aOutDirImg + "/Homol" + SH+"/";
        if (ELISE_fp::exist_file(aSHDir) == false)
            ELISE_fp::MkDir(aSHDir);
        std::string aNewDir = aSHDir + "Pastis" + aImg1;
        if (ELISE_fp::exist_file(aNewDir) == false)
            ELISE_fp::MkDir(aNewDir);
        std::string aNameFile1 = aNewDir + "/"+aImg2+".txt";

        FILE * fpTiePt1 = fopen(aNameFile1.c_str(), "w");
        for (ElPackHomologue::iterator itCpl=aPack.begin();itCpl!=aPack.end() ; itCpl++)
        {
            ElCplePtsHomologues tiept = itCpl->ToCple();
            fprintf(fpTiePt1, "%lf %lf %lf %lf\n", tiept.P1().x, tiept.P1().y, tiept.P2().x, tiept.P2().y);
        }
        fclose(fpTiePt1);
    }
}

void ReProjection(Pt2dr* aPCornerL, Pt2dr* aPCornerR, cTransform3DHelmert aTrans3DH, cBasicGeomCap3D * aCamR, cGet3Dcoor a3DL, cDSMInfo aDSMInfoL, bool bPrint, double dRandomShift)
{
    double dScaleL = 1;
    double dScaleR = 1;

    double dShiftX = 0;
    double dShiftY = 0;

    if(dRandomShift < 0)
        dRandomShift = -dRandomShift;

    if(dRandomShift > 0.000001){
        srand((int)time(0));
        std::vector<double> res;
        GetRandomNum(-dRandomShift, dRandomShift, 2, res);
        dShiftX = res[0];
        dShiftY = res[1];
        printf("dShiftX, dShiftY: %.2lf %.2lf\n", dShiftX, dShiftY);
    }

    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = aPCornerL[i];
        Pt3dr aPTer1;
        bool bPreciseL;
        aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD(), true);
        aPTer1 = aTrans3DH.Transform3Dcoor(aPTer1);

        aPCornerR[i] = aCamR->Ter2Capteur(aPTer1);
        aPCornerR[i].x = aPCornerR[i].x/dScaleL/dScaleR + dShiftX;
        aPCornerR[i].y = aPCornerR[i].y/dScaleL/dScaleR + dShiftY;
    }
}

cElHomographie GetSecondaryPatch(Pt2dr* aPCornerR, Pt2dr PatchSz, std::string aImg2, std::string aSubImgR, bool bPrint, std::string aOutDirImg, std::string aOutDirRGBTxt, bool bKeepRGBPatch, double dScaleR)
{
    Pt2dr aPCornerPatch[4];
    GetCorners(Pt2dr(0, 0), PatchSz, aPCornerPatch);
    /*
    Pt2dr origin = Pt2dr(0, 0);
    aPCornerPatch[0] = origin;
    aPCornerPatch[1] = Pt2dr(origin.x+PatchSz.x, origin.y);
    aPCornerPatch[2] = Pt2dr(origin.x+PatchSz.x, origin.y+PatchSz.y);
    aPCornerPatch[3] = Pt2dr(origin.x, origin.y+PatchSz.y);
    */

    std::string aNameSave = StdPrefix(aSubImgR) + ".txt";
    SaveResampleTxt(aNameSave, aPCornerPatch, aPCornerR, 1, dScaleR);

    if(bKeepRGBPatch == true)
    {
        std::string aComBaseResample = MMBinFile(MM3DStr) + "TestLib OneReechFromAscii ";
        std::string aComResampleSndImg = aComBaseResample + aImg2  + " " + aNameSave + " Out="+aSubImgR + " Show=true";
        cout<<aComResampleSndImg<<endl;
        System(aComResampleSndImg);

        std::string aMvTif = "mv "+aSubImgR + " "+aOutDirImg+aSubImgR;
        cout<<aMvTif<<endl;
        System(aMvTif);
    }

    ElPackHomologue aPack;
    //aPack = ElPackHomologue::FromFile(aNameSave);
    for(int i=0; i<4; i++)
        aPack.Cple_Add(ElCplePtsHomologues(aPCornerPatch[i], aPCornerR[i]));
    double anEcart,aQuality;
    bool Ok;
    cElHomographie aHomo = cElHomographie::RobustInit(anEcart,&aQuality,aPack,Ok,50,80.0,2000);
    std::cout << "Ecart " << anEcart << " ; Quality " << aQuality    << " \n";

    std::string aMvTxt = "mv "+aNameSave + " "+aOutDirRGBTxt+aNameSave;
    cout<<aMvTxt<<endl;
    System(aMvTxt);

    return aHomo.Inverse();
}

void CropPatch(Pt2dr aPtOri, Pt2dr aPatchSz, std::string aImg2, std::string aOutDir, std::string aSubImgR, std::string aOutDirRGBTxt, bool bKeepRGBPatch, double dScaleL)
{
    Pt2dr aCornersL[4];
    GetCorners(Pt2dr(0, 0), aPatchSz, aCornersL);
    Pt2dr aCornersR[4];
    GetCorners(aPtOri, aPatchSz, aCornersR);
    std::string aNameSave = StdPrefix(aSubImgR) + ".txt";
    SaveResampleTxt(aNameSave, aCornersL, aCornersR, 1, dScaleL);

    std::string aMvTxt = "mv "+aNameSave + " "+aOutDirRGBTxt+aNameSave;
    cout<<aMvTxt<<endl;
    System(aMvTxt);

    if(bKeepRGBPatch == true)
    {
        std::string aClipSz = " ["+std::to_string(int(aPatchSz.x*dScaleL))+","+std::to_string(int(aPatchSz.y*dScaleL))+"] ";
        std::string aComBaseClip = MMBinFile(MM3DStr) + "ClipIm " + aImg2 + " ";
        std::string aComClipMasterImg = aComBaseClip + " ["+std::to_string(int(aPtOri.x*dScaleL))+","+std::to_string(int(aPtOri.y*dScaleL))+"] " + aClipSz + " Out="+aOutDir+"/"+aSubImgR;
        cout<<aComClipMasterImg<<endl;
        System(aComClipMasterImg);
    }
}

bool GetDSMPatch(std::string aImg, std::string aDSMDir, std::string aDSMFile, Pt2dr* ImgCorner, Pt2dr PatchSz, cGet3Dcoor a3DL, cDSMInfo aDSMInfoL, bool bPrint, std::string aOutDirDSM, std::string aOutDirDSMTxt, bool bKeepDSMPatch)
//, std::vector<Pt2dr> & DSMCoor
{
    Pt2dr PatchCorner[4];
    GetCorners(Pt2dr(0, 0), PatchSz, PatchCorner);

    std::string aDSMName = aDSMInfoL.GetDSMName(aDSMFile, aDSMDir);
    std::string aDSMImgName = aDSMDir + "_" + StdPrefix(aDSMName) + "_gray.tif_sfs." + StdPostfix(aDSMName);

    std::string aNameSave = StdPrefix(aImg); // + "_DSM";
    std::string aSubImg = aNameSave + "." + StdPostfix(aImg);
    aNameSave += ".txt";

    Pt2dr aPtInDSM[4];
    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = ImgCorner[i];
        Pt3dr aPTer1;
        bool bPreciseL;
        aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD(), true);

        aPtInDSM[i] = aDSMInfoL.Get2DcoorInDSM(aPTer1);
    }

    SaveResampleTxt(aNameSave, PatchCorner, aPtInDSM);
    /*
    FILE * fpOutput = fopen((aNameSave).c_str(), "w");
    for(int i=0; i<4; i++)
        fprintf(fpOutput, "%lf %lf %lf %lf\n", PatchCorner[i].x, PatchCorner[i].y, aPtInDSM[i].x, aPtInDSM[i].y);
    fclose(fpOutput);
    */

    std::string cmmd = "mv " + aNameSave + " " + aOutDirDSMTxt + aNameSave;
    cout<<cmmd<<endl;
    System(cmmd);

    if(bKeepDSMPatch == true)
    {
        /*
        std::string aComBaseResample = MMBinFile(MM3DStr) + "TestLib OneReechFromAscii ";
        std::string aComResampleSndImg = aComBaseResample + aDSMImgName + " " + aNameSave + " Out="+aSubImg + " Show=true";
        cout<<aComResampleSndImg<<endl;
        System(aComResampleSndImg);

        std::string cmmd = "mv " + aSubImg + " " + aOutDirDSM + aSubImg;
        cout<<cmmd<<endl;
        System(cmmd);
        */

        std::string cmmd = "python3 /home/lulin/Documents/Code/DSM2ImgEqualPatch.py --DSMFile " + aDSMDir + "/" + aDSMName + " --TxtFile " + aOutDirDSMTxt + aNameSave + "  --OutFile " + aOutDirDSM + aSubImg;
        cout<<cmmd<<endl;
        System(cmmd);
    }

    return true;
}

void MakeTrainingData(std::string aDir,std::string aImg1, std::string aImg2, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR, std::string aOri1, std::string aOri2, std::string aDSMFileGTL, std::string aDSMFileGTR, std::string aDSMDirGTL, std::string aDSMDirGTR, std::string aOriGT1, std::string aOriGT2, cInterfChantierNameManipulateur * aICNM, cTransform3DHelmert aTrans3DHL, cTransform3DHelmert aTrans3DHR, bool bPrint, Pt2dr aPatchSz, Pt2di seed, int nGap, double dThres, std::string SH, bool bTile, bool bSaveHomol, double dRandomShift, std::string aScene, bool bKeepRGBPatch, bool bKeepDSMPatch, double dScaleL, double dScaleR)
{
    if (ELISE_fp::exist_file(aImg1) == false || ELISE_fp::exist_file(aImg2) == false)
    {
        cout<<aImg1<<" or "<<aImg2<<" didn't exist, hence skipped"<<endl;
        return;
    }

    //Tiff_Im aRGBIm1(aImg1.c_str());
    Tiff_Im aRGBIm1 = Tiff_Im::StdConvGen((aImg1).c_str(), -1, true ,true);
    Pt2di ImgSzL = aRGBIm1.sz();
    ImgSzL.x /= dScaleL;
    ImgSzL.y /= dScaleL;
    //Tiff_Im aRGBIm2(aImg2.c_str());
    Tiff_Im aRGBIm2 = Tiff_Im::StdConvGen((aImg2).c_str(), -1, true ,true);
    Pt2di ImgSzR = aRGBIm2.sz();
    ImgSzR.x /= dScaleR;
    ImgSzR.y /= dScaleR;
    printf("%s: %d, %d\n%s: %d, %d\n", aImg1.c_str(), ImgSzL.x, ImgSzL.y, aImg2.c_str(), ImgSzR.x, ImgSzR.y);

    int aType = eTIGB_Unknown;
    std::string aNameOriL = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    std::string aNameOriR = aICNM->StdNameCamGenOfNames(aOri2, aImg2);
    //cBasicGeomCap3D * aCamL = cBasicGeomCap3D::StdGetFromFile(aNameOriL,aType);
    cBasicGeomCap3D * aCamR = cBasicGeomCap3D::StdGetFromFile(aNameOriR,aType);
    cGet3Dcoor a3DL(aNameOriL);
    cDSMInfo aDSMInfoL = a3DL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cGet3Dcoor a3DR(aNameOriR);
    cDSMInfo aDSMInfoR = a3DR.SetDSMInfo(aDSMFileR, aDSMDirR);

    std::string aNameOriGTL = aICNM->StdNameCamGenOfNames(aOriGT1, aImg1);
    std::string aNameOriGTR = aICNM->StdNameCamGenOfNames(aOriGT2, aImg2);
    cBasicGeomCap3D * aCamGTL = cBasicGeomCap3D::StdGetFromFile(aNameOriGTL,aType);
    cBasicGeomCap3D * aCamGTR = cBasicGeomCap3D::StdGetFromFile(aNameOriGTR,aType);
    cGet3Dcoor a3DGTL(aNameOriGTL);
    cDSMInfo aDSMInfoGTL = a3DGTL.SetDSMInfo(aDSMFileGTL, aDSMDirGTL);
    cGet3Dcoor a3DGTR(aNameOriGTR);
    cDSMInfo aDSMInfoGTR = a3DGTR.SetDSMInfo(aDSMFileGTR, aDSMDirGTR);

    Pt2dr aPCornerR[4];
    GetCorners(Pt2dr(0, 0), Pt2dr(ImgSzR.x, ImgSzR.y), aPCornerR);
    /*
    Pt2dr origin = Pt2dr(0, 0);
    aPCornerR[0] = origin;
    aPCornerR[1] = Pt2dr(origin.x+ImgSzR.x, origin.y);
    aPCornerR[2] = Pt2dr(origin.x+ImgSzR.x, origin.y+ImgSzR.y);
    aPCornerR[3] = Pt2dr(origin.x, origin.y+ImgSzR.y);
    */

    Pt2dr aPCornerRinL[4];  //to Save zone in master image which is overlapping with secondary image
    for(int i=0; i<4; i++)
    {
        Pt2dr aPtR = aPCornerR[i];
        Pt3dr aPTer1;
        bool bPrecise;
        aPTer1 = a3DR.Get3Dcoor(aPtR, aDSMInfoR, bPrecise, bPrint);//, a3DL.GetGSD());

        aPTer1 = aTrans3DHR.Transform3Dcoor(aPTer1);
        Pt2dr ptPred = a3DL.Get2Dcoor(aPTer1);
        aPCornerRinL[i] = ptPred;

        CheckRange(0, ImgSzL.x, aPCornerRinL[i].x);
        CheckRange(0, ImgSzL.y, aPCornerRinL[i].y);

        if(bPrint)
        {
            printf("%dth: CornerR: [%.2lf\t%.2lf], CornerRinTerr: [%.2lf\t%.2lf\t%.2lf], ptPred: [%.2lf\t%.2lf], CornerRinL: [%.2lf\t%.2lf]\n", i, aPCornerR[i].x, aPCornerR[i].y, aPTer1.x, aPTer1.y, aPTer1.z, ptPred.x, ptPred.y, aPCornerRinL[i].x, aPCornerRinL[i].y);
        }
    }

    double dMaxX = max(max(aPCornerRinL[0].x, aPCornerRinL[1].x), max(aPCornerRinL[2].x, aPCornerRinL[3].x));
    double dMinX = min(min(aPCornerRinL[0].x, aPCornerRinL[1].x), min(aPCornerRinL[2].x, aPCornerRinL[3].x));
    double dMaxY = max(max(aPCornerRinL[0].y, aPCornerRinL[1].y), max(aPCornerRinL[2].y, aPCornerRinL[3].y));
    double dMinY = min(min(aPCornerRinL[0].y, aPCornerRinL[1].y), min(aPCornerRinL[2].y, aPCornerRinL[3].y));

    if(dMaxX < dMinX || dMaxY < dMinY)
    {
        printf("There is no overlapping area in the image pair, hence skipped.\n");
        return;
    }
    else
        printf("Zone in master image which is overlapping with secondary image:\nMinX: %.2lf, MaxX: %.2lf MinY: %.2lf, MaxY: %.2lf\n", dMinX, dMaxX, dMinY, dMaxY);

    std::string aSceneDir = aDir + "/Tmp_TrainingPatches/";
    if (ELISE_fp::exist_file(aSceneDir) == false)
        ELISE_fp::MkDir(aSceneDir);
    aSceneDir = aSceneDir + aScene + "/";
    if (ELISE_fp::exist_file(aSceneDir) == false)
        ELISE_fp::MkDir(aSceneDir);
    std::string aOutDirDSM = aSceneDir + "DSM/";
    if (ELISE_fp::exist_file(aOutDirDSM) == false && bKeepDSMPatch==true)
        ELISE_fp::MkDir(aOutDirDSM);
    std::string aOutDirImg = aSceneDir + "RGB/";
    if (ELISE_fp::exist_file(aOutDirImg) == false && bKeepRGBPatch==true)
        ELISE_fp::MkDir(aOutDirImg);
    std::string aOutDirNpz = aSceneDir + "HomolNpz/";
    if (ELISE_fp::exist_file(aOutDirNpz) == false)
        ELISE_fp::MkDir(aOutDirNpz);
    std::string aOutDirDSMTxt = aSceneDir + "DSMTxt/";
    if (ELISE_fp::exist_file(aOutDirDSMTxt) == false)
        ELISE_fp::MkDir(aOutDirDSMTxt);
    std::string aOutDirRGBTxt = aSceneDir + "RGBTxt/";
    if (ELISE_fp::exist_file(aOutDirRGBTxt) == false)
        ELISE_fp::MkDir(aOutDirRGBTxt);

    std::vector<int> centerX;
    std::vector<int> centerY;

    if(bTile == true)
    {
        int nNumX = int((dMaxX - dMinX)/aPatchSz.x);
        int nNumY = int((dMaxY - dMinY)/aPatchSz.y);
        //to avoid blank in the resampled image
        if(false){
            nNumX--;
            nNumY--;
        }
        int nStartX = int(dMinX + ((dMaxX - dMinX) - nNumX*aPatchSz.x)/2);
        int nStartY = int(dMinY + ((dMaxY - dMinY) - nNumY*aPatchSz.y)/2);
        printf("zone in master image which is overlapping with secondary image: dMaxX, dMinX, dMaxY, dMinY: %lf, %lf, %lf, %lf\n", dMaxX, dMinX, dMaxY, dMinY);
        printf("%d, %d, %d, %d\n%lf  %lf\n", nNumX, nNumY, nStartX, nStartY, aPatchSz.x, aPatchSz.y);
        for(int i=0; i<nNumX; i++)
        {
            centerX.push_back(int(nStartX+i*aPatchSz.x));
        }
        for(int j=0; j<nNumY; j++)
        {
            centerY.push_back(int(nStartY+j*aPatchSz.y));
        }
    }
    else
    {
        srand((int)time(0));
        //randomly get the coordinate in secondary image which will be used as center for cropping patch
        GetRandomNum(dMinX, dMaxX, 1, centerX);
        GetRandomNum(dMinY, dMaxY, 1, centerY);

        //if user set CenterMatch
        if(seed.x > 0){
            if(seed.x < ImgSzL.x)
                centerX[0] = seed.x;
            else
                printf("x of CenterMatch %d should be smaller than image width %d\n", seed.x, ImgSzR.x);
        }
        if(seed.y > 0){
            if(seed.y < ImgSzL.y)
                centerY[0] = seed.y;
            else
                printf("y of CenterMatch %d should be smaller than image width %d\n", seed.y, ImgSzR.y);
        }
    }

    printf("************************************************\n%s, %s\n", aImg1.c_str(), aImg2.c_str());
    //std::vector<std::string> aMvTxtCmmd;
    std::vector<std::string> vPatchesL, vPatchesR;
    std::vector<cElHomographie> vHomoL, vHomoR;
    for(int m=0; m<int(centerX.size()); m++)
    {
        for(int n=0; n<int(centerY.size()); n++)
        {
            printf("----->>>>>>CenterMatch[%d,%d]=[%d,%d]\tImgSz: %d\t%d\n", m, n, centerX[m], centerY[n], ImgSzL.x, ImgSzL.y);

            Pt2dr aPCenterL = Pt2dr(centerX[m], centerY[n]);
            if(CheckRange(aPatchSz.x/2, dMaxX-aPatchSz.x/2, aPCenterL.x)==false ||
            CheckRange(aPatchSz.y/2, dMaxY-aPatchSz.y/2, aPCenterL.y)==false)
            {
                printf("There is no enough overlapping area in the patch pair, hence skipped.\n");
                continue;
            }

            Pt2dr aPtOriL = Pt2dr(aPCenterL.x-aPatchSz.x/2, aPCenterL.y-aPatchSz.y/2);
            //printf("aPtOriL: %.2lf, %.2lf, %.2lf, %.2lf, %.2lf\n", aPtOriL.x, aPtOriL.y, aPCenterL.x, aPCenterL.y, aPatchSz.x);

            Pt2dr aPatchCornerL[4];
            GetCorners(aPtOriL, aPatchSz, aPatchCornerL);
            /*
            origin = aPtOriL;
            aPatchCornerL[0] = origin;
            aPatchCornerL[1] = Pt2dr(origin.x+aPatchSz.x, origin.y);
            aPatchCornerL[2] = Pt2dr(origin.x+aPatchSz.x, origin.y+aPatchSz.y);
            aPatchCornerL[3] = Pt2dr(origin.x, origin.y+aPatchSz.y);
            */

            Pt2dr aPatchCornerR[4];
            ReProjection(aPatchCornerL, aPatchCornerR, aTrans3DHL, aCamR, a3DL, aDSMInfoL, bPrint, dRandomShift);
            bool bOutOfBorder = false;
            for(int k=0; k<4; k++){
                if(aPatchCornerR[k].x<0 || aPatchCornerR[k].x>= ImgSzR.x || aPatchCornerR[k].y<0 || aPatchCornerR[k].y>= ImgSzR.y)
                    bOutOfBorder = true;
                if(bPrint)
                    printf("%dth: PatchCornerL: [%.2lf\t%.2lf]  PatchCornerR: [%.2lf\t%.2lf]\n", k, aPatchCornerL[k].x, aPatchCornerL[k].y, aPatchCornerR[k].x, aPatchCornerR[k].y);
            }
            //if(FallInBox(aPatchCornerR, Pt2dr(0,0), ImgSzR) == false)
            if(bOutOfBorder == true)
            {
                if(bPrint)
                {
                    printf("Reprojected RGB patch out of border, hence skipped.\n");
                    printf("patch: ");
                    for(int i=0; i<4; i++)
                    {
                        printf("[%.2lf,%.2lf] ", aPatchCornerR[i].x, aPatchCornerR[i].y);
                    }
                    printf("out of border:\n[%d,%d] [%d,%d]\n", 0, 0, ImgSzR.x, ImgSzR.y);
                }
                continue;
            }


            if(CheckInDSMBorder(aPatchCornerR, aPatchSz, a3DR, aDSMInfoR, bPrint) == false || CheckInDSMBorder(aPatchCornerL, aPatchSz, a3DL, aDSMInfoL, bPrint) == false){
                printf("DSM patch out of border or mask, hence skipped.\n");
                //continue;
            }

            std::string aSubImgL = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_" + std::to_string(m) + "_" + std::to_string(n) + "_L." + StdPostfix(aImg1);
            std::string aSubImgR = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_" + std::to_string(m) + "_" + std::to_string(n) + "_R." + StdPostfix(aImg1);

            //crop master RGB patch
            CropPatch(aPtOriL, aPatchSz, aImg1, aOutDirImg, aSubImgL, aOutDirRGBTxt, bKeepRGBPatch, dScaleL);

            //Crop roughly aligned patches in secondary image
            cElHomographie aHomo = GetSecondaryPatch(aPatchCornerR, aPatchSz, aImg2, aSubImgR, bPrint, aOutDirImg, aOutDirRGBTxt, bKeepRGBPatch, dScaleR);


            cElComposHomographie aFstHX(1, 0, aPatchCornerL[0].x);
            cElComposHomographie aFstHY(0, 1, aPatchCornerL[0].y);
            cElComposHomographie aFstHZ(0, 0,        1);
            cElHomographie  aFstH =  cElHomographie(aFstHX,aFstHY,aFstHZ);
/*
            cElComposHomographie aUnitHX(1, 0, 0);
            cElComposHomographie aUnitHY(0, 1, 0);
            cElComposHomographie aUnitHZ(0, 0, 1);
            cElHomographie  aSndH =  cElHomographie(aUnitHX,aUnitHY,aUnitHZ);

            Pt2dr aPCornerPatch[4];
            GetCorners(Pt2dr(0,0), aPatchSz, aPCornerPatch);
            ElPackHomologue aPack;
                for(int k=0; k<4; k++)
                {
                    aPack.Cple_Add(ElCplePtsHomologues(aPCornerPatch[k], aPCornerR[k]));
                }

                double anEcart,aQuality;
                bool Ok;
                aSndH = cElHomographie::RobustInit(anEcart,&aQuality,aPack,Ok,50,80.0,2000);
                */
                vPatchesL.push_back(aSubImgL);
                vHomoL.push_back(aFstH);
                vPatchesR.push_back(aSubImgR);
                vHomoR.push_back(aHomo.Inverse());

            //Crop master DSM patch
            GetDSMPatch(aSubImgL, aDSMDirL, aDSMFileL, aPatchCornerL, aPatchSz, a3DL, aDSMInfoL, bPrint, aOutDirDSM, aOutDirDSMTxt, bKeepDSMPatch);

            //Crop secondary DSM patch
            GetDSMPatch(aSubImgR, aDSMDirR, aDSMFileR, aPatchCornerR, aPatchSz, a3DR, aDSMInfoR, bPrint, aOutDirDSM, aOutDirDSMTxt, bKeepDSMPatch);

            //get tie points in patch pair
            //GetGTTiePts(aPtOriL, aPatchSz, aPatchSz, ImgSzR, aTrans3DHL, aCamR, a3DR, aDSMInfoR, aCamL, a3DL, aDSMInfoL, bPrint, aHomo, aOutDirImg, aSubImgL, aSubImgR, SH, nGap, dThres, bSaveHomol);
            GetGTTiePts(aPtOriL, aPatchSz, aPatchSz, ImgSzR, aTrans3DHL, aCamGTR, a3DGTR, aDSMInfoGTR, aCamGTL, a3DGTL, aDSMInfoGTL, bPrint, aHomo, aOutDirImg, aOutDirNpz, aSubImgL, aSubImgR, SH, nGap, dThres, bSaveHomol);


/*
            //Same center, keep GSD and rotation difference
            if(false)
            {
                //Crop master patch
                Pt3dr aPTer1;
                bool bPreciseR;
                aPTer1 = a3DR.Get3Dcoor(aPCenterL, aDSMInfoR, bPreciseR, bPrint);//, a3DR.GetGSD());

                aPTer1 = aTrans3DHR.Transform3Dcoor(aPTer1);
                Pt2dr aPCenterL = a3DL.Get2Dcoor(aPTer1);
                printf("CenterMatch in master image: %.0lf\t%.0lf\n", aPCenterL.x, aPCenterL.y);
                printf("CenterMatch in secondary image: %.0lf\t%.0lf\n", aPCenterL.x, aPCenterL.y);

                Pt2dr aPtOriL = Pt2dr(aPCenterL.x-aPatchSz.x/2, aPCenterL.y-aPatchSz.y/2);
                CropPatch(aPtOriL, aPatchSz, aImg1, aOutDirImg, aSubImgL);

                Pt2dr aPatchCornerR[4];
                origin = aPtOriL;
                aPatchCornerR[0] = origin;
                aPatchCornerR[1] = Pt2dr(origin.x+aPatchSz.x, origin.y);
                aPatchCornerR[2] = Pt2dr(origin.x+aPatchSz.x, origin.y+aPatchSz.y);
                aPatchCornerR[3] = Pt2dr(origin.x, origin.y+aPatchSz.y);
                GetDSMPatch(aSubImgR, aDSMDirL, aDSMFileL, aPatchCornerR, aPatchSz, a3DL, aDSMInfoL, bPrint, aOutDirDSM, aOutDirDSMTxt);
                //RotateImgBy90Deg1("Tmp_TrainingPatches", aDSML, aDSML+"_90.tif");
                //RotateImgBy90Deg1("./", aImg1, aImg1+"_90.tif");
            }
*/
        }
    }
    WriteXml(aImg1, aImg2, aSceneDir+"SubPatch.xml", vPatchesL, vPatchesR, vHomoL, vHomoR, bPrint);
/*
    for(int i=0; i<int(aMvTxtCmmd.size()); i++){
        cout<<aMvTxtCmmd[i]<<endl;
        System(aMvTxtCmmd[i]);
    }

    //test tie points in image pair
    if(false){
        cElComposHomographie aUnitX(1, 0, 0);
        cElComposHomographie aUnitY(0, 1, 0);
        cElComposHomographie aUnitZ(0, 0, 1);
        cElHomographie  aUnitH =  cElHomographie(aUnitX,aUnitY,aUnitZ);
        GetGTTiePts(Pt2dr(0,0), Pt2dr(ImgSzR.x,ImgSzR.y), Pt2dr(ImgSzL.x,ImgSzL.y), ImgSzL, aTrans3DHR, aCamL, a3DL, aDSMInfoL, aCamR, a3DR, aDSMInfoR, bPrint, aUnitH, aDir, aImg1, aImg2, SH, nGap, dThres, false);
    }
    */
}

int MakeOneTrainingData_main(int argc,char ** argv)
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aImg1;
    std::string aImg2;

    std::string aOri1;
    std::string aOri2;
    std::string aDSMDirL = "";
    std::string aDSMDirR = "";
    std::string aDSMFileL = "MMLastNuage.xml";
    std::string aDSMFileR = "MMLastNuage.xml";

    std::string aOriGT1 = "";
    std::string aOriGT2 = "";
    std::string aDSMDirGTL = "";
    std::string aDSMDirGTR = "";
    std::string aDSMFileGTL = "";
    std::string aDSMFileGTR = "";

    std::string aPara3DHL = "";
    std::string aPara3DHR = "";

    std::string SH = "-GT";

    bool bTile = false;
    bool bSaveHomol = false;
    bool bPrepareDSM = true;
    double dRandomShift = 0;

    std::string aScene = "0001";

    bool bKeepRGBPatch = false;
    bool bKeepDSMPatch = false;
    bool bKeepTxt = true;

    double dScaleL = 1;
    double dScaleR = 1;

    int nGap = 5;
    double dThres = 3;
    Pt2dr aPatchSz = Pt2dr(640, 480);
    Pt2di seed = Pt2di(-1, -1);
    ElInitArgMain
     (
         argc,argv,
         LArgMain()   << EAMC(aImg1,"Master image name")
                << EAMC(aImg2,"Secondary image name")
                << EAMC(aOri1,"Rough orientation of master image (Ori1)")
                << EAMC(aOri2,"Rough orientation of secondary image (Ori2)"),
         LArgMain()
                << aCAS3D.ArgBasic()
                << EAM(aDSMDirL, "DSMDirL", true, "Rough DSM of master image (for improving the reprojecting accuracy), Def=none")
                << EAM(aDSMDirR, "DSMDirR", true, "Rough DSM of secondary image (for improving the reprojecting accuracy), Def=none")
                << EAM(aDSMFileL, "DSMFileL", true, "Rough DSM File of master image, Def=MMLastNuage.xml")
                << EAM(aDSMFileR, "DSMFileR", true, "Rough DSM File of secondary image, Def=MMLastNuage.xml")
                << EAM(aPara3DHL, "Para3DHL", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master image to secondary image, Def=none")
                << EAM(aPara3DHR, "Para3DHR", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of secondary image to master image, Def=none")
                << EAM(aPatchSz, "PatchSz", true, "Patch size, which means the input images will be croped into patches of this size, Def=[640, 480]")
                << EAM(seed, "CenterMatch", true, "the coordinate in secondary image which will be used as center for cropping patch (for developpers only), Def=[-1, -1] ([-1, -1] means randomly choose center)")
                << EAM(nGap, "Gap", false, "Gap for extracting GT tie pts (for developpers only), Def=5")
                << EAM(dThres, "Thres", false, "Threshold for checking 3D distance for GT tie pts (for developpers only), Def=3")
                << EAM(SH,"OutSH",true,"Output Homologue extenion for NB/NT mode of GT tie points, Def=-GT")
                << EAM(bTile,"Tile",true,"crop multiple patches by tiling, Def=false")
                << EAM(bSaveHomol,"SaveHomol",true,"Save Homologue in MicMac format, Def=false")
                << EAM(aOriGT1, "OriGT1", true, "GT orientation of master image (for generating GT tie points), Def=Ori1")
                << EAM(aOriGT2, "OriGT2", true, "GT orientation of secondary image (for generating GT tie points), Def=Ori2")
                << EAM(aDSMDirGTL, "DSMDirGTL", true, "GT DSM of master image (for generating GT tie points), Def=DSMDirL")
                << EAM(aDSMDirGTR, "DSMDirGTR", true, "GT DSM of secondary image (for generating GT tie points), Def=DSMDirR")
                << EAM(aDSMFileGTL, "DSMFileGTL", true, "GT DSM File of master image (for generating GT tie points), Def=DSMFileL")
                << EAM(aDSMFileGTR, "DSMFileGTR", true, "GT DSM File of secondary image (for generating GT tie points), Def=DSMFileL")
                << EAM(bPrepareDSM,"PrepareDSM",true,"copy DSM equalized image to the work directory and remove it after processing, Def=true")
                << EAM(dRandomShift,"RanShift",true,"Use GT Orientation and DSM combined with random shift within \"RanShift\" to synthesize the roughly aligned RGB and DSM patches, Def=0 (use rough orientation and DSM without random shift)")
                << EAM(aScene, "Scene", true, "Output folder name of scene, Def=0001")
                << EAM(bKeepRGBPatch,"KeepRGBPatch",true,"Keep the RGBpatches, Def=false")
                << EAM(bKeepDSMPatch,"KeepDSMPatch",true,"Keep the DSM patches, Def=false")
                << EAM(bKeepTxt,"KeepTxt",true,"Keep the txt files for resampling RGB and DSM patches, Def=true")
                << EAM(dScaleL,"ScaleL",true,"The factor used to scale the master images (for developpers only, when you want to use images with higher resolution instead), Def=1")
                << EAM(dScaleR,"ScaleR",true,"The factor used to scale the master images (for developpers only, when you want to use images with higher resolution instead), Def=1")
     );

    if(bKeepRGBPatch==false)
        bSaveHomol = false;

     if(aOriGT1.length() == 0)
         aOriGT1 = aOri1;
     if(aOriGT2.length() == 0)
         aOriGT2 = aOri2;
     if(aDSMDirGTL.length() == 0)
         aDSMDirGTL = aDSMDirL;
     if(aDSMDirGTR.length() == 0)
         aDSMDirGTR = aDSMDirR;
     if(aDSMFileGTL.length() == 0)
         aDSMFileGTL = aDSMFileL;
     if(aDSMFileGTR.length() == 0)
         aDSMFileGTR = aDSMFileR;

     //if use random shift to synthesize data, the rough orientation and DSM will be ignored
     if(fabs(dRandomShift) > 0.00001){
         aOri1 = aOriGT1;
         aOri2 = aOriGT2;
         aDSMDirL = aDSMDirGTL;
         aDSMDirR = aDSMDirGTR;
         aDSMFileL = aDSMFileGTL;
         aDSMFileR = aDSMFileGTR;
     }

     StdCorrecNameOrient(aOri1,"./",true);
     StdCorrecNameOrient(aOri2,"./",true);

     StdCorrecNameOrient(aOriGT1,"./",true);
     StdCorrecNameOrient(aOriGT2,"./",true);

     cTransform3DHelmert aTrans3DHL(aPara3DHL);
     cTransform3DHelmert aTrans3DHR(aPara3DHR);

     std::string aDSMImgNameL;
     std::string aDSMImgNameR;
     if(bPrepareDSM == true){
         aDSMImgNameL = PrepareDSM(aDSMDirL, aDSMFileL);
         aDSMImgNameR = PrepareDSM(aDSMDirR, aDSMFileR);
     }

     MakeTrainingData( aCAS3D.mDir, aImg1,  aImg2, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR,  aOri1, aOri2, aDSMFileGTL, aDSMFileGTR, aDSMDirGTL, aDSMDirGTR,  aOriGT1, aOriGT2, aCAS3D.mICNM, aTrans3DHL, aTrans3DHR, aCAS3D.mPrint, aPatchSz, seed, nGap, dThres, SH, bTile, bSaveHomol, dRandomShift, aScene, bKeepRGBPatch, bKeepDSMPatch, dScaleL, dScaleR);

     if(bPrepareDSM == true){
         RemoveDSM(aDSMImgNameL);
         RemoveDSM(aDSMImgNameR);
     }

     if(bKeepTxt == false){
         std::string aOutDirDSMTxt = aCAS3D.mDir + "/Tmp_TrainingPatches/"+aScene+"/DSMTxt/";
         if (ELISE_fp::exist_file(aOutDirDSMTxt) == true){
             std::string cmmd = "rm -r " + aOutDirDSMTxt;
             cout<<cmmd<<endl;
             System(cmmd);
        }
         else
             printf("%s did not exist.\n", aOutDirDSMTxt.c_str());
         std::string aOutDirRGBTxt = aCAS3D.mDir + "/Tmp_TrainingPatches/"+aScene + "/RGBTxt/";
         if (ELISE_fp::exist_file(aOutDirRGBTxt) == true){
             std::string cmmd = "rm -r " + aOutDirRGBTxt;
             cout<<cmmd<<endl;
             System(cmmd);
        }
         else
             printf("%s did not exist.\n", aOutDirRGBTxt.c_str());
     }

   return EXIT_SUCCESS;
}

int MakeTrainingData_main(int argc,char ** argv)
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aFullPattern1;
    std::string aFullPattern2;

    std::string aOri1;
    std::string aOri2;
    std::string aDSMDirL = "";
    std::string aDSMDirR = "";
    std::string aDSMFileL = "MMLastNuage.xml";
    std::string aDSMFileR = "MMLastNuage.xml";

    std::string aOriGT1 = "";
    std::string aOriGT2 = "";
    std::string aDSMDirGTL = "";
    std::string aDSMDirGTR = "";
    std::string aDSMFileGTL = "";
    std::string aDSMFileGTR = "";

    std::string aPara3DHL = "";
    std::string aPara3DHR = "";

    std::string SH = "-GT";

    bool bTile = false;
    bool bSaveHomol = false;

    double dRandomShift = 25;

    std::string aScene = "";
    std::string aScenePostFix = "";

    bool bKeepRGBPatch = false;
    bool bKeepDSMPatch = false;
    bool bKeepTxt = true;

    double dScaleL = 1;
    double dScaleR = 1;

    int nGap = 5;
    double dThres = 3;
    Pt2dr aPatchSz = Pt2dr(640, 480);
    Pt2di seed = Pt2di(-1, -1);
    ElInitArgMain
     (
         argc,argv,
         LArgMain()   << EAMC(aFullPattern1,"Master image name (Dir+Pattern, or txt file of image list)")
                << EAMC(aFullPattern2,"Secondary image name (Dir+Pattern, or txt file of image list)")
                << EAMC(aOri1,"Ori1: Orientation of master image")
                << EAMC(aOri2,"Ori2: Orientation of secondary image"),
         LArgMain()
                << aCAS3D.ArgBasic()
                << EAM(aDSMDirL, "DSMDirL", true, "DSM of master image (for improving the reprojecting accuracy), Def=none")
                << EAM(aDSMDirR, "DSMDirR", true, "DSM of secondary image (for improving the reprojecting accuracy), Def=none")
                << EAM(aDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
                << EAM(aDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml")
                << EAM(aPara3DHL, "Para3DHL", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master image to secondary image, Def=none")
                << EAM(aPara3DHR, "Para3DHR", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of secondary image to master image, Def=none")
                << EAM(aPatchSz, "PatchSz", true, "Patch size, which means the input images will be croped into patches of this size, Def=[640, 480]")
                << EAM(seed, "CenterMatch", true, "the coordinate in secondary image which will be used as center for cropping patch (for developpers only), Def=[-1, -1] ([-1, -1] means randomly choose center)")
                << EAM(nGap, "Gap", false, "Gap for extracting GT tie pts (for developpers only), Def=5")
                << EAM(dThres, "Thres", false, "Threshold for checking 3D distance for GT tie pts (for developpers only), Def=3")
                << EAM(SH,"OutSH",true,"Output Homologue extenion for NB/NT mode of GT tie points, Def=-GT")
                << EAM(bTile,"Tile",true,"crop multiple patches by tiling, Def=false")
                << EAM(bSaveHomol,"SaveHomol",true,"Save Homologue in MicMac format, Def=false")
                << EAM(aOriGT1, "OriGT1", true, "GT orientation of master image (for generating GT tie points), Def=Ori1")
                << EAM(aOriGT2, "OriGT2", true, "GT orientation of secondary image (for generating GT tie points), Def=Ori2")
                << EAM(aDSMDirGTL, "DSMDirGTL", true, "GT DSM of master image (for generating GT tie points), Def=DSMDirL")
                << EAM(aDSMDirGTR, "DSMDirGTR", true, "GT DSM of secondary image (for generating GT tie points), Def=DSMDirR")
                << EAM(aDSMFileGTL, "DSMFileGTL", true, "GT DSM File of master image (for generating GT tie points), Def=DSMFileL")
                << EAM(aDSMFileGTR, "DSMFileGTR", true, "GT DSM File of secondary image (for generating GT tie points), Def=DSMFileL")
                << EAM(dRandomShift,"RanShift",true,"Use GT Orientation and DSM combined with random shift within \"RanShift\" to synthesize the roughly aligned RGB and DSM patches, Def=25 (use rough orientation and DSM without random shift)")
                << EAM(aScene, "Scene", true, "Output folder name of scene, Def=MasterImageName")
                << EAM(aScenePostFix, "ScenePostFix", true, "PostFix of scene, Def=none")
                << EAM(bKeepRGBPatch,"KeepRGBPatch",true,"Keep the RGBpatches, Def=false")
                << EAM(bKeepDSMPatch,"KeepDSMPatch",true,"Keep the DSM patches, Def=false")
                << EAM(bKeepTxt,"KeepTxt",true,"Keep the txt files for resampling RGB and DSM patches, Def=true")
                << EAM(dScaleL,"ScaleL",true,"The factor used to scale the master images (for developpers only, when you want to use images with higher resolution instead), Def=1")
                << EAM(dScaleR,"ScaleR",true,"The factor used to scale the master images (for developpers only, when you want to use images with higher resolution instead), Def=1")
     );

     std::vector<std::string> aVIm1;
     std::vector<std::string> aVIm2;
     GetImgListVec(aFullPattern1, aVIm1);
     GetImgListVec(aFullPattern2, aVIm2);

     if(aOriGT1.length() == 0)
         aOriGT1 = aOri1;
     if(aOriGT2.length() == 0)
         aOriGT2 = aOri2;
     if(aDSMDirGTL.length() == 0)
         aDSMDirGTL = aDSMDirL;
     if(aDSMDirGTR.length() == 0)
         aDSMDirGTR = aDSMDirR;
     if(aDSMFileGTL.length() == 0)
         aDSMFileGTL = aDSMFileL;
     if(aDSMFileGTR.length() == 0)
         aDSMFileGTR = aDSMFileR;

     //if use random shift to synthesize data, the rough orientation and DSM will be ignored
     if(fabs(dRandomShift) > 0.00001){
         aOri1 = aOriGT1;
         aOri2 = aOriGT2;
         aDSMDirL = aDSMDirGTL;
         aDSMDirR = aDSMDirGTR;
         aDSMFileL = aDSMFileGTL;
         aDSMFileR = aDSMFileGTR;
     }

     cTransform3DHelmert aTrans3DHL(aPara3DHL);
     cTransform3DHelmert aTrans3DHR(aPara3DHR);

     std::string aImg1;
     std::string aImg2;

     std::list<std::string> aComm;

     std::string aOptPara="";
     if (EAMIsInit(&aDSMDirL))           aOptPara += " DSMDirL=" + aDSMDirL;
     if (EAMIsInit(&aDSMDirR))           aOptPara += " DSMDirR=" + aDSMDirR;
     if (EAMIsInit(&aDSMFileL))          aOptPara += " DSMFileL=" + aDSMFileL;
     if (EAMIsInit(&aDSMFileR))          aOptPara += " DSMFileR=" + aDSMFileR;
     if (EAMIsInit(&aPara3DHL))          aOptPara += " Para3DHL=" + aPara3DHL;
     if (EAMIsInit(&aPara3DHR))          aOptPara += " Para3DHR=" + aPara3DHR;
     if (EAMIsInit(&aPatchSz))           aOptPara += " PatchSz=[" + ToString(aPatchSz.x) + "," + ToString(aPatchSz.y) + "] ";
     if (EAMIsInit(&seed))               aOptPara += " CenterMatch=[" + ToString(seed.x) + "," + ToString(seed.y) + "]";
     if (EAMIsInit(&nGap))               aOptPara += " Gap=" + ToString(nGap);
     if (EAMIsInit(&dThres))             aOptPara += " Thres=" + ToString(dThres);
     if (EAMIsInit(&SH))                 aOptPara += " OutSH=" + SH;
     if (EAMIsInit(&bTile))              aOptPara += " Tile=" + ToString(bTile);
     if (EAMIsInit(&bSaveHomol))         aOptPara += " SaveHomol=" + ToString(bSaveHomol);
     if (EAMIsInit(&aOriGT1))            aOptPara += " OriGT1=" + aOriGT1;
     if (EAMIsInit(&aOriGT2))            aOptPara += " OriGT2=" + aOriGT2;
     if (EAMIsInit(&aDSMDirGTL))         aOptPara += " DSMDirGTL=" + aDSMDirGTL;
     if (EAMIsInit(&aDSMDirGTR))         aOptPara += " DSMDirGTR=" + aDSMDirGTR;
     if (EAMIsInit(&aDSMFileGTL))        aOptPara += " DSMFileGTL=" + aDSMFileGTL;
     if (EAMIsInit(&aDSMFileGTR))        aOptPara += " DSMFileGTR=" + aDSMFileGTR;
     if (EAMIsInit(&dScaleL))        aOptPara += " ScaleL=" + ToString(dScaleL);
     if (EAMIsInit(&dScaleR))        aOptPara += " ScaleR=" + ToString(dScaleR);
     //if (EAMIsInit(&aScene))             aOptPara += " Scene=" + aScene;
     if (bKeepTxt == false)               aOptPara += " KeepTxt=true";        //avoid to remove the same buffer folder in each single command
     if (EAMIsInit(&bKeepRGBPatch))           aOptPara += " KeepRGBPatch="+ToString(int(bKeepRGBPatch));
     if (EAMIsInit(&bKeepDSMPatch))           aOptPara += " KeepDSMPatch="+ToString(int(bKeepDSMPatch));
     aOptPara += " PrepareDSM=false";   //avoid to prepare and remove the same DSM in each single command
     aOptPara += " RanShift=" + ToString(dRandomShift);

     std::string aDSMImgNameL = PrepareDSM(aDSMDirL, aDSMFileL);
     std::string aDSMImgNameR = PrepareDSM(aDSMDirR, aDSMFileR);

     int nIdx = 0;
     for(int i=0; i<int(aVIm1.size()); i++)
     {
         aImg1 = aVIm1[i];
         for(int j=0; j<int(aVIm2.size()); j++)
         {
             aImg2 = aVIm2[j];
             /*
             printf("*********************************%d********************************\n", nIdx);
             printf("%s, %s\n", aImg1.c_str(), aImg2.c_str());
             //MakeTrainingData( aCAS3D.mDir, aImg1,  aImg2, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR,  aOri1, aOri2, aDSMFileGTL, aDSMFileGTR, aDSMDirGTL, aDSMDirGTR,  aOriGT1, aOriGT2, aCAS3D.mICNM, aTrans3DHL, aTrans3DHR, aCAS3D.mPrint, aPatchSz, seed, nGap, dThres, SH, bTile, bSaveHomol);
             */

             std::string aOneComm = MMBinFile(MM3DStr) + "TestLib MakeOneTrainingData " + aImg1 + BLANK + aImg2 + BLANK + aOri1 + BLANK + aOri2 + aOptPara;
             if(aScene.length() == 0)
                 aOneComm += " Scene=" + StdPrefix(aImg1) + aScenePostFix;
             else
                 aOneComm += " Scene=" + aScene;
             //cout<<aOneComm<<endl;
             aComm.push_back(aOneComm);
             nIdx++;
         }
     }

     cEl_GPAO::DoComInParal(aComm);

     RemoveDSM(aDSMImgNameL);
     RemoveDSM(aDSMImgNameR);

     if(bKeepTxt == false){
         std::string cmmd;
         std::string aOutDirDSMTxt = aCAS3D.mDir + "/Tmp_TrainingPatches/"+aScene+"/DSMTxt/";
         //if (ELISE_fp::exist_file(aOutDirDSMTxt) == true){
             cmmd = "rm -r " + aOutDirDSMTxt;
             cout<<cmmd<<endl;
             System(cmmd);
         /*}
         else
             printf("%s did not exist.\n", aOutDirDSMTxt.c_str());*/
         std::string aOutDirRGBTxt = aCAS3D.mDir + "/Tmp_TrainingPatches/"+aScene + "/RGBTxt/";
         //if (ELISE_fp::exist_file(aOutDirRGBTxt) == true){
             cmmd = "rm -r " + aOutDirRGBTxt;
             cout<<cmmd<<endl;
             System(cmmd);
         /*}
         else
             printf("%s did not exist.\n", aOutDirRGBTxt.c_str());*/
     }

     std::string cmmd = "python3 /home/lulin/Documents/Code/Txt2Npz.py --DirName "+aCAS3D.mDir+"/Tmp_TrainingPatches/"+aScene+"/HomolNpz/ --RemoveOriFile 1";
     cout<<cmmd<<endl;

   return EXIT_SUCCESS;
}

