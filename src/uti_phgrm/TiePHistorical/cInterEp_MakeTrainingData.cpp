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


void CropPatch(Pt2dr aPtOri, Pt2dr aPatchSz, std::string aImg2, std::string aOutDir, std::string aSubImg1)
{
    std::string aClipSz = " ["+std::to_string(int(aPatchSz.x))+","+std::to_string(int(aPatchSz.y))+"] ";

    std::string aComBaseClip = MMBinFile(MM3DStr) + "ClipIm " + aImg2 + " ";
    std::string aComClipMasterImg = aComBaseClip + " ["+std::to_string(int(aPtOri.x))+","+std::to_string(int(aPtOri.y))+"] " + aClipSz + " Out="+aOutDir+"/"+aSubImg1;
    cout<<aComClipMasterImg<<endl;
    System(aComClipMasterImg);
}

std::string GetDSMPatch(std::string aImg, std::string aDSMDir, std::string aDSMFile, Pt2dr* ImgCorner, Pt2dr PatchSz, cGet3Dcoor a3DL, cDSMInfo aDSMInfoL, bool bPrint, std::string aOutDirDSM, std::string aOutDirTxt)
//, std::vector<Pt2dr> & DSMCoor
{
    Pt2dr PatchCorner[4];
    Pt2dr origin = Pt2dr(0, 0);
    PatchCorner[0] = origin;
    PatchCorner[1] = Pt2dr(origin.x+PatchSz.x, origin.y);
    PatchCorner[2] = Pt2dr(origin.x+PatchSz.x, origin.y+PatchSz.y);
    PatchCorner[3] = Pt2dr(origin.x, origin.y+PatchSz.y);

    std::string aDSMName = aDSMInfoL.GetDSMName(aDSMFile, aDSMDir);
    std::string aDSMImgName = StdPrefix(aDSMName) + "_gray.tif_sfs." + StdPostfix(aDSMName);

    if (ELISE_fp::exist_file(aDSMDir + "/" + aDSMImgName) == false){
        cout<<"aDSMImgName: "<<aDSMImgName<<endl;
        std::string cmmd = "mm3d TestLib DSM_Equalization "+aDSMDir+" DSMFile="+aDSMFile;
        cout<<cmmd<<endl;
        System(cmmd);
        cmmd = "mm3d TestLib Wallis "+StdPrefix(aDSMName)+"_gray.tif Dir="+aDSMDir;
        cout<<cmmd<<endl;
        System(cmmd);
    }

    std::string aNameSave = StdPrefix(aImg) + "_DSM";
    std::string aSubImg2 = aNameSave + "." + StdPostfix(aImg);
    aNameSave += ".txt";
    FILE * fpOutput = fopen((aNameSave).c_str(), "w");

    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = ImgCorner[i];
        Pt3dr aPTer1;
        bool bPreciseL;
        aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD(), true);
        Pt2dr aPtInDSM = aDSMInfoL.Get2DcoorInDSM(aPTer1);
        fprintf(fpOutput, "%lf %lf %lf %lf\n", PatchCorner[i].x, PatchCorner[i].y, aPtInDSM.x, aPtInDSM.y);
    }
    fclose(fpOutput);

    std::string cmmd = "cp " + aDSMDir + "/" + aDSMImgName + " " + aDSMImgName;
    cout<<cmmd<<endl;
    System(cmmd);
    std::string aComBaseResample = MMBinFile(MM3DStr) + "TestLib OneReechFromAscii ";
    std::string aComResampleSndImg = aComBaseResample + aDSMImgName + " " + aNameSave + " Out="+aSubImg2 + " Show=true";
    cout<<aComResampleSndImg<<endl;
    System(aComResampleSndImg);

    cmmd = "mv " + aSubImg2 + " " + aOutDirDSM + aSubImg2;
    cout<<cmmd<<endl;
    System(cmmd);
    cmmd = "mv " + aNameSave + " " + aOutDirTxt + aNameSave;
    cout<<cmmd<<endl;
    System(cmmd);
    cmmd = "rm " + aDSMImgName;
    cout<<cmmd<<endl;
    System(cmmd);

    return aSubImg2;
}

void GetGTTiePts(Pt2dr aPtOriR, Pt2dr PatchSz, Pt2di ImgSz1, cTransform3DHelmert aTrans3DHR, cBasicGeomCap3D * aCamL, cGet3Dcoor a3DL, cDSMInfo aDSMInfoL, cBasicGeomCap3D * aCamR, cGet3Dcoor a3DR, cDSMInfo aDSMInfoR, bool bPrint, cElHomographie aHomo, std::string input_dir, std::string aImg1, std::string aImg2, std::string SH, int nGap, double dThres)
{
    ElPackHomologue aPack;
    Pt2dr aPtL, aPtR, aPtRinPatch;
    int i, j;
    //int nGap = 20;
    int nWid = int(PatchSz.x/nGap);
    int nHei = int(PatchSz.y/nGap);

    Pt3dr aPTer1, aPTer2;

    //double dThres = 3;

    for(i=0; i<nWid; i++){
        aPtRinPatch.x = i*nGap;
        aPtR.x = aPtOriR.x + aPtRinPatch.x;
        for(j=0; j<nHei; j++){
            aPtRinPatch.y = j*nGap;
            aPtR.y = aPtOriR.y + aPtRinPatch.y;

            //compute tie point in left image
            bool bPrecise;
            aPTer1 = a3DR.Get3Dcoor(aPtR, aDSMInfoR, bPrecise, bPrint);//, a3DL.GetGSD(), true);
            aPTer1 = aTrans3DHR.Transform3Dcoor(aPTer1);
            aPtL = aCamL->Ter2Capteur(aPTer1);

            //check out of border
            if(aPtL.x < 0 || aPtL.y < 0 || aPtL.x >= ImgSz1.x || aPtL.y >= ImgSz1.y)
            {
                continue;
            }


            //check depth
            aPTer2 = a3DL.Get3Dcoor(aPtL, aDSMInfoL, bPrecise, bPrint);
            double dDis = pow((pow((aPTer1.x-aPTer2.x), 2), pow((aPTer1.y-aPTer2.y), 2), pow((aPTer1.z-aPTer2.z), 2)), 0.5);
            if(dDis > dThres)
                continue;

            //transform pt in left image to pt in left patch
            aPtL = aHomo(aPtL);
            aPack.Cple_Add(ElCplePtsHomologues(aPtL, aPtRinPatch));
        }
    }

    std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+SH;
    std::string aComInv = "mm3d SEL" + BLANK + input_dir + BLANK + aImg2 + BLANK + aImg1 + BLANK + "KH=NT SzW=[600,600] SH="+SH;
    printf("%s\n%s\n", aCom.c_str(), aComInv.c_str());

    std::string aSHDir = input_dir + "/Homol" + SH+"/";
    if (ELISE_fp::exist_file(aSHDir) == false)
        ELISE_fp::MkDir(aSHDir);
    std::string aNewDir = aSHDir + "Pastis" + aImg1;
    if (ELISE_fp::exist_file(aNewDir) == false)
        ELISE_fp::MkDir(aNewDir);
    std::string aNameFile1 = aNewDir + "/"+aImg2+".txt";

    aNewDir = aSHDir + "Pastis" + aImg2;
    if (ELISE_fp::exist_file(aNewDir) == false)
        ELISE_fp::MkDir(aNewDir);
    std::string aNameFile2 = aNewDir + "/"+aImg1+".txt";

    FILE * fpTiePt1 = fopen(aNameFile1.c_str(), "w");
    FILE * fpTiePt2 = fopen(aNameFile2.c_str(), "w");
    for (ElPackHomologue::iterator itCpl=aPack.begin();itCpl!=aPack.end() ; itCpl++)
    {
        ElCplePtsHomologues tiept = itCpl->ToCple();
        fprintf(fpTiePt1, "%lf %lf %lf %lf\n", tiept.P1().x, tiept.P1().y, tiept.P2().x, tiept.P2().y);
        fprintf(fpTiePt2, "%lf %lf %lf %lf\n", tiept.P2().x, tiept.P2().y, tiept.P1().x, tiept.P1().y);
    }
    fclose(fpTiePt1);
    fclose(fpTiePt2);
}

cElHomographie GetSecondaryPatch(Pt2dr* aPCornerL, Pt2dr* aPCornerR, Pt2di ImgSzR, Pt2dr PatchSz, cTransform3DHelmert aTrans3DH, cBasicGeomCap3D * aCamR, cGet3Dcoor a3DL, cDSMInfo aDSMInfoL, std::string aOutImg2, std::string aSubImg1, bool bPrint, std::string aOutDirImg, std::string aOutDirTxt)
{
    double dScaleL = 1;
    double dScaleR = 1;

    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = aPCornerL[i];
        Pt3dr aPTer1;
        bool bPreciseL;
        aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD(), true);
        aPTer1 = aTrans3DH.Transform3Dcoor(aPTer1);

        aPCornerR[i] = aCamR->Ter2Capteur(aPTer1);
        aPCornerR[i].x = aPCornerR[i].x/dScaleL/dScaleR;
        aPCornerR[i].y = aPCornerR[i].y/dScaleL/dScaleR;

        if(bPrint)
        {
            printf("%dth: CornerL: [%.2lf\t%.2lf], ImEtProf2Terrain: [%.2lf\t%.2lf\t%.2lf], CornerR: [%.2lf\t%.2lf]\n", i, aPCornerL[i].x, aPCornerL[i].y, aPTer1.x, aPTer1.y, aPTer1.z, aPCornerR[i].x, aPCornerR[i].y);
        }
    }

    Pt2dr aPCornerPatch[4];
    Pt2dr origin = Pt2dr(0, 0);
    aPCornerPatch[0] = origin;
    aPCornerPatch[1] = Pt2dr(origin.x+PatchSz.x, origin.y);
    aPCornerPatch[2] = Pt2dr(origin.x+PatchSz.x, origin.y+PatchSz.y);
    aPCornerPatch[3] = Pt2dr(origin.x, origin.y+PatchSz.y);

    std::string aNameSave = StdPrefix(aSubImg1) + ".txt";
    //if the patch is not out of the image border
    if(FallInBox(aPCornerR, Pt2dr(0,0), ImgSzR) == true)
    {
        FILE * fpOutput = fopen((aNameSave).c_str(), "w");
        for(int i=0; i<4; i++)
        {
            fprintf(fpOutput, "%lf %lf %lf %lf\n", aPCornerPatch[i].x, aPCornerPatch[i].y, aPCornerR[i].x, aPCornerR[i].y);
        }
        fclose(fpOutput);
        //cout<<aNameSave<<" saved"<<endl;

        std::string aComBaseResample = MMBinFile(MM3DStr) + "TestLib OneReechFromAscii ";
        std::string aComResampleSndImg = aComBaseResample + aOutImg2  + " " + aNameSave + " Out="+aSubImg1 + " Show=true";
        cout<<aComResampleSndImg<<endl;
        System(aComResampleSndImg);

        std::string aMvTif = "mv "+aSubImg1 + " "+aOutDirImg+aSubImg1;
        cout<<aMvTif<<endl;
        System(aMvTif);
        /*
        std::string aMvTxt = "mv "+aNameSave + " "+aOutDirTxt+aNameSave;
        cout<<aMvTxt<<endl;
        System(aMvTxt);*/
    }
    else
    {
        if(bPrint)
            cout<<aNameSave<<" out of border, hence the current patch is not saved"<<endl;
    }

    ElPackHomologue aPack = ElPackHomologue::FromFile(aNameSave);
    double anEcart,aQuality;
    bool Ok;
    cElHomographie aHomo = cElHomographie::RobustInit(anEcart,&aQuality,aPack,Ok,50,80.0,2000);
    std::cout << "Ecart " << anEcart << " ; Quality " << aQuality    << " \n";
    return aHomo.Inverse();
}

void MakeTrainingData(std::string aDir,std::string aImg1, std::string aImg2, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR, std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, cTransform3DHelmert aTrans3DHL, cTransform3DHelmert aTrans3DHR, bool bPrint, Pt2dr aPatchSz, Pt2di seed, int nGap, double dThres, std::string SH)
{
    if (ELISE_fp::exist_file(aImg1) == false || ELISE_fp::exist_file(aImg2) == false)
    {
        cout<<aImg1<<" or "<<aImg2<<" didn't exist, hence skipped"<<endl;
        return;
    }

    Tiff_Im aRGBIm1(aImg1.c_str());
    Pt2di ImgSzL = aRGBIm1.sz();
    Tiff_Im aRGBIm2(aImg2.c_str());
    Pt2di ImgSzR = aRGBIm2.sz();

    std::string aNameOriL = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    std::string aNameOriR = aICNM->StdNameCamGenOfNames(aOri2, aImg2);
    int aType = eTIGB_Unknown;
    cBasicGeomCap3D * aCamL = cBasicGeomCap3D::StdGetFromFile(aNameOriL,aType);
    cBasicGeomCap3D * aCamR = cBasicGeomCap3D::StdGetFromFile(aNameOriR,aType);
    cGet3Dcoor a3DL(aNameOriL);
    cDSMInfo aDSMInfoL = a3DL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cGet3Dcoor a3DR(aNameOriR);
    cDSMInfo aDSMInfoR = a3DR.SetDSMInfo(aDSMFileR, aDSMDirR);

    Pt2dr aPCornerL[4];
    Pt2dr origin = Pt2dr(0, 0);
    aPCornerL[0] = origin;
    aPCornerL[1] = Pt2dr(origin.x+ImgSzL.x, origin.y);
    aPCornerL[2] = Pt2dr(origin.x+ImgSzL.x, origin.y+ImgSzL.y);
    aPCornerL[3] = Pt2dr(origin.x, origin.y+ImgSzL.y);

    Pt2dr aPLPredinR[4];  //to Save zone in secondary image which is overlapping with left image
    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = aPCornerL[i];
        Pt3dr aPTer1;
        bool bPreciseL;
        aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD());

        aPTer1 = aTrans3DHL.Transform3Dcoor(aPTer1);
        Pt2dr ptPred = a3DR.Get2Dcoor(aPTer1);
        aPLPredinR[i] = ptPred;

        CheckRange(0, ImgSzR.x, aPLPredinR[i].x);
        CheckRange(0, ImgSzR.y, aPLPredinR[i].y);

        if(bPrint)
        {
            printf("%dth: CornerL: [%.2lf\t%.2lf], ImEtProf2Terrain: [%.2lf\t%.2lf\t%.2lf], CornerR: [%.2lf\t%.2lf], CornerRNew: [%.2lf\t%.2lf]\n", i, aPCornerL[i].x, aPCornerL[i].y, aPTer1.x, aPTer1.y, aPTer1.z, ptPred.x, ptPred.y, aPLPredinR[i].x, aPLPredinR[i].y);
        }
    }
    std::vector<int> centerX;
    std::vector<int> centerY;

    double dMaxX = max(max(aPLPredinR[0].x, aPLPredinR[1].x), max(aPLPredinR[2].x, aPLPredinR[3].x));
    double dMinX = min(min(aPLPredinR[0].x, aPLPredinR[1].x), min(aPLPredinR[2].x, aPLPredinR[3].x));
    double dMaxY = max(max(aPLPredinR[0].y, aPLPredinR[1].y), max(aPLPredinR[2].y, aPLPredinR[3].y));
    double dMinY = min(min(aPLPredinR[0].y, aPLPredinR[1].y), min(aPLPredinR[2].y, aPLPredinR[3].y));
    printf("dMaxX: %lf\tdMaxX: %lf\tdMaxX: %lf\tdMaxX: %lf\n", dMaxX, dMinX, dMaxY, dMinY);

    if(dMaxX < dMinX || dMaxY < dMinY)
    {
        printf("There is no overlapping area in the image pair, hence skipped.\n");
        return;
    }

    srand((int)time(0));
    //randomly get the coordinate in secondary image which will be used as center for cropping patch
    GetRandomNum(dMinX, dMaxX, 1, centerX);
    GetRandomNum(dMinY, dMaxY, 1, centerY);

    //if user set CenterMatch
    if(seed.x > 0){
        if(seed.x < ImgSzR.x)
            centerX[0] = seed.x;
        else
            printf("x of CenterMatch %d should be smaller than image width %d\n", seed.x, ImgSzR.x);
    }
    if(seed.y > 0){
        if(seed.y < ImgSzR.y)
            centerY[0] = seed.y;
        else
            printf("y of CenterMatch %d should be smaller than image width %d\n", seed.y, ImgSzR.y);
    }

    if(bPrint)
        printf("CenterMatch: %d\t%d\tImgSz: %d\t%d\n", centerX[0], centerY[0], ImgSzR.x, ImgSzR.y);

    std::string aOutDir = aDir + "/Tmp_Patches/";
    if (ELISE_fp::exist_file(aOutDir) == false)
        ELISE_fp::MkDir(aOutDir);
    std::string aOutDirDSM = aDir + "/Tmp_Patches/DSM/";
    if (ELISE_fp::exist_file(aOutDirDSM) == false)
        ELISE_fp::MkDir(aOutDirDSM);
    std::string aOutDirTxt = aDir + "/Tmp_Patches/Txt/";
    if (ELISE_fp::exist_file(aOutDirTxt) == false)
        ELISE_fp::MkDir(aOutDirTxt);
    std::string aOutDirImg = aDir + "/Tmp_Patches/Img/";
    if (ELISE_fp::exist_file(aOutDirImg) == false)
        ELISE_fp::MkDir(aOutDirImg);

    int m = 0;
    int n = 0;

    //Crop secondary patch
    Pt2dr aPCenterR = Pt2dr(centerX[0], centerY[0]);
    if(CheckRange(aPatchSz.x/2, dMaxX-aPatchSz.x/2, aPCenterR.x)==false ||
    CheckRange(aPatchSz.y/2, dMaxY-aPatchSz.y/2, aPCenterR.y)==false)
    {
        printf("There is no enough overlapping area in the image pair, hence skipped.\n");
        return;
    }
    Pt2dr aPtOriR = Pt2dr(aPCenterR.x-aPatchSz.x/2, aPCenterR.y-aPatchSz.y/2);
    std::string aSubImg1 = StdPrefix(aImg2) + "_" + StdPrefix(aImg1) + "_" + std::to_string(m) + "_" + std::to_string(n) + "_L." + StdPostfix(aImg2);
    CropPatch(aPtOriR, aPatchSz, aImg2, aOutDirImg, aSubImg1);

    Pt2dr aPatchCornerR[4];
    origin = aPtOriR;
    aPatchCornerR[0] = origin;
    aPatchCornerR[1] = Pt2dr(origin.x+aPatchSz.x, origin.y);
    aPatchCornerR[2] = Pt2dr(origin.x+aPatchSz.x, origin.y+aPatchSz.y);
    aPatchCornerR[3] = Pt2dr(origin.x, origin.y+aPatchSz.y);
    //Crop secondary DSM patch
    GetDSMPatch(aSubImg1, aDSMDirR, aDSMFileR, aPatchCornerR, aPatchSz, a3DR, aDSMInfoR, bPrint, aOutDirDSM, aOutDirTxt);

    std::string aSubImg2 = StdPrefix(aImg2) + "_" + StdPrefix(aImg1) + "_" + std::to_string(m) + "_" + std::to_string(n) + "_R." + StdPostfix(aImg2);

    //Same center, keep GSD and rotation difference
    if(false)
    {
        //Crop master patch
        Pt3dr aPTer1;
        /*if(bDSMR == true)
        {*/
            bool bPreciseR;
            aPTer1 = a3DR.Get3Dcoor(aPCenterR, aDSMInfoR, bPreciseR, bPrint);//, a3DR.GetGSD());
        /*}
        else
        {
            aPTer1 = a3DR.GetRough3Dcoor(aPCenterR);
        }*/

        aPTer1 = aTrans3DHR.Transform3Dcoor(aPTer1);
        Pt2dr aPCenterL = a3DL.Get2Dcoor(aPTer1);
        printf("CenterMatch in master image: %.0lf\t%.0lf\n", aPCenterL.x, aPCenterL.y);
        printf("CenterMatch in secondary image: %.0lf\t%.0lf\n", aPCenterR.x, aPCenterR.y);

        Pt2dr aPtOriL = Pt2dr(aPCenterL.x-aPatchSz.x/2, aPCenterL.y-aPatchSz.y/2);
        CropPatch(aPtOriL, aPatchSz, aImg1, aOutDirImg, aSubImg2);

        Pt2dr aPatchCornerL[4];
        origin = aPtOriL;
        aPatchCornerL[0] = origin;
        aPatchCornerL[1] = Pt2dr(origin.x+aPatchSz.x, origin.y);
        aPatchCornerL[2] = Pt2dr(origin.x+aPatchSz.x, origin.y+aPatchSz.y);
        aPatchCornerL[3] = Pt2dr(origin.x, origin.y+aPatchSz.y);
        GetDSMPatch(aSubImg1, aDSMDirL, aDSMFileL, aPatchCornerL, aPatchSz, a3DL, aDSMInfoL, bPrint, aOutDirDSM, aOutDirTxt);
        //RotateImgBy90Deg1("Tmp_Patches", aDSML, aDSML+"_90.tif");
        //RotateImgBy90Deg1("./", aImg1, aImg1+"_90.tif");
    }

    //Crop roughly aligned patches in master image
    if(true)
    {
        Pt2dr aPatchCornerL[4];
        cElHomographie aHomo = GetSecondaryPatch(aPatchCornerR, aPatchCornerL, ImgSzR, aPatchSz, aTrans3DHR, aCamL, a3DR, aDSMInfoR, aImg1, aSubImg2, bPrint, aOutDirImg, aOutDirTxt);
        for(int k=0; k<4; k++)
            printf("%dth: CornerL: [%.2lf\t%.2lf]\n", k, aPatchCornerL[k].x, aPatchCornerL[k].y);
        GetDSMPatch(aSubImg2, aDSMDirL, aDSMFileL, aPatchCornerL, aPatchSz, a3DL, aDSMInfoL, bPrint, aOutDirDSM, aOutDirTxt);

        GetGTTiePts(aPtOriR, aPatchSz, ImgSzL, aTrans3DHR, aCamL, a3DL, aDSMInfoL, aCamR, a3DR, aDSMInfoR, bPrint, aHomo, aOutDirImg, aSubImg1, aSubImg2, SH, nGap, dThres);
    }
}

int MakeTrainingData_main(int argc,char ** argv)
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aImg1;
    std::string aImg2;
    std::string aOri1;
    std::string aOri2;

    std::string aDSMDirL = "";
    std::string aDSMDirR = "";
    std::string aDSMFileL;
    std::string aDSMFileR;

    aDSMFileL = "MMLastNuage.xml";
    aDSMFileR = "MMLastNuage.xml";

    std::string aPara3DHL = "";
    std::string aPara3DHR = "";

    std::string SH = "-GT";

    int nGap = 5;
    double dThres = 3;
    Pt2dr aPatchSz = Pt2dr(640, 480);
    Pt2di seed = Pt2di(-1, -1);
    ElInitArgMain
     (
         argc,argv,
         LArgMain()   << EAMC(aImg1,"Master image name")
                << EAMC(aImg2,"Secondary image name")
                << EAMC(aOri1,"Orientation of master image")
                << EAMC(aOri2,"Orientation of secondary image"),
         LArgMain()
                << aCAS3D.ArgBasic()
                << EAM(aDSMDirL, "DSMDirL", true, "DSM of master image (for improving the reprojecting accuracy), Def=none")
                << EAM(aDSMDirR, "DSMDirR", true, "DSM of secondary image (for improving the reprojecting accuracy), Def=none")
                << EAM(aDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
                << EAM(aDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml")
                << EAM(aPara3DHL, "Para3DHL", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master image to secondary image, Def=none")
                << EAM(aPara3DHR, "Para3DHR", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of secondary image to master image, Def=none")
                << EAM(aPatchSz, "PatchSz", true, "Patch size, which means the input images will be croped into patches of this size, Def=[640, 480]")
                << EAM(seed, "CenterMatch", true, "the coordinate in secondary image which will be used as center for cropping patch (for developpers only), Def=[-1, -1]")
                << EAM(nGap, "Gap", false, "Gap for extracting GT tie pts (for developpers only), Def=5")
                << EAM(dThres, "Thres", false, "Threshold for checking 3D distance for GT tie pts (for developpers only), Def=3")
                << EAM(SH,"OutSH",true,"Output Homologue extenion for NB/NT mode of GT tie points, Def=-GT")
     );
     StdCorrecNameOrient(aOri1,"./",true);
     StdCorrecNameOrient(aOri2,"./",true);

     cTransform3DHelmert aTrans3DHL(aPara3DHL);
     cTransform3DHelmert aTrans3DHR(aPara3DHR);

     MakeTrainingData( aCAS3D.mDir, aImg1,  aImg2, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR,  aOri1, aOri2, aCAS3D.mICNM, aTrans3DHL, aTrans3DHR, aCAS3D.mPrint, aPatchSz, seed, nGap, dThres, SH);

   return EXIT_SUCCESS;
}

