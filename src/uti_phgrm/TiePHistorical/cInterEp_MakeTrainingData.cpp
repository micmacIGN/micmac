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



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe à
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
aooter-MicMac-eLiSe-25/06/2007*/

/*
void RotateImgBy90Deg1(std::string aDir, std::string aImg1, std::string aNameOut)
{
    Tiff_Im aIm1((aDir+"/"+aImg1).c_str());
    Pt2di ImgSzL = aIm1.sz();

    aNameOut = aDir+"/"+aNameOut;

    //std::string aNameOut = aDir+"/"+StdPrefix(aImg1)+"_R90.tif";

    L_Arg_Opt_Tiff aLArg = Tiff_Im::Empty_ARG;
    aLArg = aLArg + Arg_Tiff(Tiff_Im::ANoStrip());

    cout<<aImg1<<": "<<aIm1.phot_interp()<<", "<<Tiff_Im::RGBPalette<<";"<<(aIm1.phot_interp() == Tiff_Im::RGBPalette)<<endl;
    //cout<<aIm1.pal()<<endl;
    Tiff_Im TiffOut  =     (aIm1.phot_interp() == Tiff_Im::RGBPalette)  ?
                           Tiff_Im
                           (
                              aNameOut.c_str(),
                              Pt2di(ImgSzL.y, ImgSzL.x),
                              aIm1.type_el(),
                              Tiff_Im::No_Compr,
                              aIm1.pal(),
                              aLArg
                          )                    :
                           Tiff_Im
                           (
                              aNameOut.c_str(),
                              Pt2di(ImgSzL.y, ImgSzL.x),
                              aIm1.type_el(),
                              Tiff_Im::No_Compr,
                              aIm1.phot_interp(),
                              aLArg
                          );

    TIm2D<float, double> aTImProfPx(ImgSzL);
    ELISE_COPY
    (
    aTImProfPx.all_pts(),
    aIm1.in(),
    aTImProfPx.out()
    );

    TIm2D<float, double> aTImProfPxTmp(Pt2di(ImgSzL.y, ImgSzL.x));
    ELISE_COPY
    (
    aTImProfPxTmp.all_pts(),
    aTImProfPx.in()[Virgule(FY,FX)],
    aTImProfPxTmp.out()
    );

    //flip
    ELISE_COPY
    (
    TiffOut.all_pts(),
    aTImProfPxTmp.in(0)[Virgule(ImgSzL.y-FX,FY)],
    TiffOut.out()
    );
}
*/

/*
bool GetRandomCenter(Pt2dr* aPLPredinR, std::vector<int> & centerX, std::vector<int> & centerY)
{
    double dMaxX = max(max(aPLPredinR[0].x, aPLPredinR[1].x), max(aPLPredinR[2].x, aPLPredinR[3].x));
    double dMinX = min(min(aPLPredinR[0].x, aPLPredinR[1].x), min(aPLPredinR[2].x, aPLPredinR[3].x));
    double dMaxY = max(max(aPLPredinR[0].y, aPLPredinR[1].y), max(aPLPredinR[2].y, aPLPredinR[3].y));
    double dMinY = min(min(aPLPredinR[0].y, aPLPredinR[1].y), min(aPLPredinR[2].y, aPLPredinR[3].y));
    printf("dMaxX: %lf\tdMaxX: %lf\tdMaxX: %lf\tdMaxX: %lf\n", dMaxX, dMinX, dMaxY, dMinY);

    if(dMaxX < dMinX || dMaxY < dMinY)
        return false;

    srand((int)time(0));
    GetRandomNum(dMinX, dMaxX, 1, centerX);
    GetRandomNum(dMinY, dMaxY, 1, centerY);
    return true;
}

void CheckRange(int nMin, int nMax, int & value)
{
    if(value < nMin)
        value = nMin;
    if(value > nMax)
        value = nMax;
}
*/



void CropPatch(Pt2dr aPtOri, Pt2dr aPatchSz, std::string aImg2, std::string aOutDir, int m, int n)
{
    std::string aClipSz = " ["+std::to_string(int(aPatchSz.x))+","+std::to_string(int(aPatchSz.y))+"] ";

    std::string aSubImg1 = StdPrefix(aImg2) + "_" + std::to_string(m) + "_" + std::to_string(n) + "." + StdPostfix(aImg2);
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
    std::string aNameSave = StdPrefix(aImg) + "_DSM";
    std::string aSubImg2 = aNameSave + "." + StdPostfix(aImg);
    aNameSave += ".txt";
    FILE * fpOutput = fopen((aNameSave).c_str(), "w");

    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = ImgCorner[i];
        Pt3dr aPTer1;
        /*if(bDSM == true)
        {*/
            bool bPreciseL;
            aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD(), true);
            //Pt2dr ptPred = a3DL.Get2Dcoor(aPTer1);
            //printf("aPL.x, aPL.y, ptPred.x, ptPred.y: %.2lf\t%.2lf, %.2lf\t%.2lf\n", aPL.x, aPL.y, ptPred.x, ptPred.y);
        /*}
        else
        {
            aPTer1 = a3DL.GetRough3Dcoor(aPL);
        }*/
        Pt2dr aPtInDSM = aDSMInfoL.Get2DcoorInDSM(aPTer1);
        //DSMCoor.push_back(aPtInDSM);
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



//std::string GetDSMPatch(std::string aImg, std::string aDSMDir, std::string aDSMFile, Pt2dr aPtCenter, Pt2dr PatchSz, cGet3Dcoor a3DL, TIm2D<float,double> aTImProfPxL, bool bDSM)
void GetSecondaryPatch(Pt2dr* aPCornerL, Pt2dr* aPCornerR, Pt2di ImgSzR, Pt2dr PatchSz, cTransform3DHelmert aTrans3DH, cBasicGeomCap3D * aCamR, cGet3Dcoor a3DL, cDSMInfo aDSMInfoL, std::string aOutImg2, std::string aOriginImg1, int m, int n, bool bPrint, std::string aOutDirImg, std::string aOutDirTxt)
{
    double dScaleL = 1;
    double dScaleR = 1;

    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = aPCornerL[i];
        Pt3dr aPTer1;
        /*if(bDSM == true)
        {*/
            bool bPreciseL;
            aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD(), true);
        /*}
        else
        {
            aPTer1 = a3DL.GetRough3Dcoor(aPL);
        }*/
        aPTer1 = aTrans3DH.Transform3Dcoor(aPTer1);

        aPCornerR[i] = aCamR->Ter2Capteur(aPTer1);
        aPCornerR[i].x = aPCornerR[i].x/dScaleL/dScaleR;
        aPCornerR[i].y = aPCornerR[i].y/dScaleL/dScaleR;

        if(bPrint)
        {
            printf("%dth: CornerL: [%.2lf\t%.2lf], ImEtProf2Terrain: [%.2lf\t%.2lf\t%.2lf], CornerR: [%.2lf\t%.2lf]\n", i, aPCornerL[i].x, aPCornerL[i].y, aPTer1.x, aPTer1.y, aPTer1.z, aPCornerR[i].x, aPCornerR[i].y);
            //printf("ImEtZ2Terrain: [%.2lf\t%.2lf\t%.2lf]\n", Pt_H_sol.x, Pt_H_sol.y, Pt_H_sol.z);
        }
    }

    Pt2dr aPCornerPatch[4];
    Pt2dr origin = Pt2dr(0, 0);
    aPCornerPatch[0] = origin;
    aPCornerPatch[1] = Pt2dr(origin.x+PatchSz.x, origin.y);
    aPCornerPatch[2] = Pt2dr(origin.x+PatchSz.x, origin.y+PatchSz.y);
    aPCornerPatch[3] = Pt2dr(origin.x, origin.y+PatchSz.y);

    std::string aNameSave = StdPrefix(aOutImg2) + "_" + StdPrefix(aOriginImg1) + "_" + std::to_string(m) + "_" + std::to_string(n);
    std::string aSubImg2 = aNameSave + "." + StdPostfix(aOutImg2);
    aNameSave += ".txt";
    //cout<<aNameSave<<endl;
    if(FallInBox(aPCornerR, Pt2dr(0,0), ImgSzR) == true)
    {
        FILE * fpOutput = fopen((aNameSave).c_str(), "w");
        for(int i=0; i<4; i++)
        {
            fprintf(fpOutput, "%lf %lf %lf %lf\n", aPCornerPatch[i].x, aPCornerPatch[i].y, aPCornerR[i].x, aPCornerR[i].y);
        }
        fclose(fpOutput);

        std::string aComBaseResample = MMBinFile(MM3DStr) + "TestLib OneReechFromAscii ";
        std::string aComResampleSndImg = aComBaseResample + aOutImg2  + " " + aNameSave + " Out="+aSubImg2 + " Show=true";
        cout<<aComResampleSndImg<<endl;
        System(aComResampleSndImg);

        //std::string aOutDir = "./Tmp_Patches";
        std::string aMvTxt = "mv "+aNameSave + " "+aOutDirTxt+aNameSave;
        std::string aMvTif = "mv "+aSubImg2 + " "+aOutDirImg+aSubImg2;
        System(aMvTxt);
        System(aMvTif);
    }
    else
    {
        if(bPrint)
            cout<<aNameSave<<" out of border, hence the current patch is not saved"<<endl;
    }
}

void MakeTrainingData(std::string aDir,std::string aImg1, std::string aImg2, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR, std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, cTransform3DHelmert aTrans3DHL, cTransform3DHelmert aTrans3DHR, bool bPrint, Pt2dr aPatchSz, Pt2di seed)
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
    cGet3Dcoor a3DL(aNameOriL);
    cDSMInfo aDSMInfoL = a3DL.SetDSMInfo(aDSMFileL, aDSMDirL);
    /*TIm2D<float,double> aTImProfPxL(a3DL.GetDSMSz(aDSMFileL, aDSMDirL));
    bool bDSML = false;
    if(aDSMDirL.length() > 0)
    {
        bDSML = true;
        aTImProfPxL = a3DL.SetDSMInfo(aDSMFileL, aDSMDirL);
    }*/
    cGet3Dcoor a3DR(aNameOriR);
    cDSMInfo aDSMInfoR = a3DR.SetDSMInfo(aDSMFileR, aDSMDirR);
    /*TIm2D<float,double> aTImProfPxR(a3DR.GetDSMSz(aDSMFileR, aDSMDirR));
    bool bDSMR = false;
    if(aDSMDirR.length() > 0)
    {
        bDSMR = true;
        aTImProfPxR = a3DR.SetDSMInfo(aDSMFileR, aDSMDirR);
    }*/

    Pt2dr aPCornerL[4];
    Pt2dr origin = Pt2dr(0, 0);
    aPCornerL[0] = origin;
    aPCornerL[1] = Pt2dr(origin.x+ImgSzL.x, origin.y);
    aPCornerL[2] = Pt2dr(origin.x+ImgSzL.x, origin.y+ImgSzL.y);
    aPCornerL[3] = Pt2dr(origin.x, origin.y+ImgSzL.y);

    Pt2dr aPLPredinR[4];
    for(int i=0; i<4; i++)
    {
        Pt2dr aPL = aPCornerL[i];
        Pt3dr aPTer1;
        /*if(bDSML == true)
        {*/
            bool bPreciseL;
            aPTer1 = a3DL.Get3Dcoor(aPL, aDSMInfoL, bPreciseL, bPrint);//, a3DL.GetGSD());
        /*}
        else
        {
            aPTer1 = a3DL.GetRough3Dcoor(aPL);
        }*/

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
    CropPatch(aPtOriR, aPatchSz, aImg2, aOutDirImg, m, n);

    Pt2dr aPatchCornerR[4];
    origin = aPtOriR;
    aPatchCornerR[0] = origin;
    aPatchCornerR[1] = Pt2dr(origin.x+aPatchSz.x, origin.y);
    aPatchCornerR[2] = Pt2dr(origin.x+aPatchSz.x, origin.y+aPatchSz.y);
    aPatchCornerR[3] = Pt2dr(origin.x, origin.y+aPatchSz.y);
    GetDSMPatch(aImg2, aDSMDirR, aDSMFileR, aPatchCornerR, aPatchSz, a3DR, aDSMInfoR, bPrint, aOutDirDSM, aOutDirTxt);

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
        CropPatch(aPtOriL, aPatchSz, aImg1, aOutDirImg, m, n);

        Pt2dr aPatchCornerL[4];
        origin = aPtOriL;
        aPatchCornerL[0] = origin;
        aPatchCornerL[1] = Pt2dr(origin.x+aPatchSz.x, origin.y);
        aPatchCornerL[2] = Pt2dr(origin.x+aPatchSz.x, origin.y+aPatchSz.y);
        aPatchCornerL[3] = Pt2dr(origin.x, origin.y+aPatchSz.y);
        GetDSMPatch(aImg1, aDSMDirL, aDSMFileL, aPatchCornerL, aPatchSz, a3DL, aDSMInfoL, bPrint, aOutDirDSM, aOutDirTxt);
        //RotateImgBy90Deg1("Tmp_Patches", aDSML, aDSML+"_90.tif");
        //RotateImgBy90Deg1("./", aImg1, aImg1+"_90.tif");
    }

    //roughly aligned patches
    if(true)
    {
        int aType = eTIGB_Unknown;
        cBasicGeomCap3D * aCamL = cBasicGeomCap3D::StdGetFromFile(aNameOriL,aType);
        Pt2dr aPatchCornerL[4];
        GetSecondaryPatch(aPatchCornerR, aPatchCornerL, ImgSzR, aPatchSz, aTrans3DHR, aCamL, a3DR, aDSMInfoR, aImg1, aImg2, m, n, bPrint, aOutDirImg, aOutDirTxt);
        for(int k=0; k<4; k++)
            printf("%dth: CornerL: [%.2lf\t%.2lf]\n", k, aPatchCornerL[k].x, aPatchCornerL[k].y);
        GetDSMPatch(aImg1, aDSMDirL, aDSMFileL, aPatchCornerL, aPatchSz, a3DL, aDSMInfoL, bPrint, aOutDirDSM, aOutDirTxt);
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
                << EAM(seed, "CenterMatch", true, "the coordinate in master image which will be used as center for cropping patch (for debugging), Def=[-1, -1]")
                //<< EAM(bPrint, "Print", false, "Print corner coordinate, Def=false")
     );
     StdCorrecNameOrient(aOri1,"./",true);
     StdCorrecNameOrient(aOri2,"./",true);

     cTransform3DHelmert aTrans3DHL(aPara3DHL);
     cTransform3DHelmert aTrans3DHR(aPara3DHR);

     MakeTrainingData( aCAS3D.mDir, aImg1,  aImg2, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR,  aOri1, aOri2, aCAS3D.mICNM, aTrans3DHL, aTrans3DHR, aCAS3D.mPrint, aPatchSz, seed);

   return EXIT_SUCCESS;
}

