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

Pt2dr Get3DCoorFromOrthoDSM(std::vector<Pt2dr> vPt2D, std::vector<Pt3dr> & vPt3D, std::string aDSMDir, std::string aDSMFile, std::string aOrthoDir, std::string aOrthoFile)
{
    aDSMDir += "/";
    cout<<aDSMDir + aDSMFile<<endl;
    cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
    (
    aDSMDir + aDSMFile,
    StdGetFileXMLSpec("SuperposImage.xml"),
    "XML_ParamNuage3DMaille",
    "XML_ParamNuage3DMaille"
    );

    Pt2di aDSMSz = aNuageIn.NbPixel();

    cImage_Profondeur aImProfPx = aNuageIn.Image_Profondeur().Val();
    std::string aImName = aDSMDir + aImProfPx.Image();
    Tiff_Im aImProfPxTif(aImName.c_str());

    Pt2di aSzOut = aDSMSz;
    TIm2D<float,double> aTImProfPx(aSzOut);
    ELISE_COPY
    (
    aTImProfPx.all_pts(),
    aImProfPxTif.in(),
    aTImProfPx.out()
    );

    std::string tfwFile = aOrthoDir+"/"+StdPrefix(aOrthoFile)+".tfw";
    std::vector<double> aTmp;
    ReadTfw(tfwFile, aTmp);
    Pt2dr aOrthoResolPlani = Pt2dr(aTmp[0], aTmp[3]);
    Pt2dr aOrthoOriPlani = Pt2dr(aTmp[4], aTmp[5]);

    cFileOriMnt aFOM = StdGetFromPCP(aDSMDir+StdPrefix(aImProfPx.Image())+".xml",FileOriMnt);

    Pt2dr aDSMOriPlani = aFOM.OriginePlani();
    Pt2dr aDSMResolPlani = aFOM.ResolutionPlani();


    printf("%s; aDSMSz: [%d, %d]; aOrthoOriPlani: [%.6lf, %.6lf]; aOrthoResolPlani: [%.6lf, %.6lf]\n", aImName.c_str(), aDSMSz.x, aDSMSz.y, aOrthoOriPlani.x, aOrthoOriPlani.y, aOrthoResolPlani.x, aOrthoResolPlani.y);

    for(unsigned int i=0; i<vPt2D.size(); i++)
    {
        Pt2di aPt = Pt2di(vPt2D[i].x, vPt2D[i].y);
        double dX, dY;
        dX = aPt.x*aOrthoResolPlani.x + aOrthoOriPlani.x;
        dY = aPt.y*aOrthoResolPlani.y + aOrthoOriPlani.y;

        Pt2di aPtDSM;
        aPtDSM.x = int((dX-aDSMOriPlani.x)*1.0/aDSMResolPlani.x);
        aPtDSM.y = int((dY-aDSMOriPlani.y)*1.0/aDSMResolPlani.y);

        //printf("[%d, %d], [%d, %d], [%.2lf, %.2lf]\n", aPt.x, aPt.y, aPtDSM.x, aPtDSM.y, dX, dY);

        double dZ =  aTImProfPx.get(aPtDSM);

        vPt3D.push_back(Pt3dr(dX, dY, dZ));
    }
    printf("%d tie points processed\n", int(vPt2D.size()));
    return Pt2dr(aOrthoResolPlani.x, aOrthoResolPlani.y);
}

Pt2dr Get3DCoorFromDSM(std::vector<Pt2dr> vPt2D, std::vector<Pt3dr> & vPt3D, std::string aDSMDir, std::string aDSMFile)
{
    aDSMDir += "/";
    cout<<aDSMDir + aDSMFile<<endl;
    cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
    (
    aDSMDir + aDSMFile,
    StdGetFileXMLSpec("SuperposImage.xml"),
    "XML_ParamNuage3DMaille",
    "XML_ParamNuage3DMaille"
    );

    Pt2di aDSMSz = aNuageIn.NbPixel();

    cImage_Profondeur aImProfPx = aNuageIn.Image_Profondeur().Val();
    std::string aImName = aDSMDir + aImProfPx.Image();
    Tiff_Im aImProfPxTif(aImName.c_str());

    Pt2di aSzOut = aDSMSz;
    TIm2D<float,double> aTImProfPx(aSzOut);
    ELISE_COPY
    (
    aTImProfPx.all_pts(),
    aImProfPxTif.in(),
    aTImProfPx.out()
    );

    cFileOriMnt aFOM = StdGetFromPCP(aDSMDir+StdPrefix(aImProfPx.Image())+".xml",FileOriMnt);

    Pt2dr aOriPlani = aFOM.OriginePlani();
    Pt2dr aResolPlani = aFOM.ResolutionPlani();
    printf("%s; aDSMSz: [%d, %d]; aOriPlani: [%.6lf, %.6lf]; aResolPlani: [%.6lf, %.6lf]\n", aImName.c_str(), aDSMSz.x, aDSMSz.y, aOriPlani.x, aOriPlani.y, aResolPlani.x, aResolPlani.y);

    for(unsigned int i=0; i<vPt2D.size(); i++)
    {
        Pt2di aPt = Pt2di(vPt2D[i].x, vPt2D[i].y);
        double dX, dY;
        dX = aPt.x*aResolPlani.x + aOriPlani.x;
        dY = aPt.y*aResolPlani.y + aOriPlani.y;

        //printf("[%d, %d], [%.2lf, %.2lf]\n", aPt.x, aPt.y, dX, dY);

        double dZ =  aTImProfPx.get(aPt);

        vPt3D.push_back(Pt3dr(dX, dY, dZ));
    }
    printf("%d tie points processed\n", int(vPt2D.size()));
    return Pt2dr(aResolPlani.x, aResolPlani.y);
}

void Save3DTxt(std::vector<Pt3dr> vPt3D, std::string aOutTxt)
{
    FILE * fpOutput = fopen((aOutTxt).c_str(), "w");

    for(unsigned int i=0; i<vPt3D.size(); i++)
    {
        Pt3dr aPtTerr = vPt3D[i];
        fprintf(fpOutput, "%lf %lf %lf\n", aPtTerr.x, aPtTerr.y, aPtTerr.z);
    }

    fclose(fpOutput);
}

void CreateGCPs(std::string aDSMGrayImgDir, std::string aRGBImgDir, std::string aDSMGrayImg1, std::string aDSMGrayImg2, std::string aImgList1, std::string aImgList2, std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, std::string aDSMDirL, std::string aDSMDirR, std::string aDSMFileL, std::string aDSMFileR, std::string aOut2DXml1, std::string aOut2DXml2, std::string aOut3DXml1, std::string aOut3DXml2, std::string aCreateGCPsInSH, std::string aOrthoDirL, std::string aOrthoDirR, std::string aOrthoFileL, std::string aOrthoFileR)
{
    bool bUseOrtho = false;
    if (aOrthoDirL.length()>0 && aOrthoDirR.length()>0 && ELISE_fp::exist_file(aOrthoDirL+"/"+aOrthoFileL) == true && ELISE_fp::exist_file(aOrthoDirR+"/"+aOrthoFileR) == true)
        bUseOrtho = true;

    std::string aDir_inSH = aDSMGrayImgDir + "/Homol" + aCreateGCPsInSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aDSMGrayImg1 + "/"+aDSMGrayImg2+".txt";
        if (ELISE_fp::exist_file(aNameIn) == false)
        {
            cout<<aNameIn<<"didn't exist hence skipped."<<endl;
            return;
        }
        ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    std::vector<Pt2dr> vPt2DL, vPt2DR;
    std::vector<Pt3dr> vPt3DL, vPt3DR;

    int nOriPtNum = 0;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       vPt2DL.push_back(p1);
       vPt2DR.push_back(p2);
       nOriPtNum++;
    }
    cout<<"Correspondences number: "<<nOriPtNum<<endl;


    if(bUseOrtho == true)
    {
        Get3DCoorFromOrthoDSM(vPt2DL, vPt3DL, aDSMDirL, aDSMFileL, aOrthoDirL, aOrthoFileL);
        Get3DCoorFromOrthoDSM(vPt2DR, vPt3DR, aDSMDirR, aDSMFileR, aOrthoDirR, aOrthoFileR);
    }
    else
    {
        Get3DCoorFromDSM(vPt2DL, vPt3DL, aDSMDirL, aDSMFileL);
        Get3DCoorFromDSM(vPt2DR, vPt3DR, aDSMDirR, aDSMFileR);
    }

    Save3DXml(vPt3DL, aRGBImgDir+"/"+aOut3DXml1);
    Save3DXml(vPt3DR, aRGBImgDir+"/"+aOut3DXml2);

    //std::string aOut3DTxt1 = aRGBImgDir+"/"+StdPrefix(aOut3DXml1)+".txt";
    //Save3DTxt(vPt3DL, aOut3DTxt1);
    //std::string aOut3DTxt2 = aRGBImgDir+"/"+StdPrefix(aOut3DXml2)+".txt";
    //Save3DTxt(vPt3DR, aOut3DTxt2);

    std::vector<string> vImgList1;
    std::vector<string> vImgList2;
    GetImgListVec(aImgList1, vImgList1);
    GetImgListVec(aImgList2, vImgList2);
    /*
    std::string s;
    ifstream in1(aRGBImgDir+"/"+aImgList1);
    while(getline(in1,s))
    {
        vImgList1.push_back(s);
    }

    ifstream in2(aRGBImgDir+"/"+aImgList2);
    while(getline(in2,s))
    {
        vImgList2.push_back(s);
    }
    */

    Get2DCoor(aRGBImgDir, vImgList1, vPt3DL, aOri1, aICNM,  aOut2DXml1);
    Get2DCoor(aRGBImgDir, vImgList2, vPt3DR, aOri2, aICNM,  aOut2DXml2);
    printf("xdg-open %s\n", aOut2DXml1.c_str());
    printf("xdg-open %s\n", aOut3DXml1.c_str());
    printf("xdg-open %s\n", aOut2DXml2.c_str());
    printf("xdg-open %s\n", aOut3DXml2.c_str());
}

int CreateGCPs_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aDSMGrayImgDir;

   std::string aDSMGrayImg1;
   std::string aDSMGrayImg2;

   std::string aRGBImgDir;

   std::string aImgList1;
   std::string aImgList2;

   std::string aOri1;
   std::string aOri2;

   std::string aDSMDirL;
   std::string aDSMDirR;
   std::string aDSMFileL = "MMLastNuage.xml";
   std::string aDSMFileR = "MMLastNuage.xml";

   std::string aOrthoDirL;
   std::string aOrthoDirR;
   std::string aOrthoFileL;
   std::string aOrthoFileR;

   aOrthoDirL = "";
   aOrthoDirR = "";
   aOrthoFileL = "Orthophotomosaic.tif";
   aOrthoFileR = "Orthophotomosaic.tif";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDSMGrayImgDir,"The directory of 'gray image of DSM' or orthophoto")
                    << EAMC(aDSMGrayImg1,"The 'gray image of DSM' or orthophoto of epoch1")
                    << EAMC(aDSMGrayImg2,"The 'gray image of DSM' or orthophoto of epoch2")
                    << EAMC(aRGBImgDir,"The directory of RGB image")
                    << EAMC(aImgList1,"ImgList1: All RGB images in epoch1 (Dir+Pattern, or txt file of image list)")
                    << EAMC(aImgList2,"ImgList2: All RGB images in epoch2 (Dir+Pattern, or txt file of image list)")
               << EAMC(aOri1,"Orientation of images in epoch1")
               << EAMC(aOri2,"Orientation of images in epoch2")
               << EAMC(aDSMDirL,"DSM direcotry of epoch1")
               << EAMC(aDSMDirR,"DSM direcotry of epoch2"),
        LArgMain()
                    //<< aCAS3D.ArgBasic()
                    << aCAS3D.ArgCreateGCPs()
                    << EAM(aDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
                    << EAM(aDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")
               << EAM(aOrthoDirL, "OrthoDirL", true, "Orthophoto directory of epoch1 (if this parameter is set, it means the tie points are on orthophotos instead of DSMs), Def=none")
               << EAM(aOrthoDirR, "OrthoDirR", true, "Orthophoto directory of epoch2 (if this parameter is set, it means the tie points are on orthophotos instead of DSMs), Def=none")
               << EAM(aOrthoFileL, "OrthoFileL", true, "Orthophoto file of epoch1, Def=Orthophotomosaic.tif")
               << EAM(aOrthoFileR, "OrthoFileR", true, "Orthophoto file of epoch2, Def=Orthophotomosaic.tif")

    );

   aCAS3D.CorrectXmlFileName(aCAS3D.mCreateGCPsInSH, aOri1, aOri2);

   CreateGCPs(aDSMGrayImgDir, aRGBImgDir, aDSMGrayImg1, aDSMGrayImg2, aImgList1, aImgList2, aOri1, aOri2, aCAS3D.mICNM, aDSMDirL, aDSMDirR, aDSMFileL, aDSMFileR, aCAS3D.mOut2DXml1, aCAS3D.mOut2DXml2, aCAS3D.mOut3DXml1, aCAS3D.mOut3DXml2, aCAS3D.mCreateGCPsInSH, aOrthoDirL, aOrthoDirR, aOrthoFileL, aOrthoFileR);

   return EXIT_SUCCESS;
}




/******************************InlierRatio********************************/
void InlierRatio(std::string aDSMGrayImgDir, std::string aTransFile, std::string aDSMGrayImg1, std::string aDSMGrayImg2, std::string aDSMDirL, std::string aDSMDirR, std::string aDSMFileL, std::string aDSMFileR, std::string aCreateGCPsInSH, std::string aOrthoDirL, std::string aOrthoDirR, std::string aOrthoFileL, std::string aOrthoFileR, double threshold)
{
    bool bUseOrtho = false;
    if (aOrthoDirL.length()>0 && aOrthoDirR.length()>0 && ELISE_fp::exist_file(aOrthoDirL+"/"+aOrthoFileL) == true && ELISE_fp::exist_file(aOrthoDirR+"/"+aOrthoFileR) == true)
        bUseOrtho = true;

    std::string aCom = "mm3d SEL" + BLANK + aDSMGrayImgDir + BLANK + aDSMGrayImg1 + BLANK + aDSMGrayImg2 + BLANK + "KH=NT SzW=[600,600] SH="+aCreateGCPsInSH;
    std::string aComInv = "mm3d SEL" + BLANK + aDSMGrayImgDir + BLANK + aDSMGrayImg2 + BLANK + aDSMGrayImg1 + BLANK + "KH=NT SzW=[600,600] SH="+aCreateGCPsInSH;
    printf("%s\n%s\n", aCom.c_str(), aComInv.c_str());

    std::string aDir_inSH = aDSMGrayImgDir + "/Homol" + aCreateGCPsInSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aDSMGrayImg1 + "/"+aDSMGrayImg2+".txt";
        if (ELISE_fp::exist_file(aNameIn) == false)
        {
            cout<<aNameIn<<"didn't exist hence skipped."<<endl;
            return;
        }
        ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    std::vector<Pt2dr> vPt2DL, vPt2DR;
    std::vector<Pt3dr> vPt3DL, vPt3DR;

    int nOriPtNum = 0;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       vPt2DL.push_back(p1);
       vPt2DR.push_back(p2);
       nOriPtNum++;
    }
    cout<<"Correspondences number: "<<nOriPtNum<<endl;

    Pt2dr aResol(0,0);
    if(bUseOrtho == true)
    {
        Get3DCoorFromOrthoDSM(vPt2DL, vPt3DL, aDSMDirL, aDSMFileL, aOrthoDirL, aOrthoFileL);
        aResol = Get3DCoorFromOrthoDSM(vPt2DR, vPt3DR, aDSMDirR, aDSMFileR, aOrthoDirR, aOrthoFileR);
    }
    else
    {
        Get3DCoorFromDSM(vPt2DL, vPt3DL, aDSMDirL, aDSMFileL);
        aResol = Get3DCoorFromDSM(vPt2DR, vPt3DR, aDSMDirR, aDSMFileR);
    }

    if (ELISE_fp::exist_file(aTransFile) == false){
        printf("%s not exist, hence skipped\n", aTransFile.c_str());
        return;
    }

    cXml_ParamBascRigide  *  aXBR = OptStdGetFromPCP(aTransFile,Xml_ParamBascRigide);
    cSolBasculeRig aSBR = Xml2EL(*aXBR);
    int nPtNum = vPt3DL.size();
    int nInlier = 0;

    double dAve=0;
    FILE * fpOutput = fopen((aDSMGrayImgDir+"/Eval"+aCreateGCPsInSH+".txt").c_str(), "w");
    for(int i=0; i<nPtNum; i++)
    {
        Pt3dr aP1 = vPt3DL[i];
        Pt3dr aP2 = vPt3DR[i];

        Pt3dr aP2Pred = aSBR(aP1);
        double dist = pow(pow(aP2Pred.x-aP2.x,2) + pow(aP2Pred.y-aP2.y,2) + pow(aP2Pred.z-aP2.z,2), 0.5);
        dAve += dist;

        fprintf(fpOutput, "%d %lf", i, dist);
        if(dist < threshold)
            nInlier++;
        else
            fprintf(fpOutput, "  *Outlier*");
        fprintf(fpOutput, "\n");
    }
    dAve = dAve*1.0/nPtNum;
    fprintf(fpOutput, "------------------------------------------------\n");
    fprintf(fpOutput, "Total tie point: %d;  Outlier: %d;  Inlier: %d\n", nPtNum, nPtNum-nInlier, nInlier);
    fprintf(fpOutput, "Total Ave: %lf (%.2lf pix)\n", dAve, dAve/aResol.x);
    fclose(fpOutput);
    printf("--->>>Total Pt: %d; Total inlier: %d; Inlier Ratio: %.2lf%%\n", nPtNum, nInlier, nInlier*100.0/nPtNum);
    printf("Total Ave: %lf (%.2lf pix)\n", dAve, dAve/aResol.x);
    cout<<"Tie points difference saved in:"<<endl;
    cout<<"gedit "<<aDSMGrayImgDir+"/Eval"+aCreateGCPsInSH+".txt"<<endl;
}

int InlierRatio_main(int argc,char ** argv)
{
   std::string aDSMGrayImgDir;

   std::string aDSMGrayImg1;
   std::string aDSMGrayImg2;

   //std::string aRGBImgDir;
   std::string aTransFile;

   std::string aDSMDirL;
   std::string aDSMDirR;
   std::string aDSMFileL = "MMLastNuage.xml";
   std::string aDSMFileR = "MMLastNuage.xml";

   std::string aOrthoDirL;
   std::string aOrthoDirR;
   std::string aOrthoFileL;
   std::string aOrthoFileR;

   aOrthoDirL = "";
   aOrthoDirR = "";
   aOrthoFileL = "Orthophotomosaic.tif";
   aOrthoFileR = "Orthophotomosaic.tif";

   double aThreshold = 30;
   std::string aInSH = "";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDSMGrayImgDir,"The directory of 'gray image of DSM' or orthophoto")
                    << EAMC(aDSMGrayImg1,"The 'gray image of DSM' or orthophoto of epoch1")
                    << EAMC(aDSMGrayImg2,"The 'gray image of DSM' or orthophoto of epoch2")
                    //<< EAMC(aRGBImgDir,"The directory of RGB image")
                    << EAMC(aTransFile, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master images to secondary images")
                    << EAMC(aDSMDirL,"DSM direcotry of epoch1")
                    << EAMC(aDSMDirR,"DSM direcotry of epoch2"),
        LArgMain()
                    //<< aCAS3D.ArgBasic()
//                    << aCAS3D.ArgCreateGCPs()
                   << EAM(aInSH,"InSH",true,"Input Homologue extenion for NB/NT mode, Def=none")
                    << EAM(aThreshold,"Thres",true,"Threshold in 3D to define inlier, Def=30")
                    << EAM(aDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
                    << EAM(aDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")
                    << EAM(aOrthoDirL, "OrthoDirL", true, "Orthophoto directory of epoch1 (if this parameter is set, it means the tie points are on orthophotos instead of DSMs), Def=none")
                    << EAM(aOrthoDirR, "OrthoDirR", true, "Orthophoto directory of epoch2 (if this parameter is set, it means the tie points are on orthophotos instead of DSMs), Def=none")
                    << EAM(aOrthoFileL, "OrthoFileL", true, "Orthophoto file of epoch1, Def=Orthophotomosaic.tif")
                    << EAM(aOrthoFileR, "OrthoFileR", true, "Orthophoto file of epoch2, Def=Orthophotomosaic.tif")
    );

   InlierRatio(aDSMGrayImgDir, aTransFile, aDSMGrayImg1, aDSMGrayImg2, aDSMDirL, aDSMDirR, aDSMFileL, aDSMFileR, aInSH, aOrthoDirL, aOrthoDirR, aOrthoFileL, aOrthoFileR, aThreshold);

   return EXIT_SUCCESS;
}

/******************************EvalOri********************************/

void ReadPt3D(std::string FileName, std::vector<Pt3dr>& aVPt)
{
    cDicoAppuisFlottant aDico = StdGetFromPCP(FileName,DicoAppuisFlottant);
    std::list< cOneAppuisDAF > aOneAppuisDAFList= aDico.OneAppuisDAF();

    for (std::list< cOneAppuisDAF >::iterator itP=aOneAppuisDAFList.begin(); itP != aOneAppuisDAFList.end(); itP ++)
    {
        aVPt.push_back(itP->Pt());
    }
    cout<<aVPt.size()<<" points in "<<FileName<<endl;
}

int EvalOri_main(int argc,char ** argv)
{
    std::string aImgList;
    std::string aOri;
    std::string a2DXml;
    std::string a3DXml;

    std::string aOut3DXml = "3DCoords.xml";

    double aThreshold = 30;
    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aImgList,"images (Dir+Pattern)")
                    << EAMC(aOri,"Orientation of images")
                    << EAMC(a3DXml,"Xml files of 3D obersevations of the GCPs")
                    << EAMC(a2DXml,"Xml files of 2D obersevations of the GCPs"),
         LArgMain()
                << EAM(aThreshold,"Thres",true,"Threshold in 3D to define inlier, Def=30")
                //<< EAM(aOut3DXml,"Out3DXml",true,"Output name of xml file that recorded the estimated 3D obersevations of the GCPs, Def=GCP-S3D.xml")
    );

    StdCorrecNameOrient(aOri,"./",true);


    std::string aOriTmp = aOri;
    if(aOriTmp.substr(0,4) != "Ori-")
        aOriTmp = "Ori-" + aOri;
    std::string aComm = MMBinFile(MM3DStr) + "TestLib PseudoIntersect " + aImgList + BLANK + aOriTmp + BLANK + a2DXml;// + " Out="+aOut3DXml;
    cout<<aComm<<endl;
    System(aComm);

    std::vector<Pt3dr> aVPt1, aVPt2;
    ReadPt3D(a3DXml, aVPt1);
    ReadPt3D(aOut3DXml, aVPt2);

    int nSz = aVPt1.size();
    if(int(aVPt2.size()) < nSz)
        nSz = aVPt2.size();
    Pt3dr aAve = Pt3dr(0,0,0);
    for(int i=0; i<nSz; i++){
        Pt3dr ptTmp = aVPt1[i];
        printf("PtRef: [%.5lf, %.5lf, %.5lf]; ", ptTmp.x, ptTmp.y, ptTmp.z);
        ptTmp = aVPt2[i];
        printf("PtEstimated: [%.5lf, %.5lf, %.5lf]; ", ptTmp.x, ptTmp.y, ptTmp.z);
        Pt3dr ptTmp1 = aVPt1[i];
        printf("Diff: [%.5lf, %.5lf, %.5lf]\n", fabs(ptTmp.x-ptTmp1.x), fabs(ptTmp.y-ptTmp1.y), fabs(ptTmp.z-ptTmp1.z));
        aAve.x += fabs(aVPt1[i].x-aVPt2[i].x);
        aAve.y += fabs(aVPt1[i].y-aVPt2[i].y);
        aAve.z += fabs(aVPt1[i].z-aVPt2[i].z);
    }
    printf("--------------------------\n");
    //printf("aAve: [%.5lf, %.5lf, %.5lf]\n", aAve.x, aAve.y, aAve.z);
    aAve = aAve/3;
    printf("aAve: [%.5lf, %.5lf, %.5lf]  %.5lf\n", aAve.x, aAve.y, aAve.z, (aAve.x+aAve.y+aAve.z)*1.0/3);

    return EXIT_SUCCESS;
}
