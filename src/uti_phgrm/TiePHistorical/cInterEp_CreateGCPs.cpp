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

void Get3DCoorFromDSM(std::vector<Pt2dr> vPt2D, std::vector<Pt3dr> & vPt3D, std::string aDSMDir, std::string aDSMFile)
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

    for(unsigned int i=0; i<vPt2D.size(); i++)
    {
        Pt2di aPt = Pt2di(vPt2D[i].x, vPt2D[i].y);
        double dX, dY;
        dX = aPt.x*aResolPlani.x + aOriPlani.x;
        dY = aPt.y*aResolPlani.y + aOriPlani.y;

        double dZ =  aTImProfPx.get(aPt);

        vPt3D.push_back(Pt3dr(dX, dY, dZ));
    }
}

void Save3DXml(std::vector<Pt3dr> vPt3D, std::string aOutXml)
{
    cDicoAppuisFlottant aDAFout;

    for(unsigned int i=0; i<vPt3D.size(); i++)
    {
        cOneAppuisDAF anAp;

        anAp.Pt() = vPt3D[i];
        anAp.NamePt() = std::to_string(i);
        anAp.Incertitude() = Pt3dr(1,1,1);
        aDAFout.OneAppuisDAF().push_back(anAp);
    }

    MakeFileXML(aDAFout, aOutXml);
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

/*
void GetIm2DCoorFromDSM(std::vector<Pt2dr>& vPt2D, std::vector<Pt3dr> vPt3D, std::string aDSMDirL, std::string aDSMFileL)
{

}
*/

bool GetBoundingBox(std::string aRGBImgDir, std::string aImg1, cBasicGeomCap3D * aCamL, Pt3dr& minPt, Pt3dr& maxPt)
{
    if (ELISE_fp::exist_file(aRGBImgDir+"/"+aImg1) == false)
    {
        cout<<aRGBImgDir+"/"+aImg1<<" didn't exist, hence skipped"<<endl;
        return false;
    }

    Tiff_Im aRGBIm1((aRGBImgDir+"/"+aImg1).c_str());
    Pt2di aImgSz = aRGBIm1.sz();

    Pt2dr aPCorner[4];
    Pt2dr origin = Pt2dr(0, 0);
    aPCorner[0] = origin;
    aPCorner[1] = Pt2dr(origin.x+aImgSz.x, origin.y);
    aPCorner[2] = Pt2dr(origin.x+aImgSz.x, origin.y+aImgSz.y);
    aPCorner[3] = Pt2dr(origin.x, origin.y+aImgSz.y);

    //double prof_d = aCamL->GetVeryRoughInterProf();
    //prof_d = 11.9117;
    //double prof_d = aCamL->GetProfondeur();
    double dZ = aCamL->GetAltiSol();
    //cout<<"dZ: "<<dZ<<endl;

    Pt3dr ptTerrCorner[4];
    for(int i=0; i<4; i++)
    {
        Pt2dr aP1 = aPCorner[i];
        //ptTerrCorner[i] = aCamL->ImEtProf2Terrain(aP1, prof_d);
        ptTerrCorner[i] = aCamL->ImEtZ2Terrain(aP1, dZ);
    }

    minPt = ptTerrCorner[0];
    maxPt = ptTerrCorner[0];
    for(int i=0; i<4; i++){
        Pt3dr ptCur = ptTerrCorner[i];
        //cout<<i<<": "<<ptCur.x<<"; "<<ptCur.y<<"; "<<ptCur.z<<endl;
        if(minPt.x > ptCur.x)
            minPt.x = ptCur.x;
        if(maxPt.x < ptCur.x)
            maxPt.x = ptCur.x;

        if(minPt.y > ptCur.y)
            minPt.y = ptCur.y;
        if(maxPt.y < ptCur.y)
            maxPt.y = ptCur.y;

        if(minPt.z > ptCur.z)
            minPt.z = ptCur.z;
        if(maxPt.z < ptCur.z)
            maxPt.z = ptCur.z;
    }
    return true;
}

void Get2DCoor(std::string aRGBImgDir, std::vector<string> vImgList1, std::vector<Pt3dr> vPt3DL, std::string aOri1, cInterfChantierNameManipulateur * aICNM, std::string aOut2DXml)
{
    StdCorrecNameOrient(aOri1,"./",true);

    //std::string aKeyOri1 = "NKS-Assoc-Im2Orient@-" + aOri1;

    cSetOfMesureAppuisFlottants aSOMAFout;
    for(unsigned int i=0; i<vImgList1.size(); i++)
    {
        std::string aImg1 = vImgList1[i];
        //cout<<aKeyOri1<<endl;
        //std::string aIm1OriFile = aICNM->Assoc1To1(aKeyOri1,aImg1,true);
        std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1); //aICNM->Assoc1To1(aKeyOri1,aImg1,true);
        //cout<<aIm1OriFile<<endl;

        int aType = eTIGB_Unknown;
        cBasicGeomCap3D * aCamL = cBasicGeomCap3D::StdGetFromFile(aIm1OriFile,aType);

        Pt3dr minPt, maxPt;
        GetBoundingBox(aRGBImgDir, aImg1, aCamL, minPt, maxPt);

        cMesureAppuiFlottant1Im aMAF;
        aMAF.NameIm() = aImg1;
        for(unsigned int j=0; j<vPt3DL.size(); j++)
        {
            Pt3dr ptCur = vPt3DL[j];
            //if current 3d point is out of the border of the current image, skip
            //because sometimes a 3d point that is out of border will get wrong 2D point from command XYZ2Im
            if(ptCur.x<minPt.x || ptCur.y<minPt.y || ptCur.x>maxPt.x || ptCur.y>maxPt.y)
                continue;

            Pt2dr aPproj = aCamL->Ter2Capteur(ptCur);
/*
            if(aImg1 == "OIS-Reech_IGNF_PVA_1-0__1970__C3544-0221_1970_CDP6452_1457.tif" || aImg1 == "OIS-Reech_IGNF_PVA_1-0__1970__C3544-0221_1970_CDP6452_1405.tif")
                printf("%s  %lf %lf\n", aImg1.c_str(), aPproj.x, aPproj.y);
*/
            cOneMesureAF1I anOM;
            anOM.NamePt() = std::to_string(j);
            anOM.PtIm() = aPproj;
            aMAF.OneMesureAF1I().push_back(anOM);
        }
        aSOMAFout.MesureAppuiFlottant1Im().push_back(aMAF);
        /*
        std::string acmmd = aComBase + aIm1OriFile + " "+ aRGBImgDir+"/"+aOut3DTxt1 + " "+aRGBImgDir+"/"+aOut2DXml1;
        cout<<acmmd<<endl;
        aLComXYZ2Im.push_back(acmmd);
        */
    }
    //cEl_GPAO::DoComInSerie(aLComXYZ2Im);
    MakeFileXML(aSOMAFout, aOut2DXml);
}

void CreateGCPs(std::string aDSMGrayImgDir, std::string aRGBImgDir, std::string aDSMGrayImg1, std::string aDSMGrayImg2, std::string aImgList1, std::string aImgList2, std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, std::string aDSMDirL, std::string aDSMDirR, std::string aDSMFileL, std::string aDSMFileR, std::string aOut2DXml1, std::string aOut2DXml2, std::string aOut3DXml1, std::string aOut3DXml2, std::string aCreateGCPsInSH)
{
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

    int nPtNum = 0;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       vPt2DL.push_back(p1);
       vPt2DR.push_back(p2);
       nPtNum++;
    }
    cout<<"Correspondences number: "<<nPtNum<<endl;

    Get3DCoorFromDSM(vPt2DL, vPt3DL, aDSMDirL, aDSMFileL);
    Save3DXml(vPt3DL, aRGBImgDir+"/"+aOut3DXml1);

    Get3DCoorFromDSM(vPt2DR, vPt3DR, aDSMDirR, aDSMFileR);
    Save3DXml(vPt3DR, aRGBImgDir+"/"+aOut3DXml2);

    //std::string aOut3DTxt1 = aRGBImgDir+"/"+StdPrefix(aOut3DXml1)+".txt";
    //Save3DTxt(vPt3DL, aOut3DTxt1);
    //std::string aOut3DTxt2 = aRGBImgDir+"/"+StdPrefix(aOut3DXml2)+".txt";
    //Save3DTxt(vPt3DR, aOut3DTxt2);

    std::vector<string> vImgList1;
    std::vector<string> vImgList2;

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

    Get2DCoor(aRGBImgDir, vImgList1, vPt3DL, aOri1, aICNM,  aOut2DXml1);
    Get2DCoor(aRGBImgDir, vImgList2, vPt3DR, aOri2, aICNM,  aOut2DXml2);
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

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDSMGrayImgDir,"The directory of gray image of DSM")
                    << EAMC(aDSMGrayImg1,"The gray image of DSM of epoch1")
                    << EAMC(aDSMGrayImg2,"The gray image of DSM of epoch2")
                    << EAMC(aRGBImgDir,"The directory of RGB image")
                    << EAMC(aImgList1,"ImgList1: The list that contains all the RGB images of epoch1")
                    << EAMC(aImgList2,"ImgList2: The list that contains all the RGB images of epoch2")
               << EAMC(aOri1,"Orientation of images in epoch1")
               << EAMC(aOri2,"Orientation of images in epoch2")
               << EAMC(aDSMDirL,"DSM direcotry of epoch1")
               << EAMC(aDSMDirR,"DSM direcotry of epoch2"),
        LArgMain()
                    //<< aCAS3D.ArgBasic()
                    << aCAS3D.ArgCreateGCPs()
                    << EAM(aDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
                    << EAM(aDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")

    );

   CreateGCPs( aDSMGrayImgDir, aRGBImgDir, aDSMGrayImg1, aDSMGrayImg2, aImgList1, aImgList2, aOri1, aOri2, aCAS3D.mICNM, aDSMDirL, aDSMDirR, aDSMFileL, aDSMFileR, aCAS3D.mOut2DXml1, aCAS3D.mOut2DXml2, aCAS3D.mOut3DXml1, aCAS3D.mOut3DXml2, aCAS3D.mCreateGCPsInSH);

   return EXIT_SUCCESS;
}
