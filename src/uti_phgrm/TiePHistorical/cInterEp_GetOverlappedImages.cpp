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




bool IsOverlapped(std::string aImg1, std::string aImg2, std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, cTransform3DHelmert aTrans3DHL, bool bPrint)
{
    if (ELISE_fp::exist_file(aImg1) == false || ELISE_fp::exist_file(aImg2) == false)
    {
        cout<<aImg1<<" or "<<aImg2<<" didn't exist, hence skipped"<<endl;
        return false;
    }


    //Tiff_Im aRGBIm1(aImg1.c_str());
    Tiff_Im aRGBIm1 = Tiff_Im::StdConvGen((aImg1).c_str(), -1, true ,true);
    Pt2di ImgSzL = aRGBIm1.sz();
    //Tiff_Im aRGBIm2(aImg2.c_str());
    Tiff_Im aRGBIm2 = Tiff_Im::StdConvGen((aImg2).c_str(), -1, true ,true);
    Pt2di ImgSzR = aRGBIm2.sz();

    //cout<<"Left img size: "<<ImgSzL.x<<", "<<ImgSzL.y<<endl;
    //cout<<"Right img size: "<<ImgSzR.x<<", "<<ImgSzR.y<<endl;

    Pt2dr origin = Pt2dr(0,0);

    Pt2dr aPCornerL[4];
    aPCornerL[0] = origin;
    aPCornerL[1] = Pt2dr(origin.x+ImgSzL.x, origin.y);
    aPCornerL[2] = Pt2dr(origin.x, origin.y+ImgSzL.y);
    aPCornerL[3] = Pt2dr(origin.x+ImgSzL.x, origin.y+ImgSzL.y);

    Pt2dr aPCornerR[4];
    aPCornerR[0] = origin;
    aPCornerR[1] = Pt2dr(origin.x+ImgSzR.x, origin.y);
    aPCornerR[2] = Pt2dr(origin.x, origin.y+ImgSzR.y);
    aPCornerR[3] = Pt2dr(origin.x+ImgSzR.x, origin.y+ImgSzR.y);

/*
    std::string aNameOriL = aOri +"/Orientation-"+aImg1+".xml";
    std::string aNameOriR = aOri +"/Orientation-"+aImg2+".xml";
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
*/
    int aType = eTIGB_Unknown;
    std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    std::string aIm2OriFile = aICNM->StdNameCamGenOfNames(aOri2, aImg2);
    cBasicGeomCap3D * aCamL = cBasicGeomCap3D::StdGetFromFile(aIm1OriFile,aType);
    cBasicGeomCap3D * aCamR = cBasicGeomCap3D::StdGetFromFile(aIm2OriFile,aType);
    double dZL = aCamL->GetAltiSol();
    double dZR = aCamR->GetAltiSol();
    if(bPrint == true)
    {
        cout<<"dZL: "<<dZL<<endl;
        cout<<"dZR: "<<dZR<<endl;
        /*
        Pt3dr ptt = Pt3dr(736136,6259014,138);
        Pt2dr aP1 = aCamR->Ter2Capteur(ptt);
        cout<<" pred: "<<aP1.x<<"; "<<aP1.y<<endl;

        ptt = Pt3dr(726796,6269100,138);
        aP1 = aCamR->Ter2Capteur(ptt);
        cout<<" pred: "<<aP1.x<<"; "<<aP1.y<<endl;

        ptt = Pt3dr(744136,6249560,138);
        aP1 = aCamR->Ter2Capteur(ptt);
        cout<<" pred: "<<aP1.x<<"; "<<aP1.y<<endl;
        */
    }

    Pt3dr aCornerTerrL[4];
    Pt3dr aCornerTerrR[4];
    for(int i=0; i<4; i++)
    {
        Pt3dr Pt_H_d;
        Pt2dr aP1;
        aP1 = aPCornerL[i];
        Pt_H_d = aCamL->ImEtZ2Terrain(aP1, dZL);
        aCornerTerrL[i] = aTrans3DHL.Transform3Dcoor(Pt_H_d);
        if(bPrint == true)
        {
            cout<<"----------------"<<endl;
            cout<<i<<" aP1: "<<aP1.x<<"; "<<aP1.y<<endl;
            cout<<i<<" aCornerTerrL: "<<Pt_H_d.x<<"; "<<Pt_H_d.y<<"; "<<Pt_H_d.z<<endl;
            //aP1 = aCamL->Ter2Capteur(Pt_H_d);
            //cout<<i<<" aP1: "<<aP1.x<<"; "<<aP1.y<<endl;
        }

        aP1 = aPCornerR[i];
        Pt_H_d = aCamR->ImEtZ2Terrain(aP1, dZR);
        aCornerTerrR[i] = Pt_H_d;
        if(bPrint == true)
        {
            cout<<"--------"<<endl;
            cout<<i<<" aP1: "<<aP1.x<<"; "<<aP1.y<<endl;
            cout<<i<<"aCornerTerrR: "<<Pt_H_d.x<<"; "<<Pt_H_d.y<<"; "<<Pt_H_d.z<<endl;
            //aP1 = aCamR->Ter2Capteur(Pt_H_d);
            //cout<<i<<" aP1: "<<aP1.x<<"; "<<aP1.y<<endl;
        }
    }

    bool bRes = true;

    Pt3dr minPtL, maxPtL;
    GetBoundingBox(aCornerTerrL, 4, minPtL, maxPtL);
    Pt3dr minPtR, maxPtR;
    GetBoundingBox(aCornerTerrR, 4, minPtR, maxPtR);
    if(minPtL.x > maxPtR.x ||maxPtL.x < minPtR.x || minPtL.y > maxPtR.y ||maxPtL.y < minPtR.y){
        if(bPrint==true)
            printf("The 2 images have no overlapped area in Terr.\n");
        bRes = false;
    }

    if(bPrint==true)
    {
        printf("[minPtL.x, maxPtR.x]: [%.2lf, %.2lf]\n[maxPtL.x, minPtR.x]: [%.2lf, %.2lf]\n[minPtL.y, maxPtR.y]: [%.2lf, %.2lf]\n[maxPtL.y, minPtR.y]: [%.2lf, %.2lf]\n", minPtL.x, maxPtR.x, maxPtL.x, minPtR.x, minPtL.y, maxPtR.y, maxPtL.y, minPtR.y);
        printf("bRes: %d\n", bRes);
        for(int i=0; i<4; i++)
        {
            Pt3dr Pt_H_d = aCornerTerrL[i];
            printf("Left: %ith corner: %lf, %lf, %lf\n", i, Pt_H_d.x, Pt_H_d.y, Pt_H_d.z);

            Pt_H_d = aCornerTerrR[i];
            printf("Right: %ith corner: %lf, %lf, %lf\n", i, Pt_H_d.x, Pt_H_d.y, Pt_H_d.z);
        }
        printf("************%s\n", aIm1OriFile.c_str());
        printf("************%s\n", aIm2OriFile.c_str());
    }

    return bRes;
}

void GetOverlappedImages(std::string aImgList1, std::string aImgList2, std::string aOri1, std::string aOri2, std::string aDir, std::string aOut, cInterfChantierNameManipulateur * aICNM, cTransform3DHelmert aTrans3DHL, bool bPrint)
{
    std::vector<string> vImgList1;
    std::vector<string> vImgList2;
    GetImgListVec(aImgList1, vImgList1);
    GetImgListVec(aImgList2, vImgList2);
    /*
    if (ELISE_fp::exist_file(aImgList1) == false)
        printf("File %s does not exist.\n", aImgList1.c_str());
    if (ELISE_fp::exist_file(aImgList2) == false)
        printf("File %s does not exist.\n", aImgList2.c_str());

    std::string s;

    ifstream in1(aDir+aImgList1);
    printf("Images in %s:\n", aImgList1.c_str());
    while(getline(in1,s))
    {
        vImgList1.push_back(s);
        printf("%s\n", s.c_str());
    }
    //printf("%d images in %s\n", int(vImgList1.size()), aImgList1.c_str());

    ifstream in2(aDir+aImgList2);
    printf("Images in %s:\n", aImgList2.c_str());
    while(getline(in2,s))
    {
        vImgList2.push_back(s);
        printf("%s\n", s.c_str());
    }
    //printf("%d images in %s\n", int(vImgList2.size()), aImgList2.c_str());

    //std::string aKeyOri1 = "NKS-Assoc-Im2Orient@-" + aOri1;
    //std::string aKeyOri2 = "NKS-Assoc-Im2Orient@-" + aOri2;
    */

    unsigned int m, n;
    cSauvegardeNamedRel aRel;
    int nOverlappedPair = 0;
    for(m=0; m<vImgList1.size(); m++)
    {
        std::string aImg1 = vImgList1[m];
        //std::string aIm1OriFile = aICNM->Assoc1To1(aKeyOri1,aImg1,true);
        for(n=0; n<vImgList2.size(); n++)
        {
            std::string aImg2 = vImgList2[n];
            //std::string aIm2OriFile = aICNM->Assoc1To1(aKeyOri2,aImg2,true);

            if(IsOverlapped(aImg1, aImg2, aOri1, aOri2, aICNM, aTrans3DHL, bPrint) == true)
            {
                aRel.Cple().push_back(cCpleString(aImg1, aImg2));
                nOverlappedPair++;
            }
        }
    }
    MakeFileXML(aRel,aDir+aOut);
    cout<<"Overlapped image pairs: "<<nOverlappedPair<<endl;
    cout<<"xdg-open "<<aDir+aOut<<endl;
}

int GetOverlappedImages_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aImgList1;
   std::string aImgList2;
   std::string aOri1;
   std::string aOri2;

   std::string aPara3DH = "";

   //bool bPrint = false;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()
               << EAMC(aOri1,"Orientation of master image")
               << EAMC(aOri2,"Orientation of secondary image")
               //<< EAMC(aDir,"Work directory")
               << EAMC(aImgList1,"ImgList1: RGB images in epoch1 for extracting inter-epoch correspondences (Dir+Pattern, or txt file of image list)")
               << EAMC(aImgList2,"ImgList2: RGB images in epoch2 for extracting inter-epoch correspondences (Dir+Pattern, or txt file of image list)"),
        LArgMain()
                    << aCAS3D.ArgBasic()
                    << aCAS3D.ArgGetOverlappedImages()
               << EAM(aPara3DH, "Para3DH", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master image to secondary image, Def=none")
               //<< EAM(bPrint, "Print", false, "Print corner coordinate, Def=false")

    );

   StdCorrecNameOrient(aOri1,"./",true);
   StdCorrecNameOrient(aOri2,"./",true);

   cTransform3DHelmert aTrans3DH(aPara3DH);

   std::string aOutPairXml = aCAS3D.mOutPairXml;
   if(aOutPairXml.length() == 0)
       aOutPairXml = "OverlappedImages"+RemoveOri(aOri1)+"_"+RemoveOri(aOri2)+".xml";
   printf("%s to be saved...\n", aOutPairXml.c_str());

   GetOverlappedImages(aImgList1, aImgList2, aOri1, aOri2, aCAS3D.mDir, aOutPairXml, aCAS3D.mICNM, aTrans3DH, aCAS3D.mPrint);

   return EXIT_SUCCESS;
}
