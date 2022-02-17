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

Ce logiciel est un programme informatique servant à la aise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, aodifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de aodification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les aêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la aodification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe à
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
à l'utiliser et l'exploiter dans les aêmes conditions de sécurité.

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
aooter-MicMac-eLiSe-25/06/2007*/

void DSM_Equalization(std::string aImName, std::string aDSMDir, std::string aDSMFile, std::string aOutImg, double dSTDRange)
{
    bool bMasq = false;
    std::string aOutDir = "";
    std::string aMasqName = "";

    Pt2di aDSMSz = Pt2di(0,0);

    if(aImName.length() == 0){
        aOutDir = aDSMDir;
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
    }
    else{
        Tiff_Im aIm(aImName.c_str());
        aDSMSz = aIm.sz();
    }

    TIm2D<float,double> aTImMasq(aDSMSz);
    if (ELISE_fp::exist_file(aMasqName) == true){
        Tiff_Im aImMasqTif(aMasqName.c_str());
        ELISE_COPY
        (
        aTImMasq.all_pts(),
        aImMasqTif.in(),
        aTImMasq.out()
        );
        bMasq = true;
    }

    Tiff_Im aImDSMTif(aImName.c_str());
    TIm2D<float,double> aTImDSM(aDSMSz);
    ELISE_COPY
    (
    aTImDSM.all_pts(),
    aImDSMTif.in(),
    aTImDSM.out()
    );

    //cout<<aTImMasq.get(Pt2di(0, 0))<<",,,,,,,"<<aTImMasq.get(Pt2di(1000, 800))<<"\n";

    int nValidPxNum = 0;

    int i, j;
    double dMean = 0;
    double dMax = -99999;
    double dMin = 99999;
    for(i=0; i<aDSMSz.x; i++)
    {
        for(j=0; j<aDSMSz.y; j++)
        {
            int nVal =  1;
            if(bMasq == true)
                nVal = aTImMasq.get(Pt2di(i, j));
            if(nVal > 0)
            {
                double dZ =  aTImDSM.get(Pt2di(i, j));
                dMean += dZ;
                nValidPxNum++;

                if(dZ > dMax)
                    dMax = dZ;
                if(dZ < dMin)
                    dMin = dZ;
            }
        }
    }
    dMean /= nValidPxNum;

    double dSTD = 0;
    for(i=0; i<aDSMSz.x; i++)
    {
        for(j=0; j<aDSMSz.y; j++)
        {
            int nVal =  1;
            if(bMasq == true)
                nVal = aTImMasq.get(Pt2di(i, j));
            if(nVal > 0)
            {
                double dZ =  aTImDSM.get(Pt2di(i, j));
                dSTD += pow(dZ-dMean, 2);
            }
        }
    }
    dSTD = pow(dSTD/nValidPxNum, 0.5);

    cout<<"dMean: "<<dMean<<"; dSTD: "<<dSTD<<endl;
    cout<<"dMax: "<<dMax<<"; dMin: "<<dMin<<endl;

    double aMaxAlti = dMean+dSTD*dSTDRange;
    double aMinAlti = dMean-dSTD*dSTDRange;

    double dScale = 255/(aMaxAlti-aMinAlti);
    double dTranslation = aMinAlti;

    printf("dSTDRange: %.2lf, dSTD*dSTDRange: %.2lf\n", dSTDRange, dSTD*dSTDRange);
    printf("aMaxAlti(dMean+dSTD*dSTDRange): %.2lf, aMinAlti(dMean-dSTD*dSTDRange): %.2lf, dScale(255/(aMaxAlti-aMinAlti)): %.2lf, dTranslation(aMinAlti): %.2lf\n", aMaxAlti, aMinAlti, dScale, dTranslation);

    if(aOutImg.length()==0)
        aOutImg = StdPrefix(aImName) + "_gray.tif";
    else
        aOutImg = aOutDir + aOutImg;
    //cout<<aOutImg<<endl;

    //cout<<dScale<<",,,"<<dTranslation<<endl;
    //cout<<aOutImg<<endl;

    std::string aSubStr; // = " + (" + std::to_string(dTranslation);
//    std::string aSubStr = " - " + std::to_string(dTranslation);
    if(dTranslation < 0)
        aSubStr = " + " + std::to_string(fabs(dTranslation));
    else
        aSubStr = " + (-" + std::to_string(dTranslation) + ")";
    std::string aComNikrup = MMBinFile(MM3DStr) + "Nikrup \"* "  + std::to_string(dScale) + aSubStr + " " + aImName + "\" " + aOutImg;
    std::string aComTo8Bits = MMBinFile(MM3DStr) + "to8Bits " + aOutImg + " UseSigne=false";

    std::string aComRename = "mv " + StdPrefix(aOutImg) + "_8Bits." + StdPostfix(aOutImg) + " " + aOutImg;

    cout<<aComNikrup<<"\n"<<aComTo8Bits<<"\n"<<aComRename<<endl;
    System(aComNikrup);
    System(aComTo8Bits);
    System(aComRename);

    cout<<"xdg-open "<<aOutImg<<endl;
}

int DSM_Equalization_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aDSMDir;
   std::string aDSMFile = "MMLastNuage.xml";
   std::string aOutImg;
   std::string aInImg = "";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDSMDir, "DSM directory"),
        LArgMain()
                    //<< aCAS3D.ArgBasic()
                    << aCAS3D.ArgDSM_Equalization()
                    << EAM(aInImg, "InImg", true, "File name of input image to be equalized (if this parameter is set, DSM direcotry and DSMFile will be ignored), Def=none")
                    << EAM(aDSMFile, "DSMFile", true, "The xml file that recorded the structure information of the DSM, Def=MMLastNuage.xml")
                    << EAM(aOutImg, "OutImg", true, "Output image name, Def='input'_gray.tif")

    );

   DSM_Equalization(aInImg, aDSMDir, aDSMFile, aOutImg, aCAS3D.mSTDRange);

   return EXIT_SUCCESS;
}
