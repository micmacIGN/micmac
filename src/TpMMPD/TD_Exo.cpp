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

#include "StdAfx.h"
#include "TpPPMD.h"

// mm3d ElDcraw -t 0  -d -T -4   IMG_1364-CL1.CR2 

/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/


/*
   Par exemple :

       mm3d TestLib TD_Test Orientation-IMG_0016.CR2.xml AppuisTest-IMG_0016.CR2.xml
*/

int TD_Exo0(int argc,char ** argv)
{
    std::string aNameIm; // Chaine de caractere pour nom du fichier

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name of image"),
        LArgMain()  
    );

    // Maintenant aNameIm contient le premier argument de la ligne de commande

    // Cree une image en ram a partir du fichier
    cTD_Im aIm = cTD_Im::FromString(aNameIm);

    // Affiche la taille de l'image
    std::cout << "Bonjour , NameIm=" << aNameIm  
              << " Sz=" <<  aIm.Sz().x 
              << " : " <<  aIm.Sz().y << "\n";

    // Cree une image vide de la meme taille
    cTD_Im aImRes(aIm.Sz().x,aIm.Sz().y);

    double aVMax = (1<<16) -1; // Valeur Max sur 16 Bits
    // Parcours tous les pixels
    for (int aX=0 ; aX<aIm.Sz().x ; aX++)
    {
        for (int aY=0 ; aY<aIm.Sz().y ; aY++)
        {
            double aVal = aIm.GetVal(aX,aY); // Recupere la valeur
            aImRes.SetVal(aX,aY,aVMax-aVal); // L'inverse
        }
    }
    // Sauvegarde dans Neg
    aImRes.Save("Neg.tif");

    return EXIT_SUCCESS;
}

int TD_Exo1(int argc,char ** argv)
{
    std::string aNameIm; // Chaine de caractere pour nom du fichier

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name of image"),
        LArgMain()  
    );

    // Maintenant aNameIm contient le premier argument de la ligne de commande

    // Cree une image en ram a partir du fichier
    cTD_Im aIm = cTD_Im::FromString(aNameIm);

    // Affiche la taille de l'image
    std::cout << "Bonjour , NameIm=" << aNameIm  
              << " Sz=" <<  aIm.Sz().x 
              << " : " <<  aIm.Sz().y << "\n";

    int TX2 = aIm.Sz().x / 2;
    int TY2 = aIm.Sz().y / 2;

    // Cree une image vide de la meme taille
    cTD_Im aImR(TX2,TY2);
    cTD_Im aImV(TX2,TY2);
    cTD_Im aImB(TX2,TY2);

    // Parcours tous les pixels
    for (int aX=0 ; aX<TX2; aX++)
    {
        for (int aY=0 ; aY<TY2 ; aY++)
        {
            aImR.SetVal(aX,aY,aIm.GetVal(2*aX,2*aY));
            aImV.SetVal(aX,aY,aIm.GetVal(2*aX+1,2*aY));
            aImB.SetVal(aX,aY,aIm.GetVal(2*aX+1,2*aY+1));
        }
    }
    // Sauvegarde dans Neg
    aImR.SaveRGB("RGB.tif",aImV,aImB);

    aImR.Save("R.tif");
    aImV.Save("V.tif");
    aImB.Save("B.tif");
    std::cout << "Bonjour monde : Exo1\n";
    return EXIT_SUCCESS;

    // double x,y;
    // aIm.GetVal(Pt2dr(x,y));
}

float DiffVignette(const cTD_Im & aIm1,const cTD_Im & aIm2,double aRatio,Pt2di aC1,Pt2di aC2,int aSzW)
{
    float aSomDif=0;
    float aSomI1 = 0;

    for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)  // Parcour la vignette en x
    {
        for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)  // Parcour la vignette en y
        {
            Pt2di aPIm1 = aC1 + Pt2di(aDx,aDy);  // pt de la vignette dans I1
            Pt2di aPIm2 = aC2 + Pt2di(aDx,aDy);
            if (aIm1.Ok(aPIm1.x,aPIm1.y) && aIm2.Ok(aPIm2.x,aPIm2.y)) // Si les 2 pts sont ds l'image
            {
                aSomI1 += aIm1.GetVal(aPIm1);
                aSomDif += std::abs(aIm1.GetVal(aPIm1) - aRatio*aIm2.GetVal(aPIm2));
            }
        }
    }
    if (aSomI1 !=0) 
       return  aSomDif/aSomI1 ; 
    else
       return 10.0;
}

int TD_Exo2(int argc,char ** argv)
{
    std::string aNameIm1; // Chaine de caractere pour nom du fichier 1
    std::string aNameIm2; // Chaine de caractere pour nom du fichier 2
    int         aSzW;
    double      aRatio;
    Pt2di       aDec;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of image")
                    << EAMC(aNameIm2,"Name of image")
                    << EAMC(aRatio,"Ratio I1/I2")
                    << EAMC(aSzW,"Size of window")
                    << EAMC(aDec,"Decalage"),
        LArgMain()  
    );

    // Cree une image en ram a partir du fichier
    cTD_Im aIm1 = cTD_Im::FromString(aNameIm1);
    cTD_Im aIm2 = cTD_Im::FromString(aNameIm2);
    Pt2di aSz = aIm1.Sz();

    cTD_Im aImDif(aSz.x,aSz.y);  // Image resultante des differences

    for (int aX=0 ; aX<aSz.x ;aX++) // parcours toute l'image
    {
        for (int aY=0 ; aY<aSz.y ;aY++)
        {
             Pt2di aP(aX,aY);  // Pt dans l'image 1
             float aDif = DiffVignette(aIm1,aIm2,aRatio,aP,aP+aDec,aSzW);
             aImDif.SetVal(aX,aY,aDif);
        }
    }
    // Sauve avec un nom qui depend du decalage
    aImDif.Save("Dif_"+ ToString(aDec.x) + "_" + ToString(aDec.y) + ".tif");

    return EXIT_SUCCESS;
}

int TD_Exo3(int argc,char ** argv)
{
    std::string aNameIm1; // Chaine de caractere pour nom du fichier 1
    std::string aNameIm2; // Chaine de caractere pour nom du fichier 2
    int         aSzW;
    double      aRatio;
    int         aDecMax;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of image")
                    << EAMC(aNameIm2,"Name of image")
                    << EAMC(aRatio,"Ratio I1/I2")
                    << EAMC(aSzW,"Size of window")
                    << EAMC(aDecMax,"Decalage"),
        LArgMain()  
    );

    // Cree une image en ram a partir du fichier
    cTD_Im aIm1 = cTD_Im::FromString(aNameIm1);
    cTD_Im aIm2 = cTD_Im::FromString(aNameIm2);
    Pt2di aSz = aIm1.Sz();

    cTD_Im aImDecX(aSz.x,aSz.y);  // Image resultante des differences
    cTD_Im aImDecY(aSz.x,aSz.y);  // Image resultante des differences

    for (int aX=0 ; aX<aSz.x ;aX++) // parcourt toute l'image
    {
        for (int aY=0 ; aY<aSz.y ;aY++)
        {
             Pt2di aP(aX,aY);  // Pt dans l'image 1
             float aDifMin = 1e10;
             int aDxMin=0,aDyMin=0;
             for (int aDecX=-aDecMax ; aDecX<=aDecMax ; aDecX++)  // Parcourt tout les decalage
             {
                 for (int aDecY=-aDecMax ; aDecY<=aDecMax ; aDecY++)
                 {
                      float aDif = DiffVignette(aIm1,aIm2,aRatio,aP,aP+Pt2di(aDecX,aDecY),aSzW);
                      if (aDif < aDifMin) // Si la dif est meilleure
                      {   // Sauve dif et decalage
                          aDifMin = aDif;
                          aDxMin = aDecX;
                          aDyMin = aDecY;
                      }
                 }
             }
             aImDecX.SetVal(aX,aY,aDxMin);
             aImDecY.SetVal(aX,aY,aDyMin);
        }
    }
    // Sauve avec un nom qui depend du decalage
    aImDecX.Save("DecX.tif");
    aImDecY.Save("DecY.tif");

    return EXIT_SUCCESS;
}



int TD_Exo4(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo5(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo6(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo7(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo8(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

int TD_Exo9(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}



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
Footer-MicMac-eLiSe-25/06/2007*/
