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
#include "../TpMMPD/TpPPMD.h"


/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/


/*
   Par exemple :

       mm3d TestLib TD_Test Orientation-IMG_0016.CR2.xml AppuisTest-IMG_0016.CR2.xml
*/


double ScoreScol(const  cTD_Camera & aCam,const cTD_SetAppuis & aSetGCP)
{
    double aScore = 0;
    for (int aKP=0 ; aKP<int(aSetGCP.PTer().size()) ; aKP++)
    {
         Pt3dr aPTer = aSetGCP.PTer()[aKP];
         Pt2dr aPIm  = aSetGCP.PIm()[aKP];

         Pt2dr aPProj = aCam.Ter2Image(aPTer);

         double aD = euclid (aPIm,aPProj);
         aScore +=  aD / (10+aD);
    }
    return aScore;
}



int TD_Sol1(int argc,char ** argv)
{
    std::string aNameCam,aNameAppuis;
    int aNbTest = 10000;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameCam,"Name of camera")
                    << EAMC(aNameAppuis,"Name of GCP"),
        LArgMain()  << EAM(aNbTest,"NbTest",true,"Do no stuff")
    );

    cTD_Camera aCam(aNameCam);
    cTD_SetAppuis aSetGCP(aNameAppuis);


    int aNbPts = (int)aSetGCP.PTer().size();
    int aK1Max=-1,aK2Max=-1,aK3Max=-1;
    double aSurfMax=0;
    for (int aK=0 ; aK< aNbTest ; aK++)
    {
        if ((aK%100)==0) 
           std::cout << "RESTE " << aNbTest-aK << "\n";

        int aK1 = TD_EntierAleatoire(aNbPts);
        int aK2 = TD_EntierAleatoire(aNbPts);
        int aK3 = TD_EntierAleatoire(aNbPts);
        if ((aK1!=aK2) && (aK1!=aK3) && (aK2!=aK3))
        {
             Pt2dr  aP1 = aSetGCP.PIm()[aK1];
             Pt2dr  aP2 = aSetGCP.PIm()[aK2];
             Pt2dr  aP3 = aSetGCP.PIm()[aK3];

             double aSurf = fabs((aP1-aP2) ^ (aP2-aP3));
             if (aSurf> aSurfMax)
             {
                 aSurfMax = aSurf;
                 aK1Max = aK1;
                 aK2Max = aK2;
                 aK3Max = aK3;
             }
        }
    }
    Pt2dr aPIm1 = aSetGCP.PIm()[aK1Max];
    Pt2dr aPIm2 = aSetGCP.PIm()[aK2Max];
    Pt2dr aPIm3 = aSetGCP.PIm()[aK3Max];
    Pt3dr aPTer1 = aSetGCP.PTer()[aK1Max];
    Pt3dr aPTer2 = aSetGCP.PTer()[aK2Max];
    Pt3dr aPTer3 = aSetGCP.PTer()[aK3Max];


     double aStepR3 = 1 /pow(euclid(aCam.SzCam())/2.0,3.0);
     std::cout << "STEP R3 " << aStepR3 << "\n";

     double aScoreMin = 1e20;
     double aFMin = 0;
     double aR3Min = 0;

     for (double aDF =  -100 ; aDF<=100 ; aDF +=10)
     {
          std::cout << "DF = " << aDF << "\n";
          for (double aDR =  -200 ; aDR<=200 ; aDR +=10)
          {
              cTD_Camera  aNewC  =  aCam.NewCam(aCam.Focale()+aDF,aStepR3*aDR);
              std::vector<cTD_Camera> aSols = aNewC.RelvtEspace
                                    (
                                          aPTer1, aPIm1,
                                          aPTer2, aPIm2,
                                          aPTer3, aPIm3
                                    );

              for (int aKS=0 ; aKS < int(aSols.size()) ; aKS++)
              {
                   double aScore = ScoreScol(aSols[aKS],aSetGCP);
                   if (aScore<aScoreMin)
                   {
                      aScoreMin = aScore;
                      aFMin = aSols[aKS].Focale() ; 
                      aR3Min = aSols[aKS].R3() ; 

                      std::cout << "SCORE " << aScoreMin << " " << aFMin << " " << aR3Min << "\n";
                  }
              }
          }
     }
     std::cout << "END-SCORE " << aScoreMin << " " << aFMin << " " << aR3Min << "\n";



    return 0;
}



int TD_Sol2(int argc,char ** argv)
{
    std::string aNameCam,aNameAppuis;
    int aNbTest = 10000;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameCam,"Name of camera")
                    << EAMC(aNameAppuis,"Name of GCP"),
        LArgMain()  << EAM(aNbTest,"NbTest",true,"Do no stuff")
    );

    cTD_Camera aCam(aNameCam);
    cTD_SetAppuis aSetGCP(aNameAppuis);

    int aNbPts = (int)aSetGCP.PTer().size();

    
    double aScoreMin = 1e20;
    cTD_Camera aBestCam = aCam;
    for (int aK=0 ; aK< aNbTest ; aK++)
    {
        if ((aK%100)==0) 
           std::cout << "RESTE " << aNbTest-aK << "\n";

        int aK1 = TD_EntierAleatoire(aNbPts);
        int aK2 = TD_EntierAleatoire(aNbPts);
        int aK3 = TD_EntierAleatoire(aNbPts);

        if ((aK1!=aK2) && (aK1!=aK3) && (aK2!=aK3))
        {
            std::vector<cTD_Camera> aSols = aCam.RelvtEspace
                                    (
                                          aSetGCP.PTer()[aK1], aSetGCP.PIm()[aK1],
                                          aSetGCP.PTer()[aK2], aSetGCP.PIm()[aK2],
                                          aSetGCP.PTer()[aK3], aSetGCP.PIm()[aK3]
                                    );


            for (int aKS=0 ; aKS<int(aSols.size()) ; aKS++)
            {
                  double aScore =  ScoreScol(aSols[aKS],aSetGCP);

                  if (aScore<aScoreMin)
                  {
                       aScoreMin = aScore;
                       aBestCam = aSols[aKS];
                  }
            }
        }
     }
     aBestCam.Save(StdPrefix(aNameCam) + "_Save.xml");


     return 0;
}

float Correl(const cTD_Im & aIm1,const Pt2di & aP1,const cTD_Im & aIm2,const Pt2di & aP2,int aSzW)
{
// static int aCpt = 0 ; aCpt++;
     double aS   = 0;
     double aS1  = 0;
     double aS2  = 0;
     double aS11 = 0;
     double aS22 = 0;
     double aS12 = 0;

     for (int aDx = -aSzW ; aDx<=aSzW ; aDx++)
     {
         for (int aDy = -aSzW ; aDy<=aSzW ; aDy++)
         {
              int aXIm1 = aP1.x + aDx;
              int aYIm1 = aP1.y + aDy;
              if (! aIm1.Ok(aXIm1,aYIm1)) return -2;

              int aXIm2 = aP2.x + aDx;
              int aYIm2 = aP2.y + aDy;
              if (! aIm2.Ok(aXIm2,aYIm2)) return -2;

              float aV1 = aIm1.GetVal(aXIm1,aYIm1);
              float aV2 = aIm2.GetVal(aXIm2,aYIm2);


              aS++;
              aS1 += aV1;
              aS11 += aV1 * aV1;
              aS2 += aV2;
              aS22 += aV2 * aV2;
              aS12 += aV1 * aV2;
         }
     }

     aS1 /= aS;
     aS2 /= aS;
     aS11 = aS11 / aS  - aS1 * aS1;
     aS22 = aS22 / aS  - aS2 * aS2;
     aS12 = aS12 / aS  - aS1 * aS2;

     double aVar = aS11 * aS22;
     if (aVar < 1e-5) return -2;

     return  aS12 / sqrt(aVar);
}

int TD_Sol3(int argc,char ** argv)
{
    std::string aNameIm1;
    std::string aNameIm2;
    int aIntPax=30;
    int aSzW=2;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm1,"Name of image1")
                    << EAMC(aNameIm2,"Name of image2"),
        LArgMain()  << EAM(aIntPax,"IntPax",true,"Intervale of paralaxe")
                    << EAM(aSzW,"SzW",true,"Size of Window")
    );

    cTD_Im aIm1 = cTD_Im::FromString(aNameIm1);
    Pt2di aSz = aIm1.Sz();

    cTD_Im aIm2 = cTD_Im::FromString(aNameIm2);
    cTD_Im aRes(aSz.x,aSz.y);



    for (int anY=aSzW ; anY< (aSz.y-aSzW) ; anY++)
    {
         std::cout << "Y=" << anY << "\n";
         for (int anX=0 ; anX<aSz.x ; anX++)
         {
              double aPxMax = -100;
              double aCorMax = -1;
              for (int aP=-aIntPax ; aP<=aIntPax ; aP++)
              {
                  Pt2di aP1(anX,anY);
                  Pt2di aP2(anX+aP,anY);
                  float aCor = Correl(aIm1,aP1,aIm2,aP2,aSzW);
                  if (aCor > aCorMax)
                  {
                      aCorMax = aCor;
                      aPxMax = aP;
                  }
              }
              aRes.SetVal(anX,anY,aPxMax);
         }
    }

    aRes.Save("Px.tif");

    return 0;
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
