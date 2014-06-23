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

    MicMa cis an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"

/*
Parametre de Tapas :

   - calibration In : en base de donnees ou deja existantes.


*/

int EstimFlatField_main(int argc,char ** argv)
{
    std::string aFullDir,aDir,aPat;
    std::string aNameOut;
    double aResol=1.0;
    double aDilate=1.0;
    int aNbMed = 1;
    int aNbMedSsRes = 3;
    bool ByMoy = false;
    double TolMed = 0.25;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Images = Dir + Pat", eSAM_IsPatFile)
                    << EAMC(aResol,"Resolution "),
    LArgMain()  << EAM(aNbMed,"NbMed",true)
                    << EAM(aNameOut,"Out",true,"Name of result")
                    << EAM(aDilate,"SousResAdd",true)
                    << EAM(aNbMedSsRes,"NbMedSsRes",true)
                    << EAM(TolMed,"TolMed",true)
                    << EAM(ByMoy,"ByMoy",true,"Average or median (def=false")
    );

    if (!MMVisualMode)
    {
    SplitDirAndFile(aDir,aPat,aFullDir);

    if (aNameOut=="")
       aNameOut = "FlatField.tif";
    aNameOut = aDir + aNameOut;

    cTplValGesInit<std::string> aTplN;
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::StdAlloc(0,0,aDir,aTplN);

    std::list<std::string> aLName = aICNM->StdGetListOfFile(aPat);
    Pt2di aSzIm(-1,-1);
    double aNbPix=-1;
    Im2D_REAL4  aImIn(1,1);
    Im2D_REAL8  aFFRes1(1,1);
    Pt2di aSzF(-1,-1);


    int aNbIm = aLName.size();
    int aCpt = aNbIm;

    std::vector<Im2D_REAL4> aVImRed;

    // for (std::list<std::string>::const_iterator itN=aLName.begin(); itN!=aLName.end() ; itN++)
    // for (std::list<std::string>::reverse_iterator itN=aLName.rbegin(); itN!=aLName.rend() ; itN++)
     for (std::list<std::string>::const_iterator itN=aLName.begin(); itN!=aLName.end() ; itN++)
     {
         std::cout << "To Do " << aCpt << *itN << "\n";
         Tiff_Im  aTIn = Tiff_Im::StdConvGen(aDir+*itN,1,true);
         if (aSzIm.x<0)
         {
            aSzIm = aTIn.sz();
            aImIn = Im2D_REAL4(aSzIm.x,aSzIm.y,0.0);
            if (ByMoy)
               aFFRes1 = Im2D_REAL8(aSzIm.x,aSzIm.y,0.0);
            aNbPix = aSzIm.x*aSzIm.y;
            aSzF = round_up(Pt2dr(aSzIm)/aResol);
         }
         else
         {
             if (aSzIm!=aTIn.sz())
             {
                 std::cout << "For Image " << *itN << "\n";
                 ELISE_ASSERT(false,"Different size");
             }
         }
         double aSom = 0;
         // ELISE_COPY(aImIn.all_pts(),aTIn.in(),aImIn.out()|sigma(aSom));

         Fonc_Num aFIN = aTIn.in();
         ELISE_COPY(aImIn.all_pts(),Rconv(aFIN),aImIn.out()|sigma(aSom));
         double aMoy = aSom/aNbPix;

         if (ByMoy)
         {
            ELISE_COPY
            (
                 aImIn.all_pts(),
                 aFFRes1.in()+(aImIn.in()/aMoy),
                 aFFRes1.out()
            );
         }
         else
         {
             Im2D_REAL4 aIRed(aSzF.x,aSzF.y);
             ELISE_COPY
             (
                   aIRed.all_pts(),
                   StdFoncChScale
                   (
                       aImIn.in_proj() / aMoy,
                       Pt2dr(0.0,0.0),
                       Pt2dr(aResol,aResol),
                       Pt2dr(aDilate,aDilate)
                   ),
                   aIRed.out()
             );
             aVImRed.push_back(aIRed);
         }

         aCpt--;
    }


    Fonc_Num aF;

    if (ByMoy)
    {

       std::cout << "Filtrage Median "<< aNbMed << "\n";
       ELISE_COPY
       (
             aFFRes1.all_pts(),
             MedianBySort(aFFRes1.in_proj()/aNbIm,aNbMed),
             aFFRes1.out()
       );


       aF =  StdFoncChScale
                   (
                       aFFRes1.in_proj() / aNbIm,
                       Pt2dr(0.0,0.0),
                       Pt2dr(aResol,aResol),
                       Pt2dr(aDilate,aDilate)
                   );
       if (aNbMedSsRes)
         aF = MedianBySort(aF,aNbMedSsRes);
    }
    else
    {
         int aMoyMed = 2;
         Im2D_REAL4 aRes = ImMediane<float,double>(aVImRed,-1e30,0.0,TolMed);
         aF = aRes.in_proj();

         for (int aK=0 ; aK<3 ; aK++)
             aF = MedianBySort(aF,4);

         for (int aK=0 ; aK<3 ; aK++)
             aF = rect_som(aF,aMoyMed)/ElSquare(1+2*aMoyMed);
    }


    Tiff_Im aTOut
            (
                 aNameOut.c_str(),
                 aSzF ,
                 GenIm::real8,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );
    ELISE_COPY
    (
         aTOut.all_pts(),
         aF,
         aTOut.out()
    );

    }

    return EXIT_SUCCESS;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
