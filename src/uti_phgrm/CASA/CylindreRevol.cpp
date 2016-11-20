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

#include "Casa.h"




void cAppli_Casa::TestCylindreRevolution
     (
          cOneSurf_Casa & aSurf,
          const cFaceton & aFc1,
          const cFaceton & aFc2
     )
{
   std::vector<cFaceton> & aVF = aSurf.mVF;
   Pt3dr aPMoy = aSurf.mFMoy->Centre();

   Pt3dr aNorm = aFc1.Normale() ^ aFc2.Normale();
   ElSeg3D aD1 =  aFc1.DroiteNormale();
   ElSeg3D aD2 =  aFc2.DroiteNormale();
   Pt3dr aP = aD1.PseudoInter(aD2);

   ElSeg3D aGen(aP,aP+aNorm);

   double aS1=0;
   double aSD1=0;
   double aSD2=0;

   for (int aK=0 ; aK<int(aVF.size()); aK++)
   {
        double aDist = aGen.DistDoite(aVF[aK].Centre());
        aS1++;
        aSD1+=aDist;
        aSD2+=ElSquare(aDist);
   }
   aSD1 /= aS1;
   aSD2 /= aS1;
   aSD2 -= ElSquare(aSD1);

   double aVar = sqrt(aSD2);


   if (aVar < aSurf.mBestScore)
   {
         std::cout << "   DistDroite = " << aSD1 << " SigmaDist= " << aVar << "\n";
         aSurf.mBestScore = aVar;

/*
         Pt3dr aProj = aGen.ProjOrtho(aPMoy);
         Pt3dr aDir = vunit(aPMoy-aProj);

         cCylindreRevolution  aCyl
                              (
                                  aGen,
                                  aProj + aDir * aSD1
                              );
*/
          // Valeur au hasard pour le premier argument
          cCylindreRevolution  aCyl = cCylindreRevolution::WithRayFixed(true,aGen,aSD1,aPMoy);
                                    

          if (mBestCyl)  
              *mBestCyl = aCyl;
          else
             mBestCyl = new cCylindreRevolution(aCyl);
   }
  
}

void cAppli_Casa::EstimeCylindreRevolution
     (
            cOneSurf_Casa & aSurf,
            const cSectionEstimSurf & aSES
     )
{
    std::cout << "Begin Ransac \n";
    delete mBestCyl;
    mBestCyl = 0;
    aSurf.mBestScore = 1e30;
    std::vector<cFaceton> & aVF = aSurf.mVF;
    std::cout << "Nb Facette " << aVF.size() << "\n";
    for (int aK=0 ; aK<aSES.NbRansac().Val() ; aK++)
    {
        int aK1 = NRrandom3((int)aVF.size());
        int aK2 = NRrandom3((int)aVF.size());
        if (aK1!=aK2)
        {
           TestCylindreRevolution(aSurf,aVF[aK1],aVF[aK2]);
           aK++;
        }
    }

    aSurf.mISAF = &(mSetEq.AllocCylindre("Casa",*mBestCyl));
    std::cout << "End Ransac \n";


/*
    for (int aK=0 ; aK<int(aVF.size()) ; aK++)
    {
       Pt3dr aEucl = aVF[aK].Centre();
       Pt3dr aParam = mBestCyl->E2UVL(aEucl);
       Pt3dr aBackE = mBestCyl->UVL2E(aParam);
    } 
*/
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
