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


/*
    Acceleration :
      - Presel sur point les plus stables
      - calcul de distance de stabilite ? => Uniquement si pas Invar Ech !!!!
      - Apres pre-sel, a simil (ou autre) :
             * selection des point dans  regions homologues
             * indexation

    Plus de points :
        SIFT Criteres ?
*/

#include "NewRechPH.h"



/*************************************************/
/*                                               */
/*           cAFM_Im_Sec                         */
/*                                               */
/*************************************************/


//  aTBuf.oset(Pt2di(aKRho,aKTeta),aVal);


double DistHistoGrad(cCompileOPC & aMast,int aShift,cCompileOPC & aSec)
{
    Pt2di aSzInit = aMast.mSzIm;
    Pt2di aSzG (aSzInit.x-1,aSzInit.y);

    TIm2D<INT1,INT>  aImM (aMast.mOPC.ImLogPol());
    TIm2D<INT1,INT>  aImS ( aSec.mOPC.ImLogPol());

    double aSomEcPds = 0;
    double aSomPds = 0;

/*
    Im2D_REAL4 aMGx(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTMGx(aMGx);
    Im2D_REAL4 aMGy(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTMGy(aMGy);
    double aSumGM = 0;
    Im2D_REAL4 aMN(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTMN(aMN);

    Im2D_REAL4 aSGx(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTSGx(aSGx);
    Im2D_REAL4 aSGy(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTSGy(aSGy);
    double aSumGS = 0;
    Im2D_REAL4 aSN(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTSN(aSN);
*/
    
    double aSomEcRad =0;
    double aSomEcGrad =0;
    int aNb=0;
    for (int aKRho=0 ; aKRho<aSzG.x; aKRho++)
    {
        for (int aKTeta=0 ; aKTeta<aSzG.y; aKTeta++)
        {
             Pt2di aPM (aKRho  ,  aKTeta);
             Pt2di aPMr(aPM.x+1,  aPM.y);
             Pt2di aPMt(aPM.x  , (aPM.y+1)%aSzG.y);


             Pt2di aPS  (aKRho  ,  (aKTeta+aShift + aSzG.y) %aSzG.y);
             Pt2di aPSr (aPS.x+1,aPS.y);
             Pt2di aPSt (aPS.x  ,(aPS.y+1)%aSzG.y);

             Pt2dr aGradM (aImM.get(aPM)-aImM.get(aPMr),aImM.get(aPM)-aImM.get(aPMt));
             double aNormM = euclid(aGradM);

             Pt2dr aGradS (aImS.get(aPS)-aImS.get(aPSr),aImS.get(aPS)-aImS.get(aPSt));
             double aNormS = euclid(aGradS);
             double aPds = sqrt(sqrt(aNormM*aNormS));
             // double aPds = sqrt(aNormM*aNormS);
             aSomEcRad += ElAbs(aImM.get(aPM)-aImS.get(aPS));
             aSomEcGrad += dist4(aGradM-aGradS);
             aNb++;
             if (aPds > 0)
             {
                aGradM = aGradM / aNormM;
                aGradS = aGradS / aNormS;
                Pt2dr aPEcart = aGradM / aGradS;
                aPEcart = aPEcart - Pt2dr(1,0);
                aSomEcPds +=  aPds * euclid(aPEcart);
                aSomPds += aPds;
             }
        }
    }
/*
Pt2dr aPM = aMast.mOPC.Pt();
Pt2dr aPS = aSec.mOPC.Pt();
std::cout << "PppPpp " << aPM << " " << aPS << (aPM+aPS) / 2.0 << aPM-aPS 
          << " ECART RAD " << aSomEcRad/aNb 
          << " ECART Grad " << aSomEcGrad/aNb 
          << "\n";
std::cout << "mShitfBestmShitfBest " << aMast.mShitfBest 
          << " " << (aSomEcPds / aSomPds) 
          <<  aSzInit 
          << "\n";
*/

    return aSomEcPds / aSomPds;
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
aooter-MicMac-eLiSe-25/06/2007*/
