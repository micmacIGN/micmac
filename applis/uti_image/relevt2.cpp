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
#include "general/all.h"
#include "private/all.h"


int main(int argc,char ** argv)
{

  // Partie simulation de donnees 
  Pt3dr aTR = Pt3dr(0.0,0.0,10.0);
  REAL aTeta01 = 0.04;
  REAL aTeta02 = -0.05;
  REAL aTeta12 = 0.03;
  ElRotation3D aRotM2C(aTR,aTeta01,aTeta02,aTeta12);
  
  CamStenopeIdeale aCamSimul(1.0,Pt2dr(0,0));
  aCamSimul.SetOrientation(aRotM2C);

  Pt3dr aP0 (5,4,1);
  Pt3dr aP1 (-4,5,2);
  Pt3dr aP2 (3,-6,1);

    
  Pt2dr aQ0 = aCamSimul.R3toF2(aP0);
  Pt2dr aQ1 = aCamSimul.R3toF2(aP1);
  Pt2dr aQ2 = aCamSimul.R3toF2(aP2);

  // Partie utilisation du relevement dans l'espace
  std::list<ElRotation3D> aLRot;
  CamStenopeIdeale aCamInc(1.0,Pt2dr(0,0));
  aCamInc.OrientFromPtsAppui(aLRot,aP0,aP1,aP2,aQ0,aQ1,aQ2);


  // Partie verification 
   for (std::list<ElRotation3D>::iterator anItR = aLRot.begin(); anItR!=aLRot.end() ; anItR++)
   {
       cout << "----------------------------\n";
       cout << anItR->teta01() << " " <<  anItR->teta02() << " " << anItR->teta12() << "\n";
       ElRotation3D aRotC2M = anItR->inv();
       ElMatrix<REAL> aMat = anItR->Mat();
       Pt3dr aImI = aMat * Pt3dr(1,0,0);
       cout << aImI.x << " " << aImI.y << " " << aImI.z << "\n";
       cout << aMat * Pt3dr(0,1,0) << "\n";
       cout << aMat * Pt3dr(0,0,1) << "\n";
       cout << "DIST = " << aRotM2C.Mat().L2(aMat) << "\n";
       cout << "DIST = " << euclid(aRotM2C.tr()- anItR->tr()) << "\n";
       cout << "COPT = " << aRotC2M.tr() << "\n";
   }

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
