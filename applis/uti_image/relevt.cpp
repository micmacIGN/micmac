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


static  list<Pt3dr> aLP3;
static  list<Pt2dr> aLP2;
static CamStenopeIdeale * pCam;

void AddPt(Pt3dr aP,REAL anErr)
{
   aLP3.push_back(aP);
   aLP2.push_back(pCam->R3toF2(aP)+Pt2dr(NRrandC(),NRrandC())*anErr);
}

void AddPt(Pt3dr aP3,Pt2dr aP2)
{
    aLP3.push_back(aP3);
    aLP2.push_back(aP2);
}

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
  pCam  = &aCamSimul;

  AddPt(Pt3dr(-18061.5,41158,27928.8),Pt2dr(1224,117));
  AddPt(Pt3dr(-13100.3,40971.7,17641.5),Pt2dr(1429,498));
  AddPt(Pt3dr(-8212.36,36677.2,3949.58),Pt2dr(1620,1098));
  AddPt(Pt3dr(-7013.52,35142,-4484.05),Pt2dr( 1685,1596));
  AddPt(Pt3dr(-19391.2,27868.5,-7424.43),Pt2dr(792,1819));

// 1 -18061.5 41158 27928.8
// 2 -13100.3 40971.7 17641.5
// 3 -8212.36 36677.2 3949.58
// 4 -7013.52 35142 -4484.05
// 5 -19391.2 27868.5 -7424.43
// 6 -20221.1 31052.1 -527.415
// 7 -20510.1 33945.8 6726.41
// 8 -11517.8 38550.3 6505.82
// 9 -21180.6 36844.8 17854.6



// 1 1224 117
// 2 1429 498
// 3 1620 1098
// 4 1685 1596
// 5 792 1819
// 6 875 1361
// 7 955 977
// 8 1476 986
// 9 1010 486


/*
  AddPt(Pt3dr (5,4,1),0.0);
  AddPt(Pt3dr (-4,5,2),0.0);
  AddPt(Pt3dr (3,-6,1),0.0);
  AddPt(Pt3dr (4,-7,2),0.01);
*/

    
  // Partie utilisation du relevement dans l'espace
  CamStenopeIdeale aCamInc(2128.51,Pt2dr(1298.47,972.67));

  REAL aDist;
  ElRotation3D aRot = aCamInc.CombinatoireOFPA(10,aLP3,aLP2,&aDist);


  // Partie verification 
   {
       ElRotation3D *  anItR = &aRot;
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
