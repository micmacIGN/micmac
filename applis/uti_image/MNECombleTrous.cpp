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
#include <algorithm>


extern void GenCodeLaserImage(INT);

int main(int argc,char ** argv)
{

   Tiff_Im  aFile("/data1/MNT/ZCarto_ground_filtre.tif");

   Pt2di aSz = aFile.sz();

   Video_Win aW = Video_Win::WStd(aSz,0.15);

   ELISE_COPY
   (
       aFile.all_pts(),
       mod(aFile.in(),256),
       aW.ogray()
   );

   cOriMntCarto aOriMnt("/data1/MNT/ZCarto_ground_filtre.ori");

   Ori3D_Std aOriIm1 ("/data1/MNT/ZCarto.ori");

   Pt3dr aC1 =  aOriIm1.orsommet_de_pdv_terrain();
   Pt2dr aCIm1 = aOriMnt.TerrainToPix(Pt2dr(aC1.x,aC1.y)) ;
   Pt2dr aCIm2 = aCIm1 + Pt2dr(-1000,200);

   cout << aC1 << aCIm1  << "\n";

   aW.draw_circle_abs(aCIm1,5.0,aW.pdisc()(P8COL::red));
   aW.draw_circle_abs(aCIm2,5.0,aW.pdisc()(P8COL::red));
   aW.draw_seg(aCIm1,aCIm2,aW.pdisc()(P8COL::green));

   ELISE_COPY
   (
        line_map_rect(Pt2di(aCIm1-aCIm2),Pt2di(0,0),aSz),
        P8COL::red,
        aW.odisc()
   );

   getchar();
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
