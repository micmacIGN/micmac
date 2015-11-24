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
Im2D_INT4  Ok_eLise()
{
  Im2D_INT4 res(108,58,0);
  res.SetLine(0,0,108,1);
  res.SetLine(0,1,108,1);
  res.SetLine(0,2,108,1);
  res.SetLine(0,3,108,1);
  res.SetLine(0,4,108,1);
  res.SetLine(0,5,43,1);
  res.SetLine(45,5,63,1);
  res.SetLine(0,6,42,1);
  res.SetLine(46,6,62,1);
  res.SetLine(0,7,42,1);
  res.SetLine(46,7,62,1);
  res.SetLine(0,8,42,1);
  res.SetLine(46,8,62,1);
  res.SetLine(0,9,108,1);
  res.SetLine(0,10,21,1);
  res.SetLine(25,10,83,1);
  res.SetLine(0,11,17,1);
  res.SetLine(28,11,80,1);
  res.SetLine(0,12,15,1);
  res.SetLine(28,12,80,1);
  res.SetLine(0,13,14,1);
  res.SetLine(17,13,6,1);
  res.SetLine(29,13,79,1);
  res.SetLine(0,14,12,1);
  res.SetLine(15,14,9,1);
  res.SetLine(29,14,79,1);
  res.SetLine(0,15,11,1);
  res.SetLine(14,15,11,1);
  res.SetLine(28,15,80,1);
  res.SetLine(0,16,11,1);
  res.SetLine(14,16,10,1);
  res.SetLine(28,16,14,1);
  res.SetLine(46,16,62,1);
  res.SetLine(0,17,10,1);
  res.SetLine(13,17,11,1);
  res.SetLine(27,17,13,1);
  res.SetLine(46,17,62,1);
  res.SetLine(0,18,9,1);
  res.SetLine(13,18,30,1);
  res.SetLine(46,18,62,1);
  res.SetLine(0,19,9,1);
  res.SetLine(13,19,15,1);
  res.SetLine(39,19,4,1);
  res.SetLine(46,19,12,1);
  res.SetLine(66,19,5,1);
  res.SetLine(72,19,36,1);
  res.SetLine(0,20,9,1);
  res.SetLine(13,20,15,1);
  res.SetLine(39,20,4,1);
  res.SetLine(46,20,10,1);
  res.SetLine(59,20,6,1);
  res.SetLine(68,20,2,1);
  res.SetLine(72,20,36,1);
  res.SetLine(0,21,9,1);
  res.SetLine(14,21,17,1);
  res.SetLine(35,21,8,1);
  res.SetLine(46,21,9,1);
  res.SetLine(58,21,9,1);
  res.SetLine(69,21,1,1);
  res.SetLine(72,21,36,1);
  res.SetLine(0,22,9,1);
  res.SetLine(14,22,17,1);
  res.SetLine(34,22,9,1);
  res.SetLine(46,22,8,1);
  res.SetLine(57,22,11,1);
  res.SetLine(72,22,36,1);
  res.SetLine(0,23,9,1);
  res.SetLine(16,23,15,1);
  res.SetLine(34,23,9,1);
  res.SetLine(46,23,8,1);
  res.SetLine(56,23,13,1);
  res.SetLine(72,23,10,1);
  res.SetLine(89,23,11,1);
  res.SetLine(102,23,6,1);
  res.SetLine(0,24,10,1);
  res.SetLine(19,24,12,1);
  res.SetLine(34,24,9,1);
  res.SetLine(46,24,7,1);
  res.SetLine(56,24,13,1);
  res.SetLine(72,24,8,1);
  res.SetLine(83,24,5,1);
  res.SetLine(90,24,9,1);
  res.SetLine(103,24,5,1);
  res.SetLine(0,25,11,1);
  res.SetLine(21,25,10,1);
  res.SetLine(34,25,9,1);
  res.SetLine(46,25,7,1);
  res.SetLine(55,25,15,1);
  res.SetLine(72,25,7,1);
  res.SetLine(82,25,7,1);
  res.SetLine(91,25,8,1);
  res.SetLine(103,25,5,1);
  res.SetLine(0,26,13,1);
  res.SetLine(20,26,11,1);
  res.SetLine(34,26,9,1);
  res.SetLine(46,26,7,1);
  res.SetLine(55,26,15,1);
  res.SetLine(72,26,6,1);
  res.SetLine(81,26,8,1);
  res.SetLine(92,26,7,1);
  res.SetLine(103,26,5,1);
  res.SetLine(0,27,12,1);
  res.SetLine(16,27,15,1);
  res.SetLine(34,27,9,1);
  res.SetLine(46,27,7,1);
  res.SetLine(55,27,15,1);
  res.SetLine(72,27,6,1);
  res.SetLine(80,27,10,1);
  res.SetLine(92,27,16,1);
  res.SetLine(0,28,10,1);
  res.SetLine(13,28,18,1);
  res.SetLine(34,28,9,1);
  res.SetLine(46,28,7,1);
  res.SetLine(56,28,14,1);
  res.SetLine(72,28,5,1);
  res.SetLine(80,28,10,1);
  res.SetLine(93,28,15,1);
  res.SetLine(0,29,9,1);
  res.SetLine(12,29,19,1);
  res.SetLine(34,29,9,1);
  res.SetLine(46,29,7,1);
  res.SetLine(56,29,15,1);
  res.SetLine(72,29,5,1);
  res.SetLine(80,29,10,1);
  res.SetLine(93,29,15,1);
  res.SetLine(0,30,8,1);
  res.SetLine(11,30,20,1);
  res.SetLine(34,30,9,1);
  res.SetLine(46,30,8,1);
  res.SetLine(57,30,20,1);
  res.SetLine(80,30,10,1);
  res.SetLine(93,30,15,1);
  res.SetLine(0,31,7,1);
  res.SetLine(10,31,21,1);
  res.SetLine(34,31,9,1);
  res.SetLine(46,31,8,1);
  res.SetLine(58,31,18,1);
  res.SetLine(80,31,10,1);
  res.SetLine(93,31,15,1);
  res.SetLine(0,32,6,1);
  res.SetLine(9,32,22,1);
  res.SetLine(34,32,9,1);
  res.SetLine(46,32,9,1);
  res.SetLine(60,32,16,1);
  res.SetLine(93,32,15,1);
  res.SetLine(0,33,5,1);
  res.SetLine(8,33,23,1);
  res.SetLine(34,33,9,1);
  res.SetLine(46,33,9,1);
  res.SetLine(64,33,12,1);
  res.SetLine(79,33,29,1);
  res.SetLine(0,34,5,1);
  res.SetLine(8,34,23,1);
  res.SetLine(34,34,9,1);
  res.SetLine(46,34,11,1);
  res.SetLine(67,34,9,1);
  res.SetLine(79,34,29,1);
  res.SetLine(0,35,4,1);
  res.SetLine(8,35,23,1);
  res.SetLine(34,35,9,1);
  res.SetLine(47,35,11,1);
  res.SetLine(69,35,7,1);
  res.SetLine(79,35,29,1);
  res.SetLine(0,36,4,1);
  res.SetLine(7,36,17,1);
  res.SetLine(26,36,5,1);
  res.SetLine(34,36,6,1);
  res.SetLine(49,36,13,1);
  res.SetLine(70,36,7,1);
  res.SetLine(80,36,28,1);
  res.SetLine(0,37,3,1);
  res.SetLine(7,37,15,1);
  res.SetLine(25,37,6,1);
  res.SetLine(34,37,32,1);
  res.SetLine(71,37,6,1);
  res.SetLine(80,37,28,1);
  res.SetLine(0,38,3,1);
  res.SetLine(8,38,13,1);
  res.SetLine(25,38,6,1);
  res.SetLine(34,38,34,1);
  res.SetLine(72,38,5,1);
  res.SetLine(80,38,12,1);
  res.SetLine(93,38,15,1);
  res.SetLine(0,39,3,1);
  res.SetLine(8,39,13,1);
  res.SetLine(24,39,7,1);
  res.SetLine(34,39,35,1);
  res.SetLine(72,39,6,1);
  res.SetLine(81,39,11,1);
  res.SetLine(93,39,15,1);
  res.SetLine(0,40,4,1);
  res.SetLine(9,40,11,1);
  res.SetLine(23,40,8,1);
  res.SetLine(34,40,16,1);
  res.SetLine(51,40,19,1);
  res.SetLine(73,40,6,1);
  res.SetLine(81,40,10,1);
  res.SetLine(92,40,8,1);
  res.SetLine(102,40,6,1);
  res.SetLine(0,41,4,1);
  res.SetLine(11,41,7,1);
  res.SetLine(21,41,10,1);
  res.SetLine(34,41,16,1);
  res.SetLine(51,41,2,1);
  res.SetLine(54,41,16,1);
  res.SetLine(73,41,6,1);
  res.SetLine(82,41,8,1);
  res.SetLine(92,41,7,1);
  res.SetLine(103,41,5,1);
  res.SetLine(0,42,5,1);
  res.SetLine(20,42,11,1);
  res.SetLine(34,42,15,1);
  res.SetLine(50,42,3,1);
  res.SetLine(54,42,16,1);
  res.SetLine(73,42,8,1);
  res.SetLine(83,42,6,1);
  res.SetLine(91,42,8,1);
  res.SetLine(103,42,5,1);
  res.SetLine(0,43,6,1);
  res.SetLine(18,43,13,1);
  res.SetLine(34,43,15,1);
  res.SetLine(50,43,3,1);
  res.SetLine(54,43,17,1);
  res.SetLine(73,43,9,1);
  res.SetLine(89,43,11,1);
  res.SetLine(102,43,6,1);
  res.SetLine(0,44,8,1);
  res.SetLine(15,44,16,1);
  res.SetLine(34,44,15,1);
  res.SetLine(50,44,3,1);
  res.SetLine(54,44,17,1);
  res.SetLine(73,44,35,1);
  res.SetLine(0,45,31,1);
  res.SetLine(34,45,15,1);
  res.SetLine(50,45,3,1);
  res.SetLine(55,45,16,1);
  res.SetLine(73,45,35,1);
  res.SetLine(0,46,31,1);
  res.SetLine(34,46,14,1);
  res.SetLine(50,46,3,1);
  res.SetLine(55,46,15,1);
  res.SetLine(72,46,36,1);
  res.SetLine(0,47,31,1);
  res.SetLine(34,47,14,1);
  res.SetLine(50,47,3,1);
  res.SetLine(56,47,14,1);
  res.SetLine(72,47,36,1);
  res.SetLine(0,48,31,1);
  res.SetLine(34,48,13,1);
  res.SetLine(50,48,3,1);
  res.SetLine(57,48,12,1);
  res.SetLine(72,48,36,1);
  res.SetLine(0,49,31,1);
  res.SetLine(34,49,12,1);
  res.SetLine(50,49,3,1);
  res.SetLine(55,49,1,1);
  res.SetLine(58,49,11,1);
  res.SetLine(71,49,37,1);
  res.SetLine(0,50,30,1);
  res.SetLine(35,50,8,1);
  res.SetLine(50,50,3,1);
  res.SetLine(55,50,2,1);
  res.SetLine(59,50,8,1);
  res.SetLine(70,50,38,1);
  res.SetLine(0,51,27,1);
  res.SetLine(50,51,3,1);
  res.SetLine(54,51,5,1);
  res.SetLine(68,51,40,1);
  res.SetLine(0,52,53,1);
  res.SetLine(54,52,8,1);
  res.SetLine(66,52,42,1);
  res.SetLine(0,53,108,1);
  res.SetLine(0,54,108,1);
  res.SetLine(0,55,108,1);
  res.SetLine(0,56,108,1);
  res.SetLine(0,57,108,1);

// ELISE_COPY(res.all_pts(),1-res.in(),res.out());
return res;
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
