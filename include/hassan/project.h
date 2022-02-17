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
#ifndef _HASSAN_PROJECT_H
#define _HASSAN_PROJECT_H


class Project
{
     public : 
          REAL4        _niv0;
          REAL4        _niv1;
          Ori3D_Std    _o3;
          Pt2di        _p0ph;
          Pt2di        _sz; 
          Im2D_U_INT1  _iphoto;
          Pt3dr        _p0ph3d;
          Pt3dr        _ori_lum;
 
          Project(Pt3dr, Pt3dr, Ori3D_Std);
          Project(Pt3dr, Pt3dr, REAL4, REAL4, Ori3D_Std);
          Im2D_U_INT1 image();
          
          void set_lum(Pt3dr ori_lum){_ori_lum = ori_lum/sqrt(scal(ori_lum,ori_lum));} 
          void lancer_de_rayon(ElList<Batiment>,REAL4, REAL4, Video_Win);
          void l_d_r_s_o(ElList<Batiment>, REAL4, REAL4, Video_Win);
          bool l_d_r_s_o(ElList<Batiment>, REAL4, REAL4, int, int, Batiment&, Facette_3d&);
          void l_d_r_z(Video_Win);
          void l_d_r_z(Facette_3d, Video_Win);
          void l_d_r_z(ElList<Facette_3d>, Video_Win);
          void l_d_r_z(ElList<Batiment>, Video_Win);
          void affiche(Video_Win);
          
      private:
          void set_param(Pt3dr,Pt3dr);
          REAL4 const INFINI;


};


#endif // _HASSAN_PROJECT_H

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
