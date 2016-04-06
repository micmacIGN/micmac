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
#ifndef _HASSAN_H_CORREL_H
#define _HASSAN_H_CORREL_H

class H_Graphe;

class H_Wcor: public virtual Wcor
{
      public :

          H_Wcor
          (    Cub3DTer,
               char * name
          );
   
          U_INT1 radio_pt_terrain(Pt3dr);
          U_INT1 radio_pt_terrain(Pt3dr,Im2D_U_INT1);
          U_INT1 radio_biline_pt_terrain(Pt3dr);
          U_INT1 radio_biline_pt_terrain(Pt3dr,Im2D_U_INT1);
          U_INT1 radio_bicube_pt_terrain(Pt3dr);
          U_INT1 radio_bicube_pt_terrain(Pt3dr,Im2D_U_INT1);

          REAL   b_sur_h(Wcor Wb, REAL z1=30, REAL z2=60);

          Pt3dr  phot_to_ter(Pt2dr p, REAL z);
          Pt2dr  ter_to_phot(Pt3dr p);
          Facette_2d terrain_to_phot( Facette_2d f, REAL z);
          Facette_2d terrain_to_phot( Facette_3d f);

};

class H_WcorVis :  public WcorVis
{
     public :

          H_WcorVis
          (    Cub3DTer,
               char * name,
               Video_Display,
               Elise_Set_Of_Palette
          );

          H_WcorVis
          (    Cub3DTer,
               char * name,
               Video_Win W
          );

          void affiche_facette(Facette_2d f, INT color,  bool etiquet = false);
          void desaffiche_facette(Facette_2d f, bool etiquet = false);

          void affiche_facette(Facette_3d f,INT color, bool etiquet = false);
          void desaffiche_facette(Facette_3d f, bool etiquet = false);

          void affiche_facette(ElList<Facette_3d>  lf,INT color, bool etiquet = false);
          void desaffiche_facette(ElList<Facette_3d> lf, bool etiquet = false);

          void affiche_facette(ElFilo<Facette_3d>&  f_f,INT color, bool etiquet = false);
          void desaffiche_facette(ElFilo<Facette_3d>& f_f, bool etiquet = false);

          void affiche_facette(Facette_2d f, REAL z, INT color,  bool etiquet = false);
          void desaffiche_facette(Facette_2d f, REAL z, bool etiquet = false);

          void affiche_facette(ElList<Facette_2d>  lf, REAL z, INT color,  bool etiquet = false);
          void desaffiche_facette(ElList<Facette_2d> lf, REAL z, bool etiquet = false);

          void affiche_facette(ElFilo<Facette_2d>&  f_f, REAL z, INT color,  bool etiquet = false);
          void desaffiche_facette(ElFilo<Facette_2d>& f_f, REAL z, bool etiquet = false);

          void affiche_graphe(H_Graphe& graphe, INT color);
          void desaffiche_graphe(H_Graphe& graphe);

          INT  etiquet_in();
          Im2D_U_INT2 etiquet(){return _etiquet;}

          U_INT1 radio_pt_terrain(Pt3dr);
          U_INT1 radio_pt_terrain(Pt3dr,Im2D_U_INT1);
          U_INT1 radio_biline_pt_terrain(Pt3dr);
          U_INT1 radio_biline_pt_terrain(Pt3dr,Im2D_U_INT1);
          U_INT1 radio_bicube_pt_terrain(Pt3dr);
          U_INT1 radio_bicube_pt_terrain(Pt3dr,Im2D_U_INT1);

          REAL   b_sur_h(WcorVis Wb, REAL z1=30, REAL z2=60);

          Pt3dr  phot_to_ter(Pt2dr p, REAL z);
          Pt2dr  ter_to_phot(Pt3dr p);
          Facette_2d terrain_to_phot( Facette_2d f, REAL z);
          Facette_2d terrain_to_phot( Facette_3d f);

    private :
    
          Im2D_U_INT2 _etiquet;
};

extern void correlation(Im2D_U_INT1 im_a, Im2D_U_INT1 im_b, Im2D_U_INT1 masq, Im2D_REAL4 coef, INT semi_fenet, Output win = 0);
extern void correlation_ad(Im2D_U_INT1 im_a, Im2D_U_INT1 im_b, Im2D_U_INT1 masq, Im2D_REAL4 coef, INT semi_fenet, Output win = 0);
extern void correlation_ad_non_cent(Im2D_U_INT1 im_a, Im2D_U_INT1 im_b, Im2D_U_INT1 masq, Im2D_REAL4 coef, INT semi_fenet, Output win = 0);
extern REAL correlation_ad_glob(Im2D_U_INT1 im_a, Im2D_U_INT1 im_b, Im2D_U_INT1 masq);
extern REAL correlation_ad_glob_non_cent(Im2D_U_INT1 im_a, Im2D_U_INT1 im_b, Im2D_U_INT1 masq);


#endif // _ELISE_GENERAL_H_CORREL_H

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
