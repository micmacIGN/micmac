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
#ifndef    _HASSAN_FILTRAGE_H
#define    _HASSAN_FILTRAGE_H


void             up_date_facettes(   ElFilo<Facette_3d>& f_f, U_INT1* et_f);

void             filtrer_plan_selon_mne(   ElFilo<Hplan>& lp, Boite&b_c, Facette_3d f, REAL z_min, REAL z_max);


void             transforme_plan_facettes(  Hplan& plan, ElFilo<Facette_3d>& f_f, Facette_2D f, REAL z_min, REAL z_max);
void             transforme_plans_facettes(  ElFilo<Hplan>& lp, ElFilo<Facette_3d>& f_f, Facette_2D f, REAL z_min, REAL z_max);


void             intersection_plans(  ElFilo<Hplan>& lp, Boite& b_c, ElFilo<Facette_3d>& f_f, Facette_2D f, REAL z_min, REAL z_max);


void             intersection_facettes(  ElFilo<Facette_3d>& f_f);
void             intersection_facettes(  ElFilo<Facette_3d>& f_f, ElFilo<Facette_3d>& f_sortie, INT n_facet);

void             filtrer_plan_inf_sup(  ElFilo<Facette_3d>& f_f, Im1D_U_INT1 im_et_f, REAL z_min, REAL z_max);
void             filtrer_plan_inf_sup(  ElFilo<Facette_3d>& f_f, REAL z_min, REAL z_max);

void             filtrer_selon_le_mne(  ElFilo<Facette_3d>& f_f, Im1D_U_INT1 im_et_f, Boite& b_c, INT pourcentage = 5);
void             filtrer_selon_le_mne(  ElFilo<Facette_3d>& f_f, Boite& b_c, INT pourcentage);
void             filtrage_meil(  ElFilo<Facette_3d>& f_f, Boite& b_c, INT nb_max);

void             arrange_selon_le_mne(  ElFilo<Facette_3d>& f_f, ElFilo<INT>& indice, Boite& b_c);

void             coef_mne(  Boite& b_c, Boite& b_mne);
void             coef_mne(  ElFilo<Facette_3d>& f_f, Boite& b_c, Boite& b_mne);
void             coef_mne(  ElFilo<Facette_3d>& f_f, Boite& b_c, Mne& mne);
void             recherche_mne(  ElFilo<Facette_3d>& f_f, Boite& b_c);
void             recherche_mne_prog_dyn(  ElFilo<Facette_3d>& f_f, Boite& b_c);

void             filtrer_facettes_sup_inf_mne(  ElFilo<Facette_3d>& f_f, U_INT1* et_f, Boite& b_c);

void             meilleurs_n_plans( ElFilo<Hplan>& pl_f, ElFilo<REAL8>& a_pl_f, INT n);

void             enlever_plans_horisontaux_mult( ElFilo<Hplan>& pl_f);

void             affiche_facettes( ElFilo<Facette_3d>& f_f, U_INT1* et_f, ElList<Facette_3d> l_facades, Video_Win W );
void             affiche_facettes( ElFilo<Facette_3d>& f_f, ElList<Facette_3d> l_facades, Video_Win W );


Liste_Pts_INT2   intersection_facette_mne( Facette_3d f, Im2D_U_INT1 mne, Im2D_U_INT1 masq, INT decal = 0);
Im2D_U_INT1      image_intersection_facette_mne( Facette_3d f, Im2D_U_INT1 mne, Im2D_U_INT1 masq, INT decal = 0);
Im2D_U_INT1      image_intersection_facette_mne( Facette_3d f, Boite& b, Im2D_U_INT1 masq, INT decal = 0);

void             affiche_image_intersection_facette_mne( Facette_3d f, Boite& b, Im2D_U_INT1 masq, Video_Win W, INT max_palet, INT decal = 0, INT test_stab = 0);

REAL             poids_facette_coef(Facette_3d f, H_WcorVis Wa, H_WcorVis Wb, Pt3dr pas, INT cor_fnt, INT sdng);
REAL             poids_facette_mne( Facette_3d f, Im2D_U_INT1 mne, Im2D_U_INT1 masq, INT decal = 0 , INT test_stab = 0);
REAL             poids_segment_mne( Pt3dr p0, Pt3dr p1, Im2D_U_INT1 mne, INT decal = 0);

REAL             filtrage_morpho(ElFilo<Facette_3d>& f_f, Boite& b_mne, Im2D_U_INT1 masq, INT seuil, INT decal = 0, INT test_stab = 0);
REAL             filtrage_morpho(ElFilo<Facette_3d>& f_f, Boite& b_cor, INT seuil, INT decal = 0, INT test_stab = 0);

void             filtrage_normale(ElFilo<Hplan>& f_f,      ElFilo<REAL>& poids, REAL angl_min = 25);  //degree
void             filtrage_normale(ElFilo<Facette_3d>& f_f, ElFilo<REAL>& poids, REAL angl_min = 25, REAL dist = 1);  //degree et metre
#endif   //_HASSAN_FILTRAGE_H

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
