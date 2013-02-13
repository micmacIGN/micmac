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
#ifndef _HASSAN_HASSAN_FO_H
#define _HASSAN_HASSAN_FO_H





/*****************************************************/
/**************** Fonction special *******************/
/*****************************************************/

extern bool resoud_syst_lin(int n, REAL*tab);
extern bool resoud_syst_lin(int n, REAL4*tab);
extern bool resoud_syst_lin_koleski(int n, REAL4*tab);

extern bool moindre_carre(INT n, INT m, REAL* tab);  
extern bool moindre_carre(INT n, INT m, REAL* tab, REAL* poids, REAL* param);   



extern bool egal(Pt3dr p1, Pt3dr p2);
extern bool intersect(Pt2dr p0,Pt2dr p1,Pt2dr p2,Pt2dr p3,Pt2dr& p_int);

extern ElList<Pt2di>   isolement3(Im2D_U_INT1 im,Im2D<INT,INT> im_reg, Output w,INT nb_colour);
extern ElList<Pt2di>   isolement2(Im2D_U_INT1 im, Im2D<INT,INT> im_reg, Output w);
extern ElList<Pt2di>   isolement(Im2D_U_INT1 im, Im2D<INT,INT> im_reg, Output w);
extern ElList<Hregion> isolement(Im2D_U_INT1 im, Output w);


extern int   fonccomp8(const void* , const void* );
extern int   fonccomp(const void* , const void* );
extern int   fonccomp_INT(const void* , const void* );
extern REAL4 median(Im2D_REAL4 , Facette_2d , REAL,REAL);


extern void affich_facette(Facette_2d f,Output w,int pal);
extern void affich_facette(Facette_2d f,Output w,Fonc_Num fn);
extern void affich_facette(ElList<Facette_2d>l,Output w,int pal);
extern void affich_facette(ElList<Facette_2d>l,Output w,Fonc_Num fn);


extern Pt3dr phot_to_terrain(Pt2di p2, REAL z, Wcor w);
extern Pt2di phot_to_phot(Pt2di p2, REAL z, Wcor wa,Wcor wb);
extern Pt3dr phot_phot_to_terrain(Wcor Wa, Pt2di pa, Wcor Wb, Pt2di pb);


extern Pt2di              terrain_to_phot(Pt3dr, Wcor);

extern Pt2di              terrain_to_imagette(Pt2dr pt, Pt2dr p0, Pt2dr v, REAL pas);
extern Facette_2d         terrain_to_imagette(Facette_2d f, Pt2dr p0, Pt2dr v, REAL pas);

extern Facette_2d         terrain_to_ortho(Facette_2d f, Pt2dr p0, Pt2dr d);
extern ElList<Facette_2d> terrain_to_ortho(ElList<Facette_2d> l, Pt2dr p0, Pt2dr d);

extern Facette_2d         terrain_to_phot(Facette_3d f,Wcor w);
extern ElList<Facette_2d> terrain_to_phot(ElList<Facette_3d> lf,Wcor w);
extern Facette_2d         terrain_to_phot(Facette_2d f,Wcor w, REAL z_ter);

extern Facette_2d         ortho_to_terrain(Facette_2d f, Pt2dr p0, Pt2dr d);
extern ElList<Facette_2d> ortho_to_terrain(ElList<Facette_2d> l, Pt2dr p0, Pt2dr d);

extern Facette_2d         phot_to_terrain(Facette_2d f,Wcor w, REAL z_ter);


extern Pt3dr              carte_to_terrain(Pt3dr p,Wcor w);
extern Facette_2d         carte_to_terrain(Facette_2d f,Wcor w, REAL z_lamb);
extern ElList<Facette_2d> carte_to_terrain(ElList<Facette_2d>l,Wcor w, REAL z_lamb);

extern Facette_2d         carte_to_ortho(Facette_2d f,Wcor w,REAL z_lamb, Pt2dr p0, Pt2dr d);
extern ElList<Facette_2d> carte_to_ortho(ElList<Facette_2d> l,Wcor w,REAL z_lamb, Pt2dr p0, Pt2dr d);

extern bool               mnt_oliv(Pt3dr& p, Wcor w);

extern void               facette_to_polygone(Facette_2d f, Im2D_U_INT1 im);
extern Facette_2d         seg_to_pt(ElList<Facette_2d>lsegf);

extern void       recal_xy(Pt2di& p0, Pt2di& p1,REAL t);
extern void       recal_xy(Pt2dr& p0, Pt2dr& p1,REAL t);
extern void       recal_xy(Pt2dr& p0, Pt2dr& p1, Pt2dr& p2, REAL t);
extern Pt2dr      recal_xy(Pt2dr  p0, Pt2dr  p1, Pt2dr  p2, REAL t1, REAL t2);
extern Facette_2d recal_xy(Facette_2d f,REAL dist);
extern Facette_2d recal_xyz( WcorVis Wa, 
                             WcorVis Wb,
                             Im2D_U_INT1 gradg,
                             Im2D_U_INT1 gradd,
                             Facette_2d f, 
                             REAL z, 
                             REAL dz,  
                             REAL dxy,
                             REAL pas,
                             REAL param_xy=1.5,
                             REAL param_z=1
                           );

extern Facette_2d recal_seg_xy(Facette_2d f,REAL dist);
extern Facette_3d recal_seg_z(Facette_3d  f,REAL dist);

extern REAL4 median_recal(Im2D_U_INT1 grad, Facette_2d f);
extern REAL4 moyen_recal(Im2D_U_INT1 grad, Facette_2d f);

extern Facette_2d seg_recal(Facette_2d f, Im2D_U_INT1 grad, REAL dist);
extern Facette_3d seg_recal(Facette_2d sega, Facette_2d segb, WcorVis Wa, WcorVis Wb,REAL z_lamb_0,REAL z_lamb_1);
extern REAL scor(Facette_2d f, WcorVis W, REAL4 z,Im2D_U_INT1 grad);
extern REAL scor(Facette_3d f, WcorVis W,Im2D_U_INT1 grad);
extern REAL scor(Facette_3d f, WcorVis W1, Im2D_U_INT1 grad1, WcorVis W2, Im2D_U_INT1 grad2);





// eliminer les points ayant un distance moins que le semi fenetre p(x,y)
// les points n'ont pas le meme poids :  la liste est ordonner en importante incrementee

template <class type1, class type2> ElList<Pt2di> nettoyer(ElList<Pt2di> lp, Pt2di p, Im2D<type1, type2> tamp);

ElList<Pt2di> nettoyer(ElList<Pt2di> lp, Pt2di p);




//filtrage  par le max pour l'image 3D 

template <class type1, class type2> void cube_max(Im3D<type1, type2> Im, Pt3di p); 
template <class type1, class type2> void cube_min(Im3D<type1, type2> Im, Pt3di p); 
template <class type1, class type2> void ouverture(Im3D<type1, type2> Im, Pt3di p); 
template <class type1, class type2> void fermeture(Im3D<type1, type2> Im, Pt3di p); 
template<class type1, class type2>  void affich_Im2D(  Im2D<type1 , type2> im,
                                                       Output w,
                                                       INT nb_color
                                                    );

template <class type1, class type2> void affich_Im3D( Im3D<type1, type2> im,
                                                      Output w,
                                                      INT color,
                                                      INT n_colone,
                                                      INT dierect = 1      //1 : x, 2 : y et 3 : z
                                                    );


extern Im2D_REAL4 norm_gradient_deriche(Im2D_U_INT1 im, REAL alpha);


//co_aligner les segments de facette ayant de meme direction

extern Facette_2d co_aligner(Facette_2d f);
extern Facette_3d co_aligner(Facette_3d f, REAL precision);
extern void       co_aligner(ElFilo<Facette_3d>& filo, REAL precision);


//   correlation    

extern Fonc_Num Hcor(Fonc_Num f1,Fonc_Num f2,Pt2di p);


//   correlation avec une contrainte de region de l'image gauche    

extern Fonc_Num Hcor(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3,Pt2di p);


//   correlation avec une contrainte de contour de l'image gauche    

extern Fonc_Num Hcorcont(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3,Pt2di p);



//   filtrage moindre carres avec une contrainte de region de l'image gauche    

extern Fonc_Num Hmdcar(Fonc_Num f1,Fonc_Num f2,Pt2di p,INT nParam);




//   filtrage moindre carres avec des contraintes :
//      1-region de l'image gauche 
//      2-poid de coef de cor   

extern Fonc_Num Hmdcarp(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3,Pt2di p,INT nParam);
extern Disc_Pal palette_64();

#endif // _HASSANL_HASSAN_FO_H

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
