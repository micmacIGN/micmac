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
#ifndef _HASSAN_BOITE_H
#define _HASSAN_BOITE_H

#define DEBUG_HASSAN   1

class Hdroite;
class Hplan;
class H_Graphe;
class Mne;
/////////////////////////////////////////////////

class Boite : public PRC0
{
   public :

       Im3D_U_INT1 _boite;
       Im2D_U_INT1 _masq;
       Pt3dr       _p0;
       Pt3dr       _pas;
       Pt3dr       _d1;
       Pt3dr       _d2;
       Pt3dr       _d3;
 
       Boite();
       Boite(Pt3dr p0,Pt3dr p1,Pt3dr p2,Pt3dr p3,Pt3dr pas);
       Boite(Facette_2d f, REAL z_min, REAL z_max, Pt3dr pas);
       Boite(Facette_3d f,Pt3dr pas,bool vertical=true);
       Boite(Boite& b);
       Boite(Boite& b, Pt3dr orig, Pt3di taille);
       Boite(Mne& mne);

       bool        inside(Pt3di p){return (p.x > 0 && p.y  > 0 && p.z > 0 && p.x < tx() && p.y < ty() && p.z < tz());}
       
       Im3D_U_INT1 boite(){return _boite;};
       U_INT1      boite(INT x,INT y,INT z){return _boite.data()[z][y][x];};
       U_INT1      boite(Pt3di p){return _boite.data()[p.z][p.y][p.x];};
       void        boite(INT x,INT y,INT z,U_INT1 v){_boite.data()[z][y][x]=v;};
       void        boite(Pt3di p,U_INT1 v){_boite.data()[p.z][p.y][p.x]=v;};
       U_INT1      boite_bicube_h(Pt3dr p);
       
       Im2D_U_INT1 masq(){return _masq;};
       U_INT1      masq(INT x,INT y){return _masq.data()[y][x];};
       U_INT1      masq(Pt2di p){return _masq.data()[p.y][p.x];};
       void        masq(INT x,INT y, U_INT1 v){_masq.data()[y][x]=v;};
       void        masq(Pt2di p, U_INT1 v){_masq.data()[p.y][p.x]=v;};
       void        masq(Im2D_U_INT1 masq){ELISE_COPY(_masq.all_pts(),masq.in(0),_masq.out());};
       
       INT   tx(){return _boite.tx();};
       INT   ty(){return _boite.ty();};
       INT   tz(){return _boite.tz();};
       Pt3dr p0(){return _p0;};
       Pt3dr pas(){return _pas;};
       Pt3dr d1(){return _d1;};
       Pt3dr d2(){return _d2;};
       Pt3dr d3(){return _d3;};
       
       bool               terrain_to_boite(Pt3dr,Pt3di&);
       Pt3dr              terrain_to_boite(Pt3dr);
       Facette_3d         terrain_to_boite(Facette_3d f, bool real_int = false);
       Facette_2d         terrain_to_boite(Facette_2d f, bool real_int = false);
       ElList<Facette_3d> terrain_to_boite(ElList<Facette_3d> f);
       void               terrain_to_boite(ElFilo<Facette_3d> entree, ElFilo<Facette_3d> sortie);
       Hdroite            terrain_to_boite(Hdroite dr);
       Hplan              terrain_to_boite(Hplan pl);
       void               terrain_to_boite(H_Graphe& graphe, bool real_to_int = true);
       
       bool           boite_to_terrain(Pt3di,Pt3dr&);
       Pt3dr          boite_to_terrain(Pt3dr);
       Facette_3d     boite_to_terrain(Facette_3d f);
       Hdroite        boite_to_terrain(Hdroite dr);
       Hplan          boite_to_terrain(Hplan pl);
       ElList<Hplan>  boite_to_terrain(ElList<Hplan> pl);
       void           boite_to_terrain(ElFilo<Facette_3d> entree, ElFilo<Facette_3d> sortie);
       void           boite_to_terrain(H_Graphe& graphe);
       
       void   charger(Wcor, U_INT1 (* ree_f)(U_INT1**, INT, INT ,Pt2dr) );
       void   charger(Wcor, Im2D_U_INT1 im, U_INT1 (* ree_f)(U_INT1**, INT, INT ,Pt2dr) );

       void   initialiser(U_INT1 v=0);
       U_INT1 get(INT x,INT y,INT z){return _boite.data()[z][y][x];}; 
       void   put(INT x,INT y,INT z,U_INT1 v){_boite.data()[z][y][x]=v;};
       
       ElList<Pt3di> intersect(Boite& b);
       
       static void correl(Boite& b_g, Boite& b_d, Boite& b_c, INT semi_fenet,INT dif_ng=40); 
              void correl(Boite& b, Boite& b_c, INT semi_fenet,INT dif_ng=40 )
                         {correl(*this,b,b_c,semi_fenet,dif_ng);};
 
       static void eqrt_type(Boite& b,Boite& b_e, INT semi_fenet = 1);
              void eqrt_type(Boite& b_e, INT semi_fenet = 1)
                            {eqrt_type(*this,b_e, semi_fenet);};

       REAL   esperence();
       REAL   eqrt();
       REAL   covariance(Boite& b);
       void   histograme(Output out, INT coulor,  INT x, INT y);
       void   histograme_x(Output out, INT coulor,  INT x, INT y);
       void   histograme_y(Output out, INT coulor,  INT x, INT y);
       void   histograme_z(Output out, INT coulor,  INT x, INT y);

       void   prog_dyn(REAL par1=.16, REAL par2=1.5, REAL alpha = .75);
       void   prog_dyn(Mne& mne, REAL seuil = 0.5, REAL par1=.16, REAL par2=1.5, REAL alpha = .75);

       void                 get_mne(Mne& mne, REAL seuil = 0.5); 
       
       REAL scor_segment(Pt3dr p0, Pt3dr p1);
       REAL scor_contour(Facette_3d f, bool local=true, bool vertical=false);
       REAL scor_surface(Facette_3d f);
       INT  facette_surface(Facette_3d f);
       INT  facette_surface_interne(Facette_3d f);
       
       Facette_3d recalage_xyz(Boite& grad_g, Boite& grad_d, Facette_2d f, REAL dxy, REAL dz, REAL z_init);
       Facette_3d recalage_xyz(Boite& grad_d, Facette_2d f, REAL dxy, REAL dz, REAL z_init)
                               {return recalage_xyz(*this,grad_d,f,dxy,dz,z_init);};
       
       Facette_3d recalage_z(Boite& grad_g, Boite& grad_d, Facette_3d f, REAL dz, REAL z_init);
       Facette_3d recalage_z(Boite& grad_d, Facette_3d f, REAL dz, REAL z_init)
                               {return recalage_z(*this,grad_d,f,dz,z_init);};

       void affiche_cube(WcorVis W, INT col = 2);
       void affiche_boite(Output wgry);

       void tracer(Facette_3d f, INT color, bool contour = false);
       void tracer(ElList<Facette_3d> lf, bool terrain = false, bool contour = false, INT coulor = 255);
       void tracer(ElFilo<Facette_3d>& lf, bool terrain = false, bool contour = false, INT coulor = 255);
       void tracer(Facette_3d f, INT color, Im3D_INT4 etiquet, bool contour = false);
       void tracer(ElList<Facette_3d> lf, Im3D_INT4 etiquet, bool terrain = false, bool contour = false);
       void tracer(ElFilo<Facette_3d>& f_f, U_INT1* et_f, Im3D_INT4 etiquet, bool terrain = false, bool contour = false, INT color = -1);
       void tracer(ElFilo<Facette_3d>& f_f, U_INT1* et_f, ElFilo<INT>& nb_f_pl_f, ElFilo<Hplan>& pl_f, Im3D_INT4 etiquet);
       void tracer(Hplan pl, INT color);
       void tracer(Hdroite dr, INT color);

       void max_z();
       void valeur_correspond();
       void surface_sup_inf_continue();

       ElList<REAL> get_perpend_directions(Facette_2d f, REAL dif = 5, REAL long_min = 3);

       void MNE(Hdroite dr);
       Liste_Pts_INT2 flux(Facette_3d f, bool ter = false);
       Liste_Pts_INT2 flux_3D(Facette_3d f, bool ter = false);
       Im2D_U_INT1 facette_to_image(Facette_3d f, bool ter = false);
};


#endif // _HASSAN_BOITE_H

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
