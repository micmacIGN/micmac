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
#ifndef _ELISE_GENERAL_HASSAN_3D_H
#define _ELISE_GENERAL_HASSAN_3D_H
//#include<values.h>

class Hdroite;

class Data_Hdroite : public RC_Object
{
   friend class Hdroite;

   public :
   private:
   
      Data_Hdroite();
      Data_Hdroite(Pt3dr p0, Pt3dr p1);
      virtual ~Data_Hdroite();
      
      Pt3dr _p0;
      Pt3dr _p1;
      Pt3dr _ray;
      bool  _nu;
      REAL  _dist;
      
      void p0(Pt3dr p);
      void p1(Pt3dr p);
      void inv_ray();
};

class Hdroite : public PRC0
{
   public :
   
      Hdroite();
      Hdroite(Pt3dr p0, Pt3dr p1);
      
      Pt3dr p0();
      Pt3dr p1();
      Pt3dr ray();
      bool  empty();
//      Hdroite* ptr();
      
      REAL         distance();
      REAL         distance(Pt3dr p);
      static REAL  distance(Hdroite& d1, Hdroite& d2);
      REAL         distance(Hdroite& d);
      
      bool         appartenir_drt(Pt3dr p, REAL precision);
      bool         appartenir_seg(Pt3dr p, REAL precision);
      bool         espace_seg(Pt3dr p);
      
      Pt3dr        project_pt_drt(Pt3dr p);
      Pt3dr        rayon_ort(Hdroite& d);
      Hdroite      droite_ort(Hdroite& d);
      
      void         trans_p0(REAL dist);
      void         trans_p1(REAL dist);
      void         trans(REAL dist);
      void         trans(Pt3dr dif);
      void         trans(Pt3dr& p,REAL dist);
      
      void         rot(Pt3dr& p, REAL alpha);   //alpha en degree
      void         rot(Hdroite& d, REAL alpha);
      
      Pt3dr         coord(REAL t);
      ElList<Pt3di> flux();
      Liste_Pts_INT2 flux_el();
      
      bool         inter_drt(Hdroite& d, REAL dist, Pt3dr& p);
      bool         inter_seg(Hdroite& d, REAL dist, Pt3dr& p);
      bool         intersect(Facette_2d f);
      bool         intersect(Facette_3d f);
      ElList<Facette_2d> partition_z(Facette_2d f);
      Facette_2d         associer(Facette_2d f);
      
   private:
   
      inline  Data_Hdroite * dtd();
};

//////////////////////////////////////////////////////////////////////////

class Hplan;

class Data_Hplan : public RC_Object
{
   friend class Hplan;

   public :
   private:
   
      Data_Hplan();
      Data_Hplan(Pt3dr p0, Pt3dr p1, Pt3dr p2, bool ort=true);
      Data_Hplan(Facette_3d f, bool ort=true);
      virtual ~Data_Hplan();
      
      Pt3dr _p0;
      Pt3dr _p1;
      Pt3dr _p2;
      Pt3dr _r_x;
      Pt3dr _r_y;
      Pt3dr _r_z;
      REAL  _poids;
      bool  _nu;
      
      void trans(Pt3dr p);
};


class Hplan : public PRC0
{
   public :
   
      Hplan();
      Hplan(Pt3dr p0, Pt3dr p1, Pt3dr p2, bool ort=true);
      Hplan(Pt3dr pt, Pt3dr normal);

      Hplan(REAL A, REAL B, REAL C, REAL D); //, bool ort=true);
      Hplan(REAL teta, REAL phi, REAL rho);                //cos(t)cos(p)x+sin(t)cos(p)y+sin(p)z-rho=0   radian

      Hplan(Facette_3d f);
      
      Pt3dr p0();
      Pt3dr p1();
      Pt3dr p2();
      Pt3dr ray_x();
      Pt3dr ray_y();
      Pt3dr ray_z();
      REAL  teta();
      REAL  phi();
      REAL  rho();
      REAL  A();
      REAL  B();
      REAL  C();
      REAL  D();
      bool  empty();
      // REAL  poids(){return dtd()->_poids;}
      // void set_poids(REAL poids){dtd()->_poids = poids;}
//      Hplan* ptr();
      
      REAL         distance(Pt3dr p);                         //positive ou negative
      REAL         distance(Facette_3d f);                    //distance moyenne
      REAL         distance_max(Facette_3d f);                    //distance moyenne
      
      bool         appartenir_plan(Pt3dr p, REAL precision);
      bool         into_trian(Pt3dr p);                       //l'appartenance est deja testee
      bool         appartenir_trian(Pt3dr p, REAL precision);
      bool         espace_trian(Pt3dr p);
      
      Pt3dr        project_pt_plan(Pt3dr p);
      
      void         trans(Pt3dr dif);                           //dans l'espace
      void         trans(REAL dist1, REAL dist2);              //dans le plan
      void         rot_x(REAL alpha);                          //alpha en degree
      void         rot_y(REAL alpha);              
      void         rot_z(REAL alpha);              
      void         rot_d(Hdroite d, REAL alpha);
      bool         balancer(Hdroite d, REAL alpha);
      
      Pt3dr         coord(Pt2dr p);                            // plan to espace
      Pt2dr         coord(Pt3dr p);                            //appartenance stisfaite
      ElList<Pt2dr> coord(ElList<Pt3dr> lp); 
      Facette_2d    coord(Facette_3d f);
      bool          coord_x(Pt3dr& p); 
      bool          coord_y(Pt3dr& p); 
      bool          coord_z(Pt3dr& p);
      
      bool          coord_x(ElList<Pt3dr> lp);
      bool          coord_y(ElList<Pt3dr> lp);
      bool          coord_z(ElList<Pt3dr> lp);
      
      Liste_Pts_INT2 flux();
      Liste_Pts_INT2 flux(Facette_3d f, bool contour = false);
      
      bool          intersect(Hdroite d, Pt3dr& p);
      bool          intersect(Hdroite d, Pt2dr& p);
      bool          inter_trian(Hdroite d, Pt3dr& p);
      
      bool          conforme(Hplan& pl){ return ( ptr() == pl.ptr() ); }

      Facette_3d    projection( Facette_2d f, Hdroite drt, REAL z = 0.);

      bool               intersect(Hplan& pl, Hdroite& d);
      ElList<Facette_3d> partage(Facette_3d f);
      ElList<Facette_3d> partage(ElList<Facette_3d> lf);
      void               partage(ElFilo<Facette_3d>& filo_f);
      INT                haut_milieu_bas(Facette_3d f);       // 1 : haut, 2 : milieu, 3 : bas
      ElList<Facette_3d> haut(ElList<Facette_3d> lf);
      ElList<Facette_3d> milieu(ElList<Facette_3d> lf);
      ElList<Facette_3d> bas(ElList<Facette_3d> lf);
      bool               touch(Facette_3d f, REAL dist = 0);
      bool               intersect(Facette_3d f);
      ElList<Facette_3d> no_touch(ElList<Facette_3d> lf);

      REAL               angle_normale(Hplan& pl);
      
   private:
   
       // inline  Data_Hplan * dtd();
};

//////////////////////////////////////////////////////////////////////////////

class Hcamera
{
   public : 
   
      Hplan        _plan;
      Hplan        _plan_init;
      Im2D_U_INT1  _iphoto;
      Im2D_REAL8   _z_buf;
      REAL         _dist_focal;
      REAL         _resolu_film;
      Pt2dr        _p0ph;
      Pt3dr        _cent_opt;
      Pt3dr        _trans;
 
      Hcamera();
      Hcamera(Hplan pl, Pt2di size, REAL resolu_film = .0003);
      
      void        resolution(REAL r){_resolu_film = r;}
      
      void        zoom(REAL zo){_dist_focal *= (1+zo);_cent_opt = _plan.p0() - _plan.ray_z() * _dist_focal;}
      void        trans_x(REAL dx){_trans.x +=dx;_plan.trans(Pt3dr(dx,0,0));} 
      void        trans_y(REAL dy){_trans.y +=dy;_plan.trans(Pt3dr(0,dy,0));}
      void        trans_z(REAL dz){_trans.z +=dz;_plan.trans(Pt3dr(0,0,dz));}
      void        trans(Pt3dr dif){_trans = _trans + dif;_plan.trans(dif);}
      void        trans_phot(Pt2di dif){_p0ph = _p0ph + (Pt2dr)dif * _resolu_film;}   
      void        rot_x(REAL alpha){_plan.rot_x(alpha);}    
      void        rot_y(REAL alpha){_plan.rot_y(alpha);}     
      void        rot_z(REAL alpha){_plan.rot_z(alpha);} 
                    
      Im2D_U_INT1 photo(){return _iphoto;}
      INT         tx(){return _iphoto.tx();} 
      INT         ty(){return _iphoto.ty();} 
      Pt3dr       trans(){return _trans;}
      Pt2di       trans_phot(){return _p0ph;} 
      REAL        resolution(){return _resolu_film;}       
      
      void effacer();
      Pt3dr projter(Pt3dr p, U_INT1 v);
      void projter(Facette_3d f,U_INT1 v, bool lin=true);
      void projter(Facette_3d f,Boite b, bool lin=true);
      void afficher(Video_Win); 
};


/////////////////////////////////////////////////////////////////////////////////////////////////
class Camera
{
   public :
      
      Camera(  Video_Win W, INT nb_color, ElList<Pt3dr> pts, bool avant_arrier = false);
      Camera(  Video_Win W, INT nb_color, ElList<Facette_3d> lface, bool avant_arrier = false);
      Camera(  Video_Win W, INT nb_color, ElFilo<Facette_3d>& lface, bool avant_arrier = false);

   private :
      
      REAL _m00, _m01, _m02, _m03;
      REAL _m10, _m11, _m12, _m13;
      REAL _m20, _m21, _m22, _m23;
      REAL _m30, _m31, _m32, _m33;
      REAL _zoom;
      REAL _teta;
      REAL _phi;
      REAL _rho;
      REAL _d_zoom;
      REAL _d_teta;
      REAL _d_phi;
      REAL _d_rho;
      REAL _x_c;
      REAL _y_c;
      REAL _z_c;
      INT  _tx_1_2;
      INT  _ty_1_2;
      bool _avant_arrier;

      ElList<Pt3dr> _donnees;
      ElList<Facette_3d> _lface;

      ElList<Pt2di> _donnees_p;
      ElList<ElList<Pt2di> > _ll;

      INT       _nb_color;
      Video_Win _W;

                                      // mouvement de camera

      void zoom_up();
      void zoom_down(); 
      void tourne_teta_up();
      void tourne_teta_down();
      void tourne_phi_up();
      void tourne_phi_down();
      void approche_up();
      void approche_down();
      void tourne_tete_teta_up();
      void tourne_tete_teta_down();
      void tourne_tete_phi_up();
      void tourne_tete_phi_down();

                                        //parametre de camera

      void calcul_matrice();
      void reperer();
      void zoom_adjuster();
      Pt2di projeter(Pt3dr p) {
                                 return Pt2di(
                                                (INT)(_m00 * p.x + _m01 * p.y + _m02 * p.z + _m03),
                                                (INT)(_m10 * p.x + _m11 * p.y + _m12 * p.z + _m13)
                                             );
                              }

                                        // afficahge et desaffichage

      void afficher_pts();
      void afficher_face( INT color = 2);
      void afficher();
      void desafficher_pts();
      void desafficher_face();
      void desafficher();
      void reafficher();
      void interface();
};



std::ostream & operator << (std::ostream & os, Hplan & f);
std::istream & operator >> ( std::istream &is, Hplan & f);



#endif

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
