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
#ifndef _HASSAN_FACETTE_H
#define _HASSAN_FACETTE_H

#define DEBUG_HASSAN   1

// template  <class Type> class Tab_CPT_REF;

template  <class Type> class Data_Tab_CPT_REF : public RC_Object
{
     friend class Tab_CPT_REF<Type>;

     public :
     private :

         Data_Tab_CPT_REF(const Type * objects,INT nb);
         Data_Tab_CPT_REF(INT nb);
         virtual ~Data_Tab_CPT_REF();

         Type *   _objects;
         INT        _nb;
         INT        _capa;
         void  push(Type);
};

/*
template  <class Type> class Tab_CPT_REF : public PRC0
{
     public :
      
           Tab_CPT_REF(Type p0,Type p1,Type p2);
           Tab_CPT_REF(Type p0,Type p1,Type p2,Type p3);
           Tab_CPT_REF(const Type *objects,INT nb);
           Tab_CPT_REF(INT nb);

           INT   nb();
           Type & operator [](INT i);
           void  push(Type);

     private :

          inline  Data_Tab_CPT_REF<Type> * dtd();
};
*/

typedef    Tab_CPT_REF<Pt2dr>      Facette_2D;


/////////////////////////////////////////////////////////////////////////////////

class Facette_3d;
class Facette_2d;

class Data_facette_3d : public RC_Object
{
     friend class Facette_3d;

     public :
     private :

         Data_facette_3d(const Pt3dr * objects,INT nb);
         Data_facette_3d(INT nb);
         virtual ~Data_facette_3d();

         Pt3dr *  _objects;
         INT      _nb;
         INT      _capa;
         Pt3dr    _norm;
         bool     _statu;
         bool     _visu;
         U_INT1   _coulor;
         REAL     _poids;
         INT      _refplan;
         void     push(Pt3dr);
         void     poids(REAL pds){_poids = pds;}
         REAL     poids(){return _poids;}
};

class Facette_3d : public PRC0
{
     public :

           Facette_3d();
           Facette_3d(Pt3dr p0,Pt3dr p1,Pt3dr p2);
           Facette_3d(Pt3dr p0,Pt3dr p1,Pt3dr p2,Pt3dr p3);
           Facette_3d(const Pt3dr *objects,INT nb);
           Facette_3d(Facette_2D f2d, REAL4 z=0);
           Facette_3d(Facette_2d f2d, REAL4 z=0);
           Facette_3d(INT nb);

           Facette_3d trans(Facette_2D f2d, REAL4 z=0);
           Facette_3d trans(Facette_2d f2d, REAL4 z=0);
           
           Facette_2D to_2D();
           Facette_2d to_2d();

           void set(Pt3dr p0,Pt3dr p1,Pt3dr p2);
           void set(Pt3dr p0,Pt3dr p1,Pt3dr p2,Pt3dr p3);
           void set(const Pt3dr *objects,INT nb);
           void set(INT);
           void set(INT, Pt3dr v);
           void set(U_INT1 coul);
           bool test();
           bool test_v(Pt3dr ray);
           void normer();
           void initialiser();
           bool intersect_ray(Pt3dr ray, Pt3dr p, REAL4& t);
           bool test_into(Pt3dr pi);
           ElList<Facette_3d> intersection(Facette_3d f);
           ElList<Facette_3d> intersection(ElList<Facette_3d> lf);
           void intersection(ElFilo<Facette_3d>& f_f);

           void chc(Pt3dr trans);
           void rot(Pt3dr per);
           void rot(REAL4 teta);


           REAL surface();               //cas facette plane
           Pt3dr centre_gravite();

//           Facette_3d* ptr();
           INT   nb();
           Pt3dr  norm();
           bool  statu();
           bool  visu();
           U_INT1 coulor();
           void setrefplan(INT ref);
           INT  refplan();

           // void  poids(REAL pds){dtd()->poids(pds);}
           // REAL  poids()        {return dtd()->poids();}


           Pt3dr & operator [](INT i);
		   std::ostream& operator >>(std::ofstream& os);
		   std::istream& operator <<(std::istream& is);
           void  push(Pt3dr);

     private :

          // inline  Data_facette_3d * dtd();
};


///////////////////////////////////////////////////////////////////////////////// 
class Facette
{
    public :
      Pt3dr* point;
      INT    nb;
      Pt3dr  norm;
      bool   statu;
      bool   visu;
      U_INT1 coulor;

      Facette();
      Facette(Pt3dr p0, Pt3dr p1, Pt3dr p2);
      Facette(Pt3dr p0, Pt3dr p1, Pt3dr p2, Pt3dr p3);
      Facette(INT, Pt3dr*);
      void set(Pt3dr, Pt3dr, Pt3dr);
      void set(Pt3dr, Pt3dr, Pt3dr, Pt3dr);
      void set(INT, Pt3dr*);
      void set(INT);
      void set(INT, Pt3dr);
      void set_coulor(U_INT1 coul){coulor = coul;}
      bool test();
      bool test_v(Pt3dr);
      void normer();
      bool intersect_ray_plan(Pt3dr, Pt3dr, Pt3dr&, REAL4&);
      bool intersect_ray(Pt3dr, Pt3dr, REAL4&);
      bool test_into(Pt3dr);
      virtual ~Facette(){ delete [] point; }
          
};

/////////////////////////////////////////////////////////////////////////////////

class Facette_2d;

class Data_facette_2d : public RC_Object
{
     friend class Facette_2d;

     public :
     private :

         Data_facette_2d(const Pt2dr * objects,INT nb);
         Data_facette_2d(INT nb);
         virtual ~Data_facette_2d();

         Pt2dr *  _objects;
         INT      _nb;
         INT      _capa;
         void     push(Pt2dr);
};

class Facette_2d : public PRC0
{
     public :

           Facette_2d();
           Facette_2d(Pt2dr p0,Pt2dr p1,Pt2dr p2);
           Facette_2d(Pt2dr p0,Pt2dr p1,Pt2dr p2,Pt2dr p3);
           Facette_2d(const Pt2dr *objects,INT nb);
           Facette_2d(Facette_2D f2d);
           Facette_2d(INT nb);

           INT   nb();
           Pt2dr & operator [](INT i);
           void  push(Pt2dr);
           Facette_2D to_2D();
           ElList<REAL> get_perpend_directions(REAL dif = 5, REAL long_min = 3);

           bool if_into(Pt2dr);
           bool if_intersect(Facette_2d f, REAL petit_surf = .00001);

     private :

          inline  Data_facette_2d * dtd();
};

#include "gpc.h"
void gpc_polygon_to_facette_2d(gpc_polygon& p, ElFilo<Facette_2d>& f_f);
void facette_2d_to_gpc_polygon(Facette_2d f, gpc_polygon& p);

class Hplan;
extern void partage_facette_3d(Facette_3d f, Hplan plan, ElList<Facette_3d>& l_f);

#endif // _HASSAN_FACETTE_H

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
