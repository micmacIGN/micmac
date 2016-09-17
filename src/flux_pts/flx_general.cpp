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



/*********************************************************************/
/*                                                                   */
/*         Arg_Flux_Pts_Comp                                         */
/*                                                                   */
/*********************************************************************/

Arg_Flux_Pts_Comp::Arg_Flux_Pts_Comp() :
      _sz_buf (Elise_Std_Max_Buf) 
{
}

Arg_Flux_Pts_Comp::Arg_Flux_Pts_Comp(INT SzBuf) :
      _sz_buf (SzBuf) 
{
}


/*********************************************************************/
/*                                                                   */
/*         Pack_Of_Pts                                               */
/*                                                                   */
/*********************************************************************/

Pack_Of_Pts::~Pack_Of_Pts(){}

Pack_Of_Pts::Pack_Of_Pts(INT dim,INT sz_buf,type_pack type) :
    _nb     (0)     ,
    _dim    (dim)   ,
    _sz_buf (sz_buf),
    _type   (type)
{
}


void * Pack_Of_Pts::adr_coord() const
{
     elise_internal_error
      ("should never be in Pack_Of_Pts::adr_coord",__FILE__,__LINE__);
    return 0;
}


Pack_Of_Pts * Pack_Of_Pts::new_pck(INT dim,INT sz_buf,type_pack type)
{
    switch (type)
    {
         case rle :
              return RLE_Pack_Of_Pts::new_pck(dim,sz_buf);
         case integer :
              return Std_Pack_Of_Pts<INT>::new_pck(dim,sz_buf);
         case real :
              return Std_Pack_Of_Pts<REAL>::new_pck(dim,sz_buf);
         default :
               elise_fatal_error
               ("incoherence in Pack_Of_Pts::Pack_Of_Ptsnew_pck",
                __FILE__,__LINE__);
               return 0;
    }
}


void Pack_Of_Pts::convert_from(const Pack_Of_Pts * pts) 
{
     switch (pts->_type)
     {
            case rle :
                 convert_from_rle
                 (
                     SAFE_DYNC(const RLE_Pack_Of_Pts *,pts)
                 );
            return;

            case integer :
                 convert_from_int
                 (
                     SAFE_DYNC(const Std_Pack_Of_Pts<INT> *,pts)
                 );
            return;

            case real :
                 convert_from_real
                 (
                     SAFE_DYNC(const Std_Pack_Of_Pts<REAL> *,pts)
                 );
            return;

     }
}

void Pack_Of_Pts::convert_from_rle(const RLE_Pack_Of_Pts *)
{
     elise_internal_error
     (
          "Impossible convertions of Pack_Of_Pts",
           __FILE__,
           __LINE__
     );
}

void Pack_Of_Pts::convert_from_int(const Std_Pack_Of_Pts<INT>  *) 
{
     elise_internal_error
     (
          "Impossible convertions of Pack_Of_Pts",
           __FILE__,
           __LINE__
     );
}

void Pack_Of_Pts::convert_from_real(const Std_Pack_Of_Pts<REAL>  *)
{
     elise_internal_error
     (
          "Impossible convertions of Pack_Of_Pts",
           __FILE__,
           __LINE__
     );
}

Pack_Of_Pts::type_pack Pack_Of_Pts::type_common(type_pack t1,type_pack t2)
{
    ASSERT_INTERNAL
    (
          (rle < integer) && (integer < real),
          "Bad odrder for type in Pack_Of_Pts"
    );

    return (type_pack) ElMax((INT)t1,(INT)t2);
}




/***************************************************************/
/*                                                             */
/*          Flux_Pts  : constructors                           */
/*                                                             */
/***************************************************************/


       // -----------  constructors  ---------------------



Flux_Pts::Flux_Pts(Flux_Pts_Not_Comp * flx) :
    PRC0(flx)
{
}
                  // see also at end of this file

      //-------------------------------------------------

Flux_Pts_Computed * Flux_Pts::compute (const Arg_Flux_Pts_Comp & arg)
{
    ASSERT_TJS_USER
    (    _ptr != 0,
       "USE OF Flux_Pts  non initialized"
    );

    return SAFE_DYNC(Flux_Pts_Not_Comp *,_ptr)->compute(arg);
}


/***************************************************************/
/*                                                             */
/*          Flux_Pts_Convertion                                */
/*                                                             */
/***************************************************************/

class Flux_Pts_Convertion : public Flux_Pts_Computed
{
      public :
          Flux_Pts_Convertion
          (
                 Flux_Pts_Computed *,
                 Pack_Of_Pts::type_pack   
          );
          virtual ~Flux_Pts_Convertion() 
          { 
                 delete    _flx;
                 delete _pack_res;
          }


     private :

          Flux_Pts_Computed *       _flx;
          //Pack_Of_Pts::type_pack    _type;
          Pack_Of_Pts *             _pack_res;


          const Pack_Of_Pts * next()
          {
                const Pack_Of_Pts * pck = _flx->next();
                if (! pck) return 0;
                _pack_res->convert_from(pck);
                return _pack_res;
          }
};

Flux_Pts_Convertion::Flux_Pts_Convertion
(
       Flux_Pts_Computed * flx ,
       Pack_Of_Pts::type_pack type
)   :
        Flux_Pts_Computed   (flx->dim(),type,flx->sz_buf()),
        _flx                (flx),
        _pack_res           (Pack_Of_Pts::new_pck(dim(),sz_buf(),type))
{
}

/***************************************************************/
/*                                                             */
/*          Flux_Pts_Computed                                  */
/*                                                             */
/***************************************************************/



Flux_Pts_Computed::~Flux_Pts_Computed()
{ 
}


Flux_Pts_Computed::Flux_Pts_Computed
   (  INT                        dim,
      Pack_Of_Pts::type_pack     type,
      INT                        sz_buf
   ) :
     _dim    (dim),
     _type   (type),
     _sz_buf (sz_buf)
{
}

bool Flux_Pts_Computed::is_line_map_rect()
{
   return false;
}

bool Flux_Pts_Computed::is_rect_2d(Box2di &)
{
   return false;
}


REAL  Flux_Pts_Computed::average_dist()
{
   elise_internal_error
       ("use of average_dist in a non linear flux",__FILE__,__LINE__);
   return 0.0;  // completely arbitrary
}

Flux_Pts_Computed * Flux_Pts_Computed::convert(Pack_Of_Pts::type_pack type) 
{
     if (type == _type)
        return this;
     else
        return new  Flux_Pts_Convertion(this,type);
}

void Flux_Pts_Computed::type_common
    (
             Flux_Pts_Computed ** flx1,
             Flux_Pts_Computed ** flx2
    )
{
    Pack_Of_Pts::type_pack  type
          = Pack_Of_Pts::type_common
            (
                (*flx1)->type(),
                (*flx2)->type()
            );

    (*flx1) = (*flx1)->convert(type);
    (*flx2) = (*flx2)->convert(type);
}

/***************************************************************/
/*                                                             */
/*          RLE_Flux_Pts_Computed                              */
/*                                                             */
/***************************************************************/


RLE_Flux_Pts_Computed::RLE_Flux_Pts_Computed(INT dim,INT sz_buf) :
      Flux_Pts_Computed (dim,Pack_Of_Pts::rle,sz_buf),
      _rle_pack  (RLE_Pack_Of_Pts::new_pck(dim,sz_buf))
{
}


RLE_Flux_Pts_Computed::~RLE_Flux_Pts_Computed()
{
     delete _rle_pack;
}

/***************************************************************/
/*                                                             */
/*          Elise_Rect                                         */
/*                                                             */
/***************************************************************/

Box2dr Elise_Rect::ToBoxR(void) const
{
    ELISE_ASSERT(_dim==2,"Bad Dim in  Elise_Rect::ToBoxR");
    return Box2dr
           (
                Pt2dr(_p0[0],_p0[1]),
                Pt2dr(_p1[0],_p1[1])
           );
}


Elise_Rect::Elise_Rect(const INT * p0,const INT * p1, INT dim)
{
   convert(_p0,p0,dim);
   convert(_p1,p1,dim);
   _dim = dim;
}

Elise_Rect::Elise_Rect(Pt2di p0,Pt2di p1)
{
    INT t0[2],t1[2];
    p0.to_tab(t0);
    p1.to_tab(t1);

    *this = Elise_Rect(t0,t1,2);
}



Flux_Pts Rectang_Object::all_pts()  const
{
      Elise_Rect r = box();
      return rectangle (r._p0,r._p1,r._dim);
}

Flux_Pts Rectang_Object::interior(INT sz)  const
{
      Elise_Rect r = box();
      sz = ElAbs(sz);
      for (INT d=0 ; d<r._dim;d++)
      {
          r._p0[d] += sz;
          r._p1[d] -= sz;
      }
      return rectangle (r._p0,r._p1,r._dim);
}


Flux_Pts Rectang_Object::border(INT b)  const
{
      Elise_Rect r = box();
      return border_rect (r._p0,r._p1,b,r._dim);
}


INT Rectang_Object::x0() const 
{
      Elise_Rect r = box();
      ASSERT_USER((r._dim == 1),"dim != 1 for Rectang_Object::x0");
      return r._p0[0];
}

INT Rectang_Object::x1() const 
{
      Elise_Rect r = box();
      ASSERT_USER((r._dim == 1),"dim != 1 for Rectang_Object::x1");
      return r._p1[0];
}


Pt2di Rectang_Object::p0() const 
{
      Elise_Rect r = box();
      ASSERT_USER
      (
         (r._dim == 2),
         "dim != 2 for Rectang_Object::p0"
      );
      return Pt2di(r._p0[0],r._p0[1]);
}

Pt2di Rectang_Object::p1() const 
{
      Elise_Rect r = box();
      ASSERT_USER
      (
         (r._dim == 2),
         "dim != 2 for Rectang_Object::p0"
      );
      return Pt2di(r._p1[0],r._p1[1]);
}


Fonc_Num Rectang_Object::inside()  const
{
      Elise_Rect r = box();
      return ::inside(r._p0,r._p1,r._dim);
}

Flux_Pts Rectang_Object::lmr_all_pts(Pt2di dir) const 
{
      Elise_Rect r = box();
      ASSERT_USER((r._dim == 2),"dim != 2 for lmr_all_pts(Pt2di dir)");
      return line_map_rect(dir,p0(),p1());
}


Flux_Pts Rectang_Object::lmr_all_pts(INT dir) const 
{
      Elise_Rect r = box();
      ASSERT_USER((r._dim == 1),"dim != 1 for lmr_all_pts(INT dir)");
      return line_map_rect(dir,x0(),x1());
}

Rectang_Object::~Rectang_Object() {}

/**************************************************************/
/*                                                            */
/*    Flux Pts decribed by user coordinates                   */
/*                                                            */
/**************************************************************/

class Flux_Pts_Coord_user_Comp : public Flux_Pts_Computed
{
   public :
          Flux_Pts_Coord_user_Comp(const Arg_Flux_Pts_Comp &,Pack_Of_Pts * pack);
          Pack_Of_Pts  * _pack;

         const Pack_Of_Pts * next()
         {
              Pack_Of_Pts * p = _pack;
              _pack = 0;
              return p;
         }

};

Flux_Pts_Coord_user_Comp::Flux_Pts_Coord_user_Comp
      (const Arg_Flux_Pts_Comp &,Pack_Of_Pts * pack) :
          Flux_Pts_Computed(pack->dim(),pack->type(),pack->pck_sz_buf()),
          _pack (pack)
{
}

class Flux_Pts_Coord_user_Not_Comp : public Flux_Pts_Not_Comp
{
   public :
        virtual  ~Flux_Pts_Coord_user_Not_Comp(){ delete _pack; }

        Flux_Pts_Coord_user_Not_Comp(Pack_Of_Pts *,bool to_dup);

        Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp & arg)
        {
            Flux_Pts_Computed * res = 
                new Flux_Pts_Coord_user_Comp(arg,_pack);

           return split_to_max_buf(res,arg);
        }

   private :
        Pack_Of_Pts  * _pack;
        
};

Flux_Pts_Coord_user_Not_Comp::Flux_Pts_Coord_user_Not_Comp(Pack_Of_Pts * pack,bool to_dup) :
     _pack (to_dup ? pack->dup() : pack)
{
}



Flux_Pts::Flux_Pts(ElList<Pt2di> l) :
    PRC0(new Flux_Pts_Coord_user_Not_Comp(lpt_to_pack(l),false))
{
}

Flux_Pts::Flux_Pts(Pt2di p) :
    PRC0(new Flux_Pts_Coord_user_Not_Comp(lpt_to_pack(ElList<Pt2di>()+p),false))
{
}



/**************************************************************/
/**************************************************************/
/**************************************************************/
/***                                                        ***/
/***  Chang of coordinates on flux of poinst                ***/
/***                                                        ***/
/**************************************************************/
/**************************************************************/
/**************************************************************/



        /*  Flux_Chc_Comp  */

class Flux_Chc_Comp :  public Flux_Pts_Computed
{
   public :

     Flux_Chc_Comp
     (
             const Arg_Flux_Pts_Comp &,
             Flux_Pts_Computed * _flx,
             Fonc_Num_Computed  *_f
     );

     virtual  ~Flux_Chc_Comp()
     {
           delete _f;
           delete _flx;
     }


   private :


      Flux_Pts_Computed * _flx;
      Fonc_Num_Computed * _f;

      const Pack_Of_Pts * next()
      {
           const Pack_Of_Pts * p = _flx->next();

          if (p)
             return _f->values(p);
          else
             return 0;
      }
};

Flux_Chc_Comp::Flux_Chc_Comp
(
   const Arg_Flux_Pts_Comp & arg,
   Flux_Pts_Computed *       flx,
   Fonc_Num_Computed *       f
)  :

   Flux_Pts_Computed(f->idim_out(),f->type_out(),arg.sz_buf()),
   _flx (flx),
   _f   (f)
{
}


        /*  Flux_Chc_Comp  */

class Flux_Chc_Not_Comp :  public Flux_Pts_Not_Comp
{
      public :

          Flux_Chc_Not_Comp(Flux_Pts,Fonc_Num);

      private :

          Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp & arg)
          {
               Flux_Pts_Computed * flx_c = _flx.compute(arg);
               Fonc_Num_Computed * f_c =   _f.compute(Arg_Fonc_Num_Comp(flx_c));

               return new Flux_Chc_Comp(arg,flx_c,f_c);
          }

          Flux_Pts   _flx;
          Fonc_Num   _f;
};


Flux_Chc_Not_Comp::Flux_Chc_Not_Comp(Flux_Pts flx,Fonc_Num f) :
      _flx    (flx),
      _f      (f  )
{
}



Flux_Pts Flux_Pts::chc(Fonc_Num f)
{
     return new Flux_Chc_Not_Comp(*this,f);
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
