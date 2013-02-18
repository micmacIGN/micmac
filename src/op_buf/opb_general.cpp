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
/*         Arg_FNOPB                                                 */
/*                                                                   */
/*********************************************************************/

Arg_FNOPB::Arg_FNOPB(Fonc_Num f,Box2di side,GenIm::type_el Type) :
      _f        (f),
      _side     (side),
      _type     (Type)
{
}

Arg_FNOPB::Arg_FNOPB(Fonc_Num f,Box2di side) :
      _f        (f),
      _side     (side),
      _type     (  (f.really_a_fonc() && f.integral_fonc(true)) ? 
                    GenIm::int4                                 : 
                    GenIm::real8
                )
{
}

const Arg_FNOPB Arg_FNOPB::def 
                (
                    Fonc_Num::No_Fonc,
                    Box2di(Pt2di(0,0),Pt2di(0,0))
                );

INT Arg_FNOPB::DimF() const
{
   return really_an_arg() ?  _f.dimf_out() : 0;
}


/*********************************************************************/
/*                                                                   */
/*         Buf_Fonc_Op_buf                                           */
/*                                                                   */
/*********************************************************************/

class Buf_Fonc_Op_buf : public Mcheck
{
      public :
		 virtual ~Buf_Fonc_Op_buf();

         friend class Fonc_Num_OP_Buf;

         static Buf_Fonc_Op_buf *
                new_bfob
                (
                       const Arg_Fonc_Num_Comp &,
                       Arg_FNOPB                ,
                       Box2di                    rect,
                       bool                      aWlineInit
                );

         // the arg to buf fonction is completly unused, so one can pass
         // (REAL *) 0; aim od this args is to allow a generic call in templates

         virtual void  *buf_values()  = 0;
         GenIm::type_el type () const {return _type;}
         // virtual bool   integral_buf() const =   0;


         virtual void  convert_data(INT * dest,INT nb,INT dim,INT y,INT x0Src)  = 0;
         virtual void  convert_data(REAL * dest,INT nb,INT dim,INT y,INT x0Src)  = 0;

      protected :

         Buf_Fonc_Op_buf
         (
            Flux_Pts_Computed  * flx,
            Fonc_Num_Computed * ,
            Box2di              rect,
            Box2di              side,
            GenIm::type_el      Type
         );


         Flux_Pts_Computed * _flx_comp;
         Fonc_Num_Computed * _f;
         GenIm::type_el      _type;
         INT                 _sz_buf;
         INT                 _d_out;
         Box2di              _rect;
         Box2di              _side;
         RLE_Pack_Of_Pts *   _pack_f;

         INT mInitX0;
         INT mInitX1;

         INT mInitY0; // version intiales non englobant 0
         INT mInitY1; // version intiales non englobant 0

         INT mY0;
         INT mY1 ;

         // mX0 & mX1 so that the buffering rectangle always includes the
         // image; useful when this buffer is used in line as a temporary result

         INT mX0;
         INT mX1;


      private :

         virtual void  compute_last_line(INT y)=0;

         void          init_comput_line(INT y);
         // first time; do not initialize last line

        

};



Buf_Fonc_Op_buf::Buf_Fonc_Op_buf
(
     Flux_Pts_Computed  * flx,
     Fonc_Num_Computed *  f,
     Box2di               rect,
     Box2di               side,
     GenIm::type_el       Type
)    :
     _flx_comp  (flx),
     _f         (f),
     _type      (Type),
     _sz_buf    (flx->sz_buf()),
     _d_out     (f->idim_out()),
     _rect      (rect),
     _side      (side),
     _pack_f    (RLE_Pack_Of_Pts::new_pck(2,flx->sz_buf())),
     mInitX0        (side._p0.x +rect._p0.x),
     mInitX1        (side._p1.x +rect._p1.x),
     mInitY0    (side._p0.y),
     mInitY1    (side._p1.y+1),
     mY0        (ElMin(0,side._p0.y)),
     mY1        (ElMax(0,side._p1.y)+1),
     mX0       (ElMin(0,side._p0.x) +rect._p0.x),
     mX1       (ElMax(0,side._p1.x) +rect._p1.x)
{
}

void Buf_Fonc_Op_buf::init_comput_line(INT y)
{
     for (INT dy=mY0 -mY1 + 1; dy<0 ; dy++)
         compute_last_line(dy+y);
}

Buf_Fonc_Op_buf::~Buf_Fonc_Op_buf()
{
    delete _pack_f;
    delete _f;
    delete _flx_comp;
}

/*********************************************************************/
/*                                                                   */
/*         Buf_Fonc_OB_TPL<TyData,TyBase>                            */
/*                                                                   */
/*********************************************************************/


template <class TyData,class TyBase> class Buf_Fonc_OB_TPL 
                                           : public Buf_Fonc_Op_buf
{
      public :
         Buf_Fonc_OB_TPL<TyData,TyBase>
         (
            Flux_Pts_Computed  * flx,
            Fonc_Num_Computed *,
            Box2di             rect,
            Box2di             side,
            GenIm::type_el     type,
            bool               wVInit 
         );
          virtual ~Buf_Fonc_OB_TPL();

   
      protected :

      private :
            virtual void  compute_last_line(INT y);


            TyData *** vbuf;  //  vbuf[d][y][x]
            TyData ***   mLineInit;  //  vbuf[d][x]
            static const bool _is_integral;

            virtual void * buf_values()                { return vbuf;}


         virtual void  convert_data( INT * dest,INT nb,INT dim,INT y,INT x0Src) ;
         virtual void  convert_data(REAL * dest,INT nb,INT dim,INT y,INT x0Src);
};


template <class TyData,class TyBase>
         void Buf_Fonc_OB_TPL<TyData,TyBase>::convert_data
         ( 
                INT * dest,
                INT nb,
                INT dim,
                INT y,
                INT x0Src
          )
{
   ::convert(dest,mLineInit[dim][y]+x0Src,nb);
}


template <class TyData,class TyBase>
         void Buf_Fonc_OB_TPL<TyData,TyBase>::convert_data
         ( 
                REAL * dest,
                INT nb,
                INT dim,
                INT y,
                INT x0Src
          )
{
   ::convert(dest,mLineInit[dim][y]+x0Src,nb);
}






template <class TyData,class TyBase>  
         Buf_Fonc_OB_TPL<TyData,TyBase>::~Buf_Fonc_OB_TPL()
{
       if (mLineInit)
          DELETE_TAB_MATRICE(mLineInit,_d_out,Pt2di(mX0,0),Pt2di(mX1,mY1));
       DELETE_TAB_MATRICE(vbuf,_d_out,Pt2di(mX0,mY0),Pt2di(mX1,mY1));
}


template <class TyData,class TyBase>  
         Buf_Fonc_OB_TPL<TyData,TyBase>::Buf_Fonc_OB_TPL
(
       Flux_Pts_Computed * flx,
       Fonc_Num_Computed * f,
       Box2di              rect,
       Box2di              side,
       GenIm::type_el      type,
       bool                aWithLineInit
)   :
    Buf_Fonc_Op_buf(flx,f,rect,side,type)
{
    mLineInit = aWithLineInit ?  NEW_TAB_MATRICE(_d_out,Pt2di(mX0,0),Pt2di(mX1,mY1),TyData) : 0;

    vbuf = NEW_TAB_MATRICE(_d_out,Pt2di(mX0,mY0),Pt2di(mX1,mY1),TyData);
}

template <class TyData,class TyBase> 
         void Buf_Fonc_OB_TPL<TyData,TyBase>::compute_last_line(INT anY)
{
      // [1] : Rotate data;
      for (INT d=0 ; d<_d_out ; d++)
           rotate_plus_data(vbuf[d],mY0,mY1);

      // 2 : compute
      for (INT x=mX0; x<mX1; x+=_sz_buf)
      {
           INT nb = ElMin(mX1-x,_sz_buf);
           _pack_f->pt0()[0] = x;
           _pack_f->pt0()[1] = anY + mY1-1;
           _pack_f->set_nb(nb);

           Std_Pack_Of_Pts<TyBase> * vals 
                   = SAFE_DYNC
                     (
                         Std_Pack_Of_Pts<TyBase> *,
                         const_cast<Pack_Of_Pts *>(_f->values(_pack_f))
                     );

           for (INT d=0 ; d < _d_out ; d++)
               convert
               (
                   vbuf[d][mY1-1]+x,
                   vals->_pts[d],
                   nb
               );
      }

      if (mLineInit)
      {
           for (INT d=0 ; d<_d_out ; d++)
           {
               rotate_plus_data(mLineInit[d],0,mY1);
               convert
               (
                   mLineInit[d][mY1-1]+mX0,
                   vbuf[d][mY1-1]+mX0,
                   mX1-mX0
               );
           }
      }
}


Buf_Fonc_Op_buf * Buf_Fonc_Op_buf::new_bfob
(
       const Arg_Fonc_Num_Comp & arg,
       Arg_FNOPB                 afn,
       Box2di                    rect,
       bool                      aWlineInit
)
{
     INT sz_buf = arg.flux()->sz_buf();
     Box2di side = afn._side;

     Flux_Pts_Computed * rect_comp 
          = RLE_Flux_Pts_Computed::rect_2d_interface
            (
               rect._p0+side._p0,
               rect._p1+side._p1,
               sz_buf
            );

     Fonc_Num_Computed * fc = afn._f.compute
                              (
                                  Arg_Fonc_Num_Comp(rect_comp)
                              );

     if (afn.type() == GenIm::int4)
        return new  Buf_Fonc_OB_TPL<INT,INT>
                    (rect_comp,fc,rect,side,afn.type(),aWlineInit);

     if (afn.type() == GenIm::real8)
        return new  Buf_Fonc_OB_TPL<REAL,REAL>
                    (rect_comp,fc,rect,side,afn.type(),aWlineInit);


     if (afn.type() == GenIm::u_int1)
        return new Buf_Fonc_OB_TPL<U_INT1,INT>
                   (rect_comp,fc,rect,side,afn.type(),aWlineInit);


     if (afn.type() == GenIm::real4)
        return new Buf_Fonc_OB_TPL<REAL4,REAL>
                   (rect_comp,fc,rect,side,afn.type(),aWlineInit);

     if (afn.type() == GenIm::u_int2)
        return new Buf_Fonc_OB_TPL<U_INT2,INT>
                   (rect_comp,fc,rect,side,afn.type(),aWlineInit);


     if (afn.type() == GenIm::int1)
        return new Buf_Fonc_OB_TPL<INT1,INT>
                   (rect_comp,fc,rect,side,afn.type(),aWlineInit);



    elise_internal_error
    (
        "Unexpected Type in Buf_Fonc_Op_buf::new_bfob",
         __FILE__,__LINE__
    );
    return 0;
}


/*********************************************************************/
/*                                                                   */
/*              Fonc_Num_OP_Buf                                      */
/*                                                                   */
/*********************************************************************/

const Box2di Fonc_Num_OP_Buf::_bBid (Pt2di(0,0),Pt2di(0,0));

Fonc_Num_OP_Buf::Fonc_Num_OP_Buf
(
      const Arg_Fonc_Num_Comp & arg,
      Arg_FNOPB                 ar0,
      Arg_FNOPB                 ar1,
      Arg_FNOPB                 ar2,
      bool                      aLineInit
)    
{
     Box2di              rect (Pt2di(0,0),Pt2di(0,0));


     ASSERT_TJS_USER
     (
           arg.flux()->is_rect_2d(rect),
           "Use of ``buffered rectangular filter'' with a non `` 2D-rectangular'' flux"
     );

     _x0 = rect._p0.x;
     _x1 = rect._p1.x;
     _y0 = rect._p0.y;
     _y1 = rect._p1.y;

     _nbf = 0;


     if (ar0.really_an_arg())
     {
        _nbf = 1;
        _buf_foncs[0] = Buf_Fonc_Op_buf::new_bfob(arg,ar0,rect,aLineInit);

          Box2di  boxb  = kth_box_buf(0);
          Box2di side_0 = ar0.side();
          _x0_buf  = boxb._p0.x;
          _x1_buf  = boxb._p1.x;
          _y0_buf  = boxb._p0.y;
          _y1_buf  = boxb._p1.y;
          _x0_side = side_0._p0.x;
          _x1_side = side_0._p1.x;
          _y0_side = side_0._p0.y;
          _y1_side = side_0._p1.y;

     }

     if (ar1.really_an_arg())
     {
         _nbf = 2;
         _buf_foncs[1] = Buf_Fonc_Op_buf::new_bfob(arg,ar1,rect,aLineInit);
     }

     if (ar2.really_an_arg())
     {
         _nbf = 3;
         _buf_foncs[2] = Buf_Fonc_Op_buf::new_bfob(arg,ar2,rect,aLineInit);
     }

     _first = true;
     _last_y =  rect._p0.y -1;
}

Fonc_Num_OP_Buf::~Fonc_Num_OP_Buf()
{
     for (INT ifonc = 0; ifonc < _nbf ; ifonc++)
         delete _buf_foncs[ifonc];
}

void Fonc_Num_OP_Buf::pre_new_line(bool)
{
}

void Fonc_Num_OP_Buf::post_new_line(bool)
{
}

Box2di Fonc_Num_OP_Buf::kth_box_buf(INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf),
          "Inoherence in Fonc_Num_OP_Buf::kth_box_buf"
     );

     Buf_Fonc_Op_buf *  bf =_buf_foncs[k];
     return Box2di
            (
                 Pt2di(bf->mInitX0,bf->mInitY0),
                 Pt2di(bf->mInitX1,bf->mInitY1)
            );
}

Fonc_Num_Computed * Fonc_Num_OP_Buf::kth_fonc(INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf),
          "Inoherence in Fonc_Num_OP_Buf::kth_fonc"
     );

     return _buf_foncs[k]->_f;
}

INT Fonc_Num_OP_Buf::DimKthFonc(INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf) ,
          "Inoherence in Fonc_Num_OP_Buf::DimKthFonc"
     );
     return (_buf_foncs[k]->_d_out);
}





REAL *** Fonc_Num_OP_Buf::kth_buf(REAL *,INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf) &&( _buf_foncs[k]->type()==GenIm::real8),
          "Inoherence in Fonc_Num_OP_Buf::kth_buf"
     );
     return (REAL ***) (_buf_foncs[k]->buf_values());
}

INT *** Fonc_Num_OP_Buf::kth_buf(INT *,INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf) &&( _buf_foncs[k]->type()==GenIm::int4),
          "Inoherence in Fonc_Num_OP_Buf::kth_buf"
     );
     return (INT ***) (_buf_foncs[k]->buf_values());
}

U_INT1 *** Fonc_Num_OP_Buf::kth_buf(U_INT1 *,INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf) &&( _buf_foncs[k]->type()==GenIm::u_int1),
          "Inoherence in Fonc_Num_OP_Buf::kth_buf"
     );
     return (U_INT1 ***) (_buf_foncs[k]->buf_values());
}


U_INT2 *** Fonc_Num_OP_Buf::kth_buf(U_INT2 *,INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf) &&( _buf_foncs[k]->type()==GenIm::u_int2),
          "Inoherence in Fonc_Num_OP_Buf::kth_buf"
     );
     return (U_INT2 ***) (_buf_foncs[k]->buf_values());
}




REAL4 *** Fonc_Num_OP_Buf::kth_buf(REAL4 *,INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf) &&( _buf_foncs[k]->type()==GenIm::real4),
          "Inoherence in Fonc_Num_OP_Buf::kth_buf"
     );
     return (REAL4 ***) (_buf_foncs[k]->buf_values());
}

INT1 *** Fonc_Num_OP_Buf::kth_buf(INT1 *,INT k)
{
     ASSERT_INTERNAL
     (
          (k>=0) &&(k<_nbf) &&( _buf_foncs[k]->type()==GenIm::int1),
          "Inoherence in Fonc_Num_OP_Buf::kth_buf"
     );
     return (INT1 ***) (_buf_foncs[k]->buf_values());
}



void Fonc_Num_OP_Buf::convert_data(INT Kth,INT * dest,INT nb,INT dim,INT y,INT x0Src)
{
     ASSERT_INTERNAL
     (
          (Kth>=0) &&(Kth<_nbf),
          "Inoherence in Fonc_Num_OP_Buf::kth_buf"
     );
     _buf_foncs[Kth]->convert_data(dest,nb,dim,y,x0Src);
}

void Fonc_Num_OP_Buf::convert_data(INT Kth,REAL * dest,INT nb,INT dim,INT y,INT x0Src)
{
     ASSERT_INTERNAL
     (
          (Kth>=0) &&(Kth<_nbf),
          "Inoherence in Fonc_Num_OP_Buf::kth_buf"
     );
     _buf_foncs[Kth]->convert_data(dest,nb,dim,y,x0Src);
}








void  Fonc_Num_OP_Buf::maj_buf_values(INT y)
{

     _y_cur = y;

     if (_first)
     {
        for (INT  ifonc=0 ; ifonc<_nbf ; ifonc++)
            _buf_foncs[ifonc]->init_comput_line(y);
     }

     if (y != _last_y)
     {
         ELISE_ASSERT(y==(_last_y+1),"Fonc_Num_OP_Buf::maj_buf_values");
         pre_new_line(_first);
         for (INT  ifonc=0 ; ifonc<_nbf ; ifonc++)
             _buf_foncs[ifonc]->compute_last_line(y);
         post_new_line(_first);
     }


     _last_y = y;
     _first = false;
}


/*********************************************************************/
/*                                                                   */
/*              Fonc_Num_OPB_TPL<Type>                               */
/*                                                                   */
/*********************************************************************/


template <class Type> 
          Fonc_Num_OPB_TPL<Type>::Fonc_Num_OPB_TPL
          (
                const Arg_Fonc_Num_Comp & arg,
                INT                     dim_out,
                Arg_FNOPB               ar0,
                Arg_FNOPB               ar1,
                Arg_FNOPB               ar2,
                bool                    aCatFoncInit
          )  :

             Fonc_Num_Comp_TPL<Type>
             (
                 arg,
                 dim_out+(aCatFoncInit?(ar0.DimF()+ar1.DimF()+ar2.DimF()):0),
                 arg.flux(),
                 true
             ),
             Fonc_Num_OP_Buf(arg,ar0,ar1,ar2,aCatFoncInit),
             mCatFoncInit   (aCatFoncInit),
             mDimOutSpec    (dim_out)
{
     _buf_res = NEW_MATRICE
                (
                     Pt2di(_x0,0),
                     Pt2di(_x1,mDimOutSpec),
                     Type
                );

      if (mCatFoncInit)
      {
         for (INT d= mDimOutSpec; d<this->_dim_out ; d++)
         {
             this->_pack_out->_pts[d] = NEW_VECTEUR(0,arg.flux()->sz_buf(),Type);
         }
      }
}


template <class Type>  const  Pack_Of_Pts *
          Fonc_Num_OPB_TPL<Type>::values(const Pack_Of_Pts * gen_pts)
{

     const RLE_Pack_Of_Pts * rle_pack 
           =  SAFE_DYNC(const RLE_Pack_Of_Pts *,gen_pts);

     INT x0 = rle_pack->pt0()[0];
     INT y  = rle_pack->pt0()[1];
     INT nb = rle_pack->nb();

    maj_buf_values(y);

    this->_pack_out->set_nb(nb);
    for (INT d = 0; d < mDimOutSpec ; d++)
    {
        this->_pack_out->_pts[d] = _buf_res[d] + x0;
    }
    if (mCatFoncInit)
    {
       INT dRes = mDimOutSpec;

       for (INT kFonc=0; kFonc<_nbf ; kFonc++)
       {
           INT dimF = DimKthFonc(kFonc);

           for (INT kdf = 0; kdf < dimF ; kdf++)
           {
                convert_data
                (
                     kFonc,
                     this->_pack_out->_pts[dRes++],
                     nb,
                     kdf,
                     0,
                     x0
                );
           }
       }
    }

    return this->_pack_out;
}


template <class Type>   Fonc_Num_OPB_TPL<Type>::~Fonc_Num_OPB_TPL()
{
      if (mCatFoncInit)
      {
         for (INT d= mDimOutSpec; d<this->_dim_out ; d++)
         {
             DELETE_VECTOR(this->_pack_out->_pts[d],0);
         }
      }
      DELETE_MATRICE(_buf_res, Pt2di(_x0,0), Pt2di(_x1,mDimOutSpec));
}


template class Fonc_Num_OPB_TPL<REAL>;
template class Fonc_Num_OPB_TPL<INT>;



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
