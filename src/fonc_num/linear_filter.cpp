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


template <class Type> class Linear_Gen_Comp : public Fonc_Num_Comp_TPL<Type>
{
    public :


      Linear_Gen_Comp
      (
           const Arg_Fonc_Num_Comp &,
           Box2di                  box,
           Flux_Pts_Computed *     flx_int,
           Fonc_Num_Computed       *f
      );


      virtual ~Linear_Gen_Comp()
      {
          delete _flx_int;
          delete _f;
          delete _rle_line;
      }

    private :

      const Pack_Of_Pts * values(const Pack_Of_Pts * pts);

      Fonc_Num_Computed *             _f;
      Flux_Pts_Computed *             _flx_int;
      Box2di                          _box;
      INT                             _last_y;
      const Std_Pack_Of_Pts<Type> *   _last_pack;
      RLE_Pack_Of_Pts *               _rle_line;
};



template <class Type> const Pack_Of_Pts * 
            Linear_Gen_Comp<Type>::values(const Pack_Of_Pts * pts)
{
      const RLE_Pack_Of_Pts * rle_pack = pts->rle_cast();

      INT x0 = rle_pack->pt0()[0];
      INT y0 = rle_pack->pt0()[1];


      if (y0 != _last_y)
      {
          _rle_line->set_pt0(Pt2di(x0,y0));
          _rle_line->set_nb(_box._p1.x-x0);
          const Pack_Of_Pts * p = _f->values(_rle_line);
         _last_pack  = p->std_cast((Type *) 0);
         _last_y = y0;
      }

      INT dx0 = x0 -_box._p0.x;
      this->_pack_out->interv(_last_pack,dx0,dx0+pts->nb());
      return this->_pack_out;
}

template <class Type> Linear_Gen_Comp<Type>::Linear_Gen_Comp
(
    const Arg_Fonc_Num_Comp & arg,
    Box2di                  box,
    Flux_Pts_Computed *     flx_int,
    Fonc_Num_Computed       * f
)   :
    Fonc_Num_Comp_TPL<Type>(arg,f->idim_out(),arg.flux(),true),
    _f           (f),
    _flx_int     (flx_int),
    _box         (box),
    _last_y      (box._p0.y-1),
    _last_pack   (0),
    _rle_line    (RLE_Pack_Of_Pts::new_pck(2,box._p1.x-box._p0.x))
{
}


class Linear_Gen_Not_Comp : public Fonc_Num_Not_Comp
{

     public :


          Linear_Gen_Not_Comp(Fonc_Num f,const char * name)
            : _f (f), _name (name) {}

     private :

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
                if (arg.flux()->is_line_map_rect())
                   return _f.compute(arg);
                
                Box2di b(Pt2di(0,0),Pt2di(0,0));;
                Tjs_El_User.ElAssert
                (
                    arg.flux()->is_rect_2d(b),
                   EEM0 
                      << "incompatible flux for function : \"" 
                      << _name << "\"\n" 
                      << "|    (it  is a \"linear operator\","
                      << "it requires a flux pts of kind\n"
                      << "|   \"rectangle \" or \"line_map_rect\")\n"
                );



                Flux_Pts_Computed * flxi = 
                          RLE_Flux_Pts_Computed::rect_2d_interface
                          (
                              b._p0,b._p1,b._p1.x-b._p0.x
                          );
                Fonc_Num_Computed * fc = _f.compute(Arg_Fonc_Num_Comp(flxi));

                if (fc->integral())
                   return new Linear_Gen_Comp<INT>(arg,b,flxi,fc);
                else
                   return new Linear_Gen_Comp<REAL>(arg,b,flxi,fc);
          }


          virtual bool  integral_fonc (bool iflx) const 
          {return _f.integral_fonc(iflx);}

          virtual INT dimf_out() const {return _f.dimf_out();}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

          Fonc_Num       _f;
          const char *   _name;

          virtual Fonc_Num deriv(INT k) const 
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          

          virtual void  show(ostream & os) const 
          {
                os << "[Linear Filter]";
          }
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for linear Filter");
                return 0;
          }              

};

Fonc_Num r2d_adapt_filtr_lin(Fonc_Num f,const char * name)
{
    return new Linear_Gen_Not_Comp(f,name);
}

/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/
/*****                                                                        ****/
/*****                                                                        ****/
/*****           ASSOCIATIVE                                                  ****/
/*****                                                                        ****/
/*****                                                                        ****/
/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/




   /*************************************************************************/
   /*                                                                       */
   /*         Linear_Integr_Comp                                            */
   /*                                                                       */
   /*************************************************************************/

template <class Type> class Linear_Integr_Comp : public  Fonc_Num_Comp_TPL<Type>
{
      public :
          Linear_Integr_Comp
          (
               const Arg_Fonc_Num_Comp &,
               const OperAssocMixte &   op,
               Fonc_Num_Computed *      f,
               const OperAssocMixte *   op2 = 0
          );
          virtual ~Linear_Integr_Comp() 
          {
               DELETE_TAB (_buf_rev);
               delete _f;
          }

      private :

          const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
          {
               Std_Pack_Of_Pts<Type> * vals =
                   SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(_f->values(pts)));
               INT nb = pts->nb();
               for (int d=0 ; d<this->_dim_out ; d++)
               {
                   _op.integral(this->_pack_out->_pts[d],vals->_pts[d],nb);
                   if (_op2)
                   {
                        convert(_buf_rev,vals->_pts[d],nb);
                        auto_reverse_tab(_buf_rev,nb);
                        _op.integral(_buf_rev,_buf_rev,nb);
                        auto_reverse_tab(_buf_rev,nb);
                        _op2->t0_opeg_t1(this->_pack_out->_pts[d],_buf_rev,nb);
                   }
               }
               this->_pack_out->set_nb(nb);
               return this->_pack_out;
          }

          const OperAssocMixte &  _op;
          const OperAssocMixte *  _op2;
          Fonc_Num_Computed *     _f;
          Type *                  _buf_rev;
};

template <class Type> Linear_Integr_Comp<Type>::Linear_Integr_Comp
(
     const Arg_Fonc_Num_Comp &    arg,
     const OperAssocMixte &       op,
     Fonc_Num_Computed       *    f,
     const OperAssocMixte *       op2
)     :
      Fonc_Num_Comp_TPL<Type>(arg,f->idim_out(),arg.flux()),
      _op  (op),
      _op2 (op2),
      _f   (f),
      _buf_rev    (NEW_TAB(arg.flux()->sz_buf(),Type))
{
}

class Linear_Integr_Not_Comp : public Fonc_Num_Not_Comp
{

     public :

          Linear_Integr_Not_Comp(const OperAssocMixte & op, Fonc_Num f,const OperAssocMixte * op2 = 0): 
              _op (op),   _f (f) , _op2 (op2)
          {}

     private :

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
               Fonc_Num_Computed * fc = _f.compute(arg);
               if (fc->type_out() == Pack_Of_Pts::integer)
                  return  new Linear_Integr_Comp<INT>(arg,_op,fc,_op2);
               else
                  return  new Linear_Integr_Comp<REAL>(arg,_op,fc,_op2);
          }

          const OperAssocMixte & _op;
          Fonc_Num  _f;
          const OperAssocMixte * _op2;

          virtual bool  integral_fonc (bool iflx) const 
          {return _f.integral_fonc(iflx);}

          virtual INT dimf_out() const {return _f.dimf_out();}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

          virtual Fonc_Num deriv(INT k) const 
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          virtual void  show(ostream & os) const 
          {
                os << "[Linear Filter]";
          }
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for linear Filter");
                return 0;
          }              
};


Fonc_Num lin_cumul_ass
          (
              const OperAssocMixte & op,
              Fonc_Num f,
              const OperAssocMixte * op2 
          )
{
      return r2d_adapt_filtr_lin
             (
                  new Linear_Integr_Not_Comp(op,f,op2),
                  op.name()
             );
}

Fonc_Num lin_cumul_sum(Fonc_Num f) 
{
     return lin_cumul_ass(OpSum,f);
}
Fonc_Num lin_cumul_max(Fonc_Num f) 
{
     return lin_cumul_ass(OpMax,f);
}
Fonc_Num lin_cumul_min(Fonc_Num f) 
{
     return lin_cumul_ass(OpMin,f);
}

Fonc_Num linear_integral(Fonc_Num f) 
{
     return lin_cumul_sum(f);
}


Fonc_Num min_linear_cum2sens_max(Fonc_Num f) 
{
     return lin_cumul_ass(OpMax,f,&OpMin);
}

Fonc_Num max_linear_cum2sens_min(Fonc_Num f) 
{
     return lin_cumul_ass(OpMin,f,&OpMax);
}


   /*************************************************************************/
   /*                                                                       */
   /*         Linear_Red_Comp                                               */
   /*                                                                       */
   /*************************************************************************/

template <class Type> class Linear_Red_Comp : public  Fonc_Num_Comp_TPL<Type>
{
      public :


          Linear_Red_Comp
          (
               const Arg_Fonc_Num_Comp &,
               const OperAssocMixte &   op,
               Fonc_Num_Computed *      f,
               INT                      x0,
               INT                      x1
          );

          virtual ~Linear_Red_Comp()
          {
                DELETE_VECTOR(_in,_xM0);
                DELETE_VECTOR(_buf_ar,_x0);
                DELETE_VECTOR(_buf_av,_x0);
                delete _f;
          }

      private :

          const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
          {
               Std_Pack_Of_Pts<Type> * vals =
                   SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(_f->values(pts)));

               INT nb = pts->nb();

               this->_pack_out->set_nb(nb);
               set_cste(_in+_xM0,_neutre,-_xM0);
               set_cste(_in+nb,_neutre,_xM1);

               for (int d=0 ; d<this->_dim_out ; d++)
               {
                   convert(_in,vals->_pts[d],nb);
                   _op.reduce_seg(this->_pack_out->_pts[d],_in,_buf_ar,_buf_av,0,nb,_x0,_x1);
               }

               return this->_pack_out;
          }


          const OperAssocMixte & _op;
          Fonc_Num_Computed * _f;
          INT     _x0;
          INT     _x1;

          INT     _xM0;
          INT     _xM1;

          Type *  _buf_ar;
          Type *  _buf_av;
          Type *  _in;
          Type    _neutre;
};



template <class Type> Linear_Red_Comp<Type>::Linear_Red_Comp
(
     const Arg_Fonc_Num_Comp & arg,
     const OperAssocMixte &   op,
     Fonc_Num_Computed       * f,
     INT                      x0,
     INT                      x1
)     :
      Fonc_Num_Comp_TPL<Type>(arg,f->idim_out(),arg.flux()),
      _op  (op),
      _f   (f),
      _x0  (x0),
      _x1  (x1),
      _xM0 (ElMin(x0,0)),
      _xM1 (ElMax(x1,0))
{
      _op.set_neutre(_neutre);

      _buf_av = NEW_VECTEUR(_x0,_x1+arg.flux()->sz_buf(),Type);
      _buf_ar = NEW_VECTEUR(_x0,_x1+arg.flux()->sz_buf(),Type);
      _in     = NEW_VECTEUR(_xM0,_xM1+arg.flux()->sz_buf(),Type);
}


class Linear_Red_Not_Comp : public Fonc_Num_Not_Comp
{

     public :
          Linear_Red_Not_Comp
          (
                 const OperAssocMixte &,
                 Fonc_Num,
                 INT,
                 INT,
                 const char *
          );

     private :

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {

               Fonc_Num_Computed * fc = _f.compute(arg);
               if (fc->type_out() == Pack_Of_Pts::integer)
                  return new Linear_Red_Comp<INT>(arg,_op,fc,_x0,_x1);
               else
                  return new Linear_Red_Comp<REAL>(arg,_op,fc,_x0,_x1);
          }

          const OperAssocMixte & _op;
          Fonc_Num  _f;
          INT _x0;
          INT _x1;

          virtual bool  integral_fonc (bool iflx) const 
          {return _f.integral_fonc(iflx);}

          virtual INT dimf_out() const {return _f.dimf_out();}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

          virtual Fonc_Num deriv(INT k) const 
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          virtual void  show(ostream & os) const 
          {
                os << "[Linear Filter]";
          }
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for linear Filter");
                return 0;
          }              
           
};

Linear_Red_Not_Comp::Linear_Red_Not_Comp
(
        const OperAssocMixte & op,
        Fonc_Num               f,
        INT                    x0,
        INT                    x1,
        const char *           name
)     :
      _op   (op),
      _f    (f ),
      _x0   (x0),
      _x1   (x1)
{
       Tjs_El_User.ElAssert
       ( 
           (_x0<=_x1),
           EEM0 << "function \"" << name << "\" : " 
                << "limits x0, x1 of, do verify x0 <= x1\n"
                << "|   x0= " << x0 << " , x1 = " << x1
       );

}


Fonc_Num linear_red
         (
              const OperAssocMixte & op,
              Fonc_Num f,
              INT x0,
              INT x1
         )
{
      return r2d_adapt_filtr_lin
             (
                 new Linear_Red_Not_Comp(op,f,x0,x1,op.name()),
                 op.name()
             );
}

Fonc_Num linear_som(Fonc_Num f,INT x0,INT x1)
{
         return linear_red(OpSum,f,x0,x1);
}


Fonc_Num linear_max(Fonc_Num f,INT x0,INT x1)
{
         return linear_red(OpMax,f,x0,x1);
}

Fonc_Num linear_min(Fonc_Num f,INT x0,INT x1)
{
         return linear_red(OpMin,f,x0,x1);
}



/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/
/*****                                                                        ****/
/*****                                                                        ****/
/*****           SHADING                                                      ****/
/*****                                                                        ****/
/*****                                                                        ****/
/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/

/*
   Pour l'apparent "paradoxe" de la prise en compte de average_euclid_line_seed,
   voici un raisonnement qui (provisoirement) m'enleve mes doutes :

      * quand on est (par ex) en vecteur (1,1) on analyse une contraction
         d'un facteur sqrt(2) de la coupe reelle;

      * dans le cas binary shadin ou projection, on a un pente qui vient d'un
        MNT "non contratce", donc on la multiplie par sqrt(2) pour l'exporter
        vers le MNT normal;

      * dans le cas "gray level shading" il s'agit d'un pente calculee sur le
        MNT contracte que l'on souhaite exporter vers le MNT reel; on divise
        donc par sqrt(2);
        
*/

       /***********************************************************************/
       /***********************************************************************/
       /*                                                                     */
       /*               BINARY SHADING                                        */
       /*                                                                     */
       /***********************************************************************/
       /***********************************************************************/


              /*************************************************************************/
              /*                                                                       */
              /*         Linear_Bin_Shading_Comp                                       */
              /*                                                                       */
              /*************************************************************************/
           
class Linear_Bin_Shading_Comp : public  Fonc_Num_Comp_TPL<INT>
{
      
      public :

          Linear_Bin_Shading_Comp
          (
               const Arg_Fonc_Num_Comp  &,
               Fonc_Num_Computed *      f,
               REAL                   steep
          );

          virtual ~Linear_Bin_Shading_Comp() {delete _f;}

          const Pack_Of_Pts * values(const Pack_Of_Pts * pts);

      private :

          Fonc_Num_Computed * _f;
          REAL                _steep;
          bool                _neg_st;
};

Linear_Bin_Shading_Comp::Linear_Bin_Shading_Comp
(
        const Arg_Fonc_Num_Comp  & arg,
        Fonc_Num_Computed *      f,
        REAL                   steep
)     :
      Fonc_Num_Comp_TPL<INT>(arg,f->idim_out(),arg.flux()),
      _f (f),
      _steep                (ElAbs(steep)*arg.flux()->average_dist()),
      _neg_st               (steep < 0)
{
}




const Pack_Of_Pts * Linear_Bin_Shading_Comp::values(const Pack_Of_Pts * pts)
{
      Std_Pack_Of_Pts<REAL> * vals =
                   SAFE_DYNC(Std_Pack_Of_Pts<REAL> *,const_cast<Pack_Of_Pts *>(_f->values(pts)));

      if (_neg_st)
         vals->auto_reverse();

      INT nb = pts->nb();
      this->_pack_out->set_nb(nb);
      INT *  out = this->_pack_out->_pts[0];
      REAL * in = vals->_pts[0];

 
      for (int x0=0 ; x0<nb ;)
      {
          out[x0] = 0;

          INT x1;
          for 
          (
                x1 = x0+1; 
                (x1 <nb) &&(in[x1]-in[x0] < _steep * (x0-x1)) ;
                x1 ++
          )
             out[x1] = 1;

           x0 = x1;
      }

      if (_neg_st)
      {
         vals->auto_reverse();
         this->_pack_out->auto_reverse();
      }

      return this->_pack_out;
}

              /*************************************************************************/
              /*                                                                       */
              /*         Linear_Bin_Shading_Not_Comp                                   */
              /*                                                                       */
              /*************************************************************************/

class Linear_Bin_Shading_Not_Comp : public Fonc_Num_Not_Comp
{

     public :
          Linear_Bin_Shading_Not_Comp
          (
                 Fonc_Num,
                 REAL
          );

     private :

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {

               Fonc_Num_Computed * fc = _f.compute(arg);

               Tjs_El_User.ElAssert
               (
                   fc->idim_out() == 1,
                   EEM0 << "function \"binary_shading\" : "
                        << "dim out of Fonc_Num should be 1\n"
                        << "|     (dim out = "
                        <<  fc->idim_out() << ")"
               );
              
               fc = convert_fonc_num(arg,fc,arg.flux(),Pack_Of_Pts::real);
               return  new Linear_Bin_Shading_Comp(arg,fc,_steep);
          }

          Fonc_Num  _f;
          REAL _steep ;

          virtual bool  integral_fonc (bool) const 
          {return true;}

          INT dimf_out() const {return _f.dimf_out();}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


          virtual Fonc_Num deriv(INT k) const 
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          virtual void  show(ostream & os) const 
          {
                os << "[Linear Filter]";
          }
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for linear Filter");
                return 0;
          }              
};

Linear_Bin_Shading_Not_Comp::Linear_Bin_Shading_Not_Comp(Fonc_Num f,REAL steep) :
           _f (f),
           _steep (steep)
{
}

Fonc_Num binary_shading(Fonc_Num f,REAL steep)
{
      return r2d_adapt_filtr_lin
             (
                   new Linear_Bin_Shading_Not_Comp(f,steep),
                   "binary_shading"
             );
}



       /***********************************************************************/
       /***********************************************************************/
       /*                                                                     */
       /*               GRAY LEVEL SHADING                                    */
       /*                                                                     */
       /***********************************************************************/
       /***********************************************************************/

              /*************************************************************************/
              /*                                                                       */
              /*         Linear_Gray_Level_Shading_Comp                                */
              /*                                                                       */
              /*************************************************************************/
           
class Linear_Gray_Level_Shading_Comp : public  Fonc_Num_Comp_TPL<REAL>
{
      
      public :

          Linear_Gray_Level_Shading_Comp
          (
               const Arg_Fonc_Num_Comp  &,
               Fonc_Num_Computed *      f
          );

          virtual ~Linear_Gray_Level_Shading_Comp()
          {
               DELETE_VECTOR(_in,-1);
               delete _f;
          }


          const Pack_Of_Pts * values(const Pack_Of_Pts * pts);

      private :

          Fonc_Num_Computed * _f;

          // variable +or- global used during computation

          REAL     _corr_dir;
          INT       _nb;
          REAL *   _out;
          REAL *   _in;

          inline REAL  comp_steep(INT x1,INT x2) {return (_in[x1] - _in[x2]) /(x2-x1);}

          INT  compute_rec(INT x0,INT x_vis);
};

Linear_Gray_Level_Shading_Comp::Linear_Gray_Level_Shading_Comp
(
        const Arg_Fonc_Num_Comp  & arg,
        Fonc_Num_Computed *      f
)     :
      Fonc_Num_Comp_TPL<REAL>(arg,f->idim_out(),arg.flux()),
      _f (f),
      _corr_dir              (arg.flux()->average_dist())
{
      _in = NEW_VECTEUR(-1,1+arg.flux()->sz_buf(),REAL);
}

/*
    Assume that the ray tangently illuminating x0, comes from x_0_vis;
*/

INT  Linear_Gray_Level_Shading_Comp::compute_rec(INT x0,INT x_0_vis)
{

     REAL  steep_x0 =  comp_steep(x0,x_0_vis);
     _out[x0]  = steep_x0 ;

     REAL  min_steep =-1e35; // Just to avoid warnings
     for (INT x = x0+1;  ;) 
     {
          REAL  steep_x = comp_steep(x,x0);

          if (steep_x + 1e-7 < steep_x0)
             return x;
          if ((x==x0+1) || (steep_x < min_steep))
          {
             min_steep = steep_x;
             _out[x] = steep_x;
             x++;
          }
          else
              x = compute_rec(x-1,x0);
     }

     elise_fatal_error("incoherence in gray-level shading",__FILE__,__LINE__);
     return _nb;
}


const Pack_Of_Pts * Linear_Gray_Level_Shading_Comp::values(const Pack_Of_Pts * pts)
{

      Std_Pack_Of_Pts<REAL> * vals =
                   SAFE_DYNC(Std_Pack_Of_Pts<REAL> *,const_cast<Pack_Of_Pts *>(_f->values(pts)));

      _nb = pts->nb();
      this->_pack_out->set_nb(_nb);
      if (! _nb)
         return this->_pack_out;

      _out = this->_pack_out->_pts[0];
      _out[0] = 0.0;
      if (_nb == 1)
           return this->_pack_out;

      REAL * in = vals->_pts[0];
      convert(_in,in,_nb);

       REAL vmax = _in[0];
       REAL vmin = _in[0];
       {  // use this stupid {} because fucking visual
	    for (int i =0; i<_nb ; i++)
            {
                vmax = ElMax(vmax,_in[i]);
                vmin = ElMin(vmin,_in[i]);
	    } 
       }

       REAL fact = (vmax == vmin) ? 1.0 : (vmax-vmin);
       for (int i =0; i<_nb ; i++)
            _in[i] = ( _in[i]-vmin) / fact;
  
      _in[-1] = -1e30;
      _in[_nb] = 1e40;


 #if (DEBUG_INTERNAL)
     INT verif_nb = compute_rec(0,-1);
     ASSERT_INTERNAL(verif_nb==_nb,"incoherence in gray level shading");
#else
      compute_rec(0,-1);
#endif

      // ASSERT_INTERNAL(compute_rec(0,-1)==_nb,"incoherence in gray level shading");


      REAL fact_corr = fact / _corr_dir;
      for (int x = 0; x < _nb ; x++)
          if (_out[x] < 0.0)
             _out[x] = 0.0;
          else
             _out[x] *= fact_corr;
      
      return this->_pack_out;
}



              /*************************************************************************/
              /*                                                                       */
              /*         Linear_Gray_Level_Shading_Not_Comp                            */
              /*                                                                       */
              /*************************************************************************/

class Linear_Gray_Level_Shading_Not_Comp : public Fonc_Num_Not_Comp
{

     public :
          Linear_Gray_Level_Shading_Not_Comp ( Fonc_Num);

     private :

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {

               Fonc_Num_Computed * fc = _f.compute(arg);

               Tjs_El_User.ElAssert
               (
                   fc->idim_out() == 1,
                   EEM0 << "function \"gray_level_shading\" : "
                        << "dim out of Fonc_Num should be 1\n"
                        << "|     (dim out = "
                        <<  fc->idim_out() << ")"
               );
            
               fc = convert_fonc_num(arg,fc,arg.flux(),Pack_Of_Pts::real);

               return new Linear_Gray_Level_Shading_Comp(arg,fc);
          }

          Fonc_Num  _f;
          virtual bool  integral_fonc (bool) const 
          {return false;}

          virtual INT dimf_out() const {return _f.dimf_out();}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

          virtual Fonc_Num deriv(INT k) const 
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          virtual void  show(ostream & os) const 
          {
                os << "[Linear Filter]";
          }
          REAL ValFonc(const PtsKD &) const
          {
                ELISE_ASSERT(false,"No ValFonc for linear Filter");
                return 0;
          }              
};


Linear_Gray_Level_Shading_Not_Comp::Linear_Gray_Level_Shading_Not_Comp(Fonc_Num f) :
           _f (f)
{
}

Fonc_Num gray_level_shading(Fonc_Num f)
{
      return r2d_adapt_filtr_lin
             (
                new Linear_Gray_Level_Shading_Not_Comp(f),
                "gray_level_shading"
             );
}


   /*************************************************************************/
   /*                                                                       */
   /*         Linear_Integr_2sens                                           */
   /*                                                                       */
   /*************************************************************************/

#if (0)
template <class Type> class Linear_2sens_Integr_Comp : public  Fonc_Num_Comp_TPL<Type>
{
      public :
          Linear_Integr_Comp
          (
               const Arg_Fonc_Num_Comp &,
               const OperAssocMixte &   op1,
               const OperAssocMixte &   op2,
               Fonc_Num_Computed *      f
          );
          virtual ~Linear_Integr_Comp() {delete _f;}

      private :

          const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
          {
               Std_Pack_Of_Pts<Type> * vals =
                   SAFE_DYNC(Std_Pack_Of_Pts<Type> *,_f->values(pts));
               INT nb = pts->nb();
               for (int d=0 ; d<this->_dim_out ; d++)
               {
                   _op1.integral(this->_pack_out->_pts[d],vals->_pts[d],nb);
                }
               this->_pack_out->set_nb(nb);
               return this->_pack_out;
          }

          const OperAssocMixte & _op1;
          const OperAssocMixte & _op2;
          Type *                 _buf_rev;
          Fonc_Num_Computed *      _f;
};

template <class Type>  Linear_2sens_Integr_Comp<Type>::Linear_2sens_Integr_Comp
(
     const Arg_Fonc_Num_Comp & arg,
     const OperAssocMixte &   op1,
     const OperAssocMixte &   op2,
     Fonc_Num_Computed       * f
)     :
      Fonc_Num_Comp_TPL<Type>(arg,f->idim_out(),arg.flux()),
      _op1        (op1),
      _op2        (op2),
      _buf_rev    (NEW_TAB(arg.flux()->sz_buf(),Type)),
      _f          (f)
{
}

class Linear_Integr_Not_Comp : public Fonc_Num_Not_Comp
{

     public :

          Linear_Integr_Not_Comp(const OperAssocMixte & op, Fonc_Num f): 
              _op (op),   _f (f)
          {}

     private :

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
               Fonc_Num_Computed * fc = _f.compute(arg);
               if (fc->type_out() == Pack_Of_Pts::integer)
                  return  new Linear_Integr_Comp<INT>(arg,_op,fc);
               else
                  return  new Linear_Integr_Comp<REAL>(arg,_op,fc);
          }

          const OperAssocMixte & _op;
          Fonc_Num  _f;

          virtual bool  integral_fonc (bool iflx) const 
          {return _f.integral_fonc(iflx);}

          virtual INT dimf_out() const {return _f.dimf_out();}
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

          virtual Fonc_Num deriv(INT k) const 
          {
                ELISE_ASSERT(false,"No derivation for formal filters");
                return 0;
          }
          virtual void  show(ostream & os) const 
          {
                os << "[Linear Filter]";
          }
};


Fonc_Num lin_cumul_ass
          (
              const OperAssocMixte & op,
              Fonc_Num f
          )
{
      return r2d_adapt_filtr_lin
             (
                  new Linear_Integr_Not_Comp(op,f),
                  op.name()
             );
}

Fonc_Num lin_cumul_sum(Fonc_Num f) 
{
     return lin_cumul_ass(OpSum,f);
}
#endif


Im2D_U_INT1  Shading
             (
                  Pt2di    aSz,
                  Fonc_Num aFMnt,
                  INT      aNbDir, 
                  REAL Anisotropie
             )
{
   Im2D_REAL4 aShade(aSz.x,aSz.y);
   ELISE_COPY(aShade.all_pts(),0,aShade.out());

   REAL SPds = 0;

   for (int i=0; i<aNbDir; i++)
   {
       REAL Teta  = (2*PI*i) / aNbDir ;
       Pt2dr U(cos(Teta),sin(Teta));
       Pt2di aDir = Pt2di( U * (aNbDir * 4.0));
       REAL Pds = (1-Anisotropie) + Anisotropie *ElSquare(1.0 - euclid(U,Pt2dr(0,1))/2);
       Symb_FNum Gr = (1-cos(PI/2-atan(gray_level_shading(aFMnt)))) *255.0;
       SPds  += Pds;
       ELISE_COPY
       (
            line_map_rect(aDir,Pt2di(0,0),aSz),
            aShade.in()+Pds*Gr,
            aShade.out() 
        );
   }

   Im2D_U_INT1 aRes(aSz.x,aSz.y);
   ELISE_COPY
   (
       aRes.all_pts(),
       Max(0,Min(255,aShade.in()/SPds)),
       aRes.out()
   );

   return aRes;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
