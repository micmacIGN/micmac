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


/***************************************************************/
/*                                                             */
/*             Fonc_Num_Comp_TPL                               */
/*                                                             */
/***************************************************************/




template <class Type> Fonc_Num_Comp_TPL<Type>::Fonc_Num_Comp_TPL
                     (  const Arg_Fonc_Num_Comp & arg,
                        INT dim_out,
                        Flux_Pts_Computed * flx,
                        bool sz_buf_0
                     ) :


        Fonc_Num_Computed
        (
             arg,
             dim_out,
             Fonc_Num_Comp_TPL<Type>::class_pack_out
        ),

        _pack_out ( Std_Pack_Of_Pts<Type>::new_pck(dim_out,sz_buf_0 ? 0 : flx->sz_buf()))
        
{
};

template <class Type> Fonc_Num_Comp_TPL<Type>::~Fonc_Num_Comp_TPL()
{
    delete _pack_out;
}

template <> Pack_Of_Pts::type_pack Fonc_Num_Comp_TPL<INT>::class_pack_out = Pack_Of_Pts::integer;
template <> Pack_Of_Pts::type_pack Fonc_Num_Comp_TPL<REAL>::class_pack_out = Pack_Of_Pts::real;

#if ElTemplateInstantiation
	template class Fonc_Num_Comp_TPL<INT>;
	template class Fonc_Num_Comp_TPL<REAL>;
#endif

/***************************************************************/
/*                                                             */
/*             FNC_Computed                                    */
/*                                                             */
/*   (= fonction numeric constant, computed)                   */
/*                                                             */
/***************************************************************/


template <class Type> class FNC_Computed : 
         public Fonc_Num_Comp_TPL<Type>
{
    public :
        virtual const Pack_Of_Pts * values(const Pack_Of_Pts *);
        FNC_Computed(const Arg_Fonc_Num_Comp &,Flux_Pts_Computed *,Type);
        virtual bool  icste(INT *);
        Type _val;
};


template <class Type> FNC_Computed<Type>::FNC_Computed 
        (const Arg_Fonc_Num_Comp & arg,Flux_Pts_Computed * flx,Type val) :

             Fonc_Num_Comp_TPL<Type>(arg,1,flx),
             _val (val)
{


     set_cste
     (
        this->_pack_out->_pts[0],
        val,
         flx->sz_buf()
     );
}



template <class Type>  const Pack_Of_Pts * FNC_Computed<Type>::values
                         (const Pack_Of_Pts * in)
{
   this->_pack_out->set_nb(in->nb());
   return this->_pack_out;
}

template <class Type> bool FNC_Computed<Type>::icste(INT * v)
{
     *v = (INT) _val;
     return true;
}

/***************************************************************/
/*                                                             */
/*             FNC_Not_Comp                                    */
/*                                                             */
/*   (= fonction numeric constant, not computed)               */
/*                                                             */
/***************************************************************/


template <class Type> class   FNC_Not_Comp : public Fonc_Num_Not_Comp
{
     public :
         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);
         FNC_Not_Comp
         (
              Type val,
              bool WithDerForm,
              INT IndDerFormel,
              const std::string & aNVar 
         );

        
        Type         _val;
        bool         mWithDerForm;
        INT          mIndDerFormel;
        std::string  mNVar;
     private :
        static const bool is_integral;

        virtual bool  integral_fonc (bool) const 
        { return is_integral;}

        virtual INT dimf_out() const { return 1;}
 
        void VarDerNN(ElGrowingSetInd &)const {}

        virtual Fonc_Num deriv(INT k) const 
        {
               return mWithDerForm ?  (k==mIndDerFormel) : 0;
        }
        REAL ValFonc(const PtsKD &) const
        {
            return _val;
        }          
        REAL ValDeriv(const PtsKD &,INT k) const
        {
            return 0;
        }          
	 INT DegrePoly() const {return 0;} 


        Fonc_Num::tKindOfExpr  KindOfExpr()
        {
            if (IsVarSpec()) 
               return Fonc_Num::eIsVarSpec;

            return  is_integral ? Fonc_Num::eIsICste : Fonc_Num::eIsRCste;
        }


       INT CmpFormelIfSameKind(Fonc_Num_Not_Comp * aF2)
       {
            if (IsVarSpec())
               return  CmpTertiare((Fonc_Num_Not_Comp *)this,aF2);

            return CmpTertiare(_val,((FNC_Not_Comp<Type>*) aF2)->_val);
       }




	 void compile (cElCompileFN &); 
        virtual void  show(ostream & os) const 
        {
            if (IsVarSpec())
               os << mNVar.c_str() ;
			else if (mWithDerForm)
              os << "[V"  << (-mIndDerFormel) << ":" <<  _val << "]";
            else
              os << _val;
        }
        virtual bool  is0() const
        {
               return( _val == 0) && (!mWithDerForm);
        }
        virtual bool  is1() const
        {
               return (_val == 1) && (!mWithDerForm);
        }

        bool IsCsteRealDim1(REAL & aVal) const
        {
              if (mWithDerForm)
                 return false;

              aVal = _val;
              return true;
        }
       bool IsVarSpec () const {return mWithDerForm && (mNVar != "");}
};

template <> const bool FNC_Not_Comp<INT>::is_integral = true;
template <> const bool FNC_Not_Comp<REAL>::is_integral = false;

template <class Type> FNC_Not_Comp<Type>::FNC_Not_Comp
                (Type val,bool WithDerForm,INT IndDerFormel,const std::string & aNVar) :
      _val           (val),
      mWithDerForm   (WithDerForm),
      mIndDerFormel  (IndDerFormel),
      mNVar          (aNVar)
{
}


template <class Type>  void FNC_Not_Comp<Type>::compile (cElCompileFN & anEnv)
{
   if (IsVarSpec())
   {
      anEnv.PutVarLoc(cVarSpec(this));
      // anEnv << anEnv.PreVarLoc() << mNVar ;
   }
   else
      anEnv << _val;
}

template <class Type>  Fonc_Num_Computed * FNC_Not_Comp<Type>::compute(const Arg_Fonc_Num_Comp & arg)
{
     return new FNC_Computed<Type>(arg,arg.flux(),_val);
}

typedef FNC_Not_Comp<INT>    int_FNC_Not_Comp;
typedef FNC_Not_Comp<REAL>  real_FNC_Not_Comp;

template class  FNC_Not_Comp<INT>;
template class  FNC_Not_Comp<REAL>;

//=============================== cVarSpec =====================

Fonc_Num Fonc_Num::deriv(cVarSpec  aCDer) const
{
    return deriv(aCDer.IndexeDeriv());
}



Fonc_Num::Fonc_Num(cTagCsteDer,REAL aVal, INT anInd,const std::string & aName) :
    PRC0(new real_FNC_Not_Comp(aVal,true,anInd,aName))
{
}

INT cVarSpec::TheCptIndexeNumerotation = -1;

cVarSpec::cVarSpec(REAL aVal,const std::string & aName) :
    Fonc_Num(Fonc_Num::cTagCsteDer(),aVal,TheCptIndexeNumerotation--,aName)
{
}

INT cVarSpec::IndexeDeriv()
{
    return ((FNC_Not_Comp<double> *) _ptr)->mIndDerFormel;
}

void  cVarSpec::Set(REAL aVal)
{
    ((FNC_Not_Comp<double> *) _ptr)->_val = aVal;
}

double *  cVarSpec::AdrVal() const
{
    return & (((FNC_Not_Comp<double> *) _ptr)->_val);
}

cVarSpec::cVarSpec(class Fonc_Num_Not_Comp * aPtr) :
    Fonc_Num (aPtr)
{
}

const std::string & cVarSpec::Name() const
{
    return ((FNC_Not_Comp<double> *) _ptr)->mNVar;
}

/***************************************************************/

/***************************************************************/

/***************************************************************/
/*                                                             */
/*    The two constructor that tranform constants  in fonc_num */
/*                                                             */
/***************************************************************/




Fonc_Num::Fonc_Num(INT v) :
    PRC0(new int_FNC_Not_Comp(v,false,-1,""))
{
}


Fonc_Num::Fonc_Num(INT v1,INT v2) 
{
    * this = Virgule(Fonc_Num(v1),v2);
}

Fonc_Num::Fonc_Num(INT v1,INT v2,INT v3) 
{
    * this = Virgule(Fonc_Num(v1),v2,v3);
}

Fonc_Num::Fonc_Num(INT v1,INT v2,INT v3,INT v4) 
{
    * this =Virgule (Fonc_Num(v1),v2,v3,v4);
}


Fonc_Num::Fonc_Num(Pt2di p)
{
    * this = Fonc_Num(p.x,p.y);
}

Fonc_Num::Fonc_Num(Pt2dr p)
{
    * this = Fonc_Num(p.x,p.y);
}


Fonc_Num::Fonc_Num(REAL v) :
    PRC0(new real_FNC_Not_Comp(v,false,-1,""))
{
}


Fonc_Num::Fonc_Num(REAL v1,REAL v2) 
{
    * this = Virgule(Fonc_Num(v1),v2);
}

Fonc_Num::Fonc_Num(REAL v1,REAL v2,REAL v3) 
{
    * this = Virgule(Fonc_Num(v1),v2,v3);
}

Fonc_Num::Fonc_Num(REAL v1,REAL v2,REAL v3,REAL v4) 
{
    * this = Virgule(Fonc_Num(v1),v2,v3,v4);
}


/***************************************************************/
/*                                                             */
/*             RLE_Fonc_Num_clip_def                           */
/*                                                             */
/***************************************************************/

template  <class Type> class RLE_Fonc_Num_clip_def : public Fonc_Num_Comp_TPL<Type>
{
       public :

          virtual ~RLE_Fonc_Num_clip_def()
          {
              if  (_flx_to_flush)
                  delete _flx_to_flush;

              DELETE_VECTOR(_p0,0);
              DELETE_VECTOR(_p1,0);
              delete _cliped;
              delete _f;
          }

          virtual const Pack_Of_Pts * values(const Pack_Of_Pts * pts);

          RLE_Fonc_Num_clip_def
          (       const Arg_Fonc_Num_Comp & arg,
                  Fonc_Num_Computed * f,
                  Flux_Pts_Computed * flux,
                  const INT * p0,
                  const INT * p1,
                  Type        def_val,
                  INT         rab_p0,
                  INT         rab_p1,
                  bool        flush_flx
          ) ;


            
          Flux_Pts_Computed        * _flx_to_flush;
          Fonc_Num_Computed        * _f;
          INT                       _dim_pts;
          INT *                     _p0;
          INT *                     _p1;
          Type                      _def_val;
          RLE_Pack_Of_Pts *         _cliped;
};

template  <class Type> RLE_Fonc_Num_clip_def<Type>::RLE_Fonc_Num_clip_def
                        (       const Arg_Fonc_Num_Comp & arg,
                                Fonc_Num_Computed * f,
                                Flux_Pts_Computed * flux,
                                const INT * p0,
                                const INT * p1,
                                Type        def_val,
                                INT         rab_p0,
                                INT         rab_p1,
                                bool        flush_flx
                        ) :

              Fonc_Num_Comp_TPL<Type>(arg,f->idim_out(),flux) ,
              _flx_to_flush (flush_flx ? flux : 0),
              _f(f),
              _dim_pts (arg.flux()->dim()),
              _p0 (NEW_VECTEUR(0,_dim_pts,INT)),
              _p1 (NEW_VECTEUR(0,_dim_pts,INT)),
              _def_val(def_val),
              _cliped (RLE_Pack_Of_Pts::new_pck(flux->dim(),RLE_Pack_Of_Pts::_sz_buf_infinite))
{
   for (INT d=0; d <_dim_pts; d++)
   {
       _p0[d] = p0[d] + rab_p0;
       _p1[d] = p1[d] - rab_p1;
   }
}

template  <class Type> const Pack_Of_Pts * RLE_Fonc_Num_clip_def<Type>::values(const Pack_Of_Pts * pts)
{
   INT clip_deb = _cliped->clip(SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(pts)),_p0,_p1);

   INT nb_clip,nb_tot;
   if ( (nb_clip = _cliped->nb()) != (nb_tot = pts->nb()))
   {
       if (nb_clip)
       {
          Std_Pack_Of_Pts<Type> * val_cliped =
                 SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(_f->values(_cliped)));
 
          for (int d = 0; d < this->_dim_out ; d++)
          {
              Type * tab  =  this->_pack_out->_pts[d];
              set_cste(tab,_def_val,clip_deb);
              convert(tab+clip_deb,val_cliped->_pts[d],nb_clip);
              set_cste(tab+nb_clip+clip_deb,_def_val, nb_tot-nb_clip-clip_deb);
          }

       }
       else
       {
          for (int d = 0; d < this->_dim_out ; d++)
              set_cste(this->_pack_out->_pts[d],_def_val,nb_tot);
       }
       this->_pack_out->set_nb(nb_tot);
       return this->_pack_out;
   }
   else
      return _f->values(pts);
}
typedef   RLE_Fonc_Num_clip_def<INT>  RLE_cfn_I;;
typedef   RLE_Fonc_Num_clip_def<REAL> RLE_cfn_R;;

/***************************************************************/
/*                                                             */
/*            Not_Rle_Fonc_Num_clip_def                        */
/*                                                             */
/***************************************************************/

template  <class Ty_Fonc,class Ty_Pts> 
          class Not_Rle_Fonc_Num_clip_def : public Fonc_Num_Comp_TPL<Ty_Fonc>
{
       public :

          virtual ~Not_Rle_Fonc_Num_clip_def()
          {
              if  (_flx_to_flush)
                  delete _flx_to_flush;
              delete _pack_filtr;
              delete _pack_pts_filtred;
              DELETE_VECTOR(_p1,0);
              DELETE_VECTOR(_p0,0);
              delete _f;
          }

          virtual const Pack_Of_Pts * values(const Pack_Of_Pts * pts);

          Not_Rle_Fonc_Num_clip_def
          (       const Arg_Fonc_Num_Comp & arg,
                  Fonc_Num_Computed * f,
                  Flux_Pts_Computed * flux,
                  bool                flush_flx,
                  const INT * p0,
                  const INT * p1,
                  Ty_Fonc    def_val,
                  Ty_Pts     rab_p0,
                  Ty_Pts     rab_p1
          ) ;


            
          Flux_Pts_Computed        * _flx_to_flush;
          Fonc_Num_Computed        * _f;
          INT                        _dim_fonc;
          INT                        _dim_pts;
          INT                        _sz_buf;
          Ty_Pts *             _p0;
          Ty_Pts *             _p1;
          Ty_Fonc                   _def_val;

          Std_Pack_Of_Pts<Ty_Pts>   *   _pack_pts_filtred;
          Std_Pack_Of_Pts<INT>      *   _pack_filtr;

};

template  <class Ty_Fonc,class Ty_Pts> 
          Not_Rle_Fonc_Num_clip_def<Ty_Fonc,Ty_Pts>::Not_Rle_Fonc_Num_clip_def
          (     
                  const Arg_Fonc_Num_Comp & arg,
                  Fonc_Num_Computed * f,
                  Flux_Pts_Computed * flux,
                  bool                flush_flx,
                  const INT * p0,
                  const INT * p1,
                  Ty_Fonc    def_val,
                  Ty_Pts     rab_p0,
                  Ty_Pts     rab_p1
          ) :
               Fonc_Num_Comp_TPL<Ty_Fonc>(arg,f->idim_out(),flux) ,
                _flx_to_flush (flush_flx ?flux:0),
               _f         (f),
               _dim_fonc  (f->idim_out()),
               _dim_pts   (arg.flux()->dim()),
               _sz_buf    (arg.flux()->sz_buf()),
               _p0        (NEW_VECTEUR(0,_dim_pts,Ty_Pts)),
               _p1        (NEW_VECTEUR(0,_dim_pts,Ty_Pts)),
               _def_val   (def_val),
               _pack_pts_filtred  (Std_Pack_Of_Pts<Ty_Pts>::new_pck(_dim_pts,_sz_buf)),
               _pack_filtr        (Std_Pack_Of_Pts<INT>::new_pck(1,_sz_buf))
{
   for (INT d=0; d <_dim_pts; d++)
   {
       _p0[d] = p0[d] + rab_p0;
       _p1[d] = p1[d] - rab_p1;
   }
}


template  <class Ty_Fonc,class Ty_Pts> 
          const Pack_Of_Pts *  Not_Rle_Fonc_Num_clip_def<Ty_Fonc,Ty_Pts>::values
                                                       (const Pack_Of_Pts * gen_pts)
{
    const Std_Pack_Of_Pts<Ty_Pts> * pts =
           SAFE_DYNC(const Std_Pack_Of_Pts<Ty_Pts> *,gen_pts);

    INT nb_tot = pts->nb();
    Ty_Pts ** coord = pts->_pts;


     INT * filtr = _pack_filtr->_pts[0];
     compute_inside(filtr,coord,nb_tot,_dim_pts,_p0,_p1);

     _pack_pts_filtred->set_nb(0);
     pts->select_tab(_pack_pts_filtred,filtr);

     const Std_Pack_Of_Pts<Ty_Fonc> *  pack_val_filtred =
               SAFE_DYNC(const Std_Pack_Of_Pts<Ty_Fonc> *,_f->values(_pack_pts_filtred));

     for (int d=0 ; d<_dim_fonc; d++)
     {
          Ty_Fonc * v_out = this->_pack_out->_pts[d];
          Ty_Fonc * v_filtr = pack_val_filtred->_pts[d];
          for(int i = 0, ifiltr =0; i<nb_tot; i++)
              v_out[i] = filtr[i] ? v_filtr[ifiltr++] : _def_val;
     }

    this->_pack_out->set_nb(nb_tot);
    return this->_pack_out;

}



typedef   Not_Rle_Fonc_Num_clip_def<INT ,INT > NotRLE_cfn_II;
typedef   Not_Rle_Fonc_Num_clip_def<REAL,INT > NotRLE_cfn_RI;
typedef   Not_Rle_Fonc_Num_clip_def<INT ,REAL> NotRLE_cfn_IR;
typedef   Not_Rle_Fonc_Num_clip_def<REAL,REAL> NotRLE_cfn_RR;

/***************************************************************/
/*                                                             */
/*            Interface                                        */
/*                                                             */
/***************************************************************/


Fonc_Num_Computed * clip_fonc_num_def_val
                    (       const Arg_Fonc_Num_Comp & arg,
                            Fonc_Num_Computed * f,
                            Flux_Pts_Computed * flux,
                            const INT * _p0,
                            const INT * _p1,
                            REAL        def_val,
                            REAL        rab_p0 ,
                            REAL        rab_p1 ,
                            bool        flush_flx
                    )
{


    switch (flux->type())
    {

      case Pack_Of_Pts::rle : 
        if (f->integral() )
            return new RLE_cfn_I
                   (arg,f,flux,_p0,_p1,(INT)def_val,(INT)rab_p0,(INT)rab_p1,flush_flx);
        else  
             return new RLE_cfn_R
                    (arg,f,flux,_p0,_p1,     def_val,(INT)rab_p0,(INT)rab_p1,flush_flx);


      case Pack_Of_Pts::integer :
        if (f->integral())                                                      
            return new NotRLE_cfn_II
                    (arg,f,flux,flush_flx,_p0,_p1,(INT)def_val,(INT)rab_p0,(INT)rab_p1);
        else   
	   return new NotRLE_cfn_RI
                  (arg,f,flux,flush_flx,_p0,_p1,def_val,(INT)rab_p0,(INT)rab_p1);

      case Pack_Of_Pts::real :
        if ( f->integral())                                                      
           return  new NotRLE_cfn_IR
                   (arg,f,flux,flush_flx,_p0,_p1,(INT)def_val,rab_p0,rab_p1);
        else
           return new NotRLE_cfn_RR
                  (arg,f,flux,flush_flx,_p0,_p1,def_val,rab_p0,rab_p1);


      default :
         elise_internal_error("clip_fonc_num_def_val",__FILE__,__LINE__);
         return 0   ;
    }
}


/**************************************************************/
/*                                                            */
/*     User's Interface  to clip-fonc                         */
/*                                                            */
/**************************************************************/


class  Fonc_Num_Clip_Not_Comp : public Fonc_Num_Not_Comp
{
       public  :

        Fonc_Num_Clip_Not_Comp
        (
            Fonc_Num f,
            REAL     v,
            INT      * p0,
            INT      * p1,
            INT      dim
        )   :
            _f     (f),
            _v     (v),
            _dim   (dim)
        {
             convert (_p0,p0,_dim);
             convert (_p1,p1,_dim);
        }



       private :
           bool integral_fonc(bool int_flx) const
           {
                return _f.integral_fonc(int_flx);
           }
           int  dimf_out() const
           {
                return _f.dimf_out();
           }


           Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
           {
              Flux_Pts_Computed * flxc = 0;
              Box2di bx(Pt2di(0,0),Pt2di(0,0));

               Tjs_El_User.ElAssert
               (  arg.flux()->dim()==_dim, // ai remplace >= par ==, why >=  ?
                   EEM0 << "function  \"clip_dep\" : " 
                        << "unexpected dim of flux  \n"
                        << "|    expected "  << _dim
                        << " , got  : "   << arg.flux()->dim()
               );


              if (arg.flux()->is_rect_2d(bx))
              {
                 flxc = RLE_Flux_Pts_Computed::rect_2d_interface
                        (
                           Sup(bx._p0,Pt2di(_p0[0],_p0[1])),
                           Inf(bx._p1,Pt2di(_p1[0],_p1[1])),
                           arg.flux()->sz_buf()
                        );
              }
              else
              {
                 flxc = arg.flux();
                 flxc = flx_interface(flxc->dim(),flxc->type(),flxc->sz_buf());
              }
              Arg_Fonc_Num_Comp fnarg = Arg_Fonc_Num_Comp(flxc);

              Fonc_Num_Computed * fc = _f.compute(fnarg);
              return clip_fonc_num_def_val
                     (fnarg,fc,fnarg.flux(), 
                      _p0,_p1,_v,0.0,0.0,flxc != arg.flux()
                     );
           }

           
           Fonc_Num _f;
           REAL     _v;
           INT      _dim;
           INT _p0[Elise_Std_Max_Dim];
           INT _p1[Elise_Std_Max_Dim];
           virtual Fonc_Num deriv(INT) const
           {
               ELISE_ASSERT(false,"No Deriv for clip");
               return 0;
           }
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}
           REAL ValFonc(const PtsKD &) const
           {
               ELISE_ASSERT(false,"No ValFonc for clip");
               return 0;
           }          


           virtual void show(ostream & os) const
           {
               os << "clip()";
           }                
};

Fonc_Num  clip_def
          ( 
              Fonc_Num f,
              REAL     v,
              INT  *   p0,
              INT  *   p1,
              INT      dim
          )
{
    return new Fonc_Num_Clip_Not_Comp(f,v,p0,p1,dim);
}

Fonc_Num  clip_def
          ( 
              Fonc_Num f,
              REAL     v,
              Pt2di    pt0,
              Pt2di    pt1
          )
{
     INT t0[2],t1[2];

     pt0.to_tab(t0);
     pt1.to_tab(t1);
     return clip_def(f,v,t0,t1,2);
}


/***************************************************************/
/*                                                             */
/*             FNC_Computed                                    */
/*                                                             */
/*   (= fonction numeric constant, computed)                   */
/*                                                             */
/***************************************************************/


template <class Type> class FN_KTH_proj : 
         public Fonc_Num_Comp_TPL<Type>
{
    public :

        FN_KTH_proj
        (
             const Arg_Fonc_Num_Comp &,
             Flux_Pts_Computed *,
             Fonc_Num_Computed *,
             const std::vector<INT>  & aVKth
        );
        virtual ~FN_KTH_proj(){delete _f;}

   private :

        virtual const Pack_Of_Pts * values(const Pack_Of_Pts *);
        virtual bool  icste(INT *);

        std::vector<INT>        mVKth;
        Fonc_Num_Computed *     _f;
};


template <class Type> FN_KTH_proj<Type>::FN_KTH_proj 
(    
            const Arg_Fonc_Num_Comp & arg,
            Flux_Pts_Computed * flx,
            Fonc_Num_Computed *  f,
            const std::vector<INT>  & aVKth
) :
             Fonc_Num_Comp_TPL<Type>(arg,(int) aVKth.size(),flx,true),
             mVKth  (aVKth),
             _f (f)
{
}



template <class Type>  const Pack_Of_Pts * FN_KTH_proj<Type>::values
                         (const Pack_Of_Pts * in)
{
   this->_pack_out->set_nb(in->nb());
   Std_Pack_Of_Pts<Type> * aPckIn =  
           SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(_f->values(in)));

    for (INT d=0 ; d<(INT)mVKth.size() ; d++)
    {
         this->_pack_out->_pts[d] = aPckIn->_pts[mVKth[d]];
    }
/*
   Std_Pack_Of_Pts<Type> * aPckIn =  
           SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(_f->values(in)))->_pts[_kth];
   _pack_out->_pts[0] = 
        SAFE_DYNC(Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(_f->values(in)))->_pts[_kth];
*/
   return this->_pack_out;
}

template <class Type> bool FN_KTH_proj<Type>::icste(INT * v)
{
    return _f->icste(v);
}
/*
*/

/***************************************************************/
/*                                                             */
/*             FN_Permut_NComp                                 */
/*                                                             */
/*   (= fonction numeric constant, not computed)               */
/*                                                             */
/***************************************************************/


class FN_Permut_NComp : public Fonc_Num_Not_Comp
{
     public :
         FN_Permut_NComp(Fonc_Num,const std::vector<INT> &);

     private :
         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);


        std::vector<INT>  mKth;
        Fonc_Num _f;

        virtual bool  integral_fonc (bool iflx) const 
        { return _f.integral_fonc(iflx);}

        virtual INT dimf_out() const { return (INT) mKth.size();}

        void VarDerNN(ElGrowingSetInd & aSet)const {_f.VarDerNN(aSet);}

        virtual Fonc_Num deriv(INT kd) const
        {
             return _f.permut(mKth).deriv(kd);
        }
        virtual void show(ostream & os) const
        {
               _f.show(os);
               os << "kth_proj(" << mKth << ")";
        }                
        REAL ValFonc(const PtsKD & pts) const
        {
             ELISE_ASSERT
             (
                 _f.dimf_out()==1 ,
                 "dimOut !=1; in KthProj, ValFonc"
             );
             return _f.ValFonc(pts);
        }          
};

FN_Permut_NComp::FN_Permut_NComp(Fonc_Num f,const std::vector<INT> & aVKth) :
       mKth (aVKth),
       _f   (f)
{
    ELISE_ASSERT((INT)aVKth.size() >= 1,"Bad Size in Permut Coord");

    INT aDout = f.dimf_out();
    for (INT d=0 ; d<(INT)mKth.size() ; d++)
    {
        if (! ((mKth[d]>=0) && (mKth[d]<aDout)))
	{
           std::cout << "INTERV = [0 " << aDout 
	             << "[ ; Val [" << d << "]=" << mKth[d] << "\n";
           ELISE_ASSERT
           (
                  false,
                 "Bad Size in Permut Coord"
           );
	}
    }
}



Fonc_Num_Computed * FN_Permut_NComp::compute(const Arg_Fonc_Num_Comp & arg)
{
     Fonc_Num_Computed * fc = _f.compute(arg);

     if (fc->integral()) 
         return new FN_KTH_proj<INT>(arg,arg.flux(),fc,mKth);
     else
	 return new FN_KTH_proj<REAL>(arg,arg.flux(),fc,mKth);
}

Fonc_Num Fonc_Num::permut(const std::vector<INT> & aVKth) const
{
   return new FN_Permut_NComp(*this,aVKth);
}

Fonc_Num Fonc_Num::kth_proj(INT kth) const
{
   return permut(std::vector<INT>(1,kth));
}


Fonc_Num Fonc_Num::v0() { return kth_proj(0);}
Fonc_Num Fonc_Num::v1() { return kth_proj(1);}
Fonc_Num Fonc_Num::v2() { return kth_proj(2);}


Fonc_Num Fonc_Num::shift_coord(int shift) const
{
    std::vector<INT> mVKth;
    INT dim = dimf_out();

    for (INT d=0 ; d<dim; d++)
       mVKth.push_back(mod(d-shift,dim));

    return permut(mVKth);
}




/***************************************************************/
/*                                                             */
/*             RANDOM                                          */
/*                                                             */
/***************************************************************/


class RAN_Computed : 
         public Fonc_Num_Comp_TPL<REAL>
{
    public :
        virtual const Pack_Of_Pts * values(const Pack_Of_Pts *);
        RAN_Computed(const Arg_Fonc_Num_Comp &,Flux_Pts_Computed *);
};


RAN_Computed::RAN_Computed 
        (const Arg_Fonc_Num_Comp & arg,Flux_Pts_Computed * flx) :

             Fonc_Num_Comp_TPL<REAL>(arg,1,flx)
{
}



const Pack_Of_Pts * RAN_Computed::values
                         (const Pack_Of_Pts * in)
{
   INT nb = in->nb();

   _pack_out->set_nb(nb);
   REAL * out = SAFE_DYNC(Std_Pack_Of_Pts<REAL> *,_pack_out)->_pts[0];

   for (INT i=0 ; i<nb ; i++)
       out[i] = NRrandom3();
   return _pack_out;
}


/***************************************************************/
/*                                                             */
/*             RAN_Not_Computed                                */
/*                                                             */
/***************************************************************/


class   RAN_Not_Computed : public Fonc_Num_Not_Comp
{
     public :
         RAN_Not_Computed() {};

     private :
         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
         {
            return new RAN_Computed(arg,arg.flux());
         }


        virtual bool  integral_fonc (bool) const { return false;}

        virtual INT dimf_out() const { return 1;}

        virtual Fonc_Num deriv(INT kd) const
        {
               ELISE_ASSERT(false,"No Deriv for random");
               return 0;
        }
        void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}

        virtual void show(ostream & os) const
        {
               os << "frandr()";
        }                
        REAL ValFonc(const PtsKD &) const
        {
             ELISE_ASSERT(false,"No ValFonc for random");
             return 0;
        }          
};

Fonc_Num frandr()
{
     return new RAN_Not_Computed();
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
