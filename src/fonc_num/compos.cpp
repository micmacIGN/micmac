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


/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/
/*****                                                                        ****/
/*****                                                                        ****/
/*****           COMPOSITION                                                  ****/
/*****                                                                        ****/
/*****                                                                        ****/
/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/

   /*************************************************************************/
   /*                                                                       */
   /*         Compos_Func_Comp                                              */
   /*                                                                       */
   /*************************************************************************/

class Compos_Func_Comp : public  Fonc_Num_Computed
{
      
      public :

          Compos_Func_Comp
          (
               const Arg_Fonc_Num_Comp &,
               Fonc_Num_Computed *f,
               Fonc_Num_Computed *c,
               Flux_Pts_Computed *flx_interf
          );

          virtual ~Compos_Func_Comp()
          {
             delete _f;
             delete _flx_interf;
             delete _c;
          };

          const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
          {
               return _f->values(_c->values(pts));
          }

      private :

          Fonc_Num_Computed * _f;
          Fonc_Num_Computed * _c;
          Flux_Pts_Computed *_flx_interf;
};
Compos_Func_Comp::Compos_Func_Comp
(
     const Arg_Fonc_Num_Comp & arg,
      Fonc_Num_Computed *f,
      Fonc_Num_Computed *c,
      Flux_Pts_Computed *flx_interf
) :
     Fonc_Num_Computed(arg,f->idim_out(),f->type_out()),
    _f (f),
    _c (c),
    _flx_interf (flx_interf)
{
}

   /*************************************************************************/
   /*                                                                       */
   /*         Compos_Func_Not_Comp                                          */
   /*                                                                       */
   /*************************************************************************/

class Compos_Func_Not_Comp : public  Fonc_Num_Not_Comp
{
      public :

          Compos_Func_Not_Comp(Fonc_Num f,Fonc_Num c);

          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
          {
              Fonc_Num_Computed *f,*c;
              Flux_Pts_Computed *flx_interface;

              c = _c.compute(arg);
              flx_interface = interface_flx_chc(arg.flux(),c);
              f =_f.compute(Arg_Fonc_Num_Comp(flx_interface));
              return new Compos_Func_Comp(arg,f,c,flx_interface);
          }

      private :

           Fonc_Num _f;
           Fonc_Num _c;

          virtual bool  integral_fonc (bool iflx) const 
          {
               return _f.integral_fonc( _c.integral_fonc(iflx));
          }

         virtual INT dimf_out() const
         {
                return _f.dimf_out();
         }
         void VarDerNN(ElGrowingSetInd & aSet)const
         {
              _c.VarDerNN(aSet);
              _f.VarDerNN(aSet);
         }

         virtual Fonc_Num deriv(INT k) const
         {
              return _c.deriv(k) * _f.deriv(k)[_c];
         }

         virtual REAL ValDeriv(const  PtsKD &  pts,INT k) const 
         {
               ELISE_ASSERT
               (
                    _c.dimf_out() ==1,
                    "Use compos, with val and dim!=1"
               );
               REAL c = _c.ValFonc(pts);
               PtsKD vals(&c,1);
               return _c.ValDeriv(pts,k)*_f.ValDeriv(vals,k);
         }


         virtual void  show(ostream & os) const   
         {
              _f.show(os); 
              os << "[";
              _c.show(os); 
              os << "]";
         }
         virtual REAL  ValFonc(const PtsKD & pts) const
         {
               ELISE_ASSERT
               (
                    _c.dimf_out() ==1,
                    "Use compos, with val and dim!=1"
               );
               REAL c = _c.ValFonc(pts);
               PtsKD vals(&c,1);
               return _f.ValFonc(vals);
         }
};

Compos_Func_Not_Comp::Compos_Func_Not_Comp(Fonc_Num f,Fonc_Num c) :
    _f (f) ,
    _c (c)
{
} 

     // interface function 

Fonc_Num Fonc_Num::operator[](Fonc_Num fcoord)
{
      return new Compos_Func_Not_Comp(*this,fcoord);
}


/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/
/*****                                                                        ****/
/*****                                                                        ****/
/*****           TRANSLATION                                                  ****/
/*****                                                                        ****/
/*****                                                                        ****/
/*********************************************************************************/
/*********************************************************************************/
/*********************************************************************************/

   /*************************************************************************/
   /*                                                                       */
   /*         Compos_Func_Comp                                              */
   /*                                                                       */
   /*************************************************************************/

class Trans_Func_Comp : public  Fonc_Num_Computed
{
      
      public :

          Trans_Func_Comp
          (
               const Arg_Fonc_Num_Comp &,
               Fonc_Num_Computed *f,
               const INT * tr,
               Flux_Pts_Computed *flx_interf
          );

         virtual ~Trans_Func_Comp()
         {
              delete _flx_interf;
              delete _pack_tr;
              delete _f;
         }


      private :
        
         const Pack_Of_Pts * values(const Pack_Of_Pts * pts)
         {
               _pack_tr->trans(pts,_tr);
               return _f->values(_pack_tr);
         }


         Fonc_Num_Computed *  _f;
         Pack_Of_Pts       *  _pack_tr;
         INT                  _tr[Elise_Std_Max_Dim];
         Flux_Pts_Computed * _flx_interf;
};


Trans_Func_Comp::Trans_Func_Comp
(
               const Arg_Fonc_Num_Comp & arg,
               Fonc_Num_Computed *        f,
               const INT *                tr,
               Flux_Pts_Computed *flx_interf
)    :
     Fonc_Num_Computed  (arg,f->idim_out(),f->type_out()),
     _f                 (f),
     _pack_tr           (   Pack_Of_Pts::new_pck
                            (    arg.flux()->dim(),
                                 arg.flux()->sz_buf(),
                                 arg.flux()->type()
                            )
                        ),
     _flx_interf        (flx_interf)
{
     convert(_tr,tr,arg.flux()->dim());
}

   /*************************************************************************/
   /*                                                                       */
   /*         Trans_Func_Not_Comp                                           */
   /*                                                                       */
   /*************************************************************************/


class Trans_Func_Not_Comp  : public  Fonc_Num_Not_Comp
{
      public :
         Trans_Func_Not_Comp(Fonc_Num f,const INT * _tr,INT dim);
 
      private :

         Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg)
         {
             Tjs_El_User.ElAssert
             (
                 _dim == (arg.flux()->dim()),
                 EEM0 << "unexpected dimension in Fonc translation\n"
                      << "|    (expected " << _dim
                      << " , got " << arg.flux()->dim() << ")"
             );
             Fonc_Num_Computed * fc;

             Flux_Pts_Computed * flx_int = tr_flx_interface(arg.flux(),_tr);
             fc = _f.compute(Arg_Fonc_Num_Comp(flx_int));
             return new Trans_Func_Comp(arg,fc,_tr,flx_int);
         }

         virtual bool  integral_fonc (bool iflux) const 
         {return  _f.integral_fonc(iflux);}

         virtual INT dimf_out() const
         {
                return _f.dimf_out();
         }
         void VarDerNN(ElGrowingSetInd & aSet) const{_f.VarDerNN(aSet);}

         Fonc_Num   _f;
         INT        _dim;
         INT        _tr[Elise_Std_Max_Dim];


         virtual Fonc_Num deriv(INT k) const
         {
              return trans(_f.deriv(k),_tr,_dim);
         }
         virtual void  show(ostream & os) const   
         {
               os << "trans(";
               _f.show(os);
               os << ",...)";
         }

         virtual REAL  ValFonc(const PtsKD &) const
         {
               ELISE_ASSERT(false,"No val for trans");
               return 0;
         }
};


Trans_Func_Not_Comp::Trans_Func_Not_Comp
(
     Fonc_Num f,
     const INT * tr,
     INT dim
)    :
     _f   (f),
     _dim (dim)
{
     convert(_tr,tr,dim);
}



   /*************************************************************************/
   /*                                                                       */
   /*         trans                                                         */
   /*                                                                       */
   /*************************************************************************/


Fonc_Num trans(Fonc_Num f,const INT* tr,INT dim)
{
    return new Trans_Func_Not_Comp(f,tr,dim);
}


Fonc_Num trans(Fonc_Num f,Pt2di p)
{
    INT tr[2];
    p.to_tab(tr);

    return trans(f,tr,2);
}


Fonc_Num trans(Fonc_Num f,INT x)
{
    return trans(f,&x,1);
}



Fonc_Num trans(Fonc_Num f,Pt3di p)
{
    INT tr[3];
    tr[0] = p.x;
    tr[1] = p.y;
    tr[2] = p.z;

    return trans(f,tr,3);
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
