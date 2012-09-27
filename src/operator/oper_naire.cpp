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




typedef  void (* FONC_NAIRE_RR)(REAL** ,REAL** , INT);
typedef  void (* FONC_NAIRE_II )(INT ** ,INT ** , INT);
typedef  void (* FONC_NAIRE_IR )(INT ** ,REAL**, INT);
typedef  void (* FONC_NAIRE_RI )(REAL** ,INT **, INT);


/***************************************************************/
/*                                                             */
/*             OP_Un_TPL                                       */
/*                                                             */
/***************************************************************/

template <class TOut,class TIn>
          class OP_un_Ndim_Comp : public  Fonc_Num_Comp_TPL<TOut>
{
     public :
         OP_un_Ndim_Comp
         (       const Arg_Fonc_Num_Comp &,
                 Fonc_Num_Computed * f,
                 Flux_Pts_Computed * flx,
                 INT                 dim_out,
                 void  (* fonc) (TOut **,TIn **,INT)
         );

         virtual ~OP_un_Ndim_Comp(void)
         {
              delete _f;
         }

      private :
         virtual const Pack_Of_Pts * values(const Pack_Of_Pts *);

         Fonc_Num_Computed * _f;
         void  (* _fonc) (TOut **,TIn **,INT);
};

typedef OP_un_Ndim_Comp<REAL,REAL> OP_un_Ndim_Comp_RR;
typedef OP_un_Ndim_Comp<REAL,INT > OP_un_Ndim_Comp_RI;
typedef OP_un_Ndim_Comp<INT ,REAL> OP_un_Ndim_Comp_IR;
typedef OP_un_Ndim_Comp<INT ,INT > OP_un_Ndim_Comp_II;

template <class TOut,class TIn> const Pack_Of_Pts *
     OP_un_Ndim_Comp<TOut,TIn>::values(const Pack_Of_Pts * pts)
{

     const Pack_Of_Pts * v = _f->values(pts);
     _fonc
     (
         this->_pack_out->_pts,
         v->std_cast((TIn *)0)->_pts,
         pts->nb()
     );
     this->_pack_out->set_nb(pts->nb());

     return this->_pack_out;
}

template <class TOut,class TIn> OP_un_Ndim_Comp<TOut,TIn>::OP_un_Ndim_Comp
(
   const Arg_Fonc_Num_Comp & arg,
   Fonc_Num_Computed * f,
   Flux_Pts_Computed * flx,
   INT   dim_out,
   void  (* fonc) (TOut **,TIn **,INT)
)  :
     Fonc_Num_Comp_TPL<TOut>(arg,dim_out,flx),
     _f (f),
     _fonc (fonc)
{
     El_Internal.ElAssert
     (
         f != 0,
         EEM0 << "convertion error in OP_un_Ndim_Comp"
     );
}






class OP_Naire_Not_Comp :  public  Fonc_Num_Not_Comp 
{
    public :
         OP_Naire_Not_Comp
         (
             Fonc_Num f,
             bool     real,
             INT      (*  dim_out)(INT),
             bool     (*  accept_dim_in)(INT),
             FONC_NAIRE_RR    frr,
             FONC_NAIRE_II    fii,
             FONC_NAIRE_RI    fri,
             FONC_NAIRE_IR    fir
         )   :
             _f (f), _real(real),_dim_out(dim_out),
             _accept_dim_in(accept_dim_in),
              _frr (frr), _fii (fii), _fri (fri), _fir (fir)
         {
         }
         static INT d3(INT) {return 3;}
         static bool ad3(INT d) {return  d==3;}

    private :
         Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &) ;
         bool integral_fonc(bool ) const 
         {
             return ! _real;
         }

         INT dimf_out() const 
         {
             return _dim_out(_f.dimf_out());
         }
         void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


         Fonc_Num _f;
         bool _real;
         const char * _name;
         INT  (* _dim_out)(INT) ;
         bool (* _accept_dim_in)(INT) ;

         FONC_NAIRE_RR   _frr;
         FONC_NAIRE_II   _fii;
         FONC_NAIRE_RI   _fri;
         FONC_NAIRE_IR   _fir;

};

Fonc_Num_Computed * OP_Naire_Not_Comp::compute
                    (const Arg_Fonc_Num_Comp & arg)
{
      Fonc_Num_Computed * fc = _f.compute(arg);

       Tjs_El_User.ElAssert
       (
             _accept_dim_in(fc->idim_out()),
             EEM0 
               << "   Unexpected dimension for operator " << _name << "\n"
               << "|    dim = " << fc->idim_out()
       );

      INT dout = _dim_out(fc->idim_out());

      if (fc->integral())
      {
         if( _real)
            return new OP_un_Ndim_Comp_RI(arg,fc,arg.flux(),dout,_fri);
         else
            return new OP_un_Ndim_Comp_II(arg,fc,arg.flux(),dout,_fii);
      }
      else
      {
         if ( _real)
           return new OP_un_Ndim_Comp_RR(arg,fc,arg.flux(),dout,_frr);
         else
           return new OP_un_Ndim_Comp_IR(arg,fc,arg.flux(),dout,_fir);
      }

}


Fonc_Num op_un_3d_real(Fonc_Num f,FONC_NAIRE_RR frr)
{
    return new   OP_Naire_Not_Comp
                 (   Rconv(f),true,
                     OP_Naire_Not_Comp::d3,
                     OP_Naire_Not_Comp::ad3,
                     frr,0,0,0
                 );
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
