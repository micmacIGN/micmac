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

//#pragma implementation


/***************************************************************/
/*                                                             */
/*          Out_Reduction_Computed                             */
/*                                                             */
/***************************************************************/


template <class TFonc,class TObj> class Out_Reduction_Computed : 
                                        public Output_Computed
{
     public : 


       void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals)
       {
            const Std_Pack_Of_Pts<TFonc> * v = 
                  SAFE_DYNC(const Std_Pack_Of_Pts<TFonc> *,vals);

            for (int i=0; i<_nb ; i++)
                _res_fonc[i] = _op.red_tab( v->_pts[i],pts->nb(),_res_fonc[i]);

       } 

       Out_Reduction_Computed(const OperAssocMixte & op,TObj * res, INT nb) :
               Output_Computed(nb),
              _op     (op),
              _res_user    (res),
              _res_fonc (NEW_VECTEUR(0,nb,TFonc)),
              _nb     (nb)
        {
             for(int i=0 ; i<_nb ; i++)
                _op.set_neutre(_res_fonc[i]);
        }

          ~Out_Reduction_Computed()
          {
               convert(_res_user,_res_fonc,_nb);
               DELETE_VECTOR(_res_fonc,0);
          }

      private :
          const OperAssocMixte & _op;
          TObj                 * _res_user;
          TFonc                * _res_fonc;
          INT                   _nb;

};

/***************************************************************/
/*                                                             */
/*          Out_Reduction_Not_Comp                             */
/*                                                             */
/***************************************************************/



template <class TObj> class Out_Reduction_Not_Comp : public Output_Not_Comp
{
      public :


          Out_Reduction_Not_Comp(const OperAssocMixte & op,TObj * res, INT nb) :
              _op     (op),
              _res    (res),
              _nb     (nb)
          {
          }

          Output_Computed * compute (const Arg_Output_Comp & arg )
          {

               Tjs_El_User.ElAssert
               (  
                    arg.fonc()->idim_out() >= _nb,
                   EEM0 << "incompatible dimension in Out Reduction |\n" 
                        << " dim fonc = " <<  arg.fonc()->idim_out()
                        <<  "   required : " << _nb
               );

               if (arg.fonc()->integral())
                   return new Out_Reduction_Computed<INT,TObj> (_op,_res,_nb) ;
               else
                   return new Out_Reduction_Computed<REAL,TObj> (_op,_res,_nb) ;
          }
         
      private :
          const OperAssocMixte & _op;
          TObj                 * _res;
          INT                    _nb;
};



/***************************************************************/
/*                                                             */
/*          interface function                                 */
/*                                                             */
/***************************************************************/

template <class Type>    Output  reduc(const OperAssocMixte & op,Type * res,INT nb)
{
      return new Out_Reduction_Not_Comp<Type>(op,res,nb);
}

template <class Type>    Output  reduc(const OperAssocMixte & op,Type & res)
{
      return  reduc(op,&(res),1);
}

     //------------------------------------

template <class Type>    Output  sigma(Type * res,INT nb)
{
    return reduc(OpSum,res,nb);
}

template <class Type>    Output  sigma(Type & res)
{
    return sigma(&res,1);
}

/*
Output  sigma(Pt2di p)
{
    return sigma(&p.x,2);
}
*/


     //------------------------------------

template <class Type>    Output  VMax(Type * res,INT nb)
{
    return reduc(OpMax,res,nb);
}

template <class Type>    Output  VMax(Type & res)
{
    return VMax(&res,1);
}


     //------------------------------------

template <class Type>    Output  VMin(Type * res,INT nb)
{
    return reduc(OpMin,res,nb);
}

template <class Type>    Output  VMin(Type & res)
{
    return VMin(&res,1);
}



/***************************************************************/
/*                                                             */
/*          Witch_Optim_Computed                               */
/*                                                             */
/***************************************************************/


template <class TFonc,class TFlx> class Witch_Optim_Computed : 
                                        public Output_Computed
{
     public : 


       void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals)
       {
            TFonc * v = vals->std_cast((TFonc *) 0)->_pts[0];
            TFlx  ** p = pts->std_cast((TFlx *) 0)->_pts;

            INT nb = pts->nb();

            for (int k=0; k<nb ; k++)
            {
                if ((v[k] > _optim) == _mode_max)
                {
                      _optim = v[k];
                      for (INT d = 0; d <_dim ; d++)
                          _popt[d] = p[d][k];
                }
            } 
       } 

       Witch_Optim_Computed(bool mode_max,TFlx * popt,INT dim) :
               Output_Computed   (1),
               _mode_max         (mode_max),
               _popt             (popt),
               _dim              (dim)
        {
               if (mode_max)
                  OpMax.set_neutre(_optim);
               else
                  OpMin.set_neutre(_optim);
        }

        ~Witch_Optim_Computed()
        {
        }

      private :
          bool         _mode_max;
          TFlx *       _popt;
          INT          _dim;
          TFonc        _optim;

};


template <class TFlx> class Witch_Optim_Not_Comp : public Output_Not_Comp
{
      public :


          Witch_Optim_Not_Comp(bool mode_max,TFlx * popt, INT dim) :
               _mode_max         (mode_max),
               _popt             (popt),
               _dim              (dim)
          {
          }

          Output_Computed * compute (const Arg_Output_Comp & arg )
          {
               Tjs_El_User.ElAssert
               (  
                    arg.flux()->dim() == _dim,
                   EEM0 << "incompatible dimension in WitchMax/WitchMin |\n" 
                        << " dim flux = " <<  arg.flux()->dim()
                        <<  "   required : " << _dim
               );

               Output_Computed * o = 0;


               if (arg.fonc()->integral())
                   o = new Witch_Optim_Computed<INT,TFlx> (_mode_max,_popt,_dim);
               else
                   o =  new Witch_Optim_Computed<REAL,TFlx> (_mode_max,_popt,_dim);

               return out_adapt_type_pts(arg,o,_tpts_required);
          }
         
      private :

          static const Pack_Of_Pts::type_pack    _tpts_required;
          bool                                   _mode_max;
          TFlx *                                 _popt;
          INT                                    _dim;
};

template <> const Pack_Of_Pts::type_pack Witch_Optim_Not_Comp<INT >::_tpts_required = Pack_Of_Pts::integer;
template <> const Pack_Of_Pts::type_pack Witch_Optim_Not_Comp<REAL>::_tpts_required = Pack_Of_Pts::real;



template <class Type>    Output  WhichMin(Type * pts,INT dim)
{
    return new Witch_Optim_Not_Comp<Type>(false,pts,dim);
}

template <class Type>    Output  WhichMin(Type & res)
{
    return WhichMin(&res,1);
}


template <class Type>    Output  WhichMax(Type * pts,INT dim)
{
    return new Witch_Optim_Not_Comp<Type>(true,pts,dim);
}

template <class Type>    Output  WhichMax(Type & res)
{
    return WhichMax(&res,1);
}




     //------------------------------------
     //------------------------------------
     //------------------------------------
     //------------------------------------

void HHHH_INSTANTIATE_REDUC()
{
    {
       INT x;
       VMin(x); VMax(x); sigma(x);
    }

    {
       REAL x;
       VMin(x); VMax(x); sigma(x);
    }
}

#define INSTANTIATE_REDUC(Type)\
template Output WhichMax(Type &);\
template Output WhichMin(Type &);\
template Output sigma(Type &);\
template Output VMax<Type>(Type &);\
template Output VMin<Type>(Type &);\
template Output reduc(OperAssocMixte const &,Type &);\
template Output WhichMax(Type *,INT);\
template Output WhichMin(Type *,INT);\
template Output sigma(Type *,INT);\
template Output VMax(Type *,INT);\
template Output VMin(Type *,INT);\
template Output reduc(OperAssocMixte const &,Type *,INT);


INSTANTIATE_REDUC(INT);
INSTANTIATE_REDUC(REAL);

template <> Output WhichMax<REAL16>(REAL16 *,int)
{
   return Output::onul(1);
}
template <> Output WhichMin<REAL16>(REAL16 *,int)
{
   return Output::onul(1);
}

template Output VMax(REAL16 *,int);
template Output VMin(REAL16 *,int);
template Output sigma(REAL16 *,int);
template Output VMax(REAL16 &);
template Output VMin(REAL16 &);
template Output sigma(REAL16 &);





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
