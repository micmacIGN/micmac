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

#if Compiler_Gpp2_7_2
#pragma implementation
#pragma interface
#endif


/********************************************************************/
/*                                                                  */
/*       Out_adapt_type_Fonc                                        */
/*                                                                  */
/********************************************************************/

template <class  TyFonc,class TyOut> class Out_adapt_type_Fonc
         : public Output_Computed
{
        
      public :

           virtual ~Out_adapt_type_Fonc()
           {
               delete _pck_adapt;
               delete _o;
           }

           virtual void update( const Pack_Of_Pts * pts,
                                const Pack_Of_Pts * vals)
           {
                   _pck_adapt->convert_from(vals);
                   _o->update(pts,_pck_adapt);
           }


           Out_adapt_type_Fonc
           (
                const Arg_Output_Comp & arg,
                Output_Computed        * o
           )  :
                 Output_Computed(o->dim_consumed()),
                 _o             (o),
                 _pck_adapt     (Std_Pack_Of_Pts<TyOut>::new_pck
                                     (  o->dim_consumed(),
                                        arg.flux()->sz_buf()
                                     )
                                )
           {
           }

      private :
          Output_Computed *          _o;
          Std_Pack_Of_Pts<TyOut> *   _pck_adapt;
};


Output_Computed * out_adapt_type_fonc 
                  (    const Arg_Output_Comp & arg,
                       Output_Computed       * o,
                       Pack_Of_Pts::type_pack type_wished
                  )
{

    if (type_wished == arg.fonc()->type_out())
       return o;

    if (type_wished == Pack_Of_Pts::integer)
       return new Out_adapt_type_Fonc<REAL,INT>(arg,o);
    else
       return new  Out_adapt_type_Fonc<INT,REAL>(arg,o);
}

/********************************************************************/
/*                                                                  */
/*       RLE_clip_output<Type>                                      */
/*                                                                  */
/********************************************************************/


template <class Type> class RLE_clip_output : public Output_Computed
{
    public :
       RLE_clip_output
       (
            Output_Computed * o,
            const Arg_Output_Comp & arg,
            const INT * p0,
            const INT * p1
       );
        


       virtual ~RLE_clip_output()
       {
              delete _val_cliped;
              delete _pts_cliped;
              delete _o;
       }
    private :

       virtual void update( const Pack_Of_Pts * pts,
                            const Pack_Of_Pts * vals)
       {
           INT n0 = _pts_cliped->clip(SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(pts)),_p0,_p1);
           _val_cliped->interv
           (    SAFE_DYNC( Std_Pack_Of_Pts<Type> *,const_cast<Pack_Of_Pts *>(vals)),
                n0,
                n0+_pts_cliped->nb()
           );
           _o->update(_pts_cliped,_val_cliped);
       }


       Output_Computed *         _o;
       RLE_Pack_Of_Pts *         _pts_cliped;
       Std_Pack_Of_Pts<Type> *   _val_cliped;   

       INT  _p0[Elise_Std_Max_Dim];
       INT  _p1[Elise_Std_Max_Dim];
};


template <class Type> 
       RLE_clip_output<Type>::RLE_clip_output
       (
            Output_Computed * o,
            const Arg_Output_Comp & arg,
            const INT * p0,
            const INT * p1
       ) :
         Output_Computed(o->dim_consumed()),
         _o (o),
         _pts_cliped (RLE_Pack_Of_Pts::new_pck
                        ( arg.flux()->dim(),
                          RLE_Pack_Of_Pts::_sz_buf_infinite
                        )
                     ),
         _val_cliped (Std_Pack_Of_Pts<Type>::new_pck
                                     (o->dim_consumed(),0)
                     )
{
     convert(_p0,p0,arg.flux()->dim());
     convert(_p1,p1,arg.flux()->dim());
}


/********************************************************************/
/*                                                                  */
/*       STD_clip_output<Type_val,Type_Pts>                         */
/*                                                                  */
/********************************************************************/

template <class Type_val,class Type_Pts> class STD_clip_output : public Output_Computed
{
    public :
       STD_clip_output
       (
            Output_Computed * o,
            const Arg_Output_Comp & arg,
            const INT * p0,
            const INT * p1
       );
        
       virtual ~STD_clip_output()
       {
              delete _val_cliped;
              delete _pts_cliped;
              delete _pack_filtr;
              delete _o;
       }

    private :
       virtual void update( const Pack_Of_Pts * pts,
                            const Pack_Of_Pts * vals)
       {
             const Std_Pack_Of_Pts<Type_val> * pvals =
                   SAFE_DYNC(const Std_Pack_Of_Pts<Type_val> *,vals);
             const Std_Pack_Of_Pts<Type_Pts> *      ppts
                   =  SAFE_DYNC(const  Std_Pack_Of_Pts<Type_Pts> *,pts);

             INT * filtr = _pack_filtr->_pts[0];

             ::compute_inside(filtr,ppts->_pts,ppts->nb(),ppts->dim(),_p0,_p1);

             _pts_cliped->set_nb(0);
             _val_cliped->set_nb(0);
             ppts->select_tab(_pts_cliped,filtr);
             pvals->select_tab(_val_cliped,filtr);

           _o->update(_pts_cliped,_val_cliped);
       }


       Output_Computed *             _o;
       Std_Pack_Of_Pts<INT>      *   _pack_filtr;
       Std_Pack_Of_Pts<Type_Pts> *   _pts_cliped;
       Std_Pack_Of_Pts<Type_val> *   _val_cliped;   

       INT  _p0[Elise_Std_Max_Dim];
       INT  _p1[Elise_Std_Max_Dim];
};


template <class Type_val,class Type_Pts>
       STD_clip_output<Type_val,Type_Pts>::STD_clip_output
       (
            Output_Computed * o,
            const Arg_Output_Comp & arg,
            const INT * p0,
            const INT * p1
       ) :
         Output_Computed(o->dim_consumed()),
         _o (o),
         _pack_filtr     (Std_Pack_Of_Pts<INT>::new_pck
                                         (1,arg.flux()->sz_buf())
                         ),
         _pts_cliped     (Std_Pack_Of_Pts<Type_Pts>::new_pck
                                         (arg.flux()->dim(),arg.flux()->sz_buf())
                         ),
         _val_cliped     (Std_Pack_Of_Pts<Type_val>::new_pck
                                         (o->dim_consumed(),arg.flux()->sz_buf())
                         )
{
     convert(_p0,p0,arg.flux()->dim());
     convert(_p1,p1,arg.flux()->dim());
}


/********************************************************************/
/*                                                                  */
/*       Top Level Interface Function.                              */
/*                                                                  */
/********************************************************************/

/*

     Il n'y a pas pour l'instant d'interface utilisateur a clip_out_put;

     Si on le fait, ne pas oublier de faire un flucx interface pour
    eviter un bug tel que :

      Fnum_Symb f (FX);

      copy
      (
           ...,
           f,
           clip(o<<(f+1)) | ...
      );

*/

Output_Computed * clip_out_put
                 ( 
                       Output_Computed * o,
                       const Arg_Output_Comp & arg,
                       const INT * p0,
                       const INT * p1
                  ) 
{
    switch(arg.flux()->type())
    {

        case Pack_Of_Pts::rle :

	   if (arg.fonc()->integral())
              return new RLE_clip_output<INT> (o,arg,p0,p1);
           else 
	      return new RLE_clip_output<REAL>(o,arg,p0,p1);


        case Pack_Of_Pts::integer :

             if (arg.fonc()->integral())        
                return new STD_clip_output<INT,INT>  (o,arg,p0,p1);
             else 
	        return new STD_clip_output<REAL,INT> (o,arg,p0,p1);

        default :
            elise_internal_error
                ("can only handle RLE and integer mode in out clip",__FILE__,__LINE__);
            return 0;
    }
}


Output_Computed * clip_out_put
                 ( 
                       Output_Computed * o,
                       const Arg_Output_Comp & arg,
                       Pt2di p0,
                       Pt2di p1
                  ) 
{
    INT c0[2];     p0.to_tab(c0);
    INT c1[2];     p1.to_tab(c1);

    return clip_out_put(o,arg,c0,c1);
}





/********************************************************************/
/*                                                                  */
/*       Out_adapt_type_Pts                                         */
/*                                                                  */
/********************************************************************/

class Out_adapt_type_Pts : public Output_Computed
{
        
      public :

           virtual ~Out_adapt_type_Pts()
           {
               delete _pck_adapt;
               delete _o;
           }

           virtual void update( const Pack_Of_Pts * pts,
                                const Pack_Of_Pts * vals)
           {
                   _pck_adapt->convert_from(pts);
                   _o->update(_pck_adapt,vals);
           }


           Out_adapt_type_Pts
           (
                const Arg_Output_Comp & arg,
                Output_Computed        * o,
                Pack_Of_Pts::type_pack  type_pack
           )  :
                 Output_Computed(o->dim_consumed()),
                 _o             (o),
                 _pck_adapt     (Pack_Of_Pts::new_pck
                                     (  arg.flux()->dim(),
                                        arg.flux()->sz_buf(),
                                        type_pack
                                     )
                                )
           {
           }

      private :
          Output_Computed *          _o;
          Pack_Of_Pts     *          _pck_adapt;
};


Output_Computed * out_adapt_type_pts 
                  (    const Arg_Output_Comp & arg,
                       Output_Computed       * o,
                       Pack_Of_Pts::type_pack type_wished
                  )
{

    if (type_wished == arg.flux()->type())
       return o;
    else
       return new Out_adapt_type_Pts(arg,o,type_wished);
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
