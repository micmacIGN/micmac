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



template <class Type>  
         class FoncATrou_OPB_Comp : public Simple_OPBuf1<INT,INT>
{
   public :

     FoncATrou_OPB_Comp(Liste_Pts<Type,INT>,INT X0,INT X1,INT Y0,INT Y1);
     FoncATrou_OPB_Comp(Liste_Pts<Type,INT>);
     virtual ~FoncATrou_OPB_Comp();

   // private :

     void  calc_buf
           (
               INT **     output,
               INT ***    input
           );

     virtual Simple_OPBuf1<INT,INT> * dup_comp();


     Im2D<Type,INT>         _values;
     Liste_Pts<Type,INT>    _l;
     Type  **                    _v;
     Type  *                     _tx;
     INT *                       _cpt;
     INT                         _dim_vals;
};

template <class Type>  
         FoncATrou_OPB_Comp<Type>::~FoncATrou_OPB_Comp()
{
    if (_cpt)
       DELETE_VECTOR(_cpt,y0());
}



template <class Type>  
         FoncATrou_OPB_Comp<Type>::FoncATrou_OPB_Comp
         (
                 Liste_Pts<Type,INT>  l
         )   :
         _values      (1,1),
         _l            (l),
         _cpt         (0),
         _dim_vals    (l.card()-2)
{
}





template <class Type>  
         FoncATrou_OPB_Comp<Type>::FoncATrou_OPB_Comp
         (
                 Liste_Pts<Type,INT>  l,
                 INT                       X0,
                 INT                       X1,
                 INT                       Y0,
                 INT                       Y1
         )   :
         _values      (1,1),
         _l            (1),
         _cpt         (NEW_VECTEUR(Y0,Y1+1,INT)),
         _dim_vals    (l.dim()-2)
{
	INT k;

     set_cste(_cpt+Y0,0,Y1-Y0+1);

     Im2D<Type,INT> XYV = l.image();

     INT nb_pts = 0;
     Type * tx = XYV.data()[0];
     Type * ty = XYV.data()[1];
     Type **   xyv  = XYV.data();


   //----  filtre les pts hors du rectangle ---------

     INT dim =        XYV.ty();

     {
          INT nb_pts_tot = XYV.tx();
          for( k=0; k<nb_pts_tot; k++)
              if (
                         (tx[k]>=X0) 
                      && (tx[k]< X1)
                      && (ty[k]>=Y0) 
                      && (ty[k]< Y1)
                 )
              {
                  for(INT d=0; d<dim; d++)
                     xyv[d][nb_pts] = xyv[d][k];
                  nb_pts++;
              }
     }

   //--------- histo en Y + cumule -----------------------
     for( k=0; k<nb_pts; k++)
         _cpt[ty[k]]++;
     for (INT y = Y0+1; y<=Y1; y++)
          _cpt[y] +=  _cpt[y-1];


    //---------------- range en y croissant ----------------
     _values = Im2D<Type,INT> (nb_pts,_dim_vals+1);
     _tx     = _values.data()[0];
     _v      = _values.data()+1;

     for( k=0; k<nb_pts ; k++)
     {
         INT adr =  --_cpt[ty[k]];
         _tx[adr] =  tx[k];
         for (INT d = 0; d<_dim_vals; d++)
             _v[d][adr] = xyv[d+2][k];
     }
}

template <class Type>   
         Simple_OPBuf1<INT,INT> *
         FoncATrou_OPB_Comp<Type>::dup_comp()
{

    return new
           FoncATrou_OPB_Comp<Type>
           (
                _l,
                x0(),
                x1(),
                y0(),
                y1()
           );
}


template <class Type>   
         void FoncATrou_OPB_Comp<Type>::calc_buf
         (
               INT **     output,
               INT ***    input
         )
{
    for (INT d= 0; d<dim_out() ; d++)
    {
        convert
        (
             output[d]+x0(),
             input[ElMin(d,dim_in()-1)][0]+x0(),
             tx()
        );
    }

     INT a0 = _cpt[ycur()];
     INT a1 = _cpt[ycur()+1];
     for (INT a=a0 ; a<a1 ; a++)
     {
         INT x = _tx[a];
         for (INT d= 0; d<dim_out() ; d++)
             output[d][x] = _v[d][a];
     }
}


template <class Type> Fonc_Num Tfonc_a_trou(Fonc_Num f,Liste_Pts<Type,INT> l)
{
     Tjs_El_User.ElAssert
     (
          f.dimf_out() <= l.dim()-2,
          EEM0 << "fonc_a_trou  : Fonc_Num.dimf_out() > Liste_Pts.dim()-2"
     );

     return create_op_buf_simple_tpl
            (
                 new FoncATrou_OPB_Comp<Type> (l),
                 0,
                 f,
                 l.dim()-2,
                 Box2di(0)
            );
}


Fonc_Num fonc_a_trou(Fonc_Num f,Liste_Pts<INT,INT> l)
{
    return Tfonc_a_trou(f,l);
}

Fonc_Num fonc_a_trou(Fonc_Num f,Liste_Pts<U_INT2,INT> l)
{
    return Tfonc_a_trou(f,l);
}

Fonc_Num fonc_a_trou(Fonc_Num f,Liste_Pts<INT2,INT> l)
{
    return Tfonc_a_trou(f,l);
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
