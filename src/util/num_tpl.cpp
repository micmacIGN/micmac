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



ElTmplSpecNull INT   ElStdTypeScal<INT>::RtoT(REAL v) { return round_ni(v);}
ElTmplSpecNull INT   ElStdTypeScal<INT>::RTtoT(REAL v) { return round_ni(v);}
ElTmplSpecNull REAL  ElStdTypeScal<REAL>::RtoT(REAL v) { return v;}
ElTmplSpecNull REAL  ElStdTypeScal<REAL>::RTtoT(REAL v) { return v;}


ElTmplSpecNull REAL16  ElStdTypeScal<REAL16>::RtoT(REAL v) { return v;}
ElTmplSpecNull REAL16  ElStdTypeScal<REAL16>::RTtoT(REAL v) { return v;}



ElTmplSpecNull float   ElStdTypeScal<float>::RtoT(REAL v) { return float(v);}
ElTmplSpecNull float   ElStdTypeScal<float>::RTtoT(REAL v) { return float(v);}


template <class Type> void set_cste(Type * t,Type cste,INT nb)
{
    if (sizeof(Type)==1)
    {
      U_INT1 * ui1 = (U_INT1 *) & cste;
       memset(t,*ui1,nb);
    }
    if (cste == (Type) 0)
    {
       memset(t,0,sizeof(Type)*nb);
       return;
    }

    for (int i=0; i<nb ; i++)
        t[i] = cste;
}

template <class Type> void set_fonc_id(Type * t,Type v0,INT nb)
{
    for (int i=0; i<nb ; i++)
        t[i] = v0++;
}

template <class Type> void binarise(Type * t,Type v0,INT nb)
{
    for (int i=0; i<nb ; i++)
        if (t[i])
           t[i] = v0;
}

template <class Type> void binarise(Type * vout,const Type * vin,Type v0,INT nb)
{
    for (int i=0; i<nb ; i++)
        vout[i] = (vin[i] ? v0 : 0);
}

template <class Type> void neg_binarise
                           (Type * vout,const Type * vin,Type v0,INT nb)
{
    for (int i=0; i<nb ; i++)
        vout[i] = (vin[i] ? 0 : v0 );
}





template <class Type> bool values_positive (const Type *t,INT nb)
{
   for(int i=0 ; i<nb ; i++)
      if (t[i] < 0)
         return false;

   return true;
}

template <class Type> bool values_positive_strict (const Type *t,INT nb)
{
   for(int i=0 ; i<nb ; i++)
      if (t[i] <= 0)
         return false;

   return true;
}

template <class Type> bool values_all_inf_to
             (const Type *t,INT nb,Type v_max)
{
   for(int i=0 ; i<nb ; i++)
      if (t[i] >= v_max)
         return false;

   return true;
}


template <class Type> INT index_values_out_of_range
             (const Type *t,INT nb,Type v_min,Type v_max)
{
   for(int i=0 ; i<nb ; i++)
      if ( (t[i] < v_min) || (t[i] >= v_max))
         return i;

   return INDEX_NOT_FOUND;
}

template <class Type> bool values_in_range
             (const Type *t,INT nb,Type v_min,Type v_max)
{

   return
          index_values_out_of_range(t,nb,v_min,v_max)
       == INDEX_NOT_FOUND;
}



template <class Tout,class Tin> class EL_CONVERT
{
  public :
     static inline void raw_conv(Tout * out,const Tin * input,INT nb);
     static inline void soft_conv(Tout * out,const Tin * input,INT nb);

};

template <class Tout,class Tin>
          void EL_CONVERT<Tout,Tin>::raw_conv(Tout * out,const Tin * input,INT nb)
{
           memcpy(out,input,nb*sizeof(*input));
}

#if (! Compiler_Gpp2_7_2)
template <> void EL_CONVERT<INT,INT>::soft_conv(INT * out,const INT * input,INT nb)
{
    raw_conv(out,input,nb);
}

template <> void EL_CONVERT<REAL,REAL>::soft_conv(REAL * out,const REAL * input,INT nb)
{
    raw_conv(out,input,nb);
}

template <> void EL_CONVERT<REAL4,REAL4>::soft_conv(REAL4 * out,const REAL4 * input,INT nb)
{
    raw_conv(out,input,nb);
}

template <> void EL_CONVERT<U_INT1,U_INT1>::soft_conv(U_INT1 * out,const U_INT1 * input,INT nb)
{
    raw_conv(out,input,nb);
}
#endif


template <class Tout,class Tin> void EL_CONVERT<Tout,Tin>::soft_conv
                                    (Tout * out,const Tin * input,INT nb)
{
   int i;
   for(i=0 ; i<nb ; i++)
      out[i] = (Tout) input[i];
}




template <class Tout,class Tin> void convert
                                     (Tout * out,const Tin * input,INT nb)
{
    EL_CONVERT<Tout,Tin>::soft_conv(out,input,nb);
}






template <class Type> void auto_reverse_tab (Type * in_out,INT nb)
{
   Type tmp;

   for(int iplus=0, imoins = nb-1 ; iplus<imoins ; iplus++,imoins--)
   {
       tmp = in_out[iplus];
       in_out[iplus] = in_out[imoins];
       in_out[imoins] = tmp;
   }
}




template <class Type> int index_values_null(const Type * t,INT nb)
{
    for (int i=0; i<nb ; i++)
        if (t[i] == 0)
           return i;

    return INDEX_NOT_FOUND;
}

template <class Type> INT index_vmax(const Type * t,INT nb)
{
    El_Internal.ElAssert(nb>0,EEM0<< "Bas index_vmax");

    INT imax = 0;
    Type vmax = t[0];

    for (int i=1; i<nb ; i++)
        if (t[i] >vmax)
        {
           imax = i;
           vmax = t[i];
        }

    return imax;
}


         /****** Operator ***************/

template <class Type> void tab_Abs (Type * out,const Type * in,INT nb)
{
   for(int i=0 ; i<nb ; i++)
      out[i] = (in[i] > 0) ? in[i] : - in[i];
}

template <class Type> void tab_minus1 (Type * out,const Type * in,INT nb)
{
   for(int i=0 ; i<nb ; i++)
      out[i] =  - in[i];
}


template <class Type> void tab_square (Type * out,const Type * in,INT nb)
{
   for(int i=0 ; i<nb ; i++)
      out[i] =  in[i] * in[i];
}

template <class Type> void tab_cube (Type * out,const Type * in,INT nb)
{
   for(int i=0 ; i<nb ; i++)
      out[i] =  in[i] * ElSquare(in[i]);
}

template <class Type> void tab_pow4 (Type * out,const Type * in,INT nb)
{
   for(int i=0 ; i<nb ; i++)
      out[i] =  ElSquare(ElSquare(in[i]));
}

template <class Type> void tab_pow5 (Type * out,const Type * in,INT nb)
{
   for(int i=0 ; i<nb ; i++)
      out[i] =  ElSquare(ElSquare(in[i])) * in[i];
}

template <class Type> void tab_pow6 (Type * out,const Type * in,INT nb)
{
   for(int i=0 ; i<nb ; i++)
   {
      Type aV2 = ElSquare(in[i]);
      out[i] =  ElSquare(aV2) * aV2;
   }
}

template <class Type> void tab_pow7 (Type * out,const Type * in,INT nb)
{
   for(int i=0 ; i<nb ; i++)
   {
      Type aV2 = ElSquare(in[i]);
      out[i] =  ElSquare(aV2) * aV2 * in[i];
   }
}





template <class Type> void compute_inside
                          (
                               INT * res,
                               const Type * tx,
                               const Type * ty,
                               INT nb,
                               Type x0,
                               Type y0,
                               Type x1,
                               Type y1
                          )
{
     for (int i=0 ; i<nb ; i++)
     {
         Type x = *(tx++);
         Type y = *(ty++);
         res[i] =   (x >= x0)
                 && (x <  x1)
                 && (y >= y0)
                 && (y <  y1);
     }
}

template <class Type> void compute_inside
                          (
                               INT * res,
                               const Type * tx,
                               const Type * ty,
                               const Type * tz,
                               INT nb,
                               Type x0,
                               Type y0,
                               Type z0,
                               Type x1,
                               Type y1,
                               Type z1
                          )
{
     for (int i=0 ; i<nb ; i++)
     {
         Type x = *(tx++);
         Type y = *(ty++);
         Type z = *(tz++);
         res[i] =   (x >= x0)
                 && (x <  x1)
                 && (y >= y0)
                 && (y <  y1)
                 && (z >= z0)
                 && (z <  z1)
         ;
     }
}


template <class Type> void compute_inside
                          ( INT * res, const Type * tx, INT nb, Type x0, Type x1)
{
     for (int i=0 ; i<nb ; i++)
         res[i] = ((tx[i]>=x0) && (tx[i] < x1));
}


template <class Type>
        void compute_inside
            (
               INT * res,
               const Type * const *  coord,
               INT nb,
               INT dim,
               const Type *p0,
               const Type *p1
            )

{
    switch(dim)
    {
        case 1 : compute_inside
			     (
					 res,
					 coord[0],
					 nb,p0[0],p1[0]
				  );
                break;
        case 2 : compute_inside
			     (
					 res,
					 (const Type *) coord[0],
					 (const Type *) coord[1],
					 nb,p0[0],p0[1],p1[0],p1[1]
			     );
                break;

        case 3 : compute_inside
			     (
					 res,
					 (const Type *) coord[0],
					 (const Type *) coord[1],
					 (const Type *) coord[2],
					 nb,p0[0],p0[1],p0[2],p1[0],p1[1],p1[2]
			     );
                break;

        default :
               elise_internal_error("compute_inside with dim >2",__FILE__,__LINE__);
    }
}


template <class Type> void rotate_plus_data(Type * tab,INT i0,INT i1)
{
     Type tmp = tab[i0];
     for (INT i = i0+1; i<i1 ; i++)
         tab[i-1] = tab[i];
     tab[i1-1] = tmp;
}

template <class Type> void rotate_moins_data(Type * tab,INT i0,INT i1)
{

     Type tmp = tab[i1-1];
     for (INT i = i1-1; i>i0 ; i--)
          tab[i] = tab[i-1];
     tab[i0] = tmp;
}

template <class Type> void proj_in_seg
                            (
                                Type *      out,
                                const Type * in,
                                Type v_min ,
                                Type v_max,
                                INT  nb
                            )
{
    v_max--;
    while(nb--)
    {
         Type v = *(in++);
         *(out++) =
                     (v < v_min)               ?
                     v_min                     :
                     ( (v>v_max) ? v_max : v)  ;
    }
}







template <class Type> Type * dup(const Type * in,INT nb)
{
     Type * res = NEW_VECTEUR(0,nb,Type);
     convert(res,in,nb);
     return res;
}

char * dup(const char * in)
{
    return dup(in,1+(int) strlen(in));
}

char * cat(const char * ch1,const char * ch2)
{
    char * res = NEW_VECTEUR(0,(int) strlen(ch1)+(int)strlen(ch2)+1,char);
    strcpy(res,ch1);
    strcat(res,ch2);
    return res;
}

template  void  convert(REAL16 * ,  const U_INT1 *  ,INT);
template  void  convert(REAL16 * ,  const U_INT2 *  ,INT);
template  void  convert(REAL16 * ,  const INT1 *  ,INT);
template  void  convert(REAL16 * ,  const INT2 *  ,INT);
template  void  convert(REAL16 * ,  const INT *  ,INT);
template  void  convert(REAL16 * ,  const REAL16 *  ,INT);
template  void  convert(REAL16 * ,  const REAL8  *  ,INT);
template  void  convert(REAL16 * ,  const REAL4  *  ,INT);


template  void  convert(_INT8 * ,  const char  *  ,INT);
template  void  convert(_INT8 * ,  const U_INT1  *  ,INT);
template  void  convert(_INT8 * ,  const INT1  *  ,INT);
template  void  convert(_INT8 * ,  const U_INT2  *  ,INT);
template  void  convert(_INT8 * ,  const INT2  *  ,INT);
template  void  convert(_INT8 * ,  const INT  *  ,INT);
template  void  convert(_INT8 * ,  const REAL4  *  ,INT);
template  void  convert(_INT8 * ,  const REAL8  *  ,INT);
template  void  convert(_INT8 * ,  const REAL16  *  ,INT);

template  void  convert(U_INT4 * ,  const char  *  ,INT);
template  void  convert(U_INT4 * ,  const U_INT1  *  ,INT);
template  void  convert(U_INT4 * ,  const INT1  *  ,INT);
template  void  convert(U_INT4 * ,  const U_INT2  *  ,INT);
template  void  convert(U_INT4 * ,  const INT2  *  ,INT);
template  void  convert(U_INT4 * ,  const INT  *  ,INT);
template  void  convert(U_INT4 * ,  const REAL4  *  ,INT);
template  void  convert(U_INT4 * ,  const REAL8  *  ,INT);
template  void  convert(U_INT4 * ,  const REAL16  *  ,INT);
template  void  convert(U_INT4 * ,  const _INT8  *  ,INT);




template  void  convert(U_INT1 * ,  const _INT8  *  ,INT);
template  void  convert(INT1 * ,  const _INT8  *  ,INT);
template  void  convert(U_INT2 * ,  const _INT8  *  ,INT);
template  void  convert(INT2 * ,  const _INT8  *  ,INT);
template  void  convert(INT * ,  const _INT8  *  ,INT);
template  void  convert(REAL8 * ,  const _INT8  *  ,INT);
template  void  convert(REAL4 * ,  const _INT8  *  ,INT);
template  void  convert(REAL16 * ,  const _INT8  *  ,INT);
template  void  convert(_INT8 * ,  const _INT8  *  ,INT);


template  void  convert(U_INT1 * ,  const U_INT4  *  ,INT);
template  void  convert(INT1 * ,  const U_INT4  *  ,INT);
template  void  convert(U_INT2 * ,  const U_INT4  *  ,INT);
template  void  convert(INT2 * ,  const U_INT4  *  ,INT);
template  void  convert(INT * ,  const U_INT4  *  ,INT);
template  void  convert(REAL8 * ,  const U_INT4  *  ,INT);
template  void  convert(REAL4 * ,  const U_INT4  *  ,INT);
template  void  convert(REAL16 * ,  const U_INT4  *  ,INT);
template  void  convert(_INT8 * ,  const U_INT4  *  ,INT);
template  void  convert(U_INT4 * ,  const U_INT4  *  ,INT);



template  void  convert(REAL * ,  const U_INT1 *  ,INT);
template  void  convert(REAL * ,  const U_INT2 *  ,INT);
template  void  convert(REAL * ,  const INT1 *  ,INT);
template  void  convert(INT  * ,  const REAL4 *  ,INT);

template  void  convert(INT  * ,  const char *  ,INT);

#define INSTANCIATE_TYPE_BITM_GEN(Type,TypeBase)\
template  void  convert(Type * ,  const Type *  ,INT);\
template  void  convert(Type *     ,  const INT *   ,INT);\
template  void  convert(Type *     ,  const REAL *  ,INT);\
template  void  convert(TypeBase * ,  const Type *  ,INT);\
template  void  set_cste(Type * ,  Type  ,INT);


INSTANCIATE_TYPE_BITM_GEN(U_INT1,INT);
INSTANCIATE_TYPE_BITM_GEN(INT1,INT);
INSTANCIATE_TYPE_BITM_GEN(U_INT2,INT);
INSTANCIATE_TYPE_BITM_GEN(INT2,INT);
INSTANCIATE_TYPE_BITM_GEN(REAL4,REAL);

template  void  set_cste(U_INT4 * ,  U_INT4  ,INT);


#define INSTANCIATE_TYPE_BITM_BASE(Type)\
template  void  convert(Type *     ,  const INT *   ,INT);\
template  void  convert(Type *     ,  const REAL *  ,INT);\
template  void  set_cste(Type * ,  Type  ,INT);\
template void tab_Abs(Type *, const Type *,INT);\
template void tab_minus1(Type *, const Type *,INT);\
template void tab_square(Type *, const Type *,INT);\
template void tab_cube(Type *, const Type *,INT);\
template void tab_pow4(Type *, const Type *,INT);\
template void tab_pow5(Type *, const Type *,INT);\
template void tab_pow6(Type *, const Type *,INT);\
template void tab_pow7(Type *, const Type *,INT);\
template void binarise(Type *, const Type *,Type,INT);\
template void neg_binarise(Type *, const Type *,Type,INT);\
template void binarise(Type *, Type,INT);\
template bool values_positive (const Type *t,INT nb);\
template bool values_positive_strict (const Type *t,INT nb);\
template void set_fonc_id(Type * t,Type v0,INT nb);\
template bool values_in_range (const Type *t,INT nb,Type v_min,Type v_max);\
template int index_values_null(const Type * t,INT nb);\
template void compute_inside(INT *,const Type * const *,INT,INT, const Type *,const Type *);;\
template bool values_all_inf_to (const Type *t,INT nb,Type v_max);\
template void auto_reverse_tab (Type * in_out,INT nb);\
template void auto_reverse_tab (Type ** in_out,INT nb);\
template void rotate_plus_data(Type * tab,INT i0,INT i1);\
template void rotate_plus_data(Type ** tab,INT i0,INT i1);\
template void rotate_moins_data(Type * tab,INT i0,INT i1);\
template void proj_in_seg(Type *,const Type *,Type,Type,INT);\
template INT index_vmax(const Type * t,INT nb);\
template Type * dup(const Type * in,INT nb);\
template INT index_values_out_of_range (const Type *t,INT nb,Type v_min,Type v_max);\


template INT index_values_out_of_range (const REAL16 *t,INT nb,REAL16 v_min,REAL16 v_max);
template INT index_values_out_of_range (const _INT8 *t,INT nb,_INT8 v_min,_INT8 v_max);
template  void  set_cste(REAL16 * ,  REAL16  ,INT);


INSTANCIATE_TYPE_BITM_BASE(REAL);
INSTANCIATE_TYPE_BITM_BASE(INT);
template INT index_values_out_of_range(U_INT1 const *, INT, U_INT1, U_INT1);
template void rotate_plus_data(INT1 **, INT, INT);
template void rotate_plus_data(U_INT1 **, INT, INT);
template void rotate_plus_data(U_INT2 **, INT, INT);
template void binarise(U_INT1 *, U_INT1, INT);
template void rotate_plus_data(REAL4 **, INT, INT);
template void rotate_moins_data(REAL4 **, INT, INT);
template void rotate_moins_data(REAL8 **, INT, INT);
template void rotate_moins_data(INT4 **, INT, INT);

template void set_cste(char *, char, int);

template void rotate_plus_data(class Br_Skel_Vect ***, INT, INT);




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
