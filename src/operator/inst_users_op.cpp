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
/*             Fnum_Ecart_Circ                                 */
/*                                                             */
/***************************************************************/

class Fnum_Ecart_Circ : public  Simple_OP_UN<REAL>
{
     public :
         Fnum_Ecart_Circ(REAL per,int din) : 
                 _per (per), 
                 _tmp ((din>=0) ? (NEW_TAB(din+1,REAL))  : 0) 
         {}


         virtual  ~Fnum_Ecart_Circ() 
         { 
                if (_tmp) 
                    DELETE_TAB(_tmp);
         }

     private :

         virtual Simple_OP_UN<REAL> *  dup_comp(const Arg_Comp_Simple_OP_UN & arg)
         {
              return new Fnum_Ecart_Circ(_per,arg.dim_in());
         }
      
         virtual void calc_buf
                      (  
                         REAL ** output,
                         REAL ** input,
                         INT        nb,
                         const Arg_Comp_Simple_OP_UN  &
                      );

         REAL _per;
         REAL * _tmp;
};

void Fnum_Ecart_Circ::calc_buf 
( 
        REAL ** output,
        REAL ** input,
        INT        nb,
        const Arg_Comp_Simple_OP_UN  & arg
)
{
    INT din = arg.dim_in();

    for (INT i=0; i<nb ; i++)
    {
        for (INT d=0; d<din ; d++)
            _tmp[d] = input[d][i];
        elise_sort(_tmp,din);
        _tmp[din] = _tmp[0] + _per;
        REAL ec_max = _tmp[1]-_tmp[0];
        {
            for (INT d =0; d<din; d++)
                 ec_max = ElMax (ec_max,_tmp[d+1]-_tmp[d]);
        }
        output[0][i] = _per -ec_max;
    }
}


Fonc_Num ecart_circ(Fonc_Num f,REAL per)
{
    return create_users_oper
           (
               0,
               new Fnum_Ecart_Circ(per,-1),
               f,
               1
           );
}



/***************************************************************/
/*                                                             */
/*             Fnum_Grad_Bilin                                 */
/*                                                             */
/***************************************************************/

class Fnum_Grad_Bilin : public  Simple_OP_UN<REAL>
{
     public :
         Fnum_Grad_Bilin(Im2D_U_INT1 b) : _b (b), _d (b.data()) {}
     private :


         Im2D_U_INT1 _b;
         U_INT1 **   _d;
         virtual void calc_buf
                      (  
                           REAL ** output,
                           REAL ** input,
                           INT        nb,
                           const Arg_Comp_Simple_OP_UN  &
                      );

};

void Fnum_Grad_Bilin::calc_buf
( 
        REAL ** output,
        REAL ** input,
        INT        nb,
        const Arg_Comp_Simple_OP_UN  & arg
)
{
    REAL * x = input[0];
    REAL * y = input[1];

    REAL * gx = output[0];
    REAL * gy = output[1];


    Tjs_El_User.ElAssert
    (
        arg.dim_in() == 2,
        EEM0 << "need 2-d Func for grad_bilin"
    );


    Tjs_El_User.ElAssert
    ( 
             values_in_range(x,nb,1.0,_b.tx()-1.0)
          && values_in_range(y,nb,1.0,_b.ty()-1.0),
          EEM0 << "Out of range in grad_bilin \n"
    );


   for (int i=0 ; i<nb ; i++)
   {
       INT xi,yi;
       REAL x0 = x[i];
       REAL y0 = y[i];
       REAL p_1x = x0 - (xi= (INT) x0);
       REAL p_1y = y0 - (yi= (INT) y0);
       REAL p_0x = 1.0-p_1x;
       REAL p_0y = 1.0-p_1y;

       INT v_00 = _d[yi][xi];
       INT v_10 = _d[yi][xi+1];
       INT v_01 = _d[yi+1][xi];
       INT v_11 = _d[yi+1][xi+1];

       gx[i] =   p_0y * (v_10-v_00) + p_1y * (v_11-v_01);
       gy[i] =   p_0x * (v_01-v_00) + p_1x * (v_11-v_10);
   }
}

Fonc_Num grad_bilin(Fonc_Num f,Im2D_U_INT1 b)
{
    return create_users_oper
           (
               0,
               new Fnum_Grad_Bilin(b),
               f,
               2
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
