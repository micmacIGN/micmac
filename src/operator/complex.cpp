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


Fonc_Num cast_complex(Fonc_Num f)
{
     switch(f.dimf_out())
     {
           case 2 : return f;
           case 1 : return  Virgule (f,Fonc_Num(0.0));

           default :
              elise_fatal_error("incompatible dimension in complex-operator",__FILE__,__LINE__);
              return 0;
     }
}

/*****************************************************************/
/*              Unary operator                                   */
/*****************************************************************/


            /*===================================*/
            /*         square_compl              */
            /*===================================*/

void  square_compl_gen
      (  
               bool  mWithDivNorm,
               REAL ** out,
               REAL ** in,
               INT     nb,
               const Arg_Comp_Simple_OP_UN&
      )
{
    REAL * x = out[0];
    REAL * y = out[1];
    const REAL * tx = in[0];
    const REAL * ty = in[1];
    REAL x0,y0;

    for(int i=0; i<nb ; i++)
    {
        x0 = *(tx++);
        y0 = *(ty++);
        x[i] = x0*x0-y0*y0;
        y[i] = 2*x0*y0;
        if (mWithDivNorm)
        {
            double aN  = hypot(x0,y0);
            if (aN > 0)
            {
                 x[i] /= aN;
                 y[i] /= aN;
            }
        }
    }
}

void  square_compl
      (  
               REAL ** out,
               REAL ** in,
               INT     nb,
               const Arg_Comp_Simple_OP_UN& anArg
      )
{
   square_compl_gen(false,out,in,nb,anArg);
}
Fonc_Num squarec(Fonc_Num f)
{
     return create_users_oper(0,square_compl,cast_complex(f),2);
}


void  square_compl_divN
      (  
               REAL ** out,
               REAL ** in,
               INT     nb,
               const Arg_Comp_Simple_OP_UN& anArg
      )
{
   square_compl_gen(true,out,in,nb,anArg);
}
Fonc_Num squarec_divN(Fonc_Num f)
{
     return create_users_oper(0,square_compl_divN,cast_complex(f),2);
}



            /*===================================*/
            /*         divc                      */
            /*===================================*/

void  inv_compl
      (  
               REAL ** out,
               REAL ** in,
               INT     nb,
               const Arg_Comp_Simple_OP_UN&
      )
{
    REAL * x = out[0];
    REAL * y = out[1];
    const REAL * tx = in[0];
    const REAL * ty = in[1];
    REAL x0,y0,n2;
    
    ASSERT_USER
    (
        index_values_complex_nul(tx,ty,nb) == INDEX_NOT_FOUND,
        "null complexe in divc"
    );

    for(int i=0; i<nb ; i++)
    {
        x0 = *(tx++);
        y0 = *(ty++);
        n2 = x0*x0+y0*y0;
        x[i] = x0/ n2;
        y[i] = -y0/n2;
    }
}


Fonc_Num divc(Fonc_Num f)
{
     return create_users_oper
            (
                 0,
                 inv_compl,
                 cast_complex(f),
                 2
            );
}

            /*===================================*/
            /*         polar                     */
            /*===================================*/


void  polar
      (  
               REAL ** out,
               REAL ** in,
               INT     nb,
               const Arg_Comp_Simple_OP_UN&
      )
{
    REAL * rho  = out[0];
    REAL * teta = out[1];
    const REAL * tx = in[0];
    const REAL * ty = in[1];
    
    ASSERT_USER
    (
        index_values_complex_nul(tx,ty,nb) == INDEX_NOT_FOUND,
        "null complexe in polar"
    );

    for(int i=0; i<nb ; i++)
    {
        *(rho++)  =  hypot(*tx,*ty);
        *(teta++) =  atan2(*(ty++),*(tx++));
    }
}


Fonc_Num polar(Fonc_Num f)
{
     return create_users_oper
            (
                 0,
                 polar,
                 cast_complex(f),
                 2
            );
}
Fonc_Num TronkUC(Fonc_Num aF)
{
   return Max(0,Min(255,aF));
}

Fonc_Num  Hypot(Fonc_Num aFxyz)
{
   Fonc_Num aRes = 0;
   for (int aK=0 ; aK<aFxyz.dimf_out() ; aK++)
       aRes = aRes + Square(aFxyz.kth_proj(aK));

   return sqrt(aRes);
}

Fonc_Num  Scal(Fonc_Num aF1,Fonc_Num aF2)
{
   ELISE_ASSERT(aF1.dimf_out()==aF2.dimf_out(),"Dims diff in Scal");
   Fonc_Num aRes = 0;
   for (int aK=0 ; aK<aF1.dimf_out() ; aK++)
       aRes = aRes + aF1.kth_proj(aK)*aF2.kth_proj(aK);

   return aRes;
}



Fonc_Num  Hypot(Fonc_Num aFx,Fonc_Num aFy)
{
   return Hypot(Virgule(aFx,aFy));
}


             /*===================================*/
             /*         polar-def                 */
             /*===================================*/
 
/*
 class Polar_Def_Opun : public Simple_OP_UN<REAL>
 {
 
    public :
       Polar_Def_Opun(REAL teta_def) : _teta_def (teta_def) {}
 
 	  static Fonc_Num polar(Fonc_Num f,REAL teta0);
 
    private  :
       virtual void calc_buf
       (
            REAL **,
            REAL**,
            INT,
            const Arg_Comp_Simple_OP_UN  &
       );
       REAL _teta_def;
 
 };
*/


void  Polar_Def_Opun::calc_buf
      (
           REAL **   out,
           REAL **   in,
           INT       nb,
           const Arg_Comp_Simple_OP_UN  &
      )
{
    REAL * rho = out[0];
    REAL * teta = out[1];
    const REAL * tx = in[0];
    const REAL * ty = in[1];

    for(int i=0; i<nb ; i++)
       if (*tx || *ty)
       {
           *(rho++)  =  hypot(*tx,*ty);
           *(teta++) =  atan2(*(ty++),*(tx++));
       }
       else
       {
           *(rho++)  =  0.0;
           *(teta++) =  _teta_def;
           tx++;
           ty++;
       }
}



Fonc_Num Polar_Def_Opun::polar(Fonc_Num f,REAL teta0)
{
     return create_users_oper
            (
                 0,
                 new Polar_Def_Opun(teta0),
                 cast_complex(f),
                 2
            );
}


Fonc_Num polar(Fonc_Num f,REAL teta0)
{
   return  Polar_Def_Opun::polar(f,teta0);
}

/*****************************************************************/
/*              Binary operator                                  */
/*****************************************************************/


            /*===================================*/
            /*         mul_compl                 */
            /*===================================*/

void  mul_compl(    REAL ** out,
                    REAL ** in0,
                    REAL ** in1,
                    INT     nb,
                    const Arg_Comp_Simple_OP_BIN&
              )
{

    REAL * x = out[0];
    REAL * y = out[1];


    const REAL * x0 = in0[0];
    const REAL * y0 = in0[1];
    const REAL * x1 = in1[0];
    const REAL * y1 = in1[1];


    for(int i=0; i<nb ; i++)
    {
        x[i] = x0[i]*x1[i] - y0[i]*y1[i];
        y[i] = x0[i]*y1[i] + y0[i]*x1[i];
    }
}



Fonc_Num mulc(Fonc_Num f1,Fonc_Num f2)
{
     return create_users_oper
            (
                 0,
                 mul_compl,
                 cast_complex(f1),
                 cast_complex(f2),
                 2
            );
}


            /*===================================*/
            /*         div_compl                 */
            /*===================================*/

void  div_compl(    REAL ** out,
                    REAL ** in0,
                    REAL ** in1,
                    INT     nb,
                    const Arg_Comp_Simple_OP_BIN&
              )
{
    REAL * x = out[0];
    REAL * y = out[1];

    const REAL * tx0 = in0[0];
    const REAL * ty0 = in0[1];
    const REAL * tx1 = in1[0];
    const REAL * ty1 = in1[1];
    
    ASSERT_USER
    (
        index_values_complex_nul(tx1,ty1,nb) == INDEX_NOT_FOUND,
        "null complexe in divc"
    );

    REAL x1,y1,n2;
    for(int i=0; i<nb ; i++)
    {
        x1 = *(tx1++);
        y1 = *(ty1++);
        n2 = x1*x1+y1*y1;
        x[i] = (tx0[i]*x1 +ty0[i]*y1)/n2;
        y[i] = (-tx0[i]*y1+ ty0[i]*x1)/n2;
    }
}
Fonc_Num divc(Fonc_Num f1,Fonc_Num f2)
{
     return create_users_oper
            (
                 0,
                 div_compl,
                 cast_complex(f1),
                 cast_complex(f2),
                 2
            );
}

/***************************************

void  pow_real_compl(    REAL * x,REAL * y,
                         const REAL * tx0,
                         const REAL * tx1,const REAL*  ty1,
                         INT nb,
                         Data_Arg_Opt *
                )
{
    REAL x0,x1,y1;
    REAL rho,teta;
    
    ASSERT_USER
    (
        index_values_complex_nul(tx1,ty1,nb) == INDEX_NOT_FOUND,
        "null complexe in divc"
    );

    for(int i=0; i<nb ; i++)
    {
        x0 = *(tx0++);
        x1 = *(tx1++);
        y1 = *(ty1++);
        if (x0 >= 0)
        {
           rho  = pow(x0,x1);
           teta = log(x0) * y1;
        }
        else
        {
           rho  = pow(-x0,x1) * exp(-PI*y1);
           teta = log(-x0) * y1 +PI * x1;
        }

        x[i] = rho * cos(teta);
        y[i] = rho * sin(teta);
    }
}
*********************************/



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
