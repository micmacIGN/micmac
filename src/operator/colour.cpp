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

static inline REAL get_sat(REAL i,REAL vmi)
{
    if (vmi <0)
       return (i != 0) ?    (-vmi/i) : 1e20;
    else
       return (i != 1.0) ? (vmi/(1-i)) : 1e20;
}

static REAL get_sat (REAL i,Pt3dr vmi)
{
    REAL s = get_sat(i,vmi.x);
    s = ElMax(s,get_sat(i,vmi.y));
    s = ElMax(s,get_sat(i,vmi.z));

    return s;
}

static Pt3dr VecRed =   Pt3dr(2,-1,-1)/euclid(Pt3dr(2,-1,-1));
static Pt3dr VecGreen = Pt3dr(0,1,-1) /euclid(Pt3dr(0,1,-1));

REAL Elise_colour::adj_rvb(REAL v)
{
     return ElMax(0.0,ElMin(1.0,v));
}

void Elise_colour::to_its(REAL & i,REAL & t, REAL & s)
{

     REAL x = r();
     REAL y = g();
     REAL z = b();

     i = (x+y+z)/3;
     Pt3dr Pts(x-i,y-i,z-i);
     s = get_sat(i,Pts);
     s = ElMin(s,1.0);


     REAL cr = scal(Pts,VecRed);
     REAL cg = scal(Pts,VecGreen);

     if (cr || cg)
     {
         t = atan2(cg,cr) / (2*PI);
         while (t < 0.0) t += 1;
         if (t>=1.0)
            t = 0.0;
     }
     else
        t = 0;
}

Elise_colour Elise_colour::its(REAL  i,REAL  t, REAL  s)
{
     t        *= 2 * PI;

     Pt3dr vmi = VecRed*cos(t)+VecGreen*sin(t);
     
     REAL sat = get_sat(i,vmi);

     Pt3dr c = Pt3dr(i,i,i) + vmi*(s/sat);

    return Elise_colour::rgb
           (
                 adj_rvb(c.x),
                 adj_rvb(c.y),
                 adj_rvb(c.z)
           );
     
}


void  rgb_to_its 
      (
          REAL ** out_put,
          REAL ** in_put,
          INT     nb,
          const Arg_Comp_Simple_OP_UN& arg
      )
{
    Tjs_El_User.ElAssert
    (
        arg.dim_in() == 3,
        EEM0 << "bas dim fonc in rgb_to_its"
    );

    REAL * r = in_put[0];
    REAL * g = in_put[1];
    REAL * b = in_put[2];

    REAL * i = out_put[0];
    REAL * t = out_put[1];
    REAL * s = out_put[2];

    for (INT k=0 ; k< nb ; k++)
    {
        Elise_colour c = Elise_colour::rgb(r[k]/255,g[k]/255,b[k]/255);
        c.to_its(i[k],t[k],s[k]);
        i[k] *= 255;
        t[k] *= 255;
        s[k] *= 255;
    }
}

Fonc_Num rgb_to_its(Fonc_Num f)
{
    return create_users_oper
           (
               0,
               rgb_to_its,
               f,
               3
           );
}


void  its_to_rgb 
      (
          REAL ** out_put,
          REAL ** in_put,
          INT     nb,
          const Arg_Comp_Simple_OP_UN& arg
      )
{
    Tjs_El_User.ElAssert
    (
        arg.dim_in() == 3,
        EEM0 << "bas dim fonc in rgb_to_its"
    );

    REAL * i = in_put[0];
    REAL * t = in_put[1];
    REAL * s = in_put[2];

    REAL * r = out_put[0];
    REAL * g = out_put[1];
    REAL * b = out_put[2];

    for (INT k=0 ; k< nb ; k++)
    {
        Elise_colour c = Elise_colour::its(i[k]/255,t[k]/255,s[k]/255);
        r[k] = 255 * c.r();
        g[k] = 255 * c.g();
        b[k] = 255 * c.b();
    }
}

Fonc_Num its_to_rgb(Fonc_Num f)
{
    return create_users_oper
           (
               0,
               its_to_rgb,
               f,
               3
           );
}




/***************************************************************/
/*                                                             */
/*             Pal_to_rgb_comp                                 */
/*                                                             */
/***************************************************************/

class Pal_to_rgb_comp : public Simple_OP_UN<INT>
{
    public :
         Pal_to_rgb_comp
         (
               Elise_Palette pal,
               Data_Elise_Palette  * dep
         )  :
            _pal (pal),
            _dep (dep)
         {
         }

    private :
        virtual void  calc_buf
                      (
                           INT ** output,
                           INT ** input,
                           INT        nb,
                           const Arg_Comp_Simple_OP_UN  &

                      );

        Elise_Palette         _pal;
        Data_Elise_Palette  * _dep;
};

void  Pal_to_rgb_comp::calc_buf 
      (
           INT ** output,
           INT ** input,
           INT        nb,
           const Arg_Comp_Simple_OP_UN  & arg
      )
{
    Tjs_El_User.ElAssert
    (
        arg.dim_in() == _dep->dim_pal(),
        EEM0 << "bas dim fonc in Elise_Palette::to_rgb"
    );

    _dep->to_rgb(output,input,nb);

}



Fonc_Num Elise_Palette::to_rgb(Fonc_Num f)
{
    return create_users_oper
           (
               new Pal_to_rgb_comp(*this,dep()),
               0,
               f,
               3
           );
}

/*
    Formule elementaire "pure"  (cad ni tronuqge ni shit dand [0 255])
*/

void  rgb_to_yuv
      (
           REAL8 & y,
           REAL8 & u,
           REAL8 & v,
           REAL8   r,
           REAL8   g,
           REAL8   b
      )
{
    y = 0.299   * r  +    0.587 * g  +   0.114 * b;
    u = -0.1687 * r  +  -0.3313 * g  +     0.5 * b;
    v =     0.5 * r  +  -0.4187 * g  +  0.0813 * b;
}

void mpeg_rgb_to_yuv
    (
           REAL8 & y,
           REAL8 & u,
           REAL8 & v,
           REAL8   r,
           REAL8   g,
           REAL8   b
    )
{
    rgb_to_yuv(y,u,v,r,g,b);
    y = (219.0/255.0)*y+16;
    u = (224.0/255.0)*u+128;
    v = (224.0/255.0)*v+128;
}


void  mpeg_rgb_to_yuv 
      (
          REAL ** out_put,
          REAL ** in_put,
          INT     nb,
          const Arg_Comp_Simple_OP_UN& arg
      )
{
    Tjs_El_User.ElAssert
    (
        arg.dim_in() == 3,
        EEM0 << "bas dim fonc in rgb_to_its"
    );

    REAL * r = in_put[0];
    REAL * g = in_put[1];
    REAL * b = in_put[2];

    REAL * y = out_put[0];
    REAL * u = out_put[1];
    REAL * v = out_put[2];

    for (INT k=0 ; k< nb ; k++)
        mpeg_rgb_to_yuv(y[k],u[k],v[k],r[k],g[k],b[k]);
}

Fonc_Num mpeg_rgb_to_yuv(Fonc_Num f)
{
    return create_users_oper
           (
               0,
               mpeg_rgb_to_yuv,
               f,
               3
           );
}





void  TrueCol16Bit_RGB::BufRGB2I
      (
          INT ** out_put,
          INT ** in_put,
          INT     nb,
          const Arg_Comp_Simple_OP_UN& arg
      )
{
    Tjs_El_User.ElAssert
    (
        arg.dim_in() == 3,
        EEM0 << "bas dim fonc in rgb_to_its"
    );

    INT * r = in_put[0];
    INT * g = in_put[1];
    INT * b = in_put[2];

    INT * ind = out_put[0];

    for (INT k=0 ; k< nb ; k++)
    {
        ind[k] = IndFromObj(RGB_Int(r[k],g[k],b[k]));
    }
}

Fonc_Num TrueCol16Bit_RGB::RGB2I(Fonc_Num f)
{
    return create_users_oper(BufRGB2I,0,f,1);
}


void  TrueCol16Bit_RGB::BufI2RGB
      (
          INT ** out_put,
          INT ** in_put,
          INT     nb,
          const Arg_Comp_Simple_OP_UN& arg
      )
{
    Tjs_El_User.ElAssert
    (
        arg.dim_in() == 1,
        EEM0 << "bas dim fonc in rgb_to_its"
    );

    INT * r = out_put[0];
    INT * g = out_put[1];
    INT * b = out_put[2];

    INT * ind = in_put[0];

    for (INT k=0 ; k< nb ; k++)
    {
        RGB_Int aRGB = ObjFromInd(ind[k]);
        r[k] = aRGB._r;
        g[k] = aRGB._g;
        b[k] = aRGB._b;
    }
}

Fonc_Num TrueCol16Bit_RGB::I2RGB(Fonc_Num f)
{
    return create_users_oper(BufI2RGB,0,f,3);
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
