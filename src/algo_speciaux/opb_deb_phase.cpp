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


class Integ_grad_Opb : public Simple_OPBuf1 <INT,INT>
{
    public :

      Integ_grad_Opb (INT v_min,Im1D_INT4  im) :
          _v_min (v_min),
          _v_max (im.tx()+v_min),
          _im    (im),
          _pond  (im.data()-v_min)
      {}
 
   private :

      void  calc_buf(INT ** output,INT *** input);
      void verif_values(INT * grad) const;

      INT                     _v_min;
      INT                     _v_max;
      Im1D_INT4               _im;
      INT *                   _pond;
};



void Integ_grad_Opb::verif_values(INT * grad) const
{
     if (El_User_Dyn.active())
     {
         for (INT x = x0(); x <x1(); x++)
         {
             INT v = grad[x];

             if ( (v<_v_min)||(v>=_v_max)||((-v)<_v_min)||((-v)>=_v_max))
                El_User_Dyn.ElAssert
                (
                    0,
                    EEM0  << "Out of interval in Integ_grad_Opb_Comp \n"
                       << "|  Got value " << v << " or " << -v
                       << ", expected values in [" << _v_min 
                       << "," << _v_max << "["
                );
         }
     }
}

void Integ_grad_Opb::calc_buf(INT ** output,INT *** input) 
{
     
     Tjs_El_User.ElAssert
     (
          dim_in() == 3,
          EEM0 << "Integ_grad_Opb requires dim out = 3 for func"
     );


     INT ** pot_in = input[0];
     INT ** gx_in  = input[1];
     INT ** gy_in  = input[2];

     INT * pot_in_0 = pot_in[0];
     INT * pot_in_p1 = pot_in[1];
     INT * pot_in_m1 = pot_in[-1];
     INT * gx_in_0   = gx_in[0];
     INT * gy_in_0   = gy_in[0];
     INT * gy_in_m1   = gy_in[-1];


     INT * pot_out = output[0];
     INT * gx_out  = output[1];
     INT * gy_out  = output[2];


     verif_values(gx_in_0);
     verif_values(gy_in_0);

     
     for (INT x =x0(); x < x1(); x++)
     {
         INT dxp1 =  - gx_in_0[x];
         INT dxm1 =    gx_in_0[x-1];
         INT dyp1 = -  gy_in_0[x];
         INT dym1 =    gy_in_m1[x];

         INT Pdxp1 = _pond[dxp1];
         INT Pdxm1 = _pond[dxm1];
         INT Pdyp1 = _pond[dyp1];
         INT Pdym1 = _pond[dym1];

         INT new_v =    (pot_in_0[x+1]+dxp1) * Pdxp1
                      + (pot_in_0[x-1]+dxm1) * Pdxm1
                      + (pot_in_p1[x] + dyp1) *Pdyp1  
                      + (pot_in_m1[x] +dym1)  *Pdym1;




          INT sum_pds = Pdxp1+Pdxm1+Pdyp1+Pdym1;
          pot_out[x] =  new_v/sum_pds;
          gx_out[x]  = gx_in_0[x];
          gy_out[x]  = gy_in_0[x];
     }

}

Fonc_Num  integr_grad (Fonc_Num f,INT v_min,Im1D_INT4 p)
{

    return create_op_buf_simple_tpl
           (
                 new Integ_grad_Opb(v_min,p),
                 0,
                 f,
                 3,
                 Box2di(1)
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
