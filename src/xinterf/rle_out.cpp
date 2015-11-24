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

/*****************************************************************/
/*                                                               */
/*                  Out_Ras_W_Comp                               */
/*                                                               */
/*****************************************************************/

Out_Ras_W_Comp::~Out_Ras_W_Comp() {}

Out_Ras_W_Comp::Out_Ras_W_Comp
(
           const Arg_Output_Comp &     arg ,
           const Data_El_Geom_GWin *   geom,
           Data_Elise_Raster_W *       derw,
           Data_Elise_Palette  *       dep
) :
        GW_computed(arg,geom,dep)           ,
       _derd      (derw->_derd)             ,
       _derw      (derw)                    ,
       _ddp       (derw->_ddsop->ddp_of_dep(dep))  ,
       _dep       (dep),
       _first     (true),
       _byte_pp   (_derd->_byte_pp)

{
}



/*****************************************************************/
/*                                                               */
/*                  RLE_Fen_X11_Computed                         */
/*                                                               */
/*****************************************************************/

/****************** La classe des des fenetres X11, vue de l'interieur ****/

class  RLE_Out_Ras_W_Comp : public  Out_Ras_W_Comp
{
   public :
      RLE_Out_Ras_W_Comp
      (
           const Arg_Output_Comp &  ,
           const Data_El_Geom_GWin *,
           Data_Elise_Raster_W     *,
           Data_Elise_Palette      *,
           bool  OnYDiff
      );


      virtual void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);
      void    flush_image(void);
      ~RLE_Out_Ras_W_Comp(void)
      {
           flush_image();
           DELETE_VECTOR(_line_lut,0);
           DELETE_VECTOR(_cpbli  ,0);
           DELETE_VECTOR(_zx  ,_pu1.x);
           DELETE_VECTOR(_wx0  ,_pu1.x);
      }

    private :

      INT       _u_last_y;
      INT       _w_last_x0;
      INT       _w_last_x1;
      U_INT1 *  _line_lut;
      U_INT1 *  _cpbli;
      Pt2di     _pu1,_pu2;

      U_INT2 *  _zx;
      U_INT2 *  _wx0;
      bool      mOnYDiff;
      bool      _LineInProgr;

};


RLE_Out_Ras_W_Comp::RLE_Out_Ras_W_Comp
(
      const Arg_Output_Comp &  arg   ,
      const Data_El_Geom_GWin * geom ,
      Data_Elise_Raster_W     * derw ,
      Data_Elise_Palette      * pal,
      bool                      OnYDiff
) :
      Out_Ras_W_Comp(arg,geom,derw,pal),
      mOnYDiff      (OnYDiff)
      // _u_last_y,_w_last_x0 and _w_last_x1 initialized in update when (! _first)
{
   

    geom->box_user_geom(_pu1,_pu2);
    _line_lut = _derd->alloc_line_buf(_pu2.x-_pu1.x+2);
    _cpbli    = _derd->alloc_line_buf(_sz.x);

    REAL tx = _tr.x;
    REAL sx = _sc.x;


    _zx  = NEW_VECTEUR(_pu1.x,_pu2.x,U_INT2);
    _wx0 = NEW_VECTEUR(_pu1.x,_pu2.x,U_INT2);

    for (INT x = _pu1.x; x<_pu2.x ; x++)
    {
        INT wx0,wx1;
        interval_user_to_window(wx0,wx1,x,x+1,tx,sx);

        wx0 = ElMax(0,wx0);
        wx1 = ElMin(geom->sz().x,wx1);

        _zx[x] = wx1 -wx0;
        _wx0[x] = wx0;
    }

}




void RLE_Out_Ras_W_Comp::flush_image(void)
{
   if (    (! _first)   //  !! : fisrt => _w_last_x .... = rubbish
        && (_w_last_x0<_w_last_x1) // do not want to know how are handled empty lines
      )
   {
       INT wy0,wy1;
       interval_user_to_window(wy0,wy1,_u_last_y,_u_last_y+1,_tr.y,_sc.y);
       wy0 = ElMax(0,wy0);
       wy1 = ElMin(_sz.y,wy1);
// std::cout << "WY " << wy0 << " " << wy1  << " " << _w_last_x0 << " " << _w_last_x1 << "\n";
       if (wy0 != wy1)
       {
            memcpy
            (
                _derw->_bli+_w_last_x0 * _byte_pp,
                _cpbli+_w_last_x0 * _byte_pp,
                (_w_last_x1-_w_last_x0) * _byte_pp
            );

            for (INT wy = wy0; wy < wy1 ; wy++)
                _derw->flush_bli(_w_last_x0,_w_last_x1,wy);
           _derd->disp_flush();
       }

   }
}


void RLE_Out_Ras_W_Comp::update(const Pack_Of_Pts * p,const Pack_Of_Pts * v)
{
    const Std_Pack_Of_Pts<INT> * ivals
         = SAFE_DYNC(const Std_Pack_Of_Pts<INT> *,v);
    RLE_Pack_Of_Pts * rle_pack = SAFE_DYNC(RLE_Pack_Of_Pts *,const_cast<Pack_Of_Pts *>(p));

    INT nb = rle_pack->nb();
    if (! nb) return;

    INT ** vals = ivals->_pts;

     // _dep->verif_values_out(vals,nb);



    INT ux0 = rle_pack->x0();
    INT ux1 = ux0 + nb;
    INT uy  = rle_pack->pt0()[1];

    if (mOnYDiff)
    {
       int wy0,wy1;
       interval_user_to_window(wy0,wy1,uy,uy+1,_tr.y,_sc.y);
       if (wy0==wy1) return;
/*
       if (_first || (wy0!= _w_last_y))
       {
          _LineInProgr = true;
       }
       else if   (ux0!=_u_last_x1)
       {
            _LineInProgr = false;
       }

       _w_last_y = wy0;
       _u_last_x1 = ux1;
      

       if (  ! _LineInProgr)
       {
            return;
       }
*/
    }


     _dep->verif_values_out(vals,nb);


    REAL tx = _tr.x;
    REAL sx = _sc.x;

    INT out_wx0,out_wx1;
     // these values may get out ``slightly'' (no more than 1  user pixel) of window
    interval_user_to_window(out_wx0,out_wx1,ux0,ux1,tx,sx);
    INT wx0 =  ElMax(0,out_wx0);
    INT wx1 = ElMin(_sz.x,out_wx1);


    // An easy way to treat the firt call is to consider that a previous empty segment
    // was preceding;
    if (_first)
    {
       _first = false;
       _w_last_x0 = _w_last_x1 = wx0;
       _u_last_y = uy-1;
    }

    // if the curent segment is not a prolongation of previous one
    if ((_u_last_y != uy) || (_w_last_x1 != wx0))
    {
        flush_image();
       _w_last_x0 = wx0;
       _u_last_y = uy;
    }
    _w_last_x1 = wx1;


    // for optmization purpose, treat separately
    // the cases :
    //    * sx = 1      (most frequent and fast)
    //    * sx integer  (not so rare and quite quick)
    //    * other case  (should be less frequent, and more complex to handle)



    if (sx == 1.0)
    {
        //  sx = 1 : copy with an eventual offset
        _dep->lutage (_derd,_cpbli,wx0,wx1,
                      _ddp->lut_compr(),vals,wx0 - out_wx0);
    }
    else
    {
       _dep->lutage (_derd,_line_lut,0,nb,
                      _ddp->lut_compr(),vals,0);

        U_INT1 * ddbli = _cpbli + _wx0[ux0] * _byte_pp ;
        U_INT2 * _tz = _zx+ux0;

        switch (_derd->_cmod)
        {
              case Indexed_Colour :
              {
                      for (int i=0 ; i<nb ; i++)
                      {
                          INT zx = *(_tz++);
                          INT c = _line_lut[i];
                          for (INT iz = 0; iz < zx; iz ++)
                              *(ddbli++) = c;
                      }
              }
              break;

              case True_16_Colour :
              {
                 U_INT2 * ll2= (U_INT2 *) _line_lut;
                 U_INT2 * dd2= (U_INT2 *) ddbli;
                 for (int i=0 ; i<nb ; i++)
                 {
                     INT c = ll2[i];
                     INT zx = *(_tz++);
                     for (INT iz = 0; iz < zx; iz ++)
                         *(dd2++) = c;
                 }
              }
              break;

              case True_24_Colour :
              {
                  INT r,g,b;
                  U_INT1 * ll1= _line_lut;
                  INT  r_ind = _derd->_r_ind;
                  INT  g_ind = _derd->_g_ind;
                  INT  b_ind = _derd->_b_ind;

                  for (int i=0 ; i<nb ; i++)
                  {
                     INT zx = *(_tz++);
                      r = ll1[r_ind];
                      g = ll1[g_ind];
                      b = ll1[b_ind];
                      for (INT iz = 0; iz < zx; iz ++)
                      {
                          ddbli[r_ind] = r;
                          ddbli[g_ind] = g;
                          ddbli[b_ind] = b;
                          ddbli += _byte_pp;
                      }
                      ll1 += _byte_pp;
                  }
               }
               break;
       }
    }
}


Output_Computed * Data_Elise_Raster_W::rle_out_comp
(
      const Data_El_Geom_GWin * geom,
      const Arg_Output_Comp & arg,
      Data_Elise_Palette *    p,
      bool                    OnYDiff
)
{
     return  new RLE_Out_Ras_W_Comp(arg,geom,this,p,OnYDiff);
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
