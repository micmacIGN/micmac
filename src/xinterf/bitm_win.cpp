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



/****************************************************************/
/*                                                              */
/*     Data_El_Bitmaped_Display                                 */
/*                                                              */
/****************************************************************/

class Data_El_Bitmaped_Display : public Data_Elise_Raster_D
{

      public :
          Data_El_Bitmaped_Display(const char*,INT nb);
          virtual ~Data_El_Bitmaped_Display();
          virtual void augmente_pixel_index(INT nb);
          virtual void disp_flush();
          virtual void set_cur_coul(INT coul);
          virtual void _inst_set_line_witdh(REAL) ;


};



void Data_El_Bitmaped_Display::set_cur_coul(INT coul)
{
    _cur_coul = coul;
}



Data_El_Bitmaped_Display::Data_El_Bitmaped_Display
(
     const char* NAME,
     INT nb
) :
      Data_Elise_Raster_D(NAME,nb)
{
      init_mode(8);
}

Data_El_Bitmaped_Display::~Data_El_Bitmaped_Display()
{
}

void Data_El_Bitmaped_Display::disp_flush() {}

void Data_El_Bitmaped_Display::augmente_pixel_index(INT nb)
{
    ASSERT_TJS_USER
    (
          (nb>=1) &&(nb <= _nb_pix_ind_max) ,
          " Bitmaped window cannot manage over 256 color"
    );

    for (int i=_nb_pix_ind; i<nb; i++)
        _pix_ind[i] = i;
   _nb_pix_ind = nb;
}


void Data_El_Bitmaped_Display::_inst_set_line_witdh(REAL){}

/****************************************************************/
/*                                                              */
/*     El_Bitmaped_Display                                      */
/*                                                              */
/****************************************************************/

class  El_Bitmaped_Display : public PRC0
{
      public :
          El_Bitmaped_Display(const char *,INT nb);
          Data_El_Bitmaped_Display * debd();
};


El_Bitmaped_Display::El_Bitmaped_Display(const char * NAME,INT nb) :
    PRC0(new Data_El_Bitmaped_Display(NAME,nb))
{
}


Data_El_Bitmaped_Display * El_Bitmaped_Display::debd()
{
    return SAFE_DYNC(Data_El_Bitmaped_Display *,_ptr);
}

/****************************************************************/
/*                                                              */
/*     Data_Elise_Bitmaped_Win                                  */
/*                                                              */
/****************************************************************/

class Data_Elise_Bitmaped_Win : public Data_Elise_Raster_W
{


      public :

          virtual ~Data_Elise_Bitmaped_Win();

          Data_Elise_Bitmaped_Win
          (
              El_Bitmaped_Display,
              Pt2di              ,
              Im2D_U_INT1      im,
              Elise_Set_Of_Palette
          );


          El_Bitmaped_Display         _ebd;
          Data_El_Bitmaped_Display * _debd;
          Im2D_U_INT1                  _im;

          U_INT1 **    _i;
          INT          _tx;
          INT          _ty;


      // DRAWING :

                // internal utilities

          inline void set_pixel(INT x,INT y,INT c)
          {
               if ((x>=0) && (x<_tx) && (y>=0) && (y<_ty))
                  _i[y][x] = c;
          }
          void draw_rect(INT x,INT y,INT w,INT h,INT c);

                //    raster functions

          virtual void  flush_bli(INT x0,INT x1,INT y);
          virtual void  rast_draw_pixels(struct El_RW_Point *,INT );
          virtual void  rast_draw_big_pixels(El_RW_Rectangle *,INT nb);

          void make_file_im(const char * name,bool Tif);


               // vector functions

          virtual void  _inst_draw_seg(Pt2dr,Pt2dr);
          virtual void  _inst_draw_circle(Pt2dr,Pt2dr);

};




Data_Elise_Bitmaped_Win::Data_Elise_Bitmaped_Win
(
        El_Bitmaped_Display  ebd,
        Pt2di                sz,
        Im2D_U_INT1          im,
        Elise_Set_Of_Palette sop
)    :
      Data_Elise_Raster_W(ebd.debd(),sz,sop)  ,
      _ebd  (ebd)                             ,
      _debd (ebd.debd()),
      _im   (im),
      _i    (_im.data()),
      _tx   (_im.tx()),
      _ty   (_im.ty())

{
   _bli = NEW_VECTEUR(0,sz.x,U_INT1);
}



Data_Elise_Bitmaped_Win::~Data_Elise_Bitmaped_Win()
{
     DELETE_VECTOR(_bli,0);
}


void  Data_Elise_Bitmaped_Win::flush_bli(INT x0,INT x1,INT y)
{
     memcpy(_im.data()[y]+x0,_bli+x0,x1-x0);
}

void  Data_Elise_Bitmaped_Win::rast_draw_pixels(struct El_RW_Point * pts,INT nb )
{
      INT c = _debd->cur_coul();
      for (int i=0; i<nb ; i++)
          set_pixel(pts[i].x,pts[i].y,c);
}

void Data_Elise_Bitmaped_Win::draw_rect(INT x,INT y,INT w,INT h,INT c)
{
   INT x0 = ElMax(x,0);
   INT y0 = ElMax(y,0);
   INT x1 = ElMin(x+w,_tx);
   INT y1 = ElMin(y+h,_ty);

   for (y=y0 ; y<y1; y++)
       for (x=x0 ; x<x1; x++)
           _i[y][x] = c;
}

void  Data_Elise_Bitmaped_Win::rast_draw_big_pixels(El_RW_Rectangle * r,INT nb)
{
      INT c = _debd->cur_coul();
      for (int i=0; i<nb ; i++)
          draw_rect
          (
               r[i].x,
               r[i].y,
               r[i].width,
               r[i].height,
               c
          );
}

void Data_Elise_Bitmaped_Win::_inst_draw_seg(Pt2dr p1,Pt2dr p2)
{
     bitm_marq_line
     (
           _i,
           _tx,
           _ty,
           round_ni(p1),
           round_ni(p2),
           _debd->cur_coul(),
           _debd->line_witdh()
     );
}

void Data_Elise_Bitmaped_Win::_inst_draw_circle(Pt2dr centre,Pt2dr ray)
{
    ELISE_COPY
    (
         ellipse(centre,ray.x,ray.y,0.0,true),
         _debd->cur_coul(),
         _im.oclip()
    );
}



void  Data_Elise_Bitmaped_Win::make_file_im(const char * name,bool Tif)
{

      Elise_colour tabc[256];
      _ddsop->get_tab_col(tabc,256);

      if (Tif)
      {
          Disc_Pal pal(tabc,256);
          Tiff_Im ImOut
                  (
                        name,
                        Pt2di(_tx,_ty),
                        GenIm::u_int1,
                        Tiff_Im::LZW_Compr,
                        pal
                 );

          ELISE_COPY(_im.all_pts(),_im.in(),ImOut.out());

      }
      else  // Gif
      {
            ELISE_COPY
            (
                _im.all_pts(),
                _im.in(),
                Gif_Im::create(name,Pt2di(_tx,_ty),tabc,8)
            );
      }
}

/****************************************************************/
/*                                                              */
/*     Bitm_Win                                                 */
/*                                                              */
/****************************************************************/


void Bitm_Win::make_tif(const char * name)
{
     debw()->make_file_im(name,true);
}

void Bitm_Win::make_gif(const char * name)
{
     debw()->make_file_im(name,false);
}






Bitm_Win::Bitm_Win
(
     const char *            NAME,
     Elise_Set_Of_Palette    sop,
     Pt2di                   sz
) :
       El_Window
       (
            new Data_Elise_Bitmaped_Win
                (El_Bitmaped_Display(NAME,256),sz,Im2D_U_INT1(sz.x,sz.y),sop),
            Pt2dr(0.0,0.0),
            Pt2dr(1.0,1.0)
       )
{
}


Bitm_Win::Bitm_Win
(
     const char *            NAME,
     Elise_Set_Of_Palette    sop,
     Im2D_U_INT1             IM
) :
       El_Window
       (
            new Data_Elise_Bitmaped_Win
                (El_Bitmaped_Display(NAME,256),Pt2di(IM.tx(),IM.ty()),IM,sop),
            Pt2dr(0.0,0.0),
            Pt2dr(1.0,1.0)
       )
{
}




Bitm_Win::Bitm_Win(Bitm_Win w,Pt2dr tr,Pt2dr sc) :
      El_Window(w.debw(),tr,sc)
{
}

Bitm_Win Bitm_Win::chc(Pt2dr tr,Pt2dr sc)
{
    return Bitm_Win(*this,tr,sc);
}


         //-------------------------------------------

Im2D_U_INT1 Bitm_Win::im() const
{
     return SAFE_DYNC(Data_Elise_Bitmaped_Win *,degraw())->_im;
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
