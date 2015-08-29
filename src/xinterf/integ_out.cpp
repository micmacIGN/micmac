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
/*                  PInt_Cste_Out_Ras_W_Comp                     */
/*                                                               */
/*****************************************************************/

class  PInt_Cste_Out_Ras_W_Comp : public  Out_Ras_W_Comp
{
   public :

      virtual void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);

      PInt_Cste_Out_Ras_W_Comp
      (
             const Arg_Output_Comp & ,
             const Data_El_Geom_GWin *,
             Data_Elise_Raster_W     *,
             Data_Elise_Palette      *,
             INT * coul
      );
      virtual ~PInt_Cste_Out_Ras_W_Comp()
      {
          _derd->disp_flush();
          _pts.destr();
          _rects.destr();
      }

    protected :
      INT    _coul;

    private :

      Elise_Pile<El_RW_Point>      _pts;
      Elise_Pile<El_RW_Rectangle>  _rects;
};



PInt_Cste_Out_Ras_W_Comp::PInt_Cste_Out_Ras_W_Comp
(
             const Arg_Output_Comp &       arg,
             const Data_El_Geom_GWin *     geom,
             Data_Elise_Raster_W     *     derw,
             Data_Elise_Palette      *     DEP,
             INT *  coul
) :
            Out_Ras_W_Comp(arg,geom,derw,DEP)
{
     if  ((_sc.x==1.0) && (_sc.y == 1.0))
         _pts = Elise_Pile<El_RW_Point>  (arg.flux()->sz_buf());
     else
         _rects = Elise_Pile<El_RW_Rectangle>(arg.flux()->sz_buf());

     if (coul)
     {
         _dep->verif_value_out(coul);
         _coul = _dep->ilutage(_derd,_ddp->lut_compr(),coul);
     }
}


void PInt_Cste_Out_Ras_W_Comp::update
(
         const Pack_Of_Pts * p,
         const Pack_Of_Pts *
)
{
    const Std_Pack_Of_Pts<INT> * pts = SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>(p));
    INT nb = pts->nb();
    INT * x = pts->_pts[0];
    INT * y = pts->_pts[1];

    REAL sx = _sc.x;
    REAL sy = _sc.y;

    REAL tx = _tr.x;
    REAL ty = _tr.y;


    _derd->set_cur_coul(_coul);

    if ((sx == 1.0) && (sy == 1.0))
    {
         Pt2di tr ( deb_interval_user_to_window(0,tx,sx),
                    deb_interval_user_to_window(0,ty,sy)
                  );


         _pts.reset(0);
         for (int i=0; i<nb ; i++)
             _pts.qpush(El_RW_Point(x[i]+tr.x,y[i]+tr.y));

         _derw->rast_draw_pixels(_pts.ptr(),_pts.nb());
    }
    else if  (  ((INT) sx == sx)  &&  ((INT) sy == sy))
    {
         INT zx = (int) sx;
         INT zy = (int) sy;
         INT wx0 = deb_interval_user_to_window(0,tx,sx);
         INT wy0 = deb_interval_user_to_window(0,ty,sy);

         _rects.reset(0);
         for (int i=0; i<nb ; i++)
             _rects.qpush(El_RW_Rectangle(zx*x[i]+wx0,zy*y[i]+wy0,zx,zy));

         _derw->rast_draw_big_pixels(_rects.ptr(),_rects.nb());
    }
    else
    {
         INT wx0,wx1;
         INT wy0,wy1;

         _rects.reset(0);
         for (int i=0; i<nb ; i++)
         {
            interval_user_to_window(wx0,wx1,x[i],x[i]+1,tx,sx);
            interval_user_to_window(wy0,wy1,y[i],y[i]+1,ty,sy);
            _rects.qpush(El_RW_Rectangle(wx0,wy0,wx1-wx0,wy1-wy0));
         }

         _derw->rast_draw_big_pixels(_rects.ptr(),_rects.nb());
    }
}


Output_Computed * Data_Elise_Raster_W::pint_cste_out_comp
(
      const Data_El_Geom_GWin * geom,
      const Arg_Output_Comp & arg   ,
      Data_Elise_Palette    * DEP,
      INT        * cste
)
{
     return  new PInt_Cste_Out_Ras_W_Comp(arg,geom,this,DEP,cste);
}


/*****************************************************************/
/*                                                               */
/*                  PInt_No_Cste_Out_Ras_W_Comp                  */
/*                                                               */
/*****************************************************************/

const INT SZ_REBUF_X11 =  10000;

class  PInt_No_Cste_Out_Ras_W_Comp : public PInt_Cste_Out_Ras_W_Comp
{
   public :
      PInt_No_Cste_Out_Ras_W_Comp
      (
             const Arg_Output_Comp & ,
             const Data_El_Geom_GWin *,
             Data_Elise_Raster_W     *,
             Data_Elise_Palette      *
      );

      virtual void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);
      virtual    ~PInt_No_Cste_Out_Ras_W_Comp(void);

  private :

     void vider_buf();

     INT _nb_pts_buf; // number of points currently bufferized
             // buffer for x,y and coul
     INT2    * _buf_x;
     INT2    * _buf_y;
     U_INT1  * _buf_c;
     INT        _sz_buf;
     Std_Pack_Of_Pts<INT> * _pts_sorted;

          // more or less a curser on _pts_sorted
          // do not use the Curser_on_PoP class because the need of have exact split
          // on colour make it useless
     Std_Pack_Of_Pts<INT> * _map_psort;

     INT        _cpt_coul[257];
};




PInt_No_Cste_Out_Ras_W_Comp::PInt_No_Cste_Out_Ras_W_Comp
(
             const Arg_Output_Comp & arg ,
             const Data_El_Geom_GWin * geom,
             Data_Elise_Raster_W     * derw,
             Data_Elise_Palette      * DEP
)  :
          PInt_Cste_Out_Ras_W_Comp(arg,geom,derw,DEP,0),
         _nb_pts_buf (0)   ,
         _buf_x      (STD_NEW_TAB(SZ_REBUF_X11,INT2)),
         _buf_y      (STD_NEW_TAB(SZ_REBUF_X11,INT2)),
         _buf_c      (STD_NEW_TAB(SZ_REBUF_X11,U_INT1)),
         _sz_buf     (arg.flux()->sz_buf()),
         _pts_sorted (Std_Pack_Of_Pts<INT>::new_pck(2,SZ_REBUF_X11)),
         _map_psort  (Std_Pack_Of_Pts<INT>::new_pck(2,0))
{
}


PInt_No_Cste_Out_Ras_W_Comp::~PInt_No_Cste_Out_Ras_W_Comp()
{
    vider_buf();
    delete _map_psort;
    delete _pts_sorted;
    STD_DELETE_TAB(_buf_c);
    STD_DELETE_TAB(_buf_y);
    STD_DELETE_TAB(_buf_x);
}

void PInt_No_Cste_Out_Ras_W_Comp::vider_buf()
{
    memset(_cpt_coul,0,257*sizeof(int));
    INT i; // Fuuuuuck to Visual
    for (i = 0; i <_nb_pts_buf ; i++)
        _cpt_coul[_buf_c[i]] ++;

    for (i = 1; i < 257 ; i++)
         _cpt_coul[i] += _cpt_coul[i-1];

    INT adr;
    INT * x = _pts_sorted->_pts[0];
    INT * y = _pts_sorted->_pts[1];

    for (i = 0; i <_nb_pts_buf ; i++)
    {
         adr = --(_cpt_coul[_buf_c[i]]);
         x[adr] = _buf_x[i];
         y[adr] = _buf_y[i];
    }


    for (i = 0; i < 256 ; i++)
    {
        INT a0 = _cpt_coul[i];
        INT a1 = _cpt_coul[i+1];
        if (a0 != a1)
           for (INT a = a0; a < a1 ; a += _sz_buf)
           {
               if (_derd->_cmod == Indexed_Colour)
                  _coul = i;
               else
                  _coul = _dep->ilutage(_derd,_ddp->lut_compr(),&i);
               _map_psort->set_nb(ElMin(_sz_buf,a1-a));
               _map_psort->_pts[0] = x+a;
               _map_psort->_pts[1] = y+a;
               PInt_Cste_Out_Ras_W_Comp::update(_map_psort,0);
           }
    }
    _nb_pts_buf = 0;

    _derd->disp_flush();
}


void PInt_No_Cste_Out_Ras_W_Comp::update
       (const Pack_Of_Pts * p,const Pack_Of_Pts * v)
{
    const Std_Pack_Of_Pts<INT> * pts =  SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>(p));
    const Std_Pack_Of_Pts<INT> * vals = SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>(v));

    const INT * x = pts->_pts[0];
    const INT * y = pts->_pts[1];
    INT nb =  pts->nb();
    INT ** c = vals->_pts;


   _dep->verif_values_out(c,nb);


    INT nbbp_last_vide = _nb_pts_buf;
    INT i_last_vide = 0;

    for (int i=0; i<nb ; i++)
    {
         _buf_x[_nb_pts_buf]   = x[i];
         _buf_y[_nb_pts_buf++]   = y[i];
         if (_nb_pts_buf == SZ_REBUF_X11)
         {
            _dep->clutage
            (
               _derd,
               _buf_c,
               nbbp_last_vide,
               _nb_pts_buf,
               _ddp->lut_compr(),
               c,
               i_last_vide
            );
            vider_buf();
            i_last_vide = i+1;
            nbbp_last_vide = _nb_pts_buf;
         }
    }
    _dep->clutage
    (
         _derd,
         _buf_c,
         nbbp_last_vide,
         _nb_pts_buf,
         _ddp->lut_compr(),
         c,
         i_last_vide
    );
}

/*****************************************************************/
/*                                                               */
/*                  PInt_NoC_TrueC_ORW_Comp                      */
/*                                                               */
/*****************************************************************/

class  PInt_NoC_TrueC_ORW_Comp : public  Out_Ras_W_Comp
{
   public :

      virtual void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);

      PInt_NoC_TrueC_ORW_Comp
      (
             const Arg_Output_Comp & ,
             const Data_El_Geom_GWin *,
             Data_Elise_Raster_W     *,
             Data_Elise_Palette      *
      );
      virtual     ~PInt_NoC_TrueC_ORW_Comp(void)
      {
          _derd->disp_flush();
          _pts.destr();
          _rects.destr();
          DELETE_VECTOR(_coul,0);
      }

    protected :

    private :

      INT    * _coul;
      Elise_Pile<El_RW_Point>      _pts;
      Elise_Pile<El_RW_Rectangle>  _rects;
};



PInt_NoC_TrueC_ORW_Comp::PInt_NoC_TrueC_ORW_Comp
(
             const Arg_Output_Comp &       arg,
             const Data_El_Geom_GWin *     geom,
             Data_Elise_Raster_W     *     derw,
             Data_Elise_Palette      *     DEP
) :
            Out_Ras_W_Comp(arg,geom,derw,DEP)
{
     if  ((_sc.x==1.0) && (_sc.y == 1.0))
         _pts = Elise_Pile<El_RW_Point> (arg.flux()->sz_buf());
     else
         _rects = Elise_Pile<El_RW_Rectangle>(arg.flux()->sz_buf());

     _coul = NEW_VECTEUR(0,arg.flux()->sz_buf(),INT);
}


void PInt_NoC_TrueC_ORW_Comp::update
(
         const Pack_Of_Pts * p,
         const Pack_Of_Pts * v
)
{
    const Std_Pack_Of_Pts<INT> * pts  = SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>(p));
    const Std_Pack_Of_Pts<INT> * vals = SAFE_DYNC(Std_Pack_Of_Pts<INT> *,const_cast<Pack_Of_Pts *>(v));
    INT nb = pts->nb();
    INT * x = pts->_pts[0];
    INT * y = pts->_pts[1];

    REAL sx = _sc.x;
    REAL sy = _sc.y;

    REAL tx = _tr.x;
    REAL ty = _tr.y;

    _dep->ilutage(_derd,_coul,vals->nb(),_ddp->lut_compr(),vals->_pts);

    if ((sx == 1.0) && (sy == 1.0))
    {
         Pt2di tr ( deb_interval_user_to_window(0,tx,sx),
                    deb_interval_user_to_window(0,ty,sy)
                  );

         _pts.reset(0);
         for (int i=0; i<nb ; i++)
             _pts.qpush(El_RW_Point(x[i]+tr.x,y[i]+tr.y));

         _derw->rast_draw_col_pix(_coul,_pts.ptr(),pts->nb());
    }
    else if  (  ((INT) sx == sx)  &&  ((INT) sy == sy))
    {
         INT zx = (int) sx;
         INT zy = (int) sy;
         INT wx0 = deb_interval_user_to_window(0,tx,sx);
         INT wy0 = deb_interval_user_to_window(0,ty,sy);

         _rects.reset(0);
         for (int i=0; i<nb ; i++)
             _rects.qpush(El_RW_Rectangle(zx*x[i]+wx0,zy*y[i]+wy0,zx,zy));

         _derw->rast_draw_col_big_pixels(_coul,_rects.ptr(),_rects.nb());
    }
    else
    {
         INT wx0,wx1;
         INT wy0,wy1;

         _rects.reset(0);
         for (int i=0; i<nb ; i++)
         {
            interval_user_to_window(wx0,wx1,x[i],x[i]+1,tx,sx);
            interval_user_to_window(wy0,wy1,y[i],y[i]+1,ty,sy);
            _rects.qpush(El_RW_Rectangle(wx0,wy0,wx1-wx0,wy1-wy0));
         }

         _derw->rast_draw_col_big_pixels(_coul,_rects.ptr(),_rects.nb());
    }
}

/*****************************************************************/
/*                                                               */
/*                                                               */
/*****************************************************************/


Output_Computed * Data_Elise_Raster_W::pint_no_cste_out_comp
(
      const Data_El_Geom_GWin * geom,
      const Arg_Output_Comp & arg   ,
      Data_Elise_Palette      * DEP
)
{
    if ((_derd->_cmod==Indexed_Colour)||(DEP->dim_pal()==1))
	   return new PInt_No_Cste_Out_Ras_W_Comp(arg,geom,this,DEP);
    else
	   return new PInt_NoC_TrueC_ORW_Comp(arg,geom,this,DEP);
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
