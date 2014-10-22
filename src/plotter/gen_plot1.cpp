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


/********************************************************************************/
/********************************************************************************/
/********************************************************************************/
/*                                                                              */
/*       Optional args to plotters                                              */
/*                                                                              */
/********************************************************************************/
/********************************************************************************/
/********************************************************************************/

L_Arg_Opt_Pl1d  Empty_LAO_Pl1d = L_Arg_Opt_Pl1d();


         //  generic : quite poor !

class Data_Arg_Opt_Plot1d : public RC_Object
{
   friend class Data_Param_Plot_1d;
   private :
      virtual void update_dpp1d(Data_Param_Plot_1d *) = 0;
};


Arg_Opt_Plot1d::Arg_Opt_Plot1d(Data_Arg_Opt_Plot1d * daop1d) :
    PRC0(daop1d)
{
}
         
         //===================
         //      Box Arg
         //===================

class Data_PlBox : public Data_Arg_Opt_Plot1d
{
     friend class PlBox;

     private :
        Data_PlBox(Box2dr box) : _box (box) {}

        virtual void update_dpp1d(Data_Param_Plot_1d * dpp1d)
        {
            dpp1d->_box = _box;
        }

        Box2dr _box;
};

PlBox::PlBox(Pt2dr p1,Pt2dr p2)    :
    Arg_Opt_Plot1d(new Data_PlBox(Box2dr(p1,p2)))
{
}

PlBox::PlBox(REAL x0,REAL y0,REAL x1,REAL y1)    :
    Arg_Opt_Plot1d(new Data_PlBox(Box2dr(Pt2dr(x0,y0),Pt2dr(x1,y1))))
{
}

         //===================
         //     int args 
         //===================

class DPl_Arg_int : public Data_Arg_Opt_Plot1d
{
     friend class PlModePl;

     typedef enum  mode
     {
          plot_mode
     } mode;

     private :
        DPl_Arg_int(INT VAL,mode MODE) : _val (VAL), _mode(MODE) {}

        virtual void update_dpp1d(Data_Param_Plot_1d * dpp1d)
        {
          
           switch (_mode)
           {
              case plot_mode : dpp1d->_mode_plots = (Plots::mode_plot) _val; break;

           }
        }

        INT _val;
        mode _mode;
};


PlModePl::PlModePl(Plots::mode_plot mode) :
   Arg_Opt_Plot1d(new DPl_Arg_int(mode,DPl_Arg_int::plot_mode))
{
}


         //===================
         //     real args 
         //===================

class DPl_Arg_real : public Data_Arg_Opt_Plot1d
{
     friend class PlOriY;
     friend class PlScaleY;
     friend class PlStepX;

     typedef enum  mode
     {
          ori_y,
          scale_y,
          step_x
     } mode;

     private :
        DPl_Arg_real(REAL VAL,mode MODE) : _val (VAL), _mode(MODE) {}

        virtual void update_dpp1d(Data_Param_Plot_1d * dpp1d)
        {
          
           switch (_mode)
           {
              case ori_y : dpp1d->_ori_y = _val; break;

              case scale_y : dpp1d->_y_scale = _val; break;

              case step_x : dpp1d->_x_step = _val; break;
           }
        }

        REAL _val;
        mode _mode;
};


PlOriY::PlOriY(REAL Oy) : 
   Arg_Opt_Plot1d(new DPl_Arg_real(Oy,DPl_Arg_real::ori_y))
{
}

PlScaleY::PlScaleY(REAL Sy) : 
   Arg_Opt_Plot1d(new DPl_Arg_real(Sy,DPl_Arg_real::scale_y))
{
}


PlStepX::PlStepX(REAL st) : 
   Arg_Opt_Plot1d(new DPl_Arg_real(st,DPl_Arg_real::step_x))
{
}
         //===================
         //     Fill Style 
         //===================

class Data_Plot_Fill_St : public Data_Arg_Opt_Plot1d
{
     friend class PlClearSty;
     friend class PlotFilSty;

     private :

        typedef enum mode {clear,fil_plot} mode;

        Data_Plot_Fill_St(Fill_St FST,mode MODE) : _fst (FST), _mode (MODE) {}

        virtual void update_dpp1d(Data_Param_Plot_1d * dpp1d)
        {
            switch (_mode)
            {
                 case clear:
                     dpp1d->_clear_style = _fst;
                 break;

                 case fil_plot:
                     dpp1d->_plot_fill_style = _fst;
                 break;
            }
        }

        Fill_St _fst;
        mode    _mode;
};


PlClearSty::PlClearSty(Fill_St FST) : 
   Arg_Opt_Plot1d(new Data_Plot_Fill_St(FST,Data_Plot_Fill_St::clear))
{
}

PlClearSty::PlClearSty(Col_Pal col) : 
   Arg_Opt_Plot1d(new Data_Plot_Fill_St(col,Data_Plot_Fill_St::clear))
{
}



PlotFilSty::PlotFilSty(Fill_St FST) : 
   Arg_Opt_Plot1d(new Data_Plot_Fill_St(FST,Data_Plot_Fill_St::fil_plot))
{
}

PlotFilSty::PlotFilSty(Col_Pal col) : 
   Arg_Opt_Plot1d(new Data_Plot_Fill_St(col,Data_Plot_Fill_St::fil_plot))
{
}


         //===================
         //     Line Style 
         //===================

class Data_Plot_Line_St : public Data_Arg_Opt_Plot1d
{
     friend class PlBoxSty;
     friend class PlotLinSty;
     friend class PlAxeSty;

     private :
        typedef enum mode {box,axe,plot} mode;

        Data_Plot_Line_St(Line_St LST,mode MODE) : _lst (LST), _mode (MODE) {}
        Data_Plot_Line_St(Col_Pal c,INT e,mode MODE) : _lst (Line_St(c,e)), _mode (MODE) {}

        virtual void update_dpp1d(Data_Param_Plot_1d * dpp1d)
        {
            switch (_mode)
            {
                 case box:
                     dpp1d->_box_styles = _lst;
                 break;
                 case axe:
                     dpp1d->_axes_styles = _lst;
                 break;
                 case plot:
                     dpp1d->_plot_style = _lst;
                 break;
            }
        }

        Line_St _lst;
        mode    _mode;
};


PlBoxSty::PlBoxSty(Line_St LST) : 
   Arg_Opt_Plot1d(new Data_Plot_Line_St(LST,Data_Plot_Line_St::box))
{
}

PlBoxSty::PlBoxSty(Col_Pal c,INT e)   : 
   Arg_Opt_Plot1d(new Data_Plot_Line_St(c,e,Data_Plot_Line_St::box))
{
}



PlAxeSty::PlAxeSty(Line_St LST) : 
   Arg_Opt_Plot1d(new Data_Plot_Line_St(LST,Data_Plot_Line_St::axe))
{
}

PlAxeSty::PlAxeSty(Col_Pal c,INT e)   : 
   Arg_Opt_Plot1d(new Data_Plot_Line_St(c,e,Data_Plot_Line_St::axe))
{
}




PlotLinSty::PlotLinSty(Line_St LST) : 
   Arg_Opt_Plot1d(new Data_Plot_Line_St(LST,Data_Plot_Line_St::plot))
{
}

PlotLinSty::PlotLinSty(Col_Pal c,INT e)   : 
   Arg_Opt_Plot1d(new Data_Plot_Line_St(c,e,Data_Plot_Line_St::plot))
{
}


         //===================
         //     Boolean args 
         //===================

class DPl_Arg_bool : public Data_Arg_Opt_Plot1d
{
     friend class PlClipY;
     friend class PlAutoScalOriY;
     friend class PlAutoScalY;
     friend class PlShAxes;
     friend class PlAutoClear;

     typedef enum  mode
     {
          clip_y,
          aut_scor_y,
          aut_sc_y,
          sh_ax,
          auto_clear
     } mode;

     private :
        DPl_Arg_bool(bool VAL,mode MODE) : _val (VAL), _mode(MODE) {}

        virtual void update_dpp1d(Data_Param_Plot_1d * dpp1d)
        {
           switch (_mode)
           {
              case clip_y : dpp1d->_clip_y               = _val; break;
              case aut_scor_y : dpp1d->_auto_ori_scale_y = _val; break;
              case aut_sc_y : dpp1d->_auto_scale_y       = _val; break;
              case sh_ax : dpp1d->_show_axes             = _val; break;
              case auto_clear : dpp1d->_auto_clear       = _val; break;
           }
        }

        bool _val;
        mode _mode;
};


PlClipY::PlClipY(bool  cly) : 
   Arg_Opt_Plot1d(new DPl_Arg_bool(cly,DPl_Arg_bool::clip_y))
{
}

PlAutoScalOriY::PlAutoScalOriY(bool  cly) : 
   Arg_Opt_Plot1d(new DPl_Arg_bool(cly,DPl_Arg_bool::aut_scor_y))
{
}

PlAutoScalY::PlAutoScalY(bool  cly) : 
   Arg_Opt_Plot1d(new DPl_Arg_bool(cly,DPl_Arg_bool::aut_sc_y))
{
}

PlShAxes::PlShAxes(bool  cly) : 
   Arg_Opt_Plot1d(new DPl_Arg_bool(cly,DPl_Arg_bool::sh_ax))
{
}

PlAutoClear::PlAutoClear(bool  cly) : 
   Arg_Opt_Plot1d(new DPl_Arg_bool(cly,DPl_Arg_bool::auto_clear))
{
}


         //===================
         //     Interval args 
         //===================

class DPl_Arg_Interv : public Data_Arg_Opt_Plot1d
{
     friend class PlIntervPlotX;
     friend class PlIntervBoxX;

     typedef enum  mode
     {
          xplot,
          xbox
     } mode;

     private :

        DPl_Arg_Interv(Interval VAL,mode MODE) : _val (VAL), _mode(MODE) {}

        virtual void update_dpp1d(Data_Param_Plot_1d * dpp1d)
        {

           switch (_mode)
           {
              case xplot : dpp1d->_int_plot_x       = _val; break;
              case xbox :  dpp1d->_int_box_x        = _val; 
                           dpp1d->_int_plot_x       = _val;
                            break;
           }
        }

        Interval     _val;
        mode         _mode;
};


PlIntervPlotX::PlIntervPlotX(Interval  intX) : 
   Arg_Opt_Plot1d(new DPl_Arg_Interv(intX,DPl_Arg_Interv::xplot))
{
}

PlIntervPlotX::PlIntervPlotX(REAL v1,REAL v2) : 
   Arg_Opt_Plot1d(new DPl_Arg_Interv(Interval(v1,v2),DPl_Arg_Interv::xplot))
{
}




PlIntervBoxX::PlIntervBoxX(Interval  intX) : 
   Arg_Opt_Plot1d(new DPl_Arg_Interv(intX,DPl_Arg_Interv::xbox))
{
}

PlIntervBoxX::PlIntervBoxX(REAL v1,REAL v2) : 
   Arg_Opt_Plot1d(new DPl_Arg_Interv(Interval(v1,v2),DPl_Arg_Interv::xbox))
{
}






/********************************************************************************/
/********************************************************************************/
/********************************************************************************/
/*                                                                              */
/*       Parameter of 1-D plotter                                               */
/*                                                                              */
/********************************************************************************/
/********************************************************************************/
/********************************************************************************/


          /*****************************************************************/
          /*                                                               */
          /*                  Data_Param_Plot_1d                           */
          /*                                                               */
          /*****************************************************************/



         //---------------------------------------------
         //---------------------------------------------
         //---------------------------------------------
         //---------------------------------------------
         //---------------------------------------------

void  Data_Param_Plot_1d::compute_args()
{


          /*--------------------------------------------------------------


             [1]  For computing tr.x, sc.x, we must solve the equation :

                   rto_window_geom(_int_box_x.v0, tr.x,sc.x) =  _box._p0.x
                   rto_window_geom(_int_box_x.v1, tr.x,sc.x) = _box._p1.x

             [2]   _sc.y is given by : _sc.x *  -1 *_y_scale;
                                           ===
             [3]   _tr.y is given by 
                      rto_window_geom(0, tr.y,sc.y)
                      = (1-_ori_y ) * _box.p1.y + _ori_y * box.p1.y
          ---------------------------------------------------------------------*/

     compute_tr_sc_from_exemple
     (
             _tr.x,_sc.x,
             _int_box_x._v0 -0.5,_int_box_x._v1 +0.5,
             _box._p0.x,_box._p1.x
     );

     _sc.y = - _sc.x * _y_scale;

      set_tr_y();
     _axes_lst  =    _axes_styles.dlst();
     _box_lst  =     _box_styles.dlst();
     _plot_lst  =    _plot_style.dlst();
     _clear_st  =    _clear_style.dfst();
     _plot_fstyle = _plot_fill_style.dfst();
}


void  Data_Param_Plot_1d::set_tr_y()
{
     compute_tr_from_exemple
     (
         _tr.y,_sc.y,0,
         (1-_ori_y)* _box._p1.y+_ori_y*_box._p0.y 
     );


     _y_interv = Interval
                 (
                      rto_user_geom(_box._p1.y,_tr.y,_sc.y),
                      rto_user_geom(_box._p0.y,_tr.y,_sc.y)
                 );

     _w         =    _w.chc(_tr,_sc);
     _wgeo      =    _w.degeow();
}

void  Data_Param_Plot_1d::use_arg_opts(L_Arg_Opt_Pl1d l)
{
      for(; !(l.empty()); l = l.cdr())
         l.car().daop1d()->update_dpp1d(this);
}


Pt2dr Data_Param_Plot_1d::to_win(Pt2dr p)
{
    return rto_window_geom(p,_tr,_sc);
}
         //---------------------------------------------
         //---------------------------------------------
         //---------------------------------------------
         //---------------------------------------------
         //---------------------------------------------

Data_Param_Plot_1d::Data_Param_Plot_1d
(
         El_Window      W ,
         Line_St        Axes_Style  ,
         Line_St        Splot_Style  ,
         Interval       X_Interv,
         L_Arg_Opt_Pl1d  l
) :

     _w                 (W),
     _axes_styles       (Axes_Style),
     _plot_style        (Splot_Style),
     _int_box_x         (X_Interv),
     _int_plot_x        (X_Interv),
     _mode_plots        (Plots::line),
     _box_styles        (Axes_Style),
     _clear_style       (Axes_Style.col()),
     _plot_fill_style   (Splot_Style.col()),
     _box               ( W.box().ToBoxR()),
     _x_step            (1.0),
     _y_scale           (1.0),
     _ori_y             (0.5),
     _show_axes         (false),
     _clip_y            (true),
     _auto_scale_y      (false),
     _auto_ori_scale_y  (false),
      _auto_clear       (false),
     _y_interv          (0.0,0.0)  // arbitrary to satisfy compiler => really set in compute_args
{
      // a fisrt computation to ensure that all args 
      // are initialized ones (else, if, for example,_auto_ori_scale_x
      // is true, tr.x and scale. are not initialized)
      compute_args();
      use_arg_opts(l);
      compute_args();
}


          /*****************************************************************/
          /*                                                               */
          /*                  Param_Plot_1d                                */
          /*                                                               */
          /*****************************************************************/

Param_Plot_1d::Param_Plot_1d
(
         El_Window          W ,
         Line_St            Axes_Style  ,
         Line_St            Splot_Style  ,
         Interval           X_Interv,
         L_Arg_Opt_Pl1d     laopl1d 
) :
  PRC0
  (
       new Data_Param_Plot_1d
           (W,Axes_Style,Splot_Style,X_Interv,laopl1d)
  )
{
}


/********************************************************************************/
/********************************************************************************/
/********************************************************************************/
/*                                                                              */
/*                    1-D plotter                                               */
/*                                                                              */
/********************************************************************************/
/********************************************************************************/
/********************************************************************************/



          /*****************************************************************/
          /*                                                               */
          /*                  Data_Plot_1d                                 */
          /*                                                               */
          /*****************************************************************/

void Data_Plot_1d::init_arg_tmp(L_Arg_Opt_Pl1d l)
{
      *(_tmp_par.dpp1d()) = *(_param.dpp1d());
      _tmp_par.dpp1d()->use_arg_opts(l);
      _tmp_par.dpp1d()->compute_args();
}

void Data_Plot_1d::set(L_Arg_Opt_Pl1d l)
{
     _param.dpp1d()->use_arg_opts(l);
}



Data_Plot_1d::Data_Plot_1d
(
     El_Window        W,
     Line_St          Ax_St,
     Line_St          Sp_St,
     Interval         Int_x,
     L_Arg_Opt_Pl1d   laopl1d 
) :
    _param   (W,Ax_St,Sp_St,Int_x,laopl1d),
    _tmp_par (W,Ax_St,Sp_St,Int_x,laopl1d)
{
}


          /*****************************************************************/
          /*                                                               */
          /*                  Plot_1d                                      */
          /*                                                               */
          /*****************************************************************/

Plot_1d::Plot_1d
(
         El_Window            W ,
         Line_St              Axes_Style  ,
         Line_St              Splot_Style  ,
         Interval             X_Interv,
         L_Arg_Opt_Pl1d       laopl1d 
) :
  PRC0
  (
       new
            Data_Plot_1d
            (W,Axes_Style,Splot_Style,X_Interv,laopl1d)
  )
{
}


void Plot_1d::set(L_Arg_Opt_Pl1d l)
{
      dp1d()->set(l);
}

El_Window  Plot_1d::permanent_window()
{
     return  dp1d()->_param.dpp1d()->_w;
}
  
El_Window  Plot_1d::last_window()
{
     return  dp1d()->_tmp_par.dpp1d()->_w;
}


L_Arg_Opt_Pl1d NewlArgPl1d(Arg_Opt_Plot1d arg)
{
    return Empty_LAO_Pl1d + arg;
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
