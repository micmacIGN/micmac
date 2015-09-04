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



#ifndef _ELISE_PRIVATE_PLOT1D_H
#define _ELISE_PRIVATE_PLOT1D_H


class Data_Param_Plot_1d : public RC_Object
{
     friend class Data_Plot_1d;
     friend class Param_Plot_1d;
     friend class Out_Plot1d_Comp;
     friend class Plot_1d;


         // OK, OK this class seems to have much to many friends to be honest,
         // but is there another way to manage the optional arguments of plotters

     friend class Data_Plot_Line_St;
     friend class DPl_Arg_real;
     friend class DPl_Arg_bool;
     friend class DPl_Arg_Interv;
     friend class Data_PlBox;
     friend class Data_Plot_Fill_St;
     friend class DPl_Arg_int;

     private :

        // "The" constructor

           Data_Param_Plot_1d(El_Window,Line_St axes,Line_St plot,Interval,L_Arg_Opt_Pl1d);



        // What makes it a garbage collected object


        // "utilitary methods"

                 // compute the calculated parameters out
                 // of fundamentals paramaters

           void  compute_args();
           void  set_tr_y();

           void  use_arg_opts(L_Arg_Opt_Pl1d);
           Pt2dr to_win(Pt2dr);

         // drawing utilities

             void clear(bool auto_flush = true);
             void show_axes(bool auto_flush = true);
             void show_box(bool auto_flush = true);
             void plot(Fonc_Num f);
             void plot(REAL * x ,REAL * y,INT nb);
             void show_habillage();

             Flux_Pts all_pts();


         // Fundamuntals Non optional  parameters 

         El_Window    _w;
         Line_St      _axes_styles;
         Line_St      _plot_style;

         Interval     _int_box_x;  

         // Fundamuntals optional  parameters 


         Interval     _int_plot_x;   // def = _int_box_x

         Plots::mode_plot  _mode_plots;  // def value line

         Line_St      _box_styles;         // def value _axes_styles
         Fill_St      _clear_style;
         Fill_St      _plot_fill_style;    // def value colour of _axes_styles

         Box2dr      _box;      // def values =  w.box()
         REAL        _x_step;   // def values =   1.0
         REAL        _y_scale;  // relative to x_scale, def value = 1.0
         REAL        _ori_y;    // relative to box def value = 0.5
                                // set 0.0 for a function > 0

         bool         _show_axes; // def value = false
         bool         _clip_y;   // def value = true 
         bool          _auto_scale_y;  // is true scale_y is ajusted 
                                       // to tangent ymax or ymin
                                       //  def value = false
         bool          _auto_ori_scale_y;  // is true scale_y  and ori_y
                                           // are ajusted to tangent 
                                           // both ymax or ymin def value = false
         bool          _auto_clear;        // def = false

         // calculated parameters 

               // so that 
         Pt2dr                        _tr;
         Pt2dr                        _sc;
         Interval                    _y_interv;  
         Data_El_Geom_GWin          *  _wgeo;
         Data_Line_St  *               _axes_lst;
         Data_Line_St  *               _box_lst;
         Data_Line_St  *               _plot_lst;
         Data_Fill_St  *               _clear_st;
         Data_Fill_St  *               _plot_fstyle;
};

class Param_Plot_1d : public PRC0
{
     friend  class Data_Plot_1d;
     friend  class Out_Plot1d_Comp;
     friend class Plot_1d;

     private :
         Param_Plot_1d
         (
             El_Window,
             Line_St axes,
             Line_St plot,
             Interval,
             L_Arg_Opt_Pl1d
         );

         inline class Data_Param_Plot_1d * dpp1d() const 
                {return SAFE_DYNC(class Data_Param_Plot_1d *,_ptr);}
};

class Data_Plot_1d : public RC_Object
{
    friend class Plot_1d;
    friend class Out_Plot1d_Comp;

    private :

    // What makes it a garbage collected object


    // ctor
           Data_Plot_1d(El_Window,Line_St axes,Line_St plot,Interval,L_Arg_Opt_Pl1d);


   //  utilitary
           void init_arg_tmp(L_Arg_Opt_Pl1d);
           void set(L_Arg_Opt_Pl1d);


    //  drawing
           void show_axes(L_Arg_Opt_Pl1d);
           void clear(L_Arg_Opt_Pl1d);
           void show_box(L_Arg_Opt_Pl1d);

           Flux_Pts all_pts(L_Arg_Opt_Pl1d);

           void plot(Fonc_Num f,L_Arg_Opt_Pl1d);

           


    // DATA

                     // used to store the permanent options
           Param_Plot_1d _param;       

                     // used for storing the option temporar to one time ploting
           Param_Plot_1d _tmp_par;  
       

};



#endif //  _ELISE_PRIVATE_PLOT1D_H

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
