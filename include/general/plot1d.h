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



#ifndef _ELISE_GENERAL_PLOT1D_H
#define _ELISE_GENERAL_PLOT1D_H



class Plots
{
    public :

           typedef enum 
           {
               line,
               draw_box,
               fill_box,
               draw_fill_box,
               dirac
               //  dots,
           } mode_plot;

           typedef enum
           {
                cart,
                polar,
                user
           }  mode_coord;

           typedef enum
           {
                 no_par,
                 param
           } mode_param;
};


class Arg_Opt_Plot1d : public PRC0
{
      friend class Data_Param_Plot_1d;
      protected :
         Arg_Opt_Plot1d (class Data_Arg_Opt_Plot1d *);
      private :
          class Data_Arg_Opt_Plot1d * daop1d() const
                {return SAFE_DYNC(class Data_Arg_Opt_Plot1d *,_ptr);}
};

typedef ElList<Arg_Opt_Plot1d> L_Arg_Opt_Pl1d;

extern L_Arg_Opt_Pl1d       Empty_LAO_Pl1d;

L_Arg_Opt_Pl1d NewlArgPl1d(Arg_Opt_Plot1d arg);

class Plot_1d : public PRC0
{
      public :
          friend class Out_Plot1d_Not_Comp;

          Plot_1d(El_Window,
                  Line_St axes,
                  Line_St plot,
                  Interval,
                  L_Arg_Opt_Pl1d = Empty_LAO_Pl1d);

          void set(L_Arg_Opt_Pl1d );

          void show_axes(L_Arg_Opt_Pl1d = Empty_LAO_Pl1d);
          void clear(L_Arg_Opt_Pl1d = Empty_LAO_Pl1d);
          void show_box(L_Arg_Opt_Pl1d = Empty_LAO_Pl1d);

          void plot(Fonc_Num f,L_Arg_Opt_Pl1d = Empty_LAO_Pl1d);

          Output  out();
          Flux_Pts all_pts();

          El_Window permanent_window();
          El_Window last_window();

      private :

          Flux_Pts all_pts(L_Arg_Opt_Pl1d);
          Output  out(L_Arg_Opt_Pl1d);
          class Data_Plot_1d * dp1d () const
                {return SAFE_DYNC(Data_Plot_1d *,_ptr);}
};

        //======================
        // INT values args
        //======================

class PlModePl : public Arg_Opt_Plot1d  // origin of y
{
    public :
        PlModePl(Plots::mode_plot);
};

        //======================
        // real values  args
        //======================

class PlOriY : public Arg_Opt_Plot1d  // origin of y
{
      public :
        PlOriY (REAL);
};

class PlScaleY : public Arg_Opt_Plot1d  // origin of y
{
      public :
        PlScaleY (REAL);
};

class PlStepX : public Arg_Opt_Plot1d  // step in x
{
      public :
        PlStepX (REAL);
};

       //======================
       // line style args
       //======================

class PlBoxSty : public Arg_Opt_Plot1d  // boxe style
{
      public :
        PlBoxSty (Line_St);
        PlBoxSty (Col_Pal,INT);
};



class PlotLinSty : public Arg_Opt_Plot1d // plot style
{
      public :
        PlotLinSty (Line_St);
        PlotLinSty (Col_Pal,INT);

};

class PlAxeSty : public Arg_Opt_Plot1d  // axes styles
{
      public :
        PlAxeSty (Line_St);
        PlAxeSty (Col_Pal,INT);
};

       //======================
       // fill  style args
       //======================

class PlClearSty : public Arg_Opt_Plot1d  // boxe style
{
      public :
        PlClearSty (Fill_St);
        PlClearSty (Col_Pal);
};


class PlotFilSty : public Arg_Opt_Plot1d  // boxe style
{
      public :
        PlotFilSty (Fill_St);
        PlotFilSty (Col_Pal);
};



     //======================
     // boolean args 
     //======================

                 
class PlClipY : public Arg_Opt_Plot1d  // y clipping
{
      public :
        PlClipY (bool);
};

class PlAutoScalOriY : public Arg_Opt_Plot1d  // y clipping
{
      public :
        PlAutoScalOriY (bool);
};

class PlAutoScalY : public Arg_Opt_Plot1d  // y clipping
{
      public :
        PlAutoScalY (bool);
};

class PlShAxes : public Arg_Opt_Plot1d  // y clipping
{
      public :
        PlShAxes (bool);
};

class PlAutoClear : public Arg_Opt_Plot1d  // y clipping
{
      public :
        PlAutoClear (bool);
};


      //======================
      // intervalw arg
      //======================

class PlIntervBoxX : public Arg_Opt_Plot1d   // box
{
      public :
        PlIntervBoxX(REAL,REAL);
        PlIntervBoxX(Interval);
};


class PlIntervPlotX : public Arg_Opt_Plot1d   // box
{
      public :
        PlIntervPlotX(REAL,REAL);
        PlIntervPlotX(Interval);
};





      //======================
      // box arg
      //======================

class PlBox : public Arg_Opt_Plot1d   // box
{
      public :
        PlBox(Pt2dr,Pt2dr);
        PlBox(REAL x0,REAL y0,REAL x1,REAL y1);
};





#endif //  _ELISE_GENERAL_PLOT1D_H

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
