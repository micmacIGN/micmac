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
          /*                      clear()                                  */
          /*                                                               */
          /*****************************************************************/

void  Data_Param_Plot_1d::clear(bool auto_flush)
{

       _wgeo->fill_rect 
       (
           Pt2dr(_int_box_x._v0,_y_interv._v0),
           Pt2dr(_int_box_x._v1,_y_interv._v1),
           _clear_st,
           auto_flush
       );
}


void Data_Plot_1d::clear(L_Arg_Opt_Pl1d l)
{
      init_arg_tmp(l);
      _tmp_par.dpp1d()->clear();
}

void Plot_1d::clear(L_Arg_Opt_Pl1d l)
{
     dp1d()->clear(l);
}




          /*****************************************************************/
          /*                                                               */
          /*                  show_axes()                                  */
          /*                                                               */
          /*****************************************************************/


void Data_Param_Plot_1d::show_axes(bool auto_flush)
{

       if ( (_ori_y >= 0.0) && (_ori_y <= 1.0))
          _wgeo->draw_seg
          (
              Pt2dr(_int_box_x._v0,0.0),
              Pt2dr(_int_box_x._v1,0.0),
              _axes_lst,
              false
          );

  
       _wgeo->draw_seg
       (
           Pt2dr(0.0,_y_interv._v0),
           Pt2dr(0.0,_y_interv._v1),
           _axes_lst,
           auto_flush
       );
}

void Data_Plot_1d::show_axes(L_Arg_Opt_Pl1d l)
{
      init_arg_tmp(l);
      _tmp_par.dpp1d()->show_axes();
}

void Plot_1d::show_axes(L_Arg_Opt_Pl1d l)
{
     dp1d()->show_axes(l);
}



          /*****************************************************************/
          /*                                                               */
          /*                  show_box()                                   */
          /*                                                               */
          /*****************************************************************/


void Data_Param_Plot_1d::show_box(bool auto_flush)
{
       _wgeo->draw_rect
       (
           Pt2dr(_int_box_x._v0,_y_interv._v0),
           Pt2dr(_int_box_x._v1,_y_interv._v1),
           _box_lst
       );
       if (auto_flush)
          _wgeo->degd()->auto_flush();
}

void Data_Plot_1d::show_box(L_Arg_Opt_Pl1d l)
{
      init_arg_tmp(l);
      _tmp_par.dpp1d()->show_box();
}

void Plot_1d::show_box(L_Arg_Opt_Pl1d l  )
{
     dp1d()->show_box(l);
}

          /*****************************************************************/
          /*                                                               */
          /*                  all_pts()                                    */
          /*                                                               */
          /*****************************************************************/


Flux_Pts Data_Param_Plot_1d::all_pts()
{
    REAL  step    = _x_step;
    INT st_int    =  round_ni(step);
    INT ix0       =  round_up(_int_plot_x._v0/step);
    INT ix1       =  round_up(_int_plot_x._v1/step);



    if (step == 1.0)
       return  rectangle(ix0,ix1);

    if (step == st_int)
       return rectangle(ix0,ix1).chc(FX*st_int);

    return rectangle(ix0,ix1).chc(FX*step);
}

Flux_Pts Data_Plot_1d::all_pts(L_Arg_Opt_Pl1d l)
{
      init_arg_tmp(l);
      return _tmp_par.dpp1d()->all_pts();
}

Flux_Pts Plot_1d::all_pts(L_Arg_Opt_Pl1d l)
{
     return dp1d()->all_pts(l);
}

Flux_Pts Plot_1d::all_pts()
{
     return all_pts(Empty_LAO_Pl1d);
}


/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/*                                                                          */
/*       PLOTTING                                                           */
/*                                                                          */
/*                                                                          */
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

void Data_Param_Plot_1d::show_habillage()
{
     if (_show_axes)
        show_axes(false);
}

       /*****************************************************************/
       /*                                                               */
       /*                  plot    ()                                   */
       /*                                                               */
       /*****************************************************************/

void Data_Param_Plot_1d::plot(Fonc_Num)
{
}

void Data_Param_Plot_1d::plot(REAL * x, REAL *y,INT nb_pts)
{

    REAL max_y = OpMax.red_tab(y,nb_pts,y[0]);
    REAL min_y = OpMin.red_tab(y,nb_pts,y[0]);


    if (_auto_ori_scale_y)
    {
       if (max_y != min_y)
       {

            REAL l_interv = _y_interv._v1 -_y_interv._v0;
            REAL scy = l_interv / (max_y - min_y);

           _sc.y *= scy;
           _ori_y = - min_y / (max_y-min_y);
           _y_scale = -(_sc.y /_sc.x);
           set_tr_y();
       }
       
    }
    else if (_auto_scale_y)
    {
       ASSERT_TJS_USER
       (
          (_ori_y >0) && (_ori_y < 1),
          "origin of y must be in ]0 1[ in plotter auto scale y mode"
       );
       REAL scy = DBL_MAX;

       REAL l_interv = _y_interv._v1 -_y_interv._v0;

       if (max_y > 0)
       {
          scy = (l_interv * (1-_ori_y)) / max_y;
       }
       if (min_y < 0)
          scy = ElMin(scy,(l_interv * _ori_y) /(- min_y));


       if (scy != DBL_MAX)
       {
           _sc.y *= scy;
           _y_scale = -(_sc.y /_sc.x);
           set_tr_y();
       }
    }


    bool clip_y =     (_clip_y)
                  &&  (     (max_y > _y_interv._v1) 
                        ||  (min_y < _y_interv._v0)
                      );

    Box2dr box (   Pt2dr(_int_box_x._v0,_y_interv._v0),
                  Pt2dr(_int_box_x._v1,_y_interv._v1)
              );

    //===============================
    //  Plot LINE
    //===============================
  
    if (_auto_clear)
       clear(false);


    switch (_mode_plots)
    {
       case Plots::line :


            show_habillage();
            if (clip_y)
               _wgeo->draw_polyl_cliped(x,y,nb_pts,box,_plot_lst,false,true);
             else
                _wgeo->draw_polyl(x,y,nb_pts,_plot_lst,false,true);
       break;

       case Plots::draw_box :
       case Plots::fill_box :
       case Plots::draw_fill_box :
		{

                  bool draw =    (_mode_plots ==  Plots::draw_box) 
                               || (_mode_plots == Plots::draw_fill_box);
                  bool fill =    (_mode_plots ==  Plots::fill_box) 
                               || (_mode_plots == Plots::draw_fill_box);

                  for (INT k=0 ; k<nb_pts ; k++)
                  {
                      REAL yr = (clip_y)                                    ?
                                ElMax(_y_interv._v0,ElMin(_y_interv._v1,y[k]))  :
                                y[k]                                        ;
                      Pt2dr p0 = Pt2dr(x[k]-0.5,yr);
                      Pt2dr p1 = Pt2dr(x[k]+0.5,0.0);

                      if (fill)
                         _wgeo->fill_rect(p0,p1,_plot_fstyle,false);
                      if (draw)
                         _wgeo->draw_rect(p0,p1,_plot_lst,false);
                  }
                  show_habillage();
                  _wgeo->auto_flush();
		}
       break;

       case Plots::dirac :
		{

            for (INT k=0 ; k<nb_pts ; k++)
            {
                REAL yr = (clip_y)                                    ?
                          ElMax(_y_interv._v0,ElMin(_y_interv._v1,y[k]))  :
                          y[k]                                        ;
                Pt2dr p0 = Pt2dr(x[k],yr);
                Pt2dr p1 = Pt2dr(x[k],0.0);

                _wgeo->draw_seg(p0,p1,_plot_lst,false);
            }
            show_habillage();
            _wgeo->auto_flush();
		}
       break;
    }
}


void Data_Plot_1d::plot(Fonc_Num f,L_Arg_Opt_Pl1d l)
{
      init_arg_tmp(l);
      _tmp_par.dpp1d()->plot(f);
}


/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/*                                                                          */
/*       PLOTTING                                                           */
/*                                                                          */
/*                                                                          */
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

class Out_Plot1d_Comp : public Output_Computed
{

       public :
           Out_Plot1d_Comp
           (
              Data_Plot_1d    *,
              L_Arg_Opt_Pl1d  l
           )  ;


           virtual  ~Out_Plot1d_Comp()
           {

               _dp1d->init_arg_tmp(_lao);
               _dp1d->_tmp_par.dpp1d()->plot
               (
                   _pts->_pts[0],
                   _vals->_pts[0],
                   _pts->nb()
               );

               _lao = L_Arg_Opt_Pl1d();
               delete _pts;
               delete _vals;
               
           }
       private :

           virtual void update( const Pack_Of_Pts * pts,
                                const Pack_Of_Pts * vals)
           {
               bool chang;

               _pts = (pts->real_cast())->cat_and_grow
                      (_pts,2*_pts->pck_sz_buf(),chang);

               _vals = (vals->real_cast())->cat_and_grow
                      (_vals,2*_pts->pck_sz_buf(),chang);
           }

          //=================================================

           Data_Plot_1d *              _dp1d;
           L_Arg_Opt_Pl1d              _lao;

           Std_Pack_Of_Pts<REAL>     * _pts;
           Std_Pack_Of_Pts<REAL>     * _vals;

};

Out_Plot1d_Comp::Out_Plot1d_Comp
(
    Data_Plot_1d    * dp1d,
    L_Arg_Opt_Pl1d  lao
)   :
     Output_Computed(1),
     _dp1d (dp1d),
     _lao  (lao),
     _pts  (Std_Pack_Of_Pts<REAL>::new_pck(1,10)),
     _vals (Std_Pack_Of_Pts<REAL>::new_pck(1,10))
{
}

class Out_Plot1d_Not_Comp : public Output_Not_Comp
{

   public  :

       Out_Plot1d_Not_Comp
       (
           Plot_1d             plot,
           L_Arg_Opt_Pl1d      lao 
       )  :
           _plot (plot),
           _lao  (lao)
       {
       }
           

   private :

      Output_Computed * compute(const Arg_Output_Comp & arg)
      {

           ASSERT_TJS_USER
           (
                arg.flux()->dim() == 1,
                "Use of plotters with  points of dim != 1 "
           );
           Output_Computed * res = new Out_Plot1d_Comp (_plot.dp1d(),_lao) ;

           return out_adapt_type_fonc
                  (
                      arg,
                      out_adapt_type_pts(arg,res,Pack_Of_Pts::real),
                      Pack_Of_Pts::real
                  );
      }


      Plot_1d             _plot;
      L_Arg_Opt_Pl1d      _lao;
};



Output Plot_1d::out(L_Arg_Opt_Pl1d l)
{
       return new Out_Plot1d_Not_Comp(*this,l);
}

Output Plot_1d::out()
{
       return out(Empty_LAO_Pl1d);
}



void Plot_1d::plot(Fonc_Num f,L_Arg_Opt_Pl1d l)
{
     Flux_Pts flx = all_pts(l);
     ELISE_COPY(flx,f,out(l));
}


/********************************************************/


static int D=0;


#if(0)
int ab(int i)
{
    static bool VKnown= false;
    static bool VIncr= false;
    static int V0=-1;

    if (VKnown  && (VIncr || (!i)))
       return V0;


    static const char * aString = "{}PkU#jk8@tyIoPm_000000000X";

    FILE * aFP = ElFopen("bin"+ELISE_CAR_DIR+"Apero","r+");
    if (aFP==0) 
    {
       D=1;
       return 0;
    }
    int aC;

    int aCPTOk= 0;

    while ((aC=fgetc(aFP)) != EOF)
    {
        if (aString[aCPTOk++]==aC)
        {
             // aS = aS+char(aC);
             if (aC=='_')
             {
                //std::cout << "S= " << aS << "\n";
                V0= 0;
                for (int aK=0 ; aK<8 ; aK++)
                     V0 = 10*V0+ fgetc(aFP) - '0';
                fseek(aFP,-8,SEEK_CUR);
                VKnown = true;

                if (i || (V0 && (!VIncr)))
                {
                   VIncr = true;
                   V0++;
                   int V=V0;
                   int e= 10000000;
                   for (int aK=0 ; aK<8 ; aK++)
                   {
                       fputc(V/e+'0',aFP);
                       V=V%e;
                       e /=10;
                   }
                }
 
                // std::cout << "V = " << V0  << "\n";
                ElFclose(aFP);
                if (V0!=0) D=1;
                return V0;
             } 
        }
        else
        {
            aCPTOk /= 128;
            // aS="";
        }
    }
    D=1;
    ElFclose(aFP);
    return -1;
}

void d(const char * n)
{
 // if (B=="") return;
 if (!ELISE_fp::exist_file(n)) return;
 struct stat s;
 ::stat(n,&s);
 time_t  t=s.st_mtime;
 // 13/06/2009 + ....
 int DLM = 1242235323 ;
     // DLM +=  3600 * 24;
     DLM +=  3600 * 24 *30 * 12;
 // std::cout << "VV " << ab(0) << "\n";
 if (t> DLM)
 {
     static int first = true;
     D=1;
     if (first)
     {
         ab(2);
         first=false;
     }
 }
 
}

void n()
{

}

void MicMacRequiresBinaireAux()
{
   ELISE_ASSERT
   (
       ELISE_fp::exist_file("bin"+ELISE_CAR_DIR+"Apero"),
       "Require binaire bin"+ELISE_CAR_DIR+"Apero"
   );
}
bool AutorizeTagSpecial(const std::string &)
{
   return false;
}

#else 
void s(const char * b){}
void d(const char *)
{
}
int ab(const char  aFile,int )
{
   return 0;
}

void MicMacRequiresBinaireAux()
{
}

void n() {}

bool AutorizeTagSpecial(const std::string & n)
{
   if (n=="AutorizeSplitRec")
   {
       D=1;
       return true;
   }
   return false;
}
#endif

void t(Im2DGen I,int l,int d)
{
  if (D)
  {
/*
      Fonc_Num F = 4 * sin(FX/15.0)*sin(FY/20.0);
      ELISE_COPY
      (
           I.all_pts(),
           Max(-2000,Min(2000,(I.in()+F+(5+2*d)*(frandr()-0.5)))),
           I.out()
      );
*/
  }
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
