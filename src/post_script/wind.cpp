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

/**********************************************************************/
/*                                                                    */
/*                Data_Elise_PS_Disp                                  */
/*                                                                    */
/**********************************************************************/

void Data_Elise_PS_Disp::set_active_window(Data_Elise_PS_Win * w)
{
     if (_num_last_act_win == w->_ps_num)
        return;

     if (_num_last_act_win != -1)
     {
        _active_pal = 0;
        reinit_cp();
     }
     prim_win_coord(_fd,w); _fd << "\n";

     _num_last_act_win = w->_ps_num;
}



void Data_Elise_PS_Disp::prim_win_coord(ofstream & f,class Data_Elise_PS_Win * w)
{
     f << ElPsPREFIX << "Wc" << w->_ps_num ;
}


void Data_Elise_PS_Disp::ins_window(Data_Elise_PS_Win * w)
{

      // fill_rect(w->_p0,w->_szpica);
      if (_nb_win)
      {
         _p0_box = Inf(_p0_box,w->_p0);
         _p1_box = Inf(_p1_box,w->_p1);
      }
      else
      {
         _p0_box = w->_p0;
         _p1_box = w->_p1;
      }
      _nb_win++;


      REAL sc_x = w->_scale.x *w->_geo_sc.x;
      REAL sc_y = -w->_scale.y *w->_geo_sc.y;
      REAL tr_x = (w->_p0.x -w->_geo_tr.x * sc_x);
      REAL tr_y = (w->_p0.y +w->_szpica.y-w->_geo_tr.y * sc_y);


      if ( _lgeo_clip.new_get_num
                 (
                      w->_p0,
                      w->_szpica,
                      Pt2dr(sc_x,sc_y),
                      Pt2dr(tr_x,tr_y),
                      w->_ps_num
                 )
          )
      {
           INT num_cl;
           if (_lclip.new_get_num
                      (
                           w->_p0,
                           w->_szpica,
                           Pt2dr(sc_x,sc_y),
                           Pt2dr(tr_x,tr_y),
                           num_cl
                      )
               )
            {
                 _fh << "/" << "Elclip" << num_cl << "\n";
                 _fh << "{\n";
                 _fh << "newpath " <<w->_p0.x<< " " <<  w->_p0.y<< " moveto \n";
                 _fh <<  w->_szpica.x << " " <<0<< " rlineto \n";
                 _fh << 0 << " " <<  w->_szpica.y << " rlineto \n";
                 _fh << -w->_szpica.x << " " <<  0 << " rlineto \n";
                 _fh << "closepath clip\n";
                 _fh << "} def\n";

            }


          _fh << "/";   prim_win_coord(_fh,w);  _fh << "\n";
          _fh << "{\n";
          _fh << "grestore gsave \n";

          _fh << "Elclip" << num_cl << "\n";

          _fh << tr_x << " " << tr_y << " translate \n";
          _fh << sc_x << " " << sc_y << " scale \n";
          _fh << "}\n";
          _fh << "def\n";
      }
}


/**********************************************************************/
/*                                                                    */
/*                Data_Elise_PS_Win                                   */
/*                                                                    */
/**********************************************************************/

bool Data_Elise_PS_Win::adapt_vect()
{
    return false;
}


void Data_Elise_PS_Win::_inst_set_col(Data_Col_Pal * p)
{
     _psd->set_active_palette(p->pal(),false);
     _psd->set_cur_color(p->cols());
     
}

void Data_Elise_PS_Win::_inst_draw_seg(Pt2dr p1,Pt2dr p2)
{
     _psd->line(p1,p2);
}


void Data_Elise_PS_Win::set_active()
{
     _psd->set_active_window(this);
}

void Data_Elise_PS_Win::_inst_fill_rectangle(Pt2dr p1,Pt2dr p2)
{
     _psd->fill_rect(p1,p2-p1);
}

void Data_Elise_PS_Win::_inst_draw_rectangle(Pt2dr p1,Pt2dr p2)
{
     _psd->dr_rect(p1,p2-p1);
}

void Data_Elise_PS_Win::_inst_draw_polyl(const REAL * x,const REAL *y,INT nb)
{
    _psd->draw_poly(x,y,nb);
}


void Data_Elise_PS_Win::_inst_draw_circle (Pt2dr centre ,Pt2dr radius)
{
     _psd->dr_circle(centre,euclid(radius));
}


Data_Elise_PS_Win::Data_Elise_PS_Win
(
      PS_Display psdisp,
      Pt2di sz,
      Pt2dr p0,
      Pt2dr p1,
      Pt2dr geo_tr,
      Pt2dr geo_sc
)   :
    Data_Elise_Gra_Win
    (
          psdisp.depsd(),
          sz,
          psdisp.depsd()->_sop,
          false
    ) ,
    _p0_ori  (p0),
    _p1_ori  (p1),
    _geo_tr  (geo_tr),
    _geo_sc  (geo_sc),
    _Ptr_psd (psdisp),
    _psd     (psdisp.depsd())
{
   p0 = p0 * PS_Display::picaPcm; 
   p1 = p1 * PS_Display::picaPcm; 
   pt_set_min_max(p0,p1);

   _p0 = p0;
   _p1 = p1;
   _szpica = p1-p0;
   _sz = sz;
   _scale = Pt2dr(_szpica.x/_sz.x,_szpica.y/_sz.y);

   _psd->ins_window(this);
}


Data_Elise_Gra_Win * Data_Elise_PS_Win::dup_geo(Pt2dr geo_tr,Pt2dr  geo_sc)
{

   Data_Elise_Gra_Win * res =
         new Data_Elise_PS_Win 
         (
            _Ptr_psd,
            _sz,
            _p0_ori,
            _p1_ori,
            geo_tr,
            geo_sc
         );

  return res;
}




/**********************************************************************/
/*                                                                    */
/*                PS_Window                                           */
/*                                                                    */
/**********************************************************************/


PS_Window::PS_Window
(
          PS_Display psd,
          Pt2di sz,
          Pt2dr p0,
          Pt2dr p1
) :
      El_Window
      (
          new Data_Elise_PS_Win (psd,sz,p0,p1,Pt2dr(0,0),Pt2dr(1,1)),
          Pt2dr(0,0),
          Pt2dr(1,1)
      )
{
}

PS_Window::PS_Window
(
     Data_Elise_PS_Win * w,
     Pt2dr               tr,
     Pt2dr               sc
) :
      El_Window(w->dup_geo(tr,sc),tr,sc)
{
}


PS_Window PS_Window::chc(Pt2dr tr,Pt2dr sc)
{
   return   PS_Window(depw(),tr,sc);
}
          

PS_Window::PS_Window () :
    El_Window(0)
{
}

PS_Window  PS_Window::WStd
           (
	       const std::string & aName,
               Pt2di aSz,
	       Pt2dr aMargin
           )
{
    PS_Display aDisp
               (
	           aName.c_str(),
		   "Mon beau fichier ps",
		   Elise_Set_Of_Palette::TheFullPalette()
               );

     return aDisp.w_centered_max(aSz,aMargin);
}


/**********************************************************************/
/*                                                                    */
/*                Data_Elise_PS_Disp::NumGeAlloc                      */
/*                                                                    */
/**********************************************************************/

Data_Elise_PS_Disp::LGeom::~LGeom()
{
    if (_next) delete _next;
}

Data_Elise_PS_Disp::NumGeAlloc::~NumGeAlloc()
{
    if (_lg) delete _lg;
}

Data_Elise_PS_Disp::LGeom::LGeom
( 
          Pt2dr p1,
          Pt2dr p2,
          Pt2dr tr,
          Pt2dr sc,
          INT   num,
          LGeom  * next
)  :
  _p1  (p1),
  _p2  (p2),
  _tr  (tr),
  _sc  (sc),
  _num (num),
  _next (next)
{
}

Data_Elise_PS_Disp::NumGeAlloc::NumGeAlloc(bool box_only) :
     _lg(0),
     _num  (0),
     _box_only (box_only)
{
}


bool Data_Elise_PS_Disp::NumGeAlloc::new_get_num
     ( 
          Pt2dr p1,
          Pt2dr p2,
          Pt2dr tr,
          Pt2dr sc,
          INT &  num
     )
{
     for (LGeom * lg=_lg ; lg!=0 ; lg = lg->_next)
     {
         if (
                   (euclid(p1-lg->_p1) < 1e-6)
                && (euclid(p2-lg->_p2) < 1e-6)
                && (
                        _box_only
                     || (
                               (euclid(tr-lg->_tr) < 1e-6)
                            && (euclid(sc-lg->_sc) < 1e-6)
                        )
                   )
            )
          {
               num = lg->_num;
               return false;
          }
     }
     num =  _num;
     _lg = new LGeom(p1,p2,tr,sc,_num++,_lg);
     return true;
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
