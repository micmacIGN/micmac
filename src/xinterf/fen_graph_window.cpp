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
/*                  Graph_8_neigh_Win_Comp                       */
/*                                                               */
/*****************************************************************/

class  Graph_8_neigh_Win_Comp : public  Output_Computed
{
   public :

      Graph_8_neigh_Win_Comp
      (
             const Data_El_Geom_GWin *,
             Data_Line_St            *dlst,
             bool                    sym,
             bool                    ortho 
      );




      virtual ~Graph_8_neigh_Win_Comp()
      {
             _degw->_degd->disp_flush();
      }

    protected :
    private :

       virtual void update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals);

       Pt2dr                          _tr;
       Pt2dr                          _sc;
       Data_Line_St               * _dlst;
       Data_Elise_Gra_Win         * _degw;
       bool                         _sym;
       bool                         _ortho;
};

Graph_8_neigh_Win_Comp::Graph_8_neigh_Win_Comp
(
     const Data_El_Geom_GWin *    geom,
     Data_Line_St            *    dlst,
     bool                         sym,
     bool                         ortho
)  :
      Output_Computed(1),
      _tr            (geom->tr()),
      _sc            (geom->sc()),
      _dlst          (dlst),
      _degw          (geom->degw()),
      _sym           (sym),
      _ortho         (ortho)
{
}

void Graph_8_neigh_Win_Comp::update
     (
           const Pack_Of_Pts * gpts,
           const Pack_Of_Pts * gvals
     )
{
      _degw->set_line_style(_dlst);

      const Std_Pack_Of_Pts<REAL>* pts = gpts->real_cast();
      const Std_Pack_Of_Pts<INT> * vals = gvals->int_cast();

      REAL * x = pts->_pts[0];
      REAL * y = pts->_pts[1];
      INT  * v = vals->_pts[0];


      REAL x1[NB_BUF_DRAW_POLY]; 
      REAL x2[NB_BUF_DRAW_POLY]; 
      REAL y1[NB_BUF_DRAW_POLY]; 
      REAL y2[NB_BUF_DRAW_POLY]; 

      INT nb_in_buf = 0;
      INT nb_pts = gpts->nb();

      Pt2dr p0dir[8];
      Pt2dr tdir[8];
      for (INT f = 0; f<8 ; f++)
      {
          tdir[f] =    rto_window_geom(Pt2dr(TAB_8_NEIGH[f]),_tr,_sc)
                    -  rto_window_geom(Pt2dr(0,0),_tr,_sc);
          p0dir[f] = Pt2dr(0,0);
          if (_ortho)
          {
              Pt2dr m = (tdir[f]+p0dir[f]) / 2.0;
              Pt2dr v = (tdir[f]-p0dir[f])/2.0;
              tdir[f] = m + v * Pt2dr(0,1);
              p0dir[f]= m - v * Pt2dr(0,1);
          }
          else if (! _sym)
              tdir[f] = tdir[f] / 2.0;
      }

      for(INT k=0; k<nb_pts; k++)
      {
          INT flags = v[k];
          Pt2dr p1 = rto_window_geom(Pt2dr(x[k],y[k]),_tr,_sc);
          for (INT f = 0; f<8 ; f++)
              if (flags & (1 << f))
              {
                 x1[nb_in_buf  ] = round_ni_inf(p1.x + p0dir[f].x);
                 y1[nb_in_buf  ] = round_ni_inf(p1.y + p0dir[f].y);
                 x2[nb_in_buf  ] = round_ni_inf(p1.x + tdir[f].x);
                 y2[nb_in_buf++] = round_ni_inf(p1.y + tdir[f].y);
                 if (nb_in_buf == NB_BUF_DRAW_POLY)
                 {
                     _degw->_inst_draw_poly_segs(x1,y1,x2,y2,nb_in_buf);
                     nb_in_buf = 0;
                 }
              }
      }
      _degw->_inst_draw_poly_segs(x1,y1,x2,y2,nb_in_buf);
}


class  Graph_8_neigh_Not_Win_Comp : public   Output_Not_Comp
{
       public :
          Graph_8_neigh_Not_Win_Comp(Line_St,El_Window,bool,bool);

       private :

          Line_St    _lst;   // dlst()
          El_Window  _w;  // dagrw
          bool       _sym;
          bool       _ortho;


          virtual  Output_Computed * compute(const Arg_Output_Comp & arg)
          {
               ASSERT_TJS_USER
               (  arg.flux()->dim() == 2,
                  "DIM of PtS != 2, in graph-window writing"
               );

               Output_Computed * res =
                                     new
                                     
                                           Graph_8_neigh_Win_Comp
                                           (_w.degeow(),_lst.dlst(),_sym,_ortho)
                                     ;

               res = out_adapt_type_fonc(arg,res,Pack_Of_Pts::integer);
               res = out_adapt_type_pts (arg,res,Pack_Of_Pts::real);

               return res;
          }
};

Graph_8_neigh_Not_Win_Comp::Graph_8_neigh_Not_Win_Comp
(
    Line_St    lst,
    El_Window  w,
    bool       sym,
    bool       ortho
)  :
     _lst   (lst),
     _w     (w  ),
     _sym   (sym),
     _ortho (ortho)
{
}


Output El_Window::out_graph(Line_St lst,bool sym)
{
   return new Graph_8_neigh_Not_Win_Comp(lst,*this,sym,false);
}

Output El_Window::out_fr_graph(Line_St lst)
{
   return new Graph_8_neigh_Not_Win_Comp(lst,*this,false,true);
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
