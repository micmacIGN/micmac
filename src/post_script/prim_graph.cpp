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

void Data_Elise_PS_Disp::line(Pt2dr p1,Pt2dr p2)
{
    Ps_Pts(_fd,p1,3);
    Ps_Pts(_fd,p2,3);
    _line.put_prim(this);
}


void Data_Elise_PS_Disp::fill_rect(Pt2dr p1,Pt2dr p2)
{
    Ps_Pts(_fd,p1,3);
    Ps_Pts(_fd,p2,3);
    _fd << "rectfill\n";
}

void Data_Elise_PS_Disp::dr_rect(Pt2dr p1,Pt2dr p2)
{
    Ps_Pts(_fd,p2.yx(),3);
    Ps_Pts(_fd,p1,3);
    _dr_rect.put_prim(this);
}


void Data_Elise_PS_Disp::dr_circle(Pt2dr centre,REAL radius)
{
    Ps_Pts(_fd,centre,3);
    Ps_Real_Prec(_fd,radius,3);
    _fd << " ";
    _dr_circ.put_prim(this);
}




void Data_Elise_PS_Disp::draw_poly(const REAL * x,const REAL *y,INT nb)
{
    if (nb<=1) return;

// Try to see if x is regulary spaced 

    REAL dx = (x[nb-1] - x[0]) / (nb-1);
    REAL tol = 1e-6;
    REAL delta_max   = 0;

    {
       REAL x0 = x[0];
       for (INT k = 0 ; (k< nb)&&(delta_max < tol) ; k++)
          delta_max = ElMax(ElAbs(x[k]-(x0+k*dx)),delta_max);
    }

    
    if (delta_max<tol)
    {
         for (INT k = nb-1; k>=1 ; k--)
         {
              Ps_Real_Prec(_fd,y[k]-y[k-1],4);
              _fd << "\n";
         }
         Ps_Real_Prec(_fd,dx,4); _fd << "\n";
         _fd << nb-1 << "\n";
         Ps_Pts(_fd,Pt2dr(x[0],y[0]),3);
     
         _dr_polyFxy.put_prim(this);
    }
    else
    {
         for (INT k = nb-1; k>=1 ; k--)
         {
              Ps_Pts(_fd,Pt2dr(x[k]-x[k-1],y[k]-y[k-1]),4);
              _fd << "\n";
         }
         _fd << nb-1 << "\n";
         Ps_Pts(_fd,Pt2dr(x[0],y[0]),3);
     
         _dr_poly.put_prim(this);
    }
}


/***************************************************************/

REAL Data_Elise_PS_Disp::compute_sz_pixel(Pt2dr sz,Pt2dr margin,Pt2di nb,Pt2dr inside_margin)
{
     return ElMin
            (
                 ((_sz_page.x -2*margin.x) -(inside_margin.x*(nb.x-1))) / (nb.x * sz.x),
                 ((_sz_page.y -2*margin.y) -(inside_margin.y*(nb.y-1))) / (nb.y * sz.y)
            );
}

Box2dr  Data_Elise_PS_Disp::box_all_w(Pt2dr sz,Pt2dr margin,Pt2di nb,Pt2dr inside_margin)
{
       REAL sz_pixel = compute_sz_pixel(sz,margin,nb,inside_margin);

       Pt2dr sz_all =  sz.mcbyc(Pt2dr(nb)) *sz_pixel + inside_margin.mcbyc(Pt2dr(nb.x-1.0,nb.y-1.0)); 
       Pt2dr p0 = (_sz_page-sz_all)/2;

       return Box2dr(p0,_sz_page-p0);
}



PS_Window    PS_Display::w_centered_max(Pt2di sz,Pt2dr margin)
{
    Box2dr b =  depsd()->box_all_w(Pt2dr(sz),margin,Pt2di(1,1),Pt2dr(0.0,0.0));
    return PS_Window(*this,sz,b._p0,b._p1);
}

          //===============================
          //   Data_Mat_PS_Window
          //===============================


class Data_Mat_PS_Window : public RC_Object
{
     public :

          Data_Mat_PS_Window(PS_Display The_disp,Pt2di sz,Pt2dr margin,Pt2di nb,Pt2dr inside_margin);
          virtual ~Data_Mat_PS_Window();

          PS_Window kthw(INT x,INT y);

     private  :

         PS_Window ** _w;
         Pt2di        _nb;
};


PS_Window Data_Mat_PS_Window::kthw(INT x,INT y)
{
    Tjs_El_User.ElAssert
    (
          Pt2di(x,y).in_box(Pt2di(0,0),_nb),
          EEM0 << "invalid request for windows of PS MAT WINDOW"
    );

    return _w[y][x];
}



Data_Mat_PS_Window::Data_Mat_PS_Window
(
       PS_Display         The_disp,
       Pt2di sz,
       Pt2dr margin,
       Pt2di nb,
       Pt2dr inside_margin
)   :
    _w      (0),
    _nb     (nb)
{
    Data_Elise_PS_Disp * disp = The_disp.depsd();

    _w  = NEW_VECTEUR(0,nb.y,PS_Window *);
    {
        for (INT y = 0; y<nb.y ; y++)
            _w[y] = new PS_Window [nb.x];
    }
    
    Box2dr b =  disp->box_all_w(Pt2dr(sz),margin,nb,inside_margin);
    REAL sz_pix = disp->compute_sz_pixel(Pt2dr(sz),margin,nb,inside_margin);

    Pt2dr sz_w      =  Pt2dr(sz)    * sz_pix;
    Pt2dr sz_w_marg = inside_margin + sz_w;

    {
        for (INT y = 0; y<nb.y ; y++)
            for (INT x = 0; x<nb.x ; x++)
            {
                 INT yi = nb.y-1-y;
                 Pt2dr p0 = b._p0 + Pt2dr(x,yi).mcbyc(sz_w_marg);
                 Pt2dr p1 = p0 + sz_w;
                 _w[y][x] = PS_Window(The_disp,sz,p0,p1);
            }
    }
}


Data_Mat_PS_Window::~Data_Mat_PS_Window()
{
    for (INT y =_nb.y-1 ; y>=0 ; y--)
        delete [] _w[y];
     DELETE_VECTOR(_w,0);
}



PS_Window Mat_PS_Window::operator() (int x,int y)
{
   return ((Data_Mat_PS_Window *) _ptr)->kthw(x,y);
}


Mat_PS_Window::Mat_PS_Window
(
       PS_Display         disp,
       Pt2di sz,
       Pt2dr margin,
       Pt2di nb,
       Pt2dr inside_margin
)    :
     PRC0
     (
         new Data_Mat_PS_Window
             (
                 disp,
                 sz,
                 margin,
                 nb,
                 inside_margin
             )

     )
{
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
