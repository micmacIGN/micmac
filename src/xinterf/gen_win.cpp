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
/*                  GW_computed                                  */
/*                                                               */
/*****************************************************************/

GW_computed::~GW_computed(){}

GW_computed::GW_computed
(
       const Arg_Output_Comp & ,
       const Data_El_Geom_GWin * degw,
       Data_Elise_Palette      * pal
) :
       Output_Computed(pal->dim_pal()),
      _sz ( degw->sz()),
      _tr ( degw->tr()),
      _sc ( degw->sc()),
      _geom (degw)
{
      degw->box_user_geom(_up0,_up1);
}



/*****************************************************************/
/*                                                               */
/*                  DE_GW_Not_Comp                               */
/*                                                               */
/*****************************************************************/

class  DE_GW_Not_Comp : public Output_Not_Comp
{
   public :


      DE_GW_Not_Comp(El_Window ,Elise_Palette PAL,bool OnYDiff=false);

      Output_Computed * compute(const Arg_Output_Comp & );

   private :

      El_Window          _ew;
      Elise_Palette      _pal;
      bool               mOnYDiff;
};

DE_GW_Not_Comp::DE_GW_Not_Comp(El_Window ew,Elise_Palette PAL,bool OnYDiff) :
    _ew      (ew),
    _pal     (PAL),
    mOnYDiff (OnYDiff)
{
}


Output_Computed * DE_GW_Not_Comp::compute(const Arg_Output_Comp & arg)
{
      ASSERT_TJS_USER
      (
           arg.flux()->dim() == 2,
           "need 2-dimensional sets to write in raster Graphic Window"
      );

      // get the general representation of window
      Data_El_Geom_GWin  * geom = _ew.degeow();
      ASSERT_TJS_USER
      (
             (geom->handle_rastser_scale_negatif()) || ((geom->sc().x>0) && (geom->sc().y>0)),
             "can only handle positive scales in raster mode (for now)"
      );
      Data_Elise_Gra_Win * degw = _ew.degraw();
      degw->_degd->reinit_cp();
      degw->warn_graph();


      (degw->sop()).pal_is_loaded(_pal);
      Data_Elise_Palette *   dep  = _pal.dep();
      dep->verif_out_put(arg);

      Output_Computed * res;
      switch (arg.flux()->type())
      {
              case Pack_Of_Pts::rle   :
                     res = degw->rle_out_comp(geom,arg,dep,mOnYDiff);
              break;

           case Pack_Of_Pts::integer   :

              INT v[Elise_Std_Max_Dim];
              if (arg.fonc()->icste(v))
                 res = degw->pint_cste_out_comp(geom,arg,dep,v);
              else
                 res = degw->pint_no_cste_out_comp(geom,arg,dep);
           break;

           default :
               elise_fatal_error
               ("Raster Windows cannot manage real points (for now)",__FILE__,__LINE__);
               res = 0;
      }

     res = out_adapt_type_fonc (arg,res,Pack_Of_Pts::integer);
     {
         Pt2di p0,p1;
         geom->box_user_geom(p0,p1);
         res = clip_out_put(res,arg,p0,p1);
     }

      return res;
}

/*****************************************************************/
/*                                                               */
/*                  El_Window                                    */
/*                                                               */
/*****************************************************************/

El_Window::~El_Window() {}


El_Window::El_Window(Data_El_Geom_GWin * degw) :
      PRC0(degw)
{
}

El_Window::El_Window(Data_Elise_Gra_Win * W,Pt2dr tr,Pt2dr sc) :
     PRC0 (new Data_El_Geom_GWin(W,tr,sc))
{
}


El_Window El_Window::chc(Pt2dr tr,Pt2dr sc)
{
    if (! degeow()->_dnext)
         return El_Window(degraw()->dup_geo(tr,sc),tr,sc);

    El_Window wn = degeow()->_next.chc(tr,sc);

    return new Data_El_Geom_GWin
              (
                   degraw()->dup_geo(tr,sc),
                   tr,
                   sc,
                   wn.degeow()
              );
/*
*/
}

Pt2di El_Window::sz() const
{
    return degeow()->sz();
}


El_Window El_Window::operator |(El_Window w)
{
    return new Data_El_Geom_GWin
              (
                   degraw(),
                   Pt2dr(0,0),
                   Pt2dr(1,1),
                   w.chc(Pt2dr(0,0),Pt2dr(1,1)).degeow()
              );
}




         //---------------------------------------------


Output El_Window::out(Elise_Palette pal,bool OnYDif)
{
   Output o = new DE_GW_Not_Comp(*this,pal,OnYDif);

   if (degeow()->_dnext)
      o = o | degeow()->_next.out(pal);

   return  o;
}

Data_Elise_Gra_Win * El_Window::degraw() const
{
    return degeow()->degw();
}




Elise_Rect El_Window::box() const
{
     return degeow()->box();
}

Output El_Window::out(TYOFPAL::t type,bool OnYDif)
{

      return out(degraw()->sop().pal_of_type(type),OnYDif);
}

Output El_Window::ogray()
{
    return out(TYOFPAL::gray);
}

Output El_Window::orgb()
{
    return out(TYOFPAL::rgb);
}


Output El_Window::odisc(bool OnYDif)
{
    return out(TYOFPAL::disc,OnYDif);
}

Output El_Window::obicol()
{
    return out(TYOFPAL::bicol);
}

Output El_Window::ocirc()
{
    return out(TYOFPAL::circ);
}

Output El_Window::olin1()
{
    return out(TYOFPAL::lin1);
}


void El_Window::draw_seg(Pt2dr p1,Pt2dr p2,Line_St lst)
{
     degeow()->draw_seg(p1,p2,lst.dlst());
}


void El_Window::draw_rect(Pt2dr p1,Pt2dr p2,Line_St lst)
{
     degeow()->draw_rect(p1,p2,lst.dlst());
}


void El_Window::fill_rect(Pt2dr p1,Pt2dr p2,Fill_St fst)
{
     degeow()->fill_rect(p1,p2,fst.dfst());
}


void El_Window::draw_circle_loc(Pt2dr p1,REAL radius,Line_St lst)
{
     degeow()->draw_circle(p1,Pt2dr(radius,radius),lst.dlst(),true);
}

Pt2dr El_Window::U2W(Pt2dr aP)
{
    return degeow()->prto_window_geom(aP);
}
Pt2dr El_Window::W2U(Pt2di aP)
{
    return degeow()->to_user_geom(aP);
}


void El_Window::draw_ellipse_loc
     (
     Pt2dr aCentre,REAL A,REAL B,REAL C,Line_St aLst,
         INT Nb
      )
{
   InvertParamEllipse(A,B,C,A,B,C);

   Pt2dr LastP;
   for (INT aK= 0 ; aK<= Nb ; aK++)
   {
        Pt2dr P = aCentre+ImAppSym(A,B,C,Pt2dr::FromPolar(1.0,(aK*2*PI)/Nb));
    if (aK !=0)
           draw_seg(LastP,P,aLst);
    // cout << LastP << P << "\n";
    LastP = P;
   }
}

void El_Window::draw_circle_abs(Pt2dr p1,REAL radius,Line_St lst)
{
     degeow()->draw_circle(p1,Pt2dr(radius,radius),lst.dlst(),false);
}


void El_Window::fixed_string(Pt2dr pt,const char * name,Col_Pal col,bool draw_image)
{
/*
     if (name[0] == 0)
     {
            std::cout << "NOTRTTT  El_Window::fixed_string " << "\n";
            return;
     }
*/
     degeow()->fixed_string(pt,name,col.dcp(),draw_image);
}


Elise_Set_Of_Palette  El_Window::sop()
{
   return degraw()->sop();
}



Elise_Palette  El_Window::palette(TYOFPAL::t type)
{
      return degraw()->sop().pal_of_type(type);
}

Disc_Pal El_Window::pdisc()
{
    return palette(TYOFPAL::disc);
}
RGB_Pal El_Window::prgb()
{
    return palette(TYOFPAL::rgb);
}

Gray_Pal El_Window::pgray()
{
    return palette(TYOFPAL::gray);
}

Circ_Pal El_Window::pcirc()
{
    return palette(TYOFPAL::circ);
}






void El_Window::draw_arrow
     (
         Pt2dr p0, Pt2dr p1, Line_St LAxe, Line_St LPointe,
         REAL size_pointe, REAL pos , REAL teta
     )
{
     draw_seg(p0,p1,LAxe);

     Pt2dr q0 = barry(1-pos,p0,p1);
     Pt2dr dir_pte = Pt2dr::FromPolar(1.0,teta);
     Pt2dr tgt = vunit(p0-p1) * size_pointe;

     draw_seg(q0,q0+tgt*dir_pte,LPointe);
     draw_seg(q0,q0+tgt/dir_pte,LPointe);
}

void El_Window::draw_arrow
     (
         Pt2dr p0, Pt2dr p1, Line_St lst,
         REAL size_pointe, REAL pos , REAL teta
     )
{
     draw_arrow(p0,p1,lst,lst,size_pointe,pos,teta);
}

void El_Window::draw_rect(Box2dr aBox,Line_St aLst)
{
   draw_rect(aBox._p0,aBox._p1,aLst);
}

void El_Window::draw_poly(const std::vector<Pt2dr> & aV,Line_St aLst,bool isFerm)
{
   int aNb = (int) aV.size();
   for (int aK=0 ; aK<aNb-!isFerm ; aK++)
       draw_seg(aV[aK],aV[(aK+1)%aNb],aLst);
}

void El_Window::draw_poly_ferm(const std::vector<Pt2dr> & aV,Line_St aLst)
{
    draw_poly(aV,aLst,true);
}

void El_Window::draw_poly_ouv(const std::vector<Pt2dr> & aV,Line_St aLst)
{
    draw_poly(aV,aLst,false);
}



/*****************************************************************/

/*****************************************************************/
/*                                                               */
/*                  Data_El_Geom_GWin                            */
/*                                                               */
/*****************************************************************/


Data_El_Geom_GWin::Data_El_Geom_GWin
(
      Data_Elise_Gra_Win * degw,
      Pt2dr tr,
      Pt2dr sc,
      Data_El_Geom_GWin * next
) :
     _egw       (degw),
     _degw      (degw),
     _dnext     (next),
     _next      (next),
     _tr        (tr.x,tr.y),
     _sc        (sc.x,sc.y),
     _num_geow  (degw->_degd->_nb_geow++),
     _adapt_vect (degw->adapt_vect())
{
}


/*-------------------------------------------------------------------

     Link between the doc X11_coord.tex and function elise :

      -  E+ : round_up, E- : round_down, I+  : round_ni,
      -  E++ : round_Uup, E-- : round_Ddown,

      -  rto_window_geom : FWx, FWy:
      -  ito_window_geom : I+ o FWx       (and I+ o FWy)
      -  to_window_geom   : I+ o FW
      -  to_user_geom  :  FW-1
      -  cor_tr_pix    : Cor
      - box_user_geom : compute GW-1 of windows box

-----------------------------------------------------------------------*/

void compute_tr_from_exemple (REAL & tr,REAL sc,REAL u,REAL w)
{
    ASSERT_INTERNAL((sc!=0),"/ by 0 in compute_tr_from_exemple");

    tr = (u-w/sc) / SIGN_TO_USER_GEOM;
}

void compute_tr_sc_from_exemple
            (REAL & tr,REAL & sc,REAL u1,REAL u2,REAL w1,REAL w2)
{
    ASSERT_INTERNAL((u1!=u2),"/ by 0 in compute_tr_sc_from_exemple");

    sc = (w1-w2) / (u1-u2);
    compute_tr_from_exemple(tr,sc,u1,w1);
}



Pt2di Data_El_Geom_GWin::pito_window_geom(Pt2dr p) const
{
     return Pt2di (  ito_window_geom(p.x,_tr.x,_sc.x),
                     ito_window_geom(p.y,_tr.y,_sc.y)
                  );
}

Pt2dr Data_El_Geom_GWin::prto_window_geom(Pt2dr p) const
{
     if ( ! _adapt_vect)
        return p;

     return Pt2dr (  rto_window_geom(p.x,_tr.x,_sc.x),
                     rto_window_geom(p.y,_tr.y,_sc.y)
                  );
}

Pt2dr Data_El_Geom_GWin::to_user_geom(Pt2di p) const
{
    return Pt2dr
          (
             rto_user_geom(p.x,_tr.x,_sc.x),
             rto_user_geom(p.y,_tr.y,_sc.y)
          );
}

void interval_window_to_user(INT & a1,INT & a2,INT u1,INT u2,REAL t,REAL s)
{
     if (s >= 1.0)
     {
        a1 = round_down(0.5+rto_user_geom(u1,t,s));
        a2 = round_Uup(0.5+rto_user_geom(u2-1,t,s));
     }
     else
     {
        a1 = round_up(rto_user_geom(u1-0.5,t,s));
        a2 = round_up(rto_user_geom(u2-0.5,t,s));
     }
}




void interval_user_to_window(INT & u1,INT & u2,INT a1,INT a2,REAL t,REAL s)
{
     u1 = deb_interval_user_to_window(a1,t,s);
     u2 = deb_interval_user_to_window(a2,t,s);
}




REAL cor_tr_pix(REAL scale)
{
    return  (0.5 * (scale-1) / scale);
}



void Data_El_Geom_GWin::box_user_geom(Pt2di & res1,Pt2di & res2) const
{
    interval_window_to_user(res1.x,res2.x,0,sz().x,_tr.x,_sc.x);
    interval_window_to_user(res1.y,res2.y,0,sz().y,_tr.y,_sc.y);
}



Elise_Rect  Data_El_Geom_GWin::box() const
{
   Pt2di p1,p2;

   box_user_geom(p1,p2);
   return Elise_Rect(p1,p2);
}


void Data_El_Geom_GWin::fixed_string
(
    Pt2dr               pt,
    const char *        name,
    Data_Col_Pal *      col,
    bool                draw_image,
    bool                auto_flush
)
{
    if (_dnext)
       _dnext->fixed_string(pt,name,col,draw_image,auto_flush);

    _degw->set_col(col);

    _degw->_inst_fixed_string(prto_window_geom(pt),name,draw_image);

    if (auto_flush)
       _degw->_degd->auto_flush();
}





void Data_El_Geom_GWin::draw_seg
(
       Pt2dr p1,
       Pt2dr p2,
       Data_Line_St * lst,
       bool auto_flush
)
{
     if (_dnext)
        _dnext->draw_seg(p1,p2,lst,auto_flush);

     _degw->set_line_style(lst);
     _degw->_inst_draw_seg
     (
          prto_window_geom(p1),
          prto_window_geom(p2)
     );
     if (auto_flush)
        _degw->_degd->auto_flush();
}


void Data_El_Geom_GWin::draw_circle
(
       Pt2dr centre,
       Pt2dr radius,
       Data_Line_St * lst,
       bool r_loc ,
       bool auto_flush
)
{
     if (_dnext)
        _dnext->draw_circle(centre,radius,lst,r_loc,auto_flush);

     _degw->set_line_style(lst);
     if (r_loc)
        radius =     prto_window_geom(radius)
                  -  prto_window_geom(Pt2dr(0.0,0.0));
     radius = Pt2dr(ElAbs(radius.x),ElAbs(radius.y));

     _degw->_inst_draw_circle(prto_window_geom(centre),radius);
     if (auto_flush)
        _degw->_degd->auto_flush();
}

void Data_El_Geom_GWin::draw_seg_cliped
(
       Pt2dr p0,
       Pt2dr p1,
       Data_Line_St * lst,
       Box2dr         box,
       bool auto_flush
)
{
   if (_dnext)
      _dnext->draw_seg_cliped(p0,p1,lst,box,auto_flush);

   Seg2d s = Seg2d(p0,p1).clip(box);
   if (! s.empty())
      draw_seg(s.p0(),s.p1(),lst,auto_flush);
}


void Data_El_Geom_GWin::draw_rect
(
        Pt2dr p1,
        Pt2dr p2,
        Data_Line_St * lst,
        bool auto_flush
)
{
     if (_dnext)
        _dnext->draw_rect(p1,p2,lst,auto_flush);

     p1 = prto_window_geom(p1);
     p2 = prto_window_geom(p2);
     pt_set_min_max(p1,p2);
     _degw->set_line_style(lst);
     _degw->_inst_draw_rectangle(p1,p2);
     if (auto_flush)
         _degw->_degd->auto_flush();
}


void Data_El_Geom_GWin::fill_rect
(
        Pt2dr p1,
        Pt2dr p2,
        Data_Fill_St * fst,
        bool auto_flush
)
{
     if (_dnext)
        _dnext->fill_rect(p1,p2,fst,auto_flush);

     p1 = prto_window_geom(p1);
     p2 = prto_window_geom(p2);
     pt_set_min_max(p1,p2);
     _degw->set_fill_style(fst);
     _degw->_inst_fill_rectangle(p1,p2);
     if (auto_flush)
         _degw->_degd->auto_flush();
}


void Data_El_Geom_GWin::draw_polyl
(
     const REAL *     x,
     const REAL *     y,
     INT              nb,
     Data_Line_St *   lst,
     bool             circ,
     bool             auto_flush
)
{
    if (_dnext)
       _dnext->draw_polyl(x,y,nb,lst,circ,auto_flush);

    _degw->set_line_style(lst);

    REAL xbuf[NB_BUF_DRAW_POLY];
    REAL ybuf[NB_BUF_DRAW_POLY];

    INT nb_buf = 0;

    for (int k=0; k<nb ; k++)
    {
        Pt2dr p = prto_window_geom(Pt2dr(x[k],y[k]));
        xbuf[nb_buf] = p.x;
        ybuf[nb_buf++] = p.y;

        if (nb_buf == NB_BUF_DRAW_POLY)
        {
            _degw->_inst_draw_polyl(xbuf,ybuf,nb_buf);
            nb_buf = 0;
             k--;
        }
    }
    if (nb_buf > 1)
       _degw->_inst_draw_polyl(xbuf,ybuf,nb_buf);

    if (circ && ( nb > 1))
        _degw->_inst_draw_seg
         (
              prto_window_geom(Pt2dr(x[nb-1],y[nb-1])),
              prto_window_geom(Pt2dr(x[0],y[0]))
         );

    if (auto_flush)
        _degw->_degd->auto_flush();
}


void Data_El_Geom_GWin::draw_polyl_cliped
(
         const REAL * x,const REAL *y,INT nb,Box2dr box, Data_Line_St * lst,
         bool circ,
         bool auto_flush
)
{
    if (_dnext)
       _dnext->draw_polyl_cliped(x,y,nb,box,lst,circ,auto_flush);

    _degw->set_line_style(lst);

    REAL x0[NB_BUF_DRAW_POLY];
    REAL y0[NB_BUF_DRAW_POLY];
    REAL x1[NB_BUF_DRAW_POLY];
    REAL y1[NB_BUF_DRAW_POLY];

    INT nb_buf = 0;
    for (INT k0 = 0; k0 <nb-1 ;)
    {
        INT k1 = k0;
        while
        (
                 (k1<nb-1)
              && (box.inside(Pt2dr(x[k1],y[k1])))
              && (box.inside(Pt2dr(x[k1+1],y[k1+1])))
        )
           k1 ++;

        if (k1 > k0)
        {
            draw_polyl(x+k0,y+k0,k1-k0+1,lst,false,false);
            k0 = k1;
        }
        else
        {
             Seg2d seg = Seg2d(x[k0],y[k0],x[k0+1],y[k0+1]).clip(box);

             if (! seg.empty())
             {
                  Pt2dr p0 = prto_window_geom(seg.p0());
                  Pt2dr p1 = prto_window_geom(seg.p1());

                  x0[nb_buf]   =    p0.x;
                  y0[nb_buf]   =    p0.y;
                  x1[nb_buf]   =    p1.x;
                  y1[nb_buf++] =    p1.y;
             }
             if (nb_buf == NB_BUF_DRAW_POLY)
             {
                _degw->_inst_draw_poly_segs(x0,y0,x1,y1,nb_buf);
                nb_buf = 0;
             }
             k0++;
        }
    }

    if (nb_buf)
       _degw->_inst_draw_poly_segs(x0,y0,x1,y1,nb_buf);

    if (circ && (nb > 1))
       draw_seg_cliped
       (
            Pt2dr(x[nb-1],y[nb-1]),
            Pt2dr(x[0],y[0]),
            lst,
            box,
            false
       );

    if (auto_flush)
        _degw->_degd->auto_flush();
}





/*****************************************************************/
/*                                                               */
/*                  Elise_Gra_Window                             */
/*                                                               */
/*****************************************************************/

Elise_Gra_Window::Elise_Gra_Window(Data_Elise_Gra_Win * degw) :
    PRC0(degw)
{
}

/*****************************************************************/
/*                                                               */
/*                  Data_Elise_Gra_Win                           */
/*                                                               */
/*****************************************************************/


void Data_Elise_Gra_Win::set_col(Data_Col_Pal * col)
{
     if (! col->eg_dcp(_degd->_last_cp))
     {
        _degd->_last_cp = *col;
        _inst_set_col(col);
     }
}

void Data_Elise_Gra_Win::set_line_style(Data_Line_St * dlst)
{
     set_active();
     _degd->set_line_witdh(dlst->witdh());
      set_col(dlst->dcp());
}

void Data_Elise_Gra_Win::set_fill_style(Data_Fill_St * dfst)
{
     set_active();
     set_col(dfst->dcp());
}


Data_Elise_Gra_Win::Data_Elise_Gra_Win
(
    Data_Elise_Gra_Disp * degd,
     Pt2di sz,
     Elise_Set_Of_Palette sop,
     bool INT_W
)  :
    _sz       (sz),
    _sop      (sop),
    _degd     (degd),
    _int_w    (INT_W),
    _num_w    (degd->_nb_graw++)
{
    _degd->reinit_cp();
}


void Data_Elise_Gra_Win::warn_graph(){};



void Data_Elise_Gra_Win::set_sop(Elise_Set_Of_Palette sop)
{
    _sop = sop;
    _degd->reinit_cp();
}



        /*   _inst_draw_rectangle, _inst_draw_polyl, _inst_draw_poly_segs

            The default values of some drawing functions. They are
          unoptimals because they just call back elementary primitives
          (like "_inst_draw_seg") and consequently do not take
          benefit of some special optimization (like X bufferization).

            Anyway they are useful because :

              o   there is sometime no much better to do (as for example
                  with bitmaped windows);

              o   they allow a faster portage to new graphical enviroment.

        */

void  Data_Elise_Gra_Win::_inst_draw_rectangle(Pt2dr p1,Pt2dr p2)
{
       p2 = Pt2dr(p2.x-1,p2.y-1);
       Pt2dr q1 (p1.x,p2.y);
       Pt2dr q2 (p2.x,p1.y);

      _inst_draw_seg(p1,q1);
      _inst_draw_seg(p1,q2);

      _inst_draw_seg(p2,q1);
      _inst_draw_seg(p2,q2);
}

void  Data_Elise_Gra_Win::_inst_draw_polyl
      (const REAL * x,const REAL *y,INT nb)
{
      for (int k=0 ; k<nb-1; k++)
          _inst_draw_seg(Pt2dr(x[k],y[k]),Pt2dr(x[k+1],y[k+1]));

}

void  Data_Elise_Gra_Win::_inst_draw_poly_segs
      (const REAL * x1,const REAL *y1,
       const REAL * x2,const REAL *y2,INT nb)
{
      for (int k=0 ; k<nb; k++)
          _inst_draw_seg(Pt2dr(x1[k],y1[k]),Pt2dr(x2[k],y2[k]));

}

void Data_Elise_Gra_Win::set_active()
{
}

void Data_Elise_Gra_Win::_inst_fixed_string(Pt2dr,const char *,bool){}


/*****************************************************************/
/*                                                               */
/*                  Data_Elise_Raster_W                          */
/*                                                               */
/*****************************************************************/

bool Data_Elise_Raster_W::adapt_vect()
{
    return true;
}

Data_Elise_Gra_Win * Data_Elise_Raster_W::dup_geo(Pt2dr,Pt2dr)
{
   return this;
}

Data_Elise_Raster_W::Data_Elise_Raster_W
     (  Data_Elise_Raster_D * derd,
        Pt2di sz,
        Elise_Set_Of_Palette sop
      ) :
      Data_Elise_Gra_Win(derd,sz,sop,true),
      _bli ((U_INT1 *) 0),
      _derd  (derd),
      _ddsop ( _derd->get_comp_pal(sop))

{
}

void Data_Elise_Raster_W::rset_sop(Elise_Set_Of_Palette sop)
{
    set_sop(sop);
    _ddsop = _derd->get_comp_pal(sop);
}

void Data_Elise_Raster_W::_inst_set_col(Data_Col_Pal * col)
{
     _derd->set_cur_coul(col->get_index_col(_ddsop));
}

void  Data_Elise_Raster_W::_inst_fill_rectangle(Pt2dr p0,Pt2dr p1)
{
     Pt2di q0 = round_ni(p0);
     Pt2di q1 = round_ni(p1);

     El_RW_Rectangle rect (q0.x,q0.y,q1.x-q0.x,q1.y-q0.y);
     rast_draw_big_pixels(&rect,1);
}

void  Data_Elise_Raster_W::rast_draw_col_pix
      (
           INT * col,
           El_RW_Point * pts,
           INT nb
      )
{
      for (INT i=0 ; i<nb ; i++)
      {
          _derd->set_cur_coul(col[i]);
          rast_draw_pixels(pts+i,1);
      }
}

void  Data_Elise_Raster_W::rast_draw_col_big_pixels
      (
           INT * col,
           El_RW_Rectangle * rect,
           INT nb
      )
{
      for (INT i=0 ; i<nb ; i++)
      {
          _derd->set_cur_coul(col[i]);
          rast_draw_big_pixels(rect+i,1);
      }
}

/******************************************************************/
/******************************************************************/
/******************************************************************/
/******************************************************************/

const Pt2dr Video_Win::_tr00(0.0,0.0);
const Pt2dr Video_Win::_sc11(1.0,1.0);




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
