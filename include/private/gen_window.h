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

/*


class Data_Elise_Raster_W : public Data_Elise_Gra_Win

class Data_Elise_Video_Win : public Data_Elise_Raster_W

class Data_El_Geom_GWin : public RC_Object
{
      class    Data_Elise_Gra_Win * _degw;
}
*/


#ifndef ELISE_PRIVATE_GEN_WINDOW
#define ELISE_PRIVATE_GEN_WINDOW


#if (ELISE_VIDEO_WIN== ELISE_VW_X11)

typedef struct El_RW_Point
{
   short x; short y;
   inline El_RW_Point(int X,int Y) : x (X), y(Y){}
} El_RW_Point;

typedef struct El_RW_Rectangle
{
    short x, y;
    unsigned short width, height;
   inline El_RW_Rectangle(int X,int Y,int W,int H) : x (X), y(Y), width(W),height(H){}
} El_RW_Rectangle;

#else  // ELISE_VIDEO_WIN== ELISE_VW_X11

         //      TO CHANGE WITH NT GRAPHISM
typedef struct El_RW_Point
{
   short x; short y;
   inline El_RW_Point(int X,int Y) : x (X), y(Y){}
} El_RW_Point;

typedef struct El_RW_Rectangle
{
    short x, y;
    unsigned short width, height;
   inline El_RW_Rectangle(int X,int Y,int W,int H) : x (X), y(Y), width(W),height(H){}
} El_RW_Rectangle;

#endif  // ELISE_VIDEO_WIN== ELISE_VW_X11







/*****************************************************************/
/*                  Graph window(coord manip)                    */
/*****************************************************************/

// As I have had some methaphysical doubts about the good
// sign, I introduce this constant to be abble to change later.

const INT SIGN_TO_USER_GEOM = 1;

inline REAL rto_window_geom(REAL p,REAL t,REAL s)
{
    return (s*(p-(SIGN_TO_USER_GEOM *t)));
}

inline REAL rto_user_geom(REAL p,REAL t,REAL s)
{
    return (p/s+(SIGN_TO_USER_GEOM * t));
}


inline INT ito_window_geom(REAL p,REAL t,REAL s)
{
    return round_ni(rto_window_geom(p,t,s));
}


inline INT deb_interval_user_to_window(INT a,REAL t,REAL s)
{
     return  round_up(rto_window_geom(a-0.5,t,s));
}

extern void interval_window_to_user(INT & a1,INT & a2,INT u1,INT u2,REAL t,REAL s);

extern void interval_user_to_window(INT & u1,INT & u2,INT a1,INT a2,REAL t,REAL s);


// compt tr,sc so that rto_window_geom(Ui,tr,sc) = Wi

extern void compute_tr_sc_from_exemple
            (REAL & tr,REAL & sc,REAL u1,REAL u2,REAL w1,REAL w2);

extern void compute_tr_from_exemple
            (REAL & tr,REAL sc,REAL u,REAL w);

inline Pt2dr rto_window_geom(Pt2dr p,Pt2dr t,Pt2dr s)
{
    return Pt2dr
           (
                rto_window_geom(p.x,t.x,s.x),
                rto_window_geom(p.y,t.y,s.y)
           );
}

Pt2dr rto_user_geom(Pt2dr p,Pt2dr t,Pt2dr s);


/*****************************************************************/
/*  display independant classes                                  */
/*****************************************************************/

class Data_Elise_Gra_Disp : public RC_Object
{
      friend class Data_Elise_Gra_Win;
      friend class Data_El_Geom_GWin;

      public :
            void reinit_cp();
            virtual void disp_flush() = 0;
            inline  void auto_flush()   {if(_auto_flush) disp_flush();}
            inline REAL line_witdh() { return _last_line_width;}


      protected :
            Data_Elise_Gra_Disp();
            void degd_flush();

      private :

            virtual void _inst_set_line_witdh(REAL) =0;
            inline void set_line_witdh(REAL lw)
            {
                   if (lw != _last_line_width)
                   {
                       _last_line_width = lw;
                       _inst_set_line_witdh(lw);
                   }
            }

            Data_Col_Pal      _last_cp;
            REAL              _last_line_width;
            bool              _auto_flush;  // when true all graphics are flushed immediately

            INT               _nb_graw;
            INT               _nb_geow;

};


const INT NB_BUF_DRAW_POLY = 100;
class Data_Elise_Gra_Win : public RC_Object
{

   friend class DE_GW_Not_Comp;
   friend class Data_El_Geom_GWin;
   friend class Graph_8_neigh_Win_Comp;
   friend class PS_Out_RLE_computed;
   friend class Ps_Multi_Filter;
   friend class PS_Pts_Not_RLE;

   public :
      inline Pt2di sz(){return _sz;}
      inline Elise_Set_Of_Palette sop(){return  _sop;}

       // for those window who have some special thing to do before
       // displaying can begin
      virtual  void warn_graph();


      inline  bool  int_w() const {return _int_w;}

      INT      num_w(){ return    _num_w;}  // num in display
        virtual  Data_Elise_Gra_Win * dup_geo(Pt2dr tr,Pt2dr sc) = 0;
   protected :

     
      Pt2di                    _sz;
      Elise_Set_Of_Palette    _sop;
      Data_Elise_Gra_Disp   *_degd;

      Data_Elise_Gra_Win(Data_Elise_Gra_Disp *,Pt2di sz,Elise_Set_Of_Palette sop,bool int_w);
      
      void set_sop(Elise_Set_Of_Palette);
      virtual void set_active();

   protected :

        virtual  bool adapt_vect() = 0; 

        void set_line_style(Data_Line_St *);
        void set_fill_style(Data_Fill_St *);
        void set_col(Data_Col_Pal * col);

        virtual  void _inst_set_col(Data_Col_Pal *) = 0;

        virtual   Output_Computed 
                  * rle_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & arg,
                       Data_Elise_Palette *,
                       bool  OnYDiff
                   ) = 0;


        virtual   Output_Computed 
                  * pint_cste_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & arg,
                       Data_Elise_Palette *,
                       INT        * cste
                   ) = 0;


        virtual   Output_Computed 
                  * pint_no_cste_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & arg,
                       Data_Elise_Palette *
                   ) = 0;

        const bool   _int_w; // is it a window to integer 
                             // coordinates (like Video) or 
                             // a window to real coordinate (like PS)

        INT          _num_w;  // num in display


             //  DRAWING PRIMITIVES

                   // elementary : without def values

        virtual void  _inst_draw_seg(Pt2dr,Pt2dr) = 0;

             

        virtual void  _inst_fill_rectangle(Pt2dr,Pt2dr) = 0;  

        virtual void  _inst_draw_circle(Pt2dr centre,Pt2dr ray /* ray.x != ray.y */) = 0;


                   // with def values (eventually call back elementary primitives)

        virtual void  _inst_draw_rectangle(Pt2dr,Pt2dr);  // def value : 4 draw seg

        virtual void  _inst_draw_polyl(const REAL * x,const REAL *y,INT nb);
                      // def value : draw nb-1 seg


        virtual void  _inst_draw_poly_segs
                       (const REAL * x1,const REAL *y1,
                        const REAL * x2,const REAL *y2,INT nb);
                      // def value : draw nb seg

        // def value do nothing (do only on X11)
        virtual void  _inst_fixed_string(Pt2dr,const char * name,bool draw_image);
};


class Elise_Gra_Window : public PRC0 
{
      friend class Data_El_Geom_GWin;
      private :
         Elise_Gra_Window(Data_Elise_Gra_Win *);
};

class Data_El_Geom_GWin : public RC_Object
{
   friend class El_Window;
   friend class Data_Param_Plot_1d;


   public :
         inline Pt2dr tr() const {return _tr;}
         inline Pt2dr sc() const {return _sc;}
         inline class    Data_Elise_Gra_Win * degw() const {return _degw;}
         inline Pt2di     sz() const { return _degw->sz();}

         inline void auto_flush() const {_degw->_degd->auto_flush();}

          Pt2di pito_window_geom(Pt2dr p) const;
          Pt2dr prto_window_geom(Pt2dr p) const;
          Pt2dr to_user_geom(Pt2di p) const;
          void  box_user_geom(Pt2di & res1,Pt2di & res2) const;
          INT   num_geow() {return _num_geow;}
          bool  handle_rastser_scale_negatif() { return !_adapt_vect;}

   private :

      Data_Elise_Gra_Disp * degd(){return _degw->_degd;}
      Data_El_Geom_GWin
      (
            Data_Elise_Gra_Win *,
            Pt2dr,
            Pt2dr,
            Data_El_Geom_GWin * next =0
      );

      Elise_Gra_Window            _egw;
      class    Data_Elise_Gra_Win * _degw;

      Data_El_Geom_GWin            *_dnext;
      El_Window                    _next;

      Pt2dr                  _tr;
      Pt2dr                  _sc;
      INT                    _num_geow;  // num in display
      bool                   _adapt_vect; 
      
      void draw_circle(Pt2dr,Pt2dr,Data_Line_St *,bool r_loc,bool auto_flush = true);

      void draw_seg(Pt2dr,Pt2dr,Data_Line_St *,bool auto_flush = true);
      void draw_seg_cliped(Pt2dr,Pt2dr,Data_Line_St *,Box2dr,bool auto_flush = true);
      void draw_rect(Pt2dr,Pt2dr,Data_Line_St *,bool auto_flush = true);
      void fill_rect(Pt2dr,Pt2dr,Data_Fill_St *,bool auto_flush = true);
      void fixed_string
           (Pt2dr,const char * name,Data_Col_Pal * col,bool draw_image,bool auto_flush = true);

      void draw_polyl
      (
         const REAL * x,const REAL *y,INT nb, Data_Line_St *,
         bool circ,
         bool auto_flush = true
      );
      void draw_polyl_cliped
      (
         const REAL * x,const REAL *y,INT nb,Box2dr b, Data_Line_St *,
         bool circ,
         bool auto_flush = true
      );

      Elise_Rect  box() const;
};

class GW_computed  : public Output_Computed
{
    protected :

       GW_computed(const Arg_Output_Comp &,const Data_El_Geom_GWin *,Data_Elise_Palette *);

       Pt2di  _sz;
       Pt2dr  _tr;
       Pt2dr  _sc;

       Pt2di  _up0;  // user coordinates limites
       Pt2di  _up1;

       const Data_El_Geom_GWin * _geom;
       virtual ~GW_computed();
};


/*****************************************************************/
/*     Raster Window (= X11, bitmap)                             */
/*****************************************************************/

class  Out_Ras_W_Comp : public  GW_computed
{
   public :

      Out_Ras_W_Comp
      (
           const Arg_Output_Comp &      ,
           const class Data_El_Geom_GWin    * ,
           class Data_Elise_Raster_W  * ,
           Data_Elise_Palette *       pal
      );

    protected :

      class Data_Elise_Raster_D   *    _derd;
      class Data_Elise_Raster_W   *    _derw;
      Data_Disp_Pallete           *     _ddp;
      Data_Elise_Palette          *     _dep;
      bool                            _first;
      INT                             _byte_pp; // _derd->_byte_pp
      virtual ~Out_Ras_W_Comp();

};

class Data_Elise_Raster_W : public Data_Elise_Gra_Win
{
   friend class Out_Ras_W_Comp;
   friend class RLE_Out_Ras_W_Comp;
   friend class PInt_Cste_Out_Ras_W_Comp;
   friend class PInt_NoC_TrueC_ORW_Comp;
   friend class Video_Win;

      protected :
         Data_Elise_Raster_W(class Data_Elise_Raster_D *,
                             Pt2di,
                             Elise_Set_Of_Palette);

         void rset_sop(Elise_Set_Of_Palette);

         virtual void  flush_bli(INT x0,INT x1,INT y) = 0;
         virtual void  rast_draw_pixels(El_RW_Point *,INT nb) = 0;
         virtual void  rast_draw_big_pixels(El_RW_Rectangle *,INT nb) = 0;

         virtual void  rast_draw_col_pix(INT * col,El_RW_Point *,INT nb);
         virtual void  rast_draw_col_big_pixels(INT *,El_RW_Rectangle *,INT nb);

         virtual void  _inst_fill_rectangle(Pt2dr,Pt2dr);  

         virtual  void _inst_set_col(Data_Col_Pal *);
         virtual   Output_Computed 
                  * rle_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & arg,
                       Data_Elise_Palette *,
                       bool  OnYDiff
                   );

        virtual   Output_Computed 
                  * pint_cste_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & arg,
                       Data_Elise_Palette *,
                       INT        * cste
                   );

        virtual   Output_Computed 
                  * pint_no_cste_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & arg,
                       Data_Elise_Palette *
                   );
         U_INT1 *                   _bli;  // buffer line image 
         Data_Elise_Raster_D *     _derd;
         Data_Disp_Set_Of_Pal *   _ddsop;

         virtual  bool adapt_vect() ; 
         virtual  Data_Elise_Gra_Win * dup_geo(Pt2dr tr,Pt2dr sc);
};


class Data_Elise_Raster_D : public Data_Elise_Gra_Disp
{

      public :


          void init_mode(INT depth);

          Elise_mode_raster_color _cmod;
          INT        _byte_pp;  // byte per pixel

          INT        _r_nbb;
          INT        _g_nbb;
          INT        _b_nbb;

          // only used for true_16_col mode

          INT        _r_mult;
          INT        _g_mult;
          INT        _b_mult;

          // only used for true_16_col or true_24_col mode
          INT        _r_shift;
          INT        _g_shift;
          INT        _b_shift;
	  //
          // only used for true_16_col or true_24_col mode
          INT        _r_mask;
          INT        _g_mask;
          INT        _b_mask;


          // only used for  true_24_col mode (value 0,1 or 2)

          INT        _r_ind;
          INT        _g_ind;
          INT        _b_ind;

          inline  INT rgb_to_16(INT r,INT g,INT b)
          {
                  return
                              ((r * _r_mult) / 256) << _r_shift
                           |  ((g * _g_mult) / 256) << _g_shift
                           |  ((b * _b_mult) / 256) << _b_shift ;
          };

          inline  INT rgb_to_24(INT r,INT g,INT b)
          {
                  return
                              (r  << _r_shift)
                           |  (g  << _g_shift)
                           |  (b  << _b_shift) ;
          };



          U_INT1 * alloc_line_buf(INT nb);

          inline INT cur_coul() { return _cur_coul;}
          virtual void set_cur_coul(INT coul) = 0;
          Data_Disp_Set_Of_Pal * get_comp_pal(Elise_Set_Of_Palette);

          void read_rgb_line(U_INT1 * r,U_INT1 * g,U_INT1 *b,INT nb,U_INT1 * Xim); 
          void write_rgb_line(U_INT1 * Xim,INT nb,U_INT1 * r,U_INT1 * g,U_INT1 *b); 

      protected :

          Data_Elise_Raster_D(const char * name,INT nb_pix_ind_max);
          virtual    ~Data_Elise_Raster_D();

          virtual void augmente_pixel_index(INT nb) = 0;

          L_Disp_SOP   _ldsop;

          INT             _nb_pix_ind_max;
          unsigned long   * _pix_ind;

          INT             _nb_pix_ind;
          INT _cur_coul;

          INT          _depth;
};





#endif  // !  ELISE_PRIVATE_GEN_WINDOW

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
