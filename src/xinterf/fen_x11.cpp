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


#if (ELISE_X11)
#include "private/video_win.h"

#define X11_2PI 23040  // 360 * 64

extern "C" 
{
  INT always_true (Display*, XEvent*,XPointer)
  {
    return(True);
  }
};

class X11ValideEvent
{
   public :
      virtual bool ok(XEvent &) { return true;}
  virtual ~X11ValideEvent() {}
};


class X11ValidButonThisWindow : public X11ValideEvent
{
      Window  _w;

   public :
    
       X11ValidButonThisWindow(Window w) : _w (w) {}
       virtual bool ok(XEvent & e) { return e.xbutton.window == _w;}
};



/*
*/



class Data_El_Video_Display : public Data_Elise_Raster_D
{
     friend class Video_Display;
     friend class Data_Elise_Video_Win;

     public :

          struct LW
          {
              
                LW *                    _n;
                Data_Elise_Video_Win *  _w;
          };
          void add_w(Data_Elise_Video_Win *);
          Data_Elise_Video_Win *  get_w(Window,bool svp = false);


          void init_xi(XImage *  xi);
          XImage  _xi0;
          bool    _xi_first;

          virtual ~Data_El_Video_Display();
          virtual void disp_flush();

          Data_El_Video_Display(const char * name,void * Id);


          virtual void augmente_pixel_index(INT nb);
          void  load_palette(Elise_Set_Of_Palette);

          // identic (set_cur_coul calls vset_cur_coul) 
          void set_cur_coul(INT coul);            // virtual 
          inline void vset_cur_coul(INT coul);    // inline

          LW *  _lw;
          Display *    _disp;
          Window       _root_w;
          Pt2di        mSzEcr;
          int          _num_ecr;
          Screen *     _screen;
          GC           _gc;
          Visual *     _visual;
          INT          _depth;
          XVisualInfo  _xvi;
          Colormap     _cmap;

          XColor     * _lut;
          bool         _pal_load;
          bool         _warn_no_pal_load;

          virtual void warn_graph();
          virtual void _inst_set_line_witdh(REAL);

          void  init_bits_acc
                (
                            INT & aMemoMask,
                            INT & shift,
                            INT & mult,
                            INT & ind,
                            INT & nbb,
                            INT   mask
                );

           Clik clik
		(
			X11ValideEvent & press,
			bool             wait_press,
			X11ValideEvent & release,
			bool             wait_release
		);

           void empty_events();


           XFontStruct *  _FixedFont;
           XFontStruct *  _CurFont;


           void SetCurFonte(XFontStruct *);
           XFontStruct * SetFixedFonte();
		   void grab(Data_Elise_Video_Win * wel,Grab_Untill_Realeased &);


    private :
         Data_El_Video_Display(const Data_El_Video_Display &);
};



class DataElXim : public RC_Object
{
     public :

         void * operator new    (size_t sz);
         void operator delete   (void * ptr);
	
         INT tx() const {return _sz.x;}
         INT ty() const {return _sz.y;}
         typedef U_INT1 tValueElem;
         tValueElem * adr_data(Pt2di pt) {return _data+adr_bxi(pt);}



    Pt2di sz() const{return _sz;}
	DataElXim(Data_Elise_Video_Win *,Pt2di sz);
	void init(Data_Elise_Video_Win *,Pt2di sz);

	DataElXim(Data_Elise_Video_Win *,Pt2di sz,Fonc_Num,Data_Elise_Palette *);
	void init(Data_Elise_Video_Win *w,Fonc_Num,Data_Elise_Palette * dep);

	Data_Elise_Video_Win * _w;
      	XImage *              _xim;
      	INT                   _bbl; // byte per line
      	INT                   _xofs; // ofs pix 0
      	INT                   _szpix; // byte per pixel
	U_INT1 *              _data;
	Pt2di                 _sz;
	void load();
	void load(Pt2di p0Src,Pt2di p0Dest,Pt2di sz);

        void write_image_per
             (
	         Pt2di p0src, 
	         Pt2di p0dest,
	         Pt2di sz, 
                 DataElXim &
             );

        void write_image_per
             (
	         Pt2di p0src, 
	         Pt2di p0dest,
	         Pt2di sz
             );

        void fill_with_el_image 
             (
                   Pt2di                    p0src,
                   Pt2di                    p0dest,
                   Pt2di                    sz,
                   std::vector<Im2D_INT4> & Images,
                   Data_Elise_Palette *     dep
            );

            void read_write_ElIm
            (
	          Pt2di       aP0El, 
	          Pt2di       aP0Xim,
	          Pt2di       aSz, 
                  Im2D_U_INT1 anImR,
                  Im2D_U_INT1 anImG,
                  Im2D_U_INT1 anImB,
                  bool        ModeRead
            );



      inline INT adr_bxi(Pt2di p);
};



const int NB_COUL_X11_LUT = 256;



class Data_Elise_Video_Win : public Data_Elise_Raster_W
{

   friend class Video_Win;
   friend class Out_Ras_W_Comp;
   friend class RLE_Out_Ras_W_Comp;
   friend class Data_El_Video_Display;
   friend class DataElXim;
   friend class InitDataElXim;


   public :
       void set_cl_coord(Pt2dr,Pt2dr);
       Pt2dr to_user_geom(Pt2dr p);
       void raise();
       void lower();
       void move_to(const Pt2di &);
       void move_translate(const Pt2di &);
   private :

		void grab( Grab_Untill_Realeased &);

      Data_Elise_Video_Win
      (
             Video_Display              ,
             Elise_Set_Of_Palette       ,
             Pt2di                    p0,
             Pt2di                    sz,
             INT                      border_witdh,
             Window *                 aX11IdPtr,
             Data_Elise_Video_Win *   aSoeur = 0,
             Video_Win::ePosRel       aPosRel = Video_Win::eDroiteH
      );

      Data_Elise_Video_Win vdup(Data_Elise_Video_Win);

      void set_title(const char *);
      void clear();

      void disp_flush() {_devd->disp_flush();}


      virtual void  flush_bli(INT x0,INT x1,INT y);
      virtual void  rast_draw_pixels(El_RW_Point *,INT) ;
      virtual void  rast_draw_big_pixels(El_RW_Rectangle *,INT nb);
      virtual void  rast_draw_col_pix(INT *,El_RW_Point *,INT) ;
      virtual void  rast_draw_col_big_pixels(INT *,El_RW_Rectangle *,INT nb);

      inline DataElXim * alloc_big_image();
      XImage *  AllocXim(Pt2di sz);
      inline void load_pal_im (Data_Elise_Palette * dep);
      void write_image 
           (
	      Pt2di p0src, 
	      Pt2di p0dest,
	      Pt2di sz, 
	      INT *** Im, 
              Data_Elise_Palette * dep,
              DataElXim * = 0
           );


      void load_image(Pt2di p0src,Pt2di p0dest,Pt2di sz,DataElXim * =0);
      void translate(Pt2di tr);
      void image_translate(Pt2di tr,DataElXim * = 0);

      EliseStdImageInteractor  *  mInteractor;
      Video_Display               _vd;
      Data_El_Video_Display *     _devd;

      Window                   _w;
      Window                   _mother;
      XImage *                 _xi;

      DataElXim  *             _DBXI;
      Data_Elise_Palette *     _dep_wi;      
      Data_Disp_Pallete  *     _ddp_wi;      
      U_INT2 *                 _lut;


      Pt2dr                    _tr;
      Pt2dr                    _sc;
      Pt2di                    _p0;
      Pt2di *                  _SzMere;
      Pt2di                    _p1;
      INT                      _bord;
      Pt2di PBrd() const {return Pt2di(_bord,_bord);}
      Pt2di EnCombr() const {return  PBrd()*2+_sz;}

    //------------------------------------
    //   X11 Window creation
    //------------------------------------

     static Window  simple_w
                    (
                         Data_El_Video_Display *,
                         Pt2di               p0,
                         Pt2di               sz,
                         Window              mother,
                         INT                 border_witdh
                    );

     static Window  very_simple_w
                    (
                         Data_El_Video_Display *,
                         Pt2di               p0,
                         Pt2di               sz,
                         INT                 border_witdh,
                         Window &            mother
                    );

      virtual void warn_graph();
      virtual void _inst_draw_seg(Pt2dr,Pt2dr);
      virtual void _inst_draw_circle(Pt2dr,Pt2dr);
      virtual void _inst_draw_rectangle(Pt2dr,Pt2dr);
      void  _inst_draw_polyl (const REAL * x,const REAL *y,INT nb);
      virtual void  _inst_draw_poly_segs
                     (const REAL * x1,const REAL *y1,
                      const REAL * x2,const REAL *y2,INT nb);


      void _inst_fixed_string(Pt2dr pt,const char * name,bool draw_image);
      std::string InstGetString
                  (
                        const Pt2dr & aPt,
                        Data_Col_Pal *      colStr,
                        Data_Col_Pal *      colEras,
                        const std::string & aStr0
                  );

       Pt2di InstSizeFixedString(const std::string aStr);

};



/******************************************************************/
/*                                                                */
/*        X11 window creation  (Data_Elise_Video_Win   )          */
/*                                                                */
/******************************************************************/

                 //    X11 window creation 

Window   Data_Elise_Video_Win::simple_w
         (
                Data_El_Video_Display * ded,
                Pt2di               p0,
                Pt2di               sz,
                Window              mother,
                INT                 border_witdh
         )
{
   Window  w;
   XEvent glob_event;

  XSetWindowAttributes attr;

  attr.colormap =  ded->_cmap;
  attr.background_pixel = 0;
  int val_mask = CWColormap | CWBackPixel;

  w = XCreateWindow
  (
           ded->_disp,
           mother,
           p0.x,p0.y,
           sz.x,sz.y,
           border_witdh, 
           ded->_xvi.depth,
           InputOutput,
           ded->_xvi.visual,
           val_mask,
           &attr
  );


  XMapWindow(ded->_disp,w);
  XSelectInput
  (
         ded->_disp,
         w,
            ExposureMask
         |  ButtonPressMask
         |  ButtonReleaseMask
         |  KeyPressMask
  );

  if (mother == ded->_root_w)
  {
     XWindowEvent(ded->_disp,w,ExposureMask,&glob_event);
     XMoveWindow(ded->_disp,w,p0.x,p0.y);
     XMapWindow(ded->_disp,w);

  }
  {
      XSetWindowAttributes att;

      att.backing_store = Always;
      XChangeWindowAttributes(ded->_disp,w,CWBackingStore,&att);

  }

  ded->disp_flush();
  return w;
}


Window   Data_Elise_Video_Win::very_simple_w
         (
                Data_El_Video_Display * ded,
                Pt2di               p0,
                Pt2di               sz,
                INT                 border_witdh,
                Window              &mother
         )
{

   

    mother = simple_w(ded,p0,sz+Pt2di(2,2)*border_witdh,ded->_root_w,border_witdh);
    Window w  = simple_w(ded,Pt2di(0,0),sz,mother,border_witdh);

    return w;
}

void Data_Elise_Video_Win::raise()
{
    XRaiseWindow(_devd->_disp,_w);
    disp_flush();
}
void Data_Elise_Video_Win::lower()
{
    XLowerWindow(_devd->_disp,_w);
    disp_flush();
}

void Data_Elise_Video_Win::move_to(const Pt2di & aP)
{
   _p0 = aP;
    XMoveWindow(_devd->_disp,_w,aP.x,aP.y);
    XMapWindow(_devd->_disp,_w);
    disp_flush();
}

void Data_Elise_Video_Win::move_translate(const Pt2di & aP)
{
   move_to(aP+_p0);
}


/******************************************************************/
/*                                                                */
/*        Data_Elise_Video_Win                                    */
/*                                                                */
/******************************************************************/

void show_mask(char * mes,INT mask,INT nb)
{
     cout << mes ;
     for (INT b = nb-1 ; b>=0 ; b--)
         cout << ((mask & (1<<b)) ? "+" : "-");
     cout << "\n" ;
}

XImage *  Data_Elise_Video_Win::AllocXim(Pt2di sz)
{
    
    //std::cout << "On passe ici"<<std::endl;
       return  XGetImage
               (
                   _devd->_disp,
                /*_w*/XDefaultRootWindow(_devd->_disp),
                   0,0,
                   sz.x,sz.y,
                   AllPlanes,ZPixmap
               );            
/*
	INT Lx = (_sz.x *_devd->_depth + 7) / 8;
	Lx = ((Lx+3) /4) * 4;

 	return  XCreateImage
		(
                     _devd->_disp,
                     _devd->_visual,
                     _devd->_depth,
                     ZPixmap,
                     0,
		     new char [ty*Lx],
		     _sz.x,
		     ty,
		     32,
		     Lx
		);
*/
}

void  * DataElXim::operator new(size_t sz)
{
    void * ptr = Elise_Calloc(1,sz);
    return   ptr;
}
 
 
void   DataElXim::operator delete (void * ptr)
{
    Elise_Free(ptr);
}                   


void DataElXim::init(Data_Elise_Video_Win * w,Pt2di sz)
{

    _w = w;

    if ((sz.x > w->_sz.x) || (sz.y>w->_sz.y))
    {
        std::cout << "DataElXim::init " << sz << " " << w->_sz << "\n";
        ELISE_ASSERT
        (
              (sz.x <= w->_sz.x) && (sz.y<=w->_sz.y),
              "ElXim size to big for Window "
        );
    }
    _xim =  w->AllocXim(sz);
    ELISE_ASSERT(_xim,"Can't alloc Ximage");
    _bbl    = _xim->bytes_per_line; 
    _xofs   = _xim->xoffset; 
    _szpix  = w->_devd->_byte_pp; 
    _data   = (U_INT1 *) _xim->data;
    _sz     = sz;
}
DataElXim::DataElXim(Data_Elise_Video_Win * w,Pt2di sz) 
{
   init(w,sz);
}

DataElXim::DataElXim
(
     Data_Elise_Video_Win * w,
     Pt2di sz,
     Fonc_Num f,
     Data_Elise_Palette * dep
) 
{
   init(w,sz);
   init(w,f,dep);
}

class InitDataElXim : public Simple_OPBuf1<INT,INT>
{
   public :
     	void  calc_buf (INT ** output,INT *** input); 
	InitDataElXim
        (
		DataElXim * 		XIM,
	 	Data_Elise_Palette * 	DEP,
		Data_Elise_Video_Win * 	W
	)   :
	    _xim (XIM),
	    _dep (DEP),
	    _w	 (W)
	{
	}

	DataElXim *                _xim;
	Data_Elise_Palette *       _dep;
        Data_Elise_Video_Win *     _w;
};

typedef int * IntPtr_fen_x11; // For fucking Workshop 5.0

void InitDataElXim::calc_buf(INT ** output,INT *** input)
{
    static INT ** l =  new IntPtr_fen_x11  [10];

    for (INT d=0; d<dim_in();  d++)
        l[d]  = &(input[d][0][0]);
    _w->write_image
    (
       Pt2di(0,0),
       Pt2di(0,ycur()),
       Pt2di(tx(),1),
       &l,
       _dep,
       _xim
    );
}

void DataElXim::init
     (
         Data_Elise_Video_Win * w,
         Fonc_Num f,
         Data_Elise_Palette * dep
     )
{
     ELISE_COPY
     (
        rectangle(Pt2di(0,0),_sz),
        create_op_buf_simple_tpl
        (
              new InitDataElXim(this,dep,w),
              0,
              f,
              1,
              Box2di(Pt2di(0,0),Pt2di(0,0))
        ),
	Output::onul()
     );
}

void DataElXim::load()
{
      _w->load_image(Pt2di(0,0),Pt2di(0,0),_sz,this);
}

void DataElXim::load(Pt2di p0Src,Pt2di p0Dest,Pt2di sz)
{
      _w->load_image(p0Src,p0Dest,sz,this);
}



ElXim::ElXim(Video_Win w,Pt2di sz,Fonc_Num f,Elise_Palette pal) :
	PRC0(new  DataElXim(w.devw(),sz,f,pal.dep()))
{
}

ElXim::ElXim(Video_Win w,Pt2di sz) :
	PRC0(new  DataElXim(w.devw(),sz))
{
}

ElXim::ElXim(DataElXim * aDTXI) :
	PRC0(aDTXI)
{
}

DataElXim * ElXim::dex()
{
     return SAFE_DYNC (DataElXim *,_ptr);
}
void ElXim::load()
{
	dex()->load();
}

void ElXim::write_image_per(Pt2di   p0src,Pt2di  p0dest,Pt2di  sz)
{
     dex()->write_image_per(p0src,p0dest,sz);
}

void ElXim::load(Pt2di   p0src,Pt2di  p0dest,Pt2di  sz)
{
	dex()->load(p0src,p0dest,sz);
}

void ElXim::fill_with_el_image
    (
          Pt2di p0src,
          Pt2di p0dest,
          Pt2di sz,
          std::vector<Im2D_INT4> & Images,
          Elise_Palette  aPal
    )
{
    dex()->fill_with_el_image
    (
        p0src,
        p0dest,
        sz,
        Images,
        aPal.dep()
    );
}

void ElXim::read_in_el_image
             (
                  Pt2di       aP0Src,
                  Pt2di       aP0Dest,
                  Pt2di       aSz,
                  Im2D_U_INT1 anImR,
                  Im2D_U_INT1 anImG,
                  Im2D_U_INT1 anImB
             )
{
     dex()->read_write_ElIm
     (
        aP0Dest,aP0Src,aSz,
        anImR,anImG,anImB,
        true
     );
}

void ElXim::fill_with_el_image
             (
                  Pt2di       aP0Src,
                  Pt2di       aP0Dest,
                  Pt2di       aSz,
                  Im2D_U_INT1 anImR,
                  Im2D_U_INT1 anImG,
                  Im2D_U_INT1 anImB
             )
{
     dex()->read_write_ElIm
     (
        aP0Src,aP0Dest,aSz,
        anImR,anImG,anImB,
        false
     );
}




DataElXim * Data_Elise_Video_Win::alloc_big_image()
{
    if (! _DBXI)
    {
       _DBXI = new  DataElXim  (this,_sz);
    }

//std::cout << "ABI " << _DBXI << "\n";
    return _DBXI;
}


void Data_Elise_Video_Win::load_pal_im
     (
        Data_Elise_Palette * dep
     )
{

     if (dep != _dep_wi)
     {
         _dep_wi = dep;
	 _ddp_wi = _ddsop->ddp_of_dep(dep);
	 _lut = _ddp_wi->lut_compr();
     }

}

void Data_Elise_Video_Win::write_image 
     (
	Pt2di p0src,
	Pt2di p0dest,
	Pt2di sz,
	INT *** Im,  // [y][ch][x]
	Data_Elise_Palette * dep,
	DataElXim *          xim
     )
{
   if (! xim)
   {
      alloc_big_image();
      xim = _DBXI;
   }
   AdaptParamCopyTrans(p0src,p0dest,sz,Pt2di(1<<30,1<<30),xim->_sz);
   load_pal_im(dep);

   // p0dest.SetSup(Pt2di(0,0));
   // sz.SetInf(_sz-p0dest);


   if ((sz.x >0) && (_sz.y>0))
   {
   	for (INT dy =0 ; dy<sz.y ; dy++)
	{
		dep->lutage
		(
			_devd,
			xim->_data +  xim->_xofs + (p0dest.y+dy) * xim->_bbl,
			p0dest.x, p0dest.x+sz.x,
			_lut,
			Im[p0src.y+dy],p0src.x
		);
	}
   }
}


void DataElXim::fill_with_el_image 
     (
            Pt2di                    p0src,
            Pt2di                    p0dest,
            Pt2di                    sz,
            std::vector<Im2D_INT4> & Images,
            Data_Elise_Palette *     dep
     )
{
   static const INT NbChMax  = 10;
   INT NbCh = dep->dim_pal();
   INT * Im[NbChMax];  
   ELISE_ASSERT
   (
       (NbCh<=NbChMax) && (NbCh<=(INT)Images.size()),
       "Bad dim in DataElXim::write_image"
   );

   AdaptParamCopyTrans(p0src,p0dest,sz,Images[0].sz(),this->_sz);

   INT ** I = Im;

   for (INT y=0 ; y<sz.y ; y++)
   {
        INT ySrc = y+p0src.y;
        for (INT iCh=0; iCh<NbCh; iCh++)
            I[iCh] = Images[iCh].data()[ySrc];
        _w->write_image 
        (
              Pt2di(p0src.x,0), 
              Pt2di(p0dest.x,p0dest.y+y), 
              Pt2di(sz.x,1), 
              &I, 
              dep, 
              this
        );
   }
}



INT DataElXim::adr_bxi(Pt2di p)
{
    return _xofs + _szpix * p.x +_bbl * p.y;
}



void Data_Elise_Video_Win::load_image 
     (
	Pt2di p0src,
	Pt2di p0dest,
	Pt2di sz,
	DataElXim * xim
     )
{
      if (! xim)
      {
         alloc_big_image();
	     xim = _DBXI;
      }
     AdaptParamCopyTrans(p0src,p0dest,sz,xim->_sz,this->_sz);
     if ((sz.x<=0)||(sz.y<=0))
        return;

      warn_graph();

      XPutImage
      (   _devd->_disp,
	  _w, 
	 _devd->_gc, 
	  xim->_xim,
          p0src.x,p0src.y,
          p0dest.x,p0dest.y,
          sz.x,sz.y
      );
      disp_flush();
}

void DataElXim::write_image_per
     (
	Pt2di       p0src, 
	Pt2di       p0dest,
	Pt2di       sz
     )
{
    _w->alloc_big_image();
    _w->_DBXI->write_image_per(p0src,p0dest,sz,*this);
}


void DataElXim::read_write_ElIm
     (
	  Pt2di       aP0ElIm, 
	  Pt2di       aP0Xim,
	  Pt2di       aSz, 
          Im2D_U_INT1 anImR,
          Im2D_U_INT1 anImG,
          Im2D_U_INT1 anImB,
          bool        ModeRead
     )
{
     AdaptParamCopyTrans
     (
        aP0ElIm,aP0Xim,aSz,
        Inf(anImR.sz(),Inf(anImG.sz(),anImB.sz())),
        sz()
     );

     U_INT1 ** aDataR = anImR.data();
     U_INT1 ** aDataG = anImG.data();
     U_INT1 ** aDataB = anImB.data();

     for (INT y = 0 ; y<aSz.y; y++)
     {
        U_INT1 * r = aDataR[y+aP0ElIm.y]+aP0ElIm.x;
        U_INT1 * g = aDataG[y+aP0ElIm.y]+aP0ElIm.x;
        U_INT1 * b = aDataB[y+aP0ElIm.y]+aP0ElIm.x;

        U_INT1 * Xim = adr_data(Pt2di(aP0Xim.x,y+aP0Xim.y));

        if (ModeRead)
             _w->_devd->read_rgb_line(r,g,b,aSz.x,Xim);
        else
             _w->_devd->write_rgb_line(Xim,aSz.x,r,g,b);

     }
}




void DataElXim::write_image_per
     (
	  Pt2di p0src, 
	  Pt2di p0dest,
	  Pt2di sz, 
          DataElXim & Im2
     )
{
    p0dest = Inf(_sz,Sup(p0dest,Pt2di(0,0)));
    sz = Inf(sz,_sz-p0dest);
    if ((sz.x<=0) || (sz.y<=0)) 
       return;
    p0src = Pt2di
             (
                mod(p0src.x,Im2._sz.x),
                mod(p0src.y,Im2._sz.y)
             );

    INT X0src = p0src.x;
    INT X0Dest = p0dest.x;
    INT SZX_residuel = sz.x;
    while (SZX_residuel >0)
    {
	INT szx = ElMin(SZX_residuel,Im2._sz.x-X0src);

        for (INT y =0; y<sz.y ; y++)
        {
	     INT id0 = this->adr_bxi(Pt2di(X0Dest,y+p0dest.y));
	     INT id1 = this->adr_bxi(Pt2di(X0Dest+szx,y+p0dest.y));
	     INT is0 = Im2.adr_bxi(Pt2di(X0src,mod(y+p0src.y,Im2._sz.y)));
	     memcpy(_data+id0,Im2._data+is0, id1-id0);
	}

        SZX_residuel -= szx;
        X0src = 0;
        X0Dest += szx;
    }
}




void Data_Elise_Video_Win::image_translate(Pt2di tr,DataElXim * xim)
{
    if (! xim)
    {
       alloc_big_image();
       xim = _DBXI;
    }

    AutoTranslateData(tr,*xim);
}

void Data_Elise_Video_Win::translate(Pt2di tr)
{
	INT sx =ElMax(-tr.x,0);
	INT sy =ElMax(-tr.y,0);
	INT dx =ElMax(tr.x,0);
	INT dy =ElMax(tr.y,0);

	INT szx = _sz.x-ElAbs(tr.x);
	INT szy = _sz.y-ElAbs(tr.y);
        if ((szx <=0) || (szy<0))
           return;


	XCopyArea
	(
         	_devd->_disp,
         	_w,
         	_w,
         	_devd->_gc,
		sx,sy,
		szx,szy,
		dx,dy
	);
      disp_flush();
}


/*
*/


Data_Elise_Video_Win::Data_Elise_Video_Win
(
         Video_Display            vd,
         Elise_Set_Of_Palette     sop ,
         Pt2di                    p0,
         Pt2di                    sz,
         INT                      border_witdh,
         Window *                 aX11IdPtr,
         Data_Elise_Video_Win *   aSoeur ,
         Video_Win::ePosRel       aPosRel
)  :
   Data_Elise_Raster_W(vd.devd(),sz,sop),
   mInteractor (0),
   _vd (vd),
   _devd (vd.devd()),
   _DBXI (0),
   _dep_wi (0),
   _tr  (0.0,0.0),
   _sc  (1.0,1.0),
   _bord (border_witdh)
{

    if (aX11IdPtr)
    {
       _p0 = Pt2di(0,0); 
       _w = *aX11IdPtr;
       _SzMere = new Pt2di(EnCombr());
    }
    else if (aSoeur == 0)
    {
       _w = very_simple_w(_devd,p0,sz,border_witdh,_mother);
       _p0 = Pt2di(0,0); 
       _SzMere = new Pt2di(EnCombr());
    }
    else
    {
       Pt2di PF0(0,0);
       Pt2di PF1(0,0);

       if (aPosRel == Video_Win::eDroiteH)
       {
            PF0 = Pt2di(1,0);
            PF1 = Pt2di(0,0);
       }
       else if (aPosRel == Video_Win::eBasG)
       {
            PF0 = Pt2di(0,1);
            PF1 = Pt2di(0,0);
       }
       else if (aPosRel == Video_Win::eSamePos)
       {
            PF0 = Pt2di(0,0);
            PF1 = Pt2di(0,0);
       }


       _p0 =    aSoeur->_p0
             +  PF0.mcbyc(aSoeur->EnCombr())
             -  PF1.mcbyc(EnCombr());

       _SzMere = aSoeur->_SzMere;
       _SzMere->SetSup(_p0 + EnCombr());
       _mother = aSoeur->_mother;

//-OK- std::cout << "xazerty   1 \n"; getchar();
       if (1)
       {
          XResizeWindow(_devd->_disp,_mother,_SzMere->x,_SzMere->y);
          XMapWindow(_devd->_disp,_mother);
          disp_flush();

          XSetWindowAttributes attr;

          attr.colormap =  _devd->_cmap;
          attr.background_pixel = 0;
          attr.backing_store = Always;
          attr.save_under = True;
          int val_mask = CWColormap | CWBackPixel | CWBackingStore | CWSaveUnder;

          _w = XCreateWindow
          (
                   _devd->_disp,
                   _mother,
                   _p0.x,_p0.y,
                   sz.x,sz.y,
                   border_witdh, 
                   _devd->_xvi.depth,
                   InputOutput,
                   _devd->_xvi.visual,
                   val_mask,
                   &attr
          );
          XMapWindow(_devd->_disp,_w);
          SleepProcess(0.2);
  // std::cout << "xazerty   9 \n"; getchar();
          XSelectInput
          (
                 _devd->_disp,
                 _w,
                    ExposureMask
                 |  ButtonPressMask
                 |  ButtonReleaseMask
                 |  KeyPressMask
          );
//-OK- std::cout << "xazerty   10 \n"; getchar();
       }
       else
          _w = very_simple_w(_devd,_p0,sz,border_witdh,_mother);

    }

//-OK-   std::cout << "xazerty   10 \n"; getchar();

    // Modif GM: je n'ai rien compris au code, mais je constate sous MacOS
    // une erreur :
    /*
     X Error of failed request:  BadMatch (invalid parameter attributes)
     Major opcode of failed request:  73 (X_GetImage)
     Serial number of failed request:  20
     Current serial number in output stream:  20
     */ 
    // lorsque l'on passe dans le if(1) ci-dessus
    // empiriquement, l'erreur ne se produit pas si on utilise
    // XDefaultRootWindow(_devd->_disp) plutot que _w
    //std::cout << "On passe la"<<std::endl;
    _xi = XGetImage
    (
     _devd->_disp,
     /*_w*/XDefaultRootWindow(_devd->_disp),
     0,0,
     sz.x,1,
     AllPlanes,ZPixmap
     );
    
//-OK-  std::cout << "xazerty   11 \n"; getchar();
    _bli = (U_INT1 *)_xi->data;
//-OK-  std::cout << "xazerty   12 \n"; getchar();
    _devd->init_xi(_xi);
//-OK- std::cout << "xazerty   13 \n"; getchar();
    _devd->add_w(this);
//-OK- std::cout << "xazerty   14 \n"; getchar();

    if (aX11IdPtr ==0)
       set_title("Fenetre ELISE");

//-OK- std::cout << "xazerty   15 \n"; getchar();
}

void Data_Elise_Video_Win::set_title(const char * name)
{
  XSizeHints hints;
  hints.flags = 0;

  XSetStandardProperties
  (
         _devd->_disp,
         _mother,
         name,
         name,
         None,
         (char **)NULL,
         0,
         &hints
  );
    disp_flush();
}

void Data_Elise_Video_Win::clear()
{
    XClearWindow(_devd->_disp,_w);
    disp_flush();
}


void Data_Elise_Video_Win::warn_graph()
{
    _devd->warn_graph();
}

     //   Method specific to rastered windows


void Data_Elise_Video_Win::flush_bli(INT x0,INT x1,INT y)
{
      XPutImage
      (   _devd->_disp, _w, _devd->_gc, _xi,
          x0,0,     x0,y,       (x1-x0),1
      );
}


void  Data_Elise_Video_Win::rast_draw_pixels(El_RW_Point * pts,INT nb) 
{
      XDrawPoints
      (
        _devd->_disp, _w, _devd->_gc,
         (XPoint *) pts, nb,CoordModeOrigin
      );
}


void  Data_Elise_Video_Win::rast_draw_col_pix
      (
           INT * c,
           El_RW_Point * pts,
           INT nb
      ) 
{
     for (INT i = 0; i < nb ; i++)
     {
        _devd->vset_cur_coul(c[i]);
         XDrawPoint
         (
             _devd->_disp,
             _w,
             _devd->_gc,
             pts[i].x,
             pts[i].y
         );
     }
}

void  Data_Elise_Video_Win::rast_draw_col_big_pixels
      (
           INT * c,
           El_RW_Rectangle * rect,
           INT nb
      )
{
     for (INT i=0; i<nb ; i++)
     {
        _devd->vset_cur_coul(c[i]);
         XFillRectangle
         (
             _devd->_disp,
             _w,
             _devd->_gc,
             rect[i].x,
             rect[i].y,
             rect[i].width,
             rect[i].height
         );
     }
}

void  Data_Elise_Video_Win::rast_draw_big_pixels(El_RW_Rectangle * rects,INT nb)
{
      XFillRectangles
      (
           _devd->_disp, _w, _devd->_gc,
           (XRectangle *)rects,nb
      );
}


     //   Method specific to graphical  windows

void Data_Elise_Video_Win::_inst_draw_seg(Pt2dr p1,Pt2dr p2)
{
     XDrawLine
     (
         _devd->_disp,
         _w,
         _devd->_gc,
         round_ni(p1.x),
         round_ni(p1.y),
         round_ni(p2.x),
         round_ni(p2.y)
     ); 
}



void Data_Elise_Video_Win::_inst_draw_circle(Pt2dr centre,Pt2dr radius)
{
     XDrawArc
     (
         _devd->_disp,
         _w,
         _devd->_gc,
         round_ni(centre.x -radius.x),
         round_ni(centre.y-radius.y),
         round_ni(2*radius.x),
         round_ni(2*radius.y),
         0,
         X11_2PI
     ); 
}

void Data_Elise_Video_Win::_inst_draw_rectangle(Pt2dr p1,Pt2dr p2)
{

     XDrawRectangle
     (
         _devd->_disp,
         _w,
         _devd->_gc,
         round_ni(p1.x),
         round_ni(p1.y),
         ElMax(1,round_ni(p2.x-p1.x)-1),
         ElMax(1,round_ni(p2.y-p1.y)-1)
     ); 
}


void  Data_Elise_Video_Win::_inst_draw_polyl
      (const REAL * x,const REAL *y,INT nb)
{
      ASSERT_INTERNAL
      (
             nb <= NB_BUF_DRAW_POLY,
             "insufficcient buf in Data_Elise_Video_Win::_inst_draw_polyl"
      );

      XPoint pts[NB_BUF_DRAW_POLY];
      for (int k=0 ; k<nb; k++)
      {
           pts[k].x = round_ni(x[k]);
           pts[k].y = round_ni(y[k]);
      }
      XDrawLines
      (
         _devd->_disp,
         _w,
         _devd->_gc,
         pts,
         nb,
         CoordModeOrigin
      );

}

void  Data_Elise_Video_Win::_inst_draw_poly_segs
        (const REAL * x1,const REAL *y1,
         const REAL * x2,const REAL *y2,INT nb)
{
    ASSERT_INTERNAL
    (
             nb <= NB_BUF_DRAW_POLY,
             "insufficcient buf in Data_Elise_Video_Win::_inst_draw_poly_segs"
    );

    XSegment segs[NB_BUF_DRAW_POLY];
    for (int k=0 ; k<nb; k++)
    {
           segs[k].x1 = round_ni(x1[k]);
           segs[k].y1 = round_ni(y1[k]);
           segs[k].x2 = round_ni(x2[k]);
           segs[k].y2 = round_ni(y2[k]);
    }
    XDrawSegments
    (
         _devd->_disp,
         _w,
         _devd->_gc,
         segs,
         nb
    );
}


void Data_Elise_Video_Win::set_cl_coord(Pt2dr tr,Pt2dr sc)
{
    _tr = tr;
    _sc = sc;
}

Pt2dr Data_Elise_Video_Win::to_user_geom(Pt2dr p)
{
      return rto_user_geom(p,_tr,_sc);
}

#define X_RETOUR_CHARIOT 13
#define X_ERASE 8
#define X_Arrow_Recule_Deb 1  // Crt A
#define X_Arrow_Recule 2  // Crt B
#define X_Arrow_Avance 6  // Ctr F
#define X_Arrow_Avance_Fin 5  // Ctr E



Pt2di Data_Elise_Video_Win::InstSizeFixedString(const std::string aStr)
{
    
   XFontStruct * aFont = _devd->SetFixedFonte();
   XCharStruct aXCS;
   int aDirRet,aFontAccRet,aFontDescRet;

   XTextExtents(aFont,aStr.c_str(),aStr.size(),&aDirRet,&aFontAccRet,&aFontDescRet,&aXCS);


   return Pt2di(aXCS.rbearing - aXCS.lbearing ,aXCS.ascent + aXCS.descent);
}


int PopBackString(std::string & aStr)
{
     int aL = aStr.size();
     if (aL)
     {
        int aRes = aStr[aL-1];
        aStr = aStr.substr(0,aL-1); // pop_back en C++11
        return aRes;
     }

     return 0;
}
int PopFrontString(std::string & aStr)
{
     int aL = aStr.size();
     if (aL)
     {
        int aRes = aStr[0];
        aStr = aStr.substr(1,std::string::npos); // pop_back en C++11
        return aRes;
     }

     return 0;
}

std::string Data_Elise_Video_Win::InstGetString
            (
                 const Pt2dr & aPt, 
                 Data_Col_Pal *      colStr,
                 Data_Col_Pal *      colEras,
                 const std::string & aStr0
            )
{
   std::string aStrAv = aStr0;
   std::string aStrApr;


 //  XSelectInput(ecr->X11_ecran,fen->X11_fen,KeyPressMask);

   bool cont = true;

   XEvent event;
   XFontStruct * aFont = _devd->SetFixedFonte();
   set_col(colStr);
   std::string aBar = "|";
   int aLargBar = XTextWidth(aFont,aBar.c_str(),aBar.size());

// XSelectInput(ecr->X11_ecran,fen->X11_fen,KeyPressMask);

   while (cont)
   {
      std::string aMes = aStrAv + aStrApr + "  ";
      set_col(colStr);
      XDrawImageString (_devd->_disp,_w,_devd->_gc,aPt.x,aPt.y, aMes.c_str(),aMes.size());

      int aLarg= ElMax(2,XTextWidth(aFont,aStrAv.c_str(),aStrAv.size()));
      set_col(colEras);
      XDrawString (_devd->_disp,_w,_devd->_gc,aPt.x+aLarg - aLargBar/2 ,aPt.y+2, aBar.c_str(),aBar.size());
      XDrawString (_devd->_disp,_w,_devd->_gc,aPt.x+aLarg - aLargBar/2 ,aPt.y-2, aBar.c_str(),aBar.size());
      char mes[4];

       while (!  XCheckMaskEvent(_devd->_disp,KeyPressMask,&event)) ;
       int nb = XLookupString(&(event.xkey),mes,2,(KeySym *)NULL,(XComposeStatus *)NULL);


       if (nb)
       {
           if (mes[0] == X_RETOUR_CHARIOT)
               cont = false;
            else if (mes[0] == X_ERASE)
            {
               PopBackString(aStrAv);
            }
            else if (mes[0] == X_Arrow_Recule)
            {
                int aC = PopBackString(aStrAv);
                if (aC)
                   aStrApr = char(aC) + aStrApr;
            }
            else if (mes[0] == X_Arrow_Avance)
            {
                int aC = PopFrontString(aStrApr);
                if (aC)
                   aStrAv =  aStrAv + char(aC);
            }
            else if (mes[0] == X_Arrow_Recule_Deb)
            {
                 aStrApr = aStrAv + aStrApr;
                 aStrAv = "";
            }
            else if (mes[0] == X_Arrow_Avance_Fin)
            {
                 aStrAv = aStrAv + aStrApr;
                 aStrApr = "";
            }
            else
            {
                if (isprint(mes[0]))
                   aStrAv += mes[0];
                else
                   std::cout << "Non printable caracter " << int(mes[0]) << "\n";
            }

/*

            std::string aMes = aStrAv + aStrApr + "  ";
            set_col(colStr);
            XDrawImageString (_devd->_disp,_w,_devd->_gc,aPt.x,aPt.y, aMes.c_str(),aMes.size());

            int aLarg= ElMax(2,XTextWidth(aFont,aStrAv.c_str(),aStrAv.size()));
            set_col(colEras);
            XDrawString (_devd->_disp,_w,_devd->_gc,aPt.x+aLarg - aLargBar/2 ,aPt.y+2, aBar.c_str(),aBar.size());
            XDrawString (_devd->_disp,_w,_devd->_gc,aPt.x+aLarg - aLargBar/2 ,aPt.y-2, aBar.c_str(),aBar.size());
*/
       }

   }
   return  aStrAv + aStrApr;
}


/*
std::string Data_Elise_Video_Win::InstGetString
            (
                 const Pt2dr & aPt, 
                 Data_Col_Pal *      colStr,
                 Data_Col_Pal *      colEras
            )
{
   std::vector<char> aRes;


 //  XSelectInput(ecr->X11_ecran,fen->X11_fen,KeyPressMask);

   bool cont = true;
   int i=0;
   int larg =0;

   XEvent event;
   XFontStruct * aFont = _devd->SetFixedFonte();
   set_col(colStr);

// XSelectInput(ecr->X11_ecran,fen->X11_fen,KeyPressMask);

   while (cont)
   {
      char mes[4];

       while (!  XCheckMaskEvent(_devd->_disp,KeyPressMask,&event)) ;
       int nb = XLookupString(&(event.xkey),mes,2,(KeySym *)NULL,(XComposeStatus *)NULL);

       char * aStr = &(aRes[0]);

       if (nb)
       {
           if (mes[0] == X_RETOUR_CHARIOT)
               cont = false;
            else if (mes[0] == X_ERASE)
            {
               if (i)
               {
                    i--;
                    larg -= XTextWidth(aFont,aStr+i,1);
                    set_col(colEras);
                    XDrawString (_devd->_disp,_w,_devd->_gc,  aPt.x+larg,aPt.y, aStr+i,1);
                    set_col(colStr);
                    aRes.pop_back();
               }
 
            }
            else
            {
                aRes.push_back( mes[0]);
                aStr = &(aRes[0]);
                
                XDrawImageString (_devd->_disp,_w,_devd->_gc,  aPt.x+larg,aPt.y, aStr+i,1);
                //XDrawString (_devd->_disp,_w,_devd->_gc,  aPt.x+larg,aPt.y, aStr+i,1);
                larg += XTextWidth(aFont,aStr+i,1);
                i++;
            }

       }

   }
   aRes.push_back(0);
   return std::string(&(aRes[0]));
}
*/




void  Data_Elise_Video_Win::_inst_fixed_string(Pt2dr pt,const char * name,bool draw_image)
{
    _devd->SetFixedFonte();
    if (draw_image)
       XDrawImageString
       (
             _devd->_disp,
             _w,
             _devd->_gc,     
             round_ni(pt.x),
             round_ni(pt.y),
             name,
             strlen(name)
       );
    else
       XDrawString
       (
             _devd->_disp,
             _w,
             _devd->_gc,     
             round_ni(pt.x),
             round_ni(pt.y),
             name,
             strlen(name)
       );
    disp_flush();
}

void   Data_Elise_Video_Win::grab(Grab_Untill_Realeased & gur)
{
		_devd->grab(this,gur);
}

/******************************************************************/
/*                                                                */
/*                    Video_Win                                   */
/*                                                                */
/******************************************************************/

Video_Win::Video_Win   
(
       Video_Display            d  ,
       Elise_Set_Of_Palette  esop  ,
       Pt2di                    p0 ,
       Pt2di                    sz ,
       INT          border_witdh 
)   :
    El_Window
    (
        new Data_Elise_Video_Win(d,esop,p0,sz,border_witdh,(Window *)0),
        Pt2dr(0.0,0.0),
        Pt2dr(1.0,1.0)
    )
{
}

Video_Win::Video_Win
(
       Video_Win                aSoeur,
       ePosRel                  aPos,
       Pt2di                    sz ,
       INT                      border_witdh
)   :
    El_Window
    (
        new Data_Elise_Video_Win(aSoeur.disp(),aSoeur.sop(),Pt2di(0,0),sz,border_witdh,(Window *)0,aSoeur.devw(),aPos),
        Pt2dr(0.0,0.0),
        Pt2dr(1.0,1.0)
    )
{
}

Video_Win::Video_Win
(
     Video_Win_LawLevelId,
     void *    anId,
     void *    aScrId,
     Pt2di     aSz
) :
    El_Window
    (
        new Data_Elise_Video_Win
            (
                 Video_Display(Video_Win_LawLevelId(),aScrId),
                 Elise_Set_Of_Palette::TheFullPalette(),
                 Pt2di(0,0),
                 aSz,
                 1,
                 (Window *)anId
            ),
        Pt2dr(0.0,0.0),
        Pt2dr(1.0,1.0)
    )
{
}



Video_Win::Video_Win (class Data_Elise_Video_Win * w,Pt2dr tr,Pt2dr sc) :
       El_Window (w,tr,sc)
{
}


Video_Win Video_Win::chc(Pt2dr tr,Pt2dr sc,bool SetClikCoord)
{
    Video_Win W = Video_Win(devw(),tr,sc);
    if (SetClikCoord)
        W.set_cl_coord(tr,sc);
    return W;
}

Video_Win  * Video_Win::PtrChc(Pt2dr tr,Pt2dr sc,bool SetClikCoord )
{
    Video_Win * pW = new Video_Win(devw(),tr,sc);
    if (SetClikCoord)
        pW->set_cl_coord(tr,sc);
    return pW;
}

       //-----------------------------------

void Video_Win::set_sop(Elise_Set_Of_Palette sop)
{
    SAFE_DYNC(Data_Elise_Raster_W *,degraw())->rset_sop(sop);
}

Data_Elise_Video_Win * Video_Win::devw()
{
     return SAFE_DYNC (Data_Elise_Video_Win *,degraw());
}

const Data_Elise_Video_Win * Video_Win::devw() const
{
     return SAFE_DYNC (Data_Elise_Video_Win *,degraw());
}



bool Video_Win::operator == (const Video_Win & w2) const         
{
    return devw() == w2.devw();
}


void Video_Win::set_cl_coord(Pt2dr tr,Pt2dr sc)
{
      devw()->set_cl_coord(tr,sc);
}

void Video_Win::set_title(const char * name)
{
      devw()->set_title(name);
}

void Video_Win::clear()
{
      devw()->clear();
}

Clik  Video_Win::clik_in()
{
      X11ValideEvent  OK;
      X11ValidButonThisWindow  ThisW(devw()->_w);

      return devw()->_devd->clik(ThisW,true,OK,true);
}

std::string Video_Win::GetString(const Pt2dr & aPt,Col_Pal aColDr,Col_Pal aColErase,const std::string & aStr0)
{
   return devw()->InstGetString(U2W(aPt),aColDr.dcp(),aColErase.dcp(),aStr0);
}

Pt2di Video_Win::SizeFixedString(const std::string aStr)
{
   return devw()->InstSizeFixedString(aStr);
}



Pt2di Video_Win::fixed_string_middle(int aPos,const std::string &  name,Col_Pal aCol,bool draw_im)
{
   return fixed_string_middle(Box2di(Pt2di(0,0),sz()),aPos,name,aCol,draw_im);
}

Pt2di Video_Win::fixed_string_middle(const Box2di & aBox,int aPos,const std::string &  name,Col_Pal aCol,bool draw_im)
{
    Pt2di aLarg = SizeFixedString(name);
    int anY = aBox._p0.y +  (aBox.sz().y + aLarg.y) / 2;
    int aRab=ElAbs(aPos);

    // int anY = (sz().y + aLarg.y) / 2;
    int anX = 0;
    if (aPos < 0)  anX=aRab;
    else if (aPos > 0)  anX = aBox.sz().x - aLarg.x -aRab;
    else  anX = (aBox.sz().x - aLarg.x) / 2;

    anX += aBox._p0.x;

    Pt2di aRes(anX,anY);
    fixed_string(Pt2dr(aRes),name.c_str(),aCol,draw_im);

    return aRes;
}



void Video_Win::grab(Grab_Untill_Realeased & gur)
{
    devw()->grab(gur);
}

void Video_Win::write_image
      (
              Pt2di p0src,
              Pt2di p0dest,
              Pt2di sz,
              INT *** Im,
              Elise_Palette pal
      )
{

	 devw()->write_image(p0src,p0dest,sz,Im,pal.dep());
}

void Video_Win::load_image(Pt2di p0src,Pt2di p0dest,Pt2di sz)
{
     devw()->load_image(p0src,p0dest,sz);
}

void Video_Win::load_image()
{
     devw()->load_image(Pt2di(0,0),Pt2di(0,0),sz());
}

void Video_Win::load_image(Pt2di p0,Pt2di p1)
{
     devw()->load_image(p0,p0,p1-p0);
}
                               
void Video_Win::translate(Pt2di tr)
{
     devw()->translate(tr);
}

void Video_Win::image_translate(Pt2di tr)
{
     devw()->image_translate(tr);
}

Video_Display    Video_Win::disp()
{
	return  devw()->_devd;
}


ElXim  Video_Win::StdBigImage()
{
   return ElXim(devw()-> alloc_big_image());
}


ElList<Pt2di> Video_Win::GetPolyg(Line_St aLST,INT aButonEnd)
{
    ElList<Pt2di> aLRes;

    for (;;)
    {
       Clik   aCl = clik_in();
       if (aCl._b == aButonEnd)
          return aLRes;
       if (! aLRes.empty())
          draw_seg ( Pt2dr(aCl._pt), Pt2dr(aLRes.car()), aLST);
       aLRes = aLRes+Pt2di(aCl._pt);
    }
    // SHOULD NOT BE HERE
    return aLRes;
}

Pt2dr Video_Win::to_user_geom(Pt2dr p)
{
   return devw()->to_user_geom(p);
}


EliseStdImageInteractor * Video_Win::Interactor()
{
      return devw()->mInteractor;
}

void  Video_Win::SetInteractor(EliseStdImageInteractor * anI)
{
      devw()->mInteractor = anI;
}

Video_Win  Video_Win::WSzMax(Pt2dr aSzTarget,Pt2dr aSzMax,double & aZoom)
{
    aZoom = aSzMax.RatioMin(aSzTarget);
    Pt2di aSzReal = round_ni(aSzTarget*aZoom);
    return Video_Win::WStd(round_ni(Pt2dr(aSzReal)/aZoom),aZoom);
    
}

Video_Win  Video_Win::WSzMax(Pt2dr aSzTarget,Pt2dr aSzMax)
{
     double aZoom;
     return WSzMax(aSzTarget,aSzMax,aZoom);
}


Video_Win  Video_Win::LoadTiffWSzMax(const std::string &aNameTiff,Pt2dr aSzMax,double & aZoom)
{
    Tiff_Im aTif = Tiff_Im::StdConvGen(aNameTiff,1,false);
    Video_Win aW = Video_Win::WSzMax(Pt2dr(aTif.sz()),aSzMax,aZoom);
    Video_Win aW0 = aW.chc(Pt2dr(0,0),Pt2dr(1,1));

    ELISE_COPY
    (
        aW0.all_pts(),
        StdFoncChScale(aTif.in_proj(),Pt2dr(0,0),Pt2dr(1/aZoom,1/aZoom),Pt2dr(1,1)),
        aW0.ogray()
    );

    return aW;
}


Video_Win  Video_Win::chc_fit_sz(Pt2dr aSz,bool ClikCoord)
{
    REAL aZoom = Pt2dr(sz()).RatioMin(aSz);
    return chc(Pt2dr(0,0),Pt2dr(aZoom,aZoom),ClikCoord);

}

void Video_Win::raise()
{
    devw()->raise();
}
void Video_Win::lower()
{
    devw()->lower();
}
void Video_Win::move_to(const Pt2di& aP)
{
    devw()->move_to(aP);
}

void Video_Win::move_translate(const Pt2di& aP)
{
    devw()->move_translate(aP);
}

/*****************************************************************/
/*                                                               */
/*                  Data_El_Video_Display                        */
/*                                                               */
/*****************************************************************/

void  Data_El_Video_Display::init_bits_acc
      (
          INT & aMemoMask,
          INT & shift,
          INT & mult,
          INT & ind,
          INT & nbb,
          INT   mask
      )
{ 
     aMemoMask = mask;
     if (_cmod == Indexed_Colour)
        return;

     INT b0;
     for (b0 = 0; (b0 < 32) && ((mask&(1<<b0))==0)  ; b0++);
     ASSERT_INTERNAL(b0!=32,"unexpected bits mask in XImage");

     INT b1;
     for (b1 = b0; (b1 < 32) && ((mask&(1<<b1))!=0)  ; b1++);
     ASSERT_INTERNAL((b1!=32)&&(b1!=b0),"unexpected bits mask in XImage");

     if (_cmod == True_24_Colour)
        ASSERT_INTERNAL
        (
            ((b0%8)==0) && ((b1%8)==0),
            "unexpected bits mask in 24 bits-XImage"
        );

     nbb = b1-b0;
     shift = b0;
     mult  = 1<<nbb ;

     ind = b0 / 8;

}

void Data_El_Video_Display::init_xi(XImage *  xi)
{
    INT bpp = round_ni(xi->bytes_per_line/(REAL)xi->width);
    if (! _xi_first)
    {
       ASSERT_INTERNAL
       (
              (xi->bitmap_bit_order == _xi0.bitmap_bit_order)
          &&  (xi->byte_order       == _xi0.byte_order      )
          &&  (xi->bits_per_pixel   == _xi0.bits_per_pixel  )
          &&  (xi->red_mask         == _xi0.red_mask        )
          &&  (xi->blue_mask        == _xi0.blue_mask       )
          &&  (xi->green_mask       == _xi0.green_mask      )   
          &&  (bpp                  ==  _byte_pp            )   ,
          "do not handle Variable XImage on the same display"
       );
       return;
     }
     _xi_first = false;
     _xi0 = *xi;

     _byte_pp = bpp;

     if  (_xi0.byte_order != LSBFirst)
     {
         _r_ind = 3-_r_ind;
         _g_ind = 3-_g_ind;
         _b_ind = 3-_b_ind;
     }
}

void Data_El_Video_Display::augmente_pixel_index(INT nb)
{
    if (_cmod != Indexed_Colour)
       return;

    ASSERT_TJS_USER
    (
          (nb>=1) &&(nb <= _nb_pix_ind_max) ,
          "Elise X11 cannot manage over 256 color"
    );

    if (nb > _nb_pix_ind)
    {
        unsigned long  plane_mask[1];

        INT ok_alloc = XAllocColorCells
                       (
                            _disp,
                            _cmap,
                            False,
                            plane_mask,
                            0,
                            _pix_ind + _nb_pix_ind,
                            nb-_nb_pix_ind
                       );
         _nb_pix_ind = nb;

         ASSERT_TJS_USER(ok_alloc,"Allocation de couleur impossible");
    }

}


void Data_El_Video_Display::empty_events()
{
    XEvent event;
    while (XCheckIfEvent(_disp,&event,always_true,(char *) NULL));

}

Clik::Clik(Video_Win w,Pt2dr pt,INT b,U_INT state) :
      _w   (w),
      _pt  (pt),
      _b   (b),
	  _state (state)
{
}


bool  Clik::b1Pressed() const
{
	return (_state & Button1Mask) != 0;
}

bool  Clik::b2Pressed() const 
{
    return (_state & Button2Mask) != 0;
}                             

bool  Clik::b3Pressed() const 
{
    return (_state & Button3Mask) != 0;
}                             

bool  Clik::controled() const 
{
    return (_state & ControlMask) != 0;
}                             

bool  Clik::shifted() const
{
    return (_state & ShiftMask) != 0;
}                             




Clik Data_El_Video_Display::clik
     (
           X11ValideEvent & press,
           bool             wait_press,
           X11ValideEvent & release,
           bool             wait_release
     )
{
   XEvent e;

   if (wait_press)
   {
   	empty_events();
   	while (
               	(!XCheckMaskEvent(_disp,ButtonPressMask,&e))
           	||  (! press.ok(e))
         	);
   }

   empty_events();


   if (wait_release)
   {
   	while (
               	(!XCheckMaskEvent(_disp,ButtonReleaseMask,&e))
           	||  (! release.ok(e))
         	);
    	empty_events();
   }
   
   Data_Elise_Video_Win *  w = get_w(e.xbutton.window,true);
   if (w==0)
      return clik(press,wait_press,release,wait_release);

   Pt2dr ploc = Pt2dr(e.xbutton.x,e.xbutton.y);

   return Clik(w,w->to_user_geom(ploc),e.xbutton.button,e.xbutton.state);
}


void   Data_El_Video_Display::grab(Data_Elise_Video_Win * wel, Grab_Untill_Realeased & gur)
{
		Window w = wel->_w;
    	empty_events();

		bool first = true;
		INT xl=-100000,yl=-100000;

		while (true)
		{
			 Window root,child; 
             int xr,yr,xc,yc;
			 U_INT State =0;

			 if ( XQueryPointer
					(
						_disp,w,&root,&child,
                        &xr,&yr,&xc,&yc,
						&State
					)
				)
			 {
				 if (root == _root_w)
				 {
					 bool Mvt =   first || (xc!=xl) || (yc!= yl);
					 first = false;
					 if (Mvt || (!gur.OnlyMvmt()))
                     {
						xl = xc; yl = yc;
						gur.GUR_query_pointer(Clik(wel,Pt2dr(xc,yc),-1,State),Mvt);
                     }
				 }
			 }
             XEvent e;
             if (XCheckMaskEvent(_disp,ButtonReleaseMask,&e))
			 {
   				Pt2dr p = Pt2dr(e.xbutton.x,e.xbutton.y);
				gur.GUR_button_released(Clik(wel,p,e.xbutton.button,e.xbutton.state));
				 return;
			 }
		}
}


void Data_El_Video_Display::warn_graph()
{
    if  ((!_pal_load) && (!_warn_no_pal_load))
    {
        _warn_no_pal_load  = true;
        cout << "Warning : no loaded palette on video display\n";
    }
}

void Data_El_Video_Display::vset_cur_coul(INT coul)
{
    if (coul != _cur_coul)
    {
        XSetForeground(_disp,_gc,coul);
        _cur_coul = coul;
    }    
}

void Data_El_Video_Display::set_cur_coul(INT coul)
{
    vset_cur_coul(coul);
}


INT intens_0_1_to_ushrt(REAL intens)
{
    return  ElMax(0,ElMin((INT)USHRT_MAX, (INT)(USHRT_MAX * intens)));
}

void Data_El_Video_Display::load_palette(Elise_Set_Of_Palette esop)
{
     _pal_load          = true;
     if (_cmod != Indexed_Colour)
        return;

     Data_Disp_Set_Of_Pal * ddsop = get_comp_pal(esop);

     INT nb_tot = 0;

     for (INT i_pal = 0; i_pal < ddsop->nb(); i_pal++)
     {
         Data_Disp_Pallete  * pc =  ddsop->kth_ddp(i_pal);
         Data_Elise_Palette * p  = pc->dep_of_ddp();
         INT nb_sup              = p->nb();


         for (INT i_coul=0; i_coul<nb_sup; i_coul++)
         {
             Elise_colour col = p->kth_col(i_coul);
             INT          iX  = i_coul + nb_tot;

             _lut[iX].red    = intens_0_1_to_ushrt(col.r());
             _lut[iX].green  = intens_0_1_to_ushrt(col.g());
             _lut[iX].blue   = intens_0_1_to_ushrt(col.b());

             _lut[iX].pixel = _pix_ind[iX];
             _lut[iX].flags = (DoRed | DoGreen | DoBlue);
             _lut[iX].pad = 0;
         }

         nb_tot += nb_sup;
     }
     XStoreColors(_disp,_cmap,_lut,nb_tot);
     disp_flush();
}


void Data_El_Video_Display::add_w(Data_Elise_Video_Win * w)
{
     LW * nlw  = new LW;
     nlw->_w = w;
     nlw->_n = _lw;
    _lw = nlw;
}

Data_Elise_Video_Win * Data_El_Video_Display::get_w(Window w,bool svp)
{
    for(LW * l = _lw; l ; l = l->_n)
       if (l->_w->_w ==  w)
          return l->_w;

    El_Internal.ElAssert
    (
       svp,
       EEM0 << "Cannot get window in Data_El_Video_Display::get_w \n"
    );
    return 0;
}

void Data_El_Video_Display::SetCurFonte(XFontStruct * XFS)
{
    if (_CurFont != XFS)
    {
       _CurFont = XFS;
        XSetFont(_disp,_gc,XFS->fid);
    }
}

XFontStruct * Data_El_Video_Display::SetFixedFonte()
{
    if (_FixedFont == 0)
    {
        _FixedFont =  XLoadQueryFont(_disp,"fixed");
        ELISE_ASSERT(_FixedFont != 0,"Data_El_Video_Display::SetFixedFonte");
    }
    SetCurFonte(_FixedFont);

    return _FixedFont;
}





Data_El_Video_Display::Data_El_Video_Display(const char * name,void * Id) :
     Data_Elise_Raster_D(name,NB_COUL_X11_LUT)
{

  _lw = 0;
  _lut = NEW_VECTEUR(0,_nb_pix_ind_max,XColor);
  if (Id)
    _disp = (Display *) Id;
  else
     _disp        =    XOpenDisplay(name);
   ASSERT_TJS_USER
   (
      _disp != (Display *) NULL,
      "Ouverture X11 ecr imposible"
   );

   _root_w    =   XDefaultRootWindow(_disp);
   _num_ecr   =   XDefaultScreen(_disp);
   _depth     =   DefaultDepth(_disp,_num_ecr);
   _screen    =   DefaultScreenOfDisplay(_disp);

    _xi_first   = true;

    // cout << "Depth Ecran = " << _depth << "\n";


    init_mode(_depth);
    switch (_depth)
    {
           case 8:
           {
              ASSERT_TJS_USER
              (
                   XMatchVisualInfo(_disp,_num_ecr,_depth,PseudoColor,&_xvi),
                   " cannot match PseudoColor on 8-bits display"
              );
              _visual   =   XDefaultVisualOfScreen(_screen);
              _cmap     =   DefaultColormap(_disp,_num_ecr);
           }
           break;

           case 16 :
           case 24 :
           {
              ASSERT_TJS_USER
              (
                   XMatchVisualInfo(_disp,_num_ecr,_depth,TrueColor,&_xvi),
                   " cannot match TrueColor on 16 or 24-bits display"
              );
              _cmap =  XCreateColormap
                        (
                          _disp,
                          _root_w,
                          _xvi.visual,
                          AllocNone // needed, TrueColor => read only
                        );
              _visual     =   _xvi.visual;



               init_bits_acc(_r_mask,_r_shift,_r_mult,_r_ind,_r_nbb,_xvi.red_mask  );
               init_bits_acc(_g_mask,_g_shift,_g_mult,_g_ind,_g_nbb,_xvi.green_mask);
               init_bits_acc(_b_mask,_b_shift,_b_mult,_b_ind,_b_nbb,_xvi.blue_mask );
           }
           break;

           default :
           elise_fatal_error
           (
               " Elise X11 : handle only 8, 16 or 24 bit depth display",
               __FILE__,__LINE__
           );
    }

   _gc          =   XDefaultGC(_disp,_num_ecr);

   _pal_load          = false;
   _warn_no_pal_load  = false;


#if (DEBUG_INTERNAL)

      // some verifications because the equivalence El_RW_Point <=> Xpoint
      //  is rather managed sadly.

   {
      El_RW_Point *p = new El_RW_Point(244,592);
      XPoint * xp  = reinterpret_cast<XPoint *> (p);
      ASSERT_INTERNAL
      (
              (sizeof(El_RW_Point)== sizeof(XPoint))
         &&   (xp->x == 244) 
         &&   (xp->y == 592),
        "bad equivalence between XPoint and El_RW_Point"
      );
      delete p;
   }

   {
      El_RW_Rectangle *r= new El_RW_Rectangle(11,22,33,44);
      XRectangle *xr = reinterpret_cast<XRectangle *> (r);
      ASSERT_INTERNAL
      (
              (sizeof(El_RW_Rectangle)== sizeof(XRectangle))
         &&   (xr->x      == 11) 
         &&   (xr->y      == 22)
         &&   (xr->width  == 33) 
         &&   (xr->height == 44),
        "bad equivalence between El_RW_Rectangle and XRectangle"
      );
      delete r;
   }
#endif // DEBUG_INTERNAL

   _FixedFont = 0;
   _CurFont = 0;

   XWindowAttributes anAttrs;
   XGetWindowAttributes(_disp,_root_w,&anAttrs);
   mSzEcr = Pt2di(anAttrs.width,anAttrs.height);
}


void Data_El_Video_Display::disp_flush(){XFlush(_disp);}


Data_El_Video_Display::~Data_El_Video_Display()
{
      DELETE_VECTOR(_lut,0);
}

void Data_El_Video_Display::_inst_set_line_witdh(REAL lw)
{
     XSetLineAttributes
     (
          _disp,
          _gc,
          round_ni(lw),
          LineSolid,
          CapRound,
          JoinRound
     );
}


/*****************************************************************/
/*                                                               */
/*                  Video_Display                                */
/*                                                               */
/*****************************************************************/

Video_Display::Video_Display(const char * name) :
     PRC0 (new Data_El_Video_Display(name,(void *)0))
{
}

Video_Display::Video_Display(Video_Win_LawLevelId,void * Id) :
     PRC0 (new Data_El_Video_Display((char *)0,Id))
{
}





Video_Display::Video_Display(Data_El_Video_Display * disp) :
     PRC0 (disp)
{
}





Data_El_Video_Display * Video_Display::devd()
{
    return SAFE_DYNC(Data_El_Video_Display *,_ptr);
}


const Data_El_Video_Display * Video_Display::devd() const
{
    return SAFE_DYNC(const Data_El_Video_Display *,_ptr);
}




void  Video_Display::load(Elise_Set_Of_Palette sop)
{
       devd()->load_palette(sop);
}




Clik  Video_Display::clik()
{
       X11ValideEvent  OK;

       return devd()->clik(OK,true,OK,true);
}

Clik  Video_Display::clik_press()
{
       X11ValideEvent  OK;
       return devd()->clik(OK,true,OK,false);
}

Clik  Video_Display::clik_release()
{
       X11ValideEvent  OK;
       return devd()->clik(OK,false,OK,true);
}

INT Video_Display::Depth() const
{
   return devd()->_depth;
}

Pt2di Video_Display::Sz() const
{
   return devd()->mSzEcr;
}


bool Video_Display::TrueCol() const
{
   return (devd()->_cmod==True_16_Colour) || (devd()->_cmod==True_24_Colour);
}



//   HJMPD

HJ_PtrDisplay Video_Win::display()                     
{
     return devw()->_devd->_disp;
}

HJ_Window   Video_Win::window() 
{
     return devw()->_w;
}


void Video_Win::DumpImage(const std::string & aName)
{
    Pt2di aSz = sz();

    ElXim  anIm (*this,aSz);

    Im2D_U_INT1 aIR(aSz.x,aSz.y);
    Im2D_U_INT1 aIG(aSz.x,aSz.y);
    Im2D_U_INT1 aIB(aSz.x,aSz.y);


   anIm.read_in_el_image(Pt2di(0,0),Pt2di(0,0),aSz,aIR,aIG,aIB);


   Tiff_Im::Create8BFromFonc
   (
        aName,
        aSz,
        Virgule(aIR.in(),aIG.in(),aIB.in())
   );
}


/***************************************************/
/*                                                 */
/*            cFenMenu                             */
/*                                                 */
/***************************************************/

cFenMenu::cFenMenu(Video_Win aWSoeur,const Pt2di & aSzCase,const Pt2di & aNb) : 
   mW       (aWSoeur,Video_Win::eSamePos,aSzCase.mcbyc(aNb)),
   mSzCase  (aSzCase),
   mNb      (aNb)
{
   mW.lower();
}


Video_Win cFenMenu::W() 
{
  return mW;
}

Pt2di  cFenMenu::Pt2Case(const Pt2di & aP) const
{
    return Sup
           (
               Pt2di(0,0),
               Inf( aP.dcbyc(mSzCase),mNb-Pt2di(1,1))
           );
}

Pt2di  cFenMenu::Case2Pt(const Pt2di & aCase) const
{
     return  aCase.mcbyc(mSzCase);
}

void cFenMenu::ColorieCase(const Pt2di & aKse,Col_Pal aCol,int aBrd)
{
   mW.fill_rect
   (
         Pt2dr(Case2Pt(aKse)),
         Pt2dr(Case2Pt(aKse+Pt2di(1,1))) -Pt2dr(aBrd,aBrd),
         Fill_St(aCol)
   );
}


void cFenMenu::StringCase(const Pt2di &  aKse,const std::string & aStr, bool center)
{
    Pt2dr aDeb = Pt2dr(Case2Pt(aKse));
    aDeb = aDeb + Pt2dr(0.0,mSzCase.y/2.0 + 5);

    if (center)
       aDeb =  aDeb + Pt2dr(mSzCase.x/2.0 - aStr.size()*3.5,0);
    else
       aDeb = aDeb + Pt2dr(5,0);

    mW.fixed_string(aDeb,aStr.c_str(),mW.pdisc()(P8COL::black));
}


/***************************************************/
/*                                                 */
/*            cFenOuiNon                           */
/*                                                 */
/***************************************************/

cFenOuiNon::cFenOuiNon
(
   Video_Win aWSoeur,
   const Pt2di & aSzCase
) :
  cFenMenu(aWSoeur,aSzCase,Pt2di(1,3))
{
}

bool cFenOuiNon::Get(const std::string & aMes)
{
  mW.raise();
  ColorieCase(Pt2di(0,0),mW.prgb()(255,255,255),1);
  ColorieCase(Pt2di(0,1),mW.prgb()(128,255,128),1);
  ColorieCase(Pt2di(0,2),mW.prgb()(255,128,128),1);

  StringCase(Pt2di(0,0),aMes,true);
  StringCase(Pt2di(0,1),"Oui",true);
  StringCase(Pt2di(0,2),"Non",true);

  Clik aClk = mW.clik_in();

  Pt2di aKse = Pt2Case(Pt2di(aClk._pt));
  mW.lower();


  return aKse.y==1;
}


#endif  //  (ELISE_VIDEO_WIN== ELISE_VW_X11)








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
