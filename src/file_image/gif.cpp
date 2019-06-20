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



class GIF_Color
{
  public :
     U_INT1 red;
     U_INT1 green;
     U_INT1 blue;

     static void read_tab(GIF_Color *tgc,INT nb,ELISE_fp & fp);
};

void GIF_Color::read_tab(GIF_Color * tgc,INT nb,ELISE_fp & fp)
{
     U_INT1 c[3];
     for (INT i=0; i<nb ; i++)
     {
          fp.read(c,1,3);
          tgc[i].red   = c[0];
          tgc[i].green = c[1];
          tgc[i].blue  = c[2];
     }
}


static INT per_interlac_gif[4] = {8,8,4,2};
static INT deb_interlac_gif[4] = {0,4,2,1};

class GIF_Global
{
     public :

       static GIF_Color * read_col_pal(ELISE_fp  fp, INT nb);

       static void skip_data_bloc(ELISE_fp  fp);

        typedef enum
        {
            Image_Separator  = 0x2C,
            Extension  =       0x21,
            Trailer    =       0x3b
        }   Introducer;

        typedef enum
        {
            Graphic_Control = 0xF9,
            Comment         = 0xFE,
            Plain_Text      = 0x01

        }  Ext_Type;


};

class Data_GifIm : public ElDataGenFileIm
{
    friend class Gif_Im;
    friend class Gif_Tile_F2d;

    public :
          Pt2di sz() {return Pt2di(_im_w,_im_h);}
          const char * name() {return _name;}

    private :

        virtual ~Data_GifIm(){ delete _tprov_name;}

        Data_GifIm(const char *,ELISE_fp ,class Data_Giff * g);


       Im2D<U_INT1,INT>     im();

     //===========================

        Disc_Pal  _pal;

        Tprov_char *     _tprov_name;
        const char *           _name;

        INT  _left_pos;
        INT  _top_pos;

        INT  _im_w;
        INT  _im_h;

        INT         _nb_bits_lzw_init;

        bool        _loc_col_tab;
        bool        _interlaced;
        bool        _sorted_col_tab;
        INT         _sz_col_tab;
        tFileOffset _offs_data;


        virtual   Fonc_Num in()     ;
        virtual   Fonc_Num in(REAL) ;
        virtual   Output out()      ;

};


class Data_Giff : public RC_Object
{
      friend class Gif_File;
      friend class Data_GifIm;

      private :

         Data_Giff(const char * name);


         Gif_Im kth_im(INT nb);

     //=============================================

         L_Gif_Im  _lgi;
         Disc_Pal  _pal;


         typedef enum
         {
             v87a,
             v89a
         } version;

         //  char      _signature [3];  do not store it just verification
         //  char      _version[3]; prefere enum as storage

        version     _numv;
        U_INT2      _scr_w;
        U_INT2      _scr_h;

                  // packed
        INT         _sz_col_tab;
        bool        _sorted_col_tab;
        INT         _nb_bit_per_col;
        bool        _glob_col_tab;

        INT         _index_back;
        REAL        _aspect_ratio;

        INT         _nb_im;
};

class GIF_char_flow : public Flux_Of_Byte
{
    public :
      GIF_char_flow(ELISE_fp,bool read);
      virtual ~GIF_char_flow();

    private :
      void read_pack();

      U_INT1 _buf[256];
      U_INT1 Getc();
      void Putc(U_INT1);

      void flush_out_put();

      ELISE_fp  _fp;
      INT  _nb_in_pack;
      INT  _i_pack;
      bool _read;

      virtual tFileOffset tell() {return _fp.tell();}
};


Disc_Pal GIF_palette(ELISE_fp  fp,INT nb)
{
    Elise_colour  tec[256];
    GIF_Color     tgc[256];

    GIF_Color::read_tab(tgc,nb,fp);

    for (INT i=0; i<nb ; i++)
        tec[i] = Elise_colour::rgb
                 (
                     tgc[i].red/255.0,
                     tgc[i].green/255.0,
                     tgc[i].blue/255.0
                 );

   return Disc_Pal(tec,nb);
}



/******************************************************/
/*                                                    */
/*                GIF_Global                          */
/*                                                    */
/******************************************************/



GIF_Color * GIF_Global::read_col_pal(ELISE_fp  fp,INT nb)
{
     GIF_Color * res = NEW_VECTEUR(0,nb,GIF_Color);
     GIF_Color::read_tab(res,nb,fp);
     return res;
}

void GIF_Global::skip_data_bloc(ELISE_fp fp)
{
    INT c;
    while((c = fp.lsb_read_U_INT1()))
         fp.seek_cur(c);
}


/******************************************************/
/*                                                    */
/*                Data_Giff                           */
/*                                                    */
/******************************************************/

Gif_Im Data_Giff::kth_im(INT kth)
{
    ASSERT_TJS_USER
    (
        (kth >=0) && (kth < _nb_im),
        "invalid order of image in  Data_Giff::kth_im\n"
    );


    L_Gif_Im l = _lgi;
    for( INT i= kth +1; i < _nb_im; i++)
        l = l.cdr();

    return l.car();
}


Data_Giff::Data_Giff(const char * name) :
     _lgi (),
     _pal (Disc_Pal::PBIN())
{
    ELISE_fp  fp (name,ELISE_fp::READ);
    char c[4];
    c[3] = 0;

    fp.read(c,sizeof(c[0]),3);
    ASSERT_TJS_USER
    (
          ! strncmp(c,"GIF",3),
          "NOT a GIF FILE"
    );


    fp.read(c,sizeof(c[0]),3);

    if (! strncmp(c,"87a",3))
       _numv = v87a;
    else if (! strncmp(c,"89a",3))
       _numv = v89a;
    else
       elise_fatal_error("unknown version of GIF file",__FILE__,__LINE__);

    _scr_w = fp.lsb_read_U_INT2();
    _scr_h = fp.lsb_read_U_INT2();

     U_INT1 packed = fp.lsb_read_U_INT1();

     _sz_col_tab     =  1 << (sub_bit(packed,0,3)+1);
     _sorted_col_tab = (_numv >= v89a) && (kth_bit(packed,3) == 1);
     _nb_bit_per_col = sub_bit(packed,4,7) + 1;
     _glob_col_tab   = (kth_bit(packed,7) == 1);



     _index_back = fp.lsb_read_U_INT1();

     U_INT1 ratio = fp.lsb_read_U_INT1();
     _aspect_ratio = ratio ? ((ratio+15.0) / 64.0) : 1.0;

     if (_glob_col_tab)
        _pal =  GIF_palette(fp,_sz_col_tab);


    _nb_im = 0;
    INT   intro;
    while (
                 ((intro = fp.fgetc()) !=  GIF_Global::Trailer)
               && (intro != ELISE_fp::eof)  // Tolerance to bad gif writters
          )
    {

          if (intro == GIF_Global::Image_Separator)
          {
               _nb_im++;
               _lgi = _lgi + Gif_Im(name,fp,this);
               GIF_Global::skip_data_bloc(fp);
          }
          else if (intro == GIF_Global::Extension)
          {
                U_INT1 ext_type = fp.lsb_read_U_INT1();

                switch (ext_type)
                {
                      case  GIF_Global::Graphic_Control :
                      case  GIF_Global::Comment         :
                      case  GIF_Global::Plain_Text      :

                      break;


                      default :;
/*
                           elise_fatal_error
                           ("unknown gif extension \n",__FILE__,__LINE__);
*/
                }
                GIF_Global::skip_data_bloc(fp);
          }
          else
              elise_fatal_error("unknown kind of gif block \n",__FILE__,__LINE__);
    }
    fp.close();
}


/******************************************************/
/*                                                    */
/*                Gif_File                            */
/*                                                    */
/******************************************************/

Gif_File::Gif_File(const char * name) :
   PRC0(new Data_Giff(name))
{
}

INT Gif_File::nb_im() const
{
    return dgi()->_nb_im;
}

Gif_Im Gif_File::kth_im(INT kth) const
{
    return dgi()->kth_im(kth);
}

/******************************************************/
/*                                                    */
/*                Gif_Fich_Im2d                       */
/*                                                    */
/******************************************************/

class Gif_Fich_Im2d  : public Std_Bitm_Fich_Im_2d
{
   friend class Gif_Tile_F2d;

   public :
       Gif_Fich_Im2d(Flux_Pts_Computed * flx,Data_GifIm *);
   //private :
   //    Data_GifIm * _dgi;
};


class Gif_Tile_F2d : public Tile_F2d
{
    public :
        Gif_Tile_F2d(Data_GifIm *);
        virtual ~Gif_Tile_F2d();

    private :

        virtual void read_seg(Fich_Im2d *,void * buf,INT x0,INT x1);
        virtual void seek_in_line(Fich_Im2d * ,INT x0,INT x1);
        void r_new_line(Fich_Im2d *,INT);


        Packed_LZW_Decompr_Flow  * _LZW_flow;
        Packed_LZW_Decompr_Flow  * _interlac_LF[4];
        bool                       _interlaced;
        INT                        _nb_lzw;
};



Gif_Tile_F2d::~Gif_Tile_F2d()
{
    for (INT i = 0; i < _nb_lzw ; i++)
        delete _interlac_LF[i];
}

void Gif_Tile_F2d::read_seg(Fich_Im2d *,void * buf,INT x0,INT x1)
{

     _LZW_flow->Read((U_INT1 *)buf,x1-x0);
}

void Gif_Tile_F2d::seek_in_line(Fich_Im2d * ,INT x0,INT x1)
{
     _LZW_flow->Rseek(x1-x0);
}

void Gif_Tile_F2d::r_new_line(Fich_Im2d *,INT y)
{
    if (_interlaced)
       switch (y % 8)
       {
           case 0 :
               _LZW_flow = _interlac_LF[0];
           break;

           case 4 :
               _LZW_flow = _interlac_LF[1];
           break;

           case 2 :
           case 6 :
               _LZW_flow = _interlac_LF[2];
           break;

           default :
               _LZW_flow = _interlac_LF[3];
           break;
       }
}


  // Herit of Tile_F2d(0) because :
  // do not want to flush automatically  _pfob (woul be more
  // complicated to flush the 3 other flow)

Gif_Tile_F2d::Gif_Tile_F2d(Data_GifIm * dgi) :
    Tile_F2d(0),
   _nb_lzw(0)
{
    _interlaced = dgi->_interlaced;
    _nb_lzw =  _interlaced ? 4 : 1;

     for (int ilzw = 0; ilzw < _nb_lzw ; ilzw++)
     {
          ELISE_fp  fp (dgi->_name,ELISE_fp::READ);
          fp.seek_begin(dgi->_offs_data);

          GIF_char_flow  * GCF = new GIF_char_flow(fp,true);
          _interlac_LF[ilzw] =
                new Packed_LZW_Decompr_Flow
                    (
                        GCF,
                        true,
                        false,
                        LZW_Protocols::gif,
                        dgi->_nb_bits_lzw_init
                    );
          for (int j = 0 ; j < ilzw ; j++)
          {
               for
               (
                   INT y =deb_interlac_gif[j];
                   y < dgi->sz().y       ;
                   y += per_interlac_gif[j]
               )
                   _interlac_LF[ilzw]->Rseek(dgi->sz().x);
          }
     }
     _LZW_flow = _interlac_LF[0];

}


Gif_Fich_Im2d::Gif_Fich_Im2d
(
        Flux_Pts_Computed * flx,
        Data_GifIm * dgi
) :
     Std_Bitm_Fich_Im_2d
     (
          flx,
          dgi->sz(),
          dgi->sz(),
          1,
          dgi->name(),
          alloc_im1d(GenIm::u_int1,flx->sz_buf()),
          true
     )//,
     //_dgi (dgi)
{
     init_tile(new Gif_Tile_F2d(dgi),0,1,true);
}


class Gif_Im_Not_Comp : public Fonc_Num_Not_Comp
{
     public :

         Gif_Im_Not_Comp(Gif_Im,bool,REAL);

     private :

        Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);

        Gif_Im   _gi;
        bool     _with_def_value;
        REAL     _def_value;

        virtual bool  integral_fonc (bool) const
        {return true;}

        virtual INT dimf_out() const {return 1;}

         void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}
};

Fonc_Num_Computed * Gif_Im_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{
      return
          fonc_num_std_f2d
          (
                arg,
                 new Gif_Fich_Im2d(arg.flux(),_gi.dgi()) ,
                _with_def_value,
                _def_value
          );
}

Gif_Im_Not_Comp::Gif_Im_Not_Comp(Gif_Im gi,bool wdf,REAL def_v) :
     _gi (gi),
     _with_def_value(wdf),
     _def_value(def_v)
{
}


Fonc_Num Gif_Im::in()
{
     return new Gif_Im_Not_Comp(*this,false,0.0);
}

Fonc_Num Gif_Im::in(INT dfv)
{
     return new Gif_Im_Not_Comp(*this,true,dfv);
}


/******************************************************/
/*                                                    */
/*                Data_GifIm                          */
/*                                                    */
/******************************************************/


Im2D<U_INT1,INT> Data_GifIm::im()
{
    Im2D<U_INT1,INT> I(_im_w,_im_h);
    U_INT1 **   i = I.data();
    ELISE_fp  fp (_name,ELISE_fp::READ);

    fp.seek_begin(_offs_data);
    GIF_char_flow  * GCF = new GIF_char_flow(fp,true);

    Packed_LZW_Decompr_Flow  * LZW_flow
       =  new Packed_LZW_Decompr_Flow
              (
                 GCF,
                 true,
                 false,
                 LZW_Protocols::gif,
                 _nb_bits_lzw_init
              );

    if (_interlaced)
    {
        for (INT k = 0; k<4 ; k++)
        {
           INT per = per_interlac_gif[k];
           INT deb = deb_interlac_gif[k];
           for (INT y =deb  ; y<_im_h ; y+= per)
               LZW_flow->Read(i[y],_im_w);
        }
    }
    else
    {
        for (INT y =0 ; y<_im_h ; y++)
        {
           for (int x = 0 ; x <_im_w ; x+= 5)
           {
               LZW_flow->Read(i[y]+x,ElMin(5,_im_w-x));
           }
        }
    }

    //  LZW_flow->Assert_end_code();
    //  suppresse this because some GIF file forget it

    delete LZW_flow;

    return I;
}



Data_GifIm::Data_GifIm(const char * name,ELISE_fp  fp,Data_Giff  * gh) :
     _pal (Disc_Pal::PBIN())
{
     _tprov_name   = dup_name_std(name);
     _name         = _tprov_name->coord();
     _left_pos   = fp.lsb_read_U_INT2();
     _top_pos    = fp.lsb_read_U_INT2();
     _im_w       = fp.lsb_read_U_INT2();
     _im_h       = fp.lsb_read_U_INT2();



      U_INT1 packed = fp.lsb_read_U_INT1();

     _loc_col_tab     =  (sub_bit(packed,7,8)!=0);
     _interlaced      =  (sub_bit(packed,6,7)!=0);


     if (_loc_col_tab)
     {
         _sorted_col_tab  =     (gh->_numv >= Data_Giff::v89a)
                             && sub_bit(packed,5,6);
         // [3-5[  reserved
         _sz_col_tab     =  1 << (sub_bit(packed,0,3)+1);
        _pal =  GIF_palette(fp,_sz_col_tab);
     }
     else
     {
           ASSERT_TJS_USER
           (
                gh->_glob_col_tab,
                "neither global nor local colour table in GIF file"
           );
           _sorted_col_tab = gh->_sorted_col_tab;
           _sz_col_tab     = gh->_sz_col_tab;
           _pal       = gh->_pal;
     }
     _nb_bits_lzw_init  =  fp.lsb_read_U_INT1();
     _offs_data = fp.tell();


     INT sz_exp = ElMax(4,_sz_col_tab);

     if ( sz_exp != (1<<_nb_bits_lzw_init))
        cout << _nb_bits_lzw_init << " bits in LZW coder \n";


     ASSERT_INTERNAL
     (
          sz_exp == (1<<_nb_bits_lzw_init),
          "incoherent number of LZW bits inits in GIF file"
     );

     INT SZ[2];
     sz().to_tab(SZ);
     ElDataGenFileIm::init
     (
          2,
          SZ,
          1,
          false,
          true,
          _nb_bits_lzw_init,
          SZ,
          true
     );
}
Fonc_Num Data_GifIm::in()
{
    return Gif_Im(this).in();
}
Fonc_Num Data_GifIm::in(REAL val)
{
    return Gif_Im(this).in((INT)val);
}
Output   Data_GifIm::out()
{
    Tjs_El_User.ElAssert
    (
         false,
         EEM0 << "Bad Assertion in Data_GifIm::out"
    );
    return Output::onul();
}

/******************************************************/
/*                                                    */
/*                Gif_Im                              */
/*                                                    */
/******************************************************/
Gif_Im::Gif_Im(Data_GifIm * dgi) :
     ElGenFileIm(dgi)
{
}

Gif_Im::Gif_Im(const char * name,ELISE_fp  fp,class Data_Giff *gh) :
     ElGenFileIm(new Data_GifIm(name,fp,gh))
{
}


Gif_Im::Gif_Im(const char * name) : ElGenFileIm(0)
{
     *this = Gif_File(name).kth_im(0);
}


Disc_Pal     Gif_Im::pal()
{
    return dgi()->_pal;
}

Im2D_U_INT1   Gif_Im::im()
{
    return dgi()->im();
}

Pt2di Gif_Im::sz()
{
    return dgi()->sz();
}

/******************************************************/
/*                                                    */
/*                GIF_char_flow                       */
/*                                                    */
/******************************************************/

void GIF_char_flow::read_pack()
{
    _nb_in_pack = _fp.fgetc();
    ASSERT_INTERNAL(_nb_in_pack != ELISE_fp::eof,"unexpected EOF in GIF");
    _i_pack = 0;
    _fp.read(_buf,sizeof(_buf[0]),_nb_in_pack);

}


GIF_char_flow::GIF_char_flow(ELISE_fp  fp,bool read) :
    _fp   (fp),
    _read (read)
{
     if (_read)
     {
        read_pack();
     }
     else
     {
        _nb_in_pack = 0;
     }
}



void GIF_char_flow::flush_out_put()
{
     if (_nb_in_pack)
     {
       _fp.write_U_INT1(_nb_in_pack);
       _fp.write(_buf,sizeof(_buf[0]),_nb_in_pack);
       _nb_in_pack = 0;
     }
}


void GIF_char_flow::Putc(U_INT1 c)
{
      _buf[_nb_in_pack++] = c;
      if (_nb_in_pack == 255)
         flush_out_put();
}

U_INT1 GIF_char_flow::Getc()
{
    if (_i_pack == _nb_in_pack)
       read_pack();

    ASSERT_INTERNAL
    (
        _nb_in_pack,
        "unexcpeted empty data stream in GIF FILE"
    );
    return _buf[_i_pack++];
}


GIF_char_flow::~GIF_char_flow()
{
    if (! _read)
    {
        flush_out_put();
       _fp.write_U_INT1(0); // null data block
       _fp.write_U_INT1(GIF_Global::Trailer);
    }
    _fp.close();
}

/*****************************************************************/
/*****************************************************************/

/*****************************************************************/
/*****************************************************************/
/*****************************************************************/
/*****************************************************************/
/*****************************************************************/

#if (ELISE_X11)
void test_gif(char * name,Video_Win W,Video_Display D)
{
    INT nb_col = 150;

    ELISE_fp  fp;
    if (! fp.ropen(name,true))
    {
        cout << "cannnot open : " << name << "\n";
        return;
    }
    fp.close();


    Gif_File gf (name);

    cout << "\n\n" << "[" << name << "]" <<  " NB im : " << gf.nb_im() << "\n\n";

for (INT kth = 0; kth < gf.nb_im(); kth++)
{
   Gif_Im  i = gf.kth_im(kth);
   Disc_Pal    p  = i.pal();
   Gray_Pal    GP(30);
   Im2D<U_INT1,INT> i2 = i.im();

   Pt2di sz = W.sz();
   REAL sc = ElMin(sz.x / (REAL) i2.tx(),sz.y / (REAL) i2.ty());




   Im1D_INT4 lut(p.nb_col());
   Video_Win W2 = W.chc(Pt2dr(0,0),Pt2dr(sc,sc));
   if (p.nb_col() < nb_col)
   {
      ELISE_COPY(lut.all_pts(),FX,lut.out());
   }
   else
   {
       p = p.reduce_col(lut,nb_col);
   }


   Elise_Set_Of_Palette SOP(ElList<Elise_Palette>()+Elise_Palette(p)+Elise_Palette(GP));
   W.set_sop(SOP);
   D.load(SOP);
   if (kth == 0)
      ELISE_COPY(W.all_pts(),128,W.ogray());
   ELISE_COPY(i2.all_pts(),lut.in()[i2.in(0)],W2.out(p));
}
    getchar();
}
#endif


class GIF_Out_Comp :  public Output_Computed
{
      public :

         GIF_Out_Comp
         (
             char *             name,
             Pt2di              sz,
             Elise_colour *     tec,
             INT                nbb
         );
         ~GIF_Out_Comp();

      private :

         virtual void update( const Pack_Of_Pts * pts,
                              const Pack_Of_Pts * vals);

         Packed_LZW_Decompr_Flow * _lzw_fl;
};

GIF_Out_Comp::GIF_Out_Comp
(
    char *             name,
    Pt2di              sz,
    Elise_colour *     tec,
    INT                nbb
)   :
     Output_Computed (1)
{
    INT       nb_coul = 1<<nbb;
    ELISE_fp  fp (name,ELISE_fp::WRITE);

    fp.write("GIF89a",1,6);

    fp.lsb_write_U_INT2(sz.x);
    fp.lsb_write_U_INT2(sz.y);

    {
       U_INT1 packed =0;

       packed = set_sub_bit(packed,nbb-1  ,0,3); // _sz_col_tab
       packed = set_sub_bit(packed,0      ,3,4); // _sorted_col_tab
       packed = set_sub_bit(packed,7      ,4,7); // _nb_bit_per_col
       packed = set_sub_bit(packed,1      ,7,8); // _glob_col_tab
       fp.write_U_INT1(packed);
    }

    fp.write_U_INT1(0);             // _index_back
    fp.write_U_INT1(49);            // _aspect_ratio of 1.0

    for (INT i=0; i<nb_coul ; i++)
    {
         U_INT1 c[3];
         c[0] = round_ni(tec[i].r() * 255);
         c[1] = round_ni(tec[i].g() * 255);
         c[2] = round_ni(tec[i].b() * 255);
         fp.write(c,1,3);
    }


    fp.write_U_INT1(GIF_Global::Image_Separator);


    fp.lsb_write_U_INT2(0);
    fp.lsb_write_U_INT2(0);
    fp.lsb_write_U_INT2(sz.x);
    fp.lsb_write_U_INT2(sz.y);

    {
       U_INT1 packed = 0;

       packed = set_sub_bit(packed,0 ,7,8); // no local colour table
       packed = set_sub_bit(packed,0 ,6,7); // not interlaced

       fp.write_U_INT1(packed);
    }


    INT nbb_init = ElMax(4,nbb);
    fp.write_U_INT1(nbb_init);     // nb bits lzw init

    GIF_char_flow  * GCF = new GIF_char_flow(fp,false);

    _lzw_fl = new Packed_LZW_Decompr_Flow
                  (
                      GCF,
                      false,
                      false,
                      LZW_Protocols::gif,
                      nbb_init
                  );
}

GIF_Out_Comp::~GIF_Out_Comp()
{
    _lzw_fl->reset();
    delete _lzw_fl;
}



void GIF_Out_Comp::update
     (
         const Pack_Of_Pts * pts,
         const Pack_Of_Pts * vals
     )
{
    const Std_Pack_Of_Pts<INT> * ivals = vals->std_cast((INT *)0);
    _lzw_fl->Write(ivals->_pts[0],pts->nb());
}


class GIF_Out_NotComp : public  Output_Not_Comp
{
      public :
         GIF_Out_NotComp
         (
             const char *       name,
             Pt2di              sz,
             Elise_colour *     tec,
             INT                nbb
         ) ;

        ~GIF_Out_NotComp()
        {
             delete _tprov_name;
             STD_DELETE_TAB(_cols);
        }

      private :
         virtual  Output_Computed * compute(const Arg_Output_Comp &);

         Tprov_char *     _tprov_name;
         char *           _name;
         Elise_colour *   _cols;
         Pt2di            _sz;
         INT              _nbb;
};


GIF_Out_NotComp::GIF_Out_NotComp
(
   const char *       name,
   Pt2di              sz,
   Elise_colour *     tec,
   INT                nbb
)
{
    _tprov_name   = dup_name_std(name);
    _name         = _tprov_name->coord();
    _sz           = sz;
    _nbb           = ElMax(4,nbb);
    
#if (ELISE_windows & !ELISE_MinGW)
    _cols        = NEW_TAB(1i64<<_nbb,Elise_colour);
    memcpy(_cols,tec,sizeof(Elise_colour)*(1i64<<nbb));
#else
    _cols        = NEW_TAB(1<<_nbb,Elise_colour);
    memcpy(_cols,tec,sizeof(Elise_colour)*(1<<nbb));
#endif

    for (INT k=(1<<nbb);  k<(1<<_nbb); k++)
         _cols[k] = Elise_colour::rgb(0,0,0);
}



Output_Computed * GIF_Out_NotComp::compute(const Arg_Output_Comp & arg)
{
    Box2di b(Pt2di(0,0),Pt2di(0,0));

    bool ok = arg.flux()->is_rect_2d(b);
    Tjs_El_User.ElAssert
    (
         ok ,
         EEM0 << "Non rectangular Flux for Gif file\n"
              << "|   File = " << _name
    );
    Tjs_El_User.ElAssert
    (
         ok && (b._p0 == Pt2di(0,0)) && (b._p1 == _sz),
         EEM0 << "Bad rectangle for gif file"
              << "|   File = " << _name  << "\n"
              << "|   rect = " <<  b
               << "; expecting" << Box2di(Pt2di(0,0),_sz) << "\n"
    );

    return out_adapt_type_fonc
           (
               arg,
               new GIF_Out_Comp(_name,_sz,_cols,_nbb),
               Pack_Of_Pts::integer
           );
}



Output  Gif_Im::create
        (
            const char *       name,
            Pt2di              sz,
            Elise_colour *     tec,
            INT                nbb
        )
{
     return new  GIF_Out_NotComp (name,sz,tec,nbb);
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
