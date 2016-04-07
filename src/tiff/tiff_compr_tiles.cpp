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

#include <cstring>


/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***                   Tiff_Tiles_Cpr                                ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/


class Tiff_Tiles_Cpr : public Tiff_Tiles
{
     public :
          Tiff_Tiles_Cpr(bool bin_up);

          void seek_in_line(Fich_Im2d *,INT x0,INT x1);
          void seek_pack_line(Fich_Im2d * fich,INT y0,INT y1,bool read_mode);
          virtual  void use_this_tile(Fich_Im2d *,bool read);

          //===============================================
          // method to be redefined in inheriting classes
          //===============================================

                  //  READ MODE

          virtual  void compr_r_use_this_tile (Tiff_file_2d *)
          {};

          virtual  void compr_w_use_this_tile (Tiff_file_2d *)
          {};

          virtual  void compr_r_new_tile(Tiff_file_2d *)
          {} ;

          virtual  void compr_w_new_tile(Tiff_file_2d *)
          {} ;

          virtual  void compr_r_end_tile(Tiff_file_2d *)
          {} ;

          virtual  void compr_w_end_tile(Tiff_file_2d *)
          {} ;



          //===============================================
          // method redefinition of Tiff_Tiles
          //===============================================

                  //  READ MODE


          virtual  void r_use_this_tile(class Fich_Im2d *);
          virtual  void r_new_tile(class Fich_Im2d *) ;
          virtual  void r_new_line(Fich_Im2d * f2d,INT);
          void     read_seg(Fich_Im2d *,void * buf,INT x0,INT x1);
          virtual  void r_end_tile(Fich_Im2d * f2d);
          virtual void  UcompLine
                        (
                            Packed_Flux_Of_Byte * pfob,
                            U_INT1 * res,
                            INT nb_byte,
                            INT nb_el
                        ) = 0;

                  //  WRITE MODE

          virtual  void w_use_this_tile(class Fich_Im2d *);
          virtual  void w_new_tile(class Fich_Im2d *) ;
          void     write_seg(Fich_Im2d *,void * buf,INT x0,INT x1);
          virtual  void w_end_line(Fich_Im2d * f2d,INT);
          virtual  void w_end_tile(Fich_Im2d * f2d);


          virtual void CompLine
                       (
                            Packed_Flux_Of_Byte * pfob,
                            const U_INT1 * line,
                            INT nb_byte,
                            INT nb_el
                       ) =0;
          U_INT1 * _packed;
          U_INT1 * _un_packed;
          U_INT1 ** _matr_packed;
          DATA_Tiff_Ifd *  _dti;

          bool             _bin_up;  // binary unpacked


          virtual ~Tiff_Tiles_Cpr();
          GenIm           _gi;
          tFileOffset     _offs_deb;
          INT _tx;
};



          /*********************************************/
          /*   Commom to read and write                */
          /*********************************************/

Tiff_Tiles_Cpr::Tiff_Tiles_Cpr(bool bin_up) :
       Tiff_Tiles    (),
       _packed       (0),
       _un_packed    (0),
       _matr_packed  (0),
       _dti          (0),
       _bin_up       (bin_up),
       _gi           (alloc_im1d(GenIm::u_int1,1)),
       _offs_deb     (0)   // just to initialize everything
{
}



Tiff_Tiles_Cpr::~Tiff_Tiles_Cpr()
{
     if (_un_packed)
     {
         if (_packed != _un_packed)
            STD_DELETE_TAB_USER(_packed);
         STD_DELETE_TAB_USER(_un_packed);

     }
     if (_matr_packed)
         DELETE_MATRICE_ORI
         (
             _matr_packed,
             _dti->_line_byte_sz_tiles.CKK_Byte4AbsLLO(),
             _dti->_sz_tile.y
         );
}


void Tiff_Tiles_Cpr::use_this_tile(class Fich_Im2d * f2d,bool read_mode)
{
     Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
     DATA_Tiff_Ifd * dti = tf2d->_dti;

     _dti = dti;
     if (read_mode)
        _pfob = Tiff_Tiles::init_pfob(dti,tf2d,read_mode);

     _tx = (dti->_sz_tile.x+7+dti->_nb_chan_per_tile) *dti->_sz_byte_pel_unpacked;

     _un_packed = STD_NEW_TAB_USER(_tx,U_INT1);
     MEM_RAZ(_un_packed,_tx);

     if (dti->_nbb_ch0 < 8)
         _packed = STD_NEW_TAB_USER(dti->_line_byte_sz_tiles.CKK_Byte4AbsLLO(),U_INT1);
     else
       _packed = _un_packed;

     _gi = alloc_im1d(GenIm::u_int1,_tx,_un_packed);

     if ((!read_mode) && (!tf2d->_single_tile))
     {
         _matr_packed = NEW_MATRICE_ORI
                        (
                            dti->_line_byte_sz_tiles.CKK_Byte4AbsLLO(),
                            dti->_sz_tile.y,
                            U_INT1
                        );
     }
}

void Tiff_Tiles_Cpr::seek_pack_line(Fich_Im2d * f2d,INT y0,INT y1,bool read_mode)
{
     Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
     DATA_Tiff_Ifd * dti = tf2d->_dti;

     for (INT y=y0; y<y1; y++)
     {
         El_Internal.ElAssert
         (
            read_mode,
            EEM0 << "Tiff_Tiles_Cpr::seek_pack_line in write mode"
         );
         UcompLine
         (
              _pfob,
              0,
              dti->_line_byte_sz_tiles.CKK_Byte4AbsLLO(),
              dti->_line_el_sz_tiles.CKK_Byte4AbsLLO()
         );
     }
}

void Tiff_Tiles_Cpr::seek_in_line(Fich_Im2d *,INT ,INT )
{
}

          /*********************************************/
          /*             WRITE                         */
          /*********************************************/

void Tiff_Tiles_Cpr::w_use_this_tile(class Fich_Im2d * f2d)
{
    use_this_tile(f2d,false);
    Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);

    tf2d->init_stdpf();
    compr_w_use_this_tile(SAFE_DYNC(Tiff_file_2d *,f2d));
}

void Tiff_Tiles_Cpr::w_new_tile(class Fich_Im2d * f2d)
{
    Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);

    DATA_Tiff_Ifd * dti = tf2d->_dti;

    if (! _matr_packed)
    {
       _offs_deb = tf2d->_fp.tell();
       dti->set_offs_tile
       (
           tf2d->_fp,
           _n_tile,
           _last_til_Y,
           tf2d->_kth_ch,
           _offs_deb
       );
   }
   compr_w_new_tile(tf2d);
}

void   Tiff_Tiles_Cpr::write_seg(Fich_Im2d * f2d,void * buf,INT x0,INT x1)
{


     Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
     DATA_Tiff_Ifd * dti = tf2d->_dti;

     memcpy
     (
          _un_packed+x0*dti->_sz_byte_pel_unpacked,
          buf,
          (x1-x0)*dti->_sz_byte_pel_unpacked
     );
}


void Tiff_Tiles_Cpr::w_end_line(Fich_Im2d * f2d,INT y)
{
     Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
     DATA_Tiff_Ifd * dti = tf2d->_dti;


     {
          INT x0 = _sz_tile_log  * dti->_sz_byte_pel_unpacked;
          INT x1 = _sz_tile_phys * dti->_sz_byte_pel_unpacked;

          INT v0 = (x0 > 0) ? _un_packed[x0-1] : 0;
          for (INT x = x0; x < x1 ; x++)
              _un_packed[x] = v0;
     }


     if (dti->_predict == Tiff_Im::Hor_Diff)
        _gi.tiff_predictor
        (
             dti->_sz_tile.x,
             dti->_nb_chan_per_tile,
             1<<dti->_nbb_ch0,
             true
        );

    if (_matr_packed)
    {
        if (_packed != _un_packed)
        {
           Tabul_Bits_Gen::pack
           (
                 _matr_packed[y],
                 _un_packed,
                 dti->_line_el_sz_tiles.CKK_IntBasicLLO(),
                 dti->_nbb_ch0,
                 dti->_msbit_first
           );
        }
        else
        {
           // MODIF 26/12/00
           // convert(_matr_packed[y],_un_packed,dti->_line_el_sz_tiles);
           convert(_matr_packed[y],_un_packed,dti->_line_byte_sz_tiles.CKK_IntBasicLLO());
        }
    }
    else
    {
        if (_bin_up)
        {
           CompLine
           (
                    tf2d->_stdpf,
                    _un_packed,
                    dti->_line_byte_sz_tiles.CKK_IntBasicLLO(),
                    dti->_line_el_sz_tiles.CKK_IntBasicLLO()
            );
        }
        else
        {
             if (_packed != _un_packed)
             {
                 Tabul_Bits_Gen::pack
                 (
                     _packed,
                     _un_packed,
                     dti->_line_el_sz_tiles.CKK_IntBasicLLO(),
                     dti->_nbb_ch0,
                     dti->_msbit_first
                 );
              }
              CompLine
              (
                    tf2d->_stdpf,
                    _packed,
                    dti->_line_byte_sz_tiles.CKK_IntBasicLLO(),
                    dti->_line_el_sz_tiles.CKK_IntBasicLLO()
              );
         }
    }
}

void Tiff_Tiles_Cpr::w_end_tile(Fich_Im2d * f2d)
{
    Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
    DATA_Tiff_Ifd * dti = tf2d->_dti;

    INT end_y = dti->_tiled ? dti->_sz_tile.y : (_last_y+1);
    // Has for effect of enventually repeat las line
    for (INT y = _last_y+1; y <end_y ; y++)
         w_end_line(f2d,y);

    if (_matr_packed)
    {
       _offs_deb = tf2d->_fp.tell();
       dti->set_offs_tile
       (
           tf2d->_fp,
           _n_tile,
           _last_til_Y,
           tf2d->_kth_ch,
           _offs_deb
       );



       for (INT y=0; y <end_y ; y++)
           if (_bin_up)
           {
                if (_packed != _un_packed)
                {
                    Tabul_Bits_Gen::unpack
                    (
                          _un_packed,
                          _matr_packed[y],
                          dti->_line_el_sz_tiles.CKK_IntBasicLLO(),
                          dti->_nbb_ch0,
                          dti->_msbit_first
                    );
                    CompLine
                    (
                        tf2d->_stdpf,
                        _un_packed,
                        dti->_line_byte_sz_tiles.CKK_IntBasicLLO(),
                        dti->_line_el_sz_tiles.CKK_IntBasicLLO()
                    );
                }
                else
                {
                    CompLine
                    (
                        tf2d->_stdpf,
                        _matr_packed[y],
                        dti->_line_byte_sz_tiles.CKK_IntBasicLLO(),
                        dti->_line_el_sz_tiles.CKK_IntBasicLLO()
                    );
                }
           }
           else
           {
                CompLine
                (
                    tf2d->_stdpf,
                    _matr_packed[y],
                    dti->_line_byte_sz_tiles.CKK_IntBasicLLO(),
                    dti->_line_el_sz_tiles.CKK_IntBasicLLO()
                );
           }
    }

    compr_w_end_tile(tf2d);

    dti->set_count_tile
    (
        tf2d->_fp,
        _n_tile,
        _last_til_Y,
        tf2d->_kth_ch,
        tf2d->_fp.tell()-_offs_deb
    );
}


          /*********************************************/
          /*             READ                          */
          /*********************************************/

void Tiff_Tiles_Cpr::r_use_this_tile(class Fich_Im2d * f2d)
{
    use_this_tile(f2d,true);
    compr_r_use_this_tile(SAFE_DYNC(Tiff_file_2d *,f2d));
}

void Tiff_Tiles_Cpr::r_new_tile(class Fich_Im2d * f2d)
{
    Tiff_Tiles::new_tile(f2d,true);
    compr_r_new_tile(SAFE_DYNC(Tiff_file_2d *,f2d));
}


void Tiff_Tiles_Cpr::r_new_line(Fich_Im2d * f2d,INT)
{
     Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
     DATA_Tiff_Ifd * dti = tf2d->_dti;



     if (_bin_up)
     {
         UcompLine
         (
            _pfob,
            _un_packed,
            dti->_line_byte_sz_tiles.CKK_Byte4AbsLLO(),
            dti->_line_el_sz_tiles.CKK_Byte4AbsLLO()
         );
     }
     else
     {
         UcompLine
         (
            _pfob,
            _packed,
            dti->_line_byte_sz_tiles.CKK_Byte4AbsLLO(),
            dti->_line_el_sz_tiles.CKK_Byte4AbsLLO()
         );
         if (_packed != _un_packed)
         {
             Tabul_Bits_Gen::unpack
             (
                 _un_packed,
                 _packed,
                 dti->_line_el_sz_tiles.CKK_IntBasicLLO(),
                 dti->_nbb_ch0,
                 dti->_msbit_first
             );
          }
      }

      if (dti->_predict == Tiff_Im::Hor_Diff)
         _gi.tiff_predictor
         (
                dti->_sz_tile.x,
                dti->_nb_chan_per_tile,
                1<<dti->_nbb_ch0,
                false
         );
}


void   Tiff_Tiles_Cpr::read_seg(Fich_Im2d * f2d,void * buf,INT x0,INT x1)
{
     Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
     DATA_Tiff_Ifd * dti = tf2d->_dti;

     memcpy
     (
          buf,
          _un_packed+x0*dti->_sz_byte_pel_unpacked,
          (x1-x0)*dti->_sz_byte_pel_unpacked
     );
}

void Tiff_Tiles_Cpr::r_end_tile(Fich_Im2d * f2d)
{
    Tiff_file_2d * tf2d = SAFE_DYNC(Tiff_file_2d *,f2d);
    compr_r_end_tile(tf2d);
}


/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***                   Tiff_Tiles_PckBits                            ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/


class Tiff_Tiles_PckBits : public Tiff_Tiles_Cpr
{
     public :

         Tiff_Tiles_PckBits (INT NbByteEl) :
              Tiff_Tiles_Cpr(false) ,
              mNbByteEl(NbByteEl)
         {
         };

         void UcompLine
              (
                     Packed_Flux_Of_Byte * pfob ,
                     U_INT1 * res,
                     INT nb_byte,
                     INT nb_el
              )
         {
              if (mNbByteEl==1)
                 PackBitsUCompr(pfob,res,nb_byte);
              else if (mNbByteEl==2)
                 PackBitsUCompr_B2(pfob,res,nb_el);
              else
                 Tjs_El_User.ElAssert(0,EEM0<<"Bad Nb Byte in Tiff_Tiles_PckBits");


         }
         void CompLine
              (
                   Packed_Flux_Of_Byte * pfob,
                   const U_INT1 * line,
                   INT nb_byte,
                   INT nb_el
               )
         {
             if (mNbByteEl==1)
                PackBitsCompr(pfob,line,nb_byte);
             else if (mNbByteEl==2)
                PackBitsCompr_B2(pfob,line,nb_el);
             else
               Tjs_El_User.ElAssert(0,EEM0<<"Bad Nb Byte in Tiff_Tiles_PckBits");
         }


         INT mNbByteEl;
};


/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***                   Tiff_LZW                                      ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/


class Tiff_Tiles_LZW : public Tiff_Tiles_Cpr
{
     public :
        Tiff_Tiles_LZW () : Tiff_Tiles_Cpr(false), _lzw_pack(0) {};
        virtual  ~Tiff_Tiles_LZW()
        {
              if(_lzw_pack) delete _lzw_pack;
        }

     private :

        void compr_r_use_this_tile (Tiff_file_2d *  tf2d)
        {
            _lzw_pack = new Packed_LZW_Decompr_Flow
                            (
                                 new UnPacked_FOB(_pfob,false),
                                 true,
                                 tf2d->_dti->_msbit_first,
                                 LZW_Protocols::tif,
                                 8
                             );
        }
        void compr_w_use_this_tile (Tiff_file_2d *  tf2d)
        {
            _lzw_pack = new Packed_LZW_Decompr_Flow
                            (
                                 new UnPacked_FOB
                                     (tf2d->_stdpf,false),
                                 false,
                                 tf2d->_dti->_msbit_first,
                                 LZW_Protocols::tif,
                                 8
                            );
        }


        virtual  void compr_r_new_tile(Tiff_file_2d *)
        {
             _lzw_pack->reset();
        }
        virtual   void compr_w_end_tile(Tiff_file_2d *)
        {
             _lzw_pack->reset();
        }




        void UcompLine
              (
                     Packed_Flux_Of_Byte *,
                     U_INT1 * res,
                     INT nb_byte,
                     INT
              )
        {
             _lzw_pack->Read(res,nb_byte);
        }

        void CompLine
             (
                 Packed_Flux_Of_Byte * ,
                 const U_INT1 * res,
                 INT nb_byte,
                 INT
             )
        {
            _lzw_pack->Write(res,nb_byte);
        }


        Packed_LZW_Decompr_Flow * _lzw_pack;
};

/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***                   Tiff_Tiles_Ccit3_1D                           ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/


class Tiff_Tiles_Ccit3_1D : public Tiff_Tiles_Cpr
{
     public :

         Tiff_Tiles_Ccit3_1D ():
                  Tiff_Tiles_Cpr(true),
                  _hflx(0)
         {}


         Huff_Ccitt_1D_Codec * _hflx;

         void compr_r_use_this_tile (Tiff_file_2d *  tf2d)
         {
            _hflx = new Huff_Ccitt_1D_Codec
                        (
                                 _pfob,
                                 true,
                                 tf2d->_dti->_msbit_first,
                                 false
                        );
         }

         void UcompLine(Packed_Flux_Of_Byte *,U_INT1 * res,INT,INT nb_el)
         {
              _hflx->read(res,nb_el);
         }

         void compr_w_use_this_tile (Tiff_file_2d *  tf2d)
         {
            _hflx = new Huff_Ccitt_1D_Codec
                        (
                                 tf2d->_stdpf,
                                 false,
                                 tf2d->_dti->_msbit_first,
                                 false
                        );
         }
         void CompLine(Packed_Flux_Of_Byte *,const U_INT1 * im,INT,INT nb_el)
         {
              _hflx->write(im,nb_el);
         }

         ~Tiff_Tiles_Ccit3_1D() {if (_hflx) delete _hflx;}


};


/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***                   Tiff_Tiles_Ccit4_2D_T6                        ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/


class Tiff_Tiles_Ccit4_2D_T6 : public Tiff_Tiles_Cpr
{
     public :

         Tiff_Tiles_Ccit4_2D_T6 ():
                  Tiff_Tiles_Cpr(true),
                  _hflx(0)
         {}


         Huff_Ccitt_2D_T6 * _hflx;

         void compr_r_use_this_tile (Tiff_file_2d *  tf2d)
         {
            _hflx = new Huff_Ccitt_2D_T6
                        (
                                 _pfob,
                                 true,
                                 tf2d->_dti->_msbit_first,
                                 false,
                                 tf2d->_dti->_sz_tile.x
                        );
         }

         void UcompLine
              (
                  Packed_Flux_Of_Byte *,
                  U_INT1 * res,
                  INT,
                  INT /* nb_el */
              )
         {
              _hflx->read(res);
         }

         void compr_r_new_tile(Tiff_file_2d * tf2d)
         {
              _hflx->new_block( tf2d->_dti->_sz_tile.x );
         }

         void compr_r_end_tile(Tiff_file_2d * tf2d)
         {
              _hflx->end_block(_last_y== tf2d->_dti->_sz_tile.y);
         }





         void compr_w_new_tile(Tiff_file_2d * tf2d)
         {
              _hflx->new_block( tf2d->_dti->_sz_tile.x );
         }

         void compr_w_end_tile(Tiff_file_2d *)
         {
              _hflx->end_block(true);
         }

         void compr_w_use_this_tile (Tiff_file_2d *  tf2d)
         {
              _hflx = new Huff_Ccitt_2D_T6
                          (
                                 tf2d->_stdpf,
                                 false,
                                 tf2d->_dti->_msbit_first,
                                 false,
                                 tf2d->_dti->_sz_tile.x
                          );
         }

         void CompLine(Packed_Flux_Of_Byte *,const U_INT1 * vals,INT,INT)
         {
              _hflx->write(vals);
         }



          ~Tiff_Tiles_Ccit4_2D_T6() {if (_hflx) delete _hflx;}


};


/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***                   Tiff_Tiles_MPD_T6                             ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/


class Tiff_Tiles_MPD_T6 : public Tiff_Tiles_Cpr
{
     public :

         Tiff_Tiles_MPD_T6 ():
                  Tiff_Tiles_Cpr(true),
                  _hflx(0)
         {}


         MPD_CCIT_T6 * _hflx;

         void compr_r_use_this_tile (Tiff_file_2d *  tf2d)
         {
            _hflx = new MPD_CCIT_T6
                        (
                                 _pfob,
                                 true,
                                 tf2d->_dti->_msbit_first,
                                 false,
                                 tf2d->_dti->_sz_tile.x,
                                 tf2d->_dti->_nbb_ch0
                        );
         }

         void UcompLine
              (
                  Packed_Flux_Of_Byte *,
                  U_INT1 * res,
                  INT,
                  INT /* nb_el */
              )
         {
              _hflx->read(res);
         }

         void compr_r_new_tile(Tiff_file_2d * tf2d)
         {
              _hflx->new_block( tf2d->_dti->_sz_tile.x );
         }

         void compr_r_end_tile(Tiff_file_2d * tf2d)
         {
              _hflx->end_block(_last_y== tf2d->_dti->_sz_tile.y);
         }





         void compr_w_new_tile(Tiff_file_2d * tf2d)
         {
              _hflx->new_block( tf2d->_dti->_sz_tile.x );
         }

         void compr_w_end_tile(Tiff_file_2d *)
         {
              _hflx->end_block(true);
         }

         void compr_w_use_this_tile (Tiff_file_2d *  tf2d)
         {
              _hflx = new MPD_CCIT_T6
                          (
                                 tf2d->_stdpf,
                                 false,
                                 tf2d->_dti->_msbit_first,
                                 false,
                                 tf2d->_dti->_sz_tile.x,
                                 tf2d->_dti->_nbb_ch0
                          );
         }

         void CompLine(Packed_Flux_Of_Byte *,const U_INT1 * vals,INT,INT)
         {
              _hflx->write(vals);
         }



         ~Tiff_Tiles_MPD_T6() {if (_hflx) delete _hflx;}


};


/***********************************************************************/
/***********************************************************************/
/***********************************************************************/
/***********************************************************************/


Tiff_Tiles * Tiff_file_2d::alloc_tile()
{
     switch(_dti->_mode_compr)
     {

          case Tiff_Im::No_Compr:
               return new Tiff_Tiles_NC();

          case Tiff_Im::PackBits_Compr:
               return new Tiff_Tiles_PckBits(1);

          case Tiff_Im::NoByte_PackBits_Compr:
               return new Tiff_Tiles_PckBits(_dti->_nbb_ch0/8);

          case Tiff_Im::LZW_Compr:
               return new Tiff_Tiles_LZW();

          case Tiff_Im::CCITT_G3_1D_Compr:
               return new Tiff_Tiles_Ccit3_1D();

          case Tiff_Im::Group_4FAX_Compr:
               return new Tiff_Tiles_Ccit4_2D_T6();

          case Tiff_Im::MPD_T6:
               return new Tiff_Tiles_MPD_T6();
     }


     Tjs_El_User.ElAssert
     (
        0,
        EEM0 << "Compression mode not handled by Elise \n"
             << "|  FILE : " <<  _name
             << "|  compr: " << Tiff_Im::name_compr(_dti->_mode_compr)
    );
    return 0;
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
