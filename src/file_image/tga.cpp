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


class Data_TGA_File : public RC_Object
{
      friend class Tga_Im;
      friend class TGA_tile;
      friend class Tga_Fich_Im2d;
      friend class TGA_RLE_Flx_byte;
      friend class Tga_Im_Not_Comp;

      public :

          Pt2di sz() {return Pt2di(_szx,_szy);}
          const char * name() {return _name;}
          INT  nb_chanel() const;

          const char * why_cant_use_it();

          INT nb_byte_per_pix () const {return _nb_byte_pix;}
          tFileOffset offset_image() const {return _offs_im;}

          bool  top_to_down   () const   {return _top_to_down;}
          bool  left_to_right () const   {return _left_to_right;}


          Std_Bitm_Fich_Im_2d::r_special_transf  r_sptr() const
          {
             return _r_spec_transf;
          }

          Std_Bitm_Fich_Im_2d::w_special_transf  w_sptr() const
          {
             return _w_spec_transf;
          }

          virtual ~Data_TGA_File(){delete _tprov_name;}

      private :

          Data_TGA_File(const char * name);


          Tprov_char *     _tprov_name;
          char *           _name;


          INT              _id_length;

          bool             _cmap_present;

          bool                     _im_present;
          Tga_Im::mode_compr       _im_compr;
          Tga_Im::type_of_image    _im_type;


          INT              _cm_first_entr_ind;
          INT              _cm_length;
          INT              _cm_entr_sz;


          INT              _x0;
          INT              _y0;
          INT              _szx;
          INT              _szy;

          INT              _pix_depth;
          INT              _nb_byte_pix;
          INT              _alpha_channel; // I do not use this for now
          bool             _top_to_down;
          bool             _left_to_right;

          tFileOffset              _offs_pal;
          tFileOffset              _offs_im;

          Std_Bitm_Fich_Im_2d::r_special_transf  _r_spec_transf;
          Std_Bitm_Fich_Im_2d::w_special_transf  _w_spec_transf;

};


static const char * (Name_tga_compr)[2] = {"No compr", "Rle compr"};
static const char * (Name_tga_ty_im)[3] = {"Col maped", "True color","Black & White"};

/****************************************************************************************************/
/****************************************************************************************************/
/****************************************************************************************************/
/****************************************************************************************************/

void tga_16_r_spec_transf(Std_Pack_Of_Pts_Gen * pts,const void * buf)
{
     INT ** v     = (INT **) (pts->adr_coord());
     INT * r = v[0];
     INT * g = v[1];
     INT * b = v[2];

     U_INT2 * tga = reinterpret_cast<U_INT2 *> (const_cast<void *>(buf));
     INT nb       = pts->nb();

     for (INT i=0 ; i<nb ; i++)
     {
          r[i] = (  ((tga[i]) &      (0x1F)) * 255  ) / 0x1F;
          g[i] = (  ((tga[i] >> 5) & (0x1F)) * 255  ) / 0x1F;
          b[i] = (  ((tga[i] >> 10)& (0x1F)) * 255  ) / 0x1F;
     }
}
void tga_16_w_spec_transf(const Std_Pack_Of_Pts_Gen * pts,void * buf)
{
     const INT * const * v     =  pts->std_cast((INT *)0)->_pts;
     const INT * r = v[0];
     const INT * g = v[1];
     const INT * b = v[2];

     U_INT2 * tga = (U_INT2 *) buf;
     INT nb       = pts->nb();

     for (INT i=0 ; i<nb ; i++)
         tga[i] =
                     ((r[i] * 0x20) / 256)
                |   (((g[i] * 0x20) / 256) << 5 )
                |   (((b[i] * 0x20) / 256) << 10);
}

void tga_32_r_spec_transf(Std_Pack_Of_Pts_Gen * pts,const void * buf)
{
     INT ** v     = (INT **) (pts->adr_coord());
     INT * r = v[0];
     INT * g = v[1];
     INT * b = v[2];

     U_INT1 * tga = reinterpret_cast<U_INT1* >(const_cast<void *> (buf));
     INT nb       = pts->nb();

     for (INT i=0 , i4 =0; i<nb ; i++ , i4+=4)
     {
          r[i] = tga[i4+0];
          g[i] = tga[i4+1];
          b[i] = tga[i4+2];
     }
}
void tga_32_w_spec_transf(const Std_Pack_Of_Pts_Gen * pts ,void * buf)
{
     const INT * const * v     =  pts->std_cast((INT *)0)->_pts;
     const INT * r = v[0];
     const INT * g = v[1];
     const INT * b = v[2];

     U_INT1 * tga = (U_INT1 *) buf;
     INT nb       = pts->nb();

     for (INT i=0 , i4 =0; i<nb ; i++ , i4+=4)
     {
          tga[i4+0] = r[i];
          tga[i4+1] = g[i];
          tga[i4+2] = b[i];
          tga[i4+3] = 0;
     }
}



     /*------------------------------------------------------------------*/
     /*                                                                  */
     /*           Data_TGA_File                                          */
     /*                                                                  */
     /*------------------------------------------------------------------*/


INT  Data_TGA_File::nb_chanel() const
{
     switch(_im_type)
     {
          case Tga_Im::true_col   :
               return 3;
          break;

          case   Tga_Im::bw_image :
                 return 1;
          break;

          default : return -1;
     }
}


const char * Data_TGA_File::why_cant_use_it()
{
     if (! _im_present)
        return "no image in this TGA file";


     switch(_im_type)
     {
           case  Tga_Im::col_maped :
                 return "do not handle TGA-color maped image";
           break;


           case Tga_Im::true_col :

                if (
                          (_pix_depth != 15)
                     &&   (_pix_depth != 16)
                     &&   (_pix_depth != 24)
                     &&   (_pix_depth != 32)
                   )
                   return "handle only 15,16, 24 or 32 bits TGA-image in true color";
           break;

          case Tga_Im::bw_image :

                if (
                          (_pix_depth != 1)
                      &&  (_pix_depth != 2)
                      &&  (_pix_depth != 4)
                      &&  (_pix_depth != 8)
                   )
                   return "handle only 1,2,4 or 8 bits TGA-image in black&white mode";
           break;
     }

     switch(_im_compr)
     {
          case Tga_Im::no_compr:
          case Tga_Im::rle_compr:
          break;

     };

     return (char *) 0;
}






Data_TGA_File::Data_TGA_File(const char * name)
{
    _tprov_name   = dup_name_std(name);
    _name         = _tprov_name->coord();

    ELISE_fp fp (name,ELISE_fp::READ);

   // Field 1 : Id length

    _id_length = fp.lsb_read_U_INT1();


   // Field 2 :  Color Map Type

     switch (fp.lsb_read_U_INT1())
     {
               case  0 : _cmap_present = false; break;
               case  1 : _cmap_present = true; break;
               default :  elise_fatal_error
                          (
                              "unknown Cmap presence specifier in TGA file",
                              __FILE__,__LINE__
                          );

     };


   // Field 3 :  Image Type

    _im_present = true;
	INT imt;
     switch (imt = fp.lsb_read_U_INT1())
     {
            case 0 :  _im_present = false; break;

            case 1: case 2: case 3:
                 _im_compr = Tga_Im::no_compr;
                 _im_type = (Tga_Im::type_of_image) (imt-1);
            break;

            case 9 : case 10 : case 11 :
                 _im_compr = Tga_Im::rle_compr;
                 _im_type = (Tga_Im::type_of_image) (imt-9);
            break;
                 
            default :  elise_fatal_error
                        (
                          "unknown image type field in TGA file",
                              __FILE__,__LINE__
                        );
     }



     // Field 4 :  Color Map specification

      _cm_first_entr_ind = fp.lsb_read_U_INT2() ;     // [4.1]
      _cm_length         = fp.lsb_read_U_INT2() ;     // [4.2]
      _cm_entr_sz        = fp.lsb_read_U_INT1() ;     // [4.3]

      if (!_cmap_present)
         ASSERT_TJS_USER
         (
            (! _cm_first_entr_ind) && (! _cm_length) && (! _cm_entr_sz),
            "Incoherence in TGA file (field 4 != 0 without color map)"
         );


    // Field 5 :  Image Specification field


      _x0  = fp.lsb_read_U_INT2() ;
      _y0  = fp.lsb_read_U_INT2() ;
      _szx = fp.lsb_read_U_INT2() ;
      _szy = fp.lsb_read_U_INT2() ;
      _pix_depth = fp.lsb_read_U_INT1() ;
      _nb_byte_pix = nb_bits_to_nb_byte(_pix_depth);

      {
           U_INT1 packed = fp.lsb_read_U_INT1();

           _alpha_channel = sub_bit(packed,0,4);
           _left_to_right  = !(kth_bit(packed,4));
           _top_to_down    = (kth_bit(packed,5)!=0);
      }


   
      _offs_pal = fp.tell();
      _offs_im  = _offs_pal + _cm_length * nb_bits_to_nb_byte(_cm_entr_sz) ;

      if (_nb_byte_pix == 2)
      {
         _r_spec_transf = tga_16_r_spec_transf;
         _w_spec_transf = tga_16_w_spec_transf;
      }
      else if (_nb_byte_pix == 4)
      {
         _r_spec_transf = tga_32_r_spec_transf;
         _w_spec_transf = tga_32_w_spec_transf;
      }
      else
      {
         _r_spec_transf = 0;
         _w_spec_transf = 0;
      }

      fp.close();
      cout << " _id_length : " << _id_length << "\n";
      cout << (_cmap_present ? "color map in file" : "no color map in file") << "\n";

      if (_im_present)
         cout << Name_tga_compr[(INT)_im_compr] << " , " << Name_tga_ty_im[(INT)_im_type] << "\n";
      else
          cout << "no image present";

      cout << "x0,y0 [" << _x0 << "," << _y0
           << "] size [" << _szx << "," << _szy << "]"
           <<  " nb bits / pix " << _pix_depth << "\n";
      cout << " Left to right : " << _left_to_right 
           << " . Top to down : " << _top_to_down << "\n";

}

/********************************************************************************/
/********************************************************************************/
/********************************************************************************/
/********************************************************************************/

     /*------------------------------------------------------------------*/
     /*                                                                  */
     /*           Tga_Tiles                                              */
     /*                                                                  */
     /*------------------------------------------------------------------*/

class TGA_tile : public Tile_F2d
{
    public :
        TGA_tile(Data_TGA_File * dtga,bool t_to_d,bool l_to_r) ;

    private :


        Data_TGA_File * _dtga;
        bool    _t_to_d;
        //bool    _l_to_r;


    // redefine, just to avoid useless call to new line
    void seek_pack_line(Fich_Im2d *,INT,INT,bool) 
    {
    }

    void r_new_line(Fich_Im2d *,INT y)
    {
           if (! _t_to_d)
              y = _dtga->_szy-y-1;
           (SAFE_DYNC(Std_Packed_Flux_Of_Byte *,_pfob))->Aseek(y*_dtga->_szx);
    }

    void w_new_line(Fich_Im2d * f,INT y)
    {
         r_new_line(f,y);
    }


};



TGA_tile::TGA_tile(Data_TGA_File * dtga,bool t_to_d,bool l_to_r) :
   Tile_F2d
   (
      new Std_Packed_Flux_Of_Byte
          (dtga->name(),dtga->_nb_byte_pix,dtga->_offs_im,ELISE_fp::READ)
   ),
   _dtga     (dtga),
   _t_to_d   (t_to_d)//,
   //_l_to_r   (l_to_r)
{
}


class TGA_RLE_Flx_byte : public Packed_Flux_Of_Byte
{
    public :
        TGA_RLE_Flx_byte(Data_TGA_File * dtga); 
        virtual  ~TGA_RLE_Flx_byte() {_fp.close();}

    private :

        virtual tFileOffset tell()
        {
            return _fp.tell();
        }

        tFileOffset             _nb_buffered;
        U_INT1          _buf_rle[3];
        bool            _rle_state;
        ELISE_fp         _fp;


        tFileOffset Read(U_INT1 * cbuf,tFileOffset nb_el);

        bool      compressed() const { return true;}
};

TGA_RLE_Flx_byte::TGA_RLE_Flx_byte(Data_TGA_File * dtga) :
     Packed_Flux_Of_Byte(dtga->_nb_byte_pix),
     _nb_buffered  (0)
{
      _fp.open(dtga->_name,ELISE_fp::READ);
      _fp.seek(dtga->_offs_im,ELISE_fp::sbegin);
}


tFileOffset TGA_RLE_Flx_byte::Read(U_INT1 * cbuf,tFileOffset nb_elo)
{
   int nb_el = nb_elo.CKK_IntBasicLLO();
   INT nb_el_red = -12345; // warn init

    for
    ( 
        int sum_nb_el =0;
        sum_nb_el < nb_el;
        sum_nb_el += nb_el_red
    )
    {
       if (_nb_buffered == 0)
       {
           INT packed = _fp.lsb_read_U_INT1(); 
           _rle_state = (kth_bit(packed,7) != 0);
           _nb_buffered = 1+ sub_bit(packed,0,7);
           if (_rle_state)
              _fp.read(_buf_rle,_sz_el,1); 
       }
       nb_el_red = ElMin(_nb_buffered.CKK_IntBasicLLO(),nb_el-sum_nb_el);
       _nb_buffered -= nb_el_red;

       if (_rle_state)
       {
          for (INT iel =0; iel<nb_el_red ; iel++)
              for (INT ibyt =0 ; ibyt<_sz_el ; ibyt++)
                  *(cbuf++) =   _buf_rle[ibyt];
       }
       else
       {
          _fp.read(cbuf,_sz_el,nb_el_red); 
          cbuf += _sz_el * nb_el_red;
       }
    }
 
    return nb_el;
}


     /*------------------------------------------------------------------*/
     /*                                                                  */
     /*           Tga_Fich_Im2d                                          */
     /*                                                                  */
     /*------------------------------------------------------------------*/

class  Tga_Fich_Im2d : public Std_Bitm_Fich_Im_2d
{
   friend class Tga_Tiles;

   public :
       Tga_Fich_Im2d
       (
             Flux_Pts_Computed * flx,
             Data_TGA_File *
       );
   private :
       Data_TGA_File * _dtga;

      void post_traite(Std_Pack_Of_Pts_Gen * pack)
      {
           if (_dtga->_im_type == Tga_Im::true_col)
              pack->rgb_bgr();
           if (       (_dtga->_im_type == Tga_Im::bw_image)
                        &&   (_dtga->_pix_depth != 8)
              )
           {
               INT * v = ((INT **) pack->adr_coord())[0];
               INT nb = pack->nb();
               INT vmax = (1<<_dtga->_pix_depth) -1;
               for (INT i=0 ; i<nb ; i++)
                   v[i] = (v[i] * 255) / vmax;
           }
      }

       const Pack_Of_Pts * pre_traite
       (
                         const Pack_Of_Pts * values,
                         Pack_Of_Pts *       empty_buf,
                         Pack_Of_Pts *       buf
       );

};

const Pack_Of_Pts * Tga_Fich_Im2d::pre_traite
(
                         const Pack_Of_Pts * values,
                         Pack_Of_Pts *       empty_buf,
                         Pack_Of_Pts *       buf
)
{
   if (_dtga->_im_type == Tga_Im::true_col)
   {
      Std_Pack_Of_Pts<INT> * e = const_cast<Std_Pack_Of_Pts<INT> *>(empty_buf->std_cast((INT *)0));

      e->rgb_bgr(values->std_cast());
      return e;

   }
   if (       (_dtga->_im_type == Tga_Im::bw_image)
              &&   (_dtga->_pix_depth != 8)
      )
   {
      const Std_Pack_Of_Pts<INT> * pv = values->std_cast((INT *)0);
      Std_Pack_Of_Pts<INT> * pb = const_cast<Std_Pack_Of_Pts<INT> *>(buf->std_cast((INT *)0));
      INT nb = pv->nb();
      pb->set_nb(nb);
      INT * v = pv->_pts[0];
      INT * b = pb->_pts[0];
      INT vmax = (1<<_dtga->_pix_depth);
      for (INT i=0 ; i<nb ; i++)
          b[i] = (v[i] * vmax) / 256;
      return buf;
   }

   return values;
}



Tga_Fich_Im2d::Tga_Fich_Im2d
(
        Flux_Pts_Computed * flx,
        Data_TGA_File * dtga
) :
  Std_Bitm_Fich_Im_2d
  (
       flx,
       dtga->sz(),
       dtga->sz(),
       dtga->nb_chanel(),
       dtga->name(),
       alloc_im1d(GenIm::u_int1,flx->sz_buf() *dtga->nb_byte_per_pix()),
       (dtga->_im_compr!=Tga_Im::no_compr),
       dtga->nb_byte_per_pix(),
       dtga->r_sptr(),
       dtga->w_sptr()
   ),
   _dtga (dtga)
{

    if (dtga->_im_compr == Tga_Im::no_compr)
       init_tile
       (
             new TGA_tile (dtga,dtga->top_to_down(),dtga->left_to_right()) ,
             0,
             1,
             true
       );
    else
       init_tile
       (
           new Tile_F2d (new TGA_RLE_Flx_byte(dtga)) ,
           0,
           1,
           true
       );
}




     /*------------------------------------------------------------------*/
     /*                                                                  */
     /*           Tga_Im_Not_Comp                                        */
     /*                                                                  */
     /*------------------------------------------------------------------*/

class Tga_Im_Not_Comp : public Fonc_Num_Not_Comp
{
     public :

         Tga_Im_Not_Comp(Tga_Im,bool,REAL);

     private :

        Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);

        Tga_Im   _tga;
        bool     _with_def_value;
        REAL     _def_value;

        virtual bool  integral_fonc (bool ) const 
        {return true;}

        virtual INT dimf_out() const
        {
                return _tga.dtga()->nb_chanel();
        }
        void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}
};

Fonc_Num_Computed * Tga_Im_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{
    Data_TGA_File * dtga = _tga.dtga();

    const char * ch = dtga->why_cant_use_it();
    ASSERT_TJS_USER((!ch),ch);


    Std_Bitm_Fich_Im_2d * SBFI2d 
                          = new Tga_Fich_Im2d(arg.flux(),dtga);

    return fonc_num_std_f2d
           (       arg,
                   SBFI2d,
                   _with_def_value,
                   _def_value
           );
};


Tga_Im_Not_Comp::Tga_Im_Not_Comp(Tga_Im tga,bool wdfv,REAL def_val)   :
      _tga              (tga    ),
      _with_def_value   (wdfv   ),
      _def_value        (def_val)
{
}

     /*------------------------------------------------------------------*/
     /*                                                                  */
     /*           Tga_Im                                                 */
     /*                                                                  */
     /*------------------------------------------------------------------*/

Tga_Im::Tga_Im(const char * name) :
    PRC0(new Data_TGA_File(name))
{
}


Fonc_Num Tga_Im::in()
{
    return new Tga_Im_Not_Comp(*this,false,0.0);
}

Fonc_Num Tga_Im::in(INT def_val)
{
    return new Tga_Im_Not_Comp(*this,true,def_val);
}


bool                   Tga_Im::im_present() const
{
     return dtga()->_im_present;
}


Tga_Im::type_of_image  Tga_Im::toi()        const
{
     return dtga()->_im_type;
}


/***************************************************************/
/***************************************************************/
/***************************************************************/


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
