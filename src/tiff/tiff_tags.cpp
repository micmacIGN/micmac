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


Pseudo_Tiff_Arg::Pseudo_Tiff_Arg() :
    _bidon (true)
{
}

Pseudo_Tiff_Arg::Pseudo_Tiff_Arg
                 (
                     tFileOffset      offs0,
                     Pt2di            sz,
                     GenIm::type_el   type_im,
                     INT              nb_chan,
                     bool             chunk_conf,
                     Pt2di            sz_tile,
                     bool             clip_tile,
                     bool             create
                 ) :
    _bidon      (false),
    _offs0      (offs0),
    _sz         (sz),
    _type_im    (type_im),
    _nb_chan    (nb_chan),
    _chunk_conf (chunk_conf),
    _sz_tile    (sz_tile),
    _clip_tile  (clip_tile),
    _create     (create)
{
}


INT   Pseudo_Tiff_Arg::byte_sz_tile(Pt2di Ktile)  const
{
    INT nbb = nbb_type_num(_type_im);
    INT padconstr   = (nbb < 8) ? (8/nbb) : 1;

    INT sz_tile_log_x  = ElMin(_sz_tile.x,_sz.x -Ktile.x*_sz_tile.x);
    INT sz_tile_phys_x = _clip_tile ? sz_tile_log_x : _sz_tile.x;
    sz_tile_phys_x *= chan_by_plan();
    sz_tile_phys_x = (sz_tile_phys_x + padconstr-1)/padconstr;
    sz_tile_phys_x *= padconstr;

    INT sz_tile_log_y  = ElMin(_sz_tile.y,_sz.y -Ktile.y*_sz_tile.y);
    INT sz_tile_phys_y = _clip_tile ? sz_tile_log_y : _sz_tile.y;

    return 
               ((sz_tile_phys_x * nbb) / 8)
             * sz_tile_phys_y ;
}

// Compte non tenu du multi-canal en planar-conf
INT   Pseudo_Tiff_Arg::nb_tile_x()  const
{
      return (_sz.x+_sz_tile.x-1)/_sz_tile.x;
}

INT   Pseudo_Tiff_Arg::nb_tile_y()  const
{
      return (_sz.y+_sz_tile.y-1)/_sz_tile.y;
}

INT   Pseudo_Tiff_Arg::nb_tile()  const
{
     return nb_tile_x() * nb_tile_y();
}

INT Pseudo_Tiff_Arg::nb_plan()  const
{
   return _chunk_conf ? 1 : _nb_chan;
}

INT Pseudo_Tiff_Arg::chan_by_plan()  const
{
   return _chunk_conf ? _nb_chan : 1 ;
}

tFileOffset Pseudo_Tiff_Arg::sz_tot() const
{
    tFileOffset res = _offs0;

    INT nb_pl = nb_plan();
    INT nb_tx = nb_tile_x();
    INT nb_ty = nb_tile_y();


    for (INT y = 0; y<nb_ty ; y++)
        for (INT x = 0; x<nb_tx ; x++)
            res += byte_sz_tile(Pt2di(x,y))*nb_pl;

    return res;
}



/***********************************************************/
/***********************************************************/
/***                                                     ***/
/***                                                     ***/
/***                                                     ***/
/***********************************************************/
/***********************************************************/

GenIm::type_el  Tiff_Im::to_Elise_Type_Num(FIELD_TYPE ftype,const char * aName)
{
     switch(ftype)
     {
           case eBYTE  : return GenIm::u_int1;
           case eASCII : return GenIm::u_int1;
           case eSHORT : return GenIm::u_int2;
           case eLONG  : return GenIm::u_int4;

           case eSSHORT   : return GenIm::int2;
           case eSLONG  : return GenIm::int4;


           case e_SLONG8  : return GenIm::int8;
           case e_LONG8   : return GenIm::int8;

           default    :;
     }

     if (aName !=0)
     {
          std::cout << "For Name= " << aName  << " Type " << ftype<< "\n";
     }
     elise_internal_error
     (
           "incoherent call to Tiff_Im::to_Elise_Type_Num",
           __FILE__,
           __LINE__
     );

     return GenIm::no_type;  // N'importe quoi
}

const char * Tiff_Im::name_compr(INT compr)
{

    switch (compr)
    {
          case No_Compr           :    return "Uncompressed";
          case CCITT_G3_1D_Compr  :    return "CCIT-1D";
          case Group_3FAX_Compr   :    return "FAX3";
          case Group_4FAX_Compr   :    return "FAX4";
          case LZW_Compr          :    return "LZW";
          case JPEG_Compr         :    return "JPEG";
          case MPD_T6             :    return "MPD_T6";
          case  PackBits_Compr    :    return  "PackBits";
          case  NoByte_PackBits_Compr    :    return  "NoBytePackBits";
    }
	
    El_Internal.ElAssert
    (
        0,
        EEM0 
           << "Unvalid Tiff Compression Mode in Tiff_Im::name_compr"
    );
    return 0;
}

bool Tiff_Im::mode_compr_bin(INT mode)
{
     return
               (mode == CCITT_G3_1D_Compr)
          ||   (mode == Group_3FAX_Compr)
          ||   (mode == Group_4FAX_Compr)     ;
}

const char * Tiff_Im::name_phot_interp(INT phi)
{

    switch (phi)
    {
          case WhiteIsZero        :    return "White=0";
          case BlackIsZero        :    return "Black=0";
          case RGB                :    return "RBG";
          case RGBPalette         :    return "RGB/LUT";
          case TranspMask         :    return "Tranparency";
          case CMYK               :    return "CMYK";
          case  YCbCr             :    return  "YCbCr";
          case  CIELab            :    return  "CIELab";
          case  PtDeLiaison       :    return  "PointDeLiaisons";
          case  PtDAppuisDense       :    return  "PointsD'AppuisDenses";
    }
    El_Internal.ElAssert
    (
        0,
        EEM0 
           << "Unvalid photogram Mode in Tiff_Im::name_phot_interp"
    );
    return 0;
}

const char * Tiff_Im::name_plan_conf(INT plconf)
{

    switch (plconf)
    {
          case Chunky_conf        :    return "Chunky";
          case Planar_conf        :    return "Planar";
    }
    El_Internal.ElAssert
    (
        0,
        EEM0 
           << "Unvalid Tiff Planar Configuration in Tiff_Im::name_plan_conf"
    );
    return 0;
}

const char * Tiff_Im::name_resol_unit(INT unit)
{

    switch (unit)
    {
          case No_Unit        :    return "Undefined unit";
          case Inch_Unit      :    return "Inch";
          case Cm_Unit        :    return "Centimeters";
    }
    El_Internal.ElAssert
    (
        0,
        EEM0 
           << "Unvalid Tiff Resolution Unit in Tiff_Im::name_resol_unit"
    );
    return 0;
}

const char * Tiff_Im::name_data_format(INT form)
{

    switch (form)
    {
          case Unsigned_int      :    return "U_Int";
          case Signed_int        :    return "S_Int";
          case IEEE_float        :    return "Float";
          case Undef_data        :    return "Undef";
    }
    El_Internal.ElAssert
    (
        0,
        EEM0 
           << "Unvalid Tiff Data Format in Tiff_Im::name_data_format"
    );
    return 0;
}


const char * Tiff_Im::name_predictor(INT pred)
{

    switch (pred)
    {
          case No_Predic      :    return "No Pred";
          case Hor_Diff       :    return "Hor Diff";
    }
    El_Internal.ElAssert
    (
        0,
        EEM0 
           << "Unvalid Tiff Predictor in Tiff_Im::name_predictor"
    );
    return 0;
}


Pt2di  Tiff_Im::std_sz_tile_of_nbb (INT nbb)
{
     if (nbb <= 2)
        return Pt2di(2048,2048);

     if (nbb <= 8)
        return Pt2di(1024,1024);

     if (nbb <= 32)
        return Pt2di(512,512);

     return Pt2di(256,256);
}

INT  Tiff_Im::nb_chan_of_phot_interp(PH_INTER_TYPE phot_interp)
{
     switch(phot_interp)
     {
          case WhiteIsZero :
          case BlackIsZero :
          case RGBPalette :
          case TranspMask :
               return 1;

          case RGB :
          case YCbCr :
          case CIELab :
               return 3;

          case PtDAppuisDense :
          case CMYK :
               return 4;

          case PtDeLiaison :
               return 5;
     }
     return 1;
}

const ElSTDNS string   Tiff_Im::Str_No_Compr("NoCompr");
const ElSTDNS string   Tiff_Im::Str_CCITT_G3_1D_Compr("CCITTG31D");
const ElSTDNS string   Tiff_Im::Str_Group_3FAX_Compr("FAX3");
const ElSTDNS string   Tiff_Im::Str_Group_4FAX_Compr("FAX4");
const ElSTDNS string   Tiff_Im::Str_LZW_Compr("LZW");
const ElSTDNS string   Tiff_Im::Str_JPEG_Compr("JPEG");
const ElSTDNS string   Tiff_Im::Str_MPD_T6("MPTDT6");
const ElSTDNS string   Tiff_Im::Str_PackBits_Compr("PackBits");
const ElSTDNS string   Tiff_Im::Str_NoBytePackBits_Compr("NoBytePackBits");

Tiff_Im::COMPR_TYPE Tiff_Im::mode_compr(const ElSTDNS string & str)
{
	if (str ==Str_No_Compr)
		return  No_Compr;
	if (str ==Str_CCITT_G3_1D_Compr)
		return  CCITT_G3_1D_Compr;
	if (str ==Str_Group_3FAX_Compr)
		return  Group_3FAX_Compr;
	if (str ==Str_Group_4FAX_Compr)
		return  Group_4FAX_Compr;
	if (str ==Str_LZW_Compr)
		return  LZW_Compr;
	if (str ==Str_JPEG_Compr)
		return  JPEG_Compr;
	if (str ==Str_MPD_T6)
		return  MPD_T6;
	if (str ==Str_PackBits_Compr)
		return  PackBits_Compr;
	if (str ==Str_NoBytePackBits_Compr)
		return  NoByte_PackBits_Compr;

	ELISE_ASSERT(false,"Inc in Tiff_Im::mode_compr(const ElSTDNS string & str)");
	return  PackBits_Compr;
}

/***********************************************************/
/***********************************************************/
/***                                                     ***/
/***           TIFF_TAG_VALUE                            ***/
/***                                                     ***/
/***********************************************************/
/***********************************************************/


class TIFF_TAG_VALUE
{
      public :

          TIFF_TAG_VALUE (ELISE_fp,DATA_Tiff_Ifd *);
          static void skeep_value(ELISE_fp,DATA_Tiff_Ifd *);


          tFileOffset  get_offset(ELISE_fp);

          std::string getstring(ELISE_fp);
          _INT8 * get_tabi(ELISE_fp);
          tFileOffset * get_taboffset(ELISE_fp);


          _INT8  iget1v() ; // only for tags with exactly
          REAL  rget1v(); // 1 values of integer of real type

          Tiff_Im::FIELD_TYPE _field_type;

          bool _dereferenced;
          bool _integral;
          _INT8   _ivalues[8];  //
		  enum
		  {
				_NB_MAX_RVALUES = 6
		  };
          REAL8 _rvalues[_NB_MAX_RVALUES];  //
          INT _nb_log;
          INT _nb_phys;  // for tags which are  offset to real data,
                         // _nb_phys = 1   whatever maybe _nb

          _INT8 * read_values(GenIm::type_el,_INT8 *,INT nb,ELISE_fp fp);

};




void TIFF_TAG_VALUE:: skeep_value(ELISE_fp fp,DATA_Tiff_Ifd * aDTI)
{
    fp.seek_cur(aDTI->SzTag()-2);
}

_INT8 TIFF_TAG_VALUE::iget1v()
{
    El_Internal.ElAssert
    (
          _integral && (! _dereferenced) && (_nb_log ==1),
          EEM0 << "incorrect call to TIFF_TAG_VALUE::iget1v"
    );
    return _ivalues[0];
}

REAL TIFF_TAG_VALUE::rget1v()
{
    El_Internal.ElAssert
    (
          (!_integral) && (! _dereferenced) && (_nb_log ==1),
          EEM0 << "incorrect call to TIFF_TAG_VALUE::rget1v"
    );
    return _rvalues[0];
}



_INT8 * TIFF_TAG_VALUE::read_values
      (
          GenIm::type_el type,
          _INT8 * tab,
          INT nb,
          ELISE_fp fp
      )
{
    if (! tab)
       tab = STD_NEW_TAB_USER(nb,_INT8); 
    if (type==GenIm::int8)
    {
         fp.read(tab,sizeof(*tab),nb);
         if (! fp.byte_ordered())
            byte_inv_tab(tab,sizeof(*tab),nb);
    }
    else
    {

       GenIm tamp = alloc_im1d(type,nb);
       DataGenIm * dtamp = tamp.data_im();

       fp.read(dtamp->data_lin_gen(),dtamp->sz_el(),nb);
       if (! fp.byte_ordered())
          byte_inv_tab(dtamp->data_lin_gen(),dtamp->sz_el(),nb);

       dtamp->int8_input_rle(tab,nb,dtamp->data_lin_gen(),0);
     }

    return tab;
}

tFileOffset  TIFF_TAG_VALUE::get_offset(ELISE_fp fp)
{
    if (_dereferenced)
    {
       // return tFileOffset::FromReinterpretInt(_ivalues[0]);
       return tFileOffset(_ivalues[0]);
    }
    else
       return fp.tell()-4;
}


tFileOffset * TIFF_TAG_VALUE::get_taboffset(ELISE_fp fp)
{
   tFileOffset * aRes =  STD_NEW_TAB_USER(_nb_log,tFileOffset);

   _INT8  * aRI = get_tabi(fp);

   for (int aK=0 ; aK<_nb_log ; aK++)
   {
       // aRes[aK] = tFileOffset::FromReinterpretInt(aRI[aK]);
       aRes[aK] = tFileOffset(aRI[aK]);
   }

   return aRes;
}



_INT8 * TIFF_TAG_VALUE::get_tabi(ELISE_fp fp)
{
    El_Internal.ElAssert
    (
          _integral ,
          EEM0 << "incorrect call to TIFF_TAG_VALUE::get_tabi"
    );

    _INT8 * res = STD_NEW_TAB_USER(_nb_log,_INT8);

    if (_dereferenced)
    {
         tFileOffset offs_cur = fp.tell();

         // fp.seek_begin(tFileOffset::FromReinterpretInt(_ivalues[0]));
         fp.seek_begin(tFileOffset(_ivalues[0]));

         TIFF_TAG_VALUE::read_values
         (
              Tiff_Im::to_Elise_Type_Num(_field_type,fp.NameFile().c_str()),
              res,
              _nb_log,
              fp
          );
          fp.seek_begin(offs_cur);

if (0&& MPD_MM())
{
    std::cout << "get_tabi " << offs_cur << " " << res[0] <<  "\n";
    getchar();
}
    }
    else
        convert(res,_ivalues,_nb_log);

    return res;
}

std::string TIFF_TAG_VALUE::getstring(ELISE_fp fp)
{
   _INT8 * aTab = get_tabi(fp);
   std::string aRes;
   for (int aK=0 ; aK<_nb_log ; aK++)
   {
       aRes= aRes + char(aTab[aK]);
   }
   STD_DELETE_TAB_USER(aTab);
   return aRes;
}

TIFF_TAG_VALUE::TIFF_TAG_VALUE(ELISE_fp fp,DATA_Tiff_Ifd * aDTI)
{
     _field_type = (Tiff_Im::FIELD_TYPE)  fp.read_U_INT2();
     // _nb_log     = fp.read_INT4();
     _nb_log     = aDTI->LireNbVal(fp) ; // fp.read_INT4();
     _nb_phys    = _nb_log;

if (0&& MPD_MM())
{
   std::cout << "_field_type_field_type " << _field_type << "\n";
}

     int aLimNbByte = aDTI->MaxNbByteTagValNonDefer();

     if (_field_type != Tiff_Im::eRATIONNAL)
     {
         _integral = true;
         GenIm::type_el  el_ty =  Tiff_Im::to_Elise_Type_Num(_field_type,fp.NameFile().c_str());
         INT nb_byte = nbb_type_num(el_ty)/8;
         INT byte_sz = nb_byte * _nb_log;


         if (byte_sz <= aLimNbByte)
         {
            read_values(el_ty,_ivalues,_nb_log,fp);
            if (byte_sz < aLimNbByte)
               fp.seek_cur(aLimNbByte-byte_sz);
            _dereferenced = false;
         }
         else
         {
             _nb_phys = 1;
             // _ivalues[0] =  fp.read_INT4();
             _ivalues[0] =  aDTI->LireOffset(fp);
             _dereferenced = true;
         }
     }
     else
     {
         _integral = false;
         El_Internal.ElAssert
         (
               _nb_log <= _NB_MAX_RVALUES,
               EEM0 
                   << "Didn't know TIFF rationnal coul have more than "
	           << (INT) _NB_MAX_RVALUES  << " values"
         );
         /// tFileOffset offs_goto = fp.read_FileOffset4();
 
         int byte_sz = _nb_log * 8;
         bool DeRef = (byte_sz >aLimNbByte);
         
         tFileOffset off_cur;
         if (DeRef)
         {
             tFileOffset offs_goto = aDTI->LireOffset(fp);
             off_cur   = fp.tell();
             fp.seek_begin(offs_goto);
         }


         for (INT k =0; k < _nb_log ; k++)
         {
              INT p = fp.read_INT4();
              INT q = fp.read_INT4();
              if (q==0)
                  _rvalues[k] = InfRegex;
              else
                 _rvalues[k] = p / (double) q;
         }
         if (DeRef)
             fp.seek_begin(off_cur);

         _dereferenced = false;  // On a tout lu donc on fait croire que DeRef est false
     }

}

/***********************************************************/
/***********************************************************/
/***                                                     ***/
/***           TAG_TIF                                   ***/
/***                                                     ***/
/***********************************************************/
/***********************************************************/

class TAG_TIF
{
      public :
         friend void  lire_all_tiff_tag(DATA_Tiff_Ifd *,ELISE_fp);
         friend void  lire_all_tiff_tag(DATA_Tiff_Ifd * ,const Pseudo_Tiff_Arg &);
         friend void  write_all_tiff_tag(DATA_Tiff_Ifd *,ELISE_fp);
  virtual ~TAG_TIF() {}

      protected :

          typedef enum
          {
             ImageWidth                = 256,
             ImageLength               = 257,
             BitPerSample              = 258,
             Compression               = 259,
             PhotometricInterpretation = 262,
             FillOrder                 = 266,
             StripOffset               = 273,
             Orientation               = 274,
             SamplesPerPixel           = 277,
             RowPerStrip               = 278,
             StripByteCount            = 279,
             MinSampleValue            = 280,
             MaxSampleValue            = 281,
             XResolution               = 282,
             YResolution               = 283,
             PlanarConfiguration       = 284,
             T6Options                 = 293,
             ResolutionUnit            = 296,
             Predictor                 = 317,
             PaletteColor              = 320,
             TileWidth                 = 322,
             TileLength                = 323,
             TileOffsets               = 324,
             TileByteCounts            = 325,
             SampleFormat              = 339,

	     TileFileWidth             = 40000,
	     TileFileLength            = 40001,
             eTagExifTiff_ShutterSpeed = 37377,
             eTagExifTiff_IsoSpeed     = 34855,
             eTagExifTiff_Date         = 36867,
             eTagExifTiff_Aperture     = 37378,
             eTagExifTiff_FocalLength  = 37386,
             eTagExifTiff_FocalEqui35Length  = 41989,
             eTagExifTiff_Camera       = 50708


          }  TAGS_ID;

          TAG_TIF(TAGS_ID id,bool real_field) : 
               _id(id) ,
               _real_field(real_field) 
          {}

          TAGS_ID id() const {return _id;};

          void read_vmodif(DATA_Tiff_Ifd::vmodif &,TIFF_TAG_VALUE & v,ELISE_fp fp);
          bool  write_vmodif(DATA_Tiff_Ifd *,DATA_Tiff_Ifd::vmodif, ELISE_fp fp,Tiff_Im::FIELD_TYPE);

           void tag_set_tile_offset
                (DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,
                  ELISE_fp fp,const char * mes);
           void tag_set_tile_byte_count
                (DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,
                  ELISE_fp fp,const char * mes);

          void write_tiles_offset(DATA_Tiff_Ifd * Di,ELISE_fp fp);
          void write_tiles_byte_count(DATA_Tiff_Ifd * Di,ELISE_fp fp);

          //  Put values if they fit in 4-byte, else 
          // memo offset of byte 8 of tiff (offset to data)

          void write_string0(DATA_Tiff_Ifd *,ELISE_fp,const std::string &);
          void write_int_0 (DATA_Tiff_Ifd *,ELISE_fp,INT aVal,Tiff_Im::FIELD_TYPE);
          void Offset_write_value_0 (DATA_Tiff_Ifd *,ELISE_fp,const tFileOffset   *,INT nb);

          void write_Rvalue_0 (DATA_Tiff_Ifd *,ELISE_fp,const REAL  *,INT nb,Tiff_Im::FIELD_TYPE);
          void write_Ivalue_0 (DATA_Tiff_Ifd *,ELISE_fp,const _INT8   *,INT nb,Tiff_Im::FIELD_TYPE);

          // void write_tab_int_0 (ELISE_fp,INT* aVal,int,Tiff_Im::FIELD_TYPE);

          virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg & pta
                       ) = 0;

      private :

          virtual bool write_value (DATA_Tiff_Ifd * Di,ELISE_fp fp) = 0;

          void write_header_value
              (DATA_Tiff_Ifd *Di,ELISE_fp,Tiff_Im::FIELD_TYPE,INT nb);

          virtual void tag_use_value
                       (DATA_Tiff_Ifd *,TIFF_TAG_VALUE &,ELISE_fp fp) = 0;
          static void  lire_1_tag(DATA_Tiff_Ifd *,ELISE_fp);


          static TAG_TIF  * tags_from_id(TAGS_ID id);

		  enum 
		  {
			  NB_TAG = 34
		  };
          static TAG_TIF * TAB_TAG_TIFF[NB_TAG];

          // Put physically the values at EOF

          void physical_write_ivals(ELISE_fp,const _INT8 *,INT nb,Tiff_Im::FIELD_TYPE);



          void memo_offset_tag(DATA_Tiff_Ifd * Di,ELISE_fp fp,INT nb,Tiff_Im::FIELD_TYPE);


          void write_value_dereferenced(DATA_Tiff_Ifd *Di,ELISE_fp fp);
          void WriteOneRVal(ELISE_fp fp,const double & aVal);

          //==============================================


          const TAGS_ID       _id;
          const bool          _real_field;


          Tiff_Im::FIELD_TYPE    _type_field;
          tFileOffset            _offset_tag;   
          INT                       _nb;
          const      _INT8 *        _ivals;
          const      REAL  *        _rvals;
          bool                      _used;
          
};



void TAG_TIF::physical_write_ivals
     (   ELISE_fp fp,
         const _INT8 * v,
         INT nb,
         Tiff_Im::FIELD_TYPE type
     )
{
    GenIm::type_el  el_ty =  Tiff_Im::to_Elise_Type_Num(type,fp.NameFile().c_str());
    INT byte_by_el = nbb_type_num(el_ty)/8;

    GenIm tamp = alloc_im1d(el_ty,nb);
    DataGenIm * dtamp = tamp.data_im();

    dtamp->out_rle(dtamp->data_lin_gen(),nb,v,0);

    fp.write(dtamp->data_lin_gen(),byte_by_el,nb);
}

void TAG_TIF::memo_offset_tag(DATA_Tiff_Ifd * Di,ELISE_fp fp,INT nb, Tiff_Im::FIELD_TYPE type)
{
     _type_field = type;
     _nb = nb;
     _offset_tag = fp.tell();
     fp.seek_cur(Di->SzPtr());  
}


void TAG_TIF::write_header_value
     (
        DATA_Tiff_Ifd * Di,
        ELISE_fp fp,
        Tiff_Im::FIELD_TYPE type,
        INT nb
     )
{
     fp.write_U_INT2(_id);
     fp.write_U_INT2(type);
     //  fp.write_INT4(nb);
     Di->WriteNbVal(fp,nb);
}

void TAG_TIF::Offset_write_value_0 (DATA_Tiff_Ifd * Di,ELISE_fp fp,const tFileOffset   * v,INT nb)
{
// std::cout << "SZOF " << sizeof(tFileOffset) << " " << sizeof(int) << "\n";
    ELISE_ASSERT(sizeof(tByte4AbsFileOffset)==sizeof(int),"write_value_0 tFileOffset/int");

    _INT8 * aTabOffsetI = STD_NEW_TAB_USER(nb,_INT8);
    for (int aK=0 ; aK<nb ; aK++)
    {
         aTabOffsetI[aK] = v[aK].BasicLLO();
         //    aTabOffsetI[aK] = v[aK].ToReinterpretInt();
    }
    write_Ivalue_0(Di,fp,aTabOffsetI,nb,Tiff_Im::eLONG);
}

void TAG_TIF::write_Ivalue_0
     (   
         DATA_Tiff_Ifd * Di,
         ELISE_fp fp,
         const _INT8 * v,
         INT nb,
         Tiff_Im::FIELD_TYPE type
     )
{
    write_header_value(Di,fp,type,nb);

    El_Internal.ElAssert
    ( ! _real_field,
      EEM0 << "error in TAG_TIF::write_value_0(..,INT *,..)"
    );
    
    GenIm::type_el  el_ty =  Tiff_Im::to_Elise_Type_Num(type,fp.NameFile().c_str());
    INT nb_byte = (nbb_type_num(el_ty)/8) * nb;

    int aNbByteMax = Di->MaxNbByteTagValNonDefer();
    if (nb_byte <= aNbByteMax) 
    {
        physical_write_ivals(fp,v,nb,type);
        if (nb_byte < aNbByteMax)
           fp.seek_cur(aNbByteMax-nb_byte);
        _offset_tag = tFileOffset::NoOffset;
    }
    else
    {
         _ivals = v;
         memo_offset_tag(Di,fp,nb,type);
    }
}

          // void write_value_0 (ELISE_fp,const _INT8   *,INT nb,Tiff_Im::FIELD_TYPE);
void TAG_TIF::write_int_0 (DATA_Tiff_Ifd  * Di,ELISE_fp aFp,INT aVal,Tiff_Im::FIELD_TYPE aType)
{
    _INT8 aValI8 = aVal;
    write_Ivalue_0(Di,aFp,&aValI8,1,aType);
}

void TAG_TIF::write_string0(DATA_Tiff_Ifd * Di,ELISE_fp fp ,const std::string & aStr)
{
   const char * aC = aStr.c_str();
   int aNb = (int)strlen(aC);

   _INT8 * aTab = STD_NEW_TAB_USER(aNb,_INT8);
   convert(aTab,aC,aNb);
   write_Ivalue_0(Di,fp,aTab,aNb,Tiff_Im::eASCII);

   // STD_DELETE_TAB_USER(aTab);
   
}



void TAG_TIF::write_Rvalue_0
     (   
         DATA_Tiff_Ifd * Di,
         ELISE_fp fp,
         const REAL * v,
         INT nb,
         Tiff_Im::FIELD_TYPE type
     )
{
    write_header_value(Di,fp,type,nb);

    El_Internal.ElAssert
    ( 
          _real_field && (type == Tiff_Im::eRATIONNAL),
          EEM0 << "error in TAG_TIF::write_value_0(..,REAL *,..)"
    );
    
    int aNbByte = nb * 8;
    int aNbByteMax = Di->MaxNbByteTagValNonDefer();

    if (aNbByte <=aNbByteMax)
    {
        for (int aK=0 ; aK<nb ; aK++)
            WriteOneRVal(fp,v[aK]);
        if (aNbByte < aNbByteMax)
           fp.seek_cur(aNbByteMax-aNbByte);
        _offset_tag = tFileOffset::NoOffset;
    }
    else
    {
        _rvals = v; 
        memo_offset_tag(Di,fp,nb,type);
    }
}

void TAG_TIF::WriteOneRVal(ELISE_fp fp,const double & aVal)
{
    INT p,q;
    rationnal_approx(aVal,p,q);
    fp.write_INT4(p);
    fp.write_INT4(q);
}


void TAG_TIF::write_value_dereferenced(DATA_Tiff_Ifd *Di,ELISE_fp fp)
{
    
    if (_offset_tag == tFileOffset::NoOffset) 
      return;

    tFileOffset where = fp.tell();
    fp.seek_begin(_offset_tag);


    Di->WriteOffset(fp,where);
    // fp.write_   FileOffset4(where);
    fp.seek_end(0);


    if (_real_field)
    {
       for (INT i=0; i <_nb; i++)
       {
           WriteOneRVal(fp,_rvals[i]);
       }
    }
    else
    {
         physical_write_ivals(fp,_ivals,_nb,_type_field);
    }
}

        //==============================

void TAG_TIF::write_tiles_offset(DATA_Tiff_Ifd * Di,ELISE_fp fp)
{
     Offset_write_value_0
     (
          Di,
          fp,
          Di->_tiles_offset,
          Di->_nb_tile_tot.BasicLLO()
     );
}

void TAG_TIF::write_tiles_byte_count(DATA_Tiff_Ifd * Di,ELISE_fp fp)
{
     Offset_write_value_0
     (
          Di,
          fp,
          Di->_tiles_byte_count,
          Di->_nb_tile_tot.BasicLLO()
     );
}

void TAG_TIF::tag_set_tile_offset
     (DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp,const char *) 
{
      Di->_offs_toffs = v.get_offset(fp);
      Di->_tiles_offset = v.get_taboffset(fp);
}

void TAG_TIF::tag_set_tile_byte_count
     (DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp,const char * ) 
{
      Di->_offs_bcount = v.get_offset(fp);
      Di->_tiles_byte_count = v.get_taboffset(fp);
}


void TAG_TIF::read_vmodif
     (
           DATA_Tiff_Ifd::vmodif &   vmod,
           TIFF_TAG_VALUE &          val,
           ELISE_fp                  fp
     )
{
     vmod._offs = val.get_offset(fp);
     vmod.init(val.get_tabi(fp),val._nb_log);
}

bool TAG_TIF::write_vmodif
     (
          DATA_Tiff_Ifd * Di,
          DATA_Tiff_Ifd::vmodif     vmod, 
          ELISE_fp                  fp,
          Tiff_Im::FIELD_TYPE       type
     )
{
     if (vmod._vals)
     {
        write_Ivalue_0(Di,fp,vmod._vals,vmod._nb,type);
     }
     return  vmod._vals != 0;
}


TAG_TIF  * TAG_TIF::tags_from_id(TAGS_ID id)
{
    for (INT i =0 ; i<NB_TAG ; i++)
        if (TAB_TAG_TIFF[i]->_id == id)
           return TAB_TAG_TIFF[i];
    return 0;
}

void TAG_TIF::lire_1_tag(DATA_Tiff_Ifd * DTIfd,ELISE_fp fp)
{
     TAGS_ID id_tag  = (TAGS_ID) fp.read_U_INT2();

     TAG_TIF * tag = tags_from_id(id_tag);


     if (tag)
     {
        TIFF_TAG_VALUE value (fp,DTIfd);
        tag->tag_use_value(DTIfd,value,fp);
     }
     else
     {
        TIFF_TAG_VALUE::skeep_value(fp,DTIfd);
     }
if (0&& MPD_MM())
{
     std::cout << "DONNNE lire_1_tag \n";
     getchar();
}

}




/***********************************************************/
/***********************************************************/
/***                                                     ***/
/***           unique interface                          ***/
/***                                                     ***/
/***********************************************************/
/***********************************************************/

void  lire_all_tiff_tag(DATA_Tiff_Ifd * DTIfd,ELISE_fp fp)
{
      INT nb_tag = DTIfd->LireNbTag(fp);


      for (INT i = 0; i < nb_tag ; i++)
      {
           
           TAG_TIF::lire_1_tag(DTIfd,fp);
      }
}

void  lire_all_tiff_tag(DATA_Tiff_Ifd * DTIfd,const Pseudo_Tiff_Arg & pta)
{
      for (int itag=0; itag<TAG_TIF::NB_TAG ; itag++)
          TAG_TIF::TAB_TAG_TIFF[itag]->pseudo_read(DTIfd,pta);
}


void   write_all_tiff_tag(DATA_Tiff_Ifd * DTIfd,ELISE_fp fp)
{
       tFileOffset where = fp.tell();

       // nb tag, will fil once I know how many are really used
       // fp.write_U_INT2(0);  
       DTIfd->WriteNbTag(fp,0); 
       INT nb_tag = 0;

       for (INT i =0; i<TAG_TIF::NB_TAG ; i++)
       {
           TAG_TIF * tag = TAG_TIF::TAB_TAG_TIFF[i];
           tag->_used = tag->write_value(DTIfd,fp);
           if (tag->_used)
              nb_tag++;
       }
       //  fp.write_INT4(0);  
       DTIfd->WriteOffset(fp,tFileOffset(0));  // offset to next ifd


       {
          for (INT i =0; i<TAG_TIF::NB_TAG ; i++)
          {
              TAG_TIF * tag = TAG_TIF::TAB_TAG_TIFF[i];
              if (tag->_used)
              {
                  tag->write_value_dereferenced(DTIfd,fp);
              }
          }
       }

       fp.seek_begin(where);
       // fp.write_U_INT2(nb_tag);
       DTIfd->WriteNbTag(fp,nb_tag);

       fp.seek_end(0);
}

/***********************************************************/
/***********************************************************/
/***                                                     ***/
/***           Ensemble des tags                         ***/
/***                                                     ***/
/***********************************************************/
/***********************************************************/


  //===============================================
  //===============================================
  //===============================================

           /* 256 : TAG_TIF_SZX */

class TAG_TIF_SZX :  public  TAG_TIF
{
      public :
        TAG_TIF_SZX () : TAG_TIF(ImageWidth,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_sz.x = v.iget1v();

        }
        bool write_value (DATA_Tiff_Ifd * Di,ELISE_fp fp)
        {
          write_int_0 (Di,fp,Di->_sz.x,Tiff_Im::eLONG);
          return true;
        }
        static TAG_TIF_SZX The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *Di,
                           const Pseudo_Tiff_Arg & pta
                       )
        {
            Di->_sz.x = pta._sz.x;
        }

  virtual ~TAG_TIF_SZX() {}
};




TAG_TIF_SZX TAG_TIF_SZX::The_only_one;


  //===============================================
  //===============================================
  //===============================================

           /* 257 : TAG_TIF_SZY */

class TAG_TIF_SZY :  public  TAG_TIF
{
      public :
        TAG_TIF_SZY () : TAG_TIF(ImageLength,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_sz.y = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
              write_int_0(Di,fp,Di->_sz.y,Tiff_Im::eLONG);
              return true;
        }
        static TAG_TIF_SZY The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *Di,
                           const Pseudo_Tiff_Arg & pta
                       )
        {
            Di->_sz.y = pta._sz.y;
        }
  virtual ~TAG_TIF_SZY() {}
};

TAG_TIF_SZY TAG_TIF_SZY::The_only_one;


  //===============================================
  //===============================================
  //===============================================

           /* 258 : TAG_TIF_BIT_P_CHAN */

class TAG_TIF_BIT_P_CHAN :  public  TAG_TIF
{
      public :
        TAG_TIF_BIT_P_CHAN () : TAG_TIF(BitPerSample,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
              Di->_bits_p_chanel = v.get_tabi(fp);
        }
        bool write_value (DATA_Tiff_Ifd * Di,ELISE_fp fp)
        {
          write_Ivalue_0(Di,fp,Di->_bits_p_chanel,Di->_nb_chanel, Tiff_Im::eSHORT);
          return true;
        }
        static TAG_TIF_BIT_P_CHAN The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *Di,
                           const Pseudo_Tiff_Arg & pta
                       )
        {
            Di->_bits_p_chanel = STD_NEW_TAB_USER(pta._nb_chan,_INT8);
            for (INT c =0; c<pta._nb_chan; c++)
                Di->_bits_p_chanel[c] = nbb_type_num(pta._type_im);
        }
  virtual ~TAG_TIF_BIT_P_CHAN() {}
};

TAG_TIF_BIT_P_CHAN  TAG_TIF_BIT_P_CHAN::The_only_one;

  //===============================================
  //===============================================
  //===============================================

           /* 259 : TAG_TIF_COMPR */

class TAG_TIF_COMPR :  public  TAG_TIF
{
      public :
        TAG_TIF_COMPR () : TAG_TIF(Compression,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_mode_compr = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          write_int_0(Di,fp,Di->_mode_compr,Tiff_Im::eSHORT);
          return true;
        }
        static TAG_TIF_COMPR The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *Di,
                           const Pseudo_Tiff_Arg &
                       )
        {
            Di->_mode_compr = Tiff_Im::No_Compr;
        }
  virtual ~TAG_TIF_COMPR() {}
};

TAG_TIF_COMPR  TAG_TIF_COMPR::The_only_one;



  //===============================================
  //===============================================
  //===============================================

           /* 262 : TAG_TIF_PHIT_INT */

class TAG_TIF_PHIT_INT :  public  TAG_TIF
{
      public :
        TAG_TIF_PHIT_INT () : TAG_TIF(PhotometricInterpretation,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_phot_interp = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          write_int_0(Di,fp,Di->_phot_interp,Tiff_Im::eSHORT);
          return true;
        }
        static TAG_TIF_PHIT_INT The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *Di,
                           const Pseudo_Tiff_Arg &
                       )
        {
            Di->_phot_interp = Tiff_Im::BlackIsZero;
        }
  virtual ~TAG_TIF_PHIT_INT() {}
};

TAG_TIF_PHIT_INT  TAG_TIF_PHIT_INT::The_only_one;

  //===============================================
  //===============================================
  //===============================================

           /* 266 : TAG_TIF_FILL_ORDER */

class TAG_TIF_FILL_ORDER :  public  TAG_TIF
{
      public :
        TAG_TIF_FILL_ORDER () : TAG_TIF(FillOrder,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_msbit_first = (v.iget1v()==1);
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
	    int msbf = Di->_msbit_first;
            write_int_0(Di,fp,msbf,Tiff_Im::eSHORT);
            return true;
        }
        static TAG_TIF_FILL_ORDER The_only_one;


        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *Di,
                           const Pseudo_Tiff_Arg & pta
                       )
        {
            Di->_msbit_first =   (  nbb_type_num(pta._type_im)>=8)
                               ||  msbf_type_num(pta._type_im);
        }
};

TAG_TIF_FILL_ORDER  TAG_TIF_FILL_ORDER::The_only_one;


  //===============================================
  //===============================================
  //===============================================

           /* 273 : TAG_TIF_STRIP_OFFS */

class TAG_TIF_STRIP_OFFS :  public  TAG_TIF
{
      public :
        TAG_TIF_STRIP_OFFS () : TAG_TIF(StripOffset,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
              tag_set_tile_offset(Di,v,fp,"TAG_TIF_STRIP_OFFS");
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {

          if ((! Di->_tiled) && Di->_tiles_offset)
	  {
		  write_tiles_offset(Di,fp);
		  return true;
	  }
          return  false;
        }
        static TAG_TIF_STRIP_OFFS The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }

};

TAG_TIF_STRIP_OFFS  TAG_TIF_STRIP_OFFS::The_only_one;


  //===============================================
  //===============================================
  //===============================================

           /* 274 : TAG_TIF_STRIP_OFFS */

class TAG_TIF_ORIENTATION :  public  TAG_TIF
{
      public :
        TAG_TIF_ORIENTATION () : TAG_TIF(Orientation,false) {}

        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_orientation = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            write_int_0(Di,fp,Di->_orientation,Tiff_Im::eSHORT);
            return true;
        }

        static TAG_TIF_ORIENTATION The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }

};

TAG_TIF_ORIENTATION  TAG_TIF_ORIENTATION::The_only_one;


  //===============================================
  //===============================================
  //===============================================


           /* 277 :  TAG_TIF_NB_CHAN */

class TAG_TIF_NB_CHAN :  public  TAG_TIF
{
      public :
        TAG_TIF_NB_CHAN () : TAG_TIF(SamplesPerPixel,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_nb_chanel = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          write_int_0(Di,fp,Di->_nb_chanel,Tiff_Im::eSHORT);
          return true;
        }
        static TAG_TIF_NB_CHAN The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg & pta
                       )
        {
           Di->_nb_chanel = pta._nb_chan;
        }
};

TAG_TIF_NB_CHAN  TAG_TIF_NB_CHAN::The_only_one;


  //===============================================
  //===============================================
  //===============================================

           /* 278 :  TAG_TIF_ROW_P_STR */

class TAG_TIF_ROW_P_STR :  public  TAG_TIF
{
      public :
        TAG_TIF_ROW_P_STR () : TAG_TIF(RowPerStrip,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_sz_tile.y = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if (! Di->_tiled)
            {
               write_int_0(Di,fp,Di->_sz_tile.y,Tiff_Im::eLONG);
            }
            return (! Di->_tiled);
        }
        static TAG_TIF_ROW_P_STR The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};

TAG_TIF_ROW_P_STR  TAG_TIF_ROW_P_STR::The_only_one;

  //===============================================
  //===============================================
  //===============================================

           /* 279 : TAG_TIF_STRIP_OFFS */

class TAG_TIF_STRIP_BYTEC :  public  TAG_TIF
{
      public :
        TAG_TIF_STRIP_BYTEC () : TAG_TIF(StripByteCount,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
              tag_set_tile_byte_count(Di,v,fp,"TAG_TIF_STRIP_BYTEC");
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
             if ((! Di->_tiled) && Di->_tiles_byte_count)
	     {
                 write_tiles_byte_count(Di,fp);
                 return true;
	     }
             return  false;
        }
        static TAG_TIF_STRIP_BYTEC The_only_one;


        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * ,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};

TAG_TIF_STRIP_BYTEC  TAG_TIF_STRIP_BYTEC::The_only_one;


  //===============================================
  //===============================================
  //===============================================

             // MaxSampleValue            = 281,

           /* 280 : TAG_TIF_VMIN_SAMP */

class TAG_TIF_VMIN_SAMP :  public  TAG_TIF
{
      public :
        TAG_TIF_VMIN_SAMP () : TAG_TIF(MinSampleValue,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
             read_vmodif(Di->_mins,v,fp);
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
           return write_vmodif(Di,Di->_mins,fp,Tiff_Im::eSHORT);
        }
        static TAG_TIF_VMIN_SAMP The_only_one;


        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * ,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};

TAG_TIF_VMIN_SAMP  TAG_TIF_VMIN_SAMP::The_only_one;




  //===============================================
  //===============================================
  //===============================================

           /* 281 : TAG_TIF_VMAX_SAMP */

class TAG_TIF_VMAX_SAMP :  public  TAG_TIF
{
      public :
        TAG_TIF_VMAX_SAMP () : TAG_TIF(MaxSampleValue,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
             read_vmodif(Di->_maxs,v,fp);
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
           return write_vmodif(Di,Di->_maxs,fp,Tiff_Im::eSHORT);
        }
        static TAG_TIF_VMAX_SAMP The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * ,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};

TAG_TIF_VMAX_SAMP  TAG_TIF_VMAX_SAMP::The_only_one;

  //===============================================
  //===============================================
  //===============================================

           /* 282 : TAG_TIF_XRESOL */

class TAG_TIF_XRESOL :  public  TAG_TIF
{
      public :
        TAG_TIF_XRESOL () : TAG_TIF(XResolution,true) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_resol.x =  v.rget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          write_Rvalue_0(Di,fp,&(Di->_resol.x),1,Tiff_Im::eRATIONNAL);
          return true;
        }
        static TAG_TIF_XRESOL The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg & 
                       )
        {
             Di->_resol.x = 1;
        }
};

TAG_TIF_XRESOL  TAG_TIF_XRESOL::The_only_one;





  //===============================================
  //===============================================
  //===============================================

           /* 283 : TAG_TIF_XRESOL */

class TAG_TIF_YRESOL :  public  TAG_TIF
{
      public :
        TAG_TIF_YRESOL () : TAG_TIF(YResolution,true) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_resol.y =  v.rget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          write_Rvalue_0(Di,fp,&(Di->_resol.y),1,Tiff_Im::eRATIONNAL);
          return true;
        }
        static TAG_TIF_YRESOL The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg & 
                       )
        {
             Di->_resol.y = 1;
        }
};

TAG_TIF_YRESOL  TAG_TIF_YRESOL::The_only_one;


  //===============================================
  //===============================================
  //===============================================


           /* 284 : TAG_TIF_PLAN_CONFIG */

class TAG_TIF_PLAN_CONFIG :  public  TAG_TIF
{
      public :
        TAG_TIF_PLAN_CONFIG () : TAG_TIF(PlanarConfiguration,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_plan_conf =  v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          write_int_0(Di,fp,Di->_plan_conf,Tiff_Im::eSHORT);
          return true;
        }
        static TAG_TIF_PLAN_CONFIG The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg &  pta
                       )
        {
                Di->_plan_conf = pta._chunk_conf      ?
                                 Tiff_Im::Chunky_conf :
                                 Tiff_Im::Planar_conf ;
        }
};

TAG_TIF_PLAN_CONFIG  TAG_TIF_PLAN_CONFIG::The_only_one;





  //===============================================
  //===============================================
  //===============================================

           /* 293 : TAG_T6_OPTIONS */

class TAG_T6_OPTIONS :  public  TAG_TIF
{
      public :
        TAG_T6_OPTIONS () : TAG_TIF(T6Options,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              INT i =  v.iget1v();

              if (kth_bit(i,1))
                 Di->_ccitt_ucomp = true;
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          INT v =0;
          if (Di->_mode_compr == Tiff_Im::Group_4FAX_Compr)
          {
             write_int_0(Di,fp,v,Tiff_Im::eLONG);
          }
          return Di->_mode_compr == Tiff_Im::Group_4FAX_Compr;
        }
        static TAG_T6_OPTIONS The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * ,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};

TAG_T6_OPTIONS  TAG_T6_OPTIONS::The_only_one;






  //===============================================
  //===============================================
  //===============================================

           /* 296 : TAG_TIF_RESOL_UNIT */

class TAG_TIF_RESOL_UNIT :  public  TAG_TIF
{
      public :
        TAG_TIF_RESOL_UNIT () : TAG_TIF(ResolutionUnit,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_res_unit =  v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
             write_int_0(Di,fp,Di->_res_unit,Tiff_Im::eSHORT);
             return true;
        }
        static TAG_TIF_RESOL_UNIT The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * ,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};


TAG_TIF_RESOL_UNIT  TAG_TIF_RESOL_UNIT::The_only_one;



  //===============================================
  //===============================================
  //===============================================
           /* 317 : TAG_TIF_PREDICTOR */

class TAG_TIF_PREDICTOR :  public  TAG_TIF
{
      public :
        TAG_TIF_PREDICTOR () : TAG_TIF(Predictor,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_predict =  v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
// certain lecteur ne comprennent pas ce tag, donc quand valeur par
// defaut, on ne met rien
             if(Di->_predict != Tiff_Im::No_Predic)
             {
                write_int_0(Di,fp,Di->_predict,Tiff_Im::eSHORT);
                return true;
             }
             else
                return false;
        }
        static TAG_TIF_PREDICTOR The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * ,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};


TAG_TIF_PREDICTOR  TAG_TIF_PREDICTOR::The_only_one;



  //===============================================
  //===============================================
  //===============================================

           /* 320 :  TAG_TIF_RGB_PALETTE */

class TAG_TIF_RGB_PALETTE :  public  TAG_TIF
{
      public :
        TAG_TIF_RGB_PALETTE () : TAG_TIF(PaletteColor,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
              _INT8 * pal = v.get_tabi(fp);
              Di->_nb_pal_entry = v._nb_log;
              Di->_palette = STD_NEW_TAB_USER(Di->_nb_pal_entry,_INT8);
              convert(Di->_palette,pal,Di->_nb_pal_entry);
              STD_DELETE_TAB_USER(pal);
        }
        bool write_value (DATA_Tiff_Ifd * DI,ELISE_fp fp)
        {
             if(DI->_palette != 0)
             {
                 write_Ivalue_0(DI,fp,DI->_palette,DI->_nb_pal_entry,Tiff_Im::eSHORT);
             }
             return (DI->_palette != 0);
        }
        static TAG_TIF_RGB_PALETTE The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * ,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};

TAG_TIF_RGB_PALETTE  TAG_TIF_RGB_PALETTE::The_only_one;




  //===============================================
  //===============================================
  //===============================================

           /* 322 :  TAG_TIF_TILE_WIDTH */

class TAG_TIF_TILE_WIDTH :  public  TAG_TIF
{
      public :
        TAG_TIF_TILE_WIDTH () : TAG_TIF(TileWidth,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_sz_tile.x = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          if (Di->_tiled)
          {
             write_int_0(Di,fp,Di->_sz_tile.x,Tiff_Im::eLONG);
          }
          return Di->_tiled;
        }
        static TAG_TIF_TILE_WIDTH The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg & pta
                       )
        {
           Di->_sz_tile.x = pta._sz_tile.x;
        }
};

TAG_TIF_TILE_WIDTH  TAG_TIF_TILE_WIDTH::The_only_one;



  //===============================================
  //===============================================
  //===============================================

           /* 323 :  TAG_TIF_TILE_LENGTH */

class TAG_TIF_TILE_LENGTH :  public  TAG_TIF
{
      public :
        TAG_TIF_TILE_LENGTH () : TAG_TIF(TileLength,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->_sz_tile.y = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          if (Di->_tiled)
          {
             write_int_0(Di,fp,Di->_sz_tile.y,Tiff_Im::eLONG);
          }
          return Di->_tiled;
        }
        static TAG_TIF_TILE_LENGTH The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg & pta
                       )
        {
           Di->_sz_tile.y = pta._sz_tile.y;
        }
};

TAG_TIF_TILE_LENGTH  TAG_TIF_TILE_LENGTH::The_only_one;



  //===============================================
  //===============================================
  //===============================================

           /* 324 : TAG_TIF_STRIP_OFFS */

class TAG_TIF_TILE_OFFS :  public  TAG_TIF
{
      public :
        TAG_TIF_TILE_OFFS () : TAG_TIF(TileOffsets,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {

              tag_set_tile_offset(Di,v,fp,"TAG_TIF_TILE_OFFS");
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          if (Di->_tiled && Di->_tiles_offset)
	  {
              write_tiles_offset(Di,fp);
              return true;
	  }
          return false;
        }
        static TAG_TIF_TILE_OFFS The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg & pta
                       );
};

void   TAG_TIF_TILE_OFFS::pseudo_read(DATA_Tiff_Ifd * Di,const Pseudo_Tiff_Arg & pta)
{
    Di->_clip_last = pta._clip_tile;
    Di->_offs_toffs = -100;
    Di->_offs_bcount = -100;

    INT nb_tile = pta.nb_tile();
    INT nb_plan = pta.nb_plan();

    Di->_tiles_offset =     STD_NEW_TAB_USER(nb_tile*nb_plan,tFileOffset);
    Di->_tiles_byte_count = STD_NEW_TAB_USER(nb_tile*nb_plan,tFileOffset);

    tFileOffset offs = pta._offs0;

    INT k =0;
    INT nb_tx = pta.nb_tile_x();
    INT nb_ty = pta.nb_tile_y();

    for (INT pl = 0 ; pl <nb_plan ; pl++)
    {
        for (INT y = 0; y<nb_ty ; y++)
        {
             for (INT x = 0; x<nb_tx ; x++)
             {
                  Di->_tiles_byte_count[k] = tFileOffset(pta.byte_sz_tile(Pt2di(x,y)));
                  Di->_tiles_offset[k] = tFileOffset(offs);
                  offs += Di->_tiles_byte_count[k];
                  k++;
             }
        }
    }
}

TAG_TIF_TILE_OFFS  TAG_TIF_TILE_OFFS::The_only_one;



  //===============================================
  //===============================================
  //===============================================

           /* 325 : TAG_TIF_TILE_BYTEC */

class TAG_TIF_TILE_BYTEC :  public  TAG_TIF
{
      public :
        TAG_TIF_TILE_BYTEC () : TAG_TIF(TileByteCounts,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
              tag_set_tile_byte_count(Di,v,fp,"TAG_TIF_TILE_BYTEC");

if (0&& MPD_MM())
{
    std::cout << "XXXXX  " <<  Di->_tiles_byte_count[0] << " " <<  Di->_tiles_byte_count[1] << "\n";
    getchar();
}

        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
          if (Di->_tiled && Di->_tiles_byte_count)
	  {
              write_tiles_byte_count(Di,fp);
              return true;
	  }
          return false;
        }
        static TAG_TIF_TILE_BYTEC The_only_one;


        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *,
                           const Pseudo_Tiff_Arg &
                       )
        {
        }
};

TAG_TIF_TILE_BYTEC  TAG_TIF_TILE_BYTEC::The_only_one;



  //===============================================
  //===============================================
  //===============================================

           /* 339 : TAG_TIF_DATA_FORMAT */

class TAG_TIF_DATA_FORMAT :  public  TAG_TIF
{
      public :
        TAG_TIF_DATA_FORMAT () : TAG_TIF(SampleFormat,false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
              Di->_data_format = v.get_tabi(fp);
              for (int aK=0 ; aK<v._nb_log ; aK++)
              {
                   // On supporte un bug GeoView qui genere parfois des Undef_data
                   if (Di->_data_format[aK] ==  Tiff_Im::Undef_data)
                   {
                        if (aK==0)
                           Di->_data_format[aK] = Tiff_Im::Unsigned_int;
                        else
                           Di->_data_format[aK] = Di->_data_format[0];
                   }
              }
              
        }
        bool write_value (DATA_Tiff_Ifd * Di,ELISE_fp fp)
        {
          write_Ivalue_0(Di,fp,Di->_data_format,Di->_nb_chanel, Tiff_Im::eSHORT);
          return true;
        }
        static TAG_TIF_DATA_FORMAT The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * Di,
                           const Pseudo_Tiff_Arg & pta
                       );
};


void   TAG_TIF_DATA_FORMAT::pseudo_read
       (
            DATA_Tiff_Ifd * Di,
            const Pseudo_Tiff_Arg & pta
       )
{
    Tiff_Im::SAMPLE_FORMAT  format;

    if (type_im_integral(pta._type_im))
    {
       format =   signed_type_num(pta._type_im)  ?
                  Tiff_Im::Signed_int            :
                  Tiff_Im::Unsigned_int          ;
    }
    else
       format = Tiff_Im::IEEE_float;

    Di->_data_format = STD_NEW_TAB_USER(pta._nb_chan,_INT8);
    for (INT c=0 ; c<pta._nb_chan ; c++)
        Di->_data_format[c] = format;

}

TAG_TIF_DATA_FORMAT  TAG_TIF_DATA_FORMAT::The_only_one;


  //===============================================
  //===============================================
  //===============================================

           /* 40000 :  cTAG_TIF_TileFileWidth */

class cTAG_TIF_TileFileWidth :  public  TAG_TIF
{
      public :
        cTAG_TIF_TileFileWidth () : TAG_TIF(TileFileWidth, false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->mUseFileTile = true;
              Di->mSzFileTile.x = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if ( Di->mUseFileTile)
            {
               write_int_0(Di,fp,Di->mSzFileTile.x,Tiff_Im::eLONG);
            }
            return (Di->mUseFileTile != 0);
        }
        static cTAG_TIF_TileFileWidth The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }

  virtual ~cTAG_TIF_TileFileWidth() {}
};

cTAG_TIF_TileFileWidth  cTAG_TIF_TileFileWidth::The_only_one;


  //===============================================
  //===============================================
  //===============================================

           /* 40001 :  cTAG_TIF_TileFileLength */

class cTAG_TIF_TileFileLength :  public  TAG_TIF
{
      public :
        cTAG_TIF_TileFileLength () : TAG_TIF(TileFileLength, false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->mUseFileTile = true;
              Di->mSzFileTile.y = v.iget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if ( Di->mUseFileTile)
            {
               write_int_0(Di,fp,Di->mSzFileTile.y,Tiff_Im::eLONG);
            }
            return  (Di->mUseFileTile != 0);
        }
        static cTAG_TIF_TileFileLength The_only_one;

        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd *,
                           const Pseudo_Tiff_Arg & 
                       )
        {
        }
};

cTAG_TIF_TileFileLength  cTAG_TIF_TileFileLength::The_only_one;

  //   

           /* 40001 :  cTAG_TIF_TileFileLength */

/*
             eTagExifTiff_IsoSpeed     = 34855,
             eTagExifTiff_Date Date    = 36867,
             eTagExifTiff_Aperture Ap  = 37378,
             eTagExifTiff_FocalLength FL = 37386,
             eTagExifTiff_Camera       Cam = 50708
*/

/***********************************************************/
/***                                                     ***/
/***                    TAGS EXIF ET TIFS                ***/
/***                                                     ***/
/***********************************************************/

      // =====    cTAG_TIF_ExifTiff_IsoSpeed   =====

class cTAG_TIF_ExifTiff_IsoSpeed :  public  TAG_TIF
{
      public :
        cTAG_TIF_ExifTiff_IsoSpeed () : TAG_TIF(eTagExifTiff_IsoSpeed, true) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->mExifTiff_IsoSpeed = v.rget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if ( Di->mExifTiff_IsoSpeed>=0)
            {
               write_Rvalue_0(Di,fp,&(Di->mExifTiff_IsoSpeed),1,Tiff_Im::eRATIONNAL);
               return  true;
            }
            return  false;
        }
        static cTAG_TIF_ExifTiff_IsoSpeed The_only_one;
        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * aDI,
                           const Pseudo_Tiff_Arg & 
                       )
        {
            aDI->mExifTiff_IsoSpeed  = -1;
        }
};
cTAG_TIF_ExifTiff_IsoSpeed  cTAG_TIF_ExifTiff_IsoSpeed::The_only_one;

      // =====    cTAG_TIF_ExifTiff_Aperture   =====

class cTAG_TIF_ExifTiff_Aperture :  public  TAG_TIF
{
      public :
        cTAG_TIF_ExifTiff_Aperture () : TAG_TIF(eTagExifTiff_Aperture, true) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->mExifTiff_Aperture = v.rget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if ( Di->mExifTiff_Aperture>=0)
            {
               write_Rvalue_0(Di,fp,&(Di->mExifTiff_Aperture),1,Tiff_Im::eRATIONNAL);
               return  true;
            }
            return  false;
        }
        static cTAG_TIF_ExifTiff_Aperture The_only_one;
        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * aDI,
                           const Pseudo_Tiff_Arg & 
                       )
        {
            aDI->mExifTiff_Aperture  = -1;
        }
};
cTAG_TIF_ExifTiff_Aperture  cTAG_TIF_ExifTiff_Aperture::The_only_one;

      // =====    cTAG_TIF_ExifTiff_FocalLength   =====

class cTAG_TIF_ExifTiff_FocalLength :  public  TAG_TIF
{
      public :
        cTAG_TIF_ExifTiff_FocalLength () : TAG_TIF(eTagExifTiff_FocalLength, true) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->mExifTiff_FocalLength = v.rget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if ( Di->mExifTiff_FocalLength>=0)
            {
               write_Rvalue_0(Di,fp,&(Di->mExifTiff_FocalLength),1,Tiff_Im::eRATIONNAL);
               return  true;
            }
            return  false;
        }
        static cTAG_TIF_ExifTiff_FocalLength The_only_one;
        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * aDI,
                           const Pseudo_Tiff_Arg & 
                       )
        {
            aDI->mExifTiff_FocalLength  = -1;
        }
};
cTAG_TIF_ExifTiff_FocalLength  cTAG_TIF_ExifTiff_FocalLength::The_only_one;

    
      // =====    cTAG_TIF_ExifTiff_FocalEqui35Length   =====

class cTAG_TIF_ExifTiff_FocalEqui35Length :  public  TAG_TIF
{
      public :
        cTAG_TIF_ExifTiff_FocalEqui35Length () : TAG_TIF(eTagExifTiff_FocalEqui35Length, true) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->mExifTiff_FocalEqui35Length = v.rget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if ( Di->mExifTiff_FocalEqui35Length>=0)
            {
               write_Rvalue_0(Di,fp,&(Di->mExifTiff_FocalEqui35Length),1,Tiff_Im::eRATIONNAL);
               return  true;
            }
            return  false;
        }
        static cTAG_TIF_ExifTiff_FocalEqui35Length The_only_one;
        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * aDI,
                           const Pseudo_Tiff_Arg & 
                       )
        {
            aDI->mExifTiff_FocalEqui35Length  = -1;
        }
};
cTAG_TIF_ExifTiff_FocalEqui35Length  cTAG_TIF_ExifTiff_FocalEqui35Length::The_only_one;





      // =====    cTAG_TIF_ExifTiff_ShutterSpeed   =====

class cTAG_TIF_ExifTiff_ShutterSpeed :  public  TAG_TIF
{
      public :
        cTAG_TIF_ExifTiff_ShutterSpeed () : TAG_TIF(eTagExifTiff_ShutterSpeed, true) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp) 
        {
              Di->mExifTiff_ShutterSpeed = v.rget1v();
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if ( Di->mExifTiff_ShutterSpeed>=0)
            {
               write_Rvalue_0(Di,fp,&(Di->mExifTiff_ShutterSpeed),1,Tiff_Im::eRATIONNAL);
               return  true;
            }
            return  false;
        }
        static cTAG_TIF_ExifTiff_ShutterSpeed The_only_one;
        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * aDI,
                           const Pseudo_Tiff_Arg & 
                       )
        {
            aDI->mExifTiff_ShutterSpeed  = -1;
        }
};
cTAG_TIF_ExifTiff_ShutterSpeed  cTAG_TIF_ExifTiff_ShutterSpeed::The_only_one;

    
      // =====    cTAG_TIF_ExifTiff_Camera   =====

class cTAG_TIF_ExifTiff_Camera :  public  TAG_TIF
{
      public :
        cTAG_TIF_ExifTiff_Camera () : TAG_TIF(eTagExifTiff_Camera, false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {
              Di->mExifTiff_Camera = v.getstring(fp);
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if ( Di->mExifTiff_Camera != "" )
            {
               write_string0(Di,fp,Di->mExifTiff_Camera);
               return  true;
            }
            return  false;
        }
        static cTAG_TIF_ExifTiff_Camera The_only_one;
        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * aDI,
                           const Pseudo_Tiff_Arg & 
                       )
        {
            aDI->mExifTiff_Camera  = "";
        }
};
cTAG_TIF_ExifTiff_Camera  cTAG_TIF_ExifTiff_Camera::The_only_one;
    

      // =====    cTAG_TIF_ExifTiff_Date   =====

class cTAG_TIF_ExifTiff_Date :  public  TAG_TIF
{
      public :
        cTAG_TIF_ExifTiff_Date () : TAG_TIF(eTagExifTiff_Date, false) {}
         
        void tag_use_value(DATA_Tiff_Ifd * Di,TIFF_TAG_VALUE & v,ELISE_fp fp) 
        {

if (0&& MPD_MM())
{
    std::cout << "Nb=" << v._nb_log  << " Dddatte [" << v.getstring(fp)  << "]\n";
}


              Di->mExifTiff_Date = cElDate::FromString(v.getstring(fp));
        }
        bool write_value (DATA_Tiff_Ifd *Di,ELISE_fp fp)
        {
            if (! Di->mExifTiff_Date.IsNoDate()) 
            {
               write_string0(Di,fp,ToString(Di->mExifTiff_Date));
               return  true;
            }
            return  false;
        }
        static cTAG_TIF_ExifTiff_Date The_only_one;
        virtual void   pseudo_read
                       (
                           DATA_Tiff_Ifd * aDI,
                           const Pseudo_Tiff_Arg & 
                       )
        {
            aDI->mExifTiff_Date  = cElDate::NoDate;
        }
};
cTAG_TIF_ExifTiff_Date  cTAG_TIF_ExifTiff_Date::The_only_one;
    
        


/***********************************************************/
/***********************************************************/
/***                                                     ***/
/***           TABLEAU DE L'ENSEMBLE DES TAGS            ***/
/***                                                     ***/
/***********************************************************/
/***********************************************************/


TAG_TIF * TAG_TIF::TAB_TAG_TIFF[NB_TAG] =
{
          &(TAG_TIF_SZX::The_only_one),              // 256
          &(TAG_TIF_SZY::The_only_one),              // 257
          &(TAG_TIF_BIT_P_CHAN::The_only_one),       // 258
          &(TAG_TIF_COMPR::The_only_one),            // 259
          &(TAG_TIF_PHIT_INT::The_only_one),         // 262
          &(TAG_TIF_FILL_ORDER::The_only_one),       // 266
          &(TAG_TIF_STRIP_OFFS::The_only_one),       // 273
          &(TAG_TIF_ORIENTATION::The_only_one),      // 274
          &(TAG_TIF_NB_CHAN::The_only_one),          // 277
          &(TAG_TIF_ROW_P_STR::The_only_one),        // 278
          &(TAG_TIF_STRIP_BYTEC::The_only_one),      // 279
          &(TAG_TIF_VMIN_SAMP::The_only_one),        // 280
          &(TAG_TIF_VMAX_SAMP::The_only_one),        // 281
          &(TAG_TIF_XRESOL::The_only_one),           // 282
          &(TAG_TIF_YRESOL::The_only_one),           // 283
          &(TAG_TIF_PLAN_CONFIG::The_only_one),      // 284
          &(TAG_T6_OPTIONS::The_only_one),           // 293
          &(TAG_TIF_RESOL_UNIT::The_only_one),       // 296
          &(TAG_TIF_PREDICTOR::The_only_one),        // 317
          &(TAG_TIF_RGB_PALETTE::The_only_one),      // 320
          &(TAG_TIF_TILE_WIDTH::The_only_one),       // 322
          &(TAG_TIF_TILE_LENGTH::The_only_one),      // 323
          &(TAG_TIF_TILE_OFFS::The_only_one),        // 324
          &(TAG_TIF_TILE_BYTEC::The_only_one),       // 325
          &(TAG_TIF_DATA_FORMAT::The_only_one),      // 339
          &(cTAG_TIF_ExifTiff_IsoSpeed::The_only_one),     // 34855,
          &(cTAG_TIF_ExifTiff_Date::The_only_one),         // 36867,,
          &(cTAG_TIF_ExifTiff_ShutterSpeed::The_only_one), // 37377
          &(cTAG_TIF_ExifTiff_Aperture::The_only_one),     // 37378,
          &(cTAG_TIF_ExifTiff_FocalLength::The_only_one),  // 37386,
          &(cTAG_TIF_TileFileWidth::The_only_one),         // 40 000
          &(cTAG_TIF_TileFileLength::The_only_one),        // 40 001
          &(cTAG_TIF_ExifTiff_FocalEqui35Length::The_only_one),  // 41989,
          &(cTAG_TIF_ExifTiff_Camera::The_only_one)         // 50708
};


/*
             eTagExifTiff_IsoSpeed     = 34855,
             eTagExifTiff_Date         = 36867,
             eTagExifTiff_Aperture     = 37378,
             eTagExifTiff_FocalLength  = 37386,
             eTagExifTiff_Camera       = 50708
*/


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
