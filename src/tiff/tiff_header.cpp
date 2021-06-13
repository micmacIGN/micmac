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


int DefValueBigTif = 0;



/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              DATA_tiff_header                                   ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/

bool IsNamePxm(const std::string & post)
{
    return    (post == "PBM") || (post == "PGM") || (post == "PPM")
           || (post == "pbm") || (post == "pgm") || (post == "ppm");
}

bool IsNameSunRaster(const std::string & post)
{
    return    (post == "RS") || (post == "rs");
}

bool IsNameHDR(const std::string & post)
{
   return  (post == "HDR");
}
bool IsNameXML(const std::string & post)
{
   return  (post == "xml");
}

bool IsTiffisablePost(const std::string & post)
{
    return
                 IsNameHDR(post)
              || IsNameXML(post)
              || IsNameSunRaster(post)
              || IsNamePxm(post);
}

bool IsKnownTifPost(const std::string & aPost)
{
    return      (aPost == "tif")
             || (aPost == "tiff")
             || (aPost == "TIF")
             || (aPost == "TIFF")
             || (aPost == "Tiff")
             || (aPost == "Tif");
}


bool IsKnownJPGPost(const std::string & aPost)
{
    return    (aPost == "jpg")
           || (aPost == "JPG")
           || (aPost == "JPEG")
           || (aPost == "jpeg")
           || (aPost == "Jpg")
           || (aPost == "Jpeg")
      ;
}

bool IsPostfixedJPG(const std::string & aName)
{
   return IsPostfixed(aName) && IsKnownJPGPost(StdPostfix(aName));
}


bool IsKnownNotRawPost(const std::string & aPost)
{
    return    (aPost == "png")
           || (aPost == "PNG");

}
bool IsPostfixedNotRaw(const std::string & aName)
{
   return IsPostfixed(aName) && IsKnownNotRawPost(StdPostfix(aName));
}

std::string NameInDicoGeom(const std::string & aDir,const std::string & aName,const std::string & aDef)
{

   std::string aFullRes = aDir+aName;
   if (ELISE_fp::exist_file(aFullRes) )
      return aFullRes;

   aFullRes = MMDir() + "include"+ELISE_CAR_DIR+"XML_User"+ELISE_CAR_DIR+"DicoCalibGeom"+ELISE_CAR_DIR + aName;
   if (ELISE_fp::exist_file(aFullRes) )
      return aFullRes;

   aFullRes = MMDir() + "include"+ELISE_CAR_DIR+"XML_MicMac"+ELISE_CAR_DIR+"DicoCalibGeom"+ELISE_CAR_DIR + aName;
   if (ELISE_fp::exist_file(aFullRes) )
      return aFullRes;


   return aDef;
}

std::string StdFileCalibOfImage
            (
                 const std::string & aFullName,
                 const std::string & aPrefix,
                 const std::string & aDef,
                 cMetaDataPhoto & aMDP
            )
{
   std::string aDir = DirOfFile(aFullName);
   if (aMDP.IsNoMTD())
       aMDP = cMetaDataPhoto::CreateExiv2(aFullName);

// std::cout << "AAA " << aPrefix << "#"<< aPrefix << "#" << aDef << "#" << aMDP.Cam(true) << "\n"; getchar();

   cCameraEntry * aCE = CamOfName(aMDP.Cam(true) );
   // MPD => sinon blocage qd pas de camera ds xif
   // cCameraEntry * aCE = CamOfName(aMDP.Cam() );

   if (aCE==0)
      return aDef;

   std::string aRes =    aPrefix
                     + "-Calib-"
                     + aCE->ShortName() +  "-"
                     + ToString(round_ni(10*aMDP.FocMm())) + ".xml";


    return NameInDicoGeom(aDir,aRes,aDef);
}



std::string StdNameBayerCalib(const std::string & aFullName)
{
   if (IsPostfixedJPG(aFullName) ||  Tiff_Im::IsTiff(aFullName.c_str(),true) || IsPostfixedNotRaw(aFullName))
      return "";

   cMetaDataPhoto aMDP;
   return StdFileCalibOfImage(aFullName,"Bayer","",aMDP);
}

std::string StdNameGeomCalib(const std::string & aFullName)
{
   std::string aPost = "Raw";
   if (IsPostfixedJPG(aFullName))
       aPost = "Jpg";

   if (Tiff_Im::IsTiff(aFullName.c_str(),true))
      aPost = "Tif";

   cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(aFullName);

   Pt2di aSz = aMDP.XifSzIm();
   aPost = aPost+ ToString(aSz.x)+"x"+ ToString(aSz.y);

   return StdFileCalibOfImage(aFullName,"Geom"+aPost,"",aMDP);
}


/*
  Tiff :  5634,3754
  RAW  :  5616,3744

*/



bool Tiff_Im::IsTiff(const char * name,bool AcceptUnPrefixed )
{
    if (IsPostfixed(name))
    {
        std::string aPost = StdPostfix(name);
        return   IsKnownTifPost(aPost);
    }
    if (! AcceptUnPrefixed)  return false;

    if (sizeofile(name) < 4) return false;

    ELISE_fp fp(name,ELISE_fp::READ);
    INT byte_order_flag = fp.lsb_read_U_INT2();
    bool byte_ordered = false;

    switch (byte_order_flag)
    {
         case Tiff_Im::LSBYTE :
              byte_ordered = (! MSBF_PROCESSOR());
         break;

         case Tiff_Im::MSBYTE :
         // case 0x3550 :
              byte_ordered = MSBF_PROCESSOR();
         break;

         default :
              return false;
              fp.close();
         break;
    }
    fp.set_byte_ordered(byte_ordered);
    INT version = fp.read_U_INT2();
    fp.close();
    return  (version == Tiff_Im::THE_STD_VERSION) || (version == Tiff_Im::BIGTIF_VERSION) ;
}



DATA_tiff_header::DATA_tiff_header(const char * name)
{
    _tprov_name   = dup_name_std(name);
    _name         = _tprov_name->coord();

    ELISE_fp fp(_name,ELISE_fp::READ);

    INT byte_order_flag = fp.lsb_read_U_INT2();
    switch (byte_order_flag)
    {
         case Tiff_Im::LSBYTE :
              _byte_ordered = (! MSBF_PROCESSOR());
         break;

         case Tiff_Im::MSBYTE :
         // case 0x3550 :
              _byte_ordered = MSBF_PROCESSOR();
         break;

         default :
              BasicErrorHandler();
              printf("XX=%x\n",byte_order_flag);
              Tjs_El_User.ElAssert
              (  0,
                 EEM0 <<  "Incoherent byte order flag for tiff file \n"
                      << "|    file : "  <<  _name  << "\n"
                      << "|   got following byte_order_flag : "
                      << byte_order_flag

              );
    };

    fp.set_byte_ordered(_byte_ordered);

    mVersion = fp.read_U_INT2();
    mBigTiff = (mVersion == Tiff_Im::BIGTIF_VERSION);


    Tjs_El_User.ElAssert
    (
            (mVersion == Tiff_Im::THE_STD_VERSION)
         || (mVersion == Tiff_Im::BIGTIF_VERSION),
         EEM0 <<  "Incoherent version number for tiff file \n"
              << "|    file : "  <<  _name  << "\n"
              << "|   got following version number : "
              << mVersion
    );

    if (mBigTiff)
    {
       int aK8 = fp.read_U_INT2();
       ELISE_ASSERT(aK8==Tiff_Im::BIGTIF_K8,"Bad majic in BigTiff");
       int aK0 = fp.read_U_INT2();
       ELISE_ASSERT(aK0==Tiff_Im::BIGTIF_K0,"Bad majic in BigTiff");
    }
    InitBigTiff();
    fp.close();

}

void DATA_tiff_header::InitBigTiff()
{
    mOffsetIfd0 = mBigTiff ? Tiff_Im::BIGTIF_OFSS_IFD0 : Tiff_Im::STD_OFSS_IFD0 ;
    mSzTag      = mBigTiff ? Tiff_Im::BIGTIF_SZ_TAG    : Tiff_Im::STD_SZ_TAG    ;
}


    // BIG TIFF HANDLING 

bool   DATA_tiff_header::BigTiff() const
{
   return mBigTiff;
}







DATA_tiff_header::~DATA_tiff_header()
{
     delete _tprov_name;
}

tFileOffset DATA_tiff_header::ReadFileOffset(ELISE_fp & aFp) const
{
   return mBigTiff ? aFp.read_FileOffset8() :  aFp.read_FileOffset4();
}



ELISE_fp DATA_tiff_header::kth_file(INT & nb,bool read)
{

// static int aCpt=0; aCpt++;
// bool Bug = (aCpt==2);
// if (MPD_MM()) std::cout << "AAAAAAAAAAAAA " << Bug << "\n";
    ELISE_fp fp
             (  _name,
                 read   ?   ELISE_fp::READ  :  ELISE_fp::READ_WRITE
             );

    fp.set_byte_ordered(_byte_ordered);

    // fp.seek_begin(Tiff_Im::OFSS_IFD0);
    fp.seek_begin(mOffsetIfd0);

    INT i=0 ;
    tFileOffset offs = 0;

    // for (offs = fp.read_FileOffset4(); offs.BasicLLO() && (i<nb) ; i++)
    for (offs = ReadFileOffset(fp); offs.BasicLLO() && (i<nb) ; i++)
    {
// if (MPD_MM()) std::cout << "CCCCCCC\n";
          fp.seek_begin(offs);
          U_INT8 nb_tag =  mBigTiff ? fp.read_U_INT8()  : fp.read_U_INT2();
          fp.seek_cur(nb_tag*mSzTag);
          offs = ReadFileOffset(fp);
// if (MPD_MM()) std::cout << "DDDDDD\n";
    }
// if (MPD_MM()) std::cout << "HHHHHHHHHh\n";

    if (offs.BasicLLO())
    {
       fp.seek_begin(offs);
    }
    else
    {
       i--;
    }

    nb = i;
    return fp;
}

INT  DATA_tiff_header::nb_im()
{
    INT nb = 1000000000;

    ELISE_fp  fp = kth_file(nb,true);
    fp.close();

    return nb+1;
}


Tiff_Im DATA_tiff_header::kth_im(INT kth)
{
    INT kth_got =  kth;
    ELISE_fp  fp = kth_file(kth_got,true);

    if (kth != kth_got)
    {
       fp.close();
       INT number_im = nb_im();

       Tjs_El_User.ElAssert
       (
            kth == kth_got,
            EEM0 << " Invalid image num required for Tiff file \n"
                 << "|     File = " << _name   << "\n"
                 << "|  num required : " << kth
                 << "   number of image present " << number_im
       );
    }


    Tiff_Im Image(new DATA_Tiff_Ifd(_byte_ordered,mBigTiff,fp,_name,Pseudo_Tiff_Arg()));
    fp.close();

    return Image;
}




/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              Tiff_File                                          ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/

Tiff_File::Tiff_File(const char * name) :
     PRC0 (new DATA_tiff_header(name))
{
}

DATA_tiff_header * Tiff_File::dth()
{
    return SAFE_DYNC(DATA_tiff_header *,_ptr);
}

INT Tiff_File::nb_im()
{
    return dth()->nb_im();
}

Tiff_Im Tiff_File::kth_im(INT kth)
{
     return  dth()->kth_im(kth);
}

/****************************************************************/
/****************************************************************/
/****************************************************************/
/****************************************************************/



/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              DATA_Tiff_Ifd                                      ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/

Fonc_Num DATA_Tiff_Ifd::in()
{
    return Tiff_Im(this).in();
}

Fonc_Num DATA_Tiff_Ifd::in(REAL val)
{
    return Tiff_Im(this).in(val);
}

Output DATA_Tiff_Ifd::out()
{
    return Tiff_Im(this).out();
}




void DATA_Tiff_Ifd::vmodif::flush()
{
     if (_vals)
        STD_DELETE_TAB_USER(_vals);
}

DATA_Tiff_Ifd::vmodif::vmodif() :
    _vals (0),
    _nb (0),
    _offs(0)
{
}

void DATA_Tiff_Ifd::vmodif::init(_INT8 v0,INT nb)
{
    flush();
    _nb = nb;
    _vals =  STD_NEW_TAB_USER(nb,_INT8);
    for (int i=0; i<nb; i++)
        _vals[i] = v0;
}

void DATA_Tiff_Ifd::vmodif::init(_INT8 *v,INT nb)
{
    flush();
    _nb = nb;
    _vals =  v;
}

void DATA_Tiff_Ifd::vmodif::init_if_0(_INT8 v0,INT nb)
{
    if (! _vals)
        init(v0,nb);
}


const tFileOffset Tiff_Im::UN_INIT_TILE(0xFFFFFFFFu);




DATA_Tiff_Ifd::DATA_Tiff_Ifd
(     const char                  * name,
      Pt2di                       sz,
      GenIm::type_el              type,
      Tiff_Im::COMPR_TYPE         compr,
      Tiff_Im::PH_INTER_TYPE      phot_interp,
      Disc_Pal *                  aPal,
      Elise_colour                * tab_c,
      INT                         nb_col,
      L_Arg_Opt_Tiff              l_arg_opt,
      int *                       aIntPtrBigTif
) :
    mExifTiff_Date (cElDate::NoDate)
{
    _byte_ordered = true;
    _clip_last = false;


     // ================================

    if (phot_interp == Tiff_Im::RGBPalette)
    {
       Tjs_El_User.ElAssert
       (
           tab_c!=0,
           EEM0 << "Give a disc pallte to create Indexed Color Tiff File"
       );
    }



    D_Tiff_ifd_Arg_opt  args_opt;
    args_opt.modif(l_arg_opt);

    _sz = sz;
    _nbb_ch0 = nbb_type_num(type);

    // Quand elles sont equivalente, tjs preferer la
    // compression reconnue dans le standard
    if (_nbb_ch0 <= 8 && (compr== Tiff_Im::NoByte_PackBits_Compr))
       compr = Tiff_Im::PackBits_Compr;

    _sz_tile = Tiff_Im::std_sz_tile_of_nbb(_nbb_ch0);


    _nb_chanel =  Tiff_Im::nb_chan_of_phot_interp(phot_interp);
    _bits_p_chanel = STD_NEW_TAB_USER(_nb_chanel,_INT8);
    _data_format = STD_NEW_TAB_USER(_nb_chanel,_INT8);

    _palette = 0;
    _nb_pal_entry  = -1;

    if (phot_interp == Tiff_Im::RGBPalette)
    {
       Tjs_El_User.ElAssert
       (
           nb_col == (1<<_nbb_ch0),
           EEM0 << "nb color of Pal incoherent with type for Tiff File"
       );
       _nb_pal_entry  = 3 * nb_col;

       _palette = STD_NEW_TAB_USER(_nb_pal_entry,_INT8);

       for (int c=0; c<nb_col ; c++)
       {
          _palette[c         ]=(INT)(tab_c[c].r()*Tiff_Im::MAX_COLOR_PAL);
          _palette[c+nb_col  ]=(INT)(tab_c[c].g()*Tiff_Im::MAX_COLOR_PAL);
          _palette[c+2*nb_col]=(INT)(tab_c[c].b()*Tiff_Im::MAX_COLOR_PAL);
       }
       DELETE_VECTOR(tab_c,0);
    }

    INT df0;
    if (type_im_integral(type))
    {
         if (signed_type_num(type))
            df0 = Tiff_Im::Signed_int;
         else
            df0 = Tiff_Im::Unsigned_int;
    }
    else
       df0 = Tiff_Im::IEEE_float;
    double  aNbOctetTot = 0;
    for (INT ch =0; ch < _nb_chanel ; ch++)
    {
         _bits_p_chanel[ch] = _nbb_ch0;
         _data_format[ch] = df0;
         aNbOctetTot +=  _bits_p_chanel[ch];
    }
    aNbOctetTot /= 8;
    double  aSzNCompr = double(_sz.x) * double(_sz.y) * aNbOctetTot;
    double  aMaxSzFile = 4e9;

    {
         Pt2di aSzFT = args_opt.mSzFileTile;
         int aMaxTile = round_ni(sqrt((aMaxSzFile /aNbOctetTot)) *0.9);

         if ((aSzFT.x ==-1) || (aSzFT.x>5e4))
         {
              if (aSzNCompr >aMaxSzFile)
                args_opt.mSzFileTile = Pt2di(aMaxTile,aMaxTile);
         }
         else if (aSzFT.x >0)
         {
              double aSzTile = double(aSzFT.x) * double(aSzFT.y) * aNbOctetTot;
              if (aSzTile > aMaxSzFile)
                args_opt.mSzFileTile = Pt2di(aMaxTile,aMaxTile);
         }
    }


    _mode_compr   = compr;
    // many tiff reader do not handle prediction for nbbits < 8
    _predict      =  ((compr==Tiff_Im::LZW_Compr) && (_nbb_ch0>=8))  ?
                     Tiff_Im::Hor_Diff               :
                     Tiff_Im::No_Predic              ;

    _msbit_first  = (_nbb_ch0>=8) || msbf_type_num(type);
    _plan_conf    = Tiff_Im::Chunky_conf;
    _res_unit     = Tiff_Im::No_Unit;
    _orientation  = 1;
    _resol        = Pt2dr(1,1);
    _phot_interp  = phot_interp;
    _ccitt_ucomp  = false;
    mExifTiff_FocalEqui35Length = -1;
    mExifTiff_FocalLength = -1;
    mExifTiff_ShutterSpeed = -1;
    mExifTiff_Aperture = -1;
    mExifTiff_IsoSpeed = -1;

    if (_mode_compr == Tiff_Im::MPD_T6)
    {
       Tjs_El_User.ElAssert
       (
               (_nb_chanel == 1)
           &&  (_nbb_ch0 <= 8),
           // &&  (_msbit_first),
           EEM0 << "MPD T6 compression requires 1 channel, MSBF and depth <= 8"
       );

       if (_nbb_ch0 == 1)
          _mode_compr =  Tiff_Im::Group_4FAX_Compr;

    }


           // Use  args_opt

    if (args_opt._predictor != -1)
       _predict = args_opt._predictor;


    if (args_opt._sz_tile.x != -1)
       _sz_tile = args_opt._sz_tile;
    else if (args_opt._row_per_strip != -1)
       _sz_tile = Pt2di(_sz.x,args_opt._row_per_strip);
    else if (args_opt._no_strip)
       _sz_tile = _sz;

   _orientation = args_opt._orientation;

   if (args_opt._plan_conf != -1)
       _plan_conf = args_opt._plan_conf;


    if (args_opt._init_min_maxs)
    {
        _mins.init(args_opt._mins,_nb_chanel);
        _maxs.init(args_opt._maxs,_nb_chanel);
    }

    if (args_opt._res_unit != -1)
    {
       _res_unit = args_opt._res_unit;
       _resol    = args_opt._resol;
    }

    if (args_opt.mExifTiff_FocalLength >=0)
      mExifTiff_FocalLength = args_opt.mExifTiff_FocalLength;
    if (args_opt.mExifTiff_FocalEqui35Length >=0)
      mExifTiff_FocalEqui35Length = args_opt.mExifTiff_FocalEqui35Length;
    if (args_opt.mExifTiff_ShutterSpeed >=0)
      mExifTiff_ShutterSpeed = args_opt.mExifTiff_ShutterSpeed;
    if (args_opt.mExifTiff_Aperture >=0)
      mExifTiff_Aperture = args_opt.mExifTiff_Aperture;
    if (args_opt.mExifTiff_IsoSpeed >=0)
      mExifTiff_IsoSpeed = args_opt.mExifTiff_IsoSpeed;
    if (args_opt.mExifTiff_Camera != "")
      mExifTiff_Camera = args_opt.mExifTiff_Camera;
    if (!args_opt.mExifTiff_Date.IsNoDate())
      mExifTiff_Date = args_opt.mExifTiff_Date;

    //==============================================================

    Tjs_El_User.ElAssert
    (
              (_mode_compr == Tiff_Im::No_Compr)
         ||   (_mode_compr == Tiff_Im::PackBits_Compr)
         ||   (_mode_compr == Tiff_Im::NoByte_PackBits_Compr)
         ||   (_mode_compr == Tiff_Im::LZW_Compr)
         ||   (_mode_compr == Tiff_Im::CCITT_G3_1D_Compr)
         ||   (_mode_compr == Tiff_Im::Group_4FAX_Compr)
         ||   (_mode_compr == Tiff_Im::MPD_T6),
         EEM0 << "Cannot create Tiff file with required compression\n"
              << "| file = " << name
              << " compr = " << Tiff_Im::name_compr(_mode_compr) << "\n"
     );
            // END INITIALIZATION

    post_init(name);



    // GESTION du DALLAGE de fichier

    bool CreateSubTile = false;
    mUseFileTile = false;

    if ((args_opt.mSzFileTile.x != -1 )  &&  (_nbb_ch0>=8))
    {
       mSzFileTile =  Pt2di(ElAbs(args_opt.mSzFileTile.x),ElAbs(args_opt.mSzFileTile.y));
       CreateSubTile = args_opt.mSzFileTile.x > 0;

       if (CreateSubTile)
       {
            // Doivent etre pas superieur a taille du fichier
           mSzFileTile.SetInf(_sz);
            // Doivent etre multiple du dallage
           mSzFileTile.x = round_up(mSzFileTile.x,_sz_tile.x);
           mSzFileTile.y = round_up(mSzFileTile.y,_sz_tile.y);
       }


       mUseFileTile =  (mSzFileTile.x<_sz.x) || (mSzFileTile.y<_sz.y) || (! CreateSubTile);

    }

    mBigTiff = false;
    if ((aIntPtrBigTif!=0) && (*aIntPtrBigTif==-1))
    {
    }
    else if ((aIntPtrBigTif!=0) && (*aIntPtrBigTif==1))
    {
      mUseFileTile = 0;
      mBigTiff = true;
    }
    else // (aIntBigTif==0)
    {
       
       // if (MPD_MM()) std::cout << "BIGTIF suspended momentally \n";
      // mUseFileTile = 0;
      // mBigTiff =  aSzNCompr > aMaxSzFile;
    }

    ELISE_fp fp(name,ELISE_fp::WRITE);
    fp.write_U_INT2(MSBF_PROCESSOR() ? Tiff_Im::MSBYTE : Tiff_Im::LSBYTE);
    fp.write_U_INT2(mBigTiff ? Tiff_Im::BIGTIF_VERSION : Tiff_Im::THE_STD_VERSION);

    if (mBigTiff)
    {
       fp.write_U_INT2(Tiff_Im::BIGTIF_K8);
       fp.write_U_INT2(Tiff_Im::BIGTIF_K0);
    }
    // InitBigTiff();


    tFileOffset aOffsData0 = fp.tell() +  SzPtr();

    // GESTION du dallage interne

    if (mUseFileTile)
    {
         if((_mode_compr != Tiff_Im::No_Compr) || (_nbb_ch0<8))
         {
              cout << "FILE = " << name << "\n";
              ELISE_ASSERT
              (
                 false,
             "TileFile incompatible avec les modes compresses\n"
              );
         }
        _tiles_offset = 0;
        _tiles_byte_count = 0;
    ElList<Arg_Tiff> aLNoTF =
                    l_arg_opt
                  + Arg_Tiff(Tiff_Im::AFileTiling(Pt2di(-1,1)))
                          + Arg_Tiff(Tiff_Im::ATiles(_sz_tile));


       if (CreateSubTile)
       {
      for (int aX0 = 0,iX=0; aX0<_sz.x ; aX0+= mSzFileTile.x,iX++)
          {
         for (int aY0 = 0,iY=0; aY0<_sz.y ; aY0+= mSzFileTile.y,iY++)
         {
                std::string aName = NameTileFile(Pt2di(iX,iY));
        Pt2di aSzTF = Inf(mSzFileTile,_sz-Pt2di(aX0,aY0));

        if (aPal)
        {
                   Tiff_Im(aName.c_str(),aSzTF,type,compr,*aPal,aLNoTF);
        }
        else
        {
                   Tiff_Im(aName.c_str(),aSzTF,type,compr,phot_interp,aLNoTF);
        }
         }
          }
        }
        WriteOffset(fp,aOffsData0);
    }
    else
    {
        _tiles_offset = STD_NEW_TAB_USER(_nb_tile_tot.BasicLLO(),tFileOffset);
        _tiles_byte_count = STD_NEW_TAB_USER(_nb_tile_tot.BasicLLO(),tFileOffset);

        if ( _mode_compr == Tiff_Im::No_Compr)
        {
              for(tFileOffset i = 0 ; i <_nb_tile_tot; i++)
              {
                  _tiles_offset[i.BasicLLO()] = tFileOffset(aOffsData0) + i * _byte_sz_tiles;
                  _tiles_byte_count[i.BasicLLO()] =  _byte_sz_tiles;
              }
              WriteOffset(fp,tFileOffset(aOffsData0)+_nb_tile_tot*_byte_sz_tiles);
              //  fp.write_FileOffset4(tFileOffset(8)+_nb_tile_tot*_byte_sz_tiles);


              fp.write_dummy(_nb_tile_tot*_byte_sz_tiles);

        }
        else
        {
              for(tFileOffset i = 0 ; i <_nb_tile_tot; i++)
              {
                  _tiles_offset[i.BasicLLO()] =   Tiff_Im::UN_INIT_TILE;
                  _tiles_byte_count[i.BasicLLO()] =  Tiff_Im::UN_INIT_TILE;
              }
              WriteOffset(fp,aOffsData0);
              // fp.write_INT4(8);
        }
    }
    write_all_tiff_tag(this,fp);
    fp.close();

}

/*
U_INT8 DATA_Tiff_Ifd::LireNbTag(ELISE_fp & aFp) const
{
    return   mBigTiff ? aFp.read_U_INT8()  : aFp.read_U_INT2();
}
*/


    // BIG TIFF HANDLING 
bool   DATA_Tiff_Ifd::BigTiff() const
{
   return mBigTiff;
}
int  DATA_Tiff_Ifd::SzTag() const
{
    // return mBigTif  ? Tiff_Im::BIGTIF_SZ_TAG : Tiff_Im::STD_SZ_TAG;
    return 4 + 2 * MaxNbByteTagValNonDefer();
}
int DATA_Tiff_Ifd::MaxNbByteTagValNonDefer() const
{
    return mBigTiff  ? 8 : 4;
}
int DATA_Tiff_Ifd::SzPtr() const
{
    return mBigTiff  ? 8 : 4;
}

     // Nb Val

U_INT8  DATA_Tiff_Ifd::LireNbVal(ELISE_fp & aFp) const
{
   return mBigTiff  ?  aFp.read_U_INT8()  : aFp.read_U_INT4();
}
void  DATA_Tiff_Ifd::WriteNbVal(ELISE_fp & aFp,U_INT8 aVal)
{
   if (mBigTiff)
      aFp.write_U_INT8(aVal);
   else
      aFp.write_U_INT4(aVal);
}


     // Nb Tag
U_INT8 DATA_Tiff_Ifd::LireNbTag(ELISE_fp & aFp) const
{
    return   mBigTiff ? aFp.read_U_INT8()  : aFp.read_U_INT2();
}
void  DATA_Tiff_Ifd::WriteNbTag(ELISE_fp & aFp,U_INT8 aVal)
{
   if (mBigTiff)
      aFp.write_U_INT8(aVal);
   else
      aFp.write_U_INT2(aVal);
}



U_INT8  DATA_Tiff_Ifd::LireOffset(ELISE_fp & aFp) const
{
   return mBigTiff  ?  aFp.read_U_INT8()  : aFp.read_U_INT4();
}
void  DATA_Tiff_Ifd::WriteOffset(ELISE_fp & aFp,tFileOffset anOff)
{
   if (mBigTiff)
      aFp.write_U_INT8(anOff.BasicLLO());
   else
      aFp.write_U_INT4(anOff.BasicLLO());
}



extern INT aEliseCptFileOpen;



cMetaDataPhoto  DATA_Tiff_Ifd::MDP()
{
    return cMetaDataPhoto
           (
                _name,
                _sz,
                mExifTiff_Camera,
                mExifTiff_Date,
                mExifTiff_FocalLength,
                mExifTiff_FocalEqui35Length,
                mExifTiff_ShutterSpeed,
                mExifTiff_Aperture,
                mExifTiff_IsoSpeed,
                "",
                "",
                "",
                _bits_p_chanel[0]
           );
}


DATA_Tiff_Ifd::DATA_Tiff_Ifd
(
       bool  byte_ordered,
       bool  aBigTiff,
       ELISE_fp fp,
       const char *name,
       const Pseudo_Tiff_Arg & pta
) :
    mExifTiff_Date (cElDate::NoDate)
{

/*
if (MPD_MM() && (!pta._bidon))
{
   std::cout << "OfffsstPTA= " << pta._offs0.CKK_AbsLLO() << " " << int(aBigTiff)<< "\n";
}
else
{
     std::cout << "OfffssBiiiddddon \n";
}
*/

    mUseFileTile = false;
    mSzFileTile = Pt2di(-10000,-10000);
    _byte_ordered = byte_ordered;
    mBigTiff      = aBigTiff;


    if (! pta._bidon)
    {
       if ((pta._create) && (!ELISE_fp::exist_file(name)))
       {
           fp = ELISE_fp(name,ELISE_fp::WRITE);
           fp.write_dummy(pta.sz_tot());
           fp.close();
       }
       fp = ELISE_fp(name,ELISE_fp::READ);
    }

    _clip_last = false;

  //===================================
  //   Give initial values to tags
  //===================================

       // No def values tags

    _sz = Pt2di(-1,-1);
    //  _phot_interp = (Tiff_Im::PH_INTER_TYPE) -1;
    _phot_interp = Tiff_Im::BlackIsZero;  // Be lenient because standard is not well supported
    _tiles_offset = 0;
    _tiles_byte_count = 0;
    _palette = 0;
    _nb_pal_entry  = -1;


       // Tags with def values, but needing results of other tags

    _data_format = (_INT8 *) 0;
    _bits_p_chanel = (_INT8 *) 0;
    _sz_tile = Pt2di(-1,-1);



       // Tags with, therotically no def values, but I am tolerant ....
    _resol     = Pt2dr(1,1);

       // Tags with def values computable now

    _res_unit     = Tiff_Im::Inch_Unit;
    _orientation = 1;
    _nb_chanel = 1;
    _mode_compr = Tiff_Im::No_Compr;
    _predict  = Tiff_Im::No_Predic;
    _msbit_first = true;
    _plan_conf = Tiff_Im::Chunky_conf;
    _ccitt_ucomp  = false;

    mExifTiff_FocalLength = -1;
    mExifTiff_FocalEqui35Length = -1;
    mExifTiff_ShutterSpeed = -1;
    mExifTiff_Aperture = -1;
    mExifTiff_IsoSpeed = -1;
    mExifTiff_Camera = "";

  //===================================
  //   Read tags values from file
  //===================================

   if (pta._bidon)
   {
       lire_all_tiff_tag(this,fp);
   }
   else
   {
       lire_all_tiff_tag(this,pta);
   }

/*
std::cout << "XIFtif:APRES FL " << mExifTiff_FocalLength << "\n";
std::cout << "XIFtif:APRES SS " << mExifTiff_ShutterSpeed << "\n";
std::cout << "XIFtif:APRES AP " << mExifTiff_Aperture << "\n";
std::cout << "XIFtif:APRES IS " << mExifTiff_IsoSpeed << "\n";
std::cout << "XIFtif:APRES Cam " << mExifTiff_Camera << "\n";
std::cout << "XIFtif:APRES Date " << ToString(mExifTiff_Date) << "\n";
*/



  //===================================
  //   Complete def values
  //===================================

    if (! _bits_p_chanel)
    {
         _bits_p_chanel = STD_NEW_TAB_USER(_nb_chanel,_INT8);
         for (INT i =0; i<_nb_chanel; i++)
             _bits_p_chanel[i] = 1;
    }
    if (! _data_format)
    {
        _data_format = STD_NEW_TAB_USER(_nb_chanel,_INT8);
        for (INT ch =0; ch < _nb_chanel ; ch++)
             _data_format[ch] = Tiff_Im::Unsigned_int;
    }



    // also _tiles_byte_count is a required TAGS , it appears to be absent
    // of some files. As I do not need it for now ...


    if (!(  (_sz.x != -1) && (_sz.y != -1)
             && ((_tiles_offset != 0) || mUseFileTile)
             && (_resol.x != -1) && (_resol.y != -1)
             && (_phot_interp != -1)
          )
       )
    {
         std::cout << "Sz " << _sz
                    << " TileOffset " << _tiles_offset
                    << " Resol " <<   _resol
                    << " Ph Interp " << _phot_interp
                    << "\n";
          // Les tags vraiment necessaires
          if (!(  (_sz.x != -1) && (_sz.y != -1)
                   && ((_tiles_offset != 0) || mUseFileTile)
               )
             )
          {
               ELISE_ASSERT(false,"uncomplete TIFF Information File Directory");
          }
    }


/*
    Tjs_El_User.ElAssert
    (
                (_sz.x != -1) && (_sz.y != -1)
             && ((_tiles_offset != 0) || mUseFileTile)
             && (_resol.x != -1) && (_resol.y != -1)
          // && (_tiles_byte_count != 0)
          && (_phot_interp != (Tiff_Im::PH_INTER_TYPE) -1),
          EEM0 << "uncomplete TIFF Information File Directory"
    );
*/


    if (_phot_interp==Tiff_Im::RGBPalette)
    {
        Tjs_El_User.ElAssert
        (
             _palette !=0,
             EEM0 << "Missing palette for Indexed Color Tiff file "
                  << name
        );
        Tjs_El_User.ElAssert
        (
             _nb_pal_entry == 3*(1<<_bits_p_chanel[0]),
             EEM0 << "Bad palette entry number for Tiff file"
                  << name
        );
    }
    // else : accepte useless palette

    post_init(name);


    if (! pta._bidon)
    {
       fp.close();
    }
}

std::string DATA_Tiff_Ifd::NameTileFile(Pt2di aITF)
{
    return    StdPrefix(_name)
        + "_Tile_"
        + ToString(aITF.x)
        + "_"
        + ToString(aITF.y)
        + ".tif";
}

void DATA_Tiff_Ifd::post_init(const char * name)
{

    _tprov_name   = dup_name_std(name);
    _name         = _tprov_name->coord();


     // I refuse to handle variable depth by channels.
    _nbb_ch0 = _bits_p_chanel[0];
    _nbb_tot = _nbb_ch0 * _nb_chanel;

    Tjs_El_User.ElAssert
    (        (_nbb_ch0==1)
         ||  (! Tiff_Im::mode_compr_bin(_mode_compr)),
         EEM0 << "Bits/sample should be 1 for selected compression mode\n"
              << "|    Tiff file = " << _name << "\n"
              << "|    compr = " << Tiff_Im::name_compr(_mode_compr)
              << " ; Bits/sample = " << _nbb_ch0
    );

   if (_sz_tile.y == -1)   // if not stripped nor tiled
       _sz_tile.y = _sz.y;

   if (_sz_tile.x == -1)   // if stripped
       _sz_tile.x = _sz.x;

   _nb_tile.x = (_sz.x+_sz_tile.x-1)/_sz_tile.x;
   _nb_tile.y = (_sz.y+_sz_tile.y-1)/_sz_tile.y;
   _nb_tile_tot = _nb_tile.x *  _nb_tile.y;
   _tiled       = (_sz_tile.x != _sz.x);

    if (_plan_conf == Tiff_Im::Planar_conf)
        _nb_tile_tot *= _nb_chanel;

    INT nb_bits_el_til  =  (_plan_conf == Tiff_Im::Chunky_conf) ?
                           _nbb_tot                                :
                           _nbb_ch0                                ;

    _line_byte_sz_tiles = ((_sz_tile.x * nb_bits_el_til + 7) / 8);
    _byte_sz_tiles = _line_byte_sz_tiles * _sz_tile.y;

    _type_el = type_el();
    _unpacked_type_el = (_nbb_ch0 < 8) ? GenIm::u_int1 : _type_el;
    _padding_constr   = (_nbb_ch0 < 8) ? (8/_nbb_ch0) : 1;

    _sz_byte_pel_unpacked =
               (nbb_type_num(_unpacked_type_el)/8)
             * (_plan_conf == Tiff_Im::Chunky_conf ? _nb_chanel : 1);
    _nb_chan_per_tile =
          (_plan_conf == Tiff_Im::Chunky_conf ? _nb_chanel : 1);

    _line_el_sz_tiles  = _nb_chan_per_tile * _sz_tile.x;
    _padded_line_el_sz_tiles  = (_line_byte_sz_tiles * 8) / _nbb_ch0;


    _maxs.init_if_0((1<<ElMin(16, _nbb_ch0)) -1,_nb_chanel);
    _mins.init_if_0(0,_nb_chanel);

     /*--------------------------------------------*/


    if (Tiff_Im::mode_compr_bin(_mode_compr))
    {
        Tjs_El_User.ElAssert
        (        (_nbb_ch0==1)
             &&  (_nb_chanel == 1),
             EEM0 << "Bits/sample and nb chanel should be 1 \n"
                  << "|    for selected compression mode\n"
                  << "|    Tiff file = " << _name << "\n"
                  << "|    compr = " << Tiff_Im::name_compr(_mode_compr)
                  << " ; Bits/sample = " << _nbb_ch0
                  << " ; nb chanel = " << _nb_chanel
        );
   }

   {
       int   SZ[2],SZ_TILE[2];
       _sz.to_tab(SZ);
       _sz_tile.to_tab(SZ_TILE);

       ElDataGenFileIm::init
       (
           2,
           SZ,
           _nb_chanel,
           signed_type_num(_type_el),
           type_im_integral(_type_el),
           _nbb_ch0,
           SZ_TILE,
           (Tiff_Im::COMPR_TYPE)_mode_compr != Tiff_Im::No_Compr
       );
   }


   mNbTTByTF = Pt2di
           (
                  mSzFileTile.x/_sz_tile.x  ,
                  mSzFileTile.y/_sz_tile.y
           );
   // std::cout << "ghgjYujg " << mSzFileTile << _sz_tile << mNbTTByTF <<  name << "\n";
}

Pt2di DATA_Tiff_Ifd::SzFileTile() const {return mSzFileTile;}
Pt2di DATA_Tiff_Ifd::NbTTByTF() const  {return mNbTTByTF;}

DATA_Tiff_Ifd::~DATA_Tiff_Ifd()
{
     _maxs.flush();
     _mins.flush();
     delete _tprov_name;
     STD_DELETE_TAB_USER(_bits_p_chanel);
     if (_tiles_offset)
        STD_DELETE_TAB_USER(_tiles_offset);
     STD_DELETE_TAB_USER(_data_format);
     if (_tiles_byte_count)
         STD_DELETE_TAB_USER(_tiles_byte_count);
     if (_palette)
         STD_DELETE_TAB_USER(_palette);
}


void DATA_Tiff_Ifd::show()
{
     std::cout  << "MSBF "  <<  MSBF_PROCESSOR() << " Byte Order " << _byte_ordered  << " BigTiff " << mBigTiff << "\n";
     cout << "TIFF FILE : " << _name << "\n";
     cout << _name << " SIZE (" << _sz.x << "," << _sz.y << ")"
          << "; SZ TILES (" << _sz_tile.x << "," << _sz_tile.y << ")"
          << "; NB CHANNEL " << _nb_chanel;

     cout << "; BITS FORMAT/CH [";
     for (INT i =0 ; i<_nb_chanel ; i++)
     {
         if (i) cout <<  ",";
         cout << _bits_p_chanel[i];
         cout <<" "<< Tiff_Im::name_data_format(_data_format[i]);
     }
     cout << "]\n";

     cout << "MIN MAX [" << _mins._vals[0] << "," << _maxs._vals[0] << "]\n";

     cout << "Compression [" << Tiff_Im::name_compr(_mode_compr) <<"]"
          << "; Predictor [" <<Tiff_Im::name_predictor(_predict) << "]"
          << "; Photometric Interp ["
          << Tiff_Im::name_phot_interp(_phot_interp) << "]";
     cout << ";  Bits Order [" << (_msbit_first ? "MSBF" : "LSBF");
     cout << "]\n";

     cout << "Orientation " << _orientation
          << " Resolution : "
          << "x = "   <<  _resol.x
          << ", y = " <<  _resol.y
          << " per " << Tiff_Im::name_resol_unit(_res_unit) << "\n";

     cout << "Plan Conf [" << Tiff_Im::name_plan_conf(_plan_conf) <<"]";
     cout << "]\n";

     if (_tiles_offset)
     {
       cout << "TILES : ";
     {
         for (int  i =0 ; i<ElMin(6,_nb_tile_tot.CKK_IntBasicLLO()) ; i++)
         {
             cout << "(" << _tiles_offset[i].BasicLLO() << "," ;
            if   (_tiles_byte_count)
                  cout << _tiles_byte_count[i].BasicLLO();
            else
                  cout << "?";
            cout     << ")";
        }
     }
       if (_nb_tile_tot.BasicLLO()>5) cout << "...";
       cout << "\n";
     }
     else
         cout << "TileFiles\n";

     std::cout << "FocMm " << mExifTiff_FocalLength << ", FEquiv35 " << mExifTiff_FocalLength << "\n";
}

static char Buf_WECU[200];

const char * DATA_Tiff_Ifd::why_elise_cant_use()
{
      if (
                 (_mode_compr != Tiff_Im::No_Compr)
            &&   (_mode_compr != Tiff_Im::PackBits_Compr)
            &&   (_mode_compr != Tiff_Im::NoByte_PackBits_Compr)
            &&   (_mode_compr != Tiff_Im::LZW_Compr)
            &&   (_mode_compr != Tiff_Im::CCITT_G3_1D_Compr)
            &&   (_mode_compr != Tiff_Im::Group_4FAX_Compr)
            &&   (_mode_compr != Tiff_Im::MPD_T6)
         )
      {
         sprintf
         (
            Buf_WECU,
            "Do not handle mode compression [%s]",
            Tiff_Im::name_compr(_mode_compr)
         );
         return Buf_WECU;
      }

      if (refuse_for_ever())
         return  "Only handle constant dept by channel, and depth in {1,2,4,8}";


      if (_ccitt_ucomp)
         return "Do not handle CCITT uncompressed mode";
      return 0;
}

bool DATA_Tiff_Ifd::refuse_for_ever()
{

      // Refuse to handle variable depth
      for (INT ch =1;  ch<_nb_chanel ; ch++)
          if (_nbb_ch0 != _bits_p_chanel[ch])
             return true;

      // Refuse to handle depth not power of two
      if (! is_pow_of_2(_nbb_ch0))
         return true;

      // Refuse to handle variable data format
      {
         for (INT ch =1;  ch<_nb_chanel ; ch++)
              if (_data_format[0] != _data_format[ch])
                 return true;
      }

      switch(_data_format[0])
      {
           case Tiff_Im::Unsigned_int :
           case Tiff_Im::Undef_data   :
                if (_nbb_ch0 >16)
                   return true;
           break;

           case Tiff_Im::Signed_int :
                if ( (_nbb_ch0<8) || (_nbb_ch0 >32))
                   return true;
           break;


           case Tiff_Im::IEEE_float :
                if ( (_nbb_ch0<32) || (_nbb_ch0 >64))
                   return true;
           break;
      }

       return false;
}


GenIm::type_el  DATA_Tiff_Ifd::type_el()
{

    if (_data_format[0] == Tiff_Im::IEEE_float)
    {
        switch (_nbb_ch0)
        {
              case 32 : return GenIm::real4;
              case 64 : return GenIm::real8;
        }
    }
    else if (_data_format[0] == Tiff_Im::Unsigned_int) 
    {
         switch (_nbb_ch0)
         {
               case 1 : return (    _msbit_first       ?
                                    GenIm::bits1_msbf  :
                                    GenIm::bits1_lsbf
                               );

               case 2 : return (    _msbit_first       ?
                                    GenIm::bits2_msbf  :
                                    GenIm::bits2_lsbf
                               );

               case 4 : return (    _msbit_first       ?
                                    GenIm::bits4_msbf  :
                                    GenIm::bits4_lsbf
                               );

               case 8 : return GenIm::u_int1;
               case 16 : return GenIm::u_int2;
               case 32 : return GenIm::int4;
          }
    }
    else if (_data_format[0] == Tiff_Im::Signed_int)
    {
         switch (_nbb_ch0)
         {
               case 8 : return GenIm::int1;
               case 16 : return GenIm::int2;
               case 32 : return GenIm::int4;
         }
    }

    std::cout << " TIFF DATA FORMAT " << _data_format[0]   << " NbBits " << _nbb_ch0
              << " Float=" << Tiff_Im::IEEE_float 
              << " Unsigned=" <<  Tiff_Im::Unsigned_int 
              << " Signed="<< Tiff_Im::Signed_int << "\n";

    elise_internal_error
    (
        "incoherence in DATA_Tiff_Ifd::type_el()",
        __FILE__,__LINE__
    );
    return GenIm::no_type;
}



INT DATA_Tiff_Ifd::num_tile(INT tx,INT ty,INT kth)
{
    if (_plan_conf == Tiff_Im::Chunky_conf)
       return  ty*_nb_tile.x+tx;
    else
       return  (_nb_tile.y*kth+ty)*_nb_tile.x+tx;
}

tFileOffset DATA_Tiff_Ifd::offset_tile(INT tx,INT ty,INT kth)
{
    return  _tiles_offset[num_tile(tx,ty,kth)];
}

tFileOffset DATA_Tiff_Ifd::byte_count_tile(INT tx,INT ty,INT kth)
{
    return  _tiles_byte_count[num_tile(tx,ty,kth)];
}


void DATA_Tiff_Ifd::set_value_tile
                (
                    ELISE_fp & fp,
                    INT tx,
                    INT ty,
                    INT kth_ch,
                    tFileOffset value,
                    tFileOffset offset_file,
                    tFileOffset * tab_val
                )
{


   tFileOffset nt = num_tile(tx,ty,kth_ch);
   tFileOffset offs_cur = fp.tell();

   fp.seek_begin(offset_file+nt*4);
   tFileOffset where = fp.tell();
   tFileOffset old_value = LireOffset(fp) ;  // fp.read_FileOffset4();

   Tjs_El_User.ElAssert
   (
        old_value == Tiff_Im::UN_INIT_TILE,
        EEM0 <<  "multiple write in same tile for compressed tiff file \n"
             <<  "|  file = " << _name
             <<  "; tile = " << tx << "," << ty << "\n"
   );

   fp.seek_begin(where); // 
   // fp.seek_cur(-4); // P bbbb
   // fp.write_FileOffset4(value);
   WriteOffset(fp,value);
   tab_val[nt.CKK_Byte4AbsLLO()] = value;

   fp.seek_begin(offs_cur);
}

void DATA_Tiff_Ifd::set_offs_tile
                (
                    ELISE_fp  & fp,
                    INT tx,
                    INT ty,
                    INT kth_ch,
                    tFileOffset value
                )
{
       set_value_tile(fp,tx,ty,kth_ch,value,_offs_toffs,_tiles_offset);
}

void DATA_Tiff_Ifd::set_count_tile
                (
                    ELISE_fp &  fp,
                    INT tx,
                    INT ty,
                    INT kth_ch,
                    tFileOffset value
                )
{
       set_value_tile(fp,tx,ty,kth_ch,value,_offs_bcount,_tiles_byte_count);
}


Disc_Pal   DATA_Tiff_Ifd::pal()
{
    Tjs_El_User.ElAssert
    (
             _palette !=0,
             EEM0 << "Request for palette on non Color Indexed Tiff File"
    );


    int nb_col = 1<< _nbb_ch0;
    Elise_colour * tabc = NEW_VECTEUR(0,nb_col,Elise_colour);

    for (int c=0 ; c <nb_col ; c++)
    {
         tabc[c] = Elise_colour::rgb
                   (
                       _palette[c         ]/ (REAL)Tiff_Im::MAX_COLOR_PAL,
                       _palette[c+nb_col  ]/ (REAL)Tiff_Im::MAX_COLOR_PAL,
                       _palette[c+2*nb_col]/ (REAL)Tiff_Im::MAX_COLOR_PAL
                   );
    }

    Disc_Pal p(tabc,nb_col);
    DELETE_VECTOR(tabc,0);
    return p;
}

/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              Tiff_Im                                            ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/

Tiff_Im::Tiff_Im(DATA_Tiff_Ifd * DTIFD) :
      ElGenFileIm(DTIFD)
{
}

Tiff_Im::Tiff_Im(const char * name) :
      ElGenFileIm(0)
{
    Tiff_File TFile (name);
    *this = TFile.kth_im(0);
}


Tiff_Im::Tiff_Im
(     const char                  * name,
      Pt2di                       sz,
      GenIm::type_el              type,
      Tiff_Im::COMPR_TYPE      compr,
      Tiff_Im::PH_INTER_TYPE   Phot_interp,
      L_Arg_Opt_Tiff              l,
      int *                       aPtrBigTif
)     :

      ElGenFileIm ( new DATA_Tiff_Ifd (name,sz,type,compr,Phot_interp,0,0,0,l,aPtrBigTif))
{
    *this = Tiff_Im(name);
}


Tiff_Im::Tiff_Im
(     const char                  * name,
      Pt2di                       sz,
      GenIm::type_el              type,
      Tiff_Im::COMPR_TYPE      compr,
      Disc_Pal                 The_Pal,
      L_Arg_Opt_Tiff              l,
      int *                       aPtrBigTif
)     :
      ElGenFileIm
      (  new DATA_Tiff_Ifd
             (
                 name,
                 sz,
                 type,
                 compr,
                 RGBPalette,
                 &The_Pal,
                 The_Pal.create_tab_c(),
                 The_Pal.nb_col(),
                 l,
                 aPtrBigTif
              )
      )
{
    *this = Tiff_Im(name);
}


Tiff_Im Tiff_Im::CreateIfNeeded
        (
             bool  &                     IsModified,
             const std::string &         aName,
             Pt2di                       aSz,
             GenIm::type_el              aType,
             COMPR_TYPE                  aCompr,
             PH_INTER_TYPE               aPhotInterp,
             L_Arg_Opt_Tiff              aListArgOpt,
             int *                       aPtrBigTif
        )
{
    if (ELISE_fp::exist_file(aName))
    {
       Tiff_Im aFile = Tiff_Im::UnivConvStd(aName);

       if (
             (aSz == aFile.sz())
          && (aType == aFile.type_el())
          && (aCompr ==  aFile.mode_compr())
          && (aPhotInterp == aFile.phot_interp())
      )
       {
           IsModified = false;
           return aFile;
       }

    }

    IsModified = true;
    return Tiff_Im(aName.c_str(),aSz,aType,aCompr,aPhotInterp,aListArgOpt,aPtrBigTif);
}




DATA_Tiff_Ifd * Tiff_Im::dtifd()
{
     return SAFE_DYNC(DATA_Tiff_Ifd *,_ptr);
}

void Tiff_Im::show()
{
    dtifd()->show();
}

const char * Tiff_Im::why_elise_cant_use()
{
      return dtifd()->why_elise_cant_use();
}

bool Tiff_Im::can_elise_use()
{
      return dtifd()->why_elise_cant_use() == 0;
}

void Tiff_Im::verif_usable(bool)
{
     Tjs_El_User.ElAssert
     (
           can_elise_use(),
           EEM0 << "Cannot use : " << dtifd()->_name << "\n"
                << " | : " << why_elise_cant_use()
     );
}



cMetaDataPhoto Tiff_Im::MDP()
{
     return dtifd()->MDP();
}




INT Tiff_Im::nb_chan()
{
     return dtifd()->_nb_chanel;
}

Pt2di Tiff_Im::sz()
{
     return dtifd()->_sz;
}

INT Tiff_Im::bitpp()
{
     return dtifd()->_nbb_ch0;
}

Tiff_Im::COMPR_TYPE Tiff_Im::mode_compr()
{
    return (Tiff_Im::COMPR_TYPE) dtifd()->_mode_compr;
}
Pt2di Tiff_Im::SzFileTile() {return dtifd()->SzFileTile();}
Pt2di Tiff_Im::NbTTByTF()   {return dtifd()->NbTTByTF();}
std::string Tiff_Im::NameTileFile(Pt2di aTile)   {return dtifd()->NameTileFile(aTile);}




Pt2dr Tiff_Im::resol()
{
     return dtifd()->_resol;
}

Tiff_Im::RESOLUTION_UNIT Tiff_Im::resunit()
{
    return (RESOLUTION_UNIT) dtifd()->_res_unit;
}

Tiff_Im::PH_INTER_TYPE Tiff_Im::phot_interp()
{
    return (PH_INTER_TYPE) dtifd()->_phot_interp;
}

Disc_Pal   Tiff_Im::pal()
{
    return dtifd()->pal();
}

GenIm::type_el   Tiff_Im::type_el()
{
    return dtifd()->type_el();
}

tFileOffset  Tiff_Im::offset_tile(INT x,INT y,INT kth_ch)
{

    return dtifd()->offset_tile(x,y,kth_ch);
}

tFileOffset  Tiff_Im::byte_count_tile(INT x,INT y,INT kth_ch)
{
    return dtifd()->byte_count_tile(x,y,kth_ch);
}


Pt2di Tiff_Im::nb_tile()
{
    return  dtifd()->_nb_tile;
}

Pt2di Tiff_Im::sz_tile()
{
    return  dtifd()->_sz_tile;
}

Tiff_Im::PLANAR_CONFIG  Tiff_Im::plan_conf()
{
    return (Tiff_Im::PLANAR_CONFIG) dtifd()->_plan_conf;
}

const char * Tiff_Im::name()
{
    return dtifd()->name();
}

bool  Tiff_Im::byte_ordered()
{
    return dtifd()->_byte_ordered;
}

bool Tiff_Im::BigTif() const
{
   return (const_cast<Tiff_Im*>(this))->dtifd()->BigTiff();
}



const L_Arg_Opt_Tiff Tiff_Im::Empty_ARG;

Elise_Palette  Tiff_Im::std_pal(Video_Win W)
{

    switch(phot_interp())
    {
        case  RGB :
                return W.prgb();

        case  RGBPalette :
                return pal();


        case  WhiteIsZero :
        case  BlackIsZero :
                return W.pgray();

        default :
            Tjs_El_User.ElAssert
            (
                false,
                EEM0 << "Un Handled kind of palette in Tiff_Im::std_pal "
                     << "; file = " << name()
            );

    }
    return W.pgray();
}

Tiff_Im  Tiff_Im::StdConv(const ElSTDNS string & aName)
{
   return BasicConvStd(aName);
}
/*
*/

Tiff_Im  Tiff_Im::UnivConvStd(const ElSTDNS string & aName)
{
   return StdConvGen(aName,-1,true,false);
}



Tiff_Im  Tiff_Im::BasicConvStd(const ElSTDNS string & Name)
{
    if (IsPostfixed(Name))
    {
       ElSTDNS string post = StdPostfix(Name);

       if ((post=="tif") || (post=="tiff") || (post=="TIF") || (post=="TIFF") || (post=="Tif"))
           return Tiff_Im(Name.c_str());

       if ((post=="RS") || (post=="rs") )
           return Elise_Tiled_File_Im_2D::sun_raster(Name.c_str()).to_tiff();

       if (     (post=="pbm") || (post=="PBM")
                ||  (post=="pgm") || (post=="PGM")
                ||  (post=="ppm") || (post=="PPM")
              )
              return Elise_File_Im::pnm(Name.c_str()).to_tiff();

   }

   cSpecifFormatRaw *   aSFR = GetSFRFromString(Name);
   if (aSFR && (! aSFR->BayPat().IsInit()))
   {
        return Elise_Tiled_File_Im_2D::XML(Name).to_tiff();
   }

    if (IsPostfixed(Name) && (StdPostfix(Name)=="HDR"))
    {
        return Elise_Tiled_File_Im_2D::HDR(Name).to_tiff();
    }

   {
      ElSTDNS string Name_Head = Name+ElSTDNS string(".header");
      if (ELISE_fp::exist_file(Name_Head.c_str()))
         return Elise_Tiled_File_Im_2D::Saphir
                (
                   Name.c_str(),
                   Name_Head.c_str()
                ).to_tiff();
   }



 // cout << Name.c_str() << "Is Thom File = " << IsThomFile(Name) << "\n";

   if (IsThomFile(Name))
   {
      ThomParam aTP(Name.c_str());

      Tiff_Im aRes = Elise_Tiled_File_Im_2D::Thom(Name.c_str()).to_tiff();
      return aRes;
   }



    if (IsPostfixed(Name))
    {
       ElSTDNS string post = StdPostfix(Name);
       Tjs_El_User.ElAssert
       (
          false,
          EEM0 << "Tiff_Im::StdConv, do not Handle postfix : ["
             << post.c_str() <<"]" <<"\n"
             <<"(Full Name  = " << Name.c_str() <<")"
       );
    }
    else
    {
       Tjs_El_User.ElAssert
       (
          false,
          EEM0 << "Not Postifxed/Not Spahir "
               <<"(Full Name  = " << Name.c_str() <<")"
       );
    }


    return Tiff_Im(Name.c_str());
}


std::string Tiff_Im::GetNameOfFileExist(const std::string & aName)
{
   std::string aTest = aName+ ".tif";
   if (ELISE_fp::exist_file(aTest))
      return aTest;

   aTest =  aName;
   if (ELISE_fp::exist_file(aTest))
      return aTest;

   aTest =  aName+ ".TIF";
   if (ELISE_fp::exist_file(aTest))
      return aTest;


   aTest =  aName+ ".tiff";
   if (ELISE_fp::exist_file(aTest))
      return aTest;

   aTest =  aName+ ".TIFF";
   if (ELISE_fp::exist_file(aTest))
      return aTest;

   cout << "FOR NAME = [" << aName << "]\n";
   ELISE_ASSERT(false," cannot Get Tiff_Im::GetNameOfFileExist");
   return "TYTUYtyhhhKKll";
}


/***********************************************************************/
/***********************************************************************/
/***                                                                 ***/
/***                                                                 ***/
/***              Elise_Tiled_File_Im_2D                             ***/
/***                                                                 ***/
/***                                                                 ***/
/***********************************************************************/
/***********************************************************************/

Elise_Tiled_File_Im_2D::~Elise_Tiled_File_Im_2D() {}


Elise_Tiled_File_Im_2D::Elise_Tiled_File_Im_2D
(
     const char *     name,
     Pt2di            sz,
     GenIm::type_el   type,
     INT              dim_out,
     Pt2di            sz_tiles,
     bool             clip_last_tile,
     bool             chunk,
     tFileOffset      offset_0      ,
     bool             create       ,
     bool             byte_ordered
) :
     ElGenFileIm
     (
            new DATA_Tiff_Ifd
            (
                 byte_ordered,
                 false,
                 ELISE_fp(),
                 name,
                 Pseudo_Tiff_Arg
                 (
                      offset_0,
                      sz,
                      type,
                      dim_out,
                      chunk,
                      sz_tiles,
                      clip_last_tile,
                      create
                 )
            )
     )
{
    Tjs_El_User.ElAssert
    (
          (nbb_type_num(type)>=8)
       || (dim_out==1),
       EEM0  << "Elise_Tiled_File_Im_2D \n"
             << "Do not handle combinaison : \n"
             << "    Bits Image + Multiple Channel  "
    );
}


DATA_Tiff_Ifd * Elise_Tiled_File_Im_2D::dtifd()
{
     return SAFE_DYNC(DATA_Tiff_Ifd *,_ptr);
}

Fonc_Num Elise_Tiled_File_Im_2D::in()
{
    return Tiff_Im(dtifd()).in();
}

Fonc_Num Elise_Tiled_File_Im_2D::in(REAL def)
{
    return Tiff_Im(dtifd()).in(def);
}

Output Elise_Tiled_File_Im_2D::out()
{
    return Tiff_Im(dtifd()).out();
}

Tiff_Im  Elise_Tiled_File_Im_2D::to_tiff()
{
    return Tiff_Im(dtifd());
}


Im2DGen Tiff_Im::ReadIm()
{
     Im2DGen aRes = D2alloc_im2d(type_el(),sz().x,sz().y);
     ELISE_COPY(aRes.all_pts(),in(),aRes.out());
     return aRes;
}


Tiff_Im Tiff_Im::CreateFromIm(std::vector<Im2DGen> aV,const std::string & aName,L_Arg_Opt_Tiff ArgOp)
{
    Tiff_Im aRes
            (
                 aName.c_str(),
                 aV[0].sz(),
                 aV[0].TypeEl(),
                 No_Compr,
                 (aV.size() == 3) ? RGB : BlackIsZero,
                 ArgOp
            );

    Fonc_Num In = aV[0].in();
    for (INT aK=1 ; aK<INT(aV.size()) ; aK++)
        In = Virgule(In,aV[aK].in());

    ELISE_COPY(aV[0].all_pts(),In,aRes.out());
    return aRes;
}


Tiff_Im Tiff_Im::CreateFromIm(Im2DGen aI,const std::string & aName,L_Arg_Opt_Tiff ArgOp)
{
  std::vector<Im2DGen> aV;
  aV.push_back(aI);
  return CreateFromIm(aV,aName,ArgOp);
}



Tiff_Im Tiff_Im::CreateFromFonc
        (
            const std::string & aName,
            Pt2di aSz,
            Fonc_Num aFonc,
            GenIm::type_el aTEl,
            COMPR_TYPE  aModeCompr
        )
{
    Tiff_Im aRes
            (
                 aName.c_str(),
                 aSz,
         aTEl,
                 aModeCompr,
         (aFonc.dimf_out() == 1) ? BlackIsZero : RGB
        );
    ELISE_COPY(aRes.all_pts(),aFonc,aRes.out());
    return aRes;
}


Tiff_Im Tiff_Im::CreateFromFonc
        (
            const std::string & aName,
            Pt2di aSz,
            Fonc_Num aFonc,
            GenIm::type_el aTEl
        )
{
   return CreateFromFonc(aName,aSz,aFonc,aTEl,No_Compr);
}

Tiff_Im Tiff_Im::Create8BFromFonc
        (
               const std::string & aName,
               Pt2di aSz,
               Fonc_Num aFonc,
               COMPR_TYPE  aModeCompr
        )
{
   return  CreateFromFonc(aName,aSz,aFonc,GenIm::u_int1,aModeCompr);
}

Tiff_Im Tiff_Im::Create8BFromFonc
        (
               const std::string & aName,
               Pt2di aSz,
               Fonc_Num aFonc
        )
{
   return Create8BFromFonc(aName,aSz,aFonc,No_Compr);
}

Tiff_Im Tiff_Im::LZW_Create8BFromFonc
        (
               const std::string & aName,
               Pt2di aSz,
               Fonc_Num aFonc
        )
{
   return Create8BFromFonc(aName,aSz,aFonc,LZW_Compr);
}




Elise_Palette StdPalOfFile(const std::string & aName,Video_Win aW)
{
   // Tiff_Im  aFile = Tiff_Im::BasicConvStd(aName);
   Tiff_Im  aFile = Tiff_Im::StdConvGen(aName,-1,false);
   return aFile.std_pal(aW);
}


std::vector<Im2DGen *>  Tiff_Im::VecOfIm(Pt2di aSz)
{
   std::vector<Im2DGen *> aRes;
   int aNbC = nb_chan();
   for (int aK=0 ; aK<aNbC ; aK++)
   {
       aRes.push_back(Ptr_D2alloc_im2d(type_el(),aSz.x,aSz.y));
   }
   return aRes;
}


std::vector<Im2D_REAL4>  Tiff_Im::VecOfImFloat(Pt2di aSz)
{
   std::vector<Im2D_REAL4> aRes;
   int aNbC = nb_chan();
   for (int aK=0 ; aK<aNbC ; aK++)
   {
       aRes.push_back(Im2D_REAL4(aSz.x,aSz.y));
   }
   return aRes;
}







std::vector<Im2DGen *> Tiff_Im::ReadVecOfIm()
{
    std::vector<Im2DGen *> aRes = VecOfIm(this->sz());

    ELISE_COPY(this->all_pts(),this->in(),StdOut(aRes));

    return aRes;
}



Output   StdOut(const std::vector<Im2DGen *> & aV)
{
  Output aRes = aV[0]->out();
  for (int aKC=1 ; aKC<int(aV.size()) ; aKC++)
     aRes =  Virgule(aRes, aV[aKC]->out());
  return aRes;
}

Fonc_Num StdInput(const std::vector<Im2DGen *> & aV)
{
  Fonc_Num aRes = aV[0]->in();
  for (int aKC=1 ; aKC<int(aV.size()) ; aKC++)
     aRes =  Virgule(aRes, aV[aKC]->in());
  return aRes;
}


L_Arg_Opt_Tiff  ArgOpTiffMDP(const cMetaDataPhoto & aMDP,bool SVP)
{
   if (aMDP.IsNoMTD())
   {
      return Tiff_Im::Empty_ARG;
   }

   return     Tiff_Im::Empty_ARG
           +  Arg_Tiff(Tiff_Im::AExifTiff_FocalLength(aMDP.FocMm(SVP)))
           +  Arg_Tiff(Tiff_Im::AExifTiff_FocalEqui35Length(aMDP.Foc35(SVP)))
           +  Arg_Tiff(Tiff_Im::AExifTiff_Aperture(aMDP.Diaph(true)))
           +  Arg_Tiff(Tiff_Im::AExifTiff_IsoSpeed(aMDP.IsoSpeed(true)))
           +  Arg_Tiff(Tiff_Im::AExifTiff_ShutterSpeed(aMDP.ExpTime(true)))
           +  Arg_Tiff(Tiff_Im::AExifTiff_Camera(aMDP.Cam(SVP)))
           +  Arg_Tiff(Tiff_Im::AExifTiff_Date(cElDate(aMDP.Date(true))));

}

L_Arg_Opt_Tiff  ArgOpTiffMDP(const std::string & aNF)
{
   return ArgOpTiffMDP(Tiff_Im(aNF.c_str()).MDP(),true);
}


Tiff_Im MMIcone(const std::string & aName)
{
// std::cout << "IIIICOne" << MMDir() << "\n";
    return Tiff_Im::BasicConvStd(MMDir()+"data"+ELISE_CAR_DIR+aName +".tif");

}

Fonc_Num Tiff_Im::in_bool()
{
    return in_bool(in());
}

Fonc_Num Tiff_Im::in_bool_proj()
{
    return in_bool(in_proj());
}


Fonc_Num Tiff_Im::in_bool(Fonc_Num aFonc)
{
       int aV0 = VCentrale_type_num(type_el());

       switch (phot_interp())
       {
            case  Tiff_Im::BlackIsZero :
                  return aFonc >=  aV0;
            break;

            case  Tiff_Im::WhiteIsZero :
                  return  aFonc < aV0;
            break;

            case  Tiff_Im::RGB :
            {
                return aFonc.v0() >= aV0;
            }
            break;


            default :
               ELISE_ASSERT
               (
                    false,
                    "Photo-interpretation type non connu pour un masque"
               );
            break;

       }

       return 0;
}



Tiff_Im  Tiff_Im::StdConvGen(const ElSTDNS string & Name,int aNbChan,bool Bits16,bool ExigNoCompr)
{
   return Tiff_Im::BasicConvStd(NameFileStd(Name,aNbChan,Bits16,ExigNoCompr));
}
/*
*/


/*
*/

static std::string NameAdapt(const std::string  &aFullName, const std::string & aPost,bool toMkDir)
{
   static std::string aDirAdd = std::string("Tmp-MM-Dir")+ELISE_CAR_DIR;

   std::string aDir,aNameOri;
   SplitDirAndFile(aDir,aNameOri,aFullName);
   if (toMkDir)
      ELISE_fp::MkDir(aDir+aDirAdd);

   if ( isUsingSeparateDirectories() ){
      aDir = MMTemporaryDirectory();
      aDirAdd = "";
   }

   return aDir+aDirAdd+aNameOri + aPost + ".tif";
}

extern void  OkStat(const std::string & aFile,struct stat & status);

void TestDate(const std::string & aF1,const std::string & aF2)
{
        struct stat status1;   OkStat(aF1,status1);
        struct stat status2;  OkStat(aF2,status2);
#if (ELISE_unix)
        std::cout << aF1 << " " << status1.st_mtim.tv_sec << " " << status1.st_mtim.tv_nsec << "\n";
        std::cout << aF2 << " " << status2.st_mtim.tv_sec << " " << status2.st_mtim.tv_nsec << "\n";
#endif
}

bool   Tiff_Im::IsNameInternalTile(const std::string & aNameTiled,cInterfChantierNameManipulateur * anICNM)
{
    std::string aNameNoTiled = anICNM->Assoc1To1("NKS-Assoc-Tile2File",aNameTiled,true);

    return aNameNoTiled != "NONE";
}

Tiff_Im  Tiff_Im::Dupl(const std::string& aName)
{
   return Tiff_Im
          (
              aName.c_str(),
              sz(),
              type_el(),
              mode_compr(),
              phot_interp()
          );
}

Tiff_Im StdTiffFromName(const std::string & aFullNameOri)
{
   if (Tiff_Im::IsTiff(aFullNameOri.c_str(),true))
      return Tiff_Im(aFullNameOri.c_str());

   if (IsPostfixed(aFullNameOri))
   {
       ElSTDNS string post = StdPostfix(aFullNameOri);
       if (IsNamePxm(post))
       {
           return Elise_File_Im::pnm(aFullNameOri.c_str()).to_tiff();
       }
       if (IsNameSunRaster(post))
       {
           return Elise_Tiled_File_Im_2D::sun_raster(aFullNameOri.c_str()).to_tiff();
       }
       if (IsNameHDR(post))
       {
           return Elise_Tiled_File_Im_2D::HDR(aFullNameOri.c_str()).to_tiff();
       }
       if (IsNameXML(post))
       {
           return Elise_Tiled_File_Im_2D::XML(aFullNameOri.c_str()).to_tiff();
       }
 //  static Elise_Tiled_File_Im_2D HDR(const std::string & aNameHdr);
          //  static Elise_Tiled_File_Im_2D XML(const std::string & aNameHdr);

   }


   cInterfChantierNameManipulateur::Glob();
   cSpecifFormatRaw *   aSFR = GetSFRFromString(aFullNameOri);
   bool aRawTiff = aSFR && (!aSFR->BayPat().IsInit());

   if (aRawTiff) 
      return Tiff_Im (Elise_Tiled_File_Im_2D::XML(aFullNameOri).to_tiff() );

   

/*
   Elise_File_Im 
   std::cout << "Warrnnn  probable impossicle forcing to tiff file; in " << __FILE__ << " at " << __LINE__ << "\n";
   std::cout << " For File " <<  aFullNameOri << "\n";
*/

   return Tiff_Im(aFullNameOri.c_str());
}



std::string NameFileStd
            (
                const std::string & aFullNameOri,
                int aNbChanSpec,
                bool RequireBits16,
                bool ExigNoCompr,
                bool Create,
                bool ExigB8
            )
{
   cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::Glob();
   cSpecifFormatRaw *   aSFR = GetSFRFromString(aFullNameOri);
   bool aRawTiff = aSFR && (!aSFR->BayPat().IsInit());

/*
if (MPD_MM())
{
    std::cout << "RAWWWWWtif " << aRawTiff  << " " << aFullNameOri 
                        << " NbC=" << aNbChanSpec 
                        << " R16=" << RequireBits16 
                        << " ENC=" << ExigNoCompr 
                        << " Cre=" << Create 
                        << " E8B=" <<  ExigB8
                        << " PID=" << mm_getpid()
                        << " TIF=" <<  Tiff_Im::IsTiff(aFullNameOri.c_str(),true)
                        << "\n";
    getchar();
}
*/

   if (IsPostfixed(aFullNameOri))
   {
       ElSTDNS string post = StdPostfix(aFullNameOri);
       if (IsTiffisablePost(post))
          return aFullNameOri;
/*
       if (
                 (post == "HDR")
              || (post == "xml")
              || (post == "RS") || (post == "rs")
              || (post == "PBM") || (post == "PGM") || (post == "PPM")
              || (post == "pbm") || (post == "pgm") || (post == "ppm")
          )
          return aFullNameOri;
*/
   }

   std::string Post8B= "";

   Tiff_Im *aTif = 0;
   bool isTiff = Tiff_Im::IsTiff(aFullNameOri.c_str(),true);
   int aNbChanIn = -1;
   bool Bits16 = RequireBits16 && (!IsPostfixedJPG(aFullNameOri));

   if (isTiff)  // Si fichier tiff , le Bits16 sera calcule plus tard
   {
   }
   else
   {
       cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(aFullNameOri);
       int aNbbMDP = aMDP.NbBits(true);
       if ((aNbbMDP>0) && (aNbbMDP<=8))
       {
           Bits16 = false;
       }
   }
   bool Conv16to8=false;

   if (isTiff || aRawTiff )
   {
       //  if (MPD_MM()) std::cout << "[[[ aaaaAA\n";
       aTif =   aRawTiff                                                           ?
                new Tiff_Im (Elise_Tiled_File_Im_2D::XML(aFullNameOri).to_tiff() ) :
                new Tiff_Im (aFullNameOri.c_str())                                 ;
       //  if (MPD_MM()) std::cout << "]]] aaaaAA\n";
       if ((aTif->phot_interp()==Tiff_Im::RGBPalette) && (aNbChanSpec>=0))
       {
             std::cout << "WARRRRRNNNNNNNNNNN   Color Palette Tiff im may be wrongly interpreted\n";
       }
       aNbChanIn = aTif->nb_chan();
       if (aNbChanSpec<=0)
       {
           aNbChanSpec = aNbChanIn;
       }
       if (aTif->bitpp() <=8)
       {
            Bits16 = false;
       }
       if ((ExigB8) && (aTif->bitpp() > 8))
       {
                 Post8B= "_8B";
                 Conv16to8 = true;
       }
       else if (((aTif->mode_compr() == Tiff_Im::No_Compr)|| (!ExigNoCompr)) && (aNbChanIn==aNbChanSpec))
       {
       //  if (MPD_MM()) std::cout << "YYyyyyYYYYYY\n";
           delete aTif;
           return aFullNameOri;
       }
       //  if (MPD_MM()) std::cout << "ZZZzzz\n";
   }
   else
   {
       if (aNbChanSpec<=0)
       {
          aNbChanSpec=3;
       }
   }


   // std::string aNewName =  aDir+aDirAdd+StdPrefixGen(aNameOri) + ".tif";

   std::string aPost ="_Ch" + ToString(aNbChanSpec) + (Bits16 ? "_16B" :  Post8B) ;
   std::string aNewName =  NameAdapt(aFullNameOri,aPost,false);


   if ((! Create) || ELISE_fp::exist_file(aNewName))
   //  if ((! Create) ||  FileStrictPlusRecent(aNewName,aFullNameOri) )
   {
       return aNewName;
   }
   NameAdapt(aFullNameOri,"",true);

// std::cout << "xcderrr  " << aFullNameOri << "\n"; getchar();

   if (aTif)
   {
       Tiff_Im::PH_INTER_TYPE aPhOut = Tiff_Im::PtDAppuisDense;
       Symb_FNum  aSIn (aTif->in());
       Fonc_Num aFin = aTif->in();
       if (aNbChanSpec==1)
       {
           if (aICNM==0)  aICNM = cInterfChantierNameManipulateur::Glob();
           if (aICNM==0)  aICNM = cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aFullNameOri));
           std::vector<double> aVPds;
           ElArgMain<std::vector<double> > anArg(aVPds,"toto",true);
           std::string aNamePds = aICNM->Assoc1To1("NKS-Assoc-Pds-Channel",NameWithoutDir(aFullNameOri),true);
           anArg.InitEAM(aNamePds,ElGramArgMain::StdGram);
           ELISE_ASSERT(int(aVPds.size()) >= aNbChanIn,"Channel > nb of pds in tiff => Gray");

           double aSomPds = 0;
           aFin = 0.0;
           bool AllP1 = true;
           for (int aKC=0 ; aKC<aNbChanIn ; aKC++)
           {
               double aPds = aVPds[aKC];
               // FromString(aPds,aICNM->Assoc1To2("NKS-Assoc-Pds-Channel",aFullNameOri,ToString(aKC),true));
               aFin  =  aFin + aPds * aSIn.kth_proj(aKC);
               aSomPds  += aPds;
               AllP1 = AllP1 && (aPds==1);
           }
           if (! AllP1) 
              std::cout << "PDS " << aVPds << " for " <<  aFullNameOri << "\n";
           aFin  = aFin / aSomPds;
           aPhOut = Tiff_Im::BlackIsZero;

/*

           if (aNbChanIn==4) // Maybe RGB+IR ? ToDo !
           {
               aFin  = (aSIn.v0() + aSIn.v1()+ aSIn.v2()+ aSIn.kth_proj(3)) / 4;
           }
           else if (aNbChanIn==3)
           {
               aFin  = (aSIn.v0() + aSIn.v1()+ aSIn.v2()) / 3;
           }
           else if (aNbChanIn==1)
           {
           }
           else  if (true)
           {
               aFin = 0;
               for (int aKC=0 ; aKC<aNbChanIn; aKC++)
               {
                  aFin = aFin + aSIn.kth_proj(aKC);
               }
               aFin = aFin / aNbChanIn;
           }
           else
           {
              std::cout  << "For Name " << aFullNameOri << "\n";
              ELISE_ASSERT(false,"Unexpected color combinaison");
           }
*/
       }
       else if (aNbChanSpec==3)
       {
           aPhOut = Tiff_Im::RGB;
           if (aNbChanIn==1)
           {
                aFin = Virgule(aSIn.v0(),aSIn.v0(),aSIn.v0());
           }
           else if (aNbChanIn==2)
           {
                aFin = Virgule(aSIn.v0(),(aSIn.v0()+aSIn.v1())/2,aSIn.v1());
           }
           else if (aNbChanIn==3)
           {
           }
           else  if (true)  // Classique "shift" des fauses couleurs : V B PIR
           {
               aFin = Virgule(aSIn.kth_proj(aNbChanIn-3),aSIn.kth_proj(aNbChanIn-2),aSIn.kth_proj(aNbChanIn-1));
           }
           else
           {
              std::cout  << "For Name " << aFullNameOri << "\n";
              ELISE_ASSERT(false,"Unexpected color combinaison");
           }
       }
       else
       {
           ELISE_ASSERT(false,"AdaptNbChan bad Nb Channel ");
       }


        if ((!RequireBits16) && (aTif->bitpp() > 8))
        {
             int aVSom[3];
             ELISE_COPY(aTif->all_pts(),aFin,VMax(aVSom,aNbChanSpec));

             aFin = (aFin * 255) / ((aNbChanSpec==1) ? aVSom[0] : Virgule(aVSom[0],aVSom[1],aVSom[2]));
        }

        Tiff_Im aNewTF  =   Tiff_Im
                            (
                                aNewName.c_str(),
                                aTif->sz(),
                                Conv16to8 ?  GenIm::u_int1 : aTif->type_el(),
                                Tiff_Im::No_Compr,
                                aPhOut,
                                ArgOpTiffMDP(aFullNameOri)
                            );

         ELISE_COPY(aTif->all_pts(),aFin,aNewTF.out());
   }
   else
   {

       std::string aNameCal = StdNameBayerCalib(aFullNameOri);


      bool DoReech = (aNameCal!= "") ;
      std::string  aNameCoul = (aNbChanSpec==1) ? "G" : "C";
      std::string  aNameReech =  DoReech ? "R" : "B";

       //std::string aStr =    MMDir()+ "bin"+ELISE_CAR_DIR+"MpDcraw "
       std::string aStr =  MM3dBinFile_quotes("MpDcraw")
                           + ToStrBlkCorr(aFullNameOri) + " "
                           + std::string(" Add16B8B=0 ")
                           + std::string(" ConsCol=0 ")
                           + std::string(" ExtensionAbs=None ")
                           + std::string(" 16B=") + (Bits16?"1 ":"0 ")
                           // + ((aNbChanSpec==1)? " GB=1 "  : " CB=1 ")
                           + std::string(" " + aNameCoul + aNameReech + "=1 ")
                           + (DoReech ?  std::string(" Cal=" + aNameCal + " ") : "")
                           + " NameOut=" + aNewName
                           // MPD : je ne comprend plus pourquoi il faut anihiler le flat field dans ces conditions
                           // + " UseFF="  + (  (Bits16||(aNbChanSpec==3)) ? "0" : "1")  // Flat Field en Gray-8Bits
                         ;

       if (! Bits16)
       {
             if (aICNM==0)  aICNM = cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aFullNameOri));
             aStr = aStr + " Gamma=" + aICNM->Assoc1To1("NKS-Assoc-STD-Gama8Bits",NameWithoutDir(aFullNameOri),true);
             aStr = aStr + " EpsLog=" +  aICNM->Assoc1To1("NKS-Assoc-STD-EpsLog8Bits",NameWithoutDir(aFullNameOri),true);
       }



       std::cout << "nnnnnnnn " << aStr << "\n";
        System(aStr.c_str());
   }

   delete aTif;
/*
   if ( FileStrictPlusRecent(aNewName,aFullNameOri,-120) )
   {
      TestDate(aNewName,aFullNameOri);
      std::cout << "FOR FILE " <<aFullNameOri << "\n";
      ELISE_ASSERT(false,"File has probably a date in futur (you may use touch to change that)");
   }
*/

   return aNewName;
}


int TestDupBigTiff(int argc,char ** argv)
{
    std::string aNameTifIn;
    bool BigTif = true;
    bool DoCopy = true;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameTifIn,"Name File In ", eSAM_IsPatFile),
        LArgMain()  << EAM(BigTif,"BigTif",true,"Generate big tif files (Def=true)",eSAM_IsBool)
                    << EAM(DoCopy,"Cp",true,"Do Copy (Def=true)",eSAM_IsBool)
    );

    std::string aNameTifOut = StdPrefix(aNameTifIn) + "-DupBT.tif";

    Tiff_Im aTiffIn(aNameTifIn.c_str());


    int aIntBigTif  =  (BigTif ? 1 : -1);
    Tiff_Im aTiffOut
            (
                aNameTifOut.c_str(),
                aTiffIn.sz(),
                aTiffIn.type_el(),
                aTiffIn.mode_compr(),
                aTiffIn.phot_interp(),
                Tiff_Im::Empty_ARG,
                &aIntBigTif
            );

     if (DoCopy)
     {
        // ELISE_COPY(aTiffIn.all_pts(),aTiffIn.in(),aTiffOut.out());
        Pt2di aSz = aTiffOut.sz();
        ELISE_COPY(rectangle(Pt2di(0,0),aSz),aTiffIn.in(),aTiffOut.out());
     }


    return EXIT_SUCCESS;
}




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
