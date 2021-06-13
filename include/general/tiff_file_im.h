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



#ifndef _ELISE_GENERAL_TIFF_FILE_IM_H
#define _ELISE_GENERAL_TIFF_FILE_IM_H

extern "C"
{
    int el_dcraw_main (int argc, const char **argv);
};



template <class Type> class PackB_IM;



class Tiff_File : public PRC0
{
      public :
          Tiff_File(const char *);
          INT nb_im();
          class Tiff_Im kth_im(INT kth);

      private :
         class DATA_tiff_header * dth();
};

class Arg_Tiff : public PRC0
{
     friend class D_Tiff_ifd_Arg_opt;

     protected :
          Arg_Tiff (class Data_Arg_Tiff *);
     private :
          inline class Data_Arg_Tiff * dat() const;
};
typedef ElList<Arg_Tiff> L_Arg_Opt_Tiff;

L_Arg_Opt_Tiff  ArgOpTiffMDP(const cMetaDataPhoto &,bool SVP=false);
L_Arg_Opt_Tiff  ArgOpTiffMDP(const std::string & aNameF );


extern int DefValueBigTif;

class Tiff_Im : public ElGenFileIm
{
      friend class DATA_tiff_header;
      friend class Tiff_Im_Not_Comp;
      friend class Tiff_Out_Not_Comp;
      friend class DATA_Tiff_Ifd;
      friend class Elise_Tiled_File_Im_2D;
      friend class Tiff_Tiles;


      public :


       static bool IsTiff(const char * name,bool AcceptUnPrefixed = true);


        typedef enum
        {
                LSBYTE = 0x4949,
                MSBYTE = 0x4D4D

        } LMSBYTEF;

        typedef enum
        {
                THE_STD_VERSION = 0x2A,
                BIGTIF_VERSION = 0x2B,
                STD_OFSS_IFD0   = 4,
                BIGTIF_OFSS_IFD0   = 8,
                BIGTIF_K8   = 8,
                BIGTIF_K0   = 0,
                STD_SZ_TAG      = 12,
                BIGTIF_SZ_TAG   = 20


        } VERSION;

        static const tFileOffset UN_INIT_TILE;

        typedef enum
        {
               eBYTE = 1,
               eASCII = 2,
               eSHORT = 3,
               eLONG  = 4,
               eRATIONNAL = 5,
               // EXTENSION Signed de Tiff 6.0
                 eSBYTE = 6,
                 eUNDEFINED  = 7,
                 eSSHORT  = 8,
                 eSLONG  = 9,
                 eSRATIONNAL = 10,
                 eFLOAT = 11,
                 eDOUBLE = 12,

               //  13 14 15 ????
            
               e_LONG8 = 16,
               e_SLONG8 = 17
               // e_IFD8 = 18,

        } FIELD_TYPE;

        static GenIm::type_el  to_Elise_Type_Num(FIELD_TYPE,const char * aNameFile);

        typedef enum
        {
             No_Compr            = 1,
             CCITT_G3_1D_Compr   = 2,
             Group_3FAX_Compr    = 3,
             Group_4FAX_Compr    = 4,
             LZW_Compr           = 5,
             JPEG_Compr          = 6,
             MPD_T6              = 40644,
             PackBits_Compr      = 32773,
             NoByte_PackBits_Compr      = 32774

        }  COMPR_TYPE;
        /*
         * NoByte_PackBits_Compr
         *      Variante, pour les images de + de 8bits,
         *      ou ne redcoupe pas l'image en byte avant de la comprimer
         *      en general bcp + efficace (car les bytes constiutant la meme
         *      valeur de pixel on puet de raison d'etre egaux)
         */
        static const char * name_compr(INT);
        static bool mode_compr_bin(INT);

        typedef enum
        {
             WhiteIsZero   = 0,
             BlackIsZero   = 1,
             RGB           = 2,
             RGBPalette    = 3,
             TranspMask    = 4,
             CMYK          = 5,
             YCbCr         = 6,
             CIELab        = 8,
         PtDeLiaison   = 10531, // Une valeur peu susceptible de telescopage
         // PtDAppuisDense : il y 4 canaux,Pds, xy et z superposables aux images
         PtDAppuisDense  = 10532
        }  PH_INTER_TYPE;

        static const char * name_phot_interp(INT);
        static INT  nb_chan_of_phot_interp(PH_INTER_TYPE);
        static Pt2di  std_sz_tile_of_nbb (INT nbb);

        typedef enum
        {
             No_Unit         = 1,
             Inch_Unit       = 2,
             Cm_Unit         = 3
        }  RESOLUTION_UNIT;
        static const char * name_resol_unit(INT);

        typedef enum
        {
              Chunky_conf  = 1,
              Planar_conf  = 2
        }  PLANAR_CONFIG;
        static const char * name_plan_conf(INT);

        typedef enum
        {
              Unsigned_int  = 1,
              Signed_int    = 2,
              IEEE_float    = 3,
              Undef_data    = 4
        }  SAMPLE_FORMAT;
        static const char * name_data_format(INT);

        typedef enum
        {
              No_Predic = 1,
              Hor_Diff  = 2
        }
        PREDICTOR;


        enum
        {
            MAX_COLOR_PAL = ((1<<16)-1)
        };


        static const char * name_predictor(INT);


      public :

          static  const L_Arg_Opt_Tiff Empty_ARG;

          class AExifTiff_FocalEqui35Length : public Arg_Tiff
          {
               public : AExifTiff_FocalEqui35Length(REAL);
          };
          class AExifTiff_FocalLength : public Arg_Tiff
          {
               public : AExifTiff_FocalLength(REAL);
          };
          class AExifTiff_ShutterSpeed : public Arg_Tiff
          {
               public : AExifTiff_ShutterSpeed(REAL);
          };



          class AExifTiff_Aperture : public Arg_Tiff
          {
               public : AExifTiff_Aperture(REAL);
          };
          class AExifTiff_IsoSpeed : public Arg_Tiff
          {
               public : AExifTiff_IsoSpeed(REAL);
          };
          class AExifTiff_Date : public Arg_Tiff
          {
               public : AExifTiff_Date(const cElDate &);
          };
          class AExifTiff_Camera : public Arg_Tiff
          {
               public : AExifTiff_Camera(const std::string &);
          };





          // Resolution

          class AResol : public Arg_Tiff
          {
               public : AResol(REAL,RESOLUTION_UNIT);
                        AResol(Pt2dr,RESOLUTION_UNIT);
          };

          // Predictor
          class APred : public Arg_Tiff
          {
               public : APred(PREDICTOR);
          };

          // size of tiles,  Row per strip, No strip.

          class ATiles : public Arg_Tiff
          {
               public : ATiles(Pt2di sz_tiles);
          };
          class AStrip : public Arg_Tiff
          {
               public : AStrip(INT row_per_strip);
          };

          class ANoStrip : public Arg_Tiff
          {
               public : ANoStrip();
          };

      // Pour dallage par fichiers  -1,-1 => Pas de dallage
      class AFileTiling : public Arg_Tiff
      {
            public : AFileTiling(Pt2di sz_dalle);
      };




          class AOrientation : public Arg_Tiff
          {
               public : AOrientation(INT);
          };

          // Planar Configuration

          class APlanConf : public Arg_Tiff
          {
               public : APlanConf(PLANAR_CONFIG);
          };

          // Planar Configuration

          class AMinMax  : public Arg_Tiff
          {
               public : AMinMax(U_INT2,U_INT2);
               // Tiff does not handle signed of 4-byte values for
               // these tags. That's a pity
          };

          Tiff_Im(const char *);  // initialize to the first image

          // Binary, Gray level, RGB

          Tiff_Im(  const char                  *,
                    Pt2di                       sz,
                    GenIm::type_el              type,
                    COMPR_TYPE                  compr,
                    PH_INTER_TYPE               phot_interp,
                    L_Arg_Opt_Tiff              l = Empty_ARG,
                    int * BigTiff               = &DefValueBigTif
                 );

      // Cree le fichier si il n'existe pas  ou si  une de
      // ses caracteristique est differente de celle passee
          static Tiff_Im CreateIfNeeded
                 (
                      bool  &                     IsModified,
                      const std::string &         ,
                              Pt2di                       sz,
                              GenIm::type_el              type,
                              COMPR_TYPE                  compr,
                              PH_INTER_TYPE               phot_interp,
                              L_Arg_Opt_Tiff              l = Empty_ARG,
                              int * BigTiff               = &DefValueBigTif
                        );

          // Color Indexed

          Tiff_Im(  const char                  *,
                    Pt2di                       sz,
                    GenIm::type_el              type,
                    COMPR_TYPE                  compr,
                    Disc_Pal                    pal,
                    L_Arg_Opt_Tiff              l = Empty_ARG,
                    int * BigTiff               = &DefValueBigTif
                 );



                  // to create a tiff file

          const char * why_elise_cant_use();
          bool    can_elise_use();
          void show();

          PH_INTER_TYPE  phot_interp();
          Pt2dr resol();
          RESOLUTION_UNIT resunit();
          GenIm::type_el  type_el();
          bool            BigTif() const;

          Disc_Pal              pal();
          Elise_Palette         std_pal(Video_Win);
          Pt2di sz();
          Pt2di sz_tile();
          Pt2di nb_tile();
          Pt2di SzFileTile();
          Pt2di NbTTByTF();
          std::string NameTileFile(Pt2di aITF);



          INT   nb_chan();
          cMetaDataPhoto MDP() ;
          INT   bitpp();
          bool byte_ordered();
          COMPR_TYPE mode_compr();
          PLANAR_CONFIG  plan_conf();
          const char * name();

          Fonc_Num in();
          Fonc_Num in_proj();
          Fonc_Num in(REAL def_out);

          // Renvoie 0 ou 1 + Gray adapt
          Fonc_Num in_bool();
          Fonc_Num in_bool_proj();
          Fonc_Num in_bool(Fonc_Num );

      typedef enum {eModeCoulStd,eModeCoulGray,eModeCoulRGB} eModeCoul;
      typedef enum {eModeNoProl,eModeProlProj,eModeProlDef}  eModeProl;
      Fonc_Num  in_gen(eModeCoul,eModeProl,REAL aDef=0.0);

          Output out();
          PackB_IM<U_INT1> un_load_pack_bit_U_INT1();
          PackB_IM<U_INT2> un_load_pack_bit_U_INT2();
          bool OkFor_un_load_pack_bit_U_INT1();
          bool OkFor_un_load_pack_bit_U_INT2();

      tFileOffset   offset_tile(INT x,INT y,INT kth_ch);
          tFileOffset   byte_count_tile(INT x,INT y,INT kth_ch);

          // N'initialise pas
          std::vector<Im2DGen *>  VecOfIm(Pt2di aSz);
          std::vector<Im2DGen *>  ReadVecOfIm();
          std::vector<Im2D_REAL4>  VecOfImFloat(Pt2di aSz);

          static const ElSTDNS string Str_No_Compr;		// NoCompr
          static const ElSTDNS string Str_CCITT_G3_1D_Compr; 	// CCITTG31D
          static const ElSTDNS string Str_Group_3FAX_Compr;	// FAX3
          static const ElSTDNS string Str_Group_4FAX_Compr;	// FAX4
          static const ElSTDNS string Str_LZW_Compr;		// LZW
          static const ElSTDNS string Str_JPEG_Compr;		// JPEG
          static const ElSTDNS string Str_MPD_T6; 		// MPTDT6
          static const ElSTDNS string Str_PackBits_Compr;	// PackBits
          static const ElSTDNS string Str_NoBytePackBits_Compr;	// PackBits

      static Tiff_Im  StdConv(const ElSTDNS string & Name); // => remis  pour compat  avec certain binaires (=BasicConvStd)
      static Tiff_Im  BasicConvStd(const ElSTDNS string & Name);  // Que les tif ou assim
      static Tiff_Im  UnivConvStd(const ElSTDNS string & Name);   // cree des tifs cache pour les RAW-JPGS (uses StdConvGen)
          // Convertit (avec NameFileStd) les jpg , raw et tif comprs
      static Tiff_Im  StdConvGen(const ElSTDNS string & Name,int aNbChan,bool Cons16B,bool ExigNoCompr = true);
      static Tiff_Im  SpecialFilterSift(const ElSTDNS string & Name);

      // Test dand l'ordre aName + ".tif" et aName
      static std::string GetNameOfFileExist(const std::string & aName);

      static COMPR_TYPE mode_compr(const ElSTDNS string &);
         Tiff_Im(class DATA_Tiff_Ifd *);

         Im2DGen ReadIm();
         static Tiff_Im CreateFromIm(Im2DGen,const std::string &,L_Arg_Opt_Tiff              l = Empty_ARG);
         static Tiff_Im CreateFromIm(std::vector<Im2DGen>,const std::string &,L_Arg_Opt_Tiff              l = Empty_ARG);

     static Tiff_Im Create8BFromFonc(const std::string &,Pt2di,Fonc_Num,COMPR_TYPE);
     static Tiff_Im Create8BFromFonc(const std::string &,Pt2di,Fonc_Num);
     static Tiff_Im LZW_Create8BFromFonc(const std::string &,Pt2di,Fonc_Num);

     static Tiff_Im CreateFromFonc(const std::string &,Pt2di,Fonc_Num,GenIm::type_el);
     static Tiff_Im CreateFromFonc(const std::string &,Pt2di,Fonc_Num,GenIm::type_el,COMPR_TYPE);

           Tiff_Im  Dupl(const std::string&); // Avec meme carac mais nom different

         static void SetDefTileFile(int);
         static int DefTileFile() ;

         // Ne regarde que le nom
         static bool   IsNameInternalTile(const std::string &,cInterfChantierNameManipulateur *);


       private :

         void verif_usable(bool mode_read);
         class DATA_Tiff_Ifd * dtifd();
          Fonc_Num in(bool with_def,REAL def_out);

         static int mDefTileFile ;
};


//  MMDirr + data + Name + .tif
Tiff_Im MMIcone(const std::string &);
Im2D_Bits<1> MMStrIcone(const std::string &);

Output   StdOut(const std::vector<Im2DGen *> &);
Fonc_Num StdInput(const std::vector<Im2DGen *> &);

template<class Type,class TypeBase> Output   StdOut  (std::vector<Im2D<Type,TypeBase> > &);
template<class Type,class TypeBase> Fonc_Num StdInput(std::vector<Im2D<Type,TypeBase> > &);

class cLazyTiffFile
{
    public :
       cLazyTiffFile(const std::string & aNameFile);
        ~cLazyTiffFile();
       Tiff_Im StdImage() const;
       Tiff_Im ImGray8B() const;
    private :
        std::string mNameFile;
        mutable Tiff_Im *   mFile;
        mutable Tiff_Im *   mFileGray8B;
};

Elise_Palette StdPalOfFile(const std::string & aName,Video_Win);

void MakeTiffRed2
     (
          const std::string & aNameFul,
          const std::string & aNameRed
     );


void MakeTiffRed2
     (
          const std::string & aNameFul,
          const std::string & aNameRed,
          GenIm::type_el        aType,
          int                   aDiv,
          bool                  HasVS,
          REAL                  aVSpec
      );

// Pour reduire un fichier binaire (resultat en FAX4)
void MakeTiffRed2Binaire
     (
          const std::string &   aNameFul,
          const std::string &   aNameRed,
          REAL                  aRatio,
          Tiff_Im::COMPR_TYPE,
          GenIm::type_el,
          Pt2di                 aSzTile,
          Pt2di                 aSzRed=Pt2di(-1,-1),
          bool                  DynOType=false
     );

void MakeTiffRed2BinaireWithCaracIdent
     (
          const std::string &   aNameFul,
          const std::string &   aNameRed,
          REAL                  aRatio,
          Pt2di                 aSzRed=Pt2di(-1,-1)
     );


Tiff_Im PastisTif(const std::string &  aName);
std::string PastisNameFileStd(const std::string & aFullNameOri);

// A priori ExigB8 est redondant avec cons16B, mais ai des doutes et pour assurer la compat ....
std::string NameFileStd(const std::string & aFullNameOri,int aNbChan,bool cons16B,bool ExigNoCompr = true,bool Create= true,bool ExigB8=false);
bool IsKnownTifPost(const std::string & aPost);
bool IsKnownJPGPost(const std::string & aPost);
bool IsPostfixedJPG(const std::string & aName);


extern void getPastisGrayscaleFilename(const std::string & aParamDir, const string &i_baseName, int i_resolution, string &o_grayscaleFilename);




extern void test_huff();
#endif // _ELISE_GENERAL_TIFF_FILE_IM_H

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
