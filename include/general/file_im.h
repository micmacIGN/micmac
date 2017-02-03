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
    As I do not know standards for manipulating "Images" of
   dimension "1,2,3,..." (== array), we define a simple format.
*/

//    Elis_File_Im is actually a smart pointer to Data_Elis_File_Im
//  that realy contain  the information.

#ifndef _ELISE_FILEIM_H  // general
#define _ELISE_FILEIM_H



class ElDataGenFileIm;
class Data_GifIm;

class ElGenFileIm :  public PRC0,
                     public Rectang_Object
{
     public :

          virtual ~ElGenFileIm();

      // caracteristique logique :

          INT              Dim()       const;
          const int *      Sz()        const;
          INT              NbChannel() const;
          Pt2di 			Sz2() const;  // Error if Dim != 1


      // caracteristique physique liees a la  representation
      // des valeurs numeriques

          bool       SigneType()      const;
          bool       IntegralType()   const ;
          int        NbBits()         const;
          GenIm::type_el type_el();

    // carateristique d'organisation du fichier

          const int *      SzTile() const;
          bool       Compressed() const;

          Fonc_Num in();
          Fonc_Num in(REAL val);
          Output   out();


     protected :

          ElGenFileIm(ElDataGenFileIm *);

     private :

         virtual Elise_Rect box() const;
         const ElDataGenFileIm * edgfi() const;
         ElDataGenFileIm * edgfi();

};

template <class Type>  Im2D<Type,typename El_CTypeTraits<Type>::tBase> LoadFileIm(ElGenFileIm,Type * = 0);

class Elise_Tiled_File_Im_2D;
class Data_Elise_File_Im;

class Elise_File_Im : public ElGenFileIm
{
      friend class DataGenIm;
      friend class Data_Elise_File_Im;

      public :
   /* Use this constructor for :
       1- Declare a file from another compatible format;
       2- To specify the paramater of a file to create.
   */
         virtual ~Elise_File_Im();
         Elise_File_Im
         (
               const char *     name,
               INT        dim,        // 2 for usual images
               INT *      sz,         // tx,ty for usual images
               GenIm::type_el,        // U_INT1,INT ....
               INT        dim_out,    // 1 for gray level, 3 for RVB ...
               tFileOffset        offset_0,   // size of header to pass
               INT        _szd0 = -1,  // see just down
               bool       create = false      // if does not exist
         );


          // _szd0 : the "physical" size in first dim,  currently
          //        _szd0 = sz[0] and this  assumed when _szd0 is given
          //         the default values -1; however, it can differ
          //         for padding reason (for example, with a 1 bits images,
          //         padded for each line, we may have _szd0 = 16, when
          //         sz[0] = 13


         // to create a pnm,pgm,ppm file
         static Elise_File_Im pbm (const char *,Pt2di  sz,char ** comment = 0);
         static Elise_File_Im pgm (const char *,Pt2di  sz,char ** comment = 0);
         static Elise_File_Im ppm (const char *,Pt2di  sz,char ** comment = 0);


         // to open an alreaduy created  file
         static Elise_File_Im pnm(const char *);


         // for simple 1d-file
         Elise_File_Im
         (
               const char *     name,
               INT        sz,      // tx,ty for usual images
               GenIm::type_el,      // U_INT1,INT ....
               tFileOffset    offset_0 = 0,  // size of header to pass
               bool       create = false      // if does not exist
         );
         // for simple 2d-file
         Elise_File_Im
         (
               const char *     name,
               Pt2di       sz,      // tx,ty for usual images
               GenIm::type_el,      // U_INT1,INT ....
               tFileOffset    offset_0 = 0,  // size of header to pass
               bool       create = false      // if does not exist
         );

         // for simple 3d-file
         Elise_File_Im
         (
               const char *     name,
               Pt3di       sz,      // tx,ty for usual images
               GenIm::type_el,      // U_INT1,INT ....
               tFileOffset    offset_0 = 0,  // size of header to pass
               bool       create = false      // if does not exist
         );

     /* Use this constructor for a file created under Elise.
         Elise_File_Im(const char * name);
     */

         Fonc_Num in();
         Fonc_Num in(REAL);

     // Image file are, by default, always cliped when used as
     // output.
         Output out();

     // Use this if you really do not want cliping. BUT, be sure
     // that you do not get out of the file.
         Output onotcl();

        virtual Elise_Rect box() const;

        Elise_Tiled_File_Im_2D  to_elise_tiled(bool byte_ordered = true);
        Tiff_Im to_tiff(bool byte_ordered = true);

      private  :

         Elise_File_Im(Data_Elise_File_Im *);

         static Elise_File_Im pnm
                (
                   const char *,
                   char **   comment,
                   Pt2di  sz,
                   GenIm::type_el,
                   INT    dim,
                   INT    mode_pnm
                );
          Data_Elise_File_Im * defi() const
          {
                return SAFE_DYNC(Data_Elise_File_Im *,_ptr);
          }
};

class Elise_Tiled_File_Im_2D : public ElGenFileIm
{

      public :
   /* Use this constructor for :
       1- Declare a file from another compatible format;
       2- To specify the paramater of a file to create.
   */
         virtual ~Elise_Tiled_File_Im_2D();
     static const bool DefCLT         ;
     static const bool DefChunk       ;
     static const int  DefOffset0     ;
     static const bool DefCreate      ;
     static const bool DefByteOrdered ;

         Elise_Tiled_File_Im_2D
         (
               const char *     name                   ,
               Pt2di            sz                     ,
               GenIm::type_el   type                   ,
               INT              dim_out                ,
               Pt2di            sz_tiles               ,
               bool             clip_last_tile = DefCLT,         // false ,
               bool             chunk          = DefChunk,       // true  ,
               tFileOffset              offset_0       = DefOffset0,     // 0     ,
               bool             create         = DefCreate,      // false ,
               bool             byte_ordered   = DefByteOrdered  // true
         );

          Fonc_Num in();
          Fonc_Num in(REAL def_out);
          Output out();

          static Elise_Tiled_File_Im_2D HDR(const std::string & aNameHdr);
          static Elise_Tiled_File_Im_2D XML(const std::string & aNameHdr);
          static Elise_Tiled_File_Im_2D Saphir
                 (const char * name_file,const char * name_header);
          static Elise_Tiled_File_Im_2D  sun_raster(const char *);
          static Elise_Tiled_File_Im_2D Thom (const char * name_file);

          Tiff_Im to_tiff();

      private  :

          class DATA_Tiff_Ifd * dtifd();
};



/*************************************************************/
/*************************************************************/
/*************************************************************/
/*************************************************************/

void test_gif(char  * name,Video_Win,Video_Display);


class Gif_Im : public ElGenFileIm
{
      friend class Data_Giff;
      friend class Data_GifIm;
      friend class Gif_Im_Not_Comp;
      friend void instatiate_liste();

      public :
          Gif_Im(const char * name);
          Im2D_U_INT1     im();
          Disc_Pal              pal();
          Fonc_Num              in();
          Fonc_Num              in(INT);
          Pt2di                 sz();

         static  Output create  (
                                     const char *             name,
                                     Pt2di              sz,
                                     Elise_colour *     tec,
                                     INT                nbb
                                );


      private :
           Gif_Im(const char * name,class ELISE_fp fp,class Data_Giff *gh);
           Gif_Im(Data_GifIm *);
           class Data_GifIm * dgi()
                 { return SAFE_DYNC(class Data_GifIm *,_ptr);}

};

typedef ElList<Gif_Im> L_Gif_Im;



class Gif_File : public PRC0
{
    public :
      Gif_File(const char * name);

      INT      nb_im   ()        const;
      Gif_Im   kth_im  (INT)     const;


    private :
       class Data_Giff * dgi() const
             {return SAFE_DYNC(Data_Giff *,_ptr);}
};



/*************************************************************/
/*************************************************************/
/*************************************************************/
/*************************************************************/


class  Tga_Im : public PRC0
{
    friend class Tga_Im_Not_Comp;
    public :

       typedef enum
       {
               col_maped = 0,
               true_col  = 1,
               bw_image  = 2
       } type_of_image;

       typedef enum
       {
              no_compr,
              rle_compr
       } mode_compr;






      Tga_Im(const char * name);
      Fonc_Num              in();
      Fonc_Num              in(INT);

      bool                im_present() const;
      type_of_image       toi()        const;


    private :
       class Data_TGA_File * dtga() const
             {return SAFE_DYNC(Data_TGA_File *,_ptr);}
};



/*************************************************************/
/*************************************************************/
/*************************************************************/
/*************************************************************/

class  Bmp_Im : public PRC0
{
    friend class Bmp_Im_Not_Comp;
    friend class Bmp_Out_Not_Comp;
    public :


       typedef enum
       {
             col_maped,
             true_col
       } type_of_image;

       typedef enum
       {
              no_compr  = 0,
              rle_8bits = 1,
              rle_4bits = 2
       } mode_compr;

      Bmp_Im(const char * name);
      Output              out();
      Fonc_Num              in();
      Fonc_Num              in(INT);
      INT      bpp() const;   // bits per pixel
      Disc_Pal   pal() const ;  // error when bpp() == 24
      mode_compr  compr();
      Pt2di       sz();

      bool                im_present() const;
      // type_of_image       toi()        const;


    private :
       class Data_BMP_File * dbmp() const
             {return SAFE_DYNC(class Data_BMP_File *,_ptr);}
};

extern void test_ps();

class  ThomParam
{
      public :
           ThomParam(const char * name_file);
           Elise_Tiled_File_Im_2D   file(const char * );

      //private :
           std::string ORIGINE,OBJECTIF,DATE,FORMAT;
           INT MAXIMG,MINIMG,mCOULEUR,mCAMERA;
           INT FOCALE,TDI,TAILLEPIX,NBCOL,NBLIG;
           std::string NOM;
           REAL EXPOTIME,DIAPHRAGME;
           INT OFFSET;
           std::string MERE;
           INT BLANC;
       INT BIDON;
           INT BYTEORD;
};

void MakeFileThomVide
     (
          const std::string & aNameVide,
          const std::string& aNamePlein
     );

Im2D_Bits<1> MasqImThom
             (
                const std::string & aNameThom,
                const std::string & aNameFileXML,
                Box2di &
             );

void ThomCorrigeCourrantObscur(Im2D_U_INT2,const Box2di&);

#endif  //  _ELISE_FILEIM_H

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
