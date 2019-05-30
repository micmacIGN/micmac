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



#ifndef _ELISE_PRIVATE_POSTSCRIPT_H
#define _ELISE_PRIVATE_POSTSCRIPT_H

#define ElPsPREFIX "El"

extern void  Ps_Hex_Code_Col(std::ofstream & fp,REAL v);
extern void  Ps_Hex_Code_Int(std::ofstream & fp,INT v);

extern void Ps_Real_Prec(std::ofstream & fp,REAL v,INT nb_apr_virg);
extern void Ps_Real_Prec(std::ofstream & fp,Elise_colour c,INT nb_apr_virg);
extern void Ps_Real_Col_Prec(std::ofstream & fp,INT,INT,INT,INT nb_apr_virg);

extern void Ps_Pts(std::ofstream & fp,Pt2dr,INT nb_apr_virg);

class Elise_PS_Palette
{
     public :

        typedef enum {indexed, gray, rgb} ColDevice;

        virtual ~Elise_PS_Palette();
        Elise_PS_Palette(const char * name,class Data_Elise_Palette *);

        void        load(std::ofstream &,bool image);
        virtual  void        load_def(std::ofstream &) = 0 ;
        virtual  void        im_decode(std::ofstream &) = 0 ;
        virtual  void        set_cur_color(std::ofstream &,const int *) = 0;
        virtual ColDevice cdev() const = 0;

        virtual void use_colors(U_INT1 **,INT **,INT dim,INT nb);


        char   *             _name;
        class Data_Elise_Palette * _pal;
        bool                       _used;
        bool                       _image_used;

};

class Data_Elise_PS_Disp : public Data_Elise_Gra_Disp
{


       friend class Data_Elise_PS_Win;
       friend class PS_Display;
       friend class Ps_Ucompr_Filter;
       friend class Ps_PackBits_Filter;
       friend class Ps_LZW_Filter;
       friend class Ps_Multi_Filter;
       friend class PS_Out_RLE_computed;
       friend class PS_Pts_Not_RLE;
       friend class Data_Mat_PS_Window;

       public :
            
            typedef void ( * defV_ul_action)(Data_Elise_PS_Disp *);

             class LGeom : public Mcheck
             {
                public :
                 LGeom(Pt2dr,Pt2dr,Pt2dr,Pt2dr,INT,LGeom * );
                 Pt2dr _p1;
                 Pt2dr _p2;
                 Pt2dr _tr;
                 Pt2dr _sc;
                 INT   _num;
                 LGeom * _next;
                 virtual ~LGeom();
             };

             class  NumGeAlloc : public Mcheck
             {
                public :
                   virtual ~NumGeAlloc();
                   NumGeAlloc(bool box_only);
                   bool  new_get_num( Pt2dr p1,Pt2dr p2,
                                      Pt2dr tr,Pt2dr sc,INT &);
                private :
                  LGeom * _lg; 
                  INT     _num;
                  bool    _box_only;
             };

            class defV  : public Mcheck  // value definition
            {
                  public :

                     defV 
                     (
                          const char * name, 
                          const char * prim,
                          defV_ul_action = NO_ACT_INIT ,  // use action (alwways)
                          defV_ul_action = NO_ACT_INIT   // load action (only first time)
                     );

                     void load_prim(Data_Elise_PS_Disp *);

                     static void act_use_1LigI(Data_Elise_PS_Disp *);
                     static void  no_act_init(Data_Elise_PS_Disp *);

                     static void act_use_F1Ucomp(Data_Elise_PS_Disp *);
                     static void act_use_F1RLE(Data_Elise_PS_Disp *);
                     static void act_use_F1LZW(Data_Elise_PS_Disp *);

                  protected :
                      const char *  _name;
                      defV_ul_action  _act_use;


                  private :
                      defV_ul_action  _act_load;
                      bool           _init;
                      const char *   _proc;
            };

            class defF : public defV  // proc, no arg, def
            {
                  public :

                     defF 
                     (
                          const char * name, 
                          const char * prim,
                          defV_ul_action = NO_ACT_INIT ,  
                          defV_ul_action = NO_ACT_INIT  
                     );
                     void put_prim(Data_Elise_PS_Disp *);

                  private :
            };

            class RdefF : public Mcheck 
            {
                   public :

                     RdefF ();
                     void put_prim(Data_Elise_PS_Disp *,defF *);

                  private :
                     defF   * _defF;
            };

            class defI : private defV  // proc, one int arg, def
            {
                  public :

                     defI (const char * name, const char * prim);
                     void put_prim(Data_Elise_PS_Disp *,INT i);
                     void over_prim(Data_Elise_PS_Disp *);

                  private :
                     int  _i;
            };


       private :
            // class defV;


            REAL   compute_sz_pixel(Pt2dr sz,Pt2dr margin,Pt2di nb,Pt2dr inside_margin);
            Box2dr   box_1_w        (Pt2dr sz,Pt2dr margin,Pt2di k,Pt2dr inside_margin);
            Box2dr   box_all_w      (Pt2dr sz,Pt2dr margin,Pt2di nb,Pt2dr inside_margin);

            // typedef void (defV:: * defV_ul_action)(Data_Elise_PS_Disp *);

            static  const defV_ul_action  NO_ACT_INIT;


            friend class defV;
            friend class defF;
            friend class defI;

           enum {max_str = 500};

           Data_Elise_PS_Disp
           (
                  const char * name,
                  const char * title,
                  Elise_Set_Of_Palette,
                  bool          auth_lzw,
                  Pt2dr         sz_page
           );
           void  comment(const char *);

           virtual ~Data_Elise_PS_Disp();

           void ins_window(Data_Elise_PS_Win *);

           void disp_flush() {}
           void _inst_set_line_witdh(REAL);

           char *                  _name;
           std::ofstream                _fp;
        // data stream (containing  graphic primitives)
           char *                  _name_data;
           std::ofstream                _fd;
        // header stream (containing  global procedures, dictionary ....)
           char *                  _name_header;
           std::ofstream                _fh;


           Elise_Set_Of_Palette   _sop;
           INT                    _offs_bbox;
           Pt2dr                  _p0_box;
           Pt2dr                  _p1_box;
           INT                    _nb_win;


           void add_file(std::ofstream &f,const char * name);

           void prim_win_coord(std::ofstream & f,class Data_Elise_PS_Win *);
           void line(Pt2dr p1,Pt2dr p2);
           void fill_rect(Pt2dr p1,Pt2dr p2);
           void dr_rect(Pt2dr p1,Pt2dr p2);
           void dr_circle(Pt2dr centre,REAL radius);
           void draw_poly(const REAL * x,const REAL *y,INT nb);

           void set_active_window(class Data_Elise_PS_Win *);
           void set_active_palette(Elise_Palette,bool image);
           void set_cur_color(const int *);
        
           Elise_PS_Palette * get_ps_pal(Data_Elise_Palette *);

           void use_conv_colors
                (
                    Data_Elise_Palette *,
                    U_INT1 **,INT **,INT dim,INT nb
                );

           void Lpts_put_prim(bool cste);

           INT                     _nbpal;
           Elise_PS_Palette  **    _teps;
           Pt2dr                   _sz_page;
           INT                      _num_last_act_win;
           Data_Elise_Palette   *  _active_pal;
           Elise_PS_Palette     *  _act_ps_pal;
           bool                    _use_lzw;
           bool                    _use_pckb;

           defF                     _FUcomp;
           defF                     _FRLE;
           defF                     _FLZW;
           defF                     _1LigI;

           defF                     _F1Ucomp;
           defF                     _F1RLE;
           defF                     _F1LZW;

           defF                     _MUcomp;
           defF                     _MRLE;
           defF                     _MLZW;

           defF                     _line;
           defF                     _dr_circ;
           defF                     _dr_rect;
           defF                     _dr_poly;
           defF                     _dr_polyFxy;
           defF                     _DicIm;
           defV                     _MatIm;

           defF                     _StrX;
           defF                     _StrY;
           defF                     _StrC0;
           defF                     _StrC1;
           defF                     _StrC2;
           defF                   * _StrC[3];
           defF                     _LStr85;

           defF                     _LptsImCste;
           defF                     _LptsImInd;
           defF                     _LptsImGray;
           defF                     _LptsImRGB;

           defI                     _x0Im;
           defI                     _y0Im;

           defI                     _txIm;
           defI                     _tyIm;
           defI                     _nbbIm;

           NumGeAlloc               _lclip;
           NumGeAlloc               _lgeo_clip;
};


/**********************************************************************/
/*                                                                    */
/*                Data_Elise_PS_Win                                   */
/*                                                                    */
/**********************************************************************/


class Data_Elise_PS_Win  : public Data_Elise_Gra_Win
{
    public :
        
        friend class Data_Elise_PS_Disp;
        friend class PS_Window;
        friend class Ps_Multi_Filter;
        friend class PS_Out_RLE_computed;
        friend class PS_Pts_Not_RLE;

    private :



        Data_Elise_PS_Win(PS_Display,Pt2di sz,Pt2dr p0,Pt2dr p1,Pt2dr geo_tr,Pt2dr geo_sc);

         Output_Computed * rle_out_comp
         (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & ,
                       Data_Elise_Palette *,
                       bool  OnYDiff
         );

         Output_Computed * pint_cste_out_comp
         (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & ,
                       Data_Elise_Palette *,
                       INT        * 
         );


        Output_Computed * pint_no_cste_out_comp
        (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & ,
                       Data_Elise_Palette *
        );


        void set_active();
        void _inst_set_col(Data_Col_Pal *);
        void _inst_draw_seg(Pt2dr,Pt2dr);
        void _inst_fill_rectangle(Pt2dr,Pt2dr);
        void _inst_draw_rectangle(Pt2dr,Pt2dr);
        void _inst_draw_circle(Pt2dr centre,Pt2dr ray /* ray.x != ray.y */);
        void _inst_draw_polyl(const REAL * x,const REAL *y,INT nb);


        virtual  bool adapt_vect() ;
        virtual  Data_Elise_Gra_Win * dup_geo(Pt2dr tr,Pt2dr sc);


        Pt2dr   _p0_ori;   // in initial unity (= cm for now)
        Pt2dr   _p1_ori;   // in initial unity (= cm for now)


        Pt2dr   _p0;      // bottom left corner (in pica)
        Pt2dr   _p1;      // bottom left corner (in pica)
        Pt2dr   _szpica;  // physical size of whole window  (in pica !)
        Pt2dr   _scale;   // physical size of ecah pixel
        Pt2di   _sz;      // number of pixel

        Pt2dr   _geo_tr;
        Pt2dr   _geo_sc;
        INT     _ps_num;


        PS_Display           _Ptr_psd;
        Data_Elise_PS_Disp * _psd;
        
};


class Ps_Multi_Filter : public Mcheck
{
      public :
         virtual ~Ps_Multi_Filter();
         Ps_Multi_Filter(Data_Elise_PS_Disp *);

         bool put
              (
                   Data_Elise_PS_Win * w,
                   Elise_Palette       p,
                   U_INT1 ***          d,
                   INT                 dim_out,
                   Pt2di               sz,
                   Pt2di               p0,
                   Pt2di               p1,
                   INT                nb_byte_max  // <0 if no limit
              );


         bool put
              (
                   Data_Elise_PS_Win * w,
                   Data_Col_Pal  *     dcp,
                   U_INT1 **           d,
                   Pt2di               sz,
                   Pt2di               p0,
                   Pt2di               p1,
                   INT                nb_byte_max
              );


      protected :

      private :
         Data_Elise_PS_Disp *    _psd;
         INT                     _nb_filter;
         class Ps_Filter *       _filter[3];

         bool put
              (
                   Data_Elise_PS_Win * w,
                   Elise_Palette       p,
                   U_INT1 ***          d,
                   INT                 dim_out,
                   Pt2di               sz,
                   Pt2di               p0,
                   Pt2di               p1,
                   bool                mask,
                   INT                 nb_byte_max
              );
};

/**************************************************************/
/*                                                            */
/*         PS_A85                                             */
/*                                                            */
/**************************************************************/

/*
         A class for coding binary data according to
      ASCII85
*/

class PS_A85
{
     public :
         PS_A85(std::ostream & fd) :
            _fd(fd),
            _nb (0)
         {
             reinit_Base256();
         }



        virtual void put(INT);
        virtual void close_block() ;
	virtual ~PS_A85() {}

       private :

        PS_A85(const PS_A85 &);

        void reinit_Base256();
        void putchar_maxline(char c);

        std::ostream &        _fd;
        INT               _nb;

        double            sB256;
        INT               nB256;

        static   const  INT  coeffB256[4];
        static   const  INT  coeffB85[5];
        void put85(INT);

        enum {max_line = 80};
};

class cConvertBaseXXX
{
    public :
        void PutC(char aC,std::ostream &);
        void PutNC(const void *,int aNbC,std::ostream &);
        void Close(std::ostream &);

        int GetC(std::istream &);
        void  GetNC(void *,int aNbC,std::istream &);
        static cConvertBaseXXX StdBase64();
    private :
        cConvertBaseXXX
        (
              int aBaseIn,   // ex :256
              int aNbBasIn,  // ex : 4
              int aBaseOut,   // ex : 85
              int aNbBaseOut,  // ex : 5
              const std::string & aSetChar, // ex "AB...abc...012...+\" en Base64
              int aNbMaxParLine  // ex : 80  en postcript
        );
        const int           mBaseIn;
        const int           mNbBaseIn;
        int                 mNbCCur;
        long long int       mCurI;
        const int           mBaseOut;
        const int           mNbBaseOut;
        static const int    theMaxNbOut = 10;
        const std::string   mSetChar;
        int                 mLutInv[256];
        long long int       mPowbaseIn[theMaxNbOut];
        long long int       mPowbaseOut[theMaxNbOut];
        const int           mNbMaxParLine;
        int                 mNbInCurLine;
};






#endif // _ELISE_PRIVATE_POSTSCRIPT_H

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
