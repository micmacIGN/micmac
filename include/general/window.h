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



#ifndef _ELISE_X11_INTERFACE_H
#define _ELISE_X11_INTERFACE_H



// Video_Display : a `` classical'' X11 display on X11
//               : a `` ?????''                 on NT

class Clik;
class Grab_Untill_Realeased;
class Video_Win;
class Data_Elise_Video_Win;
class Data_El_Video_Display;
class Data_El_Geom_GWin;
class ElXim;
class PS_Window;
class Data_Elise_PS_Win;

class Video_Win_LawLevelId {};

Elise_Set_Of_Palette GlobPal();
Elise_Set_Of_Palette GlobPal(int aNbR,int aNbV,int aNbB,int aNbGray,int aNbCirc);
Elise_Set_Of_Palette RGB_Gray_GlobPal();


class  Video_Display : public PRC0
{
     friend class Data_Elise_Video_Win;
     friend class Video_Win;

     public :
         Video_Display(const char * name);
         Video_Display(Video_Win_LawLevelId,void * Id);
         void  load(Elise_Set_Of_Palette);
         Clik   clik();
         Clik   clik_press();
         Clik   clik_release();
		 INT  Depth() const; // Nombre de bits par pixel
		 Pt2di  Sz() const; // Nombre de pixels
		 bool TrueCol() const; // false si col-indexe

     private :
         Video_Display(Data_El_Video_Display *);
         Data_El_Video_Display * devd();
         const Data_El_Video_Display * devd() const;
		 
};



// Elise_GWindow  something that can be a display window
// or a postcript window


class El_Window  : public PRC0        ,
                   public Rectang_Object
{
    friend class DE_GW_Not_Comp;
    friend class Data_Param_Plot_1d;
    friend class Graph_8_neigh_Not_Win_Comp;
    friend class Data_El_Geom_GWin;


    public :



        virtual ~El_Window();
        Output out(Elise_Palette,bool OnYDif=false);
        Output ogray();
        Output orgb();
        Output odisc(bool OnYDif=false);
        Output obicol();
        Output ocirc();
        Output olin1();


        Disc_Pal pdisc();
        RGB_Pal  prgb();
        Gray_Pal pgray();
        Circ_Pal pcirc();

        Output out_graph(Line_St,bool sym = true);
        Output out_fr_graph(Line_St);

        El_Window operator  | (El_Window);


//  PW = Sc * (PU-Tr) 
//  PU = Tr + PW /Sc


        El_Window chc(Pt2dr tr,Pt2dr sc);
        Pt2di sz() const;
        virtual Elise_Rect box() const;

        void draw_circle_loc(Pt2dr,REAL,Line_St);  // radius depends from coord
        void draw_circle_abs(Pt2dr,REAL,Line_St);  // radius does not depend from coord

	// Bovin, le fait via un polygone
        void draw_ellipse_loc
             (
	        Pt2dr,REAL A,REAL B,REAL C,Line_St,
		  INT Nb=100);  // radius does not depend from coord

        void draw_seg(Pt2dr,Pt2dr,Line_St);
        void draw_rect(Pt2dr,Pt2dr,Line_St);
        void draw_rect(Box2dr,Line_St);
        void fill_rect(Pt2dr,Pt2dr,Fill_St);

        void draw_poly(const std::vector<Pt2dr> &,Line_St,bool isFerm);
        void draw_poly_ouv(const std::vector<Pt2dr> &,Line_St);
        void draw_poly_ferm(const std::vector<Pt2dr> &,Line_St);


        void hach(ElFifo<Pt2dr> & poly,Pt2dr dir,REAL esp,Line_St);
        void hach(std::vector<Pt2dr> & poly,Pt2dr dir,REAL esp,Line_St);


        void draw_arrow
             (
                Pt2dr, Pt2dr, Line_St Style_axe, Line_St Style_pointe,
                REAL size_pointe, REAL pos = 0.5, REAL teta = (PI/4.0)
             );
        void draw_arrow
             (
                Pt2dr, Pt2dr, Line_St Axe_and_Pointe,
                REAL size_pointe, REAL pos = 0.5, REAL teta = (PI/4.0)
             );




              // cache misere, en attendant d'avoir 
              // installe un systeme de fonte portable correct
        void fixed_string(Pt2dr pt,const char * name, Col_Pal,bool draw_im = false);
		Elise_Set_Of_Palette sop();


         Pt2dr U2W(Pt2dr aP);
         Pt2dr W2U(Pt2di aP);
    protected :
         El_Window(class Data_Elise_Gra_Win *,Pt2dr,Pt2dr);
         El_Window(Data_El_Geom_GWin *);
        inline class Data_El_Geom_GWin * degeow() const 
                {return SAFE_DYNC(class Data_El_Geom_GWin *,_ptr);}

        class Data_Elise_Gra_Win * degraw() const; 

    private :
        Output out(TYOFPAL::t,bool OnYDif=false);
        Elise_Palette palette(TYOFPAL::t);
};



class EliseStdImageInteractor;
class Video_Win   :  public El_Window
{
    friend class ElXim;
    public :

        void DumpImage(const std::string & aName);


         void raise();
         void lower();
         void move_to(const Pt2di&);
         void move_translate(const Pt2di&);

         EliseStdImageInteractor * Interactor();
	 void  SetInteractor(EliseStdImageInteractor *);
         
         static Video_Win  WStd(Pt2di sz,REAL zoom,bool all_pal= true,bool SetClikCoord = true);
         static Video_Win  WStd(Pt2di sz,REAL zoom,Video_Win,bool SetClikCoord = true);
         static Video_Win  WSzMax(Pt2dr aSzTarget,Pt2dr aSzMax);
         static Video_Win  WSzMax(Pt2dr aSzTarget,Pt2dr aSzMax,double & aZoom);
         static Video_Win  LoadTiffWSzMax(const std::string &aNameTiff,Pt2dr aSzMax,double & aZoom);

         static Video_Win *  PtrWStd(Pt2di sz,bool all_pal= true,const Pt2dr & aScale=Pt2dr(1,1));

         static Output  WiewAv(Pt2di sz,Pt2di szmax = Pt2di(500,500));

         void set_sop(Elise_Set_Of_Palette);

         // Video_Win (Pt2di);          

         Video_Win
         (
                Video_Display          ,
                Elise_Set_Of_Palette   ,
                Pt2di                  ,
                Pt2di                  ,
                INT          border_witdh = 5
         );

         typedef enum
         {
             eDroiteH,
             eBasG,
             eSamePos
         }  ePosRel;

         Video_Win
         (
                Video_Win   aSoeur,
                ePosRel     aPos,
                Pt2di       aSz,
                INT         border_witdh = 5
         );

        class HJ_PtrDisplay  display();   //  HJMPD
        class HJ_Window      window();    //  HJMPD


         Video_Win (Video_Win_LawLevelId,void * IdW,void * IdScreen,Pt2di sz);

		 bool operator == (const Video_Win &) const;

        Pt2dr to_user_geom(Pt2dr p);



        void clear();

        void set_title(const char * name);
        void  set_cl_coord(Pt2dr,Pt2dr);
        Video_Win chc(Pt2dr tr,Pt2dr sc,bool SetClikCoord = false);
        Video_Win * PtrChc(Pt2dr tr,Pt2dr sc,bool SetClikCoord = false);


        Video_Win  chc_fit_sz(Pt2dr aSz,bool SetClikCoord = false);




        std::string GetString(const Pt2dr & aP,Col_Pal aColDr,Col_Pal aColErase,const std::string & aStr0="");
        Pt2di SizeFixedString(const std::string aStr);
        // Pos <0 => gauchen Pos >0 => droite , 0 middle
        Pt2di fixed_string_middle(const Box2di & aBox,int aPos,const std::string &  name, Col_Pal,bool draw_im = false);
        Pt2di fixed_string_middle(int aPos,const std::string &  name, Col_Pal,bool draw_im = false);
        Clik   clik_in();
        ElList<Pt2di> GetPolyg(Line_St,INT aButonEnd);
		void grab(Grab_Untill_Realeased &);
		Video_Display    disp();

		// Manipulation d'images
		// Ignorent les coordonnees 

	 void write_image
           (
              Pt2di p0src,
              Pt2di p0dest,
              Pt2di sz,
              INT *** Im,
              Elise_Palette 
           );
      	void load_image(Pt2di p0src,Pt2di p0dest,Pt2di sz); 
      	void load_image(Pt2di p0,Pt2di p1);
      	void load_image();
      	void translate(Pt2di);
      	void image_translate(Pt2di);
        ElXim  StdBigImage();
         
     private :
         static const Pt2dr  _tr00;
         static const Pt2dr  _sc11;

         friend class Data_El_Video_Display;
         class Data_Elise_Video_Win * devw();
         const class Data_Elise_Video_Win * devw() const;
         Video_Win(  class Data_Elise_Video_Win *,
                           Pt2dr tr = _tr00,
                           Pt2dr sc = _sc11
                   );
};

class cFenMenu
{
      public :
         cFenMenu(Video_Win aWSoeur,const Pt2di & aSzCase,const Pt2di & aNb);
         int Get();
         Video_Win W();

         void ColorieCase(const Pt2di & aKse,Col_Pal,int aBrd);
         void StringCase(const Pt2di & aKse,const std::string &,bool center);
         Pt2di  Pt2Case(const Pt2di & aP) const;
         Pt2di  Case2Pt(const Pt2di & aP) const;
      protected :
         Video_Win mW;
      private :
         Pt2di     mSzCase;
         Pt2di      mNb;
};



class cFenOuiNon : public cFenMenu
{
      public :
         cFenOuiNon(Video_Win aWSoeur,const Pt2di & aSzCase);
         bool Get(const std::string &);
      public :
};


class DataElXim;
class ElXim : public PRC0
{
     public :

        friend class Video_Win;

        ElXim(Video_Win,Pt2di,Fonc_Num,Elise_Palette);
        ElXim(Video_Win,Pt2di);
        void load();
        void load(Pt2di   p0src,Pt2di  p0dest,Pt2di  sz);
	void write_image_per(Pt2di   p0src,Pt2di  p0dest,Pt2di  sz); 
		// recopie, periodiquement, la petite image sur la grande image
		// associee a la fenetre
		

	void fill_with_el_image
             (
                 Pt2di p0src,
                 Pt2di p0dest,
                 Pt2di sz, 
		 std::vector<Im2D_INT4> & Images,
                 Elise_Palette
             );

        void read_in_el_image
             ( 
                  Pt2di       aP0Src,
                  Pt2di       aP0Dest,
                  Pt2di       aSz,
                  Im2D_U_INT1 anImR,
                  Im2D_U_INT1 anImG,
                  Im2D_U_INT1 anImB
             );

	void fill_with_el_image
             ( 
                  Pt2di       aP0Src,
                  Pt2di       aP0Dest,
                  Pt2di       aSz,
                  Im2D_U_INT1 anImR,
                  Im2D_U_INT1 anImG,
                  Im2D_U_INT1 anImB
             );



     private :
        DataElXim * dex();
        ElXim(DataElXim *);
};           

class cElImageFlipper
{
	public :
		cElImageFlipper
	        (
		     Pt2di aSz,
		     Fonc_Num aIm1,
		     Fonc_Num aIm2
                );
		void Flip(INT aNbTime,REAL aCadence);
	private :
	    Pt2di     mSz;
            Video_Win *  pW;
	    ElXim        mXIm1;
	    ElXim        mXIm2;
        // ElXim(Video_Win,Pt2di,Fonc_Num,Elise_Palette);
};


class Clik 
{
     public :

        Clik   (Video_Win,Pt2dr,INT,U_INT state);

        Video_Win    _w;
        Pt2dr        _pt;
        INT          _b;
		U_INT 		 _state;

		bool		 b1Pressed() const;
		bool		 b2Pressed() const;
		bool		 b3Pressed() const;
		bool		 controled() const;
		bool		 shifted() const;
};  

class Grab_Untill_Realeased
{
       public :
             Grab_Untill_Realeased(bool ONLYMVT = true);

             virtual void  GUR_query_pointer(Clik,bool) =0;
             virtual void  GUR_button_released(Clik);
					 
             bool OnlyMvmt() {return _OnlyMvmt;}

	     virtual ~Grab_Untill_Realeased() {}

       private  :
             bool _OnlyMvmt;
};                      

class Bitm_Win  : public El_Window 
{
    public :

       Bitm_Win
       (
           const char *,
           Elise_Set_Of_Palette,
           Pt2di sz
       );

       Bitm_Win
       (
           const char *,
           Elise_Set_Of_Palette,
           Im2D_U_INT1
       );


       Im2D_U_INT1 im() const;

        Bitm_Win chc(Pt2dr tr,Pt2dr sc);

        void make_tif(const char * name);
        void make_gif(const char * name);

    private :
         Bitm_Win(Bitm_Win,Pt2dr,Pt2dr);
         class Data_Elise_Bitmaped_Win * debw()
         {
                return (class Data_Elise_Bitmaped_Win *) degraw();
         }
};


class  PS_Display : public PRC0
{
     public : 
         friend class PS_Window;
         friend class Mat_PS_Window;
         friend class Data_Mat_PS_Window;
         friend class Data_Elise_PS_Win;
         friend class Data_Elise_PS_Disp;

         static const REAL picaPcm; 
         static const Pt2dr A4;
         static const Pt2dr A3;
         static const Pt2dr A2;
         static const Pt2dr A1;
         static const Pt2dr A0;



         PS_Display
         (
               const char * name,
               const char * title,
               Elise_Set_Of_Palette,
               bool       auth_lzw = true,
               Pt2dr      sz_page = A4  // in cm
         );
         void comment(const char *);

         PS_Window    w_centered_max(Pt2di sz,Pt2dr margin);


    private :
         class Data_Elise_PS_Disp * depsd()
         {
             return (class Data_Elise_PS_Disp *) _ptr;
         }

};

class  PS_Window  :  public El_Window
{
    public :
       friend class Data_Elise_PS_Disp;
       friend class Data_Mat_PS_Window;

       PS_Window
       (
          PS_Display,
          Pt2di sz,
          Pt2dr p0,
          Pt2dr p1
       );

       PS_Window chc(Pt2dr,Pt2dr);

       static PS_Window  WStd
              (
	           const std::string & aName,
		   Pt2di aSz,
		   Pt2dr aMargin = Pt2dr(2.0,2.0)
              );


    private :
       PS_Window(Data_Elise_PS_Win *,Pt2dr,Pt2dr);
       PS_Window();
       class Data_Elise_PS_Win * depw()
       {
             return (class Data_Elise_PS_Win *) degraw();
       }
};


class Mat_PS_Window : public PRC0
{
      public :

          Mat_PS_Window  (PS_Display,Pt2di sz,Pt2dr margin,Pt2di nb,Pt2dr inside_margin);
          PS_Window operator() (int x,int y);

      private :
};

class DATA_DXF_WRITER;
class Seg2d;

class DXF_Writer :  public PRC0
{
      public :
          DXF_Writer
          (
               const char * ,
               Box2di        ,
               bool         InvY = true
          );


          void PutPt0(Pt2dr);
          void PutSeg(Seg2d s,const char * Layer);
          void PutVertex(Pt2dr,const char * Layer);
          void PutPolyline(const ElFilo<Pt2dr> &,const char * Layer,bool circ = false);

      private  :

          DATA_DXF_WRITER  * ddw();
};





#endif // _ELISE_X11_INTERFACE_H







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
