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



#ifndef _ELISE_GENERAL_COLOUR_H
#define _ELISE_GENERAL_COLOUR_H

class Elise_colour;

Elise_colour operator - (Elise_colour,Elise_colour);
Elise_colour operator + (Elise_colour,Elise_colour);
Elise_colour operator * (REAL,Elise_colour);

class Elise_colour
{
    public :

        REAL eucl_dist (const Elise_colour &); 

        friend Elise_colour operator - (Elise_colour,Elise_colour);
        friend Elise_colour operator + (Elise_colour,Elise_colour);
        friend Elise_colour operator * (REAL,Elise_colour);

        friend  Elise_colour som_pond(Elise_colour C1,REAL pds,Elise_colour C2);
                  // pds (*this) + (1-pds) c2

        static inline Elise_colour rgb(REAL rr,REAL gg,REAL bb)   
        {
               return Elise_colour(rr,gg,bb);
        }

        static Elise_colour cmy(REAL,REAL,REAL);   
        static Elise_colour rand();   
        static Elise_colour gray(REAL);


        static const Elise_colour   red;
        static const Elise_colour   green;
        static const Elise_colour   blue;

        static const Elise_colour   cyan;
        static const Elise_colour   magenta;
        static const Elise_colour   yellow;

        static const Elise_colour   black;
        static const Elise_colour   white;

        static const Elise_colour   medium_gray;
        static const Elise_colour   brown;
        static const Elise_colour   orange;
        static const Elise_colour   pink;
        static const Elise_colour   kaki;
        static const Elise_colour   golfgreen;
        static const Elise_colour   coterotie;
        static const Elise_colour   cobalt;
        static const Elise_colour   caramel;
        static const Elise_colour   bishop;
        static const Elise_colour   sky;
        static const Elise_colour   salmon;
        static const Elise_colour   emerald;

        inline REAL r() const { return _rgb[0];}
        inline REAL g() const { return _rgb[1];}
        inline REAL b() const { return _rgb[2];}
		inline REAL GrayVal() const {return (r()+g()+b()) /3.0;}
		inline REAL MinRGB() const {return ElMin3(r(),g(),b());}
		inline REAL MaxRGB() const {return ElMax3(r(),g(),b());}

        Elise_colour() {*this = rgb(0,0,0);};

        void to_its(REAL & i,REAL & t, REAL & s);
        static Elise_colour its(REAL i,REAL t,REAL s);

    private :

        REAL    _rgb[3];
        inline Elise_colour(REAL rr,REAL gg,REAL bb)
        {
             _rgb[0] = rr; _rgb[1] = gg; _rgb[2] = bb;
        }

        static REAL adj_rvb(REAL);


       friend class Disc_Pal GIF_palette(class ELISE_fp,INT nb);
};

typedef ElList<class Elise_colour> L_El_Col;


class Data_Elise_Palette;

class TYOFPAL
{
      public :

         typedef enum _type_of_pal
         {
              gray,
              disc,
              bicol,
              circ,
              tricol,
              rgb,
              lin1
         }
         t;
};

class  Elise_Palette : public PRC0
{
       friend class Data_Disp_Pallete;
       friend class  Elise_Set_Of_Palette;
       friend class  DE_GW_Not_Comp;
       friend class  Data_Col_Pal;
       friend class  Data_Elise_PS_Disp;
       friend class  PS_Out_RLE_computed;
       friend class  PS_Pts_Not_RLE;
       friend class  Disc_Pal;
       friend class  RGB_Pal;
       friend class  Gray_Pal;
       friend class  Circ_Pal;
       friend class  Video_Win;
       friend class  ElXim;

       public :
           INT  nb();
           Fonc_Num to_rgb(Fonc_Num);
		   INT  dim_pal();
		   enum {MaxDimPal = 3};


       protected :
           Elise_Palette(Data_Elise_Palette *);
           inline Data_Elise_Palette * dep() const
           {
               return SAFE_DYNC(Data_Elise_Palette *,_ptr);
           }
       private :

           class Elise_PS_Palette * ps_comp(const char * name);

           inline TYOFPAL::t type_pal() const;
};
typedef ElList<Elise_Palette> L_El_Palette;

ElList<Elise_Palette> NewLElPal(Elise_Palette aPal);
ElList<Elise_colour> NewLElCol(const Elise_colour & c);


class  Lin1Col_Pal : public Elise_Palette 
{
     public :
         Lin1Col_Pal(Elise_colour,Elise_colour,INT nb);
         class Col_Pal operator () (INT);
};

class  Gray_Pal : public Elise_Palette 
{
     friend class  El_Window;
     public :
         Gray_Pal(INT nb);
         class Col_Pal operator () (INT);
	 private :
		 Gray_Pal(Elise_Palette);
};

class  BiCol_Pal : public Elise_Palette
{
       public :  
           BiCol_Pal (      
                            Elise_colour c0,
                            Elise_colour c1,
                            Elise_colour c2,
                            INT nb1,
                            INT nb2
                     );

          class Col_Pal operator () (INT,INT);
};


class  TriCol_Pal : public Elise_Palette
{
       public :  
           TriCol_Pal (      
                            Elise_colour c0,
                            Elise_colour c1,
                            Elise_colour c2,
                            Elise_colour c3,
                            INT nb1,
                            INT nb2,
                            INT nb3
                     );

          class Col_Pal operator () (INT,INT,INT);
};


class  RGB_Pal : public Elise_Palette
{
       friend class El_Window;
       public :  
           RGB_Pal (INT nb1, INT nb2, INT nb3);

          class Col_Pal operator () (INT,INT,INT);

       private :
         RGB_Pal(Elise_Palette);
};


class  P8COL  
{
    public :
        enum
        {
             white,
             black,
             red,
             green,
             blue,
             cyan,
             magenta,
             yellow
        };
};


// PNCOL : faite pour etre comptaible avec P8COL, succeptible
// d'evoluer par adjonction de nouvelle couleur (d'ou N indetermine)

class  PNCOL  
{
    public :
        enum
        {
             white,
             black,
             red,
             green,
             blue,
             cyan,
             magenta,
             yellow,
             orange,
             pink,
             brown,
             salmon,
             emerald,
             cobalt
        };
};






class Disc_Pal :  public Elise_Palette
{
       friend class Tiff_Im;
       friend class DATA_Tiff_Ifd;
       friend class El_Window;
       public :  
          Disc_Pal      (   
                            L_El_Col,
                            bool reverse = false
                        );

         Disc_Pal(Elise_colour *,INT nb);
         void  getcolors(Elise_colour *);
         Elise_colour *  create_tab_c();  // allocated by NEW_VECTEUR

         // Just for compatiblity with my old clisp data
         static  Disc_Pal    clisp_pal(INT nb);

         static  Disc_Pal    P8COL();
         static  Disc_Pal    PNCOL();
         static  Disc_Pal    PBIN();
         static  Disc_Pal    PCirc(INT = 256); // Palette en teinte, 
         class Col_Pal operator () (INT);
         INT  nb_col();

         class DEP_Discrete * depd() 
               {return SAFE_DYNC(class DEP_Discrete *,_ptr);}

         Disc_Pal reduce_col(Im1D_INT4 lut,INT nb_cible);

      private :
         Disc_Pal();
         Disc_Pal(Elise_Palette);
};


class Circ_Pal : public Elise_Palette
{
       public :  
			friend class El_Window;
            Circ_Pal (   
                         L_El_Col,
                         INT NB,
                         bool reverse = false
                     );

           static Circ_Pal PCIRC6(INT NB);
           class Col_Pal operator () (INT);
	  private :
		   Circ_Pal(Elise_Palette);
};


class Elise_Set_Of_Palette : public PRC0
{
      public    :
           static Elise_Set_Of_Palette  TheFullPalette(); 

           Elise_Set_Of_Palette(L_El_Palette);

           L_El_Palette   lp() const;
           bool operator == (Elise_Set_Of_Palette) const;
           INT  som_nb_col() const;

           void  pal_is_loaded(Elise_Palette) const;
           Elise_Palette pal_of_type(TYOFPAL::t) const;

      protected :
      private   :
};




#endif  // ! _ELISE_GENERAL_COLOUR_H

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
