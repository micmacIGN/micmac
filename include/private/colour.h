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



#ifndef _ELISE_PRIVATE_COLOUR_H
#define _ELISE_PRIVATE_COLOUR_H






typedef enum
{
   Indexed_Colour,
   True_16_Colour,
   True_24_Colour
} Elise_mode_raster_color;


class Data_Elise_Palette : public RC_Object
{
     friend class Elise_Palette;
     friend class Data_Elise_PS_Disp;
     friend class PS_Pts_Not_RLE;

     public :
         TYOFPAL::t typal() {return _typal;}
          
         INT     i1() const {return _i1;}
         INT     i2()  const{return _i2;}
         INT     nb()  const{return _nb;}

         virtual Elise_colour kth_col(INT i) = 0;
         virtual Elise_colour rkth_col(INT i,REAL);

         virtual void to_rgb(INT ** rgb,INT ** input,INT nb) ;


         virtual bool is_gray_pal();
         virtual void init_true() = 0;

         // compr_indice : used only for 1-D palette where it allows
         // to have a single lut for 
         // "lut : values to [0-nb["  o "lut [0 nb[ to pixel indexes"

         virtual INT compr_indice(INT i) const; 

         virtual void verif_out_put(const Arg_Output_Comp &);

         virtual void verif_values_out(Const_INT_PP,INT nb) const = 0;

         // lutage return a data close to what is needed for
         // example in Ximage; out will have  exactly 
         // the size needed by display-depth

         virtual void lutage (class Data_Elise_Raster_D *,
                              U_INT1 * out,INT x0,INT x1,
                              U_INT2 * lut,
                              Const_INT_PP vals,INT x0_val) const = 0;

         // ilutage return a set of INT independantly of 
         // display's depth

         void ilutage (class Data_Elise_Raster_D *,
                              INT * out,INT nb,
                              U_INT2 * lut,
                              Const_INT_PP vals);

 
         // out is 1-D, U_INT1; if display is an indexed colour
         // call the standard lut, else if the palette is mono-dim
         // just copy the results; else generate an error

         void clutage (class Data_Elise_Raster_D *,
                              U_INT1 * out,INT x0,INT x1,
                              U_INT2 * lut,
                              Const_INT_PP vals,INT x0_val) const;



        INT  ilutage(class Data_Elise_Raster_D *, U_INT2 * lut,const INT * col);
        void verif_value_out(const INT *);
   
         virtual INT dim_pal() const = 0;
         // With PS file, for example, bi -color palette generates RGB values
         // (PS-image do not handle 2 components); default : call "dim_pal"
         virtual INT ps_dim_out() const;

     protected :


         virtual    ~Data_Elise_Palette();
         Data_Elise_Palette(TYOFPAL::t,int NB,INT N1,INT N2);

         INT                 _nb;
         INT                 _i1;
         INT                 _i2;
         TYOFPAL::t          _typal;

         inline     void ok_ind_col(INT i);


         virtual class Elise_PS_Palette * ps_comp(const char * name);
         virtual void  ps_end(class Elise_PS_Palette *,std::ofstream &);
         virtual void _inst_to_rgb(INT ** rgb,INT ** input,INT nb) = 0;
};


/****************************************************************/
/****************************************************************/
/****************************************************************/


class Data_Elise_Set_Of_Palette : public RC_Object
{
    friend class Elise_Set_Of_Palette;

    private :

       Data_Elise_Set_Of_Palette(L_El_Palette);
       L_El_Palette     _lp;
};



class  Data_Disp_Pallete
{

     friend class Data_Disp_Set_Of_Pal;

     public :
            inline U_INT2  * lut_compr() const {return _lut_compr;}
            inline Data_Elise_Palette * dep_of_ddp() {return _dep;}
     private :

       Data_Disp_Pallete(class Data_Elise_Raster_D *,
                         Elise_Palette,
                         unsigned long *);

       ~Data_Disp_Pallete();

       Data_Elise_Palette * _dep;

       U_INT2   * _lut_compr;
       INT      _i1;
       INT      _i2;
};

class Data_Disp_Set_Of_Pal : public RC_Object
{

     public :
         Data_Disp_Pallete * ddp_of_dep
             (Data_Elise_Palette * pal,bool svp = false)
         {
             return (_last_dep==pal) ? _last_ddp : _priv_ddp_of_dep(pal,svp);
         }

         inline  INT nb() { return _nb;}
         inline  Data_Disp_Pallete * kth_ddp(INT i) { return _ddp + i;}
         inline  Elise_Set_Of_Palette  esop() {return _esop;}

         Data_Disp_Set_Of_Pal
         (
              class Data_Elise_Raster_D *,
              Elise_Set_Of_Palette,
              unsigned long *
         );

         inline class Data_Elise_Raster_D * derd(){return _derd;}

         virtual ~Data_Disp_Set_Of_Pal();

         

         INT get_tab_col(Elise_colour *,INT nb_max);

     private :


         Elise_Set_Of_Palette                     _esop;
         Data_Disp_Pallete                       * _ddp;
         INT                                        _nb;
         Data_Disp_Pallete                       * _last_ddp;
         Data_Elise_Palette                      * _last_dep;
         class Data_Elise_Raster_D              * _derd;

         Data_Disp_Pallete * _priv_ddp_of_dep
                             (Data_Elise_Palette *,bool svp = false);
};

class  Disp_Set_Of_Pal : public PRC0
{
   friend void instatiate_liste();
   friend class Data_Elise_Raster_D;

    private :
       Disp_Set_Of_Pal(class Data_Disp_Set_Of_Pal *);
       class Data_Disp_Set_Of_Pal *  ddsop();
};


typedef ElList <Disp_Set_Of_Pal> L_Disp_SOP;


#endif  // ! _ELISE_PRIVATE_COLOUR_H





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
