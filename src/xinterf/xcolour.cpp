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


inline U_INT1  AUC(INT c)
{
   if (c < 0) return 0;
   if (c > 255) return 255;
   return (U_INT1) c;
}


void Ps_Hex_Code_Col(ofstream & fp,Elise_colour c)
{
     Ps_Hex_Code_Col(fp,c.r());
     Ps_Hex_Code_Col(fp,c.g());
     Ps_Hex_Code_Col(fp,c.b());
}

ElList<Elise_colour> NewLElCol(const Elise_colour & c)
{
   return ElList<Elise_colour>() + c;
}


ElList<Elise_Palette> NewLElPal (Elise_Palette aPal)
{
    return ElList<Elise_Palette>() + aPal;
}


/*****************************************************************/
/*                                                               */
/*                  Elise_PS_Palette                             */
/*                                                               */
/*****************************************************************/

Elise_PS_Palette::Elise_PS_Palette(const char * name,Data_Elise_Palette * pal) :
    _name (dup(name)),
    _pal  (pal),
    _used (false),
    _image_used (false)
{
}

Elise_PS_Palette::~Elise_PS_Palette()
{
   DELETE_VECTOR(_name,0);
}

void Elise_PS_Palette::use_colors(U_INT1 ** out,INT ** in,INT dim,INT nb)
{
    for (INT d=0; d<dim ; d++)
        convert(out[d],in[d],nb);
}

void Elise_PS_Palette::load(ofstream & fd, bool image)
{
     _used = true;
     if (image)
        _image_used = true;
     fd << _name << "L" << "\n";
}

/*****************************************************************/
/*                                                               */
/*                  RGB_tab_compil                               */
/*                                                               */
/*****************************************************************/


class RGB_tab_compil  : public Mcheck
{
   public :

     static RGB_tab_compil * NEW_one(INT nb1,INT nb2);
     virtual  ~RGB_tab_compil();
     U_INT1 * _r;
     U_INT1 * _g;
     U_INT1 * _b;
     INT      _nb1;

     void set_kth_col(Elise_colour c,INT i);

   private :
     RGB_tab_compil(INT nb1,INT nb2);
};



RGB_tab_compil::RGB_tab_compil(INT nb1,INT nb2)
{
     _r = NEW_VECTEUR(nb1,nb2,U_INT1);
     _g = NEW_VECTEUR(nb1,nb2,U_INT1);
     _b = NEW_VECTEUR(nb1,nb2,U_INT1);

     _nb1 = nb1;
}



RGB_tab_compil * RGB_tab_compil::NEW_one(INT nb1,INT nb2)
{
    return new RGB_tab_compil(nb1,nb2);
}

RGB_tab_compil::~RGB_tab_compil()
{
    DELETE_VECTOR(_r,_nb1);
    DELETE_VECTOR(_g,_nb1);
    DELETE_VECTOR(_b,_nb1);
}

void RGB_tab_compil::set_kth_col(Elise_colour c,INT i)
{
     _r[i] = round_ni(c.r()*255);
     _g[i] = round_ni(c.g()*255);
     _b[i] = round_ni(c.b()*255);
}

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
/*******                COLOURS                                            *****/
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/



/*****************************************************************/
/*                                                               */
/*                  Elise_colour                                 */
/*                                                               */
/*****************************************************************/

REAL  Elise_colour::eucl_dist(const Elise_colour & c2)
{
     return
              ElSquare(r()-c2.r())
            + ElSquare(g()-c2.g())
            + ElSquare(b()-c2.b());
}

Elise_colour operator - (Elise_colour c1,Elise_colour c2)
{
    return Elise_colour::rgb(c1.r()-c2.r(),c1.g()-c2.g(),c1.b()-c2.b());
}

Elise_colour operator + (Elise_colour c1,Elise_colour c2)
{
    return Elise_colour::rgb(c1.r()+c2.r(),c1.g()+c2.g(),c1.b()+c2.b());
}


Elise_colour operator * (REAL f ,Elise_colour c2)
{
    return Elise_colour::rgb(f*c2.r(),f*c2.g(),f*c2.b());
}




Elise_colour  som_pond(Elise_colour C1,REAL pds,Elise_colour C2)
{
    return pds * C1  + (1-pds) *C2;
}


Elise_colour  Elise_colour::cmy(REAL c,REAL m,REAL y)
{
    return Elise_colour::rgb(1-c,1-m,1-y);
}

Elise_colour  Elise_colour::gray(REAL gray)
{
    return Elise_colour::rgb(gray,gray,gray);
}

Elise_colour  Elise_colour::rand()
{
   return Elise_colour::rgb(NRrandom3(),NRrandom3(),NRrandom3());
}

const Elise_colour Elise_colour::red     = Elise_colour::rgb(1,0,0);
const Elise_colour Elise_colour::green   = Elise_colour::rgb(0,1,0);
const Elise_colour Elise_colour::blue    = Elise_colour::rgb(0,0,1);

const Elise_colour Elise_colour::cyan    = Elise_colour::rgb(0,1,1);
const Elise_colour Elise_colour::magenta = Elise_colour::rgb(1,0,1);
const Elise_colour Elise_colour::yellow  = Elise_colour::rgb(1,1,0);

const Elise_colour Elise_colour::black   = Elise_colour::rgb(0,0,0);
const Elise_colour Elise_colour::white   = Elise_colour::rgb(1,1,1);

const Elise_colour Elise_colour::medium_gray   = Elise_colour::rgb(0.6,0.6,0.6);
const Elise_colour Elise_colour::brown   = Elise_colour::rgb(0.8,0.5,0.2);
const Elise_colour Elise_colour::orange  = Elise_colour::rgb(1.0,0.5,0.0);
const Elise_colour Elise_colour::pink    = Elise_colour::rgb(1.0,0.5,0.5);
const Elise_colour Elise_colour::kaki    = Elise_colour::rgb(0.6,0.7,0.1);
const Elise_colour Elise_colour::golfgreen = Elise_colour::rgb(0.15,0.7,0.5);
const Elise_colour Elise_colour::coterotie = Elise_colour::rgb(0.75,0.0,0.0);
const Elise_colour Elise_colour::cobalt    = Elise_colour::rgb(0.25,0.25,0.75);
const Elise_colour Elise_colour::caramel    = Elise_colour::rgb(0.9,0.75,0.4);
const Elise_colour Elise_colour::bishop     = Elise_colour::rgb(0.3,0.05,0.33);
const Elise_colour Elise_colour::sky     = Elise_colour::rgb(0.5,0.7,0.8);
const Elise_colour Elise_colour::salmon     = Elise_colour::rgb(0.8,0.4,0.3);
const Elise_colour Elise_colour::emerald     = Elise_colour::rgb(0.05,0.8,0.7);




/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
/*******                PALETTES                                           *****/
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

/*****************************************************************/
/*                                                               */
/*                  Data_Elise_Palette                           */
/*                                                               */
/*****************************************************************/

Elise_PS_Palette * Data_Elise_Palette::ps_comp(const char *)
{
     El_Internal.ElAssert(false,EEM0<<"ps_comp Uninplemented PS Pallete");
     return 0;
}


INT  Data_Elise_Palette::ps_dim_out() const
{
     return dim_pal();
}

void Data_Elise_Palette::ps_end(Elise_PS_Palette *,ofstream &)
{
     El_Internal.ElAssert(false,EEM0<<"ps_end Uninplemented PS Pallete");
}

Data_Elise_Palette::Data_Elise_Palette(TYOFPAL::t TYPAL,INT NB,INT I1,INT I2)  :
      _nb    (NB) ,
      _i1    (I1),
      _i2    (I2),
      _typal (TYPAL)
{
}

Data_Elise_Palette::~Data_Elise_Palette()
{
}


void Data_Elise_Palette::ok_ind_col(INT i)
{
     ASSERT_TJS_USER((i>=0)&&(i<_nb),"invalide colour index");
}

INT Data_Elise_Palette::compr_indice(INT i)  const
{
  return i;
}


void Data_Elise_Palette::verif_out_put(const Arg_Output_Comp & arg)
{
   Tjs_El_User.ElAssert
   (
      arg.fonc()->idim_out() >= dim_pal(),
      EEM0 << "Unsifficent fonc-dim for palette "
           << " func-dim = " <<  arg.fonc()->idim_out()
           << "; dim pal = " <<   dim_pal()
   );
}


Elise_colour Data_Elise_Palette::rkth_col(INT ,REAL)
{
    elise_internal_error("Data_Elise_Palette::rkth_col",__FILE__,__LINE__);

    return Elise_colour::rgb(0,0,0);
}

bool Data_Elise_Palette::is_gray_pal()
{
     return false;
}


INT  Data_Elise_Palette::ilutage
     (  Data_Elise_Raster_D * derd,
        U_INT2 * lut,
        const INT * col
     )
{
   const INT *pc[3] = {col,col+1,col+2};

   U_INT2 icoul[2];
   U_INT1 * u1ic = (U_INT1 *) icoul;

   lutage(derd,u1ic,0,1,lut,pc,0);

   switch (derd->_cmod)
   {
        case Indexed_Colour :
             return u1ic[0];
        break;

        case True_16_Colour :
             return icoul[0];
        break;

        case True_24_Colour :
             return derd->rgb_to_24(u1ic[derd->_r_ind],
                                    u1ic[derd->_g_ind],
                                    u1ic[derd->_b_ind]);
        break;
   }

   return -188998;
}

void Data_Elise_Palette::verif_value_out(const INT * col)
{
   const INT *pc[3] = {col,col+1,col+2};
   verif_values_out(pc,1);
}


void  Data_Elise_Palette::ilutage
     (
        Data_Elise_Raster_D * derd,
        INT * out,INT nb,
        U_INT2 * lut,
        Const_INT_PP vals
     )
{
    const INT sz_buf = 50;
    U_INT2 buf_loc[2 * sz_buf];

    for (INT i0=0; i0<nb ; i0+=sz_buf )
    {
       INT i1 = ElMin(nb,i0+sz_buf);
       lutage
       (
           derd,
           (U_INT1 *)buf_loc,
           0,i1-i0,lut,vals,i0
       );

       switch (derd->_cmod)
       {
            case Indexed_Colour :
            {
                 U_INT1 * b1 = ((U_INT1 *) buf_loc) -i0;
                 for (INT i=i0 ; i<i1 ; i++)
                     out[i] = b1[i];
            }
            break;

            case True_16_Colour :
            {
                 U_INT2 * b2 =  buf_loc -i0;
                 for (INT i=i0 ; i<i1 ; i++)
                     out[i] = b2[i];
            }
            break;

            case True_24_Colour :
            {
                 U_INT1 * b1 = ((U_INT1 *) buf_loc);

                 for (INT i=i0 ; i<i1 ; i++,b1+= derd->_byte_pp)
                 {
                     out[i] =  derd->rgb_to_24(  b1[derd->_r_ind],
                                                 b1[derd->_g_ind],
                                                 b1[derd->_b_ind]
                                              );
                 }
            }
            break;
       }

    }
}

void  Data_Elise_Palette::clutage
     (
        Data_Elise_Raster_D * derd,
        U_INT1 * out,INT X0,INT X1,
        U_INT2 * lut,
        Const_INT_PP vals,
        INT X0_vals
     )  const
{
    if (derd->_cmod == Indexed_Colour)
    {
       lutage(derd,out,X0,X1,lut,vals,X0_vals);
       return;
    }

    if (dim_pal() == 1)
    {
        out += X0 ;
        const INT  * v = vals[0] + X0_vals;
        INT nb = X1-X0;

        for (INT i=0 ; i<nb ; i++)
            out[i] = v[i];
        return;
    }


    elise_internal_error("invalid clutage",__FILE__,__LINE__);
}


void Data_Elise_Palette::to_rgb(INT ** rgb,INT ** input,INT nb)
{
      verif_values_out(input,nb);
      init_true();
      _inst_to_rgb(rgb,input,nb);
}



/*****************************************************************/
/*                                                               */
/*                  DEP_Mono_Canal                               */
/*                                                               */
/*****************************************************************/



class Elise_Indexed_PS_Palette : public Elise_PS_Palette
{
   public :
      Elise_Indexed_PS_Palette
      (
             const char * name,
             Data_Elise_Palette * pal,
             INT                  nb_usable
      )  :
         Elise_PS_Palette(name,pal),
         _nb_used (0)
      {
          set_cste(_lut,(INT)unused,256);

         // Stupid, but corrige somme buggy printers
          for (INT k=0; k<nb_usable; k++)
               use_colors(k);

      }

      enum {unused = -1};

      INT  use_colors(INT v)
      {
           if (_lut[v] == unused)
           {
              _lut_inv[_nb_used] = v;
              _lut[v] = _nb_used++;
           }
           return _lut[v];
      }

      virtual void use_colors(U_INT1 ** out,INT ** in,INT, INT nb)
      {
          for (INT i=0; i<nb ; i++)
             out[0][i] =  use_colors(in[0][i]);
      }

      INT  _lut[256];
      INT  _lut_inv[256];
      INT  _nb_used;

      void im_decode(ofstream & fp)
      {
           fp << "[0 256]";
      }

      void load_def(ofstream & fp)
      {
           fp << _name << " setcolorspace ";
      };

      void set_cur_color(ofstream &fp,const INT * v)
      {
           fp <<  use_colors(v[0]) << " setcolor\n";
      }

      virtual ColDevice cdev() const {return indexed;}
};



class DEP_Mono_Canal : public  Data_Elise_Palette
{
     protected :
         DEP_Mono_Canal(TYOFPAL::t,int NB,INT N1,INT N2);
         virtual INT compr_indice(INT) const ;

         virtual void verif_values_out(Const_INT_PP,INT nb) const;
         virtual void lutage (Data_Elise_Raster_D *,
                              U_INT1 * out,INT x0,INT x1,
                              U_INT2 * lut,
                              Const_INT_PP vals,INT x0_val) const;
         virtual INT dim_pal() const {return 1;}

         void init_true();
         virtual ~DEP_Mono_Canal()
         {
             if (_true_c)
                 delete _true_c;
         }

         RGB_tab_compil * _true_c;

         void mono_canal_ps_end(Elise_PS_Palette *geps,ofstream & fp,bool );

         virtual void _inst_to_rgb(INT ** rgb,INT ** input,INT nb);
};



void  DEP_Mono_Canal::mono_canal_ps_end
      (Elise_PS_Palette *geps,ofstream & fp,bool real)
{
     Elise_Indexed_PS_Palette  * eip = (Elise_Indexed_PS_Palette *) geps;

     if (eip->_nb_used)
     {
        fp << "/"
           << eip->_name
           << " [/Indexed /DeviceRGB "
           << (eip->_nb_used-1)
           << "<\n";

        for (int i = 0; i<eip->_nb_used; i++)
        {
              fp << " ";
              if (real)
                 Ps_Hex_Code_Col(fp, rkth_col(0,eip->_lut_inv[i]/256.0));
              else
                 Ps_Hex_Code_Col(fp, kth_col(eip->_lut_inv[i]));
              fp << "\n";
        }

        fp << ">] def \n";
     }
}


DEP_Mono_Canal::DEP_Mono_Canal
       (TYOFPAL::t TYPAL,INT NB,INT I1,INT I2) :
       Data_Elise_Palette(TYPAL,NB,I1,I2),
       _true_c (0)
{
     ASSERT_TJS_USER(NB > 1,"need at leats two colour for palettes");
}




void DEP_Mono_Canal::verif_values_out(Const_INT_PP vals,INT nb) const
{
    ASSERT_USER
    (
         values_in_range(vals[0],nb,_i1,_i2),
         "values out of input range palette"
    );
}

INT DEP_Mono_Canal::compr_indice(INT k) const
{
    return  ((k-_i1) * _nb) / (_i2-_i1);
}

void DEP_Mono_Canal::lutage (  Data_Elise_Raster_D * derd,
                               U_INT1 * out,INT x0,INT x1,
                               U_INT2 * lut,
                               Const_INT_PP vals,INT x0_val) const
{
    out += x0 * derd->_byte_pp;
    const INT  * v = vals[0] + x0_val;
    INT nb = x1-x0;

    switch(derd->_cmod)
    {
         case  Indexed_Colour :
         {
               while (nb--)
                   *(out++) = (U_INT1) lut[*(v++)];
         }
         break;

         case True_16_Colour :
         {
               U_INT2 *  out2 = (U_INT2 *) out;
               while (nb--)
                   *(out2++) = lut[*(v++)];
         }
         break;

         case True_24_Colour :
         {
               INT bpp = derd->_byte_pp;
               while (nb--)
               {
                   out[derd->_r_ind] = _true_c->_r[*v];
                   out[derd->_g_ind] = _true_c->_g[*v];
                   out[derd->_b_ind] = _true_c->_b[*(v++)];
                   out +=  bpp;
               }
         }
         break;
    }
}


void DEP_Mono_Canal::init_true()
{
     if (! _true_c)
     {
          _true_c = RGB_tab_compil::NEW_one(_i1,_i2);
          for (int i=_i1; i<_i2 ; i++)
          {
               Elise_colour c = rkth_col(i,(i-_i1) / (_i2-_i1+1.0));
               _true_c->set_kth_col(c,i);
          }
     }
}


void DEP_Mono_Canal::_inst_to_rgb(INT ** rgb,INT ** input,INT nb)
{
     for (INT k =0; k<nb ; k++)
     {
        INT v = mod256(input[0][k]);
        rgb[0][k] = AUC(_true_c->_r[v]);
        rgb[1][k] = AUC(_true_c->_g[v]);
        rgb[2][k] = AUC(_true_c->_b[v]);
     }
}


/*****************************************************************/
/*                                                               */
/*                  DEP_Interpole                                */
/*                                                               */
/*****************************************************************/


class DEP_Interpole : public  DEP_Mono_Canal
{
     friend class Lin1Col_Pal;

     public :
     protected :

         DEP_Interpole
         (
                TYOFPAL::t,
                Elise_colour,
                Elise_colour,
                INT NB      ,
                INT I1      ,
                INT I2
         );

     private :
         virtual Elise_colour  kth_col(INT i);
         virtual Elise_colour rkth_col(INT i,REAL p);

          Elise_colour _c1;
          Elise_colour _c2;

         Elise_PS_Palette * ps_comp(const char * name)
         {
              return  new Elise_Indexed_PS_Palette(name,this,256);
         }

         void ps_end(Elise_PS_Palette * p,ofstream & f)
         {
              mono_canal_ps_end(p,f,true);
         }

};

DEP_Interpole::DEP_Interpole
       (TYOFPAL::t TYPAL,Elise_colour C1,Elise_colour C2,INT NB,INT I1,INT I2) :
       DEP_Mono_Canal(TYPAL,NB,I1,I2),
       _c1  (C1),
       _c2  (C2)
{
}

Elise_colour DEP_Interpole::kth_col(INT i)
{
    ok_ind_col(i);
    return   som_pond(_c1,1.0- i/(double) (_nb-1),_c2);
}

Elise_colour DEP_Interpole::rkth_col(INT,REAL p)
{
      return   som_pond(_c1,1-p,_c2);
}


           /*---------------------*/
           /*     MonoCol_Pal     */
           /*---------------------*/


Lin1Col_Pal::Lin1Col_Pal(Elise_colour c1,Elise_colour c2,int NB) :
      Elise_Palette(new DEP_Interpole(TYOFPAL::lin1,c1,c2,NB,0,256))
{
}

Col_Pal Lin1Col_Pal::operator ()(INT c1)
{
    return Col_Pal(*this,c1);
}



/*****************************************************************/
/*                                                               */
/*                  DEP_Gray_Level                               */
/*                                                               */
/*****************************************************************/


class Elise_Gray_PS_Palette : public Elise_PS_Palette
{
   public :
      Elise_Gray_PS_Palette
      (
             const char * name,
             Data_Elise_Palette * pal
      )  :
         Elise_PS_Palette(name,pal)
      {}

      void im_decode(ofstream & fp)
      {
           fp << "[0 1]";
      }

      void load_def(ofstream & fp)
      {
          fp << "/DeviceGray setcolorspace ";
      };

      void set_cur_color(ofstream & fp,const INT * v)
      {
           Ps_Real_Prec(fp, v[0] /256.0,3);
           fp <<   " setgray\n";
      }

      virtual ColDevice cdev() const {return gray;}
};


class DEP_Gray_Level : public DEP_Interpole
{
    friend class Gray_Pal;

    public :
    private :
         DEP_Gray_Level(INT nb);


         virtual void lutage (Data_Elise_Raster_D *,
                              U_INT1 * out,INT x0,INT x1,
                              U_INT2 * lut,
                              Const_INT_PP vals,INT x0_val) const;

         bool is_gray_pal()
         {
              return true;
         }

      Elise_PS_Palette * ps_comp(const char * name)
      {
           return  new Elise_Gray_PS_Palette(name,this);
      }

      void ps_end(Elise_PS_Palette *,ofstream &){};
};



DEP_Gray_Level::DEP_Gray_Level(INT NB) :
     DEP_Interpole
     (
         TYOFPAL::gray,
          Elise_colour::black,
          Elise_colour::white,
          NB                 ,
          0                  ,
          256
     )
{
}


void DEP_Gray_Level::lutage (  Data_Elise_Raster_D * derd,
                               U_INT1 * out,INT x0,INT x1,
                               U_INT2 * lut,
                               Const_INT_PP vals,INT x0_val) const
{

    if (derd->_cmod == True_24_Colour)
    {
         out += x0 * derd->_byte_pp;
         const INT  * v = vals[0] + x0_val;
         INT nb = x1-x0;

         INT bpp = derd->_byte_pp;
         INT r_ind = derd->_r_ind;
         INT g_ind = derd->_g_ind;
         INT b_ind = derd->_b_ind;
         while (nb--)
         {
              out[r_ind] =  // a case where do not need _r_ind...
              out[g_ind] =
              out[b_ind] = _true_c->_b[*(v++)];
              out += bpp;
         }
    }
    else
       DEP_Mono_Canal::lutage(derd,out,x0,x1,lut,vals,x0_val);
}

           /*------------------*/
           /*     Gray_Pal     */
           /*------------------*/


Gray_Pal::Gray_Pal(int NB) :
      Elise_Palette(new DEP_Gray_Level(NB))
{
}

Col_Pal Gray_Pal::operator ()(INT c1)
{
    return Col_Pal(*this,c1);
}

Gray_Pal::Gray_Pal(Elise_Palette p) :
      Elise_Palette(p.dep())
{
}


/*****************************************************************/
/*                                                               */
/*                  DEP_Discrete                                 */
/*                                                               */
/*****************************************************************/

Elise_colour * alloc_tab_coul(L_El_Col l,bool reverse)
{

    INT nb = l.card();
    Elise_colour * res = NEW_VECTEUR(0,nb,Elise_colour);
    for (INT i=0; i<nb ; i++)
    {
        res[reverse ?(nb-i-1):i] = l.car();
        l = l.cdr();
    }
    return res;
}

        //===============================

class DEP_Discrete : public  DEP_Mono_Canal
{
    friend class Disc_Pal;

    public :
    private :
         DEP_Discrete(L_El_Col,bool reverse);
         DEP_Discrete(const Elise_colour *,INT NB);
         virtual Elise_colour kth_col(INT i);

         Elise_colour * _tabc;
         virtual Elise_colour rkth_col(INT i,REAL p);
         virtual ~DEP_Discrete() { DELETE_VECTOR(_tabc,0); }

         void  getcolors(Elise_colour * tabc)
         {
              memcpy(tabc,_tabc,nb()*sizeof(Elise_colour));
         }


         Elise_PS_Palette * ps_comp(const char * name)
         {
              return  new Elise_Indexed_PS_Palette(name,this,_nb);
         }

         void ps_end(Elise_PS_Palette * p,ofstream & f)
         {
              mono_canal_ps_end(p,f,false);
         }

};



DEP_Discrete::DEP_Discrete(L_El_Col l,bool reverse = false) :
    DEP_Mono_Canal (TYOFPAL::disc,l.card(),0,l.card()),
    _tabc          (alloc_tab_coul(l,reverse))
{
}

DEP_Discrete::DEP_Discrete(const Elise_colour * TABC,INT NB) :
    DEP_Mono_Canal (TYOFPAL::disc,NB,0,NB),
    _tabc          (NEW_VECTEUR(0,NB,Elise_colour))
{
    for ( INT i=0 ; i<NB ; i++)
        _tabc[i] = TABC[i];
}

Elise_colour DEP_Discrete::kth_col(INT i)
{
    ok_ind_col(i);
    return _tabc[i];
}

Elise_colour DEP_Discrete::rkth_col(INT i,REAL)
{
    return _tabc[i];
}


           /*------------------*/
           /*     Disc_Pal     */
           /*------------------*/

Disc_Pal::Disc_Pal(L_El_Col l,bool reverse ) :
    Elise_Palette (new DEP_Discrete(l,reverse))
{
}

Disc_Pal::Disc_Pal(Elise_colour * TABC,INT NB)   :
    Elise_Palette (new DEP_Discrete(TABC,NB))
{
}

Disc_Pal::Disc_Pal()   :
    Elise_Palette (0)
{
}

Disc_Pal::Disc_Pal(Elise_Palette p) :
      Elise_Palette(p.dep())
{
}

INT Disc_Pal::nb_col()
{
    return depd()->nb();
}

void Disc_Pal::getcolors(Elise_colour * tabc)
{
     depd()->getcolors(tabc);
}

Elise_colour * Disc_Pal::create_tab_c()
{
     Elise_colour * tabc = NEW_VECTEUR(0,nb_col(),Elise_colour);
     depd()->getcolors(tabc);
     return tabc;
}


Disc_Pal  Disc_Pal::reduce_col(Im1D_INT4 lut,INT nb_cible)
{
     ASSERT_TJS_USER
     (
         lut.tx() == depd()->nb(),
         "Size of lut != number of colours init in Disc_Pal::reduce_col"
     );

     nb_cible = ElMin(nb_cible,depd()->nb());


     INT * nearest = lut.data();
     INT nb_init = lut.tx();

     Elise_colour * col_in = depd()->_tabc;
     Elise_colour * col_out = NEW_VECTEUR(0,nb_cible,Elise_colour);
     REAL         * dist    =  NEW_VECTEUR(0,nb_init,REAL);


     // [1] Select arbritrary the first colour as selected target

     for (int i = 0; i < nb_init ; i++)
     {
         nearest[i] = 0;
         dist[i] = col_in[0].eucl_dist( col_in[i]);
     }
     col_out[0] = col_in[0];

    // [2] complete iteratively with the most eloigned colour


     for (int i_targ =1 ; i_targ < nb_cible ; i_targ ++)
     {
          // [2.1] get the most eloigned colour
          REAL d_max = -1;
          INT  i_max = -1;

          {
	      INT i_init;
              for (i_init = 0; i_init < nb_init ; i_init++)
                  if (dist[i_init] > d_max)
                  {
                       d_max = dist[i_init];
                       i_max = i_init;
                  }
          }

          // [2.2]
          Elise_colour new_c = col_in[i_max];
          col_out[i_targ] = new_c;

          {
	      INT i_init;
              for (i_init = 0; i_init < nb_init ; i_init++)
              {
                  REAL d= new_c.eucl_dist(col_in[i_init]);
                  if ((d < dist[i_init]) || (i_init==i_max))
                  {
                       dist[i_init] = d;
                       nearest[i_init] =  i_targ;
                  }
              }
          }
     }

     DELETE_VECTOR(dist,0);

     Disc_Pal res (col_out,nb_cible);
     DELETE_VECTOR(col_out,0);
     return res;
}

Disc_Pal  Disc_Pal::P8COL()
{
    return  Disc_Pal
            (
                                NewLElCol(Elise_colour::white)
                                  + Elise_colour::black
                                  + Elise_colour::red
                                  + Elise_colour::green
                                  + Elise_colour::blue
                                  + Elise_colour::cyan
                                  + Elise_colour::magenta
                                  + Elise_colour::yellow,
                                true
            );
};


Disc_Pal  Disc_Pal::PNCOL()
{
    return  Disc_Pal
            (
                                NewLElCol(Elise_colour::white)
                                  + Elise_colour::black
                                  + Elise_colour::red
                                  + Elise_colour::green
                                  + Elise_colour::blue
                                  + Elise_colour::cyan
                                  + Elise_colour::magenta
                                  + Elise_colour::yellow
                                  + Elise_colour::orange
                                  + Elise_colour::pink
                                  + Elise_colour::brown
                                  + Elise_colour::salmon
                                  + Elise_colour::emerald
                                  + Elise_colour::cobalt

                               ,  true
            );
};


Disc_Pal  Disc_Pal::PCirc(INT aNb)
{
   Elise_colour * aTab = new Elise_colour [aNb];

   for (INT aK=0; aK< aNb ; aK++)
       aTab[aK] =  Elise_colour::its(0.5,aK/(REAL)aNb,1);

   Disc_Pal aRes(aTab,aNb);
   delete [] aTab;
   return aRes;
}







Disc_Pal  Disc_Pal::PBIN()
{
    return  Disc_Pal
            (
                 NewLElCol(Elise_colour::white) + Elise_colour::black,
                 true
            );
};

Col_Pal Disc_Pal::operator ()(INT c1)
{
    return Col_Pal(*this,c1);
}

Disc_Pal  Disc_Pal::clisp_pal(INT nb)
{
    nb = ElMax(nb,30);

    Elise_colour * tabc = new Elise_colour [nb];

    for (int i=0 ; i < nb ; i++)
        tabc[i] = Elise_colour::rgb(0.0,0.0,0.0);


    tabc[0] = Elise_colour::white;
    tabc[1] = Elise_colour::black;
    tabc[2] = Elise_colour::yellow;
    tabc[3] = Elise_colour::medium_gray;
    tabc[4] = Elise_colour::brown;
    tabc[5] = Elise_colour::blue;;
    tabc[6] = Elise_colour::magenta;
    tabc[7] = Elise_colour::green;
    tabc[8] = Elise_colour::cyan;
    tabc[9] = Elise_colour::red;
    tabc[10] = Elise_colour::orange;
    tabc[11] = Elise_colour::pink;
    tabc[12] = Elise_colour::kaki;
    tabc[13] = Elise_colour::golfgreen;
    tabc[14] = Elise_colour::coterotie;
    tabc[15] = Elise_colour::cobalt;
    tabc[16] = Elise_colour::caramel;
    tabc[17] = Elise_colour::bishop;
    tabc[18] = Elise_colour::sky;
    tabc[19] = Elise_colour::salmon;
    tabc[20] = Elise_colour::emerald;

    Disc_Pal p (tabc,nb);
    delete [] tabc;
    return p;
};

/*****************************************************************/
/*                                                               */
/*                  DEP_Circ                                     */
/*                                                               */
/*****************************************************************/

class DEP_circ : public  DEP_Mono_Canal
{
    friend class Circ_Pal;

    public :
    private :
         DEP_circ(L_El_Col,INT nb,bool reverse = false);
         virtual Elise_colour kth_col(INT i);
         virtual Elise_colour rkth_col(INT,REAL);

         Elise_colour * _tabc;
         INT            _nb_c0;

         virtual void verif_values_out(Const_INT_PP,INT) const {};
         virtual void lutage (Data_Elise_Raster_D *,
                              U_INT1 * out,INT x0,INT x1,
                              U_INT2 * lut,
                              Const_INT_PP vals,INT x0_val) const;

         virtual ~DEP_circ();

         Elise_PS_Palette * ps_comp(const char * name)
         {
              return  new Elise_Indexed_PS_Palette(name,this,256);
         }

         void ps_end(Elise_PS_Palette * p,ofstream & f)
         {
              mono_canal_ps_end(p,f,true);
         }
};


DEP_circ::DEP_circ(L_El_Col l,INT nb,bool reverse) :
    DEP_Mono_Canal(TYOFPAL::circ,nb,0,256),
    _tabc          (alloc_tab_coul(l,reverse)),
    _nb_c0         (l.card())
{
}

DEP_circ::~DEP_circ()
{
    DELETE_VECTOR(_tabc,0);
}

Elise_colour DEP_circ::kth_col(INT i)
{
    ok_ind_col(i);

    REAL  ic = (i * _nb_c0) / (REAL) _nb;
    INT ic1 = 	(INT) ic;
    REAL pds1 = 1.0-(ic-ic1);
    return  pds1 * _tabc[ic1%_nb_c0] + (1-pds1) * _tabc[(ic1+1)%_nb_c0];
}

Elise_colour DEP_circ::rkth_col(INT,REAL p)
{
    REAL  ic = p * _nb_c0;
    INT ic1 = 	(INT) ic;
    REAL pds1 = 1.0-(ic-ic1);
    return  pds1 * _tabc[ic1%_nb_c0] + (1-pds1) * _tabc[(ic1+1)%_nb_c0];
}




void DEP_circ::lutage (  Data_Elise_Raster_D * derd,
                         U_INT1 * out,INT x0,INT x1,
                         U_INT2 * lut,
                         Const_INT_PP vals,INT x0_val) const
{
    out += x0 * derd->_byte_pp;
    const INT  * v = vals[0] + x0_val;
    INT nb = x1-x0;

    switch(derd->_cmod)
    {
         case  Indexed_Colour :
         {
               while (nb--)
                   *(out++) = (U_INT1) lut[mod256(*(v++))];
         }
         break;

         case True_16_Colour :
         {
               U_INT2 *  out2 = (U_INT2 *) out;
               while (nb--)
                   *(out2++) = lut[mod256(*(v++))];
         }
         break;

         case True_24_Colour :
         {
              INT vmod;
              INT bpp = derd->_byte_pp;
               while (nb--)
               {
                   vmod = mod256(*(v++));
                   out[derd->_r_ind]  = _true_c->_r[vmod];
                   out[derd->_g_ind]  = _true_c->_g[vmod];
                   out[derd->_b_ind]  = _true_c->_b[vmod];
                   out += bpp;
               }
         }
         break;
    }
}




           /*------------------*/
           /*     Circ_Pal     */
           /*------------------*/


Circ_Pal Circ_Pal::PCIRC6(INT NB)
{
    return  Circ_Pal
            (
                                NewLElCol(Elise_colour::red)
                                  + Elise_colour::yellow
                                  + Elise_colour::green
                                  + Elise_colour::cyan
                                  + Elise_colour::blue
                                  + Elise_colour::magenta,
                                  NB
            );
}

Circ_Pal::Circ_Pal (
                          L_El_Col l,
                          INT      NB,
                          bool reverse
                   )  :
    Elise_Palette (new DEP_circ(l,NB,reverse))
{
}

Col_Pal Circ_Pal::operator ()(INT c1)
{
    return Col_Pal(*this,c1);
}

Circ_Pal::Circ_Pal(Elise_Palette p) :
      Elise_Palette(p.dep())
{
}



/*****************************************************************/
/*                                                               */
/*                  DEP_Bi_Col                                   */
/*                                                               */
/*****************************************************************/


class Elise_PS_RGB_Palette : public Elise_PS_Palette
{
   public :
      Elise_PS_RGB_Palette
      (
             const char * name,
             Data_Elise_Palette * pal
      )  :
         Elise_PS_Palette(name,pal)
      {
      }

      void im_decode(ofstream & fp)
      {
           fp << "[0 1 0 1 0 1]";
      }

      void load_def(ofstream & fp)
      {
           fp << "/DeviceRGB setcolorspace ";
      };

      void set_cur_color(ofstream &fp,const INT * v)
      {
           Ps_Real_Col_Prec(fp,v[0],v[1],v[2],2);
           fp << " setrgbcolor\n";
      }

      virtual ColDevice cdev() const {return rgb;}

};

class Elise_PS_BiCol_Palette : public Elise_PS_RGB_Palette
{
      public :

          Elise_PS_BiCol_Palette
          (
                 const char * name,
                 Data_Elise_Palette * pal
          )  :
             Elise_PS_RGB_Palette(name,pal)
          {
          }

      private :

         virtual void set_cur_color(ofstream &fp,const INT * v);
         virtual void use_colors(U_INT1 ** out,INT ** in,INT dim,INT nb);
};

class DEP_Bi_Col    : public  Data_Elise_Palette
{
     friend class Elise_PS_BiCol_Palette;
     friend class BiCol_Pal;

     public :
     protected :

         DEP_Bi_Col
         (
                Elise_colour,
                Elise_colour,
                Elise_colour,
                INT NB1      ,
                INT NB2
         );

         virtual INT dim_pal() const {return 2;}
         virtual INT ps_dim_out() const {return 3;}
         virtual void verif_values_out(Const_INT_PP,INT nb) const;

         virtual void lutage (Data_Elise_Raster_D *,
                              U_INT1 * out,INT x0,INT x1,
                              U_INT2 * lut,
                              Const_INT_PP vals,INT x0_val) const;
         virtual ~DEP_Bi_Col();
     private :
         virtual Elise_colour kth_col(INT i);

          Elise_colour  _c0;
          Elise_colour  _c01;
          Elise_colour  _c02;

          INT           _nb1;
          INT           _nb2;

         void init_true();

         RGB_tab_compil * _true_c1;
         RGB_tab_compil * _true_c2;

         void ps_use_colors(U_INT1& r,U_INT1 &g,U_INT1 & b,INT k1,INT k2)
         {
             init_true();
             r  =  _true_c1->_r[k1]+_true_c2->_r[k2];
             g  =  _true_c1->_g[k1]+_true_c2->_g[k2];
             b  =  _true_c1->_b[k1]+_true_c2->_b[k2];
         }

         Elise_PS_Palette * ps_comp(const char * name)
         {
              return  new Elise_PS_BiCol_Palette (name,this);
         }

         void ps_end(Elise_PS_Palette * ,ofstream & ){}
         virtual void _inst_to_rgb(INT ** rgb,INT ** input,INT nb);

};

void Elise_PS_BiCol_Palette::set_cur_color(ofstream &fp,const INT * v)
{
     U_INT1 r,g,b;

     ((DEP_Bi_Col *)_pal)->ps_use_colors(r,g,b,v[0],v[1]);
     Ps_Real_Col_Prec(fp,r,g,b,2);
     fp << " setrgbcolor\n";
}
void Elise_PS_BiCol_Palette::use_colors(U_INT1 ** out,INT ** in,INT,INT nb)
{
     for (INT k=0 ; k<nb ; k++)
         ((DEP_Bi_Col *)_pal)->ps_use_colors
                               (
                                   out[0][k],
                                   out[1][k],
                                   out[2][k],
                                   in[0][k],
                                   in[1][k]
                               );
}

DEP_Bi_Col::~DEP_Bi_Col()
{
     if (_true_c1)
     {
         delete _true_c1;
         delete _true_c2;
     }
}

void DEP_Bi_Col::init_true()
{
     if (! _true_c1)
     {
          _true_c1 = RGB_tab_compil::NEW_one(0,256);
          _true_c2 = RGB_tab_compil::NEW_one(0,256);
          for (int i =0; i < 256 ; i++)
          {
               Elise_colour c1 = _c0 +  (i / 255.0) * _c01;
               Elise_colour c2 =  (i / 255.0) * _c02;
               _true_c1->set_kth_col(c1,i);
               _true_c2->set_kth_col(c2,i);
          }
     }
}


DEP_Bi_Col::DEP_Bi_Col
(
                Elise_colour c0,
                Elise_colour c1,
                Elise_colour c2,
                INT NB1      ,
                INT NB2
)  :
       Data_Elise_Palette(TYOFPAL::bicol,NB1*NB2,0,NB1*NB2),
       _c0 (c0),
       _c01 (c1-c0),
       _c02 (c2-c0),
       _nb1 (NB1),
       _nb2 (NB2),
       _true_c1(0),
       _true_c2(0)
{
    ASSERT_TJS_USER
    (
       (NB1 >= 1) && (NB1 < 256) && (NB2 >= 1) && (NB2 < 256),
       "nb colours must be in ranges [1 255] for bicolor palettes"
    );
}


void DEP_Bi_Col::verif_values_out(Const_INT_PP vals,INT nb) const
{
    ASSERT_USER
    (
             values_in_range(vals[0],nb,0,256)
         &&  values_in_range(vals[1],nb,0,256)  ,
         "values out of input range palette"
    );
}


Elise_colour DEP_Bi_Col::kth_col(INT i)
{
    ok_ind_col(i);
    int i1 = i % _nb1;
    int i2 = i / _nb1;

    return
            _c0
          + (i1 / (REAL) (_nb1 -1)) * _c01
          + (i2 / (REAL) (_nb2 -1))  *_c02;
}



void DEP_Bi_Col::lutage (Data_Elise_Raster_D * derd,
                         U_INT1 * out,INT x0,INT x1,
                         U_INT2 * lut,
                         Const_INT_PP vals,INT x0_val) const
{
    out += x0 * derd->_byte_pp;
    const INT  * v1 = vals[0] + x0_val;
    const INT  * v2 = vals[1] + x0_val;
    INT nb = x1-x0;

    switch(derd->_cmod)
    {
        case Indexed_Colour :
        {
             for (int i=0 ; i < nb ; i++)
                  out[i] = (U_INT1) lut[  (v1[i] * _nb1)/256 + ((v2[i]* _nb2) / 256) * _nb1];
        }
        break;

        case True_16_Colour :
        {
             U_INT2 * out2 = (U_INT2 *) out;
             INT vi1,vi2;

             for (int i=0 ; i < nb ; i++)
             {
                 vi1 = v1[i];
                 vi2 = v2[i];

                 out2[i] = derd->rgb_to_16
                           (
                              _true_c1->_r[vi1] +  _true_c2->_r[vi2],
                              _true_c1->_g[vi1] +  _true_c2->_g[vi2],
                              _true_c1->_b[vi1] +  _true_c2->_b[vi2]
                           );
             }
        }
        break;

        case True_24_Colour :
        {
             INT vi1,vi2;
             INT bpp = derd->_byte_pp;
             for (int i=0 ; i < nb ; i++)
             {
                 vi1 = v1[i];
                 vi2 = v2[i];

                 out[derd->_r_ind] =  _true_c1->_r[vi1]+_true_c2->_r[vi2];
                 out[derd->_g_ind] =  _true_c1->_g[vi1]+_true_c2->_g[vi2];
                 out[derd->_b_ind] =  _true_c1->_b[vi1]+_true_c2->_b[vi2];
                 out += bpp;
             }
        }
        break;
    }
}


void DEP_Bi_Col::_inst_to_rgb(INT ** rgb,INT ** input,INT nb)
{
     for (INT k =0; k<nb ; k++)
     {
        INT v1 = mod256(input[0][k]);
        INT v2 = mod256(input[1][k]);
        rgb[0][k] = AUC(_true_c1->_r[v1]+ _true_c2->_r[v2]);
        rgb[1][k] = AUC(_true_c1->_g[v1]+ _true_c2->_g[v2]);
        rgb[2][k] = AUC(_true_c1->_b[v1]+ _true_c2->_b[v2]);
     }
}

           /*-------------------*/
           /*     BiCol_Pal     */
           /*-------------------*/

BiCol_Pal::BiCol_Pal (
                            Elise_colour C0,
                            Elise_colour C1,
                            Elise_colour C2,
                            INT NB1,
                            INT NB2
                     )  :
      Elise_Palette(new DEP_Bi_Col(C0,C1,C2,NB1,NB2))
{
}


Col_Pal BiCol_Pal::operator ()(INT c1,INT c2)
{
    return Col_Pal(*this,c1,c2);
}


/*****************************************************************/
/*                                                               */
/*                  DEP_Tri_Col                                  */
/*                                                               */
/*****************************************************************/



class DEP_Tri_Col    : public  Data_Elise_Palette
{

     friend class TriCol_Pal;

     public :
     protected :

         virtual ~DEP_Tri_Col();
         DEP_Tri_Col
         (
               TYOFPAL::t,
                Elise_colour,
                Elise_colour,
                Elise_colour,
                Elise_colour,
                INT NB1      ,
                INT NB2      ,
                INT NB3
         );

         virtual INT dim_pal() const {return 3;}
         virtual void verif_values_out(Const_INT_PP,INT nb) const;

         virtual void lutage (Data_Elise_Raster_D *,
                              U_INT1 * out,INT x0,INT x1,
                              U_INT2 * lut,
                              Const_INT_PP vals,INT x0_val) const;
     private :
         virtual Elise_colour kth_col(INT i);

          Elise_colour  _c0;
          Elise_colour  _c01;
          Elise_colour  _c02;
          Elise_colour  _c03;

          INT           _nb1;
          INT           _nb2;
          INT           _nb3;


          INT           _nb12;

           void init_true();

           RGB_tab_compil * _true_c1;
           RGB_tab_compil * _true_c2;
           RGB_tab_compil * _true_c3;

           virtual void _inst_to_rgb(INT ** rgb,INT ** input,INT nb);
};


DEP_Tri_Col::DEP_Tri_Col
(
                TYOFPAL::t  top,
                Elise_colour c0,
                Elise_colour c1,
                Elise_colour c2,
                Elise_colour c3,
                INT NB1      ,
                INT NB2      ,
                INT NB3
)  :
       Data_Elise_Palette(top,NB1*NB2*NB3,0,NB1*NB2*NB3),
       _c0 (c0),
       _c01 (c1-c0),
       _c02 (c2-c0),
       _c03 (c3-c0),
       _nb1 (NB1),
       _nb2 (NB2),
       _nb3 (NB3),
       _nb12 (NB1 * NB2),
       _true_c1 (0),
       _true_c2 (0),
       _true_c3 (0)
{
    ASSERT_TJS_USER
    (
       (NB1 >= 1) && (NB1 < 256) && (NB2 >= 1) && (NB2 < 256) && (NB3 >= 1) && (NB3 < 256),
       "nb colours must be in ranges [1 255] for tricolor palettes"
    );
}

DEP_Tri_Col::~DEP_Tri_Col()
{
    if (_true_c1)
    {
         delete _true_c1;
         delete _true_c2;
         delete _true_c3;
    }
}

void DEP_Tri_Col::init_true()
{
     if (! _true_c1)
     {
          _true_c1 = RGB_tab_compil::NEW_one(0,256);
          _true_c2 = RGB_tab_compil::NEW_one(0,256);
          _true_c3 = RGB_tab_compil::NEW_one(0,256);
          for (int i =0; i < 256 ; i++)
          {
               Elise_colour c1 = _c0 +  (i / 255.0) * _c01;
               Elise_colour c2 =  (i / 255.0) * _c02;
               Elise_colour c3 =  (i / 255.0) * _c03;
               _true_c1->set_kth_col(c1,i);
               _true_c2->set_kth_col(c2,i);
               _true_c3->set_kth_col(c3,i);
          }
     }
}

void DEP_Tri_Col::verif_values_out(Const_INT_PP vals,INT nb) const
{
    ASSERT_USER
    (
             values_in_range(vals[0],nb,0,256)
         &&  values_in_range(vals[1],nb,0,256)
         &&  values_in_range(vals[2],nb,0,256)  ,
         "values out of input range palette"
    );
}


Elise_colour DEP_Tri_Col::kth_col(INT i)
{
    ok_ind_col(i);

    int i1 = i % _nb1;
    int i2 = (i / _nb1) % _nb2;
    int i3 = i / (_nb12);

    return
            _c0
          + (i1 / (REAL) (_nb1 -1)) * _c01
          + (i2 / (REAL) (_nb2 -1))  *_c02
          + (i3 / (REAL) (_nb3 -1))  *_c03;
}

void DEP_Tri_Col::lutage(  Data_Elise_Raster_D * derd,
                           U_INT1 * out,INT x0,INT x1,
                           U_INT2 * lut,
                           Const_INT_PP vals,INT x0_val) const
{
    out += x0 * derd->_byte_pp;
    const INT  * v1 = vals[0] + x0_val;
    const INT  * v2 = vals[1] + x0_val;
    const INT  * v3 = vals[2] + x0_val;
    INT nb = x1-x0;


    switch(derd->_cmod)
    {
        case Indexed_Colour :
        {
             for (int i=0 ; i < nb ; i++)
                 out[i] = (U_INT1)
				          lut
                          [
                                     (v1[i] * _nb1)/256
                                   + ((v2[i]* _nb2) / 256) * _nb1
                                   + ((v3[i]* _nb3) / 256) * _nb12
                          ];
        }
        break;

        case True_16_Colour :
        {
             U_INT2 * out2 = (U_INT2 *) out;
             INT vi1,vi2,vi3;

             for (int i=0 ; i < nb ; i++)
             {
                 vi1 = v1[i];
                 vi2 = v2[i];
                 vi3 = v3[i];

                 out2[i] = derd->rgb_to_16
                           (
                              _true_c1->_r[vi1] +  _true_c2->_r[vi2] +_true_c3->_r[vi3],
                              _true_c1->_g[vi1] +  _true_c2->_g[vi2] +_true_c3->_g[vi3],
                              _true_c1->_b[vi1] +  _true_c2->_b[vi2] +_true_c3->_b[vi3]
                           );
             }
        }
        break;

        case True_24_Colour :
        {
             INT bpp = derd->_byte_pp;
             INT vi1,vi2,vi3;
             for (int i=0 ; i < nb ; i++)
             {
                 vi1 = v1[i];
                 vi2 = v2[i];
                 vi3 = v3[i];

                 out[derd->_r_ind] =  _true_c1->_r[vi1] +
                                      _true_c2->_r[vi2] +
                                      _true_c3->_r[vi3];

                 out[derd->_g_ind] =  _true_c1->_g[vi1] +
                                      _true_c2->_g[vi2] +
                                      _true_c3->_g[vi3];

                 out[derd->_b_ind] =  _true_c1->_b[vi1] +
                                      _true_c2->_b[vi2] +
                                      _true_c3->_b[vi3];
                  out += bpp;
             }
        }
        break;
    }
}

void DEP_Tri_Col::_inst_to_rgb(INT ** rgb,INT ** input,INT nb)
{
     for (INT k =0; k<nb ; k++)
     {
        INT v1 = mod256(input[0][k]);
        INT v2 = mod256(input[1][k]);
        INT v3 = mod256(input[2][k]);
        rgb[0][k] = AUC(_true_c1->_r[v1]+ _true_c2->_r[v2] +  _true_c3->_r[v3]);
        rgb[1][k] = AUC(_true_c1->_g[v1]+ _true_c2->_g[v2] +  _true_c3->_g[v3]);
        rgb[2][k] = AUC(_true_c1->_b[v1]+ _true_c2->_b[v2] +  _true_c3->_b[v3]);
     }
}


           /*-------------------*/
           /*    TriCol_Pal     */
           /*-------------------*/

TriCol_Pal::TriCol_Pal (
                            Elise_colour C0,
                            Elise_colour C1,
                            Elise_colour C2,
                            Elise_colour C3,
                            INT NB1,
                            INT NB2,
                            INT NB3
                       )  :
      Elise_Palette(new DEP_Tri_Col(TYOFPAL::tricol,C0,C1,C2,C3,NB1,NB2,NB3))
{
}


Col_Pal TriCol_Pal::operator ()(INT c1,INT c2,INT c3)
{
    return Col_Pal(*this,c1,c2,c3);
}

/*****************************************************************/
/*                                                               */
/*                  DEP_RGB_Col                                  */
/*                                                               */
/*****************************************************************/




class DEP_RGB_Col    : public  DEP_Tri_Col
{

     friend class RGB_Pal;

     public :
     protected :

         DEP_RGB_Col
         (
                INT NB1      ,
                INT NB2      ,
                INT NB3
         );

         virtual void lutage (Data_Elise_Raster_D *,
                              U_INT1 * out,INT x0,INT x1,
                              U_INT2 * lut,
                              Const_INT_PP vals,INT x0_val) const;

         Elise_PS_Palette * ps_comp(const char * name)
         {
              return  new Elise_PS_RGB_Palette (name,this);
         }

         void ps_end(Elise_PS_Palette * ,ofstream & ){}


     private :

     private :
};

DEP_RGB_Col::DEP_RGB_Col(INT NB1,INT NB2,INT NB3) :
    DEP_Tri_Col
    (
          TYOFPAL::rgb,
          Elise_colour::black,
          Elise_colour::red,
          Elise_colour::green,
          Elise_colour::blue,
          NB1,
          NB2,
          NB3
    )
{
}

void DEP_RGB_Col::lutage(  Data_Elise_Raster_D * derd,
                           U_INT1 * out,INT x0,INT x1,
                           U_INT2 * lut,
                           Const_INT_PP vals,INT x0_val) const
{

    if (derd->_cmod == Indexed_Colour)
    {
       DEP_Tri_Col::lutage(derd,out,x0,x1,lut,vals,x0_val);
       return;
    }
    out += x0 * derd->_byte_pp;
    const INT  * v1 = vals[0] + x0_val;
    const INT  * v2 = vals[1] + x0_val;
    const INT  * v3 = vals[2] + x0_val;
    INT nb = x1-x0;


    switch(derd->_cmod)
    {
        case Indexed_Colour :
        break;

        case True_16_Colour :
        {
             U_INT2 * out2 = (U_INT2 *) out;

             for (int i=0 ; i < nb ; i++)
                 out2[i] = derd->rgb_to_16 ( v1[i], v2[i], v3[i]);
        }
        break;

        case True_24_Colour :
        {
             INT bpp = derd->_byte_pp;
             INT r_ind = derd->_r_ind;
             INT g_ind = derd->_g_ind;
             INT b_ind = derd->_b_ind;
             for (int i=0 ; i<nb ; i++)
             {
                 out[r_ind] =  v1[i];
                 out[g_ind] =  v2[i];
                 out[b_ind] =  v3[i];
                 out += bpp;
             }
        }
        break;
    }
}

           /*-------------------*/
           /*    RGB_Pal        */
           /*-------------------*/

RGB_Pal::RGB_Pal (INT NB1, INT NB2, INT NB3)  :
      Elise_Palette(new DEP_RGB_Col(NB1,NB2,NB3))
{
}


Col_Pal RGB_Pal::operator ()(INT c1,INT c2,INT c3)
{
    return Col_Pal(*this,c1,c2,c3);
}

RGB_Pal::RGB_Pal(Elise_Palette p) :
      Elise_Palette(p.dep())
{
}

/*****************************************************************/
/*                                                               */
/*                  Elise_Palette                                */
/*                                                               */
/*****************************************************************/


Elise_Palette::Elise_Palette(Data_Elise_Palette * p) :
    PRC0(p)
{
}


TYOFPAL::t Elise_Palette::type_pal() const
{
    return dep()->typal();
}




INT Elise_Palette::nb()
{
    return dep()->nb();
}

Elise_PS_Palette * Elise_Palette::ps_comp(const char * name)
{
      return dep()->ps_comp(name);
}

INT Elise_Palette::dim_pal()
{
   return dep()->dim_pal();
}

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
/*******         SET OF PALETTES                                           *****/
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/


/*****************************************************************/
/*                                                               */
/*                  Data_Elise_Set_Of_Palette                    */
/*                                                               */
/*****************************************************************/


Data_Elise_Set_Of_Palette::Data_Elise_Set_Of_Palette
(
     L_El_Palette lp
) :
  _lp (lp)
{
}


/*****************************************************************/
/*                                                               */
/*                  Elise_Set_Of_Palette                         */
/*                                                               */
/*****************************************************************/

Elise_Set_Of_Palette::Elise_Set_Of_Palette
(
      L_El_Palette  lp
) :
    PRC0(new Data_Elise_Set_Of_Palette(lp))
{
}

L_El_Palette Elise_Set_Of_Palette::lp() const
{
    return SAFE_DYNC(Data_Elise_Set_Of_Palette *,_ptr)->_lp;
}


bool Elise_Set_Of_Palette::operator == (Elise_Set_Of_Palette p2) const
{
     return _ptr == p2._ptr;
}


INT Elise_Set_Of_Palette::som_nb_col() const
{
    int som = 0;
    for ( L_El_Palette l = lp() ; !(l.empty()) ; l = l.cdr())
        som += l.car().nb();

   return som;
}

void  Elise_Set_Of_Palette::pal_is_loaded(Elise_Palette p) const
{
    for ( L_El_Palette l = lp() ; !(l.empty()) ; l = l.cdr())
        if (p._ptr == l.car()._ptr)
           return ;

    elise_fatal_error("unactived palette",__FILE__,__LINE__);
}

Elise_Palette  Elise_Set_Of_Palette::pal_of_type(TYOFPAL::t  tpal) const
{
    for ( L_El_Palette l = lp() ; !(l.empty()) ; l = l.cdr())
        if (tpal == l.car().type_pal())
           return  l.car();

    elise_fatal_error("unactived palette",__FILE__,__LINE__);
    return Gray_Pal(0);
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
