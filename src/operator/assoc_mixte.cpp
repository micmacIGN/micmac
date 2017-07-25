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


void OperBinMixte::t0_eg_t1_op_t2(REAL16 * t0,const REAL16 * t1,const REAL16 *t2,INT nb) const
{
   ELISE_ASSERT(false,"OperBinMixte::t0_eg_t1_op_t2");
}
void OperBinMixte::t0_eg_t1_op_t2(_INT8 * t0,const _INT8 * t1,const _INT8 *t2,INT nb) const
{
   ELISE_ASSERT(false,"OperBinMixte::t0_eg_t1_op_t2");
}




/****************************************************************/
/*                                                              */
/*     Elementary definition                                    */
/*                                                              */
/****************************************************************/

      // plus_elem

class plus_elem
{
  public :
    static inline  int    op(int a,int b)       {return a + b;}
    static inline  double op(double a,double b) {return a + b;}

    static inline  int    inv_bin(int a,int b)       {return a - b;}
    static inline  double inv_bin(double a,double b) {return a - b;}

    static inline  int    inv_eq(int &a,int b)       {return a -=  b;}
    static inline  double inv_eq(double &a,double b) {return a -=  b;}

    static inline  void    op_eq(int &a,int b)        { a += b;}
    static inline  void    op_eq(double &a,double b)  { a += b;}

    static const double       r_neutre;
    static const int          i_neutre;
    static const std::string  name;

    static const OperAssocMixte & optab;

    static inline Fonc_Num opf(Fonc_Num f1,Fonc_Num f2) { return f1+f2;}

};

const double plus_elem::r_neutre = 0.0;
const int    plus_elem::i_neutre = 0;
const std::string plus_elem::name = "+";

      // mul_elem

class mul_elem
{
  public :
    static inline  int    op(int a,int b)       {return a * b;}
    static inline  double op(double a,double b) {return a * b;}

    static inline  void    op_eq(int &a,int b)        { a *= b;}
    static inline  void    op_eq(double &a,double b)  { a *= b;}

    static const double r_neutre ;
    static const int    i_neutre ;
    static const std::string  name;

    static inline Fonc_Num opf(Fonc_Num f1,Fonc_Num f2) { return f1*f2;}

    static const OperAssocMixte & optab;
};

const double mul_elem::r_neutre = 1.0;
const int    mul_elem::i_neutre = 1;
const std::string mul_elem::name = "*";



      // max_elem

class max_elem
{
  public :
    static inline  int    op(int a,int b)       {return (a>b) ? a : b;}
    static inline  double op(double a,double b) {return (a>b) ? a : b;}

    static inline  void    op_eq(int &a,int b)        { if(a<b) a = b;}
    static inline  void    op_eq(double &a,double b)  { if(a<b) a = b;}

    static const REAL    r_neutre;
    static const INT     i_neutre;
    static const std::string  name;

    static inline Fonc_Num opf(Fonc_Num f1,Fonc_Num f2) { return Max(f1,f2);}
    static const OperAssocMixte & optab;
};

const double max_elem::r_neutre = -DBL_MAX;
const int    max_elem::i_neutre = INT_MIN;
const std::string max_elem::name = "max";


      // min_elem

class min_elem
{
  public :
    static inline  int    op(int a,int b)       {return (a<b) ? a : b;}
    static inline  double op(double a,double b) {return (a<b) ? a : b;}

    static inline  void    op_eq(int &a,int b)        { if(a>b) a = b;}
    static inline  void    op_eq(double &a,double b)  { if(a>b) a = b;}

    static const REAL    r_neutre;
    static const INT     i_neutre;
    static const std::string  name;

    static inline Fonc_Num opf(Fonc_Num f1,Fonc_Num f2) { return Min(f1,f2);}
    static const OperAssocMixte & optab;
};

const double min_elem::r_neutre = DBL_MAX;
const int    min_elem::i_neutre = INT_MAX;
const std::string min_elem::name = "min";
/****************************************************************/
/*                                                              */
/*     Reduction on a segment                                   */
/*                                                              */
/****************************************************************/
/*
    Effectue une dilatation (mono-dimensionnelle) en niveau de gris.

    Il est necessaire que :

        * tous les tableaux (in,out,buf_av, buf_ar) soient
          des espaces memoires distincts; la seule exception
          concerne in et out qui peuvent etre eventuellement egaux;

        * tous les tableaux soient indexables sur
                 x_min+dx0 <= x <x_max+dx1

        * x_min < x_max, dx0 <= dx1 (en fait, je ne sais pas
          si c'est vraiment necessaire, mais je sais que
          je n'ai pas envie d'y reflechir);


    Fonctionnellement, on a, a la fin :


            *  pour  x_min <= X < x_max

                     out[X] =  Max in(X+dx) (avec dx0 <= dx <= dx1);

    L'algorithme utilise est celui de Truc-Muche (voir reference
    Pierre Soille) qui en utilisant une "bufferisation avant-arriere"
    a un temps de calcul independant de |dx1-dx0|.


    Globalement, je ne suis pas sur que ce soit vraiment tres utile,
    mais qu'est-ce que c'est joli comme algo !


    N.B: Ca pourrait etre programme de maniere + generique car le principe
    de l'algo est adaptable a tout operateur associatif. Cela dit:
         - Min est facilement emulable par - (Max(-f));
         - || et && ne sont que des cas particulier de Max et Min;
         - + a son propre algo, sans doute plus rapide;
         - * a peu d'interet (et a la rigueur emulable par
             exp(Sigma(log(f))))
         - reste la cas de &,| et ^; a voir ? Mais faire des "reduction
           associative" ultra-rapide pour les operateurs bits a bits,
           c'est sans doute pas une priorite.
*/


//  Should be function template; but my version of compiler does not
//  support explicit instanciation of function template; so : a class

template <class elem,class Type>  class tpl_red_seg
{
   public :
     static void f
     (
            Type * out,
            const Type * in,
            Type * buf_av,
            Type * buf_ar,
            INT    x_min,
            INT    x_max,
            INT    dx0,
            INT    dx1,
            Type   neutre
     );

     static void trivial_f
     (
           Type * out,
           const Type * in,
           Type * buf_av,
           INT    x_min,
           INT    x_max,
           INT    dx0,
           INT    dx1
     );
};



    // when dx1-dx0 is small : better use trivial algorithm
    // be carrefull to the fact that, often, in = out (because it is
    // ok with general algo)

template <class elem,class Type> void tpl_red_seg<elem,Type>::trivial_f
(
      Type * out,
      const Type * in,
      Type * buf_av,
      INT    x_min,
      INT    x_max,
      INT    dx0,
      INT    dx1
)
{

     Type * bopt = buf_av+x_min+dx0;
     INT nb = x_max-x_min;
     const Type * i0 = in+x_min+dx0;
     Type * o0 = out+x_min;

     switch (dx1-dx0)
     {

           case 0 :
           {
                 convert(bopt , i0   , nb);
                 convert(o0   , bopt , nb);
                 return;
           }

           case 1 :
           {
                 elem::optab.t0_eg_t1_op_t2(bopt,i0,i0+1,nb);
                 convert(o0,bopt,nb);
                 return;
           }

           case 2 :
           {
                 for (INT k = 0; k < nb ; k++,i0 ++)
                     bopt[k] = elem::op(i0[0], elem::op(i0[1],i0[2]));
                 convert(o0,bopt,nb);
                 return;
           }

           case 3 :
           {
                 for (INT k = 0; k < nb ; k++,i0 ++)
                     bopt[k] = elem::op(elem::op(i0[0],i0[1]), elem::op(i0[2],i0[3]));
                 convert(o0,bopt,nb);
                 return;
           }

           default :
                  elise_internal_error
                  (
                     "tpl_red_seg<elem,Type>::trivial_f",
                      __FILE__,__LINE__
                  );

     }
}




template <class elem,class Type> void tpl_red_seg<elem,Type>::f
(
      Type * out,
      const Type * in,
      Type * buf_av,
      Type * buf_ar,
      INT    x_min,
      INT    x_max,
      INT    dx0,
      INT    dx1,
      Type   neutre
)
{
     set_min_max(dx0,dx1);


     if (dx1-dx0 <=3)
     {
          trivial_f(out,in,buf_av,x_min,x_max,dx0,dx1);
          return;
     }


    INT  per;
    INT  X_max,X_min;

    per = dx1-dx0 + 1;
    X_min = x_min + dx0;
    X_max = x_max + dx1;

    // une passe en avant;

    INT4 x; // FUUUCCKKK to visual

    buf_av[X_min] = in[X_min];
    for (x =X_min+1; x<X_max ; x++)
        if (x%per)
           buf_av[x] = elem::op(buf_av[x-1],in[x]);
        else
           buf_av[x] = in[x];


    // une passe en arriere;

    buf_ar[X_max-1] =  in[X_max-1];
    for (x = X_max-2; x >= X_min ; x--)
        if (x%per)
           buf_ar[x] = elem::op(buf_ar[x+1],in[x]);
        else
           buf_ar[x] = neutre;

    // Conclusion :

    for (x=x_min ; x<x_max ; x++)
        out[x] = elem::op(buf_ar[x+dx0],buf_av[x+dx1]);
}


template <class elem,class Type>  class grp_tpl_red_seg
{
   public :

     static void f
     (
            Type * out,
            const Type * in,
            Type * buf_av,
            INT    x_min,
            INT    x_max,
            INT    dx0,
            INT    dx1,
            Type   neutre
     );
};

template <class elem,class Type> void grp_tpl_red_seg<elem,Type>::f
(
         Type * out,
         const Type * in,
         Type * buf_av,
         INT    x_min,
         INT    x_max,
         INT    dx0,
         INT    dx1,
         Type   neutre
)
{
     set_min_max(dx0,dx1);

     if (dx1-dx0 <=1)
     {
          tpl_red_seg<elem,Type>::trivial_f(out,in,buf_av,x_min,x_max,dx0,dx1);
          return;
     }

     Type * tmp  = buf_av + x_min + dx0;
     const Type * ptr_ar = in + x_min + dx0;
     const Type * ptr_av = in + x_min + dx0;

     Type res = neutre;

     for (INT dx = dx0; dx <dx1 ; dx++)
         elem::op_eq(res,*(ptr_av++));

     for (INT x = x_min; x <x_max ; x++)
     {
         elem::op_eq(res,*(ptr_av++));
         *(tmp++) = res;
         elem::inv_eq(res,*(ptr_ar++));
     }

     convert(out+x_min,buf_av + x_min + dx0,x_max-x_min);

}



/****************************************************************/
/*                                                              */
/*     OpMIxteTpl                                               */
/*                                                              */
/****************************************************************/

REAL16 OperAssocMixte::opel(REAL16,REAL16) const
{
    ELISE_ASSERT(false,"OperAssocMixte::opel");
    return 0;
}
_INT8 OperAssocMixte::opel(_INT8,_INT8) const
{
    ELISE_ASSERT(false,"OperAssocMixte::opel");
    return 0;
}

OperAssocMixte::OperAssocMixte(Id Theid) : _id(Theid) {}


static std::map<std::string,OperAssocMixte *>   TheMapAssoc;

OperAssocMixte * OperAssocMixte::GetFromName(const std::string & aName,bool Svp)
{
    OperAssocMixte * aRes = TheMapAssoc[aName];

    if ((aRes==0) && (! Svp))
    {
        std::cout << "Cannot get OperAssocMixte with name = " << aName << "\n";
        ELISE_ASSERT(false,"OperAssocMixte::GetFromName");
    } 
    return aRes;
}

// OperAssocMixte & Oper

template <class elem> class  OpMIxteTpl : public OperAssocMixte
{
     public :


     static  const Id _cl_id;

     OpMIxteTpl() :
             OperAssocMixte(_cl_id)
     {
          TheMapAssoc[elem::name] = this;
     }

     REAL opel(REAL v1,REAL v2) const {return elem::op(v1,v2);}
     INT  opel(INT  v1,INT  v2) const {return elem::op(v1,v2);}


    //***********************
    // Operator on functions
    //***********************

     Fonc_Num opf(Fonc_Num f1,Fonc_Num f2) const {return elem::opf(f1,f2);}

    //***********************
    // reduction on a segment
    //***********************

     void reduce_seg
        ( INT * out, const INT * in, INT * buf_av, INT * buf_ar,
          INT   x_min, INT   x_max, INT  dx0, INT   dx1)  const
      {
          tpl_red_seg<elem,INT>::f(out,in,buf_av,buf_ar,x_min,x_max,dx0,dx1,elem::i_neutre);
      }

     void reduce_seg
        ( REAL * out, const REAL * in, REAL * buf_av, REAL * buf_ar,
          INT   x_min, INT   x_max, INT  dx0, INT   dx1)  const
      {
          tpl_red_seg<elem,REAL>::f(out,in,buf_av,buf_ar,x_min,x_max,dx0,dx1,elem::r_neutre);
      }


    //******************
    // neutral element
    //******************

      double rneutre() const { return elem::r_neutre;}
      int    ineutre() const { return elem::i_neutre;}

    //******************
    // Reduction
    //******************

      REAL red_tab(const REAL * tab,INT nb,REAL    v_init) const
      {
         for(int i = 0; i<nb; i++)
            elem::op_eq(v_init,tab[i]);

         return v_init;
      }

      INT red_tab(const INT * tab,INT nb,INT    v_init) const
      {
          for(int i = 0; i<nb; i++)
             elem::op_eq(v_init,tab[i]);

          return v_init;
      }

    //******************
    //  t0 = t1 op t2
    //******************

      void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const REAL *t2,INT nb) const
      {
           for(int i=0; i<nb ; i++)
              t0[i] = elem::op(t1[i],t2[i]);
      }

      void t0_eg_t1_op_t2(REAL * t0,const INT  * t1,const REAL *t2,INT nb) const
      {
           for(int i=0; i<nb ; i++)
               t0[i] = elem::op((REAL)t1[i],t2[i]);
      }

      void t0_eg_t1_op_t2(REAL * t0,const REAL  * t1,const INT *t2,INT nb) const
      {
           for(int i=0; i<nb ; i++)
              t0[i] = elem::op(t1[i],(REAL)t2[i]);
      }

      void t0_eg_t1_op_t2(INT * t0,const INT  * t1,const INT *t2,INT nb) const
      {
           for(int i=0; i<nb ; i++)
              t0[i] = elem::op(t1[i],t2[i]);
      }

    //**************************
    //    t0_opeg_t1
    //**************************

      void t0_opeg_t1(INT * t0,const INT  * t1,INT nb) const
      {
           for(int i=0; i<nb ; i++)
              elem::op_eq(t0[i],t1[i]);
      }

      void t0_opeg_t1(REAL * t0,const REAL  * t1,INT nb) const
      {
           for(int i=0; i<nb ; i++)
              elem::op_eq(t0[i],t1[i]);
      }


    //**************************
    //   integral
    //**************************

      void integral(INT *out,const INT * in,INT nb)  const
      {
           if (! nb) return;
           out[0] = in[0];

           for(int i=1; i<nb ; i++)
              out[i] = elem::op(out[i-1],in[i]);
      }

      void integral(REAL *out,const REAL * in,INT nb)  const
      {
           if (! nb) return;
           out[0] = in[0];

           for(int i=1; i<nb ; i++)
              out[i] = elem::op(out[i-1],in[i]);
      }

      virtual const char * name () const;
};


/****************************************************************/
/*                                                              */
/*     GrpOpMIxteTpl                                            */
/*                                                              */
/****************************************************************/


template <class elem> class  GrpOpMIxteTpl : public OpMIxteTpl<elem>
{
     public :


    //***********************
    // reduction on a segment
    //***********************

     void reduce_seg
        ( INT * out, const INT * in, INT * buf_av, INT * ,
          INT   x_min, INT   x_max, INT  dx0, INT   dx1)  const
      {
          grp_tpl_red_seg<elem,INT>::f
             (out,in,buf_av,x_min,x_max,dx0,dx1,elem::i_neutre);
      }

     void reduce_seg
        ( REAL * out, const REAL * in, REAL * buf_av, REAL *,
          INT   x_min, INT   x_max, INT  dx0, INT   dx1)  const
      {
          grp_tpl_red_seg<elem,REAL>::f
             (out,in,buf_av,x_min,x_max,dx0,dx1,elem::r_neutre);
      }

      virtual void t0_eg_t1_opinv_t2
              (REAL *t0,const REAL * t1,const REAL * t2,INT nb) const
      {
              for (INT k=0 ; k<nb ; k++)
                  t0[k] = elem::inv_bin(t1[k],t2[k]);
      }

      virtual void t0_eg_t1_opinv_t2
              (INT *t0,const INT * t1,const INT * t2,INT nb) const
      {
              for (INT k=0 ; k<nb ; k++)
                  t0[k] = elem::inv_bin(t1[k],t2[k]);
      }

      virtual bool grp_oper() const
      {
          return true;
      }

      void t0_opinveg_t1(INT * t0,const INT  * t1,INT nb) const
      {
           for(int i=0; i<nb ; i++)
              elem::inv_eq(t0[i],t1[i]);
      }

      void t0_opinveg_t1(REAL * t0,const REAL  * t1,INT nb) const
      {
           for(int i=0; i<nb ; i++)
              elem::inv_eq(t0[i],t1[i]);
      }


};




GrpOpMIxteTpl<plus_elem>  OpMIxteTpl_plus_The_only_one;
const OperAssocMixte & OpSum =  OpMIxteTpl_plus_The_only_one;
template <> const OperAssocMixte::Id OpMIxteTpl<plus_elem>::_cl_id
    = OperAssocMixte::Sum;
ElTmplSpecNull const char * OpMIxteTpl<plus_elem>::name() const
{
    return "OpSum";
}
// Initialisation ici, car initialisation par OpMin est
// source d'erreur (= depend de l'ordre choisie). Ici
// en le faisant sur l'adresse d'un objet ca marche
// car c'est en fait une constante (resolue a l'edition
// de lien)

const OperAssocMixte & plus_elem::optab
    = OpMIxteTpl_plus_The_only_one;

    //================================================

OpMIxteTpl<mul_elem>  OpMIxteTpl_mul_The_only_one;
const OperAssocMixte & OpMul =  OpMIxteTpl_mul_The_only_one;
template <> const OperAssocMixte::Id OpMIxteTpl<mul_elem>::_cl_id
    = OperAssocMixte::Mul;

ElTmplSpecNull const char * OpMIxteTpl<mul_elem>::name() const
{
    return "OpMul";
}
const OperAssocMixte & mul_elem::optab
    = OpMIxteTpl_mul_The_only_one;

    //====================================================

OpMIxteTpl<max_elem>  OpMIxteTpl_max_The_only_one;
const OperAssocMixte & OpMax =  OpMIxteTpl_max_The_only_one;
template <> const OperAssocMixte::Id OpMIxteTpl<max_elem>::_cl_id
    = OperAssocMixte::Max;

ElTmplSpecNull const char * OpMIxteTpl<max_elem>::name() const
{
    return "OpMax";
}
const OperAssocMixte & max_elem::optab
    = OpMIxteTpl_max_The_only_one;

    //====================================================

OpMIxteTpl<min_elem>  OpMIxteTpl_min_The_only_one;
const OperAssocMixte & OpMin =  OpMIxteTpl_min_The_only_one;
template <> const OperAssocMixte::Id OpMIxteTpl<min_elem>::_cl_id
    = OperAssocMixte::Min;

ElTmplSpecNull const char * OpMIxteTpl<min_elem>::name() const
{
    return "OpMin";
}


const OperAssocMixte & min_elem::optab
    = OpMIxteTpl_min_The_only_one;



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
