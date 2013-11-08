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

#define ELISE_NAME_DATA_DIR "../data/"
#define  DIRECT_NAME  "/"

/***********************************************************************/
/***********************************************************************/

            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Kieme_Opb_Comp                 */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */

class Post_trait_Opb_flag
{
     public :
     
          virtual void post_proc(INT *,INT nb) = 0;
  virtual ~Post_trait_Opb_flag() {}
};

class OPB_Flaggage : public Fonc_Num_OPB_TPL<INT>
{
         public :

               OPB_Flaggage
               (
                    const Arg_Fonc_Num_Comp & arg,
                    INT                     dim_out,
                    Fonc_Num                f0,
                    Box2di                  box,
                    Post_trait_Opb_flag &   post
               );
               virtual ~OPB_Flaggage();

         private :

            virtual void post_new_line(bool);
            void flaggage_line(INT d,INT y);

            INT ** _buf_flag;
            Post_trait_Opb_flag &      _post;
};

OPB_Flaggage::~OPB_Flaggage()
{
     DELETE_MATRICE(_buf_flag,Pt2di(_x0_buf,0),Pt2di(_x1_buf,_dim_out));
}

OPB_Flaggage::OPB_Flaggage
(
       const Arg_Fonc_Num_Comp &    arg,
       INT                          dim_out,
       Fonc_Num                     f0,
       Box2di                       box,
       Post_trait_Opb_flag &        post
)  :

       Fonc_Num_OPB_TPL<INT>(arg,dim_out,Arg_FNOPB(f0,box)),
       _buf_flag(NEW_MATRICE(Pt2di(_x0_buf,0),Pt2di(_x1_buf,dim_out),INT)),
       _post(post)
{
}

void OPB_Flaggage::flaggage_line(INT d,INT y)
{

     INT * l  = kth_buf((INT *)0,0)[d][y];
     INT * bf = _buf_flag[d];
     INT  flag  = 1 << ( _y1_side-_y0_side);
     for (INT x = _x0_buf; x<_x1_buf ; x++)
     {
         if (l[x])
            bf[x] = ( bf[x]>>1) | flag;
         else
            bf[x] = ( bf[x]>>1); 
     }

}

void OPB_Flaggage::post_new_line(bool first)
{
     if (first)
     {
         for (INT d =0 ; d<_dim_out ; d++)
         {
              for (INT x=_x0_buf;  x<_x1_buf ; x++)
                   _buf_flag[d][x]=0;
              for (INT y=_y0_buf ; y<_y1_buf-1 ; y++)
                   flaggage_line(d,y);
         }
     }

     
     for (INT d =0 ; d<_dim_out ; d++)
     {
          flaggage_line(d,_y1_buf-1);
          INT * bf = _buf_flag[d];
          INT * res  = _buf_res[d];
          INT nby = (_y1_side-_y0_side+1);
          INT nbxy = nby * (_x1_side-_x0_side);
          INT v = 0;
          for (INT x = _x0+_x0_side ; x<= _x0+_x1_side ; x++)
              v = v | (bf[x] << (nby*(x-(_x0+_x0_side))));
	  {
		for (INT x = _x0  ; x< _x1  ; x++)
		{
			  res[x] = v | ( bf[x+_x1_side] << nbxy);
			v = res[x] >> nby;
		}
	  }
          _post.post_proc(res+_x0,_x1-_x0);
      }
}



            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           OPB_Flag_Not_Comp              */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */


class OPB_Flag_Not_Comp : public Fonc_Num_Not_Comp
{
      public :
          OPB_Flag_Not_Comp
          (
               Fonc_Num                    f0,
               Box2di                      side_0,
               Post_trait_Opb_flag &       post
          );

      private :

          virtual bool  integral_fonc (bool) const
          {
               return true;
          }

          virtual INT dimf_out() const {return _f.dimf_out();}
          void VarDerNN (ElGrowingSetInd &) const{ELISE_ASSERT(false,"No VarDerNN");}



          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg);

          Fonc_Num                  _f;
          Box2di                    _side;
          Post_trait_Opb_flag &     _post;
};

OPB_Flag_Not_Comp::OPB_Flag_Not_Comp
(
       Fonc_Num                    f0,
       Box2di                      side_0,
       Post_trait_Opb_flag &       post
)  :
   _f         (Iconv(f0)),
   _side      (side_0),
   _post      (post)
{
}

Fonc_Num_Computed * OPB_Flag_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{

    return new OPB_Flaggage (arg,dimf_out(),_f,_side,_post);
}



/*************************************************************/

class No_Post_trait_Opb_flag : public Post_trait_Opb_flag
{
     public :
         void post_proc(INT *,INT){};
  virtual ~No_Post_trait_Opb_flag() {}
          static No_Post_trait_Opb_flag _the_only_one;
};
No_Post_trait_Opb_flag No_Post_trait_Opb_flag::_the_only_one;


Fonc_Num  flag_vois (Fonc_Num  f,Box2di side)
{
    return new OPB_Flag_Not_Comp
               (f,side,No_Post_trait_Opb_flag::_the_only_one);
}


/*************************************************************/


class Tab_UINT1_Opb3_Flag : public Post_trait_Opb_flag
{
     public :
         // Probleme avec le 4-voisinage a revoir
         // static Tab_Bits_PT_Opb_flag eros_4_hom;
         void post_proc(INT * v,INT nb)
         {
              for (INT k=0 ; k<nb ; k++)
                  v[k] = _lut[v[k]];
         }
  virtual ~Tab_UINT1_Opb3_Flag() {}
         static Tab_UINT1_Opb3_Flag  FREEM8_TRIGO;
         static Tab_UINT1_Opb3_Flag  FREEM4_TRIGO;

     private :

         Tab_UINT1_Opb3_Flag (U_INT1 * lut) : _lut (lut) {}
         U_INT1 * _lut;

};


Tab_UINT1_Opb3_Flag  Tab_UINT1_Opb3_Flag::FREEM8_TRIGO
                     (FLAG_FRONT_8_TRIGO);

Tab_UINT1_Opb3_Flag  Tab_UINT1_Opb3_Flag::FREEM4_TRIGO
                     (FLAG_FRONT_4_TRIGO);


Fonc_Num  flag_front4 (Fonc_Num  f)
{
    return new OPB_Flag_Not_Comp
               (  f,
                  Box2di(Pt2di(-1,-1),Pt2di(1,1)),
                  Tab_UINT1_Opb3_Flag::FREEM4_TRIGO
               );
}


Fonc_Num  flag_front8 (Fonc_Num  f)
{
    return new OPB_Flag_Not_Comp
               (  f,
                  Box2di(Pt2di(-1,-1),Pt2di(1,1)),
                  Tab_UINT1_Opb3_Flag::FREEM8_TRIGO
               );
}

/*************************************************************/


class Tab_Bits_PT_Opb_flag : public Post_trait_Opb_flag
{
     public :
         // Probleme avec le 4-voisinage a revoir
         // static Tab_Bits_PT_Opb_flag eros_4_hom;
         static Tab_Bits_PT_Opb_flag eros_8_hom;
  virtual ~Tab_Bits_PT_Opb_flag() {}

     private :
         Tab_Bits_PT_Opb_flag (const char * name);
         const U_INT1 * _lut;
         File_Tabulated<U_INT1> _tab;
         void post_proc(INT * v,INT nb);

};

Tab_Bits_PT_Opb_flag 
 Tab_Bits_PT_Opb_flag::eros_8_hom
 (
	ELISE_NAME_DATA_DIR 
	"Tabul"  DIRECT_NAME "erod_8"
 );

Tab_Bits_PT_Opb_flag::Tab_Bits_PT_Opb_flag(const char * name):
       _tab(name)
{
}

void Tab_Bits_PT_Opb_flag::post_proc(INT * v,INT nb)
{
     _lut = _tab.ptr();
     for (INT x =0; x<nb; x++)
         v[x] = kth_bit(_lut,v[x]);
}



Fonc_Num  erod_8_hom (Fonc_Num  f)
{
    return new OPB_Flag_Not_Comp
               (  f,
                  Box2di(Pt2di(-1,-1),Pt2di(2,2)),
                  Tab_Bits_PT_Opb_flag::eros_8_hom
               );
}

/*********************************************************************/
/*                                                                   */
/*   Manipulation  on 8-neigboors-flagged graph                      */
/*                                                                   */
/*********************************************************************/

class Opb_Sym_Flag : public Simple_OPBuf1<INT,INT>
{
     public :

       typedef enum
       {
            sym,
            ferm_sym,
            ouv_sym
       } ModeSym;


       Opb_Sym_Flag(ModeSym ms,INT dim_in_vraie,INT dim_to_sym) : 
         _ms            (ms)          ,
         _dim_in_vraie  (dim_in_vraie),
         _dim_to_sym    (dim_to_sym)
       {};


     private :

       void sym_line(INT l);

       virtual void  calc_buf(INT ** output,INT *** input);


       ModeSym               _ms;
       INT         _dim_in_vraie;
       INT           _dim_to_sym;
};


void Opb_Sym_Flag::sym_line(INT l)
{
	INT x;

    // to force initialization of tabulation
    SkVein();

    for (INT d = 0; d < _dim_to_sym ; d++)
    {
         INT * i = _in[d][l];
         INT ** o = _in[d+_dim_in_vraie]+l;
         INT * o1 = o[1];
         for ( x = x0() ; x < x1() ; x++)
             o1[x] = 0;
         for ( x = x0()-1 ; x <= x1() ; x++)
         {
             for 
             (
                 SkVein::ldir * l = SkVein::BitsOfFlag[i[x]&255]; 
                 l; 
                 l=l->_next
             )
                 o[l->_y][x+l->_x] |=  (l->_flag_dsym);
        }
    }
}



void  Opb_Sym_Flag::calc_buf(INT ** out,INT ***) 
{
	INT x, d;

   for (INT l= (first_line() ? -1 : 1); l <= 1; l ++)
       sym_line(l);

    for ( d = 0; d < _dim_to_sym ; d++)
    {
         INT * i = _in[d][0];
         INT * s = _in[d+_dim_in_vraie][0];
         INT * o = out[d];

         switch (_ms)
         {
             case sym :
                  convert(o+x0(),s+x0(),tx());
             break;

             case ferm_sym :
                  for ( x = x0() ; x < x1() ; x++)
                      o[x] = s[x] | i[x];
             break;

             case ouv_sym :
                  for ( x = x0() ; x < x1() ; x++)
                      o[x] = s[x] & i[x];
             break;
         }
    }

    for ( d = _dim_to_sym ; d < _dim_in_vraie ; d++)
         convert
         (
                out[d]+x0(),
               _in[d][0]+x0(),
               tx()
         );
}

Fonc_Num op_sym_gen
         (
              Fonc_Num                f,
              Opb_Sym_Flag::ModeSym  ms,
              bool                   ident = false
         )
{
    INT dim = f.dimf_out ();
    
    for (INT d =0; d<dim ; d++)
        f = Virgule(f,Fonc_Num(0));

    return create_op_buf_simple_tpl
           (
                new Opb_Sym_Flag (ms,dim,(ident?1:dim)),
                0,
                f,
                dim, 
                Box2di(2)
           );
}

Fonc_Num  nflag_sym(Fonc_Num f)
{
    return op_sym_gen(f,Opb_Sym_Flag::sym);
}

Fonc_Num  nflag_close_sym(Fonc_Num f)
{
    return op_sym_gen(f,Opb_Sym_Flag::ferm_sym);
}

Fonc_Num  nflag_open_sym(Fonc_Num f)
{
    return op_sym_gen(f,Opb_Sym_Flag::ouv_sym);
}

Fonc_Num  nflag_open_sym_id(Fonc_Num f)
{
    return op_sym_gen(f,Opb_Sym_Flag::ouv_sym,true);
}


/*********************************************************************/
/*                                                                   */
/*   Creation de graphe de pixels                                    */
/*                                                                   */
/*********************************************************************/

template <class Type> void flag_min
                           (
                                Type **                   outs,
                                Type ***                  ins,
                                INT                       dk,
                                const Simple_OPBuf_Gen & arg
                           )
{
   for (INT d =0; d<arg.dim_out(); d++)
   {
        Type * out = outs[d];
        Type ** in = ins[d];
        for (INT x=arg.x0() ;  x<arg.x1() ; x++)
        {
            Type Vmin = in[0][x];
            INT  KMin = -1;
            for (INT k=0; k< 4 ; k+= dk)
            {
                 Pt2di p = TAB_8_NEIGH[k];
                 if (in[p.y][x+p.x]<Vmin)
                 {
                     KMin = k;
                     Vmin = in[p.y][x+p.x];
                 }
            }
			{
            for (INT k=4; k< 8 ; k+= dk)
            {
                 Pt2di p = TAB_8_NEIGH[k];
                 if (in[p.y][x+p.x]<=Vmin)
                 {
                     KMin = k;
                     Vmin = in[p.y][x+p.x];
                 }
            }
			}
            out[x] = (KMin <0) ? 0 : (1<<KMin );
        }
   }

}

template <class Type> void flag_min_8
                           (
                                Type ** outs,
                                Type *** ins,
                                const Simple_OPBuf_Gen & arg
                           )

{
    flag_min(outs,ins,1,arg);
}


Fonc_Num flag_min8(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                flag_min_8,
                flag_min_8,
                f,
                f.dimf_out(),
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}


void flag_visu4front
     (
            INT **                   outs,
            INT ***                  ins,
            const Simple_OPBuf_Gen & arg
     )
{
    INT * out = outs[0];
    INT ** in = ins[0];
    for (INT x=arg.x0() ;  x<arg.x1() ; x++)
    {
        out[x] = 0;
        for (INT k=0; k< 8 ; k+= 2)
        {
            Pt2di p = TAB_8_NEIGH[k];
            if (in[p.y][x+p.x] != in[0][x])
               out[x] |= 1<< k;
        }
    }
}

Fonc_Num flag_visu4front(Fonc_Num f)
{
     ELISE_ASSERT(f.dimf_out()==1,"Bad Dim in flag_visu4front");
     return create_op_buf_simple_tpl
            (
                flag_visu4front,
                0,
                f,
                1,
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
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
