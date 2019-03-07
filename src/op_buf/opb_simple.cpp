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


/*********************************************************************/
/*                                                                   */
/*         Courbe de Niveau                                                 */
/*                                                                   */
/*********************************************************************/

template <class Type> void  calc_cdn
                           (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg
                           )
{
     Tjs_El_User.ElAssert
     (
          arg.dim_in() == 1,
          EEM0 << "bobs_grad requires dim out = 1 for func"
     );

   Type * lP = in[0][-1];
   Type * l0 = in[0][0];
   Type * l1 = in[0][1];

   Type * l  = out[0];

   for (INT x=arg.x0() ;  x<arg.x1() ; x++)
   {
         l[x] =       (l0[x]<l0[x+1])
                   || (l0[x]<l0[x-1])
                   || (l0[x]<l1[x])
                   || (l0[x]<lP[x]) ;
   }
}

Fonc_Num cdn(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                calc_cdn,  // Nouvelle syntaxe
                0,
                f,
                1,
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}

/*********************************************************************/
/*                                                                   */
/*         Laplacien                                                 */
/*                                                                   */
/*********************************************************************/

template <class Type> void  calc_Laplacien
                           (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg
                           )
{
     Tjs_El_User.ElAssert
     (
          arg.dim_in() == 1,
          EEM0 << "bobs_grad requires dim out = 1 for func"
     );

   Type * lP = in[0][-1];
   Type * l0 = in[0][0];
   Type * l1 = in[0][1];

   Type * l  = out[0];

   for (INT x=arg.x0() ;  x<arg.x1() ; x++)
   {
         l[x] =            -  1*lP[x]
               -1* l0[x-1] +  4*l0[x] - 1*l0[x+1]
                         -  1*l1[x] ;
   }
}

Fonc_Num Laplacien(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                calc_Laplacien,  // Nouvelle syntaxe
                calc_Laplacien,
                f,
                1,
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}


/*********************************************************************/
/*                                                                   */
/*         Robert's gradient                                         */
/*                                                                   */
/*********************************************************************/

template <class Type> void  calc_bobs_grad
                           (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg
                           )
{
     Tjs_El_User.ElAssert
     (
          arg.dim_in() == 1,
          EEM0 << "bobs_grad requires dim out = 1 for func"
     );

   Type * l0 = in[0][0];
   Type * l1 = in[0][1];

   Type * gx = out[0];
   Type * gy = out[1];

   for (INT x=arg.x0() ;  x<arg.x1() ; x++)
   {
       gx[x] = l0[x+1] - l0[x];
       gy[x] = l1[x] -l0[x];
   }
}

Fonc_Num bobs_grad(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                calc_bobs_grad,  // Nouvelle syntaxe
                calc_bobs_grad,
                f,
                2,
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}

/*********************************************************************/
/*                                                                   */
/*         der_sec_calc                                              */
/*                                                                   */
/*********************************************************************/

template <class Type> void  der_sec_calc
                            (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg
                            )
{
     Tjs_El_User.ElAssert
     (
          arg.dim_in() == 1,
          EEM0 << "sec_deriv requires dim out = 1 for func"
     );

    Type * lm1 = in[0][-1];
    Type * l0  = in[0][0];
    Type * l1  = in[0][1];

    Type * dxx = out[0];
    Type * dxy = out[1];
    Type * dyy = out[2];

    for (INT x=arg.x0() ;  x<arg.x1() ; x++)
    {
        dxx[x] = (l0[x+1] + l0[x-1] - 2 * l0[x]);
        dxy[x] = (l1[x+1] + lm1[x-1] -l1[x-1] -lm1[x+1])/4;
        dyy[x] = (l1[x]   + lm1[x]  - 2 * l0[x]);
    }
}

Fonc_Num sec_deriv(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                der_sec_calc,
                der_sec_calc,
                f,
                3,
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}

/*********************************************************************/
/*                                                                   */
/*         grad_crois                                                */
/*                                                                   */
/*********************************************************************/

template <class Type> void calc_grad_crois
                           (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg
                           )
{
     Tjs_El_User.ElAssert
     (
          arg.dim_in() == 1,
          EEM0 << "grad_crois requires dim out = 1 for func"
     );

   Type * l0 = in[0][0];
   Type * l1 = in[0][1];

   Type * gxy = out[0];

   for (INT x=arg.x0() ;  x<arg.x1() ; x++)
       gxy[x] = sqrt
                (
                    ElSquare(l0[x+1] - l1[x])
                 +  ElSquare(l0[x]   - l1[x+1])
                );

}

Fonc_Num grad_crois(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                0,
                calc_grad_crois,
                f,
                1,
                Box2di(Pt2di(0,0),Pt2di(1,1))
            );
}

/*********************************************************************/
/*                                                                   */
/*         Courbure Tangente aux courbe de niveaux                   */
/*                                                                   */
/*********************************************************************/

void  calc_courb_tgt ( REAL ** out, REAL *** in, const Simple_OPBuf_Gen & arg)
{
    Tjs_El_User.ElAssert
    (
          arg.dim_in() == 2,
          EEM0 << "courb tgt requires dim out = 1 for func"
    );

    REAL** i0 = in[0]; // Image
    REAL** p0 = in[1]; // power
    REAL*  o0 = out[0];

    for (INT x=arg.x0() ;  x<arg.x1() ; x++)
    {
       REAL gx = (i0[0][x+1] - i0[0][x-1]) / 2;
       REAL gy = (i0[1][x]   - i0[-1][x] ) / 2;
       REAL g2 = (gx * gx + gy * gy);

       if (g2 == 0)
       {
          o0[x] = 0;
       }
       else
       {
          double p = p0[0][x];
          if (p !=1)
             g2 = pow(g2,p);
          REAL c_xx = (i0[0][x+1] + i0[0][x-1]  - 2 * i0[0][x]);
          REAL c_yy = (i0[1][x]   + i0[-1][x]   - 2 * i0[0][x]);
          REAL c_xy = (i0[1][x+1] + i0[-1][x-1] - i0[1][x-1] - i0[-1][x+1]) / 4;

          o0[x] =     ( c_xx * gy * gy - 2 * c_xy * gx * gy + c_yy * gx * gx) / g2;
        }
   }
}

Fonc_Num courb_tgt(Fonc_Num FCourb,Fonc_Num FExpos)
{
     return create_op_buf_simple_tpl
            (
                0,  // Nouvelle syntaxe
                calc_courb_tgt,
                Virgule(FCourb,FExpos),
                1,
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}


Fonc_Num courb_tgt(Fonc_Num f)
{
     return courb_tgt(f,1);
}

/*********************************************************************/
/*                                                                   */
/*               Filtre Reduc Binaire                                */
/*                                                                   */
/*********************************************************************/

template <class Type>  class cFilRedBin_OPBuf1  : public Simple_OPBuf1<Type,Type>
{
     public :
         cFilRedBin_OPBuf1
         (
                bool aRedX,
                bool aRedY,
                int aDiv,
                bool  aConsValSpec,
                const Type & aValSpec
         ) :
              Simple_OPBuf1<Type,Type> (),
              mRedX        (aRedX),
              mRedY        (aRedY),
          mDiv         (aDiv),
              mConsValSpec (aConsValSpec),
              mValSpec     (aValSpec)
         {
         }
   private :
      bool mRedX;
      bool mRedY;
      int  mDiv;
      bool mConsValSpec;
      Type mValSpec;

      virtual void  calc_buf(Type ** output,Type *** input);
};


template <class Type>  void cFilRedBin_OPBuf1<Type>::calc_buf
                            (
                                 Type ** output,
                                 Type *** input
                            )
{
    for (INT d=0; d<this->dim_out() ; d++)
    {
          Type * o = output[d];
          Type ** i = input[d];

           Type * lP = i[-1];
           Type * lC = i[0];
           Type * lN = i[1];

          if (mRedX && mRedY)
          {
             if ((this->ycur()%2)==0)
             {
                for (int x=this->x0(); x< this->x1(); x++)
                {
                   if ((x%2)==0)
                   {
                      o[x] = (
                                lP[x-1] + 2*lP[x] +   lP[x+1]
                            + 2*lC[x-1] + 4*lC[x] + 2*lC[x+1]
                            +   lN[x-1] + 2*lN[x] +   lN[x+1]
                          ) / mDiv;
                      if (
                              mConsValSpec
                           && (
                                      (lP[x-1] == mValSpec)
                                 ||   (lP[x]   == mValSpec)
                                 ||   (lP[x+1] == mValSpec)

                                 ||   (lC[x-1] == mValSpec)
                                 ||   (lC[x]   == mValSpec)
                                 ||   (lC[x+1] == mValSpec)

                                 ||   (lN[x-1] == mValSpec)
                                 ||   (lN[x]   == mValSpec)
                                 ||   (lN[x+1] == mValSpec)
                              )
                         )
                      {
                         o[x] = mValSpec;
                      }
                   }
                   else
                   {
                      o[x] = 0;
                   }
                }
             }
             else
             {
                for (int x=this->x0(); x< this->x1(); x++)
                {
                   o[x] = 0;
                }
             }
          }
          else if (mRedX)
          {
                for (int x=this->x0(); x< this->x1(); x++)
                {
                   if ((x%2)==0)
                   {
                      o[x] = ( lC[x-1] + 2*lC[x] + lC[x+1])/mDiv;
                      if (
                              mConsValSpec
                           && (
                                      (lC[x-1] == mValSpec)
                                 ||   (lC[x]   == mValSpec)
                                 ||   (lC[x+1] == mValSpec)
                              )
                         )
                      {
                         o[x] = mValSpec;
                      }
                   }
                   else
                   {
                      o[x] = 0;
                   }
                }
          }
          else if (mRedY)
          {
             if ((this->ycur()%2)==0)
             {
                for (int x=this->x0(); x< this->x1(); x++)
                {
                      o[x] =(lP[x]+2*lC[x]+lN[x])/mDiv;
                      if (
                              mConsValSpec
                           && (
                                      (lP[x]   == mValSpec)
                                 ||   (lC[x]   == mValSpec)
                                 ||   (lN[x]   == mValSpec)
                              )
                         )
                      {
                         o[x] = mValSpec;
                      }
                }
             }
             else
             {
                for (int x=this->x0(); x< this->x1(); x++)
                {
                   o[x] = 0;
                }
             }
          }
     }
}

Fonc_Num reduc_binaire_gen
         (
              Fonc_Num f,
              bool aRedX,
              bool aRedY,
              int aDiv,
              bool HasValSpec,
              REAL aValSpec =0
          )
{

    if ((! aRedX) && (! aRedY))
       return f;
    return create_op_buf_simple_tpl
           (
                new  cFilRedBin_OPBuf1<INT> (aRedX,aRedY,aDiv,HasValSpec,round_ni(aValSpec)) ,
                new  cFilRedBin_OPBuf1<REAL>(aRedX,aRedY,aDiv,HasValSpec,aValSpec),
                f,
                f.dimf_out(),
                Box2di(Pt2di(1,1),Pt2di(-1,-1))
           );
}

Fonc_Num reduc_binaire (Fonc_Num f)
{
    return reduc_binaire_gen(f,true,true,16,false);
}

Fonc_Num reduc_binaire_X (Fonc_Num f)
{
    return reduc_binaire_gen(f,true,false,4,false);
}

Fonc_Num reduc_binaire_Y (Fonc_Num f)
{
    return reduc_binaire_gen(f,false,true,4,false);
}



#if (ELISE_X11)
Video_Win * TheWinAffRed = 0;
#endif

void MakeTiffRed2Gen
     (
          int                   aNbReduc,
          const std::string &   aNameFul,
          const std::string &   aNameRed,
          bool                  UseType,
          GenIm::type_el        aTypeDem,
          int                   aDiv,
          bool                  HasVS,
          REAL                  aVSpec
     )
{
  // Reductions multiples, pas au point
   ELISE_ASSERT(aNbReduc==1,"MakeTiffRed2Gen::aNbReduc");


    Tiff_Im aTifIn = Tiff_Im::StdConvGen(aNameFul.c_str(),-1,true,true);
    Pt2di aSz = aTifIn.sz();
    Pt2di aSzRed = (aSz+Pt2di(1,1))/2;

    GenIm::type_el aTypeOut = UseType ? aTypeDem : aTifIn.type_el();
    Tiff_Im aTifRed
            (
                 aNameRed.c_str(),
                 aSzRed,
                 aTypeOut,
                 Tiff_Im::No_Compr,
                 aTifIn.phot_interp()
            );

    Fonc_Num aFonc = aTifIn.in_proj();

    if (! type_im_integral(aTypeOut))
    {
       aFonc = Rconv(aFonc);
    }

    for (int aK=0 ; aK< aNbReduc ; aK++)
        aFonc = reduc_binaire_gen(aFonc,true,true,aDiv,HasVS,aVSpec);

    int aMax,aMin;
    if (type_im_integral(aTypeOut) && (aDiv!=16))
    {
       min_max_type_num(aTypeOut,aMin,aMax);
       if (aMax>0)
          aFonc = Max(aMin,Min(aMax-1,aFonc));
    }

    Output anOut = aTifRed.out();
    for (int aK=0 ; aK< aNbReduc ; aK++)
        anOut = Filtre_Out_RedBin(anOut);

#if (ELISE_X11)
   if (TheWinAffRed)
   {
       Pt2dr aSz (TheWinAffRed->sz());
       Pt2dr aSzIn (aTifIn.sz());
       anOut = anOut | (TheWinAffRed->chc(Pt2dr(0,0),aSz.dcbyc(aSzIn),false).odisc(true) << P8COL::red);
   }
#endif

    ELISE_COPY(aTifIn.all_pts(),aFonc,anOut);
}

void MakeTiffRed2BinaireWithCaracIdent
     (
          const std::string &   aNameFul,
          const std::string &   aNameRed,
          REAL                  aRatio,
          Pt2di                 aSzRed
     )
{
    Tiff_Im aTifIn = Tiff_Im::BasicConvStd(aNameFul.c_str());

    MakeTiffRed2Binaire
    (
        aNameFul,
        aNameRed,
        aRatio,
        aTifIn.mode_compr(),
        aTifIn.type_el(),
        aTifIn.sz_tile(),
        aSzRed,
        true
    );
}


void MakeTiffRed2Binaire
     (
          const std::string &   aNameFul,
          const std::string &   aNameRed,
          REAL                  aRatio ,
          Tiff_Im::COMPR_TYPE   aModeCompr,
          GenIm::type_el        aType,
          Pt2di                 aSzTileFile,
          Pt2di                 aSzRed,
          bool                  DynOType
     )
{
    // Tiff_Im aTifIn = Tiff_Im::StdConvGen(aNameFul.c_str(),-1,true,true);
    Tiff_Im aTifIn = Tiff_Im::BasicConvStd(aNameFul.c_str());
    Pt2di aSz = aTifIn.sz();
    if (aSzRed==Pt2di(-1,-1))
       aSzRed = (aSz+Pt2di(1,1))/2;

    Tiff_Im aTifRed
            (
                 aNameRed.c_str(),
                 aSzRed,
                 aType,
                 // GenIm::bits1_msbf,
                 aModeCompr,
                 // Tiff_Im::No_Compr,
                 //Tiff_Im::Group_4FAX_Compr,
                 aTifIn.phot_interp(),
                   Tiff_Im::Empty_ARG
        + Arg_Tiff(Tiff_Im::ANoStrip())
                //+  Arg_Tiff(Tiff_Im::AFileTiling(aSzTileFile))
            );

    Im2D_Bits<1> aIm (aSz.x,aSz.y);
    ELISE_COPY(aTifIn.all_pts(),aTifIn.in_bool(),aIm.out());
    Fonc_Num aFonc = aIm.in_proj();

    int aSeuil = round_ni(aRatio*16);
    aSeuil = ElMax(0,ElMin(15,aSeuil));
    aFonc = reduc_binaire_gen(aFonc,true,true,1,false,0) > aSeuil;

    int aMul = 1;
    if (DynOType)
    {
        int aVMin,aVMax;
        min_max_type_num(aType,aVMin,aVMax);
        aMul = aVMax-1;
    }

    ELISE_COPY
    (
        rectangle(Pt2di(0,0),aSzRed*2+Pt2di(2,2)),
        aFonc * aMul,
        Filtre_Out_RedBin(aTifRed.out())
    );
}

void MakeTiffRed2
     (
          int   aNbReduc,
          const std::string & aNameFul,
          const std::string & aNameRed
     )
{
     MakeTiffRed2Gen
     (
          1,
          aNameFul,
      aNameRed,
      false,
      GenIm::u_int1, // Inutile
      16,
          false,
          0       // Inutile
     );
}

void MakeTiffRed2
     (
          const std::string & aNameFul,
          const std::string & aNameRed
     )
{
    MakeTiffRed2(1,aNameFul,aNameRed);
}

void MakeTiffRed2
     (
          const std::string & aNameFul,
          const std::string & aNameRed,
      GenIm::type_el        aType,
      int                   aDiv,
          bool                  HasVS,
          REAL                  aVSpec
     )
{
     MakeTiffRed2Gen
     (
          1,
          aNameFul,
      aNameRed,
      true,
      aType,
      aDiv,
          HasVS,
          aVSpec
     );
}

/*********************************************************************/
/*                                                                   */
/*               som_masq                                            */
/*                                                                   */
/*********************************************************************/


const Pt2di som_masq_Centered(0x7FFFFFFF,-0x7FFFFFFF);

void  compute_def_dec_som_masq(Pt2di & dec,INT tx,INT ty)
{
    if (dec==som_masq_Centered)
    {
          Tjs_El_User.ElAssert
          (
              (tx%2) && (ty%2),
              EEM0 << "need odd size for centered som_masq"
          );

          dec =  -Pt2di(tx/2,ty/2);
    }
}


template <class Type>  class som_masq_OPBuf1  : public Simple_OPBuf1<Type,Type>
{
   public :

      som_masq_OPBuf1
      (
             Im2D<Type,Type> b
      )   :
          _d (b.data()),
          _b (b)
      {
      }

   private :

      virtual void  calc_buf(Type ** output,Type *** input);



      Type    **       _d;
      Im2D<Type,Type> _b;
};



template <class Type> void som_masq_OPBuf1<Type>::calc_buf
                     (
                           Type ** output,
                           Type *** input
                     )
{
    for (INT d=0; d<this->dim_out() ; d++)
    {
          Type * o = output[d];
          Type ** i = input[d];
          set_cste(o+this->x0(),(Type)0,this->tx());
          for (int dy = this->dy0() ; dy <= this->dy1() ; dy++)
               for (int dx = this->dx0() ; dx <= this->dx1() ; dx++)
               {
                   Type val = _d[dy-this->dy0()][dx-this->dx0()];
                   if (val != 0)
                   {
                       Type * l = i[dy]+dx;
                       for (int x=this->x0(); x< this->x1(); x++)
                           o[x] += val * l[x];
                   }
               }
    }
}


Fonc_Num som_masq (Fonc_Num f,Im2D_REAL8 rfiltr,Pt2di dec)
{
    compute_def_dec_som_masq(dec,rfiltr.tx(),rfiltr.ty());

    Im2D_INT4 ifiltr(rfiltr.tx(),rfiltr.ty());

    for (INT x = 0; x < rfiltr.tx(); x++)
       for (INT y = 0; y < rfiltr.ty(); y++)
           ifiltr.data()[y][x] = round_ni(rfiltr.data()[y][x]);

    return create_op_buf_simple_tpl
           (
                new  som_masq_OPBuf1<INT>(ifiltr),
                new  som_masq_OPBuf1<REAL>(rfiltr),
                f,
                f.dimf_out(),
                Box2di
                (
                    dec,
                    dec + Pt2di(rfiltr.tx()-1,rfiltr.ty()-1)
                )
           );
}


/*********************************************************************/
/*                                                                   */
/*               rle_som_masq_binaire                                */
/*                                                                   */
/*********************************************************************/

template <class Type>  class RLE_SMB_OPBuf1  : public Simple_OPBuf1<Type,Type>
{
   public :

      RLE_SMB_OPBuf1(Im2D_U_INT1 b,Pt2di dec,Type val_out);

   private :

      virtual void  calc_buf(Type ** output,Type *** input);
      void cumul_line(Type *);

      Im2D_U_INT1       _b;
      ElFifo<Pt3di>     _FifRle;  // x=>x0, y=> y0, z => x1
      Pt3di *           _rle;
      INT               _nb_rle;
      Type              _val_out;
};

template <class Type>
         RLE_SMB_OPBuf1<Type>::RLE_SMB_OPBuf1
         (
              Im2D_U_INT1  b,
              Pt2di        dec,
              Type         val_out
         ) :
    _b       (b),
    _val_out (val_out)
{
    U_INT1 **  d = b.data();
    INT ty = b.ty();
    INT tx = b.tx();

    for (INT y=0; y<ty; y++)
    {
        INT x1 = -1;
        INT x0 ;
        while (x1 < tx-1 )
        {
             x0 = x1+1;
             while ((x0 < tx) && (d[y][x0] == 0)) x0++;
             x1 = x0;
             if (x0 < tx)
             {
                 while ((x1 < tx) && (d[y][x1] != 0)) x1++;
                 x1 --;
                 _FifRle.pushlast(Pt3di(x0-1+dec.x,y+dec.y,x1+dec.x));
             }
        }
    }
    _rle    = _FifRle.tab();
    _nb_rle = _FifRle.nb();
}


template <class Type> void  RLE_SMB_OPBuf1<Type>::cumul_line(Type * l)
{
     for (INT x = this->x0Buf()+1; x< this->x1Buf() ; x++)
         l[x] += l[x-1];
}

template <class Type> void RLE_SMB_OPBuf1<Type>::calc_buf
                     (
                           Type ** output,
                           Type *** input
                     )
{
    Tjs_El_User.ElAssert
    (
          this->dim_in() == 2,
          EEM0 << "rle_som_masq_binaire requires dim out = 1 for func"
    );

    Type * o   = output[0];
    Type ** i  = input[0];
    Type * sel = input[1][0];

     if (this->first_line())
        for (int dy = this->dy0() ; dy<this->dy1() ; dy++)
            cumul_line(i[dy]);
     cumul_line(i[this->dy1()]);

     for (int x = this->x0() ; x < this->x1() ; x++)
     {
         if (sel[x])
         {
            Type res = 0;
            for (INT k=0; k < _nb_rle ; k++)
            {
        Pt3di seg = _rle[k];
                res +=  i[seg.y][seg.z+x]-i[seg.y][seg.x+x];
            }
             o[x] = res;
         }
         else
             o[x] = _val_out;
     }
}

Fonc_Num rle_som_masq_binaire
         (Fonc_Num f,Im2D_U_INT1  filtr,REAL val_out,Pt2di dec)
{
    compute_def_dec_som_masq(dec,filtr.tx(),filtr.ty());

    return create_op_buf_simple_tpl
           (
                new  RLE_SMB_OPBuf1<INT>(filtr,dec,round_ni(val_out)),
                new  RLE_SMB_OPBuf1<REAL>(filtr,dec,val_out),
                f,
                1,
                Box2di
                (
                    dec + Pt2di(-1,0),
                    dec + Pt2di(filtr.tx()-1,filtr.ty()-1)
                )
           );
}

/*********************************************************************/
/*                                                                   */
/*         Reduction sur voisinage   flague                          */
/*                                                                   */
/*********************************************************************/

template <class Type> class RedFlagSom
{
    public :
         static const Type neutre;
         static void SetEq(Type & var,Type val) { var += val;}
};
template <> const REAL RedFlagSom<REAL>::neutre = 0.0;
template <> const INT  RedFlagSom<INT>::neutre = 0;



template <class Type,class Oper> void  oper_red_flag
                           (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg,
                                Oper *
                           )
{
   Type * Flags = in[arg.dim_in()-1][0];


   for (INT d=0 ; d<arg.dim_in()-1 ; d++)
   {
       Type * Res = out[d];
       Type ** Vals  = in[d];


       for (INT x=arg.x0() ;  x<arg.x1() ; x++)
       {
           INT flag = (INT) Flags[x];
           Type res = Oper::neutre;
           for (INT k=0 ; k<9 ; k++)
               if (flag & (1<<k))
                  Oper::SetEq(res,Vals[TAB_9_NEIGH[k].y][x+TAB_9_NEIGH[k].x]);

           Res[x] = res;
       }
   }
}

template <class Type> void oper_red_flag_som ( Type ** out, Type *** in, const Simple_OPBuf_Gen & arg)
{
    oper_red_flag(out,in,arg,(RedFlagSom<Type> *) 0);
}

Fonc_Num red_flag_som(Fonc_Num Fvals,Fonc_Num Fflags)
{
     ELISE_ASSERT(Fflags.dimf_out()==1,"bad dim in red_flag_som");
     return create_op_buf_simple_tpl
            (
                oper_red_flag_som,
                oper_red_flag_som,
                Virgule(Fvals,Fflags),
                Fvals.dimf_out(),
                Box2di(Pt2di(-1,-1),Pt2di(1,1))
            );
}


/*********************************************************************/
/*                                                                   */
/*         Pente crete                                               */
/*                                                                   */
/*********************************************************************/

template <class Type> inline Type FreemanVals(Type ** Im,INT x,INT k)
{
   return Im[TAB_8_NEIGH[k].y][x+TAB_8_NEIGH[k].x];
}

template <class Type> void calc_pente_crete_gen
                           (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg,
                                bool  ModeFonc
                           )
{

   for (INT x=arg.x0() ;  x<arg.x1() ; x++)
   {
       Type  CritMax = -1;
       INT   FlagRes = 0;

       for (INT k=0 ; k<4 ; k++)
       {
           Type CritCrete = 0;
           Type CritPente = 0;
           Type CritPente_A = 0;
           Type CritPente_C = 0;

           for (INT d=0 ; d< arg.dim_in() ; d++)
           {
               Type vB =    in[d][0][x]
                         +  FreemanVals(in[d],x,k)
                         +  FreemanVals(in[d],x,k+4);

               Type vA =    FreemanVals(in[d],x,k+1)
                         +  FreemanVals(in[d],x,k+2)
                         +  FreemanVals(in[d],x,k+3);

               Type vC =    FreemanVals(in[d],x,k+5)
                         +  FreemanVals(in[d],x,k+6)
                         +  FreemanVals(in[d],x,k+7);

                CritCrete   += ElAbs(2*vB-(vA+vC));
                CritPente   += ElAbs(vA-vC);
                CritPente_A += ElAbs(vA-vB);
                CritPente_C += ElAbs(vC-vB);
           }
           if (CritCrete > CritPente)
           {
               if (CritCrete > CritMax)
               {
                    CritMax = CritCrete;
                    FlagRes = (1 << k) | (1 << (k+4)) | (1<<8);
               }
           }
           else
           {
               if (CritPente > CritMax)
               {
                    CritMax = CritPente;
                    if (CritPente_A < CritPente_C)
                        FlagRes = (1 << (k+1))  | (1 << (k+2)) | (1 << (k+3));
                    else
                        FlagRes = (1 << ((k+5)%8)) | (1 << ((k+6)%8)) | (1 << ((k+7)%8));
               }
           }
       }
       if (ModeFonc)
       {
           {
           for (INT d=0; d<arg.dim_in() ; d++)
                out[d][x] = 0;
           }
           INT NbF = 0;
           for (INT k=0 ; k<9; k++)
           {
               if (FlagRes & (1<<k))
               {
                   NbF++;
                   for (INT d=0; d<arg.dim_in() ; d++)
                        out[d][x] +=   FreemanVals(in[d],x,k);
               }
           }
           {
           for (INT d=0; d<arg.dim_in() ; d++)
                out[d][x] /= NbF;
           }
       }
       else
       {
          out[0][x] = FlagRes;
       }
   }
}

template <class Type> void calc_flag_pente_crete
                           (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg
                           )
{
    calc_pente_crete_gen(out,in,arg,false);
}

Fonc_Num flag_pente_crete(Fonc_Num f)
{
     return Iconv
            (
                 create_op_buf_simple_tpl
                 (
                    calc_flag_pente_crete,  // Nouvelle syntaxe
                    calc_flag_pente_crete,
                    f,
                    1,
                    Box2di(Pt2di(-1,-1),Pt2di(1,1))
                 )
            );

}

template <class Type> void calc_filtre_pente_crete
                           (
                                Type ** out,
                                Type *** in,
                                const Simple_OPBuf_Gen & arg
                           )
{
    calc_pente_crete_gen(out,in,arg,true);
}

Fonc_Num filtre_pente_crete(Fonc_Num f)
{
     return Iconv
            (
                 create_op_buf_simple_tpl
                 (
                    calc_filtre_pente_crete,  // Nouvelle syntaxe
                    calc_filtre_pente_crete,
                    f,
                    f.dimf_out(),
                    Box2di(Pt2di(-1,-1),Pt2di(1,1))
                 )
            );

}


/*********************************************************************/
/*                                                                   */
/*         Pente crete                                               */
/*                                                                   */
/*********************************************************************/

template <class Type> void CalcMedianBySort
                           (
                                Type ** out,   // out[c][x]
                                Type *** in,   // in[c][y][x]      c=>channel
                                const Simple_OPBuf_Gen & arg
                           )
{

   for (INT d=0 ; d< arg.dim_in() ; d++)
   {
       Type * res = out[d];
       Type ** Im = in[d];
       std::vector<Type> Vals;

       // INT aK = ((arg.dx1()- arg.dx0() +1) * (arg.dy1()- arg.dy0() +1)) / 2;



       for (INT x=arg.x0() ;  x<arg.x1() ; x++)
       {
          Vals.clear();

          for (INT dx =  arg.dx0() ; dx <= arg.dx1() ; dx++)
              for (INT dy =  arg.dy0() ; dy <= arg.dy1() ; dy++)
                  Vals.push_back(Im[dy][x+dx]);
          Type aMed = MedianeSup(Vals);
          res[x] = aMed;

       }
   }
}

Fonc_Num MedianBySort(Fonc_Num f,INT NbMed)
{
     return
            create_op_buf_simple_tpl
            (
                    CalcMedianBySort,  // Nouvelle syntaxe
                    CalcMedianBySort,
                    f,
                    f.dimf_out(),
                    Box2di(Pt2di(-NbMed,-NbMed),Pt2di(NbMed,NbMed))
            );

}

/*********************************************************************/
/*                                                                   */
/*         Ombrage K Lips                                            */
/*                                                                   */
/*********************************************************************/

// Normalement valent false, mais pour tester le mecanisme ~ et dup ...
#define DupcOmbrageKL  0
#define TestDupAndKill 0

class cOmbrageKL  : public Simple_OPBuf1<double,double>
{
   public :

        cOmbrageKL(double aPente,int aNbV) ;
   private :
        void  calc_buf(double ** output,double *** input);

#if DupcOmbrageKL		
		Simple_OPBuf1<double,double> * dup_comp() {return new cOmbrageKL(mPente,mNbV);}
#endif

#if TestDupAndKill 
        ~cOmbrageKL() 
        {
              std::cout << "KILL==cOmbrageKL\n"; getchar();
        }
#endif

        double mPente;
        int    mNbV;
        std::vector<double> mVPds;
};


cOmbrageKL::cOmbrageKL(double aPente,int aNbV) :
    mPente (aPente),
    mNbV   (aNbV)
{
    for (int anY=-aNbV  ; anY<=aNbV ; anY++)
    {
        for (int anX=-aNbV  ; anX<=aNbV ; anX++)
        {
             mVPds.push_back(aPente * euclid(Pt2di(anX,anY)));
        }
    }
}


void  cOmbrageKL::calc_buf(double ** output,double *** input)
{
// static int aCpt=0; aCpt++;
    ELISE_ASSERT(this->dim_in()==2,"Incoherence in cOmbrageKL::calc_buf");

    double  ** aProf = input[0];
    double  ** aMasq = input[1];
    double * anO = output[0];

    for (INT anX=x0() ;  anX<x1() ; anX++)
    {
// bool Bug = ((aCpt==19) && (anX==241));

         double aDifMax  = 0.0;
         if (aMasq[0][anX])
         {
             double aP0 = aProf[0][anX];
             int aCpt=0;
             for (int aDy=-mNbV  ; aDy<=mNbV ; aDy++)
             {
                 double * aM = aMasq[aDy]+anX;
                 double * aP = aProf[aDy]+anX;

                 for (int aDx=-mNbV  ; aDx<=mNbV ; aDx++)
                 {
                     if (aM[aDx])
                     {
                         ElSetMax(aDifMax,aP[aDx] -(aP0+mVPds[aCpt]));
                     }
                     aCpt++;
                 }
             }
         }
         anO[anX] = aDifMax;
    }
}
Fonc_Num OmbrageKL(Fonc_Num Prof,Fonc_Num Masq,double aPente,int aSzV)
{
     return create_op_buf_simple_tpl
            (
                    0,
                    new  cOmbrageKL(aPente,aSzV),
                    Virgule(Prof,Masq),
                    1,
                    Box2di(Pt2di(-aSzV,-aSzV),Pt2di(aSzV,aSzV))
            );

}

/*********************************************************************/
/*                                                                   */
/*         Dilate cond : limite ici a une dilatation une fois,       */
/*           Les dilate N sont geres par N Iter                      */
/*                                                                   */
/*********************************************************************/


class cDilateCondOPB  : public Simple_OPBuf1<int,int>
{
   public :

        cDilateCondOPB(bool aV4) :
           mV4     (aV4),
           mNbVois (mV4 ? 4 : 8),
           mVois   (mV4 ? TAB_4_NEIGH :TAB_8_NEIGH)
        {
        }
   private :
        void  calc_buf(int ** output,int *** input);

        bool           mV4;
        int            mNbVois;
        const Pt2di *  mVois;
};



void  cDilateCondOPB::calc_buf(int ** output,int *** input)
{
// static int aCpt=0; aCpt++;
    ELISE_ASSERT(this->dim_in()==2,"Incoherence in cOmbrageKL::calc_buf");

    int  ** aFoncToDil = input[0];
    int  ** aFoncCond = input[1];
    int * anO = output[0];

    for (INT anX=x0() ;  anX<x1() ; anX++)
    {
         bool Ok = false;
         if (aFoncToDil[0][anX])
         {
              Ok=true;
         }
         else if (aFoncCond[0][anX])
         {
              for (int aK=0 ; (!Ok) && (aK<mNbVois) ; aK++)
              {
                   const Pt2di * aV = mVois+ aK;
                   Ok = (aFoncToDil[aV->y][anX+aV->x] != 0);
              }
         }
         anO[anX] = Ok;
    }
}

Fonc_Num FoncDilatCond(Fonc_Num f2Dil,Fonc_Num fCond,bool aV4)
{
     int aSzV=1;
     return create_op_buf_simple_tpl
            (
                    new  cDilateCondOPB(aV4),
                    0,
                    Virgule(f2Dil,fCond),
                    1,
                    Box2di(Pt2di(-aSzV,-aSzV),Pt2di(aSzV,aSzV))
            );

}

Fonc_Num NFoncDilatCond(Fonc_Num f2Dil,Fonc_Num fCond,bool aV4,int aNb)
{
   Fonc_Num aRes = f2Dil;
   for (int aK=0 ; aK<aNb ; aK++)
   {
       aRes = FoncDilatCond(aRes,fCond,aV4);
   }
   return aRes;
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
