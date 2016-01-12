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
/*         Arg_Output_Comp                                           */
/*                                                                   */
/*********************************************************************/

Arg_Output_Comp::Arg_Output_Comp(Flux_Pts_Computed * flx,Fonc_Num_Computed * fonc) :
      _flux (flx),
      _fonc (fonc)
{
}


/***************************************************************/
/*                                                             */
/*          Output_Computed                                    */
/*                                                             */
/***************************************************************/


Output_Computed::Output_Computed(INT dim_consumed) :
        _dim_consumed (dim_consumed)
{
}
Output_Computed::~Output_Computed() {}



/***************************************************************/
/*                                                             */
/*          Output                                             */
/*                                                             */
/***************************************************************/



Output::Output(Output_Not_Comp * out) :
     PRC0(out)
{
}

Output_Computed * Output::compute(const Arg_Output_Comp & arg)
{
    ASSERT_TJS_USER
  (    _ptr != 0,
       "USE OF OutPut non initialized"
  );
   return SAFE_DYNC(Output_Not_Comp *,_ptr)->compute(arg);
}


       //   NUL OUTPUT

class Out_Nul_Comp :  public Output_Computed
{
      public :

         Out_Nul_Comp(INT dim_cons) : Output_Computed(dim_cons) {}

      private :

         void update( const Pack_Of_Pts *, const Pack_Of_Pts *) {}
};

class Out_Nul_Not_Comp :  public Output_Not_Comp
{

      public :

          Out_Nul_Not_Comp(INT DIM) : _dim_cons(DIM) {}

      private :

          Output_Computed * compute(const Arg_Output_Comp &)
          {
               return new Out_Nul_Comp(_dim_cons);
          }

          INT _dim_cons;
};

Output Output::onul(INT dim)
{
   return new Out_Nul_Not_Comp(dim);
}


/***************************************************************/
/*                                                             */
/*          PushValues                                         */
/*                                                             */
/***************************************************************/

template <class Type>
         void push_val_typed
              (
                       // Type *,
                       Std_Pack_Of_Pts<Type > ** ppi ,
                       const Pack_Of_Pts * vals
              )
{
       bool chang;
       // Std_Pack_Of_Pts<Type > ** ppi =
                // SAFE_DYNC(Std_Pack_Of_Pts<Type>**,pack_push);
       Std_Pack_Of_Pts<Type > * pvi =
                SAFE_DYNC(Std_Pack_Of_Pts<Type>*,const_cast<Pack_Of_Pts *>(vals));
       *ppi = pvi->cat_and_grow(*ppi,2*(*ppi)->pck_sz_buf(),chang);
}


          //=========================
          //   Push_Values_Out_Comp
          //=========================


template <class Type> class Push_Values_Out_Comp :  public Output_Computed
{
      public :

         Push_Values_Out_Comp
         (
              const Arg_Output_Comp &,
               Std_Pack_Of_Pts<Type >   ** pack_push
         ) :
           Output_Computed((*pack_push)->dim()),
           _pack_push     (pack_push)
         {
         }


         virtual void update( const Pack_Of_Pts * ,
                              const Pack_Of_Pts * vals)
         {
         push_val_typed(_pack_push,vals);
         }

      private :
          Std_Pack_Of_Pts<Type > **     _pack_push;
};

          //==============================
          //   Push_Values_Out_Not_Comp
          //==============================

template <class Type> class Push_Values_Out_Not_Comp :  public Output_Not_Comp
{

      public :

          Push_Values_Out_Not_Comp(Std_Pack_Of_Pts<Type > ** pack_push) :
              _pack_push (pack_push)
          {
          }

      private :

          virtual  Output_Computed * compute(const Arg_Output_Comp & arg)
          {

                 ASSERT_INTERNAL
                 (  (*_pack_push)->dim() <= arg.fonc()->idim_out(),
                    "insufficient dim in Push_Values_Out_Not_Comp"
                 );
                 return
                     out_adapt_type_fonc
                     (
                          arg,
                          new Push_Values_Out_Comp<Type>(arg,_pack_push),
                          (*_pack_push)->type()
                     );
          }

          Std_Pack_Of_Pts<Type > ** _pack_push;
};


          //==============================
          //   push_values
          //==============================

template<class Type> Output  push_values (Std_Pack_Of_Pts<Type > ** pp)
{
    return new Push_Values_Out_Not_Comp<Type>(pp);
}

template Output  push_values (Std_Pack_Of_Pts<int> ** pp);
template Output  push_values (Std_Pack_Of_Pts<double> ** pp);


Output Virgule(Output f1,Output f2,Output f3)
{
     return Virgule(f1, Virgule(f2 , f3));
}

Output Virgule(Output f1,Output f2,Output f3,Output f4)
{
     return Virgule(f1, Virgule(f2 , f3, f4));
}

Output Virgule(Output f1,Output f2,Output f3,Output f4,Output f5)
{
     return Virgule(f1, Virgule(f2 , f3, f4 ,f5));
}

Output Virgule(Output f1,Output f2,Output f3,Output f4,Output f5,Output f6)
{
     return Virgule(f1, Virgule(f2 , f3, f4 ,f5,f6));
}



/****************************************************************/
/*                                                              */
/*     Data_El_Video_Display                                    */
/*                                                              */
/****************************************************************/
#if (ELISE_NO_VIDEO)


ElXim::ElXim(Video_Win,Pt2di sz,Fonc_Num ,Elise_Palette )
{
    ELISE_ASSERT(false,"ElXim::ElXim in ELISE_NO_VIDEO mode");
}

ElXim::ElXim(Video_Win,Pt2di)
{
    ELISE_ASSERT(false,"ElXim::ElXim in ELISE_NO_VIDEO mode");
}

void Video_Win::write_image ( Pt2di, Pt2di, Pt2di, INT ***, Elise_Palette)
{
    ELISE_ASSERT(false,"Video_Win::write_image  in ELISE_NO_VIDEO mode");
}

void Video_Win::image_translate(Pt2di )
{
    ELISE_ASSERT(false,"Video_Win::image_translate  in ELISE_NO_VIDEO mode");
}


bool  Clik::b1Pressed() const
{
    ELISE_ASSERT(false,"Video_Win::Clik  in ELISE_NO_VIDEO mode");
    return false;
}
bool  Clik::b2Pressed() const
{
    ELISE_ASSERT(false,"Video_Win::Clik  in ELISE_NO_VIDEO mode");
    return false;
}
bool  Clik::b3Pressed() const
{
    ELISE_ASSERT(false,"Video_Win::Clik  in ELISE_NO_VIDEO mode");
    return false;
}
bool  Clik::controled() const
{
    ELISE_ASSERT(false,"Video_Win::Clik  in ELISE_NO_VIDEO mode");
    return false;
}
bool  Clik::shifted() const
{
    ELISE_ASSERT(false,"Video_Win::Clik  in ELISE_NO_VIDEO mode");
    return false;
}


Video_Display    Video_Win::disp()
{
    ELISE_ASSERT(false,"Video_Win::disp in ELISE_NO_VIDEO mode");
    return Video_Display("JJJJ");
}
void ElXim::fill_with_el_image
    (
          Pt2di, Pt2di, Pt2di,
          std::vector<Im2D_INT4> &, Elise_Palette
    )
{
    ELISE_ASSERT(false,"ElXim::fill_with_el_image in ELISE_NO_VIDEO mode");
}
void ElXim::fill_with_el_image
             (
                  Pt2di, Pt2di, Pt2di,
                  Im2D_U_INT1, Im2D_U_INT1, Im2D_U_INT1
             )
{
    ELISE_ASSERT(false,"ElXim::fill_with_el_image in ELISE_NO_VIDEO mode");
}

void Video_Win::grab(Grab_Untill_Realeased & )
{
    ELISE_ASSERT(false,"Video_Win::grab in ELISE_NO_VIDEO mode");
}

ElXim  Video_Win::StdBigImage()
{
   ELISE_ASSERT(false,"Video_Win::StdBigImage in ELISE_NO_VIDEO mode");
   return ElXim(*this,Pt2di(0,0));
}

void Video_Win::load_image(Pt2di ,Pt2di )
{
   ELISE_ASSERT(false,"Video_Win::load_image in ELISE_NO_VIDEO mode");
}

void ElXim::write_image_per(Pt2di   ,Pt2di  ,Pt2di  )
{
   ELISE_ASSERT(false,"Video_Win::write_image_per in ELISE_NO_VIDEO mode");
}
void Video_Win::translate(Pt2di)
{
   ELISE_ASSERT(false,"Video_Win::translate in ELISE_NO_VIDEO mode");
}
void ElXim::load()
{
   ELISE_ASSERT(false,"ElXim::load in ELISE_NO_VIDEO mode");
}
void ElXim::load(Pt2di,Pt2di,Pt2di)
{
   ELISE_ASSERT(false,"ElXim::load in ELISE_NO_VIDEO mode");
}






Clik::Clik(Video_Win w,Pt2dr pt,INT b,U_INT state) :
      _w   (w),
      _pt  (pt),
      _b   (b),
      _state (state)
{
}

class Data_El_Video_Display : public  Data_Elise_Gra_Disp
{

      public :

          virtual void disp_flush() {};
          virtual void _inst_set_line_witdh(REAL) {} ;

          Data_El_Video_Display() {};
          virtual ~Data_El_Video_Display() {};




};


Video_Display::Video_Display(const char *) :
   PRC0(new  Data_El_Video_Display)
{
}

void Video_Display::load(Elise_Set_Of_Palette)
{
}


Clik   Video_Display::clik()
{
     ELISE_ASSERT(false,"Video_Display::clik in ELISE_NO_VIDEO mode");
     return Clik
           (
                Video_Win
                (
                    *this,
                    Elise_Set_Of_Palette(NewLElPal(Gray_Pal(10))),
                    Pt2di(0,0),
                    Pt2di(10,10)
                ),
                Pt2dr(0,0),
                1,
                0
           );
}


Clik   Video_Display::clik_press()
{
    return clik();
}

Data_El_Video_Display * Video_Display::devd()
{
    return SAFE_DYNC(Data_El_Video_Display *,_ptr);
}




class  Data_Elise_Video_Win  : public Data_Elise_Gra_Win
{
      public :

       Data_Elise_Video_Win(Video_Display VD,Elise_Set_Of_Palette sop,Pt2di sz) :
            Data_Elise_Gra_Win ( VD.devd(), sz, sop, false),
            mVD (VD)
       {
       }


       virtual  Data_Elise_Gra_Win * dup_geo(Pt2dr ,Pt2dr) {return this;}
       virtual  bool adapt_vect()  {return false;}
       virtual  void _inst_set_col(Data_Col_Pal *) {}

        virtual   Output_Computed
                  * rle_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & ,
                       Data_Elise_Palette * dep,
                       bool OnYDiff

                   )
                   {
                          return new Out_Nul_Comp(dep->dim_pal() );
                   }

        virtual   Output_Computed
                  * pint_cste_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & ,
                       Data_Elise_Palette *  dep,
                       INT        *
                   )
                   {
                          return new Out_Nul_Comp(dep->dim_pal() );
                   }


        virtual   Output_Computed
                  * pint_no_cste_out_comp
                  (
                       const Data_El_Geom_GWin *,
                       const Arg_Output_Comp & ,
                       Data_Elise_Palette * dep
                   )
                   {
                          return new Out_Nul_Comp(dep->dim_pal() );
                   }

       virtual void  _inst_draw_seg(Pt2dr,Pt2dr) {}

       virtual void  _inst_fill_rectangle(Pt2dr,Pt2dr) {}

       virtual void  _inst_draw_circle(Pt2dr ,Pt2dr ) {}

    private :
       Video_Display mVD;
};

void  Video_Win::set_sop(Elise_Set_Of_Palette) {}
void  Video_Win::raise(){}
void  Video_Win::lower(){}
void  Video_Win::move_to(const Pt2di&){}

Video_Win::Video_Win
(
       Video_Display            d  ,
       Elise_Set_Of_Palette  esop  ,
       Pt2di                     ,
       Pt2di                    sz ,
       INT
)   :
    El_Window
    (
        new Data_Elise_Video_Win(d,esop,sz),
        Pt2dr(0.0,0.0),
        Pt2dr(1.0,1.0)
    )
{
}

Video_Win::Video_Win
(
       Video_Win                aSoeur,
       ePosRel                  aPos,
       Pt2di                    sz ,
       INT                      border_witdh
): El_Window((Data_El_Geom_GWin *)NULL)
{
   ELISE_ASSERT(false, "Video_Win::Video_Win in ELISE_NO_VIDEO mode");
}




Video_Win::Video_Win (class Data_Elise_Video_Win * w,Pt2dr tr,Pt2dr sc) :
       El_Window (w,tr,sc)
{
}


Video_Win Video_Win::chc(Pt2dr tr,Pt2dr sc,bool)
{
    return Video_Win(devw(),tr,sc);
}

Data_Elise_Video_Win * Video_Win::devw()
{
     return SAFE_DYNC (Data_Elise_Video_Win *,degraw());
}


void Video_Win::set_cl_coord(Pt2dr ,Pt2dr )
{
}

void Video_Win::set_title(const char * )
{
}

void Video_Win::clear()
{
}

Clik  Video_Win::clik_in()
{
     return Clik(*this,Pt2dr(1,1),1,0);
}



Video_Win *  Video_Win::PtrChc(class Pt2d<double>,class Pt2d<double>,bool)
{
    ELISE_ASSERT(false,"Video_Win::PtrChc in ELISE_NO_VIDEO mode");
    return 0;
}
 bool Video_Win::operator==(class Video_Win const &)const
 {
    ELISE_ASSERT(false,"Video_Win::== in ELISE_NO_VIDEO mode");
    return true;
 }
Video_Win Video_Win::chc_fit_sz(class Pt2d<double>,bool)
{
    ELISE_ASSERT(false,"Video_Win::chc_fit_sz in ELISE_NO_VIDEO mode");
    return *this;
}



#endif


Elise_Set_Of_Palette  Elise_Set_Of_Palette ::TheFullPalette()
{
    Disc_Pal  Pdisc = Disc_Pal::PNCOL();
    Gray_Pal  Pgr (30);
    Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
    RGB_Pal   Prgb  (2,2,2);

    return Elise_Set_Of_Palette
           (
                NewLElPal(Pdisc)
              + Elise_Palette(Pgr)
              + Elise_Palette(Prgb)
              + Elise_Palette(Pcirc)
           );
}


/*
Video_Win Video_Win::WStd(Pt2di sz,REAL zoom,bool all_pal,bool SetClikCoord)
{
        Disc_Pal  Pdisc = Disc_Pal::PNCOL();
        Gray_Pal  Pgr (30);
        Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
        RGB_Pal   Prgb  (2,2,2);
        Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
        if (! all_pal)
            SOP = Elise_Set_Of_Palette (NewLElPal(Pdisc));

        Video_Display Ecr((char *) NULL);
        Ecr.load(SOP);
        Video_Win   W  (Ecr,SOP,Pt2di(50,50),sz*zoom);
        return W.chc(Pt2dr(-0.5,-0.5),Pt2dr(zoom,zoom),SetClikCoord);
}
*/

Elise_Set_Of_Palette GlobPal(int aNbR,int aNbV,int aNbB,int aNbGray,int aNbCirc)
{
   Disc_Pal  Pdisc = Disc_Pal::PNCOL();
   Gray_Pal  Pgr (aNbGray);
   Circ_Pal  Pcirc = Circ_Pal::PCIRC6(aNbCirc);
   RGB_Pal   Prgb  (aNbR,aNbV,aNbB);
   Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));

   return SOP;
}
Elise_Set_Of_Palette GlobPal()
{
   return GlobPal(2,2,2,30,30);
}

Elise_Set_Of_Palette RGB_Gray_GlobPal()
{
   return GlobPal(6,6,6,24,2);
}



Video_Win * Video_Win::PtrWStd(Pt2di sz,bool all_pal,const Pt2dr & aScale)
{
    Disc_Pal  Pdisc = Disc_Pal::PNCOL();
    Gray_Pal  Pgr (30);
    Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
    RGB_Pal   Prgb  (2,2,2);
    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
    if (! all_pal)
        SOP = Elise_Set_Of_Palette (NewLElPal(Pdisc));

    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);
    Video_Win * aRes =  new Video_Win (Ecr,SOP,Pt2di(50,50),sz);

    aRes = aRes->PtrChc(Pt2dr(0,0),aScale);

    return aRes;
}

Video_Win Video_Win::WStd(Pt2di sz,REAL zoom,bool all_pal,bool SetClikCoord)
{
    Video_Win * aPW = PtrWStd(Pt2di(Pt2dr(sz)*zoom),all_pal);

    Video_Win aRes = aPW->chc(Pt2dr(-0.5,-0.5),Pt2dr(zoom,zoom),SetClikCoord);
    delete aPW;

    return aRes;
}







Video_Win Video_Win::WStd(Pt2di sz,REAL zoom,Video_Win Soeur,bool SetClikCoord)
{
     Video_Win   W  (Soeur.disp(),Soeur.sop(),Pt2di(50,50),Pt2di(Pt2dr(sz)*zoom));
     return W.chc(Pt2dr(-0.5,-0.5),Pt2dr(zoom,zoom),SetClikCoord);
}

Output Video_Win::WiewAv(Pt2di sz,Pt2di szmax)
{
    REAL zoom = ElMin(szmax.x/(REAL)sz.x,szmax.y/(REAL)sz.y);

    Video_Win res  = WStd(sz,zoom,false);

    Video_Win w0 = res.chc(Pt2dr(0,0),Pt2dr(1,1));
    ELISE_COPY(w0.all_pts(),P8COL::black,w0.odisc());

    return res.odisc(true) << (int) P8COL::red;
}







Grab_Untill_Realeased::Grab_Untill_Realeased(bool ONLYMVT) :
        _OnlyMvmt (ONLYMVT)
{
}


void Grab_Untill_Realeased::GUR_button_released(Clik)
{
}




void AdaptParamCopyTrans
     (
        INT&  X0src,
        INT&  X0dest,
        INT&  NB,
        INT   NbSrc,
        INT   NbDest
     )
{
   INT x0 = ElMin3(0,X0dest,X0src);
   X0src  -= x0;
   X0dest -= x0;
   NB += x0;
   NB = ElMax(0,ElMin3(NB,NbSrc-X0src,NbDest-X0dest));
}

void AdaptParamCopyTrans
     (
        Pt2di&  p0src,
        Pt2di&  p0dest,
        Pt2di&  sz,
        Pt2di   SzSrc,
        Pt2di   SzDest
     )
{
   AdaptParamCopyTrans(p0src.x,p0dest.x,sz.x,SzSrc.x,SzDest.x);
   AdaptParamCopyTrans(p0src.y,p0dest.y,sz.y,SzSrc.y,SzDest.y);
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
