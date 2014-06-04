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


INT nb_pts_in_box(Box2di side)
{
    return  (ElAbs(side._p1.x-side._p0.x)+1)
          * (ElAbs(side._p1.y-side._p0.y)+1);
}
static bool Debug=false;
/*************************************************/
/*                                               */
/*   Histo_Kieme                                 */
/*                                               */
/*************************************************/


Histo_Kieme::Histo_Kieme(INT max_vals) :
     _max_vals  (max_vals),
      mPopTot   (0)
{
}

Histo_Kieme::~Histo_Kieme() {}

void Histo_Kieme::verif_vals(const INT *vals,INT nb)
{
    INT index =    index_values_out_of_range(vals,nb,0,_max_vals);

    Tjs_El_User.ElAssert
    (
       index == INDEX_NOT_FOUND,
       EEM0 << "Out  of range in a buffered operator related to order\n"
            << "|   Expected values in [0 " << _max_vals << "[\n"
            << "|   got value " << vals[index] << "\n"
    );
}

void Histo_Kieme::add(INT radiom)
{
    AddPop(radiom,1);
}

void Histo_Kieme::sub(INT radiom)
{
    AddPop(radiom,-1);
}


INT Histo_Kieme::RKthVal(REAL aProp,INT adef)
{
    if (mPopTot <3) 
       return adef;

    ELISE_ASSERT((aProp>=0) && (aProp<1.0),"Out Range in Histo_Kieme::RKthVal\n");

    REAL aV = aProp * (mPopTot-1);
    INT aV0 = round_down(aV);
    REAL aP1 = aV-aV0;
    REAL aP0 = 1-aP1;
    INT aV1 = aV0+1;



    int aRes = round_ni ( aP0*kth_val(aV0) + aP1 * kth_val(aV1));
/*
    if (aRes > 50000)
    {
        Debug = true;
        cout << aP0 << " " << aV0 << " " << kth_val(aV0) << "\n";
        cout << aP1 << " " << aV1 << " " << kth_val(aV1) << "\n";
        cout <<  aProp << " " << mPopTot << "\n";
        getchar();
    }
*/
    return aRes;

}

//================================================================
//================================================================
/*
      BinTree_HK "fundamental equation"

      h[i] = h[2i] + h[2*i+1] 

      A BinTree_HK has always a power ot two nb vals;
      If it has 256 vals :
          * indexe between 256 <= i < 512 give the 
            number of pixel having i-256 has rediom
          *  indexe between 128 <= i < 512  are
             the sum of couple od radiom 256 <= i < 512
             (for example h[200] contains h[400]+h[401])
          * ....
*/

class BinTree_HK : public Histo_Kieme
{
   public :
      
      BinTree_HK(INT max_vals);
      virtual ~BinTree_HK();

   private :

       void   AddPop(INT radiom,INT aPop);
       virtual INT kth_val(INT kth);
       virtual INT rank(INT radiom);
       virtual void raz();

       INT * _h;
};

BinTree_HK::BinTree_HK(INT max_vals) :
     Histo_Kieme (Pow_of_2_sup(max_vals)),
     _h          (NEW_VECTEUR(0,2*_max_vals,INT))
{
   raz();
}


void BinTree_HK::AddPop(INT radiom,INT aPop)
{
   if (aPop)
   {
      mPopTot += aPop;
      for(radiom += _max_vals ; radiom ; radiom /= 2)
          _h[radiom]+= aPop;
   }
}



INT BinTree_HK::rank(INT radiom)
{
    radiom += _max_vals;
    INT res = _h[radiom];

    for( ; radiom >1 ; radiom /= 2)
       if (radiom%2) 
          res += _h[radiom-1];
    return res;
}

INT BinTree_HK::kth_val (INT kth)
{
   INT i;

   for ( i = 1 ;
         i < _max_vals;
       )
   {
      i *= 2;
      if (kth >= _h[i])
      {
          kth -= _h[i];
          i++;
      }
   }
   return(i - _max_vals); // _max_vals <= i < 2*_max_vals
}




void BinTree_HK::raz()
{
    mPopTot = 0;
    mem_raz(_h,2*_max_vals*sizeof(*_h));
}

BinTree_HK::~BinTree_HK()
{
     DELETE_VECTOR(_h,0);
}

//================================================================
//================================================================

class LastRank_HK : public Histo_Kieme
{
   public :
      
      LastRank_HK(INT max_vals);
      virtual ~LastRank_HK();

   private :


       void   AddPop(INT radiom,INT aPop);
       virtual void raz();
       virtual INT rank(INT radiom);
       virtual INT kth_val(INT kth);

       INT * _h;

       INT _last_val;
       INT _rank_last_val;
};

LastRank_HK::LastRank_HK(INT max_vals) :
    Histo_Kieme  (max_vals),
    _h           (NEW_VECTEUR(-1,max_vals,INT))
{
   raz();
}

LastRank_HK::~LastRank_HK()
{
    DELETE_VECTOR(_h,-1);
}


void LastRank_HK::AddPop(INT radiom,INT aPop)
{
   if (aPop)
   {
       mPopTot  += aPop;
       _h[radiom] += aPop;
       if (radiom<=_last_val) 
          _rank_last_val+=aPop;
   }
   
}

void LastRank_HK::raz()
{
    mPopTot = 0;
    _last_val = -1;
    _rank_last_val = 0;
    set_cste(_h-1,0,_max_vals+1);
}

INT LastRank_HK::rank(INT radiom)
{
    while (_last_val < radiom)
    {
          _last_val++;
          _rank_last_val += _h[_last_val];
    }
    while (_last_val > radiom)
    {
          _rank_last_val -= _h[_last_val];
          _last_val--;
    }

    return _rank_last_val;
}


INT LastRank_HK::kth_val(INT kth)
{
    if (Debug)
    {
          cout << "k " << kth
               << " Rk " << _rank_last_val
               << " Lv " << _last_val << "\n";
    }
    while ( _rank_last_val>kth)
    {
          _rank_last_val -= _h[_last_val];
          _last_val--;
    }

    while ( _rank_last_val<=kth)
    {
          _last_val++;
          _rank_last_val += _h[_last_val];
    }

    return _last_val;
}
 
//================================================================
//================================================================

Histo_Kieme * Histo_Kieme::New_HK(mode_h mode,INT max_vals)
{
     switch(mode)
     {
          case bin_tree :
               return new BinTree_HK (max_vals);

          case last_rank :
               return new LastRank_HK(max_vals);

          default :
          break;
     }

     elise_internal_error
     (
         "Illicit call to Histo_Kieme::New_HK",
          __FILE__,__LINE__
     );
     return 0;
}

Histo_Kieme::mode_h  Histo_Kieme::Opt_HK(INT ty,INT max_vals)
{
    REAL CostOpBT = 2.1;  // Piffometre
    REAL CostBinTree = El_logDeux(max_vals) * ty * 2 * CostOpBT;
    REAL CostLastRank = max_vals + 2*ty;

    // cout << "COST RANK " << CostBinTree << " " << CostLastRank << "\n";
    if (CostBinTree < CostLastRank)
       return bin_tree;
    else
       return last_rank;
}


Histo_Kieme::mode_h  Histo_Kieme::Opt_HK(mode_h aModePref,INT ty,INT max_vals)
{
   return (aModePref == undef) ? Opt_HK(ty,max_vals) : aModePref;
}

/***********************************************************************/
/***********************************************************************/

            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Kieme_Opb_Comp                 */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */

class Kieme_Opb_Comp : public Fonc_Num_OPB_TPL<INT>
{
         public :

               Kieme_Opb_Comp
               (
                    const Arg_Fonc_Num_Comp & arg,
                    INT                     dim_out,
                    Fonc_Num                f0,
                    Box2di                  box,
                    REAL                    kth,
                    Histo_Kieme::mode_h     mode,
                    INT                     max_vals,
                    Histo_Kieme::mode_res   mode_res,
                    bool                    aCatInit,
                    bool                    aModePond
               );
               virtual ~Kieme_Opb_Comp();

         private :

            virtual void post_new_line(bool);

            virtual void   verif_line(INT y);

           INT            mIKth;
           REAL           mRKth;
           Histo_Kieme *  _hk;
           Histo_Kieme::mode_res _mode_res;
           bool     mModePond;
           INT **    mLines;
           INT **    mPond;


           void Add(int x,int y,int aSgn)
           {
                _hk->AddPop
                (
                    mLines[y][x],
                    (mModePond ? mPond[y][x] * aSgn : aSgn)
                );
           }
};

Kieme_Opb_Comp::~Kieme_Opb_Comp()
{
     delete _hk;
}

Kieme_Opb_Comp::Kieme_Opb_Comp
(
       const Arg_Fonc_Num_Comp &    arg,
       INT                          dim_out,
       Fonc_Num                     f0,
       Box2di                       box,
       REAL                         kth,
       Histo_Kieme::mode_h          mode,
       INT                          max_vals,
       Histo_Kieme::mode_res        mode_res,
       bool                         aCatInit,
       bool                         aModePond
)  :

       Fonc_Num_OPB_TPL<INT>
       (
               arg,
               dim_out,
               Arg_FNOPB(f0,box),
               Arg_FNOPB::def,
               Arg_FNOPB::def,
               aCatInit
        ),
       mIKth        (round_ni(kth)),
       mRKth        (kth/nb_pts_in_box(box)),
       _hk          (Histo_Kieme::New_HK(mode,max_vals)),
       _mode_res    (mode_res),
       mModePond    (aModePond)
{
}


void    Kieme_Opb_Comp::verif_line(INT yloc)
{
        for (INT d=0; d<mDimOutSpec ; d++)
        {
            INT * line = kth_buf((INT *)0,0)[d][yloc];
            _hk->verif_vals(line+_x0_buf,_x1_buf-_x0_buf);
        }
}


void Kieme_Opb_Comp::post_new_line(bool first)
{
     if (first)
     {
         for (INT y = _y0_buf ; y < _y1_buf-1 ; y++)
             verif_line(y);
     }

     verif_line(_y1_buf-1);

     if (mModePond) 
        mPond =  kth_buf((INT *)0,0)[1];
     
     for (INT d =0 ; d<mDimOutSpec ; d++)
     {
          _hk->raz();
          mLines = kth_buf((INT *)0,0)[d];

          for (INT dy=_y0_side ; dy<=_y1_side ; dy++)
              for (INT dx=_x0_side ; dx<_x1_side ; dx++)
                  Add(_x0+dx,dy,1);
         
          for 
          (
               INT     x = _x0              , 
                       x_sub=_x0+_x0_side   ,
                       x_add=_x0+_x1_side   ;
               x < _x1 ; 
               x++,x_sub++,x_add++
          )
          {
               for (INT dy=_y0_side ; dy<=_y1_side ; dy++)
                   Add(x_add,dy,1);
               switch(_mode_res)
               {
                    case Histo_Kieme::KTH :
                        _buf_res[d][x] = mModePond ? _hk->RKthVal(mRKth,-1) : _hk->kth_val(mIKth);
                    break;

                    case Histo_Kieme::RANK :
                        _buf_res[d][x] = _hk->rank(mLines[0][x]);
                    break;
               }
	       {
                    for (INT dy=_y0_side ; dy<=_y1_side ; dy++)
                        Add(x_sub,dy,-1);
               }
          }
     }
}



            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */
            /*                                          */
            /*           Kieme_Opb_Not_Comp             */
            /*                                          */
            /* -  -  -  -  -  -  -  -  -  -  -  -  -  - */


class Kieme_Opb_Not_Comp : public Fonc_Num_Not_Comp
{
      public :
          Kieme_Opb_Not_Comp
          (
               Fonc_Num                    f0,
               REAL                        kth,
               Box2di                      side_0,
               INT                         max_vals,
               Histo_Kieme::mode_res       mode_res,
               Histo_Kieme::mode_h         aModeH,
               bool                        aCatInit,
               bool                        aModePond
          );

      private :

          virtual bool  integral_fonc (bool) const
          {
               return true;
          }

          virtual INT dimf_out() const 
          { 
                   if (mModePond) return 1 +  (mCatInit?1:0);

                   return (mCatInit?2:1)*_f.dimf_out();
          }
          void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}


          Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & arg);

          Fonc_Num                  _f;
          REAL                      _kth;
          Box2di                    _side;
          INT                       _max_vals;
          Histo_Kieme::mode_h       _mode;
          Histo_Kieme::mode_res     _mode_res;
          bool                      mCatInit;
          bool                      mModePond;
};

Kieme_Opb_Not_Comp::Kieme_Opb_Not_Comp
(
       Fonc_Num                    f0,
       REAL                        kth,
       Box2di                      side_0,
       INT                         max_vals,
       Histo_Kieme::mode_res       mode_res,
       Histo_Kieme::mode_h         aModeH,
       bool                        aCatInit,
       bool                        aModePond
)  :
   _f         (Iconv(f0)),
   _kth       (kth),
   _side      (side_0),
   _max_vals  (max_vals),
   _mode      (Histo_Kieme::Opt_HK(aModeH,side_0._p1.y-side_0._p0.y+1,max_vals)),
   _mode_res  (mode_res),
   mCatInit   (aCatInit),
   mModePond  (aModePond)
{
  if (aModePond)
  {
      ELISE_ASSERT
      (
         _f.dimf_out()==2,
         "Bad Dim for Kieme_Opb_Not_Comp in mode Pond"
      );
  }
}

Fonc_Num_Computed * Kieme_Opb_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{

    return new Kieme_Opb_Comp
               (
                   arg,
                   (mModePond ? 1 : _f.dimf_out()),
                   _f,
                   _side,
                   _kth,
                   _mode,
                   _max_vals,
                   _mode_res,
                   mCatInit,
                   mModePond
               );
         
}


Fonc_Num RectKth_Pondere
          (
               Fonc_Num    f1,
               Fonc_Num    f2,
               REAL         kth,
               INT          side,
               INT         max_vals
          )
{
  cout << "!!!!!!!!!!!!!!!!!!!!!   Non benche !!!!!!!!!!!!\n";
   ELISE_ASSERT
   (
      (f1.dimf_out()==1) && (f2.dimf_out() ==1),
      "Bad Dim in Rect_Pondere"
   ); 
   Pt2di PSz(side,side);

  return new Kieme_Opb_Not_Comp
             (
                 Virgule(f1,f2),
                 kth,
                 Box2di(-PSz,PSz),
                 max_vals,
                 Histo_Kieme::KTH,
                 Histo_Kieme::undef,
                 false,
                 true
             );
}


Fonc_Num  rect_kth
          (
               Fonc_Num    f,
               double         kth,
               Box2di      side,
               INT         max_vals,
               bool        aCatInit,
               bool        aModePond,
               Histo_Kieme::mode_h aModeH
          )
{
    kth = ElMax(0.0,ElMin(nb_pts_in_box(side)-1.0,kth));
    
    return new Kieme_Opb_Not_Comp
              (f,kth,side,max_vals,Histo_Kieme::KTH,aModeH,aCatInit,aModePond);
}

Fonc_Num  rect_kth
          (
              Fonc_Num f,
              double kth,
              Pt2di side,
              INT max_vals,
              bool aCatInit,
              bool aModePond,
              Histo_Kieme::mode_h aModeH
           )
{
          return  rect_kth(f,kth,Box2di(-side,side),max_vals,aCatInit,aModePond,aModeH);
}

Fonc_Num  rect_kth
          (
              Fonc_Num f,
              double kth,
              INT x,
              INT max_vals,
              bool aCatInit,
              bool aModePond,
              Histo_Kieme::mode_h aModeH
          )
{
          return  rect_kth(f,kth,Pt2di(x,x),max_vals,aCatInit,aModePond,aModeH);
}


Fonc_Num  rect_median 
          (
              Fonc_Num f,
              Box2di side,
              INT max_vals,
              bool aCatInit,
              bool aModePond,
              Histo_Kieme::mode_h aModeH
          )
{
    return 
         rect_kth
         (
             f,
             nb_pts_in_box(side)/2.0,
             side,
             max_vals,
             aCatInit,
             aModePond,
             aModeH
         );
}

Fonc_Num  rect_median 
          (
              Fonc_Num f,
              Pt2di p,
              INT max_vals,
              bool aCatInit,
              bool aModePond,
              Histo_Kieme::mode_h aModeH
          )
{
    return rect_median  (f,Box2di(-p,p),max_vals,aCatInit,aModePond,aModeH);
}

Fonc_Num  rect_median 
          (
             Fonc_Num f,
             INT  x,
             INT max_vals,
             bool aCatInit,
             bool aModePond,
             Histo_Kieme::mode_h aModeH
          )
{
    return rect_median  (f,Pt2di(x,x),max_vals,aCatInit,aModePond,aModeH);
}



Fonc_Num  rect_rank
          (
               Fonc_Num    f,
               Box2di      side,
               INT         max_vals,
               bool        aCatInit
          )
{
    return new Kieme_Opb_Not_Comp
               (
                  f,
                  -1,
                  side,
                  max_vals,
                  Histo_Kieme::RANK,
                  Histo_Kieme::undef,
                  aCatInit,
                  false
               );
}

Fonc_Num  rect_egal_histo
          (
               Fonc_Num    f,
               Box2di      side,
               INT         max_vals,
               bool        aCatInit
          )
{
    return   rect_rank(f,side,max_vals,aCatInit)
           * ( ((REAL) (max_vals -1)) / nb_pts_in_box(side));
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
