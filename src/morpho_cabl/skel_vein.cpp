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


SKVRLE::SKVRLE(INT x0,INT x1) :
    _x0 (x0),
    _x1(x1)
{}

SKVRLE::SKVRLE()  :
   _x0(0),
   _x1(0)
{
}

INT SKVRLE::x0() { return _x0;}
INT SKVRLE::x1() { return _x1;}
INT SKVRLE::nbx(){ return _x1-_x0;}




/***************************************************************************/
/*                                                                         */
/*         SkVein                                                          */
/*                                                                         */
/***************************************************************************/

int SkVein::v8x[8] = { 1, 1, 0,-1,-1,-1, 0, 1};
int SkVein::v8y[8] = { 0, 1, 1, 1, 0,-1,-1,-1};

int SkVein::v4x[4] = { 1, 0,-1, 0};
int SkVein::v4y[4] = { 0, 1, 0,-1};

U_INT1 SkVein::dir_sym[8];
U_INT1 SkVein::dir_suc[8];
U_INT1 SkVein::dir_pred[8];
U_INT1 SkVein::flag_dsym[8];

SkVein::ldir * SkVein::BitsOfFlag[256];
SkVein::cc_vflag * SkVein::CC_cx8[256];
SkVein::cc_vflag * SkVein::CC_cx4[256];

U_INT1 SkVein::NbBitFlag[256];
U_INT1 SkVein::FlagIsSing[256];
U_INT1 SkVein::UniqueBits[256];

bool SkVein::_init = false;


SkVein::ldir::ldir(ldir * next,int dir) :
     _next        (next),
     _dir         (dir) ,
     _x           (v8x[dir]),
     _y           (v8y[dir]),
     _flag_dsym   (flag_dsym[dir]),
     _flag        (1<<dir)
{}

/*
    Given a "flag" corresponding to a set of bits of
    V8+, given an initial bit "b",  put in cc the flag of the
    connected component  of "b".
*/

void SkVein::flag_cc(int & cc,int b,int flag)
{
   if (
          (cc & (1<<b))
       || (! (flag & (1<<b)))
      )
      return;
   cc |= (1<<b);
   flag_cc(cc,dir_suc[b],flag);
   flag_cc(cc,dir_pred[b],flag);
   if (! (b%2))
   {
        flag_cc(cc,dir_suc[dir_suc[b]],flag);
        flag_cc(cc,dir_pred[dir_pred[b]],flag);
   }
}


SkVein::cc_vflag::cc_vflag(cc_vflag * next,int flag) :
   _dirs (BitsOfFlag[flag]),
   _next (next)
{
}

SkVein::cc_vflag * SkVein::cc_vflag::lcc_of_flag
                   (
                         int & flag,
                         bool cx8,
                         cc_vflag * res
                   )
{

    for (int b=0 ; b<8 ; b++)
        if (
                 (flag & (1<<b))
             &&  ( cx8 ||  (! (b%2)))
           )
        {
           int cc = 0;
           flag_cc(cc,b,flag);
           flag &= ~cc;
           res = new cc_vflag(res,cc);
        }

    return res;
}


SkVein::SkVein()
{

    if (_init) return ;
    _init = true;

// Then, init all tabulations required by algorithm

    {   // Stupid "{}" to please FUCKING visual c++
        for (int b=0; b<8 ; b++)
        {
             dir_sym[b] = (U_INT1)((b+4)%8);
             flag_dsym[b] = (U_INT1)(1 << dir_sym[b]);
             dir_suc[b] = (U_INT1)((b+1)%8);
             dir_pred[b] = (U_INT1) ((b+7)%8);
        }
    }

    {   // Stupid "{}" to please FUCKING visual c++
       for (int f=0; f<256; f++)
       {
            UniqueBits[f] = 255;
            BitsOfFlag[f] = 0;
            NbBitFlag[f] = 0;
            for (int b=0; b<8; b++)
                if (f&(1<<b))
                {
                   BitsOfFlag[f] = new ldir (BitsOfFlag[f],b);
                   NbBitFlag[f]++;
                }
            FlagIsSing[f] = (NbBitFlag[f]==1);
       }
    }

    {
       for (int f=0; f<256; f++)
       {
            int flag = f;
            CC_cx4[f] = cc_vflag::lcc_of_flag(flag,false,0);
            CC_cx8[f] = cc_vflag::lcc_of_flag(flag,true,CC_cx4[f]);
       }
    }
    for (int b=0; b<8 ; b++)

#if (ELISE_windows & !ELISE_MinGW)
      UniqueBits[1i64<<b] = (U_INT1) b;
#else
      UniqueBits[1<<b] = (U_INT1) b;
#endif
}


               //============================
               // Just for verifiication
               //============================

void SkVein::show(ldir * l)
{
    cout << "(";
    for (int i=0; l; i++,l = l->_next)
    {
        if (i) cout << " ";
        cout << l->_dir;
    }
    cout << ")";
}

void SkVein::show(cc_vflag * l)
{
    cout << "(";
    for (int i=0; l; i++,l=l->_next)
    {
        if (i) cout << " ";
        show (l->_dirs);
    }
    cout << ")";
}

void SkVein::showcc(int i,bool cx8)
{
    cout << i << " ;";
    cout << (cx8 ? " [V8] ;" : " [V4] ;");
    for (int b =0; b<8; b++)
        cout << ((i &(1<<b)) ? "*" : ".");
    cout << " ; ";
    show(cx8 ? CC_cx8[i] : CC_cx4[i]);
    cout << "\n";
}


/***************************************************************************/
/*                                                                         */
/*         SkVeinIm<Type>                                                  */
/*                                                                         */
/***************************************************************************/

template <> const int SkVeinIm<U_INT1>::min_val = 0;
template <> const int SkVeinIm<U_INT1>::max_val = 255;

template <> const int SkVeinIm<INT4>::min_val = -0x7fffffff;
template <> const int SkVeinIm<INT4>::max_val =  0x7fffffff;

template <class Type>
        void SkVeinIm<Type>::close_sym(ElPartition<SKVRLE> & rle)
{
    set_brd((Type)0,1);

    INT y,x,x0,x1,ks;
    ElSubFilo <SKVRLE>  segs;

    for (y=0; y<_ty ; y++)
    {
        segs = rle[y];
        for (ks=0; ks < segs.nb() ; ks++)
        {
            x0 = segs[ks].x0();
            x1 = segs[ks].x1();
            for (x=x0; x<x1 ; x++)
                for (ldir * l = BitsOfFlag[_d[y][x]]; l; l=l->_next)
                     _d[y+l->_y][x+l->_x] |= (U_INT1)(l->_flag_dsym);
        }
    }
}

template <class Type>
        void SkVeinIm<Type>::open_sym(ElPartition<SKVRLE> & rle)
{
    set_brd((Type)0,1);

    INT y,x,x0,x1,ks;
    ElSubFilo <SKVRLE>  segs;


    for (y=0; y<_ty ; y++)
    {
        segs = rle[y];
        for (ks=0; ks < segs.nb() ; ks++)
        {
            x0 = segs[ks].x0();
            x1 = segs[ks].x1();
            for (x=x0; x<x1 ; x++)
                for (ldir * l = BitsOfFlag[_d[y][x]]; l; l=l->_next)
                     if (! (_d[y+l->_y][x+l->_x]&l->_flag_dsym))
                        _d[y][x] ^=  (U_INT1)(l->_flag);
        }
   }
}




template <class Type>
        void SkVeinIm<Type>::pdsd32vein1l
        (
             INT4 *                  pds,
             ElPartition<SKVRLE> &   rle,
             INT4                      y,
             INT4 *                 Dpout
        )
{
      Type * lm1 = _d[y-1];
      Type * l   = _d[y];
      Type * lp1 = _d[y+1];

      memcpy(pds,Dpout,_tx*sizeof(*pds));
      ElSubFilo<SKVRLE>  segs = rle[y];

      INT ks;
      INT x,x0,x1;
      for (ks = 0; ks<segs.nb(); ks++)
      {
           x0 = segs[ks].x0();
           x1 = segs[ks].x1();
           for (x=x0; x<x1 ; x++)
               pds[x] =  pds_x    *  x
                       + pds_y    *  y
                       + pds_Mtrx *  l[x+1]
                       + pds_Mtry *  lp1[x]
                       + pds_moy  *  (l[x-1] + lm1[x])
                       + pds_cent *  l[x];
      }
}

template <class Type> void
         SkVeinIm<Type>::perm_circ_lines()
{
    Type * l0 = _d[0];
    for (int y = 1; y<_ty ; y++)
        _d[y-1] = _d[y];
     _d[_ty-1] = l0;
}




template <class Type>
        void SkVeinIm<Type>::d32_veinerize
        (
             ElPartition<SKVRLE> &  rle,
             SkVeinIm<U_INT1>       veines ,
             bool                   cx8
        )
{
    set_brd((Type)0,1);

     cc_vflag ** tab_CC =  &CC_cx8[0];
     Tjs_El_User.ElAssert
     (
          (veines._tx==_tx)&&(veines._ty==_ty),
          EEM0 << "Incompatible sizes in d32_veinerize"
     );
     veines.reset();
     U_INT1 ** _dv = veines._d;

     SkVeinIm<INT4> ipds(0,_tx,3);
     Im1D<INT4,INT4> PdsOut(_tx,pds_out);
     INT4 * DPout = PdsOut.data();

     INT4 ** pds = ipds._d+1;
     pdsd32vein1l(pds[-1],rle,1,DPout);
     pdsd32vein1l(pds[ 0],rle,2,DPout);

     INT4 pk[8];

     for(INT4 y=2; y<_ty-2;y++)
     {
         pdsd32vein1l(pds[1],rle,y+1,DPout);
         ElSubFilo<SKVRLE>  segs = rle[y];
         INT ks;
         for (ks=0 ; ks <segs.nb() ; ks++)
         {
              INT x0 = segs[ks].x0();
              INT x1 = segs[ks].x1();
              INT x;
              for (x = x0; x< x1; x++)
              {
                     INT4 flag = 0;
                     INT4 vc = pds[0][x];
                     for (INT4 k=0; k<8 ; k++)
                     {
                         pk[k] =  pds[v8y[k]][x+v8x[k]]-vc;
                         if (
                                  (pk[k]>0)
                              && (      cx8
                                    ||  (! (k&1))
                                    ||  _d[y+v8y[dir_suc[k]]][x+v8x[dir_suc[k]]]
                                    ||  _d[y+v8y[dir_pred[k]]][x+v8x[dir_pred[k]]]
                                 )
                            )
                         {
                            if (k&1) pk[k] +=
                                       (_d[y][x]-_d[y+v8y[k]][x+v8x[k]])*fact_corr_diag;
                            flag |= (1<<k);
                         }
                     }
                     for (cc_vflag * cc = tab_CC[flag]; cc; cc = cc->_next)
                     {
                         int pmax = pds_out;
                         int k_max = -1;
                         for (ldir * l = cc->_dirs;l;l=l->_next)
                         {
                             if (pk[l->_dir] > pmax)
                             {
                                pmax = pk[l->_dir];
                                k_max = l->_dir;
                             }
                         }
                         _dv[y][x] |= (U_INT1)(1<<k_max);
                     }
              }
         }
         ipds.perm_circ_lines();
     }
}

template <class Type> void SkVeinIm<Type>::reset()
{
    for (int y=0; y<_ty ; y++)
        memset(_d[y],0,sizeof(Type)*_tx);
}

template <class Type> void SkVeinIm<Type>::dist32
         (
            ElPartition<SKVRLE> &   rle,
            int                     vmax
         )
{
    Tjs_El_User.ElAssert
    (
         (vmax<=max_val) && (vmax>=min_val),
          EEM0 << "bad val max in  SkVeinIm<Type>::d32"
    );

    set_brd((Type)0,2);
    binarize(rle,(Type)vmax);

    {
       INT y,ks,nx;
       Type *l,*lm1;

       for (y =0; y< _ty; y++)
       {
           ElSubFilo<SKVRLE>  segs = rle[y];
           for (ks=0; ks<segs.nb() ; ks++)
           {
                l = _d[y]+segs[ks].x0();
                lm1 = _d[y-1]+segs[ks].x0();

                for(nx = segs[ks].nbx() ; nx;   nx--,l++,lm1++)
                {
                    ElSetMin(*l,l  [-1]+2);
                    ElSetMin(*l,lm1[-1]+3);
                    ElSetMin(*l,lm1[ 0]+2);
                    ElSetMin(*l,lm1[ 1]+3);
                }

           }
       }
    }

    {
       INT y,ks,nx;
       Type *l,*lp1;

       for (y =_ty-1; y>=0 ; y--)
       {
           ElSubFilo<SKVRLE>  segs = rle[y];
           for (ks=segs.nb()-1; ks>=0 ; ks--)
           {
                l = _d[y]+segs[ks].x1()-1;
                lp1 = _d[y+1]+segs[ks].x1()-1;

                for(nx = segs[ks].nbx() ; nx;   nx--,l--,lp1--)
                {
                    ElSetMin(*l,l  [ 1]+2);
                    ElSetMin(*l,lp1[-1]+3);
                    ElSetMin(*l,lp1[ 0]+2);
                    ElSetMin(*l,lp1[ 1]+3);
                }

           }
       }
    }
}

template <class Type> void SkVeinIm<Type>::binarize
                           (
                                ElPartition<SKVRLE> &   rle,
                                Type val
                           )
{
    INT4 y,x0,x1;

    rle.clear();

    for (y =0; y<_ty ; y++)
    {
        Type * l = _d[y];

        x0 =0;
        while (x0 < _tx)
        {
             while((!l[x0]) && (x0 < _tx)) x0++;
             if (x0 < _tx)
             {
                 for (x1 = x0 ; l[x1] && (x1 < _tx); x1++)
                    l[x1] = val;
                 rle.add(SKVRLE(x0,x1));
                 x0 = x1;
             }
        }
        rle.close_cur();
   }
}

template <class Type> void SkVeinIm<Type>::set_brd(Type val,int width)
{

     width =  SkVein::ElMax(width,0);
     width =  SkVein::ElMin(width,(_tx+1)/2);
     width =  SkVein::ElMin(width,(_ty+1)/2);

    for (int w=0 ; w<width ; w++)
    {
        {
            int  tyMw  = _ty-w-1;
            for (int x = 0 ; x < _tx ; x++)
                _d[w][x] = _d[tyMw][x] = val;
        }
        {
            int  txMw  = _tx-w-1;
            for (int y=0 ; y<_ty ; y++)
                _d[y][w] = _d[y][txMw] = val;
        }
    }
}

template <class Type> SkVeinIm<Type>::SkVeinIm(Type ** d,int tx,int ty) :
        _d    (d),
        _tx   (tx),
        _ty   (ty),
        _free (false)
{
     if (!_d)
     {
        _free = true;
        _d = new Type * [_ty];
        for (int y=0; y<_ty ; y++)
            _d[y] = new Type [_tx];
     }
}

template <class Type> void
      SkVeinIm<Type>::push_extr(ElPartition<SKVRLE> & rle,ElFifo<Pt2di>  &PF,U_INT2 ** som)
{
   ElSubFilo<SKVRLE> segs;
   INT y,x,x0,x1,ks;
   U_INT2 * s;
   Type * l ;

   for (y=0 ; y<_ty ; y++)
   {
       segs = rle[y];
       s = som[y];
       memset(s,0,_tx * sizeof(*s));
       l = _d[y];

       for (ks =0 ; ks<segs.nb() ; ks++)
       {
            x0 = segs[ks].x0();
            x1 = segs[ks].x1();
            for (x=x0 ; x<x1 ; x++)
            {
                if (FlagIsSing[l[x]])
                   PF.pushlast(Pt2di(x,y));
                 s[x] = 1;
            }
       }
   }
}

template <class Type> void
      SkVeinIm<Type>::push_extr(ElPartition<SKVRLE> &     rle,ElFifo<Pt2di> &PF)
{
   ElSubFilo<SKVRLE> segs;
   INT y,x,x0,x1,ks;
   Type * l ;

   for (y=0 ; y<_ty ; y++)
   {
       segs = rle[y];
       l = _d[y];

       for (ks =0 ; ks<segs.nb() ; ks++)
       {
            x0 = segs[ks].x0();
            x1 = segs[ks].x1();
            for (x=x0 ; x<x1 ; x++)
            {
                if (FlagIsSing[l[x]])
                   PF.pushlast(Pt2di(x,y));
            }
       }
   }
}


template <class Type>
        INT4  SkVeinIm<Type>::calc_surf(Pt2di  p)
{
   bool acons = false;
   INT4 som   = 1;

   for (INT4 k=0 ; k<8 ; k++)
   {
        if (    (_dvein[p.y+v8y[k]][p.x+v8x[k]] & flag_dsym[k])
             && (! (_dvein[p.y][p.x] & (1<<k)))
           )
        {
           INT4 somv = calc_surf(Pt2di(p.x+v8x[k],p.y+v8y[k]));
           if (somv == -1)
           {
              _dvein[p.y][p.x] |= (U_INT1)(1<<k);
              acons = true;
           }
           else
              som += somv;
        }
   }

   if (acons)
      return -1;
   if (     ( som < _surf)
        || ((INT4)som < (INT4) (_ang * square(_d[p.y][p.x])))
      )
      return som;
   else

      return -1;
}

template <class Type> bool  SkVeinIm<Type>::get_arc_sym(INT4
x,INT4 y)
{
    for (ldir * l = BitsOfFlag[_d[y][x]]; l; l=l->_next)
        if( _d[y+l->_y][x+l->_x] & l->_flag_dsym)
          return true;

   return false;
}

template <class Type> void  SkVeinIm<Type>::get_arc_sym(ElFifo< Pt2di> & P)
{
    for (INT4 y=0; y<_ty ; y++)
        for (INT4 x=0; x<_tx ; x++)
            if (get_arc_sym(x,y))
              P.pushlast(Pt2di(x,y));
}

template <class Type> void   SkVeinIm<Type>::isolated_points
                           (
                             SkVeinIm<U_INT1>  veines,
                             ElFifo<Pt3di> &    FGermes
                           )
{
   U_INT1 ** dv  = veines._d;
   for (int y=0 ; y<_ty ; y++)
       for (int x=0 ; x<_tx ; x++)
           if (_d[y][x] && (! dv[y][x]))
           {
              bool got8 = false;
              for (int k8 =0; (k8<8) && (!got8); k8++)
                  if (_d[y+v8y[k8]][x+v8x[k8]])
                     got8 = true;
              if (! got8)
                 FGermes.pushlast(Pt3di(x,y,0));
           }
}



template <class Type> void   SkVeinIm<Type>::pruning
                           (
                             ElPartition<SKVRLE> &     rle,
                             SkVeinIm<U_INT1>          veines,
                             U_INT2 **                 som,
                             double                    ang,
                             INT4                      surf,
                             bool                      skgermes,
                             ElFifo<Pt3di> &           FGermes
                           )
{
   ang /= 8.0;
   ElFifo<Pt2di>  FS(4*(_tx+_ty)); // for singleton

   // for centers of "empty" shades
   ElFifo<Pt3di>   FCent(_tx+_ty);

   if (som)
       veines.push_extr(rle,FS,som);
   else
       veines.push_extr(rle,FS);

   U_INT1 ** dv = veines._d;

   U_INT2 * sv=0;

   _ang  = ang;
   _surf = surf;
   _dvein = veines._d;

   while (! FS.empty())
   {
       Pt2di p = FS.popfirst();
       INT4 flag = dv[p.y][p.x];
       if (flag)
       {
           ELISE_ASSERT
           (
               FlagIsSing[flag],
               "Bad Assertion in SkVeinIm<Type>::pruning"
           );

           INT4 k = UniqueBits[flag];
           INT4 xv = p.x + v8x[k];
           INT4 yv = p.y + v8y[k];
           if (som)
           {
               sv = som[yv]+xv;
               *sv += som[p.y][p.x];
           }

            U_INT1 & fv = dv[yv][xv] ;
            fv ^=  flag_dsym[k]; // Kill arc from xv,yv to p

            bool suppres = true;

            switch(NbBitFlag[fv])
            {
                 case 1 :
                      if (som)
                          suppres =    (*sv < surf)
                                    || (*sv < ang * square(_d[yv][xv]));
                       if (suppres)
                          FS.pushlast(Pt2di(xv,yv));
                  break;

                  case 0 :
                       FCent.pushlast(Pt3di(xv,yv,k));
                  break;
            }
        }
   }

   if (! som)
   {
        {
           ElFifo<Pt2di> Cycles(4 *(_tx+_ty));
           veines.get_arc_sym(Cycles);
           while(! Cycles.empty())
                calc_surf(Cycles.popfirst());
        }

       for (int i=0 ; i<FCent.nb() ; i++)
       {
           Pt3di p = FCent[i];
           calc_surf(Pt2di(p.x,p.y));
       }
   }

   while (! FCent.empty())
   {
       Pt3di p = FCent.popfirst();
       if (! veines.get_arc_sym(p.x,p.y))
          FGermes.pushlast(p);
   }

   if (skgermes)
   {
       while (! FGermes.empty())
       {
           Pt3di p = FGermes.popfirst();
           dv[p.y][p.x] |= flag_dsym[p.z];
       }
   }
}

template <class Type> SkVeinIm<Type>::~SkVeinIm()
{
    if (_free)
    {
         for (int y=_ty-1;  y>=0 ; y--)
         {
            delete [] _d[y];
         }
         delete [] _d;
    }
};



template <class Type>  void  SkVeinIm<Type>::prolong_extre (ElPartition<SKVRLE> & rle)
{
     ElFifo<Pt2di> F(_tx+_ty);

     INT y,x,x0,x1,ks;
     ElSubFilo <SKVRLE>  segs;

     for (y=0; y<_ty ; y++)
     {
          segs = rle[y];
          for (ks=0; ks < segs.nb() ; ks++)
          {
             x0 = segs[ks].x0();
             x1 = segs[ks].x1();
             for (x = x0; x< x1; x++)
                 if(FlagIsSing[_d[y][x]])
                 {
                     INT4  k =  UniqueBits[_d[y][x]];
                     if (_d[y+v8y[k]][x+v8x[k]] & flag_dsym[k])
                     {
                        F.pushlast (Pt2di(x,y));
                        Pt2di extre = prolong_extre(x,y,k);
                        F.pushlast (extre);
                     }
                 }
          }
     }

     ElFifo<Pt3di> Chem(100);
     while (! F.empty())
     {
           Pt2di singl = F.popfirst();
           Pt2di extr  = F.popfirst();
           while (extr != singl)
           {
                INT4 k = UniqueBits[_d[extr.y][extr.x]];
                Tjs_El_User.ElAssert
                (
                     k<8,
                     EEM0 << "Basd Assertion in SkVeinIm<Type>::prolong_extre"
                );
                extr =Pt2di(extr.x+v8x[k],extr.y+v8y[k]);
                Chem.pushlast(Pt3di(extr.x,extr.y,k));
           }
           while(! Chem.empty())
           {
               Pt3di p = Chem.popfirst();
               _d[p.y][p.x] |= flag_dsym[p.z];
           }
     }
}


template <class Type>  void SkVeinIm<Type>::prolong_extre_rec
                            (
                                  int x,
                                  int y,
                                  int kpere,
                                  int &nb,
                                  ElFifo<Pt3di> &f
                            )
{
     nb++;
     bool got4 = false;
     for (INT4 k4 =0; (!got4)&&(k4<4); k4++)
         if (_d[y+v4y[k4]][x+v4x[k4]]==0)
            got4 = true;

     bool dead_end = true;
     for (INT4 k8 =1; (k8<8); k8++)
     {
         int k = (k8+kpere) %8;
         if (_d[y+v8y[k]][x+v8x[k]] &flag_dsym[k])
         {
             dead_end = false;
             prolong_extre_rec
             (
                 x+v8x[k],
                 y+v8y[k],
                 dir_sym[k],
                 nb,
                 f
             );
         }
     }
     if (got4 && dead_end)
        f.pushlast(Pt3di(x,y,nb));

     nb++;
}


template <class Type>  Pt2di SkVeinIm<Type>::prolong_extre(int x,int y,int k)
{
    int nb = 0;
    ElFifo<Pt3di> F(100);

    prolong_extre_rec(x,y,k,nb,F);

    INT4 dif_min = nb;
    nb /= 2;

    Pt2di res (x,y);
    while (! F.empty())
    {
        Pt3di p = F.popfirst();
        INT4 dif = abs(p.z-nb);
        if (dif < dif_min)
        {
            res = Pt2di(p.x,p.y);
           dif_min = dif;
        }
    }
    return res;
}


ResultVeinSkel VeinerizationSkeleton
(
     unsigned char **    result,
     unsigned char **    image,
     int                 tx,
     int                 ty,
     int                 surf_threshlod,
     double              angular_threshlod,
     bool                skel_of_disk,
     bool                prolgt_extre,
     bool                with_result,
     bool                cx8,
     unsigned short **   tmp

)
{
    bool mode_veine = (surf_threshlod<0) &&(angular_threshlod<0);


    SkVeinIm<unsigned char> DistSV(image,tx,ty);
    SkVeinIm<unsigned char> VeinSV(result,tx,ty);
    ElPartition<SKVRLE>  rle;


    DistSV.dist32(rle);
    DistSV.d32_veinerize(rle,VeinSV,cx8);
    VeinSV.close_sym(rle);

    ElFifo<Pt3di>   FGermes(tx+ty);
    if (! mode_veine)
    {
       DistSV.pruning
       (
           rle,
           VeinSV,
           tmp,
           angular_threshlod,
           surf_threshlod,
           skel_of_disk,
           FGermes
       );

       if (prolgt_extre)
           VeinSV.prolong_extre(rle);
    }


    ResultVeinSkel res;
    res.x = 0;
    res.y = 0;
    res.nb = 0;

    if (with_result)
    {
       DistSV.isolated_points(VeinSV,FGermes);
       res.nb = FGermes.nb();
       res.x = new  unsigned short [res.nb];
       res.y = new  unsigned short [res.nb];
       for (int k=0; k< res.nb; k++)
       {
            Pt3di p= FGermes[k];
            res.x[k] = p.x;
            res.y[k] = p.y;
       }
    }


    if (! mode_veine)
       VeinSV.open_sym(rle);
    return res;
}

void freeResultVeinSkel(ResultVeinSkel * res)
{
     if (res->x)
     {
         delete [] res->y;
         delete [] res->x;
     }
     res->x=0;
     res->y=0;
     res->nb = 0;
}

const unsigned char * NbBitsOfFlag()
{
      return SkVein::NbBitFlag;
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
