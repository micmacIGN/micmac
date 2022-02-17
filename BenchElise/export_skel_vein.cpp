/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/



/*

Copyright (C) 1998 Marc PIERROT DESEILLIGNY

   Skeletonization by veinerization. 

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28


   Detail of the algoprithm in Deseilligny-Stamon-Suen
   "Veinerization : a New Shape Descriptor for Flexible
    Skeletonization" in IEEE-PAMI Vol 20 Number 5, pp 505-521

    It also give the signification of main parameters.
*/



#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <limits>
#include <string>
#include <new>
#include <assert.h>
#include <float.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "export_skel_vein.h"




template <class Type> class SkVFifo;
template <class Type> class SkVeinIm;


class SkVein
{

    public :

        typedef int            INT4;
        typedef unsigned char  U_INT1;
        typedef unsigned short U_INT2;
        typedef signed char    INT1;


        static void toto();

    // protected : 

        SkVein();
        static inline int abs(int v){ return (v<0) ? -v : v;}
        static inline int min(int v1,int v2) {return (v1<v2) ? v1 : v2;}
        static inline int max(int v1,int v2) {return (v1>v2) ? v1 : v2;}
        static inline void set_min(int &v1,int v2) {if (v1 > v2) v1 = v2;}
        static inline int square(int x) {return x*x;}

        enum
        {
           sz_brd = 3
        };


       enum
       {
           pds_x = 3,
           pds_y = pds_x * 3,
           pds_tr_x = pds_y * 3,
           pds_tr_y = pds_tr_x * 7,
           pds_moy  = pds_tr_y * 7,
           pds_cent  = pds_moy * 29, 
           pds_Mtrx = pds_moy + pds_tr_x,
           pds_Mtry = pds_moy + pds_tr_y,
           fact_corr_diag = pds_cent / 3,
           pds_out = -1000
       };


        // Data structure to have list of direction
        class ldir
        {
           public :
             ldir (ldir * next,int dir);

             ldir * _next;
             int     _dir;
             int     _x;
             int     _y;
             int     _flag_dsym;
             int     _flag;
        };



        // for a given flag, gives the list of active bits
        static ldir * (BitsOfFlag[256]);

        // for a given flag,  the number of bits active
        static U_INT1    NbBitFlag[256];
        // for a given flag,  =1 if NbBitFlag == 1 and 0 elsewhere
        static U_INT1    FlagIsSing[256];

        static U_INT1    UniqueBits[256];

        class cc_vflag
        {
           public :
              ldir       * _dirs;
              cc_vflag * _next;
              cc_vflag(cc_vflag * next,int flag);

              static cc_vflag * lcc_of_flag(int & flag,bool cx8,cc_vflag *);
        };


        static cc_vflag * (CC_cx8[256]);
        static cc_vflag * (CC_cx4[256]);

        static void flag_cc(int & cc,int b,int flag);


        static int v8x[8];
        static int v8y[8];

        static int v4x[4];
        static int v4y[4];

        static U_INT1 flag_dsym[8];
        static U_INT1 dir_sym[8];
        static U_INT1 dir_suc[8];
        static U_INT1 dir_pred[8];

        class Pt
        {
            public :
                bool operator != (const Pt & p)
                {
                     return (x != p.x) || (y != p.y);
                }

                Pt(INT4 X,INT4 Y) : x ((U_INT2) X), y ((U_INT2) Y) {}
                Pt(): x(0),y(0) {}
                U_INT2 x,y;
        };

        class Pt3
        {
            public :
                Pt3(INT4 X,INT4 Y,INT4 Z) : x ((U_INT2) X), y ((U_INT2) Y) , z 
(Z) {}
                Pt3(): x(0),y(0), z(0) {}
                U_INT2 x,y;
                INT4   z;
        };


// Bench purpose
        static void show(ldir *);
        static void show(cc_vflag *);
        static void showcc(int i,bool cx8);

    private : 
        static bool _init;
};



template <class Type> class SkVeinIm : public SkVein
{

         static const int max_val;
         static const int min_val;
     public :

         SkVeinIm(Type ** d,int tx,int ty);
         ~SkVeinIm();

          void close_sym();
          void open_sym();
          void set_brd(Type val,int width);
          void binarize(Type val);
          void d32(int vmax = max_val-1);
          void pdsd32vein1l(SkVein::INT4 * pds,SkVein::INT4 y);
          void d32_veinerize (SkVeinIm<U_INT1>  veines,bool cx8);
          void reset();
          void perm_circ_lines();

          void push_extr(SkVFifo<Pt> &);
          void push_extr(SkVFifo<Pt> &,U_INT2 ** som);

          void get_arc_sym(SkVFifo<Pt> &);
          bool get_arc_sym(INT4 x,INT4 y);
          void  prolong_extre();
          Pt  prolong_extre(int x,int y,int k);
          void  prolong_extre_rec(int x,int y,int k,int & nb,SkVFifo<Pt3> &f);

          void  pruning
          (
                  SkVeinIm<U_INT1>  veines,
                  U_INT2 **         som,
                  double            ang, 
                  INT4              surf,
                  bool              skp,
                  SkVFifo<Pt3> &    Fgermes
          );

          void  isolated_points
          (
                  SkVeinIm<U_INT1>  veines,
                  SkVFifo<Pt3> &    Fgermes
          );

        int  calc_surf(Pt );


     // private :

         Type **           _d;
         SkVein::INT4     _tx;
         SkVein::INT4     _ty;
         bool           _free;

        // to pass as paremeter in calc_surf
        double _ang;
        INT4   _surf;
        U_INT1  ** _dvein;
};

template <class Type> class SkVFifo : public SkVein
{
     public :
        SkVFifo(int capa);
        bool   empty() const {return _nb == 0;}
        int    nb () { return _nb;}
        Type & operator [] (int k) 
        { 
             assert ((k>=0)&&(k<_nb));
             return _tab[(k+_begin)%_capa];
        }
        inline Type     getfirst();
        inline void   pushlast(Type);
        ~SkVFifo();



      // private :
        void incr_capa();
        Type * _tab;
        int  _nb;
        int   _capa;
        int   _begin;

        SkVFifo(const SkVFifo<Type>&);

};


/***************************************************************************/
/*                                                                         */
/*         SkVFifo                                                         */
/*                                                                         */
/***************************************************************************/

template <class Type> SkVFifo<Type>::SkVFifo(int capa)  :
      _tab   (new Type [capa]) ,
      _nb    (0),
      _capa  (capa),
      _begin (0)
{
};

/*
template <class Type> SkVFifo<Type>::SkVFifo(const SkVFifo<Type>& f) :
      _tab   (new Type [f._capa]) ,
      _nb    (f._nb),
      _capa  (f._capa),
      _begin (f._begin)
{
    memcpy(_tab,f._tab,_capa*sizeof(Type));
}
*/


template <class Type> Type SkVFifo<Type>::getfirst()
{
    assert(_nb !=0);
    Type res = _tab[_begin];
    _begin = (_begin+1)%_capa;
    _nb--;
    return res;
}


template <class Type> void SkVFifo<Type>::incr_capa()
{
   Type * newtab = new Type [2*_capa];
   int nb_at_end = _capa-_begin;

   memcpy(newtab,_tab+_begin,nb_at_end*sizeof(Type));
   memcpy(newtab+nb_at_end,_tab,_begin*sizeof(Type));

   delete [] _tab;
   _tab = newtab;
   _begin = 0;
   _capa *= 2;
}

template <class Type> void SkVFifo<Type>::pushlast(Type p)
{
   if (_nb == _capa) incr_capa();
   _tab[(_begin+_nb++)%_capa] = p;
}


template <class Type> SkVFifo<Type>::~SkVFifo()
{
   delete [] _tab;
}



/***************************************************************************/
/*                                                                         */
/*         SkVein                                                          */
/*                                                                         */
/***************************************************************************/

int SkVein::v8x[8] = { 1, 1, 0,-1,-1,-1, 0, 1};
int SkVein::v8y[8] = { 0, 1, 1, 1, 0,-1,-1,-1};

int SkVein::v4x[4] = { 1, 0,-1, 0};
int SkVein::v4y[4] = { 0, 1, 0,-1};

SkVein::U_INT1 SkVein::dir_sym[8];
SkVein::U_INT1 SkVein::dir_suc[8];
SkVein::U_INT1 SkVein::dir_pred[8];
SkVein::U_INT1 SkVein::flag_dsym[8];

SkVein::ldir * (SkVein::BitsOfFlag[256]);
SkVein::cc_vflag * (SkVein::CC_cx8[256]);
SkVein::cc_vflag * (SkVein::CC_cx4[256]);

SkVein::U_INT1 SkVein::NbBitFlag[256];
SkVein::U_INT1 SkVein::FlagIsSing[256];
SkVein::U_INT1 SkVein::UniqueBits[256];

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
       UniqueBits[1<<b] = (U_INT1) b;
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

const int SkVeinIm<SkVein::U_INT1>::min_val = 0;
const int SkVeinIm<SkVein::U_INT1>::max_val = 255;

const int SkVeinIm<SkVein::INT4>::min_val = -0x7fffffff;
const int SkVeinIm<SkVein::INT4>::max_val =  0x7fffffff;

template <class Type> 
        void SkVeinIm<Type>::close_sym()
{
    set_brd((Type)0,1);

    for (int y=0; y<_ty ; y++)
        for (int x=0; x<_tx ; x++)
            for (ldir * l = BitsOfFlag[_d[y][x]]; l; l=l->_next)
                 _d[y+l->_y][x+l->_x] |= (U_INT1)(l->_flag_dsym);
}

template <class Type> 
        void SkVeinIm<Type>::open_sym()
{
    set_brd((Type)0,1);

    for (int y=0; y<_ty ; y++)
        for (int x=0; x<_tx ; x++)
            for (ldir * l = BitsOfFlag[_d[y][x]]; l; l=l->_next)
                 if (! (_d[y+l->_y][x+l->_x]&l->_flag_dsym))
                    _d[y][x] ^=  (U_INT1)(l->_flag);
}




template <class Type> 
        void SkVeinIm<Type>::pdsd32vein1l
        (
             SkVein::INT4 * pds,
             SkVein::INT4 y
        )
{
      Type * lm1 = _d[y-1]; 
      Type * l   = _d[y]; 
      Type * lp1 = _d[y+1]; 

      for (INT4 x=0; x<_tx; x++)
      {
          if (l[x])
             pds[x] =  pds_x    *  x
                      + pds_y    *  y
                      + pds_Mtrx *  l[x+1]
                      + pds_Mtry *  lp1[x]
                      + pds_moy  *  (l[x-1] + lm1[x])
                      + pds_cent *  l[x];
          else
             pds[x] = pds_out;
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
             SkVeinIm<SkVein::U_INT1>  veines ,
             bool              cx8
        )
{
    set_brd((Type)0,1);

     cc_vflag ** tab_CC = cx8 ? &CC_cx8[0] : &CC_cx4[0];
     assert((veines._tx==_tx)&&(veines._ty==_ty));
     veines.reset();
     U_INT1 ** _dv = veines._d; 

     SkVeinIm<INT4> ipds(0,_tx,3);
     
     INT4 ** pds = ipds._d+1;
     pdsd32vein1l(pds[-1],1);
     pdsd32vein1l(pds[0],2);

     INT4 pk[8];

     for(INT4 y=2; y<_ty-2;y++)
     {
         pdsd32vein1l(pds[1],y+1);
         for (INT4 x = 2; x<_tx-2; x++)
         {
             if (pds[0][x] > 0)
             {
                INT4 flag = 0;
                INT4 vc = pds[0][x];
                for (INT4 k=0; k<8 ; k++)
                {
                    pk[k] =  pds[v8y[k]][x+v8x[k]]-vc;
                    if (pk[k]>0)
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

template <class Type> void SkVeinIm<Type>::d32(int vmax)
{
    assert((vmax<=max_val) && (vmax>=min_val));

    binarize((Type)vmax);
    set_brd((Type)0,2);

    {
       for (INT4 y =0; y< _ty; y++)
       {
           Type * l0 = _d[y];
           for ( INT4 x = 0; x< _tx ; x++)
              if (l0[x])
              {
                  Type * l = l0 +x;
                  int v = l[0]-2;
                  SkVein::set_min(v,l[-1]);

                  Type * lp = _d[y-1]+x;
                  SkVein::set_min(v,lp[0]);
                  v--;
                  SkVein::set_min(v,lp[1]);
                  SkVein::set_min(v,lp[-1]);

                  l[0] = (Type)(v+3);
              }
       }
    }
    {
       for (INT4 y =_ty-1; y>=0; y--)
       {
           Type * l0 = _d[y];
           for ( INT4 x =_tx-1 ; x>=0 ; x--)
              if (l0[x])
              {
                  Type * l = l0 +x;
                  int v = l[0]-2;
                  SkVein::set_min(v,l[1]);

                  Type * lp = _d[y+1]+x;
                  SkVein::set_min(v,lp[0]);
                  v--;
                  SkVein::set_min(v,lp[1]);
                  SkVein::set_min(v,lp[-1]);

                  l[0] = (Type)(v+3);
              }
       }
    }
}

template <class Type> void SkVeinIm<Type>::binarize(Type val)
{
    for (INT4 y =0; y<_ty ; y++)
    {
        Type * line = _d[y];
        for (INT4 x =0; x<_tx ; x++)
            if(line[x]) 
              line[x] = val;
   }
}

template <class Type> void SkVeinIm<Type>::set_brd(Type val,int width)
{

     width =  SkVein::max(width,0);
     width =  SkVein::min(width,(_tx+1)/2);
     width =  SkVein::min(width,(_ty+1)/2);
     
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
      SkVeinIm<Type>::push_extr(SkVFifo<SkVein::Pt>  &PF,SkVein::U_INT2 ** som)
{
   for (int y=0 ; y<_ty ; y++)
   {
       Type * l = _d[y];
       U_INT2 * s = som[y];
       for (int x=0 ; x<_tx ; x++)
       {
           if (FlagIsSing[l[x]])
              PF.pushlast(Pt(x,y));
            s[x] = (l[x] != 0);
       }
   }
}

template <class Type> void 
      SkVeinIm<Type>::push_extr(SkVFifo<SkVein::Pt> &PF)
{
   for (int y=0 ; y<_ty ; y++)
   {
       Type * l = _d[y];
       for (int x=0 ; x<_tx ; x++)
           if (FlagIsSing[l[x]])
              PF.pushlast(Pt(x,y));
   }
}


template <class Type> 
        SkVein::INT4  SkVeinIm<Type>::calc_surf(SkVein::Pt  p)
{
   bool acons = false;
   INT4 som   = 1;

   for (INT4 k=0 ; k<8 ; k++)
   {
        if (    (_dvein[p.y+v8y[k]][p.x+v8x[k]] & flag_dsym[k])
             && (! (_dvein[p.y][p.x] & (1<<k)))
           )
        {
           INT4 somv = calc_surf(Pt(p.x+v8x[k],p.y+v8y[k]));
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

template <class Type> bool  SkVeinIm<Type>::get_arc_sym(SkVein::INT4 
x,SkVein::INT4 y)
{
    for (ldir * l = BitsOfFlag[_d[y][x]]; l; l=l->_next)
        if( _d[y+l->_y][x+l->_x] & l->_flag_dsym)
          return true;

   return false;
}

template <class Type> void  SkVeinIm<Type>::get_arc_sym(SkVFifo< SkVein::Pt> & 
P)
{
    for (INT4 y=0; y<_ty ; y++)
        for (INT4 x=0; x<_tx ; x++)
            if (get_arc_sym(x,y))
              P.pushlast(Pt(x,y)); 
}

template <class Type> void   SkVeinIm<Type>::isolated_points
                           (
                             SkVeinIm<SkVein::U_INT1>  veines,
                             SkVFifo<SkVein::Pt3> &    FGermes
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
                 FGermes.pushlast(Pt3(x,y,0));    
           }
}



template <class Type> void   SkVeinIm<Type>::pruning
                           (
                             SkVeinIm<SkVein::U_INT1>  veines,
                             SkVein::U_INT2 **         som,
                             double                    ang, 
                             SkVein::INT4              surf,
                             bool                      skgermes,
                             SkVFifo<SkVein::Pt3> &    FGermes
                           )
{
   ang /= 8.0;
   SkVFifo<Pt>  FS(4*(_tx+_ty)); // for singleton

   // for centers of "empty" shades
   SkVFifo<Pt3>   FCent(_tx+_ty);

   if (som)
       veines.push_extr(FS,som);
   else
       veines.push_extr(FS);

   U_INT1 ** dv = veines._d;

   U_INT2 * sv=0;

   _ang  = ang;
   _surf = surf;
   _dvein = veines._d;

   while (! FS.empty())
   {
       Pt p = FS.getfirst();
       INT4 flag = dv[p.y][p.x];
       if (flag)
       {
           assert(FlagIsSing[flag]);

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
                          FS.pushlast(Pt(xv,yv));
                  break;

                  case 0 :
                       FCent.pushlast(Pt3(xv,yv,k));
                  break;
            }
        }
   }

   if (! som)
   {
        {
           SkVFifo<Pt> Cycles(4 *(_tx+_ty));
           veines.get_arc_sym(Cycles);
           while(! Cycles.empty())
                calc_surf(Cycles.getfirst());
        }

       for (int i=0 ; i<FCent.nb() ; i++)
       {
           Pt3 p = FCent[i];
           calc_surf(Pt(p.x,p.y));
       }
   }

   while (! FCent.empty())
   {
       Pt3 p = FCent.getfirst();
       if (! veines.get_arc_sym(p.x,p.y))
          FGermes.pushlast(p);
   }

   if (skgermes)
   {
       while (! FGermes.empty())
       {
           Pt3 p = FGermes.getfirst();
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



template <class Type>  void  SkVeinIm<Type>::prolong_extre ()
{
      SkVFifo<Pt> F(_tx+_ty);

      for (INT4 y = 0; y<_ty; y++)
           for (INT4 x = 0; x<_tx; x++)
               if(FlagIsSing[_d[y][x]])
               {
                   INT4  k =  UniqueBits[_d[y][x]];
                   if (_d[y+v8y[k]][x+v8x[k]] & flag_dsym[k])
                   {
                      F.pushlast (Pt(x,y));
                      Pt extre = prolong_extre(x,y,k);
                      F.pushlast (extre);
                   }
               }

     SkVFifo<Pt3> Chem(100);
     while (! F.empty())
     {
           Pt singl = F.getfirst();
           Pt extr  = F.getfirst();
           while (extr != singl)
           {
                INT4 k = UniqueBits[_d[extr.y][extr.x]];
                assert(k<8);
                extr =Pt(extr.x+v8x[k],extr.y+v8y[k]);
                Chem.pushlast(Pt3(extr.x,extr.y,k));
           }
           while(! Chem.empty())
           {
               Pt3 p = Chem.getfirst();
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
                                  SkVFifo<SkVein::Pt3> &f
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
        f.pushlast(Pt3(x,y,nb));

     nb++;
}


template <class Type>  SkVein::Pt SkVeinIm<Type>::prolong_extre(int x,int y,int 
k)
{
    int nb = 0;
    SkVFifo<Pt3> F(100);
    
    prolong_extre_rec(x,y,k,nb,F);

    INT4 dif_min = nb;
    nb /= 2;

    Pt res (x,y);
    while (! F.empty())
    {
        Pt3 p = F.getfirst();
        INT4 dif = abs(p.z-nb);
        if (dif < dif_min)
        {
            res = Pt(p.x,p.y);
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
     unsigned short **   tmp
     
)
{

    bool mode_veine = (surf_threshlod<0) &&(angular_threshlod<0);


    SkVeinIm<unsigned char> DistSV(image,tx,ty);
    SkVeinIm<unsigned char> VeinSV(result,tx,ty);


    DistSV.d32();
    DistSV.d32_veinerize(VeinSV,true);
    VeinSV.close_sym();

    SkVFifo<SkVein::Pt3>   FGermes(tx+ty);
    if (! mode_veine)
    {
       DistSV.pruning
       (
           VeinSV,
           tmp,
           angular_threshlod,
           surf_threshlod,
           skel_of_disk,
           FGermes
       );

       if (prolgt_extre)
           VeinSV.prolong_extre();
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
            SkVein::Pt3 p= FGermes[k];
            res.x[k] = p.x;
            res.y[k] = p.y;
       }
    }


    if (! mode_veine)
       VeinSV.open_sym();
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







