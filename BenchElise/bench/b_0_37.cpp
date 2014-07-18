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


#define  NoOperatorVirgule
#include "StdAfx.h"
#include "bench.h"

#include <map>

class cEl_BT2D
{
    public : 
       cEl_BT2D(REAL aVal) {Init(aVal);}
       cEl_BT2D()          {Init(NRrandom3());}
       cEl_BT2D(const cEl_BT2D & anEl)          {Init(anEl.mVal);}

       ~cEl_BT2D()          {theCpt--;}
       REAL Val() const {return mVal;}
       static INT Cpt() {return theCpt;}

    private :
        void Init(REAL aVal)
        {
             mVal = aVal;
             theCpt++;
        }

        REAL mVal;
        static INT theCpt;
};
int cEl_BT2D::theCpt = 0;


class cBenchEl_BT2D
{
     public :

        cBenchEl_BT2D(Pt2di aP0,Pt2di aP1) :
            mT2D(Box2di(aP0,aP1),true)
        {
        }

        void Add(Pt2di aP,const cEl_BT2D & anObj)
        {
               mT2D.Get(aP) = anObj;
               mMap[aP] = anObj;
        }

        void Resize(Pt2di aP0,Pt2di aP1)
        {
              mT2D.SetSize(Box2di(aP0,aP1));
              mMap.clear();
        }
         void Verif();
          

     private :
         cElBoxTab2DResizeableCreux<cEl_BT2D> mT2D;
         std::map<Pt2di,cEl_BT2D>             mMap;

};

void cBenchEl_BT2D::Verif()
{
    Pt2di aP0 = mT2D.P0() - Pt2di(5,5);
    Pt2di aP1 = mT2D.P1() + Pt2di(5,5);

    REAL Som1 = 0;
    for (INT y = aP0.y ; y<aP1.y ; y++)
    {
        for (INT x = aP0.x ; x<aP1.x ; x++)
        {
            Pt2di aP(x,y);
            bool  inTab = mT2D.inside(aP);
            bool inMap =  (mMap.find(aP)  != mMap.end());
            BENCH_ASSERT(inTab==inMap);
            if (inTab)
            {
                REAL vTab = mT2D.Get(aP).Val();
                REAL vMap = mMap[aP].Val();
                BENCH_ASSERT(ElAbs(vTab-vMap)<epsilon);
                Som1 += vMap;
            }
        }
    }

    REAL Som2 = 0;
    for 
    (
          tT2drCrNE::const_iterator it = mT2D.NonEmpty().begin();
          it != mT2D.NonEmpty().end();
          it++
    )
    {
         Som2 += mT2D.Get(*it).Val();
    }

    BENCH_ASSERT(ElAbs(Som1-Som2)<epsilon);
}




void     bench_tab2D_dyn()
{

    for (INT aK = 0 ; aK< 100 ; aK++)
    {
       {

          //Pt2di  aP0 = (Pt2dr(NRrandC(),NRrandC()) * 20);
          //Pt2di  aP1 = aP0+ (Pt2dr(NRrandom3(),NRrandom3()) * 20 + Pt2dr(2,2));
	  Pt2di  aP0(Pt2dr(NRrandC(),NRrandC()) * 20); // __NEW
	  Pt2di  aP1( Pt2dr(aP0) + (Pt2dr(NRrandom3(),NRrandom3()) * 20 + Pt2dr(2,2)) ); // __NEW

          cBenchEl_BT2D aB(aP0,aP1);

           for (INT aT=0 ; aT < 20 ; aT++)
           {

                Pt2di aSz = aP1-aP0;

                for ( INT aNb = ElMin(100,aSz.x*aSz.y); aNb>=0 ; aNb--)
                {
                      Pt2di aP =   aP0  
                                 + Pt2di
                                   (
                                       ElMin(aSz.x-1,INT(NRrandom3()*aSz.x)),
                                       ElMin(aSz.y-1,INT(NRrandom3()*aSz.y))
                                   );
                      
                      cEl_BT2D anEl(NRrandom3());
                      aB.Add(aP,anEl);
                }
                aB.Verif();

                
                //aP0 = (Pt2dr(NRrandC(),NRrandC()) * 20);
                //aP1 = aP0+ (Pt2dr(NRrandom3(),NRrandom3()) * 20 + Pt2dr(2,2));
                aP0 = Pt2di( Pt2dr(NRrandC(),NRrandC()) * 20); // __NEW
                aP1 = Pt2di( Pt2dr(aP0)+ (Pt2dr(NRrandom3(),NRrandom3()) * 20 + Pt2dr(2,2)) ); // __NEW
                aB.Resize(aP0,aP1);
           }
        }
    }
    
}

/*************************************************/

template <class T,const int NB>  class OneBench_ElPL
{
    private :

    public :
       OneBench_ElPL(ElPackList<T,NB> * res);

       ElPackList<T,NB> _l;
       INT              _i0;
       INT              _i1;

       void  pp(double proba_push = 0.5); // pop/push

       void verif();
};


template <class T,const int NB> 
         OneBench_ElPL<T,NB>::OneBench_ElPL(ElPackList<T,NB> * res) :
            _l   (res),
            _i0  ((INT) (2000 * (NRrandom3()-0.5))),
            _i1  (_i0)
{
}

template <class T,const int NB> 
         void OneBench_ElPL<T,NB>::verif()
{
    BENCH_ASSERT(_l.size() == (_i1-_i0));
    typename ElPackList<T,NB>::iterator it = _l.begin();
    typename ElPackList<T,NB>::iterator end = _l.end();

    INT i = _i0;

    for (; !(it == end); it++,i++)
    {
        BENCH_ASSERT(*it==i);
    }

    BENCH_ASSERT(i == _i1);
}

template <class T,const int NB> 
         void OneBench_ElPL<T,NB>::pp(double proba_push)
{
    if ((_l.size()) && (NRrandom3()>proba_push)) // pop
    {
        if (NRrandom3() > 0.5)
        {
           INT x = _l.pop_front();
           BENCH_ASSERT(x==_i0++);
        }
        else
        {
           INT x = _l.pop_back();
           BENCH_ASSERT(x==--_i1);
        }
    }
    else
    {
        if (NRrandom3() > 0.5)
        {
            _l.push_front(--_i0);
        }
        else
        {
            _l.push_back(_i1++);
        }
    }


    verif();
}




template <class T,const int NB>  class Bench_ElPL
{
    private :

    public :
       Bench_ElPL(bool with_res);

       ElPackList<T,NB>      _res;
       OneBench_ElPL<T,NB>   _l1;
       OneBench_ElPL<T,NB>   _l2;
};


template <class T,const int NB> 
          Bench_ElPL<T,NB>::Bench_ElPL(bool with_res) :
     _res (),
     _l1  (with_res ?  &_res : 0),
     _l2  (with_res ?  &_res : 0)
{
     for (int I =0 ; I< 10 ; I++)
        _l1.pp(0.9);
     for (int J =0 ; J< 10 ; J++)
        _l1.pp(0.1);
     for (int K =0 ; K< 100000 ; K++)
     {
        if (NRrandom3() > 0.5)
           _l1.pp(0.45);
        else
           _l2.pp(0.45);
     }
}

void bench_pack_list()
{
    for (int k=0; k<2 ; k++)
    {
        Bench_ElPL<INT,1> bepl1(k==0);
        Bench_ElPL<INT,2> bepl2(k==0);
        Bench_ElPL<INT,3> bepl3(k==0);
    }
}


class Compare_xpy
{
   public :
      bool operator () (Pt2di p1,Pt2di p2)
      {
           return (p1.x+p1.y) < (p2.x+p2.y);
      }
};


void bench_heap()
{
     Pt2di sz(20,50);
     Im2D_INT4 I1(sz.x,sz.y,0);
     Im2D_INT4 I2(sz.x,sz.y,0);

     INT4 ** i1 = I1.data();
     INT4 ** i2 = I2.data();

    Compare_xpy cmp = Compare_xpy();
     ElHeap<Pt2di,Compare_xpy> H(cmp);

	 INT k;
     for ( k=0; k<30000; k++)
     {
          Pt2di p = Pt2di((INT)(sz.x*NRrandom3()),(INT)(sz.y*NRrandom3()));
          p = Inf(p,sz);
          i1[p.y][p.x] ++;
          H.push(p);
     }

     Pt2di last;
     k =0;
     Pt2di p;
     while(H.pop(p))
     {
         if (k) 
            BENCH_ASSERT((last.x+last.y)<=(p.x+p.y));
         i2[p.y][p.x] ++;
         k++;
     }
     INT mdif;
     ELISE_COPY(I1.all_pts(),Abs(I1.in()-I2.in()),VMax(mdif));
     BENCH_ASSERT(mdif==0);

}


#define NB 3
void bench_borned_integer_heap()
{
     INT max_dif = 10;
     ElBornedIntegerHeap<Pt2di,NB> H2(max_dif+1);

     Compare_xpy cmp = Compare_xpy();
     ElHeap<Pt2di,Compare_xpy> H1(cmp);

     INT u = (INT) (10000 * (NRrandom3()-0.5));
     Pt2di p (u-10,10);
     H1.push(p);
     H2.push(p,u);

     for (INT nb =0; nb<10000; nb++)
     {
         BENCH_ASSERT(H1.nb() == H2.nb());

          Pt2di p1;
          Pt2di p2;
          INT Ind2=-1234567;
          BENCH_ASSERT(H2.pop(p2,Ind2));
          BENCH_ASSERT(H1.pop(p1));


          BENCH_ASSERT(p1==p2);
          BENCH_ASSERT(Ind2 == (p2.x + p2.y));

          INT nb_vois = ElMin(9*max_dif-H1.nb(),1+(INT)(3*NRrandom3()));

          for (INT v=0 ; v<nb_vois ; v++)
          {
               u = (INT) (max_dif * NRrandom3()) + Ind2;
               Pt2di NewP (u-10,10);
               H1.push(NewP);
               H2.push(NewP,u);
          }
     }


     Pt2di p1;
     Pt2di p2;
     INT Ind2;
     while (H1.pop(p1) && H2.pop(p2,Ind2))
     {
          BENCH_ASSERT(H2.pop(p2,Ind2));
          BENCH_ASSERT(H1.pop(p1));
          BENCH_ASSERT(p1==p2);
          BENCH_ASSERT(Ind2 == (p2.x + p2.y));
     }
     BENCH_ASSERT((H1.nb()==0) && (H2.nb()==0));
}


void bench_ext_stl_0()
{
     bench_heap();
     bench_tab2D_dyn();
     bench_borned_integer_heap();
     bench_pack_list();
     printf("OK bench_ext_stl_0\n");
}











