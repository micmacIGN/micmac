#include "StdAfx.h"
#include "bench.h"

#include <list>


static Pt2dr PtRand()
{
    return Pt2dr(NRrandom3(),NRrandom3());
}

static Pt2dr PtRand(Box2di box,REAL ampl)
{
      Pt2dr  pt = PtRand();
      pt = (pt * (2*ampl+1)) -Pt2dr(ampl,ampl);
      return Pt2dr(box._p0) + pt.mcbyc(Pt2dr(box.sz()));
}


/************************************************************/


void bench_dist_pt_box(Box2di box,INT nb)
{
    Box2di  bp1 (box._p0,box._p1+Pt2di(1,1));
    for (INT k=0 ; k< nb ; k++)
    {
          //Pt2di pt = PtRand(box,1.0);
          Pt2di pt( PtRand(box,1.0) ); // __NEW
          INT d2;

          ELISE_COPY
          (
                rectangle(box._p0,box._p1+Pt2di(1,1)),
                Square(FX-pt.x)+ Square(FY-pt.y),
                VMin(d2)
          );
          BENCH_ASSERT
          (
              ElAbs(d2- box.SquareDist(Pt2dr(pt))) < epsilon
          );

          //BENCH_ASSERT(bp1.Intersecte(pt) ==  (d2==0));
          BENCH_ASSERT(bp1.Intersecte(Pt2dr(pt)) ==  (d2==0)); // __NEW
    }
}

void bench_dist_pt_box()
{
     bench_dist_pt_box
     (
         Box2di(Pt2di(-10,-12),Pt2di(12,14)),1000
     );
}

REAL BovDistSegBox(const Box2di & b,const SegComp & s)
{
     if (
               b.Intersecte(s.p0())
          ||   b.Intersecte(s.p1())
        )
        return 0.0;
     Pt2di pt[4];
     b.Corners(pt);
     return ElMin4
            (
               //s.square_dist(SegComp::seg,SegComp(pt[0],pt[1]),SegComp::seg),
               //s.square_dist(SegComp::seg,SegComp(pt[1],pt[2]),SegComp::seg),
               //s.square_dist(SegComp::seg,SegComp(pt[2],pt[3]),SegComp::seg),
               //s.square_dist(SegComp::seg,SegComp(pt[3],pt[0]),SegComp::seg)
               s.square_dist(SegComp::seg,SegComp( Pt2dr(pt[0]), Pt2dr(pt[1]) ),SegComp::seg), // __NEW
               s.square_dist(SegComp::seg,SegComp( Pt2dr(pt[1]), Pt2dr(pt[2]) ),SegComp::seg), // __NEW
               s.square_dist(SegComp::seg,SegComp( Pt2dr(pt[2]), Pt2dr(pt[3]) ),SegComp::seg), // __NEW
               s.square_dist(SegComp::seg,SegComp( Pt2dr(pt[3]), Pt2dr(pt[0]) ),SegComp::seg)  // __NEW
            );
}

void bench_dist_seg_box(Box2di b,INT nb)
{
     Pt2di pt[4];
     b.Corners(pt);
     for (int kp=0; kp<4 ;  kp++)
         if (pt[kp] == pt[(kp+1)%4])
            return;


     for (INT k= 0; k < nb ; k++)
     {
         Pt2dr p1 = PtRand(b,2);
         Pt2dr p2 = PtRand(b,2);
         if (p1 != p2)
         {
              SegComp S(p1,p2);

              REAL d1 = BovDistSegBox(b,S);
              REAL d2 = b.SquareDist(S);
              BENCH_ASSERT(ElAbs(d1-d2)<epsilon);
         }
     }
}

void bench_dist_seg_box()
{
     bench_dist_seg_box
     (
          Box2di(Pt2di(-20,23),Pt2di(15,34)),
          10000
     );

     for (INT k=0; k < 50 ; k++)
         bench_dist_seg_box
         (
              Box2di
              (
                 Pt2di((PtRand()-Pt2dr(0.5,0.5)) * 50),
                 Pt2di((PtRand()-Pt2dr(0.5,0.5)) * 50)
              ),
              1000
         );
}

/************************************************************/
/************************************************************/
/************************************************************/


class CBenchElSList
{
     public :
           typedef ElSlist<INT>::iterator  ElIt;
           typedef ElSTDNS list<INT>::iterator     LIt;

           ElSlist<INT>      _l1;
           ElSTDNS list<INT>         _l2;
           ElSlist<INT> *    _res;
           INT               _sz;
           INT               _som;

           CBenchElSList( ElSlist<INT> * Reserve) :
               _res (Reserve),
               _sz  (0),
               _som (0)
           {
           }

           void add(INT v)
           {
                _l1.push_front(v,_res);
                _l2.push_front(v);
                _sz ++;
                _som += v;
           }

           void  remove(INT v)
           {
                _sz -= _l1.remove(v,_res);
                _l2.remove(v);
                _som -= v;
           }

           void clear() 
           {
                _sz = 0;
                _som = 0;
                _l1.clear();
                _l2.clear();
           }

           void verif();

};

void CBenchElSList::verif()
{
    ElIt it1 = _l1.begin();
    LIt it2  = _l2.begin();
    INT som = 0;
    INT sz  = 0;

    while((it1!=_l1.end()) &&  (it2!=_l2.end()))
    {
        BENCH_ASSERT(*it1 == *it2);
        som += *it1;
        sz ++;
        it1++;
        it2++;
    }
    BENCH_ASSERT((it1==_l1.end()) && (it2==_l2.end()));
    BENCH_ASSERT(_som == som);
    BENCH_ASSERT(_sz  == sz);
}

void bench_ElSList(ElSlist<INT> * res)
{
     CBenchElSList Check( res);
     Check.add(0);
     Check.add(0);
     Check.add(1);
}

/************************************************************/
/************************************************************/
/************************************************************/



void bench_ElSList()
{
     bench_ElSList(0);
     ElSlist<INT>   res;
     bench_ElSList(&res);
}


/************************************************************/
/************************************************************/
/************************************************************/

void bench_qt_support()
{
     bench_dist_seg_box();
     bench_ElSList();
     bench_dist_pt_box();
}

