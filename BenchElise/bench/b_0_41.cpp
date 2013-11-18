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

#include "StdAfx.h"
#include "bench.h"


// template class ElChainePcc<int,int>;
// template class CmpElChainePcc<int,int>;
// template class ElPcc<int,int>;

/*****************************************************/
/*                                                   */
/*       ALGOS                                       */
/*                                                   */
/*****************************************************/

class  SGR_41
{
     public :
        SGR_41(Pt2di pt) : _pt (pt) {}
        SGR_41()  {}
        Pt2di _pt;
     private  :
};



typedef  ElSom<SGR_41,int>                     TSom41;
typedef  ElArc<SGR_41,int>                     TArc41;
typedef  ElGraphe<SGR_41,int>                  TGr41;
typedef  ElSomIterator<SGR_41,int>             TSIt41;
typedef  ElArcIterator<SGR_41,int>             TAIt41;
typedef  ElPartition<TArc41 *>                 TPart41;
typedef  ElSubGraphe<SGR_41,int>               TSubGr41;
typedef  ElPcc<SGR_41,int>                     TPcc41;
typedef  ElEmptySubGrapheSom<SGR_41,int>       TEmpGr41;


class TSubGr41_xy2 : public TSubGr41
{
      //virtual Pt2dr  pt(TSom41 & s) { return s.attr()._pt;}
      virtual Pt2dr  pt(TSom41 & s) { return Pt2dr( s.attr()._pt );} // __NEW
      bool inS(TSom41 & s)
      {
          return 
                    (((s.attr()._pt.x) %2) == 0)
                 || (((s.attr()._pt.y) %2) == 0) ;
      }
};

class TSubGr41_xeg : public TSubGr41
{
      //virtual Pt2dr  pt(TSom41 & s) { return s.attr()._pt;}
      virtual Pt2dr  pt(TSom41 & s) { return Pt2dr( s.attr()._pt );} // __NEW
      bool inA(TArc41 & a)
      {
          return a.s1().attr()._pt.x == a.s2().attr()._pt.x;
      }
};

class TSubGr41_xeg_y0 : public TSubGr41
{
      //virtual Pt2dr  pt(TSom41 & s) { return s.attr()._pt;}
      virtual Pt2dr  pt(TSom41 & s) { return Pt2dr( s.attr()._pt );} // __NEW
      bool inA(TArc41 & a)
      {
          return 
                  (a.s1().attr()._pt.x == a.s2().attr()._pt.x)
              ||  (a.s1().attr()._pt.y == 0);
      }
};




class TPtSubGr41 : public TSubGr41
{
      //virtual Pt2dr  pt(TSom41 & s) { return s.attr()._pt;}
      virtual Pt2dr  pt(TSom41 & s) { return Pt2dr( s.attr()._pt );} // __NEW
};



#if ElTemplateInstantiation
template class ElSom<SGR_41,int>;
template class ElArc<SGR_41,int>;
#endif
template class ElGraphe<SGR_41,int>;
template class ElSomIterator<SGR_41,int>;
template class ElArcIterator<SGR_41,int>;


// template TArc41 * succ_trigo(TArc41 &,TSubGr41 &,bool);
template bool face_trigo(TArc41 &,TSubGr41 &,ElFilo<TArc41 *> &);
template bool all_face_trigo(TGr41&,TSubGr41&,ElPartition<TArc41 *> &);

//static Pt2dr PtArc41(TArc41 * a) {return a->s1().attr()._pt;}
static Pt2dr PtArc41(TArc41 * a) {return Pt2dr( a->s1().attr()._pt );} // __NEW

static const Pt2di SZ_PCC(10,10);

void bench_graphe_grille(Pt2di sz,TGr41 & gr)
{
       TPtSubGr41 GrAll;
       TPart41    Part;

       all_face_trigo
       (
            gr,
            (TSubGr41 &) GrAll,
            Part
       );

       ElFifo <Pt2dr> Fp2(2,true);

       BENCH_ASSERT(Part.nb() ==  ((sz.x-1)*(sz.y-1) +1));
       INT nb4 =0;
       INT perim_th = (sz.x-1 + sz.y-1) * 2;

       for (INT f=0; f< Part.nb() ; f++)
       {
           INT nbf = Part[f].nb();
           BENCH_ASSERT((nbf == 4) || (nbf==perim_th));

           copy_on(Fp2,Part[f],PtArc41);
           REAL surf = surf_or_poly(Fp2);
           Pt2dr cdg =  barrycentre(Fp2);

           if (nbf == 4)
           {
               nb4 ++;
               BENCH_ASSERT (ElAbs(ElAbs(surf)-1.0) < epsilon);
               REAL x = cdg.x - round_down(cdg.x);
               REAL y = cdg.y - round_down(cdg.y);
               BENCH_ASSERT
               (
                       (ElAbs(x-0.5) < epsilon)
                   &&  (ElAbs(y-0.5) < epsilon)
               );
           }
           else
           {
                BENCH_ASSERT (ElAbs(surf-(sz.x-1)*(sz.y-1))< epsilon);
                BENCH_ASSERT(euclid(cdg,(Pt2dr(sz)-Pt2dr(1,1))/2.0) < epsilon);
           }
       }
       BENCH_ASSERT(Part.nb() == nb4+1);

       TSubGr41_xy2 Gr_xy2;
       all_face_trigo
       (
            gr,
            (TSubGr41 &) Gr_xy2,
            Part
       );
       BENCH_ASSERT(Part.nb() ==  (((sz.x-1)/2)*((sz.y-1)/2) +1));


       TSubGr41_xeg    Gr_xeq;
       all_face_trigo
       (
            gr,
            (TSubGr41 &) Gr_xeq,
            Part
       );
       BENCH_ASSERT(Part.nb() ==  sz.x);


       TSubGr41_xeg_y0    Gr_xeq_y0;
       all_face_trigo
       (
            gr,
            (TSubGr41 &) Gr_xeq_y0,
            Part
       );
       BENCH_ASSERT(Part.nb() ==  1);

}

void bench_pcc(TGr41 & gr, TSom41 *** TABS,Pt2di sz)
{
     TPcc41                  PCC;
     TPtSubGr41              GrAll;
     TSubGr41_xeg            GrXeq;
     TSubGr41_xy2            Gr_xy2;
     TEmpGr41                GrEmpty;


     for (int k =0; k < 300  ; k ++)
     {
          INT x1 = ((INT)(NRrandom3() * 100)) % sz.x;
          INT y1 = ((INT)(NRrandom3() * 100)) % sz.y;
          INT x2 = (x1+13*k)%sz.x;
          INT y2 = (y1+3*k)%sz.y;
          Pt2di p1 (x1,y1);
          Pt2di p2 (x2,y2);
          TSom41 & s1  = *(TABS[y1][x1]);
          TSom41 & s2  = *(TABS[y2][x2]);
          
          //TSom41 * res = PCC.pcc(s1,s2,GrAll);
          TSom41 * res = PCC.pcc(s1,s2,GrAll,eModePCC_Somme); // __NEW
          BENCH_ASSERT(PCC.pds(s2)==dist4(p1-p2));

          ElFilo<TSom41 *> f41;
          PCC.chemin(f41,s2);
          BENCH_ASSERT(PCC.pds(s2)==f41.nb()-1);
          for (INT S=1 ; S<f41.nb() ; S++)
               BENCH_ASSERT(gr.arc_s1s2(*f41[S-1],*f41[S]));


          //res = PCC.pcc(s1,s2,GrXeq);
          res = PCC.pcc(s1,s2,GrXeq,eModePCC_Somme); // __NEW
          BENCH_ASSERT((res==&s2) == (x1==x2));

          //res = PCC.pcc(s1,GrEmpty,GrXeq);
          res = PCC.pcc(s1,GrEmpty,GrXeq,eModePCC_Somme); // __NEW
          BENCH_ASSERT(res==0);
          for 
          (
                TSIt41 sit = gr.begin(GrAll);
                sit.go_on();
                sit ++
          )
          {
              TSom41 & s = (*sit);
              BENCH_ASSERT(PCC.reached(s) == (s.attr()._pt.x == x1));
              if (PCC.reached(s))
              {
                  INT y = s.attr()._pt.y;
                  INT x = s.attr()._pt.x;
                  BENCH_ASSERT(PCC.pds(s) == ElAbs(y-y1));
                  if (y == y1)
                     BENCH_ASSERT(PCC.pere(s) == 0);
                  else if (y> y1)
                     BENCH_ASSERT(PCC.pere(s) == TABS[y-1][x]);
                  else
                     BENCH_ASSERT(PCC.pere(s) == TABS[y+1][x]);
              }
          }

          //res = PCC.pcc(s1,s2,Gr_xy2);
          res = PCC.pcc(s1,s2,Gr_xy2,eModePCC_Somme); // __NEW
          BENCH_ASSERT
          (
                (res==&s2) 
            ==  (
                       ((x1%2==0) || (y1%2==0))
                   &&  ((x2%2==0) || (y2%2==0))
                )
          );
          if (res==&s2)
             BENCH_ASSERT(PCC.pds(s2)==dist4(p1-p2));
     }
}

void bench_graphe_grille(Pt2di sz,INT nb=1,bool pcc = false)
{
      TGr41 gr;
      TSom41 *** TABS = NEW_MATRICE_ORI(sz.x,sz.y,TSom41 *);

	  INT x;
      for ( x =0; x<sz.x; x++)
          for (int y =0; y<sz.y; y++)
          {
             TSom41 & s = gr.new_som(SGR_41(Pt2di(x,y)));
             TABS[y][x] = & s;
          }

      for ( x =0; x<sz.x; x++)
          for (int y =0; y<sz.y; y++)
          {
              if (y+1 < sz.y)
                 gr.add_arc(*TABS[y][x],*TABS[y+1][x],10,10);
              if (x+1 < sz.x)
                 gr.add_arc(*TABS[y][x],*TABS[y][x+1],10,10);
          }

       for (INT k=0; k<nb ; k++)
            bench_graphe_grille(sz,gr);

       if (pcc)
       {
           bench_pcc(gr,TABS,sz);
       }

       
       DELETE_MATRICE_ORI(TABS,sz.x,sz.y);
}


void bench_graphe_grille()
{
    bench_graphe_grille(SZ_PCC,1,true);

    bench_graphe_grille(Pt2di(3,3),100);
    bench_graphe_grille(Pt2di(9,7));
    bench_graphe_grille(Pt2di(12,21));

    for (INT k = 0; k<50 ; k++)
        bench_graphe_grille
        (
              Pt2di(3,3)
	      +Pt2di
               (
		  Pt2dr
		  (
                     NRrandom3()*10.0,
                     NRrandom3()*10.0
		  )
	       )
	);
}



void bench_algo_graphe_0()
{
     bench_graphe_grille();
     printf("OK algo graphe 000000 \n");
}
