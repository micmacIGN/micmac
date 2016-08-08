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



/*****************************************************/
/*                                                   */
/*       ALGOS                                       */
/*                                                   */
/*****************************************************/



template class ElSom<int,int>;
template class ElArc<int,int>;
template class ElGraphe<int,int>;
template class ElSomIterator<int,int>;
template class ElArcIterator<int,int>;









static void test_alloc_tab_dyn(INT nb)
{
     All_Memo_counter MC_INIT;
     stow_memory_counter(MC_INIT);
     {
         ElGraphe<INT,INT> Gr;

          for (INT k =0; k<nb; k++)
              Gr.new_som(k);
     }
     verif_memory_state(MC_INIT);
}

static void test_alloc_tab_dyn()
{

     test_alloc_tab_dyn(0);
     test_alloc_tab_dyn(1);
     test_alloc_tab_dyn(10);
     test_alloc_tab_dyn(256);
     test_alloc_tab_dyn(512);
     test_alloc_tab_dyn(256*256);
     test_alloc_tab_dyn(256*256+1);
     test_alloc_tab_dyn(256*256-1);
     test_alloc_tab_dyn(1000);
     test_alloc_tab_dyn(200000);
}


class VerifIter
{
      public :
          static INT cple(INT v1,INT v2) 
          { 
                return  (v1 < v2) ? (1000*v1+v2) : (v1+1023*v2);
          }
          typedef ElSomIterator<INT,INT>  SIter;
          typedef ElArcIterator<INT,INT>  AIter;
          typedef ElSom<INT,INT>          TSom;
          typedef ElArc<INT,INT>          TArc;
          typedef ElSubGraphe<INT,INT>    SubGrAll;

          static  bool in_atsom(INT attr)  { return  attr %2  == 0;}
          static  bool in_atarc(INT attr)  { return  (attr %7)%3  == 0;}
          // static  bool in_atarc(INT attr)  { return  true;}

          class SSgrS2 : public SubGrAll
          {
               public :
                  bool inS(TSom & s)  { return  in_atsom(s.attr());}
                  bool inA(TArc & a)  { return  in_atarc(a.attr());}
          };

          TSom &  add_som(INT k) 
          {
              _s+= k;
              if (in_atsom(k)) _s2+= k;
              return _gr.new_som(k);
          }

          TArc &    add_arc(TSom & s1, TSom & s2)
          {
              INT a12 = cple(s1.attr(),s2.attr());
              INT a21 = cple(s2.attr(),s1.attr());
              if (in_atsom( s1.attr()) && in_atsom(s2.attr()))
              {
                  if (in_atarc(a12)) _sa2  += a12;
                  if (in_atarc(a21)) _sa2  += a21;
              }
              _sa += a12+a21;
              return  _gr.add_arc(s1,s2,a12,a21);
          }

          void   check_remove_arc(TArc & a)
          {
                TSom & s1 = a.s1();
                TSom & s2 = a.s2();
                INT a12 = cple(s1.attr(),s2.attr());
                INT a21 = cple(s2.attr(),s1.attr());
                if (in_atsom( s1.attr()) && in_atsom(s2.attr()))
                {
                    if (in_atarc(a12)) _sa2  -= a12;
                    if (in_atarc(a21)) _sa2  -= a21;
                }
                _sa -= a12+a21;
          }

          void   remove_arc(TArc & a)
          {
                check_remove_arc(a);
                a.remove();
          }

          void remove_som(TSom & s)
          {
              INT k = s.attr();
              _s-= k;
              if (in_atsom(k)) _s2-= k;
 
              SubGrAll  All;
              for (AIter it = s.begin(All); it.go_on() ; it++)
                  check_remove_arc(*it);
              s.remove();
          }
          

          void verif_som ()
          {
              SubGrAll  All;
              SIter End = _gr.end(All);
              INT s =0;

			  {
				 for (SIter  it = _gr.begin(All) ; it != End; it++)
				 {
					 s+= (*it).attr();
				}
			  }
              BENCH_ASSERT(s == _s);

              SSgrS2 Gr2;
              INT s2 = 0;
			  {
				 for (SIter it = _gr.begin(Gr2); it.go_on(); it++)
					 s2 += (*it).attr();
			  }
              BENCH_ASSERT(s2 == _s2);
          }

          void verif_arc ()
          {
              SubGrAll  All;
              SIter End = _gr.end(All);
              REAL sa =0;
              INT nb_12 = 0;
              INT nb_21 = 0;

			  {
				 for (SIter it = _gr.begin(All); it != End; it++)
				 {
					 TSom & s = (*it);
					 for (AIter ait = s.begin(All); ait.go_on(); ait++)
					 {
						  TArc & a = (*ait);
						  sa += a.attr();
						  INT n1 = a.s1().num();
						  INT n2 = a.s2().num();
						  BENCH_ASSERT(n1!=n2);
						  if (n1>n2) 
						    nb_12 ++; 
						 else 
						  nb_21++;
					}
				}
			  }
              BENCH_ASSERT(sa == _sa);
              BENCH_ASSERT(nb_12 == nb_21);

              SSgrS2 Gr2;
              REAL sa2 = 0.0;;
              for (SIter it = _gr.begin(Gr2); it != End; it++)
              {
                  TSom & s = (*it);
                  AIter AEnd = s.end(Gr2);
                  INT nb_succ =0;
                  for (AIter ait = s.begin(Gr2); ait != AEnd; ait++)
                  {
                      TArc & a = (*ait);
                      sa2 += a.attr();
                      nb_succ ++;
                  }
                  BENCH_ASSERT(nb_succ==s.nb_succ(Gr2));
              }
              BENCH_ASSERT(sa2 == _sa2);
           }
          

          VerifIter() : _s (0) , _s2(0), _sa(0.0) ,_sa2(0.0) {}
          ElGraphe<INT,INT> & gr () {return _gr;};
          void verif_alloc_arc();
          void verif_arc_s1s2();
          void bug();
          void bug2();
          void show();

      //PRIVATE :
      private : // __NEW
          ElGraphe<INT,INT> _gr;
          INT    _s;
          INT    _s2;
          REAL    _sa;
          REAL    _sa2;
};


void  VerifIter::verif_arc_s1s2()
{
     INT NBS = 50;
     INT NBV = 6;

     ElFilo<TSom *> ts;
     for (INT k=0; k<NBS ; k++)
     {
         TSom & s = add_som(k);
         ts.pushlast(&s);
     }

	 INT k1;
     for ( k1=0; k1<NBS ; k1++)
     {
          for (INT k2=k1+1; (k2<NBS) && (k2<(k1+NBV)) ; k2++)
              add_arc(*ts[k1],*ts[k2]);
     }

     for ( k1=0; k1<NBS ; k1++)
          for (INT k2= 0; k2<NBS ; k2++)
          {
              TArc * a= _gr.arc_s1s2(*ts[k1],*ts[k2]);
              BENCH_ASSERT
              (
                   (a==0) == ((k1==k2) || (ElAbs(k1-k2)>=NBV))
              );
              if (a)
                 BENCH_ASSERT
                 (
                          (a->s1().attr() == k1)
                       && (a->s2().attr() == k2)
                 );
          }
}

void VerifIter::bug()
{
  for (INT NBF =0; NBF < 30; NBF+= 1)
  {
     verif_som();
     verif_arc();
     ElFifo<TSom *> f;
	 INT k;
     for ( k = 0; k<NBF ; k++)
         f.pushlast(&add_som(k));
     verif_som();
     verif_arc();

	 INT k1;
     for ( k1=0; k1<NBF ; k1++)
         for (INT k2=k1+1; k2<NBF ; k2++)
             add_arc(*f[k1],*f[k2]);
     verif_som();
     verif_arc();

     for ( k1=0; k1<NBF ; k1++)
         for (INT k2=k1+1; k2<NBF ; k2++)
             remove_arc(*_gr.arc_s1s2(*f[k1],*f[k2]));

     for ( k = 0; k<NBF ; k++)
         remove_som(*f[k]);
     verif_som();
     verif_arc();
  }
}

void VerifIter::verif_alloc_arc()
{
  for (INT NBF =0; NBF < 4; NBF++)
  {
cout << "verif_alloc_arc \n";
     INT NBS = 1000;
     for (INT k = 0; k < NBS ; k++)
          add_som(k);

     SubGrAll  All;
     SIter End = _gr.end(All);
     for (SIter it1 = _gr.begin(All); it1 != End; it1++)
     {
          SIter it2 = it1;
          it2++;
          for (INT k=0 ;(it2 != End) && (k<4) ; it2++,k++)
          {
               add_arc(*it1,*it2);
          }
     }
     verif_arc();

     INT nbrem=0;
	 {
		for (SIter its=_gr.begin(All) ; its!=End ; its++)
		{
			TSom & s1 = *its;
			AIter AEnd = s1.end(All);
			for (AIter ita = s1.begin(All); ita != AEnd ; ita++)
			{
					TArc & a = (*ita);
					TSom & s2 = a.s2();
					if (s2.attr() == s1.attr()+1)
					{
						nbrem++;
						ita++;
						remove_arc(a);
					}
			  }
		 }
	 }
     verif_arc();

	 {
		 for (SIter its=_gr.begin(All) ; its.go_on() ; its++)
		 {
			  TSom & s = *its;
			  INT k = s.attr();
			  if (   ((k>=100) && (k<200))
				   || (NRrandom3() < 0.333)
				 )
				 remove_som(s);
		 }
	 }
     verif_som();
     verif_arc();

     for (SIter its=_gr.begin(All) ; its.go_on() ; its++)
     {
         remove_som(*its);
     }
     verif_som();
     verif_arc();
  }
}


static void test_gra_iter()
{
     VerifIter  vi; 
     for (INT k = 0; k < 10000 ; k++)
     {
           vi.add_som(k);
           if (k%1000 == 4) vi.verif_som();
     }
}

void VerifIter::show()
{
     SubGrAll  All;
     cout << "----------------------\n";
     for (SIter its = _gr.begin(All); its.go_on(); its++)
     {
         cout << "  " << (*its).attr() << " -> " ;
         for (AIter ita = (*its).begin(All); ita.go_on()  ; ita++)
             cout << "[" << (*ita).s1().attr()
                  << "," << (*ita).s2().attr() << "]";
         cout << "\n";
     }
     cout << "----------------------\n";
}

void VerifIter::bug2()
{
  INT K =0;
  for (INT NBF =0; NBF < 4; NBF++)
  {
     INT NBS = 3;
     for (INT k = 0; k < NBS ; k++)
         _gr.new_som(K++);

     SubGrAll  All;
     for (SIter it1 = _gr.begin(All); it1.go_on(); it1++)
     {
          SIter it2 = it1;
          it2++;
          for ( ;(it2.go_on() )  ; it2++)
          {
               _gr.add_arc(*it1,*it2,0);
          }
     }

     for (SIter its=_gr.begin(All) ; its.go_on() ; its++)
     {
          TSom & s1 = *its;
          while(s1.begin(All).go_on())
               (* s1.begin(All)).remove();
     }
  }
}



void bench_graphe_elem()
{
      {
           VerifIter vi;
           vi.bug2();
      }
      {
           VerifIter vi;
           vi.verif_alloc_arc();
      }
      {
           VerifIter vi;
           vi.verif_arc_s1s2();
      }
      {
           VerifIter vi;
           vi.bug();
      }
      test_gra_iter();
      test_alloc_tab_dyn();

      printf("OK GRAPHE ELEM \n");
}




