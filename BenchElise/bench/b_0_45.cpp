#define NoTemplateOperatorVirgule
#define NoSimpleTemplateOperatorVirgule
#include "StdAfx.h"
#include <list>
#include <algorithm>



/**********************************************/
/*                                            */
/*           BENCH                            */
/*                                            */
/**********************************************/


static Pt2dr PtRand()
{
    return Pt2dr(NRrandom3(),NRrandom3());
}       

class  TQdt
{
     public :

          TQdt(Pt2dr pt) : _pt (pt) , _id (_ID++) {}
          TQdt() :  _id (-1) {}

          Pt2dr  _pt;
          INT   _id;


          bool operator < (const TQdt & p2) const
          {
               return _id < p2._id;
          }

          bool operator == (const TQdt & p2) const
          {
               return _id == p2._id;
          }

      private :
          static INT _ID;
};
INT TQdt::_ID = 0;

/*
typedef   Pt2dr (* TypeFPRIM)(const TQdt &);
Pt2dr   Pt_of_TQdt (const TQdt & tq) {return tq._pt;}
*/

class TypeFPRIM
{
	public :
		Pt2dr   operator() (const TQdt & tq) const {return tq._pt;}
};
TypeFPRIM Pt_of_TQdt;



class  STQdt
{
     public :

          STQdt(Pt2dr p1,Pt2dr p2) : _p1 (p1), _p2(p2) , _id (_ID++) {}
          STQdt() :  _id (-1) {}

          Pt2dr  _p1;
          Pt2dr  _p2;
          INT   _id;

          SegComp  seg() const {return  SegComp(_p1,_p2);}


          bool operator < (const STQdt & p2) const
          {
               return _id < p2._id;
          }

          bool operator == (const STQdt & p2) const
          {
               return _id == p2._id;
          }

      private :
          static INT _ID;
};
INT STQdt::_ID = 0;


/*
typedef   SegComp (* TypeSFPRIM)(const STQdt &);
SegComp   Seg_of_STQdt (const STQdt & tq) {return tq.seg();}
*/

class TypeSFPRIM
{
	public :
		SegComp   operator () (const STQdt & tq)  const {return tq.seg();}
};

TypeSFPRIM Seg_of_STQdt;




class   BenchQdt
{
      public :

         BenchQdt(Box2dr BOX,INT NBOBJMAX,REAL SZMIN) :
              Box   (BOX),
              NbObj (NBOBJMAX),
              qdt   (Pt_of_TQdt,Box,NBOBJMAX,SZMIN),
              Sqdt  (Seg_of_STQdt,Box,NBOBJMAX,SZMIN)
         {
         }
         
         void insert(const STQdt  & s,bool svp = false)
         {
             if  (
                        (euclid(s._p1,s._p2) > 1.0)
                     && (Sqdt.insert(s,svp))
                 )
                 SGSet.insert(s);
         }

         void insert(const TQdt  & p,bool svp = false)
         {
             if (qdt.insert(p,svp))
                 GSet.insert(p);
         }

         void remove(const TQdt  & p)
         {
              qdt.remove(p);
              GSet.erase(p);
         }
         void remove(const STQdt  & p)
         {
              Sqdt.remove(p);
              SGSet.erase(p);
         }



         void remove(REAL proba);
         void sremove(REAL proba);
         void verif(Pt2dr p0,REAL ray);
         void verif(Pt2dr p0,Pt2dr p1,REAL ray);
         void verif_box(Pt2dr p0,Pt2dr p1,REAL ray);

         void sverif(Pt2dr p0,REAL ray);
         void sverif(Pt2dr p0,Pt2dr p1,REAL ray);
         void sverif_box(Pt2dr p0,Pt2dr p1,REAL ray);
         const Box2dr                    Box;
         const INT                       NbObj;
         ElQT<TQdt,Pt2dr,TypeFPRIM>      qdt;
         ElSTDNS set<TQdt>                       GSet;

         ElQT<STQdt,SegComp,TypeSFPRIM>  Sqdt;
         ElSTDNS set<STQdt>                      SGSet;
        
};

void BenchQdt::remove(REAL proba)
{
     // ElSTDNS list<TQdt> l (GSet.begin(),GSet.end());
     ElSTDNS list<TQdt> l;
	 std::copy(GSet.begin(),GSet.end(),ElSTDNS back_inserter(l));

     for (ElSTDNS list<TQdt>::iterator it = l.begin(); it != l.end() ; it++)
         if (NRrandom3() < proba)
            remove(*it);
}


void BenchQdt::sremove(REAL proba)
{
     // ElSTDNS list<STQdt> l (SGSet.begin(),SGSet.end());
     ElSTDNS list<STQdt> l;
	 std::copy(SGSet.begin(),SGSet.end(),std::back_inserter(l));
     for (ElSTDNS list<STQdt>::iterator it = l.begin(); it != l.end() ; it++)
         if (NRrandom3() < proba)
            remove(*it);
}


void BenchQdt::verif(Pt2dr p0,REAL ray)
{
     ElSTDNS set<TQdt> SSup;
     ElSTDNS set<TQdt> SInf;
     ElSTDNS set<TQdt> S0;

     qdt.RVoisins(SSup,p0,ray+1e-5);
     qdt.RVoisins(SInf,p0,ray-1e-5);

     for (ElSTDNS set<TQdt>::iterator it =GSet.begin(); it!=GSet.end() ; it++)
         if (euclid(p0,(*it)._pt) <ray)
            S0.insert(*it);


     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(S0.begin(),S0.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),S0.begin(),S0.end()));


}

void BenchQdt::verif(Pt2dr p0,Pt2dr p1,REAL ray)
{
     if (euclid(p0,p1) < 1.0) 
        return;

     ElSTDNS set<TQdt> SSup;
     ElSTDNS set<TQdt> SInf;
     ElSTDNS set<TQdt> S0;
     SegComp s01 (p0,p1);

     qdt.RVoisins(SSup,s01,ray+1e-5);
     qdt.RVoisins(SInf,s01,ray-1e-5);

     for (ElSTDNS set<TQdt>::iterator it =GSet.begin(); it!=GSet.end() ; it++)
         if (sqrt(s01.square_dist_seg((*it)._pt)) <ray)
            S0.insert(*it);

     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(S0.begin(),S0.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),S0.begin(),S0.end()));

}

void BenchQdt::verif_box(Pt2dr p0,Pt2dr p1,REAL ray)
{
     if (euclid(p0,p1) < 1.0) 
        return;

     ElSTDNS set<TQdt> SSup;
     ElSTDNS set<TQdt> SInf;
     ElSTDNS set<TQdt> S0;
     Box2dr  box (p0,p1);

     qdt.RVoisins(SSup,box,ray+1e-5);
     qdt.RVoisins(SInf,box,ray-1e-5);

     for (ElSTDNS set<TQdt>::iterator it =GSet.begin(); it!=GSet.end() ; it++)
         if (sqrt(box.SquareDist((*it)._pt)) <ray)
            S0.insert(*it);

     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(S0.begin(),S0.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),S0.begin(),S0.end()));
}






void BenchQdt::sverif(Pt2dr p0,REAL ray)
{
     ElSTDNS set<STQdt> SSup;
     ElSTDNS set<STQdt> SInf;
     ElSTDNS set<STQdt> S0;

     Sqdt.RVoisins(SSup,p0,ray+1e-5);
     Sqdt.RVoisins(SInf,p0,ray-1e-5);
     REAL R2 = ElSquare(ray);

     for (ElSTDNS set<STQdt>::iterator it =SGSet.begin(); it!=SGSet.end() ; it++)
     {
         SegComp S((*it)._p1,(*it)._p2);
         if (S.square_dist_seg(p0) <R2)
            S0.insert(*it);
     }

     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(S0.begin(),S0.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),S0.begin(),S0.end()));

}

void BenchQdt::sverif(Pt2dr p0,Pt2dr p1,REAL ray)
{
     if (euclid(p0,p1) < 1.0) 
        return;
     ElSTDNS set<STQdt> SSup;
     ElSTDNS set<STQdt> SInf;
     ElSTDNS set<STQdt> S0;
     SegComp s01 (p0,p1);

     Sqdt.RVoisins(SSup,s01,ray+1);
     Sqdt.RVoisins(SInf,s01,ElMax(ray-1,0.0));

     for (ElSTDNS set<STQdt>::iterator it =SGSet.begin(); it!=SGSet.end() ; it++)
         if (sqrt(s01.square_dist(SegComp::seg,(*it).seg(),SegComp::seg)) <ray)
            S0.insert(*it);

     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(S0.begin(),S0.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),S0.begin(),S0.end()));

}

void BenchQdt::sverif_box(Pt2dr p0,Pt2dr p1,REAL ray)
{
     if (euclid(p0,p1) < 1.0) 
        return;
     ElSTDNS set<STQdt> SSup;
     ElSTDNS set<STQdt> SInf;
     ElSTDNS set<STQdt> S0;
     Box2dr  box (p0,p1);

     Sqdt.RVoisins(SSup,box,ray+1);
     Sqdt.RVoisins(SInf,box,ElMax(ray-1,0.0));

     for (ElSTDNS set<STQdt>::iterator it =SGSet.begin(); it!=SGSet.end() ; it++)
         if (sqrt(box.SquareDist((*it).seg())) <ray)
            S0.insert(*it);


     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(S0.begin(),S0.end(),SInf.begin(),SInf.end()));
     BENCH_ASSERT(ElSTDNS includes(SSup.begin(),SSup.end(),S0.begin(),S0.end()));

}







void BenchQdtPt()
{
    Pt2di P0(5,5), P1(512+5,512+5);
    Box2dr aBox(P0,P1);

    BenchQdt  bench(aBox, 5, 20.0);

    for (INT x = 10; x< 500 ; x+= 20)
        for (INT y = 10; y< 500 ; y+= 20)
        {
             TQdt Tq(Pt2dr(x,y));
             bench.insert(Tq);
        }

	{
		for (INT k=0 ; k< 985; k++)
		{
			bench.insert(TQdt(Pt2dr(100,100)));
		}
	}

   for (INT k= 0; k< 100 ; k++)
   {
        bench.insert
        (
            STQdt(PtRand()*500,PtRand()*500),
            true
        );
   }

   for (int K=0; K<200;K++)
   {
       for (int r=0; r< 10 ; r++)
           bench.insert(TQdt(PtRand() * 500),true);

       for (INT k= 0; k< 10 ; k++)
       {
            bench.insert
            (
                STQdt(PtRand()*500,PtRand()*500),
                true
            );
       }

       Pt2dr p1 = PtRand () * 500;
       REAL ray = ElSquare(NRrandom3 ()) * 500;
       Pt2dr p2 = PtRand () * 500;

       bench.verif(p1,ray);
       bench.verif(p1,p2,ray);
       bench.verif_box(p1,p2,ray);
       bench.sverif(p1,ray);
       bench.sverif(p1,p2,ray);
       bench.sverif_box(p1,p2,ray);

       bench.remove(0.1);
       bench.sremove(0.2);
   }
}


void bench_qt()
{
    BenchQdtPt();
    cout << "OK  bench_qt()\n";
}






