#ifndef DEF_REQUETE
#define DEF_REQUETE

#include "Point.h"
 
template <class Obj> class TypeFPRIM
{
	public :
		Pt2dr   operator() (const Obj & tq) const;
};
template <class Obj> class TypeFPRIMHomol
{
	public : Pt2dr   operator() (const Obj & tq) const;
};


template <class TF, class Obj> class   BenchQdt
{
      public :

         BenchQdt(TF Pt_of_Obj, Box2dr BOX,INT NBOBJMAX,REAL SZMIN);
         bool insert(const Obj p, bool svp=false);
         void remove(const Obj  & p);
	 void clear();
         void voisins(Pt2dr p0,REAL ray, ElSTDNS set<Obj>& S0);
	 void KPPVois(Pt2dr p0, std::list<Obj>& S0 ,int aNb, double aDistInit, double aFact, int aNbMax);

         const Box2dr                    Box;
         const INT                       NbObj;
         const REAL                      SZmin;

      private :
         ElQT<Obj,Pt2dr,TF>      qdt;
};

#include "Requete_def.h"

#endif

