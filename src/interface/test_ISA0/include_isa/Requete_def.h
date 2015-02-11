
template <class Obj> Pt2dr TypeFPRIM<Obj>::operator() (const Obj & tq) const {return tq.GetPt2dr();}
template <class Obj> Pt2dr TypeFPRIMHomol<Obj>::operator() (const Obj & tq) const {return tq.GetPt2drHomol();}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////:

template <class TF, class Obj> BenchQdt<TF,Obj>::BenchQdt(TF Pt_of_Obj, Box2dr BOX,INT NBOBJMAX,REAL SZMIN) :
              Box   (BOX), NbObj (NBOBJMAX), SZmin(SZMIN), qdt (Pt_of_Obj,Box,NBOBJMAX,SZMIN)  {}
         
template <class TF, class Obj> bool BenchQdt<TF,Obj>::insert(const Obj p,bool svp) {
             return qdt.insert(p,svp);
}

template <class TF, class Obj> void BenchQdt<TF,Obj>::remove(const Obj  & p) {
	qdt.remove(p);
}

template <class TF, class Obj> void BenchQdt<TF,Obj>::clear() {
	qdt.clear();
}

template <class TF, class Obj> void BenchQdt<TF,Obj>::voisins(Pt2dr p0,REAL ray, ElSTDNS set<Obj>& S0) {
     qdt.RVoisins(S0,p0,ray);
}

template <class TF, class Obj> void BenchQdt<TF,Obj>::KPPVois(Pt2dr p0, list<Obj>& S0 ,int aNb, double aDistInit, double aFact, int aNbMax) {
     S0=qdt.KPPVois(p0,aNb,aDistInit,aFact,aNbMax);
}


