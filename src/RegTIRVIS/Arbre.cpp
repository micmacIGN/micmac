#include "Arbre.h"
#include "StdAfx.h"

Pt2dr FPRIMTp::operator() (const pair<int,Pt2dr> & tq) const {return tq.second;}


ArbreKD::ArbreKD(FPRIMTp Pt_of_Obj, Box2dr BOX,INT NBOBJMAX,REAL SZMIN) :
              Box   (BOX), NbObj (NBOBJMAX), SZmin(SZMIN), qdt (Pt_of_Obj,Box,NBOBJMAX,SZMIN)  {}

bool ArbreKD::insert(const pair<int,Pt2dr> p,bool svp) {
             return qdt.insert(p,svp);
}

void ArbreKD::remove(const pair<int,Pt2dr>  & p) {
    qdt.remove(p);
}

void ArbreKD::clear() {
    qdt.clear();
}

void ArbreKD::voisins(Pt2dr p0,REAL ray, ElSTDNS set<pair<int,Pt2dr> >& S0) {
     qdt.RVoisins(S0,p0,ray);
}
