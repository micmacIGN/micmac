#ifndef ARBRE_H_
#define  ARBRE_H_
#include "StdAfx.h"
#define ElSTDNS std::


class FPRIMTp
{
    public :
        Pt2dr operator() (const pair<int,Pt2dr> & tq) const;
};
class ArbreKD
{
      public :

         ArbreKD (FPRIMTp Pt_of_Obj, Box2dr BOX,INT NBOBJMAX,REAL SZMIN);
         bool insert(const pair<int,Pt2dr> p,bool svp=false);
         void remove(const pair<int,Pt2dr>  & p);
         void clear();
         void voisins(Pt2dr p0,REAL ray, ElSTDNS set<pair<int,Pt2dr> > & S0);

         const Box2dr                    Box;
         const INT                       NbObj;
         const REAL                      SZmin;

      private :
         ElQT<pair<int,Pt2dr>,Pt2dr,FPRIMTp>      qdt;
};

#endif
