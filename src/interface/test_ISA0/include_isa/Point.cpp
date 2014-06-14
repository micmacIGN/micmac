#include "StdAfx.h"
#include "Point.h"

using namespace std;


  Point::Point() {nbHom=0;}
  Point::Point(Coord c):coord(c) {nbHom=0;}
  Point::Point(const Point& P): coord(P.GetCoord()), nbHom(P.GetNbHom()) {
	for (list<Coord>::const_iterator itC=P.begin(); itC!=P.end(); itC++)
		push_back(*itC);
  }
	

  void Point::SetCoord(Coord c) {coord.SetX(c.GetX()); coord.SetX(c.GetX()); coord.SetX(c.GetX());}
 
  Coord Point::GetCoord() const {return coord;}
  int Point::GetNbHom() const {return nbHom;}


  void Point::AddHom(Coord c) {
	push_back(c);
	nbHom++;
  }
  Pt2dr Point::GetPt2dr() const {return coord.GetPt2dr();}//inline
  Pt2dr Point::GetPt2drHomol() const {return (*begin()).GetPt2dr();}//inline


/////////////////////////////////////////////////////////////////////////

  Point1::Point1(Coord c): Point(c) {
	suppr=false;
	carre[0]=0; carre[1]=0;
  }

  void Point1::SetSuppr(bool b) {suppr = b;}
  void Point1::SetCoherence(float coh) {coherence = coh;}

  bool Point1::GetSuppr() const {return suppr;}
  float Point1::GetCoherence() const {return coherence;}

  float Point1::GetCohCarre() const {
	if(carre[0]+carre[1]>0) return float(carre[0])/float(carre[0]+carre[1]);
	else return 0;
  }

  void Point1::IncrCarre(bool b) {
	if(b) carre[0]++;
	else carre[1]++;
  }

  list<Point1*>* Point1::GetListVoisins(int i) {return (i==1)? &(voisins.first) : &(voisins.second);}
  void Point1::AddVoisin(Point1* P, int i) {
	list<Point1*>* l=GetListVoisins(i);
	if(!((*this) == (*P))) (*l).push_back(P);
  }
  void Point1::UpDateVoisins() {
	for (int i=1; i<3; i++) {
		list<Point1*>* l=GetListVoisins(i);
		for (list<Point1*>::iterator itP=(*l).begin(); itP!=(*l).end(); itP++) {
			if ((*(*itP)).GetSuppr()) {(*l).erase(itP);itP--;}
		}
	}
  }
  void Point1::CalculeCoherence(bool unicite) {
	if (unicite) {
		voisins.first.unique();
		voisins.second.unique();
	}
	int nbCoh=0;
	for (list<Point1*>::const_iterator it=voisins.first.begin(); it!=voisins.first.end(); it++) {
		if((**it).FindPoint(voisins.second.begin(), voisins.second.end())!=0) nbCoh++;
	}
	SetCoherence(float(nbCoh)/(float(voisins.first.size())));
  }
  void Point1::ClearVoisins() {
	voisins.first.clear();
	voisins.second.clear();
  }
  Point1** Point1::FindPoint(list<Point1*>::iterator itBegin, list<Point1*>::iterator itEnd) {
    for (list<Point1*>::iterator itP=itBegin; itP!=itEnd; itP++){
	  if ((*this)==(**itP)) {
		return &(*itP);
	  }
    }	
    return 0;
  }

  bool Point1::operator < (const Point1 & p1) const { return (GetCoherence()<p1.GetCoherence()); }
  bool Point1::operator == (const Point1 & p1) const { return (Point(*this).GetCoord()==Point(p1).GetCoord() && *(begin())==*(p1.begin())); }
  Pt2dr Point1::GetPt2drHomol() const {return (*begin()).GetPt2dr();}

//////////////////////////////////////////////////////////////////////////

  PtrPt::PtrPt() : pointeur(0) {}
  PtrPt::PtrPt(Point1* P) : pointeur(P) {//
	/*Point1* ptr;
	ptr=&P;
	pointeur=ptr;*/
	//cout << pointeur << "\n";
  };

  Pt2dr PtrPt::GetPt2dr() const {return Point(*pointeur).GetPt2dr();}
  Pt2dr PtrPt::GetPt2drHomol() const {return (*pointeur).GetPt2drHomol();}

  bool PtrPt::operator < (const PtrPt & p1) const { return (*pointeur<*(p1.pointeur)); }
  bool PtrPt::operator == (const PtrPt & p1) const { return (*pointeur==*(p1.pointeur)); }

//////////////////////////////////////////////////////////////////////////

  Point2::Point2(Point1 P): Point(P) {
	select=false;
  }

  void Point2::SetSelect(bool b) {select = b;}
  bool Point2::GetSelect() const {return select;}

  bool Point2::operator < (const Point2 & p2) const
  {
       if (GetCoord().GetImg()<p2.GetCoord().GetImg()) {return true;}
       else if ((GetCoord().GetImg()==p2.GetCoord().GetImg()) && (GetNbHom()>p2.GetNbHom())) {return true;}
	else if ((GetCoord().GetImg()==p2.GetCoord().GetImg()) && (GetNbHom()==p2.GetNbHom())) {return (GetCoord() < p2.GetCoord());}
       else {return false;}
  }
  bool Point2::operator == (const Point2 & p2) const { return  GetCoord() == p2.GetCoord(); }
  bool Point2::operator == (const Coord & c) const { return (GetCoord()==c); }

//////////////////////////////////////////////////////////////////////////

  ListPt::ListPt() {}

  void ListPt::ReplacePoint(Point1 pt) {
		list<Point2>::iterator it=find(begin(),end(),Point(pt).GetCoord());
	if (it!=end()){
		(*it).AddHom(*(pt.begin()));//ajoute l'homologue
	}
	 else {
		push_back(Point2(pt));
	}
  }
