#include "StdAfx.h"
#include "Coord.h"
#include "Image.h"

using namespace std;

 
  Coord::Coord() {}
  Coord::Coord(float i, float j, int k):x(i), y(j), img(k) {}

  void Coord::SetX(float a) {x = a;}
  void Coord::SetY(float b) {y = b;}
  void Coord::SetImg(int i) {img = i;}
  //void Coord::SetDistAlignt(float da) {distAlignt = da;}

  float Coord::GetX() const {return x;}
  float Coord::GetY() const {return y;}
  int Coord::GetImg() const {return img;}
  //float Coord::GetDistAlignt() const {return distAlignt;}

  Pt2dr Coord::GetPt2dr() const {
		return Pt2dr(x,y);
  }

  bool Coord::isInZone(const REAL* recouvrement) const {
	REAL Xmin=*recouvrement;
	REAL Ymin=*(recouvrement+1);
	REAL Xmax=*(recouvrement+2);
	REAL Ymax=*(recouvrement+3);

	if (x<Xmin || x>Xmax || y<Ymin || y>Ymax) return false;
	return true;
  }

  bool Coord::operator == (const Coord& c) const {
	  if (x==c.GetX() && y==c.GetY() && img==c.GetImg()) {return true;}
	  else {return false;}
  }
  bool Coord::operator < (const Coord& c) const {
	  if (img<c.GetImg()) {return true;}
	  else if (img==c.GetImg() && x<c.GetX()) {return true;}
	  else if (img==c.GetImg() && x==c.GetX() && y<c.GetY()) {return true;}
	  else {return false;}
  }

