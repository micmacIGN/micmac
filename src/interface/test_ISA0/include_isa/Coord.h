#ifndef DEF_COORD
#define DEF_COORD

#include "Image.h"


class Coord{
public:
  Coord();
  Coord(float i, float j, int k);

  void SetX(float a);
  void SetY(float b);
  void SetImg(int i);
  //void SetDistAlignt(float da);

  float GetX() const;
  float GetY() const;
  int GetImg() const;
  //float GetDistAlignt() const;

  Pt2dr GetPt2dr() const;

  bool isInZone(const REAL* recouvrement) const;

  bool operator == (const Coord& c)const ;
  bool operator < (const Coord& c)const ;

  
private:
  float x;
  float y;
  int img;
};



#endif

