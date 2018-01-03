#ifndef IMAGE_H_
#define IMAGE_H_
#include "StdAfx.h"
/********************************************************************/
/*                                                                  */
/*             Color                                                */
/*                                                                  */
/********************************************************************/
class Color
{
  public:
    Color(U_INT2 r,U_INT2 g,U_INT2 b):mR(r),mG(g),mB(b){} // constructor
    void setR(U_INT2 r){mR=r;}
    void setG(U_INT2 g){mG=g;}
    void setB(U_INT2 b){mB=b;}
    U_INT2 r(){return mR;}
    U_INT2 g(){return mG;}
    U_INT2 b(){return mB;}
  protected:
    U_INT2 mR;
    U_INT2 mG;
    U_INT2 mB;
};
// define  a class where the colour is stored
/********************************************************************/
/*                                                                  */
/*             ColorImg                                             */
/*                                                                  */
/********************************************************************/
class ColorImg
{
  public:
    ColorImg(std::string filename);
    ColorImg(Pt2di sz);
    ~ColorImg();
    Color get(Pt2di pt);
    Color getr(Pt2dr pt);
    void set(Pt2di pt, Color color);
    void setChannels(int chan);
    int getChannnels();
    void write(std::string filename);
    ColorImg ResampleColorImg(double aFact);
    Pt2di sz(){return mImgSz;}
  protected:
    std::string mImgName;
    Pt2di mImgSz;
    Im2D<U_INT2,INT4> *mImgR;
    Im2D<U_INT2,INT4> *mImgG;
    Im2D<U_INT2,INT4> *mImgB;
    TIm2D<U_INT2,INT4> *mImgRT;
    TIm2D<U_INT2,INT4> *mImgGT;
    TIm2D<U_INT2,INT4> *mImgBT;
    int mChannels;
};
#endif
