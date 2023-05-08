#include "MMVII_PCSens.h"
// #include "MMVII_BundleAdj.h"


/* We have the image formula w/o distorsion:


*/

namespace MMVII
{

class cHomogCpleIm
{
      public :
           cHomogCpleIm(const cPt2dr &,const cPt2dr &);
           cPt2dr  mP1;
           cPt2dr  mP2;
};

class cHomogCpleDir
{
      public :
           cHomogCpleDir(const cPt3dr &,const cPt3dr &);
           cPt3dr  mP1;
           cPt3dr  mP2;
};


// class 



}; // MMVII




