#include "MMVII_Ptxd.h"


/**
   \file CalibratedSpaceResection.cpp

   \brief file for implementing the space resection algorithm in calibrated case

      3 direction   A,B,C   we assume ||A|| = ||B|| = ||C|| = 1
      We parametrize 3 point on the bundle by 2 parameters b & c:
           PA  = A  (arbitrarily we fix on this bundle)
	   PB  = B(1+b)
	   PC  = C(1+c)

     We know  GA,GB,GC  the ground point

     |PA-PB|^2    |GA-GB|^2
      -----   =  ---------  = rA
     |PA-PC|^2    |GA-GC|^2


     (A-(1+b)B)^2 = ra (A-(1+c)C)^2

     1 -2(1+b)AB  +(1+b)^2 B2 = rA(1 -2(1+c) AC + (1+c)^2 C2)

      B2 b^2   + b * 2 (B2-AB) + (1-2AB+B2 -rA(1 -2(1+c) AC + (1+c)^2 C2)) = 0

*/

namespace MMVII
{

	/*
struct  cPair3D3D
{
     public :
          cPair2D3D(const cPt2dr &,const cPt3dr &);
          cPt2dr mP2;
          cPt3dr mP3;
};
*/


class cElemSpaceResection
{
      public :
           // cElemSpaceResection
};


}; // MMVII

