#ifndef  _MMVII_Interpolators_H_
#define  _MMVII_Interpolators_H_

namespace MMVII
{
/*  ********************************* */
/*       Kernels                      */
/* ********************************** */

/// A kernel, approximating "gauss"

/**  a quick kernel, derivable, with support in [-1,1], coinciding with bicub in [-1,1] 
     not really gauss but has a "bell shape"
     1 +2X^3 -3X^2  , it's particular of bicub with Deriv(1) = 0
*/

/// If we dont need any kernel interface keep it simple 
tREAL8 CubAppGaussVal(const tREAL8&);


class cRessampleWeigth
{
    public :
         static cRessampleWeigth  GaussBiCub(const cPt2dr & aPtsIn,const cAff2D_r & aMapIn, double aSzK);
    // private :
         std::vector<cPt2di>  mVPts;
         std::vector<double>  mVWeight;
};


};

#endif  //  _MMVII_Interpolators_H_
