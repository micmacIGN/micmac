#ifndef _FORMULA_IMAGES_DEFORM_H_
#define _FORMULA_IMAGES_DEFORM_H_
#include "MMVII_TplSymbImage.h"
#include "MMVII_util_tpl.h"

/**
    \brief  class to generate code for images transformation by mimnization
*/

// #include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;



namespace MMVII
{

/**  Game example, class for computing homothety bewteen two images 

     We have a modele fonction M (which can be an image or not) and the image I
     of that regisetr this function after certain transformation. The transformation
     is both geomtric and radiometric, the unknown are :

         * RadTr & RadSc for the radiometry
         * Tr (translation) and S (scale) for the geometry. The equation we have is :
     
         _I_ ( GTr + GS(x,y)) = RadTr +  RadSc * _M_ (x,y)
*/

class cDeformImHomotethy
{
  public :
    // temporay def value for backward compat
    cDeformImHomotethy(bool isLinearGrad=false)  :
          mLinearGrad  (isLinearGrad)
    {
    }

    static std::vector<std::string> VNamesUnknowns() 
    { 
        return {"RadSc","RadTr","GeomSc","GeomTrX","GeomTrY"};  // 2 radiometry + 3 geometry
    }
    std::vector<std::string> VNamesObs()   const
    { 
           return Append
                  (
		       // 5 or 6 obs for linear-grad or bilinear interpol of Im
                       mLinearGrad ?  FormalGradInterpol_NameObs("H")  : FormalBilinIm2D_NameObs("H") ,  
                       std::vector<std::string>{"xMod","yMod","ValueMod"} // x,y of point, value of modele
                  );
    }

    std::string FormulaName() const { return  mLinearGrad ?  "LGrad_DeformImHomotethy" : "DeformImHomotethy";}

    template <typename tUk,typename tObs> 
             std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )  const
    {
          size_t IndBilin = 0;
          size_t IndXModele = (size_t) (mLinearGrad ? FormalGradInterpolIm2D_NbObs : FormalBilinIm2D_NbObs ) + IndBilin;

	  // extract observation on model 
          const auto & xModele    = aVObs[IndXModele];
          const auto & yModele    = aVObs[IndXModele+1];
          const auto & vModelInit = aVObs[IndXModele+2];

	  // extract unknowns
          const auto & aRadSc     = aVUk[0];
          const auto & aRadTr     = aVUk[1];
          const auto & aGeomScale = aVUk[2];
          const auto & aGeomTrx   = aVUk[3];
          const auto & aGeomTry   = aVUk[4];

	  // compute pixel homologous to model in image
          auto  xIm = aGeomTrx + aGeomScale *  xModele;
          auto  yIm = aGeomTry + aGeomScale *  yModele;

	  // compute formula of bilinear interpolation
          auto aValueIm =   mLinearGrad                                             ?
		            FormalGradInterpolIm2D_Formula(aVObs,IndBilin,xIm,yIm)  :
		            FormalBilinIm2D_Formula(aVObs,IndBilin,xIm,yIm)         ;
	  // take into account radiometric transform
          auto aValueModele = aRadTr + aRadSc * vModelInit;

	  // residual is simply the difference  between both value
          return { aValueModele - aValueIm};
     }

  private :
     bool mLinearGrad; // Do we use the linear-grad of the bilinear model 
};



// ------------------------------------------------------------------
// Test affinity estimation (now for the coded target detection)
// ------------------------------------------------------------------



class cDeformImAffinity
{
  public :
    cDeformImAffinity() 
    {
    }

    static std::vector<std::string> VNamesUnknowns() 
    { 
        return {"RadSc","RadTr","GeomScA11", "GeomScA12", "GeomScA21", "GeomScA22", "GeomTrX","GeomTrY"};  // 2 radiometry + 6 geometry
    }
    static const std::vector<std::string> VNamesObs()      
    { 
           return Append
                  (
                       FormalBilinIm2D_NameObs("H") ,  // 6 obs for bilinear interpol of Im
                       std::vector<std::string>{"xMod","yMod","ValueMod"} // x,y of point, value of modele
                  );
    }

    std::string FormulaName() const { return "DeformImAffinity";}

    template <typename tUk,typename tObs> 
             static std::vector<tUk> formula
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  ) // const
    {
          size_t IndBilin = 0;
          size_t IndX = FormalBilinIm2D_NbObs+ IndBilin;

	  // extract observation on model 
          const auto & xModele    = aVObs[IndX];
          const auto & yModele    = aVObs[IndX+1];
          const auto & vModelInit = aVObs[IndX+2];

	  // extract unknowns
          const auto & aRadSc     = aVUk[0];
          const auto & aRadTr     = aVUk[1];
          const auto & GeomScA11  = aVUk[2];
          const auto & GeomScA12  = aVUk[3];
          const auto & GeomScA21  = aVUk[4];
          const auto & GeomScA22  = aVUk[5];
          const auto & aGeomTrx   = aVUk[6];
          const auto & aGeomTry   = aVUk[7];

	  // compute pixel homologous to model in image
          auto  xIm = aGeomTrx + GeomScA11 *  xModele + GeomScA12 * yModele;
          auto  yIm = aGeomTry + GeomScA21 *  xModele + GeomScA22 * yModele;

	  // compute formula of bilinear interpolation
          auto aValueIm = FormalBilinIm2D_Formula(aVObs,IndBilin,xIm,yIm);
	  // take into account radiometric transform
          auto aValueModele = aRadTr + aRadSc * vModelInit;

	  // residual is simply the difference  between both value
          return { aValueModele - aValueIm};
     }
};




};//  namespace MMVII

#endif // _FORMULA_IMAGES_DEFORM_H_
