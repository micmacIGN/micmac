#ifndef _FORMULAS_TRIANGLES_DEFORM_H_
#define _FORMULAS_TRIANGLES_DEFORM_H_

#include "MMVII_TplSymbTriangle.h"
#include "MMVII_util_tpl.h"

#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"

/**
    \file   Formulas_TrianglesDeform.h
    \brief  class to generate code for triangle deformations by minimization
**/

using namespace NS_SymbolicDerivative;

namespace MMVII
{
  class cTriangleDeformation
  {
  public:
    cTriangleDeformation()
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return std::vector<std::string>{"GeomTrXPointA", "GeomTrYPointA", "RadTranslationPointA", "RadScalingPointA",
                                                                                             "GeomTrXPointB", "GeomTrYPointB", "RadTranslationPointB", "RadScalingPointB",
                                                                                             "GeomTrXPointC", "GeomTrYPointC", "RadTranslationPointC", "RadScalingPointC"}; }
    static const std::vector<std::string> VNamesObs()
    {
      return Append(
          std::vector<std::string>{"PixelCoordinatesX", "PixelCoordinatesY", "AlphaCoordPixel", "BetaCoordPixel", "GammaCoordPixel", "IntensityImPre"},
          FormalBilinIm2D_NameObs("T") // 6 obs for bilinear interpol of Im
      );
    }

    std::string FormulaName() const { return "TriangleDeformation"; }

    template <typename tUk, typename tObs>
    static std::vector<tUk> formula(
        const std::vector<tUk> &aVUnk,
        const std::vector<tObs> &aVObs)
    {

      // extract observation
      const auto &aXCoordinate = aVObs[0];
      const auto &aYCoordinate = aVObs[1];
      const auto &aAlphaCoordinate = aVObs[2];
      const auto &aBetaCoordinate = aVObs[3];
      const auto &aGammaCoordinate = aVObs[4];
      const auto &aIntensityImPre = aVObs[5];

      // extract unknowns
      const auto &aGeomTrXPointA = aVUnk[0];
      const auto &aGeomTrYPointA = aVUnk[1];
      const auto &aRadTranslationPointA = aVUnk[2];
      const auto &aRadScalingPointA = aVUnk[3];
      const auto &aGeomTrXPointB = aVUnk[4];
      const auto &aGeomTrYPointB = aVUnk[5];
      const auto &aRadTranslationPointB = aVUnk[6];
      const auto &aRadScalingPointB = aVUnk[7];
      const auto &aGeomTrXPointC = aVUnk[8];
      const auto &aGeomTrYPointC = aVUnk[9];
      const auto &aRadTranslationPointC = aVUnk[10];
      const auto &aRadScalingPointC = aVUnk[11];

      // Apply barycentric translation formula to coordinates
      auto aXTri = aXCoordinate + aAlphaCoordinate * aGeomTrXPointA + aBetaCoordinate * aGeomTrXPointB + aGammaCoordinate * aGeomTrXPointC;
      auto aYTri = aYCoordinate + aAlphaCoordinate * aGeomTrYPointA + aBetaCoordinate * aGeomTrYPointB + aGammaCoordinate * aGeomTrYPointC;

      // Apply barycentric interpolation to radiometric factors
      auto aRadTranslation = aAlphaCoordinate * aRadTranslationPointA + aBetaCoordinate * aRadTranslationPointB + aGammaCoordinate * aRadTranslationPointC;
      auto aRadScaling = aAlphaCoordinate * aRadScalingPointA + aBetaCoordinate * aRadScalingPointB + aGammaCoordinate * aRadScalingPointC;

      // compute formula of bilinear interpolation
      auto aBilinearValueTri = FormalBilinTri_Formula(aVObs, TriangleDisplacement_NbObs, aXTri, aYTri);

      // Take into account radiometry in minimisation process
      auto aRadiometryValueTri = aRadScaling * aIntensityImPre + aRadTranslation;

      // residual is simply the difference between values in before image and estimated value in new image
      return {aRadiometryValueTri - aBilinearValueTri};
    }
  };

  class cTriangleDeformationTrRad
  {
  public:
    cTriangleDeformationTrRad()
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return std::vector<std::string>{"GeomTrXPointA", "GeomTrYPointA", "RadTranslationPointA", "RadScalingPointA",
                                                                                             "GeomTrXPointB", "GeomTrYPointB", "RadTranslationPointB", "RadScalingPointB",
                                                                                             "GeomTrXPointC", "GeomTrYPointC", "RadTranslationPointC", "RadScalingPointC"}; }
    static const std::vector<std::string> VNamesObs()
    {
      return Append(
          std::vector<std::string>{"PixelCoordinatesX", "PixelCoordinatesY", "AlphaCoordPixel", "BetaCoordPixel", "GammaCoordPixel", "IntensityImPre"},
          FormalBilinIm2D_NameObs("T") // 6 obs for bilinear interpol of Im
      );
    }

    std::string FormulaName() const { return "TriangleDeformationTrRad"; }

    template <typename tUk, typename tObs>
    static std::vector<tUk> formula(
        const std::vector<tUk> &aVUnk,
        const std::vector<tObs> &aVObs)
    {

      // extract observation
      const auto &aXCoordinate = aVObs[0];
      const auto &aYCoordinate = aVObs[1];
      const auto &aAlphaCoordinate = aVObs[2];
      const auto &aBetaCoordinate = aVObs[3];
      const auto &aGammaCoordinate = aVObs[4];
      const auto &aIntensityImPre = aVObs[5];

      // extract unknowns
      const auto &aGeomTrXPointA = aVUnk[0];
      const auto &aGeomTrYPointA = aVUnk[1];
      const auto &aRadTranslationPointA = aVUnk[2];
      const auto &aRadScalingPointA = aVUnk[3];
      const auto &aGeomTrXPointB = aVUnk[4];
      const auto &aGeomTrYPointB = aVUnk[5];
      const auto &aRadTranslationPointB = aVUnk[6];
      const auto &aRadScalingPointB = aVUnk[7];
      const auto &aGeomTrXPointC = aVUnk[8];
      const auto &aGeomTrYPointC = aVUnk[9];
      const auto &aRadTranslationPointC = aVUnk[10];
      const auto &aRadScalingPointC = aVUnk[11];

      // Apply barycentric translation formula to coordinates
      auto aXTri = aXCoordinate + aAlphaCoordinate * aGeomTrXPointA + aBetaCoordinate * aGeomTrXPointB + aGammaCoordinate * aGeomTrXPointC;
      auto aYTri = aYCoordinate + aAlphaCoordinate * aGeomTrYPointA + aBetaCoordinate * aGeomTrYPointB + aGammaCoordinate * aGeomTrYPointC;

      // Apply barycentric interpolation to radiometric factors
      auto aRadTranslation = aAlphaCoordinate * aRadTranslationPointA + aBetaCoordinate * aRadTranslationPointB + aGammaCoordinate * aRadTranslationPointC;
      auto aRadScaling = aAlphaCoordinate * aRadScalingPointA + aBetaCoordinate * aRadScalingPointB + aGammaCoordinate * aRadScalingPointC;

      // compute formula of bilinear interpolation
      auto aBilinearValueTri = FormalBilinTri_Formula(aVObs, TriangleDisplacement_NbObs, aXTri, aYTri);

      // Take into account radiometry in minimisation process
      auto aRadiometryValueTri = aRadScaling * aBilinearValueTri + aRadTranslation;

      // residual is simply the difference between values in before image and estimated value in new image
      return {aIntensityImPre - aRadiometryValueTri};
    }
  };

  class cTriangleDeformationTranslation
  {
  public:
    cTriangleDeformationTranslation()
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return Append(std::vector<std::string>{"GeomTrXPointA", "GeomTrYPointA"},
                                                                           std::vector<std::string>{"GeomTrXPointB", "GeomTrYPointB"},
                                                                           std::vector<std::string>{"GeomTrXPointC", "GeomTrYPointC"}); }
    static const std::vector<std::string> VNamesObs()
    {
      return Append(
          std::vector<std::string>{"PixelCoordinatesX", "PixelCoordinatesY", "AlphaCoordPixel", "BetaCoordPixel", "GammaCoordPixel", "IntensityImPre"},
          FormalBilinIm2D_NameObs("T") // 6 obs for bilinear interpol of Image
      );
    }

    std::string FormulaName() const { return "TriangleDeformationTranslation"; }

    template <typename tUk, typename tObs>
    static std::vector<tUk> formula(
        const std::vector<tUk> &aVUnk,
        const std::vector<tObs> &aVObs)
    {
      // extract observation
      const auto &aXCoordinate = aVObs[0];
      const auto &aYCoordinate = aVObs[1];
      const auto &aAlphaCoordinate = aVObs[2];
      const auto &aBetaCoordinate = aVObs[3];
      const auto &aGammaCoordinate = aVObs[4];
      const auto &aIntensityImPre = aVObs[5];

      // extract unknowns
      const auto &aGeomTrXPointA = aVUnk[0];
      const auto &aGeomTrYPointA = aVUnk[1];
      const auto &aGeomTrXPointB = aVUnk[2];
      const auto &aGeomTrYPointB = aVUnk[3];
      const auto &aGeomTrXPointC = aVUnk[4];
      const auto &aGeomTrYPointC = aVUnk[5];

      auto aXTri = aXCoordinate + aAlphaCoordinate * aGeomTrXPointA + aBetaCoordinate * aGeomTrXPointB + aGammaCoordinate * aGeomTrXPointC;
      auto aYTri = aYCoordinate + aAlphaCoordinate * aGeomTrYPointA + aBetaCoordinate * aGeomTrYPointB + aGammaCoordinate * aGeomTrYPointC;

      // compute formula of bilinear interpolation
      auto aEstimatedTranslatedValueTri = FormalBilinTri_Formula(aVObs, TriangleDisplacement_NbObs, aXTri, aYTri);

      // residual is simply the difference between values in before image and estimated value in new image.
      return {aIntensityImPre - aEstimatedTranslatedValueTri};
    }
  };

  class cTriangleDeformationRadiometry
  {
  public:
    cTriangleDeformationRadiometry()
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return Append(std::vector<std::string>{"RadTranslationPointA", "RadScalingPointA"},
                                                                           std::vector<std::string>{"RadTranslationPointB", "RadScalingPointB"},
                                                                           std::vector<std::string>{"RadTranslationPointC", "RadScalingPointC"}); }

    static const std::vector<std::string> VNamesObs()
    {
      return Append(
          std::vector<std::string>{"PixelCoordinatesX", "PixelCoordinatesY", "AlphaCoordPixel", "BetaCoordPixel", "GammaCoordPixel", "IntensityImPre"},
          FormalBilinIm2D_NameObs("T") // 6 obs for bilinear interpol of Im
      );
    }

    std::string FormulaName() const { return "TriangleDeformationRadiometry"; }

    template <typename tUk, typename tObs>
    static std::vector<tUk> formula(
        const std::vector<tUk> &aVUnk,
        const std::vector<tObs> &aVObs)
    {
      // extract observation
      const auto &aXCoordinate = aVObs[0];
      const auto &aYCoordinate = aVObs[1];
      const auto &aAlphaCoordinate = aVObs[2];
      const auto &aBetaCoordinate = aVObs[3];
      const auto &aGammaCoordinate = aVObs[4];
      const auto &aIntensityImPre = aVObs[5];

      // extract unknowns
      const auto &aRadTranslationPointA = aVUnk[0];
      const auto &aRadScalingPointA = aVUnk[1];
      const auto &aRadTranslationPointB = aVUnk[2];
      const auto &aRadScalingPointB = aVUnk[3];
      const auto &aRadTranslationPointC = aVUnk[4];
      const auto &aRadScalingPointC = aVUnk[5];

      // Apply barycentric interpolation to radiometric factors
      auto aRadTranslation = aAlphaCoordinate * aRadTranslationPointA + aBetaCoordinate * aRadTranslationPointB + aGammaCoordinate * aRadTranslationPointC;
      auto aRadScaling = aAlphaCoordinate * aRadScalingPointA + aBetaCoordinate * aRadScalingPointB + aGammaCoordinate * aRadScalingPointC;

      // compute formula of bilinear interpolation
      auto aBilinearValueTri = FormalBilinTri_Formula(aVObs, TriangleDisplacement_NbObs, aXCoordinate, aYCoordinate);

      // Take into account radiometry in minimisation process
      auto aRadiometryValueTri = aRadScaling * aIntensityImPre + aRadTranslation;

      // residual is simply the difference between values in before image and estimated value in new image
      return {aRadiometryValueTri - aBilinearValueTri};
    }
  };

class cTriangleDeformationRad
  {
  public:
    cTriangleDeformationRad()
    {
    }

    static const std::vector<std::string> VNamesUnknowns() { return Append(std::vector<std::string>{"RadTranslationPointA", "RadScalingPointA"},
                                                                           std::vector<std::string>{"RadTranslationPointB", "RadScalingPointB"},
                                                                           std::vector<std::string>{"RadTranslationPointC", "RadScalingPointC"}); }

    static const std::vector<std::string> VNamesObs()
    {
      return Append(
          std::vector<std::string>{"PixelCoordinatesX", "PixelCoordinatesY", "AlphaCoordPixel", "BetaCoordPixel", "GammaCoordPixel", "IntensityImPre"},
          FormalBilinIm2D_NameObs("T") // 6 obs for bilinear interpol of Im
      );
    }

    std::string FormulaName() const { return "TriangleDeformationRad"; }

    template <typename tUk, typename tObs>
    static std::vector<tUk> formula(
        const std::vector<tUk> &aVUnk,
        const std::vector<tObs> &aVObs)
    {
      // extract observation
      const auto &aXCoordinate = aVObs[0];
      const auto &aYCoordinate = aVObs[1];
      const auto &aAlphaCoordinate = aVObs[2];
      const auto &aBetaCoordinate = aVObs[3];
      const auto &aGammaCoordinate = aVObs[4];
      const auto &aIntensityImPre = aVObs[5];

      // extract unknowns
      const auto &aRadTranslationPointA = aVUnk[0];
      const auto &aRadScalingPointA = aVUnk[1];
      const auto &aRadTranslationPointB = aVUnk[2];
      const auto &aRadScalingPointB = aVUnk[3];
      const auto &aRadTranslationPointC = aVUnk[4];
      const auto &aRadScalingPointC = aVUnk[5];

      // Apply barycentric interpolation to radiometric factors
      auto aRadTranslation = aAlphaCoordinate * aRadTranslationPointA + aBetaCoordinate * aRadTranslationPointB + aGammaCoordinate * aRadTranslationPointC;
      auto aRadScaling = aAlphaCoordinate * aRadScalingPointA + aBetaCoordinate * aRadScalingPointB + aGammaCoordinate * aRadScalingPointC;

      // SymbPrint(aRadTranslation, "RadTranslation");
      // SymbPrint(aRadScaling, "RadScaling");

      // compute formula of bilinear interpolation
      auto aBilinearValueTri = FormalBilinTri_Formula(aVObs, TriangleDisplacement_NbObs, aXCoordinate, aYCoordinate);

      // Take into account radiometry in minimisation process
       auto aRadiometryValueTri = aRadScaling * aBilinearValueTri + aRadTranslation;

      // residual is simply the difference between values in before image and estimated value in new image
      return {aIntensityImPre - aRadiometryValueTri};
    }
  };

}; // MMVII

#endif // _FORMULAS_TRIANGLES_DEFORM_H_