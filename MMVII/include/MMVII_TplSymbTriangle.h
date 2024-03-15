#ifndef _MMVII_TplSymbTriangle_
#define _MMVII_TplSymbTriangle_

#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"

#include "MMVII_Ptxd.h"
#include "MMVII_Geom2D.h"

#include "../src/MeshDisplacement/TriangleDeformation.h"

/**
 \file MMVII_TplSymbTriangle.h
 \brief Contains helpers for triangle as formula
**/

using namespace NS_SymbolicDerivative;

namespace MMVII
{

    /**
    Compute the formula for composition of a function and an image.

     Let I be an image, considered as function  R^2 -> R using an interpolation model, this computes
     the formula corresponding to the function

          x,y  ->   I(Fx(x,y),Fy(x,y))

     We use here the bilinear interpolation.  Let X0,Y0 be integers such that:

             X0 <= Fx(x,y) < X0+1
             Y0 <= Fy(x,y) < Y0+1
     Let X1=X0+1 , Y1=Y0+1, X=Fx(x,y),Y=Fy(x,y)

     Let I00= I[X0,Y0]   , I10 = I[X1,Y0] ...  the  bilinear formula is :

     Bil(I,X,Y) =
                    I00  * (X1-X) * (Y1-Y)
                  + I10  * (X-X0) * (Y1-Y)
                  + I01  * (X1-X) * (Y-Y0)
                  + I11  * (X-X0) * (Y-Y0)

     In this function the value  X0, Y0,  I00, I10, I01, I11  are communicated as element
     of the observation vector aVObs, starting from aKObs0.
    **/

    template <class TypeFunc, class TypeObs>
    TypeFunc FormalBilinTri_Formula(
        const std::vector<TypeObs> &aVObs,
        int aKObs0,
        const TypeFunc &FX,
        const TypeFunc &FY)
    {
        TypeFunc aX0(aVObs.at(aKObs0));
        TypeFunc aY0(aVObs.at(aKObs0 + 1));
        TypeFunc aCst1 = CreateCste(1.0, aX0); // create a symbolic formula for constant 1

        TypeFunc aWX1 = FX - aX0;     // weight for I10 and I11
        TypeFunc aWX0 = aCst1 - aWX1; // weight for I00 and I01
        TypeFunc aWY1 = FY - aY0;     // weight for I10 and I11
        TypeFunc aWY0 = aCst1 - aWY1; // weight for I00 and I10

        return aWX0 * aWY0 * aVObs.at(aKObs0 + 2)    // I00
               + aWX1 * aWY0 * aVObs.at(aKObs0 + 3)  // I10
               + aWX0 * aWY1 * aVObs.at(aKObs0 + 4)  // I01
               + aWX1 * aWY1 * aVObs.at(aKObs0 + 5); // I11
    }

    //  standard name for observation
    std::vector<std::string> FormalBilinIm2D_NameObs(const std::string &aPrefix);

    template <class Type>
    void FormalInterpBarycenter_SetObs(
        std::vector<Type> &aVObs, // vector of observation to fill
        const int aK0,            // first index where fill the vector
        const cPtInsideTriangles &aPixInsideTriangle)
    {
        // push integer coordinate of point
        SetOrPush(aVObs, aK0, Type(aPixInsideTriangle.GetCartesianCoordinates().x()));
        SetOrPush(aVObs, aK0 + 1, Type(aPixInsideTriangle.GetCartesianCoordinates().y()));
        SetOrPush(aVObs, aK0 + 2, Type(aPixInsideTriangle.GetBarycenterCoordinates().x()));
        SetOrPush(aVObs, aK0 + 3, Type(aPixInsideTriangle.GetBarycenterCoordinates().y()));
        SetOrPush(aVObs, aK0 + 4, Type(aPixInsideTriangle.GetBarycenterCoordinates().z()));
        SetOrPush(aVObs, aK0 + 5, Type(aPixInsideTriangle.GetPixelValue()));
    }

    /*
    This is the "companion" function of  FormalBilinIm2D_Formula, it fills
    the vector aVObs with X0,Y0,I00, that will be used in FormalBilinIm2D_Formula.
    */

    template <class Type, class TypeIm>
    void FormalBilinTri_SetObs(
        std::vector<Type> &aVObs,     // vector of observation to fill
        const int aK0,                // first index where fill the vector
        const cPt2dr aPtIm,           // point in image
        const cDataIm2D<TypeIm> &aDIm // image
    )
    {
        // compute coordinate of left-high corner of including pixel
        cPt2di aP0 = Pt_round_down(aPtIm);

        // push integer coordinate of point
        SetOrPush(aVObs, aK0, Type(aP0.x()));
        SetOrPush(aVObs, aK0 + 1, Type(aP0.y()));

        // push values of image at its 4 corners
        SetOrPush(aVObs, aK0 + 2, (Type)aDIm.GetV(aP0));
        SetOrPush(aVObs, aK0 + 3, (Type)aDIm.GetV(aP0 + cPt2di(1, 0)));
        SetOrPush(aVObs, aK0 + 4, (Type)aDIm.GetV(aP0 + cPt2di(0, 1)));
        SetOrPush(aVObs, aK0 + 5, (Type)aDIm.GetV(aP0 + cPt2di(1, 1)));
    }

    constexpr int TriangleDisplacement_NbObs = 6;
};

#endif //  _MMVII_TplSymbTriangle_