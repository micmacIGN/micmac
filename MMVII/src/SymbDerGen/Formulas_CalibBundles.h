#ifndef _FORMULA_CALIB_BUNDLE_H_
#define _FORMULA_CALIB_BUNDLE_H_

#include "MMVII_PhgrDist.h"
#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"

/** classes for "small" Pose bundles , or more generally
   bundle where the camera are calibrated and we directly maniplutae the 3D direction.

   This optimized for "small" bundle in the sense that :
     * the parmatrization is "ad hoc" : No Unkown for first cam, 5 unknown for second cam
     * in the 2 pose case, we  have option without the unkwnon 3D point, to avoid Schurr complement
*/

namespace MMVII
{
using namespace NS_SymbolicDerivative;


/* *************************************************** */
/*                                                     */
/*         Case with Unknown 3D point                  */
/*                                                     */
/* *************************************************** */

/*  In this case the equation will enforce a local-3D point to be on the local bundle. There is 3 variant
 *  on how we measure the belonging of point to bundle.
 */

template <typename tUk>
   std::vector<tUk> ResiduAngular(const  cPtxd<tUk,3>& aDirBundle,const  cPtxd<tUk,3>& aP3dLoc)
{
    cPtxd<tUk,3> aPt = (aDirBundle ^ aP3dLoc ) * (1.0/ Scal(aDirBundle,aP3dLoc)) ;
    return {aPt.x(),aPt.y(),aPt.z()};
}

template <typename tUk>
      std::vector<tUk> ResiduProduit(const  cPtxd<tUk,3>& aDirBundle,const  cPtxd<tUk,3>& aP3dLoc)
{
       cPtxd<tUk,3> aPt = (aDirBundle ^ aP3dLoc ) ;
       return {aPt.x(),aPt.y(),aPt.z()};
}

template <typename tUk>
      std::vector<tUk> ResidualBundle_PGround
      (const  cPtxd<tUk,3>& aDirBundle,const  cPtxd<tUk,3>& aPGround,eModResBund aMode)
{
    if (aMode==eModResBund::eAngle)
       return ResiduAngular(aDirBundle,aPGround);

    MMVII_INTERNAL_ASSERT_always(aMode==eModResBund::eProduct,"ResidualBundle")
    return ResiduProduit(aDirBundle,aPGround);
}


/// Class for first camera, no unknown for this camera, as it is the refernce
class cFormulaBundleElem_Cam1
{
      public :

           cFormulaBundleElem_Cam1(eModResBund aMode) : mMode (aMode) {}

           std::string FormulaName() const { return "BunleElem_Cam1_" +E2Str(mMode);}

           /// the only unknown is the 3D point (camera is the reference system)
           std::vector<std::string>  VNamesUnknowns()  const {return   NamesP3("PGround") ;}

           /// Only obs for bundle
           std::vector<std::string>    VNamesObs() const{  return   NamesP3("Bundle");};

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
                   cPtxd<tUk,3> aPGround = VtoP3(aVUk);  //
                   cPtxd<tUk,3> aDirBundle = VtoP3(aVObs);  //
                   return  ResidualBundle_PGround(aDirBundle,aPGround,mMode);
           }

         private :
              eModResBund mMode;
};

/// Class for second camera,  the base is unkwnon but unitary, the rotation is unknown
class cFormulaBundleElem_Cam2
{
   public :

        cFormulaBundleElem_Cam2(eModResBund aMode) : mMode (aMode) {}

        std::string FormulaName() const { return "BunleElem_Cam2_"+E2Str(mMode);}

        /// Unknowns are 3d point,  Base (unitar), Orientation of Camera
        std::vector<std::string>  VNamesUnknowns()  const
        {
             return  Append(NamesP3("PGround"),NamesP2("DuDv2"), NamesP3("Omega"));
        }

        /// Observations are Bundle, Obs of Unitary Base, Of rotation
        std::vector<std::string>    VNamesObs() const
        {
            return  {Append(NamesP3("Bundle"),NamesObsP3Norm("Base"),NamesMatr("Rot",cPt2di(3,3)))};
        };

        template <typename tUk>
                    std::vector<tUk> formula
                    (
                       const std::vector<tUk> & aVUk,
                       const std::vector<tUk> & aVObs
                    ) const
        {
                size_t aIndUk =0;
                size_t aIndObs = 0;

                cPtxd<tUk,3> aPGround = VtoP3AutoIncr(aVUk,&aIndUk);      // Extract PGround
                cPtxd<tUk,3> aDirBundle = VtoP3AutoIncr(aVObs,&aIndObs);  // Extract dir of bundle
                cP3dNorm<tUk> aBase (aVUk,&aIndUk,aVObs,&aIndObs);        // Extract the base unitary
                cRot3dF<tUk>  aRot (aVUk,&aIndUk,aVObs,&aIndObs);         // Extarct the rotation

                return  ResidualBundle_PGround
                        (
                            aRot.Value(aDirBundle), // Direction of bundle set in the coordinate system
                            aPGround-aBase.CurPt(), // Vector Cam->PGround
                            mMode
                         );
        }

      private :
        eModResBund mMode;

};


/* *************************************************** */
/*                                                     */
/*          Case W/O Unknown 3D point                  */
/*                                                     */
/* *************************************************** */

/* Formula "12" we express residual w/o using the unknown ground point that lead to Schurr complement.
 *
 *     * let B the base,
 *     *  U1,U2 the dir bundle with ||U1|| = ||U2|| = 1.
 *     * S = U1.U2
 *     *  let N = U1 ^ U2 / ||U1 ^U2|| the normal, with ||N|| = 1
 *     *  let I be the intersection of bundles  (0,U1) et (B,U2)
 *
 *  We write :    B = a U1 + b U2 + c N [1]
 *
 *  Using scalar product on [1] with N, we get  :
 *
 *     *  c = B . N = [B U1 U2] / ||U1 ^U2||
 *
 *   This lead  c=Dist=ResiduDist12 and the simplified value ResiduDet12 = [B U1 U2]
 *
 *  Taking scalar with U1 and U2 on [1], we get :
 *
 *      [1 S] [a]   [B.U1]         [a]    [ B.U1   -  S B.U2]
 *      [S 1] [b] = [B.U2]  then   [b]  = [-S B.U1 +  B.U2  ]  / (1 - S^2)
 *
 *         And the angular residual are :  c/a and c/b
 */

/** Simplest case, we just use the determinanr [B,U1,U2],  surprisingly the moste stable on simulations, almost
 always converge */
template <typename tUk>
      std::vector<tUk>
        ResiduDet12(const  cPtxd<tUk,3>& aBase,const  cPtxd<tUk,3>& aDirB1,const  cPtxd<tUk,3>& aDirB2)
{
   return {Scal(aBase,aDirB1^aDirB2)};
}

/// Case we use the distance of intersection
template <typename tUk>
   std::vector<tUk>
        ResiduDist12(const  cPtxd<tUk,3>& aBase,const  cPtxd<tUk,3>& aDirB1,const  cPtxd<tUk,3>& aDirB2)
{
   return {Scal(aBase,aDirB1^aDirB2) / Norm2(aDirB1^aDirB2) };
}

 /// Case we use the 2 angles for intersection
template <typename tUk>
   std::vector<tUk>
     ResiduAng12(const  cPtxd<tUk,3>& aBase,const  cPtxd<tUk,3>& aDirB1,const  cPtxd<tUk,3>& aDirB2)
{
       tUk S   = Scal(aDirB1,aDirB2);
       tUk BU1 = Scal(aBase ,aDirB1);
       tUk BU2 = Scal(aBase ,aDirB2);

       tUk aDet = ( CreateCste(1.0,S) - Square(S));

       tUk a = (BU1 - S *   BU2) / aDet;
       tUk b = (- S * BU1 + BU2) / aDet;
       tUk c = Scal(aBase,aDirB1^aDirB2) / Norm2(aDirB1^aDirB2);

       return {c/a,c/b};
}


template <typename tUk>
         std::vector<tUk> ResidualBundle_12
         (const  cPtxd<tUk,3>& aBase,const  cPtxd<tUk,3>& aDirB1,const  cPtxd<tUk,3>& aDirB2,eModResBund aMode)
{
       if (aMode==eModResBund::eDet12)
          return ResiduDet12(aBase,aDirB1,aDirB2);

       if (aMode==eModResBund::eDist12)
           return ResiduDist12(aBase,aDirB1,aDirB2);

       MMVII_INTERNAL_ASSERT_always(aMode==eModResBund::eAng12,"ResidualBundle_12")

       return ResiduAng12(aBase,aDirB1,aDirB2);
}


/// Class  integrating 2 camera w/o ; formula w/o Unknown Ground Point
class cFormulaBundleElem_CamDet12
{
   public :

       cFormulaBundleElem_CamDet12(eModResBund aMode) : mMode (aMode) {}

        std::string FormulaName() const { return "BunleElem_"+ToStr(mMode);}

        /// Unkowns : Base & Rot of Cam2
        std::vector<std::string>  VNamesUnknowns()  const
        {
             return  Append(NamesP2("DuDv2"), NamesP3("Omega"));
        }

        /// Obs : bundle 1, bundle 1, Base of Cam2, Rotation of Cam2
        std::vector<std::string>    VNamesObs() const
        {
            return  {Append(NamesP3("Bund1"),NamesP3("Bund2"),NamesObsP3Norm("Base"),NamesMatr("Rot",cPt2di(3,3)))};
        };

        template <typename tUk>
                    std::vector<tUk> formula
                    (
                       const std::vector<tUk> & aVUk,
                       const std::vector<tUk> & aVObs
                    ) const
        {
                size_t aIndUk =0;
                size_t aIndObs = 0;
                cPtxd<tUk,3> aDirB1 = VtoP3AutoIncr(aVObs,&aIndObs);  //
                cPtxd<tUk,3> aDirB2Loc = VtoP3AutoIncr(aVObs,&aIndObs);  //Dir bundle in Cam2 sys
                cP3dNorm<tUk> aBase (aVUk,&aIndUk,aVObs,&aIndObs);
                cRot3dF<tUk>  aRot (aVUk,&aIndUk,aVObs,&aIndObs);

                cPtxd<tUk,3>  aDirB2 = aRot.Value(aDirB2Loc);  // Dir Bundle in Cam1 Sys

                return  ResidualBundle_12(aBase.CurPt(),aDirB1,aDirB2,mMode) ;
        }

      private :
         eModResBund mMode;

};

};


#endif  //  _FORMULA_CALIB_BUNDLE_H_
