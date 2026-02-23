#ifndef _FORMULA_CALIB_BUNDLE_H_
#define _FORMULA_CALIB_BUNDLE_H_

#include "MMVII_PhgrDist.h"
#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"

/* classes for "small" Pose bundles , or more generally
   bundle where the camera are calibrated and we directly maniplutae the 3D direction
*/

namespace MMVII
{
using namespace NS_SymbolicDerivative;


template <typename tUk>
   std::vector<tUk> ResiduAngular(const  cPtxd<tUk,3>& aDirBundle,const  cPtxd<tUk,3>& aPGround)
{
    cPtxd<tUk,3> aPt = (aDirBundle ^ aPGround ) * (1.0/ Scal(aDirBundle,aPGround)) ;
    return {aPt.x(),aPt.y(),aPt.z()};
}

template <typename tUk>
      std::vector<tUk> ResiduProduit(const  cPtxd<tUk,3>& aDirBundle,const  cPtxd<tUk,3>& aPGround)
{
       cPtxd<tUk,3> aPt = (aDirBundle ^ aPGround ) ;
       return {aPt.x(),aPt.y(),aPt.z()};
}

template <typename tUk>
      std::vector<tUk> ResidualBundle
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

           std::string FormulaName() const { return "BunleElem_Cam1_Mode" +ToStr(int(mMode));}

           std::vector<std::string>  VNamesUnknowns()  const
           {
                return   NamesP3("PGround") ;
           }

           std::vector<std::string>    VNamesObs() const
           {
               return   NamesP3("Bundle");
           };

           template <typename tUk>
                       std::vector<tUk> formula
                       (
                          const std::vector<tUk> & aVUk,
                          const std::vector<tUk> & aVObs
                       ) const
           {
                   cPtxd<tUk,3> aPGround = VtoP3(aVUk);  //
                   cPtxd<tUk,3> aDirBundle = VtoP3(aVObs);  //
                   return  ResidualBundle(aDirBundle,aPGround,mMode);
           }

         private :
              eModResBund mMode;
};

/// Class for second camera,  the base is unkwnon but unitary, the rotation is unknown
class cFormulaBundleElem_Cam2
{
   public :

       cFormulaBundleElem_Cam2(eModResBund aMode) : mMode (aMode) {}

        std::string FormulaName() const { return "BunleElem_Cam2_Mode"+ToStr(int(mMode));}

        std::vector<std::string>  VNamesUnknowns()  const
        {
             return  Append(NamesP3("PGround"),NamesP2("DuDv2"), NamesP3("Omega"));
        }

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
                cPtxd<tUk,3> aPGround = VtoP3AutoIncr(aVUk,&aIndUk);  //
                cPtxd<tUk,3> aDirBundle = VtoP3AutoIncr(aVObs,&aIndObs);  //
                cP3dNorm<tUk> aBase (aVUk,&aIndUk,aVObs,&aIndObs);
                cRot3dF<tUk>  aRot (aVUk,&aIndUk,aVObs,&aIndObs);

                return  ResidualBundle(aRot.Value(aDirBundle),aPGround-aBase.CurPt(),mMode);
        }

      private :
        eModResBund mMode;

};


};


#endif  //  _FORMULA_CALIB_BUNDLE_H_
