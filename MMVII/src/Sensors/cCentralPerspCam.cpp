#include "include/MMVII_all.h"
#ifdef _OPENMP
#include <omp.h>
#endif


/**
   \file cCentralPerspCam.cpp

   \brief implementation  of functionnality for intrincic calibration of 
*/


namespace MMVII
{
class cPixelDomain ;
class cCalibStenPerfect ;
class cPerspCamIntrCalib ;
	
class cPixelDomain : public cDataBoundedSet<tREAL8,2>
{
      public :
           cPixelDomain(const cPt2di &aSz);
           virtual ~ cPixelDomain();
           virtual cPixelDomain *  Dup_PS () const;  ///< default work because deleted in mother class

      private :
           cBox2di    mBox;
};



class cCalibStenPerfect : public cDataInvertibleMapping<tREAL8,2>
{
     public :
         typedef tREAL8               tScal;
         typedef cPtxd<tScal,2>       tPt;
         typedef std::vector<tPt>     tVecPt;

	 cCalibStenPerfect(tScal aFoc,const tPt & aPP);
         cCalibStenPerfect(const cCalibStenPerfect & aPS);  ///< default wouldnt work because deleted in mother class
	 cCalibStenPerfect MapInverse() const;

	 tPt  Value(const tPt& aPt) const override {return mPP + aPt*mF;}
	 const  tVecPt &  Inverses(tVecPt &,const tVecPt & ) const override;
	 const  tVecPt &  Values(tVecPt &,const tVecPt & ) const override;

     private :
         tScal  mF;   ///<  Focal
         tPt    mPP;  ///<  Principal point
};




/** this the class for computing the intric calibration of perspective camera :
 
    The intric calibration compute the maping from a 3D camera coordinate to image coordinates.
    So it is a mapping R3 -> R2,   and as is heritates from cDataMapping<tREAL8,3,2>

    The mapping  is made by compination  of 3 functions :

         * mProjDir R3->R2 , the projection function, it can be stenope(x/z,y/z), fish-eye , 360 degre -> (teta,phi)  ...
	   it belongs to a finite set of  possibility code by enumeration eProjPC;  for each model it has no parameter

	 * dirtortion  R2->R2 , its a function close to identity (at least ideally)

 */

class cPerspCamIntrCalib : public cDataMapping<tREAL8,3,2>
{
	public :
            typedef tREAL8               tScal;
            typedef cPtxd<tScal,2>       tPtOut;
            typedef cPtxd<tScal,3>       tPtIn;
            typedef std::vector<tPtIn>   tVecIn;
            typedef std::vector<tPtOut>  tVecOut;

	    cPerspCamIntrCalib
            (
                  eProjPC        aTypeProj,           ///< type of projection 
		  const cPt3di & aDeg,                ///< degrees of distorstion  Rad/Dec/Univ
		  const std::vector<double> & aVObs,  ///< vector of constants, or void
		  const cCalibStenPerfect &,           ///< Calib w/o dist
                  const  cPixelDomain  &,              ///< sz, domaine of validity in pixel
		  const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
		  int aSzBuf                          ///< sz of buffers in computatio,
            );

	    ~cPerspCamIntrCalib();

	    const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;
	    // const  tVecOut &  Inverses(tVecIn &,const tVecOut & ) const;
	private :

            // cSphereBoundedSet<tREAL8,2>          mPNormSpace; // validity domain pixel-normalize (PP/F) space

	    eProjPC                              mTypeProj;
            cDataMapCalcSymbDer<tREAL8,3,2>*     mDir_Proj;
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mDir_Dist_Val;
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mDir_Dist_Der;
	    cCalibStenPerfect                    mCSPerfect;    ///<
	    cCalibStenPerfect                    mCSPInv;
            cPixelDomain *                       mPixDomain;   ///< validity domain in pixel
	    cDataMappedBoundedSet<tREAL8,2>*     mPhgrDomain;  ///<  validity in F/PP corected space,
            // cDataMapCalcSymbDer<tREAL8,3,2>   * mProjInv;
};

cPerspCamIntrCalib::cPerspCamIntrCalib
(
      eProjPC        aTypeProj,           ///< type of projection 
      const cPt3di & aDeg,                ///< degrees of distorstion  Rad/Dec/Univ
      const std::vector<double> & aVObs,  ///< vector of constants, or void
      const cCalibStenPerfect & aCSP,           ///< Calib w/o dist
      const  cPixelDomain  & aPixDomain,              ///< sz, domaine of validity in pixel
      const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
      int aSzBuf                          ///< sz of buffers in computation
)  :
    mTypeProj       (aTypeProj),
    mDir_Proj       (nullptr),
    mDir_Dist_Val   (nullptr),
    mDir_Dist_Der   (nullptr),
    mCSPerfect      (aCSP),
    mCSPInv         (mCSPerfect.MapInverse()),
    mPixDomain      (aPixDomain.Dup_PS()),
    mPhgrDomain     (new cDataMappedBoundedSet<tREAL8,2>(mPixDomain,&mCSPInv,false,false))
{
}


/* ******************************************************* */
/*                                                         */
/*                 cCalibStenPerfect                        */
/*                                                         */
/* ******************************************************* */

cCalibStenPerfect::cCalibStenPerfect(tScal aFoc,const tPt & aPP) :
    mF   (aFoc),
    mPP  (aPP)
{
}
cCalibStenPerfect::cCalibStenPerfect(const cCalibStenPerfect & aCS) :
    cCalibStenPerfect(aCS.mF,aCS.mPP)
{
}

cCalibStenPerfect cCalibStenPerfect::MapInverse() const
{
    //  aQ= PP+ aP * F  ;  aP = (aQ-PP) /aF
    return  cCalibStenPerfect(  1.0/mF  ,  -mPP/mF  );
}

const  typename cCalibStenPerfect::tVecPt &  cCalibStenPerfect::Values(tVecPt & aVOut,const tVecPt & aVIn) const
{
     const size_t aNbIn = aVIn.size();
     aVOut.resize(aNbIn);

#ifdef _OPENMP
#pragma omp parallel for
#endif
     for (size_t aK=0; aK < aNbIn; aK++) 
     {
	     aVOut[aK] = mPP + aVIn[aK] * mF;
     }
     return aVOut;
}

const  typename cCalibStenPerfect::tVecPt &  cCalibStenPerfect::Inverses(tVecPt & aVOut,const tVecPt & aVIn) const
{
     const size_t aNbIn = aVIn.size();
     aVOut.resize(aNbIn);

#ifdef _OPENMP
#pragma omp parallel for
#endif
     for (size_t aK=0; aK < aNbIn; aK++) 
     {
	     aVOut[aK] = (aVIn[aK] - mPP) / mF;
     }
     return aVOut;
}




}; // MMVII

