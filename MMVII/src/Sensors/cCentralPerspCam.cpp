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

template <class Type,const int Dim> class cDataInvertOfMapping : public cDataInvertibleMapping <Type,Dim>
{
   public :
         typedef  cDataInvertibleMapping<Type,Dim>  tIMap;
         typedef cPtxd<Type,Dim>                    tPt;
         typedef std::vector<tPt>                   tVecPt;

         cDataInvertOfMapping(const tIMap * aMapToInv,bool toAdopt);
         ~cDataInvertOfMapping();

         const  tVecPt &  Inverses(tVecPt &,const tVecPt & ) const;
         const  tVecPt &  Values(tVecPt &,const tVecPt & ) const;
   private :
         const tIMap * mMapToInv;
         bool          mAdopted;
};


template <class Type,const int Dim> 
    cDataInvertOfMapping<Type,Dim>::cDataInvertOfMapping(const tIMap * aMapToInv,bool toAdopt) :
         mMapToInv(aMapToInv),
         mAdopted (toAdopt)
{
}

template <class Type,const int Dim> 
    cDataInvertOfMapping<Type,Dim>::~cDataInvertOfMapping()
{
    if (mAdopted)
       delete mMapToInv;
}

template <class Type,const int Dim> 
         const  std::vector<cPtxd<Type,Dim>>  & 
                 cDataInvertOfMapping<Type,Dim>::Inverses(tVecPt & aVOut,const tVecPt & aVIn) const
{
    return  mMapToInv->Values(aVOut,aVIn);
}

template <class Type,const int Dim> 
         const  std::vector<cPtxd<Type,Dim>>  & 
                 cDataInvertOfMapping<Type,Dim>::Values(tVecPt & aVOut,const tVecPt & aVIn) const
{
    return  mMapToInv->Inverses(aVOut,aVIn);
}




template  class  cDataInvertOfMapping<tREAL8,2>;

/*template
template <class Type,const int Dim> class cDeformDataBoundedSet : public cDataBoundedSet<Type,Dim>
{
     public :
         typedef  cDataBoundedSet<Type,Dim>         tSetUp;
         typedef  cDataInvertibleMapping<Type,Dim>  tIMap;


};
*/



class cPixelSpace : public cDataBoundedSet<tREAL8,2>
{
      public :
           cPixelSpace(const cPt2di &aSz);
           virtual ~ cPixelSpace();
           virtual cPixelSpace *  Dup_PS () const;  ///< default work because deleted in mother class

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
         cCalibStenPerfect(const cCalibStenPerfect & aPS);  ///< default work because deleted in mother class
	 const  tVecPt &  Inverses(tVecPt &,const tVecPt & ) const override;
	 const  tVecPt &  Values(tVecPt &,const tVecPt & ) const override;

	 const tPt  &PP() const {return mPP;} ///< accessor
	 const tScal  F() const {return mF;} ///< accessor
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
                  const  cPixelSpace  &,              ///< sz, domaine of validity in pixel
		  const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
		  int aSzBuf                          ///< sz of buffers in computatio,
            );

	    ~cPerspCamIntrCalib();

	    const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;
	    // const  tVecOut &  Inverses(tVecIn &,const tVecOut & ) const;
	private :

            cPixelSpace *                        mPixSpace;   // validity domain in pixel
            // cSphereBoundedSet<tREAL8,2>          mPNormSpace; // validity domain pixel-normalize (PP/F) space

	    eProjPC                              mTypeProj;
            cDataMapCalcSymbDer<tREAL8,3,2>*     mDir_Proj;
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mDir_Dist_Val;
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mDir_Dist_Der;
	    cCalibStenPerfect                    mCSPerfect;
            // cDataMapCalcSymbDer<tREAL8,3,2>   * mProjInv;
};

cPerspCamIntrCalib::cPerspCamIntrCalib
(
      eProjPC        aTypeProj,           ///< type of projection 
      const cPt3di & aDeg,                ///< degrees of distorstion  Rad/Dec/Univ
      const std::vector<double> & aVObs,  ///< vector of constants, or void
      const cCalibStenPerfect & aCSP,           ///< Calib w/o dist
      const  cPixelSpace  & aPixSpace,              ///< sz, domaine of validity in pixel
      const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
      int aSzBuf                          ///< sz of buffers in computation
)  :
    mPixSpace       (nullptr),
    mTypeProj       (aTypeProj),
    mDir_Proj       (nullptr),
    mDir_Dist_Val   (nullptr),
    mDir_Dist_Der   (nullptr),
    mCSPerfect      (aCSP)
{
     mPixSpace =  aPixSpace.Dup_PS();
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

