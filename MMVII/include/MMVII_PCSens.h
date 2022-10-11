#ifndef  _MMVII_PCSENS_H_
#define  _MMVII_PCSENS_H_


using namespace NS_SymbolicDerivative;

namespace MMVII
{
/** \file MMVII_PCSens.h
    \brief Interface class for central perspective sensors
*/

class cDefProjPerspC;
class cCalibStenPerfect ;
class cPerspCamIntrCalib ;
class cSensorCamPC ;

/**  Interface class for description of domaines where of a projection is defined in 3d-ground-space and 2-space-image */
class cDefProjPerspC
{
       public :
          /// signed belonging function for 3-d points
          virtual tREAL8  P3DIsDef(const cPt3dr &) const = 0 ;
          /// signed belonging function for 2-d points
          virtual tREAL8  P2DIsDef(const cPt2dr &) const =0 ;

          /// Radial symetry, true for physcicall based model, false for ex with equirect
          virtual bool  HasRadialSym() const ;
          static const cDefProjPerspC & ProjOfType(eProjPC);
};

/**  Class for modelisation of intrisic calibration w/o distorsion */

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

         const tScal& F()  const;   ///<  Focal
         const tPt  & PP() const;  ///<  Principal point
         tScal& F()  ;   ///<  Focal
         tPt  & PP() ;  ///<  Principal point
     private :
         tScal  mF;   ///<  Focal
         tPt    mPP;  ///<  Principal point
};

/**  helper for cPixelDomain, as the cPixelDomain must be serialisable we must separate the
 * minimal data for description, with def contructor from the more "sophisticated" object  */


class cDataPerspCamIntrCalib
{
   public :
      void AddData(const cAuxAr2007 & anAux);

      cDataPerspCamIntrCalib();
      cDataPerspCamIntrCalib
      (
	    const std::string  &   aName,
            eProjPC        aTypeProj,           ///< type of projection
            const cPt3di & aDegDir,             ///< degrees of distorstion  Rad/Dec/Univ
            const  std::vector<double> & aVParams,  ///< vector of distorsion
            const cCalibStenPerfect & aCSP,           ///< Calib w/o dist
            const  cDataPixelDomain  & aPixDomain,              ///< sz, domaine of validity in pixel
            const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
            int aSzBuf                         ///< sz of buffers in computation
      )  ;
      cDataPerspCamIntrCalib(const std::string & aName,eProjPC,const cPt3di &,double aFoc,cPt2di & aNbPix,int aSzBuf=-1);


   protected :
      std::string                    mName;
      eProjPC                        mTypeProj;           ///< type of projection
      cPt3di                         mDir_Degr;             ///< degrees of distorstion  Rad/Dec/Univ
      std::vector<cDescOneFuncDist>  mDir_VDesc;
      mutable std::vector<double>    mVTmpCopyParams;     ///< tempo copy of param, used 4 serialization
      cCalibStenPerfect              mCSPerfect;                ///< Calib w/o dist
      cDataPixelDomain               mDataPixDomain;              ///< sz, domaine of validity in pixel
      cPt3di                         mInv_Degr;       ///< degree of inverse approx by least square
      int                            mSzBuf;                         ///< sz of buffers in computation
};


/** this the class for computing the intric calibration of perspective camera :

    The intric calibration compute the maping from a 3D camera coordinate to image coordinates.
    So it is a mapping R3 -> R2,   and as is heritates from cDataMapping<tREAL8,3,2>

    The mapping  is made by compination  of 3 functions :

         * mProjDir R3->R2 , the projection function, it can be stenope(x/z,y/z), fish-eye , 360 degre -> (teta,phi)  ...
           it belongs to a finite set of  possibility code by enumeration eProjPC;  for each model it has no parameter

         * dirtortion  R2->R2 , its a function close to identity (at least ideally)

 */

class cPerspCamIntrCalib : public cDataMapping<tREAL8,3,2>,
                           public cObjWithUnkowns<tREAL8>,
			   public cDataPerspCamIntrCalib
{
        public :
            typedef tREAL8               tScal;
            typedef cPtxd<tScal,2>       tPtOut;
            typedef cPtxd<tScal,3>       tPtIn;
            typedef std::vector<tPtIn>   tVecIn;
            typedef std::vector<tPtOut>  tVecOut;

    // ================== construction of object ===============
            static cPerspCamIntrCalib * Alloc(const cDataPerspCamIntrCalib &);



                ///  Update parameter of lsq-peudso-inverse distorsion taking into account direct
            void UpdateLSQDistInv();

                /// manye delete in destructor ....
            ~cPerspCamIntrCalib();

	     void AddData(const cAuxAr2007 & anAux); ///< serialization for export
	     void  ToFile(const std::string & ) const ; ///< export in xml/dmp ...  
	     static cPerspCamIntrCalib * FromFile(const std::string &); ///< create form xml/dmp ...


    // ==================   geometric points computation ===================
            const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;
            const  tVecIn  &  Inverses(tVecIn &,const tVecOut & ) const;

            // for a point in pixel coordinates, indicate how much its invert projection is defined, not parallized !
            tREAL8  InvProjIsDef(const tPtOut & aPix ) const;


    // ==================   Accessors & Modifiers ===================
            const double & F() const;   ///< access to focal
            const cPt2dr & PP() const;  ///< acess to principal point
            const cPt3di & DegDir() const;  ///< acess to direct degrees

	    std::vector<double> & VParamDist();
	    const std::vector<double> & VParamDist() const;

            void SetThresholdPhgrAccInv(double); ///< modifier of threshold for accuracy inversion, photogrametric unit
            void SetThresholdPixAccInv(double);  ///< modifier of threshold for accuracy inversion, pixel  unit

    // ==================   Test & Bench ===================

            ///  For test, put random param while take care of being invertible
            void InitRandom(double aAmpl);
            ///  Test the accuracy of "guess" invert
            void TestInvInit(double aTolApprox,double aTolAccurate);

    // ==================   use in bundle adjustment ===================

             void PutUknowsInSetInterval() override ;///< describes its unknowns
             void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion

	     /// return calculator adapted to model of camera (degree, projection)
             cCalculator<double> * EqColinearity(bool WithDerives,int aSzBuf);

       private :
	    ///  real constructor not accessible directly, must use allocator
            cPerspCamIntrCalib(const cDataPerspCamIntrCalib &);
	     ///  big object, no valuable copy
            cPerspCamIntrCalib(const cPerspCamIntrCalib &) = delete;

         // ==================   DATA PART   ===================
	 //
                // comon to dir & inverse
            // eProjPC                              mTypeProj;
            // int                                  mSzBuf;
            const cDefProjPerspC &               mDefProj;
            cPixelDomain                         mPixDomain;              ///< sz, domaine of validity in pixel

                // ------------ parameters for direct projection  DirBundle -> pixel ------------
                //
            // cPt3di                               mDir_Degr;
            // std::vector<cDescOneFuncDist>        mDir_VDesc;  ///< contain a "high" level description of dist params
           //  std::vector<tREAL8>                  mDir_Params;    ///< Parameters of distorsion -> deprecated, redundant
            cDataMapCalcSymbDer<tREAL8,3,2>*     mDir_Proj;   ///< direct projection  R3->R2
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mDir_Dist;   ///< direct disorstion  R2->R2
            // cCalibStenPerfect                    mCSPerfect;  ///< R2-phgr -> pixels
            // cPixelDomain *                       mPixDomain;  ///< validity domain in pixel
               //  -------------  now for "inversion"  pix->DirBundle --------------------
                //
            cCalibStenPerfect                    mInv_CSP;
            cDataMappedBoundedSet<tREAL8,2>*     mPhgrDomain;  ///<  validity in F/PP corected space, initialization use mInv_CSP
            // cPt3di                               mInv_Degr;
            std::vector<cDescOneFuncDist>        mInv_VDesc;  ///< contain a "high" level description of dist params
            std::vector<tREAL8>                  mInv_Params;    ///< Parameters of distorsion
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mInvApproxLSQ_Dist;   ///< approximate LSQ invert disorstion  R2->R2
            cCalculator<tREAL8> *                mInv_BaseFDist;  ///<  base of function for inverse distortion
            cLeastSqCompMapCalcSymb<tREAL8,2,2>* mInv_CalcLSQ;  ///< structure for least square estimation
            cDataIIMFromMap<tREAL8,2> *          mDist_DirInvertible; ///< accurate inverse, use approx + iterative
            cDataMapCalcSymbDer<tREAL8,2,3>*     mInv_Proj;   ///< direct projection  R2->R3
            tREAL8                               mThreshJacPI; ///< threshlod for jacobian in pseudo inversion

            double                               mThresholdPhgrAccInv; ///< threshold for accurracy in inversion (photogram units)
            double                               mThresholdPixAccInv;  ///< threshold for accurracy in inversion (pixels    units)
            int                                  mNbIterInv;           ///< maximal number of iteration in inversion
            // cDataMapCalcSymbDer<tREAL8,3,2>   * mProjInv;
};

void AddData(const cAuxAr2007 & anAux,cPerspCamIntrCalib &);

/**  Class for modelizing the geometry of perspective-central image, contain essentially a pose (Centre+rotation)
 *   and a pointer to a (generally) shared internall calibration
 */

class cSensorCamPC : public cSensorImage
{
     public :
         typedef cIsometry3D<tREAL8>  tPose;   /// transformation Cam to Word

         cSensorCamPC(const tPose & aPose,cPerspCamIntrCalib * aCalib);
         cPt2dr Ground2Image(const cPt3dr &) const override;

	 // different accessor to the pose
         const tPose &   Pose()   const;
         const cPt3dr &  Center() const;
         cPt3dr  AxeI()   const;
         cPt3dr  AxeJ()   const;
         cPt3dr  AxeK()   const;

	 // access to tiny rotation used in bundled adjustment
         const cPt3dr &  Omega()  const;

         // interaction in unknowns
         void PutUknowsInSetInterval() override ;  // add the interval on udpate
         void OnUpdate() override;                 // "reaction" after linear update

         size_t NumXCenter() const;  /// num of Center().x when used as cObjWithUnkowns (y and z follow)
         size_t NumXOmega() const;   /// num of mOmega().x when used as cObjWithUnkowns (y and z follow)
     private :
        cSensorCamPC(const cSensorCamPC&) = delete;

        cIsometry3D<tREAL8>  mPose;   ///< transformation Cam to Word
        cPerspCamIntrCalib * mCalib;  ///< pointer to internal calibration
        cPt3dr               mOmega;  ///< vector for tiny rotation when used in unknown, mW  in code gene ...
};




};

#endif  //  _MMVII_PCSENS_H_
