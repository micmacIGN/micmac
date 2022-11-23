#ifndef  _MMVII_PCSENS_H_
#define  _MMVII_PCSENS_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_Sensor.h"
#include "MMVII_Geom3D.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{
/** \file MMVII_PCSens.h
    \brief Interface class for central perspective sensors
 *  
 *  The two main classes are :
 *     - cPerspCamIntrCalib ->  intrincic calibration
 *     - cSensorCamPC       ->  extrinsic calibration = Pose + (cPerspCamIntrCalib *)
 *
 *  The most tricky thing is read/write of this object (on xml, dmp, json ...) . Mostly because the
 *  same intrinsic calibration will be shared by several (many) cSensorCamPC, this has to be the case
 *  as in bundle, we want to have one only set of unknows for the same internal calib.
 *
 *  So the policy is as follow for reading calibration:
 *
 *     - inside a process, when reading an internal calibration, a dictionnary name->object is maintained,
 *     if the name has already been encountered the same object is returned;
 *
 *     - the cSensorCamPC must not free their internal calib, because its shared, so this freeing
 *     is done at end of this application  using the " cMMVII_Appli::AddObj2DelAtEnd" method
 *     (an altenative would have been to used shared_ptr ...)
 *
 *  For writing, identically, a memory of files having already been writen is maitained, and if the file has
 *  be already encouterd nothing is done.
 *
 *  Both decision, should be adequate for most situation and are consequently the by defauly choice. If necessary
 *  new option can be added.  For reading, when processing a file already encountered, for wich we have an existing calib,
 *  we may have 3 option :
 *
 *      - return the existing calib as is (the default choice, the only one for now)
 *      - return a new object
 *      - return the same object but update it
 *
*/

class cDefProjPerspC;  // interface for nowing if where a proj is def
class cCalibStenPerfect ;  // pure intrinsic calib (only PP + F)
class cDataPerspCamIntrCalib;  // primary data part of a cPerspCamIntrCalib -> minimal data to save in a file to be able to reconstruct a cal
class cPerspCamIntrCalib ; // object allowing computation of internal calib = cDataPerspCamIntrCalib + many calculators computed from data
class cSensorCamPC ; // Sensor for one image = Pose + pointer to cPerspCamIntrCalib

/**  Interface class for description of domaines where of a projection is defined in 3d-ground-space and 2-space-image 
 *
 *   For example, for othographic proj :
 *
 *     - 3D is def if Z>0  => P3DIsDef = Z
 *     - 2D if Def if ||P|| <1  => P2DIsDef = 1 - sqrt(X^2+Y^2) 
 * */
class cDefProjPerspC
{
       public :
          /// signed belonging function for 3-d points
          virtual tREAL8  P3DIsDef(const cPt3dr &) const = 0 ;
          /// signed belonging function for 2-d points
          virtual tREAL8  P2DIsDef(const cPt2dr &) const =0 ;

          /// Radial symetry, true for physcicall based model, false for ex with equirect
          virtual bool  HasRadialSym() const ;
	  /// return an object from its enum (for a given enum, will be always the same)
          static const cDefProjPerspC & ProjOfType(eProjPC);
};

/**  Class for modelisation of intrisic calibration w/o distorsion : essentially the mapping
 *
 *      (photogrammetric space)     Q -> PP + F * Q   (pixel space)
 *
 */

class cCalibStenPerfect : public cDataInvertibleMapping<tREAL8,2>
{
     public :
         typedef tREAL8               tScal;
         typedef cPtxd<tScal,2>       tPt;
         typedef std::vector<tPt>     tVecPt;

         cCalibStenPerfect(tScal aFoc,const tPt & aPP);
         cCalibStenPerfect(const cCalibStenPerfect & aPS);  ///< default wouldnt work because deleted in mother class
         cCalibStenPerfect MapInverse() const;

         tPt    Value(const tPt& aPt) const override {return mPP + aPt*mF;}
         tPt    Inverse(const tPt& aPt) const override {return (aPt-mPP) / mF;}
         const  tVecPt &  Inverses(tVecPt &,const tVecPt & ) const override;
         const  tVecPt &  Values(tVecPt &,const tVecPt & ) const override;

         const tScal& F()  const;   ///<  Focal
         const tPt  & PP() const;  ///<  Principal point
         tScal& F()  ;   ///<  Focal
         tPt  & PP() ;  ///<  Principal point
     private :
         tScal  mF;   ///<  Focal
         // std::string  mUnused; ///< To check if PP & F need to be consecutive; OK it works
         tPt    mPP;  ///<  Principal point
};

/**  helper for cPerspCamIntrCalib, as the cPerspCamIntrCalib must be serialisable we must separate the
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

      void PushInformation(const std::string &);
      std::vector<std::string> & VecInfo() ;

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
      std::vector<std::string>       mVectInfo;  ///< vector of potential commentarys
};


/** this the class for computing the intric calibration of perspective camera :

    The intric calibration compute the maping from a 3D camera coordinate to image coordinates.
    So it is a mapping R3 -> R2,   and as is heritates from cDataMapping<tREAL8,3,2>

    The mapping  is made by compination  of 3 functions :

         * mProjDir R3->R2 , the projection function, it can be stenope(x/z,y/z), fish-eye , 360 degre -> (teta,phi)  ...
           it belongs to a finite set of  possibility code by enumeration eProjPC;  for each model it has no parameter

         * dirtortion  R2->R2 , its a function close to identity (at least ideally)
	 
	 * cCalibStenPerfect  R2->R2  transformat additmentional unit in pixels

 */

class cPerspCamIntrCalib : public cObj2DelAtEnd,
	                   public cDataMapping<tREAL8,3,2>,
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
	     void  ToFileIfFirstime(const std::string & ) const ; ///< to avoid many write 4 same cam
	     static cPerspCamIntrCalib * FromFile(const std::string &); ///< create form xml/dmp ...
             static std::string  PrefixName() ;


    // ==================   geometric points computation ===================
            const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;
            const  tVecIn  &  Inverses(tVecIn &,const tVecOut & ) const;
	    tPtIn  Inverse(const tPtOut &) const;

            // for a point in pixel coordinates, indicate how much its invert projection is defined, not parallized !
            tREAL8  InvProjIsDef(const tPtOut & aPix ) const;


    // ==================   Accessors & Modifiers ===================
            const double & F() const;   ///< access to focal
            const cPt2dr & PP() const;  ///< acess to principal point
            const cPt3di & DegDir() const;  ///< acess to direct degrees
            const std::string & Name() const;   ///< Name of the file

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

	     void UpdateCSP();  ///< when PP/F modified

	     const cPt2di & SzPix() const;

	     /// Used by CamPC
	     double Visibility(const cPt3dr &) const ;

	     double VisibilityOnImFrame(const cPt2dr &) const;
	    ///  real constructor (accessible directly because RemanentObjectFromFile)
            cPerspCamIntrCalib(const cDataPerspCamIntrCalib &);
       private :
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

         cSensorCamPC(const std::string & aNameImage,const tPose & aPose,cPerspCamIntrCalib * aCalib);
         cPt2dr Ground2Image(const cPt3dr &) const override;

	 double Visibility(const cPt3dr &) const override;
	 double VisibilityOnImFrame(const cPt2dr &) const override;


         cPt3dr Ground2ImageAndDepth(const cPt3dr &) const override;
         cPt3dr ImageAndDepth2Ground(const cPt3dr & ) const override;


	 // different accessor to the pose
         const tPose &   Pose()   const;
         const cPt3dr &  Center() const;
         cPt3dr  AxeI()   const;
         cPt3dr  AxeJ()   const;
         cPt3dr  AxeK()   const;

	 cPerspCamIntrCalib * InternalCalib();

	 // access to tiny rotation used in bundled adjustment
         const cPt3dr &  Omega()  const;

         // interaction in unknowns
         void PutUknowsInSetInterval() override ;  // add the interval on udpate
         void OnUpdate() override;                 // "reaction" after linear update

         size_t NumXCenter() const;  /// num of Center().x when used as cObjWithUnkowns (y and z follow)
         size_t NumXOmega() const;   /// num of mOmega().x when used as cObjWithUnkowns (y and z follow)

         void AddData(const cAuxAr2007 & anAux);
	 void  ToFile(const std::string & ) const ; ///< export in xml/dmp ...  
	 static cSensorCamPC * FromFile(const std::string &); ///< create form xml/dmp ...
	 static  std::string  NameOri_From_Image(const std::string & aNameImage);

							      
         const cPt2di & SzPix() const;

         static std::string  PrefixName() ;
         std::string  V_PrefixName() const override;


     private :
        cSensorCamPC(const cSensorCamPC&) = delete;

        cIsometry3D<tREAL8>  mPose;   ///< transformation Cam to Word
        cPerspCamIntrCalib * mInternalCalib;  ///< pointer to internal calibration
        cPt3dr               mOmega;  ///< vector for tiny rotation when used in unknown, mW  in code gene ...
	std::string          mTmpNameCalib; ///< use as tmp var in addata
};




};

#endif  //  _MMVII_PCSENS_H_
