#ifndef  _MMVII_PCSENS_H_
#define  _MMVII_PCSENS_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_Sensor.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"

/*   Cx Cy Cz  Wx Wy Wz   =>  external parameters
*    F   PPx PPy          =>  Focal & Principal point
*    K1 K2 K3  ....       =>  Radial
*    b1 b2      x_0_2     =>  General
*    p1 p2 (p3 p4  ..)    =>  Decentrique
*/

using namespace NS_SymbolicDerivative;

namespace MMVII
{
extern bool BUGCAL ;


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
class cMapPProj2Im ;  // pure intrinsic calib (only PP + F)
class cMapIm2PProj ;  // pure intrinsic calib (only PP + F)
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
class cDefProjPerspC : public cDataBoundedSet<tREAL8,2> 
{
       public :
          static constexpr tREAL8 DefRhoMax = -1.0;

          /// signed belonging function for 3-d points
          virtual tREAL8  P3DIsDef(const cPt3dr &) const = 0 ;
          /// signed belonging function for 2-d points
          virtual tREAL8  P2DIsDef(const cPt2dr &) const =0 ;

	  virtual tREAL8 Insideness(const tPt &) const override;


          /// Radial symetry, true for physcicall based model, false for ex with equirect
          virtual bool  HasRadialSym() const ;
	  /// return an object from its enum (for a given enum), it teta max has, if value is default each type will adapt it to its means 
          static const cDefProjPerspC * ProjOfType(eProjPC,tREAL8 aTetaMax=DefRhoMax);

          virtual ~ cDefProjPerspC();

	  cDefProjPerspC(tREAL8 aRhoMax);
	  cDefProjPerspC(cPt2dr aRhoMax);
	  tREAL8  mRhoMax;

};

/**  Class for modelisation of intrisic calibration w/o distorsion : essentially the mapping
 *
 *      (photogrammetric space)     Q -> PP + F * Q   (pixel space)
 *
 */

class cMapPProj2Im : public cInvertMappingFromElem<cHomot2D<tREAL8> >
{
     public :
         cMapPProj2Im(tREAL8 aFoc,const tPt & aPP);
         cMapPProj2Im(const cMapPProj2Im & aPS);  ///< default wouldnt work because deleted in mother class
         cMapIm2PProj MapInverse() const;

         const tREAL8& F()  const;   ///<  Focal
         const tPt  & PP() const;  ///<  Principal point
         tREAL8& F()  ;   ///<  Focal
         tPt  & PP() ;  ///<  Principal point
     private :
};


class cMapIm2PProj :  public cInvertMappingFromElem<cHomot2D<tREAL8> >
{
    public :
         cMapIm2PProj(const cHomot2D<tREAL8> &);
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
            const cMapPProj2Im & aMapP2I,           ///< Calib w/o dist
            const  cDataPixelDomain  & aPixDomain,              ///< sz, domaine of validity in pixel
            const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
            int aSzBuf                         ///< sz of buffers in computation
      )  ;
      cDataPerspCamIntrCalib(const std::string & aName, eProjPC, const cPt3di &, double aFoc, const cPt2di &aNbPix,
                             bool PPIsRel=true,const cPt2dr & aPP = cPt2dr(0.5,0.5), int aSzBuf=-1);

      void PushInformation(const std::string &);
      std::vector<std::string> & VecInfo() ;

      const cMapPProj2Im& MapPProj2Im() const { return mMapPProj2Im;}

   protected :
      std::string                    mName;
      eProjPC                        mTypeProj;           ///< type of projection
      cPt3di                         mDir_Degr;             ///< degrees of distorstion  Rad/Dec/Univ
      std::vector<cDescOneFuncDist>  mDir_VDesc;
      mutable std::vector<double>    mVTmpCopyParams;     ///< tempo copy of param, used 4 serialization
      cMapPProj2Im                   mMapPProj2Im;                ///< Calib w/o dist
      cDataPixelDomain               mDataPixDomain;              ///< sz, domaine of validity in pixel
      cPt3di                         mInv_Degr;       ///< degree of inverse approx by least square
      int                            mSzBuf;                         ///< sz of buffers in computation
      std::vector<std::string>       mVectInfo;  ///< vector of potential commentarys
};

void AddData(const cAuxAr2007 & anAux,cDataPerspCamIntrCalib & aPCIC);



/** this the class for computing the intric calibration of perspective camera :

    The intric calibration compute the maping from a 3D camera coordinate to image coordinates.
    So it is a mapping R3 -> R2,   and as is heritates from cDataMapping<tREAL8,3,2>

    The mapping  is made by compination  of 3 functions :

         * mProjDir R3->R2 , the projection function, it can be stenope(x/z,y/z), fish-eye , 360 degre -> (teta,phi)  ...
           it belongs to a finite set of  possibility code by enumeration eProjPC;  for each model it has no parameter

         * dirtortion  R2->R2 , its a function close to identity (at least ideally)
	 
	 * cMapPProj2Im  R2->R2  transformat additmentional unit in pixels

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

	    /**  Generate random calib, with assurance that distorsion will be inverible, 
	       the KDeg (in 0,1,2)  pick one of the pre-defined degree */
            static cPerspCamIntrCalib * RandomCalib(eProjPC aTypeProj,int aKDeg);


	    /** Given a set of 2D-3D correspondance, make a pose estimation using space resection,  
                   NbTriplet :  number of triplet tested in ransac
                   Real8     :  is the internal computation done on 8/16 byte floating point (theorically more accuracy)
                   aNbPtsMeasures : possibly reduce the number of point for measuring accuracy (not for triplet selection)

		   return "best" pose + score as Avg Residual (angular not pixel)
            */
             cWhichMin<tPoseR,tREAL8>  RansacPoseEstimSpaceResection
		    (const cSet2D3D & aSet0,size_t aNbTriplet,bool Real8=true, int aNbPtsMeasures=-1,cTimerSegm * =nullptr);

	    /**  Acces to the elementay space resection method : get a list of pose corresponding to a triplet of 2D-3D corresp*/
	    std::list<tPoseR >  ElemPoseEstimSpaceResection(const cPair2D3D&,const cPair2D3D&,const cPair2D3D&);

                ///  Update parameter of lsq-peudso-inverse distorsion taking into account direct
            void UpdateLSQDistInv();

            void UpdateLSQDistIfRequired() const;

                /// manye delete in destructor ....
            ~cPerspCamIntrCalib();

	     void AddData(const cAuxAr2007 & anAux); ///< serialization for export
	     void  ToFile(const std::string & ) const ; ///< export in xml/dmp ...  
	     void  ToFileIfFirstime(const std::string & ) const ; ///< to avoid many write 4 same cam
	     static cPerspCamIntrCalib * FromFile(const std::string &,bool Remanent=true); ///< create form xml/dmp ...
	     /// Prefix used for calib per image
             static std::string  PrefixName() ;
	     /// Prefix used for shared calibration
             static std::string  SharedCalibPrefixName() ;


    // ==================   geometric points computation ===================
            const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;
            const  tVecIn  &  DirBundles(tVecIn &,const tVecOut & ) const;
	    tPtIn  DirBundle(const tPtOut &) const;

            // for a point in pixel coordinates, indicate how much its invert projection is defined, not parallized !
            tREAL8  InvProjIsDef(const tPtOut & aPix ) const;

	    /** return the value of an undistorded camera, note tha with fisheyes it can trow points to infinite ... but
	     dont see what can be done ... */
	    tPtOut Undist(const tPtOut &) const;



    // ==================   Accessors & Modifiers ===================
            const double & F() const;   ///< access to focal
            const cPt2dr & PP() const;  ///< acess to principal point
            const cPt3di & DegDir() const;  ///< acess to direct degrees
            const std::string & Name() const;   ///< Name of the file
            void SetName(const std::string &) ; ///< Change the name
 
            eProjPC TypeProj() const;           ///< type of projection

	    const std::vector<double> & VParamDist() const;  ///< vector of dist param
	    std::vector<double> & VParamDist();    ///< vector of dist param
            const   std::vector<cDescOneFuncDist> &  VDescDist() const;  ///< desc of dist param
	           //  ===  Acess to individuald dist values
	    int IndParamDistFromName(const std::string&,bool SVP=false) const; ///< get index of param from its name, -1 if none & SVP

	    ///  List of adresses of parameters that contain
	    void  GetAdrInfoParam(cGetAdrInfoParam<tREAL8> &) override;
	    double  ParamDist(const std::string &) const; ///< recover param of dist from its name
	    void    SetParamDist(const std::string &,const double &) ; ///< set  value of dist from its name
	    bool    IsNameParamDist(const std::string &) const;  ///< Is it a valuable name of distosion param

            void SetThresholdPhgrAccInv(double); ///< modifier of threshold for accuracy inversion, photogrametric unit
            void SetThresholdPixAccInv(double);  ///< modifier of threshold for accuracy inversion, pixel  unit

            const cDataMapping<tREAL8,3,2>* Dir_Proj() const; ///< access to direct projection as a cDataMapping FOR_PYTHON
            const cDataMapping<tREAL8,2,2>* Dir_Dist() const; ///< access to direct distorsion as a cDataMapping FOR_PYTHON
            const cDataMapping<tREAL8,2,3>* Inv_Proj() const; ///< access to inverse projection as a cDataMapping FOR_PYTHON
            const cDataInvertibleMapping<tREAL8,2>* Dir_DistInvertible() const; ///< access to inverse distorsion as a cDataMapping FOR_PYTHON

            /// point on grid  InPixel -> apply or not F/PP
	    std::vector<cPt2dr>  PtsSampledOnSensor(int aNbByDim,bool InPixel) const ;

	    const cPixelDomain & PixelDomain() const ;

    // ==================   Test & Bench ===================

            ///  For test, put random param while take care of being invertible
            void InitRandom(double aAmpl);
            ///  Test the accuracy of "guess" invert
            void TestInvInit(double aTolApprox,double aTolAccurate);

    // ==================   use in bundle adjustment ===================

             void PutUknowsInSetInterval() override ;///< describes its unknowns
             void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion

	     /// return calculator adapted to model of camera (degree, projection)
             cCalculator<double> * EqColinearity(bool WithDerives,int aSzBuf,bool ReUse);

	     void UpdateMapProj2Im();  ///< when PP/F modified

	     const cPt2di & SzPix() const;

	     /// Used by CamPC
	     double DegreeVisibility(const cPt3dr &) const ;

	     double DegreeVisibilityOnImFrame(const cPt2dr &) const;


	    ///  real constructor (accessible directly because RemanentObjectFromFile)
            cPerspCamIntrCalib(const cDataPerspCamIntrCalib &);
	    
	    /// For inversion, or sampling point, we need seed that is +- corresponding of sensor midle, befor dist
	    cPt2dr PtSeedInv() const;

       private :
	     ///  big object, no valuable copy
            cPerspCamIntrCalib(const cPerspCamIntrCalib &) = delete;

	    void CheckBeforeInverse(const tVecOut &) const;
	    void CheckBeforeInverse(const tPtOut & ) const;


         // ==================   DATA PART   ===================
	 //
                // comon to dir & inverse
            // eProjPC                              mTypeProj;
            // int                                  mSzBuf;
	    bool                                 mVoidDist;  /// special behavior is requires with deg=[0,0,0] 
            const cDefProjPerspC *               mDefProj;    ///  Prof function
            cPixelDomain                         mPixDomain;              ///< sz, domaine of validity in pixel

                // ------------ parameters for direct projection  DirBundle -> pixel ------------
                //
            // cPt3di                               mDir_Degr;
            // std::vector<cDescOneFuncDist>        mDir_VDesc;  ///< contain a "high" level description of dist params
           //  std::vector<tREAL8>                  mDir_Params;    ///< Parameters of distorsion -> deprecated, redundant
            cDataMapCalcSymbDer<tREAL8,3,2>*     mDir_Proj;   ///< direct projection  R3->R2
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mDir_Dist;   ///< direct disorstion  R2->R2
               //  -------------  now for "inversion"  pix->DirBundle --------------------
                //
            cMapIm2PProj                         mMapIm2PProj;
            cDataMappedBoundedSet<tREAL8,2>*     mPhgrDomain;  ///<  validity in F/PP corected space, initialization use mInv_CSP
            // cPt3di                               mInv_Degr;
            std::vector<cDescOneFuncDist>        mInv_VDesc;  ///< contain a "high" level description of dist params
            std::vector<tREAL8>                  mInv_Params;    ///< Parameters of distorsion
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mInvApproxLSQ_Dist;   ///< approximate LSQ invert disorstion  R2->R2
            cCalculator<tREAL8> *                mInv_BaseFDist;  ///<  base of function for inverse distortion
            cLeastSqCompMapCalcSymb<tREAL8,2,2>* mInv_CalcLSQ;  ///< structure for least square estimation


            double                               mThresholdPhgrAccInv; ///< threshold for accurracy in inversion (photogram units)
            double                               mThresholdPixAccInv;  ///< threshold for accurracy in inversion (pixels    units)
            int                                  mNbIterInv;           ///< maximal number of iteration in inversion
            tREAL8                               mThreshJacPI; ///< threshlod for jacobian in pseudo inversion

            cDataIIMFromMap<tREAL8,2> *          mDist_DirInvertible; ///< accurate inverse, use approx + iterative
            cDataMapCalcSymbDer<tREAL8,2,3>*     mInv_Proj;   ///< direct projection  R2->R3
            bool                                 mInvIsUpToDate;        

            // cDataMapCalcSymbDer<tREAL8,3,2>   * mProjInv;
};

void AddData(const cAuxAr2007 & anAux,cPerspCamIntrCalib &);

/** Class for modelising a pose when it is used as unknwon in non linear system
 */

class cPoseWithUK :  public cObjWithUnkowns<tREAL8>
{
     public :
         /// Fill with dummy value for case where default constructor is required
	 cPoseWithUK();

	 void PushObs(std::vector<double> &,bool TransposeMatr);

	 cPoseWithUK(const tPoseR & aPose);
         void SetPose(const tPoseR & aPose);

	 // different accessor to the pose
         const tPoseR &   Pose()   const;
         const cPt3dr &  Center() const;
         const cPt3dr &  Tr() const;
         const cPt3dr &  Omega() const;
         cPt3dr &  Omega() ;
         cPt3dr &  Center() ;
         cPt3dr &  Tr() ;
         cPt3dr  AxeI()   const;
         cPt3dr  AxeJ()   const;
         cPt3dr  AxeK()   const;
         tPoseR &   Pose()   ;

	 /// The interval is specified as the object can be used as helperto to other classes
         void PutUknowsInSetInterval(cSetInterUK_MultipeObj<tREAL8> * aSetInterv) ;  
         void OnUpdate() override;                 // "reaction" after linear update
	 void  GetAdrInfoParam(cGetAdrInfoParam<tREAL8> &) override;
	 // std::vector<tPtrOUK>  GetAllUK() override;

     private :
         void PutUknowsInSetInterval() override ;  // add the interval on udpate

         tPoseR     mPose;   ///< transformation Cam to Word
         cPt3dr     mOmega;  ///< vector for tiny rotation when used in unknown, mW  in code gene ...
};
void AddData(const  cAuxAr2007 & anAux,cPoseWithUK & aPUK);


/**  Class for modelizing the geometry of perspective-central image, contain essentially a pose (Centre+rotation)
 *   and a pointer to a (generally) shared internall calibration
 */
class cSensorCamPC : public cSensorImage
{
     public :
	 typedef cObjWithUnkowns<tREAL8> * tPtrOUK;
         typedef cIsometry3D<tREAL8>  tPose;   /// transformation Cam to Word

         cSensorCamPC(const std::string & aNameImage,const tPose & aPose,cPerspCamIntrCalib * aCalib);

	 /// specialize chang sys 
         cSensorCamPC * PCChangSys(cDataInvertibleMapping<tREAL8,3> &) const ;
	 /// generic chang sys  (just call specialized)
         cSensorImage * SensorChangSys(cDataInvertibleMapping<tREAL8,3> &) const override ;

         /// Create form  Un-Calibrated-Space-Resection
         static cSensorCamPC * CreateUCSR(const cSet2D3D&,const cPt2di & aSzCam,const std::string&,bool Real16=true);

         cPt2dr Ground2Image(const cPt3dr &) const override;

	 double DegreeVisibility(const cPt3dr &) const override;
	 double DegreeVisibilityOnImFrame(const cPt2dr &) const override;

	 const cPixelDomain & PixelDomain() const override;

         void SetPose(const tPose & aPose);

         cPt3dr Ground2ImageAndDepth(const cPt3dr &) const override;
         cPt3dr ImageAndDepth2Ground(const cPt3dr & ) const override;
         tSeg3dr  Image2Bundle(const cPt2dr &) const override;


         std::vector<cPt2dr>  PtsSampledOnSensor(int aNbByDim) const override;

         ///  residual of projection as angle between directions, work with any lenses
         tREAL8  AngularProjResiudal(const cPair2D3D&) const;
         ///  average of AngularProjResiudal
         tREAL8  AvgAngularProjResiudal(const cSet2D3D&) const;

	 ///  real center
	 cPt3dr  PseudoCenterOfProj() const override;

	 const cPt3dr * CenterOfPC() const override;
         /// Return the calculator, adapted to the type, for computing colinearity equation
         cCalculator<double> * EqColinearity(bool WithDerives,int aSzBuf,bool ReUse) override;
	 /// Push the current rotation, as equation are fixed using delta-rot
	 void PushOwnObsColinearity( std::vector<double> &) override;


	 /// return the pose of aCam2 relatively to Cam1
	 tPose RelativePose(const cSensorCamPC& aCame) const;


	 // different accessor to the pose
         const tPose &   Pose()   const;
         const cPt3dr &  Center() const;
         cPt3dr &  Center() ;
         cPt3dr  AxeI()   const;
         cPt3dr  AxeJ()   const;
         cPt3dr  AxeK()   const;
	 cPoseWithUK & Pose_WU();

	 cPerspCamIntrCalib * InternalCalib() const;

	 // access to tiny rotation used in bundled adjustment
         const cPt3dr &  Omega()  const;
         cPt3dr &  Omega()  ;

         // interaction in unknowns
         void PutUknowsInSetInterval() override ;  // add the interval on udpate
         void OnUpdate() override;                 // "reaction" after linear update

	 /// contain itself + internal calib
	 std::vector<tPtrOUK>  GetAllUK() override;

	 /// retur
	 void  GetAdrInfoParam(cGetAdrInfoParam<tREAL8> &) override;


         size_t NumXCenter() const;  /// num of Center().x when used as cObjWithUnkowns (y and z follow)
         size_t NumXOmega() const;   /// num of mOmega().x when used as cObjWithUnkowns (y and z follow)

         void AddData(const cAuxAr2007 & anAux);
	 void  ToFile(const std::string & ) const override ; ///< export in xml/dmp ...  
	 static cSensorCamPC * FromFile(const std::string &,bool Remanent=true); ///< create form xml/dmp ...
	 static  std::string  NameOri_From_Image(const std::string & aNameImage);

							      
         const cPt2di & SzPix() const;

         static std::string  PrefixName() ;
         std::string  V_PrefixName() const override;

         static void BenchOneCalib(cPerspCamIntrCalib * aCalib);

	 cPt3dr  Pt_L2W(const cPt3dr &) const;  ///< Coordinat local of cam to coordinate word for a "point"
	 cPt3dr  Pt_W2L(const cPt3dr &) const;  ///< Coordinat word to coordinate local of cam for a "point"
	 cPt3dr  Vec_L2W(const cPt3dr &) const;  ///< Coordinat local of cam to coordinate word for a "vector"
	 cPt3dr  Vec_W2L(const cPt3dr &) const;  ///< Coordinat word to coordinate local of cam for a "vector"

     private :
        void Bench();
        cSensorCamPC(const cSensorCamPC&) = delete;

	cPoseWithUK          mPose_WU;
	/*
        cIsometry3D<tREAL8>  mPose;   ///< transformation Cam to Word
        cPt3dr               mOmega;  ///< vector for tiny rotation when used in unknown, mW  in code gene ...
	*/
        cPerspCamIntrCalib * mInternalCalib;  ///< pointer to internal calibration
	std::string          mTmpNameCalib; ///< use as tmp var in addata
};


class cCamSimul : public cMemCheck
{
   public :
      static cCamSimul * Alloc2VIewTerrestrial(eProjPC aProj1,eProjPC aProj2,bool SubVert);

      ~cCamSimul();

      cPt3dr mCenterGround;

      //   Geometry of acquisition
      tREAL8 mProfMin;
      tREAL8 mProfMax;
      tREAL8 mBsHMin;
      tREAL8 mBsHMax;
      tREAL8 mRandInterK;

      static void BenchMatEss(cTimerSegm * aTS,bool PerfInter);

      void TestCam(cSensorCamPC * aCam) const;
   private :
      void AddCam(cPerspCamIntrCalib *,bool SubVert);
      void AddCam(eProjPC aProj1,bool SubVert);

      cCamSimul();
      ///  is the new center sufficiently far, but not too much
      bool ValidateCenter(const cPt3dr & aP) const;

      ///  Generatea new valide point
      cPt3dr  GenValideCenter(bool SubVert) const;
      /// Generate a point w/o constraint
      cPt3dr  GenAnyCenter(bool SubVert) const;

      std::vector<cSensorCamPC *>         mListCam;
      std::vector<cPerspCamIntrCalib *>   mListCalib;


      // cSetHomogCpleIm
};

};

#endif  //  _MMVII_PCSENS_H_
