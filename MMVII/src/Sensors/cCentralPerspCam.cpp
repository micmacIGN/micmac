#include "include/MMVII_all.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/*
 CamStenope * Std_Cal_From_File
             (
                 const std::string & aNameFile,
                 const std::string &  aNameTag = "CalibrationInternConique"
             );
*/

/**
   \file cCentralPerspCam.cpp

   \brief implementation  of functionnality for intrincic calibration of 
*/

using namespace NS_SymbolicDerivative;

namespace MMVII
{

class cPixelDomain ;
class cCalibStenPerfect ;
class cPerspCamIntrCalib ;

template <class Type> class cOneInteralvUnkown;
template <class Type> class cSetIntervUK_OneObj;
template <class Type> class cSetInterUK_MultipeObj;
template <class Type> class cObjWithUnkowns;

template <class Type> class cOneInteralvUnkown
{
     public :
        Type * mVUk;
	size_t mNb;
	cOneInteralvUnkown(Type * aVUk,size_t aNb)  : mVUk (aVUk) , mNb (aNb) {}
};

template <class Type> class cSetIntervUK_OneObj
{
     public :
	 cSetIntervUK_OneObj(cObjWithUnkowns<Type>   *anObj) : mObj (anObj) {}

         cObjWithUnkowns<Type>   *         mObj;
         std::vector<cOneInteralvUnkown<Type>>   mVInterv;

};

template <class Type> class cSetInterUK_MultipeObj
{
        public :

           cSetInterUK_MultipeObj();

           void  AddOneObj(cObjWithUnkowns<Type> *);
	   std::vector<Type>  VUnKnowns();

	   void AddOneInterv(Type * anAdr,size_t aSz) ;
	   void AddOneInterv(std::vector<Type> & aV) ;

        private :
           void IO_UnKnowns(std::vector<Type> & aV,bool forExport);
           std::vector<cSetIntervUK_OneObj<Type> >  mVVInterv;
	   size_t                                    mNbUk;
};

template <class Type> class cObjWithUnkowns
{
       public :
	       virtual void SetUnknowns(cSetInterUK_MultipeObj<Type> &) = 0;
       private :
};


/* ******************************** */
/*       cSetInterUK_MultipeObj     */
/* ******************************** */

template <class Type> void  cSetInterUK_MultipeObj<Type>::AddOneObj(cObjWithUnkowns<Type> * anObj)
{
	mVVInterv.push_back(cSetIntervUK_OneObj<Type>(anObj));
	anObj->SetUnknowns(*this);
}

template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(Type * anAdr,size_t aSz) 
{
    mNbUk += aSz;
    mVVInterv.back().mVInterv.push_back(cOneInteralvUnkown<Type>(anAdr,aSz));
}
template <class Type> void cSetInterUK_MultipeObj<Type>::AddOneInterv(std::vector<Type> & aV)
{
    AddOneInterv(aV.data(),aV.size());
} 

template <class Type> cSetInterUK_MultipeObj<Type>::cSetInterUK_MultipeObj() :
    mNbUk (0)
{
}

template <class Type> void cSetInterUK_MultipeObj<Type>::IO_UnKnowns(std::vector<Type> & aVect,bool forExport)
{
    size_t anIndex=0;

    for (const auto &   aVinterv : mVVInterv) // parse object
    {
        for (const auto & anInterv : aVinterv) // parse interv of 1 object
	{
            for (size_t aK=0 ; aK<anInterv.mNb ; aK++)  // parse element of the interv
	    { 
                Type & aVal = aVect(anIndex++);
		if (forExport)
                    aVal =  anInterv.mVUk[aK];
		else
                    anInterv.mVUk[aK] = aVal;
	    }
	}
    }
}

template <class Type> std::vector<Type>  cSetInterUK_MultipeObj<Type>::VUnKnowns()
{
    std::vector<Type> aRes(mNbUk);
    IO_UnKnowns(aRes,true);
    /*
    size_t anIndex=0;

    for (const auto &   aVinterv : mVVInterv) // parse object
    {
        for (const auto & anInterv : aVinterv) // parse interv of 1 object
	{
            for (size_t aK=0 ; aK<anInterv.mNb ; aK++)  // parse element of the interv
                aRes(anIndex++) =  anInterv.mVUk[aK];
	}
    }
    */

    return aRes;
}

template class cObjWithUnkowns<tREAL8>;




	
class cPixelDomain : public cDataBoundedSet<tREAL8,2>
{
      public :
           cPixelDomain(const cPt2di &aSz);
           virtual ~ cPixelDomain();
           virtual cPixelDomain *  Dup_PS () const;  ///< default work because deleted in mother class

	   const cPt2di & Sz() const;
      private :
           cPt2di     mSz;
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

         const tScal& F()  const;   ///<  Focal
         const tPt  & PP() const;  ///<  Principal point
         tScal& F()  ;   ///<  Focal
         tPt  & PP() ;  ///<  Principal point
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

class cPerspCamIntrCalib : public cDataMapping<tREAL8,3,2>,
	                   public cObjWithUnkowns<tREAL8>
{
	public :
            typedef tREAL8               tScal;
            typedef cPtxd<tScal,2>       tPtOut;
            typedef cPtxd<tScal,3>       tPtIn;
            typedef std::vector<tPtIn>   tVecIn;
            typedef std::vector<tPtOut>  tVecOut;

    // ================== construction of object ===============
	    cPerspCamIntrCalib
            (
                  eProjPC        aTypeProj,           ///< type of projection 
		  const cPt3di & aDeg,                ///< degrees of distorstion  Rad/Dec/Univ
		  const std::vector<double> & aVParams,  ///< vector of distorsion
		  const cCalibStenPerfect &,           ///< Calib w/o dist
                  const  cPixelDomain  &,              ///< sz, domaine of validity in pixel
		  const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
		  int aSzBuf                          ///< sz of buffers in computatio,
            );


	    // allocation with minimal number of parameters
	    cPerspCamIntrCalib(eProjPC,const cPt3di &,double aFoc,cPt2di & aNbPix,int aSzBuf=-1);

	        ///  Update parameter of lsq-peudso-inverse distorsion taking into account direct
	    void UpdateLSQDistInv();

	        /// manye delete in destructor ....
	    ~cPerspCamIntrCalib();

    // ==================   points computation ===================
	    const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;
	    const  tVecIn  &  Inverses(tVecIn &,const tVecOut & ) const;

	    // for a point in pixel coordinates, indicate how much its invert projection is defined, not parallized !
	    tREAL8  InvProjIsDef(const tPtOut & aPix ) const;

    // ==================   Accessors & Modifiers ===================
	    const double & F() const;   ///< access to focal
	    const cPt2dr & PP() const;  ///< acess to principal point

	    void SetThresholdPhgrAccInv(double); ///< modifier of threshold for accuracy inversion, photogrametric unit
	    void SetThresholdPixAccInv(double);  ///< modifier of threshold for accuracy inversion, pixel  unit


    // ==================   Test & Bench ===================
		
	    ///  For test, put random param while take care of being invertible
	    void InitRandom(double aAmpl);
	    ///  Test the accuracy of "guess" invert
	    void TestInvInit(double aTolApprox,double aTolAccurate);
	    
    // ==================   use in bundle adjustment ===================

	     void SetUnknowns(cSetInterUK_MultipeObj<tREAL8> &) override ;
	private :
            cPerspCamIntrCalib(const cPerspCamIntrCalib &) = delete;

	        // comon to dir & inverse
	    eProjPC                              mTypeProj;
            int                                  mSzBuf;
	    const cDefProjPerspC &               mDefProj;
	        // parameters for direct projection  DirBundle -> pixel
	    cPt3di                               mDir_Degr;
	    std::vector<cDescOneFuncDist>        mDir_VDesc;  ///< contain a "high" level description of dist params
	    std::vector<tREAL8>                  mDir_Params;    ///< Parameters of distorsion
            cDataMapCalcSymbDer<tREAL8,3,2>*     mDir_Proj;   ///< direct projection  R3->R2
            cDataNxNMapCalcSymbDer<tREAL8,2>*    mDir_Dist;   ///< direct disorstion  R2->R2
	    cCalibStenPerfect                    mCSPerfect;  ///< R2-phgr -> pixels
            cPixelDomain *                       mPixDomain;  ///< validity domain in pixel
                // now for "inversion"  pix->DirBundle
	    cCalibStenPerfect                    mInv_CSP;
	    cDataMappedBoundedSet<tREAL8,2>*     mPhgrDomain;  ///<  validity in F/PP corected space,
	    cPt3di                               mInv_Degr;
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

//       cDataIIMFromMap(tMap aMap,const tPt &,tMap aRoughInv,const Type& aDistTol,int aNbIterMax);


class cSensorImage  :  public cObjWithUnkowns<tREAL8>
{
     public :
         virtual cPt2dr Ground2Image(const cPt3dr &) const = 0;
};

class cSensorCamPC : public cSensorImage
{
     public :
	 typedef cIsometry3D<tREAL8>  tPose;   /// transformation Cam to Word
	 void SetUnknowns(cSetInterUK_MultipeObj<tREAL8> &) override ;


	 cSensorCamPC(const tPose & aPose,cPerspCamIntrCalib * aCalib);
         cPt2dr Ground2Image(const cPt3dr &) const override;

	 cPt3dr  Center() const;
	 cPt3dr  AxeI()   const;
	 cPt3dr  AxeJ()   const;
	 cPt3dr  AxeK()   const;
	 tPose   Pose()   const;
     private :
	cSensorCamPC(const cSensorCamPC&) = delete;
	     
        cPerspCamIntrCalib * mCalib;
	cIsometry3D<tREAL8>  mPose;   /// transformation Cam to Word
};

cPt2dr cSensorCamPC::Ground2Image(const cPt3dr & aP) const 
{
	//  mPose(0,0,0) = Center, then mPose Cam->Word, then we use Inverse, BTW Inverse is as efficient as direct
	return mCalib->Value(mPose.Inverse(aP));
}

cPt3dr cSensorCamPC::Center() const {return mPose.Tr();}
cPt3dr cSensorCamPC::AxeI() const {return mPose.Rot().AxeI();}
cPt3dr cSensorCamPC::AxeJ() const {return mPose.Rot().AxeI();}
cPt3dr cSensorCamPC::AxeK() const {return mPose.Rot().AxeJ();}

/* ******************************************************* */
/*                                                         */
/*                 cPerspCamIntrCalib                      */
/*                                                         */
/* ******************************************************* */

cPerspCamIntrCalib::cPerspCamIntrCalib
(
      eProjPC        aTypeProj,           ///< type of projection 
      const cPt3di & aDegDir,             ///< degrees of distorstion  Rad/Dec/Univ
      const std::vector<double> & aVParams,  ///< vector of constants, or void
      const cCalibStenPerfect & aCSP,           ///< Calib w/o dist
      const  cPixelDomain  & aPixDomain,              ///< sz, domaine of validity in pixel
      const cPt3di & aDegPseudoInv,       ///< degree of inverse approx by least square
      int aSzBuf                          ///< sz of buffers in computation
)  :
	// ------------ global -------------
    mTypeProj           (aTypeProj),
    mSzBuf              (aSzBuf),
    mDefProj            (cDefProjPerspC::ProjOfType(mTypeProj)),
	// ------------ direct -------------
    mDir_Degr           (aDegDir),
    mDir_VDesc          (DescDist(aDegDir)),
    mDir_Params         (aVParams),
    mDir_Proj           (nullptr),
    mDir_Dist           (nullptr),
    mCSPerfect          (aCSP),
    mPixDomain          (aPixDomain.Dup_PS()),
	// ------------ inverse -------------
    mInv_CSP            (mCSPerfect.MapInverse()),
    mPhgrDomain         (new cDataMappedBoundedSet<tREAL8,2>(mPixDomain,&mInv_CSP,false,false)),
    mInv_Degr           (aDegPseudoInv),
    mInv_VDesc          (DescDist(mInv_Degr)),
    mInv_Params         (mInv_VDesc.size(),0.0),
    mInvApproxLSQ_Dist  (nullptr),
    mInv_BaseFDist      (nullptr),
    mInv_CalcLSQ        (nullptr),
    mDist_DirInvertible (nullptr),
    mInv_Proj           (nullptr),
    mThreshJacPI        (0.5),
    mNbIterInv          (10)
{
     SetThresholdPixAccInv(1e-3);

        // 1 - construct direct parameters
	
    // correct vect param, when first use, parameter can be empty meaning all 0  
    if (mDir_Params.size() != mDir_VDesc.size())
    {
       MMVII_INTERNAL_ASSERT_strong(mDir_Params.empty(),"cPerspCamIntrCalib Bad size for params");
       mDir_Params.resize(mDir_VDesc.size(),0.0);
    }
    
    mDir_Proj = new  cDataMapCalcSymbDer<tREAL8,3,2>
                     (
                          EqCPProjDir(mTypeProj,false,mSzBuf),   // equatio, w/o derivative
                          EqCPProjDir(mTypeProj,true,mSzBuf),    // equation with derivatives
			  std::vector<double>(),                 // parameters, empty here
			  true                                   // equations are "adopted" (i.e will be deleted in destuctor)
                     );

    // TO CHANGE SUPRRESS DIR PARAM GET ACESS TO mDir_Dist
    MMVII_WARGING("TO CHANGE SUPRRESS DIR PARAM GET ACESS TO mDir_Dist");
    mDir_Dist = NewMapOfDist(mDir_Degr,mDir_Params,mSzBuf);

        // 2 - construct direct parameters

}

cPerspCamIntrCalib::cPerspCamIntrCalib(eProjPC aTypeProj,const cPt3di & aDeg,double aFoc,cPt2di & aNbPix,int aSzBuf) :
	cPerspCamIntrCalib
	(
	    aTypeProj, 
	    aDeg,
	    std::vector<double>(),
            cCalibStenPerfect(aFoc,ToR(aNbPix)/2.0),
            cPixelDomain(aNbPix),
	    aDeg,
	    aSzBuf
	)
{
}

cPerspCamIntrCalib::~cPerspCamIntrCalib()
{
     delete mPhgrDomain;	
     delete mPixDomain;	
     delete mDir_Dist;
     delete mDir_Proj;

     delete mInvApproxLSQ_Dist;
     delete mInv_BaseFDist;
     delete mInv_CalcLSQ;
     delete mDist_DirInvertible;
     delete mInv_Proj;
}


void cPerspCamIntrCalib::SetUnknowns(cSetInterUK_MultipeObj<tREAL8> & aSet) 
{
    aSet.AddOneInterv(&mCSPerfect.F(),3);
    aSet.AddOneInterv(mDir_Dist->VObs());
	/*
     aVect.push_back(&mCSPerfect.F());
     aVect.push_back(&mCSPerfect.PP().x());
     aVect.push_back(&mCSPerfect.PP().y());

     for (auto & aCoef :  mDir_Dist->VObs())
        aVect.push_back(&aCoef);
	*/
}
	     

	     //  geometric manips

const  std::vector<cPt2dr> &  cPerspCamIntrCalib::Values(tVecOut & aV3 ,const tVecIn & aV0 ) const 
{
     static tVecOut aV1,aV2;
     mDir_Proj->Values(aV1,aV0);
     mDir_Dist->Values(aV2,aV1);
     mCSPerfect.Values(aV3,aV2);
     
     return aV3;
}


const  std::vector<cPt3dr> &  cPerspCamIntrCalib::Inverses(tVecIn & aV3 ,const tVecOut & aV0 ) const 
{
     static tVecOut aV1,aV2;
     mInv_CSP.Values(aV1,aV0);
     mDist_DirInvertible->Inverses(aV2,aV1);
     mInv_Proj->Values(aV3,aV2);
     
     return aV3;
}

tREAL8  cPerspCamIntrCalib::InvProjIsDef(const tPtOut & aPix ) const
{
    return mDefProj.P2DIsDef(mDist_DirInvertible->Inverse(mInv_CSP.Value(aPix)));
}


void cPerspCamIntrCalib::UpdateLSQDistInv()
{
    // allocate obect, just need to be done once
    if (mInvApproxLSQ_Dist==nullptr)
    {
        mInvApproxLSQ_Dist  = NewMapOfDist(mInv_Degr,mInv_Params,mSzBuf);
        mInv_BaseFDist = EqBaseFuncDist(mInv_Degr,mSzBuf);
        mInv_CalcLSQ   = new cLeastSqCompMapCalcSymb<tREAL8,2,2>(mInv_BaseFDist);
	mDist_DirInvertible = new   cDataIIMFromMap<tREAL8,2> (mDir_Dist,mInvApproxLSQ_Dist,mThresholdPhgrAccInv,mNbIterInv,false,false);

        mInv_Proj = new  cDataMapCalcSymbDer<tREAL8,2,3>
                         (
                              EqCPProjInv(mTypeProj,false,mSzBuf),   // equatio, w/o derivative
                              EqCPProjInv(mTypeProj,true,mSzBuf),    // equation with derivatives
			      std::vector<double>(),                 // parameters, empty here
			      true                                   // equations are "adopted" (i.e will be deleted in destuctor)
                         );
    }

    // create structure for map inversion
    cComputeMapInverse aCMI
    (
       mThreshJacPI,         ///< Threshold on jacobian to ensure inversability
       cPt2dr(0,0),          ///< Seed point, in input space
       mInv_VDesc.size(),    ///< Approximate number of point (in the biggest size), here +or- less square of minimum
       (*mPhgrDomain),       ///< Set of validity, in output space
       (*mDir_Dist),         ///< Maping to invert : InputSpace -> OutputSpace
       (* mInv_CalcLSQ),     ///< Structure for computing the invert on base of function using least square
       false                 ///< Not in  Test
   );
   aCMI.DoAll(mInv_Params); // compute the parameters
   mInvApproxLSQ_Dist->SetObs(mInv_Params); // set these parameters in approx inverse
}

void cPerspCamIntrCalib::SetThresholdPhgrAccInv(double aThr)
{
    mThresholdPhgrAccInv = aThr;
    mThresholdPixAccInv = aThr * F();
}

void cPerspCamIntrCalib::SetThresholdPixAccInv(double aThr)
{
     SetThresholdPhgrAccInv(aThr/F());
}

const double & cPerspCamIntrCalib::F()  const {return mCSPerfect.F() ;}
const cPt2dr & cPerspCamIntrCalib::PP() const {return mCSPerfect.PP();}

       /* ================================================================== */
       /*                 BENCH VERIF PART                                   */
       /* ================================================================== */

void cPerspCamIntrCalib::TestInvInit(double aTolApprox,double aTolAccurate)
{
     {
         // generate 2d-point ine photogram coordinate , after distorsion
         double aRhoMax =  mPhgrDomain->Box().DistMax2Corners(cPt2dr(0,0));
         std::vector<cPt2dr>  aVPt1;
         mPhgrDomain->GridPointInsideAtStep(aVPt1,aRhoMax/10.0);

	 //  undist them by approx-lsq invers
         std::vector<cPt2dr>  aVPt2; // undist
         mInvApproxLSQ_Dist->Values(aVPt2,aVPt1);

	 // distord them back, should have  aVPt3 ~ aVPt1
         std::vector<cPt2dr>  aVPt3;
         mDir_Dist->Values(aVPt3,aVPt2);

	 //  undist them more accurately with predictive + iterative
         std::vector<cPt2dr>  aVPt4; // undist
         mDist_DirInvertible->Inverses(aVPt4,aVPt1);

	 // distord them back, should have  aVPt5 ~ aVPt1 (more accurateky)
         std::vector<cPt2dr>  aVPt5;
         mDir_Dist->Values(aVPt5,aVPt4);

         double aSD12=0;  // som dist  V1/V2 to have a referebce
         double aSD23=0;  // som dist  V1/V2 to have another referebce
         double aSD13=0;  //  V1/V3 should be low
         double aSD15=0;  //  V1/V5 should be very low
         for (size_t aKPt=0 ; aKPt<aVPt1.size() ; aKPt++)
         {
		 //  add all that, use square dist for efficiency
              aSD12 +=  SqN2(aVPt1.at(aKPt)-aVPt2.at(aKPt));
              aSD23 +=  SqN2(aVPt2.at(aKPt)-aVPt3.at(aKPt));
              aSD13 +=  SqN2(aVPt1.at(aKPt)-aVPt3.at(aKPt));
              aSD15 +=  SqN2(aVPt1.at(aKPt)-aVPt5.at(aKPt));

         }
	     // transform sum of square dist  an averager of distance
         aSD12 = std::sqrt(aSD12/aVPt1.size());
         aSD23 = std::sqrt(aSD23/aVPt1.size());
         aSD13 = std::sqrt(aSD13/aVPt1.size());
         aSD15 = std::sqrt(aSD15/aVPt1.size());

         MMVII_INTERNAL_ASSERT_bench((aSD13/aSD12<aTolApprox),"Test approx inv");
         MMVII_INTERNAL_ASSERT_bench((aSD15/aSD12<aTolAccurate),"Test approx inv");
     }

     {
         // generate 2D point on grid
         std::vector<cPt2dr>  aVPt0;
         mPixDomain->GridPointInsideAtStep(aVPt0,Norm2(mPixDomain->Sz())/20.0);

	 // filter them because witj some projection point to far are not invetrible
         std::vector<cPt2dr>  aVPt1;
	 for (const auto & aPt0 : aVPt0)
             if (InvProjIsDef(aPt0) >= 1e-2)
                aVPt1.push_back(aPt0);

	 // compute direction of bundles
         std::vector<cPt3dr>  aVPt2;
	 Inverses(aVPt2,aVPt1);

	 // back project on images
         std::vector<cPt2dr>  aVPt3;
	 Values(aVPt3,aVPt2);

         // aSD13 -> som dist between initial points and their back proj
         double aSD13=0;  
         for (size_t aKPt=0 ; aKPt<aVPt1.size() ; aKPt++)
         {
              double aD =  SqN2(aVPt1.at(aKPt)-aVPt3.at(aKPt));
	      MMVII_INTERNAL_ASSERT_tiny(ValidFloatValue(aD),"Bad value in TestInvInit");
              aSD13 += aD;
	 }

         aSD13 = std::sqrt(aSD13/aVPt1.size())  / mCSPerfect.F()  ;
         MMVII_INTERNAL_ASSERT_bench(aSD13<1e-8,"Test approx inv");
     }
}

void cPerspCamIntrCalib::InitRandom(double aAmpl)
{
     double aRhoMax =  mPhgrDomain->Box().DistMax2Corners(cPt2dr(0,0));

     cRandInvertibleDist  aParamRID ( mDir_Degr, aRhoMax, RandUnif_0_1(), aAmpl);

     mDir_Dist->SetObs(aParamRID.VParam());
}
 

void BenchCentralePerspective(cParamExeBench & aParam,eProjPC aTypeProj)
{
    // StdOut() <<  "Pppp " << E2Str(aTypeProj) << "\n";
    tREAL8 aDiag = 1000 * (1+10*RandUnif_0_1());
    cPt2di aSz (aDiag*(1+RandUnif_0_1()),aDiag*(1+RandUnif_0_1()));
    cPt2dr aPP(   aSz.x()*(0.5+0.1*RandUnif_C())  , aSz.y()*(0.5+0.1*RandUnif_C())  );
    tREAL8  aFoc =  aDiag * (0.2 + 3.0*RandUnif_0_1());

    for (int aK=0 ; aK<2 ; aK++)
    {
       cPerspCamIntrCalib aCam
       (
          aTypeProj,
	  (aK==0) ? cPt3di(2,0,0) : cPt3di(3,1,1),
	  std::vector<double>(),
	  cCalibStenPerfect(aFoc,aPP),
	  cPixelDomain(aSz),
	  (aK==0) ? cPt3di(5,1,1) :cPt3di(7,2,5),
	  100
       );
       aCam.SetThresholdPhgrAccInv(1e-9);

       aCam.InitRandom(0.1);
       aCam.UpdateLSQDistInv();
       aCam.TestInvInit((aK==0) ? 1e-3 : 1e-2, 1e-4);
    }
}

void BenchCentralePerspective_ImportV1(cParamExeBench & aParam,const std::string & aName)
{
     std::string aFullName = cMMVII_Appli::CurrentAppli().InputDirTestMMVII() + "Ori-MMV1" +  StringDirSeparator() + aName;

     cExportV1StenopeCalInterne  aExp(aFullName);


     /*
     cCalculator<double> * anEqL =EqColinearityCamPPC(aExp.eProj,cPt3di(3,1,1),true,10);
     StdOut() << "aFullNameaFullName " << aFullName  << " " << aExp.mCorresp.Pairs().size()  << " " << anEqL << "\n";

     delete anEqL;
     */
}

void BenchCentralePerspective(cParamExeBench & aParam)
{
    if (! aParam.NewBench("CentralPersp")) return;

    BenchCentralePerspective_ImportV1(aParam,"AutoCal_Foc-11500_Cam-imx477imx477-1.xml");
    BenchCentralePerspective_ImportV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml");

    StdOut() << "BenchCentralePerspectiveBenchCentralePerspective\n";  getchar();



    int aNbTime = std::min(20,3+aParam.Level());
    for (int aTime=0 ; aTime<aNbTime ; aTime++)
    {
        for (int aKEnum=0 ; aKEnum<int(eProjPC::eNbVals) ; aKEnum++)
        {
            BenchCentralePerspective(aParam,eProjPC(aKEnum));
        }
        // StdOut()  << "TTTtt " << aTime<< "\n"; //getchar();
    }


    cCalibStenPerfect aCS(1,cPt2dr(0,0));
    MMVII_INTERNAL_ASSERT_bench(&(aCS.F())+1 == &(aCS.PP().x()) ,"Assertion cCalibStenPerfect memory model");
    MMVII_INTERNAL_ASSERT_bench(&(aCS.F())+2 == &(aCS.PP().y()) ,"Assertion cCalibStenPerfect memory model");

    aParam.EndBench();
}

/* ******************************************************* */
/*                                                         */
/*                    cPixelDomain                         */
/*                                                         */
/* ******************************************************* */

cPixelDomain::~cPixelDomain()
{
}

cPixelDomain::cPixelDomain(const cPt2di &aSz) :
     cDataBoundedSet<tREAL8,2>(cBox2dr(cPt2dr(0,0),ToR(aSz))),
     mSz  (aSz)
{
}

cPixelDomain *  cPixelDomain::Dup_PS () const
{
    return new cPixelDomain(mSz);
}

const cPt2di & cPixelDomain::Sz() const {return mSz;}

/* ******************************************************* */
/*                                                         */
/*                 cCalibStenPerfect                       */
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

const double & cCalibStenPerfect::F()  const {return mF ;}
const cPt2dr & cCalibStenPerfect::PP() const {return mPP;}
double & cCalibStenPerfect::F()  {return mF ;}
cPt2dr & cCalibStenPerfect::PP() {return mPP;}

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

