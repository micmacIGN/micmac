#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_PhgrDist.h"
#include "../Serial/Serial.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "cExternalSensor.h"
#include "../SymbDerGen/Formulas_GenSensor.h"


// Pour avoir des exemple de functor d'équation de colinearité "simple",
// voir  "CodeGen_cEqColinearityCamPPC_Stenope_Dist_Rad0_Dec0_XY0Val.cpp"

using namespace NS_SymbolicDerivative;

namespace MMVII
{


/* *********************************************** */
/*                                                 */
/*             cExternalSensorModif2D              */
/*                                                 */
/* *********************************************** */

class cExternalSensorModif2D  : public cSensorImage
{
   public :

   // - - - - - - - Constructors and destructor - - - - - - - - 

         cExternalSensorModif2D();
         cExternalSensorModif2D
         (
                const std::string& aFolderOri ,
                const std::string& aNameImage ,
                cSensorImage * aSI ,
                int aDegree 
         ) ;
         ~cExternalSensorModif2D();


   // - - - - - - - Serialization (read/write)  - - - - - - - - 
              
         void InitSensor(const cPhotogrammetricProject &);  ///< once the data has been read, finish init by computing sensor
         void InitEquation() ; ///< compute the euqation of distortion
	 /// try to read a sensor of type "cExternalSensorModif2D" associated to an image						      
         static cExternalSensorModif2D * TryRead(const cPhotogrammetricProject &,const std::string&aNameImage);

	 ///  Method to describe the data
         void AddData(const  cAuxAr2007 & anAux);

         void ToFile(const std::string & aNameFile) const override;  ///< Save the object on ai file  "aNameFile"
         std::string  V_PrefixName() const  override; ///< virtual name of the class
         static std::string  StaticPrefixName();      ///< static name of the class

   // - - - - - - - Simulation  - - - - - - - - 
  
         /// Create perturbation, used to check correctness in simulation
         void PerturbateRandom(tREAL8 anAmpl,bool Show);

    private :
	
    //  - - - - - - -  Methods for being a geometric sensor (or just a sensor in fact)   -  - - - - - - 
         /// most fundamental method : 
         tSeg3dr  Image2Bundle(const cPt2dr &) const override;
         /// Basic method  GroundCoordinate ->  image coordinate of projection
         cPt2dr Ground2Image(const cPt3dr &) const override;
         ///    Method specialized, more efficent than using bundles
         cPt3dr ImageAndZ2Ground(const cPt3dr &) const override;

     //  - - - - - - -   Method for indicating the validity of the sensor -  - - - - - - 
  
	  /// What is the image domain
	  virtual const cPixelDomain & PixelDomain() const override;
	  ///  How much is a point inside the validity domain
          double DegreeVisibility(const cPt3dr & aPGround) const override;
	  /// Has the sensor a Z Interval of validity ?
	  bool  HasIntervalZ()  const override;
          /// return the Z validity interval (if exist , else error ...)
          cPt2dr GetIntervalZ() const override;

     //  - - - - - - -  Methods for being a Differentiable sensor -  - - - - - - 
    
         cPt3dr  EpsDiffGround2Im(const cPt3dr &) const override; ///< Default for use in finite difference
         // tProjImAndGrad  DiffGround2Im(const cPt3dr & aP) const override;    TODO later for efficiency & accuracy
         virtual cPt3dr  PseudoCenterOfProj() const  override;  /// required, not very usefull for now ...


     //  - - - - - - -  Methods for being an adjustable sensor  = communication with bundle adjustment kernel  -  - - - - - - 

	      /// return the calculator of residual in colinearity equation
         cCalculator<double> * CreateEqColinearity(bool WithDerives,int aSzBuf,bool ReUse) override;
	      /// add, when required, the unknowns in equation 
         void PutUknowsInSetInterval() override;
	      /// add, when required, the observation (rather context dependant constant) in colinearity equation 
         void PushOwnObsColinearity( std::vector<double> &,const cPt3dr &)  override;
          
         size_t  NbParam() const;  ///< Number of parameters, computed from degree
				    
                 // ====  Method to override in derived classes  =====

         /// For "initial" coordinat (pixel) to "final"
         cPt2dr  InitPix2Correc (const cPt2dr & aP0) const ;
         /// For "final" to "initial" coordinat (pixel)
         cPt2dr  Correc2InitPix (const cPt2dr & aP0) const ;

     // - - - - - - - - - - - - -    DATA PART - - - - - - - - - - - - - - - - - - - - -

         std::string             mDirSensInit; ///< Folder where is located the initial sensor
         std::string             mNameImage;   ///< Name of the attached image
	 int                     mDegree;      ///< Maxima total degree of monoms
         std::vector<tREAL8>     mVParams;     ///< Coefficient of monoms in correction
						       
	 cSensorImage *                 mSensorInit;   ///< The initial sensor, for which we compute a correctio,
         cCalculator<double> *          mEqCorrec2ImInit;    ///< Functor that compute the distorstion
};


/* =============================================== */
/*                                                 */
/*                 cSensorImage                    */
/*                                                 */
/* =============================================== */

    //-------------------------------------------------------------------------------
    //------------------------ Create/Read/Write   ----------------------------------
    //-------------------------------------------------------------------------------


               // -  -  -  -  -  - Creators and Destructor -  -  -  -  -  -

cExternalSensorModif2D::cExternalSensorModif2D() :
    cSensorImage (MMVII_NONE),
    mSensorInit  (nullptr),
    mEqCorrec2ImInit   (nullptr)
{
}

cExternalSensorModif2D::cExternalSensorModif2D
(
      const std::string& aFolderOri,
      const std::string& aNameImage,
      cSensorImage * aSI,
      int            aDegree
)  :
    cSensorImage (aNameImage),
    mDirSensInit (aFolderOri),
    mNameImage   (aNameImage),
    mDegree      (aDegree),
    mVParams     (NbParam(),0.0),
    mSensorInit  (aSI)
{
    //  read the 2-D polynomial equation
    InitEquation();

    if (mSensorInit)  // Put eventually the coordinate system in which the sensor works
       TransferateCoordSys(*mSensorInit);
}

size_t  cExternalSensorModif2D::NbParam() const 
{
   if (mDegree<0) return 0;
   return Polyn2DDescDist(mDegree).size();
}

void cExternalSensorModif2D::InitEquation() 
{
/*
    mEqCorrec2ImInit   =  (mDegree>=0)                                                       ?
		    EqDistPol2D(mDegree,WithDerivate::No,1,ReUseMode::Yes)  :
		    nullptr
		 ;    
*/
}

cExternalSensorModif2D::~cExternalSensorModif2D()
{
}

void cExternalSensorModif2D::InitSensor(const cPhotogrammetricProject & aPhP)
{
     // Read the sensor init : indicate the folder where it is located
     mSensorInit = aPhP.ReadSensorFromFolder(mDirSensInit,mNameImage,true,false);
     // Initialize  the coordinate system
     TransferateCoordSys(*mSensorInit);
}

               // -  -  -  -  -  - naming an object  -  -  -  -  -  -

std::string  cExternalSensorModif2D::StaticPrefixName()    { return  "SensModifPol2D";}
std::string  cExternalSensorModif2D::V_PrefixName() const  {return StaticPrefixName();}

               // -  -  -  -  -  - read an object  -  -  -  -  -  -

cExternalSensorModif2D * cExternalSensorModif2D::TryRead
                  (
                        const cPhotogrammetricProject & aPhProj,
                        const std::string & aNameImage
                  )
{
   return nullptr;
   // [1] ToDoBAPOL2 Try read w, use method SimpleTplTryRead

   // [2] ToDoBAPOL2 if fail (adequate name did not exist)  or already exist : stop here

   // [3] ToDoBAPOL2  else, we have the "data part", we can finish the intializatio, InitSensor/InitEquation
}

cSensorImage * SensorTryReadSensM2D (const cPhotogrammetricProject & aPhProj, const std::string & aNameImage)
{
    return cExternalSensorModif2D::TryRead(aPhProj,aNameImage);
}
               // -  -  -  -  -  - write an object  -  -  -  -  -  -

void cExternalSensorModif2D::AddData(const  cAuxAr2007 & anAux0)
{
     // ToDoBAPOL2 
     // We must "communicate" to Aux0 all the data of the sensor for write&read. Also
     // we want to embed everyting in a global Tag "SensorCorrecPol2D"
     // 
     //   The easy part is mDirSensInit, mNameImage, mDegree .
     //   For the coefficients mVParams its a bit more complicated because :
     //
     //       *  when reading, we must fix the size, that depend of the degree,
     //       * and for cosmetic issue, we want to give interpretable name to the tags,
     //         for this we use Polyn2DDescDist, that give high level description
     //

    // read/write the 3 "easy" fields

    // in read mode, we must fix the size before parsing all parameters

    // now read/write the coefficients
}

void AddData(const  cAuxAr2007 & anAux,cExternalSensorModif2D & aExtSM2d)
{
    aExtSM2d.AddData(anAux);
}


void cExternalSensorModif2D::ToFile(const std::string & aNameFile) const 
{
     SaveInFile(const_cast<cExternalSensorModif2D &>(*this),aNameFile);
}

cPt3dr  cExternalSensorModif2D::EpsDiffGround2Im(const cPt3dr & aP) const 
{
	return mSensorInit->EpsDiffGround2Im(aP);
}
cPt3dr  cExternalSensorModif2D::PseudoCenterOfProj() const  
{
      return mSensorInit->PseudoCenterOfProj();
}
					       
/*
tProjImAndGrad  cExternalSensorModif2D::DiffGround2Im(const cPt3dr & aP) const
{
     // TODO LATER : this is an approximatin as we neglect correction, later add the derivate of correction an give the exact formula
     return mSensorInit->DiffGround2Im (aP);
}

*/

    //-------------------------------------------------------------------------------
    //---------------------- FOR BUNDLE ADJUSTMENT  ---------------------------------
    //-------------------------------------------------------------------------------
    
cCalculator<double> * cExternalSensorModif2D::CreateEqColinearity(bool WithDerive,int aSzBuf,bool ReUse) 
{
    // the sensor dont store its colinearity equation, it furnish it to the bundle adjusment when it is
    // required
   
    return nullptr; // ToDoBAPOL2
}

void cExternalSensorModif2D::PutUknowsInSetInterval() 
{
     // the unknowns that the sensor want to estimate are the coefficient of the monom
     //    ToDoBAPOL2
}

void cExternalSensorModif2D::PushOwnObsColinearity
     (
         std::vector<double> & aVObs, //  vector where must write our observation/context
	 const cPt3dr & aPGround      //  3D ground point (current estimation of unknown point)
     )
{
   //  Copy the string extracted from file "SymbDerGen/Formulas_GenSensor.h"
   //
   // "IObs","JObs"  |,|      "P0x","P0y","P0z"   |,|     "IP0","JP0"  |,|      "dIdX","dIDY","dIdZ"  |,|  "dJdX","dJDY","dJdZ"
  

   // "IObs","JObs",  are the coordinate of the measured point, we dont add them, it is the job of BA-Kernel

   // ToDoBAPOL2  "P0x","P0y","P0z"  : estimation of 3D point

    // Now we compute the jacobian  ToDoBAPOL2
    

     //  ToDoBAPOL2   "IP0","JP0"  :  projection of estimated point
     //  ToDoBAPOL2   "dIdX","dIDY","dIdZ" :  gradient of I-coordinate of  projection, in estimated point
     //  ToDoBAPOL2   "dJdX","dJDY","dJdZ" :  gradient of J-coordinate of  projection, in estimated point
}

    //-------------------------------------------------------------------------------
    //---------------------- GEOMETRIC FUNCTIONS    ---------------------------------
    //-------------------------------------------------------------------------------

      //------------------------ 3D/2D correspondances ---------------------------
  
tSeg3dr  cExternalSensorModif2D::Image2Bundle(const cPt2dr & aPixIm) const 
{
	// it just the bundle of initial sensor, taking into account corrected point  (Init->Correc)
	tSeg3dr aSeg =   mSensorInit->Image2Bundle(aPixIm);  // ToDoBAPOL2
	return aSeg;
}

cPt2dr cExternalSensorModif2D::Ground2Image(const cPt3dr & aPGround) const
{
	// it is just the projection of initial sensor + correction (Correc ->Init)
	return  mSensorInit->Ground2Image(aPGround); // ToDoBAPOL2
}

cPt3dr cExternalSensorModif2D::ImageAndZ2Ground(const cPt3dr & aPxyz) const 
{
     // its just the inverse projection taking into account correction
     cPt2dr aPCor = InitPix2Correc(cPt2dr(aPxyz.x(),aPxyz.y()));
     return mSensorInit->ImageAndZ2Ground(cPt3dr(aPCor.x(),aPCor.y(),aPxyz.z()));
}

      //------------------------ VALIDITY DOMAINS -----------------------------------------

const cPixelDomain & cExternalSensorModif2D::PixelDomain() const
{
     //  an approximation, but correction are supposed to be small
      return mSensorInit->PixelDomain();
}

double cExternalSensorModif2D::DegreeVisibility(const cPt3dr & aPGround) const 
{
     // it has the same visibilit than the initial sensor
	return mSensorInit->DegreeVisibility(aPGround);
}
bool  cExternalSensorModif2D::HasIntervalZ()  const
{
	return mSensorInit->HasIntervalZ();
}
cPt2dr cExternalSensorModif2D::GetIntervalZ() const 
{
	return mSensorInit->GetIntervalZ();
}

      //------------------------ 2D deformation -----------------------------------------

cPt2dr  cExternalSensorModif2D::Correc2InitPix (const cPt2dr & aP0) const
{
     // We must do the correction using the funcot mEqCorrec2ImInit
     // As the functors only use  vector in/out we do the transo
     //
     //          ToStdVecto        mEqIma2End->DoOneEval(Unk,Obs)                    FromStdVector
     //    P0   -------> vector   ----------------------------------> corr vector ------------------> PCor
     //
     return cPt2dr(0,0);   // ToDoBAPOL2
}


cPt2dr  cExternalSensorModif2D::InitPix2Correc (const cPt2dr & aPInit) const
{
    // For now implement a very basic "fix point" method for inversion
     tREAL8 aThresh = 1e-3;
     int aNbIterMax = 10;

     cPt2dr aPEnd = aPInit;
     bool GoOn = true;
     int aNbIter = 0;
     while (GoOn)
     {
	  cPt2dr aNewPInit  = Correc2InitPix(aPEnd);
	  // We make the taylor expansion assume Correc2InitPix ~Identity
	  // Correc2InitPix (aPEnd + aPInit - aNewPInit) ~ Correc2InitPix (aPEnd) + aPInit - aNewPInit = aNewPInit +aPInit - aNewPInit
	  aPEnd += aPInit - aNewPInit;
          aNbIter++;
	  GoOn = (aNbIter<aNbIterMax) && (Norm2(aNewPInit-aPInit)>aThresh);
     }
     // StdOut() << "NBITER=" << aNbIter << "\n";
     return aPEnd;
}
    //-------------------------------------------------------------------------------
    //----------------------  FOR CHECK/SIMULATION  ---------------------------------
    //-------------------------------------------------------------------------------

void cExternalSensorModif2D::PerturbateRandom(tREAL8 anAmpl,bool Show)
{
    anAmpl /= mVParams.size();
    tREAL8 aNorm = Norm2(Sz());

    std::vector<cDescOneFuncDist>  aVDesc =  Polyn2DDescDist(mDegree);
    for (size_t aK=0 ; aK<mVParams.size() ; aK++)
    {
        mVParams[aK] = (RandUnif_C() * anAmpl ) / std::pow(aNorm,aVDesc[aK].mDegTot-1);
    }

    if (Show)
    {
       for (int aK=0 ; aK<20 ; aK++)
       {
          cPt2dr aP0 = MulCByC(ToR(Sz()) ,cPt2dr::PRand());

          cPt2dr aP1 = Correc2InitPix(aP0);
          cPt2dr aP2 = InitPix2Correc(aP1);

          StdOut() << " Pts=" << aP0  << " Dist=" << aP1-aP0  <<  " Dist/Inv" << (aP2 -aP0)*1e10 << "\n";
       }
    }
}


/* =============================================== */
/*                                                 */
/*                 cAppliImportExternSensor        */
/*                                                 */
/* =============================================== */

class cAppliParametrizeSensor : public cMMVII_Appli
{
     public :

        cAppliParametrizeSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	std::vector<std::string>  Samples() const override;

        void TestTuto() const;

	void ImportOneImage(const std::string &);

        cPhotogrammetricProject  mPhProj;

        // --- Mandatory ----
        std::string                 mNameImagesIn;
	std::vector<std::string>    mPatChgName;

        // --- Optionnal ----
	int          mDegreeCorr;
	tREAL8       mRanPert;
	bool         mShowPert;
	bool         mTutoTest;

     // --- Internal ----
};

cAppliParametrizeSensor::cAppliParametrizeSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mShowPert        (false),
   mTutoTest        (false)
{
}


cCollecSpecArg2007 & cAppliParametrizeSensor::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return anArgObl
      <<   Arg2007(mNameImagesIn,"Name of input sensor gile", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
      <<   mPhProj.DPOrient().ArgDirInMand()
      <<   mPhProj.DPOrient().ArgDirOutMand()
      <<   Arg2007(mDegreeCorr,"Degree of correction for sensors")
   ;
}

cCollecSpecArg2007 & cAppliParametrizeSensor::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mRanPert,"RandomPerturb","Random perturbation of initial 2D-Poly",{eTA2007::Tuning})
           << AOpt2007(mShowPert,"ShowPert","Do we show the result of pert",{eTA2007::HDV,eTA2007::Tuning})
           << AOpt2007(mTutoTest,"TutoTest","Do we show the result of pert",{eTA2007::HDV,eTA2007::Tuning})
   ;
}

std::vector<std::string>  cAppliParametrizeSensor::Samples() const
{
   return {
              "MMVII OriParametrizeSensor AllIm.xml ..."
	};
}


void  cAppliParametrizeSensor::ImportOneImage(const std::string & aNameIm)
{
    cSensorImage * aSensInit = mPhProj.ReadSensor(aNameIm,DelAuto::Yes,SVP::No);

    cExternalSensorModif2D aSensM2D(mPhProj.DPOrient().DirIn(),aNameIm,aSensInit,mDegreeCorr);

    if (IsInit(&mRanPert))
       aSensM2D.PerturbateRandom(mRanPert,mShowPert);
    
    mPhProj.SaveSensor(aSensM2D);
}

void cAppliParametrizeSensor::TestTuto() const
{
     std::vector<cDescOneFuncDist>  aVD = Polyn2DDescDist(mDegreeCorr);

     StdOut() <<  " =============  MONOMS DESCRIPTOR ================= \n";
     for (const auto & aDesc : aVD)
         StdOut() << " Deg=" << aDesc.mDegMon  << " Name=" << aDesc.mName << "\n";

     cDistPolyn2D aDist(mDegreeCorr,false);
     NS_SymbolicDerivative::cCoordinatorF<double>  aCFD("Test",1,{"x","y"},aDist.VNamesParam());

     auto aXY = aCFD.VUk();
     auto aParam = aCFD.VObs();

     auto aF = aDist.Funcs(aXY.at(0),aXY.at(1),aParam,0);
     StdShowTreeFormula(aF.at(0));


}


int cAppliParametrizeSensor::Exe()
{
    mPhProj.FinishInit();

    if (mTutoTest)
       TestTuto();

    for (const auto & aNameIm :  VectMainSet(0))
    {
         ImportOneImage(aNameIm);
    }

    // TestRPCProjections(mNameSensorIn);

    return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_ParametrizeSensor(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliParametrizeSensor(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecParametrizeSensor
(
     "OriParametrizeSensor",
      Alloc_ParametrizeSensor,
      "Import an External Sensor",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);

};
