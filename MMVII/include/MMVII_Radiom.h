#ifndef  _MMVII_RADIOM_H_
#define  _MMVII_RADIOM_H_

#include "MMVII_AllClassDeclare.h"
#include "MMVII_memory.h"
#include "MMVII_Ptxd.h"
#include "MMVII_SysSurR.h"
#include "MMVII_PhgrDist.h"


namespace MMVII
{
/* ======================================================*/
/* =======   Classes for storing radiometry  ============*/
/* ======================================================*/

class cImageRadiomData;   // Radiometry for one image
class cFusionIRDSEt;      // Fusion index of radiometry

/*   Basically, for each image in a cImageRadiomData, we will have for each point : 
 *       *  anIndex, apoint, one a several radiometry
 *   In cFusionIRDSEt, we will make the "fusion" of index allowing to retrieve the information belongin to the same point.
 *       *  tImInd : is pair num of image + num of point in this image
 *       *  tV1Index : is a set of tImInd, it contains all the merged point
 *       *  cFusionIRDSEt is essentially a set of tV1Index
 */

/* ======================================================*/
/* =======   Classes for radiometric modelization  ======*/
/* =======   of sensor/images                      ======*/
/* ======================================================*/

class cCalibRadiomSensor ;  // base class for radiometric modelizaion of the sensor
class cDataRadialCRS;       // data-part of cRadialCRS (separation for serialization)
class cRadialCRS ;          // class for radial sensor vignettage
class cCalibRadiomIma ;     // base class for per/image  correction
class cCalRadIm_Pol ;       // class for polynomial per image correction

/**  Store the radiometric data for one image. For one image we store for each point :

     - the radiometry
     - the coordinate of this points in image (optional)
     - an index (long int) that allow to recover its homologous in other image
*/


class cImageRadiomData : public cMemCheck
{
   public :
        typedef size_t               tIndex;
        typedef cPt2df               tPtMem;     ///< Point are store on float (4 byte) 
        typedef std::vector<tPtMem>  tVPt;       ///< vector of  pts 
        typedef tU_INT2              tRadiom;    ///< Type of integer on which we store each radiometry
	typedef std::vector<tRadiom> tVRadiom;   ///< Radiometry 

	/// Constructor allocate structure for future storing
        cImageRadiomData(const std::string & aNameIm,int aNbChanel,bool withPoint);
	void AddData(const  cAuxAr2007 & anAux); ///< Serialization
	/// Create an object from file
	static cImageRadiomData * FromFile(const std::string & aNameFile);
	/// Write an object on a file
	void    ToFile(const std::string & aNameFile) const;

	/// Generate standard namefile from the name of image, must be callable w/o object
	static std::string NameFileOfImage(const std::string&);
	/// Name file from an object 
	std::string NameFile() const;

	///  Put in "aRes" the num of common index with "aIRD2",  method is non const as reorder may be necessary
	void GetIndexCommon(std::vector<tPt2di> & aRes,cImageRadiomData & aIRD2);

	// only one of this 4 method is accessible, depending how was created (point or not, nb channel)
	void AddObsGray(tIndex,tRadiom);  ///< add for 1 chanel w/o point
	void AddObsGray(tIndex,tRadiom,const tPtMem &);  ///< add for 1 chanel with point
	void AddObsRGB(tIndex,tRadiom,tRadiom,tRadiom);  ///< add for 3 chanels w/o point
	void AddObsRGB(tIndex,tRadiom,tRadiom,tRadiom,const tPtMem &); ///< add for 3 chanels wth point

	///  switch on one other AddObs depending on internal variable
	void AddObs_Adapt(tIndex,tRadiom,tRadiom,tRadiom,const tPtMem &);

						
        void MakeOrdered();  /// order if necessary , "mIndexWellOrdered" allow to do it only when required
	static void Bench(cParamExeBench & aParam);

	const std::vector<tIndex> & VIndex()             const;  ///< Accessor
	const tVPt  &               VPts()               const;  ///< Accessor
	const tVRadiom            & VRadiom(size_t aKCh) const;  ///< Accessor
	size_t  LimitIndex() const ;  

	const tPtMem  & Pt  (size_t) const;
	tRadiom         Gray(size_t) const;  

   private :
	void AddIndex(tIndex);
	void CheckAndAdd(tIndex ,tRadiom ,int aNbCh,bool WithPoint);

	bool                   mIndexWellOrdered; ///< Flag used to avoid useless reordering
	std::string            mNameIm;     ///<  Name of image
	int                    mNbChanel;   ///<  Number of channel
	bool                   mWithPoints; ///< do we store also the points

	std::vector<tIndex>    mVIndex;      ///< Indexe that allow to recover homologous point
	size_t                 mLimitIndex;  ///< Indexes in [0 mLimitIndex[   in fact the MaxIndex+1
	tVPt                   mVPts;        ///< Optional Pts
	std::vector<tVRadiom>  mVVRadiom;    ///< Vector of radiometry [Chanel][Index]

};

/*  Class for doing the fusion of index. For a given index, we will have a direct access to:
 *
 *     - the image containing this index
 *     - the position of this index in the IRD data
 */

class cFusionIRDSEt
{
     public :
          typedef size_t                  tIndex;
          typedef cPt2di                  tImInd;  ///<  P.x() => num of image; P.y() => position of the index
          typedef std::vector<tImInd>     tV1Index;


          cFusionIRDSEt(size_t aMax);
          void Resize(size_t aMax);  ///< reset capacity, fix it befor use (capacity not adapted)
          /// Add in the data all the index for corresponding image
          void AddIndex(int aNumIm, const std::vector<tIndex> &);

          const std::vector<tV1Index > & VVIndexes() const;  ///<  Accessor
          void FilterSzMin(size_t aSzMin); ///<  Supress tV1Index of size  < aSzMin
     private :
           /// For index I mVVIndexes[I] : all images+ position in data containing this index
           std::vector<tV1Index > mVVIndexes;
};

/* ******************************************************** */
/*                                                          */
/*        Classes 4 "sensor" radiometric calibration        */
/*                                                          */
/* ******************************************************** */

/**  Base class for radiometric calibration of a sensor */
class cCalibRadiomSensor : 
                             public cMemCheck,
			     public cObjWithUnkowns<tREAL8>
{
       public :
           /// constructor, just share the name / identifier
           cCalibRadiomSensor();
           virtual ~cCalibRadiomSensor();
	   /// Allocator : switch on derived class according to name prefix
           static cCalibRadiomSensor * FromFile(const std::string & aNameFile);

	   /// Save in file,  defined in each derived class
           virtual void ToFile(const std::string & aNameFile) const = 0;

	   /// Save only is first time (in the same process) for this name 
           void ToFileIfFirstime(const std::string & aNameFile) const;

	   /// Function to divide radiometry for normalisation
           // virtual tREAL8  CorrectRadiom(const tREAL8&,const cPt2dr &) const =  0;
	   virtual cPt2dr  AddMul_CC(const cPt2dr & aPt) const = 0;


           /// Accessor
           virtual const std::string & NameCal() const =0;

	    //  Vector of observation, used in equation to compute normalized coordinate
	    virtual const std::vector<tREAL8> & VObs(const cPt2dr & ) const =0;

	    /// Not good design because it suppose model is radial,  but do it for now
	    virtual int NbParamRad() const =0 ;
	    virtual bool WithCste() const  =0;
	    virtual int  DegPol() const  =0;
	    virtual const std::vector<tREAL8> & CoeffRad() const =0;
	    virtual tREAL8 & Cste2Add() =0;

       protected :
           // std::string            mNameCal;   ///< Name of file
};

/** As ususal with complexe object serialisable, separate fundamental data 
 * from "living" object */

class cDataRadialCRS
{
      public :
           cDataRadialCRS
           (
               const cPt2dr & aCenter,size_t aDegRad,const cPt2di & aSzPix,const std::string & aNameCal,
	       bool WithCste=false,int aDegPol=-1
           );
           cDataRadialCRS();  ///< usefull for AddData .
	   void AddData(const cAuxAr2007 & anAux);

           std::string            mNameCal;   ///< Name of file
           cPt2dr                 mCenter;    ///< Center of symetry
           std::vector<double>    mCoeffRad;  ///< Coeff of radial pol R2 R4 ...
	   bool                   mWithAddCste;
	   tREAL8                 mCste2Add;
	   int                    mDegPol;
           std::vector<double>    mCoeffPol;  ///< Coeff of Polynom if exist
           cPt2di                 mSzPix;     ///< Size in pixel, for info
           tREAL8                 mScaleNor;
};

/**  class for radial calibration radiometric of sensor , 
     caracterized by a symetry center and a even polynomial 
*/
class cRadialCRS : public cCalibRadiomSensor,
      	           public cObj2DelAtEnd,
	           public cDataRadialCRS
{
    public :
        cRadialCRS(const cPt2dr & aCenter,size_t aDegRad,const cPt2di & aSzPix,const std::string &,bool WithCste=false,int aDegPol=-1);
        cRadialCRS(const cDataRadialCRS&);
        virtual ~cRadialCRS();

        void  AddData(const cAuxAr2007 & anAux);
        static cRadialCRS * FromFile(const std::string & aNameFile);
        void ToFile(const std::string & aNameFile) const override;

        // normalize coordinate to have non dimensional equarion
        // tREAL8  NormalizedRho2(const cPt2dr & aPt) const;
        // tREAL8  CorrectRadiom(const tREAL8&,const cPt2dr &) const override;
	cPt2dr  AddMul_CC(const cPt2dr & aPt) const override;
        std::vector<double> &  CoeffRad();

         /// Accessor
        const std::string & NameCal() const override;

	const std::vector<tREAL8> & VObs(const cPt2dr & ) const override;
        int NbParamRad() const override;
        bool WithCste() const override ;
        int  DegPol() const  override;
        const std::vector<tREAL8> & CoeffRad() const override;
	tREAL8 & Cste2Add() override;
    private :
	cRadialCRS (const cRadialCRS&) = delete;
	void PutUknowsInSetInterval() override ;

	mutable std::vector<tREAL8>                  mVObs; ///< Vector of obs use in normalization 4 computation
	NS_SymbolicDerivative::cCalculator<double> * mCalcFF;
};

/* ******************************************************** */
/*                                                          */
/*    Classes 4 "per image" radiometric calibration         */
/*                                                          */
/* ******************************************************** */

/**  Base-class for calibration of radiometry of each individual image */
class cCalibRadiomIma : public cMemCheck,
                        public cObjWithUnkowns<tREAL8>
{
        public :
            virtual tREAL8  ImageCorrec(tREAL8 aGray,const cPt2dr &) const  =0;
            virtual cPt3dr  ImageCorrec(const cPt3dr & aCoul,const cPt2dr &) const  =0;
            virtual void  ToFile(const std::string &) const =0; ///< export in xml/dmp ...  
	    virtual const std::string & NameIm() const = 0;
	    virtual ~cCalibRadiomIma() ;  ///< nothing to do, but maybe in derived classes
            virtual cCalibRadiomSensor &  CalibSens() =0;

	    virtual std::vector<double> &  Params() = 0;
	    virtual const std::vector<cDescOneFuncDist> & VDesc()  const =0 ;
	    
	    /// Return the index in global equation (begin to Uk0) corresponding to degree, -1 if SVP&NotFound
	    int  IndDegree(const cPt2di & aDegree,bool SVP=false) const;
	    ///  IndDegre of 0,0 the Cst-polynom
	    int  IndCste() const;

	    virtual NS_SymbolicDerivative::cCalculator<double> * ImaEqual() = 0;
	    virtual NS_SymbolicDerivative::cCalculator<double> * ImaStab () = 0;
	    // Vector of observation , probably comme from sensor
	    // virtual const std::vector<tREAL8> & VObs(const cPt2dr & ) const = 0;
	    //  Vector of param containing Sensor + Owns
	    // std::vector<double>   ParamGlob() =0 ;
            virtual int MaxDegree() const =0;
        protected :
	    cCalibRadiomIma();

};


/**   calibration of radiometry with cte model/image */
class cCalRadIm_Pol : public  cCalibRadiomIma
{
        public :
            cCalRadIm_Pol(); ///< For AddData
            ~cCalRadIm_Pol(); ///< For AddData
            cCalRadIm_Pol(cCalibRadiomSensor *,int  aDegree,const std::string & aNameIm);
            void  AddData(const cAuxAr2007 & anAux);

            void  ToFile(const std::string &) const override ; ///< export in xml/dmp ...  
            static cCalRadIm_Pol * FromFile(const std::string &); ///< create form xml/dmp ...

            tREAL8  ImageCorrec(tREAL8 aGray,const cPt2dr &) const  override;
            cPt3dr  ImageCorrec(const cPt3dr & aCoul,const cPt2dr &) const  override;

            cCalibRadiomSensor &  CalibSens() override;
	    const std::string & NameIm() const override;

	    std::vector<double> &  Params() override ;
	    const std::vector<cDescOneFuncDist> & VDesc() const override ;
	    NS_SymbolicDerivative::cCalculator<double> * ImaEqual() override;
	    NS_SymbolicDerivative::cCalculator<double> * ImaStab() override;
            int MaxDegree() const override;

        public :
	    /// Correction w/o sensor
            tREAL8  ImageOwnDivisor(const cPt2dr &) const  ;

	    cCalRadIm_Pol (const cCalRadIm_Pol&) = delete;
	    void PutUknowsInSetInterval() override ;
	    void PostInit();

            cCalibRadiomSensor *                         mCalibSens;
	    std::string                                  mNameCalib;
	    int                                          mDegree;
	    std::string                                  mNameIm;
	    std::vector<tREAL8>                          mCoeffPol;
	    NS_SymbolicDerivative::cCalculator<double> * mImaOwnCorr;
	    NS_SymbolicDerivative::cCalculator<double> * mImaEqual;
	    NS_SymbolicDerivative::cCalculator<double> * mImaStab;
};




};

#endif  //   #define  _MMVII_RADIOM_H_
