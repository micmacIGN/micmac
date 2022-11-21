#ifndef  _MMVII_RADIOM_H_
#define  _MMVII_RADIOM_H_

#include "MMVII_AllClassDeclare.h"
#include "MMVII_memory.h"
#include "MMVII_Ptxd.h"


namespace MMVII
{
class cImageRadiomData;
class cFusionIRDSEt;
class cCalibRadiomSensor ;
class cRadialCRS ; 
class cCalRadIm_Cst ; 
class cComputeCalibRadIma ;

class cImageRadiomData : public cMemCheck
{
   public :
        typedef size_t               tIndex;
        typedef cPt2df               tPtMem;
        typedef std::vector<tPtMem>  tVPt;
        typedef tU_INT2              tRadiom;
	typedef std::vector<tRadiom> tVRadiom;

        cImageRadiomData(const std::string & aNameIm,int aNbChanel,bool withPoint);
	static cImageRadiomData * FromFile(const std::string & aNameFile);
	void    ToFile(const std::string & aNameFile) const;
	static std::string NameFileOfImage(const std::string&);
	std::string NameFile() const;

	// non const <= possible reordering
	void GetIndexCommon(std::vector<tPt2di> &,cImageRadiomData &);

	void AddObsGray(tIndex,tRadiom);
	void AddObsGray(tIndex,tRadiom,const tPtMem &);
	void AddObsRGB(tIndex,tRadiom,tRadiom,tRadiom);
	void AddObsRGB(tIndex,tRadiom,tRadiom,tRadiom,const tPtMem &);

	///  switch on one other AddObs depending on internal variable
	void AddObs_Adapt(tIndex,tRadiom,tRadiom,tRadiom,const tPtMem &);

	void AddData(const  cAuxAr2007 & anAux); ///< Serialization
						
        void MakeOrdered();
	static void Bench(cParamExeBench & aParam);

	const std::vector<tIndex> & VIndex()             const;
	const tVPt  &               VPts()               const;
	const tVRadiom            & VRadiom(size_t aKCh) const;
	size_t  LimitIndex() const ;  

	const tPtMem  & Pt  (size_t) const;
	tRadiom         Gray(size_t) const;  


   private :
	void AddIndex(tIndex);
	void CheckAndAdd(tIndex ,tRadiom ,int aNbCh,bool WithPoint);

	bool                   mIndexWellOrdered;
	std::string            mNameIm;
	int                    mNbChanel;
	bool                   mWithPoints;

	std::vector<tIndex>    mVIndex;
	size_t                 mLimitIndex;   // Indexes in [0 mLimitIndex[   in fact the MaxIndex+1
	tVPt                   mVPts;
	std::vector<tVRadiom>  mVVRadiom;

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

/**  Base class for radiometric calibration of a sensor */
class cCalibRadiomSensor :   public cObj2DelAtEnd,
                             public cMemCheck
{
       public :
           /// constructor, just share the name / identifier
           cCalibRadiomSensor(const std::string & aNameCal);
	   /// Allocator : switch on derived class according to name prefix
           static cCalibRadiomSensor * FromFile(const std::string & aNameFile);

	   /// Save in file,  defined in each derived class
           virtual void ToFile(const std::string & aNameFile) const = 0;

	   /// Save only is first time (in the same process) for this name 
           void ToFileIfFirstime(const std::string & aNameFile) const;

	   /// Function to divide radiometry for normalisation
           virtual tREAL8  FlatField(const cPt2dr &) const =  0;

           /// Accessor
           const std::string & NameCal() const;
       protected :
           std::string            mNameCal;   ///< Name of file
};


/**  class for radial calibration radiometric of sensor , 
     caracterized by a symetry center and a even polynomial 
*/
class cRadialCRS : public cCalibRadiomSensor
{
    public :
        cRadialCRS();  ///< usefull for addata .
        cRadialCRS(const cPt2dr & aCenter,size_t aDegRad,const cPt2di & aSzPix,const std::string &);

        void  AddData(const cAuxAr2007 & anAux);
        static cRadialCRS * FromFile(const std::string & aNameFile);
        void ToFile(const std::string & aNameFile) const override;

        tREAL8  NormalizedRho2(const cPt2dr & aPt) const;
        tREAL8  FlatField(const cPt2dr &) const override;
        std::vector<double> &  CoeffRad();


    private :

        cPt2dr                 mCenter;    ///< Center of symetry
        std::vector<double>    mCoeffRad;  ///< Coeff of radial pol R2 R4 ...
        cPt2di                 mSzPix;     ///< Size in pixel, for info
        tREAL8                 mScaleNor;  ///< Scale of normalization
};

/**  Base-class for calibration of radiometry */
class cCalibRadiomIma : public cMemCheck
{
        public :
            virtual tREAL8  ImageCorrec(const cPt2dr &) const   = 0;
            virtual void  ToFile(const std::string &) const =0; ///< export in xml/dmp ...  
	    const std::string & NameIm() const;
	    virtual ~cCalibRadiomIma() ;  ///< nothing to do, but maybe in derived classes
        protected :
	    cCalibRadiomIma(const std::string & aNameIm);

	    std::string mNameIm;
};

/**   calibration of radiometry with cte model/image */
class cCalRadIm_Cst : public  cCalibRadiomIma
{
        public :
            cCalRadIm_Cst(); ///< For AddData
            cCalRadIm_Cst(cCalibRadiomSensor *,const std::string & aNameIm);
            void  AddData(const cAuxAr2007 & anAux);

            void  ToFile(const std::string &) const override ; ///< export in xml/dmp ...  
            static cCalRadIm_Cst * FromFile(const std::string &); ///< create form xml/dmp ...


            tREAL8  ImageCorrec(const cPt2dr &) const  override;

            tREAL8 & DivIm();
            const tREAL8 & DivIm() const ;
            cCalibRadiomSensor &  CalibSens();

        public :
             cCalibRadiomSensor *  mCalibSens;
             tREAL8                mDivIm;
             mutable std::string   mTmpCalib;
};




};

#endif  //   #define  _MMVII_RADIOM_H_
