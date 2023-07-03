#ifndef  _MMVII_MEASURES_IM_H_
#define  _MMVII_MEASURES_IM_H_

namespace MMVII
{

/** \file MMVII_MeasuresIm.h
    \brief contains classes for storing images measurement

*/

struct cPair2D3D;
struct cSet2D3D;
class cMesIm1Pt;
class cSetMesPtOf1Im;

class cMesIm1Pt;
class cSetMesPtOf1Im;
class cMes1GCP;
class cSetMesGCP;

class cMultipleImPt;
class cSetMesImGCP;



/** class for representing  a 3D point paired with it 2d image projection */
 
struct  cPair2D3D
{
     public :
          cPair2D3D(const cPt2dr &,const cPt3dr &);
          cPt2dr mP2;
          cPt3dr mP3;
};
 
/** A cPair2D3D + a Weight */
struct  cWeightedPair2D3D : public cPair2D3D
{
     public :
          cWeightedPair2D3D(const cPair2D3D&,double aWeight=1.0);
          cWeightedPair2D3D(const cPt2dr&,const cPt3dr&,double aWeight=1.0);

	  double mWeight;
};



/**  class for representing  set of pairs 2-3  */
struct cSet2D3D : public cMemCheck
{
     public :
         typedef cWeightedPair2D3D                tPair;
         typedef std::vector<tPair>   tCont2D3D;

         void AddPair(const tPair &);
         void AddPair(const cPt2dr&,const cPt3dr&,double aWeight=1.0);

         const tCont2D3D &  Pairs() const;
	 const tPair &  KthPair(int aK) const;
	 size_t NbPair() const;
         void  Clear() ;

	 /// compute  weighted centroid
	 cWeightedPair2D3D  Centroid() const;

	 /// subsract a pair to all
	 void Substract(const cPair2D3D&);
     private :
        tCont2D3D  mPairs;
};

/**  class for representing  the measure of a point in an image */
class cMesIm1Pt
{
     public :
        cMesIm1Pt(const cPt2dr & aPt,const std::string & aNameIm,tREAL4 aSigma);
        cMesIm1Pt();

        cPt2dr         mPt;
        std::string    mNamePt;
        tREAL4         mSigma2[3];  // xx xy yy
};

/** class for representing a set of measure in an image*/
class cSetMesPtOf1Im : public cMemCheck
{
     public :
          cSetMesPtOf1Im(const std::string & aNameIm);
          cSetMesPtOf1Im();
	  static cSetMesPtOf1Im  FromFile(const std::string & aNameFile);
          void AddMeasure(const cMesIm1Pt &);
          void AddData(const  cAuxAr2007 & anAux);

	  void ToFile(const std::string & aNameFile) const;
	  static std::string StdNameFileOfIm(const std::string &);
	  std::string StdNameFile() const;

	  const std::string & NameIm() const;  ///<  Accessor
          const std::vector<cMesIm1Pt> &  Measures() const;
          std::vector<cMesIm1Pt> &  Measures() ;
	  static  const std::string ThePrefixFiles;

          const cMesIm1Pt &  MeasuresOfName(const std::string & aNamePt) const;
          cMesIm1Pt &  MeasuresOfName(const std::string & aNamePt) ;
          bool  NameHasMeasure(const std::string & aNamePt) const;
          cMesIm1Pt *  NearestMeasure(const cPt2dr &) ;
     private :
          cMesIm1Pt *  PrivateMeasuresOfName(const std::string & aNamePt,bool SVP) const;

          std::string             mNameIm;
          std::vector<cMesIm1Pt>  mMeasures;
};

/**  class for representing  the measure of a 3D point (Ground Control Point) */
class cMes1GCP
{
     public :
        cMes1GCP(const cPt3dr & aPt,const std::string & aNamePt,tREAL4 aSigma);
        cMes1GCP();

        cPt3dr         mPt;
        std::string    mNamePt;
        tREAL4         mSigma2[6];  //  xx xy xz yy yz zz
};

/**  A set of cMes1GCP */
class cSetMesGCP : public cMemCheck
{
    public :
          cSetMesGCP();
          cSetMesGCP(const std::string &aNameSet);
	  static cSetMesGCP  FromFile(const std::string & aNameFile);
	  void    ToFile(const std::string & aNameFile);

          void AddMeasure(const cMes1GCP &);
          void AddData(const  cAuxAr2007 & anAux);

	  void ToFile(const std::string & aNameFile) const;
	  static std::string StdNameFileOfSet(const std::string &);
	  std::string StdNameFile() const;

          const std::vector<cMes1GCP> &   Measures() const;  ///< Accessor
	  static  const std::string ThePrefixFiles;

    private :
	  std::string              mNameSet;
          std::vector<cMes1GCP>    mMeasures;
};

/**  Class for reprenting the same point in different image, maybe same class
 * used for GCP and tie points */

class cMultipleImPt 
{
      public :
              cMultipleImPt(int aNum3DP);   ///< Cstr, num of GCP of -1 for tie point

              /// Add One image measurement, 4 now WithSigma must be false
              void Add(const cMesIm1Pt & ,int aNumIm,bool WithSigma);

              ///  Return if any the point of one image
              const cPt2dr * PtOfIm(int) const;

	      const std::vector<cPt2dr> & VMeasures() const;  ///< Accessor
              const std::vector<int>    & VImages()   const;  ///< Accessor
      private :
              int                             mNumPt;
              std::vector<cPt2dr>             mVMeasures;
              std::vector<tREAL4>             mVSigma;  // optionnal
              std::vector<int>                mVImages;
};

/**  Class for storing a data base of GCP :  3D measures + 2D image measure
 *   The link between different measures is done using name of points.
 *
 *   The mMesGCP  and mMesIm are corresponinf i.e  mMesGCP[k] <-> mMesIm[k] 
 */
class cSetMesImGCP : public cMemCheck
{
    public :
            cSetMesImGCP();

            ///  Add one set of 3D measures (often called only once), all calls must occur before AddMes2D
            void AddMes3D(const cSetMesGCP &);
	    /// For a single GCP (called by AddMes3D)
	    void Add1GCP(const cMes1GCP &);
	    ///  Add mesure on 1 images, close the possibility for further call to AddMes3D
            void AddMes2D(const cSetMesPtOf1Im &,cSensorImage* =nullptr,eLevelCheck OnNonExistP=eLevelCheck::Warning);

	    /// return a set of mesure as 2d/3d corresp
            void ExtractMes1Im(cSet2D3D&,const std::string &aNameIm);

            const std::vector<cMes1GCP> &        MesGCP() const ; ///< Accessor
            std::vector<cMes1GCP> &        MesGCP() ; ///< Accessor
            const std::vector<cMultipleImPt> &   MesImOfPt() const ;  ///< Accessor
	    const std::vector<cSensorImage*> &   VSens() const ;  ///< Accessor
								
	    tREAL8 AvgSqResidual() const;
								  
	    /// suppress mMesGCP & mMesIm with no images measure (eventually can give higher threshold) 
	    cSetMesImGCP * FilterNonEmptyMeasure(int NbMeasureMin=1) const;

            const cSetMesPtOf1Im  & MesImInitOfName(const std::string &) const;
	    const cMes1GCP &        MesGCPOfName(const std::string &) const;
	    bool  NameIsGCP(const std::string &) const;
	    cSetMesGCP  ExtractSetGCP(const std::string & aName) const;
    private :


            // cSetMesImGCP(const  cSetMesImGCP & ) = delete;

            bool                         mPhaseGCPFinished;
            std::vector<cMes1GCP>        mMesGCP;      
            std::vector<cMultipleImPt>   mMesImOfPt;  ///< after compilation, sorted by point
            std::vector<cSetMesPtOf1Im>  mMesImInit;  ///< initial structuration, sorted by image

            t2MapStrInt  m2MapPtInt; //
            t2MapStrInt  m2MapImInt; //

	    std::vector<cSensorImage*> mVSens;
};


class cHomogCpleIm
{
      public :
           cHomogCpleIm(const cPt2dr &,const cPt2dr &);
           cHomogCpleIm();  ///< required 4 serialization of vect of Cple

           cPt2dr  mP1;
           cPt2dr  mP2;
	   const cPt2dr & Pt(bool First) const {return First ? mP1 : mP2;}

	   void AddData(const  cAuxAr2007 & anAux);
};
void AddData(const  cAuxAr2007 & anAux,cHomogCpleIm &);

class cSetHomogCpleIm
{
      public :
	cSetHomogCpleIm(size_t aReserve=10);
	void AddData(const  cAuxAr2007 & anAux);
	void ToFile(const std::string &) const;
	static cSetHomogCpleIm FromFile(const std::string &) ;
	void   InitFromFile(const std::string &) ;

        const std::vector<cHomogCpleIm> & SetH() const;

	void Add(const cHomogCpleIm &);
	void Clear();


      private :
        std::vector<cHomogCpleIm>  mSetH;
};

void AddData(const  cAuxAr2007 & anAux,cSetHomogCpleIm &);

/**  Interface class  for importing homolog point in  MMVII from : MMV1 , simulation, MMVII (to come ...)
*/

class cInterfImportHom : public cMemCheck
{
      public :
          virtual void GetHom(cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2) const = 0;
          virtual bool HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const = 0;

          static cInterfImportHom *  CreateImportV1(const std::string & aDir,const std::string & aSubDir,const std::string & aExt="dat") ;

	  virtual ~cInterfImportHom();
      private :

          // std::string  mDirGlob;
          // std::string  mExtIn;
          // std::string  mExt;
          // std::string  mKHIn;
          // cElemAppliSetFile                 mEASF;
          // cInterfChantierNameManipulateur * mICNM ;
};


/**   This class store multiple homologous point, ie after fusion of  
 *    points computed by pair of images
 */

class cSetMultipleTiePoints : public cMemCheck
{
     public :
        typedef std::vector<int>     tConfigIm;  // A config is a set of num of images
        typedef std::vector<cPt2dr>  tPtsOfConfig;

        cSetMultipleTiePoints(const  std::vector<std::string> & aVNames,cInterfImportHom * =nullptr);

	/// Data allow to iterate on multiple points
        const std::map<tConfigIm,tPtsOfConfig> &  Pts() const;

	/// Method use in construction
        void AddPMul(const tConfigIm&,const std::vector<cPt2dr> &);

	/// Serialization, to see later
        void  AddData(const cAuxAr2007 & anAux);

	/// Used in bench to compare fusion with simulation
        void TestEq(cSetMultipleTiePoints &) const;

	/// From a linear vector to set of vector, for easiness of manip, but to avoid in efficient use
        std::vector<tPtsOfConfig > PUnMixed(const tConfigIm &,bool Sorted) const;

        const std::vector<std::string> & VNames() const; ///< Accessor
     private  :
        std::vector<std::string>          mVNames;
        std::map<tConfigIm,tPtsOfConfig>  mPts;
};



};

#endif  //  _MMVII_MEASURES_IM_H_
