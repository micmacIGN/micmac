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
struct cSet2D3D
{
     public :
         typedef cWeightedPair2D3D                tPair;
         typedef std::vector<tPair>   tCont2D3D;

         void AddPair(const tPair &);
         void AddPair(const cPt2dr&,const cPt3dr&,double aWeight=1.0);

         const tCont2D3D &  Pairs() const;
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
class cSetMesPtOf1Im
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
	  static  const std::string ThePrefixFiles;

     private :
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
class cSetMesGCP
{
    public :
          cSetMesGCP();
          cSetMesGCP(const std::string &aNameSet);
	  static cSetMesGCP  FromFile(const std::string & aNameFile);

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
      private :
              int                             mNumPt;
              std::vector<cPt2dr>             mVMeasures;
              std::vector<tREAL4>             mVSigma;  // optionnal
              std::vector<int>                mVImages;
};

/**  Class for storing a data base of GCP :  3D measures + 2D image measure
 *   The link between different measures is done using name of points.
 */
class cSetMesImGCP
{
    public :
            cSetMesImGCP();

            ///  Add one set of 3D measures (often called only once), all calls must occur before AddMes2D
            void AddMes3D(const cSetMesGCP &);
            void AddMes2D(const cSetMesPtOf1Im &);

            void ExtractMes1Im(cSet2D3D&,const std::string &aNameIm);
    private :

            cSetMesImGCP(const  cSetMesImGCP & ) = delete;

            bool                                 mPhaseGCPFinished;
            std::vector<cMes1GCP>                mMesGCP;
            std::vector<cMultipleImPt>  mMesIm;

            t2MapStrInt  m2MapPtInt; //
            t2MapStrInt  m2MapImInt; //
};




};

#endif  //  _MMVII_MEASURES_IM_H_
