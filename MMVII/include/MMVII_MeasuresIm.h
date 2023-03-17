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


/** class for representing  a 3D point paired with it 2d image projection */
 
struct  cPair2D3D
{
     public :
          cPair2D3D(const cPt2dr &,const cPt3dr &);
          cPt2dr mP2;
          cPt3dr mP3;
};
 
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
        cMesIm1Pt(const cPt2dr & aPt,const std::string & aNameIm,tREAL8 aSigma);
        cMesIm1Pt();

        cPt2dr         mPt;
        std::string    mNamePt;
        tREAL8         mSigma2[3];  // xx xy yy
};

/** class for representing a set of measure in an image*/
class cSetMesPtOf1Im
{
     public :
          cSetMesPtOf1Im(const std::string & aNameIm);
          void AddMeasure(const cMesIm1Pt &);
          void AddData(const  cAuxAr2007 & anAux);

	  void ToFile(const std::string & aNameFile) const;
	  static std::string StdNameFileOfIm(const std::string &);
	  std::string StdNameFile() const;

     private :
          std::string             mNameIm;
          std::vector<cMesIm1Pt>  mMeasures;
};

/**  class for representing  the measure of a point in an image */
class cMes1GCP
{
     public :
        cMes1GCP(const cPt3dr & aPt,const std::string & aNameIm,tREAL8 aSigma);
        cMes1GCP();

        cPt3dr         mPt;
        std::string    mNamePt;
        tREAL8         mSigma2[6];  //  xx xy xz yy yz zz
};



};

#endif  //  _MMVII_MEASURES_IM_H_
