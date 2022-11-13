#ifndef  _MMVII_RADIOM_H_
#define  _MMVII_RADIOM_H_

#include "MMVII_AllClassDeclare.h"
#include "MMVII_memory.h"


namespace MMVII
{

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



};

#endif  //   #define  _MMVII_RADIOM_H_
