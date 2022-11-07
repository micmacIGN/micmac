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
        typedef cPt2di               tPtMem;
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
	const std::vector<tPtMem> & VPts()               const;
	const tVRadiom            & VRadiom(size_t aKCh) const;
   private :
	void AddIndex(tIndex);
	void CheckAndAdd(tIndex ,tRadiom ,int aNbCh,bool WithPoint);

	bool                   mIndexWellOrdered;
	std::string            mNameIm;
	int                    mNbChanel;
	bool                   mWithPoints;

	std::vector<tIndex>    mVIndex;
	std::vector<tPtMem>    mVPts;
	std::vector<tVRadiom>  mVVRadiom;
};


};

#endif  //   #define  _MMVII_RADIOM_H_
