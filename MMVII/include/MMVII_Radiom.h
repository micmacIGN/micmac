#ifndef  _MMVII_RADIOM_H_
#define  _MMVII_RADIOM_H_

#include "MMVII_AllClassDeclare.h"


namespace MMVII
{

class cImageRadiomData
{
   public :
        typedef size_t               tIndex;
        typedef cPt2di               tPtMem;
        typedef tU_INT2              tRadiom;
	typedef std::vector<tRadiom> tVRadiom;

        cImageRadiomData(const std::string & aNameIm,int aNbChanel,bool withPoint);

	void AddObsGray(tIndex,tRadiom);
	void AddObsGray(tIndex,tRadiom,const tPtMem &);

	void AddData(const  cAuxAr2007 & anAux); ///< Serialization
        void MakeOrdered(cSetIntDyn &);
   private :
	void AddIndex(tIndex);

	bool                   mIndexWellOrdered;
	std::string            mNameIm;
	int                    mNbChanel;
	bool                   mWithPoints;

	std::vector<tIndex>    mVIndex;
	std::vector<tPtMem>    mVPts;
	std::vector<tVRadiom>  mVRadiom;
};

};

#endif  //   #define  _MMVII_RADIOM_H_
