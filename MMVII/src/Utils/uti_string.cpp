#include "include/MMVII_all.h"

namespace MMVII
{

// cGestObjetEmpruntable<cCarLookUpTable>   cCarLookUpTable::msGOE;

void  cCarLookUpTable::Init(const std::string& aStr,char aC)
{
    MMVII_INTERNAL_ASSERT_medium(!mInit,"Multiple init of  cCarLookUpTable");
    mInit= true;
    for (const char * aPtr = aStr.c_str() ; *aPtr ; aPtr++)
        mTable[*aPtr] = aC;
    mIns = aStr;
}

void  cCarLookUpTable::UnInit()
{
    MMVII_INTERNAL_ASSERT_medium(mInit,"Multiple Uninit of  cCarLookUpTable");
    mInit= false;
    for (const char * aPtr = mIns.c_str() ; *aPtr ; aPtr++)
        mTable[*aPtr] = 0;
    mIns = "";
}

cCarLookUpTable::cCarLookUpTable() :
     mInit(false)
{
    MEM_RAZ(mTable,1);
}

// const char * cCarLookUpTable:Table() 



/*
void SplitString(std::vector<std::string > & aRes,const std::string aSpace)
{
    cCarLookUpTable *  aLut = cMMVII_Appli::TheAppli().GetLut(aSpace,1);
}
*/


};

