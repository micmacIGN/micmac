#include "MMVII_util.h"

namespace MMVII
{
/*=============================================*/
/*                                             */
/*      cMultipleOfs::                      */
/*                                             */
/*=============================================*/
cMultipleOfs::cMultipleOfs(std::ostream & aOfs) : mOfsCreated(nullptr)
{
    Add(aOfs);
}
cMultipleOfs::cMultipleOfs(const std::string & aS, bool ModeAppend)
{
    mOfsCreated = nullptr;
    mOfsCreated = new cMMVII_Ofs(aS,ModeAppend);
    Add(mOfsCreated->Ofs());
}
cMultipleOfs::~cMultipleOfs()
{
    if (mOfsCreated != nullptr) delete mOfsCreated;
}

void cMultipleOfs::Add(std::ostream & aOfs)
{
    mVOfs.push_back(&aOfs);
}
void cMultipleOfs::Clear()
{
    mVOfs.clear();
}

};

