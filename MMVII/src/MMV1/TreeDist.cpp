#define WITH_MMVII true

#if (WITH_MMVII)
// #include "include/MMVII_all.h"
#include "include/V1VII.h"
using namespace MMVII;
#else             //========================================================== WITH_MMVI
class cMemCheck
{
};
#include <typeinfo>
#include <math.h>
#include <cassert>

//#include <memory>
//#include <map>
//#include <iostream>
//#include "memory.h"
//#include <memory>
//#include <iostream>
//#include <fstream>
//#include <string>
//#include <vector>
//#include <list>
//#include <map>
//#include <ctime>
//#include <chrono>
//#include <cmath>
//#include <algorithm>
//#include <sstream>
#endif   //========================================================== WITH_MMVI

namespace NS_MMVII_FastTreeDist
{
class cFTD_OneLevel;
class cFastTreeDist;

class cFTD_OneLevel : public cMemCheck
{
    public :
       cFTD_OneLevel(const cFastTreeDist & aFTD);
    private :
       const cFastTreeDist &  mFTD;
 
       // std::vector<bool>  mVIn; ///< Indicate the som present at this level
};

class cAdjGraphSym : public cMemCheck
{
    public :
       cAdjGraphSym(const size_t & aNbSom);
    protected :
        void InitAdj(const std::vector<int> & aVS1,const std::vector<int> & aVS2); 
        void AssertOk(int aNumS) const;

        size_t mNbSom;
        size_t mNbLevel;
        std::vector<int>    mNbSucc; /// Adj list
};

class cFastTreeDist : public cAdjGraphSym
{
    public :
        //    =====  Constructor / Destructor =====
            /// Copy has no added value, so forbid it to avoid unwanted use
        cFastTreeDist(const cFastTreeDist &) = delete;
            /// just need to know the number of summit to allocate
        cFastTreeDist(const size_t & aNbSom);
        
        /// 
        void MakeDist(const std::vector<int> & aVS1,const std::vector<int> & aVS2); 
    private :
 
        // size_t mNbSom;
        // size_t mNbLevel;
        // std::vector<int>    mNbSucc; /// Adj list
};

/* ---------------------------------- */
/* ---------------------------------- */
/* |                                | */
/* |         cFTD_OneLevel          | */
/* |                                | */
/* ---------------------------------- */
/* ---------------------------------- */

cFTD_OneLevel:: cFTD_OneLevel(const cFastTreeDist & aFTD) :
   mFTD  (aFTD)
{
}

/* ---------------------------------- */
/* ---------------------------------- */
/* |                                | */
/* |       cAdjGraphSym             | */
/* |                                | */
/* ---------------------------------- */
/* ---------------------------------- */

cAdjGraphSym::cAdjGraphSym(const size_t & aNbSom) :
    mNbSom    (aNbSom),
    mNbLevel  (1+ceil(log(1+aNbSom)/log(2.0))),
    mNbSucc   (mNbSom)
{
}

void cAdjGraphSym::InitAdj(const std::vector<int> & aVS1,const std::vector<int> & aVS2)
{
   // -- 0 --   Create adjajency 

       // 0.0  Resset counting number of succ
   for (auto & aCpt : mNbSucc)
      aCpt=0;

       // 0.1  count the number of succ
   assert(aVS1.size()==aVS2.size());
   for (size_t aKS=0 ; aKS<aVS1.size() ; aKS++)
   {
       // chekc they are valide
       AssertOk(aVS1[aKS]);
       AssertOk(aVS2[aKS]);
       // update number of succ
       mNbSucc[aVS1[aKS]]++;
       mNbSucc[aVS2[aKS]]++;
   }
}

void cAdjGraphSym::AssertOk(int aNumS) const
{
    assert(aNumS>=0);
    assert(aNumS<int(mNbSom));
}

/* ---------------------------------- */
/* ---------------------------------- */
/* |                                | */
/* |       cFastTreeDist            | */
/* |                                | */
/* ---------------------------------- */
/* ---------------------------------- */


cFastTreeDist::cFastTreeDist(const size_t & aNbSom) :
    cAdjGraphSym (aNbSom)
{
}

void cFastTreeDist::MakeDist(const std::vector<int> & aVS1,const std::vector<int> & aVS2)
{
}
   
};



namespace MMVII
{
};

