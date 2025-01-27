#include "MMVII_SysSurR.h"

/* ===================================================== */
/* ===================================================== */
/* ===================================================== */


namespace MMVII
{

class cBenchLstSqEstimUncert
{
   public :
          cBenchLstSqEstimUncert(int aDim,const std::vector<int> & aVecNbByObs);

    private :
         int                   mDim;
         cDecomposPAdikVar     mDecompos;

};

/*
tREAL8  EigenDeg(std::vector<cDenseVect<tREAL8> aVV)
{
    // not sur it's the good convention, maybe an error ?
    if (aVV.empty()) 
       return 0.0;
    int aDimVect = aVV.at(0).Sz();


    int aDimMat = 
}
*/


cBenchLstSqEstimUncert::cBenchLstSqEstimUncert(int aDim,const std::vector<int> & aVecNbByObs) :
    mDim       (aDim),
    mDecompos  (aVecNbByObs)
{
/*
    for (size_t aK=0 ; aK<aVecNbByObs.size() ; K++)
    {
         cDenseVect<tREAL8> aV(mDim,eModeInitImage::eMIA_RandCenter);
    }


    StdOut()<< " ============ LSQT4 :" << aVecNbByObs << "\n";
    for (int aK=0 ; aK<mDecompos.MulBase() ;aK++)
    {
        StdOut() << "  * V=" <<  mDecompos.DecomposSizeBase(aK) << "\n";

    }
*/
}

void BenchLstSqEstimUncert(cParamExeBench & aParam)
{
    if (! aParam.NewBench("LstSqUncert")) return;

    cBenchLstSqEstimUncert  aLstQ(1,{2,3});

   

    aParam.EndBench();
}





};


