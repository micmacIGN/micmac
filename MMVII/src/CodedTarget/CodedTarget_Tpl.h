#ifndef _CODED_TARGET_TPL_H_
#define _CODED_TARGET_TPL_H_
#include "MMVII_memory.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_SetITpl.h"
#include "FilterCodedTarget.h"   // Contains image processing that may be not specific to coded target

namespace MMVII
{
// Everything specific to coded target in this namespace
namespace  cNS_CodedTarget
{

template   <class TypeExtract,class TypeFSel> 
	void OneMatchOnGT
	     (
	        cGeomSimDCT & aGSD,
	        std::vector<TypeExtract *> &aVExtr,
	        tREAL8 aDMaxMatch, 
	        const TypeFSel & aFuncSel
	      )
{
     // strtucture for extracting min
     cWhichMin<TypeExtract*,double>  aWMin(nullptr,1e10);

     for (auto aPtrDCT : aVExtr)
     {
         if (aFuncSel(aPtrDCT))
             aWMin.Add(aPtrDCT,SqN2(aPtrDCT->mPt-aGSD.mC));
     }

     if (aWMin.ValExtre() < Square(aDMaxMatch))
     {
        aGSD.mResExtr = aWMin.IndexExtre(); // the simul memorize its detected
        aGSD.mResExtr->mGT =& aGSD;         // the detected memorize its ground truth
     }
     else
     {
     }
}

template   <class TypeExtract,class TypeFSel> 
   int AllMatchOnGT
       (
	     cResSimul & aGTResSim,
             std::vector<TypeExtract *> &aVExtr,
	     tREAL8 aDMaxMatch,
	     bool ShowMSG,
	     const TypeFSel & aFuncSel
       )
{
    int aNbGTMatched = 0;
    for (auto & aGSD : aGTResSim.mVG)
    {
        OneMatchOnGT(aGSD,aVExtr,aDMaxMatch,aFuncSel);
        if (aGSD.mResExtr )
           aNbGTMatched++;
        else
        {
           if (ShowMSG)
              StdOut() << " UNMATCH000 at " << aGSD.mC << "\n";
        }
    }

    if (ShowMSG)
       StdOut()  << "GT-MATCHED : %:" << (100.0*aNbGTMatched) /aGTResSim.mVG.size() << " on " << aGTResSim.mVG.size() << " total-GT\n";

    return aNbGTMatched;
}


};     // MMVII
};     // cNS_CodedTarget
#endif // _CODED_TARGET_TPL_H_

