#ifndef  _MMVII_GEOM_TPL_H_
#define  _MMVII_GEOM_TPL_H_

#include "MMVII_Matrix.h"
#include "MMVII_Triangles.h"
#include "MMVII_ImageInfoExtract.h"


namespace MMVII
{
template <class Type,class FuncDist,class Param>
    void Tpl_SelectOnSpatialCriteria
         (
            std::vector<Type> & aVecOut,
            const std::vector<Type> & aVecIn,
            size_t aNbSel,
            const FuncDist & aFDist,
            const Param &  aParam
         )
{
   // if size in is < to required, just fix Out=In
   if (aVecIn.size()<=aNbSel)
   {
       aVecOut = aVecIn;
       return;
   }

   size_t aNbIn = aVecIn.size();
   std::vector<tREAL8> aVDistMin(aNbIn,1e8); // store Min Dist to selected object

   while (aNbSel)
   {
       aNbSel--;
       // extract object which is at max dist of  selct
       cWhichMax<size_t,tREAL8> aMaxD;
       for (size_t aKObj=0 ; aKObj<aNbIn ; aKObj++)
           aMaxD.Add(aKObj,tREAL8(aVDistMin.at(aKObj)));

       // Add the object to result
       size_t aKMax = aMaxD.IndexExtre();
       const Type & aNewObj = aVecIn.at(aKMax);
       aVecOut.push_back(aNewObj);

       // update distance to selected object
       for (size_t aKObj=0 ; aKObj<aNbIn ; aKObj++)
       {
           const Type & aCurObj = aVecIn.at(aKObj);
           UpdateMin(aVDistMin.at(aKObj),aFDist(aCurObj,aNewObj,aParam));
       }
   }
}



};


#endif  // _MMVII_GEOM_TPL_H_

