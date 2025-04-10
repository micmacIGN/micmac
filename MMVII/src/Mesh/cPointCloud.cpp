#include "MMVII_PointCloud.h"
#include "MMVII_2Include_Serial_Tpl.h"



namespace MMVII
{
cPointCloud::cPointCloud(bool isM8) :
   mMode8 (isM8)
{
}

void cPointCloud::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("M8",anAux),mMode8);
    MMVII::AddData(cAuxAr2007("Params",anAux),mParams);
    MMVII::AddData(cAuxAr2007("Offset",anAux),mOffset);
    MMVII::AddData(cAuxAr2007("PtsR",anAux),mPtsR);
    MMVII::AddData(cAuxAr2007("PtsF",anAux),mPtsF);
}

void AddData(const  cAuxAr2007 & anAux,cPointCloud & aPC)
{
   aPC.AddData(anAux);
}


cBox3dr  cPointCloud::Box() const
{
   cTplBoxOfPts<tREAL8,3> aTplB;
   for (int aK=0 ; aK<NbPts() ; aK++)
       aTplB.Add(KthPt(aK));
   return aTplB.CurBox();
}



};

