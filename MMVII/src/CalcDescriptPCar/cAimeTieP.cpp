#include "include/MMVII_all.h"
#include "AimeTieP.h"


namespace MMVII
{
/* ================================= */
/*          cProtoAimeTieP           */
/* ================================= */
cProtoAimeTieP::cProtoAimeTieP(const cPt2dr & aPt,int aNumOct,int aNumIm,float aScaleInO,float aScaleAbs) :
   mPt        (aPt),
   mNumOct    (aNumOct),
   mNumIm     (aNumIm),
   mScaleInO  (aScaleInO),
   mScaleAbs  (aScaleAbs)
{
}

const cPt2dr & cProtoAimeTieP::Pt() const  {return mPt;}
int   cProtoAimeTieP::NumOct() const       {return mNumOct;}
int   cProtoAimeTieP::NumIm() const        {return mNumIm;}
float cProtoAimeTieP::ScaleInO() const     {return mScaleInO;}
float cProtoAimeTieP::ScaleAbs() const     {return mScaleAbs;}

};
