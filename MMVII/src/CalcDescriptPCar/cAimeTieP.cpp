#include "include/MMVII_all.h"


namespace MMVII
{

class cAimeDescriptor
{
     public :
         cAimeDescriptor();
         cIm2D<tU_INT1>   ILP();
     private :
        cIm2D<tU_INT1>   mILP;   ///< mImLogPol
};

class cAimePCar
{
     public :
     private :
};

/* ================================= */
/*          cProtoAimeTieP           */
/* ================================= */

template<class Type> 
   cProtoAimeTieP<Type>::cProtoAimeTieP
   (
        cGP_OneImage<Type> * aGPI,
        const cPt2di & aPInit
   ) :
   mGPI        (aGPI),
   mPImInit    (aPInit),
   mPFileInit  (mGPI->Im2File(ToR(mPImInit)))
{
}

template<class Type> int   cProtoAimeTieP<Type>::NumOct()   const {return mGPI->Oct()->NumInPyr();}
template<class Type> int   cProtoAimeTieP<Type>::NumIm()    const {return mGPI->NumInOct();}
template<class Type> float cProtoAimeTieP<Type>::ScaleInO() const {return mGPI->ScaleInO();}
template<class Type> float cProtoAimeTieP<Type>::ScaleAbs() const {return mGPI->ScaleAbs();}

template<class Type> void   cProtoAimeTieP<Type>::FillAPC(cAimePCar &)
{
}


template class cProtoAimeTieP<tREAL4>;
template class cProtoAimeTieP<tINT2>;


/* ================================= */
/*          cAimeDescriptor          */
/* ================================= */


cAimeDescriptor:: cAimeDescriptor() :
   mILP  (cPt2di(1,1))
{
}

/* ================================= */
/*             cAimePCar             */
/* ================================= */



};
