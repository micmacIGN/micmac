#include "include/MMVII_all.h"

namespace MMVII
{


template <class Type> cRotation3D<Type>::cRotation3D(const cDenseMatrix<Type> & aMat,bool RefineIt) :
   mMat (aMat)
{
   MMVII_INTERNAL_ASSERT_always((! RefineIt),"Refine to write in Rotation ...");
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::CompleteRON(const tPt & aP0Init)
{
    cPtxd<Type,3> aP0 = VUnit(aP0Init);
    cPtxd<Type,3> aP1 = VUnit(VOrthog(aP0));
    cPtxd<Type,3> aP2 = aP0 ^ aP1;

    return  cRotation3D<Type>(MatFromCols(aP0,aP1,aP2),false);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::CompleteRON(const tPt & aP0Init,const tPt & aP1Init)
{
    cPtxd<Type,3> aP0 = VUnit(aP0Init);
    cPtxd<Type,3> aP2 = VUnit(aP0 ^ aP1Init);
    cPtxd<Type,3> aP1 = aP2 ^ aP0;

    return  cRotation3D<Type>(MatFromCols(aP0,aP1,aP2),false);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::RotFromAxe(const tPt & aP0,Type aTeta)
{
   // Compute a repair with P0 as first axes
   cRotation3D<Type> aRepAxe = CompleteRON(aP0);
   // Extract two other axes
   tPt  aP1 = tPt::Col(aRepAxe.mMat,1);
   tPt  aP2 = tPt::Col(aRepAxe.mMat,2);

   Type aCosT = cos(aTeta); 
   Type aSinT = sin(aTeta);
   // In plane P1,P2 we have the classical formula of 2D rotation
   tPt aQ1 =   aCosT*aP1 + aSinT*aP2;
   tPt aQ2 =  -aSinT*aP1 + aCosT*aP2;
   //  Mat * (aP0,aP1,aP2) = (aP0,aQ1,aQ2)

   return cRotation3D<Type>(MatFromCols(aP0,aQ1,aQ2)* MatFromCols(aP0,aP1,aP2).Transpose(),false);
}

template <class Type> cRotation3D<Type>  cRotation3D<Type>::RandomRot()
{
   tPt aP0 = tPt::PRandUnit();
   tPt aP1 = tPt::PRandUnit();
   while(Cos(aP0,aP1)>0.99)
       aP1 = tPt::PRandUnit();
   return CompleteRON(aP0,aP1);
}

template <class Type> void cRotation3D<Type>::ExtractAxe(tPt & anAxe,Type & aTeta)
{
    cDenseVect<Type> aDVAxe =  mMat.EigenVect(1.0);
    anAxe =  cPtxd<Type,3>::FromVect(aDVAxe);

/*
{
Type aVer;
   std::cout << "VVV " << aVer << "AAAA " << anAxe << " " << mMat * anAxe << "\n";
}
*/

    cRotation3D<Type> aRep = CompleteRON(anAxe);
    cPtxd<Type,3> aP1 = cPtxd<Type,3>::Col(aRep.mMat,1);
    cPtxd<Type,3> aP2 = cPtxd<Type,3>::Col(aRep.mMat,2);

    cPtxd<Type,3> aQ1 = Direct(aP1);
    Type aCosT = Cos(aP1,aQ1);
    Type aSinT = Cos(aP2,aQ1);

    cPt2dr  aRhoTeta = ToPolar(cPt2dr(aCosT,aSinT));  // To change with templatized ToPolar when exist

    MMVII_INTERNAL_ASSERT_medium(std::abs(aRhoTeta.x()-1.0)<1e-5,"Axes from rot");
    aTeta = aRhoTeta.y();
}



/*
    U D tV X =0   U0 t.q D(U0) = 0   , Ker => U0 = tV X,    X = V U0
*/


/*

void F()
{
  cRotation3D<double>  aR = cRotation3D<double>::RandomRot();
}
template   cRotation3D<double> cRotation3D<double>::RandomRot();
*/

/* ========================== */
/*          ::                */
/* ========================== */


/*
*/
#define MACRO_INSTATIATE_PTXD(TYPE)\
template class  cRotation3D<TYPE>;

/*
template  cRotation3D<TYPE>  cRotation3D<TYPE>::CompleteRON(const tPt & );\
template  cRotation3D<TYPE>  cRotation3D<TYPE>::CompleteRON(const tPt &,const tPt &);\
template  cRotation3D<TYPE>::cRotation3D(const cDenseMatrix<TYPE> & ,bool ) ;\
template  cRotation3D<TYPE>  cRotation3D<TYPE>::RotFromAxe(const tPt & ,TYPE );\
template  cRotation3D<TYPE> cRotation3D<TYPE>::RandomRot();
*/

/*
template <class Type> cRotation3D<Type>  cRotation3D<Type>::CompleteRON(const tPt & aP0Init)
*/


MACRO_INSTATIATE_PTXD(tREAL4)
MACRO_INSTATIATE_PTXD(tREAL8)
MACRO_INSTATIATE_PTXD(tREAL16)



};
