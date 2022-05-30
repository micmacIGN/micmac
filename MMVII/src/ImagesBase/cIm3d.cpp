#include "include/MMVII_all.h"
#include "include/MMVII_TplLayers3D.h"

namespace MMVII
{

/**  Class used to represent "non full" 3D images (or more generally container for any
    object) domain defined by ZMin/ZMax.

    As it may be instandiated with any value, put it in header with all inline ....
    
    As usuall  there is the cLayerData3D / cLayer3D
 */


#if (0)
template <class TObj,class TLayer>  class cLayer3D  
{
       public :
           typedef cLayerData3D<TObj,TLayer>  tLD3d;
           typedef cIm2D<TLayer>     tIm;

           cLayer3D(const tIm & aZMin,const tIm & aZMax) :
              mSPtr  (new tLD3d(aZMin,aZMax)),
              mDL3d  (mSPtr.get())
           {
           }

	   const tLD3d & LD3D() const  {return *mDL3d;}
	   tLD3d & LD3D()              {return *mDL3d;}

       private :
            std::shared_ptr<tLD3d> mSPtr;  ///< shared pointer to real image , allow automatic deallocation
            tLD3d *                mDL3d;   ///
};
#endif

/* ****************************************************** */
/*                                                        */
/*             cDataIm3D<Type>                            */
/*                                                        */
/* ****************************************************** */

template <class Type>  cDataIm3D<Type>::cDataIm3D(const cPt3di & aSz,Type * aRawDataLin,eModeInitImage aModeInit) : 
    cDataTypedIm<Type,3> (cPt3di(0,0,0),aSz,aRawDataLin,aModeInit),
    mRawData3D           (cMemManager::AllocMat<tPVal>(Sz().y(),Sz().z()))
{

    Type *  aRDL =  tBI::mRawDataLin ;   // Pointer to raw data, will be increased in loop
    for (int aZ=0 ; aZ<Sz().z() ;aZ++)
    {
        for (int aY=0 ; aY<aSz.y() ; aY++)
        {
            mRawData3D[aZ][aY]  = aRDL;
	    aRDL += Sz().x();
        }
    }
}

template <class Type>  cDataIm3D<Type>::~cDataIm3D()
{
    cMemManager::FreeMat(mRawData3D,Sz().z());
}

/* ************************************************** */
/*                                                    */
/*             cIm3D<Type>                            */
/*                                                    */
/* ************************************************** */

template <class Type>  cIm3D<Type>::cIm3D(const cPt3di & aSz,Type * aRawDataLin,eModeInitImage aModeInit) :
   mSPtr(new cDataIm3D<Type>(aSz,aRawDataLin,aModeInit)),
   mPIm (mSPtr.get())
{
}

template <class Type>  cIm3D<Type>::cIm3D(const cPt3di & aSz) :
    cIm3D<Type>(aSz,nullptr,eModeInitImage::eMIA_NoInit)
{
}
/* ************************************************** */
/*                                                    */
/*             BENCH                                  */
/*                                                    */
/* ************************************************** */

template <class Type> Type  TestFuncXYZ(const cPt3di & aP)
{
    double aVal =  aP.x() - aP.y() * 2.31 +  (3.0*aP.z())/(1.5+aP.x()+aP.z());
    return Type(aVal);
}

cPt2di   TestPtXYZ(const cPt2di & aP,const int aZ)
{
    return cPt2di
	   (
	          TestFuncXYZ<int>(cPt3di(aP.x(),aP.y(),aZ)),
	          TestFuncXYZ<int>(cPt3di(aZ,aP.x(),aP.y()))
	   );
}

static int  TestFuncZMin(const cPt2di & aP)
{
    return  1.2 + aP.x() -1.35*aP.y();
}
static int  TestFuncNbZ(const cPt2di & aP)
{
    return   (1 + aP.x() + aP.y()) % 4;
}

void TestLayer3D(const cPt2di & aSz)
{
   cIm2D<tINT2> aZMin(aSz);
   cIm2D<tINT2> aZMax(aSz);

   int aNbZ = 0;
   for (const auto & aP : aZMin.DIm())
   {
       int aZ0 = TestFuncZMin(aP);
       int aZ1 = aZ0 + TestFuncNbZ(aP);
       aZMin.DIm().SetV(aP,aZ0);
       aZMax.DIm().SetV(aP,aZ1);
       aNbZ += aZ1-aZ0;
   }
   // cLayer3D<cPt2di,tINT2> aL3d(aZMin,aZMax);
   cLayer3D<cPt2di,tINT2> aL3d = cLayer3D<cPt2di,tINT2>::Empty();
   aL3d =  cLayer3D<cPt2di,tINT2>(aZMin,aZMax);
   cLayerData3D<cPt2di,tINT2> & aLD = aL3d.LD3D();

   cPt2di aCheckP(0,0);
   for (const auto & aP : aZMin.DIm())
   {
       int aZ0 = aLD.ZMin(aP);
       int aZ1 = aLD.ZMax(aP);
       aNbZ -= aZ1-aZ0;

       for (int aZ= aZ0 ; aZ< aZ1 ;aZ++)
       {
            cPt2di aVal = TestPtXYZ(aP,aZ); 
            aLD.SetV(aP,aZ,aVal);
	    aCheckP = aCheckP +aVal;
       }
   }

   for (const auto & aP : aZMin.DIm())
   {
       int aZ0 = aLD.ZMin(aP);
       int aZ1 = aLD.ZMax(aP);
       for (int aZ= aZ0 ; aZ< aZ1 ;aZ++)
       {
            cPt2di aVal = aLD.GetV(aP,aZ);
            MMVII_INTERNAL_ASSERT_bench(aVal==TestPtXYZ(aP,aZ),"Layer 3D Read/Write ");
	    aCheckP = aCheckP -aVal;
       }
   }

   MMVII_INTERNAL_ASSERT_bench(aNbZ==0,"TplBenchIm3D Nb Elem");
   MMVII_INTERNAL_ASSERT_bench(aCheckP==cPt2di(0,0),"TplBenchIm3D Nb Elem");
}


template  class cLayerData3D<cPt2di,tINT2>;

/*  Check what you write is what you read + memory alloc/unalloc + number of elem
 */
template <class Type> void TplBenchIm3D(const cPt3di & aSz)
{
    cIm3D<Type>     aIm(aSz);
    cDataIm3D<Type>&aDIm = aIm.DIm();

    for (const auto & aP : aDIm)
    {
        Type aVal = TestFuncXYZ<Type>(aP);
        aDIm.SetV(aP,aVal);
    }
    int aNb = aSz.x()* aSz.y() * aSz.z();
    for (const auto & aP : aDIm)
    {
        Type aVal = TestFuncXYZ<Type>(aP);
	MMVII_INTERNAL_ASSERT_bench(aVal==aDIm.GetV(aP),"TplBenchIm3D image get/set");
        aNb--;
    }
    MMVII_INTERNAL_ASSERT_bench(aNb==0,"TplBenchIm3D Nb Elem");
}

void BenchIm3D()
{
     cBox3di aBox(cPt3di::PCste(1),cPt3di::PCste(20));
     for (int aK=0 ; aK<100 ; aK++)
     {
         cPt3di aP =  aBox.GeneratePointInside();
         TplBenchIm3D<tU_INT1>(aP);
         TplBenchIm3D<tINT1>(aP);
         TplBenchIm3D<tREAL4>(aP);

         TestLayer3D(cPt2di(aP.x(),aP.y()));
     }
}


/* ************************************************** */
/*                                                    */
/*             INSTANCIATION                           */
/*                                                    */
/* ************************************************** */

#define INSTANTIATE_IM3D(Type)\
template  class cIm3D<Type>;\
template  class cDataIm3D<Type>;

INSTANTIATE_IM3D(tINT1)
INSTANTIATE_IM3D(tREAL4)

/*
*/
INSTANTIATE_IM3D(tINT2)
INSTANTIATE_IM3D(tINT4)

INSTANTIATE_IM3D(tU_INT1)
INSTANTIATE_IM3D(tU_INT2)
INSTANTIATE_IM3D(tU_INT4)


INSTANTIATE_IM3D(tREAL8)
INSTANTIATE_IM3D(tREAL16)



};
