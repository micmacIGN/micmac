#ifndef  _MMVII_TplLayer3D_H_
#define  _MMVII_TplLayer3D_H_

#include "MMVII_Matrix.h"

namespace MMVII
{


/** \file MMVII_TplLayers3D.h
    \brief  Class for images/object between 2 layers, as used in multi-scale match
    and SGM
*/

template <class TObj,class TLayer>  class cLayerData3D
{
    public :
        friend class cLayer3D<TObj,TLayer>;
        friend void TestLayer3D(const cPt2di & aSz);

        typedef TObj              tVal;
        typedef tVal*             tPVal;
        typedef tPVal*            tPPVal;
        typedef cIm2D<TLayer>     tIm;
        typedef cDataIm2D<TLayer> tDIm;

        // Not private because called by shared_ptr ...
        //
        ~cLayerData3D()
        {
             cMemManager::FreeMat(mP3Obj,mSzXY.y());
             cMemManager::Free(mVecObj);
        }
        const tDIm & DZMin() const  {return mDZMin;}  ///< Accessor to bellow layer
        const tDIm & DZMax() const  {return mDZMax;}  ///< Accessor to uper Layer
        int ZMin(const cPt2di & aP) const {return mDZMin.GetV(aP);} ///< Bellow for one pix
        int ZMax(const cPt2di & aP) const {return mDZMax.GetV(aP);}  ///< Up for one pix
        bool IsInside(const cPt2di &aP,const int & aZ) const  ///< Is the 3 point in the volume where data are stored
        {
             return  mDZMin.Inside(aP) && (aZ>=ZMin(aP)) && (aZ<ZMax(aP)) ;
        }
        const tVal & GetV(const cPt2di & aP,const int & aZ)  const  ///< Read value
        {
           AssertInside(aP,aZ);
           return Value(aP,aZ);
        }
        void  SetV(const cPt2di & aP,const int & aZ,const tVal & aVal)  ///< Write Value
        {
           AssertInside(aP,aZ);
           Value(aP,aZ) = aVal;
        }
   private :
        cLayerData3D(const cLayerData3D &) =  delete;
        void AssertInside(const cPt2di &aP,const int & aZ) const
        {
            MMVII_INTERNAL_ASSERT_tiny(IsInside(aP,aZ),"Outside in cLayerData3D");
        }
        tVal & Value(const cPt2di &aP,const int & aZ)              {return  mP3Obj[aP.y()][aP.x()][aZ];}
        const tVal & Value(const cPt2di &aP,const int & aZ) const  {return  mP3Obj[aP.y()][aP.x()][aZ];}
        /*
        Type & Value(const cPt3di & aP)               {return mRawData3D[aP.z()][aP.y()][aP.x()];} ///< Data Access
        const Type & Value(const cPt3di & aP) const   {return mRawData3D[aP.z()][aP.y()][aP.x()];} /// Const Data Access
        */

        cLayerData3D(const tIm & aZMin,const tIm & aZMax) :
           mZMin    (aZMin.Dup()),
           mZMax    (aZMax.Dup()),
           mDZMin   (mZMin.DIm()),
           mDZMax   (mZMax.DIm()),
           mSzXY     (mDZMin.Sz()),
           mNbTotEl (0)
        {
            MMVII_INTERNAL_ASSERT_medium(mDZMin.Sz()==mDZMax.Sz(),"Different size");
            MMVII_INTERNAL_ASSERT_medium(mDZMin.P0()==cPt2di(0,0),"Non origin ZMin in cLayerData3D");
            MMVII_INTERNAL_ASSERT_medium(mDZMax.P0()==cPt2di(0,0),"Non origin ZMin in cLayerData3D");
            for (const auto& aP:mDZMin)
            {
                int aNbEl = mDZMax.GetV(aP) - mDZMin.GetV(aP);
                MMVII_INTERNAL_ASSERT_tiny(aNbEl>=0,"Negative number in cLayerData3D");
                mNbTotEl += aNbEl;
            }
            MMVII_INTERNAL_ASSERT_tiny(mNbTotEl>0,"Null e number tot in cLayerData3D");
            mVecObj =  cMemManager::Alloc<tVal>(mNbTotEl);
            mP3Obj  =  cMemManager::AllocMat<tVal*>(mSzXY.x(),mSzXY.y());
            mNbTotEl = 0;
            for (const auto& aP:mDZMin)
            {
                int aNbEl = mDZMax.GetV(aP) - mDZMin.GetV(aP);
                mP3Obj[aP.y()][aP.x()] = mVecObj  + mNbTotEl - mDZMin.GetV(aP);
                mNbTotEl += aNbEl;
            }
        }

        tIm        mZMin;
        tIm        mZMax;
        tDIm&      mDZMin;
        tDIm&      mDZMax;
        cPt2di     mSzXY;
        int        mNbTotEl;
        tVal *     mVecObj;  ///< raw pointer to vect of all data
        tVal ***   mP3Obj;   ///< 3d pointer directly accessible
};

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

	   // Return a "fake" layer 3D 1x1x1 voxel, usuable for default initialisation
	   static cLayer3D  Empty()
	   {
                return cLayer3D
			(
			    tIm(cPt2di(1,1),nullptr,eModeInitImage::eMIA_Null),
			    tIm(cPt2di(1,1),nullptr,eModeInitImage::eMIA_V1)
			);
	   }

           const tLD3d & LD3D() const  {return *mDL3d;}
           tLD3d & LD3D()              {return *mDL3d;}

       private :
            std::shared_ptr<tLD3d> mSPtr;  ///< shared pointer to real image , allow automatic deallocation
            tLD3d *                mDL3d;   ///
};


};

#endif  //   _MMVII_TplLayer3D_H_
