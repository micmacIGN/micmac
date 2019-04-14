#ifndef  _MMVII_Images_H_
#define  _MMVII_Images_H_
namespace MMVII
{


/** \file MMVII_Images.h
    \brief Application for storing
*/

/** The purpose of this  class is to make some elementary test on image class,
for this it is declared friend  of many image class, allowing to test atomic function
who  will be private for standard classes/functions .
*/

class cBenchBaseImage;
/// Idem cBenchBaseImage
class cBenchImage;

template <const int Dim>  class cRectObj;

/**  Class allow to iterate the pixel of an image (in fact a rectangular object) using the
same syntax than stl iterator => for (auto aP : anIma) ....
*/
template <const int Dim>  class cRectObjIterator
{
     public :
        friend class cRectObj<Dim>;

        bool operator == (const cRectObjIterator<Dim> aIt2) {return  mPCur==aIt2.mPCur;}
        bool operator != (const cRectObjIterator<Dim> aIt2) {return  mPCur!=aIt2.mPCur;}
        cPtxd<int,Dim> & operator * () {return mPCur;}
        const cPtxd<int,Dim> & operator * () const {return mPCur;}
        cPtxd<int,Dim> * operator ->() {return &mPCur;}
        const cPtxd<int,Dim> * operator ->() const {return &mPCur;}

        /// standard prefix incrementation
        cRectObjIterator<Dim> &  operator ++(); 
        /// Just a "facility" to allow post-fix
        cRectObjIterator<Dim> &  operator ++(int) {return ++(*this);} 
     private :
        cRectObjIterator(cRectObj<Dim> & aRO,const  cPtxd<int,Dim> & aP0) : mRO (&aRO),mPCur (aP0) {}
        cRectObj<Dim> * mRO;  ///< The rectangular object
        cPtxd<int,Dim> mPCur; ///< The current point 
};




template <const int Dim>  class cRectObj
{
    public : 
        typedef cRectObjIterator<Dim> iterator;
        


        cRectObj(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1);


        const cPtxd<int,Dim> & P0() const {return mP0;} ///< Origin of object
        const cPtxd<int,Dim> & P1() const {return mP1;} ///< End of object
        const cPtxd<int,Dim> & Sz() const {return mSz;} ///< Size of object

        const tINT8 & NbElem() const {return mNbElem;}  ///< Number of "pixel"

        /// Is this point/pixel/voxel  inside
        bool Inside(const cPtxd<int,Dim> & aP) const {return SupEq(aP,mP0) && InfStr(aP,mP1);}

        /// Assert that it is inside
        void AssertInside(const cPtxd<int,Dim> & aP) const
        {
             MMVII_INTERNAL_ASSERT_tiny(Inside(aP),"Point out of image");
        }
        // const cPtxd<int,Dim>  Begin() {return mP0;}
        // const cPtxd<int,Dim>  End()   {return mP1;}
        // tINT8   NumInBloc(const cPtxd<int,Dim> & aP);
        iterator &  begin() {return mBegin;}
        iterator &  end()   {return mEnd;}

    protected :
        cPtxd<int,Dim> mP0;
        cPtxd<int,Dim> mP1;
        cPtxd<int,Dim> mSz;
        cRectObjIterator<Dim> mBegin;
        cRectObjIterator<Dim> mEnd;
        cPtxd<tINT8,Dim> mSzCum;
        tINT8            mNbElem;
};

/*
template <> inline tINT8  cRectObj<1>::NumInBloc(const cPtxd<int,1> & aP)
{
    return (aP.x() - mP0.x());
}
template <> inline tINT8  cRectObj<2>::NumInBloc(const cPtxd<int,2> & aP)
{
    return (aP.x() - mP0.x()) + (aP.y()-mP0.y()) * mSzCum.x();
}
template <> inline tINT8  cRectObj<3>::NumInBloc(const cPtxd<int,3> & aP)
{
    return (aP.x() - mP0.x()) + (aP.y()-mP0.y()) * mSzCum.x() + (aP.z()-mP0.z()) * mSzCum.y() ;
}
*/


template <> inline cRectObjIterator<1> &  cRectObjIterator<1>::operator ++() { mPCur.x()++; return *this;}
template <> inline cRectObjIterator<2> &  cRectObjIterator<2>::operator ++() 
{
    mPCur.x()++; 
    if (mPCur.x() == mRO->P1().x())
    {
        mPCur.x() = mRO->P0().x();
        mPCur.y()++;
    }

    return *this;
}
template <> inline cRectObjIterator<3> &  cRectObjIterator<3>::operator ++() 
{
    mPCur.x()++; 
    if (mPCur.x() == mRO->P1().x())
    {
        mPCur.x() = mRO->P0().x();
        mPCur.y()++;
        if (mPCur.y() == mRO->P1().y())
        {
           mPCur.y() = mRO->P0().y();
           mPCur.z()++;
        }
    }

    return *this;
}



template <class Type,const int Dim> class cBaseImage : public cRectObj<Dim>,
                                                       public cMemCheck
{
    public :
        friend class cBenchImage;
        friend class cBenchBaseImage;

        typedef Type  tVal;
        typedef tNumTrait<Type> tTraits;
        typedef typename tTraits::tBase  tBase;
        typedef cRectObj<Dim>            tRO;

        const tINT8 & NbElem() const {return tRO::NbElem();}
        const cPtxd<int,Dim> & P0() const {return tRO::P0();}
        const cPtxd<int,Dim> & P1() const {return tRO::P1();}
        const cPtxd<int,Dim> & Sz() const {return tRO::Sz();}


        //========= fundamental access to values ============

           /// Get Value, integer coordinates
        virtual const Type & V_GetV(const cPtxd<int,Dim> & aP)  const = 0;
           /// Set Value, integer coordinates
        virtual void V_SetV(const  cPtxd<int,Dim> & aP,const tBase & aV) = 0;

        //========= Test access ============

        inline bool Inside(const cPtxd<int,Dim> & aP) const {return tRO::Inside(aP);}
        inline bool OkOverFlow(const tBase & aV) const {return tTraits::OkOverFlow(aV);}

        cBaseImage (const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,Type * DataLin=0);
        ~cBaseImage();
    protected :
        void AssertNonOverFlow(const tBase & aV) const
        {
             MMVII_INTERNAL_ASSERT_tiny(OkOverFlow(aV),"Point out of image");
        }

        bool   mDoAlloc;
        Type *   mDataLin;
};





template <class Type>  class cDataIm2D  : public cBaseImage<Type,2>
{
    public :
        friend class cBenchImage;
        friend class cBenchBaseImage;
        typedef Type  tVal;
        typedef cBaseImage<Type,2>   tBI;
        typedef cRectObj<2>          tRO;
        typedef typename tBI::tBase  tBase;

        //========= fundamental access to values ============

           /// Get Value
        const Type & GetV(const cPt2di & aP)  const
        {
            tRO::AssertInside(aP);
            return  Value(aP);
        }
        const Type & V_GetV(const cPtxd<int,2> & aP)  const override {return GetV(aP);}

        void SetV(const cPt2di & aP,const tBase & aV)
        { 
            tRO::AssertInside(aP);
            tBI::AssertNonOverFlow(aV);
            Value(aP) = aV;
        }
        void V_SetV(const  cPtxd<int,2> & aP,const tBase & aV) override {SetV(aP,aV);}

        //========= Access to sizes ============
        const cPt2di &  Sz()  const {return tRO::Sz();}
        const int    &  SzX() const {return Sz().x();}
        const int    &  SzY() const {return Sz().y();}

        const cPt2di &  P0()  const {return tRO::P0();}
        const int    &  X0()  const {return P0().x();}
        const int    &  Y0()  const {return P0().y();}

        const cPt2di &  P1()  const {return tRO::P1();}
        const int    &  X1()  const {return P1().x();}
        const int    &  Y1()  const {return P1().y();}

        // const cPt2di &  Sz()  const {return cRectObj<2>::Sz();}


        // typedef Type* tPtrVal;
        // typedef typename  cBaseImage<Type,2>::tBase  tBase;

        cDataIm2D(const cPt2di & aP0,const cPt2di & aP1,Type * DataLin=0);
        ~cDataIm2D();
    protected :
    private :
        Type & Value(const cPt2di & aP)   {return mData[aP.y()][aP.x()];}
        const Type & Value(const cPt2di & aP) const   {return mData[aP.y()][aP.x()];}
        Type ** mData;
};



};

#endif  //  _MMVII_Images_H_
