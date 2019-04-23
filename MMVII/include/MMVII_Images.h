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


template <const int Dim>  class cRectObjIterator;
template <const int Dim>  class cRectObj;
template <class Type,const int Dim> class cDataImGen ;
template <class Type>  class cDataIm2D  ;
template <class Type>  class cIm2D  ;

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
        static const cRectObj Empty00;

        cRectObj(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1);


        const cPtxd<int,Dim> & P0() const {return mP0;} ///< Origin of object
        const cPtxd<int,Dim> & P1() const {return mP1;} ///< End of object
        const cPtxd<int,Dim> & Sz() const {return mSz;} ///< Size of object

        const tINT8 & NbElem() const {return mNbElem;}  ///< Number of "pixel"

        // Boolean operators
           /// Is this point/pixel/voxel  inside
        bool Inside(const cPtxd<int,Dim> & aP) const  {return SupEq(aP,mP0) && InfStr(aP,mP1);}
        cPtxd<int,Dim> Proj(const cPtxd<int,Dim> & aP) const {return PtInfStr(PtSupEq(aP,mP0),mP1);}
        bool operator == (const cRectObj<Dim> aR2) const ;
        bool  IncludedIn(const cRectObj<Dim> &)const;
        cRectObj<Dim> Translate(const cPtxd<int,Dim> & aPt)const;

        /// Assert that it is inside
        void AssertInside(const cPtxd<int,Dim> & aP) const
        {
             MMVII_INTERNAL_ASSERT_tiny(Inside(aP),"Point out of image");
        }
        //  --- Iterator ----------------
        iterator &  begin() {return mBegin;}
        iterator &  end()   {return mEnd;}
        iterator   begin() const {return mBegin;}
        iterator   end()   const {return mEnd;}

        //  ---  object generation ----------------

             /// [0,1] * => Rect
        cPtxd<int,Dim>  FromNormaliseCoord(const cPtxd<double,Dim> &) const ;
        static cPtxd<double,Dim>  RandomNormalised() ;
        cPtxd<int,Dim>  GeneratePointInside() const;
        cRectObj<Dim>  GenerateRectInside(double aPowSize=1.0) const;

    protected :
        cPtxd<int,Dim> mP0;
        cPtxd<int,Dim> mP1;
        cPtxd<int,Dim> mSz;
        cRectObjIterator<Dim> mBegin;
        cRectObjIterator<Dim> mEnd;
        cPtxd<tINT8,Dim> mSzCum;
        tINT8            mNbElem;
};

typedef  cRectObj<1> cRect1;
typedef  cRectObj<2> cRect2;
typedef  cRectObj<3> cRect3;


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



template <class Type,const int Dim> class cDataImGen : public cRectObj<Dim>,
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

        Type * DataLin() {return  mDataLin;}
        void InitRandom();

        //========= fundamental access to values ============

           /// Get Value, integer coordinates
        virtual const Type & V_GetV(const cPtxd<int,Dim> & aP)  const = 0;
           /// Set Value, integer coordinates
        virtual void V_SetV(const  cPtxd<int,Dim> & aP,const tBase & aV) = 0;

        //========= Test access ============

        inline bool Inside(const cPtxd<int,Dim> & aP) const {return tRO::Inside(aP);}
        inline bool OkOverFlow(const tBase & aV) const {return tTraits::OkOverFlow(aV);}

        cDataImGen (const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,Type * DataLin=0);
        virtual ~cDataImGen();
    protected :
        void AssertNonOverFlow(const tBase & aV) const
        {
             MMVII_INTERNAL_ASSERT_tiny(OkOverFlow(aV),"Value out of image");
        }

        bool   mDoAlloc;
        Type *   mDataLin;
};

class cDataFileIm2D : public cRect2
{
     public :
        const cPt2di & Sz() const ;
        const int  & NbChannel ()  const ;
        const eTyNums &   Type ()  const ;
        const std::string &  Name() const;
        /// Create a descriptor on existing file
        static cDataFileIm2D Create(const std::string & aName);
        /// Create the file before returning the descriptor
        static cDataFileIm2D Create(const std::string & aName,eTyNums,const cPt2di & aSz,int aNbChan=1);

        virtual ~cDataFileIm2D();
        
     private :
        cDataFileIm2D(const std::string &,eTyNums,const cPt2di & aSz,int aNbChannel) ;

        std::string  mName;
        eTyNums      mType;
        int          mNbChannel;
};


template <class Type>  class cDataIm2D  : public cDataImGen<Type,2>
{
    public :
        friend class cBenchImage;
        friend class cBenchBaseImage;
        friend class cIm2D<Type>;

        typedef Type  tVal;
        typedef cDataImGen<Type,2>   tBI;
        typedef cRect2               tRO;
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

        void SetVTrunc(const cPt2di & aP,const tBase & aV)
        { 
            tRO::AssertInside(aP);
            Value(aP) = tNumTrait<Type>::Trunc(aV);
        }


        void V_SetV(const  cPtxd<int,2> & aP,const tBase & aV) override {SetV(aP,aV);}

        //========= Access to sizes, only alias/facilities ============
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
        // typedef typename  cDataImGen<Type,2>::tBase  tBase;

        void Read(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
        void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
        virtual ~cDataIm2D();
    protected :
    private :
        cDataIm2D(const cDataIm2D<Type> &) = delete;
        cDataIm2D(const cPt2di & aP0,const cPt2di & aP1,Type * DataLin=0);


        Type & Value(const cPt2di & aP)   {return mData[aP.y()][aP.x()];}
        const Type & Value(const cPt2di & aP) const   {return mData[aP.y()][aP.x()];}
        Type ** mData;
};

template <class Type>  class cIm2D  
{
    public :
       typedef cDataIm2D<Type>  tDIM;
       cIm2D(const cPt2di & aP0,const cPt2di & aP1,Type * DataLin=0);
       cIm2D(const cPt2di & aSz,Type * DataLin=0);


       tDIM & Im() {return *(mSPtr.get());}
      
       void Read(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
       void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
       static cIm2D<Type> FromFile(const std::string& aName);

       // void Read(const cDataFileIm2D &,cPt2di & aP0,cPt3dr Dyn /* RGB*/);  // 3 to 1
       // void Read(const cDataFileIm2D &,cPt2di & aP0,cIm2D<Type> aI2,cIm2D<Type> aI3);  // 3 to 3
    private :
       std::shared_ptr<tDIM> mSPtr;
};



};

#endif  //  _MMVII_Images_H_
