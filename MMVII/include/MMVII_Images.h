#ifndef  _MMVII_Images_H_
#define  _MMVII_Images_H_
namespace MMVII
{


/** \file MMVII_Images.h
    \brief Classes for storing images in RAM, possibly N dimention
*/

/** The purpose of this  class is to make some elementary test on image class,
for this it is declared friend  of many image class, allowing to test atomic function
who  will be private for standard classes/functions .
*/

class cBenchBaseImage;
/// Idem cBenchBaseImage
class cBenchImage;


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



/**  Class representing N-dim rectangle in pixel,
     Many basic services to test and visit the rectangle.
     Image (RAM), image (file), windows ... will inherit/contain

     Basically defined by integer point limits P0 and P1, and contain all the pixel Pix
     such that :
           P0[aK] <= Pix [aK] < P1[aK]  for K in [0,Dim[
*/

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
           /// Specialistion 1D
        bool Inside(const int & aX) const  
        {
           // static_assert(Dim==1,"Bas dim for integer access");
           return (aX>=mP0.x()) && (aX<mP1.x());
        }
        cPtxd<int,Dim> Proj(const cPtxd<int,Dim> & aP) const {return PtInfStr(PtSupEq(aP,mP0),mP1);}
        bool operator == (const cRectObj<Dim> aR2) const ;
        bool  IncludedIn(const cRectObj<Dim> &)const;
        cRectObj<Dim> Translate(const cPtxd<int,Dim> & aPt)const;

        /// Assert that it is inside
        template <class TypeIndex> void AssertInside(const TypeIndex & aP) const
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
        cPtxd<int,Dim> mP0;           ///< "smallest"
        cPtxd<int,Dim> mP1;           ///< "highest"
        cPtxd<int,Dim> mSz;           ///<  Size
        cRectObjIterator<Dim> mBegin; ///< Beging iterator
        cRectObjIterator<Dim> mEnd;   ///< Ending iterator
        cPtxd<tINT8,Dim> mSzCum;      ///< Cumlated size : Cum[aK] = Cum[aK-1] * Sz[aK]
        tINT8            mNbElem;     ///< Number of pixel = Cum[Dim-1]
    private :
        cRectObj(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,bool AllowEmpty);
};

typedef  cRectObj<1> cRect1;
typedef  cRectObj<2> cRect2;
typedef  cRectObj<3> cRect3;



/* Iterator allowing to visit rectangles */

template <> inline cRectObjIterator<1> &  cRectObjIterator<1>::operator ++() 
{ 
   mPCur.x()++; 
   return *this;
}
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


/**  Abstract class allowing to manipulate images independanlty of their type
*/


template <const int Dim> class cDataGenUnTypedIm : public cRectObj<Dim>,
                                                   public cMemCheck
{
      public :
        typedef cRectObj<Dim>            tRO;
        const  cRectObj<Dim> & RO() {return *this;}

        cDataGenUnTypedIm(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1);

         
           // Get Value, integer coordinates
                /// Pixel -> Integrer Value
        virtual int VI_GetV(const cPtxd<int,Dim> & aP)  const =0;
                /// Pixel -> float Value
        virtual double VD_GetV(const cPtxd<int,Dim> & aP)  const =0;
           // Set Value, integer coordinates
                /// Set Pixel Integrer Value
        virtual void VI_SetV(const  cPtxd<int,Dim> & aP,const int & aV) =0;
                /// Set Pixel Float Value
        virtual void VD_SetV(const  cPtxd<int,Dim> & aP,const double & aV)=0 ;
};

/**  Classes for   ram-image containg a given type of pixel
*/

template <class Type,const int Dim> class cDataTypedIm : public cDataGenUnTypedIm<Dim>
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

        Type * RawDataLin() {return  mRawDataLin;}
        void InitRandom();

        //========= Test access ============

        inline bool Inside(const cPtxd<int,Dim> & aP) const {return tRO::Inside(aP);}
        inline bool OkOverFlow(const tBase & aV) const {return tTraits::OkOverFlow(aV);}

        cDataTypedIm (const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,Type * DataLin=0);
        virtual ~cDataTypedIm();
    protected :
        void AssertNonOverFlow(const tBase & aV) const
        {
             MMVII_INTERNAL_ASSERT_tiny(OkOverFlow(aV),"Value out of image");
        }
        void DupIn(cDataTypedIm<Type,Dim> &) const;

        bool   mDoAlloc;  ///< was data allocated by the image (must know 4 free)
        Type *   mRawDataLin; ///< raw data containing pixel values
};


/** Class for file image, basic now, will evolve but with (hopefully)
    a same/similar interface.
 
    What the user must know if the image exist  :
        * size, channel, type of value
        * read write an area (rectangle at least) from this file, method are
          in the template class for specialization to a given type

    Create a file with given spec (size ....)
*/


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

        std::string  mName;      ///< Name on the disk
        eTyNums      mType;      ///< Type of value for pixel
        int          mNbChannel; ///< Number of channels
};

/**  Class for 2D image in Ram of a given type :
        * there is no copy constructor, and only shared pointer can be allocated
        * algorithm will work on these images (pointers, ref)
        * all acces are cheked (in non optimized versions)
*/

template <class Type>  class cDataIm2D  : public cDataTypedIm<Type,2>
{
    public :
        friend class cBenchImage;
        friend class cBenchBaseImage;
        friend class cIm2D<Type>;

        typedef Type  tVal;
        typedef cDataTypedIm<Type,2>   tBI;
        typedef cRectObj<2>               tRO;
        typedef typename tBI::tBase  tBase;

        //========= fundamental access to values ============

           /// Get Value
        const Type & GetV(const cPt2di & aP)  const
        {
            tRO::AssertInside(aP);
            return  Value(aP);
        }

          /// Set Value
        void SetV(const cPt2di & aP,const tBase & aV)
        { 
            tRO::AssertInside(aP);
            tBI::AssertNonOverFlow(aV);
            Value(aP) = aV;
        }

          /// Trunc then set value
        void SetVTrunc(const cPt2di & aP,const tBase & aV)
        { 
            tRO::AssertInside(aP);
            Value(aP) = tNumTrait<Type>::Trunc(aV);
        }

          // Interface as generic image

        int     VI_GetV(const cPt2di& aP)  const override;
        double  VD_GetV(const cPt2di& aP)  const override;
        void VI_SetV(const  cPt2di & aP,const int & aV)    override ;
        void VD_SetV(const  cPt2di & aP,const double & aV) override ;

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



        ///  Read file image 1 to 1
        void Read(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  
        ///  Write file image 1 to 1
        void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
        virtual ~cDataIm2D();
    protected :
    private :
        cDataIm2D(const cDataIm2D<Type> &) = delete;  ///< No copy constructor for big obj, will add a dup()
        cDataIm2D(const cPt2di & aP0,const cPt2di & aP1,Type * DataLin=0); ///< Called by shared ptr (cIm2D)


        
        Type & Value(const cPt2di & aP)   {return mRawData2D[aP.y()][aP.x()];} ///< Data Access
        const Type & Value(const cPt2di & aP) const   {return mRawData2D[aP.y()][aP.x()];} /// Cont Data Access


        Type ** mRawData2D;  ///< Pointers on DataLin
};

/**  Class for allocating and storing 2D images
*/

template <class Type>  class cIm2D  
{
    public :
       typedef cDataIm2D<Type>  tDIM;
       cIm2D(const cPt2di & aP0,const cPt2di & aP1,Type * DataLin=0);
       cIm2D(const cPt2di & aSz,Type * DataLin=0);


       // tDIM & Im() {return *(mSPtr.get());}
       tDIM & DIm() {return *(mPIm);}
       const tDIM & DIm() const {return *(mPIm);}
      
       void Read(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
       void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
       static cIm2D<Type> FromFile(const std::string& aName);

       // void Read(const cDataFileIm2D &,cPt2di & aP0,cPt3dr Dyn /* RGB*/);  // 3 to 1
       // void Read(const cDataFileIm2D &,cPt2di & aP0,cIm2D<Type> aI2,cIm2D<Type> aI3);  // 3 to 3
       cIm2D<Type>  Dup() const;
    private :
       std::shared_ptr<tDIM> mSPtr;  ///< shared pointer to real image
       tDIM *                mPIm;
};

/**  Class for 1D image in Ram of a given type :
*/

template <class Type>  class cDataIm1D  : public cDataTypedIm<Type,1>
{
    public :
        friend class cIm1D<Type>;
        friend class cDenseVect<Type>;


        typedef Type  tVal;
        typedef cDataTypedIm<Type,1>   tBI;
        typedef cRectObj<1>               tRO;
        typedef typename tBI::tBase  tBase;

        //========= fundamental access to values ============

           /// Get Value
        const Type & GetV(const int & aP)  const
        {
            tRO::AssertInside(aP);
            return  Value(aP);
        }
        const Type & GetV(const cPt1di & aP)  const {return GetV(aP.x());}
        /// Used by matrix/vector interface 
        Type & GetV(const int & aP) { tRO::AssertInside(aP); return  Value(aP); }

          /// Set Value
        void SetV(const int & aP,const tBase & aV)
        { 
            tRO::AssertInside(aP);
            tBI::AssertNonOverFlow(aV);
            Value(aP) = aV;
        }
        void SetV(const  cPt1di & aP,const tBase & aV) {SetV(aP.x(),aV);}

          /// Trunc then set value
        void SetVTrunc(const int & aP,const tBase & aV)
        { 
            tRO::AssertInside(aP);
            Value(aP) = tNumTrait<Type>::Trunc(aV);
        }
        void SetVTrunc(const  cPt1di & aP,const tBase & aV) {SetVTrunc(aP.x(),aV);}
        const int    &  Sz() const  {return tRO::Sz().x();}
        const int    &  X0()  const {return tRO::P0().x();}
        const int    &  X1()  const {return tRO::P1().x();}

          // Interface as generic image

        int     VI_GetV(const cPt1di& aP)  const override;
        double  VD_GetV(const cPt1di& aP)  const override;
        void VI_SetV(const  cPt1di & aP,const int & aV)    override ;
        void VD_SetV(const  cPt1di & aP,const double & aV) override ;

        //========= Access to sizes, only alias/facilities ============
        virtual ~cDataIm1D();
    protected :
    private :
        Type * RawData1D() {return mRawData1D;}  ///< Used by matrix/vector interface

        cDataIm1D(const cDataIm1D<Type> &) = delete;  ///< No copy constructor for big obj, will add a dup()
        cDataIm1D(const cPt1di & aP0,const cPt1di & aP1,Type * DataLin=0); ///< Called by shared ptr (cIm2D)

        
        Type & Value(const int & aX)   {return mRawData1D[aX];} ///< Data Access
        const Type & Value(const int & aX) const   {return mRawData1D[aX];} /// Cont Data Access

        Type * mRawData1D;  ///< Offset vs DataLin
};

/**  Class for allocating and storing 1D images
*/

template <class Type>  class cIm1D  
{
    public :
       friend class cDenseVect<Type>;
       typedef cDataIm1D<Type>  tDIM;
       cIm1D(const int & aP0,const int & aP1,Type * DataLin=0);
       cIm1D(const int & aSz,Type * DataLin=0);

       // tDIM & Im() {return *(mSPtr.get());}
       tDIM & DIm() {return *(mPIm);}
       const tDIM & DIm() const {return *(mPIm);}
       cIm1D<Type>  Dup() const;
      
       // void Read(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
       // void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::Empty00);  // 1 to 1
       // static cIm2D<Type> FromFile(const std::string& aName);

       // void Read(const cDataFileIm2D &,cPt2di & aP0,cPt3dr Dyn /* RGB*/);  // 3 to 1
       // void Read(const cDataFileIm2D &,cPt2di & aP0,cIm2D<Type> aI2,cIm2D<Type> aI3);  // 3 to 3
    private :
       std::shared_ptr<tDIM> mSPtr;  ///< shared pointer to real image
       tDIM *                mPIm;
};


};

#endif  //  _MMVII_Images_H_
