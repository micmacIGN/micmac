#ifndef  _MMVII_Images_H_
#define  _MMVII_Images_H_
namespace MMVII
{


/** \file MMVII_Images.h
    \brief Classes for storing images in RAM, possibly N dimention
*/


/**  Class allow to iterate the pixel of an image (in fact a rectangular object) using the
same syntax than stl iterator => for (auto aP : anIma) ....
*/
template <const int Dim>  class cPixBoxIterator
{
     public :
        typedef cPtxd<int,Dim>        tPt;
        typedef cPixBoxIterator<Dim>  tIter;
        typedef cPixBox<Dim>          tPB;
        friend class cPixBox<Dim>;

        bool operator == (const tIter& aIt2) const {return  mPCur==aIt2.mPCur;}  ///< Equal iff current point are =
        bool operator != (const tIter& aIt2) const {return  mPCur!=aIt2.mPCur;}  ///< !Equal iif not equal ...
        tPt & operator * () {return mPCur;}         ///< classic operator dereference
        tPt & operator * () const {return mPCur;}   ///< classic operator dereference
        tPt * operator ->() {return &mPCur;}        ///< classic operator dereference
        tPt * operator ->() const {return &mPCur;}  ///< classic operator dereference

        /// standard prefix incrementation
        tIter &  operator ++(); 
        /// Just a "facility" to allow post-fix
        tIter &  operator ++(int) {return ++(*this);} 
     protected :
        
        cPixBoxIterator(tPB & aRO,const  tPt & aP0) : mRO (&aRO),mPCur (aP0) {}

        tPB * mRO;  ///< The rectangular object
        tPt             mPCur; ///< The current point 
};

/**  Class representing N-dim rectangle in pixel,
     Many basic services to test and visit the rectangle.
     Image (RAM), image (file), windows ... will inherit/contain

     Basically defined by integer point limits P0 and P1, and contain all the pixel Pix
     such that :
           P0[aK] <= Pix [aK] < P1[aK]  for K in [0,Dim[
*/

template <const int Dim>  class cPixBox : public cTplBox<int,Dim>
{
    public : 
        typedef int                   tScalPt;
        typedef cPtxd<tScalPt,Dim>        tPt;
        typedef cTplBox<tScalPt,Dim>      tBox;
        typedef cPixBoxIterator<Dim> iterator; ///< For auto ...
        static const cPixBox<Dim>    TheEmptyBox;
        //  --- Iterator ----------------
        iterator &  begin() {return mBegin;}   ///< For auto
        iterator &  end()   {return mEnd;}   ///< For auto
        const iterator &  begin() const {return mBegin;}   ///< For auto
        const iterator &  end()   const {return mEnd;}   ///< For auto
        tINT8     IndexeLinear(const tPt &) const; ///< Num of pixel when we iterate
        tPt     FromIndexeLinear(tINT8 ) const; ///< Num of pixel when we iterate
        /// Required by iterators  as they do not copy well becaus of ptr
        cPixBox(const cPixBox<Dim> &) ;
        cPixBox(const tPt & aP0,const tPt & aP1,bool AllowEmpty = false);
        static cPixBox<Dim>  BoxWindow(int aSz); ///<  Box of window around pix !! symetric   [-Sz,+aSz] 
        static cPixBox<Dim>  BoxWindow(const tPt &aC,int aSz); ///<  with center 
        /// It may be convenient as conversion, as tool may retun TplBox, and others may need to iterate on it
        cPixBox(const cTplBox<int,Dim> &);
        // Position of point relative to PixBox
          /// D is the num coordina, 0 on the border, <0 out, value is the signed margin
        int Interiority(const int  aCoord,int aD) const;  
        int Interiority(const tPt& aP    ,int aD) const;  ///< D is 
        int Interiority(const tPt& aP           ) const;  ///< Min of previous
        int WinInteriority(const tPt& aP,const tPt& aWin,int aD) const;  ///< D is 

        ///  return normalized coordinate assuming a circular topology where begin = end in all dimension
        tPt  CircNormProj(const tPt &) const;

        cBorderPixBox<Dim>  Border(int aSz) const;

        inline bool InsideBL(const cPtxd<double,Dim> & aP) const; ///< Inside for Bilin
        inline void AssertInsideBL(const cPtxd<double,Dim> & aP) const
        {
             MMVII_INTERNAL_ASSERT_tiny(InsideBL(aP),"Outside image in bilinear mode");
        }
        // Call SignalAtFrequence with linear index
        bool SignalAtFrequence(const tPt & anIndex,double aFreq) const;

    private :
        iterator  mBegin; ///< Beging iterator
        iterator  mEnd;   ///< Ending iterator
};

template <> inline  bool cPixBox<1>::InsideBL(const cPtxd<double,1> & aP) const
{
    return (aP.x() >= tBox::mP0.x()) &&  ((aP.x()+1) <  tBox::mP1.x());
}

template <> inline  bool cPixBox<2>::InsideBL(const cPtxd<double,2> & aP) const
{
    return   (aP.x() >= tBox::mP0.x()) &&  ((aP.x()+1) <  tBox::mP1.x())
          && (aP.y() >= tBox::mP0.y()) &&  ((aP.y()+1) <  tBox::mP1.y())
    ;
}
template <> inline  bool cPixBox<3>::InsideBL(const cPtxd<double,3> & aP) const
{
    return   (aP.x() >= tBox::mP0.x()) &&  ((aP.x()+1) <  tBox::mP1.x())
          && (aP.y() >= tBox::mP0.y()) &&  ((aP.y()+1) <  tBox::mP1.y())
          && (aP.z() >= tBox::mP0.z()) &&  ((aP.z()+1) <  tBox::mP1.z())
    ;
}

#ifndef FORSWIG
extern template const cPixBox<2>     cPixBox<2>::TheEmptyBox;  // Pb Clang, requires explicit declaration
#endif

typedef  cPixBox<1> cRect1;
typedef  cPixBox<2> cRect2;
typedef  cPixBox<3> cRect3;


/// Iterator on the border of rectangle
/** Iterator on the border of rectangle, many image processing require some specific
precaution to be taken on border (sides effect);

    Method : first do a "standard" iteration inside the rectangle, then if X=left border

 */
template <const int Dim>  class cBorderPixBoxIterator : public cPixBoxIterator<Dim>
{
   public :
       typedef cPtxd<int,Dim>        tPt;
       typedef cBorderPixBoxIterator<Dim>  tIter;
       typedef cPixBoxIterator<Dim>  tPBI;
       typedef cBorderPixBox<Dim>    tBPB;

       cBorderPixBoxIterator(tBPB & ,const  tPt & aP0);
        /// standard prefix incrementation
       tIter &  operator ++();
        /// Just a "facility" to allow post-fix
       tIter &  operator ++(int);

   private :
       tBPB *  mBPB;  ///< Pointer to the border for handling "special" incrementation
};

///  Class allowing to iterate on the border itsef

/**  To iterate on the border itself 

     Stantard inteface to iteratible object
     Containd the full rectangle itself + the interior rectangle
 */


template <const int Dim>  class cBorderPixBox
{
   public :
       typedef cBorderPixBoxIterator<Dim>  iterator;
       typedef cPixBox<Dim>          tPB;
       typedef cPtxd<int,Dim>        tPt;

       cBorderPixBox(const tPB & aRO,const tPt & aSz) ;
       cBorderPixBox(const tPB & aRO,int aSz) ;
       cBorderPixBox(const cBorderPixBox<Dim> &);

       iterator &  begin() {return mBegin;}   ///< For auto
       iterator &  end()   {return mEnd;}   ///< For auto
       iterator   begin() const  {return mBegin;}   ///< For auto
       iterator   end()   const  {return mEnd;}   ///< For auto
       tPB &    PB();
       void IncrPt(tPt &); ///< Make the incrementation specific to border
   private :
       tPB              mPB; ///< The Pix Box arround the border
       tPt              mSz;  ///< Sz of border
       cTplBox<int,Dim> mBoxInt;  ///< Interior box
       int              mX0;      ///< first X of interior box
       int              mX1;      ///< last X of interior box
       iterator         mBegin;   ///< memorization of iterator , begin
       iterator         mEnd;     ///< memorization of iterator , end
};



///  This class will be used to parse big image files and reassembling the results

/** This class will be usefull to parse big box (image files typically) and reassembling the results

    At it creation , create a "small" box of index,  then user can :
       * iterate on these index via  BoxIndex
       * for a given index, recuparate the  output for writing , ouput box are a partition of global 
         box
       * for a given index recurate input box, they add a margin to output box, and are always
         included in global box
*/


template <const int Dim>  class cParseBoxInOut
{
     public :
        // -------- Typdef section ----------
        typedef cPixBox<Dim>        tBox;
        typedef cPtxd<int,Dim>      tPt;
        typedef cParseBoxInOut<Dim> tThis;

        // -------- creators ----------
        static tThis  CreateFromSize(const tBox &, const tPt & aSz); ///< Give the size of boxes
        static tThis  CreateFromSizeCste(const tBox&, int aSz);  ///< Give a constant size
        static tThis  CreateFromSzMem(const tBox&, double AvalaibleMem); ///< Allocate  approximate mem by tiles

        // -------- Manipulation ----------

        /** Box of indexes : "small" number of tiles , (0,0)=> top left box ..., to be used in for(auto..) */
        const tBox & BoxIndex() const;  
        tBox  BoxOut(const tPt & anIndex) const; ///< return OutBox from an index created by BoxIndex
        tBox  BoxIn(const tPt & anIndex,const tPt& anOverlap) const;  ///< Idem but add an overlap
        tBox  BoxIn(const tPt & anIndex,const int anOverlap) const;   ///< Add a constant overlap in all direction


     private :
        cParseBoxInOut(const tBox & aBoxGlob,const tBox & aBoxIndexe); ///< Create from given indexe box
        tPt Index2Glob(const tPt &) const; ///< Map mBoxIndex  to mBoxGlob

        tBox  mBoxGlob;  ///< Box with want to parse
        tBox  mBoxIndex; ///< Box of index of sub box
};




/* Iterator allowing to visit rectangles */

template <> inline cPixBoxIterator<1> &  cPixBoxIterator<1>::operator ++() 
{ 
   mPCur.x()++; 
   return *this;
}
template <> inline cPixBoxIterator<2> &  cPixBoxIterator<2>::operator ++() 
{
    // std::cout << "OPPPPPP " << mPCur << " " << mRO->P0() << mRO->P1() << "\n";
    mPCur.x()++; 
    if (mPCur.x() == mRO->P1().x())
    {
        mPCur.x() = mRO->P0().x();
        mPCur.y()++;
    }

    return *this;
}
template <> inline cPixBoxIterator<3> &  cPixBoxIterator<3>::operator ++() 
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



///  Abstract class allowing to manipulate images independanlty of their type

/**  This class define an interface that allow to manipulate any image

     This generic class heritate from PixBox and  define several pure
     virtual methods to access to pixel read/write 
*/


template <const int Dim> class cDataGenUnTypedIm : public cPixBox<Dim>,
                                                   public cMemCheck
{
      public :
        typedef cPixBox<Dim>            tPB;
        const   tPB  & RO() {return *this;}

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


///  Classes for   ram-image containg a given type of pixel
/**  Classes for   ram-image containg a given type of pixel

      It contains a linear buffer allow to store the pixel value.
      Some basic vector like manipulation can so be done indepentanly of the dimension
      (like add, subb etc ...)
*/

template <class Type,const int Dim> class cDataTypedIm : public cDataGenUnTypedIm<Dim>
{
    public :
        // tINT8     IndexeLinear(const tPt &) const; ///< Num of pixel when we iterate
        //    tPB::AssertInside(aP);
        

     // ======================================

        typedef Type  tVal;
        typedef tNumTrait<Type> tTraits;
        typedef typename tTraits::tBase  tBase;
        typedef cPixBox<Dim>            tPB;

        const tINT8 & NbElem() const {return tPB::NbElem();} ///< Number total of pixel
        const cPtxd<int,Dim> & P0() const {return tPB::P0();}  ///< facility
        const cPtxd<int,Dim> & P1() const {return tPB::P1();}  ///< facility
        const cPtxd<int,Dim> & Sz() const {return tPB::Sz();}  ///< facility

        Type * RawDataLin() {return  mRawDataLin;}  ///< linear raw data
        const Type * RawDataLin() const {return  mRawDataLin;}  ///< linear raw data

        Type & GetRDL(int aK)             {return  mRawDataLin[aK];} ///<  Kth val
        const Type & GetRDL(int aK) const {return  mRawDataLin[aK];} ///<  Kth val

        void InitRandom();    ///< uniform, float in [0,1], integer in [Min,Max] of Type
        void InitRandom(const Type &aV0,const Type & aV1);  ///< uniform float in [V0, V2[
        void InitRandomCenter();    ///< uniform, float in [-1,1], integer in [Min,Max] of Type
        void InitCste(const Type & aV); ///< Constant value
        void InitBorder(const Type & aV); ///< Set Value on Border 1 pixel
        void InitInteriorAndBorder(const Type & aVInterior,const Type & aVBorder); ///< Init Inter + Init Border

        void InitId();                  ///< Identity, only avalaible for 2D-squares images
        void InitNull();                ///< Null, faster than InitCste(0)
        void InitDirac(const cPtxd<int,Dim> & aP,const Type &  aVal=1);  ///<  Create aDirac Image, 0 execpt 1 in P, used in test
        void InitDirac(const Type &  aVal=1);  ///<  Create aDirac with val in center
        void Init(eModeInitImage);      ///< swicth to previous specialized version

        //========= Test access ============

        inline bool Inside(const cPtxd<int,Dim> & aP) const {return tPB::Inside(aP);} ///< Is Point inside def
        inline bool ValueOk(const tBase & aV) const {return tTraits::ValueOk(aV);}  ///< Is value Ok (overflow, nan, infty..)

        cDataTypedIm (const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,
                      Type * DataLin=nullptr,eModeInitImage=eModeInitImage::eMIA_NoInit);  ///< Only cstr
        virtual ~cDataTypedIm(); ///<   Big obj, do it virtual
        // All distance-norm are  normalized/averaged , so that const image has a norm equal to the constante
        double L1Dist(const cDataTypedIm<Type,Dim> & aV) const;  ///< Distance som abs
        double L2Dist(const cDataTypedIm<Type,Dim> & aV) const;  ///< Dist som square
        double LInfDist(const cDataTypedIm<Type,Dim> & aV) const; ///< Dist max
        double L1Norm() const;   ///< Norm som abs
        double L2Norm() const;   ///< Norm square
        double LInfNorm() const; ///< Nomr max

        Type     MinVal() const;
        Type     MaxVal() const;
        tREAL16  SomVal() const;
        tREAL16  MoyVal() const;

        void DupIn(cDataTypedIm<Type,Dim> &) const;  ///< Duplicate raw data
        void DupInVect(std::vector<Type> &) const;  ///< Duplicate raw data in a vect

        // Defaults values quitt slow but may be usefull
        int VI_GetV(const cPtxd<int,Dim> & aP)  const override;
                /// Pixel -> float Value
        double VD_GetV(const cPtxd<int,Dim> & aP)  const override;
           // Set Value, integer coordinates
                /// Set Pixel Integrer Value
        void VI_SetV(const  cPtxd<int,Dim> & aP,const int & aV) override;
                /// Set Pixel Float Value
        void VD_SetV(const  cPtxd<int,Dim> & aP,const double & aV)override;
        void Resize(const cPtxd<int,Dim> & aP0,const cPtxd<int,Dim> & aP1,eModeInitImage=eModeInitImage::eMIA_NoInit);
    protected :

        ///< Test 4 writing
        void AssertValueOk(const tBase & aV) const
        {
             MMVII_INTERNAL_ASSERT_tiny(ValueOk(aV),"Invalid Value for image");
        }

        bool   mDoAlloc;  ///< was data allocated by the image (must know 4 free)
        Type *   mRawDataLin; ///< raw data containing pixel values
        int      mNbElemMax;  ///< maxim number reached untill now, for resize
};


///   Class for file image 2D

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
        const cPt2di & Sz() const ;  ///< From cRect2
        const int  & NbChannel ()  const ;  ///< std accessor
        const eTyNums &   Type ()  const ;  ///< std accessor
        const std::string &  Name() const;  ///< std accessor
        /// Create a descriptor on existing file
        static cDataFileIm2D Create(const std::string & aName,bool ForceGray);
        /// Create the file before returning the descriptor
        static cDataFileIm2D Create(const std::string & aName,eTyNums,const cPt2di & aSz,int aNbChan=1);

        static cDataFileIm2D Empty();

        virtual ~cDataFileIm2D();
        
     private :
        cDataFileIm2D(const std::string &,eTyNums,const cPt2di & aSz,int aNbChannel) ;

        cMemCheck    mMemCheck;  ///< Inheritage may be multiple, member will have the same effect
        std::string  mName;      ///< Name on the disk
        eTyNums      mType;      ///< Type of value for pixel
        int          mNbChannel; ///< Number of channels
};

/// Size differnce of associated file images
cPt2di DifInSz(const std::string & aN1,const std::string & aN2);
/// Total diff of values of associated file images
double DifAbsInVal(const std::string & aN1,const std::string & aN2,double aDef=-1);


///  Class for 2D image in Ram of a given type :
/**  Class for 2D image in Ram of a given type :
        * there is no copy constructor, and only shared pointer can be allocated
        * algorithm will work on these images (pointers, ref)
        * all acces are cheked (in non optimized versions)
     Class that store an image will store cIm2D
      
     This "pattern" will be used in many case : a class, uncopiable, that is used
     for storing, and create shared pointer, a class for storing
*/

template <class Type>  class cDataIm2D  : public cDataTypedIm<Type,2>
{
    public :
        friend class cIm2D<Type>;


        typedef Type  tVal;
        typedef tVal* tPVal;
        typedef cDataTypedIm<Type,2>   tBI;
        typedef cPixBox<2>               tPB;
        typedef typename tBI::tBase  tBase;
        typedef cDataIm2D<Type>      tIm;

        //========= fundamental access to values ============

        void  AddVBL(const cPt2dr & aP,const double & aVal)  
        {
           tPB::AssertInsideBL(aP);
           AddValueBL(aP,aVal);
        }
       /// Bilinear valie
       inline double GetVBL(const cPt2dr & aP) const 
       {
           tPB::AssertInsideBL(aP);
           return  ValueBL(aP);
       }
       inline double DefGetVBL(const cPt2dr & aP,double aDef) const
       {
            if (tPB::InsideBL(aP))
               return ValueBL(aP);
            return aDef;
       }

           /// Get Value, check access in non release mode
        const Type & GetV(const cPt2di & aP)  const
        {
            tPB::AssertInside(aP);
            return  Value(aP);
        }
        /* No  Type & GetV() or  Type & operator()   ... as it does not allow
           to check values
        */

        /// Get Value with def when out side
        tBase  DefGetV(const cPt2di & aP,const tBase & aDef )  const
        {
            if (tPB::Inside(aP))
               return Value(aP);
            return aDef;
        }

          /// Set Value, check point and value in  non release mode
        void SetV(const cPt2di & aP,const tBase & aV)
        { 
            tPB::AssertInside(aP);
            tBI::AssertValueOk(aV);
            Value(aP) = aV;
        }

          /// Trunc then set value, no check on value
        void SetVTrunc(const cPt2di & aP,const tBase & aV)
        { 
            tPB::AssertInside(aP);
            Value(aP) = tNumTrait<Type>::Trunc(aV);
        }

        /// Increment Value, check acces and overflow
        const Type & AddVal(const cPt2di & aP,const Type & aValAdd )  
        {
            tPB::AssertInside(aP);
            tBase aRes = Value(aP) + aValAdd;
            tBI::AssertValueOk(aRes);
            return (Value(aP) = aRes);
        }

          // Interface as generic image

        int     VI_GetV(const cPt2di& aP)  const override; ///< call GetV
        double  VD_GetV(const cPt2di& aP)  const override; ///< call GetV
        void VI_SetV(const  cPt2di & aP,const int & aV)    override ; ///< call SetV
        void VD_SetV(const  cPt2di & aP,const double & aV) override ; ///< call SetV

        // ==  raw pointer on origin of line
        const Type * GetLine(int aY)  const;
        Type * GetLine(int aY) ;
        //========= Access to sizes, only alias/facilities ============

        const cPt2di &  Sz()  const {return tPB::Sz();}  ///< Std Accessor
        const int    &  SzX() const {return Sz().x();}   ///< Std Accessor
        const int    &  SzY() const {return Sz().y();}   ///< Std Accessor

        const cPt2di &  P0()  const {return tPB::P0();}  ///< Std Accessor
        const int    &  X0()  const {return P0().x();}   ///< Std Accessor
        const int    &  Y0()  const {return P0().y();}   ///< Std Accessor

        const cPt2di &  P1()  const {return tPB::P1();}  ///< Std Accessor
        const int    &  X1()  const {return P1().x();}   ///< Std Accessor
        const int    &  Y1()  const {return P1().y();}   ///< Std Accessor

        const tINT8 & NbPix() const {return tPB::NbElem();} ///< Number total of pixel

        void Resize(const cPt2di& aP0,const cPt2di & aP1,eModeInitImage=eModeInitImage::eMIA_NoInit);
        void Resize(const cPt2di& aSz,eModeInitImage=eModeInitImage::eMIA_NoInit);


        ///  Read file image 1 channel to 1 channel
        void Read(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox);  
        ///  Write file image 1 channel to 1 channel
        void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox) const;  // 1 to 1
        void Write(const cDataFileIm2D &,const tIm &aIG,const tIm &aIB,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox) const;  // 1 to 1
        virtual ~cDataIm2D();  ///< will delete mRawData2D

        void ToFile(const std::string& aName) const; ///< Create a File having same size/type ...
        void ClipToFile(const std::string& aName,const cRect2&) const; ///< Create a Clip File of Box
        void ToFile(const std::string& aName,const tIm &aIG,const tIm &aIB) const; ///< Create a File having same size/type ...
        
        /// Raw image, lost all waranty is you use it...
        tVal ** ExtractRawData2D() {return mRawData2D;}
        const tPVal * ExtractRawData2D() const {return mRawData2D;}
    protected :
    private :
        void PostInit();
        cDataIm2D(const cDataIm2D<Type> &) = delete;  ///< No copy constructor for big obj, will add a dup()
        cDataIm2D(const cPt2di & aP0,const cPt2di & aP1,
                 Type * DataLin=nullptr,eModeInitImage=eModeInitImage::eMIA_NoInit); ///< Called by shared ptr (cIm2D)


        
        Type & Value(const cPt2di & aP)   {return mRawData2D[aP.y()][aP.x()];} ///< Data Access
        const Type & Value(const cPt2di & aP) const   {return mRawData2D[aP.y()][aP.x()];} /// Const Data Access

        double  ValueBL(const cPt2dr & aP)  const ///< Bilinear interpolation
        {
            int aX0 = round_down(aP.x());  ///<  "Left" limit of  pixel
            int aY0 = round_down(aP.y());  ///<  "Up" limit of pixel

            double aWeigthX1 = aP.x() - aX0;
            double aWeightX0 = 1-aWeigthX1;
            double aWeightY1 = aP.y() - aY0;

            const Type  * aL0 = mRawData2D[aY0  ] + aX0;
            const Type  * aL1 = mRawData2D[aY0+1] + aX0;

            return  (1-aWeightY1) * (aWeightX0*aL0[0]  + aWeigthX1*aL0[1])
                  +     aWeightY1 * (aWeightX0*aL1[0]  + aWeigthX1*aL1[1])  ;
        } 

        void  AddValueBL(const cPt2dr & aP,const double & aVal)  ///< Bilinear interpolation
        {
            int aX0 = round_down(aP.x());  ///<  "Left" limit of  pixel
            int aY0 = round_down(aP.y());  ///<  "Up" limit of pixel

            double aWeigthX1 = aP.x() - aX0;
            double aWeightX0 = 1-aWeigthX1;
            double aWeightY1 = aP.y() - aY0;
            double aWeightY0 = 1 - aWeightY1;

            Type  * aL0 = mRawData2D[aY0  ] + aX0;
            Type  * aL1 = mRawData2D[aY0+1] + aX0;

            aL0[0] += aWeightY0  * aWeightX0 *  aVal;
            aL0[1] += aWeightY0  * aWeigthX1 *  aVal;
            aL1[0] += aWeightY1  * aWeightX0 *  aVal;
            aL1[1] += aWeightY1  * aWeigthX1 *  aVal;

        } 

        void AssertYInside(int Y) const
        {
             MMVII_INTERNAL_ASSERT_tiny((Y>=Y0())&&(Y<Y1()),"Point out of image");
        }

        int     mSzYMax;     ///< For resize
        tPVal * mRawData2D;  ///< Pointers on DataLin
};



///  Class for memorzing 2D images
/**  Class for allocating and storing 2D images
     This is no more than a shared ptr on a cDataIm2D
*/

template <class Type>  class cIm2D  
{
    public :
       typedef cDataIm2D<Type>  tDIM;

       /// Alow to allocate image with origin not in (0,0)
       cIm2D(const cPt2di & aP0,const cPt2di & aP1, Type * DataLin=nullptr,eModeInitImage=eModeInitImage::eMIA_NoInit); 
       /// Image with origin on (0,0)
       cIm2D(const cPt2di & aSz,Type * DataLin=0,eModeInitImage=eModeInitImage::eMIA_NoInit);

       /// Create an image and initialize it with the file
       cIm2D(const cBox2di & aSz,const cDataFileIm2D & aDataF);



       tDIM & DIm() {return *(mPIm);}  ///< return raw pointer
       const tDIM & DIm() const {return *(mPIm);} ///< const version 4 raw pointer
      
       void Read(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox);  ///< 1 to 1
       void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox) const;  // 1 to 1

       static cIm2D<Type> FromFile(const std::string& aName);  ///< Allocate and init from file
       static cIm2D<Type> FromFile(const std::string& aName,const cBox2di & );  ///< Allocate and init from file

       // void Read(const cDataFileIm2D &,cPt2di & aP0,cPt3dr Dyn /* RGB*/);  // 3 to 1
       // void Read(const cDataFileIm2D &,cPt2di & aP0,cIm2D<Type> aI2,cIm2D<Type> aI3);  // 3 to 3

       cIm2D<Type>  Dup() const;  ///< create a duplicata
       void  DecimateInThis(int aFact,const cIm2D<Type> & I) ;  ///< Decimate I in this ,just take on pix out of N
       cIm2D<Type>  Decimate(int aFact) const;  ///< create decimated image, just take on pix out of N
       cPt2di       SzDecimate(int aFact) const;  ///< Sometime it may be usefull to know the size before computing it
       /**  Apply gaussian filter, use a temporary float */
       cIm2D<Type>  GaussFilter(double , int aNbIterExp=3) const;  
       /**  Apply gaussian filter before dezoom to have good ressampling, may be a bit slow
            Dilate => to change defautl gaussian kernel */
       cIm2D<Type>  GaussDeZoom(int aFact, int aNbIterExp=3,double Dilate=1.0) const;  

       /** Transposition, needed it once, maybe usefull later */
       cIm2D<Type> Transpose() const;

    private :
       std::shared_ptr<tDIM> mSPtr;  ///< shared pointer to real image , allow automatic deallocation
       tDIM *                mPIm;   ///< raw pointer on mSPtr, a bit faster to store it ?
};

/// Generate an image of the string, using basic font, implemanted with a call to mmv1
cIm2D<tU_INT1> ImageOfString(const std::string & ,int aSpace);


///  Class for 1D image in Ram of a given type
/**  Class for 1D image in Ram of a given type :
     Same pattern than 2D image
*/

template <class Type>  class cDataIm1D  : public cDataTypedIm<Type,1>
{
    public :
        friend class cIm1D<Type>;
        friend class cDenseVect<Type>;


        typedef Type  tVal;
        typedef cDataTypedIm<Type,1>   tBI;
        typedef cPixBox<1>             tPB;
        typedef typename tBI::tBase  tBase;

        //========= fundamental access to values ============

           /// Get Value
        const Type & GetV(const int & aP)  const
        {
            tPB::AssertInside(aP);
            return  Value(aP);
        }
        inline double GetVBL(const double & aP) const 
        {
           tPB::AssertInsideBL(cPt1dr(aP));
           return  ValueBL(aP);
        }
        const Type & GetV(const cPt1di & aP)  const {return GetV(aP.x());}
        /// Used by matrix/vector interface 
        Type & GetV(const int & aP) { tPB::AssertInside(aP); return  Value(aP); }


        const Type & CircGetV(const int & aP)  const {return Value(tPB::CircNormProj(cPt1di(aP)).x());}
          /// Set Value
        void SetV(const int & aP,const tBase & aV)
        { 
            tPB::AssertInside(aP);
            tBI::AssertValueOk(aV);
            Value(aP) = aV;
        }

        void AddV(const int & aP,const tBase & aV2Add)
        {
            tPB::AssertInside(aP);
            tVal & aVP =   Value(aP); 
            tBI::AssertValueOk(aVP+aV2Add);
            aVP += aV2Add;
        }
        


        void SetV(const  cPt1di & aP,const tBase & aV) {SetV(aP.x(),aV);}

          /// Trunc then set value
        void SetVTrunc(const int & aP,const tBase & aV)
        { 
            tPB::AssertInside(aP);
            Value(aP) = tNumTrait<Type>::Trunc(aV);
        }
        void SetVTrunc(const  cPt1di & aP,const tBase & aV) {SetVTrunc(aP.x(),aV);}
        const int    &  Sz() const  {return tPB::Sz().x();}
        const int    &  X0()  const {return tPB::P0().x();}
        const int    &  X1()  const {return tPB::P1().x();}

          // Interface as generic image

        int     VI_GetV(const cPt1di& aP)  const override;
        double  VD_GetV(const cPt1di& aP)  const override;
        void VI_SetV(const  cPt1di & aP,const int & aV)    override ;
        void VD_SetV(const  cPt1di & aP,const double & aV) override ;

        void Resize(const cPt1di& aP0,const cPt1di & aP1,eModeInitImage=eModeInitImage::eMIA_NoInit);
        void Resize(int aSz,eModeInitImage=eModeInitImage::eMIA_NoInit);
        //========= Access to sizes, only alias/facilities ============
        virtual ~cDataIm1D();
        /// Raw image, lost all waranty is you use it...
        tVal * ExtractRawData1D() {return mRawData1D;}

        inline tBase  SomInterv(int aX0,int aX1) const;
        inline tREAL8  AvgInterv(int aX0,int aX1) const;

    protected :
    private :
        void PostInit();
        Type * RawData1D() {return mRawData1D;}  ///< Used by matrix/vector interface

        cDataIm1D(const cDataIm1D<Type> &) = delete;  ///< No copy constructor for big obj, will add a dup()
        cDataIm1D(const cPt1di & aP0,const cPt1di & aP1,
                      Type * DataLin=nullptr,eModeInitImage=eModeInitImage::eMIA_NoInit); ///< Called by shared ptr (cIm2D)

        
        Type & Value(const int & aX)   {return mRawData1D[aX];} ///< Data Access
        const Type & Value(const int & aX) const   {return mRawData1D[aX];} /// Cont Data Access

        double  ValueBL(const double & aX)  const ///< Bilinear interpolation
        {
            int aX0 = round_down(aX);  ///<  "Left" limit of  pixel
            double aWeigthX1 = aX - aX0;
            double aWeightX0 = 1-aWeigthX1;
            const Type  * aL = mRawData1D + aX0;

            return   (aWeightX0*aL[0]  + aWeigthX1*aL[1]);
        } 


        Type * mRawData1D;  ///< Offset vs DataLin
};

///  Class for allocating and storing 1D images
/**  Class for allocating and storing 1D images
    Same pattern than  cDataIm1D/cIm1D
*/

template <class Type>  class cIm1D  
{
    public :
       friend class cDenseVect<Type>;
       typedef cDataIm1D<Type>  tDIM;
       cIm1D(const int & aP0,const int & aP1,Type * DataLin=nullptr,eModeInitImage=eModeInitImage::eMIA_NoInit);
       cIm1D(const int & aSz, Type * DataLin=nullptr,eModeInitImage=eModeInitImage::eMIA_NoInit); 

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

/**  Class used to represent "full" 3D images containing *** Object
 */


template <class Type>  class cDataIm3D  : public cDataTypedIm<Type,3>
{
    public :
        friend class cIm3D<Type>;

        typedef Type   tVal;
        typedef tVal*  tPVal;
        typedef tPVal* tPPVal;
        typedef cDataTypedIm<Type,3>   tBI;
        typedef cPixBox<3>               tPB;
        typedef typename tBI::tBase  tBase;
        typedef cDataIm3D<Type>      tIm;
        const   cPt3di & Sz() const {return tBI::Sz();}

        const Type & GetV(const cPt3di & aP)  const
        {
            tPB::AssertInside(aP);
            return  Value(aP);
        }
        void SetV(const cPt3di & aP,const tBase & aV)
        {
            tPB::AssertInside(aP);
            tBI::AssertValueOk(aV);
            Value(aP) = aV;
        }
        // Not private because called by shared_ptr ...
        virtual ~cDataIm3D();

    private :
        cDataIm3D(const cDataIm3D &) = delete;
        Type & Value(const cPt3di & aP)               {return mRawData3D[aP.z()][aP.y()][aP.x()];} ///< Data Access
        const Type & Value(const cPt3di & aP) const   {return mRawData3D[aP.z()][aP.y()][aP.x()];} /// Const Data Access

        cDataIm3D(const cPt3di & aSz,Type * aRawDataLin,eModeInitImage aModeInit) ;
        tPPVal * mRawData3D;
};

/** Smart Pointer on 3d images */
template <class Type>  class cIm3D
{
    public :
        typedef cDataIm3D<Type>  tDIM;

        cIm3D(const cPt3di & aSz,Type * aRawDataLin,eModeInitImage aModeInit) ;
        cIm3D(const cPt3di & aSz);
        tDIM & DIm() {return *(mPIm);}  ///< return raw pointer
        const tDIM & DIm() const {return *(mPIm);} ///< const version 4 raw pointer
        // ~cIm3D();

    private :
        std::shared_ptr<tDIM> mSPtr;  ///< shared pointer to real image , allow automatic deallocation
        tDIM *                mPIm;   ///< raw pointer on mSPtr, a bit faster to store it ?
};



template <class TypeH,class TypeCumul>  class cHistoCumul
{
    public :
         cHistoCumul();  ///< For case where default constructor are required
         cHistoCumul(int aNbVal);
         void AddV(const int & aP,const TypeH & aV2Add);
         void MakeCumul();
         tREAL8  PropCumul(const int & aP) const; 
         tREAL8  PropCumul(const tREAL8 & aP) const; 
         const cDataIm1D<TypeH>&   H() const;
         void AddData(const cAuxAr2007 & anAux);
	 
          //  Different stats on the distribution of errors
          double  PercBads(double aThr) const;     // Classical % of  bads value over a threshold
          double  AvergBounded(double aThr,bool Apod=false) const; // Average of value bounded by a threshold  Min(T,X)
          double  ApodAverg(double aThr) const; //  Appodised bounded  T - T^2/(T+x)
          double  QuantilValue(double aThr) const; //  Value Over given quantille

	  int IndexeLowerProp(const double  aProp) const;

    private :
	 void AssertCumulDone() const;
        // void AddV(const int & aP,const tBase & aV2Add)
         int                 mNbVal;
         cIm1D<TypeH>        mH;
         cDataIm1D<TypeH>*   mDH;
         cIm1D<TypeCumul>      mHC;
         cDataIm1D<TypeCumul>* mDHC;
         bool                mHCOk;
         tREAL8              mPopTot;
    
};

class cTabulFonc1D : public cFctrRR
{
     public  :
       double F (double) const override;  ///< Virtual usable as cFctRR

       cTabulFonc1D(const cFctrRR & aFctr,double XMin,double XMax,int aNbStep);
       virtual ~ cTabulFonc1D() = default;

     private  :
       inline int    ToIntCoord(double aX) const;
       inline double ToRealCoord(int   aI) const;

       double  mXMin;
       double  mXMax;
       int     mNbStep;
       double  mStep;
       double  mValXMin;
       double  mValXMax;  
       cIm1D<double>      mIm;
       cDataIm1D<double>* mDIm;
};



};

#endif  //  _MMVII_Images_H_
