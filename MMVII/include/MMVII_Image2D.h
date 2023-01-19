#ifndef  _MMVII_Images2D_H_
#define  _MMVII_Images2D_H_

#include "MMVII_Images.h"

namespace MMVII
{
/** \file MMVII_Image2D.h
    \brief Classes for storing specialization to 2 images, the most frequent case 
           in photogrammetry
*/

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
	bool IsEmpty() const;
	void AssertNotEmpty() const;
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

	void CropIn(const cPt2di & aP0,const tIm &);


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
       /// return grad + value in bilinaire mode : (Gx,Gy,Val)
       inline cPt3dr  GetGradAndVBL(const cPt2dr & aP)  const
       {
           tPB::AssertInsideBL(aP);
           return  ValueAndGradBL(aP);
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
          /// "Safe" set, test P and V
        void SetVTruncIfInside(const cPt2di & aP,const tBase & aV)
        { 
            if (tPB::Inside(aP))
               SetVTrunc(aP,aV);
        }

        /// Increment Value, check acces and overflow
        const Type & AddVal(const cPt2di & aP,const Type & aValAdd )  
        {
            tPB::AssertInside(aP);
            tBase aRes = Value(aP) + aValAdd;
            tBI::AssertValueOk(aRes);
            return (Value(aP) = aRes);
        }
        /// Modify min 
        void SetMin(const cPt2di & aP,const Type & aNewVal )  
        {
            tPB::AssertInside(aP);
            tBI::AssertValueOk(aNewVal);
	    UpdateMin(Value(aP),aNewVal);
        }
        /// Modify min 
        void SetMax(const cPt2di & aP,const Type & aNewVal )  
        {
            tPB::AssertInside(aP);
            tBI::AssertValueOk(aNewVal);
	    UpdateMax(Value(aP),aNewVal);
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
        void Read(const cDataFileIm2D &,tIm &aIG,tIm &aIB,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox);  
        ///  Write file image 1 channel to 1 channel
        void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox) const;  // 1 to 1
        void Write(const cDataFileIm2D &,const tIm &aIG,const tIm &aIB,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox) const;  // 1 to 1
        virtual ~cDataIm2D();  ///< will delete mRawData2D

        void ToFile(const std::string& aName) const; ///< Create a File having same size/type ...
        void ToFile(const std::string& aName,eTyNums) const; ///< Create a File of given type, having same size ...
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

        /** Bilinear interpolation */
        double  ValueBL(const cPt2dr & aP)  const
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

        /** Bilinear interpolation + Grad of bilinear interpol */
        cPt3dr  ValueAndGradBL(const cPt2dr & aP)  const
        {
            int aX0 = round_down(aP.x());  ///<  "Left" limit of  pixel
            int aY0 = round_down(aP.y());  ///<  "Up" limit of pixel

            double aWeigthX1 = aP.x() - aX0;
            double aWeightX0 = 1-aWeigthX1;
            double aWeightY1 = aP.y() - aY0;
            double aWeightY0 = 1-aWeightY1;

            const Type  * aL0 = mRawData2D[aY0  ] + aX0;
            const Type  * aL1 = mRawData2D[aY0+1] + aX0;

            return   cPt3dr
                     (
                            aWeightY0 * (aL0[1]-aL0[0])  // Gx on line 0
                          + aWeightY1 * (aL1[1]-aL1[0]), // Gx on line 1

                            aWeightX0 * (aL1[0]-aL0[0])  // Gy on col 0
                          + aWeigthX1 * (aL1[1]-aL0[1]), // Gy on col 1

                           aWeightY0 *(aWeightX0*aL0[0] + aWeigthX1*aL0[1])
                       +   aWeightY1 *(aWeightX0*aL1[0] + aWeigthX1*aL1[1])  
                     );
        } 

        /** Bilinear interpolation */
        void  AddValueBL(const cPt2dr & aP,const double & aVal)
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

       /// Create a smart pointer on an existing allocated image 
       //  cIm2D(tDIM&); =>  DONT WORK, have to understand why !!!

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


/**  Mother Class for application that parse an image in small blocs

*/
template<class TypeEl> class  cAppliParseBoxIm
{
    public :
        typedef cIm2D<TypeEl>      tIm;
        typedef cDataIm2D<TypeEl>  tDataIm;

    protected :
	template <class Type2>   cIm2D<Type2> TmpIm() const {return cIm2D<Type2>(CurBoxIn().Sz());}
	template <class Type2>   cIm2D<Type2> APBI_ReadIm(const std::string & aName)
	{
              cIm2D<Type2> aRes = TmpIm<Type2>();
	      aRes.Read(cDataFileIm2D::Create(aName,mIsGray),CurBoxIn().P0());
	      return aRes;
        }


	template <class Type2>   void  APBI_WriteIm(const std::string & aName,cIm2D<Type2> anIm,eTyNums aTyN)
	{
             cDataFileIm2D  aDF = cDataFileIm2D::Create(aName,aTyN,mDFI2d.Sz(),1);
	     anIm.Write(aDF,CurP0(),1.0,CurBoxOutLoc());
	}
	template <class Type2>   void  APBI_WriteIm(const std::string & aName,cIm2D<Type2> anIm)
        {
               APBI_WriteIm(aName,anIm,tElemNumTrait<Type2>::TyNum());
        }

        cAppliParseBoxIm(cMMVII_Appli & anAppli,bool IsGray,const cPt2di & aSzTiles,const cPt2di & aSzOverlap,bool ParalTiles) ;
        ~cAppliParseBoxIm();

	void  APBI_ExecAll(); ///< Execute Action on all Box of file  OR  only on Test Box if exist

        cCollecSpecArg2007 & APBI_ArgObl(cCollecSpecArg2007 & anArgObl) ; ///< For sharing mandatory args
        cCollecSpecArg2007 & APBI_ArgOpt(cCollecSpecArg2007 & anArgOpt); ///< For sharing optionnal args


        tDataIm &  APBI_DIm();  ///< Accessor to loaded image
        const tDataIm &  APBI_DIm() const;  ///< Accessor to loaded image
        tIm &       APBI_Im();  ///< Accessor to loaded image
        const tIm & APBI_Im() const;  ///< Accessor to loaded image
        bool      APBI_TestMode() const; ///< Ar we in test mode
        const std::string & APBI_NameIm() const;     ///< Name of image to parse

        cBox2di       CurBoxIn()  const;   
        cPt2di        CurSzIn()  const;   
        cPt2di        CurP0()  const;   
        cBox2di       CurBoxOut() const; 
        cBox2di       CurBoxInLoc() const; 
        cBox2di       CurBoxOutLoc() const; 
        const cDataFileIm2D &  DFI2d() const;   ///< accessor

    // private :
        cAppliParseBoxIm(const cAppliParseBoxIm &) = delete;
	bool InsideParsing() const;
	void AssertInParsing() const;
	void AssertNotInParsing() const;
        tDataIm & LoadI(const cBox2di & aBox); ///< Load file for the Box, return loaded image

        bool InsideParalRecall() const ; /// Indicate if we are in a recall of a parallal excecution
        bool TopCallParallTile() const ; /// Indicate if we are at the "top level" of a call in parallel

          // mandatory params
        std::string   mNameIm;     ///< Name of image to parse
          // Optional params
        cBox2di        mBoxTest;    ///< Box for quick testing, in case we dont parse all image
	cPt2di         mSzTiles;    ///< Size of tiles to parse global file
	cPt2di         mSzOverlap;  ///< Size of overlap between each tile
        bool           mParalTiles;   ///< Loaded image
        cPt2di         mIndBoxRecal;  ///< Index for box when recalling in paral

        cMMVII_Appli & mAppli;   ///< Ineriting appli ("daughter")
        bool           mIsGray;  ///< Is it a gray file
        cParseBoxInOut<2> *mParseBox;  ///<Current structure used to parse the  box
	cPt2di         mCurPixIndex; ///< Index of parsing box
        cDataFileIm2D  mDFI2d;   ///< Data for file image to parse
        tIm            mIm;      ///< Loaded image

};

///  Create a masq image if file exist, else create a masq with 1
cIm2D<tU_INT1>  ReadMasqWithDef(const cBox2di& aBox,const std::string &);

/// Generate an image of the string, using basic font, implemanted with a call to mmv1
cIm2D<tU_INT1> ImageOfString_10x8(const std::string & ,int aSpace);
cIm2D<tU_INT1> ImageOfString_DCT(const std::string & ,int aSpace);

class cRGBImage
{
     public :
        typedef cIm2D<tU_INT1>   tIm1C;  // Type of image for 1 chanel

        cRGBImage(const cPt2di & aSz);
        cRGBImage(const cPt2di & aSz,const cPt3di & aCoul);
        void ToFile(const std::string & aName);

        static cRGBImage FromFile(const std::string& aName);  ///< Allocate and init from file
        static cRGBImage FromFile(const std::string& aName,const cBox2di & );  ///< Allocate and init from file

        void Read(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox);  ///< 1 to 1
        void Write(const cDataFileIm2D &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox) const;  // 1 to 1

        void Read(const std::string &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox);  ///< 1 to 1
        void Write(const std::string &,const cPt2di & aP0,double aDyn=1,const cRect2& =cRect2::TheEmptyBox) const;  // 1 to 1
	/*
       */

        /// set values iff param are OK,  RGB image are made for visu, not for intensive computation
        void SetRGBPix(const cPt2di & aPix,int aR,int aG,int aB);
        void SetRGBPix(const cPt2di & aPix,const cPt3di &);
        cPt3di GetRGBPix(const cPt2di & aPix) const;

        cPt3di GetRGBPixBL(const cPt2dr & aPix) const;  // Get value with BL interpol
	bool InsideBL(const cPt2dr & aPix) const;


        ///  Alpha =>  1 force colour  , 0 no effect
        void SetRGBPixWithAlpha(const cPt2di & aPix,const cPt3di &,const cPt3dr & aAlpha);
        ///  
        void SetRGBrectWithAlpha(const cPt2di & aPix,int aSzW,const cPt3di & aCoul,const double & aAlpha);

        void SetGrayPix(const cPt2di & aPix,int aGray);



        tIm1C  ImR(); ///< Accessor
        tIm1C  ImG(); ///< Accessor
        tIm1C  ImB(); ///< Accessor

        static const  cPt3di  Red;
        static const  cPt3di  Green;
        static const  cPt3di  Blue;
        static const  cPt3di  Yellow;
        static const  cPt3di  Magenta;
        static const  cPt3di  Cyan;
        static const  cPt3di  Orange;
		static const  cPt3di  White;

	/// return a lut adapted to visalise label in one chanel (blue), an maximize constrat in 2 other
	static  std::vector<cPt3di>  LutVisuLabRand(int aNbLab);

     private :
        tIm1C  mImR;
        tIm1C  mImG;
        tIm1C  mImB;
};

template <class Type> void SetGrayPix(cRGBImage& aRGBIm,const cPt2di & aPix,const cDataIm2D<Type> & aGrayIm,const double & aMul=1.0);
template <class Type> void SetGrayPix(cRGBImage& aRGBIm,const cDataIm2D<Type> & aGrayIm,const double & aMul=1.0);
template <class Type> cRGBImage  RGBImFromGray(const cDataIm2D<Type> & aGrayIm,const double & aMul=1.0);





};

#endif  //  _MMVII_Images2D_H_
