#include <bitset>
#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Sensor.h"

//  49 : 387 =  256 + 128 + 2 +1
//              00000110000011
//
//  50 : 389 = 256 + 128 + 4 +1
//              00000110000101

namespace MMVII
{

class cNormPix2Bit;
class cCircNP2B ;
class cStraightNP2B ;

/**  Generate a visualistion of target, made an external-non-friend  function
 * to test the usability.
 *
 */
void TestReloadAndShow_cFullSpecifTarget(const std::string & aDir,const std::string & aName,int aZoom)
{
    // -1- ----------- Create the object from file ------------------
    std::unique_ptr<cFullSpecifTarget> aFullSpec (cFullSpecifTarget::CreateFromFile(aName));

    // -2- ------------ generate a "standard" target image -----------
        // -2.1- generate an encoding with maximum of transition
    cOneEncoding aEnc(0,Str2BitFlag("010101010101010101010101010101"));
    aEnc.SetName("xx");

       // -2.2- generate a "standard" target image
    cFullSpecifTarget::tIm anIm = aFullSpec->OneImTarget(aEnc);
    // anIm.DIm().ToFile(aDir+"TestTarget_"+aFullSpec->Prefix()+".tif");

    // -3-  ------------------generate a high resolution to visualize
    cRGBImage aImZoom =   RGBImFromGray(anIm.DIm(),1.0,aZoom);

         // -3.1- visualize centers and corners
    aImZoom.DrawCircle(cRGBImage::Blue,aFullSpec->Center(),1.0);
    aImZoom.DrawCircle(cRGBImage::Red,aFullSpec->CornerlEl_BW(),3.0);
    aImZoom.DrawCircle(cRGBImage::Green,aFullSpec->CornerlEl_WB(),3.0);

         // -3.2- visualize the bits
    for (const auto & aC : aFullSpec->BitsCenters())
    {
        for (const auto & aR : {1.0,3.0,5.0})
            aImZoom.DrawCircle(cRGBImage::Cyan,aC,aR);
    }

         // -3.3- write the file
    aImZoom.ToFile(aDir+"TestZoom_"+aFullSpec->Prefix()+".tif");
}

void Bench_Target_Encoding()
{
    MMVII_DEV_WARNING("NO Bench_Target_Encoding");
    return;

    // -1-  Create the spec
    std::string aName =  cMMVII_Appli::InputDirTestMMVII()
	                 + "Targets" +  StringDirSeparator()  + "IGNIndoor_Nbb12_Freq2_Hamm3_Run2_3_FullSpecif.xml";

    std::unique_ptr<cFullSpecifTarget> aFullSpec (cFullSpecifTarget::CreateFromFile(aName));

    // -2-   check some elementary values
    MMVII_INTERNAL_ASSERT_bench(aFullSpec->NbBits()==12,"Bench_Target_Encoding");
    MMVII_INTERNAL_ASSERT_bench(aFullSpec->MinHammingD()==3,"Bench_Target_Encoding");
    MMVII_INTERNAL_ASSERT_bench(aFullSpec->EncodingFromName("toto")==nullptr,"Bench_Target_Encoding");
    MMVII_INTERNAL_ASSERT_bench(aFullSpec->EncodingFromCode((1<<12))==nullptr,"Bench_Target_Encoding"); // too big

    // -3-   check access to encoding from name/code 
    for (const auto & aNameT : std::vector<std::string>({"14","26","03"}))
    {
        const cOneEncoding * anEncod = aFullSpec->EncodingFromName(aNameT);
        MMVII_INTERNAL_ASSERT_bench(anEncod!=nullptr,"Bench_Target_Encoding");

        size_t aBitFlag = anEncod->Code();
        const cOneEncoding * anEncod2 = aFullSpec->EncodingFromCode(aBitFlag);
        MMVII_INTERNAL_ASSERT_bench(anEncod==anEncod2,"Bench_Target_Encoding");

        anEncod2 = aFullSpec->EncodingFromCode(aBitFlag+1);  // should ==0, due to hamming constraint
        MMVII_INTERNAL_ASSERT_bench(anEncod2==nullptr,"Bench_Target_Encoding");

        size_t anEquiCode  = N_LeftBitsCircPerm(aBitFlag,1<<12,6);
        anEncod2 = aFullSpec->EncodingFromCode(anEquiCode);  // should ==0, due to hamming constraint

        MMVII_INTERNAL_ASSERT_bench(anEncod==anEncod2,"Bench_Target_Encoding");
    }
}

/**************************************************/
/*                                                */
/*           cDecodeFromCoulBits                  */
/*                                                */
/**************************************************/

cDecodeFromCoulBits::cDecodeFromCoulBits(const cFullSpecifTarget * aSpec) :
     mSpec       (aSpec),
     mCode       (0),
     mBitsFixed  (0)
{
}

void cDecodeFromCoulBits::SetColBit(bool IsBlack,size_t aBit)
{
    MMVII_INTERNAL_ASSERT_tiny(aBit< mSpec->NbBits(),"Unvalide bit in cDecodeFromCoulBits::SetColBit");
    if (mSpec->BitIs1(!IsBlack))
       mCode.AddElem(aBit);
    else
       mCode.SuprElem(aBit);
    mBitsFixed.AddElem(aBit);
}

bool cDecodeFromCoulBits::IsComplete() const
{
    return mBitsFixed.Cardinality() == mSpec->NbBits();
}


const cOneEncoding * cDecodeFromCoulBits::Encoding() const
{
    MMVII_INTERNAL_ASSERT_tiny(IsComplete(),"Cannot decode uncomplete in cDecodeFromCoulBits::Encoding()");
    return mSpec->EncodingFromCode(mCode.FlagBits());
}


/**************************************************/
/*                                                */
/*           cParamCodedTarget                    */
/*                                                */
/**************************************************/


std::string  cParamCodedTarget::NameOfBinCode(int aNum) const
{
    MMVII_INTERNAL_ASSERT_strong(false,"OBSOLOTE TO READ IN FULL SPECIF,NameOfBinCode");

    return "-1111111";
}


void cParamCodedTarget::PCT_AddData(const cAuxAr2007 & anAuxParam,const cSpecBitEncoding * aSpec)
{
    cAuxAr2007  anAux(TheMainTag,anAuxParam);

    MMVII::EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("NbBits",anAux),mNbBit);
    //  MMVII::AddData(cAuxAr2007("SzF",anAux),mSzF);
    MMVII::AddData(cAuxAr2007("CenterF",anAux),mCenterF);
    MMVII::AddData(cAuxAr2007("CornEl1",anAux),mCornEl1);
    MMVII::AddData(cAuxAr2007("CornEl2",anAux),mCornEl2);
    MMVII::AddData(cAuxAr2007("SzCCB",anAux),mSz_CCB);
    MMVII::AddData(cAuxAr2007("ThickN_WhiteInt",anAux),mThickN_WInt);
    MMVII::AddData(cAuxAr2007("ThickN_Code",anAux),mThickN_Code);
    MMVII::AddData(cAuxAr2007("ThickN_WhiteExt",anAux),mThickN_WExt);
    MMVII::AddData(cAuxAr2007("ThickN_BorderExt",anAux),mThickN_BorderExt);
    MMVII::AddData(cAuxAr2007("ThickN_Car",anAux),mThickN_Car);
    MMVII::AddData(cAuxAr2007("ChessBoardAngle",anAux),mChessboardAng);
    MMVII::AddData(cAuxAr2007("ModeFlight",anAux),mModeFlight);
    MMVII::AddData(cAuxAr2007("CBAtTop",anAux),mCBAtTop);
    MMVII::AddData(cAuxAr2007("WithChessBoard",anAux),mWithChessboard);
    MMVII::AddData(cAuxAr2007("WhiteBackGround",anAux),mWhiteBackGround);
    MMVII::AddData(cAuxAr2007("ZeroIsBackGround",anAux),mZeroIsBackGround);
    MMVII::AddData(cAuxAr2007("AntiClockWiseBit",anAux),mAntiClockWiseBit);

    MMVII::AddData(cAuxAr2007("RayOrientTablet",anAux),mRadiusOrientTablet);
    MMVII::AddData(cAuxAr2007("CenterOrientTablet",anAux),mCenterOrientTablet);
    MMVII::AddData(cAuxAr2007("RayCenterMiniTarget",anAux),mRadiusCenterMiniTarget);

    //  MMVII::AddData(cAuxAr2007("SzHalfStr",anAux),mSzHalfStr);


    if (anAux.Input())
    {
        MMVII_INTERNAL_ASSERT_strong(aSpec!=nullptr," cParamCodedTarget::PCT_AddData no Spec in input mode");
        FinishInitOfSpec(*aSpec,false);
	FinishWoSpec();
    }
}
const std::string cParamCodedTarget::TheMainTag = "GeometryCodedTarget";
void AddData(const  cAuxAr2007 & anAux,cParamCodedTarget & aPCT)
{
   aPCT.PCT_AddData(anAux,nullptr);
}

void cParamCodedTarget::InitFromFile(const std::string & aNameFile)
{
    ReadFromFile(*this,aNameFile);
    // Finish();
}

int cParamCodedTarget::ToMultiple_2DeZoom(int aSz) const
{
    return round_to(aSz,2*mSzGaussDeZoom);
}

cPt2di cParamCodedTarget::ToMultiple_2DeZoom(const cPt2di &aSz) const
{
	return cPt2di(ToMultiple_2DeZoom(aSz.x()),ToMultiple_2DeZoom(aSz.y()));
}


void cParamCodedTarget::SetNbPixBin(int aNbPixBin)
{
    mNbPixelBin = ToMultiple_2DeZoom(aNbPixBin); // make size a multiple of 2 * zoom, to have final center at 1/2 pix
}

cParamCodedTarget::cParamCodedTarget(int aNbPixBin) :
   mType             (eTyCodeTarget::eIGNIndoor),  // used to create problem in reading serial, solved, but maintain fake-init
   mNbBit            (9),
   mWithParity       (true),
   mNbRedond         (2),
   mNbCircle         (1),
   mSzGaussDeZoom    (3),
   mNbPixelBin       (-1), // Put fake value, because init is done later
   mSz_CCB           (1),
   mThickN_WInt      (0.5),
   mThickN_Code      (0.35),
   mThickN_WExt      (0.04),
   mThickN_Car       (0.7),
   mThickN_BorderExt (0.04),
   mFactEnlargeCar   (1.0),
   mChessboardAng    (0.0),
   mWithChessboard   (true),
   mWhiteBackGround  (true),
   mZeroIsBackGround (true),
   mAntiClockWiseBit (true),
   mRadiusOrientTablet     (-1),
   mCenterOrientTablet  (0,0),
   mRadiusCenterMiniTarget (-1),
   mModeFlight       (false),  // MPD => def value was not initialized ?
   mCBAtTop          (false),//
   mDecP             ({1,1})  // "Fake" init 4 now
{
    SetNbPixBin(aNbPixBin);
}

void cParamCodedTarget::FinishInitOfSpec(const cSpecBitEncoding & aSpec,bool createInit)
{
   mType = aSpec.mType;
   cMMVII_Appli & anAppli = cMMVII_Appli::CurrentAppli();

   mNbBit = aSpec.mNbBits;
   mWithParity = aSpec.mParity;

   // if we are not in initial creation (i.e we reading existing file) all these modif that are related
   // to the fact that user did or didnt specify are meaningless, value read must not be changed
   if (createInit)
   {
       if (aSpec.mType==eTyCodeTarget::eIGNIndoor)
       {
             // Nothingto do all default value have been setled for this case
       }
       else if ((aSpec.mType==eTyCodeTarget::eIGNDroneSym) || (aSpec.mType==eTyCodeTarget::eIGNDroneTop))
       {
           anAppli.SetIfNotInit(mModeFlight,true);
           anAppli.SetIfNotInit(mCBAtTop,(aSpec.mType==eTyCodeTarget::eIGNDroneTop));
           anAppli.SetIfNotInit(mThickN_WInt,0.05);
           anAppli.SetIfNotInit(mThickN_Code,0.0);
           anAppli.SetIfNotInit(mThickN_WExt,0.0);
           anAppli.SetIfNotInit(mThickN_Car,0.3);
           anAppli.SetIfNotInit(mChessboardAng,-M_PI/4.0);
           anAppli.SetIfNotInit(mThickN_BorderExt,0.05);

           anAppli.SetIfNotInit(mRadiusOrientTablet,0.1);
           anAppli.SetIfNotInit(mCenterOrientTablet,cPt2dr(0.7,0));
       }
       else if (aSpec.mType==eTyCodeTarget::eCERN)
       {
          //  anAppli.SetIfNotInit(mNbBit,20);
           // anAppli.SetIfNotInit(mWithParity,false);

           anAppli.SetIfNotInit(mNbRedond,1);
           anAppli.SetIfNotInit(mThickN_WInt,(mNbBit==20) ? 1.5 : 1.0);
           anAppli.SetIfNotInit(mThickN_Code,(mNbBit==20) ? 1.5 : 1.0);
           anAppli.SetIfNotInit(mThickN_WExt,0.9);
           anAppli.SetIfNotInit(mThickN_BorderExt,0.10);

           anAppli.SetIfNotInit(mWithChessboard,false);
           anAppli.SetIfNotInit(mWhiteBackGround,false);
           anAppli.SetIfNotInit(mAntiClockWiseBit,false);
       }
   }
   mSzHalfStr = (aSpec.mNbDigit+1)/2;

  // StdOut() << " mThickN_CarmThickN_Car " << mThickN_Car  << " " << (aSpec.mNbDigit+1)/2 << "\n";
  //  Split string in 2
   mThickN_Car *= mSzHalfStr;
}

cPt2dr cParamCodedTarget::Pix2Norm(const cPt2dr & aPix) const
{
   return (aPix-mMidle) / mScale;
}
cPt2dr cParamCodedTarget::Pix2Norm(const cPt2di & aPix) const
{
   return Pix2Norm(ToR(aPix));
}



cPt2dr cParamCodedTarget::Norm2PixR(const cPt2dr & aP) const
{
    return mMidle + aP * mScale;
}
cPt2di cParamCodedTarget::Norm2PixI(const cPt2dr & aP) const
{
    return ToI(Norm2PixR(aP));
}

int&    cParamCodedTarget::NbRedond() {return mNbRedond;}
int&    cParamCodedTarget::NbCircle() {return mNbCircle;}

void cParamCodedTarget::FinishWoSpec()
{

  MMVII_INTERNAL_ASSERT_strong(((mNbPixelBin%2)==0),"Require odd pixel 4 binary image");

  if (mModeFlight){
    // mSzBin = cPt2di(0.95*0.707*mNbPixelBin,0.95*mNbPixelBin);

    mSzBin = ToMultiple_2DeZoom(cPt2di(mNbPixelBin,mNbPixelBin*sqrt(2)));

  } else{
    mSzBin = cPt2di(mNbPixelBin,mNbPixelBin);
  }

  double aCumulThick = 1.0;
  mRho_0_EndCCB = mSz_CCB     * aCumulThick;

  aCumulThick += mThickN_WInt;
  mRho_1_BeginCode =  mSz_CCB   * aCumulThick;

  aCumulThick += mThickN_Code;
  mRho_2_EndCode =  mSz_CCB   * aCumulThick;

  aCumulThick += mThickN_WExt;
  mRho_3_BeginCar =  mSz_CCB   * aCumulThick;


  mThickN_Car = std::min(mThickN_Car, mRho_2_EndCode * (sqrt(2)-1));
  {
      std::string aS(mSzHalfStr,'s');
      cIm2D<tU_INT1>  aImStr = ImageOfString_10x8(aS,1);
      cPt2dr aSz = ToR(aImStr.DIm().Sz());
      aSz = aSz / NormInf(aSz);

      //  (l x-R3)^ + (ly-R3)^2 = R3^2
      //  l^2 (x2+y2) - 2lR3 (x+y) + 2R3^2 - R3^2
      // For l   : a L2 + b L + C
      
      tREAL8 a = SqN2(aSz);
      tREAL8 b = -2 * mRho_3_BeginCar * Norm1(aSz);
      tREAL8 c = 2*Square(mRho_3_BeginCar) - Square(mRho_3_BeginCar);

      //  l = (-b +-sqrt(b2-4ac))/2a
      tREAL8 aDelta =  Square(b) - 4 * a * c;
      // smallest root
      tREAL8 aL1 = (-b - sqrt(aDelta)) / (2*a);
      mPSzCar  = aSz * aL1;

  }

  mRho_4_EndCar = mRho_3_BeginCar;
/*
  mRho_4_EndCar = std::max
                  (
                        mRho_3_BeginCar,
                        mRho_3_BeginCar/sqrt(2) + (mThickN_Car*mSz_CCB)
                  );
*/


  aCumulThick = mRho_4_EndCar / mSz_CCB;
  aCumulThick += mThickN_BorderExt;
  mRho_EndIm = mSz_CCB * aCumulThick; 


  // mMidle = ToR(mSzBin-cPt2di(1,1)) / 2.0 - cPt2dr(1,1) ; //  pixel center model,suppose sz=2,  pixel 0 and 1 => center is 0.5
  cPt2di aSz4Mid =  mCBAtTop ? cPt2di(mNbPixelBin,mNbPixelBin) : mSzBin ;
  mMidle = ToR(aSz4Mid-cPt2di(mSzGaussDeZoom,mSzGaussDeZoom)) / 2.0  ; //  pixel center model,suppose sz=2,  pixel 0 and 1 => center is 0.5
								     
  mScale = mNbPixelBin / (2.0 * mRho_EndIm);


  // mCenterF = mMidle / double(mSzGaussDeZoom);
  mCenterF = Norm2PixR(FromPolar(0.0,0.0))  / double(mSzGaussDeZoom);
  mSignAngle = (mAntiClockWiseBit ? 1 :-1);

  mCornEl1 = Norm2PixR(FromPolar(mRho_0_EndCCB,mChessboardAng))  / double(mSzGaussDeZoom);
  mCornEl2 = Norm2PixR(FromPolar(mRho_0_EndCCB,mChessboardAng+M_PI/2.0))  / double(mSzGaussDeZoom);

  //mCornEl1 = mCenterF + FromPolar




  if (false)
  {
      StdOut() <<  "r0 : " << mRho_0_EndCCB << std::endl
               <<  "r1 : " << mRho_1_BeginCode << "\n"
               <<  "r2 : " << mRho_2_EndCode << "\n"
               <<  "r3 : " << mRho_3_BeginCar << "\n"
               <<  "r4 : " << mRho_4_EndCar << "\n"
	       <<  "r5 : " << mRho_EndIm << "\n";

  }
}

int cParamCodedTarget::NbCodeAvalaible() const
{
   return  mDecP.MulBase();
}

int cParamCodedTarget::BaseForNum() const
{
    int aNBC = NbCodeAvalaible();
    if (aNBC < 100)
	    return 10;  // decimal number on 2 dig
    if (aNBC <= 256)
	    return 16;  // hexadecimal on 2 dig
    if (aNBC < 1296)
       return 36;  // alpha num   on 2 dig

    {
        MMVII_DEV_WARNING("Too big number of code for current system");
        return 36;
    }
    MMVII_INTERNAL_ERROR("Too big number of code for current system");
    return -1;
}

std::string cParamCodedTarget::NameOfNum(int aNum) const
{
  int aBase = BaseForNum();
  std::string aRes;
  for (int aK=0 ; aK<2 ;aK++)
  {
      int aDigit = (aNum%aBase) ;
      aNum /= aBase;
      char aCar = (aDigit < 10) ? ('0'+aDigit) : ('A'+(aDigit-10));
      aRes.push_back(aCar);
  }
  std::reverse(aRes.begin(),aRes.end());

   return aRes;
}

std::string cParamCodedTarget::NameFileOfNum(int aNum) const
{
   return "Target_" + NameOfNum(aNum) + ".tif";
}


/* *************************************************** */
/*                                                     */
/*             cNormPix2Bit                            */
/*                                                     */
/* *************************************************** */

/**  Class for specifying the mapping between image and bits, it works with "normalized"
 *   coordinates 
 */
class cNormPix2Bit
{
    public :
         /// Indicate if the point is coding a bit 
	 virtual bool    PNormIsCoding(const cPt2dr & aPt)   const = 0;
         /// Indicate the num of the bit coded
	 virtual int     BitsOfNorm    (const cPt2dr & aPt)  const = 0;

	 ///  Allocator retun one the derivate class
         static cNormPix2Bit * Alloc(const cFullSpecifTarget & aSpecif);

	 virtual ~cNormPix2Bit() {}
    protected :
};

/* *************************************************** */
/*                                                     */
/*             cCircNP2B                               */
/*                                                     */
/* *************************************************** */

/**  A "cNormPix2Bit" where the bits are code in a circle arround the checkboard */

class cCircNP2B : public  cNormPix2Bit
{
     public :
         /// Construct from the full specification
         cCircNP2B(const cFullSpecifTarget & aSpecif);
	 bool    PNormIsCoding(const cPt2dr & aPt)      const   override; ///< is it between circle
	 int     BitsOfNorm    (const cPt2dr & aPt)     const   override; ///< convert teta to a num

     private :
         /// Pre-processing, just convert to polar coordinate
	 cPt2dr  PreProcessCoord(const cPt2dr & aPt)    const; 

	 tREAL8 mRho0;    ///< minimal ray of coding part
	 tREAL8 mRho1;    ///< maximal ray of coding part
	 tREAL8 mTeta0;   ///< origin of teta for first bit
	 int    mNbBits;  ///< number of bit to code
	 tREAL8 mSignT;   ///< sign use to code the sens (clock-wise or not)
};



cCircNP2B::cCircNP2B(const cFullSpecifTarget & aSpecif) :
   mRho0   (aSpecif.Render().mRho_1_BeginCode),
   mRho1   (aSpecif.Render().mRho_2_EndCode),
   mTeta0  (aSpecif.Render().mChessboardAng),
   mNbBits (aSpecif.Specs().mNbBits),
   mSignT  (aSpecif.Render().mSignAngle)
{
}

cPt2dr  cCircNP2B::PreProcessCoord(const cPt2dr & aPt)   const 
{
    return ToPolar(aPt);
}

bool  cCircNP2B::PNormIsCoding(const cPt2dr & aPt) const 
{
    // extract the ray
    tREAL8 aRho = PreProcessCoord(aPt).x();
    // coding part between 2 circle
    return   (aRho>=mRho0)  && (aRho<mRho1) ;
}

int cCircNP2B::BitsOfNorm(const cPt2dr & aPt) const 
{
     // compute the angle, taking into account origin & sens
     tREAL8 aTeta = (PreProcessCoord(aPt).y() -mTeta0) * mSignT;
     // compute bit index from teta by mapping "[0,2PI]"  to "[0,NbBits]" 
     tREAL8 aIndex = mNbBits * (aTeta / (2*M_PI)) ;
     // assure that index is in "[0,NbBits]"
     aIndex = mod_real(aIndex,mNbBits);
     return round_down (aIndex);   
}

/* *************************************************** */
/*                                                     */
/*             cStraightNP2B                           */
/*                                                     */
/* *************************************************** */

/**  A "cNormPix2Bit" where the bits are coded on a regular grid */

class cStraightNP2B : public  cNormPix2Bit
{
     public :
         cStraightNP2B(const cFullSpecifTarget & aSpecif,bool IsSym);
	 bool    PNormIsCoding(const cPt2dr & aPt)      const   override;
	 int     BitsOfNorm    (const cPt2dr & aPt)     const   override;

     private :
         bool     mIsSym;  ///< If true, the coding part is splited in two part
	 tREAL8   mRho1;   ///< begin
         int      mNbBits;
         int      mNbBS2;
	 tREAL8   mSep2L;
};

cStraightNP2B::cStraightNP2B(const cFullSpecifTarget & aSpecif,bool IsSym) :
   mIsSym  (IsSym),
   mRho1   (aSpecif.Render().mRho_EndIm),
   mNbBits (aSpecif.Specs().mNbBits),
   mNbBS2  (mNbBits /2),
   mSep2L  ( IsSym ? 0 : (mRho1 * (1+sqrt(2)/4)))
   
{
    MMVII_INTERNAL_ASSERT_tiny((mNbBS2*2)==mNbBits,"Odd nbbits in cStraightNP2B");
     // StdOut() << "r1=" << mRho1 << " I=" << aSpecif.Render().mRho_EndIm  << " p2n:" << aSpecif.Render().Pix2Norm(cPt2di(0,0)) << std::endl;
}

bool    cStraightNP2B::PNormIsCoding(const cPt2dr & aPt) const   
{
    return  mIsSym                      ?
	    (std::abs(aPt.y()) > mRho1) :    // sym case, the coding part is symetric
	    (aPt.y() >  mRho1)        ;      // else coding part on one side
}

int   cStraightNP2B::BitsOfNorm(const cPt2dr & aPt) const
{
    bool  isLine2 =   (aPt.y()>mSep2L)  ;
    //  map  [-mRho1,+mRho1] to [0,mNbBS2]  (because x is initially signed)
    int aRes = round_down((aPt.x()+mRho1)/(2*mRho1) *mNbBS2)  ;

    // MPD 16/08/23 => correction because  want a trigonometrique ordrer of bits     0 1 2    and not  0 1 2
    // so that bit-shift correspond to rotations            ie :                     xxxxx             xxxxx
    //                                                                               5 4 3             3 4 5
    if (mIsSym && isLine2)
    {
        aRes =  mNbBS2-1 - aRes;
    }
    aRes = std::max(0,std::min(aRes,mNbBS2-1));

    //  if second line, add the NbBits/2 
    aRes =  aRes +  mNbBS2* isLine2;

    return aRes;
}

/* *************************************************** */
/*                                                     */
/*             cNormPix2Bit                            */
/*                                                     */
/* *************************************************** */


bool IsCircularTarge(eTyCodeTarget aType)
{
	return (aType==eTyCodeTarget::eCERN) || (aType==eTyCodeTarget::eIGNIndoor);
}


cNormPix2Bit * cNormPix2Bit::Alloc(const cFullSpecifTarget & aSpecif)
{
   if (IsCircularTarge(aSpecif.Type()))
      return new cCircNP2B(aSpecif);


   switch (aSpecif.Type())
   {
	   /*
         case eTyCodeTarget::eCERN :
         case eTyCodeTarget::eIGNIndoor:
	       return new cCircNP2B(aSpecif);
	       */
         case eTyCodeTarget::eIGNDroneSym:
	       return new cStraightNP2B(aSpecif,true);

         case eTyCodeTarget::eIGNDroneTop:
	       return new cStraightNP2B(aSpecif,false);
	 default :
              return nullptr;
   }

   // return nullptr;
}

/* *************************************************** */
/*                                                     */
/*                cCodedTargetPatternIm                */
/*                                                     */
/* *************************************************** */

enum class eLPT  // Label Pattern Target
           {
              eBackGround,
              eForeGround,
              eCircleSepCar,
              eBorderExt,
              eNumB0   // num first bit
           };

class cCodedTargetPatternIm 
{
     public :
          typedef tU_INT1            tElem;
          typedef cIm2D<tElem>       tIm;
          typedef cDataIm2D<tElem>   tDataIm;

          cCodedTargetPatternIm(cFullSpecifTarget &);

	  tIm  ImCoding() const;

	  tIm MakeOneImTarget(const cOneEncoding & aCode,bool is4Test = false);
     private :
	  cCodedTargetPatternIm(const cCodedTargetPatternIm &) = delete;
	  cPt2di  PDiag(tREAL8 aRhoNorm) const;


	  cFullSpecifTarget &     mSpec;
	  const cParamRenderingTarget&  mRender;
	  cPt2di                        mSzBin;
	  tIm                     mImCoding;
	  tDataIm &               mDIC;

	  tIm               mImTarget;
	  tDataIm &         mDIT;

	  tREAL8            mTeta0;
	  tREAL8            mRhoC;
	  tREAL8            mRho2C;
          bool              mWOriTab;
          tREAL8            mRayOT;
          tPt2dr            mCenterOT;
          tREAL8            mRayCT;
          tREAL8            mRay2CT;
};

cPt2di  cCodedTargetPatternIm::PDiag(tREAL8 aRhoNorm) const
{
       tREAL8 r =  aRhoNorm ;
       cPt2dr aP3r(-r,-r);
       cPt2di  aP3I = mRender.Norm2PixI(aP3r);

       return  mDIC.Proj(aP3I);
}


cCodedTargetPatternIm::cCodedTargetPatternIm(cFullSpecifTarget & aSpec) :
     mSpec       (aSpec),
     mRender     (mSpec.Render()),
     mSzBin      (mRender.mSzBin),
     mImCoding   (mSzBin),
     mDIC        (mImCoding.DIm()),
     mImTarget   (mSzBin),
     mDIT        (mImTarget.DIm()),
     mTeta0      (mRender.mChessboardAng),
     mRhoC       (mRender.mRho_0_EndCCB),
     mRho2C      (Square(mRhoC)),
     mWOriTab    (mRender.mRadiusOrientTablet >0),
     mRayOT      (mRender.mRadiusOrientTablet),
     mCenterOT   (mRender.mCenterOrientTablet * FromPolar(1.0,(M_PI/4.0+mTeta0))),
     mRayCT      (mRender.mRadiusCenterMiniTarget),
     mRay2CT     (Square(mRayCT))
{

    mDIC.InitCste(tElem(eLPT::eBackGround));

    // Structures for computing center of bits
    std::vector<cPt2dr>  aVCenters =  mSpec.BitsCenters();
    std::vector<tREAL8>  aVWeight(mSpec.NbBits(),0.0);

    // structure specifying bits location
    std::unique_ptr<cNormPix2Bit>  aP2B (cNormPix2Bit::Alloc(aSpec));

    tREAL8 aR2Sq = Square(mRender.mRho_2_EndCode);
    tREAL8 aR3Sq = Square(mRender.mRho_3_BeginCar);
    for (const auto & aPix : mDIC)
    {
       cPt2dr aPN = mSpec.Render().Pix2Norm(aPix);
       tREAL8 aR2N = SqN2(aPN);
       tREAL8 aNormInf = NormInf(aPN);
       //  ============  1  Generate the bit coding =======================
       if (aP2B->PNormIsCoding(aPN))  // if point belong to bit-coding space
       {
           int aNumB =  aP2B->BitsOfNorm(aPN);  // get value of bit
	   mDIC.SetV(aPix,int(eLPT::eNumB0)+aNumB);  // marq the image with num

	   aVWeight.at(aNumB) += 1 ;  // increment number of point for this bit
	   aVCenters.at(aNumB) += ToR(aPix);  // accumulate for centroid
       }
       //  ============  2  Generate the central circle =======================
       else if (aR2N <mRho2C)
       {
           eLPT aLab = eLPT::eForeGround;  // a priori mar circle
           if (mSpec.Render().mWithChessboard)
	   {
               // computation to separate the plane in 4 quadrant
               double PIsur2 = M_PI/2.0;
	       tREAL8 aTeta = ToPolar(aPN).y();
               int aIndTeta = round_down((aTeta-mTeta0)/PIsur2);
               if ((aIndTeta%2)==0)
                   aLab = eLPT::eBackGround;
	   }
           if (mWOriTab && (Norm1(aPN-mCenterOT) < mRayOT))
           {
		   // StdOut() <<  "RRRoot " << aPix << aPN  << mCenterOT<< " " << std::endl;
              aLab = eLPT::eForeGround;
           }

	   if ((mRayCT>0) && (SqN2(aPN)<mRay2CT))
	   {
               tREAL8 aLogR = (std::log(std::max(1.0/mRender.mScale,Norm2(aPN))/mRayCT))/std::log(2) ;
	       int aILogR = round_down(aLogR);
               aLab = ((aILogR % 2)!=0) ?  eLPT::eBackGround : eLPT::eForeGround;
	   }
	   mDIC.SetV(aPix,int(aLab));
       }
       else if ((aR2N>aR2Sq) && ( aR2N<=aR3Sq))
       {
	   mDIC.SetV(aPix,int(eLPT::eCircleSepCar));
       }

       if ((aNormInf>mRender.mRho_4_EndCar) && (aNormInf<=mRender.mRho_EndIm))
       {
            mDIC.SetV(aPix,int(eLPT::eBorderExt));
       }
    }

    // compute and memorize the center
    for (size_t aB=0 ; aB< aVWeight.size() ; aB++)
    {

       mSpec.SetBitCenter(aB,aVCenters.at(aB) / tREAL8(aVWeight.at(aB) * mSpec.DeZoomIm() ));
    }
}

cCodedTargetPatternIm::tIm cCodedTargetPatternIm::ImCoding() const {return mImCoding;}

cCodedTargetPatternIm::tIm cCodedTargetPatternIm::MakeOneImTarget(const cOneEncoding & anEnCode,bool is4Test)
{
   // compute gray level for background & foreground
   int aBG_Coul = mSpec.Render().mWhiteBackGround ? 255 : 0;
   int aFG_Coul =  255-aBG_Coul;

   tREAL8 aWSC = 0.9;

   int aColSepCirc = aFG_Coul*(1-aWSC) + aBG_Coul*aWSC;

   // by default all is backgrounf
   mDIT.InitCste(aBG_Coul);

   size_t aCode = anEnCode.Code();
   bool  BGIs_0 = mSpec.Render().mZeroIsBackGround;  // true mean bits 0 are considered background
   for (const auto & aPix : mDIC)
   {
       eLPT aLab =  eLPT(mDIC.GetV(aPix));

       if (aLab!=eLPT::eBackGround)
       {
           bool isBG = true;
           if (aLab==eLPT::eForeGround)
           {
               isBG = false;
           }
           else if (aLab>=eLPT::eNumB0)
           {
               bool BitIs_1 =  (aCode & (size_t(1)<<(int(aLab)-int(eLPT::eNumB0)))) != 0;
               isBG = BitIs_1 !=  BGIs_0;
           }

	   if (!isBG)
	   {
               mDIT.SetV(aPix,aFG_Coul);
	   }
       }

       if (is4Test &&  (aLab== eLPT::eCircleSepCar))
       {
	   mDIT.SetV(aPix,aColSepCirc);
       }
       if (is4Test &&  (aLab== eLPT::eBorderExt))
       {
	   mDIT.SetV(aPix,aColSepCirc);
       }
   }

   // ---------------  caracter insertion  --------------------

   {
        //cPt2di  aSzFt(10,10);
        std::string aFullName = anEnCode.Name();
	size_t aIndSplit = (aFullName.size() +1)/2;

	// Corners of string, 
        cPt2di  aP00 = PDiag(mRender.mRho_4_EndCar);
	// cPt2di  aP11 = PDiag(mRender.mRho_4_EndCar-mRender.mThickN_Car *mRender.mFactEnlargeCar) ; // for a 1 length caracr
	cPt2di  aP11 = aP00 + ToI(mRender.mPSzCar*mRender.mScale);

        //StdOut() << " P00=" << aP00 << " P11=" << aP11 << " SzC=" << mRender.mPSzCar << " SC=" << mRender.mScale << "\n";

	// udate highth of string, to adapt to length (aIndSplit is maximal legnt of 2 substrings)
	// int aHigth = (aP11.y()-aP00.y()) / aIndSplit;
	int aHigth = (aP11.y()-aP00.y()) ;
	aP11 =  cPt2di(aP11.x(),aP00.y() + aHigth);

	// loop for processing the 2 subsrt string 
	for (const auto & aLeftSS : {true,false} )
	{
             int aFG_StrCoul = aFG_Coul;
             // extract one of the 2 substr
             std::string aName = aFullName.substr((aLeftSS?0:aIndSplit) , (aLeftSS?aIndSplit:std::string::npos));
	     // get image of string
	     cIm2D<tU_INT1>  aImStr = ImageOfString_10x8(aName,1);
	     cDataIm2D<tU_INT1>&  aDImStr =  aImStr.DIm();

	     // compute the size of each pixel to fill the hights
	     tREAL8 aSzPixStr = aHigth / tREAL8(aDImStr.Sz().y());

	     cPt2di aPOri = aP00;
	     if (! aLeftSS)
	     {
	          int aWitdhStr = round_ni(aDImStr.Sz().x())*aSzPixStr;  // Width of string
	          //  Off + WidtStr/2 =  SzBin.x - P4.x
	          int OffsX  = mSzBin.x()  - aP00.x() -  aWitdhStr;
		  aPOri = cPt2di(OffsX,aP00.y());
	     }

	     cPt2di  aP4Sym = ToI(mSpec.Render().mMidle * 2.0) - cPt2di(1,1);

	     for (const auto & aPixStr : aDImStr)
	     {
                  bool isCar = aDImStr.GetV(aPixStr);
                  if (isCar || is4Test)
	          {
                       int aCoul = isCar ? aFG_StrCoul : 128;
                       cPt2di aP0 = aPOri+ToI(ToR(aPixStr)*aSzPixStr);
                       cPt2di aP1 = aPOri+ToI(ToR(aPixStr+cPt2di(1,1))*aSzPixStr);

		       for (const auto& aPixIm : cRect2(aP0,aP1))
		       {
                           mDIT.SetVTruncIfInside(aPixIm,aCoul);
                           mDIT.SetVTruncIfInside(aP4Sym-aPixIm,aCoul);
		       }
	          }
	     }

	}
   }


   tIm aRes = mImTarget.GaussDeZoom(mSpec.DeZoomIm());

   // in debug mode, marq with one pixel the center
   if (is4Test)
   {
      for (const auto & aC : mSpec.BitsCenters())
      {
          for (const auto aP : cRect2::BoxWindow(2))
	      aRes.DIm().SetV(ToI(aC)+aP,128);
      }
   }

   return aRes;
}

/* *************************************************** */
/*                                                     */
/*                cFullSpecifTarget                    */
/*                                                     */
/* *************************************************** */

cFullSpecifTarget::cFullSpecifTarget() :
   mCEC         (nullptr),
   mCTPI        (nullptr)
{
}

cFullSpecifTarget::cFullSpecifTarget(const cBitEncoding& aBE,const cParamRenderingTarget& aRender) :
   //cFullSpecifTarget(),
   mCEC         (nullptr),
   mCTPI        (nullptr),
   mBE          (aBE),
   mRender      (aRender),
   mBitsCenters (NbBits(),cPt2dr(0,0))
{
}

cCodedTargetPatternIm * cFullSpecifTarget::AllocCTPI()
{
    if (mCTPI==nullptr)
       mCTPI = new  cCodedTargetPatternIm(*this);

    return mCTPI;
}

cCompEquiCodes *   cFullSpecifTarget::CEC() const
{
    if (mCEC==nullptr)
    {
       size_t aPer = NbBits()/Specs().mFreqCircEq;
       mCEC = cCompEquiCodes::Alloc(NbBits(),aPer);
    }

    return mCEC;
}


cFullSpecifTarget::tIm   cFullSpecifTarget::ImagePattern()
{
	return AllocCTPI()->ImCoding();
}

cFullSpecifTarget::tIm   cFullSpecifTarget::OneImTarget(const cOneEncoding & aCode,bool ForTest)
{
	return AllocCTPI()->MakeOneImTarget(aCode,ForTest);
}



cFullSpecifTarget::~cFullSpecifTarget()
{
    delete mCTPI;
    delete mCEC;
}

const std::vector<cOneEncoding> & cFullSpecifTarget::Encodings()   const {return mBE.Encodings();}
const cSpecBitEncoding &          cFullSpecifTarget::Specs()       const {return mBE.Specs();}
const cParamRenderingTarget &     cFullSpecifTarget::Render()      const {return mRender;}
const std::string &               cFullSpecifTarget::Prefix()      const {return Specs().mPrefix;}
const std::vector<cPt2dr>&        cFullSpecifTarget::BitsCenters() const {return mBitsCenters;}
size_t                            cFullSpecifTarget::NbBits()      const {return Specs().mNbBits;}
int                               cFullSpecifTarget::DeZoomIm()    const {return mRender.mSzGaussDeZoom;}
eTyCodeTarget                     cFullSpecifTarget::Type()        const {return Specs().mType;}
size_t                            cFullSpecifTarget::MinHammingD() const {return Specs().mMinHammingD;}

cPt2dr    cFullSpecifTarget::Pix2Norm(const cPt2dr & aPix) const {return mRender.Pix2Norm(aPix*tREAL8(mRender.mSzGaussDeZoom));}
cPt2dr    cFullSpecifTarget::Norm2Pix(const cPt2dr & aPix) const {return mRender.Norm2PixR(aPix)/tREAL8(mRender.mSzGaussDeZoom);}


tREAL8 cFullSpecifTarget::Rho_0_EndCCB() const    {return mRender.mRho_0_EndCCB;}
tREAL8 cFullSpecifTarget::Rho_1_BeginCode() const {return mRender.mRho_1_BeginCode;}
tREAL8 cFullSpecifTarget::Rho_2_EndCode() const   {return mRender.mRho_2_EndCode;}
tREAL8 cFullSpecifTarget::Rho_3_BeginCar() const   {return mRender.mRho_3_BeginCar;}


const cPt2dr & cFullSpecifTarget::Center() const {return mRender.mCenterF;}
const cPt2dr & cFullSpecifTarget::CornerlEl_BW() const {return mRender.mCornEl1;}
const cPt2dr & cFullSpecifTarget::CornerlEl_WB() const {return mRender.mCornEl2;}
bool  cFullSpecifTarget::AntiClockWiseBit() const { return mRender.mAntiClockWiseBit; }
bool  cFullSpecifTarget::ZeroIsBackGround() const { return mRender.mZeroIsBackGround; }
bool  cFullSpecifTarget::WhiteBackGround() const  { return mRender.mWhiteBackGround; }
void  cFullSpecifTarget::SetWhiteBackGround(bool aWB) {mRender.mWhiteBackGround = aWB;}


bool cFullSpecifTarget::BitIs1(bool IsWhite) const
{
            //  For example 3D ICON with white pixel
	    //     true          false                     true       
	    //      !(T^ F ^T)  => 1   and that 's the case, Youpi !!
    return  !(   (IsWhite ^  mRender.mWhiteBackGround)  ^ mRender.mZeroIsBackGround );
}




void  cFullSpecifTarget::SetBitCenter(size_t aBit,const cPt2dr& aC)
{
      mBitsCenters.at(aBit) = aC;
}


std::string cFullSpecifTarget::NameOfImPattern() const
{
   return  Prefix() + "_Pattern"+ ".tif"; 
}

std::string cFullSpecifTarget::NameOfEncode(const cOneEncoding & anEncode) const
{
    return   Prefix()  +  "_Target_"+  anEncode.Name() +   ".tif";
}


const std::string cFullSpecifTarget::TheMainTag = "FullSpecifTarget";

void cFullSpecifTarget::AddData(const  cAuxAr2007 & anAuxParam)
{
     cAuxAr2007 anAux(TheMainTag,anAuxParam);

     mBE.AddData(anAux);
     mRender.PCT_AddData(anAux,&(mBE.Specs()));
     StdContAddData(cAuxAr2007("Centers",anAux),mBitsCenters);
}

void AddData(const  cAuxAr2007 & anAux,cFullSpecifTarget & aSpecif)
{
   aSpecif.AddData(anAux);
}

cFullSpecifTarget *  cFullSpecifTarget::CreateFromFile(const std::string & aName)
{
    cFullSpecifTarget * aRes = new cFullSpecifTarget;
    ReadFromFile(*aRes,aName);


    if (0)  // we dont reset nb of bit, because don want do generate comments as 100100111
    {
        for (auto & anEncod : aRes->mBE.Encodings())
            anEncod.SetNBB(aRes->NbBits());
    }

    return aRes;
}

const cOneEncoding * cFullSpecifTarget::EncodingFromCode(size_t aBitFlag) const
{
   // 1 Extract the low-standard value, if it exist
   const cCelCC * aCell = CEC()->CellOfCode(aBitFlag);
   if (aCell==nullptr) return nullptr;

   aBitFlag = aCell->mLowCode;

   // 2 Extract the encoding with adequat code
   for (const auto & anEncod : Encodings())
   {
       if (anEncod.Code() == aBitFlag)
          return &anEncod;
   }

   //  none was found
   return nullptr;
}

const cOneEncoding * cFullSpecifTarget::EncodingFromName(const std::string &aName) const
{
   //  Extract the encoding with adequate name
   for (const auto & anEncod : Encodings())
       if (anEncod.Name() == aName)
          return &anEncod;

   //  none was found
   return nullptr;
}




/* *************************************************** */
/*                                                     */
/*                cAppliGenCodedTarget                 */
/*                                                     */
/* *************************************************** */

class cAppliGenCodedTarget : public cMMVII_Appli
{
     public :
        cAppliGenCodedTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
	eTyCodeTarget      Type();

     private :


        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        int PerGen() const { return mPerGen;}   //CM: avoid mPerGen unused


	cPhotogrammetricProject  mPhgrPr;  ///< Used to generate dir visu
	int                mPerGen;  // Pattern of numbers
	int                mZoomShow;
	std::string        mNameBE;
	cBitEncoding       mBE;

	cParamCodedTarget  mPCT;
	bool               mDoMarkC;
	std::string        mPatternDoImage;
	std::string        mPrefixVisu;
	int                mNbPixBin;
        std::string        mNameOut;
        bool               mIm4Test;   ///< Do we generate image for inspection (and not for printing)
};

eTyCodeTarget cAppliGenCodedTarget::Type() {return mBE.Specs().mType ;}


cAppliGenCodedTarget::cAppliGenCodedTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhgrPr       (*this),
   mPerGen       (10),
   mDoMarkC      (false),
   mPrefixVisu   (""),
   mNbPixBin     (1800),
   mIm4Test      (false)
{
}

cCollecSpecArg2007 & cAppliGenCodedTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return
      anArgObl
          // <<   Arg2007(mNameBE,"Xml/Json name for bit encoding struct",{{eTA2007::XmlOfTopTag,cBitEncoding::TheMainTag}})
          <<   Arg2007(mNameBE,"Xml/Json/Dmp name for bit encoding struct",{{eTA2007::FileAny}})
   ;
}

// cParamCodedTarget   mRayOrientTablet

cCollecSpecArg2007 & cAppliGenCodedTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mPatternDoImage,"PatIm","Pattern for generating image (def no generation)")
          << AOpt2007(mPrefixVisu,"PrefixVisu","To add in image name when PatIm is used",{eTA2007::HDV})
          << AOpt2007(mIm4Test,"I4T","Generate image for test/inspection, not for use",{eTA2007::HDV})
          << AOpt2007(mPCT.mRadiusCenterMiniTarget,"RayMCT","Rayon \"mini\" center target (for topo)",{eTA2007::HDV})
          // << AOpt2007(mPCT.mNbBit,"NbBit","Nb Bit printed",{eTA2007::HDV})
          // << AOpt2007(mPCT.mWithParity,"WPar","With parity bit",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_WInt,"ThW0","Thickness of interior white circle",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_Code,"ThCod","Thickness of bin-coding black circle",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_WExt,"ThSepCar","Thickness of sep bin-code / ahpha code",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_Car,"ThCar","Thickness of separation alpha ccode ",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_BorderExt,"ThBExt","Thickness of border exterior",{eTA2007::HDV})
          << AOpt2007(mPCT.mChessboardAng,"Theta","Origin angle of chessboard pattern ",{eTA2007::HDV})
          << AOpt2007(mPCT.mWhiteBackGround,"WhiteBG","White back ground")
          << AOpt2007(mPCT.mModeFlight,"ModeFlight","Special mode for Patricio ",{eTA2007::HDV})
	  << AOpt2007(mPCT.mRadiusOrientTablet,"SzOrFig","Size of \"diamond\" for orientation")
          << AOpt2007(mDoMarkC,"MarkC","Mark center of bits, just for verif ",{eTA2007::HDV,eTA2007::Tuning})
          << AOpt2007(mZoomShow,"ZoomShow","Zoom to generate a high resolution check images",{eTA2007::Tuning})
          << AOpt2007(mNbPixBin,"NbPixBin","Size of binary image when printing",{eTA2007::HDV})
          << AOpt2007(mNameOut,"Out","Name for out file")
   ;
}


int  cAppliGenCodedTarget::Exe()
{
    //  Bench_Target_Encoding();

       // anAppli.SetIfNotInit(mWithParity,false);

   mPhgrPr.FinishInit();
   //if (IsInit(&mNbPixBin))
   mPCT.SetNbPixBin(mNbPixBin);

   ReadFromFile(mBE,mNameBE);
   mPCT.FinishInitOfSpec(mBE.Specs(),true);
   mPCT.FinishWoSpec();

   cFullSpecifTarget  aFullSpec(mBE,mPCT);

   // Activate the computaion of centers
   aFullSpec.ImagePattern();
   std::string aDirVisu = mPhgrPr.DirVisuAppli();

   if (IsInit(&mPatternDoImage))
   {
      //  generate the pattern image
      aFullSpec.ImagePattern().DIm().ToFile(aDirVisu+mPrefixVisu + aFullSpec.NameOfImPattern());

      // parse all encodings
      for (const auto & anEncode : aFullSpec.Encodings())
      {
          if (MatchRegex(anEncode.Name(),mPatternDoImage))
	  {
             cCodedTargetPatternIm::tIm anIm = aFullSpec.OneImTarget(anEncode,mIm4Test);

             std::string aName = aFullSpec.NameOfEncode(anEncode);
             anIm.DIm().ToFile(aDirVisu+mPrefixVisu +aName);
             StdOut() << aName << std::endl;
	  }
      }
   }

   // std::string aName = aFullSpec.Prefix()+"_FullSpecif."+TaggedNameDefSerial();
   if (! IsInit(&mNameOut)) 
   {
      // mNameOut = aFullSpec.Prefix()+"_FullSpecif."+  LastPostfix(mNameBE);
      // Modif MPD : it seems more coherent to maintain users naming  ...
      mNameOut =  LastPrefix(mNameBE) +"_FullSpecif."+  LastPostfix(mNameBE);
   }
   SaveInFile(aFullSpec, mNameOut);

   if (0)  // test reload
   {
         auto aPtr = cFullSpecifTarget::CreateFromFile(mNameOut);
	 StdOut() << "NNN=" << mNameOut << std::endl;
	 delete aPtr;
   }

   if (IsInit(&mZoomShow))
   {
      TestReloadAndShow_cFullSpecifTarget(aDirVisu,mNameOut,mZoomShow);
   }



   return EXIT_SUCCESS;
}


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_GenCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliGenCodedTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecGenCodedTarget
(
     "CodedTargetGenerate",
      Alloc_GenCodedTarget,
      "Generate images for coded target",
      {eApF::CodedTarget},
      {eApDT::Console},
      {eApDT::Image},
      __FILE__
);


};


