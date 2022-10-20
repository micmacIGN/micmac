
#include "V1VII.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_DeclareCste.h"

static bool  DEBUG_EDM = false;

// using namespace MMVII;
/*
using namespace MMVII;
using MMVII::BLANK;
using MMVII::round_ni;
using MMVII::round_up;
*/

// It's a bit strange to put local thing also in MMVII namespace, but else there is
// many conflicts between  MMV1 (include here because MicMac)  and MMVII.
// I realized that lately ....  so let it like that for now
namespace MMVII
{


void  SetBoxes12FromPx
      (
         cBox2di &          aBoxIn1,
         cBox2di &          aBoxIn2,
	 int &              aPxMin,
	 int &              aPxMax,
	 int &              aOffsetPx,
	 eModePaddingEpip   aModePad,
         const cBox2di *    aBoxFile2,
	 bool               DoDebug
      )
{
   cBox2di aBoxInit1(aBoxIn1);
   // We compute the Box2 homologous of Box1, taking into account dilatation due to px-intervall
   aBoxIn2 = DilateFromIntervPx(aBoxIn1,aPxMin,aPxMax);
   if (aBoxFile2)
      aBoxIn2 = aBoxIn2.Inter(*aBoxFile2);
   
   if (aModePad==eModePaddingEpip::eMPE_NoPad)
   {
   }
   else if (aModePad==eModePaddingEpip::eMPE_PxPos)
   {
      aBoxIn1 = cBox2di(aBoxIn1.P0(),aBoxIn1.P0()+aBoxIn2.Sz());
   }
   else if (aModePad==eModePaddingEpip::eMPE_PxNeg)
   {
      aBoxIn1 = cBox2di(aBoxIn1.P1()-aBoxIn2.Sz(),aBoxIn1.P1());
   }
   else if (aModePad==eModePaddingEpip::eMPE_SzEq)
   {
      aBoxIn1 = aBoxIn2;
   }

    //  We want to be sure that whatever happened before BoxIn1 contain the initial value BoxUtIn1
    //  (may creat bug if not) and also dont change the size now that it fit the possible requirements of deeps methods
    //  Case rare bu posible with px computed from previous steps

   if (aModePad!=eModePaddingEpip::eMPE_NoPad)
   {
      // Par ex : INIT [10 90]  IN1 [20,80] U=Init,   dX0=-10  dX1=10
      cBox2di aBoxU1 = aBoxIn1.Sup(aBoxInit1);
      int aDX0 = aBoxU1.P0().x() - aBoxIn1.P0().x();
      int aDX1 = aBoxU1.P1().x() - aBoxIn1.P1().x();

      aBoxIn1 = DilateFromIntervPx(aBoxIn1,aDX0,aDX1);
      aBoxIn2 = DilateFromIntervPx(aBoxIn2,aDX0,aDX1);
   }

   aOffsetPx = aBoxIn2.P0().x() - aBoxIn1.P0().x();
   //  Offset = Delta12, if px computed in box == 0, then "real" one will be Delta12
   aPxMin += aOffsetPx;
   aPxMax += aOffsetPx;
}


  // Put all the stuff that dont vocate to be exported in namespce
namespace NS_EpipGenDenseMatch
{

class cAppli ;  // class for main application
class cOneIm;  // class for representing on of each of both images
class cOneLevel; // class for representing each level of scale in one image
struct cParam1Match; // For parallelizing memorize the parameters on one match (used after paral)

typedef std::shared_ptr<cOneLevel> tPtrLev;
typedef std::shared_ptr<cOneIm> tPtrIm;

/************************************************/


class cOneLevel
{
       cOneLevel(const cOneLevel&) = delete;
   public :
       friend class cOneIm;
       friend class cAppli;

       // =========== Declaration ========
       cOneLevel(cOneIm &,int aLevel);

       void MakeImPx();  ///< Create the file for paralax at a given level
       void Purge(); ///< Remove file unused after each match

       /// Estimate interval of paralax in a given box
       void EstimateIntervPx(cParam1Match &,const cBox2di & aBF1,const cBox2di & aBF2,int aPaxMax) const;
       /// Once clipped match was done, save in global file
       void SaveGlobPx(const cParam1Match & aParam) const;


             // --- Many method for file name and command generation
                   
                /// Add the directory, will depend if it is an initial or created image
       std::string  StdFullName(const std::string & aName) const;
                 //  ------  Generate Commands
                   /// Generate the clipped images
       std::string StrComClipIm(bool ModeIm,const cPt2di & aInd,const cParam1Match &) const;
       std::string StrComReduce(bool ModeIm=true) const; ///< Generate the string for computing reduced images
                 //  ------  Generate name for cliped images
       std::string  NameClip(const std::string & aPrefix,const cPt2di & aInd) const; ///< Genreik name
       std::string  NameClipIm(const cPt2di & aInd) const;  ///<  Clipped input image
       std::string  NameClipMasq(const cPt2di & aInd) const;  ///< Clipped input masq
       std::string  NameClipPx(const cPt2di & aInd) const;    ///< Clipped output paralax
       std::string  NameClipDirTmp(const cPt2di & aInd) const;  ///< Directory tmp for cliped match

       // =========== Inline Definition ========
       cDataFileIm2D  DFI()   const {return cDataFileIm2D::Create(mNameIm,false);}
       cBox2di        BoxIm() const {return cBox2di(DFI().Sz());}
       std::string NameImOrMasq(bool ModeIm) const {return ModeIm ?mNameIm : mNameMasq;}
       int         Level() const {return mLevel;}
   private :
       // =========== Data ========

       cOneIm&          mIm;       ///< Ref to the image containing this level
       cAppli &         mAppli;    ///< Ref to global application
       int              mLevel;    ///< Level in the pyram (0 for full image)
       cOneLevel*       mUpLev;    ///< Uper Level , 0 for top
       cOneLevel*       mDownLev;  ///< Down Level, 0 for bottom
       std::string      mNameIm;   ///< Full Name of the image at this level
       std::string      mNameMasq; ///< Full Name of the masq
       std::string      mNamePx;   ///< For image 1, full name of paralax result
};


class cOneIm
{
       cOneIm(const cOneIm &) = delete;
    public :
       friend class cOneLevel;
       friend class cAppli;
       // =========== Declaration ========
       cOneIm(cAppli& anAppli,const std::string & aName,bool IsIm1);

       /** Create the data struct for pyramid , must be done after, Appli will need size of 
          image to compute the number of levels */
       void CreateLevels(int aNbLevels);


       // =========== Inline Definitions ========
       cOneLevel & LevAt(int aK) {return *(mLevels.at(aK));}
       
    private :
       // =========== Data ========
       cAppli &      mAppli; ///< Ref to global application
       bool                            mIsIm1; ///< Is it first image
       std::string                     mNameIm;  ///< Name of full initial Image
       std::string                     mNameMasq; ///<  Name of full initial Masq
       cDataFileIm2D                   mPFileImFull; ///< Data for File of name mNameIm1
       std::vector<tPtrLev>            mLevels;  ///< Vector of different scale
};


class cAppli : public cMMVII_Appli
{
     public :
        friend class cOneIm;
        friend class cOneLevel;

       // =========== Declaration ========
                 // --- Method to be a MMVII application
        cAppli(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        cAppliBenchAnswer BenchAnswer() const override ; ///< Has it a bench, default : no
        int  ExecuteBench(cParamExeBench &) override ;


                 // --- Others 
        void MakePyramid(); ///< Generate the pyramid of image
        void MatchOneLevel(int aLevel);  ///< Compute the matching a given level of the pyramid
        std::string ComMatch (cParam1Match &) ; ///< Command for match, depend of selected method
        void SetOutPut(std::string & aNamePx);  ///< Fix to same value aNamePx and mOutPx

       // =========== Inline Definition ========
       cOneIm & Im1() {return *(mIms.at(0));}
       cOneIm & Im2() {return *(mIms.at(1));}

     private :
       // =========== Data ========
            // Mandatory args
        eModeEpipMatch  mModeMatchFinal; ///< Method for Matching (MicMac, CNN ...) at highest resol
        std::string     mNameIm1;  ///< Name first image
        std::string     mNameIm2;  ///< Name second image

            // Optional args
        eModePaddingEpip mModePad;
        eModeEpipMatch   mModeMatchInit;  ///< Method for Matching at low resol
	double         mSzBasculeMM;  ///< sz in mega pixel of transition between MatchInit/MatchFinal
        bool           mRandPaded; ///< Force Px to be >=0 (assuming the interval prevision on paralax is right)
        cPt2di         mSzTile;    ///< Size of tiles for matching
        std::string    mOutPx;     ///< Output file
        cPt2di         mSzOverL;   ///< Size of overlap between tile to limit sides effect
        double         mIncPxProp; ///< Inc prop to size, used at first level
        int            mIncPxCste; ///< Cste to add to the interv of Px to take into account innovation
        int            mIncPxMin;  ///< Minimal size of Px interv
        double         mMaxRatio;  ///< Maximal ratio between to level of pyramid
        double         mPropEstim; ///< Proportion used in histogram of paralax 4 estimate interval
        int            mNbDecim;   ///< Nb of decimation before estimating interval

            // Optional tuning args to avoid recomputation in developpment step
        bool           mDoPyram;  ///< Do we do the pyramid
        bool           mDoClip;   ///< Do we do the  clip
        bool           mDoMatch;  ///< Do we do the match
        bool           mDoPurge;  ///< Do we purge the result

            // Computed values & auxilary methods on scales, level ...
        eModeEpipMatch  mModeMatchCur;   ///< Current Method for Matching (mModeMatchFinal or mModeMatchInit)
        int    mNbLevel ;  ///< Number of level in the pyram [0 mNbLevel]  , 0 = initial image
        double mRatioByL ; ///< Scale between 2 successive levels
	int    mMaxAmplPxCur;  ///< Maximal amplitude of Px 

        std::vector<tPtrIm>     mIms;  ///< Contain the 2 images
        bool           mDoFileCommands;  ///<  Do we store all commands
	std::string    mNameFileComs;   ///< File wher we store all commands
};

struct cParam1Match
{
    public :
        // cParam1Match(const cParam1Match&) = delete;
        cParam1Match
        (
              const cBox2di &   aBoxIn1,
              const cBox2di &   aBoxOut,
              const cPt2di &    anIndex,
              const cOneLevel & aILev1,
              const cOneLevel & aILev2
         )  :
             mBoxIn1      (aBoxIn1),
             mBoxUtiIn1   (aBoxIn1),
             mBoxIn2      (cBox2di::Empty()), // Must be computed with Paralacx interval
             mBoxOut      (aBoxOut), 
             mIndex       (anIndex),
             mClipNameIm1 ( FileOfPath(aILev1.NameClipIm(anIndex),false)),
             mClipNameIm2 ( FileOfPath(aILev2.NameClipIm(anIndex),false)),
             mClipDirTmp  ( aILev1.NameClipDirTmp(anIndex)),
             mClipNamePx  ( FileOfPath(aILev1.NameClipPx(anIndex),false)),
             mPxMin       (0),
             mPxMax       (0),
             mCanDoMatch  (true),
	     mLevel       (aILev1.Level())
         {
             static int aCpt = 0; 
             mId = aCpt++;
         };

               cBox2di           mBoxIn1; ///< Box input for image 1
               cBox2di           mBoxUtiIn1; ///< May differ from BoxIn1, in case BoxIn1 was elarged to equal sz of Box2
               cBox2di           mBoxIn2; ///< Box input for image 2
         const cBox2di           mBoxOut;  ///< Box out to save computatio,
         const cPt2di            mIndex;   ///< Index of the tile
         const std::string       mClipNameIm1;  ///< Name cliped Im1 of the tile
         const std::string       mClipNameIm2;  ///< Name cliped Im2 of the tile
         const std::string       mClipDirTmp;   ///< Name Dir Tmp for cliped match of the tile
         const std::string       mClipNamePx;   ///< Name resulting paralx of cliped match
               int               mPxMin;   ///< Min computed paralax
               int               mPxMax;   ///< Max compted paralax
               int               mOffsetPx; ///< Offset between cliped px and global px, due Box1 != Box2
               bool              mCanDoMatch;  ///< Is there enough point to do the match
               int               mId; ///< Identifier, added for debuging
	       int               mLevel;

         bool ToDebug() const {return false && DEBUG_EDM && (mLevel==3) && (mIndex==cPt2di(0,0));}
};


/*  ============================================== */
/*                                                 */
/*             cOneLevel                           */
/*                                                 */
/*  ============================================== */

std::string cOneLevel::StdFullName(const std::string & aName) const
{
    return  (mLevel==0)                                                      ?
            mIm.mAppli.DirProject()+ aName                                   :
            mIm.mAppli.DirTmpOfCmd() + "Lev" + ToStr(mLevel) + "_" + ChgPostix(aName,"tif")   ;
}

cOneLevel::cOneLevel(cOneIm &anIm,int aLevel) :
    mIm       (anIm),
    mAppli    (mIm.mAppli),
    mLevel    (aLevel),
    mUpLev    (nullptr),
    mDownLev  (nullptr),
    mNameIm   (StdFullName(mIm.mNameIm)),
    mNameMasq (StdFullName(mIm.mNameMasq)),
    mNamePx   (AddBefore(ChgPostix(mNameIm,"tif"),"Px_"))
{
   if (aLevel==0)
      mAppli.SetOutPut(mNamePx);
}

void  cOneLevel::MakeImPx()
{
   cDataFileIm2D aDataIm = cDataFileIm2D::Create(mNameIm,false);
   cDataFileIm2D::Create(mNamePx,eTyNums::eTN_REAL4,aDataIm.Sz());
}

std::string cOneLevel::StrComReduce(bool ModeIm) const
{

   return  "mm3d ScaleIm"
           + BLANK + NameImOrMasq(ModeIm)
           + BLANK + ToStr(mIm.mAppli.mRatioByL)
           + BLANK + std::string("Out=")  + mDownLev->NameImOrMasq(ModeIm)
           + std::string(ModeIm ? "" : " ModMasq=1")
   ;
}


std::string Index2Str(const cPt2di & aInd)
{
    return ToStr(aInd.x()) +  "_" + ToStr(aInd.y()) +"_" ;
}

std::string  cOneLevel::NameClip(const std::string & aPrefix,const cPt2di & aInd) const
{
     return     mIm.mAppli.DirTmpOfCmd() 
            +   aPrefix + Index2Str(aInd)
            +   ChgPostix(FileOfPath(mNameIm),"tif")
     ;
}

std::string  cOneLevel::NameClipIm(const cPt2di & aInd) const
{
     return     NameClip("ClipIm_",aInd);
}
std::string  cOneLevel::NameClipMasq(const cPt2di & aInd) const
{
     return     V1NameMasqOfIm(NameClipIm(aInd));
}

std::string  cOneLevel::NameClipPx(const cPt2di & aInd) const
{
     return     NameClip("ClipPx_",aInd);
}

std::string  cOneLevel::NameClipDirTmp(const cPt2di & aInd) const
{
     return     "ClipDirTmp_" + Index2Str(aInd) + StringDirSeparator();
}
  

std::string  cOneLevel::StrComClipIm(bool ModeIm,const cPt2di & aInd,const cParam1Match & aParam) const
{
   bool IsIm1 = mIm.mIsIm1;
   const cBox2di & aBox = IsIm1 ? aParam.mBoxIn1 : aParam.mBoxIn2;


   std::string aCom =
         "mm3d ClipIm" 
          + BLANK +  NameImOrMasq(ModeIm)
          + BLANK +  ToStrComMMV1(aBox.P0())
          + BLANK +  ToStrComMMV1(aBox.Sz())
          + BLANK +  "Out=" + (ModeIm ? NameClipIm(aInd) : NameClipMasq(aInd))
   ;

   if (IsIm1)
   {
      if (aParam.mBoxIn1.P1().x() != aParam.mBoxUtiIn1.P1().x())
         aCom = aCom + BLANK + "XMaxNot0=" + ToStr(aParam.mBoxUtiIn1.P1().x());

      if (aParam.mBoxIn1.P0().x() != aParam.mBoxUtiIn1.P0().x())
         aCom = aCom + BLANK + "XMinNot0=" + ToStr(aParam.mBoxUtiIn1.P0().x());
   }

   if (ModeIm &&   mAppli.mRandPaded)
      aCom = aCom + " AmplRandVout=255";

   return aCom;
}

void cOneLevel::SaveGlobPx(const cParam1Match & aParam) const
{
   // Read part of Cliped Px that has to be saved (e.q. without border)
   cIm2D<tREAL4>  aImClipPx(aParam.mBoxOut.Sz());
   cPt2di aDecInOut = aParam.mBoxOut.P0()-aParam.mBoxIn1.P0();

   aImClipPx.Read(cDataFileIm2D::Create(NameClipPx(aParam.mIndex),false),aDecInOut);
   cDataIm2D<tREAL4> & aDIm = aImClipPx.DIm();
   for (const auto & aP : aDIm)
   {
        aDIm.AddVal(aP, aParam.mOffsetPx);
   }
   aImClipPx.Write(cDataFileIm2D::Create(mNamePx,false),aParam.mBoxOut.P0());
}

void cOneLevel::EstimateIntervPx
     (
          cParam1Match & aParam,
          const cBox2di & aBoxFile1,
          const cBox2di & aBoxFile2,
	  int aAmplMaxPax 
     ) const
{

   //  Compute intervals of paralax
   if (mDownLev==nullptr)
   {
      // Make a rough estimation using a constant steep hypothesis
      int aIntervPx = round_up(aBoxFile2.Sz().x() * mAppli.mIncPxProp) ;
      aParam.mPxMin = -aIntervPx;
      aParam.mPxMax =  aIntervPx;
   }
   else
   {
      int aNbDecim = mAppli.mNbDecim;       // will not change so much the accuracy and much faster for sort
      double  aPropEst = mAppli.mPropEstim; // For robust estim, dont take exactly Min & Max
      double aRatio= mAppli.mRatioByL;

      // Be inialized with def values

      cDataFileIm2D aRedFilePx   = cDataFileIm2D::Create(mDownLev->mNamePx,false);
      cDataFileIm2D aRedFileMasq = cDataFileIm2D::Create(mDownLev->mNameMasq,false);


      // Box of Im1, of reduced size, include in reduce file
      cBox2di aBoxRed = (aParam.mBoxIn1.ToR()* (1.0/aRatio)).ToI().Inter(aRedFilePx);
 
      if (aBoxRed.IsEmpty()) return;

      cIm2D<tREAL4>  aImPx(aBoxRed,aRedFilePx);
      cIm2D<tU_INT1> aImMasq(aBoxRed,aRedFileMasq);

      aParam.mCanDoMatch =  BornesFonc
                            (
                               aParam.mPxMin,aParam.mPxMax,
                               aImPx,&aImMasq,
                               aNbDecim,aPropEst,aRatio
                            );
      if (! aParam.mCanDoMatch)
         return;
   }

   // Even with "flat" terrain add a minimal inc
   aParam.mPxMin -= mAppli.mIncPxCste;
   aParam.mPxMax += mAppli.mIncPxCste;
   
   // Assure that Inc is at least mIncPxMin
   if ((aParam.mPxMax - aParam.mPxMin) < (2*mAppli.mIncPxMin))
   {
       double aPxMoy = (aParam.mPxMin+aParam.mPxMax) / 2.0;
       aParam.mPxMin = aPxMoy - mAppli.mIncPxMin;
       aParam.mPxMax = aPxMoy + mAppli.mIncPxMin;
   }
   // finaly respect amplitudes max that may be imposed
   if ((aParam.mPxMax - aParam.mPxMin) > aAmplMaxPax )
   {
       double aPxMoy = (aParam.mPxMin+aParam.mPxMax) / 2.0;
       aParam.mPxMin = aPxMoy - aAmplMaxPax/2.0;
       aParam.mPxMax = aPxMoy + aAmplMaxPax/2.0;
   }

   aParam.mBoxUtiIn1 = aParam.mBoxIn1;
   // Now commpute boxes
   SetBoxes12FromPx
   (
      aParam.mBoxIn1,aParam.mBoxIn2,
      aParam.mPxMin,aParam.mPxMax,aParam.mOffsetPx,
      // mAppli.mModePad,&aBoxFile2,aParam.ToDebug()
      mAppli.mModePad,nullptr,aParam.ToDebug()
   );
   /*

   cPt2di aP0 = aParam.mBoxIn1.P0();
   cPt2di aP1 = aParam.mBoxIn1.P1();
   // We compute the Box2 homologous of Box1, taking into account dilatation due to px-intervall
   aParam.mBoxIn2 = cBox2di
                    (
                         cPt2di(aP0.x()+aParam.mPxMin,aP0.y()),
                         cPt2di(aP1.x()+aParam.mPxMax,aP1.y())
                    );
   // Memorize the value, as it was fixed "naturally" by micmac
   aParam.mBoxUtiIn1 = aParam.mBoxIn1;

   {
       cBox2di & aB1 = aParam.mBoxIn1;
       const cBox2di & aB2 = aParam.mBoxIn2;
       if (mAppli.mModePad==eModePaddingEpip::eMPE_PxPos)
       {
           aB1 = cBox2di(aB1.P0(),aB1.P0()+aB2.Sz());
       }
       else if (mAppli.mModePad==eModePaddingEpip::eMPE_PxNeg)
       {
           aB1 = cBox2di(aB1.P1()-aB2.Sz(),aB1.P1());
       }
       else if (mAppli.mModePad==eModePaddingEpip::eMPE_SzEq)
       {
           aB1 = aB2;
       }
    //  We want to be sure that whatever happened before BoxIn1 contain the initial value BoxUtIn1
    //  (may creat bug if not) and also dont change the size now that it fit the possible requirements of deeps methods
    //  Case rare bu posible with px computed from previous steps

        cPt2di  aSzB1 = aB1.Sz();
        const cPt2di & aP0U1 = aParam.mBoxUtiIn1.P0();
        if (aB1.P0().x() > aP0U1.x())
            aB1 = cBox2di(aP0U1,aP0U1+aSzB1);
           
        const cPt2di & aP1U1 = aParam.mBoxUtiIn1.P1();
        if (aB1.P1().x() < aP1U1.x())
            aB1 = cBox2di(aP1U1-aSzB1,aP1U1);

    }
   // Once images loaded in a box, the px will have to be offseted
   aParam.mOffsetPx = aParam.mBoxIn1.P0().x() -aParam.mBoxIn2.P0().x();

   // in genral X2 = X1 +PxMin,  Offset =  X1-X2 = -PxMin , and finally PxMin=0
   // a long comptation for a basic result , but we cannot set it directly like that 
   // because intersection with aBoxFile2 can change it

   aParam.mPxMin = aParam.mBoxIn2.P0().x() + aParam.mOffsetPx;
   aParam.mPxMax = aParam.mBoxIn2.P1().x() + aParam.mOffsetPx;
   */
}


void cOneLevel::Purge()
{
   // For Im1, we must remove Px & Masq one step later as it will be used in next computation
   if (mIm.mIsIm1 && mDownLev!=nullptr)
   {
       RemoveFile(mDownLev->mNamePx,false);
       RemoveFile(mDownLev->mNameMasq,false);
   }
   // At top level, dont remove initial image & masq
   if (mLevel==0)
      return;

   RemoveFile(mNameIm,false);
   if (!mIm.mIsIm1 )
       RemoveFile(mNameMasq,false);
}


/*  ============================================== */
/*                                                 */
/*             cOneIm                              */
/*                                                 */
/*  ============================================== */

cOneIm::cOneIm
(
      cAppli& anAppli,
      const std::string& aNameIm,
      bool IsIm1
) :
   mAppli       (anAppli),
   mIsIm1       (IsIm1),
   mNameIm      (aNameIm),
   mNameMasq    (V1NameMasqOfIm(mNameIm)), // Assume MMV1 convention, to see later
   mPFileImFull (cDataFileIm2D::Create(mAppli.DirProject()+mNameIm,false))
{
}

void cOneIm::CreateLevels(int aNbLevels)
{
   // create from 0 to NbLevls include
   for (int aLev=0 ; aLev <= aNbLevels ; aLev++)
   {
       mLevels.push_back(tPtrLev(new cOneLevel(*this,aLev)));
   }
   // Link the result so that each level "knows" its adjacent level in the pyramid
   for (int aLev=0 ; aLev < aNbLevels ; aLev++)
   {
      LevAt(aLev).mDownLev = &( LevAt(aLev+1));
      LevAt(aLev+1).mUpLev = &( LevAt(aLev));
   }
}

/*  ============================================== */
/*                                                 */
/*              cAppli                             */
/*                                                 */
/*  ============================================== */

cAppli::cAppli
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli(aVArgs,aSpec),
   mSzBasculeMM  (0.5),  // correspond to 2000x2000 pixel
   mRandPaded    (false),
   mSzTile     (2000,1500),
   mSzOverL    (50,30),
   mIncPxProp  (0.05),
   mIncPxCste  (50),  // (PxMax-PxMin)/2
   mIncPxMin   (100),
   mMaxRatio   (4.0),
   mPropEstim  (0.15),
   mNbDecim    (3),
   mDoPyram    (true),
   mDoClip     (true),
   mDoMatch    (true),
   mDoPurge    (true)
{
}


cCollecSpecArg2007 & cAppli::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   
   return 
      anArgObl  
         << Arg2007(mModeMatchFinal,"Matching mode at high resol",{AC_ListVal<eModeEpipMatch>()})
         << Arg2007(mNameIm1,"Name Input Image1",{eTA2007::FileImage})
         << Arg2007(mNameIm2,"Name Input Image1",{eTA2007::FileImage})
   ;
}

cCollecSpecArg2007 & cAppli::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
      anArgOpt
         << AOpt2007(mSzTile,"SzTile","Size of tiling used to split computation",{eTA2007::HDV})
         << AOpt2007(mSzOverL,"SzOverL","Size of overlap between tiles",{eTA2007::HDV})
         << AOpt2007(mModeMatchInit,"MMInit","Matching mode at low resol resol, def=mode high resol",{AC_ListVal<eModeEpipMatch>()})
         << AOpt2007(mSzBasculeMM,"SzBascMM","Sz in MegaPix of transition Init/Final for match mode",{eTA2007::HDV})
         << AOpt2007(mOutPx,CurOP_Out,"Name of Out file, def=Px_+$Im1")
         << AOpt2007(mModePad,"ModePad","Type of padding, default depend of match mode",{AC_ListVal<eModePaddingEpip>()})
         << AOpt2007(mRandPaded,"RandPaded","Generate random value for added pixel")
         // -- Tuning
         << AOpt2007(mDoPyram,"DoPyram","Compute the pyramid",{eTA2007::HDV,eTA2007::Tuning})
         << AOpt2007(mDoClip,"DoClip","Compute the clip of images",{eTA2007::HDV,eTA2007::Tuning})
         << AOpt2007(mDoMatch,"DoMatch","Do the matching",{eTA2007::HDV,eTA2007::Tuning})
         << AOpt2007(mDoPurge,"DoPurge","Do we purge the result ?",{eTA2007::HDV,eTA2007::Tuning})
   ;
}

void cAppli::SetOutPut(std::string & aNamePx)
{
   if (IsInit(&mOutPx))
      aNamePx =  DirProject() + mOutPx;
   else
     mOutPx = aNamePx;
}

std::string cAppli::ComMatch(cParam1Match & aParam) 
{
   switch (mModeMatchCur)
   {
       case  eModeEpipMatch::eMEM_MMV1 :
       {
          return    "mm3d MMTestMMVII"
                 +  BLANK  +  DirTmpOfCmd()
                 +  BLANK  +  aParam.mClipNameIm1
                 +  BLANK  +  aParam.mClipNameIm2
                 +  BLANK  +  "NbP=1"   // Allocate only one process as they are themself parallelized
                 +  BLANK  +  "DirMEC=" + aParam.mClipDirTmp  // 
                 +  BLANK  +  "FileExp=" + aParam.mClipNamePx  // 
          ;
       	  break;
       }
       case eModeEpipMatch::eMEM_PSMNet :
       {
		  std::string aDenseMDir = TopDirMMVII() + "src/DenseMatch/";
          std::string aCom = "bash " + aDenseMDir + "run.sh " 
				             //+ BLANK + "--loadmodel"  + BLANK + mTrainedModel
                             + BLANK + "--leftimg"  + BLANK + DirTmpOfCmd() + aParam.mClipNameIm1
                             + BLANK + "--rightimg" + BLANK + DirTmpOfCmd() + aParam.mClipNameIm2
                             + BLANK + "--result"     + BLANK + DirTmpOfCmd() + aParam.mClipNamePx;
				  
          
		  return aCom;
          break;
       }
       case eModeEpipMatch::eMEM_NoMatch :
       {
             std::string aNamePx = DirTmpOfCmd() + aParam.mClipNamePx;
             cPt2di aSzBox1 =  aParam.mBoxIn1.Sz();
             cPt2di aSzBox2 =  aParam.mBoxIn2.Sz();

             if ((mModePad!=eModePaddingEpip::eMPE_NoPad) && (aSzBox1!=aSzBox2))
             {
                  StdOut() << "EXECUCTE Eq: " <<   aSzBox1 << aSzBox2 << "\n"; getchar();
                  MMVII_INTERNAL_ASSERT_always(false,"Sz of boxes should be equal");
             }


             cDataFileIm2D::Create(aNamePx,eTyNums::eTN_INT1,aParam.mBoxIn1.Sz(),1);

             return "";
       }

       default : break;
   }
   return "";
}



void cAppli::MakePyramid()
{
   if (mDoPyram)
   {
      // Need uper level to compute down level : so Start from highest levels, do it serially
      for (int aLev=0 ; aLev<mNbLevel ; aLev++)
      {
          // In each level, can compute in parallel for both images
          std::list<std::string> aLComReduce;
          for (auto aIm : mIms)
          {
              aLComReduce.push_back(aIm->LevAt(aLev).StrComReduce(true));  // Image
              aLComReduce.push_back(aIm->LevAt(aLev).StrComReduce(false)); // Masq
          }
          ExeComParal(aLComReduce);
      }
   }
   else
   {
       for (int aK=0 ; aK<10 ; aK++)
           StdOut() << "!!!!! Pyramid skeeped !!!!\n";
       getchar();
   }

   // Dont try to paraliz, quite fast and would be tricky (need size of image)
   for (int aLev=0 ; aLev<=mNbLevel ; aLev++)
   {
       Im1().LevAt(aLev).MakeImPx();
   }
}

void  cAppli::MatchOneLevel(int aLevel)
{

     // This value is computed empirically : have good param (not slow down )
     //  and purge sufficiently frequently the files (not fill the HD)
     double aFreqExec = 1/ (10.0 * mNbProcAllowed);

     cOneLevel & aILev1 = Im1().LevAt(aLevel);
     cOneLevel & aILev2 = Im2().LevAt(aLevel);


     // compute match mode that depends of level
     {
          if (aLevel==0)  // whatever happen use final mode at finest level
	  {
	     mModeMatchCur = mModeMatchFinal  ;
	  }
          else if (aLevel==mNbLevel)  // else  whatever happen use initial mode at coursest level
	  {
	     mModeMatchCur = mModeMatchInit  ;
	  }
          else   // else depend of number of pixel 
	  {
             mModeMatchCur =   ((aILev1.BoxIm().NbElem() / 1e6)  < mSzBasculeMM) ?
                               mModeMatchInit                                    :
                               mModeMatchFinal                                   ;
	  }
     }
     //  now compute mode padding that depends of match mode
     if (! IsInit(&mModePad))
     {
	 int aAmplMax = 10000;
         eModePaddingEpip aModePad = eModePaddingEpip::eMPE_NoPad; 
         switch (mModeMatchCur)
         {
            case eModeEpipMatch::eMEM_MMV1    :    
		 aModePad = eModePaddingEpip::eMPE_NoPad; 
            break;

            case eModeEpipMatch::eMEM_PSMNet  :    
	         aModePad = eModePaddingEpip::eMPE_PxNeg; 
	         aAmplMax = 180;
            break;

            case eModeEpipMatch::eMEM_NoMatch :    
	          aModePad = eModePaddingEpip::eMPE_NoPad; 
            break;

            case eModeEpipMatch::eNbVals      :                   
	    break;
         }
         mMaxAmplPxCur = aAmplMax;
         if (! IsInit(&mModePad))
            mModePad = aModePad;
     }


     cBox2di aCurBoxFile1 = aILev1.BoxIm(); // Need it to parse the space
     cBox2di aCurBoxFile2 = aILev2.BoxIm(); // Need it to clip, in Interv,  to size of Im2


     cParseBoxInOut<2> aBoxParser =  cParseBoxInOut<2>::CreateFromSize(aCurBoxFile1,mSzTile);

     std::list<std::string>   aLComClip; // list of commands for clipping images
     std::list<std::string>   aLComMatch; // list of command for match cliped image
     std::list<cParam1Match>  aLParam; // list of parameter of the match
     for (auto anIndex :  aBoxParser.BoxIndex())
     {
        // Create the parameters of match, incomplete at this step
        cParam1Match aParam
                     (
                        aBoxParser.BoxIn(anIndex,mSzOverL),
                        aBoxParser.BoxOut(anIndex),
                        anIndex,aILev1,aILev2
                     );

        // The master level must compute the paralax interval  to complete param
        aILev1.EstimateIntervPx(aParam,aCurBoxFile1,aCurBoxFile2,mMaxAmplPxCur);
        // Now Param ix complete
        aLParam.push_back(aParam);
        if (aParam.mCanDoMatch)
        {
           aLComMatch.push_back(ComMatch(aParam));


           // We must create clipped images for both images and masqs
           aLComClip.push_back(aILev1.StrComClipIm(true ,anIndex,aParam));
           aLComClip.push_back(aILev1.StrComClipIm(false,anIndex,aParam));
           aLComClip.push_back(aILev2.StrComClipIm(true ,anIndex,aParam));
           aLComClip.push_back(aILev2.StrComClipIm(false,anIndex,aParam));
        }

        // At a given frequency we execute the accumulated commands

        if (aBoxParser.BoxIndex().SignalAtFrequence(anIndex,aFreqExec))
        {
            // Create all cliped images
            if (mDoClip)
               ExeComParal(aLComClip);

            // Match  all cliped images
            if (mDoMatch)
               ExeComParal(aLComMatch);

            // Save serially all cliped px in global paralax file
            for (const auto & aParam : aLParam)
            {
                if (aParam.mCanDoMatch)
                   aILev1.SaveGlobPx(aParam);
            }

            // empty all that for next computation not to redo the same stuff
            aLComClip.clear();
            aLComMatch.clear();
            aLParam.clear();

            // All the temporary clip file are no longer needed
            if (mDoPurge)
               RemovePatternFile(DirTmpOfCmd() + "Clip.*tif",false);

        }
     }
     MMVII_INTERNAL_ASSERT_always(aLComClip.empty(),"Bad size management");


     // Remove files non longer necessary
     if (mDoPurge)
     {
        aILev1.Purge();
        aILev2.Purge();
     }

}




int cAppli::Exe()
{
   SetIfNotInit(mModeMatchInit,mModeMatchFinal);
   SetIfNotInit(mDoFileCommands,!mDoPurge);
		   

   // Now the appli is completely initialized, it can be used to create object
   mIms.push_back(tPtrIm (new cOneIm (*this,mNameIm1,true )));
   mIms.push_back(tPtrIm (new cOneIm (*this,mNameIm2,false)));



   // Compute mains numeric values

   cPt2di aSzFull1 = Im1().mPFileImFull.Sz();  // Size of firt full image
   double aRatioTileFile = RatioMax(aSzFull1,mSzTile);  // Ratio between size and tiles
   mNbLevel = round_up(log(aRatioTileFile) / log(mMaxRatio));  // number of level in the pyramid
   mRatioByL = pow(aRatioTileFile,1/double(mNbLevel));  // Ratio between consecutive level

   // In each image create the structure corresponding to levels
   for (auto & aIm : mIms)
      aIm->CreateLevels(mNbLevel);


   // Compute pyramid of images
   MakePyramid();

   for (int aLevel = mNbLevel ; aLevel>=0 ; aLevel--)
   {
        MatchOneLevel(aLevel);
   }

   return EXIT_SUCCESS;
}

cAppliBenchAnswer cAppli::BenchAnswer() const 
{
   return cAppliBenchAnswer(true,1.0);
}

int  cAppli::ExecuteBench(cParamExeBench & aParam) 
{
   // As it is quite long and not randomized, dont do it each time
   if (aParam.Level() != 3) 
      return EXIT_SUCCESS;


   std::string aDirData = InputDirTestMMVII() + "EPIP/Tiny/";
   std::string aCom =    Bin2007
                       + BLANK  + mSpecs.Name()
                       + " MMV1 ImR.tif ImL.tif   Out=PxRL.tif SzTile=[300,200]"
                       + BLANK + GOP_DirProj + "=" + aDirData;

   ExtSysCall(aCom,false);

   double aDif = DifAbsInVal(aDirData+"PxRL.tif",aDirData+"RefPx_RL.tif");


   MMVII_INTERNAL_ASSERT_bench(aDif==0,"DenseMatchEpipGen : result != ref");


   return EXIT_SUCCESS;
}

/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_EpipGenDenseMatch(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli(aVArgs,aSpec));
}

}; //  NS_EpipGenDenseMatch
}


namespace MMVII
{
cSpecMMVII_Appli  TheSpecEpipGenDenseMatch
(
     "DenseMatchEpipGen",
      NS_EpipGenDenseMatch::Alloc_EpipGenDenseMatch,
      "Generik epipolar dense matching",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);

};

