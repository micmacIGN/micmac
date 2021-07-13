#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{




template <class Type> class  cTplAppliCalcDescPCar;
class cAppliCalcDescPCar;

///  Class to manage template INT/REAL in cMMVII_Appli
/**
    To avoid code duplication, only one cAppliCalcDescPCar exist,
    code that require specialisation is done in cTplAppliCalcDescPCar
*/

template <class Type> class  cTplAppliCalcDescPCar
{
    public :
        typedef cIm2D<Type>            tIm;
        typedef cGP_OneImage<Type>     tGPIm;
        typedef cGP_OneOctave<Type>    tOct;
        typedef std::shared_ptr<tOct>  tSP_Oct;
        typedef cGaussianPyramid<Type> tPyr;
        typedef std::shared_ptr<tPyr>  tSP_Pyr;



        cTplAppliCalcDescPCar(cAppliCalcDescPCar & anAppli);

        /// Run all the process
        void ExeGlob();
    private :
        /// Run the process for one image box
        void ExeOneBox(const cPt2di & aP,const cParseBoxInOut<2> & aPBI);
        tREAL4 ToStored(const tREAL4 aV)   {return (aV-mVC) *mDyn;}
        tREAL4 FromStored(const tREAL4 aV) {return aV/mDyn + mVC;}

        cAppliCalcDescPCar & mAppli;   ///< Reference the application
        bool                 mIsFloat; ///< Known from type
        tSP_Pyr              mPyr;     ///< Pointer on gaussian pyramid
        tSP_Pyr              mPyrLapl;     ///< Pointer to laplacian pyramids
        tSP_Pyr              mPyrCorner;   ///< Pointer to Corner pyramids
        tSP_Pyr              mPyrOriNom;   ///< Pointer to Normalized original pyram
        cDataFileIm2D        mDFI;     ///< Structure on image file
        tREAL4               mTargAmpl;///< Amplitude after dynamic adaptation
        cRect2               mBoxIn;   ///< Current Input Box
        cPt2di               mSzIn;    ///< Current size of inputs
        cRect2               mBoxOut;  ///< Current Output Box, inside wich we muste save data
        int                  mNbTiles;  ///< Number of tiles, usefull to know if we need to merge at end

        // To adapt dynamic  StoredVal = (RealVal-mVC) / mDyn
        tREAL4               mVC;      ///< Central Value
        tREAL4               mDyn;     ///< Dynamic of value
};


///  Application class for computation of Aime Descritor
/** 
*/

class cAppliCalcDescPCar : public cMMVII_Appli
{
     public :
        friend  class cTplAppliCalcDescPCar<tREAL4>;
        friend  class cTplAppliCalcDescPCar<tINT2>;

        cAppliCalcDescPCar(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        ~cAppliCalcDescPCar(); ///< Because of bench
   
     private :
        std::string               mNameIm;                ///< Input image pattern/name 
        bool        mIntPyram;  ///< Impose integer  image
        int         mSzTile;    ///<  sz of tiles for spliting in sub processe
        int         mOverlap;   ///< sz of  overlap between tiles
        int         mNbOct;
        int         mSzMinOct;  ///< Avoid too small octave who generate bugs
        int         mNbLevByOct;
        int         mNbOverLapByO;
        bool        mSaveIms;
        bool        mDoLapl;
        bool        mDoCorner;
        bool        mDoOriNorm;
        double      mEstSI0; // Estimation of Sigma of first image
        double      mSDON;  ///< Scale 4 Orig Normalized
        double      mCI0;   ///<  Convol Im0
        double      mCC0;   ///<  Convol Corner0
        std::string mPrefixOut; ///< Prefix 4 Out, is constructed from Image and CarPOut (inherited from Appli)
        cFilterPCar          mFPC;
};

/* =============================================== */
/*                                                 */
/*            cTplAppliCalcDescPCar<Type>          */
/*                                                 */
/* =============================================== */

template<class Type> cTplAppliCalcDescPCar<Type>::cTplAppliCalcDescPCar(cAppliCalcDescPCar & anAppli) :
   mAppli     (anAppli),
   mIsFloat   (! tElemNumTrait<Type>::IsInt()),
   mPyr       (nullptr),
   mPyrLapl   (nullptr),
   mPyrOriNom (nullptr),
   mDFI       (cDataFileIm2D::Create(mAppli.mNameIm,true)),
   mTargAmpl  (10000.0),  // Not to high => else overflow in gaussian before normalisation
   mBoxIn     (cRect2::TheEmptyBox),
   mBoxOut    (cRect2::TheEmptyBox)
{
   if (! anAppli.IsInit(&anAppli.mNbOct))
   {
      double aRatioLarg = MinAbsCoord(mDFI.Sz()) / double(anAppli.mSzMinOct) ;
      double aLog2Larg = std::log(aRatioLarg) / std::log(2);

      anAppli.mNbOct = std::min(anAppli.mNbOct,round_down(aLog2Larg));
   }
}

template<class Type> void cTplAppliCalcDescPCar<Type>::ExeGlob()
{
   cParseBoxInOut<2> aPBI = cParseBoxInOut<2>::CreateFromSizeCste(cRect2(cPt2di(0,0),mDFI.Sz()),mAppli.mSzTile);
   mNbTiles = aPBI.BoxIndex().NbElem();
   MMVII_INTERNAL_ASSERT_always(mNbTiles==1,"Merge of Tiling  to do ...");

   for (const auto & anIndex : aPBI.BoxIndex())
   {
       ExeOneBox(anIndex,aPBI);
   }
}


template<class Type>  void cTplAppliCalcDescPCar<Type>::ExeOneBox(const cPt2di & anIndex,const cParseBoxInOut<2>& aPBI)
{
    // std::string aPref = "Tile"+ToStr(anIndex.x())+ToStr(anIndex.y()) ;
    // Initialize Box, Params, gaussian pyramid
    mBoxIn = aPBI.BoxIn(anIndex,mAppli.mOverlap);
    mSzIn = mBoxIn.Sz();
    mBoxOut = aPBI.BoxOut(anIndex);
    cGP_Params aGP(mSzIn,mAppli.mNbOct,mAppli.mNbLevByOct,mAppli.mNbOverLapByO,&mAppli,true);

    // Value cPt2di(-1,-1) : special value indicating that tiles must not be written at end
    aGP.mNumTile    = (mNbTiles==1) ? cPt2di(-1,-1)  : anIndex;
    // aGP.mPrefixSave = mAppli.mPrefixOut ;
    aGP.mScaleDirOrig = mAppli.mSDON ;
    aGP.mConvolIm0 = mAppli.mCI0 ;
    aGP.mConvolC0  = mAppli.mCC0 ;

    
    if ( mAppli.IsInit(&mAppli.mEstSI0))
       aGP.mEstimSigmInitIm0 =  mAppli.mEstSI0;
    aGP.mFPC  = mAppli.mFPC;

     
    mPyr = tPyr::Alloc(aGP,mAppli.mNameIm,mBoxIn,mBoxOut);

    // Load image
    
    mVC = 0.0;
    mDyn = 1.0;
    if (mIsFloat)  // Case floating point image, just read the values
    {
        mPyr->ImTop().Read(mDFI,mBoxIn.P0());
    }
    else  // Else read the value, then adapt dynamic
    {
         cIm2D<tREAL4>  aImBuf(mSzIn); // Create a float image
         cDataIm2D<tREAL4> & aDImBuf = aImBuf.DIm(); // Extract 
         aDImBuf.Read(mDFI,mBoxIn.P0());  // Read image the file

         tREAL4  aVMin,aVMax;
         GetBounds(aVMin,aVMax,aDImBuf);   // Extract bounds of image
         mVC = (aVMin+aVMax) / 2.0;  // Central value
         if (aVMin != aVMax)  // Set dyn if possible, else will be 1.0
         {
              mDyn =  mTargAmpl / double(aVMax-aVMin);
         }
         cIm2D<Type>      aImTop  = mPyr->ImTop();
         cDataIm2D<Type>& aDImTop = aImTop.DIm();
         // Copy in top image of pyramid, the value adaptated to dynamic
         for (const auto & aP : aDImTop)
         {
// StdOut() << "ppppP " << aP << aDImBuf.GetV(aP) << " " << ToStored(aDImBuf.GetV(aP)) << "\n";
             aDImTop.SetV(aP,ToStored(aDImBuf.GetV(aP)));
         }
    }
    StdOut() << "AnIndex " << anIndex << " SsZzz " << mVC << " " << mDyn << " " << mIsFloat << "\n";


    // Compute Gaussian pyramid
    {
       cAutoTimerSegm aATS("ImGaussPyr");
       mPyr->ComputGaussianFilter();
    }
    if (mAppli.mSaveIms)
    {
       
       StdOut() << "   ############################################ \n";
       StdOut() << "   ######   NAME="  <<    mAppli.mNameIm  << "\n";
       StdOut() << "   ############################################ \n";
    }
    mPyr->SaveInFile(0,mAppli.mSaveIms);

    // Compute Normalized Original Image required
    if (mAppli.mDoOriNorm)
    {
       {
          cAutoTimerSegm aATS("ImOriNorm");
          mPyrOriNom =  mPyr->PyramOrigNormalize();
       }
       mPyrOriNom->SaveInFile(0,mAppli.mSaveIms);
    }

    // Compute Lapl by diff of gauss if required
    if (mAppli.mDoLapl)
    {
       {
          cAutoTimerSegm aATS("ImDifLapl");
          mPyrLapl =  mPyr->PyramDiff();
       }
       mPyrLapl->SaveInFile(0,mAppli.mSaveIms);
    }

    // Compute corner images required
    if (mAppli.mDoCorner)
    {
       {
          cAutoTimerSegm aATS("ImCorner");
          mPyrCorner =  mPyr->PyramCorner();
       }
       mPyrCorner->SaveInFile(0,mAppli.mSaveIms);
    }


    {  // Show times 
       // const std::vector<double>&  aVT = cMMVII_Ap_CPU::TimeSegm().Times();
       // mAppli.TimeSegm().Show();
    }
}


//const cPt2di & aSzIm0,int aNbOct,int aNbLevByOct,int aOverlap);

/* =============================================== */
/*                                                 */
/*               cAppliCalcDescPCar                */
/*                                                 */
/* =============================================== */

cAppliCalcDescPCar::~cAppliCalcDescPCar()
{
   // Because of bench case, else do not success in X::~X() of mFPC
   mFPC.FinishAC(0.05);
}

cAppliCalcDescPCar:: cAppliCalcDescPCar(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli  (aVArgs,aSpec,{eSharedPO::eSPO_CarPO}),
  mIntPyram     (false),
  mSzTile       (7000),
  mOverlap      (300),
  mNbOct        (7),
  mSzMinOct     (20),
  mNbLevByOct   (5),
  mNbOverLapByO (3),
  mSaveIms      (false),
  mDoLapl       (true),
  mDoCorner     (true),
  mDoOriNorm    (true),
  mSDON         (20.0),
  mCI0          (0.7),
  mCC0          (0.7),
  mFPC          (true) // 4 TieP
{
}

cCollecSpecArg2007 & cAppliCalcDescPCar::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameIm,"Name of input file",{{eTA2007::MPatFile,"0"}})
   ;
}

cCollecSpecArg2007 & cAppliCalcDescPCar::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
       << AOpt2007(mIntPyram,"IntPyr","Gauss Pyramid in integer",{eTA2007::HDV})
       << AOpt2007(mSzTile,"TileSz","Size of tile for spliting computation",{eTA2007::HDV})
       << AOpt2007(mOverlap,"TileOL","Overlao of tile to limit sides effects",{eTA2007::HDV})
       << AOpt2007(mNbOct,"PyrNbO","Number of octaves in Pyramid",{eTA2007::HDV})
       << AOpt2007(mSzMinOct,"SzMinOct","Minimal size for an octave",{eTA2007::HDV})
       << AOpt2007(mNbLevByOct,"PyrNbL","Number of level/Octaves in Pyramid",{eTA2007::HDV})
       << AOpt2007(mNbOverLapByO,"PyrNbOverL","Number of overlap  in Pyram(change only for Save Image)",{eTA2007::HDV})
       << AOpt2007(mSaveIms,"SaveIms","Save images (tuning/debuging/teaching)",{eTA2007::HDV})
       << AOpt2007(mSDON,"SON","Scale Orig Normalized",{eTA2007::HDV})
       << AOpt2007(mCI0,"ConvI0","Convolution top image",{eTA2007::HDV})
       << AOpt2007(mCC0,"ConvC0","Additional Corner Convolution",{eTA2007::HDV})
       << AOpt2007(mDoOriNorm,"DON","Do Original Normalized images, experimental",{eTA2007::HDV})
       << AOpt2007(mDoCorner,"DOC","Do corner images",{eTA2007::HDV})
       << AOpt2007(mEstSI0,"ESI0","Estimation of sigma of first image, by default suppose a well sampled image")
       << AOpt2007(mFPC.AutoC(),"AC","Param 4 AutoCorrel [Val,?LowVal,?LowValIntCor]",{eTA2007::HDV,{eTA2007::ISizeV,"[1,3]"}})
       << AOpt2007(mFPC.PSF(),"PSF","Param 4 Spatial Filtering [Dist,MulRay,PropNoFS]",{eTA2007::HDV,{eTA2007::ISizeV,"[3,3]"}})
       << AOpt2007(mFPC.EQsf(),"EQ","Exposant 4  Quality [AutoC,Var,Scale]",{eTA2007::HDV,{eTA2007::ISizeV,"[3,3]"}})
       << AOpt2007(mFPC.LPCirc(),"LPC","Circles of Log Pol [Rho0,DeltaI0,DeltaIm]",{eTA2007::HDV,{eTA2007::ISizeV,"[3,3]"}})
       << AOpt2007(mFPC.LPSample(),"LPS","Sampling Log Pol [NbTeta,NbRho,Mult,Census]",{eTA2007::HDV,{eTA2007::ISizeV,"[4,4]"}})
   ;
}

int cAppliCalcDescPCar::Exe() 
{
   mFPC.FinishAC(0.05);
   mFPC.Check();

   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself
      return ResultMultiSet();
/*
   {
      const std::vector<std::string> &  aVSetIm = VectMainSet(0);

      if (aVSetIm.size() != 1)  // Multiple image, run in parall 
      {
         ExeMultiAutoRecallMMVII("0",aVSetIm); // Recall with substitute recall itself
         return EXIT_SUCCESS;
      }
   }
*/
   CreateDirectories(PrefixPCar(mNameIm,""),true);
   mPrefixOut = PrefixPCarOut(mNameIm);

   // Single image, do the job ...

   if (mIntPyram)  // Case integer pyramids
   {
/*
      cTplAppliCalcDescPCar<tINT2> aTplA(*this);
      aTplA.ExeGlob();
*/
   }
   else   // Case floating points pyramids
   {
      cTplAppliCalcDescPCar<tREAL4> aTplA(*this);
      aTplA.ExeGlob();
   }
   
   return EXIT_SUCCESS;
}



/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_CalcDescPCar(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCalcDescPCar(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCalcDescPCar
(
     "TieP-AimePCar",
      Alloc_CalcDescPCar,
      "Compute caracteristic points and descriptors, using Aime method",
      {eApF::TieP,eApF::ImProc},
      {eApDT::Image},
      {eApDT::PCar},
      __FILE__
);




};
