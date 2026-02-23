//#include "MMVII_PCSens.h"

#include "MMVII_Image2D.h"
#include "MMVII_ImageMorphoMath.h"
#include "MMVII_Sensor.h"
#include "MMVII_Ptxd.h"
//#include "MMVII_ImageInfoExtract.h"
//#include "MMVII_TplGradImFilter.h"

//#include "MMVII_ExtractLines.h"
//#include "MMVII_2Include_CSV_Serial_Tpl.h"




namespace MMVII
{


/* =============================================== */
/*                                                 */
/*                 cCurvFrange                     */
/*                                                 */
/* =============================================== */

/**
 * @brief Class for creating/storing a curve in the Frange appli
 */
class cCurvFrange
{
    public :

       /// Constructor, box use to size the curve
       cCurvFrange(const cBox2di &);

       /// Create points from the average,  z is the weighting
       std::vector<cPt3dr>  ExtractCurve() const;

       /// Add a pix for averaging
       void AddPix(const cPt2di & aPix,tREAL8 aW);

       ///  Comparator used for sorting from left to right
       bool operator < (const cCurvFrange& aC2) const;
    private :
       cBox2di              mBox;
       std::vector<tWArr>   mAvX;
};


cCurvFrange::cCurvFrange(const cBox2di & aBox) :
    mBox (aBox),
    mAvX (aBox.P1().y()+1)  // +1 => P1 is include
{
}

//  Compare on P0.x for sorting left to right
bool cCurvFrange::operator < (const cCurvFrange& aC2) const
{
   return  mBox.P0().x() < aC2.mBox.P0().x();
}

// Accumulate a pixel in averaging
void cCurvFrange::AddPix(const cPt2di & aPix,tREAL8 aW)
{
   mAvX.at(aPix.y()).Add(aW,aPix.x());
}


// Extract average point with no null weighting
std::vector<cPt3dr>  cCurvFrange::ExtractCurve() const
{
   std::vector<cPt3dr> aRes;
   for (size_t anY =0 ; anY<mAvX.size() ; anY++)
   {
       const auto & anAv = mAvX.at(anY);
       if (anAv.SW() != 0)
       {
           aRes.push_back(cPt3dr(anAv.Average(),anY,anAv.SW()));
       }
   }
   return aRes;
}

/* =============================================== */
/*                                                 */
/*                 cAppliFranges                   */
/*                                                 */
/* =============================================== */


/**  An application for  extaction curves on interference images
 *   rather very specific...
 */


class cAppliFranges : public cMMVII_Appli
{
     public :
        cAppliFranges(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
     private :
        //================= typedef part ===============================
        typedef tREAL4            tElIm;
        typedef cIm2D<tElIm >     tIm;
        typedef cDataIm2D<tElIm > tDIm;

        typedef cIm2D<tU_INT1 >    tImLabel;
        typedef cDataIm2D<tU_INT1> tDImLabel;

        //================================================================
        //       METHODS DECLARATION
        //================================================================

            //------------------------- overidding cMMVII_Appli ----------------
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
         std::vector<std::string>  Samples() const override;
        virtual ~cAppliFranges();

            //------------------------- Specific method----------------
        ///  Do all the job for one image
        void  DoOneImage(const std::string & aNameIm) ;

        /// Do the job for one connected component
        void  AnalyseOneConnectedComp(const cPt2di& aPix);
        /// Validate a connected component, decision made on bounding box for now
        bool  ConnectedCompIsValide(const std::vector<cPt2di> & aVPts,const cBox2di & aBox);
        ///  Create a new curve once the CC is valid
        void  CreateNewCurve(const std::vector<cPt2di> & aVPts,const cBox2di&);
        /// convert a gray level in a weigthing inside [0,1]
        tREAL8 WeightOfGray(tREAL8 aGray) const;

        /// Generate visualisation if required
        void MakeVisu();

        void ReadImage(const cBox2di& aBox);

        //================================================================
        //       DATA DEFINITION
        //================================================================

        /// == Used for file naming, even if obviously not a photogrammetric context ========
        cPhotogrammetricProject  mPhProj;

        // ----------- Mandatory Args -----------
        std::string              mPatImage; /// Pattern of all images

        // -----------  Optionnal args  -----------

        cPt2di                   mIntY;    ///<  Y Interval of image
        cPt2dr                   mSigma;   ///<  Sigma of smoothing
        tREAL8                   mSigCurv;  ///< Sigma for comutig Cuvres
        int                      mNbIter;  ///<  Number of iteration of smoothing

        tREAL8                   mWidhMinAll;    ///< Witdh Min of Connected compenent
        tREAL8                   mWidhMinBorder; ///< Idem but for border image
        bool                     mDoVisu;        ///< Do we generate visualisation
        std::vector<std::string> mParamStack;    ///< Param for image stacking if any

        //  ----------- Variables for images -----------
        std::string              mNameCurIm;   ///< Name of current image
        cPt2di                   mCurSz;       ///< Size of current image
        tIm                      mCurIm;       ///< Loaded current image
        tDIm*                    mDCurIm;      ///< Data current im
        tIm                      mImSmooth;    ///< Smoothed image
        tDIm*                    mDImSmooth;   ///< Data smoothed
        tImLabel                 mImLabel;     ///< Label image
        tDImLabel*               mDImLabel;    ///< Data label

        //  ----------- miscellaneous -----------

        cTimerSegm               mTimeSeg;   ///< Used for timing info
        int                      mMarqCurCC; ///< Counter for connected component
        tREAL8                   mThreshW;   ///< Trehshold for white
        std::vector<cCurvFrange> mCurves;    ///< Result of curves extraction
        std::string              mIdExport;  ///< Identiant for Export (CSV/Tiff ...)
}; 

    /* =================================================== */
    /*      Overiding of  cMMVII_Appli                     */
    /* =================================================== */


cAppliFranges::cAppliFranges(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mPhProj           (*this),

    // -------- Defautl values of opt param ----
    mIntY             (900,1280),
    mSigma            (30.0,5.0),
    mNbIter           (5),
    mWidhMinAll       (200.0),
    mWidhMinBorder    (300.0),
    mDoVisu           (false),
    mParamStack       {},
    //----------  Mandatory init of images --------
    mCurIm            (cPt2di(1,1)),
    mImSmooth         (cPt2di(1,1)),
    mImLabel          (cPt2di(1,1)),
    mTimeSeg          (this)

{
    mCurves.reserve(20);  // to save memory
}

cAppliFranges::~cAppliFranges()
{
}

cCollecSpecArg2007 & cAppliFranges::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return    anArgObl
            <<  Arg2007(mPatImage,"Name of input Image", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
      ;
}

cCollecSpecArg2007 & cAppliFranges::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
            << AOpt2007(mSigma,"Sigma","Sima x/y of initial smoothig for CC",{eTA2007::HDV})
            << AOpt2007(mSigCurv,"SigCurv","Sima for smoothig curve",{eTA2007::HDV})
            << AOpt2007(mIntY,"IntY","Interval for Y",{eTA2007::HDV})
            << AOpt2007(mNbIter,"NbIt","Number of iter initial smoothing",{eTA2007::HDV})
            << AOpt2007(mDoVisu,"DoVisu","Generate Visualisation ?",{eTA2007::HDV})
            << AOpt2007(mWidhMinAll,"MinWidth","Minimal witdh, general case",{eTA2007::HDV})
            << AOpt2007(mWidhMinBorder,"BorderMinWidth","Minimal witdh for border",{eTA2007::HDV})
            << AOpt2007(mParamStack,"Stack","Stacking parm [Pat,Nb,Mode] ",{{eTA2007::ISizeV,"[3,3]"}})
     ;
}


std::vector<std::string>  cAppliFranges::Samples() const
{
   return
   {
       "MMVII ExtractFranges Retiga_000000105.tif Sigma=[10,2] DoVisu=1",
       "MMVII ExtractFranges Retiga_.*.tif"
   };
}

int cAppliFranges::Exe()
{
    mPhProj.FinishInit();

    if (RunMultiSet(0,0)) // Case several images in //
    {
       return ResultMultiSet();
    }
    DoOneImage(UniqueStr(0)); // Case 1 image (may be recalled from multiple)
    return EXIT_SUCCESS;
}

/* =================================================== */
/*              Specific functions                     */
/* =================================================== */

// Linear ressample of [Thr,255]  in [0,1]
tREAL8 cAppliFranges::WeightOfGray(tREAL8 aGray) const
{
   return  std::min(1.0,std::max(0.0,(aGray-mThreshW) /(255.0-mThreshW)));
}


bool cAppliFranges::ConnectedCompIsValide(const std::vector<cPt2di> & aVPts,const cBox2di & aBox)
{
    // CC touches  extrem left, not full -> false
    if (aBox.P0().x() <= 10) return false;

    if (aBox.Sz().x() < mWidhMinAll)  // not width enough
        return false;

    // more strict width rule for CC touching the right
    if (aBox.P1().x() >=  mCurSz.x()-10)
    {
        if  (aBox.Sz().x() < mWidhMinBorder)
            return false;
    }

    // CC must fill the entire hight
    if (aBox.P0().y()!= 1) return false;
    if (aBox.P1().y()!= mCurSz.y()-2) return false;

    // So far, so good ...
    return true;
}

void cAppliFranges::CreateNewCurve(const std::vector<cPt2di> & aVPts,const cBox2di& aBox)
{
    mCurves.push_back(cCurvFrange(aBox));  // add an empty curve
    cCurvFrange & aCurv = mCurves.back();  // reference it

    // Parse all the pixel to add them in curve
    for (const auto & aPix : aVPts)
    {
        aCurv.AddPix(aPix, WeightOfGray(mDImSmooth->GetV(aPix)));
    }
}


void  cAppliFranges::AnalyseOneConnectedComp(const cPt2di& aPix)
{
   std::vector<cPt2di> aVPts;
   // Call the method of lib to push CC in aVPts
   ConnectedComponent
   (
       aVPts ,
       *mDImLabel,
       Alloc8Neighbourhood(),
       aPix, // seed
       255,   // CC of point=255
       mMarqCurCC // Value set in point selected
   );

   // Create new box, possible for example all point have same x => Allow Empty
   cBox2di aBox = cBox2di::FromVect(aVPts,eAllowEmpty::Yes);

   if (ConnectedCompIsValide(aVPts,aBox)) // are Vpt/Box OK
   {
       CreateNewCurve(aVPts,aBox);  // is ok, create curve
       mMarqCurCC++;       // to have different label on next CC
   }
   else
   {
       mDImLabel->VPtsSetV(aVPts,1); // put 1 as label for rejected CC
   }
}

void cAppliFranges::ReadImage(const cBox2di& aBox )
{
   // if no stack, basic read
   if (!IsInit(&mParamStack))
   {
       mCurIm = tIm::FromFile(mNameCurIm,aBox);
       return;
   }

   // --- read parameters of mParamStack = [Pat,NbIm,Mode] -------
   std::string aPat = mParamStack.at(0);
   int aNbImRequired = cStrIO<int>::FromStr(mParamStack.at(1));
   int aMode = cStrIO<int>::FromStr(mParamStack.at(2));

   // --- read Names, sort and extract index of CurName --------------
   std::vector<std::string> aVName =  GetFilesFromDir(DirProject(),AllocRegex(aPat)); // read All Names
   std::sort(aVName.begin(),aVName.end()); // sort name
   const auto & anIter =  std::find(aVName.begin(),aVName.end(),mNameCurIm); // extract index of name
   if (anIter==aVName.end())
   {
       MMVII_WARNING(" Image not found in stacking, use standard read \n");
       mCurIm = tIm::FromFile(mNameCurIm,aBox);
       return;
   }
   int aKC = anIter - aVName.begin();

   // --- Compute correct interval of index of image, warantee to be inside aVName
   int aK0 = std::max(0,aKC-aNbImRequired);
   int aK1 = std::min(int(aVName.size())-1,aKC+aNbImRequired);
   int aNbImUsed = std::min(aKC-aK0,aK1-aKC);
   aK0 = aKC-aNbImUsed;
   aK1 = aKC+aNbImUsed;

  // ------- Read Images ------------------
   std::vector<tIm> aVIm;
   for (int aKIm=aK0 ; aKIm<=aK1; aKIm++)
   {
       aVIm.push_back(tIm::FromFile(aVName.at(aKIm),aBox));
   }
   mCurIm = tIm(mCurSz);


   // ------- Compute weigthing of index ------------------
   std::vector<tREAL8> aVW;
   for (size_t aKV=0 ; aKV<aVIm.size() ; aKV++)
   {
       tREAL8 aW=1.0;
       tREAL8 aRnk = (aKV+0.5) / aVIm.size();
       if (aMode==1)
           aW =  (aKV == (aVIm.size()/2)); // Median
       else if (aMode==2)
       {
           tREAL8 aTeta = aRnk * 2 * M_PI;
           aW = 1+std::cos(aTeta +M_PI);
       }
       else if (aMode==3)
       {
           aW = 0.5 - std::abs(aRnk-0.5);
       }
       aVW.push_back(aW);
   }

   //--- Compute weighted average --------------------------
   for (const auto & aPix : mCurIm.DIm())
   {
       std::vector<tREAL8> aVVals;
       for (const auto & anIm : aVIm )
           aVVals.push_back(anIm.DIm().GetV(aPix));
       std::sort(aVVals.begin(),aVVals.end());
       cWeightAv<tREAL8,tREAL8> aWAv;

       for (size_t aKV=0 ; aKV<aVVals.size() ; aKV++)
       {
           aWAv.Add(aVW.at(aKV),aVVals.at(aKV));
       }
       mCurIm.DIm().SetV(aPix,aWAv.Average());
   }
/*
   StdOut() << " STACKKK " << aVName.size()
            << " N0=" << aVName.at(0)
            << " KF=" << aKC
            << " NBI=" << aNbImUsed
            << " MODE=" << aMode
            << " W " << aVW
            << "\n";
   getchar();
*/
}

void  cAppliFranges::DoOneImage(const std::string & aNameIm)
{
    mNameCurIm = aNameIm;
    mIdExport = "Frange-MMVII-" + Prefix(mNameCurIm);
    InitReportCSV (mIdExport,"csv",false,{"Label","X","Y","Weight"});

    //  ============== Create & read the images
    cDataFileIm2D aDataFI2D = cDataFileIm2D::Create(mNameCurIm,eForceGray::No);
    cBox2di aBox(cPt2di(0,mIntY.x()), cPt2di(aDataFI2D.Sz().x(),mIntY.y()));
   // mCurIm = tIm::FromFile(mNameCurIm,aBox);
    mCurSz =  aBox.Sz();
    ReadImage(aBox);
    mDCurIm = & mCurIm.DIm();

    mImSmooth = mCurIm.Dup();
    mDImSmooth= &mImSmooth.DIm();
    mImLabel = tImLabel(mCurSz);
    mDImLabel = &mImLabel.DIm();

    // =========== create smoothing of images ==========================
    tREAL8 aFactExpX = FactExpFromSigma2(Square(mSigma.x())/mNbIter);
    tREAL8 aFactExpY = FactExpFromSigma2(Square(mSigma.y())/mNbIter);
    ExponentialFilter(true,*mDImSmooth,mNbIter,*mDImSmooth,aFactExpX,aFactExpY);

    // =================== init labeling by thresholding =========
    mThreshW = mDCurIm->MoyVal();
    for ( auto& aPix : *mDImLabel )
    {
        bool isOver = mDImSmooth->GetV(aPix)>mThreshW;
        mDImLabel->SetV(aPix,isOver ? 255 : 0);
    }
    mDImLabel->InitBorder(0); // Current precaution to avoid image outside

    // ============== Now Put in mDImSmooth , Convol on Y, or Original
    mDCurIm->DupIn(*mDImSmooth);
    if (IsInit(&mSigCurv))
    {
        tREAL8 aFactExpY = FactExpFromSigma2(Square(mSigCurv)/mNbIter);
        ExponentialFilter(true,*mDImSmooth,mNbIter,*mDImSmooth,0,aFactExpY);
    }



    // ====   Create the curves of connected component analysis =====
    mMarqCurCC = 2;
    for (const auto& aPix : *mDImLabel )
    {
         if (mDImLabel->GetV(aPix) == 255)
             AnalyseOneConnectedComp(aPix);
    }
    // sort the curve to have them from left to right, using operator <
    std::sort (mCurves.begin(),mCurves.end());

    // ===============  Generate the export =====================

    // Generate the csv
    for (size_t aKC=0 ; aKC<mCurves.size() ; aKC++)
    {
        for (const auto aPt : mCurves.at(aKC).ExtractCurve())
        {
            AddOneReportCSV(mIdExport,{ToStr(aKC),ToStr(aPt.x()),ToStr(aPt.y()),ToStr(aPt.z())});
        }
    }

    // Generate the visualisation if required
    if (mDoVisu)
       MakeVisu();
}


void cAppliFranges::MakeVisu()
{
    std::string aPref = mPhProj.DirVisuAppli() + mIdExport ;

    mDImSmooth->ToFile(aPref+"-Smooth.tif");
    mDImLabel->ToFile(aPref+"-Label.tif");

    cRGBImage aIRGB(mCurSz);

    tIm aImGray(mCurSz);
    for (const auto& aPix : *mDImLabel)
    {
        tREAL8 aVal = 0;
        if (mDImLabel->GetV(aPix) >1)
        {
           aVal = WeightOfGray(mDImSmooth->GetV(aPix)) * 255;
        }
        aImGray.DIm().SetV(aPix,aVal);
        aIRGB.SetGrayPix(aPix,aVal);
    }
    for (const auto & aFr : mCurves)
    {
        for (const auto & aPt : aFr.ExtractCurve())
        {
             aIRGB.SetRGBPix(ToI(Proj(aPt)),cRGBImage::Red);
        }
    }

    aIRGB.ToFile(aPref+"-Curves.tif");
    aImGray.DIm().ToFile(aPref+"-GrayLab.tif");
}






tMMVII_UnikPApli Alloc_AppliFranges(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliFranges(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecAppliFranges
(
     "ExtractFranges",
      Alloc_AppliFranges,
      "Extraction of Franges",
      {eApF::ImProc},
      {eApDT::Image},
      {eApDT::Console},
      __FILE__
);

#if (0)
#endif

};
