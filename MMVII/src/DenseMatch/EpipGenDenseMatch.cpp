#include "include/MMVII_all.h"

namespace MMVII
{


/*  ============================================== */
/*                                                 */
/*                                                 */
/*  ============================================== */

class cAppli_EpipGenDenseMatch : public cMMVII_Appli
{
     public :
        cAppli_EpipGenDenseMatch(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
        void MakePyramid();
        void MatchOneLevel(int aLevel);

        // Compute name
        std::string NameTopIm(bool Im1);  ///< Name of Im initial, at top of pyram
        std::string NameImOfLevel(int aLevel,bool Im1); ///< Name of image at any level of pyram

            // Mandatory args
        std::string  mNameIm1;  ///< Name first image
        std::string  mNameIm2;  ///< Name second image

            // Optional args
        cPt2di         mSzTile;   ///< Size of tiles for matching
        double         mMaxRatio; ///< Maximal ratio between to level of pyramid

            // Optional tuning args
        bool           mDoPyram;

            // Computed values & auxilary methods on scales, level ...
        int mNbLevel ;  // Number of level in the pyram [0 mNbLevel]  , 0 = initial image
        double mRatioByL ; // Scale between 2 successive levels

            // Files, use ptr because creation must be deferred in "Exe"
        std::shared_ptr<cDataFileIm2D>  mPFileIm1; ///< Data for File of name mNameIm1
        std::shared_ptr<cDataFileIm2D>  mPFileIm2; ///< Data for File of name mNameIm2
};


cAppli_EpipGenDenseMatch::cAppli_EpipGenDenseMatch
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli(aVArgs,aSpec),
   mSzTile     (1200,800),
   mMaxRatio   (4.0),
   mDoPyram    (false)
{
}


cCollecSpecArg2007 & cAppli_EpipGenDenseMatch::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
         << Arg2007(mNameIm1,"Name Input Image1",{eTA2007::FileImage})
         << Arg2007(mNameIm2,"Name Input Image1",{eTA2007::FileImage})
   ;
}

cCollecSpecArg2007 & cAppli_EpipGenDenseMatch::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
      anArgOpt
         << AOpt2007(mSzTile,"SzTile","Size of tiling used to split computation",{eTA2007::HDV})
         // -- Tuning
         << AOpt2007(mDoPyram,"DoPyram","SzTile",{eTA2007::HDV,eTA2007::Tuning})
   ;
}

std::string cAppli_EpipGenDenseMatch::NameTopIm(bool Im1)
{
   return Im1 ? mNameIm1 : mNameIm2;
}


std::string cAppli_EpipGenDenseMatch::NameImOfLevel(int aLevel,bool Im1)
{
   if (aLevel==0) return DirProject() + NameTopIm(Im1);
   return DirTmpOfCmd() + "Lev" + ToStr(aLevel) + "_" + NameTopIm(Im1);
}

void cAppli_EpipGenDenseMatch::MakePyramid()
{
   if (mDoPyram)
   {
      for (int aLev=0 ; aLev<mNbLevel ; aLev++)
      {
          // Make in parall the 2 images for a given level
          std::list<std::string> aLComReduce;
          for (int aNumIm=0 ; aNumIm<2 ; aNumIm++)
          {
              std::string aComReduce =     "mm3d ScaleIm"
                                         + BLANK + NameImOfLevel(aLev,true) 
                                         + BLANK + ToStr(mRatioByL) 
                                         + BLANK + std::string("Out=")  + NameImOfLevel(aLev+1,true) 
                                     ;
              aLComReduce.push_back(aComReduce);
          }
          ExeComParal(aLComReduce);
      }
   }
   else
   {
       for (int aK=0 ; aK<10 ; aK++)
           std::cout << "!!!!! Pyramid skeeped !!!!\n";
       getchar();
   }
}

void  cAppli_EpipGenDenseMatch::MatchOneLevel(int aLevel)
{
     std::string aCurNameI1 = NameImOfLevel(aLevel,true);
     cDataFileIm2D  aCurFileIm1 = cDataFileIm2D::Create(aCurNameI1,false);
     cPt2di aCurSz1 = aCurFileIm1.Sz();


     cParseBoxInOut<2> aPB =  cParseBoxInOut<2>::CreateFromSize(cBox2di(cPt2di(0,0),aCurSz1),mSzTile);
     for (auto anInd :  aPB.BoxIndex())
        std::cout << anInd << "\n";
}


int cAppli_EpipGenDenseMatch::Exe()
{
   mPFileIm1.reset(new cDataFileIm2D (cDataFileIm2D::Create(DirProject()+mNameIm1,false)));
   mPFileIm2.reset(new cDataFileIm2D (cDataFileIm2D::Create(DirProject()+mNameIm2,false)));


   // Compute mains numeric values

   double aRatioTileFile = RatioMax(mPFileIm1->Sz(),mSzTile);
   mNbLevel = round_up(log(aRatioTileFile) / log(mMaxRatio));
   mRatioByL = pow(aRatioTileFile,1/double(mNbLevel));

   // StdOut() << "RRR=" << aRatioTileFile << "NbLev=" << mNbLevel << " RbyL=" << mRatioByL << "\n";
   // StdOut() << "CMD=" << mSpecs.Name() << "\n";

   // Compute pyramid of images
   MakePyramid();


   return EXIT_SUCCESS;
}


/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_EpipGenDenseMatch(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_EpipGenDenseMatch(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecEpipGenDenseMatch
(
     "DenseMatchEpipGen",
      Alloc_EpipGenDenseMatch,
      "Generik epipolar dense matching",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);

};

