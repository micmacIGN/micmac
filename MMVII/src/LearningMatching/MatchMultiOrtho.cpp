#include "include/MMVII_all.h"

namespace MMVII
{

namespace  cNS_MatchMultipleOrtho
{

class cAppliMatchMultipleOrtho;

class cAppliMatchMultipleOrtho : public cMMVII_Appli
{
     public :
        typedef tU_INT1               tElemMasq;
        typedef tREAL4                tElemOrtho;
        typedef tREAL4                tElemSimil;
        typedef cIm2D<tElemMasq>      tImMasq;
        typedef cIm2D<tElemOrtho>     tImOrtho;
        typedef cIm2D<tElemSimil>     tImSimil;
        typedef cDataIm2D<tElemMasq>  tDImMasq;
        typedef cDataIm2D<tElemOrtho> tDImOrtho;
        typedef cDataIm2D<tElemSimil> tDImSimil;
	typedef std::vector<tImOrtho>  tVecOrtho;
	typedef std::vector<tImMasq>   tVecMasq;


        cAppliMatchMultipleOrtho(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
	std::string NameIm(int aKIm,int aKScale,const std::string & aPost) const
	{
             return mPrefixZ + "_I" +ToStr(aKIm) + "_S" + ToStr(aKScale) + "_"+ aPost  + ".tif";
	}
	std::string NameOrtho(int aKIm,int aKScale) const {return NameIm(aKIm,aKScale,"O");}
	std::string NameMasq(int aKIm,int aKScale) const {return NameIm(aKIm,aKScale,"M");}

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	//  One option, to replace by whatever you want
	void ComputeSimilByCorrelMaster();
	void CorrelMaster(const cPt2di &,int aKIm,bool & AllOk,float &aWeight,float & aCorrel);

	// -------------- Mandatory args -------------------
	std::string   mPrefixGlob;   // Prefix to all names
	int           mNbZ;      // Number of independant ortho (=number of Z)
	int           mNbIm;     // Number of images
	int           mNbScale;  // Number of scale in image
	cPt2di        mSzW;      // Sizeof of windows
	bool          mIm1Mast;  //  Is first image the master image ?


	// -------------- Internal variables -------------------
	tImSimil                   mImSimil;   // computed image of similarity
	std::string                mPrefixZ;   // Prefix for a gizen Z
	cPt2di                     mSzIms;     // common  size of all ortho
	std::vector<tVecOrtho>      mVOrtho;    // vector of loaded ortho at a given Z
	std::vector<tVecMasq>       mVMasq;     // vector of loaded masq  at a given Z
};


/* *************************************************** */
/*                                                     */
/*              cAppliMatchMultipleOrtho               */
/*                                                     */
/* *************************************************** */

cAppliMatchMultipleOrtho::cAppliMatchMultipleOrtho(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mImSimil      (cPt2di(1,1))
{
}


cCollecSpecArg2007 & cAppliMatchMultipleOrtho::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mPrefixGlob,"Prefix of all names")
          <<   Arg2007(mNbZ,"Number of Z/Layers")
          <<   Arg2007(mNbIm,"Number of images in one layer")
          <<   Arg2007(mNbScale,"Number of scaled in on images")
          <<   Arg2007(mSzW,"Size of window")
          <<   Arg2007(mIm1Mast,"Is first image a master image ?")
   ;
}

cCollecSpecArg2007 & cAppliMatchMultipleOrtho::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          // << AOpt2007(mStepZ, "StepZ","Step for paralax",{eTA2007::HDV})
   ;
}


void cAppliMatchMultipleOrtho::CorrelMaster
     (
         const cPt2di & aCenter,   // Central Pixel
	 int aKIm,                  // Num of Image
	 bool & AllOk,              // Is all Window in masq ?
	 float &aWeight,            //  Weight of inside pixel
	 float & aCorrel            // Correl
     )
{
    AllOk = true;
    aWeight = 0;

    cMatIner2Var<tElemOrtho> aMatI;
    for (int aKScale = 0 ; aKScale < mNbScale ; aKScale++)
    {
         const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(aKScale).DIm();
         const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(aKScale).DIm();
         const tDImOrtho & aDIO1 =  mVOrtho.at(0   ).at(aKScale).DIm();
         const tDImOrtho & aDIO2 =  mVOrtho.at(aKIm).at(aKScale).DIm();

	 double aPds = 1/(1+aKScale); // weight, more less arbitrary
         for (const auto & aLocNeigh : cRect2::BoxWindow(cPt2di(0,0),mSzW))  // Parse the window`
         {
              cPt2di  aNeigh = aCenter + aLocNeigh * (1<<aKScale);
              bool Ok = aDIM1.DefGetV(aNeigh,0) && aDIM2.DefGetV(aNeigh,0) ;  // Are both pixel valide
	      if (Ok)
	      {
                  aWeight++;
	          aMatI.Add(aPds,aDIO1.GetV(aNeigh),aDIO2.GetV(aNeigh));
	      }
	      else
	      {
                  AllOk=false;
	      }
         }
    }
    aCorrel =  aMatI.Correl(1e-15);
}

void cAppliMatchMultipleOrtho::ComputeSimilByCorrelMaster()
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
   for (const auto & aP : aDImSim)
   {
        // method : average of image all ok if any, else weighted average of partial corr
        float aSumCorAllOk = 0.0; // Sum of correl of image where point are all ok
        float aSumWeightAllOk = 0.0; //   Nb of All Ok
        float aSumCorPart  = 0.0; //  Sum of weighted partial correl
        float aSumWeightPart = 0.0; //  Sum of weight
	// Parse secondary images 
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
	{
            bool AllOk;
	    float aWeight,aCorrel;
            CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
	    if (AllOk)
	    {
               aSumCorAllOk     += aCorrel;
	       aSumWeightAllOk  += 1;
	    }
	    else
	    {
               aSumCorPart     += aCorrel * aWeight;
	       aSumWeightPart  +=   aWeight;
	    }
	}
	float aAvgCorr =  (aSumWeightAllOk !=0)            ? 
                          (aSumCorAllOk / aSumWeightAllOk) :
                          (aSumCorPart / std::max(1e-5f,aSumWeightPart)) ;

	aDImSim.SetV(aP,1-aAvgCorr);
   }
}


int  cAppliMatchMultipleOrtho::Exe()
{

   // Parse all Z
   for (int aZ=0 ; aZ<mNbZ ; aZ++)
   {
        mPrefixZ =  mPrefixGlob + "_Z" + ToStr(aZ);

        bool NoFile = ExistFile(mPrefixZ+ "_NoData");  // If no data in masq thie file exist
        bool WithFile = ExistFile(NameOrtho(0,0));
	// A little check
        MMVII_INTERNAL_ASSERT_strong(NoFile!=WithFile,"DM4MatchMultipleOrtho, incoherence file");
        if ((aZ==0)  && (true))
        {
             cDataFileIm2D aDF = cDataFileIm2D::Create(NameOrtho(0,0),false);
             StdOut() << " * NbI=" << mNbIm << " NbS=" <<  mNbScale << " NbZ=" <<  mNbZ << " Sz=" << aDF.Sz() << " SzW=" << mSzW << "\n";
        }


	if (WithFile)
        {
	    // Read  orthos and masq in  vectors of images
	    mSzIms = cPt2di(-1234,6789);
	    for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
	    {
                 mVOrtho.push_back(tVecOrtho());
                 mVMasq.push_back(tVecMasq());
                 for (int aKScale=0 ; aKScale<mNbScale ; aKScale++)
		 {
                     mVOrtho.at(aKIm).push_back(tImOrtho::FromFile(NameOrtho(aKIm,aKScale)));
		     if ((aKIm==0) && (aKScale==0))
		         mSzIms = mVOrtho[0][0].DIm().Sz();  // Compute the size at level

                     mVMasq.at(aKIm).push_back(tImMasq::FromFile(NameMasq(aKIm,aKScale)));

		     // check all images have the same at a given level
                     MMVII_INTERNAL_ASSERT_strong(mVOrtho[aKIm][aKScale].DIm().Sz()==mSzIms,"DM4O : variable size(ortho)");
                     MMVII_INTERNAL_ASSERT_strong(mVMasq [aKIm][aKScale].DIm().Sz()==mSzIms,"DM4O : variable size(masq)");
		 }
	    }
	    // Create similarity image with good size
	    mImSimil = tImSimil(mSzIms);
	    mImSimil.DIm().InitCste(2.0);   //  2 => correl of -1

            ComputeSimilByCorrelMaster();
	    mImSimil.DIm().ToFile(mPrefixZ+ "_Sim.tif"); // Save similarities
	    mVOrtho.clear();
	    mVMasq.clear();
        }
   }
   return EXIT_SUCCESS;
}



};

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
using namespace  cNS_MatchMultipleOrtho;

tMMVII_UnikPApli Alloc_MatchMultipleOrtho(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliMatchMultipleOrtho(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecMatchMultipleOrtho
(
     "DM4MatchMultipleOrtho",
      Alloc_MatchMultipleOrtho,
      "Compute similarite of overlapping ortho images",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);



};
