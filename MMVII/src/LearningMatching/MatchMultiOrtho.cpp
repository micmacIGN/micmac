#if MMVII_USE_LIBTORCH
#include "cMMVII_Appli.h"
#include "MMVII_Matrix.h"

// include model architecture 
#include "cCnnModelPredictor.h"

bool TEST=true;

void InterpolatePos(std::vector<double> aX,std::vector<double> aF, double& value)
{
    // Value to Index 
    auto i = std::lower_bound(aX.begin(), aX.end(), value); // sorted in increasing order from 0 to 1
    int k = i - aX.begin();
    //std::cout<<"KKK "<<k<<std::endl;
    int l = k ? k - 1 : 1 ;
    if(aF[k]<aF[l]) 
    {
        value = aF[k]+(value-aX[k])*(aF[l]-aF[k])/(aX[l]-aX[k]);
        //std::cout<<"values "<<value<<std::endl;
    }
    else 
    {
       value = aF[l]+(value-aX[l])*(aF[k]-aF[l])/(aX[k]-aX[l]);
        //std::cout<<"value before  "<<aF[l]<<"  values "<<value<<"  value after "<<aF[k]<<std::endl;
    }
}
 
void Tensor2Tiff(torch::Tensor aTens, std::string anImageName)
{
    Im2D<REAL4,REAL8> anIm=Im2D<REAL4,REAL8> (aTens.size(-1),aTens.size(-2));
    REAL4 ** anImD=anIm.data();
    std::memcpy((*anImD),aTens.data_ptr<REAL4>(),sizeof(REAL4)*aTens.numel());
    ELISE_COPY
    (
     anIm.all_pts(),
     anIm.in() ,
     Tiff_Im(
        anImageName.c_str(),
        anIm.sz(),
        GenIm::real4,
        Tiff_Im::No_Compr,
        Tiff_Im::BlackIsZero,
        Tiff_Im::Empty_ARG ).out()
      );
}
/*
namespace {
	void display_weights(torch::nn::Module & module)
	{
		torch::NoGradGuard no_grad;
        
        
        std::cout<<"MODULE NAME "<<module.name()<<std::endl;
        std::cout<<"MODULE PARAMETERS SIZE "<<module.parameters().size()<<std::endl;

        if (auto conv = module.as<torch::nn::Conv2d>()) {
            std::cout<<"WGHT MATRIX MIN "<<conv->weight.min()<<std::endl;
            std::cout<<"WGHT MATRIX MAX "<<conv->weight.max()<<std::endl;
            //std::cout<<"BIAS VETCOR "<<conv->bias.sizes()<<std::endl;
			}
        if (auto linear = module.as<torch::nn::Linear>()) {
            std::cout<<"WGHT MATRIX "<<linear->weight<<std::endl;
            std::cout<<"BIAS VETCOR "<<linear->bias.sizes()<<std::endl;
			}
	}
}
*/
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
     const std::string  & NameArch() const {return mArchitecture;} // ACCESSOR
     const std::string  & NameDirModel() const {return mModelBinDir;} // ACCESSOR
     private :
	std::string NameIm(int aKIm,int aKScale,const std::string & aPost) const
	{
             return mPrefixZ + "_I" +ToStr(aKIm) + "_S" + ToStr(aKScale) + "_"+ aPost  + ".tif";
	}
        std::string NameImOrg(int aKIm,int aKScale,const std::string & aPost) const
        {
             return mPrefixGlob + "_I" +ToStr(aKIm) + "_S" + ToStr(aKScale) + "_"+ aPost  + ".tif";
        }
        std::string NameImOrgEpip12(int aKIm,int aKIm2,int aKScale, const std::string & aPost) const
        {
             return mPrefixGlob + "_I" +ToStr(aKIm) + "_S" + ToStr(aKScale) + "_I" +ToStr(aKIm2) + "_S" + ToStr(aKScale) + "_"+ aPost  + ".tif";
        }
        std::string NameImEpip(int aKIm,int aKIm2,int aKScale, const std::string & aPost) const
        {
             return mPrefixGlob + "_I" +ToStr(aKIm) + "_I" +ToStr(aKIm2) +"_S" + ToStr(aKScale) + "_"+ aPost  + ".tif";
        }
	std::string NameOrtho(int aKIm,int aKScale) const {return NameIm(aKIm,aKScale,"O");}
        std::string NameGeoX (int aKIm,int aKScale) const {return NameIm(aKIm,aKScale,"GEOX");}
        std::string NameGeoY (int aKIm,int aKScale) const {return NameIm(aKIm,aKScale,"GEOY");}
        std::string NameMasq (int aKIm,int aKScale) const {return NameIm(aKIm,aKScale,"M");}
        std::string NameORIG (int aKIm,int aKScale) const {return NameImOrg(aKIm,aKScale,"ORIG");}

        std::string NameORIGMASTERGEOX (int aKIm,int aKIm2,int aKScale) const {return NameImOrgEpip12(aKIm,aKIm2,aKScale,"ORIG_GEOX");}
        std::string NameMASTEREPIP (int aKIm,int aKIm2,int aKScale) const {return NameImOrgEpip12(aKIm,aKIm2,aKScale,"Epip");}
        std::string NameSECEPIP (int aKIm,int aKIm2,int aKScale) const {return NameImEpip(aKIm,aKIm2,aKScale,"Epip");}
        std::string NameORIGMASTERGEOY (int aKIm,int aKIm2,int aKScale) const {return NameImOrgEpip12(aKIm,aKIm2,aKScale,"ORIG_GEOY");}
        std::string NameORIGMASTERMASQ (int aKIm,int aKIm2,int aKScale) const {return NameImOrgEpip12(aKIm,aKIm2,aKScale,"ORIG_Masq");}
        std::string NameORIGMASTEREpImGEOX (int aKIm,int aKIm2,int aKScale) const {return NameImOrgEpip12(aKIm,aKIm2,aKScale,"ORIG_EpIm_GEOX");}
        std::string NameORIGMASTEREpImGEOY (int aKIm,int aKIm2,int aKScale) const {return NameImOrgEpip12(aKIm,aKIm2,aKScale,"ORIG_EpIm_GEOY");}
        std::string NameORIGMASTEREpImMASQ (int aKIm,int aKIm2,int aKScale) const {return NameImOrgEpip12(aKIm,aKIm2,aKScale,"ORIG_EpIm_Masq");}

        std::string NameORIGSECGEOX (int aKIm,int aKScale) const {return NameImOrg(aKIm,aKScale,"ORIG_GEOX");}
        std::string NameORIGSECGEOY (int aKIm,int aKScale) const {return NameImOrg(aKIm,aKScale,"ORIG_GEOY");}
        std::string NameORIGSECMASQ (int aKIm,int aKScale) const {return NameImOrg(aKIm,aKScale,"ORIG_Masq");}

        std::string NameORIGSECEpImGEOX (int aKIm,int aKScale) const {return NameImOrg(aKIm,aKScale,"ORIG_EpIm_GEOX");}
        std::string NameORIGSECEpImGEOY (int aKIm,int aKScale) const {return NameImOrg(aKIm,aKScale,"ORIG_EpIm_GEOY");}
        std::string NameORIGSECEpImMASQ (int aKIm,int aKScale) const {return NameImOrg(aKIm,aKScale,"ORIG_EpIm_Masq");}


        int Exe() override;
        int ExeProjectOrigEmbeddings();
        int ExeSubPixFeats();
        int GotoEpipolar();
        int GotoHomography();
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        //  One option, to replace by whatever you want
    void ComputeSimilByCorrelMaster();
    void ComputeSimilByLearnedCorrelMaster(std::vector<torch::Tensor> * AllEmbeddings);
    void ComputeSimilByLearnedCorrelMasterEnhanced(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    void ComputeSimilByLearnedCorrelMasterEnhancedHom(std::vector<torch::Tensor> * AllOrthosEmbeddings, std::vector<bool> RelevantEmbeddings);
    void ComputeSimilByLearnedCorrelMasterEnhancedHomMV(std::vector<torch::Tensor> * AllOrthosEmbeddings,std::vector<bool> RelevantEmbeddings);
    void ComputeSimilByLearnedCorrelMasterEnhancedMVS(std::vector<torch::Tensor> * AllOrthosEmbeddings, std::vector<bool> RelevantEmbeddings);
    void ComputeSimilByLearnedCorrelMasterEnhancedMVSMAX(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    void ComputeSimilByLearnedCorrelMasterDecision();
    void ComputeSimilByLearnedCorrelMasterMaxMoy(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    void ComputeSimilByLearnedCorrelMasterMaxMoyMulScale(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    void ComputeSimilByLearnedCorrelMasterDempsterShafer(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    tREAL4 ComputeConflictBetween2SEts(tREAL4 aCorrel1, tREAL4 aCorrel2, tREAL4 aPonder1,tREAL4 aPonder2);
    tREAL4 ComputeJointMassBetween2Sets(tREAL4 aCorrel1, tREAL4 aCorrel2, tREAL4 aPonder1,tREAL4 aPonder2);
    void CorrelMaster(const cPt2di &,int aKIm,bool & AllOk,float &aWeight,float & aCorrel);
    void MakeNormalizedIms();
    void InitializePredictor ();
    torch::Tensor ToTensorGeo(tImOrtho & aGeoX,tImOrtho & aGeoY, cPt2di aDIM);
    torch::Tensor ToTensorGeo(tImOrtho & aGeoX,tImOrtho & aGeoY);
    torch::Tensor ResampleFeatureMap(torch::Tensor & aFeatMap, tImOrtho aGeoX, tImOrtho aGeoY);
    torch::Tensor Gather2D(torch::Tensor & aFeatMap, torch::Tensor anX, torch::Tensor anY);
    torch::Tensor InterpolateFeatMap(torch::Tensor & aFeatMap, tImOrtho aGeoX, tImOrtho aGeoY);
    torch::Tensor ComputeEpipolarImage(tImOrtho & aNativeGeomImage, tImOrtho & aGeoX, tImOrtho & aGeoY);
    double Interpol_Bilin(torch::Tensor & aMap,const cPt2dr & aLoc);
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
	
	// ADDED LEARNING ENV
    aCnnModelPredictor *  mCNNPredictor=nullptr;
    //bool                  mWithIntCorr=true;  // initialized in the begining
    bool                  mWithExtCorr=false;  // initialized in the begining 
    bool                  mUsePredicNet=false;
    bool                  mUseEpip= false;
    bool                  mWithMatcher3D=false;
    std::string           mArchitecture;
    std::string           mResol;
    bool                  mUseCuda=false;
    std::string           mModelBinDir;
    cPt2di                mCNNWin;
    
    // SCRIPTED NETWORKS
    torch::jit::script::Module mMSAFF;
    torch::jit::script::Module mDecisionMLP;
    MSNet_Attention mMSNet=MSNet_Attention(32);

    std::vector<tVecOrtho>      mVOrtho;    // vector of loaded ortho at a given Z
    std::vector<tVecMasq>       mVMasq;     // vector of loaded masq  at a given Z
    std::vector<tVecOrtho>      mVGEOX;     // Real offsets in X direction
    std::vector<tVecOrtho>      mVGEOY;     // Real offsets in Y direction
    std::vector<tVecOrtho>      mORIGIm;    // Original Oriented images used to generate ORTHOS
};

static const std::string TheMSNet="MSNet_Attention";
static const std::string TheUnetMlpCubeMatcher="UnetMLPMatcher";
/* *************************************************** */
/*                                                     */
/*              cAppliMatchMultipleOrtho               */
/*                                                     */
/* *************************************************** */

cAppliMatchMultipleOrtho::cAppliMatchMultipleOrtho(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mImSimil      (cPt2di(1,1)),
   mCNNWin          (0,0)
   
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
          << AOpt2007(mModelBinDir,"CNNParams" ,"Model Directory : Contient des fichiers binaires *.pt")
          << AOpt2007(mArchitecture,"CNNArch" ,"Model architecture : "+TheMSNet+" || "+TheUnetMlpCubeMatcher)
          << AOpt2007(mResol,"RESOL" ,"RESOL OPTION FOR THE MULT^ISCALE TRAINING: ")
          << AOpt2007(mUseCuda,"UseCuda","USE Cuda for inference")
          << AOpt2007(mUsePredicNet,"UsePredicNet","Use the prediction Network to compute learnt similarities")
          << AOpt2007(mUseEpip,"UseEpip","Use epipolar warping to orient features, if 0, then use homographic warping")
   ;
}


void cAppliMatchMultipleOrtho::InitializePredictor ()
{
    StdOut()<<"MODEL ARCHITECTURE:: "<<mArchitecture<<"\n";
    bool IsArchWellDefined=false;
    IsArchWellDefined =(mArchitecture==TheMSNet) || (mArchitecture==TheUnetMlpCubeMatcher);
    MMVII_INTERNAL_ASSERT_strong(IsArchWellDefined,"The network architecture should be specified :  "+TheUnetMlpCubeMatcher+ " !");
    MMVII_INTERNAL_ASSERT_strong(this->mModelBinDir!=""," Model params dir must be specified ! ");
    
    mWithExtCorr = (mArchitecture!="");
    
    if (mWithExtCorr)
    {
        if(mArchitecture==TheUnetMlpCubeMatcher)
        {
            mCNNPredictor = new aCnnModelPredictor(TheUnetMlpCubeMatcher,mModelBinDir,mUseCuda);
            mCNNPredictor->PopulateModelFeatures(mMSAFF,mUseCuda);
            mCNNWin=cPt2di(0,0);
            if (mUsePredicNet)
            {
              mCNNPredictor->PopulateModelDecision(mDecisionMLP,mUseCuda);
            }
            if (mWithMatcher3D)
            {
                // Enhance the  generated correlation coefficients using the last stage conv3d MATCHER
                //mCNNPredictor->PopulateModelMatcher(mMatcherNet);
            }
        }
    }   
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


void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMaster(std::vector<torch::Tensor> * AllOrthosEmbeddings)
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
	const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    auto MasterEmbedding=AllOrthosEmbeddings->at(0);
    int FeatSize=MasterEmbedding.size(0);
    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
        // method : average of image all ok if any, else weighted average of partial corr
        float aSumCorAllOk = 0.0; // Sum of correl of image where point are all ok
        float aSumWeightAllOk = 0.0; //   Nb of All Ok
        float aSumCorPart  = 0.0; //  Sum of weighted partial correl
        float aSumWeightPart = 0.0; //  Sum of weight
	// Parse secondary images 
	using namespace torch::indexing;
    auto aVecRef=MasterEmbedding.slice(0,0,FeatSize,1).slice(1,aP.y(),aP.y()+1,1).slice(2,aP.x(),aP.x()+1,1);
    //std::cout<<" reference vector "<<aVecRef<<std::endl;
    //int smpl=1;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
	{
        bool AllOk;
	    float aWeight,aCorrel;
        //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
        // Compute cosine simialrity with respect to master ortho embeddings 
        
        /**************************************************************************************/
        AllOk = true;
        aWeight = 0;
        const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(  0).DIm();
        for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
        {
            bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
            if (Ok)
            {
                aWeight++;
            }
            else
            {
                AllOk=false;
            }
        }
        // Compute correl separately 
        //using namespace torch::indexing;
        auto aVecOther=AllOrthosEmbeddings->at(aKIm).slice(0,0,FeatSize,1).slice(1,aP.y(),aP.y()+1,1).slice(2,aP.x(),aP.x()+1,1);
        auto aSim=torch::mm(aVecRef.view({1,FeatSize}),aVecOther.view({FeatSize,1}));
        //std::cout<<" slave vector "<<aVecOther<<std::endl;
        aCorrel=(float)aSim.item<float>();
        /**************************************************************************************/  
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

void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterEnhanced(std::vector<torch::Tensor> * AllOrthosEmbeddings)
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
	const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    //compute similarity matrices at the beginning 
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;;
    for (int k=1; k<mNbIm; k++)
    {
        // compute element wise cross product along feature size dimension
        torch::Tensor aCrossProd;
        if (mUsePredicNet)
        {
            torch::Tensor MasterSlave=torch::cat({AllOrthosEmbeddings->at(0).unsqueeze(2),AllOrthosEmbeddings->at(k).unsqueeze(2)},0);
            aCrossProd=mCNNPredictor->PredictONCUBE(mDecisionMLP,MasterSlave).squeeze();
        }
        else
        {
            aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(0),AllOrthosEmbeddings->at(k),0).squeeze();
            //std::cout<<"ORTHO EMBEDDINGS COSINE COMPUTED "<<std::endl;
        }
        //std::cout<<"    MAXXXX    "<<at::max(aCrossProd)<<"    MINNN "<<at::min(aCrossProd)<<std::endl;
        AllSimilarities->push_back(aCrossProd.to(torch::kCPU));
        // Here display similarity images of tiles 
    }
    
    // Free all ortho OneOrthoEmbeding 
    delete AllOrthosEmbeddings; 
    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
        // method : average of image all ok if any, else weighted average of partial corr
        float aSumCorAllOk = 0.0; // Sum of correl of image where point are all ok
        float aSumWeightAllOk = 0.0; //   Nb of All Ok
        float aSumCorPart  = 0.0; //  Sum of weighted partial correl
        float aSumWeightPart = 0.0; //  Sum of weight
	 // Parse secondary images 
	 using namespace torch::indexing;;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
	{
            bool AllOk;
	    float aWeight,aCorrel;
            //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
            // Compute cosine simialrity with respect to master ortho embeddings

            /**************************************************************************************/
            AllOk = true;
            aWeight = 0;
            const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(0   ).DIm();
            for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
            {
                bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
                if (Ok)
                {
                    aWeight++;
                }
                else
                {
                    AllOk=false;
                }
            }
            // Compute correl separately
            //using namespace torch::indexing;
            auto aSim=AllSimilarities->at(aKIm-1).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
            //std::cout<<" slave vector "<<aSim<<std::endl;
            aCorrel=(float)aSim.item<float>();
            /**************************************************************************************/
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

    aDImSim.SetV(aP,1.0-aAvgCorr);
   }

   // delete All Similarities 
   delete AllSimilarities;
}



void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterEnhancedHom(std::vector<torch::Tensor> * AllOrthosEmbeddings,std::vector<bool> RelevantEmbeddings)
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
        const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    //compute similarity matrices at the beginning
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;
    int ANBIM=AllOrthosEmbeddings->size();
    for (int k=1; k<ANBIM; k++)
    {
        // compute element wise cross product along feature size dimension
        torch::Tensor aCrossProd;
        if (mUsePredicNet)
        {
            namespace F=torch::nn::functional;
            // Check Features Normalizaton
            //std::cout<<"SHAPE FEATURES BEFORE NORMALIZATION "<<AllOrthosEmbeddings->at(0).sizes()<<std::endl;
            torch::Tensor MasterSlave=torch::cat({F::normalize(AllOrthosEmbeddings->at(0).unsqueeze(2), F::NormalizeFuncOptions().p(2).dim(0).eps(1e-8)),
                                                  F::normalize(AllOrthosEmbeddings->at(k).unsqueeze(2), F::NormalizeFuncOptions().p(2).dim(0).eps(1e-8))},0);
            //std::cout<<"SHAPE FEATURES AFTER NORMALIZATION "<<MasterSlave.sizes()<<std::endl;
            aCrossProd=mCNNPredictor->PredictONCUBE(mDecisionMLP,MasterSlave).squeeze();
        }
        else
        {
            aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(0),AllOrthosEmbeddings->at(k),0).squeeze();
            //std::cout<<"ORTHO EMBEDDINGS COSINE COMPUTED "<<std::endl;
        }
        //std::cout<<"    MAXXXX    "<<at::max(aCrossProd)<<"    MINNN "<<at::min(aCrossProd)<<std::endl;
        AllSimilarities->push_back(aCrossProd.to(torch::kCPU));
        // Here display similarity images of tiles
    }

    // Free all ortho OneOrthoEmbeding
    delete AllOrthosEmbeddings;
    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
        // method : average of image all ok if any, else weighted average of partial corr
        float aSumCorAllOk = 0.0; // Sum of correl of image where point are all ok
        float aSumWeightAllOk = 0.0; //   Nb of All Ok
        float aSumCorPart  = 0.0; //  Sum of weighted partial correl
        float aSumWeightPart = 0.0; //  Sum of weight
         // Parse secondary images
         using namespace torch::indexing;
        int IndImUtil=0;
        int akImRel=0;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
        {
            bool AllOk;
            float aWeight,aCorrel;
            //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
            // Compute cosine simialrity with respect to master ortho embeddings
            /**************************************************************************************/
            bool isHomCalc=ExistFile(NameORIGSECEpImGEOX(aKIm,0));
            AllOk = true;
            aWeight = 0;
            if (isHomCalc)
             {
                if(RelevantEmbeddings[akImRel])
                  {
                        const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(0   ).DIm();
                        for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
                        {
                            bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
                            if (Ok)
                            {
                                aWeight++;
                            }
                            else
                            {
                                AllOk=false;
                            }
                        }
                        // Compute correl separately
                        //using namespace torch::indexing;
                        auto aSim=AllSimilarities->at(IndImUtil).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
                        //std::cout<<" slave vector "<<aSim<<std::endl;
                        aCorrel=(float)aSim.item<float>();
                        /**************************************************************************************/
                        if (AllOk)
                        {
                           aSumCorAllOk     += aCorrel;
                           aSumWeightAllOk  += 1;
                        }
                        else
                        {
                          aSumCorPart += aCorrel * aWeight;
                           aSumWeightPart  +=   aWeight;
                        }

                        IndImUtil++;
                 }
                akImRel++;
            }
        }
        float aAvgCorr =  (aSumWeightAllOk !=0)            ?
                          (aSumCorAllOk / aSumWeightAllOk) :
                          (aSumCorPart / std::max(1e-5f,aSumWeightPart)) ;

        if (mUsePredicNet)
          {
            aDImSim.SetV(aP,1.0-aAvgCorr);
          }
        else
          {
            aDImSim.SetV(aP,(1.0-aAvgCorr)*0.5);
          }
   }

   // delete All Similarities
   delete AllSimilarities;
}


void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterEnhancedHomMV(std::vector<torch::Tensor> * AllOrthosEmbeddings,std::vector<bool> RelevantEmbeddings)
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
    //compute similarity matrices at the beginning
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;
    int ANBIM=AllOrthosEmbeddings->size();
    // full multi view setting
    for (int k=0 ; k<ANBIM-1 ; k++)
    {
        for (int j=k+1 ; j<ANBIM ; j++)
          {
            torch::Tensor aCrossProd;
            if (mUsePredicNet)
            {
                namespace F=torch::nn::functional;
                /*
                torch::Tensor MasterSlave=torch::cat({F::normalize(AllOrthosEmbeddings->at(k).unsqueeze(2), F::NormalizeFuncOptions().p(2).dim(0).eps(1e-8)),
                                                      F::normalize(AllOrthosEmbeddings->at(j).unsqueeze(2), F::NormalizeFuncOptions().p(2).dim(0).eps(1e-8))},0);
                */
                torch::Tensor MasterSlave=torch::cat({AllOrthosEmbeddings->at(k).unsqueeze(2),
                                                      AllOrthosEmbeddings->at(j).unsqueeze(2)},0);
                aCrossProd=mCNNPredictor->PredictONCUBE(mDecisionMLP,MasterSlave).squeeze();
            }
            else
            {
                aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(k),AllOrthosEmbeddings->at(j),0).squeeze();
            }
            AllSimilarities->push_back(aCrossProd.to(torch::kCPU));
          }
    }
    // Free all ortho OneOrthoEmbeding
    delete AllOrthosEmbeddings;
    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
        // method : average of image all ok if any, else weighted average of partial corr
        float aSumCorAllOk = 0.0; // Sum of correl of image where point are all ok
        float aSumWeightAllOk = 0.0; //   Nb of All Ok
        float aSumCorPart  = 0.0; //  Sum of weighted partial correl
        float aSumWeightPart = 0.0; //  Sum of weight
         // Parse secondary images
         using namespace torch::indexing;

        for (int aKIm=0 ; aKIm<mNbIm-1 ; aKIm++)
        {
          int IndImUtil=0;
          int akImRel=0;
           const tDImMasq & aDIM1  =  mVMasq.at(aKIm   ).at(0   ).DIm();

           for (int aKIm2=aKIm+1 ; aKIm2<mNbIm ; aKIm2++)
            {
                  bool AllOk;
                  float aWeight,aCorrel;
                  //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
                  // Compute cosine simialrity with respect to master ortho embeddings
                  /**************************************************************************************/
                  bool isHomCalc =(aKIm==0) ? ExistFile(NameORIGSECEpImGEOX(aKIm2,0)):
                                              ExistFile(NameORIGSECEpImGEOX(aKIm,0)) && ExistFile(NameORIGSECEpImGEOX(aKIm2,0));
                  AllOk = true;
                  aWeight = 0;
                  if (isHomCalc)
                   {
                      bool IsRel=(aKIm==0) ? RelevantEmbeddings[akImRel]:
                                             RelevantEmbeddings[akImRel] && RelevantEmbeddings[akImRel+1];
                      if(IsRel)
                        {
                              const tDImMasq & aDIM2  =  mVMasq.at(aKIm2).at(0   ).DIm();
                              for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
                              {
                                  bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
                                  if (Ok)
                                  {
                                      aWeight++;
                                  }
                                  else
                                  {
                                      AllOk=false;
                                  }
                              }
                              // Compute correl separately
                              //using namespace torch::indexing;
                              auto aSim=AllSimilarities->at(IndImUtil).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
                              //std::cout<<" slave vector "<<aSim<<std::endl;
                              aCorrel=(float)aSim.item<float>();
                              /**************************************************************************************/
                              if (AllOk)
                              {
                                 aSumCorAllOk     += aCorrel;
                                 aSumWeightAllOk  += 1;
                              }
                              else
                              {
                                aSumCorPart += aCorrel * aWeight;
                                 aSumWeightPart  +=   aWeight;
                              }

                              IndImUtil++;
                       }
                      akImRel++;
                  }
             }

        }

        float aAvgCorr =  (aSumWeightAllOk !=0)            ?
                          (aSumCorAllOk / aSumWeightAllOk) :
                          (aSumCorPart / std::max(1e-5f,aSumWeightPart)) ;

        if (mUsePredicNet)
          {
            aDImSim.SetV(aP,1.0-aAvgCorr);
          }
        else
          {
            aDImSim.SetV(aP,(1.0-aAvgCorr)*0.5);
          }
   }
   // delete All Similarities
   delete AllSimilarities;
}



void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterEnhancedMVS(std::vector<torch::Tensor> * AllOrthosEmbeddings,std::vector<bool> RelevantEmbeddings)
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
     const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    //compute similarity matrices at the beginning
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;
    int aNBCPLES=AllOrthosEmbeddings->size();
    for (int k=0; k<aNBCPLES;k+=2)
      {
        //compute similarity maps by pair of as if it is in epipolar geometry
        torch::Tensor aCrossProd;
        if (mUsePredicNet)
        {
            torch::Tensor MasterSlave=torch::cat({AllOrthosEmbeddings->at(k).unsqueeze(2),AllOrthosEmbeddings->at(k+1).unsqueeze(2)},0);
            aCrossProd=mCNNPredictor->PredictONCUBE(mDecisionMLP,MasterSlave).squeeze();
        }
        else
        {
            aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(k),AllOrthosEmbeddings->at(k+1),0).squeeze();
        }
        //std::cout<<"    MAXXXX    "<<at::max(aCrossProd)<<"    MINNN "<<at::min(aCrossProd)<<std::endl;
        AllSimilarities->push_back(aCrossProd.to(torch::kCPU));
      }
    // Free all ortho OneOrthoEmbeding
    delete AllOrthosEmbeddings;
    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
        // method : average of image all ok if any, else weighted average of partial corr
        float aSumCorAllOk = 0.0; // Sum of correl of image where point are all ok
        float aSumWeightAllOk = 0.0; //   Nb of All Ok
        float aSumCorPart  = 0.0; //  Sum of weighted partial correl
        float aSumWeightPart = 0.0; //  Sum of weight
         // Parse secondary images
         using namespace torch::indexing;
        int akImUtil=0;
        int akImRel=0;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
        {
            bool IsEpipCalcMaster=ExistFile(NameORIGMASTEREpImGEOX(0,aKIm,0));
            bool AllOk;
            float aWeight,aCorrel;
            //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
            // Compute cosine simialrity with respect to master ortho embeddings
            /**************************************************************************************/
            AllOk = true;
            aWeight = 0;
            if (IsEpipCalcMaster)
              {
                if(RelevantEmbeddings[akImRel])
                  {
                    const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(0   ).DIm();
                    for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
                    {
                        bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
                        if (Ok)
                        {
                            aWeight++;
                        }
                        else
                        {
                            AllOk=false;
                        }
                    }
                        // Compute correl separately
                        //using namespace torch::indexing;
                        auto aSim=AllSimilarities->at(akImUtil).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
                        //std::cout<<" slave vector "<<aSim<<std::endl;
                        aCorrel=(float)aSim.item<float>();
                        /**************************************************************************************/
                        if (AllOk)
                        {
                           aSumCorAllOk     += aCorrel;
                           aSumWeightAllOk  += 1;
                        }
                        else
                        {
                          aSumCorPart += aCorrel * aWeight;
                           aSumWeightPart  +=   aWeight;
                        }

                        if (0)
                          {
                            std::cout<<"Get Correl Values <<  "<<aCorrel<<"  WEIGHT: "<<aWeight<<"   "<<std::endl;
                          }

                        akImUtil++;
                  }
                akImRel++;
               }
          }


        float aAvgCorr =  (aSumWeightAllOk !=0)            ?
                          (aSumCorAllOk / aSumWeightAllOk) :
                          (aSumCorPart / std::max(1e-5f,aSumWeightPart)) ;
        if (0)
          {
            std::cout<<"Get AVG Correl Value <<  "<<aAvgCorr<<"  aCoord "<<aP<<std::endl;
          }

        if (mUsePredicNet)
          {
            aDImSim.SetV(aP,1.0-aAvgCorr);
          }
        else
          {
            aDImSim.SetV(aP,(1.0-aAvgCorr)*0.5);
          }
   }
   // delete All Similarities
   delete AllSimilarities;
}


void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterEnhancedMVSMAX(std::vector<torch::Tensor> * AllOrthosEmbeddings)
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
     const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    //compute similarity matrices at the beginning
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;

    for (int k=0; k<2*(mNbIm-1);k+=2)
      {
        //compute similarity maps by pair of as if it is in epipolar geometry
        torch::Tensor aCrossProd;
        if (mUsePredicNet)
        {
            torch::Tensor MasterSlave=torch::cat({AllOrthosEmbeddings->at(k).unsqueeze(2),AllOrthosEmbeddings->at(k+1).unsqueeze(2)},0);
            aCrossProd=mCNNPredictor->PredictONCUBE(mDecisionMLP,MasterSlave).squeeze();
        }
        else
        {
            aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(k),AllOrthosEmbeddings->at(k+1),0).squeeze();
        }
        //std::cout<<"    MAXXXX    "<<at::max(aCrossProd)<<"    MINNN "<<at::min(aCrossProd)<<std::endl;
        AllSimilarities->push_back(aCrossProd.to(torch::kCPU));
      }
    // Free all ortho OneOrthoEmbeding
    delete AllOrthosEmbeddings;
    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
         // Parse secondary images
         using namespace torch::indexing;
        float aCorMax = -1.0;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
        {
            float aCorrel,aWeight;
            aWeight = 0;
            const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(0   ).DIm();
            for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
            {
                bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
                if (Ok)
                {
                    aWeight++;
                }
                /*else
                {
                    AllOk=false;
                }*/
            }

            auto aSim=AllSimilarities->at(aKIm-1).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
            aCorrel=(float)aSim.item<float>();

            aCorrel=(aCorrel*aWeight)/(std::pow(2*mCNNWin.x()+1.0,2));

            if (0)
              {
                std::cout<<"Get Correl Values <<  "<<aCorrel<<"  WEIGHT: "<<aWeight<<"   "<<" sz : "<<2*mCNNWin.x()+1.0<<std::endl;
              }

            if (aCorrel>aCorMax) aCorMax=aCorrel;
        }

        if (0)
          {
            std::cout<<"Get Max Correl Value <<  "<<aCorMax<<"  aCoord "<<aP<<std::endl;
          }

        if (mUsePredicNet)
          {
            aDImSim.SetV(aP,1.0-aCorMax);
          }
        else
          {
            aDImSim.SetV(aP,(1.0-aCorMax)*0.5);
          }
   }
   // delete All Similarities
   delete AllSimilarities;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterDecision()
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
    const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    //compute similarity matrices at the beginning 
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;
    auto MasterOrtho=mVOrtho.at(0); //vector<tImOrtho>
    for (unsigned int i=1;i<mVOrtho.size();i++)
    {
        cPt2di aSzOrtho=mVOrtho.at(i).at(0).DIm().Sz();
        auto aCrossProd=mCNNPredictor->PredictUNetWDecision(mMSAFF,MasterOrtho,mVOrtho.at(i),aSzOrtho);
        //std::cout<<"Shape of a single similartity Map  ========> "<<aCrossProd.sizes()<<endl;
        //std::cout<<"BORNES INF ET SUP DE LA CARTE DE SIM =======> "<<at::max(aCrossProd)<<"  "<<at::min(aCrossProd)<<std::endl;
        AllSimilarities->push_back(aCrossProd);     
    }

    for (const auto & aP : aDImSim)
      {
        // method : average of image all ok if any, else weighted average of partial corr
        float aSumCorAllOk = 0.0; // Sum of correl of image where point are all ok
        float aSumWeightAllOk = 0.0; //   Nb of All Ok
        float aSumCorPart  = 0.0; //  Sum of weighted partial correl
        float aSumWeightPart = 0.0; //  Sum of weight
        // Parse secondary images
        using namespace torch::indexing;;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
	{
            bool AllOk;
	    float aWeight,aCorrel;
            //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
            // Compute cosine simialrity with respect to master ortho embeddings
            AllOk = true;
            aWeight = 0;
            const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(0   ).DIm();
            for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
            {
                bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
                if (Ok)
                {
                    aWeight++;
                }
                else
                {
                    AllOk=false;
                }
            }

            // Compute correl separately
            //using namespace torch::indexing;
            auto aSim=AllSimilarities->at(aKIm-1).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
            //std::cout<<" slave vector "<<aSim<<std::endl;
            aCorrel=(float)aSim.item<float>();

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
	double aAvgCorr =  (aSumWeightAllOk !=0)            ? 
                          (aSumCorAllOk / aSumWeightAllOk) :
                          (aSumCorPart / std::max(1e-5f,aSumWeightPart)) ;
    //std::cout<<"avant "<<aAvgCorr<<std::endl;
    // Interpolate correlation values 
    //InterpolatePos(spaceCorrDUBLIN,spaceProbDUBLIN,aAvgCorr);
    //std::cout<<"apres "<<aAvgCorr<<std::endl;
    aDImSim.SetV(aP,1.0-aAvgCorr);
   }
   // delete All Similarities 
   delete AllSimilarities;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterMaxMoy(std::vector<torch::Tensor> * AllOrthosEmbeddings)
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
    const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    //compute similarity matrices at the beginning 
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;;
    for (int k=1; k<mNbIm; k++)
    {
        
        // compute element wise cross product along feature size dimension 
        auto aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(0),AllOrthosEmbeddings->at(k),0).squeeze();
        AllSimilarities->push_back(aCrossProd);
    }
    // Free all ortho OneOrthoEmbeding 
    delete AllOrthosEmbeddings; 

    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
         // Parse secondary images  // not cpp
        tREAL4 *aTab = new tREAL4 [mNbIm-1];
        tREAL4 *aPonder=new tREAL4 [mNbIm-1];
	 using namespace torch::indexing;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
	{
            bool AllOk;
	    float aWeight,aCorrel;
            //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
            // Compute cosine simialrity with respect to master ortho embeddings

            /**************************************************************************************/
            AllOk = true;
            aWeight = 0;

            const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(0   ).DIm();
            for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
            {
                bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
                if (Ok)
                {
                    aWeight++;
                }
                else
                {
                    AllOk=false;
                }
            }
            // Compute correl separately
            //using namespace torch::indexing;
            auto aSim=AllSimilarities->at(aKIm-1).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
            aCorrel=(float)aSim.item<float>();
            /**************************************************************************************/
                if (AllOk)
                {
                aTab[aKIm-1]    = aCorrel;
                aPonder[aKIm-1] = 1.0;
                }
                else
                {
               aTab[aKIm-1]    = aCorrel ;
               aPonder[aKIm-1] = aWeight/(mCNNWin.x()*mCNNWin.y());
                }
	}
	// Moyennes deux à deux des corrélations
        tREAL4 AggCorr=-2.0;
        for (int j=0;j<mNbIm-2;j++)
        {
            for (int i=j+1;i<mNbIm-1;i++)
            {
                tREAL4 aCorr=(aTab[i]*aPonder[i]+ aTab[j]*aPonder[j])/std::max(1e-5f,aPonder[i]+aPonder[j]);
                if (AggCorr<aCorr)
                {
                    AggCorr=aCorr;
                }
            }
        }

        delete[] aTab;
        delete[] aPonder;

        if(AggCorr==-2.0) AggCorr=0.5;   // no max is found
        aDImSim.SetV(aP,1-AggCorr);
   }

   // delete All Similarities 
   delete AllSimilarities;
}

void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterMaxMoyMulScale(std::vector<torch::Tensor> * AllOrthosEmbeddings)
{
    
    // Here we jointly fuse Similarity measures from Multi-Scale information 
    // Size of embeddings vector = 4 (scales)*features
    
    // Ortho 1
    // 0 --> Resol 1
    // 1 --> Resol / 2
    // 2 --> Resol / 4
    // 3 --> Resol / 8
    
    // Ortho 2
    // 4 --> Resol 1
    // 5 --> Resol / 2 
    // 6 --> Resol / 4
    // 7 --> Resol / 8
    
    //.....
    
    
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
	const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    //compute similarity matrices at the beginning 
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;;
    for (int scales=0;scales<4;scales++)
    {
        for (int k=1; k<mNbIm; k++)
        {
            // compute element wise cross product along feature size dimension 
            auto aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(scales),AllOrthosEmbeddings->at(4*k+scales),0).squeeze();
            AllSimilarities->push_back(aCrossProd);
        }
    }
    
    // Free all ortho OneOrthoEmbeding 
    delete AllOrthosEmbeddings; 
    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
	 // Parse secondary images 
	tREAL4 * aTab=new tREAL4 [(mNbIm-1)*4];
	tREAL4 * aPonder=new tREAL4 [(mNbIm-1)*4];
    
	 using namespace torch::indexing;
     for (int aKIm=1 ; aKIm<=(mNbIm-1)*4 ; aKIm++)
	{
	    bool AllOk;
	    float aWeight,aCorrel;
        //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
        // Compute cosine simialrity with respect to master ortho embeddings 
        
        /**************************************************************************************/
        AllOk = true;
        aWeight = 0;
        const tDImMasq & aDIM2  =  mVMasq.at(aKIm%(mNbIm-1) ? aKIm%(mNbIm-1) : (mNbIm-1)).at(0 ).DIm();
        for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
        {
            bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valid
            if (Ok)
            {
                aWeight++;
            }
            else
            {
                AllOk=false;
            }
        }
        // Compute correl separately 
        //using namespace torch::indexing;
        auto aSim=AllSimilarities->at(aKIm-1).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
        //std::cout<<" slave vector "<<aVecOther<<std::endl;
        aCorrel=(float)aSim.item<float>();
        /**************************************************************************************/  
	    if (AllOk)
	    {
            aTab[aKIm-1]    = aCorrel;
            aPonder[aKIm-1] = 1.0;
	    }
	    else
	    {
           aTab[aKIm-1]    = aCorrel ; 
           aPonder[aKIm-1] = aWeight/(mCNNWin.x()*mCNNWin.y());
	    }
	}
	// Moyennes deux à deux des corrélations
    tREAL4 AggCorr=-2.0;
    tREAL4 AggCorrMaxAllScales=-2.0;
    
    for (int scales=0;scales<4;scales++)
    {
        AggCorr=-2.0;
        for (int j=0;j<mNbIm-2;j++)
        {
            for (int i=j+1;i<mNbIm-1;i++)
            {
                tREAL4 aCorr=(aTab[i+(mNbIm-1)*scales]*aPonder[i+(mNbIm-1)*scales]+ aTab[j+(mNbIm-1)*scales]*aPonder[j+(mNbIm-1)*scales])/std::max(1e-5f,aPonder[i+(mNbIm-1)*scales]+aPonder[j+(mNbIm-1)*scales]); 
                if (AggCorr<aCorr)
                {
                    AggCorr=aCorr;
                }
            }
        }
        if (AggCorrMaxAllScales<AggCorr)
        {
            AggCorrMaxAllScales=AggCorr;
        }
    }
    if(AggCorrMaxAllScales==-2.0) AggCorrMaxAllScales=0.5;   // no max is found 
    aDImSim.SetV(aP,1-AggCorrMaxAllScales);

    //free
    delete[] aTab;
    delete[] aPonder;
   }

   // delete All Similarities 
   delete AllSimilarities;
}


void cAppliMatchMultipleOrtho::ComputeSimilByLearnedCorrelMasterDempsterShafer(std::vector<torch::Tensor> * AllOrthosEmbeddings)
{
   MMVII_INTERNAL_ASSERT_strong(mIm1Mast,"DM4MatchMultipleOrtho, for now, only handle master image mode");

   tDImSimil & aDImSim = mImSimil.DIm();
   // Parse all pixels
	const tDImMasq & aDIM1  =  mVMasq.at(0   ).at(0   ).DIm();
    //compute similarity matrices at the beginning 
    std::vector<torch::Tensor> * AllSimilarities= new std::vector<torch::Tensor>;;
    for (int k=1; k<mNbIm; k++)
    {
        // compute element wise cross product along feature size dimension 
        auto aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(0),AllOrthosEmbeddings->at(k),0).squeeze();
        AllSimilarities->push_back(aCrossProd);
    }
    // Free all ortho OneOrthoEmbeding 
    delete AllOrthosEmbeddings; 
    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
         // Parse secondary images // not cpp account for vs win
    tREAL4 *aTab    =new tREAL4 [mNbIm-1];
    tREAL4 *aPonder =new tREAL4 [mNbIm-1];
	 using namespace torch::indexing;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
	{
        bool AllOk;
	    float aWeight,aCorrel;
        //CorrelMaster(aP,aKIm,AllOk,aWeight,aCorrel);
        // Compute cosine simialrity with respect to master ortho embeddings 
        
        /**************************************************************************************/
        AllOk = true;
        aWeight = 0;
        const tDImMasq & aDIM2  =  mVMasq.at(aKIm).at(0   ).DIm();
        for (const auto & aPvoisin : cRect2::BoxWindow(aP,mCNNWin))  // Parse the window`
        {
            bool Ok = aDIM1.DefGetV(aPvoisin,0) && aDIM2.DefGetV(aPvoisin,0) ;  // Are both pixel valide
            if (Ok)
            {
                aWeight++;
            }
            else
            {
                AllOk=false;
            }
        }
        // Compute correl separately 
        auto aSim=AllSimilarities->at(aKIm-1).slice(0,aP.y(),aP.y()+1,1).slice(1,aP.x(),aP.x()+1,1);
        //std::cout<<" slave vector "<<aVecOther<<std::endl;
        aCorrel=(float)aSim.item<float>();
        /**************************************************************************************/  
	    if (AllOk)
	    {
            aTab[aKIm-1]    = aCorrel;
            aPonder[aKIm-1] = 1.0;
	    }
	    else
	    {
           aTab[aKIm-1]    = aCorrel ; 
           aPonder[aKIm-1] = aWeight/(mCNNWin.x()*mCNNWin.y());
	    }
	}
	// COmbinaison des corrélations par la méthode de D-S
	tREAL4 AggCorr;
    if (mNbIm==2) 
    {
        AggCorr=aTab[0];
    }
    else
    {
        AggCorr=ComputeJointMassBetween2Sets(aTab[0],aTab[1],aPonder[0],aPonder[1]);
        //std::cout<<" Value of correl "<<AggCorr<<std::endl;
        if (mNbIm>3)
        {
            for (int j=0;j<mNbIm-2;j++)
            {
                for (int i=(j==0) ? j+2:j+1;i<mNbIm-1;i++)
                {
                    AggCorr=ComputeJointMassBetween2Sets(AggCorr,aTab[i],1.0,aPonder[i]);
                }
            }
        }
    }
    //std::cout<<" Value of correl ))  "<<AggCorr<<std::endl;
    aDImSim.SetV(aP,1-AggCorr);

    // free
    delete [] aTab;
    delete [] aPonder;
   }

   // delete All Similarities 
   delete AllSimilarities;
}

tREAL4 cAppliMatchMultipleOrtho::ComputeConflictBetween2SEts(tREAL4 aCorrel1, tREAL4 aCorrel2, tREAL4 aPonder1,tREAL4 aPonder2)
{
    return aPonder1*aCorrel1*(1-aPonder2*aCorrel2)+ (1-aPonder1*aCorrel1)+aPonder2*aCorrel2;
}
    
tREAL4 cAppliMatchMultipleOrtho::ComputeJointMassBetween2Sets(tREAL4 aCorrel1, tREAL4 aCorrel2, tREAL4 aPonder1,tREAL4 aPonder2)
{
    tREAL4 k=ComputeConflictBetween2SEts(aCorrel1,aCorrel2,aPonder1,aPonder2);
    //std::cout<<"Conflict between both values "<<k<<std::endl;
    if (k==1.0)  // conflict between 2 Correlations measures  ==> returm mean of correl
    {
        // for now return average but if there is a conflict between correl, it should be considered as an indicator !!!
        return (aCorrel1*aPonder1+aCorrel2*aPonder2)/std::max(1e-5f,aPonder1+aPonder2);
    }
    else
    {
        return aCorrel1*aPonder1*aPonder2*aCorrel2/(1-k);
    }
}

void cAppliMatchMultipleOrtho::MakeNormalizedIms()  // Possible errors here 
{
    // NORMALIZING IMAGES BEFORE INFERENCE 
    for( auto& MsOrth: mVOrtho)
    {
        for (auto& Im:MsOrth)
        {
            Im=NormalizedAvgDev(Im,1e-4);
        }
    }
}



int  cAppliMatchMultipleOrtho::ExeSubPixFeats()
{

   // Parse all Z
   // If using a model (CNN) Initialize the predictor 
   if (mArchitecture!="")
   {
        InitializePredictor();
   }
   for (int aZ=0 ; aZ<mNbZ ; aZ++)
   {
        mPrefixZ =  mPrefixGlob + "_Z" + ToStr(aZ);

        bool NoFile = ExistFile(mPrefixZ+ "_NoData");  // If no data in masq thie file exist
        bool WithFile = ExistFile(NameOrtho(0,0));
	// A little check
        MMVII_INTERNAL_ASSERT_strong(NoFile!=WithFile,"DM4MatchMultipleOrtho, incoherence file");
        if ((aZ==0)  && (true))
        {
             cDataFileIm2D aDF = cDataFileIm2D::Create(NameOrtho(0,0),eForceGray::Yes);
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

	    
	    // NORMALIZE IF LEARNING BASED CORRELATION 
        if (mWithExtCorr)
        {
           // MakeNormalizedIms();
        
        }
    
	    // Create similarity image with good size
	    mImSimil = tImSimil(mSzIms);
	    mImSimil.DIm().InitCste(1.0);

        if(mWithExtCorr)
        {
            std::vector<torch::Tensor> * OrthosEmbeddings= new std::vector<torch::Tensor>;
            if (!mUsePredicNet)
            {
                // Inference based correlation 
                // Create Embeddings 
                // Calculate the EMBEDDINGS ONE TIME USING FOWARD OVER THE WHOLE TILEs
                for (unsigned int i=0;i<mVOrtho.size();i++)
                {
                    cPt2di aSzOrtho=mVOrtho.at(i).at(0).DIm().Sz();
                    torch::Tensor OneOrthoEmbeding;
                    if (mArchitecture==TheUnetMlpCubeMatcher)
                      {
                            OneOrthoEmbeding=mCNNPredictor->PredictUnetFeaturesOnly(mMSAFF,mVOrtho.at(i),aSzOrtho);
                      }

                    //StdOut()  <<" EMBEDDING FOR VECTOR OR FULL RESOLUTION ORTHO : "<<i<<OneOrthoEmbeding.sizes()<<"\n";
                    // store in relevant vector 
                    OrthosEmbeddings->push_back(OneOrthoEmbeding);
                }

                //
                if (mArchitecture==TheUnetMlpCubeMatcher)
                {
                    //ComputeSimilByLearnedCorrelMasterMaxMoyMulScale(OrthosEmbeddings); // Size 4*numberofOrthos
                    ComputeSimilByLearnedCorrelMasterEnhanced(OrthosEmbeddings);
                    //ComputeSimilByLearnedCorrelMasterDecision(); <<<<<<here>>>>>>
                }
                else {
                    MMVII_INTERNAL_ASSERT_strong(false,"Nothing to compute, no model architecture is provided");
                  }

            }
            //StdOut()  <<" Size OF EMBEDDINGS MS : " <<OrthosEmbeddings->size()<<"\n";

            else
              {
                  // GIVEN THE ORTHOS EMBEDDINGS, Compute Correlation for each pixel in the similarity image => index work to get vectors from from tensors
                  if (mArchitecture==TheMSNet)
                  {
                      //ComputeSimilByLearnedCorrelMasterMaxMoyMulScale(OrthosEmbeddings); // Size 4*numberofOrthos
                      ComputeSimilByLearnedCorrelMasterEnhanced(OrthosEmbeddings);
                      //ComputeSimilByLearnedCorrelMasterDecision(); <<<<<<here>>>>>>
                  }
                  if (mArchitecture==TheUnetMlpCubeMatcher)
                  {
                      ComputeSimilByLearnedCorrelMasterDecision();
                  }
                  else
                  {
                     ComputeSimilByLearnedCorrelMasterMaxMoy(OrthosEmbeddings);
                  }
              }
            
        }
        else
        {
            ComputeSimilByCorrelMaster();    
        }
        
	    mImSimil.DIm().ToFile(mPrefixZ+ "_Sim.tif"); // Save similarities
	    mVOrtho.clear();
	    mVMasq.clear();
        }
   }
   return EXIT_SUCCESS;
}

double cAppliMatchMultipleOrtho::Interpol_Bilin(torch::Tensor & aMap, const cPt2dr & aLoc)
{
    using namespace torch::indexing;

    double InterpValue=0.0;
    double aLocX=(double)aLoc.x();
    double aLocY=(double)aLoc.y();
    int y_1 = floor(aLocY);
    int x_1 = floor(aLocX);
    int y_2 = ceil(aLocY);
    int x_2 = ceil(aLocX);
    //std::cout<<" 2D SLICE TO COMPUTE INTERPOL "<<aMap.sizes()<<std::endl;
    if (x_2<aMap.size(1) && y_2<aMap.size(0))
        {
            double aMap11=aMap.index({y_1,x_1}).item<double>();
            //double aMap11=aMap.slice(y_1,y_1+1).slice(x_1,x_1+1).item<double>();
            double aMap21=aMap.index({y_2,x_1}).item<double>() ;
            //double aMap21=aMap.slice(y_2,y_2+1).slice(x_1,x_1+1).item<double>();
            double aMap12=aMap.index({y_1,x_2}).item<double>() ;
            //double aMap12=aMap.slice(y_1,y_1+1).slice(x_2,x_2+1).item<double>();
            double aMap22=aMap.index({y_2,x_2}).item<double>() ;
            //double aMap22=aMap.slice(y_2,y_2+1).slice(x_2,x_2+1).item<double>();

            double y_2_y=(double)y_2-aLocY;
            double y_y_1=aLocY-(double)y_1;
            // Interpolate values

            InterpValue= ((double)x_2-aLocX)*(aMap11*y_2_y+aMap21*y_y_1)
                    + (aLocX-(double)x_1)*(aMap12*y_2_y+aMap22*y_y_1);
        }
    //std::cout<<"INTER    PPPP "<<InterpValue<<std::endl;
    return InterpValue;
}

torch::Tensor cAppliMatchMultipleOrtho::Gather2D(torch::Tensor & aFeatMap, torch::Tensor  anX,
                                                 torch::Tensor  anY)
{
    using namespace torch::indexing;
    // Gathers a tensor given mappings anX and anY
    // Out[F,j,k] = In[F,anY[j,k], anX[j,k]]
    /*std::cout<<"INITIAL OFFSETS SHAPE "<<anY.sizes()<<"   "<<anX.sizes()<<std::endl;
    auto index_y=anY.view(anY.size(0)*anY.size(1));
    auto index_x=anX.view(anX.size(0)*anX.size(1));
    auto IN=aFeatMap.contiguous();
    IN=IN.view({-1,IN.size(1)*IN.size(2)});
    auto aInterpolFeat= IN.index({Slice(0,None,1),index_y,index_x}).view({-1,anY.size(0),anY.size(1)});
    return aInterpolFeat;*/
    /*auto IN=aFeatMap.contiguous();
    torch::Tensor Lin_idx= anY + IN.size(-1) * anX;
    std::cout<<" Linear  index "<<Lin_idx.sizes()<<std::endl;
    IN=IN.view({-1,IN.size(1)*IN.size(2)});
    return torch::gather(IN,-1,Lin_idx).view({-1,anY.size(-2),anY.size(-1)});*/

    return torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),anY,anX})});
}

torch::Tensor cAppliMatchMultipleOrtho::ToTensorGeo(tImOrtho & aGeoX,tImOrtho & aGeoY, cPt2di aDIM)
{
  using namespace torch::indexing;
  cPt2di aSzOrtho=aGeoX.DIm().Sz();
  // Generate tensor offsets
  tREAL4 ** mGeoXData=aGeoX.DIm().ExtractRawData2D();
  tREAL4 ** mGeoYData=aGeoY.DIm().ExtractRawData2D();
  // create offsets tensors for interpolation
  torch::Tensor aGeoXT=torch::from_blob((*mGeoXData), {aSzOrtho.y(),aSzOrtho.x()},
                                        torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor aGeoYT=torch::from_blob((*mGeoYData), {aSzOrtho.y(),aSzOrtho.x()},
                                        torch::TensorOptions().dtype(torch::kFloat32));
   aGeoXT=aGeoXT.index({Slice(0,aSzOrtho.y()-1,1),Slice(0,aSzOrtho.x()-1,1)});
   aGeoYT=aGeoYT.index({Slice(0,aSzOrtho.y()-1,1),Slice(0,aSzOrtho.x()-1,1)});
   return torch::stack({aGeoXT.div((float)aDIM.x()/2),aGeoYT.div((float)aDIM.y()/2)},-1).sub(1.0).unsqueeze(0);
}

torch::Tensor cAppliMatchMultipleOrtho::ToTensorGeo(tImOrtho & aGeoX,tImOrtho & aGeoY)
{
  using namespace torch::indexing;
  cPt2di aSzOrtho=aGeoX.DIm().Sz();
  // Generate tensor offsets
  tREAL4 ** mGeoXData=aGeoX.DIm().ExtractRawData2D();
  tREAL4 ** mGeoYData=aGeoY.DIm().ExtractRawData2D();
  // create offsets tensors for interpolation
  torch::Tensor aGeoXT=torch::from_blob((*mGeoXData), {aSzOrtho.y(),aSzOrtho.x()},
                                        torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor aGeoYT=torch::from_blob((*mGeoYData), {aSzOrtho.y(),aSzOrtho.x()},
                                        torch::TensorOptions().dtype(torch::kFloat32));
   //aGeoXT=aGeoXT.index({Slice(0,aSzOrtho.y()-1,1),Slice(0,aSzOrtho.x()-1,1)});
   //aGeoYT=aGeoYT.index({Slice(0,aSzOrtho.y()-1,1),Slice(0,aSzOrtho.x()-1,1)});
   return torch::stack({aGeoXT,aGeoYT},-1).unsqueeze(0);
}

torch::Tensor cAppliMatchMultipleOrtho::InterpolateFeatMap(torch::Tensor & aFeatMap,
                                                           tImOrtho aGeoX, tImOrtho aGeoY)
{
    cPt2di aSzOrtho=aGeoX.DIm().Sz();
    using namespace torch::indexing;
    auto FeatSz=aFeatMap.size(0);
    auto HFeatMap=aFeatMap.size(1);
    auto WFeatMap=aFeatMap.size(2);

    // Tensor Interpolate at once
    torch::Tensor anInterPolFeatMap=torch::zeros({FeatSz,aSzOrtho.y()-1,aSzOrtho.x()-1},
                                                 torch::TensorOptions().dtype(torch::kFloat32));

    // Generate tensor offsets
    tREAL4 ** mGeoXData=aGeoX.DIm().ExtractRawData2D();
    tREAL4 ** mGeoYData=aGeoY.DIm().ExtractRawData2D();

    // create offsets tensors for interpolation
    torch::Tensor aGeoXT=torch::from_blob((*mGeoXData), {aSzOrtho.y(),aSzOrtho.x()},
                                          torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor aGeoYT=torch::from_blob((*mGeoYData), {aSzOrtho.y(),aSzOrtho.x()},
                                          torch::TensorOptions().dtype(torch::kFloat32));

     aGeoXT=aGeoXT.index({Slice(0,aSzOrtho.y()-1,1),Slice(0,aSzOrtho.x()-1,1)});
     aGeoYT=aGeoYT.index({Slice(0,aSzOrtho.y()-1,1),Slice(0,aSzOrtho.x()-1,1)});
    //std::cout<<"  OFFSETS DIMENSIONS ------   X"<<aGeoXT.sizes()<<"   Y  "<<aGeoYT.sizes()<<std::endl;

    // Get definition Tensors
    auto DEF_X=torch::mul((aGeoXT>=0), (aGeoXT<WFeatMap));
    auto DEF_Y=torch::mul((aGeoYT>=0), (aGeoYT<HFeatMap));

    // floor and ceil
    torch::Tensor Y_1=torch::floor(aGeoYT).mul(DEF_Y);
    torch::Tensor X_1=torch::floor(aGeoXT).mul(DEF_X);

    torch::Tensor Y_2=torch::ceil(aGeoYT).mul(DEF_Y);
    torch::Tensor X_2=torch::ceil(aGeoXT).mul(DEF_X);

    aGeoXT=aGeoXT.mul(DEF_X);
    aGeoYT=aGeoYT.mul(DEF_Y);

    Y_2=Y_2.mul(Y_2<HFeatMap);
    X_2=X_2.mul(X_2<WFeatMap);

    //std::cout<<" offsets limits "<<at::max(X_2)<<" Y_2 "<<at::max(Y_2)<<" X_1  "<<at::max(X_1)<<" Y_1 "<<at::max(Y_1)<<std::endl;

    //std::cout<<"  X_2 AND Y_2 DIMENSIONS ------   X"<<Y_2.sizes()<<"   Y  "<<X_2.sizes()<<std::endl;

    //auto alongx=torch::gather(aFeatMap,-2,X_1.unsqueeze(0).repeat_interleave(FeatSz,0).to(torch::kInt64));

    //std::cout<<"ALONG XXXX "<<alongx.sizes()<<std::endl;

    //auto aMAP_11= torch::gather(alongx,-1,Y_1.unsqueeze(0).repeat_interleave(FeatSz,0).to(torch::kInt64));
    //auto aMAP_11=torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),Y_1.to(torch::kInt64),X_1.to(torch::kInt64)})});
    auto aMAP_11=aFeatMap.index({Slice(),Y_1.to(torch::kInt64),X_1.to(torch::kInt64)});
    //std::cout<<"  composed offsets "<<aMAP_11.sizes()<<std::endl;
    //auto aMAP_21=torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),Y_2.to(torch::kInt64),X_1.to(torch::kInt64)})});
    auto aMAP_21=aFeatMap.index({Slice(),Y_2.to(torch::kInt64),X_1.to(torch::kInt64)});
    //auto aMAP_12=torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),Y_1.to(torch::kInt64),X_2.to(torch::kInt64)})});
    auto aMAP_12=aFeatMap.index({Slice(),Y_1.to(torch::kInt64),X_2.to(torch::kInt64)});
    //auto aMAP_22=torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),Y_2.to(torch::kInt64),X_2.to(torch::kInt64)})});
    auto aMAP_22=aFeatMap.index({Slice(),Y_2.to(torch::kInt64),X_2.to(torch::kInt64)});
    //std::cout<<"  MAP 22 DIMENSIONS ------   X"<<aMAP_22.sizes()<<"   Y  "<<aMAP_22.sizes()<<std::endl;

    /*bool oktest1=true;
    if (oktest1)
        {
            // Tests on feature warping routine correctness
            auto aFeat= aMAP_11.index({Slice(0,None,1),50,50});
            int y_ind=(int) Y_1.index({50,50}).item<double>();
            int x_ind=(int) X_1.index({50,50}).item<double>();
            auto aFeatFromIndices=aFeatMap.index({Slice(0,None,1),y_ind,x_ind});
            auto isEqual=torch::equal(aFeat,aFeatFromIndices);
            MMVII_INTERNAL_ASSERT_strong(isEqual, "PROBLEM WITH FEATURE WARPIGN BEFORE INTERPOLATION !!!");
        }*/

    // INTERPOLATE THE WHOLE FEATURE MAP

    //             InterpValue= ((double)x_2-aLocX)*(aMap11*y_2_y+aMap21*y_y_1)
    //              + (aLocX-(double)x_1)*(aMap12*y_2_y+aMap22*y_y_1);


//                               |aMAP11 aMAP21| |Y_2 - Y|
//   out = |X_2 - X , X - X_1 |  |             | |       |
//                               |aMAP12 aMAP22| |Y - Y_1|

    auto Y_1GEOYT=aGeoYT-Y_1;
    auto Y_2GEOYT=Y_2-aGeoYT;
    auto X_1GEOXT=aGeoXT-X_1;
    auto X_2GEOXT=X_2-aGeoXT;
    Y_2GEOYT.index_put_({Y_2GEOYT==0},1);
    X_2GEOXT.index_put_({X_2GEOXT==0},1);
    auto TERM_1= torch::einsum("ijk,jk->ijk",{aMAP_11,Y_2GEOYT})+torch::einsum("ijk,jk->ijk",{aMAP_21,Y_1GEOYT});
    auto TERM_2= torch::einsum("ijk,jk->ijk",{aMAP_12,Y_2GEOYT})+torch::einsum("ijk,jk->ijk",{aMAP_22,Y_1GEOYT});
    anInterPolFeatMap=torch::einsum("ijk,jk->ijk",{TERM_1,X_2GEOXT})+torch::einsum("ijk,jk->ijk",{TERM_2,X_1GEOXT});

    /*anInterPolFeatMap=(X_2-aGeoXT)*(aMAP_11.mul(Y_2-aGeoYT)+aMAP_21.mul(aGeoYT-Y_1))
            + (aGeoXT-X_1)*(aMAP_12.mul(Y_2-aGeoYT)+aMAP_22.mul(aGeoYT-Y_1)) ;*/

    // SOME CHECKS
    bool oktest=false;

    if (oktest)
        {

            std::cout<<"------------><<<<< CHECK  CORRECTNESS OF BILIN INTERPOL "<<std::endl;
            // Tests on feature warping routine correctness
            auto aFeat11= aMAP_11.index({Slice(0,None,1),50,50});
            auto aFeat21= aMAP_21.index({Slice(0,None,1),50,50});
            auto aFeat12= aMAP_12.index({Slice(0,None,1),50,50});
            auto aFeat22= aMAP_22.index({Slice(0,None,1),50,50});

            double y1_ind=Y_1.index({50,50}).item<double>();
            double x1_ind=X_1.index({50,50}).item<double>();
            double y2_ind=Y_2.index({50,50}).item<double>();
            double x2_ind=X_2.index({50,50}).item<double>();
            double XX=aGeoXT.index({50,50}).item<double>();
            double YY=aGeoYT.index({50,50}).item<double>();

            auto res =(x2_ind-XX)*(aFeat11.mul(y2_ind-YY)+aFeat21.mul(YY-y1_ind))
                    + (XX-x1_ind)*(aFeat12.mul(y2_ind-YY)+aFeat22.mul(YY-y1_ind)) ;
            float diff=torch::sum(res-anInterPolFeatMap.index({Slice(),50,50})).squeeze().item<float>();

            //std::cout<<"  by hand interpolation   ==>  "<<res<<std::endl;
            //std::cout<<"  bulk accessed with index interpol "<<anInterPolFeatMap.index({Slice(),50,50})<<std::endl;
            //std::cout<<"Difference ==> "<<torch::sum(res-anInterPolFeatMap.index({Slice(),50,50}))<<" condition on equality"<<isEqual<<std::endl;

            std::cout<<" Error check by comparing bulk interpolator with by sample one "<<diff<<std::endl;

            MMVII_INTERNAL_ASSERT_tiny(diff>0.001, "PROBLEM WITH FEATURE WARPIGN BEFORE INTERPOLATION !!!");
        }


    return anInterPolFeatMap;
}

torch::Tensor cAppliMatchMultipleOrtho::ResampleFeatureMap(torch::Tensor & aFeatMap, tImOrtho aGeoX, tImOrtho aGeoY)
{
    // Assuming aFeatMap of shape F, H, W
    // Interpolates an embedding at real valued locations
    //cInterpolateurIm2D<float> * anInt
    cPt2di aSzOrtho=aGeoX.DIm().Sz();
    std::cout<<"  ooooooo   "<<aFeatMap.sizes()<<std::endl;

    using namespace torch::indexing;

    auto FeatSz=aFeatMap.size(0);
    auto HFeatMap=aFeatMap.size(1);
    auto WFeatMap=aFeatMap.size(2);

    // Allocate tensor
    //std::cout<<FeatSz<<"   "<<HFeatMap<<"   "<<WFeatMap<<std::endl;
    //(*anInterPolFeatMap)=torch::empty({FeatSz,aSzOrtho.y,aSzOrtho.x}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor anInterPolFeatMap=torch::zeros({FeatSz,aSzOrtho.y()-1,aSzOrtho.x()-1},torch::TensorOptions().dtype(torch::kFloat32));
    //std::cout<<" SDSDSDSDSD  "<<anInterPolFeatMap.sizes()<<std::endl;
    for (int fdim=0;fdim<FeatSz;fdim++)
        {
            // Get a 2d image like slice of the feature map tensor
            auto IM = aFeatMap.index({fdim,Slice(0,None,1),Slice(0,None,1)});
            //std::cout<<" IMIMIMIMIMI  "<<IM.sizes()<<std::endl;
            for (int x=0; x<aSzOrtho.x()-1; x++)
                {
                    for (int y=0;y<aSzOrtho.y()-1;y++)
                        {
                            // for each real valued location, interpolate features at Z
                            cPt2di anInd(x,y);
                            cPt2dr aPIm = cPt2dr( aGeoX.DIm().VD_GetV(anInd), aGeoY.DIm().VD_GetV(anInd));
                            // check if point in IM
                            if (aPIm.x()>0 && aPIm.x()<WFeatMap && aPIm.y()>0 && aPIm.y()<HFeatMap)
                                {
                                    //auto anElem=anInterPolFeatMap.index({fdim,y,x});
                                    //std::cout<<this->Interpol_Bilin(IM,aPIm)<<std::endl;
                                    //auto anElementTensor = torch::tensor({this->Interpol_Bilin(IM,aPIm)});
                                    //std::cout<<anElementTensor<<std::endl;
                                    //std::cout<<" test access to tensor element "<<anElem<<"  and the affected term  "<<torch::tensor(this->Interpol_Bilin(IM,aPIm))<<std::endl;

                                    anInterPolFeatMap.index({fdim,y,x}).copy_(torch::tensor(this->Interpol_Bilin(IM,aPIm)));
                                }
                         }
                }
        }
    return anInterPolFeatMap;
}


torch::Tensor cAppliMatchMultipleOrtho::ComputeEpipolarImage(tImOrtho & aNativeGeomImage, tImOrtho & aGeoX, tImOrtho & aGeoY)
{
  using namespace torch::indexing;
  namespace FFunc=torch::nn::functional;
  cPt2di aSzOrtho= aNativeGeomImage.DIm().Sz();
  cPt2di aSzGeo= aGeoX.DIm().Sz();
  tREAL4 ** mGeoXData=aGeoX.DIm().ExtractRawData2D();
  tREAL4 ** mGeoYData=aGeoY.DIm().ExtractRawData2D();
  tREAL4 ** mNativeData=aNativeGeomImage.DIm().ExtractRawData2D();

  // create offsets tensors for interpolation
  torch::Tensor aGeoXT=torch::from_blob((*mGeoXData), {aSzGeo.y(),aSzGeo.x()},
                                        torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor aGeoYT=torch::from_blob((*mGeoYData), {aSzGeo.y(),aSzGeo.x()},
                                        torch::TensorOptions().dtype(torch::kFloat32));

  torch::Tensor aNativeImT=torch::from_blob((*mNativeData), {aSzOrtho.y(),aSzOrtho.x()},
                                        torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).unsqueeze(0); // 1,1,H,W

   //aGeoXT=aGeoXT.index({Slice(0,aSzOrtho.y()-1,1),Slice(0,aSzOrtho.x()-1,1)});
   //aGeoYT=aGeoYT.index({Slice(0,aSzOrtho.y()-1,1),Slice(0,aSzOrtho.x()-1,1)});

   return FFunc::grid_sample(aNativeImT,
                             torch::stack({aGeoXT,aGeoYT},-1).unsqueeze(0), // 1,h,w,2
                              F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)).squeeze(); //H,W
}

int  cAppliMatchMultipleOrtho::ExeProjectOrigEmbeddings()
{
  torch::Device device(mUseCuda ? torch::kCUDA : torch::kCPU);
   // Parse all Z
   // If using a model (CNN) Initialize the predictor
   if (mArchitecture!="")
   {
        InitializePredictor();
   }

   // load Original images
   bool WithFile = ExistFile(NameORIG(0,0));
   if (WithFile)
       {
           for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
           {
                mORIGIm.push_back(tVecOrtho());
                for (int aKScale=0 ; aKScale<mNbScale ; aKScale++)
                   {
                       mORIGIm.at(aKIm).push_back(tImOrtho::FromFile(NameORIG(aKIm,aKScale)));
                   }
           }
        }

   // compute original embeddings
   std::vector<torch::Tensor> * OrigEmbeddings= new std::vector<torch::Tensor>;
   if (mWithExtCorr)
     {
       for (unsigned int i=0; i<mORIGIm.size();i++)
         {
           cPt2di aSzImOrig=mORIGIm.at(i).at(0).DIm().Sz();
           if (mArchitecture==TheUnetMlpCubeMatcher)
             {
                  OrigEmbeddings->push_back(mCNNPredictor->PredictUnetFeaturesOnly(mMSAFF,mORIGIm.at(i),aSzImOrig));
             }
         }

     }

   // Load Displacement grids in X and Y directions to apply to each embedding
   for (int aZ=0 ; aZ<mNbZ ; aZ++)
   {
        mPrefixZ =  mPrefixGlob + "_Z" + ToStr(aZ);

        bool NoFile = ExistFile(mPrefixZ+ "_NoData");  // If no data in masq thie file exist
        WithFile = ExistFile(NameOrtho(0,0));
        // A little check
        MMVII_INTERNAL_ASSERT_strong(NoFile!=WithFile,"DM4MatchMultipleOrtho, incoherence file");
        if ((aZ==0)  && (true))
        {
             cDataFileIm2D aDF = cDataFileIm2D::Create(NameOrtho(0,0),eForceGray::Yes);
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
                 mVGEOX.push_back(tVecOrtho());
                 mVGEOY.push_back(tVecOrtho());
                 for (int aKScale=0 ; aKScale<mNbScale ; aKScale++)
                    {
                        mVOrtho.at(aKIm).push_back(tImOrtho::FromFile(NameOrtho(aKIm,aKScale)));
                        /*if (aKIm==2)
                          {
                            std::cout<<"NAME GEOX :"<<NameGeoX(aKIm,aKScale)<<std::endl;
                            std::cout<<"NAME GEOY :"<<NameGeoY(aKIm,aKScale)<<std::endl;
                          }*/
                        mVGEOX.at(aKIm).push_back(tImOrtho::FromFile(NameGeoX(aKIm,aKScale)));
                        mVGEOY.at(aKIm).push_back(tImOrtho::FromFile(NameGeoY(aKIm,aKScale)));
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
              mImSimil.DIm().InitCste(1.0);   //  1.0 => correl of -1  (cube is filled with (1-corr)/2.0 when cos and 1-sim when sim)


        if(mWithExtCorr)
        {
          std::vector<torch::Tensor> * ProjectEmbeddings = new std::vector<torch::Tensor>;
              // Inference based correlation
              // Create Embeddings
              // Calculate the EMBEDDINGS ONE TIME USING FOWARD OVER THE WHOLE TILEs
              MMVII_INTERNAL_ASSERT_strong(mArchitecture==TheUnetMlpCubeMatcher, "TheUnetMlpCubeMatcher is the only option for Now");
              if (mArchitecture==TheUnetMlpCubeMatcher)
                 {
                      namespace FFunc=torch::nn::functional;
                      cPt2di aSzIm;
                      cPt2di aSzImOrig;
                      // project all except the master (i=0) embedding
                      MMVII_INTERNAL_ASSERT_strong(mVGEOX.size()>1,"No query image found!");
                      //ProjectEmbeddings->push_back(OrigEmbeddings->at(0));
                      //std::cout<<"##########################  "<<0<<"  "<<ProjectEmbeddings->at(0).sizes()<<"  ###########################"<<std::endl;
                      for (unsigned int i=0; i<mORIGIm.size();i++)
                        {
                          aSzIm=mVOrtho.at(i).at(0).DIm().Sz();
                          aSzImOrig=mORIGIm.at(i).at(0).DIm().Sz();
                          //ProjectEmbeddings->push_back(this->InterpolateFeatMap(OrigEmbeddings->at(i),mVGEOX[i][0],mVGEOY[i][0]));

                          ProjectEmbeddings->push_back(FFunc::grid_sample(OrigEmbeddings->at(i).unsqueeze(0),
                                                                          ToTensorGeo(mVGEOX[i][0],mVGEOY[i][0],aSzImOrig-cPt2di(1,1)).to(device),
                            FFunc::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)).squeeze());

                          //auto OrthoEmbedding=mCNNPredictor->PredictUnetFeaturesOnly(mMSAFF,mVOrtho.at(i),aSzIm);
                          //auto COSINE=at::cosine_similarity(OrthoEmbedding,ProjectEmbeddings->at(i),0).squeeze();
                          //std::cout<<" CORRELATION CHECK >>>  "<<at::mean(COSINE)<<std::endl;
                          //std::cout<<"##########################  "<<i<<"  "<<ProjectEmbeddings->at(i).sizes()<<"  ###########################"<<std::endl;
                        }
                      ComputeSimilByLearnedCorrelMasterEnhanced(ProjectEmbeddings);
                 }

        }
        else
        {
            ComputeSimilByCorrelMaster();
        }

            mImSimil.DIm().ToFile(mPrefixZ+ "_Sim.tif"); // Save similarities
            mVOrtho.clear();
            mVGEOX.clear();
            mVGEOY.clear();
            mVMasq.clear();
        }
   }
   return EXIT_SUCCESS;
}




int  cAppliMatchMultipleOrtho::GotoEpipolar()
{
  torch::Device device(mUseCuda ? torch::kCUDA : torch::kCPU);
   // Parse all Z
   // If using a model (CNN) Initialize the predictor
   if (mArchitecture!="")
   {
        InitializePredictor();
   }

   // load Original images
   bool WithFile = ExistFile(NameORIG(0,0));
   //bool WithEpipolar=ExistFile(NameORIGMASTEREpImGEOX(0,1,0));
   if (WithFile)
       {
           for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
           {
                mORIGIm.push_back(tVecOrtho());
                for (int aKScale=0 ; aKScale<mNbScale ; aKScale++)
                   {
                       mORIGIm.at(aKIm).push_back(tImOrtho::FromFile(NameORIG(aKIm,aKScale)));
                   }
           }
        }

   // load image to epipolar maps and back
    //std::vector<tVecOrtho> mORIG_GEOX, mORIG_GEOY;
    std::vector<tVecOrtho>    mORIG_EpIm_GEOX, mORIG_EpIm_GEOY, mEPIPS;
    //std::vector<tVecMasq>  mORIG_MASQs;
    std::vector<tVecMasq> mORIG_EpIm_MASQs;
    int anNbImUtil=0;
    //if (WithEpipolar)
    {
        int aInd=0;
        for (int akIm=1; akIm<mNbIm ; akIm++ ) // ajouter une image pour le passage aux images épipolaires
          {
            // for each secondary image fille containers twice : one for master eipolar + one for secondary

            // 1. master epipolar
            //mORIG_GEOX.push_back(tVecOrtho());
            //mORIG_GEOY.push_back(tVecOrtho());
            //mORIG_MASQs.push_back(tVecMasq());

            bool IsEpipCalc=ExistFile(NameORIGMASTEREpImGEOX(0,akIm,0));
            //std::cout<< "Fileeee       "<<NameORIGMASTEREpImGEOX(0,akIm,0)<<" EXIST : "<<IsEpipCalc<<std::endl;
            if (IsEpipCalc)
            {
                    mORIG_EpIm_GEOX.push_back(tVecOrtho());
                    mORIG_EpIm_GEOY.push_back(tVecOrtho());
                    mORIG_EpIm_MASQs.push_back(tVecMasq());
                    mEPIPS.push_back(tVecOrtho());

                    // 2. second epipolar

                    aInd+=1;
                    //mORIG_GEOX.push_back(tVecOrtho());
                    //mORIG_GEOY.push_back(tVecOrtho());
                    //mORIG_MASQs.push_back(tVecMasq());
                    mEPIPS.push_back(tVecOrtho());
                    mORIG_EpIm_GEOX.push_back(tVecOrtho());
                    mORIG_EpIm_GEOY.push_back(tVecOrtho());
                    mORIG_EpIm_MASQs.push_back(tVecMasq());

                    for (int aKScale=0; aKScale<mNbScale; aKScale++)
                      {
                        // Master
                        //mORIG_GEOX.at(aInd-1).push_back(tImOrtho::FromFile(NameORIGMASTERGEOX(0,akIm,aKScale)));
                        //mORIG_GEOY.at(aInd-1).push_back(tImOrtho::FromFile(NameORIGMASTERGEOY(0,akIm,aKScale)));
                        //mORIG_MASQs.at(aInd-1).push_back(tImMasq::FromFile(NameORIGMASTERMASQ(0,akIm,aKScale)));

                        mORIG_EpIm_GEOX.at(aInd-1).push_back(tImOrtho::FromFile(NameORIGMASTEREpImGEOX(0,akIm,aKScale)));
                        mORIG_EpIm_GEOY.at(aInd-1).push_back(tImOrtho::FromFile(NameORIGMASTEREpImGEOY(0,akIm,aKScale)));
                        mORIG_EpIm_MASQs.at(aInd-1).push_back(tImMasq::FromFile(NameORIGMASTEREpImMASQ(0,akIm,aKScale)));
                        // Secondary
                        //mORIG_GEOX.at(aInd).push_back(tImOrtho::FromFile(NameORIGSECGEOX(akIm,aKScale)));
                        //mORIG_GEOY.at(aInd).push_back(tImOrtho::FromFile(NameORIGSECGEOY(akIm,aKScale)));
                        //mORIG_MASQs.at(aInd).push_back(tImMasq::FromFile(NameORIGSECMASQ(akIm,aKScale)));

                        mORIG_EpIm_GEOX.at(aInd).push_back(tImOrtho::FromFile(NameORIGSECEpImGEOX(akIm,aKScale)));
                        mORIG_EpIm_GEOY.at(aInd).push_back(tImOrtho::FromFile(NameORIGSECEpImGEOY(akIm,aKScale)));
                        mORIG_EpIm_MASQs.at(aInd).push_back(tImMasq::FromFile(NameORIGSECEpImMASQ(akIm,aKScale)));

                        // Load bare epips
                        mEPIPS.at(aInd-1).push_back(tImOrtho::FromFile(NameMASTEREPIP(0,akIm,aKScale)));
                        mEPIPS.at(aInd).push_back(tImOrtho::FromFile(NameSECEPIP(akIm,0,aKScale)));
                      }
                    aInd+=1;
                    anNbImUtil+=1;
            }
          }
      }


    std::cout<<"ALL  PAIRS OF EPIPS SHAPE "<<mEPIPS.size()<<std::endl;
   // compute original embeddings
   std::vector<torch::Tensor> * OrigEmbeddings= new std::vector<torch::Tensor>;
   //if (WithEpipolar)
     {
       int anIndIm=1;
       for (int anInd=0;anInd<2*anNbImUtil;anInd+=2,anIndIm++)
         {
           /*auto aMasterEpip=this->ComputeEpipolarImage(mORIGIm.at(0).at(0),
                                                       mORIG_GEOX.at(anInd).at(0),
                                                       mORIG_GEOY.at(anInd).at(0));
           auto aSecEpip=this->ComputeEpipolarImage(mORIGIm.at(anIndIm).at(0),
                                                    mORIG_GEOX.at(anInd+1).at(0),
                                                    mORIG_GEOY.at(anInd+1).at(0));
                                                    */
           // compute descriptors on the epipolar images directly

           if (mArchitecture==TheUnetMlpCubeMatcher)
             {
               namespace FFunc=torch::nn::functional;
               cPt2di aSzImEpipMaster=mEPIPS.at(anInd).at(0).DIm().Sz();
               cPt2di aSzImEpipSec =mEPIPS.at(anInd+1).at(0).DIm().Sz();
               auto MasterFeat=mCNNPredictor->PredictUnetFeaturesOnly(mMSAFF,mEPIPS.at(anInd),aSzImEpipMaster);
               auto SecFeat   =mCNNPredictor->PredictUnetFeaturesOnly(mMSAFF,mEPIPS.at(anInd+1),aSzImEpipSec);

               OrigEmbeddings->push_back(FFunc::grid_sample(MasterFeat.unsqueeze(0),
                                                            ToTensorGeo(mORIG_EpIm_GEOX[anInd][0],mORIG_EpIm_GEOY[anInd][0]).to(device),
              F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)));

               OrigEmbeddings->push_back(FFunc::grid_sample(SecFeat.unsqueeze(0),
                                                            ToTensorGeo(mORIG_EpIm_GEOX[anInd+1][0],mORIG_EpIm_GEOY[anInd+1][0]).to(device),
              F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)));


               if (0)
                 {
                   // Save some images rectified back to the original image
                   if (anInd==0)
                     {
                        auto aMasterImage=ComputeEpipolarImage(mEPIPS.at(anInd).at(0),mORIG_EpIm_GEOX[anInd][0],mORIG_EpIm_GEOY[anInd][0]);
                        auto aSecImage   =ComputeEpipolarImage(mEPIPS.at(anInd+1).at(0),mORIG_EpIm_GEOX[anInd+1][0],mORIG_EpIm_GEOY[anInd+1][0]);

                        // Write images
                        Tensor2Tiff(aMasterImage,"./MASTER_IM.tif");
                        Tensor2Tiff(aSecImage,"./SEC_IM.tif");
                     }

                 }

             }
           else
             {
               MMVII_INTERNAL_ASSERT_strong(false,"DM4MatchMultipleOrtho, Model architecture not taken into account !");
             }
         }
     }

   //std::cout<<"EMBEDDINGS IN ORIGINAL IMAGE "<<OrigEmbeddings->size()<<std::endl;

   // Load Displacement grids in X and Y directions to apply to each embedding
   for (int aZ=0 ; aZ<mNbZ ; aZ++)
   {
        mPrefixZ =  mPrefixGlob + "_Z" + ToStr(aZ);

        bool NoFile = ExistFile(mPrefixZ+ "_NoData");  // If no data in masq thie file exist
        WithFile = ExistFile(NameMasq(0,0));
        // A little check
        MMVII_INTERNAL_ASSERT_strong(NoFile!=WithFile,"DM4MatchMultipleOrtho, incoherence file");
        if ((aZ==0)  && (true))
        {
             cDataFileIm2D aDF = cDataFileIm2D::Create(NameMasq(0,0),eForceGray::Yes);
             StdOut() << " * NbI=" << mNbIm << " NbS=" <<  mNbScale << " NbZ=" <<  mNbZ << " Sz=" << aDF.Sz() << " SzW=" << mSzW << "\n";
        }
         if (WithFile)
            {
                // Read  orthos and masq in  vectors of images
                mSzIms = cPt2di(-1234,6789);
                for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
                {
                           //mVOrtho.push_back(tVecOrtho());
                           mVMasq.push_back(tVecMasq());
                           mVGEOX.push_back(tVecOrtho());
                           mVGEOY.push_back(tVecOrtho());
                           for (int aKScale=0 ; aKScale<mNbScale ; aKScale++)
                              {
                                  //mVOrtho.at(aKIm).push_back(tImOrtho::FromFile(NameOrtho(aKIm,aKScale)));
                                  /*if (aKIm==2)
                                    {
                                      std::cout<<"NAME GEOX :"<<NameGeoX(aKIm,aKScale)<<std::endl;
                                      std::cout<<"NAME GEOY :"<<NameGeoY(aKIm,aKScale)<<std::endl;
                                    }*/
                                  mVGEOX.at(aKIm).push_back(tImOrtho::FromFile(NameGeoX(aKIm,aKScale)));
                                  mVGEOY.at(aKIm).push_back(tImOrtho::FromFile(NameGeoY(aKIm,aKScale)));
                                  mVMasq.at(aKIm).push_back(tImMasq::FromFile(NameMasq(aKIm,aKScale)));
                                  if ((aKIm==0) && (aKScale==0))
                                      mSzIms = mVMasq[0][0].DIm().Sz();  // Compute the size at level


                                  // check all images have the same at a given level
                                  //MMVII_INTERNAL_ASSERT_strong(mVOrtho[aKIm][aKScale].DIm().Sz()==mSzIms,"DM4O : variable size(ortho)");
                                  MMVII_INTERNAL_ASSERT_strong(mVMasq[aKIm][aKScale].DIm().Sz()==mSzIms,"DM4O : variable size(masq)");
                              }

                   }
                // Create similarity image with good size
                mImSimil = tImSimil(mSzIms);

                mImSimil.DIm().InitCste(1.0);

                if (mWithExtCorr)
                  {
                    std::vector<torch::Tensor> * ProjectEmbeddings = new std::vector<torch::Tensor>;
                      // Inference based correlation
                      // Create Embeddings
                      // Calculate the EMBEDDINGS ONE TIME USING FOWARD OVER THE WHOLE TILEs
                      MMVII_INTERNAL_ASSERT_strong(mArchitecture==TheUnetMlpCubeMatcher, "TheUnetMlpCubeMatcher is the only option for Now");
                      if (mArchitecture==TheUnetMlpCubeMatcher)
                         {
                              namespace FFunc=torch::nn::functional;
                              cPt2di aSzImOrig=mORIGIm.at(0).at(0).DIm().Sz();
                              MMVII_INTERNAL_ASSERT_strong(mVGEOX.size()>1,"No query image found!");

                              int idImUtil=0;
                              std::vector<bool> RelevantEmbeddings;
                              //for (int i=0; i<2*(mNbIm-1);i+=2,id_im++) // parcourir par couple de paires épipolaires
                              for (int id_im=1; id_im<mNbIm;id_im++) // parcourir par couple de paires épipolaires
                                {
                                // master sec images
                                  bool IsEpipCalcMaster=ExistFile(NameORIGMASTEREpImGEOX(0,id_im,0));
                                  //bool IsEpipCalcSecond=ExistFile(NameORIGSECEpImGEOX(i,0));


                                  if (IsEpipCalcMaster)
                                    {
                                      bool IsRelEmbedding=((OrigEmbeddings->at(idImUtil  ).size(-1)>=32) &&
                                                           (OrigEmbeddings->at(idImUtil  ).size(-2)>=32) &&
                                                           (OrigEmbeddings->at(idImUtil+1).size(-1)>=32) &&
                                                           (OrigEmbeddings->at(idImUtil+1).size(-2)>=32)  );

                                      RelevantEmbeddings.push_back(IsRelEmbedding);
                                      if (IsRelEmbedding)
                                        {
                                          ProjectEmbeddings->push_back(FFunc::grid_sample(OrigEmbeddings->at(idImUtil),
                                                                                          ToTensorGeo(mVGEOX[0][0],mVGEOY[0][0],aSzImOrig-cPt2di(1,1)).to(device),
                                            F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)).squeeze());
                                          //std::cout<<ProjectEmbeddings->at(idImUtil  ).device()<<std::endl;

                                          cPt2di aSzImSec=mORIGIm.at(id_im).at(0).DIm().Sz();
                                          ProjectEmbeddings->push_back(FFunc::grid_sample(OrigEmbeddings->at(idImUtil+1),
                                                                                          ToTensorGeo(mVGEOX[id_im][0],mVGEOY[id_im][0],aSzImSec-cPt2di(1,1)).to(device),
                                            F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)).squeeze());
                                        }

                                      idImUtil+=2;
                                    }
                                }
                              //std::cout<<"ARE ALL RELEVANT "<<RelevantEmbeddings<<std::endl;

                              ComputeSimilByLearnedCorrelMasterEnhancedMVS(ProjectEmbeddings,RelevantEmbeddings);
                         }
                      else
                        {
                          MMVII_INTERNAL_ASSERT_strong(false, "TheUnetMlpCubeMatcher is the only option for Now");
                        }
                  }
                else
                  {
                     ComputeSimilByCorrelMaster();
                  }
            }
            else
            {
             MMVII_INTERNAL_ASSERT_strong(false, "MMV1 Orthos projected in the geometry of master image are not created !");
            }

                mImSimil.DIm().ToFile(mPrefixZ+ "_Sim.tif"); // Save similarities
                //mVOrtho.clear();
                mVGEOX.clear();
                mVGEOY.clear();
                mVMasq.clear();
        }
   //mORIG_GEOX.clear();
   //mORIG_GEOY.clear();
   mORIG_EpIm_GEOX.clear();
   mORIG_EpIm_GEOY.clear();
   mEPIPS.clear();
   //mORIG_MASQs.clear();
   mORIG_EpIm_MASQs.clear();
   //std::cout<<"Get Similarity maps "<<std::getchar();
   return EXIT_SUCCESS;
}

int  cAppliMatchMultipleOrtho::GotoHomography()
{
  torch::Device device(mUseCuda ? torch::kCUDA : torch::kCPU);
   // Parse all Z
   // If using a model (CNN) Initialize the predictor
   if (mArchitecture!="")
   {
        InitializePredictor();
   }

   // load Original images
   bool WithFile = ExistFile(NameORIG(0,0));
   //bool WithHomography=ExistFile(NameORIGSECEpImGEOX(1,0));
   if (WithFile)
     {
         for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
         {
              mORIGIm.push_back(tVecOrtho());
              for (int aKScale=0 ; aKScale<mNbScale ; aKScale++)
                 {
                     mORIGIm.at(aKIm).push_back(tImOrtho::FromFile(NameORIG(aKIm,aKScale)));
                 }
         }
      }
   // load image to HOMOGRAPHY maps and back
    //std::vector<tVecOrtho> mORIG_GEOX, mORIG_GEOY;
    std::vector<tVecOrtho> mORIG_EpIm_GEOX, mORIG_EpIm_GEOY, mEPIPS;
    //std::vector<tVecMasq>  mORIG_MASQs;
    std::vector<tVecMasq> mORIG_EpIm_MASQs;
    int anNbImUtil=0;
    //if (WithHomography)
      {
        for (int akIm=1; akIm<mNbIm ; akIm++ ) // ajouter une image pour le passage aux images épipolaires
          {
            // for each secondary image fille containers twice : one for master eipolar + one for secondary
            // 1. secondary images to homography homography
            //mORIG_GEOX.push_back(tVecOrtho());
            //mORIG_GEOY.push_back(tVecOrtho());
            //mORIG_MASQs.push_back(tVecMasq());
            bool IsHomCalc=ExistFile(NameORIGSECEpImGEOX(akIm,0));
            if (IsHomCalc)
             {
                mEPIPS.push_back(tVecOrtho());
                mORIG_EpIm_GEOX.push_back(tVecOrtho());
                mORIG_EpIm_GEOY.push_back(tVecOrtho());
                mORIG_EpIm_MASQs.push_back(tVecMasq());

                for (int aKScale=0; aKScale<mNbScale; aKScale++)
                  {
                    // Secondary
                    //mORIG_GEOX.at(akIm-1).push_back(tImOrtho::FromFile(NameORIGSECGEOX(akIm,aKScale)));
                    //mORIG_GEOY.at(akIm-1).push_back(tImOrtho::FromFile(NameORIGSECGEOY(akIm,aKScale)));
                    //mORIG_MASQs.at(akIm-1).push_back(tImMasq::FromFile(NameORIGSECMASQ(akIm,aKScale)));

                    mORIG_EpIm_GEOX.at(anNbImUtil).push_back(tImOrtho::FromFile(NameORIGSECEpImGEOX(akIm,aKScale)));
                    mORIG_EpIm_GEOY.at(anNbImUtil).push_back(tImOrtho::FromFile(NameORIGSECEpImGEOY(akIm,aKScale)));
                    mORIG_EpIm_MASQs.at(anNbImUtil).push_back(tImMasq::FromFile(NameORIGSECEpImMASQ(akIm,aKScale)));
                    // Load bare homography warped tiles
                    mEPIPS.at(anNbImUtil).push_back(tImOrtho::FromFile(NameSECEPIP(akIm,0,aKScale)));
                  }
                anNbImUtil+=1;
            }
        }
      }

   // compute original embeddings
   std::vector<torch::Tensor> * OrigEmbeddings= new std::vector<torch::Tensor>;
   //if (WithHomography)
     {
       cPt2di aSzImOrig=mORIGIm.at(0).at(0).DIm().Sz();
       auto MasterFeat=mCNNPredictor->PredictUnetFeaturesOnly(mMSAFF,mORIGIm.at(0),aSzImOrig);
       OrigEmbeddings->push_back(MasterFeat.squeeze());
       for (int anInd=0;anInd<anNbImUtil;anInd++)
         {
           if (mArchitecture==TheUnetMlpCubeMatcher)
             {
               namespace FFunc=torch::nn::functional;
               cPt2di aSzImEpipSec =mEPIPS.at(anInd).at(0).DIm().Sz();
               auto SecFeat   =mCNNPredictor->PredictUnetFeaturesOnly(mMSAFF,mEPIPS.at(anInd),aSzImEpipSec);
               OrigEmbeddings->push_back(FFunc::grid_sample(SecFeat.unsqueeze(0),
                                                            ToTensorGeo(mORIG_EpIm_GEOX[anInd][0],mORIG_EpIm_GEOY[anInd][0]).to(device),
              F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)).squeeze());
               if (0)
                 {
                   // Save some images rectified back to the original image
                   if (anInd==0)
                     {
                        auto aSecImage =ComputeEpipolarImage(mEPIPS.at(anInd).at(0),mORIG_EpIm_GEOX[anInd][0],mORIG_EpIm_GEOY[anInd][0]);

                        // Write images
                        //Tensor2Tiff(aMasterImage,"./MASTER_IM.tif");
                        Tensor2Tiff(aSecImage,"./SEC_IM.tif");
                     }
                 }
             }
           else
             {
               MMVII_INTERNAL_ASSERT_strong(false,"DM4MatchMultipleOrtho, Model architecture not taken into account !");
             }
         }
       std::cout<<"COMPUTED EMBEDDINGS =====>   ==============    "<<OrigEmbeddings->size()<<std::endl;
     }

   // Load Displacement grids in X and Y directions to apply to each embedding
   for (int aZ=0 ; aZ<mNbZ ; aZ++)
   {
        mPrefixZ =  mPrefixGlob + "_Z" + ToStr(aZ);

        bool NoFile = ExistFile(mPrefixZ+ "_NoData");  // If no data in masq thie file exist
        WithFile = ExistFile(NameMasq(0,0));
        // A little check
        MMVII_INTERNAL_ASSERT_strong(NoFile!=WithFile,"DM4MatchMultipleOrtho, incoherence file");
        if ((aZ==0)  && (true))
        {
             cDataFileIm2D aDF = cDataFileIm2D::Create(NameMasq(0,0),eForceGray::Yes);
             StdOut() << " * NbI=" << mNbIm << " NbS=" <<  mNbScale << " NbZ=" <<  mNbZ << " Sz=" << aDF.Sz() << " SzW=" << mSzW << "\n";
        }

        if (WithFile)
          {
                // Read  orthos and masq in  vectors of images
                mSzIms = cPt2di(-1234,6789);
                for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
                {
                     //mVOrtho.push_back(tVecOrtho());
                     mVMasq.push_back(tVecMasq());
                     mVGEOX.push_back(tVecOrtho());
                     mVGEOY.push_back(tVecOrtho());
                     for (int aKScale=0 ; aKScale<mNbScale ; aKScale++)
                        {
                            //mVOrtho.at(aKIm).push_back(tImOrtho::FromFile(NameOrtho(aKIm,aKScale)));
                            /*if (aKIm==2)
                              {
                                std::cout<<"NAME GEOX :"<<NameGeoX(aKIm,aKScale)<<std::endl;
                                std::cout<<"NAME GEOY :"<<NameGeoY(aKIm,aKScale)<<std::endl;
                              }*/
                            mVGEOX.at(aKIm).push_back(tImOrtho::FromFile(NameGeoX(aKIm,aKScale)));
                            mVGEOY.at(aKIm).push_back(tImOrtho::FromFile(NameGeoY(aKIm,aKScale)));
                            mVMasq.at(aKIm).push_back(tImMasq::FromFile(NameMasq(aKIm,aKScale)));
                            if ((aKIm==0) && (aKScale==0))
                                mSzIms = mVMasq[0][0].DIm().Sz();  // Compute the size at level

                            // check all images have the same size at a given level
                            //MMVII_INTERNAL_ASSERT_strong(mVOrtho[aKIm][aKScale].DIm().Sz()==mSzIms,"DM4O : variable size(ortho)");
                            MMVII_INTERNAL_ASSERT_strong(mVMasq[aKIm][aKScale].DIm().Sz()==mSzIms,"DM4O : variable size(masq)");
                        }
                }
                // Create similarity image with good size
                mImSimil = tImSimil(mSzIms);
                //std::cout<<"mSzIms   ::::::  "<<mSzIms<<std::endl;

                mImSimil.DIm().InitCste(1.0);

                if (mWithExtCorr)
                  {
                    std::vector<torch::Tensor> * ProjectEmbeddings = new std::vector<torch::Tensor>;
                      // Inference based correlation
                      // Create Embeddings
                      // Calculate the EMBEDDINGS ONE TIME USING FOWARD OVER THE WHOLE TILEs
                      MMVII_INTERNAL_ASSERT_strong(mArchitecture==TheUnetMlpCubeMatcher, "TheUnetMlpCubeMatcher is the only option for Now");
                      if (mArchitecture==TheUnetMlpCubeMatcher)
                         {
                          namespace FFunc=torch::nn::functional;
                          cPt2di aSzImOrig;
                          // project all except the master (i=0) embedding
                          MMVII_INTERNAL_ASSERT_strong(mVGEOX.size()>1,"No query image found!");
                          //ProjectEmbeddings->push_back(OrigEmbeddings->at(0));
                          //std::cout<<"##########################  "<<0<<"  "<<ProjectEmbeddings->at(0).sizes()<<"  ###########################"<<std::endl;
                          int idIndUtil=0;
                          std::vector<bool> RelevantEmbeddings;
                          for (int i=0; i<mNbIm;i++)
                            {
                              aSzImOrig=mORIGIm.at(i).at(0).DIm().Sz();
                              //ProjectEmbeddings->push_back(this->InterpolateFeatMap(OrigEmbeddings->at(i),mVGEOX[i][0],mVGEOY[i][0]));

                              if (i==0)
                                {
                              ProjectEmbeddings->push_back(FFunc::grid_sample(OrigEmbeddings->at(i).unsqueeze(0),
                                                                              ToTensorGeo(mVGEOX[i][0],mVGEOY[i][0],aSzImOrig-cPt2di(1,1)).to(device),
                                FFunc::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)).squeeze());

                              idIndUtil++;
                                }
                              else
                                {
                                  bool isHomCalc=ExistFile(NameORIGSECEpImGEOX(i,0));
                                  if (isHomCalc)
                                    {
                                      bool IsRelEmbedding=((OrigEmbeddings->at(idIndUtil  ).size(-1)>=32) &&
                                                           (OrigEmbeddings->at(idIndUtil  ).size(-2)>=32) );

                                      //std::cout<<"RELEVANT EMBEDDINGS "<<OrigEmbeddings->at(idIndUtil  ).sizes()<<std::endl;
                                      RelevantEmbeddings.push_back(IsRelEmbedding);

                                      if(IsRelEmbedding)
                                        {

                                          ProjectEmbeddings->push_back(FFunc::grid_sample(OrigEmbeddings->at(idIndUtil).unsqueeze(0),
                                                                                          ToTensorGeo(mVGEOX[i][0],mVGEOY[i][0],aSzImOrig-cPt2di(1,1)).to(device),
                                            FFunc::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)).squeeze());

                                        }
                                      idIndUtil++;

                                    }
                                }


                              //auto OrthoEmbedding=mCNNPredictor->PredictUnetFeaturesOnly(mMSAFF,mVOrtho.at(i),aSzIm);
                              //auto COSINE=at::cosine_similarity(OrthoEmbedding,ProjectEmbeddings->at(i),0).squeeze();
                              //std::cout<<" CORRELATION CHECK >>>  "<<at::mean(COSINE)<<std::endl;
                              //std::cout<<"##########################  "<<i<<"  "<<ProjectEmbeddings->at(i).sizes()<<"  ###########################"<<std::endl;
                            }
                              //std::cout<<"PROJECTED EMBEDDINGS     ===================>   "<<ProjectEmbeddings->size()<<" RELEVANT EMBEDDINGS "<<RelevantEmbeddings<<std::endl;
                              ComputeSimilByLearnedCorrelMasterEnhancedHomMV(ProjectEmbeddings,RelevantEmbeddings);
                         }
                      else
                        {
                          MMVII_INTERNAL_ASSERT_strong(false, "TheUnetMlpCubeMatcher is the only option for Now");
                        }


                  }
                else
                  {
                     ComputeSimilByCorrelMaster();
                  }
            }
            else
            {
             MMVII_INTERNAL_ASSERT_strong(false, "MMV1 Orthos projected in the geometry of master image are not created !");
            }

                mImSimil.DIm().ToFile(mPrefixZ+ "_Sim.tif"); // Save similarities
                //mVOrtho.clear();
                mVGEOX.clear();
                mVGEOY.clear();
                mVMasq.clear();
        }
   //mORIG_GEOX.clear();
   //mORIG_GEOY.clear();
   mORIG_EpIm_GEOX.clear();
   mORIG_EpIm_GEOY.clear();
   mEPIPS.clear();
   //mORIG_MASQs.clear();
   mORIG_EpIm_MASQs.clear();
   return EXIT_SUCCESS;
}

int cAppliMatchMultipleOrtho::Exe()
{
  //int aResol=std::atoi(mResol.c_str());
  if (mUseEpip)
    {
      return GotoEpipolar();
    }
  else
    {
      return GotoHomography();
    }
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
#endif