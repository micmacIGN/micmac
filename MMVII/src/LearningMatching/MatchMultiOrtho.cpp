#include "include/MMVII_all.h"

// include model architecture 
#include "cCnnModelPredictor.h"

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
	std::string NameOrtho(int aKIm,int aKScale) const {return NameIm(aKIm,aKScale,"O");}
	std::string NameMasq(int aKIm,int aKScale) const {return NameIm(aKIm,aKScale,"M");}

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	//  One option, to replace by whatever you want
	void ComputeSimilByCorrelMaster();
    void ComputeSimilByLearnedCorrelMaster(std::vector<torch::Tensor> * AllEmbeddings);
    void ComputeSimilByLearnedCorrelMasterEnhanced(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    void ComputeSimilByLearnedCorrelMasterMaxMoy(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    void ComputeSimilByLearnedCorrelMasterMaxMoyMulScale(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    void ComputeSimilByLearnedCorrelMasterDempsterShafer(std::vector<torch::Tensor> * AllOrthosEmbeddings);
    tREAL4 ComputeConflictBetween2SEts(tREAL4 aCorrel1, tREAL4 aCorrel2, tREAL4 aPonder1,tREAL4 aPonder2);
    tREAL4 ComputeJointMassBetween2Sets(tREAL4 aCorrel1, tREAL4 aCorrel2, tREAL4 aPonder1,tREAL4 aPonder2);
	void CorrelMaster(const cPt2di &,int aKIm,bool & AllOk,float &aWeight,float & aCorrel);
    void MakeNormalizedIms();
    void InitializePredictor ();
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
    bool                  mWithIntCorr=true;  // initialized in the begining 
    bool                  mWithExtCorr=false;  // initialized in the begining 
    std::string           mArchitecture;
    std::string           mResol;
    std::string           mModelBinDir;
    cPt2di                mCNNWin;
    
    // Networks architectures 
    ConvNet_Fast mNetFastStd= ConvNet_Fast(3,4);  // Conv Kernel= 3x3 , Convlayers=4
    ConvNet_FastBn  mNetFastMVCNN=ConvNet_FastBn(3,7);// Conv Kernel= 3x3 , Convlayers=7
    ConvNet_FastBnRegister mNetFastMVCNNReg=ConvNet_FastBnRegister(3,5,1,112,torch::kCPU);// changed from 64 to 112
    Fast_ProjectionHead mNetFastPrjHead=Fast_ProjectionHead(3,5,1,1,112,112,64,torch::kCPU);
    //MSNet_Attention mMSNet=MSNet_Attention(32);
    //MSNetHead mMSNet=MSNetHead(32);
    torch::jit::script::Module mMSNet /* MSNet_AttentionCustom mMSNet=MSNet_AttentionCustom(32)*/;
    FastandHead mNetFastMVCNNMLP=FastandHead(3,5,4,1,184,184,9,64,torch::kCPU);
    SimilarityNet mNetFastMVCNNDirectSIM=SimilarityNet(3,5,4,1,184,184,64,torch::kCPU);
    //FastandHead mNetFastMVCNNMLP; // Fast MVCNN + MLP for Multiview Features Aggregation
    // LATER SLOW NET 
    ConvNet_Slow mNetSlowStd=ConvNet_Slow(3,4,4); // Conv Kernel= 3x3 , Convlayers=4, Fully Connected Layers =4

	std::vector<tVecOrtho>      mVOrtho;    // vector of loaded ortho at a given Z
	std::vector<tVecMasq>       mVMasq;     // vector of loaded masq  at a given Z
};

// ARCHITECTURES OF CNN TRAINED 
static const std::string TheFastArch = "MVCNNFast";
static const std::string TheFastArchReg = "MVCNNFastReg";
static const std::string TheFastandPrjHead = "MVCNNFastProjHead";
static const std::string TheFastStandard = "MCNNStd";
static const std::string TheFastArchWithMLP= "MVCNNFastMLP";
static const std::string TheFastArchDirectSim="MVCNNFastDirectSIM";
//static const std::string TheMSNet="MSNetHead";
static const std::string TheMSNet="MSNet_Attention";
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
          << AOpt2007(mModelBinDir,"CNNParams" ,"Model Directory : Contient des fichiers binaires *.bin")
          << AOpt2007(mArchitecture,"CNNArch" ,"Model architecture : "+TheFastArch+" || "+TheFastStandard+" || "+TheFastArchWithMLP)
          << AOpt2007(mResol,"RESOL" ,"RESOL OPTION FOR THE MULTISCALE TRAINING: ")
   ;
}


void cAppliMatchMultipleOrtho::InitializePredictor ()
{
    StdOut()<<"MODEL ARCHITECTURE:: "<<mArchitecture<<"\n";
    bool IsArchWellDefined=false;
    IsArchWellDefined = (mArchitecture==TheFastArch) ||  (mArchitecture==TheFastArchReg) ||  (mArchitecture==TheFastStandard) || (mArchitecture==TheFastArchWithMLP) || (mArchitecture==TheFastArchDirectSim) || (mArchitecture==TheFastandPrjHead)|| (mArchitecture==TheMSNet) ;
    MMVII_INTERNAL_ASSERT_strong(IsArchWellDefined,"The network architecture should be specified : "+TheFastArch+" || "+TheFastStandard 
        +" || "+TheFastArchWithMLP+" || "+TheFastArchDirectSim+ " !");      
    MMVII_INTERNAL_ASSERT_strong(this->mModelBinDir!=""," Model params dir must be specified ! ");
    
    mWithExtCorr = (mArchitecture!="");
    
    if (mWithExtCorr)
    {
        // ARCHITECTURE and Location of Model Binaries 
        if(mArchitecture==TheFastArch)
        {
            mCNNPredictor = new aCnnModelPredictor(TheFastArch,mModelBinDir);
            // CREATE AN INSTANCE OF THE NETWORK 
            torch::Device device(torch::kCPU);
            mNetFastMVCNN->createModel(184,7,1,3,device); // becareful to change these values with respect to network architecture
				
				
            // Populate layers by learned weights and biases 
            mCNNPredictor->PopulateModelFromBinaryWithBN(mNetFastMVCNN);
            
            mCNNWin=mCNNPredictor->GetWindowSizeBN(mNetFastMVCNN);
				
            //Add padding to maintain the same size as output 
            auto Fast=mNetFastMVCNN->getFastSequential(); 
            
            // ACTIVATE PADDING (NOW DEACTIVATED)
            
            size_t Sz=Fast->size();
            size_t cc=0;
            for (cc=0;cc<Sz;cc++)
            {
                std::string LayerName=Fast->named_children()[cc].key();
                if (LayerName.rfind(std::string("conv"),0)==0)
                {   //torch::nn::Conv2dImpl *mod=Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>();
                        //std::cout<<"condition verified on name of convolution "<<std::endl;
                    Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>()->options.padding()=1;
                }
            }
        }
        if(mArchitecture==TheFastArchReg)
        {
            mCNNPredictor = new aCnnModelPredictor(TheFastArchReg,mModelBinDir);
            mCNNPredictor->PopulateModelFromBinaryWithBNReg(mNetFastMVCNNReg);
            
            //mCNNWin=mCNNPredictor->GetWindowSizeBNReg(mNetFastMVCNNReg);  just changed to test
			mCNNWin=cPt2di(7,7);	
            //Add padding to maintain the same size as output 
            auto Fast=mNetFastMVCNNReg->mFast; 
            size_t Sz=Fast->size();
            size_t cc=0;
            for (cc=0;cc<Sz;cc++)
            {
                std::string LayerName=Fast->named_children()[cc].key();
                if (LayerName.rfind(std::string("conv"),0)==0)
                {  
                    Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>()->options.padding()=1;
                }
            }
        }
        if(mArchitecture==TheFastandPrjHead)
        {
            mCNNPredictor = new aCnnModelPredictor(TheFastandPrjHead,mModelBinDir);
            mCNNPredictor->PopulateModelPrjHead(mNetFastPrjHead);
            
            mCNNWin=mCNNPredictor->GetWindowSizePrjHead(mNetFastPrjHead);
				
            //Add padding to maintain the same size as output 
            auto Fast=mNetFastPrjHead->mFast; 
            size_t Sz=Fast->size();
            size_t cc=0;
            for (cc=0;cc<Sz;cc++)
            {
                std::string LayerName=Fast->named_children()[cc].key();
                if (LayerName.rfind(std::string("conv"),0)==0)
                {  
                    Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>()->options.padding()=1;
                }
            }
        }
        if(mArchitecture==TheMSNet)
        { 
            mCNNPredictor = new aCnnModelPredictor(TheMSNet,mModelBinDir);
            mCNNPredictor->PopulateModelMSNetHead(mMSNet);
    
            mCNNWin=cPt2di(7,7); // The chosen window size is 7x7
            //Add padding to maintain the same size as output 
               /* auto common=mMSNet->common; 
                for (auto& module : common->children())
                {
                    if(auto* conv2d = module->as<torch::nn::Conv2d>())
                        {
                            conv2d->as<torch::nn::Conv2dImpl>()->options.padding()=1;
                        }
                }*/
            
        }
        if(mArchitecture==TheFastArchWithMLP)
        {
            mCNNPredictor = new aCnnModelPredictor(TheFastArchWithMLP,mModelBinDir);
            // CREATE AN INSTANCE OF THE NETWORK 
            //auto cuda_available = torch::cuda::is_available();
            //torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
            //mNetFastMVCNNMLP=FastandHead(3,7,4,1,184,184,3,64,device); // not to change for the moment 
            //mNetFastMVCNNMLP->to(devicecuda);
            // Populate layers by learned weights and biases 
            mCNNPredictor->PopulateModelFastandHead(mNetFastMVCNNMLP);
            StdOut()<<"MODEL LOADED-------> "<<"\n";
            //mNetFastMVCNNMLP->to(torch::kCPU);
            mCNNWin=mCNNPredictor->GetWindowSizeFastandHead(mNetFastMVCNNMLP);
				
            //Add padding to maintain the same size as output 
            auto Fast=mNetFastMVCNNMLP->mFast; 
            
            // ACTIVATE PADDING (NOW DEACTIVATED)
            
            size_t Sz=Fast->size();
            size_t cc=0;
            for (cc=0;cc<Sz;cc++)
            {
                std::string LayerName=Fast->named_children()[cc].key();
                if (LayerName.rfind(std::string("conv"),0)==0)
                {   //torch::nn::Conv2dImpl *mod=Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>();
                        //std::cout<<"condition verified on name of convolution "<<std::endl;
                    Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>()->options.padding()=1;
                }
            }
        }
        if(mArchitecture==TheFastArchDirectSim)
        {
            mCNNPredictor = new aCnnModelPredictor(TheFastArchDirectSim,mModelBinDir);
            StdOut()<<"LOADING NETWORKKKK   ------> "<<"\n";
            mCNNPredictor->PopulateModelSimNet(mNetFastMVCNNDirectSIM);
            StdOut()<<"MODEL LOADED-SIMILARITY NETWORK    ------> "<<"\n";
            //mNetFastMVCNNMLP->to(torch::kCPU);
            mCNNWin=mCNNPredictor->GetWindowSizeSimNet(mNetFastMVCNNDirectSIM);
				
            //Add padding to maintain the same size as output 
            auto Fast=mNetFastMVCNNDirectSIM->mFast; 
            
            // ACTIVATE PADDING (NOW DEACTIVATED)
            
            size_t Sz=Fast->size();
            size_t cc=0;
            for (cc=0;cc<Sz;cc++)
            {
                std::string LayerName=Fast->named_children()[cc].key();
                if (LayerName.rfind(std::string("conv"),0)==0)
                {   //torch::nn::Conv2dImpl *mod=Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>();
                        //std::cout<<"condition verified on name of convolution "<<std::endl;
                    Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>()->options.padding()=1;
                }
            }
        }
        else if (mArchitecture==TheFastStandard)
        {
           mCNNPredictor = new aCnnModelPredictor(TheFastStandard,mModelBinDir); 
           
           // CREATE A CNN MODULE AND LOAD PARAMS 
            mNetFastStd->createModel(64,4,1,3);
            
            // Populate layers by learned weights and biases 
            mCNNPredictor->PopulateModelFromBinary(mNetFastStd);
            mCNNWin=mCNNPredictor->GetWindowSize(mNetFastStd);
            //Add padding to maintain the same size as input
            auto Fast=mNetFastStd->getFastSequential(); 
            
            size_t Sz=Fast->size();
            size_t cc=0;
            for (cc=0;cc<Sz;cc++)
            {
                std::string LayerName=Fast->named_children()[cc].key();
                if (LayerName.rfind(std::string("conv"),0)==0)
                    
                    {   //torch::nn::Conv2dImpl *mod=Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>();
                        //std::cout<<"condition verified on name of convolution "<<std::endl;
                        Fast->named_children()[cc].value().get()->as<torch::nn::Conv2dImpl>()->options.padding()=1;
                    }
	         }   // DO NOT APPLY PADDING TO GET A VECTOR EMBDEDDING OF THE PATCH 
	         // PADDING IS USED WHENEVER THE WHOLE TILE IS CONCERNED 
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
        //std::cout<<"correl val "<<aCorrel<<std::endl;
        /*if (smpl){
            std::cout<<"correl val "<<aCorrel<<std::endl;
            smpl--;
        }*/
        //std::cout<<" CORREL "<<aCorrel<<std::endl;
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
        auto aCrossProd=at::cosine_similarity(AllOrthosEmbeddings->at(0),AllOrthosEmbeddings->at(k),0).squeeze();
        AllSimilarities->push_back(aCrossProd);
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
        std::cout<<" slave vector "<<aSim<<std::endl;
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

   // delete All Similarities 
   delete AllSimilarities;
}


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
        //StdOut()<<"Cross Product values "<<aCrossProd.min()<<"   "<<aCrossProd.max()<<"\n";
        AllSimilarities->push_back(aCrossProd);
    }
    // Free all ortho OneOrthoEmbeding 
    delete AllOrthosEmbeddings; 

    //std::cout<<" feature vector size : "<<FeatSize<<std::endl;
   for (const auto & aP : aDImSim)
   {
	 // Parse secondary images 
    tREAL4 aTab[mNbIm-1];
    tREAL4 aPonder[mNbIm-1];
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
        //std::cout<<" slave vector "<<aSim<<std::endl;
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
    tREAL4 aTab[(mNbIm-1)*4];
    tREAL4 aPonder[(mNbIm-1)*4];
    
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
	 // Parse secondary images 
    tREAL4 aTab[mNbIm-1];
    tREAL4 aPonder[mNbIm-1];
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



int  cAppliMatchMultipleOrtho::Exe()
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

	    
	    // NORMALIZE IF LEARNING BASED CORRELATION 
        if (mWithExtCorr)
        {
           // MakeNormalizedIms();
        
        }
    
	    // Create similarity image with good size
	    mImSimil = tImSimil(mSzIms);
	    mImSimil.DIm().InitCste(2.0);   //  2 => correl of -1
    

        if(mWithExtCorr)
        {
            // Inference based correlation 
            std::vector<torch::Tensor> * OrthosEmbeddings= new std::vector<torch::Tensor>;
            // Create Embeddings 
            // Calculate the EMBEDDINGS ONE TIME USING FOWARD OVER THE WHOLE TILEs
            for (unsigned int i=0;i<mVOrtho.size();i++)
            {
                cPt2di aSzOrtho=mVOrtho.at(i).at(0).DIm().Sz();
                // Initialize Feature Size ;
                /*int FeatSize=0;
                if (mArchitecture==TheFastStandard) FeatSize=64 ;
                if (mArchitecture==TheFastArchReg) FeatSize=64 ;  // change here to account for a new model 
                if (mArchitecture==TheFastandPrjHead) FeatSize=112 ; 
                if (mArchitecture==TheFastArch)   FeatSize=184;
                if (mArchitecture==TheFastArchWithMLP)   FeatSize=184;
                if (mArchitecture==TheFastArchDirectSim)   FeatSize=184;*/
                //  Initialize Embeddings and store them for correlation calculus 
                //auto OneOrthoEmbeding=torch::empty({1,FeatSize,aSzOrtho.y(),aSzOrtho.x()},torch::TensorOptions().dtype(torch::kFloat32));
                
                torch::Tensor OneOrthoEmbeding;
                if (mArchitecture==TheFastStandard)
                    {
                       OneOrthoEmbeding=mCNNPredictor->PredictTile(mNetFastStd,mVOrtho.at(i).at(0),aSzOrtho);
                    }
                else if (mArchitecture==TheFastArch)
                    {
                       OneOrthoEmbeding=mCNNPredictor->PredictWithBNTile(mNetFastMVCNN,mVOrtho.at(i).at(0),aSzOrtho);
                    }
                else if (mArchitecture==TheFastArchReg)
                    {
                        OneOrthoEmbeding=mCNNPredictor->PredictWithBNTileReg(mNetFastMVCNNReg,mVOrtho.at(i).at(0),aSzOrtho);
                    }
                else if (mArchitecture==TheFastandPrjHead)
                    {
                       // std::cout<<"ORTOHS SIZES :  ====> "<<aSzOrtho<<std::endl;
                        OneOrthoEmbeding=mCNNPredictor->PredictPrjHead(mNetFastPrjHead,mVOrtho.at(i).at(0),aSzOrtho);
                    }
                else if (mArchitecture==TheMSNet)
                    {
                      //  OneOrthoEmbeding=mCNNPredictor->PredictMSNetCommon(mMSNet,mVOrtho.at(i),aSzOrtho);
                       // OneOrthoEmbeding=mCNNPredictor->PredictMSNet(mMSNet,mVOrtho.at(i),aSzOrtho);
                        
                        /*int Resol=1;
                        if (mResol!="")
                        {
                            Resol=std::atoi(mResol.c_str());
                        auto CommonEmbedding=mCNNPredictor->PredictMSNet(mMSNet,mVOrtho.at(i),aSzOrtho);
                        switch (Resol)
                        {
                            case 1:
                                OneOrthoEmbeding=mCNNPredictor->PredictMSNet1(mMSNet,CommonEmbedding);
                                break;
                            case 2:
                                OneOrthoEmbeding=mCNNPredictor->PredictMSNet2(mMSNet,CommonEmbedding);
                                break;
                            case 4:
                                OneOrthoEmbeding=mCNNPredictor->PredictMSNet3(mMSNet,CommonEmbedding);
                                break;
                            case 8:
                                OneOrthoEmbeding=mCNNPredictor->PredictMSNet4(mMSNet,CommonEmbedding);
                                break;
                            default:
                                // Full Resolution Inference 
                                OneOrthoEmbeding=mCNNPredictor->PredictMSNet1(mMSNet,CommonEmbedding);
                                break;
                        }*/
                    
                        OneOrthoEmbeding=mCNNPredictor->PredictMSNetHead(mMSNet,mVOrtho.at(i),aSzOrtho);
                    }
                else if (mArchitecture==TheFastArchWithMLP)
                    {
                        OneOrthoEmbeding=mCNNPredictor->PredictFastWithHead(mNetFastMVCNNMLP,mVOrtho.at(i).at(0),aSzOrtho);
                    }
                else if (mArchitecture==TheFastArchDirectSim)
                    {
                        OneOrthoEmbeding=mCNNPredictor->PredictSimNetConv(mNetFastMVCNNDirectSIM,mVOrtho.at(i).at(0),aSzOrtho);
                    }
                
                 StdOut()  <<" EMBEDDING FOR VECTOR OR FULL RESOLUTION ORTHO : "<<i<<OneOrthoEmbeding.sizes()<<"\n";
            
                // store in relevant vector 
                OrthosEmbeddings->push_back(OneOrthoEmbeding);
            }
            //StdOut()  <<" Size OF EMBEDDINGS MS : " <<OrthosEmbeddings->size()<<"\n";
            
            // GIVEN THE ORTHOS EMBEDDINGS, Compute Correlation for each pixel in the similarity image => index work to get vectors from from tensors 
            if (mArchitecture==TheMSNet)
            {
                //ComputeSimilByLearnedCorrelMasterMaxMoyMulScale(OrthosEmbeddings); // Size 4*numberofOrthos
                ComputeSimilByLearnedCorrelMasterMaxMoy(OrthosEmbeddings);
            }
            else
            {
               ComputeSimilByLearnedCorrelMasterMaxMoy(OrthosEmbeddings); 
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
