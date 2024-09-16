
#include "MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"

//#include "include/MMVII_Tpl_Images.h"
//#include "include/MMVII_TplLayers3D.h"
#include <thread>
// included model cnn 
#include "cCnnModelPredictor.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_TplLayers3D.h"


/*
   C = (1-(1-L) ^2)
*/

void InterpolateCorrRatio(std::vector<double> aX,std::vector<double> aF, double& value)
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

bool sortspecific(const std::pair<int,std::pair<torch::Tensor,int>> &a,
              const std::pair<int,std::pair<torch::Tensor,int>> &b)
{
    return (a.first < b.first);
}

namespace MMVII
{

namespace  cNS_FillCubeCost
{

class cAppliFillCubeCost;
struct cOneModele
{
    public :
        typedef cIm2D<tREAL4>  tImRad; 
        cOneModele
        (
            const std::string & aNameModele,
            cAppliFillCubeCost  & aAppliLearn
            //aCnnModelPredictor  & aPredicor
        );
        /*cOneModele
        (
            const std::string & aNameModele,
            cAppliFillCubeCost  & aAppliLearn,
            const std::string & aModelBinDir,
            const std::string & aModelArch
            //aCnnModelPredictor  & aPredicor
        );*/

        double ComputeCost(bool &Ok,const cPt2di & aPC1,const cPt2di & aPC2,int aDZ) const;
        void CalcCorrelExterneTerm(const cBox2di & aBoxInitIm1,int aPxMin,int aPxMax);
        void CalcCorrelExterneRecurs(const cBox2di & aBoxIm1);
        void CalcCorrelExterne();
        
        
       // ADDED METHODS FOR MVCNN
        void CalcCorrelMvCNN();
        void UseZInfZSup();

        cAppliFillCubeCost  * mAppli;
        std::string           mNameModele;
        bool                  mWithIntCorr;
        bool                  mWithExtCorr;
        // ADDED MVCNN
        bool                  mWIthMVCNNCorr;
	    bool                  mWithStatModele;
        cHistoCarNDim         mModele;
        cPyr1ImLearnMatch *   mPyrL1;
        cPyr1ImLearnMatch *   mPyrL2;
        int                   mSzW;
        cPt2di                mPSzW;
        
        // instantiate a pointer to null CNN PREDICTOR
        aCnnModelPredictor * mCNNPredictor=nullptr;
        std::string mArchitecture ="";
        std::string mModelBinDir="";
        cPt2di              mCNNWin;
        
        // Networks architectures 
        ConvNet_Fast mNetFastStd= ConvNet_Fast(3,4);  // Conv Kernel= 3x3 , Convlayers=4
        ConvNet_FastBn  mNetFastMVCNN=ConvNet_FastBn(3,5);// Conv Kernel= 3x3 , Convlayers=7
        ConvNet_FastBnRegister mNetFastMVCNNReg=ConvNet_FastBnRegister(3,5,1,64,torch::kCPU);
        Fast_ProjectionHead mNetFastPrjHead=Fast_ProjectionHead(3,5,1,1,112,112,64,torch::kCPU);
        FastandHead mNetFastMVCNNMLP=FastandHead(3,5,4,1,184,184,9,64,torch::kCPU);
        SimilarityNet mNetFastMVCNNDirectSIM=SimilarityNet(3,5,4,1,184,184,64,torch::kCPU);
        //FastandHead mNetFastMVCNNMLP; // Fast MVCNN + MLP for Multiview Features Aggregation
        // LATER SLOW NET 
        ConvNet_Slow mNetSlowStd=ConvNet_Slow(3,4,4); // Conv Kernel= 3x3 , Convlayers=4, Fully Connected Layers =4
        torch::jit::script::Module mMSNet;
        torch::jit::script::Module mDecisionNet;
        torch::jit::script::Module mMatcherNet;
        
};
static const std::string TheNameCorrel  = "MMVIICorrel";
static const std::string TheNameExtCorr = "ExternCorrel";
static const std::string TheNameCNNCorr = "MVCNNCorrel";

// ARCHITECTURES OF CNN TRAINED 

static const std::string TheFastArch = "MVCNNFast";
static const std::string TheFastandPrjHead = "MVCNNFastProjHead";
static const std::string TheFastArchReg = "MVCNNFastReg";
static const std::string TheFastStandard = "MCNNStd";
static const std::string TheFastArchWithMLP= "MVCNNFastMLP";
static const std::string TheFastArchDirectSim="MVCNNFastDirectSIM";
static const std::string TheMSNet="MSNet";
static const std::string TheUnetMlpCubeMatcher="UnetMLPMatcher";
//.....................................................

class cAppliFillCubeCost : public cAppliLearningMatch
{
     public :
        typedef tINT2                      tElemZ;
        typedef cIm2D<tElemZ>              tImZ;
        typedef cDataIm2D<tElemZ>          tDataImZ;
        typedef cIm2D<tREAL4>              tImRad;
        typedef cDataIm2D<tREAL4>          tDataImRad;
        typedef cLayer3D<float,tElemZ>     tLayerCor;
        typedef cIm2D<tREAL4>              tImPx;
        typedef cDataIm2D<tU_INT1>         tDataImMasq;
        typedef cDataIm2D<tREAL4>          tDataImPx;
        typedef cIm2D<tREAL4>              tImFiltred;
        typedef cDataIm2D<tREAL4>          tDataImF;
        typedef cGaussianPyramid<tREAL4>   tPyr;
        typedef std::shared_ptr<tPyr>      tSP_Pyr;
        typedef cPyr1ImLearnMatch *        tPtrPyr1ILM;

        cAppliFillCubeCost(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
	double ComputCorrel(const cPt2di & aPI1,const cPt2dr & aPI2,int mSzW) const;
    // ADDED CORREL USING CNN 
    double ComputCorrelMVCNN(const cPt2di & aPI1,const cPt2dr & aPI2,int mSzW) const;
    
    /*************************************************************************/
    //void Predict(Tiff_Im & ImLeft, Tiff_Im & ImRight, ConvNet_Fast & Net);
    torch::Tensor ReadBinaryFile(std::string aFileName, torch::Tensor aHost);
    void PopulateModelFromBinary(ConvNet_Fast Net,std::vector<std::string> Names,std::string aDirModel);
    int GetWindowSize(ConvNet_Fast & Network);
    void OneSliceSimil(
                cPt2di aPx,
                std::vector<std::pair<int,std::pair<torch::Tensor,int>>> * AllSimils,
                std::vector<cOneModele*> VMods,
                torch::Tensor & EmbedL,
                torch::Tensor & EmbedR,
                tDataImZ & aDZMIN,
                tDataImZ & aDZMAX,
                int FeatSize);
    /*************************************************************************/
	const tDataImRad & DI1() {return *mDI1;}
	const tDataImRad & DI2() {return *mDI2;}
	
	/************************** MAY BE  REMOVE THIS LATER ********************/
	// accessor to NORMALIZED IMAGES 
    const tImRad & IMNorm1() {return mImNorm1;}
    const tImRad & IMNorm2() {return mImNorm2;}
	const tDataImRad & NDI1() {return *mDINorm1;}
	const tDataImRad & NDI2() {return *mDINorm2;}
    /*************************************************************************/
    
        cPyr1ImLearnMatch * PyrL1 () {return PyrL(mPyrL1,mBoxGlob1,mNameI1);}
        cPyr1ImLearnMatch * PyrL2 () {return PyrL(mPyrL2,mBoxGlob2,mNameI2);}
        tREAL8     StepZ() const {return mStepZ;}
        bool Ok1(int aX) const {return Ok(aX,mVOk1);}
        bool Ok2(int aX) const {return Ok(aX,mVOk2);}
        const cAimePCar & PC1(int aX) const {return mVPC1.at(aX);}
        const cAimePCar & PC2(int aX) const {return mVPC2.at(aX);}
        const cBox2di  & BoxGlob1() const {return mBoxGlob1;}  ///< Accessor
        const cBox2di  & BoxGlob2() const {return mBoxGlob2;}  ///< Accessor
        const std::string   & NameI1() const {return mNameI1;}  ///< Accessor
        const std::string   & NameI2() const {return mNameI2;}  ///< Accessor
        
        const std::string  & NameArch() const {return mModelArchitecture;} // ACCESSOR
        const std::string  & NameDirModel() const {return mModelBinaries;} // ACCESSOR

	cBox2di BoxFile1() const {return cDataFileIm2D::Create(mNameI1,false);}
	cBox2di BoxFile2() const {return cDataFileIm2D::Create(mNameI2,false);}

	int  SzW() const {return mSzW;}
	bool InterpolLearn() const {return mInterpolLearn;}
	double ExpLearn()    const {return mExpLearn; }
	double FactLearn()   const {return mFactLearn; }
        const cFilterPCar  & FPC() const {return mFPC;}  ///< Used to compute Pts

	const tImZ  & ImZMin() {return  mImZMin;}
	const tImZ  & ImZMax() {return  mImZMax;}
	void MakeNormalizedIm();
        // if with decision Network 
        bool mWithMatcher3D=false;
        bool mUseCuda=false;
        bool mUsePredicNet=false;

	// -------------- Internal variables -------------------
     private :

        cPyr1ImLearnMatch * PyrL (tPtrPyr1ILM & aPtrPyr,const cBox2di & aBoxI,const std::string & aNameIm)
	{
	   if (aPtrPyr==nullptr)
              aPtrPyr = new cPyr1ImLearnMatch(aBoxI,aBoxI,aNameIm,*this,mFPC,false);
	   return aPtrPyr;
	}

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	void PushCost(double aCost);
        void PushCostdbl(double aCost);
        bool Ok(int aX,const std::vector<bool> &  aV) const
	{
            return (aX>=0) && (aX<int(aV.size())) && (aV.at(aX)) ;
	}

	void MakeLinePC(int aYLoc,bool Im1);
	// -------------- Mandatory args -------------------
	std::string   mNameI1;
	std::string   mNameI2;
	std::string   mNameModele;
	cPt2di        mP0Z;  // Pt corresponding in Im1 to (0,0)
	cBox2di       mBoxGlob1;  // Box to Load, taking into account siwe effect
	cBox2di       mBoxGlob2;
	std::string   mNamePost;

	// -------------- Optionnal args -------------------
	tREAL8        mStepZ;
	bool          mCmpCorLearn; //  Create a comparison between Correl & Learn
	bool          mInterpolLearn; // Use interpolation mode for learned cost
	double        mExpLearn; // Exposant to adapt learned cost
	double        mFactLearn; // Factor to adapt learned cost
	std::string   mNameCmpModele;
	int           mSzW;
    
    // ADDED CNN PARAMS 
    std::string mModelBinaries;
    std::string mModelArchitecture;

	// -------------- Internal variables -------------------
	
	std::string StdName(const std::string & aPre,const std::string & aPost);

        int         mNbCmpCL;
	    cIm2D<tREAL8>  mImCmp;
        std::string mNameZMin;
        std::string mNameZMax;
        std::string mNameCube;
        cMMVII_Ofs* mFileCube;

	tImZ        mImZMin;
	tImZ        mImZMax;
	tImRad      mIm1;
	tDataImRad  *mDI1;
	tImRad      mIm2;
	tDataImRad  *mDI2;

        // Normalized images, in radiometry, avearge 0, std dev 1.0
	tImRad      mImNorm1;
	tDataImRad  *mDINorm1;
	tImRad      mImNorm2;
	tDataImRad  *mDINorm2;
	tLayerCor   mLayerCor;

	double      ToCmpCost(double aCost) const;

        cPyr1ImLearnMatch * mPyrL1;
        cPyr1ImLearnMatch * mPyrL2;
        cFilterPCar  mFPC;  ///< Used to compute Pts
        std::vector<bool>         mVOk1;
        std::vector<cAimePCar>    mVPC1;
        std::vector<bool>         mVOk2;
        std::vector<cAimePCar>    mVPC2;
};


/* *************************************************** */
/*                                                     */
/*                   cOneModele                        */
/*                                                     */
/* *************************************************** */


cOneModele::cOneModele
(
    const std::string & aNameModele,
    cAppliFillCubeCost  & aAppliLearn
    //aCnnModelPredictor & aPredicor
) :
   mAppli          (&aAppliLearn),
   //**************************
   //mPredictor      (&aPredicor)
   //**************************,
   mNameModele     (aNameModele),
   mWithIntCorr    (mNameModele==TheNameCorrel),
   mWithExtCorr    (mNameModele==TheNameExtCorr),
   /***********************************************************************/
   mWIthMVCNNCorr (mNameModele==TheNameCNNCorr),
   /***********************************************************************/
   mWithStatModele (! (mWithIntCorr || mWithExtCorr || mWIthMVCNNCorr)),
   mPyrL1          (nullptr),
   mPyrL2          (nullptr),
   mSzW            (mAppli->SzW()),
   mPSzW           (mSzW,mSzW),
   mCNNWin          (0,0)
{
    if (mWithStatModele)
    {
       ReadFromFile(mModele,mNameModele);
       mPyrL1 = mAppli->PyrL1 ();
       mPyrL2 = mAppli->PyrL2 ();
    }
    else if (mWithExtCorr)
    {
         CalcCorrelExterne();
    }
    // ADDED MVCNN here 
    else if (mWIthMVCNNCorr)
    {
        // ARCHITECTURE and Location of Model Binaries 
        
        mArchitecture=mAppli->NameArch();
        mModelBinDir =mAppli->NameDirModel();
        
        MMVII_INTERNAL_ASSERT_strong(mArchitecture!="","The network architecture should be specified : "+TheFastArch+" || "+TheFastStandard 
            +" || "+TheFastArchWithMLP+" || "+TheFastArchDirectSim+ " !");      
        MMVII_INTERNAL_ASSERT_strong(mModelBinDir!=""," Model params dir must be specified ! ");
        //CalcCorrelMvCNN();
        if(mArchitecture==TheFastArch)
        {
            mCNNPredictor = new aCnnModelPredictor(TheFastArch,mModelBinDir,mAppli->mUseCuda);
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
            mCNNPredictor = new aCnnModelPredictor(TheFastArchReg,mModelBinDir,mAppli->mUseCuda);
            mCNNPredictor->PopulateModelFromBinaryWithBNReg(mNetFastMVCNNReg);
            
            mCNNWin=mCNNPredictor->GetWindowSizeBNReg(mNetFastMVCNNReg);
				
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
            mCNNPredictor = new aCnnModelPredictor(TheFastandPrjHead,mModelBinDir,mAppli->mUseCuda);
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
        
        if(mArchitecture==TheFastArchWithMLP)
        {
            mCNNPredictor = new aCnnModelPredictor(TheFastArchWithMLP,mModelBinDir,mAppli->mUseCuda);
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
            mCNNPredictor = new aCnnModelPredictor(TheFastArchDirectSim,mModelBinDir,mAppli->mUseCuda);
            mCNNPredictor->PopulateModelMSNetHead(mMSNet);  // load script module 
            mCNNWin=cPt2di(7,7); // The chosen window size is 7x7
        }
        else if (mArchitecture==TheFastStandard)
        {
           mCNNPredictor = new aCnnModelPredictor(TheFastStandard,mModelBinDir,mAppli->mUseCuda);
           
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
        else if(mArchitecture==TheMSNet)
        { 
            mCNNPredictor = new aCnnModelPredictor(TheMSNet,mModelBinDir,mAppli->mUseCuda);
            mCNNPredictor->PopulateModelFeatures(mMSNet);
            mCNNWin=cPt2di(7,7); 
            if (mAppli->mUsePredicNet)
            {
              mCNNPredictor->PopulateModelDecision(mDecisionNet);  
            }
        } 
        else if (mArchitecture==TheUnetMlpCubeMatcher)
            {
                mCNNPredictor = new aCnnModelPredictor(TheUnetMlpCubeMatcher,mModelBinDir,mAppli->mUseCuda);
                mCNNPredictor->PopulateModelFeatures(mMSNet);
                mCNNWin=cPt2di(1,1);
                if (mAppli->mUsePredicNet)
                {
                  mCNNPredictor->PopulateModelDecision(mDecisionNet);
                }
                if (mAppli->mWithMatcher3D)
                    {
                        // Enhance the  generated correlation coefficients using the last stage conv3d MATCHER
                        //mCNNPredictor->PopulateModelMatcher(mMatcherNet);
                    }
            }
    }
}
void cOneModele::CalcCorrelExterneTerm(const cBox2di & aBoxInitIm1,int aPxMin,int aPxMax)
{
     // cBox2di aBoxDil = aBoxInitIm1.Inter
}

void cOneModele::CalcCorrelExterneRecurs(const cBox2di & aBoxIm1)
{
     cPt2di aP0Glob = mAppli->BoxGlob1().P0();
     int aMinPxMin = 1e6;
     int aMaxPxMax = -1e6;
     int aTotPx = 0;
     int aNbPx = 0;
     const cAppliFillCubeCost::tDataImZ & aDIZMin = mAppli->ImZMin().DIm();
     const cAppliFillCubeCost::tDataImZ & aDIZMax = mAppli->ImZMax().DIm();

     cRect2 aR2(aBoxIm1.P0(),aBoxIm1.P1());
     for (const auto & aP : aR2)
     {
          cPt2di aPLoc = aP-aP0Glob;
	  int aPxMin = aDIZMin.GetV(aPLoc);
	  int aPxMax = aDIZMax.GetV(aPLoc);
	  UpdateMin(aMinPxMin,aPxMin);
	  UpdateMax(aMaxPxMax,aPxMax);
	  aTotPx += aPxMax - aPxMin;
	  aNbPx++;
     }

     double aAvgPx = aTotPx / double(aNbPx);
     int   aIntervPx = (aMaxPxMax-aMinPxMin);

     bool   isTerminal = aIntervPx < 2 * aAvgPx;

     if ((aIntervPx*aIntervPx) > 1e8)
     {
        isTerminal = false;
     }

     {
        int aSzMin = MinAbsCoord(aBoxIm1.Sz());
	if (aSzMin<200)
           isTerminal =true;
     }


     if (isTerminal)
     {
         CalcCorrelExterneTerm(aBoxIm1,aMinPxMin,aMaxPxMax);
     }
     else
     {
        cPt2di aP0 = aBoxIm1.P0();
        cPt2di aP1 = aBoxIm1.P1();
        std::vector<int> aVx{aP0.x(),(aP0.x()+aP1.x())/2,aP1.x()};
        std::vector<int> aVy{aP0.y(),(aP0.y()+aP1.y())/2,aP1.y()};
        for (int aKx=0 ; aKx<2 ; aKx++)
        {
             for (int aKy=0 ; aKy<2 ; aKy++)
             {
                  cPt2di aQ0(aVx.at(aKx),aVy.at(aKy));
                  cPt2di aQ1(aVx.at(aKx+1),aVy.at(aKy+1));
                  CalcCorrelExterneRecurs(cBox2di(aQ0,aQ1));
             }
        }
     }
}

void cOneModele::CalcCorrelExterne()
{
   mAppli->MakeNormalizedIm();
   CalcCorrelExterneRecurs(mAppli->BoxGlob1());
}

void cOneModele::CalcCorrelMvCNN()
{
   mAppli->MakeNormalizedIm();
}


void UseZInfZSup()
{
    std::cout<<"Handling Nappes englobantes "<<std::endl;

}

double cOneModele::ComputeCost(bool & Ok,const cPt2di & aPC1,const cPt2di & aPC20,int aDZ) const
{
    Ok = false;
    double aCost= 1.0;
    if (mWithStatModele)
    {
       int aX1 =  aPC1.x();
       int aX2 =  aPC20.x() + aDZ;
       aCost = 0.5;
       if (mAppli->Ok1(aX1) && mAppli->Ok2(aX2))
       {
           cVecCaracMatch aVCM(*mPyrL1,*mPyrL2,mAppli->PC1(aX1),mAppli->PC2(aX2));
	   aCost = 1-mModele.HomologyLikelihood(aVCM,mAppli->InterpolLearn());
           aCost = mAppli->FactLearn() * pow(std::max(0.0,aCost),mAppli->ExpLearn());
           Ok = true;
       };
    }
    else if (mWithIntCorr)
    {
        cPt2dr aPC2Z(aPC20.x()+aDZ*mAppli->StepZ(),aPC20.y());
        double aCorrel = 0.0;

        if (WindInside4BL(mAppli->DI1(),aPC1,mPSzW) && WindInside4BL(mAppli->DI2(),aPC2Z,mPSzW))
        {
            aCorrel = mAppli->ComputCorrel(aPC1,aPC2Z,mSzW);
	    Ok = true;
	}
	aCost=(1-aCorrel)/2.0;
    }
    else if (mWithExtCorr)
    {
        //COMPUTE CORREL USING STATISTICAL LEARNING 
        
    }
    /*else if (mWIthMVCNNCorr)
    {
        // COMPUTE CORREL WITH MVCNN CORREL 
        cPt2dr aPC2Z(aPC20.x()+aDZ*mAppli->StepZ(),aPC20.y());
        double aCorrel = 0.0;  
        // BATCH MODE PASSING THE WHOLE TILES TO THE NETWORKS TO GET EMBEDDING VECTORS THAN DOING SIMPLE COSINE_SIMILARITY ON INDIVIDUAL VECTORS
        

        
        if (WindInside4BL(mAppli->DI1(),aPC1,mCNNWin) && WindInside4BL(mAppli->DI2(),aPC2Z,mCNNWin))  // check for Limits of inclusion  !!!!
        {
            
            //aCorrel = mAppli->ComputCorrelMVCNN(aPC1,aPC2Z,mSzW);
            // CALL MODEL PREDICTOR and NOT mAppli -> .....
            // LEFT PATCH SIZE 
            //
            //
            //PREVIOUSLY USED TO CREATE A PATCH AND DIRECTLUY PASS IT FORWARD 
            //
            //
            //cPt2di p1Uleft(aPC1.x()-round_ni(mCNNWin.x()/2),aPC1.y()-round_ni(mCNNWin.y()/2));
            cPt2di p1lRight(aPC1.x()+round_ni(mCNNWin.x()/2),aPC1.y()+round_ni(mCNNWin.y()/2));
            
            // RIGHT PATCH SIZE 
            cPt2di p2Uleft(aPC2Z.x()-round_ni(mCNNWin.x()/2),aPC2Z.y()-round_ni(mCNNWin.y()/2));
            cPt2di p2lRight(aPC2Z.x()+round_ni(mCNNWin.x()/2),aPC2Z.y()+round_ni(mCNNWin.y()/2));
            
            cBox2di aBoxL(p1Uleft,p1lRight); 
            cBox2di aBoxR(p2Uleft,p2lRight); 
                
            //Patch Left
            tImRad PatchL(aBoxL.P0(),aBoxL.P1(),*(mAppli->NDI1().ExtractRawData2D()), eModeInitImage::eMIA_V1);
            tImRad PatchR(aBoxR.P0(),aBoxR.P1(),*(mAppli->NDI2().ExtractRawData2D()), eModeInitImage::eMIA_V1);
            
            
           if (mArchitecture==TheFastStandard)
             {
               aCorrel=mCNNPredictor->Predict(mNetFastStd,PatchL,PatchR,mCNNWin);
             }
           else if (mArchitecture==TheFastArch)
             {
               aCorrel=mCNNPredictor->PredictWithBN(mNetFastMVCNN,PatchL,PatchR,mCNNWin);
             }
           else
           {
             // NOTHING FOR NOW  
               aCorrel=0.0;
           }
           
            
            //aCorrel=mCNNPredictor->Predict
            
	    Ok = true;
	}
	aCost=(1-aCorrel)/2.0;
        
    }*/
    return aCost;
}

/* *************************************************** */
/*                                                     */
/*              cAppliFillCubeCost                     */
/*                                                     */
/* *************************************************** */

cAppliFillCubeCost::cAppliFillCubeCost(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mBoxGlob1            (cBox2di::Empty()),
   mBoxGlob2            (cBox2di::Empty()),
   mStepZ               (1.0),
   mCmpCorLearn         (true),
   mInterpolLearn       (true),
   mExpLearn            (0.5),
   mFactLearn           (0.33333),
   mSzW                 (3),
   mNbCmpCL             (200),
   mImCmp               (cPt2di(mNbCmpCL+2,mNbCmpCL+2),nullptr,eModeInitImage::eMIA_Null),
   mFileCube            (nullptr),
   mImZMin              (cPt2di(1,1)),
   mImZMax              (cPt2di(1,1)),
   mIm1                 (cPt2di(1,1)),
   mDI1                 (nullptr),
   mIm2                 (cPt2di(1,1)),
   mDI2                 (nullptr),
   mImNorm1             (cPt2di(1,1)),
   mDINorm1             (nullptr),
   mImNorm2             (cPt2di(1,1)),
   mDINorm2             (nullptr),
   mLayerCor            (tLayerCor::Empty()),
   mPyrL1               (nullptr),
   mPyrL2               (nullptr),
   mFPC                 (false)
{
    mFPC.FinishAC();
    mFPC.Check();
}

void cAppliFillCubeCost::MakeNormalizedIm()
{
    if (mDINorm1!= nullptr) return;

    mImNorm1 = NormalizedAvgDev(mIm1,1e-4);
    mDINorm1 = &(mImNorm1.DIm());

    mImNorm2 = NormalizedAvgDev(mIm2,1e-4);
    mDINorm2 = &(mImNorm2.DIm());

    mLayerCor  = tLayerCor(mImZMin,mImZMax);
}


double cAppliFillCubeCost::ToCmpCost(double aCost) const
{
   return mNbCmpCL * std::max(0.0,std::min(1.0,aCost));
}

cCollecSpecArg2007 & cAppliFillCubeCost::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameI1,"Name of first image")
          <<   Arg2007(mNameI2,"Name of second image")
          <<   Arg2007(mNameModele,"Name for modele : .*dmp|MMVIICorrel|MVCNNCorrel")
          <<   Arg2007(mP0Z,"Origin in first image")
          <<   Arg2007(mBoxGlob1,"Box to read 4 Im1")
          <<   Arg2007(mBoxGlob2,"Box to read 4 Im2")
          <<   Arg2007(mNamePost,"Post fix for other names (ZMin,ZMax,Cube)")
   ;
}

cCollecSpecArg2007 & cAppliFillCubeCost::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          // << AOpt2007(mStepZ, "StepZ","Step for paralax",{eTA2007::HDV})
          << AOpt2007(mNameCmpModele, "ModCmp","Modele for Comparison")
          << AOpt2007(mSzW, "SzW","Size for windows to match",{eTA2007::HDV})
          << AOpt2007(mModelBinaries,"CNNParams" ,"Model Directory : Contient des fichiers binaires *.bin")
          << AOpt2007(mModelArchitecture,"CNNArch" ,"Modek architecture : "+TheFastArch+" || "+TheFastStandard+" || "+TheFastArchWithMLP)
          << AOpt2007(mUseCuda,"UseCuda","USE CUDA TO LAUNCH MODELS")
          << AOpt2007(mUsePredicNet,"UsePredicNet","Use the mlp to compute learnt similarities")
   ;
}

std::string cAppliFillCubeCost::StdName(const std::string & aPre,const std::string & aPost)
{
	return aPre + "_" + mNamePost + "." + aPost;
}

double cAppliFillCubeCost::ComputCorrel(const cPt2di & aPCI1,const cPt2dr & aPCI2,int aSzW) const
{
   cMatIner2Var<tREAL4> aMat;

   for (int aDx=-aSzW ; aDx<=aSzW  ; aDx++)
   {
       for (int aDy=-aSzW ; aDy<=aSzW  ; aDy++)
       {
            aMat.Add
            (
                mDI1->GetV  (aPCI1+cPt2di(aDx,aDy)),
                mDI2->GetVBL(aPCI2+cPt2dr(aDx,aDy))
            );
       }
   }

   return aMat.Correl();
}


double cAppliFillCubeCost::ComputCorrelMVCNN(const cPt2di & aPCI1,const cPt2dr & aPCI2,int aSzW) const
{
   // LEFT PATCH SIZE 
   double Corr=0.0;
   return Corr;
}


void cAppliFillCubeCost::PushCost(double aCost)
{
   tU_INT2 aICost = round_ni(1e4*(std::max(0.0,std::min(1.0,aCost))));
   mFileCube->Write(aICost);
}
void cAppliFillCubeCost::PushCostdbl(double aCost)
{
   mFileCube->Write(aCost);
}

void cAppliFillCubeCost::MakeLinePC(int aYLoc,bool Im1)
{
   if (mPyrL1==nullptr)
      return;

   MMVII_INTERNAL_ASSERT_strong(mStepZ==1.0,"For now do not handle StepZ!=1 with model");

   std::vector<bool>     & aVOK  = Im1 ? mVOk1      : mVOk2;
   std::vector<cAimePCar>& aVPC  = Im1 ? mVPC1      : mVPC2;
   const cBox2di         & aBox  = Im1 ? mBoxGlob1  : mBoxGlob2;
   cPyr1ImLearnMatch     & aPyrL = Im1 ? *mPyrL1    : *mPyrL2;

   aVOK.clear();
   aVPC.clear();

   for (int aX=aBox.P0().x() ; aX<=aBox.P1().x()  ; aX++)
   {
       cPt2di aPAbs (aX,aYLoc+mP0Z.y());
       cPt2di aPLoc = aPAbs - aBox.P0();
       aVOK.push_back(aPyrL.CalculAimeDesc(ToR(aPLoc)));
       aVPC.push_back(aPyrL.DupLPIm());
   }
}

void cAppliFillCubeCost::OneSliceSimil(
                                        cPt2di aPx,
                                        std::vector<std::pair<int,std::pair<torch::Tensor,int>>> * AllSimils,
                                        std::vector<cOneModele*> VMods,
                                        torch::Tensor & EmbedL,
                                        torch::Tensor & EmbedR,
                                        tDataImZ & aDZMIN,
                                        tDataImZ & aDZMAX,
                                        int FS)
{
    //std::cout<<"APIX"<<aPx<<std::endl;
    cPt2di aPAbs = aPx + mP0Z;
    //std::cout<<"APABS "<<aPAbs<<std::endl;
    cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
    //std::cout<<"aPC1 "<<aPC1<<std::endl;
    cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
    //std::cout<<aPC1.y()<<"     "<<aPC1.x()<<std::endl;
    using namespace torch::indexing;
    auto aVecL=EmbedL.index({0,Slice(0,FS,1),aPC1.y(),aPC1.x()}).unsqueeze(0).unsqueeze(2).unsqueeze(3);
    int liminf=round_ni(aPC20.x()+aDZMIN.GetV(aPx)*this->StepZ());
    int limsup=round_ni(aPC20.x()+aDZMAX.GetV(aPx)*this->StepZ());
    if (liminf<0 && limsup>0)
    {
            liminf=0;
    }
    /*std::cout<<"   "<<
               round_ni(aPC20.x()+aDZMIN.GetV(aPix)*this->StepZ())<<
               "   "<<aPC20.x()+aDZMAX.GetV(aPix)*this->StepZ()<<std::endl;*/
    auto aVecR=EmbedR.index(
                {0,
                 Slice(0,FS,1),
                 aPC20.y(),
                 Slice(liminf,
                       limsup,
                       1)
                }
                );
    // UNSQUEEZE LOST DIMENSIONS
    aVecR=aVecR.unsqueeze(0).unsqueeze(2);
    /*std::cout<<"The candidates tensor sizes "<<aVecR.sizes()<<"   "<<
               round_ni(aPC20.x()+aDZMIN.GetV(aPix)*this->StepZ())<<
               "   "<<aPC20.x()+aDZMAX.GetV(aPix)*this->StepZ()<<std::endl;*/
    // One MLP forward over the slice of possible disparities
    auto aRepeatedVecL=at::repeat_interleave(aVecL,aVecR.size(-1),-1);
    //std::cout<<"THE repeated Tensr size "<<aRepeatedVecL.sizes()<<std::endl;
    auto SimSlice=VMods.at(0)->mCNNPredictor->PredictDecisionNet(VMods.at(0)->mDecisionNet,aRepeatedVecL,aVecR);
    AllSimils->push_back(std::make_pair(aPx.x(),std::make_pair(SimSlice,liminf)));
}

int  cAppliFillCubeCost::Exe()
{

   // Compute names
   mNameZMin = StdName("ZMin","tif");
   mNameZMax = StdName("ZMax","tif");
   mNameCube = StdName("MatchingCube","data");
   //  Read images 
   mImZMin = tImZ::FromFile(mNameZMin);
   tDataImZ & aDZMin = mImZMin.DIm();
   mImZMax = tImZ::FromFile(mNameZMax);
   tDataImZ & aDZMax = mImZMax.DIm();

   mIm1 = tImRad::FromFile(mNameI1,mBoxGlob1);
   mDI1 = &(mIm1.DIm());
   mIm2 = tImRad::FromFile(mNameI2,mBoxGlob2);
   mDI2 = &(mIm2.DIm());

   mFileCube = new cMMVII_Ofs(mNameCube, eFileModeOut::CreateText);

   mCmpCorLearn = IsInit(&mNameCmpModele);
   std::vector<cOneModele*> aVMods;
   aVMods.push_back(new cOneModele(mNameModele,*this));
   if (mCmpCorLearn)
       aVMods.push_back(new cOneModele(mNameCmpModele,*this));

    if (aVMods.at(0)->mWIthMVCNNCorr)
    {
        //auto cuda_available = torch::cuda::is_available();
        auto cuda_available=false;
        torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
        aVMods.at(0)->CalcCorrelMvCNN();
        // Calculate the EMBEDDINGS ONE TIME USING FOWARD OVER THE WHOLE TILEs
        cPt2di aSzL = this->NDI1().Sz();      
        cPt2di aSzR = this->NDI2().Sz();
        int FeatSize=0;
        if (aVMods.at(0)->mArchitecture==TheFastStandard) FeatSize=64 ;
        if (aVMods.at(0)->mArchitecture==TheFastandPrjHead) FeatSize=64 ;
        if (aVMods.at(0)->mArchitecture==TheFastArchReg) FeatSize=64 ;
        if (aVMods.at(0)->mArchitecture==TheFastArch)   FeatSize=184;
        if (aVMods.at(0)->mArchitecture==TheFastArchWithMLP)   FeatSize=184;
        if (aVMods.at(0)->mArchitecture==TheFastArchDirectSim)   FeatSize=64;
        if (aVMods.at(0)->mArchitecture==TheMSNet) FeatSize=64 ;
        if (aVMods.at(0)->mArchitecture==TheUnetMlpCubeMatcher) FeatSize=64 ;
        torch::Tensor LREmbeddingsL,LREmbeddingsR;
        if (aVMods.at(0)->mArchitecture==TheFastStandard)
             {
               LREmbeddingsL=aVMods.at(0)->mCNNPredictor->PredictTile(aVMods.at(0)->mNetFastStd,this->IMNorm1(),aSzL);
               LREmbeddingsR=aVMods.at(0)->mCNNPredictor->PredictTile(aVMods.at(0)->mNetFastStd,this->IMNorm2(),aSzR);
             }
        else if (aVMods.at(0)->mArchitecture==TheFastArch)
             {
               LREmbeddingsL=aVMods.at(0)->mCNNPredictor->PredictWithBNTile(aVMods.at(0)->mNetFastMVCNN,this->IMNorm1(),aSzL);
               LREmbeddingsR=aVMods.at(0)->mCNNPredictor->PredictWithBNTile(aVMods.at(0)->mNetFastMVCNN,this->IMNorm2(),aSzR);
             }
        else if (aVMods.at(0)->mArchitecture==TheFastArchReg)
             {
               LREmbeddingsL=aVMods.at(0)->mCNNPredictor->PredictWithBNTileReg(aVMods.at(0)->mNetFastMVCNNReg,this->IMNorm1(),aSzL);
               LREmbeddingsR=aVMods.at(0)->mCNNPredictor->PredictWithBNTileReg(aVMods.at(0)->mNetFastMVCNNReg,this->IMNorm2(),aSzR);
             }
        else if (aVMods.at(0)->mArchitecture==TheFastandPrjHead)
             {
               LREmbeddingsL=aVMods.at(0)->mCNNPredictor->PredictPrjHead(aVMods.at(0)->mNetFastPrjHead,this->mIm1,aSzL);
               LREmbeddingsR=aVMods.at(0)->mCNNPredictor->PredictPrjHead(aVMods.at(0)->mNetFastPrjHead,this->mIm2,aSzR);
             }
        else if (aVMods.at(0)->mArchitecture==TheFastArchWithMLP)
             {
               LREmbeddingsL=aVMods.at(0)->mCNNPredictor->PredictFastWithHead(aVMods.at(0)->mNetFastMVCNNMLP,this->IMNorm1(),aSzL);
               LREmbeddingsR=aVMods.at(0)->mCNNPredictor->PredictFastWithHead(aVMods.at(0)->mNetFastMVCNNMLP,this->IMNorm2(),aSzR);
             }
        else if (aVMods.at(0)->mArchitecture==TheFastArchDirectSim)
             {
               LREmbeddingsL=aVMods.at(0)->mCNNPredictor->PredictMSNetTileFeatures(aVMods.at(0)->mMSNet,this->mIm1,aSzL); // chnaged mMSNET
               LREmbeddingsR=aVMods.at(0)->mCNNPredictor->PredictMSNetTileFeatures(aVMods.at(0)->mMSNet,this->mIm2,aSzR);  // chnaged mMSNET
             }
        else if (aVMods.at(0)->mArchitecture==TheMSNet)
             {
               LREmbeddingsL=aVMods.at(0)->mCNNPredictor->PredictMSNetTile(aVMods.at(0)->mMSNet,this->mIm1,aSzL);
               LREmbeddingsR=aVMods.at(0)->mCNNPredictor->PredictMSNetTile(aVMods.at(0)->mMSNet,this->mIm2,aSzR);
             }
        else if (aVMods.at(0)->mArchitecture==TheUnetMlpCubeMatcher)
             {
               LREmbeddingsL=aVMods.at(0)->mCNNPredictor->PredictMSNetTileFeatures(aVMods.at(0)->mMSNet,this->mIm1,aSzL);
               LREmbeddingsR=aVMods.at(0)->mCNNPredictor->PredictMSNetTileFeatures(aVMods.at(0)->mMSNet,this->mIm2,aSzR);
             }

        StdOut()  <<" EMBEDDING TENSOR SIZE LEFT  "<<LREmbeddingsL.sizes()<<"\n";
        StdOut()  <<" EMBEDDING TENSOR SIZE RIGHT  "<<LREmbeddingsR.sizes()<<"\n";
        // DIMS OF LREmbeddings == {2,FEAT_VECTOR_SIZE=184 OU 64, TILE_HEIGHT,TILE_WIDTH }
        // Perform COSINE METRIC TO GET CORRELATION VALUES BETWEEN EMBDEDDINGS
        // RESET DEVICES
        LREmbeddingsL=LREmbeddingsL.to(device);
        LREmbeddingsR=LREmbeddingsR.to(device);
        
        cPt2di aPix;
        using namespace torch::indexing;
        if (mUsePredicNet && !mWithMatcher3D)
        {
            LREmbeddingsL=LREmbeddingsL.unsqueeze(0);
            LREmbeddingsR=LREmbeddingsR.unsqueeze(0);
            //std::vector<std::pair<torch::Tensor,int>> * AllSimilResults=new std::vector<std::pair<torch::Tensor,int>>;

            /*std::vector<std::pair<int,std::pair<torch::Tensor,int>>> * AllSimilResults=new std::vector<std::pair<int,std::pair<torch::Tensor,int>>>;

            for (aPix.y()=0 ; aPix.y()<aSzL.y() ; aPix.y()++)
            {
                    int nthreads = (int)std::thread::hardware_concurrency();
                    //std::cout<<"NUMBER OF POSSIBLE THREADS AT ONCE "<<nthreads<<std::endl;
                   // SCALING THE PROCESSES THROUGH columns
                    int SCALES=aSzL.x()/nthreads;
                    int RESTE=aSzL.x()%nthreads;
                    //std::cout<<"SCALES  AND REST "<<SCALES<<"    "<<RESTE<<std::endl;
                    std::vector<std::thread> * AllThreads=new std::vector<std::thread>;
                    for (int ss=0;ss<SCALES;ss++)
                    {
                        for (aPix.x()=ss*nthreads;aPix.x()<(ss+1)*nthreads;aPix.x()++)
                          {
                            //std::cout<<"COORDINATES "<<aPix.x()<<std::endl;
                            AllThreads->push_back(std::thread([&]{
                                    this->OneSliceSimil(
                                                aPix,
                                                AllSimilResults,
                                                aVMods,
                                                LREmbeddingsL,
                                                LREmbeddingsR,
                                                aDZMin,
                                                aDZMax,
                                                FeatSize
                                                );
                                }
                            ));
                          }
                        aPix.x()--;
                       for (int ll=0;ll<nthreads;ll++)
                       {
                           AllThreads->at(ll).join();
                       }
                       AllThreads->clear();
                    }
                    //std::cout<<"APIX   "<<aPix<<std::endl;
                    // Calculer le  reste des operations
                    if (RESTE>0)
                       {
                            for (aPix.x()=SCALES*nthreads;aPix.x()<SCALES*nthreads+RESTE;aPix.x()++)
                              {
                                AllThreads->push_back(std::thread([&]{
                                        this->OneSliceSimil(
                                                    aPix,
                                                    AllSimilResults,
                                                    aVMods,
                                                    LREmbeddingsL,
                                                    LREmbeddingsR,
                                                    aDZMin,
                                                    aDZMax,
                                                    FeatSize
                                                    );
                                    }
                                ));
                              }
                           for (int cc=0;cc<RESTE;cc++)
                           {
                               AllThreads->at(cc).join();
                           }
                        }
                    // Arranger les résultats des threads
                    std::sort(AllSimilResults->begin(),AllSimilResults->end(),sortspecific);
                    std::cout<<"ALLSIMILS <<   "<<AllSimilResults->size()<<std::endl;
                    for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                        {
                            std::cout<<" SIMIL INDEX "<<AllSimilResults->at(aPix.x()).first<<std::endl;
                        }
                    delete AllThreads;
                    // Parcourir tous les slices de similarité pour remplir les nappes englobantes
                    for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                       {
                            cPt2di aPAbs = aPix + mP0Z;
                            cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                            cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                            for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
                            {
                                cPt2di aPC2Z(round_ni(aPC20.x()+aDz*this->StepZ()),aPC20.y());
                                double aTabCost[2]={1.0,1.0};
                                bool   aTabOk[2]={false,false};
                                for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                                        {
                                            bool IsInside=WindInside4BL(this->DI1(),aPC1,aVMods[aK]->mCNNWin) && WindInside4BL(this->DI2(),aPC2Z,aVMods[aK]->mCNNWin);
                                            if(IsInside)
                                            {
                                                using namespace torch::indexing;
                                                auto aSim=AllSimilResults->at(aPix.x()).second.first.index({aPC2Z.x()-AllSimilResults->at(aPix.x()).second.second});
                                                double corr=(double)aSim.item<float>();
                                                aTabCost[aK] = 1-corr;
                                                aTabOk[aK]=true;
                                            }
                                        }
                                PushCost(aTabCost[0]);
                                if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
                                {
                                       double aC0 = ToCmpCost(aTabCost[0]);
                                       double aC1 = ToCmpCost(aTabCost[1]);
                                       mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
                                }
                            }
                       }
                    AllSimilResults->clear();
            }
            delete AllSimilResults;
            */
            //========================================================================================
            /*for (aPix.y()=0 ; aPix.y()<aSzL.y() ; aPix.y()++)
            {
                for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                {
                        cPt2di aPAbs = aPix + mP0Z;
                        cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                        cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                        using namespace torch::indexing;
                        auto aVecL=LREmbeddingsL.index({0,Slice(0,FeatSize,1),aPC1.y(),aPC1.x()}).unsqueeze(0).unsqueeze(2).unsqueeze(3);
                        int liminf=round_ni(aPC20.x()+aDZMin.GetV(aPix)*this->StepZ());
                        int limsup=round_ni(aPC20.x()+aDZMax.GetV(aPix)*this->StepZ());
                        if (liminf<0 && limsup>0)
                        {
                                liminf=0;
                        }
                        auto aVecR=LREmbeddingsR.index(
                                    {0,
                                     Slice(0,FeatSize,1),
                                     aPC20.y(),
                                     Slice(liminf,
                                           limsup,
                                           1)
                                    }
                                    );

                        // UNSQUEEZE LOST DIMENSIONS
                        aVecR=aVecR.unsqueeze(0).unsqueeze(2);
                        //std::cout<<"The candidates tensor sizes "<<aVecR.sizes()<<"   "<<
                         //          round_ni(aPC20.x()+aDZMin.GetV(aPix)*this->StepZ())<<
                          //        "   "<<aPC20.x()+aDZMax.GetV(aPix)*this->StepZ()<<std::endl;
                        // One MLP forward over the slice of possible disparities
                        auto aRepeatedVecL=at::repeat_interleave(aVecL,aVecR.size(-1),-1);
                        //std::cout<<"THE repeated Tensr size "<<aRepeatedVecL.sizes()<<std::endl;
                        auto SimSlice=aVMods.at(0)->mCNNPredictor->PredictDecisionNet(aVMods.at(0)->mDecisionNet,aRepeatedVecL,aVecR);
                        //std::cout<<"The similarity Slice "<<SimSlice.numel()<<std::endl;
                        for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
                        {
                            cPt2di aPC2Z(round_ni(aPC20.x()+aDz*this->StepZ()),aPC20.y());
                            double aTabCost[2]={1.0,1.0};
                            bool   aTabOk[2]={false,false};
                            for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                                    {
                                        bool IsInside=WindInside4BL(this->DI1(),aPC1,aVMods[aK]->mCNNWin) && WindInside4BL(this->DI2(),aPC2Z,aVMods[aK]->mCNNWin);
                                        if(IsInside)
                                        {
                                            using namespace torch::indexing;
                                            auto aSim=SimSlice.index({aPC2Z.x()-liminf});
                                            double corr=(double)aSim.item<float>();
                                            aTabCost[aK] = 1-corr;
                                            aTabOk[aK]=true;
                                        }
                                    }
                            PushCost(aTabCost[0]);
                            if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
                            {
                                   double aC0 = ToCmpCost(aTabCost[0]);
                                   double aC1 = ToCmpCost(aTabCost[1]);
                                   mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
                            }
                        }
                }
            }*/


            //=======================================================================================
            LREmbeddingsL=LREmbeddingsL.squeeze(0);
            LREmbeddingsR=LREmbeddingsR.squeeze(0);
            for (aPix.y()=0 ; aPix.y()<aSzL.y() ; aPix.y()++)
            {
                    int  aMinZmin=1e8;
                    int  aMaxZmax=-1e8;
                    for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                    {
                            if (aDZMin.GetV(aPix)<aMinZmin) aMinZmin=aDZMin.GetV(aPix);
                            if (aDZMax.GetV(aPix)>aMaxZmax) aMaxZmax=aDZMax.GetV(aPix);
                    }
                    // FILL COST VOLUME SLICE ON ONE LINE
                    //auto cuda_available = torch::cuda::is_available();
                    torch::Device TheAvailDevice(mUseCuda ? torch::kCUDA : torch::kCPU);
                    ELISE_ASSERT(aMaxZmax-aMinZmin, "PAX INTERVAL NULL !");
                    using namespace torch::indexing;

                    auto aSlcL=LREmbeddingsL.index({Slice(0,FeatSize,1),
                                                    Slice(aPix.y()+mP0Z.y()-mBoxGlob1.P0().y()
                                                    ,aPix.y()+mP0Z.y()-mBoxGlob1.P0().y()+1,1),
                                                    Slice(mP0Z.x()-mBoxGlob1.P0().x(),
                                                    aSzL.x()+mP0Z.x()-mBoxGlob1.P0().x(),1)}); // FeatSize,1,W
                    //std::cout<<"SLICE LEFT IMAGE "<<aSlcL.sizes()<<std::endl;
                    //aSlcL=aSlcL.unsqueeze(1);
                    //std::cout<<"SLICE LEFT IMAGE "<<aSlcL.sizes()<<std::endl;
                    int Intervalle_DPAX=round_ni(aMaxZmax-aMinZmin);// /this->StepZ());
                    torch::Tensor CUBE= torch::ones({2*FeatSize,Intervalle_DPAX,1,aSzL.x()},torch::TensorOptions().dtype(torch::kFloat32).device(TheAvailDevice)).mul(1.0);

                    ELISE_ASSERT(Intervalle_DPAX==aMaxZmax-aMinZmin, "ISSUE WITH PAWX INTERVAL ");

                    for (int dd=0;dd<Intervalle_DPAX;dd++)
                        {
                            // Get relevant right features

                            CUBE.index({Slice(0,FeatSize,1),dd,Slice(0,None,1),Slice(0,None,1)}).copy_(aSlcL);
                            //std::cout<<"COPIED SLICE LEFT IN THE COST VOLUME "<<std::endl;

                            int lim_inf=round_ni(mP0Z.x()-mBoxGlob2.P0().x()+dd+aMinZmin);
                            int lim_sup=round_ni(mP0Z.x()-mBoxGlob2.P0().x()+dd+aMinZmin+aSzL.x());

                            // CONDITIONS ON CUBE LIMITS
                            // CONDITIONS ON CUBE LIMITS
                            if(lim_inf<0 && lim_sup>aSzR.x())
                              {
                                auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                                   Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                                   ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                                   Slice(0,aSzR.x(),1)}); // FEatSize,1,W
                                //std::cout<<"SLICE RIGHT "<<aSlcR.sizes()<<std::endl;
                                //std::cout<<"liminf <0    "<<lim_inf<<"  limsup    "<<lim_sup<<std::endl;
                                CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,
                                            Slice(0,None,1),Slice(-lim_inf,aSzR.x()-lim_inf,1)}).copy_(aSlcR);

                              }
                            else if (lim_inf<0)
                                {
                                 auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                                    Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                                    ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                                    Slice(0,lim_sup,1)}); // FEatSize,1,W

                                 //aSlcR=aSlcR.unsqueeze(1);
                                 //std::cout<<"SLICE RGHT "<<aSlcR.sizes()<<std::endl;
                                 //std::cout<<"liminf <0 "<<std::endl;
                                 CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,Slice(0,None,1),Slice(-lim_inf,CUBE.size(3),1)}).copy_(aSlcR);
                                }
                            else if (lim_sup>aSzR.x())
                                {
                                    auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                                       Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                                       ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                                       Slice(lim_inf,aSzR.x(),1)}); // FEatSize,1,W
                                    //aSlcR=aSlcR.unsqueeze(1);
                                    //std::cout<<"limsup sup "<<std::endl;
                                    //std::cout<<"SLICE RGHT SUPPP  "<<aSlcR.sizes()<<std::endl;
                                    CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,Slice(0,None,1),Slice(0,aSlcR.size(2),1)}).copy_(aSlcR);
                                }
                            else
                                {
                                    // TAKE USUAL RANGES
                                    auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                                       Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                                       ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                                       Slice(lim_inf,lim_sup,1)}); // FEatSize,1,W
                                    //aSlcR=aSlcR.unsqueeze(1);
                                    //std::cout<<"SLICE RGHT  HABITUEL "<<aSlcR.sizes()<<std::endl;
                                    CUBE.index({Slice(FeatSize,2*FeatSize,1),
                                                dd,
                                                Slice(0,None,1),Slice(0,None,1)}).copy_(aSlcR);
                                }
                        }

                    // For One Line Get all the similarity Values
                    //std::cout<<"GENERATED CUBE OF FEATURES "<<std::endl;
                    // infer to get the similarity cost volume
                    //auto aSimilCUBE=aVMods.at(0)->mCNNPredictor->PredictCUBE(aVMods.at(0)->mDecisionNet,aVMods.at(0)->mMatcherNet,CUBE);
                    auto aSimilCUBE=aVMods.at(0)->mCNNPredictor->PredictONCUBE(aVMods.at(0)->mDecisionNet, CUBE);
                    CUBE.resize_(at::IntArrayRef{0});
                    //std::cout<<" SIMIL CUBE IS GENERATED =====+>   "<<aSimilCUBE.sizes()<<std::endl;
                    if (aSimilCUBE.dim()==4)
                    {
                        aSimilCUBE.squeeze();
                    }
                    // Fill cube cost
                    for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                    {
                            cPt2di aPAbs = aPix + mP0Z;
                            cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                            cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                            for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
                            {
                                double aTabCost[2]={1.0,1.0};
                                bool   aTabOk[2]={false,false};
                                cPt2di aPC2Z(round_ni(aPC20.x()+aDz*this->StepZ()),aPC20.y());  // INTEG FOR NOW
                                for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                                        {
                                            bool IsInside=WindInside4BL(this->DI1(),aPC1,aVMods[aK]->mCNNWin) && WindInside4BL(this->DI2(),aPC2Z,aVMods[aK]->mCNNWin);
                                            if(IsInside)
                                            {
                                                //auto aVecR=LREmbeddingsR.slice(2,aPC2Z.y(),aPC2Z.y()+1).slice(3,aPC2Z.x(),aPC2Z.x()+1);
                                                using namespace torch::indexing;
                                                auto aSim=aSimilCUBE.index({Slice(aDz-aMinZmin,aDz-aMinZmin+1,1),
                                                                                Slice(aPix.x(),aPix.x()+1,1)});
                                                ELISE_ASSERT(aSim.item<float>()<=1.0 && aSim.item<float>()>=0, "Similarity values issue not in bound 0 ,1 ");
                                                aTabCost[aK] =(1-(double)aSim.item<float>());
                                                aTabOk[aK]=true;
                                            }
                                        }
                                PushCost(aTabCost[0]);
                                if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
                                {
                                        double aC0 = ToCmpCost(aTabCost[0]);
                                        double aC1 = ToCmpCost(aTabCost[1]);
                                        mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
                                }
                            }
                    }

            }
           //========================================================================================

            /*for (aPix.y()=0 ; aPix.y()<aSzL.y() ; aPix.y()++)
            {
                for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                {
                        cPt2di aPAbs = aPix + mP0Z;
                        cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                        cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                        using namespace torch::indexing;
                        auto aVecL=LREmbeddingsL.index({0,Slice(0,FeatSize,1),aPC1.y(),aPC1.x()}).unsqueeze(0).unsqueeze(2).unsqueeze(3);
                        for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
                        {
                            double aTabCost[2]={1.0,1.0};
                            bool   aTabOk[2]={false,false};
                            // Get location of the pixel for which to compute correl given the limits of lower and upper layers (NAPPES ENGMOBANTES)
                            cPt2di aPC2Z(round_ni(aPC20.x()+aDz*this->StepZ()),aPC20.y());  // INTEG FOR NOW   
                            //StdOut() <<" COORDINATE AT RIGHT IM "<<aPC2Z.x()<<"\n";
                            for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                                    {
                                        bool IsInside=WindInside4BL(this->DI1(),aPC1,aVMods[aK]->mCNNWin) && WindInside4BL(this->DI2(),aPC2Z,aVMods[aK]->mCNNWin);
                                        if(IsInside)
                                        {
                                            //auto aVecR=LREmbeddingsR.slice(2,aPC2Z.y(),aPC2Z.y()+1).slice(3,aPC2Z.x(),aPC2Z.x()+1);
                                            using namespace torch::indexing;
                                            //StdOut() <<" shape element embedding "<<LREmbeddingsR.sizes()<<"\n";
                                            auto aVecR=LREmbeddingsR.index({0,Slice(0,FeatSize,1),aPC2Z.y(),aPC2Z.x()}).unsqueeze(0).unsqueeze(2).unsqueeze(3);
                                            //StdOut() <<" device of element "<<aVecR.device()<<"\n";
                                            auto aSim=aVMods.at(0)->mCNNPredictor->PredictDecisionNet(aVMods.at(0)->mDecisionNet,aVecL,aVecR);
                                            double corr=(double)aSim.item<float>();
                                            aTabCost[aK] =(1-corr);
                                            aTabOk[aK]=true;
                                        }
                                    }
                            PushCost(aTabCost[0]);
                            if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
                            {
                                    double aC0 = ToCmpCost(aTabCost[0]);
                                    double aC1 = ToCmpCost(aTabCost[1]);
                                    mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
                            }
                        }
                }
            }*/

        }
        else if(mUsePredicNet && mWithMatcher3D)

        {
            // CREATE THE COST VOLUME AND PUSH COST AT RELEVANT NAPPE LOCATIONS

            // calculer le min sur les nappes
            int  aMinZmin=1e8;
            int  aMaxZmax=-1e8;
            cPt2di aPix;

            for (aPix.y()=0 ; aPix.y()<aSzL.y() ; aPix.y()++)
            {
                for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                {
                        if (aDZMin.GetV(aPix)<aMinZmin) aMinZmin=aDZMin.GetV(aPix);
                        if (aDZMax.GetV(aPix)>aMaxZmax) aMaxZmax=aDZMax.GetV(aPix);
                }
            }
            // Construct a cube of features to be forwarded into the network
            torch::Device TheAvailDevice(mUseCuda ? torch::kCUDA : torch::kCPU);
            ELISE_ASSERT(aMaxZmax-aMinZmin, "PAX INTERVAL NULL !");


            using namespace torch::indexing;

            auto aSlcL=LREmbeddingsL.index({Slice(0,FeatSize,1),
                                            Slice(mP0Z.y()-mBoxGlob1.P0().y()
                                            ,aSzL.y()+mP0Z.y()-mBoxGlob1.P0().y(),1),
                                            Slice(mP0Z.x()-mBoxGlob1.P0().x(),
                                            aSzL.x()+mP0Z.x()-mBoxGlob1.P0().x(),1)}); // FeatSize,H,W

            int Intervalle_DPAX=round_ni(aMaxZmax-aMinZmin);// /this->StepZ());
            torch::Tensor CUBE= torch::ones({2*FeatSize,Intervalle_DPAX,aSzL.y(),aSzL.x()},torch::TensorOptions().dtype(torch::kFloat32).device(TheAvailDevice)).mul(0.5);

            ELISE_ASSERT(Intervalle_DPAX==aMaxZmax-aMinZmin, "ISSUE WITH PAX INTERVAL ");

            for (int dd=0;dd<Intervalle_DPAX;dd++)
                {
                    // Get relevant right features

                    CUBE.index({Slice(0,FeatSize,1),dd,Slice(0,None,1),Slice(0,None,1)}).copy_(aSlcL);

                    int lim_inf=round_ni(mP0Z.x()-mBoxGlob2.P0().x()+dd+aMinZmin);
                    int lim_sup=round_ni(mP0Z.x()-mBoxGlob2.P0().x()+dd+aMinZmin+aSzL.x());

                    // CONDITIONS ON CUBE LIMITS
                    if(lim_inf<0 && lim_sup>aSzR.x())
                      {
                        auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                           Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                           ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                           Slice(0,aSzR.x(),1)}); // FEatSize,1,W
                        //std::cout<<"SLICE RIGHT "<<aSlcR.sizes()<<std::endl;
                        //std::cout<<"liminf <0    "<<lim_inf<<"  limsup    "<<lim_sup<<std::endl;
                        CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,
                                    Slice(0,None,1),Slice(-lim_inf,aSzR.x()-lim_inf,1)}).copy_(aSlcR);

                      }
                    else if (lim_inf<0)
                        {
                         auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                            Slice(mP0Z.y()-mBoxGlob2.P0().y()
                                                            ,aSzL.y()+mP0Z.y()-mBoxGlob2.P0().y(),1),
                                                            Slice(0,lim_sup,1)}); // FEatSize,H,W
                         //std::cout<<"liminf <0 "<<std::endl;
                         CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,Slice(0,None,1),Slice(-lim_inf,CUBE.size(3),1)}).copy_(aSlcR);
                        }
                    else if (lim_sup>aSzR.x())
                        {
                            auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                               Slice(mP0Z.y()-mBoxGlob2.P0().y()
                                                               ,aSzL.y()+mP0Z.y()-mBoxGlob2.P0().y(),1),
                                                               Slice(lim_inf,aSzR.x(),1)}); // FEatSize,H,W
                            //std::cout<<"limsup sup "<<std::endl;
                            CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,Slice(0,None,1),Slice(0,aSlcR.size(2),1)}).copy_(aSlcR);
                        }
                    else
                        {
                            // TAKE USUAL RANGES
                            auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                               Slice(mP0Z.y()-mBoxGlob2.P0().y()
                                                               ,aSzL.y()+mP0Z.y()-mBoxGlob2.P0().y(),1),
                                                               Slice(lim_inf,lim_sup,1)}); // FEatSize,H,W
                            CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,Slice(0,None,1),Slice(0,None,1)}).copy_(aSlcR);
                        }
                }

            std::cout<<"GENERATED CUBE OF FEATURES "<<std::endl;
            // Clear embeddings
            LREmbeddingsL.resize_(at::IntArrayRef{0});
            LREmbeddingsR.resize_(at::IntArrayRef{0});
            // infer to get the similarity cost volume
            //auto aSimilCUBE=aVMods.at(0)->mCNNPredictor->PredictCUBE(aVMods.at(0)->mDecisionNet,aVMods.at(0)->mMatcherNet,CUBE);
            auto aSimilCUBE=aVMods.at(0)->mCNNPredictor->PredictONCUBE(aVMods.at(0)->mDecisionNet,
                                                                       /*aVMods.at(0)->mMatcherNet,*/
                                                                       CUBE);
            CUBE.resize_(at::IntArrayRef{0});
            std::cout<<" SIMIL CUBE IS GENERATED =====+>   "<<aSimilCUBE.sizes()<<std::endl;
            /*std::cout<<"A sample of data "<<
                       aSimilCUBE.slice(0,5,6,1)<<std::endl;*/

            if (aSimilCUBE.dim()==4)
            {
                aSimilCUBE.squeeze();
            }
            // Refill tables defining layers of NAPPES WITH GENERATED CORREL
                for (aPix.y()=0 ; aPix.y()<aSzL.y() ; aPix.y()++)
                {
                    for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                    {
                            cPt2di aPAbs = aPix + mP0Z;
                            cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                            cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                            for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
                            {
                                double aTabCost[2]={1.0,1.0};
                                bool   aTabOk[2]={false,false};
                                cPt2di aPC2Z(round_ni(aPC20.x()+aDz*this->StepZ()),aPC20.y());  // INTEG FOR NOW
                                for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                                        {
                                            bool IsInside=WindInside4BL(this->DI1(),aPC1,aVMods[aK]->mCNNWin) && WindInside4BL(this->DI2(),aPC2Z,aVMods[aK]->mCNNWin);
                                            if(IsInside)
                                            {
                                                //auto aVecR=LREmbeddingsR.slice(2,aPC2Z.y(),aPC2Z.y()+1).slice(3,aPC2Z.x(),aPC2Z.x()+1);
                                                using namespace torch::indexing;
                                                auto aSim=aSimilCUBE.index({Slice(aDz-aMinZmin,aDz-aMinZmin+1,1),
                                                                                Slice(aPix.y(),aPix.y()+1,1),
                                                                                Slice(aPix.x(),aPix.x()+1,1)});
                                                ELISE_ASSERT(aSim.item<float>()<=1.0 && aSim.item<float>()>=0, "Similarity values issue not in bound 0 ,1 ");
                                                aTabCost[aK] =(1-(double)aSim.item<float>());
                                                aTabOk[aK]=true;
                                            }
                                        }
                                PushCostdbl(aTabCost[0]);
                                if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
                                {
                                        double aC0 = ToCmpCost(aTabCost[0]);
                                        double aC1 = ToCmpCost(aTabCost[1]);
                                        mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
                                }
                            }
                    }
                }
        }
        else
        {
            /*for (aPix.y()=0 ; aPix.y()<aSzL.y() ; aPix.y()++)
            {
                for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                {
                        cPt2di aPAbs = aPix + mP0Z;
                        cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                        cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                        //auto aVecL=LREmbeddingsL.slice(2,aPC1.y(),aPC1.y()+1).slice(3,aPC1.x(),aPC1.x()+1);
                        using namespace torch::indexing;
                        auto aVecL=LREmbeddingsL.index({Slice(0,FeatSize,1),aPC1.y(),aPC1.x()});
                        //StdOut() <<"pax limits "<<"MIN "<<aDZMin.GetV(aPix)<<" MAX "<<aDZMax.GetV(aPix)<<"\n";
                        for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
                        {
                            double aTabCost[2]={1.0,1.0};
                            bool   aTabOk[2]={false,false};
                            // Get location of the pixel for which to compute correl given the limits of lower and upper layers (NAPPES)
                            cPt2di aPC2Z(round_ni(aPC20.x()+aDz*this->StepZ()),aPC20.y());  // INTEG FOR NOW   
                            for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                                    {
                                        bool IsInside=WindInside4BL(this->DI1(),aPC1,aVMods[aK]->mCNNWin) && WindInside4BL(this->DI2(),aPC2Z,aVMods[aK]->mCNNWin);
                                        if(IsInside)
                                        {   
                                            using namespace torch::indexing;
                                            //auto aPCI1=LREmbeddingsL.index({Slice(0,FeatSize,1),Slice(aPC1.y()-1,aPC1.y()+2,1),Slice(aPC1.x()-1,aPC1.x()+2,1)});
                                            //auto aPCI2=LREmbeddingsR.index({Slice(0,FeatSize,1),Slice(aPC2Z.y()-1,aPC2Z.y()+2,1),Slice(aPC2Z.x()-1,aPC2Z.x()+2,1)});
                                            //auto aSim=F::cosine_similarity(aPCI1, aPCI2, F::CosineSimilarityFuncOptions().dim(0)).mean().squeeze();
                                            auto aVecR=LREmbeddingsR.index({Slice(0,FeatSize,1),aPC2Z.y(),aPC2Z.x()});
                                            auto aSim=F::cosine_similarity(aVecL, aVecR, F::CosineSimilarityFuncOptions().dim(0)).squeeze();
                                            //StdOut() <<aSim<<"\n";
                                            aTabCost[aK] =(1-(double)aSim.item<float>())/2.0 ;
                                            aTabOk[aK]=true;
                                        }
                                    }
                            PushCost(aTabCost[0]);
                            if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
                            {
                                    double aC0 = ToCmpCost(aTabCost[0]);
                                    double aC1 = ToCmpCost(aTabCost[1]);
                                    mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
                            }
                        }
                }
            }*/

             // ACCELERATED COMPUTATION OF COSINE SIMILARITY BASED ON FEATURES ONLY
                //=======================================================================================
                //LREmbeddingsL=LREmbeddingsL.squeeze(0);
                //LREmbeddingsR=LREmbeddingsR.squeeze(0);
                for (aPix.y()=0 ; aPix.y()<aSzL.y() ; aPix.y()++)
                {
                        int  aMinZmin=1e8;
                        int  aMaxZmax=-1e8;
                        for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                        {
                                if (aDZMin.GetV(aPix)<aMinZmin) aMinZmin=aDZMin.GetV(aPix);
                                if (aDZMax.GetV(aPix)>aMaxZmax) aMaxZmax=aDZMax.GetV(aPix);
                        }
                        // FILL COST VOLUME SLICE ON ONE LINE
                        //auto cuda_available = torch::cuda::is_available();
                        torch::Device TheAvailDevice(mUseCuda ? torch::kCUDA : torch::kCPU);
                        ELISE_ASSERT(aMaxZmax-aMinZmin, "PAX INTERVAL NULL !");
                        using namespace torch::indexing;

                        auto aSlcL=LREmbeddingsL.index({Slice(0,FeatSize,1),
                                                        Slice(aPix.y()+mP0Z.y()-mBoxGlob1.P0().y()
                                                        ,aPix.y()+mP0Z.y()-mBoxGlob1.P0().y()+1,1),
                                                        Slice(mP0Z.x()-mBoxGlob1.P0().x(),
                                                        aSzL.x()+mP0Z.x()-mBoxGlob1.P0().x(),1)}); // FeatSize,1,W
                        //std::cout<<"SLICE LEFT IMAGE "<<aSlcL.sizes()<<std::endl;
                        //aSlcL=aSlcL.unsqueeze(1);
                        //std::cout<<"SLICE LEFT IMAGE "<<aSlcL.sizes()<<std::endl;
                        int Intervalle_DPAX=round_ni(aMaxZmax-aMinZmin);// /this->StepZ());
                        torch::Tensor CUBE= torch::ones({2*FeatSize,Intervalle_DPAX,1,aSzL.x()},torch::TensorOptions().dtype(torch::kFloat32).device(TheAvailDevice)).mul(0.5);
                        //std::cout<<"CUBE CONTAINEER SHAPE "<<CUBE.sizes()<<std::endl;
                        ELISE_ASSERT(Intervalle_DPAX==aMaxZmax-aMinZmin, "ISSUE WITH PAWX INTERVAL ");

                        for (int dd=0;dd<Intervalle_DPAX;dd++)
                            {
                                // Get relevant right features

                                CUBE.index({Slice(0,FeatSize,1),dd,Slice(0,None,1),Slice(0,None,1)}).copy_(aSlcL);
                                //std::cout<<"COPIED SLICE LEFT IN THE COST VOLUME "<<std::endl;

                                int lim_inf=round_ni(mP0Z.x()-mBoxGlob2.P0().x()+dd+aMinZmin);
                                int lim_sup=round_ni(mP0Z.x()-mBoxGlob2.P0().x()+dd+aMinZmin+aSzL.x());

                                // CONDITIONS ON CUBE LIMITS
                                if(lim_inf<0 && lim_sup>aSzR.x())
                                  {
                                    auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                                       Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                                       ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                                       Slice(0,aSzR.x(),1)}); // FEatSize,1,W
                                    //std::cout<<"SLICE RIGHT "<<aSlcR.sizes()<<std::endl;
                                    //std::cout<<"liminf <0    "<<lim_inf<<"  limsup    "<<lim_sup<<std::endl;
                                    CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,
                                                Slice(0,None,1),Slice(-lim_inf,aSzR.x()-lim_inf,1)}).copy_(aSlcR);

                                  }
                                else if (lim_inf<0)
                                    {
                                     auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                                        Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                                        ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                                        Slice(0,lim_sup,1)}); // FEatSize,1,W

                                     //aSlcR=aSlcR.unsqueeze(1);
                                     //std::cout<<"SLICE RIGHT "<<aSlcR.sizes()<<std::endl;
                                     //std::cout<<"liminf <0    "<<lim_inf<<"  limsup    "<<lim_sup<<std::endl;
                                     CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,Slice(0,None,1),
                                                 Slice(-lim_inf,CUBE.size(3),1)}).copy_(aSlcR);
                                    }
                                else if (lim_sup>aSzR.x())
                                    {
                                        auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                                           Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                                           ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                                           Slice(lim_inf,aSzR.x(),1)}); // FEatSize,1,W
                                        //aSlcR=aSlcR.unsqueeze(1);
                                        //std::cout<<"limsup sup "<<std::endl;
                                        //std::cout<<"SLICE RGHT SUPPP  "<<aSlcR.sizes()<<std::endl;
                                        CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,Slice(0,None,1),Slice(0,aSlcR.size(2),1)}).copy_(aSlcR);
                                    }
                                else
                                    {
                                        // TAKE USUAL RANGES
                                        auto aSlcR=LREmbeddingsR.index({Slice(0,FeatSize,1),
                                                                           Slice(aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()
                                                                           ,aPix.y()+mP0Z.y()-mBoxGlob2.P0().y()+1,1),
                                                                           Slice(lim_inf,lim_sup,1)}); // FEatSize,1,W
                                        //aSlcR=aSlcR.unsqueeze(1);
                                        //std::cout<<"SLICE RGHT  HABITUEL "<<aSlcR.sizes()<<std::endl;
                                        CUBE.index({Slice(FeatSize,2*FeatSize,1),dd,Slice(0,None,1),Slice(0,None,1)}).copy_(aSlcR);
                                    }
                            }

                        // For One Line Get all the similarity Values
                        //std::cout<<"GENERATED CUBE OF FEATURES "<<std::endl;
                        // infer to get the similarity cost volume
                        //auto aSimilCUBE=aVMods.at(0)->mCNNPredictor->PredictCUBE(aVMods.at(0)->mDecisionNet,aVMods.at(0)->mMatcherNet,CUBE);
                        //auto aSimilCUBE=aVMods.at(0)->mCNNPredictor->PredictONCUBE(aVMods.at(0)->mDecisionNet, CUBE);
                        //::cosine_similarity(aVecL, aVecR, F::CosineSimilarityFuncOptions().dim(0)).squeeze();
                        //auto aCUBEL=CUBE.index({Slice(0,FeatSize,1),Slice(0,None,1),Slice(0,None,1),Slice(0,None,1)});
                        //std::cout<<"CUBE LEFT "<<aCUBEL.sizes()<<std::endl;

                        auto  aSimilCUBE=F::cosine_similarity(CUBE.index({Slice(0,FeatSize,1),Slice(0,None,1),Slice(0,None,1),Slice(0,None,1)}),
                                                              CUBE.index({Slice(FeatSize,2*FeatSize,1),Slice(0,None,1),Slice(0,None,1),Slice(0,None,1)}),
                                                              F::CosineSimilarityFuncOptions().dim(0)).squeeze().to(torch::kCPU);

                        //std::cout<<"SIML CUBE "<<aSimilCUBE.sizes()<<std::endl;
                        CUBE.resize_(at::IntArrayRef{0});
                        //std::cout<<" SIMIL CUBE IS GENERATED =====+>   "<<aSimilCUBE.sizes()<<std::endl;
                        /*if (aSimilCUBE.dim()==4)
                        {
                            aSimilCUBE.squeeze();
                        }*/
                        // Fill cube cost

                        for (aPix.x()=0 ; aPix.x()<aSzL.x() ; aPix.x()++)
                        {
                                cPt2di aPAbs = aPix + mP0Z;
                                cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                                cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                                for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
                                {
                                    double aTabCost[2]={1.0,1.0};
                                    bool   aTabOk[2]={false,false};
                                    cPt2di aPC2Z(round_ni(aPC20.x()+aDz*this->StepZ()),aPC20.y());  // INTEG FOR NOW
                                    for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                                            {
                                                bool IsInside=WindInside4BL(this->DI1(),aPC1,aVMods[aK]->mCNNWin) && WindInside4BL(this->DI2(),aPC2Z,aVMods[aK]->mCNNWin);
                                                if(IsInside)
                                                {
                                                    //auto aVecR=LREmbeddingsR.slice(2,aPC2Z.y(),aPC2Z.y()+1).slice(3,aPC2Z.x(),aPC2Z.x()+1);
                                                    using namespace torch::indexing;
                                                    auto aSim=aSimilCUBE.index({Slice(aDz-aMinZmin,aDz-aMinZmin+1,1),
                                                                                    Slice(aPix.x(),aPix.x()+1,1)}); //"Similarity values issue not in bound -1 ,1 "+
                                                    ELISE_ASSERT(aSim.item<float>()<=1.01 && aSim.item<float>()>=-1.01, std::to_string(aSim.item<float>()).c_str());
                                                    aTabCost[aK] =(1-(double)aSim.item<float>())/2.0;
                                                    aTabOk[aK]=true;
                                                }
                                            }
                                    PushCost(aTabCost[0]);
                                    if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
                                    {
                                            double aC0 = ToCmpCost(aTabCost[0]);
                                            double aC1 = ToCmpCost(aTabCost[1]);
                                            mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
                                    }
                                }
                        }

                }


        }
        
    }
    else     
    {
            cPt2di aSz = aDZMin.Sz();
            cPt2di aPix;

            int aCpt=0;

            for (aPix.y()=0 ; aPix.y()<aSz.y() ; aPix.y()++)
            {
                StdOut() << "Line " << aPix.y() << " on " << aSz.y()  << "\n";
                //MakeLinePC(aPix.y(),true );
                //MakeLinePC(aPix.y(),false);
                for (aPix.x()=0 ; aPix.x()<aSz.x() ; aPix.x()++)
                {
                        cPt2di aPAbs = aPix + mP0Z;
                        cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                        cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                        for (int aDz=aDZMin.GetV(aPix) ; aDz<aDZMax.GetV(aPix) ; aDz++)
                        {
                        double aTabCost[2];
                        bool   aTabOk[2];
                    for (int aK=0 ; aK<int(aVMods.size()) ; aK++)
                                aTabCost[aK] = aVMods[aK]->ComputeCost(aTabOk[aK],aPC1,aPC20,aDz);
                        aCpt++;
                        PushCost(aTabCost[0]);

                    if (mCmpCorLearn && aTabOk[0] && aTabOk[1])
                    {
                            double aC0 = ToCmpCost(aTabCost[0]);
                            double aC1 = ToCmpCost(aTabCost[1]);
                            mImCmp.DIm().AddVBL(cPt2dr(aC1,aC0),1.0);
                    }
                        }
                }
            }
   }


   if (mCmpCorLearn)
   {
       mImCmp.DIm().ToFile("CmpCorrLearn_"+ mNamePost + ".tif");
       //BREAK_POINT("Compare Corr/Learned made");
   }

   delete mFileCube;
   delete mPyrL1;
   delete mPyrL2;
   DeleteAllAndClear(aVMods);

   return EXIT_SUCCESS;
}



};

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
using namespace  cNS_FillCubeCost;

tMMVII_UnikPApli Alloc_FillCubeCost(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliFillCubeCost(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecFillCubeCost
(
     "DM4FillCubeCost",
      Alloc_FillCubeCost,
      "Fill a cube with matching costs",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::ToDef},
      __FILE__
);



};
