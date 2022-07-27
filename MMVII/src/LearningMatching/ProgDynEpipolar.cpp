#include "include/MMVII_all.h"
//#include "include/V1VII.h"
#include "include/MMVII_Tpl_Images.h"
#include "LearnDM.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fstream>
#include "../../bin/SGM_CUDA/cudInfer.cuh"


namespace F = torch::nn::functional;
torch::Device TheCPUDevice(torch::kCPU);
auto cuda_available=torch::cuda::is_available();
//torch::Device TheGPUDevice=cuda_available ? torch::kCUDA : torch::kCPU;
torch::Device TheGPUDevice(torch::kCPU);
/*********************************************************************/
template <typename T> void Tensor2File(torch::Tensor a, std::string fname, std::string Type)
{
   //Store Tensor 
   T * TensorContent=a.data_ptr<T>();
   FILE *finaldestination = fopen(fname.c_str(), "wb");
   fwrite(TensorContent, sizeof(T), a.numel(), finaldestination);
   fclose(finaldestination);
   //Store dimensions 
   std::string dimFname=fname;
   dimFname.append(".dim");
   std::ofstream finaldestinationDim(dimFname.c_str());
   
   for (int dd=0;dd<a.dim();dd++)
   {
      finaldestinationDim<<a.size(dd)<<std::endl;
   }
   finaldestinationDim.close();
   
   //Store data type 
   std::ofstream datatypetensor(fname.append(".type").c_str());
   datatypetensor<<Type<<std::endl;
   datatypetensor.close();
}

/*********************************************************************/
namespace MMVII
{
    struct OptimOptions   
    {
        int L1            =  4;    // For Homogenous regions definition : threshold on region extent
        tREAL4 cbca_i1    =  2;
        tREAL4 cbca_i2    =  4;
        tREAL4 tau1       =  0.05; // For Homogenous regions definition : threshold on intensity
        tREAL4 sgm_i      =  1;
        tREAL4 sgm_q1     =  4.5; // P1 AND p2 ARE SCALED by sgm_q1 if on of the image have high gradient
        tREAL4 sgm_q2     =  2.0;// P1 AND p2 ARE SCALED by sgm_q1*sgm_q2 if both images have high gradient
        tREAL4 alpha1     =  3.0; // P1 and P2 are scaled by alpha1 if in transverse parallax direction : less disparity change ==> lower penalisation
        tREAL4 tau_so     =  0.1;
        tREAL4 blur_sigma =  1.7;
        tREAL4 blur_t     =  2.0;
    }; 
class cAppliProgDynEpipolar: public cAppliLearningMatch
{
     public :
        typedef cIm2D<tREAL4>              tImRad;   
        typedef cDataIm2D<tREAL4>          tDataImRad;

        cAppliProgDynEpipolar(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        void PopulateModel(torch::jit::script::Module & Network, std::string ModelPath);
        void DoInference();
        void DoInferenceFillCost();
        void DoInferenceFillCostOneForward();
        torch::Tensor gaussian(float sigma);
        // -------------- Mandatory args -------------------
        std::string mNameI1;
        std::string mNameI2;
        std::string mNameImPax;
        std::string mModelPath,mDecisonModelPath;
        std::string P1,P2;
        tREAL4 mP1=0.02;
        tREAL4 mP2=1.0;
        int mDispRange=192;
        cBox2di BoxFile1() const {return cDataFileIm2D::Create(mNameI1,false);}
        cBox2di BoxFile2() const {return cDataFileIm2D::Create(mNameI2,false);}

        tImRad      mIm1;
        tDataImRad  *mDI1;
        tImRad      mIm2;
        tDataImRad  *mDI2;
        tImRad      mImDisp;
        tDataImRad  *mDIImDisp;
        torch::jit::script::Module mSimilarityModel;
        torch::jit::script::Module mDecisionNetwork;
        torch::Tensor aLeftImageT,aRightImageT;
        // Aggregation parameters 
        OptimOptions AggregParams;
     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
};

cAppliProgDynEpipolar::cAppliProgDynEpipolar(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mIm1                 (cPt2di(1,1)),
   mDI1                 (nullptr),
   mIm2                 (cPt2di(1,1)),
   mDI2                 (nullptr),
   mImDisp              (cPt2di(1,1)),
   mDIImDisp            (nullptr)
   
{
}



cCollecSpecArg2007 & cAppliProgDynEpipolar::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mModelPath,"Model script module full path ")
          <<   Arg2007(mDecisonModelPath,"Model script module of similarity computation full path ")
          <<   Arg2007(mNameI1,"Name of first image")
          <<   Arg2007(mNameI2,"Name of second image")
          <<   Arg2007(mNameImPax,"Name of prallax image")
          //<<   Arg2007(mDispRange, "Upper bound of disparity range lower is 0 by default: Let ")
   ;
}

cCollecSpecArg2007 & cAppliProgDynEpipolar::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(P1, "P1","PENALTY FOR SUFFICIENTLY CLOSE DISPARITIES AT NEIGHBOURING REGIONS",{eTA2007::HDV})
           << AOpt2007(P2, "P2","PENALTY COEFFICIENT for LARGER DISPARITIES",{eTA2007::HDV})
   ;
}

void cAppliProgDynEpipolar::PopulateModel(torch::jit::script::Module & Network, std::string ModelPath)
{
    Network=torch::jit::load(ModelPath);
    Network.to(TheGPUDevice);
    torch::NoGradGuard no_grad;
    Network.eval();  
}

/***********************************************************************/
torch::Tensor cAppliProgDynEpipolar::gaussian(float sigma)
{
   int kr=ceil(sigma/3);
   int ks=kr*2+1;
   torch::Tensor K=torch::empty({ks,ks},torch::TensorOptions().dtype(torch::kFloat32));
   for (int i=0;i<ks;i++)
   {
     for (int j=0;j<ks;j++)
        {
			float y=i-kr;
			float x=j-kr;
			float val=exp(-(x * x + y * y) / (2 * sigma * sigma));
            K.index_put_({i,j},val);
        }
   }
   return K;
}


/***********************************************************************/
void cAppliProgDynEpipolar::DoInferenceFillCost()
{
    torch::Tensor vol = torch::ones({1, mDispRange, aLeftImageT.size(2), aLeftImageT.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor disp = torch::ones({1, 1, aLeftImageT.size(2), aLeftImageT.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    vol=vol.mul(0.5);
    using namespace torch::indexing;
    for (int d=0;d<mDispRange;d++)
    {
        torch::Tensor aSlicedLeft=aLeftImageT.slice(3,d,aLeftImageT.size(3),1); // left image slice at certain disparities 
        torch::Tensor aSlicedRight=aRightImageT.slice(3,0,aRightImageT.size(3)-d,1); 
        // Inference and fill cost volume with 1-Similarity 
        torch::jit::IValue inp(torch::cat({aSlicedLeft,aSlicedRight},0));
        std::vector<torch::jit::IValue> allinp={inp};
        torch::NoGradGuard no_grad_guard;
        auto out=mSimilarityModel.forward(allinp);
        auto Simil=out.toTensor().squeeze().to(TheCPUDevice);
        vol.index({0,d,Slice(0,None,1),Slice(d,aLeftImageT.size(3),1)}).copy_(Simil.mul(-1).add(1));
    }
    //Cost Based Cross Aggregation CBCA
    torch::Tensor x0c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor x1c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    cudaDeviceSynchronize();
    Cross(aLeftImageT, x0c, AggregParams.L1, AggregParams.tau1); 
    cudaDeviceSynchronize();
    Cross(aRightImageT, x1c, AggregParams.L1, AggregParams.tau1); 
    cudaDeviceSynchronize();
    torch::Tensor tmp_cbca = torch::empty({1, mDispRange, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    for (int i=0;i<AggregParams.cbca_i1;i++)
    {
        std::cout<<"================> COST AGGREGATION"<<std::endl;
        CrBaCoAgg(x0c,x1c,vol,tmp_cbca,-1);
        vol.copy_(tmp_cbca);
    }
    // SEMI GLOBAL MATCHING 
    //vol=vol.transpose(1,2).transpose(2,3).clone();
    vol=at::transpose(at::transpose(vol,1,2),2,3).contiguous();
    torch::Tensor out = torch::zeros({1, vol.size(1), vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor tmp = torch::zeros({vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    for (int i=0;i<AggregParams.sgm_i;i++)
    {
        out=out.mul(0.0);
        std::cout<<"================> SGM SGM SGM SGM SGM "<<std::endl;
        sgm2(aLeftImageT, aRightImageT, vol, out, tmp, mP1, mP2, AggregParams.tau_so,
            AggregParams.alpha1, AggregParams.sgm_q1, AggregParams.sgm_q2, -1);
        vol.copy_(out.div(4.0));
    }
    vol=vol.reshape({1,mDispRange,aLeftImageT.size(2),aLeftImageT.size(3)});
    vol.copy_(at::transpose(at::transpose(out,2,3),1,2).contiguous());
    vol=vol.div(4.0);
    std::cout<<"Left and Right cost volumes are constructed !"<<std::endl;
    std::tuple<torch::Tensor, torch::Tensor> d_Tpl = at::min(vol,1);
    torch::Tensor indexes=std::get<1>(d_Tpl);
    disp.index({0,0,Slice(0,None,1),Slice(0,None,1)}).copy_(indexes.squeeze()); 
    
    
    // Disparity postprocessing 
    torch::Tensor out3 = torch::zeros(disp.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor out4 = torch::zeros(disp.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor out5 = torch::zeros(disp.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));    
    subpixel_enchancement(disp, vol, out3, mDispRange);                 
    median2d(out3,out4,5);   
    mean2d(out4, gaussian(AggregParams.blur_sigma), out5, AggregParams.blur_t);      
    
    // The output disparity map can no be stored with the same size as the left tile or image
    tREAL4 ** tdispData=mDIImDisp->ExtractRawData2D();
    std::memcpy((*tdispData),out5.data_ptr<tREAL4>(),sizeof(tREAL4)*out5.numel());
    mDIImDisp->ToFile(mNameImPax);
}


/***********************************************************************/
void cAppliProgDynEpipolar::DoInferenceFillCostOneForward()
{
    torch::Tensor vol = torch::ones({1, mDispRange, aLeftImageT.size(2), aLeftImageT.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor disp = torch::ones({1, 1, aLeftImageT.size(2), aLeftImageT.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    vol=vol.mul(0.5);
    using namespace torch::indexing;
    // Forward Features Only 
    torch::jit::IValue inp(torch::cat({aLeftImageT,aRightImageT},0));
    std::vector<torch::jit::IValue> allinp={inp};
    torch::NoGradGuard no_grad_guard;
    auto Feats=mSimilarityModel.forward(allinp);
    auto Features=Feats.toTensor(); 
    
    auto l=Features.slice(0,0,1);
    auto r=Features.slice(0,1,2); 
    for (int d=0;d<mDispRange;d++)
    {
        torch::Tensor aSlicedLeft=l.slice(3,d,aLeftImageT.size(3),1); // left image slice at certain disparities 
        torch::Tensor aSlicedRight=r.slice(3,0,aRightImageT.size(3)-d,1); 
        // Inference and fill cost volume with 1-Similarity 
        torch::jit::IValue inp(torch::cat({aSlicedLeft,aSlicedRight},1));
        std::vector<torch::jit::IValue> allinp={inp};
        torch::NoGradGuard no_grad_guard;
        auto out=mDecisionNetwork.forward(allinp);
        auto Simil=out.toTensor().squeeze().to(TheCPUDevice);
        vol.index({0,d,Slice(0,None,1),Slice(d,aLeftImageT.size(3),1)}).copy_(torch::sigmoid(Simil).mul(-1).add(1));
    }
    //Cost Based Cross Aggregation CBCA
    torch::Tensor x0c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor x1c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    cudaDeviceSynchronize();
    Cross(aLeftImageT, x0c, AggregParams.L1, AggregParams.tau1); 
    cudaDeviceSynchronize();
    Cross(aRightImageT, x1c, AggregParams.L1, AggregParams.tau1); 
    cudaDeviceSynchronize();
    torch::Tensor tmp_cbca = torch::empty({1, mDispRange, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    for (int i=0;i<AggregParams.cbca_i1;i++)
    {
        std::cout<<"================> COST AGGREGATION"<<std::endl;
        CrBaCoAgg(x0c,x1c,vol,tmp_cbca,-1);
        vol.copy_(tmp_cbca);
    }
    // SEMI GLOBAL MATCHING 
    //vol=vol.transpose(1,2).transpose(2,3).clone();
    vol=at::transpose(at::transpose(vol,1,2),2,3).contiguous();
    torch::Tensor out = torch::zeros({1, vol.size(1), vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor tmp = torch::zeros({vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    for (int i=0;i<AggregParams.sgm_i;i++)
    {
        out=out.mul(0.0);
        std::cout<<"================> SGM SGM SGM SGM SGM "<<std::endl;
        sgm2(aLeftImageT, aRightImageT, vol, out, tmp, mP1, mP2, AggregParams.tau_so,
            AggregParams.alpha1, AggregParams.sgm_q1, AggregParams.sgm_q2, -1);
        vol.copy_(out.div(4.0));
    }
    vol=vol.reshape({1,mDispRange,aLeftImageT.size(2),aLeftImageT.size(3)});
    vol.copy_(at::transpose(at::transpose(out,2,3),1,2).contiguous());
    vol=vol.div(4.0);
    //  ANOTHER CBCA 2
    for (int i=0;i<AggregParams.cbca_i2;i++)
    {
        std::cout<<"================> COST AGGREGATION "<<std::endl;
        CrBaCoAgg(x0c, x1c, vol, tmp_cbca, -1);
        cudaDeviceSynchronize();
        vol.copy_(tmp_cbca);
    }   
    
    std::tuple<torch::Tensor, torch::Tensor> d_Tpl = at::min(vol,1);
    torch::Tensor indexes=std::get<1>(d_Tpl);
    disp.index({0,0,Slice(0,None,1),Slice(0,None,1)}).copy_(indexes.squeeze()); 
    // Disparity postprocessing 
    torch::Tensor out3 = torch::zeros(disp.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor out4 = torch::zeros(disp.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor out5 = torch::zeros(disp.sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));    
    subpixel_enchancement(disp, vol, out3, mDispRange);                 
    median2d(out3,out4,5);   
    mean2d(out4, gaussian(AggregParams.blur_sigma), out5, AggregParams.blur_t);      
    
    // The output disparity map can no be stored with the same size as the left tile or image
    tREAL4 ** tdispData=mDIImDisp->ExtractRawData2D();
    std::memcpy((*tdispData),out5.data_ptr<tREAL4>(),sizeof(tREAL4)*out5.numel());
    mDIImDisp->ToFile(mNameImPax);
}
/***********************************************************************/
void cAppliProgDynEpipolar::DoInference()
{
    int mb_directions[1]={-1};
    /*******************************************************************/
    torch::Tensor vols = torch::ones({2, mDispRange, aLeftImageT.size(2), aLeftImageT.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor disp = torch::ones({2, 1, aLeftImageT.size(2), aLeftImageT.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    vols=vols.mul(0.5);
    using namespace torch::indexing;
    for (int d=0;d<mDispRange;d++)
    {
        torch::Tensor aSlicedLeft=aLeftImageT.slice(3,d,aLeftImageT.size(3),1); // left image slice at certain disparities 
        torch::Tensor aSlicedRight=aRightImageT.slice(3,0,aRightImageT.size(3)-d,1); 
        // Inference and fill cost volume with 1-Similarity 
        torch::jit::IValue inp(torch::cat({aSlicedLeft,aSlicedRight},0));
        std::vector<torch::jit::IValue> allinp={inp};
        torch::NoGradGuard no_grad_guard;
        auto out=mSimilarityModel.forward(allinp);
        auto Simil=out.toTensor().squeeze().to(TheCPUDevice);
        //std::cout<<vols.index({0,d,Slice(0,None,1),Slice(d,aLeftImageT.size(3),1)}).sizes()<<std::endl;
        std::cout<<vols.index({0,d,Slice(0,None,1),Slice(d,aLeftImageT.size(3),1)}).sizes()<<std::endl;
        //std::cout<<"simil "<<Simil.mul(-1).add(1).sizes()<<std::endl;
        vols.index({0,d,Slice(0,None,1),Slice(d,aLeftImageT.size(3),1)}).copy_(Simil.mul(-1).add(1));
        vols.index({1,d,Slice(0,None,1),Slice(0,aLeftImageT.size(3)-d,1)}).copy_(Simil.mul(-1).add(1));
    }
    std::cout<<"Left and Right cost volumes are constructed !"<<std::endl;
    
    for (auto direction : mb_directions)
    {
        auto vol=vols.slice(0,direction == -1 ? 0 : 1, direction == -1 ? 1 : 2,1).contiguous();
        torch::Tensor x0c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
        torch::Tensor x1c=torch::empty({1, 4, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
        cudaDeviceSynchronize();
        Cross(aLeftImageT, x0c, AggregParams.L1, AggregParams.tau1); 
        cudaDeviceSynchronize();
        Cross(aRightImageT, x1c, AggregParams.L1, AggregParams.tau1); 
        cudaDeviceSynchronize();
        torch::Tensor tmp_cbca = torch::empty({1, mDispRange, vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
        for (int i=0;i<AggregParams.cbca_i1;i++)
        {
            std::cout<<"================> COST AGGREGATION"<<std::endl;
            CrBaCoAgg(x0c,x1c,vol,tmp_cbca,direction);
            vol.copy_(tmp_cbca);
        }
        tmp_cbca=tmp_cbca.mul(0);
        cudaDeviceSynchronize();
        // SGM 
        vol=vol.transpose(1,2).transpose(2,3).clone().contiguous();
        torch::Tensor out = torch::zeros({1, vol.size(1), vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
        torch::Tensor tmp = torch::zeros({vol.size(2), vol.size(3)},torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
        for (int i=0;i<AggregParams.sgm_i;i++)
        {
            std::cout<<"================> SGM SGM SGM SGM SGM "<<std::endl;
            sgm2(aLeftImageT, aRightImageT, vol, out, tmp, mP1, mP2, AggregParams.tau_so,
                AggregParams.alpha1, AggregParams.sgm_q1, AggregParams.sgm_q2, direction);
            vol.copy_(out.div(4.0));
        }
        vol=vol.reshape({1,mDispRange,aLeftImageT.size(2),aLeftImageT.size(3)});
        //vol.copy_(out.transpose(3,2).transpose(2,1).div(4.0));
        //  ANOTHER CBCA 2
        for (int i=0;i<AggregParams.cbca_i2;i++)
        {
            std::cout<<"================> COST AGGREGATION "<<std::endl;
            CrBaCoAgg(x0c, x1c, vol, tmp_cbca, direction);
            cudaDeviceSynchronize();
            vol.copy_(tmp_cbca);
        }
        std::tuple<torch::Tensor, torch::Tensor> d_Tpl = at::min(vol,1);
        torch::Tensor indexes=std::get<0>(d_Tpl);
        //std::cout<<"indexes shape "<<indexes.sizes()<<std::endl;
        disp.index({direction == -1 ? 0 : 1,0,Slice(0,None,1),Slice(0,None,1)}).copy_(indexes.squeeze()); 
        vols.index({direction == -1 ? 0 : 1,Slice(0,None,1),Slice(0,None,1),Slice(0,None,1)}).copy_(vol.squeeze());
    }
    
    // All Subsequent steps that allow to handle filtering and interpolation 
    /*torch::Tensor outlier = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor out = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor out2 = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    */
    torch::Tensor out3 = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor out4 = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    torch::Tensor out5 = torch::zeros(disp.slice(0,1,2,1).sizes(),torch::TensorOptions().dtype(torch::kFloat32).device(TheCPUDevice));
    
    /*outlier_detection(disp.slice(0,1,2,1), disp.slice(0,0,1,1), outlier, mDispRange);
    interpolate_occlusion(disp.slice(0,1,2,1), outlier,out);       
    interpolate_mismatch(out, outlier,out2);  */
    
    subpixel_enchancement(disp.slice(0,1,2,1), vols.slice(0,0,1,1), out3, mDispRange);                 
    median2d(out3,out4,5);   
    mean2d(out4, gaussian(AggregParams.blur_sigma), out5, AggregParams.blur_t);   
    
    std::cout<<"Saving the image "<<std::endl;    
    // The output disparity map can no be stored with the same size as the left tile or image
    tREAL4 ** tdispData=mDIImDisp->ExtractRawData2D();

    std::memcpy((*tdispData),out5.data_ptr<tREAL4>(),sizeof(tREAL4)*out5.numel());
    mDIImDisp->ToFile(mNameImPax);
}


int  cAppliProgDynEpipolar::Exe()
{
   if (P1!="") mP1=std::stof(P1);
   if (P2!="") mP2=std::stof(P2);
   mIm1 = tImRad::FromFile(mNameI1);
   mDI1 = &(mIm1.DIm());
   mIm2 = tImRad::FromFile(mNameI2);
   mDI2 = &(mIm2.DIm());
   cPt2di aSz1 = mDI1->Sz();
   cPt2di aSz2 = mDI2->Sz();
   mImDisp = tImRad(aSz1,nullptr,eModeInitImage::eMIA_Null);
   mDIImDisp = &(mImDisp.DIm());
   MMVII_INTERNAL_ASSERT_always(aSz1.x()==aSz2.x() && aSz1.y()==aSz2.y(), "Images sizes should match !");
   // Generate Tensors from both images and assert equality between left and right 
   tREAL4 ** mL1Data=mDI1->ExtractRawData2D();
   tREAL4 ** mL2Data=mDI2->ExtractRawData2D();
   aLeftImageT=torch::from_blob((*mL1Data), {1,1,aSz1.y(),aSz1.x()}, torch::TensorOptions().dtype(torch::kFloat32)).to(TheGPUDevice);
   aRightImageT=torch::from_blob((*mL2Data), {1,1,aSz2.y(),aSz2.x()}, torch::TensorOptions().dtype(torch::kFloat32)).to(TheGPUDevice);
   // Load Model 
   this->PopulateModel(mSimilarityModel,mModelPath);
   this->PopulateModel(mDecisionNetwork,mDecisonModelPath);
   // Do inference and compute the disparity map 
   this->DoInferenceFillCostOneForward();
   
   return EXIT_SUCCESS;
}
   
tMMVII_UnikPApli Alloc_ProgDynEpipolar(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliProgDynEpipolar(aVArgs,aSpec));
}

cSpecMMVII_Appli TheSpecProgDynEpipolar
(
     "SGMCUDA_IN_MM",
      Alloc_ProgDynEpipolar,
      "Stereo matching in Epipolar geometry and Optimization with SGM CUDA",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::ToDef},
      __FILE__
);



};
