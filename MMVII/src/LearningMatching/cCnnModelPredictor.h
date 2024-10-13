#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
/*#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <vector>*/
#include "MMVII_all.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "V1VII.h"
//#include <fcntl.h>
/*#include "cConvnet_Fast.h"
#include "cConvnet_Slow.h"
#include "cConvNet_Fast_BN.h"*/
#include "cMSNet.h"

#ifdef _WIN32
#include <Windows.h>
#endif

namespace F = torch::nn::functional;

namespace MMVII
{
class aCnnModelPredictor
{
    public :
       typedef cIm2D<tREAL4> tTImV2;
       aCnnModelPredictor(std::string anArchitecture, std::string aModelBinDir, bool Cuda);
       
       torch::Tensor PredictMSNet(MSNet Net,std::vector<tTImV2> aTilL,cPt2di aPSz);
       torch::Tensor PredictMSNetTile(torch::jit::script::Module mNet, tTImV2 aPatchLV, cPt2di aPSz);
       torch::Tensor PredictMSNetAtt(MSNet_Attention Net,std::vector<tTImV2> aTilL,cPt2di aPSz);
       torch::Tensor PredictMSNetHead(torch::jit::script::Module mNet, std::vector<tTImV2> aPatchLV, cPt2di aPSz);
       torch::Tensor PredictUNetWDecision(torch::jit::script::Module mNet, std::vector<tTImV2> aMasterP,std::vector<tTImV2> aPatchLV, cPt2di aPSz);
       torch::Tensor PredictUnetFeaturesOnly(torch::jit::script::Module mNet,std::vector<tTImV2> aPatchLV, cPt2di aPSz);
       torch::Tensor PredictUnetFeaturesOnly(torch::jit::script::Module mNet,torch::Tensor aPAllSlaves);
       torch::Tensor PredictMSNetTileFeatures(torch::jit::script::Module mNet, tTImV2 aPatchLV, cPt2di aPSz);
       torch::Tensor PredictDecisionNet(torch::jit::script::Module mNet, torch::Tensor Left, torch::Tensor Right);
       torch::Tensor PredictONCUBE(torch::jit::script::Module mMlp,/*torch::jit::script::Module mMatcher,*/ torch::Tensor &aCube);
       
       // 3 MOdels used to coompute features, compute similarities, enhance similarities in the 3D space
       void PopulateModelFeatures(torch::jit::script::Module & Network);
       void PopulateModelFeatures(torch::jit::script::Module & Network,bool DeviceCuda);
       void PopulateModelDecision(torch::jit::script::Module & Network);
       void PopulateModelDecision(torch::jit::script::Module & Network,bool DeviceCuda);
       void PopulateModelMatcher(torch::jit::script::Module & Network); 
       void PopulateModelMSNetHead(torch::jit::script::Module & Net);
       
       std::string Architecture(){return mArchitecture;};  // it is not useful 
        std::vector<std::string > mSetModelBinaries;
        std::string mArchitecture;
        std::string mDirModel;
        bool IsCuda=false;
};

    
};
