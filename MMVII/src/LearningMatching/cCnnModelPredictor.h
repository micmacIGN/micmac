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
#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"
#include "include/V1VII.h"
#include <fcntl.h>
#include "cConvnet_Fast.h"
#include "cConvnet_Slow.h"
#include "cConvNet_Fast_BN.h"
#include "cMSNet.h"

namespace F = torch::nn::functional;

namespace MMVII
{
    
    
template <typename T>  void Tensor2File(torch::Tensor a, std::string fname, std::string Type);
torch::Tensor ReadBinaryFile(std::string filename, torch::Tensor Host);

class aCnnModelPredictor
{
    public :
       typedef cIm2D<tREAL4> tTImV2;
       aCnnModelPredictor(std::string anArchitecture, std::string aModelBinDir);
       double Predict(ConvNet_Fast Net, tTImV2 aPatchL, tTImV2 aPatchR,cPt2di aPSz);
       double PredictWithBN(ConvNet_FastBn Net, tTImV2 aPatchL, tTImV2 aPatchR,cPt2di aPSz);
       
       double PredictSlow(ConvNet_Slow Net, tTImV2 aPatchL, tTImV2 aPatchR,cPt2di aPSz);
       
       torch::Tensor PredictTile(ConvNet_Fast  Net, tTImV2 aTilL,cPt2di aPSz);
       torch::Tensor PredictFastWithHead(FastandHead Net, tTImV2 aTilL,cPt2di aPSz);
       torch::Tensor PredictSimNetConv(SimilarityNet Net, tTImV2 aTilL,cPt2di aPSz);
       torch::Tensor PredictSimNetMLP(SimilarityNet Net, torch::Tensor Left, torch::Tensor Right);
       torch::Tensor PredictWithBNTile(ConvNet_FastBn  Net, tTImV2 aTilL,cPt2di aPSz);
       torch::Tensor PredictWithBNTileReg(ConvNet_FastBnRegister  Net, tTImV2 aTilL,cPt2di aPSz);
       torch::Tensor PredictSlowTile(ConvNet_Slow  Net, tTImV2 aTilL, tTImV2 aTilR,cPt2di aPSz);
       
       torch::Tensor PredictPrjHead(Fast_ProjectionHead Net,tTImV2 aTilL,cPt2di aPSz);
       
       torch::Tensor PredictMSNet(MSNet Net,std::vector<tTImV2> aTilL,cPt2di aPSz);
       torch::Tensor PredictMSNetTile(torch::jit::script::Module mNet, tTImV2 aPatchLV, cPt2di aPSz);
       torch::Tensor PredictMSNetAtt(MSNet_Attention Net,std::vector<tTImV2> aTilL,cPt2di aPSz);
       torch::Tensor PredictMSNetHead(/*MSNetHead*/ torch::jit::script::Module mNet, std::vector<tTImV2> aPatchLV, cPt2di aPSz);
       torch::Tensor PredictUNetWDecision(torch::jit::script::Module mNet, std::vector<tTImV2> aMasterP,std::vector<tTImV2> aPatchLV, cPt2di aPSz);
       torch::Tensor PredictUnetFeaturesOnly(torch::jit::script::Module mNet,std::vector<tTImV2> aPatchLV, cPt2di aPSz);
       torch::Tensor PredictMSNet1(MSNet Net,torch::Tensor X);
       torch::Tensor PredictMSNet2(MSNet Net,torch::Tensor X);
       torch::Tensor PredictMSNet3(MSNet Net,torch::Tensor X);
       torch::Tensor PredictMSNet4(MSNet Net,torch::Tensor X);
       torch::Tensor PredictMSNetTileFeatures(torch::jit::script::Module mNet, tTImV2 aPatchLV, cPt2di aPSz);
       torch::Tensor PredictDecisionNet(torch::jit::script::Module mNet, torch::Tensor Left, torch::Tensor Right);
       torch::Tensor PredictONCUBE(torch::jit::script::Module mMlp,/*torch::jit::script::Module mMatcher,*/ torch::Tensor &aCube);
       torch::Tensor PredictMSNetCommon(MSNet Net,tTImV2 aTilL,cPt2di aPSz);
       
       

       // 3 MOdels used to coompute features, compute similarities, enhance similarities in the 3D space
       void PopulateModelFeatures(torch::jit::script::Module & Network);
       void PopulateModelDecision(torch::jit::script::Module & Network);
       void PopulateModelMatcher(torch::jit::script::Module & Network);
       
       torch::Tensor ReadBinaryFile(std::string aFilename, torch::Tensor aHost);
       void PopulateModelFromBinary(ConvNet_Fast Net);
       void PopulateModelFromBinaryWithBN(ConvNet_FastBn Net);
       void PopulateModelFromBinaryWithBNReg(ConvNet_FastBnRegister Net); // DONE
       
       void  PopulateModelPrjHead(Fast_ProjectionHead Net);
       
       void PopulateModelMSNet(MSNet Net);
       void PopulateModelMSNetAtt(MSNet_Attention Net);
       void PopulateModelMSNetHead(/*MSNetHead*/  torch::jit::script::Module & Net);
       
       
       void PopulateModelFastandHead(FastandHead Network);
       void PopulateModelSimNet(SimilarityNet Network);
       void PopulateSlowModelFromBinary(ConvNet_Slow Net);
       cPt2di GetWindowSize(ConvNet_Fast Network);
       cPt2di GetWindowSizeBN(ConvNet_FastBn Network);
       cPt2di GetWindowSizeBNReg(ConvNet_FastBnRegister Network);  // DONE REPLACED  ConvNet_FastBnRegister 
       
       cPt2di GetWindowSizeFastandHead(FastandHead Network);
       cPt2di GetWindowSizeSimNet(SimilarityNet Network);
       cPt2di GetSlowWindowSize(ConvNet_Slow Network);
       
       cPt2di GetWindowSizePrjHead(Fast_ProjectionHead Network);
        
       std::string Architecture(){return mArchitecture;};  // it is not useful 
       
                std::vector<std::string > mSetModelBinaries;
        std::string mArchitecture;
        std::string mDirModel;
        /*ConvNet_Fast mNetFastStd;
        ConvNet_FastBn mNetFastMVCNN;
        ConvNet_Slow mNetSlowStd;
        */
};

    
};
