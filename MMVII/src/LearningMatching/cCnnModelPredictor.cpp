/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr
   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
// Tool for calculating disparity between two Tiles using a CNN Trained Model 
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
/*#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include "cConvnet_Fast.h"
#include "cConvnet_Slow.h"
#include "cConvNet_Fast_BN.h"*/
#include "cCnnModelPredictor.h"

namespace F = torch::nn::functional;

namespace MMVII
{
/**************************************************************************/
aCnnModelPredictor::aCnnModelPredictor(std::string anArchitecture, std::string aModelBinDir, bool Cuda):
    mArchitecture(anArchitecture),IsCuda(Cuda)
{
    // FILL THE SET OF BINARY FILES NAMES 
    std::string aModelPat,aDirModel;
	SplitDirAndFile(aDirModel, aModelPat, aModelBinDir,false);
	cInterfChantierNameManipulateur * aICNMModel=cInterfChantierNameManipulateur::BasicAlloc(aDirModel);
	mSetModelBinaries = *(aICNMModel->Get(aModelPat));
    mDirModel=aDirModel;
}

/***********************************************************************/
void aCnnModelPredictor::PopulateModelMSNetHead(/*MSNetHead Network*/ torch::jit::script::Module & Network)
{
    //StdOut()<<"TO LOAD MODEL "<<"\n";
    std::string aModel=mDirModel+mSetModelBinaries.at(0); // just one pickled model 
    Network=torch::jit::load(aModel);
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    Network.to(device);
    StdOut()<<"TORCH LOAD  "<<"\n";
}
/***********************************************************************/
void aCnnModelPredictor::PopulateModelFeatures(torch::jit::script::Module & Network)
{
    // add a convention on Model Name TAKE FOR EXAMPLES FEATURES AS A KEY FOR THE FEATURE MODULE 
    std::string aModel;
    for (unsigned int i=0;i<mSetModelBinaries.size();i++)
    {
        if (mSetModelBinaries.at(i).find("FEATURES") != std::string::npos)
        {
            aModel=mDirModel+mSetModelBinaries.at(i);
            std::cout<<"Models checked "<<mSetModelBinaries.at(i)<<std::endl;
            break;
        }
    }
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    //auto cuda_available =torch::cuda::is_available();
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    StdOut()<<"Model Name "<<aModel<<"\n";
    torch::Device device(IsCuda ? torch::kCUDA : torch::kCPU);
    Network=torch::jit::load(aModel);
    Network.to(device);
    StdOut()<<"MODEL FEATURES LOADED !!!!!! "<<"\n";
}

/***********************************************************************/
void aCnnModelPredictor::PopulateModelFeatures(torch::jit::script::Module & Network,bool DeviceCuda)
{
    // add a convention on Model Name TAKE FOR EXAMPLES FEATURES AS A KEY FOR THE FEATURE MODULE
    std::string aModel;
    for (unsigned int i=0;i<mSetModelBinaries.size();i++)
    {
        if (mSetModelBinaries.at(i).find("FEATURES") != std::string::npos)
        {
            aModel=mDirModel+mSetModelBinaries.at(i);
            std::cout<<"Models checked "<<mSetModelBinaries.at(i)<<std::endl;
            break;
        }
    }
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    //auto cuda_available = WhichDevice;  //torch::cuda::is_available();
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    StdOut()<<"Model Name "<<aModel<<"\n";
    //auto cuda_available=false;
    torch::Device device(DeviceCuda ? torch::kCUDA : torch::kCPU);
    try
    {
            Network= torch::jit::load(aModel);
    }
    catch (std::exception& e)
    {
            std::cout << e.what() << std::endl;
    }
    //Network=torch::jit::load(aModel);
    Network.to(device);
    StdOut()<<"MODEL FEATURES LOADED !!!!!! "<<"\n";
}
/***************************************************************************************/

void aCnnModelPredictor::PopulateModelDecision(torch::jit::script::Module & Network)
{
    // add a convention on Model Name TAKE FOR EXAMPLES FEATURES AS A KEY FOR THE FEATURE MODULE 
    std::string aModel;
    for (unsigned int i=0;i<mSetModelBinaries.size();i++)
    {
        if (mSetModelBinaries.at(i).find("DECISION_NET") != std::string::npos)
        {
            std::cout<<"Models checked for the decision NEtwork"<<mSetModelBinaries.at(i)<<std::endl;
            aModel=mDirModel+mSetModelBinaries.at(i);
            break;
        }
    }
    //auto cuda_available = torch::cuda::is_available();
    torch::Device device(IsCuda ? torch::kCUDA : torch::kCPU);
    Network=torch::jit::load(aModel);
    Network.to(device);
    StdOut()<<"MODEL DECISION LOADED !!  "<<"\n";
}

/***************************************************************************************/

void aCnnModelPredictor::PopulateModelDecision(torch::jit::script::Module & Network,bool DeviceCuda)
{
    // add a convention on Model Name TAKE FOR EXAMPLES FEATURES AS A KEY FOR THE FEATURE MODULE
    std::string aModel;
    for (unsigned int i=0;i<mSetModelBinaries.size();i++)
    {
        if (mSetModelBinaries.at(i).find("DECISION_NET") != std::string::npos)
        {
            std::cout<<"Models checked for the decision NEtwork"<<mSetModelBinaries.at(i)<<std::endl;
            aModel=mDirModel+mSetModelBinaries.at(i);
            break;
        }
    }
    torch::Device device(DeviceCuda ? torch::kCUDA : torch::kCPU);
    Network=torch::jit::load(aModel);
    Network.to(device);
    StdOut()<<"MODEL DECISION LOADED !!  "<<"\n";
}
/***************************************************************************************/

void aCnnModelPredictor::PopulateModelMatcher(torch::jit::script::Module & Network)
{
    // add a convention on Model Name TAKE FOR EXAMPLES FEATURES AS A KEY FOR THE FEATURE MODULE
    std::string aModel;
    for (unsigned int i=0;i<mSetModelBinaries.size();i++)
    {
        if (mSetModelBinaries.at(i).find("MATCHER_NET") != std::string::npos)
        {
            std::cout<<"Models checked for the decision NEtwork"<<mSetModelBinaries.at(i)<<std::endl;
            aModel=mDirModel+mSetModelBinaries.at(i);
            break;
        }
    }
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    Network=torch::jit::load(aModel);
    Network.to(device);
    StdOut()<<"MODEL MATCHER LOADED !!  "<<"\n";
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNet(MSNet mNet, std::vector<tTImV2> aPatchLV, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    torch::Tensor aPAllScales=torch::empty({1,4,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));;
    for (int cc=0;cc<(int) aPatchLV.size();cc++)
    {
        tREAL4 ** mPatchLData=aPatchLV.at(cc).DIm().ExtractRawData2D();
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
        aPAllScales.index_put_({cc},aPL);
    }

    // 4 scale tensor is needed for now test by passing the same tensor at each stage of the network 
    /*torch::Tensor a4ScaleTens=aPL.repeat_interleave(4,1);
    std::cout<<" a4ScaleTens size "<<a4ScaleTens.sizes()<<std::endl;
    assert
    (
      (a4ScaleTens.size(1)==4)  
    );*/
    auto output=mNet->forward(aPAllScales).squeeze();
    return output;
    
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictUNetWDecision(torch::jit::script::Module mNet, std::vector<tTImV2> aMasterP,std::vector<tTImV2> aPatchLV, cPt2di aPSz)
{
    //auto cuda_available = torch::cuda::is_available();
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    auto cuda_available=false;
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    //std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    //std::cout<<"master vector sizes <<   "<<aMasterP.size()<<std::endl;
    //std::cout<<"slaves vector sizes <<   "<<aPatchLV.size()<<std::endl;
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard no_grad;
    mNet.eval();
    torch::Tensor aPAllMasters=torch::empty({(int) aMasterP.size(),aPSz.y(),aPSz.x()},
                                            torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor aPAllSlaves=torch::empty({(int) aPatchLV.size(),aPSz.y(),aPSz.x()},
                                           torch::TensorOptions().dtype(torch::kFloat32));
    for (int cc=0;cc<(int) aMasterP.size();cc++)
    {
        tREAL4 ** mPatchLData=aMasterP.at(cc).DIm().ExtractRawData2D();
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,aPSz.y(),aPSz.x()},
                                           torch::TensorOptions().dtype(torch::kFloat32));
        //normalize apl
        //auto std=aPL.std();
        //aPL=aPL.sub(aPL.mean());
        //aPL=aPL.div(std.add(1e-12));
        //std::cout<<"  PATCH CONTENT "<<aPL<<std::endl;
        aPL=aPL.div(255.0);
        aPL=(aPL.sub(0.4353755468)).div(0.19367880);
        aPAllMasters.index_put_({cc},aPL);
    }
    //StdOut()<<"master "<<aPAllMasters.sizes()<<"\n";
    for (int cc=0;cc<(int) aPatchLV.size();cc++)
    {
        tREAL4 ** mPatchLData=aPatchLV.at(cc).DIm().ExtractRawData2D();
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,aPSz.y(),aPSz.x()},
                                           torch::TensorOptions().dtype(torch::kFloat32));
        //auto std=aPL.std();
        //aPL=aPL.sub(aPL.mean());
        //aPL=aPL.div(std.add(1e-12));
        aPL=aPL.div(255.0);
        aPL=(aPL.sub(0.4353755468)).div(0.19367880);
        aPAllSlaves.index_put_({cc},aPL);
    }
    auto aPAll=torch::cat({aPAllMasters.unsqueeze(0),aPAllSlaves.unsqueeze(0)},0).to(device); // tensor of size 2,1,W,H
    //StdOut()<<"Patches "<<aPAll.sizes()<<"\n";
    torch::jit::IValue inp(aPAll);
    std::vector<torch::jit::IValue> allinp={inp};
    auto out=mNet.forward(allinp);
    auto output=out.toTensor().squeeze();
    return output.to(torch::kCPU);
}

/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictUnetFeaturesOnly(torch::jit::script::Module mNet,std::vector<tTImV2> aPatchLV, cPt2di aPSz)
{
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    //std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    torch::Device device(IsCuda ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard no_grad;
    mNet.eval();
    torch::Tensor aPAllSlaves=torch::empty({(int) aPatchLV.size(),aPSz.y(),aPSz.x()},
                                           torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    for (int cc=0;cc<(int) aPatchLV.size();cc++)
    {
        tREAL4 ** mPatchLData=aPatchLV.at(cc).DIm().ExtractRawData2D();
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,aPSz.y(),aPSz.x()},
                                           torch::TensorOptions().dtype(torch::kFloat32));
        aPL=aPL.div(255.0);
        //aPL=(aPL.sub(at::min(aPL))).div(at::max(aPL)-at::min(aPL));
        //aPL=(aPL.sub(0.4353755468)).div(0.19367880); //0.434583236,0.1948717255
        //aPL=(aPL.sub(0.434583236)).div(0.1948717255);
        // Aerial data 22 cm resolution
        // Images Gregoire Maillet :
        //     ----> aPL=(aPL.sub(0.20912810375666974)).div(0.08828173006933751);
        //aPL=(aPL.sub(0.49877)).div(0.0895);
        aPL=(aPL.sub(0.3489)).div(0.25);
        // Images Vaihingen
        //aPL=(aPL.sub(0.37205569556786616)).div(0.15318937508043667);
        // Zone 1 Toulouse
        //aPL=(aPL.sub(0.27758397098638876)).div(0.19629460091512405);
        // Zone urbaine Toulouse
        //aPL=(aPL.sub(0.31662110673531746)).div(0.22801173902559266);
        // Toulouse umbra Urban Dense
         //aPL=(aPL.sub(0.38217)).div(0.307);
         //aPL=(aPL.sub(0.42512)).div(0.18);
        //aPL=(aPL.sub(at::mean(aPL))).div(at::std(aPL)+1e-6);
        aPAllSlaves.index_put_({cc},aPL.to(device));
    }
    // rotate by 90°
    //aPAllSlaves=aPAllSlaves.rot90(1,{1,2});
    torch::jit::IValue inp(aPAllSlaves.unsqueeze(0));
    std::vector<torch::jit::IValue> allinp={inp};
    auto out=mNet.forward(allinp);
    auto output=out.toTensor().squeeze();
    //output=output.rot90(3,{1,2});
    // annuler la rotation de 90°
    return output;//.to(torch::kCPU);
}

/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictUnetFeaturesOnly(torch::jit::script::Module mNet,torch::Tensor aPAllSlaves)
{
    //std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    torch::Device device(IsCuda ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard no_grad;
    mNet.eval();
    aPAllSlaves=aPAllSlaves.to(device);
    torch::jit::IValue inp(aPAllSlaves.unsqueeze(0).unsqueeze(0));
    std::vector<torch::jit::IValue> allinp={inp};
    auto out=mNet.forward(allinp);
    auto output=out.toTensor().squeeze();
    return output;//.to(torch::kCPU);
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNetTile(torch::jit::script::Module mNet, tTImV2 aPatchLV, cPt2di aPSz)
{
    auto cuda_available = torch::cuda::is_available();
    //std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    torch::Device device(cuda_available ? torch::kCPU : torch::kCPU);
    torch::NoGradGuard no_grad;
    mNet.eval();
    tREAL4 ** mPatchLData=aPatchLV.DIm().ExtractRawData2D();
    torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    torch::jit::IValue inp(aPL);
    std::vector<torch::jit::IValue> allinp={inp};
    auto out=mNet.forward(allinp);
    auto output=out.toTensor();
    return output;
}
/**********************************************************************************************************************/
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNetTileFeatures(torch::jit::script::Module mNet, tTImV2 aPatchLV, cPt2di aPSz)
{
    //auto cuda_available = torch::cuda::is_available();
    //std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    torch::Device device(IsCuda ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard no_grad;
    mNet.eval();
    tREAL4 ** mPatchLData=aPatchLV.DIm().ExtractRawData2D();
    torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    // Normalize The tile with respect to the dataset configuration
    // print image content
    //std::cout<<"TILE CONTENT  ========= >  "<<aPL<<std::endl;
    aPL=aPL.div(255.0);
    aPL=(aPL.sub(0.4357159999)).div(0.1951853861);
    //aPL=(aPL.sub(at::min(aPL))).div(at::max(aPL)-at::min(aPL));
    //***********************************aPL=(aPL.sub(0.4357159999)).div(0.1951853861); //0.4357159999,0.1951853861 0.434583236,0.1948717255

    //normalize with MONTPELLIER PLEAIDES 50 CM DATASET

    //aPL=(aPL.sub(0.3489)).div(0.25); //0.4357159999,0.1951853861 0.434583236,0.1948717255
    //aPL=(aPL.sub(0.45)).div(0.117);
    //aPL=(aPL.sub(aPL.mean())).div(aPL.std()+1e-8);
    torch::jit::IValue inp(aPL);
    std::vector<torch::jit::IValue> allinp={inp};
    //std::cout<<"IVALUE CREATED "<<std::endl; 
    auto out=mNet.forward(allinp);
    auto output=out.toTensor().squeeze();
    return output;  // not to KCPU because some calculation on similarity is to be perfomed 
}
/**********************************************************************************************************************/
/*********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictDecisionNet(torch::jit::script::Module mNet, torch::Tensor Left, torch::Tensor Right)
{
    auto cuda_available = torch::cuda::is_available();
    //std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    //Left=Left.to(device);
    //Right=Right.to(device);
    torch::NoGradGuard no_grad;
    mNet.eval();
    auto CatTensor=torch::cat({Left,Right},1); // to get a size of {1,FeatsSIZE}
    //CatTensor=CatTensor.squeeze(0).unsqueeze(3);
    torch::jit::IValue inp(CatTensor);
    std::vector<torch::jit::IValue> allinp={inp};
    torch::Tensor OutSim=mNet.forward(allinp).toTensor().squeeze();
    return torch::sigmoid(OutSim).to(torch::kCPU);
}

/*********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictONCUBE(torch::jit::script::Module mMlp,/*torch::jit::script::Module mMatcher,*/ torch::Tensor & aCube)
{
    //torch::Device device(torch::kCUDA);
    torch::NoGradGuard no_grad;
    mMlp.eval();
    torch::jit::IValue inp(aCube);
    std::vector<torch::jit::IValue> allinp={inp};
    torch::Tensor OutSimBrut=mMlp.forward(allinp).toTensor().sigmoid();
    // construct CONCAT CUBE AND SIMIL
    /*auto ConcatCube=torch::cat({aCube.unsqueeze(0),OutSimBrut.unsqueeze(0).unsqueeze(0)},1);
    std::cout<<"THE AGGREGATED CUBE OF DATA "<<ConcatCube.sizes()<<std::endl;
    // Second Forward
    //torch::jit::IValue inp2(OutSimBrut.unsqueeze(0).unsqueeze(0));
    torch::jit::IValue inp2(ConcatCube);
    //std::cout<<"Similarity  before match shape "<<OutSimBrut.sizes()<<std::endl;
    allinp.clear();
    allinp.push_back(inp2);
    mMatcher.eval();
    torch::Tensor OutSim=mMatcher.forward(allinp).toTensor().sigmoid().squeeze();
    allinp.clear();*/
    return OutSimBrut.to(torch::kCPU);
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNetAtt(MSNet_Attention mNet, std::vector<tTImV2> aPatchLV, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    torch::Tensor aPAllScales=torch::empty({4,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    //std::cout<<"SIZE OF MSCALE TILES "<<aPatchLV.size()<<std::endl;
    for (int cc=0;cc<(int) aPatchLV.size();cc++)
    {
        StdOut()<<"Size of tile Mul Scale is "<<aPatchLV.at(cc).DIm().Sz()<<"\n";
        tREAL4 ** mPatchLData=aPatchLV.at(cc).DIm().ExtractRawData2D();
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
        aPAllScales.index_put_({cc},aPL);
    }
    //downscale and upscale


    aPAllScales=aPAllScales.unsqueeze(0);
    // 4 scale tensor is needed for now test by passing the same tensor at each stage of the network 
    /*torch::Tensor a4ScaleTens=aPL.repeat_interleave(4,1);
    std::cout<<" a4ScaleTens size "<<a4ScaleTens.sizes()<<std::endl;
    assert
    (
      (a4ScaleTens.size(1)==4)  
    );*/

    auto output=mNet->forward(aPAllScales).squeeze();
    return output;
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNetHead(/*MSNetHead*/ torch::jit::script::Module mNet, std::vector<tTImV2> aPatchLV, cPt2di aPSz)
{
    auto cuda_available = torch::cuda::is_available();
    std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet.eval();
    torch::Tensor aPAllScales=torch::empty({(int) aPatchLV.size(),aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));;
    for (int cc=0;cc<(int) aPatchLV.size();cc++)
    {
        tREAL4 ** mPatchLData=aPatchLV.at(cc).DIm().ExtractRawData2D();
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
        aPAllScales.index_put_({cc},aPL);
    }
    // 4 scale tensor is needed for now test by passing the same tensor at each stage of the network 
    /*torch::Tensor a4ScaleTens=aPL.repeat_interleave(4,1);
    std::cout<<" a4ScaleTens size "<<a4ScaleTens.sizes()<<std::endl;
    assert
    (
      (a4ScaleTens.size(1)==4)  
    );*/
    aPAllScales=aPAllScales.unsqueeze(0).to(device);
    StdOut()<<"Patches "<<aPAllScales.sizes()<<"\n";
    torch::jit::IValue inp(aPAllScales);
    std::vector<torch::jit::IValue> allinp={inp};
    auto out=mNet.forward(allinp);
    auto output=out.toTensor().squeeze();
    return output.to(torch::kCPU);
}

};
/**********************************************************************************************************************/

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3  
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
