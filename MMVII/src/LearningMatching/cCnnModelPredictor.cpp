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
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include "cConvnet_Fast.h"
#include "cConvnet_Slow.h"
#include "cConvNet_Fast_BN.h"
#include "cCnnModelPredictor.h"

namespace F = torch::nn::functional;

namespace MMVII
{
/*********************************************************************/
template <typename T>  void Tensor2File(torch::Tensor a, std::string fname, std::string Type)
{
   //Store Tensor 
   T * TensorContent=a.data_ptr<T>();
   FILE *finaldestination = fopen(fname.c_str(), "wb");
   fwrite(TensorContent, sizeof(T), a.numel(), finaldestination);
   fclose(finaldestination);
   //Store dimensions 
   std::string dimFname=fname;
   dimFname.append(".dim");
   std::ofstream finaldestinationDim(dimFname.c_str()); // CHANGE LATER BY "cMMVII_Ofs"
   
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
/***********************************************************************/
/***********************************************************************/
torch::Tensor ReadBinaryFile(std::string filename, torch::Tensor Host)
{
  	int fd;
  	float *TensorContent;
  	fd = open(filename.c_str(), O_RDONLY);
  	TensorContent = static_cast<float*>(mmap(NULL, Host.numel() * sizeof(float), PROT_READ, MAP_SHARED, fd, 0));
	torch::Tensor Temp=torch::from_blob(TensorContent, Host.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
  	//showTensor<float>(Host,2,3);
  	close(fd);
  	return Temp;
}


/**************************************************************************/
aCnnModelPredictor::aCnnModelPredictor(std::string anArchitecture, std::string aModelBinDir):
    mArchitecture(anArchitecture)
{
    // FILL THE SET OF BINARY FILES NAMES 
    std::string aModelPat,aDirModel;
	SplitDirAndFile(aDirModel, aModelPat, aModelBinDir,false);
	cInterfChantierNameManipulateur * aICNMModel=cInterfChantierNameManipulateur::BasicAlloc(aDirModel);
	mSetModelBinaries = *(aICNMModel->Get(aModelPat));
    mDirModel=aDirModel;
}

/***************************************************************************/
torch::Tensor aCnnModelPredictor::ReadBinaryFile(std::string filename, torch::Tensor Host)
{
  	int fd;
  	float *TensorContent;
  	fd = open(filename.c_str(), O_RDONLY);
  	TensorContent = static_cast<float*>(mmap(NULL, Host.numel() * sizeof(float), PROT_READ, MAP_SHARED, fd, 0));
	torch::Tensor Temp=torch::from_blob(TensorContent, Host.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
  	//showTensor<float>(Host,2,3);
  	close(fd);
  	return Temp;
}
/***************************************************************************/
void aCnnModelPredictor::PopulateModelFromBinary(ConvNet_Fast Network)
{
	auto Fast=Network->getFastSequential();
	int Sz=mSetModelBinaries.size();
	int cnt=0;
	for (int i=0;i<(int)Fast->size();i++)
    { 
		std::sort (mSetModelBinaries.begin(), mSetModelBinaries.end()); // find a better way to serialize a model weights and biases 
		std::string LayerName=Fast->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0)
		{
			auto bias1=mDirModel+mSetModelBinaries.at(cnt);
			auto weight1=mDirModel+mSetModelBinaries.at(cnt+Sz/2); 
			std::cout<<"BIAS "<<bias1<<std::endl;
			std::cout<<"WGHT "<<weight1<<std::endl;
			torch::Tensor Weights=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
			Weights=ReadBinaryFile(weight1,Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data());
			//memcpy to copy content into conv2D weight 
			std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().data_ptr<float>(),Weights.data_ptr<float>(),sizeof(float)*Weights.numel());
			//std::cout<<"inner weight sizes "<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().sizes()<<std::endl;
			//showTensor<float>(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().slice(0,0,1,1).slice(1,0,1,1),2,3,"weights");
			//std::cout<<"WEIGHTSSSS "<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().slice(0,0,6,1).slice(1,0,1,1)<<std::endl;
			torch::Tensor Biases=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
			Biases=ReadBinaryFile(bias1,Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data());
			std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().data_ptr<float>(),Biases.data_ptr<float>(),sizeof(float)*Biases.numel());
			//std::cout<<"BIASSS  "<<Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data()<<std::endl;
			cnt++;
		}

	}
}

/*void PopulateModelFromFolder(MSNet Model, std::string FolderLocation)
{
     Load named tensors from folder location
}*/
/***********************************************************************/
void aCnnModelPredictor::PopulateModelFromBinaryWithBN(ConvNet_FastBn Network)
{
	
	auto Fast=Network->getFastSequential();
	int Sz=mSetModelBinaries.size();
	int cnt=0;
	std::sort (mSetModelBinaries.begin(), mSetModelBinaries.end()); // find a better way to serialize a model weights and biases 
	for (int i=0;i<(int)Fast->size();i++)
    { 
		std::string LayerName=Fast->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0 || LayerName.rfind(std::string("BatchNorm"),0)==0)
		{
			if (LayerName.rfind(std::string("conv"),0)==0)
			{
				//get relevant bias binary 
				int cc=0;
				bool found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("tensorBias_conv")+std::to_string(i),0)==0) 
					{
						found=true;
						
					}
					cc++;
				}
				auto bias1=mDirModel+mSetModelBinaries.at(cc-1);
				std::cout<<"BIAS "<<i<<" "<<bias1<<std::endl;
				cc=0;
				found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("tensorWeight_conv")+std::to_string(i),0)==0) 
					{
						found=true;
					}
					cc++;
				}
				auto weight1=mDirModel+mSetModelBinaries.at(cc-1); 
				std::cout<<"WEIGHTS "<<weight1<<std::endl;
				torch::Tensor Weights=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				Weights=ReadBinaryFile(weight1,Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data());
				std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().data_ptr<float>(),Weights.data_ptr<float>(),sizeof(float)*Weights.numel());
				torch::Tensor Biases=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				Biases=ReadBinaryFile(bias1,Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data());
				std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().data_ptr<float>(),Biases.data_ptr<float>(),sizeof(float)*Biases.numel());
				cnt++;
		    }
		    else
		    {
				//std::cout<<"SECOND CONDITION "<<std::endl;
				//get relevant bias binary 
				int cc=0;
				bool found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("tensorGamma_batchnorm")+std::to_string(i),0)==0) 
					{
						found=true;
						
					}
					cc++;
				}
				auto Gamma=mDirModel+mSetModelBinaries.at(cc-1);
				std::cout<<"Gamma "<<i<<" "<<Gamma<<std::endl;
				cc=0;
				found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("tensorBeta_batchnorm")+std::to_string(i),0)==0) 
					{
						found=true;
					}
					cc++;
				}
				auto Beta=mDirModel+mSetModelBinaries.at(cc-1); 
				std::cout<<"Beta "<<Beta<<std::endl;
                
                //RUNNING MEAN 
				cc=0;
				found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("tensorAverage_batchnorm")+std::to_string(i),0)==0) 
					{
						found=true;
					}
					cc++;
				}
				auto AVG=mDirModel+mSetModelBinaries.at(cc-1); 
				std::cout<<"RUNNING AVERAGE "<<AVG<<std::endl;
                
                // RUNNING VARIANCE 
				cc=0;
				found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("tensorVariance_batchnorm")+std::to_string(i),0)==0) 
					{
						found=true;
					}
					cc++;
				}
				auto VAR=mDirModel+mSetModelBinaries.at(cc-1); 
				std::cout<<"RUNNING VARIANCE "<<VAR<<std::endl;
                 
                // TENSOR ARE FOUND THEN FILL RELEVANT TENSOR DATA IN BATCH NORM LAYER 
                
				torch::Tensor Weights=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->weight.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				std::cout<<"NAMES "<<Weights.sizes()<<std::endl;
				Weights=ReadBinaryFile(Gamma,Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->weight.data());
				//memcpy to copy content into conv2D weight 
				//std::cout<<"READ BINARY 1"<<std::endl;
				std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->weight.data().data_ptr<float>(),Weights.data_ptr<float>(),sizeof(float)*Weights.numel());
				//std::cout<<"READ BINARY 2"<<std::endl;
				torch::Tensor Biases=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->bias.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				//std::cout<<"BIASES "<<Biases.sizes()<<std::endl;
				Biases=ReadBinaryFile(Beta,Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->bias.data());
				//std::cout<<"READ BINARY 3"<<std::endl;
				std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->bias.data().data_ptr<float>(),Biases.data_ptr<float>(),sizeof(float)*Biases.numel());
                
                // RUNNINH MEAN TENSOR 
				torch::Tensor Average=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->running_mean.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				//std::cout<<"BIASES "<<Biases.sizes()<<std::endl;
				Average=ReadBinaryFile(AVG,Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->running_mean.data());
				//std::cout<<"READ BINARY 3"<<std::endl;
				std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->running_mean.data().data_ptr<float>(),Average.data_ptr<float>(),sizeof(float)*Average.numel());
                
                
                //RUNNING VARIANCE TENSOR 
				torch::Tensor Variance=torch::empty(Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->running_var.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				//std::cout<<"BIASES "<<Biases.sizes()<<std::endl;
				Variance=ReadBinaryFile(VAR,Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->running_var.data());
				//std::cout<<"READ BINARY 3"<<std::endl;
				std::memcpy(Fast->named_children()[i].value().get()->as<torch::nn::BatchNorm2dImpl>()->running_var.data().data_ptr<float>(),Variance.data_ptr<float>(),sizeof(float)*Variance.numel());
                
				cnt++;
			}
		}
	}
}
/***********************************************************************/
void aCnnModelPredictor::PopulateModelFastandHead(FastandHead Network)
{
    StdOut()<<"TO LOAD MODEL "<<"\n";
    std::string aModel=mDirModel+mSetModelBinaries.at(0); // just one pickled model 
    StdOut()<<" Model NAME "<<aModel<<"\n";
    torch::load(Network,aModel);
    StdOut()<<"TORCH LOAD  "<<"\n";
}
/***********************************************************************/
void aCnnModelPredictor::PopulateModelFromBinaryWithBNReg(ConvNet_FastBnRegister Network)
{
    //StdOut()<<"TO LOAD MODEL "<<"\n";
    std::string aModel=mDirModel+mSetModelBinaries.at(0); // just one pickled model 
    //StdOut()<<" Model NAME "<<aModel<<"\n";
    torch::load(Network,aModel);
    //StdOut()<<"TORCH LOAD  "<<"\n";
}

/***********************************************************************/
void aCnnModelPredictor::PopulateModelPrjHead(Fast_ProjectionHead Network)
{
    //StdOut()<<"TO LOAD MODEL "<<"\n";
    std::string aModel=mDirModel+mSetModelBinaries.at(0); // just one pickled model 
    torch::load(Network,aModel);
    //StdOut()<<"TORCH LOAD  "<<"\n";
}
/***********************************************************************/
void aCnnModelPredictor::PopulateModelMSNet(MSNet Network)
{
    //StdOut()<<"TO LOAD MODEL "<<"\n";
    std::string aModel=mDirModel+mSetModelBinaries.at(0); // just one pickled model 
    torch::load(Network,aModel);
    //StdOut()<<"TORCH LOAD  "<<"\n";
}
/***********************************************************************/
void aCnnModelPredictor::PopulateModelMSNetAtt(MSNet_Attention Network)
{
    //StdOut()<<"TO LOAD MODEL "<<"\n";
    std::string aModel=mDirModel+mSetModelBinaries.at(0); // just one pickled model 
    torch::load(Network,aModel);
    //StdOut()<<"TORCH LOAD  "<<"\n";
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
    auto cuda_available = torch::cuda::is_available();
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    StdOut()<<"Model Name "<<aModel<<"\n";
    //auto cuda_available=false;
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    Network=torch::jit::load(aModel);
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
    auto cuda_available = torch::cuda::is_available();
    //auto cuda_available=false;
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
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
/***********************************************************************/
void aCnnModelPredictor::PopulateModelSimNet(SimilarityNet Network)
{
//     /StdOut()<<"TO LOAD MODEL "<<"\n";
    std::string aModel=mDirModel+mSetModelBinaries.at(0); // just one pickled model 
    //StdOut()<<" Model NAME "<<aModel<<"\n";
    torch::load(Network,aModel);
    //StdOut()<<"TORCH LOAD  "<<"\n";
}
/***********************************************************************/
void aCnnModelPredictor::PopulateSlowModelFromBinary(ConvNet_Slow Network)
{
	auto Slow=Network->getSlowSequential();
	int Sz=mSetModelBinaries.size();
	int cnt=0;
	std::sort (mSetModelBinaries.begin(), mSetModelBinaries.end()); // find a better way to serialize a model weights and biases 
	for (int i=0;i<(int)Slow->size();i++)
    { 
		std::string LayerName=Slow->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0 || LayerName.rfind(std::string("Linear"),0)==0)
		{
			if (LayerName.rfind(std::string("conv"),0)==0)
			{
				//get relevant bias binary 
				int cc=0;
				bool found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("Biases_")+std::to_string(i+1)+"_",0)==0) 
					{
						found=true;
						
					}
					cc++;
				}
				auto bias1=mDirModel+mSetModelBinaries.at(cc-1);
				std::cout<<"BIAS "<<i<<" "<<bias1<<std::endl;
				cc=0;
				found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("Weights_")+std::to_string(i+1)+"_",0)==0) 
					{
						found=true;
					}
					cc++;
				}
				auto weight1=mDirModel+mSetModelBinaries.at(cc-1); 
				std::cout<<"WEIGHTS "<<weight1<<std::endl;
				torch::Tensor Weights=torch::empty(Slow->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				Weights=ReadBinaryFile(weight1,Slow->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data());
				std::memcpy(Slow->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->weight.data().data_ptr<float>(),Weights.data_ptr<float>(),sizeof(float)*Weights.numel());
				torch::Tensor Biases=torch::empty(Slow->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				Biases=ReadBinaryFile(bias1,Slow->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data());
				std::memcpy(Slow->named_children()[i].value().get()->as<torch::nn::Conv2dImpl>()->bias.data().data_ptr<float>(),Biases.data_ptr<float>(),sizeof(float)*Biases.numel());
				cnt++;
		    }
		    else
		    {
				std::cout<<"SECOND CONDITION "<<std::endl;
				//get relevant bias binary 
				int cc=0;
				bool found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("Biases_")+std::to_string(i)+"_",0)==0) 
					{
						found=true;
						
					}
					cc++;
				}
				auto bias1=mDirModel+mSetModelBinaries.at(cc-1);
				std::cout<<"BIAS "<<i<<" "<<bias1<<std::endl;
				cc=0;
				found=false;
				while(cc<Sz && !found)
				{
					if (mSetModelBinaries.at(cc).rfind(std::string("Weights_")+std::to_string(i)+"_",0)==0) 
					{
						found=true;
					}
					cc++;
				}
				auto weight1=mDirModel+mSetModelBinaries.at(cc-1); 
				std::cout<<"WEIGHTS "<<weight1<<std::endl;
				torch::Tensor Weights=torch::empty(Slow->named_children()[i].value().get()->as<torch::nn::LinearImpl>()->weight.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				std::cout<<"NAMES "<<Weights.sizes()<<std::endl;
				Weights=ReadBinaryFile(weight1,Slow->named_children()[i].value().get()->as<torch::nn::LinearImpl>()->weight.data());
				//memcpy to copy content into conv2D weight 
				//std::cout<<"READ BINARY 1"<<std::endl;
				std::memcpy(Slow->named_children()[i].value().get()->as<torch::nn::LinearImpl>()->weight.data().data_ptr<float>(),Weights.data_ptr<float>(),sizeof(float)*Weights.numel());
				//std::cout<<"READ BINARY 2"<<std::endl;
				torch::Tensor Biases=torch::empty(Slow->named_children()[i].value().get()->as<torch::nn::LinearImpl>()->bias.data().sizes(),torch::TensorOptions().dtype(torch::kFloat32));
				//std::cout<<"BIASES "<<Biases.sizes()<<std::endl;
				Biases=ReadBinaryFile(bias1,Slow->named_children()[i].value().get()->as<torch::nn::LinearImpl>()->bias.data());
				//std::cout<<"READ BINARY 3"<<std::endl;
				std::memcpy(Slow->named_children()[i].value().get()->as<torch::nn::LinearImpl>()->bias.data().data_ptr<float>(),Biases.data_ptr<float>(),sizeof(float)*Biases.numel());
				cnt++;
			}
		}
	}
}

/***********************************************************************/

cPt2di aCnnModelPredictor::GetWindowSize(ConvNet_Fast Network)
{ 
	int ws=1;
	auto Fast=Network->getFastSequential();
	for (int i=0;i<(int)Fast->size();i++)
    { 
     	std::string LayerName=Fast->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0)
		{
		ws=ws+Network->getKernelSize()-1;
	    }
	}
	cPt2di aPt(round_up(ws/2),round_up(ws/2));
	return aPt;
}

/***********************************************************************/
cPt2di aCnnModelPredictor::GetWindowSizeBN(ConvNet_FastBn  Network)
{ 
	int ws=1;
	auto Fast=Network->getFastSequential();
	for (int i=0;i<(int)Fast->size();i++)
    { 
     	std::string LayerName=Fast->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0)
		{
		ws=ws+Network->getKernelSize()-1;
	    }
	}
	cPt2di aPt(round_up(ws/2),round_up(ws/2));
	return aPt;
}
/***********************************************************************/
cPt2di aCnnModelPredictor::GetWindowSizeBNReg(ConvNet_FastBnRegister  Network)
{ 
	int ws=1;
	auto Fast=Network->mFast;
	for (int i=0;i<(int)Fast->size();i++)
    { 
     	std::string LayerName=Fast->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0)
		{
		ws=ws+Network->getKernelSize()-1;
	    }
	}
	cPt2di aPt(round_up(ws/2),round_up(ws/2));
	return aPt;
}

/***********************************************************************/
cPt2di aCnnModelPredictor::GetWindowSizePrjHead(Fast_ProjectionHead  Network)
{ 
	int ws=1;
	auto Fast=Network->mFast;
	for (int i=0;i<(int)Fast->size();i++)
    { 
     	std::string LayerName=Fast->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0)
		{
		ws=ws+Network->getKernelSize()-1;
	    }
	}
	cPt2di aPt(round_up(ws/2),round_up(ws/2));
	return aPt;
}

/***********************************************************************/
cPt2di aCnnModelPredictor::GetWindowSizeFastandHead(FastandHead  Network)
{ 
	int ws=1;
	auto Fast=Network->mFast;
	for (int i=0;i<(int)Fast->size();i++)
    { 
     	std::string LayerName=Fast->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0)
		{
		ws=ws+Network->getKernelSize()-1;
	    }
	}
	cPt2di aPt(round_up(ws/2),round_up(ws/2));
	return aPt;
}


/***********************************************************************/
cPt2di aCnnModelPredictor::GetWindowSizeSimNet(SimilarityNet Network)
{ 
	int ws=1;
	auto Fast=Network->mFast;
	for (int i=0;i<(int)Fast->size();i++)
    { 
     	std::string LayerName=Fast->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0)
		{
		ws=ws+Network->getKernelSize()-1;
	    }
	}
	cPt2di aPt(round_up(ws/2),round_up(ws/2));
	return aPt;
}
/***********************************************************************/

cPt2di aCnnModelPredictor::GetSlowWindowSize(ConvNet_Slow  Network)
{ 
	int ws=1;
	auto Slow=Network->getSlowSequential();
	for (int i=0;i<(int)Slow->size();i++)
    { 
     	std::string LayerName=Slow->named_children()[i].key();
		if (LayerName.rfind(std::string("conv"),0)==0)
		{
		ws=ws+Network->getKernelSize()-1;
	    }
	}
	cPt2di aPt(round_up(ws/2),round_up(ws/2));
	return aPt;
}
/*********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictTile(ConvNet_Fast mNet, tTImV2 aPatchL, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
    
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	
	/*X_batch.index_put_({0},aPL);
	X_batch.index_put_({1},aPR);*/
    /*******************************************************************/
    auto output=mNet->forward_but_Last(aPL).squeeze();
    
    // OUTPUT IS OF SIZE {2,FeatureSIZE,1,1} //NO PADDING USED OTHERWISE Will be {2,FeatureSIZE,WinSIZE,WINSIZE}
    
    // PERFORM COSINE SIMILARITY BETWEEN BOTH INSTANCES 
    //auto aSim=F::cosine_similarity(output.slice(0,0,1,1), output.slice(0,1,2,1), F::CosineSimilarityFuncOptions().dim(1)).squeeze();
    
    return output;
}
/*********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictFastWithHead(FastandHead mNet, tTImV2 aPatchL, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
    
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	
	/*X_batch.index_put_({0},aPL);
	X_batch.index_put_({1},aPR);*/
    /*******************************************************************/
    auto output=mNet->forwardInfer(aPL).squeeze();
    
    // REPRODUCE THE SAME PATCH FOR ENTRY INTO  MLP 
    /*std::cout<<"OUPUT OF CONV SIZE "<<output.sizes()<<std::endl;
    output=output.repeat({1,9,1,1}).squeeze();
    output=at::transpose(at::transpose(output,0,1),1,2).contiguous();
    std::cout<<"OUPUT OF CONV SIZE "<<output.sizes()<<std::endl;
    // FORWARD THROUGH MLP 
    output=mNet->forwardMLP(output).squeeze();
    output=at::transpose(at::transpose(output,2,1),1,0).contiguous();
    // NORMALIZE VECTORS 
    
    output=torch::nn::functional::normalize(output, F::NormalizeFuncOptions().p(2).dim(0).eps(1e-8));
    std::cout<<"NEW SIZE "<<output.sizes()<<std::endl;*/
    // OUTPUT IS OF SIZE {2,FeatureSIZE,1,1} //NO PADDING USED OTHERWISE Will be {2,FeatureSIZE,WinSIZE,WINSIZE}
    
    // PERFORM COSINE SIMILARITY BETWEEN BOTH INSTANCES 
    //auto aSim=F::cosine_similarity(output.slice(0,0,1,1), output.slice(0,1,2,1), F::CosineSimilarityFuncOptions().dim(1)).squeeze();
    
    return output;
}
/*********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictSimNetConv(SimilarityNet mNet, tTImV2 aPatchL, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
    
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    auto output=mNet->forwardConv(aPL);
    return output;
}
/*********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictSimNetMLP(SimilarityNet mNet, torch::Tensor Left, torch::Tensor Right)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    // TENSOR SHOUD BE OF SHAPE {1,FeatsSIZE,N,N} WHERE N IS the size of TILE BUT FOR NOW {1,FeatsSIZE,1,1} for a single embedding vector *
    auto CatTensor=torch::cat({Left,Right},1); // to get a size of {1,FeatsSIZE}
    torch::Tensor OutSim=mNet->forwardMLP(CatTensor).squeeze();
    return OutSim;
}
/*************************************************************************************************************************/
double aCnnModelPredictor::Predict(ConvNet_Fast mNet, tTImV2 aPatchL, tTImV2 aPatchR, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::Tensor X_batch=torch::empty({2,1,aPSz.y(),aPSz.x()},torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::NoGradGuard no_grad;
	mNet->eval();
    
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
	tREAL4 ** mPatchRData=aPatchR.DIm().ExtractRawData2D();
    
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor aPR=torch::from_blob((*mPatchRData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	
	X_batch.index_put_({0},aPL);
	X_batch.index_put_({1},aPR);
    /*******************************************************************/
    auto output=mNet->forward_but_Last(X_batch);
    
    // OUTPUT IS OF SIZE {2,FeatureSIZE,1,1} //NO PADDING USED OTHERWISE Will be {2,FeatureSIZE,WinSIZE,WINSIZE}
    
    // PERFORM COSINE SIMILARITY BETWEEN BOTH INSTANCES 
    auto aSim=F::cosine_similarity(output.slice(0,0,1,1), output.slice(0,1,2,1), F::CosineSimilarityFuncOptions().dim(1)).squeeze();
    
    return (double) aSim.item<float>();
}


/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictWithBNTile(ConvNet_FastBn mNet, tTImV2 aPatchL, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    
    //StdOut() << "load data RAW " << "\n";
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
	//tREAL4 ** mPatchRData=aPatchR.DIm().ExtractRawData2D();
    //StdOut() << "loaded data RAW " << "\n";
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	//torch::Tensor aPR=torch::from_blob((*mPatchRData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    //StdOut() << "TENSOR CREATED  " << "\n";
	/*X_batch.index_put_({0},aPL);
	X_batch.index_put_({1},aPR);*/
    //StdOut() << "loaded data RAW " << "\n";
    /*******************************************************************/
    auto output=mNet->forward(aPL).squeeze();
    //StdOut () <<output<<"\n";
    
    // OUTPUT IS OF SIZE {2,FeatureSIZE,1,1} //NO PADDING USED OTHERWISE Will be {2,FeatureSIZE,WinSIZE,WINSIZE}
    
    // PERFORM COSINE SIMILARITY BETWEEN BOTH INSTANCES 
    //auto aSim=F::cosine_similarity(output.slice(0,0,1,1), output.slice(0,1,2,1), F::CosineSimilarityFuncOptions().dim(1)).squeeze();
    return output;
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictPrjHead(Fast_ProjectionHead mNet, tTImV2 aPatchL, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    auto output=mNet->forwardConv(aPL).squeeze();
    return output;
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
    torch::Tensor aPAllMasters=torch::empty({(int) aMasterP.size(),aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor aPAllSlaves=torch::empty({(int) aPatchLV.size(),aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    for (int cc=0;cc<(int) aMasterP.size();cc++)
    {
        tREAL4 ** mPatchLData=aMasterP.at(cc).DIm().ExtractRawData2D();
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
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
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
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
    //auto cuda_available = torch::cuda::is_available();
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    auto cuda_available=false;
    //<$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$>
    //std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard no_grad;
    mNet.eval();
    torch::Tensor aPAllSlaves=torch::empty({(int) aPatchLV.size(),aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    for (int cc=0;cc<(int) aPatchLV.size();cc++)
    {
        tREAL4 ** mPatchLData=aPatchLV.at(cc).DIm().ExtractRawData2D();
        torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
        aPL=aPL.div(255.0);
        aPL=(aPL.sub(0.4353755468)).div(0.19367880);
        aPAllSlaves.index_put_({cc},aPL);
    }
    torch::jit::IValue inp(aPAllSlaves.unsqueeze(0));
    std::vector<torch::jit::IValue> allinp={inp};
    auto out=mNet.forward(allinp);
    auto output=out.toTensor().squeeze();
    return output.to(torch::kCPU);
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
    auto cuda_available = torch::cuda::is_available();
    //auto cuda_available = false;
    //std::cout<<"Cuda is available ? "<<cuda_available<<std::endl;
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard no_grad;
    mNet.eval();
    tREAL4 ** mPatchLData=aPatchLV.DIm().ExtractRawData2D();
    torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    // Normalize The tile with respect to the dataset configuration
    // print image content
    //std::cout<<"TILE CONTENT  ========= >  "<<aPL<<std::endl;
    aPL=(aPL.div(255.0));
    aPL=(aPL.sub(0.4357159999)).div(0.1951853861); //0.4357159999,0.1951853861
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
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNet1(MSNet mNet,torch::Tensor X)
{
	torch::NoGradGuard no_grad;
	mNet->eval();
    auto output=mNet->forwardRes1(X).squeeze();
    return output;
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNet2(MSNet mNet,torch::Tensor X)
{
	torch::NoGradGuard no_grad;
	mNet->eval();
    auto output=mNet->forwardRes2(X).squeeze();
    return output;
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNet3(MSNet mNet,torch::Tensor X)
{
	torch::NoGradGuard no_grad;
	mNet->eval();
    auto output=mNet->forwardRes3(X).squeeze();
    return output;
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNet4(MSNet mNet,torch::Tensor X)
{
	torch::NoGradGuard no_grad;
	mNet->eval();
    auto output=mNet->forwardRes4(X).squeeze();
    return output;
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictMSNetCommon(MSNet mNet, tTImV2 aPatchL, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    // 4 scale tensor is needed for now test by passing the same tensor at each stage of the network 
    /*torch::Tensor a4ScaleTens=aPL.repeat_interleave(4,1);
    std::cout<<" a4ScaleTens size "<<a4ScaleTens.sizes()<<std::endl;
    assert
    (
      (a4ScaleTens.size(1)==4)  
    );*/
    auto output=mNet->forwardConvCommon(aPL);
    return output;
}
/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictWithBNTileReg(ConvNet_FastBnRegister mNet, tTImV2 aPatchL, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::NoGradGuard no_grad;
	mNet->eval();
    
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    
    //StdOut() << "load data RAW " << "\n";
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
	//tREAL4 ** mPatchRData=aPatchR.DIm().ExtractRawData2D();
    //StdOut() << "loaded data RAW " << "\n";
    
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	//torch::Tensor aPR=torch::from_blob((*mPatchRData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    //StdOut() << "TENSOR CREATED  " << "\n";
	/*X_batch.index_put_({0},aPL);
	X_batch.index_put_({1},aPR);*/
    //StdOut() << "loaded data RAW " << "\n";
    /*******************************************************************/
    auto output=mNet->forward(aPL).squeeze();
    //StdOut () <<output<<"\n";
    
    // OUTPUT IS OF SIZE {2,FeatureSIZE,1,1} //NO PADDING USED OTHERWISE Will be {2,FeatureSIZE,WinSIZE,WINSIZE}
    
    // PERFORM COSINE SIMILARITY BETWEEN BOTH INSTANCES 
    //auto aSim=F::cosine_similarity(output.slice(0,0,1,1), output.slice(0,1,2,1), F::CosineSimilarityFuncOptions().dim(1)).squeeze();
    return output;
}
/**********************************************************************************************************************/
double aCnnModelPredictor::PredictWithBN(ConvNet_FastBn mNet, tTImV2 aPatchL, tTImV2 aPatchR, cPt2di aPSz)
{
	torch::Device device(torch::kCPU);
	torch::Tensor X_batch=torch::empty({2,1,aPSz.y(),aPSz.x()},torch::TensorOptions().dtype(torch::kFloat32).device(device));
	torch::NoGradGuard no_grad;
	mNet->eval();
    
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    
    //StdOut() << "load data RAW " << "\n";
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
	tREAL4 ** mPatchRData=aPatchR.DIm().ExtractRawData2D();
    //StdOut() << "loaded data RAW " << "\n";
    
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor aPR=torch::from_blob((*mPatchRData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
    //StdOut() << "TENSOR CREATED  " << "\n";
	X_batch.index_put_({0},aPL);
	X_batch.index_put_({1},aPR);
    //StdOut() << "loaded data RAW " << "\n";
    /*******************************************************************/
    auto output=mNet->forward(X_batch);
    //StdOut () <<output<<"\n";
    
    // OUTPUT IS OF SIZE {2,FeatureSIZE,1,1} //NO PADDING USED OTHERWISE Will be {2,FeatureSIZE,WinSIZE,WINSIZE}
    
    // PERFORM COSINE SIMILARITY BETWEEN BOTH INSTANCES 
    auto aSim=F::cosine_similarity(output.slice(0,0,1,1), output.slice(0,1,2,1), F::CosineSimilarityFuncOptions().dim(1)).squeeze();
    
    return (double) aSim.item<float>();
}



/**********************************************************************************************************************/
torch::Tensor aCnnModelPredictor::PredictSlowTile(ConvNet_Slow mNet, tTImV2 aPatchL, tTImV2 aPatchR, cPt2di aPSz)
{
    
    // to be checked later =======================================++++++><+++++++
	torch::Device deviceCUDA(torch::kCUDA);
	torch::Device devicecpu(torch::kCPU);
	torch::Tensor X_batch=torch::empty({2,1,aPSz.y(),aPSz.x()},torch::TensorOptions().dtype(torch::kFloat32).device(devicecpu));
	torch::NoGradGuard no_grad;
	mNet->eval();
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
	tREAL4 ** mPatchRData=aPatchR.DIm().ExtractRawData2D();
    
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor aPR=torch::from_blob((*mPatchRData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	X_batch.index_put_({0},aPL);
	X_batch.index_put_({1},aPR);
	// CREATE A COST VOLUME FOR THE SLOW CONFIGURATION CNN+MLP
	auto outputConv=mNet->ForwardConv(X_batch);  // size output conv ==> {2,nbfeats,1,1}
    auto l = outputConv.slice(0,0,1,1); // Left at certain disparities 
    auto r = outputConv.slice(0,1,2,1); // Right at certain
    torch::Tensor InMLP=torch::cat({l,r},1);
    auto aSim=mNet->ForwardMLP(InMLP).squeeze();
    return aSim;
    
    // WORK TO DO ON TENSORS TO PRODUCE N D ARRAY THAT COMPUTES COSINE METRIC BETWEEN VECTORS 
}
/**********************************************************************************************************************/
double aCnnModelPredictor::PredictSlow(ConvNet_Slow mNet, tTImV2 aPatchL, tTImV2 aPatchR, cPt2di aPSz)
{
    
    // to be checked later =======================================++++++><+++++++
	torch::Device deviceCUDA(torch::kCUDA);
	torch::Device devicecpu(torch::kCPU);
	torch::Tensor X_batch=torch::empty({2,1,aPSz.y(),aPSz.x()},torch::TensorOptions().dtype(torch::kFloat32).device(devicecpu));
	torch::NoGradGuard no_grad;
	mNet->eval();
    // TENSOR FROM PATCHES AND FILL BOTH IN ONE
    tREAL4 ** mPatchLData=aPatchL.DIm().ExtractRawData2D();
	tREAL4 ** mPatchRData=aPatchR.DIm().ExtractRawData2D();
    
	torch::Tensor aPL=torch::from_blob((*mPatchLData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor aPR=torch::from_blob((*mPatchRData), {1,1,aPSz.y(),aPSz.x()}, torch::TensorOptions().dtype(torch::kFloat32));
	X_batch.index_put_({0},aPL);
	X_batch.index_put_({1},aPR);
	// CREATE A COST VOLUME FOR THE SLOW CONFIGURATION CNN+MLP
	auto outputConv=mNet->ForwardConv(X_batch);  // size output conv ==> {2,nbfeats,1,1}
    auto l = outputConv.slice(0,0,1,1); // Left at certain disparities 
    auto r = outputConv.slice(0,1,2,1); // Right at certain
    torch::Tensor InMLP=torch::cat({l,r},1);
    auto aSim=mNet->ForwardMLP(InMLP).squeeze();
    return (double) aSim.item<float>();
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
