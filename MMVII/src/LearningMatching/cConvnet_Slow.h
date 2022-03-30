#pragma once
#include <torch/torch.h>
namespace F = torch::nn::functional;

/**********************************************************************/
/**********************************************************************/
class ReshapeImpl:public torch::nn::Module{	
	
public:
      ReshapeImpl(int64_t BS,int64_t NbDim):mBS(BS),mNbDim(NbDim)
      {};
      int64_t mBS;
      int64_t mNbDim;
      torch::Tensor forward (torch::Tensor input)
      {
          return input.reshape({mBS,mNbDim});  // Reshape the layer to output a vector 
	   };
};
TORCH_MODULE(Reshape);
/**********************************************************************/

/**********************************************************************/
class ConvNet_SlowImpl : public torch::nn::Module {
public:
      ConvNet_SlowImpl(int64_t kern,int64_t hidden1, int64_t hidden2):mkernel(kern),mNbHiddenLayers(hidden1),mNbHiddenLayers2(hidden2)
       { };
/**********************************************************************/
    void createModel(int64_t mfeatureMaps, int64_t mn_input_plane,int64_t mks,int64_t mfeatureMaps2,int64_t BS)
    {
        for (auto i=0; i<mNbHiddenLayers;i++)
        {
            if (i==0) // Initial image: it will take the number of channels of the patch$
            {
    		  mSlow->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
              //mSlow->push_back(std::string("BathNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps)));
              mSlow->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    		}
    		else 
    		{
    		mSlow->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
            //mSlow->push_back(std::string("BathNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps)));
            mSlow->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    	    }
    	}
    	mSlow->push_back(std::string("Reshape"),Reshape(BS,2*mfeatureMaps)); //Reshape
    	
    	// Add Fully Connected Layers 
    	 for (auto i=0; i<mNbHiddenLayers2;i++)
    	{
			if (i==0)
			{
			mSlow->push_back(std::string("Linear")+std::to_string(mNbHiddenLayers+i), torch::nn::Linear(torch::nn::LinearOptions(2*mfeatureMaps,mfeatureMaps2).bias(true)));
			mSlow->push_back(std::string("ReLU")+std::to_string(mNbHiddenLayers+i),torch::nn::ReLU());
			}
			else
			{
			mSlow->push_back(std::string("Linear")+std::to_string(mNbHiddenLayers+i), torch::nn::Linear(torch::nn::LinearOptions(mfeatureMaps2, mfeatureMaps2).bias(true)));
			mSlow->push_back(std::string("ReLU")+std::to_string(mNbHiddenLayers+i),torch::nn::ReLU());
			}
		}
		mSlow->push_back(std::string("Linear")+std::to_string(mNbHiddenLayers+mNbHiddenLayers2), torch::nn::Linear(torch::nn::LinearOptions(mfeatureMaps2, 1).bias(true)));
		mSlow->push_back(std::string("Sigmoid"), torch::nn::Sigmoid());
    }
/**********************************************************************/

    torch::Tensor forward(torch::Tensor x)   
    {
    	auto& model_ref = *mSlow;
        for (auto module : model_ref)
        {
    		x=module.forward(x);
    	}
    	return x;
    }

/***********************************************************************/
    torch::Tensor ForwardConv(torch::Tensor x)   
    {
    
        int64_t cc=0;
    	auto& model_ref = *mSlow;
        for (auto module : model_ref)
        {
    		if (cc<2 * mNbHiddenLayers)
    		{
				//std::cout<<"Layers "<<model_ref.named_children()[cc].key()<<std::endl;
				x=module.forward(x);
			}
    		cc++;
    	}
    	return x;
    }
/***********************************************************************/
    torch::Tensor ForwardMLP(torch::Tensor x)   
    {
        int64_t cc=0; // account for the reshape layer that is added
    	auto& model_ref = *mSlow;
        for (auto module : model_ref)
        {
    		if (cc>2*mNbHiddenLayers)
    		{
				//std::cout<<"Layers "<<model_ref.named_children()[cc].key()<<std::endl;
				x=module.forward(x);
			}
    		cc++;
    	}
    	return x;
    }
    
/***********************************************************************/
    torch::nn::Sequential getSlowSequential()
    {
		return this->mSlow;
	}
/***********************************************************************/
	int64_t getKernelSize()
	{ return this->mkernel;}
/***********************************************************************/
   int64_t mkernel;
   int64_t mNbHiddenLayers;
   int64_t mNbHiddenLayers2;
private:
   torch::nn::Sequential mSlow;
};

TORCH_MODULE(ConvNet_Slow);





