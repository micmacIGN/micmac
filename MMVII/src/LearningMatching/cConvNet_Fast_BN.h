#pragma once
#include <torch/torch.h>
namespace F = torch::nn::functional;




/**********************************************************************/
class NormL22Impl:public torch::nn::Module{
	
public:
      NormL22Impl()
      {};
      torch::Tensor forward (torch::Tensor input)
      {
		  //torch::Tensor out=F::normalize(input, F::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
		  //std::cout<<"effect of normalization "<<out.index({0})<<std::endl;
          return F::normalize(input, F::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
	   };
};
TORCH_MODULE(NormL22);
/**********************************************************************/
/**********************************************************************/
class ConvNet_FastBnImpl : public torch::nn::Module {
public:
      ConvNet_FastBnImpl(int64_t kern, int64_t nbHidden):mkernel(kern),mnbHidden(nbHidden)
       { };
/**********************************************************************/
    void createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane,int64_t mks, torch::Device device)
    {
        for (auto i=0; i<mNbHiddenLayers-1;i++)
        {
            if (i==0) // Initial image: it will take the number of channels of the patch$
            {
    		  mFast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
              mFast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
              mFast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    		}
    		else 
    		{
    		mFast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
            mFast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
            mFast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    	    }
    	}
    	mFast->push_back(std::string("conv")+std::to_string(mNbHiddenLayers-1),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
    	
        mFast->push_back("NormL22",NormL22());
        //mFast->push_back("StereoJoin",StereoJoin1());
    }
/**********************************************************************/
    torch::Tensor forward(torch::Tensor x)   
    {
    	auto& model_ref = *mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }
    

/***********************************************************************/    
        torch::Tensor forwardTo4(torch::Tensor x)   
    {
        //int counto4=0;
        bool EnoughForward=false;
        size_t cc=0;
        auto& model_ref = *mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
            std::string LayerName=mFast->named_children()[cc].key();
            if (LayerName.rfind(std::string("conv"),0)==0 && (!EnoughForward))
            {
                x=module.forward(x);
            }
            
            if (LayerName.rfind(std::string("BatchNorm"),0)==0 && (!EnoughForward))
            {
                x=module.forward(x);
            }
            
            if (LayerName.rfind(std::string("ReLU"),0)==0 && (!EnoughForward))
            {
                x=module.forward(x);
            }
            if (cc==15)
            {
               EnoughForward=true;                
            }
            // DO THE LAST NORM L2 
            std::string NormLayer=mFast->named_children()[cc].key();
            if (LayerName.rfind(std::string("NormL22"),0)==0)
            {
              x=module.forward(x);  
            }
            cc++;
    	}
    	return x;
    }

    /***********************************************************************/    
        torch::Tensor forwardWithoutBN(torch::Tensor x)   
    {
        bool EnoughForward=false;
        size_t cc=0;
        auto& model_ref = *mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
            std::string LayerName=mFast->named_children()[cc].key();
            if (LayerName.rfind(std::string("conv"),0)==0)
            {
                x=module.forward(x);
            }
            if (LayerName.rfind(std::string("ReLU"),0)==0 && (!EnoughForward))
            {
                x=module.forward(x);
            }
            if (LayerName.rfind(std::string("NormL22"),0)==0)
            {
              x=module.forward(x);  
            }
            cc++;
    	}
    	return x;
    }

/***********************************************************************/
    torch::Tensor forward_but_Last(torch::Tensor x)   
    {
    
        size_t Sz=this->getFastSequential()->size();
        size_t cc=0;
    	auto& model_ref = *mFast;
        for (auto module : model_ref)
        {
    		if (cc<Sz-1)
    		{x=module.forward(x);
			}
    		cc++;
    	}
    	return x;
    }
/***********************************************************************/
    torch::nn::Sequential getFastSequential()
    {
		return this->mFast;
	}
/***********************************************************************/
	int64_t getKernelSize()
	{ return this->mkernel;}
/***********************************************************************/
 private:
   int64_t mkernel;
   int64_t mnbHidden;
   torch::nn::Sequential mFast;
};

TORCH_MODULE(ConvNet_FastBn);



/**********************************************************************/
class ConvNet_FastBnRegisterImpl : public torch::nn::Module {
public:
      ConvNet_FastBnRegisterImpl(int64_t kern, int64_t nbHidden, int64_t input_plane,int64_t mfeatureMaps, torch::Device device):mn_input_plane(input_plane), mkernel(kern),mnbHidden(nbHidden), mFeats(mfeatureMaps)
       { 
         mFast=torch::nn::Sequential(createModel(mFeats,mnbHidden,mn_input_plane ,mkernel,device));
         register_module("mFast",mFast);
       };
/**********************************************************************/
    torch::nn::Sequential createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane,int64_t mks, torch::Device device)
    {
        torch::nn::Sequential Fast;
        for (auto i=0; i<mNbHiddenLayers-1;i++)
        {
            if (i==0) // Initial image: it will take the number of channels of the patch$
            {
    		  Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
              Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
              Fast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    		}
    		else 
    		{
    		Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
            Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
            Fast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    	    }
    	}
    	Fast->push_back(std::string("conv")+std::to_string(mNbHiddenLayers-1),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
    	
        Fast->push_back("NormL2",NormL2());
        return Fast;
    }
/**********************************************************************/

    torch::Tensor forward(torch::Tensor x)   
    {
    	auto& model_ref = * mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }

/***********************************************************************/
/***********************************************************************/
	int64_t getKernelSize()
	{ return this->mkernel;}
/***********************************************************************/    
	int64_t getFeats()
	{ return this->mFeats;}
/***********************************************************************/
   int64_t mn_input_plane;
   int64_t mkernel;
   int64_t mnbHidden;
   int64_t mFeats;
   torch::nn::Sequential mFast{ nullptr };
};

TORCH_MODULE(ConvNet_FastBnRegister);


/**********************************************************************/
class ConvNet_FastBnRegisterLReLUImpl : public torch::nn::Module {
public:
      ConvNet_FastBnRegisterLReLUImpl(int64_t kern, int64_t nbHidden, int64_t input_plane,int64_t mfeatureMaps, torch::Device device):mn_input_plane(input_plane), mkernel(kern),mnbHidden(nbHidden), mFeats(mfeatureMaps)
       { 
         mFast=torch::nn::Sequential(createModel(mFeats,mnbHidden,mn_input_plane ,mkernel,device));
         register_module("mFast",mFast);
       };
/**********************************************************************/
    torch::nn::Sequential createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane,int64_t mks, torch::Device device)
    {
        torch::nn::Sequential Fast;
        for (auto i=0; i<mNbHiddenLayers-1;i++)
        {
            if (i==0) // Initial image: it will take the number of channels of the patch$
            {
    		  Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
              Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
              Fast->push_back(std::string("LReLU")+std::to_string(i),torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
    		}
    		else 
    		{
    		Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
            Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
            Fast->push_back(std::string("LReLU")+std::to_string(i),torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
    	    }
    	}
    	Fast->push_back(std::string("conv")+std::to_string(mNbHiddenLayers-1),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
    	
        Fast->push_back("NormL2",NormL2());
        return Fast;
    }
/**********************************************************************/

    torch::Tensor forward(torch::Tensor x)   
    {
    	auto& model_ref = * mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }

/***********************************************************************/
/***********************************************************************/
	int64_t getKernelSize()
	{ return this->mkernel;}
/***********************************************************************/    
	int64_t getFeats()
	{ return this->mFeats;}
/***********************************************************************/
   int64_t mn_input_plane;
   int64_t mkernel;
   int64_t mnbHidden;
   int64_t mFeats;
   torch::nn::Sequential mFast{ nullptr };
};

TORCH_MODULE(ConvNet_FastBnRegisterLReLU);

/**********************************************************************/
class ConvNet_Fast_LReLUBNImpl : public torch::nn::Module {
public:
      ConvNet_Fast_LReLUBNImpl(int64_t kern, int64_t nbHidden, int64_t input_plane,int64_t mfeatureMaps, torch::Device device):mn_input_plane(input_plane), mkernel(kern),mnbHidden(nbHidden), mFeats(mfeatureMaps)
       { 
         mFast=torch::nn::Sequential(createModel(mFeats,mnbHidden,mn_input_plane ,mkernel,device));
         register_module("mFast",mFast);
       };
/**********************************************************************/
    torch::nn::Sequential createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane,int64_t mks, torch::Device device)
    {
        torch::nn::Sequential Fast;
        for (auto i=0; i<mNbHiddenLayers-1;i++)
        {
            if (i==0) // Initial image: it will take the number of channels of the patch$
            {
    		  Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
              Fast->push_back(std::string("LReLU")+std::to_string(i),torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
              Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));

    		}
    		else 
    		{
    		Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
            Fast->push_back(std::string("LReLU")+std::to_string(i),torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
            Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
    	    }
    	}
    	Fast->push_back(std::string("conv")+std::to_string(mNbHiddenLayers-1),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
    	
        Fast->push_back("NormL2",NormL2());
        return Fast;
    }
/**********************************************************************/

    torch::Tensor forward(torch::Tensor x)   
    {
    	auto& model_ref = * mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }

/***********************************************************************/
/***********************************************************************/
	int64_t getKernelSize()
	{ return this->mkernel;}
/***********************************************************************/    
	int64_t getFeats()
	{ return this->mFeats;}
/***********************************************************************/
   int64_t mn_input_plane;
   int64_t mkernel;
   int64_t mnbHidden;
   int64_t mFeats;
   torch::nn::Sequential mFast{ nullptr };
};

TORCH_MODULE(ConvNet_Fast_LReLUBN);

/************************************************************************************************************/
/**********************************************************************/

class FastandHeadImpl : public torch::nn::Module {
public:
      FastandHeadImpl(int64_t kern, int64_t nbHidden,int64_t nbHidden2, int64_t input_plane,int64_t mfeatureMaps,int64_t mfeatMaps2,int64_t smpl,int64_t bsize,
                      torch::Device device):mkernel(kern),mnbHidden(nbHidden),mnbHidden2(nbHidden2),mn_input_plane(input_plane), mFeats(mfeatureMaps),mfeatureMaps2(mfeatMaps2),
                      SAMPLE_DEPTH(smpl),BS(bsize)
       { 
         mFast=torch::nn::Sequential(createModel(mFeats,mnbHidden,mn_input_plane ,mkernel,device));
         Head=torch::nn::Sequential(createHead());
         register_module("mFast",mFast);
         register_module("Head" ,Head );
       };
       FastandHeadImpl(){};
/**********************************************************************/
    torch::nn::Sequential createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane,int64_t mks, torch::Device device)
    {
        torch::nn::Sequential Fast;
        for (auto i=0; i<mNbHiddenLayers-1;i++)
        {
            if (i==0) // Initial image: it will take the number of channels of the patch$
            {
    		  Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
              Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
              Fast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    		}
    		else 
    		{
    		Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
            Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
            Fast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    	    }
    	}
    	Fast->push_back(std::string("conv")+std::to_string(mNbHiddenLayers-1),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
    	
        Fast->push_back("NormL2",NormL2());
        return Fast;
    }
/**********************************************************************/
    torch::nn::Sequential createHead()
    {
        torch::nn::Sequential Head;
        /*float Expansion;
        if (SAMPLE_DEPTH*mFeats>mfeatureMaps2) Expansion=mfeatureMaps2/(SAMPLE_DEPTH*mFeats);*/
        int FeatureMaps[mnbHidden2];
        for (int i=0;i<mnbHidden2;i++)
        {
            FeatureMaps[i]=std::round((mfeatureMaps2*i+SAMPLE_DEPTH*mFeats*(mnbHidden2-1-i))/(mnbHidden2-1));
        }
        
    	 for (int i=0; i< mnbHidden2-1;i++)
    	{

			Head->push_back(std::string("Linear")+std::to_string(mnbHidden+i), torch::nn::Linear(torch::nn::LinearOptions(FeatureMaps[i],FeatureMaps[i+1]).bias(true)));
			Head->push_back(std::string("ReLU")+std::to_string(mnbHidden+i),torch::nn::ReLU());
		}
        Head->push_back(std::string("Linear")+std::to_string(mnbHidden+mnbHidden2), torch::nn::Linear(torch::nn::LinearOptions(mfeatureMaps2, mfeatureMaps2).bias(true)));
        return Head;
    }    
/**********************************************************************/

    torch::Tensor forward(torch::Tensor x)   
    {
    	auto& model_ref = * mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	// Reshape 
    	x=x.reshape({BS,SAMPLE_DEPTH*mFeats});
        
    	auto& header_ref = * Head;
        for (auto module : header_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }

/***********************************************************************/
    torch::Tensor forwardInfer(torch::Tensor x)   
    {
    	auto& model_ref = * mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }
/***********************************************************************/
    torch::Tensor forwardMLP(torch::Tensor x)   
    {
    	auto& model_ref = * Head;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }
/***********************************************************************/
	int64_t getKernelSize()
	{ return this->mkernel;}
/***********************************************************************/    
	int64_t getFeats()
	{ return this->mFeats;}
/***********************************************************************/   
	int64_t getFeats2()
	{ return this->mfeatureMaps2;}
        void setBS(int64_t bs)
          {this->BS=bs;}
/***********************************************************************/
   int64_t mkernel;
   int64_t mnbHidden;
   int64_t mnbHidden2;
   int64_t mn_input_plane;
   int64_t mFeats;
   int64_t mfeatureMaps2;
   int64_t SAMPLE_DEPTH;
   int64_t BS;
   torch::nn::Sequential mFast{ nullptr };
   torch::nn::Sequential Head { nullptr };
};

TORCH_MODULE(FastandHead);


/*********************************************************************************************/


class SimilarityNetImpl : public torch::nn::Module {
public:
      SimilarityNetImpl(int64_t kern, int64_t nbHidden,int64_t nbHidden2, int64_t input_plane,int64_t mfeatureMaps,int64_t mfeatMaps2,int64_t bsize,
                      torch::Device device):mkernel(kern),mnbHidden(nbHidden),mnbHidden2(nbHidden2),mn_input_plane(input_plane), mFeats(mfeatureMaps),mfeatureMaps2(mfeatMaps2),BS(bsize)
       { 
         mFast=torch::nn::Sequential(createModel(mFeats,mnbHidden,mn_input_plane ,mkernel,device));
         Head=torch::nn::Sequential(createHead());
         register_module("mFast",mFast);
         register_module("Head" ,Head );
       };
/**********************************************************************/
    torch::nn::Sequential createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane,int64_t mks, torch::Device device)
    {
        torch::nn::Sequential Fast;
        for (auto i=0; i<mNbHiddenLayers-1;i++)
        {
            if (i==0) // Initial image: it will take the number of channels of the patch$
            {
    		  Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
              Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
              Fast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    		}
    		else 
    		{
    		Fast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
            Fast->push_back(std::string("BatchNorm")+std::to_string(i),torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
            Fast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    	    }
    	}
    	Fast->push_back(std::string("conv")+std::to_string(mNbHiddenLayers-1),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
    	
        Fast->push_back("NormL2",NormL2());
        return Fast;
    }
/**********************************************************************/
    torch::nn::Sequential createHead()
    {
        torch::nn::Sequential Head;
    	// Add Fully Connected Layers 
    	 for (auto i=0; i<mnbHidden2;i++)
    	{
			if (i==0)
			{
			Head->push_back(std::string("Linear")+std::to_string(mnbHidden+i), torch::nn::Linear(torch::nn::LinearOptions(2*mFeats,mfeatureMaps2).bias(true)));
			Head->push_back(std::string("ReLU")+std::to_string(mnbHidden+i),torch::nn::ReLU());
			}
			else
			{
			Head->push_back(std::string("Linear")+std::to_string(mnbHidden+i), torch::nn::Linear(torch::nn::LinearOptions(mfeatureMaps2,mfeatureMaps2).bias(true)));
			Head->push_back(std::string("ReLU")+std::to_string(mnbHidden+i),torch::nn::ReLU());
			}
		}
		Head->push_back(std::string("Linear")+std::to_string(mnbHidden+mnbHidden2),
                        torch::nn::Linear(torch::nn::LinearOptions(mfeatureMaps2, 1).bias(true)));
		Head->push_back(std::string("Sigmoid"), torch::nn::Sigmoid());
        return Head;
    }    
/**********************************************************************/

    torch::Tensor forward(torch::Tensor x)   
    {
        x=this->forwardConv(x);
    	// Reshape 
    	x=x.reshape({BS,2*mFeats});
        
        x=this->forwardMLP(x);
        return x;
    }

/***********************************************************************/
    torch::Tensor forwardConv(torch::Tensor x)   
    {
    	auto& model_ref = * mFast;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }
/***********************************************************************/
    torch::Tensor forwardMLP(torch::Tensor x)   
    {
    	auto& model_ref = * Head;
        for (auto module : model_ref)
        {
			//std::cout<<x.is_cuda()<<std::endl;
    		x=module.forward(x);
    	}
    	return x;
    }
/***********************************************************************/
	int64_t getKernelSize()
	{ return this->mkernel;}
/***********************************************************************/    
	int64_t getFeats()
	{ return this->mFeats;}
/***********************************************************************/   
	int64_t getFeats2()
	{ return this->mfeatureMaps2;}
/***********************************************************************/
   int64_t mkernel;
   int64_t mnbHidden;
   int64_t mnbHidden2;
   int64_t mn_input_plane;
   int64_t mFeats;
   int64_t mfeatureMaps2;
   int64_t BS;
   torch::nn::Sequential mFast{ nullptr };
   torch::nn::Sequential Head { nullptr };
};

TORCH_MODULE(SimilarityNet);

class Fast_ProjectionHeadImpl : public torch::nn::Module {
public:
	Fast_ProjectionHeadImpl(int64_t kern, int64_t nbHidden, int64_t nbHidden2, int64_t input_plane, int64_t mfeatureMaps, int64_t mfeatMaps2, int64_t mfeaturesOut,
		torch::Device device) :mkernel(kern), mnbHidden(nbHidden), mnbHidden2(nbHidden2), mn_input_plane(input_plane), mFeats(mfeatureMaps), mfeatureMaps2(mfeatMaps2),
		mfeatsOut(mfeaturesOut)
	{
		mFast = torch::nn::Sequential(createModel(mFeats, mnbHidden, mn_input_plane, mkernel, device));
		Head = torch::nn::Sequential(createHead());
		register_module("mFast", mFast);
		register_module("Head", Head);
	};
	/**********************************************************************/
	torch::nn::Sequential createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane, int64_t mks, torch::Device device)
	{
		torch::nn::Sequential Fast;
		for (auto i = 0; i < mNbHiddenLayers - 1; i++)
		{
			if (i == 0) // Initial image: it will take the number of channels of the patch$
			{
				Fast->push_back(std::string("conv") + std::to_string(i), torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
				Fast->push_back(std::string("BatchNorm") + std::to_string(i), torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
				Fast->push_back(std::string("ReLU") + std::to_string(i), torch::nn::ReLU());
			}
			else
			{
				Fast->push_back(std::string("conv") + std::to_string(i), torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
				Fast->push_back(std::string("BatchNorm") + std::to_string(i), torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(mfeatureMaps).track_running_stats(true)));
				Fast->push_back(std::string("ReLU") + std::to_string(i), torch::nn::ReLU());
			}
		}
		Fast->push_back(std::string("conv") + std::to_string(mNbHiddenLayers - 1), torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));

		return Fast;
	}
	/**********************************************************************/
	torch::nn::Sequential createHead()
	{
		torch::nn::Sequential Head;
		// Add Fully Connected Layers 
		for (auto i = 0; i < mnbHidden2; i++)
		{
			if (i == 0)
			{
				Head->push_back(std::string("Linear") + std::to_string(mnbHidden + i), torch::nn::Linear(torch::nn::LinearOptions(mFeats, mfeatureMaps2).bias(true)));
				//Head->push_back(std::string("BN")+std::to_string(mnbHidden+i),torch::nn::BatchNorm1d(torch::nn::BatchNormOptions(mfeatureMaps2).track_running_stats(true)));
				Head->push_back(std::string("ReLU") + std::to_string(mnbHidden + i), torch::nn::ReLU());
			}
			else
			{
				Head->push_back(std::string("Linear") + std::to_string(mnbHidden + i), torch::nn::Linear(torch::nn::LinearOptions(mfeatureMaps2, mfeatureMaps2).bias(true)));
				//  Head->push_back(std::string("BN")+std::to_string(mnbHidden+i),torch::nn::BatchNorm1d(torch::nn::BatchNormOptions(mfeatureMaps2).track_running_stats(true)));
				Head->push_back(std::string("ReLU") + std::to_string(mnbHidden + i), torch::nn::ReLU());
			}
		}
		Head->push_back(std::string("Linear") + std::to_string(mnbHidden + mnbHidden2),
			torch::nn::Linear(torch::nn::LinearOptions(mfeatureMaps2, mfeatsOut).bias(true)));
		Head->push_back("NormL2", NormL2());
		return Head;
	}
	/**********************************************************************/

	torch::Tensor forward(torch::Tensor x)
	{
		auto& model_ref = *mFast;
		for (auto module : model_ref)
		{
			//std::cout<<x.is_cuda()<<std::endl;
			x = module.forward(x);
		}
		// SQUUEEZE TO REDUCE UNUSED DIMENSIONS  
		x = x.squeeze();

		auto& header_ref = *Head;
		for (auto module : header_ref)
		{
			x = module.forward(x);
		}
		return x;
	}
	/***********************************************************************/
	torch::Tensor forwardInfer(torch::Tensor x)
	{
		auto& model_ref = *mFast;
		for (auto module : model_ref)
		{
			//std::cout<<x.is_cuda()<<std::endl;
			x = module.forward(x);
		}
		//x=x.squeeze();
        x=x.permute({0,2,3,1});
        
		auto& header_ref = *Head;
        int cc=0;
        for (auto module : header_ref)
		{
            //if (Head->named_children()[cc].key()!="NormL2")
            if (cc<3)
            {
                //std::cout<<" Module forward name  : "<<Head->named_children()[cc].key()<<std::endl;
               x = module.forward(x);
            }
            
            cc++;
		}
        x=x.permute({0,3,1,2});
        x=F::normalize(x, F::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
		return x;
	}
	/***********************************************************************/	
	torch::Tensor forwardConv(torch::Tensor x)
	{
		auto& model_ref = *mFast;
		for (auto module : model_ref)
		{
			//std::cout<<x.is_cuda()<<std::endl;
			x = module.forward(x);
		}
		// normalise data before return 
		x=F::normalize(x, F::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
		return x;
	}
	/***********************************************************************/
	int64_t getKernelSize()
	{
		return this->mkernel;
	}
	/***********************************************************************/
	int64_t getFeats()
	{
		return this->mFeats;
	}
	/***********************************************************************/
	int64_t getFeats2()
	{
		return this->mfeatureMaps2;
	}
	/***********************************************************************/
	int64_t mkernel;
	int64_t mnbHidden;
	int64_t mnbHidden2;
	int64_t mn_input_plane;
	int64_t mFeats;
	int64_t mfeatureMaps2;
	int64_t mfeatsOut;
	torch::nn::Sequential mFast{ nullptr };
	torch::nn::Sequential Head{ nullptr };
};

TORCH_MODULE(Fast_ProjectionHead);
/***********************************************************************/

