#pragma once
#include <torch/torch.h>
namespace F = torch::nn::functional;





/**********************************************************************/
class NormL2:public torch::nn::Module{
	
public:
      NormL2()
      {};
      torch::Tensor forward (torch::Tensor input)
      {
		  //torch::Tensor out=F::normalize(input, F::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
		  //std::cout<<"effect of normalization "<<out.index({0})<<std::endl;
          return F::normalize(input, F::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
	   };
};

/**********************************************************************/
/**********************************************************************/


class StereoJoin1 : public torch::nn::Module {
public:
     StereoJoin1 ()
     {
	 };
     std::vector <torch::Tensor> slice_input(torch::Tensor input) // the vector contains input_L, et input_R
     {
        std::vector <torch::Tensor> out;
        torch::Tensor input_L = torch::empty({input.size(0)/2, input.size(1),input.size(2),input.size(3)},torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor input_R = torch::empty({input.size(0)/2, input.size(1),input.size(2),input.size(3)},torch::TensorOptions().dtype(torch::kFloat32));
        // fill tensors left and right 
        for (int i=0;i<input.size(0)/2;i++)
        {
			//std::cout<<"sizes for checkoing "<<input.index({2*i}).sizes()<<std::endl;
           input_L.index_put_({i},input.index({2*i}));
           input_R.index_put_({i},input.index({2*i+1}));
         }
        return {input_L,input_R};
	  };
     /****updateOutput*****/
     
     torch::Tensor forward (torch::Tensor input)
     {
		//std::cout<<"sizes for checkoing "<<input.sizes()<<std::endl;
       std::vector <torch::Tensor> out = this->slice_input(input);
       torch::Tensor tmp=torch::empty({input.size(0)/2, input.size(1),input.size(2),input.size(3)},torch::TensorOptions().dtype(torch::kFloat32));
       tmp.reshape_as(out.at(0));     // changing private attributes
       tmp=out.at(0).mul(out.at(1));  //multilying both left and right tensors
       return torch::sum(tmp,1);            // Summing for each row 
     };
     //torch::Tensor updateOutput(torch::Tensor input);
     //void updateGradInput(torch::Tensor input, torch::Tensor gradOutput);
	};
/**********************************************************************/
/**********************************************************************/
/**********************************************************************/
/*                HINGE LOSS FUNCTION                                 */
/**********************************************************************/
class SiameseLoss: public torch::nn::Module {
public:
      SiameseLoss (double margin):mMargin(margin){};
      double getMargin()
      {
		  return this->mMargin;
	   };
	  void setMargin (double Margin)
	  {
         this->mMargin=Margin;
	  };
      /**********************************************************************/
      torch::Tensor forward_on_contrastive (torch::Tensor x1,torch::Tensor x2, int label)
      {
      	//torch::nn::CosineSimilarity CosineSim=torch::nn::CosineSimilarity();// to check later !!!!!!!!!
      	torch::Tensor similarity=F::cosine_similarity(x1, x2, F::CosineSimilarityFuncOptions().dim(1));
      	torch::Tensor clmp=at::mul(at::sub(similarity,this->getMargin()),-1);
      	torch::Tensor loss_c=at::mean(at::mul(at::pow(2,similarity),1-label)+at::mul(at::pow(2,at::clamp(clmp,0.0)),label));
      	return loss_c;
      };
      
      torch::Tensor forward(torch::Tensor input, torch::Tensor target)
      {
        // get hinge loss for each couple of data 
        torch::Tensor pair=torch::empty({input.size(0)/2},torch::TensorOptions().dtype(torch::kInt32));
        torch::Tensor Impair=torch::empty({input.size(0)/2},torch::TensorOptions().dtype(torch::kInt32));
        
        for (int i=0;i<input.size(0)/2;i++)
        {
			pair.index_put_({i},2*i);
			Impair.index_put_({i},2*i+1);
		}
        torch::Tensor similarity_plus=torch::index_select(input,0,pair);
        torch::Tensor similarity_minus=torch::index_select(input,0,Impair);
        similarity_plus=torch::squeeze(similarity_plus);
        similarity_minus=torch::squeeze(similarity_minus);
        auto metric=similarity_minus.add(this->getMargin()).sub(similarity_plus); //correction error in loss function 
        metric=torch::fmax(metric,torch::zeros({input.size(0)/2}));
        metric=torch::mean(metric,0,1);
        return metric;
       } ;
       
private:
    double mMargin=0.2;
};

/**********************************************************************/
class ConvNet_FastImpl : public torch::nn::Module {
public:
      ConvNet_FastImpl(int64_t kern, int64_t nbHidden):mkernel(kern),mnbHidden(nbHidden)
       { };
/**********************************************************************/
    void createModel(int64_t mfeatureMaps, int64_t mNbHiddenLayers, int64_t mn_input_plane,int64_t mks)
    {
        for (auto i=0; i<mNbHiddenLayers-1;i++)
        {
            if (i==0) // Initial image: it will take the number of channels of the patch$
            {
    		  mFast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mn_input_plane, mfeatureMaps, mks).stride(1).padding(0)));  // we ll see types of padding later
              mFast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    		}
    		else 
    		{
    		mFast->push_back(std::string("conv")+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
            mFast->push_back(std::string("ReLU")+std::to_string(i),torch::nn::ReLU());
    	    }
    	}
    	mFast->push_back(std::string("conv")+std::to_string(mNbHiddenLayers-1),torch::nn::Conv2d(torch::nn::Conv2dOptions(mfeatureMaps, mfeatureMaps, mks).stride(1).padding(0)));
    	
        mFast->push_back("NormL2", NormL2());
        mFast->push_back("StereoJoin",StereoJoin1());
    }
/**********************************************************************/
    torch::Tensor forward(torch::Tensor x)   
    {
    	auto& model_ref = *mFast;
        for (auto module : model_ref)
        {
    		x=module.forward(x);
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
    		{
				x=module.forward(x);
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
   // add hidden layers to sequential staff
   torch::nn::Sequential mFast;
};

TORCH_MODULE(ConvNet_Fast);





