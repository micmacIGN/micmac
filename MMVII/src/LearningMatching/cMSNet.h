#include <torch/torch.h>
#include <utility>
#include <set>

inline torch::nn::Conv2d conv3x3(int64_t in_channels, int64_t out_channels, int64_t stride,int64_t pad) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
        .stride(stride)
        .padding(pad)
        .bias(false));
};

inline torch::nn::Conv2d conv1x1(int64_t in_channels, int64_t out_channels, int64_t stride,int64_t pad) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
        .stride(stride)
        .padding(pad)
        .bias(false));
};



class ResidualBlockImpl : public torch::nn::Module {
 public:
    ResidualBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride,int64_t pad, torch::nn::Sequential downsample = nullptr):
        conv1(conv3x3(in_channels, out_channels, stride,pad)),
        bn1(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels).track_running_stats(true))),
        conv2(conv3x3(out_channels, out_channels,stride,pad)),
        bn2(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels).track_running_stats(true))),
        downsampler(downsample) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", relu);
        register_module("conv2", conv2);
        register_module("bn2", bn2);

        if (downsampler) {
            register_module("downsampler", downsampler);
        }
    };
    torch::Tensor forward(torch::Tensor x) {
        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = relu->forward(out);
        out = conv2->forward(out);
        out = bn2->forward(out);

        auto residual = downsampler ? downsampler->forward(x) : x;
        //std::cout<<" signal "<<out.sizes()<<std::endl;
        //std::cout<<" Residual "<<residual.sizes()<<std::endl;
        out += residual;
        //out = relu->forward(out); no non linear activation after sum 
        return out;
    }
 private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::ReLU relu;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Sequential downsampler;
};

TORCH_MODULE(ResidualBlock);


class MSCAMImpl : public torch::nn::Module {
    
 public:
    MSCAMImpl(int64_t in_channels, int64_t r)
    {
        inter_channels=in_channels/r;
        torch::nn::Sequential alocal_attention;
        
        alocal_attention->push_back(conv1x1(in_channels,inter_channels,1,0));
        alocal_attention->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(inter_channels).track_running_stats(true)));
        alocal_attention->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
        
        alocal_attention->push_back(conv1x1(inter_channels,in_channels,1,0));
        alocal_attention->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(in_channels).track_running_stats(true)));
        

        local_attention=torch::nn::Sequential(alocal_attention);
        
        torch::nn::Sequential aglobal_attention;
        
        aglobal_attention->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        
        aglobal_attention->push_back(conv1x1(in_channels,inter_channels,1,0));
        aglobal_attention->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(inter_channels).track_running_stats(true)));
        aglobal_attention->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
        
        aglobal_attention->push_back(conv1x1(inter_channels,in_channels,1,0));
        aglobal_attention->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(in_channels).track_running_stats(true)));
        
        global_attention=torch::nn::Sequential(aglobal_attention);
        
        register_module("local_attention", local_attention);
        register_module("global_attention", global_attention);
    };
    
    torch::Tensor forward(torch::Tensor X, torch::Tensor Y)
    {
        auto X_a=X+Y;
        //X_a is the addition of Features at Scale R/pow(2,k) and R/pow(2,k+1)
        auto xl=local_attention->forward(X_a);  // BS,FT,7X7
        auto xg=global_attention->forward(X_a); // BS,FT,7X7
        auto xlg= xg + xl; // addition 
        auto weight= torch::sigmoid(xlg);
        return X.mul(weight) + Y.mul(weight.mul(-1.0).add(1.0));
    }

 private:
        int64_t inter_channels;
        torch::nn::Sequential local_attention{nullptr};  // local features more textural 
        torch::nn::Sequential global_attention{nullptr}; // larger scale features more context
};

TORCH_MODULE(MSCAM);

/*******************************************************************************************/
class MSNetImpl:public torch::nn::Module
{
    public:
        // Multi Scale Network 
        MSNetImpl(int64_t inplanes)
        {
            min_planes=inplanes;
            // Create sub modules
            
            // Common layer weight sharing betrween scales
            torch::nn::Sequential afirstconv;
            afirstconv->push_back(conv3x3(1,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            afirstconv->push_back(conv3x3(min_planes,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            afirstconv->push_back(conv3x3(min_planes,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));    
            firstconv=torch::nn::Sequential(afirstconv);
            
            
            //SINGLE SCALE Separate layers 
            
            layer1=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer2=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer3=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer4=_make_layer(32,64,3,1,1); //3 residual blocks 
            
            
            // AGGREG
            torch::nn::Sequential Aggregator; // start reducing dimension of W x H  if 7x7 then 3 conv layers 
            Aggregator->push_back(conv3x3(64*4,128,1,0));
            Aggregator->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(128).track_running_stats(true)));
            Aggregator->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            Aggregator->push_back(conv3x3(128,128,1,0));   
            Aggregator->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(128).track_running_stats(true)));
            Aggregator->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            Aggregator->push_back(conv3x3(128,128,1,0));    // 1x1 
            
            common=torch::nn::Sequential(Aggregator);
            
            register_module("firstconv",firstconv);
            register_module("layer1",layer1);
            register_module("layer2",layer2);
            register_module("layer3",layer3);
            register_module("layer4",layer4);
            register_module("common",common);
        };
        
        torch::nn::Sequential _make_layer(int inplanes, int outplanes,int blocks,int stride,int pad)
        {
            torch::nn::Sequential Out;
            if (inplanes!=outplanes)
            {
                torch::nn::Sequential downsampling;
                downsampling->push_back(conv3x3(inplanes,outplanes,1,1));
                downsampling->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(outplanes).track_running_stats(true)));
                Out->push_back(ResidualBlock(inplanes,outplanes,stride,pad,downsampling));  
            }
            else
            {
                Out->push_back(ResidualBlock(inplanes,outplanes,stride,pad));      // no downsampler nullptr
            }
            for (int i=1;i<blocks;i++)
            {
                Out->push_back(ResidualBlock(outplanes,outplanes,stride,pad));
            }
            return Out;
        }
        
        torch::Tensor forward(torch::Tensor x)
        {
            // shape of X {BS,4,7x7} or {BS,4,5x5}
            auto grps=x.chunk(4,1);
            auto x1=firstconv->forward(grps[0]);
            //std::cout<<" x1 shape "<<x1.sizes()<<std::endl;
            auto x2=firstconv->forward(grps[1]);
            //std::cout<<" x2 shape "<<x2.sizes()<<std::endl;
            auto x3=firstconv->forward(grps[2]);
            //std::cout<<" x3 shape "<<x3.sizes()<<std::endl;
            auto x4=firstconv->forward(grps[3]);
            //std::cout<<" x4 shape "<<x4.sizes()<<std::endl;
            
            x1=layer1->forward(x1);
            //std::cout<<" x1 after specialized module : "<<x1.sizes()<<std::endl;
            x2=layer2->forward(x2);
            //std::cout<<" x2 after specialized module : "<<x2.sizes()<<std::endl;
            x3=layer3->forward(x3);
            //std::cout<<" x3 after specialized module : "<<x3.sizes()<<std::endl;
            x4=layer4->forward(x4);
            //std::cout<<" x4 after specialized module : "<<x4.sizes()<<std::endl;
            
            auto x_all=torch::cat({x1,x2,x3,x4},1); //dim {Bs,64*4,W,H}
            //std::cout<<" xall after concat : "<<x_all.sizes()<<std::endl;
            x_all=common->forward(x_all); 
            //std::cout<<" xall after aggregation module  : "<<x_all.sizes()<<std::endl;
            
            // check if it needs to be normalized   
            
            return x_all;  //dim {Bs,128,1,1} A multi scale representation
        }
        
        torch::Tensor forwardFullRes1(torch::Tensor x)
        {
            auto grps=x.chunk(4,1);
            auto x1=firstconv->forward(grps[0]);
            x1=layer2->forward(x1);
            return x1;
        }
        torch::Tensor forwardRes1(torch::Tensor x)
        {
            //auto x1=firstconv->forward(x);
            auto x1=layer1->forward(x);
            return x1;
        }
        torch::Tensor forwardRes2(torch::Tensor x)
        {
            //auto x2=firstconv->forward(x);
            auto x2=layer2->forward(x);
            return x2;
        }
        torch::Tensor forwardRes3(torch::Tensor x)
        {
            //x3=firstconv->forward(x);
            auto x3=layer3->forward(x);
            return x3;
        }
        torch::Tensor forwardRes4(torch::Tensor x)
        {
            //auto x4=firstconv->forward(x);
            auto x4=layer4->forward(x);
            return x4;
        }
        
        torch::Tensor forwardConvCommon(torch::Tensor x)
        {
            auto xCom=firstconv->forward(x);         
            return xCom;
        }
    //private:
        int64_t min_planes=32;
        torch::nn::Sequential firstconv{nullptr};  // weight sharing sub module 
        torch::nn::Sequential layer1{nullptr};     // specialized modules 
        torch::nn::Sequential layer2{nullptr};     // .. 
        torch::nn::Sequential layer3{nullptr};     // ..
        torch::nn::Sequential layer4{nullptr};     // ..
        torch::nn::Sequential common{nullptr};     // Aggregation module last features
};

TORCH_MODULE(MSNet);


class MSNetHeadImpl:public torch::nn::Module
{
    public:
        MSNetHeadImpl(int64_t inplanes)
        {
            min_planes=inplanes;
            // Create sub modules
            
            // Common layer weight sharing betrween scales
            torch::nn::Sequential afirstconv;
            afirstconv->push_back(conv3x3(1,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            afirstconv->push_back(conv3x3(min_planes,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            afirstconv->push_back(conv3x3(min_planes,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));    
            firstconv=torch::nn::Sequential(afirstconv);
            
            
            //SINGLE SCALE Separate layers 
            
            layer1=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer2=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer3=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer4=_make_layer(32,64,3,1,1); //3 residual blocks 
            
            
            // AGGREG
            
            /*
            torch::nn::Sequential Aggregator; 
            Aggregator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64*4, 64*4).bias(true)));
            Aggregator->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1d(64 * 4)));
            Aggregator->push_back(torch::nn::ReLU());
            Aggregator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64 * 4, 128).bias(true)));
            Aggregator->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1d(128)));
            Aggregator->push_back(torch::nn::ReLU());
            Aggregator->push_back(torch::nn::Linear(torch::nn::LinearOptions(128, 128).bias(true)));
            common=torch::nn::Sequential(Aggregator);
            */
            torch::nn::Sequential Aggregator;
            Aggregator->push_back(torch::nn::Linear(torch::nn::LinearOptions(64*4, 384).bias(true)));
            Aggregator->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(384).affine(true).track_running_stats(true)));
            Aggregator->push_back(torch::nn::ReLU());
            Aggregator->push_back(torch::nn::Linear(torch::nn::LinearOptions(384, 384).bias(true)));
            Aggregator->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(384).affine(true).track_running_stats(true)));
            Aggregator->push_back(torch::nn::ReLU());
            Aggregator->push_back(torch::nn::Linear(torch::nn::LinearOptions(384, 384).bias(true)));
            Aggregator->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(384).affine(true).track_running_stats(true)));
            Aggregator->push_back(torch::nn::ReLU());
            Aggregator->push_back(torch::nn::Linear(torch::nn::LinearOptions(384, 256).bias(true))); 
            common=torch::nn::Sequential(Aggregator);            
            
            register_module("firstconv",firstconv);
            register_module("layer1",layer1);
            register_module("layer2",layer2);
            register_module("layer3",layer3);
            register_module("layer4",layer4);
            register_module("common",common);
        };
 
        torch::nn::Sequential _make_layer(int inplanes, int outplanes,int blocks,int stride,int pad)
        {
            torch::nn::Sequential Out;
            if (inplanes!=outplanes)
            {
                torch::nn::Sequential downsampling;
                downsampling->push_back(conv3x3(inplanes,outplanes,1,1));
                downsampling->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(outplanes).track_running_stats(true)));
                Out->push_back(ResidualBlock(inplanes,outplanes,stride,pad,downsampling));  
            }
            else
            {
                Out->push_back(ResidualBlock(inplanes,outplanes,stride,pad));      // no downsampler nullptr
            }
            for (int i=1;i<blocks;i++)
            {
                Out->push_back(ResidualBlock(outplanes,outplanes,stride,pad));
                Out->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ 1,1 }).stride(1).padding(0)));
            }
            return Out;
        }
        torch::Tensor forwardTestConv(torch::Tensor x)
        {
            auto grps=x.chunk(4,1);
            auto x1=firstconv->forward(grps[0]);
            //std::cout<<" x1 shape "<<x1.sizes()<<std::endl;
            auto x2=firstconv->forward(grps[1]);
            //std::cout<<" x2 shape "<<x2.is_cuda() <<std::endl;
            auto x3=firstconv->forward(grps[2]);
            //std::cout<<" x3 shape "<<x3.sizes()<<std::endl;
            auto x4=firstconv->forward(grps[3]);
            //std::cout<<" x4 shape "<<x4.sizes()<<std::endl;
            
            x1=layer1->forward(x1);
        // std::cout<<" x1 after specialized module : "<<x1.sizes()<<std::endl;
            x2=layer2->forward(x2);
            //std::cout<<" x2 after specialized module : "<<x2.is_cuda() <<std::endl;
            x3=layer3->forward(x3);
            //std::cout<<" x3 after specialized module : "<<x3.sizes()<<std::endl;
            x4=layer4->forward(x4);
            //std::cout<<" x4 after specialized module : "<<x4.sizes()<<std::endl;
            

            // pooling the size of the patch size 
            //x1= F::max_pool2d(x1, F::MaxPool2dFuncOptions(7).stride(1)); // B,C,1,1
            //std::cout<<" After maxpool operation size of X1  "<<x1.sizes()<<std::endl;
            //x2= F::max_pool2d(x2, F::MaxPool2dFuncOptions(7).stride(1)); // B,C,1,1
            //x3= F::max_pool2d(x3, F::MaxPool2dFuncOptions(7).stride(1)); // B,C,1,1
            //x4= F::max_pool2d(x4, F::MaxPool2dFuncOptions(7).stride(1)); // B,C,1,1

            auto x_all=torch::cat({x1,x2,x3,x4},1); //dim {Bs,64*4,W,H}
            return x_all.squeeze();
        }
        
       torch::Tensor forward(torch::Tensor x)
        {
            // shape of X {BS,4,7x7} or {BS,4,5x5}
            auto grps=x.chunk(4,1);
            auto x1=firstconv->forward(grps[0]);
            //std::cout<<" x1 shape "<<x1.sizes()<<std::endl;
            auto x2=firstconv->forward(grps[1]);
            //std::cout<<" x2 shape "<<x2.is_cuda() <<std::endl;
            auto x3=firstconv->forward(grps[2]);
            //std::cout<<" x3 shape "<<x3.sizes()<<std::endl;
            auto x4=firstconv->forward(grps[3]);
            //std::cout<<" x4 shape "<<x4.sizes()<<std::endl;
            
            x1=layer1->forward(x1);
        // std::cout<<" x1 after specialized module : "<<x1.sizes()<<std::endl;
            x2=layer2->forward(x2);
            //std::cout<<" x2 after specialized module : "<<x2.is_cuda() <<std::endl;
            x3=layer3->forward(x3);
            //std::cout<<" x3 after specialized module : "<<x3.sizes()<<std::endl;
            x4=layer4->forward(x4);
            //std::cout<<" x4 after specialized module : "<<x4.sizes()<<std::endl;
            

            // pooling the size of the patch size 
            //x1= F::max_pool2d(x1, F::MaxPool2dFuncOptions(7).stride(1)); // B,C,1,1
            //std::cout<<" After maxpool operation size of X1  "<<x1.sizes()<<std::endl;
            //x2= F::max_pool2d(x2, F::MaxPool2dFuncOptions(7).stride(1)); // B,C,1,1
            //x3= F::max_pool2d(x3, F::MaxPool2dFuncOptions(7).stride(1)); // B,C,1,1
            //x4= F::max_pool2d(x4, F::MaxPool2dFuncOptions(7).stride(1)); // B,C,1,1

            auto x_all=torch::cat({x1,x2,x3,x4},1); //dim {Bs,64*4,W,H}
            x_all = x_all.squeeze();
            // FOR MLP THERE NEED TO BE CONCAT 
            //std::cout<<"TEnsor sizes before transpose "<<x_all.sizes()<<std::endl;
            x_all=x_all.permute({1,2,0}); // size of (w,h,Featuresize)
        // std::cout<<" xall after concat : "<<x_all.sizes()<<std::endl;
            //std::cout<<"TEnsor sizes after transpose "<<x_all.sizes()<<std::endl;
            
            namespace F=torch::nn::functional;
            
            /*for (auto& module : common->children())
             {
                if(auto* linear = module->as<torch::nn::Linear>())
                    {
                        x_all=linear->forward(x_all);
                    }
                 if(auto* bn = module->as<torch::nn::BatchNorm1d>())
                    {
                        // return to initial config 
                        x_all=x_all.transpose(3,2).transpose(2,1);
                        x_all=bn->forward(x_all);
                        // go back again 
                        x_all=x_all.transpose(1,2).transpose(2,3); 
                    }
                 if(auto* rl = module->as<torch::nn::ReLU>())
                    {
                        x_all=rl->forward(x_all);
                    }
            }*/  
            // perform hand-crafted normalization on the set of tiles using running statistics from the model
            
            for (auto& module : common->children())
             {
                if(auto* linear = module->as<torch::nn::Linear>())
                    {
                        x_all=linear->forward(x_all);
                    }
                 if(auto* bn = module->as<torch::nn::BatchNorm1d>())
                    {
                        // return to initial config 
                        //auto x_all0=x_all.transpose(3,2).transpose(2,1);
                       /* std::cout<<"RUNNING MEAN TENSOR SIZE "<<bn->running_mean.sizes()<<std::endl;
                        std::cout<<"RUNNING STDEV TENSOR SIZE "<<bn->running_var.sizes()<<std::endl;
                        std::cout<<"RUNNING gamma TENSOR SIZE "<<bn->weight.sizes()<<std::endl;
                        std::cout<<"RUNNING beta TENSOR SIZE "<<bn->bias.sizes()<<std::endl;*/
                        
                        //x_all=bn->forward(x_all0);

                        //auto x_norm=x_all.sub(bn->running_mean).div(torch::sqrt(bn->running_var.add(0.00001)));
                        //x_all=bn->weight.mul(x_norm).add(bn->bias);
                        //assert(torch::equal(x_all,x_hand));
                        // go back again 
                        //x_all=x_all.transpose(1,2).transpose(2,3); 
                        x_all=bn->forward(x_all.permute({0,2,1})).permute({0,2,1});
                        //std::cout<<"Entered batch Norm "<<std::endl;
                    }
                 if(auto* rl = module->as<torch::nn::ReLU>())
                    {
                        x_all=rl->forward(x_all);
                    }
            }             
             
            //x_all=x_all.squeeze();
            //std::cout<<" xall after aggregation module  : "<<x_all.is_cuda()<<std::endl;
            x_all=x_all.transpose(2,1).transpose(1,0);  // back to  {feature size,Height,Width}
            // check if it needs to be normalized   
            return x_all;  //dim {Bs,128} A multi scale representation
        }        
        int64_t min_planes=32;
        torch::nn::Sequential firstconv{nullptr};  // weight sharing sub module 
        torch::nn::Sequential layer1{nullptr};     // specialized modules 
        torch::nn::Sequential layer2{nullptr};     // .. 
        torch::nn::Sequential layer3{nullptr};     // ..
        torch::nn::Sequential layer4{nullptr};     // ..
        torch::nn::Sequential common{nullptr};     // Aggregation module last features
};

TORCH_MODULE(MSNetHead);

/*******************************************************************************************/
class MSNet_AttentionImpl:public torch::nn::Module
{
    public:
        // Multi Scale Network 
        MSNet_AttentionImpl(int64_t inplanes)
        {
            min_planes=inplanes;
            // Create sub modules
            
            // Common layer weight sharing betrween scales
            torch::nn::Sequential afirstconv;
            afirstconv->push_back(conv3x3(1,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            afirstconv->push_back(conv3x3(min_planes,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            afirstconv->push_back(conv3x3(min_planes,min_planes,1,1));
            afirstconv->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(min_planes).track_running_stats(true)));
            afirstconv->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));    
            firstconv=torch::nn::Sequential(afirstconv);
            
            
            //SINGLE SCALE Separate layers 
            
            layer1=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer2=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer3=_make_layer(32,64,3,1,1); //3 residual blocks 
            layer4=_make_layer(32,64,3,1,1); //3 residual blocks 
            
            // Create Gradual from Coarse to fine Feature Fusion Module 
            
            MultiScaleFeatureFuser3_4=torch::nn::Sequential(MSCAMImpl(64,4));
            MultiScaleFeatureFuser2_3_4=torch::nn::Sequential(MSCAMImpl(64,4));    
            MultiScaleFeatureFuser1_2_3_4=torch::nn::Sequential(MSCAMImpl(64,4));
            
            
            
            // COMMON ARCHITRECTURE TO REDUCE THE SPATILA DIMENSION OF FEATURES
            torch::nn::Sequential acommon; // start reducing dimension of W x H  if 7x7 then 3 conv layers 
            acommon->push_back(conv3x3(64,64,1,0));
            acommon->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(64).track_running_stats(true)));
            acommon->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            acommon->push_back(conv3x3(64,64,1,0));   
            acommon->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(64).track_running_stats(true)));
            acommon->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
            
            acommon->push_back(conv3x3(64,64,1,0));    // 1x1 
            
            common=torch::nn::Sequential(acommon);
            
            register_module("firstconv",firstconv);
            register_module("layer1",layer1);
            register_module("layer2",layer2);
            register_module("layer3",layer3);
            register_module("layer4",layer4);
            register_module("MultiScaleFeatureFuser3_4",MultiScaleFeatureFuser3_4);
            register_module("MultiScaleFeatureFuser2_3_4",MultiScaleFeatureFuser2_3_4);
            register_module("MultiScaleFeatureFuser1_2_3_4",MultiScaleFeatureFuser1_2_3_4);
            register_module("common",common);
        };
        
        torch::nn::Sequential _make_layer(int inplanes, int outplanes,int blocks,int stride,int pad)
        {
            torch::nn::Sequential Out;
            if (inplanes!=outplanes)
            {
                torch::nn::Sequential downsampling;
                downsampling->push_back(conv3x3(inplanes,outplanes,1,1));
                downsampling->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(outplanes).track_running_stats(true)));
                Out->push_back(ResidualBlock(inplanes,outplanes,stride,pad,downsampling));  
            }
            else
            {
                Out->push_back(ResidualBlock(inplanes,outplanes,stride,pad));      // no downsampler nullptr
            }
            for (int i=1;i<blocks;i++)
            {
                Out->push_back(ResidualBlock(outplanes,outplanes,stride,pad));
            }
            return Out;
        }
        
        torch::Tensor forward(torch::Tensor x)
        {
            // shape of X {BS,4,7x7} or {BS,4,5x5}
            auto grps=x.chunk(4,1);
            auto x1=firstconv->forward(grps[0]);
            //std::cout<<" x1 shape "<<x1.sizes()<<std::endl;
            auto x2=firstconv->forward(grps[1]);
            //std::cout<<" x2 shape "<<x2.sizes()<<std::endl;
            auto x3=firstconv->forward(grps[2]);
            //std::cout<<" x3 shape "<<x3.sizes()<<std::endl;
            auto x4=firstconv->forward(grps[3]);
            //std::cout<<" x4 shape "<<x4.sizes()<<std::endl;
            
            x1=layer1->forward(x1);
            //std::cout<<" x1 after specialized module : "<<x1.sizes()<<std::endl;
            x2=layer2->forward(x2);
            //std::cout<<" x2 after specialized module : "<<x2.sizes()<<std::endl;
            x3=layer3->forward(x3);
            //std::cout<<" x3 after specialized module : "<<x3.sizes()<<std::endl;
            x4=layer4->forward(x4);
            //std::cout<<" x4 after specialized module : "<<x4.sizes()<<std::endl;
            
            
            //return x2;
            // APPLY THE SELF ATTENTION BASED FUSION MODULES 
            
            
            auto x_3_4=MultiScaleFeatureFuser3_4->forward(x3,x4);
            auto x_2_3_4=MultiScaleFeatureFuser2_3_4->forward(x2,x_3_4);
            auto x_all = MultiScaleFeatureFuser1_2_3_4->forward(x1,x_2_3_4);
            
            
            x_all=common->forward(x_all); 
            //std::cout<<" xall after aggregation module  : "<<x_all.sizes()<<std::endl;
            
            
            // check if it needs to be normalized   
            
            return x_all;  //dim {Bs,128,1,1} A multi scale representation
        }
        
        torch::Tensor forwardFullRes1(torch::Tensor x)
        {
            auto grps=x.chunk(4,1);
            auto x1=firstconv->forward(grps[0]);
            x1=layer1->forward(x1);
            return x1;
        }
    //private:
        int64_t min_planes=32;
        torch::nn::Sequential firstconv{nullptr};  // weight sharing sub module 
        torch::nn::Sequential layer1{nullptr};     // specialized modules 
        torch::nn::Sequential layer2{nullptr};     // .. 
        torch::nn::Sequential layer3{nullptr};     // ..
        torch::nn::Sequential layer4{nullptr};     // ..
        torch::nn::Sequential MultiScaleFeatureFuser3_4{nullptr}; // Feature fusion based on attention
        torch::nn::Sequential MultiScaleFeatureFuser2_3_4{nullptr}; // Feature fusion based on attention
        torch::nn::Sequential MultiScaleFeatureFuser1_2_3_4{nullptr}; // Feature fusion based on attention
        torch::nn::Sequential common{nullptr};     // further training module last features
};

TORCH_MODULE(MSNet_Attention);

