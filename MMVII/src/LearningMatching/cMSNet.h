#include <torch/torch.h>
#include <utility>
#include <set>

inline torch::nn::Conv2d conv3x3(int64_t in_channels, int64_t out_channels, int64_t stride,int64_t pad) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
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


