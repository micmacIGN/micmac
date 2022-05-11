#include "cPatchMS.h"

/******************************************************/
/**
 * 
 * 
 * ImagePyramid  Classs
 * 
 * 
 * 
 * */
/******************************************************/

ImagePyramid::ImagePyramid( const int nLevels, const float scaleFactor)
{
    m_nLevels = nLevels;
    m_scaleFactor = scaleFactor;
}

void ImagePyramid::TensorPyramid(torch::Tensor & imT)
{
    m_imPyrT.clear();
    m_imPyrT.resize(m_nLevels);
    m_imPyrT[0]=torch::empty({1,1,imT.size(2),imT.size(3)},torch::TensorOptions().dtype(torch::kFloat32));
    namespace F= torch::nn::functional;
    auto kern=this->GaussKern(2,1.0);
    //std::cout<<"Noyau gaussien 0 "<<kern<<std::endl;
    auto imTcnv=F::conv2d(imT, kern.unsqueeze(0).unsqueeze(0), F::Conv2dFuncOptions().padding(2));
    m_imPyrT[0].copy_(imTcnv);
    if(m_nLevels > 1)
    {
        for (int lvl = 1; lvl < m_nLevels; lvl++)
        {
            //std::cout<<" Value "<<lvl<<std::endl;
            float scale = 1 / std::pow(m_scaleFactor, (float)lvl);
            auto kern=this->GaussKern(2,scale);
            int Newsizex=round(imT.size(3)*scale);
            int Newsizey=round(imT.size(2)*scale);
            m_imPyrT[lvl]=torch::empty({1,1,Newsizey,Newsizex},torch::TensorOptions().dtype(torch::kFloat32));
            auto aCropT=this->resizeT(imT,Newsizex,Newsizey);
            m_imPyrT[lvl].copy_(F::conv2d(aCropT, kern.unsqueeze(0).unsqueeze(0), F::Conv2dFuncOptions().padding(2)));
        }
    }
    //std::cout<<"Pyramid tensor created ! "<<std::endl;
}

torch::Tensor ImagePyramid::GaussKern(int LOWPASS_R, float scale)
{
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*scale*scale); 
    torch::Tensor kernel=torch::empty({2*LOWPASS_R+1,2*LOWPASS_R+1},torch::TensorOptions().dtype(torch::kFloat32));
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) 
    {
      for (int i=-LOWPASS_R;i<=LOWPASS_R;i++)
      {
         float val= (float)expf(-((double)j*j+(double)i*i)*ivar2);
         kernelSum+=val;
         kernel.index_put_({i+LOWPASS_R,j+LOWPASS_R},val);
      }
    }
    kernel=kernel.div(kernelSum);
    return kernel;
}

torch::Tensor ImagePyramid::resizeT(torch::Tensor im, int Newsizex, int Newsizey)
{
    namespace F = torch::nn::functional;
    auto Out = F::interpolate(
        im,
        F::InterpolateFuncOptions()
                .mode(torch::kNearest)
                .size(std::vector<int64_t>({Newsizey, Newsizex}))
        );
    return Out;
}

ImagePyramid::~ImagePyramid()
{
}
/******************************************************/
/**
 * 
 * 
 * MulScalePatchGen  Classs
 * 
 * 
 * 
 * */
/******************************************************/
MulScalePatchGen::MulScalePatchGen(int nblevels,int SizePatchx, int SizePatchy):

    mnblevels(nblevels),
    mSizePatchx(SizePatchx),
    mSizePatchy(SizePatchy)
{    
}
torch::Tensor MulScalePatchGen::GenPatchMS(torch::Tensor aFullResImage, double aLocx, double aLocy )
{
    //N.B SizePatch Must Fit to coarse Resolution pyramid image
    float scaleFactor=1.316f; // window size of 21 at coarse scale 
    int aLocIx=(int)aLocx;
    int aLocIy=(int)aLocy;
    float scaleF = 1.0 / std::pow(scaleFactor, mnblevels);
    //std::cout<<"SCALE F "<<scaleF<<std::endl;
    int SizeImage0[2]={(int)lroundf((1/scaleF)*(float)mSizePatchx),(int)lroundf((1/scaleF)*(float)mSizePatchy)};
    //std::cout<<"Size tile "<<SizeImage0[2]<<std::endl;
    // Image of Pyramid : SIZE = SizeImage0
    int supplem0=SizeImage0[0]%2;
    int supplem1=SizeImage0[1]%2;
    
    // Check if computed size does not go outside image shape 
    bool IsContext= ( (aLocIx-SizeImage0[0]/2)>0 )&& ((aLocIx+SizeImage0[0]/2+supplem0) <  aFullResImage.size(-1))
                   && ( (aLocIy-SizeImage0[1]/2)>0 ) && ((aLocIy+SizeImage0[1]/2+supplem1) <  aFullResImage.size(-2));
    assert// very dangerous
    (
        IsContext
    );
    
    using namespace torch::indexing;
    // take the relevant slice from the original tensor 
    torch::Tensor aSliceOfImage= torch::empty({1,1,SizeImage0[1],SizeImage0[0]},torch::TensorOptions().dtype(torch::kFloat32));
    aSliceOfImage.copy_(aFullResImage.slice(-2,aLocIy-SizeImage0[1]/2,aLocIy+SizeImage0[1]/2+supplem1,1).slice(-1,aLocIx-SizeImage0[0]/2,aLocIx+SizeImage0[0]/2+supplem0,1));
    ImagePyramid PatchPyram=ImagePyramid(mnblevels, scaleFactor);
    PatchPyram.TensorPyramid(aSliceOfImage);
    
    std::vector<torch::Tensor> FromTensor=PatchPyram.getImPyrT();
    
    torch::Tensor MultiResPatch= torch::empty({mnblevels,mSizePatchy,mSizePatchx},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    
    for (int i=0; i<(int)FromTensor.size();i++)
    {
        int aCenterx=FromTensor.at(i).size(-1)/2;
        int aCentery=FromTensor.at(i).size(-2)/2;
        int supp0=mSizePatchx%2;
        int supp1=mSizePatchy%2;
        int p0[2]={aCenterx-mSizePatchx/2,aCentery-mSizePatchy/2};
        int p1[2]={aCenterx+mSizePatchx/2+supp0,aCentery+mSizePatchy/2+supp1};
        // Get the patch at each level
        torch::Tensor aPatch=torch::empty({1,1,mSizePatchy,mSizePatchx},torch::TensorOptions().dtype(torch::kFloat32));
        aPatch.copy_(FromTensor.at(i).slice(-2,p0[1],p1[1],1).slice(-1,p0[0],p1[0],1));
        MultiResPatch.index_put_({i},aPatch.index({0})); // not clean implementation
    }
     //MultiResPatch= MultiResPatch.unsqueeze(0);
    
    return MultiResPatch;
}   

MulScalePatchGen::~MulScalePatchGen()
{
}
