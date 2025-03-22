#include <torch/torch.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>



/*
torch::Tensor readTIFFIM(torch::Tensor Host, std::string ImgFullPath, int NbrChannels, bool Is16Bit )
{
	Tiff_Im aTifIm = Tiff_Im::StdConvGen(ImgFullPath,NbrChannels,0);
	// IMage to Tensor 
	Im2D<REAL4,REAL8>* mIm=new Im2D<REAL4,REAL8>(aTifIm.sz().x,aTifIm.sz().y);
	ELISE_COPY(mIm->all_pts(),aTifIm.in(),mIm->out());
	REAL4 ** mImData=mIm->data();
	Host=torch::from_blob((*mImData), {1,1,aTifIm.sz().y,aTifIm.sz().x}, torch::TensorOptions().dtype(torch::kFloat32));
	return Host;
}
*/
/*
template <class Type,class TyBase> void ToFile(TIm2D<Type,TyBase> anImage, std::string anImageName)
{
    ELISE_COPY
    (
    anImage.all_pts(),
    anImage.in() ,
    Tiff_Im(
        anImageName.c_str(),
        anImage.sz(),
        GenIm::real4,
        Tiff_Im::No_Compr,
        Tiff_Im::BlackIsZero,
        Tiff_Im::Empty_ARG ).out()
    );
}*/


/*void Tensor2Tiff(torch::Tensor aTens, std::string anImageName)
{
    Im2D<REAL4,REAL8> anIm=Im2D<REAL4,REAL8> (aTens.size(-1),aTens.size(-2));
    REAL4 ** anImD=anIm.data();
    std::memcpy((*anImD),aTens.data_ptr<REAL4>(),sizeof(REAL4)*aTens.numel());
    ELISE_COPY
    (
     anIm.all_pts(),
     anIm.in() ,
     Tiff_Im(
        anImageName.c_str(),
        anIm.sz(),
        GenIm::real4,
        Tiff_Im::No_Compr,
        Tiff_Im::BlackIsZero,
        Tiff_Im::Empty_ARG ).out()
      );   
}*/
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
class ImagePyramid
{
public:
    ImagePyramid(const int nLevels, const float scaleFactor = 1.6f);
    torch::Tensor GaussKern(int LOWPASS_R, float scale);
    void TensorPyramid(torch::Tensor & imT);
    torch::Tensor resizeT(torch::Tensor im, int Newsizex, int Newsizey);
    ~ImagePyramid();
    const std::vector< torch::Tensor > getImPyrT() const { return m_imPyrT; }
private:
    std::vector<torch::Tensor > m_imPyrT;
    int m_nLevels;
    float m_scaleFactor;
};
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

class MulScalePatchGen
{
public:
    MulScalePatchGen(int nblevels,int SizePatchx, int SizePatchy);
    torch::Tensor GenPatchMS(torch::Tensor aFullResPatch, double aLocx, double aLocy);
    ~MulScalePatchGen();
    //torch::Tensor GenAug();
    int mnblevels;
    int mSizePatchx,mSizePatchy;
};
