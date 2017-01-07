#include "ZBufferRaster.h"

cAppliZBufferRaster::cAppliZBufferRaster(
                                            cInterfChantierNameManipulateur * aICNM,
                                            const std::string & aDir,
                                            const std::string & anOri,
                                            vector<cTri3D> & aVTri,
                                            vector<string> & aVImg
                                        ):
    mICNM (aICNM),
    mDir  (aDir),
    mOri  (anOri),
    mVTri (aVTri),
    mVImg (aVImg)
{
    cout<<"cAppliZBufferRaster : "<<endl;
    cout<<"mDir "<<mDir<<endl;
    cout<<" mOri "<<mOri<<endl;
    cout<<" nb Tri "<<mVTri.size()<<endl;
    cout<<" nb Img "<<mVImg.size()<<endl;



}

void  cAppliZBufferRaster::DoAllIm()
{
    for (int aKIm=0; aKIm<int(mVImg.size()); aKIm++)
    {
       cout<<"Im "<<mVImg[aKIm]<<endl;
       cImgZBuffer * aZBuf =  new cImgZBuffer(this, mVImg[aKIm]);
       for (int aKTri=0; aKTri<int(mVTri.size()); aKTri++)
          aZBuf->LoadTri(mVTri[aKTri]);
    }
}
