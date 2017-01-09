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
    mVImg (aVImg),
    mNInt (0),
    mW    (0)
{
}

void  cAppliZBufferRaster::DoAllIm()
{
    for (int aKIm=0; aKIm<int(mVImg.size()); aKIm++)
    {
       cout<<"Im "<<mVImg[aKIm]<<endl;
       cImgZBuffer * aZBuf =  new cImgZBuffer(this, mVImg[aKIm]);
       for (int aKTri=0; aKTri<int(mVTri.size()); aKTri++)
       {
          if (aKTri % 200 == 0)
            cout<<"["<<(aKTri*100.0/mVTri.size())<<" %]"<<endl;
          aZBuf->LoadTri(mVTri[aKTri]);
       }
       aZBuf->normalizeIm(aZBuf->ImZ(), 0.0, 255.0);
       //save Image ZBuffer to disk
       string fileOut = mVImg[aKIm] + "_ZBuffer.tif";
       ELISE_COPY
       (
           aZBuf->ImZ().all_pts(),
           aZBuf->ImZ().inside() ,
           Tiff_Im(
               fileOut.c_str(),
               aZBuf->ImZ().sz(),
               GenIm::real8,
               Tiff_Im::No_Compr,
               Tiff_Im::BlackIsZero,
               Tiff_Im::Empty_ARG ).out()
       );
       if (mNInt != 0)
       {
           aZBuf->normalizeIm(aZBuf->ImZ(), 0.0, 255.0);
           if (mW ==0)
           {
               double aZ = 0.5;
               mW = Video_Win::PtrWStd(Pt2di(aZBuf->ImZ().sz()*aZ), true, Pt2dr(aZ, aZ));
               mW->set_sop(Elise_Set_Of_Palette::TheFullPalette());
           }

           if (mW)
           {
               mW->set_title( (mVImg[aKIm] + "_ZBuf").c_str());
               //ELISE_COPY(mImZ.all_pts(),mImZ.in(),mW->ogray());
               ELISE_COPY(   aZBuf->ImZ().all_pts(),
                             aZBuf->ImZ().in(),
                             mW->ogray()
                             );
               //mW->clik_in();
           }
       }
       cout<<"Finish Img Cont..Nb mCntTriValab : "<<aZBuf->CntTriValab()<<endl;
       getchar();
    }
}
