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
    mW    (0),
    mSzW  (Pt2di(500,500)),
    mReech(1.0)
{
}

void  cAppliZBufferRaster::DoAllIm()
{
    for (int aKIm=0; aKIm<int(mVImg.size()); aKIm++)
    {
       cout<<"Im "<<mVImg[aKIm]<<endl;
       ElTimer aChrono;
       cImgZBuffer * aZBuf =  new cImgZBuffer(this, mVImg[aKIm]);
       for (int aKTri=0; aKTri<int(mVTri.size()); aKTri++)
       {
          if (aKTri % 200 == 0)
            cout<<"["<<(aKTri*100.0/mVTri.size())<<" %]"<<endl;
          aZBuf->LoadTri(mVTri[aKTri]);
       }
       //save Image ZBuffer to disk
       string fileOut = mVImg[aKIm] + "_ZBuffer.tif";
       ELISE_COPY
               (
                   aZBuf->ImZ().all_pts(),
                   aZBuf->ImZ().in_proj(),
                   Tiff_Im(
                       fileOut.c_str(),
                       aZBuf->ImZ().sz(),
                       GenIm::real8,
                       Tiff_Im::No_Compr,
                       aZBuf->Tif().phot_interp()
                       ).out()

                   );
       //=======================================
       if (mNInt != 0)
       {
           aZBuf->normalizeIm(aZBuf->ImZ(), 0.0, 255.0);

           if (aZBuf->ImZ().sz().x >= aZBuf->ImZ().sz().y)
           {
               double scale =  double(aZBuf->ImZ().sz().x) / double(aZBuf->ImZ().sz().y) ;
               mSzW = Pt2di(mSzW.x , round_ni(mSzW.x/scale));
           }
           else
           {
               double scale = double(aZBuf->ImZ().sz().y) / double(aZBuf->ImZ().sz().x);
               mSzW = Pt2di(round_ni(mSzW.y/scale) ,mSzW.y);
           }
           Pt2dr aZ(double(mSzW.x)/double(aZBuf->ImZ().sz().x) , double(mSzW.y)/double(aZBuf->ImZ().sz().y) );

           if (mW ==0)
           {
               mW = Video_Win::PtrWStd(mSzW, true, aZ);
               mW->set_sop(Elise_Set_Of_Palette::TheFullPalette());
           }

           if (mW)
           {
               mW->set_title( (mVImg[aKIm] + "_ZBuf").c_str());
               ELISE_COPY(   aZBuf->ImZ().all_pts(),
                             aZBuf->ImZ().in(),
                             mW->ogray()
                             );
               //mW->clik_in();
           }
       }
       cout<<"Finish Img Cont.. - Nb Tri Valab : "<<aZBuf->CntTriValab()<<" -Time: "<<aChrono.uval()<<endl;
       getchar();
    }
}
