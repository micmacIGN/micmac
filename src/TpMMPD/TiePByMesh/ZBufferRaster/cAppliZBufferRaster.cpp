#include "ZBufferRaster.h"
/***********************************************************************************/
/*********************************** Constructor ***********************************/
/***********************************************************************************/
cAppliZBufferRaster::cAppliZBufferRaster(cInterfChantierNameManipulateur * aICNM,
                                            const std::string & aDir,
                                            const std::string & anOri,
                                            vector<cTri3D> & aVTri,
                                            vector<string> & aVImg,
                                            bool aNoTif,
                                            cParamZbufferRaster aParam
                                         ):
    mICNM (aICNM),
    mDir  (aDir),
    mOri  (anOri),
    mVTri (aVTri),
    mVImg (aVImg),
    mNInt (0),
    mW    (0),
    mSzW  (Pt2di(500,500)),
    mReech(1.0),
    mDistMax (TT_DISTMAX_NOLIMIT),
    mIsTmpZBufExist (ELISE_fp::IsDirectory(aDir + "Tmp-ZBuffer/")),
    mNoTif (aNoTif),
    mMethod (3),
    mAccNbImgVisible (aVTri.size(), Pt2di(0,0)),
    mvImgVisibleFarScene (aVImg.size() , false),
    mParam (aParam)
{
    if ( !mIsTmpZBufExist)
    {
        ELISE_fp::MkDir(mDir + "Tmp-ZBuffer/");
    }
}

/***********************************************************************************/
/*********************************** SetNameMesh ***********************************/
/***********************************************************************************/

void cAppliZBufferRaster::SetNameMesh(string & aNameMesh)
{
    mNameMesh = aNameMesh;
    ELISE_fp::MkDirSvp(mDir + "Tmp-ZBuffer/" + aNameMesh);
}


/***********************************************************************************/
/*********************************** DoAllIm ***************************************/
/***********************************************************************************/

void  cAppliZBufferRaster::DoAllIm(vector<vector<bool> > & aVTriValid)
{
 //cout<<"=======ZBUF : DistMax "<<this->DistMax()<<" Method "<<this->Method()<<" NInt "<<this->NInt()<<" Reech "<<this->Reech()<<" surf "<<this->SEUIL_SURF_TRIANGLE()<<endl;
    for (int aKIm=0; aKIm<int(mVImg.size()); aKIm++)
    {
        if (aKIm % 10 == 0 && mVImg.size() > 30)
             cout<<" ++ Im : ["<<(aKIm*100.0/mVImg.size())<<" %]"<<endl;
        string path = mDir + "Tmp-ZBuffer/" + mNameMesh + "/" + mVImg[aKIm] + "/";
        string fileOutZBuf = path + mVImg[aKIm] + "_ZBuffer_DeZoom" + ToString(int(1.0/mReech)) + ".tif";
        string fileOutLbl = path + mVImg[aKIm] + "_TriLabel_DeZoom" +  ToString(int(1.0/mReech)) + ".tif";
        ElTimer aChrono;
        cImgZBuffer * aZBuf = new cImgZBuffer(this, mVImg[aKIm], mNoTif, aKIm);
       if ( ELISE_fp::exist_file(fileOutLbl) && ELISE_fp::exist_file(fileOutZBuf))
       {
           cout<<mVImg[aKIm]<<" existed in Tmp-ZBuffer ! . Skip"<<endl;
           aZBuf->ImportResult(fileOutLbl, fileOutLbl);
           mIsTmpZBufExist = true;
       }
       else
       {
           for (int aKTri=0; aKTri<int(mVTri.size()); aKTri++)
           {
               if (aKTri % 200 == 0 && mNInt != 0)
                   cout<<"["<<(aKTri*100.0/mVTri.size())<<" %]"<<endl;
               aZBuf->LoadTri(mVTri[aKTri]);
           }
           //save Image ZBuffer to disk

           ELISE_fp::MkDirSvp(path);

           ELISE_COPY
                   (
                       aZBuf->ImZ().all_pts(),
                       aZBuf->ImZ().in_proj(),
                       Tiff_Im(
                           fileOutZBuf.c_str(),
                           aZBuf->ImZ().sz(),
                           GenIm::real8,
                           Tiff_Im::No_Compr,
                           Tiff_Im::BlackIsZero
                           ).out()

                       );

           if (mWithImgLabel)
           {
               ELISE_COPY
                   (
                       aZBuf->ImInd().all_pts(),
                       aZBuf->ImInd().in_proj(),
                       Tiff_Im(
                           fileOutLbl.c_str(),
                           aZBuf->ImInd().sz(),
                           GenIm::real8,
                           Tiff_Im::No_Compr,
                           Tiff_Im::BlackIsZero
                           //aZBuf->Tif().phot_interp()
                           ).out()

                       );
           }
       }
       aVTriValid.push_back(aZBuf->TriValid());
       if (mNInt != 0)
       {
           cout<<"Im "<<mVImg[aKIm]<<endl;
           cout<<"Finish Img Cont.. - Nb Tri Traiter : "<<aZBuf->CntTriTraite()<<" -Time: "<<aChrono.uval()<<endl;

           //=======================================
           aZBuf->normalizeIm(aZBuf->ImZ(), 0.0, 255.0);
           aZBuf->normalizeIm(aZBuf->ImInd(), 0.0, 255.0);

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
               if (mWithImgLabel)
               {
                   if (mWLbl ==0)
                   {
                       mWLbl = Video_Win::PtrWStd(mSzW, true, aZ);
                       mWLbl->set_sop(Elise_Set_Of_Palette::TheFullPalette());
                   }
               }
           }

           if (mW)
           {
               mW->set_title( (mVImg[aKIm] + "_ZBuf").c_str());
               ELISE_COPY(   aZBuf->ImZ().all_pts(),
                             aZBuf->ImZ().in(),
                             mW->ogray()
                             );
               mW->clik_in();
           }

           if (mWithImgLabel && mWLbl)
           {

               mWLbl->set_title( (mVImg[aKIm] + "_Label").c_str());
               ELISE_COPY(   aZBuf->ImInd().all_pts(),
                             aZBuf->ImInd().in(),
                             mWLbl->ogray()
                             );

           }
           getchar();
       }
       if (mWithImgLabel)
       {
           for (double i=0; i<aZBuf->TriValid().size(); i++)
           {
               if(aZBuf->TriValid()[i] == true)
                   aZBuf->CntTriValab()++;
           }
           cout<<mVImg[aKIm]<<" : "<<aZBuf->CntTriValab()<<" tri valid"<<endl;
       }
       delete aZBuf;
    }

}
/***********************************************************************************/
/*********************************** DoAllIm ***************************************/
/***********************************************************************************/

void cAppliZBufferRaster::DoAllIm()
{
    DoAllIm(mTriValid);
}

