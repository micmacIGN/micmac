#include "ZBufferRaster.h"

cImgZBuffer::cImgZBuffer(cAppliZBufferRaster * anAppli ,const std::string & aNameIm):

    mAppli    (anAppli),

    mNameIm   (aNameIm),
    mTif      (Tiff_Im::StdConv(mAppli->Dir() + mNameIm)),
    mCam      (mAppli->ICNM()->StdCamOfNames(aNameIm,mAppli->Ori())),
    mSzIm     (mTif.sz()),
    mImZ      (mSzIm.x, mSzIm.y, tElZBuf(0.0)),
    mTImZ     (mImZ),
    mMasqTri  (1,1),
    mTMasqTri (mMasqTri),
    mMasqIm   (1,1),
    mTMasqIm  (mMasqIm),
    mW        (0),
    mCntTri   (0)


{
    cout<<"Dans constructor cImgZBuffer"<<endl;
}

void cImgZBuffer::LoadTri(cTri3D aTri3D)
{
    cTri2D aTri = aTri3D.reprj(mCam);
    if (aTri.IsInCam())
    {
        //creat image Im2D from ImTif
        ELISE_COPY(mImZ.all_pts(),mTif.in(),mImZ.out());

        //creat masque for triangle zone on image (255=triangle)
        mMasqTri =  Im2D_Bits<1>(mSzIm.x,mSzIm.y,0);
        mTMasqTri = TIm2DBits<1> (mMasqTri);
        ElList<Pt2di>  aLSmTri;
        aLSmTri = aLSmTri + round_ni(aTri.P1());
        aLSmTri = aLSmTri + round_ni(aTri.P2());
        aLSmTri = aLSmTri + round_ni(aTri.P3());
        ELISE_COPY(polygone(aLSmTri), 1, mMasqTri.oclip());

        //grab coordinate all pixel in triangle
        vector<Pt2dr> aVPtsInTri;
        Flux2StdCont(aVPtsInTri , select(mImZ.all_pts(),mMasqTri.in()) );
        cout<<"Nb Pts In Flux :"<<aVPtsInTri.size()<<endl;



/*
        if (mW ==0)
        {
             double aZ = 0.25;
             mW = Video_Win::PtrWStd(Pt2di(mSzIm*aZ), true, Pt2dr(aZ, aZ));
             mW->set_sop(Elise_Set_Of_Palette::TheFullPalette());
        }

        if (mW)
        {
             mW->set_title(ToString(mCntTri).c_str());
             //ELISE_COPY(mImZ.all_pts(),mImZ.in(),mW->ogray());
             ELISE_COPY(   select(mImZ.all_pts(),mMasqTri.in()),
                           255,
                           mW->ogray()
                       );
             mW->clik_in();

        }
*/
        //Creer masque polygon
        //prendre tout les pixels dans tri
        //labeliser avec Z
    }
    mCntTri++;
}

