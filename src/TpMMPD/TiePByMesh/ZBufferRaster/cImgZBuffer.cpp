#include "ZBufferRaster.h"

cImgZBuffer::cImgZBuffer(cAppliZBufferRaster * anAppli ,const std::string & aNameIm):

    mAppli    (anAppli),

    mNameIm   (aNameIm),
    mTif      (Tiff_Im::StdConv(mAppli->Dir() + mNameIm)),
    mCam      (mAppli->ICNM()->StdCamOfNames(aNameIm,mAppli->Ori())),
    mSzIm     (mTif.sz()),
    mImZ      (mSzIm.x, mSzIm.y, tElZBuf(-1.0)),
    mTImZ     (mImZ),
    mMasqTri  (1,1),
    mTMasqTri (mMasqTri),
    mMasqIm   (1,1),
    mTMasqIm  (mMasqIm),
    mW        (0),
    mCntTri   (0),
    mCntTriValab (0)
{
    cout<<"Dans constructor cImgZBuffer"<<endl;
}

bool cImgZBuffer::updateZ(tImZBuf & ImZ, Pt2dr & pxl, double & prof_val)
{
    Pt2di pxlI(pxl);
/*
    if (ImZ.GetI(pxlI) > prof_val)
    {
        ImZ.SetR(pxlI , prof_val);
        return true;
    }
    else
        return false;
*/

    ImZ.SetR(pxlI , prof_val);
    return true;
}

void cImgZBuffer::LoadTri(cTri3D aTri3D)
{
    cTri2D aTri = aTri3D.reprj(mCam);
    if (
            aTri.IsInCam() &&
            //aTri.orientToCam(mCam) &&
            -aTri.surf() > TT_SEUIL_SURF
       )
    {
        //creat masq rectangle local autour triangle
        Pt2dr aPMin = Inf(Inf(aTri.P1(),aTri.P2()),aTri.P3());
        Pt2dr aPMax = Sup(Sup(aTri.P1(),aTri.P2()),aTri.P3());

        Pt2di mDecal = round_down(aPMin);
        Pt2di mSzRec  = round_up(aPMax-aPMin);


        //creat masque for triangle zone on image (255=triangle)
        mMasqTri =  Im2D_Bits<1>(mSzIm.x,mSzIm.y,0);
        mTMasqTri = TIm2DBits<1> (mMasqTri);
        ElList<Pt2di>  aGlobTri;
        aGlobTri = aGlobTri + round_ni(aTri.P1());
        aGlobTri = aGlobTri + round_ni(aTri.P2());
        aGlobTri = aGlobTri + round_ni(aTri.P3());
        ELISE_COPY(polygone(aGlobTri), 1, mMasqTri.oclip());


        //creat masque local for triangle zone on rectangle (255=triangle)
        Im2D_Bits<1> mMasqLocalTri(mSzRec.x,mSzRec.y,0);
        ElList<Pt2di>  aLTri;
        aLTri = aLTri + round_ni(aTri.P1()-Pt2dr(mDecal));
        aLTri = aLTri + round_ni(aTri.P2()-Pt2dr(mDecal));
        aLTri = aLTri + round_ni(aTri.P3()-Pt2dr(mDecal));
        ELISE_COPY(polygone(aLTri),1,mMasqLocalTri.oclip());

        //grab coordinate all pixel in triangle
        vector<Pt2dr> aVPtsInTri;
        //Flux2StdCont(aVPtsInTri , select(mImZ.all_pts(),mMasqTri.in()) );
        for (int aKx=0; aKx<mMasqLocalTri.sz().x; aKx++)
        {
            for (int aKy=0; aKy<mMasqLocalTri.sz().y; aKy++)
            {
                Pt2di aPt(aKx, aKy);
                if (mMasqLocalTri.GetI(aPt)!=0);
                {
                    Pt2di aPtGlob(aPt+mDecal);
                    if (mImZ.Inside(aPtGlob))
                        aVPtsInTri.push_back(Pt2dr(aPtGlob));
                }
            }
        }


        //update ZBuffer
        for (uint aKPxl=0; aKPxl<aVPtsInTri.size(); aKPxl++)
        {
            Pt2dr pxlInTri = aVPtsInTri[aKPxl];
            double prof = aTri.profOfPixelInTri(pxlInTri, aTri3D, mCam);
            bool isUpdate = updateZ(mImZ, pxlInTri, prof);
        }
        mCntTriValab++;


        if (mAppli->NInt() > 1)
        {
            if (mW ==0)
            {
                double aZ = 0.25;
                mW = Video_Win::PtrWStd(Pt2di(mSzIm*aZ), true, Pt2dr(aZ, aZ));
                mW->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            }

            if (mW)
            {
                mW->set_title(ToString(mCntTri).c_str());
                ELISE_COPY(   select(mImZ.all_pts(),mMasqTri.in()),
                              255,
                              mW->ogray()
                              );
                //mW->clik_in();
            }
        }
    }
    mCntTri++;
}

void cImgZBuffer::normalizeIm(tImZBuf & aImZ, double valMin, double valMax)
{
    double minProf;
    double maxProf;
    aImZ.substract(-1.0);
    aImZ.getMinMax(minProf, maxProf);
    cout<<"Min Max Prof : "<<minProf<<" "<<maxProf;
    aImZ.multiply(255.0/maxProf);
    aImZ.getMinMax(minProf, maxProf);
    cout<<" - Norm "<<minProf<<" "<<maxProf<<endl;



}

