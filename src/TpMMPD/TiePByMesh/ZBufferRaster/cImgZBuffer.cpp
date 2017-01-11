#include "ZBufferRaster.h"

cImgZBuffer::cImgZBuffer(cAppliZBufferRaster * anAppli ,const std::string & aNameIm):

    mAppli    (anAppli),

    mNameIm   (aNameIm),
    mTif      (Tiff_Im::StdConv(mAppli->Dir() + mNameIm)),
    mSzIm     (mTif.sz()),
    mCam      (mAppli->ICNM()->StdCamOfNames(aNameIm,mAppli->Ori())),
    mImZ      (round_ni(mSzIm.x*mAppli->Reech()), round_ni(mSzIm.y*mAppli->Reech()), tElZBuf(-1.0)),
    mTImZ     (mImZ),
    mMasqTri  (1,1),
    mTMasqTri (mMasqTri),
    mMasqIm   (1,1),
    mTMasqIm  (mMasqIm),
    mW        (0),
    mCntTri   (0),
    mCntTriValab (0)
{
    cout<<"Dans constructor cImgZBuffer : "<<mImZ.sz()<<mSzIm<<endl;
    if (mAppli->Reech() != 1.0)
    {
       mSzIm = mImZ.sz();
    }
}

bool cImgZBuffer::updateZ(tImZBuf & ImZ, Pt2dr & pxl, double & prof_val)
{
    Pt2di pxlI(pxl);
    if (ImZ.Inside(pxlI))
    {
        double prof_old = ImZ.GetR(pxlI);

        if (prof_old == TT_DEFAULT_PROF_NOVISIBLE)
        {
            ImZ.SetR_SVP(pxlI , prof_val);
            return true;
        }
        else if (prof_old != TT_DEFAULT_PROF_NOVISIBLE && prof_old > prof_val)
        {
            ImZ.SetR_SVP(pxlI , prof_val);
            return true;
        }
        else
            return false;
    }
    else
    {
        return false;
    }
}

void cImgZBuffer::LoadTri(cTri3D aTri3D)
{
    if (mAppli->DistMax() != TT_DISTMAX_NOLIMIT)
    {
        if (aTri3D.dist2Cam(mCam) > mAppli->DistMax())
        {
            return;
        }
    }
    cTri2D aTri = aTri3D.reprj(mCam);
    if (mAppli->Reech() != TT_SCALE_1)
    {
        //Reech coordonee dans aTri2D
        aTri.SetReech(mAppli->Reech());
    }

    if (
            aTri.IsInCam() &&
            -aTri.surf() > TT_SEUIL_SURF
       )
    {
        if (mAppli->NInt() > 1)
        {
            //creat masq rectangle local autour triangle
            Pt2dr aPMin = Inf(Inf(aTri.P1(),aTri.P2()),aTri.P3());
            Pt2dr aPMax = Sup(Sup(aTri.P1(),aTri.P2()),aTri.P3());

            Pt2di mDecal = round_down(aPMin);
            Pt2di mSzRec  = round_up(aPMax-aPMin);

            //creat masque local for triangle zone on rectangle (255=triangle)
            Im2D_Bits<1> mMasqLocalTri(mSzRec.x,mSzRec.y,0);
            ElList<Pt2di>  aLTri;
            aLTri = aLTri + round_ni(aTri.P1()-Pt2dr(mDecal));
            aLTri = aLTri + round_ni(aTri.P2()-Pt2dr(mDecal));
            aLTri = aLTri + round_ni(aTri.P3()-Pt2dr(mDecal));
            ELISE_COPY(polygone(aLTri),1,mMasqLocalTri.oclip());

            //creat masque global for triangle zone on image (255=triangle)
            mMasqTri =  Im2D_Bits<1>(mSzIm.x,mSzIm.y,0);
            mTMasqTri = TIm2DBits<1> (mMasqTri);
            ElList<Pt2di>  aGlobTri;
            aGlobTri = aGlobTri + round_ni(aTri.P1());
            aGlobTri = aGlobTri + round_ni(aTri.P2());
            aGlobTri = aGlobTri + round_ni(aTri.P3());
            ELISE_COPY(polygone(aGlobTri), 1, mMasqTri.oclip());
        }

        //grab coordinate all pixel in triangle
        /*===method 1====*/
        //vector<Pt2dr> aVPtsInTri;
        //Flux2StdCont(aVPtsInTri , select(mImZ.all_pts(),mMasqTri.in()) );

        /*===method 2====*/
        /*
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
        */

        /*===method 3====*/

        cElTriangleComp aElTri(aTri.P1(), aTri.P2(), aTri.P3());
        std::vector<cSegEntierHor> aRasTri;
        RasterTriangle(aElTri, aRasTri);
        for (uint aKSeg=0; aKSeg<aRasTri.size(); aKSeg++)
        {
            cSegEntierHor aSeg = aRasTri[aKSeg];
            for (uint aKPt=0; aKPt<aSeg.mNb; aKPt++)
            {
                Pt2dr aPtRas(aSeg.mP0.x + aKPt, aSeg.mP0.y);
                double prof = aTri.profOfPixelInTri(aPtRas, aTri3D, mCam);
                bool isUpdate = updateZ(mImZ, aPtRas, prof);
            }
        }
        mCntTriValab++;

        //Display masq Triangle global
        if (mAppli->NInt() > 1)
        {
            if (mW ==0)
            {
                double aZ = 0.5;
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
                for (uint aKSeg=0; aKSeg<aRasTri.size(); aKSeg++)
                {
                    cSegEntierHor aSeg = aRasTri[aKSeg];
                    mW->draw_seg(Pt2dr(aSeg.mP0), Pt2dr(aSeg.mP0.x + aSeg.mNb, aSeg.mP0.y), mW->pdisc()(P8COL::green));
                }
                mW->clik_in();
            }
        }
    }
    mCntTri++;
}

void cImgZBuffer::normalizeIm(tImZBuf & aImZ, double valMin, double valMax)
{
    double minProf;
    double maxProf;

    aImZ.getMinMax(minProf, maxProf);
    cout<<"Min Max Prof : "<<minProf<<" "<<maxProf;
    aImZ.substract(minProf - valMin);
    aImZ.getMinMax(minProf, maxProf);
    aImZ.multiply(valMax/maxProf);


    aImZ.getMinMax(minProf, maxProf);
    cout<<" -> Norm "<<minProf<<" "<<maxProf<<endl;
}

