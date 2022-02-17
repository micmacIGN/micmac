#include "ZBufferRaster.h"

cImgZBuffer::cImgZBuffer(cAppliZBufferRaster * anAppli , const std::string & aNameIm, bool & aNoTif, int aInd):

    mAppli    (anAppli),
    mNameIm   (aNameIm),
    mInd      (aInd),
    mTif      (Tiff_Im::UnivConvStd(mAppli->Dir() + aNameIm)),
    mSzIm     (mTif.sz()),
    mCamGen   (mAppli->ICNM()->StdCamGenerikOfNames(mAppli->Ori(),mNameIm)),
    mImZ      (round_ni(mSzIm.x*mAppli->Reech()), round_ni(mSzIm.y*mAppli->Reech()), tElZBuf(-1.0)),
    mTImZ     (mImZ),
    mImInd    (round_ni(mSzIm.x*mAppli->Reech()), round_ni(mSzIm.y*mAppli->Reech()), tElZBuf(-1.0)),
    mTImInd   (mImInd),
    mMasqTri  (1,1),
    mTMasqTri (mMasqTri),
    mMasqIm   (1,1),
    mTMasqIm  (mMasqIm),
    mW        (0),
    mCntTri   (0),
    mCntTriValab (0),
    mCntTriTraite (0)
{
    mTriValid.resize(mAppli->VTri().size(), false);    //initializer vector avec taille et valeur default
    if (mAppli->Reech() != 1.0)
    {
       mSzIm = mImZ.sz();
    }
}

void cImgZBuffer::updateZ(tImZBuf & ImZ, Pt2dr & pxl, double & prof_val, double & ind_val)
{
    Pt2di pxlI(pxl);
    if (ImZ.Inside(pxlI))
    {
        double prof_old = ImZ.GetR(pxlI);
        double ind_old = -1.0;
        if (mAppli->WithImgLabel())
            ind_old =  mImInd.GetR(pxlI);
        if (prof_old == TT_DEFAULT_PROF_NOVISIBLE)
        {
            ImZ.SetR_SVP(pxlI , prof_val);
            if (mAppli->WithImgLabel() && prof_val!=TT_DEFAULT_PROF_NOVISIBLE)
            {
                mImInd.SetR_SVP(pxlI, ind_val);
                mTriValid[ind_val] = true;
            }
            return;
        }
        else if (prof_old != TT_DEFAULT_PROF_NOVISIBLE && prof_old > prof_val)
        {
            ImZ.SetR_SVP(pxlI , prof_val);
            if (mAppli->WithImgLabel())
            {
                mImInd.SetR_SVP(pxlI, ind_val);
                mTriValid[ind_val] = true;
                mTriValid[ind_old] = false;
            }
            return;
        }
        else
            return;
    }
    else
    {
        return;
    }
}

void cImgZBuffer::LoadTri(cTri3D aTri3D)
{

    if (mAppli->DistMax() != TT_DISTMAX_NOLIMIT)
    {
        if (aTri3D.dist2Cam(mCamGen) > mAppli->DistMax())
        {
            return;
        }
    }
    cTri2D aTri = aTri3D.reprj(mCamGen);
    if (mAppli->Param().mInverseOrder)
    {
        aTri.InverseOrder() = true;
    }
    if (mAppli->Param().mFarScene)
    {
        if (
               aTri.IsInCam()
           )
        {
            mAppli->AccNbImgVisible()[int(aTri3D.Ind())].x = int(aTri3D.Ind());
            mAppli->AccNbImgVisible()[int(aTri3D.Ind())].y++;
            mAppli->vImgVisibleFarScene()[Ind()] = true;
        }
    }
    if (mAppli->Reech() != TT_SCALE_1)
    {
        //Reech coordonee dans aTri2D
        aTri.SetReech(mAppli->Reech());
    }

    if (
            aTri.IsInCam() &&
            -aTri.surf() > Appli()->SEUIL_SURF_TRIANGLE()
       )
    {
        if (mAppli->NInt() > 1 || this->Appli()->Method() == 2 || this->Appli()->Method() == 1)
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
        if (this->Appli()->Method() == 1)
        {
            vector<Pt2dr> aVPtsInTri;
            Flux2StdCont(aVPtsInTri , select(mImZ.all_pts(),mMasqTri.in()) );
            for (int aKPt=0; aKPt<(int)aVPtsInTri.size(); aKPt++)
            {
                Pt2dr aPtRas = aVPtsInTri[aKPt];
                double prof = aTri.profOfPixelInTri(aPtRas, aTri3D, mCamGen, Appli()->Param().mSafe);
                cImgZBuffer::updateZ(mImZ, aPtRas, prof, aTri3D.Ind());
            }
        }

        /*===method 2====*/
//        if (this->Appli()->Method() == 2)
//        {
//            for (int aKx=0; aKx<mMasqLocalTri.sz().x; aKx++)
//            {
//                for (int aKy=0; aKy<mMasqLocalTri.sz().y; aKy++)
//                {
//                    Pt2di aPt(aKx, aKy);
//                    if (mMasqLocalTri.GetI(aPt)!=0);
//                    {
//                        Pt2di aPtGlob(aPt+mDecal);
//                        if (mImZ.Inside(aPtGlob))
//                            aVPtsInTri.push_back(Pt2dr(aPtGlob));
//                    }
//                }
//            }
//            for (int aKPt=0; aKPt<aVPtsInTri.size(); aKPt++)
//            {
//                Pt2dr aPtRas = aVPtsInTri[aKPt];
//                double prof = aTri.profOfPixelInTri(aPtRas, aTri3D, mCamGen);
//                cImgZBuffer::updateZ(mImZ, aPtRas, prof, aTri3D.Ind());
//            }
//        }

        /*===method 3====*/
        std::vector<cSegEntierHor> aRasTri;
        if (this->Appli()->Method() == 3)
        {
        cElTriangleComp aElTri(aTri.P1(), aTri.P2(), aTri.P3());
        RasterTriangle(aElTri, aRasTri);
        for (uint aKSeg=0; aKSeg<aRasTri.size(); aKSeg++)
        {
            cSegEntierHor aSeg = aRasTri[aKSeg];
            for (int aKPt=0; aKPt<aSeg.mNb; aKPt++)
            {
                Pt2dr aPtRas(aSeg.mP0.x + aKPt, aSeg.mP0.y);
                double prof = aTri.profOfPixelInTri(aPtRas, aTri3D, mCamGen, Appli()->Param().mSafe);
                cImgZBuffer::updateZ(mImZ, aPtRas, prof, aTri3D.Ind());
            }
        }
        }


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
        //===========================//
        mCntTriTraite++;
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

/***********************************************************************************/
/*********************************** ImportResult***********************************/
/***********************************************************************************/

void cImgZBuffer::ImportResult(string & fileTriLbl, string & fileZBuf)
{
    ELISE_ASSERT(ELISE_fp::exist_file(fileTriLbl),"File Img Label not found");
    ELISE_ASSERT(ELISE_fp::exist_file(fileZBuf),"File Img ZBuf not found");
    Tiff_Im aImInd = Tiff_Im::StdConv(fileTriLbl);
    //Tiff_Im aImZBuf = Tiff_Im::StdConv(fileZBuf);
    ELISE_COPY(mImInd.all_pts(), aImInd.in(), mImInd.out());
    //ELISE_COPY(mImZ.all_pts(), aImZBuf.in(), mImZ.out());
    if (Appli()->Param().mFarScene)
    {
        cout<<"Far scene is computed by existed result in Tmp-ZBuffer"<<endl;
    }
    Pt2di aP;
    for (aP.x = 0; aP.x < mImInd.sz().x; aP.x++)
    {
        for (aP.y = 0; aP.y < mImInd.sz().y; aP.y++)
        {
            //double aIndTri = mImInd.GetR(aP);
            double aIndTri = mTImInd.get(aP);
            if (aIndTri  != tElZBuf(-1.0))
            {
               mTriValid[int(aIndTri)] = true;
               if (Appli()->Param().mFarScene)
               {
                    Appli()->AccNbImgVisible()[int(aIndTri)].x = int(aIndTri);
                    Appli()->AccNbImgVisible()[int(aIndTri)].y++;
               }
            }
        }
    }
}

