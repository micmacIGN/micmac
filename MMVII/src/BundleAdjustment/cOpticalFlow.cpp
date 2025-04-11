#include "cOpticalFlow.h"
#include "../ImagesBase/cGdalApi.h"
#include "MMVII_Error.h"

namespace MMVII
{

template <class TypeIm> cOpticalFlow<TypeIm>::cOpticalFlow(const cIm2D<TypeIm> & aIm1,
                                   const cIm2D<TypeIm> & aIm2, tImMasq & aImMasq1):
    mDispX(cPt2di(1,1)),
    mDDispX(nullptr),
    mDispY(cPt2di(1,1)),
    mDDispY(nullptr),
    mImGradX(cPt2di(1,1)),
    mDImGradX(nullptr),
    mImGradY(cPt2di(1,1)),
    mDImGradY(nullptr),
    mImGradt(cPt2di(1,1)),
    mDImGradt(nullptr),
    mDMasq1(nullptr),
    mInterp(nullptr)
{
    // Init disps and grads,
    // Initialize image gradients
    const cDataIm2D<TypeIm> * aDIm1 =&(aIm1.DIm());
    const cDataIm2D<TypeIm> * aDIm2 =&(aIm2.DIm());

    mDispX= tImDispl(aDIm1->Sz());
    mDispY= tImDispl(aDIm1->Sz());


    mImGradX= tImDispl(aDIm1->Sz());
    mImGradY= tImDispl(aDIm1->Sz());
    mImGradt= tImDispl(aDIm1->Sz());

    mDDispX=&(mDispX.DIm());
    mDDispX->InitCste(0);


    mDDispY=&(mDispY.DIm());
    mDDispY->InitCste(0);


    mDImGradX=&(mImGradX.DIm());
    mDImGradX->InitCste(0);

    mDImGradY=&(mImGradY.DIm());
    mDImGradY->InitCste(0);

    mDImGradt=&(mImGradt.DIm());
    mDImGradt->InitCste(0);

    mDMasq1=&(aImMasq1.DIm());

    StdOut()<<"Init images "<<std::endl;

    std::vector<std::string> aParamInt {"Scale","2","100","Cubic","-0.5"};//{"Tabul","1000","SinCApod","10","10"};
    mInterp = cDiffInterpolator1D::AllocFromNames(aParamInt);


    // compute gradients
    cPt2di aPix;
    for (aPix.x()=0;aPix.x()<aDIm1->SzX();aPix.x()++)
    {
        for (aPix.y()=0;aPix.y()<aDIm1->SzY();aPix.y()++)
        {
            if (mDMasq1->Inside(aPix))
            {
                if (mDMasq1->GetV(aPix))
                {
                    if (aDIm1->InsideInterpolator(*mInterp,ToR(aPix),1.0))
                    {
                        auto aVGr = aDIm1->GetValueAndGradInterpol(*mInterp,ToR(aPix));
                        mDImGradX->SetV(aPix,aVGr.second.x());
                        mDImGradY->SetV(aPix,aVGr.second.y());
                        mDImGradt->SetV(aPix,aDIm2->GetV(aPix)-aDIm1->GetV(aPix));
                    }
                }
            }
        }
    }

    // INIT MATRIX of OBS A
    tREAL8 aNumPix=mDDispX->SzX()*mDDispX->SzY();
    mmatA = SparseMatrix<tREAL8> (2*aNumPix,
                                 2*aNumPix);
    mmatB=SparseMatrix<tREAL8> (2*aNumPix,1);
}


template <class TypeIm> cOpticalFlow<TypeIm>::cOpticalFlow(const std::string & aNameIm1,
                                   const std::string & aNameIm2,
                                   const std::string & aNameMasq1):
    mDispX(cPt2di(1,1)),
    mDDispX(nullptr),
    mDispY(cPt2di(1,1)),
    mDDispY(nullptr),
    mImGradX(cPt2di(1,1)),
    mDImGradX(nullptr),
    mImGradY(cPt2di(1,1)),
    mDImGradY(nullptr),
    mImGradt(cPt2di(1,1)),
    mDImGradt(nullptr),
    mDMasq1(nullptr),
    mInterp(nullptr)
{
    // read images

    cIm2D<TypeIm> aIm1= cIm2D<TypeIm>::FromFile(aNameIm1);
    cIm2D<TypeIm> aIm2= cIm2D<TypeIm>::FromFile(aNameIm2);
    cIm2D<tU_INT1> aImMasq1= cIm2D<tU_INT1>::FromFile(aNameMasq1);


    StdOut()<<"  aIM1 "<<aIm1.DIm().Sz()<<" aIM2 "<<aIm2.DIm().Sz()<<std::endl;
    const cDataIm2D<TypeIm> * aDIm1 =&(aIm1.DIm());
    const cDataIm2D<TypeIm> * aDIm2 =&(aIm2.DIm());

    mDispX= tImDispl(aDIm1->Sz());
    mDispY= tImDispl(aDIm1->Sz());


    mImGradX= tImDispl(aDIm1->Sz());
    mImGradY= tImDispl(aDIm1->Sz());
    mImGradt= tImDispl(aDIm1->Sz());

    mDDispX=&(mDispX.DIm());
    mDDispX->InitCste(0);


    mDDispY=&(mDispY.DIm());
    mDDispY->InitCste(0);


    mDImGradX=&(mImGradX.DIm());
    mDImGradX->InitCste(0);

    mDImGradY=&(mImGradY.DIm());
    mDImGradY->InitCste(0);

    mDImGradt=&(mImGradt.DIm());
    mDImGradt->InitCste(0);

    mDMasq1=&(aImMasq1.DIm());


    StdOut()<<"Masq SIZE "<<mDMasq1->Sz()<<std::endl;


    std::vector<std::string> aParamInt {"Scale","2","100","Cubic","-0.5"};//{"Tabul","1000","SinCApod","10","10"};
    mInterp = cDiffInterpolator1D::AllocFromNames(aParamInt);


    // compute gradients
    /*cPt2di aPix;
    //cPt2di aTab[4]={cPt2di(0,-1),cPt2di(0,1), cPt2di(-1,0), cPt2di(1,0)};

    for (aPix.x()=0;aPix.x()<aDIm1->SzX();aPix.x()++)
    {
        for (aPix.y()=0;aPix.y()<aDIm1->SzY();aPix.y()++)
        {
            if (mDMasq1->Inside(aPix))
            {
                if (mDMasq1->GetV(aPix))
                {
                    if (aDIm1->InsideInterpolator(*mInterp,ToR(aPix),1.0))
                    {
                        //MMVII_INTERNAL_ASSERT_strong(aDIm2->InsideInterpolator(*mInterp,ToR(aPix),1.0),"IM2 NOT INSIDE INTERPO ! ");
                        auto aVGr = aDIm1->GetValueAndGradInterpol(*mInterp,ToR(aPix));
                        mDImGradX->SetV(aPix,aVGr.second.x());
                        mDImGradY->SetV(aPix,aVGr.second.y());
                        mDImGradt->SetV(aPix,aDIm2->GetV(aPix)-aDIm1->GetV(aPix));
                    }
                }
            }
        }
    }*/

    // compute gradients
    cPt2di aPix;
    //cPt2di aTab[4]={cPt2di(0,-1),cPt2di(0,1), cPt2di(-1,0), cPt2di(1,0)};
    tREAL8 aIX,aIY,aIT;
    int aCntx;
    int aCnty;
    int aCntt;
    for (aPix.x()=0;aPix.x()<aDIm1->SzX();aPix.x()++)
    {
        for (aPix.y()=0;aPix.y()<aDIm1->SzY();aPix.y()++)
        {
            aIX=0.0;
            aIY=0.0;
            aIT=0.0;
            aCntx=0;
            aCnty=0;
            aCntt=0;
            if (mDMasq1->Inside(aPix))
            {
                if (mDMasq1->GetV(aPix))
                {
                    if (aDIm1->InsideInterpolator(*mInterp,ToR(aPix),1.0))
                    {
                        aCntx+=2;
                        aCnty+=2;
                        aCntt++;
                        //MMVII_INTERNAL_ASSERT_strong(aDIm2->InsideInterpolator(*mInterp,ToR(aPix),1.0),"IM2 NOT INSIDE INTERPO ! ");
                        auto aVGr1 = aDIm1->GetValueAndGradInterpol(*mInterp,ToR(aPix));
                        auto aVGr2 = aDIm2->GetValueAndGradInterpol(*mInterp,ToR(aPix));
                        aIX+=aVGr1.second.x()+aVGr2.second.x();
                        aIY+=aVGr1.second.y()+aVGr2.second.y();
                        aIT+=aDIm2->GetV(aPix)-aDIm1->GetV(aPix);
                        if (aDIm1->InsideInterpolator(*mInterp,ToR(aPix+cPt2di(0,1)),1.0))
                        {
                            auto aVGr1y=aDIm1->GetValueAndGradInterpol(*mInterp,ToR(aPix+cPt2di(0,1)));
                            auto aVGr2y=aDIm2->GetValueAndGradInterpol(*mInterp,ToR(aPix+cPt2di(0,1)));
                            aCntx+=2;
                            aIX+=aVGr1y.second.x()+aVGr2y.second.x();
                            aIT+=aDIm2->GetV(aPix+cPt2di(0,1))-aDIm1->GetV(aPix+cPt2di(0,1));
                            aCntt++;
                        }
                        if (aDIm1->InsideInterpolator(*mInterp,ToR(aPix+cPt2di(1,0)),1.0))
                        {
                            auto aVGr1x=aDIm1->GetValueAndGradInterpol(*mInterp,ToR(aPix+cPt2di(1,0)));
                            auto aVGr2x=aDIm2->GetValueAndGradInterpol(*mInterp,ToR(aPix+cPt2di(1,0)));
                            aCnty+=2;
                            aIY+=aVGr1x.second.y()+aVGr2x.second.y();
                            aIT+=aDIm2->GetV(aPix+cPt2di(1,0))-aDIm1->GetV(aPix+cPt2di(1,0));
                            aCntt++;
                        }

                        if (aDIm1->InsideInterpolator(*mInterp,ToR(aPix+cPt2di(1,1)),1.0))
                        {
                            aIT+=aDIm2->GetV(aPix+cPt2di(1,1))-aDIm1->GetV(aPix+cPt2di(1,1));
                            aCntt++;
                        }

                        mDImGradX->SetV(aPix,aIX/aCntx);
                        mDImGradY->SetV(aPix,aIY/aCnty);
                        mDImGradt->SetV(aPix,aIT/aCntt);
                    }
                }
            }
        }
    }

    // INIT MATRIX of OBS A
    /*tREAL8 aNumPix=mDDispX->SzX()*mDDispX->SzY();
    mmatA = SparseMatrix<tREAL8> (2*aNumPix,
                                 2*aNumPix);
    mmatB=SparseMatrix<tREAL8> (2*aNumPix,1);*/
}

template <class TypeIm>void cOpticalFlow<TypeIm>::InitMat(bool isA)
{
    cPt2di aPix;
    tREAL8 aCXX,aCYY,aCXY, aCXT, aCYT;
    if (isA)
    {
        mmatA.setZero();
        std::vector<Eigen::Triplet<tREAL8>> CoeffsA;

        int aOffset=0;
        for (aPix.x()=0;aPix.x()<mDDispX->SzX();aPix.x()++)
        {
            for (aPix.y()=0;aPix.y()<mDDispX->SzY();aPix.y()++)
            {
                if (mDMasq1->Inside(aPix))
                {
                    if (mDMasq1->GetV(aPix))
                    {
                        aCXX=1+mLamda*pow(mDImGradX->GetV(aPix),2);
                        aCYY=1+mLamda*pow(mDImGradY->GetV(aPix),2);
                        aCXY=mLamda*mDImGradX->GetV(aPix)*mDImGradY->GetV(aPix);
                        // Coeffs A
                        //StdOut()<<2*aPix.y()+aOffset<<"  "<<mmatA.cols()<<std::endl;
                        MMVII_INTERNAL_ASSERT_strong(2*aPix.y()+aOffset<mmatA.cols(),"assertion cols out !");
                        CoeffsA.push_back(Triplet<tREAL8>(2*aPix.y()+aOffset,
                                                          2*aPix.y()+aOffset,
                                                          aCXX));
                        CoeffsA.push_back(Triplet<tREAL8>(2*aPix.y()+aOffset+1,
                                                          2*aPix.y()+aOffset,
                                                          aCXY));
                        CoeffsA.push_back(Triplet<tREAL8>(2*aPix.y()+aOffset,
                                                          2*aPix.y()+aOffset+1,
                                                          aCXY));
                        CoeffsA.push_back(Triplet<tREAL8>(2*aPix.y()+aOffset+1,
                                                          2*aPix.y()+aOffset+1,
                                                          aCYY));
                    }
                }
            }
            aOffset+=2*mDDispX->SzY();
        }

        mmatA.setFromTriplets(CoeffsA.begin(),CoeffsA.end());

        /*if (1)
        {
            // test matrix construction
            int aNumPix=mDDispX->SzX()*mDDispX->SzY();
            cPt2di aSzMat(aNumPix,aNumPix);
            cIm2D<tREAL8> aIMat=cIm2D<tREAL8>(aSzMat,
                                                nullptr,
                                            eModeInitImage::eMIA_Null);
            eTyNums aTypeF2 = tElemNumTrait<tREAL8>::TyNum();
            cDataFileIm2D  aFileIm = cDataFileIm2D::Create("./MatrixSample.tif",
                                                          aTypeF2,aSzMat,1);

            cPt2di aPix;
            for (aPix.x()=0; aPix.x()<aSzMat.x();aPix.x()++)
            {
                for(aPix.y()=0; aPix.y()<aSzMat.y();aPix.y()++)
                {
                    aIMat.DIm().SetV(aPix,mmatA.coeff(aPix.y(),aPix.x()));
                }
            }
            aIMat.Write(aFileIm,cPt2di(0,0));
        }*/
    }
    else
    {
        cPt2di aTab[4]={cPt2di(0,-1),cPt2di(0,1), cPt2di(-1,0), cPt2di(1,0)};
        mmatB.setZero();
        std::vector<Eigen::Triplet<tREAL8>> CoeffsB;
        int aOffset=0;
        for (aPix.x()=0;aPix.x()<mDDispX->SzX();aPix.x()++)
        {
            for (aPix.y()=0;aPix.y()<mDDispX->SzY();aPix.y()++)
            {
                for (auto aP : aTab)
                {
                    tREAL8 anAvgX=0.0;
                    tREAL8 anAvgY=0.0;
                    int aCnt=0;
                    if (mDMasq1->Inside(aPix+aP))
                    {
                        if (mDMasq1->GetV(aPix+aP)) // aPixOffset (0,-1) (0,1) (-1,0) (1,0)
                        {
                            anAvgX+=mDDispX->GetV(aPix+aP);
                            anAvgY+=mDDispY->GetV(aPix+aP);
                            aCnt++;
                        }
                            MMVII_INTERNAL_ASSERT_strong(mDDispY->Inside(aPix),"PIX NOT INSIDE IMAGE ");
                            aCXT=(anAvgX/aCnt)-mLamda*mDImGradX->GetV(aPix)*mDImGradt->GetV(aPix);
                            aCYT=(anAvgY/aCnt)-mLamda*mDImGradY->GetV(aPix)*mDImGradt->GetV(aPix);
                            // Coeffs B
                            CoeffsB.push_back(Triplet<tREAL8>(2*aPix.y()+aOffset,0,aCXT));
                            CoeffsB.push_back(Triplet<tREAL8>(2*aPix.y()+aOffset+1,0,aCYT));
                    }

                }
            }
        aOffset+=2*mDDispX->SzY();

        }

        mmatB.setFromTriplets(CoeffsB.begin(),CoeffsB.end());
    }
}


template <class TypeIm> std::pair<tREAL8,tREAL8> cOpticalFlow<TypeIm>::diff(tDImDispl & anActualDisplX,tDImDispl & anActualDisplY)
{
    tREAL8 diff_dispx=0.0;
    tREAL8 diff_dispy=0.0;

    cPt2di aPix;
    int tNBPixValid=0;
    for (aPix.x()=0;aPix.x()<mDDispX->SzX();aPix.x()++)
    {
        for (aPix.y()=0;aPix.y()<mDDispX->SzY();aPix.y()++)
        {
            if (mDMasq1->GetV(aPix))
            {
                diff_dispx+=abs(anActualDisplX.GetV(aPix)-mDDispX->GetV(aPix));
                diff_dispy+=abs(anActualDisplY.GetV(aPix)-mDDispY->GetV(aPix));
                tNBPixValid++;
            }
        }
    }
    return std::make_pair(diff_dispx/tNBPixValid,diff_dispy/tNBPixValid);
}


template <class TypeIm> void cOpticalFlow<TypeIm>::udpateDispl(tDImDispl & anActDispX,
                                       tDImDispl & anActDispY,
                                       SparseMatrix<tREAL8> & aSol)
{
    anActDispX.InitCste(0.0);
    anActDispY.InitCste(0.0);
    // update displacements maps based on computed solution
    cPt2di aPix;
    int aOffset=0;
    for (aPix.x()=0;aPix.x()<mDDispX->SzX();aPix.x()++)
    {
        for (aPix.y()=0;aPix.y()<mDDispX->SzY();aPix.y()++)
        {
            if (mDMasq1->GetV(aPix))
            {
                anActDispX.SetV(aPix,aSol.coeff(2*aPix.y()+aOffset,0));
                anActDispY.SetV(aPix,aSol.coeff(2*aPix.y()+aOffset+1,0));
            }
        }
        aOffset+=2*mDDispX->SzY();
    }
}


template <class TypeIm> void cOpticalFlow<TypeIm>::udpateDisplDirect(tDImDispl & anActDispX,
                                       tDImDispl & anActDispY)
{
    anActDispX.InitCste(0.0);
    anActDispY.InitCste(0.0);
    // update displacements maps based on computed solution
    cPt2di aPix;
    tREAL8 aNewX,aNewY;
    tREAL8 aGX,aGY,aGt,aX_ba,aY_ba,aRatio;
    for (aPix.x()=0;aPix.x()<mDDispX->SzX();aPix.x()++)
    {
        for (aPix.y()=0;aPix.y()<mDDispX->SzY();aPix.y()++)
        {
            if(mDMasq1->Inside(aPix))
            {
                if (mDMasq1->GetV(aPix))
                {
                    aGX=mDImGradX->GetV(aPix);
                    aGY=mDImGradY->GetV(aPix);
                    aGt=mDImGradt->GetV(aPix);
                    aX_ba=mDDispX->GetV(aPix);
                    aY_ba=mDDispY->GetV(aPix);
                    aRatio =(aGX*aX_ba+aGY*aY_ba+aGt)/((1/mLamda)+pow(aGX,2)+pow(aGY,2));
                    aNewX  = aX_ba - aGX*aRatio;
                    aNewY  = aY_ba - aGY*aRatio;
                    anActDispX.SetV(aPix,aNewX);
                    anActDispY.SetV(aPix,aNewY);
                }
            }
        }
    }
}


template <class TypeIm> void cOpticalFlow<TypeIm>::refreshDisp(tDImDispl & anActDispX,
                                       tDImDispl & anActDispY)
{
    cPt2di aPix;
    std::pair<cPt2di,tREAL8> aTab[8]={{cPt2di(0,-1),1/6},{cPt2di(0,1),1/6}, {cPt2di(-1,0),1/6}, {cPt2di(1,0),1/6},
                                          {cPt2di(-1,-1),1/12},{cPt2di(-1,1),1/12}, {cPt2di(1,-1),1/12}, {cPt2di(1,1),1/12}};
    tREAL8 anAvgX;
    tREAL8 anAvgY;
    mDDispX->InitCste(0.0);
    mDDispY->InitCste(0.0);
    for (aPix.x()=0;aPix.x()<mDDispX->SzX();aPix.x()++)
    {
        for (aPix.y()=0;aPix.y()<mDDispX->SzY();aPix.y()++)
        {
            anAvgX=0.0;
            anAvgY=0.0;
            for (auto aP : aTab)
            {
                if (mDMasq1->Inside(aPix+aP.first))
                {
                    if (mDMasq1->GetV(aPix+aP.first))
                    {
                        anAvgX+=aP.second*anActDispX.GetV(aPix+aP.first);
                        anAvgY+=aP.second*anActDispY.GetV(aPix+aP.first);
                    }
                    //MMVII_INTERNAL_ASSERT_strong(mDDispY->Inside(aPix),"PIX NOT INSIDE IMAGE ");
                    mDDispX->SetV(aPix,anAvgX);
                    mDDispX->SetV(aPix,anAvgY);
                }

            }
        }
    }
}

template <class TypeIm> void cOpticalFlow<TypeIm>::SolveDisp(std::string & aNameOut)
{
    tDImDispl anActDisX= tDImDispl(mDDispX->P0(),mDDispX->P1());
    tDImDispl anActDisY= tDImDispl(mDDispY->P0(),mDDispY->P1());

    StdOut()<<" actual DisX DisY  "<<anActDisX.Sz()<<"   "<<anActDisY.Sz()<<std::endl;

    InitMat(true);
    InitMat(false);
    // Solve
    //SparseLU<SparseMatrix<tREAL8>> solver;
    SimplicialCholesky<SparseMatrix<tREAL8>> cholesky(mmatA);
    //solver.analyzePattern(mmatA);
    //solver.factorize(mmatA);
    //solver.compute(mmatA);
    SparseMatrix<tREAL8> aSol;
    tREAL8 dx=1.0;
    tREAL8 dy=1.0;
    int NbIter=1000;

    while ((dx>1e-5) && (dy>1e-5) && NbIter)
    {
        aSol=cholesky.solve(mmatB);
        // fill Actual displacements
        // anActualDisX, anActualDisY from aSol
        udpateDispl(anActDisX,anActDisY,aSol);
        // Recompute mmatB
        auto Dxy=diff(anActDisX,anActDisY);
        dx=Dxy.first;
        dy=Dxy.second;
        StdOut()<<" avant  "<<mDDispX->Sz()<<"   "<<mDDispY->Sz()<<std::endl;

        // update displacements
        mDDispX->DupIn(anActDisX);
        mDDispY->DupIn(anActDisY);

        StdOut()<<" apres  "<<mDDispX->Sz()<<"   "<<mDDispY->Sz()<<std::endl;

        // update mmatB
        InitMat(false);
        NbIter--;
        StdOut()<<"DX__ "<<dx<<" DY__ "<<dy<<std::endl;
    }

    StdOut()<<" apres  "<<mDDispX->Sz()<<"   "<<mDDispY->Sz()<<std::endl;
    cDataFileIm2D aDF=cDataFileIm2D::Create(aNameOut,
                                              eTyNums::eTN_REAL8,
                                              mDDispX->Sz());

    // add geotiff transform
    std::vector<const cDataIm2D<tREAL8>*> aVIms({mDDispX});

    cGdalApi::ReadWrite(cGdalApi::IoMode::Write,
                        aVIms,
                        aDF,
                        cPt2di(0,0),
                        1.0,
                        cPixBox<2>(mDDispX->P0(),mDDispX->P1()));
}




template <class TypeIm> void cOpticalFlow<TypeIm>::SolveDispDirect(std::string & aNameOut, tREAL8 * TRANSFORM)
{
     ///<  U_{kl}= U_{kl}0 -....

     tDImDispl anActDisX= tDImDispl(mDDispX->P0(),mDDispX->P1());
     tDImDispl anActDisY= tDImDispl(mDDispY->P0(),mDDispY->P1());
     anActDisX.InitCste(0.0);
     anActDisY.InitCste(0.0);

     tREAL8 dx=1.0;
     tREAL8 dy=1.0;
     int NbIter=1000;
     cPt2di aPix;
     while ((dx>1e-5) && (dy>1e-5) && NbIter)
        {

         // actualiser deplacement
         udpateDisplDirect(anActDisX,anActDisY);

         auto Dxy=diff(anActDisX,anActDisY);
         dx=Dxy.first;
         dy=Dxy.second;

         //StdOut()<<"DX  DY  "<<dx<<"   "<<dy<<std::endl;

         // refresh saved displacements
         refreshDisp(anActDisX,anActDisY);
        NbIter--;
        }

    std::string aNameDX=aNameOut+"_x.tif";
    std::string aNameDY=aNameOut+"_y.tif";

     cDataFileIm2D aDFX=cDataFileIm2D::Create(aNameDX,
                                               eTyNums::eTN_REAL8,
                                               anActDisX.Sz());
    cDataFileIm2D aDFY=cDataFileIm2D::Create(aNameDY,
                                               eTyNums::eTN_REAL8,
                                               anActDisY.Sz());
     // add geotiff transform
     std::vector<const cDataIm2D<tREAL8>*> aVImXs({&anActDisX});
     std::vector<const cDataIm2D<tREAL8>*> aVImYs({&anActDisY});

     cGdalApi::ReadWrite(cGdalApi::IoMode::Write,
                         aVImXs,
                         aDFX,
                         cPt2di(0,0),
                         1.0,
                         cPixBox<2>(anActDisX.P0(),anActDisX.P1()),
                         TRANSFORM);

     cGdalApi::ReadWrite(cGdalApi::IoMode::Write,
                         aVImYs,
                         aDFY,
                         cPt2di(0,0),
                         1.0,
                         cPixBox<2>(anActDisY.P0(),anActDisY.P1()),
                         TRANSFORM);

     //StdOut()<<"FilWrite "<<std::endl;
}



template <class TypeIm> void cOpticalFlow<TypeIm>::saveFlow(std::string & aNameOut, tREAL8 * transform)
{
    cDataFileIm2D aDF=cDataFileIm2D::Create(aNameOut,
                                              eTyNums::eTN_REAL8,
                                              mDDispX->Sz());

    // add geotiff transform
    std::vector<const cDataIm2D<tREAL8>*> aVIms({mDDispX});

    cGdalApi::ReadWrite(cGdalApi::IoMode::Write,
                        aVIms,
                        aDF,
                        cPt2di(0,0),
                        1.0,
                        cPixBox<2>(mDDispX->P0(),mDDispX->P1()));
}


template <class TypeIm> cOpticalFlow<TypeIm>::~cOpticalFlow()
{
    delete mInterp;
    //delete mDDispX;
    //delete mDDispY;
    //delete mDImGradX;
    //delete mDImGradY;
    //delete mDImGradt;
}



/* ========================== */
/*     INSTANTIATION          */
/* ========================== */

#define INSTANTIATE_FLOW(TYPE)\
template class cOpticalFlow<TYPE>;

INSTANTIATE_FLOW(tU_INT1);
INSTANTIATE_FLOW(tU_INT2);
INSTANTIATE_FLOW(tREAL4);
INSTANTIATE_FLOW(tREAL8);

};
