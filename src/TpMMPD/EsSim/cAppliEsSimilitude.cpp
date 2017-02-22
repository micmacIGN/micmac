#include "EsSimilitude.h"

//==================== Si je mets ca dans .h, ca va pas content parce que multiple define ================
cParamEsSim::cParamEsSim(string & aDir, string & aImgX, string & aImgY, Pt2dr & aPtCtr, int & aSzW, Pt3di & aDispParam, int & nInt, Pt2di & aNbGrill, double & aSclDepl, bool & aSaveImg):
    mDir (aDir),
    mImgX (aImgX),
    mImgY (aImgY),
    mPtCtr (aPtCtr),
    mSzW (aSzW),
    mDispSz (Pt2di(aDispParam.x, aDispParam.y)),
    mZoom  (aDispParam.z),
    mInt   (nInt),
    mNbGrill (aNbGrill),
    mSclDepl (aSclDepl),
    mSaveImg (aSaveImg)
{}
//==========================================

cAppliEsSim::cAppliEsSim(cParamEsSim * aParam):
    mParam (aParam),
    mWX (0),
    mWY (0),
    aData1   (NULL)
{
    mImgX = new cImgEsSim(mParam->mImgX, this);
    mImgY = new cImgEsSim(mParam->mImgY, this);
    //calcul grill
    Pt2di aSzImg = mImgX->Tif().sz();
    Pt2di aStepGrill( int(aSzImg.x/mParam->mNbGrill.x), int(aSzImg.y/mParam->mNbGrill.y));
    for (int aStepX=0; aStepX<aSzImg.x; aStepX=aStepX+aStepGrill.x)
    {
        for (int aStepY=0; aStepY<aSzImg.y; aStepY=aStepY+aStepGrill.y)
        {
            Pt2di aPtCtr = Pt2di( (aStepX+aStepGrill.x + aStepX)/2, (aStepY+aStepGrill.y + aStepY)/2);
            mVaP0Grill.push_back(aPtCtr);
        }
    }
    cout<<"Nb Vignette : "<<mVaP0Grill.size()<<endl;
}

/*===================cAppliEsSim::creatHomol====================*/
/*========create pairs of tie points for all the pixels=========*/

void cAppliEsSim::creatHomol(cImgEsSim * aImgX, cImgEsSim * aImgY)
{
    Pt2di aSz = aImgX->ImgDep().sz(); //get the size of images
    Pt2di aK(0,0);
    for (aK.x=0; aK.x<aSz.x; aK.x++)
    {
        for (aK.y=0; aK.y<aSz.y; aK.y++)
        {
            Pt2dr aPt = Pt2dr(aK);
            ElCplePtsHomologues aCpl(
                                     aPt,
                                     aPt + Pt2dr(aImgX->ImgDep().GetR(aK), aImgY->ImgDep().GetR(aK))
                                    ); // get pairs of tie points for all the pixels
            mHomolDep.Cple_Add(aCpl);
        }
    }
}

/*================cAppliEsSim::writeHomol=================*/
/*=========write pairs of tiep oints in a pack============*/

void cAppliEsSim::writeHomol(ElPackHomologue & aPack)
{
    string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string("_Depl")
                        +  std::string("@")
                        +  std::string("dat");
    cInterfChantierNameManipulateur * mICNM = cInterfChantierNameManipulateur::BasicAlloc(mParam->mDir);
    string cleNomHomolOut = mICNM->Assoc1To2(aKHOutDat, mImgX->Name(), mImgY->Name(), true);
    cout<<"Save Homol : "<<cleNomHomolOut<<endl;
    aPack.StdPutInFile(cleNomHomolOut);
}

/*=================cAppliEsSim::getHomolInVgt================*/
/*get tie points of the vignette and indicate if the vignette is inside the original image*/

bool cAppliEsSim::getHomolInVgt (ElPackHomologue & aPack, Pt2dr & aPtCtr, int & aSzw)
{
    Pt2dr aP0(aPtCtr.x-aSzw,aPtCtr.y-aSzw); //up-left point of the vignette
    Pt2dr aP1(aPtCtr.x+aSzw,aPtCtr.y+aSzw); //down-right point of the vignette
    if (mImgX->IsInside(aP0) && mImgX->IsInside(aP1)) //check if the vignette is in the original image
    {
        Pt2di aK(0,0);
        for (aK.x = aP0.x; aK.x<=aP1.x; aK.x++)
        {
            for (aK.y = aP0.y; aK.y<=aP1.y; aK.y++)
            {
               Pt2dr aPt = Pt2dr(aK);
               ElCplePtsHomologues aCpl(
                                        aPt,
                                        aPt + Pt2dr(mImgX->ImgDep().GetR(aK), mImgY->ImgDep().GetR(aK))
                                       ); // get pairs of tie points of the vignette
               aPack.Cple_Add(aCpl);
            }
        }
        return true;
    }
    return false;
}

/*=================cAppliEsSim::EsSimFromHomolPack=================*/
/*======estimate the rotation and translation from deplacement======*/
bool cAppliEsSim::EsSimFromHomolPack (ElPackHomologue & aPack, Pt2dr & rotCosSin, Pt2dr & transXY)
{
    L2SysSurResol aSys(4); // indicate the nb of params being estimated by MSE
    aData1 = NULL;
    for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    {
            Pt2dr aPt = itP->P1();
            Pt2dr aDepl = itP->P2()-itP->P1();
            double coeffX[4] = {aPt.x, -aPt.y, 1.0, 0.0};
            double coeffY[4] = {aPt.y, aPt.x, 0.0, 1.0};
            double delX = aDepl.x;
            double delY = aDepl.y;
            aSys.AddEquation(1.0, coeffX, delX);
            aSys.AddEquation(1.0, coeffY, delY);
    }
    bool solveOK = true;
    Im1D_REAL8 aResol1 = aSys.GSSR_Solve(&solveOK);
    aData1 = aResol1.data();

    if (solveOK != false)
    {
        cout<<"Estime : A B C D = "<<aData1[0]<<" "<<aData1[1]<<" "<<aData1[2]<<" "<<aData1[3]<<endl;
        rotCosSin.x = aData1[0];
        rotCosSin.y = aData1[1];
        transXY.x = aData1[2];
        transXY.y = aData1[3];
    }
    else
        cout<<"Can't estime"<<endl;
    return solveOK;
}

//====================================
bool cAppliEsSim::EsSimAndDisp (Pt2dr & aPtCtr, int & aSzw, Pt2dr & rotCosSin, Pt2dr & transXY)
{
    tImgEsSim aVigX;
    tImgEsSim aVigY;
    if ( mImgX->getVgt (aVigX, aPtCtr, aSzw) && mImgY->getVgt(aVigY, aPtCtr, aSzw) )
    {
        cout<<"Get Vignette : Sz X: "<<aVigX.sz()<<" - Y : "<<aVigY.sz()<<endl;
        int aZ = mParam->mZoom;
        if (mWX == 0)
        {
            mWX = Video_Win::PtrWStd(mParam->mDispSz*aZ,true,Pt2dr(aZ,aZ));
            mWX = mWX-> PtrChc(Pt2dr(0,0),Pt2dr(aZ,aZ),true);
            std::string aTitleX = std::string("Vignet X");
            mWX->set_title(aTitleX.c_str());
        }
        if (mWX)
        {
            //normalize to affiche
            tImgEsSim  mDisplay;
            mDisplay.Resize(aVigX.sz());
            mImgX->normalize(aVigX, mDisplay, 0.0, 255.0);
            ELISE_COPY(mDisplay.all_pts(),mDisplay.in(),mWX->ogray());
        }
        if (mWY == 0)
        {
            mWY = Video_Win::PtrWStd(mParam->mDispSz*aZ,true,Pt2dr(aZ,aZ));
            mWY = mWY-> PtrChc(Pt2dr(0,0),Pt2dr(aZ,aZ),true);
            std::string aTitleY = std::string("Vignet Y");
            mWY->set_title(aTitleY.c_str());
        }
        if (mWY)
        {
            //normalize to affiche
            tImgEsSim  mDisplay;
            mDisplay.Resize(aVigY.sz());
            mImgY->normalize(aVigY, mDisplay, 0.0, 255.0);
            ELISE_COPY(mDisplay.all_pts(),mDisplay.in(),mWY->ogray());
            mWY->clik_in();
        }
        //Estimation similitude
        L2SysSurResol aSys(4);
        double* aData1 = NULL;
        for (int aKx=0; aKx<aVigX.sz().x; aKx++)
        {
            for (int aKy=0; aKy<aVigX.sz().y; aKy++)
            {
                double coeffX[4] = {double(aKx + mImgX->Decal().x), double(-(aKy + mImgX->Decal().y)), 1.0, 0.0};
                double coeffY[4] = {double(aKy + mImgX->Decal().x), double(aKx + mImgX->Decal().x), 0.0, 1.0};
                double delX = aVigX.GetR(Pt2di(aKx, aKy));
                double delY = aVigY.GetR(Pt2di(aKx, aKy));
                aSys.AddEquation(1.0, coeffX, delX);
                aSys.AddEquation(1.0, coeffY, delY);
            }
        }
        bool solveOK = true;
        Im1D_REAL8 aResol1 = aSys.GSSR_Solve(&solveOK);
        aData1 = aResol1.data();
        if (solveOK != false)
        {
            cout<<"Estime : A B C D = "<<aData1[0]<<" "<<aData1[1]<<" "<<aData1[2]<<" "<<aData1[3]<<endl;
            rotCosSin.x = aData1[0];
            rotCosSin.y = aData1[1];
            transXY.x = aData1[2];
            transXY.y = aData1[3];

            return true;
        }
        else
        {
            cout<<"Can't estime"<<endl;
            return false;
        }
    }
    return true;
}

//====================================
bool cAppliEsSim::EsSimEnGrill(vector<Pt2di> aVPtCtrVig, int & aSzw, Pt2dr & rotCosSinAll, Pt2dr & transXYAll)
{
    vector<Pt2dr> result;
    Video_Win * mWDepl = 0;
    Pt2di aSzIm = mImgX->Tif().sz();
    Pt3di mSzW = Pt3di(mParam->mDispSz.x, mParam->mDispSz.y, mParam->mZoom);

    ElPackHomologue aPackAll;

    for (uint aKPt=0; aKPt<aVPtCtrVig.size(); aKPt++)
    {
        ElPackHomologue aPack;
        Pt2dr rotCosSin;
        Pt2dr transXY;
        Pt2dr aPtCtr = Pt2dr(aVPtCtrVig[aKPt]);

        Pt2dr aP0 = aPtCtr + Pt2dr(aSzw,aSzw);
        Pt2dr aP1 = aPtCtr - Pt2dr(aSzw,aSzw);

        if ( mImgX->IsInside(aP0) && mImgX->IsInside(aP1) )
        {
            getHomolInVgt (aPack, aPtCtr, aSzw);
            getHomolInVgt (aPackAll, aPtCtr, aSzw);
            EsSimFromHomolPack (aPack, rotCosSin, transXY);

            //exporter local estimation (aVPtCtrVig[aKPt] - rotCosSin - transXY)



            if (mParam->mInt != 0)
            {
                if (mWDepl == 0)
                {
                    //=====initialiser avec scale
                    if (aSzIm.x >= aSzIm.y)
                    {
                        double scale =  double(aSzIm.x) / double(aSzIm.y) ;
                        mSzW.x = mSzW.x;
                        mSzW.y = round_ni(mSzW.x/scale);
                    }
                    else
                    {
                        double scale = double(aSzIm.y) / double(aSzIm.x);
                        mSzW.x = round_ni(mSzW.y/scale);
                        mSzW.y = mSzW.y;
                    }
                    Pt2dr aZ(double(mSzW.x)/double(aSzIm.x) , double(mSzW.y)/double(aSzIm.y) );

                    if (mWDepl ==0)
                    {
                        mWDepl = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
                        mWDepl->set_sop(Elise_Set_Of_Palette::TheFullPalette());
                        mWDepl->set_title("Deplacement");
                    }
                    //===========================
                }
                if (mWDepl)
                {
                   mWDepl->draw_circle_loc(aPtCtr,3,mWDepl->pdisc()(P8COL::green));
                   mWDepl->draw_seg(aPtCtr, aPtCtr+ transXY*mParam->mSclDepl, mWDepl->pdisc()(P8COL::red));
                }
            }
        }

    }
    EsSimFromHomolPack (aPackAll, rotCosSinAll, transXYAll);
    double scale = rotCosSinAll.x > 0 ? euclid(rotCosSinAll) : -euclid(rotCosSinAll);
    cout<<endl<<rotCosSinAll<<" "<<transXYAll<<" -scale= "<<scale<< endl;


    if (mWDepl && mParam->mInt != 0)
    {
       mWDepl->draw_circle_loc(Pt2dr(aSzIm/2),3,mWDepl->pdisc()(P8COL::cyan));
       mWDepl->draw_seg(Pt2dr(aSzIm/2), Pt2dr(aSzIm/2)+ transXYAll*mParam->mSclDepl, mWDepl->pdisc()(P8COL::yellow));
       mWDepl->clik_in();
    }
    return true;
}
