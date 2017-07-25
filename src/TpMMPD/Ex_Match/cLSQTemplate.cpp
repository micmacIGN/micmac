#include "cLSQTemplate.h"

cLSQMatch::cLSQMatch(cImgMatch * aTmpl, cImgMatch * aImg):
    mTemplate (aTmpl),
    mImg      (aImg),
    mICNM     (aImg->ICNM()),
    mcurrErr  (0.0),
    mImRes    (aImg->SzIm().x, aImg->SzIm().y),
    mWTemplate (0),
    mWTarget   (0),
    mMinErr    (DBL_MAX),
    mPtMinErr  (-1,-1)
{
    mInterpol = new cInterpolBilineaire<double>;
    cout<<"In const cLSQMatch"<<endl;
}

void cLSQMatch::update(double CurErr, Pt2dr aPt)
{
    if (CurErr < mMinErr)
    {
        mMinErr = CurErr;
        mPtMinErr = aPt;
    }
}


bool cLSQMatch::DoMatchbyLSQ()
{
    mcurrErr = 0.0;
    // for each pixel couple from Template & Target :
    Pt2dr aPt(0,0);

    // In Template
    tIm2DM  aTmp = mTemplate->Im2D();
    tTIm2DM aTTmp= mTemplate->TIm2D();

    // In Target
    tIm2DM  aTarget = mImg->CurImgetIm2D();
    tTIm2DM aTTarget = mImg->CurImgetTIm2D();

    if (mParam.mDisp)
    {
        // Display Template & Target before solve
        //mWTemplate->set_sop(Elise_Set_Of_Palette::TheFullPalette());
        //mWTarget->set_sop(Elise_Set_Of_Palette::TheFullPalette());
        cout<<"Disp : "<<"Tmp : "<<aTmp.sz()<<" Tar : "<<aTarget.sz()<<endl;
        if (mWTemplate == 0)
        {
            mWTemplate = Video_Win::PtrWStd(aTmp.sz(),true,Pt2dr(1,1));
            std::string aTitle = std::string("Tmp");
            mWTemplate->set_title(aTitle.c_str());
        }
        if (mWTarget == 0)
        {
            mWTarget = Video_Win::PtrWStd(aTarget.sz(),true,Pt2dr(1,1));
            std::string aTitle = std::string("Tar");
            mWTarget->set_title(aTitle.c_str());
        }
        if (mWTemplate)
        {
            ELISE_COPY(aTmp.all_pts(),aTmp.in(),mWTemplate->ogray());
        }
        if (mWTarget)
        {
            ELISE_COPY(aTarget.all_pts(),aTarget.in(),mWTarget->ogray());
        }
        mWTarget->clik_in();
    }

    L2SysSurResol aSys(4);  // 4 variable: (A, B, trX, trY)
    //Value Init :
    double h0=0.0;
    double h1=1.0;
    double aCoeff[4];
    double Trx = 0.0;
    double Try = 0.0;


    for (aPt.x=0; aPt.x<aTmp.sz().x; aPt.x++)
    {
        for (aPt.y=0; aPt.y<aTmp.sz().y; aPt.y++)
        {
            //bool BUG = (cnt==804);
            //get data in Template
            double aVTmp = aTmp.Get(aPt, *mInterpol, -1.0);
            //get data in Target
            Pt3dr aVImg = mInterpol->GetValDer(aTarget.data(), aPt);
            double aVImgVal = aVImg.z;
            double aDxImg = aVImg.x;
            double aDyImg = aVImg.y;
            // former equation
            double aB = aVTmp - h0 - h1*aVImgVal;
            aCoeff[0] = 1.0;
            aCoeff[1] = aVImgVal;
            aCoeff[2] = h1*aDxImg;
            aCoeff[3] = h1*aDyImg;
            aSys.AddEquation(1.0, aCoeff, aB);
        }
    }
    bool OK = false;
    aSys.Solve(&OK);
    Im1D_REAL8 aSol = aSys.Solve(&OK);
    double * aDS = aSol.data();
    if (OK)
    {
        // estimate error
        Pt2dr aPtErr(0,0);
        h0 = h0 + aDS[0];   // 0.0
        h1 = h1 + aDS[1];   // 1.0
        Trx = Trx + aDS[2];
        Try = Try + aDS[3];
        for (aPtErr.x=0; aPtErr.x<aTmp.sz().x; aPtErr.x++)
        {
            for (aPtErr.y=0; aPtErr.y<aTmp.sz().y; aPtErr.y++)
            {
                double aVTmp = aTmp.Get(aPtErr, *mInterpol, -1.0);
                double aVImg = aTmp.Get(aPtErr + Pt2dr(Trx,Try), *mInterpol, -1.0);
                mcurrErr += ElSquare(aVTmp -h0 -h1*aVImg);
            }
        }
        mcurrErr = sqrt(mcurrErr/(aTmp.sz().x*aTmp.sz().y));
        //cout<<"Err : "<<mcurrErr<<" - "<<h0<<" "<<h1<<" "<<Trx<<" "<<Try<<endl;
        update(mcurrErr, mImg->CurPt());
        mImRes.SetR_SVP(Pt2di(mImg->CurPt()),mcurrErr);
        return true;
    }
    else
    {
        mImRes.SetR_SVP(Pt2di(mImg->CurPt()),mcurrErr);
        return false;
    }
}

bool cLSQMatch::DoMatchbyCorel()
{
    Pt2dr aPt(0,0);
    // In Template
    tIm2DM  aTmp = mTemplate->Im2D();
    tTIm2DM aTTmp= mTemplate->TIm2D();

    // In Target
    tIm2DM  aTarget = mImg->CurImgetIm2D();
    tTIm2DM aTTarget = mImg->CurImgetTIm2D();

    // Correlation b/w Template & Target
    return true;
}



int LSQMatch_Main(int argc,char ** argv)
{
   string aTmpl ="";
   string aImg = "";
   string aDir = "./";
   cParamLSQMatch aParam;
   aParam.mDisp = false;

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aTmpl, "Image Template",  eSAM_IsExistFile)
                     << EAMC(aImg, "Target Image to search for template",  eSAM_IsExistFile),
         LArgMain()   
                     << EAM(aParam.mDisp, "Disp", true, "Display ? (click to Tar image)")
               );
         cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
         cout<<"Lire Img Target"<<endl;
         cImgMatch * aImgTarget = new cImgMatch(aImg, anICNM);
         cout<<"Lire Img Template"<<endl;
         cImgMatch * aImgTmplt = new cImgMatch(aTmpl, anICNM);
         aImgTmplt->Load();
         cout<<"Create matching"<<endl;
         cLSQMatch * aMatch = new cLSQMatch(aImgTmplt, aImgTarget);
         aMatch->Param() = aParam;

         Pt2dr aPt(0,0);
         double aStep = 1.0;
         int cnt = 0;
         for (aPt.x=0; aPt.x < aImgTarget->SzIm().x; aPt.x = aPt.x + aStep)
         {
             for (aPt.y=0; aPt.y<aImgTarget->SzIm().y; aPt.y = aPt.y + aStep)
             {
                if (aImgTarget->GetImget(aPt, aImgTmplt->SzIm()))
                {
                    aMatch->DoMatchbyLSQ();
                    cnt++;
                    //aMatch->DoMatchbyCorel();
                    if (cnt % 1000 == 0)
                      cout<<"["<<(cnt*100.0/(aImgTarget->SzIm().x * aImgTarget->SzIm().y))<<" %]"<<endl;
                }
             }
         }
         // write Image residue to Disk
         string imRes = "imres.tif";
         ELISE_COPY
                 (
                     aMatch->ImRes().all_pts(),
                     aMatch->ImRes().in_proj(),
                     Tiff_Im(
                         imRes.c_str(),
                         aMatch->ImRes().sz(),
                         GenIm::real8,
                         Tiff_Im::No_Compr,
                         Tiff_Im::BlackIsZero
                         //aZBuf->Tif().phot_interp()
                         ).out()

                     );
         cout<<"MinErr : "<<aMatch->MinErr()<<" - Pt: "<<aMatch->PtMinErr()<<endl;
    return EXIT_SUCCESS;
}
