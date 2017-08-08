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
}

void cLSQMatch::update(double CurErr, Pt2dr aPt)
{
    if (CurErr < mMinErr)
    {
        mMinErr = CurErr;
        mPtMinErr = aPt;
    }
}

/*
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

    //er
    double aReg=0.00001;

    for (aPt.x=20; aPt.x<aTmp.sz().x-20; aPt.x++)
    {
        for (aPt.y=20; aPt.y<aTmp.sz().y-20; aPt.y++)
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

    for(int aK=0; aK<4; aK++)
    {
        aSys.AddTermQuad(aK,aK,aReg);
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
                //mcurrErr += ElSquare(aVTmp - aVImg);
            }
        }
        mcurrErr = sqrt(mcurrErr/(aTmp.sz().x*aTmp.sz().y));
        cout<<"Err : "<<mcurrErr<<" - "<<h0<<" "<<h1<<" "<<Trx<<" "<<Try<<endl;
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
*/


/*
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
            aCoeff[0] = aVTmp;
            aCoeff[1] = h1; //1.0
            aCoeff[2] = -aDxImg;
            aCoeff[3] = -aDyImg;
            aSys.AddEquation(1.0, aCoeff, aVImgVal);
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
                //mcurrErr += ElSquare(aVTmp - aVImg);
            }
        }
        mcurrErr = sqrt(mcurrErr/(aTmp.sz().x*aTmp.sz().y));
        cout<<"Err : "<<mcurrErr<<" - "<<h0<<" "<<h1<<" "<<Trx<<" "<<Try<<endl;
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
*/

bool cLSQMatch::MatchbyLSQ(
                                Pt2dr aPt1,
                                const tIm2DM & aImg1,   // template
                                const tIm2DM & aImg2,   // image
                                Pt2dr aPt2,
                                Pt2di aSzW,
                                double aStep,
                                Im1D_REAL8 & aSol,
                                ElAffin2D & aTrans12
                          )
{
    /*
     * Matching by LSQ between  aImg1 & aImg2, with point init correspondant aPt1, aPt2
     * Model : A*V1 + B = V2 + dV2/dx*Trx + dV2/dy*TrY
     *
     */
    Pt2dr aPt(0,0);

    int aNbInconnu=8;
    switch (mParam.mCase)
    {
        case 0: // only trans
            {
                aNbInconnu = 2;
                break;
            }
        case 1: // trans + Aff
            {
                aNbInconnu = 6;
                break;
            }
        case 2: // Trans + Aff + Radio
            {
                aNbInconnu = 8;
                break;
            }
        case 3: // Trans + Radio
            {
                aNbInconnu = 4;
                break;
            }
        case 4: // Aff
            {
                aNbInconnu = 4;
                break;
            }
    }
    L2SysSurResol aSys(aNbInconnu);
    double mCoeff[8];
    double sqr_residu=0;
    for (aPt.x = -aSzW.x; aPt.x < aSzW.x; aPt.x = aPt.x + aStep)
    {
        for (aPt.y = -aSzW.y; aPt.y < aSzW.y; aPt.y = aPt.y + aStep)
        {
            Pt2dr aPC1 = aPt1 + aPt;
            Pt2dr aPC2 = aTrans12(aPC1);
            double aV1 = mInterpol->GetVal(aImg1.data(),aPC1);    // value of center point (point master)
            Pt3dr aNewVD2= mInterpol->GetValDer(aImg2.data(),aPC2);   // Get intensity & derive value of point 2nd img
            double aGr2X = aNewVD2.x;  // derive en X
            double aGr2Y = aNewVD2.y;  // derive en Y
            double aV2   = aNewVD2.z;  // valeur d'intensite
/*          // This code works also
            mCoeff[0] = aV1 ; // A
            mCoeff[1] = 1.0 ; // B
            mCoeff[2] = -aGr2X; // im00.x
            mCoeff[3] = -aGr2Y;  // im00.y

            if (mParam.mAff)
            {
                mCoeff[4] = -aGr2X*aPt1.x ; // A
                mCoeff[5] = -aGr2Y*aPt1.x ; // B
                mCoeff[6] = -aGr2X*aPt1.y; // im00.x
                mCoeff[7] = -aGr2Y*aPt1.y;  // im00.y
            }

            aSys.AddEquation(1.0,mCoeff,aV2);
*/
            switch (mParam.mCase)
            {
                case 0: // only trans
                    {
                        mCoeff[0] = aGr2X ; // Trx
                        mCoeff[1] = aGr2Y ; // Try
                        break;
                    }
                case 1: // trans + Aff
                    {
                        mCoeff[0] = aGr2X ; // Trx
                        mCoeff[1] = aGr2Y ; // Try
                        mCoeff[2] = aGr2X*aPC1.x ; // im10
                        mCoeff[3] = aGr2Y*aPC1.x ;
                        mCoeff[4] = aGr2X*aPC1.y; // im01
                        mCoeff[5] = aGr2Y*aPC1.y;
                        break;
                    }
                case 2: // Trans + Aff + Radio
                    {
                        mCoeff[0] = aGr2X ; // Trx
                        mCoeff[1] = aGr2Y ; // Try
                        mCoeff[2] = aGr2X*aPC1.x ; // im10
                        mCoeff[3] = aGr2Y*aPC1.x ;
                        mCoeff[4] = aGr2X*aPC1.y; // im01
                        mCoeff[5] = aGr2Y*aPC1.y;
                        mCoeff[6] = aV2 ; // A
                        mCoeff[7] = 1.0 ; // B
                        break;
                        }
                case 3: // Trans + Radio
                    {
                        mCoeff[0] = aGr2X ; // Trx
                        mCoeff[1] = aGr2Y ; // Try
                        mCoeff[2] = aV2 ; // A
                        mCoeff[3] = 1.0 ; // B
                        break;
                    }
                case 4: // Aff
                    {
                        mCoeff[0] = aGr2X*aPC1.x ; // im10
                        mCoeff[1] = aGr2Y*aPC1.x ;
                        mCoeff[2] = aGr2X*aPC1.y; // im01
                        mCoeff[3] = aGr2Y*aPC1.y;
                        break;
                    }
            }
            aSys.AddEquation(1.0,mCoeff,aV1-aV2);
            sqr_residu+=(aV1-aV2)*(aV1-aV2);
        }
    }
    cout<<"sqr_residu: "<<sqr_residu<<endl;
    // ======= regulisation ====== //
    /*
    double aReg=0.0001;
    for(int aK=0; aK<aNbInconnu; aK++)
    {
        aSys.AddTermQuad(aK,aK,aReg);
    }
    */
    bool OK = false;
    aSol = aSys.Solve(&OK);
    cout<<"Retour solve: "<<OK<<endl;
    return OK;
}
/********************************************************************************/
/*                                                                              */
/*                  Correlation entiere                                         */
/*                                                                              */
/********************************************************************************/

double Tst_Correl1Win
                             (
                                const tIm2DM & Im1,
                                const Pt2di & aP1,
                                const tIm2DM & Im2,
                                const Pt2di & aP2,
                                const Pt2di   aSzW,
                                const int   aStep
                             )
{
    /*
     * This function compute correlation score between 2 images patch
     * As an input, given Im1, Im2 & 2 image patche center coordinate correspondant aP1 & aP2
     * Given also half size of image patch in [x,y] as parameter aSzW
     * Given also pixel sampling factor aStep. if aStep=1 => pixel entier is add to correlation compute
     */

     if (
             ! ((Im1.Inside(aP1-aSzW) && Im1.Inside(aP1+aSzW))
            && (Im2.Inside(aP2-aSzW)) && Im2.Inside(aP2+aSzW))
        )
             return (TT_DefCorrel);
     Pt2di aVois;
     RMat_Inertie aMatr;

     for  (aVois.x = -aSzW.x ; aVois.x<=aSzW.x  ; aVois.x++)
     {
          for  (aVois.y = -aSzW.y ; aVois.y<=aSzW.y  ; aVois.y++)
          {
               aMatr.add_pt_en_place(Im1.GetR(aP1+aVois*aStep),Im2.GetR(aP2+aVois*aStep));
          }
     }

     return aMatr.correlation();
}




cResulRechCorrel   Tst_Correl
                      (
                             const tIm2DM & Im1,
                             const Pt2di & aP1,
                             const tIm2DM & Im2,
                             const Pt2di & aP2,
                             const Pt2di aSzW,
                             const int   aStep,
                             const Pt2di   aSzRech,    // size of search zone
                             tIm2DM & ImScore
                      )
{
    /*
     * This function search for best match position by correlation
     * As an input, given Im1, Im2 & 2 image patche center coordinate correspondant aP1 & aP2
     * Given also half size of image patch of Img1 in [x,y] as parameter aSzW
     * Given also pixel sampling factor aStep. if aStep=1 => pixel entier is add to correlation compute
     * Given also half size of searching zone around aP2, with parameter aSzRech
     * Output stock in class cResulRechCorrel, contain coordinate of matched on Img2, score.
     * Image of score of correlation is provide also as output parameter ImScore
     */
    double aScoreMax = -1e30;
    Pt2di  aDecMax;
    Pt2di  aP;
    for (aP.x=-aSzRech.x ; aP.x<= aSzRech.x ; aP.x = aP.x + aStep)
    {
        for (aP.y=-aSzRech.y ; aP.y<= aSzRech.y ; aP.y = aP.y + aStep)
        {
             double a2Sol  = Tst_Correl1Win(Im1,aP1,Im2,aP2+aP,aSzW,1);

             if (a2Sol > aScoreMax)
             {
                 aScoreMax = a2Sol;
                 aDecMax = aP;
             }
             ImScore.SetR_SVP(Pt2di(aP2 + aP),a2Sol);
        }
     }

     return cResulRechCorrel(Pt2dr(aP2+aDecMax),aScoreMax);
}

/********************************************************************************/
/*                                                                              */
/*                  Correlation (all is real input)                             */
/*                                                                              */
/********************************************************************************/

bool  InsideREAL(const Im2DGen & Im, const Pt2dr & aP)
{
   return    (aP.x>=0)
          && (aP.y>=0)
          && (aP.x<Im.sz().x)
          && (aP.y<Im.sz().y);
}


double dblTst_Correl1Win
                             (
                                tIm2DM & Im1,           // can't use "const tIm2DM" because "Get" member function isn't "const"
                                const Pt2dr & aP1,
                                tIm2DM & Im2,
                                const Pt2dr & aP2,
                                const Pt2dr   aSzW,
                                const int aStepPxl,
                                const cInterpolateurIm2D<double> & aInterPol
                             )
{
    /*
     * This function compute correlation score between 2 images patch
     * As an input, given Im1, Im2 & 2 image patche center coordinate correspondant aP1 & aP2
     * Given also half size of image patch in [x,y] as parameter aSzW
     * Given also pixel sampling factor aStep. if aStep=1 => pixel entier is add to correlation compute
     * Logic with aStep here is if we want to do correlation faster by over-sample image (every 2 pixels for ex)
     */

    // Compute point most up & most down for two images
    // Image 1 & 2 : check if center patch point aP with window size aSzW is inside
    Pt2dr aPtSupIm1 = aP1 + aSzW;
    Pt2dr aPtInfIm1 = aP1 - aSzW;
    Pt2dr aPtSupIm2 = aP2 + aSzW;
    Pt2dr aPtInfIm2 = aP2 - aSzW;
    if (!(
            (InsideREAL(Im1, aPtSupIm1) && InsideREAL(Im1, aPtInfIm1))
            &&
            (InsideREAL(Im2, aPtSupIm2) && InsideREAL(Im2, aPtInfIm2))
       ) )
            return (TT_DefCorrel);

     double aDefInter = -1.0;
     Pt2dr aVois;
     RMat_Inertie aMatr;

     for  (aVois.x = -aSzW.x ; aVois.x<=aSzW.x  ; aVois.x = aVois.x+aStepPxl)
     {
          for  (aVois.y = -aSzW.y ; aVois.y<=aSzW.y  ; aVois.y = aVois.y+aStepPxl)
          {
               Pt2di aPtIm1 = Pt2di(aP1+aVois);         // for a template => interpolator nearest (in case of even size template)
               aMatr.add_pt_en_place(
                                        Im1.GetR(aPtIm1),
                                        //Im1.Get(aP1+aVois, aInterPol, aDefInter),
                                        Im2.Get(aP2+aVois, aInterPol, aDefInter)
                                    );
          }
     }

     return aMatr.correlation();
}




cResulRechCorrel   dblTst_Correl
                      (
                             tIm2DM & Im1,
                             const Pt2dr & aP1,
                             tIm2DM & Im2,
                             const Pt2dr & aP2,
                             const Pt2dr aSzW,
                             const double   aStep,
                             const int   aStepPxl,
                             const Pt2dr   aSzRech,    // size of search zone
                             tIm2DM & ImScore,
                             const cInterpolateurIm2D<double> & aInterPol,
                             bool & OK
                      )
{
    /*
     * This function search for best match position by correlation
     * As an input, given Im1, Im2 & 2 image patche center coordinate correspondant aP1 & aP2
     * Given also half size of image patch of Img1 in [x,y] as parameter aSzW
     * Given also windows movement step aStep (correlation sub pixel if aStep < 1)
     * Given also half size of searching zone around aP2, with parameter aSzRech
     * Output stock in class cResulRechCorrel, contain coordinate of matched on Img2, score.
     * Image of score of correlation is provide also as output parameter ImScore
     */

    Pt2dr aPtSupIm1 = aP1 + aSzW;
    Pt2dr aPtInfIm1 = aP1 - aSzW;
    int aSzInterpol = aInterPol.SzKernel();
    Pt2dr aPtSupIm2 = aP2 + aSzRech + Pt2dr(aSzInterpol, aSzInterpol);
    Pt2dr aPtInfIm2 = aP2 - aSzRech - Pt2dr(aSzInterpol, aSzInterpol);
    cout<<aPtSupIm1<<aPtInfIm1<<" "<<aPtSupIm2<<aPtInfIm2<<endl;
    if (!(
            (InsideREAL(Im1, aPtSupIm1) && InsideREAL(Im1, aPtInfIm1))
            &&
            (InsideREAL(Im2, aPtSupIm2) && InsideREAL(Im2, aPtInfIm2))
       ) )
    {
            OK=false;
            return (cResulRechCorrel (Pt2dr(-1,-1),TT_DefCorrel));
    }

    double aScoreMax = -1e30;
    Pt2dr  aDecMax;
    Pt2dr  aP;
    for (aP.x=-aSzRech.x ; aP.x<= aSzRech.x ; aP.x = aP.x + aStep)
    {
        for (aP.y=-aSzRech.y ; aP.y<= aSzRech.y ; aP.y = aP.y + aStep)
        {
            // cout<<" + Pt: "<<aP1<<aP2+aP;
             double a2Sol  = dblTst_Correl1Win(Im1,aP1,Im2,aP2+aP,aSzW,aStepPxl,aInterPol);
            // cout<<" -Scr :"<<a2Sol<<endl;
             if (a2Sol > aScoreMax)
             {
                 aScoreMax = a2Sol;
                 aDecMax = aP;
             }
             ImScore.SetR_SVP(Pt2di(aP2 + aP),a2Sol);
        }
     }
     OK = true;
     return cResulRechCorrel(Pt2dr(aP2+aDecMax),aScoreMax);
}



/********************************************************************************/
/*                                                                              */
/*                  Main function                                               */
/*                                                                              */
/********************************************************************************/

int LSQMatch_Main(int argc,char ** argv)
{
   string aTmpl ="";
   string aImg = "";
   string aDir = "./";
   cParamLSQMatch aParam;
   aParam.mDisp = false;
   aParam.mStepCorrel = 1.0;
   aParam.mStepLSQ = 1.0;
   aParam.mStepPxl = 1;
   aParam.mNbIter = 1;
   int method = 0;
   aParam.mAff = false;
   aParam.mRadio = false;
   aParam.mCase = 3;

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aTmpl, "Image Template",  eSAM_IsExistFile)
                     << EAMC(aImg, "Target Image to search for template",  eSAM_IsExistFile),
         LArgMain()   
                     << EAM(aParam.mDisp, "Disp", true, "Display ? (click to Tar image)")
                     << EAM(aParam.mStepCorrel, "StepCor", true, "Step of windows movement in Correlation")
                     << EAM(aParam.mStepPxl, "StepPix", true, "Step of pixel sampling in 1 Correlation")
                     << EAM(aParam.mStepLSQ, "StepLSQ", true, "Step of pixel sampling in LSQ")
                     << EAM(aParam.mNbIter, "NbIter", true, "Number of LSQ iteration (def=1)")
                     //<< EAM(aParam.mAff, "Aff", true, "Estimate Affine part in LSQ - total 8 param if true (def=false)")
                     //<< EAM(aParam.mRadio, "Radio", true, "Estimate Radiometry part in LSQ - (def=false)")
                     << EAM(aParam.mCase, "Case", true, "0 = Trans, 1 = Trans + Aff, 2 = Trans + Aff + Radio,  3 = Trans + Radio, 4 = Aff - def = 3")
                     << EAM(method, "Meth", true, "method corelation (0, 1=cCorrelImage)")
               );
         cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
         cImgMatch * aImgTarget = new cImgMatch(aImg, anICNM);
         aImgTarget->Load();
         cImgMatch * aImgTmplt = new cImgMatch(aTmpl, anICNM);
         aImgTmplt->Load();
         cInterpolBilineaire<double> * aInterpolBilin = new cInterpolBilineaire<double>;
         cResulRechCorrel aResCorrel(Pt2dr(-1,-1),TT_DefCorrel);

         Pt2dr aPt1 = Pt2dr(aImgTmplt->Im2D().sz()-Pt2di(1,1))/2;       // (1,1) to have a good pixel index
         Pt2dr aPt2 = Pt2dr(aImgTarget->Im2D().sz()-Pt2di(1,1))/2;
         Pt2dr aSzRech = Pt2dr(aImgTarget->Im2D().sz() - Pt2di(1,1))/2  - Pt2dr(2.0,2.0); // (2,2) for a rab
         Pt2dr aSzW = Pt2dr(aImgTmplt->Im2D().sz()-Pt2di(1,1))/2;       // Sz win not include center pixel

         cout<<"P1 : "<<aPt1<<" -P2 : "<<aPt2<<" -SzWin : "<<aSzW<<" -SzRech :"<<aSzRech<<endl;

         tIm2DM aImgScoreCorrel(aImgTarget->Im2D().sz().x, aImgTarget->Im2D().sz().y);
         bool OK = false;
         ElTimer a;
if (method == 0)
{
         /*================ Corrrelation real ==================*/
         Pt2dr aPt(0.0,0.0);
         //double aStep = 1.0;
         aResCorrel =    dblTst_Correl
                                                                   (
                                                                          aImgTmplt->Im2D(),
                                                                          aPt1,
                                                                          aImgTarget->Im2D(),
                                                                          aPt2,
                                                                          aSzW,
                                                                          aParam.mStepCorrel,
                                                                          aParam.mStepPxl,
                                                                          aSzRech,
                                                                          aImgScoreCorrel,
                                                                          *aInterpolBilin,
                                                                          OK
                                                                   );
}
         /*================ Corrrelation by cCorrelImage ==================*/
if (method == 1)
{
         // Load Image to type Im2D<unsigned int, int>
         Pt2di aSzImTarget = aImgTarget->Im2D().sz();
         Pt2di aSzImTmp = aImgTmplt->Im2D().sz();
         tIm2DcCorrel aCImTmpl(aSzImTmp.x, aSzImTmp.y);
         tIm2DcCorrel aCImTarget(aSzImTarget.x, aSzImTarget.y);
         ELISE_COPY(aCImTmpl.all_pts(), aImgTmplt->Tif().in(), aCImTmpl.out());
         ELISE_COPY(aCImTarget.all_pts(), aImgTarget->Tif().in(), aCImTarget.out());
         cout<<"Im Init OK"<<endl;
         // Creat image patch as object of class cCorrelImage
         cCorrelImage::setSzW(aSzW.x);  // this is static variable => set it before creat object
         cCorrelImage ImPatchTmpl;
         ImPatchTmpl.getWholeIm(&aCImTmpl);

         cCorrelImage ImPatchTarget;
         bool corOK = false;

         Pt2dr aPt(0,0);
         OK = true;
         for (aPt.x=0; aPt.x<aSzImTarget.x; aPt.x = aPt.x + (int)aParam.mStepCorrel)        // class cCorrelImage don't support real coordinate
         {
             for (aPt.y=0; aPt.y<aSzImTarget.y; aPt.y = aPt.y + (int)aParam.mStepCorrel)
             {
                 Pt2dr aPtInf = aPt-Pt2dr(ImPatchTmpl.getmSz());
                 // if image patch is inside image
                 if (    (aPtInf.x >= 0 && aPtInf.y >= 0)
                      && (aPtInf.x < aSzImTarget.x && aPtInf.y < aSzImTarget.y) )
                 {
                      // Load image patch from target image
                      ImPatchTarget.getFromIm(&aCImTarget, aPt.x, aPt.y);
                      double aScore = ImPatchTmpl.CrossCorrelation(ImPatchTarget);
                      corOK = true;
                      if (aScore > aResCorrel.mCorrel)
                      {
                          aResCorrel.mCorrel = aScore;
                          aResCorrel.mPt = Pt2dr(aPt);
                      }
                      aImgScoreCorrel.SetR_SVP(Pt2di(aPt), aScore);
                 }
                 else
                 {
                      corOK = false;
                      aImgScoreCorrel.SetR_SVP(Pt2di(aPt),-1.0);
                 }
             }
         }
}
cout<<endl<<endl<<"Correlation Time : "<<a.uval()<<endl<<endl;
string imScore = "imScore.tif";
ELISE_COPY
        (
            aImgScoreCorrel.all_pts(),
            aImgScoreCorrel.in_proj(),
            Tiff_Im(
                imScore.c_str(),
                aImgScoreCorrel.sz(),
                GenIm::real8,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
                //aZBuf->Tif().phot_interp()
                ).out()

            );
if (OK)
   cout<<"Correl : "<<aResCorrel.mCorrel<<" - Pt: "<<aResCorrel.mPt<<endl;
else
    cout<<"Correl false"<<endl;


cResulRechCorrel aResCorrelOrg(aResCorrel.mPt, aResCorrel.mCorrel);

         /*================= LSQ =====================*/
         // Do refine matching by LSQ
         aImgTarget->GetImget(aResCorrel.mPt, aImgTmplt->SzIm());
         cLSQMatch * aMatch = new cLSQMatch(aImgTmplt, aImgTarget);
         aMatch->Param() = aParam;

         int aNbInconnu = 8;
         switch (aParam.mCase)
         {
             case 0: // only trans
                 {
                     aNbInconnu = 2;
                     break;
                 }
         case 1: // trans + Aff
             {
                 aNbInconnu = 6;
                 break;
             }
         case 2: // Trans + Aff + Radio
             {
                 aNbInconnu = 8;
                 break;
             }
         case 3: // Trans + Radio
             {
                 aNbInconnu = 4;
                 break;
             }
         case 4: // Aff
             {
                 aNbInconnu = 4;
                 break;
             }
         }


    ElAffin2D aTransAffFinal(Pt2dr(aResCorrelOrg.mPt - aPt1),
                             Pt2dr(1,0),
                             Pt2dr(0,1));
    cout<<"Translation initiale: "<<aTransAffFinal.I00()<<endl;

    for (int aK=0; aK<aParam.mNbIter; aK++)
    {
        Pt2dr aPtRes = aTransAffFinal(Pt2dr(42,20));
        cout<<"Le Pt [42,20] devient : "<<aPtRes<<" -Res: "<<
              aImgTarget->Im2D().Get(aPtRes, *aInterpolBilin, -1.0) - aImgTmplt->Im2D().GetR(Pt2di(42,20))
           <<endl;

         Im1D_REAL8 aSol(aNbInconnu);
         ElAffin2D aTransAffInit(ElAffin2D::Id());

         aMatch->MatchbyLSQ (
                                 aPt1,
                                 aImgTmplt->Im2D(),
                                 aImgTarget->Im2D(),
                                 aResCorrel.mPt,
                                 Pt2di(aImgTmplt->Im2D().sz()/2),
                                 aParam.mStepLSQ,
                                 aSol,
                                 aTransAffFinal
                             );


         double * aResLSQ = aSol.data();
/*
         if (aParam.mCase == 0 || aParam.mCase == 3)    // trans only
         {
             ElAffin2D aT(aTransAffInit.trans(Pt2dr(aResLSQ[0],aResLSQ[1])));
             aTransAffFinal.update(aT);
         }
         else
         {
             if (aParam.mCase == 4)
             {
                 ElAffin2D aT(
                                 aTransAffInit + ElAffin2D(   Pt2dr(0,0),
                                                              Pt2dr(aResLSQ[0], aResLSQ[1]),
                                                              Pt2dr(aResLSQ[2], aResLSQ[3])
                                                          )
                             );
                aTransAffFinal.update(aT);
             }
             else
             {
                 ElAffin2D aT(
                                 aTransAffInit + ElAffin2D(   Pt2dr(aResLSQ[0], aResLSQ[1]),
                                                              Pt2dr(aResLSQ[2], aResLSQ[3]),
                                                              Pt2dr(aResLSQ[4], aResLSQ[5])
                                                          )
                             );
                 aTransAffFinal.update(aT);
             }
         }

*/
                         //cout<<aResLSQ[0]<<" "<<aResLSQ[1]<<endl;
         if (aParam.mCase == 0 || aParam.mCase == 3)    // trans only or Trans + Radio
         {
             cout<<"Delta: "<<Pt2dr(aResLSQ[0], aResLSQ[1])<<endl;
             aTransAffFinal = aTransAffFinal + ElAffin2D(   Pt2dr(aResLSQ[0],aResLSQ[1]),
                                                            Pt2dr(0,0),
                                                            Pt2dr(0,0)
                                                        );
         }
         else
         {
             if (aParam.mCase == 4) // Aff Only
             {

                 aTransAffFinal = aTransAffFinal  + ElAffin2D(  Pt2dr(0,0),
                                                                Pt2dr(aResLSQ[0], aResLSQ[1]),
                                                                Pt2dr(aResLSQ[2], aResLSQ[3])
                                                             );

             }
             else
             {  // Trans + Aff
                  cout<<"Delta: "<<Pt2dr(aResLSQ[0], aResLSQ[1])<<" "<<Pt2dr(aResLSQ[2], aResLSQ[3])
                          <<" "<<Pt2dr(aResLSQ[4], aResLSQ[5])<<endl;
                  aTransAffFinal = aTransAffFinal  + ElAffin2D(   Pt2dr(aResLSQ[0], aResLSQ[1]),
                                                                  Pt2dr(aResLSQ[2], aResLSQ[3]),
                                                                  Pt2dr(aResLSQ[4], aResLSQ[5])
                                                              );
             }
         }

         cout<<"==== Iter "<<"["<<aK<<"]"<<" ====="<<endl;

         cout<<"    ER **I00: "<< aTransAffFinal.I00() <<" -I01: "<< aTransAffFinal.I01() <<" -I10: "<<aTransAffFinal.I10();

         if (aParam.mCase == 2)
            cout<<" -A: "<<aResLSQ[6]<<" -B: "<<aResLSQ[7];
         if (aParam.mCase == 3)
             cout<<" -A: "<<aResLSQ[2]<<" -B: "<<aResLSQ[3];

           cout<<endl;

         cout<<"    **Before : "<<aResCorrel.mPt;

         // update matched result
         aResCorrel.mPt = aTransAffFinal(aPt1);
         cout<<" - After LSQ: "<<aResCorrel.mPt<<endl<<endl;
         //aMatch->DoMatchbyLSQ();

    }
    cout<<"TransAff : "<<aTransAffFinal.I00()<<aTransAffFinal.I01()<<aTransAffFinal.I10()<<endl;

    // Export to view result
    /*
    ElAffin2D aTrAff(
                        aTransAffFinal.I00() + Pt2dr(aResCorrelOrg.mPt - aPt1),
                        aTransAffFinal.I10(),
                        aTransAffFinal.I01()
                    );


    cout<<"TransAff : "<<aTrAff.I00()<<aTrAff.I01()<<aTrAff.I10()<<endl;
   */


    cout<<endl<<endl;
    tIm2DM aImgVerif(aImgTmplt->Im2D().sz().x, aImgTmplt->Im2D().sz().y);
    Pt2di aPt(0,0);
    OK = true;
    for (aPt.x=0; aPt.x<aImgTmplt->Im2D().sz().x; aPt.x++)        // class cCorrelImage don't support real coordinate
    {
        for (aPt.y=0; aPt.y<aImgTmplt->Im2D().sz().y; aPt.y++)
        {
            Pt2dr aPxlTarget = aTransAffFinal(Pt2dr(aPt));
            double aVal = aImgTarget->Im2D().Get(aPxlTarget, *aInterpolBilin, -1.0);
            //cout<<aPt<<" -> "<<aPxlTarget<<" : "<<aVal<<endl;
            aImgVerif.SetR_SVP(
                                aPt,
                                aVal
                              );
        }
    }
    string imVerif = aTmpl + "_Verif.tif";
    ELISE_COPY
            (
                aImgVerif.all_pts(),
                aImgVerif.in_proj(),
                Tiff_Im(
                    imVerif.c_str(),
                    aImgVerif.sz(),
                    GenIm::real8,
                    Tiff_Im::No_Compr,
                    Tiff_Im::BlackIsZero
                    ).out()

                );





    return EXIT_SUCCESS;
}
