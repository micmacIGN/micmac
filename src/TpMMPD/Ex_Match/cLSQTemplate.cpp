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
                                Im1D_REAL8 & aSol
                          )
{
    /*
     * Matching by LSQ between  aImg1 & aImg2, with point init correspondant aPt1, aPt2
     * Model : A*V1 + B = V2 + dV2/dx*Trx + dV2/dy*TrY
     *
     */
    Pt2dr aPt(0,0);
    L2SysSurResol aSys(4);
    double mCoeff[4];
    for (aPt.x = -aSzW.x; aPt.x < aSzW.x; aPt.x = aPt.x + aStep)
    {
        for (aPt.y = -aSzW.y; aPt.y < aSzW.y; aPt.y = aPt.y + aStep)
        {
            Pt2dr aPC1 = aPt1 + aPt;
            Pt2dr aPC2 = aPt2 + aPt;
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

            aSys.AddEquation(1.0,mCoeff,aV2);
*/
            mCoeff[0] = aV2 ; // A
            mCoeff[1] = 1.0 ; // B
            mCoeff[2] = aGr2X; // im00.x
            mCoeff[3] = aGr2Y;  // im00.y

            aSys.AddEquation(1.0,mCoeff,aV1-aV2);
        }
    }
    bool OK = false;
    aSol = aSys.Solve(&OK);
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
               );
         cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
         cout<<"Lire Img Target"<<endl;
         cImgMatch * aImgTarget = new cImgMatch(aImg, anICNM);
         aImgTarget->Load();
         cout<<"Lire Img Template"<<endl;
         cImgMatch * aImgTmplt = new cImgMatch(aTmpl, anICNM);
         aImgTmplt->Load();
         cInterpolBilineaire<double> * aInterpolBilin = new cInterpolBilineaire<double>;

         /*================ Corrrelation real ==================*/
         Pt2dr aPt(0.0,0.0);
         //double aStep = 1.0;
         tIm2DM aImgScoreCorrel(aImgTarget->Im2D().sz().x, aImgTarget->Im2D().sz().y);
         bool OK = false;

         Pt2dr aPt1 = Pt2dr(aImgTmplt->Im2D().sz()-Pt2di(1,1))/2;       // (1,1) to have a good pixel index
         Pt2dr aPt2 = Pt2dr(aImgTarget->Im2D().sz()-Pt2di(1,1))/2;
         Pt2dr aSzRech = Pt2dr(aImgTarget->Im2D().sz() - Pt2di(1,1))/2  - Pt2dr(2.0,2.0); // (2,2) for a rab
         Pt2dr aSzW = Pt2dr(aImgTmplt->Im2D().sz()-Pt2di(1,1))/2;       // Sz win not include center pixel




         cout<<"P1 : "<<aPt1<<" -P2 : "<<aPt2<<" -SzWin : "<<aSzW<<" -SzRech :"<<aSzRech<<endl;

         cResulRechCorrel aResCorrel =    dblTst_Correl
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


         /*================= LSQ =====================*/
         // Do refine matching by LSQ
         cout<<"Create matching"<<endl;
         aImgTarget->GetImget(aResCorrel.mPt, aImgTmplt->SzIm());
         cLSQMatch * aMatch = new cLSQMatch(aImgTmplt, aImgTarget);
         aMatch->Param() = aParam;

    for (int aK=0; aK<aParam.mNbIter; aK++)
    {
         Im1D_REAL8 aSol(4);

         aMatch->MatchbyLSQ (
                                 aPt1,
                                 aImgTmplt->Im2D(),
                                 aImgTarget->Im2D(),
                                 aResCorrel.mPt,
                                 Pt2di(aImgTmplt->Im2D().sz()/2),
                                 aParam.mStepLSQ,
                                 aSol
                             );



         double * aResLSQ = aSol.data();
         cout<<"==== Iter "<<"["<<aK<<"]"<<" ====="<<endl;
         cout<<"    **A: "<<aResLSQ[0]<<" -B: "<<aResLSQ[1]<<" -TrX: "<<aResLSQ[2]<<" -TrY: "<<aResLSQ[3]<<endl;
         cout<<"    **Before : "<<aResCorrel.mPt<<" - After LSQ: "<<aResCorrel.mPt + Pt2dr(aResLSQ[2], aResLSQ[3])<<endl;

         //aMatch->DoMatchbyLSQ();
         // update matched result
         aResCorrel.mPt = aResCorrel.mPt + Pt2dr(aResLSQ[2], aResLSQ[3]);
    }
    return EXIT_SUCCESS;
}
