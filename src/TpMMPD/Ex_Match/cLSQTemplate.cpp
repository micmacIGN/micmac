#include "cLSQTemplate.h"

cLSQMatch::cLSQMatch(cImgMatch * aTmpl, cImgMatch * aImg):
    mTemplate (aTmpl),
    mImg      (aImg),
    mICNM     (aImg->ICNM()),
    mcurrErr  (0.0),
    mImRes    (aImg->SzIm().x, aImg->SzIm().y)
{
    mInterpol = new cInterpolBilineaire<double>;
    cout<<"In const cLSQMatch"<<endl;
}

bool cLSQMatch::DoMatch()
{
    cout<<"Do Match"<<endl;
    mcurrErr = 0.0;
    // for each pixel couple from Template & Target :
    Pt2dr aPt(0,0);
    tIm2DM  ImgIm2D = mImg->Im2D();
    tTIm2DM  ImgTIm2D = mImg->TIm2D();

    tIm2DM  ImgetIm2D = mTemplate->CurImgetIm2D();
    tTIm2DM ImgetTIm2D = mTemplate->CurImgetTIm2D();

    L2SysSurResol aSys(4);  // 4 variable: (A, B, trX, trY)
    //Value Init :
    double h0=0.0;
    double h1=1.0;
    double aCoeff[4];
    double Trx = 0.0;
    double Try = 0.0;
    for (aPt.x=0; aPt.x<ImgetIm2D.sz().x; aPt.x++)
    {
        for (aPt.y=0; aPt.y<ImgetIm2D.sz().y; aPt.y++)
        {
            double aVTmp = ImgetIm2D.Get(aPt, *mInterpol, -1.0);
            Pt3dr aVImg = mInterpol->GetValDer(ImgIm2D.data(), aPt);
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
        h0 = h0 + aDS[0];
        h1 = h1 + aDS[1];
        Trx = Trx + aDS[2];
        Try = Try + aDS[3];
        for (aPtErr.x=0; aPtErr.x<ImgetIm2D.sz().x; aPtErr.x++)
        {
            for (aPtErr.y=0; aPtErr.y<ImgetIm2D.sz().y; aPtErr.y++)
            {
                double aVTmp = ImgetIm2D.Get(aPtErr, *mInterpol, -1.0);
                double aVImg = ImgIm2D.Get(aPtErr + Pt2dr(Trx,Try), *mInterpol, -1.0);
                mcurrErr += ElSquare(aVTmp - h0 - h1*aVImg);
            }
        }
        cout<<"Err : "<<mcurrErr<<" - "<<h0<<" "<<h1<<" "<<Trx<<" "<<Try<<endl;
        return true;
    }
    else
    {
        return false;
    }
}


int LSQMatch_Main(int argc,char ** argv)
{
   string aTmpl ="";
   string aImg = "";
   string aDir = "./";

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aTmpl, "Image Template",  eSAM_IsExistFile)
                     << EAMC(aImg, "Target Image to search for template",  eSAM_IsExistFile),
         LArgMain()   
               );
         cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
         cout<<"Lire Img Target"<<endl;
         cImgMatch * aImgTarget = new cImgMatch(aImg, anICNM);
         cout<<"Lire Img Template"<<endl;
         cImgMatch * aImgTmplt = new cImgMatch(aTmpl, anICNM);
         cout<<"Create matching"<<endl;
         cLSQMatch * aMatch = new cLSQMatch(aImgTmplt, aImgTarget);

         Pt2dr aPt(0,0);
         double aStep = 1.0;
         for (aPt.x=0; aPt.x < aImgTarget->SzIm().x; aPt.x = aPt.x + aStep)
         {
             for (aPt.y=0; aPt.y<aImgTarget->SzIm().y; aPt.y = aPt.y + aStep)
             {
                if (aImgTarget->GetImget(aPt, aImgTmplt->SzIm()))
                    aMatch->DoMatch();
             }
         }

    return EXIT_SUCCESS;
}
