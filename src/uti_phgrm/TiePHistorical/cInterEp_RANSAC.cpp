/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "TiePHistorical.h"

//#include "../../util/affin2d.cpp"
#include <ctime>

extern ElSimilitude SimilRobustInit(const ElPackHomologue & aPackFull,double aPropRan,int aNbTir);

void RANSAC3D(std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, std::string input_dir, std::string aImg1, std::string aImg2, std::string inSH, std::string outSH, int aNbTir, double threshold, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR, bool bPrint, bool bCheckFile, cTransform3DHelmert aTrans3DHL, int nMinPt)
{
    cout<<aImg1<<" "<<aImg2<<endl;
    //printf("iteration number: %d; thresh: %lf\n", aNbTir, threshold);

    bool bInverse = false;
    std::string aDir_inSH = input_dir + "/Homol" + inSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";
        if (ELISE_fp::exist_file(aNameIn) == false)
        {
            aDir_inSH = input_dir + "/Homol" + inSH+"/";
            aNameIn = aDir_inSH +"Pastis" + aImg2 + "/"+aImg1+".txt";

            if (ELISE_fp::exist_file(aNameIn) == false)
            {
                cout<<aNameIn<<" didn't exist hence skipped (RANSAC3D)."<<endl;
                return;
            }
            bInverse = true;
        }
        ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    /*
    if (bCheckFile == true && ELISE_fp::exist_file(aNameOut) == true)
    {
        cout<<aNameOut<<" already exist, hence skipped"<<endl;
        return;
    }
    */
        if(IsHomolFileExist(input_dir, aImg1, aImg2, outSH, bCheckFile) == true)
            return;

    std::vector<Pt3dr> aV1;
    std::vector<Pt3dr> aV2;
    std::vector<Pt2dr> a2dV1;
    std::vector<Pt2dr> a2dV2;
    std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    std::string aIm2OriFile = aICNM->StdNameCamGenOfNames(aOri2, aImg2);
    cGet3Dcoor a3DCoorL(aIm1OriFile);
    //TIm2D<float,double> aTImProfPxL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cDSMInfo aDSMInfoL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cGet3Dcoor a3DCoorR(aIm2OriFile);
    //TIm2D<float,double> aTImProfPxR = a3DCoorR.SetDSMInfo(aDSMFileR, aDSMDirR);
    cDSMInfo aDSMInfoR = a3DCoorR.SetDSMInfo(aDSMFileR, aDSMDirR);

    double dGSD1 = a3DCoorL.GetGSD();
    double dGSD2 = a3DCoorR.GetGSD();
    double dRefGSD = dGSD2;
    cout<<"GSD of master image: "<<dGSD1<<endl;
    if(aTrans3DHL.GetApplyTrans() == true){
         cout<<"GSD of master image after transformation: "<<dGSD1*aTrans3DHL.GetScale()<<endl;
         //dRefGSD = (dGSD1*aTrans3DHL.GetScale()+dGSD2)*0.5;
    }
     cout<<"GSD of secondary image: "<<dGSD2<<endl;

     if(threshold < 0){
         threshold = 10*dRefGSD;
         //printf("GSD1: %.2lf, GSD2: %.2lf, RefGSD: %.2lf, 3DRANTh: %.2lf\n", dGSD1*aTrans3DHL.GetScale(), dGSD2, dRefGSD, threshold);
     }

    //std::vector<int> aValidPt;
    //ElPackHomologue aPackInsideBorder;
    //transform 2D tie points into 3D

     int nOriPtNum = Get3DTiePt(aPackFull, a3DCoorL, a3DCoorR, aDSMInfoL, aDSMInfoR, aTrans3DHL, aV1, aV2, a2dV1, a2dV2, bPrint, bInverse);

    if(bPrint)
    {
        printf("Finished transforming %d tie points into 3D.\n", nOriPtNum);
    }

    int nPtNum = aV1.size();
    cout<<"nOriPtNum: "<<nOriPtNum<<";  InsideBorderPtNum:  "<<nPtNum;
    printf(";  iteration number: %d; thresh: %lf\n", aNbTir, threshold);

    if(nPtNum<nMinPt)
    {
        printf("InsideBorderPtNum (%d) is less than %d, hence skipped.\n", nPtNum, nMinPt);
        return;
    }

    srand((int)time(0));
    std::vector<ElCplePtsHomologues> inlierFinal;
    RANSAC3DCore(aNbTir, threshold, aV1, aV2, a2dV1, a2dV2, inlierFinal);

    int nMaxInlier = inlierFinal.size();
    cout<<"---------------------------------"<<endl;
    printf("--->>>Total OriPt: %d; Total InsideBorderPt: %d;; Total inlier: %d; Inlier Ratio (3DRANSAC): %.2lf%%\n", nOriPtNum, nPtNum, nMaxInlier, nMaxInlier*100.0/nPtNum);

    SaveHomolTxtFile(input_dir, aImg1, aImg2, outSH, inlierFinal, false);

    std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    std::string aComInv = "mm3d SEL" + BLANK + input_dir + BLANK + aImg2 + BLANK + aImg1 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    cout<<aCom<<endl<<aComInv<<endl<<"nOriPtNum: "<<nOriPtNum<<";  InsideBorderPtNum:  "<<nPtNum<<";  nFilteredPtNum: "<<inlierFinal.size()<<endl;

    return;
}

bool CheckSclRot(double aSclL, double aRotL, double aSclR, double aRotR, double dScale, double threshScale, double dAngle, double threshAngle)
{
    const double d2PI = 3.1415926*2;
    SetAngleToValidRange(dAngle, d2PI);
    double dScaleRatio = aSclR/aSclL;
    double dAngleDif = aRotR - aRotL;
    SetAngleToValidRange(dAngleDif, d2PI);

    if((dScaleRatio < dScale*(1-threshScale)) || (dScaleRatio > dScale*(1+threshScale)))
        return false;

    if((dAngleDif < dAngle-threshAngle) || (dAngleDif > dAngle+threshAngle))
        return false;

    return true;
}

void RANSAC2D(std::string input_dir, std::string aImg1, std::string aImg2, std::string inSH, std::string outSH, int aNbTir, double thresh, bool bCheckSclRot, double threshScale, double threshAngle, int nMinPt)
{
    printf("iteration number: %d; thresh: %lf\n", aNbTir, thresh);

    std::string aDir_inSH = input_dir + "/Homol" + inSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";

    bool bInverse = false;
    if (ELISE_fp::exist_file(aNameIn) == false)
    {
        aDir_inSH = input_dir + "/Homol" + inSH+"/";
        aNameIn = aDir_inSH +"Pastis" + aImg2 + "/"+aImg1+".txt";
        if (ELISE_fp::exist_file(aNameIn) == false)
        {
            cout<<aNameIn<<" didn't exist hence skipped (RANSAC2D)."<<endl;
            return;
        }
        bInverse = true;
    }
    ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    /******************************Read scale and rotation**********/
    std::vector<Pt2dr> aSclRotV1;
    std::vector<Pt2dr> aSclRotV2;
    if(bCheckSclRot){
        std::string aNameSclRot = input_dir + "/Homol" + inSH+"_SclRot"+"/" +"Pastis" + aImg1 + "/"+aImg2+".txt";

        if (ELISE_fp::exist_file(aNameSclRot) == false)
        {
            cout<<aNameSclRot<<" didn't exist hence skipped (RANSAC2D)."<<endl;
            return;
        }

        ElPackHomologue aPackHomoSclRot =  ElPackHomologue::FromFile(aNameSclRot);
        for (ElPackHomologue::iterator itCpl=aPackHomoSclRot.begin();itCpl!=aPackHomoSclRot.end(); itCpl++)
        {
           ElCplePtsHomologues cple = itCpl->ToCple();
           aSclRotV1.push_back(cple.P1());
           aSclRotV2.push_back(cple.P2());
        }
    }

    /******************************Read tie points**********/
    std::vector<Pt2dr> aV1;
    std::vector<Pt2dr> aV2;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();

       if(bInverse == false)
       {
           aV1.push_back(cple.P1());
           aV2.push_back(cple.P2());
       }
       else
       {
           aV2.push_back(cple.P1());
           aV1.push_back(cple.P2());
       }
    }

    int i, j;
    int nPtNum = aV1.size();
    //int nMinPt = 5;

    cout<<"Input tie point number: "<<nPtNum;
    printf(";  iteration number: %d; thresh: %lf\n", aNbTir, thresh);

    if(nPtNum<nMinPt)
    {
        printf("Input tie point number (%d) is less than %d, hence skipped.\n", nPtNum, nMinPt);
        return;
    }

    int nMaxInlier = 0;
    srand((int)time(0));

    std::vector<ElCplePtsHomologues> inlierCur;
    std::vector<ElCplePtsHomologues> inlierFinal;

    ElSimilitude aSim;

    bool bConsis;
    double dScale = 1;
    double dAngle = 0;
    double aSclL, aRotL, aSclR, aRotR;

    double aEpslon = 0.0001;
    //std::string aFinalMsg = "";
    for(j=0; j<aNbTir; j++)
    {
        ElPackHomologue aPackSeed;
        std::vector<int> res;

        Pt2dr aDiff1, aDiff2;
        do
        {
            res.clear();
            GetRandomNum(0, nPtNum, 2, res);
            aDiff1 = aV1[res[0]] - aV1[res[1]];
            aDiff2 = aV2[res[0]] - aV2[res[1]];
        }
        //in case duplicated points
        while((fabs(aDiff1.x) < aEpslon && fabs(aDiff1.y) < aEpslon) || (fabs(aDiff2.x) < aEpslon && fabs(aDiff2.y) < aEpslon));
        //while{(aV1[res[0]].x - aV1[res[1]].x)};
        //printf("%dth seed: %d, %d\n", j, res[0], res[1]);

        /********************Check if the 2 seed points are consistent in scale and rotation********************/
        if(bCheckSclRot == true)
        {
            i = 0;
            aSclL = aSclRotV1[res[i]].x;
            aRotL = aSclRotV1[res[i]].y;
            aSclR = aSclRotV2[res[i]].x;
            aRotR = aSclRotV2[res[i]].y;
            dScale = aSclR/aSclL;
            dAngle = aRotR - aRotL;
            i = 1;
            aSclL = aSclRotV1[res[i]].x;
            aRotL = aSclRotV1[res[i]].y;
            aSclR = aSclRotV2[res[i]].x;
            aRotR = aSclRotV2[res[i]].y;
            bConsis = CheckSclRot(aSclL, aRotL, aSclR, aRotR, dScale, threshScale, dAngle, threshAngle);
            if(bConsis == false)
                continue;

            dScale = (dScale + aSclR/aSclL)/2;
            dAngle = (dAngle + aRotR - aRotL)/2;
        }

        Pt2dr tr, sc;

        for(i=0; i<2; i++)
        {
            aPackSeed.Cple_Add(ElCplePtsHomologues(aV1[res[i]],aV2[res[i]]));
        }
        double aPropRan = 0.8;
/*
        std::string aTmp = input_dir + "/Homol-SIFT2Step-Rough-GlobalR3D/" +"Pastis" + aImg1 + "/"+aImg2+".txt";
        aPackSeed =  ElPackHomologue::FromFile(aTmp);
*/
        ElSimilitude aSimCur = SimilRobustInit(aPackSeed,aPropRan,1);

        tr = aSimCur.tr();
        sc = aSimCur.sc();
        //printf("inter: %d; translation: %lf  %lf  %lf  %lf\n", j, tr.x, tr.y, sc.x, sc.y);

        int nInlier =0;
        for(i=0; i<nPtNum; i++)
        {
            Pt2dr aP1 = aV1[i];
            Pt2dr aP2 = aV2[i];

            Pt2dr aP2Pred = aSimCur(aP1);
            double dist = pow(pow(aP2Pred.x-aP2.x,2) + pow(aP2Pred.y-aP2.y,2), 0.5);
            //printf("%d %lf\n", i, dist);
            bConsis = true;
            if(bCheckSclRot == true)
            {
                aSclL = aSclRotV1[i].x;
                aRotL = aSclRotV1[i].y;
                aSclR = aSclRotV2[i].x;
                aRotR = aSclRotV2[i].y;
                bConsis = CheckSclRot(aSclL, aRotL, aSclR, aRotR, dScale, threshScale, dAngle, threshAngle);
            }
            if(dist < thresh && bConsis == true)
            {
               // printf("%dth: %d, %.2lf  %.2lf  %.2lf  %.2lf\n", i, bConsis, aSclL, aRotL, aSclR, aRotR);
                inlierCur.push_back(ElCplePtsHomologues(aP1, aP2));
                nInlier++;
            }
        }
        if(nInlier > nMaxInlier)
        {
            nMaxInlier = nInlier;
            aSim = aSimCur;
            inlierFinal = inlierCur;

            //char aCh[1024];
            printf("Iter: %d/%d; seed: [%d,%d]; Translation: [%.2lf, %.2lf]; Scl: %.2lf; Rot: %.2lf;  nInlier/nPtNum: %d/%d\n", j, aNbTir, res[0], res[1], tr.x, tr.y, sc.x, sc.y, nMaxInlier, nPtNum);
            //aFinalMsg = aCh;
            //cout<<aFinalMsg<<endl;
            //printf("Iter: %d/%d; nMaxInlier/nPtNum: %d/%d; ", j, aNbTir, nMaxInlier, nPtNum);
            //printf("translation: %lf  %lf  %lf  %lf, seed: [%d,%d]\n", tr.x, tr.y, sc.x, sc.y, res[0], res[1]);
        }
        /*
        else{
            printf("Iter: %d/%d, seed: %d, %d, %d;  ", j, aNbTir, res[0], res[1], res[2]);
            printf(" nMaxInlier: %d, nOriPtNum: %d\n", nMaxInlier, nOriPtNum);
        }
        */
        inlierCur.clear();
    }
    /******************************end random perform**********/

    std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    std::string aComInv = "mm3d SEL" + BLANK + input_dir + BLANK + aImg2 + BLANK + aImg1 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    cout<<"---------------------------------"<<endl;
    printf("%s\n%s\n--->>>Total Pt: %d; Total inlier: %d; Inlier Ratio (2DRANSAC): %.2lf%%\n", aCom.c_str(), aComInv.c_str(), nPtNum, nMaxInlier, nMaxInlier*100.0/nPtNum);
    //cout<<"Final: "<<aFinalMsg<<endl;

    /****************Save points****************/
    SaveHomolTxtFile(input_dir, aImg1, aImg2, outSH, inlierFinal, false);
    /*
    std::string aDir_outSH = input_dir + "/Homol" + outSH+"/";
    ELISE_fp::MkDir(aDir_outSH);
    aDir_outSH = aDir_outSH + "Pastis" + aImg1;
    ELISE_fp::MkDir(aDir_outSH);
    std::string aNameOut = aDir_outSH + "/"+aImg2+".txt";

    cout<<"Output: "<<aNameOut<<endl;
    FILE * fpOutput = fopen(aNameOut.c_str(), "w");
    for (unsigned int i=0; i<inlierFinal.size(); i++)
    {
       ElCplePtsHomologues cple = inlierFinal[i];
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       fprintf(fpOutput, "%lf %lf %lf %lf\n",p1.x,p1.y,p2.x,p2.y);
    }
    fclose(fpOutput);
    */

    /*
    //ElSimilitude aSim = SimilRobustInit(aPackFull,aPropRan,aNbTir);

    int nOriPtNum = 0;
    int nFilteredPtNum = 0;
    FILE * fpOutput = fopen(aNameOut.c_str(), "w");
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();
       Pt2dr p2Pred = aSim(p1);

       nOriPtNum++;

       if(fabs(p2.x-p2Pred.x)<thresh && fabs(p2.y-p2Pred.y)<thresh)
       {
           fprintf(fpOutput, "%lf %lf %lf %lf\n",p1.x,p1.y,p2.x,p2.y);
           nFilteredPtNum++;
       }
    }
    fclose(fpOutput);

    cout<<"nOriPtNum: "<<nOriPtNum<<";  nFilteredPtNum: "<<nFilteredPtNum<<endl;
    */

    return;
}


void RANSACHomography(std::string input_dir, std::string aImg1, std::string aImg2, std::string inSH, std::string outSH, int aNbTir, double thresh, bool bCheckSclRot, double threshScale, double threshAngle, int nMinPt)
{
    printf("iteration number: %d; thresh: %lf\n", aNbTir, thresh);

    std::string aDir_inSH = input_dir + "/Homol" + inSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";

    bool bInverse = false;
    if (ELISE_fp::exist_file(aNameIn) == false)
    {
        aNameIn = aDir_inSH +"Pastis" + aImg2 + "/"+aImg1+".txt";
        if (ELISE_fp::exist_file(aNameIn) == false)
        {
            cout<<aNameIn<<" didn't exist hence skipped (RANSAC2D)."<<endl;
            return;
        }
        bInverse = true;
    }
    ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    /******************************Read tie points**********/
    std::vector<Pt2dr> aV1;
    std::vector<Pt2dr> aV2;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();

       if(bInverse == false)
       {
           aV1.push_back(cple.P1());
           aV2.push_back(cple.P2());
       }
       else
       {
           aV2.push_back(cple.P1());
           aV1.push_back(cple.P2());
       }
    }

    int i, j;
    int nPtNum = aV1.size();
    //int nMinPt = 5;

    cout<<"Input tie point number: "<<nPtNum;
    printf(";  iteration number: %d; thresh: %lf\n", aNbTir, thresh);

    if(nPtNum<nMinPt)
    {
        printf("Input tie point number (%d) is less than %d, hence skipped.\n", nPtNum, nMinPt);
        return;
    }

    int nMaxInlier = 0;
    srand((int)time(0));

    std::vector<ElCplePtsHomologues> inlierCur;
    std::vector<ElCplePtsHomologues> inlierFinal;
/*
    ElSimilitude aSim;

    bool bConsis;
    double dScale = 1;
    double dAngle = 0;
    double aSclL, aRotL, aSclR, aRotR;
*/
    double aEpslon = 0.0001;
    //std::string aFinalMsg = "";
    for(j=0; j<aNbTir; j++)
    {
        ElPackHomologue aPackSeed;
        std::vector<int> res;

        Pt2dr aDiff1, aDiff2;
        do
        {
            res.clear();
            GetRandomNum(0, nPtNum, 4, res);
            aDiff1 = aV1[res[0]] - aV1[res[1]];
            aDiff2 = aV2[res[0]] - aV2[res[1]];
        }
        //in case duplicated points
        while((fabs(aDiff1.x) < aEpslon && fabs(aDiff1.y) < aEpslon) || (fabs(aDiff2.x) < aEpslon && fabs(aDiff2.y) < aEpslon));
        //while{(aV1[res[0]].x - aV1[res[1]].x)};
        //printf("%dth seed: %d, %d\n", j, res[0], res[1]);

        //Pt2dr tr, sc;

        for(i=0; i<4; i++)
        {
            aPackSeed.Cple_Add(ElCplePtsHomologues(aV1[res[i]],aV2[res[i]]));
        }
        double anEcart,aQuality;
        bool Ok;
        cElHomographie aH1To2 = cElHomographie::RobustInit(anEcart,&aQuality,aPackSeed,Ok,50,80.0,2000);

        int nInlier =0;
        for(i=0; i<nPtNum; i++)
        {
            Pt2dr aP1 = aV1[i];
            Pt2dr aP2 = aV2[i];

            Pt2dr aP2Pred = aH1To2(aP1);
            double dist = pow(pow(aP2Pred.x-aP2.x,2) + pow(aP2Pred.y-aP2.y,2), 0.5);
            //printf("%d %lf\n", i, dist);
            if(dist < thresh)
            {
               // printf("%dth: %d, %.2lf  %.2lf  %.2lf  %.2lf\n", i, bConsis, aSclL, aRotL, aSclR, aRotR);
                inlierCur.push_back(ElCplePtsHomologues(aP1, aP2));
                nInlier++;
            }
        }
        if(nInlier > nMaxInlier)
        {
            nMaxInlier = nInlier;
            //aSim = aSimCur;
            inlierFinal = inlierCur;

            //char aCh[1024];
            printf("---Iter: %d/%d; seed: [%d,%d,%d,%d]; nInlier/nPtNum: %d/%d\n", j, aNbTir, res[0], res[1], res[2], res[3], nMaxInlier, nPtNum);
            aH1To2.Show();
            //aFinalMsg = aCh;
            //cout<<aFinalMsg<<endl;
            //printf("Iter: %d/%d; nMaxInlier/nPtNum: %d/%d; ", j, aNbTir, nMaxInlier, nPtNum);
            //printf("translation: %lf  %lf  %lf  %lf, seed: [%d,%d]\n", tr.x, tr.y, sc.x, sc.y, res[0], res[1]);
        }
        /*
        else{
            printf("Iter: %d/%d, seed: %d, %d, %d;  ", j, aNbTir, res[0], res[1], res[2]);
            printf(" nMaxInlier: %d, nOriPtNum: %d\n", nMaxInlier, nOriPtNum);
        }
        */
        inlierCur.clear();
    }
    /******************************end random perform**********/

    std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    std::string aComInv = "mm3d SEL" + BLANK + input_dir + BLANK + aImg2 + BLANK + aImg1 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    cout<<"---------------------------------"<<endl;
    printf("%s\n%s\n--->>>Total Pt: %d; Total inlier: %d; Inlier Ratio (2DRANSAC): %.2lf%%\n", aCom.c_str(), aComInv.c_str(), nPtNum, nMaxInlier, nMaxInlier*100.0/nPtNum);
    //cout<<"Final: "<<aFinalMsg<<endl;

    /****************Save points****************/
    std::string aSaveTxt = SaveHomolTxtFile(input_dir, aImg1, aImg2, outSH, inlierFinal, false);

    ElPackHomologue aPackFinal =  ElPackHomologue::FromFile(aSaveTxt);
    double anEcart,aQuality;
    bool Ok;
    cElHomographie aHomo = cElHomographie::RobustInit(anEcart,&aQuality,aPackFinal,Ok,50,80.0,2000);
    cout<<"Final homography"<<endl;
    aHomo.Show();

    return;
}

int R2D(int argc,char ** argv, const std::string &aArg="")
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aImg1;
    std::string aImg2;

    std::string aStrType;

    bool bCheckSclRot = false;
    double aThreshScale = 0.2;
    double aThreshAngle = 30;

    int aMinPt = 5;

    int aTransType = 1;

    ElInitArgMain
     (
         argc,argv,
         LArgMain()  << EAMC(aStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeRHP))
                << EAMC(aImg1,"Master image name")
                << EAMC(aImg2,"Secondary image name"),
         LArgMain()
                     << aCAS3D.ArgBasic()
                     << aCAS3D.Arg2DRANSAC()
                     << EAM(bCheckSclRot, "CheckSclRot", true, "Check the scale and rotation consistency (please make sure you saved the scale and rotation in \"Homol'2DRANInSH'_SclRot\" if you set this parameter to true), Def=false")
                     << EAM(aThreshScale, "ScaleTh",true, "The threshold for checking scale ratio, Def=0.2; (0.2 means the ratio of master and secondary SIFT scale between [(1-0.2)*Ref, (1+0.2)*Ref] is considered valide. Ref is automatically calculated by reprojection.)")
                     << EAM(aThreshAngle, "AngleTh",true, "The threshold for checking angle difference, Def=30; (30 means the difference of master and secondary SIFT angle between [Ref - 30 degree, Ref + 30 degree] is considered valide. Ref is automatically calculated by reprojection.)")
                     << EAM(aMinPt,"MinPt",true,"Minimun number of input correspondences required, Def=5")
                     << EAM(aTransType,"TransType",true,"Type of transformation model (1 means 2D similarity, 2 means homography), Def=1")
     );

    aThreshAngle = aThreshAngle*3.14/180;

    if(aCAS3D.mR2DOutSH.length() == 0)
        aCAS3D.mR2DOutSH = aCAS3D.mR2DInSH + "-2DRANSAC";

    if(aTransType == 2)
        RANSACHomography(aCAS3D.mDir, aImg1, aImg2, aCAS3D.mR2DInSH, aCAS3D.mR2DOutSH, aCAS3D.mR2DIteration, aCAS3D.mR2DThreshold, bCheckSclRot, aThreshScale, aThreshAngle, aMinPt);
    else
        RANSAC2D(aCAS3D.mDir, aImg1, aImg2, aCAS3D.mR2DInSH, aCAS3D.mR2DOutSH, aCAS3D.mR2DIteration, aCAS3D.mR2DThreshold, bCheckSclRot, aThreshScale, aThreshAngle, aMinPt);

    return 0;
}

int R3D(int argc,char ** argv, const std::string &aArg="")
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aImg1;
    std::string aImg2;

    std::string aOri1;
    std::string aOri2;

    std::string aStrType;

    std::string aDSMDirL;
    std::string aDSMDirR;
    std::string aDSMFileL;
    std::string aDSMFileR;

    aDSMFileL = "MMLastNuage.xml";
    aDSMFileR = "MMLastNuage.xml";

    std::string aPara3DHL = "";
    bool bCheckFile = false;

    ElInitArgMain
     (
         argc,argv,
         LArgMain()  << EAMC(aStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeRHP))
                << EAMC(aImg1,"Master image name")
                << EAMC(aImg2,"Secondary image name")
                << EAMC(aOri1,"Orientation of master image")
                << EAMC(aOri2,"Orientation of secondary image"),
         LArgMain()
                     << aCAS3D.ArgBasic()
                     << aCAS3D.Arg3DRANSAC()
                << EAM(aDSMDirL, "DSMDirL", true, "DSM directory of master image, Def=none")
                << EAM(aDSMDirR, "DSMDirR", true, "DSM directory of secondary image, Def=none")
                << EAM(aDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
                << EAM(aDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml")
                << EAM(aPara3DHL, "Para3DHL", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master image to secondary image, Def=none")
                   << EAM(bCheckFile, "CheckFile", true, "Check if the result files of inter-epoch correspondences exist (if so, skip to avoid repetition), Def=false")
     );

    if(aCAS3D.mR3DOutSH.length() == 0)
        aCAS3D.mR3DOutSH = aCAS3D.mR3DInSH + "-3DRANSAC";

    //RANSAC3D(aCAS3D.mOri, aCAS3D.mDir, aImg1, aImg2, aCAS3D.mR3DInSH, aCAS3D.mR3DOutSH, aCAS3D.mIteration, aR3DThreshold, aCAS3D.mDSMFileL, aCAS3D.mDSMFileR, aCAS3D.mDSMDirL, aCAS3D.mDSMDirR);

    StdCorrecNameOrient(aOri1,"./",true);
    StdCorrecNameOrient(aOri2,"./",true);
/*
     std::string aKeyOri1 = "NKS-Assoc-Im2Orient@-" + aOri1;
     std::string aKeyOri2 = "NKS-Assoc-Im2Orient@-" + aOri2;

     std::string aIm1OriFile = aCAS3D.mICNM->Assoc1To1(aKeyOri1,aImg1,true);
     std::string aIm2OriFile = aCAS3D.mICNM->Assoc1To1(aKeyOri2,aImg2,true);
*/

    cTransform3DHelmert aTrans3DHL(aPara3DHL);
    RANSAC3D(aOri1, aOri2, aCAS3D.mICNM, aCAS3D.mDir, aImg1, aImg2, aCAS3D.mR3DInSH, aCAS3D.mR3DOutSH, aCAS3D.mR3DIteration, aCAS3D.mR3DThreshold, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR, aCAS3D.mPrint, bCheckFile, aTrans3DHL, aCAS3D.mMinPt);

    return 0;
}

int RANSAC_main(int argc,char ** argv)
{

    bool aModeHelp=true;
    eRANSAC_HistoP aType=eNbTypeRHP;
    StdReadEnum(aModeHelp,aType,argv[1],eNbTypeRHP);

    std::string TheType = argv[1];

    if (TheType == "R2D")
    {
        int aRes = R2D(argc, argv, TheType);
        return aRes;
    }
    else if (TheType == "R3D")
    {
        int aRes = R3D(argc, argv, TheType);
        return aRes;
    }
    return EXIT_SUCCESS;
}

