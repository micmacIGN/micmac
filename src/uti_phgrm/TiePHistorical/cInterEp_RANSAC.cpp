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
    //printf("iteration number: %d; thresh: %lf\n", aNbTir, threshold);

    std::string aDir_inSH = input_dir + "/Homol" + inSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";
        if (ELISE_fp::exist_file(aNameIn) == false)
        {
            cout<<aNameIn<<"didn't exist hence skipped."<<endl;
            return;
        }
        ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    std::string aDir_outSH = input_dir + "/Homol" + outSH+"/";
    ELISE_fp::MkDir(aDir_outSH);
    aDir_outSH = aDir_outSH + "Pastis" + aImg1;
    ELISE_fp::MkDir(aDir_outSH);
    std::string aNameOut = aDir_outSH + "/"+aImg2+".txt";

    if (bCheckFile == true && ELISE_fp::exist_file(aNameOut) == true)
    {
        cout<<aNameOut<<" already exist, hence skipped"<<endl;
        return;
    }
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
    cout<<"GSD of master image: "<<dGSD1<<endl;
    if(aTrans3DHL.GetApplyTrans() == true)
         cout<<"GSD of master image after transformation: "<<dGSD1*aTrans3DHL.GetScale()<<endl;
     cout<<"GSD of secondary image: "<<dGSD2<<endl;

     if(threshold < 0)
         threshold = 10*dGSD2;

    int nOriPtNum = 0;
    std::vector<int> aValidPt;
    ElPackHomologue aPackInsideBorder;
    //transform 2D tie points into 3D
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       if(bPrint)
           cout<<nOriPtNum<<"th tie pt: "<<p1.x<<" "<<p1.y<<" "<<p2.x<<" "<<p2.y<<endl;

       bool bValidL, bValidR;
       Pt3dr pTerr1 = a3DCoorL.Get3Dcoor(p1, aDSMInfoL, bValidL, bPrint);//, dGSD1);
       pTerr1 = aTrans3DHL.Transform3Dcoor(pTerr1);
       Pt3dr pTerr2 = a3DCoorR.Get3Dcoor(p2, aDSMInfoR, bValidR, bPrint);//, dGSD2);

       if(bValidL == true && bValidR == true)
       {
           aV1.push_back(pTerr1);
           aV2.push_back(pTerr2);
           a2dV1.push_back(p1);
           a2dV2.push_back(p2);
           aPackInsideBorder.Cple_Add(cple);
           aValidPt.push_back(nOriPtNum);
       }
       else
       {
           if(false)
               cout<<nOriPtNum<<"th tie pt out of border of the DSM hence skipped"<<endl;
       }
       nOriPtNum++;
    }

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

    cSolBasculeRig aSBR = cSolBasculeRig::Id();
    cSolBasculeRig aSBRBest = cSolBasculeRig::Id();
    int i, j;
    int nMaxInlier = 0;
    srand((int)time(0));

    std::vector<ElCplePtsHomologues> inlierCur;
    std::vector<ElCplePtsHomologues> inlierFinal;

    for(j=0; j<aNbTir; j++)
    {
        cRansacBasculementRigide aRBR(false);

        std::vector<int> res;

        Pt3dr aDiff;
        double aEpslon = 0.0000001;
        bool bDupPt;
        //in case duplicated points
        do
        {
            res.clear();
            bDupPt = false;
            GetRandomNum(0, nPtNum, 3, res);
            for(i=0; i<3; i++)
            {
                aDiff = aV1[res[i]] - aV1[res[(i+1)%3]];
                if((fabs(aDiff.x) < aEpslon) && (fabs(aDiff.y) < aEpslon) && (fabs(aDiff.z) < aEpslon))
                {
                    bDupPt = true;
                    //printf("Duplicated 3D pt seed: %d, %d; Original index of 2D pt: %d %d\n ", res[i], res[i+1], aValidPt[res[i]], aValidPt[res[i+1]]);
                    break;
                }
                aDiff = aV2[res[i]] - aV2[res[(i+1)%3]];
                if((fabs(aDiff.x) < aEpslon) && (fabs(aDiff.y) < aEpslon) && (fabs(aDiff.z) < aEpslon))
                {
                    bDupPt = true;
                    //printf("Duplicated 3D pt seed: %d, %d; Original index of 2D pt: %d %d\n ", res[i], res[i+1], aValidPt[res[i]], aValidPt[res[i+1]]);
                    break;
                }
            }
        }
        while(bDupPt == true);

        for(i=0; i<3; i++)
        {
            aRBR.AddExemple(aV1[res[i]],aV2[res[i]],0,"");
            inlierCur.push_back(ElCplePtsHomologues(a2dV1[res[i]], a2dV2[res[i]]));
        }

        aRBR.CloseWithTrGlob();
        aRBR.ExploreAllRansac();
        aSBR = aRBR.BestSol();

        int nInlier =3;
        ElPackHomologue::iterator itCpl=aPackInsideBorder.begin();
        for(i=0; i<nPtNum; i++)
        {
            Pt3dr aP1 = aV1[i];
            Pt3dr aP2 = aV2[i];

            Pt3dr aP2Pred = aSBR(aP1);
            double dist = pow(pow(aP2Pred.x-aP2.x,2) + pow(aP2Pred.y-aP2.y,2) + pow(aP2Pred.z-aP2.z,2), 0.5);
            //printf("%d %lf\n", i, dist);
            if(dist < threshold)
            {
                inlierCur.push_back(itCpl->ToCple());
                nInlier++;
            }
            itCpl++;
        }
        if(nInlier > nMaxInlier)
        {
            nMaxInlier = nInlier;
            aSBRBest = aSBR;
            inlierFinal = inlierCur;
            printf("Iter: %d/%d, seed: %d, %d, %d;  ", j, aNbTir, res[0], res[1], res[2]);
            printf(" nMaxInlier: %d, nOriPtNum: %d\n", nMaxInlier, nOriPtNum);
        }
        /*
        else{
            printf("Iter: %d/%d, seed: %d, %d, %d;  ", j, aNbTir, res[0], res[1], res[2]);
            printf(" nMaxInlier: %d, nOriPtNum: %d\n", nMaxInlier, nOriPtNum);
        }
        */
        inlierCur.clear();
    }

    FILE * fpOutput = fopen(aNameOut.c_str(), "w");
    for (unsigned int i=0; i<inlierFinal.size(); i++)
    {
       ElCplePtsHomologues cple = inlierFinal[i];
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       fprintf(fpOutput, "%lf %lf %lf %lf\n",p1.x,p1.y,p2.x,p2.y);
    }
    fclose(fpOutput);

    std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    std::string aComInv = "mm3d SEL" + BLANK + input_dir + BLANK + aImg2 + BLANK + aImg1 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    cout<<aCom<<endl<<aComInv<<endl<<"nOriPtNum: "<<nOriPtNum<<";  InsideBorderPtNum:  "<<nPtNum<<";  nFilteredPtNum: "<<inlierFinal.size()<<endl;

    return;
}

void RANSAC2D(std::string input_dir, std::string aImg1, std::string aImg2, std::string inSH, std::string outSH, int aNbTir, double thresh)
{
    printf("iteration number: %d; thresh: %lf\n", aNbTir, thresh);

    std::string aDir_inSH = input_dir + "/Homol" + inSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";

    if (ELISE_fp::exist_file(aNameIn) == false)
    {
        cout<<aNameIn<<"didn't exist hence skipped."<<endl;
        return;
    }
    ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    /******************************random perform**********/
    std::vector<Pt2dr> aV1;
    std::vector<Pt2dr> aV2;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();

       aV1.push_back(cple.P1());
       aV2.push_back(cple.P2());
    }

    int i, j;
    int nPtNum = aV1.size();

    int nMaxInlier = 0;
    srand((int)time(0));

    std::vector<ElCplePtsHomologues> inlierCur;
    std::vector<ElCplePtsHomologues> inlierFinal;

    ElSimilitude aSim;

    double aEpslon = 0.0001;
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

        Pt2dr tr, sc;

        for(i=0; i<2; i++)
        {
            aPackSeed.Cple_Add(ElCplePtsHomologues(aV1[res[i]],aV2[res[i]]));
        }
        double aPropRan = 0.8;
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
            if(dist < thresh)
            {
                inlierCur.push_back(ElCplePtsHomologues(aP1, aP2));
                nInlier++;
            }
        }
        if(nInlier > nMaxInlier)
        {
            nMaxInlier = nInlier;
            aSim = aSimCur;
            inlierFinal = inlierCur;
            printf("Iter: %d/%d; nMaxInlier/nPtNum: %d/%d; ", j, aNbTir, nMaxInlier, nPtNum);
            printf("translation: %lf  %lf  %lf  %lf, seed: [%d,%d]\n", tr.x, tr.y, sc.x, sc.y, res[0], res[1]);
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
    printf("%s\n%s\n", aCom.c_str(), aComInv.c_str());

    /****************Save points****************/
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

int R2D(int argc,char ** argv, const std::string &aArg="")
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aImg1;
    std::string aImg2;

    std::string aStrType;

    ElInitArgMain
     (
         argc,argv,
         LArgMain()  << EAMC(aStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeRHP))
                << EAMC(aImg1,"Master image name")
                << EAMC(aImg2,"Secondary image name"),
         LArgMain()
                     << aCAS3D.ArgBasic()
                     << aCAS3D.Arg2DRANSAC()
     );

    if(aCAS3D.mR2DOutSH.length() == 0)
        aCAS3D.mR2DOutSH = aCAS3D.mR2DInSH + "-2DRANSAC";

    RANSAC2D(aCAS3D.mDir, aImg1, aImg2, aCAS3D.mR2DInSH, aCAS3D.mR2DOutSH, aCAS3D.mR2DIteration, aCAS3D.mR2DThreshold);

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

