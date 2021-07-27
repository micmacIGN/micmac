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

void GetRandomNum(int nMin, int nMax, int nNum, std::vector<int> & res)
{
    //srand((int)time(0));
    int idx = 0;
    for(int i=0; i<nNum; i++)
    {
        bool bRepeat = false;
        int nIter = 0;
        do
        {
            bRepeat = false;
            nIter++;
            idx = rand() % (nMax - nMin) + nMin;
            //printf("For %dth seed, %dth generation, random value: %d\n", i, nIter, idx);
            for(int j=0; j<int(res.size()); j++)
            {
                if(idx == res[j]){
                    bRepeat = true;
                    break;
                }
            }
        }
        while(bRepeat == true);
        res.push_back(idx);
    }
}

void RANSAC3D(std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, std::string input_dir, std::string aImg1, std::string aImg2, std::string inSH, std::string outSH, int aNbTir, double threshold, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR)
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

        std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
        cout<<aCom<<endl;

    std::string aDir_outSH = input_dir + "/Homol" + outSH+"/";
    ELISE_fp::MkDir(aDir_outSH);
    aDir_outSH = aDir_outSH + "Pastis" + aImg1;
    ELISE_fp::MkDir(aDir_outSH);
    std::string aNameOut = aDir_outSH + "/"+aImg2+".txt";

    std::vector<Pt3dr> aV1;
    std::vector<Pt3dr> aV2;
    std::vector<Pt2dr> a2dV1;
    std::vector<Pt2dr> a2dV2;
    std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    std::string aIm2OriFile = aICNM->StdNameCamGenOfNames(aOri2, aImg2);
    cGet3Dcoor a3DCoorL(aIm1OriFile);
    TIm2D<float,double> aTImProfPxL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cGet3Dcoor a3DCoorR(aIm2OriFile);
    TIm2D<float,double> aTImProfPxR = a3DCoorR.SetDSMInfo(aDSMFileR, aDSMDirR);

    double dGSD1 = a3DCoorL.GetGSD();
    double dGSD2 = a3DCoorR.GetGSD();
     cout<<"GSD of first image: "<<dGSD1<<endl;
     cout<<"GSD of second image: "<<dGSD2<<endl;

     if(threshold < 0)
         threshold = 10*dGSD2;

    int nOriPtNum = 0;
    std::vector<int> aValidPt;
    ElPackHomologue aPackInsideBorder;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       //cout<<nTodel<<"th tie pt: "<<p1.x<<" "<<p1.y<<" "<<p2.x<<" "<<p2.y<<endl;

       bool bValidL, bValidR;
       Pt3dr pTerr1 = a3DCoorL.Get3Dcoor(p1, aTImProfPxL, bValidL, dGSD1);
       Pt3dr pTerr2 = a3DCoorR.Get3Dcoor(p2, aTImProfPxR, bValidR, dGSD2);

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

    int nPtNum = aV1.size();
    cout<<"nOriPtNum: "<<nOriPtNum<<";  InsideBorderPtNum:  "<<nPtNum;
    printf(";  iteration number: %d; thresh: %lf\n", aNbTir, threshold);

    if(nPtNum<3)
    {
        printf("InsideBorderPtNum (%d) is less than 3, hence skipped.\n", nPtNum);
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

    cout<<"nOriPtNum: "<<nOriPtNum<<";  InsideBorderPtNum:  "<<nPtNum<<";  nFilteredPtNum: "<<inlierFinal.size()<<endl;

    return;
}

void RANSAC2D(std::string input_dir, std::string aImg1, std::string aImg2, std::string inSH, std::string outSH, int aNbTir, double thresh)
{
    double aPropRan = 0.8;

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

        Pt2dr tr, sc;
/*
        res[0] = 427;
        res[1] = 449;
        tr = aV1[res[0]];
        sc = aV1[res[1]];
        printf("inter: %d; translation: %lf  %lf  %lf  %lf\n", j, tr.x, tr.y, sc.x, sc.y);
        tr = aV2[res[0]];
        sc = aV2[res[1]];
        printf("inter: %d; translation: %lf  %lf  %lf  %lf\n", j, tr.x, tr.y, sc.x, sc.y);

        Pt2dr ttt = aV1[res[0]] - aV1[res[1]];
        cout<<ttt.x<<",,,,,"<<ttt.y<<endl;
*/
        for(i=0; i<2; i++)
        {
            aPackSeed.Cple_Add(ElCplePtsHomologues(aV1[res[i]],aV2[res[i]]));
        }
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

    /****************Save points****************/
    std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    cout<<aCom<<endl;

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
                << EAMC(aImg1,"First image name")
                << EAMC(aImg2,"Second image name"),
         LArgMain()
                     << aCAS3D.ArgBasic()
                     << aCAS3D.Arg2DRANSAC()
     );

    if(aCAS3D.mRANSACOutSH.length() == 0)
        aCAS3D.mRANSACOutSH = aCAS3D.mRANSACInSH + "-2DRANSAC";

    RANSAC2D(aCAS3D.mDir, aImg1, aImg2, aCAS3D.mRANSACInSH, aCAS3D.mRANSACOutSH, aCAS3D.mR2DIteration, aCAS3D.mR2DThreshold);

    return 0;
}
/*
double GetGSD(std::string aImg1, std::string aOri1, cInterfChantierNameManipulateur * aICNM)
{
    int aType = eTIGB_Unknown;
    std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    cBasicGeomCap3D * aCam1 = cBasicGeomCap3D::StdGetFromFile(aIm1OriFile,aType);
    double dZL = aCam1->GetAltiSol();

    Pt2dr aCent(double(aCam1->SzBasicCapt3D().x)/2,double(aCam1->SzBasicCapt3D().y)/2);
    Pt2dr aCentNeigbor(aCent.x+1, aCent.y);

    Pt3dr aCentTer = aCam1->ImEtZ2Terrain(aCent, dZL);
    Pt3dr aCentNeigborTer = aCam1->ImEtZ2Terrain(aCentNeigbor, dZL);


    //double dist = pow(pow(aCentTer.x-aCentNeigborTer.x,2) + pow(aCentTer.y-aCentNeigborTer.y,2) + pow(aCentTer.z-aCentNeigborTer.z,2), 0.5);

    double dist = pow(pow(aCentTer.x-aCentNeigborTer.x,2) + pow(aCentTer.y-aCentNeigborTer.y,2), 0.5);

    return dist;
}
*/
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

    ElInitArgMain
     (
         argc,argv,
         LArgMain()  << EAMC(aStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeRHP))
                << EAMC(aImg1,"First image name")
                << EAMC(aImg2,"Second image name")
                << EAMC(aOri1,"Orientation of first image")
                << EAMC(aOri2,"Orientation of second image"),
         LArgMain()
                     << aCAS3D.ArgBasic()
                     << aCAS3D.Arg3DRANSAC()
                << EAM(aDSMDirL, "DSMDirL", true, "DSM directory of first image, Def=none")
                << EAM(aDSMDirR, "DSMDirR", true, "DSM directory of second image, Def=none")
                << EAM(aDSMFileL, "DSMFileL", true, "DSM File of first image, Def=MMLastNuage.xml")
                << EAM(aDSMFileR, "DSMFileR", true, "DSM File of second image, Def=MMLastNuage.xml")

     );

    if(aCAS3D.mRANSACOutSH.length() == 0)
        aCAS3D.mRANSACOutSH = aCAS3D.mRANSACInSH + "-3DRANSAC";

    //RANSAC3D(aCAS3D.mOri, aCAS3D.mDir, aImg1, aImg2, aCAS3D.mRANSACInSH, aCAS3D.mRANSACOutSH, aCAS3D.mIteration, aR3DThreshold, aCAS3D.mDSMFileL, aCAS3D.mDSMFileR, aCAS3D.mDSMDirL, aCAS3D.mDSMDirR);

    StdCorrecNameOrient(aOri1,"./",true);
    StdCorrecNameOrient(aOri2,"./",true);
/*
     std::string aKeyOri1 = "NKS-Assoc-Im2Orient@-" + aOri1;
     std::string aKeyOri2 = "NKS-Assoc-Im2Orient@-" + aOri2;

     std::string aIm1OriFile = aCAS3D.mICNM->Assoc1To1(aKeyOri1,aImg1,true);
     std::string aIm2OriFile = aCAS3D.mICNM->Assoc1To1(aKeyOri2,aImg2,true);
*/

    RANSAC3D(aOri1, aOri2, aCAS3D.mICNM, aCAS3D.mDir, aImg1, aImg2, aCAS3D.mRANSACInSH, aCAS3D.mRANSACOutSH, aCAS3D.mR3DIteration, aCAS3D.mR3DThreshold, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR);

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

/*
int RANSAC_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aImg1;
   std::string aImg2;


   std::string aStrType;

   if(argc < 2)
   {
       cout<<"not enough parameters"<<endl;
       return 0;
   }

   //cout<<argv[1]<<endl;

   //ReadType(argv[1]);
   bool aModeHelp=true;
   eRANSAC_HistoP aType=eNbTypeRHP;
   StdReadEnum(aModeHelp,aType,argv[1],eNbTypeRHP);

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeRHP))
               << EAMC(aImg1,"First image name")
               << EAMC(aImg2,"Second image name"),
        LArgMain()
                    << aCAS3D.ArgBasic()
               //???
                    << aCAS3D.ArgRANSAC()
    );

   if(aCAS3D.mRANSACOutSH.length() == 0)
       aCAS3D.mRANSACOutSH = aCAS3D.mRANSACInSH + "-RANSAC";

   if(aStrType == "R2D")
       RANSAC2D(aCAS3D.mDir, aImg1, aImg2, aCAS3D.mRANSACInSH, aCAS3D.mRANSACOutSH, aCAS3D.mIteration, aCAS3D.mRANSACThreshold);

   if(aStrType == "R3D")
       RANSAC3D(aCAS3D.mOri, aCAS3D.mDir, aImg1, aImg2, aCAS3D.mRANSACInSH, aCAS3D.mRANSACOutSH, aCAS3D.mIteration, aCAS3D.mRANSACThreshold, aCAS3D.mDSMFileL, aCAS3D.mDSMFileR, aCAS3D.mDSMDirL, aCAS3D.mDSMDirR);

   return EXIT_SUCCESS;
}
*/
