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



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe à
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
aooter-MicMac-eLiSe-25/06/2007*/

//use co-registered orientation or DSM to predict key points in another image
void PredictKeyPt(std::vector<Pt2dr>& aVPredL, std::vector<Siftator::SiftPoint> aVSiftL, std::string aDSMFileL, std::string aDSMDirL, std::string aNameOriL, std::string aNameOriR, cTransform3DHelmert aTrans3DH)
{
    bool bDSM = false;

    cGet3Dcoor a3DL(aNameOriL);
    TIm2D<float,double> aTImProfPxL(a3DL.GetDSMSz(aDSMFileL, aDSMDirL));
    if(aDSMDirL.length() > 0)
    {
        bDSM = true;
        aTImProfPxL = a3DL.SetDSMInfo(aDSMFileL, aDSMDirL);
    }
    cGet3Dcoor a3DR(aNameOriR);

    int nSizeL = aVSiftL.size();

    double dGSD1 = a3DL.GetGSD();
     cout<<"GSD of first image: "<<dGSD1<<endl;

    //printf("---------------------\n");
    for(int i=0; i<nSizeL; i++)
    {
        Pt2dr aPL = Pt2dr(aVSiftL[i].x, aVSiftL[i].y);
        Pt3dr aPTer1;
        if(bDSM == true)
        {
            bool bPreciseL;
            aPTer1 = a3DL.Get3Dcoor(aPL, aTImProfPxL, bPreciseL, dGSD1);
        }
        else
        {
            aPTer1 = a3DL.GetRough3Dcoor(aPL);
        }

        /*
        if(i>0.45*nSizeL && i<0.46*nSizeL)
            printf("%dth: %lf, %lf, %lf; ", i, aPTer1.x, aPTer1.y, aPTer1.z);
        aPTer1 = aTrans3DH.Transform3Dcoor(aPTer1);
        if(i>0.45*nSizeL && i<0.46*nSizeL)
            printf("%lf, %lf, %lf; ", aPTer1.x, aPTer1.y, aPTer1.z);

        if(i == 49139)
            printf("%lf, %lf, %lf; ", aPTer1.x, aPTer1.y, aPTer1.z);
*/
        aPTer1 = aTrans3DH.Transform3Dcoor(aPTer1);
        Pt2dr aPLPred = a3DR.Get2Dcoor(aPTer1);

/*
        //if(i>0.45*nSizeL && i<0.46*nSizeL)
        //if(aPLPred.x >0 && aPLPred.x<1700 && aPLPred.y >0 && aPLPred.y<1700)
        if(i == 49139)
        {
            printf("%dth: ", i);
            printf("%lf, %lf, %lf; ", aPTer1.x, aPTer1.y, aPTer1.z);
            printf("%lf, %lf, %lf, %lf\n", aPL.x, aPL.y, aPLPred.x, aPLPred.y);
        }

        //pick a subregion for test, in order to be faster
        if(i<0.4*nSizeL || i>0.6*nSizeL)
            aPLPred = Pt2dr(-1,-1);
*/
        aVPredL.push_back(aPLPred);
    }
}

//for the key points in one image (first or second image), find their nearest neighbor in another image and record it
void MatchOneWay(std::vector<int>& matchIDL, std::vector<Siftator::SiftPoint> aVSiftL, std::vector<Siftator::SiftPoint> aVSiftR, std::vector<Pt2dr> aVPredL, Pt2di ImgSzR, double dScale, double dAngle, double threshScale, double threshAngle, bool bCheckScale, bool bCheckAngle, double aSearchSpace, bool bPredict, bool bRatioT)
{
    int nSIFT_DESCRIPTOR_SIZE = 128;
    const double dPie2 = 3.14*2;

    int nSizeL = aVSiftL.size();
    int nSizeR = aVSiftR.size();

    int nStartL = 0;
    int nStartR = 0;
    int nEndL = nSizeL;
    int nEndR = nSizeR;

    std::time_t t1 = std::time(nullptr);
    std::cout << std::put_time(std::localtime(&t1), "%Y-%m-%d %H:%M:%S") << std::endl;

    long nSkiped = 0;
    float alpha = 2;

    int i, j, k;
    int nProgress = nSizeL/10;
    for(i=nStartL; i<nEndL; i++)
    {
        if(i%nProgress == 0)
        {
            printf("%.2lf%%\n", i*100.0/nSizeL);
        }

        double dBoxLeft  = 0;
        double dBoxRight = 0;
        double dBoxUpper = 0;
        double dBoxLower = 0;

        double dEuDisMin = DBL_MAX;
        double dEuDisSndMin = DBL_MAX;
        int nMatch = -1;

        double x, y;
        if(bPredict == true)
        {
            x = aVPredL[i].x;
            y = aVPredL[i].y;

            //if predicted point is out of the border of the other image, skip searching
            if(x<0 || x> ImgSzR.x || y<0 || y>ImgSzR.y)
            {
                matchIDL.push_back(-1);
                continue;
            }

            dBoxLeft  = x - aSearchSpace;
            dBoxRight = x + aSearchSpace;
            dBoxUpper = y - aSearchSpace;
            dBoxLower = y + aSearchSpace;
        }

        for(j=nStartR; j<nEndR; j++)
        {
            if(bPredict == true)
            {
                if(aVSiftR[j].x<=dBoxLeft || aVSiftR[j].x>= dBoxRight || aVSiftR[j].y<= dBoxUpper || aVSiftR[j].y>= dBoxLower)
                {
                    nSkiped++;
                    continue;
                }
            }
            if(bCheckScale == true)
            {
                double dScaleDif = fabs(aVSiftR[j].scale/aVSiftL[i].scale - dScale);
                if(dScaleDif > threshScale)
                    continue;
            }
            if(bCheckAngle == true)
            {
                double dAngleDif = fabs(aVSiftR[j].angle-aVSiftL[i].angle - dAngle);
                if((dAngleDif > threshAngle) && (dAngleDif < dPie2-threshAngle))
                    continue;
            }

            double dDis = 0;
            for(k=0; k<nSIFT_DESCRIPTOR_SIZE; k++)
            {
                double dDif = aVSiftL[i].descriptor[k] - aVSiftR[j].descriptor[k];
                dDis += pow(dDif, alpha);
            }
            dDis = pow(dDis, 1.0/alpha);

            //save first and second nearest neigbor
            if(dDis < dEuDisMin)
            {
                dEuDisMin = dDis;
                nMatch = j;
                if(dEuDisMin > dEuDisSndMin)
                {
                    dEuDisSndMin = dEuDisMin;
                }
            }
            else if(dDis < dEuDisSndMin)
            {
                dEuDisSndMin = dDis;
            }
        }

        if(bRatioT == true && dEuDisMin/dEuDisSndMin > 0.8)
            nMatch = -1;
        matchIDL.push_back(nMatch);
    }
    std::time_t t2 = std::time(nullptr);
    std::cout << std::put_time(std::localtime(&t2), "%Y-%m-%d %H:%M:%S") << std::endl;
}

void MutualNearestNeighbor(bool bMutualNN, std::vector<int> matchIDL, std::vector<int> matchIDR, std::vector<Pt2di> & match)
{
    int nStartL = 0;
    int nStartR = 0;
    int nEndL = matchIDL.size();
    int nEndR = matchIDR.size();

    int i, j;
    if (bMutualNN == true){
        printf("Mutual nearest neighbor applied.\n");
        for(i=nStartL; i<nEndL; i++)
        {
            j = matchIDL[i-nStartL];

            if(j-nStartR < 0 || j-nStartR >= nEndR)
                 continue;
            if(matchIDR[j-nStartR] == i)
            {
                    Pt2di mPair = Pt2di(i, j);
                    match.push_back(mPair);
            }
        }
    }
    else
    {
        printf("Mutual nearest neighbor NOT applied.\n");
        for(i=nStartL; i<nEndL; i++)
        {
            j = matchIDL[i-nStartL];
            if(j-nStartR < 0 || j-nStartR >= nEndR)
                 continue;
            Pt2di mPair = Pt2di(i, j);
            match.push_back(mPair);

            //if the current pair is not mutual, save the other pair
            int nMatch4j = matchIDR[j-nStartR];
            if(nMatch4j != i && nMatch4j >= nStartL && nMatch4j-nStartL<nEndL)
            {
                Pt2di mPair = Pt2di(i, j);
                match.push_back(mPair);
            }
        }
    }
}

//transform the descriptor to rootSIFT if neccessary
void AmendSIFTKeyPt(std::vector<Siftator::SiftPoint>& aVSiftL, bool aRootSift)
{
    int nSizeL = aVSiftL.size();

    int nSIFT_DESCRIPTOR_SIZE = 128;
    for(int i=0; i<nSizeL; i++)
    {
        if(aRootSift == true)
        {
            double dSum = 0;
            for(int j=0; j<nSIFT_DESCRIPTOR_SIZE; j++)
                dSum += aVSiftL[i].descriptor[j];
            for(int j=0; j<nSIFT_DESCRIPTOR_SIZE; j++)
            {
                aVSiftL[i].descriptor[j] = sqrt(aVSiftL[i].descriptor[j]/dSum);
            }
        }
    }
}

void GuidedSIFTMatch(std::string aDir,std::string aImg1, std::string aImg2, std::string outSH, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR, std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, bool bRootSift, double aSearchSpace, bool bPredict, bool bRatioT, bool bMutualNN, cTransform3DHelmert aTrans3DHL, cTransform3DHelmert aTrans3DHR, bool bCheckScale, bool bCheckAngle, double dScale=1, double dAngle=0)
{
    if (ELISE_fp::exist_file(aImg1) == false || ELISE_fp::exist_file(aImg2) == false)
    {
        cout<<aImg1<<" or "<<aImg2<<" didn't exist, hence skipped"<<endl;
        return;
    }

    Tiff_Im aRGBIm1(aImg1.c_str());
    Pt2di ImgSzL = aRGBIm1.sz();
    Tiff_Im aRGBIm2(aImg2.c_str());
    Pt2di ImgSzR = aRGBIm2.sz();

    std::string aImg1Key = aImg1.substr(0, aImg1.rfind(".")) + ".key";
    std::string aImg2Key = aImg2.substr(0, aImg2.rfind(".")) + ".key";

    //*********** 1. read SIFT key-pts
    std::vector<Siftator::SiftPoint> aVSiftL;
    std::vector<Siftator::SiftPoint> aVSiftR;
    if(read_siftPoint_list(aImg1Key,aVSiftL) == false || read_siftPoint_list(aImg2Key,aVSiftR) == false)
    {
        cout<<"Read SIFT of "<<aImg1Key<<" or "<<aImg2Key<<" went wrong."<<endl;
        return;
    }

    cout<<"Key point number of first image: "<<aVSiftL.size()<<endl;
    cout<<"Key point number of second image: "<<aVSiftR.size()<<endl;

    AmendSIFTKeyPt(aVSiftL, bRootSift);
    AmendSIFTKeyPt(aVSiftR, bRootSift);

    //*********** 2. Predict SIFT key-pts
    std::vector<Pt2dr> aVPredL;
    std::vector<Pt2dr> aVPredR;
    std::string aNameOriL = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
    std::string aNameOriR = aICNM->StdNameCamGenOfNames(aOri2, aImg2);
    PredictKeyPt(aVPredL, aVSiftL, aDSMFileL, aDSMDirL, aNameOriL, aNameOriR, aTrans3DHL);
    PredictKeyPt(aVPredR, aVSiftR, aDSMFileR, aDSMDirR, aNameOriR, aNameOriL, aTrans3DHR);

    //*********** 3. match SIFT key-pts
    double threshScale, threshAngle;
    threshScale = 0.05;
    threshAngle = 3.14*10/180;

    if(bCheckScale == true)
        printf("Check scale. dScale: %lf; threshScale: %lf\n", dScale, threshScale);
    else
        printf("won't check scale\n");
    if(bCheckAngle == true)
        printf("Check angle. dAngle: %lf; threshAngle: %lf\n", dAngle, threshAngle);
    else
        printf("won't check angle\n");

    std::vector<int> matchIDL;
    std::vector<int> matchIDR;
    printf("*****processing Left*****\n");
    MatchOneWay(matchIDL, aVSiftL, aVSiftR, aVPredL, ImgSzR, dScale, dAngle, threshScale, threshAngle, bCheckScale, bCheckAngle, aSearchSpace, bPredict, bRatioT);
    printf("*****processing Right*****\n");
    MatchOneWay(matchIDR, aVSiftR, aVSiftL, aVPredR, ImgSzL, dScale, dAngle, threshScale, threshAngle, bCheckScale, bCheckAngle, aSearchSpace, bPredict, bRatioT);

    //cout<<matchIDL.size()<<",,,,"<<matchIDR.size()<<endl;

    std::vector<Pt2di> match;
    MutualNearestNeighbor(bMutualNN, matchIDL, matchIDR, match);

    cout<<"Extracted tie point number: "<<match.size()<<endl;

    std::string aCom = "mm3d SEL" + BLANK + aDir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    cout<<aCom<<endl;

    //*********** 4. Save tie pt
    std::string aSHDir = aDir + "/Homol" + outSH + "/";
    ELISE_fp::MkDir(aSHDir);
    std::string aNewDir = aSHDir + "Pastis" + aImg1;
    ELISE_fp::MkDir(aNewDir);
    std::string aNameFile = aNewDir + "/"+aImg2+".txt";
    FILE * fpTiePt1 = fopen(aNameFile.c_str(), "w");

    aNewDir = aSHDir + "Pastis" + aImg2;
    ELISE_fp::MkDir(aNewDir);
    aNameFile = aNewDir + "/"+aImg1+".txt";
    FILE * fpTiePt2 = fopen(aNameFile.c_str(), "w");
/*
    std::string comSEL = "mm3d SEL " + aDir + " " + aImg1 + " " + aImg2 + " KH=NT SzW=[600,600] SH=" + outSH;
    cout<<comSEL<<endl;
*/
    int nTiePtNum = match.size();
    for(int i = 0; i < nTiePtNum; i++)
    {
        int idxL = match[i].x;
        int idxR = match[i].y;
        Pt2dr aP1, aP2;
        aP1 = Pt2dr(aVSiftL[idxL].x, aVSiftL[idxL].y);
        aP2 = Pt2dr(aVSiftR[idxR].x, aVSiftR[idxR].y);
        fprintf(fpTiePt1, "%lf %lf %lf %lf\n", aP1.x, aP1.y, aP2.x, aP2.y);
        fprintf(fpTiePt2, "%lf %lf %lf %lf\n", aP2.x, aP2.y, aP1.x, aP1.y);
    }
    fclose(fpTiePt1);
    fclose(fpTiePt2);
}

void ExtractSIFT(std::string aFullName, std::string aDir)
{
    cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aFullName));
    cout<<aFullName<<endl;

    //Tiff_Im::StdConvGen(aFullName,1,true,true);
    Tiff_Im::StdConvGen(aFullName,1,false,true);

    std::string aGrayImgName = aFullName + "_Ch1.tif";

    //if RGB image
    if( ELISE_fp::exist_file(aDir + "/Tmp-MM-Dir/" + aGrayImgName) == true)
    {
        std::string aComm;
        aComm = "mv " + aDir + "/Tmp-MM-Dir/" + aGrayImgName + " " + aGrayImgName;
        cout<<aComm<<endl;
        System(aComm);

        aComm = MMBinFile(MM3DStr) + "SIFT " + aGrayImgName;
        cout<<aComm<<endl;
        System(aComm);

        aComm = "mv " + StdPrefix(aGrayImgName)+".key" + " "+StdPrefix(aFullName)+".key";
        cout<<aComm<<endl;
        System(aComm);

        aComm = "rm " + aGrayImgName;
        cout<<aComm<<endl;
        System(aComm);
    }
    //gray image
    else
    {
        std::string aCom = MMBinFile(MM3DStr) + "SIFT " + aFullName;
        cout<<aCom<<endl;
        System(aCom);
    }
}

int GuidedSIFTMatch_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aImg1;
   std::string aImg2;
   std::string aOri1;
   std::string aOri2;

   std::string aDSMDirL = "";
   std::string aDSMDirR = "";
   std::string aDSMFileL;
   std::string aDSMFileR;

   aDSMFileL = "MMLastNuage.xml";
   aDSMFileR = "MMLastNuage.xml";

   std::string aPara3DHL = "";
   std::string aPara3DHR = "";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()   << EAMC(aImg1,"First image name")
               << EAMC(aImg2,"Second image name")
               << EAMC(aOri1,"Orientation of first image")
               << EAMC(aOri2,"Orientation of second image"),
        LArgMain()
                    << aCAS3D.ArgBasic()
                    << aCAS3D.ArgGuidedSIFT()
               << EAM(aDSMDirL, "DSMDirL", true, "DSM of first image (for improving the reprojecting accuracy), Def=none")
               << EAM(aDSMDirR, "DSMDirR", true, "DSM of second image (for improving the reprojecting accuracy), Def=none")
               << EAM(aDSMFileL, "DSMFileL", true, "DSM File of first image, Def=MMLastNuage.xml")
               << EAM(aDSMFileR, "DSMFileR", true, "DSM File of second image, Def=MMLastNuage.xml")
               << EAM(aPara3DHL, "Para3DHL", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of first image to second image, Def=none")
               << EAM(aPara3DHR, "Para3DHR", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of second image to first image, Def=none")

    );
    StdCorrecNameOrient(aOri1,"./",true);
    StdCorrecNameOrient(aOri2,"./",true);
/*
    std::string aKeyOri1 = "NKS-Assoc-Im2Orient@-" + aOri1;
    std::string aKeyOri2 = "NKS-Assoc-Im2Orient@-" + aOri2;

    std::string aNameOriL = aCAS3D.mICNM->Assoc1To1(aKeyOri1,aImg1,true);
    std::string aNameOriR = aCAS3D.mICNM->Assoc1To1(aKeyOri2,aImg2,true);

    cout<<aNameOriL<<endl;
    cout<<aNameOriR<<endl;

    std::string aFullName = aImg2;
    cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aFullName));
    cout<<aFullName<<endl;

    //Tiff_Im::StdConvGen(aFullName,1,true,true);
    Tiff_Im::StdConvGen(aFullName,1,false,true);
    //DoSimplePastisSsResol(aImg2,-1,1);
    cout<<"DoSimplePastisSsResol"<<endl;

    aFullName = aImg1;
    cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aFullName));
    cout<<aFullName<<endl;

    //Tiff_Im::StdConvGen(aFullName,1,true,true);
    Tiff_Im::StdConvGen(aFullName,1,false,true);
    //DoSimplePastisSsResol(aImg2,-1,1);
    cout<<"DoSimplePastisSsResol"<<endl;

    return 0;
*/
   //if SIFT key point is not extracted before (aCAS3D.mSkipSIFT = true), extract SIFT first
   if(aCAS3D.mSkipSIFT == false)
   {
       ExtractSIFT(aImg1, aCAS3D.mDir);
       ExtractSIFT(aImg2, aCAS3D.mDir);
       /*
       std::string aCom = MMBinFile(MM3DStr) + "SIFT " + aImg1;
       cout<<aCom<<endl;
       System(aCom);

       aCom = MMBinFile(MM3DStr) + "SIFT " + aImg2;
       cout<<aCom<<endl;
       System(aCom);
       */
   }

   if (aPara3DHL.length() > 0 && ELISE_fp::exist_file(aPara3DHL) == false)
   {
       printf("File %s does not exist.\n", aPara3DHL.c_str());
        return 0;
    }
   if (aPara3DHR.length() > 0 && ELISE_fp::exist_file(aPara3DHR) == false)
   {
       printf("File %s does not exist.\n", aPara3DHR.c_str());
        return 0;
    }

   cTransform3DHelmert aTrans3DHL(aPara3DHL);
   cTransform3DHelmert aTrans3DHR(aPara3DHR);

   GuidedSIFTMatch( aCAS3D.mDir, aImg1,  aImg2,  aCAS3D.mGuidedSIFTOutSH, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR,  aOri1, aOri2, aCAS3D.mICNM, aCAS3D.mRootSift, aCAS3D.mSearchSpace, aCAS3D.mPredict, aCAS3D.mRatioT, aCAS3D.mMutualNN, aTrans3DHL, aTrans3DHR, aCAS3D.mCheckScale, aCAS3D.mCheckAngle, aCAS3D.mScale, aCAS3D.mAngle);

   return EXIT_SUCCESS;
}
