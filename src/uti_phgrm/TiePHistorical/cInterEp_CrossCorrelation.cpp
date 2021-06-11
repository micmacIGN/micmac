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

void ReadXml1(std::string & aImg1, std::string & aImg2, std::string aSubPatchXml, std::vector<std::string>& vPatchesL, std::vector<std::string>& vPatchesR, std::vector<cElHomographie>& vHomoL, std::vector<cElHomographie>& vHomoR)
{
    cout<<aSubPatchXml<<endl;
    cSetOfPatches aSOMAF = StdGetFromSI(aSubPatchXml, SetOfPatches);

    std::list<cMes1Im>::const_iterator itIms = aSOMAF.Mes1Im().begin();

    cMes1Im aIms1 = * itIms;
    itIms++;
    cMes1Im aIms2 = * itIms;

    aImg1 = aIms1.NameIm();
    aImg2 = aIms2.NameIm();

    for(std::list<cOnePatch1I>::const_iterator itF = aIms1.OnePatch1I().begin() ; itF != aIms1.OnePatch1I().end() ; itF++)
    {
        cOnePatch1I aMAF = *itF;
        vPatchesL.push_back(aMAF.NamePatch());
        vHomoL.push_back(aMAF.PatchH());
    }

    for(std::list<cOnePatch1I>::const_iterator itF = aIms2.OnePatch1I().begin() ; itF != aIms2.OnePatch1I().end() ; itF++)
    {
        cOnePatch1I aMAF = *itF;
        vPatchesR.push_back(aMAF.NamePatch());
        vHomoR.push_back(aMAF.PatchH());
    }
}

//if the window around the point aPt is out of the border of the current patch, move to an adjoint patch
Pt2di VerifyPatch(Pt2dr aPt, Pt2dr aPatchSz, int aWindowSize)
{
    Pt2di aRes(0,0);
    int aBorder = int(aWindowSize/2);

    if(aPt.x - aBorder < 0)
        aRes.x = -1;
    else if(aPt.x + aBorder >= aPatchSz.x)
        aRes.x = 1;

    if(aPt.y - aBorder < 0)
        aRes.y = -1;
    else if(aPt.y + aBorder >= aPatchSz.y)
        aRes.y = 1;

    return aRes;
}

bool GetPxVal(std::string aDir, std::string aImg, int aWindowSize, Pt2dr aPt, std::vector<int>& aPxVal1)
{
    Pt2di aP1InPatch(aPt.x, aPt.y);
    /*
    cout<<"---------------"<<endl;
    cout<<aDir<<endl;
    cout<<aImg<<endl;
    cout<<aP1InPatch.x<<",,,"<<aP1InPatch.y<<endl;
    cout<<"---------------"<<endl;
*/

    int i, j;

    Tiff_Im aRGBPatch1((aDir+"/"+aImg).c_str());
    Pt2di ImgSz = aRGBPatch1.sz();
    TIm2D<U_INT1,INT> aTImProfPx(ImgSz);
    ELISE_COPY
    (
    aTImProfPx.all_pts(),
    aRGBPatch1.in(),
    aTImProfPx.out()
    );
    int aHalfWindow = int(aWindowSize/2);
    //printf("%d %d %d %d %d \n", aHalfWindow, aP1InPatch.x-aHalfWindow, aP1InPatch.x+aHalfWindow, aP1InPatch.y-aHalfWindow, aP1InPatch.y+aHalfWindow);
    for(i=aP1InPatch.x-aHalfWindow; i<aP1InPatch.x+aHalfWindow; i++)
    {
        for(j=aP1InPatch.y-aHalfWindow; j<aP1InPatch.y+aHalfWindow; j++)
        {
            if(i<0 || i>=ImgSz.x || j<0 || j>=ImgSz.y)
                return false;
            int nVal = aTImProfPx.get(Pt2di(i,j));
            aPxVal1.push_back(nVal);
        }
    }

    return true;
}

double GetMean(std::vector<int> aPxVal1)
{
    double nMean1 = 0;

    for(unsigned int i=0; i<aPxVal1.size(); i++)
    {
        nMean1 += aPxVal1[i];
    }
    nMean1 /= aPxVal1.size();

    return nMean1;
}

double CalcCorssCorr(std::vector<int> aPxVal1, std::vector<int> aPxVal2)
{
    double aMean1 = GetMean(aPxVal1);
    double aMean2 = GetMean(aPxVal2);

    //cout<<aMean1<<","<<aMean2<<endl;

    for(unsigned int i=0; i<aPxVal1.size(); i++)
    {
        aPxVal1[i] -= aMean1;
        aPxVal2[i] -= aMean2;
    }

    double aInter = 0;
    double aIntraL = 0;
    double aIntraR = 0;
    for(unsigned int i=0; i<aPxVal1.size(); i++)
    {
        aInter += aPxVal1[i]*aPxVal2[i];
        aIntraL += aPxVal1[i]*aPxVal1[i];
        aIntraR += aPxVal2[i]*aPxVal2[i];
    }

    double dCorr = aInter/pow(aIntraL, 0.5)/pow(aIntraR, 0.5);

    return dCorr;
}

void CrossCorrelation(std::string aDir, std::string outSH, std::string inSH, std::string aSubPatchXml, Pt2dr aPatchSz, Pt2dr aBufferSz, std::string aPatchDir, int aWindowSize, double aThreshold, bool bCheckFile)
{
    if(aPatchSz.x < aWindowSize || aPatchSz.y < aWindowSize)
    {
        cout<<"Patch size is smaller than window size, hence skipped."<<endl;
        return;
    }

    std::string aImg1, aImg2;
    std::vector<std::string> vPatchesL, vPatchesR;
    std::vector<cElHomographie> vHomoL, vHomoR;

    ReadXml1(aImg1, aImg2, aPatchDir+"/"+aSubPatchXml, vPatchesL, vPatchesR, vHomoL, vHomoR);

    // Save tie pt
    std::string aSHDir = aDir + "/Homol" + outSH + "/";
    ELISE_fp::MkDir(aSHDir);
    std::string aNewDir = aSHDir + "Pastis" + aImg1;
    ELISE_fp::MkDir(aNewDir);
    std::string aNameFile1 = aNewDir + "/"+aImg2+".txt";

    aNewDir = aSHDir + "Pastis" + aImg2;
    ELISE_fp::MkDir(aNewDir);
    std::string aNameFile2 = aNewDir + "/"+aImg1+".txt";

    std::string aCom = "mm3d SEL" + BLANK + aDir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
    cout<<aCom<<endl;

    if (bCheckFile == true && ELISE_fp::exist_file(aNameFile1) == true && ELISE_fp::exist_file(aNameFile2) == true)
    {
        cout<<aNameFile1<<" already exist, hence skipped"<<endl;
        return;
    }


    Tiff_Im aRGBIm1((aDir+"/"+aImg1).c_str());
    Pt2di ImgSzL = aRGBIm1.sz();
/*
    TIm2D<U_INT1,INT> aTImProfPx(ImgSzL);

    //Pt2di aSzOut = mDSMSz;
    //TIm2D<float,double> aTImProfPx(aSzOut);
    ELISE_COPY
    (
    aTImProfPx.all_pts(),
    aRGBPatch1.in(),
    aTImProfPx.out()
    );



    int i, j;
    i= 2;
    j = 3;

    //aTImProfPx.getr(i,j);
    aTImProfPx.get(Pt2di(i,j));

    Tiff_Im aRGBIm2((aDir+"/"+aImg2).c_str());
    Pt2di ImgSzR = aRGBIm2.sz();
*/

    Pt2dr CoreaPatchSz;
    CoreaPatchSz.x = aPatchSz.x - aBufferSz.x*2;
    CoreaPatchSz.y = aPatchSz.y - aBufferSz.y*2;

    Pt2dr PatchNum;
    PatchNum.x = ceil(ImgSzL.x*1.0/CoreaPatchSz.x);
    PatchNum.y = ceil(ImgSzL.y*1.0/CoreaPatchSz.y);

    std::string aDir_inSH = aDir + "/Homol" + inSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";
    if (ELISE_fp::exist_file(aNameIn) == false)
    {
        cout<<aNameIn<<"didn't exist hence skipped."<<endl;
        return;
    }
    ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);


    std::vector<ElCplePtsHomologues> inlier;
    int nPtNum = 0;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin(); itCpl!=aPackFull.end() ; itCpl++)
    {
        Pt2dr aP1, aP2;
        aP1 = itCpl->ToCple().P1();
        aP2 = itCpl->ToCple().P2();

        int aIdxX, aIdxY, aIdxL, aIdxR;
        aIdxX = int(aP1.x/CoreaPatchSz.x);
        aIdxY = int(aP1.y/CoreaPatchSz.y);
        aIdxL = aIdxX*PatchNum.y + aIdxY;
        aIdxR = aIdxL;
        cElHomographie  aH1 = vHomoL[aIdxL].Inverse();
        cElHomographie  aH2 = vHomoR[aIdxR].Inverse();
/*
        cout<<aIdxX<<";;;"<<aP1.x<<";;;"<<CoreaPatchSz.x<<endl;
        cout<<aIdxY<<";;;"<<aP1.y<<";;;"<<CoreaPatchSz.y<<endl;
        cout<<aIdxL<<",,,"<<PatchNum.x<<",,,"<<PatchNum.y<<endl;
        cout<<"--------"<<aPatchDir+"/"+vPatchesL[aIdxL]<<", "<<aPatchDir+"/"+vPatchesR[aIdxR]<<endl;
*/
        Pt2dr aP1InPatch, aP2InPatch;
        aP1InPatch = aH1(aP1);
        aP2InPatch = aH2(aP2);


        //printf("%d: original coor: %lf %lf %lf %lf\n", nPtNum, aP1.x, aP1.y, aP2.x, aP2.y);
//        cout<<nPtNum++<<endl;
/*
        printf("original coor: %lf %lf %lf %lf\n", aP1.x, aP1.y, aP2.x, aP2.y);
        printf("new coor: %lf %lf %lf %lf\n", aP1InPatch.x, aP1InPatch.y, aP2InPatch.x, aP2InPatch.y);
        cout<<aPatchDir+"/"+vPatchesL[aIdx]<<", "<<aPatchDir+"/"+vPatchesR[aIdx]<<endl;
*/
        nPtNum++;
        Pt2di res = VerifyPatch(aP2InPatch, aPatchSz, aWindowSize);
        /*
        if(nPtNum == 453)
        {
            cout<<res.x<<"::::::"<<res.y<<endl;
            printf("%d %d %d, %lf %lf\n", aIdxL, aIdxX, aIdxY, CoreaPatchSz.x, CoreaPatchSz.y);
        }
        */

        if(res.x != 0 || res.y !=0)
        {
            aIdxX = aIdxX + res.x;
            aIdxY = aIdxY + res.y;
            aIdxR = aIdxX*PatchNum.y + aIdxY;
            if(aIdxR<0 || aIdxR>=int(vPatchesR.size()))
                continue;

            //aH1 = vHomoL[aIdx].Inverse();
            aH2 = vHomoR[aIdxR].Inverse();

            //aP1InPatch = aH1(aP1);
            aP2InPatch = aH2(aP2);
        }

/*
        std::string aCom = "mm3d ClipIm " + aPatchDir+"/"+vPatchesL[aIdxL]+ " ["+ToString(int(aP1InPatch.x-int(aWindowSize/2)))+","+ToString(int(aP1InPatch.y-int(aWindowSize/2)))+"]"+ " ["+ToString(aWindowSize)+","+ToString(aWindowSize)+"]";
        cout<<aCom<<endl;
        //System(aCom);

        aCom = "mm3d ClipIm " + aPatchDir+"/"+vPatchesR[aIdxR]+ " ["+ToString(int(aP2InPatch.x-int(aWindowSize/2)))+","+ToString(int(aP2InPatch.y-int(aWindowSize/2)))+"]"+ " ["+ToString(aWindowSize)+","+ToString(aWindowSize)+"]";
                cout<<aCom<<endl;
                //System(aCom);
*/

        std::vector<int> aPxVal1, aPxVal2;
        if (ELISE_fp::exist_file(aPatchDir+"/"+vPatchesL[aIdxL]) == true && ELISE_fp::exist_file(aPatchDir+"/"+vPatchesR[aIdxR]) == true)
        {
            bool bRes1 = GetPxVal(aPatchDir, vPatchesL[aIdxL], aWindowSize, aP1InPatch, aPxVal1);
            bool bRes2 = GetPxVal(aPatchDir, vPatchesR[aIdxR], aWindowSize, aP2InPatch, aPxVal2);

            if(bRes1 == false || bRes2 == false)
            {
                printf("------Out of border-------\n %dth pt, Original coor: %lf %lf %lf %lf\n", nPtNum, aP1.x, aP1.y, aP2.x, aP2.y);
                printf("new coor: %lf %lf %lf %lf\n", aP1InPatch.x, aP1InPatch.y, aP2InPatch.x, aP2InPatch.y);
                cout<<aPatchDir+"/"+vPatchesL[aIdxL]<<", "<<aPatchDir+"/"+vPatchesR[aIdxR]<<endl;
                continue;
            }

            double dCorr = CalcCorssCorr(aPxVal1, aPxVal2);

            //cout<<nPtNum<<"th dCorr: "<<dCorr<<endl;

            if(dCorr >= aThreshold)
            {
                inlier.push_back(ElCplePtsHomologues(aP1, aP2));
            }
        }
        else
            printf("%s or %s didn't exist, hence skipped.\n", vPatchesL[aIdxL].c_str(), vPatchesR[aIdxR].c_str());
    }


    FILE * fpTiePt1 = fopen(aNameFile1.c_str(), "w");
    FILE * fpTiePt2 = fopen(aNameFile2.c_str(), "w");

    cout<<"original correspondences: "<<nPtNum<<"; survived correspondences: "<<inlier.size()<<endl;

    for (unsigned int i=0; i<inlier.size(); i++)
    {
       ElCplePtsHomologues cple = inlier[i];
       Pt2dr aP1 = cple.P1();
       Pt2dr aP2 = cple.P2();

       fprintf(fpTiePt1, "%lf %lf %lf %lf\n", aP1.x, aP1.y, aP2.x, aP2.y);
        fprintf(fpTiePt2, "%lf %lf %lf %lf\n", aP2.x, aP2.y, aP1.x, aP1.y);
    }
    fclose(fpTiePt1);
    fclose(fpTiePt2);
}

int CrossCorrelation_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aImg1;
   std::string aImg2;

   std::string aSubPatchXml = "SubPatch.xml";
   std::string aPatchDir = "./Tmp_Patches";

   Pt2dr aPatchSz(640, 480);
   Pt2dr aBufferSz(30, 60);

   bool bCheckFile = false;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aImg1,"First image name")
               << EAMC(aImg2,"Second image name"),
        LArgMain()
                    << aCAS3D.ArgBasic()
                    << aCAS3D.ArgCrossCorrelation()
                   << EAM(aPatchSz, "PatchSz", true, "Patch size, Def=[640, 480]")
                   << EAM(aBufferSz, "BufferSz", true, "Buffer sie, Def=[30, 60]")
                    << EAM(aSubPatchXml, "SubPXml", true, "The xml file name to record the homography between the patch and original image, Def=SubPatch.xml")
                   << EAM(aPatchDir, "PatchDir", true, "The input directory of patches, Def=./Tmp_Patches")
                   << EAM(bCheckFile, "CheckFile", true, "Check if the result files exist (if so, skip), Def=false")

    );
/*
   if(aSubPatchXml.length() == 0)
       aSubPatchXml = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_SubPatch.xml";
*/
   if(aCAS3D.mCrossCorrelationOutSH.length() == 0)
       aCAS3D.mCrossCorrelationOutSH = aCAS3D.mCrossCorrelationInSH + "-CrossCorrelation";

   CrossCorrelation(aCAS3D.mDir, aCAS3D.mCrossCorrelationOutSH, aCAS3D.mCrossCorrelationInSH, aSubPatchXml, aPatchSz, aBufferSz, aPatchDir, aCAS3D.mWindowSize, aCAS3D.mCrossCorrThreshold, bCheckFile);

   return EXIT_SUCCESS;
}
