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

std::vector<int> TiePtEvaluation(std::string aIm1OriFile, std::string aIm2OriFile, std::string aDir,std::string aImg1, std::string aImg2, std::string inSH, std::string aDSMFileL, std::string aDSMDirL, cTransform3DHelmert aTrans3DH, int nThreshMax, bool bPrint)
{
    std::vector<int> aStat;

    std::string aDir_inSH = aDir + "/Homol" + inSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";
        if (ELISE_fp::exist_file(aNameIn) == false)
        {
            cout<<aNameIn<<"didn't exist hence skipped."<<endl;
            return aStat;
        }
        ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    std::vector<Pt2dr> a2dPRpred;
    std::vector<Pt2dr> a2dPR;
    std::vector<double> aReproj;
    cGet3Dcoor a3DCoorL(aIm1OriFile);
    //TIm2D<float,double> aTImProfPxL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cDSMInfo aDSMInfoL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cGet3Dcoor a3DCoorR(aIm2OriFile);
    //TIm2D<float,double> aTImProfPxR = a3DCoorR.SetDSMInfo(aDSMFileR, aDSMDirR);

    int nOriPtNum = 0;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       //cout<<nTodel<<"th tie pt: "<<p1.x<<" "<<p1.y<<" "<<p2.x<<" "<<p2.y<<endl;

       bool bValidL;
       Pt3dr aPTer1 = a3DCoorL.Get3Dcoor(p1, aDSMInfoL, bValidL, bPrint);//, a3DCoorL.GetGSD());

       aPTer1 = aTrans3DH.Transform3Dcoor(aPTer1);
       Pt2dr aPLPred = a3DCoorR.Get2Dcoor(aPTer1);


       if(bValidL == true)
       {
           a2dPRpred.push_back(aPLPred);
           a2dPR.push_back(p2);

           double dist = pow(pow(aPLPred.x-p2.x,2) + pow(aPLPred.y-p2.y,2), 0.5);
           aReproj.push_back(dist);
       }
       else
       {
           if(false)
               cout<<nOriPtNum<<"th tie pt out of border of the DSM hence skipped"<<endl;
       }
       nOriPtNum++;
    }

    int nSize = aReproj.size();
    int nNb = 0;
    for(int nThresh=0; nThresh<nThreshMax; nThresh++)
    {
        nNb = 0;
        for(int j=0; j<nSize; j++)
        {
            if(aReproj[j] < nThresh)
                nNb++;
        }
        aStat.push_back(nNb);
    }
    aStat.push_back(nOriPtNum);

    if(true)
    {
        printf("%s\t%s\n", aImg1.c_str(), aImg2.c_str());
        printf("The point number with reprojection error under %d is: %d; TotalPtNum: %d\n", nThreshMax, nNb, nOriPtNum);
    }
    return aStat;
}

int TiePtEvaluation_main(int argc,char ** argv)
{
//    std::string aFullPattern1;
//    std::string aFullPattern2;
    std::string aImgList1;
    std::string aImgList2;

    cCommonAppliTiepHistorical aCAS3D;
    //std::string aDir = "./";

   std::string aImg1;
   std::string aImg2;

   std::string aOri1;
   std::string aOri2;

   std::string aDSMDirL;
   std::string aDSMFileL = "MMLastNuage.xml";
   //std::string aDSMDirR = "";
   //std::string aDSMFileR = "MMLastNuage.xml";

   std::string aPara3DHL;

   std::string aInSH;

   int nThreshMax = 10;

   std::string aNameOut = "";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()
               << EAMC(aImgList1,"ImgList1: All master images (Dir+Pattern, or txt file of image list)")
               << EAMC(aImgList2,"ImgList2: All secondary images (Dir+Pattern, or txt file of image list)")
//               << EAMC(aFullPattern1,"Master image name (Dir+Pattern)")
//               << EAMC(aFullPattern2,"Secondary image name (Dir+Pattern)")
               << EAMC(aOri1,"Orientation of master image")
               << EAMC(aOri2,"Orientation of secondary image")
               << EAMC(aDSMDirL,"DSM of master image"),
        LArgMain()
               << aCAS3D.ArgBasic()
               //<< EAM(aDir,"Dir",true,"Work directory, Def=./")
               << EAM(aDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
//               << EAM(aDSMDirR, "DSMDirR", true, "DSM of secondary image, Def=\"DSM of master image\"")
//               << EAM(aD    int nSize = aReproj.size();
               << EAM(aPara3DHL, "Para3DHL", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master image to secondary image, Def=unit matrix")
               << EAM(aInSH,"InSH",true, "Input Homologue extenion for NB/NT mode, Def=none")
               << EAM(nThreshMax, "Th", true, "The max threshold of reprojection error, Def=10")
               << EAM(aNameOut,"NameOut",true, "Output txt file that records the accuracy, Def=TiePtAccuracy-InSH.txt")

    );

   if(aNameOut.length() == 0)
   {
       aNameOut = aCAS3D.mDir + "TiePtAccuracy" + aInSH + ".txt";
   }

   cTransform3DHelmert aTrans3DHL(aPara3DHL);

   std::vector<std::string> aSetIm1;
   std::vector<std::string> aSetIm2;
   GetImgListVec(aImgList1, aSetIm1);
   GetImgListVec(aImgList2, aSetIm2);
   /*
   std::string aDirImages1,aPatIm1;
   SplitDirAndFile(aDirImages1,aPatIm1,aFullPattern1);

   cInterfChantierNameManipulateur * aICNM1=cInterfChantierNameManipulateur::BasicAlloc(aDirImages1);
   const std::vector<std::string> aSetIm1 = *(aICNM1->Get(aPatIm1));

   std::string aDirImages2,aPatIm2;
   SplitDirAndFile(aDirImages2,aPatIm2,aFullPattern2);

   cInterfChantierNameManipulateur * aICNM2=cInterfChantierNameManipulateur::BasicAlloc(aDirImages2);
   const std::vector<std::string> aSetIm2 = *(aICNM2->Get(aPatIm2));
   */

   //std::vector<std::string> aVIm1;
   //std::vector<std::string> aVIm2;

   std::vector<int> aStatTotal;
   //The last one is the total point number
   for(int k=0; k<nThreshMax+1; k++)
       aStatTotal.push_back(0);

   for (unsigned int i=0;i<aSetIm1.size();i++)
   {
       //std::cout<<" - "<<aSetIm1[i]<<std::endl;
       aImg1 = aSetIm1[i];
       for (unsigned int j=0;j<aSetIm2.size();j++)
       {
           //std::cout<<" - "<<aSetIm2[i]<<std::endl;
           aImg2 = aSetIm2[j];
           StdCorrecNameOrient(aOri1,"./",true);
           StdCorrecNameOrient(aOri2,"./",true);

            std::string aKeyOri1 = "NKS-Assoc-Im2Orient@-" + aOri1;
            std::string aKeyOri2 = "NKS-Assoc-Im2Orient@-" + aOri2;

            std::string aIm1OriFile = aCAS3D.mICNM->Assoc1To1(aKeyOri1,aImg1,true);
            std::string aIm2OriFile = aCAS3D.mICNM->Assoc1To1(aKeyOri2,aImg2,true);

           std::vector<int> aStat = TiePtEvaluation(aIm1OriFile, aIm2OriFile, aCAS3D.mDir, aImg1, aImg2, aInSH, aDSMFileL, aDSMDirL, aTrans3DHL, nThreshMax, aCAS3D.mPrint);

           if(int(aStat.size()) > nThreshMax)
               for(int k=0; k<nThreshMax+1; k++)
                   aStatTotal[k] += aStat[k];
       }
   }

   FILE * fpOutput = fopen(aNameOut.c_str(), "w");
   for(int nThresh=0; nThresh<nThreshMax+1; nThresh++)
   {
       fprintf(fpOutput, "%d %d\n", nThresh, aStatTotal[nThresh]);
   }
   fclose(fpOutput);

   printf("The total point number with reprojection error under %d is: %d; TotalPtNum: %d\n", nThreshMax, aStatTotal[nThreshMax-1], aStatTotal[nThreshMax]);
   printf("xdg-open %s\n", aNameOut.c_str());

   return EXIT_SUCCESS;
}
