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

void GlobalR3D(std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, std::string input_dir, std::vector<std::string> aVIm1, std::vector<std::string> aVIm2, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR, std::string inSH, std::string outSH, cTransform3DHelmert aTrans3DHL, int aNbTir, double threshold, int nMinPt, bool bPrint, bool bSaveHomol, bool bSaveGCP, std::string aInBascFile)
{
    /*
     std::vector<std::string> aVIm1;
     std::vector<std::string> aVIm2;

     if (ELISE_fp::exist_file(input_dir+"/"+aImgPair) == true)
     {
         GetXmlImgPair(input_dir+"/"+aImgPair, aVIm1, aVIm2);
     }
     else
     {
         GetImgListVec(aImgList1, aVIm1);
         GetImgListVec(aImgList2, aVIm2);
     }
     */

     std::vector<Pt3dr> aV1;
     std::vector<Pt3dr> aV2;
     std::vector<Pt2dr> a2dV1;
     std::vector<Pt2dr> a2dV2;

     std::vector<int> aOriPtNumV;
     std::vector<int> aInsideBorderPtNumV;

     double dGSDAve1 = 0;
     double dGSDAve2 = 0;
     int nNum = 0;

     int nOriPtNum = 0;
     std::string  aImg1, aImg2;
     aOriPtNumV.push_back(0);
     aInsideBorderPtNumV.push_back(0);
     for(int i=0; i<int(aVIm1.size()); i++)
     {
         aImg1 = aVIm1[i];
         if(1) //for(int j=0; j<int(aVIm2.size()); j++)
         {
             aImg2 = aVIm2[i];
             //cout<<i<<", "<<j<<"; "<<aImg1<<" "<<aImg2<<endl;
             std::string aDir_inSH = input_dir + "/Homol" + inSH+"/";
             std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";
             if (ELISE_fp::exist_file(aNameIn) == false)
             {
                 //cout<<aNameIn<<"didn't exist hence skipped (GlobalR3D)."<<endl;
                 aOriPtNumV.push_back(nOriPtNum);
                 aInsideBorderPtNumV.push_back(int(aV1.size()));
                 continue;
             }
             ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);
             //aPackAll.push_back(aPackFull);

             std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
             std::string aIm2OriFile = aICNM->StdNameCamGenOfNames(aOri2, aImg2);
             //cout<<aIm1OriFile<<" "<<aIm2OriFile<<endl;
             cGet3Dcoor a3DCoorL(aIm1OriFile);
             cDSMInfo aDSMInfoL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
             cGet3Dcoor a3DCoorR(aIm2OriFile);
             cDSMInfo aDSMInfoR = a3DCoorR.SetDSMInfo(aDSMFileR, aDSMDirR);

             int nOriPtNumCur = Get3DTiePt(aPackFull, a3DCoorL, a3DCoorR, aDSMInfoL, aDSMInfoR, aTrans3DHL, aV1, aV2, a2dV1, a2dV2, bPrint);
             nOriPtNum += nOriPtNumCur;
             aOriPtNumV.push_back(nOriPtNum);
             aInsideBorderPtNumV.push_back(int(aV1.size()));
             cout<<aImg1<<" "<<aImg2<<": ";
             cout<<"nOriPtNum: "<<nOriPtNumCur<<";  InsideBorderPtNum:  "<<aInsideBorderPtNumV[aInsideBorderPtNumV.size()-1] - aInsideBorderPtNumV[aInsideBorderPtNumV.size()-2]<<"; ";
             double dGSD1 = a3DCoorL.GetGSD();
             double dGSD2 = a3DCoorR.GetGSD();
             printf("dGSD1: %.5lf; dGSD2: %.5lf\n", dGSD1, dGSD2);

             if(false)
             {
                 int nSz = int(aInsideBorderPtNumV.size());
                 int nStart = aInsideBorderPtNumV[nSz-2];
                 int nEnd = aInsideBorderPtNumV[nSz-1];

                 for(int k=nStart; k<nEnd; k++)
                 {
                     Pt3dr aP1 = aV1[k];
                     Pt3dr aP2 = aV2[k];

                     Pt3dr aP2Pred = aP1;
                     printf("%dth: PtL: [%.2lf, %.2lf], PtR: [%.2lf, %.2lf]\n", i, a2dV1[k].x, a2dV1[k].y, a2dV2[k].x, a2dV2[k].y);
                     printf("%dth: PtL: [%.2lf, %.2lf, %.2lf], PtR: [%.2lf, %.2lf, %.2lf], PtRPred: [%.2lf, %.2lf, %.2lf]\n", i, aP1.x, aP1.y, aP1.z, aP2.x, aP2.y, aP2.z, aP2Pred.x, aP2Pred.y, aP2Pred.z);
                 }
             }

             if(false){
                 int nSize1 = int(aOriPtNumV.size());
                 int nSize2 = int(aInsideBorderPtNumV.size());
                 printf("aOriPtNumV.size(): %d; aOriPtNumV[-1]: %d\n", nSize1, aOriPtNumV[nSize1-1]);
                 printf("aInsideBorderPtNumV.size(): %d; aInsideBorderPtNumV[-1]: %d\n", nSize2, aInsideBorderPtNumV[nSize2-1]);
             }

             dGSDAve1 += dGSD1;
             dGSDAve2 += dGSD2;
             nNum++;

             if(bPrint)
             {
                 printf("Finished transforming %d tie points into 3D.\n", nOriPtNum);
             }
         }
     }
     int nPtNum = aV1.size();
     cout<<"nOriPtNum: "<<nOriPtNum<<";  InsideBorderPtNum:  "<<nPtNum<<endl;

     dGSDAve1 /= nNum;
     dGSDAve2 /= nNum;

     if(threshold < 0){
         threshold = 30*dGSDAve2;
         printf("Average GSD of master images: %.5lf, Average GSD of secondary images: %.5lf, 3DRANTh: %.5lf\n", dGSDAve1, dGSDAve2, threshold);
     }

     srand((int)time(0));
     std::vector<ElCplePtsHomologues> inlierFinal;
     cSolBasculeRig aSBR = cSolBasculeRig::Id();

     std::string aMsg = "";
     if (ELISE_fp::exist_file(input_dir+"/"+aInBascFile) == true)
     {
         aMsg = "Use "+input_dir+"/"+aInBascFile+" instead of automatically estimating 3D Helmet transformation.";
         //cout<<"Use "<<input_dir+"/"+aInBascFile<<" instead of automatically estimating 3D Helmet transformation."<<endl;
         cXml_ParamBascRigide  *  aXBR = OptStdGetFromPCP(input_dir+"/"+aInBascFile,Xml_ParamBascRigide);
         aSBR = Xml2EL(*aXBR);
     }
     else
     {
         if(nPtNum<nMinPt)
         {
             printf("InsideBorderPtNum (%d) is less than %d, hence skipped.\n", nPtNum, nMinPt);
             return;
         }

         printf("iteration number: %d; thresh: %lf\n", aNbTir, threshold);
         aSBR = RANSAC3DCore(aNbTir, threshold, aV1, aV2, a2dV1, a2dV2, inlierFinal);
         aMsg = "Use 3D RANSAC to automatically estimate 3D Helmet transformation using tie points in Homol" + inSH;
     }

     int nIdx = -1;
     //std::vector<Pt2dr> vPt2DL, vPt2DR;
     std::vector<Pt3dr> vPt3DL, vPt3DR;

     /*
     cSetOfMesureAppuisFlottants aSOMAFout1;
     cSetOfMesureAppuisFlottants aSOMAFout2;

     std::vector<cMesureAppuiFlottant1Im> aMAF2V;
     for(int n=0; n<int(aVIm2.size()); n++)
     {
         aImg2 = aVIm2[n];
         cMesureAppuiFlottant1Im aMAF2;
         aMAF2.NameIm() = aImg2;
         aMAF2V.push_back(aMAF2);
     }
     */

     int nTotalInlier = 0;
     for(int m=0; m<int(aVIm1.size()); m++)
     {
         aImg1 = aVIm1[m];
         //cMesureAppuiFlottant1Im aMAF1;
         //aMAF1.NameIm() = aImg1;
         if(1) //for(int n=0; n<int(aVIm2.size()); n++)
         {
             aImg2 = aVIm2[m];

             nIdx++;
             std::vector<ElCplePtsHomologues> inlierCur;
             /*
             std::string aDir_inSH = input_dir + "/Homol" + inSH+"/";
             std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";
             if (ELISE_fp::exist_file(aNameIn) == false)
             {
                 //cout<<aNameIn<<"didn't exist hence skipped."<<endl;
                 continue;
             }
             ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);
             */
             if(nIdx+1 >= int(aInsideBorderPtNumV.size()))
             {
                 cout<<"current index "<<nIdx+1<<" is bigger than aInsideBorderPtNumV.size() "<<aInsideBorderPtNumV.size()<<endl;
                 return;
             }
             int nStart = aInsideBorderPtNumV[nIdx];
             int nEnd = aInsideBorderPtNumV[nIdx+1];
             //printf("nStart, nEnd: %d, %d\n", nStart, nEnd);
             for(int i=nStart; i<nEnd; i++)
             {
                 Pt3dr aP1 = aV1[i];
                 Pt3dr aP2 = aV2[i];

                 Pt3dr aP2Pred = aSBR(aP1);
                 //printf("%dth: PtL: [%.2lf, %.2lf], PtR: [%.2lf, %.2lf]\n", i, a2dV1[i].x, a2dV1[i].y, a2dV2[i].x, a2dV2[i].y);
                 //printf("%dth: PtL: [%.2lf, %.2lf, %.2lf], PtR: [%.2lf, %.2lf, %.2lf], PtRPred: [%.2lf, %.2lf, %.2lf]\n", i, aP1.x, aP1.y, aP1.z, aP2.x, aP2.y, aP2.z, aP2Pred.x, aP2Pred.y, aP2Pred.z);
                 double dist = pow(pow(aP2Pred.x-aP2.x,2) + pow(aP2Pred.y-aP2.y,2) + pow(aP2Pred.z-aP2.z,2), 0.5);
                 if(dist < threshold)
                 {
                     //printf("inlier, %.5lf, %.5lf\n", dist, threshold);
                     //vPt2DL.push_back(a2dV1[i]);
                     //vPt2DR.push_back(a2dV2[i]);
                     vPt3DL.push_back(aV1[i]);
                     vPt3DR.push_back(aV2[i]);

                     inlierCur.push_back(ElCplePtsHomologues(a2dV1[i], a2dV2[i]));
                     nTotalInlier++;

                     /*
                     cOneMesureAF1I anOM1;
                     anOM1.NamePt() = std::to_string(i);
                     anOM1.PtIm() = a2dV1[i];
                     aMAF1.OneMesureAF1I().push_back(anOM1);

                     cOneMesureAF1I anOM2;
                     anOM2.NamePt() = std::to_string(i);
                     anOM2.PtIm() = a2dV2[i];
                     aMAF2V[n].OneMesureAF1I().push_back(anOM2);
                     */
                 }
             }

             if (bSaveHomol == true)
             {
                 SaveHomolTxtFile(input_dir, aImg1, aImg2, outSH, inlierCur);

                 std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
                 std::string aComInv = "mm3d SEL" + BLANK + input_dir + BLANK + aImg2 + BLANK + aImg1 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
                 if(inlierCur.size() >0)
                     cout<<aCom<<endl<<aComInv<<endl;
                 cout<<"nOriPtNum: "<<aOriPtNumV[nIdx+1]-aOriPtNumV[nIdx]<<" InsideBorderPtNum:  "<<nEnd-nStart<<";  nFilteredPtNum: "<<inlierCur.size()<<endl;
             }
         }
         //aSOMAFout1.MesureAppuiFlottant1Im().push_back(aMAF1);
     }

     /*
     for(int n=0; n<int(aVIm2.size()); n++)
     {
         aSOMAFout2.MesureAppuiFlottant1Im().push_back(aMAF2V[n]);
     }
     */
         int nSize1 = int(aOriPtNumV.size());
         int nSize2 = int(aInsideBorderPtNumV.size());
         printf("--->>>Total OriPt: %d; Total InsideBorderPt: %d; Total inlier: %d; Inlier Ratio (GlobalR3D): %.2lf%%\n", aOriPtNumV[nSize1-1], aInsideBorderPtNumV[nSize2-1], nTotalInlier, nTotalInlier*100.0/aInsideBorderPtNumV[nSize2-1]);
         cout<<aMsg<<endl;

     //cout<<"Total inlier: "<<nTotalInlier<<endl;

     if(bSaveGCP == true)
     {
         std::string aOut2DXml1 = input_dir + "/GCP2D_Homol" + inSH + "_" + aOri1 +".xml";
         std::string aOut2DXml2 = input_dir + "/GCP2D_Homol" + inSH + "_" + aOri2 +".xml";
         //MakeFileXML(aSOMAFout1, aOut2DXml1);
         //MakeFileXML(aSOMAFout2, aOut2DXml2);
         Get2DCoor(input_dir, aVIm1, vPt3DL, aOri1, aICNM,  aOut2DXml1);
         Get2DCoor(input_dir, aVIm2, vPt3DR, aOri2, aICNM,  aOut2DXml2);

         std::string aOut3DXml1 = input_dir + "/GCP3D_Homol" + inSH + "_" + aOri1 +".xml";
         std::string aOut3DXml2 = input_dir + "/GCP3D_Homol" + inSH + "_" + aOri2 +".xml";
         Save3DXml(vPt3DL, aOut3DXml1);
         Save3DXml(vPt3DR, aOut3DXml2);
         printf("xdg-open %s\nxdg-open %s\nxdg-open %s\nxdg-open %s\n", aOut2DXml1.c_str(), aOut2DXml2.c_str(), aOut3DXml1.c_str(), aOut3DXml2.c_str());
     }

     std::string aSBRFile = input_dir + "/Basic_Homol" + inSH + "-" + aOri1 + "-2-" + aOri2 +".xml";
     std::string aSBRInvFile = input_dir + "/Basic_Homol" + inSH + "-" + aOri2 + "-2-" + aOri1 +".xml";
     MakeFileXML(EL2Xml(aSBR), aSBRFile);
     MakeFileXML(EL2Xml(aSBR.Inv()), aSBRInvFile);
     printf("xdg-open %s\nxdg-open %s\n", aSBRFile.c_str(), aSBRInvFile.c_str());
}

int GlobalR3D_main(int argc,char ** argv)
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aImgList1;
   std::string aImgList2;

   std::string aOri1;
   std::string aOri2;

   std::string aDSMDirL;
   std::string aDSMDirR;

   std::string aDSMFileL = "MMLastNuage.xml";
   std::string aDSMFileR = "MMLastNuage.xml";
   std::string aTransFile = "";

   bool bSaveHomol = true;
   bool bSaveGCP = false;

   double aR3DIteration = 2000;
   std::string aInSH = "";
   std::string aR3DOutSH = "";
   double aR3DThreshold = -1;
   int aMinPt = 10;
   std::string aImgPair;

   ElInitArgMain
    (
        argc,argv,
               LArgMain()
                << EAMC(aOri1,"Orientation of images in epoch1")
                << EAMC(aOri2,"Orientation of images in epoch2")
                << EAMC(aImgList1,"ImgList1: All RGB images in epoch1 (Dir+Pattern, or txt file of image list)")
                << EAMC(aImgList2,"ImgList2: All RGB images in epoch2 (Dir+Pattern, or txt file of image list)")
                << EAMC(aDSMDirL,"DSM direcotry of epoch1")
                << EAMC(aDSMDirR,"DSM direcotry of epoch2"),
        LArgMain()
                    << aCAS3D.ArgBasic()
               << EAM(aImgPair,"Pair",true,"XML-File of image pair (if this parameter is defined, the input image pairs will be defnied by this instead of ImgList1 and ImgList2 will be ), Def=none")
               << EAM(aInSH,"InSH",true,"Input Homologue extenion for NB/NT mode, Def=none")
               << EAM(aR3DOutSH,"3DRANOutSH",true,"Output Homologue extenion for NB/NT mode after 3D RANSAC, Def='InSH'-GlobalR3D")
               << EAM(aR3DIteration,"3DIter",true,"3D RANSAC iteration, Def=2000")
               << EAM(aR3DThreshold,"3DRANTh",true,"3D RANSAC threshold, Def=30*(GSD of secondary image)")
               << EAM(aMinPt,"MinPt",true,"Minimun number of input correspondences required, Def=10")
               //<< aCAS3D.Arg3DRANSAC()
               << EAM(aDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
               << EAM(aDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")
               << EAM(aTransFile, "TransFile", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master images to secondary images (if not provided, it will be automatically estimated), Def=none")
               << EAM(bSaveHomol, "SaveHomol", true, "Save inlier tie points, Def=true")
               << EAM(bSaveGCP, "SaveGCP", true, "Save GCP files based on the inlier tie points, Def=false")
    );

   if (!EAMIsInit(&aR3DOutSH))
       aR3DOutSH = aInSH + "-GlobalR3D";

   StdCorrecNameOrient(aOri1,"./",true);
   StdCorrecNameOrient(aOri2,"./",true);

   cTransform3DHelmert aTrans3DHL("");

   std::vector<std::string> aVIm1;
   std::vector<std::string> aVIm2;
   if (ELISE_fp::exist_file(aCAS3D.mDir+"/"+aImgPair) == true)
   {
       GetXmlImgPair(aCAS3D.mDir+"/"+aImgPair, aVIm1, aVIm2);
   }
   else
   {
       std::vector<std::string> aVIm1Tmp;
       std::vector<std::string> aVIm2Tmp;
       GetImgListVec(aImgList1, aVIm1Tmp);
       GetImgListVec(aImgList2, aVIm2Tmp);
       for(int i=0; i<int(aVIm1Tmp.size()); i++){
           for(int j=0; j<int(aVIm1Tmp.size()); j++){
               aVIm1.push_back(aVIm1Tmp[i]);
               aVIm2.push_back(aVIm2Tmp[j]);
           }
       }
   }

   cout<<aVIm1.size()<<" image pairs to be processed."<<endl;

   GlobalR3D(aOri1, aOri2, aCAS3D.mICNM, aCAS3D.mDir, aVIm1, aVIm2, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR, aInSH, aR3DOutSH, aTrans3DHL, aR3DIteration, aR3DThreshold, aMinPt, aCAS3D.mPrint, bSaveHomol, bSaveGCP, aTransFile);

   return EXIT_SUCCESS;
}

int CoReg_GlobalR3D_main(int argc,char ** argv)
{
    cCommonAppliTiepHistorical aCAS3D;
    bool aExe = true;

   std::string aImgList1;
   std::string aImgList2;

   std::string aOri1;
   std::string aOri2;

   std::string aDSMDirL;
   std::string aDSMDirR;

   std::string aInSH = "";

   /**********************/
   std::string aFeature = "SuperGlue";
   std::string aImgPair;

   bool bCheckFile = false;

   double aCheckNb = -1;

   std::string aOutput_dir = "";

   bool bRotHyp = false;
   std::string aOutSH = "-SIFT2Step";
   std::string aOriOut = "";
   ElInitArgMain
    (
        argc,argv,
               LArgMain()
                << EAMC(aOri1,"Ori1: Orientation of images in epoch1")
                << EAMC(aOri2,"Ori2: Orientation of images in epoch2")
                << EAMC(aImgList1,"ImgList1: All RGB images in epoch1 (Dir+Pattern, or txt file of image list)")
                << EAMC(aImgList2,"ImgList2: All RGB images in epoch2 (Dir+Pattern, or txt file of image list)")
                << EAMC(aDSMDirL,"DSM direcotry of epoch1")
                << EAMC(aDSMDirR,"DSM direcotry of epoch2"),
        LArgMain()
                    << EAM(aExe,"Exe",true,"Execute all, Def=true")
                    << aCAS3D.ArgBasic()
                    << EAM(aFeature,"Feature",true,"Feature matching method used for matching (SuperGlue or SIFT), Def=SuperGlue")
               << EAM(aImgPair,"Pair",true,"XML-File of image pair (if this parameter is defined, the input image pairs will be defnied by this instead of ImgList1 and ImgList2 will be ), Def=none")

               << aCAS3D.ArgSuperGlue()
               << EAM(aOutput_dir, "OutDir", true, "The output directory of the match results of SuperGlue, Def=InDir")
               << EAM(bCheckFile, "CheckFile", true, "Check if the result files of inter-epoch correspondences exist (if so, skip to avoid repetition), Def=false")
               << EAM(aCheckNb,"CheckNb",true,"Radius of the search space for SuperGlue (which means correspondence [(xL, yL), (xR, yR)] with (xL-xR)*(xL-xR)+(yL-yR)*(yL-yR) > CheckNb*CheckNb will be removed afterwards), Def=-1 (means don't check search space)")
               << EAM(bRotHyp, "RotHyp", true, "Apply rotation hypothesis (if true, the secondary image will be rotated by 90 degreee 4 times and the matching with the largest correspondences will be kept), Def=false")

               << EAM(aOutSH, "SIFTOutSH", true, "Homologue extenion for NB/NT mode, Def=-SIFT2Step")
               << EAM(aOriOut, "OriOut", true, "Output orientation, Def='Ori1'_CoReg_'Feature'")
    );

   StdCorrecNameOrient(aOri1,"./",true);
   StdCorrecNameOrient(aOri2,"./",true);

   if(aOriOut.length() == 0)
       aOriOut = aOri1 + "_CoReg_" + aFeature;

   std::vector<std::string> aVIm1;
   std::vector<std::string> aVIm2;
   if (ELISE_fp::exist_file(aCAS3D.mDir+"/"+aImgPair) == false)
   {
       GetImgListVec(aImgList1, aVIm1);
       GetImgListVec(aImgList2, aVIm2);
       aImgPair = "PairAll.xml";
       SaveXmlImgPair(aCAS3D.mDir+"/"+aImgPair, aVIm1, aVIm2);
       aVIm1.clear();
       aVIm2.clear();
   }
   int nPairs = GetXmlImgPair(aCAS3D.mDir+"/"+aImgPair, aVIm1, aVIm2);
   cout<<nPairs<<" image pairs to be matched."<<endl;

   std::string aComm;
   if(aFeature == "SuperGlue")
   {
       if(aOutput_dir.length() == 0)
           aOutput_dir = aCAS3D.mInput_dir;

       std::string aImgPairTxt = StdPrefix(aImgPair) + ".txt";
       SaveTxtImgPair(aCAS3D.mInput_dir+"/"+aImgPairTxt, aVIm1, aVIm2);

       std::string aOptPara = aCAS3D.ComParamSuperGlue();
       if (EAMIsInit(&aOutput_dir))         aOptPara +=  " OutDir=" + aOutput_dir;
       if (EAMIsInit(&bCheckFile))          aOptPara +=  " CheckFile=" + ToString(bCheckFile);
       if (EAMIsInit(&aCheckNb))            aOptPara +=  " CheckNb=" + ToString(aCheckNb);
       if (EAMIsInit(&bRotHyp))             aOptPara +=  " RotHyp=" + ToString(bRotHyp);

       aComm = MMBinFile(MM3DStr) + "TestLib SuperGlue " + aImgPairTxt + aOptPara;
       cout<<aComm<<endl;
       if(aExe)
           System(aComm);

       aInSH = aCAS3D.mSpGlueOutSH;
   }
   else if(aFeature == "SIFT")
   {
       std::string aOptPara = " Skip2ndSIFT=1";
       if (EAMIsInit(&aOutSH))         aOptPara +=  " OutSH=" + aOutSH;
       aComm = MMBinFile(MM3DStr) + "TestLib SIFT2StepFile " + aImgPair + aOptPara;
       cout<<aComm<<endl;
       if(aExe)
           System(aComm);

       aInSH = aOutSH + "-Rough-2DRANSAC";
   }
   else
       printf("Please set Feature to SuperGlue or SIFT\n");

   std::string aOptPara = " SaveGCP=1";
   aOptPara += " InSH="+aInSH;
   aOptPara += " Pair="+aImgPair;
   aComm = MMBinFile(MM3DStr) + "TestLib GlobalR3D " + aOri1 + BLANK + aOri2 + " none none " + aDSMDirL + BLANK + aDSMDirR + aOptPara;
   cout<<aComm<<endl;
   if(aExe)
       System(aComm);

   std::vector<std::string> aVIm1Uniq;
   GetUniqImgList(aVIm1, aVIm1Uniq);
   std::string aImgPatternL = GetImgList(aVIm1Uniq);
   std::string aOut2DXml1 = aCAS3D.mDir + "/GCP2D_Homol" + aInSH + "_" + aOri1 +".xml";
   std::string aOut3DXml2 = aCAS3D.mDir + "/GCP3D_Homol" + aInSH + "_" + aOri2 +".xml";
   aComm = MMBinFile(MM3DStr) + "GCPBascule " + aImgPatternL + BLANK + aOri1 + BLANK + aOriOut + BLANK + aOut3DXml2 + BLANK + aOut2DXml1;
   cout<<aComm<<endl;
   if(aExe)
       System(aComm);

   return EXIT_SUCCESS;
}

