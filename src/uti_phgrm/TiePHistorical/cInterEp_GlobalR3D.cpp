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

void GetOrthoPt(std::string aOrthoFileL, std::vector<Pt3dr> vPt3DL, std::vector<Pt2dr>& vOrthoPtL, bool & bSaveOrthoHomo)
{
    if(ELISE_fp::exist_file(aOrthoFileL) == true){
        std::vector<double> aTmp;
        std::string aTfwFile = StdPrefix(aOrthoFileL) + ".tfw";
        ReadTfw(aTfwFile, aTmp);
        Pt2dr aOrthoResolPlani = Pt2dr(aTmp[0], aTmp[3]);
        Pt2dr aOrthoOriPlani = Pt2dr(aTmp[4], aTmp[5]);

        for(int i=0; i<int(vPt3DL.size()); i++){
            Pt3dr aPTer1 = vPt3DL[i];
            Pt2dr aPtOrtho;
            aPtOrtho.x = (aPTer1.x - aOrthoOriPlani.x)/aOrthoResolPlani.x;
            aPtOrtho.y = (aPTer1.y - aOrthoOriPlani.y)/aOrthoResolPlani.y;
            vOrthoPtL.push_back(aPtOrtho);
        }
    }
    else{
        bSaveOrthoHomo = false;
    }
}

void GetOrthoHom(std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, std::string input_dir, std::vector<std::string> aVIm1, std::vector<std::string> aVIm2, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR, std::string inSH, bool bPrint, cTransform3DHelmert aTrans3DHL, std::string aOrthoImg1, std::string aOrthoImg2, std::string aOutImg1, std::string aOutImg2)
{
    std::vector<Pt3dr> aV1;
    std::vector<Pt3dr> aV2;
    std::vector<Pt2dr> a2dV1;
    std::vector<Pt2dr> a2dV2;

    std::vector<int> aOriPtNumV;
    std::vector<int> aInsideBorderPtNumV;

    if(ELISE_fp::exist_file(aOrthoImg1) == false || ELISE_fp::exist_file(aOrthoImg1) == false){
        printf("%s or %s don't exist, hence skipped\n", aOrthoImg1.c_str(), aOrthoImg2.c_str());
        return;
    }
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

            //Get3DTiePt(aPackFull, a3DCoorL, a3DCoorR, aDSMInfoL, aDSMInfoR, aTrans3DHL, aV1, aV2, a2dV1, a2dV2, bPrint);
            int nOriPtNumCur = Get3DTiePt(aPackFull, a3DCoorL, a3DCoorR, aDSMInfoL, aDSMInfoR, aTrans3DHL, aV1, aV2, a2dV1, a2dV2, bPrint);
            nOriPtNum += nOriPtNumCur;
            aOriPtNumV.push_back(nOriPtNum);
            aInsideBorderPtNumV.push_back(int(aV1.size()));
            cout<<aImg1<<" "<<aImg2<<": ";
            cout<<"nOriPtNum: "<<nOriPtNumCur<<";  InsideBorderPtNum:  "<<aInsideBorderPtNumV[aInsideBorderPtNumV.size()-1] - aInsideBorderPtNumV[aInsideBorderPtNumV.size()-2]<<"; ";
            double dGSD1 = a3DCoorL.GetGSD();
            double dGSD2 = a3DCoorR.GetGSD();
            printf("dGSD1: %.5lf; dGSD2: %.5lf\n", dGSD1, dGSD2);
        }
    }

    bool bSaveOrthoHomo = true;
    std::vector<Pt2dr> vOrthoPtL, vOrthoPtR;
    GetOrthoPt(aOrthoImg1, aV1, vOrthoPtL, bSaveOrthoHomo);
    GetOrthoPt(aOrthoImg2, aV2, vOrthoPtR, bSaveOrthoHomo);
    printf("aOrthoImg1: %s\naOrthoImg2: %s\n", aOrthoImg1.c_str(), aOrthoImg2.c_str());

    if(bSaveOrthoHomo){
        std::vector<ElCplePtsHomologues> vOrthoHom;
        for(int i=0; i<int(vOrthoPtL.size()); i++)
            vOrthoHom.push_back(ElCplePtsHomologues(vOrthoPtL[i], vOrthoPtR[i]));

        ELISE_fp::MkDir(input_dir+"/Tmp_PileImg/");
        SaveHomolTxtFile(input_dir+"/Tmp_PileImg/", aOutImg1, aOutImg2, inSH+"-PileImg", vOrthoHom);
        cout<<"nPtNum: "<<vOrthoHom.size()<<endl;
    }
}

void GlobalR3D(std::string aOri1, std::string aOri2, cInterfChantierNameManipulateur * aICNM, std::string input_dir, std::vector<std::string> aVIm1, std::vector<std::string> aVIm2, std::string aDSMFileL, std::string aDSMFileR, std::string aDSMDirL, std::string aDSMDirR, std::string inSH, std::string outSH, cTransform3DHelmert aTrans3DHL, int aNbTir, double threshold, int nMinPt, bool bPrint, bool bSaveHomol, bool bSaveGCP, std::string aInBascFile, std::string aOut)
{
    aOut = aOut;
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
         Pt3dr aTr = aSBR.Tr();
         double aLambda = aSBR.Lambda();
         printf("Final aLambda: %.2lf, aTr: [%.2lf, %.2lf, %.2lf]\n", aLambda, aTr.x, aTr.y, aTr.z);

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

     cSauvegardeNamedRel aRel;

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

             if(inlierCur.size() > 0)
                 aRel.Cple().push_back(cCpleString(aImg1, aImg2));


             if (bSaveHomol == true)
             {
                 SaveHomolTxtFile(input_dir, aImg1, aImg2, outSH, inlierCur);
                 cout<<"nOriPtNum: "<<aOriPtNumV[nIdx+1]-aOriPtNumV[nIdx]<<" InsideBorderPtNum:  "<<nEnd-nStart<<";  nFilteredPtNum: "<<inlierCur.size();
                 printf("; Inlier Ratio: %.2lf%%\n", int(inlierCur.size())*100.0/(nEnd-nStart));
             }
         }
         //aSOMAFout1.MesureAppuiFlottant1Im().push_back(aMAF1);
     }
     MakeFileXML(aRel,input_dir+"/"+aOut);
     printf("xdg-open %s\n", (input_dir+"/"+aOut).c_str());


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
   std::string aImgPair = "";

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

   if (!EAMIsInit(&aR3DOutSH)){
       if(aTransFile.length() == 0)
           aR3DOutSH = aInSH + "-GlobalR3D";
       else
           aR3DOutSH = aInSH + "-GlobalR3DGT";
   }

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
           for(int j=0; j<int(aVIm2Tmp.size()); j++){
               aVIm1.push_back(aVIm1Tmp[i]);
               aVIm2.push_back(aVIm2Tmp[j]);
           }
       }
   }

   cout<<aVIm1.size()<<" image pairs to be processed."<<endl;

   std::string aOutImgPair = "PairAll"+aR3DOutSH+".xml";
   if(aImgPair != "")
       aOutImgPair = StdPrefix(aImgPair) + aR3DOutSH + ".xml";
   cout<<"Output pairs will be saved in "<<aOutImgPair<<endl;

   GlobalR3D(aOri1, aOri2, aCAS3D.mICNM, aCAS3D.mDir, aVIm1, aVIm2, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR, aInSH, aR3DOutSH, aTrans3DHL, aR3DIteration, aR3DThreshold, aMinPt, aCAS3D.mPrint, bSaveHomol, bSaveGCP, aTransFile, aOutImgPair);

   return EXIT_SUCCESS;
}

int GetOrthoHom_main(int argc,char ** argv)
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

   std::string aInSH = "";
   std::string aImgPair = "";

   std::string aOrthoDirL;
   std::string aOrthoDirR;
   std::string aOrthoFileL;
   std::string aOrthoFileR;

   aOrthoDirL = "";
   aOrthoDirR = "";
   aOrthoFileL = "Orthophotomosaic.tif";
   aOrthoFileR = "Orthophotomosaic.tif";

   std::string aOrthoImg1 = "";
   std::string aOrthoImg2 = "";

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

               << EAM(aDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
               << EAM(aDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")
               << EAM(aOrthoDirL, "OrthoDirL", true, "Orthophoto directory of epoch1 (if this parameter is set, it means the tie points are on orthophotos instead of DSMs), Def=none")
               << EAM(aOrthoDirR, "OrthoDirR", true, "Orthophoto directory of epoch2 (if this parameter is set, it means the tie points are on orthophotos instead of DSMs), Def=none")
               << EAM(aOrthoFileL, "OrthoFileL", true, "Orthophoto file of epoch1, Def=Orthophotomosaic.tif")
               << EAM(aOrthoFileR, "OrthoFileR", true, "Orthophoto file of epoch2, Def=Orthophotomosaic.tif")
               << EAM(aOrthoImg1, "OrthoImg1", true, "Orthophoto file of epoch1 in Tmp_PileImg folder (if this parameter is set, OrthoDirL and OrthoFileL will be ignored), Def=none")
               << EAM(aOrthoImg2, "OrthoImg2", true, "Orthophoto file of epoch2 in Tmp_PileImg folder (if this parameter is set, OrthoDirR and OrthoFileR will be ignored), Def=none")
    );

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
           for(int j=0; j<int(aVIm2Tmp.size()); j++){
               aVIm1.push_back(aVIm1Tmp[i]);
               aVIm2.push_back(aVIm2Tmp[j]);
           }
       }
   }

   std::string aOutImg1, aOutImg2;
   if(aOrthoImg1.length() == 0){
       aOrthoImg1 = aOrthoDirL + "/" + aOrthoFileL;
       aOutImg1= aOrthoDirL+".tif";
   }
   else{
       aOutImg1 = aOrthoImg1;
       aOrthoImg1 = aCAS3D.mDir+"/Tmp_PileImg/"+aOrthoImg1;
   }
   if(aOrthoImg2.length() == 0){
       aOrthoImg2 = aOrthoDirR + "/" + aOrthoFileR;
       aOutImg2= aOrthoDirR+".tif";
   }
   else{
       aOutImg2 = aOrthoImg2;
       aOrthoImg2 = aCAS3D.mDir+"/Tmp_PileImg/"+aOrthoImg2;
   }
   printf("aOrthoImg1: %s\n",aOrthoImg1.c_str());
   printf("aOrthoImg2: %s\n",aOrthoImg2.c_str());
   printf("aOutImg1: %s\n",aOutImg1.c_str());
   printf("aOutImg2: %s\n",aOutImg2.c_str());

   GetOrthoHom(aOri1, aOri2, aCAS3D.mICNM, aCAS3D.mDir, aVIm1, aVIm2, aDSMFileL, aDSMFileR, aDSMDirL, aDSMDirR, aInSH, aCAS3D.mPrint, aTrans3DHL, aOrthoImg1, aOrthoImg2, aOutImg1, aOutImg2);
   cout<<aVIm1.size()<<" image pairs processed."<<endl;

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

   double aR3DThreshold = -1;

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
               << EAM(aR3DThreshold,"3DRANTh",true,"3D RANSAC threshold, Def=30*(GSD of secondary image)")
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
   aOptPara += " 3DRANTh="+ToString(aR3DThreshold);
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

Pt2di PileImgs(std::string aDir, std::vector<std::string> aVIm1, std::string aOri1, std::string aImgList, std::string aTfwFile, cInterfChantierNameManipulateur * aICNM, std::string aResDir, std::string aMasq)
{
    std::vector<double> aTmp;
    ReadTfw(aTfwFile, aTmp);
    Pt2dr aOrthoResolPlani = Pt2dr(aTmp[0], aTmp[3]);
    Pt2dr aOrthoOriPlani = Pt2dr(aTmp[4], aTmp[5]);

    Pt2dr ptMax = Pt2dr(DBL_MIN, DBL_MIN);
    Pt2dr ptMin = Pt2dr(DBL_MAX, DBL_MAX);
    for(int i=0; i<int(aVIm1.size()); i++)
    {
        std::string aImg1 = aVIm1[i];
        //Tiff_Im aRGBIm1((aDir+aImg1).c_str());
        Tiff_Im aRGBIm1 = Tiff_Im::StdConvGen((aDir+aImg1).c_str(), -1, true ,true);
        Pt2di ImgSzL = aRGBIm1.sz();

        std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
        cGet3Dcoor a3DCoorL(aIm1OriFile);

        Pt2dr origin = Pt2dr(0,0);
        Pt2dr aPCornerL[4];
        aPCornerL[0] = origin;
        aPCornerL[1] = Pt2dr(origin.x+ImgSzL.x, origin.y);
        aPCornerL[2] = Pt2dr(origin.x+ImgSzL.x, origin.y+ImgSzL.y);
        aPCornerL[3] = Pt2dr(origin.x, origin.y+ImgSzL.y);

        for(int j=0; j<4; j++)
        {
            Pt2dr aP1 = aPCornerL[j];

            Pt3dr aPTer1 = a3DCoorL.GetRough3Dcoor(aP1);//, a3DCoorL.GetGSD());

            if(aPTer1.x > ptMax.x)
                ptMax.x = aPTer1.x;
            if(aPTer1.x < ptMin.x)
                ptMin.x = aPTer1.x;

            if(aPTer1.y > ptMax.y)
                ptMax.y = aPTer1.y;
            if(aPTer1.y < ptMin.y)
                ptMin.y = aPTer1.y;
        }
    }
    printf("ptMin: [%.2lf, %.2lf], ptMax: [%.2lf, %.2lf]\n", ptMin.x, ptMin.y, ptMax.x, ptMax.y);
    printf("aOrthoResolPlani: [%.2lf, %.2lf], aOrthoOriPlani: [%.2lf, %.2lf]\n", aOrthoResolPlani.x, aOrthoResolPlani.y, aOrthoOriPlani.x, aOrthoOriPlani.y);
    if(aOrthoResolPlani.x > 0)
        aOrthoOriPlani.x = ptMin.x;
    else
        aOrthoOriPlani.x = ptMax.x;
    if(aOrthoResolPlani.y > 0)
        aOrthoOriPlani.y = ptMin.y;
    else
        aOrthoOriPlani.y = ptMax.y;
    SaveTfw(aTfwFile, aOrthoResolPlani, aOrthoOriPlani);
    printf("aOrthoResolPlani: [%.2lf, %.2lf], aOrthoOriPlani: [%.2lf, %.2lf]\n", aOrthoResolPlani.x, aOrthoResolPlani.y, aOrthoOriPlani.x, aOrthoOriPlani.y);

    if(0){
        aTfwFile = "/mnt/e4833a33-2e75-4f51-907f-10b923e3000d/PhDFullTest/Pezenas/PileImg/Tmp_PseudoOrtho/Ortho-MEC-Malt_2015-.tfw";
        ReadTfw(aTfwFile, aTmp);
        aOrthoResolPlani = Pt2dr(aTmp[0], aTmp[3]);
        aOrthoOriPlani = Pt2dr(aTmp[4], aTmp[5]);
        printf("aOrthoResolPlani: [%.2lf, %.2lf], aOrthoOriPlani: [%.2lf, %.2lf]\n", aOrthoResolPlani.x, aOrthoResolPlani.y, aOrthoOriPlani.x, aOrthoOriPlani.y);
    }
    Pt2di aImgSz;
    aImgSz.x = abs((ptMax.x -ptMin.x)/aOrthoResolPlani.x);
    aImgSz.y = abs((ptMax.y -ptMin.y)/aOrthoResolPlani.y);
    printf(" --ImgSz %d %d\n", aImgSz.x, aImgSz.y);

    FILE * fpImgList = fopen((aResDir+aImgList).c_str(), "w");
    for(int i=0; i<int(aVIm1.size()); i++)
    {
        std::string aImg1 = aVIm1[i];
        //Tiff_Im aRGBIm1((aDir+aImg1).c_str());
        Tiff_Im aRGBIm1 = Tiff_Im::StdConvGen((aDir+aImg1).c_str(), -1, true ,true);
        Pt2di ImgSzL = aRGBIm1.sz();

        std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
        cGet3Dcoor a3DCoorL(aIm1OriFile);

        Pt2dr origin = Pt2dr(0,0);
        Pt2dr aPCornerL[4];
        aPCornerL[0] = origin;
        aPCornerL[1] = Pt2dr(origin.x+ImgSzL.x, origin.y);
        aPCornerL[2] = Pt2dr(origin.x+ImgSzL.x, origin.y+ImgSzL.y);
        aPCornerL[3] = Pt2dr(origin.x, origin.y+ImgSzL.y);

        std::string aTxt = StdPrefix(aImg1)+"_PseudoOrtho.txt";
        fprintf(fpImgList, "%s\n", (StdPrefix(aTxt)+".tif").c_str());

        FILE * fpOutput = fopen((aDir+"/"+aTxt).c_str(), "w");
        for(int j=0; j<4; j++)
        {
            Pt2dr aP1 = aPCornerL[j];

            Pt3dr aPTer1 = a3DCoorL.GetRough3Dcoor(aP1);//, a3DCoorL.GetGSD());

            Pt2dr aPtOrtho;
            aPtOrtho.x = (aPTer1.x - aOrthoOriPlani.x)/aOrthoResolPlani.x;
            aPtOrtho.y = (aPTer1.y - aOrthoOriPlani.y)/aOrthoResolPlani.y;

            fprintf(fpOutput, "%lf %lf %lf %lf\n", aPtOrtho.x, aPtOrtho.y, aPCornerL[j].x, aPCornerL[j].y);
        }
        fclose(fpOutput);

        std::string aTif = StdPrefix(aTxt)+".tif";
        std::string aComm = MMBinFile(MM3DStr) + "TestLib OneReechFromAscii " + aImg1 + BLANK + aTxt + " Show=1 Out=" + aTif;
        cout<<aComm<<endl;
        System(aComm);

        aComm = "mv "+aDir+"/"+aTif+" "+aResDir+aTif;
        cout<<aComm<<endl;
        System(aComm);

        aTif = StdPrefix(aTxt)+"_Masked.tif";
        aComm = MMBinFile(MM3DStr) + "TestLib OneReechFromAscii " + aMasq + BLANK + aTxt + " Show=1 Out=" + aTif;
        cout<<aComm<<endl;
        System(aComm);

        aComm = "mv "+aDir+"/"+aTif+" "+aResDir+aTif;
        cout<<aComm<<endl;
        System(aComm);

        aComm = "mv "+aDir+"/"+aTxt+" "+aResDir+aTxt;
        cout<<aComm<<endl;
        System(aComm);
    }
    fclose(fpImgList);

    return aImgSz;
}

int PileImgs_main(int argc,char ** argv)
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aFullPattern1;

    std::string aOri1;

    std::string aTfwFile = "";
    std::string aTifFile = "";
    std::string aOrthoDir = "";

    std::string aImgList = "";

    std::string aMasq = "Masq.tif";

    ElInitArgMain
     (
         argc,argv,
         LArgMain()   << EAMC(aFullPattern1,"Image name (Dir+Pattern, or txt file of image list)")
                << EAMC(aOri1,"Orientation of images")
                << EAMC(aOrthoDir,"OrthoDir: Orthophoto directory"),
         LArgMain()
                << aCAS3D.ArgBasic()
                << EAM(aMasq, "Masq", true, "File name of input mask, Def=Masq.tif")
                //<< EAM(aTfwFile,"TfwFile",true,"Tfw file to transform 3D points to 2D points on pseudo orthophoto, Def=none")
                << EAM(aImgList,"ImgList",true,"Output file name to record the output image list, Def=ImgList-'OrthoDir'.txt")
    );

    StdCorrecNameOrient(aOri1,"./",true);

    if(aImgList.length() == 0)
        aImgList = "ImgList-"+aOrthoDir+".txt";

    std::string aFile = "Orthophotomosaic.tif";
    //aTifFile = "PseudoOrtho_"+StdPrefix(aImgList)+"."+StdPostfix(aFile);
    aTifFile = aOrthoDir+"."+StdPostfix(aFile);
    std::string strCpImg;
    /*
    strCpImg = "cp "+aCAS3D.mDir+"/"+aOrthoDir+"/"+aFile+" "+aCAS3D.mDir+"/"+aTifFile;
    cout<<strCpImg<<endl;
    System(strCpImg);

    strCpImg = "mv "+aCAS3D.mDir+"/"+aTifFile+" "+aResDir+"/"+aTifFile;
    cout<<strCpImg<<endl;
    System(strCpImg);
    */

    //Tiff_Im aRGBIm1((aCAS3D.mDir+"/"+aOrthoDir+"/"+aFile).c_str());
    Tiff_Im aRGBIm1 = Tiff_Im::StdConvGen((aCAS3D.mDir+"/"+aOrthoDir+"/"+aFile).c_str(), -1, true ,true);
    Pt2di aImgSz = aRGBIm1.sz();
    printf("Original orthophoto size: %d, %d\n", aImgSz.x, aImgSz.y);

    aFile = "Orthophotomosaic.tfw";
    //aTfwFile = "PseudoOrtho_"+StdPrefix(aImgList)+"."+StdPostfix(aFile);
    aTfwFile = aOrthoDir+"."+StdPostfix(aFile);
    strCpImg = "cp "+aCAS3D.mDir+"/"+aOrthoDir+"/"+aFile+" "+aCAS3D.mDir+"/"+aTfwFile;
    cout<<strCpImg<<endl;
    System(strCpImg);

    std::vector<std::string> aVIm1;
    GetImgListVec(aFullPattern1, aVIm1);

    std::string aResDir = aCAS3D.mDir + "/Tmp_PseudoOrtho/";
    if(ELISE_fp::exist_file(aResDir) == false)
        ELISE_fp::MkDir(aResDir);

    aImgSz = PileImgs(aCAS3D.mDir, aVIm1, aOri1, aImgList, aTfwFile, aCAS3D.mICNM, aResDir, aMasq);



    strCpImg = "mv "+aCAS3D.mDir+"/"+aTfwFile+" "+aResDir+"/"+aTfwFile;
    cout<<strCpImg<<endl;
    System(strCpImg);

    std::string aComm = "python3 /home/lulin/Documents/Code/PileImages.py --OutImg "+aTifFile+" --ImgList "+aImgList+" --DirName "+aResDir+" --ImgSz "+ToString(aImgSz.y)+" "+ToString(aImgSz.x);
    cout<<aComm<<endl;

    return EXIT_SUCCESS;
}
