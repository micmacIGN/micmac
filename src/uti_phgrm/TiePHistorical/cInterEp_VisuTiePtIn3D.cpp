#include "TiePHistorical.h"


void Write3DCoor(std::string aNameSaveL, std::vector<Pt3dr> Pt3dL)
{
    FILE * fpOutput = fopen((aNameSaveL).c_str(), "w");
    for(unsigned int i=0; i<Pt3dL.size(); i++)
    {
        fprintf(fpOutput, "%lf %lf %lf\n", Pt3dL[i].x, Pt3dL[i].y, Pt3dL[i].z);
    }
    fclose(fpOutput);
}

int Get3DCoors(ElPackHomologue aPackFull, bool bReverse, std::string aDSMDirL, std::string aDSMDirR, std::string aDSMFileL, std::string aDSMFileR, std::string aIm1OriFile, std::string aIm2OriFile, std::vector<Pt3dr> & Pt3dL, std::vector<Pt3dr> & Pt3dR, std::vector<bool> & vecPreciseL, std::vector<bool> & vecPreciseR, cTransform3DHelmert aTrans3DHL, cTransform3DHelmert aTrans3DHR, bool bPrint, double aThres)
{
    cGet3Dcoor a3DCoorL(aIm1OriFile);
    //TIm2D<float,double> aTImProfPxL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cDSMInfo aDSMInfoL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cGet3Dcoor a3DCoorR(aIm2OriFile);
    //TIm2D<float,double> aTImProfPxR = a3DCoorR.SetDSMInfo(aDSMFileR, aDSMDirR);
    cDSMInfo aDSMInfoR = a3DCoorR.SetDSMInfo(aDSMFileR, aDSMDirR);


    int nTiePtNum = 0;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       /*
       if(bPrint)
           cout<<nTiePtNum<<"th tie pt: "<<p1.x<<" "<<p1.y<<" "<<p2.x<<" "<<p2.y<<endl;
       */

       bool bValidL, bValidR;
       Pt3dr pTerr1, pTerr2;
       if(bReverse == true)
       {
           pTerr1 = a3DCoorL.Get3Dcoor(p2, aDSMInfoL, bValidL, bPrint, aThres);//, dGSD1);
           pTerr2 = a3DCoorR.Get3Dcoor(p1, aDSMInfoR, bValidR, bPrint, aThres);//, dGSD2);
           pTerr1 = aTrans3DHR.Transform3Dcoor(pTerr1);
           pTerr2 = aTrans3DHL.Transform3Dcoor(pTerr2);
       }
       else
       {
           pTerr1 = a3DCoorL.Get3Dcoor(p1, aDSMInfoL, bValidL, bPrint, aThres);//, dGSD1);
           pTerr2 = a3DCoorR.Get3Dcoor(p2, aDSMInfoR, bValidR, bPrint, aThres);//, dGSD2);
           pTerr1 = aTrans3DHL.Transform3Dcoor(pTerr1);
           pTerr2 = aTrans3DHR.Transform3Dcoor(pTerr2);
       }
       Pt3dL.push_back(pTerr1);
       Pt3dR.push_back(pTerr2);
       vecPreciseL.push_back(bValidL);
       vecPreciseR.push_back(bValidR);

       nTiePtNum++;
    }
    return nTiePtNum;
}

void CalcDiff(std::vector<Pt3dr> Pt3dL, std::vector<Pt3dr> Pt3dR, std::vector<bool> vecPreciseL, std::vector<bool> vecPreciseR, std::string aNameSave, double dThres)
{
    Pt3dr Pt3dDiff;
    double dNorm;
    double dNormAve=0;
    double dNormMax = 0;
    int nIDNormMax = 0;
    Pt3dr Pt3dDiffMax = Pt3dr(0,0,0);
    double dNormMin = DBL_MAX;
    int nIDNormMin = 0;
    Pt3dr Pt3dDiffMin = Pt3dr(0,0,0);
    int nOutlier = 0;
    int nTotalPt = Pt3dL.size();

    FILE * fpOutput = fopen((aNameSave).c_str(), "w");
    fprintf(fpOutput, "PtID x y z Norm bPreciseL bPreciseR\n");
    for(int i=0; i<nTotalPt; i++)
    {
        Pt3dDiff = Pt3dL[i] - Pt3dR[i];
        dNorm = pow((pow(Pt3dDiff.x,2) + pow(Pt3dDiff.y,2) + pow(Pt3dDiff.z,2)), 0.5);
        dNormAve += dNorm;
        if(dNorm > dNormMax){
            dNormMax = dNorm;
            nIDNormMax = int(i);
            Pt3dDiffMax = Pt3dDiff;
        }
        if(dNorm < dNormMin){
            dNormMin = dNorm;
            nIDNormMin = int(i);
            Pt3dDiffMin = Pt3dDiff;
        }

        fprintf(fpOutput, "%d %lf %lf %lf %lf %d %d", i, Pt3dDiff.x, Pt3dDiff.y, Pt3dDiff.z, dNorm, int(vecPreciseL[i]), int(vecPreciseR[i]));
        if(dNorm > dThres){
            nOutlier++;
            fprintf(fpOutput, "  *Outlier*");
        }
        fprintf(fpOutput, "\n");
    }
    dNormAve = dNormAve*1.0/nTotalPt;
    fprintf(fpOutput, "------------------------------------------------\n");
    fprintf(fpOutput, "Total tie point: %d;  Outlier: %d;  Inlier: %d\n", nTotalPt, nOutlier, nTotalPt-nOutlier);
    fprintf(fpOutput, "Total NormAve: %lf\n", dNormAve);
    fprintf(fpOutput, "NormMin: %d %lf %lf %lf %lf\n", nIDNormMin, Pt3dDiffMin.x, Pt3dDiffMin.y, Pt3dDiffMin.z, dNormMin);
    fprintf(fpOutput, "NormMax: %d %lf %lf %lf %lf", nIDNormMax, Pt3dDiffMax.x, Pt3dDiffMax.y, Pt3dDiffMax.z, dNormMax);
    fclose(fpOutput);
    printf("Total NormAve: %lf\n", dNormAve);
}

//void VisuTiePtIn3D(std::string input_dir, std::string output_dir, std::string inSH, std::string outSH, std::string aSubPatchXml, bool bPrint)
void VisuTiePtIn3D(std::string aDir, std::vector<string> vImgList1, std::vector<string> vImgList2, std::string aInSH, std::string aOri1, std::string aOri2, std::string aDSMDirL, std::string aDSMDirR, std::string aDSMFileL, std::string aDSMFileR, cTransform3DHelmert aTrans3DHL, cTransform3DHelmert aTrans3DHR, std::string aNameSave, cInterfChantierNameManipulateur * aICNM, bool bPrint, double aThres, bool bSaveDif, double dThres)
{
    /*
    std::vector<string> vImgList1;
    std::vector<string> vImgList2;    
    GetImgListVec(aImgList1, vImgList1);
    GetImgListVec(aImgList2, vImgList2);

    std::string s;
    ifstream in1(aDir+"/"+aImgList1);
    cout<<"Images in "<<aDir+"/"+aImgList1<<":"<<endl;
    while(getline(in1,s))
    {
        vImgList1.push_back(s);
        cout<<s<<endl;
    }

    ifstream in2(aDir+"/"+aImgList2);
    cout<<"Images in "<<aDir+"/"+aImgList2<<":"<<endl;
    while(getline(in2,s))
    {
        vImgList2.push_back(s);
        cout<<s<<endl;
    }
    */

    std::string aDir_inSH = aDir + "/Homol" + aInSH + "/";

    int nTiePtNumTotal = 0;

    std::vector<Pt3dr> Pt3dL;
    std::vector<Pt3dr> Pt3dR;
    std::vector<bool> vecPreciseL;
    std::vector<bool> vecPreciseR;
    for(unsigned int i=0; i<vImgList1.size(); i++)
    {
        std::string aImg1 = vImgList1[i];
        if(1) //for(unsigned int j=0; j<vImgList2.size(); j++)
        {
            //std::string aImg2 = vImgList2[j];
            std::string aImg2 = vImgList2[i];

            std::string aNameIn = aDir_inSH + "Pastis" + aImg1 + "/" + aImg2 + ".txt";
            bool bReverse = false;
            if (ELISE_fp::exist_file(aNameIn) == false)
            {
                bReverse = true;
                aNameIn = aDir_inSH + "Pastis" + aImg2 + "/" + aImg1 + ".txt";
                if (ELISE_fp::exist_file(aNameIn) == false)
                {
                    if(bPrint)
                        printf("%s didn't exist hence skipped.\n", aNameIn.c_str());
                    continue;
                }
            }
            std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1);
            std::string aIm2OriFile = aICNM->StdNameCamGenOfNames(aOri2, aImg2);
            //cout<<aOri1<<"; "<<aImg1<<"; "<<aIm1OriFile<<endl;
            //cout<<aOri2<<"; "<<aImg2<<"; "<<aIm2OriFile<<endl;
            int pos1 = aNameIn.find("Homol")+5;
            int pos2 = aNameIn.find("Pastis")-1;
            std::string outSH = aNameIn.substr(pos1, pos2-pos1);
            std::string aCom = "mm3d SEL" + BLANK + aDir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;
            std::string aComInv = "mm3d SEL" + BLANK + aDir + BLANK + aImg2 + BLANK + aImg1 + BLANK + "KH=NT SzW=[600,600] SH="+outSH;

            ElPackHomologue aPackInLoc =  ElPackHomologue::FromFile(aNameIn);
            int nTiePtNum = Get3DCoors(aPackInLoc, bReverse, aDSMDirL, aDSMDirR, aDSMFileL, aDSMFileR, aIm1OriFile, aIm2OriFile, Pt3dL, Pt3dR, vecPreciseL, vecPreciseR, aTrans3DHL, aTrans3DHR, bPrint, aThres);
            nTiePtNumTotal += nTiePtNum;
            printf("%s\n%s\n%s: %d tie points. Total tie points: %d\n", aCom.c_str(), aComInv.c_str(), aNameIn.c_str(), nTiePtNum, nTiePtNumTotal);
        }
    }

    printf("nTiePtNumTotal: %d\n", nTiePtNumTotal);

    std::string aNameSaveL = StdPrefix(aNameSave) + "_L." + StdPostfix(aNameSave);
    Write3DCoor(aNameSaveL, Pt3dL);
    cout<<"Tie points in master images saved in:"<<endl;
    cout<<"gedit "<<aNameSaveL<<endl;

    std::string aNameSaveR = StdPrefix(aNameSave) + "_R." + StdPostfix(aNameSave);
    Write3DCoor(aNameSaveR, Pt3dR);
    cout<<"Tie points in secondary images saved in:"<<endl;
    cout<<"gedit "<<aNameSaveR<<endl;

    if(bSaveDif){
        std::string aNameSaveDif = StdPrefix(aNameSave) + "_Diff." + StdPostfix(aNameSave);
        CalcDiff(Pt3dL, Pt3dR, vecPreciseL, vecPreciseR, aNameSaveDif, dThres);
        cout<<"Tie points difference saved in:"<<endl;
        cout<<"gedit "<<aNameSaveDif<<endl;
    }
}

int VisuTiePtIn3D_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aOri1;
   std::string aOri2;

   std::string aDSMDirL = "";
   std::string aDSMDirR = "";
   std::string aDSMFileL;
   std::string aDSMFileR;

   aDSMFileL = "MMLastNuage.xml";
   aDSMFileR = "MMLastNuage.xml";

   std::string aImgList1;
   std::string aImgList2;

   std::string aNameSave = "";

   std::string aInSH = "";

   double aThres = 2;

   std::string aPara3DHL = "";
   std::string aPara3DHR = "";

   std::string aImgPair = "";

   bool bSaveDif = true;
   double dThres = 10;
   ElInitArgMain
    (
        argc,argv,
        LArgMain()
               << EAMC(aImgList1,"ImgList1: All master images (Dir+Pattern, or txt file of image list)")
               << EAMC(aImgList2,"ImgList2: All secondary images (Dir+Pattern, or txt file of image list)")
               << EAMC(aOri1,"Orientation of ImgList1")
               << EAMC(aOri2,"Orientation of ImgList2"),
        LArgMain()
               << aCAS3D.ArgBasic()
               << EAM(aInSH,"InSH",true,"Input Homologue extenion for NB/NT mode, Def=none")
               << EAM(aNameSave,"OutFile",true,"Output file name of 3D points, Def=VisuTiePtIn3D-InSH.txt")
          << EAM(aDSMDirL, "DSMDirL", true, "DSM of master image (for improving the reprojecting accuracy), Def=none")
          << EAM(aDSMDirR, "DSMDirR", true, "DSM of secondary image (for improving the reprojecting accuracy), Def=none")
          << EAM(aDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
          << EAM(aDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml")
          << EAM(aThres, "Thres", true, "The threshold of reprojection error (unit: pixel) when prejecting patch corner to DSM, Def=2")
               << EAM(aPara3DHL, "Para3DHL", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation for points in master images, Def=none")
               << EAM(aPara3DHR, "Para3DHR", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation for points in secondary images, Def=none")
               << EAM(bSaveDif, "SaveDif", false, "Save the difference of the 3D points in master and secondary images, Def=true")
               << EAM(dThres,"Th",true,"Threshold to define outlier (correspondence projected to DSM with coordinate difference larger than this value would be labelled as outlier), Def=10")
               << EAM(aImgPair,"Pair",true,"XML-File of image pair (if this parameter is defined, the input image pairs will be defnied by this instead of ImgList1 and ImgList2 will be ), Def=none")
     );

   if(aNameSave.length() == 0)
       aNameSave = "VisuTiePtIn3D-Homol"+aInSH+".txt";

   StdCorrecNameOrient(aOri1,"./",true);
   StdCorrecNameOrient(aOri2,"./",true);

   cTransform3DHelmert aTrans3DHL(aPara3DHL);
   cTransform3DHelmert aTrans3DHR(aPara3DHR);

   std::vector<string> vImgList1;
   std::vector<string> vImgList2;
   if (ELISE_fp::exist_file(aCAS3D.mDir+"/"+aImgPair) == true)
   {
       GetXmlImgPair(aCAS3D.mDir+"/"+aImgPair, vImgList1, vImgList2);
   }
   else
   {
       std::vector<std::string> aVIm1Tmp;
       std::vector<std::string> aVIm2Tmp;
       GetImgListVec(aImgList1, aVIm1Tmp);
       GetImgListVec(aImgList2, aVIm2Tmp);
       for(int i=0; i<int(aVIm1Tmp.size()); i++){
           for(int j=0; j<int(aVIm2Tmp.size()); j++){
               vImgList1.push_back(aVIm1Tmp[i]);
               vImgList2.push_back(aVIm2Tmp[j]);
           }
       }
   }

   cout<<vImgList1.size()<<" image pairs to be processed."<<endl;
   //cout<<aDir<<",,,"<<aCAS3D.mHomoXml<<endl;
   VisuTiePtIn3D(aCAS3D.mDir, vImgList1, vImgList2, aInSH, aOri1, aOri2, aDSMDirL, aDSMDirR, aDSMFileL, aDSMFileR, aTrans3DHL, aTrans3DHR, aNameSave, aCAS3D.mICNM, aCAS3D.mPrint, aThres, bSaveDif, dThres);

   return EXIT_SUCCESS;
}
