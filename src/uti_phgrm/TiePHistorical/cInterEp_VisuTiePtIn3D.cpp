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

int Get3DCoors(ElPackHomologue aPackFull, bool bReverse, std::string aDSMDirL, std::string aDSMDirR, std::string aDSMFileL, std::string aDSMFileR, std::string aIm1OriFile, std::string aIm2OriFile, std::vector<Pt3dr> & Pt3dL, std::vector<Pt3dr> & Pt3dR, cTransform3DHelmert aTrans3DHL, cTransform3DHelmert aTrans3DHR, bool bPrint, double aThres)
{
    cGet3Dcoor a3DCoorL(aIm1OriFile);
    TIm2D<float,double> aTImProfPxL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
    cGet3Dcoor a3DCoorR(aIm2OriFile);
    TIm2D<float,double> aTImProfPxR = a3DCoorR.SetDSMInfo(aDSMFileR, aDSMDirR);

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
           pTerr1 = a3DCoorL.Get3Dcoor(p2, aTImProfPxL, bValidL, bPrint, aThres);//, dGSD1);
           pTerr2 = a3DCoorR.Get3Dcoor(p1, aTImProfPxR, bValidR, bPrint, aThres);//, dGSD2);
           pTerr1 = aTrans3DHR.Transform3Dcoor(pTerr1);
           pTerr2 = aTrans3DHL.Transform3Dcoor(pTerr2);
       }
       else
       {
           pTerr1 = a3DCoorL.Get3Dcoor(p1, aTImProfPxL, bValidL, bPrint, aThres);//, dGSD1);
           pTerr2 = a3DCoorR.Get3Dcoor(p2, aTImProfPxR, bValidR, bPrint, aThres);//, dGSD2);
           pTerr1 = aTrans3DHL.Transform3Dcoor(pTerr1);
           pTerr2 = aTrans3DHR.Transform3Dcoor(pTerr2);
       }
       Pt3dL.push_back(pTerr1);
       Pt3dR.push_back(pTerr2);

       nTiePtNum++;
    }
    return nTiePtNum;
}

//void VisuTiePtIn3D(std::string input_dir, std::string output_dir, std::string inSH, std::string outSH, std::string aSubPatchXml, bool bPrint)
void VisuTiePtIn3D(std::string aDir, std::string aImgList1, std::string aImgList2, std::string aInSH, std::string aOri1, std::string aOri2, std::string aDSMDirL, std::string aDSMDirR, std::string aDSMFileL, std::string aDSMFileR, cTransform3DHelmert aTrans3DHL, cTransform3DHelmert aTrans3DHR, std::string aNameSave, cInterfChantierNameManipulateur * aICNM, bool bPrint, double aThres)
{
    std::vector<string> vImgList1;
    std::vector<string> vImgList2;

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

    std::string aDir_inSH = aDir + "/Homol" + aInSH + "/";

    int nTiePtNumTotal = 0;

    std::vector<Pt3dr> Pt3dL;
    std::vector<Pt3dr> Pt3dR;
    for(unsigned int i=0; i<vImgList1.size(); i++)
    {
        std::string aImg1 = vImgList1[i];
        for(unsigned int j=0; j<vImgList2.size(); j++)
        {
            std::string aImg2 = vImgList2[j];

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

            ElPackHomologue aPackInLoc =  ElPackHomologue::FromFile(aNameIn);
            int nTiePtNum = Get3DCoors(aPackInLoc, bReverse, aDSMDirL, aDSMDirR, aDSMFileL, aDSMFileR, aIm1OriFile, aIm2OriFile, Pt3dL, Pt3dR, aTrans3DHL, aTrans3DHR, bPrint, aThres);
            printf("%d tie points in %s\n", nTiePtNum, aNameIn.c_str());
            nTiePtNumTotal += nTiePtNum;
        }
    }

    printf("nTiePtNumTotal: %d\n", nTiePtNumTotal);

    std::string aNameSaveL = StdPrefix(aNameSave) + "_L." + StdPostfix(aNameSave);
    Write3DCoor(aNameSaveL, Pt3dL);
    cout<<"Tie points in master images saved in "<<aNameSaveL<<endl;

    std::string aNameSaveR = StdPrefix(aNameSave) + "_R." + StdPostfix(aNameSave);
    Write3DCoor(aNameSaveR, Pt3dR);
    cout<<"Tie points in secondary images saved in "<<aNameSaveR<<endl;
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

   std::string aNameSave = "VisuTiePtIn3D.txt";

   std::string aInSH = "";

   double aThres = 2;

   std::string aPara3DHL = "";
   std::string aPara3DHR = "";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()
               << EAMC(aImgList1,"ImgList1: The list that contains all the master images")
               << EAMC(aImgList2,"ImgList2: The list that contains all the secondary images")
               << EAMC(aOri1,"Orientation of ImgList1")
               << EAMC(aOri2,"Orientation of ImgList2"),
        LArgMain()
               << aCAS3D.ArgBasic()
               << EAM(aInSH,"InSH",true,"Input Homologue extenion for NB/NT mode, Def=none")
               << EAM(aNameSave,"OutFile",true,"Output file name of 3D points, Def=VisuTiePtIn3D.txt")
          << EAM(aDSMDirL, "DSMDirL", true, "DSM of master image (for improving the reprojecting accuracy), Def=none")
          << EAM(aDSMDirR, "DSMDirR", true, "DSM of secondary image (for improving the reprojecting accuracy), Def=none")
          << EAM(aDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
          << EAM(aDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml")
          << EAM(aThres, "Thres", true, "The threshold of reprojection error (unit: pixel) when prejecting patch corner to DSM, Def=2")
               << EAM(aPara3DHL, "Para3DHL", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation for points in master images, Def=none")
               << EAM(aPara3DHR, "Para3DHR", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation for points in secondary images, Def=none")
     );

   StdCorrecNameOrient(aOri1,"./",true);
   StdCorrecNameOrient(aOri2,"./",true);

   cTransform3DHelmert aTrans3DHL(aPara3DHL);
   cTransform3DHelmert aTrans3DHR(aPara3DHR);

   //cout<<aDir<<",,,"<<aCAS3D.mHomoXml<<endl;
   VisuTiePtIn3D(aCAS3D.mDir, aImgList1, aImgList2, aInSH, aOri1, aOri2, aDSMDirL, aDSMDirR, aDSMFileL, aDSMFileR, aTrans3DHL, aTrans3DHR, aNameSave, aCAS3D.mICNM, aCAS3D.mPrint, aThres);

   return EXIT_SUCCESS;
}
