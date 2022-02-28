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
#include "cnpy.h"
#include "cnpy.cpp"



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

bool CheckFile(std::string input_dir, std::string SH, std::string input_pairs)
{
    ifstream in(input_dir+input_pairs);
    std::string s;

    while(getline(in,s))
    {
        std::string aImg1,aImg2;
        std::stringstream is(s);
        is>>aImg1>>aImg2;

        std::string aSHDir = input_dir + "/Homol" + SH+"/";
        ELISE_fp::MkDir(aSHDir);
        std::string aNewDir = aSHDir + "Pastis" + aImg1;
        ELISE_fp::MkDir(aNewDir);
        std::string aNameFile1 = aNewDir + "/"+aImg2+".txt";

        aNewDir = aSHDir + "Pastis" + aImg2;
        ELISE_fp::MkDir(aNewDir);
        std::string aNameFile2 = aNewDir + "/"+aImg1+".txt";

        if (ELISE_fp::exist_file(aNameFile1) == false || ELISE_fp::exist_file(aNameFile2) == false)
        {
            return false;
        }
    }
    return true;
}

void Npz2Homol(Pt2di resize, std::string input_dir, std::string SH, std::string input_pairs, bool keepNpzFile, double aCheckNb, bool bPrint, bool bRotHyp, bool bViz)
{
    std::string aVizDir = input_dir + "/VizSpG/";
    if(bViz == true && ELISE_fp::exist_file(aVizDir) == false)
        ELISE_fp::MkDir(aVizDir);

    ifstream in(input_dir+input_pairs);
    std::string s;
    while(getline(in,s))
    {
        std::string aImg1,aImg2;
        std::stringstream is(s);
        is>>aImg1>>aImg2;

        Tiff_Im aRGBIm1((input_dir+aImg1).c_str());
        Pt2di ImgSzL = aRGBIm1.sz();
        Tiff_Im aRGBIm2((input_dir+aImg2).c_str());
        Pt2di ImgSzR = aRGBIm2.sz();

        std::vector<cElHomographie> aHomoVec;
        std::vector<std::string> aRotVec;
        aHomoVec.push_back(cElHomographie(cElComposHomographie(1,0,0),cElComposHomographie(0,1,0),cElComposHomographie(0,0,1)));
        aRotVec.push_back("");
        if(bRotHyp){
            cElComposHomographie aRotateHX(0, 1, 0);
            cElComposHomographie aRotateHY(-1, 0, ImgSzR.y);
            cElComposHomographie aRotateHZ(0, 0, 1);
            cElHomographie  aRotateH =  cElHomographie(aRotateHX,aRotateHY,aRotateHZ);
            aHomoVec.push_back(aRotateH);
            aRotVec.push_back("_R90");

            aRotateHX = cElComposHomographie(-1, 0, ImgSzR.x);
            aRotateHY = cElComposHomographie(0, -1, ImgSzR.y);
            aRotateHZ = cElComposHomographie(0, 0, 1);
            aRotateH =  cElHomographie(aRotateHX,aRotateHY,aRotateHZ);
            aHomoVec.push_back(aRotateH);
            aRotVec.push_back("_R180");

            aRotateHX = cElComposHomographie(0, -1, ImgSzR.x);
            aRotateHY = cElComposHomographie(1, 0, 0);
            aRotateHZ = cElComposHomographie(0, 0, 1);
            aRotateH =  cElHomographie(aRotateHX,aRotateHY,aRotateHZ);
            aHomoVec.push_back(aRotateH);
            aRotVec.push_back("_R270");
        }

        std::string aBestMatch = "";
        int nMaxTiePtNum = 0;
        std::vector<ElCplePtsHomologues> aTiePtVecFinal;
        for(int k=0; k<int(aRotVec.size()); k++){
            std::string aRot = aRotVec[k];
            cElHomographie  aRotateH = aHomoVec[k];

            //load the entire npz file
            std::string aFileName = StdPrefix(aImg1)+"_"+StdPrefix(aImg2)+aRot+"_matches.npz";
            std::string aFullFileName = input_dir+aFileName;
            if(bViz == true){
                std::string aPngName = StdPrefix(aFileName) + ".png";
                std::string aCom = "mv " + input_dir+aPngName + " " + aVizDir+aPngName;
                //cout<<aCom<<endl;
                System(aCom);
            }

            Tiff_Im aRGBIm2Rot((input_dir+StdPrefix(aImg2)+aRot+"."+StdPostfix(aImg2)).c_str());
            //cout<<input_dir+StdPrefix(aImg2)+aRot+"."+StdPostfix(aImg2)<<endl;
            ImgSzR = aRGBIm2Rot.sz();

            if (ELISE_fp::exist_file(aFullFileName) == false)
            {
                cout<<aFullFileName<<" didn't exist, hence skipped.\n";
                continue;
            }

            cnpy::npz_t my_npz = cnpy::npz_load(aFullFileName);

            cnpy::NpyArray arr_keypt0_raw = my_npz["keypoints0"];
            float* arr_keypt0 = arr_keypt0_raw.data<float>();

            cnpy::NpyArray arr_keypt1_raw = my_npz["keypoints1"];
            float* arr_keypt1 = arr_keypt1_raw.data<float>();
            int nPtNum1 = arr_keypt1_raw.shape[0];

            if(arr_keypt0_raw.shape[0]<=0 || arr_keypt1_raw.shape[0]<=0)
            {
                cout<<aFullFileName<<" has no key point in left or right image, hence skipped.\n";
                continue;
            }

            cnpy::NpyArray arr_matches_raw = my_npz["matches"];
            long * arr_matches = arr_matches_raw.data<long>();
            int nMatchNum = arr_matches_raw.shape[0];
            //cout<<"nMatchNum: "<<nMatchNum<<endl;


            Pt2dr ptScaleL = Pt2dr(1.0, 1.0);
            Pt2dr ptScaleR = Pt2dr(1.0, 1.0);

            //cout<<"Left img size: "<<ImgSzL.x<<", "<<ImgSzL.y<<endl;
            //cout<<"Right img size: "<<ImgSzR.x<<", "<<ImgSzR.y<<endl;

            //scale the pt if the images are resized before applying SuperGlue
            if(resize.x > 0 && resize.y > 0)
            {
                ptScaleL.x = 1.0*ImgSzL.x/resize.x;
                ptScaleL.y = 1.0*ImgSzL.y/resize.y;
                ptScaleR.x = 1.0*ImgSzR.x/resize.x;
                ptScaleR.y = 1.0*ImgSzR.y/resize.y;
            }

            int i;
            std::vector<ElCplePtsHomologues> aTiePtVec;
            int nValidMatchNum = 0;
            for(i=0; i<nMatchNum; i++)
            {
                long nMatchVal = arr_matches[i];
    /*
                if(nMatchVal != -1)
                    cout<<i<<": "<<nMatchVal<<endl;
                //continue;
    */
                //the sizes of arr_keypt0, arr_keypt1 and arr_matches are: Num1*2, Num2*2, Num1*1
                //-1 means no match for arr_keypt0[i]
                if(nMatchVal != -1)
                {
                    if(nMatchVal < 0 || nMatchVal >= nPtNum1)
                        continue;
                    Pt2dr ptL = Pt2dr(arr_keypt0[i*2]*ptScaleL.x, arr_keypt0[i*2+1]*ptScaleL.y);
                    Pt2dr ptR = Pt2dr(arr_keypt1[nMatchVal*2], arr_keypt1[nMatchVal*2+1]);
                    ptR = Pt2dr(ptR.x*ptScaleR.x, ptR.y*ptScaleR.y);
                    //printf("%d: [%.2lf, %.2lf], [%.2lf, %.2lf]", i, ptL.x, ptL.y, ptR.x, ptR.y);
                    ptR = aRotateH(ptR);
                    //printf(", [%.2lf, %.2lf]", ptR.x, ptR.y);
                    if(aCheckNb > 0 && (pow((ptL.x-ptR.x),2)+pow((ptL.y-ptR.y),2) > aCheckNb*aCheckNb)){
                        if(bPrint)
                            printf("Correspondence ([%lf,%lf], [%lf,%lf]) out of search space, hence removed.\n", ptL.x, ptL.y, ptR.x, ptR.y);
                        continue;
                    }
                    aTiePtVec.push_back(ElCplePtsHomologues(ptL, ptR));
                    nValidMatchNum++;
                }
            }
            if(nValidMatchNum > nMaxTiePtNum)
            {
                nMaxTiePtNum = nValidMatchNum;
                aTiePtVecFinal = aTiePtVec;
                aBestMatch = aImg1+" "+StdPrefix(aImg2)+aRot+"."+StdPostfix(aImg2);
            }

            cout<<"Tie-point number in "<<aFullFileName<<": "<<nValidMatchNum<<endl;
            //SaveHomolTxtFile(input_dir, aImg1, aImg2, SH+aRot, aTiePtVec);

            if(keepNpzFile == false)
            {
                std::string cmmd = "rm " + aFullFileName;
                System(cmmd);
            }
        }
        if(bRotHyp)
            cout<<"Best match in 4 rotation hypothesis: "<<aBestMatch<<endl;
        //break;

        //cout<<"nValidMatchNum: "<<nValidMatchNum<<endl;
        SaveHomolTxtFile(input_dir, aImg1, aImg2, SH, aTiePtVecFinal);

/*
        std::string aSHDir = input_dir + "/Homol" + SH+"/";
        ELISE_fp::MkDir(aSHDir);
        std::string aNewDir = aSHDir + "Pastis" + aImg1;
        ELISE_fp::MkDir(aNewDir);
        std::string aNameFile1 = aNewDir + "/"+aImg2+".txt";

        aNewDir = aSHDir + "Pastis" + aImg2;
        ELISE_fp::MkDir(aNewDir);
        std::string aNameFile2 = aNewDir + "/"+aImg1+".txt";

        FILE * fpTiePt1 = fopen(aNameFile1.c_str(), "w");
        FILE * fpTiePt2 = fopen(aNameFile2.c_str(), "w");
        for (ElPackHomologue::iterator itCpl=aTiePtVec.begin();itCpl!=aTiePtVec.end() ; itCpl++)
        {
            ElCplePtsHomologues tiept = itCpl->ToCple();
            fprintf(fpTiePt1, "%lf %lf %lf %lf\n", tiept.P1().x, tiept.P1().y, tiept.P2().x, tiept.P2().y);
            fprintf(fpTiePt2, "%lf %lf %lf %lf\n", tiept.P2().x, tiept.P2().y, tiept.P1().x, tiept.P1().y);
        }
        fclose(fpTiePt1);
        fclose(fpTiePt2);
*/
    }
}

void ReadImgPairs(std::string input_dir, std::string input_pairs)
{
    ifstream in(input_dir+input_pairs);
    std::string s;
    while(getline(in,s))
    {
        //printf("%s\n", s.c_str());
        std::string aImg1,aImg2;
        std::stringstream is(s);
        is>>aImg1>>aImg2;
        //cout<<str2<<","<<str3<<endl;

    }
}

int SuperGlue_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string input_pairs;

   bool bCheckFile = false;

   double aCheckNb = -1;

   std::string aOutput_dir = "";

   bool bRotHyp = false;

   bool bSkipSpG = false;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(input_pairs, "txt file that listed the image pairs"),
        LArgMain()
                    << aCAS3D.ArgBasic()
                    << aCAS3D.ArgSuperGlue()
               << EAM(aOutput_dir, "OutDir", true, "The output directory of the match results of SuperGlue, Def=InDir")
                    << EAM(bCheckFile, "CheckFile", true, "Check if the result files of inter-epoch correspondences exist (if so, skip to avoid repetition), Def=false")
               << EAM(aCheckNb,"CheckNb",true,"Radius of the search space for SuperGlue (which means correspondence [(xL, yL), (xR, yR)] with (xL-xR)*(xL-xR)+(yL-yR)*(yL-yR) > CheckNb*CheckNb will be removed afterwards), Def=-1 (means don't check search space)")
               << EAM(bRotHyp, "RotHyp", true, "Apply rotation hypothesis (if true, the secondary image will be rotated by 90 degreee 4 times and the matching with the largest correspondences will be kept), Def=false")

               << EAM(bSkipSpG, "SkipSpG", true, "Skip executing SuperGlue for testing custom network(for developpers only), Def=false")
               );

   std::vector<std::string> CmmdRmFilesVec;

   std::vector<std::string> input_pairsVec;
   input_pairsVec.push_back(input_pairs);

   if(bRotHyp == true)
   {
       std::string input_dir = aCAS3D.mInput_dir;
       std::string aRotate[4] = {"_R90", "_R180", "_R270"};

       for(int i=0; i<3; i++)
       {
           ifstream in(input_dir+input_pairs);
           std::string s;
           std::string input_pairsNew = StdPrefix(input_pairs)+aRotate[i]+".txt";

           FILE * fpOutput = fopen((input_dir+input_pairsNew).c_str(), "w");
           while(getline(in,s))
           {
               std::string aImg1,aImg2;
               std::stringstream is(s);
               is>>aImg1>>aImg2;

               std::string aImg2_Rotate = StdPrefix(aImg2)+aRotate[i]+"."+StdPostfix(aImg2);
               if (ELISE_fp::exist_file(input_dir+"/"+aImg2_Rotate) == false)
                   RotateImgBy90DegNTimes(input_dir, aImg2, aImg2_Rotate, i+1);
               CmmdRmFilesVec.push_back(aImg2_Rotate);

               fprintf(fpOutput, "%s %s\n", aImg1.c_str(), aImg2_Rotate.c_str());
           }
           fclose(fpOutput);
           input_pairsVec.push_back(input_pairsNew);
           CmmdRmFilesVec.push_back(input_pairsNew);
       }
   }

   if(aOutput_dir.length() == 0)
   {
       aOutput_dir = aCAS3D.mInput_dir;
   }

   std::string strOpt = aCAS3D.mStrOpt;
   std::string strMicMacDirBin = aCAS3D.mStrEntSpG;

   std::string strMicMacDirTPHisto = strMicMacDirBin;
   if(strMicMacDirBin.length()==0)
   {
       strMicMacDirBin = MMBinFile(MM3DStr);
       strMicMacDirTPHisto = strMicMacDirBin.substr(0, strMicMacDirBin.length()-9) + "src/uti_phgrm/TiePHistorical/run.sh";
   }

   bool bExe = true;
   std::string cmmd;
   std::vector<std::string> cmmdVec;
   for(int i=0; i<int(input_pairsVec.size()); i++){
       std::string input_pairsCur = input_pairsVec[i];

       cmmd = strMicMacDirTPHisto + " --input_pairs "+aCAS3D.mInput_dir+input_pairsCur+" --input_dir "+aCAS3D.mInput_dir+" --output_dir "+aOutput_dir + " --max_keypoints "+std::to_string(aCAS3D.mMax_keypoints);
       cmmdVec.push_back(cmmd);

       if(bCheckFile == true && i==0)
       {
           bool bFileExist = CheckFile(aCAS3D.mInput_dir, aCAS3D.mSpGlueOutSH, input_pairsCur);
           if(bFileExist == true)
           {
               printf("%s: Result files already exist, hence skipped\n", input_pairsCur.c_str());
               bExe = false;
           }
       }
   }

   if(bExe)
   {
       std::string aParaOpt = "";
       if(aCAS3D.mViz == true)
           aParaOpt += " --viz";
       if(aCAS3D.mModel == "indoor")
           aParaOpt += " --superglue indoor";
       else
           aParaOpt += " --superglue outdoor";
       if(aCAS3D.mResize.x > 0 && aCAS3D.mResize.y > 0)
           aParaOpt += " --resize " + std::to_string(aCAS3D.mResize.x) + " " + std::to_string(aCAS3D.mResize.y);
       else
           aParaOpt += " --resize -1";

       for(int k=0; k<int(cmmdVec.size()); k++){
           cmmd = cmmdVec[k];
           cmmd += aParaOpt + " " + strOpt;

           printf("%s\n", cmmd.c_str());
           if(bSkipSpG == false)
               System(cmmd);
       }

       Npz2Homol(aCAS3D.mResize, aCAS3D.mInput_dir, aCAS3D.mSpGlueOutSH, input_pairs, aCAS3D.mKeepNpzFile, aCheckNb, aCAS3D.mPrint, bRotHyp, aCAS3D.mViz);

       for(int k=0; k<int(CmmdRmFilesVec.size()); k++){
           std::string CmmdRmFiles = CmmdRmFilesVec[k];
           if (ELISE_fp::exist_file(CmmdRmFiles) == true)
           {
               CmmdRmFiles = "rm "+CmmdRmFiles;
               cout<<CmmdRmFiles<<endl;
               System(CmmdRmFiles);
           }
       }
    }

   return EXIT_SUCCESS;
}
