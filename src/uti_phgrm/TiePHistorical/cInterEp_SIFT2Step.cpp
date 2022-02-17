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
//#include "cnpy.cpp"

extern ElSimilitude SimilRobustInit(const ElPackHomologue & aPackFull,double aPropRan,int aNbTir);

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


ElSimilitude EstimateHomography(std::string aDir, std::string aImg1, std::string aImg2, std::string outSH, double dScale, bool bCheckFile, double dScaleL, double dScaleR, double aR2DTh1st, double aR2dIter1st, std::vector<Siftator::SiftPoint> aVSiftOriL, std::vector<Siftator::SiftPoint> aVSiftOriR, bool bPrint, bool bCheckSclRot, double aThreshScaleR2D, double aThreshAngleR2D)
{
    if(IsHomolFileExist(aDir, aImg1, aImg2, outSH+"-Rough", bCheckFile) == false)
    {
        std::vector<Siftator::SiftPoint> aVSiftL;
        std::vector<Siftator::SiftPoint> aVSiftR;
        FilterKeyPt(aVSiftOriL, aVSiftL, dScale);
        FilterKeyPt(aVSiftOriR, aVSiftR, dScale);

        printf("Original key point number of master image for 1th step: %d.\nOriginal key point number of secondary image for 1th step: %d.\nFiltered key point number of master image for 1th step: %d.\nFiltered key point number of secondary image for 1th step: %d.\n", int(aVSiftOriL.size()), int(aVSiftOriR.size()), int(aVSiftL.size()), int(aVSiftR.size()));

        std::vector<int> matchIDL;
        std::vector<int> matchIDR;
        MatchOneWay(matchIDL, aVSiftL, aVSiftR, false);
        MatchOneWay(matchIDR, aVSiftR, aVSiftL, false);

        std::vector<Pt2di> match;
        MutualNearestNeighbor(true, matchIDL, matchIDR, match);

        SaveSIFTHomolFile(aDir, aImg1, aImg2, outSH+"-Rough", match, aVSiftL, aVSiftR, bPrint, 1, 1, bCheckSclRot);
    }

    std::string CurSH = outSH+"-Rough";
    std::string aComm;

    if(IsHomolFileExist(aDir, aImg1, aImg2, CurSH+"-2DRANSAC", bCheckFile) == false)
    {
        std::string aParaOpt = " Dir="+aDir;
        if(bCheckSclRot == true)
        {
            aParaOpt += " CheckSclRot=" + ToString(bCheckSclRot);
            aParaOpt += " ScaleTh=" + ToString(aThreshScaleR2D);
            aParaOpt += " AngleTh=" + ToString(aThreshAngleR2D);
        }
        aComm = MMBinFile(MM3DStr) + "TestLib RANSAC R2D " + aImg1 + BLANK + aImg2 + " 2DRANInSH="+CurSH + " 2DRANTh="+ToString(aR2DTh1st) + " 2DIter="+ToString(aR2dIter1st) + aParaOpt;
        cout<<aComm<<endl;
        System(aComm);
    }
    /*s
    CurSH += "-2DRANSAC";
    aComm = MMBinFile(MM3DStr) + "HomolFilterMasq \"" + aImg1 + "|" + aImg2 + "\" PostIn="+CurSH + BLANK + "PostOut="+CurSH+"-dat ANM=1 ExpTxt=1 ExpTxtOut=0";
    cout<<aComm<<endl;
    System(aComm);

    std::string aNameFile1 = aDir + "/Homol" + CurSH + "-dat/Pastis" + aImg1 + "/"+aImg2+".dat";
    */
    std::string aNameFile1 = aDir + "/Homol" + CurSH + "/Pastis" + aImg1 + "/"+aImg2+".txt";
    if (ELISE_fp::exist_file(aNameFile1) == true)
    {
        ElPackHomologue aPack =  ElPackHomologue::FromFile(aNameFile1);
        double aPropRan = 0.8;
        ElSimilitude aSimCur = SimilRobustInit(aPack,aPropRan,1);
        if(false)
        {
            Pt2dr tr, sc;
            tr = aSimCur.tr();
            sc = aSimCur.sc();
            cout<<"Translation_X, Translation_Y, scale, rotation:"<<endl;
            printf("%lf  %lf  %lf  %lf\n", tr.x, tr.y, sc.x, sc.y);
        }

        return aSimCur;
    }
    else
        return ElSimilitude(Pt2dr(0,0), Pt2dr(1,0));
}

void PredictKeyPt(std::vector<Pt2dr>& aVPredL, std::vector<Siftator::SiftPoint> aVSiftL, ElSimilitude aSimCur, bool bPrint)
{
    if(1)
    {
        Pt2dr tr, sc;
        tr = aSimCur.tr();
        sc = aSimCur.sc();
        cout<<"Translation_X, Translation_Y, scale, rotation:"<<endl;
        printf("%lf  %lf  %lf  %lf\n", tr.x, tr.y, sc.x, sc.y);
    }

    int nSizeL = aVSiftL.size();

    for(int i=0; i<nSizeL; i++)
    {
        Pt2dr aPL = Pt2dr(aVSiftL[i].x, aVSiftL[i].y);
        Pt2dr aPLPred = aSimCur(aPL);

        if(bPrint)
            if(aPLPred.x>0 && aPLPred.y>0 && aPLPred.x<3533 && aPLPred.y<3533)
                printf("%dth pt: [%.2lf, %.2lf]; prediction: [%.2lf, %.2lf]\n", i, aPL.x, aPL.y, aPLPred.x, aPLPred.y);
        aVPredL.push_back(aPLPred);
    }
}
/*
ElSimilitude ReverseSimi(ElSimilitude aSimCur)
{
    double tranX, tranY, scale, rotation;
    tranX = -aSimCur.tr().x;
    tranY = -aSimCur.tr().y;
    scale = 1.0/aSimCur.sc().x;
    rotation = -aSimCur.sc().y;
    ElSimilitude aSimRev = ElSimilitude(Pt2dr(tranX, tranY), Pt2dr(scale,rotation));
    Pt2dr tr, sc;
    tr = aSimRev.tr();
    sc = aSimRev.sc();
    cout<<"Reversed similarity: Translation_X, Translation_Y, scale, rotation:"<<endl;
    printf("%lf  %lf  %lf  %lf\n", tr.x, tr.y, sc.x, sc.y);
    return aSimRev;
}
*/
void SIFT2Step(std::string aDir, std::string aImg1, std::string aImg2, std::string outSH, double dScale, bool bCheckFile, double dScaleL, double dScaleR, double aR2DTh1st, double aR2dIter1st, ElSimilitude aSimCur, double dSearchSpace, bool bPrint,  bool aCheckScale, bool aCheckAngle, double aThreshScale, double aThreshAngle, bool bSkip1stSIFT, bool bSkip2ndSIFT, bool bCheckSclRot, double aThreshScaleR2D, double aThreshAngleR2D)
{
    Tiff_Im aRGBIm1((aDir+"/"+aImg1).c_str());
    Pt2di ImgSzL = aRGBIm1.sz();
    Tiff_Im aRGBIm2((aDir+"/"+aImg2).c_str());
    Pt2di ImgSzR = aRGBIm2.sz();

    std::string aImg1Key = ExtractSIFT(aImg1, aDir, dScaleL);
    std::string aImg2Key = ExtractSIFT(aImg2, aDir, dScaleR);

/*
    std::string aImgScaledName1 = GetScaledImgName(aImg1, ImgSzL, dScaleL);
    std::string aImgScaledName2 = GetScaledImgName(aImg2, ImgSzR, dScaleR);
    std::string aImg1Key = aDir + "/Pastis/LBPp" + aImgScaledName1+".dat"; //aImg1.substr(0, aImg1.rfind(".")) + ".key";
    std::string aImg2Key = aDir + "/Pastis/LBPp" + aImgScaledName2+".dat"; //aImg2.substr(0, aImg2.rfind(".")) + ".key";

    if (ELISE_fp::exist_file(aImg1Key) == false)
        ExtractSIFT(aImg1, aDir, dScaleL);
    if (ELISE_fp::exist_file(aImg2Key) == false)
        ExtractSIFT(aImg2, aDir, dScaleR);
*/
    std::vector<Siftator::SiftPoint> aVSiftOriL;
    std::vector<Siftator::SiftPoint> aVSiftOriR;
    if(read_siftPoint_list(aImg1Key,aVSiftOriL) == false || read_siftPoint_list(aImg2Key,aVSiftOriR) == false)
    {
        cout<<"Read SIFT of "<<aImg1Key<<" or "<<aImg2Key<<" went wrong."<<endl;
        return;
    }

    ScaleKeyPt(aVSiftOriL, dScaleL);
    ScaleKeyPt(aVSiftOriR, dScaleR);

    if(bSkip1stSIFT == false)
        aSimCur = EstimateHomography(aDir, aImg1, aImg2, outSH, dScale, bCheckFile, dScaleL, dScaleR, aR2DTh1st, aR2dIter1st, aVSiftOriL, aVSiftOriR, bPrint, bCheckSclRot, aThreshScaleR2D, aThreshAngleR2D);

    if(bSkip2ndSIFT == false && IsHomolFileExist(aDir, aImg1, aImg2, outSH, bCheckFile) == false)
    {
        printf("Key point number of master image for 2th step: %d.\nKey point number of secondary image for 2th step: %d.\n", int(aVSiftOriL.size()), int(aVSiftOriR.size()));

        std::vector<Pt2dr> aVPredL;
        std::vector<Pt2dr> aVPredR;
        ElSimilitude aSimInv = aSimCur.inv();
        PredictKeyPt(aVPredL, aVSiftOriL, aSimCur, 0);
        PredictKeyPt(aVPredR, aVSiftOriR, aSimInv, 0); //ReverseSimi(aSimCur));

        bool bPredict = true;
        bool bRatioT = true;
        bool bMutualNN = true;

        std::vector<int> matchIDL;
        std::vector<int> matchIDR;
        cout<<"Start matching for left image."<<endl;
        int nMatchesL = MatchOneWay(matchIDL, aVSiftOriL, aVSiftOriR, bRatioT, aVPredL, ImgSzR, aCheckScale, aCheckAngle, dSearchSpace, bPredict, aSimCur.sc().x, aSimCur.sc().y, aThreshScale, aThreshAngle);
        cout<<"nMatchesL: "<<nMatchesL<<endl;
        cout<<"Start matching for right image."<<endl;
        int nMatchesR = MatchOneWay(matchIDR, aVSiftOriR, aVSiftOriL, bRatioT, aVPredR, ImgSzL, aCheckScale, aCheckAngle, dSearchSpace, bPredict, aSimInv.sc().x, aSimInv.sc().y, aThreshScale, aThreshAngle);
        cout<<"nMatchesR: "<<nMatchesR<<endl;

        std::vector<Pt2di> match;
        MutualNearestNeighbor(bMutualNN, matchIDL, matchIDR, match);
        SaveSIFTHomolFile(aDir, aImg1, aImg2, outSH, match, aVSiftOriL, aVSiftOriR, bPrint);

        cout<<"Extracted tie point number: "<<match.size()<<endl;
    }
}

ElSimilitude ReadSimi(std::string aDir, std::string aFile, bool& bSimi)
{
    bSimi = true;
    ifstream in(aDir +"/"+ aFile);
    if (!in) {
        cout << "error opening file " << aDir +"/"+ aFile << endl;
        bSimi = false;
        return ElSimilitude();
    }

    double tranX, tranY, scale, rotation;
    std::string s;
    getline(in,s);
    getline(in,s);
    std::stringstream is(s);
    is>>tranX>>tranY>>scale>>rotation;
    return ElSimilitude(Pt2dr(tranX, tranY), Pt2dr(scale,rotation));
}

int SIFT2Step_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aImg1;
   std::string aImg2;
   std::string aOutSH = "-SIFT2Step";
   double dScale = 3;
   bool bCheckFile = false;
   double dScaleL = 1;
   double dScaleR = 1;
   double aR2dIter1st = 1000;
   double aR2DTh1st = 30;
   std::string aSimiFile = "";
   double dSearchSpace = 100;
   bool   aCheckScale = true;
   bool   aCheckAngle = true;
   double aThreshScale = 0.2;
   double aThreshAngle = 30;

   std::string aOri1= "";
   std::string aOri2 = "";

   std::string aDSMDirL = "";
   std::string aDSMDirR = "";
   std::string aDSMFileL;
   std::string aDSMFileR;

   aDSMFileL = "MMLastNuage.xml";
   aDSMFileR = "MMLastNuage.xml";

   bool bSkip1stSIFT = false;
   bool bSkip2ndSIFT = false;

   double aR2dIter2nd=1000;
   double aR2DTh2nd=10;

   double aR3DIteration = 1000;
   double aR3DThreshold = -1;
   int aMinPt = 10;

   bool bCheckSclRot = false;
   double aThreshScaleR2D = 0.2;
   double aThreshAngleR2D = 30;

   ElInitArgMain
    (
        argc,argv,
               LArgMain()   << EAMC(aImg1,"Master image name")
                      << EAMC(aImg2,"Secondary image name"),
        LArgMain()
                    << aCAS3D.ArgBasic()
               << EAM(dScale, "Scale", true, "Scale factor to downsample images in order to estimate homography, Def=3")
               << EAM(aOutSH, "OutSH", true, "Homologue extenion for NB/NT mode, Def=-SIFT2Step")
               << EAM(bCheckFile, "CheckFile", true, "Check if the result files of correspondences exist (if so, skip to avoid repetition), Def=false")
               << EAM(dScaleL, "ScaleL", true, "Extract SIFT points on master images downsampled with a factor of \"ScaleL\", Def=1")
               << EAM(dScaleR, "ScaleR", true, "Extract SIFT points on secondary images downsampled with a factor of \"ScaleR\", Def=1")
               << EAM(aR2DTh1st,"2DRANTh1st",true,"2D RANSAC threshold after the first SIFT, Def=30")
               << EAM(aR2dIter1st,"2DIter1st",true,"2D RANSAC iteration after the first SIFT, Def=1000")

               << EAM(bCheckSclRot, "CheckSclRot", true, "Check the scale and rotation consistency in 2D RANSAC after the first SIFT (please make sure you saved the scale and rotation in \"Homol-SIFT2Step-Rough_SclRot\" if you set this parameter to true), Def=false")
               << EAM(aThreshScaleR2D, "ScaleThR2D",true, "The threshold for checking scale ratio in 2D RANSAC after the first SIFT, Def=0.2; (0.2 means the ratio of master and secondary SIFT scale between [(1-0.2)*Ref, (1+0.2)*Ref] is considered valide. Ref is automatically calculated by reprojection.)")
               << EAM(aThreshAngleR2D, "AngleThR2D",true, "The threshold for checking angle difference in 2D RANSAC after the first SIFT, Def=30; (30 means the difference of master and secondary SIFT angle between [Ref - 30 degree, Ref + 30 degree] is considered valide. Ref is automatically calculated by reprojection.)")

               << EAM(aSimiFile,"SimiFile",true,"input file that records the similarity transformation, Def=none")
               << EAM(dSearchSpace,"SearchSpace",true,"Radius of the search space for the second SIFT (the search space is the circle with the center on the predicted point), Def=100")
               << EAM(aCheckScale, "CheckScale",true, "Check the scale of the candidate tie points on SIFT, Def=true")
               << EAM(aCheckAngle, "CheckAngle",true, "Check the angle of the candidate tie points on SIFT, Def=true")
               << EAM(aThreshScale, "ScaleTh",true, "The threshold for checking scale ratio in the second SIFT, Def=0.2; (0.2 means the ratio of master and secondary SIFT scale between [(1-0.2)*Ref, (1+0.2)*Ref] is considered valide.)")
               << EAM(aThreshAngle, "AngleTh",true, "The threshold for checking angle difference in the second SIFT, Def=30; (30 means the difference of master and secondary SIFT angle between [Ref - 30 degree, Ref + 30 degree] is considered valide.)")
//               << aCAS3D.Arg2DRANSAC()
               << EAM(aR2dIter2nd,"2DIter2nd",true,"2D RANSAC iteration after the second SIFT, Def=1000")
               << EAM(aR2DTh2nd,"2DRANTh2nd",true,"2D RANSAC threshold after the second SIFT, Def=10")

               << EAM(aOri1, "OriL", true, "Orientation of master image (for applying 3D RANSAC after the second SIFT. 2D RANSAC would be applied instead, if this parameter is empty), Def=none")
               << EAM(aOri2, "OriR", true, "Orientation of secondary image (for applying 3D RANSAC after the second SIFT. 2D RANSAC would be applied instead, if this parameter is empty), Def=none")
               << EAM(aDSMDirL, "DSMDirL", true, "DSM directory of master image (for applying 3D RANSAC after the second SIFT. 2D RANSAC would be applied instead, if this parameter is empty), Def=none")
               << EAM(aDSMDirR, "DSMDirR", true, "DSM directory of secondary image (for applying 3D RANSAC after the second SIFT. 2D RANSAC would be applied instead, if this parameter is empty), Def=none")
               << EAM(aDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
               << EAM(aDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml")
//               << aCAS3D.Arg3DRANSAC()
               //<< EAM(aInSH,"3DRANInSH",true,"Input Homologue extenion for NB/NT mode for 3D RANSAC, Def=none")
               //<< EAM(aR3DOutSH,"3DRANOutSH",true,"Output Homologue extenion for NB/NT mode after 3D RANSAC, Def='InSH'-GlobalR3D")
               << EAM(aR3DIteration,"3DIter",true,"3D RANSAC iteration, Def=1000")
               << EAM(aR3DThreshold,"3DRANTh",true,"3D RANSAC threshold, Def=10*(GSD of secondary image)")
               << EAM(aMinPt,"MinPt",true,"Minimun number of input correspondences required, Def=10")

               << EAM(bSkip1stSIFT, "Skip1stSIFT", true, "Skip the first step of SIFT matching, Def=false")
               << EAM(bSkip2ndSIFT, "Skip2ndSIFT", true, "Skip the second step of SIFT matching, Def=false")
         );

   std::string aDir = aCAS3D.mDir;
       //aThreshAngleR2D = aThreshAngleR2D*3.14/180;

   string::size_type position;
   position = aDir.find_last_of("/");
   cout<<position<<endl;
   if (position != aDir.length()-1){
       aDir = aDir+"/";
   }


       bool bR3D = true;
       if(aOri1.length()==0 && aOri2.length()==0 && aDSMDirL.length()==0 && aDSMDirR.length()==0)
       {
           bR3D = false;
       }

       if (ELISE_fp::exist_file(aDir+aImg1) == false || ELISE_fp::exist_file(aDir+aImg2) == false)
       {
           cout<<aDir+aImg1<<" or "<<aDir+aImg2<<" didn't exist, hence skipped (SIFT2Step_main)"<<endl;
           return EXIT_SUCCESS;
       }

       //double dThreshAngle = dThreshAngle*3.14/180;

       ElSimilitude aSimCur;
       bool bSimi = false;
       if(aSimiFile.length() == 0)
           aSimCur = ElSimilitude();
       else
           aSimCur = ReadSimi(aDir, aSimiFile, bSimi);//aSimiVec[i];

       if(bSimi == true)
           bSkip1stSIFT = true;

       SIFT2Step(aDir, aImg1, aImg2, aOutSH, dScale, bCheckFile, dScaleL, dScaleR, aR2DTh1st, aR2dIter1st, aSimCur, dSearchSpace, aCAS3D.mPrint, aCheckScale, aCheckAngle, aThreshScale, aThreshAngle, bSkip1stSIFT, bSkip2ndSIFT, bCheckSclRot, aThreshScaleR2D, aThreshAngleR2D);

       if(bSkip2ndSIFT == false)
       {
           if(bR3D == false)
           {
               if(IsHomolFileExist(aDir, aImg1, aImg2, aOutSH+"-2DRANSAC", bCheckFile) == false)
               {
                   std::string aOptPara= " Dir="+aDir; //aCAS3D.ComParamRANSAC2D();
                   if (EAMIsInit(&aR2dIter2nd))          aOptPara +=  " 2DIter=" + ToString(aR2dIter2nd);
                   if (EAMIsInit(&aR2DTh2nd))          aOptPara +=  " 2DRANTh=" + ToString(aR2DTh2nd);
                   std::string CurSH = aOutSH;
                   std::string aComm;
                   aComm = MMBinFile(MM3DStr) + "TestLib RANSAC R2D " + aImg1 + BLANK + aImg2 + BLANK + "2DRANInSH="+CurSH + aOptPara;
                   cout<<aComm<<endl;
                   System(aComm);
               }
           }
           else {
               if(IsHomolFileExist(aDir, aImg1, aImg2, aOutSH+"-3DRANSAC", bCheckFile) == false)
               {
                   std::string aOptPara = " Dir="+aDir; //=aCAS3D.ComParamRANSAC3D();
                   if (EAMIsInit(&aR3DIteration))   aOptPara +=  " 3DIter=" + ToString(aR3DIteration);
                   if (EAMIsInit(&aR3DThreshold))   aOptPara +=  " 3DRANTh=" + ToString(aR3DThreshold);
                   if (EAMIsInit(&aMinPt))          aOptPara +=  " MinPt=" + ToString(aMinPt);
                   std::string CurSH = aOutSH;
                   aOptPara +=  " 3DRANInSH=" + CurSH;
                   aOptPara +=  " 3DRANOutSH=" + CurSH+"-3DRANSAC";
                   aOptPara +=  " DSMDirL=" + aDSMDirL;
                   aOptPara +=  " DSMDirR=" + aDSMDirR ;

                   std::string aComm;
                   aComm = MMBinFile(MM3DStr) + "TestLib RANSAC R3D " + aImg1 + BLANK + aImg2 + BLANK + aOri1 + BLANK + aOri2 + aOptPara;
                   cout<<aComm<<endl;
                   System(aComm);
               }
           }
       }

   return EXIT_SUCCESS;
}

int SIFT2StepFile_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aImgPair;
   std::string aOutSH = "-SIFT2Step";
   double dScale = 3;
   bool bCheckFile = false;
   double dScaleL = 1;
   double dScaleR = 1;
   double aR2DTh1st = 30;
   double aR2dIter1st = 1000;
   double dSearchSpace = 100;
   bool   aCheckScale = true;
   bool   aCheckAngle = true;
   double aThreshScale = 0.2;
   double aThreshAngle = 30;

   std::string aOri1 = "";
   std::string aOri2 = "";

   std::string aDSMDirL = "";
   std::string aDSMDirR = "";
   std::string aDSMFileL;
   std::string aDSMFileR;

   aDSMFileL = "MMLastNuage.xml";
   aDSMFileR = "MMLastNuage.xml";

   bool bSkip1stSIFT = false;
   bool bSkip2ndSIFT = false;

   double aR2dIter2nd=1000;
   double aR2DTh2nd=10;

   double aR3DIteration = 1000;
   double aR3DThreshold = -1;
   int aMinPt = 10;

   bool bCheckSclRot = false;
   double aThreshScaleR2D = 0.2;
   double aThreshAngleR2D = 30;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()   << EAMC(aImgPair,"XML-File of image pair"),
        LArgMain()
                    << aCAS3D.ArgBasic()
               << EAM(dScale, "Scale", true, "Scale factor to downsample images in order to estimate homography, Def=3")
               << EAM(aOutSH, "OutSH", true, "Homologue extenion for NB/NT mode, Def=-SIFT2Step")
               << EAM(bCheckFile, "CheckFile", true, "Check if the result files of correspondences exist (if so, skip to avoid repetition), Def=false")
               << EAM(dScaleL, "ScaleL", true, "Extract SIFT points on master images downsampled with a factor of \"ScaleL\", Def=1")
               << EAM(dScaleR, "ScaleR", true, "Extract SIFT points on secondary images downsampled with a factor of \"ScaleR\", Def=1")
               << EAM(aR2DTh1st,"2DRANTh1st",true,"2D RANSAC threshold after the first SIFT, Def=30")
               << EAM(aR2dIter1st,"2DIter1st",true,"2D RANSAC iteration after the first SIFT, Def=1000")

               << EAM(bCheckSclRot, "CheckSclRot", true, "Check the scale and rotation consistency in 2D RANSAC after the first SIFT (please make sure you saved the scale and rotation in \"Homol-SIFT2Step-Rough_SclRot\" if you set this parameter to true), Def=false")
               << EAM(aThreshScaleR2D, "ScaleThR2D",true, "The threshold for checking scale ratio in 2D RANSAC after the first SIFT, Def=0.2; (0.2 means the ratio of master and secondary SIFT scale between [(1-0.2)*Ref, (1+0.2)*Ref] is considered valide. Ref is automatically calculated by reprojection.)")
               << EAM(aThreshAngleR2D, "AngleThR2D",true, "The threshold for checking angle difference in 2D RANSAC after the first SIFT, Def=30; (30 means the difference of master and secondary SIFT angle between [Ref - 30 degree, Ref + 30 degree] is considered valide. Ref is automatically calculated by reprojection.)")

               << EAM(dSearchSpace,"SearchSpace",true,"Radius of the search space for the second SIFT (the search space is the circle with the center on the predicted point), Def=100")
               << EAM(aCheckScale, "CheckScale",true, "Check the scale of the candidate tie points on SIFT, Def=true")
               << EAM(aCheckAngle, "CheckAngle",true, "Check the angle of the candidate tie points on SIFT, Def=true")
               << EAM(aThreshScale, "ScaleTh",true, "The threshold for checking scale ratio, Def=0.2; (0.2 means the ratio of master and secondary SIFT scale between [(1-0.2)*Ref, (1+0.2)*Ref] is considered valide.)")
               << EAM(aThreshAngle, "AngleTh",true, "The threshold for checking angle difference, Def=30; (30 means the difference of master and secondary SIFT angle between [Ref - 30 degree, Ref + 30 degree] is considered valide.)")
//               << aCAS3D.Arg2DRANSAC()
               << EAM(aR2dIter2nd,"2DIter2nd",true,"2D RANSAC iteration after the second SIFT, Def=1000")
               << EAM(aR2DTh2nd,"2DRANTh2nd",true,"2D RANSAC threshold after the second SIFT, Def=10")

               << EAM(aOri1, "OriL", true, "Orientation of master image (for applying 3D RANSAC after the second SIFT. 2D RANSAC would be applied instead, if this parameter is empty), Def=none")
               << EAM(aOri2, "OriR", true, "Orientation of secondary image (for applying 3D RANSAC after the second SIFT. 2D RANSAC would be applied instead, if this parameter is empty), Def=none")
               << EAM(aDSMDirL, "DSMDirL", true, "DSM directory of master image (for applying 3D RANSAC after the second SIFT. 2D RANSAC would be applied instead, if this parameter is empty), Def=none")
               << EAM(aDSMDirR, "DSMDirR", true, "DSM directory of secondary image (for applying 3D RANSAC after the second SIFT. 2D RANSAC would be applied instead, if this parameter is empty), Def=none")
               << EAM(aDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
               << EAM(aDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml")
//               << aCAS3D.Arg3DRANSAC()
               << EAM(aR3DIteration,"3DIter",true,"3D RANSAC iteration, Def=1000")
               << EAM(aR3DThreshold,"3DRANTh",true,"3D RANSAC threshold, Def=10*(GSD of secondary image)")
               << EAM(aMinPt,"MinPt",true,"Minimun number of input correspondences required, Def=10")

               << EAM(bSkip1stSIFT, "Skip1stSIFT", true, "Skip the first step of SIFT matching, Def=false")
               << EAM(bSkip2ndSIFT, "Skip2ndSIFT", true, "Skip the second step of SIFT matching, Def=false")
               );

   std::vector<std::string> aOverlappedImgL;
   std::vector<std::string> aOverlappedImgR;
   int nPairNum = GetXmlImgPair(aImgPair, aOverlappedImgL, aOverlappedImgR);

   std::string aOptPara=" Dir="+aCAS3D.mDir;
   if (EAMIsInit(&dScale))            aOptPara += " Scale=" + ToString(dScale);
   if (EAMIsInit(&aOutSH))            aOptPara += " OutSH=" + aOutSH;
   if (EAMIsInit(&bCheckFile))        aOptPara += " CheckFile=" + ToString(bCheckFile);
   if (EAMIsInit(&dScaleL))           aOptPara += " ScaleL=" + ToString(dScaleL);
   if (EAMIsInit(&dScaleR))           aOptPara += " ScaleR=" + ToString(dScaleR);
   if (EAMIsInit(&aR2DTh1st))         aOptPara += " 2DRANTh1st=" + ToString(aR2DTh1st);
   if (EAMIsInit(&aR2dIter1st))       aOptPara += " 2DIter1st=" + ToString(aR2dIter1st);
   if (EAMIsInit(&dSearchSpace))      aOptPara += " SearchSpace=" + ToString(dSearchSpace);
   if (EAMIsInit(&aCheckScale))       aOptPara += " CheckScale=" + ToString(aCheckScale);
   if (EAMIsInit(&aCheckAngle))       aOptPara += " CheckAngle=" + ToString(aCheckAngle);
   if (EAMIsInit(&aThreshScale))      aOptPara += " ScaleTh=" + ToString(aThreshScale);
   if (EAMIsInit(&aThreshAngle))      aOptPara += " AngleTh=" + ToString(aThreshAngle);

   if (EAMIsInit(&aOri1))             aOptPara += " OriL=" + aOri1;
   if (EAMIsInit(&aOri2))             aOptPara += " OriR=" + aOri2;
   if (EAMIsInit(&aDSMDirL))          aOptPara += " DSMDirL=" + aDSMDirL;
   if (EAMIsInit(&aDSMDirR))          aOptPara += " DSMDirR=" + aDSMDirR;
   if (EAMIsInit(&aDSMFileL))         aOptPara += " DSMFileL=" + aDSMFileL;
   if (EAMIsInit(&aDSMFileR))         aOptPara += " DSMFileR=" + aDSMFileR;

   if (EAMIsInit(&bSkip1stSIFT))      aOptPara += " Skip1stSIFT=" + ToString(bSkip1stSIFT);
   if (EAMIsInit(&bSkip2ndSIFT))      aOptPara += " Skip2ndSIFT=" + ToString(bSkip2ndSIFT);

   if (EAMIsInit(&aR2dIter2nd))      aOptPara += " 2DIter2nd=" + ToString(aR2dIter2nd);
   if (EAMIsInit(&aR2DTh2nd))        aOptPara += " 2DRANTh2nd=" + ToString(aR2DTh2nd);

   if (EAMIsInit(&aR3DIteration))      aOptPara += " 3DIter=" + ToString(aR3DIteration);
   if (EAMIsInit(&aR3DThreshold))      aOptPara += " 3DRANTh=" + ToString(aR3DThreshold);
   if (EAMIsInit(&aMinPt))             aOptPara += " MinPt=" + ToString(aMinPt);

   if (EAMIsInit(&bCheckSclRot))      aOptPara += " CheckSclRot=" + ToString(bCheckSclRot);
   if (EAMIsInit(&aThreshScaleR2D))   aOptPara += " ScaleTh=" + ToString(aThreshScaleR2D);
   if (EAMIsInit(&aThreshAngleR2D))   aOptPara += " AngleTh=" + ToString(aThreshAngleR2D);

   std::list<std::string> aCommsSIFT;
   std::list<std::string> aComms;

   std::vector<std::string> aImgs1;
   std::vector<std::string> aImgs2;
   GetUniqImgList(aOverlappedImgL, aImgs1);
   GetUniqImgList(aOverlappedImgR, aImgs2);

   std::string aComm;
   for(int i=0; i<int(aImgs1.size()); i++)
   {
       std::string aImg1 = aImgs1[i];
           aComm = MMBinFile(MM3DStr) + "TestLib ExtractSIFT " + aImg1 + " Scale="+ToString(dScaleL);
           cout<<aComm<<endl;
           aCommsSIFT.push_back(aComm);
   }
   cout<<aImgs1.size()<<" master images"<<endl;

   for(int i=0; i<int(aImgs2.size()); i++)
   {
       std::string aImg2 = aImgs2[i];
           aComm = MMBinFile(MM3DStr) + "TestLib ExtractSIFT " + aImg2 + " Scale="+ToString(dScaleR);
           cout<<aComm<<endl;
           aCommsSIFT.push_back(aComm);
   }
   cout<<aImgs2.size()<<" secondary images"<<endl;

   cEl_GPAO::DoComInParal(aCommsSIFT);

   for(int i=0; i<nPairNum; i++)
   {
       std::string aImg1 = aOverlappedImgL[i];
       std::string aImg2 = aOverlappedImgR[i];

       if (ELISE_fp::exist_file(aImg1) == false || ELISE_fp::exist_file(aImg2) == false)
       {
           cout<<aImg1<<" or "<<aImg2<<" didn't exist, hence skipped (SIFT2StepFile_main)"<<endl;
           continue;
       }

       std::string aComm;
       aComm = MMBinFile(MM3DStr) + "TestLib SIFT2Step " + aImg1 + BLANK + aImg2 + BLANK + aOptPara;
       cout<<aComm<<endl;
       //System(aComm);
       aComms.push_back(aComm);
   }
   cEl_GPAO::DoComInParal(aComms);

   return EXIT_SUCCESS;
}

int ExtractSIFT_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aFullPattern1;
   std::string aImg1;
   double dScale = 1;

   ElInitArgMain
    (
        argc,argv,
               LArgMain() << EAMC(aFullPattern1,"Image name (Dir+Pattern, or txt file of image list)"),
        LArgMain()
               << aCAS3D.ArgBasic()
               << EAM(dScale, "Scale", true, "Scale factor to downsample images, Def=1")
               //<< EAM(bRANSAC, "RANSAC",true, "Apply RANSAC to estimate the similarity transformation parameters, Def=false")
    );

   std::vector<std::string> aVIm1;
   GetImgListVec(aFullPattern1, aVIm1);

   for(int i=0; i<int(aVIm1.size()); i++)
   {
       aImg1 = aVIm1[i];
       ExtractSIFT(aImg1, aCAS3D.mDir, dScale);
   }

   return EXIT_SUCCESS;
}

/*
bool ReadD2NetPt(std::string aDir, std::string aImg)
{
    std::string aFileName = aDir + "/" + aImg + ".d2-net";
    if (ELISE_fp::exist_file(aFileName) == false)
    {
        cout<<aFileName<<" didn't exist, hence skipped.\n";
        return false;
    }
    cnpy::npz_t my_npz = cnpy::npz_load(aFileName);

    cnpy::NpyArray arr_keypt0_raw = my_npz["keypoints"];
    //float* arr_keypt0 = arr_keypt0_raw.data<float>();
    int nPtNum1 = arr_keypt0_raw.shape[0];
    cout<<nPtNum1<<endl;

    cnpy::NpyArray arr_desc_raw = my_npz["descriptors"];
    //float* arr_desc_raw = arr_desc_raw.data<float>();
    nPtNum1 = arr_desc_raw.shape[0];
    cout<<nPtNum1<<endl;

    if(nPtNum1<=0)
    {
        cout<<aFileName<<" has no key point, hence skipped.\n";
        return false;
    }
    return true;
}

int D2NetMatch_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aImg1;
   std::string aImg2;

   ElInitArgMain
    (
        argc,argv,
               LArgMain()   << EAMC(aImg1,"Master image name")
                      << EAMC(aImg2,"Secondary image name"),
        LArgMain()
                    << aCAS3D.ArgBasic()
    );

   ReadD2NetPt(aCAS3D.mDir, aImg1);
   cout<<aImg1<<aImg2<<endl;

   return EXIT_SUCCESS;
}
*/


int ReadTiePt(std::string aDir, std::string aFile, std::vector<Pt2dr>& aPtL, std::vector<Pt2dr>& aPtR)
{
    int nPtNum = 0;
    ifstream in(aDir +"/"+ aFile);
    if (!in) {
        cout << "error opening file " << aDir +"/"+ aFile << endl;
        return nPtNum;
    }

    std::string s;
    while(getline(in,s))
    {
        double xL, yL, xR, yR;
        std::stringstream is(s);
        is>>xL>>yL>>xR>>yR;
        aPtL.push_back(Pt2dr(xL, yL));
        aPtR.push_back(Pt2dr(xR, yR));
        nPtNum++;
    }
    return nPtNum;
}

bool SaveSimi(std::string aDir, std::string aFile, ElSimilitude aSimCur)
{
    FILE * fpOutput = fopen((aDir+"/"+aFile).c_str(), "w");

    if (NULL == fpOutput)
    {
        cout<<"Open file "<<aDir+"/"+aFile<<" failed"<<endl;
        return false;
    }

    Pt2dr tr, sc;
    tr = aSimCur.tr();
    sc = aSimCur.sc();

    cout<<"Translation_X, Translation_Y, scale, rotation:"<<endl;
    printf("%lf  %lf  %lf  %lf\n", tr.x, tr.y, sc.x, sc.y);


    fprintf(fpOutput, "Translation_X, Translation_Y, scale, rotation:\n");
    fprintf(fpOutput, "%lf  %lf  %lf  %lf\n", tr.x, tr.y, sc.x, sc.y);

    return true;
}

bool Calc2DSimi(std::string aDir, std::string aTiePtFile, std::string aSimiFile)
{
    std::vector<Pt2dr> aPtL;
    std::vector<Pt2dr> aPtR;

    int nPtNum = ReadTiePt(aDir, aTiePtFile, aPtL, aPtR);

    ElPackHomologue aPack;
    for(int i=0; i<nPtNum; i++)
    {
        aPack.Cple_Add(ElCplePtsHomologues(aPtL[i],aPtR[i]));
    }
    double aPropRan = 0.8;
    ElSimilitude aSimCur = SimilRobustInit(aPack,aPropRan,1);

    SaveSimi(aDir, aSimiFile, aSimCur);
    return true;
}

int Calc2DSimi_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aTiePtFile;
   std::string aSimiFile = "Simi.txt";
   //bool bRANSAC = false;

   ElInitArgMain
    (
        argc,argv,
               LArgMain() << EAMC(aTiePtFile,"Input tie point file"),
        LArgMain()
               << aCAS3D.ArgBasic()
               << EAM(aSimiFile, "Out",true, "Name of output file that recorded the 2D similarity transformation paramters, Def=Simi.txt")
               //<< EAM(bRANSAC, "RANSAC",true, "Apply RANSAC to estimate the similarity transformation parameters, Def=false")
    );

   Calc2DSimi(aCAS3D.mDir, aTiePtFile, aSimiFile);

   return EXIT_SUCCESS;
}
