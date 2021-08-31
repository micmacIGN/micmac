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

/*******************************************/
/******* cCommonAppliTiepHistorical  *******/
/*******************************************/
cCommonAppliTiepHistorical::cCommonAppliTiepHistorical() :
        mArgBasic(new LArgMain),
        mArgRough(new LArgMain),
        mArgSuperGlue(new LArgMain),
        mArgMergeTiePt(new LArgMain),
        mArgGetPatchPair(new LArgMain),
        mArgGetOverlappedImages(new LArgMain),
        mArg2DRANSAC(new LArgMain),
        mArg3DRANSAC(new LArgMain),
        mArgGuidedSIFT(new LArgMain),
        mArgDSM_Equalization(new LArgMain),
        mArgCrossCorrelation(new LArgMain),
        mArgCreateGCPs(new LArgMain)
{
    mDir = "./";
    mAngle = 0;
    mBufferSz = Pt2dr(30, 60);
    mCheckAngle = true;
    mCheckScale = true;
    mHomoXml = "SubPatch.xml";
    mImgPair = "SuperGlueInput.txt";
    mInput_dir = "./";
    mR2DIteration = 1000;
    mR3DIteration = 1000;
    mKeepNpzFile = false;
    mMax_keypoints = 1024;
    mModel = "outdoor";
    mMutualNN = true;
    //mOutImg = "";
    mOutPairXml = "OverlappedImages.xml";
    mOutput_dir = "./";
    mSpGlueOutSH = "-SuperGlue";
    mGuidedSIFTOutSH = "-GuidedSIFT";
    mMergeTiePtOutSH = "";
    mRANSACOutSH = "";
    mMergeTiePtInSH = "";
    mRANSACInSH = "";
    mCreateGCPsInSH = "";
    mCrossCorrelationOutSH = "";
    mPatchSz = Pt2dr(640, 480);
    mPredict = true;
    mRatioT = true;
    mResize = Pt2di(640, 480);
    mRootSift = true;
    mScale = 1;
    mSearchSpace = 100;
    mSkipSIFT = false;
    mSTDRange = 5;
    mSubPatchXml = "SubPatch.xml";
    mR2DThreshold = 10;
    mR3DThreshold = -1;
    mCrossCorrThreshold = 0.6;
    mWindowSize = 32;
    mViz = false;
//    mDSMFileL = "MMLastNuage.xml";
//    mDSMFileR = "MMLastNuage.xml";
//    mDSMDirL = "";
//    mDSMDirR = "";
    mOutDir = "./Tmp_Patches";
//    mOutImg1 = "";
//    mOutImg2 = "";
    mOut2DXml1 = "OutGCP2D_epoch1.xml";
    mOut2DXml2 = "OutGCP2D_epoch2.xml";
    mOut3DXml1 = "OutGCP3D_epoch1.xml";
    mOut3DXml2 = "OutGCP3D_epoch2.xml";
    //mSubPatchLXml = "SubPL.xml";
    //mSubPatchRXml = "SubPR.xml";
    mStrEntSpG = "";
    mStrOpt = "";

        *mArgBasic
            //                        << EAM(mExe,"Exe",true,"Execute all, Def=true")
                        << EAM(mDir,"Dir",true,"Work directory, Def=./");


        *mArgRough
            << EAM(mOriIn1,"OriIn1",true,"Input Orientation of epoch1, Def=./")
            << EAM(mOriIn2,"OriIn2",true,"Input Orientation of epoch2, Def=./")
//            << EAM(mDSMDirL,"DSMDirL",true,"DSM direcotry of epoch1, Def=none")
//            << EAM(mDSMDirR,"DSMDirR",true,"DSM direcotry of epoch2, Def=none")
            << EAM(mOriOut,"OriOut",true,"Output Orientation, Def=./");



    *mArgDSM_Equalization
        //<< EAM(mDSMFileL, "DSMFile", true, "DSM File, Def=MMLastNuage.xml")
        << EAM(mSTDRange, "STDRange", true, "Only pixels with their value within STDRange times of std will be considered (in order to ignore altitude outliers), Def=5");
        //<< EAM(mOutImg, "OutImg", true, "Output image name");


    *mArgGetPatchPair
            << EAM(mPatchSz, "PatchSz", true, "Patch size, Def=[640, 480]")
            << EAM(mBufferSz, "BufferSz", true, "Buffer zone size around the patch, Def=[30, 60]")
            << EAM(mSubPatchXml, "SubPXml", true, "The output xml file name to record the homography between the patch and original image, Def=SubPatch.xml")
            //<< EAM(mDSMDirL, "DSMDirL", true, "DSM of first image (for improving the reprojecting accuracy), Def=none")
            //<< EAM(mDSMDirR, "DSMDirR", true, "DSM of second image (for improving the reprojecting accuracy), Def=none")
            << EAM(mOutDir, "OutDir", true, "Output direcotry of the patches, Def=./Tmp_Patches")
            //<< EAM(mOutImg1, "OutImg1", true, "Name of the main part of the output patches from first image, Def=first image name")
            //<< EAM(mOutImg2, "OutImg2", true, "Name of the main part of the output patches from second image, Def=second image name")
            << EAM(mImgPair, "ImgPair", true, "Output txt file that records the patch pairs, Def=SuperGlueInput.txt");


    *mArgSuperGlue
            //<< EAM(input_pairs, "input_pairs", true, "txt file that listed the image pairs")
            << EAM(mInput_dir, "InDir", true, "The input directory of the images for SuperGlue, Def=./")
            << EAM(mSpGlueOutSH, "SpGOutSH", true, "Homologue extenion for NB/NT mode of SuperGlue, Def=-SuperGlue")
            << EAM(mOutput_dir, "OutDir", true, "The output directory of the match results of SuperGlue, Def=./")
            << EAM(mResize, "Resize", true, "The goal size to resize the input image for SuperGlue, Def=[640, 480], if you don't want to resize, please set to [-1, -1]")
            << EAM(mViz, "Viz", true, "Visualize the matches and dump the plots of SuperGlue, Def=false")
            << EAM(mModel, "Model", true, "Pretrained indoor or outdoor model of SuperGlue, Def=outdoor")
            << EAM(mMax_keypoints, "MaxPt", true, "Maximum number of keypoints detected by Superpoint, Def=1024")
            << EAM(mKeepNpzFile, "KeepNpzFile", true, "Keep the original npz file that SuperGlue outputed, Def=false")
            << EAM(mStrEntSpG, "EntSpG", true, "The SuperGlue program entry, Def=../micmac/src/uti_phgrm/TiePHistorical/SuperGluePretrainedNetwork-master/match_pairs.py")
            << EAM(mStrOpt, "opt", true, "Other options for SuperGlue (for debug only), Def=none");

    *mArgMergeTiePt
        << EAM(mMergeTiePtInSH,"MergeInSH",true,"Input Homologue extenion for NB/NT mode for MergeTiePt, Def=none")
        << EAM(mHomoXml,"HomoXml",true,"Input xml file that recorded the homograhpy from patch to original image for MergeTiePt, Def=SubPatch.xml")
           //the name of MergeOutSH will be set as the same name of HomoXml, if not set by users
        << EAM(mMergeTiePtOutSH,"MergeOutSH",true,"Output Homologue extenion for NB/NT mode of MergeTiePt, Def=-SubPatch");


    *mArgCreateGCPs
        //<< EAM(mDSMDirL,"DSMDirL",true,"DSM direcotry of epoch1, Def=none")
        //<< EAM(mDSMDirR,"DSMDirR",true,"DSM direcotry of epoch2, Def=none")
        //<< EAM(mDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
        //<< EAM(mDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")
        << EAM(mCreateGCPsInSH,"CreateGCPsInSH",true,"Input Homologue extenion for NB/NT mode for CreateGCPs, Def=none")
        << EAM(mOut2DXml1,"Out2DXml1",true,"Output xml files of 2D obersevations of the GCPs in epoch1, Def=OutGCP2D_epoch1.xml")
        << EAM(mOut3DXml1,"Out3DXml1",true,"Output xml files of 3D obersevations of the GCPs in epoch1, Def=OutGCP3D_epoch1.xml")
        << EAM(mOut2DXml2,"Out2DXml2",true,"Output xml files of 2D obersevations of the GCPs in epoch2, Def=OutGCP2D_epoch2.xml")
        << EAM(mOut3DXml2,"Out3DXml2",true,"Output xml files of 3D obersevations of the GCPs in epoch2, Def=OutGCP3D_epoch2.xml");


    *mArgGetOverlappedImages
            << EAM(mOutPairXml,"OutPairXml",true,"Output Xml file to record the overlapped image pairs, Def=OverlappedImages.xml");


    *mArg2DRANSAC
        << EAM(mRANSACInSH,"2DRANInSH",true,"Input Homologue extenion for NB/NT mode for 2D RANSAC, Def=none")
        << EAM(mRANSACOutSH,"2DRANOutSH",true,"Output Homologue extenion for NB/NT mode of 2D RANSAC, Def='RANSACInSH'-2DRANSAC")
        << EAM(mR2DIteration,"2DIter",true,"2D RANSAC iteration, Def=1000")
        << EAM(mR2DThreshold,"2DRANTh",true,"2D RANSAC threshold, Def=10");

    *mArg3DRANSAC
        << EAM(mRANSACInSH,"3DRANInSH",true,"Input Homologue extenion for NB/NT mode for 3D RANSAC, Def=none")
        << EAM(mRANSACOutSH,"3DRANOutSH",true,"Output Homologue extenion for NB/NT mode of 3D RANSAC, Def='RANSACInSH'-3DRANSAC")
        << EAM(mR3DIteration,"3DIter",true,"3D RANSAC iteration, Def=1000")
        << EAM(mR3DThreshold,"3DRANTh",true,"3D RANSAC threshold, Def=10*(GSD of second image)");
           /*
        << EAM(mDSMDirL, "DSMDirL", true, "DSM directory of first image, Def=none")
        << EAM(mDSMDirR, "DSMDirR", true, "DSM directory of second image, Def=none")
        << EAM(mDSMFileL, "DSMFileL", true, "DSM File of first image, Def=MMLastNuage.xml")
        << EAM(mDSMFileR, "DSMFileR", true, "DSM File of second image, Def=MMLastNuage.xml");
               */


    *mArgGuidedSIFT
            /*
    << EAM(mDSMDirL, "DSMDirL", true, "DSM of first image (for improving the reprojecting accuracy), Def=none")
    << EAM(mDSMDirR, "DSMDirR", true, "DSM of second image (for improving the reprojecting accuracy), Def=none")
    << EAM(mDSMFileL, "DSMFileL", true, "DSM File of first image, Def=MMLastNuage.xml")
    << EAM(mDSMFileR, "DSMFileR", true, "DSM File of second image, Def=MMLastNuage.xml")
            */
    << EAM(mGuidedSIFTOutSH,"GSIFTOutSH",true,"Output Homologue extenion for NB/NT mode of Guided SIFT, Def=-GuidedSIFT")
    << EAM(mSkipSIFT,"SkipSIFT",true,"Skip extracting SIFT key points in case it is already done, Def=false")
    << EAM(mSearchSpace,"SearchSpace",true,"Radius of the search space for GuidedSIFT (the search space is the circle with the center on the predicted point), Def=100")
    << EAM(mMutualNN, "MutualNN",true, "Apply mutual nearest neighbor or not on GuidedSIFT, Def=true")
    << EAM(mRatioT, "RatioT",true, "Apply ratio test or not on GuidedSIFT, Def=true")
    << EAM(mRootSift, "RootSift",true, "Use RootSIFT as descriptor or not on GuidedSIFT, Def=true")
    << EAM(mCheckScale, "CheckScale",true, "Check the scale of the candidate tie points or not on GuidedSIFT, Def=true")
    << EAM(mCheckAngle, "CheckAngle",true, "Check the angle of the candidate tie points or not on GuidedSIFT, Def=true")
    << EAM(mPredict, "Predict",true, "Use the predicted key points to guide the matching or not, Def=true")
    << EAM(mScale, "Scale",true, "The scale used for checking the candidate tie points on GuidedSIFT, Def=1")
    << EAM(mAngle, "Angle",true, "The angle used for checking the candidate tie points on GuidedSIFT, Def=0");

    *mArgCrossCorrelation
//            << EAM(mPatchSz, "PatchSz", true, "Patch size, Def=[640, 480]")
//            << EAM(mBufferSz, "BufferSz", true, "Buffer sie, Def=[30, 60]")
            << EAM(mCrossCorrelationInSH,"CCInSH",true,"Input Homologue extenion for NB/NT mode for cross correlation, Def=none")
            << EAM(mCrossCorrelationOutSH,"CCOutSH",true,"Output Homologue extenion for NB/NT mode of cross correlation, Def='CrossCorrelationInSH'-CrossCorrelation")
               //<< EAM(mSubPatchXml, "SubPatchXml", true, "The xml file name to record the homography between the patch and original image, Def=SubPatch.xml")
            << EAM(mWindowSize, "SzW",true, "Window size of cross correlation, Def=32")
            << EAM(mCrossCorrThreshold, "CCTh",true, "Corss correlation threshold, Def=0.6");

        //StdCorrecNameOrient(mOutRPC,mDir,true);

    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
}

LArgMain & cCommonAppliTiepHistorical::ArgBasic()
{
        return * mArgBasic;
}

LArgMain & cCommonAppliTiepHistorical::ArgRough()
{
        return * mArgRough;
}

LArgMain & cCommonAppliTiepHistorical::ArgDSM_Equalization()
{
        return * mArgDSM_Equalization;
}

LArgMain & cCommonAppliTiepHistorical::ArgSuperGlue()
{
        return * mArgSuperGlue;
}

LArgMain & cCommonAppliTiepHistorical::ArgMergeTiePt()
{
        return * mArgMergeTiePt;
}

LArgMain & cCommonAppliTiepHistorical::ArgGetPatchPair()
{
        return * mArgGetPatchPair;
}

LArgMain & cCommonAppliTiepHistorical::ArgGetOverlappedImages()
{
        return * mArgGetOverlappedImages;
}

LArgMain & cCommonAppliTiepHistorical::Arg2DRANSAC()
{
        return * mArg2DRANSAC;
}

LArgMain & cCommonAppliTiepHistorical::Arg3DRANSAC()
{
        return * mArg3DRANSAC;
}

LArgMain & cCommonAppliTiepHistorical::ArgGuidedSIFT()
{
        return * mArgGuidedSIFT;
}

LArgMain & cCommonAppliTiepHistorical::ArgCrossCorrelation()
{
        return * mArgCrossCorrelation;
}

LArgMain & cCommonAppliTiepHistorical::ArgCreateGCPs()
{
        return * mArgCreateGCPs;
}

std::string cCommonAppliTiepHistorical::ComParamDSM_Equalization()
{
    std::string aCom = "";
    if (EAMIsInit(&mDir))    aCom += " Dir=" + mDir;
    if (EAMIsInit(&mSTDRange))  aCom +=  " STDRange=" + ToString(mSTDRange);

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamGetPatchPair()
{
    std::string aCom = "";
    if (EAMIsInit(&mDir))  aCom +=  " Dir=" + mDir;
    if (EAMIsInit(&mPatchSz))    aCom += " PatchSz=[" + ToString(mPatchSz.x) + "," + ToString(mPatchSz.y) + "]";
    if (EAMIsInit(&mBufferSz))    aCom += " BufferSz=[" + ToString(mBufferSz.x) + "," + ToString(mBufferSz.y) + "]";
    if (EAMIsInit(&mSubPatchXml))  aCom +=  " SubPXml=" + mSubPatchXml;
    if (EAMIsInit(&mOutDir))  aCom +=  " OutDir=" + mOutDir;
    if (EAMIsInit(&mImgPair))  aCom +=  " ImgPair=" + mImgPair;

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamSuperGlue()
{
    std::string aCom ="";
    if (EAMIsInit(&mInput_dir))   aCom +=  " InDir=" + mDir + "/" + mInput_dir;
    if (EAMIsInit(&mOutput_dir))   aCom +=  " OutDir=" + mDir + "/" + mOutput_dir;
    if (EAMIsInit(&mSpGlueOutSH))   aCom +=  " SpGOutSH=" + mSpGlueOutSH;
    if (EAMIsInit(&mResize))    aCom += " Resize=[" + ToString(mResize.x) + "," + ToString(mResize.y) + "]";
    if (EAMIsInit(&mViz))       aCom += " mViz=" + ToString(mViz);
    if (EAMIsInit(&mModel))  aCom +=  " Model=" + mModel;
    if (EAMIsInit(&mMax_keypoints))    aCom +=  " MaxPt=" + ToString(mMax_keypoints);
    if (EAMIsInit(&mKeepNpzFile))   aCom +=  " KeepNpzFile=" + ToString(mKeepNpzFile);
    if (EAMIsInit(&mStrEntSpG))  aCom +=  " EntSpG=" + mStrEntSpG;
    if (EAMIsInit(&mStrOpt))  aCom +=  " opt=\" " + mStrOpt + "\"";

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamMergeTiePt()
{
    std::string aCom = "";
    if (EAMIsInit(&mMergeTiePtInSH))   aCom +=  " MergeInSH=" + mMergeTiePtInSH;
    if (EAMIsInit(&mMergeTiePtOutSH))  aCom +=  " MergeOutSH=" + mMergeTiePtOutSH;
    if (EAMIsInit(&mHomoXml))          aCom +=  " HomoXml=" + mHomoXml;

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamRANSAC2D()
{
    std::string aCom = "";
    if (EAMIsInit(&mRANSACInSH))   aCom +=  " 2DRANInSH=" + mRANSACInSH;
    if (EAMIsInit(&mRANSACOutSH))  aCom +=  " 2DRANOutSH=" + mRANSACOutSH;
    if (EAMIsInit(&mR2DIteration))          aCom +=  " 2DIter=" + ToString(mR2DIteration);
    if (EAMIsInit(&mR2DThreshold))          aCom +=  " 2DRANTh=" + ToString(mR2DThreshold);

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamCreateGCPs()
{
    std::string aCom = "";
    if (EAMIsInit(&mCreateGCPsInSH))          aCom +=  " CreateGCPsInSH=" + mCreateGCPsInSH;
    if (EAMIsInit(&mOut2DXml1))          aCom +=  " Out2DXml1=" + mOut2DXml1;
    if (EAMIsInit(&mOut2DXml2))          aCom +=  " Out2DXml2=" + mOut2DXml2;
    if (EAMIsInit(&mOut3DXml1))          aCom +=  " Out3DXml1=" + mOut3DXml1;
    if (EAMIsInit(&mOut3DXml2))          aCom +=  " Out3DXml2=" + mOut3DXml2;

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamGuidedSIFTMatch()
{
    std::string aCom = "";
    if (EAMIsInit(&mGuidedSIFTOutSH   ))          aCom +=  " GSIFTOutSH=" + mGuidedSIFTOutSH;
    if (EAMIsInit(&mSkipSIFT   ))          aCom +=  " SkipSIFT=" + ToString(mSkipSIFT);
    if (EAMIsInit(&mSearchSpace   ))          aCom +=  " SearchSpace=" + ToString(mSearchSpace);
    if (EAMIsInit(&mMutualNN   ))          aCom +=  " MutualNN=" + ToString(mMutualNN);
    if (EAMIsInit(&mRatioT   ))          aCom +=  " RatioT=" + ToString(mRatioT);
    if (EAMIsInit(&mRootSift   ))          aCom +=  " RootSift=" + ToString(mRootSift);
    if (EAMIsInit(&mCheckScale   ))          aCom +=  " CheckScale=" + ToString(mCheckScale);
    if (EAMIsInit(&mCheckAngle   ))          aCom +=  " CheckAngle=" + ToString(mCheckAngle);
    if (EAMIsInit(&mPredict   ))          aCom +=  " Predict=" + ToString(mPredict);
    if (EAMIsInit(&mScale   ))          aCom +=  " Scale=" + ToString(mScale);
    if (EAMIsInit(&mAngle ))          aCom +=  " Angle=" + ToString(mAngle);

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamGetOverlappedImages()
{
    std::string aCom = "";
    if (EAMIsInit(&mOutPairXml))          aCom +=  " OutPairXml=" + mOutPairXml;

    return aCom;
}

std::string cCommonAppliTiepHistorical::GetFolderName(std::string strIn)
{
    std::string strOut = strIn;

    std::size_t found = strIn.find("/");
    if (found!=std::string::npos)
        strOut = strIn.substr(0, found);

    return strOut;
}

void cCommonAppliTiepHistorical::ExtractSIFT(std::string aFullName, std::string aDir)
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

/*******************************************/
/****** cAppliTiepHistoricalPipeline  ******/
/*******************************************/

std::string cAppliTiepHistoricalPipeline::StdCom(const std::string & aCom,const std::string & aPost, bool aExe)
{
    std::string  aFullCom = MMBinFile(MM3DStr) +  aCom + BLANK;
    aFullCom = aFullCom + aPost;
/*
    if(aExe == false)
    {
        std::cout << "COM= " << aFullCom << "\n";
        return aFullCom;
    }
*/
    if (aExe)
    {
       //std::cout << "---------->>>>  " << aFullCom << "\n";
       System(aFullCom);
    }
    else
       std::cout << aFullCom << "\n";

    return aFullCom;

    //std::cout << " DONE " << aCom << " in time " << mChrono.uval() << "\n";
}

std::string cAppliTiepHistoricalPipeline::GetImage_Profondeur(std::string aDSMDir, std::string aDSMFile)
{
    cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
    (
    aDSMDir + "/" + aDSMFile,
    StdGetFileXMLSpec("SuperposImage.xml"),
    "XML_ParamNuage3DMaille",
    "XML_ParamNuage3DMaille"
    );


    cImage_Profondeur aImDSM = aNuageIn.Image_Profondeur().Val();
    return aImDSM.Image();
}

int cAppliTiepHistoricalPipeline::GetTiePtNum(std::string aDir, std::string aImg1, std::string aImg2, std::string aSH)
{
    std::string aDir_inSH = aDir + "/Homol" + aSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";

    if (ELISE_fp::exist_file(aNameIn) == false)
    {
        cout<<aNameIn<<"didn't exist hence skipped."<<endl;
        return 0;
    }
    ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aNameIn);

    //    int nPtNum = int(aPackFull.end() - aPackFull.begin());
    int nPtNum = 0;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin(); itCpl!=aPackFull.end(); itCpl++)
    {
        nPtNum++;
    }

    return nPtNum;
}

int cAppliTiepHistoricalPipeline::GetOverlappedImgPair(std::string aName, std::vector<std::string>& aResL, std::vector<std::string>& aResR)
{
    //std::vector<tNamePair> aRes;
    int num = 0;
    cSauvegardeNamedRel aSNR = StdGetFromPCP(aName,SauvegardeNamedRel);
    for (const auto & aCpl : aSNR.Cple())
    {
        aResL.push_back(aCpl.N1());
        aResR.push_back(aCpl.N2());
        num++;
        //aRes.Add(tNamePair(aCpl.N1(),aCpl.N2()));
    }

    return num;
}

std::string cAppliTiepHistoricalPipeline::GetImgList(std::string aDir, std::string aFileName)
{
    std::string aRes = "\"";

    std::string s;
    ifstream in1(aDir+aFileName);
    while(getline(in1,s))
    {
        aRes += s+"|";
    }

    aRes = aRes.substr(0, aRes.length()-1) + "\"";

    return aRes;
}

void cAppliTiepHistoricalPipeline::DoAll()
{
    std::string aCom;

    std::string aBaseOutDir = mCAS3D.mOutDir;
    if(aBaseOutDir.find("/") == aBaseOutDir.length()-1)
        aBaseOutDir = aBaseOutDir.substr(0, aBaseOutDir.length()-1);
    std::string aOutDir = aBaseOutDir;

    std::string aOri1 = mOri1;
    StdCorrecNameOrient(aOri1,"./",true);
    std::string aOri2 = mOri2;
    StdCorrecNameOrient(aOri2,"./",true);

    if(mSkipCoReg == false)
    {
        printf("**************************************1- rough co-registration************************************\n");
        /********************1- rough co-registration******************/

        aOutDir = aBaseOutDir + "-CoReg";

        std::string aDSMImgNameL = GetImage_Profondeur(mCAS3D.mDir+mDSMDirL, mDSMFileL);
        std::string aDSMImgNameR = GetImage_Profondeur(mCAS3D.mDir+mDSMDirR, mDSMFileR);
        std::string aDSMImgGrayNameL = StdPrefix(aDSMImgNameL)+"_gray."+StdPostfix(aDSMImgNameL);
        std::string aDSMImgGrayNameR = StdPrefix(aDSMImgNameR)+"_gray."+StdPostfix(aDSMImgNameR);

        /**************************************/
        /* 1.1 - DSM_Equalization and wallis filter */
        /**************************************/
        StdCom("TestLib DSM_Equalization", mDSMDirL + BLANK + "DSMFile="+mDSMFileL + BLANK
                                                                    + mCAS3D.ComParamDSM_Equalization(), mExe);
        StdCom("TestLib DSM_Equalization", mDSMDirR + BLANK + "DSMFile="+mDSMFileR + BLANK
                                                                    + mCAS3D.ComParamDSM_Equalization(), mExe);

        StdCom("TestLib Wallis", aDSMImgGrayNameL + BLANK + "Dir="+mDSMDirL, mExe);
        StdCom("TestLib Wallis", aDSMImgGrayNameR + BLANK + "Dir="+mDSMDirR, mExe);

        /**************************************/
        /* 1.2 - GetPatchPair for rough co-registration */
        /**************************************/
        std::string aDSMImgWallisNameL = aDSMImgGrayNameL+"_sfs.tif";
        std::string aDSMImgWallisNameR = aDSMImgGrayNameR+"_sfs.tif";
        aCom = "";
        if (!EAMIsInit(&mCAS3D.mOutDir))   aCom +=  " OutDir=" + aOutDir;
        StdCom("TestLib GetPatchPair BruteForce", mDSMDirL+"/"+aDSMImgWallisNameL + BLANK + mDSMDirR+"/"+aDSMImgWallisNameR + BLANK + aCom + BLANK + "Rotate=1" + BLANK + mCAS3D.ComParamGetPatchPair(), mExe);

        std::string aDSMImgGrayNameRenamedL = mCAS3D.GetFolderName(mDSMDirL) + "." + StdPostfix(aDSMImgNameL);
        std::string aDSMImgGrayNameRenamedR = mCAS3D.GetFolderName(mDSMDirR) + "." + StdPostfix(aDSMImgNameR);

        std::string aRotate[4] = {"", "_R90", "_R180", "_R270"};
        std::string aFinalOutSH;
        int nMaxinlier = 0;
        //Rotate the left DSM 4 times and apply superGlue
        for(int i=0; i<4; i++)
        {
            if(mRotateDSM != -1)
            {
                std::string aRotateDSMStr = "_R" + ToString(mRotateDSM);
                if(mRotateDSM == 0)
                    aRotateDSMStr = "";
                if(aRotate[i] != aRotateDSMStr)
                {
                    printf("%dth attempt with \"%s\" doesn't match with \"%s\", hence skipped\n", i, aRotate[i].c_str(), aRotateDSMStr.c_str());
                    continue;
                }
            }
            /**************************************/
            /* 1.3 - SuperGlue for rough co-registration */
            /**************************************/
            std::string aImgPair = StdPrefix(mCAS3D.mImgPair) + aRotate[i] + "." + StdPostfix(mCAS3D.mImgPair);
            aCom = "";
            if (!EAMIsInit(&mCAS3D.mInput_dir))    aCom +=  " InDir=" + aOutDir+"/";
            if (!EAMIsInit(&mCAS3D.mOutput_dir))   aCom +=  " OutDir=" + aOutDir+"/";
            StdCom("TestLib SuperGlue", aImgPair + BLANK + aCom + BLANK + mCAS3D.ComParamSuperGlue(), mExe);


            /**************************************/
            /* 1.4 - MergeTiePt for rough co-registration */
            /**************************************/
            std::string aHomoXml = StdPrefix(mCAS3D.mHomoXml) + aRotate[i] + "." + StdPostfix(mCAS3D.mHomoXml);
            aCom = "";
            if (!EAMIsInit(&mCAS3D.mHomoXml))   aCom +=  " HomoXml=" + aHomoXml;
            if (!EAMIsInit(&mCAS3D.mMergeTiePtInSH))   aCom +=  " MergeInSH=" + mCAS3D.mSpGlueOutSH;
            StdCom("TestLib MergeTiePt", aOutDir+"/" + BLANK + aCom + BLANK + mCAS3D.ComParamMergeTiePt(), mExe);


            /**************************************/
            /* 1.5 - RANSAC R2D for rough co-registration */
            /**************************************/
            aCom = "";
            if (!EAMIsInit(&mCAS3D.mRANSACInSH))   aCom +=  " 2DRANInSH=-" + StdPrefix(aHomoXml);
            std::string aRANSACOutSH = "-" + StdPrefix(aHomoXml) + "-2DRANSAC";
            StdCom("TestLib RANSAC R2D", aDSMImgGrayNameRenamedL + BLANK + aDSMImgGrayNameRenamedR + BLANK + "Dir=" + aOutDir+"/" + BLANK + aCom + BLANK + mCAS3D.ComParamRANSAC2D(), mExe);
            int nInlier = GetTiePtNum(aOutDir, aDSMImgGrayNameRenamedL, aDSMImgGrayNameRenamedR, aRANSACOutSH);
            cout<<i<<",,"<<aRANSACOutSH<<","<<nInlier<<endl;

            if(nInlier > nMaxinlier)
            {
                nMaxinlier = nInlier;
                aFinalOutSH = aRANSACOutSH;
            }
        }
        cout<<"aFinalOutSH: "<<aFinalOutSH<<endl;


        /**************************************/
        /* 1.6 - CreateGCPs for rough co-registration */
        /**************************************/
        aCom = "";
        if (!EAMIsInit(&mCAS3D.mCreateGCPsInSH))   aCom +=  " CreateGCPsInSH=" + aFinalOutSH;
        StdCom("TestLib CreateGCPs", aOutDir + BLANK + aDSMImgGrayNameRenamedL + BLANK + aDSMImgGrayNameRenamedR + BLANK + mCAS3D.mDir + BLANK + mImgList1 + BLANK + mImgList2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + mDSMDirL + BLANK + mDSMDirR + aCom + mCAS3D.ComParamCreateGCPs(), mExe);


        /**************************************/
        /* 1.7 - GCPBascule for rough co-registration */
        /**************************************/
        aCom = "";
        std::string aImgListL = GetImgList(mCAS3D.mDir, mImgList1);
        StdCom("GCPBascule", aImgListL + BLANK + mOri1 + BLANK + mCoRegOri.substr(4,mCoRegOri.length()) + BLANK + mCAS3D.mOut3DXml2 + BLANK + mCAS3D.mOut2DXml1, mExe);
        /*
        aCom = "/home/lulin/Documents/ThirdParty/oldMicmac/micmac_old/bin/mm3d GCPBascule " + aImgListL + BLANK + mOri1 + BLANK + mCoRegOri.substr(4,mCoRegOri.length()) + BLANK + mCAS3D.mOut3DXml2 + BLANK + mCAS3D.mOut2DXml1;
        if(mExe==true)
            System(aCom);
        cout<<aCom<<endl;
        */
    }


    if(mSkipPrecise == false)
    {
        printf("**************************************2- precise matching************************************\n");
    /********************2- precise matching******************/
    aOutDir = aBaseOutDir + "-Precise";

    /**************************************/
    /* 2.1 - GetOverlappedImages */
    /**************************************/
    StdCom("TestLib GetOverlappedImages", mOri1 + BLANK + mOri2 + BLANK + mImg4MatchList1 + BLANK + mImg4MatchList2 + BLANK + mCAS3D.ComParamGetOverlappedImages() + BLANK + "Para3DH=Basc-"+aOri1+"-2-"+aOri2+".xml", mExe);

    if (ELISE_fp::exist_file(mCAS3D.mOutPairXml) == false)
    {
        cout<<mCAS3D.mOutPairXml<<" didn't exist because the pipeline is not executed, hence the precise matching commands are not shown here."<<endl;
        return;
    }

    bool aExe = false;
    std::vector<std::string> aOverlappedImgL;
    std::vector<std::string> aOverlappedImgR;
    int nPairNum = GetOverlappedImgPair(mCAS3D.mOutPairXml, aOverlappedImgL, aOverlappedImgR);

    std::list<std::string> aComList;
    std::string aComSingle;
    /**************************************/
    /* 2.2 - GetPatchPair for precise matching */
    /**************************************/
    cout<<"-------GetPatchPair-------"<<endl;
    //if(mSkipGetPatchPair == false)
    {
    for(int i=0; i<nPairNum; i++)
    {
        std::string aImg1 = aOverlappedImgL[i];
        std::string aImg2 = aOverlappedImgR[i];
/*
        cout<<"---------------------"<<i<<"th pair------------------"<<endl;
        cout<<aImg1<<" "<<aImg2<<endl;
*/
        std::string aPrefix = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_" ;
        aCom = "";
        if (!EAMIsInit(&mCAS3D.mOutDir))   aCom +=  " OutDir=" + aOutDir;
        if (!EAMIsInit(&mCAS3D.mSubPatchXml))  aCom +=  " SubPXml=" + aPrefix + mCAS3D.mSubPatchXml;
        if (!EAMIsInit(&mCAS3D.mImgPair))  aCom +=  " ImgPair=" + aPrefix + mCAS3D.mImgPair;
        //aComSingle = StdCom("TestLib GetPatchPair Guided", aImg1 + BLANK + aImg2 + BLANK + mCoRegOri + BLANK + mCoRegOri + BLANK + aCom + BLANK + mCAS3D.ComParamGetPatchPair(), aExe);
        //printf("%s\t%s\n", aOri1.c_str(), mOri1.c_str());
        aComSingle = StdCom("TestLib GetPatchPair Guided", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGetPatchPair() + BLANK + "Para3DH=Basc-"+aOri1+"-2-"+aOri2+".xml" + BLANK + "DSMDirL="+mDSMDirL, aExe);
        aComList.push_back(aComSingle);

        if(mUseDepth == true)
        {
            aComSingle = StdCom("TestLib GetPatchPair Guided", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGetPatchPair() + BLANK + "Para3DH=Basc-"+aOri1+"-2-"+aOri2+".xml" + BLANK + "DSMDirL="+mDSMDirL + BLANK + "Prefix=Depth_", aExe);
            aComList.push_back(aComSingle);
        }
    }
    /*
    for(list<std::string>::iterator it=aComList.begin();it!=aComList.end();it++)
    {
        cout<<(*it)<<endl;
    }
    */
    if(mExe && (!mSkipGetPatchPair))
        //cEl_GPAO::DoComInParal(aComList);
        //because "mm3d TestLib GetPatchPair Guided" is parallized itself, if DoComInParal here, terminal will show "make[1]: warning: -jN forced in submake: disabling jobserver mode."
        cEl_GPAO::DoComInSerie(aComList);
    }

    std::string aFeatureOutSH;

    //if(mSkipTentativeMatch == false)
    {
    /**************************************/
    /* 2.3: option 1 - SuperGlue for precise matching */
    /**************************************/
    aComList.clear();
    if(mFeature == "SuperGlue")
    {
        cout<<"-------SuperGlue-------"<<endl;
        for(int i=0; i<nPairNum; i++)
        {
            std::string aImg1 = aOverlappedImgL[i];
            std::string aImg2 = aOverlappedImgR[i];

            std::string aPrefix = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_" ;

            /**************************************/
            /* SuperGlue */
            /**************************************/
            std::string aImgPair = aPrefix + mCAS3D.mImgPair;
            aCom = "";
            if (!EAMIsInit(&mCAS3D.mInput_dir))    aCom +=  " InDir=" + aOutDir+"/";
            if (!EAMIsInit(&mCAS3D.mOutput_dir))   aCom +=  " OutDir=" + aOutDir+"/";
            aCom +=  "  CheckFile=" + ToString(mCheckFile);
            aComSingle = StdCom("TestLib SuperGlue", aImgPair + BLANK + aCom + BLANK + mCAS3D.ComParamSuperGlue(), aExe);
            aComList.push_back(aComSingle);
        }
        /*
        for(list<std::string>::iterator it=aComList.begin();it!=aComList.end();it++)
        {
            cout<<(*it)<<endl;
        }
        */
        if(mExe && (!mSkipTentativeMatch))
        {
            //cEl_GPAO::DoComInParal(aComList);
            cEl_GPAO::DoComInSerie(aComList);
        }
        aComList.clear();
        cout<<"-------MergeTiePt-------"<<endl;
        for(int i=0; i<nPairNum; i++)
        {
            std::string aImg1 = aOverlappedImgL[i];
            std::string aImg2 = aOverlappedImgR[i];

            std::string aPrefix = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_" ;

            /**************************************/
            /* MergeTiePt  */
            /**************************************/
            std::string aHomoXml = aPrefix + mCAS3D.mHomoXml;
            aCom = "";
            if (!EAMIsInit(&mCAS3D.mHomoXml))   aCom +=  " HomoXml=" + aHomoXml;
            if (!EAMIsInit(&mCAS3D.mMergeTiePtInSH))   aCom +=  " MergeInSH=" + mCAS3D.mSpGlueOutSH;
            if (!EAMIsInit(&mCAS3D.mMergeTiePtOutSH))
            {
                aCom +=  " MergeOutSH="+mCAS3D.mSpGlueOutSH;
                aFeatureOutSH = mCAS3D.mSpGlueOutSH;
            }
            else
                aFeatureOutSH = mCAS3D.mMergeTiePtOutSH;
            aComSingle = StdCom("TestLib MergeTiePt", aOutDir+"/" + BLANK + aCom + BLANK + "OutDir=" + mCAS3D.mDir + BLANK + mCAS3D.ComParamMergeTiePt(), aExe);
            aComList.push_back(aComSingle);
        }
        /*
        for(list<std::string>::iterator it=aComList.begin();it!=aComList.end();it++)
        {
            cout<<(*it)<<endl;
        }
        */
        if(mExe && (!mSkipTentativeMatch))
            cEl_GPAO::DoComInParal(aComList);
    }
    /**************************************/
    /* 2.3: option 2 - Guided SIFT for precise matching */
    /**************************************/
    else if(mFeature == "SIFT")
    {
        cout<<"-------Guided SIFT-------"<<endl;
        // Extract SIFT if SkipSIFT is set to false
        if(mExe == true && (!mSkipTentativeMatch) && mCAS3D.mSkipSIFT == false)
        {
            std::string aImgName;
            ifstream in1(mCAS3D.mDir+mImg4MatchList1);
            while(getline(in1,aImgName))
            {
                ExtractSIFT(aImgName, mCAS3D.mDir);
            }

            ifstream in2(mCAS3D.mDir+mImg4MatchList2);
            while(getline(in2,aImgName))
            {
                ExtractSIFT(aImgName, mCAS3D.mDir);
            }
        }

        for(int i=0; i<nPairNum; i++)
        {
            std::string aImg1 = aOverlappedImgL[i];
            std::string aImg2 = aOverlappedImgR[i];

            aCom = "";
            aCom +=  " SkipSIFT=true";
            aCom +=  " DSMDirL=" + mDSMDirL;
            aCom +=  " DSMDirR=" + mDSMDirR;
            if (EAMIsInit(&mDSMFileL))   aCom +=  " DSMFileL=" + mDSMFileL;
            if (EAMIsInit(&mDSMFileR))   aCom +=  " DSMFileR=" + mDSMFileR;
            aComSingle = StdCom("TestLib GuidedSIFTMatch", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGuidedSIFTMatch() + BLANK + "Para3DHL=Basc-"+aOri1+"-2-"+aOri2+".xml" + BLANK + "Para3DHR=Basc-"+aOri2+"-2-"+aOri1+".xml", aExe);

            aFeatureOutSH = mCAS3D.mGuidedSIFTOutSH;
            aComList.push_back(aComSingle);
        }
        /*
        for(list<std::string>::iterator it=aComList.begin();it!=aComList.end();it++)
        {
            cout<<(*it)<<endl;
        }
        */
        if(mExe && (!mSkipTentativeMatch))
            cEl_GPAO::DoComInParal(aComList);
    }
    else
    {
        cout<<"Please set Feature to SuperGlue or SIFT"<<endl;
        return;
    }
    cout<<"aFeatureOutSH: "<<aFeatureOutSH<<endl;
    }

    /**************************************/
    /* 2.4 - RANSAC R3D for precise matching */
    /**************************************/
    //if(mSkipRANSAC3D == false)
    {
        cout<<"-------RANSAC R3D-------"<<endl;
    aComList.clear();
    for(int i=0; i<nPairNum; i++)
    {
        std::string aImg1 = aOverlappedImgL[i];
        std::string aImg2 = aOverlappedImgR[i];

        aCom = "";
        aCom +=  " DSMDirL=" + mDSMDirL;
        aCom +=  " DSMDirR=" + mDSMDirR;
        if (!EAMIsInit(&mDSMFileL))   aCom +=  " DSMFileL=" + mDSMFileL;
        if (!EAMIsInit(&mDSMFileR))   aCom +=  " DSMFileR=" + mDSMFileR;
        if (!EAMIsInit(&mCAS3D.mRANSACInSH))   aCom +=  " 3DRANInSH=" + aFeatureOutSH;
        if (!EAMIsInit(&mCAS3D.mRANSACOutSH))   aCom +=  " 3DRANOutSH=" + aFeatureOutSH+"-3DRANSAC";
        if (EAMIsInit(&mCAS3D.mR3DIteration))   aCom +=  " 3DIter=" + ToString(mCAS3D.mR3DIteration);
        if (EAMIsInit(&mCAS3D.mR3DThreshold))   aCom +=  " 3DRANTh=" + ToString(mCAS3D.mR3DThreshold);
        aComSingle = StdCom("TestLib RANSAC R3D", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + "Dir=" + mCAS3D.mDir + BLANK + aCom, aExe);
        aComList.push_back(aComSingle);
    }
    /*
    for(list<std::string>::iterator it=aComList.begin();it!=aComList.end();it++)
    {
        cout<<(*it)<<endl;
    }
    */
    if(mExe && (!mSkipRANSAC3D))
        cEl_GPAO::DoComInParal(aComList);
    }

    //if(mSkipCrossCorr == false)
    {
        /**************************************/
        /* 2.5 - CrossCorrelation for precise matching */
        /**************************************/
        cout<<"-------CrossCorrelation-------"<<endl;
    aComList.clear();
    for(int i=0; i<nPairNum; i++)
    {
        std::string aImg1 = aOverlappedImgL[i];
        std::string aImg2 = aOverlappedImgR[i];

        std::string aPrefix = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_" ;

        aCom = "";
        if (!EAMIsInit(&mCAS3D.mCrossCorrelationInSH))   aCom +=  " CCInSH=" + aFeatureOutSH+"-3DRANSAC";
        if (!EAMIsInit(&mCAS3D.mCrossCorrelationOutSH))   aCom +=  " CCOutSH=" + aFeatureOutSH+"-3DRANSAC-CrossCorrelation";
        if (!EAMIsInit(&mCAS3D.mWindowSize))   aCom +=  " SzW=" + ToString(mCAS3D.mWindowSize);
        if (!EAMIsInit(&mCAS3D.mCrossCorrThreshold))   aCom +=  " CCTh=" + ToString(mCAS3D.mCrossCorrThreshold);
        aCom += " PatchSz=[" + ToString(mCAS3D.mPatchSz.x) + "," + ToString(mCAS3D.mPatchSz.y) + "]";
        aCom += " BufferSz=[" + ToString(mCAS3D.mBufferSz.x) + "," + ToString(mCAS3D.mBufferSz.y) + "]";
        aCom +=  " PatchDir=" + aOutDir;
        aCom +=  " SubPXml=" + aPrefix + mCAS3D.mSubPatchXml;
        //cout<<aCom<<endl;
        aCom +=  "  CheckFile=" + ToString(mCheckFile);
        //cout<<aCom<<endl;
        aComSingle = StdCom("TestLib CrossCorrelation", aImg1 + BLANK + aImg2 + BLANK + aCom, aExe);
        aComList.push_back(aComSingle);
    }
    /*
    for(list<std::string>::iterator it=aComList.begin();it!=aComList.end();it++)
    {
        cout<<(*it)<<endl;
    }
    */
    if(mExe && (!mSkipCrossCorr))
        cEl_GPAO::DoComInParal(aComList);
    }
    }
}

cAppliTiepHistoricalPipeline::cAppliTiepHistoricalPipeline(int argc,char** argv) :
        mDebug(false)

{
    mExe = true;
    mUseDepth = false;
    mDSMFileL = "MMLastNuage.xml";
    mDSMFileR = "MMLastNuage.xml";
    mFeature = "SuperGlue";
    //mCoRegOri = "Co-reg";
    mSkipCoReg = false;
    mSkipPrecise = false;
    mSkipGetPatchPair = false;
    mSkipTentativeMatch = false;
    mSkipRANSAC3D = false;
    mSkipCrossCorr = false;
    mRotateDSM = -1;
    mCheckFile = false;
    mImg4MatchList1 = "";
    mImg4MatchList2 = "";
   ElInitArgMain
   (
        argc,argv,
        LArgMain()
               << EAMC(mOri1,"Orientation of epoch1")
               << EAMC(mOri2,"Orientation of epoch2")
               << EAMC(mImgList1,"ImgList1: The list that contains all the RGB images of epoch1")
               << EAMC(mImgList2,"ImgList2: The list that contains all the RGB images of epoch2")
               << EAMC(mDSMDirL, "DSM directory of epoch1")
               << EAMC(mDSMDirR, "DSM directory of epoch2"),

        LArgMain()
               << EAM(mExe,"Exe",true,"Execute all, Def=true")
               << EAM(mImg4MatchList1,"I4ML1",true,"The list that contains the RGB images of epoch1 for extracting inter-epoch correspondences, Def=ImgList1")
               << EAM(mImg4MatchList2,"I4ML2",true,"The list that contains the RGB images of epoch2 for extracting inter-epoch correspondences, Def=ImgList2")
               << EAM(mCheckFile, "CheckFile", true, "Check if the result files exist (if so, skip), Def=false")
               << EAM(mUseDepth,"UseDep",true,"Use depth to improve perfomance, Def=false")
               << EAM(mRotateDSM,"RotateDSM",true,"The angle of rotation from the first DSM to the second DSM for rough co-registration (only 4 options available: 0, 90, 180, 270), Def=-1 (means all the 4 options will be executed, and the one with the most inlier will be kept) ")
               << EAM(mSkipCoReg, "SkipCoReg", true, "Skip the step of rough co-registration, when the input orientations of epoch1 and epoch 2 are already co-registrated, Def=false")
               << EAM(mSkipPrecise, "SkipPrecise", true, "Skip the step of the whole precise matching pipeline, Def=false")
               << EAM(mSkipGetPatchPair, "SkipGetPatchPair", true, "Skip the step of \"GetPatchPair\" in precise matching (for debug only), Def=false")
               << EAM(mSkipTentativeMatch, "SkipTentativeMatch", true, "Skip the step of SuperGlue or SIFT matching (for debug only), Def=false")
               << EAM(mSkipRANSAC3D, "SkipRANSAC3D", true, "Skip the step of 3D RANSAC (for debug only), Def=false")
               << EAM(mSkipCrossCorr, "SkipCrossCorr", true, "Skip the step of cross correlation (for debug only), Def=false")
               << EAM(mFeature,"Feature",true,"Feature matching method used for precise matching (SuperGlue or SIFT), Def=SuperGlue")
               //<< EAM(mCoRegOri,"CoRegOri",true,"Output of Co-registered orientation, Def=Co-reg")
               << mCAS3D.ArgBasic()
               << EAM(mDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
               << EAM(mDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")
               << mCAS3D.ArgDSM_Equalization()
               << mCAS3D.ArgGetPatchPair()
               << mCAS3D.ArgSuperGlue()
               << mCAS3D.ArgMergeTiePt()
               << mCAS3D.Arg2DRANSAC()
               << mCAS3D.ArgCreateGCPs()
               << mCAS3D.ArgGetOverlappedImages()
               << mCAS3D.ArgGuidedSIFT()
               << mCAS3D.Arg3DRANSAC()
               << mCAS3D.ArgCrossCorrelation()
/*
                    << EAM(mDebug, "Debug", true, "Debug mode, def false")
                    << mCAS3D.ArgBasic()
                    << mCAS3D.ArgRough()
*/
               );
   mCoRegOri = mOri2;

   if(mImg4MatchList1.length() == 0)
       mImg4MatchList1 = mImgList1;

   if(mImg4MatchList2.length() == 0)
       mImg4MatchList2 = mImgList2;

   StdCorrecNameOrient(mOri,mCAS3D.mDir,true);
}


/*******************************************/
/****** cTransform3DHelmert  ******/
/*******************************************/

cTransform3DHelmert::cTransform3DHelmert(std::string aFileName)
{
    //if(aFileName.length() == 0)
    if(ELISE_fp::exist_file(aFileName) == false)
    {
        if(aFileName.length() > 0)
            printf("File %s does not exist, hence will use unit matrix instead.\n", aFileName.c_str());
        mApplyTrans = false;
    }
    else
    {
        mApplyTrans = true;
        mTransf = OptStdGetFromPCP(aFileName, Xml_ParamBascRigide);
        mScl = mTransf->Scale();
        mTr = mTransf->Trans();
        //mRot = mTransf->ParamRotation();
    }
}

Pt3dr cTransform3DHelmert::Transform3Dcoor(Pt3dr aPt)
{
    if(mApplyTrans == false)
        return aPt;
    else
    {
        Pt3dr aPtBasc(
                    scal(mTransf->ParamRotation().L1() , aPt) * mScl + mTr.x,
                    scal(mTransf->ParamRotation().L2() , aPt) * mScl + mTr.y,
                    scal(mTransf->ParamRotation().L3() , aPt) * mScl + mTr.z
                     );

        return aPtBasc;
    }
}

/*******************************************/
/****** cGet3Dcoor  ******/
/*******************************************/

cGet3Dcoor::cGet3Dcoor(std::string aNameOri)
{
    int aType = eTIGB_Unknown;
    mCam1 = cBasicGeomCap3D::StdGetFromFile(aNameOri,aType);

    bDSM = false;
}

double cGet3Dcoor::GetGSD()
{
    double dZL = mCam1->GetAltiSol();

    Pt2dr aCent(double(mCam1->SzBasicCapt3D().x)/2,double(mCam1->SzBasicCapt3D().y)/2);
    Pt2dr aCentNeigbor(aCent.x+1, aCent.y);

    Pt3dr aCentTer = mCam1->ImEtZ2Terrain(aCent, dZL);
    Pt3dr aCentNeigborTer = mCam1->ImEtZ2Terrain(aCentNeigbor, dZL);

    double dist = pow(pow(aCentTer.x-aCentNeigborTer.x,2) + pow(aCentTer.y-aCentNeigborTer.y,2), 0.5);

    return dist;
}

Pt2di cGet3Dcoor::GetDSMSz(std::string aDSMFile, std::string aDSMDir)
{
    if(aDSMDir.length() == 0)
        return Pt2di(0,0);
    aDSMDir += "/";
    bDSM = true;

    cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
    (
    aDSMDir + aDSMFile,
    StdGetFileXMLSpec("SuperposImage.xml"),
    "XML_ParamNuage3DMaille",
    "XML_ParamNuage3DMaille"
    );

    return aNuageIn.NbPixel();
}

TIm2D<float,double> cGet3Dcoor::SetDSMInfo(std::string aDSMFile, std::string aDSMDir)
{
    //if(aDSMFile.length() > 0)
    {
        aDSMDir += "/";
        bDSM = true;

        cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
        (
        aDSMDir + aDSMFile,
        StdGetFileXMLSpec("SuperposImage.xml"),
        "XML_ParamNuage3DMaille",
        "XML_ParamNuage3DMaille"
        );

        mDSMSz = aNuageIn.NbPixel();

        cImage_Profondeur aImDSM = aNuageIn.Image_Profondeur().Val();
        std::string aImName = aDSMDir + aImDSM.Image();
        Tiff_Im aImDSMTif(aImName.c_str());

        Pt2di aSzOut = mDSMSz;
        TIm2D<float,double> aTImDSM(aSzOut);
        ELISE_COPY
        (
        aTImDSM.all_pts(),
        aImDSMTif.in(),
        aTImDSM.out()
        );

        mFOM = StdGetFromPCP(aDSMDir+StdPrefix(aImDSM.Image())+".xml",FileOriMnt);

        mOriPlani = mFOM.OriginePlani();
        mResolPlani = mFOM.ResolutionPlani();
/*
        cout<<aDSMDir<<StdPrefix(aImDSM.Image())+".xml"<<endl;
        cout<<mOriPlani.x<<", "<<mOriPlani.y<<endl;
        cout<<mResolPlani.x<<", "<<mResolPlani.y<<endl;
*/
        return aTImDSM;
    }
}

//get rough 3D coor with mean altitude
Pt3dr cGet3Dcoor::GetRough3Dcoor(Pt2dr aPt1)
{
    double dZ = mCam1->GetAltiSol();
    return mCam1->ImEtZ2Terrain(aPt1, dZ);
}

Pt2dr cGet3Dcoor::Get2Dcoor(Pt3dr aTer)
{
    Pt2dr aPLPred;
    //aPLPred = mCam1->R3toF2(aTer);
    aPLPred = mCam1->Ter2Capteur(aTer);

    return aPLPred;
}

//get 3d coordinate from DSM, if no DSM, get rough 3D coor with mean altitude
Pt3dr cGet3Dcoor::Get3Dcoor(Pt2dr aPt1, TIm2D<float,double> aTImDSM, bool& bPrecise, double dThres)
{
    bPrecise = true;

    Pt3dr aTer(0,0,0);
    Pt2dr ptPrj;

    //double dThres = 0.3;
    //tempo, check prof+dZ=posZ?
    double dZ = mCam1->GetAltiSol();
    double dDis;
    int nIter = 0;
    //printf("--------\nIter: %d, dZ: %lf, aTer.x: %lf, aTer.y: %lf, aTer.z: %lf\n", nIter, dZ, aTer.x, aTer.y, aTer.z);
//    cout<<"nIter: "<<nIter<<"; dZ: "<<dZ<<"; aTer: "<<aTer.x<<", "<<aTer.y<<", "<<aTer.z<<endl;
    do
    {
        nIter++;

        aTer = mCam1->ImEtZ2Terrain(aPt1, dZ);
        ptPrj = mCam1->Ter2Capteur(aTer);

        dDis = pow(pow(aPt1.x-ptPrj.x, 2) + pow(aPt1.y-ptPrj.y, 2), 0.5);

        if(nIter > 100)
        {
            printf("%lf %lf %lf %lf\n", aPt1.x,ptPrj.x,aPt1.y,ptPrj.y);
            printf("nIter: %d, dZ: %lf, aTer.x: %lf, aTer.y: %lf, aTer.z: %lf, dDis: %lf, dThres: %lf\n", nIter, dZ, aTer.x, aTer.y, aTer.z, dDis, dThres);
        }

        Pt2di aPt2;
        aPt2.x = int((aTer.x - mOriPlani.x)/mResolPlani.x + 0.5);
        aPt2.y = int((aTer.y - mOriPlani.y)/mResolPlani.y + 0.5);
        //out of border of the DSM
        if(aPt2.x<0 || aPt2.y<0 || aPt2.x >= mDSMSz.x || aPt2.y >= mDSMSz.y)
        {
            bPrecise = false;
            aTer = GetRough3Dcoor(aPt1);
            return aTer;
        }

        dZ =  aTImDSM.get(aPt2);
        aTer.z = dZ;
    }
    while(dDis > dThres);

    //printf("Final 3D: aTer.x: %lf, aTer.y: %lf, aTer.z: %lf\n", aTer.x, aTer.y, aTer.z);

    return aTer;
}

