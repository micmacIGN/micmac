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

#include <iomanip>

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
    //mBufferSz = Pt2dr(30, 60);
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
    mSpGlueOutSH = "-SuperGlue";
    mGuidedSIFTOutSH = "-GuidedSIFT";
    mMergeTiePtOutSH = "";
    mR2DInSH = "";
    mR2DOutSH = "";
    mR3DInSH = "";
    mR3DOutSH = "";
    mCrossCorrelationOutSH = "";
    mMergeTiePtInSH = "";
    mCreateGCPsInSH = "";
    mCrossCorrelationInSH = "";
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
//    mOutDir = "./Tmp_Patches";
//    mOutImg1 = "";
//    mOutImg2 = "";
    mOut2DXml1 = "";
    mOut2DXml2 = "";
    mOut3DXml1 = "";
    mOut3DXml2 = "";
    //mSubPatchLXml = "SubPL.xml";
    //mSubPatchRXml = "SubPR.xml";
    mStrEntSpG = "";
    mStrOpt = "";
    mPrint = false;
    mThreshScale = 0.2;
    mThreshAngle = 30;
    mMinPt = 10;

        *mArgBasic
            //                        << EAM(mExe,"Exe",true,"Execute all, Def=true")
                        << EAM(mPrint, "Print", false, "Print supplementary information, Def=false")
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
            //<< EAM(mBufferSz, "BufferSz", true, "Buffer zone size around the patch of the tiling scheme, Def=[30,60]")
            << EAM(mSubPatchXml, "SubPXml", true, "The output xml file name to record the homography between the patches and original image, Def=SubPatch.xml")
            //<< EAM(mDSMDirL, "DSMDirL", true, "DSM of master image (for improving the reprojecting accuracy), Def=none")
            //<< EAM(mDSMDirR, "DSMDirR", true, "DSM of secondary image (for improving the reprojecting accuracy), Def=none")
            //<< EAM(mOutImg1, "OutImg1", true, "Name of the main part of the output patches from master image, Def=master image name")
            //<< EAM(mOutImg2, "OutImg2", true, "Name of the main part of the output patches from secondary image, Def=secondary image name")
            << EAM(mImgPair, "ImgPair", true, "Output txt file that records the patch pairs, Def=SuperGlueInput.txt");


    *mArgSuperGlue
            //<< EAM(input_pairs, "input_pairs", true, "txt file that listed the image pairs")
            << EAM(mInput_dir, "InDir", true, "The input directory of the images for SuperGlue, Def=./")
            << EAM(mSpGlueOutSH, "SpGOutSH", true, "Homologue extenion for NB/NT mode of SuperGlue, Def=-SuperGlue")
            << EAM(mResize, "Resize", true, "The goal size for resizing the input image for SuperGlue, Def=[640,480], if you don't want to resize, please set to [-1,-1]")
            << EAM(mViz, "Viz", true, "Visualize the matches and dump the plots of SuperGlue, Def=false")
            << EAM(mModel, "Model", true, "Pretrained indoor or outdoor model of SuperGlue, Def=outdoor")
            << EAM(mMax_keypoints, "MaxPt", true, "Maximum number of keypoints detected by Superpoint, Def=1024")
            << EAM(mKeepNpzFile, "KeepNpzFile", true, "Keep the original npz file that SuperGlue outputed, Def=false")
            << EAM(mStrEntSpG, "EntSpG", true, "The SuperGlue program entry (this option is only used for developper), Def=../micmac/src/uti_phgrm/TiePHistorical/SuperGluePretrainedNetwork-master/match_pairs.py")
            << EAM(mStrOpt, "opt", true, "Other options for SuperGlue (this option is only used for developper), Def=none");

    *mArgMergeTiePt
        << EAM(mMergeTiePtInSH,"MergeInSH",true,"Input Homologue extenion for NB/NT mode for MergeTiePt, Def=none")
        << EAM(mHomoXml,"HomoXml",true,"Input xml file that recorded the homograhpy from patches to original image for MergeTiePt, Def=SubPatch.xml")
           //the name of MergeOutSH will be set as the same name of HomoXml, if not set by users
        << EAM(mMergeTiePtOutSH,"MergeOutSH",true,"Output Homologue extenion for NB/NT mode of MergeTiePt, Def=-SubPatch");


    *mArgCreateGCPs
        //<< EAM(mDSMDirL,"DSMDirL",true,"DSM direcotry of epoch1, Def=none")
        //<< EAM(mDSMDirR,"DSMDirR",true,"DSM direcotry of epoch2, Def=none")
        //<< EAM(mDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
        //<< EAM(mDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")
        << EAM(mCreateGCPsInSH,"CreateGCPsInSH",true,"Input Homologue extenion for NB/NT mode for CreateGCPs, Def=none")
        << EAM(mOut2DXml1,"Out2DXml1",true,"Output xml files of 2D obersevations of the GCPs in epoch1, Def=OutGCP2D'CreateGCPsInSH'_'Ori1'-CoReg_'Ori2'.xml")
        << EAM(mOut3DXml1,"Out3DXml1",true,"Output xml files of 3D obersevations of the GCPs in epoch1, Def=OutGCP3D'CreateGCPsInSH'_'Ori1'-CoReg_'Ori2'.xml")
        << EAM(mOut2DXml2,"Out2DXml2",true,"Output xml files of 2D obersevations of the GCPs in epoch2, Def=OutGCP2D'CreateGCPsInSH'_'Ori2'_'Ori1'-CoReg.xml")
        << EAM(mOut3DXml2,"Out3DXml2",true,"Output xml files of 3D obersevations of the GCPs in epoch2, Def=OutGCP3D'CreateGCPsInSH'_'Ori2'_'Ori1'-CoReg.xml");


    *mArgGetOverlappedImages
            << EAM(mOutPairXml,"OutPairXml",true,"Output Xml file to record the overlapped image pairs, Def=OverlappedImages.xml");


    *mArg2DRANSAC
        << EAM(mR2DInSH,"2DRANInSH",true,"Input Homologue extenion for NB/NT mode for 2D RANSAC, Def=none")
        << EAM(mR2DOutSH,"2DRANOutSH",true,"Output Homologue extenion for NB/NT mode of 2D RANSAC, Def='2DRANInSH'-2DRANSAC")
        << EAM(mR2DIteration,"2DIter",true,"2D RANSAC iteration, Def=1000")
        << EAM(mR2DThreshold,"2DRANTh",true,"2D RANSAC threshold, Def=10");

    *mArg3DRANSAC
        << EAM(mR3DInSH,"3DRANInSH",true,"Input Homologue extenion for NB/NT mode for 3D RANSAC, Def=none")
        << EAM(mR3DOutSH,"3DRANOutSH",true,"Output Homologue extenion for NB/NT mode of 3D RANSAC, Def='3DRANInSH'-3DRANSAC")
        << EAM(mR3DIteration,"3DIter",true,"3D RANSAC iteration, Def=1000")
        //<< EAM(mR3DThreshold,"3DRANTh",true,"3D RANSAC threshold, Def=10*RefGSD (if \"Para3DHL\" is valid, RefGSD is the average GSD of master and secondary image, otherwise RefGSD is the GSD of secondary image)")
        << EAM(mR3DThreshold,"3DRANTh",true,"3D RANSAC threshold, Def=10*(GSD of secondary image)")
                << EAM(mMinPt,"MinPt",true,"Minimun number of input correspondences required, Def=10");
           /*
        << EAM(mDSMDirL, "DSMDirL", true, "DSM directory of master image, Def=none")
        << EAM(mDSMDirR, "DSMDirR", true, "DSM directory of secondary image, Def=none")
        << EAM(mDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
        << EAM(mDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml");
               */


    *mArgGuidedSIFT
            /*
    << EAM(mDSMDirL, "DSMDirL", true, "DSM of master image (for improving the reprojecting accuracy), Def=none")
    << EAM(mDSMDirR, "DSMDirR", true, "DSM of secondary image (for improving the reprojecting accuracy), Def=none")
    << EAM(mDSMFileL, "DSMFileL", true, "DSM File of master image, Def=MMLastNuage.xml")
    << EAM(mDSMFileR, "DSMFileR", true, "DSM File of secondary image, Def=MMLastNuage.xml")
            */
    << EAM(mGuidedSIFTOutSH,"GSIFTOutSH",true,"Output Homologue extenion for NB/NT mode of Guided SIFT, Def=-GuidedSIFT")
    << EAM(mSkipSIFT,"SkipSIFT",true,"Skip extracting SIFT key points in case it is already done, Def=false")
    << EAM(mSearchSpace,"SearchSpace",true,"Radius of the search space for GuidedSIFT (the search space is the circle with the center on the predicted point), Def=100 (this value is based on master image, the search space on secondary image will multiply the scale difference if \"CheckScale\" is set to true)")
    << EAM(mMutualNN, "MutualNN",true, "Apply mutual nearest neighbor on GuidedSIFT, Def=true")
    << EAM(mRatioT, "RatioT",true, "Apply ratio test on GuidedSIFT, Def=true")
    << EAM(mRootSift, "RootSift",true, "Use RootSIFT as descriptor on GuidedSIFT, Def=true")
    << EAM(mCheckScale, "CheckScale",true, "Check the scale of the candidate tie points on GuidedSIFT, Def=true")
    << EAM(mCheckAngle, "CheckAngle",true, "Check the angle of the candidate tie points on GuidedSIFT, Def=true")
    //<< EAM(mScale, "Scale",true, "The scale ratio used for checking the candidate tie points on GuidedSIFT, Def=1")
    //<< EAM(mAngle, "Angle",true, "The angle difference used for checking the candidate tie points on GuidedSIFT, Def=0")
    << EAM(mThreshScale, "ScaleTh",true, "The threshold for checking scale ratio, Def=0.2; (0.2 means the ratio of master and secondary SIFT scale between [(1-0.2)*Ref, (1+0.2)*Ref] is considered valide. Ref is automatically calculated by reprojection.)")
    << EAM(mThreshAngle, "AngleTh",true, "The threshold for checking angle difference, Def=30; (30 means the difference of master and secondary SIFT angle between [Ref - 30 degree, Ref + 30 degree] is considered valide. Ref is automatically calculated by reprojection.)")
    << EAM(mPredict, "Predict",true, "Use the predicted key points to guide the matching, Def=true");

    *mArgCrossCorrelation
//            << EAM(mBufferSz, "BufferSz", true, "Buffer sie, Def=[30,60]")
            << EAM(mCrossCorrelationInSH,"CCInSH",true,"Input Homologue extenion for NB/NT mode for cross correlation, Def=none")
            << EAM(mCrossCorrelationOutSH,"CCOutSH",true,"Output Homologue extenion for NB/NT mode of cross correlation, Def='CCInSH'-CrossCorrelation")
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

void cCommonAppliTiepHistorical::CorrectXmlFileName(std::string aCreateGCPsInSH, std::string aOri1, std::string aOri2)
{
    aOri1 = RemoveOri(aOri1);
    aOri2 = RemoveOri(aOri2);
    if(mOut2DXml1.length() == 0)			mOut2DXml1 = "OutGCP2D" + aCreateGCPsInSH + "_"+aOri1+"-CoReg"+ "_"+aOri2+".xml";
    if(mOut2DXml2.length() == 0)			mOut2DXml2 = "OutGCP2D" + aCreateGCPsInSH + "_"+aOri2+ "_"+aOri1+"-CoReg"+".xml";
    if(mOut3DXml1.length() == 0)			mOut3DXml1 = "OutGCP3D" + aCreateGCPsInSH + "_"+aOri1+"-CoReg"+ "_"+aOri2+".xml";
    if(mOut3DXml2.length() == 0)			mOut3DXml2 = "OutGCP3D" + aCreateGCPsInSH + "_"+aOri2+ "_"+aOri1+"-CoReg"+".xml";
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
    //if (EAMIsInit(&mPatchSz))    aCom += " PatchSz=[" + ToString(mPatchSz.x) + "," + ToString(mPatchSz.y) + "]";
    //if (EAMIsInit(&mBufferSz))    aCom += " BufferSz=[" + ToString(mBufferSz.x) + "," + ToString(mBufferSz.y) + "]";
    if (EAMIsInit(&mSubPatchXml))  aCom +=  " SubPXml=" + mSubPatchXml;
    //if (EAMIsInit(&mOutDir))    aCom +=  " PatchOutDir=" + mOutDir;
    if (EAMIsInit(&mImgPair))  aCom +=  " ImgPair=" + mImgPair;

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamSuperGlue()
{
    //cout<<"mViz "<<mViz<<endl;
    std::string aCom ="";
    if (EAMIsInit(&mInput_dir))   aCom +=  " InDir=" + mDir + "/" + mInput_dir;
    //if (EAMIsInit(&mOutput_dir))   aCom +=  " SpGOutDir=" + mDir + "/" + mOutput_dir;
    if (EAMIsInit(&mSpGlueOutSH))   aCom +=  " SpGOutSH=" + mSpGlueOutSH;
    if (EAMIsInit(&mResize))    aCom += " Resize=[" + ToString(mResize.x) + "," + ToString(mResize.y) + "]";
    if (EAMIsInit(&mViz))       aCom += " Viz=" + ToString(mViz);
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
    if (EAMIsInit(&mR2DInSH))   aCom +=  " 2DRANInSH=" + mR2DInSH;
    if (EAMIsInit(&mR2DOutSH))  aCom +=  " 2DRANOutSH=" + mR2DOutSH;
    if (EAMIsInit(&mR2DIteration))          aCom +=  " 2DIter=" + ToString(mR2DIteration);
    if (EAMIsInit(&mR2DThreshold))          aCom +=  " 2DRANTh=" + ToString(mR2DThreshold);

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamRANSAC3D()
{
    std::string aCom = "";
    if (EAMIsInit(&mR3DIteration))   aCom +=  " 3DIter=" + ToString(mR3DIteration);
    if (EAMIsInit(&mR3DThreshold))   aCom +=  " 3DRANTh=" + ToString(mR3DThreshold);
    if (EAMIsInit(&mMinPt))   aCom +=  " MinPt=" + ToString(mMinPt);

    return aCom;
}

std::string cCommonAppliTiepHistorical::ComParamCreateGCPs()
{
    std::string aCom = "";
    if (EAMIsInit(&mCreateGCPsInSH))     aCom +=  " CreateGCPsInSH=" + mCreateGCPsInSH;
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

/*******************************************/
/****** cAppliTiepHistoricalPipeline  ******/
/*******************************************/

std::string StdCom(const std::string & aCom,const std::string & aPost, bool aExe)
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
    std::cout << aFullCom << "\n";

    if (aExe)
    {
       //std::cout << "---------->>>>  " << aFullCom << "\n";
       System(aFullCom);
    }
    //else

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

int GetTiePtNum(std::string aDir, std::string aImg1, std::string aImg2, std::string aSH)
{
    std::string aDir_inSH = aDir + "/Homol" + aSH+"/";
    std::string aNameIn = aDir_inSH +"Pastis" + aImg1 + "/"+aImg2+".txt";

    if (ELISE_fp::exist_file(aNameIn) == false)
    {
        cout<<aNameIn<<" didn't exist hence skipped (GetTiePtNum)."<<endl;
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

int SaveTxtImgPair(std::string aName, std::vector<std::string> aResL, std::vector<std::string> aResR)
{
    int num=0;
    bool bPair = true;
    FILE * fpPair = fopen(aName.c_str(), "w");

    if (NULL == fpPair)
    {
        cout<<"Open file "<<aName<<" failed"<<endl;
        bPair = false;
    }

    for(int i=0; i<int(aResL.size()); i++)
    {
        if(1) //for(int j=0; j<int(aResR.size()); j++)
        {
            if(bPair)
                fprintf(fpPair, "%s %s\n", aResL[i].c_str(), aResR[i].c_str());

            num++;
        }
    }
    fclose(fpPair);

    return num;
}

int SaveXmlImgPair(std::string aName, std::vector<std::string> aResL, std::vector<std::string> aResR)
{
    cSauvegardeNamedRel aLCpleOri;
    int num=0;
    for(int i=0; i<int(aResL.size()); i++)
    {
        for(int j=0; j<int(aResR.size()); j++)
        {
            cCpleString aCplCalc(aResL[i], aResR[j]);
            aLCpleOri.Cple().push_back(aCplCalc);
            num++;
        }
    }
    MakeFileXML(aLCpleOri, aName);
    return num;
}

int GetXmlImgPair(std::string aName, std::vector<std::string>& aResL, std::vector<std::string>& aResR)
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

bool cAppliTiepHistoricalPipeline::CheckFileComplete()
{
    std::vector<std::string> aVIm1;
    std::vector<std::string> aVIm2;

    GetImgListVec(mImgList1, aVIm1, 1);
    GetImgListVec(mImgList2, aVIm2, 1);

    unsigned int i;
    std::string aImgName;
    std::string aOri2 = RemoveOri(mOri2);

    StdCorrecNameOrient(mOri1,"./",true);
    for (i=0;i<aVIm1.size();i++)
    {
        aImgName = aVIm1[i];
        //std::string aImgOri = "./Ori-"+mOri1+"/Orientation-"+aImgName+".xml";
        std::string aImgOri = mICNM->StdNameCamGenOfNames(mOri1, aImgName);
        if (ELISE_fp::exist_file(aImgOri) == false)
        {
            cout<<"The orientation file of "<<aImgName<<" doesn't exist."<<endl;
            bFileComplete = false;
            //return false;
        }
    }

    StdCorrecNameOrient(aOri2,"./",true);
    for (i=0;i<aVIm2.size();i++)
    {
        aImgName = aVIm2[i];
        //std::string aImgOri = "./Ori-"+aOri2+"/Orientation-"+aImgName+".xml";
        std::string aImgOri = mICNM->StdNameCamGenOfNames(aOri2, aImgName);
        if (ELISE_fp::exist_file(aImgOri) == false)
        {
            cout<<"The orientation file of "<<aImgName<<" doesn't exist."<<endl;
            bFileComplete = false;
            //return false;
        }
    }

    std::string aDSMDir = "./" + mDSMDirL + "/";
    std::string aDSMFile = aDSMDir + "MMLastNuage.xml";
    if (ELISE_fp::exist_file(aDSMFile) == false)
    {
        cout<<aDSMFile<<" doesn't exist."<<endl;
        bFileComplete = false;

        //return false;
    }
    else{
        cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
        (
        aDSMFile,
        StdGetFileXMLSpec("SuperposImage.xml"),
        "XML_ParamNuage3DMaille",
        "XML_ParamNuage3DMaille"
        );

        cImage_Profondeur aImProfPx = aNuageIn.Image_Profondeur().Val();
        std::string aImName = aDSMDir + aImProfPx.Image();

        if (ELISE_fp::exist_file(aImName) == false)
        {
            cout<<aImName<<" doesn't exist."<<endl;
            bFileComplete = false;
        }
    }

    aDSMDir = "./" + mDSMDirR + "/";
    aDSMFile = aDSMDir + "MMLastNuage.xml";
    if (ELISE_fp::exist_file(aDSMFile) == false)
    {
        cout<<aDSMFile<<" doesn't exist."<<endl;
        bFileComplete = false;
        //return false;
    }
    else{
        cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
        (
        aDSMFile,
        StdGetFileXMLSpec("SuperposImage.xml"),
        "XML_ParamNuage3DMaille",
        "XML_ParamNuage3DMaille"
        );

        cImage_Profondeur aImProfPx = aNuageIn.Image_Profondeur().Val();
        std::string aImName = aDSMDir + aImProfPx.Image();

        if (ELISE_fp::exist_file(aImName) == false)
        {
            cout<<aImName<<" doesn't exist."<<endl;
            bFileComplete = false;
        }
    }
/*

    << EAMC(mOri1,"Ori1: Orientation of epoch1")
    << EAMC(mOri2,"Ori2: Orientation of epoch2")
    << EAMC(mImgList1,"ImgList1: All RGB images in epoch1 (Dir+Pattern, or txt file of image list)")
    << EAMC(mImgList2,"ImgList2: All RGB images in epoch2 (Dir+Pattern, or txt file of image list)")
    << EAMC(mDSMDirL, "DSM directory of epoch1")
    << EAMC(mDSMDirR, "DSM directory of epoch2"),
*/
    return true;
}

void cAppliTiepHistoricalPipeline::DoAll()
{
    std::string aCom;

    if(bFileComplete == false)
    {
        cout<<"Input files incomplete, therefore skipped. Please check the hints above."<<endl;
        return;
    }

    std::string aBaseOutDir = "./Tmp_Patches";
    if(aBaseOutDir.find("/") == aBaseOutDir.length()-1)
        aBaseOutDir = aBaseOutDir.substr(0, aBaseOutDir.length()-1);
    std::string aOutDir = aBaseOutDir;

    std::string aOri1 = mOri1;
    StdCorrecNameOrient(aOri1,"./",true);
    std::string aOri2 = mOri2;
    StdCorrecNameOrient(aOri2,"./",true);

    bool bUseOrtho = false;
    if (mOrthoDirL.length()>0 && mOrthoDirR.length()>0 && ELISE_fp::exist_file(mOrthoDirL+"/"+mOrthoFileL) == true && ELISE_fp::exist_file(mOrthoDirR+"/"+mOrthoFileR) == true)
        bUseOrtho = true;

    if(mSkipCoReg == false)
    {
        printf("*****************************************************************************************************\n");
        printf("************************************** (1) Rough co-registration ************************************\n");
        printf("*****************************************************************************************************\n");
        /******************** (1) rough co-registration******************/

        aOutDir = aBaseOutDir + "-CoReg";

        std::string aDSMImgNameL = GetImage_Profondeur(mCAS3D.mDir+mDSMDirL, mDSMFileL);
        std::string aDSMImgNameR = GetImage_Profondeur(mCAS3D.mDir+mDSMDirR, mDSMFileR);
        std::string aDSMImgGrayNameL = StdPrefix(aDSMImgNameL)+"_gray."+StdPostfix(aDSMImgNameL);
        std::string aDSMImgGrayNameR = StdPrefix(aDSMImgNameR)+"_gray."+StdPostfix(aDSMImgNameR);

        cout<<"############################# (1.1) Preprocess DSM #############################"<<endl;
        /**************************************/
        /* 1.1 - DSM_Equalization and wallis filter */
        /**************************************/
        if(bUseOrtho == false){
            StdCom("TestLib DSM_Equalization", mDSMDirL + BLANK + "DSMFile="+mDSMFileL + BLANK
                                                                        + mCAS3D.ComParamDSM_Equalization(), mExe);
            StdCom("TestLib DSM_Equalization", mDSMDirR + BLANK + "DSMFile="+mDSMFileR + BLANK
                                                                        + mCAS3D.ComParamDSM_Equalization(), mExe);

            StdCom("TestLib Wallis", aDSMImgGrayNameL + BLANK + "Dir="+mDSMDirL, mExe);
            StdCom("TestLib Wallis", aDSMImgGrayNameR + BLANK + "Dir="+mDSMDirR, mExe);
        }

        std::string aDSMImgWallisDirL = mDSMDirL;
        std::string aDSMImgWallisDirR = mDSMDirR;
        std::string aDSMImgWallisNameL = aDSMImgGrayNameL+"_sfs.tif";
        std::string aDSMImgWallisNameR = aDSMImgGrayNameR+"_sfs.tif";
        if(bUseOrtho == true){
            aDSMImgWallisDirL = mOrthoDirL;
            aDSMImgWallisDirR = mOrthoDirR;
            aDSMImgWallisNameL = mOrthoFileL;
            aDSMImgWallisNameR = mOrthoFileR;
        }
        std::string aDSMImgGrayNameRenamedL = mCAS3D.GetFolderName(aDSMImgWallisDirL) + "." + StdPostfix(aDSMImgNameL);
        std::string aDSMImgGrayNameRenamedR = mCAS3D.GetFolderName(aDSMImgWallisDirR) + "." + StdPostfix(aDSMImgNameR);

        cout<<"################################ (1.2) Match DSM ###############################"<<endl;
        std::string aFinalOutSH;
        if(mFeature == "SuperGlue")
        {
            /**************************************/
            /* 1.2 - GetPatchPair for rough co-registration */
            /**************************************/
            cout<<"-------Get patch pairs (One-to-many tiling)-------"<<endl;
            aCom = "";
            //if (!EAMIsInit(&mCAS3D.mOutDir))   aCom +=  " OutDir=" + aOutDir;
            //if (EAMIsInit(&mCoRegPatchLSz))
            aCom += " PatchLSz=[" + ToString(mCoRegPatchLSz.x) + "," + ToString(mCoRegPatchLSz.y) + "]";
            //if (EAMIsInit(&mCoRegBufferLSz))
            aCom += " BufferLSz=[" + ToString(mCoRegBufferLSz.x) + "," + ToString(mCoRegBufferLSz.y) + "]";
            //if (EAMIsInit(&mCoRegPatchRSz))
            aCom += " PatchRSz=[" + ToString(mCoRegPatchRSz.x) + "," + ToString(mCoRegPatchRSz.y) + "]";
            //if (EAMIsInit(&mCoRegBufferRSz))
            aCom += " BufferRSz=[" + ToString(mCoRegBufferRSz.x) + "," + ToString(mCoRegBufferRSz.y) + "]";
            StdCom("TestLib GetPatchPair BruteForce", aDSMImgWallisDirL+"/"+aDSMImgWallisNameL + BLANK + aDSMImgWallisDirR+"/"+aDSMImgWallisNameR + BLANK + aCom + BLANK + "Rotate=" + ToString(mRotateDSM) + BLANK + mCAS3D.ComParamGetPatchPair(), mExe);


            std::string aRotate[4] = {"", "_R90", "_R180", "_R270"};
            int nMaxinlier = 0;
            //Rotate the master DSM 4 times and apply superGlue
            for(int i=0; i<4; i++)
            {
                cout<<"-------"<<i+1<<"th rotation hypothesis-------"<<endl;
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
                //if (!EAMIsInit(&mCAS3D.mOutput_dir))   aCom +=  " SpGOutDir=" + aOutDir+"/";
                aCom +=  " CheckNb=\" " + ToString(mCheckNbCoReg) + "\"";
                StdCom("TestLib SuperGlue", aImgPair + BLANK + aCom + BLANK + mCAS3D.ComParamSuperGlue(), mExe);


                /**************************************/
                /* 1.4 - MergeTiePt for rough co-registration */
                /**************************************/
                std::string aHomoXml = StdPrefix(mCAS3D.mHomoXml) + aRotate[i] + "." + StdPostfix(mCAS3D.mHomoXml);
                aCom = "";
                if (!EAMIsInit(&mCAS3D.mHomoXml))   aCom +=  " HomoXml=" + aHomoXml;
                if (!EAMIsInit(&mCAS3D.mMergeTiePtInSH))   aCom +=  " MergeInSH=" + mCAS3D.mSpGlueOutSH;
                aCom +=  " PatchSz=[" + ToString(mCoRegPatchLSz.x) + "," + ToString(mCoRegPatchLSz.y) + "]";
                aCom +=  " BufferSz=[" + ToString(mCoRegBufferLSz.x) + "," + ToString(mCoRegBufferLSz.y) + "]";
                StdCom("TestLib MergeTiePt", aOutDir+"/" + BLANK + aCom + BLANK + mCAS3D.ComParamMergeTiePt(), mExe);


                /**************************************/
                /* 1.5 - RANSAC R2D for rough co-registration */
                /**************************************/
                aCom = "";
                if (!EAMIsInit(&mCAS3D.mR2DInSH))   aCom +=  " 2DRANInSH=-" + StdPrefix(aHomoXml);
                std::string aRANSACOutSH = "-" + StdPrefix(aHomoXml) + "-2DRANSAC";
                StdCom("TestLib RANSAC R2D", aDSMImgGrayNameRenamedL + BLANK + aDSMImgGrayNameRenamedR + BLANK + "Dir=" + aOutDir+"/" + BLANK + aCom + BLANK + mCAS3D.ComParamRANSAC2D(), mExe);
                int nInlier = GetTiePtNum(aOutDir, aDSMImgGrayNameRenamedL, aDSMImgGrayNameRenamedR, aRANSACOutSH);

                cout<<"Tie points: Homol"<<aRANSACOutSH<<"; Inlier number: "<<nInlier<<endl;

                if(nInlier > nMaxinlier)
                {
                    nMaxinlier = nInlier;
                    aFinalOutSH = aRANSACOutSH;
                }
            }
        }
        else if(mFeature == "SIFT")
        {
            std::string strCpImg;
            strCpImg = "cp "+aDSMImgWallisDirL+"/"+aDSMImgWallisNameL+" "+aOutDir+"/"+aDSMImgGrayNameRenamedL;
            cout<<strCpImg<<endl;
            if(mExe)            System(strCpImg);

            strCpImg = "cp "+aDSMImgWallisDirR+"/"+aDSMImgWallisNameR+" "+aOutDir+"/"+aDSMImgGrayNameRenamedR;
            cout<<strCpImg<<endl;
            if(mExe)            System(strCpImg);

            std::string aRANSACOutSH;

            if (0){
                std::string aInSH = "-SIFT";
                StdCom("Tapioca MulScale ", "\"" + aOutDir + "/" + aDSMImgGrayNameRenamedL + "|" + aDSMImgGrayNameRenamedR + "\" 500 2000 ExpTxt=1 PostFix="+aInSH, mExe);

                aCom = "";
                if (!EAMIsInit(&mCAS3D.mR2DInSH))   aCom +=  " 2DRANInSH=" + aInSH;
                aRANSACOutSH = aInSH + "-2DRANSAC";
                StdCom("TestLib RANSAC R2D", aDSMImgGrayNameRenamedL + BLANK + aDSMImgGrayNameRenamedR + BLANK + "Dir=" + aOutDir+"/" + BLANK + aCom + BLANK + mCAS3D.ComParamRANSAC2D(), mExe);
            }
            else{
                std::string aCom = "";
                if (EAMIsInit(&mScaleL))   aCom +=  " ScaleL=" + ToString(mScaleL);
                if (EAMIsInit(&mScaleR))   aCom +=  " ScaleR=" + ToString(mScaleR);

                StdCom("TestLib SIFT2Step ", aDSMImgGrayNameRenamedL + BLANK + aDSMImgGrayNameRenamedR + " Skip2ndSIFT=1 Dir="+aOutDir + aCom, mExe);

                aRANSACOutSH = "-SIFT2Step-Rough-2DRANSAC";
            }

            int nInlier = GetTiePtNum(aOutDir, aDSMImgGrayNameRenamedL, aDSMImgGrayNameRenamedR, aRANSACOutSH);
            cout<<aRANSACOutSH<<","<<nInlier<<endl;

            aFinalOutSH = aRANSACOutSH;
        }
        else
            printf("Please set Feature to SuperGlue or SIFT\n");

        cout<<"aFinalOutSH: "<<aFinalOutSH<<endl;


        /**************************************/
        /* 1.6 - CreateGCPs for rough co-registration */
        /**************************************/
        cout<<"###################### (1.3) Create virtual GCPs from DSMs ######################"<<endl;
        aCom = "";
        if (!EAMIsInit(&mCAS3D.mCreateGCPsInSH))   aCom +=  " CreateGCPsInSH=" + aFinalOutSH;
        if (EAMIsInit(&mOrthoDirL))               aCom +=  " OrthoDirL=" + mOrthoDirL;
        if (EAMIsInit(&mOrthoDirR))               aCom +=  " OrthoDirR=" + mOrthoDirR;
        if (EAMIsInit(&mOrthoFileL))              aCom +=  " OrthoFileL=" + mOrthoFileL;
        if (EAMIsInit(&mOrthoFileR))              aCom +=  " OrthoFileR=" + mOrthoFileR;
        StdCom("TestLib CreateGCPs", aOutDir + BLANK + aDSMImgGrayNameRenamedL + BLANK + aDSMImgGrayNameRenamedR + BLANK + mCAS3D.mDir + BLANK + mImgList1 + BLANK + mImgList2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + mDSMDirL + BLANK + mDSMDirR + aCom + mCAS3D.ComParamCreateGCPs(), mExe);


        mCAS3D.CorrectXmlFileName(aFinalOutSH, mOri1, mOri2);
        /**************************************/
        /* 1.7 - GCPBascule for rough co-registration */
        /**************************************/
        cout<<"######################## (1.4) 3D Helmert transformation ########################"<<endl;
        aCom = "";
        std::string aImgListL = GetImgList(mCAS3D.mDir, mImgList1, mExe);
        StdCom("GCPBascule", aImgListL + BLANK + mOri1 + BLANK + mCoRegOri1 /*mCoRegOri.substr(4,mCoRegOri.length())*/ + BLANK + mCAS3D.mOut3DXml2 + BLANK + mCAS3D.mOut2DXml1, mExe);
        /*
        aCom = "/home/lulin/Documents/ThirdParty/oldMicmac/micmac_old/bin/mm3d GCPBascule " + aImgListL + BLANK + mOri1 + BLANK + mCoRegOri.substr(4,mCoRegOri.length()) + BLANK + mCAS3D.mOut3DXml2 + BLANK + mCAS3D.mOut2DXml1;
        if(mExe==true)
            System(aCom);
        cout<<aCom<<endl;
        */
    }


    if(mSkipPrecise == false)
    {
        printf("************************************************************************************************\n");
        printf("************************************** (2) Precise matching ************************************\n");
        printf("************************************************************************************************\n");
    /********************2- precise matching******************/
    aOutDir = aBaseOutDir + "-Precise";

    /**************************************/
    /* 2.1 - GetOverlappedImages */
    /**************************************/
    cout<<"######################## (2.1) Get overlapped image pairs ########################"<<endl;
    if(mSkipGetOverlappedImages == false){
        //StdCom("TestLib GetOverlappedImages", mOri1 + BLANK + mOri2 + BLANK + mImg4MatchList1 + BLANK + mImg4MatchList2 + BLANK + mCAS3D.ComParamGetOverlappedImages() + BLANK + "Para3DH=Basc-"+aOri1+"-2-"+mCoRegOri1+".xml", mExe);
        StdCom("TestLib GetOverlappedImages", mOri1 + BLANK + mOri2 + BLANK + mImg4MatchList1 + BLANK + mImg4MatchList2 + BLANK + mCAS3D.ComParamGetOverlappedImages() + BLANK + "Para3DH="+mPara3DH, 1);
    }

    cout<<mCAS3D.mOutPairXml<<endl;
    if (ELISE_fp::exist_file(mCAS3D.mOutPairXml) == false)
    {
        cout<<mCAS3D.mOutPairXml<<" didn't exist because the pipeline is not executed, hence the precise matching commands are not shown here."<<endl;
        return;
    }

    bool aExe = false;
    std::vector<std::string> aOverlappedImgL;
    std::vector<std::string> aOverlappedImgR;
    int nPairNum = GetXmlImgPair(mCAS3D.mOutPairXml, aOverlappedImgL, aOverlappedImgR);

    std::list<std::string> aComList;
    std::string aComSingle;
    /**************************************/
    /* 2.2 - GetPatchPair for precise matching */
    /**************************************/
    //cout<<"-------GetPatchPair-------"<<endl;
    cout<<"#################### (2.2) Get patch pairs (One-to-one tiling) ####################"<<endl;
    //if(mSkipGetPatchPair == false)
    {
    for(int i=0; i<nPairNum; i++)
    {
        std::string aImg1 = aOverlappedImgL[i];
        std::string aImg2 = aOverlappedImgR[i];
/*
        cout<<"####################-------"<<i<<"th pair####################----"<<endl;
        cout<<aImg1<<" "<<aImg2<<endl;
*/
        std::string aPrefix = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_" ;
        aCom = "";
        //if (!EAMIsInit(&mCAS3D.mOutDir))   aCom +=  " OutDir=" + aOutDir;
        if (!EAMIsInit(&mCAS3D.mSubPatchXml))  aCom +=  " SubPXml=" + aPrefix + mCAS3D.mSubPatchXml;
        if (!EAMIsInit(&mCAS3D.mImgPair))   aCom +=  " ImgPair=" + aPrefix + mCAS3D.mImgPair;
        //if (EAMIsInit(&mPrecisePatchSz))
        aCom += " PatchSz=[" + ToString(mPrecisePatchSz.x) + "," + ToString(mPrecisePatchSz.y) + "]";
        if (EAMIsInit(&mDyn))               aCom += " Dyn=" + ToString(mDyn);
        //if (EAMIsInit(&mPreciseBufferSz))
        aCom += " BufferSz=[" + ToString(mPreciseBufferSz.x) + "," + ToString(mPreciseBufferSz.y) + "]";
        if (EAMIsInit(&mReprojTh))          aCom += " Thres=" + ToString(mReprojTh);
        //aComSingle = StdCom("TestLib GetPatchPair Guided", aImg1 + BLANK + aImg2 + BLANK + mCoRegOri + BLANK + mCoRegOri + BLANK + aCom + BLANK + mCAS3D.ComParamGetPatchPair(), aExe);
        //printf("%s\t%s\n", aOri1.c_str(), mOri1.c_str());
        //aComSingle = StdCom("TestLib GetPatchPair Guided", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGetPatchPair() + BLANK + "Para3DH=Basc-"+aOri1+"-2-"+mCoRegOri1+".xml" + BLANK + "DSMDirL="+mDSMDirL, aExe);
        aComSingle = StdCom("TestLib GetPatchPair Guided", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGetPatchPair() + BLANK + "Para3DH="+mPara3DH + BLANK + "DSMDirL="+mDSMDirL, aExe);
                aComList.push_back(aComSingle);

        if(mUseDepth == true)
        {
            //aComSingle = StdCom("TestLib GetPatchPair Guided", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGetPatchPair() + BLANK + "Para3DH=Basc-"+aOri1+"-2-"+mCoRegOri1+".xml" + BLANK + "DSMDirL="+mDSMDirL + BLANK + "Prefix=Depth_", aExe);
            aComSingle = StdCom("TestLib GetPatchPair Guided", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGetPatchPair() + BLANK + "Para3DH="+mPara3DH + BLANK + "DSMDirL="+mDSMDirL + BLANK + "Prefix=Depth_", aExe);
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

    std::string aRANSACInSH;
    std::string aCrossCorrInSH;
    std::string aCrossCorrOutSH;

    //if(mSkipTentativeMatch == false)
    cout<<"######################### (2.3) Get tentative tie-points #########################"<<endl;
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
            //if (!EAMIsInit(&mCAS3D.mOutput_dir))   aCom +=  " SpGOutDir=" + aOutDir+"/";
            aCom +=  "  CheckFile=" + ToString(mCheckFile);
            aCom +=  " CheckNb=\" " + ToString(mCheckNbPrecise) + "\"";
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
                aRANSACInSH = mCAS3D.mSpGlueOutSH;
            }
            else
                aRANSACInSH = mCAS3D.mMergeTiePtOutSH;
            aCom +=  " PatchSz=[" + ToString(mPrecisePatchSz.x) + "," + ToString(mPrecisePatchSz.y) + "]";
            aCom +=  " BufferSz=[" + ToString(mPreciseBufferSz.x) + "," + ToString(mPreciseBufferSz.y) + "]";
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
            std::vector<std::string> aVIm1;
            std::vector<std::string> aVIm2;
            GetImgListVec(mCAS3D.mDir+mImg4MatchList1, aVIm1, mExe);
            GetImgListVec(mCAS3D.mDir+mImg4MatchList2, aVIm2, mExe);
            for(unsigned int k=0; k<aVIm1.size(); k++)
                ExtractSIFT(aVIm1[k], mCAS3D.mDir, mScaleL);
            for(unsigned int k=0; k<aVIm2.size(); k++)
                ExtractSIFT(aVIm2[k], mCAS3D.mDir, mScaleR);
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
            if (EAMIsInit(&mScaleL))   aCom +=  " ScaleL=" + ToString(mScaleL);
            if (EAMIsInit(&mScaleR))   aCom +=  " ScaleR=" + ToString(mScaleR);
            aCom +=  "  CheckFile=" + ToString(mCheckFile);
            //aComSingle = StdCom("TestLib GuidedSIFTMatch", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGuidedSIFTMatch() + BLANK + "Para3DHL=Basc-"+aOri1+"-2-"+mCoRegOri1+".xml" + BLANK + "Para3DHR=Basc-"+aOri2+"-2-"+aOri1+".xml", aExe);
            //aComSingle = StdCom("TestLib GuidedSIFTMatch", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGuidedSIFTMatch() + BLANK + "Para3DH=Basc-"+aOri1+"-2-"+mCoRegOri1+".xml", aExe);
            aComSingle = StdCom("TestLib GuidedSIFTMatch", aImg1 + BLANK + aImg2 + BLANK + mOri1 + BLANK + mOri2 + BLANK + aCom + BLANK + mCAS3D.ComParamGuidedSIFTMatch() + BLANK + "Para3DH="+mPara3DH, aExe);


            aRANSACInSH = mCAS3D.mGuidedSIFTOutSH;
            aComList.push_back(aComSingle);
        }
        /*
        for(list<std::string>::iterator it=aComList.begin();it!=aComList.end();it++)
        {
            cout<<(*it)<<endl;
        }
        */
        if(mExe && (!mSkipTentativeMatch))
            //cEl_GPAO::DoComInParal(aComList);
            cEl_GPAO::DoComInSerie(aComList);
    }
    else
    {
        cout<<"Please set Feature to SuperGlue or SIFT"<<endl;
        return;
    }
    //cout<<"aRANSACInSH: "<<aRANSACInSH<<endl;
    }

    /**************************************/
    /* 2.4 - RANSAC R3D for precise matching */
    /**************************************/
    //if(mSkipRANSAC3D == false)
    {
        cout<<"#################### (2.4) Get enhanced tie-points (3D RANSAC) ####################"<<endl;
        //cout<<"-------RANSAC R3D-------"<<endl;
    aComList.clear();
    for(int i=0; i<nPairNum; i++)
    {
        std::string aImg1 = aOverlappedImgL[i];
        std::string aImg2 = aOverlappedImgR[i];

        aCom = mCAS3D.ComParamRANSAC3D();
        /*
        if (EAMIsInit(&mDSMFileL))   aCom +=  " DSMFileL=" + mDSMFileL;
        if (EAMIsInit(&mDSMFileR))   aCom +=  " DSMFileR=" + mDSMFileR;
        if (EAMIsInit(&mCAS3D.mR3DIteration))   aCom +=  " 3DIter=" + ToString(mCAS3D.mR3DIteration);
        if (EAMIsInit(&mCAS3D.mR3DThreshold))   aCom +=  " 3DRANTh=" + ToString(mCAS3D.mR3DThreshold);
        if (EAMIsInit(&mCAS3D.mMinPt))   aCom +=  " MinPt=" + ToString(mCAS3D.mMinPt);
        */
        aCom +=  " DSMDirL=" + mDSMDirL;
        aCom +=  " DSMDirR=" + mDSMDirR;
        if (EAMIsInit(&mCAS3D.mR3DInSH))
            aRANSACInSH = mCAS3D.mR3DInSH;
        if (EAMIsInit(&mCAS3D.mR3DOutSH))
            aCrossCorrInSH = mCAS3D.mR3DOutSH;
        else
            aCrossCorrInSH = aRANSACInSH+"-3DRANSAC";
        aCom +=  " 3DRANInSH=" + aRANSACInSH;
        aCom +=  " 3DRANOutSH=" + aCrossCorrInSH;
        //aCom +=  " Para3DHL=Basc-"+aOri1+"-2-"+mCoRegOri1+".xml";
        aCom +=  "  CheckFile=" + ToString(mCheckFile);
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
        //cEl_GPAO::DoComInSerie(aComList);
    }

    //if(mSkipCrossCorr == false)
    {
        /**************************************/
        /* 2.5 - CrossCorrelation for precise matching */
        /**************************************/
        cout<<"################### (2.5) Get final tie-points (cross correlation) ##################"<<endl;
        //cout<<"-------CrossCorrelation-------"<<endl;
    aComList.clear();
    for(int i=0; i<nPairNum; i++)
    {
        std::string aImg1 = aOverlappedImgL[i];
        std::string aImg2 = aOverlappedImgR[i];

        std::string aPrefix = StdPrefix(aImg1) + "_" + StdPrefix(aImg2) + "_" ;

        aCom = "";
        if (EAMIsInit(&mCAS3D.mCrossCorrelationInSH))
            aCrossCorrInSH = mCAS3D.mCrossCorrelationInSH;
        if (EAMIsInit(&mCAS3D.mCrossCorrelationOutSH))
            aCrossCorrOutSH = mCAS3D.mCrossCorrelationOutSH;
        else
            aCrossCorrOutSH = aCrossCorrInSH+"-CrossCorrelation";
        aCom +=  " CCInSH=" + aCrossCorrInSH;
        aCom +=  " CCOutSH=" + aCrossCorrOutSH;
        /*
        if (!EAMIsInit(&mCAS3D.mCrossCorrelationInSH)){
            aCom +=  " CCInSH=" + aRANSACInSH+"-3DRANSAC";
            aCrossCorrOutSH = aRANSACInSH+"-3DRANSAC-CrossCorrelation";
        }
        else                                             aCom +=  " CCInSH=" + mCAS3D.mCrossCorrelationInSH;
        if (!EAMIsInit(&mCAS3D.mCrossCorrelationOutSH))  aCom +=  " CCOutSH=" + aCrossCorrOutSH;
        else                                             aCom +=  " CCOutSH=" + mCAS3D.mCrossCorrelationOutSH;
        */
        aCom +=  " SzW=" + ToString(mCAS3D.mWindowSize);
        aCom +=  " CCTh=" + ToString(mCAS3D.mCrossCorrThreshold);
        //if (EAMIsInit(&mPrecisePatchSz))
        aCom += " PatchSz=[" + ToString(mPrecisePatchSz.x) + "," + ToString(mPrecisePatchSz.y) + "]";
        aCom += " BufferSz=[" + ToString(mPreciseBufferSz.x) + "," + ToString(mPreciseBufferSz.y) + "]";
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

    mOrthoDirL = "";
    mOrthoDirR = "";
    mOrthoFileL = "Orthophotomosaic.tif";
    mOrthoFileR = "Orthophotomosaic.tif";

    mFeature = "SuperGlue";
    //mCoRegOri = "Co-reg";
    mCoRegOri1 = "";
    mSkipCoReg = false;
    mSkipPrecise = false;
    mSkipGetOverlappedImages = false;
    mSkipGetPatchPair = false;
    mSkipTentativeMatch = false;
    mSkipRANSAC3D = false;
    mSkipCrossCorr = false;
    mRotateDSM = -1;
    mCheckFile = false;
    mImg4MatchList1 = "";
    mImg4MatchList2 = "";
    mCoRegPatchLSz = Pt2dr(1920, 1440);
    mCoRegBufferLSz = Pt2dr(0, 0);
    mCoRegPatchRSz = Pt2dr(1920, 1440);
    mCoRegBufferRSz = Pt2dr(0, 0);

    mPrecisePatchSz = Pt2dr(1920, 1440);
    mPreciseBufferSz = Pt2dr(-1, -1);

    mCheckNbCoReg = -1;
    mCheckNbPrecise = 100;

    mDyn = 0.1;
    mScaleL = 1;
    mScaleR = 1;

    mReprojTh = 2;

    mPara3DH = "";

   ElInitArgMain
   (
        argc,argv,
        LArgMain()
               << EAMC(mOri1,"Ori1: Orientation of epoch1")
               << EAMC(mOri2,"Ori2: Orientation of epoch2")
               << EAMC(mImgList1,"ImgList1: All RGB images in epoch1 (Dir+Pattern, or txt file of image list)")
               << EAMC(mImgList2,"ImgList2: All RGB images in epoch2 (Dir+Pattern, or txt file of image list)")
               << EAMC(mDSMDirL, "DSM directory of epoch1")
               << EAMC(mDSMDirR, "DSM directory of epoch2"),

        LArgMain()
               << EAM(mExe,"Exe",true,"Execute all, Def=true. If this parameter is set to false, the pipeline will not be executed and the command of all the submodules will be printed.")
               << EAM(mImg4MatchList1,"IL1",true,"RGB images in epoch1 for extracting inter-epoch correspondences (Dir+Pattern, or txt file of image list), Def=ImgList1")
               << EAM(mImg4MatchList2,"IL2",true,"RGB images in epoch2 for extracting inter-epoch correspondences (Dir+Pattern, or txt file of image list), Def=ImgList2")
               << EAM(mCheckFile, "CheckFile", true, "Check if the result files of inter-epoch correspondences exist (if so, skip to avoid repetition), Def=false")
               << EAM(mUseDepth,"UseDep",true,"GetPatchPair for depth maps as well (this option is only used for developper), Def=false")
               << EAM(mRotateDSM,"Rotate",true,"The angle of clockwise rotation from the master DSM/orthophoto to the secondary DSM/orthophoto for rough co-registration (only 5 options available: 0, 90, 180, 270 and -1. -1 means all the 4 rotations (0, 90, 180, 270) will be executed, and the one with the most inlier will be kept.), Def=-1")
               << EAM(mSkipCoReg, "SkipCoReg", true, "Skip the step of rough co-registration, when the input orientations of epoch1 and epoch 2 are already co-registrated, Def=false")
               << EAM(mSkipPrecise, "SkipPrecise", true, "Skip the step of the whole precise matching pipeline, Def=false")
               << EAM(mSkipGetOverlappedImages, "SkipGetOverlappedImages", true, "Skip the step of \"mGetOverlappedImages\" in precise matching (this option is used when the results of \"GetOverlappedImages\" already exist), Def=false")
               << EAM(mSkipGetPatchPair, "SkipGetPatchPair", true, "Skip the step of \"GetPatchPair\" in precise matching (this option is used when the results of \"GetPatchPair\" already exist), Def=false")
               << EAM(mSkipTentativeMatch, "SkipTentativeMatch", true, "Skip the step of \"SuperGlue\" or SIFT matching (this option is used when the results of \"SuperGlue\" or SIFT matching already exist), Def=false")
               << EAM(mSkipRANSAC3D, "SkipRANSAC3D", true, "Skip the step of \"3D RANSAC\" (this option is used when the results of \"3D RANSAC\" already exist), Def=false")
               << EAM(mSkipCrossCorr, "SkipCrossCorr", true, "Skip the step of \"cross correlation\" (this option is used when the results of \"cross correlation\" already exist), Def=false")
               << EAM(mFeature,"Feature",true,"Feature matching method used for precise matching (SuperGlue or SIFT), Def=SuperGlue")
               << EAM(mCoRegOri1,"CoRegOri1",true,"Output of orientation of epoch1 after rough co-registration, Def='Ori1'-CoReg-'Feature'")
               << EAM(mPara3DH, "Para3DH", true, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of master image to secondary image, Def=Basc-'aOri1'-2-'CoRegOri1'.xml")
               //<< mCAS3D.ArgBasic()
               << EAM(mDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
               << EAM(mDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")

               << EAM(mOrthoDirL, "OrthoDirL", true, "Orthophoto directory of epoch1 (if this parameter is set, the orthophotos would be used for rough co-registration instead of DSM), Def=none")
               << EAM(mOrthoDirR, "OrthoDirR", true, "Orthophoto directory of epoch2 (if this parameter is set, the orthophotos would be used for rough co-registration instead of DSM), Def=none")
               << EAM(mOrthoFileL, "OrthoFileL", true, "Orthophoto file of epoch1, Def=Orthophotomosaic.tif")
               << EAM(mOrthoFileR, "OrthoFileR", true, "Orthophoto file of epoch2, Def=Orthophotomosaic.tif")

               << EAM(mCoRegPatchLSz, "CoRegPatchLSz", true, "Patch size of the tiling scheme for master image in rough co-registration part, which means the master images to be matched by SuperGlue will be split into patches of this size, Def=[1920, 1440]")
               << EAM(mCoRegBufferLSz, "CoRegBufferLSz", true, "Buffer zone size around the patch of the tiling scheme for master image in rough co-registration part, Def=[0, 0]")
               << EAM(mCoRegPatchRSz, "CoRegPatchRSz", true, "Patch size of the tiling scheme for secondary image in rough co-registration part, which means the secondary images to be matched by SuperGlue will be split into patches of this size, Def=[1920, 1440]")
               << EAM(mCoRegBufferRSz, "CoRegBufferRSz", true, "Buffer zone size around the patch of the tiling scheme for secondary image in rough co-registration part, Def=[0, 0]")

               << EAM(mPrecisePatchSz, "PrecisePatchSz", true, "Patch size of the tiling scheme in precise matching part, which means the images to be matched by SuperGlue will be split into patches of this size, Def=[1920, 1440]")
               << EAM(mPreciseBufferSz, "PreciseBufferSz", true, "Buffer zone size around the patch of the tiling scheme in precise matching part, Def=10%*PrecisePatchSz")
               << mCAS3D.ArgDSM_Equalization()
               << mCAS3D.ArgGetPatchPair()
               << EAM(mDyn, "Dyn", true, "The Dyn parameter in \"to8Bits\" if the input RGB images are 16 bits, Def=0.1")
               << mCAS3D.ArgSuperGlue()
               << EAM(mCheckNbCoReg,"CheckNbCoReg",true,"Radius of the search space for SuperGlue in rough co-registration step (which means correspondence [(xL, yL), (xR, yR)] with (xL-xR)*(xL-xR)+(yL-yR)*(yL-yR) > CheckNb*CheckNb will be removed afterwards), Def=-1 (means don't check search space)")
               << EAM(mCheckNbPrecise,"CheckNbPrecise",true,"Radius of the search space for SuperGlue in precise matching step (which means correspondence [(xL, yL), (xR, yR)] with (xL-xR)*(xL-xR)+(yL-yR)*(yL-yR) > CheckNb*CheckNb will be removed afterwards), Def=100")
               << mCAS3D.ArgMergeTiePt()
               << mCAS3D.Arg2DRANSAC()
               << mCAS3D.ArgCreateGCPs()
               << mCAS3D.ArgGetOverlappedImages()
               << mCAS3D.ArgGuidedSIFT()
               << EAM(mScaleL, "ScaleL", true, "Extract SIFT points on master images downsampled with a factor of \"ScaleL\", Def=1")
               << EAM(mScaleR, "ScaleR", true, "Extract SIFT points on secondary images downsampled with a factor of \"ScaleR\", Def=1")
               << EAM(mReprojTh, "ReprojTh", true, "EThe threshold of reprojection error (unit: pixel) when prejecting patch corner to DSM, Def=2")
               << mCAS3D.Arg3DRANSAC()
               << mCAS3D.ArgCrossCorrelation()
/*
                    << EAM(mDebug, "Debug", true, "Debug mode, def false")
                    << mCAS3D.ArgBasic()
                    << mCAS3D.ArgRough()
*/
               );

   StdCorrecNameOrient(mOri1,mCAS3D.mDir,true);
   //cout<<mOri1<<endl;

   //mCoRegOri = mOri2;
   if(mCoRegOri1.length() == 0)
       mCoRegOri1 = mOri1 + "-CoReg-" + mFeature;
   mCoRegOri1 = RemoveOri(mCoRegOri1);
   mOri1 = RemoveOri(mOri1);

   if(mPara3DH.length() == 0)
       mPara3DH = "Basc-"+mOri1+"-2-"+mCoRegOri1+".xml";

   if(mPreciseBufferSz.x < 0 && mPreciseBufferSz.y < 0){
       mPreciseBufferSz.x = int(0.1*mPrecisePatchSz.x);
       mPreciseBufferSz.y = int(0.1*mPrecisePatchSz.y);
   }

   if(mImg4MatchList1.length() == 0)
       mImg4MatchList1 = mImgList1;

   if(mImg4MatchList2.length() == 0)
       mImg4MatchList2 = mImgList2;

   bFileComplete = true;
   CheckFileComplete();
}

std::string RemoveOri(std::string aOri)
{
    StdCorrecNameOrient(aOri,"./",true);
    while(aOri.substr(0,4) == "Ori-")
        aOri = aOri.substr(4, aOri.length()-4);

    return aOri;
}

/*******************************************/
/****** cTransform3DHelmert  ******/
/*******************************************/

cTransform3DHelmert::cTransform3DHelmert(std::string aFileName):
    mSBR(cSolBasculeRig::Id()),
    mSBRInv(cSolBasculeRig::Id())
{
    //if(aFileName.length() == 0)
    if(ELISE_fp::exist_file(aFileName) == false)
    {
        if(aFileName.length() > 0)
            printf("File %s does not exist, hence will use unit matrix instead.\n", aFileName.c_str());
        mApplyTrans = false;
        mScl = 1;
        mTr = Pt3dr(0,0,0);
    }
    else
    {
        mApplyTrans = true;
        mTransf = OptStdGetFromPCP(aFileName, Xml_ParamBascRigide);
        mScl = mTransf->Scale();
        mTr = mTransf->Trans();
        //mRot = mTransf->ParamRotation();

        mSBR = Xml2EL(*mTransf);
        mSBRInv = mSBR.Inv();
    }
}

bool cTransform3DHelmert::GetApplyTrans()
{
    return mApplyTrans;
}

double cTransform3DHelmert::GetScale()
{
    return mScl;
}

cSolBasculeRig cTransform3DHelmert::GetSBR()
{
    return mSBR;
}

cSolBasculeRig cTransform3DHelmert::GetSBRInv()
{
    return mSBRInv;
}

Pt3dr cTransform3DHelmert::Transform3Dcoor(Pt3dr aPt)
{
    if(mApplyTrans == false)
        return aPt;
    else
    {
        /*
        Pt3dr aPtBasc(
                    scal(mTransf->ParamRotation().L1() , aPt) * mScl + mTr.x,
                    scal(mTransf->ParamRotation().L2() , aPt) * mScl + mTr.y,
                    scal(mTransf->ParamRotation().L3() , aPt) * mScl + mTr.z
                     );*/
        Pt3dr aPtBasc = mSBR(aPt);

        return aPtBasc;
    }
}

/*******************************************/
/****** cDSMInfo  ******/
/*******************************************/

cDSMInfo::cDSMInfo(Pt2di aDSMSz, std::string aDSMFile, std::string aDSMDir) :
mTImDSM  (aDSMSz),
mTImMask (aDSMSz)
{
    mDSMSz = aDSMSz;
    bDSM = true;

    if(true)
    {
        aDSMDir += "/";

        if (ELISE_fp::exist_file(aDSMDir + aDSMFile) == false)
        {
            if(aDSMDir.length() > 1)
                printf("%s didn't exist\n", (aDSMDir + aDSMFile).c_str());
            bDSM = false;
        }
        else
        {
            cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
            (
            aDSMDir + aDSMFile,
            StdGetFileXMLSpec("SuperposImage.xml"),
            "XML_ParamNuage3DMaille",
            "XML_ParamNuage3DMaille"
            );

            mDSMSz = aNuageIn.NbPixel();

            cImage_Profondeur aImDSM = aNuageIn.Image_Profondeur().Val();

            mDSMName = aImDSM.Image();
            std::string aDSMFullName = aDSMDir + mDSMName;
            //Tiff_Im aImDSMTif(aDSMFullName.c_str());
            Tiff_Im aImDSMTif = Tiff_Im::StdConvGen((aDSMFullName).c_str(), -1, true ,true);
            ELISE_COPY
            (
            mTImDSM.all_pts(),
            aImDSMTif.in(),
            mTImDSM.out()
            );

            mMaskName = aImDSM.Masq();
            std::string aMaskFullName = aDSMDir + mMaskName;
            //Tiff_Im aImMaskTif(aMaskFullName.c_str());
            Tiff_Im aImMaskTif = Tiff_Im::StdConvGen((aMaskFullName).c_str(), -1, true ,true);
            ELISE_COPY
            (
            mTImMask.all_pts(),
            aImMaskTif.in(),
            mTImMask.out()
            );

            mFOM = StdGetFromPCP(aDSMDir+StdPrefix(mDSMName)+".xml",FileOriMnt);

            mOriPlani = mFOM.OriginePlani();
            mResolPlani = mFOM.ResolutionPlani();
        }
    }
}

double cDSMInfo::GetDSMValue(Pt2di aPt2)
{
    if(bDSM == false)
        return 0;

    return mTImDSM.get(aPt2);
}

double cDSMInfo::GetMasqValue(Pt2di aPt2)
{
    if(bDSM == false)
        return 0;

    return mTImMask.get(aPt2);
}

//get 2d coordinate in DSM
Pt2dr cDSMInfo::Get2DcoorInDSM(Pt3dr aTer)
{
    if(bDSM == false)
        return Pt2dr(0,0);

    Pt2dr aPt2;
    aPt2.x = (aTer.x - mOriPlani.x)/mResolPlani.x + 0.5;
    aPt2.y = (aTer.y - mOriPlani.y)/mResolPlani.y + 0.5;

    return aPt2;
}

Pt2di cDSMInfo::GetDSMSz(std::string aDSMFile, std::string aDSMDir)
{
    if(aDSMDir.length() == 0)
        return Pt2di(0,0);
    aDSMDir += "/";

    cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
    (
    aDSMDir + aDSMFile,
    StdGetFileXMLSpec("SuperposImage.xml"),
    "XML_ParamNuage3DMaille",
    "XML_ParamNuage3DMaille"
    );

    return aNuageIn.NbPixel();
}

std::string cDSMInfo::GetDSMName(std::string aDSMFile, std::string aDSMDir)
{
    if(bDSM == false)
        return "";

    if(aDSMDir.length() == 0)
        return "";
    aDSMDir += "/";
    cXML_ParamNuage3DMaille aNuageIn = StdGetObjFromFile<cXML_ParamNuage3DMaille>
    (
    aDSMDir + aDSMFile,
    StdGetFileXMLSpec("SuperposImage.xml"),
    "XML_ParamNuage3DMaille",
    "XML_ParamNuage3DMaille"
    );

    cImage_Profondeur aImDSM = aNuageIn.Image_Profondeur().Val();

    return aImDSM.Image();
}

Pt2dr cDSMInfo::GetOriPlani()
{
    if(bDSM == false)
        return Pt2dr(0,0);

    return mOriPlani;
}

Pt2dr cDSMInfo::GetResolPlani()
{
    if(bDSM == false)
        return Pt2dr(0,0);

    return mResolPlani;
}

Pt2di cDSMInfo::GetDSMSz()
{
    if(bDSM == false)
        return Pt2di(0,0);

    return mDSMSz;
}

bool cDSMInfo::GetIfDSMIsValid()
{
    return bDSM;
}

/*******************************************/
/****** cGet3Dcoor  ******/
/*******************************************/

cGet3Dcoor::cGet3Dcoor(std::string aNameOri)
{
    int aType = eTIGB_Unknown;
    mCam1 = cBasicGeomCap3D::StdGetFromFile(aNameOri,aType);

    mZ = mCam1->GetAltiSol();

    bDSM = false;
}

double cGet3Dcoor::GetGSD()
{
    //double dZL = mCam1->GetAltiSol();

    Pt2dr aCent(double(mCam1->SzBasicCapt3D().x)/2,double(mCam1->SzBasicCapt3D().y)/2);
    Pt2dr aCentNeigbor(aCent.x+1, aCent.y);

    Pt3dr aCentTer = mCam1->ImEtZ2Terrain(aCent, mZ);
    Pt3dr aCentNeigborTer = mCam1->ImEtZ2Terrain(aCentNeigbor, mZ);

    double dist = pow(pow(aCentTer.x-aCentNeigborTer.x,2) + pow(aCentTer.y-aCentNeigborTer.y,2), 0.5);

    return dist;
}

cDSMInfo cGet3Dcoor::SetDSMInfo(std::string aDSMFile, std::string aDSMDir)
{
    Pt2di aDSMSz = cDSMInfo::GetDSMSz(aDSMFile, aDSMDir);
    cDSMInfo aDSMInfo(aDSMSz, aDSMFile, aDSMDir);
    bDSM = aDSMInfo.GetIfDSMIsValid();
    return aDSMInfo;
}

/*
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
        //Tiff_Im aImDSMTif(aImName.c_str());
        Tiff_Im aImDSMTif = Tiff_Im::StdConvGen(aImName.c_str(), -1, true ,true);

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

        return aTImDSM;
    }
}
*/
//get rough 3D coor with mean altitude
Pt3dr cGet3Dcoor::GetRough3Dcoor(Pt2dr aPt1)
{
    //double dZ = mCam1->GetAltiSol();
    return mCam1->ImEtZ2Terrain(aPt1, mZ);
}

Pt2dr cGet3Dcoor::Get2Dcoor(Pt3dr aTer)
{
    Pt2dr aPLPred;
    //aPLPred = mCam1->R3toF2(aTer);
    aPLPred = mCam1->Ter2Capteur(aTer);

    return aPLPred;
}

//get 3d coordinate from DSM, if no DSM, get rough 3D coor with mean altitude
Pt3dr cGet3Dcoor::Get3Dcoor(Pt2dr aPt1, cDSMInfo aDSMInfo, bool& bPrecise, bool bPrint, double dThres)
{
    bPrecise = true;

    Pt3dr aTer(0,0,0);
    Pt2dr ptPrj;

    double dZ = mZ;
    double dDis = 0;
    int nIter = 0;

    if(bPrint)
        printf("--->>>AltiSol: %.2lf\n", dZ);

    if(bDSM == true)
    {
        Pt2dr aOriPlani = aDSMInfo.GetOriPlani();
        Pt2dr aResolPlani = aDSMInfo.GetResolPlani();
        Pt2di aDSMSz = aDSMInfo.GetDSMSz();
        do
        {
            aTer = mCam1->ImEtZ2Terrain(aPt1, dZ);

            Pt2di aPt2;
            aPt2.x = int((aTer.x - aOriPlani.x)/aResolPlani.x + 0.5);
            aPt2.y = int((aTer.y - aOriPlani.y)/aResolPlani.y + 0.5);

            //out of border of the DSM
            if(aPt2.x<0 || aPt2.y<0 || aPt2.x >= aDSMSz.x || aPt2.y >= aDSMSz.y)
            {
                bPrecise = false;
                if(bPrint == true)
                    printf("Point (%.2lf, %.2lf) out of border of the DSM (Projected px in DSM: %d, %d; DSM size: %d, %d), hence use average altitude %.2lf instead.\n", aPt1.x, aPt1.y, aPt2.x, aPt2.y, aDSMSz.x, aDSMSz.y, mZ);
            }
            else if(aDSMInfo.GetMasqValue(aPt2) < 0.0001){
                bPrecise = false;
                if(bPrint == true)
                    printf("Point (%.2lf, %.2lf) out of mask of the DSM (Projected px in DSM: %d, %d), hence use average altitude %.2lf instead.\n", aPt1.x, aPt1.y, aPt2.x, aPt2.y, mZ);
            }

            //don't converge
            if(nIter > 10){
                bPrecise = false;
                if(bPrint == true){
                    printf("Iteration > 100, hence use average altitude instead. ");
                    printf("aTer: %.2lf, %.2lf, %.2lf\n", aTer.x, aTer.y, aTer.z);
                }
            }

            if(bPrecise == false)
            {
                aTer = GetRough3Dcoor(aPt1);
                return aTer;
            }

            dZ =  aDSMInfo.GetDSMValue(aPt2);
            aTer.z = dZ;

            ptPrj = mCam1->Ter2Capteur(aTer);
            dDis = pow(pow(aPt1.x-ptPrj.x, 2) + pow(aPt1.y-ptPrj.y, 2), 0.5);

            if(bPrint == true)
            {
                printf("nIter: %d; PxInDSM: [%d, %d], dZ: %.2lf, aTer.x: %.2lf, aTer.y: %.2lf, aTer.z: %.2lf, dDis: %.2lf, dThres: %.2lf\n", nIter, aPt2.x, aPt2.y, dZ, aTer.x, aTer.y, aTer.z, dDis, dThres);
                printf("ptOri: %.2lf %.2lf; ptReproj: %.2lf %.2lf\n", aPt1.x,aPt1.y,ptPrj.x,ptPrj.y);
            }
            nIter++;
        }
        while(dDis > dThres);
    }
    else
        aTer = GetRough3Dcoor(aPt1);

    return aTer;
}

void GetRandomNum(int nMin, int nMax, int nNum, std::vector<int> & res)
{
    //srand((int)time(0));
    int idx = 0;
    for(int i=0; i<nNum; i++)
    {
        bool bRepeat = false;
        int nIter = 0;
        do
        {
            bRepeat = false;
            nIter++;
            idx = rand() % (nMax - nMin) + nMin;
            //printf("For %dth seed, %dth generation, random value: %d\n", i, nIter, idx);
            for(int j=0; j<int(res.size()); j++)
            {
                if(idx == res[j]){
                    bRepeat = true;
                    break;
                }
            }
        }
        while(bRepeat == true);
        res.push_back(idx);
    }
}

void GetRandomNum(double dMin, double dMax, int nNum, std::vector<double> & res)
{
    //srand((int)time(0));
    int idx = 0;
    for(int i=0; i<nNum; i++)
    {
        bool bRepeat = false;
        int nIter = 0;
        do
        {
            bRepeat = false;
            nIter++;
            idx = (rand()*1.0/RAND_MAX)*(dMax-dMin) + dMin;
            //printf("For %dth seed, %dth generation, random value: %d\n", i, nIter, idx);
            /*
            for(int j=0; j<int(res.size()); j++)
            {
                if(fabs(idx - res[j]) < 0.00001){
                    bRepeat = true;
                    break;
                }
            }
            */
        }
        while(bRepeat == true);
        res.push_back(idx);
    }
}

bool GetImgs(std::string aFullPattern, std::vector<std::string>& aVIm, bool bPrint)
{        // Initialize name manipulator & files
    std::string aDirImages,aPatIm;
    SplitDirAndFile(aDirImages,aPatIm,aFullPattern);

    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));

    if(bPrint)
        std::cout<<"Selected files:"<<std::endl;
    for (unsigned int i=0;i<aSetIm.size();i++)
    {
        if(bPrint)
            std::cout<<" - "<<aSetIm[i]<<std::endl;
        aVIm.push_back(aSetIm[i]);
    }
    return true;
}

std::string GetImgListVec(std::string aFullPattern, std::vector<std::string>& aVIm, bool bPrint)
{
    std::string aFileType = "";
    //image list
    if(aFullPattern.substr(aFullPattern.length()-4,4) == ".txt")
    {
        if (ELISE_fp::exist_file(aFullPattern) == false)
            printf("File %s does not exist.\n", aFullPattern.c_str());

        std::string s;

        ifstream in1(aFullPattern);
        if(bPrint)
            printf("Images in %s:\n", aFullPattern.c_str());
        while(getline(in1,s))
        {
            aVIm.push_back(s);
            if(bPrint)
                printf(" - %s\n", s.c_str());
        }
        aFileType = "txt";
    }
    else if(aFullPattern.substr(aFullPattern.length()-4,4) == ".xml")
    {
        if (ELISE_fp::exist_file(aFullPattern) == false)
            printf("File %s does not exist.\n", aFullPattern.c_str());

        cListOfName         aLDirMec = StdGetFromPCP(aFullPattern,ListOfName);
        auto aDir_it = aLDirMec.Name().begin();

        if(bPrint)
            printf("Images in %s:\n", aFullPattern.c_str());

        for (int i=0; i<(int)aLDirMec.Name().size(); i++)
        {
            std::string s = (*aDir_it);
            aVIm.push_back(s);
            if(bPrint)
                printf(" - %s\n", s.c_str());
            (*aDir_it++);
        }
        aFileType = "xml";
    }
    //image pattern
    else
    {
        GetImgs(aFullPattern, aVIm, bPrint);
        /*
        // Initialize name manipulator & files
        std::string aDirImages,aPatIm;
        SplitDirAndFile(aDirImages,aPatIm,aFullPattern);

        cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
        const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));

        if(bPrint)
            std::cout<<"Selected files:"<<std::endl;
        for (unsigned int i=0;i<aSetIm.size();i++)
        {
            if(bPrint)
                std::cout<<" - "<<aSetIm[i]<<std::endl;
            aVIm.push_back(aSetIm[i]);
        }
        */
    }
    return aFileType;
}

bool FallInBox(Pt2dr* aPCorner, Pt2dr aLeftTop, Pt2di aRightLower)
{
    if((aPCorner[0].x < aLeftTop.x) && (aPCorner[1].x < aLeftTop.x) && (aPCorner[2].x < aLeftTop.x) && (aPCorner[3].x < aLeftTop.x))
        return false;

    if((aPCorner[0].y < aLeftTop.y) && (aPCorner[1].y < aLeftTop.y) && (aPCorner[2].y < aLeftTop.y) && (aPCorner[3].y < aLeftTop.y))
        return false;

    if((aPCorner[0].x > aRightLower.x) && (aPCorner[1].x > aRightLower.x) && (aPCorner[2].x > aRightLower.x) && (aPCorner[3].x > aRightLower.x))
        return false;

    if((aPCorner[0].y > aRightLower.y) && (aPCorner[1].y > aRightLower.y) && (aPCorner[2].y > aRightLower.y) && (aPCorner[3].y > aRightLower.y))
        return false;

    return true;
}

void ReadXml(std::string & aImg1, std::string & aImg2, std::string aSubPatchXml, std::vector<std::string>& vPatchesL, std::vector<std::string>& vPatchesR, std::vector<cElHomographie>& vHomoL, std::vector<cElHomographie>& vHomoR)
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

void GetBoundingBox(Pt3dr* ptTerrCorner, int nLen, Pt3dr& minPt, Pt3dr& maxPt)
{
    minPt = ptTerrCorner[0];
    maxPt = ptTerrCorner[0];
    for(int i=0; i<nLen; i++){
        Pt3dr ptCur = ptTerrCorner[i];

        if(minPt.x > ptCur.x)
            minPt.x = ptCur.x;
        if(maxPt.x < ptCur.x)
            maxPt.x = ptCur.x;

        if(minPt.y > ptCur.y)
            minPt.y = ptCur.y;
        if(maxPt.y < ptCur.y)
            maxPt.y = ptCur.y;

        if(minPt.z > ptCur.z)
            minPt.z = ptCur.z;
        if(maxPt.z < ptCur.z)
            maxPt.z = ptCur.z;
    }
}

bool CheckRange(int nMin, int nMax, double & value)
{
    if(nMax < nMin)
        return false;
    if(value < nMin)
        value = nMin;
    if(value > nMax)
        value = nMax;
    return true;
}

std::string GetScaledImgName(std::string aImgName, Pt2di ImgSz, double dScale)
{
    int nSz1 = int(max(ImgSz.x, ImgSz.y)*1.0/dScale);

    //cout<<max(ImgSz.x, ImgSz.y)*1.0/nSz1<<endl;
    double dScaleNm = max(ImgSz.x, ImgSz.y)*1.0/nSz1;


    int nScaleNm = int(dScaleNm*10  + 0.5);
    /*if(dScaleNm < 9.95){
        nScaleNm = int(dScaleNm*10  + 0.5);
    }*/

    //printf("%d, %.2lf, %d\n", nSz1, dScaleNm, nScaleNm);

    std::string aImgScaledName = "Resol" + ToString(nScaleNm) + "_Teta0_" + StdPrefix(aImgName)+".tif";

    return aImgScaledName;
}

std::string ExtractSIFT(std::string aImgName, std::string aDir, double dScale)
{
    std::string aImgNameWithDir = aDir+"/"+aImgName;
    if (ELISE_fp::exist_file(aImgNameWithDir) == false)
    {
        cout<<aImgNameWithDir<<" didn't exist, hence skipped"<<endl;
        return "";
    }

    //Tiff_Im aRGBIm1(aImgNameWithDir.c_str());
    Tiff_Im aRGBIm1 = Tiff_Im::StdConvGen(aImgNameWithDir.c_str(), -1, true ,true);
    Pt2di ImgSz = aRGBIm1.sz();
    int nSz1 = int(max(ImgSz.x, ImgSz.y)*1.0/dScale);

    std::string aImgScaledName = GetScaledImgName(aImgName, ImgSz, dScale);

    std::string aImgKey = aDir + "/Pastis/LBPp" + aImgScaledName+".dat";
    if (ELISE_fp::exist_file(aImgKey) == true){
        //cout<<aImgKey<<" already exist, hence skipped"<<endl;
        return aImgKey;
    }

    std::string aComm;
    aComm = MMBinFile(MM3DStr) + "PastDevlop " + aDir+"/"+aImgName + " Sz1=" + ToString(nSz1) +" Sz2=-1";
    cout<<aComm<<endl;
    System(aComm);

    aComm = MMBinFile(MM3DStr) + "SIFT " + aDir+"/Pastis/"+aImgScaledName + " -o " + aDir+"/Pastis/LBPp"+aImgScaledName+".dat";
    cout<<aComm<<endl;
    System(aComm);
    return aImgKey;
}

void FilterKeyPt(std::vector<Siftator::SiftPoint> aVSIFTPt, std::vector<Siftator::SiftPoint>& aVSIFTPtNew, double dMinScale, double dMaxScale)
{
    int nSizeL = aVSIFTPt.size();

    for(int i=0; i<nSizeL; i++)
    {
        if(aVSIFTPt[i].scale < dMaxScale && aVSIFTPt[i].scale > dMinScale)
            aVSIFTPtNew.push_back(aVSIFTPt[i]);
    }
}

void SetAngleToValidRange(double& dAngle, double d2PI)
{
    while(dAngle > d2PI)
        dAngle = dAngle - d2PI;
    while(dAngle < 0)
        dAngle = dAngle + d2PI;
}

bool IsHomolFileExist(std::string aDir, std::string aImg1, std::string aImg2, std::string CurSH, bool bCheckFile)
{
    std::string aSHDir;
    std::string aNewDir;
    std::string aNameFile1;
    std::string aNameFile2;
    aSHDir = aDir + "/Homol" + CurSH + "/";
    aNewDir = aSHDir + "Pastis" + aImg1;
    aNameFile1 = aNewDir + "/"+aImg2+".txt";

    aNewDir = aSHDir + "Pastis" + aImg2;
    aNameFile2 = aNewDir + "/"+aImg1+".txt";

    if (bCheckFile == true && ELISE_fp::exist_file(aNameFile1) == true && ELISE_fp::exist_file(aNameFile2) == true)
    {
        cout<<aNameFile1<<" already exist, hence skipped"<<endl;
        return true;
    }
    else
        return false;
}

std::string SaveHomolTxtFile(std::string aDir, std::string aImg1, std::string aImg2, std::string CurSH, std::vector<ElCplePtsHomologues> aPack, bool bPrintSEL)
{
    if(aPack.size() <= 0){
        //printf("No tie points.\n");
        return "";
    }
    std::string aSHDir;
    std::string aNewDir;
    std::string aNameFile1;
    std::string aNameFile2;
    aSHDir = aDir + "/Homol" + CurSH + "/";
    ELISE_fp::MkDir(aSHDir);
    aNewDir = aSHDir + "Pastis" + aImg1;
    ELISE_fp::MkDir(aNewDir);
    aNameFile1 = aNewDir + "/"+aImg2+".txt";

    aNewDir = aSHDir + "Pastis" + aImg2;
    ELISE_fp::MkDir(aNewDir);
    aNameFile2 = aNewDir + "/"+aImg1+".txt";

    std::string aCom = "mm3d SEL" + BLANK + aDir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+CurSH;
    std::string aComInv = "mm3d SEL" + BLANK + aDir + BLANK + aImg2 + BLANK + aImg1 + BLANK + "KH=NT SzW=[600,600] SH="+CurSH;
    if(bPrintSEL==true && aPack.size() >0)
        printf("%s\n%s\ntie point number: %d\n", aCom.c_str(), aComInv.c_str(), int(aPack.size()));

    FILE * fpTiePt1 = fopen(aNameFile1.c_str(), "w");
    FILE * fpTiePt2 = fopen(aNameFile2.c_str(), "w");

    bool bTiePt1 = true;
    bool bTiePt2 = true;

    if (NULL == fpTiePt1)
    {
        cout<<"Open file "<<aNameFile1<<" failed"<<endl;
        bTiePt1 = false;
    }
    if (NULL == fpTiePt2)
    {
        cout<<"Open file "<<aNameFile2<<" failed"<<endl;
        bTiePt2 = false;
    }

    for (unsigned int i=0; i<aPack.size(); i++)
    {
        ElCplePtsHomologues cple = aPack[i];
        Pt2dr aP1 = cple.P1();
        Pt2dr aP2 = cple.P2();
        if(bTiePt1)
            fprintf(fpTiePt1, "%lf %lf %lf %lf\n", aP1.x, aP1.y, aP2.x, aP2.y);
        if(bTiePt2)
            fprintf(fpTiePt2, "%lf %lf %lf %lf\n", aP2.x, aP2.y, aP1.x, aP1.y);
    }

    fclose(fpTiePt1);
    fclose(fpTiePt2);

    return aNameFile1;
}

void SaveSIFTHomolFile(std::string aDir, std::string aImg1, std::string aImg2, std::string CurSH, std::vector<Pt2di> match, std::vector<Siftator::SiftPoint> aVSiftL, std::vector<Siftator::SiftPoint> aVSiftR, bool bPrint, double dScaleL, double dScaleR, bool bSaveSclRot)
{
    std::vector<ElCplePtsHomologues> aPack;
    std::vector<ElCplePtsHomologues> aPackSclRot;

    int nTiePtNum = match.size();
    for(int i = 0; i < nTiePtNum; i++)
    {
        int idxL = match[i].x;
        int idxR = match[i].y;
        Pt2dr aP1, aP2;
        aP1 = Pt2dr(dScaleL*aVSiftL[idxL].x, dScaleL*aVSiftL[idxL].y);
        aP2 = Pt2dr(dScaleR*aVSiftR[idxR].x, dScaleR*aVSiftR[idxR].y);
        aPack.push_back(ElCplePtsHomologues(aP1,aP2));

        if(bSaveSclRot){
            aP1 = Pt2dr(aVSiftL[idxL].scale, aVSiftL[idxL].angle);
            aP2 = Pt2dr(aVSiftR[idxR].scale, aVSiftR[idxR].angle);
            aPackSclRot.push_back(ElCplePtsHomologues(aP1,aP2));
        }
        if(bPrint){
            printf("%dth match, master and secondary scale and angle: %lf %lf %lf %lf\n", i, aVSiftL[idxL].scale, aVSiftL[idxL].angle, aVSiftR[idxR].scale, aVSiftR[idxR].angle);
            printf("scaleDif, angleDif: %lf %lf\n", aVSiftR[idxR].scale/aVSiftL[idxL].scale,aVSiftR[idxR].angle-aVSiftL[idxL].angle);
        }
    }

    SaveHomolTxtFile(aDir, aImg1, aImg2, CurSH, aPack);
    if(bSaveSclRot)
        SaveHomolTxtFile(aDir, aImg1, aImg2, CurSH+"_SclRot", aPackSclRot);
}

//for the key points in one image (master or secondary image), find their nearest neighbor in another image and record it
int MatchOneWay(std::vector<int>& matchIDL, std::vector<Siftator::SiftPoint> aVSiftL, std::vector<Siftator::SiftPoint> aVSiftR, bool bRatioT, std::vector<Pt2dr> aVPredL, Pt2di ImgSzR, bool bCheckScale, bool bCheckAngle, double dSearchSpace, bool bPredict, double dScale, double dAngle, double threshScale, double threshAngle)
{
    if(dScale < 0)
        dScale = -dScale;
    int nMatches = 0;
    int nSIFT_DESCRIPTOR_SIZE = 128;
    const double d2PI = 3.1415926*2;

    int nSizeL = aVSiftL.size();
    int nSizeR = aVSiftR.size();

    int nStartL = 0;
    int nStartR = 0;
    int nEndL = nSizeL;
    int nEndR = nSizeR;

    
#if !defined (__GNUC__) || (defined (__GNUC__) && __GNUC__ > 4)
    std::time_t t1 = std::time(nullptr);
    std::cout << std::put_time(std::localtime(&t1), "%Y-%m-%d %H:%M:%S") << std::endl;
#endif
    
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

            dBoxLeft  = x - dSearchSpace;
            dBoxRight = x + dSearchSpace;
            dBoxUpper = y - dSearchSpace;
            dBoxLower = y + dSearchSpace;
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
                /*
                double dScaleDif = fabs(aVSiftR[j].scale/aVSiftL[i].scale - dScale);
                if(dScaleDif > threshScale)
                    continue;
                    */
                double dScaleRatio = aVSiftR[j].scale/aVSiftL[i].scale;
                if((dScaleRatio < dScale*(1-threshScale)) || (dScaleRatio > dScale*(1+threshScale)))
                    continue;
                //printf("%.2lf ", dScaleRatio);
            }
            if(bCheckAngle == true)
            {
                /*
                double dAngleDif = fabs(aVSiftR[j].angle-aVSiftL[i].angle - dAngle);
                if((dAngleDif > threshAngle) && (dAngleDif < d2PI-threshAngle))
                    continue;
                    */
                double dAngleDif = aVSiftR[j].angle-aVSiftL[i].angle;
                SetAngleToValidRange(dAngleDif, d2PI);
                if((dAngleDif < dAngle-threshAngle) || (dAngleDif > dAngle+threshAngle))
                    continue;
                //printf("[%.2lf] ", dAngleDif);
            }

            double dDis = 0;
            for(k=0; k<nSIFT_DESCRIPTOR_SIZE; k++)
            {
                double dDif = aVSiftL[i].descriptor[k] - aVSiftR[j].descriptor[k];
                dDis += pow(dDif, alpha);
            }
            dDis = pow(dDis, 1.0/alpha);

            //save master and secondary nearest neigbor
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
        if(nMatch != -1)
            nMatches++;
        //cout<<i<<" "<<nMatch<<endl;
    }
    #if !defined (__GNUC__) || (defined (__GNUC__) && __GNUC__ > 4)
        std::time_t t2 = std::time(nullptr);
        std::cout << std::put_time(std::localtime(&t2), "%Y-%m-%d %H:%M:%S") << std::endl;
    #endif
    return nMatches;
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

void ScaleKeyPt(std::vector<Siftator::SiftPoint>& aVSIFTPt, double dScale)
{
    int nSizeL = aVSIFTPt.size();

    for(int i=0; i<nSizeL; i++)
    {
        aVSIFTPt[i].x *= dScale;
        aVSIFTPt[i].y *= dScale;
        aVSIFTPt[i].scale *= dScale;
    }
}

int Get3DTiePt(ElPackHomologue aPackFull, cGet3Dcoor a3DCoorL, cGet3Dcoor a3DCoorR, cDSMInfo aDSMInfoL, cDSMInfo aDSMInfoR, cTransform3DHelmert aTrans3DHL, std::vector<Pt3dr>& aV1, std::vector<Pt3dr>& aV2, std::vector<Pt2dr>& a2dV1, std::vector<Pt2dr>& a2dV2, bool bPrint, bool bInverse)
{
    int nOriPtNum = 0;
    for (ElPackHomologue::iterator itCpl=aPackFull.begin(); itCpl!=aPackFull.end(); itCpl++)
    {
       ElCplePtsHomologues cple = itCpl->ToCple();
       Pt2dr p1 = cple.P1();
       Pt2dr p2 = cple.P2();

       if(bPrint)
           cout<<nOriPtNum<<"th tie pt: "<<p1.x<<" "<<p1.y<<" "<<p2.x<<" "<<p2.y<<endl;

       bool bValidL, bValidR;
       Pt3dr pTerr1 = a3DCoorL.Get3Dcoor(p1, aDSMInfoL, bValidL, bPrint);//, dGSD1);
       pTerr1 = aTrans3DHL.Transform3Dcoor(pTerr1);
       Pt3dr pTerr2 = a3DCoorR.Get3Dcoor(p2, aDSMInfoR, bValidR, bPrint);//, dGSD2);

       if(bInverse == true)
       {
           Pt3dr PtTmp = pTerr1;
           pTerr1 = pTerr2;
           pTerr2 = PtTmp;
       }

       if(bValidL == true && bValidR == true)
       {
           aV1.push_back(pTerr1);
           aV2.push_back(pTerr2);
           a2dV1.push_back(p1);
           a2dV2.push_back(p2);
           //aPackInsideBorder.Cple_Add(cple);
           //aValidPt.push_back(nOriPtNum);
       }
       else
       {
           if(false)
               cout<<nOriPtNum<<"th tie pt out of border of the DSM hence skipped"<<endl;
       }
       nOriPtNum++;
    }
    return nOriPtNum;
}

cSolBasculeRig RANSAC3DCore(int aNbTir, double threshold, std::vector<Pt3dr> aV1, std::vector<Pt3dr> aV2, std::vector<Pt2dr> a2dV1, std::vector<Pt2dr> a2dV2, std::vector<ElCplePtsHomologues>& inlierFinal)
{
    double aEpslon = 0.0000001;
    cSolBasculeRig aSBR = cSolBasculeRig::Id();
    //cSolBasculeRig aSBRBest = cSolBasculeRig::Id();
    int i, j;
    int nMaxInlier = 0;
    std::vector<ElCplePtsHomologues> inlierCur;
    std::vector<Pt3dr> aInlierCur3DL;
    std::vector<Pt3dr> aInlierCur3DR;
    std::vector<Pt3dr> aInlierFinal3DL;
    std::vector<Pt3dr> aInlierFinal3DR;
    int nPtNum = aV1.size();

    for(j=0; j<aNbTir; j++)
    {
        cRansacBasculementRigide aRBR(false);

        std::vector<int> res;

        Pt3dr aDiff;
        bool bDupPt;
        //in case duplicated points
        do
        {
            res.clear();
            bDupPt = false;
            GetRandomNum(0, nPtNum, 3, res);
            for(i=0; i<3; i++)
            {
                aDiff = aV1[res[i]] - aV1[res[(i+1)%3]];
                if((fabs(aDiff.x) < aEpslon) && (fabs(aDiff.y) < aEpslon) && (fabs(aDiff.z) < aEpslon))
                {
                    bDupPt = true;
                    //printf("Duplicated 3D pt seed: %d, %d; Original index of 2D pt: %d %d\n ", res[i], res[i+1], aValidPt[res[i]], aValidPt[res[i+1]]);
                    break;
                }
                aDiff = aV2[res[i]] - aV2[res[(i+1)%3]];
                if((fabs(aDiff.x) < aEpslon) && (fabs(aDiff.y) < aEpslon) && (fabs(aDiff.z) < aEpslon))
                {
                    bDupPt = true;
                    //printf("Duplicated 3D pt seed: %d, %d; Original index of 2D pt: %d %d\n ", res[i], res[i+1], aValidPt[res[i]], aValidPt[res[i+1]]);
                    break;
                }
            }
        }
        while(bDupPt == true);

        /*
        int aTmp[3] = {44, 49, 40};
        res[0] = aTmp[0];
        res[1] = aTmp[1];
        res[2] = aTmp[2];
        */

        for(i=0; i<3; i++)
        {
            aRBR.AddExemple(aV1[res[i]],aV2[res[i]],0,"");
            inlierCur.push_back(ElCplePtsHomologues(a2dV1[res[i]], a2dV2[res[i]]));
            aInlierCur3DL.push_back(aV1[res[i]]);
            aInlierCur3DR.push_back(aV2[res[i]]);
            /*
            printf("%dth, seed: %d; [%.2lf, %.2lf, %.2lf]; [%.2lf, %.2lf, %.2lf]\n", i,res[i], aV1[res[i]].x, aV1[res[i]].y, aV1[res[i]].z, aV2[res[i]].x, aV2[res[i]].y, aV2[res[i]].z);
            printf("[%.2lf, %.2lf]; [%.2lf, %.2lf]\n", a2dV1[res[i]].x, a2dV1[res[i]].y, a2dV2[res[i]].x, a2dV2[res[i]].y);
            */
        }

        aRBR.CloseWithTrGlob();
        aRBR.ExploreAllRansac();
        bool aSolIsInit = aRBR.SolIsInit();
        //printf("Iter: %d/%d, seed: %d, %d, %d; aSolIsInit: %d \n", j, aNbTir, res[0], res[1], res[2], aSolIsInit);
        if(aSolIsInit == 0)
            continue;
        aSBR = aRBR.BestSol();

        int nInlier =3;
        for(i=0; i<nPtNum; i++)
        {
            Pt3dr aP1 = aV1[i];
            Pt3dr aP2 = aV2[i];

            Pt3dr aP2Pred = aSBR(aP1);
            double dist = pow(pow(aP2Pred.x-aP2.x,2) + pow(aP2Pred.y-aP2.y,2) + pow(aP2Pred.z-aP2.z,2), 0.5);
            if(dist < threshold)
            {
                inlierCur.push_back(ElCplePtsHomologues(a2dV1[i], a2dV2[i]));
                aInlierCur3DL.push_back(aV1[i]);
                aInlierCur3DR.push_back(aV2[i]);
                nInlier++;
            }
        }
        if(nInlier > nMaxInlier)
        {
            nMaxInlier = nInlier;
            //aSBRBest = aSBR;
            inlierFinal = inlierCur;
            aInlierFinal3DL = aInlierCur3DL;
            aInlierFinal3DR = aInlierCur3DR;
            printf("Iter: %d/%d, seed: %d, %d, %d;  ", j, aNbTir, res[0], res[1], res[2]);
            printf("nPtNum: %d, nMaxInlier: %d; ", nPtNum, nMaxInlier);

            Pt3dr aTr = aSBR.Tr();
            double aLambda = aSBR.Lambda();
            printf("aLambda: %.2lf, aTr: [%.2lf, %.2lf, %.2lf]; \n", aLambda, aTr.x, aTr.y, aTr.z);
            //ElMatrix<double> aRot = aSBR.Rot();
        }

        inlierCur.clear();
        aInlierCur3DL.clear();
        aInlierCur3DR.clear();
    }
    printf("nMaxInlier: %d\n", int(aInlierFinal3DL.size()));
    //printf("nMaxInlier: %d\n", int(inlierFinal.size()));

    cRansacBasculementRigide aRBR(false);
    for(int i=0; i<int(aInlierFinal3DL.size()); i++)
    {
        aRBR.AddExemple(aInlierFinal3DL[i], aInlierFinal3DR[i],0,"");
    }
    aRBR.CloseWithTrGlob();
    aRBR.ExploreAllRansac();
    aSBR = aRBR.BestSol();

    return aSBR;
}

void Save3DXml(std::vector<Pt3dr> vPt3D, std::string aOutXml, std::string strPrefix)
{
    cDicoAppuisFlottant aDAFout;

    for(unsigned int i=0; i<vPt3D.size(); i++)
    {
        cOneAppuisDAF anAp;

        anAp.Pt() = vPt3D[i];
        anAp.NamePt() = strPrefix + std::to_string(i);
        anAp.Incertitude() = Pt3dr(1,1,1);
        aDAFout.OneAppuisDAF().push_back(anAp);
    }

    MakeFileXML(aDAFout, aOutXml);
}

void Get2DCoor(std::string aRGBImgDir, std::vector<string> vImgList1, std::vector<Pt3dr> vPt3DL, std::string aOri1, cInterfChantierNameManipulateur * aICNM, std::string aOut2DXml)
{
    StdCorrecNameOrient(aOri1,"./",true);
//    cout<<aOri1<<endl;

    //std::string aKeyOri1 = "NKS-Assoc-Im2Orient@-" + aOri1;

    cSetOfMesureAppuisFlottants aSOMAFout;
    for(unsigned int i=0; i<vImgList1.size(); i++)
    {
        std::string aImg1 = vImgList1[i];
//        cout<<aImg1<<endl;
        //std::string aIm1OriFile = aICNM->Assoc1To1(aKeyOri1,aImg1,true);
        std::string aIm1OriFile = aICNM->StdNameCamGenOfNames(aOri1, aImg1); //aICNM->Assoc1To1(aKeyOri1,aImg1,true);
//        cout<<aIm1OriFile<<endl;

        int aType = eTIGB_Unknown;
        cBasicGeomCap3D * aCamL = cBasicGeomCap3D::StdGetFromFile(aIm1OriFile,aType);

        Pt3dr minPt, maxPt;
        GetImgBoundingBox(aRGBImgDir, aImg1, aCamL, minPt, maxPt);

        cMesureAppuiFlottant1Im aMAF;
        aMAF.NameIm() = aImg1;
        for(unsigned int j=0; j<vPt3DL.size(); j++)
        {
            Pt3dr ptCur = vPt3DL[j];
            //if current 3d point is out of the border of the current image, skip
            //because sometimes a 3d point that is out of border will get wrong 2D point from command XYZ2Im
            if(ptCur.x<minPt.x || ptCur.y<minPt.y || ptCur.x>maxPt.x || ptCur.y>maxPt.y)
                continue;

            Pt2dr aPproj = aCamL->Ter2Capteur(ptCur);

            cOneMesureAF1I anOM;
            anOM.NamePt() = std::to_string(j);
            anOM.PtIm() = aPproj;
            aMAF.OneMesureAF1I().push_back(anOM);
        }
        aSOMAFout.MesureAppuiFlottant1Im().push_back(aMAF);
    }
    MakeFileXML(aSOMAFout, aOut2DXml);
}

bool GetImgBoundingBox(std::string aRGBImgDir, std::string aImg1, cBasicGeomCap3D * aCamL, Pt3dr& minPt, Pt3dr& maxPt)
{
    if (ELISE_fp::exist_file(aRGBImgDir+"/"+aImg1) == false)
    {
        cout<<aRGBImgDir+"/"+aImg1<<" didn't exist, hence skipped"<<endl;
        return false;
    }

    //Tiff_Im aRGBIm1((aRGBImgDir+"/"+aImg1).c_str());
    Tiff_Im aRGBIm1 = Tiff_Im::StdConvGen((aRGBImgDir+"/"+aImg1).c_str(), -1, true ,true);
    Pt2di aImgSz = aRGBIm1.sz();

    Pt2dr aPCorner[4];
    Pt2dr origin = Pt2dr(0, 0);
    aPCorner[0] = origin;
    aPCorner[1] = Pt2dr(origin.x+aImgSz.x, origin.y);
    aPCorner[2] = Pt2dr(origin.x+aImgSz.x, origin.y+aImgSz.y);
    aPCorner[3] = Pt2dr(origin.x, origin.y+aImgSz.y);

    //double prof_d = aCamL->GetVeryRoughInterProf();
    //prof_d = 11.9117;
    //double prof_d = aCamL->GetProfondeur();
    double dZ = aCamL->GetAltiSol();
    //cout<<"dZ: "<<dZ<<endl;

    Pt3dr ptTerrCorner[4];
    for(int i=0; i<4; i++)
    {
        Pt2dr aP1 = aPCorner[i];
        //ptTerrCorner[i] = aCamL->ImEtProf2Terrain(aP1, prof_d);
        ptTerrCorner[i] = aCamL->ImEtZ2Terrain(aP1, dZ);
    }

    GetBoundingBox(ptTerrCorner, 4, minPt, maxPt);

    return true;
}

void RotateImgBy90Deg(std::string aDir, std::string aImg1, std::string aNameOut)
{
    //cout<<aDir<<endl;
    cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aDir+"/"+aImg1));
    //cout<<aImg1<<endl;

    Tiff_Im::StdConvGen(aDir+"/"+aImg1,1,false,true);

    std::string aGrayImgName = aImg1 + "_Ch1.tif";
    bool bRGB = false;
    //if RGB image
    if( ELISE_fp::exist_file(aDir + "/Tmp-MM-Dir/" + aGrayImgName) == true)
    {
        bRGB = true;
        std::string aComm = "mv " + aDir + "/Tmp-MM-Dir/" + aGrayImgName + " " + aDir+"/"+aGrayImgName;
        cout<<aComm<<endl;
        System(aComm);
        aImg1 = aGrayImgName;
    }

    //Tiff_Im aDSMIm1((aDir+"/"+aImg1).c_str());
    Tiff_Im aDSMIm1 = Tiff_Im::StdConvGen((aDir+"/"+aImg1).c_str(), -1, true ,true);
    Pt2di ImgSzL = aDSMIm1.sz();

    aNameOut = aDir+"/"+aNameOut;

    //std::string aNameOut = aDir+"/"+StdPrefix(aImg1)+"_R90.tif";

    Tiff_Im TiffOut  =     Tiff_Im
                           (
                              aNameOut.c_str(),
                              Pt2di(ImgSzL.y, ImgSzL.x),
                              aDSMIm1.type_el(),
                              Tiff_Im::No_Compr,
                              aDSMIm1.phot_interp(),
                              Tiff_Im::Empty_ARG
                          );

    TIm2D<float, double> aTImProfPx(ImgSzL);
    ELISE_COPY
    (
    aTImProfPx.all_pts(),
    aDSMIm1.in(),
    aTImProfPx.out()
    );

    TIm2D<float, double> aTImProfPxTmp(Pt2di(ImgSzL.y, ImgSzL.x));
    ELISE_COPY
    (
    aTImProfPxTmp.all_pts(),
    aTImProfPx.in()[Virgule(FY,FX)],
    aTImProfPxTmp.out()
    );

    //flip
    ELISE_COPY
    (
    TiffOut.all_pts(),
    aTImProfPxTmp.in(0)[Virgule(ImgSzL.y-FX,FY)],
    TiffOut.out()
    );

    if(bRGB == true){
        std::string aComm = "rm " + aDir+"/"+aGrayImgName;
        cout<<aComm<<endl;
        System(aComm);
    }
}

void RotateImgBy90DegNTimes(std::string aDir, std::string aImg1, std::string aNameOut, int nTime)
{
    RotateImgBy90Deg(aDir, aImg1, aNameOut);
    nTime--;

    for(int i=0; i<nTime; i++)
    {
        std::string aNameOutTmp = StdPrefix(aNameOut)+"_tmp."+StdPostfix(aNameOut);
        std::string strCpImg = "mv "+aDir+aNameOut+" "+aDir+aNameOutTmp;
        //cout<<strCpImg<<endl;
        System(strCpImg);

        RotateImgBy90Deg(aDir, aNameOutTmp, aNameOut);
        strCpImg = "rm "+aDir+aNameOutTmp;
        //cout<<strCpImg<<endl;
        System(strCpImg);
    }
}

void GetUniqImgList(std::vector<std::string> aInput, std::vector<std::string>& aOutput)
{
    for(int i=0; i<int(aInput.size()); i++)
    {
        std::string aImg1 = aInput[i];
        bool bRepeat1 = false;
        for(int j=0; j<int(aOutput.size()); j++){
            if(aImg1 == aOutput[j]){
                bRepeat1 = true;
                break;
            }
        }
        if(bRepeat1 == false)
        {
            aOutput.push_back(aImg1);
        }
    }
}

std::string GetImgList(std::vector<std::string> aVIm)
{
    std::string aRes = "\"";
    for(int i=0; i<int(aVIm.size()); i++)
    {
        aRes += aVIm[i]+"|";
    }
    aRes = aRes.substr(0, aRes.length()-1) + "\"";

    return aRes;
}

std::string GetImgList(std::string aDir, std::string aFileName, bool bExe)
{
    std::string aRes;

    std::vector<std::string> aVIm;
    std::string aFileType = GetImgListVec(aDir+"/"+aFileName, aVIm, bExe);
    if(aFileType == "txt")
        aRes = GetImgList(aVIm);
    else if(aFileType == "xml")
        aRes = "NKS-Set-OfFile@" + aFileName;
    else
        aRes = aFileName;

    return aRes;
}

void ReadTfw(std::string tfwFile, std::vector<double>& aTmp)
{
    aTmp.resize(6, 0);
    if (ELISE_fp::exist_file(tfwFile) == false){
        cout<<tfwFile<<" didn't exist, hence skipped."<<endl;
        return;
    }

    ifstream in(tfwFile);
    std::string s;
    //double aTmp[6];
    int idx = 0;
    while(getline(in,s))
    {
        std::stringstream is(s);
        is>>aTmp[idx];
        idx++;
    }
}

void SaveTfw(std::string tfwFile, Pt2dr aOrthoResolPlani, Pt2dr aOrthoOriPlani)
{
    FILE * fpOutput = fopen(tfwFile.c_str(), "w");
    fprintf(fpOutput, "%lf\n0\n0\n%lf\n%lf\n%lf\n", aOrthoResolPlani.x, aOrthoResolPlani.y, aOrthoOriPlani.x, aOrthoOriPlani.y);
    fclose(fpOutput);
}

/*
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
*/

int TransmitHelmert_main(int argc,char ** argv)
{
   std::string aFile1;
   std::string aFile2;

   std::string aOutFile = "";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFile1,"First input of 3D Helmert transformation parameters (A->C)")
                    << EAMC(aFile2,"Second input of 3D Helmert transformation parameters (B->C)"),
        LArgMain()
                    << EAM(aOutFile, "OutFile", true, "Output 3D Helmert transformation parameters (A->B)")
    );

   if(aOutFile.length() == 0){
       aOutFile = "TransmittedHelmert.xml";

       string::size_type Pos11 = aFile1.find("Basc-");
       string::size_type Pos12 = aFile1.find("-2-");
       string::size_type Pos21 = aFile2.find("Basc-");
       string::size_type Pos22 = aFile2.find("-2-");
       if (Pos11 != string::npos && Pos12 != string::npos && Pos21 != string::npos && Pos22 != string::npos){
           int nStart1 = Pos11 + 5;
           int nLen1 = Pos12 - Pos11 - 5;
           int nStart2 = Pos21 + 5;
           int nLen2 = Pos22 - Pos21 - 5;
           aOutFile = "Basc-" + aFile1.substr(nStart1, nLen1) + "-2-" + aFile2.substr(nStart2, nLen2) + ".xml";
           //printf("%d, %d, %d, %d. %s\n", nStart1, nLen1, nStart2, nLen2, aOutFile.c_str());
       }
   }

   cXml_ParamBascRigide  * aTransf1 = OptStdGetFromPCP(aFile1, Xml_ParamBascRigide);
   cSolBasculeRig aSBR1 = Xml2EL(*aTransf1);

   cXml_ParamBascRigide  * aTransf2 = OptStdGetFromPCP(aFile2, Xml_ParamBascRigide);
   cSolBasculeRig aSBR2 = Xml2EL(*aTransf2);
   cSolBasculeRig aSBR2Inv = aSBR2.Inv();

   Pt3dr aTr;
   //ElMatrix<double> aRot;
   double aLambda;

   aTr = aSBR2Inv(aSBR1.Tr());
   aLambda = aSBR1.Lambda()*aSBR2Inv.Lambda();
   ElMatrix<double> aRot = aSBR2Inv.Rot()*aSBR1.Rot();

   cSolBasculeRig aSBROut = cSolBasculeRig(Pt3dr(0,0,0),aTr,aRot,aLambda);

   MakeFileXML(EL2Xml(aSBROut), aOutFile);
   cout<<"xdg-open "<<aOutFile<<endl;
    /*
   Pt3dr aSrc = Pt3dr(100, 200, 300);
   Pt3dr aDes = aSBR1(aSrc);
   printf("Pred1: [%.2lf, %.2lf, %.2lf]\n", aDes.x, aDes.y, aDes.z);
   aDes = aSBR2Inv(aDes);
   printf("Pred2: [%.2lf, %.2lf, %.2lf]\n", aDes.x, aDes.y, aDes.z);
   aDes = aSBROut(aSrc);
   printf("Pred22: [%.2lf, %.2lf, %.2lf]\n", aDes.x, aDes.y, aDes.z);
    */

   return EXIT_SUCCESS;
}

void WriteXml(std::string aImg1, std::string aImg2, std::string aSubPatchXml, std::vector<std::string> vPatchesL, std::vector<std::string> vPatchesR, std::vector<cElHomographie> vHomoL, std::vector<cElHomographie> vHomoR, bool bPrint)
{
    //cout<<aSubPatchXml<<endl;
    cSetOfPatches aDAFout;

    cMes1Im aIms1;
    cMes1Im aIms2;

    aIms1.NameIm() = GetFileName(aImg1);
    aIms2.NameIm() = GetFileName(aImg2);

    //cout<<aIms1.NameIm()<<endl;

    int nPatchNumL = vPatchesL.size();
    int nPatchNumR = vPatchesR.size();
    for(int i=0; i < nPatchNumL; i++)
    {
        //cout<<i<<"/"<<nPatchNum<<endl;
        cOnePatch1I patch1;
        patch1.NamePatch() = vPatchesL[i];
        patch1.PatchH() = vHomoL[i].ToXml();
        aIms1.OnePatch1I().push_back(patch1);

        if(bPrint)
        {
            printf("%d, %s: \n", i, vPatchesL[i].c_str());
            vHomoL[i].Show();
        }
    }

    for(int i=0; i < nPatchNumR; i++)
    {
        cOnePatch1I patch2;
        patch2.NamePatch() = vPatchesR[i];
        patch2.PatchH() = vHomoR[i].ToXml();
        aIms2.OnePatch1I().push_back(patch2);

        if(bPrint)
        {
            printf("%d, %s: \n", i, vPatchesR[i].c_str());
            vHomoR[i].Show();
        }
    }

    aDAFout.Mes1Im().push_back(aIms1);
    aDAFout.Mes1Im().push_back(aIms2);

    MakeFileXML(aDAFout, aSubPatchXml);
}

std::string GetFileName(std::string strIn)
{
    std::string strOut = strIn;

    std::size_t found = strIn.find("/");
    if (found!=std::string::npos)
        strOut = strIn.substr(found+1, strIn.length());

    return strOut;
}
