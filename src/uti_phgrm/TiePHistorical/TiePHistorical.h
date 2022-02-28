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


#ifndef _TiePHisto_
#define _TiePHisto_

#include "StdAfx.h"
#include <algorithm>




class cAppliTiepHistoricalPipeline; // class managing the pipeline
class cCommonAppliTiepHistorical; // class managing parameters common to all stages of the pipeline

/****************************************/
/****** cCommonAppliTiepHistorical ******/
/****************************************/
/*
typedef enum
{
  e2D,
  e3D,
  eNbTypeRHP
} eRANSAC_HistoPipe;
*/
class cCommonAppliTiepHistorical
{
    public:

        cCommonAppliTiepHistorical();

        /* Common parameters */
//        bool                              mExe;
        bool                              mPrint;
        std::string                       mDir;

        //std::string mDir;
        std::string mPat;
        //std::string mOri;

        /* Parameters for rough co-registration */
        std::string                       mOriIn1;
        std::string                       mOriIn2;
//        std::string                       mDSMDirL;
//        std::string                       mDSMDirR;
        std::string                       mOriOut;

        /* Parameters for rough DSM_Equalization */
        //std::string                       mOutImg;
        double                            mSTDRange;

        /* Parameters for SuperGlue */
        //std::string                       input_pairs;
        std::string                       mInput_dir;
//        std::string                       mOutput_dir;
        std::string                       mSpGlueOutSH;
        Pt2di                             mResize;
        std::string                       mModel;
        int                               mMax_keypoints;
        bool                              mViz;
        bool                              mKeepNpzFile;
        std::string                       mStrEntSpG;
        std::string                       mStrOpt;

        /* Parameters for GetPatchPair */ 
        //Pt2dr                             mPatchSz;
        //Pt2dr                             mBufferSz;
        std::string                       mSubPatchXml;
        std::string                       mImgPair;
//        std::string                       mOutDir;
//        std::string                       mOutImg1;
//        std::string                       mOutImg2;

        /* Parameters for MergeTiePt */   
        std::string                       mMergeTiePtInSH;
        std::string                       mMergeTiePtOutSH;
        std::string                       mHomoXml;

        /* Parameters for 2DRANSAC */
        std::string                       mR2DInSH;
        std::string                       mR2DOutSH;
        int                               mR2DIteration;
        double                            mR2DThreshold;

        /* Parameters for 3DRANSAC */
        std::string                       mR3DInSH;
        std::string                       mR3DOutSH;
        int                               mR3DIteration;
        double                            mR3DThreshold;
        int                               mMinPt;
//        std::string                       mDSMFileL;
//        std::string                       mDSMFileR;

        /* Parameters for CreateGCPs */
        std::string                       mCreateGCPsInSH;
        std::string                       mOut2DXml1;
        std::string                       mOut2DXml2;
        std::string                       mOut3DXml1;
        std::string                       mOut3DXml2;

        /* Parameters for GetOverlappedImages */
        std::string                       mOutPairXml;

        /* Parameters for GuidedSIFT */   
        std::string                       mGuidedSIFTOutSH;
        bool                              mSkipSIFT;
        double                            mSearchSpace;
        bool                              mMutualNN;
        bool                              mRatioT;
        bool                              mRootSift;
        bool                              mCheckScale;
        bool                              mCheckAngle;
        bool                              mPredict;
        double                            mScale;
        double                            mAngle;
        double                            mThreshScale;
        double                            mThreshAngle;

        /* Parameters for CrossCorrelation */
        std::string                       mCrossCorrelationInSH;
        std::string                       mCrossCorrelationOutSH;
        int                               mWindowSize;
        double                            mCrossCorrThreshold;

        LArgMain &     ArgBasic();
        LArgMain &     ArgRough();
        LArgMain &     ArgSuperGlue();
        LArgMain &     ArgMergeTiePt();
        LArgMain &     ArgGetPatchPair();
        LArgMain &     ArgCreateGCPs();
        LArgMain &     ArgGetOverlappedImages();
        LArgMain &     Arg2DRANSAC();
        LArgMain &     Arg3DRANSAC();
        LArgMain &     ArgGuidedSIFT();
        LArgMain &     ArgDSM_Equalization();
        LArgMain &     ArgCrossCorrelation();

        cInterfChantierNameManipulateur * mICNM;

        void CorrectXmlFileName(std::string aCreateGCPsInSH, std::string aOri1, std::string aOri2);

        std::string GetFolderName(std::string strIn);

        std::string ComParamDSM_Equalization();
        std::string ComParamGetPatchPair();
        std::string ComParamSuperGlue();
        std::string ComParamMergeTiePt();
        std::string ComParamRANSAC2D();
        std::string ComParamRANSAC3D();
        std::string ComParamCreateGCPs();
        std::string ComParamGetOverlappedImages();
        std::string ComParamGuidedSIFTMatch();


private:
    LArgMain * mArgBasic;
    LArgMain * mArgRough;
    LArgMain * mArgSuperGlue;
    LArgMain * mArgMergeTiePt;
    LArgMain * mArgGetPatchPair;
    LArgMain * mArgGetOverlappedImages;
    LArgMain * mArg2DRANSAC;
    LArgMain * mArg3DRANSAC;
    LArgMain * mArgGuidedSIFT;
    LArgMain * mArgDSM_Equalization;
    LArgMain * mArgCrossCorrelation;
    LArgMain * mArgCreateGCPs;
    /*

            LArgMain * mArgRPC;
            LArgMain * mArgMM1P;
    LArgMain * mArgFuse;
    */
};


/*******************************************/
/****** cTransform3DHelmert  ******/
/*******************************************/
class cTransform3DHelmert
{
    public:

        cTransform3DHelmert(std::string aFileName);

        cSolBasculeRig GetSBR();
        cSolBasculeRig GetSBRInv();

        Pt3dr Transform3Dcoor(Pt3dr aPt);
        double GetScale();
        bool GetApplyTrans();

private:
        bool mApplyTrans;
        cXml_ParamBascRigide  *  mTransf;
        double mScl;
        Pt3dr mTr;

        cSolBasculeRig mSBR;
        cSolBasculeRig mSBRInv;
        //cTypeCodageMatr mRot;

};


/*******************************************/
/****** cDSMInfo  ******/
/*******************************************/

class cDSMInfo
{
    public:

        cDSMInfo(Pt2di aDSMSz, std::string aDSMFile, std::string aDSMDir);

        Pt2dr Get2DcoorInDSM(Pt3dr aTer);

        static Pt2di GetDSMSz(std::string aDSMFile, std::string aDSMDir);
        std::string GetDSMName(std::string aDSMFile, std::string aDSMDir);

        double GetDSMValue(Pt2di aPt2);
        double GetMasqValue(Pt2di aPt2);

        Pt2dr GetOriPlani();
        Pt2dr GetResolPlani();
        Pt2di GetDSMSz();

        bool GetIfDSMIsValid();

private:
        bool         bDSM;
        Pt2di        mDSMSz;
        cFileOriMnt  mFOM;
        Pt2dr mOriPlani;
        Pt2dr mResolPlani;

        std::string mDSMName;
        std::string mMaskName;

        TIm2D<float,double> mTImDSM;
        TIm2D<float,double> mTImMask;
};

/*******************************************/
/****** cGet3Dcoor  ******/
/*******************************************/

class cGet3Dcoor
{
    public:

        cGet3Dcoor(std::string aNameOri);

        double GetGSD();

        //TIm2D<float,double> SetDSMInfo(std::string aDSMFile, std::string aDSMDir);

        cDSMInfo SetDSMInfo(std::string aDSMFile, std::string aDSMDir);

        //Pt2di GetDSMSz(std::string aDSMFile, std::string aDSMDir);

        Pt3dr Get3Dcoor(Pt2dr aPt1, cDSMInfo aDSMInfo, bool& bValid, bool bPrint = false, double dThres = 2);

        Pt3dr GetRough3Dcoor(Pt2dr aPt1);

        Pt2dr Get2Dcoor(Pt3dr aTer);

        //Pt2dr Get2DcoorInDSM(Pt3dr aTer);

        //std::string GetDSMName(std::string aDSMFile, std::string aDSMDir);

private:
        cBasicGeomCap3D * mCam1;
        bool         bDSM;
        double mZ;
        /*Pt2di        mDSMSz;
        cFileOriMnt  mFOM;
        Pt2dr mOriPlani;
        Pt2dr mResolPlani;*/
        //cDSMInfo mDSMInfo;
        //Im2D<float,double>   mImIn;
        //TIm2D<float,double> mTImProfPx;
};

/*******************************************/
/****** cAppliTiepHistoricalPipeline  ******/
/*******************************************/


class cAppliTiepHistoricalPipeline : cCommonAppliTiepHistorical
{
    public:
        cAppliTiepHistoricalPipeline(int argc, char** argv);

        void DoAll();

    private:
        std::string GetImage_Profondeur(std::string aDSMDir, std::string aDSMFile);
//        std::string GetImgList(std::string aDir, std::string aFileName, bool bExe);


        bool        mDebug;
        cCommonAppliTiepHistorical mCAS3D;

        std::string mFeature;

        std::string mDSMDirL;
        std::string mDSMDirR;
        std::string mDSMFileL;
        std::string mDSMFileR;

        std::string mOrthoDirL;
        std::string mOrthoDirR;
        std::string mOrthoFileL;
        std::string mOrthoFileR;

        std::string mOri1;
        std::string mOri2;
        std::string mImgList1;
        std::string mImgList2;
        std::string mImg4MatchList1;
        std::string mImg4MatchList2;

        //std::string mCoRegOri;
        std::string mCoRegOri1;
        std::string mPara3DH;

        ElTimer     mChrono;

        bool mSkipCoReg;
        bool mSkipPrecise;
        bool mSkipGetOverlappedImages;
        bool mSkipGetPatchPair;
        bool mSkipTentativeMatch;
        bool mSkipRANSAC3D;
        bool mSkipCrossCorr;

        Pt2dr mCoRegPatchLSz;
        Pt2dr mCoRegBufferLSz;
        Pt2dr mCoRegPatchRSz;
        Pt2dr mCoRegBufferRSz;

        Pt2dr mPrecisePatchSz;
        Pt2dr mPreciseBufferSz;

        double mDyn;
        double mScaleL;
        double mScaleR;

        double mReprojTh;

        bool                              mExe;
        bool                              mUseDepth;
        bool                              mCheckFile;
        int                               mRotateDSM;
        double                            mCheckNbCoReg;
        double                            mCheckNbPrecise;
        /*


        std::string mPat;
        std::string mOri;



        ElTimer     mChrono;
        */
};


bool FallInBox(Pt2dr* aPCorner, Pt2dr aLeftTop, Pt2di aRightLower);
void GetRandomNum(int nMin, int nMax, int nNum, std::vector<int> & res);
void GetRandomNum(double dMin, double dMax, int nNum, std::vector<double> & res);
bool GetImgListVec(std::string aFullPattern, std::vector<std::string>& aVIm, bool bPrint=true);
void ReadXml(std::string & aImg1, std::string & aImg2, std::string aSubPatchXml, std::vector<std::string>& vPatchesL, std::vector<std::string>& vPatchesR, std::vector<cElHomographie>& vHomoL, std::vector<cElHomographie>& vHomoR);
void GetBoundingBox(Pt3dr* ptTerrCorner, int nLen, Pt3dr& minPt, Pt3dr& maxPt);
bool CheckRange(int nMin, int nMax, double & value);
std::string GetScaledImgName(std::string aImgName, Pt2di ImgSz, double dScale);
std::string ExtractSIFT(std::string aImgName, std::string aDir, double dScale=1);
int GetTiePtNum(std::string aDir, std::string aImg1, std::string aImg2, std::string aSH);
std::string StdCom(const std::string & aCom,const std::string & aPost="", bool aExe=false);
int SaveTxtImgPair(std::string aName, std::vector<std::string> aResL, std::vector<std::string> aResR);
int SaveXmlImgPair(std::string aName, std::vector<std::string> aResL, std::vector<std::string> aResR);
int GetXmlImgPair(std::string aName, std::vector<std::string>& aResL, std::vector<std::string>& aResR);
void FilterKeyPt(std::vector<Siftator::SiftPoint> aVSIFTPt, std::vector<Siftator::SiftPoint>& aVSIFTPtNew, double dMinScale, double dMaxScale=DBL_MAX);
int MatchOneWay(std::vector<int>& matchIDL, std::vector<Siftator::SiftPoint> aVSiftL, std::vector<Siftator::SiftPoint> aVSiftR, bool bRatioT, std::vector<Pt2dr> aVPredL=std::vector<Pt2dr>(), Pt2di ImgSzR=Pt2di(0,0), bool bCheckScale=0, bool bCheckAngle=0, double dSearchSpace=0, bool bPredict=0, double dScale=0, double dAngle=0, double threshScale=0, double threshAngle=0);
void MutualNearestNeighbor(bool bMutualNN, std::vector<int> matchIDL, std::vector<int> matchIDR, std::vector<Pt2di> & match);
void SetAngleToValidRange(double& dAngle, double d2PI);
void SaveSIFTHomolFile(std::string aDir, std::string aImg1, std::string aImg2, std::string CurSH, std::vector<Pt2di> match, std::vector<Siftator::SiftPoint> aVSiftL, std::vector<Siftator::SiftPoint> aVSiftR, bool bPrint=0, double dScaleL=1, double dScaleR=1, bool bSaveSclRot=false);
void SaveHomolTxtFile(std::string aDir, std::string aImg1, std::string aImg2, std::string CurSH, std::vector<ElCplePtsHomologues> aPack, bool bPrintSEL=true);
bool IsHomolFileExist(std::string aDir, std::string aImg1, std::string aImg2, std::string CurSH, bool bCheckFile);
void ScaleKeyPt(std::vector<Siftator::SiftPoint>& aVSIFTPt, double dScale);
int Get3DTiePt(ElPackHomologue aPackFull, cGet3Dcoor a3DCoorL, cGet3Dcoor a3DCoorR, cDSMInfo aDSMInfoL, cDSMInfo aDSMInfoR, cTransform3DHelmert aTrans3DHL, std::vector<Pt3dr>& aV1, std::vector<Pt3dr>& aV2, std::vector<Pt2dr>& a2dV1, std::vector<Pt2dr>& a2dV2, bool bPrint, bool bInverse=0);
cSolBasculeRig RANSAC3DCore(int aNbTir, double threshold, std::vector<Pt3dr> aV1, std::vector<Pt3dr> aV2, std::vector<Pt2dr> a2dV1, std::vector<Pt2dr> a2dV2, std::vector<ElCplePtsHomologues>& inlierFinal);
void Save3DXml(std::vector<Pt3dr> vPt3D, std::string aOutXml);
void Get2DCoor(std::string aRGBImgDir, std::vector<string> vImgList1, std::vector<Pt3dr> vPt3DL, std::string aOri1, cInterfChantierNameManipulateur * aICNM, std::string aOut2DXml);
bool GetImgBoundingBox(std::string aRGBImgDir, std::string aImg1, cBasicGeomCap3D * aCamL, Pt3dr& minPt, Pt3dr& maxPt);
void RotateImgBy90Deg(std::string aDir, std::string aImg1, std::string aNameOut);
void RotateImgBy90DegNTimes(std::string aDir, std::string aImg1, std::string aNameOut, int nTime);
std::string GetImgList(std::vector<std::string> aVIm);
std::string GetImgList(std::string aDir, std::string aFileName, bool bExe);
void GetUniqImgList(std::vector<std::string> aInput, std::vector<std::string>& aOutput);
void ReadTfw(std::string tfwFile, std::vector<double>& aTmp);
void SaveTfw(std::string tfwFile, Pt2dr aOrthoResolPlani, Pt2dr aOrthoOriPlani);
std::string RemoveOri(std::string aOri);
void WriteXml(std::string aImg1, std::string aImg2, std::string aSubPatchXml, std::vector<std::string> vPatchesL, std::vector<std::string> vPatchesR, std::vector<cElHomographie> vHomoL, std::vector<cElHomographie> vHomoR, bool bPrint);
std::string GetFileName(std::string strIn);

/****************************************/
/****** cInterEp_RoughCoReg ******/
/****************************************/
/*
class cInterEp_RoughCoReg
{
    public:

        cInterEp_RoughCoReg();

        void DSM2Gray();

private:

};
*/
#endif //  _TiePHisto_
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

