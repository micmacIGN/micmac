#ifndef SIMUROLSHUT_H
#define SIMUROLSHUT_H

#include "StdAfx.h"
#include <unordered_map>
#include "../../uti_files/CPP_XifGps2Xml.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"

typedef pair<std::string,std::string> Key; // map key <ImName,PtName> -> PtIm

class cAppli_CamXifDate;

class cIm_CamXifDate : public cIm_XifDate // Cam + exif time info of one image
{
    public:
        cIm_CamXifDate(cInterfChantierNameManipulateur* aICNM, const std::string & aName, const std::string & aOri, cElHour & aBeginTime, const Pt3d<double> &aVitesse);

        CamStenope * mCam;
        Pt3dr        mVitesse;
};

class cAppli_CamXifDate : public cAppli_XifDate // Cam + exif time info for a set of cIm_CamXifDate
{
    public:
        cAppli_CamXifDate(const std::string & aFullName, std::string & aOri, const std::string &aCalcV);

        std::unordered_map<std::string,cIm_CamXifDate> mVIm;
        std::string                          mOri;
};

class cSetTiePMul_Cam
{
    public:
        cSetTiePMul_Cam(const std::string &aSH,const cAppli_CamXifDate & anAppli);
        void ReechRS_SH(const double & aRSSpeed, const std::string &aSHOut);

    private:
        cSetTiePMul *             m_pSH;
        std::string               mSH;
        const cAppli_CamXifDate   m_Appli;
};

class cPtIm_CamXifDate
{
    public:
        cPtIm_CamXifDate(Pt2dr &aPtIm, cIm_CamXifDate &aIm_CamXifDate);

        Pt2dr mPtIm;
        const cIm_CamXifDate mIm_CamXifDate;
};

class cSetOfMesureAppuisFlottants_Cam
{
    public:
        cSetOfMesureAppuisFlottants_Cam(const std::string &aMAFIn,const cAppli_CamXifDate & anAppli);
        std::map<Key, Pt2dr> ReechRS_MAF(const double aRSSpeed); //<pair<ImName,PtName>,Pt2dr>
        void Export_MAF(const std::string & aMAFOut, const std::map<Key,Pt2dr> &aMap);
    private:
        const cAppli_CamXifDate   m_Appli;
        cSetOfMesureAppuisFlottants mDico;
        std::map<std::string,std::vector<cPtIm_CamXifDate>> mVPtIm; //<PtName,std::vector<cPtIm_CamXifDate>>
};


#endif // SIMUROLSHUT_H

