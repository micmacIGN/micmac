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
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>


#include "string.h"

#include "../schnaps.h"
#include "SimuBBA.h"
#include "SimuRolShut.h"


Pt2dr Reproj (CamStenope * aCam, const Pt3dr & aP3D, const Pt2dr & aP2D, const Pt3dr & aEcartCamCenter)
{
    Pt2dr aReprojP2D = aCam->R3toF2(aP3D);
    aCam->AddToCenterOptical(-aEcartCamCenter);
    Pt2dr aNewReprojP2D = aCam->R3toF2(aP3D);
    Pt2dr aNewP2D = aP2D + aNewReprojP2D - aReprojP2D;
    return aNewP2D;
}


/*******************************************************************/
/*                                                                 */
/*                  cIm_CamXifDate                                 */
/*                                                                 */
/*******************************************************************/

cIm_CamXifDate::cIm_CamXifDate(cInterfChantierNameManipulateur * aICNM, const string &aName, const string &aOri,cElHour & aBeginTime, const Pt3dr & aVitesse):
    cIm_XifDate(aName,aBeginTime),
    mCam(aICNM->StdCamStenOfNames(aName,aOri)),
    mVitesse(aVitesse)
{}


/*******************************************************************/
/*                                                                 */
/*                  cAppli_CamXifDate                              */
/*                                                                 */
/*******************************************************************/

cAppli_CamXifDate::cAppli_CamXifDate(const string &aFullName, string &aOri, const std::string & aCalcV):
    cAppli_XifDate(aFullName)
{
    StdCorrecNameOrient(aOri,mDir,true);
    mOri = aOri;

    // read velocity file
    std::ifstream aFile((mDir+aCalcV).c_str());

    std::map<string,Pt3dr> aMapVitesse;

    if(aFile)
    {
        //#F=Im_xy_t_Vxy_h_X_Y_Z_Vx_Vy_Vz
        std::string aNameIm;
        double xy,t,Vxy,h,X,Y,Z,aVX,aVY,aVZ;

        while(aFile >> aNameIm >> xy >> t >> Vxy >> h >> X >> Y >> Z >> aVX >> aVY >> aVZ)
        {
            aMapVitesse.insert(pair<string,Pt3dr>(aNameIm,Pt3dr(aVX,aVY,aVZ)));
        }
        aFile.close();
    }
    std::cout << "aCalcV : " << aMapVitesse.size() << " files" << endl;


    for(int i=0; i<int(mSetIm->size());i++)
    {
        mVIm.insert(std::pair<std::string,cIm_CamXifDate>(mSetIm->at(i),cIm_CamXifDate(mICNM,mSetIm->at(i),aOri,mBegin,aMapVitesse.at(mSetIm->at(i)))));
    }

    std::cout << "Nb of files:" << mVIm.size() << endl;

    ELISE_ASSERT(aMapVitesse.size()==mVIm.size(),"Nb of Im != Nb of Im velocity");
}




/*******************************************************************/
/*                                                                 */
/*                      cSetTiePMul_Cam                            */
/*                                                                 */
/*******************************************************************/
cSetTiePMul_Cam::cSetTiePMul_Cam(const std::string &aSH, const cAppli_CamXifDate &anAppli):
    m_pSH(new cSetTiePMul(0)),
    mSH(aSH),
    m_Appli(anAppli)
{
    m_pSH->AddFile(mSH);
}

void cSetTiePMul_Cam::ReechRS_SH(const double &aRSSpeed, const string &aSHOut)
{
    std::vector<cSetPMul1ConfigTPM *> aVCnf = m_pSH->VPMul();
    for(uint itCnf=0; itCnf<aVCnf.size(); itCnf++)
    {
        std::cout << "Done " << itCnf << " out of " << aVCnf.size() << endl;
        auto aCnf = aVCnf.at(itCnf);
        const std::vector<int> & aVIdIm = aCnf->VIdIm();
        std::vector<CamStenope*> aVCam;
        for(const int &aIdIm : aVIdIm)
        {
            std::string aNameIm = m_pSH->NameFromId(aIdIm);
            aVCam.push_back(m_Appli.mVIm.at(aNameIm).mCam);
        }

        for(int aKPt=0; aKPt<aCnf->NbPts(); aKPt++)
        {
            std::vector<Pt2dr> aVOldP2D;
            for(int aKIm=0; aKIm<aCnf->NbIm(); aKIm++)
            {
                aVOldP2D.push_back(aCnf->Pt(aKPt,aKIm));
            }
            ELISE_ASSERT(aVOldP2D.size() == aVCam.size(), "Size not coherent");
            ELISE_ASSERT(aVOldP2D.size() > 1 && aVCam.size() > 1, "Nb faiseaux < 2");
            Pt3dr aP3D = Intersect_Simple(aVCam , aVOldP2D);

            for(int aKIm=0; aKIm<aCnf->NbIm(); aKIm++)
            {
                std::string aNameIm = m_pSH->NameFromId(aCnf->VIdIm().at(aKIm));
                CamStenope * aCam = aVCam.at(aKIm);
                Pt2dr aOldP2D = aVOldP2D.at(aKIm);

                double aEcartTime = (aOldP2D.y-aCam->Sz().y/2) * aRSSpeed/1000/1000;
                Pt3dr aEcartCenter = m_Appli.mVIm.at(aNameIm).mVitesse * aEcartTime;

                Pt2dr aNewP2D = Reproj(aCam, aP3D, aOldP2D, aEcartCenter);

                if(IsInImage(aCam->Sz(),aNewP2D))
                    aCnf->SetPt(aKPt,aKIm,aNewP2D);
            }
        }
    }
    // output modified tie points
    std::string aNameOut0 = cSetTiePMul::StdName(m_Appli.mICNM,aSHOut,"Reech",0);
    std::string aNameOut1 = cSetTiePMul::StdName(m_Appli.mICNM,aSHOut,"Reech",1);

    m_pSH->Save(aNameOut0);
    m_pSH->Save(aNameOut1);
}


/*******************************************************************/
/*                                                                 */
/*                      cPtIm_CamXifDate                           */
/*                                                                 */
/*******************************************************************/
cPtIm_CamXifDate::cPtIm_CamXifDate(Pt2dr &aPtIm, cIm_CamXifDate &aIm_CamXifDate):
    mPtIm(aPtIm),
    mIm_CamXifDate(aIm_CamXifDate)
{}

/*******************************************************************/
/*                                                                 */
/*                cSetOfMesureAppuisFlottants_Cam                  */
/*                                                                 */
/*******************************************************************/
cSetOfMesureAppuisFlottants_Cam::cSetOfMesureAppuisFlottants_Cam(const std::string &aMAFIn,const cAppli_CamXifDate & anAppli):
    m_Appli(anAppli),
    mDico(StdGetFromPCP(aMAFIn,SetOfMesureAppuisFlottants))
{

    std::list<cMesureAppuiFlottant1Im> & aLMAF = mDico.MesureAppuiFlottant1Im();
    for(auto &aMAF : aLMAF)
    {
        const std::string & aNameIm = aMAF.NameIm();
        std::cout << aNameIm << endl;
        std::list<cOneMesureAF1I> & aLMes = aMAF.OneMesureAF1I();
        for(auto & aMes:aLMes)
        {
            const std::string & aNamePt = aMes.NamePt();
            Pt2dr aPtIm = aMes.PtIm();
            auto search = mVPtIm.find(aNamePt);
            cIm_CamXifDate aIm_XifDate = m_Appli.mVIm.at(aNameIm);
            cPtIm_CamXifDate aPtIm_CamXifDate(aPtIm,aIm_XifDate);

            if(search == mVPtIm.end())
            {
                std::vector<cPtIm_CamXifDate> aVPtIm_CamXifDate;
                aVPtIm_CamXifDate.push_back(aPtIm_CamXifDate);
                mVPtIm.insert(pair<std::string,std::vector<cPtIm_CamXifDate>>(aNamePt,aVPtIm_CamXifDate));
            }
            else
            {
                mVPtIm.at(aNamePt).push_back(aPtIm_CamXifDate);
            }
        }
    }
}

std::map<Key,Pt2dr> cSetOfMesureAppuisFlottants_Cam::ReechRS_MAF(const double aRSSpeed)
{
    std::map<Key,Pt2dr> aMap;
    int i=0, j=0;
    for(auto &aPtIm:mVPtIm)
    {
        std::vector<Pt2dr> aVOldP2D;
        std::vector<CamStenope*> aVCam;
        std::string aPtName = aPtIm.first;
        std::cout << aPtName << endl;
        std::vector<cPtIm_CamXifDate> aVPtIm_CamXifDate = aPtIm.second;
        for(auto &aPtIm_CamXifDate:aVPtIm_CamXifDate)
        {
            aVOldP2D.push_back(aPtIm_CamXifDate.mPtIm);
            aVCam.push_back(aPtIm_CamXifDate.mIm_CamXifDate.mCam);
        }
        ELISE_ASSERT(aVOldP2D.size() == aVCam.size(), "Size not coherent");
        ELISE_ASSERT(aVOldP2D.size() > 1 && aVCam.size() > 1, "Nb faiseaux < 2");

        Pt3dr aP3D = Intersect_Simple(aVCam , aVOldP2D);


        for(auto &aPtIm_CamXifDate:aVPtIm_CamXifDate)
        {
            std::string aImName = aPtIm_CamXifDate.mIm_CamXifDate.mName;
            CamStenope * aCam = aPtIm_CamXifDate.mIm_CamXifDate.mCam;

            double aEcartTime = (aPtIm_CamXifDate.mPtIm.y-aCam->Sz().y/2) * aRSSpeed/1000/1000;
            Pt3dr aEcartCenter = m_Appli.mVIm.at(aImName).mVitesse * aEcartTime;
            Pt2dr aNewP2D = Reproj(aCam, aP3D, aPtIm_CamXifDate.mPtIm, aEcartCenter);

            Key aKey = pair<string,string>(aImName,aPtName);

            if(IsInImage(aCam->Sz(),aNewP2D))
            { 
                aMap.insert(pair<Key,Pt2dr>(aKey,aNewP2D));
                i++;
            }
            else
            {
                aMap.insert(pair<Key,Pt2dr>(aKey,aPtIm_CamXifDate.mPtIm));
                j++;
            }
        }
    }

    std::cout << j << "/" << i+j << " points are not corrected!" << endl;
    return aMap;

}

void cSetOfMesureAppuisFlottants_Cam::Export_MAF(const std::string & aMAFOut, const std::map<Key,Pt2dr> & aMap)
{
    std::list<cMesureAppuiFlottant1Im> & aLMAF = mDico.MesureAppuiFlottant1Im();
    for(auto &aMAF:aLMAF)
    {
        const std::string aNameIm = aMAF.NameIm();
        std::list<cOneMesureAF1I> & aLMes = aMAF.OneMesureAF1I();
        for(auto & aMes:aLMes)
        {
            const std::string aNamePt = aMes.NamePt();
            Key aKey = pair<string,string>(aNameIm,aNamePt);
            aMes.PtIm() = aMap.at(aKey);
        }
    }
    MakeFileXML(mDico,aMAFOut);
}


/*******************************************************************/
/*                                                                 */
/*                                                                 */
/*                                                                 */
/*******************************************************************/

int SimuRolShut_main(int argc, char ** argv)
{
    std::string aPatImgs, aSH, aOri, aOriRS, aSHOut{"SimuRolShut"}, aDir, aImgs,aModifP;
    int aLine{3}, aSeed;
    std::vector<double> aNoiseGaussian(4,0.0);
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image Pattern",eSAM_IsExistFile)
                           << EAMC(aSH, "PMul File",  eSAM_IsExistFile)
                           << EAMC(aOri, "Ori",  eSAM_IsExistDirOri)
                           << EAMC(aOriRS,"Ori for modified ori files")
                           << EAMC(aModifP,"File containing pose modification for each image, file size = 1 or # of images"),
                LArgMain() << EAM(aSHOut,"Out",false,"Output name of generated tie points, default=simulated")
                           << EAM(aLine,"Line",true,"Read file containing pose modification from a certain line, def=3 (two lines for file header)")
                           << EAM(aSeed,"Seed",false,"Seed for generating gaussian noise")
                           << EAM(aNoiseGaussian,"NoiseGaussian",false,"[meanX,stdX,meanY,stdY]")
                );

    // get directory
    SplitDirAndFile(aDir,aImgs,aPatImgs);
    StdCorrecNameOrient(aOri, aDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImgs = *(aICNM->Get(aImgs));

    std::cout << aVImgs.size() << " image files.\n";

    // Generate poses corresponding to the end of exposure
    std::vector<Orientation> aVOrient;
    ReadModif(aVOrient,aModifP,aLine);

    // Copy Calib file
    cElFilename aCal = cElFilename(aDir,aICNM->StdNameCalib(aOri,aVImgs[0]));
    std::string aDirCalib, aCalib;
    SplitDirAndFile(aDirCalib,aCalib,aCal.m_basename);
    std::string aNewCalib = aDir+"Ori-"+aOriRS+"/"+aCalib;
    cElFilename aCalRS = cElFilename(aNewCalib);

    for(uint i=0; i<aVImgs.size();i++)
    {
        CamStenope * aCam = aICNM->StdCamStenOfNames(aVImgs[i],aOri);
        uint j = (aVOrient.size()==1)? 0 : i;
        aCam->AddToCenterOptical(aVOrient.at(j).Translation);
        aCam->MultiToRotation(aVOrient.at(j).Rotation);

        std::string aKeyOut = "NKS-Assoc-Im2Orient@-" + aOriRS;
        std::string aOriOut = aICNM->Assoc1To1(aKeyOut,aVImgs[i],true);
        cOrientationConique  anOC = aCam->StdExportCalibGlob();
        anOC.Interne().SetNoInit();
        anOC.FileInterne().SetVal(aNewCalib);

        std::cout << "Generate " << aOriRS << "/" << aVImgs[i] << ".xml" << endl;
        MakeFileXML(anOC,aOriOut); 
    }
    if(aCal.copy(aCalRS,true))
        std::cout << "Create Calibration file "+aNewCalib << endl;


    // gaussian noise
    if(!EAMIsInit(&aSeed))  aSeed=time(0);

    std::default_random_engine generator(aSeed);

    std::normal_distribution<double> distributionX(aNoiseGaussian[0],aNoiseGaussian[1]);
    std::normal_distribution<double> distributionY(aNoiseGaussian[2],aNoiseGaussian[3]);

    //1. lecture of tie points and orientation
    std::cout << "Loading tie points + orientation...   ";
    cSetTiePMul * pSH = new cSetTiePMul(0);
    pSH->AddFile(aSH);
    std::map<std::string,cCelImTPM *> aVName2Im = pSH->DicoIm().mName2Im;

    // load cam for all Img
    // Iterate through all elements in std::map
    vector<CamStenope*> aVCam (aVName2Im.size());//real poses
    vector<CamStenope*> aVCambis (aVName2Im.size());//generated poses
    for(auto &aName2Im:aVName2Im)
    {
        CamStenope * aCam = aICNM->StdCamStenOfNames(aName2Im.first,aOri);
        aCam->SetNameIm(aName2Im.first);
        aVCam[aName2Im.second->Id()] = aCam;

        //std::string aNamebis = aName2Im.first.substr(0,aName2Im.first.size()-aSzPF)+"_bis"+aPostfix;
        CamStenope * aCambis = aICNM->StdCamStenOfNames(aName2Im.first,aOriRS);
        aCambis->SetNameIm(aName2Im.first);
        aVCambis[aName2Im.second->Id()] = aCambis;
    }

    std::cout << "Finish loading " << pSH->VPMul().size() << " CONFIG\n";

    // declare aVStructH to stock generated tie points
    vector<ElPackHomologue> aVPack (aVName2Im.size());
    vector<int> aVIdImS (aVName2Im.size(),-1);
    StructHomol aStructH;
    aStructH.VElPackHomol = aVPack;
    aStructH.VIdImSecond = aVIdImS;
    vector<StructHomol> aVStructH (aVName2Im.size(),aStructH);

    //2. get 2D/3D position of tie points
    std::cout << "Filling ElPackHomologue...   ";

    if (EAMIsInit(&aNoiseGaussian))
    {
        std::cout << "Gaussian Noise: " << aNoiseGaussian << endl;
    }


    // parse Configs aVCnf
    std::vector<cSetPMul1ConfigTPM *> aVCnf = pSH->VPMul();
    for (auto &aCnf:aVCnf)
    {
        std::vector<int> aVIdIm =  aCnf->VIdIm();

        // Parse all pts in one Config
        for (uint aKPtCnf=0; aKPtCnf<uint(aCnf->NbPts()); aKPtCnf++)
        {
            vector<Pt2dr> aVPtInter;
            vector<CamStenope*> aVCamInter;
            vector<CamStenope*> aVCamInterbis;
            vector<int> aVIdImInter;

            // Parse all imgs for one pts
            for (uint aKImCnf=0; aKImCnf<aVIdIm.size(); aKImCnf++)
            {

                aVPtInter.push_back(aCnf->Pt(aKPtCnf, aKImCnf));
                aVCamInter.push_back(aVCam[aVIdIm[aKImCnf]]);
                aVCamInterbis.push_back(aVCambis[aVIdIm[aKImCnf]]);
                aVIdImInter.push_back(aVIdIm[aKImCnf]);
            }

            //Intersect aVPtInter:

            ELISE_ASSERT(aVPtInter.size() == aVCamInter.size(), "Size not coherent");
            ELISE_ASSERT(aVPtInter.size() > 1 && aVCamInter.size() > 1, "Nb faiseaux < 2");
            Pt3dr aPInter3D = Intersect_Simple(aVCamInter , aVPtInter);
            //std::cout << aPInter3D << endl;


            // reproject aPInter3D sur tout les images dans aVCamInter
            std::vector<Pt2dr> aVP2d;
            std::vector<CamStenope *> aVCamInterVu;
            std::vector<int> aVIdImInterVu;

            for (uint itVCI=0; itVCI < aVCamInter.size(); itVCI++)
            {
                CamStenope * aCam = aVCamInter[itVCI];
                Pt2dr aPt2d0 = aCam->R3toF2(aPInter3D);//P0

                CamStenope * aCambis = aVCamInterbis[itVCI];
                Pt2dr aPt2d1 = aCambis->R3toF2(aPInter3D);//P1

                // Pl = l*P1 + (1-l)P0
                double aY = aCam->Sz().y * aPt2d0.y /( aCam->Sz().y - aPt2d1.y + aPt2d0.y );
                double aRatio = aY / aCam->Sz().y;
                double aX = aRatio * (aPt2d1.x-aPt2d0.x) + aPt2d0.x;
                Pt2dr aPt2d = Pt2dr(aX,aY);

                //std::cout << "P0: " << aPt2d0 << " P1: " << aPt2d1 << " Pl: " << aPt2d << endl;

                if (EAMIsInit(&aNoiseGaussian))
                {
                    aPt2d.x += distributionX(generator);
                    aPt2d.y += distributionY(generator);
                }

                //std::cout << aPt2d0.x << " " << aPt2d0.y << " " << aPt2d.x << " " << aPt2d.y << " " << aPt2d1.x << " " << aPt2d1.y << " " << endl;


                //check if the point is in the camera view
                if (aCam->PIsVisibleInImage(aPInter3D) && IsInImage(aCam->Sz(),aPt2d))
                {
                    aVP2d.push_back(aPt2d);
                    //std::cout << aPt2d << endl;
                    aVCamInterVu.push_back(aCam);
                    aVIdImInterVu.push_back(aVIdImInter[itVCI]);
                }
            }

            // parse images to fill ElPackHomologue
            for (uint it1=0; it1 < aVCamInterVu.size(); it1++)
            {
                int aIdIm1=aVIdImInterVu.at(it1);
                aVStructH.at(aIdIm1).IdIm=aIdIm1;

                for (uint it2=0; it2 < aVCamInterVu.size(); it2++)
                {
                    if (it1==it2) continue;

                    int aIdIm2=aVIdImInterVu.at(it2);

                    ElCplePtsHomologues aCPH (aVP2d[it1],aVP2d[it2]);
                    aVStructH.at(aIdIm1).VElPackHomol.at(aIdIm2).Cple_Add(aCPH);
                    aVStructH.at(aIdIm1).VIdImSecond.at(aIdIm2)=aIdIm2;
                }
            }


        }
    }

    std::cout << "ElPackHomologue filled !\n";

    //writing of new tie points
    std::cout << "Writing Homol files...   ";
    //key for tie points
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
            + "_"
            +  std::string(aSHOut)
            +  std::string("@")
            +  std::string("dat");


    for (uint itVSH=0; itVSH < aVStructH.size(); itVSH++)
    {
        int aIdIm1 = aVStructH.at(itVSH).IdIm;
        CamStenope * aCam1 = aVCam.at(aIdIm1);
        std::string aNameIm1 = aCam1->NameIm();
        if (IsInList(aVImgs,aNameIm1))
        {
            for (uint itVElPH=0; itVElPH < aVStructH.at(itVSH).VElPackHomol.size(); itVElPH++)
            {
                int aIdIm2 = aVStructH.at(itVSH).VIdImSecond.at(itVElPH);
                if (aIdIm2 == -1) continue;
                CamStenope * aCam2 = aVCam.at(aIdIm2);
                std::string aNameIm2 = aCam2->NameIm();
                if (IsInList(aVImgs,aNameIm2))
                {
                    std::string aHmOut= aICNM->Assoc1To2(aKHOut, aNameIm1, aNameIm2, true);
                    ElPackHomologue aPck = aVStructH.at(aIdIm1).VElPackHomol.at(aIdIm2);
                    aPck.StdPutInFile(aHmOut);
                }
            }
        }

    }

    // write seed file
    ofstream aSeedfile;
    aSeedfile.open ("Homol_"+aSHOut+"/Seed.txt");
    std::cout << "Homol_"+aSHOut+"/Seed.txt" << endl;
    aSeedfile << aSeed << endl;
    aSeedfile.close();

    std::cout << "Finished writing Homol files ! \n";

    // convert Homol folder into new format
    std::string aComConvFH = MM3dBinFile("TestLib ConvNewFH")
                           + aImgs
                           + " All SH=_"
                           + aSHOut
                           + " ExportBoth=1";
    system_call(aComConvFH.c_str());
    std::cout << aComConvFH << endl;

    return EXIT_SUCCESS;
}

int GenerateOrient_main (int argc, char ** argv)
{
    std::string aPatImgs, aOri, aSHOut{"Modif_orient.txt"}, aDir, aImgs,aOut{"Modif_orient.txt"};
    Pt2dr aTInterv, aGauss;
    int aSeed;
    std::vector<std::string> aVTurn;
    bool aTrans{true};
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image Pattern, make sure images are listed in the right order",eSAM_IsExistFile)
                           << EAMC(aOri, "Ori",  eSAM_IsExistDirOri)
                           << EAMC(aTInterv, "Time Interval, interpolate to generate translation, [cadence (s), exposure time (ms)]")
                           << EAMC(aGauss,"Gaussian distribution parameters of angular velocity for rotation generation (radian/s), [mean,std]"),
                LArgMain() << EAM(aOut,"Out",true,"Output file name for genarated orientation, def=Modif_orient.txt")
                           << EAM(aSeed,"Seed",false,"Random engine, if not give, computer unix time is used.")
                           << EAM(aVTurn,"Turn",false,"List of image names representing flight turns (set the translation T(i) as T(i-1))")
                           << EAM(aTrans,"Trans",false,"Take into account translation, def=true")
                );

    // get directory
    SplitDirAndFile(aDir,aImgs,aPatImgs);
    StdCorrecNameOrient(aOri, aDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImgs = *(aICNM->Get(aImgs));


    int aNbIm = aVImgs.size();
    double aRatio = aTInterv.y/1000/aTInterv.x;

    std::vector<Pt3dr> aVTrans;
    // generation of translation, interpolation
    for(int i=0; i<aNbIm-1; i++)
    {
        CamStenope * aCam0 = aICNM->StdCamStenOfNames(aVImgs[i],aOri);
        Pt3dr aP0 = aCam0->PseudoOpticalCenter();

        CamStenope * aCam1 = aICNM->StdCamStenOfNames(aVImgs[i+1],aOri);
        Pt3dr aP1 = aCam1->PseudoOpticalCenter();

        Pt3dr aP = aTrans? (aP1-aP0) * aRatio : Pt3dr(0,0,0);
        //Pt3dr aP_Last = i==0 ? aP : aVTrans.back();

        if(IsInList(aVTurn,aVImgs[i]))
        {
            aP = aVTrans.back();
            std::cout << "Reset Translation for " << aVImgs[i] << endl;
        }

        aVTrans.push_back(aP);
        if(i==(aNbIm-2)) aVTrans.push_back((aP));
    }
    //std::cout << aVTrans.size() << endl;

    // generation of rotation, axis: uniform distribution, angle: gaussian distribution
    if(!EAMIsInit(&aSeed))  aSeed=time(0);
    std::default_random_engine generator(aSeed);
    std::normal_distribution<double> angle(aGauss.x,aGauss.y);
    std::uniform_real_distribution<double> axis(-1.0,1.0);

    std::vector<ElMatrix<double>> aVRot;
    for(int i=0; i<aNbIm;i++)
    {
        Pt3dr aU = Pt3dr(axis(generator),axis(generator),axis(generator)); // axis of rotation
        aU = aU / euclid(aU); // normalize

        double aTeta = angle(generator)*aTInterv.y/1000;

        Pt3dr aCol1 = Pt3dr(
                            cos(aTeta) + aU.x*aU.x*(1-cos(aTeta)),
                            aU.y*aU.x*(1-cos(aTeta)) + aU.z*sin(aTeta),
                            aU.z*aU.x*(1-cos(aTeta)) - aU.y*sin(aTeta)
                           );
        Pt3dr aCol2 = Pt3dr(
                            aU.x*aU.y*(1-cos(aTeta)) - aU.z*sin(aTeta),
                            cos(aTeta) + aU.y*aU.y*(1-cos(aTeta)),
                            aU.z*aU.y*(1-cos(aTeta)) + aU.x*sin(aTeta)
                           );
        Pt3dr aCol3 = Pt3dr(
                            aU.x*aU.z*(1-cos(aTeta)) + aU.y*sin(aTeta),
                            aU.y*aU.z*(1-cos(aTeta)) + aU.x*sin(aTeta),
                            cos(aTeta) + aU.z*aU.z*(1-cos(aTeta))
                           );

        ElMatrix<double> aRot = MatFromCol(aCol1,aCol2,aCol3);
        aVRot.push_back(aRot);
    }
    //std::cout << aVRot.size() << endl;

    ELISE_ASSERT(aVTrans.size()==aVRot.size(),"Different size for translation and rotation!");
    ofstream aFile;
    aFile.open (aOut);

    aFile << "Random engine :" << aSeed << endl;
    aFile << "#F=Tx_Ty_Tz_R00_R01_R02_R10_R11_R12_R20_R21_R22" << endl;

    for(int i=0; i<aNbIm; i++)
    {
        aFile << aVTrans.at(i).x << " "
              << aVTrans.at(i).y << " "
              << aVTrans.at(i).z << " "
              << aVRot[i](0,0) << " "
              << aVRot[i](0,1) << " "
              << aVRot[i](0,2) << " "
              << aVRot[i](1,0) << " "
              << aVRot[i](1,1) << " "
              << aVRot[i](1,2) << " "
              << aVRot[i](2,0) << " "
              << aVRot[i](2,1) << " "
              << aVRot[i](2,2) << endl;
    }

    aFile.close();

    std::cout << aOut << endl;

    return EXIT_SUCCESS;
}

int ReechRolShut_main(int argc, char ** argv)
{
    std::string aPatIm,aSH,aOri,aSHOut{"-Reech"},aMAFIn,aMAFOut{"Mesure_Finale-Reech.xml"},aCalcV;
    double aRSSpeed;
    ElInitArgMain
            (
                argc,argv,
                LArgMain() << EAMC(aPatIm,"Image pattern",eSAM_IsExistFile)
                           << EAMC(aOri,"Input orientation folder",eSAM_IsExistDirOri)
                           << EAMC(aRSSpeed,"Rolling shutter speed (us/line)")
                           << EAMC(aCalcV,"File containing camera velocity"),
                LArgMain() << EAM(aSH,"SHIn",true,"Input tie point file (new format)")
                           << EAM(aSHOut,"SHOut",true,"Folder postfix for tie point output folder, def=_Reech")
                           << EAM(aMAFIn,"MAFIn",true,"Input image measurement file")
                           << EAM(aMAFOut,"MAFOut",true,"Output image measurement file, def=Mesure_Finale-Reech.xml")
                );

    cAppli_CamXifDate anAppli_CamXifDate(aPatIm,aOri,aCalcV);

    if(EAMIsInit(&aSH))
    {
        cSetTiePMul_Cam aSetTiePMul_Cam(aSH,anAppli_CamXifDate);
        aSetTiePMul_Cam.ReechRS_SH(aRSSpeed,aSHOut);
    }
    if(EAMIsInit(&aMAFIn))
    {
        cSetOfMesureAppuisFlottants_Cam aSetOfMesureAppuisFlottants_Cam(aMAFIn,anAppli_CamXifDate);
        std::map<Key,Pt2dr> aMap = aSetOfMesureAppuisFlottants_Cam.ReechRS_MAF(aRSSpeed);
        aSetOfMesureAppuisFlottants_Cam.Export_MAF(aMAFOut,aMap);
    }

    return EXIT_SUCCESS;
}

int ExportTPM_main(int argc, char ** argv)
{
    std::string aSH, aSHFile{"PMulAll.txt"}, aOut{"multiplicity.txt"}, aDir, aSHPat;
    ElInitArgMain
            (
                argc,argv,
                LArgMain() << EAMC(aSH,"Postfix of tie point file",eSAM_IsExistFile),
                LArgMain() << EAM(aSHFile,"SHFile",true,"tie point file name (new format), def=PMulAll.txt")
                           << EAM(aOut,"Out",true,"Output file name, def=SH_multiplicity.txt")
                );

    SplitDirAndFile(aDir, aSHPat, aSH);
    //cInterfChantierNameManipulateur* aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    std::map<int,int> aMapTPM;

    cSetTiePMul * pSH=new cSetTiePMul(0);
    pSH->AddFile("Homol" + aSH + "/" + aSHFile);
    const std::vector<cSetPMul1ConfigTPM *> & aVCnf = pSH->VPMul();
    for(auto & aCnf:aVCnf)
    {
        auto search = aMapTPM.find(aCnf->NbIm());
        if(search == aMapTPM.end())
            aMapTPM.insert(pair<int,int>(aCnf->NbIm(),aCnf->NbPts()));
        else
            aMapTPM.at(aCnf->NbIm()) += aCnf->NbPts();
    }

    ofstream aTPMFile;
    aTPMFile.open(aOut);
    for(auto aTPM:aMapTPM)
    {
        std::cout << aTPM.first << " " << aTPM.second << endl;
        aTPMFile << aTPM.first << " " << aTPM.second << endl;
    }
    aTPMFile.close();
    return EXIT_SUCCESS;
}

int ReechRolShutV1_main(int argc, char ** argv)
{
    std::string aSH,aSHOut{"_Reech"},aCamVFile,aMAFIn,aMAFOut,aDir,aSHIn;
    std::vector<double> aData;
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aCamVFile,"File containg image velocity and image depth",eSAM_IsExistFile)
                           << EAMC(aData,"[image pixel height (um), focal length (mm), rolling shutter speed (us)]"),
                LArgMain() << EAM(aSH,"SHIn",true,"Input tie point file (new format)")
                           << EAM(aSHOut,"SHOut",true,"File postfix of the output tie point file (new format), def=_Reech")
                           << EAM(aMAFIn,"MAFIn",true,"Input image measurement file")
                           << EAM(aMAFOut,"MAFOut",true,"Output image measurement file")
                );
    // get directory
    SplitDirAndFile(aDir,aSHIn,aSH);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    double h = aData.at(0)/1000/1000; // um
    double f = aData.at(1)/1000; // mm
    double T = aData.at(2)/1000000; // us

    // read file containing image velocity and depth
    std::ifstream aFile(aCamVFile.c_str());

    std::map<std::string, double> aMapReechScale;
    if(aFile)
    {
        std::cout << "Read File : " << aCamVFile << endl;
        std::string aName;
        double aDiffPos,aDiffSecond,aV,aH;

        while(aFile >> aName >> aDiffPos >> aDiffSecond >> aV >> aH)
        {
            double lambda = 1 - f*aV*T/h/aH;
            std::cout << std::setprecision(10) << aName << " " << lambda << endl;
            aMapReechScale.insert(pair<std::string, double>(aName, lambda));
        }
        aFile.close();
    }

    if(EAMIsInit(&aSH))
    {
        // load tie points
        cSetTiePMul * pSH = new cSetTiePMul(0);
        pSH->AddFile(aSH);
        //std::map<std::string,cCelImTPM *> aVName2Im = pSH->DicoIm().mName2Im;

        // modification of tie points
        std::vector<cSetPMul1ConfigTPM *> aVCnf = pSH->VPMul();
        for(auto & aCnf:aVCnf)
        {
            std::vector<int> aVIdIm =  aCnf->VIdIm();
            // Parse all pts in one Config
            for(uint aKPt=0; aKPt<uint(aCnf->NbPts());aKPt++)
            {
                // Parse all imgs for one pt
                for(uint aKIm=0;aKIm<aVIdIm.size();aKIm++)
                {
                    std::string aNameIm = pSH->NameFromId(aVIdIm.at(aKIm));
                    Pt2dr aPt = aCnf->Pt(aKPt, aKIm);
                    Pt2dr aNewPt = Pt2dr(aPt.x,aPt.y*aMapReechScale[aNameIm]);
                    //std::cout << "Name:" << aNameIm << " Pt: " << aPt << " Scale:" << aMapReechScale[aNameIm] << endl;

                    aCnf->SetPt(aKPt,aKIm,aNewPt);
                }
            }
        }


        // output modifeied tie points
        std::string aNameOut0 = cSetTiePMul::StdName(aICNM,aSHOut,"Reech",0);
        std::string aNameOut1 = cSetTiePMul::StdName(aICNM,aSHOut,"Reech",1);

        pSH->Save(aNameOut0);
        pSH->Save(aNameOut1);
    }

    if(EAMIsInit(&aMAFIn))
    {
        cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aMAFIn,SetOfMesureAppuisFlottants);
        std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();

        for(auto & aMAF : aLMAF)
        {
            std::string aNameIm = aMAF.NameIm();
            //std::list<cOneMesureAF1I> & aMes = aMAF.OneMesureAF1I();
            std::cout << aNameIm << endl;

            for(auto & aOneMes : aMAF.OneMesureAF1I())
            {
                Pt2dr aPt = aOneMes.PtIm();
                std::cout << aOneMes.NamePt() << " before:" << aOneMes.PtIm();
                Pt2dr aNewPt = Pt2dr(aPt.x,aPt.y*aMapReechScale[aNameIm]);
                // aOneMes.SetPtIm(aNewPt); ==> MPD D'OU VIENT CETTE FONCTION ??? COMPILE PAS
                aOneMes.PtIm() = aNewPt;
                std::cout << " after:" << aOneMes.PtIm() << endl;
            }
        }
    MakeFileXML(aDico,aMAFOut);
    }

    return EXIT_SUCCESS;

}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
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
associés au chargement,  a  l'utilisation,  a  la modification et/ou au
développement et a  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe a
manipuler et qui le réserve donc a  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités a  charger  et  tester  l'adéquation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
a  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder a cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
