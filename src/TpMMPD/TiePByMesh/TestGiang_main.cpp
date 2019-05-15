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

#include "InitOutil.h"
#include "DrawOnMesh.h"
#include "CorrelMesh.h"
#include "Pic.h"
#include "Triangle.h"
#include <stdio.h>
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"

class cOneLoop;
Pt3dr Intersect_Simple(const std::vector<CamStenope *> & aVCS,const std::vector<Pt2dr> & aNPts2D);
void Intersect_Simple(const std::vector<cOneLoop *> &aVOneLoop, std::vector< vector<Pt3dr> > & aResultPt3D, vector<Pt3d<double> > &aResultPt3DAllImg);

// ========== PLYBascule ==========
int PlyBascule(int argc, char ** argv)
{
    string aPlyO="";
    string aPly;
    string aXML;
    bool aBin=1;

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPly, "Input Ply (mesh or C3DC dense cloud)",  eSAM_IsExistFile)
                      << EAMC(aXML, "Bascule XML (xml result from GCPBascule)", eSAM_IsExistFile),
          LArgMain()
                      << EAM(aPlyO, "Out" , true, "Name of output PLY")
                      << EAM(aBin, "Bin" , true, "Binary (1 - def) / ASCII (0)")
                );

    // Lire XML
    cXml_ParamBascRigide  *  aTransf = OptStdGetFromPCP(aXML,Xml_ParamBascRigide);
    double aScl = aTransf->Scale();
    Pt3dr aTr = aTransf->Trans();

    // Lire PLY
    cout<<"Lire PLY .... "<<endl;
    cMesh myMesh(aPly, true);
    const int nVertex = myMesh.getVertexNumber();
    cout<<"Nb Face = "<<myMesh.getFacesNumber()<<endl;
    cout<<"Nb Vertices = "<<myMesh.getVertexNumber()<<endl;

    cout<<"Bascule ...."<<endl;

    for (double aKV=0; aKV<nVertex ; aKV++)
    {
        cVertex* aV = myMesh.getVertex(aKV);
        Pt3dr aPt;
        aV->getPos(aPt);
        Pt3dr aPtBasc(
                    scal(aTransf->ParamRotation().L1() , aPt) * aScl + aTr.x,
                    scal(aTransf->ParamRotation().L2() , aPt) * aScl + aTr.y,
                    scal(aTransf->ParamRotation().L3() , aPt) * aScl + aTr.z
                     );

        aV->modPos(aPtBasc);
    }
    cout<<"Write output PLY...."<<endl;
    myMesh.write(aPlyO, aBin);
    cout<<"Done !"<<endl;
    return EXIT_SUCCESS;
}

// ========== Test Zone Trajecto Acquisition BLoc Rigid ==========
int Test_TrajectoFromOri(int argc, char ** argv)
{
    // Tracer trajectoire en concatenant tout les centres du camera dans Ori donne
    string aDir="./";
    string aPatIn;
    string aPlyOut="";
    string aPlyN;
    string aPatIm;
    string aOri, aBlinis;

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPatIn, "PatIm1",  eSAM_IsPatFile)
                      << EAMC(aOri, "Ori",  eSAM_IsExistDirOri),
          LArgMain()
                      << EAM(aPlyN, "Ply" , true, "Name of output PLY trajecto")
                      << EAM(aBlinis, "Blinis" , true, "Rigid Structur")
                );

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    SplitDirAndFile(aDir, aPatIm, aPatIn);
    StdCorrecNameOrient(aOri, aICNM->Dir());

    if (EAMIsInit(&aPlyN))
    {
        aPlyOut=aPlyN;
    }
    else
    {
        aPlyOut = aPlyOut+ "Trajecto_" + aOri + ".ply";
    }

    std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));
    ELISE_ASSERT(aSetIm.size() > 2, "Nb Img < 2 => Comment peut je trace trajecto ?");

    std::vector<CamStenope*> aVCam;

    cPlyCloud aPly;
    for (uint aKCam=0; aKCam<aSetIm.size(); aKCam++)
    {
        string aImName = aSetIm[aKCam];
        CamStenope * aCam = aICNM->StdCamStenOfNames(aImName, aOri);
        aCam->SetNameIm(aImName);
        aVCam.push_back(aCam);
    }
    ELISE_ASSERT(aSetIm.size() == aVCam.size(), "Nb Img != Nb Orientation");


    double dist=euclid(aVCam[1]->VraiOpticalCenter() - aVCam[0]->VraiOpticalCenter());
    for (uint aKCam=1; aKCam<aVCam.size(); aKCam++)
    {

            CamStenope * aCam = aVCam[aKCam-1];
            CamStenope * aCamN = aVCam[aKCam];

            Pt3dr aCentre = aCam->VraiOpticalCenter();
            Pt3dr aCentreN = aCamN->VraiOpticalCenter();

            aPly.AddSphere(Pt3di(255,0,0), aCentre, dist/20.0, 10);
            aPly.AddSeg(Pt3di(0,255,0), aCentre, aCentreN, 200);

    }
    aPly.PutFile(aPlyOut.c_str());
    return EXIT_SUCCESS;
}
// ========== Test Zone Verifier Aero par la fermeture de boucle ==========
class cOneImInLoop
{
public:
    cOneImInLoop(int IdLoop, int IdImg, string aNameIm);
    void AddPt(Pt2dr aPt);
    vector<Pt2dr> mVPt;
    int mIdLoop;
    int mIdImg;
    string mNameIm;
    CamStenope * mCam;
};

cOneImInLoop::cOneImInLoop(int IdLoop, int IdImg, string aNameIm):
    mIdLoop (IdLoop),
    mIdImg (IdImg),
    mNameIm (aNameIm),
    mCam (NULL)
{

}

void cOneImInLoop::AddPt(Pt2dr aPt)
{
    mVPt.push_back(aPt);
}

class cOneLoop
{
    public :
        cOneLoop();
        vector<cOneImInLoop *> mVIm;
        void AddIm(cOneImInLoop  * aIm);
};

cOneLoop::cOneLoop()
{}

void cOneLoop::AddIm(cOneImInLoop  * aIm)
{
    mVIm.push_back(aIm);
}




class cOnePtMeasure
{
public:
    cOnePtMeasure(string aName);
    void AddMeasure(Pt2dr aCoor, string aNameIm, CamStenope * aCam, int idLoop);
    string mNamePt;
    vector<string> mVNameIm;

    vector<CamStenope*> mVCamLoop1;
    vector<Pt2dr> mVMeasureLoop1;

    vector<CamStenope*> mVCamLoop2;
    vector<Pt2dr> mVMeasureLoop2;

    bool mInLoop1;
    bool mInLoop2;

    bool operator==(const cOnePtMeasure& r) const
    {
        return mNamePt == r.mNamePt;
    }
};

cOnePtMeasure::cOnePtMeasure(string aName):
    mNamePt (aName),
    mInLoop1 (false),
    mInLoop2 (false)
{}

void cOnePtMeasure::AddMeasure(Pt2dr aCoor, string aNameIm, CamStenope *aCam, int idLoop)
{
    if (idLoop == 1)
    {
        mVCamLoop1.push_back(aCam);
        mVMeasureLoop1.push_back(aCoor);
        mVNameIm.push_back(aNameIm);
        mInLoop1 = true;
    }
    if (idLoop == 2)
    {
        mVCamLoop2.push_back(aCam);
        mVMeasureLoop2.push_back(aCoor);
        mVNameIm.push_back(aNameIm);
        mInLoop2 = true;
    }
    return;
}

struct MatchString
{
 MatchString(const std::string & s) :
     s_(s)
 {}

 bool operator()(const cOnePtMeasure& obj) const
 {
   return obj.mNamePt == s_;
 }

 private:
   const std::string & s_;
};

bool sortAscending(double i, double j) { return i < j; }

int find_Obj_in_cOnePtMeasure(string & aNamePt, std::vector<cOnePtMeasure*> & aVPtMes)
{
    vector<string> aName;
    for (uint aKP=0; aKP<aVPtMes.size(); aKP++)
    {
        cOnePtMeasure * aPt = aVPtMes[aKP];
        aName.push_back(aPt->mNamePt);

    }
    vector<string>::iterator it;
    it = find(aName.begin(), aName.end(), aNamePt);
    if (it != aName.end())
    {
        return distance(aName.begin(), it);
    }
    else
    {
        return -1;
    }
}

int Test_CtrlCloseLoopv2(int argc, char ** argv)
{
    string aDir = "./";
    vector<string> aVPatIn;
    string aPatIn;
    string aOriA;
    string aSH ="";
    Pt2di DoTapioca(0,-1);

    Pt3di colSeg(0,255,0);
    string aPlyErrName = "PlyErr.ply";
    bool aSilent = true;
    string aXML="";

    string aCSV = "CtrlCloseLoopv2.csv";

    ElInitArgMain
    (
          argc,argv,
          LArgMain()
                      << EAMC(aPatIn, "Pat",  eSAM_None)
                      << EAMC(aOriA, "Ori",  eSAM_IsExistDirOri)
                      << EAMC(aSH, "SH file homol new format contains all selected im, "" if don't have or not sure to compute homol (set with option DoTapioca)"),
          LArgMain()
                      << EAM(aPatIn, "Pat",  true, "[Pat1, Pat2, ...]")
                      << EAM(aCSV, "CSV",  true, "CSV Filename")
                );

    std::ifstream file(aPatIn);
    std::string str;
    while (std::getline(file, str))
    {
      aVPatIn.push_back(str);
      cout<<str<<endl;
    }
    int aNbLoop = aVPatIn.size();
    cout<<"Nb Loop : "<<aVPatIn.size()<<endl;

     cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
     StdCorrecNameOrient(aOriA, aICNM->Dir());
     std::map<std::string, int> aMap_Name2LoopId;

     vector< vector<string> > aVSetIm;

     std::vector<std::string> aAllIm;


     for(uint aK=0; aK<aVPatIn.size(); aK++)
     {
        vector<std::string> aSetIm = *(aICNM->Get(aVPatIn[aK]));
        aVSetIm.push_back(*(aICNM->Get(aVPatIn[aK])));
        for (uint aK1=0; aK1<aSetIm.size(); aK1++)
        {
            aAllIm.push_back(aSetIm[aK1]);
            aMap_Name2LoopId.insert(std::pair<string, int>(aSetIm[aK1], aK));
        }
     }
     cout<<"Nb Im : "<<aAllIm.size()<<endl;

     // homologue interaction
     const std::string  aSHInStr = aSH;
     cSetTiePMul * aSetTiePMul = new cSetTiePMul(0);
     aSetTiePMul->AddFile(aSHInStr);

        // Look for a config that contains all of these images

     // this map contain all image name in homologue fileœ
     std::map<std::string,cCelImTPM *> aMap_Name2Im = aSetTiePMul->DicoIm().mName2Im;
     std::map<std::string,cCelImTPM *>::iterator aIt_Find; // iterator to check if image exist in homol pack

     vector<string> aVAllImPresent;
     vector<int> aIdAllImg;

     vector<cOneImInLoop*> aVImInLoop;

     vector<CamStenope *> aVCam;
     for (uint aK=0; aK<aAllIm.size(); aK++)
     {
         string aImName = aAllIm[aK];
         aIt_Find = aMap_Name2Im.find(aImName);
         if (aIt_Find != aMap_Name2Im.end())
         {
             // image found in homol set
             std::map<std::string,int>::iterator aIt;
             aIt = aMap_Name2LoopId.find(aImName);
             aVAllImPresent.push_back(aImName);
             CamStenope * aCam = aICNM->StdCamStenOfNames(aImName, aOriA);
             aCam->SetNameIm(aImName);
             aVCam.push_back(aCam);
             aIdAllImg.push_back(aIt_Find->second->Id());
             int aLoopId = aIt->second;
             int aImId = aIt_Find->second->Id();
             cOneImInLoop* aImLoop = new cOneImInLoop(aLoopId, aImId, aImName);
             aVImInLoop.push_back(aImLoop);
         }
         else
         {
             std::map<std::string,int>::iterator aIt; // iterator to check if image exist in homol pack
             aIt = aMap_Name2LoopId.find(aImName);
             cout<<"WARN !! : image "<<aImName<<" - loop "<<aIt->second<<endl;
         }
     }

     // intersection : Query pixel in homol point set given (nameIm, vector<string>aVAllIm)

     std::vector<cSetPMul1ConfigTPM *> aVCnf = aSetTiePMul->VPMul();

     vector<cSetPMul1ConfigTPM * > aSelectCnf;  // config contains points from all image

     cout<<"Chercher bon homol multiple... "<<endl;
     for (uint aKCnf=0; aKCnf<aVCnf.size(); aKCnf++)
     {
         cSetPMul1ConfigTPM* aCnf = aVCnf[aKCnf];
         // Get config that has more than nb of selected images
         if (aCnf->NbIm() >= (int)aVAllImPresent.size())
         {
             // Get config that contains all selected image
             vector<int> aIDImgInCnf = aCnf->VIdIm();
             bool isCnfContainAllId = true;
             for (uint aKId=0; aKId < aIdAllImg.size(); aKId++)
             {
                 int aQueryID = aIdAllImg[aKId];
                 // is this aQueryID exist in this config image list ?
                 vector<int>::iterator it_Find;
                 it_Find = find(aIDImgInCnf.begin(), aIDImgInCnf.end(), aQueryID);
                 if(it_Find == aIDImgInCnf.end())
                 {
                     isCnfContainAllId = isCnfContainAllId && false;
                 }
             }
             if (isCnfContainAllId)
             {
                 aSelectCnf.push_back(aCnf);
             }
         }
     }
     cout<<"Nb config Selected : "<<aSelectCnf.size()<<endl;
     if (!aSilent)
     {
         cout<<aSelectCnf.size()<<" cnf selected ! "<<endl;
     }
     // For each selected config, get pt 2D on each correspondant control image, intersect and suivi aussi
     for (uint aKCnf=0; aKCnf<aSelectCnf.size(); aKCnf++)
     {
         cSetPMul1ConfigTPM* aCnf = aSelectCnf[aKCnf];
         for (uint aKIdImg=0; aKIdImg<aIdAllImg.size(); aKIdImg++)
         {
             int aQueryId = aIdAllImg[aKIdImg]; // Id of an Image in pts homologue file
             // get all 2D points of image ID aQueryId in aCnf
             cOneImInLoop * aImLoop = aVImInLoop[aKIdImg];
             for (uint aKPt=0; (int)aKPt<aCnf->NbPts(); aKPt++)
             {
                 Pt2dr aPt = aCnf->GetPtByImgId(aKPt, aQueryId);
                 string aNameIm = aVAllImPresent[aKIdImg];
                 CamStenope * aCam = aVCam[aKIdImg];
                 std::map<std::string, int>::iterator aIt;
                 aIt = aMap_Name2LoopId.find(aNameIm);
                 aImLoop->AddPt(aPt);
                 aImLoop->mCam = aCam;
             }
         }
     }

     // Arrange Image in loop order
     vector<cOneLoop*> aVLoop;
     for (uint aKL=0; (int)aKL<aNbLoop; aKL++)
     {
         cOneLoop * aLoop = new cOneLoop();
         aVLoop.push_back(aLoop);
     }

     for(uint aKIm =0; aKIm<aVImInLoop.size(); aKIm++)
     {
         cOneImInLoop * aImLoop = aVImInLoop[aKIm];
         int aIDLoop = aImLoop->mIdLoop;
         aVLoop[aIDLoop]->AddIm(aImLoop);
     }
     // Intersection by Loop
     vector< vector<Pt3dr> >  aResultPt3D;
     vector<Pt3dr> aPt3dFrmAllIm;
     Intersect_Simple(aVLoop,  aResultPt3D, aPt3dFrmAllIm);

     for (uint aKL=0; (int)aKL<aNbLoop; aKL++)
     {
         vector<Pt3dr> aPts3D = aResultPt3D[aKL];
         cout<<"+ Loop "<<aKL<<" : "<<endl;

         for (uint aKPt=0; aKPt<aPts3D.size(); aKPt++)
         {
             cout<<"  - "<<aPts3D[aKPt]<<endl;
         }
     }

     cout<<"+ All Loop All Im : "<<endl;
     for (uint aKPt=0; aKPt<aPt3dFrmAllIm.size(); aKPt++)
     {
         cout<<"  - "<<aPt3dFrmAllIm[aKPt]<<endl;
     }

     // Export Point for Close Loop Test
     ofstream csvPt3d;
     csvPt3d.open(aCSV);
     for (uint aKL=0; (int)aKL<aNbLoop; aKL++)
     {
         vector<Pt3dr> aPts3D = aResultPt3D[aKL];
         csvPt3d<<aKL;
         for (uint aKPt=0; aKPt<aPts3D.size(); aKPt++)
         {
            csvPt3d<<","<<euclid(aPt3dFrmAllIm[aKPt] - aPts3D[aKPt]);
         }
         csvPt3d<<endl;
     }

     csvPt3d<<"AllIntersect";
     /*
     for (uint aKPt=0; aKPt<aPt3dFrmAllIm.size(); aKPt++)
     {
         csvPt3d<<","<<aPt3dFrmAllIm[aKPt];
     }
     csvPt3d<<endl;
*/
     csvPt3d.close();


    return EXIT_SUCCESS;
}



int Test_CtrlCloseLoop(int argc, char ** argv)
{
    string aDir = "./";
    string aPatIn1, aPatIn2;
    string aPatIm1, aPatIm2;
    string aOriA;
    string aSH ="";
    bool plot=false;
    Pt2di DoTapioca(0,-1);
    double seuilEcart = DBL_MAX;

    Pt3di colSeg(0,255,0);
    double aDynV = 1.1;
    string aPlyErrName = "PlyErr.ply";
    bool aSilent = true;
    string aXML="";

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPatIn1, "PatIm1",  eSAM_IsPatFile)
                      << EAMC(aPatIn2, "PatIm2",  eSAM_IsPatFile)
                      << EAMC(aOriA, "Ori",  eSAM_IsExistDirOri)
                      << EAMC(aSH, "SH file homol new format contains all selected im, "" if don't have or not sure to compute homol (set with option DoTapioca)"),
          LArgMain()
                      << EAM(DoTapioca, "Tapioca" , true, "Do Tapioca = [DoIt(1/0),Resolution]")
                      << EAM(plot, "plot" , true, "Plot data (with gnuplot)")
                      << EAM(seuilEcart, "DistMax" , true, "max Point distant between 2 loop (to eliminate noise) - def=Inf")
                      << EAM(colSeg, "Col" , true, "Rayon of error vector")
                      << EAM(aDynV, "Dynv" , true, "Multp factor of error vector")
                      << EAM(aPlyErrName, "Ply" , true, "Name of output error vector")
                      << EAM(aSilent, "Silent" , true, "Display just 3D error")
                      << EAM(aXML, "Saisie", true, "XML of SaisieAppuis - Ctrl on manual picked point")
                );

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    SplitDirAndFile(aDir, aPatIm1, aPatIn1);
    SplitDirAndFile(aDir, aPatIm2, aPatIn2);
    StdCorrecNameOrient(aOriA, aICNM->Dir());

    std::vector<std::string> aSetIm1 = *(aICNM->Get(aPatIm1));
    std::vector<std::string> aSetIm2 = *(aICNM->Get(aPatIm2));
    std::vector<std::string> aAllIm;
    aAllIm.insert(aAllIm.end(), aSetIm1.begin(), aSetIm1.end());
    aAllIm.insert(aAllIm.end(), aSetIm2.begin(), aSetIm2.end());

    if (!aSilent)
    {
        cout<<"Verif : Nb ImLoop 1 "<<aSetIm1.size()<<" Nb ImLoop 2 "<<aSetIm2.size()<<" Nb AllIm "<<aAllIm.size()<<endl;
    }

    if (EAMIsInit(&aXML))
    {
        cSetOfMesureAppuisFlottants aXMLSaisie = StdGetFromPCP(aXML,SetOfMesureAppuisFlottants);
        std::list<cMesureAppuiFlottant1Im> & aList_Im = aXMLSaisie.MesureAppuiFlottant1Im();
        std::vector<cOnePtMeasure*> aVPtMeasure;
        for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aList_Im.begin() ; iT1 != aList_Im.end() ; iT1++)
        {
            cMesureAppuiFlottant1Im aIm = *iT1;
            string aImName = aIm.NameIm();
            CamStenope * aCam = aICNM->StdCamStenOfNamesSVP(aImName, aOriA);

            if (aCam != 0)
            {
            aCam->SetNameIm(aImName);

            bool onL1=false;
            bool onL2=false;
            if (find(aSetIm1.begin(), aSetIm1.end(), aImName) != aSetIm1.end())
            {
                onL1=true;
            }
            if (find(aSetIm2.begin(), aSetIm2.end(), aImName) != aSetIm2.end())
            {
                onL2=true;
            }
            ELISE_ASSERT(!(onL1==true && onL2==true), "Image existid in both 2 loop => impossible");
            if (onL1 || onL2)
            {
                std::list<cOneMesureAF1I> aLst_Mes = aIm.OneMesureAF1I();
                for (std::list<cOneMesureAF1I>::iterator iTMes = aLst_Mes.begin() ; iTMes != aLst_Mes.end() ; iTMes++)
                {
                    cOneMesureAF1I aMes = *iTMes;
                    string aNamePt = aMes.NamePt();
                    Pt2dr aCoor = aMes.PtIm();

                    int itF = find_Obj_in_cOnePtMeasure(aNamePt, aVPtMeasure);
                    //std::vector<cOnePtMeasure>::iterator itF = find_if(aVPtMeasure.begin(), aVPtMeasure.end(), MatchString(aNamePt));

                    if (itF == -1)
                    {
                        cOnePtMeasure * aPtMes = new cOnePtMeasure(aNamePt);
                        if (onL1)
                            aPtMes->AddMeasure(aCoor, aImName, aCam, 1);
                        if (onL2)
                            aPtMes->AddMeasure(aCoor, aImName, aCam, 2);
                        aVPtMeasure.push_back(aPtMes);
                    }
                    else
                    {
                        cOnePtMeasure * aPtMesExist = aVPtMeasure[itF];
                        if (onL1)
                            aPtMesExist->AddMeasure(aCoor, aImName, aCam, 1);
                        if (onL2)
                            aPtMesExist->AddMeasure(aCoor, aImName, aCam, 2);
                    }
                }
            }
            }
            else
            {
                if (!aSilent)
                {
                    cout<<"Image "<<aImName<<" not selected in pattern"<<endl;
                }
            }
        }



        for (uint aKPt=0; aKPt<aVPtMeasure.size(); aKPt++)
        {
            cOnePtMeasure * aPtMes = aVPtMeasure[aKPt];
            string aNamePt = aPtMes->mNamePt;
            cout<<" + Pt : "<<aNamePt<<endl;
            Pt3dr aPt3d_Lp1;
            Pt3dr aPt3d_Lp2;
            if (aPtMes->mVCamLoop1.size() > 1)
            {
                ELISE_ASSERT(aPtMes->mVCamLoop1.size() == aPtMes->mVMeasureLoop1.size(), "Intersect Loop 1 : VCam not coherent with VMes");
                aPt3d_Lp1 = Intersect_Simple(aPtMes->mVCamLoop1, aPtMes->mVMeasureLoop1);
                cout<<"  + Loop 1: "<<aPt3d_Lp1<<endl;
            }
            if (aPtMes->mVCamLoop2.size() > 1)
            {
                ELISE_ASSERT(aPtMes->mVCamLoop2.size() == aPtMes->mVMeasureLoop2.size(), "Intersect Loop 2 : VCam not coherent with VMes");
                aPt3d_Lp2 = Intersect_Simple(aPtMes->mVCamLoop2, aPtMes->mVMeasureLoop2);
                cout<<"  + Loop 2: "<<aPt3d_Lp2<<endl;
            }
            if (aPtMes->mVCamLoop1.size() > 1 && aPtMes->mVCamLoop2.size() > 1)
            {
                cout<<"  + Delta : "<<aPt3d_Lp2-aPt3d_Lp1<<" - D="<<euclid(aPt3d_Lp2-aPt3d_Lp1)<<endl;
            }

        }
        return EXIT_SUCCESS;
    }


    std::vector<cOneImInLoop*> aVImInLoop;
    std::vector<CamStenope*> aVCam1;
    std::vector<CamStenope*> aVCam2;

    if (DoTapioca.x != 0 || (aSH==""))
    {
        cout<<"Compute Point homologues between 2 close loop"<<endl;
        std::list<std::string> aLCom;
        string cmdTapioca = MM3DStr + " Tapioca All " + "\'"+aPatIn1+"|" + aPatIn2 +"\' "
                                                     + ToString(DoTapioca.y)+ " PostFix=CtrlCloseLoop";
        string cmdConNewFH = MM3DStr + " TestLib ConvNewFH " +"\'"+aPatIn1+"|" + aPatIn2 +"\' "
                                    + "\'\' SH=CtrlCloseLoop Bin=0";

        aLCom.push_back(cmdTapioca);
        aLCom.push_back(cmdConNewFH);
        cEl_GPAO::DoComInSerie(aLCom);
        aSH = "HomolCtrlCloseLoop/PMul.txt";
    }
    // read new format points homologue
    const std::string  aSHInStr = aSH;
    cSetTiePMul * aSetTiePMul = new cSetTiePMul(0);
    aSetTiePMul->AddFile(aSHInStr);

    if (!aSilent)
    {
        cout<<"Total : "<<aSetTiePMul->DicoIm().mName2Im.size()<<" imgs"<<endl;
    }
    std::map<std::string,cCelImTPM *> aMap_Name2Im = aSetTiePMul->DicoIm().mName2Im;
    std::map<std::string,cCelImTPM *>::iterator aIt_Find;

    vector<int> aIdAllImg;
    vector<int> aIdImgPat1;
    for (uint aKIm1=0; aKIm1<aSetIm1.size(); aKIm1++)
    {
        string aImName = aSetIm1[aKIm1];
        aIt_Find = aMap_Name2Im.find(aImName);
        if (aIt_Find != aMap_Name2Im.end())
        {
            int aImId = aIt_Find->second->Id();
            aIdImgPat1.push_back(aImId);
            aIdAllImg.push_back(aImId);
            cOneImInLoop* aImLoop = new cOneImInLoop(1, aImId, aImName);
            aVImInLoop.push_back(aImLoop);

            CamStenope * aCam = aICNM->StdCamStenOfNames(aImName, aOriA);
            aCam->SetNameIm(aImName);
            aVCam1.push_back(aCam);
        }
        else
        {
            cout<<"BIG WARNING : "<<" File "<<aSH<<" don't contain image "<<aImName<<endl;
            cout<<"Please do Tapioca for all selected control image on 2 loop"<<endl;
            return EXIT_FAILURE;
        }
    }


    vector<int> aIdImgPat2;
    for (uint aKIm2=0; aKIm2<aSetIm2.size(); aKIm2++)
    {
        string aImName = aSetIm2[aKIm2];
        aIt_Find = aMap_Name2Im.find(aImName);
        if (aIt_Find != aMap_Name2Im.end())
        {
            int aImId = aIt_Find->second->Id();
            aIdImgPat2.push_back(aImId);
            aIdAllImg.push_back(aImId);
            cOneImInLoop* aImLoop = new cOneImInLoop(2, aImId, aImName);
            aVImInLoop.push_back(aImLoop);

            CamStenope * aCam = aICNM->StdCamStenOfNames(aImName, aOriA);
            aCam->SetNameIm(aImName);
            aVCam2.push_back(aCam);
        }
        else
        {
            cout<<"BIG WARNING : "<<" File "<<aSH<<" don't contain image "<<aImName<<endl;
            cout<<"Please do Tapioca for all selected control image on 2 loop"<<endl;
            return EXIT_FAILURE;
        }
    }
    if (!aSilent)
    {
        cout<<"VPMul - Nb Config: "<<aSetTiePMul->VPMul().size()<<endl;
    }
    std::vector<cSetPMul1ConfigTPM *> aVCnf = aSetTiePMul->VPMul();

    vector<cSetPMul1ConfigTPM * > aSelectCnf;
    for (uint aKCnf=0; aKCnf<aVCnf.size(); aKCnf++)
    {
        cSetPMul1ConfigTPM* aCnf = aVCnf[aKCnf];
        // Get config that has more than nb of selected images
        if (aCnf->NbIm() >= (int)aAllIm.size())
        {
            // Get config that contains all selected image
            vector<int> aIDImgInCnf = aCnf->VIdIm();
            bool isCnfContainAllId = true;
            for (uint aKId=0; aKId < aIdAllImg.size(); aKId++)
            {
                int aQueryID = aIdAllImg[aKId];
                // is this aQueryID exist in this config image list ?
                vector<int>::iterator it_Find;
                it_Find = find(aIDImgInCnf.begin(), aIDImgInCnf.end(), aQueryID);
                if(it_Find == aIDImgInCnf.end())
                {
                    isCnfContainAllId = isCnfContainAllId && false;
                }
            }
            if (isCnfContainAllId)
            {
                aSelectCnf.push_back(aCnf);
            }
        }
    }
    if (!aSilent)
    {
        cout<<aSelectCnf.size()<<" cnf selected ! "<<endl;
    }
    // For each selected config, get pt 2D on each correspondant control image
    for (uint aKCnf=0; aKCnf<aSelectCnf.size(); aKCnf++)
    {
    cSetPMul1ConfigTPM* aCnf = aSelectCnf[aKCnf];
        for (uint aKIdImg=0; aKIdImg<aIdAllImg.size(); aKIdImg++)
        {
            int aQueryId = aIdAllImg[aKIdImg];
            cOneImInLoop * aImLoop = aVImInLoop[aKIdImg];
            // get all 2D points of image ID aQueryId in aCnf
            for (uint aKPt=0; (int)aKPt<aCnf->NbPts(); aKPt++)
            {
                Pt2dr aPt = aCnf->GetPtByImgId(aKPt, aQueryId);
                aImLoop->AddPt(aPt);
            }
        }

    }
    // Intersection to get 3D points set:

        // Intersect Points on Loop 1
        vector<Pt3dr> aPtCtrlLoop1;
        cOneImInLoop * aImLoop1 = aVImInLoop[0]; // get 1st image in loop 1
        if (aImLoop1->mIdLoop == 1)
        {
            vector<Pt2dr> aVPt = aImLoop1->mVPt;
            for (uint aKPt=0; aKPt<aVPt.size(); aKPt++)
            {
                vector<Pt2dr> aVPtToIntersect;
                for (uint aKImLoop1=0; aKImLoop1<aSetIm1.size(); aKImLoop1++)
                {
                    cOneImInLoop * aIm = aVImInLoop[aKImLoop1];
                    ELISE_ASSERT(aVPt.size() == aIm->mVPt.size(), "AAAA VPT");
                    if (aIm->mIdLoop == 1)
                        aVPtToIntersect.push_back(aIm->mVPt[aKPt]);
                    else
                        ELISE_ASSERT(aIm->mIdLoop == 1,"AAAAAAAAAA Wrong Loop");
                }
                // Intersect
                aPtCtrlLoop1.push_back(Intersect_Simple(aVCam1, aVPtToIntersect));
            }
        }
        else
        {
            cout<<"Image is not in Good Loop 1??? Il y a vraiement prob la..";
        }

        // Intersect Points on Loop 2
        vector<Pt3dr> aPtCtrlLoop2;
        cOneImInLoop * aImLoop2 = aVImInLoop[aSetIm1.size()]; // get 1st image in loop 2
        if (aImLoop2->mIdLoop == 2)
        {
            vector<Pt2dr> aVPt = aImLoop2->mVPt;
            for (uint aKPt=0; aKPt<aVPt.size(); aKPt++)
            {
                vector<Pt2dr> aVPtToIntersect;
                for (uint aKImLoop2=aSetIm1.size(); aKImLoop2<aVImInLoop.size(); aKImLoop2++)
                {
                    cOneImInLoop * aIm = aVImInLoop[aKImLoop2];
                    ELISE_ASSERT(aVPt.size() == aIm->mVPt.size(), "AAAA VPT");
                    if (aIm->mIdLoop == 2)
                        aVPtToIntersect.push_back(aIm->mVPt[aKPt]);
                    else
                        ELISE_ASSERT(aIm->mIdLoop == 2,"AAAAAAAAAA Wrong Loop 2");
                }
                // Intersect
                aPtCtrlLoop2.push_back(Intersect_Simple(aVCam2, aVPtToIntersect));
            }
        }
        else
        {
            cout<<"Image is not in Good Loop 2??? Il y a vraiement prob la..";
        }

        // Export Point for Close Loop Test
        ofstream csvPt3d;
        csvPt3d.open ("CtrlCloseLoop.csv");
        csvPt3d<<"Ecart"<<endl;
        CamStenope * aCam11 = aVCam1[0];
        Pt3dr aCen1= aCam11->VraiOpticalCenter();

        vector<double> aVEcart;
        vector<double> aVEcart_X;
        vector<double> aVEcart_Y;
        vector<double> aVEcart_Z;


        cPlyCloud aPlyERROR;
        Pt3di colPt(255,0,0);

        for (uint aKPt=0; aKPt < aPtCtrlLoop1.size(); aKPt++)
        {
            double ecart = euclid(aPtCtrlLoop1[aKPt] - aPtCtrlLoop2[aKPt]);
            csvPt3d<<ecart<<endl;
            aVEcart.push_back(ecart);
            Pt3dr aVEc = (aPtCtrlLoop1[aKPt] - aPtCtrlLoop2[aKPt]).AbsP();
            aVEcart_X.push_back(aVEc.x);
            aVEcart_Y.push_back(aVEc.y);
            aVEcart_Z.push_back(aVEc.z);

            // Ply ERROR
            aPlyERROR.AddPt(colPt, aPtCtrlLoop1[aKPt]);
            Pt3dr aNorm = -aCen1 + aPtCtrlLoop1[aKPt];
            aPlyERROR.AddSeg(colSeg, aPtCtrlLoop1[aKPt], aPtCtrlLoop1[aKPt] + aNorm*(ecart*aDynV), 500);
        }
        aPlyERROR.PutFile(aPlyErrName);
        csvPt3d .close();
        if (!aSilent)
        {
            cout<<"Total : "<<aPtCtrlLoop1.size()<<" pts in control"<<endl;
        }

        // Stat on aVEcart && Sortie pt cloud d'erreur
        {
            // Sort ascending
            sort(aVEcart.begin(), aVEcart.end(), sortAscending);
            sort(aVEcart_X.begin(), aVEcart_X.end(), sortAscending);
            sort(aVEcart_Y.begin(), aVEcart_Y.end(), sortAscending);
            sort(aVEcart_Z.begin(), aVEcart_Z.end(), sortAscending);

            double aSom = 0.0;
            Pt3dr aSom3D(0.0, 0.0, 0.0);

            int aNb = 0;

            vector<int> aVRangInd;
            int aPerInit = 10;
            int aPerFinal = 100;
            int aPerStep = 10;
            for (int aKPer=aPerInit; aKPer <= aPerFinal; aKPer = aKPer + aPerStep)
            {
                aVRangInd.push_back(round_down(aVEcart.size()*((double)aKPer/100.0))-1);
            }
            int indCur=0;

            vector<double> aCurEcart;
            for (uint aKE=0; aKE<aVEcart.size(); aKE++)
            {
                aSom += aVEcart[aKE];
                aSom3D.x += aVEcart_X[aKE];
                aSom3D.y += aVEcart_Y[aKE];
                aSom3D.z += aVEcart_Z[aKE];

                aNb++;
                aCurEcart.push_back(aVEcart[aKE]);
                if ((int)aKE == aVRangInd[indCur])
                {
                    //PrintOutStat.
                    if (!aSilent)
                    {
                    cout<<endl;
                    std::cout << "==== Stat: "<<ToString(100.0*aKE/aVEcart.size())<<"% ====" <<endl
                              << " Moy= " << aSom/aNb <<endl
                              << " Med=" << KthValProp(aCurEcart,0.5)   <<endl    // score median
                              << " 20%=" << KthValProp(aCurEcart,0.2)   <<endl    // score à 20% en premier
                              << " 80%=" << KthValProp(aCurEcart,0.8)   <<endl    // score à 20% en premier
                              << " Nb=" << aCurEcart.size()             <<endl
                              << "\n";
                    }
                    indCur++;
                }
            }

            if (!aSilent)
            {
                std::cout << "==== Stat on All: ====" <<endl
                          << " Moy= " << aSom/aNb <<endl
                          << " Med=" << KthValProp(aVEcart,0.5)   <<endl    // score median
                          << " 20%=" << KthValProp(aVEcart,0.2)   <<endl    // score à 20% en premier
                          << " 80%=" << KthValProp(aVEcart,0.8)   <<endl    // score à 20% en premier
                          << " Nb=" << aVEcart.size()             <<endl
                          << "\n";

                std::cout << "==== Stat on X: ====" <<endl
                          << " Moy= " << aSom3D.x/aNb <<endl
                          << " Med=" << KthValProp(aVEcart_X,0.5)   <<endl    // score median
                          << " 20%=" << KthValProp(aVEcart_X,0.2)   <<endl    // score à 20% en premier
                          << " 80%=" << KthValProp(aVEcart_X,0.8)   <<endl    // score à 20% en premier
                          << " Nb=" << aVEcart.size()             <<endl
                          << "\n";

                std::cout << "==== Stat on Y: ====" <<endl
                          << " Moy= " << aSom3D.y/aNb <<endl
                          << " Med=" << KthValProp(aVEcart_Y,0.5)   <<endl    // score median
                          << " 20%=" << KthValProp(aVEcart_Y,0.2)   <<endl    // score à 20% en premier
                          << " 80%=" << KthValProp(aVEcart_Y,0.8)   <<endl    // score à 20% en premier
                          << " Nb=" << aVEcart.size()             <<endl
                          << "\n";

                std::cout << "==== Stat on Z: ====" <<endl
                          << " Moy= " << aSom3D.z/aNb <<endl
                          << " Med=" << KthValProp(aVEcart_Z,0.5)   <<endl    // score median
                          << " 20%=" << KthValProp(aVEcart_Z,0.2)   <<endl    // score à 20% en premier
                          << " 80%=" << KthValProp(aVEcart_Z,0.8)   <<endl    // score à 20% en premier
                          << " Nb=" << aVEcart.size()             <<endl
                          << "\n";
            }
        }

        double scaleAuto = KthValProp(aVEcart,0.8);
        if (aSilent)
        {
            cout<< " 80% (3D)=" << KthValProp(aVEcart,0.8) <<" -Nb="<<aVEcart.size()<<endl;
            cout<< " 80% (X)=" << KthValProp(aVEcart_X,0.8) <<" -Nb="<<aVEcart.size()<<endl;
            cout<< " 80% (Y)=" << KthValProp(aVEcart_Y,0.8) <<" -Nb="<<aVEcart.size()<<endl;
            cout<< " 80% (Z)=" << KthValProp(aVEcart_Z,0.8) <<" -Nb="<<aVEcart.size()<<endl;
        }
        // plot data
        if (plot && aPtCtrlLoop1.size()>0)
        {
            ofstream scriptPlot;
            scriptPlot.open ("ScriptPlot.txt");
            scriptPlot<<"set key autotitle columnhead"<<endl;
            string cmdPlot;
            if (EAMIsInit(&seuilEcart))
            {
                cmdPlot =cmdPlot +  "plot " + "[ ] [0:"+  ToString(scaleAuto) +"] 'CtrlCloseLoop.csv' with boxes";
            }
            else
            {
                cmdPlot = "plot 'CtrlCloseLoop.csv' with boxes";
            }
            scriptPlot<<cmdPlot<<endl;
            System("gnuplot --persist ScriptPlot.txt");
        }
        ELISE_fp::RmFileIfExist("ScriptPlot.txt");
        return EXIT_SUCCESS;
}



// =================== Test Zone Init block rigid ============================

// Orientation 1 camera series (cam0 par ex)
// Init tout les autres cam par cam0
extern ElMatrix<double> ImportMat(const cTypeCodageMatr & aCM);
extern cTypeCodageMatr ExportMatr(const ElMatrix<double> & aMat);

class cOneTimeStamp
{
public:
    cOneTimeStamp(string aTimeStampId, string cleAssoc, cInterfChantierNameManipulateur * aICNM, cLiaisonsSHC * aLSHC);
    void AddCam(string aCamId, string KeyIm2TimeCam);
    void AddAllCamPossible(string KeyIm2TimeCam);
    void SetCamRef(CamStenope * aCamRef, string aCamRefId);
    void SetOri(string aOri);
    void SetOriOut(string aOriOut) {mOriOut = aOriOut;}
    CamStenope * CamRef() {return mCamRef;}


    string & Id() {return mTimeId;}
    string & Ori() {return mOri;}

    bool & IsInit() {return mIsInit;}
    cInterfChantierNameManipulateur * ICNM() {return mICNM;}
    vector<CamStenope*> & VCamInBlock() {return mVCamInBlock;}
    cLiaisonsSHC * LSHC() {return mLSHC;}

    void Export(string aNameIm, string aOriOut, Pt3dr Centre, ElMatrix<double> aRot);

private:

    string mTimeId;
    bool mIsInit;
    cInterfChantierNameManipulateur * mICNM;
    cLiaisonsSHC * mLSHC;
    vector<string> mVNameCamInBlock; // name of all image in the same time stamp
    vector<bool>   mVIsCamInBlockHasOriented;
    CamStenope * mCamRef;
    vector<CamStenope*> mVCamInBlock; // name of all image in the same time stamp
    string mCamRefId;
    cParamOrientSHC mCamRefOrientSHC;
    cOrientationConique mOriRef;
    string mOri;
    string mOriOut;
};

cOneTimeStamp::cOneTimeStamp(string aTimeStampId, string cleAssoc, cInterfChantierNameManipulateur * aICNM, cLiaisonsSHC *aLSHC):
    mTimeId (aTimeStampId),
    mIsInit (true),
    mICNM (aICNM),
    mLSHC (aLSHC),
    mCamRef (NULL),
    mVCamInBlock (mLSHC->ParamOrientSHC().size())
{
}

void cOneTimeStamp::SetOri(string aOri)
{
    mOri = aOri;
    cout<<"Set Ori "<<aOri<<" "<<mOri<<endl;
}

void cOneTimeStamp::SetCamRef(CamStenope * aCamRef, string aCamRefId)
{
    cout<<"   + Set Cam Ref "<<endl;
    if (mCamRef != NULL)
    {
        cout<<"WARN ! : this time stamp has already camRef : "<<mCamRef->NameIm()<<endl;
    }
    mCamRefId = aCamRefId;
    mCamRef = aCamRef;
    std::string aKey = "NKS-Assoc-Im2Orient@-"+ mOri;
    std::string aOriPath =  mICNM->Assoc1To1(aKey,aCamRef->NameIm(),true);
    if (ELISE_fp::exist_file(aOriPath))
    {
        cout<<"   + Get OrientationConique of Cam Ref "<<endl;
        mOriRef=StdGetFromPCP(aOriPath,OrientationConique);
    }
    else
    {
        ELISE_ASSERT(ELISE_fp::exist_file(aOriPath), aOriPath.c_str());
        return;
    }
    std::list< cParamOrientSHC >::iterator aK = mLSHC->ParamOrientSHC().begin();
    while (aK != mLSHC->ParamOrientSHC().end())
    {
        cParamOrientSHC aPr =  *aK;
        string aIdCam = aPr.IdGrp();
        if (aIdCam == mCamRefId)
        {
            cout<<"   + Get cParamOrientSHC of Cam Ref .. OK "<<endl;
            mCamRefOrientSHC = *aK;
            break;
        }
        aK++;
    }
}

void cOneTimeStamp::Export(string aNameIm, string aOriOut, Pt3dr Centre, ElMatrix<double> aRot)
{
    std::string aKey = "NKS-Assoc-Im2Orient@-"+ aOriOut;
    std::string aOriPath =  mICNM->Assoc1To1(aKey,aNameIm,true);

    cOrientationConique aExport=mOriRef;
    aExport.Externe().Centre() = Centre;

    cTypeCodageMatr aValRot;
    aValRot.L1() = Pt3dr(aRot(0,0), aRot(1,0), aRot(2,0));
    aValRot.L2() = Pt3dr(aRot(0,1), aRot(1,1), aRot(2,1));
    aValRot.L3() = Pt3dr(aRot(0,2), aRot(1,2), aRot(2,2));
    aExport.Externe().ParamRotation().CodageMatr().SetVal(aValRot);

    MakeFileXML(aExport, aOriPath);

    cout<<"     ++ Export XML "<<aOriPath<<endl;
}


void cOneTimeStamp::AddCam(string aCamId, string KeyIm2TimeCam)
{
    string aCamToFind = mICNM->Assoc1To2(KeyIm2TimeCam, mTimeId, aCamId, false);
    cout<<"   +Add Cam : "<<endl;
    cout<<"   + Assoc12 "<<mTimeId<<" + "<<aCamId<<" = "<<aCamToFind<<endl;
    std::vector<std::string> aSetIm2Find = *(mICNM->Get(aCamToFind));
    if (aSetIm2Find.size() > 1)
    {
        cout<<"   + WTF ? Impossible to have more than 1 camera ID "<<aCamId<<" in block "<<mTimeId<<endl;
        return;
    }
    if (aSetIm2Find.size() == 0)
    {
        cout<<"   + Query: camID "<<aCamId<<" with TimeStamp "<<mTimeId<<" not found !"<<endl;
        return;
    }
    string aIm2Find = aSetIm2Find[0];
    // check if cam is already added
    cout<<"   + Found Cam : "<<aIm2Find<<endl;
    vector<string>::iterator itFind;
    itFind = find(mVNameCamInBlock.begin(), mVNameCamInBlock.end(), aIm2Find);
    if (itFind == mVNameCamInBlock.end())
    {
        mVNameCamInBlock.push_back(aIm2Find);
        // check if cam is already orientated
        std::string aKey = "NKS-Assoc-Im2Orient@-"+ Ori();
        std::string aNameCam =  mICNM->Assoc1To1(aKey,aIm2Find,true);
        if (ELISE_fp::exist_file(aNameCam))
        {
            cout<<"   + Cam has already oriented. Keep existed orientation !"<<endl;
            CamStenope * aCam = mICNM->StdCamStenOfNames(aIm2Find, mOri);
            mVCamInBlock.push_back(aCam);

        }
        else
        {
            cout<<"   + Cam Init by block struture. "<<endl;
            // Init cam by blinis
            CamStenope * aCamToInit = new CamStenope(*mCamRef, mCamRef->Orient());

            aCamToInit->SetNameIm(aIm2Find);
            // From CamId, get ori relative in Block Structure
            std::list< cParamOrientSHC >::iterator aK = mLSHC->ParamOrientSHC().begin();
            while (aK != mLSHC->ParamOrientSHC().end())
            {
                cParamOrientSHC aPrCamToInit =  *aK;
                string aIdCam = aPrCamToInit.IdGrp();
                if (aIdCam == aCamId)
                {
                    cout<<"    + Cam struct found in Blinis"<<endl;
                    // CamToInit in BlockRef coordinate (R31)
                    ElMatrix<double> aPrRotCam = ImportMat(aPrCamToInit.Rot());
                    Pt3dr aPrTrCam = aPrCamToInit.Vecteur();
                    // CamRef in BlockRef coordinate (R21)
                    ElMatrix<double> aPrRotCamRef = ImportMat(mCamRefOrientSHC.Rot());
                    Pt3dr aPrTrCamRef = mCamRefOrientSHC.Vecteur();
                    // CamRef in world coordinate  (R2W)
                    ElMatrix<double> aRotCamRef(3);
                    aRotCamRef = ImportMat(mOriRef.Externe().ParamRotation().CodageMatr().Val());
                    Pt3dr aCenterCamRef = mOriRef.Externe().Centre();

                    // Compute CamToInit in world coordinate R3W = R31*(R21)t*R2W
                    ElMatrix<double> R21t = aPrRotCamRef.transpose();
                    ElMatrix<double> R31R21t(3);
                    R31R21t.mul(aPrRotCam, R21t);
                    ElMatrix<double> R3W(3);    // Rotation CamToInit
                    R3W.mul(R31R21t, aRotCamRef);

                    ElMatrix<double> TrInBlock(1,3);
                    TrInBlock(0,0) = (-aPrTrCam + aPrTrCamRef).x;
                    TrInBlock(0,1) = (-aPrTrCam + aPrTrCamRef).y;
                    TrInBlock(0,2) = (-aPrTrCam + aPrTrCamRef).z;

                    ElMatrix<double> aCenterCamRefMat(1,3);
                    aCenterCamRefMat(0,0) = aCenterCamRef.x;
                    aCenterCamRefMat(0,1) = aCenterCamRef.y;
                    aCenterCamRefMat(0,2) = aCenterCamRef.z;

                    ElMatrix<double> aCenterCamToInit = aRotCamRef.transpose()*TrInBlock + aCenterCamRefMat;
                    Pt3dr aPtCenterCamToInit (aCenterCamToInit(0,0), aCenterCamToInit(0,1), aCenterCamToInit(0,2));

                    Export(aIm2Find , mOriOut , aPtCenterCamToInit, R3W);
                    // Store result in aCamToInit ???
                    //aCamToInit->SetIncCentre(aCenterCamToInit);
                    //aCamToInit->SetOrientation(R3W);
                    // cout pour verifier
                    double aL1_Cam1[3];R3W.GetLine(0,aL1_Cam1);
                    double aL2_Cam1[3];R3W.GetLine(1,aL2_Cam1);
                    double aL3_Cam1[3];R3W.GetLine(2,aL3_Cam1);
                    cout<<"    + Rot: "<<endl
                      <<"      "<<aL1_Cam1[0]<<" "<<aL1_Cam1[1]<<" "<<aL1_Cam1[2]<<endl
                      <<"      "<<aL2_Cam1[0]<<" "<<aL2_Cam1[1]<<" "<<aL2_Cam1[2]<<endl
                      <<"      "<<aL3_Cam1[0]<<" "<<aL3_Cam1[1]<<" "<<aL3_Cam1[2]<<endl
                      <<endl;
                    cout<<"    + Center = "<<aPtCenterCamToInit<<endl;


                    mVCamInBlock.push_back(aCamToInit);
                    cout<<"    + Init OK ! "<<endl;
                    break;
                }
                aK++;
            }
        }
    }
    else
    {
        cout<<"    + Cam has already added ! Quit "<<endl;
    }
    return;
}


void cOneTimeStamp::AddAllCamPossible(string KeyIm2TimeCam)
{
    // from a time stamp & list of camId, get all Image existed
    std::list< cParamOrientSHC >::iterator aK = mLSHC->ParamOrientSHC().begin();
    cout<<"   + Search all possible cam to add "<<endl;
    while (aK != mLSHC->ParamOrientSHC().end())
    {
        cParamOrientSHC aPr =  *aK;
        string aIdCam = aPr.IdGrp();
        if (aIdCam != mCamRefId)
        {
            this->AddCam(aIdCam, KeyIm2TimeCam);
        }
        aK++;
    }
}


int Test_InitBloc(int argc, char ** argv)
{
    string aDir = "./";
    string aPat, aPattern;
    string aOriA;
    string aBlinisPath = "./Blinis_Camp_Test_Blinis_GCP.xml";
    string aOriOut = "OutInitBloc";

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPattern, "PatIm",  eSAM_IsPatFile)
                      << EAMC(aOriA, "Ori",  eSAM_IsExistDirOri)
                      << EAMC(aBlinisPath, "BlinisStructCamFile" , eSAM_IsExistFile),
          LArgMain()
                      << EAM(aOriOut, "Out" , true, "Ori Out Folder, def=OutInitBloc")
                );

    SplitDirAndFile(aDir, aPat, aPattern);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    // read Blinis XML
    cStructBlockCam aSBC = StdGetFromPCP(aBlinisPath, StructBlockCam); //xml blinis
    cout<<aSBC.KeyIm2TimeCam()<<endl; // recuperer le cle assoc
    cLiaisonsSHC * aLSHC = aSBC.LiaisonsSHC().PtrVal();
    std::list< cParamOrientSHC >::iterator aK = aLSHC->ParamOrientSHC().begin();
    vector<string>aVIdCam;
    while (aK != aLSHC->ParamOrientSHC().end())
    {
        cParamOrientSHC aPr =  *aK;
        string aIdCam = aPr.IdGrp(); // $1 in image name cle inverse
        aVIdCam.push_back(aIdCam);
        ElMatrix<double> aRot = ImportMat(aPr.Rot());
        aK++;
    }
    // $2 in image name cle inverse

    // Pattern of image to init
    vector<string>  aSetIm = *(aICNM->Get(aPat));
    // Get orientation of all oriented image
    vector<CamStenope*> aVCam;
    StdCorrecNameOrient(aOriA, aICNM->Dir());
    aOriOut = "Ori-" + aOriOut;
    ELISE_fp::MkDirSvp(aOriOut);
    StdCorrecNameOrient(aOriOut, aICNM->Dir(), true);
    for (uint aKIm=0; aKIm<aSetIm.size(); aKIm++)
    {
        string aNameIm = aSetIm[aKIm];
        std::string aKey = "NKS-Assoc-Im2Orient@-"+ aOriA ;
        std::string aNameCam =  aICNM->Assoc1To1(aKey,aNameIm,true);
        if (ELISE_fp::exist_file(aNameCam))
        {
            CamStenope * aCam = aICNM->StdCamStenOfNames(aNameIm, aOriA);
            aCam->SetNameIm(aNameIm);
            aVCam.push_back(aCam);
        }
        else
        {
            cout<<" ++ "<<aNameCam<<" not existe in "<<aOriA<<endl;
        }

    }

    // On initialise les camera
    cout<<endl<<"Init les cam.."<<endl;
    std::map<std::string,cOneTimeStamp *> mMap_TimeId_cOneTimeStamp;
    std::vector<cOneTimeStamp *>          mVTimeStamp; // size = nb cam in block
    cout<<"Key Assoc : "<<aSBC.KeyIm2TimeCam()<<endl;
    cout<<"BEGIN FUSION : "<<endl;
    for (int aKP=0 ; aKP<int(aVCam.size()) ; aKP++)
    {
        // for all of oriented image
        CamStenope * aPC = aVCam[aKP];
        std::string aNamePose = aPC->NameIm();
        cout<<endl<<"++ NameIm : "<<aNamePose<<endl;

        // From name Im, get $1 et $2 (time stamp & CamID)
        std::pair<std::string,std::string> aPair = aICNM->Assoc2To1(aSBC.KeyIm2TimeCam(),aNamePose,true);
        cout<<"   + Assoc21 "<<aPair.first<<" + "<<aPair.second<<endl;
        string nTimeId = aPair.first;
        std::string nCamId = aPair.second;

        // Creat a time stamp nBlockId if it is not existed
        cOneTimeStamp *  aTimeStamp = NULL;
        if (! DicBoolFind( mMap_TimeId_cOneTimeStamp, nTimeId))
        {
            cout<<"   ++ Create Time Stamp "<<nTimeId<<endl;
            aTimeStamp = new cOneTimeStamp ( nTimeId,
                                             aSBC.KeyIm2TimeCam(),
                                             aICNM,
                                             aLSHC
                                             );
            aTimeStamp->SetOri(aOriA);
            aTimeStamp->SetOriOut(aOriOut);
            mMap_TimeId_cOneTimeStamp.insert(std::pair<string, cOneTimeStamp*>(nTimeId , aTimeStamp));
            aTimeStamp->SetCamRef(aPC, nCamId);
            aTimeStamp->AddAllCamPossible(aSBC.KeyIm2TimeCam());
            continue;
        }
        else
        {   // if time stamp is already existe
            std::map<std::string,cOneTimeStamp *>::iterator itFind;
            itFind = mMap_TimeId_cOneTimeStamp.find(nTimeId);
            // get TimeStamp

            // Add cam to TimeStamp (normally it has already added)
            if (itFind != mMap_TimeId_cOneTimeStamp.end())
            {
                cout<<"   + Time Stamp founded ! "<<endl;
                aTimeStamp = itFind->second;
                aTimeStamp->AddCam(nCamId, aSBC.KeyIm2TimeCam());

            }
            else
                cout<<" PutainNNNNNNNNN"<<endl;
        }
    }

    cout<<endl<<" ENDDDD TEST FUSIONNNN"<<endl;
    cout<<"QUIT INIT BLOC"<<endl;
return EXIT_SUCCESS;
}




// =============================================================================

// =================== Test Zone Detect image blur ============================
Im2D_REAL4 ImRead(string aNameImTif, int mRech = 1)
{
   Tiff_Im aTif = Tiff_Im::UnivConvStd(aNameImTif);

   Pt2di aSz = Pt2di(Pt2dr(aTif.sz())/double(mRech));

   Im2D_REAL4 aI(aSz.x, aSz.y);

   ELISE_COPY
   (
       aI.all_pts(),
       aTif.in(),
       aI.out()
   );
   return aI;
}

void SaveTif(Im2D_REAL4 aIm, string aSaveName)
{

        string aName =  std::string("./") + aSaveName + ".tif";

        L_Arg_Opt_Tiff aL = Tiff_Im::Empty_ARG;
        aL = aL + Arg_Tiff(Tiff_Im::ANoStrip());
        Tiff_Im aRes
                (
                   aName.c_str(),
                   aIm.sz(),
                   GenIm::u_int1,
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero,
                   aL
                );
        ELISE_COPY
        (
           aIm.all_pts(),
           aIm.in(),
           aRes.out()
        );

}

void ReSample(Im2D_REAL4 & aIm, Im2D_REAL4 & aImOut, double mRech = 1.0)
{
    aImOut.Resize(Pt2di(aIm.sz()*mRech));
    if (mRech == 1.0)
    {
        aImOut = aIm;
        return;
    }
    if (mRech < 1.0)
    {
        ELISE_COPY
        (
            aImOut.all_pts(),
            aIm.in()[Virgule(FX/mRech,FY/mRech)],
            aImOut.out()
        );
        return;
    }
    if (mRech > 1.0)
    {
        Pt2dr aRabInterpol(1,1);
        ELISE_COPY
        (
            select(  aImOut.all_pts(),
                     (
                          (FX<(aImOut.sz().x - aRabInterpol.x -mRech ))
                       && (FY<(aImOut.sz().y - aRabInterpol.y -mRech ))
                     )
                  ),
            aIm.in()[Virgule(FX/mRech,FY/mRech)],
            aImOut.out()
        );
        return;
    }
}

void Test_ELISE(string aNameImTif, double mRech = 1)
{
    Tiff_Im aTif = Tiff_Im::UnivConvStd(aNameImTif);

    Pt2di aSz = Pt2di(Pt2dr(aTif.sz())*double(mRech));

    Im2D_REAL4 aI(aTif.sz().x, aTif.sz().y);
    Im2D_REAL4 aO(aSz.x, aSz.y);

    Video_Win * mW = Video_Win::PtrWStd(aO.sz(),true,Pt2dr(1,1));

    mW->set_sop(Elise_Set_Of_Palette::TheFullPalette());

    ELISE_COPY
    (
        aI.all_pts(),
        aTif.in(),
        aI.out()
    );
// re-sample aI to aO
    if (mRech <= 1.0)
    {
        ELISE_COPY
        (
            aO.all_pts(),
            aI.in()[Virgule(FX/mRech,FY/mRech)],
            aO.out()
        );
    }
    else
    {
        Pt2dr aRabInterpol(1,1);
        ELISE_COPY
        (
            select(  aO.all_pts(),
                     (
                          (FX<(aO.sz().x - aRabInterpol.x -mRech ))
                       && (FY<(aO.sz().y - aRabInterpol.y -mRech ))
                     )
                  ),
            aI.in()[Virgule(FX/mRech,FY/mRech)],
            aO.out()
        );
    }
    SaveTif(aO, aNameImTif+"_TestELISE" );
//display aO
    ELISE_COPY
    (
        aO.all_pts(),
        aO.in()[Virgule(FX,FY)],
        mW->ogray()
    );
}


void Show(Im2D_REAL4 aIm,Fonc_Num aF, Im2D_REAL4 & aImOut, string aSaveName = "")
{
    ELISE_COPY
    (
       aIm.all_pts(),
       Max(0,Min(255,aF)),
       aImOut.out()
    );
    if (aSaveName != "")
    {
        string aName =  std::string("./") + aSaveName + ".tif";

        L_Arg_Opt_Tiff aL = Tiff_Im::Empty_ARG;
        aL = aL + Arg_Tiff(Tiff_Im::ANoStrip());
        Tiff_Im aRes
                (
                   aName.c_str(),
                   aIm.sz(),
                   GenIm::u_int1,
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero,
                   aL
                );
        ELISE_COPY
        (
           aIm.all_pts(),
           Max(0,Min(255,aF)),
           aRes.out()
        );
    }
}

void Show_LAP2(Im2D_REAL4 aIm,Fonc_Num aF, Im2D_REAL4 & aImOut, string aSaveName = "")
{
    ELISE_COPY
    (
       aIm.all_pts(),
       ElAbs(aF),
       aImOut.out()
    );
    if (aSaveName != "")
    {
        string aName =  std::string("./") + aSaveName + ".tif";

        L_Arg_Opt_Tiff aL = Tiff_Im::Empty_ARG;
        aL = aL + Arg_Tiff(Tiff_Im::ANoStrip());
        Tiff_Im aRes
                (
                   aName.c_str(),
                   aIm.sz(),
                   GenIm::u_int1,
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero,
                   aL
                );
        ELISE_COPY
        (
           aIm.all_pts(),
           Max(0,Min(255,aF)),
           aRes.out()
        );
    }
}


double Conv1Cell(Im2D_REAL4 & aImgIn, Im2D_REAL8 & aKer, Pt2di & aPos, Pt2di & aSzKer, double & aSomker)
{
    double aSom=0;
    for (int aKx=-aSzKer.x; aKx<=aSzKer.x; aKx++)
    {
        for (int aKy=-aSzKer.y; aKy<=aSzKer.y; aKy++)
        {
            Pt2di aVois(aKx, aKy);
            aSom += aImgIn.GetI(aPos + aVois) * aKer.GetI(aVois + aSzKer);
            //cout<<"Img "<<(aPos + aVois)<<aImgIn.GetI(aPos + aVois)<<" -aKer "<<(aVois + aSzKer)<<aKer.GetI(aVois + aSzKer)<<endl;
        }
    }
    return (aSom/aSomker);
}

double Convol_Withker(Im2D_REAL4 & aImgIn, Im2D_REAL8 & aKer, Im2D_REAL4 & aImgOut)
{
    aImgOut.Resize(aImgIn.sz());
    Pt2di aSzKer(round_up((aKer.sz().x-1)/2), round_up((aKer.sz().y-1)/2));
    Pt2di aRun;

    double aSomKer = aKer.som_rect();
    if (aSomKer == 0)
        aSomKer = 1;
    double Moy = 0;
    int aCnt = 0;

    for (aRun.x = aSzKer.x ;aRun.x < aImgIn.sz().x-aSzKer.x; aRun.x++)
    {
        for (aRun.y = aSzKer.y ;aRun.y < aImgIn.sz().y-aSzKer.y; aRun.y++)
        {
            double aRes = Conv1Cell(aImgIn, aKer, aRun, aSzKer, aSomKer);
            Moy += aRes;
            aCnt++;
            aImgOut.SetI_SVP(aRun, aRes);
        }
    }
    return Moy/aCnt;
}

double Average(Im2D_REAL4 & aImgIn, Pt2di aRab = Pt2di(0,0))
{
    Pt2di aRun;
    double aMoy{0};
    int aCnt{0};
    for (aRun.x = aRab.x; aRun.x < aImgIn.sz().x - aRab.x; aRun.x++)
    {
        for (aRun.y = aRab.y; aRun.y < aImgIn.sz().y - aRab.y; aRun.y++)
        {
           aMoy +=  aImgIn.GetI(aRun);
           aCnt++;
        }
    }
    return (aMoy/aCnt);
}


double Variance(Im2D_REAL4 & aImgIn, double aMoy = 0, Pt2di aRab = Pt2di(0,0))
{
    double Moy=aMoy;
    if (aMoy == 0)
    {
       Moy = aImgIn.moy_rect(Pt2dr(aRab), Pt2dr(aImgIn.sz()-(aRab+Pt2di(1,1))) );
    }
    Pt2di aRun;
    double aSumEcart{0};
    int aCnt{0};
    for (aRun.x = aRab.x; aRun.x < aImgIn.sz().x - aRab.x; aRun.x++)
    {
        for (aRun.y = aRab.y; aRun.y < aImgIn.sz().y - aRab.y; aRun.y++)
        {
           aSumEcart +=  ElSquare(aImgIn.GetI(aRun)-Moy);
           aCnt++;
        }
    }
    return (aSumEcart/aCnt);
}



// ====== Focus measurement operator =====

Im2D_REAL4 Convol_With_ELISE(string aImIn, Im2D_REAL8 & aKer)
{
    Im2D_REAL4 aIm2D = ImRead(aImIn);
    Im2D_REAL4 aIm2D_DNs(aIm2D.sz().x, aIm2D.sz().y);

    Im2D_REAL4 aRes(aIm2D.tx(),aIm2D.ty());
    Fonc_Num aF = aIm2D.in(0);
    double som_Ker = aKer.som_rect();
    if (som_Ker == 0)
        som_Ker=1.0;
   ELISE_COPY(aRes.all_pts(),som_masq(aF, aKer)/som_Ker,aRes.out());
    return aRes;
}

Im2D_REAL4 Convol_With_ELISE(Im2D_REAL4 & aImIn, Im2D_REAL8 & aKer)
{
    Im2D_REAL4 aRes(aImIn.tx(),aImIn.ty());
    Fonc_Num aF = aImIn.in(0);
    double som_Ker = aKer.som_rect();
    if (som_Ker == 0)
        som_Ker=1.0;
    ELISE_COPY(aRes.all_pts(),som_masq(aF, aKer)/som_Ker,aRes.out());
    return aRes;
}

double VarOfLap_LAP4(string aNameIm, double rech = 0.5)
{
    // Variance of Laplacian
    cout<<" + Im : "<<aNameIm;
    ElTimer aTimer;
    Im2D_REAL8 aLapl(3,3,
                        "0 1 0 "
                        "1 -4 1 "
                        " 0 1 0"
                   );
    Im2D_REAL8 aDenoise(3,3,
                        "1 1 1 "
                        "1 1 1 "
                        " 1 1 1"
                        );

    Pt2di aSzKer(round_up((aLapl.sz().x-1)/2), round_up((aLapl.sz().y-1)/2));

    Im2D_REAL4 aIm2D = ImRead(aNameIm);

    Im2D_REAL4 aIm2D_DNs(aIm2D.sz().x, aIm2D.sz().y);

    aIm2D_DNs = Convol_With_ELISE(aIm2D, aDenoise);

    Im2D_REAL4 aIm2D_Rsz;

    ReSample(aIm2D, aIm2D_Rsz, rech);

    //SaveTif(aIm2D_Rsz, aNameIm + "_Rsz");


    Im2D_REAL4 aIm2D_Lpl(aIm2D_Rsz.sz().x, aIm2D_Rsz.sz().y);
    aIm2D_Lpl = Convol_With_ELISE(aIm2D_DNs, aLapl);
    double aVar = Variance(aIm2D_Lpl, 0, aSzKer);
    cout<<" -Var :"<<aVar<<endl;
    return aVar;
}

double VarOfLap_LAP4_G(string aNameIm)
{
    // Variance of Laplacian
    ElTimer aTimer;
    Im2D_REAL8 aLapl(3,3,
                        "0 1 0 "
                        "1 -4 1 "
                        " 0 1 0"
                   );
    Im2D_REAL8 aDenoise(3,3,
                        "1 1 1 "
                        "1 1 1 "
                        " 1 1 1"
                        );
    Pt2di aSzKer(round_up((aLapl.sz().x-1)/2), round_up((aLapl.sz().y-1)/2));
    Im2D_REAL4 aIm2D = ImRead(aNameIm);
    Im2D_REAL4 aIm2D_DNs(aIm2D.sz().x, aIm2D.sz().y);
    Convol_Withker(aIm2D, aDenoise, aIm2D_DNs);
    SaveTif(aIm2D_DNs, aNameIm + "_DnsG");

    Im2D_REAL4 aIm2D_Lpl(aIm2D.sz().x, aIm2D.sz().y);
    Convol_Withker(aIm2D_DNs, aLapl, aIm2D_Lpl);
    double aVar = Variance(aIm2D_Lpl, 0, aSzKer);
    SaveTif(aIm2D_Lpl, aNameIm + "_LplG");


    return aVar;
}

double ModifLap_LAP2 (string aNameIm)
{
    // Modified Laplacian
    ElTimer aTimer;
    Im2D_REAL8 aLapl_x(1,3,
                        "-1 2 -1"
                   );
    Im2D_REAL8 aLapl_y(3,1,
                        "-1 "
                        "2 "
                        "-1"
                   );
    Im2D_REAL8 aDenoise(3,3,
                        "1 1 1 "
                        "1 1 1 "
                        " 1 1 1"
                        );

    Pt2di aSzKer(1,1);


    Im2D_REAL4 aIm2D = ImRead(aNameIm);
    Im2D_REAL4 aIm2D_DNs(aIm2D.sz().x, aIm2D.sz().y);
    aIm2D_DNs = Convol_With_ELISE(aIm2D, aDenoise);

    Im2D_REAL4 aIm2D_LplX(aIm2D.sz().x, aIm2D.sz().y);
    Im2D_REAL4 aIm2D_LplY(aIm2D.sz().x, aIm2D.sz().y);
    aIm2D_LplX = Convol_With_ELISE(aIm2D_DNs, aLapl_x);
    aIm2D_LplY = Convol_With_ELISE(aIm2D_DNs, aLapl_y);

    Im2D_REAL4 aIm2D_LplSum(aIm2D.sz().x, aIm2D.sz().y);
    aIm2D_LplX.bitwise_add(aIm2D_LplY, aIm2D_LplSum);
    double aScore = aIm2D_LplSum.som_rect(Pt2dr(aSzKer), Pt2dr(aIm2D_LplSum.sz() - (aSzKer + Pt2di(1,1))));
    return aScore;
}

double DiagonalLap_LAP3 (string aNameIm)
{
    // Diagonal Laplacian
    ElTimer aTimer;
    Im2D_REAL8 aLapl_x(1,3,
                        "-1 2 -1"
                   );
    Im2D_REAL8 aLapl_y(3,1,
                        "-1 "
                        "2 "
                        "-1"
                   );
    Im2D_REAL8 aLapl_x1(3,3,
                        "0 0 1 "
                        "0 -2 0 "
                        " 1 0 0"
                   );
    Im2D_REAL8 aLapl_x2(3,3,
                        "1 0 0 "
                        "0 -2 0 "
                        " 0 0 1"
                   );
    Im2D_REAL8 aDenoise(3,3,
                        "1 1 1 "
                        "1 1 1 "
                        " 1 1 1"
                        );

    Pt2di aSzKer(1,1);
    double aFac = 1.0/sqrt(2);
    aLapl_x1.multiply(aFac);
    aLapl_x2.multiply(aFac);

    Im2D_REAL4 aIm2D = ImRead(aNameIm);
    Im2D_REAL4 aIm2D_DNs(aIm2D.sz().x, aIm2D.sz().y);
    aIm2D_DNs = Convol_With_ELISE(aIm2D, aDenoise);


    Im2D_REAL4 aIm2D_LplX(aIm2D.sz().x, aIm2D.sz().y);
    Im2D_REAL4 aIm2D_LplY(aIm2D.sz().x, aIm2D.sz().y);
    Im2D_REAL4 aIm2D_Lplx1(aIm2D.sz().x, aIm2D.sz().y);
    Im2D_REAL4 aIm2D_Lplx2(aIm2D.sz().x, aIm2D.sz().y);
    aIm2D_LplX = Convol_With_ELISE(aIm2D_DNs, aLapl_x);
    aIm2D_LplY = Convol_With_ELISE(aIm2D_DNs, aLapl_y);
    aIm2D_Lplx1 = Convol_With_ELISE(aIm2D_DNs, aLapl_x1);
    aIm2D_Lplx2 = Convol_With_ELISE(aIm2D_DNs, aLapl_x2);

    Im2D_REAL4 aIm2D_LplSum(aIm2D.sz().x, aIm2D.sz().y);
    aIm2D_LplX.bitwise_add(aIm2D_LplY, aIm2D_LplSum);
    aIm2D_LplSum.bitwise_add(aIm2D_Lplx1, aIm2D_LplSum);
    aIm2D_LplSum.bitwise_add(aIm2D_Lplx2, aIm2D_LplSum);

    double aScore = aIm2D_LplSum.som_rect(Pt2dr(aSzKer), Pt2dr(aIm2D_LplSum.sz() - (aSzKer + Pt2di(1,1))));
    return aScore;
}


// ========================================
int Test_Conv(int argc,char ** argv)
{

    Im2D_REAL8 aDenoise(3,3,
                        "1 1 1 "
                        "1 1 1 "
                        " 1 1 1"
                        );

    string aDir = "./";
    string aPat, aPattern;
    double rech = 0.5;

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPattern, "PatIm",  eSAM_IsPatFile),
          LArgMain()
                      << EAM(rech, "rech" , true, "re sample im befor compute (faster) def=0.5 - 2 times smaller")

    );

    SplitDirAndFile(aDir, aPat, aPattern);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    vector<string>  aSetIm = *(aICNM->Get(aPat));
    vector<Pt2dr> aVPair;
    for (uint aKImg=0; aKImg<aSetIm.size(); aKImg++)
    {
        // ====== test convolution function ======
        string aIm = aSetIm[aKImg];
        double aVar = VarOfLap_LAP4(aIm, rech);
        Pt2dr aPair(double(aKImg), aVar);
        aVPair.push_back(aPair);
    }
    sortDescendPt2drY(aVPair);

    cout<<endl<<"+ Sort by sharpness (higher is sharper) : "<<endl;
    for (uint aK=0; aK<aVPair.size(); aK++)
    {
        cout<<" + "<<aSetIm[int(aVPair[aK].x)]<<" - Var "<<aVPair[aK].y<<endl;
    }
    return 0;
}
// ===================  ============================

void Test_Xml()
{
    cXml_TriAngulationImMaster aTriangulation = StdGetFromSI("Tri0.xml",Xml_TriAngulationImMaster);
    std::cout << "Name master " << aTriangulation.NameMaster() << "\n";
    cXml_Triangle3DForTieP aTri;
    aTri.P1() = Pt3dr(1,1,1);
    aTri.P2() = Pt3dr(1,1,2);
    aTri.P3() = Pt3dr(1,1,3);
    aTri.NumImSec().push_back(1);

    aTriangulation.Tri().push_back(aTri);


    aTriangulation.NameSec().push_back("toto.tif");

    MakeFileXML(aTriangulation,"Tri1.xml");
    MakeFileXML(aTriangulation,"Tri1.dmp");

    aTriangulation = StdGetFromSI("Tri1.dmp",Xml_TriAngulationImMaster);

     std::cout << "Nb tri " <<  aTriangulation.Tri().size()  << " UnPt " << aTriangulation.Tri()[1].P2() << "\n";


    exit(EXIT_SUCCESS);
}

void Test_FAST()
{
    Tiff_Im * mPicTiff = new Tiff_Im ( Tiff_Im::StdConvGen("./Test.tif",1,false));
    Pt2di mImgSz = mPicTiff->sz();
    TIm2D<double,double> * mPic_TIm2D = new TIm2D<double,double> (mPicTiff->sz());
    ELISE_COPY(mPic_TIm2D->all_pts(), mPicTiff->in(), mPic_TIm2D->out());
    //Im2D<double,double> * mPic_Im2D = new Im2D<double, double> (mPic_TIm2D->_the_im);

    Im2D_Bits<1> aMasq0  = Im2D_Bits<1>(mImgSz.x,mImgSz.y,1);
    TIm2DBits<1> TaMasq0 = TIm2DBits<1> (aMasq0);

    FastNew *aDec = new FastNew(*mPic_TIm2D , 15 , 3 , TaMasq0);
    cout<<aDec->lstPt().size()<<" pts detected "<<endl;
}
// ============== Test Draw Rectangle on PLY ===============
void JSON_WritePt3D(Pt3dr aPt, ofstream & aJSONOut, Pt3dr aOffSet = Pt3dr(0,0,0), bool isEnd=false)
{
    aPt = aPt + aOffSet;
    aJSONOut.precision(16); // std::ios_base::precision
    aJSONOut<<"{"<<endl;
    aJSONOut<<" \"type\": \"Feature\","<<endl;
    aJSONOut<<" \"geometry\": {"<<endl;
    aJSONOut<<"  \"type\": \"Point\","<<endl;
    aJSONOut<<"  \"coordinates\": ["<<std::fixed<<aPt.x<<", "<<aPt.y<<"]"<<endl;
    aJSONOut<<" },"<<endl;
    aJSONOut<<" \"properties\": {"<<endl;
    aJSONOut<<"     \"Z\":"<<aPt.z<<endl;
    aJSONOut<<" }"<<endl;
    if (isEnd)
        aJSONOut<<"}"<<endl;
    else
        aJSONOut<<"},"<<endl;
    return;
}

void JSON_WritePoly(vector<Pt3dr> aPoly, ofstream & aJSONOut, Pt3dr aOffSet = Pt3dr(0,0,0), bool isEnd=false)
{
    aJSONOut.precision(16); // std::ios_base::precision
    aJSONOut<<"{"<<endl;
    aJSONOut<<" \"type\": \"Feature\","<<endl;
    aJSONOut<<" \"geometry\": {"<<endl;
    aJSONOut<<"  \"type\": \"Polygon\","<<endl;
    aJSONOut<<"  \"coordinates\": ["<<endl;
    aJSONOut<<"    ["<<endl;
    for(uint aKP=0; aKP<aPoly.size(); aKP++)
    {
        Pt3dr aPt = aPoly[aKP];

        aPt = aPt + aOffSet;
        aJSONOut<<"     ["<<std::fixed<<aPt.x<<", "<<aPt.y<<", "<<aPt.z<<"]";
        aJSONOut<<","<<endl;
    }
    aJSONOut<<"     ["<<std::fixed<<aPoly[0].x<<", "<<aPoly[0].y<<", "<<aPoly[0].z<<"]";
    aJSONOut<<endl;
    aJSONOut<<"    ]"<<endl;
    aJSONOut<<" ]"<<endl;
    aJSONOut<<" }"<<endl;
    aJSONOut<<"},"<<endl;
    return;
}


void DrawOneFootPrintToPly(CamStenope * aCam,
                           string & aNameIm,
                           cPlyCloud & aCPlyRes,
                           Pt3di aCoul,
                           Pt2dr aResolution,
                           ofstream & aJSONOut,
                           bool isEnd=false,
                           Pt3dr aOffSetPly = Pt3dr(0,0,0),
                           Pt3dr aOffSetGeoJSON = Pt3dr(0,0,0),
                           bool aPlyEachImg = false)
{
    std::string aPathPly = "./PLYFootPrint/" + aNameIm + "_FP.ply"; // We can't get name Image from CamStenope !!!

    cPlyCloud aPly;
    cElPolygone aPolyEmprise = aCam->EmpriseSol();
    Pt3dr aCamCentre = aCam->VraiOpticalCenter();
    JSON_WritePt3D(aCamCentre, aJSONOut, aOffSetGeoJSON, false);

    //if (aPlyEachImg)
        //aPly.AddSphere(aCoul, aCamCentre, aCam->GetAltiSol()/30 ,20);
    aCPlyRes.AddSphere(aCoul, aCamCentre, aCam->GetAltiSol()/90 ,aResolution.y);

    list<cElPolygone::tContour> aContours =  aPolyEmprise.Contours();
    list<cElPolygone::tContour>::iterator it = aContours.begin();
    vector<Pt3dr> aPolyEmpreint;
    for (; it!=aContours.end(); it++)
    {
        cElPolygone::tContour aCon = *it;
        for (uint aK = 0; aK<aCon.size(); aK++)
        {
            cout<<aCon[aK]<<" ";
            aPolyEmpreint.push_back(Pt3dr(aCon[aK].x, aCon[aK].y, aCam->GetAltiSol()));
        }
    }
    cout<<endl;
    ELISE_ASSERT(aPolyEmpreint.size() == 4, "Polygon Empreint != 4");
    aCPlyRes.AddSeg(aCoul, aPolyEmpreint[0] + aOffSetPly, aPolyEmpreint[1] + aOffSetPly, aResolution.x);
    aCPlyRes.AddSeg(aCoul, aPolyEmpreint[1] + aOffSetPly, aPolyEmpreint[2] + aOffSetPly, aResolution.x);
    aCPlyRes.AddSeg(aCoul, aPolyEmpreint[2] + aOffSetPly, aPolyEmpreint[3] + aOffSetPly, aResolution.x);
    aCPlyRes.AddSeg(aCoul, aPolyEmpreint[3] + aOffSetPly, aPolyEmpreint[0] + aOffSetPly, aResolution.x);
    JSON_WritePoly(aPolyEmpreint, aJSONOut, aOffSetGeoJSON, isEnd);

    if (aPlyEachImg)
    {
        aPly.AddSeg(aCoul, aPolyEmpreint[0] + aOffSetPly, aPolyEmpreint[1] + aOffSetPly, aResolution.x);
        aPly.AddSeg(aCoul, aPolyEmpreint[1] + aOffSetPly, aPolyEmpreint[2] + aOffSetPly, aResolution.x);
        aPly.AddSeg(aCoul, aPolyEmpreint[2] + aOffSetPly, aPolyEmpreint[3] + aOffSetPly, aResolution.x);
        aPly.AddSeg(aCoul, aPolyEmpreint[3] + aOffSetPly, aPolyEmpreint[0] + aOffSetPly, aResolution.x);
        aPly.PutFile(aPathPly);

    }

    return;
}

int DroneFootPrint(int argc,char ** argv)
{
    cout<<"********************************************************"<<endl;
    cout<<"*    Draw footprint from image + orientation           *"<<endl;
    cout<<"********************************************************"<<endl;


    string aDir = "./";
    string aPat, aPattern;
    string aOri;
    string aOutPly = "FootPrint.ply";
    bool aPlyEachImg = false;
    Pt2dr aResolution = Pt2dr(2000,20); //[resol_line, resol_sphere]
    int aCodeProj = 2154;
    Pt3dr aOffSetPLY = Pt3dr(0,0,0);
    Pt3dr aOffSetGeoJSON = Pt3dr(0,0,0);

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPattern, "PatIm",  eSAM_IsPatFile)
                      << EAMC(aOri, "Ori",  eSAM_IsExistDirOri),
          LArgMain()
                      << EAM(aOutPly, "Out" , true, "PLY output - def=FootPrint.ply & FootPrint.ply.geojson (QGIS Format)")
                      << EAM(aPlyEachImg, "PlyEachImg" , true, "PLY output separately for each image - directory PLYFootPrint")
                      << EAM(aOffSetPLY, "OffSetPLY" , true, "OffSet for PLY [X,Y,Z]")
                      << EAM(aOffSetGeoJSON, "OffSetGeoJSON" , true, "OffSet for geo JSON file [X,Y,Z]")
                      << EAM(aResolution, "Resol" , true, "Resolution of line and sphere [resol_line, resol_sphere], def=[2000,20]")
                      << EAM(aCodeProj, "CodeProj" , true, "EPSG projection code. Default = 2154 (Lambert 93)")
                );

    SplitDirAndFile(aDir, aPat, aPattern);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    vector<string>  aSetIm = *(aICNM->Get(aPat));
    StdCorrecNameOrient(aOri, aICNM->Dir());

    vector<CamStenope*> aVCam (aSetIm.size());
    cPlyCloud aCPlyRes;



    if (aPlyEachImg)
    {
        ELISE_fp::MkDir("./PLYFootPrint");
    }

    ofstream aJSONOut;
    string aNameJSON = aOutPly + ".geojson";
    aJSONOut.open(aNameJSON.c_str());
    aJSONOut<<"{"<<endl;
    aJSONOut<<" \"type\": \"FeatureCollection\","<<endl;

    aJSONOut<<"\"crs\": {"<<endl;
    aJSONOut<<" \"type\": \"EPSG\","<<endl;
    aJSONOut<<" \"properties\": {\"code\": "<<aCodeProj<<"}"<<endl;
    aJSONOut<<"},"<<endl;

    aJSONOut<<" \"features\": ["<<endl;

    for (uint aKIm=0; aKIm < aSetIm.size(); aKIm++)
    {
        string aNameIm = aSetIm[aKIm];
        cout<<" ++ "<<aNameIm<<endl;
        CamStenope * aCam = aICNM->StdCamStenOfNames(aNameIm, aOri);

        /*
        A typical way to generate trivial pseudo-random numbers in a determined
                range using rand is to use the modulo of the returned value by the range span and add the initial value of the range:
        v1 = rand() % 100;         // v1 in the range 0 to 99
        v2 = rand() % 100 + 1;     // v2 in the range 1 to 100
        v3 = rand() % 30 + 1985;   // v3 in the range 1985-2014
        */

        Pt3di aCoul(rand()%255, rand()%255, rand()%255);
        bool isEnd=false;
        if (aKIm == aSetIm.size()-1)
            isEnd=true;
        DrawOneFootPrintToPly(aCam, aNameIm, aCPlyRes, aCoul, aResolution, aJSONOut, isEnd, aOffSetPLY, aOffSetGeoJSON, aPlyEachImg);
    }
    aJSONOut<<"]"<<endl;
    aJSONOut<<"}"<<endl;
    aCPlyRes.PutFile(aOutPly);

    cout<<"Test_Footprint finish"<<endl;
    return EXIT_SUCCESS;
}

    /******************************************************************************
    The main function.
    ******************************************************************************/
int TestGiang_main(int argc,char ** argv)
{

    //Test_Xml();
    //Test_FAST();
    //Test_Footprint(argc, argv);

    cout<<"********************************************************"<<endl;
    cout<<"*    TestGiang                                         *"<<endl;
    cout<<"********************************************************"<<endl;
        cout<<"dParam : param of detector : "<<endl;
        cout<<"     [FAST_Threshold]"<<endl;
        cout<<"     NO"<<endl;

        string pathPlyFileS ;
        string aTypeD="HOMOLINIT";
        string aFullPattern, aOriInput;
        string aHomolOut = "_Filtered";
        bool assum1er=false;
        int SzPtCorr = 1;int indTri=-1;double corl_seuil_glob = 0.8;bool Test=false;
        int SzAreaCorr = 5; double corl_seuil_pt = 0.9;
        double PasCorr=0.5;
        vector<string> dParam; dParam.push_back("NO");
        bool useExistHomoStruct = false;
        double aAngleF = 90;
        bool debugByClick = false;
        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                    << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri)
                    << EAMC(pathPlyFileS, "path to mesh(.ply) file - created by Inital Ori", eSAM_IsExistFile),
                    //optional arguments
                    LArgMain()
                    << EAM(corl_seuil_glob, "corl_glob", true, "corellation threshold for imagette global, default = 0.8")
                    << EAM(corl_seuil_pt, "corl_pt", true, "corellation threshold for pt interest, default = 0.9")
                    << EAM(SzPtCorr, "SzPtCorr", true, "1->3*3,2->5*5 size of cor wind for each pt interet  default=1 (3*3)")
                    << EAM(SzAreaCorr, "SzAreaCorr", true, "1->3*3,2->5*5 size of zone autour pt interet for search default=5 (11*11)")
                    << EAM(PasCorr, "PasCorr", true, "step correlation (default = 0.5 pxl)")
                    << EAM(indTri, "indTri", true, "process one triangle")
                    << EAM(assum1er, "assum1er", true, "always use 1er pose as img master, default=0")
                    << EAM(Test, "Test", true, "Test new method - correl by XML")
                    << EAM(aTypeD, "aTypeD", true, "FAST, DIGEO, HOMOLINIT - default = HOMOLINIT")
                    << EAM(dParam,"dParam",true,"[param1, param2, ..] (selon detector - NO if don't have)", eSAM_NoInit)
                    << EAM(aHomolOut, "HomolOut", true, "default = _Filtered")
                    << EAM(useExistHomoStruct, "useExist", true, "use exist homol struct - default = false")
                    << EAM(aAngleF, "angleV", true, "limit view angle - default = 90 (all triangle is viewable)")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;
        vector<double> aParamD = parse_dParam(dParam); //need to to on arg enter
        InitOutil *aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD, aHomolOut,
                                          SzPtCorr, SzAreaCorr,
                                          corl_seuil_glob, corl_seuil_pt, false, useExistHomoStruct, PasCorr, assum1er);
        aChain->initAll(pathPlyFileS);
        cout<<endl<<" +++ Verify init: +++"<<endl;
        vector<pic*> PtrPic = aChain->getmPtrListPic();
        for (uint i=0; i<PtrPic.size(); i++)
        {
            cout<<PtrPic[i]->getNameImgInStr()<<" has ";
            vector<PackHomo> packHomoWith = PtrPic[i]->mPackHomoWithAnotherPic;
            cout<<packHomoWith.size()<<" homo packs with another pics"<<endl;
            for (uint j=0; j<packHomoWith.size(); j++)
            {
                if (j!=i)
                    cout<<" ++ "<< PtrPic[j]->getNameImgInStr()<<" "<<packHomoWith[j].aPack.size()<<" pts"<<endl;
            }
        }
        vector<triangle*> PtrTri = aChain->getmPtrListTri();
        cout<<PtrTri.size()<<" tri"<<endl;
        CorrelMesh aCorrel(aChain);
        if (!Test && indTri == -1)
        {
            if (aAngleF == 90)
            {
                cout<<"All Mesh is Viewable"<<endl;
                for (uint i=0; i<PtrTri.size(); i++)
                {
                    if (useExistHomoStruct)
                        aCorrel.correlByCplExist(i);
                    else
                        aCorrel.correlInTri(i);
                }
            }
            else
            {
                cout<<"Use condition angle view"<<endl;
                for (uint i=0; i<PtrTri.size(); i++)
                {
                    if (useExistHomoStruct)
                        aCorrel.correlByCplExistWithViewAngle(i, aAngleF);
                    else
                        aCorrel.correlInTriWithViewAngle(i, aAngleF);
                }
            }
        }
        if (indTri != -1)
        {
            cout<<"Do with tri : "<<indTri<<endl;
            CorrelMesh * aCorrel = new CorrelMesh(aChain);
            if (useExistHomoStruct == false)
                aCorrel->correlInTriWithViewAngle(indTri, aAngleF, debugByClick);
            else
                aCorrel->correlByCplExistWithViewAngle(indTri, aAngleF, debugByClick);
            delete aCorrel;
        }
        if(Test)
        {



        }
        cout<<endl<<"Total "<<aCorrel.countPts<<" cpl NEW & "<<aCorrel.countCplOrg<<" cpl ORG"<<endl;
        cout<<endl;
        return EXIT_SUCCESS;
    }

int IsExtrema(TIm2D<double,double> & anIm,Pt2di aP)
{
    double aValCentr = anIm.get(aP);
    const std::vector<Pt2di> &  aVE = SortedVoisinDisk(0.5,TT_DIST_EXTREMA,true);
    int aCmp0 =0;
    for (int aKP=0 ; aKP<int(aVE.size()) ; aKP++)
    {
        int aCmp = CmpValAndDec(aValCentr,anIm.get(aP+aVE[aKP]),aVE[aKP]);
        if (aKP==0)
        {
            aCmp0 = aCmp;
            if (aCmp0==0) return 0;
        }

        if (aCmp!=aCmp0) return 0;
    }
    return aCmp0;
}

Col_Pal  ColOfType(Video_Win * mW, eTypeTieTri aType)
{
    switch (aType)
    {
          case eTTTMax : return mW->pdisc()(P8COL::red);    //max local => red
          case eTTTMin : return mW->pdisc()(P8COL::blue);   //min local => bleu
          default :;
    }
   return mW->pdisc()(P8COL::yellow);   //No Label => Jaune
}

int TestDetecteur_main(int argc,char ** argv)
{
    Pt3di mSzW;
    string aImg;
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aImg, "img",  eSAM_None)
                << EAMC(mSzW, "mSzW", eSAM_None),
                //optional arguments
                LArgMain()
                );



    if (MMVisualMode) return EXIT_SUCCESS;
    Tiff_Im * mPicTiff = new Tiff_Im ( Tiff_Im::StdConvGen(aImg,1,false));
    Pt2di aSzIm = mPicTiff->sz();
    TIm2D<double,double> mPic_TIm2D(mPicTiff->sz());
    ELISE_COPY(mPic_TIm2D.all_pts(), mPicTiff->in(), mPic_TIm2D.out());
    Im2D<double,double> * anIm = new Im2D<double, double> (mPic_TIm2D._the_im);

    Im2D_Bits<1> aMasq0  = Im2D_Bits<1>(aSzIm.x,aSzIm.y,1);
    TIm2DBits<1> TaMasq0 = TIm2DBits<1> (aMasq0);
    /* video Win */
    Video_Win * mW_Org = 0;
    Video_Win * mW_F = 0;
    Video_Win * mW_FAC = 0; //origin, fast, fast && autocorrel
    Video_Win * mW_Final = 0;

    if (EAMIsInit(&mSzW))
    {
        if (aSzIm.x >= aSzIm.y)
        {
            double scale =  double(aSzIm.x) / double(aSzIm.y) ;
            mSzW.x = mSzW.x;
            mSzW.y = round_ni(mSzW.x/scale);
        }
        else
        {
            double scale = double(aSzIm.y) / double(aSzIm.x);
            mSzW.x = round_ni(mSzW.y/scale);
            mSzW.y = mSzW.y;
        }
        Pt2dr aZ(double(mSzW.x)/double(aSzIm.x) , double(mSzW.y)/double(aSzIm.y) );

        if (mW_Org ==0)
        {
            mW_Org = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
            mW_Org->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mW_Org->set_title((aImg+"_Extr").c_str());
            ELISE_COPY(anIm->all_pts(), anIm->in(), mW_Org->ogray());
        }
        if (mW_F == 0)
        {
            mW_F = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
            mW_F->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mW_F->set_title((aImg+"_FAST").c_str());
            ELISE_COPY(anIm->all_pts(), anIm->in(), mW_F->ogray());
        }
        if (mW_FAC == 0)
        {
            mW_FAC = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
            mW_FAC->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mW_FAC->set_title((aImg+"_ACORREL").c_str());
            ELISE_COPY(anIm->all_pts(), anIm->in(), mW_FAC->ogray());
        }
        if (mW_Final == 0)
        {
            mW_Final = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
            mW_Final->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mW_Final->set_title((aImg+"_FINAL").c_str());
            ELISE_COPY(anIm->all_pts(), anIm->in(), mW_Final->ogray());
        }
    }
    mW_Final->clik_in();


    Pt2di aP;
    std::vector<cIntTieTriInterest> aListPI;
    cFastCriterCompute * mFastCC   = cFastCriterCompute::Circle(TT_DIST_FAST);

    cCutAutoCorrelDir< TIm2D<double,double> > mCutACD (mPic_TIm2D,Pt2di(0,0),TT_SZ_AUTO_COR /2.0 ,TT_SZ_AUTO_COR);
    for (aP.x=5 ; aP.x<aSzIm.x-5 ; aP.x++)
    {
        for (aP.y=5 ; aP.y<aSzIm.y-5 ; aP.y++)
        {
            int aCmp0 =  IsExtrema(mPic_TIm2D,aP);
            if (aCmp0)
            {
                eTypeTieTri aType = (aCmp0==1)  ? eTTTMax : eTTTMin;
                bool OKAutoCorrel = !mCutACD.AutoCorrel(aP,TT_SEUIL_CutAutoCorrel_INT,TT_SEUIL_CutAutoCorrel_REEL,TT_SEUIL_AutoCorrel);
                Pt2dr aFastQual =  FastQuality(mPic_TIm2D,aP,*mFastCC,aType==eTTTMax,Pt2dr(TT_PropFastStd,TT_PropFastConsec));
                bool OkFast = (aFastQual.x > TT_SeuilFastStd) && ( aFastQual.y> TT_SeuilFastCons);
                if (OkFast && OKAutoCorrel)
                    aListPI.push_back(cIntTieTriInterest(aP,aType,aFastQual.x + 2 * aFastQual.y));
                if (mW_Org)
                {
                    mW_Org->draw_circle_loc(Pt2dr(aP),1.5,ColOfType(mW_Org, aType));    // cercle grand => extrema
                    //mW_Org->draw_circle_loc(Pt2dr(aP),0.5,mW_Org->pdisc()(OkFast ? P8COL::yellow : P8COL::cyan)); //=> cercle petit => Fast : jaune  = valid ; cyan = non valid
                }
                if (mW_F)
                {
                    mW_F->draw_circle_loc(Pt2dr(aP),1.5,ColOfType(mW_F, aType));
                    if (!OkFast)
                        mW_F->draw_circle_loc(Pt2dr(aP),1.5,mW_F->pdisc()(P8COL::cyan));
                }
                if (mW_FAC)
                {
                    mW_FAC->draw_circle_loc(Pt2dr(aP),1.5,ColOfType(mW_FAC, aType));
                    if (!OKAutoCorrel)
                        mW_FAC->draw_circle_loc(Pt2dr(aP),1.5,mW_FAC->pdisc()(P8COL::yellow));
                }
                if (mW_Final && OKAutoCorrel && OkFast)
                {
                    mW_Final->draw_circle_loc(Pt2dr(aP),1.5,ColOfType(mW_Final, aType));
                }
            }
        }

    }
    cout<<"Nb Pts :"<<aListPI.size()<<endl;
    mW_FAC->clik_in();
    return EXIT_SUCCESS;
}

//-----------------TestGiang---------------//


//-----------------Test New format point Homologue---------------//
double cal_Residu( Pt3dr aPInter3D , vector<CamStenope*> & aVCamInter, vector<Pt2dr> & aVPtInter)
{
    double aMoy =0.0;
    for (uint aKCam=0; aKCam<aVCamInter.size(); aKCam++)
    {
        CamStenope * aCam = aVCamInter[aKCam];
        Pt2dr aPtMes = aVPtInter[aKCam];
        Pt2dr aPtRep = aCam->Ter2Capteur(aPInter3D);
        aMoy = aMoy + euclid(aPtMes, aPtRep);
    }
    return aMoy/aVCamInter.size();
}


Pt3di gen_coul(double val, double min, double max)
{
    if (val <= max && val >=min)
    {
        double Number = (val-min)/(max-min);
        int Green = round(255.0 - (255.0 * Number));
        int Red = round(255.0 * Number);
        int Blue = 0;
        return Pt3di(Red, Green, Blue);
    }
    else if (max == min && val == max)
        return Pt3di(0,255,0);  //green ?
    else
        return Pt3di(0,0,0);  //noir
}

Pt3di gen_coul_heat_map(double value , double minimum, double maximum)
{
    double ratio = 2 * (value-minimum) / (maximum - minimum);
    int b = int(ElMax(0.0, 255*(1 - ratio)));
    int r = int(ElMax(0.0, 255*(ratio - 1)));
    int g = 255 - b - r;
    return Pt3di(r,g,b);
}

Pt3di gen_coul_emp(int val)
{
    switch (val)
    {
        case 1:
            return Pt3di (255,0,0); //rouge
        case 2:
            return Pt3di (255,144,0);   //orange
        case 3:
            return Pt3di (255,255,0);   //jaune
        case 4:
            return Pt3di (140,255,0);   //vert jaunit
        case 5:
            return Pt3di (0,255,221);   //cyan
        default:
            return Pt3di (0,255,0);     //vert
    }

}


Pt3dr Intersect_Simple(const std::vector<CamStenope *> & aVCS,const std::vector<Pt2dr> & aNPts2D)
{

    ELISE_ASSERT(aVCS.size() == aNPts2D.size(), "In Intersect_Simple: Nb Cam & Nb Pt2dr not coherent to intersect");
    std::vector<ElSeg3D> aVSeg;

    for (int aKR=0 ; aKR < int(aVCS.size()) ; aKR++)
    {
        //CamStenope * aCam = aVCS[aKR];
        //cout<<aNPts2D[aKR]<<aCam->NameIm()<<endl;
        ElSeg3D aSeg = aVCS.at(aKR)->F2toRayonR3(aNPts2D.at(aKR));
        //cout<<"done"<<endl;
        //ElSeg3D aSeg = aVCS.at(aKR)->Capteur2RayTer(aNPts2D.at(aKR));
        aVSeg.push_back(aSeg);
    }

    Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);
    return aRes;
}

void Intersect_Simple(const std::vector<cOneLoop*> & aVOneLoop, std::vector< vector<Pt3dr> > & aResultPt3D, vector<Pt3dr> & aResultPt3DAllImg)
{
    vector<Pt2dr> aVAllPt;
    vector<CamStenope * > aVAllCam;


    cOneLoop * aLoop0 =  aVOneLoop[0];
    vector<cOneImInLoop*> aVImL0 = aLoop0->mVIm;
    cOneImInLoop * aIm0Loop0 = aVImL0[0];


    for(uint aKL = 0; aKL<aVOneLoop.size(); aKL++)
    {
       cOneLoop * aLoop =  aVOneLoop[aKL];
       vector<cOneImInLoop*> aVIm = aLoop->mVIm;

       vector<Pt3dr> aRes1Loop;

       // Get 1st image in loop
           cOneImInLoop * aIm = aVIm[0];

           vector<Pt2dr> aVPt = aIm->mVPt;

           vector<Pt2dr> aVPtCorrespontAllIm;
           vector<CamStenope * > aVCamCorrespontAllIm;
            // all point in 1st image, get correspondant in another image in loop
           for (uint aKPt=0; aKPt < aVPt.size(); aKPt++)
           {
               Pt2dr aPt1st = aVPt[aKPt];
               aVPtCorrespontAllIm.push_back(aPt1st);
               aVCamCorrespontAllIm.push_back(aIm->mCam);
               // Get correspondant point in another image in loop
               for (uint aKIm = 1; aKIm < aVIm.size(); aKIm++)
               {
                   aVPtCorrespontAllIm.push_back(aVIm[aKIm]->mVPt[aKPt]);
                   aVCamCorrespontAllIm.push_back(aVIm[aKIm]->mCam);
               }
               // Intersection
               Pt3dr aPt3d = Intersect_Simple(aVCamCorrespontAllIm , aVPtCorrespontAllIm);
               aRes1Loop.push_back(aPt3d);

               aVPtCorrespontAllIm.clear();
               aVCamCorrespontAllIm.clear();
           }
           aResultPt3D.push_back(aRes1Loop);
    }

    // Intersect all point from all image
    vector<Pt2dr> aVPtIm0L0 = aIm0Loop0->mVPt;
    for (uint aKPt=0; aKPt < aVPtIm0L0.size(); aKPt++)
    {
        Pt2dr aPt = aVPtIm0L0[aKPt];
        CamStenope *aCam = aIm0Loop0->mCam;
        aVAllPt.push_back(aPt);
        aVAllCam.push_back(aCam);
        for(uint aKL = 0; aKL<aVOneLoop.size(); aKL++)
        {
            uint aKIm;
            if (aKL == 0)
                aKIm = 1;
            else
                aKIm = 0;
            vector<cOneImInLoop*> aVIm = aVOneLoop[aKL]->mVIm;
            for (; aKIm < aVIm.size(); aKIm++)
            {
                cOneImInLoop * aIm = aVIm[aKIm];
                Pt2dr aPt1 = aIm->mVPt[aKPt];
                CamStenope * aCam1 = aIm->mCam;

                aVAllPt.push_back(aPt1);
                aVAllCam.push_back(aCam1);

            }
        }
        Pt3dr aPt3d = Intersect_Simple(aVAllCam, aVAllPt);
        aResultPt3DAllImg.push_back(aPt3d);
        aVAllPt.clear();
        aVAllCam.clear();
    }
}


void PlyPutForCC(string & aPlyResCC, vector<Pt3dr> & aVAllPtInter, vector<double> & aVResidu)
{
    ELISE_ASSERT(aVAllPtInter.size() > 0,"No Pts in PlyPutForCC");
    ELISE_ASSERT(aVResidu.size() == aVAllPtInter.size(),"Pts and Res dif size in PlyPutForCC");

    //int aNbS = aVAllPtInter.size();
    std::string aTypeXYZ = "float";

    bool aModeBin = 1; // mode bin
    std::string mode = aModeBin ? "wb" : "w";
    FILE * aFP = FopenNN(aPlyResCC,mode,"PlyPutForCC");
/*
    //Header
    fprintf(aFP,"ply\n");
    std::string aBinSpec =       MSBF_PROCESSOR() ?
                           "binary_big_endian":
                           "binary_little_endian" ;

    fprintf(aFP,"format %s 1.0\n",aModeBin?aBinSpec.c_str():"ascii");
    fprintf(aFP,"element vertex %d\n",aNbS);
    fprintf(aFP,"property %s x\n",aTypeXYZ.c_str());
    fprintf(aFP,"property %s y\n",aTypeXYZ.c_str());
    fprintf(aFP,"property %s z\n",aTypeXYZ.c_str());

    fprintf(aFP,"property float intensity\n");


    //fprintf(aFP,"property list uchar int vertex_indices\n");
    fprintf(aFP,"end_header\n");
*/
    for (uint i=0; i<aVAllPtInter.size(); i++)
    {
        Pt3dr aPt = aVAllPtInter[i];
        double aRes = aVResidu[i];
        fprintf(aFP,"%.3f %.3f %.3f %f\n",aPt.x,aPt.y,aPt.z,aRes);
    }

    ElFclose(aFP);

}


int TestGiangNewHomol_Main(int argc,char ** argv)
{
    //Test_Conv(argc, argv);
    string aDir = "./";
    string aSH="";
    string aOri="";
    Pt2dr aRange(0.0,0.0);
    bool relative = true;
    double resMaxTapas = 3.0;
    string aPlyRes="Res_";
    string aPlyEmp="Emp_";
    double seuilBH = 0.0;
    Pt3dr aHistoRes(0,0,0);
    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aSH, "Homol New Format file",  eSAM_IsExistFile)
                      << EAMC(aOri, "Ori",  eSAM_IsExistDirOri),
          LArgMain()
                      << EAM(aDir,"Dir",true,"Directory , Def=./")
                      << EAM(resMaxTapas,"seuilRes",true,"threshold of reprojection error ")
                      << EAM(seuilBH,"seuilBH",true,"threshold for rapport B/H")
                      << EAM(aRange,"aRange",true,"range to colorize reprojection error ,green->red Def= colorize as relative (min->resMax)")
                      << EAM(aPlyRes,"PlyRes",true,"Ply's name output for residus - def=Cloud_Residu.ply")
                      << EAM(aPlyEmp,"PlyEmp",true,"Ply's name output for emplacement image - def=Cloud_Emp.ply")
                      << EAM(aHistoRes,"HistoRes",true,"Histogram of residue - [Min, Max, Nb Bin]")

    );

    if (EAMIsInit(&aRange))
    {
        relative = false;
    }

    cInterfChantierNameManipulateur*  aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    StdCorrecNameOrient(aOri, aICNM->Dir());
    const std::string  aSHInStr = aSH;
    cSetTiePMul * aSHIn = new cSetTiePMul(0);
    aSHIn->AddFile(aSHInStr);

    cout<<"Total : "<<aSHIn->DicoIm().mName2Im.size()<<" imgs"<<endl;
    std::map<std::string,cCelImTPM *> VName2Im = aSHIn->DicoIm().mName2Im;
    // load cam for all Img
    // Iterate through all elements in std::map
    std::map<std::string,cCelImTPM *>::iterator it = VName2Im.begin();
    vector<CamStenope*> aVCam (VName2Im.size());
    while(it != VName2Im.end())
    {
        //std::cout<<it->first<<" :: "<<it->second->Id()<<std::endl;
        string aNameIm = it->first;
        int aIdIm = it->second->Id();
        CamStenope * aCam = aICNM->StdCamStenOfNames(aNameIm, aOri);
        aVCam[aIdIm] = aCam;
        it++;
    }
    cPlyCloud aCPlyRes;
    cPlyCloud aCPlyEmp;

    cout<<"VPMul - Nb Config: "<<aSHIn->VPMul().size()<<endl;
    std::vector<cSetPMul1ConfigTPM *> aVCnf = aSHIn->VPMul();

    vector<double> aVResidu;            // residue de tout les points dans pack
    vector<Pt3dr> aVAllPtInter;         // Coordonne 3D de tout les points dans pack
    vector<int> aVNbImgOvlap;           // Nb Overlape de tout les points 3D dans pack

    vector<int> aStats(aSHIn->NbIm());  // Vector contient multiplicite de pack, index d'element du vector <=> multiplicite, valeur d'element <=> nb point
    vector<int> aStatsInRange(aSHIn->NbIm()); // Vector contient multiplicite de pack dans 1 gamme de residue defini, index d'element du vector <=> multiplicite, valeur d'element <=> nb point
    vector<int> aStatsValid;            // Vector contient multiplicite existe de pack, valeur d'element <=> multiplicité

    int nbPtsInRange = 0;
    double resMax = 0.0;
    double resMin = DBL_MAX;
    for (uint aKCnf=1; aKCnf<aVCnf.size(); aKCnf++)
    {
        cSetPMul1ConfigTPM * aCnf = aVCnf[aKCnf];
        //cout<<"Cnf : "<<aKCnf<<" - Nb Imgs : "<<aCnf->NbIm()<<" - Nb Pts : "<<aCnf->NbPts()<<endl;
        std::vector<int> aVIdIm =  aCnf->VIdIm();

        for (uint aKPtCnf=0; aKPtCnf<uint(aCnf->NbPts()); aKPtCnf++)
        {
            vector<Pt2dr> aVPtInter;
            vector<CamStenope*> aVCamInter;


            for (uint aKImCnf=0; aKImCnf<aVIdIm.size(); aKImCnf++)
            {
                //cout<<aCnf->Pt(aKPtCnf, aKImCnf)<<" ";
                aVPtInter.push_back(aCnf->Pt(aKPtCnf, aKImCnf));
                aVCamInter.push_back(aVCam[aVIdIm[aKImCnf]]);
            }
            //cout<<endl;
            //Intersect aVPtInter:
            ELISE_ASSERT(aVPtInter.size() == aVCamInter.size(), "Size not coherent");
            ELISE_ASSERT(aVPtInter.size() > 1 && aVCamInter.size() > 1, "Nb faiseaux < 2");
            Pt3dr aPInter3D = Intersect_Simple(aVCamInter , aVPtInter);
            //calcul reprojection error :
            double resMoy = cal_Residu( aPInter3D , aVCamInter, aVPtInter);
            //cout<<resMoy<<endl;
            if (resMoy >= resMax)
            {
                if (resMoy <= resMaxTapas)
                {
                    resMax = resMoy;
                }
                else
                {
                    resMax =resMaxTapas;
                }
            }
            else
            {
                if (resMoy <= resMin)
                {
                    resMin = resMoy;
                }
            }
            if (resMoy <= resMaxTapas && resMoy>=0.0)
            {
                aVAllPtInter.push_back(aPInter3D);
                aVResidu.push_back(resMoy);
                aVNbImgOvlap.push_back(aVPtInter.size());
            }

        }
    }
    //ajout au nuage de point
    cout<<"Nb Pt 3d : "<<aVResidu.size();
    cout<<"Res max = "<<resMax<<" -res Min = "<<resMin<<endl;


    // Chercher max and min multiplicity
    int aMaxOvLap = *max_element(aVNbImgOvlap.begin(), aVNbImgOvlap.end());
    int aMinOvLap = *min_element(aVNbImgOvlap.begin(), aVNbImgOvlap.end());
    cout<<" Max Min Overlap = "<<aMaxOvLap<<" "<<aMinOvLap<<endl;

    for (uint aKPt=0; aKPt<aVAllPtInter.size(); aKPt++)
    {
        //parcourir tout les points
        if (!relative)
        {
            aCPlyRes.AddPt(gen_coul_heat_map(aVResidu[aKPt], aRange.x,  aRange.y), aVAllPtInter[aKPt]);
        }
        else
        {
            aCPlyRes.AddPt(gen_coul_heat_map(aVResidu[aKPt], resMin,  resMax), aVAllPtInter[aKPt]);
        }
        aCPlyEmp.AddPt(Pt3di(aVNbImgOvlap[aKPt],0,0), aVAllPtInter[aKPt]);
        // faut donner le pourcentage de niveau de gris
        //aCPlyEmp.AddPt(aCPlyEmp.Gray(((double(aVNbImgOvlap[aKPt]) - double(aMinOvLap))/double(aMaxOvLap-aMinOvLap))), aVAllPtInter[aKPt]);
        //aCPlyEmp.AddPt(aCPlyEmp.Gray((double(aVNbImgOvlap[aKPt])), aVAllPtInter[aKPt]);
        //===== stats Multiplicite ========
        int nbImgsVu1Pts = aVNbImgOvlap[aKPt];
        aStats[nbImgsVu1Pts]++;
        if (aVResidu[aKPt] >= aRange.x && aVResidu[aKPt]<=aRange.y)
        {
            aStatsInRange[nbImgsVu1Pts]++;
            nbPtsInRange++;
        }
        //voir si dans aStatsValid exist nbImgsVu1Pts
        if (!(std::find(aStatsValid.begin(), aStatsValid.end(), nbImgsVu1Pts) != aStatsValid.end()))
        {
            //si exist pas
            aStatsValid.push_back(nbImgsVu1Pts);
        }
        //==================================
    }
    aPlyRes = aPlyRes + aOri + ".ply";
    aPlyEmp = aPlyEmp + aOri + ".ply";
    string aPlyResCC = "Res_" + aOri +"_CC.txt";
    aCPlyRes.PutFile(aPlyRes);
    aCPlyEmp.PutFile(aPlyEmp);
    PlyPutForCC(aPlyResCC, aVAllPtInter, aVResidu);

    //===== stats Multiplicite ========
    ofstream statsFile;
    string aName = "Stats_" + aOri + ".txt";
    statsFile.open(aName.c_str());
    statsFile << "Stats Multiplicite"<<endl;
    statsFile << "Nb Pts Total : "<<aVAllPtInter.size()<<endl;
    statsFile << "NbMul  NbPts  %"<<endl;
    sort(aStatsValid.begin(), aStatsValid.end());
    for (uint ikLine=0; ikLine<aStatsValid.size(); ikLine++)
        statsFile << aStatsValid[ikLine] <<" "<<aStats[aStatsValid[ikLine]]<<" "<< (double(aStats[aStatsValid[ikLine]])/double(aVAllPtInter.size()))*100.0 <<endl;

    if (!relative)
    {
        statsFile << "============  Range  ==========="<<aRange.x<<" -> "<<aRange.y<<"  ============="<<endl;
        statsFile << "Stats Multiplicite"<<endl;
        statsFile << "Nb Pts Total : "<<nbPtsInRange<<endl;
        statsFile << "NbMul  NbPts  %"<<endl;
        for (uint ikLine=0; ikLine<aStatsValid.size(); ikLine++)
            statsFile << aStatsValid[ikLine] <<" "<<aStatsInRange[aStatsValid[ikLine]]<<" "<<(double(aStatsInRange[aStatsValid[ikLine]])/double(nbPtsInRange))*100.0 <<endl;
    }
    //==================================

    //===== Histo Residue ==============
    if (EAMIsInit(&aHistoRes))
    {
        double pas = (aHistoRes.y-aHistoRes.x)/aHistoRes.z;
        sort(aVResidu.begin(), aVResidu.end());
        int ind=0;
        statsFile<<endl<<endl<<"Histo Residue image : "<<endl<< "RangeMin RangeMax NbPts %"<<endl;
        for (double iPas=aHistoRes.x; iPas<aHistoRes.y; iPas=iPas+pas)
        {
            cout<<"Range : "<<iPas<<" - "<<iPas+pas<<" : ";
            statsFile<<iPas<<" "<<iPas+pas<<" ";
            double minRange=iPas; double maxRange=iPas+pas;
            int cntBin = 0;
            //For each range, extract vector element in range
            for (uint i=ind; i<aVResidu.size(); i++)
            {
                //parcour vector from index
                if (aVResidu[i] >= minRange)
                {
                    if (aVResidu[i] < maxRange)
                    {
                        cntBin++;
                    }
                    else
                    {
                        ind = i;
                        break;
                    }
                }
            }
            cout<<cntBin<<" "<<(double(cntBin)/aVAllPtInter.size())*100<<endl;
            statsFile<<cntBin<<" "<<(double(cntBin)/aVAllPtInter.size())*100<<endl;
        }
    }
    //==================================
    statsFile.close();


    cout<<"Nb Emplacement image : 1 rouge - 2 orange - 3 jaune - 4 vert jaune - 5 cyan - > 5 vert"<<endl;
    return EXIT_SUCCESS;
}

//= ===========================  TEST GIANG DISPLAY ALL HOMOL WITH 1 IMAGE ============================


int TestGiangDispHomol_Main(int argc,char ** argv)
{

    string aDir="./";
    string aSH="";
    string aPat;
    string aImg;
    double aZ=0.2;
    bool aTwoSens = false;
    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPat, "Pattern Image",  eSAM_IsPatFile)
                      << EAMC(aImg, "Image Master",  eSAM_IsExistFile),
          LArgMain()
                      << EAM(aSH,"SH",true,"Homol suffix")
                      << EAM(aZ,"Zoom",true,"0.2")
                      << EAM(aTwoSens,"TwoSens",true,"fault")

    );
    string mPatIm;
    SplitDirAndFile(aDir, mPatIm, aPat);
    cInterfChantierNameManipulateur *mICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    vector<string>mSetIm = *(mICNM->Get(mPatIm));
    ELISE_ASSERT(mSetIm.size()>0,"ERROR: No image found!");
    //============================================================
       //anExt = ExpTxt ? "txt" : "dat";
       string mNameHomol = "Homol"+aSH;


       string mKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                          +  std::string(mNameHomol)
                          +  std::string("@")
                          +  std::string("dat");


       vector<Pt2dr> mVPts;
       for (uint i=0; i<mSetIm.size(); i++)
       {
           string pic1 = aImg;
           string pic2 = mSetIm[i];
           cout<<pic1<<" "<<pic2<<endl;
           if (pic1 != pic2)
           {
               // ===== sens 1 =====
               string aHomoIn = mICNM->Assoc1To2(mKHIn, pic1, pic2, true);
               StdCorrecNameHomol_G(aHomoIn, aDir);
               cout<<"++ "<<aHomoIn;
               bool Exist= ELISE_fp::exist_file(aHomoIn);
               if (Exist)
               {
                   ElPackHomologue aPackIn;
                   bool Exist= ELISE_fp::exist_file(aHomoIn);
                   if (Exist)
                   {
                       aPackIn =  ElPackHomologue::FromFile(aHomoIn);
                       for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++)
                       {

                           Pt2dr aP1 = itP->P1();  //Point img1
                           mVPts.push_back(aP1);

                       }
                   }
               }
               if (aTwoSens == true)
               {
                   // ===== sens 2 =====
                   string aHomoIn = mICNM->Assoc1To2(mKHIn, pic2, pic1, true);
                   StdCorrecNameHomol_G(aHomoIn, aDir);
                   cout<<"++ "<<aHomoIn;
                   bool Exist= ELISE_fp::exist_file(aHomoIn);
                   if (Exist)
                   {
                       ElPackHomologue aPackIn;
                       bool Exist= ELISE_fp::exist_file(aHomoIn);
                       if (Exist)
                       {
                           aPackIn =  ElPackHomologue::FromFile(aHomoIn);
                           for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++)
                           {

                               Pt2dr aP2 = itP->P2();  //Point img1
                               mVPts.push_back(aP2);

                           }
                       }
                   }
               }
               //  ====== ====== ======
           }
       }

        cout<<endl<<endl<<"Total = "<<mVPts.size()<<" pts for img "<<aImg<<endl;


        Tiff_Im mPicTiff = Tiff_Im::UnivConvStd(aDir+aImg);
        Pt2di mImgSz = mPicTiff.sz();
        Video_Win * aWin;
        aWin = Video_Win::PtrWStd(Pt2di(mImgSz.x*aZ, mImgSz.y*aZ), true, Pt2dr(aZ,aZ));


        aWin->set_sop(Elise_Set_Of_Palette::TheFullPalette());
        aWin->set_title((aImg+"_ALLPTS").c_str());
        ELISE_COPY(mPicTiff.all_pts(), mPicTiff.in(), aWin->ogray());

        for (uint i=0; i<mVPts.size(); i++)
        {
            aWin->draw_circle_loc(mVPts[i], 2.0/aZ, aWin->pdisc()(P8COL::green));
        }


        aWin->clik_in();

    return EXIT_SUCCESS;
}





int Image_Vide(int argc,char ** argv)
{

    string aDir="./";

    string aPat;
    string aImg;

    string aDirOutput = "ImgVide";

    cout<<"Create fake image (zerobyte file with same name) to cheat Tapas"<<endl;
    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aPat, "Pattern Image",  eSAM_IsPatFile),
          LArgMain()
                      << EAM(aDirOutput,"OutDir",true,"Output Directory")

    );
    ELISE_fp::MkDirSvp(aDirOutput);

    string mPatIm;
    SplitDirAndFile(aDir, mPatIm, aPat);
    cInterfChantierNameManipulateur *mICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    vector<string>mSetIm = *(mICNM->Get(mPatIm));
    ELISE_ASSERT(mSetIm.size()>0,"ERROR: No image found!");

    /* Create empty images */
    for (int aKIm=0; aKIm<int(mSetIm.size()); aKIm++)
    {

        // === EWELINA Tricks ====

        string aImPath = aDirOutput + "/" + mSetIm[aKIm];
        ELISE_fp aFPOut(aImPath.c_str(),ELISE_fp::WRITE);
        aFPOut.close();


        // ==== Giang tricks =====
        /*
        // Lire TIFF input
        //cLazyTiffFile * aTiff = new cLazyTiffFile(mSetIm[aKIm]);


       // Tiff_Im * aTiff = new Tiff_Im ( Tiff_Im::StdConvGen(mSetIm[aKIm],1,false));

        Tiff_Im * aTiff = new Tiff_Im
                            ( Tiff_Im::UnivConvStd(mSetIm[aKIm]));

        cMetaDataPhoto aMTD = cMetaDataPhoto::CreateExiv2(mSetIm[aKIm]);


        cout<<"Lire EXIF ... par cMetaDataPhoto : "<<endl;
        cout<<"Cam : "<<aMTD.Foc35(0)<<endl;



        // Take meta data & Tiff Info

        // Creat Output with metadata & Info

*/
    }

    return EXIT_SUCCESS;
}


