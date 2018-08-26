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

#include "NewOri.h"
#include "../TiepTri/MultTieP.h"
#include "../Apero/cPose.cpp"
//extern bool ERupnik_MM();

struct TripleStr
{
    public :
        TripleStr(const CamStenope *aC1,CamStenope *aC2,CamStenope *aC3,
                  const int aId1,const int aId2,const int aId3) :
                  mC1(aC1),
                  mC2(aC2),
                  mC3(aC3),
                  mId1(aId1),
                  mId2(aId2),
                  mId3(aId3) {}
 
        const CamStenope * mC1;
        const CamStenope * mC2;
        const CamStenope * mC3;

        const int   mId1;
        const int   mId2;
        const int   mId3;
};

class cAppliFictObs : public cCommonMartiniAppli
{
    public:
        cAppliFictObs(int argc,char **argv);

    private:

        void Initialize();
        void CalcResidPoly();
        void GenerateFicticiousObs();

        void UpdateAR(const ElPackHomologue*,const ElPackHomologue*,const ElPackHomologue*,
                      const int); 
        void UpdateAROne(const ElPackHomologue*,
                         const CamStenope*,const CamStenope*,
                         const int,const int);



        cNewO_NameManager *  mNM;
        
        int         mNumFPts;
        int         mNbIm;


        cSetTiePMul *               mPMul; 
        cSetTiePMul *               mPMulRed; 

        std::map<int,cAccumResidu *> mAR;

        cXml_TopoTriplet            mLT;

        std::map<std::string,int>   mNameMap;
        std::map<int, TripleStr*>   mTriMap;

        const std::vector<std::string> * mSetName;
        std::string                      mDir;
        std::string                      mPattern;
        std::string                      mOut;
};

void cAppliFictObs::GenerateFicticiousObs()
{

    int aNPtNum=0;
    int aTriNum=0;
    //pour chaque triplet recouper son elipse3d et genere les obs fict
    for (auto a3 : mLT.Triplets())
    {
        std::string  aName3R = mNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
    
        cXml_Elips3D anEl = aXml3Ori.Elips();
        
        cGenGaus3D aGG1(anEl);
        std::vector<Pt3dr> aVP;

        //generate the obs fict
        aGG1.GetDistribGaus(aVP,mNumFPts,mNumFPts,mNumFPts);
        aNPtNum += (int)aVP.size();
        
//        std::cout << "(int)aVP.size() " << (int)aVP.size() << "\n";
     

        //get cams
        const CamStenope * aC1 = mTriMap[aTriNum]->mC1;
        const CamStenope * aC2 = mTriMap[aTriNum]->mC2;
        const CamStenope * aC3 = mTriMap[aTriNum]->mC3;

        //cam ids
        std::vector<int> aTriIds{mTriMap[aTriNum]->mId1,
                                 mTriMap[aTriNum]->mId2,
                                 mTriMap[aTriNum]->mId3};
        


        //back-proj the fict points to the triplet
        for (int aK=0; aK<(int)aVP.size(); aK++)
        {

            std::vector<Pt2dr> aPImV;
            aPImV.push_back(aC1->Ter2Capteur(aVP.at(aK)));
            aPImV.push_back(aC2->Ter2Capteur(aVP.at(aK)));
            aPImV.push_back(aC3->Ter2Capteur(aVP.at(aK)));

            //add the residual
            //mAR[mTriMap[aTriNum]->mId1]
    
            std::vector<float> aAttr;
            mPMulRed->AddPts(aTriIds,aPImV,aAttr);

            if (0)
            {
                std::vector<ElSeg3D> aSegV;
                std::vector<double> aVPds;

                aSegV.push_back(aC1->Capteur2RayTer(aPImV.at(0)));
                aSegV.push_back(aC2->Capteur2RayTer(aPImV.at(1)));
                aSegV.push_back(aC3->Capteur2RayTer(aPImV.at(2)));

                aVPds.push_back(1.0);
                aVPds.push_back(1.0);
                aVPds.push_back(1.0);

                bool ISOK=false;
                Pt3dr aPVerif = ElSeg3D::L2InterFaisceaux(&aVPds,aSegV,&ISOK);
                std::cout << "P=" << aVP.at(aK) << ", p=" << aPImV.at(0) << " " << aPImV.at(1) << " " << aPImV.at(2) << 
                          " \nPVerif=" << aPVerif  << " ISOK? " << ISOK << "\n";
                getchar();
            }
        }
        aTriNum++;
    }
    
    std::string aSaveTo = "Homol/PMul-" + mOut + ".txt";
    mPMulRed->Save(aSaveTo);
    std::cout << "cAppliFictObs::GenerateFicticiousObs()" << " ";    
    cout << " " << aNPtNum << " points saved to " << aSaveTo << "\n";
}

void cAppliFictObs::UpdateAROne(const ElPackHomologue* aPack,
                                const CamStenope* aC1,const CamStenope* aC2,
                                const int aC1Id,const int aC2Id)
{
    for (ElPackHomologue::const_iterator itP=aPack->begin(); itP!=aPack->end(); itP++)
    {
        Pt2dr aDir1,aDir2;

        double aRes1 = aC1->EpipolarEcart(itP->P1(),*aC2,itP->P2(),&aDir1);
        double aRes2 = aC2->EpipolarEcart(itP->P2(),*aC1,itP->P1(),&aDir2);
        //std::cout << "P1=" << itP->P1() << ", P2=" << itP->P2() << "\n";
        //std::cout << "Res1=" << aRes1 << ", Res2=" << aRes2 << "\n";

        cInfoAccumRes aInf1(itP->P1(),1.0,aRes1,aDir1);
        mAR[aC1Id]->Accum(aInf1);

        cInfoAccumRes aInf2(itP->P2(),1.0,aRes2,aDir2);
        mAR[aC2Id]->Accum(aInf2);


    }
}

void cAppliFictObs::UpdateAR(const ElPackHomologue* Pack12,
                             const ElPackHomologue* Pack13,
                             const ElPackHomologue* Pack23,
                             const int aTriId)
{

    UpdateAROne(Pack12,mTriMap[aTriId]->mC1,mTriMap[aTriId]->mC2,
                       mTriMap[aTriId]->mId1,mTriMap[aTriId]->mId2);

    UpdateAROne(Pack13,mTriMap[aTriId]->mC1,mTriMap[aTriId]->mC3,
                       mTriMap[aTriId]->mId1,mTriMap[aTriId]->mId3);

    UpdateAROne(Pack23,mTriMap[aTriId]->mC2,mTriMap[aTriId]->mC3,
                       mTriMap[aTriId]->mId2,mTriMap[aTriId]->mId3);

}


void cAppliFictObs::CalcResidPoly()
{
    //residual displacement maps
    Pt2di  aSz(100,100);
    double aRed=5;
    bool   OnlySign=true;
    int    aDegPol = 1;

    for (auto aCam : mNameMap) //because residual per pose
        mAR[aCam.second] = new cAccumResidu(aSz,aRed,OnlySign,aDegPol);


    int aTriNum=0;
    for (auto a3 : mLT.Triplets())
    {
  
        /* Cam1 - Cam2 Homol */
        //mNM->LoadHomFloats(a3.Name1(),a3.Name2(),&aVP12,&aVP21);
        const ElPackHomologue aElHom12 = mNM->PackOfName(a3.Name1(),a3.Name2());
    
        /* Cam1 - Cam3 Homol */
        //mNM->LoadHomFloats(a3.Name1(),a3.Name3(),&aVP13,&aVP31);
        const ElPackHomologue aElHom13 = mNM->PackOfName(a3.Name1(),a3.Name3());

        /* Cam2 - Cam3 Homol */
        //mNM->LoadHomFloats(a3.Name2(),a3.Name3(),&aVP23,&aVP32);
        const ElPackHomologue aElHom23 = mNM->PackOfName(a3.Name2(),a3.Name3());

 
        UpdateAR(&aElHom12,&aElHom13,&aElHom23,aTriNum);
        aTriNum++;
    }

    FILE* aFileExpImRes = FopenNN("StatRes.txt","w","cAppliFictObs::CalcResidPoly");
    cUseExportImageResidu aUEIR;
    aUEIR.SzByPair()    = 30;
    aUEIR.SzByPose()    = 50;
    aUEIR.SzByCam()     = 100;
    aUEIR.NbMesByCase() = 10;
    aUEIR.GeneratePly() = true;

    mAR[10]->Export("./","TestRes",aUEIR,aFileExpImRes);
    fclose(aFileExpImRes); 

    std::cout << "cAppliFictObs::CalcResidPoly()" << "\n";    

}

void cAppliFictObs::Initialize()
{
    //file managers
    cElemAppliSetFile anEASF(mPattern);
    mSetName = anEASF.SetIm();
    mNbIm = (int)mSetName->size();

    for (int aK=1; aK<mNbIm; aK++)
        mNameMap[mSetName->at(aK)] = aK;

    mNM = NM(mDir);


    //triplets
    std::string aNameLTriplets = mNM->NameTopoTriplet(true);
    mLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);
    
    //initialize reduced tie-points
    mPMulRed = new cSetTiePMul(0,mSetName);
   

    //update orientations in mCamMap
    int aTriNb=0;
    for (auto a3 : mLT.Triplets())
    {

        std::string  aName3R = mNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
    
     
        //get the poses 
        ElRotation3D aP1 = ElRotation3D::Id ;
        ElRotation3D aP2 = Xml2El(aXml3Ori.Ori2On1());
        ElRotation3D aP3 = Xml2El(aXml3Ori.Ori3On1());
        
        CamStenope * aC1 = mNM->CamOfName(a3.Name1());
        CamStenope * aC2 = mNM->CamOfName(a3.Name2());
        CamStenope * aC3 = mNM->CamOfName(a3.Name3());

        //should handle camera variant calibration
        if (aC1==aC2)
            aC2 = aC1->Dupl();
        if (aC1==aC3)
            aC3 = aC1->Dupl();

        //update poses 
        aC1->SetOrientation(aP1.inv());
        aC2->SetOrientation(aP2.inv());
        aC3->SetOrientation(aP3.inv());

        mTriMap[aTriNb] = new TripleStr(aC1,aC2,aC3,
                                        mNameMap[a3.Name1()],
                                        mNameMap[a3.Name2()],
                                        mNameMap[a3.Name3()]);


        aTriNb++;

    }
}

cAppliFictObs::cAppliFictObs(int argc,char **argv) :
    mNumFPts(1.0),
    mOut("ElRed")
{
    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mPattern,"Pattern of images"),
        LArgMain() << EAM (mNumFPts,"NPt",true,"Number of ficticious pts, Def=1 (1:27pts, 2:175pts)")
                   << EAM (mOut,"Out",true,"Output file name")
    );
   #if (ELISE_windows)
        replace( mPattern.begin(), mPattern.end(), '\\', '/' );
   #endif

    SplitDirAndFile(mDir,mPattern,mPattern);

    Initialize();
    
    CalcResidPoly();

    GenerateFicticiousObs();


}

int CPP_FictiveObsFin_main(int argc,char ** argv)
{
    cAppliFictObs AppliFO(argc,argv);

    return EXIT_SUCCESS;
 
}

