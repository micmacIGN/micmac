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


/* contains both triplets and couples;
   if couples then aC3=0 */
struct TripleStr
{
    public :
        
        TripleStr(const CamStenope *aC1,const int aId1,
                  const CamStenope *aC2,const int aId2,
                  const CamStenope *aC3=0,const int aId3=0) :
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
        void InitNFHom();
        void InitNFHomOne(std::string&,std::string&);
        void CalcResidPoly();
        void GenerateFicticiousObs();

        void UpdateAR(const ElPackHomologue*,const ElPackHomologue*,const ElPackHomologue*,
                      const int); 
        void UpdateAROne(const ElPackHomologue*,
                         const CamStenope*,const CamStenope*,
                         const int,const int);
        Pt2di ApplyRedFac(Pt2dr&);
        
        void SaveHomolOne(std::vector<int>& aId,
                          std::vector<Pt2dr>& aPImV,
                          std::vector<float>& aAttr);
        void SaveHomol(std::string&);        

        cNewO_NameManager *  mNM;
        
        int         mNumFPts;
        int         mNbIm;

        bool                                                    NFHom;    //new format homol
        cSetTiePMul *                                           mPMulRed; 
        std::map<std::string,ElPackHomologue *>                 mHomRed; //hom name, hom
        std::map<std::string,std::map<std::string,std::string>> mHomMap; //cam name, cam name, hom name
        std::string                                             mHomExp;
        
        cXml_TopoTriplet               mLT;
        cSauvegardeNamedRel            mLCpl;

        std::map<std::string,int>      mNameMap;
        std::map<int, TripleStr*>      mTriMap;
        std::map<int,cAccumResidu *>   mAR;

        Pt2di                       mSz;
        int                         mResPoly;
        int                         mRedFacSup;
        double                      mResMax;

        const std::vector<std::string> * mSetName;//redundant with mNameMap; best to remove and be coherent
        std::string                      mDir;
        std::string                      mPattern;
        std::string                      mOut;
};


cAppliFictObs::cAppliFictObs(int argc,char **argv) :
    mNumFPts(1.0),
    NFHom(true),
    mPMulRed(0),
    mHomExp("dat"),
    mResPoly(2),
    mRedFacSup(20),
    mResMax(5),
    mOut("ElRed")
{

    bool aExpTxt=false;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mPattern,"Pattern of images"),
        LArgMain() << EAM (mNumFPts,"NPt",true,"Number of ficticious pts, Def=1 (1:27pts, 2:175pts)")
                   << EAM (mRedFacSup,"RedFac",true,"Residual image reduction factor, Def=20")
                   << EAM (mResPoly,"Deg",true,"Degree of polyn to smooth residual images, Def=2")
                   << EAM (mResMax,"RMax",true,"Maximum residual, everything above will be filtered out, Def=5")
                   << EAM (NFHom,"NF",true,"Save homol to new format?, Def=true")
                   << EAM (aExpTxt,"ExpTxt",true,"ASCII homol?, Def=true")
                   << EAM (mOut,"Out",true,"Output file name")
    );
   #if (ELISE_windows)
        replace( mPattern.begin(), mPattern.end(), '\\', '/' );
   #endif

    aExpTxt ? mHomExp="txt" : mHomExp="dat";

    SplitDirAndFile(mDir,mPattern,mPattern);

    Initialize();
    
    CalcResidPoly();

    GenerateFicticiousObs();


}

Pt2di cAppliFictObs::ApplyRedFac(Pt2dr& aP)
{
    Pt2di aRes;
    aRes.x = round_up(aP.x/mRedFacSup) -1;
    aRes.y = round_up(aP.y/mRedFacSup) -1;

    return aRes;
}


void cAppliFictObs::GenerateFicticiousObs()
{

    int aNPtNum=0;


    /* pour chaque triplet/cple recouper son elipse3d et genere les obs fict,
       alternatively recalculate the ellipse - to do
    */
    for (auto aT : mTriMap)
    {
        
        cXml_Elips3D anEl;

        //triplets
        if (aT.second->mC3)
        {
            std::string  aName3R = mNM->NameOriOptimTriplet(true,mSetName->at(aT.second->mId1),
                                                                 mSetName->at(aT.second->mId2),
                                                                 mSetName->at(aT.second->mId1));
            cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
            anEl = aXml3Ori.Elips();
        }
        else//cple
        {
            std::string aNamOri = mNM->NameXmlOri2Im(mSetName->at(aT.second->mId1),
                                                     mSetName->at(aT.second->mId2),true);
            cXml_Ori2Im aXml2Ori = StdGetFromSI(aNamOri,Xml_Ori2Im);
            anEl = aXml2Ori.Geom().Val().Elips();
        } 
   
        cGenGaus3D aGG1(anEl);
        std::vector<Pt3dr> aVP;
    
        //generate the obs fict
        aGG1.GetDistribGaus(aVP,mNumFPts,mNumFPts,mNumFPts);
        aNPtNum += (int)aVP.size();


        //get cams
        std::vector<const CamStenope * > aVC;
        if (aT.second->mC3)
            aVC = {aT.second->mC1,
                   aT.second->mC2,
                   aT.second->mC3};
        else
            aVC = {aT.second->mC1,
                   aT.second->mC2};

        //cam ids
        std::vector<int> aTriIds;
        if (aT.second->mC3)
            aTriIds = {aT.second->mId1,
                       aT.second->mId2,
                       aT.second->mId3};
        else
            aTriIds = {aT.second->mId1,
                       aT.second->mId2};

        //back-proj the fict points to the triplet/cple
        for (int aK=0; aK<(int)aVP.size(); aK++)
        {

            std::vector<Pt2dr> aPImV;
            for (int aC=0; aC<int(aVC.size()); aC++) 
                aPImV.push_back(aVC.at(aC)->Ter2Capteur(aVP.at(aK)));

            for (int aC=0; aC<int(aVC.size()); aC++)
            {
                if ((aPImV.at(aC).x >0) && (aPImV.at(aC).x < mSz.x) &&
                    (aPImV.at(aC).y > 0) && (aPImV.at(aC).y < mSz.y))
                {

                    Pt2dr aPCor;
                    mAR[aT.second->mId1]->ExportResXY(ApplyRedFac(aPImV.at(aC)),aPCor);
                    aPImV.at(aC) += aPCor;


                }

            }

            std::vector<float> aAttr;
            SaveHomolOne(aTriIds,aPImV,aAttr);

            if (0)
            {
                std::vector<ElSeg3D> aSegV;
                std::vector<double> aVPds;
           
                for (int aC=0; aC<int(aVC.size()); aC++)
                {
                    aSegV.push_back(aVC.at(aC)->Capteur2RayTer(aPImV.at(aC))); 
                    aVPds.push_back(1.0);
                }
            
            
                bool ISOK=false;
                Pt3dr aPVerif = ElSeg3D::L2InterFaisceaux(&aVPds,aSegV,&ISOK);
                std::cout << "P=" << aVP.at(aK) << ", p=" << aPImV.at(0) << " " << aPImV.at(1) << " " << aPImV.at(2) <<
                          " \nPVerif=" << aPVerif  << " ISOK? " << ISOK << "\n";
                getchar();

            }
        }
        

    }


    /*for (auto a3 : mLT.Triplets())
    {
        std::string  aName3R = mNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
    
        cXml_Elips3D anEl = aXml3Ori.Elips();
        
        cGenGaus3D aGG1(anEl);
        std::vector<Pt3dr> aVP;

        //generate the obs fict
        aGG1.GetDistribGaus(aVP,mNumFPts,mNumFPts,mNumFPts);
        aNPtNum += (int)aVP.size();
        

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

            if ((aPImV.at(0).x > 0) && (aPImV.at(0).x < mSz.x) &&
                (aPImV.at(0).y > 0) && (aPImV.at(0).y < mSz.y) &&
                (aPImV.at(1).x > 0) && (aPImV.at(1).x < mSz.x) &&
                (aPImV.at(1).y > 0) && (aPImV.at(1).y < mSz.y) &&
                (aPImV.at(2).x > 0) && (aPImV.at(2).x < mSz.x) &&
                (aPImV.at(2).y > 0) && (aPImV.at(2).y < mSz.y))
            {
                

                Pt2dr aP1Cor; 
                mAR[mTriMap[aTriNum]->mId1]->ExportResXY(ApplyRedFac(aPImV.at(0)),aP1Cor);
                aPImV.at(0) += aP1Cor;                

             
                Pt2dr aP2Cor; 
                mAR[mTriMap[aTriNum]->mId2]->ExportResXY(ApplyRedFac(aPImV.at(1)),aP2Cor);
                aPImV.at(1) += aP2Cor;
             
                Pt2dr aP3Cor;
                mAR[mTriMap[aTriNum]->mId3]->ExportResXY(ApplyRedFac(aPImV.at(2)),aP3Cor);
                aPImV.at(2) += aP3Cor;
             
                std::vector<float> aAttr;
                SaveHomolOne(aTriIds,aPImV,aAttr);
             
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
        }
        aTriNum++;
    }*/
    
    std::string aSaveTo = "Homol" + mOut + "/PMul-" + mOut + ".txt";
    SaveHomol(aSaveTo);

    std::cout << "cAppliFictObs::GenerateFicticiousObs()" << " ";    
    cout << " " << aNPtNum << " points saved. " << "\n";
}

void cAppliFictObs::SaveHomol(std::string& aName)
{
    if (NFHom)
    {
        mPMulRed->Save(aName);
    }
    else
    {
        for (auto itH : mHomRed)
        {
            itH.second->StdPutInFile(itH.first);
        }
    }
}

void cAppliFictObs::SaveHomolOne(std::vector<int>& aId,std::vector<Pt2dr>& aPImV,std::vector<float>& aAttr)
{
    if (NFHom)
    {
        mPMulRed->AddPts(aId,aPImV,aAttr);        
    }
    else
    {
        //symmetrique
        for (int aK1=0; aK1<(int)aId.size(); aK1++)
        {
            for (int aK2=0; aK2<(int)aId.size(); aK2++)
            {
                if (aK1!=aK2)
                {
                    std::string aN1 = mSetName->at(aId.at(aK1));
                    std::string aN2 = mSetName->at(aId.at(aK2));
                   
                    std::string aNameH = mNM->ICNM()->Assoc1To2("NKS-Assoc-CplIm2Hom@"+mOut+"@"+mHomExp,aN1,aN2,true);  

                    if (DicBoolFind(mHomRed,aNameH))
                    {   

                        ElCplePtsHomologues aP(aPImV.at(aK1),aPImV.at(aK2));
                        mHomRed[aNameH]->Cple_Add(aP);

                    }
                }
            }
        }
    }
}

void cAppliFictObs::UpdateAROne(const ElPackHomologue* aPack,
                                const CamStenope* aC1,const CamStenope* aC2,
                                const int aC1Id,const int aC2Id)
{
    
    if (aC1 && aC2)
    {
        for (ElPackHomologue::const_iterator itP=aPack->begin(); itP!=aPack->end(); itP++)
        {
            Pt2dr aDir1,aDir2;
 
            double aRes1 = aC1->EpipolarEcart(itP->P1(),*aC2,itP->P2(),&aDir1);
            double aRes2 = aC2->EpipolarEcart(itP->P2(),*aC1,itP->P1(),&aDir2);
 
            if ((aRes1<mResMax) && (aRes1>-mResMax) && (aRes2>-mResMax) && (aRes2<mResMax)) 
            {
                cInfoAccumRes aInf1(itP->P1(),1.0,aRes1,aDir1);
                mAR[aC1Id]->Accum(aInf1);
 
                cInfoAccumRes aInf2(itP->P2(),1.0,aRes2,aDir2);
                mAR[aC2Id]->Accum(aInf2);
            }
            /* else   //it still somewhat strange there are pts with this large spread
            {
                std::cout << "P1=" << itP->P1() << ", P2=" << itP->P2() << " ";
                std::cout << "Res1=" << aRes1 << ", Res2=" << aRes2 << "\n";
 
            } */
 
        }
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
    bool   OnlySign=true;

    for (auto aCam : mNameMap) //because residual per pose
        mAR[aCam.second] = new cAccumResidu(mSz,mRedFacSup,OnlySign,mResPoly);


    int aTriNum=0;
    for (auto aT : mTriMap)
    {
        /* Cam1 - Cam2 Homol */
        const ElPackHomologue aElHom12 = mNM->PackOfName(mSetName->at(aT.second->mId1),mSetName->at(aT.second->mId2));

        /* Cam1 - Cam3 Homol */
        const ElPackHomologue aElHom13 = mNM->PackOfName(mSetName->at(aT.second->mId1),mSetName->at(aT.second->mId3));
    
        /* Cam2 - Cam3 Homol */
        const ElPackHomologue aElHom23 = mNM->PackOfName(mSetName->at(aT.second->mId2),mSetName->at(aT.second->mId3));

        UpdateAR(&aElHom12,&aElHom13,&aElHom23,aTriNum);
        aTriNum++;

    }

    /*int aTriNum=0;
    for (auto a3 : mLT.Triplets())
    {
  
        const ElPackHomologue aElHom12 = mNM->PackOfName(a3.Name1(),a3.Name2());
    
        const ElPackHomologue aElHom13 = mNM->PackOfName(a3.Name1(),a3.Name3());

        const ElPackHomologue aElHom23 = mNM->PackOfName(a3.Name2(),a3.Name3());

 
        UpdateAR(&aElHom12,&aElHom13,&aElHom23,aTriNum);
        aTriNum++;
    }*/

    FILE* aFileExpImRes = FopenNN("StatRes.txt","w","cAppliFictObs::CalcResidPoly");
    cUseExportImageResidu aUEIR;
    aUEIR.SzByPair()    = 30;
    aUEIR.SzByPose()    = 50;
    aUEIR.SzByCam()     = 100;
    aUEIR.NbMesByCase() = 10;
    aUEIR.GeneratePly() = true;

    mAR[0]->Export("./","TestRes",aUEIR,aFileExpImRes);
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
   
    //couples 
    std::string aNameLCple = mNM->NameListeCpleOriented(true);
    mLCpl = StdGetFromSI(aNameLCple,SauvegardeNamedRel); 
 
    //initialize reduced tie-points
    if (NFHom)
        mPMulRed = new cSetTiePMul(0,mSetName);
    else
        InitNFHom();

    //update triplet orientations in mTriMap
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

        mTriMap[aTriNb] = new TripleStr(aC1,mNameMap[a3.Name1()],
                                        aC2,mNameMap[a3.Name2()],
                                        aC3,mNameMap[a3.Name3()]);

        aTriNb++;

    }
    //update couples orientations in mTriMap
    for (auto a2 : mLCpl.Cple())
    {
        
        //poses
        bool OK;
        ElRotation3D aP1 = ElRotation3D::Id;
        ElRotation3D aP2 = mNM->OriCam2On1 (a2.N1(),a2.N2(),OK);
        if(!OK)
            std::cout << "cAppliFictObs::Initialize() warning - no elipse3D for couple " << a2.N1() << " " << a2.N2() << "\n";

        CamStenope *aC1 = mNM->CamOfName(a2.N1()); 
        CamStenope *aC2 = mNM->CamOfName(a2.N2());

        //should handle camera variant calibration
        if (aC1==aC2)
            aC2 = aC1->Dupl();

        //update poses
        aC1->SetOrientation(aP1.inv()); 
        aC2->SetOrientation(aP2.inv()); 

        mTriMap[aTriNb] = new TripleStr(aC1,mNameMap[a2.N1()],
                                        aC2,mNameMap[a2.N2()]);

        aTriNb++;
    }
    
    if (aTriNb!=0)
    {
        mSz = mTriMap[0]->mC1->Sz();
    }
    else
        ELISE_ASSERT(false,"cAppliFictObs::Initialize no couples or triplets found");
}   


void cAppliFictObs::InitNFHomOne(std::string& N1,std::string& N2)
{
    std::string aNameH = mNM->ICNM()->Assoc1To2("NKS-Assoc-CplIm2Hom@"+mOut+"@"+mHomExp,N1,N2,true);

    if (! DicBoolFind(mHomRed,aNameH))
    {
        mHomRed[aNameH] = new ElPackHomologue();
       
        std::map<std::string,std::string> aSSMap;
        aSSMap[N2] = aNameH;
        mHomMap[N1] = aSSMap;
    }
}

void cAppliFictObs::InitNFHom()
{

    for (auto a3 : mLT.Triplets())
    {
        //symmetrique points
        InitNFHomOne(a3.Name1(),a3.Name2());
        InitNFHomOne(a3.Name2(),a3.Name1());
        
        InitNFHomOne(a3.Name1(),a3.Name3());
        InitNFHomOne(a3.Name3(),a3.Name1());

        InitNFHomOne(a3.Name2(),a3.Name3());
        InitNFHomOne(a3.Name3(),a3.Name2());


            //StdPutInFile
    } 
}


int CPP_FictiveObsFin_main(int argc,char ** argv)
{
    cAppliFictObs AppliFO(argc,argv);

    return EXIT_SUCCESS;
 
}

