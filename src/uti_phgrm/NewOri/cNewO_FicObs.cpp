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

ElRotation3D TestOriConvention(std::string & aNameOri, const std::string & mNameIm1, const std::string & mNameIm2);

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
        bool IsInSz(Pt2dr&) const;

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
        
        Pt3dr       mNumFPts;
        bool        mNSym;
        int         mNbIm;

        bool                                                    NFHom;    //new format homol
        cSetTiePMul *                                           mPMulRed; 
        std::map<std::string,ElPackHomologue *>                 mHomRed; //hom name, hom
        std::map<std::string,std::map<std::string,std::string>> mHomMap; //cam name, cam name, hom name
        std::string                                             mHomExp;
    
        std::string                    mNameOriCalib;    
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
        std::string                      mPrefHom;
        std::string                      mOut;
        bool                             mPly;
};


cAppliFictObs::cAppliFictObs(int argc,char **argv) :
    mNumFPts(Pt3dr(1,1,1)),
    mNSym(false),
    NFHom(true),
    mPMulRed(0),
    mHomExp("dat"),
    mNameOriCalib(""),
    mResPoly(2),
    mRedFacSup(20),
    mResMax(5),
    mPrefHom(""),
    mOut("ElRed"),
    mPly(false)
{

    bool aExpTxt=false;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mPattern,"Pattern of images"),
        LArgMain() << EAM (mNumFPts,"NPt",true,"Number of ficticious pts, Def=1 (1:27pts, 2:175pts)")
                   << EAM (mNSym,"NSym",true,", Non-symetric point generation, Def=false")
                   << EAM (mRedFacSup,"RedFac",true,"Residual image reduction factor, Def=20")
                   << EAM (mResPoly,"Deg",true,"Degree of polyn to smooth residual images, Def=2")
                   << EAM (mResMax,"RMax",true,"Maximum residual, everything above will be filtered out, Def=5")
                   << EAM (NFHom,"NF",true,"Save homol to new format?, Def=true")
                   << EAM (aExpTxt,"ExpTxt",true,"ASCII homol?, Def=true")
                   << EAM (mPrefHom,"SH",true,"Homol post-fix, Def=\"\"")
                   << EAM (mNameOriCalib,"OriCalib",true,"Calibration folder if exists, Def=\"\"")
                   << EAM (mOut,"Out",true,"Output file name")
                   << EAM (mPly,"Ply",true,"Output ply file?, def=false")
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

bool cAppliFictObs::IsInSz(Pt2dr& aP) const
{
    if ((aP.x > 0) && (aP.x < mSz.x) &&
        (aP.y > 0) && (aP.y < mSz.y))
        return true;
    else
        return false;

}

void cAppliFictObs::GenerateFicticiousObs()
{

    int aNPtNum=0;


    /* pour chaque triplet/cple recouper son elipse3d et genere les obs fict */
    for (auto aT : mTriMap)
    {
        
        cXml_Elips3D anEl;

        //triplets
        if (aT.second->mC3)
        {

            std::string  aName3R = mNM->NameOriOptimTriplet(true,mSetName->at(aT.second->mId1),
                                                                 mSetName->at(aT.second->mId2),
                                                                 mSetName->at(aT.second->mId3));
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
        if (mNSym)
            aGG1.GetDistribGausNSym(aVP,mNumFPts.x,mNumFPts.y,mNumFPts.z);
        else
            aGG1.GetDistribGaus(aVP,mNumFPts.x,mNumFPts.y,mNumFPts.z);
        
        aNPtNum += (int)aVP.size();


        //get all cams
        std::vector<const CamStenope * > aVC;
        if (aT.second->mC3)
            aVC = {aT.second->mC1,
                   aT.second->mC2,
                   aT.second->mC3};
        else
            aVC = {aT.second->mC1,
                   aT.second->mC2};

        if (0)
        {
            std::cout << "C1=" << aT.second->mC1->Focale() << " " << aT.second->mC1->PP() << "\n"; //aT.second->mC1->Dist()
            std::cout << "C2=" << aT.second->mC2->Focale() << " " << aT.second->mC2->PP() << "\n"; //aT.second->mC2->Dist()
            std::cout << "C3=" << aT.second->mC3->Focale() << " " << aT.second->mC3->PP() << "\n"; //aT.second->mC3->Dist()
            getchar();
        }

        //all cam ids
        std::vector<int> aTriIds;
        if (aT.second->mC3)
            aTriIds = {aT.second->mId1,
                       aT.second->mId2,
                       aT.second->mId3};
        else
            aTriIds = {aT.second->mId1,
                       aT.second->mId2};


        //variable to keep track of retained points
        int a1Rec=0, a2Rec=0;
        std::vector<Pt3dr> aVPSel; 
        //back-proj the fict points to the triplet/cple
        for (int aK=0; aK<(int)aVP.size(); aK++)
        {

            std::vector<int>   aTriIdsCpy;
            std::vector<int>   aPtOutImg;
            std::vector<Pt2dr> aPImV;
            for (int aC=0; aC<int(aVC.size()); aC++)
            {

                //back-project
                Pt2dr aPt = aVC.at(aC)->Ter2Capteur(aVP.at(aK));
                //check point visibility in the image
                if (IsInSz(aPt))
                {
                    //get residual 
                    Pt2dr aCor(0.0,0.0);
                    mAR[aT.second->mId1]->ExportResXY(ApplyRedFac(aPt),aCor);

                    //check whether still inside the image
                    Pt2dr aPtCor(aPt.x+aCor.x,aPt.y+aCor.y);
                    if (IsInSz(aPtCor))
                    {
                        aPImV.push_back(aPtCor);
                        aTriIdsCpy.push_back(aTriIds.at(aC));
                    } else a2Rec++;
                } else a1Rec++;
               
                //save points if visible in at leasst 2 images 
                if (aTriIdsCpy.size() >1) 
                {
                    std::vector<float> aAttr;
                    SaveHomolOne(aTriIdsCpy,aPImV,aAttr);

                    aVPSel.push_back(aVP.at(aK));
                }


            }


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
        std::cout << "Pts out of image: " << a1Rec << " " << a2Rec << " " << int(aVPSel.size()) 
                                  << " ~" << (1-double(aVPSel.size())/ (double)aVP.size()) *100.0 << ", images:";
        for (int aC=0; aC<int(aTriIds.size()); aC++)
            std::cout << mSetName->at(aTriIds.at(aC)) << ", " ;
        std::cout << "\n";

        //print the pts
        if (mPly)
        {
            std::string Ply0Dos = "PLY-El/";
            std::string Ply1Dos = Ply0Dos + "NSym" + ToString(mNSym) + "_Pts-" + ToString(mNumFPts.x) +
                                                     ToString(mNumFPts.y) + 
                                                     ToString(mNumFPts.z) + "/" ;
            std::string Ply1File = mSetName->at(aT.second->mId1) + "-" +
                                   mSetName->at(aT.second->mId2) + "-" +
                                   (aT.second->mC3 ? mSetName->at(aT.second->mId3) : "-Cple") + "-" +
                                   "_Pts-" + ToString(mNumFPts.x) + ToString(mNumFPts.y) + ToString(mNumFPts.z);

            ELISE_fp::MkDirSvp( Ply0Dos );
            ELISE_fp::MkDirSvp( Ply1Dos );

            cPlyCloud aPlyElSel,aPlyElAll;
            for (int aK=0 ; aK<int(aVPSel.size()) ; aK++)
            {   
                aPlyElSel.AddPt(Pt3di(255,255,255),aVPSel[aK]);
            }
            for (int aK=0 ; aK<int(aVP.size()) ; aK++)
            {
                aPlyElAll.AddPt(Pt3di(255,255,255),aVP[aK]);
            }
 
            aPlyElSel.PutFile(Ply1Dos + "El-" + Ply1File + "_SEL.ply");
            aPlyElAll.PutFile(Ply1Dos + "El-" + Ply1File + "_ALL.ply");
        }
    }


    
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


    FILE* aFileExpImRes = FopenNN("StatRes.txt","w","cAppliFictObs::CalcResidPoly");
    cUseExportImageResidu aUEIR;
    aUEIR.SzByPair()    = 30;
    aUEIR.SzByPose()    = 50;
    aUEIR.SzByCam()     = 100;
    aUEIR.NbMesByCase() = 10;
    aUEIR.GeneratePly() = true;

    if (0)
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

    for (int aK=0; aK<mNbIm; aK++)
        mNameMap[mSetName->at(aK)] = aK;

    //mNM = NM(mDir);
    mNM = new cNewO_NameManager("",mPrefHom,true,mDir,mNameOriCalib,"dat");

    //triplets
    std::string aNameLTriplets = mNM->NameTopoTriplet(true);
    mLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);
    std::cout << "Triplet no: " << mLT.Triplets().size() << "\n";   

    //couples 
    std::string aNameLCple = mNM->NameListeCpleOriented(true);
    mLCpl = StdGetFromSI(aNameLCple,SauvegardeNamedRel); 
    std::cout << "Pairs no: " << mLCpl.Cple().size() << "\n";   
 
    //initialize reduced tie-points
    if (NFHom)
        mPMulRed = new cSetTiePMul(0,mSetName);
    else
        InitNFHom();


    //update triplet orientations in mTriMap
    int aTriNb=0;
    for (auto a3 : mLT.Triplets())
    {
        //verify that the triplet images are in the pattern
        if ( DicBoolFind(mNameMap,a3.Name1()) && 
             DicBoolFind(mNameMap,a3.Name2()) && 
             DicBoolFind(mNameMap,a3.Name3()) )
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
            aC1->SetOrientation(aP1);//.inv()
            aC2->SetOrientation(aP2);
            aC3->SetOrientation(aP3);

         
            mTriMap[aTriNb] = new TripleStr(aC1,mNameMap[a3.Name1()],
                                            aC2,mNameMap[a3.Name2()],
                                            aC3,mNameMap[a3.Name3()]);
         
         
            if (0)        
                std::cout << "Triplet " << aTriNb << " " << mNameMap[a3.Name1()] << "-" << a3.Name1() << " " << mNameMap[a3.Name2()] << "-" << a3.Name2() << " " << mNameMap[a3.Name3()] << "-" << a3.Name3() << "\n";
         
            aTriNb++;

        }

    }
    //update couples orientations in mTriMap
    for (auto a2 : mLCpl.Cple())
    {
        //verify that the pair images are in the pattern
        if ( DicBoolFind(mNameMap,a2.N1()) &&
             DicBoolFind(mNameMap,a2.N2()) )
        {
            //poses
            bool OK;
            ElRotation3D aP1 = ElRotation3D::Id;
            ElRotation3D aP2 = mNM->OriCam2On1 (a2.N1(),a2.N2(),OK);
            if(!OK)
                std::cout << "cAppliFictObs::Initialize() warning - no elipse3D for couple " 
                          << a2.N1() << " " << a2.N2() << "\n";
         
            CamStenope *aC1 = mNM->CamOfName(a2.N1()); 
            CamStenope *aC2 = mNM->CamOfName(a2.N2());
         
            //should handle camera-variant calibration
            if (aC1==aC2)
                aC2 = aC1->Dupl();
         
            //update poses
            aC1->SetOrientation(aP1);//.inv() 
            aC2->SetOrientation(aP2);//.inv); 
         
            mTriMap[aTriNb] = new TripleStr(aC1,mNameMap[a2.N1()],
                                            aC2,mNameMap[a2.N2()]);
            aTriNb++;
        }
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

    //triplets
    for (auto a3 : mLT.Triplets())
    {
        //symmetrique points
        InitNFHomOne(a3.Name1(),a3.Name2());
        InitNFHomOne(a3.Name2(),a3.Name1());
        
        InitNFHomOne(a3.Name1(),a3.Name3());
        InitNFHomOne(a3.Name3(),a3.Name1());

        InitNFHomOne(a3.Name2(),a3.Name3());
        InitNFHomOne(a3.Name3(),a3.Name2());


    } 

    //couples
    for (auto a2 : mLCpl.Cple())
    {
        std::string aN1 = a2.N1();
        std::string aN2 = a2.N2();
        InitNFHomOne(aN1,aN2);
        InitNFHomOne(aN2,aN1);
    }
}


int CPP_FictiveObsFin_main(int argc,char ** argv)
{
    cAppliFictObs AppliFO(argc,argv);

    return EXIT_SUCCESS;
 
}


 
int CPP_XmlOriRel2OriAbs_main(int argc,char ** argv)
{
    std::string aPattern;
    std::string aOut="Test/";
    std::string aDir;
    const std::vector<std::string> * aSetName;
    cNewO_NameManager              * aNM;   
    cCommonMartiniAppli              aCMA;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aPattern,"Pattern of images"),
        LArgMain() << EAM (aOut,"Out",true,"Output directory, Def=Test")
                   << aCMA.ArgCMA()
    );
    #if (ELISE_windows)
        replace( aPattern.begin(), aPattern.end(), '\\', '/' );
    #endif

    SplitDirAndFile(aDir,aPattern,aPattern);    
    StdCorrecNameOrient(aOut,aDir,true);


    //update the lists of couples and triplets
    std::string aCom =   MM3dBinFile("TestLib NO_AllOri2Im ") + "\"" + aPattern + "\"" + " ExpTxt=" + ToString(aCMA.mExpTxt);
    std::cout << "COM " << aCom << "\n";
    System(aCom);


    //file managers
    cElemAppliSetFile anEASF(aPattern);
    aSetName = anEASF.SetIm();
    int aNbIm = (int)aSetName->size();

    std::map<std::string,int> aNameMap;
    for (int aK=0; aK<aNbIm; aK++)
        aNameMap[aSetName->at(aK)] = aK;

    aNM = new cNewO_NameManager("","",true,aDir,aCMA.mNameOriCalib,aCMA.mExpTxt ? "txt" : "dat");

    //triplets
    std::string aNameLTriplets = aNM->NameTopoTriplet(true);

    cXml_TopoTriplet aLT;
    if (ELISE_fp::exist_file(aNameLTriplets))
    {
        aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);
        std::cout << "Triplet no: " << aLT.Triplets().size() << "\n";
    }   
    

    //couples 
    std::string aNameLCple = aNM->NameListeCpleOriented(true);
    
    cSauvegardeNamedRel aLCpl;
    if (ELISE_fp::exist_file(aNameLCple))
    {
        aLCpl = StdGetFromSI(aNameLCple,SauvegardeNamedRel);
        std::cout << "Pairs no: " << aLCpl.Cple().size() << "\n";
    }

    for (auto a3 : aLT.Triplets())
    {
        //verify that the triplet images are in the pattern
        if ( DicBoolFind(aNameMap,a3.Name1()) &&
             DicBoolFind(aNameMap,a3.Name2()) &&
             DicBoolFind(aNameMap,a3.Name3()) )
        {
            std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
            cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);


            //get the poses 
            ElRotation3D aP1 = ElRotation3D::Id ;
            ElRotation3D aP2 = Xml2El(aXml3Ori.Ori2On1());
            ElRotation3D aP3 = Xml2El(aXml3Ori.Ori3On1());

            CamStenope * aC1 = aNM->CamOfName(a3.Name1());
            CamStenope * aC2 = aNM->CamOfName(a3.Name2());
            CamStenope * aC3 = aNM->CamOfName(a3.Name3());

            //should handle camera variant calibration
            if (aC1==aC2)
                aC2 = aC1->Dupl();
            if (aC1==aC3)
                aC3 = aC1->Dupl();

            //update poses 
            aC1->SetOrientation(aP1);//.inv()
            aC2->SetOrientation(aP2);//.inv()
            aC3->SetOrientation(aP3);//.inv()

            cOrientationConique aOri1 = aC1->StdExportCalibGlob();
            cOrientationConique aOri2 = aC2->StdExportCalibGlob();
            cOrientationConique aOri3 = aC3->StdExportCalibGlob();

            ELISE_fp::MkDirSvp("Ori-"+aOut+"/");
            MakeFileXML(aOri1,"Ori-"+aOut+"/Orientation-"+StdPrefix(a3.Name1())+"."+StdPostfix(a3.Name1())+".xml");
            MakeFileXML(aOri2,"Ori-"+aOut+"/Orientation-"+StdPrefix(a3.Name2())+"."+StdPostfix(a3.Name2())+".xml");
            MakeFileXML(aOri3,"Ori-"+aOut+"/Orientation-"+StdPrefix(a3.Name3())+"."+StdPostfix(a3.Name3())+".xml");
        
        }
    }

    for (auto a2 : aLCpl.Cple())
    {
        //verify that the pair images are in the pattern
        if ( DicBoolFind(aNameMap,a2.N1()) &&
             DicBoolFind(aNameMap,a2.N2()) )
        {
            //poses
            bool OK;
            ElRotation3D aP1 = ElRotation3D::Id;
            ElRotation3D aP2 = aNM->OriCam2On1 (a2.N1(),a2.N2(),OK);

            CamStenope *aC1 = aNM->CamOfName(a2.N1());
            CamStenope *aC2 = aNM->CamOfName(a2.N2());

            //should handle camera-variant calibration
            if (aC1==aC2)
                aC2 = aC1->Dupl();

            //update poses
            aC1->SetOrientation(aP1);//.inv()
            aC2->SetOrientation(aP2);//.inv()

            cOrientationConique aOri1 = aC1->StdExportCalibGlob();
            cOrientationConique aOri2 = aC2->StdExportCalibGlob();

            ELISE_fp::MkDirSvp("Ori-"+aOut+"/");
            MakeFileXML(aOri1,"Ori-"+aOut+"/Orientation-"+StdPrefix(a2.N1())+"."+StdPostfix(a2.N1())+".xml");
            MakeFileXML(aOri2,"Ori-"+aOut+"/Orientation-"+StdPrefix(a2.N2())+"."+StdPostfix(a2.N2())+".xml");
        }
    }


    return EXIT_SUCCESS;
}


int CPP_Rot2MatEss_main(int argc,char ** argv)
{
    std::string aDir,aPattern;
    std::string aOut="MatEss";
    bool ROT2F=false;
    bool VERIF=false;
    const std::vector<std::string> * aSetName;
    cNewO_NameManager              * aNM;   
    cCommonMartiniAppli              aCMA;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aPattern,"Pattern of images"),
        LArgMain() << EAM (aOut,"Out",true,"Output directory, Def=MatEss.xml")
                   << EAM (ROT2F,"F",true,"developpers use")
                   << EAM (VERIF,"Verif",true,"developpers use")
                   << aCMA.ArgCMA()
    );
    #if (ELISE_windows)
        replace( aPattern.begin(), aPattern.end(), '\\', '/' );
    #endif

    SplitDirAndFile(aDir,aPattern,aPattern);
    StdCorrecNameOrient(aOut,aDir,true);

    //update the lists of couples and triplets
    std::string aCom =   MM3dBinFile("TestLib NO_AllOri2Im ") + "\"" + aPattern + "\"" + " ExpTxt=" + ToString(aCMA.mExpTxt);
    std::cout << "COM " << aCom << "\n";
    System(aCom);


    //file managers
    cElemAppliSetFile anEASF(aPattern);
    aSetName = anEASF.SetIm();
    int aNbIm = (int)aSetName->size();

    std::map<std::string,int> aNameMap;
    for (int aK=0; aK<aNbIm; aK++)
        aNameMap[aSetName->at(aK)] = aK;

    aNM = new cNewO_NameManager("","",true,aDir,aCMA.mNameOriCalib,aCMA.mExpTxt ? "txt" : "dat");

    //couples 
    std::string aNameLCple = aNM->NameListeCpleOriented(true);

    cSauvegardeNamedRel aLCpl;
    if (ELISE_fp::exist_file(aNameLCple))
    {
        aLCpl = StdGetFromSI(aNameLCple,SauvegardeNamedRel);
        std::cout << "Pairs no: " << aLCpl.Cple().size() << "\n";
    }
    else
    {
        ELISE_ASSERT(false,"CPP_Rot2MatEss_main no computed relative orientation in the defined pattern!")
        return EXIT_FAILURE;
    }

    for (auto a2 : aLCpl.Cple())
    {
        //verify that the pair images are in the pattern
        if ( DicBoolFind(aNameMap,a2.N1()) &&
             DicBoolFind(aNameMap,a2.N2()) )
        {
            //poses
            bool OK;
            ElRotation3D aP1 = ElRotation3D::Id;
            ElRotation3D aP2 = aNM->OriCam2On1 (a2.N1(),a2.N2(),OK);
            
            ElMatrix<double> aR  = aP2.Mat();
            Pt3dr            aTr = aP2.inv().tr();
            ElMatrix<double> aT(1,3);
            aT(0,0) = aTr.x;
            aT(0,1) = aTr.y;
            aT(0,2) = aTr.z;

            //R * [R^t t]x  , où x c'est skew matrix
            ElMatrix<double> aRtT = aR.transpose() * aT; 

            
            ElMatrix<double> aRtTx(3,3);
            aRtTx(0,1) = -aRtT(0,2);
            aRtTx(1,0) =  aRtT(0,2);
            aRtTx(0,2) =  aRtT(0,1);
            aRtTx(2,0) = -aRtT(0,1);
            aRtTx(1,2) = -aRtT(0,0);
            aRtTx(2,1) =  aRtT(0,0);

            ElMatrix<double> aMatEss = aR * aRtTx;
            std::cout << "size MatEss=" << aMatEss.Sz() << "\n";


            cTypeCodageMatr  aMatEssTCM = ExportMatr(aMatEss);
 
            std::string aNameOut = aDir + aOut + "-" + StdPrefix(a2.N1()) + "_" + StdPrefix(a2.N2()) + ".xml";
            MakeFileXML(aMatEssTCM,aNameOut);
            std::cout << "Saved to: " << aNameOut << "\n";

            //verify
            if (VERIF)
            {
                 
                cNewO_OneIm * aIm1 = new cNewO_OneIm(*aNM,a2.N1());
                cNewO_OneIm * aIm2 = new cNewO_OneIm(*aNM,a2.N2());
                
                std::vector<cNewO_OneIm *> aVI;
                aVI.push_back(aIm1);
                aVI.push_back(aIm2);
               
                tMergeLPackH aMergeStr(2,false); 
                NOMerge_AddAllCams(aMergeStr,aVI);
                aMergeStr.DoExport();               


                ElPackHomologue aPackPStd = ToStdPack(&aMergeStr,false,0);
                ElPackHomologue aPackStdRed = PackReduit(aPackPStd,1500,500);
                //ElPackHomologue aPack150 = PackReduit(aPackStdRed,100);
                //ElPackHomologue aPack30 =  PackReduit(aPack150,30);
 


                //ElRotation3D aRotVerif  = NEW_MatEss2Rot(aMatEss,(aQuick?aPack30:aPack150));
                ElRotation3D aRotVerif  = NEW_MatEss2Rot(aMatEss,aPackStdRed);
                std::cout << "aRotVerif \n" << aRotVerif.Mat()(0,0) << " " << aRotVerif.Mat()(0,1) << " " << aRotVerif.Mat()(0,2) << "\n"
                                          << aRotVerif.Mat()(1,0) << " " << aRotVerif.Mat()(1,1) << " " << aRotVerif.Mat()(1,2) << "\n"
                                          << aRotVerif.Mat()(2,0) << " " << aRotVerif.Mat()(2,1) << " " << aRotVerif.Mat()(2,2) << "\n" 
                          << "aTrVerif"    << aRotVerif.tr() << "\n";

                std::cout << "aR\n"      << aR(0,0) << " " << aR(0,1) << " " << aR(0,2) << "\n"
                                         << aR(1,0) << " " << aR(1,1) << " " << aR(1,2) << "\n"
                                         << aR(2,0) << " " << aR(2,1) << " " << aR(2,2) << "\n" 
                          << "aTr"       << aTr << "\n";

            }

        }
    }


    return EXIT_SUCCESS;
}


class cAppliMinim  : public cCommonMartiniAppli
{
    public:
        cAppliMinim(int argc,char **argv);

        void            DoCalc();

        ElRotation3D    OptimalSol(std::vector<ElMatrix<double>>& aMat, std::vector<ElSeg3D>& aDirs);
        
    private:
        ElRotation3D TestOriConvention(std::string & aNameOri, const std::string & aNameIm1, const std::string & aNameIm2) const;
        //re-written from cNewO_NameManager to avoid aN1InfN2
        ElRotation3D OriCam2On1(const std::string & aNOri1,const std::string & aNOri2,bool & OK) const;

        const std::vector<std::string>    * mSetName;
        std::map<std::string,int>           mNameMap;
        std::map<std::string,ElRotation3D*> mCplMap;
        std::map<std::string,CamStenope*>   mMapInOri;
        cNewO_NameManager                 * mNM;
        cCommonMartiniAppli                 mCMA;

        std::string                    mQIm;
        std::string                    mInOri;
        std::string                    mOut;
        cXml_TopoTriplet               mLT;
        cSauvegardeNamedRel            mLCpl;


};

cAppliMinim::cAppliMinim(int argc,char ** argv) : 
    mOut("AbsPose")
{

    std::string aDir;
    std::string aPatInOri;
    cCommonMartiniAppli aCMA;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mQIm,"The query image")
                   << EAMC(aPatInOri,"Pattern of image with known (absolute) orientation)")
                   << EAMC(mInOri,"(Absolute) orientation directory"),
        LArgMain() << EAM (mOut,"Out",true,"Output directory, Def=Test") // ???????????????????????????????????????????
                   << aCMA.ArgCMA()
    );
    #if (ELISE_windows)
        replace( aPatInOri.begin(), aPatInOri.end(), '\\', '/' );
    #endif

    SplitDirAndFile(aDir,aPatInOri,aPatInOri);
    StdCorrecNameOrient(mOut,aDir,true);
    StdCorrecNameOrient(mInOri,aDir);

    //update the lists of couples and triplets
    std::string aCom =   MM3dBinFile("TestLib NO_AllOri2Im ") + "\"" + aPatInOri + "|" + mQIm+ "\"" + " ExpTxt=" + ToString(aCMA.mExpTxt);
    if (EAMIsInit(&aCMA.mNameOriCalib))
        aCom += " OriCalib=" + aCMA.mNameOriCalib;

    std::cout << "COM " << aCom << "\n";
    System(aCom);

    //file managers
    cElemAppliSetFile anEASF(aPatInOri + "|" + mQIm);
    mSetName = anEASF.SetIm();
    int aNbIm = (int)mSetName->size();


    for (int aK=0; aK<aNbIm; aK++)
        mNameMap[mSetName->at(aK)] = aK;

    mNM = new cNewO_NameManager("","",true,aDir,aCMA.mNameOriCalib,aCMA.mExpTxt ? "txt" : "dat",mOut);

    std::cout << "======================\n" ;
    std::cout << "Query image: " << mQIm << "\n";

    //couples 
    std::string aNameLCple = mNM->NameListeCpleOriented(true);


    if (ELISE_fp::exist_file(aNameLCple))
    {
        mLCpl = StdGetFromSI(aNameLCple,SauvegardeNamedRel);
    }


    std::cout << "Known relative poses:\n" ;
    for (auto a2 : mLCpl.Cple())
    {
        if ( (DicBoolFind(mNameMap,a2.N1()) &&
             DicBoolFind(mNameMap,a2.N2()) ))
        {
            std::cout << "+ " << a2.N1() << " " << a2.N2() << "\n";
            bool OK;
            mCplMap[a2.N1() + a2.N2()] = new ElRotation3D(OriCam2On1 (a2.N1(),a2.N2(),OK));
            //mCplMap[a2.N1() + a2.N2()] = new ElRotation3D(TestOriConvention(mInOri,a2.N1(),a2.N2()));

        }
    }

    //triplets to be implemented (maybe)
    /*std::string aNameLTriplets = mNM->NameTopoTriplet(true);

    if (ELISE_fp::exist_file(aNameLTriplets))
    {
        mLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);
        std::cout << "Triplet no: " << mLT.Triplets().size() << "\n";
    }*/

    //absolute orientations
    std::cout << "Known absolute poses in " << mInOri << " :\n" ;
    for (auto a1 : *mSetName)
    {
        if (a1 != mQIm)//no InOri for the query image
        {
            mMapInOri[a1] = mNM->CamOriOfNameSVP(a1,mInOri);
            std::cout << "+ " << a1 << " " << mMapInOri[a1]->Orient().inv().tr() << "\n"; 

        }
    }

    if (0)
    {
        std::string aNameIm1 = "image_002_00112.tif";
        std::string aNameIm2 = "image_002_00119.tif";
        ElRotation3D alk = TestOriConvention(mInOri, aNameIm1, aNameIm2);        
        
        aNameIm1 = "image_002_00095.tif";
        aNameIm2 = "image_002_00119.tif";
        ElRotation3D aklm = TestOriConvention(mInOri, aNameIm1, aNameIm2);        
    } 

}

ElRotation3D cAppliMinim::OriCam2On1(const std::string & aNOri1,const std::string & aNOri2,bool & OK) const
{
    OK = false;

    std::string aN1 =  aNOri1;
    std::string aN2 =  aNOri2;

    if (!  ELISE_fp::exist_file(mNM->NameXmlOri2Im(aN1,aN2,true)))
       return ElRotation3D::Id;


    cXml_Ori2Im  aXmlO = mNM->GetOri2Im(aN1,aN2);
    OK = aXmlO.Geom().IsInit();
    if (!OK)
       return ElRotation3D::Id;
    const cXml_O2IRotation & aXO = aXmlO.Geom().Val().OrientAff();
    ElRotation3D aR12 =    ElRotation3D (aXO.Centre(),ImportMat(aXO.Ori()),true);

    OK = true;
    return aR12;

}

/*
    Calculate the absolute pose of a query image given at least two connected absolute poses (i.e. reference images)    
*/

void cAppliMinim::DoCalc()
{

    std::vector<ElMatrix<double>>  aVRQIm;
    std::vector<ElSeg3D>           aVSeg;
    std::vector<double>            aVPds;

    for (auto a2 : mMapInOri)
    {
        //absolut
        ElRotation3D aRAbs = a2.second->Orient().inv();  

        //relative
        ElRotation3D aP1 = ElRotation3D::Id;
        ElRotation3D aP2 = ElRotation3D::Id;
        ElRotation3D aRQIm = ElRotation3D::Id;


        if (DicBoolFind(mCplMap,a2.first+mQIm))
        {
            std::cout << "Pair: " << a2.first << " " << mQIm << "\n";
            aP2 = (*(mCplMap[a2.first+mQIm]));

            aRQIm = aRAbs * aP2; //

            
            if (0)
                std::cout << " atrQIm=" <<  aP2.tr() << " " << aP2.inv().tr() << "\n"; 


        }
        else if ((DicBoolFind(mCplMap,mQIm+a2.first)))
        {
            //std::cout << "Pair: " << mQIm << " " << a2.first << "\n";
            aP2 = (*(mCplMap[mQIm+a2.first]));

            aRQIm = aP2.inv() * aRAbs; // = aP2^-1 * R1abs
        
            if (0)
                std::cout << " atrQIm=" <<  aP2.tr() << " " << aP2.inv().tr() << "\n"; 
        } 
        else
        {
            std::cout << "No relative orientation for " << a2.first << "-" << mQIm << "\n";
        }


       aVRQIm.push_back(aRQIm.Mat());

       //collect 3d segments
       aVSeg.push_back(ElSeg3D(aRAbs.tr(),aRQIm.tr()));
       aVPds.push_back(1.0);

        if (0)
            std::cout << "P2 Rot\n"  << aP2.Mat()(0,0) << " " << aP2.Mat()(0,1) << " " << aP2.Mat()(0,2) << "\n"
                                     << aP2.Mat()(1,0) << " " << aP2.Mat()(1,1) << " " << aP2.Mat()(1,2) << "\n"
                                     << aP2.Mat()(2,0) << " " << aP2.Mat()(2,1) << " " << aP2.Mat()(2,2) << "\n";



        if (0)
            std::cout << "=======\n" << "Tr1 " << aRAbs.tr() << 
                           " \n" << "tr " << aRQIm.tr() << 
                           " \n" << "tr " << aP2.inv().tr() <<
                           " \n" << "Tr2 " << aRQIm.ImAff(aP2.tr()) << 
                           " \n" << "Tr2 " << aRQIm.ImAff(aP2.inv().tr()) << 
                           " \n" << "delta " << (aRAbs.tr() - aP2.tr()) <<  
                           " \n" << "tr+delta " << aP2.tr() + (aRAbs.tr() - aP2.tr()) << "\n"; 
    
    }

    ElMatrix<double> aRMoy(3,3);
    for (int aR=0; aR<int(aVRQIm.size()); aR++)
    {
        aRMoy = aRMoy + aVRQIm.at(aR);
        
        if (0)
            std::cout << "Rot of the query image\n" << aVRQIm.at(aR)(0,0) << " " << aVRQIm.at(aR)(0,1) << " " << aVRQIm.at(aR)(0,2) << "\n"
                                                << aVRQIm.at(aR)(1,0) << " " << aVRQIm.at(aR)(1,1) << " " << aVRQIm.at(aR)(1,2) << "\n"
                                                << aVRQIm.at(aR)(2,0) << " " << aVRQIm.at(aR)(2,1) << " " << aVRQIm.at(aR)(2,2) << "\n";

    }    
    aRMoy = NearestRotation(aRMoy * (1/double(aVRQIm.size())));
    
    std::cout << "======================\n" ;
    std::cout << "Rot-Moy of the query image\n" << aRMoy(0,0) << " " << aRMoy(0,1) << " " << aRMoy(0,2) << "\n"
                         << aRMoy(1,0) << " " << aRMoy(1,1) << " " << aRMoy(1,2) << "\n"
                         << aRMoy(2,0) << " " << aRMoy(2,1) << " " << aRMoy(2,2) << "\n";

    std::cout.precision(17);
    if (int(aVSeg.size()) >= 2)
    {

        ElRotation3D aTest = OptimalSol(aVRQIm,aVSeg);

        bool aIsOK;
        Pt3dr aTr = ElSeg3D::L2InterFaisceaux(&aVPds, aVSeg, &aIsOK);

        std::cout << "Tr of the query image\n" << aTr << "\n"; 
    
        std::cout << "Distance between the segments:" << "\n";
        for (int aK=0; aK<int(aVSeg.size()); aK++)
        {
            std::cout << "+ " << aVSeg.at(aK).DistDoite(aTr) << "\n";
        }


        //update \& save poses
        ElRotation3D aRes = ElRotation3D(aTr,aRMoy,true);

        CamStenope *aCQIm = mNM->CamOfName(mQIm);
        aCQIm->SetOrientation (ElRotation3D(aRes.inv().tr(),aRes.inv().Mat(),true));
        cOrientationConique aOriQIm = aCQIm->StdExportCalibGlob();

        aOriQIm.Externe().Centre() = aTr;
        
        ELISE_fp::MkDirSvp(DirOfFile(mNM->NameOriOut(mQIm)));
        MakeFileXML(aOriQIm,mNM->NameOriOut(mQIm));
        std::cout << "Saved to: " << mNM->NameOriOut(mQIm) << "\n";

        //TODO or NOT TODO: weight rotation 
        //       test on rel ori calculated from InOri, pass by Martini
        if (0)
        {
            //get min difference
            double aDifInf = 1e9;
            for (int aK=0; aK<int(aVSeg.size()); aK++)
            {
                double aDiff = euclid(aTr - aVSeg.at(aK).P1());
                if (aDiff<aDifInf)
                    aDifInf=aDiff;
            }
            //new weights
            std::vector<double> aPds2;
            for (int aK=0; aK<int(aVSeg.size()); aK++)
            {
                double aDiff = euclid(aTr - aVSeg.at(aK).P1());
                if (aDifInf == aDiff)
                    aPds2.push_back(1.0);
                else
                    aPds2.push_back( 1 / (1+ ElSquare(aDiff)) );
                
            }

            Pt3dr aTr2 = ElSeg3D::L2InterFaisceaux(&aPds2, aVSeg, &aIsOK);

            std::cout << "Tr2 of the query image\n" << aTr2 << "\n"; 
            for (int aK=0; aK<int(aVSeg.size()); aK++)
            {
                std::cout << "+ " << aVSeg.at(aK).DistDoite(aTr2) << "\n";
            }
        }

    }
    else 
        std::cout << "Not enough 3D lines to rocover the pose of the query image" << "\n";


}


ElRotation3D cAppliMinim::OptimalSol(std::vector<ElMatrix<double>>& aMat, std::vector<ElSeg3D>& aDirs)
{
    ElRotation3D aRes = ElRotation3D::Id;
    
    //do intersection of all possible pairs
    //take median with Kth
    //calculte std dev
    //sc1: remove everything over 1sigma and intersect the rest
    //sc2: take the solution that gives lowest residual

//Pt3dr aU = OneDirOrtho(aDirDroite);


    std::vector<Pt3dr> aVInter;
    for (int aK1=0; aK1<(int(aDirs.size())-1); aK1++)
    {

        for (int aK2=1; aK2<int(aDirs.size()); aK2++)
        {

            std::vector<ElSeg3D> aVSeg;
            std::vector<double>  aVPds;
            aVSeg.push_back(aDirs.at(aK1));
            aVSeg.push_back(aDirs.at(aK2));
            aVPds.push_back(1.0);
            aVPds.push_back(1.0);

            //std::cout << ",,,,,,,,,,,," << aDirs.at(aK1).P1() << " " << aDirs.at(aK2).P1() << "\n";

            bool aIsOK;
            Pt3dr aInter = ElSeg3D::L2InterFaisceaux(&aVPds, aVSeg, &aIsOK);
           
            if (aIsOK) 
            {
                //std::cout << "Inter= " << aInter << "\n"; 
                aVInter.push_back(aInter);
            }
            //else
              //  aVInter.push_back(aInter);
        }


    }


    return aRes;
    
}

ElRotation3D cAppliMinim::TestOriConvention(std::string & aNameOri, const std::string & aNameIm1, const std::string & aNameIm2) const
{
    std::cout << "InOri=" << aNameOri << ", Im1: " << aNameIm1 << ", Im2: " << aNameIm2 << "\n";

    CamStenope * aCam1 = mNM->CamOriOfNameSVP(aNameIm1,aNameOri);
    CamStenope * aCam2 = mNM->CamOriOfNameSVP(aNameIm2,aNameOri);

    // aCam2->Orient() : M =>C2  ;  aCam1->Orient().inv() :  C1=>M
    // Donc la aRot = C1=>C2
    ElRotation3D aRot = (aCam2->Orient() *aCam1->Orient().inv());
    //   Maintenat Rot C2 =>C1; donc Rot( P(0,0,0)) donne le vecteur de Base
    aRot = aRot.inv();
    aRot.tr() = vunit(aRot.tr());

    std::cout << "  +ElRot rel \n" << aRot.Mat()(0,0) << " " << aRot.Mat()(0,1) << " " << aRot.Mat()(0,2) << "\n"
                                   << aRot.Mat()(1,0) << " " << aRot.Mat()(1,1) << " " << aRot.Mat()(1,2) << "\n"
                                   << aRot.Mat()(2,0) << " " << aRot.Mat()(2,1) << " " << aRot.Mat()(2,2) << "\n";
    std::cout << "  +ElTr rel : "  << aRot.tr().x << " " << aRot.tr().y << " " << aRot.tr().z << "\n";

    return aRot;
    
}

int CPP_Rel2AbsTest_main(int argc,char ** argv)
{

    cAppliMinim aApp(argc,argv);
    aApp.DoCalc();

    return EXIT_SUCCESS;

}
