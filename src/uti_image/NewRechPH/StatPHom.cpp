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


#include "NewRechPH.h"

/*
class cPtFromCOPC
{
   public :
       Pt2dr operator() (cOnePCarac * aOPC) { return aOPC->Pt(); }
};

typedef ElQT<cOnePCarac*,Pt2dr,cPtFromCOPC> tQtOPC ;
*/





const std::string NH_KeyAssoc_Nuage =  "NKS-Assoc-NHNuageRef";
const std::string NH_KeyAssoc_PC    =  "NKS-Assoc-NHPtRef" ; 

class cAppliStatPHom;
class cOneImSPH;


class cStatOneLabel
{
    public :
       int mNbPtsIn;
       int mNbPtsSeuilDist;
};

class cAppliStatPHom
{
    public :
       friend class cOneImSPH;
       cAppliStatPHom(int argc,char ** argv);

       bool I1HasHom(const Pt2dr & aP1) 
       {
            ELISE_ASSERT(mNuage1!=0,"I1HasHom without Nuage");
            bool aRes = mNuage1->CaptHasData(mNuage1->Plani2Index(aP1));
            return aRes;
       }
       Pt2dr Hom(const Pt2dr & aP1);
       std::vector<cStatOneLabel>  & VLabs() {return mVLabs;}
       const cFitsParam & FP() const {return  mFP;}
       cFitsParam & FP() {return  mFP;}
       int &  NbMaxHighScale() {return mNbMaxHighScale;}
       int &  NbMaxValid() {return mNbMaxValid;}
       int &  NbMaxTested() {return mNbMaxTested;}
       double &  ScaleLim() {return mScaleLim;}
       tQtOPC * Qt2() {return  mQt2;}

       std::string & DirSaveIm() {return mDirSaveIm;}

    private :
       void ShowStat(const std::string & aMes,int aNb,std::vector<double>  aVR,double aPropMax=1.0);
       void TestHom();
       double EcartEpip(const Pt2dr & aP1,const Pt2dr & aP2);
       double EcartCompl(const Pt2dr & aP1,const Pt2dr & aP2);

       std::string mN1;
       std::string mN2;
       std::string mDir;
       cInterfChantierNameManipulateur * mICNM;
       cOneImSPH * mI1;
       cOneImSPH * mI2;
       std::string mOri;
       std::string mSH;
       std::string mNameNuage;

       cElNuage3DMaille * mNuage1;
       
       ElPackHomologue   mPack;
       bool              mSetPI;
       std::string       mExtInput;
       bool              mTestFlagBin;
       std::string       mExtOut;

       std::vector<cStatOneLabel>  mVLabs;
       cFitsParam                  mFP;
       int                         mNbMaxHighScale;
       int                         mNbMaxTested;
       int                         mNbMaxValid;
       double                      mScaleLim;
       double                      mSeuilBigRes;
       cPtFromCOPC                 mArgQt;
       tQtOPC *                    mQt2;
       std::string                 mDirSaveIm;
       std::string                 mPatLabels;
       cElRegex *                  mAutoLabels;
};

class cOneImSPH
{
    public :
         cOneImSPH(const std::string & aName,cAppliStatPHom & anAppli) ;
         void TestMatch(cOneImSPH & aI2,eTypePtRemark);

         cOnePCarac * Nearest(const Pt2dr&,double & aDMin,double aMinDMin);
         void Load(eTypePtRemark);
             

         cAppliStatPHom &   mAppli;
         std::string        mN;
         cBasicGeomCap3D *  mCam;
         cSetPCarac *       mCurSPC;
         std::vector<cOnePCarac*>  mCurAPC; // Classe par label
         std::vector<cOnePCarac*>  mVHom;
         Tiff_Im            mTif;
};


/*******************************************************/
/*                                                     */
/*           cOneImSPH                                 */
/*                                                     */
/*******************************************************/

double ScaleGen(const cOnePCarac & aPC)
{
   double aS = aPC.ScaleStab();
   if (aS<0) 
      aS= aPC.Scale();
   return aS;
}

void FiltrageValueHighestScale(std::vector<cOnePCarac*> & aVec,double aScaleLim)
{
    std::vector<cOnePCarac*> aRes;
    for (const auto & aPC : aVec)
    {
       if (ScaleGen(*aPC) >= aScaleLim)
           aRes.push_back(aPC);
    }
    std::cout << "Filtrage Scale " << aVec.size() << " => " << aRes.size() << "\n";
    aVec = aRes;
}

void FiltrageNbHighestScale(std::vector<cOnePCarac*> & aVec,int aNb,bool aShow)
{
    if (aVec.empty()) 
       return;
    std::vector<double> aVStab;
    for (const auto & aPC : aVec)
    {
        double aS = ScaleGen(*aPC);
        aVStab.push_back(aS);
    }

    // On selectionne ceux qui sont au dessus de l'echelle limite
    double aProp = 1-aNb/(double) aVec.size();
    double aScaleLim = KthValProp(aVStab,aProp);
    if (aShow)
         std::cout << "SCALE LIMITE ========== " << aScaleLim  << " " << aProp << "\n"; // getchar();
    FiltrageValueHighestScale(aVec,aScaleLim);
}


void cOneImSPH::Load(eTypePtRemark aLab)
{
   delete mCurSPC;
   mCurSPC =   LoadStdSetCarac(aLab,mN,mAppli.mExtInput);
   mCurAPC.clear();

   for (auto & aPt : mCurSPC->OnePCarac())
   {
      if (mAppli.mSetPI) 
      {
         aPt.Pt() = aPt.Pt0();
      }
      mCurAPC.push_back(&aPt);
   }
   if (EAMIsInit(&mAppli.NbMaxHighScale()))
      FiltrageNbHighestScale(mCurAPC,mAppli.NbMaxHighScale(),true);
   if (EAMIsInit(&mAppli.ScaleLim()))
      FiltrageValueHighestScale(mCurAPC,mAppli.ScaleLim());
}

cOneImSPH::cOneImSPH(const std::string & aName,cAppliStatPHom & anAppli) :
   mAppli   (anAppli),
   mN       (aName),
   mCam     (mAppli.mICNM->StdCamGenerikOfNames(mAppli.mOri,mN)),
   mCurSPC  (0),
   mTif     (Tiff_Im::StdConvGen(aName,1,true))
{
}


cOnePCarac * cOneImSPH::Nearest(const Pt2dr& aP0,double &aDMin,double aMinDMin)
{
    std::list<cOnePCarac *> aLVois2 = mAppli.Qt2()->KPPVois(aP0,2,100.0); // 100.0 = dist init
// if ( aLVois2.size()!=2)
    // std::cout << "OneImSPH::Neares " << aLVois2.size() << "\n";

    aDMin = 1e10;
    cOnePCarac * aRes = nullptr;
    for (auto & aPt : aLVois2)
    {
        double aD = euclid(aP0-aPt->Pt());
        if ((aD<aDMin) && (aD>aMinDMin))
        {
            aDMin = aD;
            aRes = aPt;
        }
    }
    // ELISE_ASSERT(aRes!=0,"cOneImSPH::Nearest");
    return aRes;
}

// Ajoute des exemples randomise de point dans la partie SRPC_Rand
void AddRand(cSetRefPCarac & aSRef,const std::vector<cOnePCarac*> aVP, int aNb)
{
   cRandNParmiQ aRNpQ(aNb,aVP.size());
   for (const auto & aPtr : aVP)
       if (aRNpQ.GetNext())
          aSRef.SRPC_Rand().push_back(*aPtr);
}

Im2D_INT1   ImOfCarac(const cOnePCarac & aPC,eTypeVecInvarR aType)
{
    // std::cout << "ImOfCarac " << aPC.ProfR().ImProfil().sz() << "\n";
    switch(aType)
    {
        // case eTVIR_Curve : return aPC.ProfR().ImProfil();
        case eTVIR_Curve : return aPC.InvR().ImRad();
        case eTVIR_ACR0  : return aPC.RIAC().IR0();
        case eTVIR_ACGT  : return aPC.RIAC().IGT();
        case eTVIR_ACGR  : return aPC.RIAC().IGR();
        case eTVIR_LogPol : 
        {
             Im2D_INT1  aILP =  aPC.ImLogPol();
             // Im2D_INT1 aRes(aILP.sz().x,aILP.sz.y);
             return aILP;
        }
        default: ;
    }
    ELISE_ASSERT(false,"ImOfCarac");
    return Im2D_INT1(1,1);
}



void cOneImSPH::TestMatch(cOneImSPH & aI2,eTypePtRemark aLab)
{
   cFitsOneLabel * aFOL = FOLOfLab(&(mAppli.FP()),aLab,true);
       // std::vector<cStatOneLabel>  mVLabs;
   // for (int aKL=0 ; aKL<int(eTPR_NoLabel) ; aKL++)
   Load(aLab);
   aI2.Load(aLab);
   mVHom.clear();
   {
        cStatOneLabel aSOL;
        aSOL.mNbPtsSeuilDist = 0;

        cFHistoInt aHLF;


        cSetRefPCarac aSetRef;
        int aDifMax = 3;
        std::vector<int>  aHistoScale(aDifMax+1,0);
        std::vector<int>  aHistoScaleStab(aDifMax+1,0); 
        double aSeuilDist = 2.0;  // Seuil distance en pixel
        double aSeuilProp = 0.02; // Seuil proportionnalité sur le fait d'etre le best, sur distance eucl 
                                  // des curves
        int aNbOk=0;

        const std::vector<cOnePCarac*>  &   aV1 = mCurAPC;  // Par compta avec vieux code
        const std::vector<cOnePCarac*>  &   aV2 = aI2.mCurAPC;

        std::vector<cOnePCarac>  aVObj1;
        std::vector<cOnePCarac>  aVSelObj1;
        {
           cRandNParmiQ aSelMaxTot(mAppli.NbMaxValid(),aV1.size());
           for (auto aPtr1 : aV1)
           {
               aVObj1.push_back(*aPtr1);
               if (aSelMaxTot.GetNext())
               {
                   aVSelObj1.push_back(*aPtr1);
               }
           }
        }

// std::cout << "GGGGGg " << aV1.size() << " " << aV2.size() << "\n"; getchar();

        aSOL.mNbPtsIn = aV1.size();
        //  int mNbPtsIn;
        //  int mNbPtsDist2;

        if ((!aV1.empty()) && (!aV2.empty()))
        {
            mAppli.Qt2()->clear();
            for (const auto & aP2 : aV2)
            {
               mAppli.Qt2()->insert(aP2); 
            }
            std::cout << "*************===========================================================*************\n";
            std::cout << "*************===========================================================*************\n";
            std::cout << "*************===========================================================*************\n";
            std::cout << "NbInit For " << eToString(aLab) << " sz=" << aV1.size() << " " << aV2.size() << "\n";

            std::vector<double> aVD22;
            for (int aK2=0 ; aK2< int(aV2.size()); aK2++)
            {
                 double aDist;
                 /*cOnePCarac * aP = */ aI2.Nearest(aV2[aK2]->Pt(),aDist,1e-5);
                 aVD22.push_back(aDist);
            }
            mAppli.ShowStat("Distribution du point le plus proche avec meme carac",20,aVD22);
      
 
            std::vector<double> aVD12;
            std::vector<double> aScorInvR;
            cRandNParmiQ aSelMaxTot(mAppli.NbMaxTested(),aV1.size());
            for (int aK1=0 ; aK1< int(aV1.size()); aK1++)
            {
                Pt2dr aP1 = aV1[aK1]->Pt();
                cOnePCarac * aHom = 0;

                if (aSelMaxTot.GetNext() && mAppli.I1HasHom(aP1))
                {
                    double aDist;
                    cOnePCarac * aP = aI2.Nearest(mAppli.Hom(aP1),aDist,-1.0);
                    aVD12.push_back(aDist);
                    if (aP && (aDist<aSeuilDist))
                    {
                         aSOL.mNbPtsSeuilDist++;
                         aNbOk++;
                         aHistoScale.at(ElMin(aDifMax,ElAbs(aV1[aK1]->NivScale() - aP->NivScale())))++;

               
                         if (0) // Affichage ScaleStab
                         // Conclusion, on peut sans doute limiter le nombre de point avec ScaleStab
                         // pour filtrage a priori => genre les 500 les plus stable
                         {
                            double aRatioS = aV1[aK1]->ScaleStab() / aP->ScaleStab();
                            aRatioS = ElAbs(log(aRatioS) / log(2));
                            double aEchS = ElAbs(log(aP->ScaleStab())) / log(2);
                            
                            std::cout << "RRRR " << aRatioS <<  " " << aEchS  <<  "  SN " << aP->NivScale() << "\n";
                         }
                         
                         // aHistoScaleStab.at(ElMin(aDifMax,ElAbs(aV1[aK1]->ScaleStab() - aP->ScaleStab())))++;
                         // SCore calcule sur les courbes
                         double aPropInv = 1 - ScoreTestMatchInvRad(aVSelObj1,aV1[aK1],aP);
                         
                         
                         aScorInvR.push_back(aPropInv);
                         // if (aNbOk%10) std::cout << "aNbOk++aNbOk++ " << aNbOk << "\n";
                         if (aPropInv < aSeuilProp)
                            aHom = aP;
                    }
                }

                mVHom.push_back(aHom);
                if (aHom)
                {
                    cSRPC_Truth aTruth;
                    aTruth.P1() = *(aV1[aK1]);
                    aTruth.P2() = *(aHom);
                    aSetRef.SRPC_Truth().push_back(aTruth);

                    if (aFOL)
                    {
                        cCompileOPC aPC1(*aV1[aK1]);
                        cCompileOPC aPC2(*aHom);
                        int aShift,aLevFail; 
                        aPC1.SetFlag(*aFOL);
                        aPC2.SetFlag(*aFOL);
                        aPC1.Match(aPC2,*aFOL,mAppli.FP().SeuilOL(),aShift,aLevFail,nullptr);
                        aHLF.Add(aLevFail);
                     }
                }
            }

            mAppli.ShowStat("Distance nearest to  homol ",20,aVD12,0.5);
            mAppli.ShowStat("Inv Rad ",20,aScorInvR,1.0);

            std::cout << "=======  Stat Dif echelle ==========\n";
            for (int aK=0 ; aK<=aDifMax ; aK++)
            {
               int aNb = aHistoScale.at(aK);
               if (aNb)
                  std::cout << "  * For dif="  << aK << " perc=" << (aNb * 100.0) / aNbOk << "\n";
            }

            if (aLab==eTPR_GrayMax)
            {
                std::cout << "======= HISTO LEV FAIL ===========\n";
                aHLF.Show();
            }

            // Test random, pas forcement tres interessant
            if (0)
            {
                for (int aNbB=1 ; aNbB<=2 ; aNbB++)
                {
                   cFullParamCB  aFB = RandomFullParamCB(*(aV1[0]),aNbB,3);
                   TestFlagCB(aFB,aV1,aV2,mVHom);
                }
            }

            if (mAppli.mTestFlagBin)
            {
                cFullParamCB  aFPB =   Optimize(true,aV1,aV2,mVHom,1);
                TestFlagCB(aFPB,aV1,aV2,mVHom);
            }
            AddRand(aSetRef,aV1,round_up(aSetRef.SRPC_Truth().size()/4.0));
            AddRand(aSetRef,aV2,round_up(aSetRef.SRPC_Truth().size()/4.0));

            std::string aExt =  eToString(aLab);
            if (EAMIsInit(&mAppli.mExtOut))
               aExt =  mAppli.mExtOut + "-" +  aExt;
            std::string aKey = NH_KeyAssoc_PC + "@"+aExt;
            std::string aName =  mAppli.mICNM->Assoc1To2(aKey,mN,aI2.mN,true);
            MakeFileXML(aSetRef,aName);

            // Export en forme d'imagette
            if (EAMIsInit(&(mAppli.DirSaveIm())))
            {
                // for (int aKL=0 ; aKL<int(eTPR_NoLabel) ; aKL++)
                {
                    // eTypePtRemark   aLabTPR = eTypePtRemark(aKL);
                    eTypePtRemark   aLabTPR = aLab;
                    for (int aKI=0 ; aKI<int(eTVIR_NoLabel) ; aKI++)
                    {
                        eTypeVecInvarR  aLabTVI = eTypeVecInvarR(aKI);
                        const std::vector<cSRPC_Truth> & aVT = aSetRef.SRPC_Truth(); 
                        int aNbSample = aVT.size(); 
                        if (aNbSample != 0)
                        {
                            Im2D_INT1 aI0 = ImOfCarac(aVT.at(0).P1(),aLabTVI);
std::cout << "SZZZZ " << aI0.sz() << "\n";
                            Pt2di aSz0 = aI0.sz();
                            Pt2di aSzGlob (aSz0.x,aSz0.y*aNbSample);
                            Im2D_U_INT1 aImGlob(2*aSzGlob.x,aSzGlob.y);

                            for (int aKS=0 ; aKS <aNbSample ; aKS++)
                            {
                                int aDy = aKS * aSz0.y;
                                Im2D_INT1 aIm1 =  ImOfCarac(aVT.at(aKS).P1(),aLabTVI);
                                Im2D_INT1 aIm2 =  ImOfCarac(aVT.at(aKS).P2(),aLabTVI);
                                ELISE_COPY
                                (
                                     rectangle(Pt2di(0,aDy),Pt2di(aSz0.x,aDy+aSz0.y)),
                                     128+ trans(aIm1.in(),Pt2di(0,-aDy)),
                                     aImGlob.out()
                                );
                                ELISE_COPY
                                (
                                     rectangle(Pt2di(aSz0.x,aDy),Pt2di(2*aSz0.x,aDy+aSz0.y)),
                                     128+ trans(aIm2.in(),Pt2di(-aSz0.x,-aDy)),
                                     aImGlob.out()
                                );
                            }
                            std::string aDir = DirApprentIR(mAppli.DirSaveIm(),aLabTPR,aLabTVI);
                            ELISE_fp::MkDirRec(aDir);
                            std::string aName = aDir + "Cple-"+ StdPrefix(mAppli.mN1) + "-" + StdPrefix(mAppli.mN2) + ".tif";


                            // Tiff_Im::CreateFromIm(aImGlob,aName);

                            L_Arg_Opt_Tiff        aLArg = Tiff_Im::Empty_ARG;
                            aLArg = aLArg + Arg_Tiff(Tiff_Im::ANoStrip());
                            Tiff_Im aFileSave
                                    (
                                        aName.c_str(),
                                        aImGlob.sz(),
                                        GenIm::u_int1,
                                        Tiff_Im::No_Compr,
                                        Tiff_Im::BlackIsZero,
                                        aLArg
                                    );
                            ELISE_COPY(aImGlob.all_pts(),aImGlob.in(),aFileSave.out());
                            // Im2D_INT1   ImOfCarac(const cOnePCarac & aPC,eTypeVecInvarR aType)
                            //std::string aDir = DirApprentIR(mAppli.DirSaveIm(),aLabTPR,aLabTVI);
                        }
                        
                        //ELISE_fp::MkDirRec(aDir);
                    }
                }
            }
        }

        mAppli.VLabs().push_back(aSOL);
        // getchar();
   }
}


/*******************************************************/
/*                                                     */
/*           cAppliStatPHom                            */
/*                                                     */
/*******************************************************/


Pt2dr cAppliStatPHom::Hom(const Pt2dr & aP1)
{
   ELISE_ASSERT(I1HasHom(aP1),"cAppliStatPHom::Hom");
   Pt2dr aI1 = mNuage1->Plani2Index(aP1);
   Pt3dr aPTer = mNuage1->PtOfIndexInterpol(aI1);
   return mI2->mCam->Ter2Capteur(aPTer);
}

void  cAppliStatPHom::ShowStat(const std::string & aMes,int aNB,std::vector<double>  aVR,double aPropMax)
{
    if (aVR.empty())
       return;

    std::cout << "=========  " << aMes << " ==========\n";
    for (int aK=0 ; aK< aNB ; aK++)
    {
        double aProp= ((aK+0.5) / double(aNB)) * aPropMax;
        std::cout << "E[" << round_ni(100.0*aProp) << "]= " << KthValProp(aVR,aProp) << "\n";
    }
    double aSom = 0.0;
    for (const auto & aV : aVR)
    {
        aSom += aV;
    }
    std::cout << "   MOY= " << aSom/aVR.size() << "\n";
}



double cAppliStatPHom::EcartEpip(const Pt2dr & aP1,const Pt2dr & aP2)
{
    return  mI1->mCam->EpipolarEcart(aP1,*mI2->mCam,aP2);
}


double cAppliStatPHom::EcartCompl(const Pt2dr & aP1,const Pt2dr & aP2)
{
    ELISE_ASSERT(mNuage1!=0,"cAppliStatPHom::EcartCompl");

    if (! I1HasHom(aP1)) return -1;
    return euclid(aP2-Hom(aP1));
}

void cAppliStatPHom::TestHom()
{
    std::vector<double> aVREpi;
    std::vector<double> aVRComp;
    ElPackHomologue aPackBigRes;
    for (cPackNupletsHom::iterator itP=mPack.begin() ; itP!=mPack.end() ; itP++)
    {
        double anEcartEpi = EcartEpip(itP->P1(),itP->P2());
        aVREpi.push_back(ElAbs(anEcartEpi));
        if (mNuage1)
        {
           double anEcartCompl = EcartCompl(itP->P1(),itP->P2());
           if (anEcartCompl>=0)
           {
               aVRComp.push_back(anEcartCompl);
               if (anEcartCompl > mSeuilBigRes)
               {
                  aPackBigRes.Cple_Add(ElCplePtsHomologues(itP->P1(),itP->P2()));
               }
           }
        }
    }
    //  La, on test la qualite des references , epipolaire et nuages
    ShowStat("ECAR EPIP pour les points SIFT",20,aVREpi);
    ShowStat("ECAR COMPLET pour les points SIFT",20,aVRComp);
    std::cout << " Perc with hom " << (aVRComp.size() * 100.0) / mPack.size() << "\n";

    std::string aNameBigRes =  mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@HigRStatH@dat",mN1,mN2,true);
    aPackBigRes.StdPutInFile(aNameBigRes);
/*
    std::cout << "NnnnnnnnnnNnn " << aNameBigRes << "\n";
    int aNB= 20;
    std::cout << "========= ECAR EPIP ==========\n";
    for (int aK=0 ; aK< aNB ; aK++)
    {
        double aProp= (aK+0.5) / double(aNB);
        std::cout << "E[" << round_ni(100.0*aProp) << "]= " << KthValProp(aVR,aProp) << "\n";
    }
*/
}


cAppliStatPHom::cAppliStatPHom(int argc,char ** argv) :
    mDir          ("./"),
    mSH           (""),
    mNuage1       (0),
    mSetPI        (false),
    mExtInput     ("Std"),
    mTestFlagBin  (false),
    mExtOut       (),
    mNbMaxHighScale  (100000000),
    mNbMaxTested  (30000),
    mNbMaxValid   (1000),
    mScaleLim     (0.0),
    mSeuilBigRes  (100.0),
    mPatLabels    (".*")
{
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mN1, "Name Image1")
                     << EAMC(mN2, "Name Image2")
                     << EAMC(mOri,"Orientation"),
         LArgMain()  << EAM(mSH,"SH",true,"Set of homologous point")
                     << EAM(mNameNuage,"NC",true,"Name of cloud")
                     << EAM(mSetPI,"SetPI",true,"Set Integer point, def=false,for stat")
                     << EAM(mExtInput,"ExtInput",true,"Extentsion for tieP")
                     << EAM(mExtOut,"ExtOut",true,"Extentsion for output")
                     << EAM(mNbMaxHighScale,"NbMaxHS",true,"Nb Max of high scale , def=infinity")
                     << EAM(mNbMaxValid,"NbMaxTot",true,"Nb Max for valid, def=1000")
                     << EAM(mNbMaxTested,"NbMaxTested",true,"Nb Max Testesd def=30000")
                     << EAM(mScaleLim,"ScaleLim",true,"Scale minimal, def=0")
                     << EAM(mDirSaveIm,"DSI",true,"DIR SAVE IMAGE (to create truth for learning)")
                     << EAM(mPatLabels,"PatLabs",true,"Pattern for labels (essentially tuning)")
   );

   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   StdCorrecNameOrient(mOri,mDir);
   StdCorrecNameHomol(mSH,mDir);
   mI1 = new cOneImSPH(mN1,*this);

   mAutoLabels = new cElRegex(mPatLabels,10);


   StdInitFitsPm(mFP);

/*
   if (1)
   {
       for (int aNbB=1 ; aNbB<4 ; aNbB++)
       {
          cFullParamCB  aFB = RandomFullParamCB(mI1->mSPC->OnePCarac()[0],aNbB,3);
          MakeFileXML(aFB,"Test_"+ToString(aNbB)+".xml");
       }
   }
*/

   mI2 = new cOneImSPH(mN2,*this);

   Pt2di aSzIm2 = mI2->mTif.sz();
   mQt2 = new tQtOPC (mArgQt,Box2dr(Pt2dr(-10,-10),Pt2dr(10,10)+Pt2dr(aSzIm2)),5,euclid(aSzIm2)/50.0);

   if (EAMIsInit(&mNameNuage))
   {
      if (mNameNuage=="") 
      {
          mNameNuage =  mICNM->Assoc1To1(NH_KeyAssoc_Nuage+"@.xml",mN1,true);
      }
      mNuage1 = cElNuage3DMaille::FromFileIm(mNameNuage);
   }

   mPack = mICNM->StdPackHomol(mSH,mI1->mN,mI2->mN);
   TestHom();


   for (int aKLab = 0 ; aKLab < int(eTPR_NoLabel) ; aKLab++)
   {
      eTypePtRemark aLab = eTypePtRemark(aKLab);

      if (mAutoLabels->Match(eToString(aLab)))
      {
         mI1->TestMatch(*mI2,aLab);
      }
   }

   
   std::cout << "=================================================\n"; 
   std::cout << "=================================================\n"; 
   std::cout << "=================================================\n"; 
   std::cout << "===========         STAT GLOBALE         ========\n"; 
   std::cout << "=================================================\n"; 

   std::cout << "   NB Pts Tapioca : " <<   mPack.size()  << "\n";
   int aNbSeuilD = 0;

   int aKStat=0;
   for (int aKLab=0 ; aKLab<int(eTPR_NoLabel) ; aKLab++)
   {
       eTypePtRemark aLab = eTypePtRemark(aKLab);
       if (mAutoLabels->Match(eToString(aLab)))
       {
          const cStatOneLabel & aSOL = mVLabs.at(aKStat);
          // eTypePtRemark aLab = (eTypePtRemark) aK;
          std::cout << "  Lab=" <<  eToString(aLab);
          std::cout << " NbSeuilD=" << aSOL.mNbPtsSeuilDist ;
          std::cout << " NbTot=" << aSOL.mNbPtsIn ;
          std::cout << "\n";
          aNbSeuilD +=  aSOL.mNbPtsSeuilDist;
          aKStat++;
       // std::vector<cStatOneLabel>  mVLabs;
       }
   }
   std::cout << " NbSeuilDist " << aNbSeuilD ;
   std::cout << "\n";
}


int  CPP_StatPHom(int argc,char ** argv)
{
    cAppliStatPHom anAppli(argc,argv);

    return EXIT_SUCCESS;
}


/*

extern const std::string NH_DirRefNuage;
extern const std::string NH_DirRef_PC;  // Point caracteristique
    
*/


/**************************************************************/
/*                                                            */
/*                  cAppli_RenameRef                          */
/*                                                            */
/**************************************************************/


class cAppli_RenameRef
{
    public :
         cAppli_RenameRef(int argc,char ** argv);
         void CpFile(std::string &,const std::string & aKind);
    private :
         cInterfChantierNameManipulateur * mICNM;
         std::string   mNameNuage;
         std::string   mNameIm;
};

void cAppli_RenameRef::CpFile(std::string & aName,const std::string & aKind)
{
/*
   std::string aDest = mPrefOut + "-" + aKind + ".tif";
   ELISE_fp::CpFile(DirOfFile(mNameNuage)+aName,aDest); 
   aName = NameWithoutDir(aDest);
*/
   std::string aDest =  mICNM->Assoc1To1(NH_KeyAssoc_Nuage+"@-"+aKind+".tif",mNameIm,true);
   ELISE_fp::CpFile(DirOfFile(mNameNuage)+aName,aDest); 
   aName = NameWithoutDir(aDest);
}

  
cAppli_RenameRef::cAppli_RenameRef(int argc,char ** argv) 
{
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mNameNuage, "Name Nuage")
                     << EAMC(mNameIm, "Name Image"),
         LArgMain()  
   );

   // C'est le numero d'étape d'une denomination standard
   if (mNameNuage.size()<=2)
   {
       mNameNuage = "MM-Malt-Img-"+ StdPrefix(mNameIm) + "/NuageImProf_STD-MALT_Etape_"+ mNameNuage + ".xml";
   }

   mICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
   std::string aNameOut =  mICNM->Assoc1To1(NH_KeyAssoc_Nuage+"@.xml",mNameIm,true);

   cXML_ParamNuage3DMaille aParam = StdGetFromSI(mNameNuage,XML_ParamNuage3DMaille);

   cImage_Profondeur & aIP = aParam.Image_Profondeur().Val();
   
   CpFile(aIP.Image(),"Prof");
   CpFile(aIP.Masq(),"Masq");
   if (aIP.Correl().IsInit())
      CpFile(aIP.Correl().Val(),"Correl");

   MakeFileXML(aParam,aNameOut);
/*
  <Image_Profondeur>
               <Image>Z_Num6_DeZoom4_STD-MALT.tif</Image>
               <Masq>AutoMask_STD-MALT_Num_5.tif</Masq>
               <Correl>Correl_STD-MALT_Num_5.tif</Correl>
*/
}


int  CPP_PHom_RenameRef(int argc,char ** argv)
{
   cAppli_RenameRef anAppli(argc,argv);
   return EXIT_SUCCESS;
}

/***************************************/

class cImADHB;
class cAppli_DistHistoBinaire;

class cImADHB
{
      friend class cAppli_DistHistoBinaire;
   public :
      cImADHB(const std::string & aName,cAppli_DistHistoBinaire &);
      std::string   mName;
      cAppli_DistHistoBinaire & mAppli;
      Im1D_REAL8    mH;
      cSetPCarac*   mPC;
      // Im1D_INT1     mHI;
};

class cAppli_DistHistoBinaire
{
      friend class cImADHB;
    public :
       cAppli_DistHistoBinaire(int argc,char ** argv);
    private :
       std::string mPatIm1;
       std::string mPatIm2;
       std::string mExt;
       std::string mNameXmlCOB;
       double      mFactConv;
       int         mNbConvol;
       cCompCB     mCOB;
       int         mNBB;
       int         mSzH;
       int         mNBI1;
       int         mNBI2;
       bool        mSingle;
       std::string mDir;
       cInterfChantierNameManipulateur * mICNM;
       std::vector<double>   mVProp;
       double                mScaleMin;
};

void FilterHistoFlag(Im1D_REAL8 aH,int aNbConvol,double aFactConv,bool DoNorm)
{
    int aSzH = aH.tx();
    int aNBB = round_ni(log2(aSzH));
    ELISE_ASSERT(aSzH==(1<<aNBB),"FilterHistoFlag pow 2 expected");

    for (int aK= 0 ; aK< aNbConvol ; aK++)
    {
        Im1D_REAL8    aNewH(aSzH,0.0) ;
        ELISE_COPY(aH.all_pts(),aH.in(),aNewH.out());
        for (int aFlag=0 ; aFlag<aSzH ; aFlag++)
        {
            for (int aB=0 ; aB<aNBB ; aB++)
            {
                 int aNewF = aFlag ^ (1<<aB);
                 aNewH.data()[aFlag] += aH.data()[aNewF] * aFactConv;
            }
        }

        double aSom;
        double aSomNew;
        ELISE_COPY(aNewH.all_pts(),aNewH.in(),sigma(aSomNew));
        ELISE_COPY(aH.all_pts(),aH.in(),sigma(aSom));

        ELISE_COPY(aH.all_pts(),aNewH.in() * (aSom/aSomNew),aH.out());
    }
    // Normalisation
    if (DoNorm)
    {
        double aS0,aS1,aS2;
        ELISE_COPY(aH.all_pts(),Virgule(1,aH.in(),Square(aH.in())),Virgule(sigma(aS0),sigma(aS1),sigma(aS2)));
        aS1 /= aS0;
        aS2 /= aS0;
        aS2 -= Square(aS1);
        aS2 = sqrt(ElMax(1e-10,aS2));
        ELISE_COPY(aH.all_pts(),(aH.in()-aS1)/aS2,aH.out());
    }
}

cImADHB::cImADHB(const std::string & aName,cAppli_DistHistoBinaire & anAppli) :
   mName  (aName),
   mAppli (anAppli),
   mH     (mAppli.mSzH,0.0),
   mPC    (LoadStdSetCarac(eTPR_NoLabel,mName,mAppli.mExt))
{
    int aSom = 0;
    int aSomDif = 0;
    // Histogramme des flag
    for (const auto & aP : mPC->OnePCarac())
    {
       if (aP.Kind() == eTPR_GrayMax)
       {
           aSom ++;
           cCompileOPC aCOPC(aP);
           int aBit=0;
           int aFlag = 0;
           for (const auto & aCOB : mAppli.mCOB.CompCBOneBit())
           {
              double aVCB = aCOPC.ValCB(aCOB);
              if (aVCB>=0)
                 aFlag |= 1<< aBit;
              aBit++;
           }
           if (mH.data()[aFlag]!=0) 
              aSomDif++;
           if (aP.ScaleStab()>= mAppli.mScaleMin)
           {
              mH.data()[aFlag] += 1;
           }
             
           // mH.data()[aFlag] += pow(2.0,aP.ScaleStab()/2.0);
           // std::cout << "CCCCCC " << aP.ScaleStab() << "\n";
       }
    }
    if (mAppli.mSingle)
       std::cout << "NB CASE!=0 " << aSomDif << " Tot=" << aSom << "\n";
    // Convolution
    FilterHistoFlag(mH,mAppli.mNbConvol, mAppli.mFactConv,true);
}

class cTestImD
{
    public :
        cTestImD(const std::string & aName,int aSzH,double aScore) :
            mSzH   (aSzH),
            mScore (aScore)
        {
        }
        double mSzH;
        double mScore;
        int mRnkH;
        int mRnkS;
};
bool CmpOnSzH(const cTestImD & aI1,const cTestImD & aI2)
{
    return  aI1.mSzH > aI2.mSzH;
}
bool CmpOnScore(const cTestImD & aI1,const cTestImD & aI2)
{
    return  aI1.mScore > aI2.mScore;
}


cAppli_DistHistoBinaire::cAppli_DistHistoBinaire(int argc,char ** argv) :
   mDir      ("./"),
   mVProp    ({0.1,0.25,0.5}),
   mScaleMin (-1)
{
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatIm1, "Name Image1")
                     << EAMC(mPatIm2, "Pattern Image2")
                     << EAMC(mExt,"Extension for Point Carac")
                     << EAMC(mFactConv,"Factor of convolution")
                     << EAM(mNbConvol,"Number of convolution")
                     << EAMC(mNameXmlCOB,"Name XML file for binary code computation"),
         LArgMain()  << EAM(mScaleMin,"ScaleMin",true,"Scale Stab min 4 use")
   );

   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   cElemAppliSetFile aS1(mPatIm1);
   cElemAppliSetFile aS2(mPatIm2);

   mCOB = StdGetFromNRPH(mNameXmlCOB,CompCB);
   mNBB = mCOB.CompCBOneBit().size();
   mSzH = 1 << mNBB;

   mNBI1 = aS1.SetIm()->size();
   mNBI2 = aS2.SetIm()->size();
   mSingle = (mNBI1==1) && (mNBI2==2);
   

   
   for (const auto & aN1 : *(aS1.SetIm()))
   {
      cImADHB aI1 (aN1,*this);
      std::vector<cTestImD>  aVI;
      for (const auto & aN2 : *(aS2.SetIm()))
      {
         cImADHB aI2 (aN2,*this);
         std::string aNH = mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@@dat",aN1,aN2,true);
         // int aSize = ELISE_fp::exist_file(aNH);
         int aSizeFileH = ELISE_fp::file_length(aNH);

         double aSom;
         ELISE_COPY(aI1.mH.all_pts(),aI1.mH.in()*aI2.mH.in(),sigma(aSom));
         double aScal = aSom / mSzH;

         if (mNBI1==1)
            std::cout << "  #  " << aN2 << " , S=" << aScal << " " << aSizeFileH << "\n";
         aVI.push_back(cTestImD(aN2,aSizeFileH,aScal));
      }
      std::sort(aVI.begin(),aVI.end(),CmpOnSzH);
      for (int aK=0 ; aK< int (aVI.size()) ; aK++)
          aVI[aK].mRnkH = aK;

      std::sort(aVI.begin(),aVI.end(),CmpOnScore);
      for (int aK=0 ; aK< int (aVI.size()) ; aK++)
          aVI[aK].mRnkS = aK;

      std::cout << "Im=" << aN1 << " ";
      for (const auto & aProp : mVProp)
      {
         int aRnk = round_up(aProp*aVI.size());
         int aNbOk=0;
         double aRealP = (aRnk / double(aVI.size()));
         for (const auto & anI : aVI)
         {
             if ((anI.mRnkH<aRnk) && ( anI.mRnkS <aRnk))
               aNbOk++;
         }
         std::cout << "[P=" << aProp <<  " Max=" << aRnk <<  " Esp=" << (aRnk*aRealP) << " Got=" << aNbOk << "] ";
      }
      std::cout << "\n";
   }
}


int  CPP_DistHistoBinaire(int argc,char ** argv)
{
   cAppli_DistHistoBinaire anAppli(argc,argv);
   return EXIT_SUCCESS;
}

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
