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

#include "Apero.h"
#include "../TiepTri/MultTieP.h"


class cStatResPM
{
     public :
        cStatResPM();
        void Init();
        void AddStat(double anEr,double aPds,int aMult);

        double mNb;
        double mNbNN;
        double mNbMult;
        double mSomPds;
        double mSomPdsEr;
        double mSomPdsEr2;
};



class cCam_NewBD  // Dans le "void*" des cCelImTPM
{
    public :
      cCam_NewBD(cGenPoseCam *);
      cGenPoseCam * mCam;
      int           mNbPts;
      double        mPdsNb;
      cStatResPM    mStat;
};


class cConf_NewBD  // dans le "void*" 
{
    public :
       cConf_NewBD(cManipPt3TerInc *);
       cManipPt3TerInc *  mManipP3TI;
};

class cCompile_BDD_NewPtMul
{
    public :
         cCompile_BDD_NewPtMul (const cBDD_NewPtMul &,cSetTiePMul *);
         const cBDD_NewPtMul & CBN() const;
         cSetTiePMul *         SetPM() const;
    private :
         cBDD_NewPtMul               mCBN;
         cSetTiePMul *               mSetPM;
};


/**************************************************/
/*                                                */
/*                cCam_NewBD                      */
/*                cConf_NewBD                     */
/*                cStatResPM                      */
/*                                                */
/**************************************************/

    // --- cCam_NewBD ---

cCam_NewBD::cCam_NewBD(cGenPoseCam * aCam) :
    mCam    (aCam),
    mNbPts  (0)
{
}

    // --- cConf_NewBD ---

cConf_NewBD::cConf_NewBD(cManipPt3TerInc * aManipP3TI) :
  mManipP3TI (aManipP3TI)
{
}

    // --- cStatResPM ---

cStatResPM::cStatResPM()
{
   Init();
}

void cStatResPM::Init()
{
   mNb=0;
   mNbMult=0;
   mNbNN=0;
   mSomPds=0;
   mSomPdsEr=0;
   mSomPdsEr2=0;
}

void  cStatResPM::AddStat(double anEr,double aPds,int aMult)
{
   mNb++;
   if (aPds)
   {
       mNbNN++;
       mSomPds += aPds;
       mSomPdsEr += aPds * anEr;
       mSomPdsEr2 += aPds * ElSquare(anEr);
       if (aMult>2) mNbMult++;
   }
}


/**************************************************/
/*                                                */
/*              cCompile_BDD_NewPtMul             */
/*                                                */
/**************************************************/

cCompile_BDD_NewPtMul::cCompile_BDD_NewPtMul (const cBDD_NewPtMul & aCBN,cSetTiePMul * aSet) :
    mCBN   (aCBN),
    mSetPM (aSet)
{
}

const cBDD_NewPtMul & cCompile_BDD_NewPtMul::CBN() const
{
   return mCBN;
}

cSetTiePMul *   cCompile_BDD_NewPtMul::SetPM()  const
{
   return mSetPM;
}

/**************************************************/
/*                                                */
/*                                                */
/*                                                */
/**************************************************/


void cAppliApero::CompileNewPMul()
{
    // cCompile_BDD_NewPtMul
    for (int aKB=0 ; aKB<int(mVectNewBDL.size()) ; aKB++)
    {
        const cCompile_BDD_NewPtMul & aCBN = *(mVectNewBDL[aKB]);
        cSetTiePMul *  aSTM = aCBN.SetPM() ;
        const std::vector<cSetPMul1ConfigTPM *> &  aVPMC = aSTM->VPMul();

        for (int aKC=0 ; aKC<int(aVPMC.size()) ; aKC++)
        {
           
      // cCelImTPM * aCel = aSet->CelFromInt(aVIdIm[aKIdIm]);
      // cCam_NewBD * aCamNBD = static_cast<cCam_NewBD *>(aCel->ImTPM_GetVoidData());
           cSetPMul1ConfigTPM * aConf = aVPMC[aKC];
           int aNbIm = aConf->NbIm();
           const std::vector<int> & aVIdIm = aConf->VIdIm();
           std::vector<cGenPDVFormelle *> aVGPdvF;

           for (int aKIdIm = 0 ; aKIdIm<aNbIm ; aKIdIm++)
           {
              cCelImTPM * aCel = aSTM->CelFromInt(aVIdIm[aKIdIm]);
              cCam_NewBD * aCamNBD = static_cast<cCam_NewBD *>(aCel->ImTPM_GetVoidData());
              aCamNBD->mNbPts += aConf->NbPts();
              cGenPoseCam * aCamGen = aCamNBD->mCam;
              aVGPdvF.push_back(aCamGen->PDVF());
           }

      // std::cout << "CAMMM NAME " << aCam->mCam->Name() << "\n";
            cManipPt3TerInc * aM3P =  new cManipPt3TerInc(mSetEq,0,aVGPdvF,false);
            cConf_NewBD  * aConfN = new cConf_NewBD(aM3P);
            aVPMC[aKC]->ConfTPM_SetVoidData(aConfN);
        }

        // std::cout << "UUuuuUUuu " << mVectNewBDL[aKB]->CBN().Id()  << " " <<  aVPMC.size() << "\n";    
    }
    // std::cout << "cAppliApero::CompileNewPMuxxxxx\n";
    // getchar();
}

void cAppliApero::InitNewBDL()
{
    for 
    (
         std::list<cBDD_NewPtMul>::const_iterator itBDN=mParam.BDD_NewPtMul().begin() ; 
         itBDN!=mParam.BDD_NewPtMul().end() ; 
         itBDN++
    )
    {
        InitNewBDL(*itBDN);
    }
}


void cAppliApero::InitNewBDL(const cBDD_NewPtMul & aBDN)
{
     if (mDicoNewBDL[aBDN.Id()] != 0)
     {
         std::cout << "For Id = " << aBDN.Id() << "\n";
         ELISE_ASSERT(false,"cAppliApero::InitNewBDL multiple use of id in BDD_NewPtMul");
     }
     //const std::vector<std::string> *  aSTP= cSetTiePMul::StdSetName(mICNM,aBDN.SH(),false);
     const std::vector<std::string> *  aSTP= cSetTiePMul::StdSetName_BinTxt(mICNM,aBDN.SH());

     if (aSTP->size()==0) return;

     // aVNameFilter permet de filtrer les points homologues sur les poses 
     // chargees dans Apero, c'est le meme d'une fois a l'autre
     static std::vector<std::string>  * aVNameFilter=0;
     if (aVNameFilter==0)
     {
         aVNameFilter = new std::vector<std::string>; 
         for (int aKP=0 ; aKP<int(mVecGenPose.size()) ; aKP++)
         {
              const std::string & aName = mVecGenPose[aKP]->Name();
              aVNameFilter->push_back(aName);
         }
     }

    cSetTiePMul * aSet = cSetTiePMul::FromFiles(*aSTP,aVNameFilter);

    for (int aKP=0 ; aKP<int(mVecGenPose.size()) ; aKP++)
    {
        cCelImTPM *  aCBN = aSet->CelFromName(mVecGenPose[aKP]->Name());
        if (aCBN)
        {
           aCBN->ImTPM_SetVoidData(new cCam_NewBD(mVecGenPose[aKP]) );
        }
    }

    cCompile_BDD_NewPtMul * aComp = new cCompile_BDD_NewPtMul(aBDN,aSet);
    mDicoNewBDL[aBDN.Id()] = aComp;
    mVectNewBDL.push_back(aComp);
}

bool cAppliApero::CDNP_InavlideUse_StdLiaison(const std::string & aName)
{

   cCompile_BDD_NewPtMul * aCDN = CDNP_FromName(aName);

   return (aCDN!=0) && (aCDN->CBN().SupressStdHom() );
}




cCompile_BDD_NewPtMul * cAppliApero::CDNP_FromName(const std::string & aName)
{
    std::map<std::string,cCompile_BDD_NewPtMul *>::iterator anIt = mDicoNewBDL.find(aName);

    if (anIt != mDicoNewBDL.end()) return anIt->second;
    return 0;
}


     //-----------------------------------------------------------------
     //               COMPENSATION
     //-----------------------------------------------------------------

bool DEBUG_LSQ=false;

    // Compensation par configuration

void  cAppliApero::CDNP_Compense
      (
          std::vector<cStatResPM> & aVStat,
          cSetPMul1ConfigTPM* aConf,
          cSetTiePMul* aSet,
          const cObsLiaisons & anObsOl
      )
{
   cConf_NewBD * aCNBD = static_cast<cConf_NewBD *>(aConf->ConfTPM_GetVoidData());
   cManipPt3TerInc * aMP3 = aCNBD->mManipP3TI;

   const std::vector<int> & aVIdIm = aConf->VIdIm();
   int aNbIm = aConf->NbIm();
   int aNbPts = aConf->NbPts();
   std::vector<cGenPoseCam *> aVCam;
   std::vector<cGenPDVFormelle *> aVGPdvF;

   double  aPdsNb = 0.0;
   int aNbCamOk = 0;
   for (int aKIdIm = 0 ; aKIdIm<aNbIm ; aKIdIm++)
   {
      cCelImTPM * aCel = aSet->CelFromInt(aVIdIm[aKIdIm]);
      cCam_NewBD * aCamNBD = static_cast<cCam_NewBD *>(aCel->ImTPM_GetVoidData());
      aPdsNb += aCamNBD->mPdsNb;
      cGenPoseCam * aCamGen = aCamNBD->mCam;

      aNbCamOk += aCamGen->RotIsInit();
      aVCam.push_back(aCamGen);
      aVGPdvF.push_back(aCamGen->PDVF());
      // std::cout << "CAMMM NAME " << aCam->mCam->Name() << "\n";
   }
   for (int aK=aVStat.size() ; aK<=aNbIm ; aK++)
   {
       aVStat.push_back(cStatResPM());
   }


   if (aNbCamOk<2)
      return;
   cPonderateur aPdtrIm(anObsOl.Pond(),0.0);
   aPdtrIm.SetPondOfNb(aPdsNb);

   cCompFilterProj3D * aFiltre3D = 0;
   if (anObsOl.Pond().IdFilter3D().IsInit())
       aFiltre3D = FilterOfId(anObsOl.Pond().IdFilter3D().Val());


   
   // cManipPt3TerInc

   double aLimBsHP = Param().LimBsHProj().Val();
   double aLimBsHRefut = Param().LimBsHRefut().Val();
   cArg_UPL anArgUPL = ArgUPL();

   int aNAWNF = Param().NumAttrPdsNewF().Val();

   for (int aKp=0 ; aKp<aNbPts ; aKp++)
   {
       cNupletPtsHomologues aNUpl(0);
       for (int aKIm=0 ; aKIm <aNbIm ; aKIm++)
       {
           Pt2dr aPt = aConf->Pt(aKp,aKIm);
           aNUpl.AddPts(aPt);
       }

      double aPdsIm = 1.0;
      if (aNAWNF>=0)
      {
          aPdsIm = aConf->Attr(aKp,aNAWNF);
      }

      double aResidu = 0;
      int aNbEtape = 2;

      mDebugNumPts++;
      bool ToEliminate = DebugEliminateNumTieP(mDebugNumPts);
      DEBUG_LSQ= ToEliminate;
      for (int aKEtape=0 ; aKEtape<aNbEtape ; aKEtape++)
      {
          if (true)
          {
             bool WithEq = (aKEtape!=0);
             std::vector<double>   aVpds;
             for (int aKIm=0 ; aKIm <aNbIm ; aKIm++)
             {
                 double aPds = aVCam[aKIm]->RotIsInit()  ? aPdsIm : 0.0;
                 aVpds.push_back(aPds);
             }

             const cResiduP3Inc & aRes    =  aMP3->UsePointLiaison
                                          (
                                                anArgUPL,
                                                aLimBsHP,
                                                aLimBsHRefut,
                                                0.0,
                                                aNUpl,
                                                aVpds,
                                                WithEq,
                                                0
                                          );       
              if (aRes.mOKRP3I)
              {
                   ELISE_ASSERT(int(aRes.mEcIm.size()) == aNbIm,"Incoh to check in cAppliApero::CDNP_Compense");
                   for (int aKIm=0 ; aKIm<int(aRes.mEcIm.size()) ; aKIm++)
                   {
                      if (aVpds[aKIm] >0)
                      {
                         aResidu += square_euclid(aRes.mEcIm[aKIm]);//  *ElSquare(aScN);
                         if (std_isnan(aResidu))
                         {
                             std::cout <<  aRes.mEcIm[aKIm] << " " << aKIm << " " << aVCam[aKIm]->Name() << "\n";
                             // std::cout << "CPT= " << aCpt << "\n";
                             ELISE_ASSERT(false,"Nan residu\n");
                         }
                      }
                  }
                  aResidu /= aNbCamOk;
                  aResidu = sqrt(aResidu);
                  aPdsIm = aPdtrIm.PdsOfError(aResidu);
                  aPdsIm *= pow(aNbCamOk-1,anObsOl.Pond().ExposantPoidsMult().Val());

                  if (aFiltre3D && (!aFiltre3D->InFiltre(aRes.mPTer)))
                  {
                     aPdsIm = 0.0;
                  }

                  if (aPdsIm>0)
                  {
                  }
                  else
                  {
                     aNbEtape = 0;
                  }

                  // On est forcement en aPdsIm>0, sinon on 
                  if (WithEq)
                  {
                      AddInfoImageResidu(aRes.mPTer,aNUpl,aVCam,aVpds);
                      for (int aKPose=0 ; aKPose < aNbIm ; aKPose++)
                      {
                          aVCam[aKPose]->AddPMoy
                          (
                              aNUpl.PK(aKPose),
                              aRes.mPTer,
                              aRes.mBSurH,
                              aKPose,
                              &aVpds,
                              &aVCam
                          );
                      }
                  }
              }
              else
              {
                 aNbEtape = 0;
              }
           }
      }

      for (int aKIdIm = 0 ; aKIdIm<aNbIm ; aKIdIm++)
      {
         cCelImTPM * aCel = aSet->CelFromInt(aVIdIm[aKIdIm]);
         cCam_NewBD * aCamNBD = static_cast<cCam_NewBD *>(aCel->ImTPM_GetVoidData());
         aCamNBD->mStat.AddStat(aResidu,aPdsIm,aNbCamOk);
      }
      aVStat.at(aNbCamOk).AddStat(aResidu,aPdsIm,aNbCamOk);
   }

   DEBUG_LSQ= false;
   // std::cout << "-----------------------------\n";
}

    // Compensation globale 

void cAppliApero::CDNP_Compense(const std::string & anId,const cObsLiaisons & anObsOl)
{
    mDebugNumPts = 0;

    cCompile_BDD_NewPtMul * aCDN = CDNP_FromName(anId);

     if (aCDN==0)
        return; 
     
    cSetTiePMul *  aSetPM = aCDN->SetPM() ;

    //  --  Calcul des poids images ---  NbMax / (Nb+NbMax)
    const cPonderationPackMesure & aPondIm = anObsOl.Pond();


    double aNbMax = aPondIm.NbMax().Val();
    int aNbIm =  aSetPM->NbIm();
    for  (int aKIm=0 ; aKIm<aNbIm ; aKIm++)
    {
         cCelImTPM * aCel = aSetPM->CelFromInt(aKIm);
         cCam_NewBD * aCamNBD = static_cast<cCam_NewBD *>(aCel->ImTPM_GetVoidData());
         aCamNBD->mStat.Init();
         aCamNBD->mPdsNb = aNbMax / (aNbMax+ aCamNBD->mNbPts);

/*
         std::cout << "cCam_NewBD... " << aCamNBD->mCam->Name() 
                   << " " <<  aCamNBD->mPdsNb 
                   << " " <<  aCamNBD->mNbPts << " " << aNbMax
                   << " " <<  aCamNBD->mNbPts * aCamNBD->mPdsNb
                   << "\n";
*/
    }

    //----

    const std::vector<cSetPMul1ConfigTPM *> &  aVPM = aSetPM->VPMul();
    std::cout << "cAppliApero::CDNP_Compens:NBCONFIG " <<  aVPM.size() << "\n";

    double aNbTot = 0;
    for (int aKConf=0 ; aKConf<int(aVPM.size()) ; aKConf++)
    {
        cSetPMul1ConfigTPM * aConf = aVPM[aKConf];
        aNbTot +=  aConf->NbIm() * aConf->NbPts();
    }

    ElTimer aChrono;
    double  aLastTime = aChrono.uval();
    double  aPerAff = 1.0;
    double  aNbDone = 0;

    std::vector<cStatResPM>  aVStat;

    for (int aKConf=0 ; aKConf<int(aVPM.size()) ; aKConf++)
    {
        cSetPMul1ConfigTPM * aConf = aVPM[aKConf];
        CDNP_Compense(aVStat,aConf,aSetPM,anObsOl);
        aNbDone +=  aConf->NbIm() * aConf->NbPts();

        double aTime = aChrono.uval();
// std::cout << "GGggGggg " << 
        if (round_down(aLastTime/aPerAff) != round_down(aTime/aPerAff))
        {
            double aPerc = (100.0* aNbDone)/aNbTot;
            double aSpeed = aPerc/ aTime;
            std::cout << "Done " << aPerc 
                      << " \% ;  time , done " << aTime 
                      << ", left " <<   (100-aPerc) / aSpeed << "\n";
            aLastTime = aTime;
        }
    }

    std::cout << "\n";
    std::cout << "-----------------------------------------------------------------\n";
    for (int aKS=0 ; aKS<int(aVStat.size()) ; aKS++)
    {
         const cStatResPM & aStat = aVStat[aKS];
         if (aStat.mNb)
         {
             double aRes = sqrt(aStat.mSomPdsEr2 / aStat.mSomPds);
             double aPercNN = (100.0*aStat.mNbNN) / aStat.mNb;
             std::cout << " Multipl=" << aKS
                       << " NbPts="  << aStat.mNb
                       << " Res=" <<  aRes
                       << " %NN=" <<  aPercNN
                       << "\n";
             cXmlSauvExportAperoOneMult aXmlMult;
             aXmlMult.Multiplicity() = aKS;
             aXmlMult.Residual() = aRes;
             aXmlMult.NbPts() = aStat.mNb;
             aXmlMult.PercOk() = aPercNN;
              
             CurXmlE().OneMult().push_back(aXmlMult);
         }

    }
    std::cout << "-----------------------------------------------------------------\n";
    cCam_NewBD * aCamWorstRes = 0;
    cCam_NewBD * aCamWorstPerc = 0;
    double aWorstRes = -1;
    double aWorstPerc = 200;
    double aAverRes   = 0.0;
    double aSomPds   = 0.0;

    for  (int aKIm=0 ; aKIm<aNbIm ; aKIm++)
    {
         cXmlSauvExportAperoOneIm  aXml;

         cCelImTPM * aCel = aSetPM->CelFromInt(aKIm);
         cCam_NewBD * aCamNBD = static_cast<cCam_NewBD *>(aCel->ImTPM_GetVoidData());
         const cStatResPM & aStat = aCamNBD->mStat;
      
         double aRes = sqrt(aStat.mSomPdsEr2 / aStat.mSomPds);
         double aPerc =  (100.0*aStat.mNbNN) / aStat.mNb;

         aSomPds  += aStat.mSomPds;
         aAverRes += aStat.mSomPdsEr2;

         std::cout << "For pose=" << aCamNBD->mCam->Name()
                   << " NbPts="  << aStat.mNb
                   << " Res=" <<  aRes
                   << " %NN=" <<  aPerc
                   << "\n";

         if (aPerc<aWorstPerc)
         {
            aWorstPerc = aPerc;
            aCamWorstPerc = aCamNBD;
         }
         if (aRes>aWorstRes)
         {
            aWorstRes = aRes;
            aCamWorstRes = aCamNBD;
         }

         aXml.Name() =  aCamNBD->mCam->Name();
         aXml.Residual() = aRes;
         aXml.PercOk() = aPerc;
         aXml.NbPts() = aStat.mNb;
         aXml.NbPtsMul() = aStat.mNbMult;

         CurXmlE().OneIm().push_back(aXml);
    }

    if (aSomPds) 
    {
        double aSqrtEr = sqrt(aAverRes/aSomPds);
        CurXmlE().AverageResidual() = aSqrtEr;
        std::cout << " ## Average Res " <<  aSqrtEr  << "\n";
    }
    if (aCamWorstRes) 
        std::cout << " ## Worst Res " << aWorstRes << " for " << aCamWorstRes->mCam->Name() << "\n";
    if (aCamWorstPerc) 
        std::cout << " ## Worst Perc " << aWorstPerc << " for " << aCamWorstPerc->mCam->Name() << "\n";

    std::cout << "-----------------------------------------------------------------\n";
}

/*
void cAppliApero::ObsNewBDL(const cObsLiaisons & anObsL)
{
}

void cAppliApero::ObsNewBDL()
{
}

*/


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
