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

#include "general/all.h"
#include "private/all.h"
#include "Traj_Aj.h"

using namespace NS_AJ;


      //=================================
      //   LEARN OFFSET
      //=================================

              // Stat

/*
void cAppli_Traj_AJ::Avance(const std::vector<double> & aV,int aK0,double aVMax)
{
     while( (aK0<int(aV.size()) ) && (aV[aK0]<aVMax))
     {
            aK0++;
     }
     return aK0;
}
*/


double cAppli_Traj_AJ::LearnOffsetByStatDiff(cTAj2_OneLayerIm* aLIm,cTAj2_LayerLogIm* aLLog,const cLearnByStatDiff & aLD)
{
   std::vector<double> aVDif;
   for (int aKIm = 0 ; aKIm <aLIm->NbIm() ; aKIm++)
   {
       for (int aKLog = 0 ; aKLog <aLLog->NbLog() ; aKLog++)
       {
           double aTIm = aLIm->KthIm(aKIm)->T0();
           double aTLog = aLLog->KthLog(aKLog)->T0();
           aVDif.push_back(aTLog - aTIm);
       }
   }

   std::sort(aVDif.begin(),aVDif.end());

   double a2E = 2 * aLD.MaxEcart().Val();
   int aK0 = 0;
   int aK1 = 0;
   double aScoreMax = -1;
   double aOfsetMax=0;


   for (; aK0<int(aVDif.size()) ; aK0++)
   {
        while( (aK1<int(aVDif.size()) ) && (aVDif[aK1]<aVDif[aK0]+a2E))
        {
            aK1++;
        }
        double aScore = aK1-aK0;

        if (aScore>aScoreMax)
        {
           aScoreMax = aScore;
           aOfsetMax = (aVDif[aK0] + aVDif[aK1-1]) /2.0;
           if (0)
           {
               std::cout << "OODffa " << aOfsetMax  
                         << " " << aVDif[aK0] 
                         << " " << aVDif[aK1-1] 
                         << " " << aScore << "\n";
           }
        }
   }

   ELISE_ASSERT(aScoreMax > 0, "cAppli_Traj_AJ::LearnOffsetByStatDiff");


   std::cout << "Learn Stat " << aScoreMax 
             << " OFFSET =" << aOfsetMax
             << " dans interv avec " << aLIm->NbIm()  << " images \n";

   return aOfsetMax;
}

              // Example

double cAppli_Traj_AJ::LearnOffsetByExample(cTAj2_OneLayerIm* aLIm,cTAj2_LayerLogIm* aLLog,const cLearnByExample & aLbE)
{
    cTAj2_OneImage * aIm0 =  aLIm->ImOfName(aLbE.Im0());
    int aK0Im = aIm0->Num();
    int aK0Log =  aLbE.Log0();

    int aDeltaMin = ElMax3(aLbE.DeltaMinRech(),-aK0Im,-aK0Log);
    int aDeltaMax = ElMin3(aLbE.DeltaMaxRech(),aLIm->NbIm()-aK0Im,aLLog->NbLog()-aK0Log);

// std::cout << aLIm->NbIm() << " " << aLLog->NbLog() << "\n";
// std::cout << aLbE.DeltaMaxRech() << " " << aK0Im << " " << aK0Log << "\n";

    std::cout << aDeltaMin << " " << aDeltaMax << "\n";

    std::vector<double> aVDif;
 
    for (int aDelta=aDeltaMin ; aDelta<aDeltaMax ; aDelta++)
    {
       int aKIm = aK0Im+aDelta;
       int aKLog = aK0Log+aDelta;
       double aTIm = aLIm->KthIm(aKIm)->T0();
       double aTLog = aLLog->KthLog(aKLog)->T0();
       aVDif.push_back(aTLog - aTIm);
       if (aLbE.Show().Val())
       {
           std::cout  << aLIm->KthIm(aKIm)-> Name() << " : " << aTLog - aTIm ;
           if (aKIm >= 1)
           {
              std::cout << " DTPrec " <<  aLIm->KthIm(aKIm)->T0() -aLIm->KthIm(aKIm-1)->T0() << " ";
           }
           std::cout << "\n";
       }
    }
    std::sort(aVDif.begin(),aVDif.end());
    double aVMed= (ValPercentile(aVDif,90.0)+ValPercentile(aVDif,10.0)) /2.0;

    if (aLbE.ShowPerc().Val())
    {
       std::vector<double> aVPerc; 
       aVPerc.push_back(0); aVPerc.push_back(5); aVPerc.push_back(10); aVPerc.push_back(25);
       aVPerc.push_back(50);
       aVPerc.push_back(75); aVPerc.push_back(90); aVPerc.push_back(95); aVPerc.push_back(100);
       for (int aK=0 ; aK<int(aVPerc.size()) ; aK++)
       {
           double aPerc = aVPerc[aK];
           double aValP = ValPercentile(aVDif,aPerc);
           double aRk1 = aPerc;
           double aRk2 = (aValP-aVMed+0.5) * 100.0;

           std::cout << "Dif[" << aPerc << "%]=" << aValP <<  " Coh% " << (aRk1-aRk2) << "\n";
       }
       std::cout << "    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";
    }

    return aVMed;
}


              // Global

double cAppli_Traj_AJ::LearnOffset(cTAj2_OneLayerIm* aLIm,cTAj2_LayerLogIm* aLLog,const cLearnOffset & aLOf)
{
    if (aLOf.LearnByExample().IsInit())
    {
        return LearnOffsetByExample(aLIm,aLLog,aLOf.LearnByExample().Val());
    }
    if (aLOf.LearnByStatDiff().IsInit())
    {
        return LearnOffsetByStatDiff(aLIm,aLLog,aLOf.LearnByStatDiff().Val());
    }
    ELISE_ASSERT(false,"Internal Error cAppli_Traj_AJ::LearnOffset");
    return 0;
}


      //=================================
      //   ALGO MATCH
      //=================================

void  cAppli_Traj_AJ::DoMatchNearest(cTAj2_OneLayerIm* aLIm,cTAj2_LayerLogIm* aLLog,const cMatchNearestIm & aMI)
{

    for (int aKIm = 0 ; aKIm <aLIm->NbIm() ; aKIm++)
    {
// bool DEBUG= (aKIm==100);

        cTAj2_OneImage * anIm= aLIm->KthIm(aKIm);
        for (int aKLog = 0 ; aKLog <aLLog->NbLog() ; aKLog++)
        {
            cTAj2_OneLogIm * aLog = aLLog->KthLog(aKLog);
            double aDif = ElAbs(aLog->T0() - anIm->T0() -mCurOffset);
// if (aDif<0.52) std::cout << "DIF = " << aDif << "\n";
            if (aDif < aMI.TolAmbig())
            {
               anIm->UpdateMatch(aLog,aDif);
               aLog->UpdateMatch(anIm,aDif);
            }
        }
// if (DEBUG) getchar();
    }
// std::cout << "uuuuuuuuuUUU \n"; getchar();

/*
    int aNbOk = 0;
    int aNbNoMatch = 0;
    int aNbAmbig = 0;
    for (int aKIm = 0 ; aKIm <aLIm->NbIm() ; aKIm++)
    {
        cTAj2_OneImage * anIm= aLIm->KthIm(aKIm);
        eTypeMatch aTM = anIm->QualityMatch(aMI.TolMatch());
        anIm->SetDefQualityMatch(aTM);

        if (aTM==eMatchParfait)
        {
           aNbOk++;
           aLIm->AddMatchedIm(anIm);
        }
        else if ((aTM==eNoMatch) || (aTM==eMatchDifForte))
        {
           aNbNoMatch++;
           std::cout << anIm->Name() << " UnMatched\n";
        }
        else
        {
           aNbAmbig++;
           std::cout << anIm->Name() << " Ambigu\n";
        }
    }
    std::cout << "    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";
    std::cout << "Un Matched Images " << aNbNoMatch << "\n";
    std::cout << "Ambiguous Images " << aNbAmbig << "\n";

    aNbOk = 0;
    aNbNoMatch = 0;
    aNbAmbig = 0;
    for (int aKLog = 0 ; aKLog <aLLog->NbLog() ; aKLog++)
    {
        cTAj2_OneLogIm * aLog = aLLog->KthLog(aKLog);
        eTypeMatch aTM = aLog->QualityMatch(aMI.TolMatch());

        if (aTM==eMatchParfait)
        {
           aNbOk++;
        }
        else if ((aTM==eNoMatch) || (aTM==eMatchDifForte))
        {
           aNbNoMatch++;
        }
        else
        {
           aNbAmbig++;
        }
    }
    std::cout << "Un Matched Log " << aNbNoMatch << "\n";
    std::cout << "Ambiguous Log " << aNbAmbig << "\n";
    std::cout << "    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";
*/
}

void  cAppli_Traj_AJ::DoAlgoMatchByName
      (
           cTAj2_OneLayerIm* aLIm,
           cTAj2_LayerLogIm* aLLog,
           const cMatchByName & aMN
      )
{
   std::vector<cTAj2_OneLogIm *> & aVLs = aLLog->Logs();
   std::vector<cTAj2_OneImage *> & aVIm = aLIm-> Ims();


   for (int aKLog=0 ; aKLog<int(aVLs.size()); aKLog++)
   {
       const std::string & aKey = aVLs[aKLog]->KeyIm();
       std::string aName = mICNM->Assoc1To1(aMN.KeyLog2Im(),aKey,true);
       for (int aKIm=0 ; aKIm<int(aVIm.size()); aKIm++)
       {
           if (aVIm[aKIm]->Name() == aName)
           {
                aVIm[aKIm]->UpdateMatch(aVLs[aKLog],0);
                aVLs[aKLog]->UpdateMatch(aVIm[aKIm],0);
           }
       }
   }
}


void cAppli_Traj_AJ::DoAlgoMatch(cTAj2_OneLayerIm* aLIm,cTAj2_LayerLogIm* aLLog,const cAlgoMatch & anAlgo)
{
     aLIm->ResetMatch();
     aLLog->ResetMatch();

     double aTol = 1e20;

     if (anAlgo.MatchNearestIm().IsInit())
     {
          DoMatchNearest(aLIm,aLLog,anAlgo.MatchNearestIm().Val());
           aTol = anAlgo.MatchNearestIm().Val().TolMatch();
     }
     else if (anAlgo.MatchByName().IsInit())
     {
           DoAlgoMatchByName(aLIm,aLLog,anAlgo.MatchByName().Val());
     }
     else
     {
         ELISE_ASSERT(false,"Internal Error cAppli_Traj_AJ::LearnOffset");
     }

    int aNbOk = 0;
    int aNbNoMatch = 0;
    int aNbAmbig = 0;
    for (int aKIm = 0 ; aKIm <aLIm->NbIm() ; aKIm++)
    {
        cTAj2_OneImage * anIm= aLIm->KthIm(aKIm);
        eTypeMatch aTM = anIm->QualityMatch(aTol);
        anIm->SetDefQualityMatch(aTM);

        if (aTM==eMatchParfait)
        {
           aNbOk++;
           aLIm->AddMatchedIm(anIm);
        }
        else if ((aTM==eNoMatch) || (aTM==eMatchDifForte))
        {
           aNbNoMatch++;
           std::cout << anIm->Name() << " UnMatched\n";
        }
        else
        {
           aNbAmbig++;
           std::cout << anIm->Name() << " Ambigu\n";
        }
    }
    std::cout << "    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";
    std::cout << "Un Matched Images " << aNbNoMatch << "\n";
    std::cout << "Ambiguous Images " << aNbAmbig << "\n";

    aNbOk = 0;
    aNbNoMatch = 0;
    aNbAmbig = 0;
    for (int aKLog = 0 ; aKLog <aLLog->NbLog() ; aKLog++)
    {
        cTAj2_OneLogIm * aLog = aLLog->KthLog(aKLog);
        eTypeMatch aTM = aLog->QualityMatch(aTol);

        if (aTM==eMatchParfait)
        {
           aNbOk++;
        }
        else if ((aTM==eNoMatch) || (aTM==eMatchDifForte))
        {
           aNbNoMatch++;
        }
        else
        {
           aNbAmbig++;
        }
    }
    std::cout << "Un Matched Log " << aNbNoMatch << "\n";
    std::cout << "Ambiguous Log " << aNbAmbig << "\n";
    std::cout << "    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n";

}



      //=================================
      //   GENERE ORIENT
      //=================================

void cAppli_Traj_AJ::GenerateOrient
     (
           cTAj2_OneLayerIm* aLIm,
           const cTrAJ2_SectionMatch & aSM,
           const cTrAJ2_GenerateOrient & aGO
     )
{
   bool aTFC = aGO.Teta1FromCap().Val();
   bool aDelayGps = aGO.CorrecDelayGps().IsInit();
   bool isVReq = aTFC || aDelayGps;



   if (isVReq)
   {
      ELISE_ASSERT
      (
         aSM.ModeliseVitesse().IsInit(),
         "Vitesses required in cAppli_Traj_AJ::GenerateOrient"
      );
   }



   cSysCoord *  aCS = cSysCoord::FromXML(aGO.SysCible(),mDC.c_str());
   CamStenope * aCam = Std_Cal_From_File(DC()+aGO.NameCalib());
   for (int aKIm = 0 ; aKIm <aLIm->NbIm() ; aKIm++)
   {
        cTAj2_OneImage * anIm= aLIm->KthIm(aKIm);
///std::cout << "AAAAA "  << (anIm->DefQualityMatch()==eMatchParfait) << " " << ((!isVReq ) || (anIm->VitOK())) << "\n";
        if (  
                (anIm->DefQualityMatch()==eMatchParfait)
             && ((!isVReq ) || (anIm->VitOK()))
           )
        {
///std::cout << "BBBBB\n";
           cTAj2_OneLogIm * aLog = anIm->BestMatch();
           if (0)
           {
              std::cout << anIm->Name()  << " " << aLog->KLine() << "\n";
           }
           std::string aNameOr = DC() + mICNM->Assoc1To1(aGO.KeyName(),anIm->Name(),true);

           ElMatrix<double> aRVectCam2Av =  Std_RAff_C2M(aLIm->SIm().OrientationCamera(),aLIm->SIm().ConvOrCam().Val());

           double aCap = aLog->Teta(0);
           Pt3dr aCentre = aCS->FromGeoC(aLog->PGeoC());
           if (isVReq)
           {
               Pt3dr aP0 = aCS->FromGeoC(aLog->PGeoC());
               Pt3dr aP1 = aCS->FromGeoC(aLog->PGeoC()+anIm->Vitesse());
               Pt3dr aV = aP1 - aP0;
               double aCapV = atan2(aV.y,aV.x);
               if (aTFC) 
               {
                  aCap = aCapV;
               }
               if (aDelayGps)
               {
                    aCentre = aCentre + aV * aGO.CorrecDelayGps().Val();
               }
           }
           ElMatrix<double> aRVectCap = ElMatrix<double>::Rotation(aCap,0,0);   // oZ

           ElMatrix<double> aRouli = ElMatrix<double>::Rotation(0,aLog->Teta(1),0);  // oY
           ElMatrix<double> aTangage = ElMatrix<double>::Rotation(0,0,aLog->Teta(2)); // oX


           ElMatrix<double> aRVectAv2Ter = aRVectCap * aRouli * aTangage;
if (aKIm==0)
{
 std::cout << "ROLTANG " << aLog->Teta(0) << " " << aLog->Teta(1) << " " << aLog->Teta(2) << "\n";
ElRotation3D aR (Pt3dr(0,0,0),aRVectAv2Ter);
 std::cout << "MAT " << aR.teta01() << " " << aR.teta02() << " " << aR.teta12() << "\n";
}
/*
*/

           ElMatrix<double> aRVectCam2Ter =  aLog->MatI2C() * aRVectAv2Ter * aRVectCam2Av;


            ///ShowMatr("JJJkkk",aLog->MatI2C());
 // ElMatrix<double> aRR = aLog->MatI2C() ; std::cout << aRR.Teta12() << "\n";


           ElRotation3D aRAv2Ter (aCentre,aRVectCam2Ter);
           // ElRotation3D aRAv2Ter (aCS->FromGeoC(aLog->PGeoC()),aRVectCam2Ter);
           // cOrientationExterneRigide anOER = From_Std_RAff_C2M(aRAv2Ter,true);
           aCam->SetOrientation(aRAv2Ter.inv());

           aCam->SetAltiSol(aGO.AltiSol());

           aCam->SetTime(aLog->T0());

           cOrientationConique anOC = aCam->StdExportCalibGlob(aGO.ModeMatrix().Val());
           MakeFileXML(anOC,aNameOr);

if (aKIm==0)
{
    ElCamera * aCS = Cam_Gen_From_XML(anOC,mICNM);
    ElRotation3D aR = aCS->Orient().inv();
    std::cout << "RELEC " << aR.teta01() << " " << aR.teta02() << " " << aR.teta12() << "\n";
}

           if (TraceImage(*anIm))
           {
               std::cout << "Name " << anIm->Name() << " LINE " << aLog->KLine() << " GC " << aLog->PGeoC() << " Loc " << aCS->FromGeoC(aLog->PGeoC()) << "\n";
           }

           if (0)
           {
              std::cout << "Name " << anIm->Name() 
                      << " Loc " <<  aCS->FromGeoC(aLog->PGeoC()) 
                      << " CapINS " << aLog->Teta(0)
                      << " R " << aLog->Teta(1)
                      << " T " << aLog->Teta(2);

              if ((aKIm>0) && (aKIm<aLIm->NbIm()-1))
              {
                 cTAj2_OneImage * aIPrec= aLIm->KthIm(aKIm-1);
                 cTAj2_OneImage * aINext= aLIm->KthIm(aKIm+1);
                 if (
                           (aIPrec->DefQualityMatch()==eMatchParfait)
                        && (aINext->DefQualityMatch()==eMatchParfait)
                    )
                 {
                     cTAj2_OneLogIm * aLPrec = aIPrec->BestMatch(); 
                     cTAj2_OneLogIm * aLNext = aINext->BestMatch(); 
                     Pt3dr aVec =  aCS->FromGeoC(aLNext->PGeoC()) -aCS->FromGeoC(aLPrec->PGeoC());
                     Pt2dr aV2(aVec.x,aVec.y);
                     // std::cout << aV2 ;
                     // aV2 = aV2 / Pt2dr(0,1);
                     // double aDTeta = atan2(aV2.y,aV2.x)+ aLog->Teta(0);
                     // if (aDTeta < -PI) aDTeta += 2* PI;
                     // if (aDTeta > PI) aDTeta -= 2* PI;
                     std::cout  << " Delta TRAJ  " <<  aLog->Teta(0)-atan2(aV2.y,aV2.x) ;
                 }

              }
              std::cout  << "\n";
           }

           // getchar();
        }
   }
}
      //=================================
      //   VITESSES 
      //=================================

void cAppli_Traj_AJ::DoEstimeVitesse(cTAj2_OneLayerIm * aLIm,const cTrAJ2_ModeliseVitesse & aMV)
{
   const std::vector<cTAj2_OneImage *> & aVI = aLIm->MatchedIms() ;

   for (int aK=1 ; aK<int(aVI.size()) ; aK++)
   {
        cTAj2_OneImage * aPrec = aVI[aK-1];
        cTAj2_OneImage * aNext = aVI[aK];
        
        double aDT = aPrec->BestMatch()->T0()-aNext->BestMatch()->T0();
        bool aOK = ElAbs(aDT) < aMV.DeltaTimeMax();
        
        if (aOK)
           aNext->SetLinks(aPrec);
   }
   for (int aK=0 ; aK<int(aVI.size()) ; aK++)
   {
       aVI[aK]->EstimVitesse();
       if (! aVI[aK]->VitOK())
          std::cout << "For " << aVI[aK]->Name() << " No vitesse estimed\n";
   }
}


      //=================================
      //   GLOBAL
      //=================================

void cAppli_Traj_AJ::DoOneMatch(const cTrAJ2_SectionMatch & aSM)
{
   cTAj2_OneLayerIm * aLIm = ImLayerOfId(aSM.IdIm());
   cTAj2_LayerLogIm  * aLLog = LogLayerOfId(aSM.IdLog());


   mIsInitCurOffset = false;
   if (aSM.LearnOffset().IsInit())
   {
      mIsInitCurOffset = true;
      mCurOffset = LearnOffset(aLIm,aLLog,aSM.LearnOffset().Val());
   }


   ELISE_ASSERT
   (
          mIsInitCurOffset == (!  aSM.AlgoMatch().MatchByName().IsInit()),
        "Incohe CurOffset / MatchByName"
   );


   DoAlgoMatch(aLIm,aLLog,aSM.AlgoMatch());


   if (mIsInitCurOffset) 
      std::cout << "OFFS = " << mCurOffset << "\n";


   if (aSM.ModeliseVitesse().IsInit())
   {
       DoEstimeVitesse(aLIm,aSM.ModeliseVitesse().Val());
   }

   if (aSM.GenerateOrient().IsInit())
   {
       GenerateOrient(aLIm,aSM,aSM.GenerateOrient().Val());
   }
 
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
Footer-MicMac-eLiSe-25/06/2007*/
