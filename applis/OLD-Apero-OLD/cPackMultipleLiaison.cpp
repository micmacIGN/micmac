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


#include "Apero.h"
#include "im_tpl/image.h"
#include "algo_geom/qdt_implem.h"
#include "ext_stl/numeric.h"

template  class ElQT<NS_ParamApero::cOnePtsMult *,Pt2dr,NS_ParamApero::cFctrPtsOfPMul> ;

namespace NS_ParamApero
{

#define VERIF_PackMUL 1

// bool DebugPM = false;

/**************************************************/
/*                                                */
/*             cStatObs                           */
/*                                                */
/**************************************************/

cStatObs::cStatObs(bool aAddEq) :
   mSomErPond (0),
   mAddEq     (aAddEq)
{
}

void cStatObs::AddSEP(double aSEP)
{
   mSomErPond += aSEP;
}

double cStatObs::SomErPond() const
{
   return mSomErPond;
}

bool   cStatObs::AddEq() const
{
   return mAddEq;
}


/**************************************************/
/*                                                */
/*        cOneElemLiaisonMultiple                 */
/*                                                */
/**************************************************/

cOneElemLiaisonMultiple::cOneElemLiaisonMultiple
(
    const std::string & aNameCam
)  :
   mNameCam (aNameCam),
   mPose    (0)
{
}

const std::string & cOneElemLiaisonMultiple::NameCam()
{
   return mNameCam;
}

void  cOneElemLiaisonMultiple::Compile(cAppliApero &  anAppli)
{
    if (mPose==0) 
        mPose  = anAppli.PoseFromName(mNameCam);
}

cPoseCam * cOneElemLiaisonMultiple::Pose()
{
   return mPose;
}

/**************************************************/
/*                                                */
/*           cOneCombinMult                       */
/*                                                */
/**************************************************/

cOneCombinMult::cOneCombinMult
(
    cSurfInconnueFormelle  * anEqS,
    const std::vector<cPoseCam *> & aVP,
    const std::vector<cCameraFormelle *>  & aVCF ,
    const tFixedSetInt & aFlag
)   :
   mPLiaisTer (new cManipPt3TerInc(aVCF[0]->Set(),anEqS,aVCF)),
   mVP  (aVP)
{
   for (int aK=0 ; aK<aFlag.Capacite() ; aK++)
   {
       if (aFlag.IsIn(aK))
       {
          mNumCams.push_back(aK);
       }
   }
}

cManipPt3TerInc * cOneCombinMult::LiaisTer()
{
   return mPLiaisTer;
}

const std::vector<int> & cOneCombinMult::NumCams()
{
   return mNumCams;
}

const std::vector<cPoseCam *> & cOneCombinMult::VP()
{
   return mVP;
}

int cOneCombinMult::IndOfPose(cPoseCam * aPose) const
{
    for (int aKP=0; aKP<int(mVP.size()) ; aKP++)
    {
         if (mVP[aKP]==aPose)
            return aKP;
    }
    return -1;
}

cPoseCam *  cOneCombinMult::Pose0() const
{
   return mVP.at(0);
}

cPoseCam *  cOneCombinMult::PoseK(int aK) const
{
   return mVP.at(aK);
}

void cOneCombinMult::AddLink(cAppliApero & anAppli)
{
   for (int aKP1=0; aKP1<int(mVP.size()) ; aKP1++)
   {
       for (int aKP2=aKP1; aKP2<int(mVP.size()) ; aKP2++)
       {
            anAppli.AddLinkCam(mVP[aKP1],mVP[aKP2]);
       }
   }
}

/**************************************************/
/*                                                */
/*           cOnePtsMult                         */
/*                                                */
/**************************************************/


cOnePtsMult::cOnePtsMult() :
   mMemPds (0),
   mNPts (0,1.0),
   mOCM  (0),
   mOnPlaneRapOnz (true)
{
}

bool cOnePtsMult::OnPRaz() const
{
  return mOnPlaneRapOnz;
}

void cOnePtsMult::SetOnPRaz(bool aPRaz)
{
   mOnPlaneRapOnz  = aPRaz;
}

const Pt2dr& cOnePtsMult::P0()  const
{
   return PK(0);
}

const Pt2dr& cOnePtsMult::PK(int aK ) const 
{
   return mNPts.PK(aK);
}


void cOnePtsMult::Add(int aNum,const Pt2dr & aP,bool IsFirstSet)
{
   if (mFlagI.IsIn(aNum))
   {
       if (!IsFirstSet)
          return;
#if VERIF_PackMUL
       std::cout << "NUM=" << aNum << aP << "\n";
       if (mNPts.NbPts() > 0)
          std::cout << "P0 " << mNPts.PK(0) << "\n";
       ELISE_ASSERT(false,"Multiple add in cOnePtsMult::Add");
#endif
   }
   mFlagI.Add(aNum);
   mNPts.AddPts(aP);

}

const tFixedSetInt & cOnePtsMult::Flag() const
{
   return mFlagI;
}


void cOnePtsMult::SetCombin(cOneCombinMult * anOCM)
{
   mOCM = anOCM;
}


const double & cOnePtsMult::Pds() const
{
   return mNPts.Pds();
}


cOneCombinMult * cOnePtsMult::OCM()
{
   return mOCM;
}

double & cOnePtsMult::MemPds()
{
   return mMemPds;
}




const  cNupletPtsHomologues & cOnePtsMult::NPts() const
{
   return mNPts;
}

int cOnePtsMult::NbPoseOK(bool aFullInitRequired,bool UseZU) const
{
   int aRes =0;
   const std::vector<cPoseCam *> & aVP =  mOCM->VP();

   if (UseZU && (! aVP[0]->IsInZoneU(P0())))
      return 0;

   for (int aKPose=1 ; aKPose<int(aVP.size()) ; aKPose++)
   {
       cPoseCam * aPC = aVP[aKPose];
       if  (
                 (!UseZU)
             ||  (aPC->IsInZoneU(PK(aKPose)))
           )
       {
           aRes +=    aFullInitRequired         ?
                      aPC->RotIsInit()  :
                      aPC->PreInit()    ;
       }
   }
   return aRes;

}


int   cOnePtsMult::InitPdsPMul
      (
           double aPds,
	   std::vector<double> & aVpds
      ) const
{
     aVpds.clear();
     const std::vector<cPoseCam *> & aVP =  mOCM->VP();
     int aNbRInit=0;


     for (int aKPose=0 ; aKPose<int(aVP.size()) ; aKPose++)
     {
          if ( 
                      aVP[aKPose]->RotIsInit()  
                  && aVP[0]->Calib()->IsInZoneU(PK(0))
                  && aVP[aKPose]->Calib()->IsInZoneU(PK(aKPose))
             )
         {
               aVpds.push_back(aPds);
               aNbRInit++;
         }
         else
         {
               aVpds.push_back(0);
         }
    }
    return aNbRInit;
}


ElSeg3D  cOnePtsMult::GetUniqueDroiteInit(bool UseZU)
{
   Pt2dr aPtK;
   
   const std::vector<cPoseCam *> & aVP =  mOCM->VP();
   cPoseCam * aPcK=0;
   for (int aKPose=0 ; aKPose<int(aVP.size()) ; aKPose++)
   {
       if (
                (aVP[aKPose]->RotIsInit())
             && ((!UseZU) || (aVP[aKPose]->IsInZoneU(PK(aKPose))))
          )
       {
           ELISE_ASSERT(aPcK==0,"cOnePtsMult::GetUniqueDroiteInit Non Unique");
//std::cout << "cOnePtsMult::GetUniqueDroiteInit Non Unique" << "\n";
           aPcK = aVP[aKPose];
           aPtK = PK(aKPose);
// std::cout << "XXXXX " << aPcK->Name() << " "<< aKPose << " " << aVP.size() << "\n";
       }
   }
   ELISE_ASSERT(aPcK!=0,"cOnePtsMult::GetUniqueDroiteInit Aucune");
 
   const CamStenope & aCS =   * (aPcK->CF()->CameraCourante());

// std::cout << "ttttt " << aPtK << aCS.CentreOptique() << "\n";

   return ElSeg3D
          (
              aCS.ImEtProf2Terrain(aPtK,9),
              aCS.ImEtProf2Terrain(aPtK,11)
          );
   
}

const cResiduP3Inc * cOnePtsMult::ComputeInter
                     (
                         double aPds,
                         std::vector<double> & aVPds
                     ) const
{
     int aNbRInit= InitPdsPMul(1.0,aVPds);
     if (aNbRInit>=2)
     {
         const cResiduP3Inc  & aRes =  (mOCM->LiaisTer()->UsePointLiaison(-1,-1,0.0,mNPts,aVPds,false));

         if (! aRes.mOKRP3I)
            return 0;

         return & aRes;
     }
     return 0;
}

Pt3dr InterFaisceaux
      (
           const std::vector<double> & aVPds,
           const std::vector<cPoseCam *> & aVC,
           const cNupletPtsHomologues  &   aNPt
      )
{
     std::vector<ElSeg3D> aVSeg;
     for (int aKR=0; aKR<int(aVC.size()) ; aKR++)
     {
         if ( (aKR>=int(aVPds.size())) || (aVPds[aKR] >0))
         {
             const CamStenope * aCS =   aVC[aKR]->CF()->CameraCourante();
             ElSeg3D aSeg = aCS->F2toRayonR3(aNPt.PK(aKR));
             aVSeg.push_back(aSeg);
         }
    }


    Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);
    return aRes;
}

Pt3dr TestInterFaisceaux
      (
           const std::vector<cPoseCam *> & aVC,
           const cNupletPtsHomologues  &   aNPt,
           double                          aSigma,
           bool                            Show
      )
{
   ELISE_ASSERT (aVC.size() >= 2,"TestInterFaisceaux" );
           
      double aSomPMax = 0.0;
      Pt3dr aPMax(0,0,0);

      for (int aK1=0 ; aK1 <int( aVC.size()) ; aK1++)
      {
          const CamStenope * aCS1 =   aVC[aK1]->CF()->CameraCourante();
          Pt2dr aP1 = aNPt.PK(aK1);
          ElSeg3D aSeg1 = aCS1->F2toRayonR3(aP1);
          for (int aK2=aK1+1 ; aK2 <int( aVC.size()) ; aK2++)
          {
              const CamStenope * aCS2 =   aVC[aK2]->CF()->CameraCourante();
              Pt2dr aP2 = aNPt.PK(aK2);
              ElSeg3D aSeg2 = aCS2->F2toRayonR3(aP2);

              Pt3dr aPTest = aSeg1.PseudoInter(aSeg2);
              double aSomP = 0.0;

              for (int aK3 = 0 ; aK3<int( aVC.size()) ; aK3++)
              {
                  const CamStenope * aCS3 =   aVC[aK3]->CF()->CameraCourante();
                  Pt2dr aP3 = aNPt.PK(aK3);
                  Pt2dr aQ3 = aCS3->R3toF2(aPTest);
                  double aDist = euclid(aP3,aQ3);

                  aSomP += exp(-ElSquare(aDist)/ (2*ElSquare(aSigma)));
              }

               if (aSomP>aSomPMax)
               {
                    aSomPMax = aSomP;
                    aPMax = aPTest;
               }
          }
      }

   if (Show)
   {
       for (int aK3 = 0 ; aK3<int( aVC.size()) ; aK3++)
       {
            const CamStenope * aCS3 =   aVC[aK3]->CF()->CameraCourante();
            Pt2dr aP3 = aNPt.PK(aK3);
            Pt2dr aQ3 = aCS3->R3toF2(aPMax);
            double aDist = euclid(aP3,aQ3);

            std::cout << "TFI::" << aVC[aK3]->Name() << " Residu Image= " << aDist ; 
            if (aDist > 20) std::cout << "    ########################";
            std::cout  << "\n";
       }
   }
   return aPMax;
}




Pt3dr cOnePtsMult::QuickInter(std::vector<double> & aVPds) const
{
   return InterFaisceaux(aVPds,mOCM->VP(),mNPts);
/*
     std::vector<ElSeg3D> aVSeg;
     aVSeg.clear();
     const std::vector<cPoseCam *> & aVC = mOCM->VP();
     for (int aKR=0; aKR<int(aVPds.size()) ; aKR++)
     {
         if (aVPds[aKR] >0)
         {
             CamStenope & aCS =   * (aVC[aKR]->Calib()->PIF().CurPIF());
             ElSeg3D aSeg = aCS.F2toRayonR3(PK(aKR));
             aVSeg.push_back(aSeg);
         }
    }

    Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);
    // ElSeg3D::L2InterFaisceaux(0,aVSeg,0);

    return aRes;
*/
}






int cOnePtsMult::IndOfPose(cPoseCam * aPose) const
{
   return mOCM->IndOfPose(aPose);
}

cPoseCam *  cOnePtsMult::Pose0() const
{
   return mOCM->Pose0();
}
cPoseCam *  cOnePtsMult::PoseK(int aK) const
{
   return mOCM->PoseK(aK);
}

/**************************************************/

/**************************************************/
/*                                                */
/*           cFctrPtsOfPMul                       */
/*                                                */
/**************************************************/

Pt2dr cFctrPtsOfPMul::operator()(cOnePtsMult *const &  aPMul) const
{
   return aPMul->P0();
}



/**************************************************/
/*                                                */
/*        cObsLiaisonMultiple                     */
/*                                                */
/**************************************************/


cObsLiaisonMultiple::cObsLiaisonMultiple
(
    cAppliApero &       anAppli,
    const std::string & aNamePack,
    const std::string & aName1,
    const std::string & aName2,
    bool                isFirstSet
)  :
    mAppli       (anAppli),
    mSurf        (0),
    mEqS         (0),
    mCompilePoseDone (false),
    mKIm         (0),
    mRazGlob     (0),
    mLayerImDone (false)
{
   {
     mPose1 = mAppli.PoseFromName(aName1);
     Pt2dr aSz = Pt2dr(mPose1->Calib()->SzIm());
     double aRab =  euclid(aSz)/60.0;
     Pt2dr aPRab(aRab,aRab);
     cFctrPtsOfPMul aPrim;
     mBox = Box2dr(-aPRab,aSz+aPRab);
     mIndPMul = new  tIndPMul (aPrim,mBox,20,10.0);
   }

   // std::cout << "SzC " << aPos1->Calib()->SzIm() << "\n";
   // Tiff_Im aTif1 = Tiff_Im::StdConv(mAppli.DC()+ aName1);



   // std::cout << "cObsLiaisonMultiple : " << aName1 << " " << aName2 << "\n";
   // getchar();
   AddPose(aName1,isFirstSet);
   AddPose(aName2,isFirstSet);

   AddPack(aNamePack,aName1,aName2,isFirstSet);
}

cPoseCam *  cObsLiaisonMultiple::Pose1() const
{
    return mPose1;
}


void cObsLiaisonMultiple::AddLink()
{
   for
   (
      std::map<tFixedSetInt, cOneCombinMult*>::const_iterator itD=mDicoMP3TI.begin();
      itD!=mDicoMP3TI.end();
      itD++
   )
   {
       itD->second->AddLink(mAppli);
   }
}

void cObsLiaisonMultiple::AddPack
     (
         const std::string & aNamePack,
         const std::string & aName1,
         const std::string & aName2,
         bool                IsFirstSet
     )
{
   int anInd1 = GetIndexeOfName(aName1);
   int anInd2 = GetIndexeOfName(aName2);

   ElPackHomologue aPack = ElPackHomologue::FromFile(aNamePack);
   cPoseCam * aC1 =  mAppli.PoseFromName(aName1);
   cPoseCam * aC2 =  mAppli.PoseFromName(aName2);

   // CamStenope & aCC1 = aC1->Calib()-> CamInit();
   // CamStenope & aCC2 = aC2->Calib()-> CamInit();


   ElPackHomologue aNewPack;

   for (ElPackHomologue::iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
   {
      aC1->C2MCompenseMesureOrInt(itP->P1());
      aC2->C2MCompenseMesureOrInt(itP->P2());

      if (aC1->AcceptPoint(itP->P1()) && aC2->AcceptPoint(itP->P2()))
      {
          aNewPack.Cple_Add(ElCplePtsHomologues(itP->P1(),itP->P2()));
      }
   }
   aPack = aNewPack;

   mAppli.AddLinkCam(aC1,aC2);
   // aC1->AddLink(aC2);
   // aC2->AddLink(aC1);



   for (ElPackHomologue::const_iterator itP=aPack.begin();itP!=aPack.end();itP++)
   {
       AddCple(anInd1,anInd2,itP->ToCple(),IsFirstSet);
   }
}


void cObsLiaisonMultiple::AddCple(int aK1,int aK2,const ElCplePtsHomologues& aCpl,bool IsFirstSet)
{

    std::list<cOnePtsMult *> aLPM = mIndPMul->KPPVois
                                    (
                                         aCpl.P1(),
                                         1, // 1 Voisin
                                         1e-5, // dist initiale
                                         -1, // INUTILE CAR NbTest = 1
                                         1   // 1 seul test
                                    );



    cOnePtsMult * aPM=0;
    if (aLPM.empty())
    {
         aPM = new cOnePtsMult;
         aPM->Add(aK1,aCpl.P1(),IsFirstSet);
         mVPMul.push_back(aPM);
         if (! mBox.Include(aCpl.P1()))
         {
             std::cout << "FOR Pt " <<  aCpl.P1() << " Box = " << mBox._p0 << mBox._p1 << "\n";
             ELISE_ASSERT(false,"POINT HOM OUT OF IMAGE");
         }
// std::cout << "==DO " << aCpl.P1() << aCpl.P2() << "\n";
         mIndPMul->insert(aPM);
// std::cout << "==DONE \n";
    }
    else
    {
        aPM = *(aLPM.begin());
    }
    aPM->Add(aK2,aCpl.P2(),IsFirstSet);
}

int  cObsLiaisonMultiple::GetIndexeOfName(const std::string & aName)
{
    for (int aK=0 ; aK<int(mVPoses.size()) ; aK++)
    {
        if (mVPoses[aK]->NameCam() == aName)
	   return aK;
    }
    return -1;
}


int  cObsLiaisonMultiple::IndOfCam(const cPoseCam * aCam) const
{
    for (int aK=0 ; aK<int(mVPoses.size()) ; aK++)
    {
        if (mVPoses[aK]->Pose() == aCam)
	   return aK;
    }
    return -1;
}


Pt3dr cObsLiaisonMultiple::CentreNuage() const
{
  std::vector<double> aVPds;

  const CamStenope &   aCS  = *(mPose1->CurCam());
  std::vector<double> aVProf;
  Pt2dr aPMoy(0,0);

  for (int aKPt=0 ; aKPt<int(mVPMul.size()) ;aKPt++)
  {
      cOnePtsMult& anOPM = *(mVPMul[aKPt]);
      if (anOPM.MemPds() >0)
      {
           Pt3dr aPI = anOPM.QuickInter(aVPds);
           aPMoy = aPMoy+ anOPM.P0();
           aVProf.push_back(aCS.ProfondeurDeChamps(aPI));
      }
  }

  ELISE_ASSERT(aVProf.size()!=0,"cObsLiaisonMultiple::CentreNuage No Point");

  aPMoy = aPMoy/double(aVProf.size()),

  std::sort(aVProf.begin(),aVProf.end());
  double aProf  = ValPercentile(aVProf,0.5);

  return aCS.ImEtProf2Terrain(aPMoy,aProf);
}


/*
*/



void    cObsLiaisonMultiple::AddPose(const std::string & aName,bool IsFirstSet)
{
    if (GetIndexeOfName(aName)!=-1)
    {
         if (! IsFirstSet) 
            return;
#if VERIF_PackMUL
         std::cout  << "Name =" << aName << "\n";
         ELISE_ASSERT(false,"cObsLiaisonMultiple::cAddPose");
#endif
/*
*/
    }
    mVPoses.push_back(new cOneElemLiaisonMultiple(aName));
}

bool cObsLiaisonMultiple::InitPack(ElPackHomologue & aPack, const std::string& aN2)
{
   int aKPack2=GetIndexeOfName(aN2);

   if (aKPack2<=0) 
      return false;

   aPack.clear();
   for (int aKPts=0 ; aKPts<int(mVPMul.size()) ; aKPts++)
   {
      const tFixedSetInt & aFlag =  mVPMul[aKPts]->Flag();
      if (aFlag.IsIn(aKPack2)&&aFlag.IsIn(0))
      {
             const cNupletPtsHomologues & aNP = mVPMul[aKPts]->NPts();
             aPack.Cple_Add
             (
                 ElCplePtsHomologues
                 (
                     aNP.PK(aFlag.NumOrdre(0)),
                     aNP.PK(aFlag.NumOrdre(aKPack2))
                 )
             );
      }
   }

   return true;
}

cOnePtsMult & cObsLiaisonMultiple::PMulLPP(Pt2dr aPt)
{
    return *(mIndPMul->NearestObj(aPt,10,1e5));


}

const std::vector<cOnePtsMult *> & cObsLiaisonMultiple::VPMul()
{
   return mVPMul;
}

void cObsLiaisonMultiple::AddLiaison(const std::string & aNamePack,const std::string & aName2,bool isFirstSet)
{
   AddPose(aName2,isFirstSet);
   AddPack(aNamePack,mVPoses[0]->NameCam(),aName2,isFirstSet);
}


void  cObsLiaisonMultiple::Compile
      (
	    cSurfParam  * aSurf
      )
{
    mSurf = aSurf;
    if (aSurf!=0)
    {
       mEqS = aSurf->EqSurfInc();
    }
    CompilePose();
}


void  cObsLiaisonMultiple::CompilePose()
{
    if (mCompilePoseDone)
       return;
    mCompilePoseDone = true;

    mNbPts =0.0;
    mSomPds = 0.0;

    for (int aK=0 ; aK<int(mVPoses.size()) ; aK++)
    {
        mVPoses[aK]->Compile(mAppli);
    }


    for (int aKM=0; aKM<int(mVPMul.size()); aKM++)
    {

        cOneCombinMult * anOCM = AddAFlag(*(mVPMul[aKM]));
        mVPMul[aKM]->SetCombin(anOCM);
        mSomPds += mVPMul[aKM]->Pds();
        mNbPts ++;

        ELISE_ASSERT
        (
             mVPMul[aKM]->Pose0()==mVPoses.at(0)->Pose(),
             "Coherence in cObsLiaisonMultiple::Compile"
        );
    }
    mMultPds = mNbPts / mSomPds ;

}


const std::vector<cOneElemLiaisonMultiple *> &  cObsLiaisonMultiple::VPoses() const
{
   return mVPoses;
}

void  CompleteSurfParam()
{
    ELISE_ASSERT(false,"Comlete SurfParam !!");
}

cOneCombinMult *  cObsLiaisonMultiple::AddAFlag(const cOnePtsMult & aPM)
{
    const tFixedSetInt & aFlag = aPM.Flag();

    cOneCombinMult * aRes = mDicoMP3TI[aFlag];
    if (aRes!=0)
       return aRes;

    std::vector<cCameraFormelle *>  aVCF;
    std::vector<cPoseCam *>  aVP;
    for (int aK=0 ; aK <int(mVPoses.size()) ; aK++)
    {
         if (aFlag.IsIn(aK))  // (aFlag & (1<<(aK-1))()
         {
             aVCF.push_back(mVPoses[aK]->Pose()->CF());
	     aVP.push_back(mVPoses[aK]->Pose());
         }
    }
    aRes =  new cOneCombinMult(mEqS,aVP,aVCF,aFlag);
    mDicoMP3TI[aFlag] = aRes;
    return aRes;
}

/*
cOnePtsMult * cObsLiaisonMultiple::CreateNewPM
              (
                  const std::vector<double> &       aVPds,
                  const std::vector<cPoseCam*>  &   aVPC,
                  const cNupletPtsHomologues    &   aNuple
              )
{
    
}
*/

class cOneStatDet
{
   public :
      cOneStatDet() :
          mSomPds (0),
          mSomNormPds (0),
          mNb         (0)
      {
      }

      void Add(Pt2dr aPt,double aPds)
      {
          mNb++;
          mSomPds+= aPds;
          mSomNormPds += aPds * euclid(aPt);
      }
      
   // private :
      double mSomPds;
      double mSomNormPds;
      int    mNb;

};

class cGlobStatDet
{
 
     public :
          void Add(cPoseCam * aCam,Pt2dr aPt,double aPds);
          void Show();
     private :
          std::map<cPoseCam *,cOneStatDet> mMap;
          
};

void cGlobStatDet::Add(cPoseCam * aCam,Pt2dr aPt,double aPds)
{
     mMap[aCam].Add(aPt,aPds);
}

void cGlobStatDet::Show()
{
    for (std::map<cPoseCam *,cOneStatDet>::const_iterator  itC=mMap.begin(); itC!=mMap.end(); itC++ )
    {
       std::cout << itC->first->Name()  << " "
                 << " ErMoy " << itC->second.mSomNormPds / itC->second.mSomPds
                 << " Nb=" << itC->second.mNb
                 << "\n";
    }
}



double cObsLiaisonMultiple::AddObsLM
       (
           const cPonderationPackMesure & aImPPM,
	   const cPonderationPackMesure * aPPMSurf,
           cArgGetPtsTerrain * anArgPT,
           cArgVerifAero     *  anAVA,
           cStatObs &           aSO,
           const cRapOnZ *      aRAZGlob
       )
{

  FILE * aFpRT = mAppli.FpRT() ;

  for (int aKP=0 ;  aKP<int(mVPoses.size()) ; aKP++)
  {
     mVPoses[aKP]->Pose()->ResetStatR();
  }

  cGlobStatDet  * aGSD = 0;
  if (mVPoses[0]->NameCam() =="F120_SG1L7916_MpDcraw8B_GR.tif")
  {
     aGSD = new cGlobStatDet;
  }


  cCompFilterProj3D * aFiltre3D = 0;
  if (aImPPM.IdFilter3D().IsInit())
     aFiltre3D = mAppli.FilterOfId(aImPPM.IdFilter3D().Val());

  double aMaxDistE = mAppli.Param().MaxDistErrorPtsTerr().Val();
  double aMaxDistW = mAppli.Param().MaxDistWarnPtsTerr().Val();
  double aLimBsHP = mAppli.Param().LimBsHProj().Val();
  double aLimBsHRefut = mAppli.Param().LimBsHRefut().Val();

  if (! mVPoses[0]->Pose()->RotIsInit())
     return 0;

   double aSomPSurf = 0;
   if (mEqS)
   {
       ELISE_ASSERT(aPPMSurf!=0,"No Pond for contrainte-surface");
   }

   cPonderateur aPdrtIm(aImPPM,mNbPts);
   cPonderateur aPdrtSurf = aPdrtIm;  // Pb d'init
   if (mEqS)
   {
       aPdrtSurf = cPonderateur(*aPPMSurf,mNbPts);
   }

   double aSEr2=0;
   double aSPds2=0;
   std::vector<double> aVErs;

   int aNbPdsNN=0;
   int aNbMultPdsNN=0;
   int aNbMult=0;
   ElTimer aT0;

   InitRapOnZ(aRAZGlob);


   int aNbNN=0;
   for (int aKPm=0 ; aKPm<int(mVPMul.size()) ; aKPm++)
   {
        cOnePtsMult * aPM = mVPMul[aKPm];
        aPM->MemPds() = 0;
        cOneCombinMult * aCOM = aPM->OCM();
        const  cNupletPtsHomologues & aNupl = aPM->NPts() ;
        const std::vector<cPoseCam *> & aVP = aCOM->VP();
	std::vector<double> aVpds;

        double aPds = aNupl.Pds() * mMultPds;
        int aNbRInit= aPM->InitPdsPMul(aPds,aVpds);
        if (aNbRInit>=2)
        {
             
             aNbNN++;
             static int aCpt=0; aCpt++;
             aNbMult += (aNbRInit>=3);
             const cRapOnZ * aRAZ = aPM->OnPRaz()? aRAZGlob : 0;
             const cResiduP3Inc & aRes = aCOM->LiaisTer()->UsePointLiaison(aLimBsHP,aLimBsHRefut,0.0,aNupl,aVpds,false,aRAZ);

             if (aRes.mOKRP3I)
             {

                if (anAVA  && (anAVA->VA().TypeVerif()==eVerifResPerIm))
                {
                    // double aScN =  aVP[0]->Calib()->PIF().StdSCN();
                    anAVA->AddResidu(aNupl.PK(0),aRes.mEcIm[0]);
                }


                double aDist = euclid(aRes.mPTer-aVP[0]->CurCentre());
                if (aDist >aMaxDistW)
                {
                    static bool first = true;
                    if (first)
                    {
                       first = false;
                       std::cout <<  "WWWwwwaarning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n";
	               std::cout << " Dist = " << aDist <<   " Pose 0 " << aVP[0]->Name() << "\n";
                    }
                    if (aDist >aMaxDistE)
                    {
                        for (int aKC=0 ; aKC<int(aVP.size()) ; aKC++)
                           aVP[aKC]->Trace();
                        ELISE_ASSERT(false,"Dist >  MaxDistWarnPtsTerr")
                    }
                }

	        double aResidu = 0;
	        for (int aKPose=0 ; aKPose<int(aRes.mEcIm.size()) ; aKPose++)
                {
                   if (aVP[aKPose]->RotIsInit())
                   {
	              aResidu += square_euclid(aRes.mEcIm[aKPose]);//  *ElSquare(aScN);
                      if (isnan(aResidu))
                      {
                          std::cout <<  aRes.mEcIm[aKPose] << " " << aKPose << " " << aVP[aKPose]->Name() << "\n";
                          std::cout << "CPT= " << aCpt << "\n";
                          ELISE_ASSERT(false,"Nan residu\n");
                      }
                   }
                   else
                   {
                   }
                }
                bool isInF3D = ((!aFiltre3D) || (aFiltre3D->InFiltre(aRes.mPTer))) ;
                aResidu /= (aNbRInit-1);
             //  aResidu /= ElSquare(aNbRInit-1);

                double aPdsIm = aPdrtIm.PdsOfError(sqrt(aResidu));
             // double aPAv = aPdsIm;
                aPdsIm *= pow(aNbRInit-1,aImPPM.ExposantPoidsMult().Val());

                if (aGSD && aPdsIm)
                {
	            for (int aKPose=0 ; aKPose<int(aRes.mEcIm.size()) ; aKPose++)
                    {
                        if (aVP[aKPose]->RotIsInit())
                        {
                           aGSD->Add(aVP[aKPose],aRes.mEcIm[aKPose],aPdsIm);
                        }
                    }
                }
             
                if (isInF3D)
                {
                   if (aFpRT)
                   {
                       Pt3dr aP = aRes.mPTer;
                       fprintf(aFpRT,"*%lf %lf %lf %lf\n",aP.x,aP.y,aP.z,aPdsIm);
	               for (int aKPose=0 ; aKPose<int(aRes.mEcIm.size()) ; aKPose++)
                       {
                          if (aVP[aKPose]->RotIsInit())
                          {
                              Pt2dr anEc = aRes.mEcIm[aKPose];
                              Pt2dr aPIm = aNupl.PK(aKPose);
                              cPoseCam *  aPC = aVP[aKPose];
                              const CamStenope * aCS = aPC->CurCam();
                              Pt3dr aDir = aCS->F2toDirRayonR3(aPIm);
                              // Pt2dr aPIm =  ;
                              fprintf(aFpRT,"%s %f %f %f %f %f %f %f",aPC->Name().c_str(),aPIm.x,aPIm.y,anEc.x,anEc.y,aDir.x,aDir.y,aDir.z);
                              fprintf(aFpRT,"\n");
                          }
                       }
/*
                      if (aFpRT)
                      {
                          Pt2dr anEc = aRes.mEcIm[aKPose];
                          // Pt2dr aPIm =  ;
                          fprintf(aFpRT,"*%lf %lf ",anEc.x,anEc.y);
                          fprintf(aFpRT,"\n");
                      }
*/
                   }

                   aSEr2 += aPdsIm * aResidu;
                   aSPds2 += aPdsIm ;
  
                   if (int(aImPPM.Show().Val()) >= int(eNSM_Percentile))
                   {
                      aVErs.push_back(aResidu);
                   }
                   if (int(aImPPM.Show().Val()) >= int(eNSM_CpleIm))
                   {
                      for (int aK=0 ; aK<int(aVpds.size()) ;  aK++)
                      {
                          if (aVpds[aK] > 0)
                          {
                             aVP[aK]->AddStatR(aVpds[aK]*aPdsIm,aResidu);
                          }
                      }
                   }
/*
REPERE-111
for (int aK=0 ; aK<int(aVpds.size()) ;  aK++)
{
     Pt2dr aPIm = aNupl.PK(aK);
     std::cout << "ppppPIM " << aPIm << "\n";
}
*/
                   if (ResidualStepByStep)
                   {
                      std::cout << "| | | |  : " << aResidu << " PDS=" << aPdsIm  << " Mult " << aNbRInit << "\n";
                      std::cout << "Ter " << aRes.mPTer   << "\n";
                      std::vector<ElSeg3D> aVS;
                      for (int aK=0 ; aK<int(aVpds.size()) ;  aK++)
                      {
                          cPoseCam *  aPC = aVP[aK];
                          const CamStenope * aCS = aPC->CurCam();

                          Pt2dr aPIm = aNupl.PK(aK);

                          Pt3dr aC= aCS->VraiOpticalCenter();
                          Pt3dr aDir = aCS->F2toDirRayonR3(aPIm);
                          std::cout << aVP[aK]->Name()  << aPIm <<  aCS->F2toC2(aPIm)  << "\n";
                          std::cout << "    " <<  aC << " " <<  aDir << "\n";
                          aVS.push_back(ElSeg3D(aC,aC+aDir));
                      }
                      Pt3dr aPInt = ElSeg3D::L2InterFaisceaux(0,aVS);
                      std::cout << "Faisceaux " << aPInt   << "\n";
                      getchar();
                   }
                   if (int(aImPPM.Show().Val()) >= int(eNSM_Indiv))
                   {
                      if (aNbRInit >= aImPPM.NbMinMultShowIndiv().Val())
                      {
                           std::cout << "| | | |  : " << aResidu << " PDS=" << aPdsIm  << " Mult " << aNbRInit << "\n";
                           std::cout << "Ter " << aRes.mPTer   << "\n";
                           for (int aK=0 ; aK<int(aVpds.size()) ;  aK++)
                           {
                              if (aVpds[aK] > 0)
                                 std::cout << aVP[aK]->CF()->CameraCourante()->R3toF2(aRes.mPTer) 
                                      <<  aRes.mEcIm[aK] 
                                      << aNupl.PK(aK) << "\n";
                              else
                                 std::cout << "       =   =   =   =\n";
                           }
                      }
                   }
                }



                if (anArgPT&&(aPdsIm >0) && (aRes.mBSurH > anArgPT->LimBsH()))
                {
                   double aPdsBsH  = ElMin(1.0,aRes.mBSurH / 0.1);
                   Pt2dr aP = aNupl.PK(mKIm);
                   aP = aVP[mKIm]->OrIntM2C()(aP);
                   anArgPT->AddAGP(aP,aRes.mPTer,aPdsIm*aPdsBsH,false,&aVpds,&(aCOM->VP()));
                }

                if (aPdsIm >0) 
                {
                   aPM->MemPds() = aPdsIm;
                }

                if (   ((aPdsIm >0) &&  aImPPM.Add2Compens().Val()) && isInF3D)
                {
                    aNbPdsNN++;   
                    aNbMultPdsNN += (aNbRInit>=3);
	            for (int aKPose=0 ; aKPose<int(aVpds.size()) ; aKPose++)
                    {
                        // aVpds[aKPose] *= aPdsIm;
                        if (aVpds[aKPose])
                        {
                            aVP[aKPose]->AddPMoy(aRes.mPTer,aRes.mBSurH);
                        }
                    }

                    double aPdsSurf = 0;
                    if (mEqS)
                    {
                       aPdsSurf = aPdrtSurf.PdsOfError(ElAbs(aRes.mEcSurf)) * aPds;
                    }

                     aCOM->LiaisTer()->SetTerrainInit(true);
                     aCOM->LiaisTer()->SetMulPdsGlob(aPdsIm);
	             const cResiduP3Inc & aRes2 = aCOM->LiaisTer()->UsePointLiaison(aLimBsHP,aLimBsHRefut,aPdsSurf,aNupl,aVpds,aSO.AddEq(),aRAZ);
                     aCOM->LiaisTer()->SetMulPdsGlob(1.0);
                     aCOM->LiaisTer()->SetTerrainInit(false);  // Conservatif




                    if (aRes2.mOKRP3I)
                    {
                       aSO.AddSEP(aRes2.mSomPondEr);
                       aPM->MemPds() = aPdsIm;
                    }
                    // aNb++;
                }
                if (anAVA &&  (anAVA->VA().TypeVerif()==eVerifDZ) && (aPdsIm >0) && (aNbRInit>=3))
                {
                //  Dz regarde ce qui se passe quand on supprime les points de l'image elle meme
                // dans la compensation. Sans doute pour tester les pb de biais
                    Pt3dr aPTerGlob =  aRes.mPTer;
                    std::vector<double>  aDupV  = aVpds;
                    aDupV[0] = 0;
                    aCOM->LiaisTer()->SetMulPdsGlob(aPdsIm);
                    const cResiduP3Inc & aRes2 = aCOM->LiaisTer()->UsePointLiaison(aLimBsHP,aLimBsHRefut,0.0,aNupl,aDupV,false,aRAZ);
                    aCOM->LiaisTer()->SetMulPdsGlob(1.0);
                    if (aRes2.mOKRP3I)
                    {
                       Pt3dr aPTer2 =  aRes2.mPTer;
                       double aDZ = aPTer2.z - aPTerGlob.z;
                       anAVA->AddPImDZ(aNupl.PK(mKIm),aDZ,aVpds,*aPM);
                    }
                }
             }
        }
        BugUPL = false;
   }

   aSEr2 /= aSPds2;
   int aNbP = mVPMul.size();
      
  mVPoses[0]->Pose()->SetNbPtsMulNN(aNbMultPdsNN+mVPoses[0]->Pose()->NbPtsMulNN());

   if (aSO.AddEq())
   {
       if (int(aImPPM.Show().Val()) >= int(eNSM_Paquet))
       {
          //ostream & myfile = std::cout;
      
          // myfile = std::cout;

          if (mEqS)
             mAppli.COUT() << "PDS Surf = " << aSomPSurf << "\n";
          mAppli.COUT() << "RES:["  << mVPoses[0]->NameCam() << "]"
                <<  " ER2 " << sqrt(aSEr2)
                << " Nn " << (100.0*aNbPdsNN)/double(aNbP) 
                << " Of " << aNbP
                << " Mul " << aNbMult
                << " Mul-NN " << aNbMultPdsNN
                <<  " Time " << aT0.uval()
                << "\n";
       }
       cElRegex_Ptr aFilterImAff  = mAppli.Param().Im2Aff().ValWithDef(0);  
       bool aMatchIm0 =   (aFilterImAff==0)
                       || (aFilterImAff->Match( mVPoses[0]->Pose()->Name()));

       if ((int(aImPPM.Show().Val()) >= int(eNSM_Percentile)) &&  aMatchIm0)
       {
           int aNBV = aVErs.size();
           if (aNBV>=2)
           {
               std::cout << "----- % % % % % % % % % % -----------\n";
               std::sort(aVErs.begin(),aVErs.end());
               std::vector<double> aVPerc;

               if (aImPPM.ShowPercentile().IsInit())
                  aVPerc = aImPPM.ShowPercentile().Val();
               else
               {
                   aVPerc.push_back(50);
                   aVPerc.push_back(75);
                   aVPerc.push_back(90);
               }
               for (int aK=0 ; aK<int(aVPerc.size()); aK++)
               {
                   std::cout << "Perc[" << aVPerc[aK] << "%]=" 
                         <<  ValPercentile(aVErs,aVPerc[aK])
                         << "\n";
               }
           }
           else
           {
              std::cout << "Pas assez de valeurs pour percentile\n";
           }
       }
       if ((int(aImPPM.Show().Val()) >= int(eNSM_CpleIm)) && aMatchIm0)
       {
          for (int aKP=0 ;  aKP<int(mVPoses.size()) ; aKP++)
          {
               bool aMatchImK =   aMatchIm0
                               || (aFilterImAff->Match( mVPoses[aKP]->Pose()->Name()));
               double aSom1,aSomP,aSomPR;
               if ( aMatchImK)
               {
                   mVPoses[aKP]->Pose()->GetStatR(aSomP,aSomPR,aSom1);
                   std::cout << "   " << mVPoses[aKP]->Pose()->Name();
                   if ((aSomP==0)|| (aNbNN==0))
                   {
                       std::cout << " XXX";
                   }
                   else
                   {
                       std::cout <<  " ER=" << sqrt(aSomPR/aSomP) << " " <<  " %Pts=" <<( aSom1*100)/aNbNN ;
                   }
                   std::cout << "\n";
               }
          }
       }
       if (aImPPM.GetChar().Val())
       {
          getchar();
       }

       {
         if (aGSD) aGSD->Show();
         delete aGSD;
       }
   }

   return aSEr2;
}










//   ==============================================================
//   ==============================================================
//   
//
//     CRITERE de SURFACE
//
//   ==============================================================
//   ==============================================================

void cObsLiaisonMultiple::ClearAggregImage()
{
    for (int aKP=0 ; aKP<int(mVPoses.size()) ; aKP++)
    {
         mVPoses[aKP]->Pose()->NbPLiaisCur() = 0;
         mVPoses[aKP]->Pose()->QualAZL() = 0;
         mVPoses[aKP]->Pose()->AZL().Reset();
    }
}


// Qualite des que 3eme vue, si au moins aNbPtsMin multiple,
// tjrs prio
double cObsLiaisonMultiple::QualityZonePMul
       (
            bool  UseZU,
            bool  OnInit,
            int  aNbPtsMin,
            double aExpDist,
            double aExpNb,
            bool &  GotPMul
       )
{
   int aNbPtsMult = 0;
   // int aNbPtsTot = 0;
   cAnalyseZoneLiaison aAZL;
   for (int aKPt=0 ; aKPt<int(mVPMul.size()) ; aKPt++)
   {
        cOnePtsMult * aPM = mVPMul[aKPt];
        int aNpPose = aPM->NbPoseOK(OnInit,UseZU);
        if (aNpPose>=2)
        {
            aAZL.AddPt(aPM->P0());
            aNbPtsMult++;
        }
   }

   GotPMul = (aNbPtsMult>=aNbPtsMin);
   return  (! GotPMul) ? 
            QualityZoneAlgoCV (UseZU,OnInit,0,aExpDist,aExpNb,2) / 1e5 :
            1e3+  aAZL.Score(aExpDist,aExpNb);
}


double cObsLiaisonMultiple::QualityZoneAlgoCV
       (
              bool  OnZU,
              bool  OnInit,
              int    aNbMinPts,
              double aExpDist,
              double aExpNb,
              int    aNbPoseMin
       )
{
   std::vector<double>  aVCost; 
   std::vector<cPoseCam *>   aVPC = BestPoseInitStd(OnZU,OnInit,aVCost,aNbMinPts,aExpDist,aExpNb);
   if (int(aVPC.size()) <aNbPoseMin )
      return -1;
   return aVCost[0];
}


int cObsLiaisonMultiple::NbRotPreInit() const
{
    int aRes = 0;
    for (int aKP=0 ; aKP<int(mVPoses.size()) ; aKP++)
    {
         if (mVPoses[aKP]->Pose()->PreInit())
            aRes ++;
    }
    return aRes;
}



double  cObsLiaisonMultiple::StdQualityZone
        (
              bool   OnZU,
              bool   OnInit,
              int    aNbMinPts,
              double aExpDist,
              double aExpNb,
              bool &  GotPMul
        )
{
    GotPMul = false;
    // return  mAppli.NbRotPreInit() >= 2                   ?
    return  NbRotPreInit() >= 2                                              ?
            QualityZonePMul(OnZU,OnInit,aNbMinPts,aExpDist,aExpNb,GotPMul)   :
            QualityZoneAlgoCV(OnZU,OnInit,aNbMinPts,aExpDist,aExpNb,1) ;
}

std::vector<cPoseCam *>  cObsLiaisonMultiple::BestPoseInitStd
                         (
                               bool OnZU,
                               bool OnInit,
                               std::vector<double> & aVCost,
                               int    aNbMinMul,
                               double aExpDist,
                               double aExpNb
                         )
{
    int aNbMinGlob = 10;
    ClearAggregImage();
    std::vector<cPoseCam *> aRes;

    // Accum de Qual + Mult
    for (int aKPt=0 ; aKPt<int(mVPMul.size()) ; aKPt++)
    {
        cOnePtsMult * aPM = mVPMul[aKPt];
        const std::vector<cPoseCam *> & aVPose=  aPM->OCM()->VP();
        for (int aKPose=0 ; aKPose<int(aVPose.size()) ; aKPose++)
        {
            cPoseCam * aPose = aVPose[aKPose];
            // if (OnInit ? aPose->RotIsInit() : aPose->PreInit())
            if (
                    (aPose->CanBeUSedForInit(OnInit))
                 && (      (!OnZU)
                        || (        aVPose[0]->IsInZoneU(aPM->P0())
                               &&   aVPose[aKPose]->IsInZoneU(aPM->PK(aKPose))
                           )
                    )
               )
            {
               aPose->AZL().AddPt(aPM->P0());
               aPose->NbPLiaisCur()++;
            }
        }
    }

    cPoseCam * aBestP=0;
    cPoseCam * aBestSec=0;
   
    // Calcul de Qual  et select best
    for (int aKP=0 ; aKP<int(mVPoses.size()) ; aKP++)
    {
         cPoseCam * aPose = mVPoses[aKP]->Pose();


         if (aPose->CanBeUSedForInit(OnInit) && (aPose->NbPLiaisCur()>=aNbMinMul))
         {
             aPose->QualAZL() = aPose->AZL().Score(aExpDist,aExpNb);
             if ((aBestP==0) || (aPose->QualAZL() >aBestP->QualAZL()))
             {
                  aBestSec = aBestP;
                  aBestP = aPose;
             }
             else if ((aBestSec==0) || (aPose->QualAZL() >aBestSec->QualAZL()))
             {
                   aBestSec = aPose;
             }
         }
    }


    aVCost.clear();
    if ((aBestP!=0)  && (int(aBestP->AZL().VPts().size())>=aNbMinGlob))
    {
       aRes.push_back(aBestP);
       aVCost.push_back(aBestP->QualAZL());

       if (aBestSec!=0) 
       {
          aRes.push_back(aBestSec);
          aVCost.push_back(aBestSec->QualAZL());
       }
    }




    // Calcul de Qual  et select best

    ClearAggregImage();

    return aRes;
}




//   ==============================================================
//   ==============================================================
//   
//
//     Orientation initiales par Point appuis multiples
//
//   ==============================================================
//   ==============================================================


class cAppar1Im
{
    public :
        cAppar1Im(Pt2dr aIm,ElSeg3D aSeg) :
             mIm  (aIm),
             mSeg (aSeg)
        {
        }
        Pt2dr    mIm;
        ElSeg3D  mSeg;
};



class  cIndAppui
{
    public :
       int mI1;
       int mI2;
       int mI3;

       cIndAppui(int aI1,int aI2,int aI3) 
       {
             mI1 = ElMin3(aI1,aI2,aI3);
             mI3 = ElMax3(aI1,aI2,aI3);
             mI2 = aI1+aI2+aI3 - mI1-mI3;
       }
       bool AllDif() const {return (mI1<mI2) && (mI2<mI3);}

       bool operator < (const cIndAppui & anP) const
       {
             if (mI1<anP.mI1) return true;
             if (mI1>anP.mI1) return false;

             if (mI2<anP.mI2) return true;
             if (mI2>anP.mI2) return false;

             if (mI3<anP.mI3) return true;
             if (mI3>anP.mI3) return false;

             return false;
       }
};



bool Valid3Index(const std::vector<Appar23> &  aVS2,const cIndAppui & aI)
{
   const Appar23 & Ap1 = aVS2[aI.mI1];
   const Appar23 & Ap2 = aVS2[aI.mI2];
   const Appar23 & Ap3 = aVS2[aI.mI3];

    double aEpsIm = 1e-7;
    double aEpsTer = 1e-7;

   return 
                (euclid(Ap1.pim-Ap2.pim) > aEpsIm)
            &&  (euclid(Ap1.pim-Ap3.pim) > aEpsIm)
            &&  (euclid(Ap2.pim-Ap3.pim) > aEpsIm)
            &&  (euclid(Ap1.pter-Ap2.pter) > aEpsTer)
            &&  (euclid(Ap1.pter-Ap3.pter) > aEpsTer)
            &&  (euclid(Ap2.pter-Ap3.pter) > aEpsTer);
}


void  MakeVectIndAppui
      (
            const std::vector<Appar23>  & aVS2,
            std::set<cIndAppui> & aRes,
            int aNbInd,
            int aNbResult
      )
{
    if (aNbInd < 3) 
        return;
     aRes.clear();
     bool aAllIndex = false;
     if (aNbInd < 300)  // Pour etre en dehors des debordement
     {
        int aNbMaxResult = (aNbInd*(aNbInd-1)*(aNbInd-2))/6;
        // Si la proba d'un bon tirage est trop faible
        if (aNbMaxResult < round_up( 1.5 * aNbResult))
            aAllIndex = true;
     }
     aRes.clear();

     if (aAllIndex)
     {
        for (int aK0=0 ; aK0<aNbInd ; aK0++)
        {
            for (int aK1=aK0+1 ; aK1<aNbInd ; aK1++)
            {
                for (int aK2=aK1+1 ; aK2<aNbInd ; aK2++)
                {
                     cIndAppui aIA(aK0,aK1,aK2);
                     if (Valid3Index(aVS2,aIA))
                         aRes.insert(aIA);
                }
            }
        }
     }
     else
     {
         while (int(aRes.size())<aNbResult)
         {
             cIndAppui aIA (NRrandom3(aNbInd),NRrandom3(aNbInd),NRrandom3(aNbInd));
             if (aIA.AllDif())
             {
                if (Valid3Index(aVS2,aIA))
                    aRes.insert(aIA);
             }
         }
     }
}


static inline Pt2dr  ToLoc(const ElRotation3D & aR,const Pt3dr & aPter)
{
     return ProjStenope(aR.ImRecAff(aPter));
}

//  Cette fonction de conversion de distance permet de limiter l'influence de 
// point a l'infini.

double DistConversion(double aDist)
{
   return aDist /(aDist+0.2);
}

double cObsLiaisonMultiple::CostOrientationPMult(const ElRotation3D & aR,const Appar23 & anAp) const
{
    return DistConversion(euclid(anAp.pim,ToLoc(aR,anAp.pter)));
}

double cObsLiaisonMultiple::CostOrientationPMult(const ElRotation3D & aR,const cAppar1Im & anAp) const
{
    SegComp  aSeg(ToLoc(aR,anAp.mSeg.P0()),ToLoc(aR,anAp.mSeg.P1()));


    return DistConversion(ElAbs(aSeg.ordonnee(anAp.mIm)));
}



double cObsLiaisonMultiple::CostOrientationPMult
       (
                      const ElRotation3D & aR,
                      const std::vector<Appar23> & aV2,
                      const std::vector<cAppar1Im> & aV1
       ) const
{
    double aRes2=0;
    for (int aK2=0 ; aK2<int(aV2.size()) ; aK2++)
    {
        double aC = CostOrientationPMult(aR,aV2[aK2]);
        aRes2 += aC;
    }

    double aRes1=0;
    for (int aK1=0 ; aK1<int(aV1.size()) ; aK1++)
    {
        double aC = CostOrientationPMult(aR,aV1[aK1]);
        aRes1 += aC;
    }

   return (aRes1+aRes2) / (aV2.size()+aV1.size()) ;
}




void cObsLiaisonMultiple::TestMEPAppuis
     (
        bool           UseZU,
        ElRotation3D & aRSol,
        int aNbRansac,
        const cLiaisonsInit & aLI
     )

{

   CompilePose();

   cPoseCam * aCam0 = mVPoses[0]->Pose();

   // DebugPM = (aCam0->Name() == "F011_IMG_0435_Gray.tif");
   // CompilePose();

//
//   1- On calcul combien de point multiple et simple devront etre selectionnes:
//

   int aNb1 =0;
   int aNb2 =0;
   for (int aKPt=0 ; aKPt<int(mVPMul.size()) ; aKPt++)
   {
        int aNpPose = mVPMul[aKPt]->NbPoseOK(true,UseZU);
        aNb1 += (aNpPose ==1);
        aNb2 += (aNpPose >=2);
   }
   int aNbTot = aNb1+aNb2;

   // Aucun espoir d'aucune sorte ....
   if (aNbTot < 2) 
      return;

   // if (aNbTot<aLI.NbMinPtsRanAp().Val())
   if ((aNb2 <3)||(aNbTot<aLI.NbMinPtsRanAp().Val()))
       return;
   int aNbMax =  ElMin(aLI.NbMaxPtsRanAp().Val(),aNbTot);

   // La proportion de point multiple est le max de la proportion reell
   // et de celle souhaitee
   double aProp2 = ElMax(aLI.PropMinPtsMult().Val(),aNb2/double(aNbTot));
   // Cepepdant on ne peut selectionner les points qu'une fois
   aProp2 = ElMin3(1.0,aProp2,aNb2/double(aNbMax));
   
   int aNSel2 = round_ni(aNbMax*aProp2);
   int aNSel1 = ElMin(aNb1,aNbMax-aNSel2);

   cRandNParmiQ  aSel1(aNSel1,aNb1);
   cRandNParmiQ  aSel2(aNSel2,aNb2);

   std::vector<Appar23>    aVS2;
   std::vector<cAppar1Im> aVS1;

   CamStenope & aCS0 =   * (aCam0->Calib()->PIF().CurPIF());

   for (int aKPt=0 ; aKPt<int(mVPMul.size()) ; aKPt++)
   {
        cOnePtsMult * aPM = mVPMul[aKPt];
        int aNpPose = aPM->NbPoseOK(true,UseZU);
        if ((aNpPose==1) && (aSel1.GetNext()))
        {
            Pt2dr aPIm = aCS0.F2toPtDirRayonL3(aPM->P0());
            aVS1.push_back(cAppar1Im(aPIm,aPM->GetUniqueDroiteInit(UseZU)));
        }
        else if ((aNpPose>=2) && (aSel2.GetNext()))
        {
             std::vector<double>  aVPds;
             double aPds=1;
             const cResiduP3Inc * aRes =  aPM->ComputeInter(aPds,aVPds); // UseZU est pris en compte
             if (aRes)
             {
                Pt2dr aPIm = aCS0.F2toPtDirRayonL3(aPM->P0());
                aVS2.push_back(Appar23(aPIm,aRes->mPTer));
             }
        }

   }

    ElTimer aT0;
    double aCostMin = CostOrientationPMult(aRSol,aVS2,aVS1);

    tParamAFocal aNoPAF;
    CamStenopeIdeale aCSI(true,1.0,Pt2dr(0.0,0.0),aNoPAF);
    // double aFoc =  aCSI.Focale() ;
   
    // std::cout << "Cost " << aCostMin * aFoc << " Time " << aT0.uval() << "\n";

    std::set<cIndAppui>  aSetIA;
    MakeVectIndAppui(aVS2,aSetIA,aVS2.size(),aNbRansac);
        
    bool RChanged = false; GccUse(RChanged);
    for (std::set<cIndAppui>::const_iterator itI=aSetIA.begin();  itI!=aSetIA.end(); itI++)
    {
        std::list<ElRotation3D> aLR;

        Pt3dr aPT1 = aVS2[itI->mI1].pter;
        Pt3dr aPT2 = aVS2[itI->mI2].pter;
        Pt3dr aPT3 = aVS2[itI->mI3].pter;

        Pt2dr aPI1 = aVS2[itI->mI1].pim;
        Pt2dr aPI2 = aVS2[itI->mI2].pim;
        Pt2dr aPI3 = aVS2[itI->mI3].pim;


        aCSI.OrientFromPtsAppui(aLR,aPT1,aPT2,aPT3,aPI1,aPI2,aPI3);
        for 
        (
                 std::list<ElRotation3D>::const_iterator itR = aLR.begin();
                 itR != aLR.end() ;
                 itR++
        )
        {
             // Solution "Physique" la camera voit devant
             if (
                           (itR->ImAff(aPT1).z >0)
                        && (itR->ImAff(aPT2).z >0)
                        && (itR->ImAff(aPT3).z >0)
             )
             {

                double aC = CostOrientationPMult(itR->inv(),aVS2,aVS1);
                if (aC < aCostMin)
                {
                    aCostMin = aC;
                    aRSol = itR->inv();
                    RChanged = true;
                    //std::cout  <<   aC*aFoc  << "\n";
                }
             }
         }

    }
 
// std::cout << "RCHNAGED FOR " <<  aCam0->Name() << " " << RChanged << " COST" << aCostMin << "\n";
// getchar();

// std::cout << "TestMEPAppuis " << aCam0->Name() << " CHANG " << RChanged << "\n";
}


void cObsLiaisonMultiple::TestMEPCentreInit
     (
        ElRotation3D & aR,
        const Pt3dr & aP,
        const cLiaisonsInit & aLI
     )

{

   CompilePose();

}


void cObsLiaisonMultiple::InitRapOnZ(const cRapOnZ *  aRAZGlob)
{
   if (aRAZGlob == 0) return;

   if (mRazGlob)
   {
       ELISE_ASSERT
       (
           mRazGlob->LayerIm()==aRAZGlob->LayerIm(),
           "Attempt to change to layer im"
       );
       if (aRAZGlob==mRazGlob) return;
   }
   // A CORRIGER : 
   //  delete mRazGlob;  -> ON peut pas le detruire comme ca car il est partage par tous ...
   mRazGlob = aRAZGlob;

   const std::string & aNameLayerIm =  mRazGlob->LayerIm();
   if (aNameLayerIm=="")
      return;
   if (mLayerImDone)
      return;
   mLayerImDone = true;
   cLayerImage * aLayIm = mAppli.LayersOfName(aNameLayerIm);
   for (int aKP=0 ; aKP<int(mVPoses.size()) ; aKP++)
   {
       mVPoses[aKP]->Pose()->SetCurLayer(aLayIm);
   }
   //     std::vector<cOneElemLiaisonMultiple *>     mVPoses;

   // SI les 

   for (int aKPm=0 ; aKPm<int(mVPMul.size()) ; aKPm++)
   {
        cOnePtsMult * aPM = mVPMul[aKPm];
        cOneCombinMult * aCOM = aPM->OCM();

        bool isOnPlane = false;
        const  cNupletPtsHomologues & aNupl = aPM->NPts() ;
        const std::vector<cPoseCam *> &  aVP = aCOM->VP();
        for (int aKPose=0 ; aKPose<int(aVP.size()) ; aKPose++)
        {
             Pt2dr aP = aNupl.PK(aKPose);
             int aVal = aVP[aKPose]->GetCurLayer()->LayerOfPt(aP);
             if (aVal)
                isOnPlane = true;
        }
        aPM->SetOnPRaz(isOnPlane);
   }
}


double cObsLiaisonMultiple::BasicAddObsLM 
       ( 
           const cPonderationPackMesure & aPPM,
           cStatObs &           aSO,
           const cRapOnZ *      aRAZGlob
       )
{
   if (! aPPM.Add2Compens().Val()) return 0;

   double aLimBsHP = mAppli.Param().LimBsHProj().Val();
   double aLimBsHRefut = mAppli.Param().LimBsHRefut().Val();

   if (! mVPoses[0]->Pose()->RotIsInit())
      return 0;

   cPonderateur aPdrtIm(aPPM,mNbPts);

   //int aNb=0;

   InitRapOnZ(aRAZGlob);

   for (int aKPm=0 ; aKPm<int(mVPMul.size()) ; aKPm++)
   {
        cOnePtsMult * aPM = mVPMul[aKPm];
        const cRapOnZ * aRAZ = aPM->OnPRaz()? aRAZGlob : 0;
        cOneCombinMult * aCOM = aPM->OCM();
        const  cNupletPtsHomologues & aNupl = aPM->NPts() ;
        // const std::vector<cPoseCam *> & aVP = aCOM->VP();
	std::vector<double> aVpds;

        double aPds = aNupl.Pds() * mMultPds;
        double aPMem = aPM->MemPds();
        if (aPMem>0)
        {
             aPM->InitPdsPMul(aPds*aPMem,aVpds);

	     const cResiduP3Inc & aRes2 =  aCOM->LiaisTer()->UsePointLiaison(aLimBsHP,aLimBsHRefut,0,aNupl,aVpds,aSO.AddEq(),aRAZ);
             if (aRes2.mOKRP3I)
             {
                aSO.AddSEP(aRes2.mSomPondEr);
             }
        }
   }

   return 0;
}


void AddCheck(int & aNbPos,double & aSAng, double aZ, double aDist)
{
   if (aZ<0)
   {
       aSAng += atan(aDist/(-aZ));
   }
   else
     aNbPos++;
}

void  cObsLiaisonMultiple::CheckInit()
{
   int aNb = 0 ;
   int aNbPos = 0 ;
   double SomAngNeg = 0;

   for (int aKPm=0 ; aKPm<int(mVPMul.size()) ; aKPm++)
   {
        cOnePtsMult * aPM = mVPMul[aKPm];
        cOneCombinMult * aCOM = aPM->OCM();
        const  cNupletPtsHomologues & aNupl = aPM->NPts() ;
        // const std::vector<cPoseCam *> & aVP = aCOM->VP();

        const std::vector<cPoseCam *> & aVP =  aCOM->VP();

	std::vector<double> aVpds;
        aPM->InitPdsPMul(1,aVpds);

        for (int aK=1 ; aK< int(aVpds.size()) ; aK++)
        {
           if (aVpds[aK] > 0)
           {
               // Pt2dr aP0 = aNupl.PK(0);
               // Pt2dr aPK = aNupl.PK(aK);
               const ElCamera * aCam0 = aVP[0]->CurCam();
               const ElCamera * aCamK = aVP[aK]->CurCam();
               Pt3dr aPTer = aCam0->PseudoInter(aNupl.PK(0),*aCamK,aNupl.PK(aK));

               Pt3dr aPT0 = aCam0->R3toL3(aPTer);
               Pt3dr aPTK = aCamK->R3toL3(aPTer);

               double aDist = euclid(aCam0->OrigineProf()-aCamK->OrigineProf());
               AddCheck(aNbPos,SomAngNeg,aPT0.z,aDist);
               AddCheck(aNbPos,SomAngNeg,aPTK.z,aDist);
                aNb+=2;
           }
        }

   }

   SomAngNeg  /= aNb;

    //  std::cout << "====== NB " << aNb  << " Pos " << aNbPos << " Moy Ang Neg " << SomAngNeg  << "\n"; 

    if (SomAngNeg >mAppli.Param().ThresholdWarnPointsBehind().Val())
    {

        std::string aMes =   " Cam=" + mPose1->Name()
                           + " \%Behind=" + ToString((100.0*(aNb-aNbPos) ) / aNb)
                           + " AngleBehind="  + ToString(SomAngNeg);
        cElWarning::BehindCam.AddWarn(aMes,__LINE__,__FILE__);
    }
}


std::map<std::string,cObsLiaisonMultiple *> & cPackObsLiaison::DicoMul()
{
   return mDicoMul;
}

};



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
