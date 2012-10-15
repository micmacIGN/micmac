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

namespace NS_ParamApero
{

/**************************************************/
/*                                                */
/*    cResul_RL, cCpleRRL , cAgglomRRL            */
/*                                                */
/**************************************************/

class cResul_RL
{
    public :

        cResul_RL(Pt2di aSz, double aSc) ;

	void Add(Pt2dr aP,double aVal,double aPds);
	void MakeFile(const std::string &,double aDyn,bool IsSigned);

     private :

        Pt2di      mSzInit;
        Pt2di      mSz;
	double     mSc;

	Im2D_REAL4           mResidu;
	TIm2D<REAL4,REAL8>   mTR;
	Im2D_REAL4           mPds;
	TIm2D<REAL4,REAL8>   mTP;
};
   
       //           * - * - * - * - * - * - * - * - *
      
class cAgglomRRL
{
     public :
	 cAgglomRRL(const cAppliApero &,const cExportImResiduLiaison &);

         void Add(cResul_RL &,cPoseCam * aPC1,cPoseCam * aPC2);
	 const cExportImResiduLiaison  & EIRL() const;


     private :


	 const cAppliApero &              mAppli;
         std::map<std::string,cResul_RL>  mResiduOfCalib;
	 const cExportImResiduLiaison  &  mEIRL;
	 cElRegex                         mAutomIndivIm;
};



       //           * - * - * - * - * - * - * - * - *
       //           * - * - * - * - * - * - * - * - *
       //           * - * - * - * - * - * - * - * - *
       

         //  ---   cResul_RL  ---- 

cResul_RL::cResul_RL(Pt2di aSz, double aSc) :
    mSzInit   (aSz),
    mSz       (round_up(Pt2dr(aSz)/aSc) + Pt2di(1,1)),
    mSc       (aSc),
    mResidu   (mSz.x,mSz.y,0.0),
    mTR       (mResidu),
    mPds      (mSz.x,mSz.y,0.0),
    mTP       (mPds)
{
}

void cResul_RL::Add(Pt2dr aP,double aVal,double aPds)
{
   mTR.incr(aP/mSc,aVal*aPds);
   mTP.incr(aP/mSc,aPds);
}


void cResul_RL::MakeFile(const std::string & aName,double aDyn,bool IsSigned) 
{
    Fonc_Num aFonc = aDyn * mResidu.in()/Max(mPds.in(),1e-5);

    if (IsSigned)
        aFonc = 128 + aFonc;

    Tiff_Im::Create8BFromFonc(aName,mSz,Max(0,Min(255,aFonc)));
}

         //  ---   cAgglomRRL  ---- 

cAgglomRRL::cAgglomRRL(const cAppliApero &  anAppli,const cExportImResiduLiaison & anEIRL) :
  mAppli (anAppli),
  mEIRL  (anEIRL),
  mAutomIndivIm (anEIRL.ResidusIndiv().IsInit()?anEIRL.ResidusIndiv().Val().Pattern() : ".*",10)
{
}

const cExportImResiduLiaison  & cAgglomRRL::EIRL() const
{
   return mEIRL;
}

void cAgglomRRL::Add
     (
         cResul_RL & aRL,
	 cPoseCam * aPC1,
	 cPoseCam * aPC2
     )
{
   if (mEIRL.ResidusIndiv().IsInit())
   {
       std::string aN12 = aPC1->Name() + "@" + aPC2->Name();
       if (mAutomIndivIm.Match(aN12))
       {
            std::string aNFile = MatchAndReplace
				(
				    mAutomIndivIm,
				    aN12,
				    mEIRL.ResidusIndiv().Val().Name()
				);
            aRL.MakeFile
	    (
	        mAppli.Param().DirectoryChantier().Val()+aNFile,
		mEIRL.DynIm(),
		mEIRL.Signed().Val()
            );

       }
   }
}



/**************************************************/
/*                                                */
/*              cObservLiaison_1Cple              */
/*                                                */
/**************************************************/

cObservLiaison_1Cple::cObservLiaison_1Cple
(
      const cBDD_PtsLiaisons & aBPL,
      const std::string& aNamePack,
      const std::string& aNameIm1,
      const std::string& aNameIm2
)  :
   // mPack      (ElPackHomologue::-FromFile(aNamePack)),
   mPack      (),
   mIm1       (aNameIm1),
   mIm2       (aNameIm2),
   mPose1     (0),
   mPose2     (0),
   // mCpleR1    (0),
   // mCpleR2    (0),
   mPLiaisTer (0),
   mSurf      (0),
   mEqS       (0),
   mEcMax     (0)
{
  // Partie du code obsolete , equation de co-planarite non maintenue
  ELISE_ASSERT(false,"ElPackHomologue:: - FromFile No Correction");
  std::cout << "aNamePack " << aNamePack << "\n";
  if (aBPL.SzImForInvY().IsInit())
  {
     mPack.InvY
     (
           aBPL.SzImForInvY().Val().SzIm1(),
           aBPL.SzImForInvY().Val().SzIm2()
     );
  }
}

cPoseCam * cObservLiaison_1Cple::Pose1() const
{
    return mPose1;
}

cPoseCam * cObservLiaison_1Cple::Pose2() const
{
    return mPose2;
}


int cObservLiaison_1Cple::NbH() const
{
   return mPack.size();
}

const std::string& cObservLiaison_1Cple::NameIm1()
{
   return mIm1;
}

const std::string& cObservLiaison_1Cple::NameIm2()
{
   return mIm2;
}


const ElPackHomologue & cObservLiaison_1Cple::Pack() const
{
   return mPack;
}

void  cObservLiaison_1Cple::Compile
      (
         cSurfParam *   aSurf,
         cAppliApero & anAppli
      )
{
    mSomPds =0.0;
    mNbPts = 0.0;

    mSurf = aSurf;
    if (aSurf!=0)
       mEqS = aSurf->EqSurfInc();

    mPose1 = anAppli.PoseFromName(mIm1);
    mPose2 = anAppli.PoseFromName(mIm2);
    // mCpleR1 = anAppli.SetEq().NewCpleCam(*(mPose1->CF()),*(mPose2->CF()),cNameSpaceEqF::eResiduIm1);
    // mCpleR2 = anAppli.SetEq().NewCpleCam(*(mPose1->CF()),*(mPose2->CF()),cNameSpaceEqF::eResiduIm2);

    std::vector<cCameraFormelle *>  aVCF;
    aVCF.push_back(mPose1->CF());
    aVCF.push_back(mPose2->CF());

    mPLiaisTer = new cManipPt3TerInc(aVCF[0]->Set(),mEqS,aVCF);

   for 
   (
       ElPackHomologue::tCstIter itL=mPack.begin();
       itL!=mPack.end();
       itL++
   )
   {
      mNbPts++;
      mSomPds += itL->Pds();
   }
   mMultPds = mNbPts / mSomPds ;
}


double MoyArithm(double aV1,double aV2)
{
  if ((aV1>0) && (aV2>0))
     return sqrt(aV1*aV2);

  if (aV1>0) return aV1;
  if (aV2>0) return aV2;

  ELISE_ASSERT(false,"MoyArithm");
  return -1;
}

double cObservLiaison_1Cple::EcMax() const
{
   return mEcMax;
}

double  cObservLiaison_1Cple::AddObs
        (
	   const cPonderationPackMesure & aPPM,
	   const cPonderationPackMesure * aPPMSurf
	)
{
   if (mEqS)
   {
       ELISE_ASSERT(aPPMSurf!=0,"No Pond for contrainte-surface");
   }

   cPonderateur aPdrtIm(aPPM,mPack.size());


   cPonderateur aPdrtSurf = aPdrtIm;  // Pb d'init
   if (mEqS)
   {
       aPdrtSurf = cPonderateur(*aPPMSurf,mPack.size());
   }

   double aS1=0;
   double aSEr2=0;
   double aSomPdsSurf = 0;
   mEcMax = 0.0;

   for 
   (
       ElPackHomologue::tCstIter itL=mPack.begin();
       itL!=mPack.end();
       itL++
   )
   {
      if (true)
      {
	  double aNb = itL->Pds() * mMultPds;
	  std::vector<double> aVPds;
	  aVPds.push_back(1.0);
	  aVPds.push_back(1.0);


          //const std::vector<Pt2dr> & aPTers = mPLiaisTer->ResiduPointLiaison(*itL,&aPInter);
	  const cResiduP3Inc & aRes = mPLiaisTer->UsePointLiaison(-1,-1,0.0,*itL,aVPds,false);
          double aResidu = (square_euclid(aRes.mEcIm[0])+square_euclid(aRes.mEcIm[1]));

	  ElSetMax(mEcMax,sqrt(aResidu));

	  double aPdsIm = aPdrtIm.PdsOfError(sqrt(aResidu));
          aVPds[0]= (aPdsIm*aNb);
          aVPds[1]= (aPdsIm*aNb);

	  double aPdsSurf = 0;
	  if (mEqS)
	  {
             aPdsSurf = aPdrtSurf.PdsOfError(ElAbs(aRes.mEcSurf)) *aNb;
	  }
	  aSomPdsSurf += aPdsSurf;
          mPLiaisTer->UsePointLiaison(-1,-1,aPdsSurf,*itL,aVPds,true);
	  aSEr2 += aResidu * aNb;
          aS1 += aNb;


          if (int(aPPM.Show().Val()) >= int(eNSM_Indiv))
            std::cout << "RLiais = " << sqrt(aResidu) << " pour P1 " << itL->P1() << "\n";

	    mPose1->AddPMoy(aRes.mPTer,aRes.mBSurH);
	    mPose2->AddPMoy(aRes.mPTer,aRes.mBSurH);
      }

   }
   aSEr2 /= aS1;

   if (int(aPPM.Show().Val()) >= int(eNSM_Paquet))
   {
      if (mEqS)
         std::cout << "PDS Surf = " << aSomPdsSurf << "\n";
      std::cout << "| | | RESIDU LIAISON (pixel) =  Ter-Im :" << sqrt(aSEr2)
                << " pour ["  << mIm1 << "/" << mIm2 << "]"
		<< " Max = " << mEcMax
                << "\n";
   }

// getchar();
   return aSEr2 ;
}

void cObservLiaison_1Cple::ImageResidu(cAgglomRRL & anAgl)
{
   double anEchIm =  anAgl.EIRL().ScaleIm();
   bool isSigne =  anAgl.EIRL().Signed().Val();
   cResul_RL aRL1 (mPose1->Calib()->SzIm(),anEchIm);
   cResul_RL aRL2 (mPose2->Calib()->SzIm(),anEchIm);

   // double aFoc1 = mPose1->CF()->PIF().CurFocale();
   // double aFoc2 = mPose2->CF()->PIF().CurFocale();

   for 
   (
       ElPackHomologue::tCstIter itL=mPack.begin();
       itL!=mPack.end();
       itL++
   )
   {
      std::vector<double> aVP;
      const std::vector<Pt2dr> & aPTers = mPLiaisTer->UsePointLiaison(-1,-1,0.0,*itL,aVP,false).mEcIm;
      double aR1 =0;
      double aR2 = 0;
      if (! isSigne)
      {
          aR1 = euclid(aPTers[0]);
          aR2 = euclid(aPTers[1]);
      }
      else
      {
          ELISE_ASSERT
	  (
               false,
	       "Residu-signes-plus-supportes"
	  );
      }
      aRL1.Add(itL->P1(),aR1,itL->Pds());
      aRL2.Add(itL->P2(),aR2,itL->Pds());
   }

   anAgl.Add(aRL1,mPose1,mPose2);
   anAgl.Add(aRL2,mPose2,mPose1);
}



/**************************************************/
/*                                                */
/*                 cPackObsLiaison                */
/*                                                */
/**************************************************/

cPackObsLiaison::cPackObsLiaison
(
        cAppliApero & anAppli,
        const cBDD_PtsLiaisons & aBDL,
        int                      aCpt
)  :
   mAppli     (anAppli),
   mBDL       (aBDL),
   mId        (aBDL.Id()),
   mFlagArc   (mAppli.Gr().alloc_flag_arc())
{

    ELISE_ASSERT
    (
        aBDL.KeySet().size() == aBDL.KeyAssoc().size(),
        "KeySet / KeyAssoc sizes in BDD_PtsLiaisons"
    );
    int aNbTot = 0 ;
    for (int aKS=0 ; aKS<int(aBDL.KeySet().size()) ; aKS++)
    {
        // std::string aDir = mAppli.DC() + aBDL.Directory().Val();
        // std::list<std::string> aLName = RegexListFileMatch(aDir,aBDL.PatternSel(),1,false);

		std::string keyset =  aBDL.KeySet()[aKS];
		cInterfChantierNameManipulateur * iChantierNM = mAppli.ICNM();
        const std::vector<std::string> * aVName =  iChantierNM->Get(keyset);

        aNbTot += aVName->size();


// std::cout << "=========BDL " << aVName->size() << "\n"; getchar();

        if (1)
        {
            // cElRegex anAutom(aBDL.PatternSel(),20);
            bool aFirst = true;
            for 
            (
                  std::vector<std::string>::const_iterator itN = aVName->begin();
	          itN!=aVName->end();
	          itN++
            )
            {
	        bool aMultiple =  aBDL.UseAsPtMultiple().ValWithDef(true);
	        std::string aFulName =  mAppli.DC() +*itN;
	        if (aFirst)
	        {
	            mIsMult  = aMultiple;
                }
                else
	        {
	            ELISE_ASSERT(mIsMult==aMultiple,"Incoherence Multiple/No-Multiple");
	        }

	        // std::string aN1 = MatchAndReplace(anAutom,*itN,aBDL.NameIm1());
	        // std::string aN2 = MatchAndReplace(anAutom,*itN,aBDL.NameIm2());
	        std::pair<std::string,std::string> aPair = mAppli.ICNM()->Assoc2To1(aBDL.KeyAssoc()[aKS],*itN,false);
	        std::string aN1 = aPair.first;
	        std::string aN2 = aPair.second;

// std::cout << aN1 << " " << aN2 << "\n";
// getchar();
	        if (
                           (mAppli.NamePoseIsKnown(aN1) && mAppli.NamePoseIsKnown(aN2))
                       && ((aN1!=aN2) || (! aBDL.AutoSuprReflexif().Val()))
                   )
	        {
                    cPoseCam * aC1 =  mAppli.PoseFromName(aN1);
                    cPoseCam * aC2 =  mAppli.PoseFromName(aN2);
                    bool OkGrp = true;
                    if (aBDL.IdFilterSameGrp().IsInit())
                    {
                         OkGrp = mAppli.SameClass(aBDL.IdFilterSameGrp().Val(),*aC1,*aC2);
                    }
                    // std::cout << "GRP " << aC1->Name() << " " << aC2->Name() << " " << OkGrp << "\n";
                    if (OkGrp)
                    {
                        if (aN1==aN2)
                        {
                            std::cout << "FOR NAME POSE = " << aN1 << "\n";
                            ELISE_ASSERT(false,"Point homologue image avec elle meme !! ");
                        }
                        double aPds=0;
                        int    aNbHom=0;
	                if (mIsMult)
		        {
		            // cObsLiaisonMultiple * & anOLM = mDicoMul[aN1];

		            if (DicBoolFind(mDicoMul,aN1))
		            {
		                mDicoMul[aN1]->AddLiaison(aFulName,aN2,aKS==0);
		            }
		            else
		            {
                               mDicoMul[aN1]  = new  cObsLiaisonMultiple(anAppli,aFulName,aN1,aN2,aKS==0);
		            }
                            cObsLiaisonMultiple * anObs = mDicoMul[aN1];
                            ElPackHomologue aPack;
                            anObs->InitPack(aPack,aN2);
                            aPds = mAppli.PdsOfPackForInit(aPack,aNbHom);
		        }
		        else
		        {
                            ELISE_ASSERT(aKS==0,"Multiple Sets in Pts non multiple");
	                    cObservLiaison_1Cple * anObs= new cObservLiaison_1Cple(aBDL,aFulName,aN1,aN2);
		            if (anObs->NbH() !=0)
		            {
	                      mLObs.push_back(anObs);
	                      {
                                 cObservLiaison_1Cple * aO2 = mDicObs[aN1][aN2];
                                 if (aO2 !=0)
                                 {
                                    std::cout << " For : " << mId << " " << aN1 <<  " " << aN2 <<"\n"; 
		                    ELISE_ASSERT(false,"Entree multiple\n");
                                 }
	                      }
	                      mDicObs[aN1][aN2] = anObs;
                              aPds = mAppli.PdsOfPackForInit(anObs->Pack(),aNbHom);
		            }
		        }
                        if (aBDL.SplitLayer().IsInit())
                        {
                            mAppli.SplitHomFromImageLayer(*itN,aBDL.SplitLayer().Val(),aN1,aN2);
                        }
                        mAppli.AddLinkCam(aC1,aC2);
                        if (aCpt==0)
                        {
                           tGrApero::TSom * aS1 =  mAppli.PoseFromName(aN1)->Som();
                           tGrApero::TSom * aS2 =  mAppli.PoseFromName(aN2)->Som();
                           tGrApero::TArc * anArc = mAppli.Gr().arc_s1s2(*aS1,*aS2);
                           if (!anArc) 
                           {
                              cAttrArcPose anAttr;
                              anArc = & mAppli.Gr().add_arc(*aS1,*aS2,anAttr);
                           }
                           anArc->attr().Pds() = aPds;
                           anArc->attr().Nb() = aNbHom;
                           // anArc->arc_rec().attr() = anArc->attr();
                        }
	            }
	        }
	        aFirst = false;
            }
        }
        else
        {
            // On rajoutera ici le cas ou fichier contient N Pack
        }
    }

    // for (std::list<std::string>::cons_iterator itP

    std::cout << "NB PACK PTS " << aNbTot << "\n";
    if (aNbTot == 0)
    {
        std::cout << "FOR LIAISONS " <<  aBDL.Id() << "\n";
        ELISE_ASSERT(false,"Cannot find any pack\n");
    }
}



void cPackObsLiaison::Compile()
{
   if (mIsMult)
   {
      for
      (
           std::map<std::string,cObsLiaisonMultiple *>::iterator itOML=mDicoMul.begin();
           itOML!=mDicoMul.end();
           itOML++
      )
      {
           // std::cout << "COMP " << itOML->first << "#" << itOML->second << "\n";
	   
	   cSurfParam  * aSurf = 0;
	   for (int aKP=0 ; aKP<int(mPatI1EqPl.size()); aKP++)
	       if (mPatI1EqPl[aKP]->Match(itOML->first))
	       {
	           if ((aSurf!=0) && (aSurf!=mVSurfCstr[aKP]))
		   {
		       std::cout << "For Name Im " << itOML->first << "\n";
                       ELISE_ASSERT(false,"Contrainte surfacique multiple");
		   }
	           aSurf = mVSurfCstr[aKP];
		   aSurf->SetUsed();
               }

           itOML->second->Compile(aSurf);
	   // std::cout << "END C \n";
      }
   }
   else
   {
      for
      (
           std::vector<cObservLiaison_1Cple *>::iterator itObs=mLObs.begin();
           itObs!=mLObs.end();
           itObs++
      )
      {
	   cSurfParam *aSurf = 0;
	   for (int aKP=0 ; aKP<int(mPatI1EqPl.size()); aKP++)
	   {
	       if (    (mPatI1EqPl[aKP]->Match((*itObs)->NameIm1()))
	            && (mPatI2EqPl[aKP]->Match((*itObs)->NameIm2()))
		  )
	       {
	           if ((aSurf!=0) && (aSurf!=mVSurfCstr[aKP]))
		   {
		       std::cout << "For Name Im " << (*itObs)->NameIm1()
		                 << "#" << (*itObs)->NameIm2()<< "\n";
                       ELISE_ASSERT(false,"Contrainte surfacique multiple");
		   }
	           aSurf = mVSurfCstr[aKP];
		   aSurf->SetUsed();
	       }
           }

           (*itObs)->Compile(aSurf,mAppli);
      }
   }
}

void cPackObsLiaison::AddLink()
{
    for
    (
         std::map<std::string,cObsLiaisonMultiple *>::const_iterator itO=mDicoMul.begin();
         itO!=mDicoMul.end();
         itO++
    )
    {
         itO->second->AddLink();
    }
}

std::list<cPoseCam *> cPackObsLiaison::ListCamInitAvecPtCom
                      (
                             cPoseCam *         aCam1
                      )
{
   std::set<cPoseCam *>   aRes;
   const std::string & aName1 = aCam1->Name();
   if (mIsMult)
   {
      if (DicBoolFind(mDicoMul,aName1))
      {
         const std::vector<cOneElemLiaisonMultiple *> & aVP = 
                                     mDicoMul[aName1]-> VPoses();
         for (int aK=0 ; aK<int(aVP.size()) ; aK++)
         {
             if (aVP[aK]->Pose()->RotIsInit())
             {
                aRes.insert(aVP[aK]->Pose());
             }
         }
      }
   
   }
   else
   {
       if (DicBoolFind(mDicObs,aName1))
       {
          const std::map<std::string,cObservLiaison_1Cple *> aDic = mDicObs[aName1];
          for
          (
              std::map<std::string,cObservLiaison_1Cple *>::const_iterator iT=aDic.begin();
              iT!=aDic.end();
              iT++
          )
          {
               cPoseCam * aPose2 = mAppli.PoseFromName(iT->first);
               if (aPose2->RotIsInit())
               {
                  aRes.insert(aPose2);
               }
          }
       }
   }

   return std::list<cPoseCam *>(aRes.begin(),aRes.end());
}

cObsLiaisonMultiple * cPackObsLiaison::ObsMulOfName(const std::string & aName)
{
   cObsLiaisonMultiple * aRes = mDicoMul[aName];
   if (aRes==0)
   {
       std::cout << "For Name =" << aName << "\n";
       ELISE_ASSERT(false,"cPackObsLiaison::ObsMulOfName");
   }
   return aRes;
}



bool cPackObsLiaison::InitPack
     (
          ElPackHomologue & aPack,
          const std::string& aNN1, 
          const std::string& aNN2
     )
{
     std::string aN1 = mAppli.PoseFromName(aNN1)->Name();
     std::string aN2 = mAppli.PoseFromName(aNN2)->Name();
     if (mIsMult)
     {
        if (DicBoolFind(mDicoMul,aN1) &&  mDicoMul[aN1]->InitPack(aPack,aN2))
	{
           return false;
	}

        if (DicBoolFind(mDicoMul,aN2) &&  mDicoMul[aN2]->InitPack(aPack,aN1))
	{
           aPack.SelfSwap();
           return true;
	}
     }
     else
     {
         cObservLiaison_1Cple * aO2 = mDicObs[aN1][aN2];
         if (aO2)
         {
             aPack = aO2->Pack();
	     return false;
         }
         aO2 = mDicObs[aN2][aN1];
         if (aO2)
         {
            aPack = aO2->Pack();
            aPack.SelfSwap();
            return true;
         }
    }
    std::cout << " For : " << mId << " " << aN1 <<  " " << aN2 << "\n"; 
    std::cout << " Mult " << mIsMult << "\n";
    ELISE_ASSERT(false,"Cannot find liaison");
    return false;
}

void cPackObsLiaison::AddContrainteSurfParam
     (
         cSurfParam * aSurf,
         cElRegex *  aPatI1,
         cElRegex *  aPatI2
     )
{
    mVSurfCstr.push_back(aSurf);
    mPatI1EqPl.push_back(aPatI1);
    mPatI2EqPl.push_back(aPatI2);
}


class cCpleCmpEcMax
{
   public :
      bool operator () (cObservLiaison_1Cple * const & aCpl1,
                   cObservLiaison_1Cple * const & aCpl2)
      {
         return aCpl1->EcMax() < aCpl2->EcMax();
      }
};



void  cPackObsLiaison::GetPtsTerrain
      (
          const cParamEstimPlan & aPEP,
          cSetName &                    aSelectorEstim,
          cArgGetPtsTerrain &           anArg,
          const char *                  anAttr
      )
{
    cStatObs  aSO(false);
    if ( aPEP.AttrSup().IsInit())
       anAttr =  aPEP.AttrSup().Val().c_str();
    ELISE_ASSERT
    (
         mIsMult,
         "Require PMUL for cPackObsLiaison::GetPtsTerrain"
    );
    cPonderationPackMesure aPPM = aPEP.Pond();
    aPPM.Add2Compens().SetVal(false);
    for
    (
           std::map<std::string,cObsLiaisonMultiple *>::iterator itOML=mDicoMul.begin();
           itOML!=mDicoMul.end();
           itOML++
    )
    {
          cObsLiaisonMultiple * anOLM = itOML->second;
          std::string aNameP = anOLM->Pose1()->Name();
          if (aSelectorEstim.IsSetIn(aNameP))
          {
              Im2D_Bits<1> aM(1,1);
              if (aPEP.KeyCalculMasq().IsInit())
              {
                  std::string aNameM =  
                        anAttr ?
                        mAppli.ICNM()->Assoc1To2(aPEP.KeyCalculMasq().Val(),aNameP,anAttr,true):
                        mAppli.ICNM()->StdCorrect(aPEP.KeyCalculMasq().Val(),aNameP,true);
                  aNameM = mAppli.DC() + aNameM;
                  Tiff_Im aTF = Tiff_Im::UnivConvStd(aNameM);
                  Pt2di aSz = aTF.sz();
                  aM = Im2D_Bits<1>(aSz.x,aSz.y);
                  ELISE_COPY(aTF.all_pts(),aTF.in_bool(),aM.out());
                  anArg.SetMasq(&aM);
              }
              anOLM->AddObsLM(aPPM,0,&anArg,(cArgVerifAero*)0,aSO,0);
              anArg.SetMasq(0);
          }
    }
}



double cPackObsLiaison::AddObs
       (
            const cPonderationPackMesure & aPond,
            const cPonderationPackMesure * aPondSurf,
            cStatObs & aSO,
            const cRapOnZ * aRAZ
       )
{
   double aS1=0;
   double aSEr=0;

// SPECIAL DEBUG CHOLESKY
   cElRegex aRegDebug("062",10);

   if (mIsMult)
   {
      for (int aK= 0 ; aK< 2; aK++)
      {
          for
          (
               std::map<std::string,cObsLiaisonMultiple *>::iterator itOML=mDicoMul.begin();
               itOML!=mDicoMul.end();
               itOML++
          )
          {
// std::cout << "OOLLM "<< itOML->first << "\n";
             cObsLiaisonMultiple * anOLM = itOML->second;
             cPoseCam * aPC = anOLM->Pose1() ;
             bool IsDebug = aRegDebug.Match(aPC->Name());
             bool aDoIt = (aK==0) ? IsDebug : (!IsDebug);
             if (aDoIt)
             {
                 if (aSO.AddEq() || aPondSurf || aPond.IdFilter3D().IsInit())
                 {
                    aS1++;
	            aSEr +=  anOLM->AddObsLM(aPond,aPondSurf,0,(cArgVerifAero*)0,aSO,aRAZ);
                 }
                 else
                 {
                    aSEr +=  anOLM->BasicAddObsLM (aPond,aSO,aRAZ);
                 }
             }
         }
      }
   }
   else
   {
       cCpleCmpEcMax aCmp;
       std::sort(mLObs.begin(),mLObs.end(),aCmp);
       for 
       (
           std::vector<cObservLiaison_1Cple *>::iterator it1C=mLObs.begin();
           it1C != mLObs.end();
           it1C++
       )
       {
           if (((*it1C)->Pose1()->RotIsInit())  && ((*it1C)->Pose2()->RotIsInit()))
           {
              aS1++;
              aSEr += (*it1C)->AddObs(aPond,aPondSurf);
           }
       }
   }
   aSEr /= aS1;

   if (aS1 && (int(aPond.Show().Val()) >= int(eNSM_Iter)))
   {
       mAppli.COUT() << "| | " << " RESIDU LIAISON MOYENS = "  
                 << sqrt(aSEr) << " pour " << mId << "\n";

   }

   return aSEr;
}



void  cPackObsLiaison::OneExportRL(const cExportImResiduLiaison & anEIL) const
{
   cAgglomRRL anAggl(mAppli,anEIL);
   for 
   (
       std::vector<cObservLiaison_1Cple *>::const_iterator it1C=mLObs.begin();
       it1C != mLObs.end();
       it1C++
   )
   {
        (*it1C)->ImageResidu(anAggl);
   }
}


/**************************************************/
/*                                                */
/*                  cAppliApero                   */
/*                                                */
/**************************************************/

const cBDD_PtsLiaisons & cAppliApero::GetBDPtsLiaisonOfId(const std::string & anId)
{
    for 
    (
       std::list<cBDD_PtsLiaisons>::const_iterator itB=mParam.BDD_PtsLiaisons().begin();
       itB!=mParam.BDD_PtsLiaisons().end();
       itB++
    )
    {
        if (itB->Id() == anId)
        {
           return *itB;
        }
    }
    std::cout << "For Id=[" << anId << "]\n";
    ELISE_ASSERT(false,"cAppliApero::GetBDPtsLiaisonOfId");
    return * ((cBDD_PtsLiaisons*)0);
}

void cAppliApero::InitBDDLiaisons()
{
    int aCpt = 0;
    for 
    (
       std::list<cBDD_PtsLiaisons>::const_iterator itB=mParam.BDD_PtsLiaisons().begin();
       itB!=mParam.BDD_PtsLiaisons().end();
       itB++
    )
    {
        if (aCpt==0)
           mSymbPack0 = itB->Id();
        NewSymb(itB->Id());
	mDicoLiaisons[itB->Id()] = new cPackObsLiaison(*this,*itB,aCpt);
        aCpt++;
    }
}



bool cAppliApero::InitPack
     ( 
	    const std::string& anId,
            ElPackHomologue & aPack,
            const std::string& aN1,
            const std::string& aN2
     )
{
   return PackOfInd(anId)->InitPack(aPack,aN1,aN2);
}

std::list<cPoseCam *> cAppliApero::ListCamInitAvecPtCom
                      (
                             const std::string& anId,
                             cPoseCam *         aCam1
                      )
{
   return PackOfInd(anId)->ListCamInitAvecPtCom(aCam1);
}


cPackObsLiaison * cAppliApero::PackOfInd(const std::string& anId)
{
   cPackObsLiaison * aPOL = mDicoLiaisons[anId];
   if (aPOL==0)
   {
       std::cout << "---------------- ID = " << anId <<"\n";
       ELISE_ASSERT(false,"Cannot find Id for pack liaison");
   }
   return aPOL;
}


cObsLiaisonMultiple * cAppliApero::PackMulOfIndAndNale
                     (
                                  const std::string& anId,
                                  const std::string& aName
                     )
{
     return PackOfInd(anId)->ObsMulOfName(aName);
}

bool  cAppliApero::InitPackPhgrm
     (
          const std::string& anId,
          ElPackHomologue & aPack,
          const std::string& aN1,CamStenope * aCam1,
          const std::string& aN2,CamStenope * aCam2
     )
{
// std::cout << aN1 << " " << aN2 << "\n";
   bool aRes = InitPack(anId,aPack,aN1,aN2);
   aPack = aCam1->F2toPtDirRayonL3(aPack,aCam2);
   return aRes;
}



void  cAppliApero::CompileLiaisons()
{
    for 
    (
        tDiLia::iterator itD=mDicoLiaisons.begin();
        itD!= mDicoLiaisons.end();
        itD++
    )
    {
        itD->second->Compile();
        itD->second->AddLink();
    }
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
