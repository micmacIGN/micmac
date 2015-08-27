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
#include "ext_stl/numeric.h"

namespace NS_ParamApero
{

cMTActive::cMTActive(int aNbPer) :
   mNbPer(aNbPer)
{
}

void cMTActive::SetKCur(int aKCur)
{
   mKCur = aKCur;
}

bool cMTActive::SelectVal(const Pt2dr & aP) const
{
   return SelectVal(aP.x+10.32 * aP.y);
}

bool cMTActive::SelectVal(const double & aV) const
{
   return mod(round_ni(aV*108.3871),mNbPer)!=mKCur;
}

      //=========================================


void cMTResult::AddResisu(double aVal)
{
  mVRes.back() += ElSquare(aVal);
  mVNb.back() += 1;
}

void cMTResult::NewSerie()
{
  mVRes.push_back(0);
  mVNb.push_back(0);
  std::vector<CamStenope *> aVC;
  mVCams.push_back(aVC);
}

void cMTResult::AddCam(CamStenope * aCam)
{
    mVCams.back().push_back(aCam);
}

bool cMTResult::IsActif()
{
   return mIsActif ;
}

void cMTResult::SetActif()
{
   mIsActif = true ;
}

void cMTResult::SetInactif()
{
   mIsActif = false;
}

cMTResult::cMTResult() :
   mIsActif (false)
{
}


void cAppliApero::AddMTResisu(const double & aDErr) 
{
    if (mMTRes &&  mMTRes->IsActif())
    {
               mMTRes->AddResisu(aDErr);
    }
}
  
void cMTResult::Show(FILE* aFP)
{
   fprintf(aFP,"======== ERREUR EN EXTRAPOLATION ======\n\n" );
   double aSRes=0,aSNb = 0;
   for (int aK=0 ; aK<int(mVRes.size()) ; aK++)
   {
        fprintf(aFP,"   Test %d : %f ",aK,sqrt(mVRes[aK]/mVNb[aK]));
        for (int aJ=0 ; aJ<int(mVCams[aK].size()) ; aJ++)
        {
            CamStenope * aCS=mVCams[aK][aJ];
            Pt2dr aPP = aCS->PP();
            fprintf(aFP," ; Focale %f Ppx %f Ppy %f",aCS->Focale(),aPP.x,aPP.y);
        }
        aSRes += mVRes[aK];
        aSNb += mVNb[aK];
        fprintf(aFP,"\n");
   }
   aSRes /= aSNb ;

   fprintf(aFP,"  *********************************\n");
   fprintf(aFP,"  **  MOY %f\n",sqrt(aSRes));
   fprintf(aFP,"  *********************************\n");
}


void cAppliApero::AddCamsToMTR()
{
   for (tDiCal::iterator it=mDicoCalib.begin(); it!=mDicoCalib.end() ; it++)
       mMTRes->AddCam(it->second->PIF().DupCurPIF());
     
}

/**************************************************/
/*                                                */
/*          APPUIS                                */
/*                                                */
/**************************************************/

  //    cTypeEnglob_Appuis
  //
  //

std::list<Appar23> cTypeEnglob_Appuis::CreateFromXML
                   (
                         cAppliApero & anAppli,
		         const std::string & aNameXML,
			 const cBDD_PtsAppuis& aBd,
			 cObserv1Im<cTypeEnglob_Appuis> &
                   )
{
   std::list<Appar23>  aRes =  Xml2EL(StdGetObjFromFile<cListeAppuis1Im>
                      (
                          aNameXML,
                          StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                          aBd.TagExtract().Val(),
                          // "ListeAppuis1Im",
                          "ListeAppuis1Im"
			)
                );

   std::list<Appar23> aFiltered;
   for 
   (
        std::list<Appar23>::const_iterator itA=aRes.begin();
        itA!=aRes.end();
        itA++
   )
   {
      if (anAppli.AcceptCible(itA->mNum))
      {
         aFiltered.push_back(*itA);
      }
      else
      {
      }
   }
   aRes = aFiltered;

   if (aBd.ToSubstract().IsInit())
   {
       Pt3dr aSub = aBd.ToSubstract().Val();
       for (std::list<Appar23>::iterator itA=aRes.begin(); itA!=aRes.end() ; itA++)
           itA->pter = itA->pter-aSub;
   }

   return aRes;
}

       // 
       // cOneAppuiMul
       //

cOneAppuiMul::cOneAppuiMul(const Pt3dr & aPTer,int aNum) :
    mPTer  (aPTer),
    mNum   (aNum),
    mPt    (0,1.0)
{
}

void cOneAppuiMul::AddPt(cPoseCam * aPC,const Pt2dr & aPIm)
{
    mVPds.push_back(1.0);
    mPoses.push_back(aPC);
    mPt.AddPts(aPIm);
}

const Pt3dr & cOneAppuiMul::PTer() const
{
   return mPTer;
}

int   cOneAppuiMul::NbInter() const
{
   return mVPds.size();
}

Pt3dr cOneAppuiMul::PInter() const
{
   return InterFaisceaux(mVPds,mPoses,mPt);
}





       // 
       // cPackGlobAppuis
       //

cPackGlobAppuis::cPackGlobAppuis() :
   mDicoApps (0),
   mNumCur   (0)
{
}

int  cPackGlobAppuis::GetNum(const Appar23 & anAp,const cBddApp_AutoNum & anAN)
{

    int aNum = -1;
    for
    (
       std::map<int,cOneAppuiMul *>::const_iterator itA=mDicoApps->begin();
       itA != mDicoApps->end();
       itA++
    )
    {
        double aD = euclid(anAp.pter-itA->second->PTer());
        if (aD < anAN.DistAmbiguite())
        {
             if (aD<anAN.DistFusion())
             {
                 if (aNum>=0)
                 {
                    std::cout << "NUMS =" << aNum << " " << itA->first << "\n";
                    ELISE_ASSERT(false,"Fusion , trouve plusieurs points !");
                 }
                 aNum = itA->first;
// std::cout <<  anAp.pter << "--------------------GOT FUSION \n";
             }
             else
             {
                 std::cout << "FOR " << anAp.pter << itA->second->PTer() << "\n";
                 ELISE_ASSERT(false,"Fusion point terrain ambigue ");
             }
        }
    }

    if (aNum>=0) 
       return aNum;

    return mNumCur++;
}

void  cPackGlobAppuis::AddOsb1Im
      (
         cObserv1Im<cTypeEnglob_Appuis> & anObs,
         const cBDD_PtsAppuis &                 aBDD_App
      )
{
   for
   (
       std::list<Appar23>::iterator itP=anObs.Vals().begin();
       itP!=anObs.Vals().end();
       itP++
   )
   {
       int aN = itP->mNum;
       if (aN <0)
       {
          if (aBDD_App.BddApp_AutoNum().IsInit())
          {
              aN = GetNum(*itP,aBDD_App.BddApp_AutoNum().Val());
              itP->mNum = aN;
          }
          else
          {
             ELISE_ASSERT
             (aN>=0," Num<0 For Point Id cPackGlobAppuis::AddOsb1Im");
          }
       }
// std::cout << "NUM = " << aN << "\n";
       cOneAppuiMul * & aPM = (*mDicoApps)[aN];
       if (aPM==0)
       {
          aPM = new cOneAppuiMul(itP->pter,aN);
          // std::cout << "NEW \n";
       }
       else
       {
          // std::cout << "OLD \n";
       }
       aPM->AddPt(anObs.PC(),itP->pim);
    }
}


void  cPackGlobAppuis::AddObsPack
      (
          cPackObserv1Im<cTypeEnglob_Appuis,cPackGlobAppuis> & aPack,
          const cBDD_PtsAppuis & aBDD_Ap
      )
{
   if (mDicoApps==0)
   {
        mDicoApps = new  std::map<int,cOneAppuiMul *> ;
        std::list<cObserv1Im<cTypeEnglob_Appuis> *> & aLobs = aPack.LObs();
        for
        (
              std::list<cObserv1Im<cTypeEnglob_Appuis> *>::iterator itO=aLobs.begin();
              itO!=aLobs.end();
              itO++
        )
           AddOsb1Im(**itO,aBDD_Ap);
   }
}


std::map<int,cOneAppuiMul *> *  cPackGlobAppuis::Apps()
{
   return mDicoApps;
}


/**************************************************/
/*                                                */
/*                                                */
/*                                                */
/**************************************************/

ElRotation3D cTypeEnglob_Orient::CreateFromXML
             (
                  cAppliApero &,
	          const std::string & aNameXML,
		  const cBDD_Orient &anArg,
		  cObserv1Im<cTypeEnglob_Orient> & anObs
             )
{
// std::cout << "AAAAAAAAAAAAa"  << aNameXML << "\n";
    std::string aPost = StdPostfix(aNameXML);
    cOrientationExterneRigide anOER;
    ElAffin2D anOrInt = ElAffin2D::Id();

    // eConventionsOrientation aConv = eConvApero_DistM2C;
    cConvExplicite aConvE = GlobMakeExplicite(eConvApero_DistM2C);

    if ((aPost == "ori") || (aPost == "ORI"))
    {
          Ori3D_Std anOri(aNameXML.c_str());
          anOER = From_Std_RAff_C2M(anOri.GetOrientation(),true);
    }
    else
    {
          cOrientationConique anOC = StdGetObjFromFile<cOrientationConique>
                             (
                                 aNameXML,
                                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                 "OrientationConique",
                                 "OrientationConique"
                             );
         anOrInt = AffCur(anOC);
         // AssertOrIntImaIsId(anOC);
         anOER = anOC.Externe();


         if (anArg.ConvOr().IsInit())
            aConvE = GlobMakeExplicite(anArg.ConvOr().Val());
         // aConv = anArg.ConvOr().ValWithDef(aConv);
         if (anOER.KnownConv().IsInit())
            aConvE = GlobMakeExplicite(anOER.KnownConv().Val());
         // aConv = anOER.KnownConv().ValWithDef(aConv);
         aConvE = GlobMakeExplicite(anOC.ConvOri());

    }
/*
std::cout << anOER.Centre() << "\n";
std::cout << (anOER.Centre().y -1.759e+06) << "\n";
std::cout << anOER.L1() << "\n";
std::cout << anOER.L2() << "\n";
std::cout << anOER.L3() << "\n";
*/


    anObs.mAltiSol = anOER.AltiSol().ValWithDef(ALTISOL_UNDEF());
    anObs.mProfondeur = anOER.Profondeur().ValWithDef(-1);
    anObs.mTime = anOER.Time().Val();
    anObs.mOrIntC2M = anOrInt.inv();


    //cConvExplicite aConvE = GlobMakeExplicite(aConv);

    return GlobStd_RAff_C2M (anOER,aConvE);

}

Pt3dr cTypeEnglob_Centre::CreateFromXML
             (              
                  cAppliApero & anAppli,
	          const std::string & aNameXML,
		  const cBDD_Centre &anArg,
		  cObserv1Im<cTypeEnglob_Centre> & anObs
             )
{
   Pt3dr aRes=  StdGetObjFromFile<Pt3dr>
                     (
                          aNameXML,
                          (std::string)"include"+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR+"ParamChantierPhotogram.xml",
                          anArg.Tag().Val(),
                          "Pt3dr"
                     );

    if (anArg.CalcOffsetCentre().IsInit())
    {
        const cCalcOffsetCentre & aCOC = anArg.CalcOffsetCentre().Val();
        ELISE_ASSERT
        (
             ! aCOC.OffsetUnknown().Val(),
             "CalcOffsetCentre ne gere pas (encore ?)  les offset variable"
        );
        std::string aBande = anAppli.ICNM()->Assoc1To1
                             (
                                 aCOC.KeyCalcBande(),
                                 anObs.Im(),
                                 true
                             );

        Pt3dr anOFs = anAppli.ICNM()->GetPt3dr(aCOC.IdBase(),aBande);
        
        aRes = aRes + anOFs;
        // std::cout << anOFs << "\n";
    }

    return aRes;
}


/**************************************************/
/*                                                */
/*              cObservAppuis_1Im                 */
/*                                                */
/**************************************************/

void cObserv1ImPostInit
     (
          cObserv1Im<cTypeEnglob_Appuis> & anObs,
	  const cBDD_PtsAppuis &           aBPA,
          cAppliApero & anAppli,
          const std::string& aNameIm
     )
{
   Appar23 aBary = BarryImTer( anObs.mVals);
   anObs.mBarryTer = aBary.pter;
   if (aBPA.SzImForInvY().IsInit())
      InvY(anObs.mVals,aBPA.SzImForInvY().Val(),aBPA.InvXY().Val());

   cPoseCam *  aPose = anAppli.PoseFromName(aNameIm);
   for 
   (
       std::list<Appar23>::iterator itL=anObs.mVals.begin();
       itL!=anObs.mVals.end();
       itL++
   )
   {
       aPose->C2MCompenseMesureOrInt(itL->pim);
   }
}
void cObserv1ImPostInit(cObserv1Im<cTypeEnglob_Orient> &,const cBDD_Orient&,cAppliApero & anAppli,const std::string& aNameIm)
{
}


void cObserv1ImPostInit(cObserv1Im<cTypeEnglob_Centre> &,const cBDD_Centre &,cAppliApero & anAppli,const std::string& aNameIm)
{
}




template <class  TypeEngl>
cObserv1Im<TypeEngl>::cObserv1Im   
(
      cAppliApero & anAppli,
      const std::string& aNamePack,
      const std::string& aNameIm,
      const typename TypeEngl::tArg & anArg
)  :
   mAppli (anAppli),
   mIm     (aNameIm),
   mPose   (0),
   mCF     (0),
   mVals (TypeEngl::CreateFromXML(anAppli,aNamePack,anArg,*this))
{
   cObserv1ImPostInit(*this,anArg,anAppli,aNameIm);
}


template <class  TypeEngl>
cObserv1Im<TypeEngl>::cObserv1Im   
(
      cAppliApero & anAppli,
      typename TypeEngl::tObj aVals,
      const std::string& aNameIm
)  :
   mAppli (anAppli),
   mIm     (aNameIm),
   mPose   (0),
   mCF     (0),
   mVals (aVals)
{
}

template <class  TypeEngl>
const std::string   &   cObserv1Im<TypeEngl>::Im() const
{
   return mIm;
}


template <class  TypeEngl>
cPoseCam * cObserv1Im<TypeEngl>::PC() const
{
   return mPose;
}




template <class  TypeEngl>
const typename TypeEngl::tObj  & cObserv1Im<TypeEngl>::Vals() const
{
   return mVals;
}

template <class  TypeEngl>
typename TypeEngl::tObj  & cObserv1Im<TypeEngl>::Vals() 
{
   return mVals;
}

template <class  TypeEngl>
void cObserv1Im<TypeEngl>::Compile( cAppliApero & anAppli) 
{
    if ( anAppli.PoseExist(mIm))
    {
         mPose = anAppli.PoseFromName(mIm);
         mCF = mPose->CF();
         // mCF = anAppli.PoseFromName(mIm)->CF();
    }
}


template class cObserv1Im<cTypeEnglob_Appuis >;
template class cObserv1Im<cTypeEnglob_Orient >;


double  cAppliApero::AddAppuisOnePose
      (
          const cObsAppuis & anArg,
          cObserv1Im<cTypeEnglob_Appuis> * anObs,
          std::vector<cRes1OnsAppui> * aVRes,
          cStatObs & aSO,
          double & aGlobSomErPds, double & aGlobSomPoids

      )
{

   const  std::list<Appar23> & aLAp = anObs->mVals;
   const cPonderationPackMesure & aPPM = anArg.Pond();
   cCameraFormelle & aCF = *(anObs->mCF);
   cCalibCam * aCalib = anObs->PC()->Calib();
   ElRotation3D anOr =  aCF.CurRot();
   // Pt3dr aCOpt = anOr.ImAff(Pt3dr(0,0,0));
   // Pt3dr aCDG = anObs->mBarryTer;


   cPonderateur aPdrt(aPPM,aLAp.size());


   double aSomEr= 0;
   double aNbEr = 0;

   double aSomErPds= 0;
   double aSomPoids = 0;

   int aKP=0;
         // std::cout << "SCCNnnnnnnnnn "<<  (aCalib->PIF().StdScaleN()) << "\n";
   for(std::list<Appar23>::const_iterator itA=aLAp.begin();itA!=aLAp.end();itA++)
   {
      if (aCalib->IsInZoneU(itA->pim))
      {
         Pt2dr aResidu = aCF.ResiduAppui(itA->pter,itA->pim);
// std::cout << aResidu << "\n"; getchar();
         double aDErr = euclid(aResidu)  ;//  * (aCalib->CamInit().ScaleCamNorm());
         aSomEr += ElSquare(aDErr);
         aNbEr ++;
         double aPdsE = aPdrt.PdsOfError(aDErr);

         aSomErPds +=  aPdsE *  ElSquare(aDErr);
         aSomPoids += aPdsE;


         if (aVRes)
         {
            aVRes->push_back(cRes1OnsAppui(itA->mNum,anObs->mPose,itA->pim,aResidu));
         }
         if (int(aPPM.Show().Val()) >= int (eNSM_Indiv))
         {
            std::cout << "| | | | RESIDU Appuis=" << aDErr  
                       << " Pour Pts numero " << aKP 
                       << " Id=" << itA->mNum 
                       << " Im=" << itA->pim 
                       << "\n";
            if (aDErr > 1e2) 
            {
                CamStenope * aCS = aCF.NC_CameraCourante() ;

                std::cout 
                     << "HHIG RESIDU !!!!!   Im2=" << aCS->R3toF2( itA->pter)
                     << "Residu = "<< aResidu
                    << " Ter=" << itA->pter
                    << "\n";
                // getchar();
            }
         }

         if (PIsActif(itA->pim))
         {
             Pt2dr aP = aCF.AddAppui(itA->pter,itA->pim,aSO.AddEq() ? aPdsE : 0.0);
             aSO.AddSEP(aPdsE*ElSquare(aP.x)+aPdsE*ElSquare(aP.y));
         }
         else
         {
            AddMTResisu(aDErr);
         }

         aKP++;
      }
   }

   aGlobSomErPds += aSomErPds;
   aGlobSomPoids += aSomPoids;

   aSomEr /= aNbEr;
   aSomErPds /= aSomPoids;

   if (int(aPPM.Show().Val()) >= int (eNSM_Paquet))
   {
     std::cout << "| | | RESIDU Appuis moyen =" << sqrt(aSomEr) 
               << " RPOND " << sqrt(aSomErPds)
               << " pour pose " << anObs->mIm << "\n";
   }


   return aSomEr;
}


/**************************************************/
/*                                                */
/*                 cPackObsAppuis                 */
/*                                                */
/**************************************************/

template <class TypeEngl,class TGlob> void cPackObserv1Im<TypeEngl,TGlob>::Add(cObserv1Im<TypeEngl> * anObs)
{
     mLObs.push_back(anObs);
     {
         cObserv1Im<TypeEngl> * aO2 = mDicObs[anObs->Im()];
         if (aO2 !=0)
         {
              std::cout << " For : " << mId << " " << anObs->Im()  <<"\n"; 
	      ELISE_ASSERT(false,"Entree multiple\n");
         }
     }
     mDicObs[anObs->Im()] = anObs;
}

template <class TypeEngl,class TGlob> 
cPackObserv1Im<TypeEngl,TGlob>::cPackObserv1Im
(
        cAppliApero & anAppli,
        const std::string & anId
)  :
   mAppli  (anAppli),
   mId     (anId),
   mArg    (0)
{
// std::cout << mId << " ::cPackObserv1Im    ----------\n"; getchar();
}


template <class TypeEngl,class TGlob> TGlob & cPackObserv1Im<TypeEngl,TGlob>:: Glob()
{
    return mGlob;
}



template <class TypeEngl,class TGlob>    
   typename TypeEngl::tArg &  cPackObserv1Im<TypeEngl,TGlob>::Arg()
{
   ELISE_ASSERT(mArg!=0,"cPackObserv1Im<TypeEngl,TGlob>::Arg");
   return *mArg;
}

template <class TypeEngl,class TGlob> 
cPackObserv1Im<TypeEngl,TGlob>::cPackObserv1Im
(
        cAppliApero & anAppli,
        const typename TypeEngl::tArg & anArg
)  :
   mAppli  (anAppli),
   mId     (anArg.Id()),
   mArg    (new typename TypeEngl::tArg(anArg))
{
    const std::vector<std::string> * aVName =  mAppli.ICNM()->Get(anArg.KeySet());

    anAppli.COUT() << "Pack Obs " << anArg.KeySet() << " NB " << aVName->size() << "\n";

    if (1)
    {
        for (int aK=0;aK<int(aVName->size());aK++)
        {
	    std::string aNamePack = (*aVName)[aK];

	    std::string aNameIm  = mAppli.ICNM()->Assoc1To1(anArg.KeyAssoc(),aNamePack,false);
            if ( mAppli.NamePoseIsKnown(aNameIm))
	    {

	        cObserv1Im<TypeEngl> * anObs= new  cObserv1Im<TypeEngl>(anAppli,anAppli.DC()+aNamePack,aNameIm,anArg);
                Add(anObs);
            }
            else
            {
                 static bool first = true;
                 if (first)
                    std::cout << "WARN, For Pack=" << aNamePack << " Im=" << aNameIm << " Do No exist\n";
                 first = false;
                // ELISE_ASSERT(false,"Cannot find image for pack appuis");
            }
        }
    }
    else
    {
        // On rajoutera ici le cas ou fichier contient N Pack
    }
}

template <class TypeEngl,class TGlob> 
         void cPackObserv1Im<TypeEngl,TGlob>::Compile()
{
   for 
   (
      typename  std::list<cObserv1Im<TypeEngl> *> ::iterator itOb = mLObs.begin();
      itOb != mLObs.end();
      itOb++
   )
   {
       (*itOb)->Compile(mAppli);
   }
}

template <class TypeEngl,class TGlob> cObserv1Im<TypeEngl>  & cPackObserv1Im<TypeEngl,TGlob>::Obs (const std::string & aName) 

{
     cObserv1Im<TypeEngl> * anObs = mDicObs[aName];
     if (anObs==0)
     {
          std::cout << " For : " << mId << " " << aName << "\n"; 
          std::cout << "Size Dico " << mDicObs.size()-1 << "\n";
          ELISE_ASSERT(false,"Cannot find appuis");
     }
     return *anObs;
}

template <class TypeEngl,class TGlob> cObserv1Im<TypeEngl>  * cPackObserv1Im<TypeEngl,TGlob>::PtrObs (const std::string & aName) 

{
     typename std::map<std::string,cObserv1Im<TypeEngl> *>::iterator itD = mDicObs.find(aName);
     if (itD != mDicObs.end())
        return itD->second;
     return 0;
}




template <class TypeEngl,class TGlob> 
const  typename  TypeEngl::tObj & 
    cPackObserv1Im<TypeEngl,TGlob>::Vals (const std::string & aName) 

{
    return Obs(aName).Vals();
}




template <class TypeEngl,class TGlob>
    std::list<cObserv1Im<TypeEngl> *> & cPackObserv1Im<TypeEngl,TGlob>::LObs()
{
   return mLObs;
}


template <class TypeEngl,class TGlob>
    const std::list<cObserv1Im<TypeEngl> *> & cPackObserv1Im<TypeEngl,TGlob>::LObs() const
{
   return mLObs;
}


template class cPackObserv1Im<cTypeEnglob_Appuis,cPackGlobAppuis>;
template class cPackObserv1Im<cTypeEnglob_Orient,cPackGlobVide>;
template class cPackObserv1Im<cTypeEnglob_Centre,cPackGlobVide>;

/**************************************************/
/*                                                */
/*                  cAppliApero                   */
/*                                                */
/**************************************************/

template <class Type,class TGlob> void cAppliApero::InitBDDPose
                      (
		           std::map<std::string,cPackObserv1Im<Type,TGlob> *> & aDico,
                           const std::list<typename Type::tArg>  & aL
                      )
{
    typedef typename Type::tArg tArg;
    for 
    (
       typename std::list<typename Type::tArg>::const_iterator itB=aL.begin();
       // std::list<tArg>::const_iterator itB=aL.begin();
       itB!=aL.end();
       itB++
    )
    {
        NewSymb(itB->Id());
	aDico[itB->Id()] = new cPackObserv1Im<Type,TGlob>(*this,*itB);
    }
}

template <class Type,class TGlob> void  cAppliApero::CompileObs1Im
                             (
			          std::map<std::string,cPackObserv1Im<Type,TGlob> *> & aDico
                             )
{
   for
   (
       typename std::map<std::string,cPackObserv1Im<Type,TGlob> *>::iterator itD=aDico.begin();
       itD!= aDico.end();
       itD++
   )
   {
       itD->second->Compile();
   }

}


    //
    //    COMPILATION   
    //
void cAppliApero::CompileAppuis()
{
    CompileObs1Im(mDicoAppuis);
}
void cAppliApero::CompileOsbOr()
{
    CompileObs1Im(mDicoOrient);
}
void cAppliApero::CompileObsCentre()
{
    CompileObs1Im(mDicoCentre);
    
}



    //
    //    INITIALISATION   
    //
void cAppliApero::InitBDDAppuis()
{
    InitBDDPose(mDicoAppuis,mParam.BDD_PtsAppuis());
}
void cAppliApero::InitBDDOrient()
{
    InitBDDPose(mDicoOrient,mParam.BDD_Orient());
}


typedef std::map<std::string,cPtTrajecto> tMapTraj;

void cAppliApero::InitBDDCentre()
{
    std::list<cBDD_Centre> aLGlob = mParam.BDD_Centre();
    std::list<cBDD_Centre>  aLClassik;
    for 
    (
         std::list<cBDD_Centre>::const_iterator itBddC = aLGlob.begin();
         itBddC != aLGlob.end();
         itBddC++
    )
    {
         if (itBddC->ByFileTrajecto().IsInit())
         {
             NewSymb(itBddC->Id());
             cFichier_Trajecto * aFT = GetTrajFromString(mDC+itBddC->ByFileTrajecto().Val(),true);

             cPackObserv1Im<cTypeEnglob_Centre,cPackGlobVide> * aPack = 
                            new cPackObserv1Im<cTypeEnglob_Centre,cPackGlobVide>(*this,itBddC->Id());
             mDicoCentre[itBddC->Id()] = aPack;

             cElRegex * anAutomSel=0;
             cElRegex * anAutomRefut=0;
             if (itBddC->PatternFileTrajecto().IsInit())
                 anAutomSel = new cElRegex (itBddC->PatternFileTrajecto().Val(),10);
             if (itBddC->PatternRefutFileTrajecto().IsInit())
                 anAutomRefut = new cElRegex (itBddC->PatternRefutFileTrajecto().Val(),10);
             for (int aKP = 0 ; aKP<int(mVecPose.size()) ; aKP++)
             {
                 const std::string & aNIm =  mVecPose[aKP]->Name();
                 if (
                           ((anAutomSel==0)||(anAutomSel->Match(aNIm)))
                      &&   ((anAutomRefut==0)||(! anAutomRefut->Match(aNIm)))
                    )
                 {
                     std::string anIdIm = mICNM->Assoc1To1(itBddC->KeyAssoc(),aNIm,true);
                     const tMapTraj & aMT = aFT->PtTrajecto();
                     tMapTraj::const_iterator itPT = aMT.find(anIdIm);
                     if (itPT==aMT.end())
                     {
                         std::cout << "For Key " << anIdIm << " In " << itBddC->ByFileTrajecto().Val() << "\n";
                         ELISE_ASSERT(false,"Cannot get key");
                     }
                     else 
                     {
                        cObserv1Im<cTypeEnglob_Centre> * anObs = new cObserv1Im<cTypeEnglob_Centre>(*this,itPT->second.Pt(),aNIm);
                        aPack->Add(anObs);
                     }
                 }

                 // std::cout << aNIm << " "  << anIdIm << " " << itPT->second.Pt() << "\n";
             }
             delete anAutomSel;
             delete anAutomRefut;
             // getchar();
         }
         else
         {
            aLClassik.push_back(*itBddC);
         }
    }

    InitBDDPose(mDicoCentre,aLClassik);
}




/*
const std::list<Appar23> & cAppliApero::Appuis(const std::string& anId,const std::string & aName) 
{
   return GetEntreeNonVide(mDicoAppuis,anId,"BDD Appuis")->Vals(aName);
}
*/


std::list<Appar23>  cAppliApero::GetAppuisDyn(const std::string& anId,const std::string & aName) 
{
  {
     std::map<std::string,tPackAppuis *>::iterator itAp = mDicoAppuis.find(anId);

     if (itAp!=mDicoAppuis.end())
        return itAp->second->Vals(aName);
   }

   {
      cBdAppuisFlottant * aBAF =  BAF_FromName(anId,false,true);
      if (aBAF!=0)
      {
          return aBAF->Appuis32FromCam(aName);
      }
   }



   std::cout << "For Id =" << anId << " Image=" << aName << "\n";
   ELISE_ASSERT(false,"cAppliApero::Appuis cannot get");
   static std::list<Appar23>  aResBid;
   return  aResBid;
   // return GetEntreeNonVide(mDicoAppuis,anId,"BDD Appuis")->Vals(aName);
}




std::list<Appar23>  cAppliApero::AppuisPghrm(const std::string& anId,const std::string & aName,cCalibCam * aCalib)
{
   CamStenope * aCam =   aCalib->PIF().CurPIF();
   std::list<Appar23> aRes;
   std::list<Appar23> aLin=GetAppuisDyn(anId,aName);

   for
   (
       std::list<Appar23>::const_iterator itA=aLin.begin();
       itA != aLin.end();
       itA++
   )
   {
      if (aCalib->IsInZoneU(itA->pim))
      {
          aRes.push_back(aCam->F2toPtDirRayonL3(*itA));
      }
   }

   return  aRes;

   // return aCam->F2toPtDirRayonL3(Appuis(anId,aName));
}

const ElRotation3D & cAppliApero::Orient(const std::string& anId,const std::string & aName)
{
   return GetEntreeNonVide(mDicoOrient,anId,"BDD Orient")->Vals(aName);
}

cObserv1Im<cTypeEnglob_Orient> & cAppliApero::ObsOrient(const std::string& anId,const std::string & aName)
{
   return GetEntreeNonVide(mDicoOrient,anId,"BDD Orient")->Obs(aName);
}

cObserv1Im<cTypeEnglob_Appuis>  & cAppliApero::ObsAppuis(const std::string& anId,const std::string & aName)
{
   return GetEntreeNonVide(mDicoAppuis,anId,"BDD Appuis")->Obs(aName);
}


cObserv1Im<cTypeEnglob_Centre> &  cAppliApero::ObsCentre(const std::string& anId,const std::string & aName)
{
   return GetEntreeNonVide(mDicoCentre,anId,"BDD Centre")->Obs(aName);
}


bool cAppliApero::HasObsCentre(const std::string& anId,const std::string & aName)
{
// std::cout << "AAA  "<<  anId << aName << "\n";
   return GetEntreeNonVide(mDicoCentre,anId,"BDD Centre")->PtrObs(aName)!=0;
}


cPackGlobAppuis * cAppliApero::PtrPackGlobApp(const std::string& anId,bool aSVP)
{
   std::map<std::string,tPackAppuis *>::iterator  anIt =   mDicoAppuis.find(anId);
   if (anIt == mDicoAppuis.end())
   {
      if (aSVP)
         return 0;
      std::cout << "ID = " << anId << "\n";
      ELISE_ASSERT
      (
          false,
          "cAppliApero::PtrPackGlobApp"
      );
   }
   tPackAppuis * aPck =  anIt->second;
   aPck->Glob().AddObsPack(*aPck,aPck->Arg());

   return &(aPck->Glob());
}

cPackGlobAppuis & cAppliApero::PackGlobApp(const std::string& anId)
{
   return *PtrPackGlobApp(anId,false);
}



/**************************************************/
/*                                                */
/*       RAPORT DE COMPENSATION                   */
/*                                                */
/**************************************************/

cRes1OnsAppui::cRes1OnsAppui
(
    int aNum,
    cPoseCam * aPC,
    Pt2dr aPIm, 
    Pt2dr aErIm
)  :
   mNum  (aNum),
   mPC   (aPC),
   mPIm  (aPIm),
   mErIm (aErIm)
{
}

bool operator < (const cRes1OnsAppui & aR1,const cRes1OnsAppui & aR2)
{
   return euclid(aR1.mErIm) < euclid(aR2.mErIm);
}

class cErOfR1
{
  public :
    typedef double tValue;
    double operator() (const cRes1OnsAppui & aR1) const {return  euclid(aR1.mErIm);}
};


void  cAppliApero::DoRapportAppuis
      (
          const cObsAppuis &,
          const cRapportObsAppui& aRAO,
          std::vector<cRes1OnsAppui> & aVEr
      )
{
   cPoseCam * aPC0 =  aVEr[0].mPC;
  
   Pt2di aSzIm = aPC0->Calib()->SzIm();
   Pt2dr aMil  = Pt2dr(aSzIm)/2.0;


   std::sort(aVEr.begin(),aVEr.end());
   FILE * aFP = FopenNN(mDC+aRAO.FichierTxt(),"w","cAppliApero::DoRapportAppuis");

// std::cout << "MTRRRRRRRRRRR = " << mMTRes << "\n";
   if ( mMTRes )
   {
      mMTRes->Show(stdout);
      mMTRes->Show(aFP);
   }

   int aNbC = aVEr.size();

   fprintf(aFP,"========== STAT GLOBALE =========\n");
   fprintf(aFP,"Nb Cible = %d\n",aNbC);
   fprintf(aFP,"Nb Images = %d\n",int(mVecPose.size()));

   int aNbPerc = 20;
   for (int aK=0 ; aK<=aNbPerc ; aK++)
   {
      cErOfR1 aER1;
      double aPerc = (100.0*aK)/aNbPerc;
      fprintf(aFP,"Er %f at Perc %f\n",GenValPercentile(aVEr,aPerc,aER1),aPerc);
   }

   fprintf(aFP,"========== STAT PAR RAYON =========\n");
   fprintf(aFP,"Ray Err Pop\n");
   double aRMax = euclid(aMil);
   int aNBR = 20;
   std::vector<double> aSomEr;
   std::vector<double> aNbEr;
   for (int aK=0 ; aK<aNBR ; aK++)
   {
       aSomEr.push_back(0.0);
       aNbEr.push_back(0.0);
   }
   for (int aKc=0 ; aKc<aNbC ; aKc++)
   {
       const cRes1OnsAppui & aR1 = aVEr[aKc];
       double anEr = euclid(aR1.mErIm);
       double aRay = euclid(aR1.mPIm-aMil);
       int aKr = round_down((aRay*aNBR)/aRMax);
       aKr = ElMax(0,ElMin(aKr,aNBR-1));
       aNbEr[aKr] ++;
       aSomEr[aKr] += anEr;
   }
   for (int aK=0 ; aK<aNBR ; aK++)
   {
      if (aNbEr[aK] == 0)
          fprintf(aFP,"%d ? 0\n",aK);
      else
          fprintf(aFP," %d %f %f\n",aK,aSomEr[aK]/aNbEr[aK],aNbEr[aK]);
   }



   fprintf(aFP,"========== Mesure par Mesure =========\n");
   fprintf(aFP,"Cible Image Er Perc ErX ErY PosX PosY \n");
   for (int aK=0 ; aK<aNbC ; aK++)
   {
        const cRes1OnsAppui & aR1 = aVEr[aK];
        fprintf(aFP,"%d \t %s \t %f \t %f \t %f \t %f \t %f \t %f\n",
                    aR1.mNum,
                    aR1.mPC->Name().c_str(),
                    euclid(aR1.mErIm),
                    ((100.0*aK)/ aNbC),
                    aR1.mErIm.x,
                    aR1.mErIm.y,
                    aR1.mPIm.x,
                    aR1.mPIm.y
               );
   }

   ElFclose(aFP);

   if (aRAO.ROA_FichierImg().IsInit())
   {
        const cROA_FichierImg &  aRFI = aRAO.ROA_FichierImg().Val();
        double aDZ = aRFI.Sz() / ElMax(aSzIm.x,aSzIm.y);
        
        Bitm_Win aBW("Toto.tif",RGB_Gray_GlobPal(),Pt2di(Pt2dr(aSzIm)*aDZ));
        ELISE_COPY(aBW.all_pts(),P8COL::black,aBW.odisc());
        for (int aK=0 ; aK<aNbC ; aK++)
        {
            const cRes1OnsAppui & aR1 = aVEr[aK];
            Pt2dr aPIm = aR1.mPIm * aDZ;
            Pt2dr aEr = aR1.mErIm  *aDZ * aRFI.Exag();
            Pt2dr aLarg(1,1);

            aBW.fill_rect(aPIm-aLarg,aPIm+aLarg,aBW.pdisc()(P8COL::green));

/*
            Elise_colour c = Elise_colour::its
                             (
                                 0.3 + 
                             );
            aBW.draw_seg(aPIm,aPIm+aEr,aBW.pdisc()(P8COL::red));
*/
             int aKP = aR1.mPC->NumInit();
             int aR = 64 + (50+ aKP * 35) % 196;
             int aV = 64 + (30+ aKP * 50) % 196;
             int aB = 64 + (aKP * 73) % 196;

             if (! aRAO.ColPerPose().Val())
             {
                 bool IsOut = euclid(aR1.mErIm) > aRAO.SeuilColOut().Val();
                 aR = (IsOut) ? 0 : 255;
                 aV = 0;
                 aB = (IsOut) ? 255 : 0;
             }

            aBW.draw_seg(aPIm,aPIm+aEr,aBW.prgb()(aR,aV,aB));
            // std::cout << aR << " " << aV << " " << aB << "\n";
            //aBW.draw_seg(aPIm,aPIm+aEr,aBW.prgb()(1,0,0));

        }
        std::string aNameRes = mDC+aRFI.Name();
        aBW.make_tif(aNameRes.c_str());

   }
}


/******************************************************/
/*                                                    */
/*            cCompFilterProj3D                       */
/*                                                    */
/******************************************************/

cCompFilterProj3D::cCompFilterProj3D
(
   cAppliApero &           anAppli,
   const cFilterProj3D &  aFilter
) :
    mAppli  (anAppli),
    mFilter (aFilter)
{
   std::vector<cPoseCam *> aVC = mAppli.ListPoseOfPattern(aFilter.PatternSel());
   for (int aKP=0 ; aKP<int(aVC.size()) ; aKP++)
       AddPose(aVC[aKP]);

}


bool  cCompFilterProj3D::InFiltre(const Pt3dr & aPTer) const
{
   for (int aKP=0 ; aKP<int(mCams.size()) ; aKP++)
   {
       const CamStenope *  aCS = mCams[aKP]->CF()->CameraCourante();
       Pt2dr aPIm = aCS->R3toF2(aPTer);
       if (! mVTMasq[aKP]->get(round_ni(aPIm),0))
          return false;

   }
   return true;
}

void cCompFilterProj3D::AddPose(cPoseCam *aCam)
{
// std::cout << "FFF3D " << aCam->Name() << "\n"; getchar();
   mCams.push_back(aCam);
   std::string aNM  =   mAppli.DC()
                      + mAppli.ICNM()->Assoc1To2
                        (
                            mFilter.KeyCalculMasq(),
                            aCam->Name(),
                            mFilter.AttrSup(),
                            true
                        );
   Tiff_Im aTF = Tiff_Im::StdConvGen(aNM,1,true,false);
   Pt2di aSz = aTF.sz();
   mVMasq.push_back (new Im2D_Bits<1>(aSz.x,aSz.y));
   ELISE_COPY(aTF.all_pts(),aTF.in()!=0,mVMasq.back()->out());
   mVTMasq.push_back(new TIm2DBits<1>(*mVMasq.back()));
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
