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
#include "StdAfx.h"


/************************************************************/
/*                                                          */
/*              cOneAppuisFlottant                          */
/*                                                          */
/************************************************************/

cOneAppuisFlottant::cOneAppuisFlottant
(
   cAppliApero & anAppli,
   const std::string & aName,
   bool  HasGround,
   Pt3dr const& aPt, 
   Pt3dr const& anInc,
   cBdAppuisFlottant & aBDF

) :
    mAppli (anAppli),
    mName  (aName),
    mNupl  (0),
    mMP3TI (0),
    mHasGround (HasGround),
    mPt    (aPt),
    mInc   (anInc),
    mPtRes (mPt)//,
    //mBDF   (aBDF)
{
    // std::cout << "XXXXXXXXXX " << mPt << mName << " HG:: " << mHasGround  << "\n";
    // getchar();
}

bool cOneAppuisFlottant::HasGround() const
{
   return mHasGround;
}




void cOneAppuisFlottant::AddLiaison
     (
          const std::string & aNameIm,
          const cOneMesureAF1I & aMes,
          const Pt2dr & anOffset,
          bool  aModeDr
     )
{
    cPoseCam * aPose = mAppli.PoseFromName(aNameIm);

    for (int aK=0; aK<int(mCams.size()) ; aK++)
    {
        if ((aPose==mCams[aK]) && (aModeDr==mIsDroite[aK]))
	{
           std::string aMessage = "For Im " + aNameIm +  " Pt " +  aMes.NamePt();
	   // std::cout << "For Im " << aNameIm << " Pt " << aMes.NamePt() << "\n";
           cElWarning::AppuisMultipleDefined.AddWarn(aMessage,__LINE__,__FILE__);
/*
	   ELISE_ASSERT
	   (
                 false,
		 "Liaison definie plusieurs fois"
	   );
*/
	}
    }
    Pt2dr aP = aMes.PtIm() + anOffset;
    if (! aModeDr)
       aPose->C2MCompenseMesureOrInt(aP);
    mPts.push_back(aP);
    mCams.push_back(aPose);
    mIsDroite.push_back(aModeDr);
    
}

void cOneAppuisFlottant::DoAMD(cAMD_Interf *)
{
    // std::cout << mName << " " << mMP3TI->IncInterv().NumBlocAlloc() << "\n";

}

void  cOneAppuisFlottant::Compile()
{
    mNupl = new cNupletPtsHomologues(mCams.size(),1.0);
    std::vector<cCameraFormelle *> aVCF;

    for (int aK=0; aK<int(mCams.size()) ; aK++)
    {
        mNupl->PK(aK) = mPts[aK];
	aVCF.push_back(mCams[aK]->CF());
	mPdsIm.push_back(1.0);
        if (mIsDroite[aK])
            mNupl->SetDr(aK);
    }

    mMP3TI = new cManipPt3TerInc(mAppli.SetEq(),0,aVCF);

    if (mAppli.ShowMes())
       std::cout << "NB[" << mName << "]= " << mCams.size() << "\n";


   if (0)
   {
      std::cout << "------------  " << mName << "  --------\n";
      if (mCams.size() >=2)
         TestInterFaisceaux(mCams,*mNupl,10.0,false);
   }
}

int cOneAppuisFlottant::NbMesures() const
{
   return mCams.size();
}

Pt3dr cOneAppuisFlottant::PInter() const
{
   ELISE_ASSERT(mNupl,"cOneAppuisFlottant::PInter : internal inc");
   return InterFaisceaux(mPdsIm,mCams,*mNupl);
}

double cOneAppuisFlottant::AddObs(const cObsAppuisFlottant & anObs,cStatObs & aSO,std::string & aCamMaxErr)
{
   aCamMaxErr = "";

   bool aShowDet = AuMoinsUnMatch(anObs.PtsShowDet(),mName);
   bool ShowUnUsed = aShowDet &&  anObs.ShowUnused().Val();
   if (ShowUnUsed)
       std::cout << "==== ADD Pts " << mName  << " Has Gr " << mHasGround << " Inc " << mInc << "\n";

   double aPdsIm = 1 / ElSquare(anObs.PondIm().EcartMesureIndiv());

   //  cPonderateur aPdrtIm(anObs.PondIm(),mCams.size());

// std::cout << "SHOOOW DET " << aShowDet  << ":" << (*anObs.PtsShowDet().begin())->NameExpr() << ":" << mName << "\n";

   int aNbContrainte = (mInc.x>0)  +  (mInc.y>0) +  (mInc.z>0) ;
   int aNbOK=0;
   for (int aK=0 ; aK<int(mCams.size()) ; aK++)
   {
        if (mCams[aK]->RotIsInit())
        {
	   mPdsIm[aK] = aPdsIm;
           aNbOK++;
           aNbContrainte += mNupl->IsDr(aK) ? 0 : 2 ;
        }
        else
        {
	   mPdsIm[aK] = 0;
        }
   }

   bool HasFaiscPur = (aNbOK >= 2);
   Pt3dr aPFP(0,0,0);
   if (HasFaiscPur)
   {
       aPFP = PInter();
   }


   // A verifier, mais probable que la methode de subsistution degenere
   // si il n'y a que deux  points (Lambda non inversible)
   //En fait, sans doute pas degeneree car attache au point !
   if   (aNbContrainte<3) // (|| ((aNbOK==0) && ((mInc.x <=0)  || (mInc.y <=0) || (mInc.z <=0))))
   {
      if (ShowUnUsed)
      {
          std::cout << "NOT OK 0 FOR " << mName << " NbOK " << aNbOK  << " NbContr " << aNbContrainte << "\n";
      }
      return 0.0;
   }

   if (  (! (mHasGround)) && (aNbOK<2)) return 0.0;
 
   bool aUseAppAsInit = (aNbOK<2) && mHasGround;
   aUseAppAsInit = true;


/*
   const cResiduP3Inc &  aRes = mMP3TI->UsePointLiaisonWithConstr
                                (
                                    -1,
                                    -1,
				    0.0,  // Pds Pl
                                    *mNupl,
                                    mPdsIm,
                                   aSO.AddEq(),
				   // false,
				    mPt,
				    mInc,
				    aUseAppAsInit
				);
*/
   const cResiduP3Inc &  aRes = mMP3TI->UsePointLiaisonGen
                                (
                                   -1,
                                   -1,
				   0.0,  // Pds Pl
                                   *mNupl,
                                   mPdsIm,
                                   aSO.AddEq(),
				   // false,
				    mHasGround ? &mPt  : 0,
				    mHasGround ? &mInc : 0,
				    aUseAppAsInit,
                                    0
				);


// std::cout << "  GGGGGGGGGGGGGGGGGGGgg " << mPt   << mInc << aUseAppAsInit << "\n";

   if (! aRes.mOKRP3I)
   {
      if (ShowUnUsed)
      {
          std::cout << "NOT OK (UPL) FOR " << mName << "\n";
      }
      return 0.0;
   }

   if (aShowDet )
   {
       std::cout  << "--NamePt " ;
       if (mHasGround)
          std::cout <<  mName 
                    << " Ec Estim-Ter " << mPt-aRes.mPTer 
                    << "           Dist =" << euclid(mPt-aRes.mPTer)  
                    << " ground units\n";
       else
           std::cout << "\n" ;
        std::cout << "Inc = "  << mInc << "PdsIm = " <<  mPdsIm  << "\n";

       if (HasFaiscPur)
          std::cout << "    Ecart Estim-Faisceaux " << euclid(aPFP-aRes.mPTer) << "\n";
      
   }

   FILE * aFpRT = mAppli.FpRT() ;
   if (aFpRT)
   {
      fprintf(aFpRT,"*%s %f %f %f %f %f %f\n",mName.c_str(),mPt.x,mPt.y,mPt.z,aRes.mPTer.x,aRes.mPTer.y,aRes.mPTer.z);
   }

   cPonderateur aPdrtIm(anObs.PondIm(),(double)(mCams.size()));

   double anErMax = -1;
   int aKMax = -1;
   double aSEr=0;
   double aSPds = 0;
   for (int aK=0 ; aK<int(mCams.size()) ; aK++)
   {
        if (mCams[aK]->RotIsInit())
        {
            Pt2dr anEc = aRes.mEcIm[aK];
	    double anEr = euclid(anEc);
	    double aPds = 1.0;

            if (aFpRT)
            {
                cPoseCam *  aPC = mCams[aK];
                Pt2dr anEc = aRes.mEcIm[aK];
                Pt2dr aPIm = mNupl->PK(aK);
                const CamStenope * aCS = aPC->CurCam();
                Pt3dr aDir = aCS->F2toDirRayonR3(aPIm);

                fprintf(aFpRT,"%s %f %f %f %f %f %f %f",aPC->Name().c_str(),aPIm.x,aPIm.y,anEc.x,anEc.y,aDir.x,aDir.y,aDir.z);
                fprintf(aFpRT,"\n");


            }
            if (aShowDet)
            {
               if (anObs.DetShow3D().Val())
               {
               }
	       // std::cout << "   " << mCams[aK]->Name() << " Er " << anEr ;
               if (anObs.DetShow3D().Val())
               {


                   std::cout <<  mCams[aK]->Name() << " Ec-Im-Ter " << mCams[aK]->CurCam()->R3toF2(mPt)- mNupl->PK(aK);
                   if (HasFaiscPur) 
                       std::cout << " Ec-Im-Faisceau " << mCams[aK]->CurCam()->R3toF2(aPFP)- mNupl->PK(aK);

                   std::cout << "\n";
/*
                   std::cout << " Proj-F " << mCams[aK]->CurCam()->R3toF2(aRes.mPTer)
                             << " Proj-Ter " << mCams[aK]->CurCam()->R3toF2(mPt)
                             << " mPdsIm Mes Im " << mNupl->PK(aK);
*/
               }
               if (anEr>anObs.NivAlerteDetail().Val())
               {
                   std::cout << "HORS TOL \n" ;
                   getchar();
               }
            }
	    aSEr +=  anEr*aPds;
	    aSPds += aPds;

	    if (anEr>anErMax)
	    {
                anErMax = anEr;
	        aKMax = aK;
	    }

	   mPdsIm[aK] = aPdrtIm.PdsOfError(anEr);
        }
        else
        {
           mPdsIm[aK] = 0.0;
        }
   }
/*
   const cResiduP3Inc & aRes2 = mMP3TI->UsePointLiaisonWithConstr
                                (
                                   -1,
                                   -1,
                                   0.0,
                                   *mNupl,
                                   mPdsIm,
                                   // true,
                                   aSO.AddEq(),
                                   mPt,
                                   mInc,
                                   aUseAppAsInit
                                );
*/
   mPtRes = aRes.mPTer;
   aSO.AddSEP(aRes.mSomPondEr);

   if (aShowDet)
   {
        // std::cout <<  aSO.AddEq() << " " << "\n";
        // std::cout << " Ter " << mPt  << " Delta ;  Faiscx-Ter " << aRes.mPTer -mPt  << "\n";
   }

   if (anObs.ShowSom().Val())
      std::cout <<  "      ErrMoy " << aSEr/aSPds << " pixels " << " SP=" << aSPds << " \n";
   if (anObs.ShowMax().Val())
   {
      std::cout <<  "     ErrMax = " << anErMax 
                << " For I="<< mCams[aKMax]->Name() 
                << ",  C=" << mName    << " pixels \n";
   }
   
   if (aShowDet) std::cout << "  - - - - - - - - - - - \n";
  
   aCamMaxErr =  mCams[aKMax]->Name();
   return anErMax;
}

const Pt3dr &  cOneAppuisFlottant::PtRes() const
{
   return mPtRes;
}

const std::string & cOneAppuisFlottant::Name() const
{
  return mName;
}

const Pt3dr &  cOneAppuisFlottant::PtInit() const
{
   return mPt;
}


const Pt3dr &  cOneAppuisFlottant::PInc() const
{
   return mInc;
}


void cOneAppuisFlottant::Add2Appar32(std::list<Appar23> & aL,const std::string & aNameCam,int & aNum)
{
// std::cout << "Add2Appar32 " << mName << " " <<  HasGround() << " " << mCams.size() << "\n";
    if (! HasGround()) return;
    for (int aK=0 ; aK<int(mCams.size()) ; aK++)
    {
          if (aNameCam==(mCams[aK]->Name()))
          {
              aL.push_back(Appar23(mPts[aK],mPt,aNum));
              aNum++;
          }
    }
}

/************************************************************/
/*                                                          */
/*              cBdAppuisFlottant                           */
/*                                                          */
/************************************************************/

cBdAppuisFlottant::cBdAppuisFlottant(cAppliApero & anAppli) :
   mAppli (anAppli)
{
}


void cBdAppuisFlottant::DoAMD(cAMD_Interf * anAMDI)
{
   for
   (
       std::map<std::string,cOneAppuisFlottant *>::iterator it= mApps.begin();
       it != mApps.end();
       it++
   )
   {
       it->second->DoAMD(anAMDI);
   }
}


void cBdAppuisFlottant::AddAFDico(const cDicoAppuisFlottant & aDAF)
{
   for 
   (
        std::list<cOneAppuisDAF>::const_iterator itP=aDAF.OneAppuisDAF().begin();
	itP!=aDAF.OneAppuisDAF().end();
	itP++
   )
   {
       cOneAppuisFlottant * &  anApp = mApps[itP->NamePt()];
       if (anApp!=0)
       {
           std::cout  << "For name = " << itP->NamePt() << "\n";
           ELISE_ASSERT(false,"Name conflict in cBdAppuisFlottant::Add")
       }
       anApp = new cOneAppuisFlottant(mAppli,itP->NamePt(),true,itP->Pt(),itP->Incertitude(),*this);
   }
}


void cBdAppuisFlottant::ShowError()
{
    for (int aK=0 ; aK < 4 ; aK++)
    {
       if (aK==0)  std::cout <<  "===============  Point Without GCP ==========================================\n";
       if (aK==1)  std::cout <<  "===============  Point With less than 2 image seizing =======================\n";
       if (aK==2)  std::cout <<  "===============  Point With uncertaincy <0  =================================\n";
       if (aK==3)  std::cout <<  "===============  Point OK                 ===================================\n";
       for
       (
            std::map<std::string,cOneAppuisFlottant *>::const_iterator itF= mApps.begin();
            itF!= mApps.end();
            itF++
       )
       {
             cOneAppuisFlottant * anOAF = itF->second;
             Pt3dr aPInc = anOAF->PInc();
             int aCase =3;
             if (!anOAF->HasGround()) aCase = 0;
             else if (anOAF->NbMesures()<2) aCase = 1;
             else if ((aPInc.x<=0) || (aPInc.y<=0) || (aPInc.z<=0)) aCase = 2;

             if (aCase==aK)
             {
                 std::cout << " " << anOAF->Name() << " " ;
                 if (aK==1) std::cout << " Nb Mes Im " << anOAF->NbMesures() ;
                 if (aK==2) std::cout << " Inc " << aPInc ;
                 std::cout << "\n";
                 
                 
             }


       }
    }
    std::cout <<  "=============================================================================\n";

}

void cBdAppuisFlottant::AddAFLiaison
     (
            const std::string & aNameIm,
            const cOneMesureAF1I & aMes,
            const Pt2dr & anOffset,
            bool  OkNoGr,
            bool  aModeDr
     )
{
    std::map<std::string,cOneAppuisFlottant *>::iterator iT = mApps.find(aMes.NamePt());
    if (iT== mApps.end())
    {
       if (! OkNoGr)
       {
          std::cout << "For Pt=" << aMes.NamePt() << "\n";
          ELISE_ASSERT(false,"Cannot Get point in cBdAppuisFlottant::AddLiaison");
       }
       return;
    }
    cOneAppuisFlottant * anApp = iT->second;

/*
    cOneAppuisFlottant * anApp = mApps[aMes.NamePt()];
    if (anApp==0)
    {
        if (OkNoGr)
        {
               anApp = new cOneAppuisFlottant
                           (
                                mAppli,
                                aMes.NamePt(),
                                false,
                                Pt3dr(0,0,0),
                                Pt3dr(0,0,0),
                                *this
                           );
               mApps[aMes.NamePt()] = anApp;
        }
        else
        {
            std::cout << "For name = " << aMes.NamePt() << "\n";
            ELISE_ASSERT(false,"Cannot Get point in cBdAppuisFlottant::AddLiaison");
        }
    }
*/
    anApp->AddLiaison(aNameIm,aMes,anOffset,aModeDr);
}

void cBdAppuisFlottant::Compile()
{
    for 
    (
       std::map<std::string,cOneAppuisFlottant *>::iterator it1=mApps.begin();
       it1!=mApps.end();
       it1++
    )
    {
       it1->second->Compile();
    }
}

void cBdAppuisFlottant::AddObs(const cObsAppuisFlottant & anObs,cStatObs & aSO)
{
   double anErrMax = -10;
   cOneAppuisFlottant * aPFMax=0;
   std::string aCamMax;
   
    for 
    (
       std::map<std::string,cOneAppuisFlottant *>::iterator it1=mApps.begin();
       it1!=mApps.end();
       it1++
    )
    {
       std::string aCam;
       double anErr = it1->second->AddObs(anObs,aSO,aCam);
       if ((anErr >  anErrMax) && (aCam!=""))
       {
              anErrMax = anErr;
              aPFMax = it1->second;
              aCamMax = aCam;
       }
    }
    if (aPFMax)
    {
           std::cout << "\n   ============================= ERRROR MAX PTS FL ======================\n";
           std::cout <<   "   ||    Value=" << anErrMax << " for Cam=" << aCamMax << " and Pt=" << aPFMax->Name() << "\n";
           std::cout <<   "   ======================================================================\n\n";
    }
}

void cBdAppuisFlottant::ExportFlottant(const cExportPtsFlottant & anEPF)
{
   if (anEPF.NameFileXml().IsInit())
   {
      ELISE_ASSERT(false,"cBdAppuisFlottant::ExportFlottant");
   }


   if (anEPF.NameFileTxt().IsInit())
   {
        std::string aNameF =   mAppli.DC() + anEPF.NameFileTxt().Val();
	FILE * aFP  = ElFopen(aNameF.c_str(),"w");
	std::string aNC = anEPF.TextComplTxt().Val();
        for 
        (
           std::map<std::string,cOneAppuisFlottant *>::iterator it1=mApps.begin();
           it1!=mApps.end();
           it1++
        )
        {
           cOneAppuisFlottant * anOAF = it1->second;
	   std::string aNP = anOAF->Name();
	   Pt3dr aPTer = anOAF->PtRes();
	   std::cout << aNP << " " <<  anOAF->PtRes() 
	             << "    "  << euclid(aPTer-anOAF->PtInit())
	             << "    "  << aPTer-anOAF->PtInit()
		     << "\n";
           fprintf(aFP,"%s %lf %lf %lf %s\n",aNP.c_str(),aPTer.x,aPTer.y,aPTer.z,aNC.c_str());
        }
	ElFclose(aFP);
   }
}


const std::map<std::string,cOneAppuisFlottant *> & cBdAppuisFlottant::Apps() const
{
  return mApps;
}


std::list<Appar23>  cBdAppuisFlottant::Appuis32FromCam(const std::string & aName)
{
   int aCpt=0;
   std::list<Appar23> aRes;

   for 
   (
        std::map<std::string,cOneAppuisFlottant *>::iterator it=mApps.begin();
        it!=mApps.end();
        it++
   )
   {
        it->second->Add2Appar32(aRes,aName,aCpt);
   }

   return aRes;
}


/************************************************************/
/*                                                          */
/*              cAppliApero                                 */
/*                                                          */
/************************************************************/

void cAppliApero::PreCompileAppuisFlottants()
{


   for 
   (
        std::list<cPointFlottantInc>::const_iterator itP=mParam.PointFlottantInc().begin();
	itP != mParam.PointFlottantInc().end();
	itP++
   )
   {
       cBdAppuisFlottant *aBAF = BAF_FromName(itP->Id(),true);
       std::list<std::string>  aLN =  mICNM->StdGetListOfFile(itP->KeySetOrPat());
       for 
       (
             std::list<std::string>::const_iterator itN=aLN.begin();
	     itN != aLN.end();
	     itN++
       )
       {
           cDicoAppuisFlottant aDic = StdGetObjFromFile<cDicoAppuisFlottant>
	                              (
				             mDC+*itN,
					     StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
					     "DicoAppuisFlottant",
					     "DicoAppuisFlottant"
				      );
            ModifDAF(mICNM,aDic,itP->ModifInc());
           aBAF->AddAFDico(aDic);
       }
   }
}

void cAppliApero::InitOneSetObsFlot
     (
           cBdAppuisFlottant * aBAF,
           const cSetOfMesureAppuisFlottants & aSMAF,
           const Pt2dr & anOffset,
           cSetName * aSN,
           bool       OkNoGr
      )
{
   for
   (
       std::list<cMesureAppuiFlottant1Im>::const_iterator itM=aSMAF.MesureAppuiFlottant1Im().begin();
       itM!=aSMAF.MesureAppuiFlottant1Im().end();
       itM++
   )
   {
      if (NamePoseIsKnown(itM->NameIm()))
      {
         for
         (
            std::list<cOneMesureAF1I>::const_iterator it1=itM->OneMesureAF1I().begin();
            it1!=itM->OneMesureAF1I().end();
            it1++
         )
         {
             if (aSN->IsSetIn (it1->NamePt()))
	        aBAF->AddAFLiaison(itM->NameIm(),*it1,anOffset,OkNoGr,false);
         }
      }
   }
}

void cAppliApero::InitOneSetOnsDr
     (
           cBdAppuisFlottant * aBAF,
           const cSetOfMesureSegDr & aSMS,
           const Pt2dr & anOffset,
           cSetName * aSN,
           bool       OkNoGr
      )
{
   for
   (
       std::list<cMesureAppuiSegDr1Im>::const_iterator itIm=aSMS.MesureAppuiSegDr1Im().begin();
       itIm!=aSMS.MesureAppuiSegDr1Im().end();
       itIm++
   )
   {
      std::string aNameIm = itIm->NameIm();
      if (NamePoseIsKnown(aNameIm))
      {

         for
         (
            std::list<cOneMesureSegDr>::const_iterator itMes=itIm->OneMesureSegDr().begin();
            itMes!=itIm->OneMesureSegDr().end();
            itMes++
         )
         {
             cPoseCam * aPose = PoseFromName(aNameIm);
             Pt2dr aP1 = itMes->Pt1Im();
             Pt2dr aP2 = itMes->Pt2Im();
             aPose->C2MCompenseMesureOrInt(aP1);
             aPose->C2MCompenseMesureOrInt(aP2);

/*
             Pt2dr aTgt = vunit(aP1-aP2);
             Pt2dr aNorm = aTgt * Pt2dr(0,1);
             Pt2dr aPolar = Pt2dr::polar(aNorm,0);
             double aRho = scal(aP1,aNorm);
             Pt2dr aPEqDr(aRho,aPolar.y);
*/
             Pt2dr aPEqDr = SegComp(aP1,aP2).ToRhoTeta();
             cOneMesureAF1I aMes;
             aMes.PtIm() = aPEqDr;
             for (std::list<std::string>::const_iterator itPt=itMes->NamePt().begin() ;  itPt!=itMes->NamePt().end() ; itPt++)
             {
                aMes.NamePt() = *itPt;
	        aBAF->AddAFLiaison(itIm->NameIm(),aMes,anOffset,OkNoGr,true);
             }
/*
             std::cout << "aRhoTetaaRhoTeta " <<  aPEqDr << "\n";
             if (aSN->IsSetIn (it1->NamePt()))
	        aBAF->AddAFLiaison(itM->NameIm(),*it1,anOffset,OkNoGr);
*/
         }
      }
   }
}




void cAppliApero::InitHasEqDr()
{
    for 
    (
        std::list<cBDD_ObsAppuisFlottant>::const_iterator itO=mParam.BDD_ObsAppuisFlottant().begin();
	itO!=mParam.BDD_ObsAppuisFlottant().end();
	itO++
    )
    {
         if (itO->KeySetSegDroite().IsInit())
             mHasEqDr = true;
    }
}


void cAppliApero::InitAndCompileBDDObsFlottant()
{
    for 
    (
        std::list<cBDD_ObsAppuisFlottant>::const_iterator itO=mParam.BDD_ObsAppuisFlottant().begin();
	itO!=mParam.BDD_ObsAppuisFlottant().end();
	itO++
    )
    {
         bool OkNoGr = itO->AcceptNoGround().Val();
         cBdAppuisFlottant *aBAF = BAF_FromName(itO->Id(),OkNoGr);
         cSetName * aSN = mICNM->KeyOrPatSelector(itO->NameAppuiSelector()) ;
         if (itO->KeySetOrPat().IsInit())
         {
	      std::list<std::string> aLN = mICNM->StdGetListOfFile(itO->KeySetOrPat().Val());
	      for 
	      (
	           std::list<std::string>::const_iterator itNF=aLN.begin();
	           itNF!=aLN.end();
	           itNF++
	      )
	      {
	          cSetOfMesureAppuisFlottants aSMAF = StdGetMAF(*itNF);
                  InitOneSetObsFlot(aBAF,aSMAF,itO->OffsetIm().Val(),aSN,OkNoGr);
	      }
	 }
         if (itO->KeySetSegDroite().IsInit())
         {
              std::list<std::string> aLS = mICNM->StdGetListOfFile(itO->KeySetSegDroite().Val());
	      for 
	      (
	           std::list<std::string>::const_iterator itS=aLS.begin();
	           itS!=aLS.end();
	           itS++
	      )
	      {
                    cSetOfMesureSegDr aSMS = StdGetFromPCP(mDC+*itS,SetOfMesureSegDr);
                    InitOneSetOnsDr(aBAF,aSMS,itO->OffsetIm().Val(),aSN,OkNoGr);
	      }
         }
  
    }

    for 
    (
       std::map<std::string,cBdAppuisFlottant *>::iterator itB=mDicPF.begin();
       itB!=mDicPF.end();
       itB++
    )
    {
       itB->second->Compile();
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
