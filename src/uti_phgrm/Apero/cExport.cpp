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



void cAppliApero::ExportVisuConfigPose(const cExportVisuConfigGrpPose & anEVCGP)
{
   FILE * aFP = FopenNN(DC()+anEVCGP.NameFile(),"w","ExportVisuConfigPose");
   for 
   (
       std::list<std::string>::const_iterator itS=anEVCGP.PatternSel().begin();
       itS!=anEVCGP.PatternSel().end();
       itS++
   )
   {
       fprintf(aFP,"\n\n   ============================================== \n\n");
       std::vector<cPoseCam *>  aVC  = ListPoseOfPattern(*itS);
       Pt3dr aCdg (0,0,0);
       double aSPds = 0;
       for ( int aK=0 ; aK<int(aVC.size()) ; aK++)
       {
            double aPds = 1.0;
            aCdg = aCdg  +aVC[aK]->CurCam()->PseudoOpticalCenter()*aPds;
            aSPds += aPds;
       }
       aCdg = aCdg/ aSPds;
       for ( int aK=0 ; aK<int(aVC.size()) ; aK++)
       {
           fprintf(aFP,"  ------------------\n");
           fprintf(aFP,"       Pose=%s\n",aVC[aK]->Name().c_str());
           Pt3dr aC = aVC[aK]->CurCam()->PseudoOpticalCenter();
           fprintf(aFP,"       DCDG=%f\n",euclid(aC-aCdg));
       }
   }


   ElFclose(aFP);

}

void  cAppliApero::SauvDataGrid(Im2D_REAL8 anIm,const std::string & aName)
{
     std::string aFullName = DC() + aName + ".dat";
     ELISE_fp  aFp(aFullName.c_str(),ELISE_fp::WRITE);
     aFp.write(anIm.data_lin(),sizeof(double),anIm.tx()*anIm.ty());
     aFp.close();
}



void cAppliApero::ExportCalib(const cExportCalib & anEC)
{
   cElRegex anAutom(anEC.PatternSel().Val(),20);

   for (tDiCal::const_iterator itC=mDicoCalib.begin(); itC!=mDicoCalib.end(); itC++)
   {
       cCalibCam * aCC= itC->second;
       const std::string & aNC = aCC->CCI().Name();
       if (anAutom.Match(aNC))
       {

	   std::string Engl="ExportAPERO";
	   std::string aNXml =   mOutputDirectory
                               + (   (anEC.KeyIsName().Val()) ? 
                                     anEC.KeyAssoc() : 
                                     mICNM->Assoc1To1(anEC.KeyAssoc(),aNC,true)
                                 );


           aNXml = AddPrePost(aNXml,anEC.Prefix().Val(),anEC.Postfix().Val());
           ELISE_fp::MkDirRec(aNXml);
	   /*
           std::string aNXml = 
	           DC() 
		 + anEC.Directory().Val() 
		 +  MatchAndReplace(anAutom,aNC,anEC.NameRes());
           */
           // const CamStenope * aCS = aCC->PIF().CurPIF();
           CamStenope * aCS = aCC->PIF().DupCurPIF();
           aCS->UnNormalize();
	   Pt2di aSzIm = aCC->SzIm();
           if (anEC.ExportAsNewGrid().IsInit())
           {
               const cExportAsNewGrid & anEANG = anEC.ExportAsNewGrid().Val();
               double aR = anEANG.RayonInv().Val();
               if (anEANG.RayonInvRelFE().IsInit() && aCS->IsForteDist())
               {
                   aR = (euclid(aSzIm)/2.0) * anEANG.RayonInvRelFE().Val();
               }
               if (aCC->HasRayonMax())
               {
                  if (aR<0)
                      aR = aCC->RayonMax();
                  else
                     aR = ElMin(aR,aCC->RayonMax());
               }
               aCS = cCamStenopeGrid::Alloc(aR,*aCS,anEANG.Step());
           }
           if (aCC->HasRayonMax())
           {
               double aR = aCC->RayonMax();
               aCS->SetRayonUtile(aR,30);    
           }
	   cCalibrationInternConique anOC = aCS->ExportCalibInterne2XmlStruct(aSzIm);
	   MakeFileXML(anOC,aNXml,Engl);

           aCC->Export(aNXml);


           if (anEC.ExportAsGrid().IsInit()  && anEC.ExportAsGrid().Val().DoExport().Val())
           { 
               cExportAsGrid anEAG = anEC.ExportAsGrid().Val();
               cElXMLFileIn aFileXml(DC()+anEAG.Name()+"_MetaDonnees.xml");
               ElDistRadiale_PolynImpair * aDR = aCS->Dist().DRADPol(true);

               double aFoc = aCS->Focale();
               Pt2dr  aPP =  aCS->PP();


               ELISE_ASSERT(!aCS->DistIsC2M(),"Usage obsolete de distorsion Cam->Monde");

               std::string aNameAux = anEAG.XML_Supl().IsInit()?DC()+anEAG.XML_Supl().Val():"";

               cCS_MapIm2PlanProj aMap(aCS);
               Pt2dr aRab = anEAG.RabPt().Val();
               Pt2dr aStep = anEAG.Step().Val();
               cDbleGrid aGr
                         (
                              false, // P0P1 Direct non par defaut M->C
                              true,
                              Pt2dr(0,0)-aRab,
                              Pt2dr(aCS->Sz())+aRab,
                              aStep,aMap,
                              anEAG.Name()
                        );

               aFileXml.SensorPutDbleGrid
               (
                   aCS->Sz(),
                   anEAG.XML_Autonome().Val(),
                   aGr,
                   0, //  A Priori plus utile aNameThom.c_str(),
                   (aNameAux=="") ? 0 :  aNameAux.c_str(),
                   aDR,
                   &aPP,
                   &aFoc
                );
              
                if (! anEAG.XML_Autonome().Val())
                {
                   SauvDataGrid(aGr.GrDir().DataGridX(),aGr.GrDir().NameX());
                   SauvDataGrid(aGr.GrDir().DataGridY(),aGr.GrDir().NameY());
                   SauvDataGrid(aGr.GrInv().DataGridX(),aGr.GrInv().NameX());
                   SauvDataGrid(aGr.GrInv().DataGridY(),aGr.GrInv().NameY());
                }
           }
       }
   }
}


void cAppliApero::ExportSauvAutom()
{
   std::string aStrS =  mParam.SauvAutom().ValWithDef("");
   std::string aPref = "";


/*
if (MPD_MM())
{
   std::cout << "ExportSauvAutom [" << aStrS << "]\n";
   getchar();
}
*/
   
   if (aStrS=="") 
   {
      if (! mParam.SauvAutomBasic().Val())
         return;
      aPref = MMTemporaryDirectory();
      aStrS ="Autom";
   }
   std::string  aStrSSansMinus = "-Sauv-" + aStrS + "-" + ToString(mNumSauvAuto);
   aStrS = "-" + aStrSSansMinus;

   
   cExportPose anEP;
   // anEP.KeyAssoc() = "NKS-Assoc-Im2Orient@" + aStrS;
   anEP.KeyAssoc() = "NKS-SauvAutom-Assoc-Im2Orient@" + aStrS;
   anEP.AddCalib().SetVal(true);
   anEP.FileExtern().SetVal(std::string("NKS-Assoc-FromFocMm@Ori")+aStrS+ELISE_CAR_DIR+"AutoCal@.xml");
   anEP.FileExternIsKey().SetVal(true);
   anEP.CalcKeyFromCalib().SetVal(false);
   anEP.RelativeNameFE().SetVal(true);
   anEP.ModeAngulaire().SetVal(false);
   anEP.PatternSel().SetVal(".*");

   anEP.StdNameMMDir().SetVal(aStrSSansMinus);

   anEP.NbVerif().SetVal(10);
   anEP.ShowWhenVerif().SetVal(true);
   anEP.TolWhenVerif().SetVal(1e-3);

   ExportPose(anEP,aPref);
   

   cExportCalib anEC;
   anEC.PatternSel().SetVal(".*");
   anEC.KeyAssoc() = "NKS-Assoc-FromKeyCal@"+aPref+"Ori"+aStrS+"/AutoCal@.xml";
   anEC.Prefix().SetVal("");
   anEC.Postfix().SetVal("");
   anEC.KeyIsName().SetVal(false);
   anEC.ExportAsGrid().SetNoInit();
   anEC.ExportAsNewGrid().SetNoInit();
   ExportCalib(anEC);

   mNumSauvAuto++;
}


void cAppliApero::ExportPose(const cExportPose & anEP,const std::string & aPref)
{
   cSetName *  anAutoSel = mICNM->KeyOrPatSelector(anEP.PatternSel());


   cChSysCo * aChCo = 0;
   if (anEP.ChC().IsInit())
   {
      aChCo = cChSysCo::Alloc(anEP.ChC().Val(),mDC);
   }

   // for (tDiPo::const_iterator itP=mDicoPose.begin(); itP!=mDicoPose.end(); itP++)
   for (tDiPoGen::const_iterator itP=mDicoGenPose.begin(); itP!=mDicoGenPose.end(); itP++)
   {
       cGenPoseCam * aGP =  itP->second;
       const std::string & aNP = aGP->Name();
       cPoseCam * aPC= aGP->DownCastPoseCamSVP() ;
       std::string aNXml = mOutputDirectory +  aPref + mICNM->Assoc1To1(anEP.KeyAssoc(),aNP,true);
       if (anAutoSel->IsSetIn(aNP))
       {
          if (aPC==0)
          {
              static cElRegex anExpr (".*Ori-(.*)/Orientation-.*\\.xml",10);

              if (  anEP.StdNameMMDir().IsInit())
              {
                // MakeFileXML(anEP,"toto.xml");
                 aGP->GenCurCam()->Save2XmlStdMMName(0,"",aPref+anEP.StdNameMMDir().Val());
              }
              else if (anExpr.Match(aNXml) && anExpr.Replace(aNXml))
              {
                 aGP->GenCurCam()->Save2XmlStdMMName(mICNM,aPref+anExpr.KIemeExprPar(1),aNP);
              }
              else
              {
                 ELISE_ASSERT(false,"Cannot get StdNameMMDir in cAppliApero::ExportPose");
              }
          }
          else if (anAutoSel->IsSetIn(aNP) && aPC->RotIsInit())
          {
              bool aMM = !anEP.ModeAngulaire().Val();
              double aZ= aPC->AltiSol();
              double aP= aPC->Profondeur();
	      std::string Engl="ExportAPERO";

              const CamStenope * aCS = aPC->CurCam();

	      if (aPC->PMoyIsInit())
	      {
                 Pt3dr aPM  = aPC->GetPMoy();
	         aZ = aPM.z;
	         aP =  aPC->ProfMoyHarmonik();
	      }


              if (anEP.ChC().IsInit())
              {
                  // ELISE_ASSERT(false,"CHC in Apero, inhibed : use ad-hoc command\n");
                 // On modifie, donc on travaille sur un dupl
                   CamStenope *aCS2 = aPC->DupCurCam();
                   aCS2->UnNormalize();
                   aCS2->SetProfondeur(aP);
                   std::vector<ElCamera*> aVC;
                   aVC.push_back(aCS2);
                   aChCo->ChangCoordCamera(aVC,anEP.ChCForceRot().Val());
                   aCS = aCS2;
                   aZ = aCS2->GetAltiSol();
              }


              int aNbV = anEP.NbVerif().Val();
              if (mMapMaskHom && (! aPC->HasMasqHom()) && (mParam.SauvePMoyenOnlyWithMasq().Val()))
              {
                  aZ= 0;
                  aP= 0;
                  aNbV=0;
              }


              ELISE_fp::MkDirRec(aNXml);
	      if (anEP.AddCalib().Val())
	      {
                  CamStenope * aCS2 = aPC->CamF()->PIF().DupCurPIF();
	          Pt2di aSzIm = aPC->Calib()->SzIm();
                  if (anEP.ExportAsNewGrid().IsInit())
                  {
                     cExportAsNewGrid anEG = anEP.ExportAsNewGrid().Val();
                     double aR =  anEG.RayonInv().Val();
                     if (aR<0) 
                        aR = 1.0;
                     aR = (euclid(aSzIm)/2.0) * aR;
                     aCS2 = cCamStenopeGrid::Alloc(aR,*aCS2,anEG.Step());
                  }
// GRID
                  aCS2->SetOrientation(aCS->Orient());
                  aCS2->SetTime(aPC->Time());
                  aCS2->UnNormalize();
                  const char * aNAux = 0;
                  const Pt3di *aPVerifDet=0;
                   if (anEP.VerifDeterm().IsInit())
                      aPVerifDet = & anEP.VerifDeterm().Val();
                  //const char * aNAux = aNXml.c_str();

	          cOrientationConique anOC = aCS2->ExportCalibGlob(aSzIm,aZ,aP,aNbV,aMM,aNAux,aPVerifDet);

                  if (anEP.Force2ObsOnC().IsInit())
                  {
                     const cForce2ObsOnC & aFOC = anEP.Force2ObsOnC().Val();
                     if (aPC->HasObsOnCentre())
                     {
                          anOC.Externe().Centre() = aPC->ObsCentre();
                     }
                     else
                     {
                        if (!aFOC.WhenExist().Val())
                        {
                             std::cout << "For camera " << aPC->Name() << "\n";
                             ELISE_ASSERT(false,"Camera has no center in Force2ObsOnC");
                        }
                     }
                  }


                  if (aPC->FidExist())
                  {
                       anOC.OrIntImaM2C().SetVal(El2Xml(aPC->OrIntM2C())) ;
                  }
	          if (aNbV)
	          {
	               anOC.Verif().Val().ShowMes().SetVal(anEP.ShowWhenVerif().Val());
	               anOC.Verif().Val().Tol() = anEP.TolWhenVerif().Val();
	          }
                  if (anEP.FileExtern().IsInit()) 
                  {
                      std::string aName = anEP.FileExtern().Val();
                      if ( (anEP.FileExternIsKey().Val()) || (anEP.CalcKeyFromCalib().Val()))
                      {

                         std::string aNameIn = aNP;
                         if (anEP.CalcKeyFromCalib().Val())
                         {
                             aNameIn=  aPC->Calib()->KeyId();
                         }
                         aName= mICNM->Assoc1To1(aName,aNameIn,true);
                      }
                      std::string aNFE = aName;
                      if (! anEP.RelativeNameFE().Val())
                           aNFE = DC()+aNFE;
                      anOC.RelativeNameFI().SetVal(anEP.RelativeNameFE().Val());
                      anOC.Interne().SetNoInit();
                      anOC.FileInterne().SetVal(aNFE);
	              MakeFileXML(anOC,aNXml,Engl);
                  }
                  else
                  {

	             MakeFileXML(anOC,aNXml,Engl);
                  }

                  if (0) // VERIFICATION
                  {
	             cOrientationConique aOCBIS = StdGetObjFromFile<cOrientationConique>
		                               (
					           aNXml,
						   StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
						   "OrientationConique",
						   "OrientationConique"
					       );
                     cOrientationExterneRigide anOER = aOCBIS.Externe();
                     ElRotation3D aR2 = Std_RAff_C2M(anOER,anOER.KnownConv().Val());
		     ElRotation3D aR0 = aCS->Orient().inv();
		     std::cout << aR0.ImAff(Pt3dr(0,0,0)) << aR2.ImAff(Pt3dr(0,0,0)) << "\n";
		     std::cout << aR0.IRecVect(Pt3dr(1,0,0)) << aR2.IRecVect(Pt3dr(1,0,0)) << "\n";
		     std::cout << aR0.IRecVect(Pt3dr(0,1,0)) << aR2.IRecVect(Pt3dr(0,1,0)) << "\n";
		     std::cout << aR0.IRecVect(Pt3dr(0,0,1)) << aR2.IRecVect(Pt3dr(0,0,1)) << "\n";
		     getchar();
	          }
	          // delete aCS;

	      }
	      else
              {
                  XML_SauvFile(aPC->CamF()->CurRot(),aNXml,Engl,aZ,aP,aMM);
              }
          }
       }
   }
}


void cAppliApero::ExportFlottant(const cExportPtsFlottant & anEPF)
{
   cElRegex anAutom(anEPF.PatternSel().Val(),10);
   for 
   (
       std::map<std::string,cBdAppuisFlottant *>::const_iterator itB = mDicPF.begin();
       itB != mDicPF.end();
       itB++
   )
   {
      if (anAutom.Match(itB->first))
      {
         itB->second->ExportFlottant(anEPF);
      }
   }
}

void cAppliApero::ExportAttrPose(const cExportAttrPose & anEAP)
{
   cElRegex anAutom(anEAP.PatternApply(),20);
   std::cout << "-------------------------------------\n";
   for (tDiPo::const_iterator itP=mDicoPose.begin(); itP!=mDicoPose.end(); itP++)
   {
       cPoseCam * aPC= itP->second;
       const std::string & aNP = aPC->Name();
       if (anAutom.Match(aNP))
       {
          std::cout << "Exp Attr : " << aNP << "\n";
          //  cElRegex anAutoThis(aNP,5);
          cSetName *  anAutoThis = mICNM->KeyOrPatSelector(aNP); // (aNP,5);
          cExportApero2MM anEAM;
          if (anEAP.ExportDirVerticaleLocale().IsInit())
          {
              const char * anAttr = 0;
              if (anEAP.AttrSup().IsInit())
                  anAttr = anEAP.AttrSup().Val().c_str();
              cElPlan3D aPlan = EstimPlan(anEAP.ExportDirVerticaleLocale().Val(),*anAutoThis,anAttr);
              Pt3dr aDirVisee = aPlan.Norm();
              Pt3dr aVCentreP0 = aPlan.P0()-aPC->CurCentre();
              if (scal(aDirVisee,aVCentreP0)<0)
                 aDirVisee = -aDirVisee;
              double aProf = scal(aDirVisee,aVCentreP0);
              anEAM.DirVertLoc().SetVal(aDirVisee);
              anEAM.ProfInVertLoc().SetVal(aProf);
          }
// std::cout <<  mDC+mICNM->Assoc1To1(anEAP.KeyAssoc(),aNP,true)  << "\n";
          MakeFileXML
          (
              anEAM,
              mDC+mICNM->Assoc1To2(anEAP.KeyAssoc(),aNP,anEAP.AttrSup().ValWithDef(""),true)
          );
       }
   }
}

void cAppliApero::ExportOrthoCyl
     (
           const cExportRepereLoc & anERL,
           const cExportOrthoCyl & anEOC,
           const cRepereCartesien & aCRCInit
     )
{

    cSetName * anAutomAxe =  mICNM->KeyOrPatSelector(anEOC.PatternEstimAxe());
    std::vector<Pt3dr> aVCentre;
    for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
    {
        if (anAutomAxe->IsSetIn(mVecPose[aKP]->Name()))
        {
              aVCentre.push_back(mVecPose[aKP]->CurCam()->PseudoOpticalCenter());
        }
    }


    ElSeg3D aS1 = ElSeg3D::CombinL1(aVCentre);
    ElSeg3D aS2 = ElSeg3D::CreateL2(aVCentre);

// std::cout << "sssssssssssS1 " << aS1.P0() << aS1.TgNormee() << "\n";
// std::cout << "sssssssssssS2 " << aS2.P0() << aS2.TgNormee() << "\n";

    ElSeg3D aSeg = anEOC.L2EstimAxe().Val() ? aS2 : aS1;

    cRepereCartesien aCRC = aCRCInit;

    Pt3dr & aP0 = aCRC.Ori();
    cElPlan3D  aPlan(aP0,aP0+aCRC.Ox(),aP0+aCRC.Oy());
    aP0 = aPlan.Proj(aSeg.P0());

    cProjOrthoCylindrique aPOC (cChCoCart::Xml2El(aCRC),aSeg,anEOC.AngulCorr());

    cXmlOneSurfaceAnalytique OneSurf;

    OneSurf.Id() = "TheSurf";
    OneSurf.VueDeLExterieur() = aPOC.VueDeLext();
    OneSurf.XmlDescriptionAnalytique() = aPOC.Xml();
    cXmlModeleSurfaceComplexe aCplxSur;
    aCplxSur.XmlOneSurfaceAnalytique().push_back(OneSurf);



    MakeFileXML(aCplxSur,mDC+anERL.NameRepere());
}

void cAppliApero::ExportRepereLoc(const cExportRepereLoc & anERL)
{
    cSetName * anAutoPlan = mICNM->KeyOrPatSelector(anERL.PatternEstimPl());

    cElPlan3D aPlan = EstimPlan(anERL.EstimPlanHor(),*anAutoPlan,0);

    ElRotation3D aRP = aPlan.CoordPlan2Euclid();
    Pt3dr anO = aRP.ImAff(Pt3dr(0,0,0));
    Pt3dr aVX = aRP.ImVect(Pt3dr(1,0,0));
    Pt3dr aVY = aRP.ImVect(Pt3dr(0,1,0));
    Pt3dr aVZ = aRP.ImVect(Pt3dr(0,0,1));


    std::vector<cPoseCam *> aVP = mVecPose;
    std::string aNameP1P2 = anERL.ImP1P2().ValWithDef(anERL.PatternEstimPl());
    if (anERL.P1P2Hor().Val())
    {
           Pt3dr aNormPl = vunit(Pt3dr(aVZ.x,aVZ.y,0));  
           Pt3dr aHorInPl (-aNormPl.y,aNormPl.x,0);

           Pt3dr aVertInPl = vunit(aVZ ^ aHorInPl);

           aVX = aHorInPl;
           aVY = aVertInPl;
           if (anERL.P1P2HorYVert().Val())
           {
              aVY = Pt3dr(0,0,1);
              aVZ = aVX ^ aVY;
           }
    }
    else if (aNameP1P2 != "NoP1P2")
    {
       std::string aNameP1 = aNameP1P2;
       std::string aNameP2 = aNameP1P2;
       std::string aNameOri = anERL.NameImOri().ValWithDef(aNameP1);

       Pt2dr anAxe = vunit(anERL.AxeDef().Val()).inv();
       Pt2dr aP1 = anERL.P1();
       Pt2dr aP2 = anERL.P2();
       Pt2dr anOriPl = anERL.Origine().ValWithDef(aP1);

       if (IsPostfixed(aNameP1P2) && (StdPostfix(aNameP1P2)=="xml"))
       {
            cSetOfMesureAppuisFlottants aSMAF = StdGetMAF(aNameP1P2);
            cAperoPointeMono  aPt1 = CreatePointeMono(aSMAF,"Line1");
            cAperoPointeMono  aPt2 = CreatePointeMono(aSMAF,"Line2");
            aNameP1  = aPt1.Im();
            aP1      = aPt1.Pt();
            aNameP2  = aPt2.Im();
            aP2      = aPt2.Pt();

            cAperoPointeMono  aPtOri;
            aPtOri.Im() = anERL.NameImOri().ValWithDef(aNameP1);
            aPtOri.Pt() =  aP1;
            aPtOri =  CreatePointeMono(aSMAF,"Origine",&aPtOri);


            aNameOri =  aPtOri.Im();
            anOriPl = aPtOri.Pt();
       }

       cPoseCam * aPose1 = PoseFromName  (aNameP1);
       cPoseCam * aPose2 = PoseFromName  (aNameP2);
       cPoseCam * aPoseOri = PoseFromName  (aNameOri);
       const CamStenope * aCS1 =  aPose1->CurCam();
       const CamStenope * aCS2 =  aPose2->CurCam();
       const CamStenope * aCSOri =  aPoseOri->CurCam();




       Pt3dr aQ1 = aCS1->PtFromPlanAndIm(aPlan,aP1);
       Pt3dr aQ2 = aCS2->PtFromPlanAndIm(aPlan,aP2);
       anO = aCSOri->PtFromPlanAndIm(aPlan,anOriPl);

       aVX = vunit(aQ2-aQ1) ;
       aVZ = aPlan.Norm();
       AjustNormalSortante(true,aVZ,aCS1,aP1);

       aVY = vunit(aVZ ^ aVX);
       aVX = aVY ^aVZ ;   // Sans doute inutile, mais bon  ....

       aVX = vunit(aVX*anAxe.x + aVY * anAxe.y);
       aVY = vunit(aVZ ^ aVX);
       aVX = aVY ^aVZ ;   // Sans doute inutile, mais bon  ....

       aVP.clear();
       aVP.push_back(aPose1);
    }

    double aSomZ = 0;

    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
       const CamStenope * aCS =  aVP[aKP]->CurCam();

       aSomZ += scal(aVZ,aCS->PseudoOpticalCenter()-anO);
       
    }

    if (aSomZ < 0)
    {
            aVY = - aVY;
            aVZ = - aVZ;
    }


    cRepereCartesien aRC;
    aRC.Ori()= anO;
    aRC.Ox() = aVX;
    aRC.Oy() = aVY;
    aRC.Oz() = aVZ;

    if (anERL.ExportOrthoCyl().IsInit() && anERL.ExportOrthoCyl().Val().UseIt().Val())
    {
        ExportOrthoCyl(anERL,anERL.ExportOrthoCyl().Val(),aRC);
    }
    else
    {
       MakeFileXML(aRC,mDC+anERL.NameRepere(),"RepereLoc");
    }
}
// void cAppliApero::ExportOrthoCyl(const cExportOrthoCyl & anEOC,const cRepereCartesien & aCRC)


void  cAppliApero::Export(const cSectionExport & anEx)
{
    for
    (
         std::list<cExportMesuresFromCarteProf>::const_iterator itM = anEx.ExportMesuresFromCarteProf().begin();
         itM != anEx.ExportMesuresFromCarteProf().end();
         itM++
    )
    {
        ExportMesuresFromCarteProf(*itM);
    }


    for
    (
         std::list<cExportRepereLoc>::const_iterator itA = anEx.ExportRepereLoc().begin();
         itA != anEx.ExportRepereLoc().end();
         itA++
    )
    {
        ExportRepereLoc(*itA);
    }
    
    
    for
    (
         std::list<cExportAttrPose>::const_iterator itA = anEx.ExportAttrPose().begin();
         itA != anEx.ExportAttrPose().end();
         itA++
    )
    {
        ExportAttrPose(*itA);
    }
    
    for
    (
         std::list<cExportCalib>::const_iterator itE = anEx.ExportCalib().begin();
         itE != anEx.ExportCalib().end();
         itE++
    )
    {
        ExportCalib(*itE);
    }

    for
    (
         std::list<cExportPose>::const_iterator itE = anEx.ExportPose().begin();
         itE != anEx.ExportPose().end();
         itE++
    )
    {
        ExportPose(*itE);
    }

    for
    (
         std::list<cExportVisuConfigGrpPose>::const_iterator itE = anEx.ExportVisuConfigGrpPose().begin();
         itE != anEx.ExportVisuConfigGrpPose().end();
         itE++
    )
    {
        ExportVisuConfigPose(*itE);
    }

    for
    (
         std::list<cExportImResiduLiaison>::const_iterator itE = anEx.ExportImResiduLiaison().begin();
         itE != anEx.ExportImResiduLiaison().end();
         itE++
    )
    {
       for(tDiLia::const_iterator itP=mDicoLiaisons.begin();itP!=mDicoLiaisons.end();itP++)
       {
           itP->second->OneExportRL(*itE);
       }
    }

    if (anEx.ExportPtsFlottant().IsInit())
      ExportFlottant(anEx.ExportPtsFlottant().Val());

    for 
    (
         std::list<cExportRedressement>::const_iterator itR = anEx.ExportRedressement().begin();
         itR != anEx.ExportRedressement().end();
         itR++
    )
    {
          ExportRedressement(*itR);
    }


    for 
    (
         std::list<cExportNuage>::const_iterator itEN = anEx.ExportNuage().begin();
         itEN != anEx.ExportNuage().end();
         itEN++
    )
    {
          ExportNuage(*itEN);
    }


    for 
    (
         std::list<cExportBlockCamera>::const_iterator itBC = anEx.ExportBlockCamera().begin();
         itBC != anEx.ExportBlockCamera().end();
         itBC++
    )
    {
          ExportBlockCam(*itBC);
    }

    if (anEx.ExportResiduXml().IsInit())
    {
        MakeFileXML(mXMLExport,mDC+anEx.ExportResiduXml().Val());
        MakeFileXML(mXMLExport,StdPrefix(mDC+anEx.ExportResiduXml().Val()) + ".dmp");
    }

    if (anEx.ChoixImMM().IsInit())
    {
       ExportImMM(anEx.ChoixImMM().Val());
    }
}

void  cAppliApero::InitRapportDetaille(const cTxtRapDetaille & aTRD)
{
    if (aTRD.NameFile() == "")
    {
        mFpRT = 0;
        return;
    }
    mFpRT  = FopenNN(DC()+aTRD.NameFile(),"w","Apero::TxtRapDetaille");


    fprintf(mFpRT,"** 1.0 //   Rapport Apero Version 1.0\n");
    fprintf(mFpRT,"\n*Cameras \n");
    fprintf(mFpRT,"// NomCam Focale\n\n");

    for (tDiCal::const_iterator itD=mDicoCalib.begin(); itD!=mDicoCalib.end() ; itD++)
    {
        cCalibCam * aCal = itD->second;
        CamStenope *  aCS =  (aCal->PIF().DupCurPIF());
        aCS->UnNormalize();
        fprintf(mFpRT,"%s %lf\n",itD->first.c_str(),aCS->Focale());
    }

    fprintf(mFpRT,"\n*Images  // nom_image1 NomCam Sx Sx Sx Ix Iy Iz Jx Jy Jz Kx Ky Kz \n\n");
    for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
    {
        const cPoseCam & aPC = *(mVecPose[aKP]);

        fprintf(mFpRT,"%s %s",aPC.Name().c_str(),aPC.NameCalib().c_str());
       
        const CamStenope & aCS = *(aPC.CurCam());
        ElRotation3D  aR = aCS.Orient().inv();
        Pt3dr aS = aR.ImAff(Pt3dr(0,0,0));

        fprintf(mFpRT," %lf %lf %lf",aS.x,aS.y,aS.z);
        Pt3dr aI =  aR.ImVect(Pt3dr(1,0,0));
        Pt3dr aJ =  aR.ImVect(Pt3dr(0,1,0));
        Pt3dr aK =  aR.ImVect(Pt3dr(0,0,1));
        fprintf(mFpRT," %lf %lf %lf",aI.x,aI.y,aI.z);
        fprintf(mFpRT," %lf %lf %lf",aJ.x,aJ.y,aJ.z);
        fprintf(mFpRT," %lf %lf %lf",aK.x,aK.y,aK.z);

        fprintf(mFpRT,"\n");
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
