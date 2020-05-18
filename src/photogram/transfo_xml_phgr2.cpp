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



void UseRequirement(const std::string & aDir,const cTplValGesInit<cBatchRequirement> & aTplB)
{
    if (! aTplB.IsInit())
       return;

    const cBatchRequirement & aBR = aTplB.Val();

	for 
	(
		std::list<cExeRequired>::const_iterator itE=aBR.ExeRequired().begin();
		itE!=aBR.ExeRequired().end();
		itE++
	)
		launchMake(itE->Make(), itE->Exe());

    for 
    (
        std::list<cFileRequired>::const_iterator itF=aBR.FileRequired().begin();
        itF!=aBR.FileRequired().end();
        itF++
    )
    {
          int aNbMin = itF->NbMin().Val();
          int aNbMax = itF->NbMax().ValWithDef(aNbMin);

          for 
          (
              std::list<std::string>::const_iterator itP=itF->Pattern().begin();
              itP!=itF->Pattern().end();
              itP++
          )
          {
             std::string aDir2,aPat2;
             SplitDirAndFile(aDir2,aPat2,*itP);
 
             aDir2 = aDir + aDir2;

             std::list<std::string>  aL=RegexListFileMatch(aDir2,aPat2,1,false);

             int aNb = (int)aL.size();
             if ((aNb<aNbMin) || (aNb>aNbMax))
             {
                  std::cout << "For Pattern {" << aPat2 << "} Number " 
                            << aNb  << " Intervalle " << aNbMin << " / " << aNbMax << "\n";
                  ELISE_ASSERT(false,"File number required");
             }
          }

    }




}


GenIm::type_el Xml2EL(const eTypeNumerique & aType)
{
   switch (aType)
   {
      case eTN_u_int1 : return GenIm::u_int1;
      case eTN_int1   : return GenIm::int1;
      case eTN_u_int2 : return GenIm::u_int2;
      case eTN_int2   : return GenIm::int2;
      case eTN_int4   : return GenIm::int4;
      case eTN_float   : return GenIm::real4;
      case eTN_double   : return GenIm::real8;
      case eTN_Bits1MSBF   : return GenIm::bits1_msbf;
   }
   ELISE_ASSERT(false,"unknown eTypeNumerique");
   return GenIm::u_int1;
}

Tiff_Im::COMPR_TYPE Xml2EL(const eComprTiff & aType)
{
   switch (aType)
   {
        case eComprTiff_None : return Tiff_Im::No_Compr;
        case eComprTiff_LZW :  return Tiff_Im::LZW_Compr;
        case eComprTiff_FAX4 : return Tiff_Im::Group_4FAX_Compr;
        case eComprTiff_PackBits : return Tiff_Im::PackBits_Compr;
   }
   ELISE_ASSERT(false,"unknown eComprTiff");
   return Tiff_Im::No_Compr;
}



std::string cInterfChantierNameManipulateur::NamePackWithAutoSym
            (
                        const std::string & aKey,
                        const std::string & aName1,
                        const std::string & aName2,
                        bool aSVP
            )
{
   std::string aN12_SsDir = Assoc1To2(aKey,aName1,aName2,true);
   std::string aN12 = mDir+aN12_SsDir;
   if (! ELISE_fp::exist_file(aN12))
   {
      std::string aN21 =mDir+ Assoc1To2(aKey,aName2,aName1,true);
      if (! ELISE_fp::exist_file(aN21))
      {
          if (aSVP)
             return aN12_SsDir;
          std::cout << "For K=" << aKey 
                    << " N1=" << aName1 
                    << " N2=" << aName2 << "\n";
          std::cout << aN12 << "\n";
          std::cout << aN21 << "\n";
          ELISE_ASSERT(false,"Ni fichier homoloque ni symetrique n'existe");
      }
      ElPackHomologue aPack = ElPackHomologue::FromFile(aN21);
      aPack.SelfSwap();
      aPack.StdPutInFile(aN12);
        
   }
   return aN12_SsDir;
}


bool IsActive(const cTplValGesInit<cCmdMappeur> &  aMaper)
{
   return    aMaper.IsInit()
          && aMaper.Val().ActivateCmdMap();
}

bool IsActive(const std::list<cCmdMappeur> &  aMaper)
{
  for 
  (
      std::list<cCmdMappeur>::const_iterator itC=aMaper.begin();
      itC!= aMaper.end();
      itC++
  )
  {
     if (itC->ActivateCmdMap())
        return true;
  }
  return    false;
}







template <class TOut,class TIn> void TXml2EL(TOut & aRes,const TIn & aLIn)
{
    for
    (
        typename TIn::const_iterator itIn = aLIn.begin();
        itIn != aLIn.end();
        itIn++
    )
    {
       aRes.push_back(Xml2EL(*itIn));
    }
}

template <class TOut,class TIn> void TEl2Xml(TOut & aRes,const TIn & aLIn)
{
    for
    (
        typename TIn::const_iterator itIn = aLIn.begin();
        itIn != aLIn.end();
        itIn++
    )
    {
       aRes.push_back(El2Xml(*itIn));
    }
}

    /*************************************************/


Appar23  Xml2EL(const cMesureAppuis & aMA)
{
   return Appar23(aMA.Im(),aMA.Ter(),aMA.Num().ValWithDef(-1));
}

cMesureAppuis  El2Xml(const Appar23 & anAp,int aNum)
{
   cMesureAppuis aRes;
   aRes.Im() = anAp.pim;
   aRes.Ter() = anAp.pter;
   aRes.Num().SetVal(aNum);
   //   if (anAp.NumIsInit()) aRes.Num().SetVal(anAp.GetNum());
   return aRes;
}
 
std::list<Appar23>  Xml2EL(const cListeAppuis1Im & aLA)
{
   std::list<Appar23> aRes;
   TXml2EL(aRes,aLA.Mesures());
   return aRes;
}

cListeAppuis1Im  El2Xml(const std::list<Appar23> & aLAp,const std::string &aNameImage)
{
   cListeAppuis1Im aRes;
   aRes.NameImage().SetVal(aNameImage);
   TEl2Xml(aRes.Mesures(),aLAp);

   int aCpt=0;
   for (auto & aP : aRes.Mesures())
   {
      aP.Num() = aCpt++;
   }

   return aRes;
}



cListeAppuis1Im  El2Xml
                 (
                      const std::list<Appar23> & aLAp,
                      const std::list<int> &     aLInd,
                      const std::string &aNameImage
                 )
{
    cListeAppuis1Im aRes = El2Xml(aLAp,aNameImage);
    ELISE_ASSERT
    (
        aRes.Mesures().size() == aLInd.size(),
        "El2Xml  Indexe-size != appuis-size "
    );

    std::list<int>::const_iterator itN = aLInd.begin();
    for 
    (
        std::list<cMesureAppuis>::iterator itM=aRes.Mesures().begin();
        itM!=aRes.Mesures().end();
        itM++,itN++
    )
    {
        itM->Num().SetVal(*itN);
    }
  

    return aRes;
}


template <class TCont,class TVal> bool IsInIntervalle
                      (
                            const  TCont  & aL,
                            const TVal & aVal,
                            bool  EmptyIsOk
                      )
{
    if (EmptyIsOk && (aL.empty()))
       return true;

    for (typename TCont::const_iterator it=aL.begin(); it!=aL.end() ; it++)
    {
         if ((aVal>=it->Val().x) && (aVal<=it->Val().y))
             return true;
    }
    return false;
}


/*
template <class Type> Type * GetImRemanenteFromFile(const std::string & aName)
{
   static std::map<std::string,Type *> aDic;
   Type * aRes = aDic[aName];

   if (aRes != 0) return aRes;

   Tiff_Im aTF(aName.c_str());
   Pt2di aSz = aTF.sz();

   aRes = new Type(aSz.x,aSz.y);
   ELISE_COPY(aTF.all_pts(),aTF.in(),aRes->out());
   aDic[aName] = aRes;
   return aRes;
}
*/


ElRotation3D  CombinatoireOFPA
              (
                   bool TousDevant,
                   CamStenope & aCam,
                   INT  NbTest,
                   const cListeAppuis1Im & aLA,
                   REAL * Res_Dmin
              )
{
   return aCam.CombinatoireOFPA(TousDevant,NbTest,Xml2EL(aLA),Res_Dmin);
}

bool NameFilter(const std::string & aSubD,cInterfChantierNameManipulateur * aICNM,const cNameFilter & aFilter,const std::string & aName)
{
   std::string anEntete = aICNM->Dir()+ aSubD;
   std::string aFullName = anEntete + aName;

   int aSz = aFilter.SizeMinFile().Val();
   if (aSz>=0)
   {
      if (sizeofile(aFullName.c_str()) < aSz)
         return false;
   }

   if ((aFilter.Min().IsInit())&&(aFilter.Min().Val()>aName))
      return false;

   if ((aFilter.Max().IsInit())&&(aFilter.Max().Val()<aName))
      return false;


   const std::list<Pt2drSubst> & aLFoc = aFilter.FocMm();
   if (! aLFoc.empty())
   {
      
      if (!IsInIntervalle(aLFoc,GetFocalMmDefined(aFullName),true))
      {
            return false;
      }
   }
   


   for 
   (
        std::list<cKeyExistingFile>::const_iterator itKEF=aFilter.KeyExistingFile().begin();
        itKEF!=aFilter.KeyExistingFile().end();
        itKEF++
   )
   {
       bool OKGlob = itKEF->RequireForAll();
       for 
       (
            std::list<std::string>::const_iterator itKA=itKEF->KeyAssoc().begin();
            itKA!=itKEF->KeyAssoc().end();
            itKA++
       )
       {
          std::string aNameF = anEntete + aICNM->Assoc1To1(*itKA,aName,true);
          bool fExists = ELISE_fp::exist_file(aNameF);
// std::cout << "KEY-NF " << aNameF << "\n";
          bool Ok = itKEF->RequireExist() ? fExists : (!fExists);
          if (itKEF->RequireForAll())
             OKGlob = OKGlob && Ok;
          else
             OKGlob = OKGlob || Ok;
       }
   //std::cout << "KEY-NF " << aName << " " << OKGlob << "\n";
       if (!OKGlob) 
          return false;
   }


   if (aFilter.KeyLocalisation().IsInit())
   {
       const cFilterLocalisation & aKLoc = aFilter.KeyLocalisation().Val();
       std::string aNameCam = anEntete + aICNM->Assoc1To1(aKLoc.KeyAssocOrient(),aName,true);
       ElCamera * aCam = Cam_Gen_From_File(aNameCam,"OrientationConique",aICNM);
       Im2D_Bits<1> * aMasq = GetImRemanenteFromFile<Im2D_Bits<1> > (anEntete+ aKLoc.NameMasq());

       TIm2DBits<1> TM(*aMasq);

       cFileOriMnt * anOri = RemanentStdGetObjFromFile<cFileOriMnt>
                             (
                                anEntete+aKLoc.NameMTDMasq(),
                                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                "FileOriMnt",
                                "FileOriMnt"
                             );
       // std::cout << "ADR MASQ " << aMasq << " " << anOri << "\n";
        Pt3dr aPMnt = FromMnt(*anOri,aCam->OrigineProf());
        Pt2di aP(round_ni(aPMnt.x),round_ni(aPMnt.y));
        return ( TM.get(aP,0)==0 );
   }
   
   return true;
}

bool NameFilter(cInterfChantierNameManipulateur * aICNM,const cNameFilter & aFilter,const std::string & aName)
{
   return NameFilter("",aICNM,aFilter,aName);
}

bool NameFilter(const std::string & aSubD,cInterfChantierNameManipulateur * aICNM,const cTplValGesInit<cNameFilter> & aF,const std::string & aN)
{
   if (aF.IsInit())
      return NameFilter(aSubD,aICNM,aF.Val(),aN);
   return true;
}

bool NameFilter(cInterfChantierNameManipulateur * aICNM,const cTplValGesInit<cNameFilter> & aF,const std::string & aN)
{
   return NameFilter("",aICNM,aF,aN);
}


cXML_Date XML_Date0()
{
   cXML_Date aRes;

   aRes.year() = 0;
   aRes.month() = 0;
   aRes.day() = 0;
   aRes.hour() = 0;
   aRes.minute() = 0;
   aRes.second() = 0;
   aRes.year() = 0;
   aRes.time_system() = "";

   return aRes;
}

cXML_LinePt3d MM2Matis(const Pt3dr & aP)
{
  cXML_LinePt3d aRes;
 
  aRes.x() = aP.x;
  aRes.y() = aP.y;
  aRes.z() = aP.z;

  return aRes;
}


/*
corientation MM2Matis(const cOrientationConique & anOC)
{
   const cOrientationExterneRigide & anOER = anOC.Externe();
   corientation aRes;

   aRes.version() = 1.0;
   aRes.image_name() = "UNKNOWN";
   aRes.stereopolis() = "UNKNOWN";
   aRes.image_date() = XML_Date0();

  // extrinseque
   aRes.euclidien().x() = anOER.Centre().x ;
   aRes.euclidien().y() = anOER.Centre().y;
   aRes.geodesique() = "+init=IGNF:LAMB1";

   aRes.grid_alti() ="UNKNOWN";

   aRes.easting() = 0.0;
   aRes.northing() = 0.0;
   aRes.altitude() = anOER.Centre().z;

   aRes.Image2Ground() = true;
   aRes.l1() = MM2Matis(anOER.ParamRotation().CodageMatr().Val().L1());
   aRes.l2() = MM2Matis(anOER.ParamRotation().CodageMatr().Val().L2());
   aRes.l3() = MM2Matis(anOER.ParamRotation().CodageMatr().Val().L3());

  // intrinseque

   ElCamera * aCam = Cam_Gen_From_XML(anOC,(cInterfChantierNameManipulateur *)0);
   CamStenope * aCS = aCam->CS();

   aRes.sensor().name() = "";
   aRes.calibration_date() = XML_Date0();
   aRes.serial_number() = "UNKNOWN";

   aRes.image_size().width() =  aCS->Sz().x ;
   aRes.image_size().height() = aCS->Sz().y;
   aRes.ppa().c() = aCS->PP().x;
   aRes.ppa().l() = aCS->PP().y;
   aRes.ppa().focale() = aCS->Focale();

   const cCamStenopeDistRadPol * aCamDRP = aCS->Debug_CSDRP();
   ELISE_ASSERT(aCamDRP!=0,"MicMac->con, ne gere que les modeles radiaux");

   const ElDistRadiale_PolynImpair  & aDRP = aCamDRP->DRad() ;
   aRes.distortion().pps().c() = aDRP.Centre().x;
   aRes.distortion().pps().l() = aDRP.Centre().y;

   aRes.distortion().r1() = 0.0;
   aRes.distortion().r3() = aDRP.CoeffGen(0);
   aRes.distortion().r5() = aDRP.CoeffGen(1);
   aRes.distortion().r7() = aDRP.CoeffGen(2);


   delete aCS;
   return aRes;
}

cElXMLTree * ToXmlTreeWithAttr(const corientation & anOri)
{
   cElXMLTree * aRes = ToXMLTree(anOri);

   aRes->GetUnique("geometry")->SetAttr("type","physique");
   aRes->GetUnique("euclidien")->SetAttr("type","MATISRTL");

   return aRes;
}
*/

   // = = = = = = = = = = = = = = = = = = = = = = = =

void ModifDAF
     (
          cInterfChantierNameManipulateur* aICMN,
          cDicoAppuisFlottant &            aDico,
          const cTplValGesInit<cModifIncPtsFlottant> & aModif
     )
{
    if (aModif.IsInit())
       ModifDAF(aICMN,aDico,aModif.Val());
}
void ModifDAF
     (
          cInterfChantierNameManipulateur* aICMN,
          cDicoAppuisFlottant &            aDico,
          const cModifIncPtsFlottant &     aModif
     )
{
   for 
   (
       std::list<cOneModifIPF>::const_iterator itO=aModif.OneModifIPF().begin();
       itO!=aModif.OneModifIPF().end();
       itO++
   )
   {
        ModifDAF(aICMN,aDico,*itO);
   }
}

void ModifDAF
     (
           cInterfChantierNameManipulateur* aICNM,
           cDicoAppuisFlottant &            aDico,
           const cOneModifIPF &             aModif
     )
{
   
    cSetName * aSet = aICNM->KeyOrPatSelector(aModif.KeyName());
    for 
    (
       std::list<cOneAppuisDAF>::iterator itO=aDico.OneAppuisDAF().begin();
       itO!=aDico.OneAppuisDAF().end();
       itO++
   )
   {
       if (aSet->IsSetIn(itO-> NamePt()))
       {
          Pt3dr anInc = aModif.Incertitude();
          if (aModif.IsMult().Val())
          {
             itO->Incertitude().x *= anInc.x;
             itO->Incertitude().y *= anInc.y;
             itO->Incertitude().z *= anInc.z;
          }
          else 
          {
             itO->Incertitude() = anInc;
          }
       }
   }
}


bool SameGeometrie(const cFileOriMnt & aF1,const cFileOriMnt & aF2)
{
   if (aF1.Geometrie() != aF2.Geometrie())
      return false;

   if (aF1.Geometrie() == eGeomMNTCarto )
   {
      if (aF1.NumZoneLambert().Val() != aF2.NumZoneLambert().Val())
         return false;
   }

   if (aF1.Geometrie() == eGeomMNTEuclid )
   {
      if (aF1.OrigineTgtLoc().IsInit() != aF2.OrigineTgtLoc().IsInit())
         return false;

      if (aF1.OrigineTgtLoc().IsInit() )
      {
         if (aF1.OrigineTgtLoc().Val() != aF2.OrigineTgtLoc().Val())
            return false;
      }
   }

   return true;
}


Fonc_Num  AdaptFonc2FileOriMnt
          (
                 const std::string & aContext,
                 const cFileOriMnt & anOriCible,
                 const cFileOriMnt &  aOriInit,
                 Fonc_Num            aFonc,
                 bool                aModifDyn,
                 double              aZOffset,
                 const Pt2dr &       CropInOriCible

          ) 
{
    if (! SameGeometrie(aOriInit,anOriCible))
    {
       cElWarning::GeomIncompAdaptF2Or.AddWarn
       (
            aContext,
             __LINE__,
             __FILE__
       );
       // std::cout  << "Geometrie incompatible Cible/Xml, AdaptFoncFileOriMnt\n" ;
    }


    Pt2dr aSxyC = anOriCible.ResolutionPlani();
    Pt2dr aSxyI = aOriInit.ResolutionPlani();

    double aSx = aSxyC.x / aSxyI.x;
    double aSy = aSxyC.y / aSxyI.y;

    if ((aSx<=0) || (aSy <=0))
    {
        std::cout << "Context = " << aContext << "\n";
        ELISE_ASSERT
        (
            false,
            "Signe incompatibles dans AdaptFoncFileOriMnt"
        );
    }

    Pt2dr aTr = anOriCible.OriginePlani() -  aOriInit.OriginePlani();

// std::cout << aTr << CropInOriCible << aSxyI << " " << aSx << "\n";
    aTr =  Pt2dr(aTr.x/aSxyI.x,aTr.y/aSxyI.y);
    aTr =  aTr +CropInOriCible.mcbyc(Pt2dr(aSx,aSy));

    aFonc =  StdFoncChScale_Bilin
             (
                  aFonc,
                  aTr,
                  // Pt2dr(aTr.x/aSxyI.x,aTr.y/aSxyI.y) + CropInOriCible,
                  Pt2dr(aSx,aSy)
             );

    if (aModifDyn)
    {
       double aZ0I = aOriInit.OrigineAlti();
       double aZ0C = anOriCible.OrigineAlti();
       double aStZC = anOriCible.ResolutionAlti();
       double aStZI = aOriInit.ResolutionAlti();
       aFonc = (aZOffset+ aZ0I-aZ0C)/aStZC + (aStZI/aStZC)*aFonc;
    }

    return aFonc;
}


const cOneMesureAF1I *  PtsOfName(const cMesureAppuiFlottant1Im & aSet,const std::string & aName)
{
   for 
   (
         std::list<cOneMesureAF1I>::const_iterator iT=aSet.OneMesureAF1I().begin();
         iT!=aSet.OneMesureAF1I().end();
         iT++
   )
   {
      if (iT->NamePt() == aName)
         return &(*iT);
   }
   return 0;
}

ElPackHomologue PackFromCplAPF(const cMesureAppuiFlottant1Im & aMes, const cMesureAppuiFlottant1Im & aRef)
{
   ElPackHomologue aRes;

   for 
   (
         std::list<cOneMesureAF1I>::const_iterator iT=aMes.OneMesureAF1I().begin();
         iT!=aMes.OneMesureAF1I().end();
         iT++
   )
   {
       const cOneMesureAF1I * aPt = PtsOfName(aRef,iT->NamePt());
       if (aPt!=0)
       {
           aRes.Cple_Add(ElCplePtsHomologues(iT->PtIm(),aPt->PtIm()));
       }
       else
       {
           std::cout << "For name " << iT->NamePt() << "\n";
           ELISE_ASSERT(false,"Cannot get name in merging two cMesureAppuiFlottant1Im");
       }
   }

   return aRes;
}


const std::list<std::string > * GetBestImSec(const cImSecOfMaster& anISOM,int aNb,int aNbMin,int aNbMax,bool  OkAndOutWhenNone)
{
   const std::list<std::string > * aRes = 0;
   double aScoreMax=-1;
   for 
   (
          std::list<cOneSolImageSec>::const_iterator itS=anISOM.Sols().begin();
          itS!=anISOM.Sols().end();
          itS++
   )
   {
       int aNbIm = (int)itS->Images().size();
       if ((aNb<=0) || (aNbIm == aNb))
       {
            if ((itS->Score() > aScoreMax) && (aNbIm>=aNbMin) && (aNbIm<=aNbMax))
            {
                 aScoreMax = itS->Score();
                 aRes = &(itS->Images());
            }
       }
   }
   if ((aRes==0 ) && (!OkAndOutWhenNone))
   {
       std::cout  << "For image " << anISOM.Master() << " and Nb= " << aNb << "\n";
       ELISE_ASSERT(aRes!=0,"Cannot GetBestSec");
   }
   return aRes;
}


cImSecOfMaster StdGetISOM
               (
                    cInterfChantierNameManipulateur * anICNM,
                    const std::string & aNameIm,
                    const std::string & anOri
               )
{
    std::string aKey = "NKS-Assoc-ImSec@-"+anOri;
    std::string aFile = anICNM->Dir() + anICNM->Assoc1To1(aKey,aNameIm,true);

    return StdGetFromPCP(aFile,ImSecOfMaster);
}




cEl_GPAO * DoCmdExePar(const cCmdExePar & aCEP,int aNbProcess)
{
   cEl_GPAO * aGPAO  = new cEl_GPAO;
   
   int aKT=0;
   for 
   (
       std::list<cOneCmdPar>::const_iterator itOCP=aCEP.OneCmdPar().begin();
       itOCP!=aCEP.OneCmdPar().end();
       itOCP++
   )
   {
       cElTask & aTask = aGPAO->NewTask("T_"+ToString(aKT),"");
       for 
       (
            std::list<std::string>::const_iterator itS=itOCP->OneCmdSer().begin();
            itS != itOCP->OneCmdSer().end();
            itS++
       )
	   aTask.AddBR(*itS);

       aGPAO->TaskOfName("all").AddDep(aTask);

       aKT++;
   }
   std::string aNameMkF = aCEP.NameMkF().Val();
   if (aNameMkF!="") 
   {
      aNbProcess = ElMax(1,aNbProcess);
      aGPAO->GenerateMakeFile(aNameMkF);

	  launchMake( aNameMkF, "all", aNbProcess, "-k" );

      delete aGPAO;
       ELISE_fp::RmFile(aNameMkF);
      return 0;
   }
   return aGPAO;
}


    //  ============= FORMAT AVION JAUNE ================

eUniteAngulaire AJStr2UAng(const std::string & aName)
{
   if (aName=="degree")
      return eUniteAngleDegre;

   ELISE_ASSERT(false,"Unknown unit in AJStr2UAng");
   return eUniteAngleDegre;
}

double AJ2Radian(const cValueAvionJaune & aVal)
{
   return ToRadian
          (
              aVal.value(),
              AJStr2UAng(aVal.unit())
          );
}


Pt3dr  CentreAJ(const cAvionJauneDocument & anAJD)
{
    return Pt3dr
           (
               anAJD.navigation().sommet().x(),
               anAJD.navigation().sommet().y(),
               anAJD.navigation().altitude().value()
           );
}


ElMatrix<double> RotationVecAJ(const cAvionJauneDocument & anAJD)
{
   double aCap = AJ2Radian(anAJD.navigation().capAvion());
// Envoie de la camera vers l'avion
   ElMatrix<double> aRCam2Av = ElMatrix<double>::Rotation
                               (
                                    Pt3dr(1, 0, 0),
                                    Pt3dr(0,-1, 0),
                                    Pt3dr(0, 0,-1)
                               );

   ElMatrix<double> aMCap = ElMatrix<double>::Rotation(aCap,0,0);


   ElMatrix<double> aRAv2Ter = aMCap;

   return aRAv2Ter * aRCam2Av;
}

ElRotation3D   AJ2R3d(const cAvionJauneDocument & anAJD)
{
    return ElRotation3D(CentreAJ(anAJD),RotationVecAJ(anAJD),true);
}


cOrientationExterneRigide AJ2Xml(const cAvionJauneDocument & anAJD)
{
   return From_Std_RAff_C2M(AJ2R3d(anAJD),true);
}


void AddKeySet
     (
         std::set<std::string> &          aRes,
         cInterfChantierNameManipulateur* anICNM,
         const std::string &              aKey
     )
{

    const std::vector<std::string>  * aSet = anICNM->Get(aKey);
    aRes.insert(aSet->begin(),aSet->end());
}


void AddListKeySet
     (
         std::set<std::string> &            aRes  ,
         cInterfChantierNameManipulateur*   anICNM,
         const std::list<std::string> &     aLSet
     )
{
    for 
    (
          std::list<std::string>::const_iterator itS=aLSet.begin();  
          itS!=aLSet.end(); 
          itS++
    )
      AddKeySet(aRes,anICNM,*itS);
}

std::vector<std::string> GetStrFromGenStr
                         (
                              cInterfChantierNameManipulateur* anICNM,
                              const cParamGenereStr & aPar
                         )
{
    std::set<std::string> aRes;
    aRes.insert(aPar.KeyString().begin(),aPar.KeyString().end());
    AddListKeySet(aRes,anICNM,aPar.KeySet());

    return std::vector<std::string>(aRes.begin(),aRes.end());
}


std::vector<std::string> GetStrFromGenStrRel
                         (
                              cInterfChantierNameManipulateur* anICNM,
                              const cParamGenereStrVois & aPar,
                              const std::string & aStr0
                         )
{
    std::set<std::string> aRes;

    aRes.insert(aPar.KeyString().begin(),aPar.KeyString().end());
    AddListKeySet(aRes,anICNM,aPar.KeySet());

    for 
    (
          std::list<std::string>::const_iterator itS=aPar.KeyRel().begin();  
          itS!=aPar.KeyRel().end(); 
          itS++
    )
    {
        std::vector<std::string>   aSet = anICNM->GetSetOfRel(*itS,aStr0);
        aRes.insert(aSet.begin(),aSet.end());
    }

    return std::vector<std::string>(aRes.begin(),aRes.end());
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
