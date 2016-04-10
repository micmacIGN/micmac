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






/**************************************************************/
/*                                                            */
/*          Lecture a partir de fichiers                      */
/*                                                            */
/**************************************************************/

cParamChantierPhotogram GetChantierFromFile                         
                        (
                              const std::string & aNameFile,
                              const std::string & aNameTag
                        )
{
   return StdGetObjFromFile<cParamChantierPhotogram>
          (
               aNameFile,
               StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
               aNameTag,
               "ParamChantierPhotogram"
          );
}

cParamChantierPhotogram GetChantierFromFile(const std::string & aNameFile)
{
   return GetChantierFromFile(aNameFile,"ParamChantierPhotogram");
}

/**************************************************************/
/*                                                            */
/*                   Conversions                              */
/*                                                            */
/**************************************************************/

bool operator < (const cPDV & aC1,const cPDV & aC2) 
{
      if (aC1.Bande() < aC2.Bande()) 
         return true;
      if (aC1.Bande() ==  aC2.Bande()) 
         return (aC1.IdInBande()<aC2.IdInBande());
      return false;
}

cVolChantierPhotogram MakeVolFromParam(const cParamVolChantierPhotogram & aParam)
{
   std::vector<cPDV> mPDVS;
   cVolChantierPhotogram aRes;
 
   std::string aDirOr =  aParam.Directory()+aParam.DirOrientations().Val();
   std::string aDirIm =  aParam.Directory()+aParam.DirImages().Val();
   std::list<std::string>  aListName = RegexListFileMatch
                                       (
                                            aDirOr,
                                            aParam.NameSelector()->NameExpr(),
                                            10,
                                            false
                                       );
   for 
   ( 
          std::list<std::string>::iterator itS = aListName.begin();
          itS != aListName.end();
          itS++
   )
   {
        bool OkBande = aParam.BandeIdSelector()->Match(*itS);
        ELISE_ASSERT(OkBande,"No Match for BandeIdSelector");

        OkBande =  aParam.BandeIdSelector()->Replace(aParam.NomBandeId());
        ELISE_ASSERT(OkBande,"No Match for NomBandeId");
        std::string aNameBande =  aParam.BandeIdSelector()->LastReplaced();

        OkBande =  aParam.BandeIdSelector()->Replace(aParam.NomIdInBande());
        ELISE_ASSERT(OkBande,"No Match for NomIdInBande");
        std::string anIdInBande =  aParam.BandeIdSelector()->LastReplaced();

        OkBande =  aParam.BandeIdSelector()->Replace(aParam.NomImage());
        ELISE_ASSERT(OkBande,"No Match for NomImage");
        std::string aNameIm =  aParam.BandeIdSelector()->LastReplaced();

        cPDV aPDV;
        aPDV.Orient() = aDirOr+*itS;
        aPDV.IdInBande() = anIdInBande;
        aPDV.Im() = aDirIm+aNameIm;
        aPDV.Bande() = aNameBande;
        mPDVS.push_back(aPDV);
   }

   std::sort(mPDVS.begin(),mPDVS.end());

   int aKDeb = 0;
   int aNb = (int) mPDVS.size();
   for (int aK=1 ;  aK<= aNb  ; aK++)
   {
       bool aDiff = (aK== aNb) || ( mPDVS[aK].Bande()  !=  mPDVS[aK-1].Bande());
       if (aDiff)
       {
           cBandesChantierPhotogram aBBBB;
           aRes.BandesChantierPhotogram().push_back(aBBBB);
           cBandesChantierPhotogram & aBande = aRes.BandesChantierPhotogram().back();
           aBande.IdBande() = mPDVS[aKDeb].Bande() ;

           for (int aKB = aKDeb ; aKB<aK ; aKB++)
           {
               aBande.PDVs().push_back(mPDVS[aKB]);
           }
           aKDeb = aK;
       }
   }

   return aRes;
}


cChantierPhotogram MakeChantierFromParam(const cParamChantierPhotogram & aParam)
{
    cChantierPhotogram aRes;

    for 
    (
        std::list<cParamVolChantierPhotogram>::const_iterator  iT = aParam.ParamVolChantierPhotogram().begin();
        iT != aParam.ParamVolChantierPhotogram().end();
        iT++
    )
    {
         aRes.VolChantierPhotogram().push_back(MakeVolFromParam(*iT));
    }

    return aRes;
}

std::list<cPDV> ListePDV(const cChantierPhotogram & aCh)
{
  std::list<cPDV> aRes;

  for 
  (
    std::list<cVolChantierPhotogram>::const_iterator itV=aCh.VolChantierPhotogram().begin();
    itV!=aCh.VolChantierPhotogram().end();
    itV++
  )
  {
     for 
     (
       std::list<cBandesChantierPhotogram>::const_iterator itB=itV->BandesChantierPhotogram().begin();
       itB!= itV->BandesChantierPhotogram().end();
       itB++
     )
     {
        for 
        (
          std::list<cPDV>::const_iterator itP=itB->PDVs().begin();
          itP!= itB->PDVs().end();
          itP++
        )
        {
             aRes.push_back(*itP);
        }
     }
  }

  return aRes;
}

/**************************************************************/
/*                                                            */
/*                   Liste de Couples                         */
/*                                                            */
/**************************************************************/


struct  cOriForLC
{
    // aRab en %, 0 = Id, -100 = vide 
    cOriForLC(cPDV aPDV,double aRab);

    cPDV               mPDV;
    Ori3D_Std          mOri; 
    std::vector<Pt2dr> mPts;
    std::vector<Pt2dr> mPtsDil;
    Box2dr             mBox;
    Pt2dr              mCdg;
};

cOriForLC::cOriForLC(cPDV aPDV,double aRab) :
    mPDV    (aPDV),
    mOri    (mPDV.Orient().c_str()),
    mPts    (mOri.EmpriseSol()),
    mPtsDil (DilateHomotetikCdg(mPts,aRab)),
    mBox    (BoxPts(mPtsDil)),
    mCdg    (barrycentre(mPtsDil))
{
}

cGraphePdv GrapheOfRecouvrement(const cChantierPhotogram & aCh,double aFact)
{
   El_Window * aW = 0;
   cGraphePdv aRes;

   std::list<cPDV> aLPDV = ListePDV(aCh);
   std::vector<cOriForLC> aVOri;
   bool aFirst=true;

   for 
   (
       std::list<cPDV>::const_iterator itP=aLPDV.begin();
       itP!=aLPDV.end();
       itP++
   )
   {
       aRes.PDVs().push_back(*itP);
       aVOri.push_back(cOriForLC(*itP,aFact));
       if (aFirst)
       {
          aFirst = false;
          aRes.BoxCh() = aVOri.back().mBox;
       }
       else
       {
          aRes.BoxCh() = Sup(aRes.BoxCh(),aVOri.back().mBox);
       }
   }
   int aNbIm = (int) aRes.PDVs().size();

   if (true)
   {
      Pt2dr aSzMax(800,500);
      Pt2dr  aSz = aRes.BoxCh().sz();
      double anEch =  ElMin(aSzMax.x/aSz.x,aSzMax.y/aSz.y);
      Pt2di aSzW = Pt2di(aSz *anEch + Pt2dr(3,3));

      El_Window aWVideo = Video_Win::WStd(aSzW,1.0);
      aW = new El_Window(aWVideo.chc(aRes.BoxCh()._p0,Pt2dr(anEch,anEch)));
   }

   double aNbC=0;
   double aNbCI=0;
   for (int aK1=0 ; aK1<aNbIm ; aK1++)
   {
       for (int aK2=aK1+1 ; aK2<aNbIm ; aK2++)
       {
            aNbC++;
            if (HasInter(aVOri[aK1].mPtsDil,aVOri[aK2].mPtsDil))
            {
               aNbCI++;
               cCplePDV aCple;
               aCple.Id1() = aK1;
               aCple.Id2() = aK2;
               aRes.CplePDV().push_back(aCple);
            }
       }
   }
   std::cout << "Couple Inter " << aNbCI 
             << " sur " << aNbC << " couples " 
             << " et " << aNbIm << " images "
             << "\n";

   if (aW)
   {
      for (int aK=0 ; aK<aNbIm ; aK++)
      {
          // aW->draw_rect(aVOri[aK].mBox,aW->pdisc()(P8COL::red));
          aW->draw_poly_ferm(aVOri[aK].mPtsDil,aW->pdisc()(P8COL::green));
      }
      for 
      (
           std::list<cCplePDV>::iterator itC=aRes.CplePDV().begin();
           itC!=aRes.CplePDV().end();
           itC++
      )
      {
               aW->draw_seg
               (
                    aVOri[itC->Id1()].mCdg,
                    aVOri[itC->Id2()].mCdg,
                    aW->pdisc()(P8COL::blue)
               );
      }
      getchar();
   }


   return aRes;
}


double UneUniteEnRadian(eUniteAngulaire aUnit)
{
   static double UnRadianEnDegre = PI/180.0;
   static double UnRadianEnGrade = PI/200.0;

   switch (aUnit)
   {
      case eUniteAngleDegre : return  UnRadianEnDegre;
      case eUniteAngleGrade : return   UnRadianEnGrade;
      case eUniteAngleRadian : return 1.0; ;
      case eUniteAngleUnknown : 
             ELISE_ASSERT(false,"Rotation format supports only matrixes (no angles)");
           return 1.0; ;
      default :  ;
   }
   ELISE_ASSERT(false,"ToRadian , unkown eUniteAngulaire");
   return 0;
}


double ToRadian(const double & aTeta,eUniteAngulaire aUnit)
{
    return aTeta * UneUniteEnRadian(aUnit);
}

double FromRadian(const double & aTeta,eUniteAngulaire aUnit)
{
    return aTeta / UneUniteEnRadian(aUnit);
}



cNameSpaceEqF::eTypeSysResol ToNS_EqF(eModeSolveurEq aMode)
{
   // + sale tu meurs
   return (cNameSpaceEqF::eTypeSysResol ) (int) (aMode);
}



// cCamStenopeDistRadPol 


cSimilitudePlane El2Xml(const ElSimilitude & aSim)
{
    cSimilitudePlane aRes;
    aRes.Trans() = aSim.tr();
    aRes.Scale() = aSim.sc();

    return aRes;
}

ElSimilitude Xml2EL(const cSimilitudePlane & aXML)
{
   return ElSimilitude(aXML.Trans(),aXML.Scale());
}


cAffinitePlane El2Xml(const ElAffin2D & anAff)
{
    cAffinitePlane aRes;
    aRes.I00() = anAff.I00();
    aRes.V10() = anAff.I10();
    aRes.V01() = anAff.I01();

    return aRes;
}

ElAffin2D Xml2EL(const cAffinitePlane & aXML)
{
   return ElAffin2D(aXML.I00(),aXML.V10(),aXML.V01());
}

ElAffin2D Xml2EL(const cTplValGesInit<cAffinitePlane> & aXML)
{
   if (aXML.IsInit()) return Xml2EL(aXML.Val());
   return ElAffin2D::Id();
}

void AddAffinite(cOrientationConique & anOri,const ElAffin2D & anAffAdd)
{
   ElAffin2D anAffCur = AffCur(anOri);

   anOri.OrIntImaM2C().SetVal(El2Xml(anAffAdd*anAffCur));
}

ElAffin2D AffCur(const cOrientationConique & anOri)
{
   return Xml2EL(anOri.OrIntImaM2C().ValWithDef(El2Xml(ElAffin2D::Id())));
}


void AssertOrIntImaIsId(const cOrientationConique & anOC)
{
   ELISE_ASSERT
   (
        AffCur(anOC).IsId(),
        "L'orientation interne image attendue n'est pas Identite"
   );
}


eTypeProj Xml2EL(const eTypeProjectionCam & aType)
{
   switch (aType)
   {
      case  eProjStenope:
            return eProjectionStenope;

      case  eProjOrthographique:
            return  eProjectionOrtho ;

      case  eProjGrid :
            ;
   }
   ELISE_ASSERT(false," Xml2EL(const eTypeProjectionCam & aType)");
   return  eProjectionOrtho ;
}

eTypeProjectionCam El2Xml(const eTypeProj & aType)
{

   switch (aType)
   {
       case eProjectionStenope:
           return eProjStenope ;
       case eProjectionOrtho :
            return eProjOrthographique;
   }
   ELISE_ASSERT(false," Xml2EL(const eTypeProj & aType)");
   return eProjOrthographique;
}

std::vector<cOneMesureAF1I> GetMesureOfPts(const cSetOfMesureAppuisFlottants & aSMAF,const std::string & aNamePt)
{
    std::vector<cOneMesureAF1I> aRes;

    for 
    (
          std::list<cMesureAppuiFlottant1Im>::const_iterator itMAF=aSMAF.MesureAppuiFlottant1Im().begin();
          itMAF!=aSMAF.MesureAppuiFlottant1Im().end();
          itMAF++
    )
    {
           for 
           (
               std::list<cOneMesureAF1I>::const_iterator itM=itMAF->OneMesureAF1I().begin();
               itM!=itMAF->OneMesureAF1I().end();
               itM++
           )
           {
                if (itM->NamePt()==aNamePt)
                {
                    cOneMesureAF1I aM = *itM;
                    aM.NamePt() = itMAF->NameIm();
                    aRes.push_back(aM);
                }
           }
    }

    return aRes;
}
std::vector<cOneMesureAF1I> GetMesureOfPtsIm(const cSetOfMesureAppuisFlottants & aSMAF,const std::string & aNamePt,const std::string & Im)
{
    std::vector<cOneMesureAF1I> aRes;

    for 
    (
          std::list<cMesureAppuiFlottant1Im>::const_iterator itMAF=aSMAF.MesureAppuiFlottant1Im().begin();
          itMAF!=aSMAF.MesureAppuiFlottant1Im().end();
          itMAF++
    )
    {
           if (Im==itMAF->NameIm())
           {
              for 
              (
                  std::list<cOneMesureAF1I>::const_iterator itM=itMAF->OneMesureAF1I().begin();
                  itM!=itMAF->OneMesureAF1I().end();
                  itM++
              )
              {
                   if (itM->NamePt()==aNamePt)
                   {
                       cOneMesureAF1I aM = *itM;
                       aRes.push_back(aM);
                   }
              }
           }
    }

    return aRes;
}

const cOneAppuisDAF * GetApOfName(const cDicoAppuisFlottant & aDAF,const std::string & aNamePt)
{
    for (std::list<cOneAppuisDAF>::const_iterator itA=aDAF.OneAppuisDAF().begin(); itA!=aDAF.OneAppuisDAF().end(); itA++)
    {
        if (itA->NamePt()==aNamePt)
           return &(*itA);
    }
    return 0;
}



double ToMnt(const cFileOriMnt & aFOM,const double & aZ)
{
   return aFOM.OrigineAlti() + aZ *aFOM.ResolutionAlti();
}

Pt2dr  ToMnt(const cFileOriMnt & aFOM,const Pt2dr  &aP)
{
    return   aFOM.OriginePlani()
           + aP.mcbyc(aFOM.ResolutionPlani());
}

Pt3dr  ToMnt(const cFileOriMnt & aFOM,const Pt3dr & aP)
{
    Pt2dr aP2 = ToMnt(aFOM,Pt2dr(aP.x,aP.y));
    double aZ= ToMnt(aFOM,aP.z);
 
    return Pt3dr(aP2.x,aP2.y,aZ);
}


double FromMnt(const cFileOriMnt & aFOM,const double & aZ)
{
   return (aZ-aFOM.OrigineAlti()) / aFOM.ResolutionAlti();
}
Pt2dr  FromMnt(const cFileOriMnt & aFOM,const Pt2dr  &aP)
{
    return   (aP-aFOM.OriginePlani()).dcbyc(aFOM.ResolutionPlani());
}
Pt3dr  FromMnt(const cFileOriMnt & aFOM,const Pt3dr & aP)
{
    Pt2dr aP2 = FromMnt(aFOM,Pt2dr(aP.x,aP.y));
    double aZ= FromMnt(aFOM,aP.z);
 
    return Pt3dr(aP2.x,aP2.y,aZ);
}





cGridDeform2D   ToXMLExp(const PtImGrid & aCstPIG)
{
   PtImGrid &  aPIG = const_cast<PtImGrid &>(aCstPIG);
   cGridDeform2D aRes;

   aRes.Origine() = aPIG.Origine();
   aRes.Step() =  aPIG.Step();
   aRes.ImX() =  aPIG.DataGridX();
   aRes.ImY() =  aPIG.DataGridY();

   return aRes;
}

cGridDirecteEtInverse   ToXMLExp(const cDbleGrid& aGR2)
{
    cGridDirecteEtInverse aRes;
    aRes.Directe() = ToXMLExp(aGR2.GrDir());
    aRes.Inverse() = ToXMLExp(aGR2.GrInv());
    aRes.AdaptStep() = aGR2.StepAdapted();

    return aRes;
}

/*          cCompCNFC   */

cCompCNFC::cCompCNFC(const cCalcNomFromCouple& aCNFC)  :
    mCNFC    (aCNFC),
    mAutom   (aCNFC.Pattern2Match(),20)
{
}


std::string cCompCNFC::NameCalc
            (
                 const std::string& aN1,
                 const std::string& aN2
            )
{
  return MatchAndReplace
         (
	    mAutom,
	    aN1+mCNFC.Separateur().Val()+aN2,
	    mCNFC.NameCalculated()
	 );
}

/*          cCompCNF1   */

cCompCNF1::cCompCNF1(const cCalcNomFromOne & aCNF1)  :
    mCNF1    (aCNF1),
    mAutom   (aCNF1.Pattern2Match(),20)
{
}


std::string cCompCNF1::NameCalc ( const std::string& aName)
{
  return MatchAndReplace(mAutom, aName, mCNF1.NameCalculated());
}

                     


cStdMapName2Name::cStdMapName2Name
(
    const cMapName2Name &  aMapN2N,
    cInterfChantierNameManipulateur  * anICNM
) :
   mICNM  (anICNM),
   mMapN2N (aMapN2N)
{
}


static const std::string NoDefMn2n = "NoDefMn2n";

std::string cStdMapName2Name::map_with_def
            (
                  const std::string & aName ,
                  const std::string & aDef
            )
{
   if (mMapN2N.MapByKey().IsInit())
   {
      ELISE_ASSERT(mICNM!=0,"Incoherence in cStdMapName2Namemap::map");
      std::string aRes = mICNM->Assoc1To1
                          (
                               mMapN2N.MapByKey().Val().Key(),
                               aName,
                               true
                          );
      if (mMapN2N.MapByKey().Val().DefIfFileNotExisting().Val())
      {
           if (! ELISE_fp::exist_file(mICNM->Dir()  + aRes))
              return aDef;
      }
      return aRes;
   }
   if (mMapN2N.MapN2NByAutom().IsInit())
   {
       std::vector<cOneAutomMapN2N> & aV =  mMapN2N.MapN2NByAutom().Val().OneAutomMapN2N();
       for (int aK=0 ; aK<int(aV.size()) ; aK++)
       {
            cElRegex_Ptr  aASel = aV[aK].AutomSel().ValWithDef(aV[aK].MatchPattern());
            if (aASel->Match(aName))
            {
               return MatchAndReplace
                      (
                          *aV[aK].MatchPattern(),
                          aName,
                          aV[aK].Result()
                      );
             }
       }
   }



   
   ELISE_ASSERT(aDef!=NoDefMn2n,"No match in cStdMapName2Namemap::map");
   return aDef;
}

std::string cStdMapName2Name::map(const std::string &  aName)
{
   return map_with_def(aName,NoDefMn2n);
}


cStdMapName2Name * StdAllocMn2n 
                   (
                      const cTplValGesInit<cMapName2Name> &  aMapN2N,
                       cInterfChantierNameManipulateur * anICNM
                   )
{
   return aMapN2N.IsInit()                           ?
          new cStdMapName2Name(aMapN2N.Val(),anICNM) :
          0                                          ;
}

Im1D_INT4 LutGama(int aNbVal,double aGama,double aValRef,int aMaxVal,double aCoeff)
{
// std::cout << "MMMMMMMMMMmmmmmmmm   " << aMaxVal << "\n";
    Im1D_INT4 aRes(round_ni(aNbVal*aCoeff));
    ELISE_COPY
    (
        aRes.all_pts(),
        Min(aMaxVal,round_ni(aValRef*pow((FX/aCoeff)/aValRef,1.0/aGama))),
        aRes.out()
    );
    return aRes;
}

Im1D_INT4 LutIm(const cLutConvertion & aLut,int aMinVal,int aMaxVal,double aCoeff)
{
   const std::vector<cIntervLutConvertion > & aViLC = aLut.IntervLutConvertion();
   int aNbV = (int)aViLC.size();

   Im1D_INT4 aRes(round_ni(aViLC[aNbV-1].NivIn()*aCoeff));
   ELISE_ASSERT(aViLC[0].NivIn()==0,"LutConvertion");

   for (int aK=0 ; aK<aNbV-1 ; aK++)
   {
       const cIntervLutConvertion &  aCur = aViLC[aK];
       const cIntervLutConvertion &  aNext = aViLC[aK+1];
       
       ELISE_COPY
       (
           rectangle
           (
              round_ni(aCur.NivIn()*aCoeff),
              round_ni((aNext.NivIn()+1)*aCoeff)
           ),
           aCur.NivOut() + ((FX/aCoeff)-aCur.NivIn()) * (double(aNext.NivOut()-aCur.NivOut()) / double(aNext.NivIn()-aCur.NivIn())),
           aRes.out()
       );
   }
   ELISE_COPY(aRes.all_pts(),Max(aMinVal,Min(aMaxVal,aRes.in())),aRes.out());
   return aRes;
}

Fonc_Num SafeUseLut(Im1D_INT4 aLut,Fonc_Num aF,double aCoeff)
{
   Fonc_Num aRes;
   for (int aK=0 ; aK<aF.dimf_out() ;  aK++)
   {
       Fonc_Num aFK = aLut.in()[Max(0,Min(aLut.tx()-1,round_ni(aCoeff*aF.kth_proj(aK))))];
       aRes = (aK==0) ? aFK : Virgule(aRes,aFK);
   }
   return aRes;
}




std::list<std::string> GetListFromSetSauvInFile
                        (
                              const std::string & aNameFile,
                              const std::string & aNameTag
                        )
{
   return StdGetObjFromFile<cSauvegardeSetString>
          (
               aNameFile,
               StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
               aNameTag,
               "SauvegardeSetString"
          ).Name();

}

bool RepereIsAnam(const std::string & aName,bool  &IsOrthXCSte,bool & IsAnamXCsteOfCart)
{
     cElXMLTree aTree(aName);
     cElXMLTree * aTreeSurf = aTree.Get("XmlOneSurfaceAnalytique");

     IsOrthXCSte = false;

     if (aTreeSurf) 
     {
        cXmlOneSurfaceAnalytique aXmlSurf;
        xml_init(aXmlSurf,aTreeSurf);
        cInterfSurfaceAnalytique * aSurf = cInterfSurfaceAnalytique::FromXml(aXmlSurf);
        IsOrthXCSte = aSurf->HasOrthoLoc() && aSurf->OrthoLocIsXCste();
        IsAnamXCsteOfCart =  aSurf->IsAnamXCsteOfCart();
        return true;
     }

     cElXMLTree * aRep = aTree.Get("RepereCartesien");
     if (aRep) 
        return false;
 
     std::cout << "For file " << aName << "\n";

     ELISE_ASSERT(false,"RepereIsAnam : cant  get <XmlOneSurfaceAnalytique> nor <RepereCartesien>");
     return false;
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
