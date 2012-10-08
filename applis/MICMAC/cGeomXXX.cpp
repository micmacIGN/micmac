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
#include "MICMAC.h"

namespace NS_ParamMICMAC
{

/*****************************************/
/*                                       */
/*            cGeomDiscR2                */
/*                                       */
/*****************************************/

bool IsModeInvY(const eModeGeomMNT & aMode)
{
    return    (aMode== eGeomMNTCarto)
           || (aMode== eGeomMNTEuclid) ;
}

cGeomDiscR2::cGeomDiscR2
(
    const cAppliMICMAC & anAppli
) :
  mAp    (&anAppli),
  mInvY  (IsModeInvY(anAppli.GeomMNT()))
{
}



Pt2dr   cGeomDiscR2::RDiscToR2(const Pt2dr & aPRas) const
{
    Pt2dr aPRel = (mRP0DiscClip+aPRas)*mResolDz ;

    return (mInvY)                                  ?
           Pt2dr(mP0.x+aPRel.x,mP1.y-aPRel.y)       :
           mP0+aPRel                                ;
}

Pt2dr   cGeomDiscR2::DiscToR2(const Pt2di & aPRas) const
{
   return RDiscToR2(Pt2dr(aPRas));
}

Pt2dr   cGeomDiscR2::R2ToRDisc(Pt2dr   aP) const
{
    if (mInvY)
    {
	    aP = Pt2dr(aP.x-mP0.x,mP1.y-aP.y);
    }
    else
	    aP = aP-mP0;

    return aP/mResolDz-mRP0DiscClip;
}

Pt2di   cGeomDiscR2::R2ToDisc(const Pt2dr &  aP) const
{
    return round_ni(R2ToRDisc(aP));
}

void cGeomDiscR2::SetClipInit()
{
   SetClip(Pt2di(0,0),mSzDz);
}


void cGeomDiscR2::SetDeZoom(REAL aDz)
{
   mDeZoom   = aDz;
   mResolDz  = mResol * aDz;
   mSzDz = round_ni((mP1-mP0)/mResolDz);
   SetClipInit();
   mDerX = RDiscToR2(Pt2dr(1,0))-RDiscToR2(Pt2dr(0,0));
   mDerY = RDiscToR2(Pt2dr(0,1))-RDiscToR2(Pt2dr(0,0));
}

double cGeomDiscR2::ResolDz() const { return mResolDz; }
double cGeomDiscR2::ResolZ1() const { return mResol; }

Pt2dr   cGeomDiscR2::DerX(const  Pt2di & ) const
{
   return mDerX;
}

Pt2dr   cGeomDiscR2::DerY(const  Pt2di & ) const
{
   return mDerY;
}


void cGeomDiscR2::SetClip(const Pt2di & aP0Cl,const Pt2di & aP1Cl)
{
   mP0DiscClip = aP0Cl;
   mRP0DiscClip = Pt2dr(mP0DiscClip);
   mSzClip = aP1Cl-aP0Cl;
}

const Pt2di & cGeomDiscR2::SzDz() const
{
   return mSzDz;
}

const Pt2di & cGeomDiscR2::P0Clip() const
{
   return mP0DiscClip;
}
 
Pt2di cGeomDiscR2::P1Clip()  const
{
  return mP0DiscClip + mSzClip;
}

const Pt2di & cGeomDiscR2::SzClip() const
{
   return mSzClip;
}

const Box2dr & cGeomDiscR2::BoxEngl() const
{
   return mBoxEngl;
}

const Box2dr & cGeomDiscR2::VraiBoxEngl() const
{
   return mVraiBoxEngl;
}


const double & cGeomDiscR2::DeZoom() const
{
   return mDeZoom;
}

void cGeomDiscR2::Show(const std::string  & aMes) const
{
   cout  << aMes
        << " Z " << mDeZoom
        << " SzCl " <<  SzClip() << "\n";
}

const Pt2dr & cGeomDiscR2::P0() const {return mP0;}
const Pt2dr & cGeomDiscR2::P1() const {return mP1;}


/*****************************************/
/*                                       */
/*            cGeomDiscFPx               */
/*                                       */
/*****************************************/


cGeomDiscFPx::cGeomDiscFPx
(
    const  cAppliMICMAC & anAppli
) :
   cGeomDiscR2 (anAppli)
{
}


/*************************************************************************/
/*       Initialise les parametres  definissant la geometrie terrain     */
/*   a partir des parametre de  <Section_Terrain> , voir ParamMICMAC.xml */
/*   pour d'autres commentaires                                          */
/*************************************************************************/
typedef std::list<cListePointsInclus> tLPI;
typedef std::list<Pt2dr> tLPt;

void cGeomDiscFPx::PostInit()
{
if (MPD_MM())
{
    std::cout << "VOIR===PB Z<0 en Prof Champs \n";
}


  cFileOriMnt * aFileExt = 0;
  if (mAp->FileOriMnt().IsInit())
  {
     aFileExt = new cFileOriMnt 
                    (
                             StdGetObjFromFile<cFileOriMnt>
                             (
                                  mAp->WorkDir() + mAp->FileOriMnt().Val(),
                                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                  "FileOriMnt",
                                  "FileOriMnt"
                             )
                     );
      //aFileEx
  }

  //  Calcul de la resolution terrain et, si possible,
  //  de la paralaxe moyenne a partir des geometrie de
  //  prise de vue
  mDimPx = mAp->DimPx();

  mResol = 0.0;
  for (int aD=0 ; aD<mDimPx ; aD++)
      mV0Px[aD] = 0.0;

  int   aNbPxMoyGot = 0;
  int   aNbCentreGot = 0;
  Pt3dr aCentre(0,0,0);

  for (int aK=0 ; aK<mDimPx; aK++)
      mRatioResAltPlani[aK] =  1e9;
  
  cSetName * aSelName = 0;
  if (mAp->Planimetrie().IsInit())
      aSelName = mAp->ICNM()->KeyOrPatSelector(mAp->FilterEstimTerrain().Val());

  int aNbResolGot=0;
  for (tCsteIterPDV itFI=mAp->PdvBegin(); itFI!=mAp->PdvEnd(); itFI++)  
  {
      if ((aSelName==0) || (aSelName->IsSetIn((*itFI)->Name())))
      {
         if (mAp->Anam() || mAp->RC())
         {
              aNbPxMoyGot++;
              for (int aD=0 ; aD<mDimPx ; aD++)
               {
                   mV0Px[aD] += 0;
               }

         }
         else
         {
            double aPx[theDimPxMax];
            if ((*itFI)->Geom().GetPxMoyenne_Euclid(aPx))
            {
               aNbPxMoyGot ++;
               for (int aD=0 ; aD<mDimPx ; aD++)
               {
                   mV0Px[aD] += aPx[aD];
               }
           }  
         }

         double aResol = (*itFI)->Geom().GetResolMoyenne_Euclid();
         mResol  += aResol;
         aNbResolGot++;
         double aRatio[theDimPxMax];
         if ((*itFI)->Geom().GetRatioResolAltiPlani(aRatio))
         {
           for (int aK=0; aK<mDimPx ; aK++)
              ElSetMin(mRatioResAltPlani[aK],aRatio[aK]);
         }

         if ((*itFI)->Geom().HasCentre())
         {
              aNbCentreGot ++;
              Pt3dr aC =  (*itFI)->Geom().CentreRestit();
              aCentre = aCentre + aC;
         }
      }
  }


  if (mAp->RatioAltiPlani().IsInit())
     mRatioResAltPlani[0] = mAp->RatioAltiPlani().Val();


  for (int aK=0; aK<mDimPx ; aK++)
  {
      // std::cout << "RRRRAAPP " << mRatioResAltPlani[aK] << "\n"; getchar();
      ELISE_ASSERT(mRatioResAltPlani[aK]<1e9,"Pas trouve mRatioResAltPlani");
  }


  if (aNbPxMoyGot)
  {
     for (int aD=0 ; aD<mDimPx ; aD++)
     {
         mV0Px[aD] /= aNbPxMoyGot;
     }
  }
  ELISE_ASSERT(aNbResolGot," Resolution pas trouvee");
  mResol /= aNbResolGot;
  if (aFileExt)
  {
       mResol = ElAbs(aFileExt->ResolutionPlani().x);
       //  Attention -(-x) != x  avec les flottant 
       ELISE_ASSERT
       (
            ElAbs(mResol- ElAbs(aFileExt->ResolutionPlani().y)) < (1e-8 * mResol),
           "Resol incoherente in FileOriMnt::ResolutionAlti"
       );
  }
  else if (mAp->Planimetrie().IsInit())
  {
     if  (mAp->RatioResolImage().IsInit())
         mResol *= mAp->RatioResolImage().Val();
     if  (mAp->ResolutionTerrain().IsInit())
         mResol = mAp->ResolutionTerrain().Val();
  } 


  //  Calcul des intervalles de paralaxe
  if (mAp->ModeGeomMEC() == eGeomMECIm1)
  {
     //ELISE_ASSERT(aNbPxMoyGot==0,"Redondance dans PxInit eGeomMECIm1");
     ELISE_ASSERT
     (
          !mAp->IntervAltimetrie().IsInit(),
          "IntervAltimetrie initialise en GeomIm1"
     );
     ELISE_ASSERT
     (
          mAp->IntervParalaxe().IsInit(),
          "IntervParalaxe Non initialise en GeomIm1"
     );

     if (
              (mAp->UseProfInVertLoc().Val())
           && (mAp->ExpA2Mm().ProfInVertLoc().IsInit())
        )
     {
        mV0Px[0] = mAp->ExpA2Mm().ProfInVertLoc().Val();
     }
     else if (mAp->Px1Moy().IsInit())
     {
        mV0Px[0] = mAp->Px1Moy().Val();
     }
     else
     {
         ELISE_ASSERT
         (
             aNbPxMoyGot!=0,
             "Impossible de determiner la paralaxe moyenne"
         );
     }



     if (mAp->Px2Moy().IsInit())
     {
         mV0Px[1] = mAp->Px2Moy().Val();
     }
     else
     {
         ELISE_ASSERT
         (
             aNbPxMoyGot!=0,
             "Impossible de determiner la paralaxe moyenne"
         );
     }
     mEcPxInit[0] = mAp->Px1IncCalc() + mAp->Px1PropProf().Val() * ElAbs(mV0Px[0]);
     mEcPxInit[1] = mAp->Px2IncCalc().ValWithDef(0);


     

     mEcPxZone[0] = mAp->Px1IncZonage().ValWithDef( 0);
     mEcPxZone[1] = mAp->Px2IncZonage().ValWithDef( 0);
      
  }
  else if (mAp->ModeGeomMEC() == eGeomMECTerrain)
  {
     ELISE_ASSERT
     (
          mAp->IntervAltimetrie().IsInit(),
          "IntervAltimetrie non initialise en GeomMECTerrain"
     );
     ELISE_ASSERT
     (
          !mAp->IntervParalaxe().IsInit(),
          "IntervParalaxe initialise en GeomMECTerrain"
     );


     if (
              (mAp->UseProfInVertLoc().Val())
           && (mAp->ExpA2Mm().ProfInVertLoc().IsInit())
        )
     {
        mV0Px[0] = mAp->ExpA2Mm().ProfInVertLoc().Val();
     }
     else if (mAp->ZMoyen().IsInit())
     {
         mV0Px[0] = mAp->ZMoyen().Val();
     }
     else
     {
         if (aNbPxMoyGot==0)
         {
            if (mAp->MNT_Init().IsInit())
            {
               Tiff_Im aMnt = Tiff_Im::StdConvGen(mAp->WorkDir()+mAp->MNT_Init_Image(),1,true,true);
               cFileOriMnt  aOriInit = mAp->GetOri(mAp->WorkDir()+mAp->MNT_Init_Xml());
               double aNb,aZ;
               ELISE_COPY
               (
                 aMnt.all_pts(),
                 Virgule(aMnt.in(),1),
                 Virgule(sigma(aZ),sigma(aNb))
               );
               mV0Px[0] =   (aZ /aNb)*aOriInit.ResolutionAlti() 
                          + aOriInit.OrigineAlti();
            }
            else
            {
               ELISE_ASSERT
               (
                   false,
                   "Impossible de determiner la paralaxe moyenne"
               );
            }
         }
     }

     mEcPxInit[0]  = mAp->ZIncCalc();
     if (mAp->ZIncIsProp().ValWithDef(false))
     {
          ELISE_ASSERT(aNbCentreGot,"Cannot get Centre with ZIncIsProp");
          Pt3dr aC = aCentre/double(aNbCentreGot);

          double aDistSol = ElAbs(aC.z-mV0Px[0]);
          mEcPxInit[0]  *= aDistSol ;
     }

     mEcPxZone[0] = mAp->ZIncZonage().ValWithDef(0);
  }


  for (int aK=0 ; aK<theDimPxMax ; aK++)
  {
       mEcPxInitPlus[aK] = mEcPxInitMoins[aK] = mEcPxInit[aK];
  }


  if (aFileExt)
  {
            mV0Px[0] = aFileExt->OrigineAlti();
  }
  else if (mAp->InversePx())
  {
      if (mAp->IntervSpecialZInv().IsInit())
      {
           double aPxMin =  1/(mV0Px[0]*mAp->IntervSpecialZInv().Val().MulZMax());
           double aPxMax =  1/(mV0Px[0]*mAp->IntervSpecialZInv().Val().MulZMin());

            mV0Px[0] = 1/mV0Px[0];
            mEcPxInitPlus[0] = aPxMax - mV0Px[0];
            mEcPxInitMoins[0] = mV0Px[0] -aPxMin;

            ELISE_ASSERT((mEcPxInitPlus[0]>0) && (mEcPxInitMoins[0]>0),"Error in IntervSpecialZInv");


           // mV0Px[0] =     (aPxMin+aPxMax)/2.0;
           // mEcPxInit[0] = (aPxMax-aPxMin)/2.0;
// std::cout << "jjjtPXA " << 1/mV0Px[0]  << " Resol " << mEcPxInitPlus[0] << " " << mEcPxInitMoins[0]   << "\n";
// getchar();
      }
      else
      {
        double aPxMin =  1/(mV0Px[0]+mEcPxInit[0]);
        mV0Px[0] = 1/mV0Px[0];
        mEcPxInit[0] = mV0Px[0] - aPxMin;
      }
      mEcPxZone[0] *= ElSquare(mV0Px[0]);
  }

  ELISE_ASSERT(mEcPxInit[0]>=0, "mEcPxInit[0]<0");



  // Les boites terrain sont definie en prenant les seconde
  // plus petite (resp. plus grande) valeur, puisque de toute facon
  // on ne peut pas correler avec une seule image
  cGetSec<double> cSecXMin;
  cGetSec<double> cSecXMax;
  cGetSec<double> cSecYMin;
  cGetSec<double> cSecYMax;


// Ne tient pas compte du Full Im1, au cas ou la boite est vide

  cGetSec<double> cVraiSecXMin;
  cGetSec<double> cVraiSecXMax;
  cGetSec<double> cVraiSecYMin;
  cGetSec<double> cVraiSecYMax;


  double  aPxMinZone[theDimPxMax];
  double  aPxMaxZone[theDimPxMax];
  for (int aD=0; aD<mDimPx ; aD++)
  {
      aPxMinZone[aD] = mV0Px[aD] -mEcPxZone[aD];
      aPxMaxZone[aD] = mV0Px[aD] +mEcPxZone[aD];
  }

// std::cout << "HHHHHH  " << aPxMinZone[0] << " " << aPxMaxZone[0] << "\n"; getchar();
  for (tCsteIterPDV itFI=mAp->PdvBegin(); itFI!=mAp->PdvEnd(); itFI++)  
  {
  //std::cout << (*itFI)->Name() << "  HHHHHhhYY\n";// getchar();
      if ((aSelName==0) || (aSelName->IsSetIn((*itFI)->Name())))
      {
           Box2dr aVraiE = (*itFI)->Geom().EmpriseTerrain(mV0Px,mV0Px,0);

// std::cout << "BOX-EM-TER " << aVraiE._p0 << " " << aVraiE._p1 << "\n";



  // std::cout << (*itFI)->Name() << " " << aVraiE._p0 << aVraiE._p1 << "\n";// getchar();

           cVraiSecXMin.Add( aVraiE.P0().x);
           cVraiSecXMax.Add(-aVraiE.P1().x);
           cVraiSecYMin.Add( aVraiE.P0().y);
           cVraiSecYMax.Add(-aVraiE.P1().y);
	   // Facon la + simple de forcer l'emprise du "terrain" a etre egale a image 1
           Box2dr anEmpr = (*itFI)->Geom().EmpriseTerrain(aPxMinZone,aPxMaxZone,0);
	   if (mAp->FullIm1())
           {
// anEmpr = (*(mAp->PdvBegin()))->Geom().EmpriseTerrain(aPxMinZone,aPxMaxZone,0);
             ELISE_ASSERT
             (	   
                   (aSelName==0) || (aSelName->IsSetIn((*(mAp->PdvBegin()))->Name())),
                 "Im1 Pas selectionne en Full Im1 !! "
             );
             ELISE_ASSERT
             (	   
                    ( mAp->GeomMNT() !=eGeomMNTCarto) 
                 && ( mAp->GeomMNT() !=eGeomMNTEuclid),
                 "Geometrie incompaible avec FulIm1"
             );
             Pt2di aSz = (*(mAp->PdvBegin()))->SzIm();
             anEmpr = Box2dr(Pt2dr(0,0),Pt2dr(aSz));
           }
           cSecXMin.Add( anEmpr.P0().x);
           cSecXMax.Add(-anEmpr.P1().x);
           cSecYMin.Add( anEmpr.P0().y);
           cSecYMax.Add(-anEmpr.P1().y);
     }
  }
  mVraiBoxEngl = Box2dr
                    (
	                 Pt2dr(  cVraiSecXMin.Sec(), cVraiSecYMin.Sec()),
                         Pt2dr( -cVraiSecXMax.Sec(),-cVraiSecYMax.Sec())
                    );
		    /*
  double  aLarg = ElMin
                  (
		      ElAbs(aVraiBox._p0.x-aVraiBox._p1.x),
		      ElAbs(aVraiBox._p0.y-aVraiBox._p1.y)
                  );
  */
  mBoxEngl = Box2dr
             (
                    Pt2dr(  cSecXMin.Sec(), cSecYMin.Sec()),
                    Pt2dr( -cSecXMax.Sec(),-cSecYMax.Sec())
             );

  // La boite terrain est definie soit explicitement par l'utilisateur
  // soit comme la boite englobante de la zone objet contenant au - 2 images
 
  bool isBoxInit = false;
  Box2dr aBox;
  if  (
           (mAp->Planimetrie().IsInit())
       &&  (mAp->BoxTerrain().IsInit())
      )
  {
     aBox = mAp->BoxTerrain().Val();
     isBoxInit = true;
  }

  if ((!isBoxInit) && (mAp->FileBoxMasqIsBoxTer() != ""))
  {
     if (!ELISE_fp::exist_file(mAp->FileBoxMasqIsBoxTer()))
     {
          Tiff_Im aFileMT = Tiff_Im::StdConvGen(mAp->WorkDir()+mAp->MasqueTerrain().Val().MT_Image(),1,true,true);
          int aXMin,aXMax,aYMin,aYMax;
          ELISE_COPY
          (
              select(aFileMT.all_pts(),aFileMT.in_bool()),
              Virgule(FX,FY),
              Virgule
              (
                  VMin(aXMin)|VMax(aXMax),
                  VMin(aYMin)|VMax(aYMax)
              )
          );
          cFileOriMnt aFileOri = mAp->GetOri(mAp->WorkDir()+mAp->MasqueTerrain().Val().MT_Xml());

         Pt2dr aOri = aFileOri.OriginePlani();
         Pt2dr aRes = aFileOri.ResolutionPlani();

         Pt2dr aP0 = aOri + Pt2dr(aRes.x*aXMin,aRes.y*aYMin);
         Pt2dr aP1 = aOri + Pt2dr(aRes.x*aXMax,aRes.y*aYMax);

        cBoxMasqIsBoxTer aBMT;
        aBMT.Box() = Box2dr(aP0,aP1);
        MakeFileXML(aBMT,mAp->FileBoxMasqIsBoxTer());
     }
     cBoxMasqIsBoxTer aBMT = StdGetObjFromFile<cBoxMasqIsBoxTer>
                             (
                                  mAp->FileBoxMasqIsBoxTer(),
                                  StdGetFileXMLSpec("ParamMICMAC.xml"),
                                  "BoxMasqIsBoxTer",
                                  "BoxMasqIsBoxTer"
                             );
      
     aBox = aBMT.Box();
     isBoxInit = true;
  }


  if (aFileExt)
  {
        isBoxInit = true;
        Pt2dr aPtOri = aFileExt->OriginePlani();
        Pt2dr aPtRes = aFileExt->ResolutionPlani();
        Pt2dr aNb =  Pt2dr(aFileExt->NombrePixels());
        aBox = Box2dr(aPtOri,aPtOri+aPtRes.mcbyc(aNb));
  }
  else if (mAp->Planimetrie().IsInit())
  {
     const tLPI & aLPI = mAp->ListePointsInclus();
     for (tLPI::const_iterator itLPtI=aLPI.begin();itLPtI!=aLPI.end();itLPtI++)
     {
         const char * aMes = "Impossible de trouver Im dans ListePointsInclus";
         const cPriseDeVue * aPDV = 0;
	 bool aCasTerrain = false;

	 if (itLPtI->Im() != "Terrain")
	 {
	     aPDV =mAp->PDVFromName(itLPtI->Im(),aMes);
	 }
	 else
	 {
	     aPDV = mAp->PDV1();
	     aCasTerrain = true;
	 }
	     
         const tLPt & aLPt = itLPtI->Pt();
         for (tLPt::const_iterator itP=aLPt.begin();itP!=aLPt.end();itP++)
         {
	    Pt2dr aPIm = *itP;
	    if (aCasTerrain)
	    {
	        aPIm = aPDV->Geom().GeoTerrainIntrinseque()->Objet2ImageInit_Euclid(aPIm,mV0Px);
	        aPIm = Pt2dr(round_ni(aPIm)); // A priori du pt de vue operationnelle c'est mieux de pas avoir de mm
	    }
            Pt2dr aPTer = aPDV->Geom().ImageAndPx2Obj_Euclid(aPIm,mV0Px);
            if (isBoxInit)
            {
                aBox._p0.SetInf(aPTer);
                aBox._p1.SetSup(aPTer);
            }
            else
            {
                 aBox = Box2dr(aPTer,aPTer);
            }
            isBoxInit = true;
         }
     }
  }

  if (!isBoxInit)
  {
      aBox = mBoxEngl;
  }
  mP0 = aBox._p0;
  mP1 = aBox._p1;
  mBoxEngl  = aBox;  // A priori redondance entre les 2
  if (mAp->ModeGeomMEC()==eGeomMECIm1)
  {
      mP0 = Pt2dr(round_ni(mP0));
      mP1 = Pt2dr(round_ni(mP1));
  }

  for (tCsteIterPDV itFI=mAp->PdvBegin(); itFI!=mAp->PdvEnd(); itFI++)  
  {
      (*itFI)->Geom().InitAnam(mResol,mBoxEngl);
  }

  cGeomDiscR2::SetDeZoom(1.0);
  REAL aV1[theDimPxMax] = {1.0,1.0};
  SetStep(aV1);

  mCorrectDerivee= 1.0;
  if (mAp->InversePx())
  {
      mCorrectDerivee = -1 / ElSquare(mV0Px[0]);
      double aZ0 = mV0Px[0];
      double aZ1 = aZ0+ mStepAbs[0];
      cout << "----STEP en Z " << (1/aZ0-1/aZ1) << " " <<  mV0Px[0] << " " <<  mStepAbs[0] << "\n";
      // cout << " RESOL " <<  mResol << "\n";
  }



  delete aFileExt;
}

double  cGeomDiscFPx::CorrectDerivee() const
{
   return mCorrectDerivee;
}

int cGeomDiscFPx::GetEcartInitialGen(double aPas,int aKPx,double anEcart) const
{
   return  round_ni(anEcart/(aPas*mResolDz* mRatioResAltPlani[aKPx]));
}

int cGeomDiscFPx::GetEcartInitialPlus(double aPas,int aKPx) const
{
    return GetEcartInitialGen(aPas,aKPx,mEcPxInitPlus[aKPx]);
}

int cGeomDiscFPx::GetEcartInitialMoins(double aPas,int aKPx) const
{
    return GetEcartInitialGen(aPas,aKPx,mEcPxInitMoins[aKPx]);
}


       /*
void  cGeomDiscFPx::GetEcartInt(int * aIEc) const
{
   for (int aD=0 ; aD<mDimPx ; aD++)
   {
       aIEc[aD] = round_ni(mEcPxInit[aD]/mStepAbs[aD]);
       std::cout  << " REL : " <<  mStepRel[aD] 
                  << "ABS : " << mStepAbs[aD] << ":: GetEcartInt " << aD << " = " <<  aIEc[aD]  << "\n"; getchar();
   }
}

		  */


void cGeomDiscFPx::SetStep(const REAL * aVStep)
{
   for (int aD=0 ; aD<mDimPx ; aD++)
   {
       mStepRel[aD] = aVStep[aD] ;
       mStepAbs[aD] = mStepRel[aD] * mResolDz * mRatioResAltPlani[aD];
       // std::cout <<  mStepRel[aD] << " " << mStepAbs[aD] << "\n";
   }
}

void cGeomDiscFPx::SetDeZoom(REAL aDz)
{
   cGeomDiscR2::SetDeZoom(aDz);
   SetStep(mStepRel);
}


double cGeomDiscFPx::RatioResAltiPlani() const
{
      return mRatioResAltPlani[0];
}

double cGeomDiscFPx::PasPx0() const
{
    return  mStepAbs[0];
}

double cGeomDiscFPx::PasPxRel0() const
{
    return  mStepRel[0];
}

void cGeomDiscFPx::PxDisc2PxReel(REAL * aPxR,const int * aPxD) const
{
    for (int aD=0 ; aD<mDimPx ; aD++)
    {
          aPxR[aD] = mV0Px[aD] + aPxD[aD] * mStepAbs[aD];
    }
}

void cGeomDiscFPx::PxDisc2PxReel(REAL * aPxR,const double * aPxD) const
{
    for (int aD=0 ; aD<mDimPx ; aD++)
    {
        aPxR[aD] = mV0Px[aD] + aPxD[aD] * mStepAbs[aD];
    }
}


void cGeomDiscFPx::PxReel2PxDisc(REAL * aPxD,const double * aPxR) const
{
    for (int aD=0 ; aD<mDimPx ; aD++)
    {
        aPxD[aD] = (aPxR[aD]-mV0Px[aD]) / mStepAbs[aD];
    }
}


const REAL *  cGeomDiscFPx::EcPxZone() const
{
    return mEcPxZone;
}

const REAL *  cGeomDiscFPx::V0Px() const
{
    return mV0Px;
}

Pt2di cGeomDiscFPx::NbPixel() const { return mSzDz; }
double cGeomDiscFPx::OrigineAlti() const { return  mV0Px[0]; }
double cGeomDiscFPx::ResolutionAlti() const { return  mStepAbs[0]; }

void cGeomDiscFPx::SetOriResolPlani(Pt2dr & aOriP,Pt2dr & aResolP) const
{
  aOriP = RDiscToR2(Pt2dr(0,0));
  aResolP = RDiscToR2(Pt2dr(1,1))-aOriP;
}


void cGeomDiscFPx::RemplitOri(cFileOriMnt & aFOM) const
{
  aFOM.NombrePixels()    = NbPixel();
  SetOriResolPlani(aFOM.OriginePlani() ,aFOM.ResolutionPlani());
  aFOM.OrigineAlti()     = OrigineAlti();
  aFOM.ResolutionAlti()  = ResolutionAlti();

/*
  aFOM.NombrePixels()    = mSzDz;

  Pt2dr aP0 = RDiscToR2(Pt2dr(0,0));
  Pt2dr aP1 = RDiscToR2(Pt2dr(1,1));


  aFOM.OriginePlani()    = aP0;
  aFOM.ResolutionPlani() = aP1-aP0;
  aFOM.OrigineAlti()     = mV0Px[0];
  aFOM.ResolutionAlti()  = mStepAbs[0];
*/
}

/*****************************************/
/*                                       */
/*            cLineariseProj             */
/*                                       */
/*****************************************/

// On a privilegie la simplicite et la fiabilite sur la
// rapidite (il est possible de ne faire que 3 evaluation
// de la proj image et une seule de la proj terrain)


void cLineariseProj::InitLP
     (
          Box2di               aRBT,  // Bornes incluses
          int *                aPxI,
          const cGeomImage &   aGeomIm,
          const cGeomDiscFPx & aGeomTer,
          const cEtapeMecComp& anEtape,
          const cLoadTer &     aLT,
          const Pt2di   &      aDecalRedr,
          int                  aSurEchW

     )
{
   mDecalRedr =  aDecalRedr;
   mUseDer = anEtape.UseGeomDerivable() ;
   // Avant 
   // mUseDer = anEtape.UseGeomDerivable() || (aSurEchW!=1);
   //  JE vois pas pourquoi, mais c'est pas non + utilise souvent
   //  je met un garde fou, a voir si on passe par la
   if ((! anEtape.UseGeomDerivable()) && (aSurEchW!=1))
   {
       ELISE_ASSERT
       (
          false,
          "Cas a elucider sur anEtape.UseGeomDerivable / aSurEchW"
       );
   }

   mGeomIm = & aGeomIm;
   mGeomTer = & aGeomTer;
   mLT      = & aLT;
   aGeomTer.PxDisc2PxReel(mRPx,aPxI);
   mSzRas = aRBT._p1 - aRBT._p0;
   mP0Ras = aRBT._p0;
   mCurRas = Pt2di(0,0);
   mDimPx  = aLT.NbPx();

   if (mUseDer)
   {
      Pt2dr aRasPCTer = (Pt2dr(aRBT._p0)+Pt2dr(aRBT._p1))/2.0;

      REAL aPas = 0.33;
      REAL amplX = ElMax(0.5,double(aRBT._p1.x - aRBT._p0.x));
      REAL amplY = ElMax(0.5,double(aRBT._p1.y - aRBT._p0.y));
      REAL aPasX  = amplX * aPas;
      REAL aPasY  = amplY * aPas;


      Pt2dr aP0X = ProjExacte(aRasPCTer-Pt2dr(aPasX,0));
      Pt2dr aP1X = ProjExacte(aRasPCTer+Pt2dr(aPasX,0));
      Pt2dr aP0Y = ProjExacte(aRasPCTer-Pt2dr(0,aPasY));
      Pt2dr aP1Y = ProjExacte(aRasPCTer+Pt2dr(0,aPasY));


      mDerX = (aP1X-aP0X) / (2*aPasX);
      mDerY = (aP1Y-aP0Y) / (2*aPasY);
      mCentre = (aP0X+aP1X+aP0Y+aP1Y)/4.0;
      mDebLigne  =  mCentre - mDerX*amplX*0.5 -mDerY*amplY*0.5;
      mCurIm = mDebLigne;
   }
   else
   {
      mCurIm = ProjExacte(mCurRas+mP0Ras);
   }

   if (aSurEchW !=1)
   {
       mSzRas = mSzRas * aSurEchW;
       mDerX = mDerX / double(aSurEchW);
       mDerY = mDerY / double(aSurEchW);
   }
}
      // std::cout << mDecalRedr << aP << "---->PxK=" << mRPx[aK] << " " << aPxRedr[aK] <<"\n";
      // getchar();

#define Definir_cLineariseProj_ProjExacte(aTypePt)\
Pt2dr cLineariseProj::ProjExacte(const aTypePt & aP)\
{\
   static double  aPxRedr[theDimPxMax];\
   static double  aPxTot[theDimPxMax];\
   mLT->Redressement(aPxRedr,aP-aTypePt(mDecalRedr));\
   for (int aK=0 ; aK <mDimPx ; aK++)\
   {\
      aPxTot[aK] = mRPx[aK] + aPxRedr[aK];\
   }\
   return mGeomIm->CurObj2Im(mGeomTer->RDiscToR2(Pt2dr(aP)),aPxTot);\
}\

Definir_cLineariseProj_ProjExacte(Pt2dr)
Definir_cLineariseProj_ProjExacte(Pt2di)



bool cLineariseProj::Continuer() const
{
   return mCurRas.y <= mSzRas.y;
}

void cLineariseProj::NexStep()
{
   if (mCurRas.x ==  mSzRas.x)
   {
      mCurRas.x = 0;
      mCurRas.y++;
      if (mUseDer)
      {
         mDebLigne +=mDerY;
         mCurIm=mDebLigne;
      }
      else 
      {
         if (Continuer())
            mCurIm = ProjExacte(mCurRas+mP0Ras);
      }
   }
   else
   {
      mCurRas.x++;
      if (mUseDer)
         mCurIm += mDerX;
      else 
         mCurIm = ProjExacte(mCurRas+mP0Ras);

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
