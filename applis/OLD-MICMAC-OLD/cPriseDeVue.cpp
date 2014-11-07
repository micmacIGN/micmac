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

static Pt2di aSzTileMasq(1000000,1000000);

/*****************************************/
/*           cModGeomComp                */
/*****************************************/

cModGeomComp::cModGeomComp() :
   mCoeffDilNonE (1.0)
{
}

cModGeomComp::~cModGeomComp() 
{
}

double cModGeomComp::CoeffDilNonE () const
{
   return mCoeffDilNonE;
}

void cModGeomComp::SetCoeffDilNonE(double aCoeff)
{
   mCoeffDilNonE = aCoeff;
}

class cModifGeomAffine : public cModGeomComp
{
   public :
         Pt2dr Modif2GeomInit(const Pt2dr & aP) const 
	 {
	    return mModif2GeomInit(aP);
	 }
         Pt2dr Modif2GeomActu(const Pt2dr & aP) const 
	 {
	    return mModif2GeomActu(aP);
	 }
         cModifGeomAffine(const ElAffin2D & aModif2GeomInit) :
	     mModif2GeomInit (aModif2GeomInit),
	     mModif2GeomActu (mModif2GeomInit.inv())
	  {
	     SetCoeffDilNonE
	     (
	        ( euclid(mModif2GeomInit.IVect(Pt2dr(1,0)))
		+euclid(mModif2GeomInit.IVect(Pt2dr(0,1))) ) / 2.0
             );
	  }
    private :
         ElAffin2D mModif2GeomInit;
         ElAffin2D mModif2GeomActu;
 
};

class cModifGeomPxTr : public cModGeomComp
{
   public :
       virtual ~cModifGeomPxTr() {}
       Pt2dr Modif2GeomInit(const Pt2dr & aP) const
       {
           return aP - mDir*mVPx.getprojR(aP/mSsResol);

       }
       Pt2dr Modif2GeomActu(const Pt2dr & aP) const
       {
           return aP + mDir*mVPx.getprojR(aP/mSsResol);
       }

       cModifGeomPxTr(const cCorrectionPxTransverse & aCor) :
	      mDir       (aCor.DirPx()),
              mVPx       (aCor.ValeurPx()),
	      mSsResol   (aCor.SsResol())
       {
       }
   private :
       Pt2dr                mDir;
       TIm2D<REAL4,REAL8>   mVPx;
       double               mSsResol;
};

/*****************************************/
/*             cPriseDeVue               */
/*****************************************/


cPriseDeVue::cPriseDeVue
(
    const cAppliMICMAC &  anAppli,
    const std::string &   aName,
    cInterfModuleImageLoader * aIMIL,
    int                   aNum,
    const std::string &   aNameGeom,
    const std::list< cModifieurGeometrie > & aLM
) :
   mAppli           (anAppli),
   mIMIL            (aIMIL),
   mName            (aName),
   mNameTif         (StdPrefixGen(mName)+".tif"),
   mNameGeom        (aNameGeom),
   mNum             (aNum),
   mGeom            (0),
   mGeomTerAssoc    (0),
   mSzIm            (-1,-1),
   mLoadIm          (0),
   mNameClassEquiv  (mAppli.NameClassEquiv(mName)),
   mNumEquiv        (-1),
   mNuagePredict    (0)
{

   for 
   (
       std::list<cModifieurGeometrie>::const_iterator itM=aLM.begin();
       itM != aLM.end();
       itM++
   )
   {
       if (
               (! itM->Apply().IsInit())
	    || (itM->Apply().Val()->Match(aName))
	  )
       {
          if ( itM->CropAndScale().IsInit())
	  {
               cCropAndScale aCS = itM->CropAndScale().Val();

	       double aSX = aCS.Scale().Val();
	       Pt2dr aTr  = aCS.Crop().Val();
	       double aSY = aCS.ScaleY().ValWithDef(aSX);

               mVModif.push_back
	       (
                      new cModifGeomAffine
		          (
	                      ElAffin2D(aTr,Pt2dr(aSX,0),Pt2dr(0,aSY))
                          )
               );
	  }
          else if (itM->NamePxTr().IsInit())
	  {
	      cCorrectionPxTransverse aCor=
	          StdGetObjFromFile<cCorrectionPxTransverse>
		  (
		     mAppli.WorkDir() + itM->NamePxTr().Val(),
		     mAppli.NameSpecXML(),
		     "CorrectionPxTransverse",
		     "CorrectionPxTransverse"
		  );
               mVModif.push_back(new cModifGeomPxTr(aCor));
	  }
	  else
	  {
	     ELISE_ASSERT(false,"cModifieurGeometrie");
	  }
       }
   }
}


cPriseDeVue::~cPriseDeVue()
{
    delete mLoadIm;
    delete mGeom;
    delete mIMIL;
}





/*
Fonc_Num cPriseDeVue::ImMasq()
{
}
*/

/*****************************************/
/*       ACCESSEURS                      */
/*****************************************/

const std::string & cPriseDeVue::Name() const
{
    return mName;
}

Pt2di cPriseDeVue::SzIm() const
{
   if (mSzIm == Pt2di(-1,-1))
   {
       mSzIm = Std2Elise(mIMIL->Sz(1));
   }
   return mSzIm;
}

const cLoadedImage & cPriseDeVue::LoadedIm() const
{
   return *mLoadIm;
}

cLoadedImage & cPriseDeVue::LoadedIm()
{
   return *mLoadIm;
}

bool cPriseDeVue::IsMaitre() const
{
   return mIsMaitre;
}

void cPriseDeVue::SetMaitre(bool AvecImMaitre)
{
   mIsMaitre = AvecImMaitre && (mNum==0);
}

cInterfModuleImageLoader * cPriseDeVue::IMIL()
{
   return mIMIL;
}

const cInterfModuleImageLoader * cPriseDeVue::IMIL() const
{
   return mIMIL;
}

const std::string  &  cPriseDeVue::NameClassEquiv() const
{
    return mNameClassEquiv;
}

const int & cPriseDeVue::NumEquiv() const { return mNumEquiv; }
int & cPriseDeVue::NumEquiv() { return mNumEquiv; }
int  cPriseDeVue::Num() const {return mNum;}



/*****************************************/
/*  Gestion du fichier image             */
/*****************************************/


/*****************************************/
/*  Chargement des images                */
/*****************************************/

bool cPriseDeVue::LoadImage
     (
          const cLoadTer& aLT,
          const Pt2di & aSzMaxInGeomTer,
          bool  IsFirstLoaded
     )
{
    delete mLoadIm;
    mLoadIm = 0;
    mCurEtape = mAppli.CurEtape ();

    bool WithRab = (mAppli.PtrVI() == 0);

    mGeom->SetClip
    (
         mAppli.GeomDFPx(),
	 mAppli.PtSzWMarge(),
         // mCurEtape->EtapeMEC().SzW().Val(),
         aLT.PxMin(),
         aLT.PxMax(),
         WithRab ? 50 :0 , 
         WithRab ? 5 :0
    );

    if (mGeom->ClipIsEmpty())
    {
       ELISE_ASSERT(!mIsMaitre,"PriseDeVue::Maitre&&Empty !! ");
       return false;
    }

    // std::cout  << " yyyyyyyyy " <<  mGeom->BoxClip()._p0 <<  mGeom->BoxClip()._p1<< "\n";
    // getchar();

   int aDZ = mCurEtape->DeZoomIm();
   mLoadIm = cLoadedImage::Alloc
             ( 
                 mAppli,
		 *this,
                 Geom(),
                 mGeom->BoxClip(),
                 aSzMaxInGeomTer,
                 mIMIL,
                 aDZ,
                 FileImMasqOfResol(aDZ).in(0),
		 IsFirstLoaded
             );

   delete mNuagePredict;
   mNuagePredict = 0;


   const cTplValGesInit<cNuagePredicteur> & aTNP =  mCurEtape->EtapeMEC().NuagePredicteur();

   if (aTNP.IsInit())
   {
        std::cout << "LOAD NUAGE " << mName << "\n";
        const cNuagePredicteur aNP = aTNP.Val();
        cSetName * aSN = mAppli.ICNM()->KeyOrPatSelector(aNP.Selector());
        if (aSN->IsSetIn(mName))
        {
            double aZoomN = aNP.ScaleNuage() / aDZ;
            std::string aFullNameNuage = mAppli.WorkDir()+ mAppli.ICNM()->Assoc1To1(aNP.KeyAssocIm2Nuage(),mName,true);
            std::string aDirNuage,aNameNuage;
            SplitDirAndFile(aDirNuage,aNameNuage,aFullNameNuage);
            cXML_ParamNuage3DMaille aPN3 = StdGetObjFromFile<cXML_ParamNuage3DMaille>
                                    (
                                        aFullNameNuage,
                                         StdGetFileXMLSpec("SuperposImage.xml"),
                                         "XML_ParamNuage3DMaille",
                                         "XML_ParamNuage3DMaille"
                                     );

            cParamModifGeomMTDNuage aParamModif(aZoomN,I2R(mGeom->BoxClip()));
            // cElNuage3DMaille * anEN  = cElNuage3DMaille::FromParam
            mNuagePredict = cElNuage3DMaille::FromParam
                            (
                                aPN3,
                                aDirNuage,
                                 "",
                                1.0,
                                &aParamModif
                            );
        }
        
       // mNuagePredict
   }
/*
*/

   return true;
}

double  cPriseDeVue::DzOverPredic(const Pt3dr & aP) const
{
   ELISE_ASSERT(mNuagePredict!=0,"No Nuage in cPriseDeVue::DzOverPredic");

   Pt3dr  aPIm = mNuagePredict->Loc_Euclid2ProfAndIndex(aP);
   return aPIm.z  - mNuagePredict->ProfOfIndexInterpolWithDef(Pt2dr(aPIm.x,aPIm.y),aPIm.z);
}

/*****************************************/
/*  Gestion  de la geometrie             */
/*****************************************/

std::string NamePrefOfImages(const cPriseDeVue * aIm)
{
   ELISE_ASSERT(aIm!=0,"Cannot extend correctly a Pattern");
   return StdPrefixGen(aIm->Name());
}

/*
    Les noms de fichiers de geometries sont passes sous forme de "pattern"
    indiquant comment transformer le nom de l'image.

    Soit un fichier de parametre avec :

    <TmpGeom>            = Toto
    <PatternNameGeom>    = titi_%I.tata

    Alors le fichier "UnImage.tif" aura un fichier de geometrie 
    "Toto/titi_UnImage.tata", de meme que le fichier "UnImage".

    Autrement dit "%I" est remplace par le nom de l'image apres
    supression d'une eventuelle extension.

*/

std::string  ExpendPattern
             (
                  const std::string & aPattern,
                  const cPriseDeVue * aIm1,
                  const cPriseDeVue * aIm2
             )
{
   std::string aRes;
   for (const char * aC=aPattern.c_str(); *aC ; aC++)
   {
       if (*aC == '%')
       {
          aC++;
          if (*aC == 'I')
          {
             aRes+=NamePrefOfImages(aIm1);
          }
          else if (*aC == 'J')
          {
             aRes+=NamePrefOfImages(aIm2);
          }
          else
          {
             ELISE_ASSERT(false,"Unkown pattern for image ");
          }
       }
       else
          aRes += *aC;
   }
   return  aRes;
}


const std::string & cPriseDeVue::NameGeom() const
{
       return mNameGeom;
}

cDbleGrid * cPriseDeVue::ReadGridDist() const
{
   // cDbleGrid::cXMLMode aXmlMode;
   if (NameGeom()=="GridDistId")
      return 0;
   // return  new cDbleGrid(aXmlMode,mAppli.FullDirGeom(),NameGeom());
   return cDbleGrid::StdGridPhotogram(mAppli.FullDirGeom()+NameGeom());
}

std::string  cPriseDeVue::NamePackHom(const cPriseDeVue * aPDV2) const
{
   return mAppli.NamePackHom(Name(),aPDV2->Name());
}


ElPackHomologue cPriseDeVue::ReadPackHom(const cPriseDeVue * aPDV2) const
{
   cElXMLTree aTree 
	      ( 
	          mAppli.FullDirGeom()
	       +  NamePackHom(aPDV2)
              );

   return  aTree.GetPackHomologues("ListeCpleHom");
}

CamStenope * cPriseDeVue::GetOri() const
{

   std::string aNG = mAppli.FullDirGeom()+NameGeom();
   CamStenope * aRes = CamStenope::StdCamFromFile(true,aNG.c_str(),mAppli.ICNM());
   mAppli.AnalyseOri(aRes);
   return aRes;
    
}


const cGeomImage & cPriseDeVue::Geom() const
{
   ELISE_ASSERT(mGeom !=0,"Geom Nulle dans cPriseDeVue::Geom") ;
   return *mGeom;
}

Box2dr cPriseDeVue::BoxIm() const
{
   return Box2dr
          (
              Pt2dr(0,0),
              Pt2dr(Std2Elise(IMIL()->Sz(1)))
          );
}


cGeomImage * cPriseDeVue::StdGeomTerrain()
{
    if (mAppli.GeomImages()== eGeomImageOri)
    {
           if (
                   (mAppli.GeomMNT()==eGeomMNTEuclid)
                || (mAppli.GeomMNT()==eGeomMNTFaisceauIm1ZTerrain_Px1D)
                || (mAppli.GeomMNT()==eGeomMNTFaisceauIm1ZTerrain_Px2D)
              )
              return  cGeomImage::Geom_Terrain_Ori(mAppli,*this,SzIm(),GetOri());
           else if (mAppli.GeomMNT()==eGeomMNTCarto)
              return  cGeomImage::Geom_Carto_Ori(mAppli,*this,SzIm(),GetOri());
    }
    else if (mAppli.GeomImages()== eGeomImageGrille)
    {
              return  cGeomImage::GeomImage_Grille(mAppli,*this,SzIm(),mAppli.FullDirGeom()+NameGeom());
    }
    else if (mAppli.GeomImages()== eGeomImageRTO)
    {
        return  cGeomImage::GeomImage_RTO(mAppli,*this,SzIm(),mAppli.FullDirGeom()+NameGeom());
    }
#ifdef __USE_ORIENTATIONMATIS__
    else if (mAppli.GeomImages()== eGeomImageCON)
    {
        return cGeomImage::GeomImage_CON(mAppli,*this,SzIm(),mAppli.FullDirGeom()+NameGeom());
    }
#endif
    else if (mAppli.GeomImages()==eGeomImageModule)
    {
	   return cGeomImage::GeomImage_Module
                  (
                       mAppli,*this,SzIm(),
                       mAppli.FullDirGeom()+NameGeom(),
                       mAppli.NomModule(),mAppli.NomGeometrie()
                   );
    }
    ELISE_ASSERT(false,"Incoherence dans cPriseDeVue::StdGeomTerrain");
    return 0;
}

cGeomImage & cPriseDeVue::Geom()
{
   if (mGeom !=0) 
      return *mGeom;

  int aDimFs=-1;
  bool aFaiscPrCh = false;
  bool aFaiscZTerrain = false;
  bool isSpherik = false;

  if (mAppli.GeomMNT() == eGeomMNTFaisceauIm1PrCh_Px2D)
  {
     aDimFs =2;
     aFaiscPrCh = true;
  }
  if (mAppli.GeomMNT() == eGeomMNTFaisceauIm1PrCh_Px1D)
  {
     aDimFs =1;
     aFaiscPrCh = true;
  }
  if (mAppli.GeomMNT() == eGeomMNTFaisceauPrChSpherik)
  {
     aDimFs =1;
     aFaiscPrCh = true;
     isSpherik = true;
  }



  if (mAppli.GeomMNT() == eGeomMNTFaisceauIm1ZTerrain_Px2D)
  {
     aDimFs =2;
     aFaiscZTerrain = true;
  }
  if (mAppli.GeomMNT() == eGeomMNTFaisceauIm1ZTerrain_Px1D)
  {
     aDimFs =1;
     aFaiscZTerrain = true;
  }

  if (aFaiscPrCh)
  {
     /* La notion de profondeur de champs, et surtout l'interet de l'echantinnoner
        en 1/Z est etroitement liee a la prise de vue conique.  */
     if   (mAppli.GeomImages()== eGeomImageOri)
     {
          if (mNum == 0)
          {
             mGeom = cGeomImage::GeomImage_Id_Ori(mAppli,*this,SzIm(),aDimFs,GetOri(),isSpherik);
             // std::cout << "WWWWWWWWWWWWWWWw\n"; getchar();
          }
          else
             mGeom = cGeomImage::GeomImage_Faisceau
                     (mAppli,*this,SzIm(),aDimFs,GetOri(),mAppli.PDV1()->GetOri(),isSpherik);
     }
     else
     {
     ELISE_ASSERT
     (
          false,
          "eGeomMNTFaisceauIm1PrCh necessit prise de vue conique"
     );
     }
  }
  else if (aFaiscZTerrain)
  {
       mGeomTerAssoc = StdGeomTerrain();
       if (mNum == 0)
          mGeom = cGeomImage::GeomFaisZTerMaitre(mAppli,*this,SzIm(),aDimFs,mGeomTerAssoc);
       else
          mGeom = cGeomImage::GeomFaisZTerEsclave
                  (
                        mAppli,
                        *this,
                        SzIm(),
                        aDimFs,
                        mGeomTerAssoc,
                        mAppli.PDV1()->mGeomTerAssoc
                   );
  }
  else if (mAppli.ModeGeomMEC() == eGeomMECIm1)
  {
     int aDim = 2;
     if (mAppli.GeomImages() == eGeomImage_EpipolairePure)
        aDim = 1;

     if (mNum == 0)
     {
        // En Geometrie image, les coordonnees
        // objets sont toujours confondues avec
        // celles de la premiere image
        mGeom = cGeomImage::GeomId(mAppli,*this,SzIm(),aDim);
     }
     else if ((mNum==1) || (mAppli.ModeAlloc()==eAllocAM_Surperposition))
     {
         if (mAppli.GeomImages() == eGeomImageDHD_Px)
         {
            mGeom = cGeomImage::Geom_DHD_Px
	            (
	                 mAppli,
                         *this,
	                 SzIm(),
	                 mAppli.PDV1()->ReadGridDist(),
	                 this->ReadGridDist(),
	                 mAppli.PDV1()->ReadPackHom(this),
			 2
	            );
         }

         if (mAppli.GeomImages() == eGeomImage_Hom_Px)
         {
             mGeom = cGeomImage::Geom_DHD_Px
	             (
	                 mAppli,
                         *this,
	                 SzIm(),
	                 0,
	                 0,
	                 mAppli.PDV1()->ReadPackHom(this),
			 2
	             );
         }

         if (
	         (mAppli.GeomImages() == eGeomImage_Epip)
	      || (mAppli.GeomImages() == eGeomImage_EpipolairePure)
	     )
         {
             mGeom = cGeomImage::Geom_DHD_Px
	             (
	                 mAppli,
                         *this,
	                 SzIm(),
	                 0,
	                 0,
	                 ElPackHomologue(),
			 aDim
	             );
         }
     }
  }
  else if (mAppli.ModeGeomMEC() == eGeomMECTerrain)
  {
       mGeom = StdGeomTerrain();
  }
  else if (mAppli.AucuneGeom())
   {
      return *cGeomImage::Geom_NoGeomImage(mAppli,*this,SzIm());
   }

  if (mGeom==0)
  {
     ELISE_ASSERT(false,"Cannot determine Geometrie");
  }

  mGeom->PostInitVirtual(mVModif);

   return *mGeom;
}


/*  -------------------------------------
     Gestion des masques
  ------------------------------------- */
    
std::string cPriseDeVue::OneNameMasq
            (
                const std::list<cOneMasqueImage> & aList
            ) const
{
   std::string aRes;
   for 
   (
        std::list<cOneMasqueImage>::const_iterator itM = aList.begin();
        itM !=  aList.end();
        itM++
   )
   {
      cElRegex_Ptr  anAutom = itM->PatternSel();
      if (anAutom->Match(mName))
      {
           bool aReplace =  anAutom->Replace(itM->NomMasq());
           if (! aReplace)
           {
               std::cout << "Autom = " << anAutom->NameExpr() << "\n";
               ELISE_ASSERT
               (
                    false,
                    "Erreur dans cPriseDeVue::OneNameMasq"
               );
           }
           aRes = anAutom->LastReplaced();
      }
   }
   ELISE_ASSERT
   (
         aRes != "",
         " Aucun match sur OneMasqueImage"
   );
   return aRes;
}

Fonc_Num  cPriseDeVue::FoncMasq(std::string  & aName) const
{
   aName = "M" +ToString(SzIm().x) +"x"+ ToString(SzIm().y) ;
   Pt2di aSz = Std2Elise(mIMIL->Sz(1));
   int aBrd = mAppli.BordImage().Val();

   Fonc_Num aFRes = inside(Pt2di(aBrd,aBrd),aSz-Pt2di(aBrd,aBrd));
   if (mAppli.HasVSNI())
   {
      aName = "_VSNI"+StdPrefix(mNameTif);
      std::string aName = mIMIL->NameTiffImage();
      Tiff_Im aTif(aName.c_str());
      aFRes = (aTif.in()!=mAppli.VSNI());
   }

   const std::list<cMasqImageIn> aLM = mAppli.MasqImageIn();

   for 
   ( 
      std::list<cMasqImageIn>::const_iterator itM = aLM.begin();
      itM != aLM.end();
      itM++
   )
   {
       std::string  aNameM =  OneNameMasq(itM->OneMasqueImage());
       
       if (aNameM != "PasDeMasqImage")
       {
           aName = aName + "_"+StdPrefix(aNameM);
           std::string  aFullNM =   mAppli.DirMasqueIms() + aNameM;
           aFRes = aFRes && Tiff_Im(aFullNM.c_str()).in();
       }
   }

   return aFRes;
}

std::string  cPriseDeVue::NameMasqOfResol(int aDz) const
{
    std::string aName;
    FoncMasq(aName);
// Fonc_Num  cPriseDeVue::FoncMasq(std::string  & aName) const

   return   mAppli.FullDirPyr() 
          + mAppli.PrefixMasqImRes().Val()
          + std::string("_Dz") + ToString(aDz) + std::string("_")
          + aName +".tif";
// #endif
}



Tiff_Im     cPriseDeVue::FileImMasqOfResol(int aDz) const
{
   std::string aName = NameMasqOfResol(aDz);

   Pt2di aSz = Std2Elise(mIMIL->Sz(aDz));
   if (! ELISE_fp::exist_file(aName))
   {
      std::cout << "Make Masq " << aName << "\n";
      if (aDz==1)
      {
           Tiff_Im aFile
                   (
                       aName.c_str(),
                       aSz,
                       Xml2EL(mAppli.TypeMasque().Val()),
                       Xml2EL(mAppli.ComprMasque().Val()),
                       // GenIm::bits1_msbf,
			// Tiff_Im::No_Compr,
                       //Tiff_Im::Group_4FAX_Compr,
                       Tiff_Im::BlackIsZero,
                       Tiff_Im::Empty_ARG
			+ Arg_Tiff(Tiff_Im::ANoStrip())
                        +  Arg_Tiff(Tiff_Im::AFileTiling(aSzTileMasq))
                   );
            // int aBrd = 3;
            std::string  aNameBid;
            ELISE_COPY
            (
               rectangle(Pt2di(0,0),aSz),
               FoncMasq(aNameBid),
               aFile.out()
            );

      }
      else
      {
          FileImMasqOfResol(aDz/2);
          MakeTiffRed2BinaireWithCaracIdent
          (
             NameMasqOfResol(aDz/2),
             aName,
             0.99,
             aSz
          );
      }
   }
   else
   {
        // Tiff_Im aRes = Tiff_Im::StdConvGen(aName,1,true,true);
        Tiff_Im aRes = Tiff_Im::BasicConvStd(aName);
        if (aRes.sz() != aSz)
        {
            std::cout << "For Name Im = " << aName << "\n";
            ELISE_ASSERT
            (
                false,
                "Incoherence in exiting image mask"
            );
        }

        return aRes ;
   }

   // return Tiff_Im::StdConvGen(aName,1,true,true);
   return Tiff_Im::BasicConvStd(aName);
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
