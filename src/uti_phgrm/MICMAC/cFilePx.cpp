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
#include "../src/uti_phgrm/MICMAC/MICMAC.h"


/*************************************************/
/*                                               */
/*          cOneNappePx                          */
/*                                               */
/*************************************************/



void cOneNappePx::TestDebugOPX(const std::string& aMes)
{
std::cout << "ffffffffffffff\n";
   if (mForDeZ) return;
   if (mW==0)
      mW  = Video_Win::PtrWStd(mSz);

   std::cout << "MES=" << aMes << "\n";

   ELISE_COPY(mW->all_pts(),P8COL::red,mW->odisc());

   ELISE_COPY(select(mW->all_pts(),mImPxMax.in()>=mImPxMin.in()),mImPxMax.in()-mImPxMin.in(),mW->ogray());
 
getchar();
}



 
cOneNappePx::cOneNappePx(int aKPx,Pt2di aSz) :
   mSz     (aSz),
   mPxRes  (mSz.x,mSz.y),
   mTPxRes (mPxRes),
   mPxInit (mSz.x,mSz.y),
   mTPxInit (mPxInit),
   mImPxMin  (mSz.x,mSz.y),
   mTImPxMin (mImPxMin),
   mImPxMax  (mSz.x,mSz.y),
   mTImPxMax (mImPxMax),
   mPxRedr (mSz.x,mSz.y,float(0.0)),
   mTPxRedr (mPxRedr),
   mKPx     (aKPx),
   mRedrPx  (false),
   mW         (0),
   mForDeZ    (false)
{
}


cOneNappePx::cOneNappePx
(
    TIm2DBits<1> & aMasqDZ,
    const cOneNappePx & aNap,
    const TIm2DBits<1> & aMasqInit,
    const cEquiv1D & anEqX,
    const cEquiv1D & anEqY
) :
   mSz     (anEqX.NbClasses(),anEqY.NbClasses()),
   mPxRes  (mSz.x,mSz.y),
   mTPxRes (mPxRes),
   mPxInit (mSz.x,mSz.y),
   mTPxInit (mPxInit),
   mImPxMin  (mSz.x,mSz.y),
   mTImPxMin (mImPxMin),
   mImPxMax  (mSz.x,mSz.y),
   mTImPxMax (mImPxMax),
   mPxRedr (1,1),
   mTPxRedr (mPxRedr),
   mKPx     (aNap.mKPx),
   mRedrPx  (aNap.mRedrPx),
   mW         (0),
   mForDeZ    (true)
{
// std::cout << "iiiiiiiiiiiiiiii "  << mDilatAlti  << "\n";getchar();
   for (int aClY =0 ; aClY<mSz.y ; aClY++)
   {
      int anY0,anY1;
      anEqY.ClasseOfNum(anY0,anY1,aClY);
      for (int aClX =0 ; aClX<mSz.x ; aClX++)
      {
          int anX0,anX1;
          anEqX.ClasseOfNum(anX0,anX1,aClX);
          int aPxMin =  (1<<30);
          int aPxMax = -(1<<30);
          Pt2di aP;
          bool OkMasq = false;
          int aSomRes = 0;
          int aCpt    = 0;
          for (aP.y = anY0; aP.y<anY1 ; aP.y++)
          {
             for (aP.x = anX0; aP.x<anX1 ; aP.x++)
             {
                if (aMasqInit.get(aP))
                {
                    ElSetMin(aPxMin,aNap.mTImPxMin.get(aP));
                    ElSetMax(aPxMax,aNap.mTImPxMax.get(aP));
                    OkMasq = true;
                    aCpt++;
                    aSomRes += aNap.mTPxRes.get(aP);
                }
             }
          }
          if (! OkMasq)
          {
             aPxMin = 0;
             aPxMax = 1;
             aCpt =1;
          }
          Pt2di aPDz = Pt2di(aClX,aClY);
          mTImPxMin.oset(aPDz,aPxMin);
          mTImPxMax.oset(aPDz,aPxMax);
          aMasqDZ.oset(aPDz,OkMasq);
          mTPxRes.oset(aPDz,aSomRes/aCpt);
          mTPxInit.oset(aPDz,aSomRes/aCpt);
      }
   }

   mVPxMin = aNap.mVPxMin;
   mVPxMax = aNap.mVPxMax;
   mVPxMinAvRedr = aNap.mVPxMinAvRedr;
   mVPxMaxAvRedr = aNap.mVPxMaxAvRedr;

   Im2D_Bits<1> aIDZ = aMasqDZ.Im();
   cResProj32 aRP32 = Projection32(aIDZ.in(),aIDZ.sz());
   ComplWithProj32(aRP32);
   ForceConnexions();
}

void cOneNappePx::ForceConnexions()
{

   // Modif MPD GM pour forcer aussi les connexions sur le bord
   for (int aClY =0 ; aClY<mSz.y ; aClY++)
   {
      for (int aClX =0 ; aClX<mSz.x ; aClX++)
      {
           Pt2di aP0(aClX,aClY);
           int aMin0 = mTImPxMin.get(aP0);
           int aMax0 = mTImPxMax.get(aP0);
           for (int aDx=-1 ; aDx<=1 ;aDx++)
           {
               for (int aDy=-1 ; aDy<=1 ;aDy++)
               {
                     Pt2di aP1(aClX+aDx,aClY+aDy);
                    // Modif MPD GM
                     if (  (aP1.x>=0) && (aP1.y>=0)  && (aP1.x<mSz.x) && (aP1.y<mSz.y))
                     {
                         mTImPxMin.oset(aP1,ElMin(mTImPxMin.get(aP1),aMax0-1));
                         mTImPxMax.oset(aP1,ElMax(mTImPxMax.get(aP1),aMin0+1));
                     }
               }
           }
      }
   }
}

void  cOneNappePx::ComplWithProj32(const cResProj32 & aRP32)
{
   if (aRP32.IsFull())
   {
   }
   else if (aRP32.IsInit())
   {
   // MODIF MPD 23/05/09 ; ca peut generer des tres gros espace memoire
   // inutile dans certaines config
/*
      ELISE_COPY
      (
          mImPxMin.all_pts(),
          Virgule(mImPxMin.in(),mImPxMax.in())[Virgule(aRP32.PX().in(),aRP32.PY().in())],
          Virgule(mImPxMin.out(),mImPxMax.out())
      );
*/
//
/**/  
    Symb_FNum aF ( ((mImPxMin.in()+mImPxMax.in())/2)[Virgule(aRP32.PX().in(),aRP32.PY().in())]);
    ELISE_COPY
    (
          select
          (
              mImPxMin.all_pts(),
              (aRP32.PX().in()!=FX)  || (aRP32.PY().in()!=FY)
          ),
          Virgule(aF,aF+1),
          Virgule(mImPxMin.out(),mImPxMax.out())
    );
/**/
   }
   else
   {
      ELISE_COPY
      (
          mImPxMin.all_pts(),
          Virgule(0,1),
          Virgule(mImPxMin.out(),mImPxMax.out())
      );
   }
}


void cOneNappePx::CalculBornesPax(Box2di aBox,INT & aVMin,INT & aVMax)
{
   ELISE_COPY
   (
      rectangle(aBox._p0,aBox._p1),
      Virgule(mImPxMin.in(),mImPxMax.in()),
      Virgule(VMin(aVMin),VMax(aVMax))
   );
}

bool   cOneNappePx::OkPx(const Pt2di & aP,int aPx) const
{
   return   (aPx >= mTImPxMin.get(aP))
         && (aPx <  mTImPxMax.get(aP));
}

double   cOneNappePx::ResultPx(const Pt2dr & aP,bool isOptCont)
{
   return isOptCont             ?
          mTPxRedr.getprojR(aP)  :
          mTPxRes.getprojR(aP)   ;
}

/*************************************************/
/*                                               */
/*          cLoadTer                             */
/*                                               */
/*************************************************/

cLoadTer::cLoadTer(int aNbPx,Pt2di aSz,const cEtapeMecComp & anEtape) :
  mNbPx           (aNbPx),
  mSz             (aSz),
  mImMasqSsPI     (aSz.x,aSz.y),
  mImMasqTer      (aSz.x,aSz.y),
  mTImMasqTer     (mImMasqTer),
  mImMasqTerOfPx  (aSz.x,aSz.y),
  mTImMasqTerOfPx (mImMasqTerOfPx),
  mImOneDefCorr   (aSz.x,aSz.y,0),
  mTOneDefCorr    (mImOneDefCorr),
  mImCorrel       (mSz.x,mSz.y),
  mTImCorrel      (mImCorrel),
  mImCorrelSol    (1,1),
  mSomPdsCorr     (1,1),
  mSomCorrel      (1,1),
  mSomI1I2        (1,1),
  mSomMasqI1I2    (1,1),
  mSom1           (1,1),
  mMasqGlob       (1,1)
{
   if (anEtape.GenImageCorrel() && (aSz.x >0))
      mImCorrelSol.Resize(aSz);
   for (int aK=0 ; aK<aNbPx ; aK++)
      mPxs.push_back(cOneNappePx(aK,aSz));
}


cLoadTer::cLoadTer
(
    const cLoadTer & aLT,
    const cEquiv1D & anEqX,
    const cEquiv1D & anEqY
) :
  mNbPx           (aLT.NbPx()),
  mSz             (anEqX.NbClasses(),anEqY.NbClasses()),
  mImMasqSsPI     (mSz.x,mSz.y),
  mImMasqTer      (mSz.x,mSz.y),
  mTImMasqTer     (mImMasqTer),
  mImMasqTerOfPx  (mSz.x,mSz.y),
  mTImMasqTerOfPx (mImMasqTerOfPx),
  mImOneDefCorr   (mSz.x,mSz.y,0),
  mTOneDefCorr    (mImOneDefCorr),
  mImCorrel       (1,1),
  mTImCorrel      (mImCorrel),
  mImCorrelSol    (1,1),
  mSomPdsCorr     (1,1),
  mSomCorrel      (1,1),
  mSomI1I2        (1,1),
  mSomMasqI1I2    (1,1),
  mSom1           (1,1),
  mMasqGlob       (1,1)
{
 // std::cout << "jjjjjjjjjjjjjjjjjjjjjjj\n";
   for (int aK=0 ; aK<aLT.NbPx() ; aK++)
   {
      mPxs.push_back
      (
           cOneNappePx
           (
              mTImMasqTer ,
              aLT.mPxs[aK],
              aLT.mTImMasqTer,
              anEqX,
              anEqY
           )
      );
      mPxMin[aK] = mPxs[aK].mVPxMinAvRedr;
      mPxMax[aK] = mPxs[aK].mVPxMaxAvRedr;
   }
}

Im2D_Bits<1>   cLoadTer::ImOneDefCorr()
{
   return mImOneDefCorr;
}

void  cLoadTer::SetDefCorrActif(const Pt2di & aP,int aVal)
{
    mTOneDefCorr.oset(aP,aVal);
}

bool  cLoadTer::IsDefCorrActif(const Pt2di  &aP) const
{
   return ( mTOneDefCorr.get(aP)!=0 );
}


Im2D_Bits<1> cLoadTer::ImMasqSsPI() const
{
   return mImMasqSsPI;
}


Im2D_Bits<1> cLoadTer::ImMasqTer() const
{
   return mImMasqTer;
}


cOneNappePx & cLoadTer::KthNap(int aK) { return mPxs.at(aK);}
const cOneNappePx & cLoadTer::KthNap(int aK) const { return mPxs.at(aK);}

int cLoadTer::NbPx() const
{
   return mNbPx;
}


Im2D_REAL8   cLoadTer::ImCorrel() const {return mImCorrel;}
Im2D_U_INT1  cLoadTer::ImCorrelSol () const {return mImCorrelSol;}


INT2 ** cLoadTer::GPULowLevel_ZMin() const
{
   return  mPxs[0].mImPxMin.data();
}
INT2 ** cLoadTer::GPULowLevel_ZMax() const
{
   return  mPxs[0].mImPxMax.data();
}

U_INT1 ** cLoadTer::GPULowLevel_MasqTer() const
{
   return mImMasqTer.data();
}




void cLoadTer::CalculBornesPax(Box2di aBox,int * aVMin,int *aVMax)
{
   for (int aK=0 ; aK<mNbPx ; aK++)
       mPxs[aK].CalculBornesPax(aBox,aVMin[aK],aVMax[aK]);
}

void cLoadTer::GetBornesPax(const Pt2di & aPt,int * aVMin,int *aVMax) const
{
   for (int aK=0 ; aK<mNbPx ; aK++)
   {
       aVMin[aK] = mPxs[aK].GetPxMin(aPt);
       aVMax[aK] = mPxs[aK].GetPxMax(aPt);
   }
}

Pt2di cLoadTer::Sz() const { return mSz; }

const int * cLoadTer::PxMin() const {return mPxMin;}
const int * cLoadTer::PxMax() const {return mPxMax;}
int *       cLoadTer::PxMin()       {return mPxMin;}
int *       cLoadTer::PxMax()       {return mPxMax;}

bool   cLoadTer::OkPx(const Pt2di & aP,const int * aPx) const
{
    if (! IsInMasq(aP))
       return false;
    for (int aK=0 ; aK<mNbPx; aK++)
       if (!  mPxs[aK].OkPx(aP,aPx[aK]))
          return false;
    return  true;
}

void cLoadTer::Redressement(double * aVRedr,const Pt2di & aP) const
{
    for (int aK=0 ; aK<mNbPx; aK++)
        aVRedr[aK] = mPxs[aK].Redressement(aP);
}

void cLoadTer::Redressement(double * aVRedr,const Pt2dr & aP) const
{
    for (int aK=0 ; aK<mNbPx; aK++)
        aVRedr[aK] = mPxs[aK].Redressement(aP);
}



void cLoadTer::MakeImTerOfPx(Box2di aBox,int * aPx)
{
  {
      Pt2di aP;
      for (aP.y=aBox._p0.y ; aP.y<aBox._p1.y; aP.y++)
      {
          for (aP.x=aBox._p0.x ; aP.x<aBox._p1.x; aP.x++)
          {
              mTImMasqTerOfPx.oset(aP,mTImMasqTer.get(aP));
          }
      }
   }

   for (int aK=0 ; aK<int(mPxs.size()); aK++)
   {
       Pt2di aP;
       TIm2D<INT2,INT> aTMin(mPxs[aK].mImPxMin);
       TIm2D<INT2,INT> aTMax(mPxs[aK].mImPxMax);
       int aPxK = aPx[aK];
       for (aP.y=aBox._p0.y ; aP.y<aBox._p1.y; aP.y++)
       {
           for (aP.x=aBox._p0.x ; aP.x<aBox._p1.x; aP.x++)
           {
               if 
               (
                     (aPxK < aTMin.get(aP) )
                  || (aPxK >= aTMax.get(aP))
               )
                   mTImMasqTerOfPx.oset(aP,0);
           }
       }
   }
}

/*************************************************/
/*                                               */
/*          cFilePx                              */
/*                                               */
/*************************************************/

Tiff_Im cFilePx::FileIm(bool & IsCreated) const
{
    Tiff_Im aRes = Tiff_Im::CreateIfNeeded
	    (
                IsCreated,
                mNameFile,
                mSzFile,
                mEtape.IsOptimReel() ? GenIm::real4 : GenIm::int2,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
	    );

    return aRes;
}

Tiff_Im cFilePx::FileImRel() const
{
    bool IsCreated;
    return Tiff_Im::CreateIfNeeded
	    (
                IsCreated,
                mNameFileRel,
                mSzFile,
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
	    );
}



void cFilePx::RemplitOri(cFileOriMnt & aFOM) const
{
  aFOM.NameFileMnt() = mNameFile;
}

Tiff_Im cFilePx::FileIm() const
{
   bool IsCreated;
   return FileIm(IsCreated);
}

std::string  cFilePx::NameFileGen(const std::string & aCompl)
{
   return    mAppli.FullDirMEC() + NameFileGenSsDir(aCompl);
}

std::string  cFilePx::NameFileGenSsDir(const std::string & aCompl) const
{
    return
             mNamePost
           + (
                mIsPseudoFirst  ?
                "_000"         :
                (std::string("_Num") + ToString(mEtape.Num()))
             )
           + std::string("_DeZoom")
           + ToString(mEtape.DeZoomTer())
           + std::string("_")
           + mAppli.NameChantier()
           + aCompl
           + std::string(".tif");
}

std::string  cFilePx::NameFileSsDir() const
{
   return NameFileGenSsDir("");
}


cFilePx::cFilePx
(
    const cArgOneEtapePx & anArg,
    const cAppliMICMAC &   anAppli,
    cEtapeMecComp &        anEtape,
    cFilePx *              aPredCalc,
    bool                   isPseudoFirst,
    const std::string &    aPost,
    int                    aKPx
)  :
   cArgOneEtapePx  (anArg),
   mAppli          (anAppli),
   mEtape          (anEtape),
   mPredCalc       (aPredCalc),
   mIsPseudoFirst  (isPseudoFirst),
   mNamePost       (aPost),
   mNameFile       (NameFileGen("")),
   mNameFileRel    (NameFileGen("Rel")),
   mSzFile         (anEtape.SzFile()),
   // A ce stade, Px1IncCalc n'a pas ete introduit dans DilatAlti,
   // par prudence on tient donc compte de NumEtape
   mGenFile        (anEtape.IsLast() || IsFirtOrNappeIsEpaisse()),
   mGenFileMA      (false),
   mGenFileRel     (anEtape.EtapeMEC().GenFilePxRel().ValWithDef(false)),
   mKPx            (aKPx)
{
}


void cFilePx::CreateMNTInit()
{
     // Si c'est le fichier 000, correspondant a aucune etape,
     // on l'initialise a 0 (a modifier lorsque prise en compte
     // d'un MNT
     if (mIsPseudoFirst && (mAppli.ModeAlloc()==eAllocAM_STD))
     {
         bool isNew;
         Tiff_Im  aFile = FileIm(isNew);
	 if (isNew)
	 {
            if (
                   (mAppli.ModeGeomMEC() == eGeomMECTerrain)
                && (mAppli.MNT_Init().IsInit())
               )
            {
               cFileOriMnt anOriCible =mAppli.OrientFromOneEtape(mEtape);

		Tiff_Im aFileMNT = Tiff_Im::StdConvGen
                                  (
                                     mAppli.WorkDir() +mAppli.MNT_Init().Val().MNT_Init_Image(),
                                     1,
                                     true,
                                     true
                                  );

                Fonc_Num aFMnt = mAppli.AdaptFoncFileOriMnt
                                 (
                                    anOriCible,
                                    aFileMNT.in_proj(),
                                      mAppli.WorkDir()
                                    + mAppli.MNT_Init().Val().MNT_Init_Xml(),
                                    true,
                                    mAppli.MNT_Init().Val().MNT_Offset().Val()
                                 );
                 ELISE_COPY(aFile.all_pts(),aFMnt,aFile.out());
            }
            else if (mAppli.NuageXMLInit().IsInit())
            {
                  const cNuageXMLInit & aNX = mAppli.NuageXMLInit().Val();
                  ELISE_ASSERT(!aNX.CanAdaptGeom().Val(),"Cannot adpat Geom");

                  std::string aNameOri = aNX.NameNuageXML() ;
                  std::string aNameCible =  mAppli.FirstVraiEtape()->NameXMLNuage();

                  cElNuage3DMaille * aClOri =  NuageWithoutData(aNameOri);
                  cElNuage3DMaille * aClTest = NuageWithoutDataWithModel(aNameCible,aNameOri);

                  if (GeomCompatForte(aClOri,aClTest))
                  {
                     cXML_ParamNuage3DMaille aXmlOri = XML_Nuage(aNameOri);
                     cXML_ParamNuage3DMaille aXmlCible = XML_Nuage(aNameCible);
                     // Fonc_Num aF = Pix2Pix(aXmlCible,,aXmlOri);
                     // std::cout << "TEST mAppli.NuageXMLInit " <<  aNameOri << " " << aXmlOri.Image_Profondeur().Val().ResolutionAlti() << "\n";
                     // std::cout << "UUU "  << aNameCible <<  " " << aXmlCible.Image_Profondeur().Val().ResolutionAlti() << "\n";
                     ELISE_COPY
                     (
                        aFile.all_pts(),
                        Pix2Pix(aXmlCible,aXmlOri,DirOfFile(aNameOri)),
                        aFile.out()
                     );
                     // std::cout << "AKKKKkkk " << aFile.name() << DirOfFile(aNameOri) << "\n";
                     // getchar();
                  }
                  else
                  {
                      ELISE_ASSERT(aNX.CanAdaptGeom().Val(),"Geom incompatible in NuageXMLInit");
                  }

            }
            else
            {
                 ELISE_COPY(aFile.all_pts(),0,aFile.out());
            }
	 }
     }
     if (mAppli.ModeAlloc()==eAllocAM_STD)
        FileIm();
}





void cFilePx::Show( const std::string & aMes) const
{
   cout << aMes << "-" << mNameFile << "\n";
   cout << "ALTI " << mDilatAltiMoins << " " << mDilatAltiPlus << "\n";
   if (mPredCalc)
      cout << "   " << mPredCalc->mNameFile << "\n";
}

void cFilePx::SetCaracOfZoom(cCaracOfDeZoom & aCZ)
{
   if (mDilatPlani)
       mDilatPlani += round_ni(mDilatPlaniProp* (1/aCZ.RatioPtsInt()-1));
}

bool cFilePx::NappeIsEpaisse() const
{
    return     (mDilatAltiPlus !=0) 
            || (mDilatAltiMoins!=0)
	    || (mDilatPlani !=0);
}

bool cFilePx::IsFirtOrNappeIsEpaisse() const
{
   return    NappeIsEpaisse() 
         ||  (mEtape.Num() == 0);
}

bool cFilePx::GenFile() const
{
   return mGenFile;
}





         // ACCESSEURS      

REAL  cFilePx::UserPas() const
{
   return mUserPas;
}

REAL  cFilePx::ComputedPas() const
{
   ELISE_ASSERT(mComputedPas>=0,"cFilePx::ComputedPas");
   return mComputedPas;
}

void cFilePx::InitComputedPas(double aRatio)
{
   mComputedPas = mUserPas * aRatio;

   if (mPredCalc)
   {
        mRatioDzPrec =   mPredCalc->mEtape.DeZoomTer() / mEtape.DeZoomTer();
        mRatioStepPrec =  (ComputedPas() / mPredCalc->ComputedPas()) / mRatioDzPrec;
   }
}

/*
int &  cFilePx::NCDilatAlti() 
{
   return mDilatAlti;
}
*/
int   cFilePx::DilatAltiPlus()  const
{
   return mDilatAltiPlus;
}
int   cFilePx::DilatAltiMoins()  const
{
   return mDilatAltiMoins;
}



int   cFilePx::DilatPlani()  const
{
   return mDilatPlani;
}

double   cFilePx::Regul() const 
{ 
    return mRegul; 
}

double   cFilePx::Regul_Quad() const 
{ 
    return mRegul_Quad; 
}

bool cFilePx::RedrPx() const
{
   return mRedrPx;
}

const std::string& cFilePx::NameFile() const
{
   return mNameFile;
}

void Test(const std::string & aMes,Im2D_INT2 aI1,Im2D_INT2 aI2)
{
    int aNb =0;
    int aS1 =0;
    ELISE_COPY(aI1.all_pts(),Virgule(Abs(aI1.in()-aI2.in()),1),Virgule(sigma(aNb),sigma(aS1)));
    std::cout << aMes << " " << aNb/aS1 << "\n"; 
    getchar();
}


void cFilePx::LoadNappeEstim 
     (
           const cEtapeMecComp & anETC,
           const cResProj32 & aRP32,
           Im2D_Bits<1> aIMasq,
           cOneNappePx & aNappe,
           Box2di aBoxIn
     )
{

// Boite englobante homologue en px prec, un peu elargie
    Pt2di aP0Prec = aBoxIn._p0 /mRatioDzPrec - Pt2di(2,2);
    Pt2di aP1Prec = aBoxIn._p1 /mRatioDzPrec + Pt2di(2,2);
    Pt2di aSzDz= aP1Prec-aP0Prec;

// Chargement  de la paralaxe Prec
    Im2D_INT2 aIPxPrec(aSzDz.x,aSzDz.y);

    mAppli.VerifSzFile(mPredCalc->FileIm().sz());
    ELISE_COPY
    (
       aIPxPrec.all_pts(),
       trans(mPredCalc->FileIm().in_proj(),aP0Prec),
       aIPxPrec.out()
    );
    TIm2D<INT2,INT> aTPxPrec(aIPxPrec);

    // Adaptation des tailles de nappes
    Pt2di aSz = aBoxIn.sz();
    // On va calculer l'extrapolation dans une 
    // l'image de resultat

    cGeomDiscFPx & aGeomPrec = mPredCalc->mEtape.GeomTer();
    cGeomDiscFPx & aGeom = mEtape.GeomTer();
    aGeomPrec.SetClip(aP0Prec,aP1Prec);
    for (int anX=0 ; anX<aSz.x ; anX++)
    {
        // On evite de refaire le calcul en utilisant le fait qu'on
        // sait que les transfo sont lineaire et separable, le long
        // d'une colone il suffit donc de propager un dY
        Pt2dr aPRed = aGeomPrec.R2ToRDisc(aGeom.DiscToR2(Pt2di(anX,0)));
        Pt2dr aPRed1 = aGeomPrec.R2ToRDisc(aGeom.DiscToR2(Pt2di(anX,1)));
        double aDy = aPRed1.y-aPRed.y;

        for (int anY=0 ; anY<aSz.y ; anY++)
        {
            aNappe.mTPxInit.oset
            (
                Pt2di(anX,anY),
                round_ni(aTPxPrec.getprojR(aPRed)/mRatioStepPrec)
            );
            aPRed.y += aDy;
        }
    }


    if (mEtape.PxAfterModAnIsNulle())
    {
       ELISE_COPY
       (
           aNappe.mPxInit.all_pts(),
           0,
           aNappe.mPxInit.out()
       );
    }



    // Dilatation en Alti et plani; afin que les zones hors du masque
    // n'aient pas d'influence on leur donne une valeur + ou - "infini"
    // avant la dilatatation

    Fonc_Num aFMasq = aIMasq.in(0);
    Pt2di aSzM = aIMasq.sz();


    int aRadDZMoins = 0;
    int aRadDPlMoins = 0;

    if (mKPx==0)
    {
       const cEtapeMEC &   anET = anETC.EtapeMEC() ;
       aRadDZMoins  = anET.RabZDilatAltiMoins().ValWithDef(0);
       aRadDPlMoins = anET.RabZDilatPlaniMoins().ValWithDef(0);
       ELISE_ASSERT(aRadDZMoins>=0, "RabZDilatAltiMoins  <0");
       ELISE_ASSERT(aRadDPlMoins>=0,"RabZDilatPlaniMoins <0");
    }

    // Sinon Bug laisse des valeurs infinies sur les bords
    if (mDilatPlani==0) 
       aFMasq = aFMasq || (!inside(Pt2di(1,1),aSzM-Pt2di(1,1)));

    int aMaxShrt = (1<<15)-1;

    Fonc_Num  aFMin = aNappe.mPxInit.in_proj();
    Fonc_Num  aFMax = aNappe.mPxInit.in_proj();
    if (
              (mAppli.IntervAltimetrie().IsInit())
           && (mAppli.EnveloppeMNT_INIT().IsInit())
       )
    {
        cEnveloppeMNT_INIT aEnv=mAppli.EnveloppeMNT_INIT().Val();
        Tiff_Im aTifInf  = Tiff_Im::StdConv(aEnv.ZInf());
        Tiff_Im aTifSup  = Tiff_Im::StdConv(aEnv.ZSup());

         aFMin  = trans(aTifInf.in(aMaxShrt),aBoxIn._p0);
         aFMax = trans(aTifSup.in( -aMaxShrt),aBoxIn._p0);

         ELISE_COPY
         (
             aNappe.mPxInit.all_pts(),
             (aFMin+aFMax)/2,
             aNappe.mPxInit.out()
         );


    }


    aFMin =   aFMasq      * (aFMin-mDilatAltiMoins-aRadDZMoins) + (1-aFMasq)  * aMaxShrt;
    aFMax =   aFMasq  * (aFMax+mDilatAltiPlus+1) + (1-aFMasq)  * (-aMaxShrt);



    ELISE_COPY
    (
       aNappe.mPxInit.all_pts(),
       Virgule
       ( 
           rect_min(aFMin,mDilatPlani+aRadDPlMoins),
           rect_max(aFMax,mDilatPlani)
       ),
       Virgule ( aNappe.mImPxMin.out(), aNappe.mImPxMax.out())
    );


// aNappe.TestDebugOPX("INIT");


   int aNb ;
    ELISE_COPY
    (
       select(aNappe.mPxInit.all_pts(),aIMasq.in() !=0),
       Virgule (aNappe.mImPxMin.in()  ,aNappe.mImPxMax.in()  ,1),
       Virgule (VMin(aNappe.mVPxMin)  ,VMax(aNappe.mVPxMax)  ,sigma(aNb))
    );
    if (aNb==0)
    {
      aNappe.mVPxMin = 0;
      aNappe.mVPxMax = 0;
    }

    // cout << "Nb " << aNb << " " << aNappe.mVMin << " "<< aNappe.mVMax << "\n";

    if  ((mPredCalc->mPredCalc) || (mKPx!=0))  // BUG dit "du patriarche"  !! Corrc Bug Pat =>
    {
       aNappe.ComplWithProj32(aRP32);
    }
    else
    {
          // Si est a la premiere etape, le "ComplWithProj32" a pour effet de bord de mettre
          // a 0 les bord du masque, hors 0 peut etre tres loin de la valeur moyenne de 
          // paralaxe lorsque  en geom faiscea
           ELISE_COPY
           (
                 select
                 (
                     aNappe.mPxInit.all_pts(),
                     erod_d8(aIMasq.in(0)== 0,1)
                 ),
                 Fonc_Num (0,1),
                 Virgule(aNappe.mImPxMin.out(),aNappe.mImPxMax.out())
           );
   
    }

// aNappe.TestDebugOPX("COMPL 32");
    if ( NappeIsEpaisse())
       aNappe.ForceConnexions();

    // On initialise le resultat sur la valeur init, comme ca les
    // algo peuvent eventuellement ignorer une des 2 Px 

/*
MODIF 5-2-2012   AJOUTER EN BAS CAR PAS MAL DE CHOSES FAITE ENSUITE QUI
CASSE CE QUE L'ON VIENT DE FAIOR
std::cout << "SUUUUUUUUUUUPPPPRESS\n";
    ELISE_COPY
    (
        aNappe.mPxInit.all_pts(),
        Max
        (
           aNappe.mImPxMin.in(),
           Min
           (
                aNappe.mImPxMax.in()-1,
                aNappe.mPxInit.in()
           )
        ),
        aNappe.mPxRes.out()| aNappe.mPxInit.out()
    );
*/
    // mMoyPxRed = 0.0;

    

    aNappe.mRedrPx = mRedrPx;
    aNappe.mVPxMinAvRedr = aNappe.mVPxMin;
    aNappe.mVPxMaxAvRedr = aNappe.mVPxMax;

    bool isForCont = mAppli.IsOptimCont() ;//  && (!mAppli.IsOptIdentite());
    bool doExportZAbs =  mAppli.CurEtape()->IsExportZAbs();
    ELISE_ASSERT
    (
          ! (mRedrPx && (isForCont)),
          "Incompatibilite Redressement / Optim continue"
    );

    aNappe.mIMoyPx = 0;
    aNappe.mRMoyPx = 0;


    if (mAppli.IsOptIdentite())
    {
       ELISE_COPY
       (
             aIPxPrec.all_pts(),
             aIPxPrec.in(),
                aNappe.mPxInit.out()
             |  aNappe.mImPxMin.out()
             |  (aNappe.mImPxMax.out() << (1+aIPxPrec.in()))
             |  (aNappe.mPxRedr.out())
       );
/*
        aNappe.mPxInit.all_pts(),
        Max
        (
           aNappe.mImPxMin.in(),
           Min
           (
                aNappe.mImPxMax.in()-1,
                aNappe.mPxInit.in()
           )
*/
    }
    else if (mRedrPx || isForCont)
    {

        int aIPx0[theDimPxMax] ={0,0};
        int aIPx1[theDimPxMax] ={1,1};
        double aRPx0[theDimPxMax];
        double aRPx1[theDimPxMax];
        aGeom.PxDisc2PxReel(aRPx0,aIPx0);
        aGeom.PxDisc2PxReel(aRPx1,aIPx1);
        aNappe.mNPx0 = aRPx0[aNappe.mKPx];
        aNappe.mDPx = aRPx1[aNappe.mKPx] - aNappe.mNPx0;

        TIm2D<REAL4,REAL8> aTRedrCur(aNappe.mPxRedr);


        if (mDeqRedrPx  || mAppli.IsOptimCont() )
        {
           bool Filtrage = mRedrPx;
           // Filtrage Median en Amonnt
           if (Filtrage)
           {
              int aNbIterMed = mEtape.EtapeMEC().RedrNbIterMed().ValWithDef(2);
              int aSzMed = mEtape.EtapeMEC().RedrSzMed().ValWithDef(2);

              int aVMax,aVMin;
              ELISE_COPY
              (
                 aIPxPrec.all_pts(),
                 aIPxPrec.in(),
                 VMax(aVMax)|VMin(aVMin)
              );

              Fonc_Num aF = (aIPxPrec.in_proj()-aVMin);
              for (int aK=0 ; aK<aNbIterMed ; aK++)
                  aF = rect_median(aF,aSzMed,aVMax+1-aVMin);
              ELISE_COPY(aIPxPrec.all_pts(),aF+aVMin,aIPxPrec.out());
           }

           ElImplemDequantifier aDeq(aSzDz);
           aDeq.SetTraitSpecialCuv(true);
           aDeq.DoDequantif(aSzDz,aIPxPrec.in(),true);
           Im2D_REAL4 aIPxPrecDeq(aSzDz.x,aSzDz.y);
           ELISE_COPY
           (
               aIPxPrecDeq.all_pts(),
               aDeq.ImDeqReelle(),
               aIPxPrecDeq.out()
           );
           TIm2D<REAL4,REAL8> aTRedrPrec(aIPxPrecDeq);

           double aSomPx=0;
           double aSom1=0;
           TIm2DBits<1> aTM(aIMasq);
// std::cout << " IFX " << isForCont << "WWwwwXX " << aNappe.FromDiscPx(0) << " " << aNappe.FromDiscPx(1) << " " << mRatioStepPrec << "\n"; getchar();
           for (int anX=0 ; anX<aSz.x ; anX++)
           {
           // On evite de refaire le calcul en utilisant le fait qu'on
           // sait que les transfo sont lineaire et separable, le long
           // d'une colone il suffit donc de propager un dY
               Pt2dr aPRed = aGeomPrec.R2ToRDisc(aGeom.DiscToR2(Pt2di(anX,0)));
               Pt2dr aPRed1 = aGeomPrec.R2ToRDisc(aGeom.DiscToR2(Pt2di(anX,1)));
               double aDy = aPRed1.y-aPRed.y;

               for (int anY=0 ; anY<aSz.y ; anY++)
               {
                   Pt2di aP(anX,anY);
                   double aPx = aTRedrPrec.getprojR(aPRed)/mRatioStepPrec;
                   if (aTM.get(aP,0))
                   {
                      aSomPx += aPx;
                      aSom1  += 1.0;
                   }
                   // if (! isForCont) || doExportZAbs)
                   if ((! isForCont) || doExportZAbs)
                   {
                       aPx =  aNappe.FromDiscPx(aPx) ;
                   }
                   aTRedrCur.oset(aP, aPx);
                   aPRed.y += aDy;
               }
           }
           if (mRedrPx)
           {
              if (aSom1 != 0) 
              {
                  aNappe.mIMoyPx = round_ni(aSomPx/aSom1);
                  aNappe.mRMoyPx = aNappe.FromDiscPx(aNappe.mIMoyPx);
              }
              // std::cout << "KPx "<< aNappe.mKPx << " PX MOY REDR " << aNappe.mIMoyPx << " " 
              //          << aNappe.mRMoyPx << " " << aNappe.mNPx0 << " " << aNappe.mDPx << "\n"; 
           }

              // ---int aNbIterMed = mEtape.EtapeMEC().RedrNbIterMed().Val();
           if (Filtrage)
           {
               int aNbIterMoy = mEtape.EtapeMEC().RedrNbIterMoy().ValWithDef(3);
               int aSzVoisMoy = mEtape.EtapeMEC().RedrSzMoy().ValWithDef(8);

               Fonc_Num aF = Rconv(aNappe.mPxRedr.in_proj()-aNappe.mRMoyPx);
               for (int aK= 0 ; aK<aNbIterMoy ; aK++)
                  aF = rect_som(aF,aSzVoisMoy) / double(ElSquare(1+2*aSzVoisMoy));


               ELISE_COPY
               (
                  aNappe.mPxRedr.all_pts(),
                  aF,
                  aNappe.mPxRedr.out()
               );
               if (0)
               {
                   Video_Win aW = Video_Win::WStd(aNappe.mPxRedr.sz(),1.0);
                   while (1)
                   {
                         double aDyn;
                         cin >> aDyn;
                         ELISE_COPY(aW.all_pts(),aNappe.mPxRedr.in()*aDyn,aW.ocirc());
                   }
               }
           }
/*
           Symb_FNum aSF(aNappe.mPxRedr.in() );
           Symb_FNum aSM(aIMasq.in(0)!=0);
           double aSomF = 0;
           double aSomM = 0;
           ELISE_COPY
           (
              aNappe.mPxRedr.all_pts(),
              Virgule(aSF*aSM,aSM),
              Virgule(sigma(aSomF),sigma(aSomM))
           );
           
           if (aSomM!=0) mMoyPxRed = aSomF / aSomM;
           std::cout << "MOY " << mMoyPxRed << "\n";
*/
        }
        else
        {
           ELISE_ASSERT(false,"Do Not Handle Redr Not Deq");
           for (int anX=0 ; anX<aSz.x ; anX++)
           {
               for (int anY=0 ; anY<aSz.y ; anY++)
               {
                   Pt2di aP(anX,anY);
                   double aPx = aNappe.mTPxInit.get(aP);
                   aPx =  aNappe.FromDiscPx(aPx);
                   aTRedrCur.oset(aP, aPx);
               }
           }
           cout << "!!!!!!!!!!  No-Dequant !!!!!! \n";
        }

     
        if (mRedrPx)
        {
            aNappe.mVPxMin = -mDilatAltiMoins  +aNappe.mIMoyPx;
            aNappe.mVPxMax =  mDilatAltiPlus+1+aNappe.mIMoyPx;
            ELISE_COPY(aNappe.mImPxMin.all_pts(),aNappe.mVPxMin,aNappe.mImPxMin.out());
            ELISE_COPY(aNappe.mImPxMax.all_pts(),aNappe.mVPxMax,aNappe.mImPxMax.out());
        }
    }

    // MODIF 5-2-2012 PASSE DESCENDU ICI
    // On initialise le resultat sur la valeur init, comme ca les
    // algo peuvent eventuellement ignorer une des 2 Px 
    ELISE_COPY
    (
        aNappe.mPxInit.all_pts(),
        Max
        (
           aNappe.mImPxMin.in(),
           Min
           (
                aNappe.mImPxMax.in()-1,
                aNappe.mPxInit.in()
           )
        ),
        aNappe.mPxRes.out()| aNappe.mPxInit.out()
    );


}



void cFilePx::ForceGenFileMA()
{
   mGenFileMA = true;
}

void cFilePx::SauvResulPx(Fonc_Num  aFonc,Box2di aBoxOut,Box2di aBoxIn) 
{ 
    if ((! mGenFile) && (!mGenFileMA))
       return;
    ELISE_COPY
    (
         rectangle(aBoxOut._p0,aBoxOut._p1),
         trans(aFonc,-aBoxIn._p0),
         FileIm().out() 
    );
}

void  cFilePx::SauvResulPxRel
      (
          Im2D_INT2 anIm,
          Im2D_INT2 anImInit,
          Box2di aBoxOut,
          Box2di aBoxIn
      )
{
    if (!  mGenFileRel)
       return;
    ELISE_COPY
    (
         rectangle(aBoxOut._p0,aBoxOut._p1),
         Max(0,Min(255,128+trans(anIm.in()-anImInit.in(),-aBoxIn._p0))),
         FileImRel().out()
    );

}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
