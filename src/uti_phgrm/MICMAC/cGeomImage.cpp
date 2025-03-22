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
#include "cModuleOrientation.h"
#include "cOrientationGrille.h"
#include "cOrientationRTO.h"
#include "cOrientationCon.h"




void ShowPoly(cElPolygone  aPoly)
{
  std::list<std::vector<Pt2dr> > aLC = aPoly.Contours();
  std::list<bool> aLH = aPoly.IsHole();
  std::list<bool>::iterator itH = aLH.begin();
  for( std::list<std::vector<Pt2dr> >::iterator itC=aLC.begin() ; itC!=aLC.end() ; itC++)
  {
       std::vector<Pt2dr> aC = *itC;
       std::cout << "H=" << ((*itH) ? "Hole" : "Fill") << "\n";
       for (int aK= 0 ; aK<int(aC.size()) ; aK++)
           std::cout << aC[aK] ;
       std::cout << "\n";
       itH++;
  }
}

/*****************************************/
/*                                       */
/*            cGeomBasculement3D         */
/*                                       */
/*****************************************/

cGeomBasculement3D::~cGeomBasculement3D()
{
}

/*****************************************/
/*                                       */
/*            cGeomImage                 */
/*                                       */
/*****************************************/


cGeomImage::cGeomImage
(
    const cAppliMICMAC & anAppli,
    cPriseDeVue &      aPDV,
    eTagGeometrie        aModeGeom,
    Pt2di aSzIm,
    int   aDimPx
) :
  mAppli      (anAppli),
  mPDV        (aPDV),
  mAnamSA     (anAppli.AnamSA()),
  mRC         (anAppli.RC()),
  mRCI        (anAppli.RCI()),
  mAnDeZoomM  (
                  anAppli.AnamorphoseGeometrieMNT().IsInit() ?
                  anAppli.AnamDeZoomMasq().Val()             :
                  -1
              ),
  mAnamSAIsInit  (false),
  mUseTerMasqAnam  (false),
  mDoImMasqAnam    (false),
  mMTA        (1,1),
  mTMTA       (mMTA),
  mTIncidTerr (Pt2di(1,1)),
  mDimPx      (aDimPx),
  mPIV_Done   (false),
  mCoeffDilNonE (0.0)
{    
   mSzImInit   =aSzIm;
  // std::cout <<  "KKKKK " << aPDV.Name() << " " << aSzIm  << " " << aPDV.SzIm() << "\n";
/*
   mContourIm.reserve(4);
   mContourIm.push_back(Pt2dr(0,0));
   mContourIm.push_back(Pt2dr(aSzIm.x,0));
   mContourIm.push_back(Pt2dr(aSzIm.x,aSzIm.y));
   mContourIm.push_back(Pt2dr(0,aSzIm.y));
*/
   SetDeZoomIm(1);
 
   if (mAnamSA!=0)
   {
       ELISE_ASSERT
       (
           mDimPx==1,
           "Anamorphose do not handle Dim Px 2"
       );
   }

}

bool cGeomImage::RPCIsVisible(const Pt3dr &) const
{
   return true;
}

Pt2dr cGeomImage::RPCGetAltiSolMinMax() const
{
   ELISE_ASSERT(false,"cGeomImage::RPCGetAltiSolMinMax");
   return Pt2dr(0,0);
}

bool cGeomImage::IsRPC() const 
{
    return false;
}

std::string cGeomImage::NameMasqImNadir(int aKB)
{
    return mAppli.FullDirPyr() + "MasqNadir_K" + ToString(aKB) +  "_" + mPDV.Name() + ".tif";
}

double cGeomImage::IncidTerrain(Pt2dr aPTer)
{
   ELISE_ASSERT(false,"cGeomImage::IncidTerrain");
   return 0;
}

void  cGeomImage::DoMasImNadir(TIm2D<REAL4,REAL8> &,cGeomDiscFPx &)
{
    ELISE_ASSERT(false,"cGeomImage::DoMasImNadir");
}

bool cGeomImage::MasqImNadirIsDone()
{

   ELISE_ASSERT(false,"cGeomImage::MasqImNadirIsDone");
   return false;
}

void cGeomImage::InitAnamSA(double aResol,const Box2dr & aBoxTer)
{
}

bool cGeomImage:: AcceptAnamSA() const
{
   return false;
}

bool cGeomImage::UseMasqTerAnamSA()
{
   return  mUseTerMasqAnam;
}

bool cGeomImage::IsInMasqAnamSA(Pt2dr aPTer)
{
   if (! mAnamSA)
      return true;
    ELISE_ASSERT(false,"cGeomImage::IsInMasqAnam");
    return false;
}

cGeomImageData cGeomImage::SauvData()
{
    return *this;
}

void cGeomImage::RestoreData(const cGeomImageData & aGID)
{
   static_cast<cGeomImageData &>(*this) = aGID;
}
  
cGeomImage::~cGeomImage()
{
}

Pt3dr cGeomImage::Bascule(const Pt3dr & aPIm) const
{
    double aPax[theDimPxMax] = {0,0};
    aPax[0] = aPIm.z;
    Pt2dr aPTer = ImageAndPx2Obj_Euclid(Pt2dr(aPIm.x,aPIm.y),aPax);
    return Pt3dr(aPTer.x,aPTer.y,aPIm.z);
}



ElSeg3D cGeomImage::FaisceauPersp(const Pt2dr & aPIm )  const
{
      // std::cout << "PXXXXXxxx " << mAppli 
      double aPx[2];
      aPx[1] = 0;
      double aPxMoy =  mAppli.GeomDFPxInit().V0Px()[0];

      double aPx0 = 0.99 *  aPxMoy;
      double aPx1 = 1.01 *  aPxMoy;

      // Grosse rustine
      if (ElAbs(aPx0-aPx1) < 1e-3)
      {
          aPx0 = aPxMoy - 1e-2;
          aPx1 = aPxMoy + 1e-2;
      }

      aPx[0] =   aPx0;
      Pt2dr aP2Ter0 = ImageAndPx2Obj_Euclid(aPIm,aPx);

      aPx[0] =   aPx1;
      Pt2dr aP2Ter1 = ImageAndPx2Obj_Euclid(aPIm,aPx);

      Pt3dr aP3Ter0(aP2Ter0.x,aP2Ter0.y,aPx0);
      Pt3dr aP3Ter1(aP2Ter1.x,aP2Ter1.y,aPx1);

      // ELISE_ASSERT(false,"cGeomImage::Faisceau non defini");
      return ElSeg3D(aP3Ter0,aP3Ter1);
}

Pt3dr cGeomImage::GeomFP2GeomFinale(const Pt3dr & aP)  const
{
   return aP;
}
Pt3dr cGeomImage::GeomFinale2GeomFP(const Pt3dr & aP)  const
{
   return aP;
}

Pt3dr cGeomImage::BasculeInv(const Pt3dr & aPTer) const
{
    double aPax[theDimPxMax] = {0,0};
    aPax[0] = aPTer.z;
    Pt2dr aPIm = Objet2ImageInit_Euclid(Pt2dr(aPTer.x,aPTer.y),aPax);
    return Pt3dr(aPIm.x,aPIm.y,aPTer.z);
}




const cGeomImage * cGeomImage::GeoTerrainIntrinseque() const
{
    return this;
}

cGeomImage * cGeomImage::NC_GeoTerrainIntrinseque() 
{
    return  const_cast<cGeomImage *>(GeoTerrainIntrinseque());
}


ElCamera * cGeomImage::GetCamera(const Pt2di & aSz,bool & ToDel,bool & aZUP) const
{
   ToDel = true;
   aZUP = true;
   return  cCameraOrtho::Alloc(aSz);
}


void ShowAff(const ElAffin2D & anAff)
{
   printf("[00] : %9.15f %9.15f \n",anAff.I00().x,anAff.I00().y);
   printf("[10] : %9.15f %9.15f \n",anAff.I10().x,anAff.I10().y);
   printf("[01] : %9.15f %9.15f \n",anAff.I01().x,anAff.I01().y);
}

void cGeomImage::RemplitOriXMLNuage
           (
                bool CallFromMere,
                const cMTD_Nuage_Maille &,
                const cGeomDiscFPx & aGT,
                cXML_ParamNuage3DMaille & aNuage,
                eModeExportNuage
           ) const
{
   // RPCNuageRPCNuageRPCNuage


   aNuage.PM3D_ParamSpecifs().NoParamSpecif().SetNoInit();
   cModeFaisceauxImage aMFI;
   aMFI.ZIsInverse() = false;
   aMFI.IsSpherik().SetVal(false);
   aMFI.DirFaisceaux() = Pt3dr(0,0,1);
   aNuage.PM3D_ParamSpecifs().ModeFaisceauxImage().SetVal(aMFI);

   bool ToDel;
   bool aZUP = false;
   ElCamera * aCam = GetCamera(aGT.SzDz(),ToDel,aZUP);


   Pt2dr anOrigine,aResol;
   aGT.SetOriResolPlani(anOrigine,aResol);
   ElAffin2D anAffC2M
             (
                anOrigine,
                Pt2dr(aResol.x,0.0),
                Pt2dr(0.0,aResol.y)
             );


   aCam->SetScanImaC2M(anAffC2M);

   // aNuage.Orientation().SetVal(aCam->StdExportCalibGlob());
   aNuage.Orientation() = aCam->StdExportCalibGlob(); // RPCNuageRPCNuage
   if (aZUP)
       aNuage.Orientation().ZoneUtileInPixel().SetVal(true); // RPCNuage
   if (ToDel)
      delete aCam;
}



Pt3dr cGeomImage::TerrainRest2Euclid(const Pt2dr & aP,double * aPax) const
{
   return Pt3dr(aP.x,aP.y,aPax[0]);
}

bool cGeomImage::IsId() const
{
  return false;
}

Pt2dr cGeomImage::CorrigeDist1(const Pt2dr & aP) const {return aP;}
Pt2dr cGeomImage::CorrigeDist2(const Pt2dr & aP) const {return aP;}
Pt2dr cGeomImage::InvCorrDist1(const Pt2dr & aP) const {return aP;}
Pt2dr cGeomImage::InvCorrDist2(const Pt2dr & aP) const {return aP;}
void cGeomImage::CorrectModeleAnalytique(cModeleAnalytiqueComp *)
{
    ELISE_ASSERT
    (
         false,
         "Geometrie ne comprend CorrectModeleAnalytique"
    );
}

void cGeomImage::SetDeZoomIm(int aDeZoom)
{
   mDeZoom = aDeZoom;
   mSzImDz = mSzImInit/mDeZoom;
   mP0Clip = Pt2di(0,0);
   mP1Clip = mSzImDz;
}

Pt2dr cGeomImage::CurObj2Im(Pt2dr aPTer,const REAL * aPx) const
{

   return Objet2ImageInit_Euclid(aPTer,aPx)/double(mDeZoom) - Pt2dr(mP0Clip);
}


bool cGeomImage::GetRatioResolAltiPlani(double * aRatio) const
{
    for (int aK=0 ; aK<mDimPx; aK++)
        aRatio[aK] = 1;
    return true;
}

void cGeomImage::PostInitVirtual(const std::vector<cModGeomComp *> & aVM)
{
   if (mPIV_Done)  // Evite la recursion infinie
      return;
   mPIV_Done = true;


   // mContourIm = EmpriseImage();
   cElPolygone aPol1;
   aPol1.AddContour(EmpriseImage(),false);
   cElPolygone aPol2;
   aPol2.AddContour(cGeomImage::EmpriseImage(),false);

   cElPolygone aPol12 = aPol1 * aPol2;

   mContourIm = aPol12.ContSMax();


/*
for (int aK=0; aK<int(mContourIm.size()) ; aK++)
{
   std::cout << "cGeomImage::EmpriseTerrain " <<mContourIm[aK] << "\n";
}
getchar();
*/


   NC_GeoTerrainIntrinseque()->PostInitVirtual(aVM);
   mIsIntrinseque = (this == GeoTerrainIntrinseque());

   mCoeffDilNonE = 1.0;

   if (mIsIntrinseque )
   {
      mVM = aVM;
      for (int aK=0 ; aK<int(mVM.size()) ; aK++)
          mCoeffDilNonE *= mVM[aK]->CoeffDilNonE();
   }
}

double cGeomImage::GetResolNonEuclidBySurfPixel() const
{
  static double aPx0[2]={0,0};
  Pt2dr aPMil = mSzImInit / 2.0;


  // Pt2dr aQ0 = ImageAndPx2Obj_NonEuclid(aPMil,aPx0);
  // Pt2dr aQ0x = ImageAndPx2Obj_NonEuclid(aPMil+Pt2dr(1,0),aPx0);
  // Pt2dr aQ0y = ImageAndPx2Obj_NonEuclid(aPMil+Pt2dr(0,1),aPx0);

  Pt2dr aQ0 =  ImageAndPx2Obj_Euclid(aPMil,aPx0);
  Pt2dr aQ0x = ImageAndPx2Obj_Euclid(aPMil+Pt2dr(1,0),aPx0);
  Pt2dr aQ0y = ImageAndPx2Obj_Euclid(aPMil+Pt2dr(0,1),aPx0);

  double aSurfOr = (aQ0x-aQ0) ^ (aQ0y-aQ0);

// std::cout << aPMil << aQ0 << aQ0x << aQ0y << aSurfOr << "\n";

   return sqrt(ElAbs(aSurfOr));
}

double cGeomImage::GetResolMoyenne_Euclid() const
{
  //std::cout << mCoeffDilNonE << " " << GetResolMoyenne_NonEuclid() << "\n";
   ELISE_ASSERT(mCoeffDilNonE!=0,"GetResolMoyenne_Euclid");


   return mCoeffDilNonE *GetResolMoyenne_NonEuclid();
}

void cGeomImage::PostInit()
{
    double  aOwnPx[2];
    bool    CanCal =  GetPxMoyenne_Euclid(aOwnPx,mAppli.InversePx());
    if (mAnamSA) CanCal = 0;
    if (! mAppli.Prio2OwnAltisolForEmprise().Val()) CanCal =0;



    const double * aPx0 =  CanCal ? aOwnPx : mAppli.GeomDFPxInit().V0Px();
    for (int aK=0 ; aK<int(mContourIm.size()) ; aK++)
    {
       Pt2dr aPter= ImageAndPx2Obj_Euclid(mContourIm[aK],aPx0); 
// std::cout <<  "xxxxxx   " << mContourIm[aK] << aPter << "\n";
       mContourTer.push_back(aPter);

       if (aK==0)
       {
          mBoxTerPx0._p0 = aPter;
          mBoxTerPx0._p1 = aPter;
       }
       else
       {
          mBoxTerPx0._p0.SetInf(aPter);
          mBoxTerPx0._p1.SetSup(aPter);
       }
    }
    if (mUseTerMasqAnam)
    {
        mContourTer = mAnamSAPMasq.BoxTer().Contour();
       //mAnamPMasq =  // StdGetObjFromFile<cParamMasqAnam>
      //return mAnamPMasq.BoxTer();
    }

// std::cout << "CCCCCcccterrrrrr " << mContourTer.size() << "\n";
// std::cout << mBoxTerPx0._p0 << " " << mBoxTerPx0._p1 << "\n";
// std::cout << " F " << CanCal << " DDDDDDDDDDD "<< aPx0[0] << " " << aPx0[1] << "\n";
// getchar();

    mPolygTerPx0.AddContour(mContourTer,false);

/*
std::cout << " EEEEEE mPolygTerPx0 \n";
ShowPoly(mPolygTerPx0);
getchar();
*/

    InstPostInit();

}

void cGeomImage::InstPostInit()
{
}


std::string cGeomImage::Name() const
{
   return "XNoName";
}


Box2dr cGeomImage::BoxImageOfVPts
       (
         const std::vector<Pt2dr> & aVPts,
         REAL aZoom,bool isSensTer2Im,
         const REAL * aMinPx,const REAL * aMaxPx,
         double aRab
     )  const
{
    double  aVPx[theDimPxMax] ={0,0};

    Pt2dr aPMax;
    Pt2dr aPMin;

    // Pour explorer les 2^(2+mDimPx) coins de cubes on
    // passe par un flag de bits
    for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
    {
        for (int aFlagBit=0 ; aFlagBit < (1<<mDimPx) ; aFlagBit ++)
        {
            for (int aD = 0 ; aD< mDimPx ; aD++)
                aVPx[aD] = (aFlagBit & (1<<aD)) ?  aMinPx[aD] : aMaxPx[aD] ;
            Pt2dr aP = isSensTer2Im                          ?
                       Objet2ImageInit_Euclid(aVPts[aKP],aVPx)  :
                       ImageAndPx2Obj_Euclid(aVPts[aKP],aVPx)   ;


            aP = aP/ aZoom;

            if ((aFlagBit==0) && (aKP==0))
            {
                aPMin = aP;
                aPMax = aP;
            }
            else
            {
               aPMin.SetInf(aP);
               aPMax.SetSup(aP);
            }
        }
    }

    return Box2dr
           (
              aPMin - Pt2dr(aRab,aRab),
              aPMax + Pt2dr(aRab,aRab)
           );
}

Box2dr cGeomImage::BoxImageOfBox
       (
         Pt2dr aPMinObj,Pt2dr aPMaxObj,
         REAL aZoom,bool isSensTer2Im,
         const REAL * aMinPx,const REAL * aMaxPx,
         double aRab
     )  const
{
   Box2dr aBox(aPMinObj,aPMaxObj);
   Pt2dr aP4Im[4];
   aBox.Corners(aP4Im);

   std::vector<Pt2dr> aVC(aP4Im,aP4Im+4);
   return BoxImageOfVPts
          (
               aVC,aZoom,isSensTer2Im,
               aMinPx,aMaxPx,aRab
          );
   
}

std::string cGeomImage::NameMasqAnamSA(const std::string & aPost) const
{
   return mAppli.FullDirMEC() + "Anam_" +  mAppli.NameChantier() + "_" + StdPrefix(mPDV.Name())+aPost;
}


std::vector<Pt2dr>  cGeomImage::EmpriseImage() const
{
   return Box2dr(Pt2dr(0,0),Pt2dr(mSzImInit)).Contour();
}


Box2dr cGeomImage::EmpriseTerrain
       (
            const REAL * aMinPx,const REAL * aMaxPx,
            double aRab
       )  const
{


    if (mUseTerMasqAnam)
      return mAnamSAPMasq.BoxTer();


    Box2dr aRes= BoxImageOfVPts
                 (
                    // Pt2dr(0,0),mSzImInit,
                    mContourIm,
                    1.0,false,
                    aMinPx,aMaxPx,
                    aRab
                 );
// std::cout << "cGeomImage::EmpriseTerrain123 " << aRes._p0 << " " << aRes._p1 << "\n";
     return aRes;
}


Box2dr cGeomImage::BoxImageOfBoxTerrainWithContSpec
       (
         Pt2dr aPMinObj,Pt2dr aPMaxObj,
         const REAL * aMinPx,const REAL * aMaxPx,
         double aRabGeom,
         double aRabInterp
       )  const
{
    if (mAppli.UseConstSpecIm1())
    {
        std::vector<Pt2dr>  aC = Box2dr(aPMinObj,aPMaxObj).ClipConpMax(mAppli.ContSpecIm1());
        if (aC.size()==0)
        {
           return Box2dr(mSzImDz/2,mSzImDz/2);
        }
        else
        {
             return  BoxImageOfVPts
                    (
                        aC,
                        mDeZoom,true,
                        aMinPx,aMaxPx,
                        aRabGeom/mDeZoom + aRabInterp
                    );
        }
    }
    return  BoxImageOfBox
            (
                aPMinObj,aPMaxObj,
                mDeZoom,true,
                aMinPx,aMaxPx,
                aRabGeom/mDeZoom + aRabInterp
            );
}


void cGeomImage::SetClip
     (
         Pt2dr aPMinObj,Pt2dr aPMaxObj,
         const REAL * aMinPx,const REAL * aMaxPx,
         double aRabGeom,
         double aRabInterp
     )
{
   // Pour ?? eviter des pb de singularite dans les rayons

   Box2dr aBoxTer =  BoxImageOfVPts
                  (
                       mContourIm,
                       1,false,
                       aMinPx,aMaxPx,
                       0
                  );
   aPMinObj.SetSup(aBoxTer._p0);
   aPMaxObj.SetInf(aBoxTer._p1);


    Box2dr aBox  =  BoxImageOfBoxTerrainWithContSpec
                    (
                        aPMinObj,aPMaxObj,aMinPx,aMaxPx,aRabGeom,aRabInterp
                    );
/*
    if (mAppli.UseConstSpecIm1())
    {
        std::vector<Pt2dr>  aC = Box2dr(aPMinObj,aPMaxObj).ClipConpMax(mAppli.ContSpecIm1());
        if (aC.size()==0)
        {
              aBox._p0 = aBox._p1 = mSzImDz/2;
        }
        else
        {
             aBox = BoxImageOfVPts
                    (
                        aC,
                        mDeZoom,true,
                        aMinPx,aMaxPx,
                        aRabGeom/mDeZoom + aRabInterp
                    );
        }
    }
    else
    {
        aBox = BoxImageOfBox
                  (
                       aPMinObj,aPMaxObj,
                       mDeZoom,true,
                       aMinPx,aMaxPx,
                       aRabGeom/mDeZoom + aRabInterp
                  );
    }
*/

    mP0Clip = Sup(Pt2di(0,0),round_down(aBox._p0));
    mP1Clip = Inf(mSzImDz,round_up(aBox._p1));


   // std::cout << aBox._p0 << aBox._p1 << "\n";
// std::cout <<  mP0Clip << mP1Clip << "\n";
//  std::cout << "a" << aPMinObj << " " << aPMaxObj << "\n"; getchar();
// std::cout << aMinPx[0] << " " << aMaxPx[0] << "\n";

}

void  cGeomImage::SetClip
      (
          const cGeomDiscFPx & aFPx,
          Pt2di aPVgn,
          const int * aMinPxDisc,const int * aMaxPxDisc,
          double aRabGeom,
          double aRabInterp
      )
{
    double  aRPxMin[theDimPxMax];
    double  aRPxMax[theDimPxMax];
    // Pt2di   aPVgn(aSzVgn,aSzVgn);

    Pt2dr aP0Obj = aFPx.DiscToR2(Pt2di(0,0)-aPVgn);
    Pt2dr aP1Obj = aFPx.DiscToR2(aFPx.SzClip()+aPVgn);
    pt_set_min_max(aP0Obj,aP1Obj);
    aFPx.PxDisc2PxReel(aRPxMin,aMinPxDisc);
    aFPx.PxDisc2PxReel(aRPxMax,aMaxPxDisc);

    SetClip
    (
       aP0Obj, aP1Obj,
       aRPxMin,aRPxMax,
       aRabGeom,aRabInterp
    );
}


bool cGeomImage::BoxTerHasIntersection
     (
          const cGeomDiscFPx & aFPx,
          const int * aMinPxDisc,const int * aMaxPxDisc,
          Box2dr aBoxTerRas
     ) const
{
    double  aRPxMin[theDimPxMax];
    double  aRPxMax[theDimPxMax];

    Pt2dr aP0Obj = aFPx.RDiscToR2(aBoxTerRas._p0);
    Pt2dr aP1Obj = aFPx.RDiscToR2(aBoxTerRas._p1);
    pt_set_min_max(aP0Obj,aP1Obj);
    aFPx.PxDisc2PxReel(aRPxMin,aMinPxDisc);
    aFPx.PxDisc2PxReel(aRPxMax,aMaxPxDisc);

   Box2dr aBoxIm = BoxImageOfBoxTerrainWithContSpec
                   (
                       aP0Obj,aP1Obj,
                       aRPxMin,aRPxMax,
                       0,0
                   );
/*
    Box2dr aBoxIm = BoxImageOfBox
                    (
                         aP0Obj,aP1Obj,
                         mDeZoom,true,
                         aRPxMin,aRPxMax,
                         0.0
                    );
*/

   return     (aBoxIm._p1.x >=0.0)
           && (aBoxIm._p1.y >=0.0)
           && (aBoxIm._p0.x <= mSzImDz.x)
           && (aBoxIm._p0.y <= mSzImDz.y)
           && (aBoxIm._p1.x >aBoxIm._p0.x)
           && (aBoxIm._p1.y >aBoxIm._p0.y);
}


bool cGeomImage::ClipIsEmpty() const
{
  return    (mP0Clip.x >= mP1Clip.x)
         || (mP0Clip.y >= mP1Clip.y);
}

Box2di cGeomImage::BoxClip() const
{
   return Box2di(mP0Clip,mP1Clip);
}

int cGeomImage::DimPx() const
{
   return mDimPx;
}

bool cGeomImage::GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
{
   return false;
}

bool cGeomImage::GetPxMoyenneNulle(double * aPxMoy) const
{
   for (int aD=0; aD<mDimPx; aD++)
       aPxMoy[aD]=0;
   return true;
}

void cGeomImage::RemplitOri(cFileOriMnt & aFOM) const
{
}


static double aPax00[theDimPxMax]={0.0,0.0};
Pt2dr cGeomImage::Direct(Pt2dr aP) const
{
   return  Objet2ImageInit_Euclid(aP,aPax00);
}
bool cGeomImage::OwnInverse(Pt2dr & aP) const 
{
   aP= ImageAndPx2Obj_Euclid(aP,aPax00);
   return true;
}


CamStenope *  cGeomImage::GetOri()  const
{
   return 0;
}

CamStenope *  cGeomImage::GetOriNN()  const
{
   CamStenope * aRes = GetOri();
   ELISE_ASSERT(aRes,"Impossible d'extraire l'Ori d'une geometrie");
   return aRes;
}


bool  cGeomImage::DirEpipTransv(Pt2dr &) const
{
   return false;
}



const cElPolygone        &  cGeomImage::PolygTerPx0() const
{
   return mPolygTerPx0;
}

const Box2dr             &  cGeomImage::BoxTerPx0() const
{
   return mBoxTerPx0;
}

const std::vector<Pt2dr> &  cGeomImage::ContourIm() const
{
   return mContourIm;
}

const std::vector<Pt2dr> &  cGeomImage::ContourTer() const
{
   return mContourTer;
}


double cGeomImage::CoeffDilNonE() const
{
  return mCoeffDilNonE;
}


bool  cGeomImage::IntersectEmprTer
      (
           const cGeomImage & aGeo2,
           Pt2dr & aPMoyEmpr,
           double * aSurf
      ) const
{
   if (aSurf)
      *aSurf = 0;
   if (InterVide(BoxTerPx0(),aGeo2.BoxTerPx0()))
   {
      return false;
   }


   cElPolygone aPolInter = PolygTerPx0() * aGeo2.PolygTerPx0();
   const std::list<cElPolygone::tContour> & aContInter = aPolInter.Contours();

   if (aContInter.empty())
      return false;


   double aSomS = 0;
   Pt2dr  aSomP(0,0);
   // std::cout << "============================mmmLLmmmIo====================\n";
   for (std::list<cElPolygone::tContour>::const_iterator itP=aContInter.begin(); itP!=aContInter.end();itP++)
   {
       double aS0 = surf_or_poly(*itP);
       if (aSomS && aS0)
       {
           // ELISE_ASSERT( (aSomS<0)==(aS0<0) , "Intersection d'emprises incoherente");
       }

       aSomS += aS0;
       aSomP = aSomP +  barrycentre(*itP) * aS0;


       // std::cout << "SSurff " << aS0 << "\n";
   }

   aPMoyEmpr =  aSomP / aSomS;
   if (aSurf)
   {
      *aSurf = ElAbs(aSomS);
   }


/*
   if (aContInter.size()!=1)
   {
       // std::cout << "For Cple = " << mName <<  " " << aGeo2.mName << "\n";
       ELISE_ASSERT(aContInter.size()==1,"Intersection d'emprises incoherente");
   }
   aPMoyEmpr = barrycentre(*(aContInter.begin()));
   if (aSurf)
   {
      *aSurf = ElAbs(surf_or_poly(*(aContInter.begin())));
   }
*/
   

   return true;
}
/*
void  cGeomImage::EmprTerInters(const cGeomImage & aGeo2,std::vector<Pt2dr> & aV)
{
}
*/

double cGeomImage::BSurH
       (
          const cGeomImage & aGeo2,
          const Pt2dr & aPTer
       ) const
{
    return BSurH(aGeo2,aPTer,mAppli.GeomDFPxInit().V0Px()[0]);
}

double cGeomImage::BSurH
       (
          const cGeomImage & aGeo2,
          const   Pt2dr &    aPTerA,
          double             aZA
       ) const
{
   
     double aPxA[theDimPxMax] ={0,0};
     aPxA[0] = aZA;

     double aZB = aZA-10;
     double aPxB[theDimPxMax] ={0,0};
     aPxB[0] = aZB;

     Pt2dr aPIm1 = Objet2ImageInit_Euclid(aPTerA,aPxA);
     Pt2dr aPTer1B = ImageAndPx2Obj_Euclid(aPIm1,aPxB);

     Pt2dr aPIm2 =  aGeo2.Objet2ImageInit_Euclid(aPTerA,aPxA);
     Pt2dr aPTer2B = aGeo2.ImageAndPx2Obj_Euclid(aPIm2,aPxB);


     double aRes = euclid(aPTer1B,aPTer2B) / ElAbs(aZA-aZB);

     return aRes;
}

// Pour l'instant, la version Eulid ne fait que rappeler la version NonEuclid

Pt3dr   cGeomImage::CentreRestit() const
{
    Pt3dr aP = Centre();
    if (mAnamSA)  
       aP = mAnamSA->E2UVL(aP);
    else   if (mRCI)
       aP = mRCI->FromLoc(aP);

    return aP;
}

bool  cGeomImage::HasCentre() const
{
   return false;
}

Pt3dr  cGeomImage::Centre() const
{
   ELISE_ASSERT(false,"cGeomImage::Centre");
   return Pt3dr(0,0,0);
}

Pt2dr cGeomImage::ImageAndPx2Obj_Euclid(Pt2dr aPIm,const REAL * aPx) const
{

    for (int aK =0 ; aK<int(mVM.size()) ; aK++)
        aPIm = mVM[aK]->Modif2GeomInit(aPIm);

    if (mAnamSA)
    {
         ElSeg3D aSeg = FaisceauPersp(aPIm);

         //int aNbSol;
         //Pt3dr aPTter = mAnam->SegAndL(aSeg,aPx[0],aNbSol);
         Pt3dr aPTter = mAnamSA->BestInterDemiDroiteVisible(aSeg,aPx[0]);
         return Pt2dr(aPTter.x,aPTter.y);
    }
    if (mRCI)
    {
         ElSeg3D aSeg = FaisceauPersp(aPIm);
         Pt3dr aPT0 = mRCI->FromLoc(aSeg.P0());
         Pt3dr aV01 = mRCI->FromLoc(aSeg.P1()) - aPT0;

         double aLamda = (aPx[0]-aPT0.z) / aV01.z;

         Pt3dr aPTer = aPT0  + aV01 * aLamda;
         return Pt2dr(aPTer.x,aPTer.y);
    }
    
    return  ImageAndPx2Obj_NonEuclid(aPIm,aPx);
}

Pt2dr cGeomImage::Objet2ImageInit_Euclid(Pt2dr aPTer,const REAL * aPx) const
{

    Pt2dr aPIm ;
    if (mRC)
    {
        Pt3dr aPEucl = mRC->FromLoc(Pt3dr(aPTer.x,aPTer.y,aPx[0]));
        aPIm =  Objet2ImageInit_NonEuclid(Pt2dr(aPEucl.x,aPEucl.y),&aPEucl.z);
    }
    else if (mAnamSA)
    {
        Pt3dr aPEucl = mAnamSA->UVL2E(Pt3dr(aPTer.x,aPTer.y,aPx[0]));
        aPIm =  Objet2ImageInit_NonEuclid(Pt2dr(aPEucl.x,aPEucl.y),&aPEucl.z);
    }
    else
    {
        aPIm =  Objet2ImageInit_NonEuclid(aPTer,aPx);
    }
    for (int aK =int(mVM.size())-1  ; aK>=0  ; aK--)
        aPIm = mVM[aK]->Modif2GeomActu(aPIm);

   return aPIm;
}

Pt3dr cGeomImage::Restit2Euclid(const Pt2dr & aP,double * aPax) const
{
    Pt3dr aPEucl(aP.x,aP.y,aPax[0]);
    if (mAnamSA)
    {
        aPEucl = mAnamSA->UVL2E(aPEucl);
    }
   
   return TerrainRest2Euclid(Pt2dr(aPEucl.x,aPEucl.y),&aPEucl.z);
}



bool cGeomImage::GetPxMoyenne_Euclid(double * aPxMoy,bool MakeInvIfNeeded) const
{
    bool aRes =  GetPxMoyenne_NonEuclid(aPxMoy,MakeInvIfNeeded);

    // std::cout << "Qqqqqqqqqqqqqqqqq " << aRes << " " << aPxMoy[0] << "\n";

    return aRes;
}

   //-----------------------------
   //
   //    cGeomPxToP2
   //
   //-----------------------------

class cGeomPxToP2 : public ElDistortion22_Gen
{
    public :
       cGeomPxToP2(Pt2dr aP1,const cGeomImage &,Pt2dr aPx0);
       Pt2dr Direct(Pt2dr) const ;
       void  Diff(ElMatrix<REAL> &,Pt2dr) const;
       Pt2dr GuessInv(const Pt2dr & aP) const {return Pt2dr(0,0);}
    private :
       Pt2dr mP1;
       Pt2dr mPx0;
       const cGeomImage & mGeom;
};

cGeomPxToP2::cGeomPxToP2
(
        Pt2dr aP1,
        const cGeomImage & aGeom,
        Pt2dr aPx0
) :
   mP1   (aP1),
   mPx0  (aPx0),
   mGeom (aGeom)
{
}

Pt2dr cGeomPxToP2::Direct(Pt2dr aPPx) const 
{
   double aVPx[theDimPxMax];
   aVPx[0] = aPPx.x +mPx0.x;
   aVPx[1] = aPPx.y +mPx0.y;
   Pt2dr aRes =  mGeom.Objet2ImageInit_Euclid(mP1,aVPx);
   // std::cout << "Res " << mP1 << aPPx << aRes << "\n";
   return aRes;
}

void  cGeomPxToP2::Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
{
   
   DiffByDiffFinies(aMat,aP,1e-3);

/*
  std::cout << "GI " << GuessInv(aP) << "\n";
 std::cout << "P=" << aP << "\n";
 std::cout <<  Direct(aP) 
           <<  Direct() - Direct(Pt2dr(0,0)) 
           <<  Direct(Pt2dr(0,1e-3)) - Direct(Pt2dr(0,0)) << "\n";
     std::cout << "#Dif : " << aMat(0,0) << " " << aMat(0,1) << "\n";
     std::cout << " Dif : " << aMat(1,0) << " " << aMat(1,1) << "\n";
      getchar();
*/
}



Pt2dr cGeomImage::P1P2ToPx(Pt2dr aP1,Pt2dr aP2) const
{
   ELISE_ASSERT
   (
        mAppli.DimPx()==2,
        "Bad Dim in cGeomImage::P1P2ToPx"
   );
   Pt2dr aPx0 (
                 mAppli.GeomDFPxInit().V0Px()[0],
                 mAppli.GeomDFPxInit().V0Px()[1]
              );
   cGeomPxToP2 aGPx(aP1,*this,aPx0);
   return aPx0 + aGPx.Inverse(aP2);
   
}


// Voir en fin de fichier pour les allocateur

/*****************************************/
/*                                       */
/*            cGeomImage_Id              */
/*                                       */
/*****************************************/

//  En appariement image/image, les coordonnees objet
//  sont confondues avec les coordonnees de la premiere image.
//  Il s'agit donc d'une geometrie "Identite".

class cGeomImage_Id : public cGeomImage
{
    public :
      cGeomImage_Id
      (
          const cAppliMICMAC & anAppli,
          cPriseDeVue &      aPDV,
          eTagGeometrie        aTag,
          Pt2di aSzIm,
          int   aDimPx
      )  :
         cGeomImage(anAppli,aPDV,aTag,aSzIm,aDimPx)
      {
      }

       std::string Name() const { return "cGeomImage_Id"; }

    private :
       virtual Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          return aP;
       }
       virtual Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          return aP;
       }
       virtual double GetResolMoyenne_NonEuclid() const 
       {
          return 1.0;
       }
       bool IsId() const {return true;}

       void CorrectModeleAnalytique(cModeleAnalytiqueComp *) {}
};

/*****************************************/
/*                                       */
/*            cGeomImage_NoGeom          */
/*                                       */
/*****************************************/

class cGeomImage_NoGeom : public cGeomImage
{
    public :
      cGeomImage_NoGeom
      (
         const cAppliMICMAC & anAppli,
         cPriseDeVue &      aPDV,
          Pt2di aSzIm
      )  :
         cGeomImage(anAppli,aPDV,eTagNoGeom,aSzIm,2)
      {
      }
       std::string Name() const { return "cGeomImage_NoGeom"; }

    private :
       virtual Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          ELISE_ASSERT(false,"cGeomImage_NoGeom::ImageAndPx2Obj_NonEuclid");
          return aP;
       }
       virtual Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          ELISE_ASSERT(false,"cGeomImage_NoGeom::Objet2ImageInit_NonEuclid");
          return aP;
       }
       virtual double GetResolMoyenne_NonEuclid() const 
       {
          ELISE_ASSERT(false,"cGeomImage_NoGeom::GetResolMoyenne");
          return 1.0;
       }

};

/*****************************************/
/*                                       */
/*            cGeomImage_DHD_Px          */
/*                                       */
/*****************************************/

//    Le modele DHD_Px correspond au cas ou apres
// correction d'une eventuelle distorsion, le modele
// de correspondance a priori est celui d'une homographie
// 
// Cela inclut l'autocalibration sur le mur (scene quasi-plane)
// la mise en correspondance des cannaux de la camera
// numerique (centre optique quasi confondus) ou la 
// photogrammetrie elementaire montre que pour une camera
// parfaite (donc apres correction de la distortion) la 
// correspondance est une homographie.
//
//    Cela inclut aussi des modeles plus elementaire, avec
// Dist=Identite, et homographie eventuellement reduite
// a une application lineaire, voire une homotetie rotation,
// voir  une translation, voir l'identite. Par exemple
// la mise en correspondance d'image satellite pour
// un couple le long de la trace ou en l'absence de 
// modele physique un modele approche peut etre trouve
// par simple translation.
//
//    On passe en parametre les deux distorsion sous forme
// de pointeur sur des Grille, si c'est le pointeur null
// on utilisera une correction identite.
//    L'Homographie est passee sous forme de couples homologues,
// qui seront corrige des eventuelles distorsion, ensuite l'homographie
// est estimee selon les conventions "habituelles" :
//    - 0 couple       : identite
//    - 1 couple       : translation
//    - 2 couples      : homotetie-translation
//    - 3 couples      : application affine
//    - 4 couples      : homographie (exactes)
//    - + de 4 couples : homographie (ajustees par moindres carres)


/*
{
}

*/


class cGeomImage_DHD_Px : public cGeomImage 
{
    public :
      std::string Name() const { return "cGeomImage_DHD_Px"; }
      cGeomImage_DHD_Px
      (
          const                   cAppliMICMAC & anAppli,
          cPriseDeVue &           aPDV,
          Pt2di                   aSzIm,
          cDbleGrid *             aPGr1,
          cDbleGrid *             aPGr2,
          const ElPackHomologue & aPack,
	  int                     aDimPx
      )  :
         cGeomImage(anAppli,aPDV,eTagDHD_Px,aSzIm,aDimPx),
         mPGr1 (aPGr1),
         mPGr2 (aPGr2),
         mH1To2 (cElHomographie::Id()),
         mH2To1 (cElHomographie::Id()),
         mModA  (0)
      {
         ElPackHomologue aPackCorr;
         for  
         (
              ElPackHomologue::const_iterator itC = aPack.begin();
              itC != aPack.end();
              itC++
         )
         {
            aPackCorr.Cple_Add
            (
                 ElCplePtsHomologues
                 (
                     Corr1(itC->P1()),
                     Corr2(itC->P2()),
                     itC->Pds()
                 )
            );
         }

         bool L2Estim=true;  // Eventuellement laisser le choix
         mH1To2 = cElHomographie(aPackCorr,L2Estim);

         mH2To1 = mH1To2.Inverse();

      }
      ~cGeomImage_DHD_Px()
      {
           delete mPGr1;
           delete mPGr2;
      }


    private :
       void CorrectModeleAnalytique(cModeleAnalytiqueComp * aModA)
       {
            mModA = aModA;
       }

       Pt2dr Corr1(const Pt2dr  & aP) const
       {
            return mPGr1 ? mPGr1->Direct(aP) : aP;
       }
       Pt2dr InvCorr1(Pt2dr aP) const
       {
           return mPGr1 ? mPGr1->Inverse(aP) : aP;
       }
       Pt2dr Corr2(const Pt2dr  & aP) const
       {
            return mPGr2 ? mPGr2->Direct(aP) : aP;
       }
       Pt2dr InvCorr2(Pt2dr aP) const
       {
           return mPGr2 ? mPGr2->Inverse(aP) : aP;
       }


      // inline double Px1(const REAL * aPx) const {return (mDimPx > 1) ? aPx[1] : 0;}

       virtual Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          aP = mH2To1.Direct(Corr2(aP-Pt2dr(aPx[0],Px1(aPx))));
          if (mModA)
             aP =mModA->CorrecInverse(aP);
          return InvCorr1(aP);
       }
       virtual Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          aP = Corr1(aP);
          if (mModA)
             aP =mModA->CorrecDirecte(aP);
          return   InvCorr2(mH1To2.Direct(aP))+Pt2dr(aPx[0],Px1(aPx));
       }
       virtual double GetResolMoyenne_NonEuclid() const 
       {
          return 1.0;
       }
       bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
       {
            return GetPxMoyenneNulle(aPxMoy);
       }

       Pt2dr CorrigeDist1(const Pt2dr & aP1) const
       {
           return Corr1(aP1);
       }
       Pt2dr InvCorrDist1(const Pt2dr & aP2) const
       {
           return InvCorr1(aP2);
       }

       Pt2dr CorrigeDist2(const Pt2dr & aP2) const
       {
           return mH2To1.Direct(Corr2(aP2));
       }
       Pt2dr InvCorrDist2(const Pt2dr & aP1) const
       {
           return InvCorr2(mH1To2.Direct(aP1));
       }

       // double I

      cDbleGrid *             mPGr1;
      cDbleGrid *             mPGr2;
      cElHomographie          mH1To2;
      cElHomographie          mH2To1;
      cModeleAnalytiqueComp * mModA;
};

/*****************************************/
/*                                       */
/*            cGeomImage_Terrain_Ori     */
/*                                       */
/*****************************************/

class cGeomImage_Terrain_Ori : public cGeomImage
{
    public :
      std::string Name() const { return "cGeomImage_Terrain_Ori"; }
      Pt2dr ToGeomMasqAnam(const Pt2dr & aPTer) const;

      std::string NameMasqImNadir() { return cGeomImage::NameMasqImNadir(mAppli.MMImNadir()->KBest()); }


      cGeomImage_Terrain_Ori
      (
          const cAppliMICMAC & anAppli,
          cPriseDeVue &      aPDV,
          Pt2di aSzIm,
          CamStenope *  anOri,
          bool      isCarto
      )  :
         cGeomImage (anAppli,aPDV,eTagGeom_TerrainOri,aSzIm,1),
         mOri       (anOri),
         mIsCarto   (isCarto),
         mOLiLi     (mOri->CastOliLib()),
         mLoadIncT  (false),
         mNameIncTerTif   (NameMasqAnamSA("_IncidTer.tif"))
      {
          if (mIsCarto)
          {
              ELISE_ASSERT
              (
                    mOLiLi!=0,
                    "Pas de support geom carto hors orilib"
              );
          }
          Init0MasqAnamSA();
      }

      bool IsInMasqAnamSA(Pt2dr aPTer) ;
      void Init0MasqAnamSA();
      void InitAnamSA(double aResol,const Box2dr & aBoxTer);
      bool MasqImNadirIsDone();
      void DoMasImNadir(TIm2D<REAL4,REAL8> &,cGeomDiscFPx &);


      std::vector<Pt2dr>  EmpriseImage() const
      {
          return mOri->ContourUtile();
      }


    protected :
       Pt3dr  Centre() const
       {
          return  mOri->PseudoOpticalCenter();
       }

       bool HasCentre() const
       {
            return true;
       }

       virtual Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
/// std::cout << "aaaa"  << mOri << "\n";
/// std::cout << "bbbb"  << aPx << "\n";
/// std::cout << "cccc"  << aPx[0] << "\n";

          // OO  Pt3dr aPTer =  mOri.to_terrain(aP,aPx[0]);
          Pt3dr aPTer =  mOri->ImEtZ2Terrain(aP,aPx[0]);
          // OO  (mIsCarto)
          // OO aPTer = mOri.terr_to_carte(aPTer);
          if (mOLiLi && mIsCarto)
             aPTer = mOLiLi->terr_to_carte(aPTer);
          return Pt2dr(aPTer.x,aPTer.y);
       }
       virtual Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          Pt3dr aPTer(aP.x,aP.y,aPx[0]);
          //OO  (mIsCarto)
          //OO aPTer = mOri.carte_to_terr(aPTer);
          if (mOLiLi && mIsCarto)
             aPTer = mOLiLi->carte_to_terr(aPTer);
          //OO return mOri.to_photo(aPTer);
          return mOri->R3toF2(aPTer);
       }

       virtual bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
       {
	   // OO double aZ = mOri.altisol();
	   double aZ = mOri->GetAltiSol();

           if (! ALTISOL_IS_DEF(aZ))
           {
              std::cout << "FOR " << mPDV.Name() << "\n";
	      ELISE_ASSERT(false,"ALTISOL_IS_DEF GetPxMoyenne_NonEuclid");
           }
           aPxMoy[0] = aZ;
           return true;
       }
       bool  AcceptAnamSA() const
       {
          return true;
       }

      
       double IncidTerrain(Pt2dr aPTer);  // Interf virtuelle
       double CalcIncidTerrain(Pt3dr aPTer);
       double ReadIncidTerrain(Pt2dr aPTer);

       
       double GetResolMoyenne_NonEuclid() const 
       {
// std::cout << "ORI RES " <<  mOri.resolution() << "\n";
          //OO  return ElAbs(mOri.resolution());
          if (mAnamSA|| mRC)
          {
              double aRes =  GetResolNonEuclidBySurfPixel();
              return  aRes;
/*
              Pt3dr aC = mOri->CentreOptique();
              Pt3dr aCa =  mAnam->E2UVL(aC);

              double aR1 =  ElAbs(aCa.z) / mOri->Focale();
              double aR2 =  GetResolNonEuclidBySurfPixel();
              return aR1;
*/
              // std::cout << "aaaaaaaaaNNaa " << aCa << "\n";
          }
          return ElAbs(mOri->ResolutionPDVVerticale());
       }
       void  RemplitOri(cFileOriMnt & aFOM) const
       {
           /* OO
              aFOM.NumZoneLambert().SetVal(mOri.ZoneLambert());
              if (! mIsCarto)
                 aFOM.OrigineTgtLoc().SetVal(mOri.OrigineTgtLoc());
           */
           if (mOLiLi)
           {
              aFOM.NumZoneLambert().SetVal(mOLiLi->ZoneLambert());
              if (! mIsCarto)
                 aFOM.OrigineTgtLoc().SetVal(mOLiLi->OrigineTgtLoc());
           }
       }

       CamStenope * mOri;
       bool         mIsCarto;
       Ori3D_Std *  mOLiLi;
       bool         mLoadIncT;
       std::string  mNameIncTerTif;
       // Champs speciaux Anam

       CamStenope *  GetOri()  const
       {
          return mOri;
       }

       Pt3dr GeomFP2GeomFinale(const Pt3dr & aP)  const
       {
             return mAnamSA ? mAnamSA->E2UVL(aP) : aP;
       }
       Pt3dr GeomFinale2GeomFP(const Pt3dr & aP)  const
       {
             return mAnamSA ? mAnamSA->UVL2E(aP) : aP;
       }


       ElSeg3D FaisceauPersp(const Pt2dr & aPt)  const
       {
             if (! mOri->ProfIsDef())
             {
                double aZ=0;
                if (mOri->AltisSolIsDef())
                {
                    aZ = mOri->GetAltiSol();
                }
                else
                {
                    ELISE_ASSERT(false,"Z  in cGeomImage_Terrain_Ori::Faisceau");
                }
                return ElSeg3D
                       (
                           mOri->OpticalVarCenterIm(aPt),
                           mOri->F2AndZtoR3(aPt,aZ)
                       );
             }
             //OO  double aProf = mOri.profondeur();
             double aProf = mOri->GetProfondeur();
             if ( aProf<=0)
             {
                 std::cout << "FOR " << mPDV.Name() << "\n";
                 ELISE_ASSERT
                 (
                       aProf > 0,"Prof in cGeomImage_Terrain_Ori::Faisceau"
                 );
             }
             return ElSeg3D
                    (
                        mOri->OpticalVarCenterIm(aPt),
                        mOri->ImEtProf2Terrain(aPt,aProf)
                    );

       }
};

bool cGeomImage_Terrain_Ori::MasqImNadirIsDone() { return ELISE_fp::exist_file(NameMasqImNadir()); }


void cGeomImage_Terrain_Ori::DoMasImNadir(TIm2D<REAL4,REAL8> & aImKN,cGeomDiscFPx & aGDF)
{
      Pt2dr aSzImR1 = Pt2dr(mOri->Sz());
      Pt2di aSzR = round_up(aSzImR1/mAnDeZoomM);
      Im2D_Bits<1> aMaskN(aSzR.x,aSzR.y,0);
      TIm2DBits<1> aTMaskN(aMaskN);

      Pt2di aPRed ;
      double aPx0[2] = {0,0};
      double aFact = 1 +  mAppli.MMImNadir()->IncertAngle().Val();

      for (aPRed.x=0 ; aPRed.x<aSzR.x; aPRed.x++)
      {
          for (aPRed.y=0 ; aPRed.y<aSzR.y; aPRed.y++)
          {
              Pt2di aPIm = aPRed * mAnDeZoomM;
              Pt2dr aPTer = ImageAndPx2Obj_Euclid(Pt2dr(aPIm),aPx0);
              double anIncIm =  ReadIncidTerrain(aPTer);
              if (anIncIm>=0)
              {
                  Pt2dr aPK = aGDF.R2ToRDisc(aPTer);
                  double aKInc = aImKN.getr(aPK,-1);
// std::cout << "IINNC  " << anIncIm  << " " << aKInc << aPK << " " << aImKN._the_im.sz() << "\n";
                  if (anIncIm  < (aKInc * aFact))
                  {
                     aTMaskN.oset(aPRed,1);
                  }
              }
	      // Pt3dr aPTer  = mOri->
          }
      }

      static Video_Win * aW = 0; 
      // if (aW==0)  aW =  Video_Win::PtrWStd(Pt2di(700,500));
      if (aW)
      {
          std::cout << mPDV.Name() << "\n";
          aW->set_title(mPDV.Name().c_str());
          ELISE_COPY(aW->all_pts(),P8COL::red,aW->odisc());
          ELISE_COPY(aMaskN.all_pts(),aMaskN.in(),aW->odisc());
          getchar();
      }

      Tiff_Im aTF
              (
                 NameMasqImNadir().c_str(),
                 aSzR,
                 GenIm::bits1_msbf,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
              );
       Fonc_Num aF = aMaskN.in(0);
       int aNbDil = mAppli.MMImNadir()->Dilat32().Val();
       if (aNbDil >0) aF = dilat_32(aF,aNbDil);

       int aNbErod = mAppli.MMImNadir()->Erod32().Val();
       if (aNbErod >0) aF = erod_32(aF,aNbErod);

       ELISE_COPY
       (
           aTF.all_pts(),
           aMaskN.in(0),
           aTF.out()
       );
       MakeMetaData_XML_GeoI(NameMasqImNadir(),mAnDeZoomM);
}

double cGeomImage_Terrain_Ori::CalcIncidTerrain(Pt3dr aP3A)
{
      double aEps = 1e-4 * mAnamSA->SignDZSensRayCam();
      Pt3dr aP3B (aP3A.x,aP3A.y,aP3A.z+aEps);
      Pt3dr aQ3A = mAnamSA->UVL2E(aP3A);
      Pt3dr aQ3B = mAnamSA->UVL2E(aP3B);
      Pt3dr aNormRentr= vunit(aQ3B-aQ3A);

      // double aSc = scal(aNormRentr,aSeg.TgNormee());
      Pt3dr aC = mOri->OpticalVarCenterTer(aQ3A);
      double aScalE = scal(aNormRentr,vunit(aQ3A-mOri->OpticalVarCenterTer(aQ3A)));



      Pt3dr aTgtE = vunit(aQ3A-aC);
      Pt3dr aP3C =  mAnamSA->E2UVL(aQ3A+aTgtE*aEps);
      Pt3dr aTgtUlv = vunit(aP3C-aP3A);
      double aScaleULV = aTgtUlv.z;

      if (0)
      {
          static int aNB0 =0;
          static int aNB1 =0;
          aNB0+= (aScalE<aScaleULV);
          aNB1+= (aScalE>=aScaleULV);
          std::cout << "GGGG " << aNB0 << " " << aNB1 << "\n";
      }

      return ElMin(aScalE,aScaleULV);
}
/*

               Pt2dr aP2Ter = aP0Ter + Pt2dr(aP)*aResTer;
               Pt3dr  aPA(aP2Ter.x,aP2Ter.y,0.0);
               Pt3dr  aPB(aP2Ter.x,aP2Ter.y,aEps);
               Pt3dr  aQA = mAnamSA->UVL2E(aPA);
               Pt3dr  aQB = mAnamSA->UVL2E(aPB);

               Pt3dr aNormRentr= vunit(aQB-aQA);

*/

void cGeomImage_Terrain_Ori::Init0MasqAnamSA()
{
  double DynVisu=0.0000002;  // Purement heuristique pour ce param de debugage, flemme de trouver une valeur rationnelle
  TIm2D<U_INT1,INT> aTImInit(Pt2di(1,1));
  TIm2D<U_INT1,INT> aTImOrtho(Pt2di(1,1));
  
  if (! mAnamSA) 
     return;

  std::string aNameXML =  NameMasqAnamSA("_Mtd.xml");
  std::string aNameTerTif =  NameMasqAnamSA("_MasqTer.tif");
  // std::cout << "XXMMLNAME " << aNameXML << "\n";

  mDoImMasqAnam = mAppli.MMImNadir()!=0;
  mUseTerMasqAnam = (mAppli.MMImNadir()==0) || (mAppli.MMImNadir()->MakeAlsoMaskTerrain().Val());

  if (mDoImMasqAnam)
     mDynIncidTerr = mAppli.MMImNadir()->DynIncid().Val();

  if (!ELISE_fp::exist_file(aNameXML))
  {

      static Video_Win * aW = 0; 
      // if (aW==0)  aW =  Video_Win::PtrWStd(Pt2di(900,900));

      //Pt2dr aSzImR1 = Pt2dr(mOri->Sz());
      Pt2dr aSzImR1 = Pt2dr(mOri->SzPixel());
      Pt2di aSzR = round_up(aSzImR1/mAnDeZoomM);
      Im2D_Bits<1> aMi(aSzR.x,aSzR.y,0);
      TIm2DBits<1> aTMi(aMi);

      // double aEps = 1e-4 * mAnamSA->SignDZSensRayCam();

      double aScalLim = ElMax(0.0,cos(ElMax(0.0,ElMin(PI/2,mAppli.AnamLimAngleVisib().Val()))));

if (0 && MPD_MM())
{
   std::cout << "sssSCALELIMMMM " << aScalLim << "\n";
}
      Pt2dr  aP0Ter (1e20,1e20);
      Pt2dr  aP1Ter (-1e20,-1e20);



      // 1- On calcul en geometrie image la masque des points ayant une incidence
      // inferieur a AnamLimAngleVisib
      Pt2di aP;
      int aNbTot = 0;
      int aNbOk = 0;
      for (aP.y=0 ; aP.y <aSzR.y ; aP.y++)
      {
          for (aP.x=0 ; aP.x <aSzR.x ; aP.x++)
          {


              Pt2dr aPF =Pt2dr(aP)*mAnDeZoomM;
              if (mOri->IsInZoneUtile(aPF,true)) // TRUE POU PIXEL
              {
                 ElSeg3D aSeg = mOri->F2toRayonR3(aPF);
              // donne le point, en coordonne UV d'intersection rayon incident surf L=0
                   cTplValGesInit<Pt3dr> aPGI3A = mAnamSA->InterDemiDroiteVisible(aSeg,0);
              // aTM.oset(aP,aNbSol>0);
if(0 &&  MPD_MM() )
{
/*
     std::cout << "IsInZoneUtile " << aPGI3A.IsInit()  << " NB" << mAnamSA->InterDroite(aSeg,0).size() << "\n";
     for (const auto & aP : mAnamSA->InterDroite(aSeg,0))
          std::cout << "GGGG " <<  aP << "\n";
*/
}
                  if (aPGI3A.IsInit() )
                  {
                     Pt3dr aP3A = aPGI3A.Val();
                     double aSc = CalcIncidTerrain(aP3A);
if (0 && MPD_MM())
{
   std::cout << "xxxxxx " << aSc << "\n";
}
                     if (aSc>aScalLim)
                     {
                         aNbOk++;
                         aTMi.oset(aP,1);
                         Pt2dr aP2A(aP3A.x,aP3A.y);
                         aP0Ter.SetInf(aP2A);
                         aP1Ter.SetSup(aP2A);
                     }
                     aNbTot++;

                  }
              }
          }
      }

      if (!aNbOk)
      {
          for (int aK=0 ; aK<5  ; aK++)
              std::cout << "      For image = " << mPDV.Name() << "\n";
          ELISE_ASSERT(false,"Not any point with good incidence"); 
      }

      if (aW)
      {
          // std::string aNameIm = mPDV.NameMasqOfResol(mAnDeZoomM);
          // std::cout << aNameTerTif <<  "  " << mDynIncidTerr  << "\n\n"; getchar();
          std::string aNameIm = mAppli.FullDirPyr()  +  mAppli.NameFilePyr(mPDV.Name(),mAnDeZoomM);
          std::cout << "NAAAMMEIN " << aNameIm << "\n";

          Tiff_Im aTI(aNameIm.c_str());
          aTImInit.Resize(aTI.sz());
          ELISE_COPY(aTI.all_pts(),Min(255,aTI.in()*DynVisu),aTImInit._the_im.out());
          Symb_FNum aFC (aTImInit._the_im.in());
          ELISE_COPY
          (
                aMi.all_pts(),
                Virgule
                (
                     aFC,
                     aFC,
                     aMi.in() *255
                ),
                aW->orgb()
           );
// NameMasqOfResol
          // aW->draw_rect(aP0Ter/mAnDeZoomM,aP1Ter/mAnDeZoomM,aW->pdisc()(P8COL::red));
          std::cout << " Boxx " << aP0Ter << aP1Ter << "\n";
          std::cout << "StatNormale " << (aNbOk) / double(aNbTot) << " " << aNbTot << "\n"; 
          getchar();
      }

      // Precaution anti inifini, pour eviter que sur des interection razante la boite soit 
      // trop grande, on refait un calcul en prenant le cone epsilon du milieue de l'image que l'on 
      // dilatera de 3/epsilon
      //
      //  En fait, avec le milieu on ne trouve pas toujours d'intersection, donc on modifie le code 
      // en testant plusieur points
      //  On espere que c'est devenu inutile avec le produit scalaire fait dans le 2 sens, mais on maintient quand meme avec une valeur de seuil
      // tres forte sur aMulRay

      if (0)  // JE COMPRENDS PLUS TROP A QUOI CA SERT ET CA CREE DES PBS .....
      {
           Pt2dr  aPExtr0Glob (1e20,1e20);
           Pt2dr  aPExtr1Glob (-1e20,-1e20);
           double aSurfMin = 1e60;
           bool BoxGlobInit = false;

           int aNbXY = 10;
           int aNbRay = 20;
           double aRay = euclid(mOri->Sz())/2.0;
           double aMulRay = 6.0;

           for (int aKX=1 ; aKX<aNbXY ; aKX++)
           {
               for (int aKY=1 ; aKY<aNbXY ; aKY++)
               {


                   Pt2dr aPds(double(aKX)/aNbXY,double(aKY)/aNbXY);
                   Pt2dr aCdg = aPds.mcbyc(Pt2dr(mOri->SzPixel()));
                   ElSeg3D aSegA = mOri->F2toRayonR3(aCdg);
                   cTplValGesInit<Pt3dr> aPGI3A = mAnamSA->InterDemiDroiteVisible(aSegA,0);


                   if (aPGI3A.IsInit())
                   {
                        bool AllOk=true;
                        Pt2dr  aPExtr0 (1e20,1e20);
                        Pt2dr  aPExtr1 (-1e20,-1e20);
                        Pt3dr aP3A = aPGI3A.Val();
                        Pt2dr aP2A(aP3A.x,aP3A.y);

                        double aMul = aRay * aMulRay;

                        for (int aK=0 ; aK< aNbRay ; aK++)
                        {
                            Pt2dr aDep =  Pt2dr::FromPolar(1.0,(2*PI*aK)/aNbRay);
                            Pt2dr aPB = aCdg + aDep;
                            ElSeg3D aSegB = mOri->F2toRayonR3(aPB);
                            cTplValGesInit<Pt3dr>  aPGI3B  =  mAnamSA->InterDemiDroiteVisible(aSegB,0);
                            if (aPGI3B.IsInit())
                            {
                                Pt3dr aP3B = aPGI3B.Val();
                                Pt2dr aP2B (aP3B.x,aP3B.y);

                                Pt2dr aPExtr = aP2A + (aP2B-aP2A) * aMul;
                                aPExtr0.SetInf(aPExtr);
                                aPExtr1.SetSup(aPExtr);
                            }
                            else
                            {
                               AllOk= false;
                            }
                        }

                        if (AllOk)
                        {
                            double aSurf = (aPExtr1.x-aPExtr0.x) * (aPExtr1.y-aPExtr0.y);
                            if (aSurf<aSurfMin)
                            {
                                aSurfMin = aSurf;
                                aPExtr0Glob = aPExtr0;
                                aPExtr1Glob = aPExtr1;
                                BoxGlobInit = true;
                            }
                        }
                   }
               }
           }
           ELISE_ASSERT(BoxGlobInit,"Cannot compute box in cGeomImage_Terrain_Ori::Init0MasqAnamSA");
           aP0Ter.SetSup(aPExtr0Glob);
           aP1Ter.SetInf(aPExtr1Glob);
      }


      if ( aW)
      {
         std::cout << "BOX AFETR " <<  aP0Ter << " " << aP1Ter << "\n"; 
         ELISE_COPY(aW->all_pts(),P8COL::blue,aW->odisc());
         ELISE_COPY(aMi.all_pts(),aMi.in(),aW->odisc());
         getchar();
      }

      mAnamSA->AdaptBox(aP0Ter,aP1Ter);

// std::cout << "AnamTer " << aP0Ter << " " << aP1Ter << "\n";

      double aResTer = GetResolMoyenne_NonEuclid()* mAnDeZoomM;
      // std::cout << "Rees "  << aResTer  << " " << mAnDeZoomM << "\n";
      Pt2di aSzTer = round_up((aP1Ter-aP0Ter)/aResTer);

      Im2D_INT1 aMt(aSzTer.x,aSzTer.y,0);
      TIm2D<INT1,INT> aTMt(aMt);

      if (aW)
      {
         aTImOrtho.Resize(aSzTer);
      }


      mTIncidTerr.Resize(aSzTer);
      for (aP.y=0 ; aP.y <aSzTer.y ; aP.y++)
      {
          for (aP.x=0 ; aP.x <aSzTer.x ; aP.x++)
          {
               Pt2dr aP2Ter = aP0Ter + Pt2dr(aP)*aResTer;
               Pt3dr  aPA(aP2Ter.x,aP2Ter.y,0.0);
               Pt3dr  aQA = mAnamSA->UVL2E(aPA);
               double aScal = CalcIncidTerrain(aPA);

               Pt2dr aPIm = mOri->R3toF2(aQA);
               if (   (aPIm.x >0) && (aPIm.y >0)
                   && (aPIm.x <aSzImR1.x) && (aPIm.y <aSzImR1.y)
                  )
               {
                  aTMt.oset(aP,ElMax(-127,ElMin(127,round_ni((aScal-aScalLim)*100))));
                  mTIncidTerr.oset(aP,round_ni(acos(ElMax(-1.0,ElMin(1.0,aScal)))*mDynIncidTerr));
               }
               else
               {
                  aTMt.oset(aP,-100);
                  mTIncidTerr.oset(aP,-10);
               }
               if (aW)
               {
                  aTImOrtho.oset(aP,aTImInit.getprojR(aPIm/mAnDeZoomM));
               }
          }
      }

      if (aW)
      {
         std::cout <<  "ORTHO  " <<  mPDV.Name() <<  aSzTer << "\n";

         ELISE_COPY(aW->all_pts(),P8COL::red,aW->odisc());
         
         //ELISE_COPY(aMt.all_pts(),aMt.in()>0,aW->odisc());

         ELISE_COPY(aMt.all_pts(),aTImOrtho._the_im.in(),aW->ogray());
         ELISE_COPY
         (
               aW->all_pts(),
               nflag_close_sym(flag_front8(aMt.in_proj()>0)),
               aW->out_graph(Line_St(aW->pdisc()(P8COL::red)))
         );
         getchar();


         ELISE_COPY(aMt.all_pts(),mod(aMt.in(),8),aW->odisc());
         getchar();
      }
      if (aW)
      {
          ELISE_COPY(aW->all_pts(),P8COL::red,aW->odisc());
          Im2D_INT2 aImN = mTIncidTerr._the_im;
          ELISE_COPY(aImN.all_pts(),Min(255,Max(0,aImN.in()/100)),aW->ogray());
          getchar();
      }

      if (mUseTerMasqAnam)
      {
         Tiff_Im::CreateFromIm(aMt,aNameTerTif);
      }

      if (mDoImMasqAnam)
      {
         Tiff_Im::CreateFromIm(mTIncidTerr._the_im,mNameIncTerTif);
      }


      if (aW)
      {
         std::cout << aNameTerTif << "\n";
         Tiff_Im aTF = Tiff_Im::StdConv(aNameTerTif);
         ELISE_COPY(aW->all_pts(),P8COL::yellow,aW->odisc());
         ELISE_COPY(aTF.all_pts(),mod(aTF.in(),8),aW->odisc());
         getchar();
      }


      cParamMasqAnam aPMA;
      aPMA.BoxTer() =  Box2dr(aP0Ter,aP1Ter) ;
      aPMA.Resol()  = aResTer ;
      MakeFileXML(aPMA,aNameXML);


     // std::cout << "ENNNNNNNNNNNNNNNd\n";
     // getchar();
  }
  mAnamSAIsInit = true;
  mAnamSAPMasq =  StdGetObjFromFile<cParamMasqAnam>
                (
                    aNameXML,
                    StdGetFileXMLSpec("ParamMICMAC.xml"),
                    "ParamMasqAnam",
                    "ParamMasqAnam"
                );
   
  if (mUseTerMasqAnam)
  {
      mMTA   = Im2D_INT1::FromFileStd(aNameTerTif);
      mTMTA  = TIm2D<INT1,INT>    (mMTA);
  }
/*
  if (mDoImMasqAnam)
  {
     mTIncidTerr = TIm2D<INT2,INT>(Im2D_INT2::FromFileStd(mNameIncTerTif));  
  }
*/
  // Pt3dr aP(0,0,0);
  // std::cout << Objet2ImageInit_Euclid(Pt2dr(0,0),0)
}

Pt2dr cGeomImage_Terrain_Ori::ToGeomMasqAnam(const Pt2dr & aPTer) const
{
   return (aPTer-mAnamSAPMasq.BoxTer()._p0)/mAnamSAPMasq.Resol();
}

double cGeomImage_Terrain_Ori::IncidTerrain(Pt2dr aPTer)
{
   return ReadIncidTerrain(aPTer);
}

double cGeomImage_Terrain_Ori::ReadIncidTerrain(Pt2dr aPTer)
{
   ELISE_ASSERT(mDoImMasqAnam,"cGeomImage_Terrain_Ori::IncidTerrain");
   if (! mLoadIncT)
   {
       mLoadIncT = true;
       mTIncidTerr = TIm2D<INT2,INT>(Im2D_INT2::FromFileStd(mNameIncTerTif));
   }

   // aPTer = (aPTer-mAnamSAPMasq.BoxTer()._p0)/mAnamSAPMasq.Resol();
   aPTer = ToGeomMasqAnam(aPTer);

   Pt2di aPIT = round_down(aPTer);
   if (
            (mTIncidTerr.get(aPIT,-10) < 0)
         || (mTIncidTerr.get(aPIT+Pt2di(1,0),-10) < 0)
         || (mTIncidTerr.get(aPIT+Pt2di(0,1),-10) < 0)
         || (mTIncidTerr.get(aPIT+Pt2di(1,1),-10) < 0)
       )
       return -10;

   double aRes = mTIncidTerr.getr(aPTer);
   if (aRes>=0) 
   {
      aRes /= mDynIncidTerr;
   }

   return aRes;
}




bool cGeomImage_Terrain_Ori::IsInMasqAnamSA(Pt2dr aPTer)
{
   if (! mUseTerMasqAnam)
      return true;
/*
   if (! mAnamSA)
      return true;
   if (! mUseTerMasqAnam)
      return false;
*/

   // return mTMTA.get(round_ni((aPTer-mAnamSAPMasq.BoxTer()._p0)/mAnamSAPMasq.Resol()) ,0) > 0;
   return mTMTA.get(round_ni(ToGeomMasqAnam(aPTer)) ,0) > 0;
}
 
void cGeomImage_Terrain_Ori::InitAnamSA(double aResol,const Box2dr & )
{
}



/*****************************************/
/*                                       */
/*            cGeomFaisZTerMaitre        */
/*            cGeomFaisZTerEsclave       */
/*                                       */
/*****************************************/

class cGeomFaisZTerMaitre : public cGeomImage_Id
{
    public :
       void RemplitOriXMLNuage(bool CFM,const cMTD_Nuage_Maille & mtd,const cGeomDiscFPx & aGT,cXML_ParamNuage3DMaille &aNuage ,eModeExportNuage mode) const
       {
           mGeoRef->RemplitOriXMLNuage(true,mtd,aGT,aNuage,mode);

           if (IsRPC())
           {
               aNuage.NameOri().SetVal(mPDV.NameGeom());
           }
       }


      ElCamera * GetCamera(const Pt2di & aSz,bool & ToDel,bool & aZUP) const
      {
            ToDel = false;
            ElCamera * aCam = GetOri();

            if (aCam) return aCam;
            return  cGeomImage::GetCamera(aSz,ToDel,aZUP);
      }
      Pt3dr TerrainRest2Euclid(const Pt2dr & aP,double * aPax) const
      {
         Pt2dr aRes2D =  mGeoRef->ImageAndPx2Obj_Euclid(aP,aPax);

	 return Pt3dr(aRes2D.x,aRes2D.y,aPax[0]);
      }

      cGeomFaisZTerMaitre
      (
          const cAppliMICMAC & anAppli,
          cPriseDeVue &      aPDV,
          eTagGeometrie        aTag,
          Pt2di                aSzIm,
          int                  aDimPx,
          cGeomImage *         aGeomRef
      )  :
         cGeomImage_Id  (anAppli,aPDV,aTag,aSzIm,aDimPx),
         mGeoRef        (aGeomRef)
      {

      }
      Pt3dr  Centre() const { return mGeoRef->Centre(); }
      bool  HasCentre() const { return mGeoRef->HasCentre(); }


      CamStenope *  GetOri()  const
      {
          return mGeoRef->GetOri();
      }
       std::string Name() const
       {
          return "cGeomFaisZTerMaitre::" +mGeoRef->Name();
       }
    
    
    
    // Par defaut erreur fatale si pas mode Image_Nuage
    protected :
      bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
      {
           if (mGeoRef->GetPxMoyenne_Euclid(aPxMoy))
           {
              if (mDimPx>1)
                 aPxMoy[1] = 0.0;
              return true;
           }
           return false;
           
      }
      bool GetRatioResolAltiPlani(double * aRatio) const
      {
          *aRatio = mGeoRef->GetResolMoyenne_Euclid() ;
          for (int aK=1 ; aK<mDimPx ; aK++)
              aRatio[aK] =  1.0;
          return true;
      }

      cGeomImage *  mGeoRef;
      bool IsId() const {return true;}
      const cGeomImage * GeoTerrainIntrinseque() const 
       {return mGeoRef->GeoTerrainIntrinseque();}

       bool IsRPC() const 
       {
            return mGeoRef->IsRPC();
       }
       bool RPCIsVisible(const Pt3dr & aP) const
       {
            if (! IsRPC()) return true;
           
            if ((aP.x<0) || (aP.y<0)  || (aP.x>mSzImInit.x) || (aP.y> mSzImInit.y)) return false;

            Pt2dr aZInt =  mGeoRef->RPCGetAltiSolMinMax() ;

            return  (aP.z >=aZInt.x) && (aP.z <= aZInt.y);
       }
       Pt2dr RPCGetAltiSolMinMax() const
       {
           return mGeoRef->RPCGetAltiSolMinMax();
       }

};



class cGeomFaisZTerEsclave : public cGeomFaisZTerMaitre
{
    public :
      cGeomFaisZTerEsclave
      (
          const cAppliMICMAC & anAppli,
          cPriseDeVue &      aPDV,
          Pt2di        aSzIm,
          int          aDim,
          cGeomImage*  aGeom,
          cGeomImage*  aGeomRef
      )  :
         cGeomFaisZTerMaitre (anAppli,aPDV,eTagGeomFaisZTerEsclave,aSzIm,aDim,aGeomRef),
         mGeom               (aGeom),
         mDirPx              (0,0)
      {
      }

      Pt3dr  Centre() const { return mGeom->Centre(); }
      bool  HasCentre() const { return mGeom->HasCentre(); }

       std::string Name() const
       {
          return "cGeomFaisZTerEsclave["+mGeom->Name() +"]["+mGeoRef->Name()+"]";
       }

      void InstPostInit()
      {
          double aZ0 = mAppli.GeomDFPxInit().V0Px()[0];
          Pt2dr aPM = mAppli.GeomDFPxInit().BoxEngl().milieu();
          Pt2dr aP1 = PtOfProf(aPM,aZ0);
          Pt2dr aP2 = PtOfProf(aPM,aZ0+1e-2);
          mDirPx = vunit(aP1-aP2) * Pt2dr(0,1);

          // AllTestPHom();
      }
      CamStenope *  GetOri()  const
      {
          return mGeom->GetOri();
      }


    private :
      void AllTestPHom();
      void TestPHom(Pt2dr aP1,Pt2dr aP2);

       inline Pt2dr PtOfProf(const Pt2dr & aPt,double aProf) const
       {
          static double aVPx[2] = {0,0};
          aVPx[0] = aProf;
          return mGeom->Objet2ImageInit_Euclid(mGeoRef->ImageAndPx2Obj_Euclid(aPt,aVPx),aVPx);
       }


       bool  DirEpipTransv(Pt2dr & aP) const
       {
           aP = mDirPx;
	   return true;
       }

       Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {


            Pt2dr aRes = PtOfProf(aP,aPx[0]);
            if (mDimPx > 1)
               aRes =  aRes +  mDirPx * aPx[1];
            return aRes;
       }

       Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
            if (mDimPx > 1)
               aP =  aP- mDirPx * aPx[1];
            static double aVPx[2] = {0,0};
            aVPx[0] = aPx[0];

            return mGeoRef->Objet2ImageInit_Euclid(mGeom->ImageAndPx2Obj_Euclid(aP,aVPx),aVPx);
       }
      bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
      {
           return false;
      }
      bool GetRatioResolAltiPlani(double * aRatio) const
      {
          return false;
      }

       cGeomImage *mGeom;
       Pt2dr     mDirPx;
       const cGeomImage * GeoTerrainIntrinseque() const
       {return mGeom->GeoTerrainIntrinseque();}


       bool IsRPC() const 
       {
            return mGeoRef->IsRPC() || mGeom->IsRPC();
       }
       Pt2dr RPCGetAltiSolMinMax() const
       {
           Pt2dr aRes(-1e60,1e60);
           if (mGeoRef->IsRPC())
           {
               aRes = mGeoRef->RPCGetAltiSolMinMax();
           }

           if (mGeom->IsRPC())
           {
               Pt2dr aR2 = mGeom->RPCGetAltiSolMinMax();
               aRes = Pt2dr(ElMax(aR2.x,aRes.x),ElMin(aR2.y,aRes.y));
           }

           return aRes;
       }
       bool RPCIsVisible(const Pt3dr & aP) const
       {
            if (! IsRPC()) return true;
            if (!  cGeomFaisZTerMaitre::RPCIsVisible(aP)) return false;
            static double aVPx[2] = {0,0};
            aVPx[0] = aP.z;
            Pt2dr aQ =  mGeoRef->ImageAndPx2Obj_Euclid(Pt2dr(aP.x,aP.y),aVPx);
            return mGeom->RPCIsVisible(Pt3dr(aQ.x,aQ.y,aP.z));
       }
};

void cGeomFaisZTerEsclave::TestPHom(Pt2dr aP1,Pt2dr aP2)
{
   Pt2dr aPx = P1P2ToPx(aP1,aP2);
   std::cout << aP1 << aP2 << " => " << aPx << "\n";

   CamStenope *   Ori1 = mGeoRef->GetOri();
   CamStenope *   Ori2 = mGeom->GetOri();

   double aD;
   // OO  Pt3dr aPTer = Ori1.to_terrain(aP1,Ori2,aP2,aD);
   Pt3dr aPTer = Ori1->PseudoInter(aP1,*Ori2,aP2,&aD);

   // OO  Pt3dr aPCarto = Ori1.terr_to_carte(aPTer);
   Pt3dr aPCarto = aPTer;  // PAS MIEUX A PROPOSER

   std::cout << "P Ter " << aPTer << " Dist = " << aD << "\n";
   
   std::cout << PtOfProf(aP1,aPTer.z) << "\n";

   // OO std::cout << Ori2.to_photo(Ori1.to_terrain(aP1,aPTer.z)) << "\n";
   std::cout << Ori2->R3toF2(Ori1->ImEtZ2Terrain(aP1,aPTer.z)) << "\n";

   Pt2dr aQ1 = mGeoRef->ImageAndPx2Obj_Euclid(aP1,&aPCarto.z);
   Pt2dr aR1 = mGeoRef->Objet2ImageInit_Euclid(aQ1,&aPCarto.z);

   std::cout << aP1  << "?=" << aR1 << " :: " << aQ1 << "\n";

   std::cout << "\n";
}

void cGeomFaisZTerEsclave::AllTestPHom()
{
    ElPackHomologue aPack = mAppli.PDV1()->ReadPackHom(mAppli.PDV2());
    for
    (
            ElPackHomologue::tIter anIter = aPack.begin();
            anIter != aPack.end();
            anIter++
    )
    {
            TestPHom(anIter->P1(),anIter->P2());
    }
    // getchar();
}

/**************************************************************/

Pt3dr  cAppliMICMAC::GetPtMoyOfOri(ElCamera * anOri) const
{
     double aProf = anOri->ProfIsDef()?anOri->GetProfondeur():0;
           
     bool ByProf = aProf > 0;

     if (EstimPxPrefZ2Prof().Val() && anOri->AltisSolIsDef())
        ByProf = false;
	   
     Pt3dr aPC;
     Pt2dr aSzIm = Pt2dr(anOri->Sz());
     if  (ByProf)
     {
         aPC =  anOri->ImEtProf2Terrain(aSzIm/2.0,aProf);
     }
     else
     {
         double aZ = anOri->GetAltiSol();
         ELISE_ASSERT(ALTISOL_IS_DEF(aZ),"ALTISOL_IS_DEF GetPxMoyenne_NonEuclid");
         aPC =  anOri->ImEtZ2Terrain(aSzIm/2.0,aZ);
     }

     return aPC;
}
/*
*/


/*****************************************/
/*                                       */
/*            cGeomImage_Id_Ori          */
/*            cGeomImage_Faisceau        */
/*                                       */
/*****************************************/

class cGeomImage_Id_Ori : public cGeomImage_Id
{
    public :
      Pt3dr TerrainRest2Euclid(const Pt2dr & aP,double * aPax) const
      {
          // OO  return mOriRef.ImDirEtProf2Terrain(aP,1/aPax[0],mNormPl);
          return mOriRef->ImDirEtProf2Terrain(aP,1/aPax[0],mNormPl);
      }

      std::string Name() const { return "cGeomImage_Id_Ori"; }

      cGeomImage_Id_Ori
      (
          const cAppliMICMAC & anAppli,
          cPriseDeVue &      aPDV,
          bool          byHerit,
          eTagGeometrie aTag,
          Pt2di aSzIm,
          int   aDimPx,
          CamStenope * anOriRef,
          bool         Spherik
      )  :
         cGeomImage_Id(anAppli,aPDV,aTag,aSzIm,aDimPx),
	 mSzIm   (aSzIm),
         mOriRef (anOriRef),
         mGITO   (anAppli,aPDV,aSzIm,anOriRef,false),
         mNormPl (
	              anAppli.X_DirPlanInterFaisceau().Val(),
	              anAppli.Y_DirPlanInterFaisceau().Val(),
	              anAppli.Z_DirPlanInterFaisceau().Val()
                  ),
         mIsSpherik  (Spherik)
      {
          if (anAppli.ExpA2Mm().DirVertLoc().IsInit())
             mNormPl = anAppli.ExpA2Mm().DirVertLoc().Val();

//std::cout << "NPL " << mNormPl << "\n";

          if ((!byHerit) &&  (anOriRef->HasDomaineSpecial()))
          {
              const_cast<cAppliMICMAC &>(mAppli).SetContourSpecIm1(anOriRef->ContourUtile());
          }
          if (euclid(mNormPl) < 1e-5)
	      mNormPl = mOriRef->DirK();
	  //OO     mNormPl = mOriRef.orDirK();

      }
/*
*/
      std::vector<Pt2dr>  EmpriseImage() const
      {
          return mOriRef->ContourUtile();
      }

    protected :
      bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
      {

           // Calcul d'un point "central"
	   //
	   // OO double aProf = mOriRef.profondeur();
	   // OO double aZ = mOriRef.altisol();

/*
	   double aProf = mOriRef->ProfIsDef() ? mOriRef->GetProfondeur()  : 0 ;
	   double aZ = mOriRef->GetAltiSol();
           
	   bool ByProf = aProf > 0;

	   if (mAppli.EstimPxPrefZ2Prof().Val() && ALTISOL_IS_DEF(aZ))
	      ByProf = false;
	   
           Pt3dr aPC;
	   if  (ByProf)
	   {
               // OO aPC =  mOriRef.ImEtProf_To_Terrain(mSzIm/2.0,aProf);
               aPC =  mOriRef->ImEtProf2Terrain(mSzIm/2.0,aProf);
	   }
	   else
	   {
	      ELISE_ASSERT(ALTISOL_IS_DEF(aZ),"ALTISOL_IS_DEF GetPxMoyenne_NonEuclid");
              // OO  aPC =  mOriRef.to_terrain(mSzIm/2.0,aZ);
              aPC =  mOriRef->ImEtZ2Terrain(mSzIm/2.0,aZ);
	   }
*/
           // OO  aPxMoy[0] = mOriRef.ProfInDir(aPC,mNormPl);

           Pt3dr aPC = mAppli.GetPtMoyOfOri(mOriRef);
/// std::cout << "AAAAAAAAAAlllllmmmmmm" << aPC <<  " " << mNormPl << "\n";
           aPxMoy[0] = mOriRef->ProfInDir(aPC,mNormPl);

  // std::cout << "BP " << ByProf << " "  << aPC << "\n"; getchar();


	   if (MakeInvIfNeeded)
	      aPxMoy[0] = 1/aPxMoy[0];

           if (mDimPx>1)
              aPxMoy[1] = 0.0;

/*
 std::cout << mPDV.Name() << "\n"; 
getchar();
*/
           return true;
      }

      Pt2di     mSzIm;
      CamStenope * mOriRef;
      cGeomImage_Terrain_Ori  mGITO;
      Pt3dr                   mNormPl;
      bool                    mIsSpherik;

      bool GetRatioResolAltiPlani(double * aRatio) const
      {
          return false;
      }
      bool IsId() const {return true;}

      CamStenope *  GetOri()  const
      {
          return mOriRef;
      }
      const cGeomImage * GeoTerrainIntrinseque() const
        {return  mGITO.GeoTerrainIntrinseque();}


      void RemplitOriXMLNuage
           (
                bool CFM,
                const cMTD_Nuage_Maille &,
                const cGeomDiscFPx & aGT,
                cXML_ParamNuage3DMaille & aNuage,
                eModeExportNuage
           ) const
      {
           aNuage.PM3D_ParamSpecifs().NoParamSpecif().SetNoInit();
           cModeFaisceauxImage aMFI;
           aMFI.ZIsInverse() = true;
           aMFI.IsSpherik().SetVal(mIsSpherik);
           aMFI.DirFaisceaux() = mNormPl;
           aNuage.PM3D_ParamSpecifs().ModeFaisceauxImage().SetVal(aMFI);

           aNuage.Orientation() = mOriRef->StdExportCalibGlob();
      }

};


       /*            cGeomImage_Faisceau        */

class cGeomImage_Faisceau : public cGeomImage_Id_Ori
{
    public :
       std::string Name() const {return "cGeomImage_Faisceau::" + mGITO.Name() ;}

      // interface pour cGeomBasculement3D, envoie de image vers terrain
      Pt3dr Bascule(const Pt3dr & aP3) const
      {
          //  return mOriRef.ImEtProf_To_Terrain(Pt2dr(aP3.x,aP3.y),1/aP3.z);
          // OO return mOriRef.ImDirEtProf2Terrain(Pt2dr(aP3.x,aP3.y),1/aP3.z,mNormPl);
          return mOriRef->ImDirEtProf2Terrain(Pt2dr(aP3.x,aP3.y),1/aP3.z,mNormPl);
      }
      Pt3dr BasculeInv(const Pt3dr & aPT) const 
      {
          // OO Pt2dr aPh = mOriRef.to_photo(aPT);
          Pt2dr aPh = mOriRef->R3toF2(aPT);
	  // double aZ = mOriRef.Prof(aPT);
	  // OO double aZ = mOriRef.ProfInDir(aPT,mNormPl);
	  double aZ = mOriRef->ProfInDir(aPT,mNormPl);
	  return Pt3dr(aPh.x,aPh.y,1/aZ);
      }

      cGeomImage_Faisceau
      (
          const cAppliMICMAC & anAppli,
          cPriseDeVue &      aPDV,
          Pt2di aSzIm,
          int       aDim,
          CamStenope * anOri,
          CamStenope * anOriRef,
          bool         isSpherik
      )  :
         cGeomImage_Id_Ori (anAppli,aPDV,true,eTagGeomFaisceau,aSzIm,aDim,anOriRef,isSpherik),
         mOri       (anOri),
         mDirPx     (0,0),
         mGITO      (anAppli,aPDV,aSzIm,anOri,false),
         mSpherik   (isSpherik)
      {
      }

       bool  DirEpipTransv(Pt2dr & aP) const
       {
           aP = mDirPx;
	   return true;
       }
      std::vector<Pt2dr>  EmpriseImage() const
      {
          return mOri->ContourUtile();
      }

       
       // Precaution anti singularite ou +infini devient - infini
      double Px2Prof(double aPax) const
      {
         return 1.0 / ElMax(aPax,1e-5);
      }

      bool  InstPostInitGen(bool UseDist)
      {
          if (mDimPx == 1) 
             return true;
          // OO  mOri.SetUseDist(UseDist);    A PRIORI DESUET ??? 
          // OO  mOriRef.SetUseDist(UseDist);

          double aZ0 = Px2Prof(mAppli.GeomDFPxInit().V0Px()[0]);
          Pt2dr aPM = mAppli.GeomDFPxInit().VraiBoxEngl().milieu();
          Pt2dr aP1 = PtOfProf(aPM,aZ0);
          Pt2dr aP2 = PtOfProf(aPM,aZ0+1e-2);

// std::cout << aP1 << aP2 << "\n";

          bool OK = euclid(aP1,aP2) > 1e-9;
          if (OK)
	  {
             mDirPx = vunit(aP1-aP2) * Pt2dr(0,1);
          }

          // OO mOri.SetUseDist(true);
          // OO mOriRef.SetUseDist(true);

	  return OK;
      }


      void InstPostInit()
      {

           if (!  InstPostInitGen(true))
	   {
	         if (! InstPostInitGen(false))
		 {
		     ELISE_ASSERT(false,"Inc in cGeomImage_Faisceau::InstPostInit()");
		 }
	   }
if (1)
{
   mAppli.TestPointsLiaisons(mOriRef,mOri,this);
}
      }

    private :
       inline Pt2dr PtOfProf(const Pt2dr & aPt,double aProf) const
       {


          // OO  return mOri.to_photo(mOriRef.ImDirEtProf2Terrain(aPt,aProf,mNormPl));
          return mOri->R3toF2
                 (
                        mSpherik                                             ?
                        mOriRef->ImEtProfSpherik2Terrain(aPt,aProf)          :
                        mOriRef->ImDirEtProf2Terrain(aPt,aProf,mNormPl)
                 );
       }


       Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
            Pt2dr aRes = PtOfProf(aP,Px2Prof(aPx[0]));
            if (mDimPx > 1)
               aRes =  aRes +  mDirPx * aPx[1];
            return aRes;
       }

       Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
            if (mDimPx > 1)
               aP =  aP- mDirPx * aPx[1];

	    Pt3dr aPt = mSpherik                                                                 ?
                        mOri->Im1EtProfSpherik2_To_Terrain(aP,*mOriRef,Px2Prof(aPx[0]))          :
                        mOri->Im1DirEtProf2_To_Terrain(aP,*mOriRef,Px2Prof(aPx[0]),mNormPl)      ;

            return mOriRef->R3toF2(aPt);
       }
      bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
      {
           return false;
      }


      bool GetRatioResolAltiPlani(double * aRatio) const
      {
          // OO Pt3dr aC1 = mOri.orsommet_de_pdv_terrain();
          // OO Pt3dr aC2 = mOriRef.orsommet_de_pdv_terrain();
          Pt3dr aC1 = mOri->PseudoOpticalCenter();
          Pt3dr aC2 = mOriRef->PseudoOpticalCenter();

	  double aCDNE = mGITO.CoeffDilNonE();
          // OO aRatio[0] = aCDNE* (mOri.resolution_angulaire() / euclid(aC1-aC2));
          aRatio[0] = aCDNE* (mOri->ResolutionAngulaire() / euclid(aC1-aC2));


         // std::cout << "GRRAP " << mOri->ResolutionAngulaire() << " " << euclid(aC1-aC2) << " " << mOri->Focale() << "\n";

// std::cout << "HHHH:GetRatioResolAltiPlani " << aRatio[0] << " CDNE " << aCDNE << "\n";
// std::cout << "HHHH:RA  " << mOri->ResolutionAngulaire() << " FFFFF " << 1/mOri->Focale() << "\n";


// std::cout << "HHHhhhhhhhhhhhhhhhhhhhhhhhhh " << aRatio[0] << " " << aC1 << aC2 << "\n";
// std::cout << "XXxxx " << aCDNE << " " << mOri->ResolutionAngulaire() << " " <<  euclid(aC1-aC2) << "\n";
          for (int aK=1 ; aK<mDimPx ; aK++)
              aRatio[aK] = 1.0;


          return true;
      }
      CamStenope *  GetOri()  const
      {
          return mOri;
      }
      const cGeomImage * GeoTerrainIntrinseque() const 
      {return mGITO.GeoTerrainIntrinseque();}



       CamStenope *  mOri;
       Pt2dr     mDirPx;
       cGeomImage_Terrain_Ori  mGITO;
       bool                    mSpherik;


};

/*****************************************/
/*                                       */
/*            cGeomImage_cBasic          */
/*                                       */
/*****************************************/

// It is (very naively !!) hoped that this time the cBasicGeomCap3D
// will replace all the altentaive existing general sensor; cGeomImage_cBasic
// is then the MicMac interface to cBasicGeomCap3D

class cGeomImage_cBasic : public cGeomImage
{
     public :
       std::string Name() const {return "cGeomImage_cBasic"  ;}
       virtual Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          Pt3dr aPTer =  mBGC3D->ImEtZ2Terrain(aP,aPx[0]);
          return Pt2dr(aPTer.x,aPTer.y);
       }
       virtual Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const
       {
          Pt3dr aPTer(aP.x,aP.y,aPx[0]);
          return mBGC3D->Ter2Capteur(aPTer);
       }

       virtual double GetResolMoyenne_NonEuclid() const
       {
	   return mBGC3D->GlobResol();
       }

       virtual bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
       {
	   aPxMoy[0] =  mBGC3D->PMoyOfCenter().z;
           return true;
       } 

       cGeomImage_cBasic(const cAppliMICMAC & anAppli,cPriseDeVue & aPDV,cBasicGeomCap3D * aBGC3D) :
              cGeomImage (anAppli,aPDV,eTagGeomBundleGen,aBGC3D->SzBasicCapt3D(),1),
              mBGC3D (aBGC3D)
       {
       }

       bool IsRPC() const 
       {
            return mBGC3D->IsRPC();
       }
       Pt2dr RPCGetAltiSolMinMax() const
       {
            return mBGC3D->GetAltiSolMinMax();
       }
       bool RPCIsVisible(const Pt3dr & aP) const
       {
            return mBGC3D->PIsVisibleInImage(aP);
       }

     private :
         cBasicGeomCap3D * mBGC3D;
};

cGeomImage * cGeomImage::GeomImage_Basic3D
             (
                                    const cAppliMICMAC & anAppli,
                                    cPriseDeVue &      aPDV
             )
{
    
    int aType = eTIGB_Unknown;


    return new cGeomImage_cBasic
               (
                    anAppli,
                    aPDV,
                    cBasicGeomCap3D::StdGetFromFile(aPDV.NameGeom(),aType)
               );
}


/*****************************************/
/*                                       */
/*            cGeomImage_Module          */
/*                                       */
/*****************************************/

class cGeomImage_Module : public cGeomImage
{	
  protected:
    
    virtual Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const
    {
      Pt2dr Pt(0.,0.);
      if (mModule)
	mModule->ImageAndPx2Obj(aP.x,aP.y,aPx,Pt.x,Pt.y);
      return Pt;
    }

    virtual Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const
    {
      Pt2dr Pt(0.,0.);
      if (mModule)
	mModule->Objet2ImageInit(aP.x,aP.y,aPx,Pt.x,Pt.y);
      return Pt;
    }

    virtual double GetResolMoyenne_NonEuclid() const
    {
      if (mModule)
	return mModule->GetResolMoyenne();
      return 1.;
    }

    virtual bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const
    {
      if (mModule)
	return mModule->GetPxMoyenne(aPxMoy);
      return false;
    } 

  public:

    cGeomImage_Module
    (
         const std::string & aNameModule,
         const cAppliMICMAC & anAppli,
         cPriseDeVue &      aPDV,
         Pt2di aSzIm,
         ModuleOrientation * aModule
     ) : 
      cGeomImage (anAppli,aPDV,eTagGeomModule,aSzIm,1),
      mModule    (aModule),
      mNameModule (aNameModule)
    {
    }
    
    ~cGeomImage_Module()
    {
       delete mModule;
    }
    std::string Name() const {return "Mod::" +mNameModule;}
    
    
    
    // Par defaut erreur fatale si pas mode Image_Nuage
    void RemplitOriXMLNuage(bool CFM,const cMTD_Nuage_Maille & mtd,const cGeomDiscFPx & aGT,cXML_ParamNuage3DMaille &aNuage ,eModeExportNuage mode) const
    {

        cGeomImage::RemplitOriXMLNuage(false,mtd,aGT,aNuage,mode);
        if (CFM)
        {
            cModuleOrientationFile oriFile;
            oriFile.NameFileOri()=mModule->GetFilename();
            aNuage.Orientation().ModuleOrientationFile().SetVal(oriFile); // RPCNuage
            aNuage.Orientation().TypeProj().SetVal(eProjGrid);            // RPCNuage
         }
    }

  private:
    ModuleOrientation * mModule;
    std::string         mNameModule;

};

cGeomImage * cGeomImage::GeomImage_Module
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di aSzIm,
 std::string const & nom_ori,
 std::string const & nom_module,
 std::string const & nom_geometrie
)
{
    cLibDynAllocator<ModuleOrientation> anAlloc(anAppli,nom_module,"create_orientation");
    ModuleOrientation * aModule = anAlloc.AllocObj(nom_ori,nom_geometrie);
    return new cGeomImage_Module(nom_ori,anAppli,aPDV,aSzIm,aModule);
}

cGeomImage * cGeomImage::GeomImage_Grille
( 
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di aSzIm,
 std::string const & nom_ori
)
{
    OrientationGrille * aGrille =  new OrientationGrille(nom_ori);
    aGrille->PrecisionRetour = 0.01 * aGrille->GetResolMoyenne();
    return new cGeomImage_Module(nom_ori,anAppli,aPDV,aSzIm,aGrille);
} 

cGeomImage * cGeomImage::GeomImage_RTO
(
    const cAppliMICMAC & anAppli,
    cPriseDeVue &      aPDV,
    Pt2di aSzIm,
    std::string const & nom_ori
)
{
        return new cGeomImage_Module(nom_ori,anAppli,aPDV,aSzIm,new OrientationRTO(nom_ori));
}
#ifdef __USE_ORIENTATIONIGN__
cGeomImage * cGeomImage::GeomImage_CON
(
    const cAppliMICMAC & anAppli,
    cPriseDeVue &      aPDV,
    Pt2di aSzIm,
    std::string const & nom_ori
)
{
        return new cGeomImage_Module(nom_ori,anAppli,aPDV,aSzIm,new OrientationCon(nom_ori));
}
#endif
  

/*****************************************/
/*                                       */
/*            ALLOCATEURS                */
/*                                       */
/*****************************************/


cGeomImage * cGeomImage::GeomId
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di aSzIm,
 int   aDimPx
)
{
    return new cGeomImage_Id(anAppli,aPDV,eTagGeomId,aSzIm,aDimPx);
}




cGeomImage * cGeomImage::Geom_DHD_Px
(
 const cAppliMICMAC &    anAppli,
 cPriseDeVue &      aPDV,
 Pt2di                   aSzIm,
 cDbleGrid *             aPGr1,
 cDbleGrid *             aPGr2,
 const ElPackHomologue & aPack,
 int                     aDimPx
)
{
    return new cGeomImage_DHD_Px(anAppli,aPDV,aSzIm,aPGr1,aPGr2,aPack,aDimPx);
}


cGeomImage * cGeomImage::Geom_Terrain_Ori
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di aSzIm,
 CamStenope * anOri
)
{
    return new cGeomImage_Terrain_Ori(anAppli,aPDV,aSzIm,anOri,false);
}

cGeomImage * cGeomImage::Geom_Carto_Ori
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di aSzIm,
 CamStenope * anOri
)
{
    return new cGeomImage_Terrain_Ori(anAppli,aPDV,aSzIm,anOri,true);
}


cGeomImage * cGeomImage::GeomFaisZTerMaitre
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di                aSzIm,
 int                  aDimPx,
 cGeomImage *         aGeomRef
)
{
    return new cGeomFaisZTerMaitre(anAppli,aPDV,eTagGeomFaisZTerMaitre,aSzIm,aDimPx,aGeomRef);
}


void Debug
    (
        cGeomImage *aGeom,
	Pt2dr                 aP1,
	Pt2dr                 aP2,
	Pt2dr                 aPx
    ) 
{
   std::cout << "\n";
   std::cout << "P1=" << aP1 << " ; P2=" << aP2 << "\n";
   double aVPx[2];
   aVPx[0] = 400 + aPx.x * 2.508882;
   aVPx[1] = aPx.y;
   std::cout << "P2Bis = " << aGeom->Objet2ImageInit_Euclid(aP1,aVPx) << "\n";
   std::cout  << "Px = " << aVPx[0] << " " << aVPx[1] << " " << aGeom->P1P2ToPx(aP1,aP2) << "\n";

   double aVCorPx[2]; GccUse(aVCorPx);
   aVCorPx[0] =aVPx[0];
   aVCorPx[1] =0;

    // getchar();
    std::cout << "\n";
}


void DebugPxTrsv(const cAppliMICMAC & anAp)
{
   ElPackHomologue  aPack = anAp.PDV1()->ReadPackHom(anAp.PDV2());

   CamStenope * aCam1   = anAp.PDV1()->Geom().GetOri();
   CamStenope * aCam2   = anAp.PDV2()->Geom().GetOri();


   std::cout << "CAMS " << aCam1 << " " << aCam2 << "\n";


   for  
   (
       ElPackHomologue::const_iterator itC = aPack.begin();
       itC != aPack.end();
       itC++
   )
   {
      Pt3dr  aPTer = aCam1->PseudoInter (itC->P1(),*aCam2,itC->P2());
      Pt2dr aU1 = aCam1->R3toF2(aPTer);
      Pt2dr aU2 = aCam2->R3toF2(aPTer);

       
      std::cout << itC->P1() << " " << itC->P2() << " \n";
      std::cout  <<   "  Diistt==" << euclid(aU1, itC->P1()) << " " << euclid(aU2, itC->P2())  << "\n";
      
      std::cout << anAp.PDV2()->Geom().P1P2ToPx(itC->P1(),itC->P2()) << "\n";
      std::cout <<" \n\n";
   }
   std::cout << "END DebugPxTrsv\n";
   getchar();


/*
   std::cout << "N=" << aGeom->Name() << "\n";
   Debug
   (
      aGeom,
      Pt2dr(6.315572217e+02,1.069263181e+04), // 631 10692
      Pt2dr(1.094678279e+03,1.091685468e+04),
      Pt2dr(-142.5,-19.705)
   );


   Debug
   (
      aGeom,
      Pt2dr(2.831805131e+02,1.162788928e+04), // 283 11627
      Pt2dr(7.837008666e+02,1.186328984e+04),
      Pt2dr(-166.598,-22.958)
   );

   Debug
   (
      aGeom,
      Pt2dr(2.638841489e+02,1.062902162e+04), // 264 10629
      Pt2dr(7.665206289e+02,1.086349530e+04),
      Pt2dr(-142.255,-24.7812)
   );
*/

    // getchar();
}

cGeomImage * cGeomImage::GeomFaisZTerEsclave
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di        aSzIm,
 int          aDim,
 cGeomImage*  aGeom,
 cGeomImage*  aGeomRef
)
{
    cGeomFaisZTerEsclave* aRes =  new cGeomFaisZTerEsclave(anAppli,aPDV,aSzIm,aDim,aGeom,aGeomRef);

    return aRes;
}



cGeomImage * cGeomImage::GeomImage_Id_Ori
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di aSzIm,
 int   aDim,
 CamStenope * anOri,
 bool         Spherik
)
{
    return new cGeomImage_Id_Ori(anAppli,aPDV,false,eTagGeom_IdOri,aSzIm,aDim,anOri,Spherik);
}



cGeomImage * cGeomImage::GeomImage_Faisceau
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di aSzIm,
 int   aDim,
 CamStenope * anOri,
 CamStenope * anOriRef,
 bool         isSpherik
)
{
    cGeomImage * aRes= new cGeomImage_Faisceau(anAppli,aPDV,aSzIm,aDim,anOri,anOriRef,isSpherik);
    // anAppli.TestPointsLiaisons(anOriRef,anOri,aRes);
    return  aRes;

}

/*
*/



cGeomImage * cGeomImage::Geom_NoGeomImage
(
 const cAppliMICMAC & anAppli,
 cPriseDeVue &      aPDV,
 Pt2di aSzIm
)
{
    return new  cGeomImage_NoGeom(anAppli,aPDV,aSzIm);
}

/*****************************************/
/*                                       */
/*      cGeometrieImageComp              */
/*                                       */
/*****************************************/

cGeometrieImageComp::cGeometrieImageComp
(
 const cNomsGeometrieImage & aGeom,
 const cAppliMICMAC &        anAppli
) :
    mGeom  (aGeom),
    mAppli (anAppli),
    mAutom ( aGeom.PatternSel().IsInit()? new cElRegex(aGeom.PatternSel().Val(),10,REG_EXTENDED) :0),
    mAutomI1I2 (0)
{
    if (mAutom && (!mAutom->IsOk()))
    {
        cout << "Pattern = [" << aGeom.PatternSel().Val() << "]\n";
        ELISE_ASSERT
            (
             false,
             "expression reguliere incorrecte"
            );
    }
    if (mGeom.PatternNameIm1Im2().IsInit())
    {
        mAutomI1I2 = new cElRegex(mGeom.PatternNameIm1Im2().Val(),10,REG_EXTENDED);
        if (!mAutomI1I2->IsOk())
        {
            cout << "Pattern = [" << mGeom.PatternNameIm1Im2().Val() << "]\n";
            ELISE_ASSERT
                (
                 false,
                 "expression reguliere incorrecte"
                );
        }
    }
}

bool cGeometrieImageComp::AcceptAndTransform
(
 const std::string & aNT,
 std::string &       aNameResult,
 int                 aNum
)
{
    std::string aNameTested = aNT;
    if (mGeom.AddNumToNameGeom().Val())
        aNameTested = aNameTested + "@"+ToString(aNum);
    if (mGeom.NGI_StdDir().IsInit())
    {
          const cNGI_StdDir  aNGI = mGeom.NGI_StdDir().Val();
          if (    aNGI.NGI_StdDir_Apply().IsInit()
               && (! aNGI.NGI_StdDir_Apply().Val()->Match(aNameTested))
             )
             return false;

         std::string aRes =  mAppli.ICNM()->StdNameCamGenOfNames(aNGI.StdDir(),aNameTested);
         if (aRes !="")
         {
             aNameResult= aRes;

             return true;
         }
    }
    else if (! mAutom)
    {
       ELISE_ASSERT(mGeom.FCND_Mode_GeomIm().IsInit()," No FCND_Mode_GeomIm ?? ");
       const cFCND_Mode_GeomIm  &aFCND = mGeom.FCND_Mode_GeomIm().Val();
       if (
               aFCND.FCND_GeomApply().IsInit() 
           && (! aFCND.FCND_GeomApply().Val()->Match(aNameTested))
           // &&  (! *(mAppli.ICNM()->SetIsIn(aFCND.FCND_GeomApply().Val(),aNameTested)))
	  )
	  return false;
//std::cout << "aNameTested " << aNameTested << "\n";
       aNameResult=  mAppli.ICNM()->Assoc1To1(aFCND.FCND_GeomCalc(),aNameTested,true);
//std::cout << "a<res " << aNameResult << "\n";
       return true;
    }
    else
    {

        if (! mAutom->Match(aNameTested))
            return false;

        if (mAutomI1I2)
        {
            std::string aNI1I2 = mAppli.Im1().Val()+"@"+mAppli.Im2().Val();
            bool Ok = mAutomI1I2->Match(aNI1I2);
            ELISE_ASSERT(Ok,"Cannot Match PatternNameIm1Im2"); 
            mAutomI1I2->Replace(mGeom.PatNameGeom().Val());
            aNameResult = mAutomI1I2->LastReplaced();
            return true;
        }

        mAutom->Replace(mGeom.PatNameGeom().Val());
        aNameResult =  mAutom->LastReplaced();
        return true;
    }
    return false;
}

const cTplValGesInit< cModuleImageLoader > &  
cGeometrieImageComp::ModuleImageLoader()const
{
    return mGeom.ModuleImageLoader();
}

const std::list<cModifieurGeometrie>  & 
  cGeometrieImageComp::ModG() const
{
   return mGeom.ModifieurGeometrie();
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3  
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
