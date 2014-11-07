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



     /*****************************************/
     /*                                       */
     /*                 ::                    */
     /*                                       */
     /*****************************************/


const cXmlOneSurfaceAnalytique & SFromId
      (
          const cXmlModeleSurfaceComplexe & aModCompl,
          const std::string & anId
      )
{
   for
   (
      std::list<cXmlOneSurfaceAnalytique>::const_iterator itS=aModCompl.XmlOneSurfaceAnalytique().begin();
      itS !=aModCompl.XmlOneSurfaceAnalytique().end();
      itS++
   )
   {
       if (itS->Id() == anId)
          return *itS;
   }
   ELISE_ASSERT(false,"SFromId");
   // cXmlOneSurfaceAnalytique *aMod;

   return  *((cXmlOneSurfaceAnalytique*)0);
}

cInterfSurfaceAnalytique * SFromFile
                           (
                                const std::string & aFile,
                                const std::string & anId,  // Nom
                                std::string  aTag ,
                                cXmlOneSurfaceAnalytique *  aMemXML
                           )
{

    std::string aSpecTag = "XmlModeleSurfaceComplexe";
    if (aTag=="")
       aTag = aSpecTag;


    cXmlModeleSurfaceComplexe aMC = 
             StdGetObjFromFile<cXmlModeleSurfaceComplexe>
             (
                  aFile,
                  StdGetFileXMLSpec("SuperposImage.xml"),
                  aTag,
                  aSpecTag
             );
   
     const cXmlOneSurfaceAnalytique & aSAN = SFromId(aMC,anId);

     if (aMemXML) 
        *aMemXML = aSAN;
     return cInterfSurfaceAnalytique::FromXml(aSAN);
}



     /*****************************************/
     /*                                       */
     /*   cInterfSurfAn_Formelle              */
     /*                                       */
     /*****************************************/
cInterSurfSegDroite::cInterSurfSegDroite(double aL,eTypeInterSurDemiDr aT) :
   mLamba (aL),
   mType  (aT)
{
}

cInterfSurfAn_Formelle::cInterfSurfAn_Formelle
(
   cSetEqFormelles & aSet,
   const std::string & aNameSurf
)  :
   cElemEqFormelle (aSet,false),
   mNameSurf       (aNameSurf),
   mSet            (aSet),
   mP2Proj         ("P2Proj"),
   mNameEqRat      ("c"+mNameSurf+"_EqRat_CodGen")
{
}

void cInterfSurfAn_Formelle::PostInitEqRat(bool Code2Gen)
{
    IncInterv().SetName("SurfRat");
    mLIntervEqRat.AddInterv(IncInterv());
    mFoncEqRat = cElCompiledFonc::AllocFromName(mNameEqRat);

    if (Code2Gen)
    {
       cElCompileFN::DoEverything
       (
           std::string("CodeGenere")+ELISE_CAR_DIR+"photogram"+ELISE_CAR_DIR,
           mNameEqRat,  
           EqRat(),  
           mLIntervEqRat
        );
        return;
    }
    ELISE_ASSERT
    (
       mFoncEqRat!=0,
      "Compiled code for cInterfSurfAn_Formelle"
    );
    mFoncEqRat->SetMappingCur(mLIntervEqRat,&mSet);
    mP2Proj.InitAdr(*mFoncEqRat);
    mSet.AddFonct(mFoncEqRat);
}


void cInterfSurfAn_Formelle::PostInit(bool Code2Gen)
{
    PostInitEqRat(Code2Gen);
}


double   cInterfSurfAn_Formelle::AddObservRatt
         (
              const Pt3dr & aP, 
              double aPds
         )
{
    mP2Proj.SetEtat(aP);
    return mSet.AddEqFonctToSys(mFoncEqRat,aPds,false);
}

     /*****************************************/
     /*                                       */
     /*   cInterfSurfaceAnalytique            */
     /*                                       */
     /*****************************************/


cInterfSurfaceAnalytique * cInterfSurfaceAnalytique::FromXml
                           (
                              const cXmlOneSurfaceAnalytique & aDS
                           ) 
{
   const cXmlDescriptionAnalytique & aDA  = aDS.XmlDescriptionAnalytique();
   if (aDA.Cyl().IsInit())
      return new cCylindreRevolution(cCylindreRevolution::FromXml(aDS,aDA.Cyl().Val()));

   if (aDA.OrthoCyl().IsInit())
      return new cProjOrthoCylindrique(cProjOrthoCylindrique::FromXml(aDS,aDA.OrthoCyl().Val()));

   ELISE_ASSERT(false,"cInterfSurfaceAnalytique::FromXml");
   return 0;
}

bool  cInterfSurfaceAnalytique::OrthoLocIsXCste() const
{
    return false;
}


Pt3dr cInterfSurfaceAnalytique::ToOrLoc(const Pt3dr & aP) const 
{
    ELISE_ASSERT(false,"cInterfSurfaceAnalytique::ToOrLoc");
    return Pt3dr(0,0,0);
}

Pt3dr cInterfSurfaceAnalytique::FromOrLoc(const Pt3dr & aP) const
{
    ELISE_ASSERT(false,"cInterfSurfaceAnalytique::FromOrLoc");
    return Pt3dr(0,0,0);
}




/*
std::vector<cInterSurfSegDroite>  cInterfSurfaceAnalytique::InterDroite(const ElSeg3D &,double aZ0) const 
{
   ELISE_ASSERT(false,"cInterfSurfaceAnalytique::SegAndL");
   std::vector<cInterSurfSegDroite> aRes;
   return aRes;
}
*/

Pt3dr cInterfSurfaceAnalytique::BestInterDemiDroiteVisible(const ElSeg3D & aSeg,double aZ0) const 
{
   return InterDemiDroiteVisible(true,aSeg,aZ0).Val();
}

cTplValGesInit<Pt3dr> cInterfSurfaceAnalytique::InterDemiDroiteVisible
                      (
                           bool  ForceMatch,
                           const ElSeg3D & aSeg,
                           double aZ0
                      ) const 
{
    std::vector<cInterSurfSegDroite> aVI = InterDroite(aSeg,aZ0);
    double aBestLamda = 1e30;
    int aBestK = -1;
    eTypeInterSurDemiDr aTISD = mIsVueExt  ? eSurfVI_Rent : eSurfVI_Sort ;


    for (int aK =0 ; aK<int(aVI.size()) ; aK++)
    {
       if (aVI[aK].mType== aTISD)
       {
           double aL = aVI[aK].mLamba;
           if ((aL>0) && (aL<aBestLamda))
           {
                aBestLamda = aL;
                aBestK = aK;
           }
       }
    }

    if (ForceMatch)
    {
       if (aBestK<0)
       {
          if (aVI.size()==0)
          {
               ELISE_ASSERT(false,"Cannot get val in InterDemiDroiteVisible");
          }
          aBestLamda = 0;
          for (int aK =0 ; aK<int(aVI.size()) ; aK++)
          {
              aBestLamda += aVI[aK].mLamba;
          }
          aBestLamda /= aVI.size();
       }
    }

    cTplValGesInit<Pt3dr> aRes;
    if (aBestLamda < 1e29)
    {
       Pt3dr aPE = aSeg.P0()+aSeg.TgNormee()*aBestLamda;
       // std::cout << "afgh:: " << aPE<<E2UVL(aPE)<<"\n";

       aRes.SetVal(E2UVL(aPE));
    }
    return aRes;
}


cTplValGesInit<Pt3dr> cInterfSurfaceAnalytique::InterDemiDroiteVisible(const ElSeg3D & aSeg,double aZ0) const 
{
   return InterDemiDroiteVisible(false,aSeg,aZ0);
}


cInterfSurfaceAnalytique::~cInterfSurfaceAnalytique()
{
}

cInterfSurfaceAnalytique::cInterfSurfaceAnalytique(bool isVueExt) :
   mIsVueExt (isVueExt)
{
}


int cInterfSurfaceAnalytique::SignDZSensRayCam()const
{
   return mIsVueExt ? -1 : 1;
}


bool cInterfSurfaceAnalytique::IsVueExt() const
{
   return mIsVueExt;
}


cTplValGesInit<Pt3dr> cInterfSurfaceAnalytique::PImageToSurf0
                      (const cCapture3D & aCap,const Pt2dr & aPIm) const
{
    return  InterDemiDroiteVisible(aCap.Capteur2RayTer(aPIm),0);
}
     /*****************************************/
     /*                                       */
     /*   cCylindreRevolution                 */
     /*                                       */
     /*****************************************/

cCylindreRevolution::cCylindreRevolution
(
    bool  isVueExt,
    const ElSeg3D & aSeg,
    const Pt3dr & aPOnCyl
)  :
    cInterfSurfaceAnalytique(isVueExt)
{
    mP0 = aSeg.ProjOrtho(aPOnCyl);
    mW = aSeg.TgNormee();
    mU = aPOnCyl-mP0;
    mRay = euclid(mU);
    mU = mU / mRay;
    mV = mW ^ mU;
}

bool cCylindreRevolution::HasOrthoLoc() const
{
   return false;
}


std::vector<cInterSurfSegDroite>  cCylindreRevolution::InterDroite(const ElSeg3D &aSeg,double aZ0) const 
{

    Pt3dr aV0 = aSeg.P0()-mP0;
    Pt3dr aT = aSeg.TgNormee();
    Pt3dr aU = mW;

    double aUV = scal(aU,aV0);
    double aUT = scal(aU,aT);

    // U esr norme  donc A est positif
    double aA = square_euclid(aT)-ElSquare(aUT);
    ELISE_ASSERT(aA!=0,"cCylindreRevolution::SegAndL");
    double aB = 2*(scal(aV0,aT)-aUT*aUV);
    double aC = square_euclid(aV0) - ElSquare(aUV)-ElSquare(mRay+aZ0);

    double aDelta = ElSquare(aB)-4*aA*aC;


    // std::cout << "DELTA " << aDelta << "\n";
// Si aDelta <0 on prend le point qui minise l'ecart, donc aDelta=0


    double aSqrtDelta = sqrt(ElMax(0.0,aDelta));
    double    aR1 = (-aB+aSqrtDelta)/(2*aA);
    double    aR2 = (-aB-aSqrtDelta)/(2*aA);
 

   std::vector<cInterSurfSegDroite> aRes;
   if (aDelta<0)
   {
      aRes.push_back(cInterSurfSegDroite(aR1,eSurfPseudoInter));
   }
   else if (aDelta==0)
   {
      aRes.push_back(cInterSurfSegDroite(aR1,eSurfInterTgt));
   }
   else
   {
      aRes.push_back(cInterSurfSegDroite(aR1,eSurfVI_Sort));
      aRes.push_back(cInterSurfSegDroite(aR2,eSurfVI_Rent));
   }

   return aRes;
}

/*
Pt3dr cCylindreRevolution::SegAndL(const ElSeg3D & aSeg,double aZ0,int & aNbVraiSol) const 
{

    Pt3dr aV0 = aSeg.P0()-mP0;
    Pt3dr aT = aSeg.TgNormee();
    Pt3dr aU = mW;

    double aUV = scal(aU,aV0);
    double aUT = scal(aU,aT);

    double aA = square_euclid(aT)-ElSquare(aUT);
    ELISE_ASSERT(aA!=0,"cCylindreRevolution::SegAndL");
    double aB = 2*(scal(aV0,aT)-aUT*aUV);
    double aC = square_euclid(aV0) - ElSquare(aUV)-ElSquare(mRay+aZ0);

    double aDelta = ElSquare(aB)-4*aA*aC;


    // std::cout << "DELTA " << aDelta << "\n";
// Si aDelta <0 on prend le point qui minise l'ecart, donc aDelta=0

    if (aDelta> 0) 
       aNbVraiSol = 2;
    else if (aDelta< 0)
       aNbVraiSol = 0;
    else
       aNbVraiSol = 1;

    aDelta = sqrt(ElMax(0.0,aDelta));
    double    aR1 = (-aB+aDelta)/(2*aA);
    double    aR2 = (-aB-aDelta)/(2*aA);
 
    Pt3dr aP1 = aSeg.P0() + aT * aR1;
    Pt3dr aP2 = aSeg.P0() + aT * aR2;

    Pt3dr aMil = (aSeg.P0()+aSeg.P1())/2.0;

    if (square_euclid(aP1-aMil)>square_euclid(aP2-aMil))
       ElSwap(aP1,aP2);


    // std::cout << E2UVL(aP1) << " " << aZ0 << " " << aSeg.DistDoite(aP1) << "\n";
    // getchar();

    return E2UVL(aP1);
}
*/




cCylindreRevolution cCylindreRevolution::WithRayFixed
                    (
                          bool  isVueExt,
                          const ElSeg3D & aSeg,
                          double    aRay,
                          const Pt3dr & aPOnCyl
                    )
{
    Pt3dr aProj = aSeg.ProjOrtho(aPOnCyl);
    return cCylindreRevolution
           (
                isVueExt,
                aSeg,
                aProj+ vunit(aPOnCyl-aProj)*aRay
           );
}


Pt3dr  cCylindreRevolution::POnCylInit() const
{
   return  UVL2E(Pt3dr(0,0,0));
}

Pt3dr cCylindreRevolution::E2UVL(const Pt3dr & aP) const
{
    Pt3dr aPP0 = aP-mP0;

    double aX = scal(aPP0,mU);
    double aY = scal(aPP0,mV);
    double aZ = scal(aPP0,mW);
    Pt2dr aRhoTeta = Pt2dr::polar(Pt2dr(aX,aY),0);  // Rho teta

    // On renvoie Teta-Z-Rho pour : 
    //   1 etre direct
    //   2 que Rho soit en dernier (et  donc 
    return Pt3dr
           (
                aRhoTeta.y*mRay,
                aZ,
                aRhoTeta.x-mRay
           );
}


Pt3dr cCylindreRevolution::UVL2E(const Pt3dr & aP) const
{
    double aRho =  aP.z + mRay;
    double aTeta = aP.x / mRay;
    double aZ = aP.y;

    Pt2dr aXY = Pt2dr::FromPolar(aRho,aTeta);

    return   mP0
           + mU * aXY.x
           + mV * aXY.y
           + mW * aZ;
}


const Pt3dr & cCylindreRevolution::P0() const
{
   return mP0;
}

const Pt3dr & cCylindreRevolution::W() const
{
   return mW;
}

const Pt3dr & cCylindreRevolution::U() const
{
   return mU;
}

double  cCylindreRevolution::Ray() const
{
   return mRay;
}

void cCylindreRevolution::AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const 
{
   double aSat = (aP1.x - aP0.x) / (mRay *2 * PI);
   if (aSat > 0.95)
      aP1.x  += mRay *2 * PI * 0.05;
/*

   std::cout << "Teta " 
             << aSat << " "
             << aP0.x / mRay <<  " "
             << aP1.x / mRay <<  " "
             << (aP1.x - aP0.x) / mRay << "\n";;

*/
}

Pt3dr  cCylindreRevolution::PluckerDir()
{
    return mW;
}

Pt3dr  cCylindreRevolution::PluckerOrigine()
{
   ElSeg3D aSeg(mP0,mP0+mW);
   return aSeg.ProjOrtho(Pt3dr(0,0,0));
}

cXmlCylindreRevolution cCylindreRevolution::XmlCyl() const
{
    cXmlCylindreRevolution aRes;
    aRes.P0() = mP0;
    aRes.P1() = mP0 + mW;
    aRes.POnCyl() = POnCylInit();

    return aRes;
}


cXmlDescriptionAnalytique cCylindreRevolution::Xml() const
{
   cXmlDescriptionAnalytique aRes;
   aRes.Cyl().SetVal(XmlCyl());

   return aRes;
}

cCylindreRevolution cCylindreRevolution::FromXml
                    (
                       const cXmlOneSurfaceAnalytique& aSAN,
                       const cXmlCylindreRevolution& aCyl
                    )
{
    return cCylindreRevolution
           (
               aSAN.VueDeLExterieur(),
               ElSeg3D(aCyl.P0(),aCyl.P1()),
               aCyl.POnCyl()
           );
}


     /*****************************************/
     /*                                       */
     /*   cCylindreRevolFormel                */
     /*                                       */
     /*****************************************/

cCylindreRevolFormel::cCylindreRevolFormel
(
     cSetEqFormelles & aSet,
     const cCylindreRevolution &aCyl
)  :
   cInterfSurfAn_Formelle (aSet,"Cylindre"),
   mCyl     (aCyl),
   mCurCyl  (mCyl),

   mDirPlk0 (mCyl.PluckerDir()),
   mOriPlk0 (mCyl.PluckerOrigine()),
   mRay0    (mCyl.Ray()),

   mDirPlkCur (mDirPlk0),
   mOriPlkCur (mOriPlk0),
   mRayCur    (mRay0),
   
   mIndDir   (mSet.Alloc().CurInc()),
   mDirPlkF  (mSet.Alloc().NewPt3(mDirPlkCur)),

   mIndOri   (mSet.Alloc().CurInc()),
   mOriPlkF  (mSet.Alloc().NewPt3(mOriPlkCur)),

   mIndRay   (mSet.Alloc().CurInc()),
   mRayF     (mSet.Alloc().NewF(&mRayCur)),


   mFcteurNormDir (cElCompiledFonc::FoncFixeNormEucl(&mSet,mIndDir,3,1.0)),
   mFcteurOrthogDirOri (cElCompiledFonc::FoncFixedScal(&mSet,mIndDir,mIndOri,3,0.0)),
   mTolFctrNorm   (cContrainteEQF::theContrStricte),
   mTolFctrOrtho  (cContrainteEQF::theContrStricte)
   
{
   AddFoncteurEEF(mFcteurNormDir);
   AddFoncteurEEF(mFcteurOrthogDirOri);
}



void cCylindreRevolFormel::Update_0F2D()
{
    mCurCyl = cCylindreRevolution::WithRayFixed
              (
                     mCyl.IsVueExt(),
                     ElSeg3D(mOriPlkCur,mOriPlkCur+mDirPlkCur),
                     mRayCur,
                     mCyl.POnCylInit()
              );
}

const cInterfSurfaceAnalytique & cCylindreRevolFormel::CurSurf() const
{
    return CurCyl();
}
const cCylindreRevolution & cCylindreRevolFormel::CurCyl() const
{
    return mCurCyl;
}



Fonc_Num  cCylindreRevolFormel::EqRat()
{
   Pt3d<Fonc_Num> aVec =  mP2Proj.PtF()- mOriPlkF;
   Fonc_Num aV2 = square_euclid(aVec);
   Fonc_Num aScal = scal(aVec,mDirPlkF);

   Fonc_Num aD2 = aV2 - Square(aScal);

   return sqrt(aD2)-mRayF;
}


cMultiContEQF cCylindreRevolFormel::StdContraintes()
{
  cMultiContEQF  aRes;

  mFcteurNormDir->SetCoordCur(mSet.Alloc().ValsVar());
  aRes.AddAcontrainte(mFcteurNormDir, mTolFctrNorm);

  mFcteurOrthogDirOri->SetCoordCur(mSet.Alloc().ValsVar());
  aRes.AddAcontrainte(mFcteurOrthogDirOri,mTolFctrOrtho);

  return aRes;
}

     /*****************************************/
     /*                                       */
     /*      cSetEqFormelles                  */
     /*                                       */
     /*****************************************/

cCylindreRevolFormel & cSetEqFormelles::AllocCylindre
                       (
                             const  cCylindreRevolution & aCR,
                             bool GenCode 
                       )
{
   AssertUnClosed();

   cCylindreRevolFormel * aCRF = new cCylindreRevolFormel(*this,aCR);

   aCRF->CloseEEF();
   AddObj2Kill(aCRF);

   aCRF->PostInit(GenCode);
   return *aCRF;
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
