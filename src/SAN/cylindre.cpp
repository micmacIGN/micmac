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


    cXmlModeleSurfaceComplexe * aPtrMC = OptionalGetObjFromFile_WithLC<cXmlModeleSurfaceComplexe>
                                         (
                                             0,0,
                                             aFile,
                                             StdGetFileXMLSpec("SuperposImage.xml"),
                                             aTag,
                                             aSpecTag
                                         );
    if (aPtrMC==0)
    {
           // static cInterfSurfaceAnalytique * FromCCC(const cChCoCart & );
       // cChCoCart(const Pt3dr &aOri,const Pt3dr&,const Pt3dr&,const Pt3dr&);
  // bitm.h:        static cChCoCart Xml2El(const cRepereCartesien &);

        cRepereCartesien * aXml_RC = OptStdGetFromPCP(aFile,RepereCartesien);
        if (aXml_RC)
        {
           cChCoCart aCCC = cChCoCart::Xml2El(*aXml_RC);
           return cInterfSurfaceAnalytique::FromCCC(aCCC);
        }
        std::cout << "For file =" << aFile << "\n";
        ELISE_ASSERT(false,"cannot get repair from file");
    }

    /*
    cXmlModeleSurfaceComplexe aMC = 
             StdGetObjFromFile<cXmlModeleSurfaceComplexe>
             (
                  aFile,
                  StdGetFileXMLSpec("SuperposImage.xml"),
                  aTag,
                  aSpecTag
             );
    */
   
     const cXmlOneSurfaceAnalytique & aSAN = SFromId(*aPtrMC,anId);

     if (aMemXML) 
        *aMemXML = aSAN;
     return cInterfSurfaceAnalytique::FromXml(aSAN);
}

cInterfSurfaceAnalytique * cInterfSurfaceAnalytique::FromFile(const std::string & aName)
{
   return  SFromFile(aName,"TheSurf");
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

   if (aDA.Tore().IsInit())
      return new cProjTore(cProjTore::FromXml(aDS,aDA.Tore().Val()));

   ELISE_ASSERT(false,"cInterfSurfaceAnalytique::FromXml");
   return 0;
}

bool  cInterfSurfaceAnalytique::OrthoLocIsXCste() const
{
    return false;
}


double cInterfSurfaceAnalytique::SeuilDistPbTopo() const
{
   return 0.0;
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


cInterfSurfaceAnalytique * cInterfSurfaceAnalytique::ChangeRepDictPts(const std::map<std::string,Pt3dr> &) const
{
    ELISE_ASSERT(false,"cInterfSurfaceAnalytique::ChangeRepDictPts");
    return 0;
}

cInterfSurfaceAnalytique * cInterfSurfaceAnalytique::DuplicateWithExter(bool IsExt)
{
    ELISE_ASSERT(false,"cInterfSurfaceAnalytique:: DuplicateWithExter(bool IsExt)");
    return 0;
}


cXmlModeleSurfaceComplexe cInterfSurfaceAnalytique::SimpleXml(const std::string & Id) const
{
   cXmlModeleSurfaceComplexe aRes;
   cXmlOneSurfaceAnalytique aSAN;
   aSAN.XmlDescriptionAnalytique() = Xml();
   aSAN.Id() = Id;
   aSAN.VueDeLExterieur() = mIsVueExt;
   aRes.XmlOneSurfaceAnalytique().push_back(aSAN);
   return aRes;
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
   mUnUseAnamXCSte (false),
   mIsVueExt (isVueExt)
{
}


void cInterfSurfaceAnalytique::SetUnusedAnamXCSte()
{
   mUnUseAnamXCSte = true;
}

int cInterfSurfaceAnalytique::SignDZSensRayCam()const
{
   // return mXXIsVueExt ? -1 : 1;
   return -1;  // Nouvelle convetion
   // return mIsVueExt ? -1 : 1;

}


bool cInterfSurfaceAnalytique::VueDeLext() const
{
   return mIsVueExt;
}


cTplValGesInit<Pt3dr> cInterfSurfaceAnalytique::PImageToSurf0
                      (const cCapture3D & aCap,const Pt2dr & aPIm) const
{
    return  InterDemiDroiteVisible(aCap.Capteur2RayTer(aPIm),0);
}


bool cInterfSurfaceAnalytique::IsAnamXCsteOfCart() const { return false; }




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
    cInterfSurfaceAnalytique(isVueExt),
    mSign  (isVueExt ? 1 : -1 )
{
    mP0 = aSeg.ProjOrtho(aPOnCyl);
    mW = aSeg.TgNormee();
    mU = aPOnCyl-mP0;
    mRay = euclid(mU);
    mU = mU / mRay;
    mV = mW ^ mU;

    for( int aK=0 ; aK<0 ; aK++)
    {
         Pt3dr aP(NRrandC(),NRrandC(),NRrandC());

         std::cout << "VERIF CYL " << euclid(aP-UVL2E(E2UVL(aP))) << " " << euclid(aP-E2UVL(UVL2E(aP))) << "\n";
    }

/*
    if (! isVueExt)
    {
std::cout << "Iiiiii cCylindreRevolution::cCylindreRevolution\n";
       mU = - mU;
       mV = - mV;
    }
*/
/*
*/
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

cInterfSurfaceAnalytique * cCylindreRevolution::ChangeRepDictPts(const std::map<std::string,Pt3dr> & aDic) const
{
    return CR_ChangeRepDictPts(aDic);
}

cInterfSurfaceAnalytique * cCylindreRevolution::DuplicateWithExter(bool IsExt)
{
    return CR_DuplicateWithExter(IsExt);
}




ElSeg3D  cCylindreRevolution::Axe() const
{
   return ElSeg3D(mP0,mP0+mW);
}


cCylindreRevolution * cCylindreRevolution::CR_DuplicateWithExter(bool IsExt)
{
   return new  cCylindreRevolution(IsExt,Axe(),POnCylInit());
}

cCylindreRevolution *      cCylindreRevolution::CR_ChangeRepDictPts(const std::map<std::string,Pt3dr> & aDic) const
{
    ElSeg3D aSeg = Axe();
    Pt3dr  aP0OnCyl =  POnCylInit() ;

    std::map<std::string,Pt3dr>::const_iterator itTop =    aDic.find("Top");
    std::map<std::string,Pt3dr>::const_iterator itBottom = aDic.find("Bottom");

    if ((itTop!=aDic.end()) && (itBottom!=aDic.end()))
    {
// UVL2E
        Pt3dr aTop  =     itTop->second;
        Pt3dr aBottom  =  itBottom->second;

        double aScal = aTop.y - aBottom.y;
        if (aScal<0)
        {
           aSeg  = ElSeg3D(mP0,mP0-mW);
        }
    }

    std::map<std::string,Pt3dr>::const_iterator itRight =    aDic.find("Right");
    std::map<std::string,Pt3dr>::const_iterator itLeft = aDic.find("Left");

    if ((itRight!=aDic.end())|| (itLeft!=aDic.end()))
    {
         double aPer = 2*PI*mRay;
         double aPerInf = aPer * 0.9;

         Pt3dr aPRight =   (itRight!=aDic.end())                            ?
                           itRight->second                                  :
                           (itLeft->second  + Pt3dr( aPerInf,0,0))          ;
               //-------------------------------------------------------------------
         Pt3dr aPLeft  =   (itLeft!=aDic.end())                             ?
                           itLeft->second                                   :
                           (itRight->second + Pt3dr(-aPerInf,0,0))          ;

         cCylindreRevolution aNewCyl(VueDeLext(),aSeg,aP0OnCyl);
         aPRight = aNewCyl.E2UVL(UVL2E(aPRight));
         aPLeft  = aNewCyl.E2UVL(UVL2E(aPLeft));
         while (aPRight.x>(aPLeft.x+aPer))  aPRight.x -= aPer;
         while (aPRight.x<=aPLeft.x) aPRight.x += aPer;

         // if (aSignInv) aPLeft.x +=aPer;

         aP0OnCyl = aNewCyl.UVL2E((aPRight+aPLeft)/2.0);
    }

    return new cCylindreRevolution (VueDeLext(),aSeg,aP0OnCyl);
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
               mSign* aRhoTeta.y*mRay,
                aZ,
                mSign*(aRhoTeta.x-mRay)
           );
}

double cCylindreRevolution::SeuilDistPbTopo() const
{
   return PI * mRay;
}

Pt3dr cCylindreRevolution::UVL2E(const Pt3dr & aP) const
{
    double aRho =  mSign*aP.z + mRay;
    double aTeta = (mSign*aP.x) / mRay;
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
     const std::string & aName,
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
   mDirPlkF  (mSet.Alloc().NewPt3("DirCyl-"+aName,mDirPlkCur)),

   mIndOri   (mSet.Alloc().CurInc()),
   mOriPlkF  (mSet.Alloc().NewPt3("OriCyl-"+aName,mOriPlkCur)),

   mIndRay   (mSet.Alloc().CurInc()),
   mRayF     (mSet.Alloc().NewF("Cyl-"+aName,"Ray",&mRayCur)),


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
                     mCyl.VueDeLext(),
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
                             const std::string & aName,
                             const  cCylindreRevolution & aCR,
                             bool GenCode 
                       )
{
   AssertUnClosed();

   cCylindreRevolFormel * aCRF = new cCylindreRevolFormel(aName,*this,aCR);

   aCRF->CloseEEF();
   AddObj2Kill(aCRF);

   aCRF->PostInit(GenCode);
   return *aCRF;
}

     /*****************************************/
     /*                                       */
     /*           PLY  PLY PLY                */
     /*                                       */
     /*****************************************/

void cInterfSurfaceAnalytique::V_MakePly(const cParamISAPly &, cPlyCloud &,const std::vector<ElCamera *> &,const Box2dr &,const double)
{
}

Box2dr BoxOfCam(cInterfSurfaceAnalytique & aSurf,ElCamera & aCam)
{
    Pt2dr aP0 (1e30,1e30);
    Pt2dr aP1 (-1e30,-1e30);
    int aNb= 10;
    Pt2dr aSz = Pt2dr (aCam.Sz());
    for (int anX=0 ; anX <= aNb ; anX++)
    {
        for (int anY=0 ; anY <= aNb ; anY++)
        {
             Pt2dr aP = aSz.mcbyc(Pt2dr(anX/double(aNb),anY/double(aNb)));
             cTplValGesInit<Pt3dr> aPTer = aSurf.PImageToSurf0(aCam,aP);
             if (aPTer.IsInit())
             {
                Pt2dr aQ(aPTer.Val().x,aPTer.Val().y);
                aP0.SetInf(aQ);
                aP1.SetSup(aQ);
             }
        }
    }

    
    return Box2dr(aP0,aP1);
}

void cInterfSurfaceAnalytique::MakePly(const cParamISAPly &  aParam, cPlyCloud & aPlyC,const std::vector<ElCamera *> & aVCam)
{

    Pt2dr aP0 (1e30,1e30);
    Pt2dr aP1 (-1e30,-1e30);
    std::vector<double> aVProf;

    for (int aKC = 0 ; aKC<int(aVCam.size()) ; aKC++)
    {
        ElCamera & aCam =  *(aVCam[aKC]);
        Box2dr aBox = BoxOfCam(*this,aCam);
        aP0.SetInf(aBox._p0);
        aP1.SetSup(aBox._p1);
        aVProf.push_back(aCam.GetProfondeur());
    }
    double aProf = MedianeSup(aVProf) / 10.0;

    AdaptBox(aP0,aP1);
    Box2dr aBox(aP0,aP1);

    Pt3dr anOri = UVL2E(Pt3dr(0,0,0));
    aPlyC.AddSphere(cPlyCloud::White,anOri, aParam.mSzSphere*aProf , 8);


    double aSzR = aParam.mSzRep * aProf;
    aPlyC.AddSeg(cPlyCloud::Red    ,anOri,  UVL2E(Pt3dr(aSzR,0,0)),  100);
    aPlyC.AddSeg(cPlyCloud::Green  ,anOri,  UVL2E(Pt3dr(0,aSzR,0)),  100);
    aPlyC.AddSeg(cPlyCloud::Blue   ,anOri,  UVL2E(Pt3dr(0,0,aSzR)),  100);


    int aNb = 150;
    for (int anX=0 ; anX<=aNb ; anX++)
    {
        for (int anY=0 ; anY<=aNb ; anY++)
        {
            Pt2dr aP = aBox.FromCoordLoc(Pt2dr(anX/double(aNb),anY/double(aNb)));
            aPlyC.AddPt(cPlyCloud::Black,UVL2E(Pt3dr(aP.x,aP.y,0)));
        }
    }
    V_MakePly(aParam,aPlyC,aVCam,aBox,aProf);
}


void cCylindreRevolution::V_MakePly(const cParamISAPly &, cPlyCloud & aPlyC,const std::vector<ElCamera *> &,const Box2dr & aBox,const double)
{
    for (int aK=0 ; aK<2 ; aK++)
    {
        double aY = (aK==0) ? aBox._p0.y :  aBox._p1.y;
        Pt3dr aC = UVL2E (Pt3dr(0,aY,-(mSign*mRay)));
        aPlyC.AddCercle(cPlyCloud::Magenta,aC,mW,mRay,500);
    }

    double aRab = 0.2;
    double  aY0 = barry(-aRab,aBox._p0.y,aBox._p1.y);
    double  aY1 = barry(1+aRab,aBox._p0.y,aBox._p1.y);
    aPlyC.AddSeg
    (
           cPlyCloud::Cyan,
           UVL2E(Pt3dr(0,aY0,-(mSign*mRay))),
           UVL2E(Pt3dr(0,aY1,-(mSign*mRay))),
           1000
    );
}



//  --------------- cParamISAPly ----------------------
//
//             Donne / a une profondeur de 10
//

cParamISAPly::cParamISAPly() :
     mSzRep       (0.3),
     mSzSphere    (0.05),
     mDensiteSurf (0.1)
{
}


//==================================================================
//                     cPlyCloud
//                     cPlyCloud
//                     cPlyCloud
//==================================================================


void cPlyCloud::AddPt(const tCol & aCol,const Pt3dr & aPt)
{
    if (aCol.x>=0)
    {
       mVCol.push_back(aCol);
       mVPt.push_back(aPt);
    }
}


void cPlyCloud::AddSphere(const tCol& aCol,const Pt3dr & aC,const double & aRay,const int & aNbPts)
{
    for (int anX=-aNbPts; anX<=aNbPts ; anX++)
    {
       for (int anY=-aNbPts; anY<=aNbPts ; anY++)
       {
          for (int aZ=-aNbPts; aZ<=aNbPts ; aZ++)
          {
               Pt3dr aP(anX,anY,aZ);
               aP = aP * (aRay/aNbPts);
               if (euclid(aP) <= aRay)
               {
                  AddPt(aCol,aC+aP);
               }
          }
       }
    }
}

Pt3dr Corner(const Pt3dr & aP1,const Pt3dr &aP2, int aNum)
{
   return Pt3dr
          (
             (aNum&1) ? aP1.x : aP2.x,
             (aNum&2) ? aP1.y : aP2.y,
             (aNum&4) ? aP1.z : aP2.z
          );
}

void cPlyCloud::AddCube
     (
          const tCol & aColP0,const tCol &aColP,const tCol & aColSeg,
          const Pt3dr & aP1,const Pt3dr &aP2,const double & aRay,
          const int & aNb
     )
{
    AddSphere(aColP0,Corner(aP1,aP2,0),aRay,5);
    for (int aFlag=0 ; aFlag <8 ; aFlag++)
    {
        AddSphere(aColP,Corner(aP1,aP2,aFlag),aRay,5);
    }

    for (int aFlag1=0 ; aFlag1 <8 ; aFlag1++)
    {
        for (int aFlag2=aFlag1+1 ; aFlag2 <8 ; aFlag2++)
        {
            int aDF = aFlag1 ^ aFlag2;
            if ((aDF==1) || (aDF==2) || (aDF==4))
            {
                tCol aCol = ((aFlag1&4) || (aFlag2&4)) ? aColSeg : aColP0;
                AddSeg(aCol,Corner(aP1,aP2,aFlag1),Corner(aP1,aP2,aFlag2),aNb);
            }
        }
    }
}


void  cPlyCloud::AddSeg(const tCol & aCol,const Pt3dr & aP1,const Pt3dr & aP2,const int & aNb)
{
     for (int aK=0 ; aK<= aNb ; aK++)
         AddPt(aCol,barry(double(aK)/aNb,aP1,aP2));
 
}

void  cPlyCloud::AddCercle(const tCol & aCol,const Pt3dr & aC,const Pt3dr &aNorm,const double & aRay,const int & aNb) 
{
    Pt3dr aU,aV,aW;
    aU = aNorm;
    MakeRONWith1Vect(aU,aV,aW);
    for (int aK=0 ; aK<= aNb ; aK++)
    {
        double aTeta = (2 * PI * aK) / aNb;
        AddPt(aCol,aC+ (aW*cos(aTeta) + aV*sin(aTeta)) * aRay);
    }
}

void cPlyCloud::PutDigit(char aDigit,Pt3dr aP0,Pt3dr aX,Pt3dr aY,tCol aCoul,double aLargCar,int aNbByCase)
{
    cElBitmFont & aFont =  cElBitmFont::BasicFont_10x8();

    Im2D_Bits<1>  anIm = aFont.ImChar(aDigit);
    TIm2DBits<1>  aTIm(anIm);


    Pt2di  aSz = anIm.sz();
    Pt2di aNb = aSz * aNbByCase;
    double aSc = aLargCar / aNb.x;

// std::cout << "PutDigit " << aP0 <<  " " <<  aLargCar << " \n";

    Pt2di aP;
    for (aP.x = 0 ; aP.x <aNb.x ; aP.x++)
    {
        for (aP.y = 0 ; aP.y <aNb.y ; aP.y++)
        {
             if (aTIm.get(aP/aNbByCase))
             {
                 Pt3dr aP3 = aP0 + aX*(aP.x*aSc) + aY*(aP.y*aSc);
// std::cout << aP <<  " " << aX*(aP.x*aSc) << "\n";
                 AddPt(aCoul,aP3);
             }
        }
    }
// std::cout << "HHHhh " << aSc << " " << aLargCar << " " << aNb << " NBCC " << aNbByCase  << " " << aX << " " << aY << "\n";
// getchar();

}

void cPlyCloud::PutString(std::string aDigit,Pt3dr aP0,Pt3dr aX,Pt3dr aY,tCol aCoul,double aLargCar,double aSpace,int aNbByCase, bool OnlyDigit)
{
    for (const char * aC = aDigit.c_str(); *aC ; aC++)
    {
        if ((! OnlyDigit) || isdigit(*aC))
        {
           PutDigit(*aC,aP0,aX,aY,aCoul,aLargCar,aNbByCase);
           aP0 = aP0 + aX * (aLargCar+aSpace);
        }
    }
}

void cPlyCloud::PutStringDigit(std::string aDigit,Pt3dr aP0,Pt3dr aX,Pt3dr aY,tCol aCoul,double aLargCar,double aSpace,int aNbByCase)
{
     PutString(aDigit,aP0,aX,aY,aCoul,aLargCar,aSpace,aNbByCase);
}


void cPlyCloud::PutFile(const std::string & aName)
{
    std::list<std::string> aVCom;
    std::vector<const cElNuage3DMaille *> aVNuage;
    cElNuage3DMaille::PlyPutFile
    (
          aName,
          aVCom,
          aVNuage,
          &mVPt,
          &mVCol,
          true
    );

}



const cPlyCloud::tCol cPlyCloud::White   (255,255,255);
const cPlyCloud::tCol cPlyCloud::Black   (  0,  0,  0);
const cPlyCloud::tCol cPlyCloud::Red     (255,  0,  0);
const cPlyCloud::tCol cPlyCloud::Cyan    (  0,255,255);
const cPlyCloud::tCol cPlyCloud::Green   (  0,255,  0);
const cPlyCloud::tCol cPlyCloud::Magenta (255,  0,255);
const cPlyCloud::tCol cPlyCloud::Blue    (  0,  0,255);
const cPlyCloud::tCol cPlyCloud::Yellow  (255,  0,  0);
cPlyCloud::tCol cPlyCloud::Gray(const double & aGr) 
{
   int Igr = ElMax(0,ElMin(255,round_ni(255*aGr)));
   return tCol(Igr,Igr,Igr);
}
cPlyCloud::tCol cPlyCloud::RandomColor() {
   return tCol(rand() % 256, rand() % 256, rand() % 256);
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
