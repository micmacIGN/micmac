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


/*************************************************/
/*                                               */
/*    EpipolaireCoordinate                       */
/*                                               */
/*************************************************/
void EpipolaireCoordinate::SaveOrientEpip
              (
                  const std::string &                anOri,
                  cInterfChantierNameManipulateur *  anICNM,
                  const std::string &                aNameIm,
                  const std::string &                aNameOther
               ) const
{
   //   anOri + aNameIm + aNameOther + Ori(aNameIm) + Ori(aNameOther)  +  OrientationItself
   std::string  aNameFile = anICNM->NameOrientEpipGen(anOri,aNameIm,aNameOther);

   /*std::string  aNameFile = anICNM->Assoc1To2
                        (
                            "NKS-Assoc-CplIm2OriGenEpi@"+anOri+"@txt",
                            aNameIm,aNameOther,true
                        );
   */
   ELISE_fp  aFile(aNameFile.c_str(),ELISE_fp::WRITE,false,ELISE_fp::eTxtTjs);


   aFile.write(anOri);
   aFile.write(aNameIm);
   aFile.write(anICNM->StdNameCamGenOfNames(anOri,aNameIm));
   aFile.write(aNameOther);
   aFile.write(anICNM->StdNameCamGenOfNames(anOri,aNameOther));

   CastToPol()->write(aFile);
   aFile.close();
}


void EpipolaireCoordinate::XFitHom(const ElPackHomologue &,bool aL2,EpipolaireCoordinate *)
{
   ELISE_ASSERT(false,"No def val for XFitHom");
}

bool EpipolaireCoordinate::HasXFitHom() const
{
    return false;
}

std::vector<double>  EpipolaireCoordinate::ParamFitHom() const 
{
   ELISE_ASSERT(false,"No ParamFitHom");
   return std::vector<double>();
}


void EpipolaireCoordinate::HeriteChScale(EpipolaireCoordinate & anEC,REAL aChSacle)
{
   mP0 = anEC.mP0 * aChSacle;
   mDirX = anEC.mDirX;
   mTrFin = anEC.mTrFin * aChSacle;
   if (anEC.mGridCor)
   {
       delete mGridCor;
       mGridCor = anEC.mGridCor->NewChScale(aChSacle,true);
   }
}

bool   EpipolaireCoordinate::IsEpipId() const
{
	return false;
}

EpipolaireCoordinate::EpipolaireCoordinate
    (
           Pt2dr aP0,
           Pt2dr aDirX,
           Pt2dr aTrFin
    )  :
    mGridCor (0),
    mP0(aP0),
    mDirX(aDirX),
    mTrFin (aTrFin)
{
}

EpipolaireCoordinate::~EpipolaireCoordinate()
{
   // Pb avec les copie sur mGridCor
   // delete mGridCor;
}


Pt2dr EpipolaireCoordinate::DirEpip(Pt2dr aP,REAL anEpsilon)
{
   return vunit(TransOnLineEpip(aP,anEpsilon)-aP);
}

Pt2dr EpipolaireCoordinate::Direct(Pt2dr aP) const 
{
     aP = (aP-mP0)/mDirX;
     aP =  ToCoordEpipol(aP) + mTrFin;

     if (mGridCor)
	aP.y +=  mGridCor->Value(aP);
     return aP;
}


bool EpipolaireCoordinate::OwnInverse(Pt2dr & aP) const
{
    if (mGridCor)
    {
// cout << mGridCor->Value(aP) << "\n";
	aP.y -=  mGridCor->Value(aP);
    }
   aP -= mTrFin;
   aP =  ToCoordInit(aP) * mDirX + mP0 ;
   return true;
}


Pt2dr EpipolaireCoordinate::TransOnLineEpip
      (
	Pt2dr aP,
	REAL aParal
      )
{
   return Inverse(Direct(aP)+Pt2dr(aParal,0.0));
}

Pt2dr EpipolaireCoordinate::P0() const
{
   return mP0;
}

Pt2dr EpipolaireCoordinate::DirX() const
{
   return mDirX;
}

Pt2dr EpipolaireCoordinate::TrFin() const
{
   return mTrFin;
}

const PolynomialEpipolaireCoordinate * EpipolaireCoordinate::CastToPol() const
{
   ELISE_ASSERT(false,"Bas Up-Cast to PolynomialEpipolaireCoordinate");
   return 0;
}

void   EpipolaireCoordinate::AddTrFinale(Pt2dr aTr)
{
   if (mGridCor)
       mGridCor->TranlateIn(aTr);
   mTrFin += aTr;
}
void   EpipolaireCoordinate::SetTrFinale(Pt2dr aTr)
{
   AddTrFinale(aTr-mTrFin);
}



Box2dr EpipolaireCoordinate::ImageOfBox(Box2dr aBoxIm)
{
     Pt2dr Corners[4];
     REAL xEpi[4];
     REAL yEpi[4];

     aBoxIm.Corners(Corners);
     for (INT aK=0 ; aK<4 ; aK++)
     {
          Pt2dr aPEpi =  Direct(Corners[aK]);
          xEpi[aK] = aPEpi.x;
          yEpi[aK] = aPEpi.y;
     }

     return Box2dr(xEpi,yEpi,4);
}


void   EpipolaireCoordinate::SetGridCorrec
       (
           Fonc_Num  DeltaY, 
           Fonc_Num  Pond,  
           REAL      aStepGr,
	   Box2dr    aBoxIm,
	   REAL      aRatioMin
       )
{
  Box2dr aBoxEpi = ImageOfBox(aBoxIm);
  // delete mGridCor;
  RImGrid * aGridCor = new RImGrid
	         (
		      false,
		     aBoxEpi._p0-Pt2dr(2,2),
		     aBoxEpi._p1+Pt2dr(2,2),
		     Pt2dr(aStepGr,aStepGr)
		 );

  for (INT x=0 ; x<aGridCor->SzGrid().x; x++)
      for (INT y=0 ; y<aGridCor->SzGrid().y; y++)
      {
	   ElList<Pt2di> aLC;
	   for (INT k=0 ; k<4 ; k++)
           {
                Pt2dr aP = Inverse(aGridCor->ToReal(Pt2dr(x,y)+TAB_CornerPix[k]));
		aLC = aLC+Pt2di(aP);
           }
	   Symb_FNum aSP(Rconv(Pond!=0));

           REAL Sy,Sp,SNb;
           ELISE_COPY
           (
	        polygone(aLC),
		Virgule(DeltaY*aSP,aSP,1.0),
		Virgule(sigma(Sy),sigma(Sp),sigma(SNb))
           );
	   if (Sp > SNb * aRatioMin)
	   {
               // cout << " In " << Sy/Sp << " \n";
               aGridCor->SetValueGrid(Pt2di(x,y),Sy/Sp);
	   }
           else
           {
               // cout << "------------ Out \n";
           }
      }

  aGridCor->ExtDef();

   mGridCor = aGridCor;
}

/*************************************************/
/*                                               */
/*            cMappingEpipCoord                  */
/*                                               */
/*************************************************/

cMappingEpipCoord::cMappingEpipCoord
(
    ElDistortion22_Gen * aDist,
    bool toDel
)  :
   EpipolaireCoordinate(Pt2dr(0,0),Pt2dr(1,0),Pt2dr(0,0)),
   mDist (aDist),
   mToDel (toDel)
{
}

cMappingEpipCoord::~cMappingEpipCoord()
{
   if (mToDel)
      delete mDist;
}


EpipolaireCoordinate * cMappingEpipCoord::MapingChScale(REAL aChSacle) const
{
   EpipolaireCoordinate * aRes = new cMappingEpipCoord(mDist->D22G_ChScale(aChSacle),true);
   aRes->HeriteChScale(const_cast<cMappingEpipCoord &>(*this),aChSacle);
   return aRes;
}


Pt2dr cMappingEpipCoord::ToCoordEpipol(Pt2dr aPInit) const 
{
   return mDist->Direct(aPInit);
}

Pt2dr cMappingEpipCoord::ToCoordInit(Pt2dr aPInit) const 
{
   return mDist->Inverse(aPInit);
}
bool  cMappingEpipCoord::IsEpipId() const
{
   return mDist->IsId();
}


/*************************************************/
/*                                               */
/*    PolynomialEpipolaireCoordinate             */
/*                                               */
/*************************************************/

static double  mExagEpip=1.0;
static bool mApproxInvExagEpip=false;

void SetExagEpip(double aVal,bool AcceptApprox)
{
    mExagEpip = aVal;
    mApproxInvExagEpip=AcceptApprox;
}

void  EpipolaireCoordinate::Diff(ElMatrix<REAL> &,Pt2dr) const 
{
    ELISE_ASSERT
    (
       false,
       "No PolynomialEpipolaireCoordinate::Diff"
    );
}


Pt2dr PolynomialEpipolaireCoordinate::ToCoordEpipol(Pt2dr aPInit) const
{
   ELISE_ASSERT((mExagEpip==1.0)||  (!mCorCalc) ,"Dont handle Exag + mCorCalc");

   Pt2dr aRes (aPInit.x,aPInit.y + (mPolToYEpip(aPInit)-aPInit.y) * mExagEpip);
   double aX = mCorCalc ?  (mNum0 + mNumx*aRes.x+mNumy*aRes.y)/(1.0+mDenx*aRes.x+mDeny*aRes.y) : aRes.x;

   return Pt2dr(aX,aRes.y);
}

Pt2dr PolynomialEpipolaireCoordinate::ToCoordInit(Pt2dr aP) const
{
   if (mExagEpip!=1.0)
   {
       static double aErMax = 0.0;
       ELISE_ASSERT(mApproxInvExagEpip ,"Can Invert Epip with mExagEpip");
       ELISE_ASSERT((!mCorCalc) ,"Can Invert Epip iwth Exag + CorCalc");
       double aY = aP.y;
       double aG =  aY;
       int aNb=10;
       
       for (int aK=0 ; aK<= aNb ; aK++)
       {
            aP = ToCoordEpipol(Pt2dr(aP.x,aG));
            double aEr = aP.y - aY;
            aG = aG - aEr;
            if (aK==aNb)
            {
                double aAe = ElAbs(aEr);
                if (aAe>aErMax)
                {
                    aErMax = aAe;
                    std::cout << "Errrr EpipInverse/Scale " << aK << " " << aErMax << "\n";
                }
            }
       }
       return Pt2dr(aP.x,aG);
   }
   aP.x = mCorCalc                                                                  ?  
	  (mNum0 + mNumy * aP.y - aP.x - mDeny *aP.y *aP.x) / (mDenx*aP.x -mNumx)   :
	  aP.x                                                                      ;
   return Pt2dr(aP.x,mPolToYInit(aP));
}


Polynome2dReal InvPolY(const Polynome2dReal & aPolY, REAL aDom,INT aDegre)
{
     Polynome2dReal aPolX(1,aPolY.Ampl());
     aPolX.SetDegre1(0,1,0);

     ElDistortionPolynomiale aDist( aPolX,aPolY);



     Polynome2dReal aRes = aDist.NewPolynLeastSquareInverse_OneCoord 
	     (
	        false,
		Box2dr(Pt2dr(0,0),Pt2dr(aDom,aDom)),
		aDegre
	    );

    return aRes;
}


PolynomialEpipolaireCoordinate::PolynomialEpipolaireCoordinate
(
     Pt2dr aP0,
     Pt2dr aDirX,
     const Polynome2dReal & aPolY,
     const Polynome2dReal * aPolInvY,
     REAL  aDom,
     INT   DeltaDegreInv,
     Pt2dr aTrFin
) :
   EpipolaireCoordinate(aP0,aDirX,aTrFin),
   mPolToYEpip (aPolY),
   mPolToYInit (aPolInvY ? *aPolInvY : (InvPolY(aPolY,aDom,aPolY.DMax()+DeltaDegreInv))),
   mNum0    (0.0),
   mNumx    (1.0),
   mNumy    (0.0),
   mDenx    (0.0),
   mDeny    (0.0),
   mCorCalc (false)
{
}

bool  PolynomialEpipolaireCoordinate::HasXFitHom() const
{
    return mCorCalc;
}

std::vector<double>  PolynomialEpipolaireCoordinate::ParamFitHom() const 
{
   return std::vector<double>({mNum0,mNumx,mNumy,mDenx,mDeny});
}

#define NBCOEFXFIT 5

void  PolynomialEpipolaireCoordinate::XFitHom(const ElPackHomologue & aPack,bool aL2,EpipolaireCoordinate *anEpi2)
{
   ELISE_ASSERT((!HasXFitHom()) && (!anEpi2->HasXFitHom()),"Multiple XFitHom");

   cGenSysSurResol * aSys =  nullptr;
   if (aL2)
      aSys = new L2SysSurResol(NBCOEFXFIT) ;
   else
      aSys = new SystLinSurResolu(NBCOEFXFIT,aPack.size()) ;

   aSys->SetPhaseEquation(nullptr);

    double aDx0 = 0.0;
    double aDy0 = 0.0;

    for (const auto & aCple : aPack)
    {
        Pt2dr aQ1 = Direct(aCple.P1());
        Pt2dr aQ2 = anEpi2->Direct(aCple.P2());

	/*std::cout<< "ElSwap(aQ1,aQ2)ElSwap(aQ1,aQ2)\n";
	ElSwap(aQ1,aQ2);*/

	// mC0 + mCx aQ1.x + mCy aQ1.y = aQ2.x
	// void V_GSSR_AddNewEquation(REAL aPds,REAL * aCoeff,REAL aB);
	double aCoeff[NBCOEFXFIT];
	aCoeff[0] = 1.0;
	aCoeff[1] = aQ1.x ;
	aCoeff[2] = aQ1.y;
	aCoeff[3] = -aQ1.x * aQ2.x;
	aCoeff[4] = -aQ1.y * aQ2.x;

	aDx0 += ElAbs(aQ1.x -aQ2.x);
	aDy0 += ElAbs(aQ1.y -aQ2.y);

	aSys->GSSR_AddNewEquation(1.0,aCoeff,aQ2.x,nullptr);
    }
    Im1D_REAL8 aSol = aSys->GSSR_Solve(nullptr);
    mNum0 = aSol.data()[0];
    mNumx = aSol.data()[1];
    mNumy = aSol.data()[2];
    mDenx = aSol.data()[3];
    mDeny = aSol.data()[4];
    mCorCalc = true;

    {
       double aDx1 = 0.0;
       double aDy1 = 0.0;
       double aDCoh1 = 0.0;
       double aDCoh2 = 0.0;
       for (const auto & aCple : aPack)
       {
           Pt2dr aQ1 = Direct(aCple.P1());
           Pt2dr aQ2 = anEpi2->Direct(aCple.P2());

	   aDx1 += ElAbs(aQ1.x -aQ2.x);
	   aDy1 += ElAbs(aQ1.y -aQ2.y);

	   aDCoh1 += euclid(aCple.P1()-Inverse(aQ1));
	   aDCoh2 += euclid(aCple.P2()-anEpi2->Inverse(aQ2));
	}

	// mC0 + mCx aQ1.x + mCy aQ1.y = aQ2.x
	int aNb = aPack.size();
	std::cout << "  XFITTTTTTTTTTTTTTTTTTTTTTTTT\n";
	std::cout << "BEFORE Dx:" << aDx0/aNb << " Dy:" << aDy0/aNb << "  \n";
	std::cout << "  AFTR Dx:" << aDx1/aNb << " Dy:" << aDy1/aNb << "  \n";
	std::cout << "  COHER 1:" << aDCoh1/aNb << " 2:" << aDCoh2/aNb << "  \n";
	// getchar();
    }

    delete aSys;
}



Polynome2dReal PolynomialEpipolaireCoordinate::PolToYEpip()
{
   return mPolToYEpip;
}

Polynome2dReal PolynomialEpipolaireCoordinate::PolToYInit()
{
   return mPolToYInit;
}

const PolynomialEpipolaireCoordinate * PolynomialEpipolaireCoordinate::CastToPol() const
{
   return this;
}

void PolynomialEpipolaireCoordinate::write(class  ELISE_fp & aFile) const
{
   aFile.write(P0());
   aFile.write(DirX());
   aFile.write(TrFin());
   
   aFile.write(mPolToYEpip);
   aFile.write(AmplInv());
   aFile.write(DeltaDegre());

   aFile.write(bool(mGridCor!=0));
   if (mGridCor!=0)
      mGridCor->write(aFile);
}

INT PolynomialEpipolaireCoordinate::DeltaDegre() const
{
   return mPolToYInit.DMax()-mPolToYEpip.DMax();
}

REAL  PolynomialEpipolaireCoordinate::AmplInv() const
{
    return mPolToYInit.Ampl();
}

PolynomialEpipolaireCoordinate 
    PolynomialEpipolaireCoordinate::read( ELISE_fp & aFile)
{
  Pt2dr aP0 = aFile.read(&aP0);
  Pt2dr aDirX = aFile.read(&aDirX);
  Pt2dr aTrFin = aFile.read(&aTrFin);
  Polynome2dReal aPol = aFile.read(&aPol);
  REAL anAmpl =  aFile.read(&anAmpl);
  INT aDeltaDegre = aFile.read(&aDeltaDegre);

   PolynomialEpipolaireCoordinate aRes(aP0,aDirX,aPol,0,anAmpl,aDeltaDegre);

   aRes.SetTrFinale(aTrFin);

  bool WithGrid = aFile.read((bool *)0);
  if (WithGrid)
     aRes.mGridCor = RImGrid::read(aFile);

   return aRes;
}

EpipolaireCoordinate * 
    PolynomialEpipolaireCoordinate::MapingChScale(REAL aChSacle) const
{
	return PolMapingChScale(aChSacle);
}

PolynomialEpipolaireCoordinate * 
    PolynomialEpipolaireCoordinate::PolMapingChScale
   (REAL aChSacle) const
{
     PolynomialEpipolaireCoordinate * aRes =
	           new PolynomialEpipolaireCoordinate
		   (
		        P0() * aChSacle,
			DirX(),
                        mPolToYEpip.MapingChScale(aChSacle),
                        0,
			AmplInv() * aChSacle,
			DeltaDegre(),
                        TrFin() * aChSacle

		   );

    if (mGridCor)
      aRes->mGridCor  = mGridCor->NewChScale(aChSacle,true);

    return aRes;
}



/*************************************************/
/*                                               */
/*       CpleEpipolaireCoord                     */
/*                                               */
/*************************************************/


CpleEpipolaireCoord::~CpleEpipolaireCoord()
{
   delete mEPI1;
   delete mEPI2;
}

void CpleEpipolaireCoord::SaveOrientCpleEpip
     (
                  const std::string &                anOri,
                  cInterfChantierNameManipulateur *  anICNM,
                  const std::string &                aName1,
                  const std::string &                aName2
     ) const
{
    EPI1().SaveOrientEpip(anOri,anICNM,aName1,aName2);
    EPI2().SaveOrientEpip(anOri,anICNM,aName2,aName1);
}


bool  CpleEpipolaireCoord::IsMappingEpi1() const
{
   return mEPI2->IsEpipId();
}

CpleEpipolaireCoord * CpleEpipolaireCoord::MapingChScale(REAL aChSacle) const
{
    return new CpleEpipolaireCoord
           (
               mEPI1->MapingChScale(aChSacle),
               mEPI2->MapingChScale(aChSacle)
           );
}

void CpleEpipolaireCoord:: SelfSwap()
{
   ElSwap(mEPI1,mEPI2);
}

CpleEpipolaireCoord * CpleEpipolaireCoord::Swap()
{
    return new  CpleEpipolaireCoord(mEPI2,mEPI1);
}

void CpleEpipolaireCoord::AdjustTr2Boxes(Box2dr aBoxIm1,Box2dr aBoxIm2)
{
    Box2dr aBoxEpi1 = mEPI1->ImageOfBox(aBoxIm1);
    Box2dr aBoxEpi2 = mEPI2->ImageOfBox(aBoxIm2);

    Pt2dr aP1  = aBoxEpi1._p0;
    Pt2dr aP2  = aBoxEpi2._p0;

    // REAL aX12 = ElMin(aP1.x,aP2.x);
    REAL aY12 = ElMin(aP1.y,aP2.y);

    mEPI1->AddTrFinale(Pt2dr(-aP1.x,-aY12));
    mEPI2->AddTrFinale(Pt2dr(-aP2.x,-aY12));
}

void CpleEpipolaireCoord::write(class  ELISE_fp & aFile) const
{
	mEPI1->CastToPol()->write(aFile);
	mEPI2->CastToPol()->write(aFile);
  
}

CpleEpipolaireCoord * CpleEpipolaireCoord::read( ELISE_fp & aFile)
{
   PolynomialEpipolaireCoordinate anEpi1 = 
	   PolynomialEpipolaireCoordinate::read(aFile);
   PolynomialEpipolaireCoordinate anEpi2 = 
	   PolynomialEpipolaireCoordinate::read(aFile);

   return new CpleEpipolaireCoord
	  (
	       new PolynomialEpipolaireCoordinate(anEpi1),
	       new PolynomialEpipolaireCoordinate(anEpi2)
	  );
}

CpleEpipolaireCoord::CpleEpipolaireCoord
(
    EpipolaireCoordinate * anEPI1,
    EpipolaireCoordinate * anEPI2
)  :
    mEPI1 (anEPI1),
    mEPI2 (anEPI2),
    mFact (1.0)
{
}

Pt2dr CpleEpipolaireCoord::Hom12(Pt2dr aP,REAL aParalaxe)
{
    return mEPI2->Inverse(mEPI1->Direct(aP)+Pt2dr(aParalaxe*mFact,0));
}

Pt2dr CpleEpipolaireCoord::Hom12(Pt2dr aP,Pt2dr aParalaxe)
{
    return mEPI2->Inverse(mEPI1->Direct(aP)+aParalaxe*mFact);
}



Pt2dr CpleEpipolaireCoord::Hom21(Pt2dr aP,REAL aParalaxe)
{
    return mEPI1->Inverse(mEPI2->Direct(aP)+Pt2dr(aParalaxe*mFact,0));
}

Pt2dr CpleEpipolaireCoord::Hom21(Pt2dr aP,Pt2dr aParalaxe)
{
    return mEPI1->Inverse(mEPI2->Direct(aP)+aParalaxe*mFact);
}


Pt2dr CpleEpipolaireCoord::Homol(Pt2dr aP,Pt2dr aParalaxe,bool Sens12)
{
    return  Sens12               ?
	    Hom12(aP,aParalaxe)  :
	    Hom21(aP,aParalaxe)  ;
}



const EpipolaireCoordinate & CpleEpipolaireCoord::EPI1() const { return *mEPI1; }
const EpipolaireCoordinate & CpleEpipolaireCoord::EPI2() const { return *mEPI2; }
EpipolaireCoordinate & CpleEpipolaireCoord::EPI1() { return *mEPI1; }
EpipolaireCoordinate & CpleEpipolaireCoord::EPI2() { return *mEPI2; }

/*
    On cherche a resoudre dans les reperes epipolaire (liés aux directio aDir1, 
    aDir2 comme axe des x) un couple de  transfo en y (Y1,Y2) tel que :

       Y1(P) = Y2(P) pour tous les points hom de aPackH

    Bien sur c'est indeterminé à une fonction pres , donc on impose ausi  pour lever l'arbitraire :
 
       Y1((0,y)) = y

        
*/

CpleEpipolaireCoord  * CpleEpipolaireCoord::PolynomialFromHomologue
                     (
                          bool                    UseL1,
                          CpleEpipolaireCoord  *  aSolApprox,
                          REAL                    aResiduMin,
                          const ElPackHomologue & aPackH,
                          INT   aDegre,
                          Pt2dr aDir1,
                          Pt2dr aDir2,
                          int aDeltaDeg 
                     )
{
   StatElPackH  aStat(aPackH);


   Polynome2dReal aPol1(aDegre,aStat.RMax1());
   Polynome2dReal aPol2(aDegre,aStat.RMax2());

   // Sur Pol1 on ne retient pas les term en Y^k qui sont la fonction correspondant a X=0
   INT aNbInc =0;
   for (INT k=0; k<aPol1.NbMonome() ; k++)
   {
      const Monome2dReal & aMon = aPol1.KthMonome(k);
      if (aMon.DegreX() != 0)
         aNbInc++;
      aNbInc++;
   }


   SystLinSurResolu aSys (aNbInc,aStat.NbPts()) ;
/*
   cGenSysSurResol * aSys =   (aSolApprox == (CpleEpipolaireCoord *)0 ) ?
                              new SystLinSurResolu(aNbInc,aStat.NbPts()) :
                              new L2SysSurResol(aNbInc);
*/
   Im1D_REAL8 aVecPds(aNbInc);
   REAL8 * aDVP = aVecPds.data();

   double aRMaxInv1 = 0;
   double aRMaxInv2 = 0;

   for
   (
        ElPackHomologue::const_iterator itC = aPackH.begin();
        itC != aPackH.end();
        itC++
   )
   {
      REAL aPdsResidu = 1;
      if (aSolApprox)
      {
          Pt2dr aQ1 = aSolApprox->EPI1().Direct(itC->P1());
          Pt2dr aQ2 = aSolApprox->EPI2().Direct(itC->P2());

          REAL aResidu = ElAbs(aQ1.y-aQ2.y);
          // aPdsResidu =  1/sqrt(ElSquare(aResidu)+ElSquare(aResiduMin));
          aPdsResidu =  1/(ElSquare(aResidu)+ElSquare(aResiduMin));
      }
      Pt2dr aP1 = ( itC->P1() -aStat.Cdg1()) / aDir1;
      Pt2dr aP2 = ( itC->P2() -aStat.Cdg2()) / aDir2;

      ElSetMax(aRMaxInv1,euclid(aP1));
      ElSetMax(aRMaxInv2,euclid(aP2));

      aNbInc=0;
      for (INT k=0; k<aPol1.NbMonome() ; k++)
      {
         const Monome2dReal & aMon1 = aPol1.KthMonome(k);
         if (aMon1.DegreX() != 0)
         {
            aDVP[aNbInc++] = -aMon1(aP1);
         }
         const Monome2dReal & aMon2 = aPol2.KthMonome(k);
         aDVP[aNbInc++] = aMon2(aP2);
      }
      // Equation 
      //  y1 =  Som(a1 X1^i Y1 ^j) + Som(a2 X2^i Y2 ^j) 
      //        i!=0
      // aSys->GSSR_AddNewEquation(itC->Pds()*aPdsResidu,aDVP,aP1.y);
      aSys.PushEquation(aVecPds,aP1.y,itC->Pds()*aPdsResidu);
   }

   
   bool aOk;
   // Im1D_REAL8  aSol =  aSys->GSSR_Solve(&aOk);
   Im1D_REAL8  aSol = ( (aSolApprox || (!UseL1)) ? aSys.L2Solve(&aOk) : aSys.L1Solve());

   aNbInc=0;
   {
      for (INT k=0; k<aPol1.NbMonome() ; k++)
      {
           const Monome2dReal & aMon1 = aPol1.KthMonome(k);
           if (aMon1.DegreX() != 0)
           {
               aPol1.SetCoeff(k,aSol.data()[aNbInc++]);
           }
           else
           {
               aPol1.SetCoeff(k,(aMon1.DegreY() == 1)*aStat.RMax1());
           }

           aPol2.SetCoeff(k,aSol.data()[aNbInc++]);
      }
   }




   Polynome2dReal aPolInv1(aDegre+aDeltaDeg,aRMaxInv1);
   int aNbIncInv = aPolInv1.NbMonome();
   SystLinSurResolu aSysInv1 (aNbIncInv,aStat.NbPts()) ;
   Im1D_REAL8 aVPInv1(aNbIncInv);
   REAL8 *    aDVPI1 = aVPInv1.data();

   Polynome2dReal aPolInv2(aDegre+aDeltaDeg,aRMaxInv2);
   SystLinSurResolu aSysInv2 (aNbIncInv,aStat.NbPts()) ;
   Im1D_REAL8 aVPInv2(aNbIncInv);
   REAL8 *    aDVPI2 = aVPInv2.data();

   for
   (
        ElPackHomologue::const_iterator itC = aPackH.begin();
        itC != aPackH.end();
        itC++
   )
   {
      Pt2dr aP1 = ( itC->P1() -aStat.Cdg1()) / aDir1;
      Pt2dr aP2 = ( itC->P2() -aStat.Cdg2()) / aDir2;

      Pt2dr aQ1 (aP1.x,aPol1(aP1));
      Pt2dr aQ2 (aP2.x,aPol2(aP2));

      for (int aK=0 ; aK<aNbIncInv ; aK++)
      {
           aDVPI1[aK] =  aPolInv1.KthMonome(aK)(aQ1);
           aDVPI2[aK] =  aPolInv2.KthMonome(aK)(aQ2);
      }
      aSysInv1.PushEquation(aDVPI1,aP1.y,1.0);
      aSysInv2.PushEquation(aDVPI2,aP2.y,1.0);
   }

   Im1D_REAL8  aSolInv1 =  aSysInv1.L2Solve(&aOk) ;
   Im1D_REAL8  aSolInv2 =  aSysInv2.L2Solve(&aOk) ;
      
   for (INT k=0; k<aNbIncInv ; k++)
   {
        aPolInv1.SetCoeff(k,aSolInv1.data()[k]);
        aPolInv2.SetCoeff(k,aSolInv2.data()[k]);
   }


   PolynomialEpipolaireCoordinate * anEpi1 = new PolynomialEpipolaireCoordinate
                                                 (
                                                      aStat.Cdg1(),
                                                      aDir1,
                                                      aPol1,
                                                      &aPolInv1,
                                                      aStat.RMax1()
                                                 );

   PolynomialEpipolaireCoordinate * anEpi2 = new PolynomialEpipolaireCoordinate
                                                 (
                                                      aStat.Cdg2(),
                                                      aDir2,
                                                      aPol2,
                                                      &aPolInv2,
                                                      aStat.RMax2()
                                                 );

   // delete aSys;

   return  new CpleEpipolaireCoord(anEpi1,anEpi2);
}





CpleEpipolaireCoord  * CpleEpipolaireCoord::PolynomialFromHomologue
                     (
                          bool UseL1,
                          const ElPackHomologue & aPackH,
                          INT   aDegre,
                          Pt2dr aDir1,
                          Pt2dr aDir2,
                          int   aDeltaDeg
                     )
{
      return   PolynomialFromHomologue
               (
                     UseL1,
                     (CpleEpipolaireCoord *) 0,
                     1.0,
                     aPackH, aDegre,
                     aDir1,aDir2,
                     aDeltaDeg
                );
}

CpleEpipolaireCoord  * CpleEpipolaireCoord::PolynomialFromHomologue
                     (
                          const ElPackHomologue & aPackHL1,
                          INT   aDegreL1,
                          const ElPackHomologue & aPackHL2,
                          INT   aDegreL2,
                          Pt2dr aDir1,
                          Pt2dr aDir2,
                          int   aDeltaDeg
                     )
{
    CpleEpipolaireCoord * aSol1 = PolynomialFromHomologue(true,aPackHL1,aDegreL1,aDir1,aDir2,aDeltaDeg);

    CpleEpipolaireCoord * aSol2 =  PolynomialFromHomologue(true,aSol1,0.5,aPackHL2,aDegreL2,aDir1,aDir2,aDeltaDeg);
    delete aSol1;

    return aSol2;
/*
    CpleEpipolaireCoord * aSol2 =  PolynomialFromHomologue(aSol1,0.3,aPackHL2,aDegreL2,aDir1,aDir2);
    CpleEpipolaireCoord * aSol3 =  PolynomialFromHomologue(aSol2,0.1,aPackHL2,aDegreL2,aDir1,aDir2);

    delete aSol1;
    delete aSol2;
    return aSol3;
*/
}



CpleEpipolaireCoord * CpleEpipolaireCoord::MappingEpipolaire
                      (
                             ElDistortion22_Gen * aDist,
                             bool ToDel
                      )
{
   return  new CpleEpipolaireCoord
               (
                   new cMappingEpipCoord(aDist,ToDel),
                   new cMappingEpipCoord(new  ElDistortion22_Triviale,true)
               ); 
}

CpleEpipolaireCoord * CpleEpipolaireCoord::MappEpiFromHomographie (cElHomographie anHomog)
{
   return MappingEpipolaire(new cDistHomographie(anHomog),true);
}


CpleEpipolaireCoord * 
    CpleEpipolaireCoord::MappEpiFromHomographieAndDist 
   (
        const cElHomographie & aH,
        const ElDistRadiale_PolynImpair & aDist,
	REAL  aR,
        INT   aDeg
   )
{
   return MappingEpipolaire(new cDistHomographieRadiale(aH,aDist,aR,aDeg),true);
}


CpleEpipolaireCoord * CpleEpipolaireCoord::OriEpipolaire
                      (
                            const std::string & aName1, Pt2dr aP1,
                            const std::string & aName2, Pt2dr aP2,
                            REAL aZoom
                      )
{
   Data_Ori3D_Std   * Z1Ori1 = new  Data_Ori3D_Std (aName1.c_str(),false,false,false);
   Data_Ori3D_Std   * Z1Ori2 = new  Data_Ori3D_Std (aName2.c_str(),false,false,false);

   Data_Ori3D_Std   * Ori1 = new Data_Ori3D_Std (Z1Ori1,aZoom);
   Data_Ori3D_Std   * Ori2 = new Data_Ori3D_Std (Z1Ori2,aZoom);


   delete Z1Ori1;
   delete Z1Ori2;

   aP1 = aP1 * aZoom;
   aP2 = aP2 * aZoom;
   Data_Ori3D_Std * anEpi1,* anEpi2;
   Data_Ori3D_Std::ororient_epipolaires
   (
          anEpi1,*Ori1,aP1,
          anEpi2,*Ori2,aP2
   );


   cOri3D_OneEpip * aS1 = new cOri3D_OneEpip(Ori1,anEpi1);
   cOri3D_OneEpip * aS2 = new cOri3D_OneEpip(Ori2,anEpi2);

    
   cMappingEpipCoord * aMap1 = new cMappingEpipCoord(aS1,true);
   cMappingEpipCoord * aMap2 = new cMappingEpipCoord(aS2,true);


    Pt2dr aQ1 = aMap1->Direct(aP1);
    Pt2dr aQ2 = aMap2->Direct(aP2);


    aMap1->AddTrFinale(Pt2dr(aQ2.x-aQ1.x,0));

   return new CpleEpipolaireCoord(aMap1,aMap2);

}

/****************************************************************/
/*                                                              */
/*              Fnum_O3d_phot_et_z_to_terrain                   */
/*                                                              */
/****************************************************************/

class Fnum_CpleEpi_Hom12  : public Simple_OP_UN<REAL>
{
     public :
           Fnum_CpleEpi_Hom12 (CpleEpipolaireCoord * aCple) :
              mCple (aCple)
           {
           }

     private :

         CpleEpipolaireCoord *  mCple;

         virtual void  calc_buf
         (
                           REAL ** output,
                           REAL ** input,
                           INT        nb,
                           const Arg_Comp_Simple_OP_UN  &
         ) ;
};

void Fnum_CpleEpi_Hom12 ::calc_buf
     (
              REAL ** output,
              REAL ** input,
              INT        nb,
              const Arg_Comp_Simple_OP_UN  &
     )
{
   REAL * xIm1 = input[0];
   REAL * yIm1 = input[1];
   REAL * zParax = input[2];
   REAL * tParax = input[3];


   REAL * xIm2 = output[0];
   REAL * yIm2 = output[1];


   for (INT aK=0 ; aK<nb ; aK++)
   {
       Pt2dr aP2  =  mCple->Hom12
                     (
                         Pt2dr(xIm1[aK],yIm1[aK]),
                         Pt2dr(zParax[aK],tParax[aK])
                     );

       xIm2[aK] = aP2.x;
       yIm2[aK] = aP2.y;
   }
}


Pt2d<Fonc_Num> CpleEpipolaireCoord ::Hom12
               (
                       Pt2d<Fonc_Num>     aPI1,
                       Pt2d<Fonc_Num>     aParax
               )
{
    Symb_FNum aFonc
              ( 
                    create_users_oper
                    (
                         0,
                         new Fnum_CpleEpi_Hom12(this),
                         Virgule(aPI1.x,aPI1.y,aParax.x,aParax.y),
                         4
                    )
             );

    return Pt2d<Fonc_Num>(aFonc.v0(),aFonc.v1());
}



/**********************************************/
/*                                            */
/*     EpipolaireCoordinateNoDist             */
/*                                            */
/**********************************************/

EpipolaireCoordinateNoDist::EpipolaireCoordinateNoDist
(
    Pt2dr aP0,
    Pt2dr aDirX
)  :
   EpipolaireCoordinate(aP0,aDirX,Pt2dr(0,0))
{
}

Pt2dr EpipolaireCoordinateNoDist::ToCoordEpipol(Pt2dr aPInit) const 
{
	return aPInit;
}

Pt2dr EpipolaireCoordinateNoDist::ToCoordInit(Pt2dr aPInit) const 
{
	return aPInit;
}


EpipolaireCoordinate * EpipolaireCoordinateNoDist::MapingChScale(REAL aChSacle) const
{
     EpipolaireCoordinateNoDist * aRes = new EpipolaireCoordinateNoDist(P0(),DirX());
     aRes->HeriteChScale(const_cast<EpipolaireCoordinateNoDist&>(*this),aChSacle);
     return aRes;
}

CpleEpipolaireCoord * CpleEpipolaireCoord::EpipolaireNoDist
                      (Pt2dr aPHom1,Pt2dr aPHom2,Pt2dr aDir1,Pt2dr aDir2)
{
	return new CpleEpipolaireCoord
		   (
		       new EpipolaireCoordinateNoDist(aPHom1,aDir1),
		       new EpipolaireCoordinateNoDist(aPHom2,aDir2)
		   );
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
