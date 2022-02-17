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
/*    ElDistortion22                             */
/*                                               */
/*************************************************/


REAL ElDistortion22_Gen::DistanceObjet
    (INT tagDist,const ElDistortion22_Gen & aDist,Pt2dr P0, Pt2dr P1,INT NbEch) const
{
    Pt2dr d01 = (P1-P0)/REAL (NbEch);

    INT aNB = 0;

    REAL dRes = 0;

    for (INT kX=0; kX<=NbEch ; kX++)
    {
        for (INT kY=0; kY<=NbEch ; kY++)
        {
             aNB++;
             Pt2dr aP = P0 +d01.mcbyc(Pt2dr(kX,kY));

             REAL dist = euclid(Direct(aP),aDist.Direct(aP));
             switch (tagDist)
             {
                   case  1 : dRes += dist;
                   break;

                   case  2 : dRes += ElSquare(dist);
                   break;

                   default : ElSetMax(dRes,dist);
                   break;
             }
        }
   }

    switch (tagDist)
    {
            case  1 : return dRes/aNB;
            break;

            case  2 : return sqrt(dRes/aNB);
            break;
    }
    return dRes;
}

REAL ElDistortion22_Gen::D1
    (const ElDistortion22_Gen & aDist,Pt2dr P0, Pt2dr P1,INT NbEch) const
{
   return DistanceObjet(1,aDist,P0,P1,NbEch);
}

REAL ElDistortion22_Gen::D2
    (const ElDistortion22_Gen & aDist,Pt2dr P0, Pt2dr P1,INT NbEch) const
{
   return DistanceObjet(2,aDist,P0,P1,NbEch);
}

REAL ElDistortion22_Gen::DInfini
    (const ElDistortion22_Gen & aDist,Pt2dr P0, Pt2dr P1,INT NbEch) const
{
   return DistanceObjet(-1,aDist,P0,P1,NbEch);
}


std::string ElDistortion22_Gen::Type() const
{
    return "UnknowDistType";
}

ElCamera * ElDistortion22_Gen::CameraOwner()
{
   return mCameraOwner;
}


void ElDistortion22_Gen::SetCameraOwner(ElCamera* aCam)
{
    mCameraOwner = aCam;
}


ElDistortion22_Gen::ElDistortion22_Gen() :
    mPolynInv         (0),
    mEpsInvDiff       (1e-8),
    mNbIterMaxInvDiff (1000),
    mScN              (1.0),
    mDist22Gen_UsePreConditionner (true),
    mDist22Gen_SupressPreCondInInverse (false),
    mName (0),
    mCameraOwner (0)
{
}


void ElDistortion22_Gen::SetName(const char * aName)
{
   mName = aName;
}

std::string ElDistortion22_Gen::Name() const
{
   return mName ? std::string(mName) : "Unknown";
}

void ElDistortion22_Gen::ErrorInvert() const
{
   std::cout << "\n\n   #####  For distortion of name " << Name()  << " " << (void *) this<< "#####\n\n";
   ELISE_ASSERT
   (
       false,
       "Distortion Inversion  by finite difference do not converge (probably ill-conditioned canvas)"
   );
}

const bool &   ElDistortion22_Gen::Dist22Gen_UsePreConditionner() const
{
    return mDist22Gen_UsePreConditionner;
}
void    ElDistortion22_Gen::SetDist22Gen_UsePreConditionner(bool aUPC) const
{
   const_cast<ElDistortion22_Gen *>(this)->mDist22Gen_UsePreConditionner = aUPC;
}


const bool &   ElDistortion22_Gen::Dist22Gen_SupressPreCondInInverse() const
{
    return mDist22Gen_SupressPreCondInInverse;
}
void    ElDistortion22_Gen::SetDist22Gen_SupressPreCondInInverse(bool aUPC) const
{
   const_cast<ElDistortion22_Gen *>(this)->mDist22Gen_SupressPreCondInInverse = aUPC;
}




double & ElDistortion22_Gen::ScN() {return mScN;}
const double & ElDistortion22_Gen::ScN() const {return mScN;}

void ElDistortion22_Gen::SetParamConvInvDiff(INT aNbIter,REAL aEps)
{
   mNbIterMaxInvDiff = aNbIter;
   mEpsInvDiff = aEps;
}


Pt2dr  ElDistortion22_Gen::DirectAndDer(Pt2dr aP,Pt2dr & aGradX,Pt2dr & aGradY) const
{
   ELISE_ASSERT(false,"ElDistortion22_Gen::DirectAndDer");
   return Pt2dr(0,0);
}


ElDistortion22_Gen::~ElDistortion22_Gen()
{
   delete mPolynInv;
}
extern bool BugNanFE;


Pt2dr  ElDistortion22_Gen::ComputeInvFromDirByDiff
       (
                     Pt2dr aPt,
                     Pt2dr anEstim0,
                     bool Reestim
       ) const
{

    Pt2dr aP00 = anEstim0;

static int aCpt=0 ; aCpt++;

     int aNbRee = 0;
     ElMatrix<REAL> aDiff = Diff(anEstim0);
     REAL aN = sqrt(aDiff.L2()/4.0);
     REAL Epsilon = mEpsInvDiff * aN / (1+aN);
 
    self_gaussj(aDiff);

 // std::cout << "Epsilon = " << Epsilon << "\n";

     int NbStep = 0;

     for (bool cont= true; cont; )
     {
         if (std_isnan(anEstim0.x) || std_isnan(anEstim0.y))
         {
               std::cout << "NAN Values in inversion,  target " << aPt << " ; guess " << aP00  << " Cpt= " << aCpt<< "\n";
               ErrorInvert();
         }
         Pt2dr DisEst    = Direct(anEstim0);
         if (
                   (euclid(DisEst,aPt)<Epsilon)
                || (NbStep > mNbIterMaxInvDiff)
            )
         {
            if (NbStep > mNbIterMaxInvDiff)
            {
               std::cout << "Slow convergence Values in inversion\n";
               std::cout <<  "Nb Step = " << NbStep << " Cur " << anEstim0
                         << "  RESIDU " << euclid(DisEst,aPt)
                         << " Dist " << DisEst << " Cible " << aPt  << " ESTIM0 " << aP00 << "\n";

               ErrorInvert();
            }
            cont = false;
         }
         else
         {
             NbStep++;
             if (Reestim) // || (NbStep==2) || (NbStep==5))
             {
                Diff(aDiff,anEstim0);
                self_gaussj(aDiff);
                aNbRee++;
             }
             anEstim0 += aDiff * (aPt-DisEst);
         }
     }
     static int aNbStepMax =0;
     if (NbStep > aNbStepMax)
     {
          aNbStepMax  = NbStep;

          if ((aNbStepMax > 5) && MPD_MM())
          {
               std::cout  << "NB-STEP-MAX " << aNbStepMax  << " P=" << aPt << "Inv=" << anEstim0<< "\n";
          }
     }

     return anEstim0;
}

/*
Pt2dr ElDistortion22_Gen::EstimInvByDiffen00(Pt2dr aPt) const
{
     Pt2dr aG = GuessInv(aPt);
     return   aG
              + gaussj(ElDistortion22_Gen::Diff(aG)) *(aPt-Direct(aG));
     // return  gaussj(ElDistortion22_Gen::Diff(Pt2dr(0,0))) * (aPt-Direct(Pt2dr(0,0)));
}
*/

bool ElDistortion22_Gen::OwnInverse(Pt2dr & p ) const
{
   return false;
}

ElMatrix<REAL> ElDistortion22_Gen::Diff(Pt2dr p) const
{

    ElMatrix<REAL> M(2,2);
    Diff(M,p);
    return M;
}

bool DebugCorona=false;

void ElDistortion22_Gen::DiffByDiffFinies
     (ElMatrix<REAL> & aMat,Pt2dr aP,Pt2dr Eps) const
{


     Eps = Eps/mScN;
     Pt2dr dx(Eps.x,0);
     Pt2dr dy(0,Eps.y);

if (0)
{
	DebugCorona=true;
    std::cout << "ion22_Gen::DiffByD " << aP << " " << Eps << " ScN " << mScN << "\n";
    std::cout << Direct(aP) << " " << Direct(aP+dx)-Direct(aP-dx) << "\n";
    getchar();
}

     SetCol(aMat,0,(Direct(aP+dx)-Direct(aP-dx))/(2.0*Eps.x));
     SetCol(aMat,1,(Direct(aP+dy)-Direct(aP-dy))/(2.0*Eps.y));
}
void ElDistortion22_Gen::DiffByDiffFinies
     (ElMatrix<REAL> & aMat,Pt2dr aP,REAL Eps) const
{
    DiffByDiffFinies(aMat,aP,Pt2dr(Eps,Eps));
}


bool   ElDistortion22_Gen::AcceptScaling() const
{
   return false;
}

bool  ElDistortion22_Gen::AcceptTranslate() const
{
   return false;
}

void ElDistortion22_Gen::V_SetScalingTranslate(const double &,const Pt2dr &)
{
   ELISE_ASSERT(false,"ElDistortion22_Gen::SetScaling");
}

void  ElDistortion22_Gen::SetScalingTranslate(const double & aS,const Pt2dr &aTr)
{
   bool doTr = false;
   bool doSc = false;

   if (AcceptScaling())
   {
      doSc = true;
      mScN *= aS;
   }
   else
   {
      ELISE_ASSERT(aS==1,"No Scale for this dist");
   }


   if (AcceptTranslate())
   {
      doTr = true;
   }
   else
   {
      ELISE_ASSERT(aTr == Pt2dr(0,0),"No Scale for this dist");
   }

   if (doTr || doSc)
   {
      V_SetScalingTranslate(aS,aTr);
   }
}





Pt2dr ElDistortion22_Gen::Inverse(Pt2dr aP) const
{
    if (OwnInverse(aP))
       return aP;

    if (mPolynInv)
       return mPolynInv->Direct(aP);

static int aCpt=0; aCpt++;

    // std::cout  << "CPT = " << aCpt << "\n";
  // BugFE=(aCpt==208501);
/*
if (BugFE)
*/

if (BugFE)
{
   std::cout << "ElDistortion22_Gen::Inverse\n";
   std::cout << aP << " " << GuessInv(aP) << "\n\n";
}


    return ComputeInvFromDirByDiff ( aP, GuessInv(aP), false);
}

Pt2dr ElDistortion22_Gen::GuessInv(const Pt2dr & aP) const
{
    // return Pt2dr(0,0);
    return aP;
}

Polynome2dReal  ElDistortion22_Gen::NewPolynLeastSquareInverse_OneCoord
                (
                        bool XCoord ,
                        Box2dr aDom,
                        INT  aDegre,
                        INT  aNbPts
                )
{
   if (aNbPts == -1)
   {
      aNbPts = 2 * aDegre * aDegre * aDegre + 20 * aDegre * aDegre + 200;
   }

   double anAmpl= ElMax(dist8(aDom._p0),dist8(aDom._p1));
   Polynome2dReal aRes(aDegre,anAmpl);
   INT aNbM = aRes.NbMonome();
   INT aNbD1 = round_up(sqrt(double(aNbPts)));
   SystLinSurResolu aSys(aNbM,ElSquare(1+aNbD1));
   Im1D_REAL8 aFLin(aNbM);
   REAL * aDataFL = aFLin.data();


   for (INT aKx =0 ; aKx <= aNbD1 ; aKx++)
       for (INT aKy =0 ; aKy <= aNbD1 ; aKy++)
         {
             Pt2dr aP = aDom.FromCoordLoc(Pt2dr(aKx,aKy)/double(aNbD1));
             for (INT kMon=0 ; kMon<aNbM ; kMon++)
                 aDataFL[kMon] = aRes.KthMonome(kMon)(Direct(aP)) ;
             aSys.PushEquation(aFLin,(XCoord ? aP.x : aP.y));
         }

     Im1D_REAL8 aSol = aSys.L2Solve(0);

     for (INT kMon=0 ; kMon<aNbM ; kMon++)
         aRes.SetCoeff(kMon,aSol.data()[kMon]);

    return aRes;
}


ElDistortionPolynomiale ElDistortion22_Gen::NewPolynLeastSquareInverse
                        (
                             Box2dr aDom,
                             INT  aDegre,
                             INT  aNbPts
                        )
{
   if (aNbPts == -1)
   {
      aNbPts = 2 * aDegre * aDegre * aDegre + 20 * aDegre * aDegre + 200;
   }

   return ElDistortionPolynomiale
          (
             NewPolynLeastSquareInverse_OneCoord(true,aDom,aDegre,aNbPts),
             NewPolynLeastSquareInverse_OneCoord(false,aDom,aDegre,aNbPts)
          );
}

void ElDistortion22_Gen::SaveAsGrid
     (
             const std::string & aName,
             const Pt2dr& aP0,
             const Pt2dr& aP1,
             const Pt2dr& aStep
     )
{
    cDbleGrid aDistGr(true, // PAS TRES SUR, maintien de l'existant
                      true,aP0,aP1,aStep,*this);
    aDistGr.SaveXML(aName);
}

class cElDG_ChScale : public ElDistortion22_Gen
{
    public :
          cElDG_ChScale(REAL aChSc,const ElDistortion22_Gen * aDist) :
             mChSc (aChSc),
             mDist (aDist)
          {
          }

          Pt2dr Direct(Pt2dr aP) const
          {
              return  mDist->Direct(aP/mChSc) * mChSc;
          }

          bool OwnInverse(Pt2dr & aP) const
          {
               aP = mDist->Inverse(aP/mChSc) * mChSc;
               return true;
          }
      void  Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
      {
                Diff(aMat,aP/mChSc);
      }


    private:

             REAL                           mChSc;
             const ElDistortion22_Gen *     mDist;
};

ElDistortion22_Gen  * ElDistortion22_Gen::D22G_ChScale(REAL aS) const
{
    return new cElDG_ChScale(aS,this);
}

class cElDG_Inverse : public ElDistortion22_Gen
{
    public :
          cElDG_Inverse(const ElDistortion22_Gen * aDist) :
             mDist (aDist)
          {
          }

          Pt2dr Direct(Pt2dr aP) const
          {
              return  mDist->Inverse(aP);
          }

          bool OwnInverse(Pt2dr & aP) const
          {
               aP = mDist->Direct(aP);
               return true;
          }
      void  Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
      {
                Diff(aMat,aP);
                self_gaussj(aMat);
      }

    private:
             const ElDistortion22_Gen *     mDist;
};

ElDistortion22_Gen  * ElDistortion22_Gen::CalcInverse() const
{
    return new cElDG_Inverse(this);
}

bool ElDistortion22_Gen::IsId() const {return false;}

ElDistRadiale_PolynImpair *ElDistortion22_Gen::DRADPol(bool strict)
{
    return 0;
}


Box2dr ElDistortion22_Gen::ImageOfBox(Box2dr aBox,INT aNbPtsDisc)
{
   static std::vector<Pt2dr>  aVPts;
   aBox.PtsDisc(aVPts,aNbPtsDisc);

   Pt2dr aP0 = Direct(aVPts[0]);
   Pt2dr aP1 = Direct(aVPts[0]);

   for (INT aKP=0 ; aKP<INT(aVPts.size()) ; aKP++)
   {
       aP0.SetInf(Direct(aVPts[aKP]));
       aP1.SetSup(Direct(aVPts[aKP]));
   }
   return Box2dr(aP0,aP1);
}

Box2dr ElDistortion22_Gen::ImageRecOfBox(Box2dr aBox,INT aNbPtsDisc)
{
   cElDG_Inverse aInv(this);
   return aInv.ImageOfBox(aBox,aNbPtsDisc);
}

void  ElDistortion22_Gen::Diff(ElMatrix<REAL> &,Pt2dr) const
{
   ELISE_ASSERT(false,"No ElDistortion22_Gen::Diff");
}

cPreCondGrid ElDistortion22_Gen::GetAsPreCond() const
{
    ELISE_ASSERT
    (
       false,
       "ElDistortion22_Gen as no default PreCond conversion"
    );
    cPreCondGrid aRes;
    return aRes;
}

cCalibDistortion GlobXmlDistNoVal()
{
   cCalibDistortion  aRes;

   aRes.ModNoDist().SetNoInit();
   aRes.ModRad().SetNoInit();
   aRes.ModPhgrStd().SetNoInit();
   aRes.ModUnif().SetNoInit();
   aRes.ModGridDef().SetNoInit();

   return aRes;
}

cCalibrationInterneRadiale CIR(const Pt2dr & aC)
{
   cCalibrationInterneRadiale aCIR;
   aCIR.CDist() = aC;
   aCIR.PPaEqPPs().SetNoInit();

   return aCIR;
}
cCalibDistortion FromCIR(const cCalibrationInterneRadiale & aCIR)
{
    cCalibDistortion  aRes = GlobXmlDistNoVal();
    aRes.ModRad().SetVal(aCIR);

    return aRes;
}


cCalibDistortion  ElDistortion22_Gen::XmlDistNoVal()
{
   return GlobXmlDistNoVal();
}



cCalibDistortion  ElDistortion22_Gen::ToXmlStruct(const ElCamera * aCam) const
{
   return FromCIR(CIR(aCam->Sz()/2.0));
}
/*
   cCalibDistortion aRes ;
   cModNoDist aNoDist;
   aRes.ModNoDist().SetVal(aNoDist);

   return aRes;
*/


/*************************************************/
/*                                               */
/*    ElDistortion22_Triviale                    */
/*                                               */
/*************************************************/


ElDistortion22_Triviale  ElDistortion22_Triviale::TheOne;

Pt2dr ElDistortion22_Triviale::Direct(Pt2dr p) const
{
   return p;
}

bool ElDistortion22_Triviale::OwnInverse(Pt2dr & p ) const
{
   return true;
}


void  ElDistortion22_Triviale::Diff(ElMatrix<REAL> & M,Pt2dr) const
{
     M = ElMatrix<REAL>(2);
}


ElDistortion22_Gen  * ElDistortion22_Triviale::D22G_ChScale(REAL aS) const
{
   return new ElDistortion22_Triviale;
}

bool ElDistortion22_Triviale::IsId() const {return true;}


cCalibDistortion ElDistortion22_Triviale::ToXmlStruct(const ElCamera * ) const
{
   cCalibDistortion aRes =XmlDistNoVal(); //  ElDistortion22_Gen::BasicToXmlStruct();
   cModNoDist aNoDist;
   aRes.ModNoDist().SetVal(aNoDist);

   return aRes;
}


/*************************************************/
/*                                               */
/*    ElDistRadiale                              */
/*                                               */
/*************************************************/

ElDistRadiale::ElDistRadiale(Pt2dr Centre) :
   _centre (Centre)
{
}

REAL ElDistRadiale::K0() const
{
    return 1.0;
}

Pt2dr ElDistRadiale::Direct(Pt2dr p0) const
{
      Pt2dr u = p0-_centre;

///std::cout << "RRRRR " << DistDirecteR2(square_euclid(u)) << "\n";

      return _centre +  u * (K0()*(1 +DistDirecteR2(square_euclid(u))));
}

bool ElDistRadiale::OwnInverse(Pt2dr & p0) const
{

      Pt2dr u = p0-_centre;
      REAL R = euclid(u);
      REAL dr = DistInverse(R);

      p0 =  _centre +  u * ((1+dr)/K0());
      return true;
}

REAL ElDistRadiale::DistInverse(REAL r) const
{
    REAL rk = r;
    for (INT k=0;k<10 ; k++)
        rk = r/(K0()*(1+DistDirecte(rk)));

   return 1 /(1+DistDirecte(rk)) - 1;

}


void ElDistRadiale::Diff(ElMatrix<REAL> & M,Pt2dr p0) const
{
      M.set_to_size(2,2);

      Pt2dr u       = p0-_centre;
      REAL R        = euclid(u);
      REAL dist     = DistDirecte(R);
      REAL der_dist = DerSurRho(R);

      REAL k0 = K0();

      M(0,0) = k0*(1+dist+der_dist*ElSquare(u.x));
      M(0,1) = k0*der_dist*u.x*u.y;
      M(1,0) = k0*der_dist*u.x*u.y;
      M(1,1) = k0*(1+dist+der_dist*ElSquare(u.y));
}

Pt2dr  & ElDistRadiale::Centre()
{
   return _centre;
}

const Pt2dr  & ElDistRadiale::Centre() const
{
   return _centre;
}

/*************************************************/
/*                                               */
/*    ElDistRadiale_PolynImpair                  */
/*                                               */
/*************************************************/

ElDistRadiale_PolynImpair::ElDistRadiale_PolynImpair(REAL aRMax,Pt2dr aCentre) :
     ElDistRadiale(aCentre),
     mRMax        (aRMax),
     mRMaxP2N     (1.0),
     mValRMax     (0.0),
     mDiffRMax    (0.0)
{
}

REAL ElDistRadiale_PolynImpair::RMax() const
{
   return mRMax;
}
REAL ElDistRadiale_PolynImpair::ValRMax() const
{
   return mValRMax;
}
REAL ElDistRadiale_PolynImpair::DiffRMax() const
{
   return mDiffRMax;
}

void ElDistRadiale_PolynImpair::PushCoeff(REAL aCoeff)
{
   mCoeffs.push_back(aCoeff);
   mRMaxP2N *= ElSquare(mRMax);
   mValRMax += aCoeff * mRMaxP2N * mRMax;
   mDiffRMax += aCoeff * mRMaxP2N * (1+2*mCoeffs.size());


}

void ElDistRadiale_PolynImpair::PushCoeff
     (const std::vector<REAL> & aVCoeff)
{
   for (INT aK=0 ; aK<INT(aVCoeff.size()); aK++)
       PushCoeff (aVCoeff[aK]);
}
void ElDistRadiale_PolynImpair::SetRMax(REAL aV)
{
    mRMaxP2N = 1;
    mValRMax = 0;
    mDiffRMax = 0;


    std::vector<REAL> Coeffs = mCoeffs;
    mRMax =aV;
    mCoeffs.clear();
    PushCoeff(Coeffs);

}

void  ElDistRadiale_PolynImpair::ActuRMax()
{
    SetRMax(mRMax);
}

void ElDistRadiale_PolynImpair::ActuRMaxFromDist(Pt2di aSz)
{
    SetRMax(DMaxCoins(Pt2dr(aSz),Centre()));
}


REAL ElDistRadiale_PolynImpair::DistDirecte(REAL R) const
{
    return DistDirecteR2( ElSquare(R));
}

REAL ElDistRadiale_PolynImpair::DistDirecteR2(REAL R2) const
{
    if (R2 > ElSquare(mRMax))
    {
         REAL R = sqrt(R2);
         return (mValRMax + mDiffRMax * (R-mRMax)) / R;
    }
    return DistDirecteR2NoSeuil(R2);
}

REAL ElDistRadiale_PolynImpair::DistDirecteR2NoSeuil(REAL R2) const
{
    REAL aRes = 0.0;
    REAL R2n = 1.0;

    for (INT aK=0 ; aK<INT(mCoeffs.size()) ; aK++)
    {
         R2n *= R2;
         aRes += mCoeffs[aK] * R2n;
    }
    return aRes;
}

REAL ElDistRadiale_PolynImpair::DerSurRho(REAL R) const
{
    if (R > mRMax)
    {
        ELISE_ASSERT(false,"R>mRMax : ElDistRadiale_PolynImpair::DerSurRho");
    }

    REAL R2 = ElSquare(R);
    REAL aRes = 0.0;
    REAL R2n = 1.0;

    for (INT aK=0 ; aK<INT(mCoeffs.size()) ; aK++)
    {
         aRes += mCoeffs[aK] * R2n * 2  * (aK+1);
         R2n *= R2;
    }
    return aRes;
}
INT  ElDistRadiale_PolynImpair::NbCoeff() const
{
   return (INT) mCoeffs.size();
}

INT  ElDistRadiale_PolynImpair::NbCoeffNN() const
{
   int aK = NbCoeff();
   while ((aK>0 ) && (Coeff(aK-1) ==0))
        aK--;
   return aK;
}



void ElDistRadiale_PolynImpair::VerifCoeff(INT aK) const
{
   ELISE_ASSERT
   (
       (aK>=0) && (aK<INT(mCoeffs.size())),
       "Bad Indexe in ElDistRadiale_PolynImpair"
   );
}

REAL &  ElDistRadiale_PolynImpair::Coeff(INT aK)
{
   VerifCoeff(aK);
   return  mCoeffs[aK];
}
REAL   ElDistRadiale_PolynImpair::Coeff(INT aK) const
{
   VerifCoeff(aK);
   return  mCoeffs[aK];
}
REAL   ElDistRadiale_PolynImpair::CoeffGen(INT aK) const
{
  if (aK>=INT(mCoeffs.size()))
     return  0.0;
   return Coeff(aK);
}



ElDistRadiale_PolynImpair   ElDistRadiale_PolynImpair::DistRadialeInverse
                           (REAL RhoApp,INT DeltaDeg)
{
    INT aDeg = DeltaDeg + NbCoeff();
    L2SysSurResol aSys(aDeg);
    aSys.SetPhaseEquation(0);

    static std::vector<REAL> Coeffs;

    INT NbPts =  20 * aDeg;
    for (INT aK = -NbPts ; aK <= NbPts ; aK++)
    {
         Coeffs.clear();
     REAL R = aK *  RhoApp/NbPts;
     REAL R2 = ElSquare(R);

     REAL ImR = R * (1+DistDirecteR2NoSeuil(R2));
     REAL Pds = ElAbs(R);

     REAL ImR2 = ImR*ImR;
     REAL ImR2N = 1;

     for (INT aK=0 ; aK<aDeg ; aK++)
     {
               ImR2N *= ImR2;
               Coeffs.push_back(ImR2N);
     }
     aSys.GSSR_AddNewEquation(Pds,&(Coeffs[0]),R/ImR-1.0,0);
    }
    Im1D_REAL8  aSol = aSys.Solve(0);

    ElDistRadiale_PolynImpair aRes(mRMax * (1+DistInverse(mRMax)),Centre());
    for (INT aK=0 ; aK<aDeg ; aK++)
        aRes.PushCoeff(aSol.data()[aK]);

    return aRes;
}

ElDistRadiale_PolynImpair
ElDistRadiale_PolynImpair::MapingChScale(REAL aSc) const
{
   ElDistRadiale_PolynImpair aRes(mRMax*aSc,Centre()*aSc);
   REAL S2 = ElSquare(aSc);
   REAL S2N = 1.0;

   for (INT aK =0 ; aK< NbCoeff() ; aK++)
   {
        S2N *= S2;
        aRes.PushCoeff(Coeff(aK)/S2N);
   }
   return aRes;
}


ElDistRadiale_PolynImpair ElDistRadiale_PolynImpair::DistId(REAL aRMax,Pt2dr aCentre,INT aDeg)
{
      ElDistRadiale_PolynImpair aDist(aRMax,aCentre);
      for (INT aK=0 ; aK<aDeg ; aK++)
      {
          aDist.PushCoeff(0);
      }

      return aDist;
}


void ElDistRadiale_PolynImpair::write(ELISE_fp & aFile)
{
   aFile.write(mRMax);
   aFile.write(Centre());
   aFile.write(mCoeffs);
}

ElDistRadiale_PolynImpair ElDistRadiale_PolynImpair::read(ELISE_fp & aFile)
{
     REAL aRmax = aFile.read((REAL *) 0);
     Pt2dr aP = aFile.read((Pt2dr *)0);
     std::vector<REAL>  vC = aFile.read((std::vector<REAL>  *)0);

     ElDistRadiale_PolynImpair aRes(aRmax,aP);
     aRes.PushCoeff(vC);
     return aRes;
}

ElDistRadiale_PolynImpair
   ElDistRadiale_PolynImpair::read(const std::string & aName)
{
    ELISE_fp  aFile(aName.c_str(),ELISE_fp::READ);
    ElDistRadiale_PolynImpair aRes = read(aFile);
    aFile.close();
    return aRes;
}

ElDistRadiale_PolynImpair * ElDistRadiale_PolynImpair::DRADPol(bool strict)
{
  return this;
}


bool   ElDistRadiale_PolynImpair::AcceptScaling() const
{
   return true;
}
bool   ElDistRadiale_PolynImpair::AcceptTranslate() const
{
   return true;
}
void ElDistRadiale_PolynImpair::V_SetScalingTranslate
     (
         const double & aF,
         const Pt2dr & aPP
     )
{
// std::cout << "CENTRE IN : " <<  Centre() << "\n";
   Centre() = (Centre() -aPP) / aF;
// std::cout << "CENTRE OUT : " <<  Centre() << "\n";
   for (int aK=0 ; aK<int(mCoeffs.size()) ; aK++)
       mCoeffs[aK] = mCoeffs[aK] * pow(aF,2*(1+aK));
   SetRMax(mRMax/aF);
}



// Rayon max a l'interieur duquel la fonction de
// distortion est bijective croissante

ElPolynome<REAL> ElDistRadiale_PolynImpair::PolynOfR()
{
    ElPolynome<REAL> aPol((char *)0, 1+2*((int) mCoeffs.size()));
    aPol[1] = 1.0;
    for (INT aK=0 ; aK<INT(mCoeffs.size()) ; aK++)
        aPol[3+2*aK] = mCoeffs[aK];
    return aPol;

}

REAL ElDistRadiale_PolynImpair::RMaxCroissant(REAL aBorne)
{
     ElPolynome<REAL> aPol = PolynOfR();
     Reduce(aPol);
     ElPolynome<REAL> aDer = aPol.deriv();
     std::vector<REAL> aVRroots;

     // for (INT aK=0 ; aK<= aDer.degre() ; aK++)
     //     cout << "Pol " <<  aK << " " << aDer[aK] << "\n";

     RealRootsOfRealPolynome(aVRroots,aDer,1e-7,30);

     for (INT aK=0 ; aK<INT(aVRroots.size()) ; aK++)
        if (aVRroots[aK] > 0)
           ElSetMin(aBorne,aVRroots[aK]);
     return aBorne;
}





cCalibrationInterneRadiale  ElDistRadiale_PolynImpair::ToXmlDradStruct() const
{
    cCalibrationInterneRadiale aCIR = CIR(Centre());

    int aNbcNN = NbCoeffNN();
    for (int aKC=0; aKC<aNbcNN ;  aKC++)
    {
        aCIR.CoeffDist().push_back(Coeff(aKC));
    }

    if (1)
    {
         int aDelta =  NbCoeffNN() - NbCoeff();
         ElDistRadiale_PolynImpair anInv = const_cast<ElDistRadiale_PolynImpair*>(this)->DistRadialeInverse(mRMax ,aDelta+1);
         for (int aKC=0 ; aKC<anInv.NbCoeffNN() ; aKC++)
             aCIR.CoeffDistInv().push_back(anInv.Coeff(aKC));
    }

    return aCIR;
}


cCalibDistortion ElDistRadiale_PolynImpair::ToXmlStruct(const ElCamera * ) const
{
    return FromCIR(ToXmlDradStruct());
}



/*************************************************/
/*                                               */
/*        cDistModStdPhpgr                       */
/*                                               */
/*************************************************/

cDistModStdPhpgr::cDistModStdPhpgr(const ElDistRadiale_PolynImpair & aDist) :
    ElDistRadiale_PolynImpair(aDist),
    mP1 (0),
    mP2 (0),
    mb1 (0),
    mb2 (0)
{
  // SetParamConvInvDiff(20,1e-3);
}

REAL & cDistModStdPhpgr::P1() {return mP1;}
REAL & cDistModStdPhpgr::P2() {return mP2;}
REAL & cDistModStdPhpgr::b1() {return mb1;}
REAL & cDistModStdPhpgr::b2() {return mb2;}
ElDistRadiale_PolynImpair & cDistModStdPhpgr::DRad() {return *this;}
const ElDistRadiale_PolynImpair & cDistModStdPhpgr::DRad() const {return *this;}

Pt2dr cDistModStdPhpgr::Direct(Pt2dr aP) const
{
    Pt2dr aPRad = ElDistRadiale_PolynImpair::Direct(aP);
// std::cout << "PHGR " << aP << euclid(aP-aPRad) << "\n";
    Pt2dr aDp = aP -ElDistRadiale_PolynImpair::Centre();

    REAL aDx = aDp.x;
    REAL aDy = aDp.y;
    REAL aDx2 = ElSquare(aDx);
    REAL aDy2 = ElSquare(aDy);
    REAL aDR2 = aDx2 + aDy2;

    REAL aDCx = (2*aDx2+aDR2)*mP1 + 2*aDx*aDy*mP2 + mb1*aDx + mb2 * aDy;
    REAL aDCy = (2*aDy2+aDR2)*mP2 + 2*aDx*aDy*mP1;

    return aPRad + Pt2dr(aDCx,aDCy);
}

void  cDistModStdPhpgr::Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
{
   DiffByDiffFinies(aMat,aP,1e-2*ElDistRadiale_PolynImpair::RMax());
}

ElDistRadiale_PolynImpair * cDistModStdPhpgr::DRADPol(bool strict)
{
  return  strict ? 0 : this;
}

bool cDistModStdPhpgr::OwnInverse(Pt2dr & p ) const
{
   // BUG : fait une fausse inverse !
   // return ElDistRadiale_PolynImpair::OwnInverse(p);
   return false;
}

Pt2dr cDistModStdPhpgr::GuessInv(const Pt2dr & aP) const
{
   Pt2dr aQ = aP;
   ElDistRadiale_PolynImpair::OwnInverse(aQ);

   return aQ;
}


cCalibrationInternePghrStd cDistModStdPhpgr::ToXmlPhgrStdStruct() const
{
   cCalibrationInternePghrStd aRes;
   aRes.RadialePart() = ElDistRadiale_PolynImpair::ToXmlDradStruct();
   aRes.P1().SetVal(mP1);
   aRes.P2().SetVal(mP2);
   aRes.b1().SetVal(mb1);
   aRes.b2().SetVal(mb2);

   return aRes;
}

cCalibDistortion cDistModStdPhpgr::ToXmlStruct(const ElCamera * ) const
{
   cCalibDistortion aRes = XmlDistNoVal() ; // ElDistortion22_Gen::BasicToXmlStruct();
   aRes.ModPhgrStd().SetVal(ToXmlPhgrStdStruct());

   return aRes;
}

bool cDistModStdPhpgr::AcceptScaling() const
{
    return true;
}

bool cDistModStdPhpgr::AcceptTranslate() const
{
    return true;
}


void cDistModStdPhpgr::V_SetScalingTranslate(const double & aScale,const Pt2dr & aTr)
{
   ElDistRadiale_PolynImpair::V_SetScalingTranslate(aScale,aTr);
   mP1  *=  aScale;
   mP2  *=  aScale;


   // std::cout << "PHGRSTD-ps : " << mP1 << " " << mP2 << " " << mb1  << " " << mb2 << "\n";
}


/*************************************************/
/*                                               */
/*    ElDistRadiale_Pol357                       */
/*                                               */
/*************************************************/

ElDistRadiale_Pol357::ElDistRadiale_Pol357
(
     REAL aRMax,
     Pt2dr Centre,
     REAL  C3,
     REAL  C5,
     REAL  C7
)  :
   ElDistRadiale_PolynImpair (aRMax,Centre)
{
   PushCoeff(C3);
   PushCoeff(C5);
   PushCoeff(C7);
}


/***************************************************************/
/***************************************************************/
/*****       Distortion / Poly degre 2                      ****/
/***************************************************************/
/***************************************************************/


       // PolyDegre2XY

PolyDegre2XY::PolyDegre2XY(REAL a,REAL aX,REAL aY,REAL aXX,REAL aXY,REAL aYY) :
    m    (a),
    mX   (aX),
    mY   (aY),
    mXX  (aXX),
    mXY  (aXY),
    mYY  (aYY)
{
}

REAL PolyDegre2XY::Val(Pt2dr aPt) const
{

    return   m
           + mX  * aPt.x
           + mY  * aPt.y
           + mXX * ElSquare(aPt.x)
           + mXY * aPt.x * aPt.y
           + mYY * ElSquare(aPt.y);
}

Pt2dr PolyDegre2XY::Grad(Pt2dr aPt) const
{
     return Pt2dr
            (
               mX + 2*mXX*aPt.x + mXY * aPt.y ,
               mY + 2*mYY*aPt.y + mXY * aPt.x
            );
}

          // ElDistPolyDegre2

Pt2dr ElDistPolyDegre2::Direct(Pt2dr aPt) const
{
    return Pt2dr(mPolX.Val(aPt),mPolY.Val(aPt));
}

void  ElDistPolyDegre2::Diff(ElMatrix<REAL> & aMat,Pt2dr aPt) const
{
     aMat.set_to_size(2,2);

     SetLig(aMat,0,mPolX.Grad(aPt));
     SetLig(aMat,1,mPolY.Grad(aPt));
}

ElDistPolyDegre2::ElDistPolyDegre2
(
    const PolyDegre2XY & aPolX,
    const PolyDegre2XY & aPolY,
    REAL EpsilonInv
)  :
    mPolX    (aPolX),
    mPolY    (aPolY)//,
    //mEpsilon (EpsilonInv)
{
}

/*********************************************************************/
/*                                                                   */
/*                  ElDistortionPolynomiale                          */
/*                                                                   */
/*********************************************************************/

ElDistortionPolynomiale::ElDistortionPolynomiale
(
               const Polynome2dReal & aDistX,
               const Polynome2dReal & aDistY,
               REAL                   anEpsilonInv
)  :
   mDistX      (aDistX),
   mDistY      (aDistY),
   mEpsilon    (anEpsilonInv)
{
}

ElDistortionPolynomiale::ElDistortionPolynomiale
(
               REAL                   anAmpl,
               REAL                   anEpsilonInv
)  :
   mDistX      (1,anAmpl),
   mDistY      (1,anAmpl),
   mEpsilon    (anEpsilonInv)
{
   mDistX.SetDegre1(0,1,0);
   mDistY.SetDegre1(0,0,1);
}


ElDistortionPolynomiale
   ElDistortionPolynomiale::DistId(int aDegre,double anAmpl)
{
   Polynome2dReal  aPX(aDegre,anAmpl);
   aPX.SetDegre1(0.0,1.0,0.0);

   Polynome2dReal  aPY(aDegre,anAmpl);
   aPY.SetDegre1(0.0,0.0,1.0);

   return ElDistortionPolynomiale(aPX,aPY);
}



void  ElDistortionPolynomiale::Diff(ElMatrix<REAL> & aMat,Pt2dr aPt) const
{
     aMat.set_to_size(2,2);

     SetLig(aMat,0,mDistX.grad(aPt));
     SetLig(aMat,1,mDistY.grad(aPt));
}

Pt2dr ElDistortionPolynomiale::Direct(Pt2dr aPt) const
{
    return Pt2dr(mDistX(aPt),mDistY(aPt));
}



ElDistortionPolynomiale ElDistortionPolynomiale::MapingChScale(REAL aChSacle) const
{
     return ElDistortionPolynomiale
            (
                  mDistX.MapingChScale(aChSacle),
                  mDistY.MapingChScale(aChSacle),
                  mEpsilon * aChSacle
            );
}


const Polynome2dReal & ElDistortionPolynomiale::DistX()  const {return mDistX;}
const Polynome2dReal & ElDistortionPolynomiale::DistY()  const {return mDistY;}

Fonc_Num ElDistortionPolynomiale::FNum() const
{
    return Virgule
           (
              mDistX.FNum(),
              mDistY.FNum()
           );
}

ElDistortion22_Gen  * ElDistortionPolynomiale::D22G_ChScale(REAL aS) const
{
   return new  ElDistortionPolynomiale(MapingChScale(aS));
}


/*********************************************/
/*                                           */
/*         cCS_MapIm2PlanProj                */
/*                                           */
/*********************************************/


cCS_MapIm2PlanProj::cCS_MapIm2PlanProj(CamStenope * pCam) :
   mCam  (*pCam)
{
}

Pt2dr cCS_MapIm2PlanProj::Direct(Pt2dr aP) const
{
       return mCam.F2toPtDirRayonL3(aP);
}

bool cCS_MapIm2PlanProj::OwnInverse(Pt2dr & aP) const
{

  aP = mCam.PtDirRayonL3toF2(aP);
  return true;
}

void  cCS_MapIm2PlanProj::Diff(ElMatrix<REAL> &,Pt2dr) const
{
    ELISE_ASSERT(false,"cCS_MapIm2PlanProj::Diff");
}

/*************************************************/
/*                                               */
/*    cDistPrecondRadial                         */
/*                                               */
/*************************************************/
    // ============  cDistPrecond2SinAtgtS2 ===

cDistPrecond2SinAtgtS2::cDistPrecond2SinAtgtS2
(
    double aFocApriori,
    const Pt2dr & aCentre
) :
  cDistPrecondRadial(aFocApriori,aCentre)
{
}
int cDistPrecond2SinAtgtS2::Mode() const {return  ePCR_2SinAtgtS2;}
double   cDistPrecond2SinAtgtS2::MultDirect(const double &  aV) const { return f2SAtRxS2SRx(aV);}
double   cDistPrecond2SinAtgtS2::MultInverse(const double &  aV) const {return Tg2AsRxS2SRx(aV);}

double   cDistPrecond2SinAtgtS2::DerMultDirect(const double &  aV) const { return Der2SAtRxS2SRx(aV);}

    // ============  cDistPrecondAtgt ===

cDistPrecondAtgt::cDistPrecondAtgt
(
    double aFocApriori,
    const Pt2dr & aCentre
) :
  cDistPrecondRadial(aFocApriori,aCentre)
{
}

int cDistPrecondAtgt::Mode() const {return  ePCR_Atgt;}
double   cDistPrecondAtgt::MultDirect(const double &  aV) const { return AtRxSRx(aV);}
double   cDistPrecondAtgt::MultInverse(const double &  aV) const {return TgRxSRx(aV);}
double   cDistPrecondAtgt::DerMultDirect(const double &  aV) const { return DerAtRxSRx(aV);}


     //=======================

cDistPrecondRadial::cDistPrecondRadial
(
    double aFocApriori,
    const Pt2dr & aCentre
)   :
    mF (aFocApriori),
    mC (aCentre)
{
}


Pt2dr  cDistPrecondRadial::DirectAndDer(Pt2dr aP,Pt2dr & aGradX,Pt2dr & aGradY) const
{
    Pt2dr aPC = (aP-mC);
    Pt2dr aPN = aPC/mF;
    double aX2N = ElSquare(aPN.x);
    double aY2N = ElSquare(aPN.y);
    double aXYN = aPN.x * aPN.y;
    double aR2N = aX2N + aY2N;
    double aMul =  MultDirect(aR2N);
    double aDerMul =  DerMultDirect(aR2N);

     aGradX.x = aMul + 2 * aX2N * aDerMul;
     aGradX.y =  2 * aXYN * aDerMul;

     aGradY.x  =  2 * aXYN * aDerMul;
     aGradY.y  =  aMul + 2 * aY2N * aDerMul;

     return mC + aPC * aMul;
}

Pt2dr cDistPrecondRadial::Direct(Pt2dr aP) const
{
    aP = (aP-mC)/mF;
    // return mC + aP * (mF*AtRxSRx(square_euclid(aP)));
    return mC + aP * (mF*MultDirect(square_euclid(aP)));
}

bool cDistPrecondRadial::OwnInverse(Pt2dr & aP) const
{
    aP = (aP-mC)/mF;
    // aP =  mC + aP * (mF*TgRxSRx(square_euclid(aP)));
    aP =  mC + aP * (mF*MultInverse(square_euclid(aP)));

    return true;
}

cPreCondGrid cDistPrecondRadial::GetAsPreCond() const
{
   cPreCondGrid aRes;

   cPreCondRadial aPCR;
   aPCR.C() = mC;
   aPCR.F() = mF;
   // aPCR.Mode() = ePCR_Atgt;
   aPCR.Mode() = eTypePreCondRad(Mode());

   aRes.PreCondRadial().SetVal(aPCR);

   return aRes;
}

ElDistortion22_Gen * ElDistortion22_Gen::AllocPreC
                     (
                        const NS_ParamChantierPhotogram::cPreCondGrid& aPCG
                     )
{
  if (aPCG.PreCondRadial().IsInit())
  {
     const cPreCondRadial & aPCGR = aPCG.PreCondRadial().Val();
     switch (aPCGR.Mode())
     {
          case ePCR_Atgt :
             return new cDistPrecondAtgt(aPCGR.F(),aPCGR.C());
          break;
          case ePCR_2SinAtgtS2 :
             return new cDistPrecond2SinAtgtS2(aPCGR.F(),aPCGR.C());
          break;
          case  ePCR_Stereographik :
             return new  cDistPrecondSterographique(aPCGR.F(),aPCGR.C());
          break;
          default : break;
     }
     ELISE_ASSERT
     (
        false,
        "Bad value  for eTypePreCondRad"
     );
  }

  ELISE_ASSERT
  (
     false,
     "ElDistortion22_Gen::AllocPreC"
  );
  return 0;
}


/*************************************************/
/*                                               */
/*    cDistCamStenopeGrid                        */
/*                                               */
/*************************************************/

/*
...  C2                                      F2
          PreCond             cDistCorrPC
   Monde  ----------->     ---------------->  Cam

             -------------------------->

               DistDirecte


*/




class cDistCorrPC  : public ElDistortion22_Gen
{
    public :
        cDistCorrPC
        (
             double               aRayInv,
             ElDistortion22_Gen * aPreCond,
             const CamStenope & aCS
        ) :
           mRayInv  (aRayInv),
           mPreCond (aPreCond),
           mCS      (aCS),
           mMil     (aCS.NormC2M(aCS.Sz()/2.0))
        {

// std::cout << "RIririririririri  " << aRayInv << "\n";
// getchar();
            SetParamConvInvDiff(10000000,1e-5);
        }

         Pt2dr Direct(Pt2dr aP) const;
         // Pt2dr GuessInv(const Pt2dr & aP) const {return aP;}
         Pt2dr GuessInv(const Pt2dr & aP) const {return mCS.NormC2M(aP);}

    private :
       void  Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
       {
           DiffByDiffFinies(aMat,aP,Pt2dr(1e-3,1e-3));
       }

       Pt2dr DirSsRay(Pt2dr aP) const;

       double               mRayInv;
       ElDistortion22_Gen * mPreCond;
       const CamStenope &   mCS;
       Pt2dr                mMil;
};


Pt2dr cDistCorrPC::Direct(Pt2dr aP) const
{
// std::cout << "PCD " << mRayInv << " " << mPreCond<< "\n";
   if ((mRayInv>0) && mPreCond)
   {
      Pt2dr aMP = aP-mMil;
      // double aR = euclid(DirSsRay(aP)-mMil);
      double aR = euclid(aMP);
// Pt2dr cDistCorrPC::DirSsRay(Pt2dr aP) const
 // std::cout << "R= " << aR <<  " " << euclid(DirSsRay(aP)-mMil) << "\n"; // getchar();
      if (aR>mRayInv)
      {
         double aEps=1e-3;
         //PC = projection sur le cercle
         Pt2dr aPC = mMil + vunit(aMP) * mRayInv;
         Pt2dr aDPC = DirSsRay(aPC);
         Pt2dr aDx =( DirSsRay(aPC+Pt2dr(aEps,0))-aDPC) /aEps;
         Pt2dr aDy =( DirSsRay(aPC+Pt2dr(0,aEps))-aDPC) /aEps;

         Pt2dr aPP = aP-aPC;

         return aDPC + aDx*aPP.x + aDy*aPP.y;
      }
   }
   return DirSsRay(aP);
}

Pt2dr cDistCorrPC::DirSsRay(Pt2dr aP) const
{
   ElDistortion22_Gen & aDCS = const_cast<CamStenope &>(mCS).Get_dist();
   bool aVUPC = aDCS.Dist22Gen_UsePreConditionner();
   aDCS.SetDist22Gen_UsePreConditionner(false);
   aP = aDCS.Direct(aP);
   aDCS.SetDist22Gen_UsePreConditionner(aVUPC);

   return aP;
/*
    if (mPreCond)
       aP = mPreCond->Inverse(aP);

    return mCS.DistDirecte(aP);
*/
}



    // ===================================================

cDistCamStenopeGrid::cDistCamStenopeGrid
(
        ElDistortion22_Gen * aPreC,
        cDbleGrid *          aGrid
)  :
   mPreC (aPreC),
   mGrid (aGrid)
{
}

Pt2dr cDistCamStenopeGrid::DirectAndDer(Pt2dr aP,Pt2dr & aGradX,Pt2dr & aGradY)  const
{
    Pt2dr aGPCx(1,0);
    Pt2dr aGPCy(0,1);

    if (mPreC)
       aP = mPreC->DirectAndDer(aP,aGPCx,aGPCy);

    Pt2dr aGGx,aGGy;
    aP = mGrid->ValueAndDer(aP,aGGx,aGGy);


    aGradX =  aGPCx * aGGx.x + aGPCy * aGGx.y;
    aGradY =  aGPCx * aGGy.x + aGPCy * aGGy.y;

   return aP;

}

Pt2dr cDistCamStenopeGrid::Direct(Pt2dr aP) const
{
   if (mPreC)
     aP = mPreC->Direct(aP);

   return mGrid->Direct(aP);
}

bool cDistCamStenopeGrid::OwnInverse(Pt2dr & aP) const
{
   aP = mGrid->Inverse(aP);

   if (mPreC)
     aP = mPreC->Inverse(aP);

   return true;
}


cDistCamStenopeGrid * cDistCamStenopeGrid::Alloc
                      (
                          bool P0P1IsBoxDirect,
                          double            aRayInv,
                          const CamStenope & aCS,
                          Pt2dr aStepGr,
                          bool  doDir,
                          bool  doInv
                      )
{
//std::cout << "STTTTeppGrid " << aStepGr << "\n";

    aStepGr = aStepGr * (euclid(aCS.Sz()) /6000.0);

// std::cout <<   aCS.mIntrOrImaM2C (Pt2dr(1,1)) << "\n";
   ElCamera::tOrIntIma aScaneOri = const_cast<CamStenope &>(aCS).InhibeScaneOri();


   ElDistortion22_Gen * aPreCond = aCS.DistPreCond();
   /// cDistCorrPC aDC(aRayInv,aPreCond,aCS);

   double aRab = euclid(aCS.Sz()) * 0.01;
   Pt2dr aPRab(aRab,aRab);



   cElDistFromCam  aDCS(aCS,true);


   bool aVUPC = aDCS.Dist22Gen_UsePreConditionner();
   bool aSUPCI = aDCS.Dist22Gen_SupressPreCondInInverse();

   aCS.Dist().SetDist22Gen_UsePreConditionner(false);
   aCS.Dist().SetDist22Gen_SupressPreCondInInverse(true);

// std::cout << "xxAAAA " <<  aCS.Sz() <<  " PASS " << (aStepGr/aCS.ScaleCamNorm()) / aCS.ScaleAfnt()<<"\n";
// std::cout << "Zigwy "  << aStepGr << " " << aCS.ScaleCamNorm() << " " << aCS.ScaleAfnt() << "\n";
// getchar();

// NIKRUP
//std::cout << "P0P1 " << Pt2dr(0,0)-aPRab << " " << Pt2dr(aCS.Sz())+aPRab << "\n";
//std::cout << "P0P1NORM " << aCS.NormC2M(Pt2dr(0,0)-aPRab) << " " <<  aCS.NormC2M(Pt2dr(aCS.Sz())+aPRab) << "\n";

//getchar();
//==============

   cDbleGrid * aGrid2 = new cDbleGrid
                           (
                               P0P1IsBoxDirect,
                               true,
                               aCS.NormC2M(Pt2dr(0,0)-aPRab),
                               aCS.NormC2M(Pt2dr(aCS.Sz())+aPRab),
                               (aStepGr/aCS.ScaleCamNorm()) / aCS.ScaleAfnt() ,
                               aDCS,
                               "toto",
                               doDir,
                               doInv
                           );
// std::cout << "wwwBBBB\n";


   aCS.Dist().SetDist22Gen_UsePreConditionner(aVUPC);
   aCS.Dist().SetDist22Gen_SupressPreCondInInverse(aSUPCI);
   const_cast<CamStenope &>(aCS).RestoreScaneOri(aScaneOri);

   cDistCamStenopeGrid* aRes2 =   new cDistCamStenopeGrid(aPreCond,aGrid2);


   while (0)
   {
        std::cout << "ENTER xy " << aVUPC << " " << aSUPCI << " " << aPreCond <<"\n";
        Pt2dr aPIm;
        cin >> aPIm.x >>  aPIm.y;
        Pt2dr aP1 = aRes2->Direct(aPIm);
        Pt2dr aQ1 = aDCS.Direct(aPIm);
        std::cout << "RES " <<  aP1 << " DCS " << aQ1 << "\n";
        std::cout << " CAM " <<  aCS.DistDirecte(aPIm)  << "\n";
        std::cout << "  GRID " <<  aGrid2->Direct(aPIm)  << "\n";

/*
*/
   }




   if ( aRayInv <0)
   {
      double aDMax = 0;
      double aRMax = 1e20;
      for (int aKx=0  ; aKx< aCS.Sz().x ; aKx += 20)
      {
          for (int aKy=0  ; aKy< aCS.Sz().y ; aKy += 20)
          {
            Pt2dr aPIm (aKx,aKy);
            double aRay = euclid(aPIm,Pt2dr(aCS.Sz())/2.0);
            if (aRay < aRayInv)
            {
                Pt2dr aP = aCS.NormC2M(aPIm);
                Pt2dr aQ2 = aRes2->Inverse(aP);

                Pt2dr aR0 = aCS.DistDirecte(aQ2);
                Pt2dr aR2 = aRes2->Direct(aQ2);

                double aDist = euclid(aR0,aR2);
                aDMax = ElMax(aDMax,aDist);
                if (aDist<1e-2)
                {
                    aRMax  = ElMax(aRMax,aRay);
                }
            }
          }
      }
     // std::cout << "DD-Ggridswxxxxxxxx : " << aDMax << " " << aRMax << " \n";
      if (aDMax>0.1)
      {
         std::cout << "DD-Ggridswxxxxxxxx : " << aDMax << " \n";
         ELISE_ASSERT(aDMax<0.1,"Bad grid inverse in cDistCamStenopeGrid::Alloc");
      }
   }



   return aRes2;
}




void   cDistCamStenopeGrid::Test(double aRayInv,const CamStenope & aCS,Pt2dr aStepGr)
{
   int   aStepTest=2;
   Pt2di aSzT = aCS.Sz() / aStepTest;




   //   P0P1IsBoxDirect  => On conserve la version 
   cDistCamStenopeGrid * aDist = Alloc(true,aRayInv,aCS,aStepGr);

   Im2D_REAL4 aIm(aSzT.x,aSzT.y,0.0);

   for (int anX = 0; anX<aSzT.x ; anX++)
   {
       std::cout << "COL --------------------: " << anX << "\n";
       for (int anY = 0; anY<aSzT.y ; anY++)
       {
           Pt2dr aPIm0(anX*aStepTest,anY*aStepTest);


//if (Test)  std::cout << "aPIm0 " << aPIm0 << "\n";
           Pt2dr aPMonde = aDist->Inverse(aPIm0);
//if (Test)  std::cout << "PMONDE " << aPMonde << "\n";

           Pt2dr aPIm1 = aDist->Direct(aPMonde);
           Pt2dr aPIm2 = aCS.DistDirecte(aPMonde);

            aIm.SetR(Pt2di(anX,anY),euclid(aPIm1-aPIm2));

//if (Test)            std::cout << aPIm1 << aPIm2  << " " << euclid(aPIm1-aPIm2) << "\n";
//if (Test)            std::cout << "\n";
       }
       // getchar();
   }

   Tiff_Im::Create8BFromFonc("ECART.tif",aSzT,aIm.in()*100);

   if(aDist) delete aDist;
}

NS_ParamChantierPhotogram::cCalibDistortion cDistCamStenopeGrid::ToXmlStruct(const ElCamera * ) const
{
   cCalibrationInterneGrid aGr;
   aGr.Grid() = ToXMLExp(*mGrid);

   if (mPreC)
   {
      aGr.PreCondGrid().SetVal(mPreC->GetAsPreCond());
   }


   cCalibDistortion aRes;
   aRes.ModGrid().SetVal(aGr);
   return aRes;
}

std::string cDistCamStenopeGrid::Type() const
{
   return "DistGrid";
}



#if (0)
#endif


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
