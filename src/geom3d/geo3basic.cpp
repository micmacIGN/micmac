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


Pt3dr norm_or_poly(const ElFifo<Pt3dr> & f,REAL * surf)
{
     Pt3dr res (0.0,0.0,0.0);

     for (INT k=0; k<f.nb(); k++)
         res = res + (f[k]^f[k+1]);

     REAL d = euclid(res);
     if (surf)
     {
        *surf = d;
     }
     else
     {
          ELISE_ASSERT(d!=0,"Null dist in norm_or_poly");
     }     

     if (d) 
        res = res/d;
     return res; 
}


ElSeg3D::ElSeg3D(Pt3dr aP0,Pt3dr aP1) :
    mP0(aP0),
    mP1(aP1),
    mTN (Tgt()/euclid(Tgt()))
{
}

Pt3dr ElSeg3D::Tgt() const
{
    return mP1-mP0;
}

Pt3dr ElSeg3D::TgNormee() const
{
    return mTN;
}

Pt3dr ElSeg3D::ProjOrtho(Pt3dr aPt) const
{
    return PtOfAbsc(scal(mTN,aPt-mP0));
}

Pt3dr ElSeg3D::PtOfAbsc(REAL anAbsc) const
{
    return mP0 +  mTN * anAbsc;
}

REAL ElSeg3D::AbscOfProj(Pt3dr aP) const
{
     return scal(mTN,aP-mP0);
}

REAL   ElSeg3D::DistDoite(Pt3dr aP0) const
{
    return euclid (aP0-ProjOrtho(aP0));
}

void   ElSeg3D::AbscissesPseudoInter
       (
             REAL &anAbsc1,
             REAL & anAbsc2,
             const ElSeg3D & aS2
        )
{
    // On cherche a minimiser 
    // (mP0+anAbsc1*mTN -aS2.mP0-anAbsc2*aS2.mTN)

    Pt3dr  A = mP0-aS2.mP0;
    Pt3dr  u = mTN;
    Pt3dr  v = -aS2.mTN;

    // On cherche a minimiser 
    // (A+anAbsc1*u+anAbsc2*v)


     // On calcule les terme quadratique  en 
     REAL m11 = square_euclid(u);
     REAL  m12 = scal(u,v);
     REAL m22 = square_euclid(v);

     // Et on inverse dans la foulee
     REAL det = m11 * m22 - m12*m12;
     m11 /= det;
     m22 /= det;
     ElSwap(m11,m22);
     m12 /= -det;

     // on calcule les termes lineaires

     REAL l1 = - scal(A,u);
     REAL l2 = - scal(A,v);

     // On multiplie les termes lineaire par la matrice invers
     // des termes quadratiques

    anAbsc1  = m11 * l1 + m12 * l2;
    anAbsc2  = m12 * l1 + m22 * l2;

}

void   ElSeg3D::Projections(Pt3dr & Proj2On1,const ElSeg3D & aS2,Pt3dr & Proj1On2)
{
    REAL a1,a2;
    AbscissesPseudoInter(a1,a2,aS2);
    Proj2On1 = PtOfAbsc(a1);
    Proj1On2 = aS2.PtOfAbsc(a2);
}

Pt3dr ElSeg3D::PseudoInter(const ElSeg3D & aS2)
{
    Pt3dr P1,P2;
    Projections(P1,aS2,P2);
    return (P1+P2) / 2.0;
}

Pt3dr ElSeg3D::Mil() const
{
    return (mP0+mP1)/2.0;
}


Pt3dr ElSeg3D::P0() const
{
    return mP0;
}

Pt3dr ElSeg3D::P1() const
{
    return mP1;
}

double  ElSeg3D::SomDistDroite(const std::vector<Pt3dr> & aV) const
{
    double aRes = 0.0;

    for (int aK=0 ; aK<int(aV.size()) ; aK++)
    {
        aRes = DistDoite(aV[aK]);
    }

    return aRes;
}

ElSeg3D  ElSeg3D::CombinL1(const std::vector<Pt3dr> & aV)
{
    ELISE_ASSERT(aV.size()>=2,"ElSeg3D::CombinL1");
    double aMinDist = 1e30;
    int aK1Min = -1 ;
    int aK2Min = -1 ;
    Pt3dr aCdg (0,0,0);


    
    for (int aK1=0 ; aK1<int(aV.size()) ; aK1++)
    {
        Pt3dr aP1 = aV[aK1];
        aCdg = aCdg + aP1;
        for (int aK2=aK1+1 ; aK2<int(aV.size()) ; aK2++)
        {
            Pt3dr aP2 = aV[aK2];
            ElSeg3D aSeg(aP1,aP2);
            double aD = aSeg.SomDistDroite(aV);
            if (aD<aMinDist)
            {
               aMinDist = aD;
               aK1Min = aK1;
               aK2Min = aK2;
            }
        }
    }
    aCdg = aCdg /(double)(aV.size());
    
    ElSeg3D aSeg(aV[aK1Min],aV[aK2Min]);
    aCdg = aSeg.ProjOrtho(aCdg);
    Pt3dr aTgt = aSeg.TgNormee();

    return ElSeg3D(aCdg,aCdg+aTgt);
    
}

ElSeg3D  ElSeg3D::CreateL2(const std::vector<Pt3dr> & aV)
{
    ElSeg3D aRes(Pt3dr(0,0,0),Pt3dr(1,1,1));
    cElPlan3D aPln(aV,0,&aRes);
    aPln.NoOp();

    return aRes;
}



/***********************************************/
/*                                             */
/*                cElPlan3D                    */
/*                                             */
/***********************************************/

cElPlan3D::cElPlan3D(Pt3dr aP0,Pt3dr aP1,Pt3dr aP2) :
        mNorm (vunit((aP1-aP0)^(aP2-aP0))),
	mScal (-scal(mNorm,aP0)),
	mP0   (aP0),
	mU    (vunit((aP1-aP0)-mNorm*(scal(mNorm,aP1-aP0)))),
	mV    (mNorm^mU)
{
}


extern void RansacTriplet(int & aK1,int & aK2,int & aK3,int  aNb);
void  GetNRandParmiQ(std::vector<int> &aRes,int aN,int aQ)
{
    aRes.clear();
    cRandNParmiQ aSelect(aN,aQ);
    for (int aK=0 ; aK<aQ ; aK++)
        if (aSelect.GetNext())
           aRes.push_back(aK);
}



cElPlan3D RobustePlan3D
          (
             const std::vector<Pt3dr> & aVPts,
             const std::vector<double> * aVPondInit,
             double anEffort,
             double aRatioTirage ,
             int    aNbStepLin
          )
{
    if (aVPts.size() < 3)
    {
        std::cout << "NB POINT GOT : " << aVPts.size() << "\n";
        ELISE_ASSERT(false,"Not enough  points in RobustePlan3D");
    }


    int aNbTirage = sqrt(anEffort*aRatioTirage);
    int aNbPTest = ElMin(int(aVPts.size()),round_up(anEffort/aNbTirage));
    std::vector<int> aIndPts; GetNRandParmiQ(aIndPts, aNbPTest, (int)aVPts.size());


    cElPlan3D  aRes(Pt3dr(0,0,0),Pt3dr(1,0,0),Pt3dr(0,1,0));
    double aBestDist = 1e60;

    double aProp = ElMin(0.9,(aNbPTest-3.1)/aNbPTest); // On enleve au - 3 pts

    for (int aK=0 ; aK<aNbTirage ; aK++)
    {
         int aK1,aK2,aK3;
         RansacTriplet(aK1,aK2,aK3,int(aVPts.size()));
         Pt3dr aP0 = aVPts[aK1];
         Pt3dr aP1 = aVPts[aK2];
         Pt3dr aP2 = aVPts[aK3];
         Pt3dr aNorm = (aP1-aP0)^(aP2-aP0);
         double aLongN = euclid(aNorm);
         if (aLongN >0)
         {
            aNorm = aNorm / aLongN;
            std::vector<double> aVDist;
            for (int aKInd=0 ; aKInd<int(aIndPts.size()) ; aKInd++)
            {
                aVDist.push_back(ElAbs(scal(aNorm,aVPts[aIndPts[aKInd]]-aP0)));
            }
            double aScore = KthValProp(aVDist,aProp);
            if (aScore < aBestDist)
            {
                    aBestDist = aScore;
                    aRes = cElPlan3D(aP0,aP1,aP2);
            }
         // cElPlan3D aPTest(aVPts[aK1],aVPts[aK2],aVPts[aK3]);
         }
    }
    
    double aDistMin=1e-10; // Juste anti plantage au cas ou seulement 3 pt
    double aDistMoy = ElMax(aDistMin,aBestDist);
    for (int aTime=0  ; aTime < aNbStepLin ; aTime++)
    {
        // std::cout << "RRRpplDMoy " << aDistMoy << "\n";
        std::vector<double> aPds;
        Pt3dr aP0 = aRes.P0();
        Pt3dr aNorm  = aRes.Norm();

        std::vector<double> aVPds;
        double aSomDist = 0;
        int    aNbDist = 0;

        double aSeuilDist = 5 * aDistMoy;
        while (aNbDist<3)
        {
            aVPds.clear();
            aSomDist = 0;
            aNbDist = 0;
            for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
            {
                 double aDist = ElAbs(scal(aNorm,aVPts[aKP]-aP0));
                 double aPds = 0;
                 if (aDist < aSeuilDist)
                 {
                     aPds = sqrt(1/(1+ElSquare(aDist/(2*aDistMoy))));
                     aSomDist += aDist;
                     aNbDist++;
                 }
                 if (aVPondInit)
                 {
                      aPds *= (*aVPondInit)[aKP];
                 }
                 aVPds.push_back(aPds);
            }
            aSeuilDist *= 2;
        }
        aRes = cElPlan3D(aVPts,&aVPds);

        
        aDistMoy = ElMax(aDistMin,aSomDist/aNbDist);

    }

    return aRes;
}


cElPlan3D::cElPlan3D
(
    const std::vector<Pt3dr> & aVP,
    const std::vector<double> * aVPds,
    ElSeg3D * aBestSeg
)
{
   if (aVP.size()<3)
   {
      std::cout << "NB PTS IN PLAN "  << aVP.size() << "\n";
      ELISE_ASSERT
      (
         aVP.size()>=3,
         "Pas assez de points dans cElPlan3D::cElPlan3D"
      );
   }
   ELISE_ASSERT
   (
          (aVPds==0) || (aVPds->size() == aVP.size()),
          "Poids incoherent dans cElPlan3D::cElPlan3D"
   );
   Pt3dr aCDG (0,0,0);
   INT aNbP = (INT) aVP.size();
   double aSomPds=0;

 
   for (INT aKP=0 ; aKP<aNbP ; aKP++)
   {
       double  aPds = aVPds ? (*aVPds)[aKP] : 1.0;
       aCDG = aCDG+ aVP[aKP] *aPds;
       aSomPds += aPds;
   }

   aCDG = aCDG /(REAL) aSomPds;

   ElMatrix<REAL> aM(3,3,0.0);
   for (INT aKP=0 ; aKP<aNbP ; aKP++)
   {
      double  aPds = aVPds ? (*aVPds)[aKP] : 1.0;

      Pt3dr aP =  aVP[aKP]-aCDG;

      aM(0,0) += aP.x * aP.x * aPds;
      aM(1,0) += aP.y * aP.x * aPds;
      aM(2,0) += aP.z * aP.x * aPds;

      aM(0,1) += aP.x * aP.y * aPds;
      aM(1,1) += aP.y * aP.y * aPds;
      aM(2,1) += aP.z * aP.y * aPds;

      aM(0,2) += aP.x * aP.z * aPds;
      aM(1,2) += aP.y * aP.z * aPds;
      aM(2,2) += aP.z * aP.z * aPds;
   }

   ElMatrix<REAL> aValP(1,3);
   ElMatrix<REAL> aVecP(3,3);
   std::vector<int> VPS = jacobi(aM,aValP,aVecP);

   aVecP.GetCol(VPS[0],mNorm);
   mScal = -scal(mNorm,aCDG);
   mP0 = aCDG;
   aVecP.GetCol(VPS[1],mU);
   mV = mNorm ^ mU;

   if (aBestSeg)
   {
       *aBestSeg = ElSeg3D(aCDG,aCDG+mV);
   }
}

void cElPlan3D::NoOp()
{
}


// La plan etant "presque" connu  (au moins pas n'importe quoi)
//  on resoud l'equation lineaire
//
//     Z = aX + bY + C
//

/*
void cElPlan3D::L1Ameliore(const std::vector<Pt3dr> & aVP,int aNbMax)
{
   if (aNbMax <=0) 
      aNbMax= aVP.size();
   ElRotation3D  aRE2P = CoordPlan2Euclid().Inv();
}
*/

ElRotation3D cElPlan3D::CoordPlan2Euclid()
{
    return ElRotation3D
           (
              mP0,
              MatFromCol(mU,mV,mNorm),
              true
           );
}



REAL cElPlan3D::ZOfXY(Pt2dr aP) const
{
    return  (-mScal-mNorm.x*aP.x-mNorm.y*aP.y) / mNorm.z;
}

Pt3dr cElPlan3D:: AddZ(Pt2dr aP) const
{
   return Pt3dr(aP.x,aP.y,ZOfXY(aP));
}

Pt3dr cElPlan3D::Inter(const ElSeg3D & aSeg,bool *OK) const
{
    Pt3dr aTgt = aSeg.Tgt();
    double aDenom = scal(aTgt,mNorm);
    if (aDenom==0)
    {
        ELISE_ASSERT(OK!=0,"cElPlan3D::Inter");
	*OK= false;
	return Pt3dr(0,0,0);
    }

    if (OK) 
       *OK= true;

    double aLamda = (mScal + scal(mNorm,aSeg.P0())) / aDenom;
    return aSeg.P0() - (aSeg.Tgt()*aLamda);
}

Pt3dr cElPlan3D::Inter(const cElPlan3D& aP1,const cElPlan3D & aP2,bool  & Ok) const
{
    ElMatrix<REAL> aM(3,3);
    SetLig(aM,0,mNorm);
    SetLig(aM,1,aP1.mNorm);
    SetLig(aM,2,aP2.mNorm);

    Ok = self_gaussj_svp(aM);
    return aM * Pt3dr(-mScal,-aP1.mScal,-aP2.mScal);
}
 

ElSeg3D cElPlan3D::Inter(const cElPlan3D& aP1,bool & Ok) const
{
    Pt3dr aTgt = mNorm^aP1.mNorm;
    REAL aD = euclid(aTgt);
    if (aD==0.0)
    {
        Ok= false;
	return ElSeg3D(Pt3dr(0,0,0),Pt3dr(1,0,0));
    }
    aTgt = aTgt / aD;
    ElMatrix<REAL> aM(3,3);
    SetLig(aM,0,mNorm);
    SetLig(aM,1,aP1.mNorm);
    SetLig(aM,2,aTgt);
    Ok = self_gaussj_svp(aM);

    Pt3dr aP0 = aM * Pt3dr(-mScal,-aP1.mScal,0);

    return ElSeg3D(aP0,aP0+aTgt);
}
         
const Pt3dr & cElPlan3D::Norm() const { return mNorm; }
const Pt3dr & cElPlan3D::P0() const { return mP0; }
const Pt3dr & cElPlan3D::U() const { return mU; }
const Pt3dr & cElPlan3D::V() const { return mV; }

Pt3dr cElPlan3D::Proj(const Pt3dr & aP) const
{
    return aP - mNorm *(scal((aP-mP0),mNorm));
}


void cElPlan3D::Revert() 
{
   mNorm = - mNorm;
   mScal = - mScal;
   mV    = - mV;
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
