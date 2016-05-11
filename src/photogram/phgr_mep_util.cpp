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


/*
  On ne gagne rien !!!
Time Euclid 1.42467
Time QEuclid 2.75878
Time 48Euclid 1.47062
*/
#define NBTabQE  100

double QuickDistEuclid(double anX,double anY)
{
    static double Tab[NBTabQE+2];
    static bool  First = true;
    if (First)
    {
       First = false;
       for (int aK=0 ; aK<= (NBTabQE+1) ; aK++)
       {
           Tab[aK] = euclid(Pt2dr(1.0,aK/double(NBTabQE)));
       }
    }



    anX = ElAbs(anX);
    anY = ElAbs(anY);
    if (anX < anY) ElSwap(anX,anY);

    if (anX==0) return 0;
    double aRatio = (anY/anX) * NBTabQE ;

    int anInd = round_down(aRatio);
    double aPds1 = aRatio - anInd;

    return anX * ((1-aPds1) * Tab[anInd] + aPds1 * Tab[anInd+1]);
}

void TestQE()
{
   std::vector<Pt2dr> aVP;
   for (int aK=0 ; aK< 20 ; aK++)
   {
       Pt2dr aP(NRrandC()*10,NRrandC()*10);
       std::cout << "DIST : " << euclid(aP) << " " << QuickDistEuclid(aP.x,aP.y) << " " << dist48_euclid(aP) << "\n";
       aVP.push_back(aP);
   }

   ElTimer aChrono;
   int aNb=1e7;
   double aS1=0;
   for (int aK=0 ; aK<aNb ; aK++)
       for (int aP=0 ; aP<int(aVP.size()) ; aP++)
          aS1+=euclid(aVP[aP]);
   std::cout << "Time Euclid " << aChrono.uval() << "\n";

   double aS2=0;
   ElTimer aChrono2;
   for (int aK=0 ; aK<aNb ; aK++)
       for (int aP=0 ; aP<int(aVP.size()) ; aP++)
          aS2+=QuickDistEuclid(aVP[aP].x,aVP[aP].y);
   std::cout << "Time QEuclid " << aChrono2.uval() << "\n";

   ElTimer aChrono3;
   double aS3=0;
   for (int aK=0 ; aK<aNb ; aK++)
       for (int aP=0 ; aP<int(aVP.size()) ; aP++)
          aS3+=dist48_euclid(aVP[aP]);
   std::cout << "Time 48Euclid " << aChrono3.uval()  << "\n";
   std::cout << "Soms " << aS1 << " " << aS2 << " " << aS3  << "\n";

}


double NewExactCostMEP(Pt3dr &  anI1,const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) ;


Pt3dr InterSeg(const Pt3dr & aP0,const Pt3dr & aP1,const Pt3dr & aQ0,const Pt3dr & aQ1,bool & Ok,double * aSqD)
{
/*
TIMME :
     3.03199   ElSeg3D::L2InterFaisceaux
     0.473224   InterSeg(const Pt3dr & aP0,...
     1.23799    InterSeg(const std::vector<Pt3r>

*/

     // (P0 + p (P1-P0) ) - (Q0+ q(Q1-Q0)) .  (P1-P0) = 0
     // (P0 + p (P1-P0) ) - (Q0+ q(Q1-Q0)) .  (Q1-Q0) = 0

      Pt3dr aP01 = aP1 -aP0;
      Pt3dr aQ01 = aQ1 -aQ0;
      // Normale

      Pt3dr aPQ = aQ0 - aP0;

      //    a p + b q = X
      //    c p + d q = Y

      double a = scal(aP01,aP01);
      double b = -scal(aQ01,aP01);
      double c = -b;
      double d = -scal(aQ01,aQ01);
      double X = scal(aPQ,aP01);
      double Y = scal(aPQ,aQ01);


     //  p      (d  -b)  X
     //  q  =   (-c  a)  Y


      double aDet = a*d - b*c;

      if (aDet==0)
      {
          Ok= false;
          return Pt3dr(0,0,0);
      }
      Ok = true;


      double p = ( d * X - b * Y ) / aDet;
      double q = (-c * X + a * Y ) / aDet;

      Pt3dr aPInt = aP0 + aP01 * p;
      Pt3dr aQInt =  aQ0 +aQ01 * q;

      if (aSqD) *aSqD = square_euclid(aPInt-aQInt);

      return (aPInt + aQInt) / 2.0;

}

void  CoordInterSeg(const Pt3dr & aP0,const Pt3dr & aP1,const Pt3dr & aQ0,const Pt3dr & aQ1,bool & Ok,double &p , double & q)
{
     // (P0 + p (P1-P0) ) - (Q0+ q(Q1-Q0)) .  (P1-P0) = 0
     // (P0 + p (P1-P0) ) - (Q0+ q(Q1-Q0)) .  (Q1-Q0) = 0
      Pt3dr aP01 = aP1 -aP0;
      Pt3dr aQ01 = aQ1 -aQ0;
      // Normale

      Pt3dr aPQ = aQ0 - aP0;

      //    a p + b q = X
      //    c p + d q = Y

      double a = scal(aP01,aP01);
      double b = -scal(aQ01,aP01);
      double c = -b;
      double d = -scal(aQ01,aQ01);
      double X = scal(aPQ,aP01);
      double Y = scal(aPQ,aQ01);


     //  p      (d  -b)  X
     //  q  =   (-c  a)  Y


      double aDet = a*d - b*c;

      if (aDet==0)
      {
          Ok= false;
          return ;
      }
      Ok = true;


      p = ( d * X - b * Y ) / aDet;
      q = (-c * X + a * Y ) / aDet;
}







Pt3dr InterSeg(const ElRotation3D & aR2to1 ,const Pt3dr & aQ1,const Pt3dr & aQ2,bool & Ok,double * aSquareD)
{
    Pt3dr aBase = aR2to1.tr();

    return InterSeg(Pt3dr(0,0,0),aQ1,aBase,aBase+ aR2to1.Mat()*aQ2,Ok,aSquareD);
}

Pt3dr InterSeg(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,bool & Ok,double * aSquareD)
{
    return InterSeg(aR2to1,PZ1(aP1),PZ1(aP2),Ok,aSquareD);
}

/*
      droite D : P U   ,  soit V W tq (UVW) soit une BON
      d^2 (D,Q) =  [(Q-P) .V] ^2 +  [(Q-P) .W] ^2
                = t(Q-P) [ U tU + W tW] (Q-P)
                =   t(Q -P) [ I - tU U]  (Q-P)

     Maintenta Uk Vk Wk ....

      Mk = I - tUk Uk / ||Uk|| ^2

      Som(Mk) Q = Som ( Mk Pk)

*/

Pt3dr InterSeg(const std::vector<Pt3dr> & aVP0, const std::vector<Pt3dr> & aVP1,bool & Ok)
{
    if (aVP0.size()==2)
       return InterSeg(aVP0[0],aVP1[0],aVP0[1],aVP1[1],Ok,0);
    Ok = true ;  // FAUX => A CHANGER !!!!! 
    static Im2D_REAL8 aImMat(3,3);
    static double ** aDM = aImMat.data();
    static double & aM00 = aDM[0][0];
    static double & aM11 = aDM[1][1];
    static double & aM22 = aDM[2][2];
    static double & aM10 = aDM[0][1];
    static double & aM20 = aDM[0][2];
    static double & aM21 = aDM[1][2];
    aM00  = aM11 = aM22 =  aM10 = aM20 = aM21 = 0;

    static double aV[3];
    aV[0] = aV[1] = aV[2]=0;

    int aNb = (int)aVP0.size();

    for (int aK=0 ; aK<aNb  ; aK++)
    {
        const Pt3dr & aP0 = aVP0[aK];
        Pt3dr aU = (aVP1[aK]-aP0);
        double aX2 = ElSquare(aU.x);
        double aY2 = ElSquare(aU.y);
        double aZ2 = ElSquare(aU.z);
        double aU2 = aX2 + aY2 + aZ2;

        double mxx = 1 -  aX2 / aU2;
        double myy = 1 -  aY2 / aU2;
        double mzz = 1 -  aZ2 / aU2;
        double mxy = - (aU.x * aU.y) / aU2;
        double mxz = - (aU.x * aU.z) / aU2;
        double myz = - (aU.y * aU.z) / aU2;

        aM00 += mxx;
        aM11 += myy;
        aM22 += mzz;
        aM10 += mxy;
        aM20 += mxz;
        aM21 += myz;

        aV[0] +=  mxx * aP0.x + mxy  * aP0.y + mxz * aP0.z;
        aV[1] +=  mxy * aP0.x + myy  * aP0.y + myz * aP0.z;
        aV[2] +=  mxz * aP0.x + myz  * aP0.y + mzz * aP0.z;
    }
    cMSymCoffact3x3<double>  aMCF(aDM);

    Pt3dr aPRes =  aMCF.CoffVecInv(aV);
    return aPRes;
}

Pt3dr InterSeg(const std::vector<ElSeg3D> & aVS,bool & Ok)
{
     std::vector<Pt3dr>  aVP0,aVP1;
     for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
     {
         aVP0.push_back(aVS[aKS].P0());
         aVP1.push_back(aVS[aKS].P1());
     }
     return InterSeg(aVP0,aVP1,Ok);
}

Pt3dr InterSeg(const std::vector<ElRotation3D> & aVR,const std::vector<Pt2dr> & aVP,bool & Ok,double * aResidu)
{
     std::vector<Pt3dr>  aVP0,aVP1;
     ELISE_ASSERT(aVR.size()==aVP.size(),"InterSeg Rot");
     for (int aK=0 ; aK<int(aVR.size()) ; aK++)
     {
         const ElRotation3D & aRot = aVR[aK];
         Pt2dr aPIm = aVP[aK]; 

         Pt3dr aC = aRot.tr();
         Pt3dr aDir  = vunit(aRot.Mat() * Pt3dr(aPIm.x,aPIm.y,1.0));

         aVP0.push_back(aC);
         aVP1.push_back(aC+aDir);
     }

     Pt3dr aPRes = InterSeg(aVP0,aVP1,Ok);

     if (aResidu)
     {
         *aResidu = 0;
         for (int aK=0 ; aK<int(aVR.size()) ; aK++)
         {
             Pt3dr aDir1 = vunit(aPRes-aVP0[aK]);
             Pt3dr aDir2 = vunit(aVP1[aK]-aVP0[aK]);
             *aResidu += euclid(aDir1-aDir2);
         }
         *aResidu /= aVR.size();
     }

     return aPRes;
}
/*
*/

double QualInterSeg(const std::vector<ElRotation3D> & aVR,const tMultiplePF & aVPMul)
{
   int aNbP = aVPMul[0]->size();
   int aNbC = aVR.size();
   ELISE_ASSERT(aNbC == int(aVPMul.size()),"QualInterSeg");

   std::vector<double> aVRes;
   for (int aKP=0 ; aKP<aNbP ; aKP++)
   {
       std::vector<Pt2dr> aVP;
       for (int aKC=0; aKC<aNbC ; aKC++)
       {
           aVP.push_back(ToPt2dr( (*(aVPMul[aKC]))[aKP]));
       }
       double aResidu;
       bool Ok;
       InterSeg(aVR,aVP,Ok,&aResidu);
       aVRes.push_back(aResidu);
   }

   double aResStd = KthVal(aVRes,aNbP*0.75) * 0.666;


   double aSomRes = 0;
   double aSomPds = 0;
   for (int aKP=0 ; aKP<aNbP ; aKP++)
   {
        double aRes = aVRes[aKP];
        double aPds = 1.0 / (1 + ElSquare(aRes/aResStd));
        aSomRes += aRes * aPds;
        aSomPds += aPds;
   }
   return aSomRes / aSomPds;
}


/************************************************************/
/*                                                          */
/*           "Mini"-Utilitaires                             */
/*                                                          */
/************************************************************/

void InitPackME
     (
          std::vector<Pt2dr> & aVP1,
          std::vector<Pt2dr>  &aVP2,
          std::vector<double>  &aVPds,
          const  ElPackHomologue & aPack
     )
{
   for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
   {
      aVP1.push_back(itP->P1());
      aVP2.push_back(itP->P2());
      aVPds.push_back(itP->Pds());
   }
}

//  Formule exacte et programmation simple et claire pour bench, c'est l'angle

// tr + R * M  , tR * (M-tr)

  //   Mesure d'erreur des intersections de faisceaux, "nouvelle" normalisation : c'est la  moyenne des distance de reproj en pixel
  //  C2 => C1 pour tous
  //
  //    ProjCostMEP         => reprojection en pixel
  //    DistDroiteCostMEP   => calcul des intersectio et angles en 3D
  //    PVCostMEP           => calcul par la formule du produit vectoriel
  //    LinearCostMEP       => calcul par le determinant, la moins precise mais la plus rapide

double ProjCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
   Pt3dr aQ1 = Pt3dr(aP1.x,aP1.y,1.0);
   Pt3dr aQ2 = aR2to1.Mat() * Pt3dr(aP2.x,aP2.y,1.0);
   Pt3dr aBase  = aR2to1.tr();
   bool Ok;
   Pt3dr anI1 = InterSeg(Pt3dr(0,0,0),aQ1,aBase,aBase+aQ2,Ok);
   if (! Ok) return aTetaMax;

   Pt3dr anI2 =  aR2to1.ImRecAff(anI1);

   Pt2dr anEc1  = aP1 - ProjStenope(anI1);
   Pt2dr anEc2  = aP2 - ProjStenope(anI2);

   return  GenCoutAttenueTetaMax((euclid(anEc1) + euclid(anEc2)) / 2.0  ,aTetaMax);
}

double QuickD48EProjCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
   Pt3dr aQ1 = Pt3dr(aP1.x,aP1.y,1.0);
   Pt3dr aQ2 = aR2to1.Mat() * Pt3dr(aP2.x,aP2.y,1.0);
   Pt3dr aBase  = aR2to1.tr();
   bool Ok;
   Pt3dr anI1 = InterSeg(Pt3dr(0,0,0),aQ1,aBase,aBase+aQ2,Ok);
   if (!Ok) return aTetaMax;

   Pt3dr anI2 =  aR2to1.ImRecAff(anI1);

   Pt2dr anEc1  = aP1 - ProjStenope(anI1);
   Pt2dr anEc2  = aP2 - ProjStenope(anI2);

   return  GenCoutAttenueTetaMax((dist48_euclid(anEc1) + dist48_euclid(anEc2)) / 2.0  ,aTetaMax);
}




double DistDroiteCostMEP(const ElRotation3D & aR2to1,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
   Pt3dr aQ1 = Pt3dr(aP1.x,aP1.y,1.0);
   Pt3dr aQ2 = aR2to1.Mat() * Pt3dr(aP2.x,aP2.y,1.0);
   Pt3dr aBase  = aR2to1.tr();

   ElSeg3D aS1(Pt3dr(0,0,0),aQ1);
   ElSeg3D aS2(aBase,aBase+aQ2);

   Pt3dr anI = aS1.PseudoInter(aS2);

   double d1 = aS1.DistDoite(anI);
   double d2 = aS2.DistDoite(anI);
   double D1 = euclid(anI);
   double D2 = euclid(aBase-anI);


   double aTeta =  (d1/D1 + d2/D2) / 2;
   return GenCoutAttenueTetaMax(aTeta,aTetaMax);
}

double PVCostMEP(const ElRotation3D & aR2to1,const Pt3dr & aQ1,const Pt3dr & aQ2Init,double aTetaMax)
{
   Pt3dr aQ2 = aR2to1.Mat() *  aQ2Init;
   Pt3dr aBase  = aR2to1.tr();

   Pt3dr aQ1vQ2vB = vunit(aQ1 ^ aQ2) ^ aBase;

   double aDet = Det(aQ1,aQ2,aBase);

   //   /4.0 : Adatation empirique  aux autres couts
   double aTeta = (ElAbs(aDet/scal(aQ1vQ2vB,aQ1)) +  ElAbs(aDet/scal(aQ1vQ2vB,aQ2))) / 4.0 ;

   return GenCoutAttenueTetaMax(aTeta,aTetaMax);
}

double PVCostMEP(const ElRotation3D & aR2to1,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
   return PVCostMEP(aR2to1,vunit(PZ1(aP1)), vunit(PZ1(aP2)),aTetaMax);
}

double  LinearCostMEP(const ElRotation3D & aR2to1,const Pt3dr & aQ1,const Pt3dr & aQ2,double aTetaMax)
{
      return GenCoutAttenueTetaMax(ElAbs(Det(aQ1,(aR2to1.Mat()*aQ2),aR2to1.tr())),aTetaMax);
}


double  LinearCostMEP(const ElRotation3D & aR2to1,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
    return LinearCostMEP(aR2to1,vunit(PZ1(aP1)),vunit(PZ1(aP2)),aTetaMax);
}

   // =====================================
    // Fonction sur une pack hom
    // =====================================

/*
#define DECLARE_PACK_COST_MEP (FONC_ELEM)\
double totoooooo(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax) \
{\
 return 0.0;\
}


    double aSomPCost = 0;\
    double aSomPds = 0;\
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)\
    {\
         double aPds = itP->Pds();\
         double aCost = FONC_ELEM(aRot,itP->P1(),itP->P2(),aTetaMax);\
         aSomPds += aPds;\
         aSomPCost += aPds * aCost;\
    }\
    return (aSomPCost / aSomPds);\
}
*/


double QuickD48EProjCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = QuickD48EProjCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}

double ProjCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = ProjCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}

double DistDroiteCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = DistDroiteCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}


double PVCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = PVCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}

double LinearCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = LinearCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}

Pt3dr MedianNuage(const ElPackHomologue & aPack,const ElRotation3D & aRot)
{
    std::vector<double>  aVX;
    std::vector<double>  aVY;
    std::vector<double>  aVZ;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
        bool Ok;
        Pt3dr anI = InterSeg(aRot,itP->P1(),itP->P2(),Ok);

        if (Ok)
        {
           aVX.push_back(anI.x);
           aVY.push_back(anI.y);
           aVZ.push_back(anI.z);
        }
    }
    return Pt3dr
           (
                 MedianeSup(aVX),
                 MedianeSup(aVY),
                 MedianeSup(aVZ)
           );
}



#define NBTEstTimeBundle  20

void TestLinariseAngle(const  ElPackHomologue & aPack,const ElRotation3D &aRef,double aFoc)
{
   // TestQE(); getchar();
   {
       double anAtten = -1;
       std::cout << "ProjD48 " << aFoc * QuickD48EProjCostMEP(aPack,aRef,anAtten) << "\n";
       std::cout << "ProEucl " << aFoc * ProjCostMEP(aPack,aRef,anAtten) << "\n";
       std::cout << "DDroite " << aFoc * DistDroiteCostMEP(aPack,aRef,anAtten) << "\n";
       std::cout << "ProVec  " << aFoc * PVCostMEP(aPack,aRef,anAtten) << "\n";
       std::cout << "Det     " << aFoc * LinearCostMEP(aPack,aRef,anAtten) << "\n";
   }
   double aNoise = 0.0;
   ElRotation3D aSolF = ElRotation3D::Id;
   for (int aTime =0 ; aTime < 1 ; aTime++)
   {
       ElMatrix<double> aMP =  ElMatrix<double>::Rotation(aNoise*NRrandC(),aNoise*NRrandC(),aNoise*NRrandC());
       ElRotation3D aRInit(aRef.tr(),aRef.Mat()*aMP,true);

       for (int aKTyB=0 ; aKTyB<5 ; aKTyB++)
       {
           ElRotation3D aRot = aRInit;
           cInterfBundle2Image * aIB = 0;
           if (aKTyB<2)
                aIB = cInterfBundle2Image::LineariseAngle(aPack,aFoc,aKTyB==0);
           else if (aKTyB==2)
                aIB = cInterfBundle2Image::LinearDet(aPack,aFoc);
           else if (aKTyB<5)
                aIB = cInterfBundle2Image::Bundle(aPack,aFoc,aKTyB==3);
           else
           {
               ELISE_ASSERT(false,"TestLinariseAngle");
           }


           double anErr = aIB->ErrInitRobuste(aRot);
           double  anErr0 = anErr;

           ElTimer aChrono;
           for (int aIter=0 ; aIter< NBTEstTimeBundle ; aIter++)
           {
               aRot = aIB->OneIterEq(aRot,anErr);
               // std::cout << "ERR " << aIter << " " << anErr * aFoc << "\n";
               if (aIter==0)
                  anErr0 = anErr;
           }
           std::cout << "ERR = " << anErr0*aFoc << " => " << anErr * aFoc
                     << " L2: " << aRot.Mat().L2(aRef.Mat()) << " Tr: " << euclid(aRot.tr()-vunit(aRef.tr()))
                     << " Time " << aChrono.uval()
                     << " CostLi " <<  aFoc * LinearCostMEP(aPack,aRot,-1)
                     << " CostPV " <<  aFoc * PVCostMEP(aPack,aRot,-1)
                     << " Ty " << aIB->VIB2I_NameType() << "\n";
           aSolF = aRot;
       }
       std::cout << "\n";
   }



   {
       double anAtten = -1;
       std::cout << "Det     " << aFoc * LinearCostMEP(aPack,aRef,anAtten)        << "    " << aFoc * LinearCostMEP(aPack,aSolF,anAtten)         <<"\n";
       std::cout << "ProjD48 " << aFoc * QuickD48EProjCostMEP(aPack,aRef,anAtten) << "    " << aFoc * QuickD48EProjCostMEP(aPack,aSolF,anAtten)  << "\n";
       std::cout << "ProEucl " << aFoc * ProjCostMEP(aPack,aRef,anAtten)          << "    " << aFoc * ProjCostMEP(aPack,aSolF,anAtten)           << "\n";
       std::cout << "DDroite " << aFoc * DistDroiteCostMEP(aPack,aRef,anAtten)    << "    " << aFoc * DistDroiteCostMEP(aPack,aSolF,anAtten)     << "\n";
       std::cout << "ProVec  " << aFoc * PVCostMEP(aPack,aRef,anAtten)            << "    " << aFoc * PVCostMEP(aPack,aSolF,anAtten)             << "\n";
   }
/*
 Resultat non explique , pourquoi sur le Ref PVCostMEP est homogenes aux autre, et pas sur la SolF ??

        Det     0.1467    0.302827
        ProjD48 0.155966    0.153702
        ProEucl 0.154485    0.15228
        DDroite 0.154348    0.152146
        ProVec  0.154348    0.152146

*/

   // double aS
   getchar();

/*
   for (int aK=0 ; aK< 2 ; aK++)
   {
       ElMatrix<double> aMP =  ElMatrix<double>::Rotation(aNoise*NRrandC(),aNoise*NRrandC(),aNoise*NRrandC());
       ElRotation3D aRInit(aRef.tr(),aRef.Mat()*aMP,true);

       ElRotation3D aRot = aRInit;
       cEqBundleBase * aBB = new  cEqBundleBase (false,0,aFoc,aK==0);
       ElTimer aChrono;
       double anErr = aBB->AddPackAndSolve(aPack,aRot,-1);
       for (int aKS=0 ; aKS < NBTEstTimeBundle ; aKS++)
       {
             anErr = aBB->AddPackAndSolve(aPack,aRot,anErr);
             aRot = aBB->CurSol();
       }
       std::cout << "Colin Errreur " << anErr * aFoc << " T " << aChrono.uval() << "\n";
   }
*/
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
