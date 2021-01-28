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

namespace NS_SimulIm
{

/*  
    Classes for trees
    *  A trees is made of tree-balls  
    * a tree-balls can me made of other tree-balls, or of
*/
class cSphere;
class cEllips3Dxy;
class cTreeBall;

// Generate a point randomly distributed in the unit sphere
static Pt3df   GenerateRandomPointInsSphereUnite();

class cSphere
{
   public :
      cSphere(const Pt3df & aP,float aR);
      float  Distance(const cSphere &) const;
      float  Interaction(const cSphere &) const;
      const Pt3df  & C() const {return mC;}

   private :
      Pt3df  mC;
      float  mR;
};


// An ellipsoid where
class cEllips3Dxy
{
   public :
      cEllips3Dxy(const Pt3df & aP,Pt3df aRay);
      float  MulR() const {return mR.Vol();}
      Pt3df  GeneratePointInside() const;
      void GenerateNSphereInside
           (
               std::vector<cSphere>  & aResult, 
               double aPropVol, int aNbPts, int aNbTest
           ) const;
      void GenerateNEllips
           (
                std::vector<cEllips3Dxy>  & aResult,
                int                         aNbPts,
                double                      aDensite,
                int                         aNbTest
           ) const;
      double Ray2NbPts(double aPropVol,int aNbPts) const ;
      double NbPts2Ray(double aPropVol,double aRay ) const ;

   private :
      Pt3df  FromNormalCoord(const Pt3df & aP) const;  /// [-1,1]^3 => Ellips
      Pt3df  ToNormalCoord(const Pt3df & aP) const;  /// [-1,1]^3 => Ellips

      // bool   IsInside(const Pt3df & aP);  /// Standard belonging to object


      float  Interaction(const cEllips3Dxy &) const;
      Pt3df  mC;
      Pt3df  mR;
};




class cTreeBall
{
   public :
      cTreeBall(const cEllips3Dxy & aEllips,std::vector<int>,int aLevel);
   private :
      // if terminal contains leafs, else contains tree balls
      std::list<cTreeBall>      mLSubB;
      std::vector<cSphere>  mVLeafs;
};


/***********************************************************/
/*                                                         */
/*                  ::NS_SimulIm                           */
/*                                                         */
/***********************************************************/

Pt3df   GenerateRandomPointInsSphereUnite()
{
   while (1)
   {
       // Pt3df aRes(NRrandC(),NRrandC(),NRrandC());
       Pt3df aRes = Pt3df::RandC();
       if (euclid(aRes)<=1)
          return aRes;
   }
   ELISE_ASSERT(false,"Sould not be here");
   return Pt3df(0,0,0);
}

/***********************************************************/
/*                                                         */
/*                     cSphere                             */
/*                                                         */
/***********************************************************/

float  cSphere::Distance(const cSphere & aS2) const
{
    return euclid(mC-aS2.mC) / (mR + aS2.mR);
}

float  cSphere::Interaction(const cSphere & aS2) const
{
    double aD = Distance(aS2);
    if (aD> 1) return 0;
    return ElSquare(1-aD);
}

/***********************************************************/
/*                                                         */
/*                     cEllips3Dxy                         */
/*                                                         */
/***********************************************************/

double cEllips3Dxy::Ray2NbPts(double aPropVol,int aNbPts) const 
{
    return pow((MulR()*aPropVol)/aNbPts,1/3.0);
}

double cEllips3Dxy::NbPts2Ray(double aPropVol,double aRay ) const 
{
   return  round_ni((MulR()*aPropVol) / pow(aRay,3.0));
}

void cEllips3Dxy::GenerateNSphereInside
     (
         std::vector<cSphere>  & aResult,
         double aPropVol,
         int aNbPts,
         int aNbTest
      ) const
{
   static double aEpsNorm = 0.1;
   static double aEpsZ    = 0.1;

   aResult.clear();
   double aRay = Ray2NbPts(aPropVol,aNbTest);

   for (int aKP = 0 ; aKP<aNbPts ; aKP++)
   {
       // Look for point that maximize distance to all selected in aResult
       double aScoreMax = -1;
       Pt3df  aPMaxScore(0,0,0);

       // Make several test on a random init
       for (int aKTest=0 ; aKTest<aNbTest; aKTest++)
       {
            Pt3df  aPTest= GeneratePointInside();
            // Compute de min distance
            float aDMin = 1e30; 
            for (int aKP=0 ; aKP<int(aResult.size()) ; aKP++)
            {
                ElSetMin(aDMin, square_euclid(aPTest,aResult[aKP].C()));
            }
            double aScore = aDMin;
            Pt3df aPNorm =ToNormalCoord(aPTest);
            // Add a penalization favorizing point at high distance
            aScore *=  (aEpsNorm + euclid(aPNorm)) ;
            // Add a penalization favorizing point at high altitude
            aScore *=  (aEpsZ + (1+aPNorm.z));

            if (aScoreMax<aScore)
            {
               aScoreMax = aScore;
               aPMaxScore = aPTest;
            }
       }
       aResult.push_back(cSphere(aPMaxScore,aRay));
   }
}


/*
void cEllips3Dxy::GenerateNEllips
     (
         std::vector<cEllips3Dxy>  & aResult,
         int                         aNbPts,
         double                      aDensite,
         int                         aNbTest
      ) const
{
   static const float aEpsInterv = 1e-2; // To avoid empty volume

   aResult.clear();

   float aVolTotal  =  MulR() ;
   float aVolTarget =  aVolTotal * aDensite;
   float aVolAvg    =  aVolTarget / aNbPts; ///< Average volume in each test
   float aVolDone   =  0.0;  ///< Volume filled untill now

   /// While the targeted volume is not reached
   while (aVolDone < aVolTarget)
   {
       // Generate a random volume
       float aVol = aVolAvg * NRrandInterv(aEpsInterv,2-aEpsInterv);
       Pt3df aRay =  Pt3df::Rand3().PVolTarget(aVol);

       double aDMax = -1;
       Pt3df  aPMaxD(0,0,0);

       // Make several test on a random init
       for (int aKTest=0 ; aKTest<aNbTest; aKTest++)
       {
            Pt3df  aPTest= GeneratePointInside();
            // Compute de min distance
            float aDMin = 1e30; 
            cEllips3Dxy anE(aPTest,aRay);
            for (int aKP=0 ; aKP<int(aResult.size()) ; aKP++)
            {
                ElSetMin(aDMin, square_euclid(aPTest,aResult[aKP].C()));
            }
            if (aDMax<aDMin)
            {
               aDMax = aDMin;
               aPMaxD = aPTest;
            }
       }
  
   }
      
}
*/


/*
*/
/*
*/


float  cEllips3Dxy::Interaction(const cEllips3Dxy & anE2) const
{

    // Difference of centers, normalised par by sum of ray
    Pt3df aDif  = (mC-anE2.mC).dcbyc(mR + anE2.mR);
    double aD = euclid(aDif);
   
    if (aD> 1) return 0;
    return ElSquare(1-aD);
}
Pt3df  cEllips3Dxy::GeneratePointInside() const
{
    return FromNormalCoord(GenerateRandomPointInsSphereUnite());
}

Pt3df  cEllips3Dxy::FromNormalCoord(const Pt3df & aP) const
{
   return mC + aP.mcbyc(mR); 
}

Pt3df  cEllips3Dxy::ToNormalCoord(const Pt3df & aP) const
{
   return (aP-mC).dcbyc(mR);
}

};

/***********************************************************/
/*                                                         */
/*                  ::                                     */
/*                                                         */
/***********************************************************/

using namespace NS_SimulIm;

int  CPP_SimulOneEllips(int argc,char **argv)
{
    Pt3dr aC(0,0,0);
    Pt3dr aR(0,0,0);
    int  aNbPts;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aR,"Ray")
                    << EAMC(aNbPts,"Number Pts")
        ,
        LArgMain()
                    << EAM(aC,"Center", true, "Center of tree")

    );

    cEllips3Dxy anE(Pt3df::P3ToThisT(aC),Pt3df::P3ToThisT(aR));


    // cPlyCloud  aPlyC;

    return EXIT_SUCCESS;
}


/*

class cArbreSimule
{
    public :
        cArbreSimule
        (
             double H,   // Center height
             double h,   //
             double Teta,
             double Sx,
             double Sy,
             int    aNbHoup

        );
    private :
};
*/





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
