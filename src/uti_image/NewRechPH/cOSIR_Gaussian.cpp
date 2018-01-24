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

#include "NewRechPH.h"


// Gaussian   1/(sqrt(2Pi) Sig2)  exp(-X/2 Sig^2)

// PB = > 

// Simga2 =  (2*a) /(ElSquare(1-a)); voir TestSigma2 ds NewRechPH.cpp
// a = ((Simga2+1) + sqrt(((Simga2+1)^2 -1)  ) / sigma2

double Sigma2FromFactExp(double a)
{
   return (2*a) / ElSquare(1-a);
}
double FactExpFromSigma2(double aS2)
{
    return (aS2+1 - sqrt(ElSquare(aS2+1)-ElSquare(aS2))  ) / aS2 ;
}

void TestSigma2(double a)
{
   int aNb = 100 + 100/(1-a);
   double aSom=0;
   double aSomK2=0;
   for (int aK=-aNb ; aK<= aNb ; aK++)
   {
       double aPds = pow(a,ElAbs(aK));
       aSom += aPds;
       aSomK2 += aPds * ElSquare(aK);
   }

   double aSigmaExp =  aSomK2 / aSom;
   // double aSigmaTh = (2*a) /((1+a) * ElSquare(1-a));
   // double aSigmaTh = (2*a) /(ElSquare(1-a));
   double aSigmaTh = Sigma2FromFactExp(a);

   double aATh = FactExpFromSigma2(aSigmaTh);

   std::cout << "TestSigma2 " << aSigmaExp << " EcarRel=" << aSigmaTh/aSigmaExp - 1 << " FactExp=" << a << " Sig2=" << aATh << "\n";
}

double Gauss(double aSig,double aVal)
{
   static double aS2PI = sqrt(1/(2*PI));
   double aS2 = ElSquare(aSig);

   return (aS2PI / aSig) * exp(-ElSquare(aVal)/ (2*aS2));
}
void TestGauss(double aSig)
{
   int aNbIn = 1000;
   int aNbOut = 10;
   int aNbTot = aNbIn * aNbOut;

   double aSomG = 0.0;
   double aSomGX = 0.0;
   double aSomGX2 = 0.0;

   double aStep = aSig / double(aNbIn);

   for (int aK=-aNbTot ; aK<= aNbTot ; aK++)
   {
        double aVal = aK * aStep;

        double aG = Gauss(aSig,aVal) * aStep;
        aSomG   += aG;
        aSomGX  += aG * aVal;
        aSomGX2 += aG * ElSquare(aVal);
   }
   std::cout << "IntegG=" << aSomG   
             <<  " S2=" << aSomGX2 / ElSquare(aSig) 
             << " Moy=" << aSomGX  
             << "\n";
// Gaussian   1/(sqrt(2Pi) Sig2)  exp(-X/2 Sig^2)
}

void TestEcartTypeStd()
{
   TestGauss(0.5);
   TestGauss(1);
   TestGauss(2);
   getchar();
   TestSigma2(0.1);
   TestSigma2(0.5);
   TestSigma2(0.9);
   TestSigma2(0.95);
   TestSigma2(0.99);
   getchar();

}




template <class T1> void  LocFilterGauss(T1 & anIm, double aSigmaN,int aNbIter)
{
  double aSig0 = aSigmaN / sqrt(aNbIter);
  double aF = FactExpFromSigma2(ElSquare(aSig0));

  Pt2di aSz = anIm.sz();
  Im2D_REAL4 aIP1(aSz.x,aSz.y,1.0);
  FilterExp(aIP1,aF);

  for (int aKIt=0 ; aKIt<aNbIter ; aKIt++)
  {
      FilterExp(anIm,aF);
      ELISE_COPY(anIm.all_pts(),anIm.in()/aIP1.in(),anIm.out());
  }
}


void FilterGaussProgr(tImNRPH anIm,double  aSTarget,double  aSInit,int aNbIter)
{
    aSTarget = ElSquare(aSTarget);
    aSInit = ElSquare(aSInit);
    if (aSTarget > aSInit)
    {
        LocFilterGauss(anIm,sqrt(aSTarget-aSInit),aNbIter);
    }
    else if (aSTarget==aSInit)
    {
    }
    else
    {
      ELISE_ASSERT(false,"FilterGaussProgr");
    }
}

void TestDist(Pt2di aSz,Fonc_Num aP,double aScale)
{
   Symb_FNum aSP = aP;
   double aSom,aSX,aSY,aSXX,aSYY;
   ELISE_COPY
   (
       rectangle(Pt2di(0,0),aSz),
       Virgule(aSP,aSP*FX,aSP*FY,aSP*FX*FX,aSP*FY*FY),
       Virgule(sigma(aSom),sigma(aSX),sigma(aSY),sigma(aSXX),sigma(aSYY))
   );
   aSX /= aSom;
   aSY /= aSom;
   aSXX /= aSom;
   aSYY /= aSom;
   aSXX -= ElSquare(aSX);
   aSYY -= ElSquare(aSY);

   // Il y a une difficulte, pour les vrais images on fait l'hypothese que a la premiere
   // echelle sigm=1 (on n'en sait rien, ptet + ptet - en fait)
   // mais ici avec un dirac la sigma=0 a ech =1
   // la relation est en 1+ sig ^2 = Ech ^2

   double aSigTh = sqrt(ElSquare(aScale)-1);

   std::cout << "SC= " << aScale  
             << "STAT S=" << aSom << " X=" << aSX << " Y=" << aSY
             << " SigX=" << sqrt(aSXX) / aSigTh 
             << " SigY=" << sqrt(aSYY) / aSigTh << "\n";
}

int Generate_ImagSift(int argc,char ** argv)
{
     Pt2di aSz(1000,1000);
     Im2D_REAL4 aIm(aSz.x,aSz.y);

     for (int aKx=0 ; aKx<10 ; aKx++)
     {
         for (int aKy=0 ; aKy<10 ; aKy++)
         {
             Pt2di aP0(aKx*100,aKy*100);
             Pt2di aP1((aKx+1)*100,(aKy+1)*100);
             Pt2dr aMil = Pt2dr(aP0+aP1) / 2.0;

             double aSigmaX = (0.25*aKx + 1.0*aKy  + 1);
             double aSigmaY = ( 1.0*aKx + 0.25*aKy + 1);
             double aSign = ((aKx+aKy) % 2)   ? 1 : -1;

             ELISE_COPY
             (
                  rectangle(aP0,aP1),
                  128 * (1+aSign * exp(-  ( Square(FX-aMil.x)/Square(aSigmaX) + Square(FY-aMil.y)/ Square(aSigmaY))   )),
                  aIm.out()
             );

         }
     }
     Tiff_Im::CreateFromIm(aIm,"TestSift.tif");
     return EXIT_SUCCESS;
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
aooter-MicMac-eLiSe-25/06/2007*/
