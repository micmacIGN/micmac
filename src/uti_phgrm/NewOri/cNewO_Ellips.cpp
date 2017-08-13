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

#include "NewOri.h"

/*******************************************************/
/*                                                     */
/*              cGenGaus3D                             */
/*                                                     */
/*******************************************************/
static const double A = 0.147;

//              -  x
//    Erf(x) = /      e-(t*t) dt / sqrt(pi)
//            -  -x

//  DirErFonc  =>  approx de Erf(x) 

   // source https://en.wikipedia.org/wiki/Error_function

double DirErFonc(double x)
{
  static double QsPi = 4/PI;

  double X2 = x * x;
  double AX2 = A*X2;
  double aRes =  1 - exp(-X2*   ( (QsPi+AX2)/(1+AX2))   );

  aRes = sqrt(ElMax(0.0,aRes));
  return (x>0) ? aRes : (-aRes);
}

double DerEF(double x)
{
   return 2 *exp(-x*x) / sqrt(PI);
}

double InvErrFonc(double x)
{
    static double DePiSA = 2/(PI*A);

    double X2 = x * x;
    double Log1MX2 = log(1-X2);

    double aV1 = ElSquare(DePiSA + Log1MX2/2)  - Log1MX2/A;
    aV1 = sqrt(ElMax(0.0,aV1)) - (DePiSA+Log1MX2/2);

    aV1 = sqrt(ElMax(0.0,aV1));


    return (x>0) ? aV1 : (-aV1);
}


// Tabulation 
double InvErrFoncRationel(int P,int Q)
{
    int aSign = (P>0 ? 1 : -1) * (Q>0 ? 1 : -1);
    P = ElAbs(P);
    Q = ElAbs(Q);
    static std::vector<std::vector<double>  > mBuf;

    for (int q=mBuf.size() ; q<=Q ; q++)
    {
        mBuf.push_back(std::vector<double>());
        for (int p=0 ; p<q ; p++)
            mBuf.back().push_back(InvErrFonc(p/double(q)));
    }

    return mBuf[Q][P] * aSign;
}
static double FactCorrectif(int aNb)
{
    return sqrt(2.) / (1 - 0.3333/(aNb+0.5));
}

   //=======================================



/*
               double aFact =  sqrt(2) / (1 - 0.3333/(aNb+0.5));
               double aL1 = InvErrFonc( (2*aK1) / double(2*aNb+1)) * aFact;
               double aL2 = InvErrFonc( (2*aK2) / double(2*aNb+1)) * aFact;
               double aL3 = InvErrFonc( (2*aK3) / double(2*aNb+1)) * aFact;
               Pt3dr aP  =   aCDG 
                           + (aVec1*aL1*aNewVp1) 
                           + (aVec2*aL2*aNewVp2) 
                           + (aVec3*aL3*aNewVp3)
*/

void cGenGaus3D::GetDistribGaus(std::vector<Pt3dr> & aVPts,int aN1,int aN2,int aN3)
{
   aVPts.clear();
   Pt3dr aFact1 = mVecP[0] * (FactCorrectif(aN1) * mVP[0]);
   Pt3dr aFact2 = mVecP[1] * (FactCorrectif(aN2) * mVP[1]);
   Pt3dr aFact3 = mVecP[2] * (FactCorrectif(aN3) * mVP[2]);

   for (int aK1 =-aN1 ; aK1<=aN1 ; aK1++)
   {
       for (int aK2 =-aN2 ; aK2<=aN2 ; aK2++)
       {
           for (int aK3 =-aN3 ; aK3<=aN3 ; aK3++)
           {
               Pt3dr aP  =   mCDG 
                           + aFact1 * InvErrFoncRationel(2*aK1,2*aN1+1)
                           + aFact2 * InvErrFoncRationel(2*aK2,2*aN2+1)
                           + aFact3 * InvErrFoncRationel(2*aK3,2*aN3+1)
                         ;
               aVPts.push_back(aP);
           }
       }
   }
}

void cGenGaus2D::GetDistribGaus(std::vector<Pt2dr> & aVPts,int aN1,int aN2)
{
   aVPts.clear();
   Pt2dr aFact1 = mVecP[0] * (FactCorrectif(aN1) * mVP[0]);
   Pt2dr aFact2 = mVecP[1] * (FactCorrectif(aN2) * mVP[1]);

   for (int aK1 =-aN1 ; aK1<=aN1 ; aK1++)
   {
       for (int aK2 =-aN2 ; aK2<=aN2 ; aK2++)
       {
           Pt2dr aP  =   mCDG 
                       + aFact1 * InvErrFoncRationel(2*aK1,2*aN1+1)
                       + aFact2 * InvErrFoncRationel(2*aK2,2*aN2+1) ;
           aVPts.push_back(aP);
       }
   }
}


const double & cGenGaus3D::ValP(int aK) const { return mVP[aK]; }
const Pt3dr &  cGenGaus3D::VecP(int aK) const { return mVecP[aK]; }

const double & cGenGaus2D::ValP(int aK) const { return mVP[aK]; }
const Pt2dr &  cGenGaus2D::VecP(int aK) const { return mVecP[aK]; }


cGenGaus3D::cGenGaus3D(const cXml_Elips3D & anEl)  :
    mCDG (anEl.CDG())
{
    ELISE_ASSERT(anEl.Norm() ,"cGenGaus3D::cGenGaus3D");
    static ElMatrix<double> aMCov(3,3); 
    static ElMatrix<double> aValP(3,3); 
    static ElMatrix<double> aVecP(3,3); 

    aMCov(0,0) = anEl.Sxx();
    aMCov(1,1) = anEl.Syy();
    aMCov(2,2) = anEl.Szz();
    aMCov(0,1) = aMCov(1,0) =  anEl.Sxy();
    aMCov(0,2) = aMCov(2,0) =  anEl.Sxz();
    aMCov(1,2) = aMCov(2,1) =  anEl.Syz();


    std::vector<int>  aIVP = jacobi_diag(aMCov,aValP,aVecP);

    for (int aK=0 ; aK<3 ; aK++)
    {
        mVP[aK] =  sqrt(aValP(aIVP[aK],aIVP[aK]));
        aVecP.GetCol(aIVP[aK],mVecP[aK]);
    }

/*
    Pt3dr aVec1;
    aVecP.GetCol(aIVP[0],aVec1);
    std::cout << "SZZZ " << aValP.Sz() << " " << aIVP[0] << " " << aIVP[1] << " " << aIVP[2]  << "\n";
    std::cout << "VP " << sqrt(aValP(aIVP[0],aIVP[0])) << " " << sqrt(aValP(aIVP[1],aIVP[1])) << " " << sqrt(aValP(aIVP[2],aIVP[2])) << "\n";
    std::cout  << "V1 = " << aVec1 << "\n";
*/

}

cGenGaus2D::cGenGaus2D(const cXml_Elips2D & anEl)  :
    mCDG (anEl.CDG())
{
    ELISE_ASSERT(anEl.Norm() ,"cGenGaus3D::cGenGaus3D");
    static ElMatrix<double> aMCov(2,2); 
    static ElMatrix<double> aValP(2,2); 
    static ElMatrix<double> aVecP(2,2); 

    aMCov(0,0) = anEl.Sxx();
    aMCov(1,1) = anEl.Syy();
    aMCov(0,1) = aMCov(1,0) =  anEl.Sxy();


    std::vector<int>  aIVP = jacobi_diag(aMCov,aValP,aVecP);

    for (int aK=0 ; aK<2 ; aK++)
    {
        mVP[aK] =  sqrt(aValP(aIVP[aK],aIVP[aK]));
        aVecP.GetCol(aIVP[aK],mVecP[aK]);
    }
}





/*******************************************************/
/*                                                     */
/*              cXml_Elips3D                           */
/*                                                     */
/*******************************************************/

void RazEllips(cXml_Elips3D & anEl)
{
   anEl.CDG() = Pt3dr(0,0,0);
   anEl.Sxx() = 0.0;
   anEl.Syy() = 0.0;
   anEl.Szz() = 0.0;
   anEl.Sxy() = 0.0;
   anEl.Sxz() = 0.0;
   anEl.Syz() = 0.0;
   anEl.Pds() = 0;
   anEl.Norm() = false;
}

void RazEllips(cXml_Elips2D & anEl)
{
   anEl.CDG() = Pt2dr(0,0);
   anEl.Sxx() = 0.0;
   anEl.Syy() = 0.0;
   anEl.Sxy() = 0.0;
   anEl.Pds() = 0;
   anEl.Norm() = false;
}

void AddEllips(cXml_Elips3D & anEl,const Pt3dr & aP,double aPds)
{
   ELISE_ASSERT(!anEl.Norm(),"AddEllips");
   anEl.CDG() = anEl.CDG() + aP * aPds;
   anEl.Sxx() += aPds * aP.x * aP.x;
   anEl.Syy() += aPds * aP.y * aP.y;
   anEl.Szz() += aPds * aP.z * aP.z;
   anEl.Sxy() += aPds * aP.x * aP.y;
   anEl.Sxz() += aPds * aP.x * aP.z;
   anEl.Syz() += aPds * aP.y * aP.z;
   anEl.Pds() += aPds;
}

void AddEllips(cXml_Elips2D & anEl,const Pt2dr & aP,double aPds)
{
   ELISE_ASSERT(!anEl.Norm(),"AddEllips");
   anEl.CDG() = anEl.CDG() + aP * aPds;
   anEl.Sxx() += aPds * aP.x * aP.x;
   anEl.Syy() += aPds * aP.y * aP.y;
   anEl.Sxy() += aPds * aP.x * aP.y;
   anEl.Pds() += aPds;
}

void NormEllips(cXml_Elips3D & anEl)
{
   ELISE_ASSERT(!anEl.Norm(),"AddEllips");
   anEl.Norm() = true;
   double aPds = anEl.Pds();
   anEl.CDG() = anEl.CDG() / aPds;
   Pt3dr aCdg = anEl.CDG();

   anEl.Sxx() = anEl.Sxx() / aPds - aCdg.x * aCdg.x;
   anEl.Syy() = anEl.Syy() / aPds - aCdg.y * aCdg.y;
   anEl.Szz() = anEl.Szz() / aPds - aCdg.z * aCdg.z;
   anEl.Sxy() = anEl.Sxy() / aPds - aCdg.x * aCdg.y;
   anEl.Sxz() = anEl.Sxz() / aPds - aCdg.x * aCdg.z;
   anEl.Syz() = anEl.Syz() / aPds - aCdg.y * aCdg.z;
}
void NormEllips(cXml_Elips2D & anEl)
{
   ELISE_ASSERT(!anEl.Norm(),"AddEllips");
   anEl.Norm() = true;
   double aPds = anEl.Pds();
   anEl.CDG() = anEl.CDG() / aPds;
   Pt2dr aCdg = anEl.CDG();

   anEl.Sxx() = anEl.Sxx() / aPds - aCdg.x * aCdg.x;
   anEl.Syy() = anEl.Syy() / aPds - aCdg.y * aCdg.y;
   anEl.Sxy() = anEl.Sxy() / aPds - aCdg.x * aCdg.y;
}


void TestEllips_3D()
{
    while (1)
    {
        int aNbPts =  4 + NRrandom3(20);
        cXml_Elips3D  anEl;
   
        Pt3dr aC0 (NRrandC(),NRrandC(),NRrandC());
        Pt3dr aU0 (NRrandC(),NRrandC(),NRrandC());
        Pt3dr aU1 (NRrandC(),NRrandC(),NRrandC());
        Pt3dr aU2 (NRrandC(),NRrandC(),NRrandC());
        RazEllips(anEl);
        for (int aK=0 ; aK<aNbPts ; aK++)
        {
             // Pt3dr aP NRrandC(),NRrandC(),NRrandC());
             Pt3dr aP = aC0 + aU0 *  NRrandC() + aU1 * NRrandC() + aU2 * NRrandC();
             aP = aP * 10;
             AddEllips(anEl,aP,1.0);
        }
        NormEllips(anEl);

        cGenGaus3D aGG1(anEl);
        std::vector<Pt3dr> aVP;

        aGG1.GetDistribGaus(aVP,1+NRrandom3(2),2+NRrandom3(2),3+NRrandom3(2));

        RazEllips(anEl);
        for (int aK=0 ; aK<int(aVP.size()) ; aK++)
            AddEllips(anEl,aVP[aK],1.0);

        NormEllips(anEl);
        cGenGaus3D aGG2(anEl);

        for (int aK=0 ; aK< 3 ; aK++)
        {
            std::cout << "RATIO VP " << aGG1.ValP(aK) /  aGG2.ValP(aK) << " " 
                      <<  euclid( aGG1.VecP(aK) - aGG2.VecP(aK))       << " "
                      <<  " VP="  << aGG1.ValP(aK)       << " "
                      << "\n";
        }

        getchar();
    }
}

void TestEllips_2D()
{
    while (1)
    {
        int aNbPts =  4 + NRrandom3(20);
        cXml_Elips2D  anEl;
   
        Pt2dr aC0 (NRrandC(),NRrandC());
        Pt2dr aU0 (NRrandC(),NRrandC());
        Pt2dr aU1 (NRrandC(),NRrandC());
        RazEllips(anEl);
        for (int aK=0 ; aK<aNbPts ; aK++)
        {
             // Pt3dr aP NRrandC(),NRrandC(),NRrandC());
             Pt2dr aP = aC0 + aU0 *  NRrandC() + aU1 * NRrandC() ;
             aP = aP * 10;
             AddEllips(anEl,aP,1.0);
        }
        NormEllips(anEl);

        cGenGaus2D aGG1(anEl);
        std::vector<Pt2dr> aVP;

        // aGG1.GetDistribGaus(aVP,1+NRrandom3(2),2+NRrandom3(2));
        aGG1.GetDistribGaus(aVP,1,1); // Version nimimaliste

        RazEllips(anEl);
        for (int aK=0 ; aK<int(aVP.size()) ; aK++)
            AddEllips(anEl,aVP[aK],1.0);

        NormEllips(anEl);
        cGenGaus2D aGG2(anEl);

        for (int aK=0 ; aK< 2 ; aK++)
        {
            std::cout << "RATIO VP " << aGG1.ValP(aK) /  aGG2.ValP(aK) << " " 
                      <<  euclid( aGG1.VecP(aK) - aGG2.VecP(aK))       << " "
                      <<  " VP="  << aGG1.ValP(aK)       << " "
                      << "\n";
        }

        getchar();
    }
}





void TestEllips_0()
{
   cXml_Elips3D  anEl;
   
   RazEllips(anEl);

   Pt3dr aVec1 (NRrandC(),NRrandC(),NRrandC());
   Pt3dr aVec2 (NRrandC(),NRrandC(),NRrandC());
   Pt3dr aVec3 = SchmitComplMakeOrthon(aVec1,aVec2);

   Pt3dr aCDG (10*NRrandC(),10*NRrandC(),10*NRrandC());
   // double aVp1 = pow(1+ NRrandC(),3);
   // double aVp2 = pow(1+ NRrandC(),3);
   // double aVp3 = pow(1+ NRrandC(),3);

   double aVp1 = 1;
   double aVp2 = 10;
   double aVp3 = 100;

   int aNb = 10;

   std::cout << " VEC1 " << aVec1 << "\n";

   for (int aK1 =-aNb ; aK1<=aNb ; aK1++)
   {
       for (int aK2 =-aNb ; aK2<=aNb ; aK2++)
       {
           for (int aK3 =-aNb ; aK3<=aNb ; aK3++)
           {
               Pt3dr aP  =   aCDG 
                           + (aVec1*aK1*aVp1) 
                           + (aVec2*aK2*aVp2) 
                           + (aVec3*aK3*aVp3)
                         ;
               AddEllips(anEl,aP,1.0);
           }
       }
   }
   NormEllips(anEl);
   cGenGaus3D aGG1(anEl);

   // Verifie que les vecp sont bien ceux ayant servi a generer
   if (1)
   {
       std::cout << "VALP " <<  aGG1.ValP(0) << " " << aGG1.ValP(1) << " " << aGG1.ValP(2) << "\n";
       std::cout << "VEC1 " <<  aGG1.VecP(0) << " " << aVec1 << "\n";
       std::cout << "VEC2 " <<  aGG1.VecP(1) << " " << aVec2 << "\n";
       std::cout << "VEC3 " <<  aGG1.VecP(2) << " " << aVec3 << "\n";
   }
   double aNewVp1 = aGG1.ValP(0);
   double aNewVp2 = aGG1.ValP(1);
   double aNewVp3 = aGG1.ValP(2);

   // Verifie DerEF est intregale de gaussienne et InvErrFonc * DerEF = Id
   if (0)
   {
     for (int aK=1 ; aK<=30 ; aK++)
     {
        double x = 1-pow(aK/30.0,4);
        // std::cout << aK << " " << erfcc(aK) << "\n";
        double Ix = InvErrFonc(x);
        double Eps = 1e-4;
        std::cout << "  " << x 
                  << " "  << Ix 
                  << " " << DirErFonc (Ix) 
                  << " D1=" << DerEF(Ix) 
                  << " D2=" << (DirErFonc(Ix+Eps)  - DirErFonc(Ix-Eps)) / (2*Eps)
                  << "\n";
     }
     std::cout <<  InvErrFonc(0.999) << "\n";
   }


   if (0)
   {
      for (int aK=0 ; aK< 1000; aK++)
      {
          int aQ = 1 + NRrandom3(100);
          int aP = NRrandom3(aQ-1);
          double aV1 = InvErrFoncRationel(aP,aQ);
          double aV2 =  InvErrFonc(aP/double(aQ));
          ELISE_ASSERT(ElAbs(aV1-aV2) < 1e-5,"InvErrFoncRationel check");
          std::cout << aV1 <<  " " << aV2 << "\n";
      }
   }

   RazEllips(anEl);

   aNb = 20;

   for (int aK1 =-aNb ; aK1<=aNb ; aK1++)
   {
       for (int aK2 =-aNb ; aK2<=aNb ; aK2++)
       {
           for (int aK3 =-aNb ; aK3<=aNb ; aK3++)
           {

               double aFact =  sqrt(2.) / (1 - 0.3333/(aNb+0.5));
               double aL1 = InvErrFonc( (2*aK1) / double(2*aNb+1)) * aFact;
               double aL2 = InvErrFonc( (2*aK2) / double(2*aNb+1)) * aFact;
               double aL3 = InvErrFonc( (2*aK3) / double(2*aNb+1)) * aFact;
               Pt3dr aP  =   aCDG 
                           + (aVec1*aL1*aNewVp1) 
                           + (aVec2*aL2*aNewVp2) 
                           + (aVec3*aL3*aNewVp3)
                         ;
               AddEllips(anEl,aP,1.0);
           }
       }
   }
   NormEllips(anEl);
   cGenGaus3D aGG2(anEl);
   // std::cout << "RATIO " <<  aGG.ValP(0)/aNewVp1 << " " << aGG.ValP(1)/aNewVp2 << " " << aGG.ValP(2)/aNewVp3 << "\n";
   double aR = aGG2.ValP(0)/aNewVp1;
   std::cout << "RATIO " <<  aR << " " << (1-aR) * (aNb+0.5) << "\n";

   // Verifie que les vecp sont bien ceux ayant servi a generer
   if (1)
   {
       std::cout << "VALP " <<  aGG2.ValP(0) << " " << aGG2.ValP(1) << " " << aGG2.ValP(2) << "\n";
       std::cout << "RATIO " <<  aGG2.ValP(0)/aNewVp1 << " " << aGG2.ValP(1)/aNewVp2 << " " << aGG2.ValP(2)/aNewVp3 << "\n";
       std::cout << "VEC1 " <<  aGG2.VecP(0) << " " << aVec1 << "\n";
       std::cout << "VEC2 " <<  aGG2.VecP(1) << " " << aVec2 << "\n";
       std::cout << "VEC3 " <<  aGG2.VecP(2) << " " << aVec3 << "\n";
   }

std::cout << "AAAAAAAAAAAAaaaaaa\n";
   std::vector<Pt3dr> aVP;
   aGG1.GetDistribGaus(aVP,1,2,3);
std::cout << "BBBBbbbbbbbbbb\n";
   RazEllips(anEl);
   for (int aK=0 ; aK<int(aVP.size()) ; aK++)
       AddEllips(anEl,aVP[aK],1.0);
std::cout << "CCCcccc\n";
   NormEllips(anEl);
   cGenGaus3D aGG3(anEl);

   std::cout << "RATIO GAUSS " <<  aGG3.ValP(0)/aNewVp1 << " " << aGG3.ValP(1)/aNewVp2 << " " << aGG3.ValP(2)/aNewVp3 << "\n";
}


void TestEllips()
{
    TestEllips_2D();
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
