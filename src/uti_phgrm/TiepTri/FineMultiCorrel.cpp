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

#include "TiepTri.h"

double  L1EcartSolAffine(const double & A,const double & B,const std::vector<double> & aV1,const std::vector<double> & aV2)
{
    double aSol = 0.0;
    int aNbV = aV1.size();
    for (int aK=0 ; aK<aNbV ; aK++)
        aSol += ElAbs(A*aV1[aK]+B-aV2[aK]);
    
    return aSol;
}
// Resoud A V1 + B = V2 , metrique L1 par L2 iteree


void LSQEstimateModeleAffine(double & A,double & B,const std::vector<double> & aV1,const std::vector<double> & aV2,const std::vector<double> * aVPds)
{
    int aNbV = aV1.size();
    ELISE_ASSERT(int(aV2.size())==aNbV,"EstimateModeleAffine sz incoherent");
    RMat_Inertie aMat;
    for (int aK=0 ; aK<aNbV ; aK++)
    {
         double aPds = aVPds ? (*aVPds)[aK] : 1.0;
         aMat.add_pt_en_place(aV1[aK],aV2[aK],aPds);
    }
    aMat = aMat.normalize();

    A=  sqrt(aMat.s22() / aMat.s11());
    if (aMat.s12() < 0)
    {
       A = -A;
    }
    B = aMat.s2() - aMat.s1() * A;
}



void TMA()
{
    while (1)
    {
         double A=  NRrandC();
         if (NRrandC() <0) A = 1/A;
         double B = 20 * NRrandC();
         int aNb = 10  + 100 * (1+NRrandC());

         std::vector<double> aV1;
         std::vector<double> aV2;
         for (int aK=0 ; aK< aNb ; aK++)
         {
              double x1 = NRrandC() ;
              double x2 = A*x1+ B;
              aV1.push_back(x1);
              aV2.push_back(x2);
         }
         double aSolA,aSolB;
         LSQEstimateModeleAffine(aSolA,aSolB,aV1,aV2,0);
         std::cout << " A=" << A << " " << aSolA << " B="  << B << " " << aSolB << "\n";
    }
}
/*
*/

/*
class cLSQAffineMatch
{
    public :
        cLSQAffineMatch
        (
            Pt2dr              aPC1,
            const tImTiepTri & aI1,
            const tImTiepTri & aI2,
            ElAffin2D          anAf1To2
        );

        bool OneIter(int aNbW,double aStep,bool AffineRadiom);
        const ElAffin2D &    Af1To2() const;

    private :
        Pt2dr         mPC1;
        tTImTiepTri   mTI1;
        tTImTiepTri   mTI2;
        ElAffin2D     mAf1To2;
};
*/

cLSQAffineMatch::cLSQAffineMatch
(
            Pt2dr              aPC1,
            const tImTiepTri & aI1,
            const tImTiepTri & aI2,
            ElAffin2D          anAf1To2
) :
  mPC1 (aPC1),
  mTI1 (aI1),
  mTI2 (aI2),
  mAf1To2 (anAf1To2),
  mA      (1.0),
  mB      (0.0)
{
}


bool cLSQAffineMatch::OneIter(int aNbW,double aStep,bool AffineRadiom)
{
    int aNbPixTot = round_ni(aNbW/aStep);
    double aCoeff[10];
    int aNbInc = AffineRadiom ? 10 : 8;
    L2SysSurResol aSys(aNbInc);

    double aSomDiff = 0;

    for (int aKx=-aNbPixTot ; aKx<=aNbPixTot ; aKx++)
    {
        for (int aKy=-aNbPixTot ; aKy<=aNbPixTot ; aKy++)
        {
             Pt2dr aPIm1 = mPC1 + Pt2dr(aKx*aStep,aKy*aStep);
             Pt2dr aPIm2 = mAf1To2(aPIm1);
             if ((!mTI1.Rinside_bilin(aPIm1)) || (!mTI2.Rinside_bilin(aPIm2)))
                return false;
 
             double aV1 =mTI1.getr(aPIm1);
             Pt3dr  aVD2 =  mTI2.getVandDer(aPIm2);
             double aGr2X = aVD2.x;
             double aGr2Y = aVD2.y;
             double aV2   = aVD2.z;

             // A V1(P1) + B  = V2(P2Init + im00 +  im10 * PI1.x  + im01 * aPI1.y)
             // A V1(P1) + B  = V2(P2Init) + DV2/Dx ( im00.x + im10.x * aPI1.x + im01.x  * aPI1.y)
             //                            + DV2/Dy ( im00.y + im10.y * aPI1.x + im01.y  * aPI1.y)
              aCoeff[0] = -aGr2X; // im00.x
              aCoeff[1] = -aGr2X*aPIm1.x; // im10.x
              aCoeff[2] = -aGr2X*aPIm1.y; // im01.x
              aCoeff[3] = -aGr2Y;  // im00.y
              aCoeff[4] = -aGr2Y *aPIm1.x;  // im10.y
              aCoeff[5] = -aGr2Y *aPIm1.y;  // im01.y
              aCoeff[6] = aV1 ; // A
              aCoeff[7] = 1.0 ; // B
              if (AffineRadiom)
              {
                 aCoeff[8] = aV1 * aKx;
                 aCoeff[9] = aV1 * aKy;
              }
              aSys.AddEquation(1.0,aCoeff,aV2);

              aSomDiff += ElSquare(aV2 - mA * aV1-mB);
        }
    }
    aSomDiff /= ElSquare(1+2*aNbPixTot);
    aSomDiff = sqrt(aSomDiff);


    bool Ok;
    Im1D_REAL8 aSol = aSys.Solve(&Ok);
    if (!Ok) return false;
    double * aDS = aSol.data();
    Pt2dr aI00(aDS[0],aDS[3]);
    Pt2dr aI10(aDS[1],aDS[4]);
    Pt2dr aI01(aDS[2],aDS[5]);
    
    mA = aDS[6];
    mB = aDS[7];

    // std::cout << "DdIIff=" << aSomDiff << " IIIIii  " << aI00 << " " << aI10 << " " << aI01 << "\n";

    ElAffin2D  aDeltaAff(aI00,aI10,aI01);

    mAf1To2 = mAf1To2 + aDeltaAff;

    return true;
}

const ElAffin2D &    cLSQAffineMatch::Af1To2() const
{
   return mAf1To2;
}


/*
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
aooter-MicMac-eLiSe-25/06/2007*/
