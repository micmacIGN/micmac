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

//=======================================================================================================
//   =========================  RMatr Inertie ===========================================================
//=======================================================================================================

/*
     a V1 +b = v2 =u
     
      (VV  V)     (VU)              (N    -V)               (VU
      (V   N) -2  (U)  +  UU        (-V    VV) / delta      (U

      (a)     
      (b)  =  ( N VV 
*/
//  solution optimale de a V1 +b = V2, au sens des moindres carres, Pt2dr(a,b)
// bool BUGRECT=false;

Pt2dr  LSQSolDroite(const  RMat_Inertie & aMatr,double & aDelta)
{
    aDelta = aMatr.s11() *aMatr.s() - ElSquare(aMatr.s1());

    double A = (aMatr.s12() * aMatr.s() - aMatr.s1() * aMatr.s2());
    double B = - aMatr.s1() *aMatr.s12() + aMatr.s11() * aMatr.s2();

    return Pt2dr(A,B) / aDelta;
}
Pt2dr  LSQSolDroite(const  RMat_Inertie & aMatr)
{
    double aDelta;
    return LSQSolDroite( aMatr,aDelta);
}
double   LSQResiduDroite(const  RMat_Inertie & aMatr)
{
    double aDelta;
    Pt2dr aAB = LSQSolDroite(aMatr,aDelta);
    double a = aAB.x;
    double b = aAB.y;

    double aRes =  aMatr.s11()*a*a + 2*aMatr.s1()*a*b +aMatr.s()*b*b -2*(aMatr.s12()*a+aMatr.s2()*b) + aMatr.s22();
    return aRes;
}
double   LSQMoyResiduDroite(const  RMat_Inertie & aMatr)
{
    return sqrt(ElMax(0.0,LSQResiduDroite(aMatr) / ElMax(1e-60,aMatr.s())));
}



double  L1EcartSolAffine(const double & A,const double & B,const std::vector<double> & aV1,const std::vector<double> & aV2)
{
    double aSol = 0.0;
    int aNbV = aV1.size();
    for (int aK=0 ; aK<aNbV ; aK++)
        aSol += ElAbs(A*aV1[aK]+B-aV2[aK]);
    
    return aSol;
}

//=======================================================================================================
//=======================================================================================================
//=======================================================================================================


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
  mPC1    (aPC1),
  mTI1    (aI1),
  mData1  (aI1.data()),
  mTI2    (aI2),
  mData2  (aI2.data()),
  mAf1To2 (anAf1To2),
  mA      (1.0),
  mB      (0.0)
{
}


void cLSQAffineMatch::CalcRect(tInterpolTiepTri * anInterp,double aStepTot)
{
    mPSupIm1 = mPC1;
    mPInfIm1 = mPC1;
    Pt2dr aPC2 =  mAf1To2(mPC1);

    mPSupIm2 = aPC2;
    mPInfIm2 = aPC2;
    ElAffin2D anAfRec = mAf1To2.inv();
    for (int aK=0 ; aK<8 ; aK++)
    {
         Pt2dr aVois = Pt2dr(TAB_8_NEIGH[aK]) * aStepTot;


         Pt2dr aPIm1 = mPC1 + aVois;
         Pt2dr aPIm2 = mAf1To2(aPIm1);

         Pt2dr aQIm2 =  aPC2 + aVois;
         Pt2dr aQIm1 =  anAfRec(aQIm2);

         mPInfIm1 = Inf3(mPInfIm1,aPIm1,aQIm1);
         mPSupIm1 = Sup3(mPSupIm1,aPIm1,aQIm1);
         mPInfIm2 = Inf3(mPInfIm2,aPIm2,aQIm2);
         mPSupIm2 = Sup3(mPSupIm2,aPIm2,aQIm2);
    }

    double aRab = anInterp->SzKernel() + 2;
    Pt2dr aPRab(aRab,aRab);

    mPInfIm1 = mPInfIm1 - aPRab;
    mPSupIm1 = mPSupIm1 + aPRab;
    mPInfIm2 = mPInfIm2 - aPRab;
    mPSupIm2 = mPSupIm2 + aPRab;
}

void cLSQAffineMatch::AddEqq(L2SysSurResol & aSys,const Pt2dr &aPIm1,const Pt2dr & aPC1)
{
/*
     static int aCpt=0 ; aCpt++;
     bool Bug = (aCpt==7639420) ;
*/
     // Pt2dr aPIm1 = mPC1 + Pt2dr(aKx*aStep,aKy*aStep);
     Pt2dr aPIm2 = mAf1To2(aPIm1);
/*
if (Bug)
{ 
    std::cout << "xxxxxxP1= " << aPIm1   << mTI1.sz() << " P2= " << aPIm2 << " " << mTI2.sz() << "\n";
    std::cout << " Box1=" << mPInfIm1 << mPSupIm1  << " Box2=" << mPInfIm2 << mPSupIm2 << "\n";
}
*/
     double aV1 = mInterp->GetVal(mData1,aPIm1);    // value of center point (point master)

     Pt3dr aNewVD2= mInterp->GetValDer(mData2,aPIm2);   // Get intensity & derive value of point 2nd img
     double aGr2X = aNewVD2.x;  // derive en X
     double aGr2Y = aNewVD2.y;  // derive en Y
     double aV2   = aNewVD2.z;  // valeur d'intensite

     mCoeff[NumAB] = aV1 ; // A
     mCoeff[NumAB+1] = 1.0 ; // B
     mCoeff[NumTr] = -aGr2X; // im00.x
     mCoeff[NumTr+1] = -aGr2Y;  // im00.y


     if (mAffineGeom)
     {
        mCoeff[NumAffGeom] =   -aGr2X*aPIm1.x; // im10.x
        mCoeff[NumAffGeom+1] = -aGr2Y *aPIm1.x;  // im10.y
        mCoeff[NumAffGeom+2] = -aGr2X*aPIm1.y; // im01.x
        mCoeff[NumAffGeom+3] = -aGr2Y *aPIm1.y;  // im01.y
     }

     if (mAffineRadiom)
     {
        mCoeff[NumAfRad] =   aV1 * (aPIm1.x-aPC1.x);
        mCoeff[NumAfRad+1] = aV1 * (aPIm1.y-aPC1.y);
     }
     aSys.AddEquation(1.0,mCoeff,aV2);

     mSomDiff += ElSquare(aV2 - mA * aV1-mB);
/*
if (Bug)
{
    std::cout << "Buuuug-DONNE \n";
}
*/
}

bool cLSQAffineMatch::OneIter(tInterpolTiepTri * anInterp,int aNbW,double aStep,bool AffineGeom,bool AffineRadiom)
{
// static int CPT=0; CPT++;
// std::cout << "CccCPT= " << CPT << "\n";
/*
if (CPT<=8369) return false;
*/



    mInterp = anInterp;
    mAffineGeom = AffineGeom;
    mAffineRadiom = AffineRadiom;
    // aStep = 1/NbByPix => "real size" of a pixel
    int aNbPixTot = round_ni(aNbW/aStep); // => calcul "real" window size from user given "window size" & Nb of point inside 1 pixel
    aStep = double(aNbW) / aNbPixTot;
    // double aCoeff[10];
    /* calcul number of variable for system equation :
       * No Aff, No Radio => 4 variables (2 translation part of affine, A , B)
       * No Aff, With Radio => 6 variables
       * With Aff, No Radio => 8 variables (plus 4 variable of affine part)
       * With Aff, With Radio => 10 variables
     */
    int aNbInc = 4 + (mAffineGeom ? 4 :0) + (mAffineRadiom ? 2 : 0);

    // Num* is position in array of output estimation result
    NumAB = 0;
    NumTr = NumAB +2;
    NumAffGeom = NumTr +2 ;
    NumAfRad = mAffineGeom ? (NumAffGeom+4 ) : (NumTr+2);   // Num Affine Radiometry

    L2SysSurResol aSys(aNbInc); // 4/6/8/10 variable
    mSomDiff = 0;

    CalcRect(mInterp,aNbW);     // calcul Pt Haut Gauche & Pt Bas Droite to form a rectangle on both image

    if (   (!mTI1.inside_rab(mPInfIm1,0))   // mPInfIm1 = Pt inferieur Image 1
        || (!mTI1.inside_rab(mPSupIm1,0))   // mPSupIm1 = Pt superieur Image 1
        || (!mTI2.inside_rab(mPInfIm2,0)) 
        || (!mTI2.inside_rab(mPSupIm2,0)) 
       )
       return false;            // check if rectangle is inside image


    ElAffin2D anAfRec = mAf1To2.inv();
    Pt2dr aPC2 = mAf1To2(mPC1);         // mPC1 : pt correl init sur image 1 (pt master)

    // Add equation to system
    for (int aKx=-aNbPixTot ; aKx<=aNbPixTot ; aKx++)
    {
        for (int aKy=-aNbPixTot ; aKy<=aNbPixTot ; aKy++)
        {
             Pt2dr aPVois (aKx*aStep,aKy*aStep); // pixel voisin dans coordonne global image
//static int aCpt=0; aCpt++;
//std::cout << "Cppt0=" << aCpt << "\n";
//bool aBug = (aCpt== 70700);
            Pt2dr aPIm1 = mPC1 + aPVois;    // aPIm1 : pixel voisin dans coordonne global image
//if (aBug) std::cout << " Addd1 " << aPIm1 << "\n";
            AddEqq(aSys,aPIm1,mPC1);        // 1 pixel aPIm1 dans le vignette => 1 equation

            if (1)
            {
                 Pt2dr aQIm2 = aPC2 + aPVois;
                 Pt2dr aQIm1 =   anAfRec(aQIm2);
//if (aBug) std::cout << " Addd2 " << aQIm2 << " " << aQIm1 << "\n";

                 // 1 pixel aQIm1  dans le vignette => ajoute encore 1 equation.
                 // aQIm1 est le "meme point" avec aPIm1 mais calcul avec affine inverse depuis aPC2 dans image 2nd
                 AddEqq(aSys,aQIm1,mPC1);
            }
//if (aBug) std::cout << " DddddOonnnnne \n";
        }
    }

    mSomDiff /= ElSquare(1+2*aNbPixTot);
    mSomDiff = sqrt(mSomDiff);

    bool Ok;
    Im1D_REAL8 aSol = aSys.Solve(&Ok);
    if (!Ok) return false;
    double * aDS = aSol.data();

    mA = aDS[NumAB];
    mB = aDS[NumAB+1];
    Pt2dr aI00(aDS[NumTr],aDS[NumTr+1]);    // translation part
    Pt2dr aI10(0,0);                        // if we don't re-estimate affine part => (0,0) ???? Must be identity matrix ???
    Pt2dr aI01(0,0);                        // if we don't re-estimate affine part => (0,0) ???? Must be identity matrix ???

    if (mAffineGeom)
    {
        aI10 = Pt2dr(aDS[NumAffGeom  ],aDS[NumAffGeom+1]);  // take estimate result of affine part if we do "LSQC = 1"
        aI01 = Pt2dr(aDS[NumAffGeom+2],aDS[NumAffGeom+3]);
    }
    
    // std::cout  << "llLSQ : " << aI00 << aI10 << aI01 << "\n";

    // std::cout << "DdIIff=" << aSomDiff << " IIIIii  " << aI00 << " " << aI10 << " " << aI01 << "\n";

    ElAffin2D  aDeltaAff(aI00,aI10,aI01);     // new estimated affine transformation from LSQ (transation + affine)

    mAf1To2 = mAf1To2 + aDeltaAff;  // update final result to global image coordinate

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
