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
            ===  14/02/2015 =============


J'abandonne, au moins provisoirement, la "voie" Oribundle car il faut pousser tres loin
le nombe de test en teta, sinon on manque souvent les solutions. Du coup c'est cher
en temps de calcul.


*/

/*
    Resoud l'equation :
       [aBase   aDirkA  Rot(aDirkB)] =0 ,  K in (1,2,3)
       (aBase ^aDirkA) . Rot(aDirkB) = 0
*/

class cOFB_Sol1S1T // One signe, One teta
{
    public :
       bool SameSigne(const cOFB_Sol1S1T & aS2) {return (mScal<0) == (aS2.mScal<0);}
       cOFB_Sol1S1T Pond(const cOFB_Sol1S1T&) const;
       void Show(const std::string & aMes);
       int      mSign;
       double   mScal;
       Pt3dr    mV1;
       Pt3dr    mV2;
       Pt3dr    mV3;
};


void cOFB_Sol1S1T::Show(const std::string & aMes)
{
    std::cout << aMes << "Sc " << mScal << " V " << mV1 << mV2 << mV3 << "\n";
}

cOFB_Sol1S1T cOFB_Sol1S1T::Pond(const cOFB_Sol1S1T& aS2) const
{
    double anEcart = aS2.mScal - mScal;
    double aPds1 = aS2.mScal / anEcart;
    double aPds2 = - mScal / anEcart;

    cOFB_Sol1S1T aRes;
    aRes.mSign  = mSign;
    aRes.mScal  = 0.0;
    aRes.mV1 = mV1 * aPds1 + aS2.mV1 * aPds2;
    aRes.mV2 = mV2 * aPds1 + aS2.mV2 * aPds2;
    aRes.mV3 = mV3 * aPds1 + aS2.mV3 * aPds2;

    return aRes;
}

class cOFB_Sol1T
{
     public :
        bool operator <(const cOFB_Sol1T & aS2) const {return mTeta<aS2.mTeta;}
        bool mOK;
        double mTeta;
        double mNorm;
        cOFB_Sol1S1T  mSols[2];

};

class cCmp_OFBS
{
    public :
        bool operator ()  (cOFB_Sol1T * aS1,cOFB_Sol1T * aS2) const {return aS1->mTeta < aS2->mTeta;}
};

std::list<ElMatrix<double> >  OriFromBundle
                              (
                                 Pt3dr aBase,
                                 Pt3dr aDir1A,
                                 Pt3dr aDir2A,
                                 Pt3dr aDir3A,
                                 Pt3dr aDir1B,
                                 Pt3dr aDir2B,
                                 Pt3dr aDir3B,
                                 int aNbInit,int aDepthNonSol,int aDepthSol
                              );


typedef std::pair<int,int> tPairOFBSol;

class cOriFromBundle
{
      public :

           cOriFromBundle
           (
                 Pt3dr aBase,
                 Pt3dr aDir1A,
                 Pt3dr aDir2A,
                 Pt3dr aDir3A,
                 Pt3dr aDir1B,
                 Pt3dr aDir2B,
                 Pt3dr aDir3B
           );

           void  Solve(int aNbInit,int aDepthNonSol,int aDepthSol);
           const std::list<ElMatrix<double> > & ListRots() const {return mListRots;}
     private :
           ElMatrix<double> CalculRot(const cOFB_Sol1S1T & aSol,bool Show);
           void AnalyseSol(const ElMatrix<double> & aMat) const;
           void AnalyseSol(const ElMatrix<double> & aMat,const Pt3dr & aPA, const Pt3dr & aPB) const;


           void  TestTeta(double aTeta,int aSign0,int aSign1);
           void ExploreDichotFrontNonSol(int aK0,int aK2,int aLevel);
           tPairOFBSol  ExploreDichotChSgn(int aK0,int aK2,int aLevel,int aKSign);

           Pt3dr mBase;
           Pt3dr mDir1A;
           Pt3dr mDir2A;
           Pt3dr mDir3A;
// Direction |_ a B et au mDirKA
           Pt3dr mDirOr1A;
           Pt3dr mDirOr2A;
           Pt3dr mDirOr3A;

           Pt3dr mDir1B;
           Pt3dr mDir2B;
           Pt3dr mDir3B;
           double mSc12;

           // mDir1B mYB mZB est _|   mDir1B mYB m plan que mDir1B/mDir2B
           Pt3dr  mZB;
           Pt3dr  mYB;
 // Coordonnee de mDir3B dans mDir1B mYB mZB
           double mSc3X;
           double mSc3Y;
           double mSc3Z;

  // X12 Y1 est une base du plan |_ a aDirOr1A, X12 est aussi |_ a aDirOr2B,
  // X12 Y2 est une base du plan |_ a  aDirOr2B
           Pt3dr mX12;
           Pt3dr mY1;
           Pt3dr mZ1;
           Pt3dr mY2;
           double mCosY2;
           double mSinY2;
           std::vector<cOFB_Sol1T> mVSols;

           std::list<ElMatrix<double> > mListRots;
};



cOriFromBundle::cOriFromBundle
(
      Pt3dr aBase,
      Pt3dr aDir1A,
      Pt3dr aDir2A,
      Pt3dr aDir3A,
      Pt3dr aDir1B,
      Pt3dr aDir2B,
      Pt3dr aDir3B
) :
  mBase    (vunit(aBase)) ,
  mDir1A   (vunit(aDir1A)),
  mDir2A   (vunit(aDir2A)),
  mDir3A   (vunit(aDir3A)),
  mDirOr1A (vunit(mBase^mDir1A)),
  mDirOr2A (vunit(mBase^mDir2A)),
  mDirOr3A (vunit(mBase^mDir3A)),
  mDir1B   (vunit(aDir1B)),
  mDir2B   (vunit(aDir2B)),
  mDir3B   (vunit(aDir3B)),
  mSc12    (scal(mDir1B,mDir2B)),
  mZB      (vunit(mDir1B^mDir2B)),
  mYB      (vunit(mZB^mDir1B)),
  mSc3X    (scal(mDir3B,mDir1B)),
  mSc3Y    (scal(mDir3B,mYB)),
  mSc3Z    (scal(mDir3B,mZB)),
  mX12     (vunit(mDirOr1A^mDirOr2A)),
  mY1      (vunit(mDirOr1A^mX12)),
  mZ1      (mX12 ^ mY1),
  mY2      (vunit(mDirOr2A^mX12)),
  mCosY2   (scal(mY2,mY1)),
  mSinY2   (scal(mY2,mZ1))
{
}

void cOriFromBundle::AnalyseSol(const ElMatrix<double> & aMat,const Pt3dr & aPA, const Pt3dr & aPB) const
{
     Pt3dr aBRor = aMat*aPB;
     ElSeg3D aSA(Pt3dr(0,0,0),aPA);
     ElSeg3D aSB(mBase,mBase+aBRor);

     Pt3dr anI = aSA.PseudoInter(aSB);

     double aAbsA = aSA.AbscOfProj(anI);
     double aAbsB = aSB.AbscOfProj(anI);
  
     std::cout << " "
               <<  ((aAbsA >0) ? "+ " : "- ")  
               <<  ((aAbsB >0) ? "+ " : "- ")  
               << " Z  " << aPA.z << " " << aBRor.z  << " " << anI.z << "\n";
}
void cOriFromBundle::AnalyseSol(const ElMatrix<double> & aMat) const
{
     AnalyseSol(aMat,mDir1A,mDir1B);
     AnalyseSol(aMat,mDir2A,mDir2B);
     AnalyseSol(aMat,mDir3A,mDir3B);

     ElRotation3D aR(Pt3dr(0,0,0),aMat,true);

     std::cout << "TETA " << aR.teta01() << " " << aR.teta02() <<  " " << aR.teta12() << "\n\n";
}




ElMatrix<double> cOriFromBundle::CalculRot(const cOFB_Sol1S1T & aSol,bool Show)
{
   // R [mDir1B ...] = [mV1 ...]

    ElMatrix<double>  aMatB =     MatFromCol(mDir1B,mDir2B,mDir3B);
    ElMatrix<double> aMatBinA =   MatFromCol(aSol.mV1,aSol.mV2,aSol.mV3);
    ElMatrix<double> aMat = aMatBinA * gaussj(aMatB);

// std::cout << mDir1B << mDir2B << mDir3B << "\n";
// std::cout << aSol.mV1 << aSol.mV2 << aSol.mV3 << "\n";
// std::cout << "xxxxxxxxxxxx\n";
/*
    for (int anY=0 ; anY<3 ; anY++)
    {
        for (int anX=0 ; anX<3 ; anX++)
        {
            std::cout << aMat(anX,anY) << " ";
        }
        std::cout << "\n";
    }
*/
// getchar();




    ElMatrix<double> aR = NearestRotation(aMat);
// std::cout << "yyyyyyyyyyyyyy\n";


    if (Show)
    {
        std::cout << "DET " << aR.Det() -1.0  << " Eucl " << ElMatrix<double>(3,true).L2(aR*aR.transpose())<< "\n";
        std::cout  << " SCAL " << scal(mBase^mDir1A,aR*mDir1B)  << " "
                               << scal(mBase^mDir2A,aR*mDir2B)  << " "
                               << scal(mBase^mDir3A,aR*mDir3B)  << "\n";
            
    }
    
    return aR;
}




void  cOriFromBundle::TestTeta(double aT1,int aSign0,int aSign1)
{
static int aCpt =0 ; aCpt++;
// std::cout << "Test Teta " << aCpt << " " << mVSols.size() << "\n";
bool Bug = false;// (aCpt==67);

    cOFB_Sol1T  aRes;
    aRes.mTeta = aT1;
    // L'image de  mDir1B par la rot est dans le plan |_ a mBase et mDir1A donc
    // dans mX12 mY1
    double aC1 = cos(aT1);
    double aS1 = sin(aT1);

    Pt3dr aV1 = mX12 * aC1 + mY1 * aS1;
    // std::cout << aV1 << "\n";

    // On V2 = mX12 cos(T2) + mY2 sin (T2) et V2.V1 = mSc12 par conservation
    //   V2 = mX12 C2 +   (mCosY2 mY1 + mSinY2 m Z1) S2
    //  V1.V2  = C1 C2 + S1 S2 mCosY2 = mSc12 = cos(Teta12)

    double  aSP1 = aS1 * mCosY2;

    double aNorm = sqrt(ElSquare(aC1) + ElSquare(aSP1));
    aRes.mNorm = aNorm;
if (Bug) 
{
    std::cout  << "N " << aNorm << " S " << mSc12 << " Rrr= " << aNorm/(ElAbs(mSc12)) -1  <<  "\n";
}
    if ((aNorm> 1e-15 ) && (ElAbs(mSc12)<=(aNorm*0.999999)) )
    {
         aRes.mOK = true;
         double aA3 = atan2(aSP1/aNorm,aC1/aNorm);
         double aTeta12 = acos(mSc12/aNorm);

if (Bug) 
{
    std::cout  << "N " << aNorm << " S " << mSc12 << " T12 " << aTeta12 << " A " << aA3 << "\n";
}

         //  V1.V2 /aNorm   = cos(T2-A3) =  Sc12/ Norm = cos(Teta12)    => T2 = A3 +/- Teta12

         for (int aK=aSign0 ; aK<aSign1 ; aK++)
         {
               int aS = (aK==0) ? -1 : 1 ;
               aRes.mSols[aK].mSign = aS;
               double aT2  = aA3 + aS * aTeta12;
               double aC2 =  cos(aT2);
               double aS2 =  sin(aT2);
               Pt3dr aV2 =  mX12 * aC2 + mY2 * aS2;

               Pt3dr aZ = vunit(aV1^aV2);
               Pt3dr aY = vunit(aZ^aV1);

               Pt3dr aV3 =  aV1*mSc3X + aY*mSc3Y + aZ*mSc3Z ;



               double aS3 = scal(aV3,mDirOr3A);

               aRes.mSols[aK].mScal = aS3;
               aRes.mSols[aK].mV1 = aV1;
               aRes.mSols[aK].mV2 = aV2;
               aRes.mSols[aK].mV3 = aV3;

if (Bug) std::cout << "TTtttt " << aV1 << aV2 << aV3 << "\n";

         }
     }
     else
     {
         aRes.mOK = false;
     }
    
     mVSols.push_back(aRes);
}


void  cOriFromBundle::Solve(int aNbInit,int aDepthNonSol,int aDepthSol)
{

   for (int aK=0 ; aK <=aNbInit ; aK++)
   {
       TestTeta((2*PI*aK)/aNbInit,0,2);
   }

   for (int aK=0 ; aK<aNbInit ; aK++)
   {
      if ( mVSols[aK].mOK != mVSols[aK+1].mOK)
      {
         ExploreDichotFrontNonSol(aK,aK+1,aDepthNonSol);
      }
   }

   std::sort(mVSols.begin(),mVSols.end());
   int aNB = (int)(mVSols.size() - 1);

   for (int aK=0 ; aK<aNB ; aK++)
   {
       cOFB_Sol1T & aSols0 = mVSols[aK];
       cOFB_Sol1T & aSols1 = mVSols[aK+1];
       if (aSols0.mOK && aSols1.mOK)
       {
          for (int aKSign=0 ; aKSign<2; aKSign++)
          {
              cOFB_Sol1S1T & aS0 = aSols0.mSols[aKSign];
              cOFB_Sol1S1T & aS1 = aSols1.mSols[aKSign];
              bool ChngSgn =  ! aS0.SameSigne(aS1) ; // ((aS0.mScal>0) != (aS1.mScal>0));
              if (ChngSgn)
              {
                   tPairOFBSol aPair = ExploreDichotChSgn(aK,aK+1,aDepthSol,aKSign);
                   if (mVSols[aPair.first].mOK)
                   {
                       cOFB_Sol1S1T & aNew0 = mVSols[aPair.first].mSols[aKSign];
                       cOFB_Sol1S1T & aNew1 = mVSols[aPair.second].mSols[aKSign];
                       cOFB_Sol1S1T aPond = aNew0.Pond(aNew1);
/*
std::cout << "NUMMS " << aPair.first << " " << aPair.second << "\n";
aNew0.Show("N0:");
aNew1.Show("N1:");
aPond.Show("Pd:");
*/
                        ElMatrix<double> aR = CalculRot(aPond,false);
                        mListRots.push_back(aR);

                        AnalyseSol(aR);  
                   }
              }
          }
       }
   }
}


tPairOFBSol  cOriFromBundle::ExploreDichotChSgn(int aK0,int aK2,int aLevel,int aKSign)
{
    if (aLevel<=0) return tPairOFBSol (aK0,aK2);

    double aTetaMil = (mVSols[aK0].mTeta + mVSols[aK2].mTeta) / 2.0;
    TestTeta(aTetaMil,aKSign,aKSign+1);
    int aK1 = (int)(mVSols.size() - 1);

    if (! mVSols[aK1].mOK)
    {
           return tPairOFBSol(aK1,aK1);
    }



    if (!mVSols[aK0].mSols[aKSign].SameSigne(mVSols[aK1].mSols[aKSign]))
       return ExploreDichotChSgn(aK0,aK1,aLevel-1,aKSign);
   
    return ExploreDichotChSgn(aK1,aK2,aLevel-1,aKSign);
    // const cOFB_Sol1T * aS1 = 

}


void cOriFromBundle::ExploreDichotFrontNonSol(int aK0,int aK2,int aLevel)
{
    // std::cout << "FRONT " << aS1.mTeta << "\n";
    if (aLevel<=0) return;

    double aTetaMil = (mVSols[aK0].mTeta + mVSols[aK2].mTeta) / 2.0;
    TestTeta(aTetaMil,0,2);
    int aK1 = (int)(mVSols.size() - 1);

    if ( mVSols[aK0].mOK != mVSols[aK1].mOK)
      ExploreDichotFrontNonSol(aK0,aK1,aLevel-1);
   else
      ExploreDichotFrontNonSol(aK1,aK2,aLevel-1);
}

std::list<ElMatrix<double> >  OriFromBundle
                              (
                                 Pt3dr aBase,
                                 Pt3dr aDir1A,
                                 Pt3dr aDir2A,
                                 Pt3dr aDir3A,
                                 Pt3dr aDir1B,
                                 Pt3dr aDir2B,
                                 Pt3dr aDir3B,
                                 int aNbInit,int aDepthNonSol,int aDepthSol
                              )
{
       cOriFromBundle anOFB(aBase,aDir1A,aDir2A,aDir3A,aDir1B,aDir2B,aDir3B);
       anOFB.Solve( aNbInit,aDepthNonSol,aDepthSol);
       return anOFB.ListRots();
}


static Pt3dr P3dRand()
{
   return Pt3dr(NRrandom3(),NRrandom3(),NRrandom3());
}

void TestOriBundleBasic()
{
   ElTimer aChrono;
      for (int aT=0 ; aT<10000 ; aT++)
      {
          std::list<ElMatrix<double> > aLR = OriFromBundle(P3dRand(),P3dRand(),P3dRand(),P3dRand(),P3dRand(),P3dRand(),P3dRand(),30,5,5);
      }
   std::cout << aChrono.uval() << "\n";
}

static Pt3dr GrP3dRand(int aK,int aN)
{
   return Pt3dr::TyFromSpherique
          (
                10.0 + NRrandom3(),
                ((aK*2*PI) + NRrandom3() *0.5 ) / aN,
                0.5 + NRrandom3() * 0.1
          );
}


static Pt3dr  PInBCoord(const Pt3dr & aBase,const ElMatrix<double> & aRot,const Pt3dr & aPA)
{
  // mDirOr1A (vunit(mBase^mDir1A)),
    Pt3dr aRes =  aRot.transpose() * (aBase+aPA);

     std::cout << "PInBCoord " << scal(aBase^aPA,aRot*aRes) << "\n";
    return aRes;
}


// Test "realiste" 

void TestOriBundle()
{
  
      for (int aT=0 ; aT<10000 ; aT++)
      {
             Pt3dr aPA1 = GrP3dRand(0,3);
             Pt3dr aPA2 = GrP3dRand(1,3);
             Pt3dr aPA3 = GrP3dRand(2,3);

             Pt3dr aBase = Pt3dr(0,-1,0) + P3dRand() * 0.1;
             double aT12 = 0.1*NRrandom3();
             double aT13 = 0.1*NRrandom3();
             double aT23 = 0.1*NRrandom3();
             ElMatrix<double> aRot =  ElMatrix<double>::Rotation(aT12,aT13,aT23);


             std::cout << "TETAIN " << aT12 << " " << aT13 << " " << aT23 << "\n\n";

             Pt3dr aPB1 = PInBCoord(aBase,aRot,aPA1);
             Pt3dr aPB2 = PInBCoord(aBase,aRot,aPA2);
             Pt3dr aPB3 = PInBCoord(aBase,aRot,aPA3);

              
/*
          Pt3dr aP1 =  Pt3dr(NRrandom3(),NRrandom3(),10+NRrandom3());
          Pt3dr aP1 =  Pt3dr(NRrandom3(),NRrandom3(),10+NRrandom3());
*/
            std::list<ElMatrix<double> > aLR = OriFromBundle(aBase,aPA1,aPA2,aPA3,aPB1,aPB2,aPB3,30000,10,10);
            std::cout << "================== " << aT << "\n"; getchar();
      }
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
