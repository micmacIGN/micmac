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



// static bool BUG= false;

void ShowMatr(const char * mes, ElMatrix<REAL> aMatr)
{
     cout << mes <<  "\n";
     for(INT y = 0; y< aMatr.ty() ; y++)
     {
         for(INT x = 0; x< aMatr.tx() ; x++)
            printf (" %8d",INT(aMatr(x,y)*1e4));
         printf("\n");
     }
     cout << "\n";
}


// Resoudre par moindre L1  la matrice essentielle
// dans le plan tangent a la sphere en aVect (ie le plan
// orthogonal a aVect), le resultat est norme

/*
    On optimise sous la contrainte
              aVect . Sol = 1;


      _
      \
      /_   aVect[K] Sol[K] = 1

       Pour ceci on choisit kMax, la valeur max de aVect, et on resoud
      un probleme a 8 parametres en substituant partout l'inconnue
         aVect[kMax] par

                   _
                  \
              (1- /_ aVect[K]  Sol[K])
              _______________________
                     aVect[kMax]



*/


REAL ElPackHomologue::MatriceEssentielle(cGenSysSurResol & aSys,double *  aVect,REAL   EcartForPond)
{
      aSys.GSSR_Reset(false);

      INT kMax = 0;
      REAL VMax = ElAbs(aVect[0]);

      for (INT aK = 0 ; aK < 9 ; aK++)
      {
          REAL V = ElAbs(aVect[aK]);
          if (V>VMax)
          {
             VMax = V;
             kMax = aK;
          }
      }

      double Pds[9];
      double PRed[8];


      for (iterator it=begin(); it!=end() ; it++)
      {
           double x1 = it->P1().x ;
           double y1 = it->P1().y ;
           double x2 = it->P2().x ;
           double y2 = it->P2().y ;

           Pds[0] = x1 * x2;
           Pds[1] = x1 * y2;
           Pds[2] = x1 *  1;

           Pds[3] = y1 * x2;
           Pds[4] = y1 * y2;
           Pds[5] = y1 *  1;

           Pds[6] =  1 * x2;
           Pds[7] =  1 * y2;
           Pds[8] =  1 *  1;



           INT kInsert = 0;
           for (INT aK=0 ; aK<9 ; aK++)
           {
               if (aK != kMax)
               {
                  PRed[kInsert] = Pds[aK] - Pds[kMax] * aVect[aK]/aVect[kMax];
                  kInsert++;
               }
           }

	   REAL Cste = - Pds[kMax] / aVect[kMax];

	   aSys.GSSR_AddNewEquation(it->Pds(),PRed,Cste,0);
      }

      Im1D_REAL8  aSolRed = aSys.GSSR_Solve(0);


     // Une fois trouve les K != kMax, on utilise l'equation pour determiner kMax
     {
          double aSol[9];
          REAL aNormSol = 0.0;

          INT kInsert =0;
          aSol[kMax] = 1;
          for (INT aK =0 ; aK< 9 ; aK++)
          {
              if (aK != kMax)
              {
                  aSol[aK] = aSolRed.data()[kInsert];
                  aSol[kMax] -= aSol[aK] * aVect[aK];
                  aNormSol += ElSquare(aSol[aK] );

                  kInsert++;
               }
         }

         aSol[kMax] /= aVect[kMax];
         aNormSol += ElSquare(aSol[kMax]);
         aNormSol = sqrt(aNormSol);


         for (INT aK=0 ; aK<9 ; aK++)
             aVect[aK] = aSol[aK] / aNormSol;
     }

     REAL ResiduTot = 0;
     for (iterator it=begin(); it!=end() ; it++)
     {
          double x1 = it->P1().x ;
          double y1 = it->P1().y ;
          double x2 = it->P2().x ;
          double y2 = it->P2().y ;

          REAL aResidu =     aVect[0] * x1 * x2
                           + aVect[1] * x1 * y2
                           + aVect[2] * x1 *  1

                           + aVect[3] * y1 * x2
                           + aVect[4] * y1 * y2
                           + aVect[5] * y1 *  1

                           + aVect[6] *  1 * x2
                           + aVect[7] *  1 * y2
                           + aVect[8] *  1 *  1 ;


          if (EcartForPond > 0)
          {
              it->Pds() = 1 / (1 + ElSquare(aResidu/EcartForPond));
          }
          ResiduTot +=  ElAbs (aResidu);

     }


     ResiduTot /= size();
     return ResiduTot;
}



ElMatrix<REAL> ElPackHomologue::MatriceEssentielle(bool SysL2)
{
    bool Got=false;
    REAL aResiduMin = -1;
    REAL Sol[9];

    cGenSysSurResol * aSys = 0;
    if (SysL2)
       aSys =  new L2SysSurResol(8);
    else
       aSys = new SystLinSurResolu (8,size());

    for (INT aKV1 = 0 ; aKV1 <9 ; aKV1++)
    {
       double VTEST[9];
       for (INT aK=0 ; aK<9 ; aK++)
           VTEST[aK] = (aKV1== aK);
       REAL aScoreTGT =  MatriceEssentielle(*aSys,VTEST,-1);


       if ((! Got) || (aScoreTGT<aResiduMin))
       {
           // Sol = VTEST;
	   memcpy(Sol,VTEST,sizeof(VTEST));
           aResiduMin = aScoreTGT;
           Got = true;
       }
    }

    for (INT aStep = 0 ; aStep <5 ; aStep++)
    {
          MatriceEssentielle(*aSys,Sol,-1);
    }


    ElMatrix<REAL> aRes(3,3);
    for (int aKy =0 ;aKy<3; aKy++)
    {
       for (int aKx =0 ;aKx<3; aKx++)
       {
          aRes(aKx,aKy) = Sol[aKx+3*aKy];
       }
    }

    delete aSys;

    return aRes;
}


REAL ElPackHomologue::SignInters
     (
        const ElRotation3D & aRot1to2,
	INT &          NbP1,
	INT &          NbP2
     )
{
     NbP1 = 0;
     NbP2 = 0;

     REAL aSomD = 0.0;
     for (iterator it=begin(); it!=end() ; it++)
     {
          Pt3dr aRay1(it->P1().x,it->P1().y,1);
          Pt3dr aRay2(it->P2().x,it->P2().y,1);
	  Pt3dr C0(0,0,0);

	  ElSeg3D Seg1(aRot1to2.ImAff(C0),aRot1to2.ImAff(aRay1));
	  ElSeg3D Seg2(C0,aRay2);

	  Pt3dr   aI2 = Seg1.PseudoInter(Seg2);
	  Pt3dr   aI1 = aRot1to2.ImRecAff(aI2);

	  aSomD += Seg1.DistDoite(aI2) +  Seg2.DistDoite(aI2);

	  NbP1 += (aI1.z>0) ? 1 : - 1;
	  NbP2 += (aI2.z>0) ? 1 : - 1;
     }

     return aSomD / size();
}


std::list<ElRotation3D> ElPackHomologue::MepRelStd(REAL LBase,bool SysL2)  // Coord1 -> Coord2
{
   ElMatrix<REAL> aMEss = MatriceEssentielle(SysL2);

   ElMatrix<REAL> aSvd1(3,3),aDiag(3,3),aSvd2(3,3);
   svdcmp_diag(aMEss,aSvd1,aDiag,aSvd2,true);



   aSvd1.self_transpose();

   std::list<ElRotation3D> aRes;
   for (INT sign = -1; sign <=1 ; sign += 2)
      for (INT kTeta = 0; kTeta<2 ; kTeta++)
      {
           REAL aTeta2 = PI * (kTeta+0.5);

           ElMatrix<REAL> aR1 = aSvd1;
           ElMatrix<REAL> aR2 =  ElMatrix<REAL>::Rotation(0,0,aTeta2)
		               * aSvd2;

           ElMatrix<REAL> aR2T = aR2.transpose();


           aRes.push_back(ElRotation3D(aR2T*Pt3dr(LBase*sign,0,0),aR2T*aR1,true));
      }
   return aRes;
}

ElRotation3D ElPackHomologue::MepRelPhysStd(REAL LBase,bool SysL2)
{
    std::list<ElRotation3D> aLRot = MepRelStd(LBase,SysL2);

    const ElRotation3D * aRes = &(aLRot.front());
    INT aScMin = -20 * size();

    for
    (
         std::list<ElRotation3D>::iterator it=aLRot.begin();
         it!=aLRot.end() ;
         it++
    )
    {
       INT aNb1,aNb2;
       SignInters(*it,aNb1,aNb2);
       INT aScore = aNb1+aNb2;
       if (aScore > aScMin)
       {
           aScMin = aScore;
	   aRes = &(*it);
       }
    }

    return *aRes;
}



/***********************************************/
/***********************************************/
/***********************************************/
/***********************************************/



cResMepRelCoplan::cResMepRelCoplan()
{
}

cElemMepRelCoplan & cResMepRelCoplan::BestSol()
{
   ELISE_ASSERT(mVElOk.size()!=0," cResMepRelCoplan::BestSol");
   return mVElOk[0];
}

const std::vector<cElemMepRelCoplan> & cResMepRelCoplan::VElOk() const
{
	return mVElOk;
}


class cCmpEMRCPtr
{
     public :
         bool operator()(const cElemMepRelCoplan & p1,const cElemMepRelCoplan & p2)
         {
             return p1.AngTot() < p2.AngTot();
         }
};

void cResMepRelCoplan::AddSol(const cElemMepRelCoplan & anEl)
{
   mLElem.push_front(anEl);

//std::cout << "OK MECOPLAN " << anEl.PhysOk() << "\n";
//anEl.Show();

   if (anEl.PhysOk())
   {
       mVElOk.push_back(anEl);
       cCmpEMRCPtr aCmp;
      std::sort(mVElOk.begin(),mVElOk.end(),aCmp);
   }
   mLRot.push_front(anEl.Rot());
}


const std::list<ElRotation3D> &  cResMepRelCoplan::LRot() const {return mLRot;}


Pt3dr cElemMepRelCoplan::Norm() const
{
   return mNorm;
}

static double aEpsPl = 1e-2;
static double aEpsHom = 5e-2;


cElHomographie cElemMepRelCoplan::HomCam2Plan()
{
   ElPackHomologue aPack;
   ElRotation3D  aRE2P = Plan().CoordPlan2Euclid().inv();

   double aResidu = 0;
   for (int aDx=-1; aDx<=1; aDx++)
   {
       for (int aDy=-1; aDy<=1; aDy++)
       {
           Pt2dr aPIm(aDx*aEpsHom,aDy*aEpsHom);
           Pt3dr aPTer = ImCam1(aPIm);
           Pt3dr aPPlan = aRE2P.ImAff(aPTer);

           aPack.Cple_Add(ElCplePtsHomologues(aPIm,Pt2dr(aPPlan.x,aPPlan.y)));

           aResidu += ElAbs(aPPlan.z);

       }
   }

   cElHomographie aRes(aPack,true);
   for (ElPackHomologue::iterator itC=aPack.begin(); itC!=aPack.end(); itC++)
   {
        double aD = euclid(aRes.Direct(itC->P1()) -itC->P2());
        aResidu += aD;
   }
   // std::cout << "RRRRR " << aResidu << "\n";
   if (aResidu>=1e-4)
   {
      std::cout << "\nRESIDU " << aResidu << "\n";
      ELISE_ASSERT(aResidu<1e-4,"Incoherence in cElemMepRelCoplan::HomCam2Plan");
   }

   return aRes;
}


Pt3dr cElemMepRelCoplan::ImCam1 (Pt2dr aP1)
{
     Pt3dr R1(aP1.x,aP1.y,1);
     ElSeg3D S1(Pt3dr(0,0,0),R1);

     Pt2dr aP2 = mHom.Direct(aP1);
     Pt3dr R2(aP2.x,aP2.y,1);

     ElSeg3D S2(ToR1(Pt3dr(0,0,0)),ToR1(R2));
     Pt3dr   aI = S1.PseudoInter(S2);

     return aI;
}

cElPlan3D  cElemMepRelCoplan::Plan() const
{
   return cElPlan3D(mP0,mP1,mP2);
}

const Pt3dr & cElemMepRelCoplan::P0() const { return mP0; }
const Pt3dr & cElemMepRelCoplan::P1() const { return mP1; }
const Pt3dr & cElemMepRelCoplan::P2() const { return mP2; }

Pt3dr cElemMepRelCoplan::ImCam2 (Pt2dr aP2)
{
     Pt2dr aP1 = mHomI.Direct(aP2);
     Pt3dr R1(aP1.x,aP1.y,1);
     ElSeg3D S1(Pt3dr(0,0,0),R1);

     Pt3dr R2(aP2.x,aP2.y,1);

     ElSeg3D S2(ToR1(Pt3dr(0,0,0)),ToR1(R2));
     Pt3dr   aI = S1.PseudoInter(S2);

     return aI;
}

double cElemMepRelCoplan::DPlan2()
{
   // return euclid(ToR1(Pt3dr(0,0,0)-ImCam2 (Pt2dr(0,0))));
   return euclid(ToR1(Pt3dr(0,0,0))-ImCam2 (Pt2dr(0,0)));
}


Pt3dr cElemMepRelCoplan::ToR1(Pt3dr aP2) const {return mRot.ImRecAff(aP2);}

REAL cElemMepRelCoplan::AngleNormale(Pt3dr aP)
{
   Pt3dr aV = aP-mCZCMur;
   REAL aS = -scal(mNorm,aV)/euclid(aV);
   return ElAbs(acos(ElMax(-1.0,ElMin(1.0,aS))));
}

REAL cElemMepRelCoplan::DPlan() const
{
	return euclid(mP0);
}


cElemMepRelCoplan::cElemMepRelCoplan
(
     const cElHomographie & aHom,
     const ElRotation3D &   aRot
) :
  mHom         (aHom),
  mHomI        (mHom.Inverse()),
  mRot         (aRot),
  mP0          (ImCam1(Pt2dr(0,0))),
/*
  mP1          (ImCam1(Pt2dr(1,0))),
  mP2          (ImCam1(Pt2dr(0,1))),
*/
  mP1          (ImCam1(Pt2dr(aEpsPl,0))),
  mP2          (ImCam1(Pt2dr(0,aEpsPl))),

  mNorm        (vunit( (mP1-mP0) ^ (mP2-mP0))),
  mCZCC2       (mHom.Direct(Pt2dr(0,0))/2.0),
  mCZCC1       (mHomI.Direct(mCZCC2)),
  mCZCMur      (ImCam1(mCZCC1)),
  mCOptC1      (0,0,0),
  mCOptC2      (ToR1(Pt3dr(0,0,0))),
  mProfC1      (mCZCMur.z),
  mProfC2      (mRot.ImAff(mCZCMur).z),
  mAng1        (AngleNormale(mCOptC1)),
  mAng2        (AngleNormale(mCOptC2)),
  mAngTot      (mAng1+mAng2),
  mDEuclidP    (ElAbs(scal(mNorm,mP0)))
{

     // VERIF 
/*
     if (1)
     {
          for (int aK=0; aK<10 ; aK++)
          {
              Pt2dr aI = ImCam1(aP1);
              // std::cout << "AAAAAAAAAAAAAaa\n";
              cElPlan3D aPl(mP0,mP1,mP2);

              Pt3dr aQ1 = aPl.Inter(S1);
              Pt3dr aQ2 = aPl.Inter(S2);

              double aDist = euclid(aI-aQ1) + euclid(aI-aQ2);
              std::cout << "BBBBBB " << aDist << "\n";
          }
     }
*/

}

void cElemMepRelCoplan::TestPack(const ElPackHomologue & aPack) const
{
    int aNbTot =0 , aNbPos1 = 0 , aNbPos2 = 0;
    for 
    (
         ElPackHomologue::const_iterator itP = aPack.begin();
         itP != aPack.end();
         itP++
    )
    {
         Pt2dr aP1 = itP->P1();
         Pt3dr R1(aP1.x,aP1.y,1);
         ElSeg3D S1(Pt3dr(0,0,0),R1);

         Pt2dr aP2 = itP->P2();
         Pt3dr R2(aP2.x,aP2.y,1);
         ElSeg3D S2(ToR1(Pt3dr(0,0,0)),ToR1(R2));


          Pt3dr   aI1 = S1.PseudoInter(S2);
          Pt3dr   aI2 = mRot.ImAff(aI1);

          aNbTot++;
          aNbPos1 += aI1.z > 0 ;
          aNbPos2 += aI2.z > 0 ;
    }

    std::cout << "Test pack " << aNbTot << " Pt1 " << aNbPos1 << "  Pt2 " << aNbPos2 << "\n";
}


double cElemMepRelCoplan::TestSol() const
{

   double aRes = 0;
   Pt3dr aV1 = (mP1-mP0) ;
   Pt3dr aV2 = (mP2-mP0) ;
   for (int aX= -5; aX <=5 ; aX+=10)
   {
      for (int aY= -5; aY <=5 ; aY+=10)
      {
          Pt3dr aQ1 = mP0 + aV1 * aX + aV2 * aY;
          Pt3dr aQ2 = mRot.ImAff(aQ1);

          Pt2dr aI1 (aQ1.x/aQ1.z, aQ1.y/aQ1.z);
          Pt2dr aI2 (aQ2.x/aQ2.z, aQ2.y/aQ2.z);

          aRes += euclid(aI2,mHom.Direct(aI1));
      }
   }
   return aRes;
}

void cElemMepRelCoplan::Show() const
{
     cout << "SIGNES " << mProfC1 << " "
                       << mProfC2 << " "
                       << " ANGLE : " << mAngTot <<  "\n";
     std::cout << "Check-Verif " << TestSol() << " A1 " << mAng1 << " A2 " << mAng2  << "\n";
     cout << mRot.tr()
          << " " << mRot.teta01()
          << " " << mRot.teta02()
          << " " << mRot.teta12() << "\n\n";
}
bool cElemMepRelCoplan::PhysOk() const
{
   return (mProfC1>0) && (mProfC2 > 0);
}

REAL  cElemMepRelCoplan::AngTot() const {return mAngTot;}
REAL  cElemMepRelCoplan::Ang1() const {return mAng1;}
REAL  cElemMepRelCoplan::Ang2() const {return mAng2;}

const ElRotation3D & cElemMepRelCoplan::Rot() const {return mRot;}


double cElemMepRelCoplan::DistanceEuclid() const
{
   return mDEuclidP;
}

/*
    Principe :

        1- On calcule l'homographie H qui envoie les point2 en point 1



        2- Soit le repere epipolaire (*) tel que le plan definissant la
        correspondance intersecte le plan commun des camera selon
        une droite ortohogonale a la direction des axes optiques:

        Dans ce repere , l'homographie prend la forme

                      1 0 0
             H~ =     0 A B
                      0 0 1


       On cherche a resoudre


       (*) Les epipolaire utilise ici correspondent a une convention
       dif de celle habituelle puisque ce sont les colonne (x=cste)
       qui sont invariantes. Quand je m'en suis rendu compte,
       tout mon raisonnemnt etait deja base sur la forme ci dessous
       de H~.

*/



// Diff Ensembliste
static REAL SetDiff(REAL V1,REAL V2,REAL U1,REAL U2)
{
	return ElAbs(ElMax(V1,V2)-(ElMax(U1,U2)))
	     + ElAbs(ElMin(V1,V2)-(ElMin(U1,U2)));
}

cResMepRelCoplan ElPackHomologue::MepRelCoplan(REAL LBase,bool HomEstL2)
{
    static tPairPt aP00(Pt2dr(0,0),Pt2dr(0,0));

    return MepRelCoplan(LBase,cElHomographie(*this,HomEstL2),aP00);
}


cResMepRelCoplan ElPackHomologue::MepRelCoplan(REAL LBase,cElHomographie aHom12,const tPairPt & aP00)

// cResMepRelCoplan FFFFMepRelCoplan(REAL LBase,cElHomographie aHom12,const tPairPt & aP00)
{
   cResMepRelCoplan aRes;
   //  cElHomographie aHom12(*this,HomEstL2);
   cElHomographie aHom21 = aHom12.Inverse();
   ElMatrix<REAL>  aMHom = aHom21.MatCoordHom();


   ElMatrix<REAL> aSvdInit1(3,3),aDiagInit(3,3),aSvdInit2(3,3);
   svdcmp_diag(aMHom,aSvdInit1,aDiagInit,aSvdInit2,true);
   aSvdInit1.self_transpose();

   ElMatrix<REAL> aRotSpec =  ElMatrix<REAL>::transposition(3,1,2);
   aRotSpec *=-1;


   ElMatrix<REAL> aPerm(3,3);
   ElMatrix<REAL> aPermT(3,3);


   for (INT aK=0 ; aK<3 ; aK++)
   {
       aPerm.set_shift_mat_permut(aK);
       aPermT.set_shift_mat_permut(-aK);

       ElMatrix<REAL> aSvd1 = aPermT * aSvdInit1 ;
       ElMatrix<REAL> aDiag = aPermT    * aDiagInit * aPerm;
       ElMatrix<REAL> aSvd2 = aPermT    * aSvdInit2;

       if (ElAbs(aDiag(0,0)) > 1e-5)
       {
           REAL aVP2 = aDiag(1,1) / aDiag(0,0);
           REAL aVP3 = aDiag(2,2) / aDiag(0,0);

           REAL A =  aVP2 * aVP3;

           REAL B2 = ElSquare(aVP2) +  ElSquare(aVP3) - ElSquare(A) - 1;

           if (B2 > -1e-4)
           {
               B2 = ElMax(0.0,B2);

               for (INT signB=-1; signB<=1 ; signB +=2)
               {
                   REAL B = signB*sqrt(B2);

                   ElMatrix<REAL> aH2Red(2,true);
                   aH2Red(0,0) = A;
                   aH2Red(1,0) = B;

                   ElMatrix<REAL> aS2Vred1(2,2),aDiag2(2,2),aS2Vred2(2,2);
                   svdcmp_diag(aH2Red,aS2Vred1,aDiag2,aS2Vred2,true);
                   aS2Vred1.self_transpose();
		   if (   SetDiff(aVP2,aVP3,aDiag2(0,0),aDiag2(1,1))
			>  SetDiff(aVP2,aVP3,-aDiag2(0,0),-aDiag2(1,1))
		      )
		   {
			    aDiag2 *= -1;
			    aS2Vred1 *= -1;
		   }


		    ElMatrix<REAL> aS3Vred1 = aS2Vred1.ExtensionId(1,0);
		    ElMatrix<REAL> aS3Vred2 = aS2Vred2.ExtensionId(1,0);

		    if ((aVP2>aVP3) != (aDiag2(0,0)>aDiag2(1,1)))
		    {
			    aS3Vred1 = aRotSpec * aS3Vred1;
			    aS3Vred2 = aRotSpec * aS3Vred2;
		    }

		    ElMatrix<REAL> aR1  = aS3Vred1.transpose() * aSvd1;
                    for (INT kTeta = 0; kTeta<2 ; kTeta++)
                    {
                         REAL aTeta = PI * (kTeta);
                         ElMatrix<REAL> aRTeta =
				        ElMatrix<REAL>::Rotation(0,aTeta,0);
		         ElMatrix<REAL> aR2  = aRTeta * aS3Vred2.transpose() * aSvd2;
                         ElMatrix<REAL> aR2T = aR2.transpose();

                         for (INT sign = -1; sign <=1 ; sign += 2)
		         {
                            ElRotation3D aR(aR2T*Pt3dr(0,LBase*sign,0),aR2T*aR1,true);
			    cElemMepRelCoplan anEl(aHom12,aR);
                            aRes.AddSol(anEl);
			 }
		    }
               }
           }

       }
   }

   return aRes;
}


/***********************************************/
/***********************************************/
/***********************************************/
/***********************************************/

double ElCplePtsHomologues::Profondeur(const ElRotation3D & aR) const
{
   // Tous les calculs se font en coord camera 2
   Pt3dr aC1 = aR.ImAff(Pt3dr(0,0,0));
   Pt3dr aDir1 =  aR.ImVect(Pt3dr(P1().x,P1().y,1));
   ElSeg3D aS1(aC1,aC1+aDir1);


   Pt3dr aC2 = Pt3dr(0,0,0);
   Pt3dr aDir2 =  Pt3dr(P2().x,P2().y,1);
   ElSeg3D aS2(aC2,aC2+aDir2);

   Pt3dr anI = aS1.PseudoInter(aS2);

   // double aD1 = euclid(aC1-anI);
   // double aD2 = euclid(aC2-anI);

    Pt3dr aV1 = aR.ImVect(Pt3dr(0,0,1));
    Pt3dr aV2(0,0,1);

    double aD1 = scal(aV1,anI-aC1);
    double aD2 = scal(aV2,anI-aC2);

   // std::cout /*<< aC1 << anI */ << aD1 << "   " << aD2 <<  " " << aD1/aD2 << "\n";

   return (aD1+aD2) / 2.0;
}

double ElPackHomologue::Profondeur(const ElRotation3D & aR) const
{
    double aSPds=0.0;
    double aSProf=0.0;

    for (const_iterator itP =begin() ; itP!=end(); itP++)
    {
        aSPds += itP->Pds();
	aSProf += itP->ToCple().Profondeur(aR) *  itP->Pds();
    }

    return aSProf / aSPds;
}

void ElPackHomologue::SetProfondeur(ElRotation3D & aR,double aPCible) const
{
   double aPInit = Profondeur(aR);
   aR.tr() = aR.tr() * (aPCible/aPInit);
}

static tParamAFocal aNoPAF;

void  ElPackHomologue::ProfMedCam2(std::vector<double> & VProf,const ElRotation3D & aR) const
{
    VProf.clear();
    CamStenopeIdeale aCam1(true,1.0,Pt2dr(0,0),aNoPAF);
    CamStenopeIdeale aCam2(true,1.0,Pt2dr(0,0),aNoPAF);
    aCam2.SetOrientation(aR);

    Pt3dr aC2 = aCam2.VraiOpticalCenter();

    for (const_iterator itP =begin() ; itP!=end(); itP++)
    {
        Pt3dr aPTer = aCam1.PseudoInter(itP->P1(),aCam2,itP->P2());
        VProf.push_back
        (
              scal(aPTer-aC2,aCam2.F2toDirRayonR3(itP->P2()))
        );
    }

}

double ElPackHomologue::QuickDistInter
       (
           const ElRotation3D & aR,
           const std::vector<double> & VProf
       ) const
{
    CamStenopeIdeale aCam1(true,1.0,Pt2dr(0,0),aNoPAF);
    CamStenopeIdeale aCam2(true,1.0,Pt2dr(0,0),aNoPAF);
    aCam2.SetOrientation(aR);


    ElRotation3D aR2 = aR.inv();
    ElMatrix<double> aMat=aR2.Mat();
    Pt3dr aC0,aC1,aC2;
    aMat.GetCol(0,aC0);
    aMat.GetCol(1,aC1);
    aMat.GetCol(2,aC2);

    Pt3dr aTr = aR2.tr();


    double aSPds=0.0;
    double sDist=0.0;
    int aK = 0;
    for (const_iterator itP =begin() ; itP!=end(); itP++)
    {
       double aPds = itP->Pds();
       double aLambda =  VProf[aK];
        
        Pt2dr aPIm1 = itP->P1();
        Pt2dr aPIm2 = itP->P2();
        Pt3dr aRay2 = aC0 * aPIm2.x + aC1 * aPIm2.y + aC2;

       // std::cout << aRay2 << "\n"; 
       // std::cout<< aCam2.F2toDirRayonR3(aPIm2) << "\n";

        Pt3dr aPTer1 = aTr + aRay2 * (aLambda *0.99);
        Pt3dr aPTer2 = aTr + aRay2 * (aLambda *1.01);

        Pt2dr aProj1(aPTer1.x/aPTer1.z,aPTer1.y/aPTer1.z);
        Pt2dr aProj2(aPTer2.x/aPTer2.z,aPTer2.y/aPTer2.z);
         
         Pt2dr aV12  (aProj1.y-aProj2.y,aProj2.x-aProj1.x);

         double aD2 =  ElAbs(scal(aV12,aProj1-aPIm1) / euclid(aV12));
        

/*
        SegComp aSeg(aProj1,aProj2);
        double aD3 = aSeg.dist_droite(aPIm1);
       std::cout <<  aD2/aD3 << "\n";




        Pt3dr aPTer = aCam1.PseudoInter(itP->P1(),aCam2,itP->P2());
        Pt2dr aQ1 = aCam1.R3toF2(aPTer);
        Pt2dr aQ2 = aCam2.R3toF2(aPTer);
        double aDist = euclid(aQ1,itP->P1())+euclid(aQ2,itP->P2());

        std::cout << "RATIO " <<  aDist / (2*euclid(aQ1,itP->P1())) << "\n";
*/


        aSPds += aPds;
        sDist += aPds*aD2;

        aK++;
         // std::cout << aDist << "\n";
    }

    return sDist / aSPds;
}


double ElPackHomologue::AngularDistInter(const ElRotation3D & aR) const
{
    CamStenopeIdeale aCam1(true,1.0,Pt2dr(0,0),aNoPAF);
    CamStenopeIdeale aCam2(true,1.0,Pt2dr(0,0),aNoPAF);
    aCam2.SetOrientation(aR);

    double aSPds=0.0;
    double sDist=0.0;

    for (const_iterator itP =begin() ; itP!=end(); itP++)
    {
        Pt3dr aPTer = aCam1.PseudoInter(itP->P1(),aCam2,itP->P2());
        Pt2dr aQ1 = aCam1.R3toF2(aPTer);
        Pt2dr aQ2 = aCam2.R3toF2(aPTer);
        double aPds = itP->Pds();
        double aDist = euclid(aQ1,itP->P1())+euclid(aQ2,itP->P2());

         aDist = aDist / (1+aDist);

        aSPds += aPds;
        sDist += aPds*aDist;

         // std::cout << aDist << "\n";
    }

    return sDist / aSPds;
}



ElRotation3D ElPackHomologue::MepRelGen(REAL LongBase,bool L2,double& aD,bool  Optimize)
{

/*
static int aCpt=0; aCpt++;
std::cout << "pckPCK----NB " << size() << " CPT " << aCpt << "OPT " << Optimize  << "\n";
BUG=(aCpt==10);
*/

   ElRotation3D aR1 = MepRelPhysStd(LongBase,L2);

   cResMepRelCoplan aMRC = MepRelCoplan(LongBase,L2);


   ElRotation3D aR2 = aMRC.BestSol().Rot();

   double aD1 = AngularDistInter(aR1);
   double aD2 = AngularDistInter(aR2);

// std::cout << "COST-Ess " << aD1 << " Cost Co plan " << aD2 << "\n";

   ElRotation3D aRes = (aD1<aD2) ? aR1 : aR2;
   if (Optimize)
       aRes = OptimiseMEPRel(aRes);
   aD = AngularDistInter(aRes);
 
   return aRes;
}

ElRotation3D ElPackHomologue::MepRelGen(REAL LongBase,bool L2,double& aD)
{
   return MepRelGen(LongBase,L2,aD,true);
}

ElRotation3D ElPackHomologue::MepRelGenSsOpt(REAL LongBase,bool L2,double& aD)
{
   return MepRelGen(LongBase,L2,aD,false);
}



/****************************************************************/
/*                                                              */
/*            OptimiseMEPRel                                    */
/*                                                              */
/****************************************************************/

class cIterOptimiseMEPRel : public FoncNVarND<double>
{
    public :
      cIterOptimiseMEPRel
      (
           const ElPackHomologue & aPack ,
           const ElRotation3D &    aRInit
      );
      ElRotation3D  Param2Rot(const double *);
    private :
      double ValFNV(const double *);

      const ElPackHomologue & mPack;
      Pt3dr mU;
      double mNorm;
      Pt3dr mV;
      Pt3dr mW;
      ElMatrix<double> mMatr;
      std::vector<double> mVProf;
};

double cIterOptimiseMEPRel::ValFNV(const double * aParam)
{
   return mPack.QuickDistInter(Param2Rot(aParam),mVProf);
}

cIterOptimiseMEPRel::cIterOptimiseMEPRel
(
     const ElPackHomologue & aPack ,
     const ElRotation3D &    aRInit
)   :
    FoncNVarND<double> (5),
    mPack   (aPack),
    mU      (aRInit.tr()),
    mNorm   (euclid(mU)),
    mMatr   (aRInit.Mat())
{
    mV  =  OneDirOrtho(mU);  // mV est unitaire a ce stade
    mW =  mU ^ mV;
    mV = mV * mNorm;
    mPack.ProfMedCam2(mVProf,aRInit);

    
}


ElRotation3D  cIterOptimiseMEPRel::Param2Rot(const double * aP)
{
   double aF = 0.05;
   Pt3dr aTr =    mU*cos(sqrt(ElSquare(aP[0]*aF)+ElSquare(aP[1]*aF)))
                 + mV*sin(aP[0]*aF) 
                 + mW*sin(aP[1]*aF);

   // std::cout << "XY " << mMatr.tx() << " " << mMatr.ty() << "\n";
   return ElRotation3D
          (
              vunit(aTr) * mNorm,
              mMatr * ElMatrix<double>::Rotation(aP[2]*aF,aP[3]*aF,aP[4]*aF),
              true
          );
}


ElRotation3D ElPackHomologue::OptimiseMEPRel(const ElRotation3D & aRot)
{
   double aParam[5] = {0,0,0,0,0};
   cIterOptimiseMEPRel aIOMR(*this,aRot);

   aIOMR.powel(aParam,1e-7,200);
   return aIOMR.Param2Rot(aParam);
}

/*
 * DIST 1 : 0.00152956 6.65325e-05
 * DIST 1 : 9.04631e-05 6.58614e-05
 *
 *
 * DIST 1 : 0.00152956 6.65318e-05
 * DIST 1 : 9.04631e-05 6.58614e-05
 * ERR = 0.563453
 *
 *
 */

/****************************************************************/
/*                                                              */
/*            cResolvAmbiBase                                   */
/*                                                              */
/****************************************************************/

cResolvAmbiBase::cResolvAmbiBase
(
	const ElRotation3D &  aR0,
        const ElRotation3D &  aR1
)   :
    mC0  (aR0.ImAff(Pt3dr(0,0,0))),
    mV01 (aR1.ImAff(Pt3dr(0,0,0)) -mC0),
    mR1  (aR1)

{
}


void cResolvAmbiBase::AddHom(const ElPackHomologue & aH12,const ElRotation3D & aR2)
{
   Pt3dr aC2 = aR2.ImAff(Pt3dr(0,0,0));
   for
   (
       ElPackHomologue::const_iterator itP=aH12.begin();
       itP != aH12.end();
       itP++
   )
   {
       Pt3dr aDir1 = mR1.ImVect(PZ1(itP->P1()));
       Pt3dr aDir2 = aR2.ImVect(PZ1(itP->P2()));

       double aDCst = Det(mC0-aC2,aDir1,aDir2);
       double aDLambda = Det(mV01,aDir1,aDir2);

       // std::cout << "LAMBDA " << -aDCst/aDLambda << "\n";

       mLambdas.push_back(-aDCst/aDLambda);
   }
}

double cResolvAmbiBase::SolveBase()
{
  int aSz = mLambdas.size();
  ELISE_ASSERT(aSz!=0,"cResolvAmbiBase::SolveBase :: vide!!");

  std::sort(mLambdas.begin(), mLambdas.end());

  return ( mLambdas[aSz/2] +  mLambdas[(aSz-1)/2] ) /2.0;  // Mediane
}


ElRotation3D cResolvAmbiBase::SolOrient(double & aLambda)
{
   aLambda = SolveBase();
   Pt3dr aCL =  mC0 + mV01*aLambda;
   const ElMatrix<double> & aM = mR1.Mat();

   return ElRotation3D(aCL,aM,true);
}

double SomEcartDist
       (
           const ElMatrix<REAL> &    aMat,
           const std::vector<Pt3dr>& aVDir1,
           const std::vector<Pt3dr>& aVDir2
       )
{ 
    double aRes = 0;
    for (int aK=0 ; aK<int(aVDir1.size()) ; aK++)
    {
        aRes += euclid(aMat*aVDir1[aK]-aVDir2[aK]);
    }
    return aRes;
}


ElMatrix<REAL> ElPackHomologue::MepRelCocentrique(int aNbRansac,int aNbMaxPts) const
{
   aNbMaxPts = ElMin(aNbMaxPts,size());

   std::vector<Pt3dr> aVDir1;
   std::vector<Pt3dr> aVDir2;

   cRandNParmiQ aRand(aNbMaxPts,size());

   for (tCstIter itH=begin() ; itH!=end() ; itH++)
   {
      if (aRand.GetNext())
      {
          aVDir1.push_back(vunit(PZ1(itH->P1())));
          aVDir2.push_back(vunit(PZ1(itH->P2())));
      }
   }

   ElMatrix<REAL> aRes(3,3);
   double anEcartMin = 1e60;

   while (aNbRansac)
   {
       int aKA = NRrandom3(aVDir1.size());
       int aKB = NRrandom3(aVDir2.size());
       if (aKA!=aKB)
       {
          aNbRansac--;
          ElMatrix<REAL> aMat = ComplemRotation(aVDir1[aKA],aVDir1[aKB],aVDir2[aKA],aVDir2[aKB]);
          double anEc = SomEcartDist(aMat,aVDir1,aVDir2);
          if (anEc<anEcartMin)
          {
              anEcartMin = anEc;
              aRes = aMat;
          }
       }
   }
   return aRes;
}



/****************************************************************/
/*                                                              */
/*            cResolvAmbiBase                                   */
/*                                                              */
/****************************************************************/





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
