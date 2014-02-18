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
#include "general/all.h"
#include "MICMAC.h"
#include "ext_stl/appli_tab.h"
#include "ext_stl/tab2D_dyn.h"


namespace NS_ParamMICMAC
{

Pt2di Px2Point(int * aPx) { return Pt2di(aPx[0],0); }
int CostR2I(double aCost) { return round_ni(aCost*1e4); }


/**************************************************/
/*                                                */
/*       cGBV2_CelOptimProgDyn                         */
/*                                                */
/**************************************************/
typedef unsigned int tCost;

class cGBV2_CelOptimProgDyn
{
     public :
        cGBV2_CelOptimProgDyn() :
            mCostFinal (0)
        {
        }
        typedef enum
        {
           eAvant = 0,
           eArriere = 1
        } eSens;
        void SetCostInit(int aCost)
        {
             mCostInit = aCost;
        }
        void SetBeginCumul(eSens aSens)
        {
             mCostCum[aSens] = mCostInit;
        }
        void SetCumulInitial(eSens aSens)
        {
             mCostCum[aSens] = int(1e9) ;
        }
        void UpdateCost
             (
                 const cGBV2_CelOptimProgDyn& aCel2,
                 eSens aSens,
                 int aCostTrans
             )
        {
            ElSetMin
            (
                mCostCum[aSens],
                mCostInit+aCostTrans+aCel2.mCostCum[aSens]
            );
        }

        tCost CostPassageForce() const
        {
            return mCostCum[eAvant] + mCostCum[eArriere] - mCostInit;
        }
        tCost GetCostInit() const 
        {
              return mCostInit;
        }
        const tCost & CostFinal() const { return mCostFinal; }
        tCost & CostFinal() { return mCostFinal; }
     private :
        cGBV2_CelOptimProgDyn(const cGBV2_CelOptimProgDyn &);
        tCost   mCostCum[2];
        tCost   mCostInit;
        tCost   mCostFinal;
};

typedef  cSmallMatrixOrVar<cGBV2_CelOptimProgDyn>   tCGBV2_tMatrCelPDyn;




/**************************************************/
/*                                                */
/*                     ::                         */
/*                                                */
/**************************************************/


//  COPIE TEL QUEL DEPUIS /home/mpd/ELISE/applis/Anag/SimpleProgDyn.cpp

/*
    Soit Z dans l'intervalle ouvert I1 [aZ1Min,aZ1Max[,
    on recherche dans l'intervalle ouvert I0 [aZ0Min,aZ0Max[,
    un intervalle ferme, non vide, le plus proche possible
    [aZ+aDzMin,aZ+aDzMax].

    De plus ce calcul doit generer des connexion symetrique.

    Ex  :
        I1 = [10,30[
        I0 = [5,20[

        MaxDeltaZ = 2


        Z = 13 ->    Delta = [-2,2]   // aucune contrainte
        Z = 18 ->    Delta = [-2,1]   // Pour que ca reste dans I0
        Z = 25 ->    Delta = [-6,-6]  //  Pour que l'intersection soit non vide avec I0
        Z = 10 ->    Delta = [-5,-1]  // principe de symetrie, dans l'autre sens                                      // les points [5,9] de I0 devront etre connecte a 10

*/


static inline void ComputeIntervaleDelta
              (
                  INT & aDzMin,
                  INT & aDzMax,
                  INT aZ,
                  INT MaxDeltaZ,
                  INT aZ1Min,
                  INT aZ1Max,
                  INT aZ0Min,
                  INT aZ0Max
              )
{
      aDzMin =   aZ0Min-aZ;
      if (aZ != aZ1Min)
         ElSetMax(aDzMin,-MaxDeltaZ);

      aDzMax = aZ0Max-1-aZ;
      if (aZ != aZ1Max-1)
         ElSetMin(aDzMax,MaxDeltaZ);

       // Si les intervalles sont vides, on relie
       // les bornes des intervalles a tous les points
       if (aDzMin > aDzMax)
       {
          if (aDzMax <0)
             aDzMin = aDzMax;
          else
             aDzMax = aDzMin;
       }
}

/**************************************************/
/*                                                */
/*               cProgDynOptimiseur               */
/*                                                */
/**************************************************/


class cGBV2_TabulCost
{
    public :
      void Reset(double aCostL1,double aCostL2)
      {
         mNb=0;
	 mTabul = std::vector<int>();
	 mCostL1 = aCostL1;
	 mCostL2 = aCostL2;
      }
      inline int Cost(int aDx)
      {
          aDx = ElAbs(aDx);
          for (;mNb<=aDx; mNb++)
	  {
	      mTabul.push_back
	      (
	         CostR2I
		 (
		       mCostL1 * mNb
		     + mCostL2 * ElSquare(mNb)
		 )
	      );
              // std::cout << "COST [" << mNb << "]=" <<  mTabul.back() << "\n";
	  }
	  return mTabul[aDx];
      } 

      double           mCostL1;
      double           mCostL2;
      int              mNb;
      std::vector<int> mTabul;  
};

class cGBV2_ProgDynOptimiseur 
{
    public :
         cGBV2_ProgDynOptimiseur
         (
	   Im2D_INT2  aPxMin,
	   Im2D_INT2  aPxMax 
         );
         ~cGBV2_ProgDynOptimiseur() {}
        void Local_SetCout(Pt2di aPTer,int aPX,REAL aCost);
        void Local_SolveOpt
	     (
                  int aNbDir,
                  double aPenteMax,
	          double aRegul,
	          double aRegul_Quad
             );

          
        Im2D_INT2     ImRes() {return mImRes;}

    private :

        void BalayageOneDirection(Pt2dr aDir);
        void BalayageOneLine(const std::vector<Pt2di> & aVPt);
        void BalayageOneSens
             ( 
                   const std::vector<Pt2di> & aVPt,
                   cGBV2_CelOptimProgDyn::eSens,
                   int anIndInit,
                   int aDelta,
                   int aLimite
             );
        void SolveOneEtape(int aNbDir);

        Im2D_INT2                          mXMin;
        Im2D_INT2                          mXMax;
        Pt2di                              mSz;
        int                                mNbPx;
        Im2D_INT2                          mYMin;
        Im2D_INT2                          mYMax;
        cMatrOfSMV<cGBV2_CelOptimProgDyn>       mMatrCel;
        cLineMapRect                       mLMR;
	cGBV2_TabulCost                         mTabCost[theDimPxMax];
        // int                                mCostActu[theDimPxMax];
        int                                mMaxEc[theDimPxMax];
        eModeAggregProgDyn                 mModeAgr;
        int                                mNbDir;
	double                             mPdsProgr;

        double mCostRegul[2];              // MODIF
        double mCostRegul_Quad[2];         // MODIF


        Im2D_INT2     mImRes;
	INT2 **       mDataImRes;
         
};
static cGBV2_CelOptimProgDyn aCelForInit;



cGBV2_ProgDynOptimiseur::cGBV2_ProgDynOptimiseur
(
      Im2D_INT2  aPxMin,
      Im2D_INT2  aPxMax 
) :
  mXMin                (aPxMin),
  mXMax                (aPxMax),
  mSz                  (mXMin.sz()),
  mNbPx                (1),   // MODIF OK
  mYMin                (mSz.x,mSz.y,0),
  mYMax                (mSz.x,mSz.y,1),
  mMatrCel             (
                           Box2di(Pt2di(0,0),mSz),
                           mXMin.data(),
                           mYMin.data(),
                           mXMax.data(),
                           mYMax.data(),
                           aCelForInit
                       ),
  mLMR                 (mSz),
  mImRes               (mSz.x,mSz.y),
  mDataImRes           (mImRes.data())
{
}




void cGBV2_ProgDynOptimiseur::Local_SetCout(Pt2di aPTer,int aPX,REAL aCost)
{
    mMatrCel[aPTer][Px2Point(&aPX)].SetCostInit(CostR2I(aCost));
}

void cGBV2_ProgDynOptimiseur::BalayageOneSens
     ( 
         const std::vector<Pt2di> & aVPt,
         cGBV2_CelOptimProgDyn::eSens    aSens,
         int                        anIndInit,
         int                        aDelta,
         int                        aLimite
     )
{
//ElTimer aChrono;
//static int aCpt=0; aCpt++;
   // Initialisation des couts sur les premieres valeurs
   {
      tCGBV2_tMatrCelPDyn &  aMat0 = mMatrCel[aVPt[anIndInit]];
      const Box2di & aBox0 = aMat0.Box();

      Pt2di aP0;
      for (aP0.y = aBox0._p0.y ;  aP0.y<aBox0._p1.y; aP0.y++)
      {
          for (aP0.x = aBox0._p0.x ; aP0.x<aBox0._p1.x;aP0.x++)
          {
              aMat0[aP0].SetBeginCumul(aSens);
          }
      }
   }
   
   //double aNb=0.0;

   // Propagation
   int anI0 = anIndInit; 
   while ((anI0+ aDelta)!= aLimite)
   {
        int anI1 = anI0+aDelta;
        tCGBV2_tMatrCelPDyn &  aMat1 = mMatrCel[aVPt[anI1]];
        const Box2di & aBox1 = aMat1.Box();
        Pt2di aP1;

        // Met un cout infini aux successeurs
        for (aP1.y = aBox1._p0.y ; aP1.y<aBox1._p1.y;aP1.y++)
        {
            for (aP1.x = aBox1._p0.x ; aP1.x<aBox1._p1.x ; aP1.x++)
            {
                aMat1[aP1].SetCumulInitial(aSens);
            }
        }

        // Propage 
        tCGBV2_tMatrCelPDyn &  aMat0 = mMatrCel[aVPt[anI0]];
        const Box2di & aBox0 = aMat0.Box();
        Pt2di aP0;
        for (aP0.y=aBox0._p0.y ; aP0.y<aBox0._p1.y ; aP0.y++)
        {
            int aDyMin,aDyMax;
            ComputeIntervaleDelta
            (
                 aDyMin,aDyMax,
                 aP0.y, mMaxEc[1],
                 aBox0._p0.y,aBox0._p1.y,
                 aBox1._p0.y,aBox1._p1.y
            );
            for (aP0.x=aBox0._p0.x ;  aP0.x<aBox0._p1.x ; aP0.x++)
            {
                int aDxMin,aDxMax;
                ComputeIntervaleDelta
                (
                     aDxMin,aDxMax,
                     aP0.x, mMaxEc[0],
                     aBox0._p0.x,aBox0._p1.x,
                     aBox1._p0.x,aBox1._p1.x
                );
                //aNb +=  (1+aDyMax - aDyMin)* (1+aDxMax - aDxMin);
                cGBV2_CelOptimProgDyn & aCel0 = aMat0[aP0];

/*
std::cout << aDyMin << " " << aDyMax << " " 
          << aDxMin << " " << aDxMax << "\n";
ELISE_ASSERT((aDyMin==0) && (aDyMax==0) && (aDxMin>=-1) && (aDxMax<=1),"!!!");
*/
                for (int aDy=aDyMin ; aDy<=aDyMax; aDy++)
                {
                    for (int aDx=aDxMin ; aDx<=aDxMax; aDx++)
                    {
                        aMat1[aP0+Pt2di(aDx,aDy)].UpdateCost
                        (
                           aCel0, aSens,
                           mTabCost[0].Cost(aDx) + mTabCost[1].Cost(aDy)
                           // mCostActu[0]*ElAbs(aDx)+mCostActu[1]*ElAbs(aDy)
                        );
                    }
                }
            }
        }
        anI0 = anI1;
   }
}

void cGBV2_ProgDynOptimiseur::BalayageOneLine(const std::vector<Pt2di> & aVPt)
{
    BalayageOneSens(aVPt,cGBV2_CelOptimProgDyn::eAvant,0,1,(int) aVPt.size());
    BalayageOneSens(aVPt,cGBV2_CelOptimProgDyn::eArriere,(int) (aVPt.size())-1,-1,-1);

    for (int aK=0 ; aK<int(aVPt.size()) ; aK++)
    {
        tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[aVPt[aK]];
        const Box2di &  aBox = aMat.Box();
        Pt2di aP;

        tCost aCoutMin = tCost(1e9);

        for (aP.y = aBox._p0.y ; aP.y<aBox._p1.y; aP.y++)
        {
            for (aP.x = aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
            {
                ElSetMin(aCoutMin,aMat[aP].CostPassageForce());
// std::cout <<  aMat[aP].CostPassageForce() << " " << aMat[aP].CostInit() << "\n";
            }
        }

        for (aP.y = aBox._p0.y ; aP.y<aBox._p1.y; aP.y++)
        {
            for (aP.x = aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
            {
                tCost  aNewCost = aMat[aP].CostPassageForce()-aCoutMin;
                tCost & aCF = aMat[aP].CostFinal();
                if (mModeAgr==ePrgDAgrSomme) // Mode somme
                {
                   aCF += aNewCost;
                }
                else if (mModeAgr==ePrgDAgrMax) // Mode max
                {
                   ElSetMax(aCF,aNewCost);
                }
                else if (mModeAgr==ePrgDAgrProgressif) // Mode max
                {
		    aCF= aNewCost;
                    aMat[aP].SetCostInit
		    (
		       round_ni
		       (
		             mPdsProgr*aCF 
			  + (1-mPdsProgr)* aMat[aP].GetCostInit()
			)
                    );
                }
                else  // Mode reinjection
                {
                     aCF = aNewCost;
                     aMat[aP].SetCostInit(aCF);
                }

            }
        }

    }
}


void cGBV2_ProgDynOptimiseur::BalayageOneDirection(Pt2dr aDirR)
{
     Pt2di aDirI = Pt2di(vunit(aDirR) * 20.0);
     mLMR.Init(aDirI,Pt2di(0,0),mSz);

     const std::vector<Pt2di> * aVPt;
     while ((aVPt=mLMR.Next()))
          BalayageOneLine(*aVPt);
}








void cGBV2_ProgDynOptimiseur::SolveOneEtape(int aNbDir)
{
   mModeAgr = ePrgDAgrSomme;
   mNbDir = aNbDir;
   

   for (int aKP=0 ; aKP<theDimPxMax ; aKP++)
   {
       // mCostActu[aKP] =0;
       mTabCost[aKP].Reset(0,0);
   }

   for (int aKDir=0 ; aKDir<mNbDir ; aKDir++)
   {
       mPdsProgr = (1.0+aKDir) / mNbDir;
       double aTeta =   (aKDir*PI)/mNbDir;

       Pt2dr aP = Pt2dr::FromPolar(100.0,aTeta);
       // On le met la parce que en toute rigueur ca depend de la 
       // direction, mais pour l'instant on ne gere pas cette dependance
       for (int aKP=0 ; aKP<mNbPx ; aKP++)
       {
	    mTabCost[aKP].Reset
	    (
	        mCostRegul[aKP],
	        mCostRegul_Quad[aKP]
            );
       }

       BalayageOneDirection(aP);
   }

   {
      Pt2di aPTer;
      for (aPTer.y=0 ; aPTer.y<mSz.y ; aPTer.y++)
      {
          for (aPTer.x=0 ; aPTer.x<mSz.x ; aPTer.x++)
          {
              tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[aPTer];
              const Box2di &  aBox = aMat.Box();
              Pt2di aPRX;
              for (aPRX.y=aBox._p0.y ;aPRX.y<aBox._p1.y; aPRX.y++)
              {
                  for (aPRX.x=aBox._p0.x ;aPRX.x<aBox._p1.x; aPRX.x++)
                  {
                      tCost & aCF = aMat[aPRX].CostFinal();
                      if (mModeAgr==ePrgDAgrSomme) // Mode somme
                      {
                         aCF /= mNbDir;
                      }
                      aMat[aPRX].SetCostInit(aCF);
                      aCF = 0;
                  }
              }
          }
      }
   }
}



void cGBV2_ProgDynOptimiseur::Local_SolveOpt
     (
         int aNbDir,
	 double aPenteMax,
	 double aRegul,
	 double aRegul_Quad
     )
{
  double aVPentes[theDimPxMax];

  mCostRegul[0] = aRegul;
  mCostRegul[1] = 0; 

  mCostRegul_Quad[0] = aRegul_Quad;
  mCostRegul_Quad[1] = 0 ;

  aVPentes[0] = aPenteMax;
  aVPentes[1] = 10;


  for (int aKP=0 ; aKP<mNbPx ; aKP++)
  {
       double aPente = aVPentes[aKP]; // MODIF  / mEtape.KPx(aKP).Pas();
       mMaxEc[aKP] = ElMax(1,round_ni(aPente));
  }

   SolveOneEtape(aNbDir);


   {
      Pt2di aPTer;
      for (aPTer.y=0 ; aPTer.y<mSz.y ; aPTer.y++)
      {
          for (aPTer.x=0 ; aPTer.x<mSz.x ; aPTer.x++)
          {
              tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[aPTer];
              const Box2di &  aBox = aMat.Box();
              Pt2di aPRX;
              Pt2di aPRXMin;
              tCost   aCostMin = tCost(1e9);
              for (aPRX.y=aBox._p0.y ;aPRX.y<aBox._p1.y; aPRX.y++)
              {
                  for (aPRX.x=aBox._p0.x ;aPRX.x<aBox._p1.x; aPRX.x++)
                  {
                      tCost aCost = aMat[aPRX].GetCostInit();
                      if (aCost<aCostMin)
                      {
                           aCostMin = aCost;
                           aPRXMin = aPRX;
                      }
                  }
              }
	      // MODIF 
              mDataImRes[aPTer.y][aPTer.x] = aPRXMin.x ;
          }
      }
      
   }
}



/**************************************************/
/*                                                */
/*               cSurfaceOptimiseur               */
/*                                                */
/**************************************************/


};

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
