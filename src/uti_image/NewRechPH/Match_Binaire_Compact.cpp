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

class cCompileOPC;    // Un TieP avec acces rapide aux data
class cPairOPC;       // Paire de TieP , memorise le score
class cVTestOPC;      // Un ensemble de paire, peut etre le Truth ou Random
class cCalcOB;        // Un pds de calc binaire + la memo des resultats dont deux vecteur de bit
class cSetLearnOPC;   // Une classe mere


class cCompileOPC
{
    public :
      cCompileOPC(cOnePCarac * aOPC) ;
      
      double   ValCB(const cCompCBOneBit & aCCOB) const;


      cOnePCarac * mOPC;
      INT1 **      mDR;
};

class cPairOPC
{
      public :
          cPairOPC(cOnePCarac & aP1,cOnePCarac & aP2) ;
          bool Eq(const cCompCBOneBit &) const;
          int         mNbNonEq;
      private :
          cCompileOPC mP1;
          cCompileOPC mP2;
};


class cVTestOPC
{
      public :
          cVTestOPC();
          void Compile(int aNbBitsMax);
          void Add(const TIm2DBits<1> &);
          void Reset();
          void Finish();
          double Score(const cVTestOPC &) const;
         
          std::vector<cPairOPC>  mVP; 
      private :
          int                  mNbBitsMax;
          int                  mNbPair;
          std::vector<int>     mHisto;
          std::vector<double>  mHistoCum;
};

class cCalcOB
{
    public :
       friend class cSetLearnOPC;
       cCalcOB(const  cCompCBOneBit &,const cVTestOPC & aVTruth,const cVTestOPC & aVRand);
    private :
       void Init(TIm2DBits<1> & aTB,const cVTestOPC &,int & aNbEq,double & aProp);
       // cCompCBOneBit
       cCompCBOneBit  mCOB;
       Im2D_Bits<1>   mIT;
       TIm2DBits<1>   mTIT;
       int            mNbEqT;
       Im2D_Bits<1>   mIR;
       TIm2DBits<1>   mTIR;
       int            mNbEqR;
       double         mScoreIndiv;
};


class cSetLearnOPC
{
      typedef std::pair<cCompileOPC,cCompileOPC> tLearnPOPC;
      public :
            cSetLearnOPC(cSetRefPCarac  & aSRPC,int aNbT,int aNbRand,int aNBBitMax);
            double Score(const std::vector<cCalcOB> &);
            void Test();
            cCalcOB COB(const  cCompCBOneBit &);
      private :
            // Si aNumInv<0 => Random
            // aModeRand 0 => rand X, 1=> rand Y , 2 => all
            cCompCBOneBit RandomCOB_OneInv(int aModeRand,int aNumInv,int aNbCoef);

            cVTestOPC  mVTruth;
            cVTestOPC  mVRand;
            int        mNbInvR;
            int        mNbQuantIR;
};


/**************************************************/
/*                                                */
/*             cCalcOB                            */
/*                                                */
/**************************************************/
cCalcOB::cCalcOB(const  cCompCBOneBit & aCOB,const cVTestOPC & aVTruth,const cVTestOPC & aVRand) :
   mCOB    (aCOB),
   mIT     (aVTruth.mVP.size(),1,0),
   mTIT    (mIT),
   mNbEqT  (0),
   mIR     (aVRand.mVP.size(),1,0),
   mTIR    (mIR),
   mNbEqR  (0)
{
    double aPropT,aPropR;
    Init(mTIT,aVTruth,mNbEqT,aPropT);
    Init(mTIR,aVRand,mNbEqR,aPropR);
    
    mScoreIndiv = aPropT-aPropR;
}

void cCalcOB::Init(TIm2DBits<1> & aTB,const cVTestOPC & aVOP,int &aNbEq,double & aProp)
{
    aNbEq = 0;
    int aNbP = aVOP.mVP.size();
    for (int aK=0 ; aK<aNbP ; aK++)
    {
         bool isEq = aVOP.mVP[aK].Eq(mCOB);
         aTB.oset(Pt2di(aK,0),isEq ? 1 : 0);
         if (isEq) 
            aNbEq++;
    }
    aProp = double(aNbEq) / double(aNbP);
       
}

/**************************************************/
/*                                                */
/*             cCompileOPC                        */
/*                                                */
/**************************************************/


cCompileOPC::cCompileOPC(cOnePCarac * aOPC) :
   mOPC   (aOPC),
   mDR    (mOPC->InvR().ImRad().data())
{
}

double   cCompileOPC::ValCB(const cCompCBOneBit & aCCOB) const
{
   double aRes = 0.0;
   const int * aDX    = aCCOB.IndX().data();
   const int * aDY    = aCCOB.IndY().data();
   const double * aDC = aCCOB.Coeff().data();
   for (int aK=0 ; aK<int(aCCOB.IndX().size()) ; aK++)
   {
       aRes += mDR[aDY[aK]][aDX[aK]] * aDC[aK];
   }
   return aRes;
}


/**************************************************/
/*                                                */
/*             cPairOPC                           */
/*                                                */
/**************************************************/

cPairOPC::cPairOPC(cOnePCarac & aP1,cOnePCarac & aP2) :
   mNbNonEq (0),
   mP1   (&aP1),
   mP2   (&aP2)
{
}

bool cPairOPC::Eq(const cCompCBOneBit & aCCOB) const
{
   return  ( (mP1.ValCB(aCCOB)>0) == (mP2.ValCB(aCCOB)>0));
}

/**************************************************/
/*                                                */
/*             cSetLearnOPC                       */
/*                                                */
/**************************************************/

cSetLearnOPC::cSetLearnOPC(cSetRefPCarac  & aSRPC,int aNbT,int aNbRand,int aNBBitMax)
{
    cRandNParmiQ aRand(aNbT,aSRPC.SRPC_Truth().size());
    for (auto & aPT : aSRPC.SRPC_Truth())
    {
       if (aRand.GetNext())
       {
          mVTruth.mVP.push_back(cPairOPC(aPT.P1(),aPT.P2()));
       }
    }
    int aNbR0 = aSRPC.SRPC_Rand().size();

    std::vector<cOnePCarac> & aVR = aSRPC.SRPC_Rand();
    for (int aK1=0 ; aK1<aNbR0 ; aK1++)
    {
        int aNb = round_up(aNbRand/double(aNbR0));
        for (int aN2 = 0 ; aN2<aNb ; aN2++)
        {
            int aK2 = NRrandom3(aNbR0);
            if (aK2== aK1)
               aN2--;
            else
                mVRand.mVP.push_back(cPairOPC(aVR[aK1],aVR[aK2]));
        }
    }
    mVTruth.Compile(aNBBitMax);
    mVRand.Compile(aNBBitMax);

    mNbQuantIR = aVR.at(0).InvR().ImRad().tx();
    mNbInvR    = aVR.at(0).InvR().ImRad().ty();
}


cCompCBOneBit cSetLearnOPC::RandomCOB_OneInv(int aModeRand,int aNumInv,int aNbCoef)
{
    cCompCBOneBit aRes;
    aRes.IndBit() = -1;
    std::vector<int>  aPermQ = RandPermut(mNbQuantIR);
    std::vector<int>  aPermI = RandPermut(mNbInvR);

    double aSom = 0;
    for (int aK=0 ; aK<aNbCoef ; aK++)
    {
        double aVal = (aK==(aNbCoef-1)) ? -aSom : NRrandC();
        aRes.Coeff().push_back(aVal);
            // aModeRand 0 => rand X, 1=> rand Y , 2 => all
        aRes.IndX().push_back((aModeRand==1) ? aNumInv : aPermQ[aK]);
        aRes.IndY().push_back((aModeRand==0) ? aNumInv : aPermI[aK]);
    }

    return aRes;
}

double cSetLearnOPC::Score(const std::vector<cCalcOB> & aVC)
{
   mVTruth.Reset();
   mVRand.Reset();
   for (const  auto & aCalc : aVC)
   {
      mVTruth.Add(aCalc.mTIT);
      mVRand.Add(aCalc.mTIR);
   }
   mVTruth.Finish();
   mVRand.Finish();
   return mVTruth.Score(mVRand);
}

cCalcOB cSetLearnOPC::COB(const  cCompCBOneBit & aCOB) {return cCalcOB(aCOB,mVTruth,mVRand);}

void cSetLearnOPC::Test()
{
    int aNbCoef= 5;
    for (int aK=0 ; aK<mNbInvR ; aK++)
    {
        std::vector<cCalcOB> aVC;
        for (int aK=0 ; aK< 5; aK++)
        {
            double aScMax = -1;
            cCalcOB aCOBMax =  COB(RandomCOB_OneInv(0,(aK%mNbInvR),aNbCoef));
            for (int aNbT=0 ; aNbT<1000 ; aNbT++)
            {
                 cCalcOB aCOB =  COB(RandomCOB_OneInv(0,(aK%mNbInvR),aNbCoef));

                 std::vector<cCalcOB> aVCur = aVC;
                 aVCur.push_back(aCOB);
                 double aS = Score(aVCur);

                 if (aS> aScMax)
                 {
                      aCOBMax = aCOB;
                      aScMax = aS;
                 }
            }
            aVC.push_back(aCOBMax);
            std::cout << "SSSSSSS glob : " << aScMax  << "\n";
        }
        std::cout << "=============================\n";
    }
}

void TestLearnOPC(cSetRefPCarac & aSRP)
{
    cSetLearnOPC aSLO(aSRP,2000,2000,16);
    aSLO.Test();
}


/**************************************************/
/*                                                */
/*                cVTestOPC                       */
/*                                                */
/**************************************************/

cVTestOPC::cVTestOPC() 
{
}
void cVTestOPC::Compile(int aNbBitsMax) 
{
    mNbBitsMax = aNbBitsMax;
    mNbPair    = mVP.size();
    mHisto     = std::vector<int>(aNbBitsMax+1,0);
    mHistoCum  = std::vector<double>(aNbBitsMax+1,0);
}

void cVTestOPC::Add(const TIm2DBits<1> & aTB)
{
    for (int aK=0 ; aK<int(mVP.size()) ; aK++)
    {
       if (! aTB.get(Pt2di(aK,0)))
       {
           mVP[aK].mNbNonEq++;
       }
    }
}

void cVTestOPC::Reset()
{
    for (auto & aP : mVP)
        aP.mNbNonEq = 0;
}

void cVTestOPC::Finish()
{
  // Calcul de l'histo
  for (auto & aH : mHisto)
      aH = 0;
  for (auto & aP : mVP)
     mHisto.at(aP.mNbNonEq)++;
  // Cumul
  mHistoCum[0] = mHisto[0];
  for (int aK=1 ; aK<int(mHisto.size()) ; aK++)
  {
      mHistoCum[aK] = mHistoCum[aK-1] + mHisto[aK];
  }
  // Normalization
  double aSom = mHistoCum.back();
  for (int aK=0 ; aK<int(mHisto.size()) ; aK++)
  {
      mHistoCum[aK]   /= aSom;
  }
}

double cVTestOPC::Score(const cVTestOPC & aVOP) const
{
    double aRes = -1.0;
    for (int aK=0 ; aK<int(mHistoCum.size()) ; aK++)
    {
       aRes = ElMax(aRes,mHistoCum[aK]-aVOP.mHistoCum[aK]);
    }
   
    return aRes;
}


//           double Score(cVTestOPC &);


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
