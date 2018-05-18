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

template <const int NbBit> class cTabulNbB
{
    public :
       static const int NbFlag = 1<<NbBit;
       int   mTabulNbBit[NbFlag];

       cTabulNbB();
};

template <const int NbBit> cTabulNbB<NbBit>::cTabulNbB()
{
   for (int aFlag=0 ; aFlag<NbFlag ; aFlag++)
   {
       mTabulNbBit[aFlag] = 0;
       for (int aBit=0 ; aBit<NbBit ; aBit++)
            if (aFlag & (1<<aBit))
               mTabulNbBit[aFlag] ++;
   }
}

inline int NbBitsOfFlag16(const int aFlag)
{
   static cTabulNbB<16> aTab;

   return aTab.mTabulNbBit[aFlag];
}



template <const int TheNbUI2>  
   inline tTplCodBin<TheNbUI2> operator ^ (const tTplCodBin<TheNbUI2> & aC1,const tTplCodBin<TheNbUI2> & aC2)
{
   tTplCodBin<TheNbUI2> aRes;
   for (int aK=0 ; aK<TheNbUI2 ; aK++)
      aRes.mCode[aK] = aC1.mCode[aK] ^ aC2.mCode[aK] ;
   return aRes;
}

void FFFFF()
{
   tTplCodBin<5> aC1,aC2;
   aC1 = aC1 ^ aC2;
}




static constexpr int TheNbUI2Flag = 5;
// typedef Im2D_U_INT2 tCodBin;


static constexpr int TheNbBitTabuled = 16;
static constexpr int TheNbFlagTabuled = 1<<TheNbBitTabuled;
static constexpr int TheMasqBitTabuled = TheNbFlagTabuled - 1;

int NbBitOfShortFlag(int aFlag)
{
   static int aTab[TheNbFlagTabuled];
   static bool first = true;
   if (first)
   {
      for (int aF=0 ; aF<TheNbFlagTabuled ; aF++)
      {
         aTab[aF] = 0;
         for (int aB=0 ; aB<TheNbBitTabuled ; aB++)
            if (aF & (1<<aB))
               aTab[aF] ++;
{
   // std::cout << "FFf" << aF << " " << NbBitsOfFlag16(aF) << " " << aTab[aF] << "\n";
}
      }
      first = false;
   }

   return aTab[aFlag];
}

void SetOfFlagInfNbb(std::vector<int> & aRes,int aNbBitTot,int aNbBitAct)
{
    aRes.clear();
    for (int aFlag=0 ; aFlag<(1<<aNbBitTot) ; aFlag++)
    {
       if (NbBitOfShortFlag(aFlag) <= aNbBitAct)
          aRes.push_back(aFlag);
    }
}

typedef std::vector<int> * tPtrVInt;
const std::vector<int> * FlagOfNbb(int aNbBitTot,int aNbBitAct)
{
    static std::map<Pt2di,std::vector<int> *> TheMap;
    tPtrVInt & aRes = TheMap[Pt2di(aNbBitTot,aNbBitAct)];
    if (aRes==0)
    {
       aRes = new std::vector<int>;
       SetOfFlagInfNbb(*aRes,aNbBitTot,aNbBitAct);
    }
    return aRes;
}



int NbBitOfFlag(tCodBin aFlag)
{
    U_INT2 * aData = aFlag.data_lin();
    int aTX = aFlag.tx();
    int aRes=0;
    for (int aX=0 ; aX<aTX ; aX++)
        aRes += NbBitOfShortFlag(aData[aX]);

    return aRes;
}

int NbBitDifOfFlag(tCodBin aFlag1,tCodBin aFlag2)
{
    U_INT2 * aData1 = aFlag1.data_lin();
    U_INT2 * aData2 = aFlag2.data_lin();
    int aTX = aFlag1.tx();
    int aRes=0;
    for (int aX=0 ; aX<aTX ; aX++)
        aRes += NbBitOfShortFlag(aData1[aX] ^ aData2[aX]);

    return aRes;
}

int NbBitDifOfFlag(int aFlag1,int aFlag2)
{
   return NbBitOfShortFlag(aFlag1^aFlag2);
}


/*

void TestNbBitOfFlag(int aF,int aNbB)
{
    std::cout << "Flag= " << aF << " ; NbB= " << NbBitOfFlag(aF) << " Theo=" << aNbB << "\n";
}
void TestNbBitOfFlag()
{
    TestNbBitOfFlag(1,1);
    TestNbBitOfFlag(3,2);
    TestNbBitOfFlag((1<<8) | (1<<13),2);
    TestNbBitOfFlag((1<<8) | (1<<13) | (1<<16) ,3);
    TestNbBitOfFlag((1<<8) | (1<<13) | (1<<16) | (1<<27) ,4);
}

*/


static bool CmpOnX(const Pt2dr & aP1,const Pt2dr &aP2)
{
   return aP1.x < aP2.x;
}

cCBOneBit RandomOneBit(int aIndBit,std::vector<double> & aPdsInd,int aNbCoef)
{
   int aNbInVect = aPdsInd.size();
   cCBOneBit  aVCOneB ;
   aVCOneB.IndBit() =  aIndBit;
         

   std::vector<Pt2dr> aVPt;
   for (int aKIV=0 ; aKIV<aNbInVect ; aKIV++)
   {
       aVPt.push_back(Pt2dr((1+NRrandC())*aPdsInd[aKIV],aKIV));
   }
   std::sort(aVPt.begin(),aVPt.end(),CmpOnX);

   double aSom=0;
   double aSom2=0;
   for (int aKC=0 ; aKC<aNbCoef ; aKC++)
   {
       double aVal =  (aKC==(aNbCoef-1)) ?  (-aSom) : NRrandC();
       aSom += aVal;
       aSom2 += ElSquare(aVal);
       aVCOneB.Coeff().push_back(aVal);
       int aInd = round_ni(aVPt[aKC].y);
       aVCOneB.IndInV().push_back(aInd);
       aPdsInd[aInd] += ElAbs(aVal);
   }
   double anEcart = sqrt(ElMax(1e-10,aSom2));
   for (int aKC=0 ; aKC<aNbCoef ; aKC++)
   {
       aVCOneB.Coeff()[aKC] /= anEcart;
   }
   return  aVCOneB;
}


cFullParamCB RandomFullParamCB(const cOnePCarac & aPC,const std::vector<int> & aNbBitsByVect,int aNbCoef)
{
   // int aNbTirage = 10;
   cFullParamCB aRes;
   // Uniquement pour connaitre le nombre de vect
   Im2D_INT1 aImR = aPC.InvR().ImRad();
   int aNbV = aImR.ty();
   int aIndBit=0;

   for (int aIV=0 ; aIV<aNbV ; aIV++)
   {
      aRes.CBOneVect().push_back(cCBOneVect());
      cCBOneVect & aVCBOneV = aRes.CBOneVect().back();
      aVCBOneV.IndVec() = aIV;

      int aNBB = aNbBitsByVect.at(aIV);
      int aNbInVect = aImR.tx();

      std::vector<double> aPdsInd(aNbInVect,0.5); // On biaise les stats pour privilegier la repartition des coeffs

      for (int aBit=0 ; aBit<aNBB ; aBit++)
      {
         aVCBOneV.CBOneBit().push_back(RandomOneBit(aIndBit,aPdsInd,aNbCoef));
         aIndBit++;
      }
   }
   
   return aRes;
}

cFullParamCB RandomFullParamCB(const cOnePCarac & aPC,int aNbBitsByVect,int aNbCoef)
{
   return RandomFullParamCB(aPC,std::vector<int>(100,aNbBitsByVect),aNbCoef);
}

double  ValCB(const cCBOneBit & aCB,const INT1 * aVD)
{
    double aRes=0;
    for (int aK=0 ; aK<int(aCB.Coeff().size()) ; aK++)
    {
        aRes += aCB.Coeff()[aK] * aVD[aCB.IndInV()[aK]];
    }
    return aRes;
}


void FlagCB(const cCBOneVect & aCBV,const INT1 * aVD,tCodBin aCodB) // Si IsRel part de 0
{
   U_INT2 * aData = aCodB.data_lin();
   for (const auto & aCB : aCBV.CBOneBit())
   {
        if (ValCB(aCB,aVD) > 0)
        {
           int aIB = aCB.IndBit();
           aData[aIB/16]  |=  (1<<(aIB%16));
        }
   }
}


void FlagCB(const cFullParamCB & aCB,Im2D_INT1 aIm ,tCodBin aCodB)
{
   
   for (const auto & aCBO : aCB.CBOneVect())
   {
      FlagCB(aCBO,aIm.data()[aCBO.IndVec()],aCodB);
   }
}



void SetFlagCB(const cFullParamCB & aCB,const std::vector<cOnePCarac*>  & aVPC)
{
    for (auto & aPC : aVPC)
    {
       aPC->InvR().CodeBinaire() = tCodBin(TheNbUI2Flag,1,(U_INT2)0);
       FlagCB(aCB,aPC->InvR().ImRad(),aPC->InvR().CodeBinaire());
    }
}

void AddHistoBits(std::vector<int> & aVH,cOnePCarac* aP1,cOnePCarac* aP2,double & aSom)
{
   if (aP1 && aP2)
   {
      aVH[NbBitDifOfFlag(aP1->InvR().CodeBinaire() , aP2->InvR().CodeBinaire())] ++;
      aSom++;
   }
}



void TestFlagCB(  const cFullParamCB & aCB,
                  const std::vector<cOnePCarac*>  & aV1,
                  const std::vector<cOnePCarac*>  & aV2,
                  const std::vector<cOnePCarac*>  & aHomOf1
               )
{
   SetFlagCB(aCB,aV1);
   SetFlagCB(aCB,aV2);
   std::vector<int> aVBRand(128,0);
   std::vector<int> aVBTruth(128,0);
   double aSomRand=0;
   double aSomTruth=0;

   for (int aK1=0 ; aK1<int(aV1.size()) ; aK1++)
   {
      for (int aK2=0 ; aK2<int(aV2.size()) ; aK2++)
      {
         AddHistoBits(aVBRand,aV1[aK1],aV2[aK2],aSomRand);
      }
      AddHistoBits(aVBTruth,aV1[aK1],aHomOf1[aK1],aSomTruth);
   }

   int aCumRand=0;
   int aCumTruth=0;
   for (int aKH=0 ; aKH<64 ; aKH++)
   {
      aCumRand  += aVBRand[aKH];
      aCumTruth += aVBTruth[aKH];
      double aPropRand = aCumRand / aSomRand;
      double aPropTruth = aCumTruth / aSomTruth;

      std::cout << "Nbb= " << aKH << " ; PROP " << aPropRand << " ; Truth " << aPropTruth << "\n";

      if (aPropTruth > 0.99) 
         aKH=1000;
   }
   getchar();
}

// ===================================================================
// 
//    Apprentissage 
// 
// ===================================================================
typedef const INT1 * tCPtVD;
typedef std::vector<std::pair<tCPtVD,tCPtVD> > tVPairCPVD;

tVPairCPVD  EchantPair
            (
                int aInd,
                const std::vector<cOnePCarac*>  & aV1,
                const std::vector<cOnePCarac*>  & aV2,
                int aNb
            )
{
   tVPairCPVD  aRes;
   cRandNParmiQ aRand(aNb,aV1.size()*aV2.size());
   for (int aK1=0 ; aK1<int(aV1.size()) ; aK1++)
   {
      for (int aK2=0 ; aK2<int(aV2.size()) ; aK2++)
      {
         if (aRand.GetNext())
         {
            aRes.push_back(std::pair<tCPtVD,tCPtVD>(
                             aV1[aK1]->InvR().ImRad().data()[aInd],
                             aV2[aK2]->InvR().ImRad().data()[aInd]
            ));
         }
      }
   }
   return aRes;
}

tVPairCPVD  TruthPair
            (
                int aInd,
                const std::vector<cOnePCarac*>  & aV1,
                const std::vector<cOnePCarac*>  & aVTruth
            )
{
   tVPairCPVD aRes;

   for (int aK1=0 ; aK1<int(aV1.size()) ; aK1++)
   {
      if (aV1[aK1] && aVTruth[aK1])
      {
            aRes.push_back(std::pair<tCPtVD,tCPtVD>(
                             aV1[aK1]->InvR().ImRad().data()[aInd],
                             aVTruth[aK1]->InvR().ImRad().data()[aInd]
            ));
      }
   }
   return aRes;
}


double PropEq (const cCBOneBit & aCB,const  tVPairCPVD & aVP)
{
    int aNbEq = 0;
    for (const auto & aP : aVP)
    {
         double  aV1 = ValCB(aCB,aP.first);
         double  aV2 = ValCB(aCB,aP.second);
         if ((aV1>0) == (aV2>0)) 
             aNbEq++;
    }
    return aNbEq / double(aVP.size());
}

std::vector<double> PropEq2V (const cCBOneBit & aCB1,const cCBOneBit & aCB2,const  tVPairCPVD & aVP)
{
   std::vector<double> aRes(3,0);
   for (const auto & aP : aVP)
   {
        double  aVf1 = ValCB(aCB1,(aP.first));
        double  aVs1 = ValCB(aCB1,(aP.second));

        double  aVf2 = ValCB(aCB2,(aP.first));
        double  aVs2 = ValCB(aCB2,(aP.second));
         
        int aNbEq =  ((aVf1>0) == (aVs1>0))  + ((aVf2>0) == (aVs2>0));
        aRes[aNbEq] ++;
   }
   for (auto & aV:aRes)
      aV /= aVP.size();
   return aRes;
}

double ScoreApprent2V
       (
            const cCBOneBit & aCB1,
            const cCBOneBit & aCB2,
            const  tVPairCPVD &  aVRand,
            const  tVPairCPVD &  aTruth,
            double aPdsTruth
        )
{
    std::vector<double> aVR  = PropEq2V (aCB1,aCB2,aVRand);
    std::vector<double> aVT  = PropEq2V (aCB1,aCB2,aTruth);

    double aEps = 0.3;

    double aScoreT = aVT[2] * 2  + aVT[1]  * aEps  ;
    double aScoreR =  aVR[1]    + aVR[0] * (1+aEps) ;

    return aScoreT * aPdsTruth + aScoreR;
}


double ScoreApprent
       (
            const cCBOneBit & aCB,
            const  tVPairCPVD &  aVRand,
            const  tVPairCPVD &  aTruth,
            double aPdsTruth
       )
{
    return PropEq(aCB,aTruth) * aPdsTruth -PropEq(aCB,aVRand);
}

cCBOneBit  Perturb(const cCBOneBit & aCB,double aSize)
{
   cCBOneBit aRes = aCB;
   for (auto & aC  : 	aRes.Coeff())
       aC += aSize * NRrandC();

   return aRes;
}


cCBOneVect RandOptimizeOneBit
          (
              bool DeuxVal,
              int aInd,
              int & aIndBit,
              int aNbCoef,
              const  tVPairCPVD &  aVRand,
              const  tVPairCPVD &  aTruth,
              double aPdsTruth,
              int aNbTir,
              int aTx
           )
{
    cCBOneBit aRes;
    cCBOneBit aRes2;
    double aBestScore = - 1e20;
    for (int aKT=0 ; aKT<aNbTir ; aKT++)
    {
        std::vector<double>  aPdsInd(aTx,1e-14);
        cCBOneBit aTest ;
        cCBOneBit aTest2 ;

        int SeuilT = aNbTir / 2;
        if (aKT < SeuilT)
        {
           aTest = RandomOneBit(aIndBit,aPdsInd,aNbCoef);
           aTest2 =  RandomOneBit(aIndBit+1,aPdsInd,aNbCoef);
        }
        else
        {
            double aSzP =  0.3 * ((aNbTir-aKT) / double(aNbTir-SeuilT));
            aTest = Perturb(aRes,aSzP);
            aTest2 = Perturb(aRes2,aSzP);
        }

        double aScore =   DeuxVal                                              ?
                          ScoreApprent2V(aTest,aTest2,aVRand,aTruth,aPdsTruth) :
                          ScoreApprent(aTest,aVRand,aTruth,aPdsTruth)          ;

        if (aScore > aBestScore)
        {
            aBestScore = aScore;
            aRes = aTest;
            aRes2 = aTest2;
        }
    }
    cCBOneVect aCOV;
    aCOV.IndVec() = aInd;
    aCOV.CBOneBit().push_back(aRes);
    aIndBit++;
    if (DeuxVal)
    {
       aCOV.CBOneBit().push_back(aRes2);
       aIndBit++;
    }

    return aCOV;
}

cFullParamCB  Optimize
              (
                  bool  DeuxVal,
                  const std::vector<cOnePCarac*>  & aV1,
                  const std::vector<cOnePCarac*>  & aV2,
                  const std::vector<cOnePCarac*>  & aHomOf1,
                  double aPdsTruth
               )
{
    cFullParamCB aRes;

    Im2D_INT1 aImR = aV1[0]->InvR().ImRad();

    // std::vector<const std::vector<double> *>  aVR = VRAD(aV1[0]);
    int aNbTir=1000;
    int aNBC = 3;

    int aIndBit = 0;
    for (int aInd=0 ; aInd<aImR.ty() ; aInd++)
    {
std::cout << "KIIIIII " << aInd << "\n";
         tVPairCPVD aTruthPair = TruthPair (aInd,aV1,aHomOf1);
         tVPairCPVD aRanPair = EchantPair(aInd,aV1,aV2,10*aTruthPair.size());

         int aNbCoef = ElMin(aNBC,aImR.tx());
         cCBOneVect aCOV =  RandOptimizeOneBit(DeuxVal,aInd,aIndBit,aNbCoef,aRanPair,aTruthPair,aPdsTruth,aNbTir,aImR.tx());
/*
         cCBOneVect aCOV;
         aCOV.IndVec() = aInd;
         aCOV.CBOneBit().push_back(aCOB);

         aIndBit++;
*/
         aRes.CBOneVect().push_back(aCOV);
    }

    return aRes;
}

#if (0)
#endif


/*
*/


// std::vector<const std::vector<double> *> VRAD(const cOnePCarac * aPC);





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
