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

/*
<FullParamCB  Nb="1" Class="true">
      <CBOneVect  Nb="*" Class="true" Container="std::vector">
           <CBOneBits Nb="*">
                <IndVec Nb="1" Type="int">     </IndVec>
                <Coeff  Nb="1" Type="double">  </Coeff>
                <IndBit Nb="1" Type="int">     </IndBit>
           </CBOneBits>
      </CBOneVect>
</FullParamCB>
*/

static bool CmpOnX(const Pt2dr & aP1,const Pt2dr &aP2)
{
   return aP1.x < aP2.x;
}

cFullParamCB RandomFullParamCB(const cOnePCarac & aPC,const std::vector<int> & aNbBitsByVect,int aNbCoef)
{
   // int aNbTirage = 10;
   cFullParamCB aRes;
   // Uniquement pour connaitre le nombre de vect
   std::vector<const std::vector<double> *> aVVR = VRAD(&aPC);
   int aNbV = aVVR.size();
   int aIndBit=0;

   for (int aIV=0 ; aIV<aNbV ; aIV++)
   {
      aRes.CBOneVect().push_back(cCBOneVect());
      cCBOneVect & aVCBOneV = aRes.CBOneVect().back();
      aVCBOneV.IndVec() = aIV;

      int aNBB = aNbBitsByVect.at(aIV);
      int aNbInVect = aVVR[aIV]->size();

      std::vector<double> aPdsInd(aNbInVect,0.5); // On biaise les stats pour privilegier la repartition des coeffs

      for (int aBit=0 ; aBit<aNBB ; aBit++)
      {
         aVCBOneV.CBOneBit().push_back(cCBOneBit());
         cCBOneBit & aVCOneB  = aVCBOneV.CBOneBit().back();
         aVCOneB.IndBit() =  aIndBit;
         

         std::vector<Pt2dr> aVPt;
         for (int aKIV=0 ; aKIV<aNbInVect ; aKIV++)
         {
            aVPt.push_back(Pt2dr((1+NRrandC())*aPdsInd[aKIV],aKIV));
         }
         std::sort(aVPt.begin(),aVPt.end(),CmpOnX);

// std::cout << "VPPPP " << aVPt << "\n";
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

         aIndBit++;
      }
   }
   
   return aRes;
}

cFullParamCB RandomFullParamCB(const cOnePCarac & aPC,int aNbBitsByVect,int aNbCoef)
{
   return RandomFullParamCB(aPC,std::vector<int>(100,aNbBitsByVect),aNbCoef);
}

double  ValCB(const cCBOneBit & aCB,const std::vector<double> & aVD)
{
    double aRes=0;
    for (int aK=0 ; aK<int(aCB.IndInV().size()) ; aK++)
    {
        aRes += aCB.Coeff()[aK] * aVD[aCB.IndInV()[aK]];
    }
    return aRes;
}


int FlagCB(const cCBOneVect & aCBV,const std::vector<double> & aVD,bool IsRel) // Si IsRel part de 0
{
   int aFlag = 0;
   for (const auto & aCB : aCBV.CBOneBit())
   {
        if (ValCB(aCB,aVD) > 0)
        {
           aFlag |=  (1<<aCB.IndBit());
        }
   }

   if (IsRel)
      aFlag >>= aCBV.CBOneBit()[0].IndBit();

   return aFlag;
}


/*
int CodageBinaire(const cCBOneVect & aCB,const std::vector<double> & )
{
}
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
