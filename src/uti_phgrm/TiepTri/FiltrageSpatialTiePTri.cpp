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



class cTpP_HeapParam
{
     public :
        static void SetIndex(cResulMultiImRechCorrel *  aRMIRC,int i) 
        {
                aRMIRC->HeapIndexe() = i;
        }
        static int  Index(cResulMultiImRechCorrel *   aRMIRC)
        {    
             return aRMIRC->HeapIndexe();
        }
};

class cTpP_HeapCompare
{
    public :

        bool operator () (cResulMultiImRechCorrel *  & aR1,cResulMultiImRechCorrel *  & aR2)
        {
              return aR1->Score() > aR2->Score();
        }
};

class cFuncPtOfRMICPtr
{
      public :
         Pt2dr operator () (cResulMultiImRechCorrel * aRMIRC) {return Pt2dr(aRMIRC->PMaster().mPt);}
};


typedef ElQT<cResulMultiImRechCorrel *,Pt2dr,cFuncPtOfRMICPtr> tQtTiepT;

/*
   A cette etape, les correlation ne sont pas tres precise (correlation entiere) donc on 
   fait une pre selection prudente.

   Algo :
      * choisir le point de meilleur correlation  P (Pt2f)  C (correl)  I (Image) , le selectionner
      * pour tout les voisins dans un cercle (P,Dist) ou Dist est un seuil
          * pour toute les 

*/

#define EpsilAggr 0.02
#define PowAggreg 0.02

std::vector<cResulMultiImRechCorrel *> cAppliTieTri::FiltrageSpatial
                                       (
                                           const std::vector<cResulMultiImRechCorrel *> & aVIn,
                                           double aSeuilDist,
                                           double aGainCorrel
                                       )
{
   std::vector<cResulMultiImRechCorrel *>  aResult;

   static tQtTiepT * aQdt = 0;
   static cFuncPtOfRMICPtr  aFctr;
   if (aQdt==0)
   {
       Pt2dr aSz= Pt2dr(mMasIm->Tif().sz());
       Pt2dr aRab(10,10);
       aQdt = new tQtTiepT(aFctr,Box2dr(-aRab,aSz+aRab),10,20.0);
   }


   cTpP_HeapCompare aCmp;
   ElHeap<cResulMultiImRechCorrel *,cTpP_HeapCompare,cTpP_HeapParam> aHeap(aCmp);
   for (int aK=0; aK <int(aVIn.size()) ; aK++)
   {
       aVIn[aK]->CalculScoreAgreg(EpsilAggr,PowAggreg);  // Epsilon, power
       aHeap.push(aVIn[aK]);
       aQdt->insert(aVIn[aK]);
   }

   cResulMultiImRechCorrel * aRM_1;
   // Contient les scores en fonction des numeros d'images
   std::vector<double> aVCorrel(mImSec.size(),TT_DefCorrel);
   while (aHeap.pop(aRM_1))
   {
       aResult.push_back(aRM_1);
       const std::vector<int> &  aVI_1 = aRM_1->VIndex();
       int aNbI_1 = aVI_1.size();
       const std::vector<cResulRechCorrel > & aVC_1 = aRM_1->VRRC() ;

       // Mets a jour le score fonction du numero
       for (int aK=0 ; aK<aNbI_1 ; aK++)
       {
           aVCorrel[aVI_1[aK]] =  aVC_1[aK].mCorrel;
       }

       std::set<cResulMultiImRechCorrel *> aSet;
       aQdt->RVoisins(aSet,aFctr(aRM_1),aSeuilDist);
       for (std::set<cResulMultiImRechCorrel *>::iterator itS=aSet.begin(); itS!=aSet.end() ; itS++)
       {
           cResulMultiImRechCorrel * aRM_2 = *itS;
           if (aRM_1==aRM_2)
           {
              aQdt->remove(aRM_2);
           }
           else
           {
              const std::vector<int> &  aVI_2 = aRM_2->VIndex();
              int aNbI_2 = aVI_2.size();
              const std::vector<cResulRechCorrel > & aVC_2 = aRM_2->VRRC() ;

              double aDist = euclid(aFctr(aRM_1)-aFctr(aRM_2));
              double aRabCorrel = (1-(aDist/aSeuilDist)) * aGainCorrel;

              int aNbS0 = aRM_2->NbSel();
              for (int aK=0 ; aK<aNbI_2 ; aK++)
              {
                  int aKIm = aVI_2[aK];
                  if (aVC_2[aK].mCorrel < (aVCorrel[aKIm]+aRabCorrel))
                  {
                      aRM_2->SetSelec(aK,false);
                  }
              }
              int aNbSelEnd = aRM_2->NbSel();

              if (aNbS0!=aNbSelEnd)
              {
                  aRM_2->CalculScoreAgreg(EpsilAggr,PowAggreg);  // Epsilon, power
                  if (aNbSelEnd==0)
                  {
                     aQdt->remove(aRM_2);
                     aHeap.Sortir(aRM_2);
                  }
                  else
                  {
                     aHeap.MAJ(aRM_2);
                  }
                  // if (aRM_2->NbSel()
              }
           }
/*
   double aSeuilDist,
   double aGainCorrel
*/
       }

       // Efface les score
       for (int aKIm=0 ; aKIm<aNbI_1  ; aKIm++)
       {
           aVCorrel[aVI_1[aKIm]] = TT_DefCorrel;
       }
   }
   aQdt->clear();
   aHeap.clear();

   std::cout << "FILTRAGE SPATIAL, " << aVIn.size() << " => " << aResult.size() << "\n";

   return aResult;
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
