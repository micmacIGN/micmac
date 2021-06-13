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


class cHistoInt
{
    public :
         void Add(int anInd,double aVal=1.0);
         int  Get(int anInd);
         void Show();
    private :
         std::vector<double> mVH;
};

void cHistoInt::Add(int anInd,double aVal)
{
    for (int aK=mVH.size() ; aK<= anInd ; aK++)
        mVH.push_back(0.0);
    mVH.at(anInd) += aVal;
}

int  cHistoInt::Get(int anInd)
{
    if ((anInd<0) || (anInd>=int(mVH.size()))) return 0.0;
    return mVH.at(anInd);
}
void  cHistoInt::Show()
{
    double aSom = 0;
    double aSomV = 0;
    for (int aK=0 ; aK<int(mVH.size()) ; aK++)
    {
        aSom += mVH.at(aK);
        aSomV += aK * mVH.at(aK);
    }

    double aCum=0.0;
    for (int aK=0 ; aK<int(mVH.size()) ; aK++)
    {
        double aV =  mVH.at(aK);
        
        if (aV)
        {
             printf("For %2d V=%5.2f  %%=%5.2f  %%Cumul=%5.2f\n",aK,aV, (100.0*(aV/aSom)),(100.0*(aCum/aSom)) );
             // std::cout << "For " << aK << " V=" << aV << " %=" << 100.0 * (aV/aSom) << "\n";
        }
        aCum += aV;
    }
    if (aSom)
        printf(" Average = %.3f\n",aSomV/aSom);
}

cHistoInt HistoRMI(const std::vector<cResulMultiImRechCorrel *>  & aVR)
{
    cHistoInt aRes;
    for (int aK=0 ; aK<int(aVR.size()) ; aK++)
       aRes.Add(aVR[aK]->VIndex().size());
    return aRes;
}



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
              return aR1->Score() > aR2->Score();   // compare score correl global
        }
        // est ce que objet 1 est meuilleur que 2
};

class cFuncPtOfRMICPtr
{   // argument du qauad tri
    // comment à partir un objet, je recuper sa pt2D
      public :
         Pt2dr operator () (cResulMultiImRechCorrel * aRMIRC) {return Pt2dr(aRMIRC->PtMast());}
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
#define PowAggreg 1.0

std::vector<cResulMultiImRechCorrel *> cAppliTieTri::FiltrageSpatial
                                       (
                                           const std::vector<cResulMultiImRechCorrel *> & aVIn,
                                           double aSeuilDist,
                                           double aGainCorrel
                                       )
{
   double aSign= 1;
   double ShowMult = (mCurEtape==ETAPE_FINALE);
   if (ShowMult) 
   {
       cHistoInt aH = HistoRMI(aVIn);
       std::cout << " ============== ENTREE ==============\n";
       aH.Show();
   }


if (0) // (MPD__MM())
{
    static bool First= true;
    aSign = -1;
    if (First)
    {
       getchar();
       First= false;
    }
}


   std::vector<cResulMultiImRechCorrel *>  aResult;

   static tQtTiepT * aQdt = 0;  // le quad-tri
   static cFuncPtOfRMICPtr  aFctr;
   if (aQdt==0)
   {
       Pt2dr aSz= Pt2dr(mMasIm->Tif().sz());    // taille d'espace à recuperer les objets = taille d'image
       Pt2dr aRab(10,10);
       aQdt = new tQtTiepT(aFctr,Box2dr(-aRab,aSz+aRab),10,20.0); //10=N obj Max, 20.0=Sz Min
       /*
        * Creer un nouveau Quad-tri.
        *  - Structure de Quad-Tri:
        *    + on peut prendre un ensemble de objet dans 1 espace dans 1 region defini rapidement.
        *    + La region peut defini comme un cercle autour 1 point rayon donne, 1 region autour 1 segment, 1 box...
        *    + Ici, notre espace est 1 image, objet est cResulMultiImRechCorrel* (result correl de chaque point), et le region avec l'objet defini dans l'espace par Pt2dr (coordonne de point)
        * Defini un quad tri par typedef ElQT<cResulMultiImRechCorrel *,Pt2dr,cFuncPtOfRMICPtr> tQtTiepT;
        * Ca veut dire il accept les objets type cResulMultiImRechCorrel *,
        * recherche par Pt2dr, avec le method de recuperer Pt2dr à partir de cResulMultiImRechCorrel * est definie dans cFuncPtOfRMICPtr
        * Creer un nouveau tQtTiepT :
        *   .) foncteur = cFuncPtOfRMICPtr  aFctr => method de recuperer Pt2dr à partir de pointer cResulMultiImRechCorrel *
        *   .) Box2dr(-aRab,aSz+aRab) => une region = taille d'image + rab 10 pxl (pour rassurer?), qui contient tout mes objet
        *   .) NbObjMax = 10 => une seuile de bas pour commencer à decouper l'espace objet (en 4)
        *   .) SzMin = 20.0 => ?

       */
   }

   cTpP_HeapCompare aCmp; // if aR1 > aR2
   ElHeap<cResulMultiImRechCorrel *,cTpP_HeapCompare,cTpP_HeapParam> aHeap(aCmp); // HeapParam: setIndex & getIndex
   /* == Definir un structure donne type Heap ==
    * ElHeap<cResulMultiImRechCorrel *,cTpP_HeapCompare,cTpP_HeapParam> aHeap(aCmp);
    *  .) Type objet cResulMultiImRechCorrel *
    *  .) cTpP_HeapCompare => comment heap evaluer objet
    *  .) cTpP_HeapParam => access au heap index dans l'objet
   */
   for (int aK=0; aK <int(aVIn.size()) ; aK++)
   {
       // calcul score global de correlation sur tout les coup image
       // 1 pt Mas correl sur plsr pt 2nd
       //     => calcul 1 score glob, mis a jour le score glob dans cResulMultiImRechCorrel aussi
       aVIn[aK]->CalculScoreAgreg(EpsilAggr,PowAggreg,aSign);  // Epsilon, power
       aHeap.push(aVIn[aK]);    // mets l'objet dans heap <=> push_back
       aQdt->insert(aVIn[aK]);  // mets l'objet dans le Quad-Tri
   }

   cResulMultiImRechCorrel * aRM_1;
   // Contient les scores en fonction des numeros d'images
   std::vector<double> aVCorrel(mImSec.size(),TT_DefCorrel); // mImSec = all image 2nd of this tri
   Video_Win *  aW = mMasIm->W();
   while (aHeap.pop(aRM_1))
   {
       // pop à partir un heap => recuper la "meuilleur" correlé point et l'enleve dans heap
       // aRM_1 contient le point avec Score() meuilleur
       if (aW)
       {
           aW->draw_circle_loc(aFctr(aRM_1),aSeuilDist,aW->pdisc()(P8COL::cyan));

           // std::cout << "Mult " << aRM_1->VIndex() << "\n";
           // dessine un cercle sur Img Master, au pt master, rayon TT_DefSeuilDensiteResul = 50
       }
       aResult.push_back(aRM_1);
       const std::vector<int> &  aVI_1 = aRM_1->VIndex();
       int aNbI_1 = aVI_1.size();
       const std::vector<cResulRechCorrel > & aVC_1 = aRM_1->VRRC() ;
       /*
        * aVI_1 = vector<int> contient index de tout les pt correl dans aRM_1
        * aVC_1 = vector<cResulRechCorrel > contient tout les pt correl (mPt, mScore)
        */
       // Mets a jour le score fonction du numero



       for (int aK=0 ; aK<aNbI_1 ; aK++)
       {
           aVCorrel[aVI_1[aK]] =  aVC_1[aK].mCorrel;
       }

       std::set<cResulMultiImRechCorrel *> aSet;
       // recuper tout les pts dans aSeuilDist (TT_DefSeuilDensiteResul = 50pxl) distance (region à filtrer)
       aQdt->RVoisins(aSet,aFctr(aRM_1),aSeuilDist);
       for (std::set<cResulMultiImRechCorrel *>::iterator itS=aSet.begin(); itS!=aSet.end() ; itS++)
       {   //== parcourir tout les point dans region à filtrer ==
           cResulMultiImRechCorrel * aRM_2 = *itS;
           if (aRM_1==aRM_2)
           {
              aQdt->remove(aRM_2);
              // enleve le point dans le Quad-Tri (on a pop out de heap, mais il exist encore dans le Quad-Tri)
           }
           else
           {
              const std::vector<int> &  aVI_2 = aRM_2->VIndex();
              int aNbI_2 = aVI_2.size();
              const std::vector<cResulRechCorrel > & aVC_2 = aRM_2->VRRC() ;
              /*
               * aVI_2 = vector<int> contient index de tout les pt correl dans aRM_2
               * aVC_2 = vector<cResulRechCorrel > contient tout les pt correl (mPt, mScore)
               */
              // === formule pour decider si on enleve un point ===
              double aDist = euclid(aFctr(aRM_1)-aFctr(aRM_2)); // distance euclid entre 2 point master
              double aRabCorrel = (1-pow(aDist/aSeuilDist,TT_FSExpoAtten)) * aGainCorrel;
              int aNbS0 = aRM_2->NbSel();
              for (int aK=0 ; aK<aNbI_2 ; aK++)
              {
                  int aKIm = aVI_2[aK]; // index of secondary image
                  if (aVC_2[aK].mCorrel < (aVCorrel[aKIm]+aRabCorrel))
                  /*
                   * Consider 2 point master aRM_1 & aRM_2
                   * On veut filtrer spatial autour de point aRM_1 avec 1 rayon donné
                   * On recuper tout les pts aRM_2, puis on regarde score correl du aRM_2 < aRM_1 + rab correl
                   * Si oui, on de-selection aRM_2
                   * Ca veut dire meme si aRM_2 est une score plus grand que aRM_1, mais la grandeur est pas assez grand, on jete aussi
                   */
                  {
                      aRM_2->SetSelec(aK,false); // déseletioner un pt
                  }
              }
              int aNbSelEnd = aRM_2->NbSel();

              if (aNbS0!=aNbSelEnd) // au moins 1 point dans multiple aRM_2 est deselectione
              {
                  // ==== Si rentrer ici, ca veut dire aRM_2 est modifie ====
                  aRM_2->CalculScoreAgreg(EpsilAggr,PowAggreg,aSign);  // Epsilon, power
                  if (aNbSelEnd==0)
                  {
                     aQdt->remove(aRM_2);
                     aHeap.Sortir(aRM_2);
                     delete aRM_2;
                  }
                  else
                  {
                     aHeap.MAJ(aRM_2);  // ca veut dire il est moin multiple
                     // mis à jour pour ne pas cassé la structure de heap
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

   if(aW)
   {
      std::cout << "FILTRAGE SPATIAL, " << aVIn.size() << " => " << aResult.size() << "\n";
   }

   if (ShowMult)
   {
       cHistoInt aH = HistoRMI(aResult);
       std::cout << " ============== Sortie  ==============\n";
       aH.Show();
   }

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
