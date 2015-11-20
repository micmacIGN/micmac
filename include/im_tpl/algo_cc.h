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



#ifndef _ELISE_IM_ALGO_CC
#define _ELISE_IM_ALGO_CC

Im2D_Bits<1> ImMarqueurCC(Pt2di aSz);
void ResetMarqueur(TIm2DBits<1> & aMarq,const std::vector<Pt2di> & aVPts);



// Mais en aValAff les composante connexe de coul=aValSelec de taille < aSeuilCard

template <class T1,class T2,class Action> int OneZC
                                 (
                                      const Pt2di & aPGerm, bool V4,
                                      T1 & aIm1,int aV1Sel,int aV1Aff,
                                      T2 & aIm2,int aV2Sel,
                                      Action & aOnNewPt
                                 )
{
   Pt2di * aTabV = V4 ? TAB_4_NEIGH : TAB_8_NEIGH ;
   int aNbV = V4 ? 4 : 8;

   std::vector<Pt2di>  aVec1;
   std::vector<Pt2di>  aVec2;

   std::vector<Pt2di> * aVCur = &aVec1;
   std::vector<Pt2di> * aVNext = &aVec2;

   if ((aIm1.get(aPGerm)==aV1Sel) && (aIm2.get(aPGerm)==aV2Sel) && aOnNewPt.ValidePt(aPGerm))
   {
      aIm1.oset(aPGerm,aV1Aff);
      aVCur->push_back(aPGerm);
      aOnNewPt.OnNewPt(aPGerm);
   }
   int aNbStep = 1;

   int aNbTot = 0;
   while (! aVCur->empty())
   {
       int aNbCur = (int)aVCur->size(); 
       aNbTot += aNbCur;
       aOnNewPt.OnNewStep();
       if (aOnNewPt.StopCondStep())
          return aNbTot;

       for (int aKp=0 ; aKp<aNbCur ; aKp++)
       {
           Pt2di aP = (*aVCur)[aKp];
           for (int aKv=0; aKv<aNbV ; aKv++)
           {
                 Pt2di aPV = aP+aTabV[aKv];
                 if ((aIm1.get(aPV)==aV1Sel) && (aIm2.get(aPV)==aV2Sel) && aOnNewPt.ValidePt(aPV))
                 {
                    aIm1.oset(aPV,aV1Aff);
                    aVNext->push_back(aPV);
                    aOnNewPt.OnNewPt(aPV);
                 }
           }
       }

       ElSwap(aVNext,aVCur);
       aVNext->clear();
       aNbStep++;
   }

   return aNbTot;
}

class cCC_NoActionOnNewPt
{
    public :
       void OnNewStep() {}
       void  OnNewPt(const Pt2di &) {}
       bool  StopCondStep() {return false;}
       bool ValidePt(const Pt2di &){return true;}
};



class cCC_GetVPt : public  cCC_NoActionOnNewPt
{
   public  :
       cCC_GetVPt() 
       {
       }

       void  OnNewPt(const Pt2di & aP)
       {
           mVPts.push_back(aP);
       }


       std::vector<Pt2di> mVPts;
};


template  <class Type>
          void    FiltrageCardCC(bool V4,Type & aTIm,int aValSelec,int aValAff,int aSeuilCard)
{
   Pt2di aSz = aTIm.sz();

   Im2D_Bits<1> aMasq1 = ImMarqueurCC(aSz);
   TIm2DBits<1> aTMasq1(aMasq1);

   Im2D_Bits<1> aMasq2 = ImMarqueurCC(aSz);
   TIm2DBits<1> aTMasq2(aMasq2);

   Pt2di aP;
   cCC_NoActionOnNewPt aNoAct;
   for(aP.x=0 ; aP.x<aSz.x ; aP.x++)
   {
      for(aP.y=0 ; aP.y<aSz.y ; aP.y++)
      {
          if ((aTIm.get(aP)==aValSelec) && (aTMasq1.get(aP)==1))
          {
               int aNb = OneZC(aP,V4,aTMasq1,1,0,aTIm,aValSelec,aNoAct);

               if (aNb<aSeuilCard)
               {
                    OneZC(aP,V4,aTIm,aValSelec,aValAff,aTMasq2,1,aNoAct);
               }
          }
      }
   }
}

#endif  //  _ELISE_IM_ALGO_CC











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
