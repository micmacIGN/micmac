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


#include "OriTiepRed.h"
NS_OriTiePRed_BEGIN

// bool BUGTR = true;



/**********************************************************************/
/*                                                                    */
/*                         cAppliTiepRed                              */
/*                                                                    */
/**********************************************************************/

void cAppliTiepRed::VonGruber()
{
     // int aNbSel0 =  mListSel.size();
     for (int aK1=0 ; aK1<int(mVecCam.size()) ; aK1++)
     {
         std::vector<tPMulTiepRedPtr> aVK1;
         for (int aKP=0 ; aKP<int(mVPM.size()) ; aKP++)
         {
             if (mVPM[aKP]->Merge()->IsInit(aK1))
                aVK1.push_back(mVPM[aKP]);
         }
         // std::cout << "VonGrube " << aK1 << " " << aVK1.size() << "\n";



         for (int aK2=aK1+1 ; aK2<int(mVecCam.size()) ; aK2++)
         {
              VonGruber(aVK1,mVecCam[aK1],mVecCam[aK2]);
         }
     }

     // std::cout << " VonGruber , " << aNbSel0  << " => " << mListSel.size() << "\n";
}


/*
template <class Obj,class Prim,class FPrim>
          ::cTplValGesInit<Obj>  ElQT<Obj,Prim,FPrim>::NearestObjSvp
          (
                Pt2dr aP,
                double aDistInit,
                double aDistMax
          )
{
    int aNbMax = round_up(log2(aDistMax/aDistInit));
    aDistInit = aDistMax/pow(2.0,aNbMax);

    std::list<Obj> aLObj = KPPVois(aP,1,aDistInit,2.0,aNbMax);

    cTplValGesInit<Obj> aRes;
    if (!aLObj.empty())
       aRes.SetVal(*(aLObj.begin()));
    return aRes;
}
*/


// VK1 => tout les points multiples qui contiennet Im1
void cAppliTiepRed::VonGruber(const std::vector<tPMulTiepRedPtr> & aVK1,cCameraTiepRed * aCam1,cCameraTiepRed * aCam2)
{
    std::vector<tPMulTiepRedPtr> aVK1K2;

    // On met tous les points contenant Im2 et suffisemment precis
    for (int aKP=0 ; aKP<int(aVK1.size()) ; aKP++)
    {
        if (aVK1[aKP]->Merge()->IsInit(aCam2->Num())  && (aVK1[aKP]->Prec() < (0.5+2*StdPrec())))
           aVK1K2.push_back(aVK1[aKP]);
    }
    if (aVK1K2.size() ==0) 
       return;

    // Pour tous les points visant I1I2, on initialise la distance et un gain
    // les points deja retenu sont mis dans le QDT
    double aDistMax = euclid(mBoxLocQT.sz());
    int aNbInside=0;
    for (int aKP=0 ; aKP<int(aVK1K2.size()) ; aKP++)
    {
        tPMulTiepRedPtr aSom = aVK1K2[aKP];
        aSom->SetDistVonGruber(aDistMax,*this);
        if (aSom->Selected())
        {
           mQT->insert(aSom);
           aNbInside++;
        }
    }

    // Tous les points non selectionne  sont modifie 
    // la dist moy est une heuristique de recherche

    double aDistMoy = sqrt(mBoxLocQT.surf()/(1+aNbInside));

    for (int aKP=0 ; aKP<int(aVK1K2.size()) ; aKP++)
    {
        tPMulTiepRedPtr aSom = aVK1K2[aKP];
        double aDist = aDistMax;
        if ((!aSom->Selected()) && (aNbInside > 0))
        {
           cTplValGesInit<tPMulTiepRedPtr> aTlOb = mQT->NearestObjSvp(aSom->Pt(),aDistMoy,aDistMax);
           if (aTlOb.IsInit())
           {
               tPMulTiepRedPtr aNearObj = aTlOb.Val();
               aDist = euclid(aNearObj->Pt(),aSom->Pt());
               aSom->ModifDistVonGruber(aDist,*this);
           }
        }
    }
    bool Cont=true;
    double aSeuilDist = mDistPMul * mResolQT * mMulVonGruber;

    // boucle de recherhe
    int aNbPVG=0;
    while (Cont)
    {
        double aMaxDist  = 0;
        double aGainMax  = 0;
        tPMulTiepRedPtr aMaxSom = 0;
        // Recherche du point ayant le meilleur gain, calcul du point le plus loin
        for (int aKP=0 ; aKP<int(aVK1K2.size()) ; aKP++)
        {
             tPMulTiepRedPtr aSom = aVK1K2[aKP];
             if (!aSom->Selected()) 
             {
                 if (aSom->Gain() > aGainMax)
                 {
                      aGainMax = aSom->Gain();
                      aMaxSom = aSom;
                 }
                 aMaxDist = ElMax(aMaxDist,aSom->DMin());
             }
        }

        // si la distance du point le plus loin est plus gde que le seuil,
        // on met a jour avec le meilleur point
        if ((aMaxDist> aSeuilDist) && aMaxSom)
        {
           aMaxSom->SetSelected();
           mListSel.push_back(aMaxSom);
           for (int aKP=0 ; aKP<int(aVK1K2.size()) ; aKP++)
           {
                tPMulTiepRedPtr aSom = aVK1K2[aKP];
                aSom->ModifDistVonGruber(euclid(aSom->Pt(),aMaxSom->Pt()),*this);
           }
           aNbPVG++;
        }
        else
        {
            Cont = false;
        }
    }

    // std::cout << "HHHHhh " << aVK1K2.size() << " NbVG " << aNbPVG << "\n";
    // if (aNbPVG) getchar();

    mQT->clear();
}


NS_OriTiePRed_END


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
