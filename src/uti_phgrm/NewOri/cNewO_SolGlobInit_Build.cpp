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


#include "NewOri.h"
#include "SolInitNewOri.h"


/**********************************************************/
/*                                                        */
/*          Composantes connexes                          */
/*                                                        */
/**********************************************************/

void cAppli_NewSolGolInit::ResetFlagCC()
{

    for (int  aK3=0 ; aK3<int (mV3.size()) ; aK3++)
    {
         mV3[aK3]->Flag().set_kth_false(mFlag3CC);
    }
}


void cAppli_NewSolGolInit::NumeroteCC()
{
    int aNumCC = 0;
    for (int  aK3=0 ; aK3<int (mV3.size()) ; aK3++)
    {
        cNOSolIn_Triplet * aTri0 = mV3[aK3];

        if ( !aTri0->Flag().kth(mFlag3CC))
        {
            // std::vector<cNOSolIn_Triplet*> * aCC = new std::vector<cNOSolIn_Triplet*>;
            cCC_TripSom * aNewCC3S = new cCC_TripSom;
            aNewCC3S->mNumCC = aNumCC;
            mVCC.push_back(aNewCC3S);
            std::vector<cNOSolIn_Triplet*> * aCC3 = &(aNewCC3S->mTri);
            std::vector<tSomNSI *> * aCCS = &(aNewCC3S->mSoms);

            // Calcul des triplets 
            aCC3->push_back(aTri0);
            aTri0->Flag().set_kth_true(mFlag3CC);
            aTri0->NumCC() = aNumCC;
            int aKCur = 0;
            while (aKCur!=int(aCC3->size()))
            {
               cNOSolIn_Triplet * aTri1 = (*aCC3)[aKCur];
               for (int aKA=0 ; aKA<3 ; aKA++)
               {
                  std::vector<cLinkTripl> &  aLnk = aTri1->KArc(aKA)->attr().ASym()->Lnk3();
                  for (int aKL=0 ; aKL<int(aLnk.size()) ; aKL++)
                  {
                     if (SetFlagAdd(*aCC3,aLnk[aKL].m3,mFlag3CC))
                     {
                          aLnk[aKL].m3->NumCC() = aNumCC;
                     }
/*
                     cNOSolIn_Triplet * aTri2 = aLnk[aKL].m3;
                     if (! aTri2->Flag().kth(mFlag3CC))
                     {
                        aCC3->push_back(aTri2);
                        aTri2->Flag().set_kth_true(mFlag3CC);
                        aTri2->NumCC() = aNumCC;
                     }
*/
                  }
               }
               aKCur++;
            }

            // Calcul des sommets 
            int aFlagSom = mGr.alloc_flag_som();
            for (int aKT=0 ; aKT<int(aCC3->size()) ; aKT++)
            {
                cNOSolIn_Triplet * aTri = (*aCC3)[aKT];
                for (int aKS=0 ;  aKS<3 ; aKS++)
                {
                    SetFlagAdd(*aCCS,aTri->KSom(aKS),aFlagSom);
                }
            }
            FreeAllFlag(*aCCS,aFlagSom);
            mGr.free_flag_som(aFlagSom);

            std::cout << "NbTriii " << aCC3->size() << " NbSooom " << aCCS->size() << "\n";
            aNumCC++;
        }
    }
    FreeAllFlag(mV3,mFlag3CC);
    // ResetFlagCC();
    std::cout << "NUMMMCCCC " <<  aNumCC << "\n";
}

/**********************************************************/
/*                                                        */
/*          Orientation                                   */
/*                                                        */
/**********************************************************/

bool  cAppli_NewSolGolInit::AddSOrCur(tSomNSI * aSom)
{
    return SetFlagAdd(mVSOrCur,aSom,mFlagSOrCur);
}


void cAppli_NewSolGolInit::CalculOrient(cNOSolIn_Triplet * aGerm)
{
    mFlagSOrCur = mGr.alloc_flag_som();
    
    
    for (int aKS=0 ; aKS<3 ; aKS++)
    {
         AddSOrCur(aGerm->KSom(aKS));
        // aGerm->
    }


     mGr.free_flag_som(mFlagSOrCur);
     mVSOrCur.clear();
     FreeAllFlag(mVSOrCur,mFlagSOrCur);
}



void  cAppli_NewSolGolInit::CalculOrient(cCC_TripSom * aCC)
{
     cNOSolIn_Triplet * aGerm0 =0;
     double aBesCoherCost = 1e30;

     for (int aK=0 ; aK<(aCC->mTri.size()) ; aK++)
     {
         cNOSolIn_Triplet * aTri = aCC->mTri[aK];
         if (aTri->CostArc()<aBesCoherCost)
         {
             aBesCoherCost = aTri->CostArc();
             aGerm0 = aTri;
         }
     }

     CalculOrient(aGerm0);
}


void  cAppli_NewSolGolInit::CalculOrient()
{
    for (int aKC=0 ;  aKC<int(mVCC.size()) ; aKC++)
       CalculOrient(mVCC[aKC]);
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
