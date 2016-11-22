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
#include "ReducHom.h"






double  cParamMerge::Gain // <0, veut dire on valide pas le noeud
               (
                     tNodIm * aN1,tNodIm * aN2,
                     const std::vector<cImagH*>&,
                     const std::vector<cImagH*>&,
                     const std::list<std::pair<cImagH*,cImagH*> >& aLPair,
                     int aNewNum
               )
{
    double aRes = 0.0;
    for
    (
          std::list<std::pair<cImagH*,cImagH*> >::const_iterator itL=aLPair.begin();
          itL!=aLPair.end();
          itL++
    )
    {
        cImagH* aIm1 =  itL->first;
        cImagH* aIm2 =  itL->second;
        cLink2Img *  aLnk12 = aIm1->GetLinkOfImage(aIm2);

        aRes += aLnk12->NbPts();
    }
    int aDepth = 1 + ElMax(aN1->Depth(),aN2->Depth());
    aRes = aRes / pow(2.0,aDepth);

    std::cout <<  aNewNum
              << "  Candidate " << aN1->Num() << " " << aN2->Num() << " " << aRes
               << " " << NameNode(aN1) << " " << NameNode(aN2) << " "
              << "\n";
    return aRes;
}

void cParamMerge::OnNewLeaf(tNodIm * aSingle)
{
   std::cout << "Creat Feuille " << aSingle->Val()->Name() << " " << aSingle->Num() << "\n";
}

void  cParamMerge::OnNewCandidate(tNodIm * aN1)
{
}

void cParamMerge::OnNewMerge(tNodIm * aN1)
{
    std::cout << aN1->Num() << " MERGE " ;
    for (int aK=0 ; aK<int(aN1->NbFils()) ; aK++)
        std::cout << " " << aN1->FilsK(aK)->Num();
    std::cout << "\n";
}

void cParamMerge::Vois(cImagH* anIm,std::vector<cImagH *> & aV)
{
    const tSetLinks & aLnks = anIm->Lnks();
    for (tSetLinks::const_iterator  itL=aLnks.begin(); itL!=aLnks.end(); itL++)
    {
         aV.push_back(itL->second->Dest());
    }
}

std::string NameNode(tNodIm * aN)
{
    cImagH * anI = aN->Val();
    return anI ? anI->Name() : "XXX" ;
}

// -------

template class cMergingNode<cImagH,cAttrLnkIm>;
template class cAlgoMergingRec<cImagH,cAttrLnkIm,cParamMerge>;
template class  ElHeap<cMergingNode<cImagH,cAttrLnkIm> *,cCmpMNode<cImagH,cAttrLnkIm> >;

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
