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

#include "general/sys_dep.h"

#if (ELISE_X11)

#include "Vino.h"

void cAppli_Vino::Efface(const Box2di & aBox)
{
   int aR = 0;
   mScr->VisuIm(aBox._p0-Pt2di(aR,aR),aBox._p1+Pt2di(aR,aR),false);
}

void cAppli_Vino::EffaceMessages(std::vector<Box2di> &aVBox)
{
    for (int aKB=0; aKB<int(aVBox.size()) ; aKB++)
    {
        Efface(aVBox[aKB]);
    }
    aVBox.clear();
}
void cAppli_Vino::EffaceMessageVal()
{
   EffaceMessages(mVBoxMessageVal);
}

void cAppli_Vino::EffaceMessageRelief()
{
   EffaceMessages(mVBoxMessageRelief);
}


Box2di cAppli_Vino::PutMessage
       (
             Pt2dr aPText ,
             const std::string & aMes,
             int aCoulText,
             Pt2dr aSzRelief,
             int aCoulRelief
       )
{
    Box2di aRes;
    Pt2dr aRab(2,2);

    Pt2di aSzV =  mW->SizeFixedString(aMes);
    Pt2dr aP0 = Pt2dr(aPText.x,aPText.y-aSzV.y) + Pt2dr(0,-2);
    Pt2dr aP1 = Pt2dr(aPText.x+aSzV.x,aPText.y) + Pt2dr(1,2) ;

    if (aCoulRelief>=0)
    {
       aP0 = aP0   - aSzRelief;
       aP1 = aP1   + aSzRelief;
       mW->fill_rect(aP0,aP1,mW->pdisc()(aCoulRelief));

       //    aRes._p1 = aRes._p0aRes._p1 + aSzRelief;
    }

    mW->fixed_string(Pt2dr(aPText),aMes.c_str(),mW->pdisc()(aCoulText),true);

    return Box2di(round_down(aP0-aRab),round_up(aP1+aRab));
}


void   cAppli_Vino::PutMessageRelief(int aK,const std::string & aMes)
{
     Box2di aBox = PutMessage
                   (
                       Pt2dr(20,45 + aK * 50),
                       aMes,
                       P8COL::black,
                       Pt2dr(15,15),
                       P8COL::magenta
                   );
     mVBoxMessageRelief.push_back(aBox);
}




#endif



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
