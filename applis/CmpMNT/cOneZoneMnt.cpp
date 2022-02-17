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
// #include "anag_all.h"

#include "CmpMNT.h"

namespace NS_CmpMNT
{


cOneZoneMnt::cOneZoneMnt
(
    const cAppliCmpMnt& anAppli,
    const cZoneCmpMnt & anArgZone,
    const cCompareMNT & aCmpMnt
) :
   mAppli (anAppli),
   mArgZ  (anArgZone),
   mAGlob (aCmpMnt),
   mMasq  (1,1)
{
   if (mArgZ.ContourPolyCM().IsInit())
      Init(mArgZ.ContourPolyCM().Val());
   else if (mArgZ.BoxContourCM().IsInit())
      Init(mArgZ.BoxContourCM().Val());
   else
   {
        ELISE_ASSERT(false,"cOneZoneMnt::cOneZoneMnt");
   }
}

const Box2di & cOneZoneMnt::Box() const
{
   return mBox;
}

void cOneZoneMnt::InitFlux(Flux_Pts aFlx)
{
   Pt2di aPMin,aPMax;

   ELISE_COPY
   (
       aFlx,
       Virgule(FX,FY),
       Virgule
       (
           VMin(aPMin.x)| VMax(aPMax.x),
           VMin(aPMin.y)| VMax(aPMax.y)
       )
   );
   int aRab = mAGlob.RabLoad().Val();
   Pt2di aPRab(aRab,aRab);

   aPMin = Sup(Pt2di(0,0),aPMin-aPRab);
   aPMax = Inf(aPMax+Pt2di(1,1)+aPRab,mAppli.Sz());

   mBox =  Box2di(aPMin,aPMax);
   mMasq = Im2D_Bits<1>(mBox.sz().x,mBox.sz().y,0);

   ELISE_COPY
   (
       aFlx.chc(Virgule(FX-aPMin.x,FY-aPMin.y)),
       1,
       mMasq.out()
   );
}

void cOneZoneMnt:: Init(const cContourPolyCM & aCPCm)
{
    ElList<Pt2di>  aLpt = TplStd2Elise(aCPCm.Pts());

    InitFlux(polygone(aLpt));
}   

void cOneZoneMnt:: Init(const Box2dr & aBox)
{
    InitFlux(rectangle(aBox._p0,aBox._p1));
}   


Im2D_Bits<1>   cOneZoneMnt::Masq() const
{
   return mMasq;
}

const std::string & cOneZoneMnt::Nom() const
{
   return mArgZ.NomZone().Val();
}


};



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
