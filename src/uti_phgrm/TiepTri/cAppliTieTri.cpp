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

class cCmpPt2diOnEuclid
{
   public : 
       bool operator () (const Pt2di & aP1, const Pt2di & aP2)
       {
                   return euclid(aP1) < euclid(aP2) ;
       }
};



cAppliTieTri::cAppliTieTri
(
              cInterfChantierNameManipulateur * anICNM,
              const std::string & aDir,
              const std::string & anOri,
              const cXml_TriAngulationImMaster & aTriang
)  :
     mICNM       (anICNM),
     mDir        (aDir),
     mOri        (anOri),
     mWithW      (false),
     mDisExtrema (3.0)

{
   mMasIm = new cImMasterTieTri(*this,aTriang.NameMaster());

   for (int aK=0 ; aK<int(aTriang.NameSec().size()) ; aK++)
   {
      mImSec.push_back(new cImSecTieTri(*this,aTriang.NameSec()[aK]));
   }

   int aDE = round_up(mDisExtrema);
   Pt2di aP;
   for (aP.x=-aDE ; aP.x <= aDE ; aP.x++)
   {
       for (aP.y=-aDE ; aP.y <= aDE ; aP.y++)
       {
            if (euclid(aP) <= mDisExtrema)
               mVoisExtr.push_back(aP);
       }
   }
   cCmpPt2diOnEuclid aCmp;
   std::sort(mVoisExtr.begin(),mVoisExtr.end(),aCmp);

}


void cAppliTieTri::DoAllTri(const cXml_TriAngulationImMaster & aTriang)
{
    for (int aK=0 ; aK<int(aTriang.Tri().size()) ; aK++)
    {
        DoOneTri(aTriang.Tri()[aK]);
    }
}


void cAppliTieTri::DoOneTri(const cXml_Triangle3DForTieP & aTri )
{
    mMasIm->LoadTri(aTri);
    for (int aK=0 ; aK<int(aTri.NumImSec().size()) ; aK++)
    {
        mImSec[aTri.NumImSec()[aK]]->LoadTri(aTri);
    }
}


void  cAppliTieTri::SetSzW(Pt2di aSzW, int aZoom)
{
    mSzW = aSzW;
    mZoomW = aZoom;
    mWithW = true;
}




cInterfChantierNameManipulateur * cAppliTieTri::ICNM()      {return mICNM;}
const std::string &               cAppliTieTri::Ori() const {return mOri;}
const std::string &               cAppliTieTri::Dir() const {return mDir;}

Pt2di cAppliTieTri::SzW() const {return mSzW;}
int   cAppliTieTri::ZoomW() const {return mZoomW;}
bool  cAppliTieTri::WithW() const {return mWithW;}


cImMasterTieTri * cAppliTieTri::Master() {return mMasIm;}




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
