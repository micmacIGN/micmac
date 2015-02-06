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


/**************************************************************/
/*                                                            */
/*           cProjCple                                        */
/*                                                            */
/**************************************************************/

cProjCple::cProjCple(const Pt3dr & aP1,const Pt3dr & aP2,double aPds) :
    mP1  (aP1),
    mP2  (aP2),
    mPds (aPds)
{
}

const Pt3dr & cProjCple::P1() const {return mP1;}
const Pt3dr & cProjCple::P2() const {return mP2;}

cProjCple cProjCple::Spherik(const ElCamera & aCam1,const Pt2dr & aP1,const ElCamera & aCam2,const Pt2dr &aP2,double aPds)
{
    Pt3dr aQ1 =  aCam1.F2toDirRayonL3(aP1);
    Pt3dr aQ2 =  aCam2.F2toDirRayonL3(aP2);

    return cProjCple(vunit(aQ1),vunit(aQ2),aPds);
}

static Pt3dr Proj(const Pt3dr & aP) {return Pt3dr(aP.x/aP.z,aP.y/aP.z,1.0);}

cProjCple cProjCple::Projection(const ElCamera & aCam1,const Pt2dr & aP1,const ElCamera & aCam2,const Pt2dr &aP2,double aPds)
{
    Pt3dr aQ1 =  aCam1.F2toDirRayonL3(aP1);
    Pt3dr aQ2 =  aCam2.F2toDirRayonL3(aP2);

    return cProjCple(Proj(aQ1),Proj(aQ2),aPds);
}

/**************************************************************/
/*                                                            */
/*                  cProjListHom                              */
/*                                                            */
/**************************************************************/

cProjListHom::cProjListHom
(  
      const ElCamera & aCam1,
      const ElPackHomologue & aPack12,
      const ElCamera & aCam2,const ElPackHomologue & aPack21,
      bool Spherik
)  :
   mSpherik (Spherik)
{
    std::map<Pt2dr,Pt2dr> aMapP2OfP1;
    for 
    (
          ElPackHomologue::tCstIter itH1=aPack12.begin();
          itH1 !=aPack12.end();
          itH1++
    )
    {
         ElCplePtsHomologues aCple = itH1->ToCple();
         aMapP2OfP1[aCple.P1()] =  aMapP2OfP1[aCple.P2()];
    }
}












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
