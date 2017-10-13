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


//======================  ElQTRegionPlan ==========================

REAL ElQTRegionPlan::D2(const cElTriangleComp &)  const 
{
    ELISE_ASSERT(false,"No ElQTRegionPlan::D2(const cElTriangleComp &)");
    return 0.0;
}

//======================  ElQTRegPt ===============================
//======================  ElQTRegPt ===============================
//======================  ElQTRegPt ===============================

ElQTRegPt::ElQTRegPt(Pt2dr pt) : 
    _pt (pt) 
{
}
REAL ElQTRegPt::D2(const Box2dr & box)  const
{
     return box.SquareDist(_pt);
}
REAL ElQTRegPt::D2(const Pt2dr & pt)  const
{
     return square_euclid(_pt,pt);
}
REAL ElQTRegPt::D2(const SegComp & seg)  const
{
     return seg.square_dist_seg(_pt);
}


REAL ElQTRegPt::D2(const cElTriangleComp & aTri)  const 
{
    return aTri.square_dist(_pt);
}


//======================  ElQTRegSeg ===============================
//======================  ElQTRegSeg ===============================
//======================  ElQTRegSeg ===============================

ElQTRegSeg::ElQTRegSeg(Seg2d seg) :
    _seg (seg)
{
}
REAL ElQTRegSeg::D2(const Box2dr & box)  const
{
     return box.SquareDist(_seg);
}
REAL ElQTRegSeg::D2(const Pt2dr & pt)  const
{
     return _seg.square_dist_seg(pt);
}
REAL ElQTRegSeg::D2(const SegComp & seg2)  const
{
     return _seg.square_dist(SegComp::seg,seg2,SegComp::seg);
}

//======================  ElQTRegBox ===============================
//======================  ElQTRegBox ===============================
//======================  ElQTRegBox ===============================

ElQTRegBox::ElQTRegBox(const Box2dr & box) :
    _box (box)
{
}
REAL ElQTRegBox::D2(const Box2dr &b2)  const
{
     return _box.SquareDist(b2);
}

REAL ElQTRegBox::D2(const Pt2dr &pt)  const
{
     return _box.SquareDist(pt);
}

REAL ElQTRegBox::D2(const SegComp &seg)  const
{
     return _box.SquareDist(seg);
}

//======================  NewElQdtGen ===============================
//======================  NewElQdtGen ===============================
//======================  NewElQdtGen ===============================


Box2dr NewElQdtGen::BoxQdt(const Box2dr & box)
{
    INT sz = Pow_of_2_sup(round_up(dist8(box.sz())));
    return Box2dr
           (
              box._p0,
              box._p0 + Pt2dr(sz,sz)
           );
}



NewElQdtGen::NewElQdtGen
(
     Box2dr        BOX,
     INT           NBOBJMAX,
     REAL          SzMin
)  :
          _box      (BoxQdt(BOX)),
          _NbObjMax (NBOBJMAX),
          // _SzMin    (round_ni(SzMin))
          _SzMin    (SzMin)

{
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
