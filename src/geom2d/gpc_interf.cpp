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
#include "gpc.h"



     //   ======================INTERNE ==============

Pt2dr GpcToEl(const gpc_vertex &);
cElPolygone::tContour GpcToEl(const gpc_vertex_list &);

gpc_vertex  ElToGPC(const Pt2dr &);
gpc_vertex_list ElToGPC(const cElPolygone::tContour  &);

void GpcEl_Free(gpc_vertex_list &);
void GpcEl_Free(gpc_polygon     &);

/**************************************************/
/*                                                */
/*                   ::                           */
/*                                                */
/**************************************************/
Pt2dr GpcToEl(const gpc_vertex & aVert)
{
   return Pt2dr(aVert.x,aVert.y);
}

cElPolygone::tContour GpcToEl(const gpc_vertex_list & aGVL)
{
   cElPolygone::tContour aRes;
   aRes.reserve(aGVL.num_vertices);

   for (INT aK=0; aK<aGVL.num_vertices ; aK++)
       aRes.push_back(GpcToEl(aGVL.vertex[aK]));

   return aRes;
}
gpc_vertex  ElToGPC(const Pt2dr & aP)
{
   gpc_vertex aV;
   aV.x = aP.x;
   aV.y = aP.y;
   return aV;
}

gpc_vertex_list ElToGPC(const cElPolygone::tContour  & aCont)
{
   gpc_vertex_list aVL;
   aVL.num_vertices = (int) aCont.size();
   aVL.vertex = new  gpc_vertex [aCont.size()];
   for (int aK=0 ; aK<INT( aCont.size()) ; aK++)
       aVL.vertex[aK] = ElToGPC(aCont[aK]);

   return aVL;
}

void GpcEl_Free(gpc_vertex_list & aVL)
{
   delete [] aVL.vertex ;
}

void GpcEl_Free(gpc_polygon  & aPol)
{
   for (INT aK=0 ; aK<aPol.num_contours ; aK++)
       GpcEl_Free(aPol.contour[aK]);

   delete [] aPol.hole;
   delete [] aPol.contour;
}

/**************************************************/

/**************************************************/
/*                                                */
/*             cElPolygone                        */
/*                                                */
/**************************************************/


void cElPolygone::AddContour(const tContour & aCont,bool isHole)
{
   mContours.push_back(aCont);
   mIsHole.push_back(isHole);
}

const std::list<bool> &     cElPolygone::IsHole()
{
   return  mIsHole;
}

cElPolygone::cElPolygone()
{
}

cElPolygone::cElPolygone(const gpc_polygon & aPol)
{
   for (INT aK=0 ; aK<aPol.num_contours ; aK++)
      AddContour(GpcToEl(aPol.contour[aK]),aPol.hole[aK]!=0);
}

gpc_polygon cElPolygone::ToGPC()  const
{ 
   gpc_polygon aRes;
   aRes.num_contours = (int) mContours.size();

   aRes.hole = new int [mContours.size()];
   aRes.contour = new gpc_vertex_list [mContours.size()];

   std::list<tContour>::const_iterator itC = mContours.begin();
   std::list<bool>::const_iterator     itH = mIsHole.begin();
   INT aK=0;

   for(;itC!=mContours.end(); itC++,itH++,aK++)
   {
        aRes.hole[aK] = *itH;
        aRes.contour[aK] =  ElToGPC(*itC);
   }

    return aRes;
}
// gpc_vertex_list 
cElPolygone cElPolygone::operator * (const cElPolygone & aPol)  const
{
   return GenOp(aPol,INT(GPC_INT));
}
cElPolygone cElPolygone::operator + (const cElPolygone & aPol)  const
{
   return GenOp(aPol,INT(GPC_UNION));
}
cElPolygone cElPolygone::operator - (const cElPolygone & aPol)  const
{
   return GenOp(aPol,INT(GPC_DIFF));
}
cElPolygone cElPolygone::operator ^ (const cElPolygone & aPol)  const
{
   return GenOp(aPol,INT(GPC_XOR));
}

cElPolygone cElPolygone::GenOp(const cElPolygone & aPol,INT anOp)const
{
   gpc_polygon aGpcPol1 = ToGPC();
   gpc_polygon aGpcPol2 = aPol.ToGPC();
   gpc_polygon aGpcRes;

   gpc_polygon_clip (gpc_op(anOp),&aGpcPol1,&aGpcPol2,&aGpcRes);
   cElPolygone aRes(aGpcRes);

   GpcEl_Free(aGpcPol1);
   GpcEl_Free(aGpcPol2);
   gpc_free_polygon(&aGpcRes);

   return aRes;
}

const std::list<cElPolygone::tContour> & cElPolygone::Contours() const
{
   return mContours;
}

cElPolygone::tContour  cElPolygone::ContSMax() const
{
   std::vector<Pt2dr> aRes;
   double aSMax = 0;
   for 
   (
      std::list<tContour>::const_iterator itC=mContours.begin();
      itC!=mContours.end();
      itC++
   )
   {
       double aS = ElAbs(surf_or_poly(*itC));
       if (aS>aSMax)
       {
          aRes = *itC;
          aSMax = aS;
       }
   }
   return aRes;
}

double cElPolygone::DiamSimple() const
{
   tContour aC = ContSMax();
   double aD2Max = 0.0;
   for (int aK1 = 0 ; aK1 <int(aC.size()); aK1++)
       for (int aK2 = aK1+1 ; aK2 <int(aC.size()); aK2++)
       {
            ElSetMax(aD2Max,square_euclid(aC[aK1],aC[aK2]));
       }

  return sqrt(aD2Max);
}


double  cElPolygone::Surf() const
{
   double aRes = 0;

   std::list<bool>::const_iterator     itH = mIsHole.begin()   ;
   for 
   (
       std::list<tContour>::const_iterator itC = mContours.begin() ;
       itC != mContours.end()                                    ;
       itC++,itH++
   )
   {
       double aS = ElAbs(surf_or_poly(*itC));
       if (*itH)
          aRes -= aS;
       else
          aRes += aS;
      
   }

   return aRes;
}


cElPolygone cElPolygone::FromBox(const Box2dr & aBox)
{
    std::vector<Pt2dr> aCont;
    aCont.push_back(aBox._p0);
    aCont.push_back(Pt2dr(aBox._p1.x,aBox._p0.y));
    aCont.push_back(aBox._p1);
    aCont.push_back(Pt2dr(aBox._p0.x,aBox._p1.y));

    cElPolygone aRes;
    aRes.AddContour(aCont,false);
    return aRes;
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
