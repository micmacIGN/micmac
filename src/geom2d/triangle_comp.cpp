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




// Modifie l'intervale entier [aXMinLoc,aXMaxLoc [, 
// si aXMaxLoc <= aXMinLoc  -> intervalle vide

void ModifInterv(int & aXMinLoc,int & aXMaxLoc,int aY,const SegComp & aSeg)
{
    double aY0 = aSeg.p0().y;
    double aY1 = aSeg.p1().y;

    double aX0 = aSeg.p0().x;
    double aX1 = aSeg.p1().x;

    if (aY0==aY1)
    {
        if (aX0==aX1) return;  // segment vide

        if ((aX1>aX0) == (aY>=aY0)) return;  // le demi plan contient le segment

         aXMaxLoc = aXMinLoc;  return;
    }

    int aXInter = round_ni(aX0 + ( (aY-aY0) / (aY1-aY0)  ) * (aX1-aX0));

    if (aY1>aY0) 
       ElSetMin(aXMaxLoc,aXInter);
    else
       ElSetMax(aXMinLoc,aXInter);
}


extern void RasterTriangle(const cElTriangleComp & aTri,std::vector<cSegEntierHor> & aRes)
{
    aRes.clear();

    int aYMinGlob = round_ni(ElMin3(aTri.P0().y,aTri.P1().y,aTri.P2().y));
    int aYMaxGlob = round_ni(ElMax3(aTri.P0().y,aTri.P1().y,aTri.P2().y));

    //int aYMinGlob = round_down(ElMin3(aTri.P0().y,aTri.P1().y,aTri.P2().y));
    //int aYMaxGlob = round_up(ElMax3(aTri.P0().y,aTri.P1().y,aTri.P2().y));

    int aXMinGlob = round_ni(ElMin3(aTri.P0().x,aTri.P1().x,aTri.P2().x));
    int aXMaxGlob = round_ni(ElMax3(aTri.P0().x,aTri.P1().x,aTri.P2().x));

    //int aXMinGlob = round_down(ElMin3(aTri.P0().x,aTri.P1().x,aTri.P2().x));
    //int aXMaxGlob = round_up(ElMax3(aTri.P0().x,aTri.P1().x,aTri.P2().x));

    for (int aY=aYMinGlob  ; aY<aYMaxGlob  ; aY++)
    {
        int aXMinLoc = aXMinGlob;
        int aXMaxLoc = aXMaxGlob;

        ModifInterv(aXMinLoc,aXMaxLoc,aY,aTri.S01());
        ModifInterv(aXMinLoc,aXMaxLoc,aY,aTri.S12());
        ModifInterv(aXMinLoc,aXMaxLoc,aY,aTri.S20());

        if (aXMaxLoc>aXMinLoc)
        {
            cSegEntierHor aSH;
            aSH.mP0 = Pt2di(aXMinLoc,aY);
            aSH.mNb = aXMaxLoc - aXMinLoc;
            aRes.push_back(aSH);
        }
    }
}




/*******************************************************/
/*                                                     */
/*              cElTriangleComp                        */
/*                                                     */
/*******************************************************/

const SegComp & cElTriangleComp::S01() const {return mS01;}
const SegComp & cElTriangleComp::S12() const {return mS12;}
const SegComp & cElTriangleComp::S20() const {return mS20;}


Pt2dr cElTriangleComp::P0() const { return mS01.p0(); }
Pt2dr cElTriangleComp::P1() const { return mS12.p0(); }
Pt2dr cElTriangleComp::P2() const { return mS20.p0(); }

bool cElTriangleComp::Inside(const Pt2dr & aP) const
{
    return    (mS01.ordonnee(aP) > 0)
           && (mS12.ordonnee(aP) > 0)
           && (mS20.ordonnee(aP) > 0) ;
}

Pt3dr cElTriangleComp::CoordBarry(const Pt2dr & aP) const
{
   return Pt3dr
	  (
	       mS12.ordonnee(aP) /  mS12.ordonnee(P0()),
	       mS20.ordonnee(aP) /  mS20.ordonnee(P1()),
	       mS01.ordonnee(aP) /  mS01.ordonnee(P2())
	  );
}


ElMatrix<double>  cElTriangleComp::MatCoeffBarry() const
{
   ElMatrix<double> aRes(3,3);

   mS12.CoeffFoncOrdonnee(P0(),aRes(0,0),aRes(1,0),aRes(2,0));
   mS20.CoeffFoncOrdonnee(P1(),aRes(0,1),aRes(1,1),aRes(2,1));
   mS01.CoeffFoncOrdonnee(P2(),aRes(0,2),aRes(1,2),aRes(2,2));


   return aRes;
}

Pt2dr  cElTriangleComp::FromCoordBarry(REAL aPds0,REAL aPds1,REAL aPds2) const
{
    return (P0()*aPds0 + P1()*aPds1 + P2()*aPds2)/(aPds0+aPds1+aPds2);
}

bool cElTriangleComp::ToSwap(const Pt2dr & aP0,const  Pt2dr & aP1,const Pt2dr & aP2)
{
	return ((aP1-aP0) ^ (aP2-aP0)) < 0;
}


SegComp cElTriangleComp::ReorderDirect(Pt2dr & aP0, Pt2dr & aP1,Pt2dr & aP2)
{
   if (ToSwap(aP0,aP1,aP2))
   {
	   ElSwap(aP1,aP2);
   }

   return SegComp(aP0,aP1);
}


cElTriangleComp::cElTriangleComp
(
     Pt2dr aP0,
     Pt2dr aP1,
     Pt2dr aP2
)  :
   mS01 (ReorderDirect(aP0,aP1,aP2)),
   mS12 (aP1,aP2),
   mS20 (aP2,aP0)
{
}

REAL cElTriangleComp::square_dist(Pt2dr pt) const
{
    if (Inside(pt))
       return 0.0;

    return ElMin3
	   (
	       mS01.square_dist_seg(pt),
	       mS12.square_dist_seg(pt),
	       mS20.square_dist_seg(pt)
	   );
}




/*******************************************************/
/*                                                     */
/*                                                     */
/*******************************************************/

/*
static Pt2di aSZ (400,400);
static Pt2di aRAB (200,200);

static Pt2dr RanCorner()
{
    return Pt2dr(aSZ.x*NRrandom3(),aSZ.y*NRrandom3());
}

static Pt2dr RanDelta()
{
    return Pt2dr(70*NRrandom3(),70*NRrandom3());
}



static cElTriangleComp  RanTri()
{
   return cElTriangleComp(RanCorner(),RanCorner(),RanCorner());
}


class  cIdTri
{
	public :
          cElTriangleComp operator ()(cElTriangleComp * aTri)
          {
               return *aTri;
          }
};
			

#include "algo_geom/qdt.h"
#include "algo_geom/qdt_implem.h"

void cElTriangleComp::Test()
{
    cIdTri aId;

    Video_Win  aW = Video_Win::WStd(aSZ,1.0);


    for (INT aK  =0 ; aK < 100 ; aK++)
    {
        aW.clear();
        ElQT<cElTriangleComp *,cElTriangleComp,cIdTri> aQt(aId,Box2di(-aRAB,aSZ+aRAB),8,20.0);
	for (INT aK2 = 0; aK2 < 10 ; aK2++)
	{
             Pt2dr aP = RanCorner();
	     cElTriangleComp aTri(aP+RanDelta(),aP+RanDelta(),aP+RanDelta());
	     aW.draw_seg(aTri.P0(),aTri.P1(),aW.pdisc()(P8COL::cyan));
	     aW.draw_seg(aTri.P1(),aTri.P2(),aW.pdisc()(P8COL::cyan));
	     aW.draw_seg(aTri.P2(),aTri.P0(),aW.pdisc()(P8COL::cyan));

	     aQt.insert(new cElTriangleComp(aTri));
	}

	
	for (INT aK2 = 0 ; aK2 < 3 ; aK2 ++)
	{
             Pt2dr aP = RanCorner();
	     REAL aR = 100.0 * NRrandom3();
             aW.draw_circle_loc(aP,aR,aW.pdisc()(P8COL::red));
	     std::set<cElTriangleComp *> aSet;
	     aQt.RVoisins(aSet,aP,aR);

	     for 
             (
                 std::set<cElTriangleComp *>::const_iterator It=aSet.begin();
                 It != aSet.end();
                 It++
             )
             {
                 cElTriangleComp aTr2 = **It;
	         aW.draw_seg(aTr2.P0(),aTr2.P1(),aW.pdisc()(P8COL::yellow));
	         aW.draw_seg(aTr2.P1(),aTr2.P2(),aW.pdisc()(P8COL::yellow));
	         aW.draw_seg(aTr2.P2(),aTr2.P0(),aW.pdisc()(P8COL::yellow));
             }

	}
        getchar();
    }


    for (INT aK  =0 ; aK < 100 ; aK++)
    {
        aW.clear();
	cElTriangleComp aTri = RanTri();


        for (INT aK2  =0 ; aK2 < 100 ; aK2++)
	{
           Pt2dr aP = RanCorner();
	   Pt3dr aCB = aTri.CoordBarry(aP);
	   Pt2dr aQ = aTri.FromCoordBarry(aCB.x,aCB.y,aCB.z);
	   BENCH_ASSERT(ElAbs(aCB.x+aCB.y + aCB.z-1.0)<1e-7);
	   BENCH_ASSERT(euclid(aP,aQ)<1e-7);
	   REAL aD  = aTri.square_dist(aP);
	   INT aCoul = aTri.Inside(aP) ? P8COL::green : P8COL::blue;
           aW.draw_circle_loc(aP,ElMax(1.0,sqrt(aD)),aW.pdisc()(aCoul));
        }
        for (INT aK2  =0 ; aK2 < 10 ; aK2++)
	{
            Pt2dr aP0= RanCorner();
            Pt2dr aP1= RanCorner();
	    Box2dr aB(aP0,aP1);

	    INT aCoul = P8COL::red ;
	    if (aB.Intersecte(aTri)) aCoul = P8COL::yellow;
	    if (aB.Include(aTri))    aCoul = P8COL::white;

	    Pt2dr Corn[4];
	    aB.Corners(Corn);
	    for (INT aC=0 ; aC<4 ; aC++)
                aW.draw_seg(Corn[aC],Corn[(aC+1)%4],aW.pdisc()(aCoul));
	}
	aW.draw_seg(aTri.P0(),aTri.P1(),aW.pdisc()(P8COL::cyan));
	aW.draw_seg(aTri.P1(),aTri.P2(),aW.pdisc()(P8COL::cyan));
	aW.draw_seg(aTri.P2(),aTri.P0(),aW.pdisc()(P8COL::cyan));
        getchar();
    }
}

*/





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
