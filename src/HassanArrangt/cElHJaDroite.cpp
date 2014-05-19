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


     /*********************************************/
     /*                                           */
     /*           cElHJaDroite::cPaireSom         */
     /*                                           */
     /*********************************************/

cElHJaDroite::cPaireSom::cPaireSom
(
    tSomGrPl *      aS1,
    tSomGrPl *      aS2,
    const SegComp & aSeg
) :
  mS1   (aS1),
  mS2   (aS2),
  mPt   (aS1->attr().Pt()),
  mAbsc (aSeg.abscisse(mPt))
{
}


     /*********************************************/
     /*                                           */
     /*                cElHJaDroite               */
     /*                                           */
     /*********************************************/

cElHJaDroite::cElHJaDroite
(
    const ElSeg3D & aDr,
    cElHJaPlan3D & aP1,
    cElHJaPlan3D & aP2,
    INT aNbPl
)  :
    mDr    (aDr),
    mSegPl (Proj(mDr.P0()),Proj(mDr.P1())),
    mP1    (&aP1),
    mP2    (&aP2)
{
    aP1.AddInter(*this,aP2);
    aP2.AddInter(*this,aP1);
    for(INT aK=0; aK<aNbPl ; aK++)
       mInters.push_back(0);
}

ElSeg3D cElHJaDroite::Droite() {return mDr;}

void cElHJaDroite::AddPaire(tSomGrPl * aS1,tSomGrPl * aS2)
{
     mVPaires.push_back(cPaireSom(aS1,aS2,mSegPl));
}


void cElHJaDroite::AddPoint
     (
         cElHJaPoint & aPt,cElHJaPlan3D & aP3,
         tSomGrPl *s1,cElHJaPlan3D * aP1,
         tSomGrPl *s2,cElHJaPlan3D * aP2
     )
{
   mInters.at(aP3.Num()) = & aPt;
   ELISE_ASSERT(aP1->Num()==mP1->Num(),"cElHJaDroite::cRelDP::AddSomGr");
   ELISE_ASSERT(aP2->Num()==mP2->Num(),"cElHJaDroite::cRelDP::AddSomGr");
   AddPaire(s1,s2);
}


void cElHJaDroite::MakeIntersectionEmprise
     (
         const tEmprPlani & anEmpr
     )
{
     INT aNbSom = (int) anEmpr.size();
     for (INT aK=0 ; aK<aNbSom ; aK++)
     {
         const cElHJaSomEmpr & s1 = anEmpr[aK];
         const cElHJaSomEmpr & s2 = anEmpr[(aK+1)%aNbSom];

         Pt2dr aP1 = mSegPl.to_rep_loc(s1.Pos());
         Pt2dr aP2 = mSegPl.to_rep_loc(s2.Pos());
         if ((aP1.y>0) != (aP2.y>0))
         {
	      REAL aX = aP1.x -aP1.y *((aP2.x-aP1.x) / (aP2.y-aP1.y));

              Pt2dr aPInter(aX,0.0);
	      aPInter = mSegPl.from_rep_loc(aPInter);
              cElHJaSomEmpr aSom (aPInter,&s1);
              AddPaire
              (
                 mP1->SomGrEmpr(aSom),
                 mP2->SomGrEmpr(aSom)
              );
         }
     }
}


class cCmpPairSom
{
	public :
             bool operator ()
             (
                const cElHJaDroite::cPaireSom & aS1,
                const cElHJaDroite::cPaireSom & aS2
	     )
	     {
                   return aS1.mAbsc < aS2.mAbsc;
	     }

};

void cElHJaDroite::AddArcsInterieurInGraphe
     (
          const std::vector<Pt2dr> & aEmprInit
     )
{
     cCmpPairSom aCmp;

     std::sort(mVPaires.begin(),mVPaires.end(),aCmp);
     for (INT aK=0 ; aK<INT(mVPaires.size()-1) ; aK++)
     {
         const cPaireSom & aPSA = mVPaires[aK];
         const cPaireSom & aPSB = mVPaires[aK+1];

	 Pt2dr aPt = (aPSA.mPt+aPSB.mPt) / 2.0;
	 if (PointInPoly(aEmprInit,aPt))
	 {
             ELISE_ASSERT((aPSA.mS1==aPSB.mS1)==(aPSA.mS2==aPSB.mS2),"cElHJaDroite::AddArcsInterieurInGraphe");
	     if (aPSA.mS1!=aPSB.mS1)
	     {
                tArcGrPl * aArcPl1 = mP1->NewArcInterieur(aPSA.mS1,aPSB.mS1);
                tArcGrPl * aArcPl2 = mP2->NewArcInterieur(aPSA.mS2,aPSB.mS2);

                for (INT aK=0 ; aK<2 ; aK++)
                {
                    aArcPl1->attr().SetArcHom(aArcPl2);
                    aArcPl2->attr().SetArcHom(aArcPl1);

		    // On swap aux arc reciproque
		    aArcPl1 = &(aArcPl1->arc_rec());
		    aArcPl2 = &(aArcPl2->arc_rec());
                }
             }
	 }
     }
}


// Structure provisoire pour la creation des arretes internes
/*
class cSomDrConstrGraphe
{
      public :
         cSomDrConstrGraphe()
         Pt2dr      mPos;
         REAL       mAbsc;
         tSomGrPl * mSom;
};


void cElHJaDroite::AddSegsInGraphe
     (
         const tEmprPlani & anEmpr
     )
{
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
