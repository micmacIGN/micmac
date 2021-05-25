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

SegComp::SegComp() {}

SegComp::SegComp(Pt2dr p0,Pt2dr p1):
      Seg2d        (p0,p1),
     _tangente     (vunit(p1-p0,_length)),
     _normale      (rot90(_tangente)),
     _c            (-scal(p1,_normale)),
     _a1           (abscisse(p1))
{
}

SegComp::SegComp(const Seg2d & aSeg)
{
    *this = SegComp(aSeg.p0(),aSeg.p1());
}

REAL SegComp::ordonnee(Pt2dr pt) const
{
    return scal(_normale,pt) + _c;
}

void SegComp::CoeffFoncOrdonnee
     (
          const Pt2dr& aP,
          double & aCx,
          double & aCy,
          double & aC0
     ) const
{
    double aDiv = ordonnee(aP);
    ELISE_ASSERT(aDiv!=0,"SegComp::ordonnee");
    aCx = _normale.x / aDiv;
    aCy = _normale.y / aDiv;
    aC0 = _c / aDiv;
}


REAL SegComp::ordonnee(Pt3dr aPt) const
{
    return _normale.x*aPt.x+  _normale.y*aPt.y + _c*aPt.z;
}

Fonc_Num  SegComp::ordonnee(Pt3d<Fonc_Num>  aPt) const
{
    return _normale.x*aPt.x+  _normale.y*aPt.y + _c*aPt.z;
}




REAL SegComp::abscisse(Pt2dr pt) const
{
    return scal(_tangente,pt-p0());
}

Pt2dr SegComp::to_rep_loc(Pt2dr pt) const
{
    return Pt2dr(abscisse(pt),ordonnee(pt));
}

Pt2dr SegComp::from_rep_loc(Pt2dr pt) const
{
    return   p0() 
           + _tangente * pt.x 
           + _normale  * pt.y;
}

bool SegComp::in_bande(Pt2dr pt,ModePrim mode) const
{
     switch(mode)
     {
          case droite : 
               return true;

          case demi_droite : 
               return abscisse(pt) >= 0;
          default :  
                REAL a = abscisse(pt);
                return (a>=0) && (a<= _a1);
     }
}
const Pt2dr SegComp::NoPoint(1e30,1e60);


Pt2dr   SegComp::inter(const SegComp & s2,bool & got) const
{
      REAL o0 =  ordonnee(s2.p0());
      REAL o1 =  ordonnee(s2.p1());

      got =  (o0!=o1);
      return  (got)                            ?
              barry(o1/(o1-o0),s2.p0(),s2.p1())  :
              NoPoint                          ;
}

Pt2dr   SegComp::inter(ModePrim m1,const SegComp & s2,ModePrim m2,bool & got) const
{
      Pt2dr res = inter(s2,got);
      if (got)
          got = in_bande(res,m1) && s2.in_bande(res,m2);
      return res;
}



bool SegComp::BoxContains(Pt2dr pt,REAL DLong,REAL DLarg) const
{
    REAL aVal = abscisse(pt);
    if ((aVal<-DLong) || (aVal>_length+DLong))
       return false;
    aVal = ordonnee(pt);
    return (aVal>-DLarg) && (aVal<DLarg);
}

bool SegComp::BoxContains(const Seg2d & s2 ,REAL DLong,REAL DLarg) const
{
    return     BoxContains(s2.p0(),DLong,DLarg)
           &&  BoxContains(s2.p1(),DLong,DLarg);
}


// Rho = x Cos T + y Sin T

Pt2dr SegComp::ToRhoTeta() const
{
    Pt2dr aPolar = Pt2dr::polar(normale(),0);
    double aRho = scal(p1(),normale());

    return Pt2dr(aRho,aPolar.y);

}

SegComp SegComp::FromRhoTeta(const Pt2dr &aP)
{
    double aC = cos(aP.y);
    double aS = sin(aP.y);
    double aRho = aP.x;

    Pt2dr aP0 (aC*aRho,aS*aRho);
    Pt2dr aP1  = aP0 + Pt2dr(-aS,aC);
    return SegComp(aP0,aP1);
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
