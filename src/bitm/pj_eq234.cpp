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

//
// Traduction d'un Programme de Patrick Julien
//


// Pourquoi CAS n'est pas mise a jour dans RacinesPolyneDegre2Reel
// Pourquoi  if (b<=0) ElSwap
// Pourquoi utiliser produit des racine  et pas somme
//  CAS non initialise dans EQ2


// Dans Degre 4 : R=AMINU4-B*A3+B2*AMINU2-3D0*B4  !!
// Dans Degre 4 : R=AMINU4-B*AMINU3+B2*AMINU2-3D0*B4  serait plus OK ?
// Dans Degre 3 : cas des racines triples, => if (ElAbs(r)< 1e-15)

// Dans degre 4, cas des equation quasi bicarre => if (ElAbs(Q)< 1e-10)
// Solution , avec un coup de Newton sur la racine de Deg 3





void RacinesPolyneDegre2Reel(REAL a0,REAL a1,REAL a2,INT & CAS,Pt2dr & X1,Pt2dr  & X2)
{
     ELISE_ASSERT(a0!=0, "Fisrt Term Null in EQ2DEGRE");
     REAL b = a1/a0;
     REAL c = a2/a0;
     REAL delta = b*b -4.0*c;


     if (delta > 0)
     {
	CAS = 1;
	REAL Sdelta = sqrt(delta);
	// Afin d'ajouter des quantites de meme signes
	if (b>=0)
	{
	    X1 = Pt2dr((-b-Sdelta)/2.0,0.0);
	    X2 = Pt2dr(c/X1.x,0.0);
	}
	else
	{
	    X2 = Pt2dr((-b+Sdelta)/2.0,0.0);
	    X1 = Pt2dr(c/X2.x,0.0);
	}

     }
     else if (delta==0)
     {
	CAS = 0;
        X1 = Pt2dr(-b/2.0,0);
        X2 = X1;
     }
     else
     {
	CAS = -1;
	REAL Sdelta = sqrt(-delta);
	X1 = Pt2dr(-b/2.0,-Sdelta/2.0);
	X2 = X1.conj();
     }
}






inline REAL DSign(REAL aVal,REAL aCmp)
{
   return (aCmp >0 ) ? aVal : -aVal;
}

void RacineCarreesComplexe (Pt2dr X,Pt2dr &A1,Pt2dr &A2)
{
    REAL R  = euclid(X);

    A1 = Pt2dr
	  (
	        sqrt(ElAbs(R+X.x)/2.0),
		DSign( sqrt(ElAbs(R-X.x)/2.0),X.y)
	  );
    A2 = -A1;
}







/*
 *  CAS =  1 , X1 reelle, X2,X3 complexe conj
 *  CAS =  0 , X1 reelle simple, X2,X3 reelle double
 *  CAS = -1 X1,X2,X3 reelles simple
*/

void RacinesPolyneDegre3Reel
     (
           REAL A0,REAL A1,REAL A2,REAL A3,
	   INT & CAS,
           Pt2dr & X1,Pt2dr  & X2,Pt2dr & X3
     )
{
     ELISE_ASSERT(A0!=0, "Fisrt Term Null in EQ3DEGRE");

     REAL aM1 = A1/A0;
     REAL aM2 = A2/A0;
     REAL aM3 = A3/A0;

     REAL b = aM1/3.0;
     REAL p = aM2-aM1*b;
     REAL q = aM3-b*aM2+2*b*b*b;
     REAL ppp = p*p*p;

     REAL r = q*q/4.0 + ppp/27.0;

     if (ElAbs(r)< 1e-15)
     {
         CAS = 0;
//	 REAL S1 = cbrt(-q/2.0);
	 REAL S1 = pow ( -q/2.0 , 1.0/3.0);
	 X1 = Pt2dr(2*S1,0);
	 X2 = Pt2dr(-S1,0);
	 X3 = Pt2dr(-S1,0);
     }
     else if (r>= 0)
     {
        CAS = 1;
	REAL S1,S2;
	if (q<=0)
	{
//		S1 = cbrt(-q/2.0+sqrt(r));
	   S1 = pow ( -q/2.0+sqrt(r) , 1.0/3.0);
	   S2 = -p/(3*S1);
	}
	else
	{
//           S2 = -cbrt(q/2.0+sqrt(r));
           S2 = - pow ( q/2.0+sqrt(r), 1.0/3.0);
           S1 = - p/(3*S2);
	}
	static const REAL Sqrt075 = sqrt(0.75);
	X1 = Pt2dr(S1+S2,0.0);
	X2 = Pt2dr(-X1.x/2.0,Sqrt075*(S1-S2));
	X3 = X2.conj();
     }
     else
     {
         CAS = -1;
	 REAL ro = sqrt(-ppp/27.0);
	 REAL om = acos(-q/(2*ro));
         REAL y = sqrt(-p/3)*cos(om/3.0);
	 REAL z = sqrt(-p)*sin(om/3.0);
	 X1 = Pt2dr(2*y,0);
	 X2 = Pt2dr(-y+z,0);
	 X3 = Pt2dr(-y-z,0);
     }
     X1.x -=b;
     X2.x -=b;
     X3.x -=b;

}
static bool BUG = false ;

void NewtonAmel( REAL A0,REAL A1,REAL A2,REAL A3,REAL A4,Pt2dr & aP1)
{
     for (INT aK=0 ; aK < 10 ; aK++)
     {
        Pt2dr aP2 = aP1 * aP1;
        Pt2dr aP3 = aP2 * aP1;
        Pt2dr aP4 = aP3 * aP1;

        Pt2dr F =  aP4 * A0 + aP3 *A1 + aP2 * A2 + aP1 * A3 + Pt2dr(A4,0);
        if (square_euclid(F) > 0.9e-10)
        {
           Pt2dr D =  aP3 * (A0*4) + aP2 *(A1*3) + aP1 * (A2 * 2) + Pt2dr(A3,0);
	   if (square_euclid(D) > 1e-15)
	   {
              aP1 -= F/D;
	   }
	   else
               return;
	}
	else
            return;
     }
}
	   


void RacinesPolyneDegre4Reel
     (
           REAL A0,REAL A1,REAL A2,REAL A3,REAL A4,
	   INT & CAS,
           Pt2dr & X1,Pt2dr  & X2,Pt2dr & X3,Pt2dr & X4
     )
{
static INT CPT=0;
CPT++;
// std::cout << "CPT = " << CPT << "\n";
// BUG = (CPT == 638);
     ELISE_ASSERT(A0!=0, "Fisrt Term Null in EQ4DEGRE");
     REAL aM1 = A1/A0;
     REAL aM2 = A2/A0;
     REAL aM3 = A3/A0;
     REAL aM4 = A4/A0;
     REAL B = aM1/4.0;
     REAL B2 = B*B;
     REAL B3 = B2 * B;
     REAL B4 = B3 * B;
     REAL P = aM2-6*B2;
     REAL Q = aM3-2*B*aM2+8*B3;
     REAL R = aM4-B*aM3+B2*aM2-3*B4;
     // REAL R = aM4-B*A3+B2*aM2-3*B4;
     if (ElAbs(Q)< 1e-10)
     {
         Pt2dr R1,R2;
         RacinesPolyneDegre2Reel(1.0,P,R,CAS,R1,R2);
         RacineCarreesComplexe(R1,X1,X2);
         RacineCarreesComplexe(R2,X3,X4);
	 X1.x -= B;
	 X2.x -= B;
	 X3.x -= B;
	 X4.x -= B;
     }
     else
     {
         aM1 = P/2;
	 aM2 = (P*P/4-R) /4;
	 aM3 = -Q*Q/64;
	 Pt2dr Y1,Y2,Y3;
         RacinesPolyneDegre3Reel(1,aM1,aM2,aM3,CAS,Y1,Y2,Y3);
	 REAL Y;
	 if (CAS==1)
	 {
            Y =  Y1.x ;

	 }
	 else
	 {
            if (BUG) std::cout << "Yi " << Y1 << Y2 << Y3 << "\n";
            Y =   ElMax(Y1.x,ElMax(Y2.x,Y3.x));
	 }

	 if ((ElAbs(Q)<1e-3) && ((CAS==1) || ((Y2.x<0) && (Y3.x))))
	 {
		    REAL Y2 = Y * Y;
		    REAL Y3 = Y2* Y;
		    REAL F0 = Y3 + aM1 * Y2 + aM2 * Y + aM3;
		    REAL D = Y2*2 + aM1 * Y + aM2 ;

		    Y -= F0/D;
	 }
	 REAL EPS = sqrt(Y);

	 REAL OmegA2 = -Y - aM1 + Q/(4*EPS);
	 REAL OmegB2 = -Y - aM1 - Q/(4*EPS);

	 Pt2dr OmegA(0.0,0.0), OmegB(0.0,0.0);


	 if (OmegA2 >=0) 
            OmegA.x = sqrt(OmegA2);
	 else
            OmegA.y = sqrt(-OmegA2);

	 if (OmegB2 >=0) 
            OmegB.x = sqrt(OmegB2);
	 else
            OmegB.y = sqrt(-OmegB2);

if (BUG)
{
	std::cout << Y << " aM1 " << aM1 << " Q " << Q << " EPS " << EPS << "\n";
	std::cout << Y1 << Y2 << Y3 << "\n";
	std::cout << OmegA << OmegB << "\n";
	std::cout << "Omeg2 : "  << OmegA2 << " " << OmegB2 << "\n";
}

	 X1.x = -EPS + OmegA.x -B;
	 X1.y =        OmegA.y   ;

	 X2.x = -EPS - OmegA.x -B;
	 X2.y =      - OmegA.y   ;


	 X3.x =  EPS + OmegB.x -B;
	 X3.y =        OmegB.y   ;

	 X4.x =  EPS - OmegB.x -B;
	 X4.y =      - OmegB.y   ;
     }

/*
     NewtonAmel(A0,A1,A2,A3,A4,X1);
     NewtonAmel(A0,A1,A2,A3,A4,X2);
     NewtonAmel(A0,A1,A2,A3,A4,X3);
     NewtonAmel(A0,A1,A2,A3,A4,X4);
     */
}






/*
 * Q!=0
 * 5.50671e-14 aM1 -2 Q -1.87788e-06 EPS 2.34664e-07
 * [5.50671e-14,0][0.999999,0.00222397][0.999999,-0.00222397]
 * [0,0.0246468][2.00015,0]
 * Omeg2 : -0.000607466 4.0006
 * F= [0.00241044,0]
 * FD = [0.00241044,0] [1.1404e-09,-0.197234]
 * FD = [0.00059782,1.74041e-12] [1.1404e-09,-0.197234]
 * F= [0.00241044,0]
 * FD = [0.00241044,0] [1.1404e-09,0.197234]
 * FD = [0.00059782,-1.74041e-12] [1.1404e-09,0.197234]
 * F= [0.00241044,0]
 * FD = [0.00241044,0] [16.0061,0]
 * FD = [4.53637e-07,0] [16.0061,0]
 * F= [0.00241044,0]
 * FD = [0.00241044,0] [-16.006,0]
 * FD = [4.53637e-07,0] [-16.006,0]
 * aD = 0.00059782
 *
 *
 *   eq4, CPT = 310
 *   aD = 0.000265083
 *
 *
 *
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
