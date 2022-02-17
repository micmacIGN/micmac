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





typedef struct tagGg {
  double X0, YR , C , R2 , SS ,LC0 ;
} Gg ; 	/* Lamb_geo */

typedef struct tagPof  {     
  double X0, Sino, Sink, LC0, R0, YR ;
} Pof ;	/* Geo_Lamb */

/********************************************************/
/*		Lamb_geo.c 
   Calcule la longitude Alon et la latitude Alat en grades
   d'un point de coordonnees X,Y donnees en metres 
   ( avec le chiffre indicateur de la zone Lambert pour les y)
   dans la Projection Lambert de numero Nlamb.
 
 On rappelle que la constante origine des Y vaut:
      1200000.00 Metres pour la zone Lambert I
      2200000.00         "           Lambert II
      3200000.00         "           Lambert III
      4185861.37         "           Lambert IV

 Les 4 zones Lambert-France sont caracterisees par:
 Lambert I  :      Alat>=53.5 Gr            (Parallele origine:55 Gr)
 Lambert II :    50.5 Gr<=Alat<=53.5 Gr     (Parallele origine:52 Gr)
 Lambert III: Alat<=50.5 Gr et Alon<=6.2 Gr (Parallele origine:49 Gr)
 Lambert IV : Alat<=48 Gr et Alon>=6.4 Gr (Parallele origine:46,85 Gr) */
/*******************************************************/

void Lamb_geo (double xx, 
	       double yy, 
	       int N_lamb, 
	       double *Alon, 
	       double *Alat )

{
  double U, V, P, T, S, S2, A, E;
  int Nl ;
  Gg Eve[4] ;
  
  /*   Constantes des projections Lambert-France  */
  Eve[0].X0= 600000; 
  Eve[1].X0= 600000;  
  Eve[2].X0= 600000; 
  Eve[3].X0= 234.358;
  Eve[0].YR= 6657616.674 ; 
  Eve[1].YR= 8199695.768 ;  
  Eve[2].YR= 9791905.085 ; 
  Eve[3].YR= 11239161.542 ; 
  Eve[0].C= 83.721038652;
  Eve[1].C= 87.331573464;
  Eve[2].C= 91.479819811;
  Eve[3].C= 94.838400858;
  Eve[0].R2 = 2.978557976E13;
  Eve[1].R2 = 3.5996349309E13;
  Eve[2].R2 = 4.345321265E13;
  Eve[3].R2 = 4.974904333E13;
  Eve[0].SS = 1.5208119312; 
  Eve[1].SS = 1.4579372548; 
  Eve[2].SS = 1.3918255932; 
  Eve[3].SS = 1.3425358644; 
  Eve[0].LC0= 0.9919966654;
  Eve[1].LC0= 0.9215573613;
  Eve[2].LC0= 0.8545910977;
  Eve[3].LC0= 0.8084757728;
  
  if ( yy < (double)(N_lamb * 1000000) ) 
  {
    yy = yy + (double)(N_lamb * 1000000) ;
  }
  
  Nl = N_lamb - 1 ;
  U = xx - (double) ( Eve[Nl].X0  ) ;
  V = (double) ( Eve[Nl].YR ) - yy ;
  *Alon = ( atan( U / V ) ) * Eve[Nl].C ; 
  P =  ( U * U + V * V ) / Eve[Nl].R2 ;
  E = exp( log(P)/Eve[Nl].SS - Eve[Nl].LC0  ) ;
  T = ( 1 - E ) / ( 1 + E ) ; 
  S = sin( 2 * atan(T) ) ;
  S2 = S * S ; 
  A = ( ( 1.38E-7 * S2 - 1.5707E-5 ) * S2 + 3.425046E-3 ) * S ; 
  *Alat = atan( (A + T) / (1 + A * T ) ) * 127.32395447 ; 
  
  /* Le calcul est fait en grades : en degres en sortie */
  *Alon = *Alon * 0.9 ;
  *Alat = *Alat * 0.9 ;
}




/******************************************************************/
/*            Geo_lamb.c

   Calcule les coordonnees X ,Y en metres,dans la projection
   Lambert de numero N_amb d'un point de longitude Alon et de
   latitude Alat donnees en grades.
   On rappelle que la constante origine des Y vaut:
      1200000.00 Metres pour la zone Lambert I
      2200000.00         "           Lambert II
      3200000.00         "           Lambert III
      4185861.37         "           Lambert IV
  Les 4 zones Lambert-France sont caracterisees par:
  Lambert I  :      Alat>=53.5 Gr            (Parallele origine:55 Gr)
  Lambert II :    50.5 Gr<=Alat<=53.5 Gr     (Parallele origine:52 Gr)
  Lambert III: Alat<=50.5 Gr et Alon<=6.2 Gr (Parallele origine:49 Gr)
  Lambert IV : Alat<=48 Gr et Alon>=6.4 Gr (Parallele origine:46,85 Gr) */

/******************************************************************/


void Geo_lamb( double Alon, double Alat, int N_lamb, double *x, double *y )

{
double T, Lc , R , C ;
double gAlon , gAlat ;	/* en grades */
Pof Pif[4] ;  
int Nl ;
/*   Constantes des projections Lambert-France    */
     Pif[0].X0=600000;
     Pif[1].X0=600000;
     Pif[2].X0=600000;
     Pif[3].X0=234.358;
     Pif[0].Sino=0.7604059656;
     Pif[1].Sino=0.7289686274;
     Pif[2].Sino=0.6959127966;
     Pif[3].Sino=0.6712679322;
     Pif[0].Sink=1.194442898E-2;
     Pif[1].Sink=1.145061242E-2;
     Pif[2].Sink=1.093137265E-2;
     Pif[3].Sink=1.054425202E-2;
     Pif[0].LC0=0.9919966654;
     Pif[1].LC0=0.9215573613;
     Pif[2].LC0=0.8545910977;
     Pif[3].LC0=0.8084757728;
     Pif[0].R0=5457616.674;
     Pif[1].R0=5999695.768;
     Pif[2].R0=6591905.085;
     Pif[3].R0=7053300.173;
     Pif[0].YR=6657616.674;
     Pif[1].YR=8199695.768; 
     Pif[2].YR=9791905.085;
     Pif[3].YR=11239161.542;


   Nl = N_lamb -1 ;
   gAlon = Alon * 200.0 / 180.0 ;
   gAlat = Alat * 200.0 / 180.0 ;

   T = 8.24832568E-2 * sin( gAlat * 1.5707963268E-2 ) ;
 
   Lc = log( tan( 7.8539816340E-3 * ( gAlat + 100) ) )
          - 4.12416284E-2 * log( (1+T) / (1-T) ) ;
   /* latitude croissante */
   R=Pif[Nl].R0 * exp(Pif[Nl].Sino*(Pif[Nl].LC0-Lc));   
   C=Pif[Nl].Sink * gAlon ; 

   *x = Pif[Nl].X0 + R * sin(C);
   *y = Pif[Nl].YR - R * cos(C);
/*
 *		Formulation abandonnee :
 *	  *y = *y - (double)(N_lamb * 1000000) ;
 *	on adopte desormais une convention pour laquelle
 *	Y porte les "millions" du Lambert
 */
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
