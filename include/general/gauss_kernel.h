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



#ifndef _ELISE_GENERAL_GAUSSIAN_KERNEL  // general
#define _ELISE_GENERAL_GAUSSIAN_KERNEL



/*****************************************************************/
/*                                                               */
/*   Fonctions elementaires de noyaux 1D faits notamment pour    */
/* permettre le calcul de noyau enchaine                         */
/*                                                               */
/*****************************************************************/

//  Resout l'equation  aI o I2 = aI3 , au sens des moindres carres
Im1D_REAL8 DeConvol
(
     int aC2,   // Indexe 0 dans I2
     int aSz2,  // Taille I2
     Im1D_REAL8 aI1,   // Kernel 1
     int aC1,          // Indexe 0 dans I1
     Im1D_REAL8 aI3,   // Kernel 3
     int aC3           // Indexe 0 dans I3
);
// Paramametrage standardA  0 en centre image
Im1D_REAL8 DeConvol(int aDemISz2,Im1D_REAL8 aI1,Im1D_REAL8 aI3);



// Convolution C1 et C2 = indexe 0,  lent ; pour verif DeConvol
Im1D_REAL8 Convol(Im1D_REAL8 aI1,int aC1,Im1D_REAL8 aI2,int aC2);
// Parametrage stantdard 
Im1D_REAL8 Convol(Im1D_REAL8 aI1,Im1D_REAL8 aI2);

// Force l'image à une integrale donnee
Im1D_REAL8 MakeSom(Im1D_REAL8 aIm,double aSomCible);
Im1D_REAL8 MakeSom1(Im1D_REAL8 aIm);

//  Calcul un noyau gaussien en faisant pour chaque pixel la valeur integrale.
Im1D_REAL8  GaussianKernel(double aSigma,int aNb,int aSurEch);

//  Calcul le nombre d'element pour que la gaussiennne puisse etre tronquee a Residu pres
int NbElemForGausKern(double aSigma,double aResidu);

//  Calcule un noyau gaussien
Im1D_REAL8  GaussianKernelFromResidu(double aSigma,double aResidu,int aSurEch);


// Conversion d'un noyau double (de somme 1) en entier, en conservant la somme
// (Pour une image entiere qui sera interpretee comme rationnele de quotient aMul)
Im1D_INT4 ToIntegerKernel(Im1D_REAL8 aRK,int aMul,bool aForceSym);




#endif  /* !_ELISE_GENERAL_GAUSSIAN_KERNEL */







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
