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



/*.FE{C 0.2}{Dynamically Allocated Vectors and Matrices}
            {Dynamically Allocated Vectors and Matrices}*/

/* --------------------- DECLARATIONS vmblock.h --------------------- */

#ifndef VMBLOCK_H_INCLUDED
#define VMBLOCK_H_INCLUDED

/***********************************************************************
* symbolic names that let the user choose the kind of dynamical data   *
* structure to be allocated upon calling vmalloc()                     *
***********************************************************************/

#define VEKTOR   0                /* for a REAL vector                */
/*.IX{VEKTOR}*/
#define VVEKTOR  1                /* for a vector with elements       */
/*.IX{VVEKTOR}*/
                                  /* of given size                    */
#define MATRIX   2                /* for a REAL matrix                */
/*.IX{MATRIX}*/
#define IMATRIX  3                /* for an int matrix                */
/*.IX{IMATRIX}*/
#define MMATRIX  4                /* for a matrix of 4x4 matrices     */
/*.IX{PMATRIX}*/
                                  /* (with elements of type `mat4x4') */
#define UMATRIX  5                /* for a lower triangular matrix    */
/*.IX{UMATRIX}*/
#define PMATRIX  6                /* for a matrix of points in R3     */
/*.IX{PMATRIX}*/


/***********************************************************************
* operations on a vector matrix list                                   *
***********************************************************************/

void *vminit         /* create an empty vector/matrix list ...........*/
        (
         void
        );                      /* address of list ...................*/


void *vmalloc        /* create a dynamic vector or matrix ............*/
        (
         void   *vmblock,       /* address of a vector/matrix list ...*/
         int    typ,            /* kind of vector or matrix ..........*/
         size_t zeilen,         /* length (vector) or number of rows .*/
         size_t spalten         /* number of columns or element size .*/
        );                      /* address of the created object .....*/


boolean vmcomplete   /* check vector/matrix list for lack of memory ..*/
        (
         void *vmblock          /* address of list ...................*/
        );                      /* no lack of memory? ................*/


void vmfree          /* free the memory for a vektor/matrix list .....*/
        (
         void *vmblock          /* address of list ...................*/
        );

#endif

/* ------------------------- END vmblock.h -------------------------- */

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
