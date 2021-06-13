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




/**********************************************************************/
/**********************************************************************/
/**********************************************************************/
/**********************************************************************/
/**********************************************************************/


template <class Type> Type * new_vecteur_init(INT x0,INT x1,Type v)
{
    Type * res = NEW_VECTEUR(x0,x1,Type);
    set_cste(res+x0,v,x1-x0);

    return res;
}
template int * new_vecteur_init(int,int,int);
template double * new_vecteur_init(int,int,double);


#if (DEBUG_INTERNAL)

const unsigned int maj_deb = 0xF98A523F;
const unsigned int maj_end = 0xA57EF39D;

const unsigned char maj_octet = 0xE7;

const unsigned int rubbish  = 0xFEDCBAEF;

bool init_rub = true;

void * Elise_Calloc(size_t nmemb, size_t size)
{
     int nb_oct = (int) (nmemb * size);
     INT nb8 = (int) ((nmemb * size + 7)/8); // Assert 8 is the allignment constant
     INT nb4 = nb8 *2;
     INT unused_octet = nb4*4-nb_oct;

     void * v =  calloc(nb8+2,8);
     El_Internal.ElAssert
     (
           (v!= 0) && (sizeof(unsigned int)==4),
           EEM0 << "Impossible memory allocation " 
                << ((INT)(nmemb * size)) 
                << " byte"
     );

     unsigned int * res = ((unsigned int *) v) + 2;

    {
        if (init_rub)
             for (INT i =0; i<nb4; i++) res[i] = rubbish;
        else
             for (INT i =0; i<nb4; i++) res[i] = 0;
    }

     res [-2] = nb4;       // Taille de la zone en entier 4
     res[-1]  = maj_deb;   // majic nunmber
     res[nb4] = maj_end;   // majic number
     res[nb4+1] = unused_octet;
     {
          unsigned char * uc = (unsigned char *) res;
          int nb_oct4 = 4 * nb4;
          for (int i =  nb_oct4-unused_octet; i < nb_oct4; i++)
              uc[i] = maj_octet;
     }
     return res;
}
void  Elise_Free(void * ptr)
{
     unsigned int * ip = ((unsigned int *) ptr);
     INT nb4 = ip[-2];

     El_Internal.ElAssert
     (
           (ip[-1] == maj_deb) && (ip[nb4] == maj_end),
           EEM0 << "Corrupted majic number memory management"
     );
     {
          int unused_octet = ip[nb4+1];
          unsigned char * uc = (unsigned char *) ip;
          int nb_oct4 = 4 * nb4;
          for (int i =  nb_oct4-unused_octet; i < nb_oct4; i++)
               El_Internal.ElAssert
               (
                     (uc[i] == maj_octet),
                     EEM0 << "Corrupted majic number memory management (octet)"
               );
     }
     for (INT i =-2; i<nb4+2; i++) ip[i] = rubbish;
     free(ip-2);
}
#else
void * Elise_Calloc(size_t nmemb, size_t size)
{
     return calloc(nmemb,size);
}
void  Elise_Free(void * ptr)
{
     free(ptr);
}
#endif

void    delete_vecteur(void * v,const int x1,const int sz)
{

       char * mem  =  ((char *)v) + x1*sz;

#if (DEBUG_INTERNAL)
       SUB_MEM_COUNT(MC_NTAB,mem,1);
#endif

       if (mem)
          Elise_Free(mem);
}

void delete_matrice(void ** m, const Pt2di p1,const Pt2di p2,const int sz)
{
    Pt2di    pmax =Sup(p1,p2),pmin= Inf(p1,p2);

    for( INT4 y=pmin.y ; y<pmax.y ; y++ )
       delete_vecteur(m[y],pmin.x,sz);
    DELETE_VECTOR(m,pmin.y);
}

void delete_tab_matrice(void *** m,INT nb, const Pt2di p1,const Pt2di p2,const int sz)
{

    for (INT k=0 ; k<nb ; k++)
        delete_matrice(m[k],p1,p2,sz);

    DELETE_VECTOR(m,0);
}



void *  alloc_vecteur(const int x1,const int x2,const int sz)
{
    ASSERT_INTERNAL( (x1<=x2),"incoherence in alloc_vecteur");
    char * mem = (x1 == x2) ? 0 : (char *)Elise_Calloc(x2-x1,sz);
#if (DEBUG_INTERNAL)
    ADD_MEM_COUNT(MC_NTAB,mem,1);
#endif
    return  (void *) (mem - (x1*sz));
}

void **  alloc_matrice(const Pt2di p1,const Pt2di p2,const int sz)
{
    void     ** res;

    res = NEW_VECTEUR(p1.y,p2.y,void *);
    for (int y = p1.y ; y< p2.y; y++)
         res[y] = alloc_vecteur(p1.x,p2.x,sz);

    return res;
}


void ***  alloc_tab_matrice(const INT nb,const Pt2di p1,const Pt2di p2,const int sz)
{
    void *** res;

    res = NEW_VECTEUR(0,nb,void **);
    for (INT i=0 ; i<nb ; i++)
        res[i] = alloc_matrice(p1,p2,sz);

    return res;
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
