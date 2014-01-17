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



/*******************************************************************/
/*******************************************************************/
/************************ Ptr_OCR **********************************/
/*******************************************************************/
/*******************************************************************/


//  A class to allow some check-sum on memory.


#if (DEBUG_INTERNAL)

void Memory_Counter::show(Memory_Counter m2)
{
     cout          << _name     << ": "  << 
         _sum_call -m2._sum_call << " "   << 
         _sum_size -m2._sum_size << " "   << 
         _sum_ptr  -m2._sum_ptr  << "\n";
}

void Memory_Counter::verif(Memory_Counter m2)
{
     if (    (_sum_call != m2._sum_call)
          || (_sum_size != m2._sum_size)
          || (_sum_ptr  != m2._sum_ptr)
        )
     {
         cout << " FATAL ASSUMPUTION IN MEMORY CHEKSUM \n";
         show(m2);
         ElEXIT(1,"Memory_Counter::verif");
     }
}


Memory_Counter::Memory_Counter(const char *name) :
     _name      (name),
     _sum_call (0) ,
     _sum_size (0) ,
     _sum_ptr  (0) 
{};


Memory_Counter::Memory_Counter(void)
{
}

void * Memory_Counter::add_sub_oks(void * adr,INT sz,INT sign)
{
      _sum_call += sign;
      _sum_size += sign * sz;
#if ELISE_windows
	_sum_ptr  += sign * (int) (reinterpret_cast<long long int>(adr));
#else
	_sum_ptr  += sign * reinterpret_cast<long int>(adr);
#endif
      return adr;
}

#if (! CPP_OPTIMIZE)
void * Memory_Counter::add(void * adr)
{
    return add_sub_oks(adr,1,1);
}

void Memory_Counter::sub(void * adr)
{
    add_sub_oks(adr,1,-1);
}

#endif

Memory_Counter MC_OKS ("OKS");
Memory_Counter MC_CPTR("CPTR");
Memory_Counter MC_NTAB ("NEW TAB");
Memory_Counter MC_NEW_ONE("NEW_ONE");
Memory_Counter MC_TAB_USER("NEW TAB USER");

Memory_Counter  *(TAB_MEMO_COUNTER[NB_MEMO_COUNTER]) = 
                 {   &MC_OKS,
                     &MC_CPTR,
                     &MC_NTAB,
                     &MC_NEW_ONE,
                     &MC_TAB_USER
                 };


void show_memory_state(All_Memo_counter tab)
{
       cout << " MEMORY CHECK SUM \n";
       for(int i = 0 ; i < NB_MEMO_COUNTER ; i++)
          TAB_MEMO_COUNTER[i]->show(tab[i]);
}

void verif_memory_state(All_Memo_counter tab)
{
       for(int i = 0 ; i < NB_MEMO_COUNTER ; i++)
          TAB_MEMO_COUNTER[i]->verif(tab[i]);
}

void stow_memory_counter(All_Memo_counter & tab)
{
    for(int i = 0 ; i < NB_MEMO_COUNTER ; i++)
       tab[i] = *(TAB_MEMO_COUNTER[i]);
}

#else /*  DEBUG_INTERNAL */
void show_memory_state(void){}
void verif_memory_state(void)
{
      elise_internal_error("verif_memory_state with DEBUG_INTERNAL = 0 ",__FILE__,__LINE__);
}
#endif /*  DEBUG_INTERNAL */


         //-------------------------------------------------

void * Object_For_Ever::alloc(size_t sz)
{
     return Elise_Calloc(1,sz);
}
         //-------------------------------------------------


/*************************************************************************/
/*************************************************************************/
/************************  Arg_Opt   *************************************/
/*************************************************************************/
/*************************************************************************/


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
