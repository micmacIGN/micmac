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

#ifndef _ELISE_GRAPHE_CONNEC_COMP_H
#define _ELISE_GRAPHE_CONNEC_COMP_H


template <class  AttrSom,class AttrArc,class Res,class Germ>
         void  comp_connexe_ens_flag_alloc
         (
               Res   &                            res,
               const Germ  &                      germ,
               ElSubGraphe<AttrSom,AttrArc> &     sub,
               INT                                flag_marq
         )
{
    res.clear();
    for (INT k=0; k< germ.nb() ; k++)
    {
         ElSom<AttrSom,AttrArc> * s = germ[k];
         if ((sub.inS(*s)) && (! s->flag_kth(flag_marq)))
         {
            s->flag_set_kth_true(flag_marq);
            res.pushlast(s);
         }
    }

{    for (INT k= 0; k < res.nb(); k++)
    {
         ElSom<AttrSom,AttrArc> * s1 = res[k];
         for 
         (
             ElArcIterator<AttrSom,AttrArc> it= s1->begin(sub);
             it.go_on();
             it++
         )
         {
               ElSom<AttrSom,AttrArc> * s2 = &((*it).s2());
               if (! s2->flag_kth(flag_marq))
               {
                   s2->flag_set_kth_true(flag_marq);
                   res.pushlast(s2);
               }
         }
    }
}
    
}

template <class  AttrSom,class AttrArc,class Res>
         void  comp_connexe_som_flag_alloc
         (
               Res   &                            res,
               ElSom<AttrSom,AttrArc> *           g,
               ElSubGraphe<AttrSom,AttrArc> &     sub,
               INT                                flag_marq
         )
{
    ElFilo<ElSom<AttrSom,AttrArc> *> Germ;
    Germ.pushlast(g);
    comp_connexe_ens_flag_alloc(res,Germ,sub,flag_marq);
}


template <class  AttrSom,class AttrArc,class Res>
         void  comp_connexe_som
         (
               Res   &                            res,
               ElSom<AttrSom,AttrArc> *           g,
               ElSubGraphe<AttrSom,AttrArc> &     sub
         )
{
     INT flag_marq = g->gr().alloc_flag_som();
     comp_connexe_som_flag_alloc(res,g,sub,flag_marq);

     for (INT k=0; k<res.nb() ; k++)
         res[k]->flag_set_kth_false(flag_marq);
     g->gr().free_flag_som(flag_marq);
}


template <class  AttrSom,class AttrArc>  
        void PartitionCC
        (
             ElPartition<ElSom<AttrSom,AttrArc> * >&  aRes,
             ElGraphe<AttrSom,AttrArc>  &     aGr,
             ElSubGraphe<AttrSom,AttrArc> &     sub
        )
{
    aRes.clear();
    INT flag_p = aGr.alloc_flag_som();
    set_flag_all_soms(aGr,flag_p,false);
    for
    (
          ElSomIterator<AttrSom,AttrArc> sit = aGr.begin(sub) ;
          sit.go_on()                        ;
          sit++
    )
    {
          ElSom<AttrSom,AttrArc> & s =   * sit;
          if (! s.flag_kth(flag_p))
          {
              ElFifo<ElSom<AttrSom,AttrArc> *> aCC;
              comp_connexe_som(aCC,&s,sub);
              for (int aK=0 ; aK<int(aCC.size()) ; aK++)
              {
                  aRes.add(aCC[aK]);
                  aCC[aK]->flag_set_kth(flag_p,true);
              }
              aRes.close_cur();
          }
    }

    set_flag_all_soms(aGr,flag_p,false);


    aGr.free_flag_som(flag_p);
}


template <class  AttrSom,class AttrArc,class Soms,class Arcs>
         void  arcs_entre_soms
         (
               Arcs   &                           arcs,
               Soms   &                           soms,
               ElSubGraphe<AttrSom,AttrArc> &     sub
         )
{
     arcs.clear();
     if (soms.nb() == 0) return;

     INT flag_marq = soms[0]->gr().alloc_flag_som();
{     for (INT k=0; k<soms.nb() ; k++)
         soms[k]->flag_set_kth_true(flag_marq);
}

{     for (INT k=0; k<soms.nb() ; k++)
     {
         ElSom<AttrSom,AttrArc> * s1 = soms[k];
         for 
         (
             ElArcIterator<AttrSom,AttrArc> it= s1->begin(sub);
             it.go_on();
             it++
         )
         {
               ElSom<AttrSom,AttrArc> * s2 = &((*it).s2());
               if (s2->flag_kth(flag_marq))
                   arcs.pushlast(&(*it));
         }
     }
}

{     for (INT k=0; k<soms.nb() ; k++)
         soms[k]->flag_set_kth_false(flag_marq);
}
     soms[0]->gr().free_flag_som(flag_marq);
}




#endif // _ELISE_GRAPHE_CONNEC_COMP_H










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
