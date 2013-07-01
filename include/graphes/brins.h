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

#ifndef _ELISE_GRAPHE_BRINS
#define _ELISE_GRAPHE_BRINS

/*
    Pour brins cycliques :

        - faire appel a all_brins_not_cycliques
        - marquer les sommets obtenus
        - attaquer les autres

     Est considere comme cyclique le  

      ___  /
     /   \/
     |    |
     \____/
*/

#include "graphes/uti_gr.h"

template <class  AttrSom,class AttrArc> 
         ElSom<AttrSom,AttrArc> * succ_diff
         (
               ElSom<AttrSom,AttrArc> * s,
               ElSom<AttrSom,AttrArc> * to_avoid,
               ElSubGraphe<AttrSom,AttrArc> &       sub
         )
{
      for
      (
         ElArcIterator<AttrSom,AttrArc> ait = s->begin(sub) ;
         ait.go_on()                                        ;
         ait++
      )
          if ( &((*ait).s2()) != to_avoid)
             return  &((*ait).s2());

      ELISE_ASSERT(false,"succ_diff ");
      return 0;
}


template <class  AttrSom,class AttrArc>
         void  brin_1_sens_and_not_clear
         (
               ElFifo<ElSom<AttrSom,AttrArc> *> &  res,
               ElSom<AttrSom,AttrArc> *            s1,
               ElSom<AttrSom,AttrArc> *            s2,
               ElSubGraphe<AttrSom,AttrArc> &      sub_small,
               ElSubGraphe<AttrSom,AttrArc> &      sub_big,
               INT                                 flag_marq,
               bool                                push_last
         )            
{
     while (!  s2->flag_kth(flag_marq))
     {
            s2-> flag_set_kth_true(flag_marq);
            if ( push_last)
               res.pushlast(s2);
            else
               res.pushfirst(s2);


            if  (
                      (s2->nb_succ(sub_small) != 2) 
                   || (s2->nb_succ(sub_big) != 2) 
                )
                return;

            ElSom<AttrSom,AttrArc> * sauv2 = s2;
            s2 = succ_diff(s2,s1,sub_small);
            s1 = sauv2;
     }
}

template <class  AttrSom,class AttrArc>
         void  brin_1_sens_and_not_clear
         (
               ElFifo<ElSom<AttrSom,AttrArc> *> &  res,
               ElSom<AttrSom,AttrArc> *            s1,
               ElSom<AttrSom,AttrArc> *            s2,
               ElSubGraphe<AttrSom,AttrArc> &      sub,
               INT                                 flag_marq,
               bool                                push_last
         )            
{
   brin_1_sens_and_not_clear(res,s1,s2,sub,sub,flag_marq,push_last);
}




template <class  AttrSom,class AttrArc>
         void  brin_2_sens_and_marq
         (
               ElFifo<ElSom<AttrSom,AttrArc> *> &  res,
               ElArc<AttrSom,AttrArc> &            arc,
               ElSubGraphe<AttrSom,AttrArc> &      sub1,
               ElSubGraphe<AttrSom,AttrArc> &      sub2,
               INT                                 flag_marq
         )
{                                        
    res.clear();
    brin_1_sens_and_not_clear(res,&(arc.s1()),&(arc.s2()),sub1,sub2,flag_marq,true);
    brin_1_sens_and_not_clear(res,&(arc.s2()),&(arc.s1()),sub1,sub2,flag_marq,false);

    if (
              (res.nb() > 2)
          &&  (
                      (    (res[0]->nb_succ(sub1)==2)
                        && (res[0]->nb_succ(sub2)==2)
                      )
                 ||   (     (res.top()->nb_succ(sub1)==2)
                        &&  (res.top()->nb_succ(sub2)==2)
                      )
              )
       )
    {
        ELISE_ASSERT
        (
           arc.s1().gr().arc_s1s2(*res[0],*res.top()),
           "Incohernce (Graphe Not SYM ?) in brin_2_sens_and_marq "
        );
        res.set_circ(true);
    }
    else
        res.set_circ(false);
}

template <class  AttrSom,class AttrArc>
         void  brin_2_sens_and_marq
         (
               ElFifo<ElSom<AttrSom,AttrArc> *> &  res,
               ElArc<AttrSom,AttrArc> &            arc,
               ElSubGraphe<AttrSom,AttrArc> &      sub,
               INT                                 flag_marq
         )
{
    brin_2_sens_and_marq(res,arc,sub,sub,flag_marq);
}

template <class  AttrSom,class AttrArc> 
         void  brin_2_sens
         (
               ElFifo<ElSom<AttrSom,AttrArc> *> &  res,
               ElArc<AttrSom,AttrArc> &            arc,
               ElSubGraphe<AttrSom,AttrArc> &      sub1,
               ElSubGraphe<AttrSom,AttrArc> &      sub2
         )
{                                        
   INT  flag_marq = arc.gr().alloc_flag_som();

   brin_2_sens_and_marq(res,arc,sub1,sub2,flag_marq);

   for (INT k=0; k< res.nb(); k++)
       res[k]->flag_set_kth_false(flag_marq);
   arc.gr().free_flag_som(flag_marq);
}


template <class  AttrSom,class AttrArc> 
         void  brin_2_sens
         (
               ElFifo<ElSom<AttrSom,AttrArc> *> &  res,
               ElArc<AttrSom,AttrArc> &            arc,
               ElSubGraphe<AttrSom,AttrArc> &      sub
         )
{                                        
     brin_2_sens(res,arc,sub,sub);
}


 

template <class  AttrSom,class AttrArc>
         void   all_brins_not_cycliques
         (
               ElGraphe<AttrSom,AttrArc> &               gr,
               ElSubGraphe<AttrSom,AttrArc> &            sub_small,
               ElSubGraphe<AttrSom,AttrArc> &            sub_big,
               ElPartition<ElSom<AttrSom,AttrArc> *>  &  part
         )
{
      part.clear();
      for
      (
            ElSomIterator<AttrSom,AttrArc> sit = gr.begin(sub_small) ;
            sit.go_on()                        ;
            sit++
      )
      {
           ElSom<AttrSom,AttrArc> & s =   * sit;
           INT nbS = s.nb_succ(sub_small);
           INT nbB = s.nb_succ(sub_big);

           if (nbS == 0)
           {
               part.add(&s);
               part.close_cur();
           }
           else if ((nbS==2) && (nbB == 2))
           {
               // rien c'est un sommet au milieu des brins
           }
           else
           {
                for
                (
                   ElArcIterator<AttrSom,AttrArc> ait = s.begin(sub_small) ;
                   ait.go_on()                                        ;
                   ait++
                )
                {
                    part.add(&s);
                    ElSom<AttrSom,AttrArc> * ps1 = &s;
                    ElSom<AttrSom,AttrArc> * ps0 = ps1;
                    ElSom<AttrSom,AttrArc> * ps2 = &(*ait).s2();
                    while (
                                 (ps2->nb_succ(sub_small) == 2)
                            &&   (ps2->nb_succ(sub_big) == 2)
                          )
                    {
                              part.add(ps2);
                              ElSom<AttrSom,AttrArc> * sauv2 = ps2;
                              ps2 = succ_diff(ps2,ps1,sub_small);
                              ps1 = sauv2;
                    }
                    part.add(ps2);
                    if (ps0->num() > ps2->num() )
                       part.close_cur();
                    else
                       part.remove_cur();
                }
           }
/*
           switch (s.nb_succ(sub))
           {
                 case 2 : break;

                 case 0 : 
                      part.add(&s);
                      part.close_cur();
                 break;

                 default :
                     for
                     (
                        ElArcIterator<AttrSom,AttrArc> ait = s.begin(sub) ;
                        ait.go_on()                                        ;
                        ait++
                     )
                     {
                         part.add(&s);
                         ElSom<AttrSom,AttrArc> * ps1 = &s;
                         ElSom<AttrSom,AttrArc> * ps0 = ps1;
                         ElSom<AttrSom,AttrArc> * ps2 = &(*ait).s2();
                         while (ps2->nb_succ(sub) == 2)
                         {
                              part.add(ps2);
                              ElSom<AttrSom,AttrArc> * sauv2 = ps2;
                              ps2 = succ_diff(ps2,ps1,sub);
                              ps1 = sauv2;
                         }
                         part.add(ps2);
                         if (ps0->num() > ps2->num() )
                            part.close_cur();
                         else
                            part.remove_cur();
                     }
           }
*/
       }
}


template <class  AttrSom,class AttrArc>
         void   all_brins_not_cycliques
         (
               ElGraphe<AttrSom,AttrArc> &               gr,
               ElSubGraphe<AttrSom,AttrArc> &            sub,
               ElPartition<ElSom<AttrSom,AttrArc> *>  &  part
         )
{
     all_brins_not_cycliques(gr,sub,sub,part);
}

template <class  AttrSom,class AttrArc>
         void   brins_1_sens
         (
               ElArc<AttrSom,AttrArc> *                  arc,
               ElSubGraphe<AttrSom,AttrArc> &            sub_small,
               ElSubGraphe<AttrSom,AttrArc> &            sub_big,
               INT                                       fl_marq,
               bool                                      plast,
               ElFifo<ElSom<AttrSom,AttrArc> *>  &       res
         )
{
      while (true)
      {
           if (
                    (arc->s2().nb_succ(sub_big)   != 2) 
                ||  (arc->s2().nb_succ(sub_small) != 2) 
              )
              return;

           ElArc<AttrSom,AttrArc> *  next_a = 0;

           for
           (
                  ElArcIterator<AttrSom,AttrArc> ait = arc->s2().begin(sub_small) ;
                  ait.go_on()                                             ;
                  ait++
           )
           {
                 ElArc<AttrSom,AttrArc> & new_a = *ait;
                 if (! new_a.flag_kth(fl_marq))
                 {
                     ELISE_ASSERT(next_a==0,"brins_1_sens [0]");
                     next_a = & new_a;
                 }
           }
           if (next_a == 0) 
              return;

           next_a->sym_flag_set_kth_true(fl_marq);
           res.push(&(next_a->s2()),plast);
           arc = next_a;
      }
}




template <class  AttrSom,class AttrArc>
         void   all_brins
         (
               ElGraphe<AttrSom,AttrArc> &               gr,
               ElSubGraphe<AttrSom,AttrArc> &            sub_small,
               ElSubGraphe<AttrSom,AttrArc> &            sub_big,
               ElPartition<ElSom<AttrSom,AttrArc> *>  &  part
         )
{

     ElFifo<ElSom<AttrSom,AttrArc> *>  buf;
      
      part.clear();
      INT fl_marq = gr.alloc_flag_arc();
      set_flag_all_arcs(gr,fl_marq,false);

      for
      (
            ElSomIterator<AttrSom,AttrArc> sit = gr.begin(sub_small) ;
            sit.go_on()                        ;
            sit++
      )
      {
           ElSom<AttrSom,AttrArc> & s =   * sit;
           for
           (
                  ElArcIterator<AttrSom,AttrArc> ait = s.begin(sub_small) ;
                  ait.go_on()                                             ;
                  ait++
           )
           {
                 ElArc<AttrSom,AttrArc> * arc = &(*ait);
                 if (! (arc->flag_kth(fl_marq)))
                 {
                      buf.clear();
                      arc->sym_flag_set_kth_true(fl_marq);
                      buf.pushfirst(&(arc->s1()));
                      buf.pushlast (&(arc->s2()));

                      brins_1_sens(arc,sub_small,sub_big,fl_marq,true,buf);
                      arc = &(arc->arc_rec());
                      brins_1_sens(arc,sub_small,sub_big,fl_marq,false,buf);

                      append(part.filo(),buf);
                      part.close_cur();
                 }
           }
     }

      set_flag_all_arcs(gr,fl_marq,false);
      gr.free_flag_arc(fl_marq);
}

template <class  AttrSom,class AttrArc> 
         void Ebarbule
         (
               ElGraphe<AttrSom,AttrArc> &               gr,
               ElSubGraphe<AttrSom,AttrArc> &            subgr,
               ElSubGrapheSom<AttrSom,AttrArc> &         Mendatory,
               std::vector<ElSom<AttrSom,AttrArc> *>  &  aRes
         )
{
     aRes.clear();
     INT fl_marq = gr.alloc_flag_som();
     set_flag_all_soms(gr,subgr,fl_marq,true);
     cSubGrFlagSom<ElSubGraphe<AttrSom,AttrArc> > aGrMarq(subgr,fl_marq);

     for
     (
        ElSomIterator<AttrSom,AttrArc> sit = gr.begin(aGrMarq) ;
        sit.go_on()                                          ;
        sit++
     )
     {
         ElSom<AttrSom,AttrArc> * aS = &(*sit);
         if ((!Mendatory.inS(*aS)) && (aS->nb_succ(aGrMarq)<=1))
         {
             aRes.push_back(aS);
         }
     }
     int aK= 0;
     while (aK!=int(aRes.size()))
     {
         ElSom<AttrSom,AttrArc> * aS1 = aRes[aK];
         aS1->flag_set_kth_false(fl_marq);
         for
         (
                  ElArcIterator<AttrSom,AttrArc> ait = aS1->begin(aGrMarq) ;
                  ait.go_on()                                        ;
                  ait++
         )
         {
               ElSom<AttrSom,AttrArc> * aS2 = &((*ait).s2());
               if ((!Mendatory.inS(*aS2)) && (aS2->nb_succ(aGrMarq)==1))
                   aRes.push_back(aS2);
         }


         aK++;
     }


     set_flag_all_soms(gr,subgr,fl_marq,false);
     gr.free_flag_som(fl_marq);
}
/*
*/


#endif // _ELISE_GRAPHE_BRINS










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
