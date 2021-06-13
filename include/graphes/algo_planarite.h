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

#ifndef _ELISE_GRAPHE_ALGO_PLANARITE
#define _ELISE_GRAPHE_ALGO_PLANARITE
#include "graphes/graphe.h"

         // FACES

template <class  AttrSom,class AttrArc>
         ElArc<AttrSom,AttrArc> *  succ_trigo
                                   (
                                       ElArc<AttrSom,AttrArc>       & a,
                                       ElSubGraphe<AttrSom,AttrArc> & sub,
                                       bool                         trigo = true
                                   )
{

      REAL teta = trigo ? 10 : -10;
      ElSom<AttrSom,AttrArc> * as1 = &(a.s1());
      ElSom<AttrSom,AttrArc> * succ = as1;
      ElSom<AttrSom,AttrArc> & s2 = a.s2();
      Pt2dr  p1 = sub.pt(a.s1());
      Pt2dr  p2 = sub.pt(s2);
      Pt2dr v12 = p2-p1;

      for
      (
           ElArcIterator<AttrSom,AttrArc> it = s2.begin(sub);
           it.go_on();
           it++
      )
      {
           ElSom<AttrSom,AttrArc> & s3 = (*it).s2();
           if (&s3 != as1)
           {
               REAL new_teta = angle(v12,sub.pt(s3)-p2);
               if (trigo ? (new_teta < teta) : (new_teta > teta))
               {
                   teta = new_teta;
                   succ = &s3;
               }
           }
      }
      ELISE_ASSERT(succ,"No SUCC in succ_trigo");
      return  s2.gr().arc_s1s2(s2,*succ);
}

template <class  AttrSom,class AttrArc>
         ElArc<AttrSom,AttrArc> *  succ_clock
                                   (
                                       ElArc<AttrSom,AttrArc>       & a,
                                       ElSubGraphe<AttrSom,AttrArc> & sub
                                   )
{
    return succ_trigo(a,sub,false);
}



template <class  AttrSom,class AttrArc>
         bool  face_trigo
         (
               ElArc<AttrSom,AttrArc>       &       arc_init,
               ElSubGraphe<AttrSom,AttrArc> &       sub,
               ElFilo<ElArc<AttrSom,AttrArc> *>  &  res
         )
{

      ElArc<AttrSom,AttrArc> * a0 =  &arc_init;
      ElArc<AttrSom,AttrArc> * a  = a0;

      INT cpt = 2*arc_init.s1().gr().nb_som();
      do
      {
            a = succ_trigo(*a,sub);
            res.pushlast(a);
            cpt --;
            if (cpt < 0) return false;
      }
      while  (a != a0);

      return true;
}


template <class  AttrSom,class AttrArc>
         bool  all_face_trigo
         (
               ElGraphe<AttrSom,AttrArc> &          gr,
               ElSubGraphe<AttrSom,AttrArc> &       sub,
               ElPartition<ElArc<AttrSom,AttrArc> *>  &  part
         )
{
      INT flag = gr.alloc_flag_arc();
      bool OK = true;

      part.clear();
      for
      (
            ElSomIterator<AttrSom,AttrArc> sit = gr.begin(sub) ;
            sit.go_on()                        ;
            sit++
      )
      {
           ElSom<AttrSom,AttrArc> & s =   * sit;
           for
           (
                  ElArcIterator<AttrSom,AttrArc> ait = s.begin(sub) ;
                  ait.go_on()                                        ;
                  ait++
           )
           {
                 ElArc<AttrSom,AttrArc> & a =   * ait;
                 if (! a.flag_kth(flag))
                 {
                     if (! face_trigo(a,sub,part.filo())) OK = false;
                     part.close_cur();
                     ElSubFilo<ElArc<AttrSom,AttrArc> *> NewF = part.top();
                     for (INT k=0; k<NewF.nb() ; k++)
                         NewF[k]->flag_set_kth_true(flag);
                 }
           }
      }
      ElFilo<ElArc<AttrSom,AttrArc> *> & all = part.filo();
      for (INT k=0; k<all.nb() ; k++)
           all[k]->flag_set_kth_false(flag);
      gr.free_flag_arc(flag);
      return OK;
}

template <class  AttrSom,class AttrArc>
         void  make_real_face
         (
               ElGraphe<AttrSom,AttrArc> &                     gr,
               ElPartition<ElArc<AttrSom,AttrArc> *>  &        out,
               ElPartition<ElArc<AttrSom,AttrArc> *>  &        in
         )
{
     INT flag = gr.alloc_flag_arc();

     out.clear();
	 INT ka;

     for (INT kf =0 ; kf < in.nb() ; kf++)
     {
          ElSubFilo<ElArc<AttrSom,AttrArc> *> face = in[kf];
          for ( ka =0 ; ka<face.nb() ; ka++)
              face[ka]->sym_flag_set_kth_false(flag);
     }


{     for (INT kf =0 ; kf < in.nb() ; kf++)
     {
          ElSubFilo<ElArc<AttrSom,AttrArc> *> face = in[kf];
          for ( ka =0 ; ka<face.nb() ; ka++)
              face[ka]->flag_set_kth_true(flag);

          for ( ka =0 ; ka<face.nb() ; ka++)
              if (! face[ka]->arc_rec().flag_kth(flag))
                  out.add(face[ka]);
          out.close_cur();

          for ( ka =0 ; ka<face.nb() ; ka++)
              face[ka]->flag_set_kth_false(flag);
     }
}

     gr.free_flag_arc(flag);
}
  


#endif // _ELISE_GRAPHE_ALGO_PLANARITE










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
