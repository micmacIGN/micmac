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

#ifndef _ELISE_GRAPHES_GRAPHE_IMPLEM_H 
#define  _ELISE_GRAPHES_GRAPHE_IMPLEM_H 

#include "graphes/graphe.h"


/**************************************************************/
/*                                                            */
/*                   ElTabDyn                                 */
/*                                                            */
/**************************************************************/


template<class T> ElTabDyn<T>::~ElTabDyn()
{
    if (_tvals)
    {
         for (INT k0 =0 ; k0<NB ; k0++) 
         {
             if (_tvals[k0])
             {
                 for (INT k1 =0 ; k1<NB ; k1++) 
                 {
                      if (_tvals[k0][k1])
                      {
                         STD_DELETE_TAB(_tvals[k0][k1]);
                      }
                         
                 }
                 STD_DELETE_TAB(_tvals[k0]);
             }
         }
         STD_DELETE_TAB(_tvals);
    }
}

template<class T> void ElTabDyn<T>::augment_index()
{
    if (! _tvals) 
    {
       _tvals     =  STD_NEW_TAB((INT)NB,T**); 
        MEM_RAZ(_tvals,NB);
     }
    if (! t1(_nb) ) 
    {
         t1(_nb)  =  STD_NEW_TAB((INT)NB,T*) ;
        MEM_RAZ(t1(_nb),NB);
    }
    if (! t2(_nb) ) 
    {
        t2(_nb)  =  STD_NEW_TAB((INT)NB,T);
    }
    
    _nb++;
}

template<class T> void ElTabDyn<T>::set_intexable_until(INT aNb)
{
     while (nb()<aNb) augment_index();
}

/**************************************************************/
/*                                                            */
/*                   ElSom                                    */
/*                                                            */
/**************************************************************/

template <class AttrSom,class AttrArc>
         ElSom<AttrSom,AttrArc>::ElSom
         (
               TGraphe * gr,
               const AttrSom & attr,
               INT             Num
         ) :
                 _alive   (true),
                 _gr      (gr),
                 _succ    (0),
                 _attr    (attr),
                 _num     (Num),
                 _flag    ()
{
}


template <class AttrSom,class AttrArc>
         ElSom<AttrSom,AttrArc>::~ElSom()
{
     TGraphe::kill_arc(_succ);
}

template <class AttrSom,class AttrArc> 
          ElArc<AttrSom,AttrArc> * ElSom<AttrSom,AttrArc>::_remove_succ
                                   (ElSom<AttrSom,AttrArc> * s2)
{

     for ( TArc * prec=0,*a =_succ  ;  a  ;  prec=a,a=a->_next)
         if (&(a->s2()) == s2)
         {
             if (prec)
                 prec->_next = a->_next;
             else
                 _succ = a->_next;
             a->s1()._gr->add_free(a);
             return a;
         }
    // ELISE_ASSERT(false,"ElSom::_remove_succ");

     return 0;
}



template <class AttrSom,class AttrArc> void ElSom<AttrSom,AttrArc>::remove()
{
   _gr->OnKillSom(*this);
   _alive = false;
   _gr->_free_number.pushlast(_num);
   _gr->_nb_som --;
   for (TArc * la = _succ; la; )
   {
         if (la->_s2->_remove_succ(la->_s1) ==0)
            ELISE_ASSERT(false,"ElSom::remove");
         TArc * next = la->_next;
         _gr->add_free(la);
         la = next;
    }
    _succ = 0;
}

template <class AttrSom,class AttrArc> 
         INT ElSom<AttrSom,AttrArc>::nb_succ(ElSubGraphe<AttrSom,AttrArc> & sub)
{
     if (! sub.inS(*this)) return 0;

     INT res=0;
     ElArc<AttrSom,AttrArc> * a = _succ;
     for (; a ; a = a->_next)
         res += sub.inA(*a) && sub.inS(a->s2());
     return res;
}

template <class AttrSom,class AttrArc>  
         ElSom<AttrSom,AttrArc> &  ElSom<AttrSom,AttrArc>::uniq_succ(ElSubGraphe<AttrSom,AttrArc> & sub)
{
     TArc * res = 0;
     for ( TArc * a =_succ  ;  a   ; a=a->_next)
	 {
		 if (sub.inA(*a))
		 {
			ELISE_ASSERT(res==0,"Multiple succ in uniq_succ");
			res = a;
		 }
	 }
	 ELISE_ASSERT(res!=0,"No succ in uniq_succ");
	 return res->s2();
}

template <class AttrSom,class AttrArc>  
         ElSom<AttrSom,AttrArc> &  ElSom<AttrSom,AttrArc>::uniq_pred(ElSubGraphe<AttrSom,AttrArc> & sub)
{
     TArc * res = 0;
     for ( TArc * a =_succ  ;  a   ; a=a->_next)
	 {
		 if (sub.inA(a->arc_rec()))
		 {
			ELISE_ASSERT(res==0,"Multiple succ in uniq_pred");
			res = a;
		 }
	 }
	 ELISE_ASSERT(res!=0,"No succ in uniq_pred");
	 return res->s2();
}

template <class AttrSom,class AttrArc>
         bool  ElSom<AttrSom,AttrArc>::in_sect_angulaire
               (
                     Pt2dr pt,
                     ElSubGraphe<AttrSom,AttrArc> & sub
               )
{
		Pt2dr orig = sub.pt(*this);
		pt -= orig;
		return pt.in_sect_angulaire
			   ( 
				    sub.pt(uniq_succ(sub))-orig,
				    sub.pt(uniq_pred(sub))-orig
			    );
}






/**************************************************************/
/*                                                            */
/*                   ElGraphe                                 */
/*                                                            */
/**************************************************************/

template <class  AttrSom,class AttrArc> 
         ElGraphe<AttrSom,AttrArc>::~ElGraphe()
{
     kill_arc(_larc_free);
}

template <class  AttrSom,class AttrArc> 
         void  ElGraphe<AttrSom,AttrArc>::kill_arc(ElArc<AttrSom,AttrArc> * la)
{
     while (la)
     {
           TArc * next = la->_next;
           DELETE_ONE(la);
           la = next;
     }
}


template <class  AttrSom,class AttrArc> 
         ElArc<AttrSom,AttrArc>*  ElGraphe<AttrSom,AttrArc>::arc_s1s2
         (
                ElSom<AttrSom,AttrArc> & s1,
                ElSom<AttrSom,AttrArc> & s2
         )
{
     TArc * a =s1._succ  ;  
     for 
     ( 
                                                 ;  
           a && (&(a->s2()) !=  (TSom *) &(s2))  ; 
           a=a->_next
     )
     ;

      return a;
}


template <class  AttrSom,class AttrArc> 
         ElArc<AttrSom,AttrArc>&  ElGraphe<AttrSom,AttrArc>::_add_arc
         (
                ElSom<AttrSom,AttrArc> & s1,
                ElSom<AttrSom,AttrArc> & s2,
                const AttrArc &          attr
         )
{

     TArc * res = 0;
     if (_larc_free)
     {
         res = _larc_free;
         _larc_free = _larc_free->_next;
         new (res) TArc  (s1,s2,attr);
     }
     else
       res = CLASS_NEW_ONE(TArc,(s1,s2,attr));
     res->_next = s1._succ;
    s1._succ = res;
    return * res;
}

template <class  AttrSom,class AttrArc> 
  ElSom<AttrSom,AttrArc> & ElGraphe<AttrSom,AttrArc>::new_som(const AttrSom & attr)
{
    _nb_som++;
    if (_free_number.empty())
    {
         _free_number.pushlast(_tsom.nb());
         _tsom.augment_index();
    }
    INT num = _free_number.poplast();
    _tsom[num] = TSom(this,attr,num);
    TSom & aRes =  _tsom[num];
    OnNewSom(aRes);

    return aRes;
}


/**************************************************************/
/*                                                            */
/*                   ElArc                                    */
/*                                                            */
/**************************************************************/

template <class AttrSom,class AttrArc> 
         void  ElArc<AttrSom,AttrArc>::remove()
{
     TSom & S1 = s1();
     TSom & S2 = s2();
     S1.gr().OnKillArc(*this);
     ELISE_ASSERT(S1._alive&&S2._alive,"ElArc::remove()");
     if ((S1._remove_succ(&S2)==0)||(S2._remove_succ(&S1)==0))
     {
        ELISE_ASSERT(false,"ElArc::remove()(Internal error ??? maybe)");
     }
}

#endif //  _ELISE_GRAPHES_GRAPHE_IMPLEM_H 



 /* Footer-MicMac-eLiSe-25/06/2007

    Ce logiciel est un programme informatique servant a  la mise en
    correspondances d'images pour la reconstruction du relief.

    Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
    respectant les principes de diffusion des logiciels libres. Vous pouvez
    utiliser, modifier et/ou redistribuer ce programme sous les conditions
    de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
    sur le site "http://www.cecill.info".

    En contrepartie de l'accessibilite au code source et des droits de copie,
    de modification et de redistribution accordes par cette licence, il n'est
    offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
    seule une responsabilite restreinte pese sur l'auteur du programme,  le
    titulaire des droits patrimoniaux et les concedants successifs.

    A cet egard  l'attention de l'utilisateur est attiree sur les risques
    associes au chargement, a l'utilisation, a la modification et/ou au
    developpement et a la reproduction du logiciel par l'utilisateur etant
    donne sa specificite de logiciel libre, qui peut le rendre complexe a
    manipuler et qui le reserve donc a des developpeurs et des professionnels
    avertis possedant  des  connaissances  informatiques approfondies.  Les
    utilisateurs sont donc invites a charger  et  tester  l'adequation  du
    logiciel a leurs besoins dans des conditions permettant d'assurer la
    securite de leurs systemes et ou de leurs donnees et, plus generalement,
    a l'utiliser et l'exploiter dans les memes conditions de securite.

    Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
    pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
    termes.
    Footer-MicMac-eLiSe-25/06/2007/*/
