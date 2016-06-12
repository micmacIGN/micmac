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

#ifndef _ELISE_GRAPHE_ALGO_PCC
#define _ELISE_GRAPHE_ALGO_PCC

#include "ext_stl/heap.h"


template <class  AttrSom,class AttrArc> class  ElChainePcc;
template <class  AttrSom,class AttrArc> class  CmpElChainePcc;


template <class  AttrSom,class AttrArc> class  ElChainePcc
{
      public :
           typedef ElSom<AttrSom,AttrArc>             TSom;
           typedef CmpElChainePcc<AttrSom,AttrArc>  TCECPCC;
           friend class CmpElChainePcc<AttrSom,AttrArc>;

           ElChainePcc(TSom * pere,TSom * s,REAL  pds) :
               _pere (pere),
               _s    (s   ),
               _pds  (pds )
           {
           }
           ElChainePcc() :
               _pere (0),
               _s    (0),
               _pds  (0)
           {
           }

           TSom * _pere;
           TSom * _s;
           REAL      _pds;
};

template <class  AttrSom,class AttrArc> class  CmpElChainePcc
{
     public :
           typedef ElChainePcc<AttrSom,AttrArc>  TECPCC;

           bool operator () (const TECPCC & t1,const TECPCC & t2)
           {
                return t1._pds < t2._pds;
           }

};




typedef enum
{
    eModePCC_Somme,
    eModePCC_Max,
    eModeArbre_Min
} eModeCoutArbre;



template <class  AttrSom,class AttrArc> class  ElPcc
{
      public :

          typedef ElPcc<AttrSom,AttrArc>             TElPcc;
          typedef ElArc<AttrSom,AttrArc>             TArc;
          typedef ElSom<AttrSom,AttrArc>             TSom;
          typedef ElGraphe<AttrSom,AttrArc>          TGraphe;
          typedef ElSubGraphe<AttrSom,AttrArc>       TSubGraphe;
          typedef ElArcIterator<AttrSom,AttrArc>     TArcIter;
          typedef ElSomIterator<AttrSom,AttrArc>     TSomIter;
          typedef ElChainePcc<AttrSom,AttrArc>       TECPCC;
          typedef CmpElChainePcc<AttrSom,AttrArc>    TCECPCC;



          bool  inf_dist
                (
                        TSom   &          s1,
                        TSom   &          s2,
                        TSubGraphe  &     sub,
                        REAL              DMax,
                        eModeCoutArbre        aModeCout,
                        bool              force_init = false
                );




          TSom *  pcc
                  (
                        TSom   &                              s1,
                        TSom   &                              s2,
                        TSubGraphe  &                        sub,
                        eModeCoutArbre        aModeCout,
                        bool                                 force_init = false,
                        REAL                                 DMax  = 1e60
                  );


          TSom *  pcc
                  (
                        TSom   &                              s1,
                        ElSubGrapheSom<AttrSom,AttrArc>  &   but,
                        TSubGraphe  &                        sub,
                        eModeCoutArbre        aModeCout,
                        bool                                 force_init = false,
                        REAL                                 DMax  = 1e60
                  );


          bool    inf_dist
                  (
                        ElFilo<TSom *>  &                    ts1,
                        ElSubGrapheSom<AttrSom,AttrArc>  &   but,
                        TSubGraphe  &                        sub,
                        REAL                                 DMax, 
                        eModeCoutArbre        aModeCout,
                        bool                                 force_init = false
                  );

          TSom *  pcc
                  (
                        ElFilo<TSom *>  &                    ts1,
                        ElSubGrapheSom<AttrSom,AttrArc>  &   but,
                        TSubGraphe  &                        sub,
                        eModeCoutArbre        aModeCout,
                        bool                                 force_init = false,
                        REAL                                 DMax  = 1e60
                  );

          void chemin(ElFilo<TSom *>  &,TSom &);

          ElPcc() : 
              _flag (-1),
              _heap (TCECPCC())
          {}

          bool reached(TSom & s) {return s.flag_kth(_flag);}
          REAL pds(TSom & s) 
          {
               ELISE_ASSERT(reached(s),"ElPcc::pds");
               return _pds[s.num()];
          }
          REAL pds(TSom & s,double aDef) 
          {
               return reached(s) ? _pds[s.num()] : aDef;
          }

          REAL pdsDef(TSom & s)
          {
               return pds(s,1e60);
          }




          TSom *  pere(TSom & s) 
          {
               ELISE_ASSERT(reached(s),"ElPcc::pere");
               return _pere[s.num()];
          }

          const ElFilo<TSom *> &  reached_soms() {return _reached;}


      private :

         ElFilo<TSom *>      _som_sing;
         INT                     _flag;
         ElFilo<TSom * >      _reached;
         ElTabDyn<REAL>           _pds;
         ElTabDyn<TSom * >       _pere;
         ElHeap
         <
               ElChainePcc<AttrSom,AttrArc>,
               CmpElChainePcc<AttrSom,AttrArc> 
         > _heap;
};


template <class  AttrSom,class AttrArc> 
          void ElPcc<AttrSom,AttrArc>::chemin
               (
                      ElFilo<ElSom<AttrSom,AttrArc> *> &  ts1,
                      ElSom<AttrSom,AttrArc>           &  s_init
               )
{
    ELISE_ASSERT(reached(s_init),"ElPcc::chemin");
    ts1.clear();
    TSom * s = & s_init;
    while(s)
    {
         ts1.pushlast(s);
         s = pere(*s);
    }
    
}


template <class  AttrSom,class AttrArc> 
        ElSom<AttrSom,AttrArc> *  
        ElPcc<AttrSom,AttrArc>::pcc
              (
                  ElSom<AttrSom,AttrArc>  &           s1,
                  ElSom<AttrSom,AttrArc>  &           s2,
                  ElSubGraphe<AttrSom,AttrArc>     &  sub,
                  eModeCoutArbre                          aModeCout,
                  bool                                force_init,
                  REAL                                DMax
              )
{
     ElSubGrapheSingleton<AttrSom,AttrArc> but(s2);
     return pcc(s1,but,sub,aModeCout,force_init,DMax);
}

template <class  AttrSom,class AttrArc> 
        bool  ElPcc<AttrSom,AttrArc>::inf_dist
              (
                  ElSom<AttrSom,AttrArc>  &           s1,
                  ElSom<AttrSom,AttrArc>  &           s2,
                  ElSubGraphe<AttrSom,AttrArc>     &  sub,
                  REAL                                DMax,
                  eModeCoutArbre                          aModeCout,
                  bool                                force_init
              )
{
     return pcc(s1,s2,sub,aModeCout,force_init,DMax) != 0;
}







template <class  AttrSom,class AttrArc> 
        ElSom<AttrSom,AttrArc> *  
        ElPcc<AttrSom,AttrArc>::pcc
              (
                  ElSom<AttrSom,AttrArc>  &           s1,
                  ElSubGrapheSom<AttrSom,AttrArc>  &  but,
                  ElSubGraphe<AttrSom,AttrArc>     &  sub,
                  eModeCoutArbre                          aModeCout,
                  bool                                force_init,
                  REAL                                DMax
              )
{
    _som_sing.clear();
    _som_sing.pushlast(&s1);

    return pcc(_som_sing,but,sub,aModeCout,force_init,DMax);
}


template <class  AttrSom,class AttrArc> 
        ElSom<AttrSom,AttrArc> *  
        ElPcc<AttrSom,AttrArc>::pcc
              (
                  ElFilo<ElSom<AttrSom,AttrArc> *> &  ts1,
                  ElSubGrapheSom<AttrSom,AttrArc>  &  but,
                  ElSubGraphe<AttrSom,AttrArc>     &  sub,
                  eModeCoutArbre                          aModeCout,
                  bool                                force_init,
                  REAL                                DMax
              )
{
      _heap.clear();
	  INT k;
      for ( k= 0; k<_reached.nb();k++)
          _reached[k]->flag_set_kth_false(_flag);
      _reached.clear();

      if (! ts1.nb()) return 0;

      TGraphe & gr = ts1[0]->gr();

      if (_flag == -1)
      {
          _flag = gr.alloc_flag_som();
          TSubGraphe all;
          for (TSomIter s = gr.begin(all); s.go_on() ; s++)
              (*s).flag_set_kth_false(_flag);
      }



      INT last_nb = _pere.nb();
      INT new_nb = gr.nb_som_phys();
      _pere.set_intexable_until(new_nb);
      _pds.set_intexable_until(new_nb);

	  for( k= last_nb ; k<new_nb ; k++) _pere[k] = 0;

       for ( k=0; k<ts1.nb() ; k++)
       {
           TSom * s = ts1[k];
           if ((sub.inS(*s)) || force_init)
           {
              _heap.push(TECPCC(0,s,sub.pds(*s)));
           }
       }

       TECPCC  te;
       bool cont = true;
       TSom * res = 0;
       while (cont &&  _heap.pop(te))
       {
            TSom & s = *(te._s);
            if ( te._pds <= DMax)
            {
                 if (! s.flag_kth(_flag))
                 {
                    _pds[s.num()] = te._pds;
                    _pere[s.num()] = te._pere;
                    _reached.pushlast(&s);
                    s.flag_set_kth_true(_flag);

                    if (but.inS(s))
                    {   
                        res = & s;
                        cont = false;
                    }
                    else
                    {
                        for (
                               TArcIter  ait = s.begin(sub);
                               ait.go_on()       ;
                               ait++             
                            )
                         {
                             TSom * s2 = &((*ait).s2());
                             if (! s2->flag_kth(_flag))
                             {
                                 REAL pds = sub.pds(*ait);
                                 if (aModeCout==eModePCC_Somme)
                                 {
                                    pds += te._pds;
                                 }
                                 else if (aModeCout==eModePCC_Max)
                                 {
                                    ElSetMax(pds,te._pds);
                                 }
                                 else if (aModeCout==eModeArbre_Min)
                                 {
                                 }

                                 _heap.push(TECPCC(&s,s2,pds));
                             }
                         }
                    }
                 }
            }
            else
              cont = false;
       }
       return res;
}

template <class  AttrSom,class AttrArc> 
        bool ElPcc<AttrSom,AttrArc>::inf_dist
              (
                  ElFilo<ElSom<AttrSom,AttrArc> *> &  ts1,
                  ElSubGrapheSom<AttrSom,AttrArc>  &  but,
                  ElSubGraphe<AttrSom,AttrArc>     &  sub,
                  REAL                                DMax,
                  eModeCoutArbre                          aModeCout,
                  bool                                force_init
              )
{
    return pcc(ts1,but,sub,aModeCout,force_init,DMax) != 0;
}




#endif  // _ELISE_GRAPHE_ALGO_PCC


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
