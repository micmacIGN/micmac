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



/*
      Renvoie le point le plus a droite des points le plus en bas
*/
template <class TPt> TPt pts_down_right(const ElFilo<TPt> & pts)
{
     ELISE_ASSERT(pts.nb(),"Cmp_Vect_Then_Norm::down_right");
     TPt res = pts[0];
     for (INT k=1; k<pts.nb() ; k++)
     {
         TPt p = pts[k];
         if (     (p.y<res.y)
              ||  ((p.y==res.y) && (p.x>res.x))
            )
            res = p;
     }
     return res;
}

/*
     Classe permettant (via le sort STL) de trier les
     points selon l'ordre trigo des vecteur partant
     de "_down_right".

     Les points  alignes avec "_down_right" sont departage 
     en fonction de leur normes  (de maniere croissant ou 
     decroissante suivant le valeur de "_norm_croissante").
    
      Pour des points entier, tous les calculs sont exacts
      (hors Overflow !!) : 
           * ``angle'' relatif calcule par produit vectoriel;
           * norme calculee selon d4
*/

template <class TPt> class  Cmp_Vect_Then_Norm
{

       public :

           typedef ElTyName TPt::TypeScal TScal;
           Cmp_Vect_Then_Norm
           (
                 TPt                 DN,
                 const ElFilo<TPt> & pts,
                 bool                NCr
           ) :
                _down_right         (DN),
                _pts                (pts),
                _norm_croissante    (NCr)
           {
           }


          // lorsque un des points vaut down right,
          // le produit est nul, donc la comparaison se
          // se fait sur les norme et c'est tjs down right le
          // le + petit, donc c'est (arbitraire mais) coherent

           bool operator()(INT i1,INT i2)
           {
                TPt p1 = _pts[i1] - _down_right;
                TPt p2 = _pts[i2] - _down_right;

                TScal  pvect = p1^p2;
                if (pvect) return (pvect > 0);

                TScal n1 = dist4(p1);
                TScal n2 = dist4(p2);
                return _norm_croissante ? (n1<n2) : (n1>n2) ;
                         
           }
       private  :


           TPt                       _down_right;
           const ElFilo<TPt> &       _pts;
           bool                      _norm_croissante;
};



template <class TPt> class  Env_Conv
{
     private :
         typedef ElTyName TPt::TypeScal TScal;

         ElFifo<INT> &        _ind;
         const ElFilo<TPt> &  _pts;
         bool                 _env_min;

         bool sup_inter(INT m,INT i)
         {
              TScal pvect = 
                             (_pts[_ind[m]] -_pts[_ind[m-1]])
                           ^ (_pts[_ind[i]] -_pts[_ind[m-1]]);

               return _env_min ? (pvect<=0) : (pvect <0) ;
         }


     public :
       Env_Conv
       (
              ElFifo<INT> &        ind,
              const ElFilo<TPt> &  pts,
              bool                 env_min 
        )  :
              _ind (ind),
              _pts (pts),
              _env_min (env_min)
        {}

        void calc()
        {
            _ind.set_circ(true);
            _ind.clear();
            INT nb_pts = _pts.nb();
            if (nb_pts ==0) return;

            for (INT k=0; k<nb_pts ; k++)  _ind.pushlast(k);

            TPt down_right = pts_down_right(_pts);

            Cmp_Vect_Then_Norm<TPt> cmp_crois(down_right,_pts,true);
            STDSORT(_ind.tab(),_ind.tab()+nb_pts,cmp_crois);

            {
                // supression des duplicatas 

                INT k_ins = 1;
                for (INT k=1 ; k<_ind.nb() ; k++)
                    if (_pts[_ind[k]] != _pts[_ind[k-1]])
                       _ind[k_ins++] = _ind[k];
                while (_ind.nb() > k_ins) _ind.poplast();
                nb_pts = _ind.nb();

                // evite de traiter ce cas degenere
                if (nb_pts == 1) return;
            }

            {
                TPt last = _pts[_ind.top()];
                INT k = nb_pts -1;
                while (
                          (k>=0)
                        &&( ((last-down_right) ^ (_pts[_ind[k]]-down_right)) == 0)
                      )
                      k--;
                k++;
                Cmp_Vect_Then_Norm<TPt> cmp_decrois(down_right,_pts,false);
				STDSORT(_ind.tab()+k,_ind.tab()+nb_pts,cmp_decrois);

                // si tout les poinst allignes, et env_min le test sup_iterm ne
                // fonctionne pas bien au niveau du rebroussement, donc on sort maintenant
                if ((k==0) && _env_min)
                {
                    while (_ind.nb() > 2) _ind.poplast();
                    return;
                }
            }

            INT v0 = _ind[0];
            _ind.pushlast(v0);
            INT m = 1;
            for (INT i = 2; i < _ind.nb() ; i++)
            {
                while ((m>0) && sup_inter(m,i)) 
                      m--;
                m++;
                _ind[m] = _ind[i];
            }

            while (_ind.nb()> m) _ind.poplast();
        }


};

void env_conv
     ( 
         ElFifo<INT> &          ind,
         const ElFilo<Pt2di> &  pts,
         bool                   env_min = true
     )
{
     Env_Conv<Pt2di> env(ind,pts,env_min);
     env.calc();
}

void env_conv
     ( 
         ElFifo<INT> &          ind,
         const ElFilo<Pt2dr> &  pts,
         bool                   env_min = true
     )
{
     Env_Conv<Pt2dr> env(ind,pts,env_min);
     env.calc();
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
