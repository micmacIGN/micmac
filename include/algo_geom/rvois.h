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

#ifndef _ELISE_ALGO_GEOM_RVOIS
#define _ELISE_ALGO_GEOM_RVOIS

template <class Type,class Fpt> class ElComparPtOn_x_then_y
{
     public :
        ElComparPtOn_x_then_y(Fpt fpt) : _fpt (fpt) {}
        bool operator () (const Type & o1,const Type & o2)
        {
             Pt2dr p1 = _fpt(o1);
             Pt2dr p2 = _fpt(o2);
             if (p1.x < p2.x) return true;
             if (p1.x > p2.x) return false;
             return p1.y < p2.y;
        }

        bool operator () (const Type * o1,const Type * o2) {return (*this)(*o1,*o2); }


     private :
        Fpt _fpt;
};


template <class Type,class Fpt> class ElComparPts
{
     public :
        ElComparPts(Fpt fpt) : _fpt (fpt) {}
        bool operator () (const Type & o1,const Type & o2)
        {
             return  _fpt(o1) == _fpt(o2);
        }
        bool operator () (const Type * o1,const Type * o2) {return (*this)(*o1,*o2); }

     private :
        Fpt _fpt;
};






template <class Type,class Fpt,class Act>  void
         rvoisins_sortx
         (
             Type * begin,
             Type * end,
             REAL  dist,
             Fpt   fpt,
             Act   act
         )
{
     ElComparPtOn_x_then_y<Type,Fpt> Cmp(fpt);
     STDSORT(begin,end,Cmp);

     REAL d2 = ElSquare(dist);
     for( Type * it1=begin; it1!=end ; it1++)
     {
          Pt2dr p1 = fpt(*it1);
          Type * it2 = it1;
          REAL x_lim = p1.x + dist;
          it2++;
          while ( it2 != end &&  (fpt(*it2).x < x_lim))
          {
              if (square_euclid(p1,fpt(*it2)) < d2)
                 act(*it1,*it2);
              it2++;
          }
     }

}

/*
     Tout public puisque, en l'absence de methode template, on 
    doit passer par un fonction globale.
*/

template <class Type,class Fpt> class  RvoisinsSortX
{

     public :
        class Pair
        {
            public : 
                Pair(Type * val,Pt2di pt) : _val(val) , _pt (pt) {}
                Pair() {}

                Type * _val;
                Pt2di  _pt;
        };

        class PtOfPair
        {
            public : 
              Pt2di operator () (const Pair & p) {return p._pt;};
        };


        Fpt                                      _fpt;
        ElComparPtOn_x_then_y<Pair,PtOfPair>     _cmp;
        ElFilo<Pair>                             _objs;


        RvoisinsSortX(const Fpt & fpt) : _fpt (fpt),_cmp (PtOfPair()) {}

        void init(Type * begin,Type * end);

   
};

template <class Type,class Fpt> void RvoisinsSortX<Type,Fpt>::init(Type * begin,Type * end)
{
     _objs.clear();
     for (Type * v =begin; v!= end ; v++)
         _objs.pushlast(Pair(v,_fpt(*v)));
     STDSORT(_objs.tab(),_objs.tab() +_objs.nb(), _cmp);
}

template <class Type,class Fpt,class Act>  
         bool MapRvoisinsSortX
              (
                    RvoisinsSortX<Type,Fpt> &  RV,
                    Pt2di                      pt,
                    REAL                       dist,
                    Act &                      act
              )
{
      typedef typename RvoisinsSortX<Type,Fpt>::Pair Pair;
      INT  xinf = round_down(pt.x-dist);
      INT  xsup = round_up(pt.x+dist);
      REAL d2 = ElSquare(dist);
      Pair * end = RV._objs.tab() +RV._objs.nb();

      for (
            Pair * p = lower_bound ( RV._objs.tab(), end, Pair(0,Pt2di(xinf,pt.y)), RV._cmp);
            (p!= end) && (p->_pt.x <= xsup)  ;
            p++
          )
          if (square_euclid(pt,p->_pt) < d2)
          {
               if (act(*(p->_val))) return true;
          }

     return false;
}





#endif // _ELISE_ALGO_GEOM_RVOIS


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
