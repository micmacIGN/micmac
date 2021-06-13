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
/*                                                                 */
/*           UTILITAIRES GENERAUX                                  */
/*                                                                 */
/*******************************************************************/

Pt2dlr ToPt2dlr(const Pt2dlr & aP) {return aP;}
Pt2dlr ToPt2dlr(const Pt2di & aP) {return Pt2dlr(aP.x,aP.y);}
Pt2dlr ToPt2dlr(const Pt2dr & aP) {return Pt2dlr(aP.x,aP.y);}


template <class Type>  Box2d<Type>::Box2d(Pt2di P0,Pt2di P1) : _p0 (P0.x,P0.y), _p1 (P1.x,P1.y)
{
     pt_set_min_max(_p0,_p1);
}


static Pt2di BoxUp(Pt2dr aP,INT  *) {return round_up(aP);}
static Pt2dr BoxUp(Pt2dr aP,REAL *) {return aP;}
static Pt2dlr BoxUp(Pt2dr aP,long double *){return ToPt2dlr(aP);}
static Pt2di BoxDown(Pt2dr aP,INT  *) {return round_down(aP);}
static Pt2dr BoxDown(Pt2dr aP,REAL *) {return aP;}
static Pt2dlr BoxDown(Pt2dr aP,long double *){return ToPt2dlr(aP);}

// static Pt2di BoxUp(Pt2dlr aP,INT  *) {return round_up(aP);}
// static Pt2dr BoxUp(Pt2dlr aP,REAL *) {return aP;}
// static Pt2dlr BoxUp(Pt2dlr aP,long double *){return ToPt2dlr(aP);}
// static Pt2di BoxDown(Pt2dlr aP,INT  *) {return round_down(aP);}
// static Pt2dr BoxDown(Pt2dlr aP,REAL *) {return aP;}
// static Pt2dlr BoxDown(Pt2dlr aP,long double *){return ToPt2dlr(aP);}





template <class Type>
 Box2d<double> Box2d<Type>::BoxImage(const cElMap2D & aMap) const
{
   Pt2d<Type> aC4[4];
   Corners(aC4);
   
   Pt2dr aP0=aMap(ToPt2dr(aC4[0]));
   Pt2dr aP1= aP0;

   for (int aK=1 ; aK<4 ; aK++)
   {
      aP0.SetInf(aMap(ToPt2dr(aC4[aK])));
      aP1.SetSup(aMap(ToPt2dr(aC4[aK])));
   }
   return  Box2d<double>(aP0,aP1);
}



template <class Type>   Box2d<Type>::Box2d(Pt2dr P0,Pt2dr P1) 
{
     pt_set_min_max(P0,P1);
     _p0 = BoxDown(P0,(Type *)0);
     _p1 = BoxUp  (P1,(Type *)0);
}

template <class Type>   Box2d<Type>::Box2d(Pt2dlr P0,Pt2dlr P1) 
{
     pt_set_min_max(P0,P1);
     _p0 = Pt2d<Type>(P0.x,P0.y);
     _p1 = Pt2d<Type>(P1.x,P1.y);
     //  _p1 = BoxUp  (P1,(Type *)0);
}

template <class Type>   Box2d<Type>::Box2d(Pt2d<Type> P)
{
   *this = Box2d<Type>(-P,P);
}

template <class Type>   Box2d<Type>::Box2d(Type xy)
{
   *this = Box2d<Type>(Pt2d<Type>(xy,xy));
}

template <class Type>   Box2d<Type> Box2d<Type>::trans(Pt2d<Type> tr) const
{
    return Box2d<Type>(_p0+tr,_p1+tr);
}

template <class Type>   Box2d<Type>::Box2d(const Type *x,const Type * y,INT nb)
{
    _p0.x = OpMin.red_tab(x,nb);
    _p0.y = OpMin.red_tab(y,nb);
    _p1.x = OpMax.red_tab(x,nb);
    _p1.y = OpMax.red_tab(y,nb);
}

template <class Type>   Box2d<Type>::Box2d(const Pt2d<Type> *aVPts,INT nb)
{
    ELISE_ASSERT(nb!=0,"Box2d<Type>::Box2d(const Pt2d<Type> *x,INT nb");
    _p0 = _p1 = aVPts[0];
    for (INT aK=1; aK<nb ; aK++)
    {
        _p0.SetInf(aVPts[aK]);
        _p1.SetSup(aVPts[aK]);
    }
}


template <class Type>   Box2d<Type>  Sup(const Box2d<Type> & b1,const Box2d<Type> & b2)
{
      return  Box2d<Type>
              (
                   Inf(b1._p0,b2._p0),
                   Sup(b1._p1,b2._p1)
              );
}


template <class Type>   Box2d<Type>  Inf(const Box2d<Type> & b1,const Box2d<Type> & b2)
{
      return  Box2d<Type>
              (
                   Sup(b1._p0,b2._p0),
                   Inf(b1._p1,b2._p1)
              );
}

template <class Type> Pt2dr  Box2d<Type>::FromCoordBar(Pt2dr aCBar) const
{
   return Pt2dr
	   (
	       _p0.x*(1-aCBar.x) + _p1.x*aCBar.x,
	       _p0.y*(1-aCBar.y) + _p1.y*aCBar.y

	   );
}

template <class Type> Pt2dr
   Box2d<Type>::RandomlyGenereInside() const
{
   return  FromCoordBar(Pt2dr(NRrandom3(),NRrandom3()));

}

Pt2di  RandomlyGenereInside(const Box2di & aBox)
{
   Pt2di aSz = aBox.sz();
   return aBox._p0+Pt2di(NRrandom3(aSz.x),NRrandom3(aSz.y));
}


template <class Type> double Box2d<Type>::Interiorite(const Pt2dr & aP) const
{
   return ElMin4
          (
                ElMax(0.0,double(aP.x-_p0.x)),
                ElMax(0.0,double(aP.y-_p0.y)),
                ElMax(0.0,double(_p1.x-aP.x)),
                ElMax(0.0,double(_p1.y-aP.y))
          );
}

#define INSTANTIATE_BOX2D(Type)\
template  Box2d<Type>  Sup(const Box2d<Type> & b1,const Box2d<Type> & b2);\
template  Box2d<Type>  Inf(const Box2d<Type> & b1,const Box2d<Type> & b2);




INSTANTIATE_BOX2D(INT)
INSTANTIATE_BOX2D(REAL)



template <class Type> bool Box2d<Type>::include_in(const Box2d<Type> & b2) const
{
     return    b2._p0.xety_inf_ou_egal(_p0   )
            &&    _p1.xety_inf_ou_egal(b2._p1);
}




template <class Type> Box2d<Type>  Box2d<Type>::erode(Pt2d<Type> pt) const
{
    return Box2d<Type>(_p0+pt,_p1-pt);
}                            



template <class Type> Box2d<Type> 
         Box2d<Type>::AddTol(const Box2d<Type> & aB) const
{
    return Box2d<Type>(_p0-aB._p0,_p1+aB._p1);
}                            
template <class Type> Box2d<Type> 
         Box2d<Type>::AddTol(const Pt2d<Type> & aP) const
{
    return Box2d<Type>(_p0-aP,_p1+aP);
}                            
template <class Type> Box2d<Type> 
         Box2d<Type>::AddTol(const Type & aV) const
{
    return AddTol(Pt2d<Type>(aV,aV));
}                            

template <class Type> Box2d<Type>  Box2d<Type>::dilate(Pt2d<Type> pt) const
{
    return Box2d<Type>(_p0-pt,_p1+pt);
}                            

template <class Type> Box2d<Type>  Box2d<Type>::dilate(Type xy) const
{
    return dilate(Pt2d<Type>(xy,xy));
}                            



template <class Type> bool  Box2d<Type>::inside(const Pt2d<Type> & p) const
{
    return    (p.x >= _p0.x)
         &&   (p.y >= _p0.y)
         &&   (p.x <= _p1.x)
         &&   (p.y <= _p1.y);
}

template <class Type> bool  Box2d<Type>::inside_std(const Pt2d<Type> & p) const
{
    return    (p.x >= _p0.x)
         &&   (p.y >= _p0.y)
         &&   (p.x < _p1.x)
         &&   (p.y < _p1.y);
}


template <class Type>
         bool Box2d<Type>::Include
              (
                   const cElTriangleComp & aTri
              ) const
{
        return     contains(aTri.P0())
               &&  contains(aTri.P1())
               &&  contains(aTri.P2()) ;
}
template <class Type>
         bool Box2d<Type>::Intersecte
              (
                   const cElTriangleComp & aTri
              ) const
{
  if (
            contains(aTri.P0() )
         || contains(aTri.P1() )
         || contains(aTri.P2() )
     )
        return true;

   Pt2d<Type> Corn[4];
   Corners(Corn);

   for (INT aK=0 ; aK<4 ; aK++)
       if (aTri.Inside(ToPt2dr(Corn[aK])))
           return true;

  if (
            Intersecte(aTri.S01())
         || Intersecte(aTri.S12())
         || Intersecte(aTri.S20())
     )
        return true;

  return false;
}





/*******************************************************************/
/*******************************************************************/

/*******************************************************************/
/*                                                                 */
/*           SPECIFIQUE QT                                         */
/*                                                                 */
/*******************************************************************/

/*
      Numerotation de "Freeman" etendue :

          3  2  1
          4  8  0
          5  6  7

*/

           //  GENERAUX

template <class Type> INT Box2d<Type>::freeman_pos(const Pt2dr & pt) const
{
   if (pt.x >= _p1.x)
      return ( (pt.y > _p1.y) ? 1 : ( (pt.y >= _p0.y) ? 0 : 7 ));

   if (pt.x >= _p0.x)
      return ( (pt.y > _p1.y) ? 2 : ( (pt.y >= _p0.y) ? 8 : 6 ));

   return ( (pt.y > _p1.y) ? 3 : ( (pt.y >= _p0.y) ? 4 : 5 ));         
}





template <class Type> REAL8  Box2d<Type>::Freem0SquareDist (const Pt2dr & pt) const
{
    return ElSquare(pt.x-_p1.x);
}                                               

template <class Type> REAL8  Box2d<Type>::Freem1SquareDist (const Pt2dr & pt) const
{
    return  ElSquare(pt.x-_p1.x) + ElSquare(pt.y-_p1.y);
}                                               

template <class Type> REAL8  Box2d<Type>::Freem2SquareDist (const Pt2dr & pt) const
{
    return ElSquare(pt.y-_p1.y);
}                                               

template <class Type> REAL8  Box2d<Type>::Freem3SquareDist (const Pt2dr & pt) const
{
    return  ElSquare(pt.x-_p0.x) + ElSquare(pt.y-_p1.y);
}                                               

template <class Type> REAL8  Box2d<Type>::Freem4SquareDist (const Pt2dr & pt) const
{
    return ElSquare(pt.x-_p0.x);
}                                               

template <class Type> REAL8  Box2d<Type>::Freem5SquareDist (const Pt2dr & pt) const
{
    return  ElSquare(pt.x-_p0.x) + ElSquare(pt.y-_p0.y);
}                                               

template <class Type> REAL8  Box2d<Type>::Freem6SquareDist (const Pt2dr & pt) const
{
    return ElSquare(pt.y-_p0.y);
}                                               

template <class Type> REAL8  Box2d<Type>::Freem7SquareDist (const Pt2dr & pt) const
{
    return  ElSquare(pt.x-_p1.x) + ElSquare(pt.y-_p0.y);
}                                               


template <class Type> REAL8  Box2d<Type>::Freem8SquareDist (const Pt2dr & ) const
{
    return  0.0;
}                                               


template <> Box2di::R_fonc_Pt2dr Box2di::_Tab_FreemSquareDist[9] =
                     {
                          &Box2di::Freem0SquareDist,
                          &Box2di::Freem1SquareDist,
                          &Box2di::Freem2SquareDist,
                          &Box2di::Freem3SquareDist,
                          &Box2di::Freem4SquareDist,
                          &Box2di::Freem5SquareDist,
                          &Box2di::Freem6SquareDist,
                          &Box2di::Freem7SquareDist,
                          &Box2di::Freem8SquareDist
                     };

template <> Box2dr::R_fonc_Pt2dr Box2dr::_Tab_FreemSquareDist[9] =
                     {
                          &Box2dr::Freem0SquareDist,
                          &Box2dr::Freem1SquareDist,
                          &Box2dr::Freem2SquareDist,
                          &Box2dr::Freem3SquareDist,
                          &Box2dr::Freem4SquareDist,
                          &Box2dr::Freem5SquareDist,
                          &Box2dr::Freem6SquareDist,
                          &Box2dr::Freem7SquareDist,
                          &Box2dr::Freem8SquareDist
                     };


typedef Box2d<long double>  Box2dlr;
template <> Box2dlr::R_fonc_Pt2dr Box2dlr::_Tab_FreemSquareDist[9] =
                     {
                          0,0,0, 0,0,0, 0,0,0
                     };



           //    BOX-PT

template <class Type> bool Box2d<Type>::Intersecte(const Pt2dr  & p)  const
{
    return     (p.x >= _p0.x)
            && (p.y >= _p0.y)
            && (p.x <  _p1.x)
            && (p.y <  _p1.y) ;
}

template <class Type> bool Box2d<Type>::Include(const Pt2dr  & p)  const
{
     return Intersecte(p);
}

template <class Type> REAL Box2d<Type>::SquareDist(const Pt2dr & pt) const
{

     //  return (this->*_Tab_FreemSquareDist[freeman_pos(pt)])(pt);
     // genere un warning bizzare sur WSC5 :
     // Warning (Anachronism): Pointer to non-const member function used with const object.

     return ((const_cast<Box2d<Type> *>(this))->*_Tab_FreemSquareDist[freeman_pos(pt)])(pt);
}

template <class Type> REAL Box2d<Type>::SquareDist(const Pt2dr & pt,int c) const
{
     return ((const_cast<Box2d<Type> *>(this))->*_Tab_FreemSquareDist[c])(pt);
}


template <class Type> void   Box2d<Type>::QSplit(typename Box2d<Type>::QBox & aQB) const
{
      Pt2d<Type> p01 = _p0 + sz()/2;

      aQB[0]._p0 = _p0; aQB[0]._p1 = p01;
      aQB[1]._p0 = Pt2d<Type>(p01.x,_p0.y); aQB[1]._p1 = Pt2d<Type>(_p1.x,p01.y);
      aQB[2]._p0 = Pt2d<Type>(_p0.x,p01.y); aQB[2]._p1 = Pt2d<Type>(p01.x,_p1.y);
      aQB[3]._p0 = p01; aQB[3]._p1 = _p1;
}

template <class Type> void   Box2d<Type>::QSplitWithRab(typename Box2d<Type>::QBox & aQB,Type aRab) const
{
   QSplit(aQB);

   for (int aK=0 ; aK<4 ; aK++)
   {
        aQB[aK] = aQB[aK].AddTol(aRab);
        aQB[aK] = Inf(aQB[aK],*this);
   }
}


template <class Type> void  Box2d<Type>::Corners(typename Box2d<Type>::P4 & p4) const
{
     p4[0] = _p0;
     p4[1] = Pt2d<Type>(_p1.x,_p0.y);
     p4[2] = _p1;
     p4[3] = Pt2d<Type>(_p0.x,_p1.y);
}


/*
      Ce qui suit n'est surement pas du tout dans le style Objet.
    Comme c'est plutot complique, j'importe le + directement  possible
    de clisp (en evitant surtout d'avoir a comprendre comment ca marche).

      De toute facon c'est totalement encapsule.
*/


template <class Type> class BoxFreemanCompil
{
    public :
 
       enum {ElBIDON = -1000000000};

       enum
       {
             INTER_TOUJ,      /* un en 8 ou 0-4 ou 2-6 */
                              /* positions, sans jamais d'intersection */
             CFF_00,
             CFF_11,
             CFF_13,
             CFF_01,
                              /* positions, avec parfois des intersection */
             CFF_02,
             CFF_03,
             CFF_15
       };

       INT TAB_CONF_CPLE_FREEM[9][9];
       INT TAB_IND_QX[9][9];  /* tableau des indices x des "Q-Vois" dans Freem */
       INT TAB_IND_QY[9][9];  /* tableau des indices y des "Q-Vois" dans Freem */
       INT TAB_IND_Q2X[9][9];  /* idem, quand y'a deux points */
       INT TAB_IND_Q2Y[9][9];  /* idem, quand y'a deux points */
       INT TAB_TRIGO_CF[8][8];

       BoxFreemanCompil(int);

       void symetrise_tab_cfr(INT tab [][9]);
       void init_tab_cfr(INT tab [][9]);

       REAL  D2BoxSeg(const Box2d<Type> & b,const SegComp & s);

       inline Pt2dr  PQ1(const Box2d<Type> & b,INT c1,INT c2)
       {
             return Pt2dr
                    (  
                        b.x(TAB_IND_QX[c1][c2]),
                        b.y(TAB_IND_QY[c1][c2])
                    );
       }
       inline Pt2dr  PQ2(const Box2d<Type> & b,INT c1,INT c2)
       {
             return Pt2dr
                    (  
                        b.x(TAB_IND_Q2X[c1][c2]),
                        b.y(TAB_IND_Q2Y[c1][c2])
                    );
       }
       static BoxFreemanCompil<Type>  TheBFC;
};


template <> BoxFreemanCompil<INT> BoxFreemanCompil<INT>::TheBFC(4);
template <> BoxFreemanCompil<REAL> BoxFreemanCompil<REAL>::TheBFC(4);
template <> BoxFreemanCompil<long double> BoxFreemanCompil<long double>::TheBFC(4);

//template <> BoxFreemanCompil<INT> BoxFreemanCompil<INT>::TheBFC=BoxFreemanCompil<INT>();
//template <> BoxFreemanCompil<REAL> BoxFreemanCompil<REAL>::TheBFC=BoxFreemanCompil<REAL>();


template <class Type> void BoxFreemanCompil<Type>::symetrise_tab_cfr(INT tab [][9])
{
    INT i,j;
    for (i = 0 ; i < 9 ; i++)
        for (j = 0 ; j < 9 ; j++)
            if (tab[i][j] == ElBIDON)
               tab[i][j] = tab[j][i];
}




template <class Type> void BoxFreemanCompil<Type>::init_tab_cfr(INT tab[][9])
{
   INT i,j;
    for (i = 0 ; i < 9 ; i++)
        for (j = 0 ; j < 9 ; j++)
            tab[i][j] = ElBIDON;
}


template <class Type> BoxFreemanCompil<Type>::BoxFreemanCompil(int)
{
   INT IND_X_COIN[4] = {1,0,0,1};
   INT IND_Y_COIN[4] = {1,1,0,0};


   INT i,d;

    for (i = 0 ; i < 8 ; i++)
        for (d = 0 ; d < 8 ; d++)
            TAB_TRIGO_CF[i][d] = 555;
    for (i = 0 ; i < 8 ; i++)
    {
        for (d = 1 ; d < 4 ; d++)
        {
            TAB_TRIGO_CF[i][(i+d)%8] = 1;
            TAB_TRIGO_CF[(i+d)%8][i] = 0;
        }
    }

    /* init a une valeur bidon */
    init_tab_cfr(TAB_CONF_CPLE_FREEM);
    init_tab_cfr(TAB_IND_QX);
    init_tab_cfr(TAB_IND_QY);
    init_tab_cfr(TAB_IND_Q2X);
    init_tab_cfr(TAB_IND_Q2Y);

/* couples passant par le point central */
    for (i = 0 ; i < 9 ; i++)
            TAB_CONF_CPLE_FREEM[i][8] = INTER_TOUJ;
    /* couple opposes , "4-touchant" le point central */
    for (i = 0 ; i < 4 ; i+=2)
        TAB_CONF_CPLE_FREEM[i][i+4] = INTER_TOUJ;


    /* 4- voisins dans la meme case */
    for (i = 0 ; i < 8 ; i+= 2)
        TAB_CONF_CPLE_FREEM[i][i] = CFF_00;


    /* 8- voisins dans la meme case */
    for (i = 1 ; i < 8 ; i+= 2)
    {
        TAB_IND_QX[i][i] = IND_X_COIN[i/2];
        TAB_IND_QY[i][i] = IND_Y_COIN[i/2];
        TAB_CONF_CPLE_FREEM[i][i] = CFF_11;
    }


    /*   8-voisins a deux cases d'ecart */
    for (i = 1 ; i < 8 ; i+= 2)
    {
        TAB_CONF_CPLE_FREEM[i][(i+2)%8] = CFF_13;
        TAB_IND_QX[i][(i+2)%8] = IND_X_COIN[i/2];
        TAB_IND_QY[i][(i+2)%8] = IND_Y_COIN[i/2];
        TAB_IND_Q2X[i][(i+2)%8] = IND_X_COIN[(i/2+1)%4];
        TAB_IND_Q2Y[i][(i+2)%8] = IND_Y_COIN[(i/2+1)%4];
    }

    /*   4 et 8-voisins a une case d'ecart */
    for (i = 0 ; i < 8 ; i++)
    {
        TAB_CONF_CPLE_FREEM[i][(i+1)%8] = CFF_01;
        TAB_IND_QX[i][(i+1)%8] = IND_X_COIN[i/2];
        TAB_IND_QY[i][(i+1)%8] = IND_Y_COIN[i/2];
    }

    /*   4-voisins a deux cases d'ecart */
    for (i = 0 ; i < 8 ; i+= 2)
    {
        TAB_CONF_CPLE_FREEM[i][(i+2)%8] = CFF_02;
        TAB_IND_QX[i][(i+2)%8] = IND_X_COIN[i/2];
        TAB_IND_QY[i][(i+2)%8] = IND_Y_COIN[i/2];
    }

    /*   points en position relatives du cavalier d'echec */
    for (i = 0 ; i < 8 ; i++)
    {
        TAB_CONF_CPLE_FREEM[i][(i+3)%8] = CFF_03;
        TAB_IND_QX[i][(i+3)%8] = IND_X_COIN[((i+1)/2)%4];
        TAB_IND_QY[i][(i+3)%8] = IND_Y_COIN[((i+1)/2)%4];
    }

    /*   points en diagonales opposees */
    for (i = 1 ; i < 8 ; i+= 2)
    {
        TAB_CONF_CPLE_FREEM[i][(i+4)%8] = CFF_15;
        TAB_IND_QX[i][(i+4)%8]  = IND_X_COIN[(i/2+1)%4];
        TAB_IND_QY[i][(i+4)%8]  = IND_Y_COIN[(i/2+1)%4];
        TAB_IND_Q2X[i][(i+4)%8] = IND_X_COIN[(i/2+3)%4];
        TAB_IND_Q2Y[i][(i+4)%8] = IND_Y_COIN[(i/2+3)%4];
    }
    /* symetrisation a posteriori */

    symetrise_tab_cfr(TAB_CONF_CPLE_FREEM);
    symetrise_tab_cfr(TAB_IND_QX);
    symetrise_tab_cfr(TAB_IND_QY);
    symetrise_tab_cfr(TAB_IND_Q2X);
    symetrise_tab_cfr(TAB_IND_Q2Y);
}


template <class Type> REAL  BoxFreemanCompil<Type>::D2BoxSeg(const Box2d<Type> & b,const SegComp & s)
{
   if (s.p0() == s.p1())
      return b.SquareDist(s.p0());

   INT c1 = b.freeman_pos (s.p0());
   INT c2 = b.freeman_pos (s.p1());


   switch ( TAB_CONF_CPLE_FREEM[c1][c2])
   {
      case INTER_TOUJ  : return 0.0;
      case CFF_00      :
           return ElMin(b.SquareDist(s.p0(),c1),b.SquareDist(s.p1(),c2));

      case CFF_11      :
      {
           return  s.square_dist_seg(PQ1(b,c1,c2));
      }

      case CFF_13      :
      {
           return  ElMin
                   (
                       s.square_dist_seg(PQ1(b,c1,c2)),
                       s.square_dist_seg(PQ2(b,c1,c2))
                   );
      }
      case CFF_01      :
      {
           return ElMin3
                  (
                     b.SquareDist(s.p0(),c1),
                     b.SquareDist(s.p1(),c2),
                     s.square_dist_seg(PQ1(b,c1,c2))
                  );
     }


      case CFF_02      :
      {
           REAL ord1 = s.ordonnee(PQ1(b,c1,c2));
           if (TAB_TRIGO_CF[c1][c2] )
               return ( (ord1 > 0) ? ElSquare(ord1) : 0.0);
           else
               return ( (ord1 < 0) ? ElSquare(ord1) : 0.0);
      }

      case CFF_03      :
      {
           REAL ord1 = s.ordonnee(PQ1(b,c1,c2));

            return (TAB_TRIGO_CF[c1][c2] == (ord1>0.0)) ?
                   ElSquare(ord1)                       :
                   0.0                                  ;
      }

      case CFF_15      :
      {
           REAL ord1 = s.ordonnee(PQ1(b,c1,c2));
           REAL ord2 = s.ordonnee(PQ2(b,c1,c2));

           return ((ord1 < 0) == (ord2 < 0))             ?
                    ElMin(ElSquare(ord1),ElSquare(ord2)) :
                    0.0                                  ;
      }

      default  : ;
   }

   cout << "CODE " << c1 << "  " << c2  <<  " " << TAB_CONF_CPLE_FREEM[c1][c2] << endl;

   ELISE_ASSERT(false,"Dist Box/seg ??? ");
   return 0;
}





template <class Type>  REAL Box2d<Type>::SquareDist(const SegComp& s) const
{
     return BoxFreemanCompil<Type>::TheBFC.D2BoxSeg(*this,s);
}
template <class Type>  REAL Box2d<Type>::SquareDist(const Seg2d& s) const
{
     return SquareDist(SegComp(s));
}


template <class Type>  bool Box2d<Type>::Intersecte(const class SegComp & seg) const
{
   return BoxFreemanCompil<Type>::TheBFC.D2BoxSeg(*this,seg) < 1e-8;
}

template <class Type>  bool Box2d<Type>::Intersecte(const class Seg2d & seg) const
{
   return Intersecte(SegComp(seg));
}

template <class Type>  bool Box2d<Type>::Include(const SegComp  & s)  const
{
     return Include(s.p0()) && Include(s.p1());
}
template <class Type>  bool Box2d<Type>::Include(const Seg2d  & s)  const
{
     return Include(SegComp(s));
}


template <class Type> Flux_Pts Box2d<Type>::Flux() const
{
   return rectangle(ToPt2di(_p0), ToPt2di(_p1));
}


template <class Type> 
std::vector<Pt2dr> Box2d<Type>::ClipConpMax(const std::vector<Pt2dr> & aCont)
{
    cElPolygone aP1 ;
    Box2dr aB(_p0,_p1);  // Pour forcer le type reel
    aP1.AddContour(aB.Contour(),false);


    cElPolygone aP2 ;
    aP2.AddContour(aCont,false);

    cElPolygone aP3 = aP1 * aP2;
    return  aP3.ContSMax();
}



     //     Box-Box


template <class Type>  REAL8  Box2d<Type>::SquareDist(const Box2d<Type> &b2) const
{
       return 
                 ElSquare (XInterv().dist(b2.XInterv()))
              +  ElSquare (YInterv().dist(b2.YInterv()));
}


template <class Type> 
void Box2d<Type>::PtsDisc(std::vector<Pt2dr> & aV,INT aNbPts)
{
   aV.clear();
   Pt2d<Type> aVCorn[4];
   Corners(aVCorn);

   for (INT aKC=0 ; aKC<4 ; aKC++)
   {
       Pt2dr aC0 = ToPt2dr(aVCorn[aKC]);
       Pt2dr aC1 = ToPt2dr(aVCorn[(aKC+1)%4]);
       // for (INT aKP=0 ; aKP<= aNbPts ; aKP++)
       for (INT aKP=0 ; aKP< aNbPts ; aKP++)  // Modif MPD
       {
            REAL aPds = (aNbPts-aKP) /REAL(aNbPts);
            aV.push_back(barry(aPds,aC0,aC1));
       }
   }
}

template <class Type>
std::vector<Pt2d<Type> >    Box2d<Type>::Contour() const
{
   Pt2d<Type> aP4[4];
   Corners(aP4);
   return std::vector<Pt2d<Type> >(aP4,aP4+4);
}



/*************************************************************/
/*************************************************************/
/**                                                         **/
/**   "Transformation" de box                               **/
/**                                                         **/
/*************************************************************/
/*************************************************************/


template <class Type> bool  InterVide(const Box2d<Type> & b1, const Box2d<Type> & b2)
{
   return
               (b1._p0.x >= b2._p1.x)
          ||   (b1._p0.y >= b2._p1.y)
          ||   (b2._p0.x >= b1._p1.x)
          ||   (b2._p0.y >= b1._p1.y);
}
 
void ModelBoxSubstr::AddBox(Pt2di p0,Pt2di p1)
{
    mBoxes[mNbBox++] = Box2di(p0,p1);
}
 
template class BoxFreemanCompil<INT>;
template class BoxFreemanCompil<REAL>;
template class Box2d<INT>;
template class Box2d<REAL>;
template class Box2d<long double>;
template bool InterVide(const Box2d<INT> & b1, const Box2d<INT> & b2);
template bool InterVide(const Box2d<REAL> & b1, const Box2d<REAL> & b2);

 
/*
 
    Numerotation utilisee des cotes
 
        2
     _______
    |       |
    |       |
  3 |       |  1
    |       |
    |_______|
 
       0
 
*/                                 


ModelBoxSubstr::ModelBoxSubstr(){}

ModelBoxSubstr::ModelBoxSubstr(Box2di ToAdd,Box2di ToSubstr)
{
    MakeIt(ToAdd,ToSubstr);
}


bool ModelBoxSubstr::MakeIt(Box2di A,Box2di S)
{
   mNbBox = 0;
   if  (InterVide(A,S))
   {
        AddBox(A._p0,A._p1);
        return true;
   }
 
   if (A._p0.y < S._p0.y)
   {
       Pt2di p0(ElMax(A._p0.x ,S._p0.x) , A._p0.y);
       Pt2di p1(A._p1.x               , S._p0.y);
       AddBox(p0,p1);
   }
 
 
   if (A._p1.x > S._p1.x)
   {
       Pt2di p0(S._p1.x,ElMax(A._p0.y,S._p0.y));
       Pt2di p1 = A._p1;
       AddBox(p0,p1);
   }
 
   if (A._p1.y > S._p1.y)
   {
      Pt2di p0(A._p0.x,S._p1.y);
      Pt2di p1(ElMin(A._p1.x,S._p1.x),A._p1.y);
      AddBox(p0,p1);
   }
 
   if (A._p0.x < S._p0.x)
   {
      Pt2di p0 =  A._p0;
      Pt2di p1(S._p0.x,ElMin(A._p1.y,S._p1.y));
      AddBox(p0,p1);
   }

   return false;
}
                                     

void ModelDiffBox::MakeIt(Box2di OldOne,Box2di NewOne)
{
    mInterIsEmpty = mEnPlus.MakeIt(NewOne,OldOne);
    mEnMoins.MakeIt(OldOne,NewOne);

    if (!mInterIsEmpty)
    {
           mInter = Box2di(Inf(OldOne._p0,NewOne._p0),Sup(OldOne._p1,NewOne._p1));
    }
}



Box2dr I2R(const Box2di & aB)
{
   return Box2dr(aB._p0,aB._p1);
}
Box2di R2I(const Box2dr & aB)
{
   return Box2di(round_ni(aB._p0),round_ni(aB._p1));
}

Box2di R2ISup(const Box2dr & aB)
{
   return Box2di(round_down(aB._p0),round_up(aB._p1));
}

template<class Type> std::istream & InputStrem (std::istream & ifs,Box2d<Type>  &aBox)
{

   std::vector<Type> aV;
   VElStdRead(ifs,aV,ElGramArgMain::StdGram);

   ELISE_ASSERT(aV.size()==4,"std::istream >> Box2dr  &");

   aBox = Box2d<Type>(Pt2d<Type>(aV[0],aV[1]),Pt2d<Type>(aV[2],aV[3]));

   return ifs;
}

std::istream & operator >> (std::istream & ifs,Box2dr  &aBox)
{
   return InputStrem(ifs,aBox);
}
std::istream & operator >> (std::istream & ifs,Box2di  &aBox)
{
   return InputStrem(ifs,aBox);
}

template<class Type> std::string  BoxToSring(const Box2d<Type>  &aBox)
{
   return    "[" + ToString(aBox._p0.x) + std::string(",")
                 + ToString(aBox._p0.y) + std::string(",")
                 + ToString(aBox._p1.x) + std::string(",")
                 + ToString(aBox._p1.y)
            + "]";
}

template <> std::string ToString<Box2di> (const Box2di & aBox) {return BoxToSring(aBox);}
template <> std::string ToString<Box2dr> (const Box2dr & aBox) {return BoxToSring(aBox);}


Pt2di BoxPClipedIntervC(const Box2di & aB,const Pt2di & aP)
{
   return  Pt2di
           (
               ElMax(aB._p0.x,ElMin(aP.x,aB._p1.x-1)),
               ElMax(aB._p0.y,ElMin(aP.y,aB._p1.y-1))
           );

}

ostream & operator << (ostream & ofs,const Box2di  &aBox)
{
      ofs << "[" << aBox._p0 << ";" << aBox._p1 <<"]";
      return ofs;
}
ostream & operator << (ostream & ofs,const Box2dr  &aBox)
{
      ofs << "[" << aBox._p0 << ";" << aBox._p1 <<"]";
      return ofs;
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
