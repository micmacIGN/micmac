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



#ifndef _ELISE_IM_ALGO_DIST32
#define _ELISE_IM_ALGO_DIST32

template  <class Type,class Type_Base>
          void TIm2D<Type,Type_Base>::algo_dist_env_Klisp_Sup(int aD4,int aD8)
{
    INT x,y;
    for ( y =1 ; y<_ty-1 ; y++)
        for ( x =1 ; x<_tx-1 ; x++)
        {
              ElSetMax(_d[y][x],(_d[y  ][x-1]-aD4));
              ElSetMax(_d[y][x],(_d[y-1][x  ]-aD4));
              ElSetMax(_d[y][x],(_d[y-1][x-1]-aD8));
              ElSetMax(_d[y][x],(_d[y-1][x+1]-aD8));
        }

    for ( y =_ty-2 ; y>= 1 ; y--)
        for ( x =_tx-2 ; x>= 1 ; x--)
        {
              ElSetMax(_d[y][x],(_d[y  ][x+1]-aD4));
              ElSetMax(_d[y][x],(_d[y+1][x  ]-aD4));
              ElSetMax(_d[y][x],(_d[y+1][x-1]-aD8));
              ElSetMax(_d[y][x],(_d[y+1][x+1]-aD8));
        }
}


template  <class Type,class Type_Base>
          void TIm2D<Type,Type_Base>::algo_dist_env_Klisp32_Sup()
{
      algo_dist_env_Klisp_Sup(2,3);
}



template  <class Type,class Type_Base> class Border_algo_dist_32_neg
{
    private :
       TIm2D<Type,Type_Base>  & _im;
       Pt2di                    _p0;
       Pt2di                    _p1;

       INT safe_get(Pt2di p)
       {
           return p.in_box(_p0,_p1) ?
                  _im.get(p)        :
                  -128              ;
       }

    public :
       
       typedef INT OutputFonc;

       INT get(Pt2di p)
       {
           INT res = safe_get(p);
           for (INT k=0; k <8 ; k++)
               ElSetMax(res,safe_get(p+TAB_8_NEIGH[k])-2-k%2);
           return res;
       }

       Border_algo_dist_32_neg(TIm2D<Type,Type_Base> & im,Pt2di p0,Pt2di p1) :
            _im (im),
            _p0 (p0),
            _p1 (p1)
       {
       }
     
};


template  <class Type,class Type_Base>
          void TIm2D<Type,Type_Base>::algo_dist_32_neg(Pt2di p0,Pt2di p1)
{

    pt_set_min_max(p0,p1);
    p0.SetSup(Pt2di(0,0));
    p1.SetInf(sz());


   TElCopy
   (
        TFlux_BordRect2d(p0,p1),
        TCste(0),
        *this
   );

    INT x,y;
    Type  *lm1,*l,*lp1;

    INT NBX = p1.x-1 -(p0.x+1);

    for (y = p0.y ; y<p1.y ; y++)
    {
        l = _d[y] + p0.x;
        for (x =p0.x ; x<p1.x ; x++,l++)
            *l = (*l ? 127 : -1);
    }

    for (y =p0.y +1; y<p1.y-1 ; y++)
    {
        l   = _d[y] + p0.x+1;
        lm1 = _d[y-1] + p0.x+1;
        INT  nb = NBX;
        for (; nb ; nb--,l++,lm1++)
            if (*l >0)
            {
                ElSetMin(*l,l[-1]    +2);
                ElSetMin(*l,lm1[0]   +2);
                ElSetMin(*l,lm1[-1]  +3);
                ElSetMin(*l,lm1[1]   +3);
            }
    }

    for (y =p1.y-2 ; y> p0.y ; y--)
    {
        l   = _d[y] + p1.x-2;
        lp1 = _d[y+1] + p1.x-2;
        INT  nb = NBX;
        for (;nb;nb--,l--,lp1--)
            if ( *l >0)
            {
                  ElSetMin(*l,l[1]    +2);
                  ElSetMin(*l,lp1[0]  +2);
                  ElSetMin(*l,lp1[-1] +3);
                  ElSetMin(*l,lp1[1]  +3);
            }
    }
/*
    J'arrete ici les optimisation de "bas niveau" , apparament ca
   a l'air plus lent une fois "opimise" qu'avant !! ??
*/

    for (y =p0.y ; y<p1.y ; y++)
        for (x =p0.x ; x<p1.x ; x++)
            if ( _d[y][x] <0)
              _d[y][x] = -128;


    for (y =p0.y +1; y<p1.y-1 ; y++)
        for (x =p0.x+1 ; x<p1.x-1 ; x++)
           if (_d[y][x]<0)
           {
              ElSetMax(_d[y][x],(_d[y  ][x-1]-2));
              ElSetMax(_d[y][x],(_d[y-1][x  ]-2));
              ElSetMax(_d[y][x],(_d[y-1][x-1]-3));
              ElSetMax(_d[y][x],(_d[y-1][x+1]-3));
           }

    for (y =p1.y-2 ; y> p0.y ; y--)
        for (x =p1.x-2 ; x> p0.x ; x--)
           if (_d[y][x]<0)
           {
              ElSetMax(_d[y][x],(_d[y  ][x+1]-2));
              ElSetMax(_d[y][x],(_d[y+1][x  ]-2));
              ElSetMax(_d[y][x],(_d[y+1][x-1]-3));
              ElSetMax(_d[y][x],(_d[y+1][x+1]-3));
           }

   border_algo_dist_32_neg(p0,p1);
}


template  <class Type,class Type_Base>
          void TIm2D<Type,Type_Base>::border_algo_dist_32_neg(Pt2di p0,Pt2di p1)
{
   TElCopy
   (
        TFlux_BordRect2d(p0,p1),
        Border_algo_dist_32_neg<Type,Type_Base>(*this,p0,p1),
        *this
   );
}

template  <class Type,class Type_Base>
          void TIm2D<Type,Type_Base>::algo_dist_32_neg()
{
    algo_dist_32_neg(Pt2di(0,0),sz());
}


template  <class Type,class Type_Base>
          void TIm2D<Type,Type_Base>::border_algo_dist_32_neg()
{
    border_algo_dist_32_neg(Pt2di(0,0),sz());
}



template  <class Type,class Type_Base> 
   int MaxJump
       (
           Im2D<Type,Type_Base>   aZMin,
           Im2D<Type,Type_Base>   aZMax
       )
{
    TIm2D<Type,Type_Base> aTZMin(aZMin);
    TIm2D<Type,Type_Base> aTZMax(aZMax);

    Pt2di aSz = aZMin.sz();
    ELISE_ASSERT(aZMax.sz()==aSz,"Dif size in MaxJump");
    Pt2di aP ; 

    int aRes = 0;
    for (aP.x=0 ; aP.x<aSz.x; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSz.y; aP.y++)
        {
            int aZ0 = aTZMin.get(aP);
            int aZ1 = aTZMax.get(aP)-1;
            for (int aK=0 ; aK<4 ; aK++)
            {
                 Pt2di aPV = aP + TAB_8_NEIGH[aK];
                 int azV0 = aTZMin.getproj(aPV);
                 int azV1 = aTZMax.getproj(aPV)-1;
                 ElSetMax(aRes,ElAbs(azV1-aZ0));
                 ElSetMax(aRes,ElAbs(aZ1-azV0));
            }
        }
    }
    return aRes;
}



/*

template  <class Type,class Type_Base>
void  ComplKLipsParLBas
      (
           TIm2DBits<1>      aTMaskValInit,
           TIm2DBits<1>      aTMaskValFinale,
           TIm2D<Type,Type_Base>   aTZ
      )
{
   //Show(aTZ,aTMaskValFinale,"Test1.tif");

   Pt2di aSz(aTMaskValInit.sz());
   std::vector<Pt2di> aVPts;
   Im2D<INT2,INT>  aZHerit(aSz.x,aSz.y,0);
   aZHerit.dup(aTZ._the_im);
   TIm2D<INT2,INT> aTZHerit(aZHerit);

   Pt2di aP;
   for (aP.y =1 ; aP.y<(aSz.y-1) ; aP.y++)
   {
      for (aP.x =1 ; aP.x<(aSz.x-1) ; aP.x++)
      {
           if ((! aTMaskValFinale.get(aP)) && (aTMaskValInit.get(aP)))
           {
               aTZ.oset(aP,El_CTypeTraits<Type>::MaxValue());
               aVPts.push_back(aP);
           }
      }
   }

   int aNbPts = aVPts.size();

   for (int aKIter=0 ; aKIter< 6 ; aKIter++)
   {
       int aNbUpdate = 0;
       bool Pair= ((aKIter%2)==0);

       int IndDeb = Pair ? 0       : (aNbPts-1);
       int IndOut = Pair ? aNbPts  : (-1)      ;
       int Incr   = Pair ? 1       : (-1)      ;

       for (int Ind=IndDeb ; Ind!=IndOut ; Ind+=Incr)
       {
            Pt2di aP2Cur = aVPts[Ind];
            int aVMin = aTZ.get(aP2Cur);
            for (int aKV = 0 ; aKV<8 ; aKV++)
            {
                Pt2di aPVois = aP2Cur + TAB_8_NEIGH[aKV];
                double aZAugm = aTZ.get(aPVois) + ((aKV%2) ? 3 : 2);
                if ((aZAugm < aVMin) && (aTMaskValInit.get(aPVois)))
                {
                    aVMin = (int)aZAugm;
                    aTZHerit.oset(aP2Cur,aTZHerit.get(aPVois));
                    aNbUpdate++;
                }
            }
            aTZ.oset(aP2Cur,aVMin);
       }
       // std::cout << "NNNbUodta " << aNbUpdate << "\n";
   }
   for (int Ind=0 ; Ind<aNbPts ; Ind++)
   {
      Pt2di aP2Cur = aVPts[Ind];
      aTZ.oset(aP2Cur,aTZHerit.get(aP2Cur));
   }

   //Show(aTZ,aTMaskValFinale,"Test2.tif");
   // std::cout << "Ennndxcfttt \n"; getchar();

}
*/

/*
template  <class Type,class Type_Base>
void  ComplKLipsParLBas
      (
           TIm2DBits<1>      aTMaskValInit,
           TIm2DBits<1>      aTMaskValFinale,
           TIm2D<Type,Type_Base>   aTZ
      )
{
}
*/

#endif  //  _ELISE_IM_ALGO_DIST32











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
