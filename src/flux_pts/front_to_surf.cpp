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



class QFPV_RLE
{
     public :
           QFPV_RLE(INT aX0,INT aNB,INT anY) :
                 mX0 ( aX0),
                 mNB ( aNB),
                 mY  ( anY)
           {
           }
           bool IsEmpty()
           {
               return mNB == 0;
           }
           void init_pack(RLE_Pack_Of_Pts &aPack)
           {
             aPack.set_pt0(Pt2di(mX0,mY));
             aPack.set_nb(mNB);
           }
           
     private :
           INT mX0;
           INT mNB;
           INT mY;
};



class QFPC_LineH
{
     public :
          void AddAbscisse(INT anX) { mVX.push_back(anX); }
          void Finish();
          INT  NbInterv() { return  (INT) (mVX.size()/2); }
          QFPV_RLE   KthInterv(INT aK,INT anY)
          {
               aK *=2;
               INT anX0 = mVX[aK];
               return QFPV_RLE(anX0,mVX[aK+1]-anX0,anY);
          }
     private :

          std::vector<INT> mVX;
};
void QFPC_LineH::Finish()
{
    std::sort(mVX.begin(),mVX.end());
}

class  Quick_Flux_Poly_Comp :  public RLE_Flux_Pts_Computed
{
      public :
           Quick_Flux_Poly_Comp(const std::vector<Pt2di> & ,Box2di aBox);

      private :

            std::vector<QFPV_RLE>     mRLE;
            virtual const Pack_Of_Pts * next(void);
};



Quick_Flux_Poly_Comp::Quick_Flux_Poly_Comp(const std::vector<Pt2di> & Pts,Box2di aBox) :
   RLE_Flux_Pts_Computed(2,aBox._p1.x-aBox._p0.x)
{
     INT NbPts =  (INT) Pts.size();
     INT YMin = aBox._p0.y;
     INT YMax = aBox._p1.y;
     INT NbY = YMax-YMin;


     // On initialise LinesH
     std::vector<QFPC_LineH>   LinesH;
     LinesH.reserve(NbY);
     QFPC_LineH aLine;
     for (INT anY=0 ; anY<NbY ; anY++)
     {
         LinesH.push_back(aLine);
     }

     for (INT k=0 ; k<NbPts ; k++)
     {
         Pt2di p0 = Pts[k];
         Pt2di p1 = Pts[(k+1)%NbPts];
         if(p0.y != p1.y)
         {
             Pt2dr aPR0 = Pt2dr(p0);
             Pt2dr aPR1 = Pt2dr(p1);
             Seg2d aSeg(aPR0,aPR1);
             INT y0 = ElMin(p0.y ,p1.y);
             INT y1 = ElMax(p0.y ,p1.y);
             
             for (INT y =y0 ; y<y1 ; y++)
             {
                 INT x = round_ni(aSeg.AbsiceInterDroiteHoriz(y+0.5));
                 LinesH[y-YMin].AddAbscisse(x);
             }
         }
     }

     INT NbInterv = 0;
     for (INT y=0 ; y<NbY ; y++)
     {
          LinesH[y].Finish();
          NbInterv += LinesH[y].NbInterv();
     }
     mRLE.reserve(NbInterv);

	 {
     for (INT y=0 ; y<NbY ; y++)
     {
          INT NbInt = LinesH[y].NbInterv();
          for (INT k=0 ; k<NbInt ; k++)
          {
              QFPV_RLE aRLE = LinesH[y].KthInterv(k,y+YMin);
              if (! aRLE.IsEmpty())
                 mRLE.push_back(aRLE);
          }
     }
	 }
}

const Pack_Of_Pts * Quick_Flux_Poly_Comp::next()
{
     if (mRLE.empty())
        return 0;
     mRLE.back().init_pack(*_rle_pack);
     mRLE.pop_back();
     return _rle_pack;
}


class Quick_Flux_Poly_NoComp  : public Flux_Pts_Not_Comp
{
      
      public :

           Quick_Flux_Poly_NoComp(ElList<Pt2di> aLPts) :
                 mLPts (aLPts)
           {
           }

      private :

          Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp & arg);

           ElList<Pt2di> mLPts;

};


Flux_Pts_Computed * Quick_Flux_Poly_NoComp::compute(const Arg_Flux_Pts_Comp & arg)
{

     ELISE_ASSERT(!mLPts.empty(),"Empty liste in quick poly");
     Pt2di aP0 = mLPts.car();
     Pt2di aP1 = mLPts.car();
     std::vector<Pt2di> aVpts;

     for (ElList<Pt2di> aL= mLPts ;!aL.empty(); aL = aL.cdr())
     {
         Pt2di aP = aL.car();
         aVpts.push_back(aP);
         aP0.SetInf(aP);
         aP1.SetSup(aP);
     }

     Flux_Pts_Computed *  res = new Quick_Flux_Poly_Comp(aVpts,Box2di(aP0,aP1));
     return split_to_max_buf(res,arg);
}

Flux_Pts quick_poly(ElList<Pt2di> aLPts)
{
   return new Quick_Flux_Poly_NoComp(aLPts);
}

/*********************************************************************/
/*                                                                   */
/*         Flux_To_Surf_Computed                                     */
/*                                                                   */
/*********************************************************************/




class Flux_To_Surf_Computed  : public RLE_Flux_Pts_Computed
{
    public :

       virtual ~Flux_To_Surf_Computed()
       {
            DELETE_VECTOR(_l_cur,_b._p0.x);
            DELETE_VECTOR(_l    ,_b._p0.y);
            DELETE_VECTOR(_res_x,0);
            DELETE_VECTOR(_res_evt,0);
       }


       class Arg
       {
            public :

                Arg(Flux_Pts flx) :
                    b (Pt2di(0,0),Pt2di(0,0))
                {
                    pts =  Std_Pack_Of_Pts<INT>::new_pck(2,1000);
                    ELISE_COPY
                    (
                         flx,
                         Virgule(FX,FY),
                         push_values(&pts)
                     );

                    b = Box2di(pts->_pts[0],pts->_pts[1],pts->nb());
                    b._p1 =  b._p1 + Pt2di(1,1);
                }

                Box2di b;
                Std_Pack_Of_Pts<INT> * pts;
       };

       

       class Line
       {
            public :
             INT  *  xs;
             bool *  evts;
             INT   nb;
             INT   nb_ev;
       };

       Flux_To_Surf_Computed(Arg arg,bool front);

    private :

       virtual const Pack_Of_Pts * next(void);
       void new_line(INT);


       Box2di  _b;
       bool    _last;

       Line *    _l;
       INT  *    _res_x;
       bool  *   _res_evt;
       INT  *    _l_cur;
       INT       _x_min;
       INT       _x_max;
       INT       _x0;
       INT       _x1;
       INT       _y_cur;

       INT       _with_front;
       
};



void Flux_To_Surf_Computed::new_line(INT y)
{
     _y_cur = y;
     Line * l = _l+_y_cur;
     INT  * tx  = l->xs;

     _x_min = OpMin.red_tab(tx,l->nb);
     _x_max = OpMax.red_tab(tx,l->nb);

     for (INT k = 0; k <l->nb ; k++)
         if (l->evts[k])
            _l_cur[tx[k]] ++;

     for (INT x=_x_min+1; x<=_x_max ; x++)
         _l_cur[x] = (_l_cur[x]+_l_cur[x-1])%2;

     _l_cur[_x_max+1] = 0;

	 {
		 for (INT k = 0; k <l->nb ; k++)
		{
			 _l_cur[tx[k]] = _with_front;
		}
	 }



      while ((_x_min  <= _x_max) && (! _l_cur[_x_min ])) _x_min++;
      while ((_x_min  <= _x_max) && (! _l_cur[_x_max ])) _x_max--;

     _x0 = _x_min;
     _x1 = _x0 ;

      while ((_x1 <= _x_max) && (_l_cur[_x1] ))
            _x1++;

}

const Pack_Of_Pts * Flux_To_Surf_Computed::next(void)
{
     if (_last)
        return 0;

     _rle_pack->set_pt0(Pt2di(_x0,_y_cur));
     _rle_pack->set_nb(_x1-_x0);
     
     if (_x1 == _x_max +1)
     {
         if (_y_cur == _b._p1.y-1)
            _last = true;
         else
         {
             for (INT x=_x_min ; x<=_x_max ; x++)
                 _l_cur[x]  = 0;
             new_line(_y_cur+1);
         }
     }
     else
     {
         while (! _l_cur[_x1]) _x1++;
         _x0 = _x1;
         while (_l_cur[_x1]) _x1++;
     }

     return _rle_pack;
}



Flux_To_Surf_Computed::Flux_To_Surf_Computed
(
     Arg  arg,
     bool front
)  :
   RLE_Flux_Pts_Computed(2,arg.b._p1.x-arg.b._p0.x),
   _b          (arg.b),
   _last       (false),
   _with_front (front?1:0)
{
   _l_cur =  NEW_VECTEUR(_b._p0.x,_b._p1.x+3,INT);
   set_cste(_l_cur+_b._p0.x,0,_b._p1.x+3-_b._p0.x);
   _l = NEW_VECTEUR(_b._p0.y,_b._p1.y,Line);

   INT   * y = arg.pts->_pts[1];
   INT   * x = arg.pts->_pts[0];
   INT   nb = arg.pts->nb();


   {
        // compute nb pts by line 

        for (INT ky=_b._p0.y ; ky<_b._p1.y ; ky++)
        {
            _l[ky].nb = 0;
            _l[ky].nb_ev = 0;
        }
        for (INT k=0 ; k<nb ; k++)
            _l[y[k]].nb++;
   }

   {
        // compute (end of) adresse to put x values;

        _res_x   = NEW_VECTEUR(0,nb,INT);
        _res_evt   = NEW_VECTEUR(0,nb,bool);
        _l[_b._p0.y].xs = _res_x +  _l[_b._p0.y].nb;
        _l[_b._p0.y].evts = _res_evt +  _l[_b._p0.y].nb;

        for (INT ky=_b._p0.y +1 ; ky<_b._p1.y ; ky++)
        {
            _l[ky].xs   = _l[ky-1].xs   + _l[ky].nb;
            _l[ky].evts = _l[ky-1].evts + _l[ky].nb;
        }
   }

   //  put x values ("begin by end")
   for (INT k=0 ; k<nb ; k++)
   {
        Line * l = _l + y[k];
        l->xs --;
        l->evts --;
        *(l->xs)  = x[k];
        *(l->evts)  = 
                (
                       ( y[mod(k-1,nb)] < y[k])
                   !=  ( y[mod(k+1,nb)] < y[k])
                 );
        if (*(l->evts)) 
           l->nb_ev ++;
   }
   delete arg.pts;
   new_line(_b._p0.y);
}


/*********************************************************************/
/*                                                                   */
/*         Flux_To_Surf_Not_Comp                                     */
/*                                                                   */
/*********************************************************************/


class Flux_To_Surf_Not_Comp  : public Flux_Pts_Not_Comp
{
      
      public :
           Flux_To_Surf_Not_Comp(Flux_Pts,bool front);

      private :

          Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp & arg)
          {
               Flux_Pts_Computed *  res = 
                      new  Flux_To_Surf_Computed
                          (Flux_To_Surf_Computed::Arg(_flx),_front);

                return split_to_max_buf(res,arg);
          }




          Flux_Pts _flx;
          bool     _front;
};


Flux_To_Surf_Not_Comp::Flux_To_Surf_Not_Comp(Flux_Pts flx,bool front) :
       _flx   (flx),
       _front (front)
{
}


/*********************************************************************/
/*                                                                   */
/*         Interface functions                                       */
/*                                                                   */
/*********************************************************************/


Flux_Pts disc(Pt2dr c,REAL r,bool front)
{
      return new Flux_To_Surf_Not_Comp (circle(c,r),front) ;
}

Flux_Pts ell_fill(Pt2dr c,REAL A,REAL B,REAL teta,bool front)
{
      return new Flux_To_Surf_Not_Comp(ellipse(c,A,B,teta),front) ;
}

Flux_Pts polygone(ElList<Pt2di> l,bool front)
{
      return new Flux_To_Surf_Not_Comp(line_for_poly(l),front);
}


Flux_Pts sector_ang(Pt2dr c,REAL r,REAL a0,REAL a1,bool front)
{
     return new Flux_To_Surf_Not_Comp(fr_sector_ang(c,r,a0,a1),front);
}

Flux_Pts chord_ang(Pt2dr c,REAL r,REAL a0,REAL a1,bool front)
{
     return new Flux_To_Surf_Not_Comp(fr_chord_ang(c,r,a0,a1),front);
}


Flux_Pts sector_ell(Pt2dr c,REAL A,REAL B,REAL teta,REAL a0,REAL a1,bool front) 
{
         return new Flux_To_Surf_Not_Comp (fr_sector_ell(c,A,B,teta,a0,a1),front);

}

Flux_Pts chord_ell(Pt2dr c,REAL A,REAL B,REAL teta,REAL a0,REAL a1,bool front) 
{
      return new Flux_To_Surf_Not_Comp(fr_chord_ell(c,A,B,teta,a0,a1),front);
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
