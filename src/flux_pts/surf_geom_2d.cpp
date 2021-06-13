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


/***************************************************************************/
/*                                                                         */
/*        Rectangles in N dimension                                        */
/*                                                                         */
/***************************************************************************/

class Rect_Computed  : public RLE_Flux_Pts_Computed
{
    public :

       Rect_Computed(const INT * p0,const INT * p1,INT dim);
       virtual const Pack_Of_Pts * next(void);

    private :
        
       INT _nb_pck;
       INT _dim;
       INT _p0[Elise_Std_Max_Dim];
       INT _p1[Elise_Std_Max_Dim];
       INT _p_cur[Elise_Std_Max_Dim];

       virtual  bool   is_rect_2d(Box2di & box)
       {
               if (_dim != 2)
                  return false;

               box._p0 = Pt2di(_p0[0],_p0[1]);
               box._p1 = Pt2di(_p1[0],_p1[1]);
               return true;
       }

       REAL   average_dist()
       {
              return 
                    (_dim == 2)                    ?
                    1                              :
                    Flux_Pts_Computed::average_dist();
       }
};

Rect_Computed::Rect_Computed(const INT * p0,const INT * p1,INT dim) :
             RLE_Flux_Pts_Computed(dim,p1[0]-p0[0]),
             _dim (dim)
{
      ::convert(_p0,p0,dim);
      ::convert(_p1,p1,dim);
      ::convert(_p_cur,p0,dim);
      _rle_pack->set_nb(p1[0]-p0[0]);

     _nb_pck = 1;
      for(int d =1; d<dim ; d++)
          _nb_pck *= p1[d]-p0[d];
}


const Pack_Of_Pts * Rect_Computed::next(void)
{
    if (!(_nb_pck--))
       return 0;

   _rle_pack->set_pt0(_p_cur);

   int d;
   for(d=1 ; _p_cur[d]==_p1[d]-1 ; d++)
      _p_cur[d] = _p0[d];

    _p_cur[d]++;

   return _rle_pack;
}


       //=====================
       //    Not computed
       //=====================

class Rect_Non_Comp : public Flux_Pts_Not_Comp
{
   public :

     Rect_Non_Comp(const INT *p0,const INT *p1,INT dim);
     virtual Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp &);

   private :
       INT _dim;
       INT _p0[Elise_Std_Max_Dim];
       INT _p1[Elise_Std_Max_Dim];
};

Flux_Pts_Computed * Rect_Non_Comp::compute(const Arg_Flux_Pts_Comp & arg)
{
    return split_to_max_buf(new Rect_Computed(_p0,_p1,_dim),arg);
}


Flux_Pts_Computed * RLE_Flux_Pts_Computed::rect_2d_interface
                       (
                           Pt2di p0,
                           Pt2di p1,
                           int sz_buf
                      )
{
    INT t0[2];
    INT t1[2];

    p0.to_tab(t0);
    p1.to_tab(t1);

    return
       split_to_max_buf
       (
            new Rect_Computed(t0,t1,2),
            Arg_Flux_Pts_Comp(sz_buf)
       );

}

Rect_Non_Comp::Rect_Non_Comp(const INT *p0,const INT *p1,INT dim) :
    _dim (dim)
{
     ASSERT_TJS_USER
     (
          dim <Elise_Std_Max_Dim,
          "cannot handle dimension > to Elise_Std_Max_Dim"
     );
     OpMin.t0_eg_t1_op_t2(_p0,p0,p1,dim);
     OpMax.t0_eg_t1_op_t2(_p1,p0,p1,dim);
}

Flux_Pts rectangle(const Box2di & box)
{
    return rectangle(box._p0,box._p1);
}

Flux_Pts rectangle(Pt2di p0,Pt2di p1) 
{
  INT t0[2],t1[2];
  p0.to_tab(t0);
  p1.to_tab(t1);
  return Flux_Pts (new Rect_Non_Comp(t0,t1,2));
}

Flux_Pts rectangle(const INT * p0,const INT * p1,INT dim)
{
  return Flux_Pts (new Rect_Non_Comp(p0,p1,dim));
}

Flux_Pts rectangle(INT x0,INT x1)
{
  return Flux_Pts (new Rect_Non_Comp(&x0,&x1,1));
}

Flux_Pts rectangle(Pt3di p0,Pt3di p1) 
{
  INT t0[3],t1[3];
  p0.to_tab(t0);
  p1.to_tab(t1);
  return Flux_Pts (new Rect_Non_Comp(t0,t1,3));
}



/***************************************************************************/
/*                                                                         */
/*        Border_Rect in dimension K                                       */
/*                                                                         */
/***************************************************************************/

/* Let point_in be a point, this function store in point_out
   the next point  (according to  lexicographic order) included in  rectangle
   bounded by corner_1 and corner_2, return 0 if this point is
   outside the rectangle
*/

bool next_point_inside_rectangle (        INT *  point_out
                                   ,const INT *  point_in
                                   ,const INT *  corner_1
                                   ,const INT *  corner_2
                                   ,      INT    dim)
{
   INT dim_a_incr; /* dimension a incrementer */
   INT j;

   dim_a_incr = 0;

  while ( (dim_a_incr < dim) && (point_in[dim_a_incr] >= corner_2[dim_a_incr] -1))
  {
      point_out[dim_a_incr] = corner_1[dim_a_incr];
      dim_a_incr++;
  }
  if (dim_a_incr == dim)
     return(false);

  point_out[dim_a_incr] = point_in[dim_a_incr] + 1;
  for (j = dim_a_incr+1;  j < dim ; j++)
      point_out[j] = point_in[j];

  return(true);
}


bool point_inside_rectangle (  const INT * point
                              ,const INT * corner_1
                              ,const INT * corner_2
                              ,      INT   dim)
{
    INT i;

     for (i = 0 ; i < dim ; i++)
        if ( (point[i] < corner_1[i]) || (point[i] >= corner_2[i]))
           return(false);

    return(true);
}


class Border_Rect_Comp : public Std_Flux_Of_Points<INT>
{
    public :
         virtual const Pack_Of_Pts * next(void);
         Border_Rect_Comp
         (const  Arg_Flux_Pts_Comp &,
          const INT *p1,const INT *p2,const INT *q1,const INT *q2,INT dim);


     private :

         bool _continue;
         INT  _p_cur[Elise_Std_Max_Dim];

         INT  _p1[Elise_Std_Max_Dim];
         INT  _p2[Elise_Std_Max_Dim];

         INT  _q1[Elise_Std_Max_Dim];
         INT  _q2[Elise_Std_Max_Dim];

};


const Pack_Of_Pts * Border_Rect_Comp::next()
{
    if (! _continue)
      return(0);

   INT d = dim();
   _pack->set_nb(0);

   while  (     (_pack->not_full())
            &&  (   _continue 
                  = next_point_inside_rectangle(_p_cur,_p_cur,_p1,_p2,d)
                )
          )
    {
       if ( point_inside_rectangle(_p_cur,_q1,_q2,d))
          _p_cur[0] = _q2[0];
        _pack->push(_p_cur);
    }
    return _pack;
}


Border_Rect_Comp::Border_Rect_Comp
            (
                 const  Arg_Flux_Pts_Comp & arg,
                 const  INT * p1,
                 const  INT * p2,
                 const  INT * q1,
                 const  INT * q2,
                        INT   dim
            ) :
      Std_Flux_Of_Points<int>(dim,arg.sz_buf())
{
     _continue = true;
      ::convert(_p_cur,p1,dim);
      _p_cur[0]--;  // set to point just before entering the rectangle

      ::convert(_p1,p1,dim);
      ::convert(_p2,p2,dim);
      ::convert(_q1,q1,dim);
      ::convert(_q2,q2,dim);
}


class Border_Rect_Not_Comp  : public Flux_Pts_Not_Comp
{
    public :
         virtual Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp &);

         Border_Rect_Not_Comp
         (const INT *p1,const INT *p2,const INT *q1,const INT *q2,INT dim);

     private :

         INT _dim;
         INT  _p1[Elise_Std_Max_Dim];
         INT  _p2[Elise_Std_Max_Dim];

         INT  _q1[Elise_Std_Max_Dim];
         INT  _q2[Elise_Std_Max_Dim];
};


Border_Rect_Not_Comp::Border_Rect_Not_Comp
            (
                 const  INT * p1,
                 const  INT * p2,
                 const  INT * q1,
                 const  INT * q2,
                        INT   dim
            ) 
{
     _dim = dim;

     OpMin.t0_eg_t1_op_t2(_p1,p1,p2,dim);
     OpMax.t0_eg_t1_op_t2(_p2,p1,p2,dim);

     for (int j = 0; j < dim ; j++)
     {
         ASSERT_TJS_USER
         (
             (q1[j]&&q2[j]),
             "BORDER nul in border_rect"
         );
         _q1[j] = _p1[j] + ElAbs(q1[j]);
         _q2[j] = _p2[j] - ElAbs(q2[j]);
     }
     OpMin.t0_eg_t1_op_t2(_q1,_q1,_p2,dim);
     OpMax.t0_eg_t1_op_t2(_q2,_q2,_q1,dim);
}




Flux_Pts_Computed * Border_Rect_Not_Comp::compute(const Arg_Flux_Pts_Comp & arg)
{
     return new  Border_Rect_Comp (arg,_p1,_p2,_q1,_q2,_dim);
}


Flux_Pts border_rect(  const  INT * p1,
                       const  INT * p2,
                       const  INT * q1,
                       const  INT * q2,
                       INT     dim
                     )
{
    return new Border_Rect_Not_Comp(p1,p2,q1,q2,dim);
}

Flux_Pts border_rect(  const  INT * p1,
                       const  INT * p2,
                              INT   b,
                              INT   dim
                     )
{
    INT q[Elise_Std_Max_Dim];

    set_cste(q,b,dim);
    return new Border_Rect_Not_Comp(p1,p2,q,q,dim);
}


Flux_Pts border_rect(Pt2di p1,Pt2di p2,Pt2di q1,Pt2di q2)
{
     INT P1[2],P2[2],Q1[2],Q2[2];

     p1.to_tab(P1);
     p2.to_tab(P2);
     q1.to_tab(Q1);
     q2.to_tab(Q2);

     return border_rect(P1,P2,Q1,Q2,2);
}

Flux_Pts border_rect(Pt2di p1,Pt2di p2,INT sz)
{
    return border_rect(p1,p2,Pt2di(sz,sz),Pt2di(sz,sz));
}


Flux_Pts border_rect(INT p1,INT p2,INT b1,INT b2)
{
    return border_rect(&p1,&p2,&b1,&b2,1);
}

Flux_Pts border_rect(INT p1,INT p2,INT b)
{
    return border_rect(p1,p2,b,b);
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
