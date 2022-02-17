/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/



#include "StdAfx.h"
#include "bench.h"

#include <algorithm> 


template <class TPt> TPt RandPt(TPt SZ)
{
     return TPt
            (
                (ElTyName TPt::TypeScal) (NRrandom3()*SZ.x),
                (ElTyName TPt::TypeScal) (NRrandom3()*SZ.y)
            );
}
        //  env_min

void verif_env_conv
     (
          ElFifo<INT> &          ind,
          const ElFilo<Pt2di > & pts,
          bool                   env_min       
     )
{

      if (ind.nb() > 1)
          for (INT I=0; I<ind.nb() ; I++)
              BENCH_ASSERT( pts[ind[I]] !=  pts[ind[I+1]]);

      for (INT k=0; k<ind.nb(); k++)
      {
          Pt2di p1 = pts[ind[k]];
          Pt2di p2 = pts[ind[k+1]];
          for (INT I=0; I<pts.nb() ; I++)
          {
              BENCH_ASSERT(((p2-p1)^(pts[I]-p1)) >=0); 
          }
      }

      if (env_min)
      {
          if (ind.nb() > 2)
             for (INT k=0; k<ind.nb(); k++)
             {
                 Pt2di p1 = pts[ind[k]];
                 Pt2di p2 = pts[ind[k+1]];
                 Pt2di p3 = pts[ind[k+2]];
                 BENCH_ASSERT (((p2-p1)^ (p3-p2)) >0);
             }
      }
      else
      {
          for (INT kp=0 ; kp<pts.nb() ; kp++)
          {
              Pt2di in = pts[ind[kp]];
              bool got = false;
              for (int ks=0; ks<ind.nb();ks++)
                  if (pts[ind[ks]] == in)
                     got = true;

              if (! got)
                 for (int ks=0; ks<ind.nb();ks++)
                 {
                      Pt2di p1 = pts[ind[ks]];
                      Pt2di p2 = pts[ind[ks+1]];
                      BENCH_ASSERT( ((p1-in)^(p2-in)) != 0);
                 }
          }
      }
}

template <class TPt>  
     void bench_env_conv
     (
          Pt2di      SZ,
          INT        nb,
          TPt *
     )
{
      ElFilo<TPt>  pts;

      for (INT k=0; k<nb ; k++)
          pts.pushlast(RandPt(SZ));


      ElFifo<INT> ind;
      env_conv(ind,pts,true);
      verif_env_conv(ind,pts,true);

      env_conv(ind,pts,false);
      verif_env_conv(ind,pts,false);

}

class cmpi
{
    public :
      bool operator() (INT i1, INT i2) {return i1< i2;}
};

void bench_env_conv()
{

    // Pt2di SZ(500,500);

    for (INT K = 0 ; K < 2000; K++)
    {
        INT tx = (INT) (50 * NRrandom3());
        INT nb = (INT) (NRrandom3() * tx *3);
        bench_env_conv(Pt2di(tx,tx),nb,(Pt2di *)0); 
    }
    printf("OK BENCH ENV-CONV \n");
}





