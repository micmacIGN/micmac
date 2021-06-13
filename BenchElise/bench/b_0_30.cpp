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



void bench_im3d()
{
     {
        Im3D<U_INT1,INT> i3(2,3,4);

        BENCH_ASSERT
        (
               (i3.tx() == 2)
            && (i3.ty() == 3)
            && (i3.tz() == 4)
        );
    }

    {
        Im3D<INT2,INT> i3(10,20,30);
        INT s1,Mx,My,Mz;
        ELISE_COPY
        (
             i3.all_pts(),
             Virgule(1,FX,FY,FZ),
             Virgule(sigma(s1),VMax(Mx),VMax(My),VMax(Mz))
        );
        BENCH_ASSERT((s1==6000)&&(Mx==9)&&(My==19)&&(Mz==29));


        INT Sint;
        ELISE_COPY(i3.interior(1),1,sigma(Sint));
        BENCH_ASSERT(Sint == (8*18*28));

        INT Sbord;
        ELISE_COPY(i3.border(1),1,sigma(Sbord));
        BENCH_ASSERT(Sint+Sbord == 6000);


        ELISE_COPY(i3.all_pts(),FX+FY-FZ,i3.out());

        INT2 *** d = i3.data();
        for (INT z=0; z<30 ; z++)
            for (INT y=0; y<20 ; y++)
                 for (INT x=0; x<10 ; x++)
                     BENCH_ASSERT(d[z][y][x] == x+y-z);

        INT dif;
        ELISE_COPY(i3.all_pts(),Abs(FX+FY-FZ-i3.in()),sigma(dif));
        BENCH_ASSERT(dif==0);

        ELISE_COPY
        (
           select(i3.all_pts(),(FX+FY+FZ)%2),
           Abs(FX+FY-FZ-i3.in()),
           sigma(dif)
        );
        BENCH_ASSERT(dif==0);

        ELISE_COPY
        (
           select(i3.all_pts(),(FX+FY+FZ)%2),
           12,
           i3.out()
        );
		{
	        for (INT z=0; z<30 ; z++)
		        for (INT y=0; y<20 ; y++)
			         for (INT x=0; x<10 ; x++)
				     {
					     if ((x+y+z)%2)
						    BENCH_ASSERT(d[z][y][x] == 12);
						 else
						    BENCH_ASSERT(d[z][y][x] == x+y-z);
					}
		}
    }


    {
        Im3D<REAL4,REAL> i3(12,11,10,2.0);

        ELISE_COPY
        (
            rectangle(Pt3di(5,5,5),Pt3di(60,55,50)),
            1,
            i3.histo().chc(Virgule(FX,FY,FZ)/5)
        );

        
        Im3D<REAL4,REAL> j3(12,11,10,2.0);
        ELISE_COPY
        (
              j3.all_pts(),
              2+125*((FX>0)&&(FY>0)&&(FZ>0)),
              j3.out()
        );

        REAL dmax;
        ELISE_COPY(j3.all_pts(),Abs(i3.in()-j3.in()),VMax(dmax));
        BENCH_ASSERT(dmax<epsilon);
    }


    printf("OK Im3D \n");
}
