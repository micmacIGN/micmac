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



void bench_shading(Im2D_U_INT1 bigI,Pt2di tr,Pt2di sz,Pt2di dir,double thresh)
{

   Im2D_U_INT1 I(sz.x,sz.y);
   Im2D_U_INT1 ShThr(sz.x,sz.y);

   ELISE_COPY (I.all_pts(),trans(bigI.in(),tr),I.out());

   Symb_FNum b(binary_shading(I.in(),thresh));
   Symb_FNum r(gray_level_shading(I.in()));
   INT dif;

   ELISE_COPY
   (
       line_map_rect(dir,Pt2di(0,0),sz),
       b,
       ShThr.out()
   );

   if (thresh < 0.0)
   {
        thresh = -thresh;
        dir = Pt2di(-dir.x,-dir.y);
   }

   ELISE_COPY
   (
       line_map_rect(dir,Pt2di(0,0),sz),
       (ShThr.in() != (r > thresh )),
       sigma(dif)
   );


   BENCH_ASSERT(dif == 0);
}

void benc_proj_cav(Pt2di dir,REAL z0,REAL steep, REAL eps)
{
   Pt2di sz (200,200);

   Im2D_REAL8 mnt(sz.x,sz.y);
   ELISE_COPY(mnt.all_pts(),3*cos(FX/6.0) * sin(FY/7.0),mnt.out());

   Im2D_REAL8 cx(sz.x,sz.y),cy(sz.x,sz.y);

   ELISE_COPY
   (
       line_map_rect(dir,Pt2di(0,0),sz),
       proj_cav(mnt.in(),z0,steep),
       Virgule(cx.out(),cy.out())
   );

   REAL dif;
   ELISE_COPY 
   (
        rectangle(Pt2di(10,10),Pt2di(sz.x-10,sz.y-10)),
        Abs
        (
           z0 -steep*(dir.x*(cx.in()-FX) + dir.y*(cy.in()-FY))/euclid(dir)
           - mnt.in()[Virgule(cx.in(),cy.in())]
        ),
        VMax(dif)
   );
   BENCH_ASSERT(dif < eps);
}





void bench_shading()
{
   Im2D_U_INT1 I(512,512);

   ELISE_fp fp;

   if (fp.ropen("../IM_ELISE/lena",true))
   {
       fp.close(true);
       Elise_File_Im FLena("../IM_ELISE/lena",Pt2di(512,512),GenIm::u_int1);
       ELISE_COPY(I.all_pts(),FLena.in(),I.out());
	   cout << "got a Lena \n";
   }
   else
   {
	   ELISE_COPY(I.all_pts(),(FX+FY+Square(FX+FY)/20)%256,I.out());
	   cout << "could not load Lena \n";
   }

   bench_shading(I,Pt2di(0,0),Pt2di(200,300),Pt2di(5,2),2.7);
   bench_shading(I,Pt2di(0,0),Pt2di(200,300),Pt2di(2,2),6.798);

   for (int x = 0; x < 20; x++)
       bench_shading
       (
           I,
           Pt2di(100,100),
           Pt2di(60,40),
           Pt2di(2+x,3-x),
           0.267+x*1.863
       );


   bench_shading(I,Pt2di(0,0),Pt2di(200,300),Pt2di(1,1),-2.7);

   /*
      This, legitimately does not work. Because in the inverse order
      the mappinf of lines is not exactly the same.

       bench_shading(I,Pt2di(0,0),Pt2di(200,300),Pt2di(5,2),-2.7);
   */

       // a big eps because bilinear approximation in diag use neighbours
       // unused in cav projection.
   benc_proj_cav(Pt2di(1,1) ,0.0,4.1,0.05); 

   benc_proj_cav(Pt2di(1,0) ,0.0,4.2,1e-7); 
   benc_proj_cav(Pt2di(0,-1) ,0.0,4.3,1e-7); 
   benc_proj_cav(Pt2di(1,0) ,0.0,-4.3,1e-7); 

   cout << "OK shading \n";
}




