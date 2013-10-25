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
                          

template <class Type,class TypeBase> 
        void bench_max_loc
             (
                  Im2D<Type,TypeBase>  Im,
                  Pt2di    vois,
                  Pt2di    p0, Pt2di    p1     
             )
{
     ElSTDNS vector<Pt2di> Pts;
     CalcMaxLoc<Type,TypeBase> CML;
     CML.AllMaxLoc(Pts,Im,vois,p0,p1,-10000);

     REAL Coeff_X = 1 << 20;
     REAL Coeff_Y = 1 << 10;

     Fonc_Num f  = Im.in() - FY/Coeff_Y -  FX/Coeff_X;

     Liste_Pts_INT2  lpts(2);
     ELISE_COPY
     (
        select (rectangle(p0,p1),f==(rect_max(f,vois))),
        1,
        lpts
     );
     ElSTDNS vector<Pt2di> Pts2;
     {
        Im2D_INT2  im_xy = lpts.image();
        INT2 * xPtr = im_xy.data()[0];
        INT2 * yPtr = im_xy.data()[1];
        INT NbPts = im_xy.tx();

         for (INT k=0 ; k<NbPts ; k++)
            Pts2.push_back(Pt2di(xPtr[k],yPtr[k]));
     }
     std::sort(Pts.begin(),Pts.end());
     std::sort(Pts2.begin(),Pts2.end());

     if (Pts!=Pts2) 
     {
        for (INT y=0; y<Im.ty() ; y++)
        {
            for (INT x=0; x<Im.tx() ; x++)
            {
                 printf("%4d ",Im.data()[y][x]);
            }
            cout << "\n";
        }
        cout << Pts << "\n";
        cout << Pts2 << "\n";
        getchar();
     }
}

void bench_max_loc(Pt2di SZ)
{
    Im2D_INT4 Im(SZ.x,SZ.y);
    Pt2di vois;

/*
    ELISE_COPY(Im.all_pts(),1000*frandr(),Im.out());
    vois =Pt2di (3,5);
    bench_max_loc(Im,vois,vois,SZ-vois);
*/

    ELISE_COPY(Im.all_pts(),1000*frandr(),Im.out());
    vois =Pt2di (1,1);
    bench_max_loc(Im,vois,vois,SZ-vois);
    vois =Pt2di (2,2);
    bench_max_loc(Im,vois,vois,SZ-vois);
    vois =Pt2di (5,3);
    bench_max_loc(Im,vois,vois,SZ-vois);
    bench_max_loc(Im,vois,vois*2,SZ-vois*3);

    vois =Pt2di (3,5);
    ELISE_COPY(Im.all_pts(),10*frandr(),Im.out());
    bench_max_loc(Im,vois,vois*2,SZ-vois*3);


    ELISE_COPY(Im.all_pts(),unif_noise_4(10)*3,Im.out());
    vois =Pt2di (7,5);
    bench_max_loc(Im,vois,vois,SZ-vois);

    ELISE_COPY(Im.all_pts(),frandr()>0.5,Im.out());
    vois =Pt2di (1,1);
    bench_max_loc(Im,vois,vois,SZ-vois);
    vois =Pt2di (2,2);
}


void bench_max_loc()
{
     for (INT k=0;k<10; k++)
     {
         cout << k << endl;
         bench_max_loc(Pt2di(60,60));
     }
}


void bench_hough()
{
   {
       ElHough * H  = ElHough::NewOne(Pt2di(20,0),1.0,1.0,ElHough::ModeStepAdapt,10.0,0.5); 
       delete H;
   }
   bench_max_loc();
   ElHough::BenchPolygoneClipBandeVert() ;
   printf("FIN HOUGH\n");
}


