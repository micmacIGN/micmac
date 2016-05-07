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


class BenchConc : public ParamConcOpb
{
     public :

       BenchConc(INT NbLabel,bool toDel) :
           mNbLab    (NbLabel),
           mToDelete (toDel),
           mCpt      (0)
       {
       }
     private :
       
       virtual bool ToDelete()
       {
            return mToDelete;
       }
       virtual INT ColBig()
       {
             return 2 *mNbLab + 1;
       }

       virtual INT ColSmall(const EliseRle::tContainer &,const Box2di &,INT ColInit)     
       {
             mCpt++;
             return ColInit + mNbLab;
       }

     public :

       INT  mNbLab;
       bool mToDelete;
       INT mCpt;
};


void bench_im_rle(Pt2di sz,INT NbLabel,Pt2di SzBox,INT NbLissage)
{
    typedef EliseRle::tIm  tIm;
    bool  ModeV8 = NRrandom3() > 0.5;
    Im2D<tIm,INT> Im1(sz.x,sz.y);
    Im2D<tIm,INT> Im2(sz.x,sz.y);
    Im2D<tIm,INT> Im3(sz.x,sz.y);
    Im2D<tIm,INT> ImOri(sz.x,sz.y);

    Im2D<tIm,INT> ImBox1(sz.x,sz.y,0);
    Im2D<tIm,INT> ImBox2(sz.x,sz.y,0);

    Fonc_Num f = Iconv(frandr()*20*NbLabel)%NbLabel;
    for (INT k=0 ; k<NbLissage ; k++)
        f = label_maj(f,NbLabel,Pt2di(5,5));


    ELISE_COPY(Im1.all_pts() , f-2, Im1.out()|Im2.out()|Im3.out()|ImOri.out() );
    ELISE_COPY(Im1.border(1) , 0  , Im1.out()|Im2.out()|Im3.out()|ImOri.out() );


    tIm ** data1 = Im1.data();
    tIm ** data2 = Im2.data();


    Neighbourhood V8 = Neighbourhood::v8();     
    Neighbourhood V4 = Neighbourhood::v4();     

    U_INT1 ColBig = NbLabel * 2 +1;
    EliseRle::tContainer  Rles;

    INT cptSmall1 =0; 
    INT cptSmall2 =0; 
    INT cptBig1 =0; 
    INT cptBig2 =0; 
    ELISE_COPY(select(Im1.all_pts(),Im1.in()<0),0,Im1.out());

    BenchConc aBc(NbLabel,false);
    BenchConc * aBcPtr = &aBc;

    if (NRrandom3() >0.5)
        aBcPtr = new BenchConc(NbLabel,true);

    ELISE_COPY
    (
         Im3.all_pts(),
         BoxedConc(Im3.in(0),SzBox,ModeV8,aBcPtr,(INT)(9+SzBox.y*2.9*NRrandom3())),
         Im3.out()
    );


    for (INT y=0; y<sz.y ; y++)
    {
       for (INT x=0; x<sz.x ; x++)
       {
           tIm v1 = data1[y][x];
           Neighbourhood V = ModeV8 ? V8 : V4;
           Liste_Pts_INT2 l2(2);
           if ((v1 >0) && (v1<=NbLabel))
           {
               Pt2di p0,p1;
               ELISE_COPY
               (
                   conc
                   (
                      Pt2di(x,y),
                      Im1.neigh_test_and_set (V,v1,ColBig,ColBig+1)
                   ),
                   Virgule(FX,FY),
                   (l2|p0.VMin()|p1.VMax())
              );
              p1 += Pt2di(1,1);

              if ((p1.x-p0.x<=SzBox.x) && (p1.y-p0.y<=SzBox.y))
              {
                 cptSmall1++;
                 ELISE_COPY(l2.all_pts(),v1+NbLabel,Im1.out());
              }
              else
              {
                 cptBig1++;
              }
              ELISE_COPY(rectangle(p0,p1),(ImBox1.in()+v1)%256,ImBox1.out());
           }

           tIm v2 = data2[y][x];
           if ((v2 >0) && (v2<=NbLabel))
           {
                Box2di Box =  EliseRle::ConcIfInBox
                              (
                                   Pt2di(x,y),
                                   Rles,
                                   data2,
                                   v2,ColBig,
                                   ModeV8,
                                   SzBox
                              );
                if (! Rles.empty())
                {
                   cptSmall2++;
                   EliseRle::SetIm(Rles,data2, v2+NbLabel);
                }
                else
                {
                   cptBig2++;
                }
                ELISE_COPY
                (
                    rectangle(Box._p0,Box._p1),
                    (ImBox2.in()+v2)%256,
                    ImBox2.out()
                );
           }
       }
    }
    ELISE_COPY(select(Im1.all_pts(),ImOri.in()<0),ImOri.in(),Im1.out());

    BENCH_ASSERT(cptSmall2 == cptSmall1);
    BENCH_ASSERT(cptBig2 == cptBig1);
    INT Dif1;
    ELISE_COPY
    (
       Im1.all_pts(),
       Abs(Im1.in()-Im2.in()),
       VMax(Dif1)
    );
    BENCH_ASSERT(Dif1==0);
    ELISE_COPY
    (
       Im1.all_pts(),
       Abs(ImBox1.in()-ImBox2.in()),
       VMax(Dif1)
    );
    BENCH_ASSERT(Dif1==0);


    ELISE_COPY
    (
       Im1.all_pts(),
       Abs(Im1.in()-Im3.in()),
       VMax(Dif1)
    );
    
    BENCH_ASSERT(Dif1==0);
}



void bench_im_rle()
{
     for (INT k=0 ; k < 20 ; k++)
     {
          All_Memo_counter MC_INIT;
          stow_memory_counter(MC_INIT);
          bench_im_rle
          (
                Pt2di(150,150) + Pt2di(Pt2dr(50*NRrandom3(),50*NRrandom3())),
                INT(2+6*NRrandom3()),
                Pt2di(Pt2dr(4+20*NRrandom3(),4+20*NRrandom3())),
                (INT)(1+4 * NRrandom3())
          );
          verif_memory_state(MC_INIT);      
     }
     cout << "END Bench RLE \n";
}









