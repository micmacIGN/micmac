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




template <const int nbb> class Test_im2d_bits
{
     public :
         static void test();
         static void test(INT);
         static void verif(Im2D_Bits<nbb>,Im2D_U_INT1);
};

template <const int nbb>  void 
                          Test_im2d_bits<nbb>::verif
                          (
                              Im2D_Bits<nbb>    I1,
                              Im2D_U_INT1       I2
                          )
{
    INT dif;

    ELISE_COPY
    (
           I2.all_pts(),
           I1.in() != I2.in(),
           sigma(dif)
    );
    BENCH_ASSERT(dif == 0);
}


template <const int nbb>  void Test_im2d_bits<nbb>::test()
{
       Im2D_Bits<nbb> I1(105,120);
       Im2D_U_INT1    I2(105,120);


       ELISE_COPY
       (
           I1.all_pts(),
           Iconv(frandr() * 1000) % I1.vmax(),
           I1.out() | I2.out()
       );
       verif(I1,I2); 

       ELISE_COPY
       (
           //disc(Pt2di(50,50),60),
           disc(Pt2dr(50,50),60), // __NEW
           Iconv(frandr() * 1000) % I1.vmax(),
           I1.out() | I2.out()
       );
       verif(I1,I2); 

       ELISE_COPY
       (
           I1.all_pts(),
           frandr()* I1.vmax() * 0.99,
           I1.out() | I2.out()
       );
       verif(I1,I2); 


        INT dif;
        ELISE_COPY
        (
           select(rectangle(Pt2di(-10,-15),Pt2di(134,123)),frandr()>0.5),
           I1.in(13) != I2.in(13),
           sigma(dif)
        );
        BENCH_ASSERT(dif == 0);

        ELISE_COPY
        (
           select(I1.all_pts(),frandr()>0.5),
           Iconv(frandr() * 1000) % I1.vmax(),
           I1.out() | I2.out()
        );
        verif(I1,I2); 

        REAL rdif;
        ELISE_COPY
        (
           I1.all_pts(),
           Abs(I1.in()-I2.in())[Virgule(FX*0.7654+1.5,FY*0.6543+1.9)],
           sigma(rdif)
        );
        BENCH_ASSERT(rdif <epsilon);

        ELISE_COPY
        (
           I1.all_pts().chc(Iconv(Virgule(frandr()*20,frandr()*20))),
           Iconv(frandr() * 1000) % I1.vmax(),
           I1.max_eg() | I2.max_eg()
        );
        verif(I1,I2); 
}

template <const int nbb>  void Test_im2d_bits<nbb>::test(INT v_init)
{
    INT tx = 105;
    INT ty = 120;


    Im2D_Bits<nbb> Iv(tx,ty,v_init);
    Im2D_U_INT1    I2(tx,ty,v_init);
    U_INT1 **      i2 = I2.data();
    INT dif;

    ELISE_COPY
    (
        Iv.all_pts(),
        Iv.in() != v_init,
        sigma(dif)
    );
    BENCH_ASSERT(dif == 0);

    for (INT k = 0; k < 1000 ; k++)
    {
        INT x = ElMin(tx-1,(INT)(NRrandom3() * tx));
        INT y = ElMin(ty-1,(INT)(NRrandom3() * ty));
        INT v = ElMin((INT)((1<<nbb)-1),(INT)(NRrandom3() * (1<<nbb) ));
        i2[y][x] = v;
        Iv.set(x,y,v);
    }

    for (INT x=0 ; x<tx ; x++)
        for (INT y=0 ; y<ty ; y++)
            BENCH_ASSERT(Iv.get(x,y) == i2[y][x]);

    if (nbb == 1)
    {
        tx = 16;
        ty = 10;
        U_INT1 * dLin = NEW_VECTEUR(0,20,U_INT1);
        for (INT x = 0; x<20 ; x++)
            dLin[x] = 128;

        Im2D_BitsIntitDataLin IBIDL;
        Im2D_Bits<nbb> IB1(IBIDL,tx,ty,dLin);

        INT Dif;
        ELISE_COPY
        (
            IB1.all_pts(),
            (IB1.in() != (FX%8==0)),
            sigma(Dif)
        );
        BENCH_ASSERT(Dif==0);
        // cout << "Dif NBB = " << Dif << "\n";

        DELETE_VECTOR(dLin,0);
    }

}



void bench_im2d_bits()
{
    Test_im2d_bits<1>::test();
    Test_im2d_bits<2>::test();
    Test_im2d_bits<4>::test();

    Test_im2d_bits<1>::test(0);
    Test_im2d_bits<1>::test(1);
    Test_im2d_bits<2>::test(2);
    Test_im2d_bits<4>::test(1);
    Test_im2d_bits<4>::test(5);
    Test_im2d_bits<4>::test(10);

    cout << "OK BENCH Im2D_Bits \n";
}
