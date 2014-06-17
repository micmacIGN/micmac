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
  void test_bitm_1d_out_rle(Type *,TypeBase *,INT tx,bool real_fonc)
{
    Im1D<Type,TypeBase>  b1(tx);


    {   // verif all_pts
        INT s1,sx;	

        ELISE_COPY(b1.all_pts(),1,sigma(s1));
        ELISE_COPY(b1.all_pts(),FX,sigma(sx));
        BENCH_ASSERT ((s1 == tx) && (sx == som_x(0,tx)));


        ELISE_COPY(b1.all_pts(),Virgule(1,FX),Virgule(sigma(s1),sigma(sx)));
        BENCH_ASSERT ((s1 == tx) && (sx == som_x(0,tx)));
    }

    Fonc_Num f = (FX + Square(FX)) % 33;
    if (real_fonc)
       f = Rconv(f);


    // verif out put
    ELISE_COPY(b1.all_pts(),f,b1.out());
	INT x;
    for(x = 0; x <b1.tx(); x++)
       if (b1.data()[x] != (x+x*x) % 33)
       {
		   cout << " PB ds test_bitm_1d_out_rle \n";
		   cout << "x = " << x << " im = " << b1.data()[x] << "\n";
          exit(0);
       }

    // verif out put + cliping
    ELISE_COPY(rectangle(-1023,tx+2389),f,b1.out());
    for(x = 0; x <b1.tx(); x++)
       if (b1.data()[x] != (x+x*x) % 33)
       {
		   cout << " PB ds test_bitm_1d_out_rle \n";
		   cout << "x = " << x << " im = " << b1.data()[x] << "\n";
          exit(0);
       }

}

void test_bitm_1d_out_rle()
{
     test_bitm_1d_out_rle((INT *) 0,(INT *) 0, 245,false);
     test_bitm_1d_out_rle((INT *) 0,(INT *) 0, 245,true);

     test_bitm_1d_out_rle((U_INT1 *) 0,(INT *) 0, 445,false);
     test_bitm_1d_out_rle((U_INT1 *) 0,(INT *) 0, 145,true);

     test_bitm_1d_out_rle((REAL *) 0,(REAL *) 0, 445,false);
     test_bitm_1d_out_rle((REAL *) 0,(REAL *) 0, 145,true);
}

/*******************************************/

template <class Type,class TypeBase>
     void test_bitm_1d_input_rle(Type *,TypeBase *,INT tx)
{
    Im1D<Type,TypeBase>  b1(tx);
    Im1D<Type,TypeBase>  b2(tx);


    // RLE input std
    Fonc_Num f = (FX + Square(FX)) % 33;
    ELISE_COPY(b1.all_pts(),f,b1.out());
    ELISE_COPY(b2.all_pts(),b1.in(),b2.out());

    for(int x = 0; x <b1.tx(); x++)
       if (b1.data()[x] != b2.data()[x])
       {
		   cout << " PB ds test_test_bitm_1d_input_rle \n";
          exit(0);
       }

    // RLE input with def value

    TypeBase som;
    INT x1 = 210;
    INT x2 = 520;
    ELISE_COPY(b2.all_pts(),44,b2.out());
    ELISE_COPY(rectangle(-x1,tx+x2),b2.in(55),sigma(som));

    BENCH_ASSERT (som == (44 * tx + 55 * (x1+x2)));


}

void test_bitm_1d_input_rle()
{
     test_bitm_1d_input_rle((INT *) 0,(INT *) 0, 245);
     test_bitm_1d_input_rle((U_INT1 *) 0,(INT *) 0, 445);
     test_bitm_1d_input_rle((REAL *) 0,(REAL *) 0, 445);

}

/*******************************************/



template <class Type,class TypeBase>
  void test_bitm_1d_out_integer(Type *,TypeBase *,INT tx)
{
    Im1D<Type,TypeBase>  b1(tx);
    // Im1D<Type,TypeBase>  b2(tx);

    Fonc_Num fsel = FX%2;
    Fonc_Num fv1 =  FX%78;
    Fonc_Num fv2 =  (FX + 1009/(FX+10) + Square(FX)) % 34;

   //  test avec select
    ELISE_COPY(b1.all_pts(),fv1,b1.out());
    ELISE_COPY(select(b1.all_pts(),fsel),fv2,b1.out());

    // verif out put
	int x;
    for(x = 0; x <b1.tx(); x++)
       if (    b1.data()[x] 
           !=  (   (x%2)                        ? 
                   ((x + 1009/(x+10) +x*x)%34)  :
                   (x%78)
               )
          )
       {
					  cout << " PB ds test_bitm_1d_out_integer \n";
					   cout << "x = " << x << " im = " << (TypeBase) b1.data()[x] << "\n";
					   cout << ((x + 1009/(x+10) +x*x)%34) << "\n";
          exit(0);
       }


   //  test avec select + clip

    ELISE_COPY(b1.all_pts(),fv1,b1.out());
    ELISE_COPY(select(rectangle(-1000,tx+2000),fsel),fv2,b1.oclip());

    // verif out put
    for(x = 0; x <b1.tx(); x++)
       if (    b1.data()[x] 
           !=  (   (x%2)                        ? 
                   ((x + 1009/(x+10) +x*x)%34)  :
                   (x%78)
               )
          )
       {
					   cout << " PB ds test_bitm_1d_out_integer \n";
          cout << "x = " << x << " im = " << (TypeBase) b1.data()[x] << "\n";
          cout << ((x + 1009/(x+10) +x*x)%34) << "\n";
          exit(0);
       }

}

void test_bitm_1d_out_integer()
{
     test_bitm_1d_out_integer((U_INT1 *)0,(INT *) 0,987);
     test_bitm_1d_out_integer((REAL *)0,(REAL *) 0,987);
}


/*******************************************/

template <class Type,class TypeBase>
  void test_bitm_1d_input_integer(Type *,TypeBase *,INT tx)
{
    Im1D<Type,TypeBase>  b1(tx);
    Im1D<Type,TypeBase>  b2(tx);

    Fonc_Num fsel = FX%2;
    Fonc_Num fval =  (FX + 1009/(FX+10) + Square(FX)) % 34;

   //  test avec select
    ELISE_COPY(b1.all_pts(),fval,b1.out());

    ELISE_COPY(select(b1.all_pts(),fsel!=0),b1.in(),b2.out());
    ELISE_COPY(select(b1.all_pts(),fsel==0),b1.in(),b2.out());

    INT dif;
    ELISE_COPY(b2.all_pts(),fval!=b2.in(),sigma(dif));
    BENCH_ASSERT (dif == 0);

   TypeBase som;

   ELISE_COPY(b2.all_pts(),33,b2.out());
   ELISE_COPY(select(rectangle(-150,tx+200),1),b2.in(12),sigma(som));
   BENCH_ASSERT (som == (350*12 +tx*33));


}

void test_bitm_1d_input_integer()
{
    test_bitm_1d_input_integer((INT *)0,(INT *)0,234);
    test_bitm_1d_input_integer((REAL *)0,(REAL *)0,234);
    test_bitm_1d_input_integer((REAL4 *)0,(REAL *)0,234);
}

/*******************************************/

void test_bitm_1d()
{
    test_bitm_1d_out_rle();
    test_bitm_1d_input_rle();
    test_bitm_1d_out_integer();
    test_bitm_1d_input_integer();

	cout << "OK bitm_1d \n";
}


