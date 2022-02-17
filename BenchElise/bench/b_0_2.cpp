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




       //-----------------------------------------------------
       // verification d'une ecriture simple sur une bitmap
       //-----------------------------------------------------

void verif_bitm_0()
{
    INT tx =100;
    INT ty =120;
    Im2D_U_INT1 b(tx,ty);

    ELISE_COPY (b.all_pts(),FX,b.out());
    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b.data()[y][x] != x)
            {
                 cout << " Pb avec verif bitm  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im  = " << b.data()[y][x] << "\n";
                 exit(0);
            }

}


       //-----------------------------------------------------
       // verification d'une addition melant les diff types
       //-----------------------------------------------------

void verif_bitm_1()
{
    INT tx =100;
    INT ty =120;
    Im2D_REAL8 b(tx,ty);

    ELISE_COPY (b.all_pts(),2.5+(FX+FY)+1.5+(FX+2),b.out());
    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b.data()[y][x] != 2*x + y + 6.0)
            {
                 cout << " Pb avec verif bitm  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im  = " << b.data()[y][x] << "\n";
                 exit(0);
            }

}


     //================================
     //  Verification sur une expression
     // melangeant +,*,max,min,-,/
     //================================

void verif_bitm_2()
{
    INT tx =100;
    INT ty =120;
    Im2D_REAL8 b(tx,ty);

    ELISE_COPY 
    (
          b.all_pts()
         ,    Max(FX,2.25*FY)
           +  Max(Min(FX,FY),Iconv(0.5*FY))
           +  (FX-3)/(FY+2)
           +  (FX-FY)/2.0
           -  12.5
           +  (-FX + -(FY+0.5))
           +  Abs(FX-FY)
           +  Abs(FX-FY+10.25)
           +  Square(FX)
           +  Square(FX+0.5)
         ,b.out()
    );
    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b.data()[y][x] != 
                      ElMax((REAL)x,2.25*y)
                  +   ElMax(ElMin(x,y),(int)(0.5*y))
                  +   (x-3)/(y+2)
                  +   (x-y)/2.0
                  -  12.5
                  +  (-x+ -(y+0.5))
                  +  ElAbs(x-y)
                  +  ElAbs(x-y+10.25)
                  +  (x*x)
                  +  ((x+0.5)*(x+0.5))
              )
            {
                 cout << " Pb avec verif bitm  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im  = " << b.data()[y][x] << "\n";
                 exit(0);
            }

}


    //================================
    //   verif for ==,<,!= ...
    //================================

void verif_op_comp(bool rx,bool ry)
{
    INT tx =50;
    INT ty =50;

    Im2D_U_INT1 beq(tx,ty);
    Im2D_U_INT1 bneq(tx,ty);
    Im2D_U_INT1 bis(tx,ty);
    Im2D_U_INT1 bieq(tx,ty);
    Im2D_U_INT1 bss(tx,ty);
    Im2D_U_INT1 bseq(tx,ty);

    Fonc_Num fx = (rx ? Rconv(FX) : FX);
    Fonc_Num fy = (ry ? Rconv(FY) : FY);

    ELISE_COPY(beq.all_pts(),fx==fy,beq.out());
    ELISE_COPY(beq.all_pts(),fx!=fy,bneq.out());
    ELISE_COPY(beq.all_pts(),fx<fy,bis.out());
    ELISE_COPY(beq.all_pts(),fx<=fy,bieq.out());
    ELISE_COPY(beq.all_pts(),fx>fy,bss.out());
    ELISE_COPY(beq.all_pts(),fx>=fy,bseq.out());

    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if 
            (
                    (beq.data()[y][x] !=  (y==x))
                ||  (bneq.data()[y][x] !=  (y!=x))
                ||  (bis.data()[y][x] !=  (x<y))
                ||  (bieq.data()[y][x] !=  (x<=y))
                ||  (bss.data()[y][x] !=  (x>y))
                ||  (bseq.data()[y][x] !=  (x>=y))
            )
            {
                 cout << " Pb avec verif bitm  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", beq  = " << (int) beq.data()[y][x] 
                      << "\n";
                 exit(0);
            }
}

void verif_op_comp()
{
    verif_op_comp(true,true);
    verif_op_comp(false,true);
    verif_op_comp(true,false);
    verif_op_comp(false,false);
}

    //================================
    //   verif for ==,<,!= ...
    //================================


void verif_op_bin_int()
{
    INT tx =100;
    INT ty =120;
    Im2D_INT4 b(tx,ty);

    ELISE_COPY
    (
         b.all_pts()
       ,    (FX&FY)
        +   (FX&&FY)
        +   (FX|FY)
        +   (FX||FY)
        +   (FX ^ FY)
        +   ElXor(FX,FY)
        +   FX % (FY + 1)
        +   (FX << (FY % 3))
        +   (FX >> (FY % 2))
        +   (! ((FX%7) > (FY%8)))
        +   ( (~(FX+3*FY)) & 255)
       ,b.out()
    );

    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b.data()[y][x] != 
                      (x&y)
                   +  (x&&y)
                   +  (x|y)
                   +  (x||y)
                   +  (x ^ y)
                   +  ((x&&y) ? 0 : (x||y))
                   +  x % (y +1)
                   + ( x << (y % 3))
                   + ( x >> (y % 2))
                   + ( ! ((x%7) > (y%8)))
                   + ( (~(x+3*y)) & 255)
              )
            {
                 cout << " Pb avec verif bitm  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im  = " << b.data()[y][x] << "\n";
                 exit(0);
            }



}


void verif_op_round()
{
    INT tx =100;
    INT ty =120;
    Im2D_INT4 b(tx,ty);


// LES VALEURS SONT VOLONTAIREMENT CHOISIES POUR TOUJOURS
// AVOIR DES DOUBLE LES REPRESENTANT VRAIMENT.
// Si par exemple on prennait FX*0.1 + FY*0.35, on risquerait
// d'avoir 1+epsilon avec Elise et 1-epislon avec l'expression a la
// main (car le compilateur l'aura rearrangee).

    ELISE_COPY
    (
         b.all_pts()
       ,    round_up(FX/8.0+ 3*FY/8.0)
        +   round_down(1/16.0 +5*FX/4.0 +FY/2.0)
        +   round_ni(FX/4.0 + FY/8.0)
        +   round_ni_inf(FY/4.0 + FX/8.0)
       ,b.out()
    );

    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b.data()[y][x] != 
                        round_up(x/8.0+ 3*y/8.0)
                    +   round_down(1/16.0 +5*x/4.0 +y/2.0)
                    +   round_ni(x/4.0 + y/8.0)
                    +   round_ni_inf(y/4.0 + x/8.0)
              )
            {
                 cout << " Pb avec verif_op_round   : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im  = " << b.data()[y][x] << "\n";
                 exit(0);
            }

    cout << "OK OP ROUND \n";
}



void verif_bitm_in_out()
{
    INT tx =100;
    INT ty =120;

    Im2D_INT4 b1(tx,ty);
    Im2D_INT4 b2(tx,ty);

    ELISE_COPY
    (
        b1.all_pts(),
        FX+FY,
        b1.out()
    );

    ELISE_COPY
    (
        b1.all_pts(),
        b1.in(),
        b2.out()
    );

    for (int y=0 ; y<ty ; y++)
        for (int x=0 ; x<tx ; x++)
            if (b1.data()[y][x] != b2.data()[y][x])
            {
                 cout << " Pb avec verif bitm  : " 
                      << " x  = " << x
                      << ", y  = " << y
                      << ", im  = " << b2.data()[y][x] << "\n";
                 exit(0);
            }

}


template <class Type,class TypeBase>
      void verif_bitm_def_val
      (
            Pt2di p1,
            Pt2di p2,
            Pt2di sz,
            TypeBase v_im,
            TypeBase v_def,
            Type     *
      )
{
    Im2D<Type,TypeBase> b(sz.x,sz.y);
    TypeBase s;


    ELISE_COPY(b.all_pts(),v_im,b.out());

    ELISE_COPY(rectangle(p1,p2),b.in(v_def),sigma(s));


    TypeBase s_sel;
    ELISE_COPY(select(rectangle(p1,p2),1),b.in(v_def),sigma(s_sel));

    BENCH_ASSERT
    (
          (s== (p2.x-p1.x)*(p2.y-p1.y)*v_def + sz.x*sz.y*(v_im-v_def))
       && (s == s_sel)
    );

}

void verif_bitm_def_val()
{
    verif_bitm_def_val
    (
          Pt2di(-110,-10),
          Pt2di(123,135),
          Pt2di(95,78),
          1.5,
          -3.75,
          (REAL *)0
     );

    verif_bitm_def_val
    (
          Pt2di(-110,-10),
          Pt2di(123,135),
          Pt2di(95,78),
          111.25,
          -404.50,
          (REAL4 *)0
     );


    verif_bitm_def_val
    (
          Pt2di(-10,-10),
          Pt2di(123,135),
          Pt2di(95,78),
          3,
          5,
          (INT *)0
     );

    verif_bitm_def_val
    (
          Pt2di(-10,-10),
          Pt2di(123,135),
          Pt2di(95,78),
          3,
          5,
          (U_INT1 *)0
     );

}

void verif_tx_ty(INT tx,INT ty)
{
     
    Im2D_U_INT1 b (tx,ty);

    BENCH_ASSERT( (tx == b.tx()) && (ty == b.ty()));
}


template <class Type,class TypeBase>
      void verif_val_init
      (
            Type *,
            TypeBase *,
            Type v
      )
{
   Im2D<Type,TypeBase> i2(20,30,v);
   Type ** d2 = i2.data();

   for (INT x=0;x<20;x++)
       for (INT y=0;y<30;y++)
           BENCH_ASSERT(d2[y][x] == v);

   
   Im1D<Type,TypeBase> i1(300,v);
   Type * d1 = i1.data();

   {
		for (INT x=0 ; x<300 ; x++)
			 BENCH_ASSERT(d1[x]==v);
   }
 }

void verif_val_init()
{
    verif_val_init((U_INT1 *)0  ,(INT4 *) 0     ,(U_INT1)  255);
    verif_val_init((U_INT1 *)0  ,(INT4 *) 0     ,(U_INT1)  0);
    verif_val_init((INT1 *)0    ,(INT4 *) 0     ,(INT1)   -128);

    verif_val_init((U_INT2 *)0  ,(INT4 *) 0     ,(U_INT2)   60000);
    verif_val_init((INT2 *)0    ,(INT4 *) 0     ,(INT2)     30000);
    verif_val_init((INT2 *)0    ,(INT4 *) 0     ,(INT2)    -30000);

    verif_val_init((REAL8 *)0    ,(REAL *) 0    ,(REAL8)    1.2345e2);
    verif_val_init((REAL4 *)0    ,(REAL *) 0    ,(REAL4)    1.25) ;
}


template <class Type,class TypeBase> 
        void verif_bitm_gray_im_red
             (
                   Type     *,
                   TypeBase *,
                   Pt2di    aSz,
                   INT      aZoom
             )
{
    Im2D<Type,TypeBase> anI0(aSz.x,aSz.y);

    ELISE_COPY(anI0.all_pts(),255*frandr(),anI0.out());
    
    Im2D<Type,TypeBase> anIR1 = anI0.gray_im_red(aZoom);
    Pt2di aSzRed = aSz/aZoom;
    Im2D<TypeBase,TypeBase> anIR2 (aSzRed.x,aSzRed.y,(TypeBase)0);


    ELISE_COPY
    (
        rectangle(Pt2di(0,0),aSzRed*aZoom),
        anI0.in(),
        (anIR2.histo()).chc(Virgule(FX,FY)/aZoom)
    );

     BENCH_ASSERT(anIR1.sz() == anIR2.sz());
     TypeBase aDif;
     ELISE_COPY
     (
        anIR1.all_pts(),
        Abs(anIR1.in()-anIR2.in()/ElSquare(aZoom)),
        sigma(aDif)
     );

     BENCH_ASSERT(aDif < epsilon);

}

void verif_bitm_gray_im_red()
{
    verif_bitm_gray_im_red((U_INT1 *)0,(INT *)0,Pt2di(100,200),7);
    verif_bitm_gray_im_red((U_INT2 *)0,(INT *)0,Pt2di(100,200),5);
    verif_bitm_gray_im_red((REAL *)0,(REAL *)0,Pt2di(100,200),5);
    verif_bitm_gray_im_red((REAL *)0,(REAL *)0,Pt2di(100,200),1);
}

static Pt2di aRandSz()
{
    return Pt2di 
           (
                round_ni(1+100*NRrandom3()),
                round_ni(1+100*NRrandom3())
           );
}

template <class T1,class T2> void verif_Resize(T1*,T2*)
{
    Pt2di aSz1 = aRandSz();
    Pt2di aSz2 = aRandSz();
    Pt2di aSz3 = aRandSz();


    Im2D<T1,T2> I1(aSz1.x,aSz1.y);
    Im2D<T1,T2> I2(aSz2.x,aSz2.y);

    I1.Resize(aSz3);
    I2.Resize(aSz3);

    BENCH_ASSERT(I1.tx() == aSz3.x);
    BENCH_ASSERT(I1.ty() == aSz3.y);


    REAL aNb;
    ELISE_COPY
    (
         I1.all_pts(),
         Virgule(Fonc_Num(1.0),frandr()*10),
         Virgule(sigma(aNb),I1.out()|I2.out())
    );

    REAL aDif = ElAbs(aNb - aSz3.x*aSz3.y);
    BENCH_ASSERT(aDif < epsilon);
    
    ELISE_COPY(I1.all_pts(),Abs(I1.in()-I2.in()),sigma(aDif));
    BENCH_ASSERT(aDif < epsilon);
}

void verif_Resize()
{
    for (INT aNb = 0 ; aNb < 200 ; aNb ++)
    {
         verif_Resize((U_INT1 *)0,(INT *)0);
         verif_Resize((INT2 *)0,(INT *)0);
         verif_Resize((REAL4 *)0,(REAL*)0);
    }
}

void verif_bitm()
{

    verif_Resize();
    
    verif_bitm_gray_im_red();
    All_Memo_counter MC_INIT;
    stow_memory_counter(MC_INIT);

    {
        verif_tx_ty(123,218);
        verif_tx_ty(3,8);
        verif_tx_ty(44,88);
    
        verif_bitm_0();
        verif_bitm_1();
        verif_bitm_2();
    
        verif_op_comp();
        verif_op_bin_int();
    
        verif_bitm_in_out();
        verif_bitm_def_val();
    
        verif_op_round();
        verif_val_init();
    }


    verif_memory_state(MC_INIT);
    cout << "OK verif_bitm \n";
}



