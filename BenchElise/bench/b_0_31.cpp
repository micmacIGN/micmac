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


/*
         Un exemple bebete, mais qui devrait tester
    ttes les fonctionnalites requises.
*/


template <class Type> class BENCH_SOM_UOP : public Simple_OP_UN<Type>
{
   public :
       BENCH_SOM_UOP(INT nb,Type * pond,Im1D_INT4 b,bool comp) :
           _b         (b),
           _nb_fonc   (nb),
           _pond      (NEW_TAB(nb,INT)),
           _dup_pond  (comp  ?  (NEW_TAB(nb,INT) ) : 0)
       {
           convert(_pond,pond,nb);
           if (_dup_pond)
              convert(_dup_pond,pond,nb);
       }

       ~BENCH_SOM_UOP()
       {
            DELETE_TAB(_pond);
            if (_dup_pond)
               DELETE_TAB(_dup_pond);
       }

       Simple_OP_UN<Type> * dup_comp(const Arg_Comp_Simple_OP_UN &)
       {
            return new BENCH_SOM_UOP(_nb_fonc,_pond,_b,true);
       }


       virtual void  calc_buf(Type ** output,Type ** input,INT nb,
                              const Arg_Comp_Simple_OP_UN & arg) ;

       Im1D_INT4 _b;
       INT    _nb_fonc;
       INT *      _pond;
       INT *      _dup_pond;

};


template  <class Type> void 
          BENCH_SOM_UOP<Type>::calc_buf
          (
                Type ** output,
                Type ** input,
                INT nb,
                const Arg_Comp_Simple_OP_UN & arg
          ) 
{
      INT din = arg.dim_in();
      BENCH_ASSERT(din==_nb_fonc);
      INT * p2 = _b.data();
      for (INT i =0; i<nb ; i++)
      {
          output[0][i] = 0;
          for (INT d =0; d<din ; d++)
              output[0][i] += (_pond[d]+p2[d]+_dup_pond[d]) * input[d][i];
      }
}



Fonc_Num I1_bench_user_oper_som (Fonc_Num f)
{
     INT s[4] = {1,2,3,4};
     Im1D_INT4 Pond(4);
     ELISE_COPY(Pond.all_pts(),10*(1+FX),Pond.out());

     return 
            create_users_oper 
            (
                 new BENCH_SOM_UOP<INT>(4,s,Pond,false),
                 0,
                 f,
                 1
            );        
}

Fonc_Num I2_bench_user_oper_som (Fonc_Num f1,Fonc_Num f2)
{
     INT s[2] = {20,20};
     Im1D_INT4 Pond(2);
     ELISE_COPY(Pond.all_pts(),7,Pond.out());

     return 
            create_users_oper 
            (
                 new BENCH_SOM_UOP<INT>(2,s,Pond,false),
                 0,
                 Virgule(f1,f2),
                 1
            );        
}



void bench_user_oper(Pt2di sz)
{
     Im2D_U_INT1 I1(sz.x,sz.y);
     Im2D_U_INT1 I2(sz.x,sz.y);
     Im2D_U_INT1 I3(sz.x,sz.y);
     Im2D_U_INT1 I4(sz.x,sz.y);

     Im1D_INT4 Pond(4);
     ELISE_COPY(Pond.all_pts(),10*(1+FX),Pond.out());

     ELISE_COPY
     (
         I1.all_pts(),
         Virgule(frandr(),frandr(),frandr(),frandr())*255,
         Virgule(I1.out(),I2.out(),I3.out(),I4.out())
     );

     INT dif;
     ELISE_COPY
     (
          I1.all_pts(),
          Abs
          (
             (12*I1.in() +24*I2.in() +36*I3.in() +48*I4.in() +FX )* 47
             -I2_bench_user_oper_som
              (
                  I1_bench_user_oper_som(Rconv(Virgule(I1.in(),I2.in(),I3.in(),I4.in()))),
                  FX
              )
          ),
          sigma(dif)
     );

     BENCH_ASSERT(dif == 0);
}

void bench_grad_bilin(INT a,INT b,INT c,INT d,Pt2di sz)
{
    Im2D_U_INT1 I(sz.x,sz.y);
    ELISE_COPY(I.all_pts(),a+b*FX+c*FY+d*FX*FY,I.out());

    REAL difx,dify;
    ELISE_COPY
    (
           select
           (
                I.all_pts().chc(Virgule(FX+frandr(),FY+frandr())),
                   (FX>=1.0) && (FX<sz.x-1.0)
                && (FY>=1.0) && (FY<sz.y-1.0)
           ),
           Abs(grad_bilin(Virgule(FX,FY),I) -Virgule(b+d*FY,c+d*FX)),
           Virgule(VMax(difx),VMax(dify))
    );

    BENCH_ASSERT(difx < epsilon);
    BENCH_ASSERT(dify < epsilon);
}

void bench_user_oper()
{
    bench_user_oper(Pt2di(30,20));
    bench_user_oper(Pt2di(20,30));

    bench_grad_bilin(4,3,2,1,Pt2di(10,10));
}








