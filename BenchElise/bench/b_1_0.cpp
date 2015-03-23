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



void test_EFF_Garb_Coll()
{
    int txy[2] = {-12,33};
    Elise_File_Im f("toto",2,txy,GenIm::u_int1,33,99);

    f.in();
    f.in(33);
}


/**************************************************/

const int dim = 3;

template <class Type,class Type_Base> void 
         test_fich_2d(Type *,Type_Base *,GenIm::type_el type_el,INT max_val)
{
    const INT tx = 185;
    const INT ty = 134;
    INT txy [2] = {tx,ty};
    INT   offset_0 = 123;
    char * name = "im_tmp/tmp0";

    Type buf[dim];


    /**************************************/
    /*   create a file                    */
    /**************************************/

    FILE * fp = Elise_fopen(name,"w");

    Im2D<Type,Type_Base> I1(tx,ty);
    Im2D<Type,Type_Base> I2(tx,ty);
    Im2D<Type,Type_Base> I3(tx,ty);

    // generate a "random" header
    for(INT nb = 0; nb <offset_0 ; nb++)
       fputc(nb,fp);
    

    for (INT y = 0; y < ty ; y++)
        for (INT x = 0; x < tx ; x++)
        {
            buf[0] = (x+y) %max_val;
            buf[1] = (x*x) %max_val;
            buf[2] = (103*x+447) %max_val;
            Elise_fwrite(buf,sizeof(Type),dim,fp);
        }

    Elise_fclose(fp);

    Elise_File_Im f(name,2,txy,type_el,dim,offset_0);

    /**************************************/
    /*   copy file on images              */
    /**************************************/
    copy
    (  I1.all_pts(),
       (FX%max_val, FY % max_val, (FX+FY) %max_val),
       (I1.out() , I2.out() , I3.out())
    );

    Pt2di p1 (12,8);
    Pt2di p2 (tx-5,ty-6);

    copy
    (    rectangle(p1,p2),
         f.in(),
         (I1.out() , I2.out() , I3.out())
   );

    /**************************************/
    /*   verif we have expected values    */
    /*    in  bitmaps                     */
    /**************************************/
   {
         Type ** d1 = I1.data();
         Type ** d2 = I2.data();
         Type ** d3 = I3.data();

         for (int x = 0; x < tx ; x++)
             for (int y = 0; y < ty ; y++)
                 if ((x>=p1.x) && (x<p2.x) && (y>=p1.y) && (y<p2.y))
                 {
                     if (
                               (d1[y][x] != (x+y) %max_val)
                           ||  (d2[y][x] != (x*x) %max_val)
                           ||  (d3[y][x] != (103*x+447) %max_val )
                        )
                        {
                           cout << " test_fich_2d   x, y \n" << x << " " << y << "\n";
                           cout << "d1 " << d1[y][x] <<" "<<(x+y)%max_val<<"\n";
                           cout << "d2 " << d2[y][x] <<" "<<(x*x)%max_val<<"\n";
                           cout << "d3 " << d3[y][x] <<" "<<(103*x+447)%max_val<<"\n";
                           exit(0);
                        }
                 }
                 else
                 {
                     if (
                               (d1[y][x] != (x) %max_val)
                           ||  (d2[y][x] != (y) %max_val)
                           ||  (d3[y][x] != (x+y) %max_val )
                        )
                        {
                             cout << " test_fich_2d   x, y \n" << x << " " << y;
                             exit(0);
                        }
                 }
   }


    /*************************/
    /*  verif of file with   */
    /*  def value            */
    /*************************/

    fp = Elise_fopen(name,"w");
    for(INT nb = 0; nb <offset_0 ; nb++)
       fputc(nb,fp);
    for (INT y = 0; y < ty ; y++)
        for (INT x = 0; x < tx ; x++)
        {
            buf[0] = 1;
            buf[1] = 2;
            buf[2] = 3;
            Elise_fwrite(buf,sizeof(Type),dim,fp);
        }
    Elise_fclose(fp);


    Type_Base s[3];
    p1  = Pt2di(-113,-186);
    p2  = Pt2di(153,146);
    INT  def_val = 22;
    INT S0 = tx * ty;
    INT S1 = (tx +p2.x -p1.x) * (ty +p2.y -p1.y);

    copy 
    (
         rectangle(p1,Pt2di(tx,ty)+p2),
         f.in(def_val),
         sigma(s,3)
    );

    BENCH_ASSERT
    (
              (s[0] == def_val *(S1 -S0) + 1 * S0)
         &&   (s[1] == def_val *(S1 -S0) + 2 * S0)
         &&   (s[2] == def_val *(S1 -S0) + 3 * S0)
    );
    

    /***************************************/
    /*      Writing in files               */
    /***************************************/

     p1 = Pt2di (12,8);
     p2 = Pt2di (tx-5,ty-6);

    copy 
    (
         I1.all_pts(),
         ((FX+1)%3,(FY+2)%5,(FX+FY+3)%7),
         f.out()
    );

    copy 
    (
         rectangle(p1,p2),
         ((FX+FY)%max_val,(FX*FY)%max_val,(square(FX)+square(FY))%max_val),
         f.out()
    );


    copy 
    (
         I1.all_pts(),
         f.in(),
         (I1.out(),I2.out(),I3.out())
    );
   {
         Type ** d1 = I1.data();
         Type ** d2 = I2.data();
         Type ** d3 = I3.data();

         for (int x = 0; x < tx ; x++)
             for (int y = 0; y < ty ; y++)
                 if ((x>=p1.x) && (x<p2.x) && (y>=p1.y) && (y<p2.y))
                 {
                     if (
                               (d1[y][x] != (x+y)%max_val)
                           ||  (d2[y][x] != (x*y)%max_val)
                           ||  (d3[y][x] != (x*x+y*y)%max_val)
                        )
                        {
                           cout << "test_fich_2d x,y\n" << x << " " << y << "\n";
                           cout << "d1 " << d1[y][x] <<" "<<(x+y)%max_val<<"\n";
                           cout << "d2 " << d2[y][x] <<" "<<(x*y)%max_val<<"\n";
                           cout << "d3 " << d3[y][x] <<" "<<(x*x+y*y)%max_val<<"\n";
                           exit(0);
                        }
                 }
                 else
                 {
                     if (
                               (d1[y][x] != (x+1) % 3)
                           ||  (d2[y][x] != (y+2) % 5)
                           ||  (d3[y][x] != (x+y+3) % 7)
                        )
                        {
                             cout << " test_fich_2d   x, y \n" << x << " " << y;
                             exit(0);
                        }
                 }
   }

   /* to create an empty file */
     fp = Elise_fopen(name,"w");
    Elise_fclose(fp);

}

void test_Elise_File_Format()
{
     test_EFF_Garb_Coll();
     test_fich_2d((INT2 *) 0,(INT *) 0,GenIm::int2,12340);
     test_fich_2d((REAL4 *) 0,(REAL *) 0,GenIm::real4,12340);

     cout << "OK Elise-Format simple \n";
}
