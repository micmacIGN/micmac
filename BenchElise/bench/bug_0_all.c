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



#include "general/all.h"
#include "private/all.h"

/*
   0 :    overflow
   2 :    incompatible type in reduction
   3 :    division by 0 (integer)
   4 :    division bu 0 (real)
   5 :    out of bitmap when reading in RLE mode
   6 :    use of a coordinate function wirh a dimension incomatible with flux


   11 : Fonc_Num non init;
*/

void bug_0()
{
    Im2D_U_INT1 b(300,300);
    cout << "should be : OVERFLOW on U_INT1 \n";
    copy (b.all_pts(),FX,b.out());
}



void bug_2()
{
    REAL res;
    cout << "should be :  incompatible type in Out Reduction \n";
    copy(rectangle(Pt2di(-10,-13),Pt2di(103,105)),FY+FX*3,sigma(res));
}


void bug_3()
{
    INT res;
    cout << "should be :  division by 0 \n";
    copy(rectangle(Pt2di(-10,-13),Pt2di(103,105)),FY/FX,sigma(res));
}

void bug_4()
{
    REAL res;
    cout << "should be :  division by 0 \n";
    copy(rectangle(Pt2di(-10,-13),Pt2di(103,105)),FY/(FX+1.0),sigma(res));
}

void bug_5()
{
    INT res;
    Im2D_U_INT1 b(300,300);
    cout << "should be :  out of reading rle \n";
    copy(rectangle(Pt2di(-10,-13),Pt2di(103,105)),b.in(),sigma(res));
}


void bug_6()
{
    INT res;
    cout << "should be :  incompatible dimension in  fonction coordinate \n";
    copy(rectangle(Pt2di(-10,-13),Pt2di(103,105)),FZ,sigma(res));
}


void bug_7()
{
    Im2D_U_INT1 b(300,300);
    cout << "should be :  out of domain in non RLE \n";
    copy(select(rectangle(Pt2di(-10,-13),Pt2di(103,105)),1),1,b.out());
}

void bug_8()
{
    Im2D_U_INT1 b(300,300);
    cout << "should be :  out of domain in non RLE \n";
    copy(select(rectangle(Pt2di(-10,-13),Pt2di(103,105)),1),1,b.onotcl());
}

void bug_9()
{
    Im2D_U_INT1 b(300,300);
    cout << "should be :  out of domain in  RLE \n";
    copy(rectangle(Pt2di(-10,-13),Pt2di(103,105)),1,b.onotcl());
}

void bug_10()
{
    INT res;
    Im2D_U_INT1 b(300,300);
    cout << "should be :  out of reading in integer mode\n";
    copy(select(rectangle(Pt2di(-10,-13),Pt2di(103,105)),1),b.in(),sigma(res));
}


void bug_11()
{
    Fonc_Num f;
    INT res;
    cout << "should be :  Fonc_Num non initialized\n";
    copy(rectangle(Pt2di(-10,-13),Pt2di(103,105)),f,sigma(res));
}

void bug_12()
{
    Flux_Pts flx;
    INT res;
    cout << "should be :  Flux_Pts non initialized\n";
    copy(flx,1,sigma(res));
}

void bug_13()
{
    Output o;
    cout << "should be :  OutPut non initialized\n";
    copy(rectangle(Pt2di(-10,-13),Pt2di(103,105)),1,o);
}


void bug_14()
{
    cout << "should be :   Symb_FNum is used in a context different from ... \n";

    Symb_FNum xpy (FX>FY);
    INT s;
    copy
    (
          select(rectangle(Pt2di(-10,-13),Pt2di(103,105)), !(xpy)),
          xpy,
          sigma(s)
    );
}


main(int argc,char ** argv)
{

   ASSERT_TJS_USER(argc>=2,"Not enouh arg in bug_0_all\n");
   int nbug = atoi(argv[1]);

   cout << "Num de bug teste : " << nbug << "\n";

   switch (nbug)
   {
        case 0 :
             bug_0();
        break;


        case 2 :
             bug_2();
        break;

        case 3 :
             bug_3();
        break;

        case 4 :
             bug_4();
        break;

        case 5 :
             bug_5();
        break;

        case 6 :
             bug_6();
        break;

        case 7 :
             bug_7();
        break;

        case 8 :
             bug_8();
        break;

        case 9 :
             bug_9();
        break;

        case 10 :
             bug_10();
        break;

        case 11 :
             bug_11();
        break;

        case 12 :
             bug_12();
        break;

        case 13 :
             bug_13();
        break;

        case 14 :
             bug_14();
        break;
   }

   cout << "APPARENTLY NO BUG APPEARED ??????\n";
}



