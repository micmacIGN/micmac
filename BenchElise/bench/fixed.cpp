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


#include "/home/demo/ELISE/include/general/all.h"
#include "/home/demo/ELISE/include/private/all.h"
#include "/home/demo/ELISE/include/ext_stl/fifo.h"


ostream & operator << (ostream & ofs,const Pt2dr  &p)
{
      ofs << "[x " << p.x << " ;y " << p.y <<"]";
      return ofs;
}
ostream & operator << (ostream & ofs,const Pt2di  &p)
{
      ofs << "[x " << p.x << " ;y " << p.y <<"]";
      return ofs;
}



class Fixed
{
     protected :
         enum 
         { 
              b  = 8,
              b2 = 2*b,
              Q  = 1 << b, 
              Q2 = (1<<b2)
         };
};


class SFixed : Fixed
{
     private :
       SFixed  (INT  p) : _p (p)           {}
       INT _p;

     public :
       static SFixed Create(INT  p) { return SFixed(p<<b);}
       SFixed (REAL p) : _p ((INT)(p*Q)){}
       REAL r() const {return _p / (REAL) Q;}

       SFixed  ()       : _p (0)             {}

       SFixed operator - (SFixed f2) { return SFixed(_p-f2._p);}
       SFixed operator + (SFixed f2) { return SFixed(_p+f2._p);}
       SFixed operator * (SFixed f2) { return SFixed((_p*f2._p)>>b);}
       SFixed operator / (SFixed f2) { return SFixed((_p<<b)/f2._p);}

       SFixed operator - () { return SFixed(-_p);}

       friend ostream & operator << (ostream & ofs,const SFixed  &p)
       {
               ofs << p.r() ;
               return ofs;
       }
};


class PFixed : Fixed
{
     private :
        friend class SimFixed;


        PFixed(INT x,INT y)   : _x(x), _y(y) {}
        INT  _x;
        INT  _y;
        INT N2() const {return _x*_x+_y*_y;}

     public :

       PFixed ()       : _x (0), _y(0) {}
       PFixed (REAL  x,REAL y) : _x ((INT)(x*Q)) , _y ((INT)(y*Q)) {}
       PFixed (const Pt2dr & pt )       : _x ((INT)(pt.x*Q)) , _y ((INT)(pt.y*Q)) {}
       PFixed (const Pt2di & pt )       : _x (pt.x<<b) , _y (pt.y<<b) {}

       static PFixed Create(INT  x,INT y)   { return PFixed(x<<b,y<<b);}

        friend ostream & operator << (ostream & ofs,const PFixed  &p)
        {
               ofs << "[x " << p.x() << " ;y " << p.y() <<"]";
               return ofs;
        }

        REAL x() const {return _x / (REAL) Q;}
        REAL y() const {return _y / (REAL) Q;}

        PFixed operator + (const PFixed & p2) { return PFixed(_x+p2._x,_y+p2._y);}
        PFixed operator * (const PFixed & p2) 
        { 
              return PFixed
                     (  (_x*p2._x-_y*p2._y) >> b,
                        (_x*p2._y+_y*p2._x) >> b 
                     );
        }

         PFixed inv()
         {
                INT n2 = N2();
                return PFixed
                       (
                            (_x << b2)/n2,
                           -(_y << b2)/n2
                       );
         }

        PFixed operator / (const PFixed & p2) 
        {
                INT n2 = p2.N2();
                return PFixed
                       (    
                            (( _x*p2._x+_y*p2._y)<<b)/n2,
                            ((-_x*p2._y+_y*p2._x)<<b)/n2
                       );
        }

};



class SimFixed : Fixed
{
     private :
         PFixed  _tr;
         PFixed  _sc;

     public :

         SimFixed(const  PFixed  & Tr, const  PFixed & Sc) : _tr (Tr), _sc (Sc){}
         SimFixed(const ElSimilitude & s) : _tr (s.tr()), _sc (s.sc()) {}
         SimFixed operator * (const SimFixed & s2)
         {
              return  SimFixed(_tr+_sc*s2._tr,_sc*s2._sc);
         }
         PFixed operator () (PFixed pt) { return _tr + pt * _sc;}

         friend SimFixed rtranslate(const Pt2di & trans,const SimFixed & s)
         {
              return SimFixed(PFixed(trans)+s._tr,s._sc);
         }
};



class Bench_Float
{
    public :

       ElFilo<Pt2di>  _F;
       ElSimilitude   _s;
       SimFixed       _sf;
       Im2D_U_INT1    _I;
       U_INT1  **     _i;

       inline INT i(Pt2di p) { return _i[p.y][p.x];}

       Bench_Float() :
            _s(Pt2dr(3.2,1.1),Pt2dr(3.3,4.4)),
            _sf (_s),
            _I(100,100),
            _i (_I.data())
       {
           for (INT k=0; k<100; k++) _F.pushlast(Pt2di(10,10));
       }

       Pt2dr sim();
       PFixed fsim();
       INT   ind();
       virtual INT   ivirt();
       void       hand_i_sim();
       void       hand_r_sim();

       void bench_time(INT);


        INT    imul();
        REAL   rmul();
        INT    idiv();
        INT    ishift();
        REAL   rdiv();
};


void Bench_Float::bench_time(INT nb)
{
     INT k;

      ElTimer chrono;


      chrono.reinit();
      for (k=0; k<nb ; k++) fsim();
      cout << "FIXED SIM : " << chrono.val() << "\n";

      chrono.reinit();
      for (k=0; k<nb ; k++) sim();
      cout << "SIM : " << chrono.val() << "\n";





      chrono.reinit();
      for (k=0; k<nb ; k++) ishift();
      cout << "ISHIFT : " << chrono.val() << "\n";

      chrono.reinit();
      for (k=0; k<nb ; k++) idiv();
      cout << "IDIV : " << chrono.val() << "\n";

      chrono.reinit();
      for (k=0; k<nb ; k++) rdiv();
      cout << "RDIV : " << chrono.val() << "\n";




      chrono.reinit();
      for (k=0; k<nb ; k++) imul();
      cout << "IMUL : " << chrono.val() << "\n";

      chrono.reinit();
      for (k=0; k<nb ; k++) rmul();
      cout << "RMUL : " << chrono.val() << "\n";




      chrono.reinit();
      for (k=0; k<nb ; k++) ind();
      cout << "IND : " << chrono.val() << "\n";

      chrono.reinit();
      for (k=0; k<nb ; k++) ivirt();
      cout << "VIRT : " << chrono.val() << "\n";
}

Pt2dr Bench_Float::sim()
{
    ElSimilitude s(_s);
    ElSimilitude s0(_s);
    Pt2dr res;
    Pt2dr r2;
    Pt2di tr(1,2);

    INT f;
    for (f=0; f<10; f++)
    {
        s = s0 * s0;
        s = s0 * s0;
        s = s0 * s0;
        s = s0 * s0;
    }
    return res;
}

PFixed Bench_Float::fsim()
{
    SimFixed s(_sf);
    SimFixed s0(_sf);
    PFixed res;
    PFixed r2;
    Pt2di tr(1,2);

    INT f;
    for (f=0; f<10; f++)
    {
        s = s0 * s0;
        s = s0 * s0;
        s = s0 * s0;
        s = s0 * s0;
    }
    return res;
}



INT Bench_Float::ind()
{
    INT res = 0;
    Pt2di * t = _F.tab();
    for (INT k=0; k<_F.nb() ; k++)
        res +=  _i[t[k].y][t[k].x];
    return res;
}

INT Bench_Float::ivirt()
{
    return ind();
}

REAL Bench_Float::rmul()
{
    INT f ;
    REAL r =0;
    REAL i= 0, j= 0, k= 0;
    for (f= 0 ; f< 100 ; f++,i++,j++,k++)
    {
        r += i+j*k;
        r -= i*j+k;
        r += i+j+k;
        r -= i*j*k;
    }
    return r;
}

INT Bench_Float::imul()
{
    INT f ;
    INT r =0;
    INT i= 0, j= 0, k= 0;
    for (f= 0 ; f< 100 ; f++,i++,j++,k++)
    {
        r += i+j*k;
        r -= i*j+k;
        r += i+j+k;
        r -= i*j*k;
    }
    return r;
}

INT Bench_Float::idiv()
{
    INT f ;
    INT r =0;
    INT i= 1, j= 1, k= 1;
    for (f= 0 ; f< 100 ; f++,i++,j++,k++)
    {
        r += i+j/k;
        r -= i/j+k;
        r += i+j+k;
        r -= i/j/k;
    }
    return r;
}

INT Bench_Float::ishift()
{
    INT f ;
    INT r =0;
    INT i= 1, j= 1, k= 1;
    for (f= 0 ; f< 100 ; f++,i++,j++,k++)
    {
        r += i+(j<<k);
        r -= (i<<j)+k;
        r += i+j+k;
        r -= (i<<j)<<k;
    }
    return r;
}


REAL Bench_Float::rdiv()
{
    INT f ;
    REAL r =0;
    REAL i= 1, j= 1, k= 1;
    for (f= 0 ; f< 100 ; f++,i++,j++,k++)
    {
        r += i+j/k;
        r -= i/j+k;
        r += i+j+k;
        r -= i/j/k;
    }
    return r;
}




int main(int,char **)
{
    // PFixed  p = PFixed::Create(1.2,3.4);

    {
        REAL r1 = 3.1;
        REAL r2 = 4.57;
        INT  r3 = 44;
        SFixed f1(r1);
        SFixed f2(r2);
        SFixed f3 = SFixed::Create(r3);

        cout << r1 << " " << f1 << "\n";
        cout << r1 + r2 << " " << f1  +f2 << "\n";
        cout << r1 * r2 << " " << f1  *f2 << "\n";
        cout << r1 / r2 << " " << f1  /f2 << "\n";
        cout << r3 / r2 << " " << f3  /f2 << "\n";
    }

    {
        Pt2dr r1 (3.1,5.89);
        Pt2dr r2  (4.57,-12.37);
        Pt2di  r3 (44,-66);
        ElSimilitude SR(r1,r1);

        PFixed f1(r1);
        PFixed f2(r2);
        PFixed f3 = PFixed::Create(r3.x,r3.y);
        SimFixed     SF(f1,f1);

        cout << SF(f2) << " " << SR(r2) << "\n";
        cout << r1 << " " << f1 << "\n";
        cout << r1 + r2 << " " << f1  +f2 << "\n";
        cout << r1 * r2 << " " << f1  *f2 << "\n";
        cout << r1 / r2 << " " << f1  /f2 << "\n";
        cout << Pt2dr(r3) / r2 << " " << f3  /f2 << "\n";
    }


    Bench_Float bf;
    bf.bench_time(10000);

    return 0;
}

