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



//++++++++++++++++++++++++++++++++++++
//     DMR
//++++++++++++++++++++++++++++++++++++

class BenchDMR : public NROptF1vDer
{
     public :

          BenchDMR(INT deg);
          REAL NRF1v(REAL);
          REAL DerNRF1v(REAL);
          void test();
     private :

        ElPolynome<REAL> _pol;
        ElPolynome<REAL> _dpol;
};


REAL BenchDMR::NRF1v(REAL v)
{
    return _pol(v);
}

REAL BenchDMR::DerNRF1v(REAL v)
{
    return _dpol(v);
}

BenchDMR::BenchDMR(INT deg) :
    _pol  ((char *)0,deg),
    _dpol ()
{
    for (INT k=0 ; k< deg; k++)
        _pol[k] = (NRrandom3()-0.5)*10;
    _pol[deg] = NRrandom3()+0.5;

    _dpol = _pol.deriv();
}

void BenchDMR::test()
{

    REAL ax=-1,bx=1,cx,fa,fb,fc;
    mnbrack(&ax,&bx,&cx,&fa,&fb,&fc);

    REAL xmin1;
    golden(ax,bx,cx,1e-15,&xmin1);

    if (_pol.degre() <= 4)
       BENCH_ASSERT(ElAbs(_dpol(xmin1)) < BIG_epsilon);
    BENCH_ASSERT(ElAbs(_dpol(xmin1)) < GIGANTESQUE_epsilon);

    Pt2dr aP = brent(true);
    if (_pol.degre() <= 4)
       BENCH_ASSERT(ElAbs(_dpol(aP.x)) < BIG_epsilon);
    BENCH_ASSERT(ElAbs(_dpol(aP.x)) < GIGANTESQUE_epsilon);

}

void  bench_optim_DMR()
{
     for (INT k=0; k<5000 ; k++)
     {
         All_Memo_counter MC_INIT;
         stow_memory_counter(MC_INIT);  

         {
            BenchDMR BDMR((k%3+1)*2);
            BDMR.test();
         }

         verif_memory_state(MC_INIT); 
     }
}

void bench_optim_0()
{
    for (INT k=0 ; k<1 ; k++)
    {
        cout << "K= " << k << "\n";
        {
            All_Memo_counter MC_INIT;
            stow_memory_counter(MC_INIT);  

            bench_optim_DMR();

            verif_memory_state(MC_INIT); 
        }
        cout << "END DMR \n";

        {
            All_Memo_counter MC_INIT;
            stow_memory_counter(MC_INIT);  


            verif_memory_state(MC_INIT); 
        }
        cout << "END DGC \n";
    }

    cout << "End Bench Optim0 \n";
}

