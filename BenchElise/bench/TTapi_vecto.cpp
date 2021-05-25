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



#include "api/vecto.h"
#include "stdio.h"


class MyVectoAct : public EL_CALL_BACK_VECTO
{
      public :

           void  ElVectoAction
                 (
                     ComplexI p0,
                     ComplexI p1,
                     const std::vector<ComplexI> & pts_interm,
                     const std::vector<int> & d_interm
                 )
            {
                cout << p0.real() << " " ;
                cout << p0.imag() << ";";
                cout << p1.real() << " " ;
                cout << p1.imag() << "\n";
            }

           MyVectoAct () { }

      private :
};



void t0 (ComplexI sz)
{
     char * name = "../IM_CAD/MOUSSAN_AA/Moussan_AA.tif";

     EL_API_VECTO::ParamSkel     PSkel;
     EL_API_VECTO::ParamApprox   PApp;

     EL_API_VECTO  API(PSkel,PApp,20);

     EL_API_VECTO::ParamFile PFile(name,false);


     int x,y;
     scanf("%d %d",&x,&y);

     MyVectoAct MVA; 

     API.vecto
     (
          PFile,
          MVA,
          ComplexI(x,y),
          sz
     );
}

int main(int,char**)
{
     int SZX = 200;
     int SZY = 150;

     while (true) t0 (ComplexI(SZX,SZY));
}




