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


class BugAV 
{
    public :
       BugAV 
       (
           Im2D_U_INT1   
       )  
       {}

    private  :

};
Fonc_Num TOTO1(BugAV ){return 0;}

void TATA( Fonc_Num  fonc_num ,Output output)
{
}

void  bench_bugvecto ()
{
	cout << "BEFOE IM \n";
    Im2D_U_INT1 Ifl(10,10,255);


cout << "-----------------------bench_bugvecto AAAA\n";

    cout << Ifl.ptr()  << "\n";

    INT aBid;
    TATA
    (
      TOTO1  (  BugAV (Ifl)),
      VMax(aBid)
    );


cout << "bench_vecto BBBBB\n";
exit(-1);
}




















