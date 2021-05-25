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


class InsertCopyright : public ElActionParseDir
{
   public :

     InsertCopyright
     (
          char * DirM,    // maitre
          char * DirE    // esclave
     ) :
       dirM (DirM),
       dirE (DirE)
     {}


     virtual void act(const ElResParseDir &);

     char  * dirM;
     char  * dirE;

   private :
};

void InsertCopyright::act(const ElResParseDir & erpd)
{
    if (erpd.is_dir()) return;

    char nM[300];
    char nE[300];

    sprintf(nM,"%s%s",dirM,erpd.sub(1));
    sprintf(nE,"%s%s",dirE,erpd.sub(1));

    cout << nE << " => " << nM << "\n";

    ELISE_fp  Fe(nE,ELISE_fp::WRITE);

    for (int i = 0; i< 2; i++)
    {
          char * n = (i==0) ? "copyright" : nM;
          ELISE_fp  FM(n,ELISE_fp::READ);

          INT c;
          while ( (c = FM.fgetc()) != ELISE_fp::eof)
               Fe.write_U_INT1(c);
                
          FM.close();
    }
    Fe.close();
}



void insert_copyright(char * dir_maitre)
{
    char dir_esclave[200];
    sprintf(dir_esclave,"DUP_ELISE/%s",dir_maitre);

    InsertCopyright IC(dir_maitre,dir_esclave);

    ElParseDir(dir_maitre,IC);
}


int main(int,char**)
{
     insert_copyright("src/");
     insert_copyright("include/");
}

   

