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



// OFFS = 129608
// OFFS END = 0

#include "general/all.h"
#include "private/all.h"


class TransCoder
{
    public :

         void init() { _nb =0;}
         TransCoder
         (
             int       NbPat,
             char **   Input,
             char *    Out
         )  :
            _nbpat   (NbPat),
            _inputs (Input),
            _out    (Out)
         {
             init();
         }
         void coder(char * str);
         void filecoder(const char * str);

         bool equal(const TransCoder & tc) const
         {
             return  (tc._nb == _nb) && (!strncmp(tc._res,_res,_nb));
         }

         void show()
         {
           for(int k=0; k<_nb ; k++)
              cout << _res[k];
           cout << "\n";
         }
         
    private :
         char        _res[10000];
         int         _nb;
         int         _nbpat;
         char * *    _inputs;
         char *      _out;
};

void TransCoder::coder(char *str)
{
     for(; *str; str++)
     {
         for (INT pat=0 ; pat<_nbpat; pat++)
             if (! strncmp(str,_inputs[pat],strlen(_inputs[pat])))
                _res[_nb++] = _out[pat];
     }
}
void TransCoder::filecoder(const char *str)
{
     init();
     ELISE_fp fp(str,ELISE_fp::READ);

     static const int nb_buf = 10000;
     char buf[nb_buf];
     while(1)
     {
        // si ligne trop grande, pas un fichier texte
        bool endof;
        if (! fp.fgets(buf,nb_buf,endof,true))
        {
             init();
             return;
        }
        if(endof)
        {
           fp.close();
           return;
        }
        coder(buf);
     }
}



class STLsubst : public ElActionParseDir
{
   public :

     STLsubst
     (
          char * DirM,    // maitre
          TransCoder & CM,
          char * DirE,    // esclave
          TransCoder & CE,
          int    
     ) :
       dirM (DirM),
       cm   (CM),
       dirE (DirE),
       ce   (CE)
     {}


     virtual void act(const ElResParseDir &);

     char  * dirM;
     TransCoder & cm;
     char  * dirE;
     TransCoder & ce;

   private :
};



void STLsubst::act(const ElResParseDir & res)
{
  if (res.is_dir())
     return;
   char buf[1000];
   sprintf(buf,"%s/%s",dirE,res.sub(1));

   if (! ELISE_fp::exist_file(buf))
   {
       cout << buf << "\n";
       return;
   }
   cm.filecoder(res.name());
   ce.filecoder(buf);

   if (!cm.equal(ce))
   {
       cout << "DIFF FOR : " << res.name() <<  "\n";
       cm.show();
       ce.show();
   }
}



int main(int,char**)
{
/*
   const int Nbmax = 4;
   char * (Inmax[Nbmax]) ={"max","min","swap","list"};
   char * Outmax = "AISL";
   TransCoder Cmax (Nbmax,Inmax,Outmax);

   const int NbElMax = 8;
   char * (InElMax[NbElMax]) ={"max","min","swap","list","ElMax","ElMin","ElSwap","ElList"};
   char * OutElMax = "AISLAISL";
   TransCoder CElMax (NbElMax,InElMax,OutElMax);


   Cmax.init();
   Cmax.filecoder("include/general/all.h");
   Cmax.show();

   CElMax.init();
   CElMax.filecoder("include/general/all.h");
   CElMax.show();


   STLsubst subst
   (
       "Vmaxmin/src", Cmax,
       "src" ,   CElMax,
        44
   );
   
   ElParseDir(subst.dirM,subst);

   return 0;
*/
   Tiff_File("../xvcol24.tif").nb_im();
}

   

