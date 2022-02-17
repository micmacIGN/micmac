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


All_Memo_counter MC_INIT;

const char * name_in =  "../IM_ELISE/BTIFF/f0.tif";
const char * name_out = "../IM_ELISE/BTIFF/b0.tif";

const REAL ang_degr  = -5.0;

/*********** UTIL ********************************/

class Mapping
{
    public :

      virtual Fonc_Num  OtoI(Fonc_Num) const = 0;
      virtual Fonc_Num  ItoO(Fonc_Num) const = 0;
};

class Map_tr_io : public Mapping
{
    public :

      Map_tr_io(const Mapping & map,Pt2dr tr) :
          _tr   (tr),
          _map  (map)
      {
      }

      Fonc_Num  OtoI(Fonc_Num fxy) const
      {
             return _map.OtoI(fxy-Fonc_Num(_tr.x,_tr.y));
      }

      Fonc_Num  ItoO(Fonc_Num fxy) const 
      {
             return _map.ItoO(fxy)+ Fonc_Num(_tr.x,_tr.y);
      }

   private :
      Pt2dr _tr;
      const Mapping & _map;
};


void test (const Mapping & map_ori)
{
    Tiff_Im  Im_in(name_in);
    Im_in.show();

    Pt2di szi = Im_in.sz();  // size input
    Pt2di p00(0,0);

    GenIm::type_el type_buf = type_u_int_of_nbb(Im_in.bitpp());

    /*================================================

         Compute the bounding box [s0,s1] of ,input bounding
         box [(0,0),sz], by mapping ItoO

    =================================================*/

    
    Pt2dr s0,s1;
    ELISE_COPY
    (
        rectangle(p00,szi),
        Rconv(map_ori.ItoO((FX,FY))),
          (VMin(s0.x),VMin(s0.y))
        | (VMax(s1.x),VMax(s1.y))
    );

    // add some rab. To be sure not to loose any pixel

    s0 = s0 - Pt2dr(1.0,1.0);
    s1 = s1 + Pt2dr(1.0,1.0);

    Pt2di szo = s1-s0+Pt2di(1,1);   // size output
 
    if (0)
    {
        std::cout << "s0  : " << s0.x  << " " << s0.y << "\n";
        std::cout << "s1  : " << s1.x  << " " << s1.y << "\n";
        std::cout << "szi : " << szi.x << " " << szi.y << "\n";
        std::cout << "szo : " << szo.x << " " << szo.y << "\n";
    }
    

    // define the mapping map, that adds a translation to map_ori,
    // so that the image of input rectangle has coordinate > 0

    Map_tr_io map(map_ori,-s0);

    if (0)  // some verification on new mapping
    {
        Pt2dr s0,s1;
        ELISE_COPY
        (
            rectangle(p00,szi),
            Rconv(map.ItoO((FX,FY))),
              (VMin(s0.x),VMin(s0.y))
            | (VMax(s1.x),VMax(s1.y))
        );
        std::cout << "s0/ tr : "<< s0.x << " " << s0.y << "\n";
        std::cout << "s1/ tr : "<< s1.x << " " << s1.y << "\n";


        REAL dx,dy;
        ELISE_COPY
        (
            rectangle(p00,szi),
            Abs(map.OtoI(map.ItoO((FX,FY)))-(FX,FY)),
            (VMax(dx),VMax(dy))
        );
        std::cout << "Verif Inverse : "<< dx << " " << dy << "\n";
    }

    /*================================================

         Compute the interval necessary to buffering in y.

    =================================================*/

    
    INT dy_min,dy_max;
    ELISE_COPY
    (
        rectangle(p00,szi),
        round_ni(map.OtoI((FX,FY))).v1()-FY,
        VMin(dy_min) | VMax(dy_max)
    );
    dy_min -= 2;
    dy_max += 3;
    INT dy = dy_max-dy_min;

    std::cout << "Interv y buf : "<< dy_min << " " << dy_max << "\n";


    GenIm buf = alloc_im2d(type_buf,szi.x,dy_max-dy_min);

    {
        Gray_Pal       PGray (100);
        Disc_Pal       Pdisc = Disc_Pal::P8COL();
        Elise_Set_Of_Palette SOP(newl(PGray)+Pdisc);
        Video_Display Ecr((char *) NULL);
        Ecr.load(SOP);
        Video_Win   W(Ecr,SOP,Pt2di(20,20),szo);

        GenIm verif = alloc_im2d(type_buf,szi.x,szi.y);
        ELISE_COPY
        (
            W.all_pts(),
            Im_in.in(2),
            W.odisc() | verif.out()
        );

/*
        ELISE_COPY
        (
            W.all_pts(),
            verif.in(2)[map.OtoI((FX,FY))],
            W.odisc() 
        );
*/


        ELISE_COPY
        (
            buf.all_pts(),
            Im_in.in(0),
            buf.out()
        );

        Symb_FNum  fxy (Iconv(map.OtoI((FX,FY))));



        ELISE_COPY
        (
            rectangle(p00,szo),
            trans(Im_in.in(0),Pt2di(0,dy_max)),
                buf.oclip().chc((FX,(FY+dy_max)%dy))
            |   (
                         W.odisc()
                    <<   clip_def
                         (
                             buf.in()[FX,mod(FY,dy)],
                             3,
                             p00,
                             szi
                         ) [fxy.v0(),fxy.v1()]
                )
        );



    }

    
    getchar();
}

class RotMap : public Mapping
{
    public :

      RotMap(REAL ang) : _ang (ang) {};

      virtual Fonc_Num  OtoI(Fonc_Num fxy)  const
      {
             return mulc(fxy,Fonc_Num(cos(_ang),sin(_ang)));
      }

      virtual Fonc_Num  ItoO(Fonc_Num fxy) const
      {
             return mulc(fxy,Fonc_Num(cos(-_ang),sin(-_ang)));
      }



   private :
       REAL _ang;
};




main(int,char *)
{
     ELISE_DEBUG_USER = true;
     stow_memory_counter(MC_INIT);

     {
         RotMap  rot(ang_degr * (PI/180.0));

         test(rot);
     }
     std::cout << "OK BENCH 0 \n";
     verif_memory_state(MC_INIT);
}








