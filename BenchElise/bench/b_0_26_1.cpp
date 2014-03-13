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

void RmBfiTmpFiles()
{
#if (! ELISE_windows)
	if (ELISE_fp::exist_file(ELISE_BFI_DATA_DIR "tmp"))
	{
		std::string ToExec(RM ELISE_BFI_DATA_DIR "tmp");
		ncout() << "Execute " << ToExec.c_str() << "\n";
		TheIntFuckingReturnValue=system(ToExec.c_str());
		ncout() << "Do Execute \n";
	}
#endif
}

Im2DGen  alloc_im_2D_Of_Type(Pt2di sz,GenIm::type_el type)
{
    switch(type)
    {
         case  GenIm::u_int1:
               return Im2D_U_INT1(sz.x,sz.y);

         case  GenIm::int1:
               return Im2D_INT1(sz.x,sz.y);

         case  GenIm::u_int2:
               return Im2D_U_INT2(sz.x,sz.y);

         case  GenIm::int2:
               return Im2D_INT2(sz.x,sz.y);

         case  GenIm::int4:
               return Im2D_INT4(sz.x,sz.y);

         case  GenIm::real4:
               return Im2D_REAL4(sz.x,sz.y);

         case  GenIm::real8:
               return Im2D_REAL8(sz.x,sz.y);

         case  GenIm::bits1_msbf:
         case  GenIm::bits1_lsbf:
               return Im2D_Bits<1>(sz.x,sz.y);

         case  GenIm::bits2_msbf:
         case  GenIm::bits2_lsbf:
               return Im2D_Bits<2>(sz.x,sz.y);

         case  GenIm::bits4_msbf:
         case  GenIm::bits4_lsbf:
               return Im2D_Bits<4>(sz.x,sz.y);

         default :;
    }
    BENCH_ASSERT(0);
    return Im2D_U_INT1(sz.x,sz.y);
    
} 



class Bench_tiles_elise_file
{
   public :

       void DoNoting() {};

       Bench_tiles_elise_file
       (
         Pt2di            sz,
         GenIm::type_el   type,
         INT              dim_out,
         Pt2di            sz_tiles,
         bool             clip_last_tile,
         bool             chunk          ,
         INT              offset_0          
      );
};



Bench_tiles_elise_file::Bench_tiles_elise_file
(
         Pt2di            sz,
         GenIm::type_el   type,
         INT              dim_out,
         Pt2di            sz_tiles,
         bool             clip_last_tile,
         bool             chunk          ,
         INT              offset_0          
)
{
    static INT num =0;
    num++;

    INT nbb = nbb_type_num(type) ;
    if (nbb < 8)
        dim_out =1;
   
/*
cout << "No " << num << " OFFS  " <<  offset_0 << " ; DIM : " 
     << dim_out << " "
     << " Tiles x = " << round_up((sz.x-0.5)/(REAL)sz_tiles.x)
     << "Nb File Opened = " << aEliseCptFileOpen
     << "\n";

cout <<  "Type " << (INT) type << "\n";
*/
	RmBfiTmpFiles();


    Elise_Tiled_File_Im_2D  
         ETF
         (
               ELISE_BFI_DATA_DIR "tmp",
               sz,
               type,
               dim_out,
               sz_tiles,
               clip_last_tile,
               chunk,
               offset_0,
               true
         );
    ELISE_fp fp(ELISE_BFI_DATA_DIR "tmp",ELISE_fp::READ_WRITE);
    fp.seek_begin(offset_0);

    Im2DGen I_TILE =  alloc_im_2D_Of_Type(sz_tiles,type);
    INT vmax = 1 << ElMin(15,nbb);

    if (signed_type_num(type)) vmax /= 2;

    Fonc_Num fr0 = Iconv(frandr()*10000)%vmax;
    Fonc_Num FRAND = fr0;

    ELISE_COPY
    (
        I_TILE.all_pts(),
        FRAND,
        I_TILE.out()
    );

    Im2D_REAL8 IGLOB(sz.x,sz.y);
    Fonc_Num   Bin   = IGLOB.in(0);
    Output     Bout  = IGLOB.out();

     ELISE_COPY
     (
            IGLOB.all_pts(),
            I_TILE.in()[Virgule(FX%sz_tiles.x,FY%sz_tiles.y)],
            IGLOB.out() 
      );

	INT d;
    for (d =1; d<dim_out; d++)
    {
        Im2D_REAL8 I2(sz.x,sz.y);
        ELISE_COPY(IGLOB.all_pts(),IGLOB.in(),I2.out());
        Bin  = Virgule(Bin,I2.in(0));
        Bout = Virgule(Bout,I2.out());
        FRAND = Virgule(FRAND,fr0);
    }


    for (INT pl=0; pl<(chunk?1:dim_out) ; pl++)
         for (INT y=0; y<sz.y ; y+=sz_tiles.y)
              for (INT x=0; x<sz.x ; x+=sz_tiles.x)
              {
                  INT dR =  chunk ? dim_out : 1;
                  Pt2di sz_loc ;
                  if (clip_last_tile)
                     sz_loc = Inf(sz_tiles,sz-Pt2di(x,y));
                  else
                     sz_loc = sz_tiles;

                   sz_loc.x *= dR;

                   INT pad_contr = (nbb >=8) ? 1 : (8/nbb);
                   sz_loc.x = ((sz_loc.x+pad_contr-1)/pad_contr)*pad_contr;
             
                   Im2DGen itile =  alloc_im_2D_Of_Type(sz_loc,type);


                   if ((nbb>=8) || msbf_type_num(type))
                       ELISE_COPY
                       (
                            itile.all_pts(),
                            I_TILE.in(0)[Virgule(FX/dR,FY)],
                            itile.out()
                       );
                   else
                   {
                       ELISE_COPY
                       (
                            itile.all_pts(),
                            I_TILE.in(0)[Virgule(FX/dR,FY)],
                            itile.out().chc
                             (Virgule((FX/pad_contr)*pad_contr +pad_contr-1-(FX%pad_contr),FY))
                       );
                   }
                   void * data = itile.data_lin();
              

                   INT tx = ((sz_loc.x+pad_contr-1) / pad_contr) * pad_contr;
                   INT byte_x = (tx * nbb) / 8;
                   INT byte_xy  = byte_x * sz_loc.y;

              
                   fp.write(data,1,byte_xy);
              }

    fp.close();

    INT dif[20];
    ELISE_COPY
    (
        IGLOB.all_pts(),
        Abs(Bin-ETF.in()),
        VMax(dif,dim_out)
    );

    for (d=0; d<dim_out;d++)
    {
        BENCH_ASSERT(dif[d]==0);
    }


	RmBfiTmpFiles();
}

static INT randi(INT v)
{
     return ((INT) (NRrandom3() * v)) % v;
}
static Pt2di Prandi(INT v)
{
    return Pt2di(randi(v),randi(v));
}

static bool PileOuFace() { return NRrandom3() >  0.5;}

void bench_tiles_elise_file()
{
    GenIm::type_el TabType[13] =
    {
        GenIm::u_int1, GenIm::int1,
        GenIm::u_int2, GenIm::int2,
        GenIm::int4,
        GenIm::real4, GenIm::real8,
        GenIm::bits1_msbf, GenIm::bits2_msbf, GenIm::bits4_msbf,
        GenIm::bits1_lsbf, GenIm::bits2_lsbf, GenIm::bits4_lsbf
    };


     for (INT i=0 ; i< 200; i++)
     {
         Bench_tiles_elise_file TOTO
         (
             Pt2di(50,50)+ Prandi(20),
             TabType[i%13],
             1+randi(3),
             Pt2di(5,5)+Prandi(5),
             PileOuFace(),
             PileOuFace(),
             randi(5)*10
         );
         TOTO.DoNoting();

    }
    cout << "END  bench_tiles_elise_file \n";
    
}





