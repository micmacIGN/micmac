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




void bench_tiff (El_Window   W)
{
   Pt2di sz (90,90);

    L_Arg_Opt_Tiff l;

    l = l+ Tiff_Im::AStrip(15);
    l = l+ Tiff_Im::APlanConf(Tiff_Im::Planar_conf);


    Tiff_Im  Tif
    (
         "BENCH_FILE_IM/ex.tif",
         sz,
          GenIm::bits1_msbf,
         Tiff_Im::PackBits_Compr,
         Tiff_Im::BlackIsZero,
         l
    );

     ELISE_COPY
     (
        rectangle(Pt2di(0,0),sz),
        1,
        Tif.out()
     );



    INT dif;
    ELISE_COPY
    (
        rectangle(Pt2di(0,0),sz),
        Abs(1-Tif.in()),
        sigma(dif) | W.odisc()
    );

    cout << dif << "\n";
    BENCH_ASSERT(dif < epsilon);

}


void bench_tiff_im()
{
      Disc_Pal      Pdisc = Disc_Pal::P8COL();
      

      Elise_Set_Of_Palette SOP(newl(Pdisc));
      Video_Display Ecr((char *) NULL);
      Ecr.load(SOP);
      Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(720,720));

      El_Window W8 = W.chc(Pt2dr(0,0),Pt2dr(8,8));


      for (INT i = 0 ; i<100000000 ; i++)
      {
           cout << i << "\n";
           bench_tiff    ( W8);
      }
}




