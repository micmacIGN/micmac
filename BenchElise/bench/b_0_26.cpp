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


void bench_egfi
     (
          ElGenFileIm    egfi,
          INT            Dim,
          const INT *    Sz,
          INT            NbChannel,
          bool           SigneType,
          bool           IntegralType,
          INT            NbBits,
          const INT *    SzTile,
          bool           Compressed,
          Fonc_Num       Inside
     )
{

        BENCH_ASSERT
        (
                (egfi.Dim() == Dim)
           &&   (egfi.NbChannel()== NbChannel)
           &&   (egfi.SigneType()== SigneType)
           &&   (egfi.IntegralType()== IntegralType)
           &&   (egfi.NbBits()== NbBits)
           &&   (egfi.Compressed()== Compressed)
        );

        INT nb_pts_Theo = 1;

		INT dim;
        for ( dim = 0; dim<Dim ; dim++)
        {
            BENCH_ASSERT
            (
                     (egfi.Sz()[dim] == Sz[dim])
                &&   (egfi.SzTile()[dim] == SzTile[dim])
            );
            nb_pts_Theo  *= Sz[dim]; 
        }

        INT  cmax[200];
        INT  cmin[200];
        INT  nbpts;


        ELISE_COPY
        (
              egfi.all_pts(),
              Identite(Dim),
                 VMax(cmax,Dim) 
              |  VMin(cmin,Dim) 
              |  (sigma(nbpts)<<1)
        );

        BENCH_ASSERT(nbpts == nb_pts_Theo);
        for ( dim = 0; dim<Dim ; dim++)
        {
            BENCH_ASSERT
            (
                  (cmin[dim] == 0)
             &&   (cmax[dim] == Sz[dim]-1)
            );
            cmin[dim] -= 2+ 3* dim;
            cmax[dim] += 10 - dim;
        }

        REAL8 dif_inside;
        ELISE_COPY
        (
            rectangle(cmin,cmax,Dim),
            Abs(Inside-egfi.inside()),
            VMax(dif_inside)
        );

        BENCH_ASSERT
        (
             (dif_inside <epsilon)
        );

}


void bench_Elise_File_Im_EGFI
     (
           INT                dim,    
           GenIm::type_el     type_el,     
           INT                dim_out
     )
{
   INT SZ[10] = {5,4,3,2,2,2,2,2,2,2};
   const char * name =  ELISE_BFI_DATA_DIR "file.elise";

   Elise_File_Im   EFI
                   (
                      name,
                      dim,
                      SZ,
                      type_el,
                      dim_out,
                      (INT)(10+20*NRrandom3()),
                      -1,
                      true
                   );
   Fonc_Num f = frandr();
   for (INT k =1 ; k<dim_out; k++)
       f = Virgule(f,frandr());
   ELISE_COPY
   (
         rectangle(PTS_00000000000000,SZ,dim),
         f,
         EFI.out()
   );

   bench_egfi
   (
         EFI,
         dim,
         SZ,
         dim_out,
         signed_type_num(type_el),
         type_im_integral(type_el),
         nbb_type_num(type_el),
         SZ,
         false,
         EFI.inside()
   );
}
        
void bench_Elise_File_Im_EGFI()
{
      bench_Elise_File_Im_EGFI(1,GenIm::bits1_msbf,1);
      bench_Elise_File_Im_EGFI(2,GenIm::u_int1    ,2);
      bench_Elise_File_Im_EGFI(4,GenIm::real8     ,5);
}


void bench_GIF_EGFI
     (
           Pt2di           sz,
           INT             nbb
     )
{
   const char * name =  ELISE_BFI_DATA_DIR "file.gif";
   Elise_colour cols[256];

   for (int k=0; k<256; k++)
   {
       REAL g = k/256.0;
       cols[k] = Elise_colour::rgb(g,g,g);
   }

   ELISE_COPY
   (
        rectangle(Pt2di(0,0),sz),
        frandr()*((1<<nbb)-1),
        Gif_Im::create
        (
            name,
            sz,
            cols,
            nbb
        )
   );
   Gif_Im GIF(name);

   INT SZ[2];
    sz.to_tab(SZ);
   bench_egfi
   (
         GIF,
         2,
         SZ,
         1,
         false,
         true,
         ElMax(4,nbb),
         SZ,
         true,
         inside(Pt2di(0,0),sz)
   );
}
void bench_GIF_EGFI()
{
      bench_GIF_EGFI(Pt2di(20,30),1);
      bench_GIF_EGFI(Pt2di(30,20),2);
      bench_GIF_EGFI(Pt2di(30,40),3);
      bench_GIF_EGFI(Pt2di(30,40),4);
      bench_GIF_EGFI(Pt2di(30,40),8);
}


double  rand_tiffc()
{
   return 
          round_ni(Tiff_Im::MAX_COLOR_PAL * NRrandom3())
      /   (REAL) Tiff_Im::MAX_COLOR_PAL;
}

void bench_tiff
     (
          Pt2di                    TileFile,
          Pt2di                    sz,
          Pt2di                    szt,
          INT                      nb_ch,
          GenIm::type_el           type,
          Tiff_Im::COMPR_TYPE      compr,
          bool                     std_pl_conf,
          Pt2di                    bloc,
          bool                     indexed_colour = false
     )
{
   static INT cpt = 0;
    cout << "COMPR = " << (INT) compr 
	 <<  "[NUM = " << (++cpt) << "]" 
	 <<  " Type = " << type
	 << "\n";

   REAL def_val = type_im_integral(type) ? 2.0 : 2.5;

   if (
           (Tiff_Im::mode_compr_bin(compr))
         && (type!= GenIm::bits1_msbf)
         && (type!= GenIm::bits1_lsbf)
      )
          type =  GenIm::bits1_msbf;
    
   if (
             (Tiff_Im::mode_compr_bin(compr))
         ||  (compr == Tiff_Im::MPD_T6)
      )
      nb_ch =1;

    L_Arg_Opt_Tiff l;
    if (TileFile.x >0 )
	    l = l +Arg_Tiff(Tiff_Im::AFileTiling(TileFile));
    if (szt.x != sz.x)
       l = l+ Arg_Tiff(Tiff_Im::ATiles(szt));
    else if (szt.y != sz.y)
    {
       l = l+ Arg_Tiff(Tiff_Im::AStrip(szt.y));
    }
    else
       l = l + Arg_Tiff(Tiff_Im::ANoStrip());

    Tiff_Im::PH_INTER_TYPE  ph_interp      =
                 (nb_ch==3)                ?
                 Tiff_Im::RGB              :
                 Tiff_Im::BlackIsZero      ;

    Tiff_Im::PLANAR_CONFIG  pl_conf        =
                  std_pl_conf              ?
                  Tiff_Im::Chunky_conf     :
                  Tiff_Im::Planar_conf     ;

    l = l+ Arg_Tiff(Tiff_Im::APlanConf(pl_conf));

    Tiff_Im  Tif
    (
         ELISE_BFI_DATA_DIR "ex.tif",
         sz,
         type,
         compr,
         ph_interp,
         l
    );

    Elise_colour * tabc = 0;
    int nb_col =  0;

    if (indexed_colour)
    {
       nb_col = 1 << Tif.bitpp();
       tabc = NEW_VECTEUR(0,nb_col,Elise_colour);
       for (int i =0; i< nb_col ;i++)
           tabc[i] = Elise_colour::rgb
                     (
                           rand_tiffc(),
                           rand_tiffc(),
                           rand_tiffc()
                     );

      Disc_Pal  P0(tabc,nb_col);
      Tif = Tiff_Im
            (
                   ELISE_BFI_DATA_DIR "ex.tif",
                   sz,
                   type,
                   compr,
                   P0,
                   l
            );
        
    }

    {
           INT SZ[2],SZ_TILE[2];
           sz.to_tab(SZ);
           szt.to_tab(SZ_TILE);

           bench_egfi
           (
                Tif,
                2,
                SZ,
                nb_ch,
                signed_type_num(type),
                type_im_integral(type),
                nbb_type_num(type),
                SZ_TILE,
                compr != Tiff_Im::No_Compr,
                Tif.inside()
           );
    }



    INT nbv = 1 << Tif.bitpp();
    if (!type_im_integral(type))
       nbv = 1 << 15;

    INT minv = (
                        type_im_integral(type)
                     && signed_type_num(type)
                )                                ?
                -(nbv/2)                          :
                0                                ;
    INT maxv = minv+nbv-1;

    Im2D_REAL8 I1(sz.x,sz.y,0.0);
    Im2D_REAL8 I2(sz.x,sz.y,0.0);
    Im2D_REAL8 I3(sz.x,sz.y,0.0);

    Fonc_Num  f0 =  (Iconv(frandr()*nbv) %  nbv) +minv;

    Fonc_Num finit  = 0;
    Output   Overif = I1.out();
    Fonc_Num Fverif = I1.in();

    switch(nb_ch)
    {
         case 1 : 
             finit = f0;
             Overif = I1.out();
             Fverif = I1.in(def_val);
         break;

         case 3 : 
             finit = Virgule(f0,f0,f0);       
             Overif = Virgule(I1.out(),I2.out(),I3.out());
             Fverif = Virgule(I1.in(def_val) , I2.in(def_val) , I3.in(def_val));
         break;
    };

    if ((minv >=0) && (maxv<=256))
       finit = rect_median(finit,4,maxv+1);

    {
       INT dx =  bloc.x *szt.x;
       INT dy =  bloc.y *szt.y;

       for (INT x =0; x<sz.x ; x+=dx)
           for (INT y =0; y<sz.y ; y+=dy)
           {
               Output ofile = Tif.out();
               if (NRrandom3() > 0.5)
               {
                    ElGenFileIm egfi = Tif;
                    ofile = egfi.out();
               }
               ELISE_COPY
               (
                   rectangle(Pt2di(x,y),Pt2di(x,y)+Pt2di(dx,dy)),
                   finit,
                   Overif | ofile
               );
           }
    }

    if (compr==Tiff_Im::No_Compr)
    {
        Fonc_Num fd =0;
        Output ofile = Tif.out();
        if (NRrandom3() > 0.5)
        {
            ElGenFileIm egfi = Tif;
            ofile = egfi.out();
        }
        switch(nb_ch)
        {
              case 1 :  fd = Fonc_Num(maxv/2);  break;
              case 3 :  fd = Fonc_Num(0,maxv,0); break;
        };
        ELISE_COPY
        (
            //disc(sz/2+Pt2di(NRrandom3(8)-4,NRrandom3(8)-4),20),
            disc( Pt2dr( (REAL)( sz.x/2+NRrandom3(8)-4 ), (REAL)( sz.y/2+NRrandom3(8)-4 ) ), 20 ), // __NEW
            fd,
            Overif  | ofile
        );
    }


    REAL dif[10];
    ELISE_COPY
    (
        rectangle(Pt2di(-10,-20),Pt2di(3,4)+sz),
        Abs(Fverif-Tif.in(def_val)),
        sigma(dif,nb_ch) 
    );

    REAL dtot = 0.0;
    for (INT i=0 ; i<nb_ch ; i++)
    {
        dtot += dif[i];
    }

    
    BENCH_ASSERT(dtot < epsilon);


    if ((sz.x >50) && (sz.y>50))
    {
       ELISE_COPY
       (
           //disc(sz/2,20),
           disc( Pt2dr( sz.x/2, sz.y/2 ),20), // __NEW
           Abs(Fverif-Tif.in()),
           sigma(dif,nb_ch) 
       );

       REAL dtot = 0.0;
       for (INT i=0 ; i<nb_ch ; i++)
           dtot += dif[i];

       BENCH_ASSERT(dtot < epsilon);
    }


    if (indexed_colour)
    {
       Disc_Pal P0 = Tif.pal();
       BENCH_ASSERT(P0.nb_col() == nb_col);

       Elise_colour * t2 = P0.create_tab_c();

       for (int c=0 ; c<nb_col ; c++)
       {
           REAL d = t2[c].eucl_dist(tabc[c]);
           BENCH_ASSERT(d<epsilon);
       }
        
       DELETE_VECTOR(tabc,0);
       DELETE_VECTOR(t2,0);
    }


    TheIntFuckingReturnValue=system(RM ELISE_BFI_DATA_DIR "*");
}

void bench_tiff
     (
          Tiff_Im::COMPR_TYPE   compr,
	  Pt2di                  TileFile = Pt2di(-1,-1)
     )
{
     bench_tiff          // 1
     (
          TileFile,
          Pt2di(64,64),
          Pt2di(16,16),
          1,
          GenIm::u_int1,
          compr,
          true,
          Pt2di(1,1),
          true
     );

     bench_tiff          // 1
     (
          TileFile,
          Pt2di(64,64),
          Pt2di(16,16),
          1,
          GenIm::u_int1,
          compr,
          true,
          Pt2di(1,1)
     );


    
     bench_tiff          // 2
     (
          TileFile,
          Pt2di(90,90),
          Pt2di(16,16),
          1,
          GenIm::u_int1,
          compr,
          true,
          Pt2di(1,1)
     );

     bench_tiff          // 3
     (
          TileFile,
          Pt2di(90,90),
          Pt2di(16,16),
          1,
          GenIm::u_int1,
          compr,
          true,
          Pt2di(6,6)
     );

     bench_tiff          // 4
     (
          TileFile,
          Pt2di(90,90),
          Pt2di(90,90),
          3,
          GenIm::u_int1,
          compr,
          true,
          Pt2di(1,2)
     );

     bench_tiff          // 5
     (
          TileFile,
          Pt2di(90,90),
          Pt2di(16,16),
          3,
          GenIm::u_int1,
          compr,
          true,
          Pt2di(1,2)
     );

     bench_tiff          // 6
     (
          TileFile,
          Pt2di(90,90),
          Pt2di(16,16),
          3,
          GenIm::u_int1,
          compr,
          true,
          Pt2di(6,6)
     );

     bench_tiff          // 7
     (
          Pt2di(-1,-1),
          Pt2di(90,90),
          Pt2di(16,16),
          1,
          GenIm::bits1_msbf,
          compr,
          false,
          Pt2di(2,1)
     );

     bench_tiff         // 8
     (
          Pt2di(-1,-1),
          Pt2di(90,90),
          Pt2di(32,16),
          1,
          GenIm::bits2_msbf,
          compr,
          true,
          Pt2di(1,1)
     );

     bench_tiff          // 9
     (
          Pt2di(-1,-1),
          Pt2di(90,90),
          Pt2di(16,32),
          1,
          GenIm::bits4_msbf,
          compr,
          false,
          Pt2di(1,2)
     );

     bench_tiff          // 10
     (
          Pt2di(-1,-1),
          Pt2di(90,90),
          Pt2di(16,32),
          1,
          GenIm::bits2_lsbf,
          compr,
          true,
          Pt2di(1,2)
     );

     bench_tiff          // 11
     (
          Pt2di(-1,-1),
          Pt2di(90,16),
          Pt2di(90,16),
          1,
          GenIm::bits1_lsbf,
          compr,
          true,
          Pt2di(1,2)
     );

     bench_tiff  // 12
     (
          Pt2di(-1,-1),
          Pt2di(90,90),
          Pt2di(90,90),
          1,
          GenIm::bits1_lsbf,
          compr,
          false,
          Pt2di(1,1)
     );

     bench_tiff        // 13
     (
          Pt2di(-1,-1),
          Pt2di(90,90),
          Pt2di(80,80),
          1,
          GenIm::bits1_lsbf,
          compr,
          false,
          Pt2di(1,1)
     );

     bench_tiff        // 14
     (
          Pt2di(-1,-1),
          Pt2di(90,90),
          Pt2di(90,15),
          1,
          GenIm::bits1_lsbf,
          compr,
          false,
          Pt2di(1,1)
     );

     bench_tiff        // 15
     (
          Pt2di(-1,-1),
          Pt2di(90,90),
          Pt2di(90,20),
          1,
          GenIm::bits1_lsbf,
          compr,
          false,
          Pt2di(1,1)
     );


     bench_tiff          // 16
     (
          TileFile,
          Pt2di(90,90),
          Pt2di(90,90),
          3,
          GenIm::u_int1,
          compr,
          false,
          Pt2di(1,2)
     );

     bench_tiff          // 17
     (
          TileFile,
          Pt2di(16,16),
          Pt2di(16,16),
          1,
          GenIm::u_int1,
          compr,
          true,
          Pt2di(1,1)
     );

}


void bench_create_Elise_file_Im
     (
          INT               dim,
          INT          *    sz,
          GenIm::type_el    type_el,
          INT               offset_0,
          INT               dim_out
     )
{

    TheIntFuckingReturnValue=system(RM ELISE_BFI_DATA_DIR "tmp");
    Elise_File_Im file
                  (
                       ELISE_BFI_DATA_DIR "tmp",
                       dim,
                       sz,
                       type_el,
                       dim_out,
                       offset_0,
                       -1,
                       true
                  );
    
    INT maxv = 1 << nbb_type_num(type_el);

    Fonc_Num f0 = kth_coord(0)%maxv;
    if (dim_out>1) f0 = Virgule(f0,(kth_coord(1%dim)%maxv));
    if (dim_out>2) f0 = Virgule(f0,(kth_coord(2%dim)%maxv));
    if (dim_out>3) f0 = Virgule(f0,(kth_coord(3%dim)%maxv));

    INT S0[8] ,S1[8], S2[8];
    INT p0[8] = {0,0,0,0,0,0,0,0};


    ELISE_COPY
    (
         rectangle(p0,sz,dim),
         f0,
         file.out() | sigma(S0,dim_out)
    );

    Symb_FNum f (file.in());

    ELISE_COPY
    (
         rectangle(p0,sz,dim),
         Virgule(Abs(f-f0),f),
         Virgule(sigma(S2,dim_out),sigma(S1,dim_out))
    );

    for (INT d = 0 ; d< dim_out; d++)
    {
        BENCH_ASSERT
        (   
                (S2[d] == 0)
            &&  (S1[d] == S0[d])
        );
    }
}

static Pt2di PR(INT aMod)
{
   return Pt2di(round_ni(aMod*NRrandom3()),round_ni(aMod*NRrandom3()));
}


void DebugTif();
void bench_tiff_im()
{
     DebugTif();
        // bench_tiff(Tiff_Im::No_Compr);

    bench_GIF_EGFI();
    bench_Elise_File_Im_EGFI();

    Tiff_Im  Tif
    (
		 ELISE_BFI_DATA_DIR "ex.tif",
         Pt2di(100,100),
         GenIm::u_int1,
         Tiff_Im::No_Compr,
         Tiff_Im::BlackIsZero
     );
    Tif.show();

   // Test dallage
   for (int aK=0; aK< 100 ; aK++)
   {
     bench_tiff
     (
          Pt2di(1,1)  +  PR(100),
          Pt2di(1,1)  +  PR(200),
          (Pt2di(1,1)  +  PR(3))*16,
          (NRrandom3() > 0.5) ? 1 : 3,
          (NRrandom3() > 0.5) ?  GenIm::u_int1 : GenIm::int2,
          Tiff_Im::No_Compr,
          NRrandom3() > 0.5,
          Pt2di(1,1) + PR(1)
     );

   }

for (int i = 0; i < 100 ; i++)
{

        cout << "======================================" << i << "\n";
        bench_tiff(Tiff_Im::MPD_T6);
        bench_tiff(Tiff_Im::No_Compr);
        bench_tiff(Tiff_Im::No_Compr,Pt2di(32,32));

        bench_tiff(Tiff_Im::PackBits_Compr);

        bench_tiff(Tiff_Im::Group_4FAX_Compr);
        bench_tiff(Tiff_Im::CCITT_G3_1D_Compr);
        bench_tiff(Tiff_Im::LZW_Compr);

     Pt2di aTF (32,32);
     bench_tiff
     (
          aTF,
          Pt2di(90,90),
          Pt2di(90,90),
          1,
          GenIm::u_int1,
          Tiff_Im::No_Compr,
          false,
          Pt2di(1,1)
     );

     bench_tiff
     (
          aTF,
          Pt2di(100,100),
          Pt2di(32,16),
          1,
          GenIm::u_int1,
          Tiff_Im::No_Compr,
          true,
          Pt2di(2,2)
     );

     bench_tiff
     (
          Pt2di(-1,-1),
          Pt2di(81,90),
          Pt2di(81,32),
          3,
          GenIm::u_int2,
          Tiff_Im::No_Compr,
          true,
          Pt2di(1,1)
     );

     bench_tiff
     (
          Pt2di(-1,-1),
          Pt2di(70,80),
          Pt2di(16,16),
          3,
          GenIm::real4,
          Tiff_Im::No_Compr,
          true,
          Pt2di(1,1)
     );

     bench_tiff
     (
          aTF,
          Pt2di(70,80),
          Pt2di(16,16),
          3,
          GenIm::u_int1,
          Tiff_Im::No_Compr,
          true,
          Pt2di(1,1)
     );

}
    {
         INT sz[4] = {10,9,8,7};
         bench_create_Elise_file_Im(2,sz,GenIm::u_int1,11,1);
         bench_create_Elise_file_Im(3,sz,GenIm::u_int2,222,4);
         bench_create_Elise_file_Im(2,sz,GenIm::bits4_msbf,33,1);
         bench_create_Elise_file_Im(2,sz,GenIm::bits2_msbf,33,1);
         bench_create_Elise_file_Im(2,sz,GenIm::bits1_msbf,33,1);
         bench_create_Elise_file_Im(3,sz,GenIm::bits1_msbf,33,1);
         bench_create_Elise_file_Im(2,sz,GenIm::bits1_msbf,33,4);
    }


    cout << "END TIFF \n";
}


void bench_pnm()
{

    {
        Pt2di sz(200,220);

        Elise_File_Im pgm = Elise_File_Im::pgm
		(
			ELISE_BFI_DATA_DIR "tmp.pgm",
			sz
		);

        ELISE_COPY(pgm.all_pts(),0,pgm.out());
        //ELISE_COPY(disc(Pt2di(100,100),70),FX+30,pgm.out());
        ELISE_COPY(disc(Pt2dr(100,100),70),FX+30,pgm.out()); // __NEW


        Elise_File_Im pbm = Elise_File_Im::pbm
		(
			ELISE_BFI_DATA_DIR "tmp.pbm",
			sz
		);			
        ELISE_COPY(pbm.all_pts(),0,pbm.out());
        //ELISE_COPY(disc(Pt2di(100,100),50),1,pbm.out());
        ELISE_COPY(disc(Pt2dr(100,100),50),1,pbm.out());


        Elise_File_Im ppm = Elise_File_Im::ppm
		(
			ELISE_BFI_DATA_DIR "tmp.ppm",
			sz
		);
        ELISE_COPY(ppm.all_pts(),Fonc_Num(128,128,128),ppm.out());
        //ELISE_COPY(disc(Pt2di(100,100),70),Virgule(FX+30,FY+30,128),ppm.out());
	ELISE_COPY(disc(Pt2dr(100,100),70),Virgule(FX+30,FY+30,128),ppm.out()); // __NEW

    }
    cout << "END P*M \n";

}

// Un exemple ultra simplifie ajoute a l'occasion d'un bug
void bench_tiff_2
     (
          Pt2di                    sz,
          Pt2di                    szt,
          Tiff_Im::COMPR_TYPE      compr,
          Pt2di                    bloc
     )
{
   // int nb_ch = 1;
   GenIm::type_el  type =  GenIm::bits1_msbf;
   static INT cpt = 0;
    cout << "COMPR = " << (INT) compr 
	 <<  "[NUM = " << (++cpt) << "]" 
	 <<  " Type = " << type
	 << "\n";


    L_Arg_Opt_Tiff l;
    l = l + Arg_Tiff(Tiff_Im::ANoStrip());


    Tiff_Im  Tif
    (
         ELISE_BFI_DATA_DIR "ex.tif",
         sz,
         type,
         compr,
          Tiff_Im::BlackIsZero, 
         l
    );

    Im2D_REAL8 I1(sz.x,sz.y,0.0);


    ELISE_COPY
    (
         rectangle(Pt2di(0,0),sz),
         1,
         I1.out() | Tif.out()
    );

    ELISE_COPY
    (
        rectangle(Pt2di(29,0),Pt2di(30,1)),
        0,
        I1.out()  | Tif.out()
    );
    {
       REAL dtot = 0.0; 
       ELISE_COPY ( rectangle(Pt2di(0,0),sz), Abs(I1.in()-Tif.in()), sigma(dtot));
       BENCH_ASSERT(dtot < epsilon);
    }

   for (int aK = 1 ; aK < 20 ; aK++)
   {
         ELISE_COPY
         (
             rectangle(Pt2di(30-aK,aK+10),Pt2di(30,aK+11)),
             0,
             I1.out()  | Tif.out()
         );
    REAL dtot = 0.0;
    ELISE_COPY
    (
        rectangle(Pt2di(0,0),sz),
        Abs(I1.in()-Tif.in()),
        sigma(dtot)
    );
    BENCH_ASSERT(dtot < epsilon);
   }
    ELISE_COPY
    (
        //disc(sz/2,20),
        disc( Pt2dr( sz.x/2, sz.y/2 ), 20 ),// __TMP
        0,
        I1.out()  | Tif.out()
    );


    {
       REAL dtot = 0.0; 
       ELISE_COPY ( rectangle(Pt2di(0,0),sz), Abs(I1.in()-Tif.in()), sigma(dtot));
       BENCH_ASSERT(dtot < epsilon);
    }
}


void DebugTif()
{
     bench_tiff_2          // 7
     (
          Pt2di(96,96),
          Pt2di(96,96),
          Tiff_Im::No_Compr,
          Pt2di(2,1)
     );

}
