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


/*
void MakeTiffRed2
     (
          const std::string & aNameFul,
          const std::string & aNameRed
     )
{
    Tiff_Im aTifIn(aNameFul.c_str());
    Pt2di aSz = aTifIn.sz();
    Pt2di aSzRed = (aSz+Pt2di(1,1))/2;

    Tiff_Im aTifRed
            (
                 aNameRed.c_str(),
                 aSzRed,
                 aTifIn.type_el(),
		 Tiff_Im::No_Compr,
		 aTifIn.phot_interp()
            );

    ELISE_COPY
    (
        aTifIn.all_pts(),
        reduc_binaire(aTifIn.in_proj()),
        Filtre_Out_RedBin(aTifRed.out())
    );
}
*/

void bench_reduc_image(Pt2di aSz,Tiff_Im::PH_INTER_TYPE aType)
{
    // Creation d'un fichier tiff 
    Tiff_Im aTif
            (
                 ELISE_BFI_DATA_DIR "tmp.tif",
                 aSz,
                 GenIm::u_int1,
		 Tiff_Im::No_Compr,
		 aType
            );

    INT aNbCh = aTif.nb_chan();
	    
    std::vector<Im2D_U_INT1> aVIms;
    std::vector<Im2D_U_INT1> aVImsRed2;
    Output aImsOut = Output::onul();
    Fonc_Num aFRand =0;
    Fonc_Num aImsF =0;
    Output aImsRed2Out = Output::onul();
    Pt2di aSzRed = ((aSz+Pt2di(1,1))/2);
    Pt2di a2SzRed = aSzRed*2;

    for (INT aK=0; aK<aNbCh ; aK++)
    {
        aVIms.push_back(Im2D_U_INT1(aSz.x,aSz.y));
        aVImsRed2.push_back(Im2D_U_INT1(a2SzRed.x,a2SzRed.y));
	Fonc_Num F = frandr() * 255;
	if (aK==0)
	{
           aFRand = F;
	   aImsOut = aVIms.back().out();
	   aImsF = aVIms.back().in_proj();
	   aImsRed2Out = aVImsRed2.back().out();
	}
	else
	{
           aFRand = Virgule(aFRand,F);
	   aImsOut = Virgule(aImsOut,aVIms.back().out());
	   aImsF = Virgule(aImsF,aVIms.back().in_proj());
	   aImsRed2Out = Virgule(aImsRed2Out,aVImsRed2.back().out());
        }
    }
    ELISE_COPY(aTif.all_pts(),aFRand,aTif.out()|aImsOut);

    // Verif in_proj
    INT aSigm[10];
    ELISE_COPY
    (
        rectangle(Pt2di(-2,-3),aSz+Pt2di(4,5)),
	Abs(aImsF-aTif.in_proj()),
	sigma(aSigm,aNbCh)
    );

    for (INT aK=0; aK<aNbCh ; aK++)
	BENCH_ASSERT(aSigm[aK]==0);

    // Verif Filtre_Out_RedBin
 
    Im2D_INT1 aImX(aSzRed.x,aSzRed.y);
    Im2D_INT1 aImY(aSzRed.x,aSzRed.y);

    ELISE_COPY
    (
         rectangle(Pt2di(0,0),aSz),
	 Virgule(FX%255-128,FY%255-128),
	 Filtre_Out_RedBin(Virgule(aImX.out(),aImY.out()))
    );

    INT aDifRed;
    ELISE_COPY
    (
        aImX.all_pts(),
	   Abs(aImX.in() - ((FX*2)%255-128))
	+  Abs(aImY.in() - ((FY*2)%255-128)),
	sigma(aDifRed)
    );
    BENCH_ASSERT(aDifRed==0);

    // Verif reduc_binaire
    ELISE_COPY
    (
        rectangle(Pt2di(0,0),a2SzRed),
        reduc_binaire(aTif.in_proj()),
        aImsRed2Out
    );

    Fonc_Num fRed =0;

    for (INT aK=0; aK<aNbCh ; aK++)
    {
        Im2D_U_INT1 aIRed = 
            cReducImCenteredFact2<U_INT1,U_INT1>::DoRed(aVIms[aK]);
	if (aK==0)
           fRed = aIRed.in();
	else
           fRed = Virgule(fRed,aIRed.in());
        INT aSDif,aMaxDif;
        ELISE_COPY
        (
            aIRed.all_pts(),
            Abs(aIRed.in()-aVImsRed2[aK].in()[Virgule(FX,FY)*2]),
            sigma(aSDif) | VMax(aMaxDif)
        );
	BENCH_ASSERT(aMaxDif==0);
    }

    // Verif globale
    MakeTiffRed2
    (
         ELISE_BFI_DATA_DIR "tmp.tif",
         ELISE_BFI_DATA_DIR "tmpRed.tif"
    );

    Tiff_Im aFRed(ELISE_BFI_DATA_DIR "tmpRed.tif");
    ELISE_COPY
    (
         aFRed.all_pts(),
	 Abs(aFRed.in()-fRed),
	 sigma(aSigm,aNbCh)

    );
    for (INT aK=0; aK<aNbCh ; aK++)
       BENCH_ASSERT(aSigm[aK]==0) ;
    
    // Creation d'un fichier tiff 

    system(RM ELISE_BFI_DATA_DIR "tmp.tif");
    system(RM ELISE_BFI_DATA_DIR "tmpRed.tif");
}

void bench_tile_file()
{
    Pt2di aSz(2000,2000);
    Tiff_Im  aFile1
    (
        ELISE_BFI_DATA_DIR "tmp.tif",
	aSz,
	GenIm::u_int1,
	Tiff_Im::No_Compr,
        Tiff_Im::BlackIsZero,
	   Tiff_Im::Empty_ARG
	+  Arg_Tiff(Tiff_Im::AFileTiling(Pt2di(450,450)))
    );

    Tiff_Im  aFile2
    (
        ELISE_BFI_DATA_DIR "tmp.tif"
    );

    ELISE_COPY(aFile1.all_pts(),(FX/3+FY/7)%256,aFile2.out());
    ELISE_COPY(rectangle(Pt2di(200,200),Pt2di(1500,1500)),255,aFile2.out());
    ELISE_COPY(disc(Pt2di(700,700),300),128,aFile2.out());


    Tiff_Im  aFile3
    (
        ELISE_BFI_DATA_DIR "tmp.tif3",
	aSz,
	GenIm::u_int1,
	Tiff_Im::No_Compr,
        Tiff_Im::BlackIsZero
    );

    ELISE_COPY(aFile1.all_pts(),aFile1.in(),aFile3.out());

    system(RM ELISE_BFI_DATA_DIR "tmp*.tif");

}

static std::string NameFax(int aZ)
{
       return    std::string(ELISE_BFI_DATA_DIR) 
              +  std::string("tmpFax")
              +  ToString(aZ)
              +  std::string(".tif");
}

void  bench_big_FAX4()
{
  Pt2di aSz(100000,100000);
  std::string aN1 =NameFax(1);

/*
  Tiff_Im aTif (aN1.c_str());
*/
  Tiff_Im aTif
          (
                aN1.c_str(),
                aSz, 
                GenIm::bits1_msbf,
                Tiff_Im::Group_4FAX_Compr,
                Tiff_Im::BlackIsZero,
	            Tiff_Im::Empty_ARG
	        +  Arg_Tiff(Tiff_Im::AFileTiling(aSz))
          );

   ELISE_COPY
   (
      aTif.all_pts(),
      (FX/2000+FY/3000)%2,
      aTif.out() |  Video_Win::WiewAv(aTif.sz())
   );

   for (int aZ=2 ; aZ<=1024 ; aZ *= 2)
   {
       std::string aNameFull = NameFax(aZ/2);
       std::string aNameRed = NameFax(aZ);
       MakeTiffRed2Binaire
       (
               aNameFull,
               aNameRed,
               0.5,
               aSz
       );
   }

    exit(1);
    // system(RM ELISE_BFI_DATA_DIR "tmp*.tif");
}

void bench_reduc_image()
{
     bench_big_FAX4();
     bench_tile_file();

     bench_reduc_image(Pt2di(1500,30),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(200,300),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(301,300),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(200,201),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(301,201),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(31,1500),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(3,15000),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(15000,3),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(1500,30),Tiff_Im::RGB);
     bench_reduc_image(Pt2di(200,300),Tiff_Im::BlackIsZero);
}
