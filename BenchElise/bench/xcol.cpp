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

const INT SZX = 256;
const INT SZY = 256;


void toto()
{
    Video_Display Ecr((char *) NULL);


    Disc_Pal       Pdisc  = Disc_Pal::P8COL();

    Elise_Set_Of_Palette SOP(newl(Pdisc));

    Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(SZX,SZY));
    Ecr.load(SOP);


	ELISE_COPY (W.all_pts(),P8COL::green,W.odisc());
	std::std::cout << "Green disc \n";
	std::cout << "should be : full green\n";
	getchar();

}



void test()
{
    Video_Display Ecr((char *) NULL);


    BiCol_Pal      Prb  ( 
                            Elise_colour::black,
                            Elise_colour::red,
                            Elise_colour::blue,
                            8,
                            8
                        );
    Gray_Pal       Pgray  (20);
    Disc_Pal       Pdisc  = Disc_Pal::PNCOL();

    Circ_Pal       Pcirc = Circ_Pal::PCIRC6(80);


    Col_Pal        g129 = Pgray(128);
    Col_Pal        red  = Pdisc(P8COL::red);

    RGB_Pal        Prgb(3,3,3);

    Elise_Set_Of_Palette SOP(newl(Prb)+Pgray+Pdisc+Pcirc+Prgb);



    Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(SZX,SZY));
    Ecr.load(SOP);


    getchar();


	{
    		ELISE_COPY (W.all_pts(),P8COL::green,W.odisc());
			std::cout << "Green disc \n";
    		std::cout << "should be : full green\n";
    		getchar();

			W.fill_rect(Pt2dr(0,0),Pt2dr(128,128),Pdisc(P8COL::green));
			std::cout << "Green disc \n";
    		std::cout << "should be : green on 0,0 x 128,128\n";
    		getchar();

    		ELISE_COPY (W.all_pts(),Fonc_Num(0,127,0),W.orgb());
			std::cout << "Green rgb \n";
    		std::cout << "should be : full green\n";
    		getchar();

    		ELISE_COPY (W.all_pts(),FX%256,W.out(Pgray));
    		std::cout << "should be X in gray\n";
    		getchar();
	}

    ELISE_COPY (W.all_pts(),P8COL::red,W.odisc());
    std::cout << "should be : full red\n";
    getchar();

    ELISE_COPY (W.all_pts(),P8COL::green,W.odisc());
    std::cout << "should be : full green\n";
    getchar();

    ELISE_COPY (W.all_pts(),P8COL::blue,W.odisc());
    std::cout << "should be : full blue\n";
    getchar();




    ELISE_COPY (W.all_pts(),Fonc_Num(255,0,0),W.orgb());
    W.fill_rect(Pt2di(50,50),Pt2di(100,100),Prgb(0,0,255));
    std::cout << "should be : RED background + blue rectangle\n";
    getchar();


    ELISE_COPY(select(W.all_pts(),FX>FY),FX,W.ogray());
    getchar();

    ELISE_COPY(select(W.all_pts(),FX<=FY),(FX,FY),W.obicol());
    getchar();

    {
        ELISE_COPY(W.all_pts(),Fonc_Num(255,0,255),W.orgb());
        Video_Win W2 = W.chc(Pt2dr(0,0),Pt2dr(5,5));
        ELISE_COPY(select(W2.all_pts(),FX>FY),((FX,FY,FX+FY)*8)%256,W2.orgb());
        getchar();
    }

    {
        ELISE_COPY(W.all_pts(),Fonc_Num(255,128,0),W.orgb());
        Video_Win W2 = W.chc(Pt2dr(0,0),Pt2dr(2.5,3.5));
        ELISE_COPY(select(W2.all_pts(),(FX+FY)%4),((FX,FY,FX+FY)*8)%256,W2.orgb());
        getchar();
    }


    {
        Video_Win W2 = W.chc(Pt2di(0,0),Pt2di(8,5));
        ELISE_COPY(W2.all_pts(),((FX+FY)*16)%256,W2.ogray());
        getchar();

        ELISE_COPY(W2.all_pts(),((FX,FY)*9)%256,W2.obicol());
        getchar();
    }

    {
        Video_Win W2 = W.chc(Pt2dr(0,0),Pt2dr(1.5,2.5));
        ELISE_COPY(W2.all_pts(),((FX+FY)*2)%256,W2.ogray());
        getchar();

        ELISE_COPY(W2.all_pts(),((FX,FY)*4)%256,W2.obicol());
        getchar();
    }


    ELISE_COPY(W.all_pts(),(FX+FY)/2,W.ogray());
    getchar();

    ELISE_COPY(W.all_pts(),(FX*2,FY*2)%256,W.obicol());
    getchar();

    ELISE_COPY
    (
            W.all_pts(),
            polar((FX-128,FY-128),0.0) * (255.0/(2*PI)),
            (Output::onul(1),W.ocirc())
    );

    getchar();



    ELISE_COPY(W.all_pts(),(FX,FY,0),W.orgb());
    getchar();

    ELISE_COPY(W.all_pts(),(FX,0,FY),W.orgb());
    getchar();

    ELISE_COPY(W.all_pts(),(0,FX,FY),W.orgb());
    getchar();



    {
        INT step = 2;

        for (INT x = 0; x < 128 ; x += step)
             for (INT y = 0; y < 128 ; y += step)
             {
                  W.fill_rect
                  (
                     Pt2di(x,y),
                     Pt2di(x+step,y+step),
                     Prgb(2*x,2*y,0)
                  );
                  W.fill_rect
                  (
                     Pt2di(128+x,y),
                     Pt2di(128+x+step,y+step),
                     Prgb(2*x,0,2*y)
                  );
                  W.fill_rect
                  (
                     Pt2di(x,128+y),
                     Pt2di(x+step,128+y+step),
                     Prgb(0,2*x,2*y)
                  );
             }
        getchar();
    }


    {
       INT step = 2;
       for (INT x = 0; x < 256 ; x += step)
            for (INT y = 0; y < 256 ; y += step)
                 W.fill_rect(Pt2di(x,y),Pt2di(x+step,y+step),Pgray((x+y)/2));

       getchar();
    }

    {
        for (INT i = 0 ; i < 8 ; i++)
            W.fill_rect
            (  Pt2di(10+20*i,10),
               Pt2di(30+20*i,50),
               Pdisc(i)
            );
        getchar();
    }

    {
        INT step = 2;
        for (INT x = 0; x < 256 ; x += step)
             for (INT y = 0; y < 256 ; y += step)
                  W.fill_rect(Pt2di(x,y),Pt2di(x+step,y+step),Prb(x,y));
        getchar();
    }

    {
       INT step = 2;
       for (INT x = 0; x < 256 ; x += step)
            for (INT y = 0; y < 256 ; y += step)
                 W.fill_rect(Pt2di(x,y),Pt2di(x+step,y+step),Pcirc(x+y-10));

       getchar();
    }

}

INT main(int,char **)
{
    ELISE_DEBUG_USER = true;
    All_Memo_counter MC_INIT;
    stow_memory_counter(MC_INIT);

    test();

    verif_memory_state(MC_INIT);
    std::cout << "OK LENA \n";

    return 0;
}
