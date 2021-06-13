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


template <class Type> class Bench_PackB_IM 
{
	public :
		static Im2D<Type,INT>  ImInit(Pt2di sz,Fonc_Num f);
		Bench_PackB_IM(Pt2di sz,Fonc_Num f);


                void ModifCSte(Flux_Pts aFlux,INT aVal);
                void ModifLut(Flux_Pts aFlux,Fonc_Num aFonc);

		void verif();
		void verif(Flux_Pts);
		void verif(Flux_Pts,INT def);

        void DoNothing(){};

        void TiffVerif();

	private :


		Pt2di sz;
	 	Im2D<Type,INT>  	im1;
        INT             mPer;
	 	PackB_IM<Type>	pck;
	

        Pt2dr CentreRand() {return Pt2dr(sz.x * NRrandom3(),sz.y * NRrandom3());}
        REAL RayonRand() {return ElMax(1.5,euclid(sz) *NRrandom3());}
};

template <class Type> Im2D<Type,INT> Bench_PackB_IM<Type>::ImInit(Pt2di sz,Fonc_Num f)
{
	Im2D<Type,INT> res(sz.x,sz.y);
	ELISE_COPY(res.all_pts(),f,res.out());
	return res;
}

extern INT aEliseCptFileOpen;


template <class Type> void Bench_PackB_IM<Type>::TiffVerif()
{

   Pt2di SzDalle = Pt2di(mPer,64);

    Tiff_Im  aTifFile
    (
         ELISE_BFI_DATA_DIR "ex.tif",  
         sz,
         type_of_ptr((Type *)0),
         Tiff_Im::NoByte_PackBits_Compr,
         Tiff_Im::BlackIsZero,
            L_Arg_Opt_Tiff()
           +  Arg_Tiff(Tiff_Im::ATiles(SzDalle))
    );


    

    ELISE_COPY(aTifFile.all_pts(),pck.in(),aTifFile.out());
    INT VDIF;
    ELISE_COPY(aTifFile.all_pts(),Abs(pck.in()-aTifFile.in()),VMax(VDIF));

    BENCH_ASSERT(VDIF==0);

    if (type_of_ptr((Type *)0)==GenIm::u_int1)
    {
         PackB_IM<U_INT1> aPack2 = aTifFile.un_load_pack_bit_U_INT1();
         ELISE_COPY(aTifFile.all_pts(),Abs(pck.in()-aPack2.in()),VMax(VDIF));
         BENCH_ASSERT(VDIF==0);
    }
    if (type_of_ptr((Type *)0)==GenIm::u_int2)
    {
         PackB_IM<U_INT2> aPack2 = aTifFile.un_load_pack_bit_U_INT2();
         ELISE_COPY(aTifFile.all_pts(),Abs(pck.in()-aPack2.in()),VMax(VDIF));
         BENCH_ASSERT(VDIF==0);
    }
}


template <class Type> void  Bench_PackB_IM<Type>::verif(Flux_Pts flx)
{
	INT dif;
	ELISE_COPY
	(
		flx,
 		Abs(im1.in()-pck.in()),
		VMax(dif)
	);
	BENCH_ASSERT(dif==0);
}

template <class Type> void  Bench_PackB_IM<Type>::verif(Flux_Pts flx,INT def)
{
	INT dif;
	ELISE_COPY
	(
		flx,
 		Abs(im1.in(def)-pck.in(def)),
		VMax(dif)
	);
	BENCH_ASSERT(dif==0);
}


template <class Type> 
         void  Bench_PackB_IM<Type>::ModifCSte(Flux_Pts aFlux,INT aVal)
{
    ELISE_COPY
    (
          aFlux,
          aVal,
          im1.out() | (pck.OutLut(aVal)<<FX)
    );
}


template <class Type> 
         void  Bench_PackB_IM<Type>::ModifLut(Flux_Pts aFlux,Fonc_Num aLut)
{
    ELISE_COPY
    (
          aFlux,
          aLut[im1.in(0)],
          im1.out() | (pck.OutLut(aLut)<<FX)
    );
}


template <class Type> void  Bench_PackB_IM<Type>::verif()
{
        INT def = (INT)(NRrandom3() * 1000);  
        verif( rectangle(Pt2di(1,0),Pt2di(2,1)));
	for (INT k=0; k<10 ; k++)
	{
            verif( rectangle(Pt2di(k,0),Pt2di(k+1,10)  ),def);
            verif( rectangle(Pt2di(k,0),Pt2di(k+10,10) ),def);
            verif( rectangle(Pt2di(k,0),Pt2di(k+100,10)),def);
            verif( rectangle(Pt2di(k,0),Pt2di(k+200,10)),def);
	}



	verif( im1.all_pts());
	verif( rectangle(Pt2di(-10,-20),sz+Pt2di(30,40)),def);
	verif( disc(sz/2.0,euclid(sz)/1.8),def);


	{
	for (INT k=0 ; k<10 ; k++)
	{
		Pt2dr c = sz/2.0 + Pt2dr(NRrandom3(),NRrandom3())*20;
		REAL ray = 1+NRrandom3()*100;
        verif(disc(c,ray),def);
	}
	}



    ELISE_COPY(disc(CentreRand(),RayonRand()),1,im1.out()| pck.out());
	verif(im1.all_pts());

    ELISE_COPY(disc(CentreRand(),RayonRand()),frandr()*8,im1.out()| pck.out());
	verif(im1.all_pts());

    INT NbPts = (INT)(3 + NRrandom3()*10);

    ElList<Pt2di> Lpt;
	{
    for (INT k=0; k<NbPts ; k++)
       Lpt = Lpt+Pt2di(CentreRand());
	}

    ELISE_COPY(polygone(Lpt),NRrandom3()<0.1,im1.out()| pck.out());
	verif(im1.all_pts());



      ModifCSte(rectangle(Pt2di(5,0),Pt2di(10,10)),2);
      verif(im1.all_pts());

      ModifLut(rectangle(Pt2di(0,5),Pt2di(12,12)),FX&3);
      verif(im1.all_pts());

      //ModifCSte(disc(Pt2di(50,50),20),3);
      ModifCSte(disc(Pt2dr(50,50),20),3); // __NEW
      verif(im1.all_pts());


      for (INT NbC =0 ; NbC < 20 ; NbC++)
      {
          ElList<Pt2di> lPt;
          for (INT iPt =0 ; iPt < 20; iPt ++)
          {
              lPt  = lPt + Pt2di(CentreRand());
          }
          ModifCSte(polygone(lPt),INT(NRrandom3() * 3));
          verif(im1.all_pts());
      }

      Pt2di P_00 (0,0);
      Pt2di P_10 (sz.x,0);
      Pt2di P_01 (0,sz.y);
      ElList<Pt2di> lP1;
      lP1 =  lP1 + P_00; lP1 =  lP1 + P_01; lP1 =  lP1 + P_10;

      ModifCSte(polygone(lP1),7);
      verif(im1.all_pts());





    TiffVerif();

}

template <class Type> Bench_PackB_IM<Type>::Bench_PackB_IM
(
	Pt2di SZ,
	Fonc_Num f
) :
	sz     (SZ),
	im1    (ImInit(sz,f)),
        mPer   (16 * (INT)(2+(NRrandom3()*20))),
	pck    (sz.x,sz.y,im1.in(),mPer)
{
	verif();
}


template <class Type> void bench_pack_im(Type *,INT vmax,bool quick)
{

	for (INT x=10; x< (quick ? 300 : 1200); x+= 220)
	{
cout << x << "\n";
		Bench_PackB_IM<Type>  b1(Pt2di(x,x),0);
		Bench_PackB_IM<Type>  b0(Pt2di(x,x),FX%vmax);
		Bench_PackB_IM<Type>  b2(Pt2di(x,x),(FX>FY)*vmax);
		Bench_PackB_IM<Type>  b3(Pt2di(x,x),frandr()<0.1);
		Bench_PackB_IM<Type>  b4(Pt2di(x,x),frandr()<(1/128.0));

                b0.DoNothing();
                b1.DoNothing();
                b2.DoNothing();
                b3.DoNothing();
                b4.DoNothing();
	}
}


void bench_compr_im()
{
     for (INT k=0 ; k< 3 ; k++)
     {
	    bench_pack_im((U_INT1 *)0,255,k<3) ;
	    bench_pack_im((U_INT2 *)0,6000,k<3) ;
     }
     cout << "END COMPR IM \n";
}




/***************************************************/
/***************************************************/
/***                                             ***/
/***    TestScroller                             ***/
/***                                             ***/
/***************************************************/
/***************************************************/

class TestScroller :
	   public  Grab_Untill_Realeased
{
	public :

		virtual ~TestScroller(){};
		TestScroller(Video_Win,ElImScroller & scrol);
		void GUR_query_pointer(Clik p,bool);
		void GUR_button_released(Clik p);

        void SetScale(bool augm);

        Video_Win				W;            
	    ElImScroller &			scrol;
	    Pt2di 					_p0grab;
	    REAL 					_sc0grab;
	    bool                    _mode_tr;
	    bool                    _sc_evt;

};

void TestScroller::GUR_button_released(Clik cl)
{
	if (! _mode_tr)
	{
		scrol.LoadAndVisuIm(false);
	}
}
#define FIVE_BOUT false

void TestScroller::GUR_query_pointer(Clik cl,bool)
{
	if (_mode_tr)
	{
		INT v = 1;
		if (cl.shifted()) 
			v *= 2;
		if (cl.controled()) 
			v *= 4;

    	        scrol.SetDTrW((_p0grab-Pt2di(cl._pt)) *v);
		//_p0grab = cl._pt;
		_p0grab = Pt2di( cl._pt ); // __NEW
	}
	else
	{
		  REAL scy = _p0grab.y- cl._pt.y;
		  scy /= -100.0;
		  scy = _sc0grab *pow(2.0,scy);
		  scy = ElMin(ElMax(scy,0.00),10.0);

      	  //scrol.SetScArroundPW(_p0grab,scy,true);
      	  scrol.SetScArroundPW( Pt2dr(_p0grab),scy,true); // __NEW
	}
}

void TestScroller::SetScale(bool augm)
{
    REAL fact = 1.2;
	REAL sc = scrol.sc();
	if (augm) 
       sc *= fact;
	else      
       sc /= fact;
     //scrol.SetScArroundPW(_p0grab,sc,true);
     scrol.SetScArroundPW( Pt2dr(_p0grab),sc,true); // __NEW
}




TestScroller::TestScroller
(
     Video_Win WIN,
     ElImScroller & SCROL
) :
	W			(WIN),
	scrol       (SCROL)
{


	scrol.set(Pt2dr(2000,2000),1);

/*
    	scrol.SetDTrW(Pt2di(323,300));
        for (INT k=0; k<10000; k++)
	{
	     cout << k << "\n";
    	     scrol.SetDTrW(Pt2di(1,1));
    }
*/

	while (true)
	{
	  Clik cl1   = W.disp().clik_press();
   	  // cout << "But " << cl1._b << "\n";
	  switch (cl1._b )
	  {

		   case 2 : 
    			scrol.SetDTrW(Pt2di(-10,0));
           break;
			case 1 :
		  		_mode_tr = FIVE_BOUT || (! cl1.shifted());
          		        //_p0grab = cl1._pt;
          		        _p0grab = Pt2di(cl1._pt); // __NEW
		  		_sc0grab = scrol.sc();
                if (_sc_evt)
                   //scrol.SetScArroundPW(_p0grab,scrol.sc(),false);
                   scrol.SetScArroundPW( Pt2dr(_p0grab),scrol.sc(),false); // __NEW
	      		W.grab(*this);
				_sc_evt = false;
           break;

		   case 4 : 
				SetScale(true);
				_sc_evt = true;
           break;

		   case 5 : 
				SetScale(false);
				_sc_evt = true;
           break;

    		case 3 :
			return;
      }
	}

}

void MakeFCol()
{
     Tiff_Im Tif("../TMP/f3Reduc4.tif");
     Tiff_Im NewTif
             (
                 "../TMP/Col.tif",
		 Tif.sz(),
		 GenIm::u_int1,
		 Tiff_Im::No_Compr,
		 Tiff_Im::RGB, 
		   ElList<Arg_Tiff> ()
                 + Arg_Tiff(Tiff_Im::ATiles(Pt2di(1024,1024)))
		 + Arg_Tiff(Tiff_Im::APlanConf(Tiff_Im::Chunky_conf))
             );
      ELISE_COPY
      (
         Tif.all_pts(),
         its_to_rgb
         (Virgule(
              Tif.in(0),
              FX,
	      Abs((FY%512)-256)
         )),
	NewTif.out()
      );
}

/********************************************************************/
/********************************************************************/
/********************************************************************/
/********************************************************************/

void bench_fp()
{
   // const char * ch = "../TMP/f3Reduc1.tif";
   const char * ch = "/home/data/mpd/Andalousie/andaReduc1.tif";
   Tiff_Im tiff(ch);
   ELISE_fp fp(ch, ELISE_fp::READ);
tiff.show();



   INT NbTx =  tiff.sz().x/ tiff.SzTile()[0];
   INT NbTy =  tiff.sz().y/ tiff.SzTile()[1];

   REAL ttot =0;
   INT k=0;

   INT f = 1;
   char * c = new char [tiff.SzTile()[0] * tiff.SzTile()[1]*f*3];
   for (INT x=0; x< NbTx -1; x++)
       for (INT y=0; y< NbTy ; y++)
        {
           fp.seek_begin(tiff.offset_tile(x,y,0));
           ElTimer tim;
cout << tiff.offset_tile(x,y,0) << " " << tiff.byte_count_tile(x,y,0) << "\n";
           fp.read(c,1,tiff.byte_count_tile(x,y,0)/1);
           REAL  t = tim.sval();
           ttot += t;
           k++;
           cout << "Read Time  moy " << ttot/k << " " << x << " " << y << "\n";
	}
   REAL m1 =  ttot/k ;
   ttot = 0;
   k=0;

   {
   for (INT x=0; x< NbTx ; x++)
       for (INT y=0; y< NbTy ; y++)
        {
           fp.seek_begin(0);
           ElTimer tim;
           fp.seek_begin(tiff.offset_tile(x,y,0));
           REAL  t = tim.sval();
           ttot += t;
           k++;
           cout << "Time  moy " << ttot/k << " " << x << " " << y << "\n";
		}
   }

   REAL m2 =  ttot/k ;
   cout << "Moy read = " << m1 << " Moy Seek = " << m2 << "\n";
}

     // bench_fp();
	 // Bench_Visu_ISC();






