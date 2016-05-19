/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/



#include "StdAfx.h"

std::ostream & operator << (std::ostream & ofs,const RGB_Int  &aRGB)
{
   ofs << "[r " << aRGB._r << ";g " << aRGB._g << ";b " << aRGB._b << "]";
   return ofs;
}

/************************************************************************/
/************************************************************************/
/*********                                                         ******/
/*********         GenScaleIm                                      ******/
/*********                                                         ******/
/************************************************************************/
/************************************************************************/

template <class TObj> 
         GenScaleIm<TObj>::GenScaleIm(Pt2di aSzU,Pt2di aSzW,INT NbChan) : 
		_tr 		(0,0),
		_sc         (0.0),	
		_CoeffPds	(0),
		_pW0		(0,0),
		_pW1		(0,0),
		_xU0		(0),
		_xU1		(0),
		_yU0		(0),
		_yU1		(0),
		_SzU		(aSzU),
		_u2wX		(NEW_VECTEUR(-RAB,aSzU.x+RAB,INT4)),
		_u2wY		(NEW_VECTEUR(-RAB,aSzU.y+RAB,INT4)),
		_Cw2uX      (NEW_VECTEUR(-RAB,aSzW.x+RAB,INT4)),
		_Cw2uY      (NEW_VECTEUR(-RAB,aSzW.y+RAB,INT4)),
		_SzW		(aSzW),
		_line       (NEW_MATRICE(Pt2di(-RAB,0),Pt2di(ElMax(aSzW.x,aSzU.x)+RAB,NbChan),TObj)),
		_l0			(_line[0]),
		_nb_chan	(NbChan)
{
}

template <class TObj> 
         GenScaleIm<TObj>::~GenScaleIm()
{
	DELETE_VECTOR(_u2wX		,-RAB);
	DELETE_VECTOR(_u2wY		,-RAB);
	DELETE_VECTOR(_Cw2uX	,-RAB);
	DELETE_VECTOR(_Cw2uY	,-RAB);
	DELETE_MATRICE(_line,Pt2di(-RAB,0),Pt2di(ElMax(_SzW.x,_SzU.x)+RAB,_nb_chan));
}

template <class TObj> 
         bool GenScaleIm<TObj>::do_it_gen(Pt2dr tr,REAL sc,Pt2di  pW0,Pt2di  pW1)
{

	_tr = tr;
	_sc = sc;
    XTransfo.Set(_tr.x,_sc);
    YTransfo.Set(_tr.y,_sc);


	_CoeffPds = round_ni(100.0/sc);
	pt_set_min_max(pW0,pW1);

	Pt2dr pu0 = Sup(Pt2dr(to_user(Pt2dr(pW0))),Pt2dr(0.0,0.0));
 	Pt2dr pu1 = Inf(Pt2dr(to_user(Pt2dr(pW1))),Pt2dr(_SzU));


	pW0 = Sup(Pt2di(0,0),round_down(to_window(pu0)));
	pW1 = Inf(_SzW		,round_up(to_window(pu1)));



	_pW0 = pW0;
	_pW1 = pW1;


	_xU0 = ElMax(0		, round_down(0.5+x_to_user(_pW0.x-0.5)));
	_xU1 = ElMin(_SzU.x	, round_up  (0.5+x_to_user(_pW1.x-0.5)));
	_yU0 = ElMax(0		, round_down(0.5+y_to_user(_pW0.y-0.5)));
	_yU1 = ElMin(_SzU.y	, round_up  (0.5+y_to_user(_pW1.y-0.5)));


    	if ((_xU0 >=_xU1) || (_yU0 >= _yU1))
           return false;

	for (INT ux=_xU0; ux<=_xU1 ; ux++)
	{
	     _u2wX[ux] =  FitInWX(PremPixelU2WX(ux));
    }

	for (INT uy=_yU0; uy<=_yU1 ; uy++)
	{
	     _u2wY[uy] =  FitInWY(PremPixelU2WY(uy));
    }

	for (INT wx= 0;wx<= _SzW.x ; wx++)	
	{
		 _Cw2uX[wx] = PixCenterW2U_x(wx);
	}
	for (INT wy=0;wy<= _SzW.y ; wy++)	
	{
		 _Cw2uY[wy] = PixCenterW2U_y(wy);
	}

     return true;
}


/************************************************************************/
/************************************************************************/
/*********                                                         ******/
/*********         Scale_Im_Compr                                  ******/
/*********                                                         ******/
/************************************************************************/
/************************************************************************/

template <class TObj,class TLut,class TInd > 
	void 	Scale_Im_Compr<TObj,TLut,TInd>::RunRLE(INT ux0,INT ux1,TInd ind)
{
   TObj col = _lut.ObjFromInd(ind);
 
   INT wx0 = this->_u2wX[ux0];
   this->_l0[wx0] +=    col *(_CurPds * _RLE_Pds_0[ux0]) ;
   this->_l0[wx0+1] +=    col *(_CurPds * _RLE_Pds_1[ux0]) ;
 
   INT wx1 = this->_u2wX[ux1];
   this->_l0[wx1]   -=    col *(_CurPds * _RLE_Pds_0[ux1]) ;
   this->_l0[wx1+1] -=    col *(_CurPds * _RLE_Pds_1[ux1]) ;

}                    

 
template <class TObj,class TLut,class TInd >
	void  Scale_Im_Compr<TObj,TLut,TInd>::RunLIT(INT ux0,INT ux1,const TInd * inds)
{
	for (INT ux=ux0; ux <ux1 ; ux++)
	{
		TObj col = _lut.ObjFromInd(*(inds++));
		INT wx = this->_u2wX[ux];
		this->_l0[wx  ] +=    col *(_CurPds * _LIT_Pds_0[ux]) ;
		this->_l0[wx+1] +=    col *(_CurPds * _LIT_Pds_1[ux]) ;
		if (_LIT_Pds_2[ux]) // tres souvent nulle
			this->_l0[wx+2] +=    col *(_CurPds * _LIT_Pds_2[ux]) ;
	}
/*

	for (INT ux=ux0; ux <ux1 ; ux++)
        RunRLE(ux,ux+1,inds[ux-ux0]);
*/
}





template <class TObj,class TLut,class TInd > 
	void 	Scale_Im_Compr<TObj,TLut,TInd>::DoItReduce()
{
    ElTimer aTimer;

	for (INT wy=this->_pW0.y; wy<this->_pW1.y; wy++)
	{
		INT yU0 = round_down(this->y_to_user(wy-0.5));
		INT yU1 = round_up(this->y_to_user(wy+0.5));


		ElSetMax(yU0,0);
		ElSetMin(yU1,this->_SzU.y-1);

		for (INT ux= this->_pW0.x -(Scale_Im_Compr<TObj,TLut,TInd>::RAB) ;
			ux<this->_pW1.x +(Scale_Im_Compr<TObj,TLut,TInd>::RAB); ux++)
			this->_l0[ux] = 0;



		INT pdsTot = 0;

		if (yU0<yU1)
		{

			for (INT uy= yU0; uy<=yU1 ; uy++)
			{
				REAL yW0 = ElMax(this->y_to_window(uy-0.5),wy-0.5);
				REAL yW1 = ElMin(this->y_to_window(uy+0.5),wy+0.5);
				
				INT pds = round_ni((yW1-yW0) * this->_CoeffPds);
				if (pds > 0)
				{
					pdsTot += 0;
					_CurPds = pds;
					DeCompr
					(
					        _dim->lpckb(uy),
						*this,
						this->_xU0,
						this->_xU1,
						_dim->per()
					);
					pdsTot += pds;
				}
			}
			if (pdsTot)
			{
				
				{
				for (INT ux= this->_pW0.x ;  ux<this->_pW1.x +(Scale_Im_Compr<TObj,TLut,TInd>::RAB) ; ux++)
					this->_l0[ux] += this->_l0[ux-1];
				}
				INT pxy = pdsTot * this->_CoeffPds;

				{
				for (INT ux= this->_pW0.x ;  ux<this->_pW1.x ; ux++)
					this->_l0[ux] /= pxy;
                }

                mTimeUnCompr += aTimer.uval();
				RasterUseLine
				(
					Pt2di(this->_pW0.x,wy),
					Pt2di(this->_pW1.x,wy+1),
					this->_line,
                                        1
				);
                aTimer.reinit();
			}
		}
	}
} 

template <class TObj,class TLut,class TInd> 
		class Zoom_Im_Compr 
						
{
		public :
			Zoom_Im_Compr(INT4 * U2WX,TObj * LINE,TLut & LUT) :
					_u2wX	(U2WX),
					_l0	(LINE),
					_lut 	(LUT)
			{
			}

   			inline void RunRLE(INT ux0,INT ux1,TInd ind)
			{
				INT wx0 = _u2wX[ux0];
				INT wx1 = _u2wX[ux1];
   				TObj col = _lut.ObjFromInd(ind);
				for (INT wx = wx0; wx<wx1 ; wx++)
					_l0[wx] = col;

			}

        	inline void RunLIT(INT ux0,INT ux1,const TInd * inds)
			{
				for (INT ux=ux0; ux<ux1; ux++)
				{
					INT wx0 = _u2wX[ux];
					INT wx1 = _u2wX[ux+1];
   					TObj col = _lut.ObjFromInd(*(inds++));
					for (INT wx = wx0; wx<wx1 ; wx++)
						_l0[wx] = col;
				}
			}

		private :

			INT4 *	_u2wX;
			TObj *	_l0;
			TLut &	_lut;
};


template <class TObj,class TLut,class TInd > 
	void 	Scale_Im_Compr<TObj,TLut,TInd>::DoItZoom()
{
    ElTimer aTimer;
	Zoom_Im_Compr<TObj,TLut,TInd>  ZIC(this->_u2wX,this->_l0,_lut);

	for (INT wy0=this->_pW0.y; wy0<this->_pW1.y; )
	{
		INT yU0 = round_ni(this->y_to_user(wy0));
		INT wy1 = wy0+1;

		while ( (wy1<this->_pW1.y) && (yU0==round_ni(this->y_to_user(wy1))))
			wy1++;

                yU0 = ElMax(0,ElMin(yU0,this->_SzU.y-1));
		DeCompr
                (
		     _dim->lpckb(yU0),
                     ZIC,this->_xU0,this->_xU1,_dim->per()
                );
        mTimeUnCompr += aTimer.uval();
		RasterUseLine
		(
			Pt2di(this->_pW0.x,wy0),
			Pt2di(this->_pW1.x,wy1),
			this->_line,
                        1
		);
        aTimer.reinit();

		wy0 = wy1;
	}
}



template <class TObj,class TLut,class TInd > 
	Box2di 	Scale_Im_Compr<TObj,TLut,TInd>::do_it 
			(
				Pt2dr	tr,
				REAL 	sc,
				Pt2di 	pW0, 
				Pt2di	pW1,
				bool    quick
			)
{

     if (! this->do_it_gen(tr,sc,pW0,pW1))
		return Box2di(pW0,pW0);


        for (INT c=0; c<this->_nb_chan ; c++)
            for (INT x=-(Scale_Im_Compr<TObj,TLut,TInd>::RAB); x< (Scale_Im_Compr<TObj,TLut,TInd>::RAB)+this->_SzW.x ; x++)
               this->_line[c][x] = 0;

	{
	for (INT ux=this->_xU0; ux<=this->_xU1 ; ux++)
    {
        INT pixW = this->PremPixelU2WX(ux);
        if (pixW == this->_u2wX[ux])
	    {
	         REAL p0  =  ElPixelCentre::LenghFromFinIntervPixelIncluant
                     (this->x_to_window(ElPixelCentre::DebIntervalPixel(ux)))   ;
		    _RLE_Pds_0[ux] = round_ni(this->_CoeffPds*p0);
		    _RLE_Pds_1[ux] = this->_CoeffPds-_RLE_Pds_0[ux];
	    }
        else if (pixW > this->_u2wX[ux])
        {
		    _RLE_Pds_0[ux] = 0;
            _RLE_Pds_1[ux] =  this->_CoeffPds;
        }
        else if (pixW < this->_u2wX[ux])
        {
		    _RLE_Pds_0[ux] = this->_CoeffPds;
            _RLE_Pds_1[ux] =  0;
        }
    }
	}

	{
	for (INT ux=this->_xU0; ux<this->_xU1 ; ux++)
	{
        INT pixW = this->PremPixelU2WX(ux);
        if (pixW == this->_u2wX[ux])
        {
		    if (this->_u2wX[ux] == this->_u2wX[ux+1])
		    {
			    _LIT_Pds_0[ux] = _RLE_Pds_0[ux]-_RLE_Pds_0[ux+1];
			    _LIT_Pds_1[ux] = -_LIT_Pds_0[ux];
			    _LIT_Pds_2[ux] = 0;
		    }
		    else
		    {
			    _LIT_Pds_0[ux] = _RLE_Pds_0[ux];
			    _LIT_Pds_2[ux] = -_RLE_Pds_1[ux+1];
			    _LIT_Pds_1[ux] = -(_LIT_Pds_0[ux]+_LIT_Pds_2[ux]);
		    }
		}
        else
        {
               _LIT_Pds_0[ux]=0;
               _LIT_Pds_1[ux]=0;
               _LIT_Pds_2[ux]=0;
        }
	}
	}


	if ((sc < 1.0) && (! quick))
		DoItReduce();
	else
	{
		DoItZoom();
	}


	return Box2di(pW0,pW1);
}

template <class TObj,class TLut,class TInd > 
	Scale_Im_Compr<TObj,TLut,TInd>::Scale_Im_Compr
	(
		PackB_IM<TInd> PBIM,
		Pt2di SzW
	)	:

		GenScaleIm<TObj>(PBIM.sz(),SzW,1),
	//	_lut		(),
	// Init Bidon  (refait par do_it)
		_CurPds		(0),

	//	
		_pbim		(PBIM),
		_dim		(PBIM.dpim()),
		_RLE_Pds_0	(NEW_VECTEUR(-(Scale_Im_Compr<TObj,TLut,TInd>::RAB),(Scale_Im_Compr<TObj,TLut,TInd>::_SzU.x)+(Scale_Im_Compr<TObj,TLut,TInd>::RAB),INT4)),
		_RLE_Pds_1	(NEW_VECTEUR(-(Scale_Im_Compr<TObj,TLut,TInd>::RAB),(Scale_Im_Compr<TObj,TLut,TInd>::_SzU.x)+(Scale_Im_Compr<TObj,TLut,TInd>::RAB),INT4)),
		_LIT_Pds_0	(NEW_VECTEUR(-(Scale_Im_Compr<TObj,TLut,TInd>::RAB),(Scale_Im_Compr<TObj,TLut,TInd>::_SzU.x)+(Scale_Im_Compr<TObj,TLut,TInd>::RAB),INT4)),
		_LIT_Pds_1	(NEW_VECTEUR(-(Scale_Im_Compr<TObj,TLut,TInd>::RAB),(Scale_Im_Compr<TObj,TLut,TInd>::_SzU.x)+(Scale_Im_Compr<TObj,TLut,TInd>::RAB),INT4)),
		_LIT_Pds_2	(NEW_VECTEUR(-(Scale_Im_Compr<TObj,TLut,TInd>::RAB),(Scale_Im_Compr<TObj,TLut,TInd>::_SzU.x)+(Scale_Im_Compr<TObj,TLut,TInd>::RAB),INT4)),
        mTimeUnCompr (0.0)
{
}

template <class TObj,class TLut,class TInd > 
		Scale_Im_Compr<TObj,TLut,TInd>::~Scale_Im_Compr()
{
	DELETE_VECTOR(_RLE_Pds_0,-(Scale_Im_Compr<TObj,TLut,TInd>::RAB));
	DELETE_VECTOR(_RLE_Pds_1,-(Scale_Im_Compr<TObj,TLut,TInd>::RAB));
	DELETE_VECTOR(_LIT_Pds_0,-(Scale_Im_Compr<TObj,TLut,TInd>::RAB));
	DELETE_VECTOR(_LIT_Pds_1,-(Scale_Im_Compr<TObj,TLut,TInd>::RAB));
	DELETE_VECTOR(_LIT_Pds_2,-(Scale_Im_Compr<TObj,TLut,TInd>::RAB));
}



/************************************************************************/
/************************************************************************/
/*********                                                         ******/
/*********         Lut_RGB_Int                                     ******/
/*********                                                         ******/
/************************************************************************/
/************************************************************************/

Lut_RGB_Int::Lut_RGB_Int() :
	_nb (0),
	_r (0),
	_g (0),
	_b (0)
{
}

void Lut_RGB_Int::init(Elise_colour * col,INT nb)
{
	_nb = nb ;
	_r =  NEW_VECTEUR(0,nb,INT4);
	_g =  NEW_VECTEUR(0,nb,INT4);
	_b =  NEW_VECTEUR(0,nb,INT4);
	for (INT k=0; k<_nb ; k++)
	{
		_r[k] = ElMax(0,ElMin(255,(INT)(255*col[k].r())));
		_g[k] = ElMax(0,ElMin(255,(INT)(255*col[k].g())));
		_b[k] = ElMax(0,ElMin(255,(INT)(255*col[k].b())));

	}
}

Lut_RGB_Int::~Lut_RGB_Int() 
{
	if (_r)
	{
		DELETE_VECTOR(_r,0);
		DELETE_VECTOR(_g,0);
		DELETE_VECTOR(_b,0);
	}
}

IntIdLut::IntIdLut() {}

/************************************************************************/
/************************************************************************/
/*********                                                         ******/
/*********         StdGray_Scale_Im_Compr                          ******/
/*********         RGBLut_Scale_Im_Compr                           ******/
/*********                                                         ******/
/************************************************************************/
/************************************************************************/


StdGray_Scale_Im_Compr::StdGray_Scale_Im_Compr(PackB_IM<U_INT1> PBIM,Pt2di SzW) :
		Scale_Im_Compr<INT,IntIdLut,U_INT1> (PBIM,SzW)
{
}


RGBLut_Scale_Im_Compr::RGBLut_Scale_Im_Compr
(
	PackB_IM<U_INT1>	PBIM,
	Pt2di 				SzW,
	Elise_colour * 		cols,
	INT 				nb
)	:
	Scale_Im_Compr<RGB_Int,Lut_RGB_Int,U_INT1>(PBIM,SzW)
{
	_lut.init(cols,nb);
}


RGBTrue16Col_Scale_Im_Compr::RGBTrue16Col_Scale_Im_Compr
(
         PackB_IM<U_INT2> PBIM,
         Pt2di            SzW 
)   :
    Scale_Im_Compr<RGB_Int,TrueCol16Bit_RGB,U_INT2> (PBIM,SzW)
{
}

/************************************************************************/
/************************************************************************/
/*********                                                         ******/
/*********         REDUC_Scale_Im_Compr                           ******/
/*********                                                         ******/
/************************************************************************/
/************************************************************************/


class Reducteur_Im_Compr  : public StdGray_Scale_Im_Compr
{
	public :
         Reducteur_Im_Compr (PackB_IM<U_INT1>,INT zoom);
		 void RasterUseLine(Pt2di p0,Pt2di p1,INT **,int aNbChanIn);


		~Reducteur_Im_Compr();
		enum {RAB =3};
 	    PackB_IM<U_INT1>   pbim () {return _pbim;}
	private : 

        //INT                       _zoom;
 	    PackB_IM<U_INT1>          _pbim;
 		Data_PackB_IM<U_INT1> *   _dpbim;
		U_INT1 *                  _line;
        ElSTDNS vector<U_INT2>  	  	  _LInd;
        ElSTDNS vector<U_INT2>  		  _VInd;
        ElSTDNS vector<U_INT1>  		  _Length;
        ElSTDNS vector<U_INT1>    		  _Vals;      
};


Reducteur_Im_Compr::~Reducteur_Im_Compr()
{
	 DELETE_VECTOR(_line,-RAB);
}


Reducteur_Im_Compr::Reducteur_Im_Compr
(
	PackB_IM<U_INT1>	PBIM,
	INT 				zoom
) :
	StdGray_Scale_Im_Compr  (PBIM,PBIM.sz()/zoom),
	//_zoom                   (zoom),
	 _pbim					(_SzW.x,_SzW.y,0,-128),
	_dpbim					(_pbim.dpim()),
	_line					(NEW_VECTEUR(-RAB,_SzW.x+RAB,U_INT1))
{
}


void  Reducteur_Im_Compr::RasterUseLine(Pt2di p0,Pt2di p1,INT ** im,int aNbChanIn)
{
	INT nb = p1.x-p0.x;
	convert(_line,im[0],nb);
	_dpbim->_LINES[p0.y].init
    (
        0,
        _LInd,
        _VInd,
        _Length,
        _Vals,
        _line,
        _pbim.sz().x,
        _dpbim->_per
    );          
}


ElImScroller * PckBitImScroller::Reduc(INT zoom,bool quick)
{
    Reducteur_Im_Compr  Red(_pbim,zoom);
    Red.do_it(Pt2dr(0,0),1.0/zoom,Pt2di(0,0),Red.SzW(),quick);

	return new PckBitImScroller(VisuStd(),Red.pbim(),sc_im()/zoom);
}



/************************************************************************/
/************************************************************************/
/*********                                                         ******/
/*********         INSTANCIATIONS                                  ******/
/*********                                                         ******/
/************************************************************************/
/************************************************************************/

template class Scale_Im_Compr<INT,IntIdLut,U_INT1>;
template class Scale_Im_Compr<RGB_Int,Lut_RGB_Int,U_INT1>;
template class Scale_Im_Compr<RGB_Int,TrueCol16Bit_RGB,U_INT2>;


template class GenScaleIm<int>;
template class GenScaleIm<double>;




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
