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

#ifndef _ELISE_PRIVATE_COMPR_IM_H_
#define _ELISE_PRIVATE_COMPR_IM_H_



class IntIdLut
{
	public :

		IntIdLut();
		INT  ObjFromInd(INT  x) {return x;}
};


class RGB_Int
{
	public :

		RGB_Int() :
			_r(0), _g(0), _b(0) 
		{}

		RGB_Int(INT R,INT G,INT B) :
			_r(R), _g(G), _b(B) 
		{}


		INT _r,_g,_b;
		void operator = (INT v) {_r = _g = _b = v;}

		void operator += (const RGB_Int & c2)
		{
			_r += c2._r;
			_g += c2._g;
			_b += c2._b;
		}
		void operator -= (const RGB_Int & c2)
		{
			_r -= c2._r;
			_g -= c2._g;
			_b -= c2._b;
		}
		RGB_Int operator * (INT s)
		{
			return RGB_Int(_r*s,_g*s,_b*s);
		}
		void operator /= (INT s)
		{
			_r /= s;
			_g /= s;
			_b /= s;
		}
};
extern std::ofstream & operator << (std::ofstream & ofs,const RGB_Int  &aRGB);

class Lut_RGB_Int
{
	public :	


		RGB_Int  ObjFromInd(int ind) 
		{
			return RGB_Int(_r[ind],_g[ind],_b[ind]);
		}

		Lut_RGB_Int();
		~Lut_RGB_Int();
		void init(Elise_colour *,INT nb);
		
		INT _nb;
		INT * _r;
		INT * _g;
		INT * _b;

};


class TrueCol16Bit_RGB
{
	private :	

		friend void bench_bitm_colour();
	    static void Bench();

		enum 
		{
				NbBits= 5,
				Complem8NbBit = 8-NbBits,
				TwoNbBits= NbBits*2,
				MasqBits=  ((1<<NbBits)-1)
		};
	
		static void  BufI2RGB
				     (
                           INT ** out_put,
                           INT ** in_put,
                           INT     nb,
                           const class Arg_Comp_Simple_OP_UN& arg
                     );

		static void  BufRGB2I
				     (
                           INT ** out_put,
                           INT ** in_put,
                           INT     nb,
                           const class Arg_Comp_Simple_OP_UN& arg
                     );

	public :	


		static Fonc_Num I2RGB(Fonc_Num); // equiv ObjFromInd
		static Fonc_Num RGB2I(Fonc_Num); // equiv IndFromObj

		static RGB_Int  ObjFromInd(int ind) 
		{
			return RGB_Int
			    	(
					     (ind &MasqBits)<<Complem8NbBit,
					     ((ind >> NbBits)  &MasqBits)<<Complem8NbBit,
					     ((ind >> TwoNbBits) &MasqBits)<<Complem8NbBit
					);
		};

		static int IndFromObj(RGB_Int aRBG)
		{
				return 
						 (aRBG._r>>Complem8NbBit) 
					|   ((aRBG._g>>Complem8NbBit)<<NbBits)
					|   ((aRBG._b>>Complem8NbBit)<<TwoNbBits);
		};
};


class ElPixelCentre
{
    public :
         static  INT  PixelIncluant(REAL aVal)   
		             {return round_ni(aVal);}


         static  INT  PixelInf(REAL aVal)   
		             {return round_down(aVal);}
         static  INT  PixelSup(REAL aVal)   
		             {return round_up(aVal);}

          static REAL PdsBarryPixelInf(REAL aVal)
		       {return 1.0 - (aVal-round_down(aVal));}

         static  REAL DebIntervalPixel(INT aPixel) 
		             {return aPixel-0.5;}

         static  REAL FinIntervalPixel(INT aPixel) 
		             {return 1+DebIntervalPixel(aPixel);}

	static  REAL LenghFromDebIntervPixelIncluant(REAL aVal)
		              {return aVal-DebIntervalPixel(PixelIncluant(aVal));}

	static  REAL LenghFromFinIntervPixelIncluant(REAL aVal)
		              {return 1-LenghFromDebIntervPixelIncluant(aVal);}
};


class ElTranlatScaleTransfo
{
      public :
          REAL U2W(REAL Uval) const {return (Uval-mTr) * mSc;}
          REAL W2U(REAL Wval) const {return (Wval/mSc) + mTr;}


		  // Soit W un pixel W, retourne le pixel user qui contient

		  void Set(REAL aTr,REAL aSc) {mTr=aTr;mSc=aSc;}
      // private :
          REAL                    mTr;
          REAL                    mSc;
};



template <class TObj> class GenScaleIm
{
	public :
		Pt2di SzW() const { return _SzW;}
		Pt2di SzU() const { return _SzU;}

	protected :


		bool do_it_gen(Pt2dr tr,REAL sc,Pt2di  pW0,Pt2di  pW1);

		virtual ~GenScaleIm();
		GenScaleIm(Pt2di SzU,Pt2di SzW,INT nb_chan);


		REAL x_to_window(REAL x) const {return XTransfo.U2W(x);}
		REAL x_to_user(REAL x)   const {return XTransfo.W2U(x);}

		REAL y_to_window(REAL y) const {return YTransfo.U2W(y);}
		REAL y_to_user(REAL y)   const {return YTransfo.W2U(y);}

	 	inline Pt2dr to_window(Pt2dr p) const
			{ return Pt2dr(x_to_window(p.x),y_to_window(p.y));}
	 	inline Pt2dr to_user(Pt2dr p) const 	
			{ return Pt2dr(x_to_user(p.x),y_to_user(p.y));}

		INT PremPixelU2WX(INT ux) const
		{
              return ElPixelCentre::PixelIncluant(x_to_window(ElPixelCentre::DebIntervalPixel(ux)));
		}
		INT PremPixelU2WY(INT uy) const
		{
              return ElPixelCentre::PixelIncluant(y_to_window(ElPixelCentre::DebIntervalPixel(uy)));
		}

		INT FitInWX(INT Val)   const {return  ElMax(0,ElMin(_SzW.x-1,Val));}
		INT FitInWY(INT Val)   const {return  ElMax(0,ElMin(_SzW.y-1,Val));}
		REAL FitInWX(REAL Val) const {return  ElMax(0.0,ElMin(_SzW.x-1.0,Val));}
		REAL FitInWY(REAL Val) const {return  ElMax(0.0,ElMin(_SzW.y-1.0,Val));}


		INT FitInUX(INT Val)   const {return  ElMax(0,ElMin(_SzU.x-1,Val));}
		INT FitInUY(INT Val)   const {return  ElMax(0,ElMin(_SzU.y-1,Val));}
		REAL FitInUX(REAL Val) const {return  ElMax(0.0,ElMin(_SzU.x-1.0,Val));}
		REAL FitInUY(REAL Val) const {return  ElMax(0.0,ElMin(_SzU.y-1.0,Val));}


		INT PixCenterW2U_x (INT wx) const
		{
			return FitInUX(ElPixelCentre::PixelIncluant(x_to_user(wx)));
		}
		INT PixCenterW2U_y (INT wy) const
		{
			return FitInUY(ElPixelCentre::PixelIncluant(y_to_user(wy)));
		}


		Pt2dr 		_tr;
		REAL 		_sc;
		ElTranlatScaleTransfo   XTransfo;
		ElTranlatScaleTransfo   YTransfo;
		INT		_CoeffPds;
		
		Pt2di		_pW0;
		Pt2di		_pW1;
		INT 		_xU0;
		INT		_xU1;
		INT 		_yU0;
		INT		_yU1;

		Pt2di		_SzU;
			// debut d'intervalle
		INT4 *	 	_u2wX; // -3 ... _SzU.x +3
		INT4 *	 	_u2wY; // -3 ... _SzU.x +3
			// centre de points
		INT4 *	 	_Cw2uX; // -3 ... _SzU.x +3
		INT4 *	 	_Cw2uY; // -3 ... _SzU.x +3
		Pt2di		_SzW;

		TObj **		_line;  //  -1  .. _SzW.x +1
		TObj *		_l0;  //  -1  .. _SzW.x +1
		INT         _nb_chan;
		enum {RAB = 4};

		     // Structure pour le coupage en morceau des pixels en zoom reduit



		
	private :
		GenScaleIm(const GenScaleIm &);

};




template <class TObj,class TLut,class TInd > class Scale_Im_Compr : public GenScaleIm<TObj>
{
	public :

		Box2di do_it(Pt2dr tr,REAL sc,Pt2di  pW0,Pt2di  pW1,bool quick);



	protected :
		virtual void RasterUseLine(Pt2di p0,Pt2di p1,TObj **,int aNbChanIn) = 0;
		virtual ~Scale_Im_Compr();
		Scale_Im_Compr(PackB_IM<TInd>,Pt2di SzW);

		TLut					_lut;



		INT		_CurPds;


		void DoItReduce();
		void DoItZoom();

		PackB_IM<TInd> 			_pbim;
		Data_PackB_IM<TInd> *	_dim;

		INT4 *	 		_RLE_Pds_0; // -3 ... _SzU.x +3
		INT4 *	 		_RLE_Pds_1; // -3 ... _SzU.x +3
		INT4 *	 		_LIT_Pds_0; // -3 ... _SzU.x +3
		INT4 *	 		_LIT_Pds_1; // -3 ... _SzU.x +3
		INT4 *	 		_LIT_Pds_2; // -3 ... _SzU.x +3


		REAL mTimeUnCompr; // 0.0


	private :
		Scale_Im_Compr(const Scale_Im_Compr<TObj,TLut,TInd> &);
	public :

		inline void RunRLE(INT ux0,INT ux1,TInd ind);
		inline void RunLIT(INT ux0,INT ux1,const TInd * inds);
};


class StdGray_Scale_Im_Compr : public  Scale_Im_Compr<INT,IntIdLut,U_INT1> 
{
	public :

		StdGray_Scale_Im_Compr(PackB_IM<U_INT1>,Pt2di SzW);
	private :
		StdGray_Scale_Im_Compr(const StdGray_Scale_Im_Compr &);
};


class RGBLut_Scale_Im_Compr : public  Scale_Im_Compr<RGB_Int,Lut_RGB_Int,U_INT1>
{
	public :

		RGBLut_Scale_Im_Compr
		(
			PackB_IM<U_INT1>,
			Pt2di SzW,
			Elise_colour *,
			INT nb
		);

	private :
		 RGBLut_Scale_Im_Compr(const RGBLut_Scale_Im_Compr&);
};


class RGBTrue16Col_Scale_Im_Compr : public  Scale_Im_Compr<RGB_Int,TrueCol16Bit_RGB,U_INT2>
{
    public :

		RGBTrue16Col_Scale_Im_Compr
		(
			PackB_IM<U_INT2>,
			Pt2di SzW
		);

	private :
		 RGBTrue16Col_Scale_Im_Compr(const RGBTrue16Col_Scale_Im_Compr&);
};


class PckBitImScroller : public ElImScroller ,
                         StdGray_Scale_Im_Compr
{
     public :

        Output out();
        Fonc_Num in();
        Pt2di SzIn() ;


        PckBitImScroller
        (
            Visu_ElImScr &Visu,
            PackB_IM<U_INT1>,
            REAL  sc_im 
        );



        virtual ElImScroller * Reduc(INT zoom,bool quick = false);
        virtual void SetPoly (Fonc_Num aFonc,std::vector<Pt2dr> VPts);
        virtual void ApplyLutOnPoly(Fonc_Num ,std::vector<Pt2dr>);



     private :

		virtual REAL TimeUnCompr() const; 
        void LoadXImage(Pt2di p0,Pt2di p1,bool quick);
        void RasterUseLine(Pt2di p0,Pt2di p1,INT **,int aNbChanIn);
		PckBitImScroller(const PckBitImScroller &);
};            


class  RGB_PckbImScr : public ElImScroller
{
	 public :
		RGB_PckbImScr(Visu_ElImScr & Visu,Pt2di Sz,REAL ScaleIm);

		virtual ~RGB_PckbImScr();
	 protected :
        void WriteRGBImage(Pt2di p0,Pt2di p1,RGB_Int **);
        Pt2di SzIn() ;
     private :

		Pt2di   mP0Im;
		Pt2di   mP1Im;
		INT **  mIm;
		INT *   mRIm;
		INT *   mGIm;
		INT *   mBIm;
};

class RGBLut_PckbImScr : public RGB_PckbImScr ,
                         RGBLut_Scale_Im_Compr
{
     public :

        Output out();
        Fonc_Num in();

        RGBLut_PckbImScr
        (
            Visu_ElImScr &Visu,
            PackB_IM<U_INT1>,
			Elise_colour *,
			INT nb,
            REAL  sc_im
        );

	private :

        void LoadXImage(Pt2di p0,Pt2di p1,bool quick);
        void RasterUseLine(Pt2di p0,Pt2di p1,RGB_Int **,int aNbChanIn);
		RGBLut_PckbImScr(const RGBLut_PckbImScr &);
		virtual REAL TimeUnCompr() const; 
};            



class RGBTrue16Col_PckbImScr : public RGB_PckbImScr,
                                      RGBTrue16Col_Scale_Im_Compr
{
     public :
        Output out();
        Fonc_Num in();

        RGBTrue16Col_PckbImScr
        (
            Visu_ElImScr &Visu,
            PackB_IM<U_INT2>,
            REAL  sc_im 
        );

	private :

        void LoadXImage(Pt2di p0,Pt2di p1,bool quick);
        void RasterUseLine(Pt2di p0,Pt2di p1,RGB_Int **,int aNbChanIn);
		RGBTrue16Col_PckbImScr(const RGBTrue16Col_PckbImScr &);
		virtual REAL TimeUnCompr() const; 
};






                                        


class  ElPixelUPond
{
			public :
				INT mPixU;
				INT mPds;
};

class ElIntervPixelUPond
{
       public :
              ElPixelUPond * mBegin;
              ElPixelUPond * mEnd;
              INT FirstPixel() const;
              INT LastPixel() const;
};

class ElTabIntervPUP
{
       public :
           typedef enum {Reduc,Lin,Cub} ModeAct;

           ~ElTabIntervPUP();
           ElTabIntervPUP(INT SzW,INT SzU,INT rab,ElTranlatScaleTransfo & aTransfo);
           void Actualise(INT PdsPerPixW,ModeAct);

           const ElIntervPixelUPond & IntervPixelPond(INT wPix);


		   void IntervalW2U(INT & u0,INT & u1,INT w0,INT w1);

       private :
           void Actualise_Reduc(INT PixW);
           void Actualise_Sinc(INT PixW,INT NbSinc);
           void Actualise_Lin(INT PixW);
           void Actualise_Cub(INT PixW);

		   INT FitInUser(INT);
		   void PushInterval(INT PixelU,REAL Pds);

           INT                     mSzW;
		   INT                     mIW0;
		   INT                     mIW1;
		   INT                     mNbW;
           INT                     mSzU;
           INT                     mIndTopRes;
           INT                     mNbRes;

		   REAL                   mPdsRealResiduel;
		   INT                    mPdsIntResiduel;

		   
           ElPixelUPond *          mRes;
           ElIntervPixelUPond *    mIntervs;
           ElTranlatScaleTransfo & mTransfo;
};



class IFL_LineWindow
{
	public :

		IFL_LineWindow(INT SzW,INT NbChan);
		~IFL_LineWindow();

		void reinit();
		INT mNbChan;
		INT mSzW;
		INT mLastX0;
		INT mLastX1;
		INT mLastY;
		INT ** mLine;
	private :
		IFL_LineWindow(const IFL_LineWindow &);
		void operator = (const IFL_LineWindow &);
};


template <class Type> class TilesIMFL;
template <class Type> class LoadedTilesIMFL;
template <class Type> class LoadedTilesIMFLAllocator;

// template <class Type> class  ImFileLoader : public GenScaleIm<INT> MPD
template <class Type> class  ImFileLoader : public GenScaleIm<typename El_CTypeTraits<Type>::tBase>
{
	    public :
               
                typedef typename El_CTypeTraits<Type>::tBase  tGSI;

                void ImFReInitTifFile(Tiff_Im aTif);

					 
		ImFileLoader(Tiff_Im,Pt2di SzW,ImFileLoader<Type> * ForAlloc = 0);
		virtual ~ImFileLoader();
									 
		void do_it(Pt2dr tr,REAL sc,Pt2di p0,Pt2di p1,bool quick);
													 
		ImFileLoader(ImFileLoader<Type> &,INT zoom);

		protected  :
                       Tiff_Im  Tiff();                        
			bool load_all(Pt2dr tr,REAL sc,Pt2di p0,Pt2di p1);
			Type * get_line_user(INT x0,INT x1,INT y);
			virtual void RasterUseLine(Pt2di p0,Pt2di p1,tGSI **,int aNbChanIn) =0;
																			 
                       void MakeOneLine(bool quick);
                       void MakeOneLineZooPPV();
                       void MakeOneLineReduceSomPPV();
                       void MakeOneLinePixelPond();

                       void put_tiles_in_alloc(bool FullReinit = false);
		
					   void load_this_tile(INT x,INT y);
		private :
			
			bool                             mByteOrdered;
			Tiff_Im                          * _tiff;
			INT                             _nb_chan;
			ELISE_fp *                      mFPGlob;
			Pt2di                           _nb_tile;
			Pt2di                           _sz_tile;
			ElSTDNS vector<ElSTDNS vector<TilesIMFL<Type> *> >    _tiles;
																															 
			Pt2di                             _tiles_0;
			Pt2di                             _tiles_1;
			LoadedTilesIMFLAllocator<Type> *  _alloc;
			bool                              _own_alloc;
			Type *	                          _uline;
		protected  :
			bool                            _dynamic;
		private :

			ElSTDNS vector<IFL_LineWindow *> mBufLW;
			INT ** GetLineWindow(INT x0U,INT x1U,INT y0U);
			void init_LW();



                        bool mPixReplic;
                        bool mReduc;

			INT _ux0cur;
			INT _ux1cur;
			INT _uy0cur;
			INT _uy1cur;
			INT mWyCur0;
			INT mWyCur1;
			ElTabIntervPUP   mXTabIntervales;
			ElTabIntervPUP   mYTabIntervales;

			ImFileLoader(const ImFileLoader<Type> &);

			bool mUpToDate_LastXY_GLU;
			INT mLastX0_GLU; // GLU = get_line_user
			INT mLastX1_GLU; // GLU = get_line_user
			INT mLastY_GLU; // GLU = get_line_user
                        Pt2di mSzTileFile;
                        bool  mHasTileFile;
                        
			ELISE_fp *                      mFPOfFileTile;
			Pt2di                           mCurTileOfFT;
			Pt2di                           mNbTTByF;

                        ELISE_fp * FileOfTile(Pt2di aTile);
};                                      

template <class Type> class ImFileScroller : public ImFileLoader<Type>,
                                             public ElImScroller
{
		public :
                                typedef typename El_CTypeTraits<Type>::tBase  tGSI;
                                bool CanReinitTif();
                                void ReInitTifFile(Tiff_Im aTif);


					 
				ImFileScroller
				(
					Visu_ElImScr &  V,
					Tiff_Im         tif,
					REAL            sc_im ,
					ImFileScroller*     alloc = 0
				);
                                virtual void no_use();
							 
				void LoadXImage(Pt2di p0,Pt2di p1,bool quick);
				void RasterUseLine(Pt2di p0,Pt2di p1,tGSI **,int aNbChanIn);
                virtual ElImScroller * Reduc(INT zoom,bool quick = false);
                Pt2di SzIn() ;

		private :
                                Fonc_Num in();
				ImFileScroller
				(
					ImFileScroller &,
					INT          zoom
				);
				ImFileScroller(const ImFileScroller &);
};                      

/*
    VOIR AUSSI :  "DOC/ArborScroller.fig"  pour un diagrammme de relations entre classes.


	Data_PackB_IM<TInd>  : public  RC_Object  
	-------------------

	       Memorise une image compressee et fournit une interface permettant de
		   la decompresser

	PackB_IM<TInd> 		
	-------------

	       Smart Pointer sur Data_PackB_IM<TInd> .

    GenScaleIm<TObj>  :
	------------------

	    fournit des services necessaire pour effectuer les changement
		de coordonnes entre espace user (U -> image) espace Window 
		(W-> visualisation)
     
	    La fonction do_it_gen met a jour les champs necessaire pour
	    mettre a jour la correspondances de coordonnees est :


		    bool do_it_gen(Pt2dr tr,REAL sc,Pt2di  pW0,Pt2di  pW1);

	        retourne vraie si la zone de validitee (point dans W avec homologue dans U)
		    est non vide



		TObj : type des donnees transitant (et donnant lieu a des allocation de buffer
		utilises par les derives). Ca ne semble plus  tres logique de parametrer a ce niveau 
		(raison historique ?);

	
	Scale_Im_Compr<TObj,TLut,TInd> :     public GenScaleIm<TObj>
	--------------------------------

	    Permet de decompresser de maniere generique une image de type RLE.

		La methode public est "do_it"

		      Box2di do_it(Pt2dr tr,REAL sc,Pt2di  pW0,Pt2di  pW1,bool quick);

			  Pour chaque ligne de W le necessitant , 
			  elle decompresse/met a l'echelle ce qui est necessaire
			  et appelle la methode "RasterUseLine" avec le resultat

		La methode a redefinir est "RasterUseLine":
		
		    void RasterUseLine(Pt2di p0,Pt2di p1,TObj **) = 0;

	        permet de definir ce que, dans un appel a do_it, on souhaite
			faire des donnees decompressees



	StdGray_Scale_Im_Compr :  Scale_Im_Compr<INT,IntIdLut,U_INT1>  
	-----------------------

	    Aujourd'hui plus ou - un typedef sur Scale_Im_Compr<INT,IntIdLut,U_INT1> 
		(ie heritage sans aucun "apport personnel")
	    

	
	Visu_ElImScr :
	--------------

		Une INTERFACE au sens JAVA . 
		Abstrait la notion de "fenetre raster avec gestion de buffer d'image" 
		(typiquement une ce buffer est une XImage).
	
		virtual Pt2di SzW() = 0;

		        La Visu_ElImScr doit pouboir indiquer sa taille.


		virtual void write_image(INT x0src,Pt2di p0dest,INT nb,INT ** data)  =0 ;
		   
		       Seul canal de communication permettant le transfert de donnee image.
		       La Visu_ElImScr doit memoriser la ligne data (commencant en x0src, de taille nb)
		       au point p0dest.


		virtual void load_rect_image(Pt2di p0,Pt2di p1) 

		      La fenetre doit charger les donnees image sur le rectangle p0,p1.


		virtual void translate(Pt2di) = 0

		      La fenetre doit se translater  de l'offset correspondant.



		 virtual void write_image_out(Pt2di p0_src,Pt2di p0_dest,Pt2di sz) =0;

			La fenetre doit transferer  une image "de son choix " depuis le rectangle
		    [p0_src, p0_src+sz]  vers le rectangle [p0_dest, p0_dest+sz].
			Typiquement c'est ce qui genere le motif eLiSe dans VideoWin_Visu_ElImScr.


     VideoWin_Visu_ElImScr  : public Visu_ElImScr
	 ------------------

		Classe concrete, repondant au protocole "Visu_ElImScr" par utilisation
		d'une fenetre XWindow et d'un image XImage.


	

     ElImScroller :
	 --------------

	    A terme, la partie emergee de l'iceberg, avec peut etre ElPyramScroller,
	   	(les autres classes passant dans les fichier private) de la lib.

	    Interface sur un Scroller d'image, definit les methodes necessaire pour 
		reafficher rapidement  l'image et etablir les correspondance de coordonnees.


		2 Fonction "fondamentale" : 

        	void set(Pt2dr tr,REAL sc,bool quick = false) : 

				modifie la geometrie et remet a jour l'affichage en consequence.
				si quick = true fait un affichage rapide

        	void SetDTrW(Pt2di tr);

				modifie la geometrie uniquement en translation.
				met a jour l'affichage en utilisant au max les
				fonctionnaites de translation de _visu


		2 Fonction "Auxiliaires", interface avec les precedente :

        	void SetTrU(Pt2dr tr) : tr est donnee en unite User

			void SetScArroundPW(Pt2dr PinvW,REAL sc,bool quick = false);
			   fait un changement d'echelle avec une translation telle
			   que PinvW soit invariant

		SzW() et SzU() : taille U et W
	
        Pt2dr to_win(Pt2dr p) , Pt2dr to_user(Pt2dr p)  chgt de coordonnees 





		3 Fonction qui seront rarement appelee (utile : mise au point, rafraichissement )

           void LoadIm(bool quick = false)    : charge l'image , prete a etre visulisee
           void VisuIm()                      :  visalise
           void LoadAndVisuIm(bool quick = false) :  enchaine les deux


*/


                              


class Gen_PackB_IM
{
      public :
             static inline INT CodeOfLengthRLE(INT aLength)
             {
                 return 1 + 2 *(aLength-1);
             }
             static inline INT CodeOfLengthLIT(INT aLength)
             {
                 return  2 *(aLength-1);
             }
};

template <class Type> class Line_PackB_IM  : public Gen_PackB_IM
{
	public :

                ~Line_PackB_IM();
                Line_PackB_IM();

                void init
                (       
				    INT                         BlockInit,
					ElSTDNS vector<U_INT2> &    LInd,
					ElSTDNS vector<U_INT2> &    VInd,
					ElSTDNS vector<U_INT1> &    Length,
					ElSTDNS vector<Type> &		Vals,
					const Type *		line =0,
					INT                 nb_tot =0,
					INT                 per =0
                );

           class RunsOfPer
           {
                 public :
                     ElSTDNS vector <U_INT1> mLRun;
                     ElSTDNS vector <Type>   mVRun;


                     inline int  run_length_pixel(INT ind) const
                     {
                            return mLRun[ind]/2+1; 
                     }
                     inline int  run_length_compr(INT ind) const
                     {
                            return run_rle(ind) ? 1 : run_length_pixel(ind);
                     }
                     inline bool run_rle(INT ind) const
                     {
                            return (mLRun[ind]&1) != 0; 
                     }
                     void Clear(){mLRun.clear();mVRun.clear();}

                     void PushRle(Type aVal,INT aLength)
                     {
                          mVRun.push_back(aVal);
                          mLRun.push_back(Gen_PackB_IM::CodeOfLengthRLE(aLength));
                     }

                     void PushRleSafe128(Type aVal,INT aLength)
                     {
                          while  (aLength > 128)
                          {
                              PushRle(aVal,128);
                              aLength -= 128;
                          }
                          PushRle(aVal,aLength);
                     }

                 private :
           };





	//  private : a cause des fton membre template non supportees
	public :
		// enum { MaxL = 128 };


		RunsOfPer  * mRuns;
		INT          mNbRuns;

		/*
		U_INT2 *   _indexes_Lrun;
		U_INT2 *   _indexes_Vrun;
		U_INT1 * 	_Lrun;
		Type *		_Vrun;
		*/
};

/*
template <class Type,class TAct>
void        DeCompr
            (
                 Line_PackB_IM<Type> & aLPBIM,
                 TAct & act,
                 INT x0,
                 INT x1,
                 INT per
            );

*/


template <class Type,class TAct>
inline void        DeCompr
            (
                 Line_PackB_IM<Type> & aLPBIM,
                 TAct & act,
                 INT x0,
                 INT x1,
                 INT per
            )
{
	typename Line_PackB_IM<Type>::RunsOfPer *  aROP = aLPBIM.mRuns + x0/per;
	INT Vind = 0;
	INT Lind = 0;
	INT x = (x0/per) * per;

	while (x<x0)
	{
			x += aROP->run_length_pixel(Lind);
			Vind += aROP->run_length_compr(Lind);
			Lind++;
	}

	if (x>x0)
	{
			int x_end = ElMin(x,x1);
			int LiPrec = Lind-1;
			int ViPrec = Vind-aROP->run_length_compr(LiPrec);
			INT xprec = x-aROP->run_length_pixel(LiPrec);
			Type * adr = & (aROP->mVRun[ViPrec]);

			if (aROP->run_rle(LiPrec))
				act.RunRLE(x0,x_end,*adr);
			else
				act.RunLIT(x0,x_end,adr+x0-xprec);
	}
			
	while(x<x1)
	{
			if (Lind == (INT) aROP->mLRun.size())
			{
				aROP++;
				Vind =0;
				Lind =0;
			}
			INT xn = ElMin(x1,x+aROP->run_length_pixel(Lind));
			if (aROP->run_rle(Lind))
				act.RunRLE(x,xn,aROP->mVRun[Vind]);
			else
				act.RunLIT(x,xn,&(aROP->mVRun[Vind]));
			x = xn;
			Vind += aROP->run_length_compr(Lind);
			Lind++;
	}
}                                          

template <class Type> class Init_Data_PackB_IM;
template <class Type> class DPIM_Im_Comp;
template <class Type> class DPIM_In_Not_Comp;
template <class Type> class DPIM_Out_Comp;

template <class Type> class Data_PackB_IM :	public	RC_Object 
{
	public :

		friend class Init_Data_PackB_IM<Type>;
		friend class DPIM_Im_Comp<Type>;
		friend class DPIM_In_Not_Comp<Type>;
		friend class DATA_Tiff_Ifd;
		friend class Reducteur_Im_Compr;
		friend class DPIM_Out_Comp<Type>;


		~Data_PackB_IM();
		// per <0, convention pour ne pas initialiser avec Fonc_Num
		Data_PackB_IM (INT tx,INT ty,Fonc_Num,INT per =128);


//		Fonc_Num in(INT def,bool with_def);
// Modif DB
		Pt2di sz() const {return Pt2di(_tx,_ty);}
		INT tx() const {return _tx;}
		INT ty() const {return _ty;}

		 Line_PackB_IM<Type> & lpckb(INT y) {return _LINES[y];}
		 INT per() const {return _per;}
		
	private :
		Line_PackB_IM<Type>	* _LINES;
		INT _tx,_ty,_per;

		INT _p0[2];
		INT _p1[2];
};




#endif // _ELISE_PRIVATE_COMPR_IM_H_

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
