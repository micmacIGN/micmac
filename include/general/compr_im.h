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

#ifndef _ELISE_GENERAL_COMPR_IM_H
#define _ELISE_GENERAL_COMPR_IM_H

double VerifInt(const int    * anInput,int aNb);
double VerifInt(const double * anInput,int aNb);




template <class Type> class Data_PackB_IM;

template <class Type> class PackB_IM :  public PRC0        ,
            public Rectang_Object

{
     public :
        friend class DATA_Tiff_Ifd;

        PackB_IM(INT tx,INT ty,Fonc_Num,INT per=128);
        Fonc_Num in();
        Output out();
        Output OutLut(Fonc_Num aLut);
        Fonc_Num in(INT def);
        Pt2di sz() const;
        virtual Elise_Rect box() const;

        const Data_PackB_IM<Type> * dpim() const;
        Data_PackB_IM<Type> * dpim() ;
        PackB_IM(Data_PackB_IM<Type> *);
    private :
};


class ElImScroller;
class ElImIncruster;


// Permet de coder la soustraction de deux boite
//  comme une union de boite
// (de maniere minimale et non  redondante si je ne m'abuse)

class ModelBoxSubstr
{
     public :
		bool MakeIt(Box2di ToAdd,Box2di ToSubstr); // return true if InterIsEmpty
		INT NbBox() const {return mNbBox;}
		const Box2di & Box(INT k) const {return mBoxes[k];}
		ModelBoxSubstr();
		ModelBoxSubstr(Box2di ToAdd,Box2di ToSubstr);
     private :
		void AddBox(Pt2di p0,Pt2di p1);
        INT    mNbBox;
		Box2di mBoxes[4];
};


class ModelDiffBox
{
    public :

		void MakeIt(Box2di OldOne,Box2di NewOne);

		bool  InterIsEmpty() const {return mInterIsEmpty;}
		const Box2di & Inter() const
	    {
             ELISE_ASSERT(!mInterIsEmpty,"Empty Inter in ModelDiffBox::Inter");
			 return mInter;
		}

		INT    NbEnPlus()  const {return mEnPlus.NbBox();}
		INT    NbEnMoins() const {return mEnMoins.NbBox();}

		const Box2di & EnPlus(INT k) const {return mEnPlus.Box(k);}
		const Box2di & EnMoins(INT k) const {return mEnMoins.Box(k);}

    private :

		Box2di mInter;
		bool   mInterIsEmpty;

		ModelBoxSubstr   mEnPlus;
		ModelBoxSubstr   mEnMoins;
};


void GetBoxUpdate(Pt2di tr,Box2di box,INT rab,Box2di & BX,Box2di & BY);





class cImgVisuChgDyn
{
    public :
        virtual void ChgDyn(int * anOut,const int * anInput,int aNb) = 0;
        virtual void ChgDyn(int * anOut,const double * anInput,int aNb) = 0;
    private :
};

class Visu_ElImDest;

template <class Type> class FriendVisu_ElImDest
{
   public :
       static void write_image(Visu_ElImDest&,INT  x0src,Pt2di p0dest,INT nb,Type ** data,int** aDataBuf,int aNbChanelIn);
};

class Visu_ElImDest
{
      public :
		   friend class ElImScroller;
                   friend class FriendVisu_ElImDest<int>;
                   friend class FriendVisu_ElImDest<double>;

                   virtual ~Visu_ElImDest();
		   virtual void VerifDim(INT DimOut) =0;

		   void SetGamaCorr(REAL aGamaFact);
                   void write_image(INT x0src,Pt2di p0dest,INT nb,INT ** data,int aNbChanelIn);
                   void write_image(INT x0src,Pt2di p0dest,INT nb,double ** data,int aNbChanelIn);

		   Visu_ElImDest(Pt2di aSz,INT aDimOut);

		   void SetEtalDyn(INT aVmin,INT aVmax,bool SetUse=true);
		   void SetUseEtalDyn(bool UseEtalDyn);
		   INT VMax() const;
		   INT VMin() const;

                   void SetChgDyn(cImgVisuChgDyn *);

	  protected :
		   // Visu_ElImDest();
      // private :
                   virtual void write_image_brute(INT x0src,Pt2di p0dest,INT nb,INT ** data) =0;
		   Visu_ElImDest(const Visu_ElImDest &);

		   INT             mDimOut;
		   Pt2di           mSzBigIm;
		   Pt2di           mSzBuf;
		   Im2D_INT4       mBufI;
		   INT4 **         mDataBufI;
		   bool            mUseEtalDyn;
		   INT             mVMin;
		   INT             mVMax;
		   INT             mDiff;

		   REAL           mGama;
		   Im1D_INT4      mImGamaCorr;
		   INT4 *         mDataGamaCorr;
		   bool           mUseGamaCorr;

                   cImgVisuChgDyn * mIVCD;
};





template <class Type> class Memory_ElImDest : public Visu_ElImDest
{
    public :
         Memory_ElImDest(INT NbChan,Pt2di SzTot);

         void SetDec(Pt2di pt);  // ajuste le decalage de maniere a ce que pt matche mCenter

//         INT NbCh() const;
// Modif DB : pas de code trouve

         std::vector<Im2D<Type,INT4> > & Images();
//         void RazImages();
// Modif DB : pas de code trouve

    private :

         INT    mNbChan;
         Pt2di mSzImTot;
         Pt2di mDec;

         virtual void write_image_brute(INT x0src,Pt2di p0dest,INT nb,INT ** data);
         std::vector<Im2D<Type,INT4> > mImages;
         Memory_ElImDest(const Memory_ElImDest &);

         void VerifDim(INT DimOut);
};





class Visu_ElImScr : public Visu_ElImDest
{
	  friend class ElImScroller;
	  friend class ElImIncruster;

      public :

           virtual ~Visu_ElImScr();
           virtual Pt2di SzW() = 0;
		   virtual void AdaptTiffFile(Tiff_Im,bool AdaptPal,bool ForceGray);

	 protected :
	 	Visu_ElImScr(Pt2di aSz,INT aDimOut);
     private :
		   virtual void InscrustImage(std::vector<Im2D_INT4> &,Pt2di p0Src,Pt2di P0Dest,Pt2di sz,INT NbIm) = 0;
		   virtual void InscrustImage(std::vector<Im2D_INT4> &,Pt2di p0Src,Pt2di P0Dest,Pt2di sz,INT NbIm,Elise_Palette) = 0;
           virtual void load_rect_image(Pt2di p0,Pt2di p1,bool Quick) =0;
           virtual void translate(Pt2di) = 0;
           virtual void write_image_out(Pt2di p0_src,Pt2di p0_dest,Pt2di sz) =0;
           Visu_ElImScr(const Visu_ElImScr &);
};


class VideoWin_Visu_ElImScr  : public Visu_ElImScr
{
     public :

         Video_Win  W() {return _w;}

         Pt2di SzW() {return _w.sz();}
         VideoWin_Visu_ElImScr(Video_Win,Elise_Palette,Pt2di SzImIncr = Pt2di(1,1),INT vMin=0,INT vMax=255);
         VideoWin_Visu_ElImScr(Video_Win,Tiff_Im,Pt2di SzImIncr = Pt2di(1,1),bool ForceGray=false);   // vMin,vMax =>Tiff_Im


		 virtual void AdaptTiffFile(Tiff_Im,bool AdaptPal,bool ForceGray);



		 void XIm_write_image_brute
			  (
			        Pt2di P0Src,
					Pt2di P0Dest,
					Pt2di Sz,
					Im2D_U_INT1 ImR,
					Im2D_U_INT1 ImG,
					Im2D_U_INT1 ImB
			  );

	protected :
         virtual void load_rect_image(Pt2di p0,Pt2di p1,bool Quick);
     private :
		 virtual void VerifDim(INT DimOut);
         virtual void write_image_brute(INT  x0src,Pt2di p0dest,INT nb,INT ** data);
         virtual void translate(Pt2di);
         virtual void write_image_out(Pt2di p0_src,Pt2di p0_dest,Pt2di sz);
	 virtual void InscrustImage(std::vector<Im2D_INT4> &,Pt2di p0Src,Pt2di P0Dest,Pt2di sz,INT NbIm);
	 virtual void InscrustImage(std::vector<Im2D_INT4> &,Pt2di p0Src,Pt2di P0Dest,Pt2di sz,INT NbIm,Elise_Palette);



		static ElXim StdIm(Video_Win w);


		ElXim             mBigIm;
		ElXim             mImPer;
		Pt2di             mSzImIncr;
		ElXim             mImIncr;
		Video_Win         _w;
		Elise_Palette   _pal;
		VideoWin_Visu_ElImScr(const VideoWin_Visu_ElImScr &);


};


class ElPyramScroller;

class cElScrCalcNameSsResol
{
     public :
         virtual std::string  CalculName(const std::string & aName, INT InvScale) = 0;

};

class ElImScroller
{
     public :
               virtual ElImScroller * CurScale() ;   // Default this => utilise dans Pyram
               virtual bool CanReinitTif();
               virtual void ReInitTifFile(Tiff_Im aTif);


               virtual void  SetAlwaysQuick(bool aVal);
               virtual void  SetAlwaysQuickInZoom(bool aVal);
               virtual void  SetAlwaysQuickInZoom();
               virtual void  SetAlwaysQuick();


               bool AlwaysQuick() const;
               bool AlwaysQuickZoom() const;

		virtual Output out(); //  Output::onul();
		virtual Fonc_Num in(); // 0
                virtual Pt2di SzIn() = 0;
                virtual REAL  GetValPtsR(Pt2dr aP);  // Def Erreur Fatale

                virtual void Sauv(const std::string & aName);
                virtual void SetPoly(Fonc_Num ,std::vector<Pt2dr>);
                virtual void ApplyLutOnPoly(Fonc_Num ,std::vector<Pt2dr>);

                 Flux_Pts ContVect2RasFlux(std::vector<Pt2dr> VPts);

		// DEBUG-OPTIM-MAP ;
		virtual REAL TimeLoadXIm() const; // 0.0
		virtual REAL TimeUnCompr() const; // 0.0
		virtual REAL TimeReformat() const; // 0.0



		// Si EchAcc = 0, ou si EchAcc->size() =0, alors aucune selection n'est
		// faite
	    static ElPyramScroller * StdPyramide
				     (
					Visu_ElImScr &Visu,
					const std::string &,
					std::vector<INT> * EchAcc =0,
					bool Adapt =false,
					bool ForceGray =false,
                                        cElScrCalcNameSsResol * = 0
			);

	    friend class ElPyramScroller;

        virtual ~ElImScroller();

        void LoadAndVisuIm(bool quick = false);
        void LoadAndVisuIm(Pt2di aPW0,Pt2di aPW1,bool quick = false);
        void LoadIm(bool quick = false);
        void VisuIm(bool Quick);
        void VisuIm(Pt2di PW0,Pt2di PW1,bool Quick);

        void set(Pt2dr tr,REAL sc,bool quick = false);
        void set_max(bool quick = false);
        void set_max_init();
        void SetSameGeom(const ElImScroller &);
        void SetGeomTranslated(const ElImScroller &,Pt2dr tr);

		ModelBoxSubstr SetDTrW(Pt2di tr);
        void SetTrU(Pt2dr tr);
		void SetScArroundPW(Pt2dr PinvW,REAL sc,bool quick = false);

        Pt2dr to_win(Pt2dr p)  const {return  (p-_tr) *_sc;}
        Pt2dr to_user(Pt2dr p) const
		{
				return _tr + p/_sc;
		}



         ElSimilitude to_user () const {return ElSimilitude(_tr,Pt2dr(1/_sc,0.0));}
         ElSimilitude to_win () const  {return to_user ().inv();}

         // transforme une similitude WW en une similitude UU
         ElSimilitude WW2UU (ElSimilitude aSim) const
         {
              return to_user () * aSim * to_win ();
         }



        Pt2di           SzW()  const {return  _SzW;}
        Pt2di           SzU()  const {return  _SzU;}
        // TCompl
        Pt2di           SzUIm()  const {return  Pt2di(Pt2dr(_SzU) / _sc_im);}


        void LoadXImageInVisu(Visu_ElImDest &,Pt2di p0W,Pt2di p1W,bool quick);
        void LoadXImageInVisu(Visu_ElImDest &,Pt2di p0W,Pt2di p1W,bool quick,Pt2dr tr,REAL sc);

	// Plutot utilisee pour la mise au point
        Pt2dr tr() const {return _tr;}
        REAL sc() const {return _sc;}
        REAL sc_abs() const {return _sc * _sc_im;}
        REAL sc_im() const {return  _sc_im;}

		INT DimOut() const {return mDimOut;}
        Visu_ElImScr &  VisuStd () { return mVisuStd;}

		Pt2dr TrOfScArroundPW(Pt2dr aPinvW,REAL aNewSc);
     protected :

        ElImScroller
        (
            Visu_ElImScr &Visu,
		    INT  aDimOut,
            Pt2di SzU,
            REAL  sc_im = 1.0
        );

        REAL ScMax() const;

        virtual void no_use();

        Pt2di           _SzW;
        Pt2di           _SzU;
        void write_image(INT x0src,Pt2di p0dest,INT nb,INT ** data,int aNbChanelIn);
        void write_image(INT x0src,Pt2di p0dest,INT nb,double ** data,int aNbChanelIn);
/*
        {
              mVisuCur->write_image(x0src,p0dest,nb,data);
        }
        void write_image(INT x0src,Pt2di p0dest,INT nb,double ** data)
        {
              mVisuCur->write_image(x0src,p0dest,nb,data);
        }
*/


        void SetVisuCur(Visu_ElImDest *);
        REAL mTimeLoadXIm;
        REAL mTimeReformat;




     public :

	    static ElImScroller * StdScrollIfExist(Visu_ElImScr &Visu,const std::string &,REAL scale,bool Adapt,bool ForceGray);
     private :
	    static ElImScroller * StdFileGenerique(Visu_ElImScr &Visu,const std::string &,INT  InvScale,bool Adapt,bool ForceGray,cElScrCalcNameSsResol * = 0);




		virtual void ReflexSetVisuCur(Visu_ElImDest *);

        virtual void LoadXImage(Pt2di p0W,Pt2di p1W,bool quick) =0;
        void LoadAndVerifXImage(Pt2di p0W,Pt2di p1W,bool quick) ;

        Visu_ElImScr &  mVisuStd; // celui par defaut
        Visu_ElImDest *  mVisuCur; // celui utilise dans  LoadXImage(Visu_ElImScr &,..)


		INT                     mDimOut;
        Pt2dr                       _tr;
        REAL                        _sc;
		const REAL               _sc_im;

        bool mAlwaysQuickInZoom;
        bool mAlwaysQuick;
        bool mSetInit;

        void set_geom(Pt2dr tr,REAL sc);
		bool  image_full_in() ;
		Pt2di PrefXim();
        void GetIntervUpdate(INT dx,INT tx,INT &x0,INT & x1);
        enum
        {
            RabUpdate = 2
        };
		ElImScroller(const ElImScroller &);
};

class ElPyramScroller : public ElImScroller
{
    public :
       ElPyramScroller(ElSTDNS vector<ElImScroller *> & scrolls);
       const ElSTDNS vector<ElImScroller *> & SubScrolls();


		virtual REAL TimeLoadXIm() const; // 0.0
		virtual REAL TimeUnCompr() const; // 0.0
		virtual REAL TimeReformat() const; // 0.0
		Fonc_Num in(); // 0

           virtual void  SetAlwaysQuick(bool aVal);
           virtual void  SetAlwaysQuickInZoom(bool aVal);
           virtual void  SetAlwaysQuickInZoom();
           virtual void  SetAlwaysQuick();
           virtual ElImScroller * CurScale() ; 
           virtual Pt2di SzIn() ;
	private :

           void Sauv(const std::string & aName);
           void SetPoly(Fonc_Num ,std::vector<Pt2dr>);
           virtual void ApplyLutOnPoly(Fonc_Num ,std::vector<Pt2dr>);
           void LoadXImage(Pt2di p0,Pt2di p1,bool quick);
           void ReflexSetVisuCur(Visu_ElImDest *);

           ElSTDNS vector<ElImScroller *> _subs;
           ElImScroller * _cur;
           ElPyramScroller(const ElPyramScroller &);
};

class ElImIncr_ScrAttr
{
	public :
		ElImIncr_ScrAttr() : mIsActif(true) {}
		bool mIsActif;
	private :
};


class ElImIncruster
{
		public :

			virtual ~ElImIncruster();

			ElImIncruster
			(
				Pt2di                           SzIm,
				Pt2di                           SzBrd,
				Visu_ElImScr &                  Visu,
				const ElSTDNS vector<ElImScroller*>   & Scrols,
				INT                             NbImRab = 0
			);

			void SetActiveKthIm(INT Kth,bool active);

			void BeginIncrust();
			void BeginIncrust(Pt2dr aPInvW,REAL aScale);
			void EndIncrust();

			void IncrustOrigineAtPtW(Pt2di pt);
			void IncrustOrigineAtPtW(Pt2di pt,Elise_Palette);
			void IncrustCenterAtPtW(Pt2di pt);
			void IncrustCenterAtPtW(Pt2di pt,Elise_Palette);
			void IncrSzAndReaff(Pt2di pCenter,Pt2di dSz);
			void IncrSzAndReaff(Pt2di pCenter,Pt2di dSz,Elise_Palette);

			Pt2di P0Tot() const;
			Pt2di P1Tot() const;

			Pt2di P0Uti() const;
			Pt2di P1Uti() const;

			Pt2di CurSize() const;
			Pt2di CurBrd() const;
			Pt2di BrdMax() const;
			void SetBrd(Pt2di);

			Box2di BoxIncrust() const;
			INT    NbImInMax() const;

		protected :

		private  :

			Pt2dr  TrScrol(INT k);
			REAL  ScaleScrol(INT k);


			void SetSize(Pt2di);
			void IncrSzAndReaff(Pt2di pCenter,Pt2di dSz,Elise_Palette *);
			void SetBox(Box2di NewBox,Elise_Palette * aPal);
			void SetNewBoxFromScratch(Box2di NewBox,Elise_Palette * aPal);
			void SetNewBoxDiffer(Box2di NewBox,Elise_Palette * aPal);

			void InitSz();

			void IncrustOrigineAtPtW(Pt2di pt,Elise_Palette *);
			virtual INT Filters
                                    (
                                          ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
                                          ElSTDNS vector<Im2D_INT4> &        OutPut,
                                          ElSTDNS vector<Im2D_INT4> &        InPut,
                                          Pt2di p0,
                                          Pt2di p1
                                    ) =0;
			            // return value = Nombre de cannaux utiles

			virtual bool PixelIndep() const =0; // Si PixelIndep vaut true la valeur calculee en 1 pixel
			                              // reste valable apres changement de  tr  ou de sz


			Pt2di mSzBrd0;
			Pt2di mSzImUti0;
			Pt2di mSzImTot0;

			Pt2di mSzBrdCur;
			Pt2di mSzImUtiCur;
			Pt2di mSzImTotCur;
			Pt2di mCenter;
			Pt2di mDec;

			Box2di mOldBox;


			std::vector<Im2D_INT4>           mImagesIn;
			std::vector<Im2D_INT4>           mImagesOut;

			std::vector<ElImScroller*>       mTabScr;
			std::vector<ElImIncr_ScrAttr>    mTabAttrScr;
			Pt2dr  mTrScrol;
			REAL   mScaleScrol;
			std::vector<Memory_ElImDest<INT4> *>   mMEID;
			Visu_ElImScr &                   mVisu;
			bool                             mLastDecUpToDate;
                        bool                             mSynchroGeom;
};


// Differentes classes offrant des filtres d'incrustation predefinis

class Filtr_Incr_EqDyn_Glob
{
	public :
         Filtr_Incr_EqDyn_Glob() ;
         INT Filters
             (
                     ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
                     ElSTDNS vector<Im2D_INT4> & OutPut,
                     ElSTDNS vector<Im2D_INT4> &InPut,
                     Pt2di p0,Pt2di p1,REAL Fact=1
             );

    private :

          Im1D_U_INT1 mLut;
          U_INT1 *    mDataLut;
};

class ConvMne_Filtr_Incr_EqDyn_Loc {};

class Filtr_Incr_EqDyn_Loc
{
	public :
         Filtr_Incr_EqDyn_Loc() ;
         Filtr_Incr_EqDyn_Loc(ConvMne_Filtr_Incr_EqDyn_Loc,INT ValSpec) ;
         INT Filters
             (
                  ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
                  ElSTDNS vector<Im2D_INT4> & OutPut,
                  ElSTDNS vector<Im2D_INT4> &InPut,
                  Pt2di p0,Pt2di p1,INT Brd,REAL Fact=1
             );

    private :
         bool mConvMne;
         INT  mValSpec;

};

class Filtr_Incr_EtalDyn
{
	public :
         Filtr_Incr_EtalDyn() ;
         INT Filters
             (
                  ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
                  ElSTDNS vector<Im2D_INT4> & OutPut,
                  ElSTDNS vector<Im2D_INT4> &InPut,
                  Pt2di p0,Pt2di p1,REAL Vmin,REAL Vmax
             );

    private :

};

class Filtr_Incr_NormDeriche
{
	public :
         Filtr_Incr_NormDeriche() ;
         INT Filters
             (
                  ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
                  ElSTDNS vector<Im2D_INT4> & OutPut,
                  ElSTDNS vector<Im2D_INT4> &InPut,
                  Pt2di p0,Pt2di p1,REAL FactDer,REAL Dyn
             );

    private :

};

class Filtr_Incr_AngDeriche
{
	public :
         Filtr_Incr_AngDeriche() ;
         INT Filters
             (
                  ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
                  ElSTDNS vector<Im2D_INT4> & OutPut,
                  ElSTDNS vector<Im2D_INT4> &InPut,
                  Pt2di p0,Pt2di p1,REAL FactDer
             );

    private :

};









         // Pop Up Menu avec transparence

class PopUpMenuTransp
{
      public :

        virtual ~PopUpMenuTransp();
        PopUpMenuTransp(Video_Win aW,Pt2di aSz);


        void UpP0(Pt2di p0Im);
        void UpCenter(Pt2di p0Im);
        bool PtInsideMenu(Pt2di pt);
		Box2di BoxInFen() const;


      protected :

        void Pop();
        Pt2di P0Cur() const;

        void reset_im_ori(Pt2di P0,Pt2di p1);
        void load_compute_show_rgb(Pt2di P0Src,Pt2di P0Dest,Pt2di sz);

        virtual void load_rgb(Pt2di P0Src,Pt2di P0Dest,Pt2di sz);
        void show_rgb(Pt2di P0Src,Pt2di P0Dest,Pt2di sz);

        virtual void compute_rgb ( Pt2di P0,Pt2di p1) ;


        void AdaptP0(Pt2di & p0Im);


        Pt2di         mSz;
        Video_Win     mW;
        ElXim         mWStdIm;
        ElXim         mWAffIm;

        Im2D_U_INT1   mImMotif;
        Im2D_U_INT1   mImTransp;

        Im2D_U_INT1   mImR;
        Im2D_U_INT1   mImG;
        Im2D_U_INT1   mImB;

        U_INT1**       mDataM;  // Motif
        U_INT1**       mDataT;  // Transparence
        U_INT1**       mDataR;
        U_INT1**       mDataG;
        U_INT1**       mDataB;

        bool          mUp;
        Pt2di         mP0Cur;
};

class CaseGPUMT;

class GridPopUpMenuTransp : public PopUpMenuTransp
{
    public :
		friend class CaseGPUMT;
        GridPopUpMenuTransp
        (
             Video_Win aW,
             Pt2di aSzGrid,
             Pt2di aNbGrid,
             Pt2di aBrdGrid
       );



       void  SetPtActif(Pt2di Pt);
       CaseGPUMT * PopAndGet();
	   void SetBorderMotif(Pt2di iGrid);

       Pt2di SzGrid() const {return mSzGrid;};
    private :


       void   VisuCaseGridActif(Pt2di iGrid);
       void   EffaceCaseGridActif(Pt2di iGrid);

       virtual void compute_menu_actif(Pt2di P0,Pt2di p1) ;

       Pt2di Indice2P0Grid(Pt2di iGrid) const;
       Pt2di Indice2P1Grid(Pt2di iGrid) const;
       Pt2di Pt2Indice(Pt2di Pt) const;

       Pt2di mSzGrid;
       Pt2di mNbGrid;
       Pt2di mBrdGrid;

       Pt2di mIndGrAct;
       bool  mGridIsAct;
	   std::vector<std::vector<CaseGPUMT *> > mCases;
	   CaseGPUMT *                            mCurCase;
};

class CaseGPUMT
{
		public :
			friend class GridPopUpMenuTransp;

			virtual ~CaseGPUMT();
			virtual void CaseGPUMTOnActif();
			virtual void CaseGPUMTOffActif();
			virtual void CaseGPUMTOnSelected();

			CaseGPUMT(GridPopUpMenuTransp &,const std::string & aName,Pt2di anInd,Fonc_Num anInit);

			const std::string & Name() const;

		protected:
			void SetFonc(Fonc_Num aFonc,Im2D_U_INT1);
			void SetMotif(Fonc_Num aFonc);
			void SetTransp(Fonc_Num aFonc);
		private:
            GridPopUpMenuTransp &   mGPUMT;
			std::string             mName;
			Pt2di                   mInd;
};

class BoolCaseGPUMT : public CaseGPUMT
{
	public :
			BoolCaseGPUMT
			(
			     GridPopUpMenuTransp &,
				 const std::string & aName,
				 Pt2di anInd,
				 Fonc_Num aFTrue,
				 Fonc_Num aFFalse,
				 bool     aVal
			);
			void CaseGPUMTOnSelected();
			virtual void BoolCaseGPUMTOnChState(bool NewState);
			bool Val() const;

                        void SetVal(bool );
	private :

			bool                         mVal;
			Im2D_U_INT1                  mImTrue;
			Im2D_U_INT1                  mImFalse;
};

class ChoixParmiCaseGPUMT : public CaseGPUMT
{
	public :
			ChoixParmiCaseGPUMT
			(
			     GridPopUpMenuTransp &,
				 const std::string & aName,
				 Pt2di anInd,
				 Fonc_Num aFTrue,
				 Fonc_Num aFFalse,
				 INT  anId,  // souvent manipules par switch (Enum) {}
				 ChoixParmiCaseGPUMT * oneExcluing  // 0 pour la premiere
			);
			void CaseGPUMTOnSelected();

			const ChoixParmiCaseGPUMT * Selected() const;
			INT IdSelected() const;
			virtual void ChoixParmiCaseGPUMTOnChState(bool NewState);


	private :

			INT                          mId;
			ChoixParmiCaseGPUMT **       mSelected;
			Im2D_U_INT1                  mImTrue;
			Im2D_U_INT1                  mImFalse;
};





class BiScroller : public ElImScroller
{
       public :

                 typedef enum
                 {
                     FusLut ,
                     FusRG1B2,
                     eModeMasq
                 } ModeFusion;

                 static  BiScroller  * LutColoredBiscroller
                 (
                     VideoWin_Visu_ElImScr &Visu,
                     const std::string &   NameGray,
                     const std::string &   NameCol,
                     Elise_colour * Cols,
                     INT            NbCol
                 );
                 static  BiScroller  * RG1B2Biscroller
                 (
                     VideoWin_Visu_ElImScr &Visu,
                     const std::string &   NameGray,
                     const std::string &   NameCol
                 );

                 static  BiScroller  * MasqBiscroller
                 (
                     VideoWin_Visu_ElImScr &Visu,
                     const std::string &   NameGray,
                     const std::string &   NameMasq
                 );



                 void SetTranslate(Pt2dr aTr);
                 Pt2dr  CurTr() const;

                ElImScroller & Scr1();
                ElImScroller & Scr2();
                Im2D_U_INT1  Im1();
                Im2D_U_INT1  Im2();


                Im2D_U_INT1 ImMasq();
                Pt2di SzIn() ;

                const ElSTDNS vector<ElImScroller *> & SubScrolls();


                 BiScroller
                 (
                     VideoWin_Visu_ElImScr &Visu,
                     ElImScroller * ScrGray,
                     ElImScroller * ScrCol,
                     ModeFusion     aMode,
                     Elise_colour * Cols,
                     INT            NbCol,
                     const std::string &  aNameMasq
                 );

                void GraySetGamaCorrec(REAL aGama);
                void SetRGBEntry(INT Gray,INT IndCol,const Elise_colour &);

                void SetImAct(bool,bool);
                bool Im1Act() const;
                bool Im2Act() const;
                ElImScroller & TheScrAct();
                ElImScroller & TheFirstScrAct();
                ElImScroller & OtherScrol();   // "Complem" du TheFirstScrAct

                Video_Win  W() {return mVideoVisu.W();}

           protected :
				void LoadXImage(Pt2di p0W,Pt2di p1W,bool quick);
            private :
                   virtual void MakeOneLine
                   (
                            U_INT1 * aDataImR,
                            U_INT1 * aDataImG,
                            U_INT1 * aDataImB,
                            INT      aY,
                            INT      anX0,
                            INT      anX1,
                            U_INT1 * aLineGr,
                            U_INT1 * aLineCol
                   );

                   void LutMakeOneLine
                   (
                            U_INT1 * aDataImR, U_INT1 * aDataImG, U_INT1 * aDataImB,
                            INT anY,INT      anX0, INT      anX1,
                            U_INT1 * aLineGr, U_INT1 * aLineCol
                   );

                   void RG1B2MakeOneLine
                   (
                            U_INT1 * aDataImR, U_INT1 * aDataImG, U_INT1 * aDataImB,
                            INT anY,INT      anX0, INT      anX1,
                            U_INT1 * aLineGr, U_INT1 * aLineCol
                   );

                   void MasqMakeOneLine
                   (
                            U_INT1 * aDataImR, U_INT1 * aDataImG, U_INT1 * aDataImB,
                            INT anY,INT      anX0, INT      anX1,
                            U_INT1 * aLineGr, U_INT1 * aLineCol
                   );


				enum
						{
							eIndR =0,
							eIndG =1,
							eIndB =2
						};


                U_INT1 * RGBEntry(INT Gray,INT IndCol) {return mDataTabRGB[Gray][IndCol];}


                Memory_ElImDest<U_INT1>   mMemGray;
                ElImScroller *            mScrGray;

                Memory_ElImDest<U_INT1>   mMemCol;
                ElImScroller *            mScrCol;
                ModeFusion                mModeFusion;

				VideoWin_Visu_ElImScr &  mVideoVisu;

                Im2D_U_INT1              mImGray;
                U_INT1 **                mDataGray;
                Im2D_U_INT1              mImCol;
                U_INT1 **                mDataCol;

                Pt2di                    mSzCols;
                Im3D_U_INT1              mTabRGB;
                U_INT1 ***               mDataTabRGB;



                Im2D_U_INT1   mImR;
                U_INT1 *      mDataImR;
                Im2D_U_INT1   mImG;
                U_INT1 *      mDataImG;
                Im2D_U_INT1   mImB;
                U_INT1 *      mDataImB;

                Pt2dr         mTr;
                ElSTDNS vector<ElImScroller *> mSubScrolls;
                bool                           mIm1Act;
                bool                           mIm2Act;
                Im2D_U_INT1                    mImMasq;
};

class cClikInterceptor
{
    public :
       virtual bool InterceptClik(Clik) = 0;
};

class  EliseStdImageInteractor
{
      public :

          virtual ~EliseStdImageInteractor();
          EliseStdImageInteractor
          (
                 Video_Win aW,
                 ElImScroller & aScrol,
                 INT            aButonGeom,
                 INT            aButonZoomIn = -1,
                 INT            aButonZoomOut = -1,
                 cClikInterceptor *           = 0
          );

          Clik clik_press();
          Pt2dr W2U (Pt2dr aWPoint) {return mScr.to_user(aWPoint);}
          Pt2dr U2W (Pt2dr aUPoint) {return mScr.to_win(aUPoint);}

          ElSimilitude W2U () {return mScr.to_user();}
          ElSimilitude U2W () {return mScr.to_win();}

          ElSimilitude WW2UU (ElSimilitude aSim) const {return mScr.WW2UU(aSim);}

          REAL  SC () {return mScr.sc();}



          Pt2di P0W () {return Pt2di(0,0);}
          Pt2di P1W () {return mW.sz();}

          Video_Win    WinEch1();
          // Uniquemement si image de Dim1
          Im2D_U_INT1  ImEch1(bool VisuSim = true);

          void SetModeReplicationPixel();
          void SetModeReplicationPixel(bool aVal);

          std::vector<Pt2dr> GetPolyg(INT aCoulSom,INT aCoulEdge,
                                      const std::vector<Pt2dr> &);
          std::vector<Pt2dr> GetPolyg(INT aCoulSom,INT aCoulEdge);
          std::vector<Pt2dr> GetPolyg(INT aCoulSom,INT aCoulEdge,const Pt2dr &);
          void Refresh();

      protected :

          bool ScaleMode() const;


          virtual void OnEndTranslate(Clik);
          virtual void OnEndScale(Clik);
          void ShowVect();


          void GQP(Clik ,bool); // GUR_query_pointer
          void GBR(Clik );      // GUR_button_released

          class Graber :  public  Grab_Untill_Realeased
          {
                public :
                   Graber(EliseStdImageInteractor &) ;

                private :
                    void  GUR_query_pointer(Clik ,bool);
                    void  GUR_button_released(Clik);

                    EliseStdImageInteractor & mESII;
          };


          Video_Win          mW;
          ElImScroller &     mScr;
          INT                mButtonGeom;
          INT                mButtonZoomIn;
          INT                mButtonZoomOut;
          bool               mScaleMode;
          Pt2dr              mP0Grab;
          Pt2dr              mLastPGrab;
          REAL               mScale0Grab;
          REAL               mSpeedTranslate;  // 1.0
          REAL               mYDoubleScale;    // 100.0
          REAL               mScaleMax;        // 10.0
          REAL               mScaleMin;        // 0.02
          bool               mRasterQuickZoom; // true
          Graber             *mGeoGraber;
          bool               mModeReplicationPixel;

          std::vector<Pt2dr> mPolygCur;
          INT                mCoulSomPolygCur;
          INT                mCoulEdgePolygCur;
          cClikInterceptor*  mClikIntercept;


          friend class Graber;
};





#endif // _ELISE_GENERAL_COMPR_IM_H


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
