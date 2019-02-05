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


// Histoire de tagger ts les cast qui degradent le contenu des images
// float ou double
#define CAST2INT(val) ((INT) (val))


/****************************************************************/
/*                                                              */
/*         TilesIMFL                                            */
/*                                                              */
/****************************************************************/

template <class Type > class TilesIMFL
{
	friend class ImFileLoader<Type>;
	friend class LoadedTilesIMFL<Type>;
	friend class LoadedTilesIMFLAllocator<Type>;

	public :
		TilesIMFL (Pt2di Sz,INT NbChan);
		TilesIMFL (Tiff_Im,Pt2di tile,INT NbChan);
		~TilesIMFL ();
	private :

	
		tFileOffset 				_offs;
		Pt2di 				_sz;
		Pt2di 				_itile;
		tFileOffset  			    _szbyte;
		INT 				_nb_chan;
		LoadedTilesIMFL<Type>  *  _loaded;
		
};


template <class Type > TilesIMFL<Type>::TilesIMFL(Tiff_Im tiff,Pt2di tile,INT NbChan) :
	_offs   (tiff.offset_tile(tile.x,tile.y,0)    ),
	_sz	    (tiff.sz_tile()),
	_itile  (tile),
	_szbyte (tiff.byte_count_tile(tile.x,tile.y,0)),
	_nb_chan (NbChan),
	_loaded (0)
{
}



/****************************************************************/
/*                                                              */
/*     LoadedTilesIMFL                                          */
/*                                                              */
/****************************************************************/

template <class Type >  class LoadedTilesIMFL
{
	friend class ImFileLoader<Type>;
	friend class LoadedTilesIMFLAllocator<Type>;

	public :
		LoadedTilesIMFL(Pt2di sz,INT nb_chan);

		bool fit(const TilesIMFL<Type> & til)
		{
			return (_sz==til._sz) && (_nb_chan == til._nb_chan);
		}

		Pt2di       sz()   {return _sz;}
	    Type **   data() {return _data;}

               void ReinitLL() {_last_load=0;}


    private :

        typedef  ElTyName El_CTypeTraits<Type>::tBase   tBase;
	
		Im2D<Type,tBase>  _im;
		Type *          _data_lin;
		Type **         _data;
		Pt2di	        _sz;
		INT		        _nb_chan;
		TilesIMFL<Type>  *    _last_load;
		
};


template <class Type> 
TilesIMFL<Type>::~TilesIMFL()
{
    if (_loaded)
       DELETE_ONE(_loaded);
}

template <class Type> 
TilesIMFL<Type>::TilesIMFL(Pt2di Sz,INT NbChan) :
	_offs   (-1),
	_sz	    (Sz),
	_itile  (0,0),
	_szbyte (-1),
	_nb_chan (NbChan),
	_loaded (CLASS_NEW_ONE(LoadedTilesIMFL<Type>,(Sz,NbChan)))
{
}


template <class Type>
LoadedTilesIMFL<Type>::LoadedTilesIMFL(Pt2di sz,INT nb_chan) :
	_im			(sz.x*nb_chan,sz.y,(tBase)0),
	_data_lin	(_im.data_lin()),
	_data   	(_im.data()),
	_sz			(sz),
	_nb_chan	(nb_chan),
	_last_load	(0)
{
}

template <class Type> class LoadedTilesIMFLAllocator
{
	public :
		void  get_if_loaded(TilesIMFL<Type> * );
		void  load_it(TilesIMFL<Type> * ,ELISE_fp &);
		void  put(LoadedTilesIMFL<Type> * );

		~LoadedTilesIMFLAllocator();
	private :
		ElSTDNS list<LoadedTilesIMFL<Type> *> _reserve;
};

template <class Type>  LoadedTilesIMFLAllocator<Type>::~LoadedTilesIMFLAllocator()
{

	for 
    (
        ElTyName std::list<LoadedTilesIMFL<Type> *>::iterator itl    = _reserve.begin();
        itl   != _reserve.end()  ;
        itl++
    )
		DELETE_ONE(*itl);
}

template <class Type>  void LoadedTilesIMFLAllocator<Type>:: get_if_loaded(TilesIMFL<Type> * til)
{
	for 
    (
        ElTyName ElSTDNS list<LoadedTilesIMFL<Type> *>::iterator itl    = _reserve.begin();
        itl   != _reserve.end()  ;
        itl++
    )
    {
		if ((*itl)->_last_load == til)
		{
			til->_loaded = *itl;
			_reserve.remove(*itl);
			return;
		}
     }
}


template <class Type> void  LoadedTilesIMFLAllocator<Type>::load_it(TilesIMFL<Type> * til,ELISE_fp & fp)
{
    LoadedTilesIMFL<Type> * lt = 0;

    for 
    (
        ElTyName ElSTDNS list<LoadedTilesIMFL<Type> *>::iterator itl    = _reserve.begin()			;
        (itl   != _reserve.end()) && (lt==0)  ;
        itl++
    )
    {
		if ((*itl)->fit(*til))
		{
			_reserve.remove(*itl);
		    lt = * itl;
		}
     }

	if (! lt)
	{
		lt = CLASS_NEW_ONE(LoadedTilesIMFL<Type>,(til->_sz,til->_nb_chan));
	}

	lt->_last_load  = til;
	til->_loaded    = lt;

 	fp.seek_begin(til->_offs);
	fp.read(lt->_data_lin,1,til->_szbyte);
}

template <class Type> void LoadedTilesIMFLAllocator<Type>::put(LoadedTilesIMFL<Type> * lt)
{
	_reserve.push_front(lt);
}


/****************************************************************/
/*                                                              */
/*     ElTabIntervPUP                                           */
/*                                                              */
/****************************************************************/

INT ElIntervPixelUPond::FirstPixel()  const
{
   return mBegin->mPixU;
}
INT ElIntervPixelUPond::LastPixel() const
{
   return mEnd[-1].mPixU+1;
}


ElTabIntervPUP::~ElTabIntervPUP()
{
  DELETE_VECTOR(mRes,0);
  DELETE_VECTOR(mIntervs,mIW0);
}

void ElTabIntervPUP::IntervalW2U(INT & u0,INT & u1,INT w0,INT w1)
{
   u0 = ElMax(0,mIntervs[w0].FirstPixel());
   u1 = ElMin(mSzU,mIntervs[w1-1].LastPixel());
}



ElTabIntervPUP::ElTabIntervPUP
(
     INT SzW,
     INT SzU,
     INT rab,
     ElTranlatScaleTransfo & aTransfo
)  :
   mSzW       (SzW),
   mIW0       (-rab),
   mIW1       (rab+mSzW),
   mNbW       (mIW1-mIW0),
   mSzU       (SzU),
   mIndTopRes (-1234678),
   mNbRes     (12*mNbW+SzU+2*rab),
   mRes       (NEW_VECTEUR(0,mNbRes,ElPixelUPond)),
   mIntervs   (NEW_VECTEUR(mIW0,mIW1,ElIntervPixelUPond)),
   mTransfo   (aTransfo)
{
}


INT ElTabIntervPUP::FitInUser(INT aUPix)
{
    return ElMax(0,ElMin(mSzU-1,aUPix));
}


void ElTabIntervPUP::PushInterval(INT PixU,REAL Pds)
{
    ELISE_ASSERT(mIndTopRes<mNbRes,"Stack Full in ElTabIntervPUP::PushInterval");

    mRes[mIndTopRes].mPixU = PixU;
    mRes[mIndTopRes].mPds = ElMin(round_ni((mPdsIntResiduel*Pds)/mPdsRealResiduel),mPdsIntResiduel);

    mPdsIntResiduel -= mRes[mIndTopRes].mPds;
    mPdsRealResiduel  = ElMax(mPdsRealResiduel-Pds,1e-10);
    mIndTopRes++;
}

void ElTabIntervPUP::Actualise_Lin(INT PixW) 
{
   mPdsRealResiduel = 1.0;
   REAL  PixU =  mTransfo.W2U(PixW);
   INT  PixInf = ElPixelCentre::PixelInf(PixU);
   
   PushInterval(PixInf,ElPixelCentre::PdsBarryPixelInf(PixU));
   PushInterval(PixInf+1,mPdsRealResiduel);

}


REAL sincpix(REAL x)
{
    x *= PI;
    if (ElAbs(x)<1e-3) return 1.0;
    return sin(x)/x;
}

void ElTabIntervPUP::Actualise_Sinc(INT PixW,INT NbSinc) 
{
   
   mPdsRealResiduel = 0.0;
   REAL  PixU =  mTransfo.W2U(PixW);

   INT  PixInf = ElPixelCentre::PixelInf(PixU)-NbSinc;
   INT  PixSup = PixInf+1+2*NbSinc;

   for (INT pix = PixInf; pix <=PixSup; pix++)
       mPdsRealResiduel += sincpix(PixU-pix);
    
   {
   for (INT pix = PixInf; pix <=PixSup; pix++)
         PushInterval(pix,sincpix(PixU-pix));
   }
}






void ElTabIntervPUP::Actualise_Reduc(INT PixW)
{
   REAL DebU = mTransfo.W2U(ElPixelCentre::DebIntervalPixel(PixW));
   REAL FinU = mTransfo.W2U(ElPixelCentre::FinIntervalPixel(PixW));
   mPdsRealResiduel = (FinU-DebU);

   INT PixUDeb = ElPixelCentre::PixelIncluant(DebU);
   INT PixUFin = ElPixelCentre::PixelIncluant(FinU);


   if (PixUDeb >=0 && PixUFin <mSzU)
   {
      if (PixUDeb == PixUFin)
         PushInterval(PixUDeb,FinU-DebU);
      else
      {
         PushInterval(PixUDeb,ElPixelCentre::LenghFromFinIntervPixelIncluant(DebU));
         for (INT PixU = PixUDeb+1; PixU<PixUFin; PixU++)
         {
             PushInterval(PixU,1.0);
         }
         PushInterval(PixUFin,mPdsRealResiduel);
      }

   }
}

void ElTabIntervPUP::Actualise(INT PdsPerPixW,ModeAct mode)
{
    mIndTopRes = 0;


    for (INT PixW = mIW0; PixW < mIW1; PixW++)
    {
        mIntervs[PixW].mBegin = mRes+mIndTopRes;

        mPdsIntResiduel  = PdsPerPixW;

        if (mode == Reduc)
           Actualise_Reduc(PixW);
        else
        {
           // Actualise_Sinc(PixW,5);
           Actualise_Lin(PixW);
        }
        mIntervs[PixW].mEnd = mRes+mIndTopRes;

    }
    for (INT k=0 ; k<mIndTopRes ; k++)
        mRes[k].mPixU = FitInUser(mRes[k].mPixU);
}

const ElIntervPixelUPond & ElTabIntervPUP::IntervPixelPond(INT wPix)
{
   return mIntervs[wPix];
}


/****************************************************************/
/*                                                              */
/*     IFL_LineWindow                                           */
/*                                                              */
/****************************************************************/

void  IFL_LineWindow::reinit()
{
    mLastX0 = 10000;
    mLastX1 =-10000;

    mLastY  =-100000000;
}

IFL_LineWindow::IFL_LineWindow(INT SzW,INT NbChan) :
    mNbChan (NbChan),
    mSzW    (SzW),
    mLine   (NEW_MATRICE(Pt2di(0,0),Pt2di(SzW,NbChan),INT))
{
   reinit();
}

IFL_LineWindow::~IFL_LineWindow()
{
   DELETE_MATRICE(mLine,Pt2di(0,0),Pt2di(mSzW,mNbChan));
}


/****************************************************************/
/*                                                              */
/*     ImFileLoader                                             */
/*                                                              */
/****************************************************************/




template <class Type> void ImFileLoader<Type>::init_LW()
{
/*
   for (INT k=0 ; k<12 ; k++)
      mBufLW.push_back(new IFL_LineWindow(_SzW.x,_nb_chan));
*/
   for (INT k=0 ; k<12 ; k++)
      mBufLW.push_back(new IFL_LineWindow(this->_SzW.x,_nb_chan));
}

template <class Type> INT ** ImFileLoader<Type>::GetLineWindow(INT x0U,INT x1U,INT yU)
{
   for (INT k=0 ; k<(INT)mBufLW.size() ; k++)
   {
       IFL_LineWindow * LW = mBufLW[k];
       if ((LW->mLastX0<=x0U) && (LW->mLastX1>=x1U) && (LW->mLastY==yU))
       {
          return LW->mLine;
       }
   }

   IFL_LineWindow * Res =  mBufLW[0];

   {
   for (INT k=1 ; k<(INT)mBufLW.size() ; k++)
       if (mBufLW[k]->mLastY < Res->mLastY)
          Res = mBufLW[k];
   }


   Res->mLastX0 = x0U;
   Res->mLastX1 = x1U;
   Res->mLastY  = yU;

   Type * ul = get_line_user(x0U,x1U,yU);

   for (INT c=0; c<_nb_chan ; c++)
   {
       INT * lc = Res->mLine[c];
       for (INT aWxCur = this->_pW0.x ; aWxCur< this->_pW1.x ; aWxCur++)
       {
           const ElIntervPixelUPond &  anXInterv = mXTabIntervales.IntervPixelPond(aWxCur);
           lc[aWxCur] = 0;


           for 
           (
                ElPixelUPond * anXUPixP = anXInterv.mBegin ; 
                anXUPixP!=anXInterv.mEnd ; 
                anXUPixP++
           )
           {
               lc[aWxCur] += CAST2INT(ul[anXUPixP->mPixU*_nb_chan+c] *   anXUPixP->mPds) ;
           }
       }
   }

   return  Res->mLine;
}

#define RAB_XD 1


template <class Type> ImFileLoader<Type>::ImFileLoader
(
	ImFileLoader  & Big,
	INT             zoom
)  :
	GenScaleIm<tGSI> ((Big._SzU+Pt2di(zoom-1,zoom-1))/zoom,Big.SzW(),Big._nb_chan),
    mByteOrdered    (true),
    _tiff           (0),
	_nb_chan  		(Big._nb_chan),
    mFPGlob             (0),
    _nb_tile        (1,1),
    _sz_tile        (this->_SzU),
    _uline          (NEW_VECTEUR(-this->RAB*Big._nb_chan,(this->_SzU.x+this->RAB)*Big._nb_chan,Type)),
    _dynamic        (false),
    mXTabIntervales (Big.SzW().x,Big._SzU.x,this->RAB,this->XTransfo),
    mYTabIntervales (Big.SzW().y,Big._SzU.y,this->RAB,this->YTransfo)
{
    init_LW();
    _own_alloc =  true;
    _alloc = NEW_ONE(LoadedTilesIMFLAllocator<Type>);


	_tiles.push_back(ElSTDNS vector<TilesIMFL<Type> *>());
	_tiles[0].push_back(CLASS_NEW_ONE(TilesIMFL<Type>,(this->_SzU,_nb_chan)));


	Type ** dglob = _tiles[0][0]->_loaded->data();
	INT Z2 = ElSquare(zoom);


    for (INT yt=0; yt<Big._nb_tile.y ; yt++)
	{
            INT Y0tile =  yt *Big._sz_tile.y;
            for (INT xt=0; xt<Big._nb_tile.x ; xt++)
            {
                INT X0tile =  xt *Big._sz_tile.x;
                Big.put_tiles_in_alloc();
                Big.load_this_tile(xt,yt);
                INT TxTile     = ElMin(Big._tiles[yt][xt]->_loaded->sz().x,Big.SzU().x-X0tile);
                INT TyTile     = ElMin(Big._tiles[yt][xt]->_loaded->sz().y,Big.SzU().y-Y0tile);

                Type ** dloc = Big._tiles[yt][xt]->_loaded->data();
                for 
                (
                    INT yloc =0,yglob =Y0tile ;
                    yloc <TyTile; 
                    yloc++,yglob++
                )
                {
                        for (INT c=0; c<_nb_chan ; c++)
                        {
                              Type * lloc = dloc[yloc]+c;
                              Type * lglob = dglob[yglob/zoom]+c;
                              for 
                              (
                                 INT nb=0,xloc =0,xglob = X0tile; 
                                 nb <TxTile; 
                                 nb++,xloc+=_nb_chan,xglob++
                              )
                              {
                                  lglob[(xglob/zoom)*_nb_chan] += lloc[xloc]/Z2;
                              }
                        }
                 }
            }
	}
	Big.put_tiles_in_alloc();
}




template <class Type> ImFileLoader<Type>::~ImFileLoader()
{
        for (INT y=0; y<_nb_tile.y ; y++)
            for (INT x=0; x<_nb_tile.x ; x++)
                DELETE_ONE (_tiles[y][x]);
	if (_own_alloc)
		DELETE_ONE(_alloc);
	DELETE_VECTOR(_uline,-this->RAB*_nb_chan);
    if (mFPGlob)
		delete mFPGlob;
    if (_tiff)
		delete _tiff;

    for (INT k=0; k<(INT)mBufLW.size() ; k++)
        delete mBufLW[k];
}

template <class Type> ELISE_fp * ImFileLoader<Type>::FileOfTile(Pt2di aTile)
{
   if (!mHasTileFile) return  mFPGlob;

    Pt2di aNumTF (aTile.x/mNbTTByF.x,aTile.y/mNbTTByF.y);
   
    if (aNumTF==mCurTileOfFT) return mFPOfFileTile;

    delete mFPOfFileTile;
    mCurTileOfFT = aNumTF;
    mFPOfFileTile = new ELISE_fp(_tiff->NameTileFile(mCurTileOfFT).c_str(),ELISE_fp::READ);

    return mFPOfFileTile;
}




template <class Type> ImFileLoader<Type>::ImFileLoader
(
	Tiff_Im tiff,
	Pt2di SzW,
	ImFileLoader * IMFalloc
) :
    GenScaleIm<tGSI>(tiff.sz(),SzW,tiff.NbChannel()),
    mByteOrdered (tiff.byte_ordered()),
	_tiff     (new Tiff_Im (tiff)),
	_nb_chan  (tiff.NbChannel()),
	mFPGlob	  (new ELISE_fp(tiff.name(),ELISE_fp::READ)),
	_nb_tile  (tiff.nb_tile()),
	_sz_tile  (tiff.sz_tile()),
	_uline    (NEW_VECTEUR(-this->RAB*_nb_chan,(this->_SzU.x+this->RAB)*_nb_chan,Type)),
   _dynamic    (true),
    mXTabIntervales (SzW.x,this->_SzU.x,this->RAB,this->XTransfo),
    mYTabIntervales (SzW.y,this->_SzU.y,this->RAB,this->YTransfo),
    mSzTileFile     (tiff.SzFileTile()),
    mHasTileFile    (mSzTileFile.x > 0),
    mFPOfFileTile   (0),
    mCurTileOfFT    (-1,-1),
    mNbTTByF        (tiff.NbTTByTF())
{
    init_LW();
    _own_alloc =  (IMFalloc == 0);
	if (IMFalloc)
		_alloc = IMFalloc->_alloc;
	else
		_alloc = NEW_ONE(LoadedTilesIMFLAllocator<Type>);
       
	
  	Tjs_El_User.ElAssert
    (
        tiff.mode_compr() == Tiff_Im::No_Compr,
        EEM0 << "ImFileLoader, handle UnCompressed Images "
            << "Tiff File = " << tiff.name()
    );             
  	Tjs_El_User.ElAssert
    (
        tiff.NbBits()  == (8*sizeof(Type)),
        EEM0 << "ImFileLoader, handle 8 bits images"
            << "Tiff File = " << tiff.name()
    );             
  	Tjs_El_User.ElAssert
    (
        tiff.plan_conf()  == Tiff_Im::Chunky_conf,
        EEM0 << "ImFileLoader, handle Chunky_conf as plan_conf"
            << "Tiff File = " << tiff.name()
    );             


    for (INT y=0; y<_nb_tile.y ; y++)
    {
	   _tiles.push_back(ElSTDNS vector<TilesIMFL<Type> *>());
	   for (INT x=0; x<_nb_tile.x ; x++)
	   {
//  mSzFileTile
//  mUseFileTile

// std::cout << "HHHHHH " << _nb_tile << " " << tiff.name() << " " << tiff.sz_tile()  << "\n";
// std::cout << "GGGGGG " << tiff.SzFileTile()  << " " << tiff.NbTTByTF()   << "\n";
                 
                 Tiff_Im aTiff2Load = tiff;
                 Pt2di    aTileInside(x,y);
                 if (mHasTileFile)
                 {
                      aTileInside = Pt2di(x%mNbTTByF.x,y%mNbTTByF.y);
                      Pt2di aNumTTF(x/mNbTTByF.x,y/mNbTTByF.y);
                      std::string aNameTTF = tiff.NameTileFile(aNumTTF);
                      aTiff2Load = Tiff_Im(aNameTTF.c_str());
                 }



		 _tiles[y].push_back
                 (
                     CLASS_NEW_ONE(TilesIMFL<Type>,(aTiff2Load,aTileInside,_nb_chan))
                     // CLASS_NEW_ONE(TilesIMFL<Type>,(tiff,Pt2di(x,y),_nb_chan))
                 );
	}
    }
}

template <class Type> void ImFileLoader<Type>::ImFReInitTifFile(Tiff_Im aTif)
{
    if ( mHasTileFile)
    {
         
         ELISE_ASSERT(false,"::ImFReInitTifFile with Tile File");
    }
    put_tiles_in_alloc(true);
    delete _tiff;
    delete mFPGlob;
    _tiff = new Tiff_Im (aTif);
    mFPGlob	  = new ELISE_fp(aTif.name(),ELISE_fp::READ);
}




template <class Type> void ImFileLoader<Type>::put_tiles_in_alloc(bool FullReinit)
{
     if (_dynamic)
     {
    	for (INT y=0; y<_nb_tile.y ; y++)
        {
        	for (INT x=0; x<_nb_tile.x ; x++)
         	{
            	     if (_tiles[y][x]->_loaded)
             	     {
                              if(FullReinit) 
                              {
                                  _tiles[y][x]->_loaded->ReinitLL();
                              }
                	     _alloc->put(_tiles[y][x]->_loaded);
		             _tiles[y][x]->_loaded =0;	
             	     }
         	}
	 }
      }
}

template <class Type> void ImFileScroller<Type>::no_use()
{
     if (this->_dynamic)
         this->put_tiles_in_alloc();
}

template <class Type> void ImFileLoader<Type>::load_this_tile(INT x,INT y)
{
	if ( ! _tiles[y][x]->_loaded)
        {
	   _alloc->load_it(_tiles[y][x],*FileOfTile(Pt2di(x,y)));
        }
}

template <class Type> bool ImFileLoader<Type>::load_all(Pt2dr tr,REAL sc,Pt2di p0,Pt2di p1)
{



	_tiles_0 = Pt2di(this->_xU0/_sz_tile.x,this->_yU0/_sz_tile.y);
	_tiles_1 = Pt2di((this->_xU1-1)/_sz_tile.x+1+2*RAB_XD,(this->_yU1-1)/_sz_tile.y+1);

	_tiles_0.SetSup(Pt2di(0,0));
	_tiles_1.SetInf(_nb_tile);


     if (_dynamic)
     {
			// On unload toute les dalles inutiles
			for (INT y=0; y<_nb_tile.y ; y++)
				for (INT x=0; x<_nb_tile.x ; x++)
				{
					if ( 
							(_tiles[y][x]->_loaded)
						&& (! Pt2di(x,y).in_box(_tiles_0,_tiles_1))
						)
						{
							_alloc->put(_tiles[y][x]->_loaded);
				    		_tiles[y][x]->_loaded =0;	
						}
				}

			// On partcourt les dalles utiles, non loadees, en ne cherchant que
    		// celle qui son deja loadees
	

			{
			for (INT y=_tiles_0.y; y<_tiles_1.y ; y++)
				for (INT x=_tiles_0.x; x<_tiles_1.x ; x++)
				{
					if ( ! _tiles[y][x]->_loaded)
						_alloc->get_if_loaded(_tiles[y][x]);
				}
			}
// std::cout << "Ppppppppppppp\n";

			// Cette fois, on force le load

			{
			for (INT y=_tiles_0.y; y<_tiles_1.y ; y++)
				for (INT x=_tiles_0.x; x<_tiles_1.x ; x++)
					load_this_tile(x,y);
			}
// std::cout << "GGgggggggggg\n";
    }

	return true;
}


template <class Type> Type * ImFileLoader<Type>::get_line_user(INT x0,INT x1,INT y)
{

   if (mUpToDate_LastXY_GLU)
   {
        if (    (x0>= mLastX0_GLU) 
             && (x1<=mLastX1_GLU)
             && (y == mLastY_GLU)
           )
        {
           return _uline;
        }
   }

   mUpToDate_LastXY_GLU = true;
   mLastX0_GLU = x0;
   mLastX1_GLU = x1;
   mLastY_GLU = y;


	INT itx0 =  x0/_sz_tile.x;
	INT itx1 =  (x1-1)/_sz_tile.x+1;


	INT ity =  y/_sz_tile.y;
	INT yl = y -ity*_sz_tile.y;

	for (INT itx=itx0 ; itx<itx1 ; itx++)
	{
		INT X0_tile = itx*_sz_tile.x;

		INT x0l = ElMax(x0- X0_tile,0);
		INT x1l = ElMin(x1- X0_tile,_sz_tile.x);

		convert
		(
			_uline+(X0_tile+x0l)*_nb_chan,
			_tiles[ity][itx]->_loaded->_data[yl]+x0l*_nb_chan,
                        (x1l-x0l)*_nb_chan
		);
	}

    if (! mByteOrdered)
		byte_inv_tab(_uline+x0,sizeof(Type),x1-x0);

return _uline;
}


template <class Type>  Tiff_Im ImFileLoader<Type>::Tiff()
{
   return *_tiff;
}

template <class Type>  void ImFileLoader<Type>::MakeOneLineZooPPV()
{
    Type * ul = get_line_user(_ux0cur,_ux1cur,_uy0cur);
    for (INT c=0; c<_nb_chan ; c++)
    {
       tGSI * lc = this->_line[c];
       for (INT wx = this->_pW0.x ; wx< this->_pW1.x ; wx++)
       {
          lc[wx] = (ul[this->_Cw2uX[wx]*_nb_chan+c]);
          // lc[wx] = CAST2INT(ul[this->_Cw2uX[wx]*_nb_chan+c]);
       }
    }
}

template <class Type>  void ImFileLoader<Type>::MakeOneLineReduceSomPPV()
{


    for (INT c=0; c<_nb_chan ; c++)
    {
       tGSI * lc = this->_line[c];
       for (INT wx = this->_pW0.x ; wx< this->_pW1.x ; wx++)
          lc[wx] = 0;
    }

    for (INT yu=_uy0cur; yu<_uy1cur ; yu++)
    {
         Type * ul = get_line_user(_ux0cur,_ux1cur,yu);
         for (INT c=0; c<_nb_chan ; c++)
         {
              tGSI * lc = this->_line[c];
              for (INT wx = this->_pW0.x ; wx< this->_pW1.x ; wx++)
              {
                 INT UX0 = this->_Cw2uX[wx];
                 INT UX1 = this->_Cw2uX[wx+1];

                 for (INT UX = UX0 ; UX < UX1 ; UX++)
                 {
                     // lc[wx] += CAST2INT(ul[UX*_nb_chan+c]);
                     lc[wx] += (ul[UX*_nb_chan+c]);
                 }
              }
         }
    }

    INT NbY = ElMax(1,_uy1cur-_uy0cur);
	{
    for (INT c=0; c<_nb_chan ; c++)
    {
        tGSI * lc = this->_line[c];
        for (INT wx = this->_pW0.x ; wx< this->_pW1.x ; wx++)
        {
            lc[wx] /= ElMax(1,(this->_Cw2uX[wx+1]-this->_Cw2uX[wx])) * NbY;
        }
    }
	}
}


template <class Type> void ImFileLoader<Type>::MakeOneLinePixelPond()
{


    for (INT c=0; c<_nb_chan ; c++)
    {
       tGSI * lc = this->_line[c];
       for (INT wx = this->_pW0.x ; wx< this->_pW1.x ; wx++)
          lc[wx] = 0;
    }

    const ElIntervPixelUPond &  anYInterv = mYTabIntervales.IntervPixelPond(mWyCur0);
    for (ElPixelUPond * anYUPixP = anYInterv.mBegin ; anYUPixP!=anYInterv.mEnd ; anYUPixP++)
    {
         INT anYPds = anYUPixP->mPds ;
         INT ** lW = GetLineWindow(_ux0cur,_ux1cur,anYUPixP->mPixU);


         for (INT c=0; c<_nb_chan ; c++)
         {
              tGSI * lc = this->_line[c];
              INT * lWc = lW[c];
              for (INT aWxCur = this->_pW0.x ; aWxCur< this->_pW1.x ; aWxCur++)
              {
                  lc[aWxCur] += lWc[aWxCur] * anYPds;
              }
         }
    }

    INT P2 = ElSquare(this->_CoeffPds);
	{
    for (INT c=0; c<_nb_chan ; c++)
    {
        tGSI * lc = this->_line[c];
        for (INT aWxCur = this->_pW0.x ; aWxCur< this->_pW1.x ; aWxCur++)
        {
            lc[aWxCur] /= P2;
        }
    }
	}
}



template <class Type>  void ImFileLoader<Type>::MakeOneLine(bool quick)
{
    if (mPixReplic)
    {
       MakeOneLineZooPPV();
    }
    else
    {
       MakeOneLinePixelPond();
    }
}




template <class Type> void ImFileLoader<Type>::do_it(Pt2dr tr,REAL sc,Pt2di p0,Pt2di p1,bool quick)
{

   mPixReplic = quick;
   mReduc = (sc<=1);



   if (! this->do_it_gen(tr,sc,p0,p1))
      return ;

   mUpToDate_LastXY_GLU = false;

   for (INT k=0 ; k<(INT)mBufLW.size() ; k++)
       mBufLW[k]->reinit();


    _ux0cur = this->_Cw2uX[this->_pW0.x];
    _ux1cur = ElMin(this->_Cw2uX[this->_pW1.x]+RAB_XD,this->_SzU.x-1);     



    ElTabIntervPUP::ModeAct Mode;
    if (mReduc)
        Mode = ElTabIntervPUP::Reduc;
    else
        Mode = ElTabIntervPUP::Lin ; 

    this->_CoeffPds = 100;
    mXTabIntervales.Actualise(this->_CoeffPds,Mode);
    mYTabIntervales.Actualise(this->_CoeffPds,Mode);


    {
       INT U0,U1;
       mXTabIntervales.IntervalW2U(U0,U1,this->_pW0.x,this->_pW1.x);
       U0 -= RAB_XD;
       ElSetMax(U0,0);
       ElSetMin(_ux0cur ,U0);
       ElSetMax(_ux1cur ,U1);
       ElSetMin(this->_xU0,U0);
       ElSetMax(this->_xU1,U1);
    }
    {
       INT U0,U1;
       mYTabIntervales.IntervalW2U(U0,U1,this->_pW0.y,this->_pW1.y);
       ElSetMin(this->_yU0,U0);
       ElSetMax(this->_yU1,U1);
    }


    if (! load_all(tr,sc,p0,p1))
       return;

    for (mWyCur0=this->_pW0.y;  mWyCur0<this->_pW1.y; )
    {
        _uy0cur = this->_Cw2uY[ mWyCur0];

        mWyCur1 =  mWyCur0+1;
 
        if (mPixReplic)
        {
            while ( (mWyCur1<this->_pW1.y) && (_uy0cur==this->_Cw2uY[mWyCur1]))
                mWyCur1++;
        }
        _uy1cur = this->_Cw2uY[ mWyCur1];
 

        MakeOneLine(quick);



        RasterUseLine
        (
            Pt2di(this->_pW0.x, mWyCur0),
            Pt2di(this->_pW1.x, mWyCur1),
            this->_line,
            _nb_chan
        );
	    mWyCur0 =  mWyCur1;
    }                                            
}



/****************************************************************/
/*                                                              */
/*     ImFileScroller                                           */
/*                                                              */
/****************************************************************/


template <class Type> bool ImFileScroller<Type>::CanReinitTif()
{
    return true;
}

template <class Type> void ImFileScroller<Type>::ReInitTifFile(Tiff_Im aTif)
{
   this->ImFReInitTifFile(aTif);
}


template <class Type> Fonc_Num ImFileScroller<Type>::in()
{
   return this->Tiff().in(0);
}

template <class Type> Pt2di ImFileScroller<Type>::SzIn()
{
   return this->Tiff().sz();
}

template <class Type> ImFileScroller<Type>:: ImFileScroller
(
   ImFileScroller & Big,
   INT          zoom
)  :
	ImFileLoader<Type>(Big,zoom),
	ElImScroller(Big.VisuStd(),Big.DimOut(),ImFileLoader<Type>::_SzU,Big.sc_im()/(REAL) zoom)
{
}

template <class Type> ElImScroller * ImFileScroller<Type>::Reduc(INT zoom,bool)
{
    ElImScroller * res =  new ImFileScroller (*this,zoom);
    return res;
}


void WarnIntConversion(Tiff_Im tif)
{
   static bool First = true;
   if (! First)
      return;
   First = false;

   cout << "\n\n";
   cout << "***** Warns float values converted to int in low level reading of ImFileScroller\n";
   cout << "***** File = " << tif.name() << "\n";
   cout << "***** (warn restricted to first occurence)";
   cout << "\n\n";
}

template <class Type> ImFileScroller<Type>::ImFileScroller
(
    Visu_ElImScr &   V,
    Tiff_Im 		tif,
    REAL  		sc_im ,
    ImFileScroller* 	alloc 
)   :  
	ImFileLoader<Type>(tif,V.SzW(),alloc),
	ElImScroller(V,tif.nb_chan(),tif.sz(),sc_im)
{

/*
if (MPD_MM())
{
    std::cout << "ImFileScroller<Type>::ImFil " << tif.name() 
              << " " << type_elToString(tif.type_el())
              << " " << sc_im << " " << this << "\n";
}
*/

/*
   if (! El_CTypeTraits<Type>::IsIntType())
      WarnIntConversion(tif);
*/
}




template <class Type> void ImFileScroller<Type>::LoadXImage(Pt2di p0,Pt2di p1,bool quick)
{
/*
  mm3d Vino VeryBig.tif Bilin=1

   LLL::LoadXImage 1 0.569406 0 0
   LLL::LoadXImage 0 0.569406 0 0

std::cout << "LLL::LoadXImage " << quick << " " << this->sc() << " " << this->AlwaysQuickZoom() << " " << this->AlwaysQuick() << "\n";
*/

	this->do_it(ElImScroller::tr(),ElImScroller::sc(),p0,p1,quick);
}

template <class Type> void ImFileScroller<Type>::RasterUseLine(Pt2di p0,Pt2di p1,tGSI ** l,int aNbChanIn)
{
     for (INT y= p0.y ; y<p1.y ; y++)
     {
 
          write_image (p0.x,Pt2di(p0.x,y),p1.x-p0.x,l,aNbChanIn);        
     }             
}


template class TilesIMFL<U_INT1>;
template class TilesIMFL<U_INT2>;
template class TilesIMFL<INT2>;
template class TilesIMFL<REAL4>;
template class TilesIMFL<REAL8>;

template class LoadedTilesIMFL<U_INT1>;
template class LoadedTilesIMFL<U_INT2>;
template class LoadedTilesIMFL<INT2>;
template class LoadedTilesIMFL<REAL4>;
template class LoadedTilesIMFL<REAL8>;

template class LoadedTilesIMFLAllocator<U_INT1>;
template class LoadedTilesIMFLAllocator<U_INT2>;
template class LoadedTilesIMFLAllocator<INT2>;
template class LoadedTilesIMFLAllocator<REAL4>;
template class LoadedTilesIMFLAllocator<REAL8>;

template class ImFileLoader<U_INT1>;
template class ImFileLoader<U_INT2>;
template class ImFileLoader<INT2>;
template class ImFileLoader<REAL4>;
template class ImFileLoader<REAL8>;

template class ImFileScroller<U_INT1>;
template class ImFileScroller<U_INT2>;
template class ImFileScroller<INT2>;
template class ImFileScroller<REAL4>;
template class ImFileScroller<REAL8>;





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
