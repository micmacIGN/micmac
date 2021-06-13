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

/****************************************************************/
/*                                                              */
/*          Visu_ElImDest                                       */
/*                                                              */
/****************************************************************/  

Visu_ElImDest::~Visu_ElImDest() {}



Visu_ElImDest::Visu_ElImDest(Pt2di aSz,INT aDimOut)  :

      mDimOut        (aDimOut),
      mSzBigIm       (aSz),
      mSzBuf         (aSz.x,Elise_Palette::MaxDimPal),
      mBufI          (mSzBuf.x,mSzBuf.y,0),
      mDataBufI      (mBufI.data()),
      mUseEtalDyn    (false),
      mGama         (1.0),
      mImGamaCorr   (256),
      mDataGamaCorr (mImGamaCorr.data()),
      mUseGamaCorr  (false),
      mIVCD         (0)
{
}

void Visu_ElImDest::SetChgDyn(cImgVisuChgDyn * anIVCD)
{
    mIVCD = anIVCD;
}

INT Visu_ElImDest::VMin() const
{
   return mVMin;
}
INT Visu_ElImDest::VMax() const
{
   return mVMax;
}


void Visu_ElImDest::SetEtalDyn(INT aVmin,INT aVmax,bool SetUse)
{
   mVMin = ElMin(aVmin,aVmax);
   mVMax = ElMax(aVmin,aVmax);
   mDiff = ElMax(1,mVMax-mVMin);
   if (SetUse)
      mUseEtalDyn = true;
}

void Visu_ElImDest::SetUseEtalDyn(bool UseEtalDyn)
{
   mUseEtalDyn = UseEtalDyn;
}

void Visu_ElImDest::SetGamaCorr(REAL aGamaFact)
{
   mGama = aGamaFact;
   ELISE_COPY
   (
        mImGamaCorr.all_pts(),
        255* pow((FX/255.0),1.0/aGamaFact),
        mImGamaCorr.out()
   );
   mUseGamaCorr = true;
}

template <class Type>  void FriendVisu_ElImDest<Type>::write_image(Visu_ElImDest& aVEI,INT  x0src,Pt2di p0dest,INT nb,Type ** data,int** aDataBuf,int aNbChanelIn)
{

     if (aVEI.mIVCD)
     {
         for (INT d=0 ; d<aVEI.mDimOut ; d++)
         {
              aVEI.mIVCD->ChgDyn(aDataBuf[d]+x0src,data[d]+x0src,nb);
         }
     }
     else if (aVEI.mUseEtalDyn)
     {
         for (INT d=0 ; d<aVEI.mDimOut ; d++)
         {
              Type  * dIn = data[ElMin(d,aNbChanelIn-1)]+x0src;
              INT * dOut = aDataBuf[d]+x0src;
              for (INT k=0; k<nb ; k++)
                  dOut[k] = ElMin(255,ElMax(0, ElStdTypeScal<int>::RtoT (((dIn[k]-aVEI.mVMin)*255)/aVEI.mDiff)));
         }
     }
     else if (aVEI.mUseGamaCorr)
     {
         for (INT d=0 ; d<aVEI.mDimOut ; d++)
         {
              Type  * dIn = data[ElMin(d,aNbChanelIn-1)]+x0src;
              INT * dOut = aDataBuf[d]+x0src;
              for (INT k=0; k<nb ; k++)
              {
                  dOut[k] = aVEI.mDataGamaCorr[round_ni(dIn[k])];
              }
         }
     }
     else
     {

         for (INT d=0 ; d<aVEI.mDimOut ; d++)
         {
             // convert(aDataBuf[d]+x0src,data[d]+x0src,nb);
             convert(aDataBuf[d]+x0src,data[ElMin(d,aNbChanelIn-1)]+x0src,nb);
         }
     }
     aVEI.write_image_brute(x0src,p0dest,nb,aDataBuf);
}



void Visu_ElImDest::write_image(INT  x0src,Pt2di p0dest,INT nb,INT ** data,int aNbChanelIn)
{
    FriendVisu_ElImDest<int>::write_image(*this,x0src,p0dest,nb,data,mDataBufI,aNbChanelIn);

/*
Visu_ElImDest& aVEI,INT  x0src,Pt2di p0dest,INT nb,Type ** data,Type** aDataBuf)
     INT ** aDataBuf = mDataBufI;
     INT ** ToW = data;

     if (mIVCD)
     {
         for (INT d=0 ; d<mDimOut ; d++)
         {
              mIVCD->ChgDyn(aDataBuf[d]+x0src,data[d]+x0src,nb);
         }
         ToW = aDataBuf;
     }
     else if (mUseEtalDyn)
     {
         for (INT d=0 ; d<mDimOut ; d++)
         {
              INT * dIn = data[d]+x0src;
              INT * dOut = aDataBuf[d]+x0src;
// cout << dIn << " " << dOut << " " << mVMin << " " << mDiff << "\n";
              for (INT k=0; k<nb ; k++)
                  dOut[k] = ElMin(255,ElMax(0,((dIn[k]-mVMin)*255)/mDiff));
         }
         ToW = aDataBuf;
     }
     else if (mUseGamaCorr)
     {
         for (INT d=0 ; d<mDimOut ; d++)
         {
              INT * dIn = data[d]+x0src;
              INT * dOut = aDataBuf[d]+x0src;
              for (INT k=0; k<nb ; k++)
              {
                  // ELISE_ASSERT(dIn[k]>=0&&dIn[k]<256,"Bad Vals in mUseGamaCorr");
                  dOut[k] = mDataGamaCorr[dIn[k]];
              }
         }
         ToW = aDataBuf;
     }
     write_image_brute(x0src,p0dest,nb,ToW);
*/
}

void Visu_ElImDest::write_image(INT  x0src,Pt2di p0dest,INT nb,double ** data,int aNbChanelIn)
{
    FriendVisu_ElImDest<double>::write_image(*this,x0src,p0dest,nb,data,mDataBufI,aNbChanelIn);
   // ELISE_ASSERT(false,"Visu_ElImDest::write_image double");
}


/****************************************************************/
/*                                                              */
/*          Memory_ElImDest                                     */
/*                                                              */
/****************************************************************/  



template <class Type> Memory_ElImDest<Type>::Memory_ElImDest(INT NbChan,Pt2di SzTot) :
    Visu_ElImDest(SzTot,NbChan),
    mNbChan   (NbChan),
    mSzImTot  (SzTot),
    mDec      (0,0)
{
    for (INT iCh=0; iCh<NbChan ; iCh++)
        mImages.push_back(Im2D<Type,INT4>(mSzImTot.x,mSzImTot.y));
}


template <class Type> void Memory_ElImDest<Type>::VerifDim(INT DimOut)
{
    ELISE_ASSERT(DimOut==mNbChan,"Incompatibles Dims in Memory_ElImDest::VerifDim");
}

template <class Type> void Memory_ElImDest<Type>::SetDec(Pt2di aDec) 
{
    mDec = aDec;
}

template <class Type> std::vector<Im2D<Type,INT4> > &  Memory_ElImDest<Type>::Images()
{
    return mImages;
}


template <class Type> void Memory_ElImDest<Type>::write_image_brute(INT x0src,Pt2di p0dest,INT nb,INT ** data)
{
   p0dest -= mDec;

   AdaptParamCopyTrans(x0src,p0dest.x,nb,1<<30,mSzImTot.x);


   if ((p0dest.y<0) || (p0dest.y>=mSzImTot.y) || (nb<=0))
       return;

   for (INT iCh=0 ; iCh<(INT)mImages.size() ; iCh++)
   {
       convert
       (
           mImages[iCh].data()[p0dest.y]+p0dest.x,
           data[iCh]+x0src,
           nb
       );
   }
}

template class Memory_ElImDest<U_INT1>;
template class Memory_ElImDest<INT4>;


/****************************************************************/
/*                                                              */
/*          Visu_ElImScr                                        */
/*                                                              */
/****************************************************************/  
Visu_ElImScr::~Visu_ElImScr(){}

Visu_ElImScr::Visu_ElImScr(Pt2di aSz,INT aDimOut) :
   Visu_ElImDest(aSz,aDimOut)
{
}


void Visu_ElImScr::AdaptTiffFile(Tiff_Im,bool,bool)
{
}

/****************************************************************/
/*                                                              */
/*     VideoWin_Visu_ElImScr                                    */
/*                                                              */
/****************************************************************/  



void VideoWin_Visu_ElImScr::VerifDim(INT DimOut)
{
    if (DimOut!=_pal.dim_pal())
    {
        std::cout << "DimOut " << DimOut << " Pal " << _pal.dim_pal() << "\n";
        ELISE_ASSERT
        (
             DimOut==_pal.dim_pal(),
            "Incompatibles Dims in VideoWin_Visu_ElImScr::VerifDim"
        );
    }
}


VideoWin_Visu_ElImScr::VideoWin_Visu_ElImScr
(
     Video_Win W,
     Elise_Palette Pal,
     Pt2di SzImIncr,
     INT   aVMin,
     INT   aVMax
) :
       Visu_ElImScr(W.sz(),Pal.dim_pal()),
       mBigIm  (W.StdBigImage()),
       mImPer (StdIm(W)),
       mSzImIncr (SzImIncr),
       mImIncr   (W,SzImIncr),
       _w   (W),
       _pal (Pal)
{
    SetEtalDyn(aVMin,aVMax,false);
}


VideoWin_Visu_ElImScr::VideoWin_Visu_ElImScr(Video_Win W,Tiff_Im aTifFile,Pt2di SzImIncr,bool ForceGray) :
       Visu_ElImScr(W.sz(),aTifFile.nb_chan()),
       mBigIm  (W.StdBigImage()),
       mImPer (StdIm(W)),
       mSzImIncr (SzImIncr),
       mImIncr   (W,SzImIncr),
       _w   (W),
       _pal (     ForceGray            ?
                  W.pgray()            :
                  aTifFile.std_pal(W)
            )
{
   SetUseEtalDyn (aTifFile.type_el() != GenIm::u_int1);
   min_max_type_num(aTifFile.type_el(),mVMin,mVMax);
cout << "Max Min " << mVMin << " " << mVMax  << "\n";
   SetEtalDyn(mVMin,mVMax,false);
}


void VideoWin_Visu_ElImScr::write_image_brute(INT  x0src,Pt2di p0dest,INT nb,INT ** data)
{
     _w.write_image
     (
          Pt2di(x0src,0),
          p0dest,
          Pt2di(nb,1),
          &data,
          _pal
     );
}

void VideoWin_Visu_ElImScr::AdaptTiffFile(Tiff_Im aTif,bool AdaptPal,bool ForceGray)
{
   if (AdaptPal)
   {
       if (ForceGray)
          _pal = _w.pgray();
       else if (aTif.phot_interp() == Tiff_Im::RGBPalette)
          _pal = _w.prgb();
       else
          _pal = aTif.std_pal(_w);
   }
}

void VideoWin_Visu_ElImScr::load_rect_image(Pt2di p0,Pt2di p1,bool Quick)
{
    _w.load_image(p0,p1);
}

void VideoWin_Visu_ElImScr::translate(Pt2di tr)
{
    _w.image_translate(tr);
    _w.translate(tr);
}


void VideoWin_Visu_ElImScr::write_image_out(Pt2di p0_src,Pt2di p0_dest,Pt2di sz)
{
       mImPer.write_image_per(p0_src,p0_dest,sz);
}


extern Im2D_INT4  Ok_eLise();


/*
ElXim  VideoWin_Visu_ElImScr::StdIm(Video_Win w)
{
    Im2D_INT4  E = Ok_eLise();
    Pt2di Sz0 = E.sz();

    Pt2di sz (3*Sz0.x,Sz0.y);

sz = Inf(sz,w.sz());

    Im2D_U_INT1 R(sz.x,sz.y,0);
    Im2D_U_INT1 G(sz.x,sz.y,0);
    Im2D_U_INT1 B(sz.x,sz.y,0);

    ELISE_COPY
    (
       select(R.all_pts(),! E.in()[Virgule(FX%Sz0.x,FY)]),
       // Fonc_Num(255,0,0),
       its_to_rgb(Virgule(128,(FX*255.0)/sz.x,255)),
       Virgule(R.out(),G.out(),B.out())
    );


    ElXim aRes = ElXim(w,sz,Virgule(R.in(),G.in(),B.in()),w.prgb());

    return aRes;
}

GnuLinux 256,205
*/

ElXim  VideoWin_Visu_ElImScr::StdIm(Video_Win w)
{

    Tiff_Im aGnuxTiff =   MMIcone("Gnux");
    Tiff_Im aMikMakTiff = MMIcone("MikMak");

    Pt2di aSzG = aGnuxTiff.sz();
    Pt2di aSzM = aMikMakTiff.sz();
    Pt2di aSz(2*aSzG.x,aSzG.y+aSzM.y);

    Im2D_U_INT1 R(aSz.x,aSz.y,255);
    Im2D_U_INT1 G(aSz.x,aSz.y,255);
    Im2D_U_INT1 B(aSz.x,aSz.y,255);
         
    ELISE_COPY(rectangle(Pt2di(0,0),aSzG),aGnuxTiff.in(),Virgule(R.out(),G.out(),B.out()));

    Pt2di aTr1 = Pt2di(0,aSzG.y);
    ELISE_COPY(rectangle(aTr1,aSzM+aTr1),trans(aMikMakTiff.in(),-aTr1),Virgule(R.out(),G.out(),B.out()));
    Pt2di aTr2 = Pt2di(aSzG.x,0);
    ELISE_COPY(rectangle(aTr2,aSzM+aTr2),trans(aMikMakTiff.in(),-aTr2),Virgule(R.out(),G.out(),B.out()));

    Pt2di aTr3 = Pt2di(aSzG.x,aSzM.y);
    ELISE_COPY(rectangle(aTr3,aSzG+aTr3),trans(aGnuxTiff.in(),-aTr3),Virgule(R.out(),G.out(),B.out()));


    Pt2di aSzW = w.sz(); // Inf(w.sz(),Pt2di(400,400));
    double aRatio = ElMin(double(aSzW.x-2)/aSz.x,double(aSzW.y-2)/aSz.y);
    if (aRatio<1.0)
    {
         aSz = round_ni(Pt2dr(aSz)*aRatio);
         Im2D_U_INT1 aNewR(aSz.x,aSz.y,255);
         Im2D_U_INT1 aNewG(aSz.x,aSz.y,255);
         Im2D_U_INT1 aNewB(aSz.x,aSz.y,255);
         ELISE_COPY
         (
              aNewR.all_pts(),
              Max (0,Min(255,
                     StdFoncChScale
                     (
                              //aDebug ? ((FX/30)%2) && tiff.in_proj() : tiff.in_proj(),
                              Virgule(R.in_proj(),G.in_proj(),B.in_proj()),
                              Pt2dr(0,0),
                              Pt2dr(1/aRatio,1/aRatio),
                              Pt2dr(1.0,1.0)
                     )
                  )
              ),
              Virgule(aNewR.out(),aNewG.out(),aNewB.out())
         );
         R = aNewR;
         G = aNewG;
         B = aNewB;
    }

    ElXim aRes = ElXim(w,aSz,Virgule(R.in(),G.in(),B.in()),w.prgb());


    return aRes;
}



void VideoWin_Visu_ElImScr::InscrustImage
     (
          std::vector<Im2D_INT4> & Ims,
          Pt2di p0Src,
          Pt2di p0Dest,
          Pt2di sz,
          INT NbIm
     )
{
    InscrustImage(Ims,p0Src,p0Dest,sz,NbIm,_pal);
}


void VideoWin_Visu_ElImScr::InscrustImage
     (
            std::vector<Im2D_INT4> & Ims,
            Pt2di p0Src,
            Pt2di p0Dest,
            Pt2di sz,
            INT   NbIm,
            Elise_Palette pal
     )
{
   ELISE_ASSERT(sz.xety_inf_ou_egal(mSzImIncr),"Insufficient size of XIm in InscrustImage");

   mImIncr.fill_with_el_image(p0Src,Pt2di(0,0),sz,Ims,pal);
   mImIncr.load(Pt2di(0,0),p0Dest,sz);
}


void VideoWin_Visu_ElImScr:: XIm_write_image_brute
              (
                    Pt2di P0Src,
                    Pt2di P0Dest,
                    Pt2di Sz,
                    Im2D_U_INT1 ImR,
                    Im2D_U_INT1 ImG,
                    Im2D_U_INT1 ImB
              )
{
   mBigIm.fill_with_el_image(P0Src,P0Dest,Sz,ImR,ImG,ImB);
}


/****************************************************************/
/*                                                              */
/*             ElImIncruster                                    */
/*                                                              */
/****************************************************************/  

ElImIncruster::~ElImIncruster()
{
   for (INT k=0; k<(INT)mMEID.size() ; k++)
       delete mMEID[k];
}

void ElImIncruster::InitSz()
{
   mSzImUtiCur.SetInf(mSzImUti0);
   mSzBrdCur.SetInf(mSzBrd0);
   mSzImTotCur = mSzImUtiCur + mSzBrdCur*2;
   mCenter = mSzImTotCur/2;
}

ElImIncruster::ElImIncruster
(
     Pt2di                           SzIm,
     Pt2di                           SzBrd,
     Visu_ElImScr &                  Visu,
     const ElSTDNS vector<ElImScroller*>   & Scrols,
     INT                             NbImRab
)   :
    mSzBrd0      (SzBrd),
    mSzImUti0    (SzIm),
    mSzImTot0    (mSzImUti0 + mSzBrd0*2),
    mSzBrdCur    (SzBrd),
    mSzImUtiCur  (SzIm),
    mDec         (123456,-67890),  // BullShit
    mTabScr      (Scrols),
    mTabAttrScr  (Scrols.size()),
    mVisu        (Visu),
    mLastDecUpToDate (false),
    mSynchroGeom     (false)
{
    InitSz();
    for (INT k=0; k<(INT)Scrols.size() ; k++)
    {
         mMEID.push_back(new Memory_ElImDest<INT4>(mTabScr[k]->DimOut(),mSzImTot0));
         
         std::vector<Im2D_INT4>  & ims = mMEID[k]->Images();
         for (INT ki=0 ; ki<(INT)ims.size() ; ki++)
         {
             mImagesIn.push_back(ims[ki]);
         }
    }

	{
    for (INT k=0 ; k< ((INT)mImagesIn.size()+NbImRab) ; k++)
         mImagesOut.push_back(Im2D_INT4(mSzImTot0.x,mSzImTot0.y));
	}
}

void ElImIncruster::SetActiveKthIm(INT Kth,bool active)
{
    mTabAttrScr[Kth].mIsActif = active;
}

Pt2di ElImIncruster::P0Tot() const { return  mDec;}
Pt2di ElImIncruster::P1Tot() const { return  mDec+mSzImTotCur;}

Pt2di ElImIncruster::P0Uti() const { return  mDec+mSzBrdCur;}
Pt2di ElImIncruster::P1Uti() const { return  mDec+mSzBrdCur+mSzImUtiCur;}

Pt2di ElImIncruster::CurSize() const { return mSzImUtiCur;}
Pt2di ElImIncruster::CurBrd()  const { return mSzBrdCur;}

Pt2di ElImIncruster::BrdMax()  const { return mSzBrd0;}

INT ElImIncruster::NbImInMax() const {return INT(mImagesIn.size());}


Box2di ElImIncruster::BoxIncrust() const
{
    return Box2di(P0Uti(),P1Uti());
}

void ElImIncruster::SetSize(Pt2di aSz)
{
   mSzImUtiCur = aSz;
   InitSz();
}


void ElImIncruster::SetBrd(Pt2di aBrd)
{
   mSzBrdCur = aBrd;
   InitSz();
   mLastDecUpToDate = false;
}


void ElImIncruster::IncrSzAndReaff(Pt2di PCenter,Pt2di dSz,Elise_Palette * aPal)
{
   dSz.SetInf(mSzImUti0-mSzImUtiCur);
   mDec = PCenter-mCenter;
   mSzImUtiCur  += dSz;

   SetBox (Box2di(P0Uti(),P1Uti()),aPal );
}

void ElImIncruster::IncrSzAndReaff(Pt2di PCenter,Pt2di dSz,Elise_Palette  aPal)
{
   IncrSzAndReaff(PCenter,dSz,&aPal);
}

void ElImIncruster::IncrSzAndReaff(Pt2di PCenter,Pt2di dSz)
{
   IncrSzAndReaff(PCenter,dSz,0);
}

REAL ElImIncruster::ScaleScrol(INT )
{
    return mScaleScrol;
}

Pt2dr ElImIncruster::TrScrol(INT k)
{
    return mTrScrol + (mTabScr[k]->tr() - mTabScr[0]->tr()) ;
}


void ElImIncruster::BeginIncrust(Pt2dr aPInvW,REAL aScale)
{
    mTrScrol    = mTabScr[0]->TrOfScArroundPW(aPInvW,aScale);
    mScaleScrol = aScale;
    mLastDecUpToDate = false;
}

void ElImIncruster::BeginIncrust()
{
    BeginIncrust
    (
       Pt2dr(0,0),      // Si on ne change pas le scale, la valeur n'a pas d'importance
       mTabScr[0]->sc()
    );
}

void ElImIncruster::EndIncrust()
{
    mVisu.load_rect_image(P0Tot(),P1Tot(),false);
}




void ElImIncruster::SetBox(Box2di NewBox,Elise_Palette * aPal)
{
    mDec = NewBox._p0 -mSzBrdCur;
    mSzImUtiCur  = NewBox.sz();
    InitSz();
    NewBox = Box2di(P0Uti(),P1Uti());


    bool from_scratch = (! PixelIndep()) || (! mLastDecUpToDate) ;

    if (! from_scratch)
    {
        ModelBoxSubstr NewElt(NewBox,Box2di(P0Tot(),P1Tot()));

        Pt2di BrCost = mSzBrdCur/2;
        INT cost_scratch =  NewBox.dilate(BrCost).surf();
        INT cost_dif = 0;

        for (INT k=0 ; k<NewElt.NbBox() ; k++)
            cost_dif += NewElt.Box(k).dilate(BrCost).surf();

        
        if (cost_scratch < cost_dif)
           from_scratch = true;
    }


       
    for (INT k=0 ; k<(INT)mTabScr.size() ; k++)
    {
        mMEID[k]->SetDec(mDec);
        if ((k!=0) && (mSynchroGeom))
           mTabScr[k]->SetSameGeom(*mTabScr[0]);
    }


    if (mLastDecUpToDate)
    {
       for (INT k=0 ; k<(INT)mImagesIn.size() ; k++)
          mImagesIn[k].auto_translate(-(NewBox._p0-mOldBox._p0));

       ModelBoxSubstr BoxSub(NewBox.dilate(mSzBrdCur),mOldBox.dilate(mSzBrdCur));
       for (INT kb=0 ; kb<BoxSub.NbBox() ; kb++)
       {
            Box2di BK = BoxSub.Box(kb);
            Box2di BKloc = BK.trans(-mDec);
            for (INT k=0 ; k<(INT)mImagesIn.size() ; k++)
                mImagesIn[k].raz(BKloc._p0,BKloc._p1);
			{
            for (INT k=0 ; k<(INT)mImagesOut.size() ; k++)
                mImagesOut[k].raz(BKloc._p0,BKloc._p1);
			}

			{
            for (INT k=0 ; k<(INT)mTabScr.size() ; k++)
            {
               if (mTabAttrScr[k].mIsActif)
                  mTabScr[k]->LoadXImageInVisu
                  (
                        *mMEID[k],
                        BK._p0,
                        BK._p1,
                        false,
                        TrScrol(k),
                        ScaleScrol(k)
                   );
            }
			}
       }
    }
    else
    {
        for (INT k=0 ; k<(INT)mImagesIn.size() ; k++)
            mImagesIn[k].raz();
		{
        for (INT k=0 ; k<(INT)mImagesOut.size() ; k++)
            mImagesOut[k].raz();
		}

		{
        for (INT k=0 ; k<(INT)mTabScr.size() ; k++)
        {
            if (mTabAttrScr[k].mIsActif)
               mTabScr[k]->LoadXImageInVisu
               (
                        *mMEID[k],
                        P0Tot(),
                        P1Tot(),
                        false,
                        TrScrol(k),
                        ScaleScrol(k)
                );
        }
		}
    }



    if (from_scratch)
       SetNewBoxFromScratch(NewBox,aPal);
    else
       SetNewBoxDiffer(NewBox,aPal);


   if (mLastDecUpToDate)
   {
      ModelBoxSubstr  BoxSub(mOldBox,NewBox);
      for (INT k=0 ; k<BoxSub.NbBox() ; k++)
      {
          mVisu.load_rect_image
          (
              BoxSub.Box(k)._p0,
              BoxSub.Box(k)._p1,
              false
          );
      }
   }                      
    mLastDecUpToDate = true;
    mOldBox = NewBox;
}



void ElImIncruster::SetNewBoxDiffer(Box2di NewBox,Elise_Palette * aPal)
{
   for (INT k=0 ; k<(INT)mImagesOut.size() ; k++)
       mImagesOut[k].auto_translate(-(NewBox._p0-mOldBox._p0));

   ModelBoxSubstr  BoxSub(NewBox,mOldBox);
   for (INT kb=0 ; kb<BoxSub.NbBox() ; kb++)
       SetNewBoxFromScratch(BoxSub.Box(kb),aPal);
}

void ElImIncruster::SetNewBoxFromScratch(Box2di aBox,Elise_Palette * aPal)
{

   INT NbIm = Filters(mTabAttrScr,mImagesOut,mImagesIn,aBox._p0-mDec,aBox._p1-mDec);

   Pt2di P0Loc  = aBox._p0-mDec;
   if (aPal)
   {
        mVisu.InscrustImage(mImagesOut,P0Loc,aBox._p0,aBox.sz(),NbIm,*aPal);
   }
   else
       mVisu.InscrustImage(mImagesOut,P0Loc,aBox._p0,aBox.sz(),NbIm);
}





void ElImIncruster::IncrustOrigineAtPtW(Pt2di pt,Elise_Palette  *aPal) 
{
   SetBox(Box2di(pt+mSzBrdCur,pt+mSzImUtiCur+mSzBrdCur),aPal);
}






void ElImIncruster::IncrustOrigineAtPtW(Pt2di pt)
{
   IncrustOrigineAtPtW(pt,0);
}

void ElImIncruster::IncrustOrigineAtPtW(Pt2di pt,Elise_Palette aPal)
{
   IncrustOrigineAtPtW(pt,&aPal);
}


void ElImIncruster::IncrustCenterAtPtW(Pt2di pt)
{
   IncrustOrigineAtPtW(pt-mCenter);
}

void ElImIncruster::IncrustCenterAtPtW(Pt2di pt,Elise_Palette aPal)
{
   IncrustOrigineAtPtW(pt-mCenter,aPal);
}

/****************************************************************/
/*                                                              */
/*             Filtr_Incr_EqDyn_Glob                            */
/*             Filtr_Incr_EqDyn_Loc                             */
/*                                                              */
/****************************************************************/  


           // Filtr_Incr_EqDyn_Glob

Filtr_Incr_EqDyn_Glob:: Filtr_Incr_EqDyn_Glob() :
    mLut (256),
    mDataLut (mLut.data())
{
}
 
 
INT Filtr_Incr_EqDyn_Glob::Filters
    (
       ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
       ElSTDNS vector<Im2D_INT4> &ImsOut,
       ElSTDNS vector<Im2D_INT4> &ImsIn,
       Pt2di p0,
       Pt2di p1,
       REAL Fact
    )
{

    INT iChOut = 0;
    for 
    (
        INT iChIn=0;
        (iChIn<(INT)ImsIn.size())&&(iChIn<(INT)ImsIn.size()) && (iChOut<(INT)ImsOut.size()); 
        iChIn++
    )
      if (Attrs[iChIn].mIsActif)
      {
         INT ** data_in  = ImsIn[iChIn].data();
         REAL rs0 = (p1.x-p0.x) * (p1.y-p0.y);
         REAL rs1 = 0;
         REAL rs2 =0;
 

         INT val;
         for (INT y=p0.y ;y<p1.y; y++)
         {
             for (INT x=p0.x ;x<p1.x; x++)
             {
                 val = data_in[y][x];
                 rs1 += val;
                 rs2 += ElSquare(val);
             }
         }
         rs1 /= rs0;
         rs2 = rs2/rs0 -ElSquare(rs1);
         REAL sigma =  sqrt(ElMax(0.01,rs2));


         ELISE_COPY
         (
             rectangle(p0,p1),
             255*erfcc(Fact*(ImsIn[iChIn].in()-rs1)/sigma),
             ImsOut[iChOut].out()
         );

          iChOut++;
      }
 
    return iChOut;
}
 
           // Filtr_Incr_EqDyn_Loc
                                            
Filtr_Incr_EqDyn_Loc::Filtr_Incr_EqDyn_Loc() :
     mConvMne (false)
{
}

Filtr_Incr_EqDyn_Loc::Filtr_Incr_EqDyn_Loc(ConvMne_Filtr_Incr_EqDyn_Loc,INT ValSpec) :
     mConvMne (true),
     mValSpec (ValSpec)
{
}


INT Filtr_Incr_EqDyn_Loc:: Filters
     (
           ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
           ElSTDNS vector<Im2D_INT4> & ImsOut,
           ElSTDNS vector<Im2D_INT4> & ImsIn,
           Pt2di p0,
           Pt2di p1,
           INT Brd,
           REAL Fact
     )
{
    INT iChOut = 0;
    for 
    (
        INT iChIn=0;
        (iChIn<(INT)ImsIn.size())&&(iChIn<(INT)ImsIn.size()) && (iChOut<(INT)ImsOut.size()); 
        iChIn++
    )
      if (Attrs[iChIn].mIsActif)
      {
         Im2D_INT4 im = ImsIn[iChIn];
         Symb_FNum  FoncInit(im.in(0));

          Fonc_Num Masq = mConvMne ? (FoncInit!=mValSpec) : 1.0;

         Symb_FNum  Fonc (FoncInit * Masq);
         Symb_FNum  Pond (im.inside()*Masq);

         Symb_FNum  S012 (rect_som(Virgule(Rconv(Pond),Fonc,ElSquare(Fonc)),Brd));
 
         Symb_FNum s0 (Rconv(S012.v0()));
         Symb_FNum s1 (S012.v1()/s0);
         Symb_FNum s2 (S012.v2()/s0-Square(s1));
         Symb_FNum ect  (sqrt(Max(0.01,s2)));
 
         ELISE_COPY
         (
              rectangle(p0,p1),
              255*erfcc((im.in()-s1)/ect),
              ImsOut[iChOut].out()
         );
         iChOut++;
         
      }
    return iChOut;
}



           // Filtr_Incr_EqDyn_Loc
                                            
Filtr_Incr_EtalDyn::Filtr_Incr_EtalDyn()
{
}

INT Filtr_Incr_EtalDyn:: Filters
     (
           ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
           ElSTDNS vector<Im2D_INT4> & ImsOut,
           ElSTDNS vector<Im2D_INT4> & ImsIn,
           Pt2di p0,
           Pt2di p1,
           REAL Vmin,
           REAL Vmax
     )
{
    INT iChOut = 0;
    for 
    (
        INT iChIn=0;
        (iChIn<(INT)ImsIn.size())&&(iChIn<(INT)ImsIn.size()) && (iChOut<(INT)ImsOut.size()); 
        iChIn++
    )
      if (Attrs[iChIn].mIsActif)
      {
        ELISE_COPY
        (
             rectangle(p0,p1),
             Min(255,Max(0,(ImsIn[iChIn].in()-Vmin)*(255.0/(Vmax-Vmin)))),
             ImsOut[iChOut].out()
        );
          iChOut++;
      }
   return iChOut;
}


           // Filtr_Incr_NormDeriche

Filtr_Incr_NormDeriche::Filtr_Incr_NormDeriche()
{
}

INT Filtr_Incr_NormDeriche:: Filters
     (
           ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
           ElSTDNS vector<Im2D_INT4> & ImsOut,
           ElSTDNS vector<Im2D_INT4> & ImsIn,
           Pt2di p0,
           Pt2di p1,
           REAL FactDer,
           REAL Dyn
     )
{
    INT iChOut = 0;
    for 
    (
        INT iChIn=0;
        (iChIn<(INT)ImsIn.size())&&(iChIn<(INT)ImsIn.size()) && (iChOut<(INT)ImsOut.size()); 
        iChIn++
    )
      if (Attrs[iChIn].mIsActif)
      {
        ELISE_COPY
        (
             rectangle(p0,p1),
             Min
             (
                 255,
                 Max
                 (
                     0,
                     Polar_Def_Opun::polar(deriche(ImsIn[iChIn].in_proj(),FactDer),0).v0()*Dyn
                 )
             ),
             ImsOut[iChOut].out()
        );
          iChOut++;
      }
   return iChOut;
}


           // Filtr_Incr_AngDeriche

Filtr_Incr_AngDeriche::Filtr_Incr_AngDeriche()
{
}

INT Filtr_Incr_AngDeriche:: Filters
     (
           ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
           ElSTDNS vector<Im2D_INT4> & ImsOut,
           ElSTDNS vector<Im2D_INT4> & ImsIn,
           Pt2di p0,
           Pt2di p1,
           REAL FactDer
     )
{
    INT iChOut = 0;
    for 
    (
        INT iChIn=0;
        (iChIn<(INT)ImsIn.size())&&(iChIn<(INT)ImsIn.size()) && (iChOut<(INT)ImsOut.size()); 
        iChIn++
    )
      if (Attrs[iChIn].mIsActif)
      {
          ELISE_COPY
          (
             rectangle(p0,p1),
             Polar_Def_Opun::polar(deriche(ImsIn[iChIn].in_proj(),FactDer,20),0).v1() * (256.0/(2*PI)),
             ImsOut[iChOut].out()
          );
          iChOut++;
      }
   return iChOut;
}




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
