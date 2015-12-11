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
/*     ElImScroller                                             */
/*                                                              */
/****************************************************************/


ElImScroller::ElImScroller
(
   Visu_ElImScr &Visu,
   INT          aDimOut,
   Pt2di          SzU,
   REAL           sc_im
)       :
     _SzW                (Visu.SzW()),
     _SzU                (SzU),
     mTimeLoadXIm        (0.0),
     mTimeReformat       (0.0),
     mVisuStd            (Visu),
     mVisuCur            (&mVisuStd),
     mDimOut             (aDimOut),
    _tr(0,0),
    _sc(1),
     _sc_im              (sc_im),
     mAlwaysQuickInZoom  (false),
     mAlwaysQuick        (false),
     mSetInit            (false)
{

 // std::cout << "DIMOUT " << aDimOut  << " " << sc_im  << " " << this << "\n"; 
} 


/*
*/

void ElImScroller::write_image(INT x0src,Pt2di p0dest,INT nb,INT ** data,int aNbChanelIn)
{
   mVisuCur->write_image(x0src,p0dest,nb,data,aNbChanelIn);
}
void ElImScroller::write_image(INT x0src,Pt2di p0dest,INT nb,double ** data,int aNbChanelIn)
{

   mVisuCur->write_image(x0src,p0dest,nb,data,aNbChanelIn);
}


ElImScroller * ElImScroller::CurScale() 
{
   return this;
}

bool  ElImScroller::CanReinitTif()
{
   return true;
}

void ElImScroller::ReInitTifFile(Tiff_Im aTif)
{
    ELISE_ASSERT(false,"ElImScroller::ReInitTifFile no def value");
}

void ElImScroller::SetAlwaysQuickInZoom()
{
   mAlwaysQuickInZoom = true;
}

void ElImScroller::SetAlwaysQuickInZoom(bool aVal)
{
   mAlwaysQuickInZoom = aVal;
}

void ElImScroller::SetAlwaysQuick(bool aVal)
{
   mAlwaysQuick  = aVal;
}

void ElImScroller::SetAlwaysQuick()
{
   SetAlwaysQuick(true);
}


bool ElImScroller::AlwaysQuick() const
{
    return mAlwaysQuick || (mAlwaysQuickInZoom && (_sc>1.0));
}

bool ElImScroller::AlwaysQuickZoom() const
{
   return mAlwaysQuickInZoom;
}

void ElImScroller::LoadAndVerifXImage(Pt2di p0W,Pt2di p1W,bool quick)
{
     p0W = Sup3( Pt2di(0,0),p0W,round_up(to_win(Pt2dr(0,0)) ));
     p1W = Inf3(       _SzW,p1W,round_down(to_win(Pt2dr(_SzU))       ));

     mVisuCur->VerifDim(mDimOut);
     LoadXImage(p0W,p1W,quick||AlwaysQuick());
}




ElImScroller::~ElImScroller() {}

void ElImScroller::LoadIm(bool quick)
{
   LoadAndVerifXImage(Pt2di(0,0),_SzW,quick);
}

void ElImScroller::VisuIm(bool Quick)
{
   mVisuStd.load_rect_image(Pt2di(0,0),_SzW,Quick);
}

void ElImScroller::VisuIm(Pt2di pW0,Pt2di pW1,bool Quick)
{
   mVisuStd.load_rect_image(pW0,pW1,Quick);
}

void ElImScroller::LoadAndVisuIm(bool quick)
{
  LoadIm(quick);
  VisuIm(quick);
}

void ElImScroller::LoadAndVisuIm(Pt2di aP0,Pt2di aP1,bool quick)
{
  LoadAndVerifXImage(aP0,aP1,quick);
  VisuIm(aP0,aP1,quick);
}


void ElImScroller::set_geom(Pt2dr tr,REAL sc)
{
   _tr = tr*_sc_im;
   _sc = sc / _sc_im;
}

bool ElImScroller::image_full_in()
{
   return 
            to_user(Pt2dr(0,0)).in_box(Pt2dr(0,0),Pt2dr(_SzU))
         && to_user(Pt2dr(_SzW)).in_box(Pt2dr(0,0),Pt2dr(_SzU));
}

Pt2di ElImScroller::PrefXim()
{
    return Pt2di(to_win(Pt2dr(0,0)));
}

void ElImScroller::set(Pt2dr tr,REAL sc,bool quick)
{
   set_geom(tr,sc);


   if (! image_full_in())
   {
      mVisuStd.write_image_out(-PrefXim(),Pt2di(0,0),_SzW);
   }

   LoadAndVisuIm(quick);
   mSetInit = true;
}


REAL ElImScroller::ScMax() const
{
   return SzW().RatioMin(SzU());
}

void ElImScroller::set_max(bool quick)
{
   REAL sc = ScMax();
   Pt2dr tr = (Pt2dr(SzU())-Pt2dr(SzW())/sc ) /2.0 ;


   set(tr,sc,quick);
}

void ElImScroller::set_max_init()
{
     if (! mSetInit)
     {
        _sc = ScMax() +1;
        set_max();
     }
}


void ElImScroller::SetSameGeom(const ElImScroller & aScr)
{
   set_geom(aScr.tr(),aScr.sc());
}

void ElImScroller::SetGeomTranslated(const ElImScroller & aScr,Pt2dr aTrans)
{
   set_geom(aScr.tr()+aTrans,aScr.sc());
}





Pt2dr ElImScroller::TrOfScArroundPW(Pt2dr aPinvW,REAL aNewSc)
{
    return  to_user(aPinvW) -aPinvW/aNewSc;
}



void ElImScroller::SetScArroundPW(Pt2dr aPinvW,REAL aNewSc,bool quick)
{
	set(TrOfScArroundPW(aPinvW,aNewSc),aNewSc,quick);
}


void ElImScroller::SetTrU(Pt2dr tr)
{
    SetDTrW(round_ni((_tr-tr)*_sc));
    
}

REAL  ElImScroller::GetValPtsR(Pt2dr aP)
{
    ELISE_ASSERT(false,"No ElImScroller::GetValPtsR");
    return 0.0;
}

/*

void GetIntervUpdate(INT dx,INT IntervX0,INT IntervX1,INT rab,INT &x0,INT & x1)
{
     if (dx>0)
     {
        x0 = IntervX1-dx-rab;
        x1 = IntervX1;
     }
     else
     {
        x0 = IntervX0;
        x1 = IntervX0 -dx + rab;
     }
}

void ElImScroller::GetIntervUpdate(INT dx,INT tx,INT &x0,INT & x1)
{
     ::GetIntervUpdate(dx,0,tx,round_ni(RabUpdate+_sc),x0,x1);
}


void GetBoxUpdate(Pt2di tr,Box2di box,INT rab,Box2di & BX,Box2di & BY)
{
    INT x0,x1;
    GetIntervUpdate(tr.x,box._p0.x,box._p1.x,rab,x0,x1);
    BX = Box2di( Pt2di(x0,box._p0.y), Pt2di(x1,box._p1.y));


    INT y0,y1;
    GetIntervUpdate(tr.y,box._p0.y,box._p1.y,rab,y0,y1);
    BY = Box2di( Pt2di(box._p0.x,y0), Pt2di(box._p1.x,y1));
}

*/


static void Adapt(INT & x,INT Rab)
{
     if (x>=0) 
        x+= Rab;
     else
        x-= Rab;
}

ModelBoxSubstr ElImScroller::SetDTrW(Pt2di dtrW)
{
    Pt2dr dtrU = dtrW / _sc;
     _tr   = _tr + dtrU;


  
    Pt2di DrtW = -dtrW;
    Adapt(DrtW.x,round_ni(RabUpdate+_sc));
    Adapt(DrtW.y,round_ni(RabUpdate+_sc));

    Box2di  b0(Pt2di(0,0),_SzW);
    Box2di  b1(DrtW,DrtW+_SzW);
    ModelBoxSubstr aMod(b0,b1); 

    if (dist4(dtrW) > dist4(_SzW)/3.0)
    {
	   set(_tr,_sc);
	   return aMod;
    }
    mVisuStd.translate(-dtrW);
    

    if  (! image_full_in())
    {
        for (INT k=0; k<aMod.NbBox() ; k++)
        {
            Box2di aBox = aMod.Box(k);
            mVisuStd.write_image_out(aBox._p0-PrefXim(),aBox._p0, aBox._p1-aBox._p0);
        }
    }

    for (INT k=0; k<aMod.NbBox() ; k++)
    {
        Box2di aBox = aMod.Box(k);
        LoadAndVerifXImage( aBox._p0,aBox._p1,false);
        mVisuStd.load_rect_image( aBox._p0,aBox._p1,false);
    }
    return aMod;
}

void ElImScroller::no_use()
{
}


void ElImScroller::LoadXImageInVisu
     (
          Visu_ElImDest & aTmpVisu,
          Pt2di p0W,
          Pt2di p1W,
          bool quick,
          Pt2dr aTr,
          REAL  aSc
     )
{
   Pt2dr tr0 = _tr;
   REAL  sc0 = _sc;

   set_geom(aTr,aSc);
   
   SetVisuCur( &aTmpVisu);
   LoadAndVerifXImage(p0W,p1W,quick);
   SetVisuCur(&mVisuStd);

   set_geom(tr0,sc0);
}



void ElImScroller::LoadXImageInVisu(Visu_ElImDest & aTmpVisu,Pt2di p0W,Pt2di p1W,bool quick)
{
   LoadXImageInVisu(aTmpVisu,p0W,p1W,quick,_tr,_sc);
}


void ElImScroller::SetVisuCur(Visu_ElImDest * pVEID)
{ 

    mVisuCur = pVEID;
    ReflexSetVisuCur(pVEID);
}

void ElImScroller::ReflexSetVisuCur(Visu_ElImDest * pVEID)
{
}

ElImScroller * ElImScroller::StdScrollIfExist
(
        Visu_ElImScr &aVisu,
        const std::string & aName,
        REAL scale,
        bool AdaptPal,
        bool ForceGray
)
{
    if (! ELISE_fp::exist_file(aName.c_str()))
       return 0;

  Tiff_Im aTifFile = Tiff_Im::StdConvGen(aName,-1,true);

  if (aTifFile.OkFor_un_load_pack_bit_U_INT1())
  {
        PackB_IM<U_INT1> aPckbIm = aTifFile.un_load_pack_bit_U_INT1();
        aVisu.AdaptTiffFile(aTifFile,AdaptPal,ForceGray);
        if (ForceGray || (aTifFile.phot_interp() != Tiff_Im::RGBPalette))
            return new PckBitImScroller(aVisu,aPckbIm,scale);

        Disc_Pal   aPal = aTifFile.pal(); 

        Elise_colour *  cols = aPal.create_tab_c();
        INT aNbCol = aPal.nb_col();
        RGBLut_PckbImScr * aRGB =  new RGBLut_PckbImScr (aVisu,aPckbIm,cols,aNbCol,scale);
        DELETE_VECTOR(cols,0);
        return aRGB;
  }


  switch (aTifFile.type_el())
  {
     case GenIm::u_int1 :
         aVisu.AdaptTiffFile(aTifFile,AdaptPal,ForceGray);
         return new ImFileScroller<U_INT1> (aVisu,aTifFile,scale);
     break;

     case GenIm::int2 :
         aVisu.AdaptTiffFile(aTifFile,AdaptPal,ForceGray);
         return new ImFileScroller<INT2> (aVisu,aTifFile,scale);
     break;

     case GenIm::u_int2 :
         aVisu.AdaptTiffFile(aTifFile,AdaptPal,ForceGray);
         return new ImFileScroller<U_INT2> (aVisu,aTifFile,scale);
     break;


     case GenIm::real4 :
         aVisu.AdaptTiffFile(aTifFile,AdaptPal,ForceGray);
         return new ImFileScroller<REAL4> (aVisu,aTifFile,scale);
     break;

     case GenIm::real8 :
         aVisu.AdaptTiffFile(aTifFile,AdaptPal,ForceGray);
         return new ImFileScroller<REAL8> (aVisu,aTifFile,scale);
     break;


     default :
         return 0;
  }

  return 0;
}


ElImScroller * ElImScroller::StdFileGenerique
(
       Visu_ElImScr &aVisu,
       const std::string & aName,
       INT InvScale,
       bool VisuAdaptPal ,
       bool ForceGray ,
       cElScrCalcNameSsResol * aCalcName
)
{
    static std::string Reduc("Reduc");
    static std::string tif(".tif");
    static std::string _INTERPOLE_MEAN_("_INTERPOLE_MEAN_");

    REAL aScale = 1.0/InvScale;
    char CScale[10];
    sprintf(CScale,"%d",InvScale);
    ElSTDNS string Scale(CScale);

    ElImScroller * res =0;

   // ===================
    if (aCalcName)
    {
        std::string aNewName = aCalcName->CalculName(aName,InvScale);
        res = ElImScroller::StdScrollIfExist(aVisu,aNewName,aScale,VisuAdaptPal,ForceGray);

        if (res)
        {
            return res;
        }
    }


    // Test fichier tif "Reduc"
    res = ElImScroller::StdScrollIfExist(aVisu,aName+Reduc+Scale+tif,aScale,VisuAdaptPal,ForceGray);
    if (res) 
       return res;

    // Test fichier saphir  "Reduc"
   res =   ElImScroller::StdScrollIfExist(aVisu,aName+_INTERPOLE_MEAN_+Scale,aScale,VisuAdaptPal,ForceGray);
   if (res) 
       return res;

   if (InvScale!= 1)
      return 0;

    // Test fichier tif Initiale
    res =   ElImScroller::StdScrollIfExist(aVisu,aName+tif,aScale,VisuAdaptPal,ForceGray);
    if (res) 
       return res;

    // Test fichier saphir Initiale
    res =   ElImScroller::StdScrollIfExist(aVisu,aName,aScale,VisuAdaptPal,ForceGray);
    if (res) 
       return res;

   return 0;
}


ElPyramScroller * ElImScroller::StdPyramide
                  (
                         Visu_ElImScr &aVisu,
                         const std::string & aName,
                         std::vector<INT> * EchAcc,
                         bool VisuAdaptPal ,
                         bool ForceGray ,
                         cElScrCalcNameSsResol *  aCalcName
                  )
{
     ElSTDNS vector <ElImScroller *>  VScrol; 

     for (INT InvScale=1 ; InvScale< 128 ; InvScale*=2)
     {
          bool OkScale  =true;
          if (EchAcc && EchAcc->size())
          {
               OkScale = (std::find(EchAcc->begin(),EchAcc->end(),InvScale) != EchAcc->end());
          }
          if (OkScale)
          {
             ElImScroller * aScr = StdFileGenerique(aVisu,aName,InvScale,VisuAdaptPal,ForceGray,aCalcName);
             if (aScr)
             {
                VScrol.push_back(aScr);
             }
          }
     }
     if (VScrol.size() ==0)
     {
        cout << "For " << aName << "\n";
        ELISE_ASSERT(false,"No Image in ElPyramScroller");
        return 0;
     }

    return new ElPyramScroller(VScrol);
}


Output ElImScroller::out()
{
   return Output::onul();
}

Fonc_Num ElImScroller::in()
{
   return 0;
}


void ElImScroller::Sauv(const std::string & aName)
{
    ELISE_ASSERT(false,"No Sauv for ElImScroller");
}


Flux_Pts ElImScroller::ContVect2RasFlux(std::vector<Pt2dr> VPts)
{
    ElList<Pt2di> Li;

    for 
    (
         std::vector<Pt2dr>::iterator itP = VPts.begin();
         itP != VPts.end();
         itP++
    )
    {
       Li = Li + Pt2di(*itP * sc_im());
    }
    return quick_poly(Li);
}


void ElImScroller::SetPoly (Fonc_Num aFonc,std::vector<Pt2dr> VPts)
{
    ELISE_COPY(ContVect2RasFlux(VPts),aFonc,out());
}

void ElImScroller::ApplyLutOnPoly(Fonc_Num aLut,std::vector<Pt2dr> VPts)
{
    ELISE_COPY(ContVect2RasFlux(VPts),aLut[in()],out());
}

REAL ElImScroller::TimeLoadXIm()  const { return mTimeLoadXIm;}
REAL ElImScroller::TimeUnCompr()  const { return 0.0;}
REAL ElImScroller::TimeReformat() const { return mTimeReformat;}

/****************************************************************/
/*                                                              */
/*     PckBitImScroller                                         */
/*                                                              */
/****************************************************************/

void PckBitImScroller::RasterUseLine(Pt2di p0,Pt2di p1,INT ** l,int aNbChan)
{
     ElTimer aTimer;
     for (INT y= p0.y ; y<p1.y ; y++)
     {
          write_image (p0.x,Pt2di(p0.x,y),p1.x-p0.x,l,aNbChan);
     }                      
     mTimeLoadXIm += aTimer.uval();
}

void PckBitImScroller::LoadXImage(Pt2di p0,Pt2di p1,bool quick)
{
   do_it(tr(),sc(),p0,p1,quick);
}


PckBitImScroller::PckBitImScroller
(
   Visu_ElImScr & visu,
   PackB_IM<U_INT1> pim,
   REAL             sc_im
) :
	ElImScroller(visu,1,pim.sz(),sc_im),
    StdGray_Scale_Im_Compr(pim,visu.SzW())
{
}


Output PckBitImScroller::out()
{
   return _pbim.out();
}

Fonc_Num PckBitImScroller::in()
{
   return _pbim.in();
}

Pt2di  PckBitImScroller::SzIn() 
{
   return _pbim.sz();
}


REAL PckBitImScroller::TimeUnCompr() const
{
   return mTimeUnCompr;
}

void PckBitImScroller::SetPoly (Fonc_Num aFonc,std::vector<Pt2dr> VPts)
{
    REAL aVal;

    if (aFonc.IsCsteRealDim1(aVal))
    {
       ELISE_COPY(ContVect2RasFlux(VPts),0,_pbim.OutLut(round_ni(aVal)));
    }
    else
    {
        ElImScroller::SetPoly(aFonc,VPts);
    }
}

void PckBitImScroller::ApplyLutOnPoly(Fonc_Num aLut,std::vector<Pt2dr> VPts)
{
     ELISE_COPY(ContVect2RasFlux(VPts),0,_pbim.OutLut(aLut));
}

/****************************************************************/
/*                                                              */
/*     RGB_PckbImScr et derivees                                */
/*                                                              */
/****************************************************************/


         //==================   RGB_PckbImScr ================

RGB_PckbImScr::RGB_PckbImScr
(
      Visu_ElImScr & Visu,
      Pt2di Sz,
      REAL ScaleIm
) :
  ElImScroller(Visu,3,Sz,ScaleIm),
   mP0Im    (-4,0),
   mP1Im    (Visu.SzW().x+4,3),
   mIm      (NEW_MATRICE(mP0Im,mP1Im,INT)),
   mRIm     (mIm[0]),
   mGIm     (mIm[1]),
   mBIm     (mIm[2])
{
}

Pt2di RGB_PckbImScr::SzIn() 
{
   return mP1Im-mP0Im;
}
                      


RGB_PckbImScr::~RGB_PckbImScr()
{
    DELETE_MATRICE(mIm,mP0Im,mP1Im);
}

void  RGB_PckbImScr::WriteRGBImage(Pt2di p0,Pt2di p1,RGB_Int ** Tl)
{
    
    ElTimer aTimer;
    RGB_Int * rgb = Tl[0];

    for (INT x = p0.x ; x<p1.x ; x++)
    {
       mRIm[x] = rgb[x]._r;
       mGIm[x] = rgb[x]._g;
       mBIm[x] = rgb[x]._b;
    }

     mTimeReformat += aTimer.uval();
     aTimer.reinit();
     for (INT y= p0.y ; y<p1.y ; y++)
     {
          write_image (p0.x,Pt2di(p0.x,y),p1.x-p0.x,mIm,3);
     }                      
     mTimeLoadXIm += aTimer.uval();
}

       //======  RGBLut_PckbImScr ==============================


RGBLut_PckbImScr::RGBLut_PckbImScr
(
    Visu_ElImScr &    Visu,
    PackB_IM<U_INT1>  PckbIm,
    Elise_colour *    cols,
    INT               nb,
    REAL              sc_im
)  :
   RGB_PckbImScr(Visu,PckbIm.sz(),sc_im),
   RGBLut_Scale_Im_Compr(PckbIm,Visu.SzW(),cols,nb)
{
}



void RGBLut_PckbImScr::LoadXImage(Pt2di p0,Pt2di p1,bool quick)
{
   do_it(tr(),sc(),p0,p1,quick);
}

Output RGBLut_PckbImScr::out()
{
   return _pbim.out();
}

Fonc_Num RGBLut_PckbImScr::in()
{
   return _pbim.in();
}

void  RGBLut_PckbImScr::RasterUseLine(Pt2di p0,Pt2di p1,RGB_Int ** Tl,int aNbChanIn)
{
    WriteRGBImage(p0,p1,Tl);
}

REAL RGBLut_PckbImScr::TimeUnCompr() const
{
   return mTimeUnCompr;
}

        //=============== RGBTrue16Col_PckbImScr =================


RGBTrue16Col_PckbImScr::RGBTrue16Col_PckbImScr
(
    Visu_ElImScr &    Visu,
    PackB_IM<U_INT2>  PckbIm,
    REAL              sc_im
)  :
   RGB_PckbImScr(Visu,PckbIm.sz(),sc_im),
   RGBTrue16Col_Scale_Im_Compr(PckbIm,Visu.SzW())
{
}


void RGBTrue16Col_PckbImScr::LoadXImage(Pt2di p0,Pt2di p1,bool quick)
{
   do_it(tr(),sc(),p0,p1,quick);
}

Output RGBTrue16Col_PckbImScr::out()
{
   return _pbim.out();
}

Fonc_Num RGBTrue16Col_PckbImScr::in()
{
   return _pbim.in();
}

void  RGBTrue16Col_PckbImScr::RasterUseLine(Pt2di p0,Pt2di p1,RGB_Int ** Tl,int aNbChanIn)
{
    WriteRGBImage(p0,p1,Tl);
}

REAL RGBTrue16Col_PckbImScr::TimeUnCompr() const
{
   return mTimeUnCompr;
}

/*********************************************************************/
/*                                                                   */
/*  ElPyramScroller                                                  */
/*                                                                   */
/*********************************************************************/


// Fonction identite pour contourner bug sur "," en WSCC5.0
static ElSTDNS vector<ElImScroller *>  VerifSize(ElSTDNS vector<ElImScroller *> & scrolls)
{
   ELISE_ASSERT(scrolls.size() !=0,"empty size inElPyramScroller");
   return scrolls;
}

ElPyramScroller::ElPyramScroller
(
	ElSTDNS vector<ElImScroller *> & scrolls
) :
    ElImScroller
    (
        VerifSize(scrolls)[0]->VisuStd(),
        scrolls[0]->DimOut(),
		scrolls[0]->SzUIm(),
        1.0
    ),
	_subs (scrolls),
	_cur  (0)
{
    for (INT k=1; k<(INT)scrolls.size(); k++)
        ELISE_ASSERT
        (
             scrolls[k]->DimOut() == DimOut(),
             "Variable DimOut in ElPyramScroller"
        );
}

ElImScroller * ElPyramScroller::CurScale()
{
   if (_cur) return _cur;
   return this;
}



void ElPyramScroller::LoadXImage(Pt2di p0,Pt2di p1,bool quick)
{

        ElImScroller  * ScrClosest  = 0;

   // Recherche de la plus basse resolution > a la resolution demandee
        for (INT k=0; k<(INT)_subs.size(); k++)
        {
// std::cout << "ElPyramScroller::LoadXImage  " << _subs[k]->sc_im()  << " " <<  sc_abs() << "\n";
            if (_subs[k]->sc_im() > sc_abs())
            {
                if (
                       (! ScrClosest)
                    || (_subs[k]->sc_im()<ScrClosest->sc_im())
                   )
                      ScrClosest = _subs[k];
            }
        }


   // Si pas trouvee Recherche de la plus haute resolution < a la resolution demandee
        if (! ScrClosest)
        {
            ScrClosest = _subs[0];
            for (INT k=1; k<(INT)_subs.size(); k++)
                 if (_subs[k]->sc_im() > ScrClosest->sc_im())
                    ScrClosest = _subs[k];
        }
 // std::cout << "ElPyramScroller::LoadXImage " << _cur << " => " <<   ScrClosest << "\n";

        if ((_cur!=0) && (_cur != ScrClosest))
           _cur->no_use();
        _cur = ScrClosest;
        ScrClosest->set_geom(tr(),sc_abs());
        ScrClosest->LoadXImage(p0,p1,quick|| AlwaysQuick());
}

void ElPyramScroller::Sauv(const std::string & aName)
{
    ELISE_ASSERT(false,"No Sauv for ElImScroller");
}


void ElPyramScroller::SetPoly (Fonc_Num aFonc,std::vector<Pt2dr> VPts)
{
     for (INT k=0; k<(INT)_subs.size(); k++)
         _subs[k]->SetPoly(aFonc,VPts);
}

void ElPyramScroller::ApplyLutOnPoly(Fonc_Num aLut,std::vector<Pt2dr> VPts)
{
     for (INT k=0; k<(INT)_subs.size(); k++)
         _subs[k]->ApplyLutOnPoly(aLut,VPts);
}

void ElPyramScroller::ReflexSetVisuCur(Visu_ElImDest * pVEID)
{
        for (INT k=0; k<(INT)_subs.size(); k++)
             _subs[k]->SetVisuCur(pVEID);
}


const ElSTDNS vector<ElImScroller *> & ElPyramScroller::SubScrolls()
{
   return _subs;
}


REAL ElPyramScroller::TimeLoadXIm()  const 
{ 
    REAL res =0.0;
    for (INT k=0; k<(INT)_subs.size(); k++)
        res +=  _subs[k]->TimeLoadXIm();
    return res;
}
REAL ElPyramScroller::TimeUnCompr()  const 
{ 
    REAL res =0.0;
    for (INT k=0; k<(INT)_subs.size(); k++)
        res +=  _subs[k]->TimeUnCompr();
    return res;
}
REAL ElPyramScroller::TimeReformat() const 
{ 
    REAL res =0.0;
    for (INT k=0; k<(INT)_subs.size(); k++)
        res +=  _subs[k]->TimeReformat();
    return res;
}

Fonc_Num ElPyramScroller::in()
{
   return _subs[0]->in(); 
}
Pt2di ElPyramScroller::SzIn()
{
   return _subs[0]->SzIn(); 
}

void  ElPyramScroller::SetAlwaysQuick(bool aVal)
{
   ElImScroller::SetAlwaysQuick(aVal);
   for (int aK=0 ; aK<int(_subs.size()) ; aK++)
   {
       _subs[aK]->SetAlwaysQuick(aVal);
   }
}

void  ElPyramScroller::SetAlwaysQuickInZoom(bool aVal)
{
   ElImScroller::SetAlwaysQuickInZoom(aVal);
   for (int aK=0 ; aK<int(_subs.size()) ; aK++)
   {
       _subs[aK]->SetAlwaysQuickInZoom(aVal);
   }
}

void  ElPyramScroller::SetAlwaysQuickInZoom()
{
   ElImScroller::SetAlwaysQuickInZoom();
   for (int aK=0 ; aK<int(_subs.size()) ; aK++)
   {
       _subs[aK]->SetAlwaysQuickInZoom();
   }
}

void  ElPyramScroller::SetAlwaysQuick()
{
   ElImScroller::SetAlwaysQuick();
   for (int aK=0 ; aK<int(_subs.size()) ; aK++)
   {
       _subs[aK]->SetAlwaysQuick();
   }
}


/****************************************************************/
/*                                                              */
/*********************************************************************/
/*                                                                   */
/*  ImageStd<Scroller>                                               */
/*                                                                   */
/*********************************************************************/






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
