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





/*****************************************************************************/
/*****************************************************************************/
/***                                                                       ***/
/***                                                                       ***/
/***                 BiScroller                                            ***/
/***                                                                       ***/
/***                                                                       ***/
/*****************************************************************************/
/*****************************************************************************/


bool BiScroller::Im1Act() const { return mIm1Act;}
bool BiScroller::Im2Act() const { return mIm2Act;}

void BiScroller::SetImAct(bool Im1Act,bool Im2Act)
{
    mIm1Act = Im1Act;
    mIm2Act = Im2Act;
}

ElImScroller & BiScroller::TheScrAct()
{
   if (Im1Act())
   {
       ELISE_ASSERT(!Im2Act(),"Both Im Act in TheScrAct");
       return Scr1();
   }

   ELISE_ASSERT(Im2Act(),"No Im Act in TheScrAct");
   return Scr2();
}

Pt2di BiScroller::SzIn() 
{
   return TheScrAct().SzIn();
}



ElImScroller &  BiScroller::TheFirstScrAct()
{
   if (Im1Act())
       return Scr1();

   ELISE_ASSERT(Im2Act(),"No Im Act in TheFirstScrAct");
   return Scr2();
}





BiScroller * BiScroller::LutColoredBiscroller
             (
                     VideoWin_Visu_ElImScr &aVisu,
                     const std::string &    NameGray,
                     const std::string &    NameCol,
                     Elise_colour *         Cols,
                     INT                    NbCol
             )
{
    ElPyramScroller * aGrayPyram = ElImScroller::StdPyramide(aVisu,NameGray);
    ElPyramScroller *  aColPyram = ElImScroller::StdPyramide(aVisu,NameCol,0,true,true);

    Tjs_El_User.ElAssert
    (
        aGrayPyram!=0,
        EEM0<< "Cannot Open Pyram for " << NameGray.c_str()
    );
    Tjs_El_User.ElAssert
    (
        aColPyram!=0,
        EEM0<< "Cannot Open Pyram for " << NameCol.c_str()
    );

    aColPyram->SetAlwaysQuick();
    return new BiScroller 
               (
                   aVisu,
                   aGrayPyram,
                   //aColPyram,
                   0,
                   FusLut,
                   Cols,NbCol,NameCol
                );
}

BiScroller * BiScroller::RG1B2Biscroller
             (
                     VideoWin_Visu_ElImScr &aVisu,
                     const std::string &    NameGray,
                     const std::string &    NameCol
             )
{
    ElPyramScroller * aGrayPyram = ElImScroller::StdPyramide(aVisu,NameGray);
    ElPyramScroller *  aColPyram = ElImScroller::StdPyramide(aVisu,NameCol,0,true,true);


    return new BiScroller 
               (
                   aVisu,
                   aGrayPyram,
                   aColPyram,
                   FusRG1B2,
                   0,0,""
                );
}

BiScroller  * BiScroller::MasqBiscroller
               (
                     VideoWin_Visu_ElImScr &aVisu,
                     const std::string &   NameGray,
                     const std::string &   NameMasq
                )
{
    ElPyramScroller * aGrayPyram = ElImScroller::StdPyramide(aVisu,NameGray);


    return new BiScroller 
               (
                   aVisu,
                   aGrayPyram,
                   0,
                   eModeMasq,
                   0,0,NameMasq
                );
}


BiScroller::BiScroller
( 
       VideoWin_Visu_ElImScr & aVisu,
       ElImScroller * ScrGray,    
       ElImScroller * ScrCol,    
       ModeFusion     aModeFusion,
       Elise_colour * Cols,
       INT            NbCol,
       const std::string &   mNameMasq

)   :
    ElImScroller (aVisu,1,Inf(ScrGray->SzU(),ScrCol?ScrCol->SzU():ScrGray->SzU()),1.0),


    mMemGray     (1,aVisu.SzW()),
    mScrGray     ( ScrGray),
    mMemCol      (1,aVisu.SzW()),
    mScrCol      (ScrCol),

    mModeFusion  (aModeFusion),

    mVideoVisu   (aVisu),

    mImGray      (mMemGray.Images()[0]),
    mDataGray    (mImGray.data()),
    mImCol       (mMemCol.Images()[0]),
    mDataCol     (mImCol.data()),

    mTabRGB      (3,NbCol,256),
    mDataTabRGB  (mTabRGB.data()),

    mImR         (aVisu.SzW().x,1),
    mDataImR     (mImR.data()[0]),
    mImG         (aVisu.SzW().x,1),
    mDataImG     (mImG.data()[0]),
    mImB         (aVisu.SzW().x,1),
    mDataImB     (mImB.data()[0]),
    mTr          (0.0,0.0),
    mIm1Act      (true),
    mIm2Act      (mScrCol!=0),
    mImMasq      (1,1)

{
   mSubScrolls.push_back(mScrGray);
   if (mScrCol)
       mSubScrolls.push_back(mScrCol);



    for (INT iCol = 0; iCol<NbCol ; iCol++)
        for (INT iGray=0; iGray < 256; iGray ++)
        {
             Elise_colour CGray = Elise_colour::gray(iGray/255.0);
             Elise_colour CCol = Cols[iCol];
             Elise_colour ColMixed = som_pond(CCol,0.3,CGray);
             if (iCol == 0) 
                ColMixed = CGray;
             SetRGBEntry(iGray, iCol, ColMixed);
        }


   if (mNameMasq!="")
   {
      Tiff_Im aFM = Tiff_Im::UnivConvStd(mNameMasq);
      Pt2di aSz = aFM.sz();
      mImMasq=Im2D_U_INT1 (aSz.x,aSz.y);

      Fonc_Num aFIn = aFM.in(); 
      if (aModeFusion==eModeMasq)
      {
         aFIn = (aFM.in()!=0);
      }
         
      ELISE_COPY(aFM.all_pts(),aFIn,mImMasq.out());
   }
/*
    REAL DistCible = 60.0;


    DistCible /= 255.0;
    for (INT iCol = 0; iCol<NbCol ; iCol++)
    {
        Elise_colour CCol = Cols[iCol];
        Elise_colour  aGray = Elise_colour::gray(CCol.GrayVal());


        REAL dist = ElMax(1e-10,aGray.eucl_dist(CCol));

        Elise_colour vAtDit = aGray +  (DistCible/dist) * (CCol-aGray);


        Elise_colour v0 = vAtDit - Elise_colour::gray(vAtDit.MinRGB());
        Elise_colour v1 = vAtDit + Elise_colour::gray(1.0-vAtDit.MaxRGB());

        for (INT iGray=0; iGray < 256; iGray ++)
        {
             REAL GrayLevel = iGray/255.0;
             Elise_colour ColMixed = som_pond(v0,1-GrayLevel,v1);
             if (iCol == 0) 
                ColMixed =  Elise_colour::gray(GrayLevel);
             SetRGBEntry(iGray, iCol, ColMixed);
        }
    }
*/
}

void BiScroller::LutMakeOneLine
     (
           U_INT1 * aDataImR,
           U_INT1 * aDataImG,
           U_INT1 * aDataImB,
           INT      aY,
           INT      anX0,
           INT      anX1,
           U_INT1 * aLineGr,
           U_INT1 * aLineCol
     )
{
/*
    U_INT1 * aRGB;
    for (INT x= anX0; x<anX1 ; x++)
    {
        aRGB = RGBEntry(aLineGr[x],aLineCol[x]);
        aDataImR[x] = aRGB[eIndR];
        aDataImG[x] = aRGB[eIndG];
        aDataImB[x] = aRGB[eIndB];
    }
*/
   TIm2D<U_INT1,INT> aTM(mImMasq);
   Pt2di aP(anX0,aY);

   Pt2dr aQ0 = mScrGray->to_user(Pt2dr(aP));
   Pt2dr aQ1 = mScrGray->to_user(Pt2dr(aP)+Pt2dr(1,0));
   double aDx = aQ1.x -aQ0.x;
   double aXQ0 = aQ0.x;
   int aYQ0 = round_ni(aQ0.y);

   for (; aP.x<anX1 ; aP.x++)
   {
       int  indCoul  = aTM.get(Pt2di(round_ni(aXQ0),aYQ0),0);
       int aGray = aLineGr[aP.x];

       U_INT1 * aRGB = RGBEntry(aGray,indCoul);

        aDataImR[aP.x] = aRGB[eIndR];
        aDataImG[aP.x] = aRGB[eIndG];
        aDataImB[aP.x] = aRGB[eIndB];
       // aDataImR[aP.x] = inM?aLR[aCoul]:aCoul ;
       // aDataImG[aP.x] = inM?aLG[aCoul]:aCoul ;
       // aDataImB[aP.x] = inM?aLB[aCoul]:aCoul ;
        aXQ0 +=aDx;
   }
}


void BiScroller::RG1B2MakeOneLine
     (
           U_INT1 * aDataImR,
           U_INT1 * aDataImG,
           U_INT1 * aDataImB,
           INT      aY,
           INT      anX0,
           INT      anX1,
           U_INT1 * aLine1,
           U_INT1 * aLine2
     )
{
    if (Im1Act() && Im2Act() )
    {
       for (INT x= anX0; x<anX1 ; x++)
       {
           aDataImR[x] = aLine1[x];
           aDataImG[x] = (aLine1[x]+aLine2[x])/2;
           aDataImB[x] = aLine2[x];
       }
    }
    else if (Im1Act() || Im2Act() )
    {
       U_INT1 * aLine = Im1Act() ? aLine1 : aLine2;
       for (INT x= anX0; x<anX1 ; x++)
           aDataImR[x] = aDataImG[x] = aDataImB[x] = aLine[x];
    }
}

void BiScroller::MasqMakeOneLine
     (
           U_INT1 * aDataImR,
           U_INT1 * aDataImG,
           U_INT1 * aDataImB,
           INT      aY,
           INT      anX0,
           INT      anX1,
           U_INT1 * aLine1,
           U_INT1 * aLine2
     )
{
   static U_INT1 * aLR=0;
   static U_INT1 * aLG=0;
   static U_INT1 * aLB=0;
   if (aLR==0)
   {
       aLR = new U_INT1 [256] ;
       aLG = new U_INT1 [256] ;
       aLB = new U_INT1 [256] ;

        // Pt3dr aP0(128,0,0);
        // Pt3dr aP1(255,128,255);
        // Pt3dr aP0(0,0,128);
        // Pt3dr aP1(255,128,255);
        Pt3dr aP0(0,80,0);
        Pt3dr aP1(255,255,128);

        for (int aK=0 ; aK< 256 ; aK++)
        {
          double aPds0 = 1- aK /255.0;
          double aPds1 = 1-aPds0;
          Pt3dr aCol = aP0 *aPds0 +  aP1*aPds1;
          aLR[aK] = round_ni(aCol.x);
          aLG[aK] = round_ni(aCol.y);
          aLB[aK] = round_ni(aCol.z);
        }
   }


   TIm2D<U_INT1,INT> aTM(mImMasq);
   Pt2di aP(anX0,aY);

   Pt2dr aQ0 = mScrGray->to_user(Pt2dr(aP));
   Pt2dr aQ1 = mScrGray->to_user(Pt2dr(aP)+Pt2dr(1,0));
   double aDx = aQ1.x -aQ0.x;
   double aXQ0 = aQ0.x;
   int aYQ0 = round_ni(aQ0.y);

   for (; aP.x<anX1 ; aP.x++)
   {
       bool inM = (aTM.get(Pt2di(round_ni(aXQ0),aYQ0),0)!=0);
       int aCoul = aLine1[aP.x];

       aDataImR[aP.x] = inM?aLR[aCoul]:aCoul ;
       aDataImG[aP.x] = inM?aLG[aCoul]:aCoul ;
       aDataImB[aP.x] = inM?aLB[aCoul]:aCoul ;
        aXQ0 +=aDx;
   }
}

Im2D_U_INT1 BiScroller::ImMasq() { return mImMasq;}

void BiScroller::MakeOneLine
     (
           U_INT1 * aDataImR,
           U_INT1 * aDataImG,
           U_INT1 * aDataImB,
           INT      aY,
           INT      anX0,
           INT      anX1,
           U_INT1 * aLine1,
           U_INT1 * aLine2
     )
{
     switch(mModeFusion)
     {
         case FusLut :
              LutMakeOneLine
              (
                  aDataImR,aDataImG,aDataImB,
                  aY,anX0,anX1,
                  aLine1,aLine2
              );
         break ;

         case FusRG1B2 :
              RG1B2MakeOneLine
              (
                  aDataImR,aDataImG,aDataImB,
                  aY,anX0,anX1,
                  aLine1,aLine2
              );
         break ;

         case eModeMasq :
              MasqMakeOneLine
              (
                  aDataImR,aDataImG,aDataImB,
                  aY,anX0,anX1,
                  aLine1,aLine2
              );
         break ;



         default :
              ELISE_ASSERT(false,"Unexpcted mModeFusion in BiScroller::MakeOneLine");
         break ;

     }
}



void BiScroller::LoadXImage(Pt2di p0W,Pt2di p1W,bool quick)
{

     mScrGray->SetSameGeom(*this);
     if (mScrCol)
         mScrCol->SetGeomTranslated(*this,mTr);

     if (quick)
     {
        TheFirstScrAct().LoadXImageInVisu(mVideoVisu,p0W,p1W,quick);
        return;
     }

     if (Im1Act())
        Scr1().LoadXImageInVisu(mMemGray,p0W,p1W,false);
     if (Im2Act())
        Scr2().LoadXImageInVisu(mMemCol,p0W,p1W,false);

     for (INT y=p0W.y; y<p1W.y ; y++)
     {
         MakeOneLine
         (
             mDataImR,mDataImG,mDataImB,
             y,p0W.x,p1W.x,
             mDataGray[y],mDataCol[y]
         );
          mVideoVisu.XIm_write_image_brute
          (
              Pt2di(p0W.x,0),Pt2di(p0W.x,y),Pt2di(p1W.x-p0W.x,1),
              mImR, mImG, mImB
          );
     }

}

void BiScroller::SetTranslate(Pt2dr aTr)
{
   mTr = aTr;
   LoadAndVisuIm();
}

Pt2dr BiScroller::CurTr() const
{
   return mTr;
}


void BiScroller::GraySetGamaCorrec(REAL aGama)
{
    mMemGray.SetGamaCorr(aGama);
}


void BiScroller::SetRGBEntry(INT Gray,INT IndCol,const Elise_colour & aCol)
{
    U_INT1 * aRGB = RGBEntry(Gray,IndCol);

    aRGB[eIndR] = ElMax(0,ElMin(255,round_ni(aCol.r() *255)));
    aRGB[eIndG] = ElMax(0,ElMin(255,round_ni(aCol.g() *255)));
    aRGB[eIndB] = ElMax(0,ElMin(255,round_ni(aCol.b() *255)));

}


ElImScroller & BiScroller::Scr1() { return *mScrGray;}
ElImScroller & BiScroller::Scr2() 
{ 
    ELISE_ASSERT(mScrCol!=0,"BiScroller::Scr2");
    return *mScrCol;
}

Im2D_U_INT1  BiScroller::Im1() {return mImGray;}
Im2D_U_INT1  BiScroller::Im2() {return mImCol;}


const ElSTDNS vector<ElImScroller *> & BiScroller::SubScrolls()
{
   return mSubScrolls;
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
