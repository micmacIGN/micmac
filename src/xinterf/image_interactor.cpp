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
    


/***************************************************/
/***************************************************/
/***                                             ***/
/***    EliseStdImageInteractor                  ***/
/***                                             ***/
/***************************************************/
/***************************************************/


    //=====================================
    //   EliseStdImageInteractor::Graber
    //=====================================

void  EliseStdImageInteractor::Graber::GUR_query_pointer(Clik aCl,bool aMvt)
{
     mESII.GQP(aCl,aMvt);
}

void  EliseStdImageInteractor::Graber::GUR_button_released(Clik aCl)
{
     mESII.GBR(aCl);
}

EliseStdImageInteractor::Graber::Graber(EliseStdImageInteractor & anESII) :
   mESII  (anESII)
{
}


    //=====================================
    //   EliseStdImageInteractor
    //=====================================


EliseStdImageInteractor::EliseStdImageInteractor
(
    Video_Win aW,
    ElImScroller & aScrol,
    INT            aButonGeom,
    INT            aButonZoomIn ,
    INT            aButonZoomOut,
    cClikInterceptor *   aClikInterceptor
)  :
   mW           (aW),
   mScr         (aScrol),
   mButtonGeom  (aButonGeom),
   mButtonZoomIn (aButonZoomIn),
   mButtonZoomOut(aButonZoomOut),
   mScaleMode   (false),
   mP0Grab      (-1,-1),
   mScale0Grab  (1.0),
   mSpeedTranslate (1.0),
   mYDoubleScale   (100.0),
   mScaleMax       (100.0),
   mScaleMin       (0.02),
   mRasterQuickZoom      (true),
   mGeoGraber            (NULL),
   mModeReplicationPixel (false),
   mClikIntercept        (aClikInterceptor)
{
	// NO_WARN
	mGeoGraber = new Graber(*this);

    aScrol.set_max_init();
    // mScr.set(Pt2dr(0,0),0.1);
}
   
std::vector<Pt2dr> EliseStdImageInteractor::GetPolyg
                   (
                       INT aCoulSom,
                       INT aCoulCur,
                       const std::vector<Pt2dr> & aResInit
                   )
{
   mCoulSomPolygCur = aCoulSom;
   mCoulEdgePolygCur = aCoulCur;

   mPolygCur=aResInit;

   while (1)
   {
      Clik  aCl = clik_press();
      if (aCl.shifted())
      {
           std::vector<Pt2dr>  aRes = mPolygCur;
           mPolygCur.clear();
           return aRes;
      }
      if (aCl.controled())
      {
          if (!mPolygCur.empty())
             mPolygCur.pop_back();

          mScr.LoadAndVisuIm(mModeReplicationPixel);
      }
      else
      {
         Pt2dr aPU = W2U(aCl._pt);
         mPolygCur.push_back(aPU);
      }
      ShowVect();
   }
}


std::vector<Pt2dr> EliseStdImageInteractor::GetPolyg(INT aCS,INT aCC)
{
   std::vector<Pt2dr> aRes;
   return GetPolyg(aCS,aCC,aRes);
}

std::vector<Pt2dr> EliseStdImageInteractor::GetPolyg
                   (INT aCS,INT aCC,const Pt2dr & aP0)
{
   std::vector<Pt2dr> aRes;
   aRes.push_back(aP0);
   return GetPolyg(aCS,aCC,aRes);
}




void EliseStdImageInteractor::ShowVect()
{
   INT aNbPts = (INT) mPolygCur.size();

   Line_St aLstPt(mW.pdisc()(mCoulSomPolygCur),1.0);
   Line_St aLstSeg(mW.pdisc()(mCoulEdgePolygCur),2.0);

   for (INT aK=0 ; aK<aNbPts ; aK++)
   {
      mW.draw_circle_loc(U2W(mPolygCur[aK]),2.0,aLstPt);
      if (aK != (aNbPts-1))
         mW.draw_seg(U2W(mPolygCur[aK]),U2W(mPolygCur[aK+1]),aLstSeg);
   }
}


void EliseStdImageInteractor::SetModeReplicationPixel(bool aVal)
{
   mModeReplicationPixel = aVal;
   mScr.SetAlwaysQuick(aVal);
}
void EliseStdImageInteractor::SetModeReplicationPixel()
{
   SetModeReplicationPixel(true);
}

EliseStdImageInteractor::~EliseStdImageInteractor(){
	if ( mGeoGraber!=NULL ) delete mGeoGraber;
}


void EliseStdImageInteractor::OnEndTranslate(Clik)
{
}

void EliseStdImageInteractor::OnEndScale(Clik)
{
}

void EliseStdImageInteractor::Refresh()
{
   mScr.LoadAndVisuIm(mModeReplicationPixel);
   ShowVect();
}

void EliseStdImageInteractor::GBR(Clik aCl)
{
   if (mScaleMode)
   {
      Refresh();
      // mScr.LoadAndVisuIm(mModeReplicationPixel);
      // ShowVect();
      OnEndScale(aCl);
   }
   else
   {
      Refresh();
      OnEndTranslate(aCl);
   }
}

bool EliseStdImageInteractor::ScaleMode() const
{
    return mScaleMode;
}


Clik  EliseStdImageInteractor::clik_press()
{

    while (true)
    {
        Clik cl1   =  mW.disp().clik_press();
        if (mClikIntercept && mClikIntercept->InterceptClik(cl1))
        {
        }
        else
        {
	     EliseStdImageInteractor * anI =0;
	     if (cl1._w == mW)
                 anI = this;
	     else
                 anI = cl1._w.Interactor();

	     if (anI == 0)
                return cl1;
	      if ((cl1._b== mButtonZoomIn) || (cl1._b== mButtonZoomOut))
	      {
	        double aFact = 1.2;
      	         mScr.SetScArroundPW
                 (
                     mP0Grab,
                      anI->mScr.sc() * ((cl1._b== mButtonZoomIn) ? aFact : 1/aFact),
                     false // mRasterQuickZoom
                 );
                 // mScr.LoadAndVisuIm(mModeReplicationPixel);
                  ShowVect();
                  OnEndScale(cl1);
                 // anI->mScale0Grab = anI->mScr.sc();
	      }
	      else
	      {


                  if (cl1._b != anI->mButtonGeom)
                     return cl1;

                  anI->mScaleMode =  cl1.shifted();
                  anI->mP0Grab    = cl1._pt;
                  anI->mLastPGrab    = cl1._pt;
                  anI->mScale0Grab = anI->mScr.sc();
                  anI->mW.grab( *( anI->mGeoGraber ) );
	     }
         }
    }


    ELISE_ASSERT(false,"Should Not Be Here (EliseStdImageInteractor::clik_press)");
    return mW.disp().clik_press();
}

Video_Win EliseStdImageInteractor::WinEch1()
{
      REAL  aSc = mScr.sc();
      return mW.chc(Pt2dr(0,0),Pt2dr(aSc,aSc));

}

          // Uniquemement si image de Dim1
Im2D_U_INT1  EliseStdImageInteractor::ImEch1(bool VisuSim)
{
    REAL  aSc = mScr.sc();
    Pt2di aTr = Pt2di(mScr.tr()); 



    Pt2di aSzIm = Pt2di(Pt2dr(mW.sz())/aSc);

    Im2D_U_INT1 anIm(aSzIm.x,aSzIm.y);

    ELISE_COPY
    (
        anIm.all_pts(),
        trans(mScr.in(),aTr),
        anIm.out() | (VisuSim ? WinEch1().ogray() : Output::onul())
    );

     return anIm;
}


void EliseStdImageInteractor::GQP(Clik cl,bool)
{
	 if (mScaleMode )
	 {
		 REAL scy = mP0Grab.y- cl._pt.y;
		 scy /= -mYDoubleScale;
		 scy = mScale0Grab *pow(2.0,scy);
		 scy = ElMin(ElMax(scy,mScaleMin),mScaleMax);

if (0 && MPD_MM()){
    std::cout << "EliseStdImageInteractor::GQP " << mRasterQuickZoom << "\n";
    mRasterQuickZoom = false;
}

      	        mScr.SetScArroundPW
                (
                    mP0Grab,
                    scy,
                    mRasterQuickZoom
                );
	 }
	 else
	 {
             mScr.SetDTrW(Pt2di((mLastPGrab-cl._pt) *mSpeedTranslate));
	 }
         ShowVect();
         mLastPGrab = cl._pt;
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
