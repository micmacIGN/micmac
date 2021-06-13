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
#include "general/all.h"
#include "MICMAC.h"


namespace NS_ParamMICMAC
{
#if ELISE_X11

class cMicMacWindowIm
{
     public :
        cMicMacWindowIm
        (
            cAppliMICMAC &,
            Video_Win * aPrec,
            Video_Win *aWXFracPrec,
            cPriseDeVue & aPDV
        );
        Video_Win  *WIm() {return mWIm;}
        void Load();
        void ShowFrac();
        void ShowProj(Pt2dr );
        Pt2dr  Proj(Pt3dr aP)
        {
             static double aPx[2] = {0,0};
             aPx[0] = aP.z;
             return mPDV.Geom().CurObj2Im(Pt2dr(aP.x,aP.y),aPx);
        }
     // private :
        cAppliMICMAC & mAp;
        cPriseDeVue &  mPDV;
        cLoadedImage * mLIm;
        Pt2di           mSzIm;
        Video_Win      *mWIm;
        Pt2di          mSzWZ;
        int            mZoom;
        Video_Win      *mWImZ;
        Im2D_REAL8     mIm;

        int            mNbF;
        Im2D_REAL8     mHFrac;
        Im2D_REAL8     mXFrac;
        Im2D_REAL8     mYFrac;
        int            mZFrac;
        Video_Win *    mWXFrac;
        Video_Win *    mWYFrac;
        Video_Win *    mWHFrac;

};

void cMicMacWindowIm::ShowProj(Pt2dr aPIm)
{
    std::cout << "Pt = " << aPIm << "\n";
    mWIm->draw_circle_loc(aPIm,2.0,mWIm->pdisc()(P8COL::green));

    Pt2di aP0= Pt2di(aPIm-Pt2dr(mSzWZ)/2.0);
    Pt2di aP1= aP0+mSzWZ;
    mWImZ = mWImZ->PtrChc(Pt2dr(aP0),Pt2dr(mZoom,mZoom));
    
    double aVMax,aVMin;
    ELISE_COPY
    (
       rectangle(aP0,aP1),
       mLIm->FIm(),
       mIm.out()|VMax(aVMax)|VMin(aVMin)  
    );
    ELISE_COPY
    (
       rectangle(aP0,aP1),
       (mLIm->FIm()-aVMin) * (255.0/(aVMax-aVMin)),
       mWImZ->ogray()
    );
    mWImZ->draw_circle_abs(aPIm,2.0,mWImZ->pdisc()(P8COL::green));

    mWImZ->draw_circle_abs(Pt2dr(2,2),2.0,mWImZ->pdisc()(P8COL::red));
}


void cMicMacWindowIm::ShowFrac()
{
   double aVMax,aVmin;
   ELISE_COPY
   (
       mHFrac.all_pts(),
       mHFrac.in(),
       VMax(aVMax) | VMin(aVmin)
   );
   if (mWHFrac)
   {
       ELISE_COPY
       (
          mHFrac.all_pts(),
          P8COL::red,
          mWHFrac->odisc()
       );
       ELISE_COPY
       (
          mHFrac.all_pts(),
          mHFrac.in() * (255/aVMax),
          mWHFrac->ogray()
       );
       ELISE_COPY
       (
          mXFrac.all_pts(),
          Min(mXFrac.in(),1-mXFrac.in()) * 500,
          mWXFrac->ogray()
       );
       ELISE_COPY
       (
          mYFrac.all_pts(),
          Min(mYFrac.in(),1-mYFrac.in()) * 500,
          mWYFrac->ogray()
       );
   }
}



cMicMacWindowIm::cMicMacWindowIm
(
      cAppliMICMAC & anAp,
      Video_Win * aPrec,
      Video_Win * aWXFracPrec,
      cPriseDeVue & aPDV
) :
   mAp   (anAp),
   mPDV  (aPDV),
   mSzIm (mPDV.LoadedIm().SzIm()),
   mWIm  (0),
   mSzWZ   (10,10),
   mZoom   (15),
   mWImZ   (0),
   mIm     (mSzIm.x,mSzIm.y),
   mNbF    (mAp.PtrVI()->NbDiscHistoPartieFrac().Val()),
   mHFrac  (ElMax(1,mNbF),ElMax(1,mNbF),0.0),
   mXFrac  (1,1),
   mYFrac  (1,1),
   mZFrac  (5),
   mWXFrac (0),
   mWYFrac (0),
   mWHFrac (0)
{
   if (aPrec)
   {
       mWIm =  new Video_Win(*aPrec,Video_Win::eDroiteH,mSzIm);
       mWImZ =  new Video_Win(*mWIm,Video_Win::eBasG,mSzWZ*mZoom);
   }

   if ((mNbF>0) && mWIm)
   {
      mXFrac = Im2D_REAL8(mSzIm.x,mSzIm.y);
      mYFrac = Im2D_REAL8(mSzIm.x,mSzIm.y);
      mWXFrac =  aWXFracPrec                                               ?
                   new Video_Win(*aWXFracPrec,Video_Win::eDroiteH,mSzIm)   :
                   Video_Win::PtrWStd(mSzIm)                               ;

      if (aWXFracPrec==0)
         mWXFrac->set_title("Partie fractionnaire");
      mWYFrac = new Video_Win(*mWXFrac,Video_Win::eBasG,mSzIm);
      mWHFrac = new Video_Win(*mWYFrac,Video_Win::eBasG,Pt2di(mNbF,mNbF)*mZFrac);
      mWHFrac = mWHFrac->PtrChc(Pt2dr(0,0),Pt2dr(mZFrac,mZFrac),true);
   }
}
   
void  cMicMacWindowIm::Load()
{
    mLIm  = & mPDV.LoadedIm();
    mSzIm = mLIm->SzIm();
    double aVMax,aVMin;
    ELISE_COPY
    (
       mIm.all_pts(),
       mLIm->FIm(),
       mIm.out()|VMax(aVMax)|VMin(aVMin)  
    );
    if (mWIm)
    {
        ELISE_COPY
        (
           mIm.all_pts(),
           (mLIm->FIm()-aVMin) * (255.0/(aVMax-aVMin)),
           mWIm->ogray()
        );
    }
    ELISE_COPY(mHFrac.all_pts(),0,mHFrac.out());
}
/*
*/

/*******************************************/
/*                                         */
/*                                         */
/*                                         */
/*******************************************/

class cMicMacVisu
{
     public :
         cMicMacVisu(cAppliMICMAC &,Box2di aBoxIn);
         void Load(Box2di aBoxIn);

         bool ExecOneClique();

         void ShowProjs(Pt2di aPt);

     private :
          Pt3dr Pdisc2Pter(Pt3dr aP);
          void  InitGeomPtI(Pt2di aP);
          int SimulZFrac(Pt2di aP);

          cAppliMICMAC & mAp;
          const cInterfaceVisualisation & mIV;
          bool           mVTI;
          Pt2di          mSzW;
          Video_Win *    mWTerr;

          int            mCurPxI[theDimPxMax];
          double         mCurPxR[theDimPxMax];
          Pt2di          mCurPtI;

          Pt2dr          mPTer;
          
          std::vector<cMicMacWindowIm *> mWIms;
          int  mNbF;
          Box2di mBoxIn;
          

          bool              mSimFrac;
          double            mCoutFrac;
          double            mPlanX;
          double            mPlanY;
          double            mPlanC;
          double            ZOfPlan(Pt2dr aP);
          cGenSysSurResol * mSysResolv;
          Video_Win *       mWSimFrac;
};


double   cMicMacVisu::ZOfPlan(Pt2dr aP)
{
    return    mPlanX * aP.x 
            + mPlanY * aP.y  
            + mPlanC;
}

cMicMacVisu::cMicMacVisu(cAppliMICMAC & anAp,Box2di aBoxIn) :
   mAp    (anAp),
   mIV    (*mAp.mPtrIV),
   mVTI   (mIV.VisuTerrainIm().Val()),
   mSzW   (aBoxIn.sz()),
   mWTerr ( mVTI ?
             Video_Win::PtrWStd(mSzW,mIV.ZoomTerr().Val()) :
             0
          ),
   mNbF    (mAp.PtrVI()->NbDiscHistoPartieFrac().Val()),
   mSimFrac   (mIV.SimulFrac().IsInit()),
   mSysResolv (0),
   mWSimFrac  (0)
{
    if (mVTI)
    {
          int aZ = mIV.ZoomTerr().Val();
          mWTerr = mWTerr->PtrChc(Pt2dr(0,0),Pt2dr(aZ,aZ),true);
    }
}

void cMicMacVisu::Load(Box2di aBoxIn)
{
   mBoxIn = aBoxIn;
   if (mVTI)
   {
     for (int aK=0 ; aK<1 ; aK++)
     {
        Im2D_INT2 aIm = mAp.LoadTer()->KthNap(aK).mPxInit;
        if (mSimFrac)
        {
           ELISE_COPY
           (
             aIm.all_pts(),
             aIm.in() *50,
             mWTerr->ocirc()
           );
        }
        else
        {
           int aVMax,aVMin;
           ELISE_COPY
           (
             aIm.all_pts(),
             aIm.in(),
             VMax(aVMax) | VMin(aVMin)
           );
           ELISE_COPY
           (
             aIm.all_pts(),
             (aIm.in() -aVMin) * (255.0 /ElMax(1,aVMax-aVMin)),
             mWTerr->ogray()
           );
        }
     }
   }

   
   {
   int aK=0;
   for 
   (
         tCsteIterPDV itFI=mAp.mPDVBoxGlobAct.begin(); 
         itFI!=mAp.mPDVBoxGlobAct.end(); 
         itFI++
   )
   {
         Video_Win * aW = mWIms.size() ? mWIms.back()->WIm() : mWTerr;
         Video_Win * aWXF = mWIms.size() ? mWIms.back()->mWXFrac : 0;
         cMicMacWindowIm  * aMMW=0;
         if (int(mWIms.size())<=aK) 
         {
            aMMW = new cMicMacWindowIm (mAp,aW,aWXF,**itFI);
            mWIms.push_back(aMMW);
         }
         else
         {
              aMMW = mWIms[aK];
         }
         aMMW->Load();
         aK++;
   }
   }

   if (mSimFrac)
   {
      mCoutFrac = mIV.SimulFrac().Val().CoutFrac();
      delete mSysResolv;
      // mSysResolv = new L2SysSurResol(3);
      mSysResolv = new SystLinSurResolu(3,10000);

      int aPer=10;
      Pt2di aSz = mAp.LoadTer()->KthNap(0).mPxInit.sz();
      INT2 ** aData = mAp.LoadTer()->KthNap(0).mPxInit.data();
      for (int aX0=0 ; aX0< aSz.x ; aX0+=aPer)
      {
         for (int aY0=0 ; aY0< aSz.y ; aY0+=aPer)
         {
              int aX1 = ElMin(aX0+aPer,aSz.x);
              int aY1 = ElMin(aY0+aPer,aSz.x);
              double aSx =0;
              double aSy =0;
              double aSz =0;
              double aS1 =0;
              Pt2di aP;
              for (aP.x = aX0 ; aP.x < aX1 ; aP.x++)
              {
                  for (aP.y = aY0 ; aP.y < aY1 ; aP.y++)
                  {
                     if (mAp.LoadTer()->IsInMasq(aP))
                     {
                         aS1++;
                         aSx += aP.x;
                         aSy += aP.y;
                         aSz += aData[aP.y][aP.x];
                     }
                  }
              }
              mSysResolv->GSSR_Add_EqFitPlan(aSx/aS1,aSy/aS1,aSz/aS1,aS1);
         }
      }
      mSysResolv->GSSR_SolveEqFitPlan(mPlanX,mPlanY,mPlanC);
      if ((mWSimFrac ==0) && (mWTerr!=0))
      {
          mWSimFrac = new Video_Win(*mWTerr,Video_Win::eBasG,mSzW);
          ELISE_COPY(mWSimFrac->all_pts(),P8COL::red,mWSimFrac->odisc());
      }

      Im2D_INT2  aImSim(aSz.x,aSz.y);
      Pt2di aP;
      for (aP.y =0 ; aP.y< aSz.y ; aP.y++)
      {
         for (aP.x =0 ; aP.x< aSz.x ; aP.x++)
         {
              aImSim.data()[aP.y][aP.x] = SimulZFrac(aP);
         }
         ELISE_COPY
         (
            rectangle(Pt2di(0,aP.y),Pt2di(aSz.x,aP.y+1)),
            aImSim.in() * 50,
            mWSimFrac->ocirc()
         );
      }
   }

   if (mNbF > 0)
   {
std::cout << "11111111 \n";
       Pt2di aP;
       Pt2di aSz = mAp.LoadTer()->KthNap(0).mPxInit.sz();

std::cout << "22222222 \n";
       for (aP.y=0 ; aP.y <aSz.y ; aP.y++)
       {
std::cout << "AAAA.y=" <<aP.y << "\n";
bool aBug = (aP.y==164);
           for (aP.x=0 ; aP.x <aSz.x ; aP.x++)
           {
if (aBug) std::cout << "AAAA.xxxx=" << aP.x << "\n";
                InitGeomPtI(aP);
                for  (int aK = 0 ; aK<int(mWIms.size()) ; aK++)
                {
if (aBug) std::cout << "-1-\n";
                    cMicMacWindowIm & aMMV  = *mWIms[aK];
                    Pt2dr aPIm = aMMV.mPDV.Geom().CurObj2Im(mPTer,mCurPxR);
                    double aRX = aPIm.x;
                    double aRY = aPIm.y;

if (aBug) std::cout << "-2-\n";
                    double aFracX = aRX-int(aRX);
                    double aFracY = aRY-int(aRY);

                    int aIFX = ElMax(0,ElMin(mNbF,int(aFracX*mNbF)));
                    int aIFY = ElMax(0,ElMin(mNbF,int(aFracY*mNbF)));

if (aBug) std::cout << "-3-\n";
                    aMMV.mHFrac.data()[aIFY][aIFX]++;
if (aBug) std::cout << "-4- " << aP <<  " " << aMMV.mXFrac.sz() << " " << aSz << "\n";
                    aMMV.mXFrac.data()[aP.y][aP.x] = aFracX;
if (aBug) std::cout << "-5-\n";
                    aMMV.mYFrac.data()[aP.y][aP.x] = aFracY;
if (aBug) std::cout << "-6-\n";
                }
           }
       }
std::cout << "33333333333 \n";
       for  (int aK = 0 ; aK<int(mWIms.size()) ; aK++)
           mWIms[aK]->ShowFrac();
   }

   
}

void  cMicMacVisu::InitGeomPtI(Pt2di aP)
{
   mCurPtI = aP;
   for (int aK=0 ; aK<mAp.DimPx() ; aK++)
   {
      Im2D_INT2 aIm = mAp.LoadTer()->KthNap(aK).mPxInit;
      mAp.mCurPxI[aK] = 
          mCurPxI[aK]= aIm.data()[mCurPtI.y][mCurPtI.x];

   } 
   mAp.mGeomDFPx.PxDisc2PxReel(mCurPxR,mCurPxI);
   mPTer =  mAp.mGeomDFPx.DiscToR2(mCurPtI);
}


Pt3dr cMicMacVisu::Pdisc2Pter(Pt3dr aP)
{
   double aPxI[theDimPxMax];
   for (int aK=0 ; aK<mAp.DimPx() ; aK++)
   {
          aPxI[aK]= (aK==0) ? aP.z : 0;
   } 
   double aPxRes[theDimPxMax];
   mAp.mGeomDFPx.PxDisc2PxReel(aPxRes,aPxI);
   Pt2dr aPTer =  mAp.mGeomDFPx.RDiscToR2(Pt2dr(aP.x,aP.y));
   Pt3dr aRes(aPTer.x,aPTer.y,aPxRes[0]);

   return aRes;
}

double PartieNonFrac(double aV)
{
    aV = aV -round_down(aV);
    return  ElMax(0.5-aV,aV-0.5);
}
double PartieNonFrac(Pt2dr aP)
{
   return ElSquare(PartieNonFrac(aP.x)) + ElSquare(PartieNonFrac(aP.y));
}

int cMicMacVisu::SimulZFrac(Pt2di aP)
{
    double aZR = ZOfPlan(Pt2dr(aP));
    Pt3dr aPVrai = Pdisc2Pter(Pt3dr(aP.x,aP.y,aZR));

    int  aZI = round_ni(aZR);
    int aDZ = 2;

    double aEcartMin  = 1e9;
    int    aZMin      = -111111;

// std::cout <<  " A  :cMicMacVisu::SimulZFrac\n"; 
    for (int aZ = aZI-aDZ ; aZ<=aZI+aDZ  ; aZ++)
    {
        Pt3dr aPCur = Pdisc2Pter(Pt3dr(aP.x,aP.y,aZ));
        double anEcart = 0.0;
        for  (int aK = 0 ; aK<int(mWIms.size()) ; aK++)
        {
// std::cout << "Z " << aZ << " K " << aK << "\n";
             cMicMacWindowIm & aMMV  = *mWIms[aK];
             Pt2dr aPVraIm = aMMV.Proj(aPVrai);
             Pt2dr aCurIm  = aMMV.Proj(aPCur);
             anEcart += square_euclid(aPVraIm-aCurIm) + mCoutFrac *PartieNonFrac(aCurIm) ; 
// std::cout << "END   ----- Z " << aZ << " K " << aK << "\n";
        }
        if (aEcartMin > anEcart)
        {
           aEcartMin = anEcart;
           aZMin = aZ;
        }
    }
// std::cout <<  " B  :cMicMacVisu::SimulZFrac\n"; 
    return aZMin;
}


void cMicMacVisu::ShowProjs(Pt2di aPt)
{
   InitGeomPtI(aPt);
   mWTerr->draw_circle_loc(Pt2dr(mCurPtI),2.0, mWTerr->pdisc()(P8COL::red));


   int aZI = mAp.LoadTer()->KthNap(0).mPxInit.data()[aPt.y][aPt.x] ;
   std::cout << "Z = " <<  aZI << "\n";;
   if (mSimFrac)
      std::cout << "Z Interp = " << ZOfPlan(Pt2dr(aPt)) <<  "\n";

   std::cout << "PTER " << Pdisc2Pter(Pt3dr(aPt.x,aPt.y,aZI)) << "\n";

   for (int aK=0 ; aK<mAp.DimPx() ; aK++)
       std::cout << "Px[" << aK << "]= "<<  mCurPxR[aK] << " ;";
   std::cout << "\n";

   for  (int aK = 0 ; aK<int(mWIms.size()) ; aK++)
   {
       cMicMacWindowIm & aMMV  = *mWIms[aK];
       Pt2dr aPIm = aMMV.mPDV.Geom().CurObj2Im(mPTer,mCurPxR);
       aMMV.ShowProj(aPIm);
   }
}

bool cMicMacVisu::ExecOneClique()
{
   if (! mVTI) 
      return false;
 
   Clik aCl = mWTerr->disp().clik_press();

   if (aCl._w == *mWTerr)
   {
       if (aCl._b==1)
       {
            ShowProjs(round_ni(aCl._pt));
            return true;
       }

      if (aCl._b==2)
      {
         Pt2di aDec = round_ni(aCl._pt -  Pt2dr(mBoxIn.sz())/2.0);
         mBoxIn._p0 = mBoxIn._p0 + aDec;
         mBoxIn._p1 = mBoxIn._p1 + aDec;
         mAp.DoOneBloc(mBoxIn,mBoxIn,0,mBoxIn);
      }
   } 

   return false;
}

/*******************************************/
/*                                         */
/*                                         */
/*                                         */
/*******************************************/


void cAppliMICMAC::Visualisation(Box2di aBoxIn)
{
   if (mVisu==0) 
      mVisu = new cMicMacVisu (*this,aBoxIn);
 
   mVisu->Load(aBoxIn);

   while(1) mVisu->ExecOneClique();
}

#else// ELISE_X11
void cAppliMICMAC::Visualisation(Box2di aBoxIn)
{
  ELISE_ASSERT
  (
       false,
       "Pas d'X11, pas d'interface"
  );
}
#endif // ELISE_X11

};





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
