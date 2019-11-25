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

#include "Vino.h"
#include "general/sys_dep.h"

/***********************************************************/
/*                                                         */
/*      Auto Correl                                        */
/*                                                         */
/***********************************************************/

namespace AimeImageAutoCorrel
{

     // =============== cOneICAIAC  ====   

cOneICAIAC::cOneICAIAC(int aTx,int aTy) :
     mTx     (aTx),
     mTy     (aTy),
     mImCor  (mTx,mTy),
     mTImCor (mImCor),
     mImVis  (mTx,mTy)
{
}

void cOneICAIAC::MakeImVis(bool isRobust) 
{
   mImVis = MakeImI1(isRobust,mImCor);
};
void cOneICAIAC::MakeTiff(const std::string & aName)
{
     Tiff_Im::Create8BFromFonc(aName,mImVis.sz(),mImVis.in()+128);
}


     // =============== cAimeImAutoCorr  ====   
cAimeImAutoCorr::cAimeImAutoCorr(Im2D_INT1 anIm) :
    mSz      (anIm.sz()),
    mNbR     (mSz.x),
    mNbT0    (mSz.y)
{
}



     // cCalcAimeImAutoCorr

double cCalcAimeImAutoCorr::AutoCorrelR0(int aRho,int aDTeta)
{
    ELISE_ASSERT(aDTeta>=0,"cCalcAimeImAutoCorr::AutoCorrel");
    double aRes = 0;

    for (int aKT=0 ; aKT<mNbT0 ; aKT++)
    {
        double aV0 =  mTImInit.get(Pt2di(aRho,aKT));
        double aVD =  mTImInit.get(Pt2di(aRho,(aKT+aDTeta)%mNbT0));

        aRes +=  aV0 * aVD;
    }
    return aRes;
}

double cCalcAimeImAutoCorr::AutoCorrelGT(int aRho,int aDTeta)
{
    ELISE_ASSERT(aDTeta>=0,"cCalcAimeImAutoCorr::AutoCorrel");
    double aRes = 0;

    for (int aKT=0 ; aKT<mNbT0 ; aKT++)
    {
        double aV0 =  mTImInit.get(Pt2di(aRho,aKT));
        double aV1 =  mTImInit.get(Pt2di(aRho,(aKT+1)%mNbT0));
        double aVD =  mTImInit.get(Pt2di(aRho,(aKT+aDTeta)%mNbT0));

        aRes +=  (aV1-aV0) * aVD;
    }
    return aRes;
}

double cCalcAimeImAutoCorr::AutoCorrelGR(int aRho,int aDTeta)
{
    ELISE_ASSERT(aDTeta>=0,"cCalcAimeImAutoCorr::AutoCorrel");
    double aRes = 0;

    for (int aKT=0 ; aKT<mNbT0 ; aKT++)
    {
        double aV0 =  mTImInit.get(Pt2di(aRho,aKT));
        double aV1 =  mTImInit.get(Pt2di(aRho-1,aKT));
        double aVD =  mTImInit.get(Pt2di(aRho,(aKT+aDTeta)%mNbT0));

        aRes +=  (aV1-aV0) * aVD;
    }
    return aRes;
}



#if  PB_LINK_AUTOCOR // PB LINK INCOMPREHENSIBLE  Ann + Micmac + Qt => ?@&#!!!
#else


cCalcAimeImAutoCorr::cCalcAimeImAutoCorr(Im2D_INT1 anIm,bool L1Mode) :
    cAimeImAutoCorr (anIm),
    mImInit         (anIm),
    mTImInit        (anIm),
    mL1Mode         (L1Mode),
    mIR0            (mNbR,mNbT0/2),
    mIGR            (mNbR-1,mNbT0),
    mIGT            (mNbR,mNbT0/2)
/*
    mTImCor         (mImCor),
    mImVis          (1,1)
*/
{
    int aSzTetaR0 =  mIR0.mImCor.sz().y;
    int aSzTetaGT =  mIGT.mImCor.sz().y; // GT being anti symetric, take only half size
    for (int aKT=0 ; aKT<mNbT0 ; aKT++)
    {
        double aS0=0;
        double aS1=0;
        double aS2=0;
        bool DoR0 = aKT < aSzTetaR0;
        bool DoGT = aKT < aSzTetaGT;
        for (int aKR=0 ; aKR<mNbR ; aKR++)
        {
           
            if (DoR0)
            {
                double aC = AutoCorrelR0(aKR,aKT+1);
                aS0 += 1;
                aS1 += aC;
                aS2 += ElSquare(aC);
                mIR0.mTImCor.oset(Pt2di(aKR,aKT),aC);
            }

            // double  aC = AutoCorrelGR(aKR,aKT+1); 
            if (DoGT)
            {
               mIGT.mTImCor.oset(Pt2di(aKR,aKT),AutoCorrelGT(aKR,aKT));
            }
            if (aKR>=1)
            {
               mIGR.mTImCor.oset(Pt2di(aKR-1,aKT),AutoCorrelGR(aKR,aKT));
            }
        }
        if (DoR0)
        {
            aS1 /= aS0;
            aS2 /= aS0;
            aS2 -= ElSquare(aS1);
            double aSig = sqrt(ElMax(1e-10,aS2));
            for (int aKR=0 ; aKR<mNbR ; aKR++)
            {
                double  aC =  mIR0.mTImCor.get(Pt2di(aKR,aKT));
                mIR0.mTImCor.oset(Pt2di(aKR,aKT),(aC-aS1)/aSig);
            }
        }
    }
    mIR0.MakeImVis(mL1Mode);
    mIGT.MakeImVis(mL1Mode);
    mIGR.MakeImVis(mL1Mode);
/*
    CalcImVis(mImCor,mImVis);
    double aVMax,aVMin;
    ELISE_COPY(mImCor.all_pts(),mImCor.in(),VMax(aVMax)|VMin(aVMin));
    double aDyn = 127.0/ ElMax(1e-10,ElMax(-aVMin,aVMax));
    ELISE_COPY(mImCor.all_pts(),round_ni(mImCor.in()*aDyn),mImVis.out());
*/
}
#endif 

};


#if (ELISE_X11)


/****************************************/
/*                                      */
/*          Grab Geom                   */
/*                                      */
/****************************************/


ElList<Pt2di> cAppli_Vino::GetPtsImage(bool GlobScale,bool ModeRect,bool AcceptPoint)
{

    std::string aMessage = "Clik  for polygone ; Shift Clik  to finish ; Enter 2 point for rectangle";
    if (AcceptPoint) aMessage = aMessage + "; Enter 1 point for a single pixel";
    PutMessageRelief(0,"Clik  for polygone ; Shift Clik  to finish ; Enter 2 point for rectangle");
    ElImScroller * aCurScr = GlobScale ? mScr : mScr->CurScale();

    bool Cont=true;


    ElList<Pt2di> aList;

    Pt2dr aLastW(-1,-1);

    while (Cont)
    {
        Clik aClik = mW->clik_in();
        Pt2dr aPW = aClik._pt;
        Pt2di aPU = round_ni(aCurScr->to_user(aPW));
        aList = aList + aPU;

        mW->fill_rect(aPW-Pt2dr(3,3),aPW+Pt2dr(3,3),mW->pdisc()(P8COL::green));

        if (aLastW.x >0)
        {
            mW->draw_seg(aPW,aLastW,mW->pdisc()(P8COL::red));
        }

        aLastW  = aPW;
           
        Cont = (!aClik.shifted()) ||  ((aList.card()==1) && (!AcceptPoint));
        if (ModeRect && (aList.card()>=2)) 
        {
           Cont = false;
        }
    }

    return aList;
}


Pt2dr  cAppli_Vino::ToCoordAsc(const Pt2dr & aP)
{
   return Sup(Pt2dr(0,0),Inf(Pt2dr(SzW()),mScr->to_user(aP).mcbyc(mRatioFulXY)));
}


void cAppli_Vino::ShowAsc()
{
   mWAscH->clear();
   mWAscV->clear();
   Pt2dr aP00 =  ToCoordAsc(Pt2dr(0,0));
   Pt2dr aP10 =  ToCoordAsc((Pt2dr(SzW().x,0))) ; 
   Pt2dr aP01 =  ToCoordAsc(Pt2dr(0,SzW().y));  

   

   mWAscH->fill_rect
   (
       Pt2dr(aP00.x,0),
       Pt2dr(ElMax(aP00.x+1,aP10.x),LargAsc()),
       mWAscH->pdisc()(P8COL::yellow)
   );
   mWAscV->fill_rect
   (
         Pt2dr(0,aP00.y),
         Pt2dr(LargAsc(),ElMax(aP00.y+1,aP01.y)),
         mWAscV->pdisc()(P8COL::yellow)
  );

   std::string aStrZoom = "Zoom=" + StrNbChifSign(mScr->sc(),3); // ToString(mScr->sc()); 
   mW->fixed_string(Pt2dr(5,10),aStrZoom.c_str(),mW->pdisc()(P8COL::black),true);

   ShowVect();
}


void  cAppli_Vino::GUR_query_pointer(Clik aCl,bool)
{
    if (mModeGrab==eModeGrapZoomVino)
    {
         double aDY= aCl._pt.y - mP0Click.y;
         double aMulScale = pow(2.0,aDY/SpeedZoomGrab());
         // mScr->set(mTr0,mScale0*aMulScale,true);
         mScr->SetScArroundPW(mP0Click,mScale0*aMulScale,true);
         // std::cout << "GUR_query_pointer " << mP0Click << " " << aCl._pt << "\n";
    }
    if (mModeGrab==eModeGrapTranslateVino)
    {
       mScr->set(mTr0-(aCl._pt-mP0Click)/mScale0,mScale0,false);
    }
    if (mModeGrab==eModeGrapAscX)
    {
       mScr->set(mTr0+Pt2dr(aCl._pt.x-mP0Click.x,0)/mRatioFulXY.x,mScale0,false);
       ShowAsc();
    }
    if (mModeGrab==eModeGrapAscY)
    {
       mScr->set(mTr0+Pt2dr(0,aCl._pt.y-mP0Click.y)/mRatioFulXY.y,mScale0,false);
       ShowAsc();
    }

    if (mModeGrab==eModeGrapShowRadiom)
    {
         ShowOneVal(aCl._pt);
    }

    if (mModeGrab==eModeVinoPopUp)
    {
         mPopUpCur->SetPtActif(Pt2di(aCl._pt));
    }
}

void cAppli_Vino::ZoomMolette()
{
    double aSc =  mScale0 * pow(2.0,SpeedZoomMolette()*(mBut0==5?1:-1));
    mScr->SetScArroundPW(mP0Click,aSc,false);
}

void cAppli_Vino::GetRect(Pt2dr & aP0,Pt2dr &aP1)
{
    ElList<Pt2di> aL = GetPtsImage(true,true,false);

    aP0 = Pt2dr(aL.car());
    aP1 = Pt2dr(aL.cdr().car());
    pt_set_min_max(aP0,aP1);
}

void cAppli_Vino::ZoomRect()
{
    Pt2dr aP0,aP1;
    GetRect(aP0,aP1);

    Pt2dr aSz = aP1-aP0;

    double aScale = ElMin(SzW().x/aSz.x,SzW().y/aSz.y);

    // std::cout << "SCALE " << aScale  << " " << aSz  << "\n";
    // mScr->SetScArroundPW(mScr->to_win((aP0+aP1)/2.0),aScale,false);

    Pt2dr aSzU = SzW() / aScale;
    Pt2dr aMil = (aP0+aP1)/2.0;

    mScr->set(aMil-aSzU/2.0,aScale);
    ShowAsc();
}



void cAppli_Vino::ExeClikGeom(Clik aCl)
{

     if (mShift0)
     {
         if (mCtrl0)
         {
             mScr->set_max();
         }
         else
         {
              mModeGrab = eModeGrapZoomVino;
              mW->grab(*this);
              mScr->SetScArroundPW(mP0Click,mScr->sc(),false);
         }
     }
     else
     {
         if (mCtrl0)
         {
              mScr->SetScArroundPW(mP0Click,1.0,false);
         }
         else
         {
              mModeGrab = eModeGrapTranslateVino;
              mW->grab(*this);
         }
     }
     ShowAsc();
}


/**************** A mettre dans un autre fichier Show/Inspect-Vect ******/

void cAppli_Vino::ShowVect()
{
   if (mWithPCarac) 
      ShowVectPCarac();
}

int  cAppli_Vino::IndexNearest(double aDistSeuil,const Pt2dr & aPClU,double * aDist,eTypePtRemark aType)
{
   cVecTplResRVoisin<cOnePCarac *> aVRV;
   mQTPC->RVoisins(aVRV,aPClU,aDistSeuil);


   // mAppli.Qt2()->KPPVois(aP0,2,100.0);
   cOnePCarac * aRes=nullptr;
   double aDMin=1e20;
   // for (int aKP=0 ; aKP<int(mSPC->OnePCarac().size()) ; aKP++)
   for (const auto & aPCP : static_cast<std::vector<cOnePCarac *> &>(aVRV) )
   {
       // const auto & aPC = mSPC->OnePCarac()[aKP];
       double aDist = euclid(aPCP->Pt(),aPClU);
       if (  ((aPCP->Kind()==aType)||(aType==eTPR_NoLabel)) && (aDist<aDMin))
       {
          aDMin = aDist;
          aRes = aPCP;
       }
   }
   if (aDist) 
      *aDist = aDMin;
   int anIndex =  (aRes ? aRes->Id() : -1);
   return anIndex;
}

const cOnePCarac * cAppli_Vino::Nearest(double aDistSeuil,const Pt2dr & aPClU,double * aDist,eTypePtRemark aType)
{
    int I = IndexNearest(aDistSeuil,aPClU,aDist,aType);
    if (I>=0) return &(mSPC->OnePCarac()[I]);
    return nullptr;
}

void ShowCurve(Im2D_INT1 aIm,int aY,const Pt2di & aP0,const Pt2di & aP1,Video_Win * aW,int aCoul)
{
   int aTx = aIm.tx();
   // std::cout << "TXxxx= " << aTx << "\n";
   INT1 * aV = aIm.data()[aY];
   double aVmax = aV[0];
   double aVmin = aV[0];
   for (int anX=1 ; anX<aTx ; anX++)
   {
      aVmin = ElMin(aVmin,double(aV[anX]));
      aVmax = ElMax(aVmax,double(aV[anX]));
   }

   std::vector<double> aVX;
   std::vector<double> aVY;
   for (int anX=0 ; anX<aTx ; anX++)
   {
        aVX.push_back(aP0.x+ ( anX /double(aTx-1)) * (aP1.x-aP0.x));
        aVY.push_back(aP0.y+ ((aV[anX]-aVmin) /(aVmax-aVmin) ) * (aP1.y-aP0.y));
   }
   ELISE_COPY ( rectangle(aP0,aP1), 255, aW->ogray());
   
   for (int anX=1 ; anX<int(aTx) ; anX++)
   {
       aW->draw_seg
       (
           Pt2dr(aVX[anX-1],aVY[anX-1]),
           Pt2dr(aVX[anX],aVY[anX]),
           aW->pdisc()(aCoul)
       );
   }
}


//======================================================================
//======================================================================

/*
    double aVMax,aVMin;
    ELISE_COPY(aImCor.all_pts(),aImCor.in(),VMax(aVMax)|VMin(aVMin));
    double aDyn = 127.0/ ElMax(1e-10,ElMax(-aVMin,aVMax));
    ELISE_COPY(aImCor.all_pts(),round_ni(aImCor.in()*aDyn),aImVis.out());
*/

void cAppli_Vino::ShowImCA(int aDx,int aDy,Im2D_INT1 aIm)
{
    double aVMax,aVMin;
    ELISE_COPY(aIm.all_pts(),aIm.in(),VMax(aVMax)|VMin(aVMin));
    double aDyn = 127.0/ ElMax(1e-10,ElMax(-aVMin,aVMax));


    Pt2di aSz = aIm.sz();
    ELISE_COPY
    (
        rectangle(Pt2di(aDx,aDy),Pt2di(aDx+aSz.y*mZoomCA,aDy+aSz.x*mZoomCA)),
        Max(0,Min(255,128+aDyn*aIm.in()[Virgule(FY-aDy,FX-aDx)/mZoomCA])),
        mW->ogray()
    );
}


void  cAppli_Vino::ShowSPC(const Pt2dr & aPClW)
{
   ElSimilitude aU2W = mScr->to_win();
   ElSimilitude aW2U = mScr->to_user();
   Pt2dr aPClU = aW2U(aPClW);

   mW->draw_circle_loc(aPClW,3.0,mW->pdisc()(P8COL::cyan));

   const cOnePCarac *  aNearest = Nearest(1000.0,aPClU);
   if (aNearest)
   {
       // mW->draw_circle_loc(aU2W(aNearest->Pt()),3.0,mW->pdisc()(P8COL::magenta));
       // mW->draw_circle_loc(aU2W(aNearest->Pt()),5.0,mW->pdisc()(P8COL::magenta));

       double aSc = mScr->sc();
       for (const auto & aRho : aNearest->VectRho())
       {
           mW->draw_circle_loc(aU2W(aNearest->Pt()),aSc*aRho,mW->pdisc()(P8COL::magenta));
       }
       if (!mAVSI.empty())
          std::cout << "#########################################################\n";

       std::cout << "PTT=" << aNearest->Pt() << mNameIm <<  "\n";
       std::cout << "  * Id= : " << aNearest->Id() << "\n";
       std::cout << "  * AutoC : " << aNearest->AutoCorrel() << "\n";
       std::cout << "  * Scale : "      << aNearest->Scale()      << "\n";
       std::cout << "  * SStab : "      << aNearest->ScaleStab()      << "\n";
       std::cout << "  * Contr : "      << aNearest->Contraste()  << " Rel : " << aNearest->ContrasteRel()   << "\n";
       std::cout << "Rhoooo " << aNearest->VectRho() << "\n";

       std::cout << "\n";

       {
          Im2D_INT1 aImLogT = aNearest->ImLogPol();
          Pt2di aSz = aImLogT.sz();


          ELISE_COPY
          (
              rectangle(Pt2di(0,0),Pt2di(aSz.y*mZoomCA,aSz.x*mZoomCA)),
              Max(0,Min(255,128 + 2 * aImLogT.in()[Virgule(FY,FX)/mZoomCA])),
              mW->ogray()
          );




          int aMarge = 5;
          int aSzW    = 45;
          int aPer = 8;

          bool ShowInvRad = true;
          bool ShowProfRad = true;

          std::vector<Im2D_INT1> aVIm2Show;
          if (ShowInvRad) 
             aVIm2Show.push_back(aNearest->InvR().ImRad());
          if (ShowProfRad) 
          {
             aVIm2Show.push_back(aNearest->ProfR().ImProfil());
          }

          int aCpt = 0;
          for (int  aKIm =0 ; aKIm<int(aVIm2Show.size()) ; aKIm++)
          {
               int aNbY = aVIm2Show[aKIm].sz().y;
               for (int anY=0 ; anY<aNbY ; anY++)
               {
                     int aX0 =  aMarge+(aCpt%aPer) * (aMarge+aSzW);
                     int aX1 =  aX0 + aSzW;

                     int aY0 =  aSz.x*mZoomCA + aMarge+(aCpt/aPer) * (aMarge+aSzW);
                     int aY1 = aY0 + aSzW;
                     ShowCurve
                     (
                         aVIm2Show[aKIm],
                         anY,
                         Pt2di(aX0,aY0), Pt2di(aX1,aY1),
                         mW,
                         1 + (anY%6)
                     );
                     aCpt++;
               }
          }

          ShowImCA(  aSz.y*mZoomCA + 10,0,aNearest->RIAC().IGT());
          ShowImCA(2*aSz.y*mZoomCA + 20,0,aNearest->RIAC().IGR());
          ShowImCA(3*aSz.y*mZoomCA + 30,0,aNearest->RIAC().IR0());
       }

       if (! mVptHom.empty())
       {
           int  aK = IndexNearest(1000.0,aPClU);
           ELISE_ASSERT((aK>=0) && aNearest==&(mSPC->OnePCarac()[aK]),"NEAREST !!??");
           const cOnePCarac * aPCHom = mVptHom.at(aK);
           if (aPCHom)
           {
               ElSimilitude aHU2W =  mAVSI.at(0)->mScr->to_win();
               Pt2dr aPt = aHU2W(aPCHom->Pt());
               std::cout << "HOMMM " << aPt << "\n";
               mAVSI.at(0)->ShowSPC(aPt);

               TestMatchInvRad(mSPC->OnePCarac(),aNearest,aPCHom);
           }
       }
   }
}


void cAppli_Vino::ShowVectPCarac()
{
   ElSimilitude aSim = mScr->to_win();

   if (mSPC)
   {
       for (int aKP=0 ; aKP<int(mSPC->OnePCarac().size()) ; aKP++)
       {
           const cOnePCarac & aPC = mSPC->OnePCarac()[aKP];
           Pt2dr aPU = aPC.Pt();
           Pt2dr aPW = aSim(aPU);
           if (   (aPW.x>0) && (aPW.y>0) && (aPW.x<SzW().x) && (aPW.y<SzW().y) 
               && (aPC.ContrasteRel()>mSeuilContRel) &&   (aPC.AutoCorrel()< mSeuilAC)
              )
           {
               mW->draw_circle_loc(aPW,aPC.Scale()*2*mScr->sc(),mW->pdisc()(P8COL::yellow));
 
               bool HighL =  (mVptHom.size() && mVptHom.at(aKP));
               ShowPt(aPC,aSim,mW,HighL);
               // Pt2dr aDirMS = aPC.DirMS();
               Pt2dr aDirMS = aPC.DirAC();
// std::cout << "aDirMS " << aDirMS << "\n";
               if (euclid(aDirMS) != 0)
               {
                   aDirMS =  vunit(aDirMS) * 20.0;
                   mW->draw_seg(aPW,aPW+aDirMS,mW->pdisc()(P8COL::green));
               }
           }
           else
           {
              // std::cout << "FfFffff " << aPU << " " << aPW << "\n";
           }
       }
   }
   if (mVSift.size())
   {
       for (const auto & aSP : mVSift)
       {
           Pt2dr aPU(aSP.x*mSSF,aSP.y*mSSF);
           Pt2dr aPW = aSim(aPU);
           if ((aPW.x>0) && (aPW.y>0) && (aPW.x<SzW().x) && (aPW.y<SzW().y))
           {
               mW->draw_circle_loc(aPW,3.0,mW->pdisc()(P8COL::red));
               Pt2dr aDir = Pt2dr::FromPolar(20,aSP.angle);
               mW->draw_seg(aPW,aPW+aDir,mW->pdisc()(P8COL::green));
               mW->draw_circle_loc(aPW,aSP.scale*2*mScr->sc(),mW->pdisc()(P8COL::yellow));
               // std::cout << "TETA " <<  aSP.scale << " " << aSP.angle << "\n";
               //ShowPt(aP,aSim,mW);
           }
           else
           {
              // std::cout << "FfFffff " << aPU << " " << aPW << "\n";
           }
       }
   }
   if (mWithAime)
   {
        AimeVisu();
   }
}



#endif



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
