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

#include "general/sys_dep.h"

#if (ELISE_X11)

#include "Vino.h"

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

int  cAppli_Vino::IndexNearest(const Pt2dr & aPClU,double * aDist,eTypePtRemark aType)
{
   int aRes = -1;
   double aDMin=1e20;
   for (int aKP=0 ; aKP<int(mSPC->OnePCarac().size()) ; aKP++)
   {
       const auto & aPC = mSPC->OnePCarac()[aKP];
       double aDist = euclid(aPC.Pt(),aPClU);
       if (  ((aPC.Kind()==aType)||(aType==eTPR_NoLabel)) && (aDist<aDMin))
       {
          aDMin = aDist;
          aRes = aKP;
       }
   }
   if (aDist) 
      *aDist = aDMin;
   return aRes;
}

const cOnePCarac * cAppli_Vino::Nearest(const Pt2dr & aPClU,double * aDist,eTypePtRemark aType)
{
    int I = IndexNearest(aPClU,aDist,aType);
    if (I>=0) return &(mSPC->OnePCarac()[I]);
    return nullptr;
}

void ShowCurve(Im2D_INT1 aIm,int aK,const Pt2di & aP0,const Pt2di & aP1,Video_Win * aW,int aCoul)
{
   int aTx = aIm.tx();
   INT1 * aV = aIm.data()[aK];
   double aVmax = aV[0];
   double aVmin = aV[0];
   for (int aK=1 ; aK<aTx ; aK++)
   {
      aVmin = ElMin(aVmin,double(aV[aK]));
      aVmax = ElMax(aVmax,double(aV[aK]));
   }

   std::vector<double> aVX;
   std::vector<double> aVY;
   for (int aK=0 ; aK<aTx ; aK++)
   {
        aVX.push_back(aP0.x+ ( aK /double(aTx-1)) * (aP1.x-aP0.x));
        aVY.push_back(aP0.y+ ((aV[aK]-aVmin) /(aVmax-aVmin) ) * (aP1.y-aP0.y));
   }
   ELISE_COPY ( rectangle(aP0,aP1), 255, aW->ogray());
   
   for (int aK=1 ; aK<int(aTx) ; aK++)
   {
       aW->draw_seg
       (
           Pt2dr(aVX[aK-1],aVY[aK-1]),
           Pt2dr(aVX[aK],aVY[aK]),
           aW->pdisc()(aCoul)
       );
   }
}

void  cAppli_Vino::ShowSPC(const Pt2dr & aPClW)
{
   ElSimilitude aU2W = mScr->to_win();
   ElSimilitude aW2U = mScr->to_user();
   Pt2dr aPClU = aW2U(aPClW);

   mW->draw_circle_loc(aPClW,3.0,mW->pdisc()(P8COL::cyan));

   const cOnePCarac *  aNearest = Nearest(aPClU);
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
       std::cout << "  * AutoC : " << aNearest->AutoCorrel() << "\n";
       std::cout << "  * Scale : "      << aNearest->Scale()      << "\n";
       std::cout << "  * SStab : "      << aNearest->ScaleStab()      << "\n";
       std::cout << "  * Contr : "      << aNearest->Contraste()  << " Rel : " << aNearest->ContrasteRel()   << "\n";
       std::cout << "Rhoooo " << aNearest->VectRho() << "\n";

       std::cout << "\n";

       {
          Im2D_INT1 aImLogT = aNearest->ImLogPol();
          int aZoom=10;
          Pt2di aSz = aImLogT.sz();

          ELISE_COPY
          (
              rectangle(Pt2di(0,0),Pt2di(aSz.y*aZoom,aSz.x*aZoom)),
              Max(0,Min(255,128 + 2 * aImLogT.in()[Virgule(FY,FX)/aZoom])),
              mW->ogray()
          );

          int aMarge = 5;
          int aSzW    = 45;
          int aPer = 8;

          for (int aK=0 ; aK<eTIR_NoLabel ; aK++)
          {
                int aX0 =  aMarge+(aK%aPer) * (aMarge+aSzW);
                int aX1 =  aX0 + aSzW;

                int aY0 =  aSz.x*aZoom + aMarge+(aK/aPer) * (aMarge+aSzW);
                int aY1 = aY0 + aSzW;
                ShowCurve
                (
                    aNearest->InvR().ImRad() ,
                    aK,
                    Pt2di(aX0,aY0), Pt2di(aX1,aY1),
                    mW,
                    1 + (aK%6)
                );
          }
       }

       if (! mVptHom.empty())
       {
           int  aK = IndexNearest(aPClU);
           ELISE_ASSERT(aNearest==&(mSPC->OnePCarac()[aK]),"NEAREST !!??");
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
