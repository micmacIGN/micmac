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


//
//   Ce fichier contient des fonctionnalites pour la recherche de
//   cibles dans le cas "caprice des dieux"
//

/****************************************************/
/*                                                  */
/*               cParamEllipse                      */
/*                                                  */
/****************************************************/

cParamEllipse::cParamEllipse
(
      int aNbDisc,
      Pt2dr aCentre,
      Pt2dr aDirH,
      Pt2dr aDirV,
      Pt2dr  aProjN
) :
  mComputed   (false),
  mTeta0      (PI/2),
  mNbDisc     (aNbDisc),
  mCentre     (aCentre),
  mDirH       (aDirH),
  mDirV       (aDirV),
  mProjN      (aProjN)
{
}



int cParamEllipse::NbDisc() const
{
   return mNbDisc;
}

Pt2dr cParamEllipse::Centre() const
{
   return mCentre;
}

void cParamEllipse::Compute()
{
   if (mComputed) 
      return;

   mPLarg.x = sqrt(ElSquare(mDirH.x)+ElSquare(mDirV.x));
   mPLarg.y = sqrt(ElSquare(mDirH.y)+ElSquare(mDirV.y));
   ImRON2ParmEllipse(mA,mB,mC,mDirH,mDirV);
   EllipseEq2ParamPhys(mGdAxe,mPtAxe,mTetaGA,mA,mB,mC);


   mComputed = true;
}

double cParamEllipse::GdAxe() const
{
   return mGdAxe;
}

double cParamEllipse::PtAxe() const
{
   return mPtAxe;
}

Pt2dr cParamEllipse::KiemV(int aK) const 
{
    double aTeta = mTeta0 + (2.0*PI*aK) / mNbDisc;
    return mDirH*cos(aTeta)+mDirV*sin(aTeta);
}

const Pt2dr &  cParamEllipse::DirH() const
{
   return mDirH;
}

const Pt2dr &  cParamEllipse::DirV() const
{
   return mDirV;
}

const Pt2dr &  cParamEllipse::ProjN() const
{
   return mProjN;
}

const Pt2dr &  cParamEllipse::PLarg() const
{
   return mPLarg;
}

Pt2dr   cParamEllipse::DirGAxe() const
{
   return Pt2dr(cos(mTetaGA),sin(mTetaGA));
}

Pt2dr   cParamEllipse::DirPAxe() const
{
   return DirGAxe() * Pt2dr(0.0,1.0);
}

void cParamEllipse::SetCentre(Pt2dr aP)
{   
   mCentre = aP;
}

Pt2d<Fonc_Num>  cParamEllipse::StdTransfo(Pt2d<Fonc_Num> aP)
{
   Pt2dr aDGA = DirGAxe();
   Pt2dr aPGA = DirPAxe();
   return Pt2d<Fonc_Num>
          (
              mCentre.x + aDGA.x *  aP.x + aPGA.x *  aP.y,
              mCentre.y + aDGA.y *  aP.x + aPGA.y *  aP.y
	  );
}

Pt2dr  cParamEllipse::StdTransfo(Pt2dr aP)
{
   Pt2dr aDGA = DirGAxe();
   Pt2dr aPGA = DirPAxe();
   return Pt2dr
          (
              mCentre.x + aDGA.x *  aP.x + aPGA.x *  aP.y,
              mCentre.y + aDGA.y *  aP.x + aPGA.y *  aP.y
	  );
}

double cParamEllipse::SurfInterRect(Pt2dr aP0,Pt2dr aP1)
{
   return SurfIER(Pt2dr(0,0),mA,mB,mC,aP0,aP1);
}

bool cParamEllipse::VecInEllipe(const Pt2dr & aP) const
{
   return ElSquare(mA*aP.x+mB*aP.y) +  ElSquare(mB*aP.x+mC*aP.y)  < 1.0;
}


/****************************************************/
/*                                                  */
/*               cRechercheCDD                      */
/*                                                  */
/****************************************************/


cRechercheCDD::cRechercheCDD(Im2D_REAL4 anImGlob,Video_Win * aW) :
   Optim2DParam(0.01,1e10,1e-5,false),
   mImCible  (1,1),
   mTImC     (mImCible),
   mImGlob   (anImGlob),
   mTImG     (mImGlob),
   mW        (aW)
{
}

REAL cRechercheCDD::Op2DParam_ComputeScore(REAL x,REAL y)
{
   double aRes =0;
   Pt2di aP;
   Pt2dr aC(x,y);
   for (aP.x=0; aP.x<mSz.x; aP.x++)
   {
       for (aP.y=0; aP.y<mSz.y; aP.y++)
       {
           Pt2dr aPEl = Pt2dr(aP) + Pt2dr(mDec) + aC;

	    aRes += mTImG.getr(aPEl,0) * mTImC.get(aP);
       }
   }
   return aRes;
}

void cRechercheCDD::Show(cParamEllipse & anEl)
{
   if (! mW) 
       return;

   double aRab = 2.0;
   double aZoom = 15.0;

   mW->clear();

   Pt2dr  aPRab(aRab,aRab);
   Pt2dr   aDec =  Pt2dr(round_down(-anEl.PLarg()-aPRab));
   Pt2di aDec0 = Pt2di(aDec);
   mSz = Pt2di(aDec*(-2.0));
   aDec += anEl.Centre();

   Pt2di aSzZ = Pt2di(Pt2dr(mSz) * aZoom);
   Im2D_U_INT1 aIZ(aSzZ.x,aSzZ.y);
   TIm2D<U_INT1,INT> aTIZ(aIZ);
   Pt2di aP;
   for (aP.x=0 ; aP.x<aSzZ.x ; aP.x++)
   {
       for (aP.y=0 ; aP.y<aSzZ.y ; aP.y++)
       {
           Pt2dr aPG = (Pt2dr(aP)/aZoom) + aDec;
           aTIZ.oset(aP,mTImG.get(round_ni(aPG),0));
       }
   }
   ELISE_COPY(aIZ.all_pts(),aIZ.in(),mW->ogray());


   for (int aK = 0 ; aK< anEl.NbDisc() ; aK++)
   {
        int aCoul = P8COL::red;
	mW->draw_seg
        (
             (anEl.KiemV(aK) -Pt2dr(aDec0))*aZoom,
             (anEl.KiemV(aK+1) -Pt2dr(aDec0))*aZoom,
             mW->pdisc()(aCoul)
        );
    }

}

void cRechercheCDD::RechercheCorrel(cParamEllipse & anEl)
{
   double aRab = 6.0;
   Pt2dr  aPRab(aRab,aRab);
   mDec =  round_down(-anEl.PLarg()-aPRab);
   mSz = mDec*-2;

   mImCible.Resize(mSz);
   mTImC = TIm2D<U_INT1,INT> (mImCible);

   double aFl = 1.0;  //  0.5 correspond a  juste 1 pixel
   Pt2dr  aPFL(aFl,aFl);

   Pt2di aP;
   for (aP.x=0; aP.x<mSz.x; aP.x++)
   {
       for (aP.y=0; aP.y<mSz.y; aP.y++)
       {
           Pt2dr aPEl = Pt2dr(aP + mDec);

            double aS = anEl.SurfInterRect(aPEl-aPFL,aPEl+aPFL);
	    aS /= ElSquare(2*aFl);
	    //  std::cout << aS//  << aPEl << "\n";
	    mTImC.oset(aP,aS*255);
       }
   }
   if (1)
   {
      mW->clear();
      ELISE_COPY
      (
          mImCible.all_pts(),
          mImCible.in(),
          mW->chc(Pt2dr(0,0),Pt2dr(3,3)).ogray()
      );
      /*
      Pt2di aDec = mDec+(anEl.Centre());
      ELISE_COPY
      (
          rectangle(aDec,aDec+mSz),
          255-trans(mImCible.in(),-aDec),
          Tiff_Im("toto.tif").out()
      );
      getchar();
      */
   }
   optim(anEl.Centre());
   anEl.SetCentre(Optim2DParam::param());
   // std::cout << "ZZZ : " << Optim2DParam::param() << "\n";

}

void cRechercheCDD::RechercheInit(cParamEllipse & anEl,double aRab)
{ 

   anEl.Compute();


  Pt2di aPMin;
  int aGA =  round_up(anEl.GdAxe());
  int aPA =  round_up(anEl.PtAxe());
  {
     Pt2di aSzR(round_ni(aRab),round_ni(aRab));
     int xy[2];

      Pt2d<Fonc_Num> aF = anEl.StdTransfo(Pt2d<Fonc_Num>(FX,FY));
      ELISE_COPY
      (
          rectangle(-aSzR,aSzR),
          rect_som
	  (
	     mImGlob.in_proj()[Virgule(aF.x,aF.y)],
	     Pt2di(aGA,aPA)
	  ),
          WhichMin(xy,2)
      );
      aPMin.x = xy[0];
      aPMin.y = xy[1];
  }

   if (false)
   {
      mW->clear();
      int lX = round_ni(aGA+aRab);
      int lY = round_ni(aPA+aRab);
      Pt2d<Fonc_Num> aF = anEl.StdTransfo(Pt2d<Fonc_Num>(FX-lX,FY-lY));
      ELISE_COPY
      (
           rectangle(Pt2di(0,0),Pt2di(lX,lY)*2),
	   mImGlob.in_proj()[Virgule(aF.x,aF.y)],
	   mW->ogray()
      );
      mW->draw_circle_loc(Pt2dr(aPMin+Pt2di(lX,lY)),4,mW->pdisc()(P8COL::red));
   }

   anEl.SetCentre(anEl.StdTransfo(Pt2dr(aPMin)));
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
