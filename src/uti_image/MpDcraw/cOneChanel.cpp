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
// #include "anag_all.h"

#include "StdAfx.h"

namespace NS_MpDcraw
{

cOneChanel::cOneChanel
(
    const cNChannel *   aNC,
    const std::string & aName,
    Im2D_REAL4 aFullImage,
    Pt2di      aP0,
    Pt2di      aPer // Typiquement 2,2
) :
  mNC           (aNC),
  mName         (aName),
  mSz           ( 
                     (aFullImage.tx()+aPer.x-1 - aP0.x)/aPer.x,
                     (aFullImage.ty()+aPer.y-1 - aP0.y)/aPer.y
                ),
  mP0           (aP0),
  mPer          (aPer),
  mIm           (mSz.x,mSz.y),
  mTIm          (mIm),
  mImReech      (1,1),
  mTIR          (mImReech),
  mIMasq        (1,1),
  mTIM          (mIMasq),
  mGridColM2This    (0),
  mCamCorDist       (aNC->CamCorDist()),
  mHomograRedr      (aNC->HomograRedr()),
  mInvHomograRedr   (aNC->InvHomograRedr())
{
std::cout << "COR DIST " << mCamCorDist << "\n";
    TIm2D<REAL4,REAL8>  aTFull(aFullImage);

    Pt2di aP,aPF;
    for (aP.x=0,aPF.x=aP0.x ; aP.x<mSz.x ; aP.x++,aPF.x+=aPer.x)
    {
        for (aP.y=0,aPF.y=aP0.y ; aP.y<mSz.y ; aP.y++,aPF.y+=aPer.y)
        {
           mTIm.oset(aP,aTFull.get(aPF));
        }
    }
}

Fonc_Num cOneChanel::MasqChannel() const
{
   return ((FX%mPer.x) == mP0.x) && ((FY%mPer.y) == mP0.y);
}

const std::string & cOneChanel::Name() const
{
   return mName;
}

void cOneChanel::SauvInit()
{
    if (! mNC->Arg().IsToSplit(mName))
       return;

/*
    double aMul=1.0;
    if (mName=="R") 
       aMul = mNC->Arg().WB(0);
    if ((mName=="V")||(mName=="W")) 
       aMul = mNC->Arg().WB(1);
    if (mName=="B") 
       aMul = mNC->Arg().WB(2);
*/

    std::string aName =  mNC->NameRes(mName);
    Tiff_Im aFile
            (
	        aName.c_str(),
		mIm.sz(),
                mNC->TypeOut(false,0),
                Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	    );
    ELISE_COPY(mIm.all_pts(),mIm.in(),aFile.out());
    //  Tiff_Im::Create8BFromFonc(aName,mIm.sz(),mIm.in());
}


Im2D_REAL4    cOneChanel::ImReech()
{
    return mImReech;
}

Pt2dr   cOneChanel::ToRef(const Pt2dr & aP) const
{
   return mGridToFusionSpace->Direct(aP);
}
Pt2dr   cOneChanel::FromRef(const Pt2dr & aP) const
{
   return mGridToFusionSpace->Inverse(aP);
}



/*
Pt2dr   cOneChanel::ToRef(const Pt2dr & aP) const
{
   return (mIsMaster?aP:mMast2This->Inverse(aP))*mScale;

}
Pt2dr   cOneChanel::FromRef(const Pt2dr & aP) const
{
  return mIsMaster? (aP/mScale) : mMast2This->Direct(aP/mScale);
}
*/



Pt2dr cOneChanel::Direct(Pt2dr aP) const
{
   aP =  (mIsMaster?aP:mGridColM2This->Inverse(aP))*mScale;
   if  (mCamCorDist)
      aP = mCamCorDist->DistInverse(aP);

   if (mHomograRedr)
      aP = mHomograRedr->Direct(aP);

   return aP;
}

bool cOneChanel::OwnInverse(Pt2dr & aP) const
{
   if (mHomograRedr)
      aP = mInvHomograRedr->Direct(aP);
   if  (mCamCorDist)
      aP = mCamCorDist->DistDirecte(aP);
   aP =  mIsMaster? (aP/mScale) : mGridColM2This->Direct(aP/mScale);
   return true;
}





const Pt2di & cOneChanel::P0()  const  {return mP0 ;}
const Pt2di & cOneChanel::Per() const  {return mPer;}




void cOneChanel::InitParamGeom
     (
           const std::string& aMastCh,
           double             aScale,
           cBayerCalibGeom * aBCG,
           cInterpolateurIm2D<REAL4> * anInterp
     )
{
  mScale = aScale;
  mInterp = anInterp;
  if (aMastCh==mName)
  {
     mIsMaster = true;
     std::cout << "Got Master " << aMastCh << "\n";
  }
  else
  {
      mIsMaster = false;
      cBayerGridDirecteEtInverse * aBGDI = 0;
      for
      (
         std::list<cBayerGridDirecteEtInverse>::iterator itB=aBCG->Grids().begin();
         itB!=aBCG->Grids().end();
	 itB++
      )
      {
          if ((itB->Ch1()==aMastCh) &&(itB->Ch2()==mName))
	  {
	     aBGDI = & (*itB);
	  }
      }
      if (aBGDI==0)
      {
           std::cout << "For " << aMastCh << " " << mName << "\n";
           ELISE_ASSERT(false,"Cannot get Grids");
      }

      mGridColM2This = new cDbleGrid(aBGDI->Grid());
  }

  mGridToFusionSpace =  new cDbleGrid
                        (
                            true, // P0P1IsBoxDirect, maintient du comportement actuel ? A priroi les grille de Dist Bayer sont faibles
                            true,
                            Pt2dr(0,0),
                            Pt2dr(mSz),
                            Pt2dr(20,20),
                            *this
                        );

}


void cOneChanel::MakeInitImReech()
{
  std::cout << "REECH " <<  mName << "\n";

  mSzR = round_ni(Pt2dr(mSz) * mScale);

  mImReech = Im2D_REAL4(mSzR.x,mSzR.y);
  mTIR = Im2D<REAL4,REAL8>(mImReech);
  mIMasq =  Im2D_Bits<1> (mSzR.x,mSzR.y);
  mTIM = TIm2DBits<1>(mIMasq);


  Pt2di aP;
  double aDef = -1e9;
  for (aP.x=0 ; aP.x<mSzR.x ; aP.x++)
  {
      for (aP.y=0 ; aP.y<mSzR.y ; aP.y++)
      {
          double aVal = mIm.Get(FromRef(Pt2dr(aP)),*mInterp,aDef);
	  bool Ok = aDef!=aVal;
	  mTIM.oset(aP,Ok);
	  mTIR.oset(aP,Ok?aVal:0.0);
	  /*
	  if (aP==Pt2di(1551,1235))
	  {
	      std::cout << mName << " " << FromRef(aP)  << Ok << " " << aVal << "\n";
	  }
	  */
      }
  }
}


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
