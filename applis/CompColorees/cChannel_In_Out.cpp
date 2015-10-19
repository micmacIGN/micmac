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

#include "CompColore.h"


/*************************************************/
/*                                               */
/*          cChannelIn                           */
/*                                               */
/*************************************************/


cChannelIn::cChannelIn
(
     cCC_Appli &            anAppli,
     GenIm::type_el aTypeIn,
     cCC_OneChannel & aCOC,
     Pt2di aSz,
     const cChannelCmpCol & aCCC
) :
   mAppli  (anAppli),
   mCOC    (aCOC),
   mCCC    (aCCC),
   mPds    (aCCC.Pds().Val()),
   mDyn    (aCCC.Dyn().Val()),
   mOffset (aCCC.Offset().Val()),
   // mInterp (*anInterp),
   mIm     (Ptr_D2alloc_im2d(aTypeIn,aSz.x+aRabResize,aSz.y+aRabResize)),
   mImI    (0)
   // mP0K    (mImI->SzKernel()+1,mImI->SzKernel()+ 1),
   // mP1K    (aSz - mP0K - Pt2di(1,1))
   //mImIn   (aSz.x,aSz.y,0.0),
   // mDataI  (mImIn.data())
   // mTIn    (mImIn)
{
    Resize(aSz);
}


void cChannelIn::Resize(const Pt2di & aSz)
{
    mIm->Resize(aSz);
    cTplValGesInit<double> aPBic = mAppli.CCC().ParamBiCub();
    delete mImI;
    mImI    =aPBic.IsInit()?(mIm->BiCubIm(aPBic.Val())):(mIm->BilinIm());

    if (mCCC.ParamBiCub().IsInit())
    {
        delete mImI;
        mImI    =mIm->BiCubIm(mCCC.ParamBiCub().Val());
    }

    mP0K  = Pt2di(mImI->SzKernel()+1,mImI->SzKernel()+ 1);
    mP1K  = aSz - mP0K - Pt2di(1,1);
}


cChannelIn::~cChannelIn()
{
   delete mIm;
}

Output cChannelIn::OutForInit()
{
   return mIm->out();
}

const cChannelCmpCol & cChannelIn::CCC()
{
   return mCCC;
}

double cChannelIn::Pds() const
{
   return mPds;
}

Pt2dr cChannelIn::PFus2PIm(const Pt2dr & aPFus) const
{
   return  mCOC.FromFusionSpace(aPFus);
}

double  cChannelIn::GetInterpolPFus(const Pt2dr & aPFus,bool & Ok)
{

   Pt2dr aPIm = PFus2PIm(aPFus) ;

    if (    (aPIm.x > mP0K.x)
         && (aPIm.y > mP0K.y)
         && (aPIm.x < mP1K.x)
         && (aPIm.y < mP1K.y)
       )
   {
      Ok = true;
      double aRes =  mDyn * (mImI->Get(aPIm)+mOffset);
      return aRes;
   }
   Ok= false;
   return 0;
}


/*************************************************/
/*                                               */
/*          cChanelOut                           */
/*                                               */
/*************************************************/

cChanelOut::cChanelOut(Pt2di aSz) :
  mSPds   (0),
  mNbCh   (0),
  mSz     (aSz),
  mIm     (Ptr_D2alloc_im2d(GenIm::int2,aSz.x+aRabResizeOut,aSz.y+aRabResizeOut)),
  mMasq   (aSz.x+aRabResizeOut,aSz.y+aRabResizeOut,0),
  mTM     (mMasq)
{
}

double cChanelOut::CalibRel(const cChanelOut  & aSec,double aMaxRatio,double & aSig) const
{
   double aRMoy=0;
   double aR2M=0;
   double aNb=0;
   Pt2di aP;
   for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
   {
      for (aP.y=0 ; aP.y<mSz.x ; aP.y++)
      {
         if (mTM.get(aP,0) && aSec.mTM.get(aP,0))
         {
            double aV1 = mIm->GetR(aP);
            double aV2 = aSec.mIm->GetR(aP);
            if ((aV1*aMaxRatio> aV2) && (aV2*aMaxRatio> aV1))
            {
                double aR = log(aV1/aV2);
                aRMoy += aR;
                aR2M += ElSquare(aR);
                aNb++;
            }
         }
      }
   }
   if (aNb==0) return -1;
   aRMoy /= aNb;
   aSig = aR2M/aNb - ElSquare(aRMoy);
   return exp(aRMoy);
}


void cChanelOut::ResizeAndReset(const Pt2di & aSz)
{
    mSz =aSz;
    mIm->Resize(aSz);
    ELISE_COPY(mIm->all_pts(),0,mIm->out());
    mMasq = Im2D_Bits<1>(aSz.x,aSz.y,0);   // A REJOUTER RESIZE BITS ....;
    mTM = TIm2DBits<1>(mMasq);
    mNbCh=0;
    mChIns.clear();
    mSPds=0;
}

cChanelOut::~cChanelOut()
{
   delete mIm;
}

void cChanelOut::AddChIn(cChannelIn * aCI)
{
   mNbCh++;
   mChIns.push_back(aCI);
   mSPds += aCI->Pds();
}

double  cChanelOut::GetInterpolPFus(const Pt2dr & aPFus,bool & Ok)
{
   Ok = false;

   double aRes = 0;
   double aSPds = 0;

   for (int aKC=0;  aKC<mNbCh ; aKC++)
   {
       bool Ok1;
       double aVal = mChIns[aKC]->GetInterpolPFus(aPFus,Ok1) ;
       if (Ok1)
       {
           Ok = true;
           double aPds= mChIns[aKC]->Pds();
           aSPds += aPds;
           aRes +=  aVal * aPds;
       }
   }
   if (! Ok) return 0;

   return aRes / aSPds;
}


void cChanelOut::InitByInterp()
{
   Pt2di aP;
   for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
   {
      for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
      {
          bool  Ok;
	  double aV = GetInterpolPFus(Pt2dr(aP),Ok);
          mIm->TronqueAndSet(aP,aV);
          // mTIm.oset(aP,aV);
	  mTM.oset(aP,Ok);
      }
   }
}

const  std::vector<cChannelIn *>  & cChanelOut::ChIns()
{
   return mChIns;
}

double cChanelOut::GetVal(const Pt2di & aP) const
{
   return mIm->GetR(aP);
}


Im2DGen *  cChanelOut::Im()
{
  return mIm;
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
