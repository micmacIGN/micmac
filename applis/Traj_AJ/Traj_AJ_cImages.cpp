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
#include "private/all.h"
#include "Traj_Aj.h"

using namespace NS_AJ;

/**************************************************************/
/*                                                            */
/*               cTAj2_OneImage                               */
/*                                                            */
/**************************************************************/



cTAj2_OneImage::cTAj2_OneImage
(
    cAppli_Traj_AJ & anAppli,
    cTAj2_OneLayerIm & aLayer,
    const std::string & aName
) :
   mAppli  (anAppli),
   mLayer  (aLayer),
   mName   (aName),
   mMDP    (0),
   mNum    (-1),
   mDQM    (eNoMatch),
   mNext   (this),
   mPrec   (this),
   mVitOK  (false)
{
    mMDP = new cMetaDataPhoto(cMetaDataPhoto::CreateExiv2(mAppli.DC()+aName));
}

bool  cTAj2_OneImage::VitOK() const
{
   return mVitOK;
}

Pt3dr cTAj2_OneImage::Vitesse() const
{
    ELISE_ASSERT(mVitOK," cTAj2_OneImage, Vitesse non init !! ");
    return mVitessse;
}


void cTAj2_OneImage::ResetMatch()
{
    mNext  = this;
    mPrec  = this;
    mVitOK = false;
    mLKM.Reset();
}

void cTAj2_OneImage::SetLinks(cTAj2_OneImage * aPrec)
{
   mPrec = aPrec;
   mPrec->mNext = this;
}

void cTAj2_OneImage::EstimVitesse()
{
   if (mNext==mPrec)
   {
      mVitOK = false;
   }
   else
   {
      mVitOK = true;
      cTAj2_OneLogIm * aPL = mPrec->BestMatch();
      cTAj2_OneLogIm * aNL = mNext->BestMatch();
      double aDT = aNL->T0() - aPL->T0();
      Pt3dr aDPos = aNL->PGeoC() - aPL->PGeoC();
      mVitessse = aDPos / aDT;
   }
}


void cTAj2_OneImage::InitT0(const cTAj2_OneImage & aI2)
{
    mTime2I0 = mMDP->Date().DifInSec(aI2.mMDP->Date());
}

double cTAj2_OneImage::T0() const
{
   mLayer.FinishT0();
   return mTime2I0;
}

const std::string & cTAj2_OneImage::Name() const
{
   return mName;
}

int  cTAj2_OneImage::Num() const
{
   return mNum;
}

void  cTAj2_OneImage::SetNum(int aNum) 
{
    mNum = aNum;
}


void cTAj2_OneImage::UpdateMatch(cTAj2_OneLogIm * aLog,double aDif)
{
   mLKM.Update(aLog,aDif);
}

cTAj2_OneLogIm *cTAj2_OneImage::BestMatch()
{
    return mLKM.mBestMatch;
}

eTypeMatch cTAj2_OneImage::QualityMatch(double aDif)
{
    return mLKM.QualityMatch(aDif,this);
}


void cTAj2_OneImage::SetDefQualityMatch(eTypeMatch aTM)
{
   mDQM = aTM;
}

eTypeMatch  cTAj2_OneImage::DefQualityMatch()
{
   return mDQM;
}

    //================

class cCmpT0Im
{
     public :
       bool operator()(const cTAj2_OneImage *  aI1,const cTAj2_OneImage *  aI2)
       {
             return aI1->T0() < aI2->T0();
       }
};



/**************************************************************/
/*                                                            */
/*               cTAj2_OneLayerIm                             */
/*                                                            */
/**************************************************************/

cTAj2_OneLayerIm::cTAj2_OneLayerIm(cAppli_Traj_AJ & anAppli,const cTrAJ2_SectionImages & aSIm) :
   mAppli (anAppli),
   mSIm   (aSIm),
   mFinishT0 (false)
{
}

const std::vector<cTAj2_OneImage *> & cTAj2_OneLayerIm::MatchedIms() const
{
   return mMatchedIms;
}

void cTAj2_OneLayerIm::AddMatchedIm(cTAj2_OneImage * anIm)
{
   mMatchedIms.push_back(anIm);
}

void cTAj2_OneLayerIm::ResetMatch()
{
    mMatchedIms.clear();
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
        mIms[aK]->ResetMatch();
}

cTAj2_OneImage * cTAj2_OneLayerIm::ImOfName(const std::string & aName)
{
   return GetEntreeNonVide(mDicIms,aName,"Im of Name");
}


void cTAj2_OneLayerIm::AddIm(const std::string & aName)
{
    cTAj2_OneImage * anIm = new cTAj2_OneImage(mAppli,*this,aName);
    mIms.push_back(anIm);
    mDicIms[aName] = anIm;
}

void  cTAj2_OneLayerIm::InitT0()
{
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
       mIms[aK]->InitT0(*(mIms[0]));
    }
}

const cTrAJ2_SectionImages & cTAj2_OneLayerIm::SIm() const
{
   return mSIm;
}

int cTAj2_OneLayerIm::NbIm() const
{
   return mIms.size();
}

cTAj2_OneImage * cTAj2_OneLayerIm::KthIm(int aK) const
{
   return mIms.at(aK);
}

std::vector<cTAj2_OneImage *> & cTAj2_OneLayerIm::Ims()
{
   return mIms;
}


void cTAj2_OneLayerIm::FinishT0()
{
   if (mFinishT0) return;

   mFinishT0 = true;
   InitT0();
   cCmpT0Im aCmp;
   std::sort(mIms.begin(),mIms.end(),aCmp);
   InitT0();

}


void cTAj2_OneLayerIm::Finish()
{
/*
   InitT0();
   cCmpT0Im aCmp;
   std::sort(mIms.begin(),mIms.end(),aCmp);
   InitT0();
*/

    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
       mIms[aK]->SetNum(aK);
    }

    if (0)
    {
       for (int aK=0 ; aK<int(mIms.size()) ; aK++)
       {
          std::cout << mIms[aK]->Name() << " " << mIms[aK]->T0() << "\n";
       }
    }
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
