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

#include "Digeo.h"

//#define __DEBUG_DIGEO_OLD_REFINE

/****************************************/
/*                                      */
/*             cTplOctDig               */
/*                                      */
/****************************************/

template <class Type>  
cTplOctDig<Type>::cTplOctDig
(
      cOctaveDigeo  * anOctUp,
      GenIm::type_el aTel,
      cImDigeo & aIm,
      int aNiv,
      Pt2di aSzMax
) :
   cOctaveDigeo (anOctUp,aTel,aIm,aNiv,aSzMax),
   mCube        (0)
{

}
   

template <class Type> 
    cOctaveDigeo* cTplOctDig<Type>::AllocDown
                       (
                            GenIm::type_el aTypEl,
                            cImDigeo &     aIm,
                            int            aNiv,
                            Pt2di          aSzMax
                       )
{
   // return new cTplOctDig<Type>(this,aTypEl,aIm,aNiv,aSzMax);
   return cOctaveDigeo::AllocGen(this,aTypEl,aIm,aNiv,aSzMax);
}

void cOctaveDigeo::ResizeAllImages(const Pt2di & aP)
{
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
       mVIms[aK]->ResizeImage(aP);
}



template <class Type>  
void  cTplOctDig<Type>::DoSiftExtract(int aK,const cSiftCarac & aSC)
{   
      if (! OkForSift(aK))
      {
          std::cout << "For k= " << aK << "\n";
          ELISE_ASSERT(false,"Bad K for DoSiftExtract");
      }
      
      #ifdef __DEBUG_DIGEO_OLD_REFINE
	 mVTplIms[aK]->ExtractExtremaDOG_old
	 (
	     aSC,
	     *(mVTplIms[aK-1]),
	     *(mVTplIms[aK+1]),
	     *(mVTplIms[aK+2])
	 );
      #else
	 mVTplIms[aK]->ExtractExtremaDOG( aSC, *(mVTplIms[aK-1]), *(mVTplIms[aK+1]) );
      #endif
}




template <class Type>  
void  cTplOctDig<Type>::DoSiftExtract(int aK)
{
   DoSiftExtract(aK,*mAppli.RequireSiftCarac());
}



template <class Type>  
void  cTplOctDig<Type>::DoSiftExtract(const cSiftCarac & aSC)
{

   for (int aK=1 ; (aK+2) < int(mVTplIms.size()) ; aK++)
   {
      DoSiftExtract(aK,aSC);
   }
   
}


template <class Type>  
cImInMem * cTplOctDig<Type>::AllocIm(double aResolOctaveBase,int aK,int IndexSigma)
{
    return AllocTypedIm(aResolOctaveBase,aK,IndexSigma);
}

template <class Type>
cImInMem *  cTplOctDig<Type>::GetImOfSigma(double aSig)
{
   return TypedGetImOfSigma(aSig);
}

template <class Type>
cTplImInMem<Type> *  cTplOctDig<Type>::TypedGetImOfSigma(double aSig)
{
  cTplImInMem<Type> * aRes =0;
  double  aBestSc = 1e9;
  
  for (int aK=0  ; aK<int(mVTplIms.size()) ; aK++)
  {
      cTplImInMem<Type> * aIm = mVTplIms[aK];
      double  aSc = ElAbs(aSig-aIm->ROct());
      if (aSc <aBestSc)
      {
          aBestSc = aSc;
          aRes = aIm;
      }
  }
  return aRes;
}


template <class Type>  
cTplImInMem<Type> * cTplOctDig<Type>::AllocTypedIm(double aResolOctaveBase,int aK,int IndexSigma)
{
	cTplImInMem<Type> *   aRes = new  cTplImInMem<Type>(mIm,mSzMax,mType,*this,aResolOctaveBase,aK,IndexSigma);

	if ( !mVTplIms.empty() ) aRes->SetMereSameDZ(mVTplIms.back());

	mVTplIms.push_back(aRes);
	mVIms.push_back(aRes);

	return aRes;
}

template <class Type>
void cTplOctDig<Type>::PostPyram() 
{
	if ( mVTplIms.size()==0 ) return;

	for ( size_t aKIm=0; aKIm<mVTplIms.size(); aKIm++ )
		mVDatas.push_back(mVTplIms[aKIm]->TIm().data());

	// compute differences of gaussians
	for ( size_t aKIm=0 ; aKIm<mVTplIms.size()-1; aKIm++ )
		mVTplIms[aKIm]->computeDoG( *mVTplIms[aKIm+1] );

	mCube = &(mVDatas[0]);
}

template <class Type>
Type *** cTplOctDig<Type>::Cube(){ return mCube; }

template <class Type>
cImInMem * cTplOctDig<Type>::FirstImage() { return TypedFirstImage(); }

template <class Type>
cTplImInMem<Type> * cTplOctDig<Type>::TypedFirstImage() { return mVTplIms[0]; }

template <class Type>  cTplOctDig<U_INT2> * cTplOctDig<Type>::U_Int2_This()
{
   if ( mType==GenIm::u_int2 ) return reinterpret_cast<cTplOctDig<U_INT2> *> (this);
   return NULL;
}

template <class Type>  cTplOctDig<REAL4> * cTplOctDig<Type>::REAL4_This()
{
	if ( mType==GenIm::real4 ) return reinterpret_cast<cTplOctDig<REAL4> *> (this);
   return NULL;
}

template <class Type> const std::vector<cTplImInMem<Type> *> &  cTplOctDig<Type>::VTplIms() const { return mVTplIms; }

// INSTANTIATION FORCEE

InstantiateClassTplDigeo(cTplOctDig)


/****************************************/
/*                                      */
/*             cOctaveDigeo             */
/*                                      */
/****************************************/

cOctaveDigeo::cOctaveDigeo(cOctaveDigeo * anOctUp,GenIm::type_el aType,cImDigeo & anIm,int aNiv,Pt2di aSzMax) :
   mType     (aType),
   mIm       (anIm),
   mAppli    (mIm.Appli()),
   mOctUp    (anOctUp),
   mNiv      (aNiv),
   mSzMax    (aSzMax),
   mNbImOri  (-1),
   mBoxImCalc  (mIm.BoxImCalc()._p0/mNiv,mIm.BoxImCalc()._p1/mNiv),
   mTrueSamplingPace( aNiv*anIm.Resol() )
{
}

cOctaveDigeo * cOctaveDigeo::OctUp() {return mOctUp;}

void cOctaveDigeo::SetBoxInOut(const Box2di & aBoxIn,const Box2di & aBoxOut)
{
   mBoxCurIn = Box2dr(Pt2dr(aBoxIn._p0)/mNiv,Pt2dr(aBoxIn._p1)/mNiv);
   mBoxCurOut = Box2di(aBoxOut._p0/mNiv,aBoxOut._p1/mNiv);
}

const Box2di  &  cOctaveDigeo::BoxImCalc () const { return mBoxImCalc; }
const Box2dr  &  cOctaveDigeo::BoxCurIn  () const { return mBoxCurIn; }
const Box2di  &  cOctaveDigeo::BoxCurOut () const { return mBoxCurOut; }

Pt2dr  cOctaveDigeo::ToPtImCalc(const Pt2dr& aP0) const
{
   return aP0*double(mNiv) + Pt2dr(mIm.P0Cur());
}

Pt2dr  cOctaveDigeo::ToPtImR1(const Pt2dr& aP0) const
{
   return ToPtImCalc(aP0) *mIm.Resol();
}

bool cOctaveDigeo::Pt2Sauv(const Pt2dr& aP0) const
{
	// __DEL
	return true;

   //return mIm.PtResolCalcSauv(ToPtImCalc(aP0));
}

int cOctaveDigeo::NbImOri() const
{
   ELISE_ASSERT(mNbImOri>0,"cOctaveDigeo::NbIm");
   return mNbImOri;
}

int cOctaveDigeo::lastLevelIndex() const { return mLastLevelIndex; }

void cOctaveDigeo::SetNbImOri(int aNbImOri)
{
   mNbImOri = aNbImOri;
   mLastLevelIndex = aNbImOri+2;
}

const std::vector<cImInMem *> &  cOctaveDigeo::VIms() const { return mVIms; }

std::vector<cImInMem *> &  cOctaveDigeo::VIms() { return mVIms; }

bool cOctaveDigeo::OkForSift(int aK) const
{
  return     (aK>=1) 
          && (aK+2<int(mVIms.size()))
          && mAppli.SiftCarac()
          && mAppli.Params().PyramideGaussienne().IsInit();
}


void cOctaveDigeo::DoAllExtract(int aK)
{
	mVIms.at(aK)->featurePoints().clear();
	if ( OkForSift(aK) ) DoSiftExtract( aK, *(mAppli.SiftCarac()) );
}

void cOctaveDigeo::DoAllExtract()
{
	for (int aKIm=0 ; aKIm<int(mVIms.size()) ; aKIm++)
		DoAllExtract(aKIm);
}

REAL8 cOctaveDigeo::GetMaxValue() const{ return mIm.GetMaxValue(); }

int cOctaveDigeo::NbIm() const { return (int)mVIms.size(); }
cImInMem * cOctaveDigeo::KthIm(int aK) const { return mVIms.at(aK); }
int  cOctaveDigeo::Niv() const { return mNiv; }

double cOctaveDigeo::trueSamplingPace() const { return mTrueSamplingPace; }

cOctaveDigeo * cOctaveDigeo::AllocGen
           (
                cOctaveDigeo * anUp,
                GenIm::type_el aType,
                cImDigeo & aIm,
                int aNiv,
                Pt2di aSzMax
           )
{
     switch (aType)
     {
         case GenIm::u_int2 : return new cTplOctDig<U_INT2>(anUp,aType,aIm,aNiv,aSzMax);
         case GenIm::real4 :  return new cTplOctDig<REAL4> (anUp,aType,aIm,aNiv,aSzMax);
         default : ELISE_ASSERT(false,"cImInMem");
     }
     return NULL;
}

cOctaveDigeo * cOctaveDigeo::AllocTop
           (
                GenIm::type_el aType,
                cImDigeo & aIm,
                int aNiv,
                Pt2di aSzMax
           )
{
   return AllocGen(0,aType,aIm,aNiv,aSzMax);
}

const cImDigeo & cOctaveDigeo::ImDigeo() const { return mIm; }


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
