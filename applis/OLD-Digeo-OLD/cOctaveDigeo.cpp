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
#include "Digeo.h"

namespace NS_ParamDigeo
{

/****************************************/
/*                                      */
/*             cTplOctDig               */
/*                                      */
/****************************************/

template <class Type>  
cTplOctDig<Type>::cTplOctDig
(
      GenIm::type_el aTel,
      cImDigeo & aIm,
      int aNiv,
      Pt2di aSzMax
) :
   cOctaveDigeo (aTel,aIm,aNiv,aSzMax),
   mCube        (0),
   mImBase      (new  cTplImInMem<Type>(mIm,mSzMax,mType,*this,0,-1,-1))
{
}


template <class Type>  
void  cTplOctDig<Type>::DoSiftExtract(const cSiftCarac & aSC)
{

   for (int aK=1 ; (aK+2) < int(mVTplIms.size()) ; aK++)
   {
      mVTplIms[aK]->ExtractExtremaDOG
      (
          aSC,
          *(mVTplIms[aK-1]),
          *(mVTplIms[aK+1]),
          *(mVTplIms[aK+2])
      );
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
  if (!mVTplIms.empty())
  {
     aRes->SetMereSameDZ(mVTplIms.back());
  }
  else
  {
     aRes->SetMereSameDZ(mImBase);
  }

  aRes->SetOrigOct(mImBase);
  mVTplIms.push_back(aRes);
  mVIms.push_back(aRes);

  return aRes;
}


template <class Type>
void cTplOctDig<Type>::PostPyram() 
{
    for (int aKIm=0 ; aKIm<int(mVTplIms.size()) ; aKIm++)
        mVDatas.push_back(mVTplIms[aKIm]->TIm().data());

    mCube = &(mVDatas[0]);
}

template <class Type>
Type *** cTplOctDig<Type>::Cube()
{
   return mCube;
}

template <class Type>
   cImInMem * cTplOctDig<Type>::ImBase()
{
    return TypedImBase();
}


template <class Type>
     cTplImInMem<Type> * cTplOctDig<Type>::TypedImBase()
{
   return mImBase;
}



// INSTANTIATION FORCEE

InstantiateClassTplDigeo(cTplOctDig)


/****************************************/
/*                                      */
/*             cOctaveDigeo             */
/*                                      */
/****************************************/

cOctaveDigeo::cOctaveDigeo(GenIm::type_el aType,cImDigeo & anIm,int aNiv,Pt2di aSzMax) :
   mType  (aType),
   mIm    (anIm),
   mNiv   (aNiv),
   mSzMax (aSzMax),
   mNbImOri  (-1)
{
}


int cOctaveDigeo::NbImOri() const
{
   ELISE_ASSERT(mNbImOri>0,"cOctaveDigeo::NbIm");
   return mNbImOri;
}

void cOctaveDigeo::SetNbImOri(int aNbImOri)
{
   mNbImOri = aNbImOri;
}

/*
void cOctaveDigeo::AddIm(cImInMem * aIm)
{
     mVIms.push_back(aIm);
}
*/

int cOctaveDigeo::NbIm() const { return mVIms.size(); }
cImInMem * cOctaveDigeo::KthIm(int aK) const { return mVIms.at(aK); }
int  cOctaveDigeo::Niv() const { return mNiv; }

cOctaveDigeo * cOctaveDigeo::Alloc
           (
                GenIm::type_el aType,
                cImDigeo & aIm,
                int aNiv,
                Pt2di aSzMax
           )
{
     switch (aType)
     {
         case GenIm::u_int1 : return new cTplOctDig<U_INT1>(aType,aIm,aNiv,aSzMax);
         case GenIm::u_int2 : return new cTplOctDig<U_INT2>(aType,aIm,aNiv,aSzMax);
         case GenIm::int4 :   return new cTplOctDig<INT>(aType,aIm,aNiv,aSzMax);
         case GenIm::real4 :  return new cTplOctDig<REAL4>(aType,aIm,aNiv,aSzMax);

         default :
           ;
     }

     ELISE_ASSERT(false,"cImInMem");
     return 0;
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
