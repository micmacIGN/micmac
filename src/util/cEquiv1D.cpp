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



/******************************************************/
/*                                                    */
/*    cFonc1D_HomTr                                   */
/*                                                    */
/******************************************************/

cFonc1D_HomTr::cFonc1D_HomTr
(
       const int & anA,
       const int & aB,
       const int & aC
) :
    mA    (anA),
    mB    (aB),
    mC    (aC)
{
}

int cFonc1D_HomTr::operator()(const int & anX) const
{
   return Elise_div(mA*anX + mB,mC);
}

/******************************************************/
/*                                                    */
/*    cFonc1D_EquivFus                                */
/*                                                    */

/******************************************************/

class cFonc1D_EquivFus : public cFonc1D
{
    public :
         int operator()(const int & aV) const
         {
            return mEq.NumClasse(aV) / mSzF;
         }
         cFonc1D_EquivFus(const cEquiv1D & anEq,const int & aSzF) :
               mEq (anEq),
               mSzF(aSzF)
         {
         }
    private :
         const cEquiv1D & mEq;
         int   mSzF;
};


/******************************************************/
/*                                                    */
/*    cFonc1D_EquivTrans                              */
/*                                                    */
/******************************************************/

class cFonc1D_EquivTrans : public cFonc1D
{
    public :
         int operator()(const int & aV) const
         {
            return mEq.NumClasse(aV+mTr);
         }
         cFonc1D_EquivTrans(const cEquiv1D & anEq,const int & aTr) :
               mEq (anEq),
               mTr(aTr)
         {
         }
    private :
         const cEquiv1D & mEq;
         int   mTr;
};




/******************************************************/
/*                                                    */
/*    cEquiv1D                                        */
/*                                                    */
/******************************************************/

cEquiv1D::cEquiv1D ()
{
}

void cEquiv1D::InitByFusion(const cEquiv1D & anEq,int aSzF)
{
   cFonc1D_EquivFus aFEF(anEq,aSzF) ;
   InitFromFctr(anEq.mV0,anEq.mV1,aFEF);
}

cEquiv1D::cEquiv1D (const cCstrFusion &,const cEquiv1D & anEquiv,int aFus)
{
   InitByFusion(anEquiv,aFus);
}


void cEquiv1D::InitByClipAndTr
     (
         const cEquiv1D & anEq,
         int aHomOfNewV0,
         int aNewV0,
         int aNewV1
     )
{
   cFonc1D_EquivTrans aFET(anEq,aHomOfNewV0-aNewV0);
   InitFromFctr(aNewV0,aNewV1,aFET);
}

void cEquiv1D::InitByDeZoom
     (
         const cEquiv1D & anEq,
         int aDz,
         cVectTr<int> * mLut
     )
{
    Reset(0,0);
    for (int aCl=0 ; aCl<anEq.NbClasses() ; aCl++)
    {
        int aV0_In,aV1_In;
        anEq.ClasseOfNum(aV0_In,aV1_In,aCl);

        int aV0_Out = Elise_div(aV0_In,aDz);
        if (aCl==0)
        {
           mV0 = aV0_Out;
           if (mLut)
           {
              mLut->SetDec(-aV0_In);
              mLut->clear();
           }
        }
        mDebOfClasse.push_back(aV0_Out);

        int aV1_Out;
        if (aCl == (anEq.NbClasses()-1))
        {
            aV1_Out =  round_up(aV1_In/double(aDz));
            mV1 = aV1_Out;
            mDebOfClasse.push_back(aV1_Out);
        }
        else
        {
           aV1_Out = Elise_div(aV1_In,aDz);
        }

        if (mLut)
        {
            ELISE_ASSERT
            (
                aV0_Out < aV1_Out,
                "Pb in cEquiv1D::InitByDeZoom"
            );
            for (int aV= aV0_In ; aV< aV1_In ; aV++)
            {
                mLut->push_back
                (
                   ElMax(aV0_Out,ElMin(Elise_div(aV,aDz),aV1_Out-1))
                );
            }
        }
        for (int aV= aV0_Out ; aV< aV1_Out ; aV++)
        {
            mNumOfClasse.push_back(aCl);
        }
        mNbClasses++;
    }
}

void cEquiv1D::Reset(int aV0,int aV1)
{
   mV0 =aV0;
   mV1 =aV1;
   mNumOfClasse.clear();
   mDebOfClasse.clear();
   mNbClasses =0;
}


void cEquiv1D::InitFromFctr
(
             int aV0,
             int aV1,
             const cFonc1D & aFctr
) 
{
   Reset(aV0,aV1);
   mDebOfClasse.push_back(aV0);
   int aLastCl =  aFctr(aV0);
   mNumOfClasse.push_back(mNbClasses);

   for (int aV=aV0+1; aV<aV1 ;aV++)
   {
       int aCl =  aFctr(aV) ;
       if (aCl != aLastCl)
       {
           aLastCl = aCl;
           mDebOfClasse.push_back(aV);
           mNbClasses++;
       }
       mNumOfClasse.push_back(mNbClasses);
   }
   mNbClasses ++;
   mDebOfClasse.push_back(aV1);
}

int cEquiv1D::SzMaxClasses() const
{
   int aRes = 0;
   for (int aNCL = 0; aNCL<mNbClasses ; aNCL++)
      aRes = ElMax(aRes,mDebOfClasse[aNCL+1]-mDebOfClasse[aNCL]);

  return aRes;
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
