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

/*
    Etapes pour generer le code :

     - 1 - ecrire le code cote ELISE ! (a base de FoncNum)
     - 2 -  GenCode
     - 3 - rajouter #include "../../src/GC_photogram/cEqObsRotVect_CodGen.cpp"
           dans un des codes phgr_or_code_genX.cpp

      -4 -  rajaouetr

                 #include "../../src/GC_photogram/cEqObsRotVect_CodGen.h"

                 AddEntry("cEqObsRotVect_CodGen",cEqObsRotVect_CodGen::Alloc);   
         
            dans phgr_or_code_gen00.cpp
*/

/*********************************************************/
/*                                                       */
/*        cVarEtat_PhgrF                                 */
/*                                                       */
/*********************************************************/

cVarEtat_PhgrF::cVarEtat_PhgrF(const std::string & aName) :
    mName   (aName),
    mAdr    (0)
{
}

void cVarEtat_PhgrF::InitAdr(cElCompiledFonc & aFoncC)
{
   mAdr = aFoncC.RequireAdrVarLocFromString(mName);
}

void cVarEtat_PhgrF::SetEtat(const double & aVal)
{
   ELISE_ASSERT(mAdr!=0,"cVarEtat_PhgrF::SetEtat");
   *mAdr = aVal;
}

double cVarEtat_PhgrF::GetVal() const
{
   ELISE_ASSERT(mAdr!=0,"cVarEtat_PhgrF::SetEtat");
   return *mAdr;
}


Fonc_Num cVarEtat_PhgrF::FN() const
{
   return cVarSpec(0,mName);
}


void cVarEtat_PhgrF::InitAdrSVP(cElCompiledFonc & aFoncC)
{
   mAdr = aFoncC.AdrVarLocFromString(mName);
}

void cVarEtat_PhgrF::SetEtatSVP(const double & aVal)
{
   if (mAdr)
      *mAdr = aVal;
}

/*********************************************************/
/*                                                       */
/*        cP2d_Etat_PhgrF                                */
/*                                                       */
/*********************************************************/

cP2d_Etat_PhgrF::cP2d_Etat_PhgrF (const std::string & aNamePt) :
      mVarX  (aNamePt+ "_x"),
      mVarY  (aNamePt+ "_y"),
      mPtF   (mVarX.FN(),mVarY.FN())
{
}

Pt2d<Fonc_Num>  cP2d_Etat_PhgrF::PtF() const 
{
   return mPtF;
}

void cP2d_Etat_PhgrF::InitAdr(cElCompiledFonc & aFoncC)
{
    mVarX.InitAdr(aFoncC);
    mVarY.InitAdr(aFoncC);
}
void cP2d_Etat_PhgrF::SetEtat(const Pt2dr & aP)
{
    mVarX.SetEtat(aP.x);
    mVarY.SetEtat(aP.y);
}


void cP2d_Etat_PhgrF::InitAdrSVP(cElCompiledFonc & aFoncC)
{
    mVarX.InitAdrSVP(aFoncC);
    mVarY.InitAdrSVP(aFoncC);
}
void cP2d_Etat_PhgrF::SetEtatSVP(const Pt2dr & aP)
{
    mVarX.SetEtatSVP(aP.x);
    mVarY.SetEtatSVP(aP.y);
}


Pt2dr cP2d_Etat_PhgrF::GetVal() const
{
   return  Pt2dr(mVarX.GetVal(),mVarY.GetVal());
}




/*********************************************************/
/*                                                       */
/*        cP3d_Etat_PhgrF                                */
/*                                                       */
/*********************************************************/

cP3d_Etat_PhgrF::cP3d_Etat_PhgrF (const std::string & aNamePt) :
      mVarX  (aNamePt+ "_x"),
      mVarY  (aNamePt+ "_y"),
      mVarZ  (aNamePt+ "_z"),
      mPtF   (mVarX.FN(),mVarY.FN(),mVarZ.FN())
{
}

Pt3d<Fonc_Num>  cP3d_Etat_PhgrF::PtF() const 
{
   return mPtF;
}

void cP3d_Etat_PhgrF::InitAdr(cElCompiledFonc & aFoncC)
{
    mVarX.InitAdr(aFoncC);
    mVarY.InitAdr(aFoncC);
    mVarZ.InitAdr(aFoncC);
}

void cP3d_Etat_PhgrF::SetEtat(const Pt3dr & aP)
{
    mVarX.SetEtat(aP.x);
    mVarY.SetEtat(aP.y);
    mVarZ.SetEtat(aP.z);
}

Pt3dr cP3d_Etat_PhgrF::GetVal() const
{
   return  Pt3dr(mVarX.GetVal(),mVarY.GetVal(),mVarZ.GetVal());
}


/*********************************************************/
/*                                                       */
/*        cMatr_Etat_PhgrF                               */
/*                                                       */
/*********************************************************/

cMatr_Etat_PhgrF::cMatr_Etat_PhgrF
(
    const std::string & aNameEl,
    int aTx,
    int aTy
)  :
   mMatF (aTx,aTy)
{
   for (int anY=0 ; anY<aTy ; anY++)
   {
      for (int anX=0 ; anX<aTx ; anX++)
      {
           std::string aName =   aNameEl 
	                       + "_" + ToString(anX)
			       + "_" + ToString(anY);
            mVF.push_back(cVarEtat_PhgrF(aName));
	    mMatF(anX,anY) = mVF.back().FN();
      }
   }
}

const ElMatrix<Fonc_Num>  & cMatr_Etat_PhgrF::Mat() const
{
   return mMatF;
}

void cMatr_Etat_PhgrF::InitAdr(cElCompiledFonc & aFoncC)
{
   for (int aK=0 ; aK<int(mVF.size()) ; aK++)
       mVF[aK].InitAdr(aFoncC);
}

void cMatr_Etat_PhgrF::SetEtat(const ElMatrix<double> & aM)
{
   ELISE_ASSERT
   (
      (aM.tx()==mMatF.tx()) &&  (aM.ty()==mMatF.ty()),
      "cMatr_Etat_PhgrF::SetEtat"
   );

   int aK=0;
   for (int anY=0 ; anY<aM.ty() ; anY++)
   {
      for (int anX=0 ; anX<aM.tx() ; anX++)
      {
          mVF[aK].SetEtat(aM(anX,anY));
          aK++;
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
