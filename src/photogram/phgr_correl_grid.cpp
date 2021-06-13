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



/**********************************************************************/
/*                                                                    */
/*                         cElemEqCorrelGrid                          */
/*                                                                    */
/**********************************************************************/

cElemEqCorrelGrid::cElemEqCorrelGrid
(
     cSetEqFormelles & aSet,
     INT               aNum,
     bool              GenCode
)  :
   mNN           (ToString(aNum)),
   mInterv       (std::string("ZCorrel")+mNN,aNum,aNum+1,false),
   mNameGr1      (std::string("Gr1_")+mNN),
   mNameGr2of0   (std::string("Gr2of0_")+mNN),
   mNameDGr2Dz   (std::string("DGr2Dz")+mNN),
   mNameZCur     (std::string("ZCur")+mNN),
   mGr1          (0.0,mNameGr1),
   mGr2of0       (0.0,mNameGr2of0),
   mDGr2Dz       (0.0,mNameDGr2Dz),
   mZCur         (0.0,mNameZCur),
   mAdrGr1       (0),
   mAdrGr2Of0    (0),
   mAdrDGr2Dz    (0),
   mAdrZCur      (0),
   mZ            (kth_coord(aNum)),
   mGr2ofZ       (mGr2of0 + mDGr2Dz * mZ),
   mGr2ofZCur    (mGr2of0 + mDGr2Dz *mZCur)
{
}

void cElemEqCorrelGrid::InitAdr(cElCompiledFonc * pECF,bool Im2MoyVar)
{
    mAdrGr1 = pECF->RequireAdrVarLocFromString(mNameGr1);   
    mAdrGr2Of0 = pECF->RequireAdrVarLocFromString(mNameGr2of0);   
    mAdrDGr2Dz = pECF->RequireAdrVarLocFromString(mNameDGr2Dz);
    if (! Im2MoyVar)
        mAdrZCur = pECF->RequireAdrVarLocFromString(mNameZCur);
}



/**********************************************************************/
/*                                                                    */
/*                           cEqCorrelGrid                            */
/*                                                                    */
/**********************************************************************/

std::string cEqCorrelGrid::NameType(INT aNbPix,bool Im2MoyVar)
{
   return   std::string("cEqCorrelGrid_")
          + ToString(aNbPix)
          + std::string(Im2MoyVar ? "_Im2Var" : "_Im2Fixe");
}


cEqCorrelGrid::cEqCorrelGrid
(
    cSetEqFormelles & aSet,
    INT aNbPix,
    bool Im2MoyVar,
    bool GenCode
)  :
   mSet            (aSet),
   mNbPix          (aNbPix),
   mVPix           (),
   mIm2MoyVar      (Im2MoyVar),
   mGenCode        (GenCode),
   mNameType       (NameType(aNbPix,Im2MoyVar)),
   mFoncEq         (0)
{
    Fonc_Num fS1  = 0;
    Fonc_Num fS11 = 0;
    Fonc_Num fS2  = 0;
    Fonc_Num fS22 = 0;

    for (INT aKPix=0 ; aKPix<mNbPix ; aKPix++)
    {
        mVPix.push_back(cElemEqCorrelGrid(aSet,aKPix,GenCode));
        cElemEqCorrelGrid & aPix = mVPix.back();
        mLInterv.AddInterv(aPix.mInterv);
        Fonc_Num fI1 = aPix.mGr1;
        Fonc_Num fI2 = Im2MoyVar ? aPix.mGr2ofZ :  aPix.mGr2ofZCur;
        fS1 = fS1 + fI1;
        fS11 = fS11 + Square(fI1);
        fS2 = fS2 + fI2;
        fS22 = fS22 + Square(fI2);
    }

    if (GenCode)
    {
        fS1  = fS1 /mNbPix;
        fS11 = sqrt(fS11/mNbPix - Square(fS1)+1e-3);
        fS2  = fS2 /mNbPix;
        fS22 = sqrt(fS22/mNbPix - Square(fS2)+1e-3);

        std::vector<Fonc_Num>  aVEqs; 
        for (INT aKPix=0 ; aKPix<mNbPix ; aKPix++)
	{
           cElemEqCorrelGrid & aPix = mVPix[aKPix];
           Fonc_Num fI1 = (aPix.mGr1-fS1) / fS11;
           Fonc_Num fI2 = (aPix.mGr2ofZ -fS2) / fS22;
	   aVEqs.push_back(fI1-fI2);
	}
        cElCompileFN::DoEverything
        (
               "src/GC_photogram/",
                mNameType,
                aVEqs,
                mLInterv
        );
	return;
    }
    mFoncEq = cElCompiledFonc::AllocFromName(mNameType);
    ELISE_ASSERT(mFoncEq!=0,"Cannot Find Fctr for cEqCorrelGrid");
    mFoncEq->SetMappingCur(mLInterv,&mSet);

    for (INT aKPix=0 ; aKPix<mNbPix ; aKPix++)
        mVPix[aKPix].InitAdr(mFoncEq,Im2MoyVar);
}

cElCompiledFonc * cEqCorrelGrid::Fctr()
{
    return mFoncEq;
}

cElemEqCorrelGrid & cEqCorrelGrid::KthElem(INT aK)
{
    ELISE_ASSERT((aK>=0) && (aK<mNbPix),"Inc in cEqCorrelGrid::KthElem");
    return mVPix[aK];
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
