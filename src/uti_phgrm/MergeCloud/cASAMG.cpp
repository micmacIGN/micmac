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


#include "MergeCloud.h"


//     void ComputeIncidGradProf();



cASAMG::cASAMG(cAppliMergeCloud * anAppli,cImaMM * anIma)  :
   mAppli     (anAppli),
   mIma       (anIma),
   mStdN      (cElNuage3DMaille::FromFileIm(mAppli->NameFileInput(anIma,".xml"))),
   mMasqN     (mStdN->ImDef()),
   mTMasqN    (mMasqN),
   mImCptr    (1,1),
   mTCptr     (mImCptr),
   mSz        (mStdN->SzUnique()),
   mImIncid   (mSz.x,mSz.y),
   mTIncid    (mImIncid),
   mMasqHigh  (mSz.x,mSz.y),
   mTMPH      (mMasqHigh),
   mMasqPLow  (mSz.x,mSz.y),
   mTMPL      (mMasqPLow),
   mSSIma     (mStdN->DynProfInPixel() *  mAppli->Param().ImageVariations().SeuilStrictVarIma()),
   mISOM      (StdGetISOM(anAppli->ICNM(),anIma->mNameIm,anAppli->Ori()))
{
// std::cout << "AAAAAAAAAAAAAAAAAAaa\n"; getchar();
   // mImCptr  => Non pertinent en mode envlop, a voir si reactiver en mode epi
   // Im2D_U_INT1::FromFileStd(mAppli->NameFileInput(anIma,"CptRed.tif"))),

   // ComputeIncidAngle3D();
   ComputeIncidGradProf();
   double aPente = mAppli->Param().PenteRefutInitInPixel().Val();
   ComputeIncidKLip(mMasqN.in_proj(),aPente,mMasqHigh);
   ComputeIncidKLip(mMasqN.in_proj(),aPente*2,mMasqPLow);
   
   
   Video_Win * aW = mAppli->TheWinIm(mSz);

   ComputeSubset(mAppli->Param().NbPtsLowResume(),mLowRN);

   if (mAppli->Param().VisuGrad().Val() && aW)
   {
      aW->set_title(mIma->mNameIm.c_str());

      ELISE_COPY
      (
             mImIncid.all_pts(),
             Virgule
             (
                  mImIncid.in(),
                  mImIncid.in() *  ! mMasqHigh.in(),
                  mImIncid.in() *  ! mMasqPLow.in()
             ),
             aW->orgb()
      );

       aW->clik_in();
   }
}

cImaMM * cASAMG::IMM() {return  mIma;}

const cImSecOfMaster &  cASAMG::ISOM() const
{
   return mISOM;
}

void  cASAMG::AddCloseVois(cASAMG * anA)
{
   mCloseNeigh.push_back(anA);
}

const cOneSolImageSec &  cASAMG::SolOfCostPerIm(double aCostPerIm)
{
   double aBestGain = -1e10;
   const cOneSolImageSec * aSol=0;
   for 
   (
        std::list<cOneSolImageSec>::const_iterator itS=mISOM.Sols().begin() ;
        itS !=mISOM.Sols().end() ;
        itS++
   )
   {
       double aGain = itS->Coverage() - aCostPerIm*itS ->Images().size();
       if (aGain > aBestGain)
       {
           aBestGain = aGain;
           aSol = &(*itS);
       }
   }
   return *aSol;
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
