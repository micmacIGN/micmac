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

#include "FusionNuage.h"

std::string cAppliFusionNuage::NameFileInput(const std::string & aNameIm,const std::string aPost)
{
   return mDir + "Fusion-0/NuageRed" + aPost ;
}


cAppliFusionNuage::cAppliFusionNuage
(
      const cParamFuNu  & aParam,
      const std::string & aDir,
      const std::string & aPat,
      const std::string & aKeyI2N,
      const std::string & aKeyI2ISec,
      const std::string & aKeyI2BsH
)  :
   mParam  (aParam),
   mDir    (aDir),
   mPat    (aPat),
   mKeyI2N (aKeyI2N),
   mICNM   (cInterfChantierNameManipulateur::BasicAlloc(mDir)),
   mGr     (),
   mAllGr  (),
   mFlagAIn (mGr.alloc_flag_arc()),
   mFlagATested (mGr.alloc_flag_arc()),
   mGrArcIn     (mAllGr,mFlagAIn),
   mGrArcTested     (mAllGr,mFlagATested)
{

    //=============================================================
    //        CONSTRUCTION DES SOMMETS DU GRAPHE
    //=============================================================

    const  std::vector<std::string>  * aSet = mICNM->Get(aPat);
    int aNbIm = int(aSet->size());
    for (int aKN=0 ; aKN<aNbIm ; aKN++)
    {
        const std::string & aNameIm = (*aSet)[aKN];
        std::string  aNameCl = mICNM->Assoc1To1(mKeyI2N,aNameIm,true);
        if (ELISE_fp::exist_file(aNameCl))
        {
            ElTimer Chrono;
            cElNuage3DMaille * aCloud = cElNuage3DMaille::FromFileIm(aNameCl);
            Im2D_U_INT1 aBSH(1,1);
            if (aKeyI2BsH != "")
            {
                 std::string aNameBsH = mICNM->Assoc1To1(aKeyI2BsH,aNameIm,true);
                 if (ELISE_fp::exist_file(aNameBsH))
                     aBSH = Im2D_U_INT1::FromFileStd(aNameBsH);
                  else
                  {
                      std::cout << "WaaaaarNNNnoBsH  " << aNameBsH  << "\n";
                      std::cout << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n";
                  }
            }

            std::string aNameImSec = mICNM->Assoc1To1(aKeyI2ISec,aNameIm,true);
            cImSecOfMaster  aISec = StdGetObjFromFile<cImSecOfMaster>
                                    (
                                        aNameImSec,
                                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                       "ImSecOfMaster",
                                       "ImSecOfMaster"
                                    );
              ELISE_ASSERT(aISec.ISOM_AllVois().IsInit(),"ImSecOfMaster :: no ISOM_AllVois");


            std::cout << Chrono.uval() << aBSH.sz() << "\n";
            // cFNuAttrSom anAttr(aCloud,aNameIm);
            tFNuSom & aSom = mGr.new_som(new cFNuAttrSom(aCloud,aISec,aNameIm,this,aBSH));
            mMapSom[aNameIm] = &aSom;
            mVSom.push_back(&aSom);
            std::cout << "To do " << aNbIm - aKN <<  " TTot " << Chrono.uval() << "\n";
        }
        else
        {
            std::cout << "WaaaaarNNN no  " << aNameCl  << "\n";
            std::cout << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n";
        }
    }
    mNbSom = (int)mVSom.size();

    //=============================================================
    //        CONSTRUCTION DES ARCS DU GRAPHE
    //=============================================================


   // std::vector<tFNuArc *> aVArc;
    for (int aK1=0 ; aK1 <mNbSom ; aK1++)
    {
        tFNuSom * aS1 = mVSom[aK1];
        const std::list<cISOM_Vois> & aLV = aS1->attr()->ListVoisInit();
        for 
        (
              std::list<cISOM_Vois>::const_iterator itV=aLV.begin();
              itV!=aLV.end();
              itV++
        )
        {
              tFNuSom *aS2 = mMapSom[itV->Name()];
              if (aS2)
              {
                 TestNewAndSet(aS1,aS2);
              }
        }
    }
}

tFNuArc * cAppliFusionNuage::TestNewAndSet(tFNuSom *aS1,tFNuSom *aS2)
{
   if (aS1==aS2) return 0;
   if (aS2>aS1) ElSwap(aS1,aS2);

   std::pair<tFNuSom *,tFNuSom *>  aPair(aS1,aS2);

   std::set<std::pair<tFNuSom *,tFNuSom *> >::iterator it = mTestedPair.find(aPair);

   if (it != mTestedPair.end()) 
      return 0;

   mTestedPair.insert(aPair);

   if (aS1->attr()->IsArcValide(aS2->attr()))
   {
   }
   return 0;
}

cInterfChantierNameManipulateur* cAppliFusionNuage::ICNM()
{
   return mICNM;
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
