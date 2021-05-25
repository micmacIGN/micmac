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

#include "Apero.h"

namespace NS_ParamApero
{


/******************************************************/
/*                                                    */
/*                cImageAppuis                        */
/*                                                    */
/******************************************************/

cImageAppuis::cImageAppuis
(
    const cImageAppuisDense & anIAD,
    cAppliApero &             anAppli
) :
    mIAD   (anIAD),
    mAppli (anAppli),
    mPose  (0),
    mFile  (Tiff_Im::StdConv(mAppli.DC() + anIAD.NameFile())),
    mSz    (mFile.sz()),
    mImP   (mSz.x,mSz.y),
    mTImP  (mImP),
    mImX   (mSz.x,mSz.y),
    mTImX  (mImX),
    mImY   (mSz.x,mSz.y),
    mTImY  (mImY),
    mImZ   (mSz.x,mSz.y),
    mTImZ  (mImZ)
{
   ELISE_COPY
   (
      mFile.all_pts(),
      mFile.in(),
      Virgule
      (
         mImP.out(),
         mImX.out(),
         mImY.out(),
         mImZ.out()
      )
   );
}

std::list<Appar23> cImageAppuis::AppuisFromHom
                   (
                      const ElPackHomologue & aPackH,
                      Pt3dr & aCDG
                   )
{
   std::list<Appar23> aRes;
   aCDG = Pt3dr(0,0,0);
   double aSomPds = 0;

   for
   (
        ElPackHomologue::const_iterator itH = aPackH.begin();
	itH != aPackH.end();
	itH++
   )
   {
       Pt2dr aP2 = itH->P2();
       Pt2di iP2 = round_down(aP2);
       if (
                 (iP2.x >= 0)
             &&  (iP2.y >= 0)
             &&  (iP2.x < mSz.x-1)
             &&  (iP2.y < mSz.y-1)
             &&  (mTImP.get(iP2+Pt2di(0,0)) > 0)
             &&  (mTImP.get(iP2+Pt2di(1,0)) > 0)
             &&  (mTImP.get(iP2+Pt2di(0,1)) > 0)
             &&  (mTImP.get(iP2+Pt2di(1,1)) > 0)
	  )
       {
           double aPds = mTImP.getr(aP2);
           Pt3dr aPTer
	         (
		      mTImX.getr(aP2),
		      mTImY.getr(aP2),
		      mTImZ.getr(aP2)
		 );
           aSomPds += aPds;
	   aCDG = aCDG + aPTer*aPds;
           aRes.push_back(Appar23(itH->P1(),aPTer));
       }
   }

   ELISE_ASSERT(aSomPds!=0,"Som P in AppuisFromHom");
   aCDG = aCDG / aSomPds;
   return aRes;
}


const std::string & cImageAppuis::NamePose() const
{
   return mIAD.NamePose();
}


/******************************************************/
/*                                                    */
/*              cAppliApero                           */
/*                                                    */
/******************************************************/

void cAppliApero::InitBDAppuisLiaisons()
{
   for 
   (
       std::list<cImageAppuisDense>::const_iterator itIAD=mParam.ImageAppuisDense().begin();
       itIAD!=mParam.ImageAppuisDense().end();
       itIAD++
   )
   {
      if (mDicImAp[itIAD->Id()]!=0)
      {
         std::cout << "For Id = " << itIAD->Id() << "\n";
         ELISE_ASSERT(false,"Multiple Id in ImageAppuisDense");
      }
      mDicImAp[itIAD->Id()] = new cImageAppuis(*itIAD,*this);
   }
}

cImageAppuis * cAppliApero::GetImAppuisOfId(const std::string& anId)
{
  cImageAppuis * aIA = mDicImAp[anId];
  if (aIA==0)
  {
     std::cout << "For name = " << anId << "\n";
     ELISE_ASSERT(false,"Cannot GetImAppuisOfId");
  }

  return aIA;
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
