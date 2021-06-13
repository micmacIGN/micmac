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
#include "TpPPMD.h"


/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

cTD_Im::cTD_Im(int anX,int anY) :
  mIm  (anX,anY,0.0),
  mSz  (anX,anY),
  mTIm (mIm)
{
}

cTD_Im::cTD_Im(const cTD_Im & aCI) :
   mIm (aCI.mIm),
   mSz (aCI.mSz),
   mTIm (mIm)
{
}

cTD_Im & cTD_Im::operator = (const cTD_Im & aCI)
{
   mIm = aCI.mIm;
   mSz = aCI.mSz;
   mTIm =  TIm2D<double,double>(mIm);

   return *this;
}


cTD_Im cTD_Im::FromString(const std::string & aName)
{
   Tiff_Im aTF = Tiff_Im::StdConvGen(aName,-1,true);
   Pt2di aSzIm = aTF.sz();
   cTD_Im aRes(aSzIm.x,aSzIm.y);

   ELISE_COPY(aRes.mIm.all_pts(),aTF.in(),aRes.mIm.out());

   return aRes;
}

void cTD_Im::Save(const std::string & aName)
{
    Tiff_Im  aTF
             (
                 aName.c_str(),
                 mIm.sz(),
                 GenIm::real4,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero

             );

    ELISE_COPY(mIm.all_pts(),mIm.in(),aTF.out());
}


void  cTD_Im::SaveRGB(const std::string & aName,cTD_Im & aI2,cTD_Im & aI3)
{
    Tiff_Im  aTF
             (
                 aName.c_str(),
                 mIm.sz(),
                 GenIm::real4,
                 Tiff_Im::No_Compr,
                 Tiff_Im::RGB

             );

    ELISE_COPY(mIm.all_pts(),Virgule(mIm.in(),aI2.mIm.in_proj(),aI3.mIm.in_proj()),aTF.out());
}


cTD_Im  cTD_Im::ImageMoy(int aSzW,int aNbIter)
{
   cTD_Im aRes(mSz.x,mSz.y);

   Fonc_Num aF = mIm.in_proj();
   int aNbVois = ElSquare(1+2*aSzW);
   for (int aK=0 ; aK<aNbIter ; aK++)
       aF = rect_som(aF,aSzW) / aNbVois;
   ELISE_COPY(mIm.all_pts(),aF,aRes.mIm.out());

   return aRes;
}

cTD_Im  cTD_Im::ImageReduite(double aFact)
{
   Pt2di aSzR = round_up(Pt2dr(mSz)/aFact);

   cTD_Im aRes(aSzR.x,aSzR.y);


    Fonc_Num aFIn = StdFoncChScale
                 (
                       mIm.in_proj(),
                       Pt2dr(0,0),
                       Pt2dr(aFact,aFact)
                       // aDilXY
                 );
    ELISE_COPY(aRes.mIm.all_pts(),aFIn,aRes.mIm.out());



   return aRes;

}


Pt2di cTD_Im::Sz() const 
{
   return mIm.sz();
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
