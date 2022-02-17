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

#include "cp_header.h"

cAppliCorrPont::cAppliCorrPont(Pt2di aSz) :
    mSz      (aSz),
    mIm1     (aSz.x,aSz.y),
    mD1      (mIm1.data()),
    mIm2     (aSz.x,aSz.y),
    mD2      (mIm2.data()),
    mIScore  (aSz.x,aSz.y),
    mTIScore (mIScore),
    mIPax    (aSz.x,aSz.y),
    mTIPax   (mIPax),
    mInterp  (new cInterpolSinusCardinal<double>(3))
{
}

void cAppliCorrPont::Init(const std::string & aName,Im2D_REAL8 anIm,Pt2di aDec)
{
    Tiff_Im aFile = Tiff_Im::StdConv(aName);
    ELISE_COPY
    (
         anIm.all_pts(),
	 trans(aFile.in(),aDec),
	 anIm.out()
    );
}


void cAppliCorrPont::InitFiles
     (
         const std::string & aDir,
         const std::string & aFile1,
         const std::string & aFile2,
         Pt2di aDec
     )
{
    Init(aDir+aFile1,mIm1,aDec);
    Init(aDir+aFile2,mIm2,aDec);
}

// Un peu bovin comme rab, mais dans cette appli minimaliste,
// on veut pas etre embeter ...

bool cAppliCorrPont::Inside(const Pt2dr & aP) const
{
   return    (aP.x > 1 ) && (aP.x < (mSz.x-2))
          && (aP.y > 1 ) && (aP.y < (mSz.y-2)) ;
}

double  cAppliCorrPont::Correl(const Pt2dr & aP1,const Pt2dr & aP2,int aSzV)
{
      int aSzK = mInterp->SzKernel();

      Pt2dr aSzW(aSzK+aSzV,aSzK+aSzV);

      if (    (!Inside(aP1-aSzW))
           || (!Inside(aP1+aSzW))
           || (!Inside(aP2-aSzW))
           || (!Inside(aP2+aSzW))
	 )
         return TheCorrelOut;

     Pt2dr aD;
     RMat_Inertie aMat;
     for (aD.x = -aSzV ; aD.x<= aSzV ; aD.x++)
     {
         for (aD.y = -aSzV ; aD.y<= aSzV ; aD.y++)
	 {
	     double aV1 = mInterp->GetVal(mD1,aP1+aD);
	     double aV2 = mInterp->GetVal(mD2,aP2+aD);
	     aMat.add_pt_en_place(aV1,aV2);
	 }
     }

     return aMat.correlation(1e-10);
}


double cAppliCorrPont::PxMaxCorr
       (
           double & aCorMax,
           const Pt2dr & aP12,
	   int aSzV,
           double aStep,
	   double aMinPax,
	   double aMaxPax
	)
{
    aCorMax = TheCorrelOut;
    double  aPMaxCorr = DefPaxOut;
    for (double aPax = aMinPax ; aPax <= aMaxPax ; aPax +=aStep)
    {
        double aCor = Correl(aP12,aP12+Pt2dr(aPax,0.0),aSzV);
	if (aCor==TheCorrelOut)
	{
	    aCorMax = TheCorrelOut;
	    return DefPaxOut;
	}
	if (aCor > aCorMax)
	{
	     aCorMax = aCor;
	     aPMaxCorr = aPax;
	}
    }

    return aPMaxCorr;
}


void   cAppliCorrPont::MakeCorrPonct(int aSzV,double aStep,double aMinPax,double aMaxPax)
{
    Pt2di aP;
    for (aP.x = 0 ; aP.x <mSz.x ; aP.x++)
    {
        std::cout << "PX " << aP.x << "\n";
        for (aP.y = 0 ; aP.y <mSz.y ; aP.y++)
        {
	    double aCor;
	    double aPax = PxMaxCorr(aCor,aP,aSzV,aStep,aMinPax,aMaxPax);
	    mTIPax.oset(aP,aPax);
	    mTIScore.oset(aP,aCor);
        }
    }
}

Im2D_REAL8  cAppliCorrPont::IPax() { return mIPax; }
Im2D_REAL8  cAppliCorrPont::Im1()  { return mIm1 ; }
Im2D_REAL8  cAppliCorrPont::Im2()  { return mIm2 ; }



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
