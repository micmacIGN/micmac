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


#include "CompColore.h"

using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;



#define DEF_OFSET -12349876

int main(int argc,char ** argv)
{
   ELISE_ASSERT(argc>=4,"Not Enough arg");
   std::vector<Im2D_REAL4> aVIm;
   std::vector<TIm2D<REAL4,REAL8> > aVTIm;

   std::string aDir = argv[1];
   for (int aK= 2 ; aK< argc;  aK++)
   {
       std::cout << "Begin Push " << (aK-2) << "\n";
       Im2D_REAL4::ReadAndPushTif(aVIm,Tiff_Im::StdConv(aDir+argv[aK]));
   }

    int aNbIm = aVIm.size()-1;

    Im2D_REAL4 aCible = aVIm.back();
    TIm2D<REAL4,REAL8> aTCible(aCible);
    Pt2di aSz = aCible.sz();
    
    if (0)
    {
       aVIm.push_back(Im2D_REAL4(aSz.x,aSz.y,1));
       aNbIm++;
       ElSwap(aVIm[aNbIm-1],aVIm[aNbIm]);
    }

    for (int aK=0 ; aK<= aNbIm ; aK++)
    {
       aVTIm.push_back(TIm2D<REAL4,REAL8>(aVIm[aK]));
    }

    Pt2di aP;
    ElMatrix<double> aMat(aNbIm,aNbIm,0);
    ElMatrix<double> aCol(1,aNbIm,0);
    for (aP.y= 0 ; aP.y<aSz.y ; aP.y++)
    {
       if (aP.y%10==0) 
          std::cout << "Y=" << aP.y << "\n";
       for (aP.x= 0 ; aP.x<aSz.x ; aP.x++)
       {
           bool Ok= true;
	   for (int aK=0 ; aK<= aNbIm ; aK++)
	   {
	       Ok = Ok && (aVTIm[aK].get(aP)!=0);
	   }
	   if (Ok)
	   {
	      for (int aKy=0 ; aKy< aNbIm ; aKy++)
	      {
                 aCol(0,aKy) += aVTIm[aKy].get(aP) * aTCible.get(aP) ;
	         for (int aKx=0 ; aKx< aNbIm ; aKx++)
	         {
		     aMat(aKx,aKy) += aVTIm[aKx].get(aP) *aVTIm[aKy].get(aP);
	         }
	      }
	   }
       }
    }

    for (int aKy=0 ; aKy< aNbIm ; aKy++)
    {
	for (int aKx=0 ; aKx< aNbIm ; aKx++)
	{
	   std::cout <<  aMat(aKx,aKy) << " " ;
	}
	std::cout << "\n";
    }

    ElMatrix<double>  aRes = gaussj(aMat) * aCol;

    Fonc_Num aF =0;
    for (int aK=0 ; aK<aNbIm ; aK++)
    {
        std::cout << "Coeeff " << aRes(0,aK) << "\n";
	aF = aF + aRes(0,aK) * aVIm[aK].in();
    }

    double aVMin,aVMax;
    ELISE_COPY
    (
        aCible.all_pts(),
	aF,
	VMin(aVMin)|VMax(aVMax)
    );
    std::cout << "Min-Max " << aVMin << " " << aVMax << "\n";

    // aF = (aF-aVMin) * (255.0/(aVMax-aVMin));
    Tiff_Im::Create8BFromFonc("toto.tif",aSz,Max(0,Min(255,aF)));
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
