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

#include "im_tpl/image.h"


class cMain
{
     public :
          cMain(int argc,char ** argv);
          void NoOp() {}

	  Tiff_Im aTifIn;
	  cDbleGrid::cXMLMode mXmlMode;
          cDbleGrid   aGrid;
	  Pt2di       mSzOut;
};

cMain::cMain(int argc,char ** argv) :
	 aTifIn (argv[1]),
	 aGrid  (mXmlMode,argv[2],argv[3])
{
    LArgMain argObl;
    LArgMain argFac;

    ElInitArgMain
    (
           argc-3,argv+3,
           argObl  << EAM(mSzOut),
           argFac
    );


    std::vector<Im2D_U_INT2> aVIn;
    Pt2di aSzIn = aTifIn.sz();

    Output anOutImIn = Output::onul();
    for (int aKCh=0 ; aKCh < 3; aKCh++)
    {
        aVIn.push_back(Im2D_U_INT2(aSzIn.x,aSzIn.y));
	if (aKCh==0)
            anOutImIn = aVIn.back().out();
	else
            anOutImIn =Virgule(anOutImIn, aVIn.back().out());
    }
    ELISE_COPY(aTifIn.all_pts(),aTifIn.in(),anOutImIn);

    std::vector<Im2D_U_INT2> aVRes;
    Fonc_Num aInRes=0;
    for (int aKCh=0 ; aKCh < 3; aKCh++)
    {
         aVRes.push_back(Im2D_U_INT2(mSzOut.x,mSzOut.y));
         TIm2D<U_INT2,INT> aTImOut(aVRes[aKCh]);
         TIm2D<U_INT2,INT> aTImIn(aVIn[aKCh]);

	 for (int anX=0 ; anX<mSzOut.x ;anX++)
	    for (int anY=0 ; anY<mSzOut.y ;anY++)
	    {
		    Pt2di aP(anX,anY);
		    aTImOut.oset
                    (
		        aP,
			aTImIn.get(aGrid.Direct(aP),0)
                    );
	    }
	if (aKCh==0)
            aInRes = aVRes.back().in();
	else
            aInRes =Virgule(aInRes, aVRes.back().in());
    }


    std::string aNameOut("toto.tif");
    Tiff_Im aTifOut
	    (
	        aNameOut.c_str(),
                mSzOut,
		GenIm::u_int2,
		Tiff_Im::No_Compr,
		Tiff_Im::RGB
	    );
    ELISE_COPY(aTifOut.all_pts(),aInRes,aTifOut.out());
}

int main(int argc,char ** argv)
{
	cMain aMain(argc,argv);
        aMain.NoOp();
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
