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


class cMain
{
     public :
          cMain(int argc,char ** argv);
          void NoOp() {}

	  Pt2dr ToW(Pt2dr aP) {return mOffset + aP*mScale;}
	  Pt2dr mOffset;
	  REAL  mScale;
	  Video_Win * pW;
};

cMain::cMain(int argc,char ** argv)
{
    mOffset = Pt2dr(100,100);
    mScale = 0.15;


    std::string aNameDir;
    std::string aNameFile;
    LArgMain argObl;
    LArgMain argFac;

    ElInitArgMain
    (
           argc,argv,
           argObl  << EAM(aNameDir)
	           << EAM(aNameFile),
           argFac
    );

    cElXMLTree aTree(aNameDir+aNameFile);


    PtImGrid toPix = aTree.Get("grid_inverse")->GetPtImGrid(aNameDir);
    PtImGrid toPhGr = aTree.Get("grid_directe")->GetPtImGrid(aNameDir);
    
    Pt2dr aPP = toPix.Value(Pt2dr(0,0));
    REAL Eps = 1e-3;
    REAL aFocale =euclid(aPP-toPix.Value(Pt2dr(Eps,0))) / Eps;

    INT Tx = aTree.Get("usefull-frame")->Get("w")->GetUniqueValInt();
    INT Ty = aTree.Get("usefull-frame")->Get("h")->GetUniqueValInt();

    cout << "PP = " << aPP << " FOCALE = " << aFocale << "\n";
    cout << "TXY = " << Tx << "  " << Ty << "\n";

    INT aNB = 30;

    pW = Video_Win::PtrWStd(Pt2di(850,850));

    for (INT aKX =0 ; aKX<=aNB ; aKX++)
       for (INT aKY =0 ; aKY<=aNB ; aKY++)
       {
            Pt2dr aP (Tx *aKX/REAL(aNB),Ty *aKY/REAL(aNB));
	     
	    pW->draw_circle_loc(ToW(aP),2.0,pW->pdisc()(P8COL::red));

	    Pt2dr aP2 = toPhGr.Value(aP);
	    Pt2dr aP3 = aPP + aP2*aFocale;

	    cout << aP2 << (aP3- aP) << "\n";

	    Pt2dr aQ3 = aP + (aP3-aP) * 40;

	 //   pW->draw_circle_loc(ToW(aQ3),2.0,pW->pdisc()(P8COL::green));
	   pW->draw_seg(ToW(aP),ToW(aQ3),pW->pdisc()(P8COL::green));
       }

     getchar();
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
