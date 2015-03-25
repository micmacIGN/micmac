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
#include "XML_GEN/all.h"
#include "im_tpl/image.h"

using namespace NS_ParamChantierPhotogram;

/*
  Extrait un modele radial d'une image de points homologues
*/

namespace NS_EvalCompositionGrid
{

cDbleGrid::cXMLMode  aXMLM;

class cGridComp
{
    public :
       cGridComp
       (
           const std::string & aDir,
           const std::string & aName,
	   bool  isDirect
       )  :
          mGrid    (aXMLM,aDir,aName),
	  mDirect  (isDirect)
       {
       }

       const cDbleGrid & Grid() const {return mGrid;}
       Pt2dr operator () (const Pt2dr & aP)
       {
          return mDirect ? mGrid.Direct(aP) : mGrid.Inverse(aP) ;
       }

    private  :
       cDbleGrid     mGrid;
       bool          mDirect;
};

};

using namespace NS_EvalCompositionGrid;
using namespace NS_SuperposeImage;



int main(int argc,char ** argv)
{
    ELISE_ASSERT(argc>=2,"Not Enough arg");
    cEvalComposeGrid aECG = StdGetObjFromFile<cEvalComposeGrid>
                            (
				   argv[1],
			           "include/XML_GEN/SuperposImage.xml",
				   "EvalComposeGrid",
				   "EvalComposeGrid"
			    );
    std::list<cGridComp *> aLG;


    for 
    (
        std::list<cOneGridECG>::const_iterator itG=aECG.OneGridECG().begin();
	itG != aECG.OneGridECG().end();
	itG++
    )
    {
        aLG.push_back(new cGridComp(aECG.Directory().Val(),itG->Name(),itG->Direct()));
    }
 
    const PtImGrid & aPIG = aLG.front()->Grid().GrDir();
    double aResol = aECG.Resol() ;


    Pt2di aSzSR = (aPIG.P1()-aPIG.P0())/aResol;

    Im2D_REAL8 aImDx(aSzSR.x,aSzSR.y);
    Im2D_REAL8 aImDy(aSzSR.x,aSzSR.y);


    Pt2di aPInd;
    for (aPInd.x=0 ; aPInd.x<aSzSR.x ; aPInd.x++)
    {
        for (aPInd.y=0 ; aPInd.y<aSzSR.y ; aPInd.y++)
	{
	    Pt2dr aPR0 = aPIG.P0() + Pt2dr(aPInd) * aResol;
	    Pt2dr aPR = aPR0;
	    for 
	    (
                std::list<cGridComp *>::iterator itG= aLG.begin();
		itG != aLG.end();
                itG++
	    )
	    {
	         aPR = (**itG)(aPR);
	    }
	    aImDx.data()[ aPInd.y][aPInd.x] = aPR.x-aPR0.x ;
	    aImDy.data()[ aPInd.y][aPInd.x] = aPR.y-aPR0.y;
	}
    }

    double aDyn = aECG.Dyn();
    if (aECG.NameNorm().IsInit())
    {
       Tiff_Im::Create8BFromFonc
       (
           aECG.Directory().Val() + aECG.NameNorm().Val(),
	   aImDx.sz(),
	   Max(0,Min(255,128+aDyn*Hypot(aImDx.in(),aImDy.in())))
       );
    }

    return 0;
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
