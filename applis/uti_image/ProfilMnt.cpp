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



int main(int argc,char ** argv)
{
    Pt2di aP0;
    Pt2di aSz(500,500);
    Pt2di aSzP(700,300);
    REAL aStepXY = 1;
    REAL aStepZ = 1.0;
    std::string aNameMnt;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aNameMnt) 
                    << EAM(aP0),
	LArgMain()  << EAM(aSz,"Sz",true)
                    <<  EAM(aStepZ,"StepZ",true)
    );	

    Tiff_Im aFileMnt = Tiff_Im::StdConv(aNameMnt.c_str());
    std::string aNameIm = StdPrefix(aNameMnt)+std::string("Shade.tif");
    Tiff_Im aFileIm =  Tiff_Im::StdConv(aNameIm.c_str());

    Video_Win aW = Video_Win::WStd(aSz,1.0);

    Video_Win aWP = Video_Win::WStd(aSzP,1.0);

    Im2D_REAL4 aMnt(aSz.x,aSz.y);

     ELISE_COPY(aMnt.all_pts(),trans(aFileMnt.in(),aP0),aMnt.out());
     ELISE_COPY(aMnt.all_pts(),trans(aFileIm.in(),aP0),aW.ogray());

     TIm2D<REAL4,REAL8> aTMnt(aMnt);


     while (1)
     {
          Pt2di aQ0 = aW.clik_in()._pt;
          Pt2di aQ1 = aW.clik_in()._pt;
          aW.draw_seg(aQ0,aQ1,aW.pdisc()(P8COL::red));

          std::vector<REAL> Vals;
          INT  aNb = round_ni(euclid(aQ0-aQ1)/aStepXY);
          REAL aVMin =  1e10;
          for (INT aK=0 ; aK<= aNb ; aK++)
          {
              Pt2dr aP = barry(aK/REAL(aNb),aQ1,aQ0);
              REAL aV = aTMnt.getr(aP) / aStepZ;
              Vals.push_back(aV);
              ElSetMin(aVMin,aV);
          }
          aWP.clear();


          aVMin -=2;
          for (INT aK=0 ; aK< aNb ; aK++)
          {
              Pt2dr aU1(aK,Vals[aK]-aVMin);
              Pt2dr aU2(aK+1,Vals[aK+1]-aVMin);
              aWP.draw_seg(aU1,aU2,aWP.pdisc()(P8COL::green));
          }
     }
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
