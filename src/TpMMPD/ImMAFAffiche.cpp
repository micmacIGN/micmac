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
#include <iostream>
#include <string>
#include "TiePByMesh/InitOutil.h"

int ImMAFAffiche_main(int argc,char ** argv)
{
    std::string aImg, aMAF, aDir, aNameImg, atoto;
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aImg, "Pattern of image", eSAM_IsExistFile)
                << EAMC(aMAF, "Image measurement file", eSAM_IsExistFile),
                //optional arguments
                LArgMain()
                << EAM(atoto, "toto", true, "toto")
             );
    SplitDirAndFile(aDir,aNameImg,aImg);


    //plot in a file
    Video_Win * VW = 0;
    Tiff_Im aTif = Tiff_Im::UnivConvStd(aDir+aNameImg);
    Pt2di aSzImg = aTif.sz();
    Im2D<U_INT1,INT4> I(aSzImg.x,aSzImg.y);

    Tiff_Im mPicTiff = Tiff_Im::StdConvGen(aDir + aNameImg,1,true);
    TIm2D<U_INT1,INT4> mPic_TIm2D = mPicTiff.sz();
    ELISE_COPY(mPic_TIm2D.all_pts(), mPicTiff.in(), mPic_TIm2D.out());
    I = mPic_TIm2D._the_im;
    VW = display_image(&I, aNameImg+" & MAF", VW, 0.2);


    //read all pts from MAF file
    cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aMAF,SetOfMesureAppuisFlottants);
    std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();

    for(std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end() ; iT1++)
    {

        std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();

        for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
        {
            VW = draw_pts_onVW(Pt2dr(iT2->PtIm()),VW,"red");

        }
    }
    VW->clik_in();

    return EXIT_SUCCESS;
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
