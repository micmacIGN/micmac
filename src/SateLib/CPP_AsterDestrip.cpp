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

MicMa cis an open source software specialized in image matching
for research in geographic information. MicMac is built on the
eLiSe image library. MicMac is governed by the  "Cecill-B licence".
See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"
#include <algorithm>
#include "hassan/reechantillonnage.h"


int AsterDestrip_main(int argc, char ** argv)
{
    std::string aFullNameIm, aNameIm;
    std::string aFileOut = "";
    //Reading the arguments
    ElInitArgMain
        (
        argc, argv,
        LArgMain()
        << EAMC(aFullNameIm, "Aster image file", eSAM_IsPatFile),
        LArgMain()
        << EAM(aFileOut, "Out", true, "Output xml file with RPC2D coordinates")
        );
    string aNameDir;
    SplitDirAndFile(aNameDir, aNameIm, aFullNameIm);

    if (aFileOut == "")
    {
        aFileOut = aNameDir + StdPrefix(aNameIm) + "_Destriped.tif";
    }

    //Reading the image and creating the objects to be manipulated
    Tiff_Im aTF = Tiff_Im::StdConvGen(aNameDir + aNameIm, 1, false);

    Pt2di aSz = aTF.sz();

    Im2D_REAL8  aIm(aSz.x, aSz.y);
    Im2D_U_INT1  aImOut(aSz.x, aSz.y);

    ELISE_COPY
        (
        aTF.all_pts(),
        aTF.in(),
        aIm.out()
        );

    cout << aNameIm << " loaded" << endl;
    REAL8 ** aData = aIm.data();
    //U_INT1 ** aDataOut = aImOut.data();

    //Making regular image at F=2pix
    Im2D_REAL8  aImRegul(aSz.x, aSz.y);
    REAL8 ** aDataRegul = aImRegul.data();
    {
        Im2D_REAL8  aImHalf(aSz.x / 2, aSz.y / 2);
        REAL8 ** aDataHalf = aImHalf.data();
        for (int aY = 0; aY<aSz.y/2; aY++)
        {
            for (int aX = 0; aX<aSz.x/2; aX++)
            {
                Pt2dr ptOut(aX * 2, aY * 2);
                aDataHalf[aY][aX] = Reechantillonnage::biline(aData, aSz.x, aSz.y, ptOut);
            }
        }
        cout << "Small image generated (of size : " << aSz.x / 2 << " " << aSz.y / 2 << ")" << endl;

        double min = 0, max = 0;
        for (int aY = 0; aY<aSz.y ; aY++)
        {
            for (int aX = 0; aX<aSz.x ; aX++)
            {
                Pt2dr ptOut(aX / 2, aY / 2);
                aDataRegul[aY][aX] = aData[aY][aX]-Reechantillonnage::biline(aDataHalf, aSz.x / 2, aSz.y / 2, ptOut);
                if (aDataRegul[aY][aX] < min)
                    min = aDataRegul[aY][aX];
                if (aDataRegul[aY][aX] > max)
                    max = aDataRegul[aY][aX];
            }
        }

        //Normalization
        for (int aY = 0; aY<aSz.y; aY++)
        {
            for (int aX = 0; aX<aSz.x; aX++)
            {
                aDataRegul[aY][aX] = (aDataRegul[aY][aX] - min) * 255 / max;
            }
        }

        cout << "Regul image generated" << endl;

    }

	
    Tiff_Im  aTOut
        (
        aFileOut.c_str(),
        aSz,
        GenIm::u_int1,
        Tiff_Im::No_Compr,
        Tiff_Im::BlackIsZero
        );


    ELISE_COPY
        (
        aTOut.all_pts(),
        aImOut.in(),
        aTOut.out()
        );

    return 0;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
