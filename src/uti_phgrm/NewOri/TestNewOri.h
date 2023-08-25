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

#ifndef _NEW_TESTORI_H
#define _NEW_TESTORI_H

#include<ctime>
#include "NewOri.h"
#include "../TiepTri/MultTieP.h"
#include "cNewO_SolGlobInit_PerfTri.h"

//class RandUnifQuick;
//class cAppliGenOptTriplets;
extern cSolBasculeRig  BascFromVRot
                (
                     const std::vector<ElRotation3D> & aVR1 ,
                     const std::vector<ElRotation3D> & aVR2,
                     std::vector<Pt3dr> &              aVP1,
                     std::vector<Pt3dr> &              aVP2
                );

static void SimilGlob2LocThreeV(const ElRotation3D Ri, const ElRotation3D Rj,  const ElRotation3D Rm,
                                const ElRotation3D rk_i, const ElRotation3D rk_j, const ElRotation3D rk_m,
                                ElMatrix<double>& Rk, Pt3dr& Ck, double& Lk)
{
    std::vector<ElRotation3D> aVR1;
    std::vector<ElRotation3D> aVR2;
    std::vector<Pt3dr> aVP1;
    std::vector<Pt3dr> aVP2;

    aVR1.push_back(rk_i);
    aVR1.push_back(rk_j);
    aVR1.push_back(rk_m);
    aVR2.push_back(Ri);
    aVR2.push_back(Rj);
    aVR2.push_back(Rm);

    cSolBasculeRig  aSol =  BascFromVRot(aVR1,aVR2,aVP1,aVP2);


    Rk = aSol.Rot();
    Ck = aSol.Tr();
    Lk = aSol.Lambda();

}

static void SimilGlob2LocTwoV(const ElRotation3D Ri, const ElRotation3D Rj, 
				       const ElRotation3D rk_i, const ElRotation3D rk_j,
					   ElMatrix<double>& Rk, Pt3dr& Ck, double& Lk)
{
    std::vector<ElRotation3D> aVR1;
    std::vector<ElRotation3D> aVR2;
    std::vector<Pt3dr> aVP1;
    std::vector<Pt3dr> aVP2;

    aVR1.push_back(rk_i);
    aVR1.push_back(rk_j);
    aVR2.push_back(Ri);
    aVR2.push_back(Rj);

    cSolBasculeRig  aSol =  BascFromVRot(aVR1,aVR2,aVP1,aVP2);

    Lk = aSol.Lambda();
    Rk = aSol.Rot();
    Ck = aSol.Tr();



}


class cAppliImportArtsQuad
{
	public:
		cAppliImportArtsQuad(int argc,char ** argv);

		bool ImportTracks();

		bool SaveCalib();

	private:
		cInterfChantierNameManipulateur * mICNM;
		std::string mTrackName;
		std::string mListName;
		std::string mFocalName;

		std::string mSH;
};

#endif

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

