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
#include "Apero.h"




void cPoseCam::ShowRel(const cTraceCpleCam & aTCC,const cPoseCam & aCam2) const
{
     ElRotation3D aR1=mRF->CurRot();
     ElRotation3D aR2=aCam2.mRF->CurRot();

     ElRotation3D aR2inR1 = aR1.inv() * aR2;

     std::cout << aR2inR1.ImAff(Pt3dr(0,0,0)) << " " 
               << " D=" << euclid(aR2inR1.ImAff(Pt3dr(0,0,0))) << " "
               << "\n";
     std::cout << "Teta " << aR2inR1.teta01()
               << " "     << aR2inR1.teta02()
               << " "     << aR2inR1.teta12()
	       << "\n";

    const CamStenope *  aCS1 = mCF->CameraCourante();
    const CamStenope *  aCS2 = aCam2.mCF->CameraCourante();
    for 
    (
         std::list<cTraceCpleHom>::const_iterator itTCH=aTCC.TraceCpleHom().begin();
         itTCH!=aTCC.TraceCpleHom().end();
	 itTCH++
    )
    {
         ElPackHomologue aPack;
         mAppli.InitPack(itTCH->Id(),aPack,Name(),aCam2.Name());

	 std::cout <<  "ID BD " << itTCH->Id() << "\n";
	 double aSangle = 0;
	 double aSZ = 0;
	 double aS1=0;
	 for
	 (
             ElPackHomologue::tCstIter itL=aPack.begin();
             itL!=aPack.end();
             itL++
	 )
	 {
              ElSeg3D  aSeg1 = aCS1->F2toRayonR3( itL->P1());
              ElSeg3D  aSeg2 = aCS2->F2toRayonR3( itL->P2());


	      Pt3dr aT1 = aSeg1.TgNormee();
	      Pt3dr aT2 = aSeg2.TgNormee();
              Pt3dr aPTer = aSeg1.PseudoInter(aSeg2);

	      // std::cout << "angle = " << acos(scal(aT1,aT2)) << "\n";
	      aSangle += acos(scal(aT1,aT2));
	      aS1++;
	      aSZ += aPTer.z;

	      // std::cout << itL->P1() << itL->P2() << aPTer << "\n";
	      // std::cout << "D = " << aSeg1.DistDoite(aPTer) << " " << aSeg2.DistDoite(aPTer) << "\n";
	 }
	 std::cout << "ANGLE = " << aSangle / aS1 <<  " " <<  " Z=" << aSZ/aS1 << "\n";
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
