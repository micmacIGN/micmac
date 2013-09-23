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



// Example of using solvers defined in   include/general/optim.h

void TestSys2(L2SysSurResol & aSys)
{
    bool Ok;
    Im1D_REAL8 aSol = aSys.GSSR_Solve(&Ok); 
    std::cout << "=== Ok is " << Ok << "\n";
    if (Ok)
    {
        double * aData = aSol.data();
        std::cout << "    Sol " << aData[0] << " " << aData[1] << "\n";
    }
}

int  Sample_LSQ0_main(int argc,char ** argv)
{
   std::cout << "Basic solver test \n";
   // Create L2SysSurResol to solve least square equation with 2 unknown

    bool ProvoqErr = false;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  ,
        LArgMain()  << EAM(ProvoqErr,"GenErr",true,"Generate error")
    );


   L2SysSurResol aSys(2);

   {
         double aPds[2] = {1,1};
         aSys.AddEquation(0.5,aPds,1);  // Add obs X + Y =1,  with pds 0.5
   }

   // System is not solvable now ....
   TestSys2(aSys);

   {
         double aPds[2] = {1,-1};
         aSys.AddEquation(1,aPds,1);  // Add obs X - Y =1,  with pds 1
   }

   // 2 Equation, 2 unknonw System shoud be OK now 
   TestSys2(aSys);

   {
         double aPds[2] = {1,1};
         aSys.AddEquation(1,aPds,2);  // Add obs X + Y =2,  with pds 1
   }

   {
         double aPds[2] = {-1,1};
         aSys.AddEquation(1,aPds,1);  // Add obs X - Y =1,  with pds 1
   }

   TestSys2(aSys);

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
