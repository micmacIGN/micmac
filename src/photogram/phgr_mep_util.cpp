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




/************************************************************/
/*                                                          */
/*                  Utilitaires                             */
/*                                                          */
/************************************************************/

void InitPackME
     (  
          std::vector<Pt2dr> & aVP1,
          std::vector<Pt2dr>  &aVP2,
          std::vector<double>  &aVPds,
          const  ElPackHomologue & aPack
     )
{
   for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
   {
      aVP1.push_back(itP->P1());
      aVP2.push_back(itP->P2());
      aVPds.push_back(itP->Pds());
   }
}

//  Formule exacte et programmation simple et claire pour bench, c'est l'angle

double ExactCostMEP(Pt3dr &  anI,const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) 
{
   Pt3dr aQ1 = Pt3dr(aP1.x,aP1.y,1.0);
   Pt3dr aQ2 = aRot.Mat() * Pt3dr(aP2.x,aP2.y,1.0);
   Pt3dr aBase  = aRot.tr();

   ElSeg3D aS1(Pt3dr(0,0,0),aQ1);
   ElSeg3D aS2(aBase,aBase+aQ2);

   anI = aS1.PseudoInter(aS2);
    
   double d1 = aS1.DistDoite(anI);
   double d2 = aS2.DistDoite(anI);
   double D1 = euclid(anI);
   double D2 = euclid(aBase-anI);


   double aTeta =  d1/D1 + d2/D2;
   return GenCoutAttenueTetaMax(aTeta,aTetaMax);
}

double PVExactCostMEP(const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) 
{
   Pt3dr aQ1 = vunit(PZ1(aP1));
   Pt3dr aQ2 = aRot.Mat() *  vunit(PZ1(aP2));
   Pt3dr aBase  = aRot.tr();

   Pt3dr aQ1vQ2vB = vunit(aQ1 ^ aQ2) ^ aBase;

   double aDet = Det(aQ1,aQ2,aBase);

   //   /2 pour etre coherent avec ExactCostMEP
   double aTeta = (ElAbs(aDet/scal(aQ1vQ2vB,aQ1)) +  ElAbs(aDet/scal(aQ1vQ2vB,aQ2))) / 2.0;
   
   return GenCoutAttenueTetaMax(aTeta,aTetaMax);
}


double ExactCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax) 
{
    double aSomPCost = 0;
    double aSomPds = 0;
    Pt3dr anI;

    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = ExactCostMEP(anI,aRot,itP->P1(),itP->P2(),aTetaMax);

 std::cout << "RRRrr " << aCost / PVExactCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax) << "\n";

         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return aSomPCost / aSomPds;
}


Pt3dr MedianNuage(const ElPackHomologue & aPack,const ElRotation3D & aRot)
{
    std::vector<double>  aVX;
    std::vector<double>  aVY;
    std::vector<double>  aVZ;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
        Pt3dr                anI;
        ExactCostMEP(anI,aRot,itP->P1(),itP->P2(),0.1);
        aVX.push_back(anI.x);
        aVY.push_back(anI.y);
        aVZ.push_back(anI.z);

// std::cout << "iiiiI " << anI << "\n";
    }
    return Pt3dr
           (
                 MedianeSup(aVX),
                 MedianeSup(aVY),
                 MedianeSup(aVZ)
           );
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
