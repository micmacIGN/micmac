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

#include "RTI.h"


void  cAppli_RTI::OneItereRecalRadiom
      (
           Im2D_U_INT2  aIRMaster,
           Im2D_Bits<1> aIRMasq,
           Im2D_U_INT2  aIRSec,
           int          aNbCase,
           int          aDeg
      )
{
     TIm2D<U_INT2,INT> aTRMaster(aIRMaster);
     TIm2D<U_INT2,INT> aTRSec(aIRSec);
     TIm2DBits<1> aTRMasq(aIRMasq);
        
     Pt2di aSz = aIRMaster.sz();
     for (int aCx=0 ; aCx<aNbCase ; aCx++)
     {
          int aX0 = (aSz.x * aCx) / aNbCase;
          int aX1 = (aSz.x * (aCx+1)) / aNbCase;
          for (int aCy=0 ; aCy<aNbCase ; aCy++)
          {
               int aY0 = (aSz.y * aCy) / aNbCase;
               int aY1 = (aSz.y * (aCy+1)) / aNbCase;

                Pt2dr aSomP(0,0);
                double aSomPds = 0;
                double aSomIm = 0;
                double aSomIs = 0;
                Pt2di aP;
                for (aP.x=aX0 ; aP.x<aX1; aP.x++)
                {
                     for (aP.y=aY0 ; aP.y<aY1; aP.y++)
                     {
                          if (aTRMasq.get(aP))
                          {
                              aSomP = aSomP + Pt2dr(aP);
                              aSomPds += 1;
                              aSomIm += aTRMaster.get(aP);
                              aSomIs += aTRSec.get(aP);
                          }
                     }
                }
                if (aSomPds!=0)
                {
                    aSomP = aSomP / aSomPds;
                    aSomIm /= aSomPds;
                    aSomIs /= aSomPds;
                }
          }
     }
}

void  cAppli_RTI::DoOneRecalRadiomBeton()
{
    Im2D_U_INT2 aIRMas =  Master()->ImRed();
    Im2D_Bits<1>  aIRMaq = Master()->MasqRed(aIRMas);
    Im2D_U_INT2 aIRSec =  UniqSlave()->ImRed();

    OneItereRecalRadiom
    (
         aIRMas,
         aIRMaq,
         aIRSec,
         10,
         1
    );

    std::cout << "RRRbb " << aIRMas.sz()  << " " << aIRSec.sz() << "\n";
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
