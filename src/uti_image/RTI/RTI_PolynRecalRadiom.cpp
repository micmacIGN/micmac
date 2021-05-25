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

void MakeCoeffPolynome(double & aSomCoeff,std::vector<double> & aVCoef,const Pt2dr & aP,Pt2di aDeg,double aMulI2,double * aSol)
{
     aVCoef.clear();
     std::vector<double> aMonX(1,1.0);
     std::vector<double> aMonY(1,1.0);
     int aDegTot = dist8(aDeg);
     for (int aD=1 ; aD<= aDegTot ; aD++)
     {
         aMonX.push_back(aMonX.back()*aP.x);
         aMonY.push_back(aMonY.back()*aP.y);
     }
     aSomCoeff = 0;
     // Im = S ( aij X^i Y ^j Is);
     for (int aDy=0 ; aDy<=aDeg.y ; aDy++)
     {
          for (int aDx=0 ; aDx<=aDeg.x ; aDx++)
          {
              double aMon = aMonX[aDx] * aMonY[aDy];
              if (aSol)
                 aSomCoeff += aMon * aSol[aVCoef.size()];
              aVCoef.push_back(aMulI2 * aMon);
          }
     }
}

Im2D_REAL8  cAppli_RTI::OneItereRecalRadiom
            (
               double &     aScaleRes,
               bool         aL1,
               Im2D_U_INT2  aIRMaster,
               Im2D_Bits<1> aIRMasq,
               Im2D_U_INT2  aIRSec,
               int          aNbCase,
               int          aDeg
            )
{
     cGenSysSurResol * aSys =0;
     int aNbObs = aNbCase * aNbCase;
     // int aNbEq = (aDeg * (aDeg+1) ) /2;
     int aNbEq = (aDeg+1) * (aDeg+1);
     if (aL1)
        aSys = new SystLinSurResolu(aNbEq,aNbObs);
     else
        aSys = new L2SysSurResol(aNbEq);

     TIm2D<U_INT2,INT> aTRMaster(aIRMaster);
     TIm2D<U_INT2,INT> aTRSec(aIRSec);
     TIm2DBits<1> aTRMasq(aIRMasq);
        

     aSys->SetPhaseEquation(0);
     Pt2di aSz = aIRMaster.sz();
     for (int aCx=0 ; aCx<aNbCase ; aCx++)
     {
          int aX0 = (aSz.x * aCx) / aNbCase;
          int aX1 = (aSz.x * (aCx+1)) / aNbCase;
          for (int aCy=0 ; aCy<aNbCase ; aCy++)
          {
               int aY0 = (aSz.y * aCy) / aNbCase;
               int aY1 = (aSz.y * (aCy+1)) / aNbCase;

                // Calcul des moyennes sur le rectangle
                Pt2dr aSomPts(0,0);
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
                              aSomPts = aSomPts + Pt2dr(aP);
                              aSomPds += 1;
                              aSomIm += aTRMaster.get(aP);
                              aSomIs += aTRSec.get(aP);
                          }
                     }
                }
                if (aSomPds!=0)
                {
                    aSomPts = aSomPts / aSomPds;
                    aSomIm /= aSomPds;
                    aSomIs /= aSomPds;
                    // Ajout au poly
                    std::vector<double> aVCoeff;
                    double aSomCoeff;
                    MakeCoeffPolynome(aSomCoeff,aVCoeff,aSomPts,Pt2di(aDeg,aDeg),aSomIs,(double *)0);
                    aSys->GSSR_AddNewEquation(aSomPds,VData(aVCoeff),aSomIm,(double *)0);

/*
                    std::vector<double> aMonX(1,1.0);
                    std::vector<double> aMonY(1,1.0);
                    for (int aD=1 ; aD<= aDeg ; aD++)
                    {
                       aMonX.push_back(aMonX.back()*aSomP.x);
                       aMonY.push_back(aMonY.back()*aSomP.y);
                    }
                    // Im = S ( aij X^i Y ^j Is);
                    for (int aDx=0 ; aDx<=aDeg ; aDx++)
                    {
                        for (int aDy=0 ; aDy<=aDeg ; aDy++)
                        {
                             aVCoef.push_back(aSomIs * aMonX[aDx] * aMonY[aDy]);
                        }
                    }
*/
                    
                }

                
          }
     }


     Im1D_REAL8 aSol = aSys->GSSR_Solve(0);
     REAL8 * aDS = aSol.data();

     double  aNbPol = 50;
     aScaleRes = ElMin(aSz.x/aNbPol,aSz.y/aNbPol);
     Pt2di aSzR = round_up(Pt2dr(aSz) /aScaleRes);
     std::cout << "SZR= " << aSzR << "\n";
     
     // Pt2di aSz = aIRMaster.sz();
     Im2D_REAL8 aImMul(aSzR.x,aSzR.y);
     TIm2D<REAL8,REAL8> aTMul(aImMul);
     Pt2di aP;
     for (aP.x=0 ; aP.x<aSzR.x ; aP.x++)
     {
         for (aP.y=0 ; aP.y<aSzR.y ; aP.y++)
         {
             Pt2dr aPR = Pt2dr(aP) * aScaleRes;
             double aSomCoeff;
             std::vector<double> aVCoeff;
             MakeCoeffPolynome(aSomCoeff,aVCoeff,aPR,Pt2di(aDeg,aDeg),1.0,aDS);
             aTMul.oset(aP,aSomCoeff);
         }
     }

     Tiff_Im::CreateFromIm(aImMul,"Mul.tif");

     delete aSys;
     return  aImMul;
}

void  cAppli_RTI::DoOneRecalRadiomBeton()
{
    Im2D_U_INT2 aIRMas =  Master()->MemImRed();
    Im2D_Bits<1>  aIRMaq = Master()->MasqRed(aIRMas);
    Im2D_U_INT2 aIRSec =  UniqSlave()->MemImRed();

    double aScalePol;
    Im2D_REAL8 aRes = OneItereRecalRadiom
                      (
                           aScalePol,
                           true,
                           aIRMas,
                           aIRMaq,
                           aIRSec,
                           20,
                           4
                      );

    double aScGlob = aScalePol * mParam.ScaleSSRes();
    Im2D_U_INT2 aIMas = Master()->MemImFull();
    Im2D_U_INT2 aISec = UniqSlave()->MemImFull();


    Tiff_Im::CreateFromFonc
    (
          UniqSlave()->NameDif(),
          aIMas.sz(),
          aIMas.in() - aISec.in_proj() * aRes.in(0)[Virgule(FX,FY)/aScGlob],
          GenIm::real4
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
