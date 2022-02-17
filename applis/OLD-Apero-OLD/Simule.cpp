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

#include "Apero.h"

namespace NS_ParamApero
{

void cAppliApero::InitPointsTerrain
     (
         std::list<Pt3dr>  & aL, 
         const std::string &
     )
{
     ELISE_ASSERT(false,"InitPointsTerrain(cGPtsTer_By_File &)");
}


void cAppliApero::InitPointsTerrain
     (
          std::list<Pt3dr>  & aL, 
          const cGPtsTer_By_ImProf & aGBI
     )
{
   const CamStenope * aMasterCam = 0;
   if (aGBI.ImMaitresse().IsInit())
      aMasterCam = PoseFromName(aGBI.ImMaitresse().Val())->CF()->CameraCourante();

   std::string aNF =  DC() + aGBI.File();
   Im2D_REAL4 aMnt = Im2D_REAL4::FromFileStd(aNF);
   TIm2D<REAL4,REAL8> aTMnt(aMnt);
   Pt2di aSz = aMnt.sz();

   int aNb = aGBI.NbPts();
   Pt2di aPtNbXY(0,0);
   Pt2dr aPtPas(0,0);
   if (aGBI.OnGrid())
   {
      double aPas = sqrt(aSz.x*aSz.y/double(aNb));
      aPtNbXY = Pt2di
              (
	         round_up(aSz.x/aPas),
	         round_up(aSz.y/aPas)
              );

       aNb = aPtNbXY.x * aPtNbXY.y;
       aPtPas = Pt2dr(double(aSz.x)/aPtNbXY.x,double(aSz.y)/aPtNbXY.y);
   }

   for (int aKP=0 ; aKP<aNb ; aKP++)
   {
       Pt2dr aPPl(0,0);

       if (aGBI.OnGrid())
       {
           int aKX = aKP %  aPtNbXY.x;
           int aKY = aKP /  aPtNbXY.x;
	   double aMul = 0.5 * aGBI.RandomizeInGrid().Val();
           aPPl = Pt2dr
	          (
		      (aKX+0.5+aMul*NRrandC())*aPtPas.x,
		      (aKY+0.5+aMul*NRrandC())*aPtPas.y
                  );
           // std::cout  << aKX << " " << aKY << " " << aNb << " "<< aPtNbXY << "\n";
	   // std::cout << aPtPas << "\n";
       }
       else
       {
          aPPl = Pt2dr(aSz.x*NRrandom3(),aSz.y*NRrandom3());
       }

       
       // std::cout << aPPl << "\n";
       double aZ = aTMnt.getr(aPPl);

       aZ = aGBI.Origine().z + aZ *  aGBI.Step().z;
       aPPl = Pt2dr
              (
	           aGBI.Origine().x+aPPl.x*aGBI.Step().x,
	           aGBI.Origine().y+aPPl.y*aGBI.Step().y
	      );
       Pt3dr aPTer(aPPl.x,aPPl.y,aZ);


       if (aMasterCam)
       {
           if (aGBI.DTMIsZ().Val())
              aPTer = aMasterCam->ImEtZ2Terrain(aPPl,aZ);
           else
              aPTer = aMasterCam->ImEtProf2Terrain(aPPl,aZ);
       }
       aL.push_back(aPTer);
   }
}

void cAppliApero::SimuleOneLiaison
     (
         const cGenerateLiaisons & aGL,
	 const std::list<Pt3dr>  & aLP3,
         cPoseCam & aCam1,
         cPoseCam & aCam2
     )
{
   const CamStenope * aCS1 = aCam1.CF()->CameraCourante();
   const CamStenope * aCS2 = aCam2.CF()->CameraCourante();

   ElPackHomologue aPack;

   for(std::list<Pt3dr>::const_iterator itP3=aLP3.begin();itP3!=aLP3.end();itP3++)
   {
      Pt2dr aPIm1 = aCS1->R3toF2(*itP3)+Pt2dr(NRrandC(),NRrandC())*aGL.BruitIm1();
      Pt2dr aPIm2 = aCS2->R3toF2(*itP3)+Pt2dr(NRrandC(),NRrandC())*aGL.BruitIm2();

      aPack.Cple_Add(ElCplePtsHomologues(aPIm1,aPIm2));
   }

   std::string aName = mDC+mICNM->Assoc1To2(aGL.KeyAssoc(),aCam1.Name(),aCam2.Name(),true);
   cElXMLFileIn aFileXML(aName);
   aFileXML.PutPackHom(aPack);
}


void cAppliApero::ExportOneSimule(const cExportSimulation & anES)
{
    // Generation des points terrain
    std::list<Pt3dr> aLP3;
    if (anES.GPtsTer_By_File().IsInit())
    {
         InitPointsTerrain(aLP3,anES.GPtsTer_By_File().Val());
    }
    else if (anES.GPtsTer_By_ImProf().IsInit())
    {
         InitPointsTerrain(aLP3,anES.GPtsTer_By_ImProf().Val());
    }
    else
    {
         ELISE_ASSERT(false,"ExportOneSimule  : GeneratePointsTerrains ??");
    }


    // Generation des points de liaison
    for
    (
       std::list<cGenerateLiaisons>::const_iterator itGL=anES.GenerateLiaisons().begin();
       itGL!=anES.GenerateLiaisons().end();
       itGL++
    )
    {
        cElRegex aSel1(itGL->FilterIm1().Val(),10);
        cElRegex aSel2(itGL->FilterIm2().Val(),10);

	for(tDiPo::const_iterator it1=mDicoPose.begin();it1!=mDicoPose.end();it1++)
	{
	    for(tDiPo::const_iterator it2=mDicoPose.begin();it2!=mDicoPose.end();it2++)
	    {
	        if ((it1!=it2)&&(aSel1.Match(it1->first))&&(aSel2.Match(it2->first)))
                {
		   SimuleOneLiaison(*itGL,aLP3,*(it1->second),*(it2->second));
		}
	    }
	}
    }
}
 



};

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
