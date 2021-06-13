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

#include "all_etal.h"
#include "XML_GEN/all.h"

using namespace NS_ParamChantierPhotogram;

typedef int (* tFINT)(int);

int Cdd_NumCol(int aK)
{
    if (aK < 161 )
       return (aK-1) /9;

    return aK/9;
    
}

int Clous_NumCol(int aK)
{
   return (aK%20);
}

void TraitementSpec_CdD (std::vector<cCibleCalib> &  aVCC,tFINT  aFCol)
{
   std::vector<int>    aVS1;
   std::vector<Pt2dr>  aVSP;
   std::vector<Pt2dr>  aVTgt;

   for (int aK=0 ; aK<int(aVCC.size()) ; aK++)
   {
       int aCol = aFCol(aK);
       ELISE_ASSERT(aCol>=0 || (aCol==0),"TraitementSpec_CdD");
       while ( int(aVS1.size()) <= aCol) 
       {
	   aVS1.push_back(0);
	   aVSP.push_back(Pt2dr(0,0));
       }
        aVS1[aCol]++;
	Pt3dr aP3 =aVCC[aK].Position();
        aVSP[aCol] =  aVSP[aCol]+Pt2dr(aP3.x,aP3.y);
   }

   for (int aK=0 ;aK<int(aVSP.size()) ; aK++)
   {
      aVSP[aK] = aVSP[aK] / aVS1[aK];
      aVTgt.push_back(Pt2dr(0,0));
   }

   for (int aK=0 ; aK<int(aVCC.size()) ; aK++)
   {
       int aC0  = aFCol(aK);
       Pt2dr aTgt(0,0);
       if (aC0==0)
       {
           Pt2dr aD01 = vunit(aVSP[1]-aVSP[0]);
           Pt2dr aD12 = vunit(aVSP[2]-aVSP[1]);
	   aTgt = vunit(aD01 + (aD01-aD12)/2.0);
       }
       else if (aC0 == int(aVSP.size())-1)
       {
           Pt2dr aDP1P0 = vunit(aVSP[aC0]-aVSP[aC0-1]);
           Pt2dr aDP2P1 = vunit(aVSP[aC0-1]-aVSP[aC0-2]);
	   aTgt = vunit(aDP1P0 + (aDP1P0-aDP2P1)/2.0);
       }
       else
       {
           Pt2dr aDP0 = vunit(aVSP[aC0]-aVSP[aC0-1]);
           Pt2dr aD0N = vunit(aVSP[aC0+1]-aVSP[aC0]);
	   aTgt = vunit((aDP0+aD0N)/2.0);
       }
       aVTgt[aC0] = aTgt;

       
       Pt2dr aN = aTgt * Pt2dr(0,1);

       aVCC[aK].Normale() = Pt3dr(aN.x,aN.y,0);
   }

   for (int aK=0 ;aK<int(aVSP.size()) ; aK++)
       std::cout << aK << " :: " << aVTgt[aK] << "\n";
}



void TraitementSpecHall(std::vector<cCibleCalib> &  aVCC)
{
   for (int aK=0 ; aK<int(aVCC.size()) ; aK++)
   {
       cCibleCalib & aC = aVCC[aK];
       int aEtage = round_ni(aC.Position().z/3.15);  // 1 ,2 ou 3=plafond
       ELISE_ASSERT
       (
           (aEtage>=1)  && (aEtage<=3),
	   "Etage ?? dans TraitementSpecHall"
       );

       double  FactCorrel[4] = {-1,1.1,1.2,1.4};
       double  FactRaff[4] = {-1,1.05,1.15,1.35};

       aC.FacteurElargRechCorrel().SetVal(FactCorrel[aEtage]);
       aC.FacteurElargRechRaffine().SetVal(FactRaff[aEtage]);
   }
}

void ConvertOnePolygone
     (
          Pt3dr aNorm,
          std::string aNameIn,
          std::string aNameOut,
          std::string aNamePoly,
          std::string aNameMesure,
	  bool        isPonct,
	  int         aNbRelief,
	  double *    aProfRelief,
	  double *    aRayRelief,
	  bool        isSortant,
          bool        Permut
     )
{

    cPolygoneEtal * aPE = cPolygoneEtal::FromName(aNameIn,0);
    cPolygoneCalib  aPC;

    aPC.Name() = aNameMesure;




    for 
    (
         cPolygoneEtal::tContCible::const_iterator itC=aPE->ListeCible().begin();
	 itC != aPE->ListeCible().end();
	 itC++
    )
    {

       cCibleCalib aCC;
       aCC.Id() =  (*itC)->Ind() ;
       aCC.Position() = (*itC)->Pos();
       if (Permut)
       {
            Pt3dr aP = aCC.Position();
            aCC.Position() = Pt3dr(aP.z,aP.x,aP.y);
       }
       // Defaut , normale sortante IGN 
       aCC.Normale() = aNorm;
       aCC.Ponctuel() = isPonct;
       const cMirePolygonEtal & aMire  = (*itC)->Mire();

       for (int aK=0 ; aK<aMire.NbDiam() ; aK++)
           aCC.Rayons().push_back(aMire.KthDiam(aK));

       for (int aK=0 ; aK<aNbRelief; aK++)
       {
           cCercleRelief aCR;
	   aCR.Rayon() = aRayRelief[aK];
	   aCR.Profondeur() = aProfRelief[aK];
	   aCC.CercleRelief().push_back(aCR);
       }

       aCC.Negatif().SetVal(aMire.IsNegatif());
       aCC.ReliefIsSortant() = isSortant;
       aCC.NomType() = aMire.Name();
       aCC.Qualite() = (*itC)->Qual();
       aPC.Cibles().push_back(aCC);

    }

    if (aNamePoly=="CapriceDesDieux")
    {
        TraitementSpec_CdD(aPC.Cibles(),Cdd_NumCol);
    }   
    if (aNamePoly=="Clous_CDD")
    {
        TraitementSpec_CdD(aPC.Cibles(),Clous_NumCol);
    }   
    
    if (aNamePoly== "Mtd-Hall")
    {
        TraitementSpecHall(aPC.Cibles());
    }

    MakeFileXML(aPC,aNameOut);
}

void GenerePolyTxtRect
     (
         Pt2di aC0,  // 0 ,0
         Pt2di aC1,          // 25,16
         Pt2di aCoord1,    // 47 97
         Pt2di aCoord2,   // 5436  3543
         std::string aFile3D,
         std::string aFile2D,
         Pt2di  aRabDeb,
         Pt2di  aRabFin
     )
{
   FILE * aF3 = ElFopen(aFile3D.c_str(),"w");
   FILE * aF2 = ElFopen(aFile2D.c_str(),"w");

    Pt2dr aPDelta (
                     (aCoord2.x-aCoord1.x) / (aC1.x-aC0.x),
                     (aCoord2.y-aCoord1.y) / (aC1.y-aC0.y)
                  );
    std::cout << "DELTA " << aPDelta << "\n";

    Pt2di aP;
    for  (aP.x=(aRabDeb.x+aC0.x) ;  aP.x<=(aRabFin.x+aC1.x) ; aP.x++)
    {
       for  (aP.y=(aRabDeb.y+aC0.y) ;  aP.y<=(aRabFin.y+aC1.y) ; aP.y++)
       {
            int aNum = 10000 + aP.y*100 + aP.x;

            Pt2dr aPIm (
                             aCoord1.x + aPDelta.x*(aP.x-aC0.x),
                             aCoord1.y + aPDelta.y*(aP.y-aC0.y)
                       );

           fprintf(aF3,"%d %d %d 0.0  N6 0\n",aNum,aP.x,aP.y);
           fprintf(aF2,"%d %d %d\n",aNum,round_ni(aPIm.x),round_ni(aPIm.y));

       }
    }
    ElFclose(aF3);
    ElFclose(aF2);
}

int main(int argc,char ** argv)
{
/*
   {
      double aProfClou[2] = {4.5,9.0};
      double aRayClou[2] = {20.0,20.0};
      ConvertOnePolygone
      (
          Pt3dr(0,0,0),
         "/mnt/data/Calib/References/PolygCapDieux/PolygClousCDD.txt",
         "/mnt/data/Calib/References/PolygCapDieux/PolygClousCDD.xml",
         "Clous_CDD",
         true,
         2,aProfClou,aRayClou,true
      );
   }

   {
       double aProfCdd[1] = {19.0};
       double aRayCdd[1] = {50.0};
 
       ConvertOnePolygone
       (
          Pt3dr(0,0,0),
          "/mnt/data/Calib/References/PolygCapDieux/PolyCapriceDesDieux.txt",
          "/mnt/data/Calib/References/PolygCapDieux/PolyCapriceDesDieux.xml",
          "CapriceDesDieux",
          false,
	  1,aProfCdd,aRayCdd,false
       );
   }

   ConvertOnePolygone
   (
      Pt3dr(0,-1,0),
      "/mnt/data/Calib/References/PolygIGN/IGNPoly2003",
      "/mnt/data/Calib/References/PolygIGN/IGNPoly2003.xml",
      "IGN-2003",
      false,
      0,0,0,false
   );

   ConvertOnePolygone
   (
      Pt3dr(0,-1,0),
      "/mnt/data/Calib/References/PolygIGN/IGNPoly2006",
      "/mnt/data/Calib/References/PolygIGN/IGNPoly2006.xml",
      "IGN-2006",
      false,
      0,0,0,false
   );
   */
/*

   ConvertOnePolygone
   (
      Pt3dr(0,0,-1.0),
      "/mnt/data/Calib/References/PolygHallMtd/HallMtd.txt",
      "/mnt/data/Calib/References/PolygHallMtd/HallMtd.xml",
      "Mtd-Hall",
      "Mtd-Mai-2008",
      false,
      0,0,0,false
   );

   ConvertOnePolygone
   (
      Pt3dr(0,0,-1.0),
      "/mnt/data/Calib/References/PolygHallMtd/HallMtd2.txt",
      "/mnt/data/Calib/References/PolygHallMtd/HallMtd2.xml",
      "Mtd-Hall",
      "Mtd-Recalc-PhgrStd",
      false,
      0,0,0,false
   );

   ConvertOnePolygone
   (
      Pt3dr(0,0,-1.0),
      "/mnt/data/Calib/References/PolygHallMtd/HallMtd3.txt",
      "/mnt/data/Calib/References/PolygHallMtd/HallMtd3.xml",
      "Mtd-Hall",
      "Mtd-Recalc-Brown",
      false,
      0,0,0,false
   );

   ConvertOnePolygone
   (
      Pt3dr(0,-1.0,0),
      "/media/SQP/Calib/References/IGN-Polygone-nouveau/IGN-2009.txt",
      "/media/SQP/Calib/References/IGN-Polygone-nouveau/IGN-2009.xml",
      "IGN-Polygone-2009",
      "IGN-Polygone-2009-Mesure-03-09",
      false,
      0,0,0,false
   );

   ConvertOnePolygone
   (
      Pt3dr(0,-1.0,0),
      "/media/SQP/Calib/References/PolygIgnAncien/IGNPoly2009",
      "/media/SQP/Calib/References/PolygIgnAncien/IGNPoly2009.xml",
      "IGN-Ancien-Polygone",
      "IGN-Ancien-Polygone-Mesure-Avril-2009",
      false,
      0,0,0,false
   );
   */

/*


    GenerePolyTxtRect
     (
         Pt2di(0 ,0),
         Pt2di (25,16),
         Pt2di(47,97),
         Pt2di ( 5436 ,3543),
         "/data/Calib-Nikon/Ref/Poly3D.txt",
         "/data/Calib-Nikon/Ref/PointeIm.txt",
         Pt2di(0,0),
         Pt2di(0,4)
     );

   ConvertOnePolygone
   (
      Pt3dr(0,0.0,-1.0),
      "/data/Calib-Nikon/Ref/Poly3D.txt",
      "/data/Calib-Nikon/Ref/Poly3D.xml",
      "SALLE MATIS",
      "SALLE MATIS",
      false,
      0,0,0,false
   );
*/
/*
*/


/*

   ConvertOnePolygone
   (
      Pt3dr(0,0,-1.0),
      "/media/Iomega3/Calibration/References/IGN-Polygone-nouveau/IGN-2009.txt",
      "/media/Iomega3/Calibration/References/IGN-Polygone-nouveau/MurXZ-IGN-2010.xml",
      "IGN-Polygone-2009",
      "IGN-Polygone-2010-Mesure-02-10",
      false,
      0,0,0,false,
      true
   );
*/

   ConvertOnePolygone
   (
      Pt3dr(0,0,1),
      "/media/HD-PVU2/CNR/Polygone/CNR-Polyg3D.txt",
      "/media/HD-PVU2/CNR/Polygone/CNR-Polyg3D.xml",
      "CNR-2010",
      "CNR-2010",
      false,
      0,0,0,false,
      false
   );
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
