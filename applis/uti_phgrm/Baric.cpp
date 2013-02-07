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
#include "private/all.h"
#include "XML_GEN/all.h"

//  Par exemple :
//
//   bin/Baric /media/SQP_/Calib/Canon-E0S-5D-Mark2/F100/Coul/ img_0075.cr2 Canon-100-Macro Exe=1
//
//   Pour l'instant, c'est assez lourd car pour une raison qui m'echappe, c'est batchable
//
//   Appelle MICMAC pour chaque couple, qui appelle bin/ModeleRadial
//   qui genere un modele radiale qui est converit en grille par
//   
//
//
//
//

using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;

/*********************************************/
/*                                           */
/*                ::                         */
/*                                           */
/*********************************************/



void Banniere_Baric()
{
   std::cout << "\n";
   std::cout <<  " **********************************\n";
   std::cout <<  " *     B-ayer                     *\n";
   std::cout <<  " *     A-utomatic                 *\n";
   std::cout <<  " *     R-egistration by           *\n";
   std::cout <<  " *     I-mage                     *\n";
   std::cout <<  " *     C-orrelation               *\n";
   std::cout <<  " **********************************\n";

}

/*********************************************/
/*                                           */
/*            cAppliPorto                    */
/*                                           */
/*********************************************/

class cOneCplAB
{
     public:
        cOneCplAB
	(
	    const std::string & aC1,
	    const std::string & aC2
	)  :
	   mC1   (aC1),
	   mC2   (aC2)
	{
	}

	std::string mC1;
	std::string mC2;
};



class cAppliBaric : public cAppliBatch
{
    public :
       cAppliBaric(int argc,char ** argv);

       std::string NameDCR(const std::string & aChan);

    private :
       void MECOneCpl(const std::string & aCh1,const std::string & aCh2);
       void FusioneRes();
       void Exec();


       std::string mNameCam;
       std::vector<cOneCplAB>  mCpls;
       // void Sys(const std::string & aStr,bool Svp=false);
       //  std::string NameHomologues(int aK);
};







cAppliBaric::cAppliBaric(int argc,char ** argv) :
   cAppliBatch
   (
       argc, argv,
       2, 1,
       "Micmac-LIAISON"
   )
{
    std::string aToto;
    ElInitArgMain
    (
           ARGC(),ARGV(),
           LArgMain() << EAM(mNameCam),
           LArgMain() << EAM(aToto,"Toto",true)
    );
}


void cAppliBaric::MECOneCpl
     (
         const std::string & aCh1,
         const std::string & aCh2
     )
{
    std::string aCom =
          MMDir() +  std::string("bin/MICMAC ")
       + MMDir() + std::string("applis/XML-Pattron/Param-Bayer.xml ")
       + std::string("\%Im1=") + NameDCR(aCh1) + std::string(" ")
       + std::string("\%Im2=") + NameDCR(aCh2) + std::string(" ")
       + std::string("WorkDir=") + DirChantier() + std::string(" ")
    ;

    System(aCom);
    mCpls.push_back(cOneCplAB(aCh1,aCh2));
}


void cAppliBaric::FusioneRes()
{
    cBayerCalibGeom aRes;
    for (int aK=0 ; aK<int(mCpls.size());  aK++)
    {
       std::string aName =  
                DirChantier()
              + std::string("MEC/GridBayer-")
	      + StdPrefix(CurF1())
	      +  std::string("_")
	      + mCpls[aK].mC1
	      + std::string("-")
	      + StdPrefix(CurF1())
	      +  std::string("_")
	      + mCpls[aK].mC2
	      + std::string(".xml");

        cBayerGridDirecteEtInverse aBGDI = 
	     StdGetObjFromFile<cBayerGridDirecteEtInverse>
	     (
	          aName,
		  "include/XML_GEN/SuperposImage.xml",
		  "BayerGridDirecteEtInverse",
		  "BayerGridDirecteEtInverse"
	     );
        std::cout << aName << "\n";
	aRes.Grids().push_back(aBGDI);
    }
    MakeFileXML
    (
       aRes,
       DirChantier() +mNameCam
    );
}

void cAppliBaric::Exec()
{
   // RequireBin(ThisBin(),"bin/MICMAC","MakeMICMAC");
   // RequireBin(ThisBin(),"bin/MpDcraw","MakeMpDcraw");

   System
   (
        MMDir() + std::string("bin/MpDcraw  ")
       + DirChantier() + std::string(" ")
       + CurF1() + std::string(" ")
       + std::string(" 16B=1 Split=* Prefix=1")
   );

   MECOneCpl("R","V");
   MECOneCpl("R","B");
   MECOneCpl("R","W");

   FusioneRes();
}

std::string cAppliBaric::NameDCR(const std::string & aChan)
{
   return    std::string("MpDcraw16B_")
           + StdPrefix(CurF1())
	   + std::string("_")
	   + aChan
	   + std::string(".tif");
}

   //===========================================

int main(int argc,char ** argv)
{
    // system("make -f MakeMICMAC");

    cAppliBaric aAP(argc,argv);

    aAP.DoAll();

    // std::cout << aAP.Com(0) << "\n";
    // std::cout << aAP.Com(1) << "\n";
    // std::cout << aAP.Com(2) << "\n";
    // std::cout << aAP.Com(3) << "\n";

    Banniere_Baric();

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
