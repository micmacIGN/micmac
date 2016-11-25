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
#include <string>

class cCP4dMM_Appli
{
	public :
		cCP4dMM_Appli(int argc,char ** argv);
		void SplitLine(const std::string& aS, char aSeparator, std::vector<string>& aV);
		
	private :
		std::string mDir;
		std::string mFileIn;
		std::string mOut;
		
};

//Pix4D format : imagename, GCP/Manual Tie Pointlabel, image coordinate x, image coordinate y, (optional) zoom level, (optional) image directory
//Example :      R0079100.JPG,mtp31,2018.3314,1621.3729,1.9755, C:\Users\pix4d\Desktop\marks 

void cCP4dMM_Appli::SplitLine(const std::string& aS, char aSeparator, std::vector<std::string>& aV) 
{
   int i=0;
   int j=aS.find(aSeparator);

   while (j >= 0) 
   {
      aV.push_back(aS.substr(i,j-i));
      i=++j;
      j=aS.find(aSeparator, j);

      if (j < 0) 
      {
         aV.push_back(aS.substr(i, aS.length()));
      }
   }
}

cCP4dMM_Appli::cCP4dMM_Appli(int argc,char ** argv)
{
	
	std::string aExt="";
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mDir, "Directory")
					 << EAMC(mFileIn, "Pix4D image mesurements file"),
          LArgMain() << EAM(mOut,"Out", false, "Name output file ; Def=InputFile.xml")
					 << EAM(aExt,"Ext", false, "Extension to add at the end of the image name (whithout .)")
    );
    
    std::string aFullName = mDir+mFileIn;
    
    //name output (.xml) file
    if (mOut=="")
    {
		mOut = StdPrefixGen(mFileIn) + ".xml";
    }
    
    //read input file
    ifstream aFichier(aFullName.c_str());
    
    std::vector<std::string>* aVS = NULL;
    
    cSetOfMesureAppuisFlottants aDico;

    if(aFichier)
    {
		std::string aLigne;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLigne,'\n');
			
			if(aLigne.size() != 0)
			{
				aVS = new std::vector<std::string>();
				SplitLine(aLigne, ',', *aVS);
				
				// at least : imagename, GCP/Manual Tie Pointlabel, image coordinate x, image coordinate y
				if(aVS->size() >= 4)
				{
					cMesureAppuiFlottant1Im aMAF;
					
					if(aExt == "")
						aMAF.NameIm() = aVS->at(0);
					else
						aMAF.NameIm() = aVS->at(0) + "." + aExt;
					
					cOneMesureAF1I aOAF1I;
					aOAF1I.NamePt() = aVS->at(1);
					Pt2dr aPt;
					aPt.x = atof(aVS->at(2).c_str());
					aPt.y = atof(aVS->at(3).c_str());
					aOAF1I.PtIm() = aPt;
					
					aMAF.OneMesureAF1I().push_back(aOAF1I);
					
					aDico.MesureAppuiFlottant1Im().push_back(aMAF);
				}
			
			}
		}
		
		aFichier.close();
		
		MakeFileXML(aDico,mOut);
        std::cout<<mOut<<" written."<<std::endl;
	}
		
	else
    {
		std::cout<< "Error While opening file" << '\n';
	}
          
}

int ConvP4dIm2MM_main(int argc,char ** argv)
{
	 cCP4dMM_Appli anAppli(argc,argv);
     return EXIT_SUCCESS;
}




/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les mêmes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement,  a  l'utilisation,  a  la modification et/ou au
   developpement et a  la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a  des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a  charger  et  tester  l'adequation  du
   logiciel a  leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
