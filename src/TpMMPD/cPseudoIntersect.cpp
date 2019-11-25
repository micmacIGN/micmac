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

//classe d'application
class cPI_Appli;

//structure d'un couple (camera,coord 2d)
struct CamCoord {
	CamStenope * Cam;
	Pt2dr coord2d;
};

//structure d'un couple [ nom Pt, vecteur (camera,coord 2d) ]
struct PtAllInfos {
	std::string nom;
	std::vector<CamCoord> CAC;
};

class cPI_Appli
{
     public :
          cPI_Appli(int argc,char ** argv);
          void ShowArgs();
          Pt3dr IntersectionFaisceaux(
			                          const std::vector<CamStenope *> & aVCS,
			                          const std::vector<Pt2dr> & aNPts2D,
									  double&                    aVRes
		);
     private :
          std::list<std::string> mLFile;
          std::string mFullName;
          std::string mDir; 			//dossier courant
          std::string mPat; 			//pattern des images
          std::string aOriIn;			//dossier qui contient les orientations des images
		  std::string a2dPtsFile;		//fichier .xml qui contient les pointes des images (avec SaisieAppuisInit)
		  std::string aOut;				//nom fichier de sortie
          cInterfChantierNameManipulateur * mICNM;
        
};

cPI_Appli::cPI_Appli(int argc,char ** argv) 
{
     bool aShowArgs=false;
     bool aXmlExport=true;
     std::map<std::string,int> mCamNameMap;//structure to eliminate measurements outside the pattern
     Pt3dr aInc(1,1,1);
        
     
     ElInitArgMain
     (
          argc, argv,
          LArgMain() << EAMC(mFullName,"Full Name (Dir+Pat)")
					 << EAMC(aOriIn, "Directory of input orientation",  eSAM_IsExistDirOri)
					 << EAMC(a2dPtsFile, ".xml file of 2d points", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Name output file (def=3DCoords.txt)")
                     << EAM(aXmlExport,"XmlOut",true,"Export in .xml format to use as GCP file (Def=true)")
                     << EAM(aShowArgs,"Show",true,"Gives details on arguments (Def=false)")
     );

     MakeFileDirCompl(aOriIn);
     ELISE_ASSERT(ELISE_fp::IsDirectory(aOriIn),"ERROR: Input orientation not found!");

     SplitDirAndFile(mDir, mPat, mFullName);

     mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
     mLFile = mICNM->StdGetListOfFile(mPat);

     int aK=0;
     for (auto aNameCam : mLFile)
        mCamNameMap[aNameCam] = aK++;

    if (aShowArgs) ShowArgs();
    
    if(aOut == "")
    {
		aOut = "3DCoords.txt";
	}
    
    /*
    //vecteur de noms de cameras
    std::vector<string> OriFiles(mLFile.size());
    
    //vecteur de cameras
    std::vector<CamStenope *> aCamRep(mLFile.size());
    
    //initialisation du compteur
    int cmpt1=0;
    
    std::cout<<"Reading cameras..."<<std::flush;
    for (std::list<std::string>::iterator itS=mLFile.begin(); itS != mLFile.end(); itS ++)
    {
		//remplissage avec les noms des fichiers orientations
		OriFiles.at(cmpt1) = aOriIn+"Orientation-"+*itS+".xml";
		
        if (!ELISE_fp::exist_file(OriFiles.at(cmpt1)))
            continue;

        //generation des cameras à partir des fichiers d'orientations
		aCamRep.at(cmpt1) = CamOrientGenFromFile(OriFiles.at(cmpt1),mICNM);
		
        //incrementation du compteur
		cmpt1++;
	}
    std::cout<<"done!"<<std::endl;*/
    
    cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(a2dPtsFile,SetOfMesureAppuisFlottants);
    
    std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();
    
    std::vector<ElSeg3D> aVSeg;
    
    std::vector<PtAllInfos> vPtsAI;
    
    //contient la liste des points a transformer en 3d
    std::vector<string> vNamePts;		
    
    std::cout<<"Reading points..."<<std::flush;
    for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end(); iT1++)
    {
        //take into account only images in the pattern
        const std::string aImName = iT1->NameIm();
        if (DicBoolFind(mCamNameMap,aImName))
        {
            std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();


            for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2 ++)
            {
                std::string aNamePt = iT2->NamePt();
                vNamePts.push_back(aNamePt);
            }
        }
	}
    std::cout<<"done!"<<std::endl;

    std::cout<<"Sorting points..."<<std::flush;
	//tri
	std::sort(vNamePts.begin() , vNamePts.end());
	
	//on vire les doublons
	vNamePts.erase(std::unique(vNamePts.begin(), vNamePts.end()),vNamePts.end());	
    std::cout<<"done! "<<vNamePts.size()<<" points found."<<std::endl;

    std::cout<<"Making bundles groups..."<<std::flush;
	//boucle sur le nombre de points à projeter en 3D
    for (unsigned int aKNP=0; aKNP<vNamePts.size(); aKNP++)
	{
		PtAllInfos aPtsAL;
		
		std::vector<CamCoord> vCameraEtCoord;
		
		std::string namePt = vNamePts.at(aKNP);
		
		aPtsAL.nom = namePt;
        //std::cout<<"Recherche du point "<<namePt<<"\n";
		//boucle sur les images
		for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end() ; iT1++)
		{

            //take into account only images in the pattern
            const std::string aImName = iT1->NameIm();
            if (DicBoolFind(mCamNameMap,aImName))
            {

				std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();
                //std::cout<<" Image "<<iT1->NameIm()<<":\n";
				//boucle sur tous les points saisis sur l'image courante
				for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
				{
					
                    //std::cout<<"  - pt "<<iT2->NamePt()<<": ";
                    //si je tombe sur le point courant je dois ajouter la camera + coord dans cette camera
                    if(namePt == iT2->NamePt())
					{
						std::string oriNameFile = aOriIn+"Orientation-"+iT1->NameIm()+".xml";
                        //std::cout<<"ok "<<oriNameFile<<" ";
                        if (!ELISE_fp::exist_file(oriNameFile)) continue;
						CamStenope * cameraCourante = CamOrientGenFromFile(oriNameFile,mICNM);
						Pt2dr coordCourant = iT2->PtIm();
						CamCoord aCameraEtCoord;
						aCameraEtCoord.Cam = cameraCourante;
						aCameraEtCoord.coord2d=coordCourant;
						vCameraEtCoord.push_back(aCameraEtCoord);
                        //std::cout<<"("<<vCameraEtCoord.size()<<")\n";
                    }
                    //else std::cout<<"non\n";
                }
            }
		}
		
		aPtsAL.CAC = vCameraEtCoord;
        if ( int(vCameraEtCoord.size()) >= 2 )
        {
		    vPtsAI.push_back(aPtsAL);
            std::cout<<aPtsAL.CAC.size()<<" ";
        }
    }
    std::cout<<"done!"<<std::endl;
	
    //le vecteur des points 3d a exporter
	std::vector<Pt3dr> Pts3d;

	//les residuus
	std::vector<double> aVRes;

	//le nombre d'observations
	std::vector<int>    aVObs;

    //boucle sur le nombre de points a projeter en 3d
	for(unsigned int aHG=0 ; aHG<vPtsAI.size() ; aHG++)
	{
        //vecteur de cameras
		std::vector<CamStenope *> vCSPt;
		
        //vecteur de coordonnees 2d
		std::vector<Pt2dr> vC2d;
		
		for(unsigned int aHF=0 ; aHF<vPtsAI.at(aHG).CAC.size() ; aHF++)
		{
			CamStenope * aCSPtC = vPtsAI.at(aHG).CAC.at(aHF).Cam;
			vCSPt.push_back(aCSPtC);
			Pt2dr aCoordPtC = vPtsAI.at(aHG).CAC.at(aHF).coord2d;
			vC2d.push_back(aCoordPtC);
        }

        //if (vC2d.size()<3) continue;
		double aResidu;
        Pt3dr aPt3d = IntersectionFaisceaux(vCSPt,vC2d,aResidu);
        Pts3d.push_back(aPt3d);
		aVRes.push_back(aResidu);
		aVObs.push_back(int(vCSPt.size()));
	}

	
	//export en .txt
	if (!MMVisualMode)
	{
		
		FILE * aFP = FopenNN(aOut,"w","PLY2XYZ_main");
		cElemAppliSetFile aEASF(mDir + ELISE_CAR_DIR + aOut);
		for(unsigned int aVP=0; aVP<Pts3d.size(); aVP++)
		{
			fprintf(aFP,"%s %lf %lf %lf %lf %d \n",vPtsAI[aVP].nom.c_str(),Pts3d[aVP].x,Pts3d[aVP].y,Pts3d[aVP].z, aVRes.at(aVP), aVObs.at(aVP));
			//~ std::cout << vPtsAI.at(aVP).nom << " " << Pts3d.at(aVP).x << " " << Pts3d.at(aVP).y << " " << Pts3d.at(aVP).z << "\n" ;
		}
		
        ElFclose(aFP);
        std::cout<<aOut<<" written."<<std::endl;
	}
	

    //export en .ply
    if (!MMVisualMode)
    {

        FILE * aFP = FopenNN(aOut+".ply","w","PLY2XYZ_main");
        //cElemAppliSetFile aEASF(mDir + ELISE_CAR_DIR + aOut);
        fprintf(aFP,"ply\n");
        fprintf(aFP,"format ascii 1.0\n");
        fprintf(aFP,"element vertex %lu\n",Pts3d.size());
        fprintf(aFP,"property float x\n");
        fprintf(aFP,"property float y\n");
        fprintf(aFP,"property float z\n");
        fprintf(aFP,"property uchar red\n");
        fprintf(aFP,"property uchar green\n");
        fprintf(aFP,"property uchar blue\n");
        fprintf(aFP,"element face 0\n");
        fprintf(aFP,"property list uchar int vertex_indices\n");
        fprintf(aFP,"end_header\n");

        for(unsigned int aVP=0; aVP<Pts3d.size(); aVP++)
        {
            fprintf(aFP,"%lf %lf %lf 255 0 0\n",Pts3d[aVP].x,Pts3d[aVP].y,Pts3d[aVP].z);
        }

        ElFclose(aFP);
        std::cout<<aOut<<".ply written."<<std::endl;
    }
	
	//export en .xml pour utiliser comme fichier de GCPs
	if(aXmlExport)
	{
		std::string aOutXml = StdPrefixGen(aOut) + ".xml";
		
		cDicoAppuisFlottant aDico;
	
		for (unsigned int aKP=0 ; aKP<Pts3d.size() ; aKP++)
		{
			cOneAppuisDAF aOAD;
			aOAD.Pt() = Pts3d[aKP];
			aOAD.NamePt() = vPtsAI.at(aKP).nom.c_str();
			aOAD.Incertitude() = aInc;
						
			aDico.OneAppuisDAF().push_back(aOAD);
		}

		MakeFileXML(aDico,aOutXml);
        std::cout<<aOutXml<<" written."<<std::endl;
	}

    std::cout<<"Finished!"<<std::endl;
		
}

void cPI_Appli::ShowArgs()
{
		  std::cout << "DIR = " << mDir << "\n";
		  std::cout << "Pat = " << mPat << "\n";
		  
          std::cout << "Nb Files " << mLFile.size() << "\n";
          for (
               std::list<std::string>::iterator itS=mLFile.begin();
               itS != mLFile.end();
               itS ++)
                    { std::cout << "     File = " << *itS << "\n"; }

}


Pt3dr cPI_Appli::IntersectionFaisceaux
	   (
			const std::vector<CamStenope *> & aVCS,
			const std::vector<Pt2dr> & aNPts2D,
			double &                   aResidu
		)
{
	aResidu=0;

	int aNb = int(aVCS.size());

	//vecteur d'éléments segments 3d
	std::vector<ElSeg3D> aVSeg;
	
	for (int aKR=0 ; aKR < aNb ; aKR++)
	{
		ElSeg3D aSeg = aVCS.at(aKR)->F2toRayonR3(aNPts2D.at(aKR));
		aVSeg.push_back(aSeg);
	}
    std::cout<<"Intersect "<< aNb <<" bundles...\n";
	Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);

	for (int aKR=0 ; aKR < aNb ; aKR++)
    {
		aResidu += aVSeg.at(aKR).DistDoite(aRes);
	}
	aResidu /= aNb;

    return aRes;
}

int PseudoIntersect_main(int argc,char **argv)
{
    
     cPI_Appli anAppli(argc,argv);
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
