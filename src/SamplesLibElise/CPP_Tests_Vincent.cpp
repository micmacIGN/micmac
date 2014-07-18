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

// Main header in which a lot of libraries are included
#include "StdAfx.h"
#if (ELISE_unix)
	#include "dirent.h"
#endif

// List  of classes

class cCMP_Ima;
class cCMP_Appli;

class cCMP_Ima
{
	public :
		cCMP_Ima(cCMP_Appli & anAppli,const std::string & aName, const std::string & aOri);
		Pt3dr ManipImage();
	//private :
		cCMP_Appli &   mAppli;
		CamStenope *	mCam;
		std::string mNameOri;
		std::string	 mName;
};

class cCMP_Appli
{
	public :
		// Main function
		cCMP_Appli(int argc, char** argv);
		cInterfChantierNameManipulateur * ICNM() const {return mICNM;}
		std::string NameIm2NameOri(const std::string &, const std::string &) const;
	private :
		std::string mOri1;
		std::string mOri2;
		std::string mOut;
		std::string mFullName;
		std::string mPat1;
		std::string mPat2;
		std::string mDir1;
		std::string mDir2;
		std::list<std::string> mLFile1;
		std::list<std::string> mLFile2;
		cInterfChantierNameManipulateur * mICNM;
		std::vector<cCMP_Ima *>		  mIms1;
		std::vector<cCMP_Ima *>		  mIms2;
		// const std::string & Dir() const {return mDir;}
};

/********************************************************************/
/*																  */
/*		 cCMP_Ima												 */
/*																  */
/********************************************************************/

cCMP_Ima::cCMP_Ima(cCMP_Appli & anAppli,const std::string & aName, const std::string & aOri) :
   mAppli  (anAppli),
   mName   (aName)
{
	mNameOri  = mAppli.NameIm2NameOri(mName,aOri);
	mCam	  = CamOrientGenFromFile(mNameOri,mAppli.ICNM());

}

Pt3dr cCMP_Ima::ManipImage()
{
	Pt3dr mCentreCam = mCam->VraiOpticalCenter();
	return mCentreCam;
}


/********************************************************************/
/*																  */
/*		 cCMP_Appli											   */
/*																  */
/********************************************************************/


cCMP_Appli::cCMP_Appli(int argc, char** argv)
{
	// Initialisation of the arguments

	ElInitArgMain
	(
		argc,argv,
		LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
					<< EAMC(mOri1,"First Orientation")
					<< EAMC(mOri2,"Second Orientation"),
		LArgMain()  << EAM(mOut,"Out",true,"Output result file")
	);

	// Initialize name manipulator & files
	SplitDirAndFile(mDir1,mPat1,mFullName);

	cout << "Ori1 : " << mOri1 << "\tOri2 : " << mOri2 << "\tOut : " << mOut << endl;
	// Get the list of files from the directory and pattern
	mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir1);
	mLFile1 = mICNM->StdGetListOfFile(mPat1);
	// If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
	StdCorrecNameOrient(mOri1,mDir1);

	SplitDirAndFile(mDir2,mPat2,mFullName);
	// Get the list of files from the directory and pattern
	mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir2);
	mLFile2 = mICNM->StdGetListOfFile(mPat2);
	// If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
	StdCorrecNameOrient(mOri2,mDir2);

	if (mOut == ""){mOut=mOri1+"_"+mOri2+".txt";};

	for (
			  std::list<std::string>::iterator itS=mLFile1.begin();
			  itS!=mLFile1.end();
			  itS++
			  )
	 {
		   cCMP_Ima * aNewIm1 = new  cCMP_Ima(*this,*itS,mOri1);
		   cCMP_Ima * aNewIm2 = new  cCMP_Ima(*this,*itS,mOri2);

		   mIms1.push_back(aNewIm1);
		   mIms2.push_back(aNewIm2);
	 }

	 Pt3dr mDiffCentre;
	 vector <Pt3dr> mVDCentre;
	 for (unsigned int i = 0 ; i < mIms1.size() ; i++)
	 {
		if (mIms1[i]->mName != mIms2[i]->mName) { cout << "!!!!!!!!! NOMS D'IMAGES INCOHÉRENTS !!!!!!!!!" << endl;}
		else
		{
		   Pt3dr mCentre1 = mIms1[i]->ManipImage();
		   Pt3dr mCentre2 = mIms2[i]->ManipImage();
		   mDiffCentre = mCentre1 - mCentre2;
		   cout << "Image : " << mIms1[i]->mName << " " << mDiffCentre << endl;
		   mVDCentre.push_back(mDiffCentre);
		}
	 }
/*
double dX(0),dY(0),dZ(0);
	 for (unsigned int i = 0 ; i < mVDCentre.size() ; i++)
	 {

		cout << "dX = " << dX  << "\tdY = " << dY << "\tdZ = " << dZ << endl;
		dX = dX + abs(mVDCentre[i].x);
		dY = dY + abs(mVDCentre[i].y);
		dZ = dZ + abs(mVDCentre[i].z);
	 }
	 cout << "Ecarts moyen absolus en X : " << dX/mVDCentre.size()
		  << "\t en Y : " << dY/mVDCentre.size()
		  << "\ten Z : " << dZ/mVDCentre.size() << endl;
*/
}


std::string cCMP_Appli::NameIm2NameOri(const std::string & aNameIm, const std::string & aOri) const
{
	return mICNM->Assoc1To1
	(
		"NKS-Assoc-Im2Orient@-"+aOri+"@",
		aNameIm,
		true
	);
}


/********************************************************************/
/*																  */
/*		 cTD_Camera											   */
/*																  */
/********************************************************************/

// Main exercise
int CheckOri_main(int argc, char** argv)
{
/* Retourne, à partir de 2 orientations, les différences en X, Y, Z associées à chaque caméra */
   cCMP_Appli anAppli(argc,argv);

   return EXIT_SUCCESS;
}

Pt3dr SplitToPt3dr(string inS)
{
	inS=inS.substr(1,inS.size()-2); 	// delete [ & ]
	double rX,rY,rZ;
	Pt3dr myPoint;
	for (unsigned int i =0 ; i < inS.size() ; i++)
	{
		if (inS[i] == ',')
		{
			std::string inS2 = inS.substr(0,i);
			rX = atof(inS2.c_str());
			myPoint.x=rX;
			inS=inS.substr(i+1,inS.size()-i-1);
		}
	}
	for (unsigned int i =0 ; i < inS.size() ; i++)
	{
		if (inS[i] == ',')
		{
			std::string inS2 = inS.substr(0,i);
			rY = atof(inS2.c_str());
			myPoint.y=rY;
			inS=inS.substr(i+1,inS.size()-i-1);
			rZ = atof(inS.c_str());
			myPoint.z=rZ;
		}
	}
	
	return myPoint;
}

int ResToTxt_main(int argc, char** argv)
{
/* Transforme résidus de GCPBascule (utiliser GCPBasc ... | tee myFile.txt) en un fichier "NamePt dX dY dZ sigmaX sY sZ eMoyPixel eMaxPixel")*/
	string mNameIn, mNameOut("");
	ElInitArgMain
	(
		argc,argv,
		LArgMain()  << EAMC(mNameIn,"Name of the residuals file"),
		LArgMain()  << EAM(mNameOut,"Out",true,"File to save the results")
	);
	
	ifstream fin (mNameIn.c_str());
	ofstream fout (mNameOut.c_str(),ios::out);
	string 	 mRead;
	int 	 i=0;
	double   rImMoy(0),
			 rXmoy(0),
			 rYmoy(0),
			 rZmoy(0);
	Pt3dr 	 ptRes, 
			 ptPres;
	vector <Pt3dr> ptResLs;
	vector <double> rImMoyLs,
					rImMaxLs;
			 
	while (fin >> mRead){

		if (mRead == "--NamePt"){i++;}
		if (i!=0){i++;}
		if (i==3){fout << mRead << " ";}	// Id
		if (i==6)		// [rX,rY,rZ]
		{
			ptRes = SplitToPt3dr (mRead);
			ptResLs.push_back(ptRes);
			fout << ptRes.x << " " << ptRes.y << " " << ptRes.z << " ";
		}
		if (i==13)
		{				// [pX,pY,pZ]
			ptPres = SplitToPt3dr (mRead);
			fout << ptPres.x << " " << ptPres.y << " " << ptPres.z << " ";
		}
		if (i==20)		// rImMoy
		{
			fout << mRead << " ";
			rImMoyLs.push_back(atof(mRead.c_str()));
		}
		if (i==25)
		{				// rImMax
			fout << mRead << "\n";
			rImMoyLs.push_back(atof(mRead.c_str()));
		}
		if (mRead == "For"){i=0;}
	}
	
	for (unsigned int i=0;i<ptResLs.size();i++)
	{
		rXmoy += fabs(ptResLs[i].x);
		rYmoy += fabs(ptResLs[i].y);
		rZmoy += fabs(ptResLs[i].z);
		rImMoy += rImMoyLs[i];
	}
	rXmoy = rXmoy/ptResLs.size();
	rYmoy = rYmoy/ptResLs.size();
	rZmoy = rZmoy/ptResLs.size();
	double rXYZ = sqrt(rXmoy*rXmoy + rYmoy*rYmoy + rZmoy*rZmoy);
	rImMoy = rImMoy/ptResLs.size();
	
	fout << "\nMEAN ABSOLUTE ERROR :\n"
		 << "X : " << rXmoy
		 <<" m\nY : " << rYmoy
		 <<" m\nZ : " << rZmoy
		 <<" m\nXYZ : " << rXYZ
		 <<" m\nImage : " << rImMoy << " pixel\n";
		  
	cout << "\nMEAN ABSOLUTE ERROR :\n"
		 << "X : " << rXmoy
		 <<" m\nY : " << rYmoy
		 <<" m\nZ : " << rZmoy
		 <<" m\nXYZ : " << rXYZ
		 <<" m\nImage : " << rImMoy << " pixel\n";
	fout.close();
   return EXIT_SUCCESS;
}

void WriteAppuis(vector <pair <std::string,float> > ListOfDiffAppuis, ofstream &fout)
{
	float AbsSumDiffAppuis=0;
	fout << "\nCONTROL POINTS : \n";
	for
	(
		unsigned int i=0;
		i<ListOfDiffAppuis.size();
		i++
	)
	{
		fout << ListOfDiffAppuis[i].first << "\t" << ListOfDiffAppuis[i].second << endl;
		AbsSumDiffAppuis += fabs(ListOfDiffAppuis[i].second);
	}
	fout << "MEAN ABSOLUTE ERROR ON CONTROL POINTS = " << AbsSumDiffAppuis/ListOfDiffAppuis.size()<<endl;
}

void WriteControl(vector <pair <std::string,float> > ListOfDiffControl, ofstream &fout)
{
	fout << "\nCHECK POINTS : \n";
	float AbsSumDiffControl=0;
	for
	(
		unsigned int i=0;
		i<ListOfDiffControl.size();
		i++
	)
	{
		fout << ListOfDiffControl[i].first << "\t" << ListOfDiffControl[i].second << endl;
		AbsSumDiffControl += fabs(ListOfDiffControl[i].second);
	}
	fout << "MEAN ABSOLUTE ERROR ON CHECK POINTS = " << AbsSumDiffControl/ListOfDiffControl.size() << endl; 
}

vector <string> GetFilesFromFolder (string dossier)
{
		vector <string> dirName;
#if ELISE_unix
		DIR* rep = NULL;
		struct dirent* fichierLu = NULL;
		rep = opendir(dossier.c_str());
		if (rep == NULL) 
			exit(1); 
	
		while ((fichierLu = readdir(rep)) != NULL){
			dirName.push_back(fichierLu->d_name);}
		sort(dirName.begin(),dirName.end());
		
#endif
	return dirName;
}

void Idem_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************\n";
    std::cout <<  " *     I-nterpolate              *\n";
    std::cout <<  " *     D-ense                    *\n";
    std::cout <<  " *     E-levation                *\n";
    std::cout <<  " *     M-odel                    *\n";
    std::cout <<  " *********************************\n\n";
}

int Idem_main(int argc, char** argv)
{
	std::string aGCP, 
				aMNT, 
				aOrthoName(" "),
				aNameFileTxt(" "), 
				aDir, 
				aPat, 
				aMesIm; 
	bool treatGCP(true), treatCP(true);
	int aPtSz(5),
		aImSize(3000);
		
	ElInitArgMain
	(
		argc,argv,
		LArgMain()  << EAMC(aMNT,"DEM xml file (NuageImProf...)")
					<< EAMC(aGCP,"Ground Control Points File")
					<< EAMC(aMesIm,"Image measurements file"),
		LArgMain()	<< EAM(aNameFileTxt,"Out",true,"File to store the results")
					<< EAM(aOrthoName,"Ortho",true,"Display the results on a video window")
					<< EAM(aImSize,"ImSz",true,"Rescaled ortho size ( default : 3000)")
					<< EAM(treatGCP,"GCP",true,"Interpolate on Ground Control Points {Def = true}")
					<< EAM(treatCP,"CP",true,"Interpolate on Check Points {Def = true}")
					<< EAM(aPtSz,"PtSz",true,"Size of the point (default : 10)")
	);
	ELISE_ASSERT(treatGCP || treatCP,"Either GCP or CP must be true");
	ELISE_ASSERT(aImSize>100,"Probable confusion with Final Size argument");

// Charger les GCP, calculer leur projection dans l'ortho et le MNT
	cDicoAppuisFlottant cDico=  StdGetFromPCP(aGCP,DicoAppuisFlottant);
	std::cout << "Nb Pts " <<  cDico.OneAppuisDAF().size() << "\n\n";
	std::list<cOneAppuisDAF> & aLGCP =  cDico.OneAppuisDAF();


// Charger les mesures images
	cSetOfMesureAppuisFlottants dDico=  StdGetFromPCP(aMesIm,SetOfMesureAppuisFlottants);
	std::list<cMesureAppuiFlottant1Im> & dLGCP =  dDico.MesureAppuiFlottant1Im();
	vector <std::string> aListOfApp;
	
	for 
	(
		 std::list<cMesureAppuiFlottant1Im>::iterator iT1= dLGCP.begin();
		 iT1 != dLGCP.end();
		 iT1++
	)
	{
		for
		(
			std::list<cOneMesureAF1I>::iterator iT2 = iT1->OneMesureAF1I().begin();
			iT2 != iT1->OneMesureAF1I().end();
			iT2++
		)
		{
			for 
			(
				std::list<cOneAppuisDAF>::iterator iT3= aLGCP.begin();
				iT3 != aLGCP.end();
				iT3++
			)
			{
				if (iT2->NamePt() == iT3->NamePt())
				{
					aListOfApp.push_back(iT3->NamePt());
				}
			}
		}
	}
	
	std::sort (aListOfApp.begin(), aListOfApp.end());
	vector<std::string> bListOfApp;
	bListOfApp.push_back(aListOfApp[0]);
	int i=0;
	
	for
	(
		vector <std::string>::iterator iT = aListOfApp.begin();
		iT != aListOfApp.end();
		iT++
	)
	{
		if (bListOfApp[i] != *iT)
		{
			bListOfApp.push_back(*iT);
			i++;
		}
	}	
	aListOfApp.clear();
	
	
// Récupération du dernier fichier Z_Num.xml
	std::string aDir2,aPat2,aZ_Num;
	SplitDirAndFile(aDir2,aPat2,aMNT);
	vector<std::string> aListOfFileInMEC;
	if (ELISE_unix)
	{
		aListOfFileInMEC = GetFilesFromFolder(aDir2);
	}
	
	for (unsigned int i=0;i<aListOfFileInMEC.size();i++)
	{
		if 
		(
			(aListOfFileInMEC[i].substr(0,5) == "Z_Num")
			&&
			(aListOfFileInMEC[i].substr(aListOfFileInMEC[i].size()-3,3) == "xml")
		)
		{
			aZ_Num = aDir2 + aListOfFileInMEC[i];
		}
	}
	
	cFileOriMnt bDico = StdGetFromPCP(aZ_Num,FileOriMnt);
	Pt2dr aOrgMNT = bDico.OriginePlani();
	Pt2dr aResMNT = bDico.ResolutionPlani();


// Interpole le MNT à l'emplacement de chaque GCP
 	cElNuage3DMaille *  bMNT = cElNuage3DMaille::FromFileIm(aMNT);			 
	Pt2di  aGCPinMNT;
	pair <std::string,float> aMNTinterpoled;
	vector <pair <std::string,float> > aListOfDiff;
	
	for 
	(
		 std::list<cOneAppuisDAF>::iterator iT= aLGCP.begin();
		 iT != aLGCP.end();
		 iT++
	)
	{
		aGCPinMNT.x = round_ni((iT->Pt().x - aOrgMNT.x) / aResMNT.x);
		aGCPinMNT.y = round_ni((iT->Pt().y - aOrgMNT.y) / aResMNT.y);
		
		if (bMNT->IndexHasContenu(aGCPinMNT))
		{
			Pt3dr aPTer = bMNT->PtOfIndex(aGCPinMNT);
			aMNTinterpoled.first = iT->NamePt();
			aMNTinterpoled.second = iT->Pt().z - aPTer.z;
			aListOfDiff.push_back(aMNTinterpoled);
		}
	}
	
	
// Différence entre appuis et contrôle	
	vector <pair <std::string,float> > ListOfDiffAppuis, ListOfDiffControl;
	for
	(
		unsigned int i=0;
		i < aListOfDiff.size();
		i++
	)
	{
		bool isAppuis=false;
		
		for
		(
			unsigned int j=0;
			j < bListOfApp.size();
			j++
		)
		{
			if (bListOfApp[j] == aListOfDiff[i].first)
			{
				ListOfDiffAppuis.push_back(aListOfDiff[i]);		// [Id] [dZ]
				isAppuis=true;
			}
		}
		
		if (!isAppuis)
		{
			ListOfDiffControl.push_back(aListOfDiff[i]);		// [Id] [dZ]
		}
	}
	
	float AbsSumDiffControl=0, AbsSumDiffAppuis=0;
	
	if (treatGCP)
	{
		for
		(
			unsigned int i=0;
			i<ListOfDiffAppuis.size();
			i++
		)
		{
			cout << "Control point : " << ListOfDiffAppuis[i].first << "\t" << "Difference between xml & DEM : " << ListOfDiffAppuis[i].second << endl;
			AbsSumDiffAppuis += fabs(ListOfDiffAppuis[i].second);
		}
		cout << "MEAN ABSOLUTE ERROR ON CONTROL POINTS = " << AbsSumDiffAppuis/ListOfDiffAppuis.size() << endl << endl << "Results saved in : " << aNameFileTxt << endl; 
	}
	
	if (treatCP)
	{
		for
		(
			unsigned int i=0;
			i<ListOfDiffControl.size();
			i++
		)
		{
			cout << "Check point : " << ListOfDiffControl[i].first << "\t" << "Difference between xml & DEM : " << ListOfDiffControl[i].second << endl;
			AbsSumDiffControl += fabs(ListOfDiffControl[i].second);
		}
		cout << "MEAN ABSOLUTE ERROR ON CHECK POINTS = " << AbsSumDiffControl/ListOfDiffControl.size()<<endl<<endl;
	}
	
	if (aNameFileTxt != " ")
	{
		ofstream fout (aNameFileTxt.c_str(),ios::out);
		fout << "Difference between altitude in xml file, and altitude in DEM (Id	 dZ)\n";
		if (treatGCP){WriteAppuis(ListOfDiffAppuis,fout);}
		if (treatCP){WriteControl(ListOfDiffControl,fout);}
	}

// Ecriture des résultats sur l'orthophoto (option)
	if (aOrthoName != " ")
	{		
		SplitDirAndFile(aDir,aPat,aOrthoName);
		std::string aMTD = aDir + "MTDOrtho.xml";
	// Recup param ortho
		cFileOriMnt aDico = StdGetFromPCP(aMTD,FileOriMnt);
		Pt2di aNbPixel = aDico.NombrePixels();
		Pt2dr aOrgOrt = aDico.OriginePlani();
		Pt2dr aResOrt = aDico.ResolutionPlani();

	// Calculer facteur d'échelle, lancer ScaleIm
		if ((aNbPixel.x > aImSize) | (aNbPixel.y > aImSize))
		{
			cout << "Rescaling the orthophoto..." << endl;
			int aSzMax;
			if (aNbPixel.y > aNbPixel.x)
			{
				aSzMax=aNbPixel.y;
			}
			else{aSzMax=aNbPixel.x;}
			double aScFactor = 1;
			aScFactor = aSzMax/aImSize;
			cout << "aScFactor = " << aScFactor << endl;
			stringstream ss;
			ss << aScFactor;
			
			std::string aScIm = "mm3d ScaleIm " + aOrthoName + " " + ss.str() ;
			VoidSystem(aScIm.c_str());
			
			aNbPixel.x /= aScFactor;
			aNbPixel.y /= aScFactor;
			aResOrt.x *= aScFactor;
			aResOrt.y *= aScFactor;
			aOrthoName = aOrthoName.substr(0,aOrthoName.size()-4)+"_Scaled.tif";
		}
	// Projection des GCP dans l'ortho
		vector <pair <std::string,Pt3dr> > aGCPground;
		vector <pair <std::string,Pt2di> > aGCPortho; 
		pair <std::string,Pt2di> toPushGCP;
		int aXproj, aYproj;
		
		for 
		(
			 std::list<cOneAppuisDAF>::iterator iT= aLGCP.begin();
			 iT != aLGCP.end();
			 iT++
		)
		{
			aXproj = round_ni((iT->Pt().x - aOrgOrt.x) / aResOrt.x);
			aYproj = round_ni((iT->Pt().y - aOrgOrt.y) / aResOrt.y);
			
			if 
			(
				(aXproj>=0) &
				(aYproj>=0) &
				(aXproj<aNbPixel.x) &
				(aYproj<aNbPixel.y)
			)
			{
				toPushGCP.first = iT->NamePt();
				toPushGCP.second.x = aXproj;
				toPushGCP.second.y = aYproj;
				aGCPortho.push_back(toPushGCP);			//Liste des points visibles sur l'ortho (dans la box)
			}
		}
		
		
	// Tracer des cercles sur les GCP dans l'ortho (réduite ou non)
		Tiff_Im  aImOrtho = Tiff_Im::StdConv(aOrthoName);
		std::string aNameOut = "MyQualityTime.tif";
		Im2D_U_INT1 I(aNbPixel.x,aNbPixel.y);
		
		 //  palette allocation
        Disc_Pal  Pdisc = Disc_Pal::P8COL();
        Gray_Pal  Pgr (30);
        Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
        RGB_Pal   Prgb  (5,5,5);
        Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));

        // Drawing with Elise
		Video_Display Ecr((char *) NULL);
        Ecr.load(SOP);
        Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(aNbPixel.x,aNbPixel.y));

        W.set_title("GCP in Red ; CP in blue");

		ELISE_COPY
		(
			I.all_pts(),
			aImOrtho.in(),
			W.out(Prgb)
		);
		
		bool isAppuis=false; 
		for 
		(
			unsigned int i=0;
			i<aGCPortho.size();
			i++
		)
		{   
			isAppuis=false;
			for
			(
				unsigned int j =0;
				j < ListOfDiffAppuis.size();
				j++
			)
			{
				if (aGCPortho[i].first == ListOfDiffAppuis[j].first)
				{
					isAppuis=true;
				}
			}
			
			Pt2dr aPtOrt (aGCPortho[i].second);
			
			if (isAppuis && treatGCP)
			{
				cout << "Id = " << aGCPortho[i].first << endl;
				ELISE_COPY
				(
					disc(aPtOrt,aPtSz),
					P8COL::red,
					W.out(Pdisc)
				);
			}
			else if (!isAppuis && treatCP)
			{
				cout << "Id = " << aGCPortho[i].first << endl;
				ELISE_COPY
				(
					disc(aPtOrt,aPtSz),
					P8COL::blue,
					W.out(Pdisc)
				);
			}
		}
		W.clik_in();
		
	}

	Idem_Banniere();
	
	return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
