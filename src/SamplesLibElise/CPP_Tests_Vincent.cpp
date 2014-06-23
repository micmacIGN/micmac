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
        CamStenope *    mCam;
        std::string mNameOri;
        std::string     mName;
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
        std::vector<cCMP_Ima *>          mIms1;
        std::vector<cCMP_Ima *>          mIms2;
        // const std::string & Dir() const {return mDir;}
};

/********************************************************************/
/*                                                                  */
/*         cCMP_Ima                                                 */
/*                                                                  */
/********************************************************************/

cCMP_Ima::cCMP_Ima(cCMP_Appli & anAppli,const std::string & aName, const std::string & aOri) :
   mAppli  (anAppli),
   mName   (aName)
{
    mNameOri  = mAppli.NameIm2NameOri(mName,aOri);
    mCam      = CamOrientGenFromFile(mNameOri,mAppli.ICNM());

}

Pt3dr cCMP_Ima::ManipImage()
{
    Pt3dr mCentreCam = mCam->VraiOpticalCenter();
    return mCentreCam;
}


/********************************************************************/
/*                                                                  */
/*         cCMP_Appli                                               */
/*                                                                  */
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
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

// Main exercise
int Vincent_main1(int argc, char** argv)
{
/* Retourne, à partir de 2 orientations, les différences en X, Y, Z associées à chaque caméra */
   cCMP_Appli anAppli(argc,argv);

   return EXIT_SUCCESS;
}

int Vincent_main2(int argc, char** argv)
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
    string mRead;
    int i=0;
    ofstream fout (mNameOut.c_str(),ios::out);
    while (fin >> mRead){

        if (mRead == "--NamePt"){i++;}
        if (i!=0){i++;}
        if (i==3){fout << mRead << " ";}
        if (i==6){fout << mRead << " ";}
        if (i==13){fout << mRead << " ";}
        if (i==20){fout << mRead << " ";}
        if (i==25){
            fout << mRead << "\n";}
        if (i!=0){cout << mRead << " at " << i << endl;}
        if (mRead == "For"){i=0;}
    }
    fout.close();
   return EXIT_SUCCESS;
}

int Vincent_main(int argc, char** argv)
{
	
	/* QualityTime : programme d'analyse qualitative des résultats générés.
	 * S'éxécute après une boucle de travail complète (MicMac/Malt en géométrie terrain, orthophoto générée)
	 * Le Programme prend en entrée : GCP.xml et NuageImProf-Etape9.xml; 
		* Option : Nom de Ortho-MEC-Malt ; Nom du fichier de mesures image (pour distinguer appuis/contrôle), taille  de l'image
     * Le programme en sortie va créer un dossier dans lequel :
		* Un fichier .txt de la forme :
		    * Points d'appuis (utilisés pendant le calcul)
			    * Nom du point	Différence entre altitude dans le fichier, et altitude obtenue en interpolant le MNT
		    * Idem pour les points de contrôle (si option ImageMeasure indiquée, alors on distingue contrôle/appuis)
		    * Calcul de l'écart moyen absolu, 
		* Génère des imagettes individuelles (256x256pixels) autour de chaque GCP, une croix(16x16) au centre, en haut : nom du point [différence entre xml et mnt]
		      * ULTIME : colore le MNT à pas régulier, dessine une échelle de répartitions des couleurs/altitudes sur la droite de l'image (to8bits "maîtrisé")
		* Orthophoto réduite avec emplacement des GCP (en rouge les points d'appuis, en bleu les points de contrôle), nom du point (option)	*/
 
	std::string aGCP, aMNT, aDirOF("./Ortho-MEC-Malt/"), aMesIm; 
	bool aAddNamePt(true);
	int aPtSz(10),aImSize(3000);
	
	ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aGCP,"Ground Control Points File")
					<< EAMC(aMNT,"DSM xml file (NuageImProf...)"),
        LArgMain()  << EAM(aDirOF,"DirOF",true,"Subdirectory for ortho (def in Ortho-${DirMEC})")
					<< EAM(aMesIm,"ImMeasure",true,"Image measurements file (to distinguish control & check points)")
					<< EAM(aAddNamePt,"NamePt",true,"Add the name of the point (default : true)")
					<< EAM(aImSize,"ImSz",true,"Rescaled ortho size ( default : 3000)")
					<< EAM(aPtSz,"PtSz",true,"Size of the point (default : 10)")
    );
    
    ELISE_ASSERT(aImSize<10,"Probable confusion with Final Size argument");
    
// Charger MTD-Ortho.xml
    std::string aMTD = aDirOF + "MTDOrtho.xml";
    ifstream fin (aMTD.c_str());
    
// Récupérer taille de l'ortho, nom de l'ortho, origineX, origineY, resolutionXY, origineZ, resolutionZ
    std::string aNameOrthoFull;
    float aOrX, aOrY, aResXY;
    // float aOrZ, aResZ;
    Pt2di aSizeOrtho;
    
// Calculer facteur d'échelle, lancer ScaleIm
    bool aOrthoScaled = false;
    double aScFactor = 1;
    if ((aSizeOrtho.x > aImSize) | (aSizeOrtho.y > aImSize))
    {
		aOrthoScaled = true ;
		int aSzMax;
		if (aSizeOrtho.y > aSizeOrtho.x)
		{
			aSzMax=aSizeOrtho.y;
		}
		else{aSzMax=aSizeOrtho.x;}
		aScFactor = aSzMax/aImSize;
		
		stringstream ss;
		ss << aScFactor;
		
		std::string aScIm = "mm3d ScaleIm " + aNameOrthoFull + " " + ss.str();
	}
	Pt2di aSizeOrhoReduc;
	aSizeOrhoReduc.x = aSizeOrtho.x * aScFactor;		// barbare, voudrait mieux récupérer taille de l'image scaled (ou non)
	aSizeOrhoReduc.y = aSizeOrtho.y * aScFactor;
    
// Charger les GCP, calculer leur projection dans l'ortho réduite
    vector <pair <std::string,Pt3dr> > aGCPground; 
    vector <pair <std::string,Pt2di> > aGCPortho;
    pair <std::string,Pt2di> toPushGCP;
    int aXproj, aYproj;
    for 
    (
		unsigned int i=0;
		i<aGCPground.size();
		i++
	)
	{
		aXproj = (aGCPground[i].second.x - aOrX) / aResXY;
		aYproj = (aGCPground[i].second.y - aOrY) / aResXY;
		if 
		(
			(aXproj>=0) &
			(aYproj>=0) &
			(aXproj<aSizeOrhoReduc.x) &
			(aYproj<aSizeOrhoReduc.y)
		)
		{
			toPushGCP.first = aGCPground[i].first;
			toPushGCP.second.x = aXproj;
			toPushGCP.second.y = aYproj;
			aGCPortho.push_back(toPushGCP);
		}
	}
    
// Interpoler le MNT à l'emplacement de chaque GCP (distinguer ou non control/appuis)
// Calculer différence mesure - MNT (enregistrer dans fichier txt), calculer pour la projection des GCP dans l'ortho réduite
    cElNuage3DMaille *  mMNT;
    mMNT = cElNuage3DMaille::FromFileIm(aMNT);		// mDir + mMNT ?
    
    float bOrX, bOrY,  bResXY ;		// récupérer paramètres du MNT (or + res)
    //float bOrZ,bResZ;
    Pt2di  aCGPinMNT;
    pair <std::string,float> aMNTinterpoled;
    vector <pair <std::string,float> > aListOfDiff;
    
    for 
    (
		unsigned int i=0;
		i<aGCPground.size();
		i++
	)
	{
		aCGPinMNT.x = round_ni((aGCPground[i].second.x - bOrX) / bResXY);
		aCGPinMNT.y = round_ni((aGCPground[i].second.y - bOrY) / bResXY);
		
        if (mMNT->IndexHasContenu(aCGPinMNT))
        {
			Pt3dr aPTer = mMNT->PtOfIndex(aCGPinMNT);
			aMNTinterpoled.first = aGCPground[i].first;
			aMNTinterpoled.second = aGCPground[i].second.z - aPTer.z;
			aListOfDiff.push_back(aMNTinterpoled);
		} 
	}
	
    ofstream fout("GCP-DSM_diff.txt",ios::out);
    fout << "Differences between the altitude in the xml file, and the one computed in the DSM\n";
    for
    (
		unsigned int i=0;
		i<aListOfDiff.size();
		i++
	)
	{
		fout << aListOfDiff[i].first << " " << aListOfDiff[i].second << endl;
	}
    fout.close();
    
// Charger l'ortho réduite, projeter les GCP, tracer des cercles autour, enregistrer l'image avec NamePt[diff]
	std::string aNameOrth = aNameOrthoFull;
	if (aOrthoScaled)
	{
		aNameOrth = aNameOrthoFull.substr(0,aNameOrthoFull.size()-4) + "_Scaled.tif";
	}
	
	Tiff_Im  aImOrtho = Tiff_Im::StdConv(aNameOrth);
	
	Im2D_U_INT1 I(aSizeOrhoReduc.x,aSizeOrhoReduc.y);
	ELISE_COPY
    (
		I.all_pts(),
		aImOrtho.in(),
		I.out()
	);
 /*   
    for 
    (
		unsigned int i=0;
		i<aGCPortho.size();
		i++
	)
	{     
		ELISE_COPY
		(
			disc(aGCPortho[i].second,10),
			P8COL::red,
			I.out(Pdisc)
		);
	}
  */  
    // Génèrer des imagettes individuelles (256x256pixels), les charger
    
    
    // Tracer autour de chaque GCP une croix(16x16) au centre, en haut : nom du point [diff]. Enregistrer avec NamePt[i].tif
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
