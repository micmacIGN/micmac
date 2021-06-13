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
class VT_Img;
class cCMP_Appli;
class VT_AppSelTieP;
class VT_Ortho;
class VT_Imagette;

class cCMP_Ima
{
	public :
		cCMP_Ima(cCMP_Appli & anAppli,const std::string & aName, const std::string & aOri);
		Pt3dr CentreOptique();
	//private :
		cCMP_Appli &   mAppli;
		CamStenope *	mCam;
		std::string mNameOri;
		std::string	 mName;
};

class VT_Img
{
	public :
		VT_Img(VT_AppSelTieP & anAppli,const std::string & aName, const std::string & aOri);
		Pt3dr CentreOptique();
	//private :
		VT_AppSelTieP &   mAppli;
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

typedef  std::pair<VT_Img *,VT_Img *> tPairIm;
typedef  std::map<tPairIm,ElPackHomologue> tMapH;

class VT_AppSelTieP : public cAppliWithSetImage
{
    public :

        VT_AppSelTieP(int argc, char** argv);
        cInterfChantierNameManipulateur * ICNM() const {return mICNM;}
        const std::string & Dir() const {return mDir;}
		std::string NameIm2NameOri(const std::string &, const std::string &) const;
    private :

        std::string 					  mFullName, mOri, mDir, mPat, mNameOri;
		double 							  distMax, mLmin, mLmax, mLmoy;
		bool 							  ShowStats;
		cInterfChantierNameManipulateur * mICNM;
		std::list<std::string> 			  mLFile;
		//CamStenope					    * aCam;
		tPairIm 						  aPair, aPair1, aPair2;
		std::map<std::pair<VT_Img *,VT_Img *> ,ElPackHomologue> mMapH;
		std::map<std::pair<VT_Img *,VT_Img *> ,ElPackHomologue> mMapH_no;
		std::map<std::pair<VT_Img *,VT_Img *> ,ElPackHomologue> mMapH_yes;
};

class VT_Ortho
{
	public :
		VT_Ortho(int argc, char** argv);
		vector <Box2di> MaKeBox (Pt2di mOrthoSz,Pt2di szBox);
	private :
		Pt2dr							  mOriginXY, mResoXY;
		Pt2di							  mOrthoSz, szBox;
		double			 				  mOriginZ, mResoZ;
		vector <Box2di>  				  mLBox;
		std::string 	 				  mOrtho, mDirOrtho, mDir, mOri, mPat, mPat2, mFullName;
		cInterfChantierNameManipulateur * mICNM;
		std::list<std::string> 			  mLFile;
};

class VT_Imagette
{
	public :
		VT_Imagette(VT_Ortho & anAppli, Box2di & mDef, Tiff_Im & mOrthoImg, const std::string & aNameOut, Im2D_U_INT1  & aImR);
	//private :
		double		 mOriginX, mOriginY, mOriginZ, mResoX, mResoY, mResoZ;
		Box2di 		 mDef;
		Pt2di 		 mPtMaxOr;
		Pt3dr 		 mPtMaxTer;
		VT_Ortho &   mAppli;
		std::string  mName;
	
};

/********************************************************************/
/*                                                                  */
/*                      VT_Ortho	                                */
/*                                                                  */
/********************************************************************/
VT_Ortho::VT_Ortho(int argc, char** argv):
	szBox(50,50)
{
	ElInitArgMain
	(
		argc,argv,
		LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
					<< EAMC(mOri,"Orientation")
					<< EAMC(mOrtho,"Orthophotography"),
		LArgMain()  << EAM(szBox,"SzBox",true,"Grid size (Default = [50,50])")
	);

	// Initialize name manipulator & files
	SplitDirAndFile(mDir,mPat,mFullName);

	// Get the list of files from the directory and pattern
	mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
	mLFile = mICNM->StdGetListOfFile(mPat);
	// If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
	StdCorrecNameOrient(mOri,mDir);

	// Get the ortho parameters
	SplitDirAndFile(mDirOrtho,mPat2,mOrtho);
    std::string mMTDOrthoFile = mDirOrtho + "MTDOrtho.xml";
    
	// Charger les mesures images
	cFileOriMnt mMTDOrtho=  StdGetFromPCP(mMTDOrthoFile,FileOriMnt);
	mOriginXY = mMTDOrtho.OriginePlani();
	mResoXY = mMTDOrtho.ResolutionPlani();
	mOriginZ = mMTDOrtho.OrigineAlti();
	mResoZ = mMTDOrtho.ResolutionAlti();
	mOrthoSz = mMTDOrtho.NombrePixels();
	
	//Création des imagettes
	mLBox = MaKeBox (mOrthoSz, szBox);
	std::string aKey = mDirOrtho + "Tmp/";
	ELISE_fp::MkDirRec(aKey);
    
    Tiff_Im mOrthoImg = Tiff_Im::StdConv(mOrtho);// (Tiff_Im::StdConvGen(mOrtho,1,true));
    Im2D_U_INT1  I(mOrthoSz.x,mOrthoSz.y);
    
    ELISE_COPY
    (
       mOrthoImg.all_pts(),
       mOrthoImg.in(),
       I.out()
    );
 
 
 
	Disc_Pal  Pdisc = Disc_Pal::P8COL();
	Gray_Pal  Pgr (30);
	Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
	RGB_Pal   Prgb  (5,5,5);
	Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));

	// Drawing with Elise
	Video_Display Ecr((char *) NULL);
	Ecr.load(SOP);
	Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(500,500));
	
    std::string aNameOut = mDir + mDirOrtho + "Tmp/Vignette_TEST.tif";   
    Tiff_Im  aTOut
    (
         aNameOut.c_str(),
         mOrthoSz,
         GenIm::u_int1,
         Tiff_Im::No_Compr,
         Tiff_Im::RGB
    );
    
    ELISE_COPY
    (
        I.all_pts(),
        I.in()[(FX-4000,FY)],
        aTOut.out()
    );
    W.clik_in();
    
    
    
    
    /*
	std::ostringstream myS;
	for (unsigned int i=0 ; i<mLBox.size() ; i++)
	{
		std::ostringstream myS;
		myS << i;
		std::string aNameOut = mDir + mDirOrtho + "Tmp/Vignette_" + myS.str() + ".tif";
		
		VT_Imagette * mImg1 = new  VT_Imagette(*this, mLBox[i], mOrthoImg, aNameOut, I);
		cout << mImg1->mName << " créée !" << endl;
	}*/
}

vector <Box2di> VT_Ortho::MaKeBox(Pt2di mOrthoSz, Pt2di szBox)
{
	vector <Box2di> boxList;
	int largeur, longueur;
	largeur = 1 + mOrthoSz.x/szBox.x;
	longueur = 1 + mOrthoSz.y/szBox.y;
	for (int i=0 ; i<largeur ; i++)
	{
		for (int j=0 ; j<longueur ; j++)
		{
			Pt2di basGauche, hautDroit;
			basGauche.x = i*szBox.x;
			basGauche.y = j*szBox.y;
			if ((basGauche.x + szBox.x)>mOrthoSz.x){hautDroit.x = mOrthoSz.x;}
			else{hautDroit.x = basGauche.x + szBox.x;}
			if((basGauche.y + szBox.y)>mOrthoSz.y){hautDroit.y = mOrthoSz.y;}
			else{hautDroit.y = basGauche.y + szBox.y;}
			
			boxList.push_back(Box2di (basGauche,hautDroit));
		}
	}
	
	return boxList;
}

/********************************************************************/
/*                                                                  */
/*                      VT_Imagettes                                */
/*                                                                  */
/********************************************************************/

VT_Imagette::VT_Imagette(VT_Ortho & anAppli,Box2di & mDef, Tiff_Im & mOrthoImg, const std::string & aNameOut, Im2D_U_INT1  & I) :
   mAppli  (anAppli),
   mName   (aNameOut)
{	
	Pt2di mySz;
	mySz.x = mDef._p1.x - mDef._p0.x;
	mySz.y = mDef._p1.y - mDef._p0.y;
	
	Disc_Pal  Pdisc = Disc_Pal::P8COL();
	Gray_Pal  Pgr (30);
	Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);
	RGB_Pal   Prgb  (5,5,5);
	Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));

	// Drawing with Elise
	Video_Display Ecr((char *) NULL);
	Ecr.load(SOP);
	Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(mySz.x,mySz.y));



	Tiff_Im  aTOut
    (
         aNameOut.c_str(),
         (mySz),
         GenIm::u_int1,
         Tiff_Im::No_Compr,
         Tiff_Im::RGB
    );
    
    ELISE_COPY
    (
        rectangle(mDef._p1,mDef._p0),
		I.in(),
        W.ogray()
    );
    W.clik_in();

}

/********************************************************************/
/*                                                                  */
/*                      VT_AppSelTieP                               */
/*                                                                  */
/********************************************************************/
//NB + MD 2014/08/29 renamed with prefix 'Local' as function exists in GDAL
bool LocalFileExists( const char * FileName )
{
	#if (ELISE_unix)
		FILE* fp = NULL;
		fp = fopen( FileName, "rb" );
		if( fp != NULL )
		{
			fclose( fp );
			return true;
		}
	#endif
    return false;
}

float AngleBetween3Pts(Pt3dr pt1, Pt3dr pt2, Pt3dr ptCent)
{
	Pt3dr v1 = ptCent-pt1;
	Pt3dr v2 = ptCent-pt2;
	float angle=0;
	float pi=3.14159265359;
	float n1 = sqrt( v1.x*v1.x + v1.y*v1.y + v1.z*v1.z );
	float n2 = sqrt( v2.x*v2.x + v2.y*v2.y + v2.z*v2.z );
	float s1 = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
	if (n1*n2!=0){
		angle = acos (s1/(n1*n2));
		angle = angle * 180 / pi ;}
	return angle;
}

double Dist2Line (Pt3dr pt1, Pt3dr pt2, Pt3dr pt3, Pt3dr pt4)
{
	float a = pt3.x - pt2.x;
	float b = pt3.y - pt2.y;
	float c = pt3.z - pt2.z;
    float d = pt2.x - pt1.x;
    float e = pt2.y - pt1.y;
    float f = pt2.z - pt1.z;
    float g = pt4.x - pt3.x;
    float h = pt4.y - pt3.y;
    float i = pt4.z - pt3.z;
    float j = e * i - f * h;
    float k = f * g - d * i;
    float l = d * h - e * g;
    float m = sqrt(j * j + k * k + l * l);
    float n = j / m;
    float o = k / m;
    float p = l / m;
    float q = a * n + b * o + c * p;
    float dt = fabs(q);
    
    return dt;	
}

VT_AppSelTieP::VT_AppSelTieP(int argc, char** argv):    // cAppliWithSetImage is used to initialize the images
    cAppliWithSetImage (argc-1,argv+1,0),		// it has to be used without the first argument (name of the command)
	distMax(0.005),
	ShowStats(false)
{
	std::string homName("");
	int NbVI(3);
	ElInitArgMain
	(
		argc,argv,
		LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
					<< EAMC(mOri,"Orientation"),
		LArgMain()  << EAM(distMax,"Dist",true,"Maximum distance between two bundles")
	);

	// Initialize name manipulator & files
	SplitDirAndFile(mDir,mPat,mFullName);

	// Get the list of files from the directory and pattern
	mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
	mLFile = mICNM->StdGetListOfFile(mPat);
	// If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
	StdCorrecNameOrient(mOri,mDir);

	// Get the name of tie points name using the key
    std::string mKey = "NKS-Assoc-CplIm2Hom@@dat";
    std::string aNameH ;
    int i=0;
    int j=0;
	for (
			  std::list<std::string>::iterator itS1=mLFile.begin();
			  itS1!=mLFile.end();
			  itS1++
		)
	 {			// Parcours liste d'image
		VT_Img * mImg1 = new  VT_Img(*this,*itS1,mOri);
		
		for (
			  std::list<std::string>::iterator itS2=mLFile.begin();
			  itS2!=mLFile.end();
			  itS2++
		)
		{
			
			std::string mPatHom = mDir + "Homol/Pastis" + *itS1 + "/" + *itS2 + ".dat";
			if (LocalFileExists(mPatHom.c_str()))
			{
				
				VT_Img * mImg2 = new  VT_Img(*this,*itS2,mOri);
				aNameH =  mDir
					   +  mICNM->Assoc1To2
								(
									mKey,
									*itS1,
									*itS2,
									true
								);
				
				ElPackHomologue aPack = ElPackHomologue::FromFile(aNameH);
				mLmin = 999;
				mLmax = 0;
				mLmoy = 0;
				i=0 ;
				for (
					   ElPackHomologue::iterator iTH = aPack.begin();
					   iTH != aPack.end();
					   iTH++
					 )
				{	
					i++;
 				    Pt2dr aP1 = iTH->P1();
				    Pt2dr aP2 = iTH->P2();				
					aPair.first = mImg1;
					aPair.second = mImg2;
					Pt3dr pt1 = mImg1->mCam->F2AndZtoR3(aP1,1);
					Pt3dr pt2 = mImg1->mCam->F2AndZtoR3(aP1,101);
					Pt3dr pt3 = mImg2->mCam->F2AndZtoR3(aP2,1);
					Pt3dr pt4 = mImg2->mCam->F2AndZtoR3(aP2,101);
					
					double dist=Dist2Line(pt1,pt2,pt3,pt4);
					if (dist<distMax)
					{
						mMapH[aPair].Cple_Add(ElCplePtsHomologues(aP1,aP2)); 
					    j++;
					}
					
				    else
				    {
					    mMapH_no[aPair].Cple_Add(ElCplePtsHomologues(aP1,aP2));
					}
				}
				cout << "Couple : " << *itS1 << " & " << *itS2 << "   =>   From " << i << " bundles " << j << " intersect under " << distMax << endl; 
				i = 0;
				j = 0;
			}
		}
	}
	
	std::ostringstream myS;
	myS << distMax;
	
	std::ostringstream myS2;
	myS2 << NbVI;
	std::string aKey = "NKS-Assoc-CplIm2Hom@_Dist" + myS.str() + "@dat";
    for (tMapH::iterator itM=mMapH.begin(); itM!=mMapH.end() ; itM++)       // browse the dictionnary
    {	// write in file
        VT_Img * aIm1 = itM->first.first;       // img1
        VT_Img * aIm2 = itM->first.second;      // img2
        std::string aNameH = mICNM->Assoc1To2(aKey,aIm1->mName,aIm2->mName,true);      
        itM->second.StdPutInFile(aNameH);      // save pt_im1 & pt_im2 in that file
        std::cout << aNameH << "\n";
    }
	
	cout << "END" << endl;
	/*
				    Pt3dr aTer = mImg1->mCam->PseudoInter(aP1,*(mImg2->mCam),aP2);
				    float mAngle = AngleBetween3Pts(mCentre1,mCentre2,aTer);
				    if (mAngle > minAngle )
				    {
					    mMapH[aPair].Cple_Add(ElCplePtsHomologues(aP1,aP2)); 
					    cpt1++;
				    }
				    else if (mAngle <= minAngle )
				    {
					    mMapH_no[aPair].Cple_Add(ElCplePtsHomologues(aP1,aP2)); 
					    cpt2++;
					}
					if (ShowStats)
					{
						if (mAngle < mLmin){mLmin = mAngle;}
						if (mAngle > mLmax){mLmax = mAngle;}
						mLmoy += mAngle;
					}
				}
				if (ShowStats)
				{
					mLmoy = mLmoy / (cpt1 + cpt2);
					cout << "\t" << cpt1 << " points > " << minAngle << "° on " << cpt1+cpt2 << " measures"
						 << "\tMin = " << mLmin << " Max = " << mLmax << " Moy = "  << mLmoy  << endl;
				}
			}
			//else {cout << "   !! DOESN't ExiSTS !!\n";}
		}
	}
	
	int cptr=0;
	bool isIn=false;
	if(NbVI>2)
	{
		for (tMapH::iterator itM1=mMapH_no.begin(); itM1!=mMapH_no.end() ; itM1++)
		{		
			VT_Img * aIm1 = itM1->first.first;
			VT_Img * aIm2 = itM1->first.second;
			for (tMapH::iterator itM2=mMapH_no.begin()	; itM2!=mMapH_no.end() ; itM2++)       // browse the dictionnary
			{	
				VT_Img * aIm3 = itM2->first.first;
				VT_Img * aIm4 = itM2->first.second;
				if ((aIm1->mName == aIm3->mName) && (aIm2->mName != aIm4->mName))
				{
					
					cout << cptr << " : " << aIm1->mName << " & " << aIm2->mName << " & " << aIm3->mName << " & " << aIm4->mName << endl;
					cptr++;
					for (
						ElPackHomologue::iterator iTH1 = itM1->second.begin();
						iTH1 != itM1->second.end();
						iTH1++
					)
					{
						Pt2dr aP1 = iTH1->P1();
						for (
							ElPackHomologue::iterator iTH2 = itM2->second.begin();
							iTH2 != itM2->second.end();
							iTH2++
						)
						{							
							Pt2dr aP3 = iTH2->P1();														
							if ((aP1 == aP3))
							{									
								Pt2dr aP2 = iTH1->P2();									
								Pt2dr aP4 = iTH2->P2();
								aPair1.first = aIm1;
								aPair1.second = aIm2; 
								aPair2.first = aIm3;
								aPair2.second = aIm4; 
								isIn=false;
								for (
									ElPackHomologue::iterator iTH3 = mMapH_yes[aPair1].begin();
									iTH3 != mMapH_yes[aPair1].end();
									iTH3++
								)
								{	// Check if point already in
									Pt2dr aPctrl1 = iTH3->P1();
									if (aP1 == aPctrl1)
									{
										isIn=true;
										iTH3 = mMapH_yes[aPair1].end();
										iTH3--;
									}
								}
								if (!isIn)
								{							
									for (
										ElPackHomologue::iterator iTH4 = mMapH_yes[aPair2].begin();
										iTH4 != mMapH_yes[aPair2].end();
										iTH4++
									)
									{
										Pt2dr aPctrl1 = iTH4->P1();
										if (aP1 == aPctrl1)
										{
											isIn=true;
											iTH4 = mMapH_yes[aPair2].end();
											iTH4--;
										}
									}
								}
								if ((!isIn) && (aP1 == aP3))
								{
									if (NbVI == 3)
									{	// Add 2 couples (3 images)										
										mMapH_yes[aPair1].Cple_Add(ElCplePtsHomologues(aP1,aP2));
										mMapH_yes[aPair2].Cple_Add(ElCplePtsHomologues(aP3,aP4)); 
										iTH2 =  itM2->second.end();
										iTH2--;
									}
									
									else if (NbVI == 4)
									{
										for (tMapH::iterator itM3=mMapH_no.begin()	; itM3!=mMapH_no.end() ; itM3++)       // browse the dictionnary
										{	
											VT_Img * aIm5 = itM3->first.first;
											VT_Img * aIm6 = itM3->first.second;
											if ((aIm1->mName == aIm5->mName) && (aIm2->mName != aIm6->mName) && (aIm4->mName != aIm6->mName))
											{
												for (
													ElPackHomologue::iterator iTH5 = itM3->second.begin();
													iTH5 != itM3->second.end();
													iTH5++
												)
												{
													Pt2dr aP5 = iTH5->P1();
													if (aP1 == aP5)
													{																												
														Pt2dr aP6 = iTH5->P2();
														tPairIm aPair3;
														aPair3.first = aIm5;
														aPair3.second = aIm6;
														isIn=false;
														for (
															ElPackHomologue::iterator iTH6 = mMapH_yes[aPair3].begin();
															iTH6 != mMapH_yes[aPair3].end();
															iTH6++
														)
														{	// Check if point already in
															Pt2dr aPctrl1 = iTH6->P1();
															if (aP1 == aPctrl1)
															{
																isIn=true;
																iTH6 = mMapH_yes[aPair3].end();
																iTH6--;
															}
														}
														if (!isIn)
														{	// Add 3 couples (4 images)
															mMapH_yes[aPair1].Cple_Add(ElCplePtsHomologues(aP1,aP2));
															mMapH_yes[aPair2].Cple_Add(ElCplePtsHomologues(aP3,aP4)); 
															mMapH_yes[aPair3].Cple_Add(ElCplePtsHomologues(aP5,aP6)); 
															iTH5 =  itM3->second.end();
															iTH5--;
														}
													}
												}			
											}
										}
									}
									else { cout << "***** ACHTUNG ***** NbVI == " << NbVI << " NbVI >= 5 not taken into account so far... " << endl;}
								}
							}
						}
					}
				}
			}
		}
	}

    for (tMapH::iterator itM=mMapH_yes.begin(); itM!=mMapH_yes.end() ; itM++)       
    {		// merge
        VT_Img * aIm1 = itM->first.first;       // img1
        VT_Img * aIm2 = itM->first.second;      // img2
        aPair.first = aIm1;
		aPair.second = aIm2; 
		for (
			ElPackHomologue::iterator iTH5 = mMapH_yes[aPair].begin();
			iTH5 != mMapH_yes[aPair].end();
			iTH5++
		)
		{
			Pt2dr aP1 = iTH5->P1();
			Pt2dr aP2 = iTH5->P2();
			mMapH[aPair].Cple_Add(ElCplePtsHomologues(aP1,aP2));
		}
    }
	std::ostringstream myS;
	myS << minAngle;
	
	std::ostringstream myS2;
	myS2 << NbVI;
	std::string aKey = "NKS-Assoc-CplIm2Hom@Angle" + myS.str() + "NbVI" + myS2.str() +"@dat";
    for (tMapH::iterator itM=mMapH.begin(); itM!=mMapH.end() ; itM++)       // browse the dictionnary
    {	// write in file
        VT_Img * aIm1 = itM->first.first;       // img1
        VT_Img * aIm2 = itM->first.second;      // img2
        std::string aNameH = mICNM->Assoc1To2(aKey,aIm1->mName,aIm2->mName,true);      
        itM->second.StdPutInFile(aNameH);      // save pt_im1 & pt_im2 in that file
        std::cout << aNameH << "\n";
    }
    aKey = "NKS-Assoc-CplIm2Hom@_" + myS.str() + "txt@txt";
    for (tMapH::iterator itM=mMapH.begin(); itM!=mMapH.end() ; itM++)       // browse the dictionnary
    {
        VT_Img * aIm1 = itM->first.first;       // img1
        VT_Img * aIm2 = itM->first.second;      // img2
        std::string aNameH = mICNM->Assoc1To2(aKey,aIm1->mName,aIm2->mName,true);      
        itM->second.StdPutInFile(aNameH);      // save pt_im1 & pt_im2 in that file
        std::cout << aNameH << "\n";
    }	*/

}

std::string VT_AppSelTieP::NameIm2NameOri(const std::string & aNameIm, const std::string & aOri) const
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
/*		 VT_Img												 */
/*																  */
/********************************************************************/

VT_Img::VT_Img(VT_AppSelTieP & anAppli,const std::string & aName, const std::string & aOri) :
   mAppli  (anAppli),
   mName   (aName)
{
	mNameOri  = mAppli.NameIm2NameOri(mName,aOri);
	mCam	  = CamOrientGenFromFile(mNameOri,mAppli.ICNM());

}

Pt3dr VT_Img::CentreOptique()
{
	Pt3dr mCentreCam = mCam->VraiOpticalCenter();
	return mCentreCam;
}

//Pt3dr VT_Img::2HomTo3D(



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

Pt3dr cCMP_Ima::CentreOptique()
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
		   Pt3dr mCentre1 = mIms1[i]->CentreOptique();
		   Pt3dr mCentre2 = mIms2[i]->CentreOptique();
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
	std::string inS2(""), inS3(""), inS4(""), inS5("");
	double rX,rY,rZ;
	Pt3dr myPoint;
	for (unsigned int i =0 ; i < inS.size() ; i++)
	{
		if (inS[i] == ',')
		{
			inS2 = inS.substr(0,i);
			rX = atof(inS2.c_str());
			myPoint.x=rX;
			inS3=inS.substr(i+1,inS.size()-i-1);
			i=(unsigned int)inS.size();
		}
	}
	for (unsigned int i =0 ; i < inS3.size() ; i++)
	{
		if (inS3[i] == ',')
		{
			inS4 = inS3.substr(0,i);
			rY = atof(inS4.c_str());
			myPoint.y=rY;
			inS5=inS3.substr(i+1,inS3.size()-i-1);
			rZ = atof(inS5.c_str());
			myPoint.z=rZ;
		}
	}
	
	return myPoint;
}





/********************************************************************/
/*												 				    */
/*					 Quelques applis							    */
/*																    */
/********************************************************************/


void DspVecDbl(vector <double> const& myVec){
    for(unsigned int i=0;i<myVec.size();i++){
        std::cout << std::fixed <<  myVec[i] << endl;
    }
}

void DspVecInt(vector <int> const& myVec){
    for(unsigned int i=0;i<myVec.size();i++){
        std::cout << std::fixed <<  myVec[i] << endl;
    }
}

void DspVecStr(vector <string> const& myVec){
    for(unsigned int i=0;i<myVec.size();i++){
        std::cout << std::fixed <<  myVec[i] << endl;
    }
}








int NLD_main(int argc, char** argv)
{
	std::string mNameIn, mNameOut;
	ElInitArgMain
	(
		argc,argv,
		LArgMain()  << EAMC(mNameOut,"Name of the residuals file"),
		LArgMain()  << EAM(mNameIn,"Out",true,"unsused")
	);
	
	std::string aStr = "GCPBascule  \"60.*ARW\" Rel-19F15P7 BascNLD GCP.xml MesFinal-S2D.xml PatNLD=\"(2a|2b|5a|8b|11a|14b)\"";
	std::vector <std::string> aList;	
	string str = "[1";
	string add1,add2,add3,add4,add5;
	for (int i=0 ; i<2 ; i++){
		if (i==0){add1 = "";}
		else{add1=",X";}
		for (int j=0 ; j<2 ; j++){
			if (j==0){add2 ="";}
			else{add2=",Y";}
			for (int k=0;k<2;k++){
				if (k==0){add3 = "";}
				else{add3 = ",X2";}
				for (int l=0 ; l<2 ; l++){
					if (l==0){add4="";}
					else{add4 = ",Y2";}
					for (int m=0; m<2 ; m++){
						if (m==0){add5 = "";}
						else{add5 = ",XY";}
						string str = "[1" + add1 + add2 + add3 + add4 + add5 + "]";
						aList.push_back(str);}}}}}
						
	std::string mNameOut4="NLD.sh";
	ofstream fout (mNameOut4.c_str(),ios::out);
	for (unsigned int i=0 ; i<aList.size() ; i++)
	{
		fout << aStr << " NLDegX=[1,X,Y] NLDegY=[1,X,Y] NLDegZ=" << aList[i] <<  " NLFR=false NLShow=true | tee ./RTT/Res_" << aList[i] << std::endl;
		fout << "mm3d TestLib RTT ./RTT/Res_" << aList[i] << "   Out=./RTT/" << aList[i] << std::endl;
	}	
	fout.close();	

	VoidSystem("sh NLD.sh");
	
	std::string aR,tmp;
	map <double,string> xList;
	map <double,string> yList;
	map <double,string> zList;
	map <double,string> xyzList;
	int l=0, m=0, n=0, o=0, p=0, q=0, r=0, s=0;
	for (unsigned int i=0 ; i<aList.size() ; i++)
	{
		std::string aNm = "./RTT/" + aList[i];
		ifstream fin (aNm.c_str());
		while (fin >> aR)
		{
			if (aR == "X")
			{
				l++;
			}
			if (l != 0){l++;}
			if (l == 4)
			{
				p++;
				std::ostringstream myS;		// magouille pour différencier les clés associées au map
				myS << p;					// (pour prendre en compte des configurations différentes 
				aR+= "000" + myS.str();		// donnant des résultats identiques)
				tmp = aList[i];
				
				xList[(atof(aR.c_str()))]=tmp;
				l=0;
			} 
			if (aR == "Y")
			{
				m++;
			}
			if (m != 0){m++;}
			if (m == 4)
			{
				q++;
				std::ostringstream myS;
				myS << q;
				aR+= "000" + myS.str();
				yList[(atof(aR.c_str()))]=tmp;
				m=0;
			}
			if (aR == "Z")
			{
				n++;
			}
			if (n != 0){n++;}
			if (n == 4)
			{
				r++;
				std::ostringstream myS;
				myS << r;
				aR+= "000" + myS.str();
				zList[(atof(aR.c_str()))]=tmp;
				n=0;
			}
			if (aR == "XYZ")
			{
				o++;
			}
			if (o != 0){o++;}
			if (o == 4)
			{
				s++;
				std::ostringstream myS;
				myS << s;
				aR+= "000" + myS.str();
				xyzList[(atof(aR.c_str()))]=tmp;
				o=0;
			}
		}
	}
	ofstream fout2 (mNameOut.c_str(),ios::out);
	fout2 << "xList.size() = " << xList.size() << endl << "yList.size() = " << yList.size() << endl << "zList.size() = " << zList.size() << endl;
	map<double,string>::iterator itZ=zList.begin();
	for (unsigned int i=1 ; i<(zList.size()) ; i++)
	{
		fout2 << i << "° résidus le plus faible en Z = " << itZ->first << " m pour la configuration : " << itZ->second << endl;
		for (map<double,string>::iterator itXYZ=xyzList.begin(); itXYZ!=xyzList.end(); itXYZ++)
		{
			if (itXYZ->second == itZ->second)
			{
				fout2 << "\tRésidus normalisés = " << itXYZ->first << " m\n";
				itXYZ=xyzList.end();
			}
		}
		
		itZ++;
	}
	
	return EXIT_SUCCESS;
}




bool IsInVector(std::string id, vector <std::string> aVec)
{
	bool result=true;
	for (unsigned int i=0; i<aVec.size(); i++)
	{
		if (id == aVec[i])
		{
			result = false;
		}
	}
	return result;
}


int ResToTxt_main(int argc, char** argv)
{
/* Transforme résidus de GCPBascule (utiliser GCPBasc ... | tee myFile.txt) en un fichier "NamePt dX dY dZ sigmaX sY sZ eMoyPixel eMaxPixel")*/
	string aNameIn, aNameOut(""), aGCP(""), aMesIm("");
	
	ElInitArgMain
	(
		argc,argv,
		LArgMain()  << EAMC(aNameIn,"Name of the residuals file")
					<< EAMC(aMesIm,"Image measurements file"),
		LArgMain()  << EAM(aNameOut,"Out",true,"File to save the results")
	);
	
	ifstream fin (aNameIn.c_str());
	ofstream fout (aNameOut.c_str(),ios::out);
	string tmpNameOut=aNameOut+"_GCP";
	ofstream fout2 (tmpNameOut.c_str(),ios::out);
	string 	 mRead;
	int 	 i=0, j=0;
	double   rImMoy(0),
			 rXmoy(0),
			 rYmoy(0),
			 rXYZ(0),
			 rZmoy(0);
	Pt3dr 	 ptRes, 
			 ptPres;
	vector <Pt3dr> ptResLs;
	vector <Pt3dr> ptResLs_GCP;
	vector <double> rImMoyLs,rImMaxLs;				
	vector <double> rImMoyLs_GCP,rImMaxLs_GCP;				
	
	
	cSetOfMesureAppuisFlottants dDico=  StdGetFromPCP(aMesIm,SetOfMesureAppuisFlottants);
	std::list<cMesureAppuiFlottant1Im> & dLGCP =  dDico.MesureAppuiFlottant1Im();
	vector <std::string> aListOfChk;
	
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
			aListOfChk.push_back(iT2->NamePt());
		}
	}
	
	bool isCheckPoint=false;		 
	while (fin >> mRead){

		if (mRead == "--NamePt"){
			isCheckPoint=false;
			i++;}
		else if (mRead == "For"){i=0;}
		else if (mRead == "Ctrl"){j++;}
		if (i!=0){i++;}
		if (j!=0){j++;}
		
		if (i==3){
			isCheckPoint=IsInVector(mRead,aListOfChk);
			if (isCheckPoint){fout << mRead << " ";}	// Id
			else {fout2 << mRead << " ";}}
		else if (i==6)		// [rX,rY,rZ]
		{
			ptRes = SplitToPt3dr (mRead);
			if (isCheckPoint)
			{						
				ptResLs.push_back(ptRes);	
				fout << ptRes.x << " " << ptRes.y << " " << ptRes.z << " ";
			}
			else {
				ptResLs_GCP.push_back(ptRes);
				fout2 << ptRes.x << " " << ptRes.y << " " << ptRes.z << " ";}
		}
		else if (i==13)
		{				// [pX,pY,pZ]
			ptPres = SplitToPt3dr (mRead);
			if (isCheckPoint)
			{			
				fout << ptPres.x << " " << ptPres.y << " " << ptPres.z << " ";
			}
			else {fout2 <<  ptPres.x << " " << ptPres.y << " " << ptPres.z << " ";}
		}
		else if (i==20)		// rImMoy
		{
			if (isCheckPoint)
			{	
				rImMoyLs.push_back(atof(mRead.c_str()));		
				fout << mRead << " ";
			}
			else {
				rImMoyLs_GCP.push_back(atof(mRead.c_str()));
				fout2 << mRead << " ";}
		}
		else if (i==25)
		{				// rImMax
			if (isCheckPoint)
			{	
				rImMoyLs.push_back(atof(mRead.c_str()));
				fout << mRead << "\n";
			}
			else { 				
				rImMoyLs_GCP.push_back(atof(mRead.c_str()));
				fout2 <<  mRead << "\n";}
		}
		
		if (j==3)
		{
			fout << mRead << " ";
		}
		else if (j==5)
		{
			fout << mRead << " ";
		}
		else if (j==6)
		{				// [pX,pY,pZ]
			ptRes = SplitToPt3dr (mRead.substr(2,mRead.size()-2));
			ptResLs.push_back(ptRes);
			fout << ptRes.x << " " << ptRes.y << " " << ptRes.z << "\n";
			j=0;
		}
	}
	
	//CP
	for (unsigned int i=0;i<ptResLs.size();i++)
	{
		ptRes = ptResLs[i];
		rXmoy += fabs(ptRes.x);
		rYmoy += fabs(ptRes.y);
		rZmoy += fabs(ptRes.z);
		rXYZ += sqrt(ptRes.x*ptRes.x + ptRes.y*ptRes.y + ptRes.z*ptRes.z);
		if (rImMoyLs.size()>0)
		{
			rImMoy += rImMoyLs[i];
		}
	}
	rXmoy  = rXmoy/ptResLs.size();
	rYmoy  = rYmoy/ptResLs.size();
	rZmoy  = rZmoy/ptResLs.size();
	rXYZ   = rXYZ/ptResLs.size();
	rImMoy = rImMoy/ptResLs.size();
	
	fout << "\nCHECK POINTS => MEAN ABSOLUTE ERROR :\n"
		 << "X : " << rXmoy
		 <<" m\nY : " << rYmoy
		 <<" m\nZ : " << rZmoy
		 <<" m\nXYZ : " << rXYZ ;
		 if (rImMoyLs.size()>0)
		 {
			 fout <<" m\nImage : " << rImMoy ;
		 }
	fout << " pixel\n"; 
	fout.close();
	
	cout << "\nCHECK POINTS => MEAN ABSOLUTE ERROR :\n"
	 << "X : " << rXmoy
	 <<" m\nY : " << rYmoy
	 <<" m\nZ : " << rZmoy
	 <<" m\nXYZ : " << rXYZ << " m";
	 if (rImMoyLs.size()>0)
	 {
		 cout <<" m\nImage : " << rImMoy << " pixel" ;
	 }
	 cout << "\n";
		 
	rXmoy=0;
	rYmoy=0;
	rZmoy=0;
	rXYZ=0;
	rImMoy=0;
	//GCP
	for (unsigned int i=0;i<ptResLs.size();i++)
	{
		ptRes = ptResLs_GCP[i];
		rXmoy += fabs(ptRes.x);
		rYmoy += fabs(ptRes.y);
		rZmoy += fabs(ptRes.z);
		rXYZ  += sqrt(ptRes.x*ptRes.x + ptRes.y*ptRes.y + ptRes.z*ptRes.z);
		if (rImMoyLs.size()>0)
		{
			rImMoy += rImMoyLs[i];
		}
	}
	rXmoy  = rXmoy/ptResLs_GCP.size();
	rYmoy  = rYmoy/ptResLs_GCP.size();
	rZmoy  = rZmoy/ptResLs_GCP.size();
	rXYZ   = rXYZ/ptResLs_GCP.size();
	rImMoy = rImMoy/ptResLs_GCP.size();
	
	fout2 << "\nGROUND CONTROL POINTS => MEAN ABSOLUTE ERROR :\n"
		 << "X : " << rXmoy
		 <<" m\nY : " << rYmoy
		 <<" m\nZ : " << rZmoy
		 <<" m\nXYZ : " << rXYZ ;
		 if (rImMoyLs.size()>0)
		 {
			 fout <<" m\nImage : " << rImMoy ;
		 }
	fout << " pixel\n"; 
	fout.close();
		  
	cout << "\nGROUND CONTROL POINTS => MEAN ABSOLUTE ERROR :\n"
		 << "X : " << rXmoy
		 <<" m\nY : " << rYmoy
		 <<" m\nZ : " << rZmoy
		 <<" m\nXYZ : " << rXYZ << " m";
		 if (rImMoyLs_GCP.size()>0)
		 {
			 cout <<" m\nImage : " << rImMoy << " pixel" ;
		 }
		 cout << "\n";
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

float GiveStats(vector <pair <std::string,float> > mes, string want)
{
    float total=0, ecart=0, moyenne=0, variance=0, stdev=0, max=mes[0].second;
	
	for(unsigned int i=0; i<mes.size() ; i++)
	{
		total = total + mes[i].second;
		if (fabs(max) < fabs(mes[i].second)){max=mes[i].second;}
	}
	moyenne = total / mes.size();
	if (want == "moyenne"){return moyenne;}
	if (want == "max"){return max;}
	
	total=0;
    for(unsigned int i=0; i < mes.size(); i++)
    {
		ecart = mes[i].second-moyenne;
        total = total + ecart*ecart;
    } 
    variance = total / (mes.size() -1);
    if (want == "variance"){return variance;}
    
    stdev = sqrt(variance);
    if (want == "stdev"){return stdev;}
    
    else{return -99999;}
}

int Idem_main(int argc, char** argv)
{
	/*
	std::string aGCP, 
				aMNT, 
				aOrthoName(" "),
				aNameFileTxt(" "), 
				aDir, 
				aPat, 
				aMesIm; 
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
					<< EAM(aPtSz,"PtSz",true,"Size of the point (default : 10)")
	);
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
	vector<std::string> bListOfChk;
	bListOfChk.push_back(aListOfApp[0]);
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
			bListOfChk.push_back(*iT);
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
	
	float moyenne1, moyenne2, stdev1, stdev2, max1, max2, AbsSumDiffControl=0, AbsSumDiffAppuis=0;

	moyenne1 = GiveStats(ListOfDiffAppuis,"moyenne");
	stdev1 = GiveStats(ListOfDiffAppuis,"stdev");
	max1 = GiveStats(ListOfDiffAppuis,"max");
	
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
	
	cout << "MEAN ABSOLUTE ERROR ON CONTROL POINTS = " << AbsSumDiffAppuis/ListOfDiffAppuis.size() << endl
		 << "AVERAGE ERROR = " << moyenne1 << endl
		 << "STANDARD DEVIATION = " << stdev1 << endl
		 << "ERROR MAXIMUM = " << max1 << endl ;	
	
	moyenne2 = GiveStats(ListOfDiffControl,"moyenne");
	stdev2 = GiveStats(ListOfDiffControl,"stdev");
	max2 = GiveStats(ListOfDiffControl,"max");
	
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
	
	cout << "MEAN ABSOLUTE ERROR ON CHECK POINTS = " << AbsSumDiffControl/ListOfDiffControl.size() << endl
		 << "AVERAGE ERROR = " << moyenne2 << endl
		 << "STANDARD DEVIATION = " << stdev2 << endl
		 << "ERROR MAXIMUM = " << max2 << endl ;
	
	if (aNameFileTxt != " ")
	{
		ofstream fout (aNameFileTxt.c_str(),ios::out);
		fout << "Difference between altitude in xml file, and altitude in DEM (Id	 dZ)\n";
		if (ListOfDiffAppuis.size() > 0)
		{
			WriteAppuis(ListOfDiffAppuis,fout);
		fout << "AVERAGE ERROR = " << moyenne1 << endl
			 << "STANDARD DEVIATION = " << stdev1 << endl
			 << "ERROR MAXIMUM = " << max1 << endl;
		}
		
		if (ListOfDiffControl.size() > 0)
		{
			WriteControl(ListOfDiffControl,fout);
			fout << "AVERAGE ERROR = " << moyenne2 << endl
				 << "STANDARD DEVIATION = " << stdev2 << endl
				 << "ERROR MAXIMUM = " << max2 << endl;
		}
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
			
			if (isAppuis)
			{
				cout << "Id = " << aGCPortho[i].first << endl;
				ELISE_COPY
				(
					disc(aPtOrt,aPtSz),
					P8COL::red,
					W.out(Pdisc)
				);
			}
			else if (!isAppuis)
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
	*/
	return EXIT_SUCCESS;
}

int SelTieP_main (int  argc, char** argv)
{
	VT_AppSelTieP anAppli(argc,argv);
	
	return EXIT_SUCCESS;
}

int Ortho2TieP_main (int  argc, char** argv)
{
	VT_Ortho anAppli(argc,argv);
	
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
