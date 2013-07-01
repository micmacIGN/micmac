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

    MicMa cis an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "../../include/StdAfx.h"
#include "hassan/reechantillonnage.h"
#include "../src/uti_image/Arsenic.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <math.h>

double binomial(double n, double k)
{
    double num, den ;
    if ( n < k ) 
    {
       return(0) ; 
    }
    else 
    {
	den = 1;
	num = 1 ; 
	for (int i =  1  ; i <= k   ; i = i+1)
	    den =    den * i;
	for (int j = n-k+1; j<=n; j=j+1)	
	    num = num * j;
	return(num/den);
    } 
}

void Vodka_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************\n";
    std::cout <<  " *     V-ignette                 *\n";
    std::cout <<  " *     O-f                       *\n";
    std::cout <<  " *     D-igital                  *\n";
    std::cout <<  " *     K-amera                   *\n";
    std::cout <<  " *     A-nalysis                 *\n";
    std::cout <<  " *********************************\n\n";
}

PtsHom ReadPtsHom(string aDir,std::vector<std::string> * aSetIm,std::vector<std::vector<double> > vectOfExpTimeISO,string Extension)
{
	PtsHom aPtsHomol;
	Pt2di aSz;
	//Looking for maxs of vectOfExpTimeISO
	double maxExpTime=0, maxISO=0;
	for (int i=0;i<int(vectOfExpTimeISO.size());i++){
		if(vectOfExpTimeISO[i][0]>maxExpTime){maxExpTime=vectOfExpTimeISO[i][0];}
		if(vectOfExpTimeISO[i][1]>maxISO){maxISO=vectOfExpTimeISO[i][1];}
	}

    // Permet de manipuler les ensemble de nom de fichier
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

//On parcours toutes les paires d'images différentes (->testé dans le if)
	int cpt=0;
    for (int aK1=0 ; aK1<int(aSetIm->size()) ; aK1++)
    {
		std::cout<<"Getting homologous points from: "<<(*aSetIm)[aK1]<<endl;
		    
		//Reading the image and creating the objects to be manipulated
			Tiff_Im aTF1= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK1],1,false);
			aSz = aTF1.sz();
			Im2D_REAL16  aIm1(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF1.all_pts(),
				   aTF1.in(),
				   aIm1.out()
				);

			REAL16 ** aData1 = aIm1.data();

        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
			Tiff_Im aTF2= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK2],1,false);
			Im2D_REAL16  aIm2(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF2.all_pts(),
				   aTF2.in(),
				   aIm2.out()
				);

			REAL16 ** aData2 = aIm2.data();

			string prefixe="";
            if (aK1!=aK2)
            {
               std::string aNamePack =  aDir +  aICNM->Assoc1To2
                                        (
                                           "NKS-Assoc-CplIm2Hom@"+prefixe + "@" + Extension,
                                           (*aSetIm)[aK1],
                                           (*aSetIm)[aK2],
                                           true
                                        );

				   bool Exist = ELISE_fp::exist_file(aNamePack);
                   if (Exist)
                   {
                      ElPackHomologue aPack = ElPackHomologue::FromFile(aNamePack);

                           for 
                           (
                               ElPackHomologue::const_iterator itP=aPack.begin();
                               itP!=aPack.end();
                               itP++
                           )
                           {
							   cpt++;
							   //Compute the distance between the point and the center of the image
							   double x0=aSz.x/2-0.5;
							   double y0=aSz.y/2-0.5;
							   double Dist1=sqrt(pow(itP->P1().x-x0,2)+pow(itP->P1().y-y0,2));
							   double Dist2=sqrt(pow(itP->P2().x-x0,2)+pow(itP->P2().y-y0,2));
							   //Go looking for grey value of the point, adjusted to ISO and Exposure time induced variations
							   double Grey1 =sqrt((vectOfExpTimeISO[aK1][0]*vectOfExpTimeISO[aK1][1])/(maxExpTime*maxISO))*Reechantillonnage::biline(aData1, aSz.x, aSz.y, itP->P1());
							   double Grey2 =sqrt((vectOfExpTimeISO[aK2][0]*vectOfExpTimeISO[aK2][1])/(maxExpTime*maxISO))*Reechantillonnage::biline(aData2, aSz.x, aSz.y, itP->P2());
							   //Check that the distances are different-> might be used in filter?
							   //double rap=Dist1/Dist2;
							   if(1){//(Dist1>aSz.x/3 || Dist2>aSz.x/3)){// && (rap<0.75 || rap>1.33)){Filtre à mettre en place?
								   aPtsHomol.Dist1.push_back(Dist1);
								   aPtsHomol.Dist2.push_back(Dist2);
								   aPtsHomol.Gr1.push_back(Grey1);
								   aPtsHomol.Gr2.push_back(Grey2);
							   }
                   }
				   }
                   else
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
            }
        }
    }
	int nbpts=aPtsHomol.size();
	std::cout<<"Total number tie points: "<<nbpts<<" out of "<<cpt<<endl;
	aPtsHomol.SZ=aSz;

   return aPtsHomol;
}

void Vignette_correct(string aDir,std::vector<std::string> * aSetIm, DiaphFoc diaphFoc,string aDirOut){

	//Reading vignette files

		//Reading the image and creating the objects to be manipulated

		char foc[5],dia[4];
		sprintf(foc, "%04d", int(diaphFoc.foc));
		sprintf(dia, "%03d", int(10*diaphFoc.diaph));
		string aNameVignette="Foc" + (string)foc + "Diaph" + (string)dia + ".tif";
		Tiff_Im aTFV= Tiff_Im::StdConvGen(aDir + aDirOut + aNameVignette,1,false);
		Pt2di aSz = aTFV.sz();

		Im2D_REAL4  aVignette(aSz.x,aSz.y);

		ELISE_COPY
		(
		   aTFV.all_pts(),
		   aTFV.in(),
		   aVignette.out()
		);

		REAL4 ** aDataVignette = aVignette.data();


	//Reading images file
    int nbIm=(aSetIm)->size();
    for(int i=0;i<nbIm;i++)
	{
	    string aNameIm=(*aSetIm)[i];
		cout<<"Correcting "<<aNameIm<<endl;
		string aNameOut=aDir + aDirOut + aNameIm +"_vodka.tif";

		//Reading the image and creating the objects to be manipulated
		Tiff_Im aTF= Tiff_Im::StdConvGen(aDir + aNameIm,3,false);
		Pt2di aSz = aTF.sz();

		Im2D_U_INT1  aImR(aSz.x,aSz.y);
		Im2D_U_INT1  aImG(aSz.x,aSz.y);
		Im2D_U_INT1  aImB(aSz.x,aSz.y);

		ELISE_COPY
		(
		   aTF.all_pts(),
		   aTF.in(),
		   Virgule(aImR.out(),aImG.out(),aImB.out())
		);

		U_INT1 ** aDataR = aImR.data();
		U_INT1 ** aDataG = aImG.data();
		U_INT1 ** aDataB = aImB.data();

		for (int aY=0 ; aY<aSz.y  ; aY++)
			{
				for (int aX=0 ; aX<aSz.x  ; aX++)
				{
					double aCor=aDataVignette[aY][aX];
					double R = aDataR[aY][aX] * aCor;
					double G = aDataG[aY][aX] * aCor;
					double B = aDataB[aY][aX] * aCor;
					if(R>255){aDataR[aY][aX]=255;}else if(aCor<1){continue;}else{aDataR[aY][aX]=R;}
					if(G>255){aDataG[aY][aX]=255;}else if(aCor<1){continue;}else{aDataG[aY][aX]=G;}
					if(B>255){aDataB[aY][aX]=255;}else if(aCor<1){continue;}else{aDataB[aY][aX]=B;}
				}
		}

		 Tiff_Im  aTOut
			(
				aNameOut.c_str(),
				aSz,
				GenIm::u_int1,
				Tiff_Im::No_Compr,
				Tiff_Im::RGB
			);


		 ELISE_COPY
			 (
				 aTOut.all_pts(),
				 Virgule(aImR.in(),aImG.in(),aImB.in()),
				 aTOut.out()
			 );


	}


}

/*
void Vignette_correct_old(string aDir,std::vector<std::string> * aSetIm,vector<double> aParam,string aDirOut){

	//Reading input files
    int nbIm=(aSetIm)->size();
    for(int i=0;i<nbIm;i++)
	{
	    string aNameIm=(*aSetIm)[i];
		cout<<"Correcting "<<aNameIm<<endl;
		string aNameOut=aDir + aDirOut + aNameIm +"_vodka.tif";

		//Reading the image and creating the objects to be manipulated
		Tiff_Im aTF= Tiff_Im::StdConvGen(aDir + aNameIm,3,false);
		Pt2di aSz = aTF.sz();

		Im2D_U_INT1  aImR(aSz.x,aSz.y);
		Im2D_U_INT1  aImG(aSz.x,aSz.y);
		Im2D_U_INT1  aImB(aSz.x,aSz.y);

		ELISE_COPY
		(
		   aTF.all_pts(),
		   aTF.in(),
		   Virgule(aImR.out(),aImG.out(),aImB.out())
		);

		U_INT1 ** aDataR = aImR.data();
		U_INT1 ** aDataG = aImG.data();
		U_INT1 ** aDataB = aImB.data();

		for (int aY=0 ; aY<aSz.y  ; aY++)
			{
				for (int aX=0 ; aX<aSz.x  ; aX++)
				{
					double x0=aSz.x/2;
					double y0=aSz.y/2;
					double D=pow(aX-x0,2)+pow(aY-y0,2);
					double aCor=1+aParam[0]*D+aParam[1]*pow(D,2)+aParam[2]*pow(D,3);
					double R = aDataR[aY][aX] * aCor;
					double G = aDataG[aY][aX] * aCor;
					double B = aDataB[aY][aX] * aCor;
					if(R>255){aDataR[aY][aX]=255;}else if(aCor<1){continue;}else{aDataR[aY][aX]=R;}
					if(G>255){aDataG[aY][aX]=255;}else if(aCor<1){continue;}else{aDataG[aY][aX]=G;}
					if(B>255){aDataB[aY][aX]=255;}else if(aCor<1){continue;}else{aDataB[aY][aX]=B;}
				}
		}

		 Tiff_Im  aTOut
			(
				aNameOut.c_str(),
				aSz,
				GenIm::u_int1,
				Tiff_Im::No_Compr,
				Tiff_Im::RGB
			);


		 ELISE_COPY
			 (
				 aTOut.all_pts(),
				 Virgule(aImR.in(),aImG.in(),aImB.in()),
				 aTOut.out()
			 );
	  
	}
}
*/

void Write_Vignette(string aDir, string aNameOut,vector<double> aParam,string aDirOut, Pt2di aSz){

	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);

		//Reading the image and creating the objects to be manipulated
		aNameOut=aDirOut + aNameOut;
		Tiff_Im aTF=Tiff_Im(aNameOut.c_str(), aSz, GenIm::real4, Tiff_Im::No_Compr, Tiff_Im::BlackIsZero);

		Im2D_REAL4  aIm(aSz.x,aSz.y);

		ELISE_COPY
		(
		   aTF.all_pts(),
		   aTF.in(),
		   aIm.out()
		);

		REAL4 ** aData = aIm.data();
		
		for (int aY=0 ; aY<aSz.y  ; aY++)
			{
				for (int aX=0 ; aX<aSz.x  ; aX++)
				{
					double x0=aSz.x/2;
					double y0=aSz.y/2;
					double D=pow(aX-x0,2)+pow(aY-y0,2);
					double aCor=1+aParam[0]*D+aParam[1]*pow(D,2)+aParam[2]*pow(D,3);
					if(aCor<1){aData[aY][aX]=1;}else{aData[aY][aX]=aCor;}
				}
		}
		
		 Tiff_Im  aTOut
			(
				aNameOut.c_str(),
				aSz,
				GenIm::real4,
				Tiff_Im::No_Compr,
				Tiff_Im::BlackIsZero
			);


		 ELISE_COPY
			 (
				 aTOut.all_pts(),
				 aIm.in(),
				 aTOut.out()
			 );
	  

}

vector<double> Vignette_Solve(PtsHom aPtsHomol)
{
	double distMax=sqrt(pow(float(aPtsHomol.SZ.x)/2,2)+pow(float(aPtsHomol.SZ.y)/2,2));
//Least Square
/*
	// Create L2SysSurResol to solve least square equation with 3 unknown
	L2SysSurResol aSys(3);
	int nbPtsSIFT=aPtsHomol[0].size();

  	//For Each SIFT point
	for(int i=0;i<int(nbPtsSIFT);i++){
				 double aPds[3]={(aPtsHomol[3][i]*pow(aPtsHomol[1][i],2)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],2)),
								 (aPtsHomol[3][i]*pow(aPtsHomol[1][i],4)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],4)),
								 (aPtsHomol[3][i]*pow(aPtsHomol[1][i],6)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],6))
								};
				 aSys.AddEquation(1,aPds,aPtsHomol[2][i]-aPtsHomol[3][i]);
	}

	//System has 3 unknowns and nbPtsSIFT equations (significantly more than enough)

    bool Ok;
    Im1D_REAL8 aSol = aSys.GSSR_Solve(&Ok);

	vector<double> aParam;
    if (Ok)
    {
        double* aData = aSol.data();
		aParam.push_back(aData[0]);
		aParam.push_back(aData[1]);
		aParam.push_back(aData[2]);
    }
	

	//Erreur moyenne
	
	vector<double> erreur;
	for(int i=0;i<int(aPtsHomol[0].size());i++){
		double aComputedVal=aData[0]*(aPtsHomol[3][i]*pow(aPtsHomol[1][i],2)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],2))
										+aData[1]*(aPtsHomol[3][i]*pow(aPtsHomol[1][i],4)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],4))
										+aData[2]*(aPtsHomol[3][i]*pow(aPtsHomol[1][i],6)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],6));
		double aInputVal=aPtsHomol[2][i]-aPtsHomol[3][i];	
		erreur.push_back(fabs(aComputedVal-aInputVal)*(min(aPtsHomol[0][i],aPtsHomol[1][i]))/(distMax));
		}
	double sum = std::accumulate(erreur.begin(),erreur.end(),0.0);
    double ErMoy=sum/erreur.size();
	cout<<"Mean error = "<<ErMoy<<endl;
	
//End Least Square
*/


//RANSAC
	
vector<double> aParam;
double /*ErMoyMin=10,*/ nbInliersMax=0,ErMin=10,aScoreMax=0;;
int nbPtsSIFT=aPtsHomol.size();
int nbRANSACinitialised=0;
int nbRANSACaccepted=0;
int nbRANSACmax=10000;
srand(time(NULL));//Initiate the rand value
while(nbRANSACinitialised<nbRANSACmax || nbRANSACaccepted<500)
{
	nbRANSACinitialised++;
	if(nbRANSACinitialised % 500==0 && nbRANSACinitialised<=nbRANSACmax){cout<<"RANSAC progress : "<<nbRANSACinitialised/100<<" %"<<endl;}

	L2SysSurResol aSys(3);

	//For 3 SIFT points
	for(int k=0;int(k)<3*((rand() % 8)+3);k++){
		
		int i=rand() % nbPtsSIFT;//Rand choice of a point

		double aPds[3]={(aPtsHomol.Gr2[i]*pow(aPtsHomol.Dist2[i],2)-aPtsHomol.Gr1[i]*pow(aPtsHomol.Dist1[i],2)),
						(aPtsHomol.Gr2[i]*pow(aPtsHomol.Dist2[i],4)-aPtsHomol.Gr1[i]*pow(aPtsHomol.Dist1[i],4)),
						(aPtsHomol.Gr2[i]*pow(aPtsHomol.Dist2[i],6)-aPtsHomol.Gr1[i]*pow(aPtsHomol.Dist1[i],6))
								};
				 double poids=1;//sqrt(max(aPtsHomol[1][i],aPtsHomol[0][i]));//sqrt(fabs(aPtsHomol[1][i]-aPtsHomol[0][i]));
				 aSys.AddEquation(poids,aPds,aPtsHomol.Gr1[i]-aPtsHomol.Gr2[i]);//fabs(aPtsHomol[1][i]-aPtsHomol[0][i])
	}

	//Computing the result
	bool Ok;
    Im1D_REAL8 aSol = aSys.GSSR_Solve(&Ok);
	double* aData = aSol.data();

	//Filter if computed vignette is <0 in the corners and if param 1<0 (not a possible vignette)
	double valCoin=(1+aData[0]*pow(distMax,2)+aData[1]*pow(distMax,4)+aData[2]*pow(distMax,6));

	if (Ok && aData[0]>0 && 1<=valCoin){
		nbRANSACaccepted++;
		if (nbRANSACaccepted % 50==0 && nbRANSACinitialised>nbRANSACmax){cout<<"Difficult config RANSAC progress : "<<nbRANSACaccepted/5<<"%"<<endl;}
  		//For Each SIFT point, test if in acceptable error field->compute score
		double nbInliers=0,aScore;
		vector<double> erreur;

		//Computing the distance from computed surface and data points
		for(int i=0;i<int(nbPtsSIFT);i++){
					 double aComputedVal=aData[0]*(aPtsHomol.Gr2[i]*pow(aPtsHomol.Dist2[i],2)-aPtsHomol.Gr1[i]*pow(aPtsHomol.Dist1[i],2))
										+aData[1]*(aPtsHomol.Gr2[i]*pow(aPtsHomol.Dist2[i],4)-aPtsHomol.Gr1[i]*pow(aPtsHomol.Dist1[i],4))
										+aData[2]*(aPtsHomol.Gr2[i]*pow(aPtsHomol.Dist2[i],6)-aPtsHomol.Gr1[i]*pow(aPtsHomol.Dist1[i],6));
					 double aInputVal=aPtsHomol.Gr1[i]-aPtsHomol.Gr2[i];	
					 erreur.push_back(fabs(aComputedVal-aInputVal)*(min(aPtsHomol.Dist1[i],aPtsHomol.Dist2[i]))/(distMax));
					 //Selecting inliers
					 if(fabs(aComputedVal-aInputVal)<5){
						nbInliers++;
					 }
		}
		double sum = std::accumulate(erreur.begin(),erreur.end(),0.0);
		double ErMoy=sum/erreur.size();
		aScore=(nbInliers/nbPtsSIFT)/ErMoy;
		//if(nbInliers/nbPtsSIFT>0.20 && aScoreMax<aScore){
		//if(nbInliers>nbInliersMax){
		if(aScore>aScoreMax){
			cout<<valCoin<<endl;
			nbInliersMax=nbInliers;
			ErMin=ErMoy;
			aScoreMax=aScore;
			cout<<"New Best Score (at "<<nbRANSACinitialised<<"th iteration) is : "<<aScoreMax<< " with " <<nbInliersMax/nbPtsSIFT*100<<"% of points used and Mean Error="<<ErMoy<<endl;
			aParam.clear();
			aParam.push_back(aData[0]);
			aParam.push_back(aData[1]);
			aParam.push_back(aData[2]);
		}
	}
}

std::cout << "RANSAC score is : "<<aScoreMax<<endl;

//end RANSAC

	if(aParam.size()==3){ std::cout << "Vignette parameters, with x dist from image center : (" << aParam[0] << ")*x^2+(" << aParam[1] << ")*x^4+(" << aParam[2] << ")*x^6"<<endl;}

	return aParam;
}

int  Vignette_main(int argc,char ** argv)
{

	std::string aFullPattern,aDirOut="Vignette/",InVig,InCal="";
	bool InTxt=false,DoCor=false;
	  //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern"),
            LArgMain()  << EAM(aDirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
						//<< EAM(InVig,"InVig",true,"Input vignette parameters")
						<< EAM(InTxt,"InTxt",true,"True if homologous points have been exported in txt (Defaut=false)")
						<< EAM(InCal,"InCal",true,"Name of folder with vignette calibration tif file (if previously computed)")
						<< EAM(DoCor,"DoCor",true,"Use the computed parameters to correct the images (Defaut=false)")
        );
		std::string aDir,aPatIm;
		SplitDirAndFile(aDir,aPatIm,aFullPattern);

		std::string Extension = "dat";
		if (InTxt){Extension="txt";}

		cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
		const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

		vector<DiaphFoc> vectOfDiaphFoc;
		vector<vector<string> > listOfListIm;
		vector<vector<vector<double> > > vectOfvectOfExpTimeISO;

		//Read InCal
		if (InCal!=""){
			string aPatVignette="Foc[0-9]{4}Diaph[0-9]{3}.tif";
			list<string> aSetVignette=RegexListFileMatch(aDir + InCal,aPatVignette,1,false);
			unsigned nbInCal=aSetVignette.size();
			for(unsigned i=0;i<nbInCal;i++){
				DiaphFoc diaphFoc;
				diaphFoc.diaph=(atof((aSetVignette.back().substr (12,3)).c_str())/10);
				diaphFoc.foc=(atof((aSetVignette.back().substr (3,4)).c_str()));
				diaphFoc.isComputed=true;
				vectOfDiaphFoc.push_back(diaphFoc);
				aSetVignette.pop_back();
				vector<string> newSetOfIm;listOfListIm.push_back(newSetOfIm);//init of the image groupe with diaph and Foc equal to thing read in name of tif
				vector<vector<double> > vectOfExpTimeISO; vectOfvectOfExpTimeISO.push_back(vectOfExpTimeISO);//idem with this
			}
		}
		cout<<"Found "<<vectOfDiaphFoc.size()<<" input vignette file(s)"<<endl;

		//Creating a new list of images for each combination of Diaph & Foc, and recording their ExpTime and ISO for future normalisation
		for (int j=0;j<(int)aSetIm->size();j++){
			std::string aFullName=(*aSetIm)[j];
			const cMetaDataPhoto & infoIm = cMetaDataPhoto::CreateExiv2(aFullName);
			DiaphFoc diaphFoc;diaphFoc.diaph=infoIm.Diaph();diaphFoc.foc=infoIm.FocMm();
			vector<double> expTimeISO;expTimeISO.push_back(infoIm.ExpTime());expTimeISO.push_back(infoIm.IsoSpeed());
			cout<<"Getting Diaph and Focal from "<<aFullName<<endl;
			if (vectOfDiaphFoc.size()==0){
				diaphFoc.isComputed=false;
				vectOfDiaphFoc.push_back(diaphFoc);
				vector<string> newSetOfIm;
				newSetOfIm.push_back(aFullName);
				listOfListIm.push_back(newSetOfIm);
				vector<vector<double> > vectOfExpTimeISO;
				vectOfExpTimeISO.push_back(expTimeISO);
				vectOfvectOfExpTimeISO.push_back(vectOfExpTimeISO);
			}else{
				for (int i=0;i<(int)vectOfDiaphFoc.size();i++){
					//cout<<"Banane! "<<vectOfDiaphFoc.size()<<endl;
					if (diaphFoc.diaph==vectOfDiaphFoc[i].diaph && diaphFoc.foc==vectOfDiaphFoc[i].foc){
						listOfListIm[i].push_back(aFullName);
						vectOfvectOfExpTimeISO[i].push_back(expTimeISO);
						//cout<<"Added "<<listOfListIm[i]<<" as "<<listOfListIm[i].size()<<"th image to group "<<i<<endl;
						break;
						}else{if(i==(int)vectOfDiaphFoc.size()-1){
								diaphFoc.isComputed=false;
								vectOfDiaphFoc.push_back(diaphFoc);
								vector<string>newSetOfIm;
								newSetOfIm.push_back(aFullName);
								listOfListIm.push_back(newSetOfIm);
								vector<vector<double> > vectOfExpTimeISO;
								vectOfExpTimeISO.push_back(expTimeISO);
								vectOfvectOfExpTimeISO.push_back(vectOfExpTimeISO);
								break;
							}else{continue;}}
					}
			}
		}
		cout<<"Number of different sets of images with the same Diaph-Focal combination : "<<listOfListIm.size()<<endl<<endl;
		for(int i=0;i<(int)listOfListIm.size();i++){
			if(!vectOfDiaphFoc[i].isComputed){
				std::cout << "--- Computing the parameters of the vignette effect for the set of "<<listOfListIm[i].size()<<" images with Diaph="<<vectOfDiaphFoc[i].diaph<<" and Foc="<<vectOfDiaphFoc[i].foc<<endl<<endl;

				//Avec Points homol
				PtsHom aPtsHomol=ReadPtsHom(aDir, & listOfListIm[i], vectOfvectOfExpTimeISO[i],Extension);
				//aPtsHomol est l'ensemble des vecteurs D1,D2,G1,G2,SZ;
				vector<double> aParam = Vignette_Solve(aPtsHomol);
			
		//Avec Stack
				/*
				string cmdStack="mm3d StackFlatField " + aFullPattern + " 16";
				system_call(cmdStack.c_str());

				Tiff_Im aTF= Tiff_Im::StdConvGen(aDir + "FlatField.tif",1,true);
					Pt2di aSz = aTF.sz();
					Im2D_U_INT1  aIm(aSz.x,aSz.y);
					ELISE_COPY
						(
						   aTF.all_pts(),
						   aTF.in(),
						   aIm.out()
						);

					U_INT1 ** aData = aIm.data();
		
				L2SysSurResol aSys(3);
				int x0=aSz.x/2;
				int y0=aSz.y/2;
				cout<<"Size=["<<aSz.x<<" - "<<aSz.y<<"]"<<endl;
				int cpt=0;
				for (int aY=0 ; aY<aSz.y  ; aY=aY+8)
					{
						for (int aX=0 ; aX<aSz.x  ; aX=aX+8)
						{
							double Dist=sqrt(pow(aX-x0,2)+pow(aY-y0,2));
							double aPds[3]={pow(Dist,2),pow(Dist,4),pow(Dist,6)};
							aSys.AddEquation(1,aPds,aData[aY][aX]);
		cpt++;
						}
					}
		cout<<cpt<<" points"<<endl;
		double* aParam = Vignette_Solve(aSys);
		*/


		   if (aParam.size()==0){
			   cout<<"Could'nt compute vignette parameters"<<endl;
		   }else{ 
			   
			   //Creating a flatfield tif file
				   //Creating the numerical format for the output files names
						char foc[5],dia[4];
						sprintf(foc, "%04d", int(vectOfDiaphFoc[i].foc));
						sprintf(dia, "%03d", int(10*vectOfDiaphFoc[i].diaph));
						
			   string aNameOut="Foc" + (string)foc + "Diaph" + (string)dia + ".tif";

			   Write_Vignette(aDir, aNameOut, aParam, aDirOut, aPtsHomol.SZ);
			   
			   //Set the couple of diaph foc to "computed"
			   vectOfDiaphFoc[i].isComputed=true;
				}
			}
			if (DoCor && vectOfDiaphFoc[i].isComputed==1){
			//Correction des images avec les params calculés
			cout<<"Correcting the images"<<endl;
			Vignette_correct(aDir, & listOfListIm[i],vectOfDiaphFoc[i],aDirOut);
	   
			}
		}
   Vodka_Banniere();
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

