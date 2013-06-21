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
#include <algorithm>
#include <functional>
#include <numeric>

void Arsenic_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " **********************************\n";
    std::cout <<  " *     A-utomated                 *\n";
    std::cout <<  " *     R-adiometric               *\n";
    std::cout <<  " *     S-hift                     *\n";
    std::cout <<  " *     E-qualization              *\n";
    std::cout <<  " *     N-ormalization             *\n";
    std::cout <<  " *     I-nter-images              *\n";
    std::cout <<  " *     C-orrection                *\n";
    std::cout <<  " **********************************\n\n";
}
vector<vector<double> > ReadPtsHom(string aDir,std::vector<std::string> * aSetIm,string Extension)
{

	vector<double> NbPtsCouple,Gr1,Gr2,R1,G1,B1,R2,G2,B2,X1,Y1,X2,Y2;//Elements of output (distance from SIFT pts to center for Im1 and Im2, and respective grey lvl 
	Pt2di aSz;

    // Permet de manipuler les ensemble de nom de fichier
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

//On parcours toutes les paires d'images différentes (->testé dans le if)
    for (int aK1=0 ; aK1<int(aSetIm->size()) ; aK1++)
    {
		std::cout<<"Getting homologous points from: "<<(*aSetIm)[aK1]<<endl;
		    
		//Reading the image and creating the objects to be manipulated
			Tiff_Im aTF1= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK1],3,false);
			aSz = aTF1.sz();
			Im2D_REAL16  aIm1R(aSz.x,aSz.y);
			Im2D_REAL16  aIm1G(aSz.x,aSz.y);
			Im2D_REAL16  aIm1B(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF1.all_pts(),
				   aTF1.in(),
				   Virgule(aIm1R.out(),aIm1G.out(),aIm1B.out())
				);

			REAL16 ** aDataR1 = aIm1R.data();
			REAL16 ** aDataG1 = aIm1G.data();
			REAL16 ** aDataB1 = aIm1B.data();

        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
			if (aK1!=aK2)
            {
			Tiff_Im aTF2= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK2],3,false);
			Im2D_REAL16  aIm2R(aSz.x,aSz.y);
			Im2D_REAL16  aIm2G(aSz.x,aSz.y);
			Im2D_REAL16  aIm2B(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF2.all_pts(),
				   aTF2.in(),
				   Virgule(aIm2R.out(),aIm2G.out(),aIm2B.out())
				);

			REAL16 ** aDataR2 = aIm2R.data();
			REAL16 ** aDataG2 = aIm2G.data();
			REAL16 ** aDataB2 = aIm2B.data();

			string prefixe="";
            
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
						int cpt=0;
							for 
							(
								ElPackHomologue::const_iterator itP=aPack.begin();
								itP!=aPack.end();
								itP++
							)
							{
								cpt++;
								//Go looking for grey value of the point, adjusted to ISO and Exposure time induced variations
								double Red1   =Reechantillonnage::biline(aDataR1, aSz.x, aSz.y, itP->P1());
								double Green1 =Reechantillonnage::biline(aDataG1, aSz.x, aSz.y, itP->P1());
								double Blue1  =Reechantillonnage::biline(aDataB1, aSz.x, aSz.y, itP->P1());
								double Red2   =Reechantillonnage::biline(aDataR2, aSz.x, aSz.y, itP->P2());
								double Green2 =Reechantillonnage::biline(aDataG2, aSz.x, aSz.y, itP->P2());
								double Blue2  =Reechantillonnage::biline(aDataB2, aSz.x, aSz.y, itP->P2());;
							Gr1.push_back((Red1+Green1+Blue1)/3);
							Gr2.push_back((Red2+Green2+Blue2)/3);
							R1.push_back(Red1);
							G1.push_back(Green1);
							B1.push_back(Blue1);
							R2.push_back(Red2);
							G2.push_back(Green2);
							B2.push_back(Blue2);
							X1.push_back(itP->P1().x);
							Y1.push_back(itP->P1().y);
							X2.push_back(itP->P2().x);
							Y2.push_back(itP->P2().y);
							}
						NbPtsCouple.push_back(double(cpt));
				   }
                   else{
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
					  NbPtsCouple.push_back(0);
				   }
            }
        }
    }
	int nbpts=G1.size();
	vector<vector<double> > aPtsHomol;
	vector<double> SZ;
	SZ.push_back(aSz.x);SZ.push_back(aSz.y);
	aPtsHomol.push_back(NbPtsCouple);
	aPtsHomol.push_back(Gr1);
	aPtsHomol.push_back(R1);
	aPtsHomol.push_back(G1);
	aPtsHomol.push_back(B1);
	aPtsHomol.push_back(Gr2);
	aPtsHomol.push_back(R2);
	aPtsHomol.push_back(G2);
	aPtsHomol.push_back(B2);
	aPtsHomol.push_back(SZ);
	aPtsHomol.push_back(X1);
	aPtsHomol.push_back(Y1);
	aPtsHomol.push_back(X2);
	aPtsHomol.push_back(Y2);
   return aPtsHomol;
}

double oneParamRANSAC(vector<double> im1, vector<double> im2){

	srand(time(NULL));//Initiate the rand value
	double aScoreMax=5;
	double bestK=1;
	for(unsigned i=0;i<100;i++){
		int j=rand() % im1.size();
		double K=im1[j]/im2[j];

		//Counting Inliers
		double nbInliers=0;
		double aScore=0;
		for(unsigned m=0;m<im1.size();m++){
			if(fabs((im1[m]/im2[m])/K-1)<0.50){nbInliers++;aScore=aScore+fabs((im1[m]/im2[m])-K);}
		}
		aScore=aScore/nbInliers;
		if(0.3<nbInliers/im1.size() && aScoreMax>aScore){aScoreMax=aScore;bestK=K;}
	}
	//cout<<aScoreMax<< " - ";
return bestK;
}


vector<vector<double> > Egalisation_factors(vector<vector<double> > aPtsHomol, int nbIm, int aMasterNum, int aDegPoly){

	vector<vector<double> > Gr(nbIm);

// Create L2SysSurResol to solve least square equation with nbIm unknown
	L2SysSurResol aSysRInit(nbIm);
	L2SysSurResol aSysGInit(nbIm);
	L2SysSurResol aSysBInit(nbIm);

	
//Finding and Selecting the brightest image for reference
int nbParcouru=0;
for (int i=0;i<int(aPtsHomol[0].size());i++){

	int numImage1=(i/(nbIm));
	int numImage2=i-numImage1*(nbIm);

if (numImage1!=numImage2){
	if(aPtsHomol[0][i]!=0){//if there are homologous points between images
		vector<double> GrIm1(aPtsHomol[1].begin()+nbParcouru,aPtsHomol[1].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> GrIm2(aPtsHomol[5].begin()+nbParcouru,aPtsHomol[5].begin()+nbParcouru+aPtsHomol[0][i]-1);
		
		nbParcouru=nbParcouru+aPtsHomol[0][i];

		std::copy (GrIm1.begin(),GrIm1.end(),back_inserter(Gr[numImage1]));
		std::copy (GrIm2.begin(),GrIm2.end(),back_inserter(Gr[numImage2]));
	}
}
}

if(aMasterNum==-1){
	double Gmax = 0;
	for(int i=0;i<int(nbIm);i++)
	{
		double meanGr = std::accumulate(Gr[i].begin(),Gr[i].end(),0.0)/Gr[i].size();
		if(meanGr>Gmax){Gmax=meanGr;aMasterNum=i;}
	}
	cout<<"The brightest image (chosen as Master) is NUM "<<aMasterNum<<endl;
}

//Solve with equation : ( model is : alpha1*G1-alpha2*G2=0 )

for(int i=0;i<int(nbIm);i++){
	double grMax=*max_element(Gr[i].begin(),Gr[i].end());
	vector<double> aCoefsFixe(nbIm,0.0);
	aCoefsFixe[i]=1;
	double * coefsFixeAr=&aCoefsFixe[0];
	aSysRInit.AddEquation(pow(float(nbParcouru),1),coefsFixeAr,255/grMax);
	aSysGInit.AddEquation(pow(float(nbParcouru),1),coefsFixeAr,255/grMax);
	aSysBInit.AddEquation(pow(float(nbParcouru),1),coefsFixeAr,255/grMax);
}


//The brightest image is fixed, using superbig weight in least square:
double grMax=*max_element(Gr[aMasterNum].begin(),Gr[aMasterNum].end());
cout<<grMax<<endl;
vector<double> aCoefsFixe(nbIm,0.0);
aCoefsFixe[aMasterNum]=1;
double * coefsFixeAr=&aCoefsFixe[0];
aSysRInit.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,255/grMax);
aSysGInit.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,255/grMax);
aSysBInit.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,255/grMax);

cout<<"Solution of zeros preventing equations written"<<endl;

nbParcouru=0;
//For each linked couples :
for (int i=0;i<int(aPtsHomol[0].size());i++){

	int numImage1=(i/(nbIm));
	int numImage2=i-numImage1*(nbIm);
if (numImage1!=numImage2){
		if(aPtsHomol[0][i]!=0){//if there are homologous points between images
		vector<double> RIm1(aPtsHomol[2].begin()+nbParcouru,aPtsHomol[2].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> GIm1(aPtsHomol[3].begin()+nbParcouru,aPtsHomol[3].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> BIm1(aPtsHomol[4].begin()+nbParcouru,aPtsHomol[4].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> RIm2(aPtsHomol[6].begin()+nbParcouru,aPtsHomol[6].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> GIm2(aPtsHomol[7].begin()+nbParcouru,aPtsHomol[7].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> BIm2(aPtsHomol[8].begin()+nbParcouru,aPtsHomol[8].begin()+nbParcouru+aPtsHomol[0][i]-1);
		nbParcouru=nbParcouru+aPtsHomol[0][i];
		
		//adding equations for each point 
		for (int j=0;j<int(RIm1.size());j++){
			vector<double> aCoefsR(nbIm, 0.0);
			vector<double> aCoefsG(nbIm, 0.0);
			vector<double> aCoefsB(nbIm, 0.0);
			aCoefsR[numImage1]=RIm1[j];
			aCoefsG[numImage1]=GIm1[j];
			aCoefsB[numImage1]=BIm1[j];
					
			aCoefsR[numImage2]=-RIm2[j];
			aCoefsG[numImage2]=-GIm2[j];
			aCoefsB[numImage2]=-BIm2[j];
			
			double * coefsArR=&aCoefsR[0];
			double * coefsArG=&aCoefsG[0];
			double * coefsArB=&aCoefsB[0];
			double aPds=1;
			if(numImage1==aMasterNum || numImage2==aMasterNum){aPds=20;}
			aSysRInit.AddEquation(aPds,coefsArR,0);
			aSysGInit.AddEquation(aPds,coefsArG,0);
			aSysBInit.AddEquation(aPds,coefsArB,0);
		}
	}
}
}

cout<<"Solving the initial system"<<endl;
vector<vector<double> > aParam3Chan;

bool Ok1,Ok2,Ok3;
Im1D_REAL8 aSolR = aSysRInit.GSSR_Solve(&Ok1);
Im1D_REAL8 aSolG = aSysGInit.GSSR_Solve(&Ok2);
Im1D_REAL8 aSolB = aSysBInit.GSSR_Solve(&Ok3);
vector<double> aParamR, aParamG, aParamB;

if (Ok1 && Ok2 && Ok3)
{
    double* aDataR = aSolR.data();
    double* aDataG = aSolG.data();
    double* aDataB = aSolB.data();
	cout<<aSysRInit.ResiduOfSol(aDataR)<<endl;
	cout<<aSysGInit.ResiduOfSol(aDataG)<<endl;
	cout<<aSysBInit.ResiduOfSol(aDataB)<<endl;
	for(unsigned i=0;i<int(nbIm);i++){
		cout<<"For im NUM "<<i<<" CorR = "<<aDataR[i]<<" CorG = "<<aDataG[i]<<" CorB = "<<aDataB[i]<<endl;
		aParamR.push_back(aDataR[i]);
		aParamG.push_back(aDataG[i]);
		aParamB.push_back(aDataB[i]);
	}
	//Normalize the result :
	double maxFactorR=1/(*max_element(aParamR.begin(),aParamR.end()));
	double maxFactorG=1/(*max_element(aParamG.begin(),aParamG.end()));
	double maxFactorB=1/(*max_element(aParamB.begin(),aParamB.end()));
	for(unsigned i=0;i<int(nbIm);i++){
		if(maxFactorR>1){aParamR[i]=aParamR[i]*maxFactorR;}
		if(maxFactorG>1){aParamG[i]=aParamG[i]*maxFactorG;}
		if(maxFactorB>1){aParamB[i]=aParamB[i]*maxFactorB;}
	}
	aParam3Chan.push_back(aParamR);
	aParam3Chan.push_back(aParamG);
	aParam3Chan.push_back(aParamB);
}


//return aParam3Chan;


//Introducing more parameters (model is : alpha1*G1+beta1*G1*X1+gamma1*G1*Y1-alpha2*G2-beta2*G2*X2-gamma2*G2*Y2=0 )


	int nbParam=aDegPoly*2+1;//nb param per in the model
// Create L2SysSurResol to solve least square equation with nbIm unknown (equa is Kij*Ki-Kj=0)
	L2SysSurResol aSysR(nbParam*nbIm);
	L2SysSurResol aSysG(nbParam*nbIm);
	L2SysSurResol aSysB(nbParam*nbIm);


for(int i=0;i<int(nbIm);i++){
	double grMax=*max_element(Gr[i].begin(),Gr[i].end());
	vector<double> aCoefsFixe(nbParam*nbIm,0.0);
	aCoefsFixe[nbParam*i]=1;
	double * coefsFixeAr=&aCoefsFixe[0];
	aSysR.AddEquation(pow(float(nbParcouru),2),coefsFixeAr,aParam3Chan[0][i]);
	aSysG.AddEquation(pow(float(nbParcouru),2),coefsFixeAr,aParam3Chan[1][i]);
	aSysB.AddEquation(pow(float(nbParcouru),2),coefsFixeAr,aParam3Chan[2][i]);
	for (int a=1;a<nbParam;a++){
		vector<double> aCoefsFixe(nbParam*nbIm,0.0);
		aCoefsFixe[nbParam*i+a]=1;
		double * coefsFixeAr=&aCoefsFixe[0];
		aSysR.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,0);
		aSysG.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,0);
		aSysB.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,0);
	}
}

/*
		//The brightest image is fixed, using superbig weight in least square:
	vector<double> aCoefsFixe2(nbParam*nbIm,0.0);
	aCoefsFixe2[nbParam*aMasterNum]=1;
	double * coefsFixe2Ar=&aCoefsFixe2[0];
	aSysR.AddEquation(pow(float(nbParcouru),1),coefsFixe2Ar,255/grMax);
	aSysG.AddEquation(pow(float(nbParcouru),1),coefsFixe2Ar,255/grMax);
	aSysB.AddEquation(pow(float(nbParcouru),1),coefsFixe2Ar,255/grMax);
	for (int a=1;a<nbParam;a++){
		vector<double> aCoefsFixe(nbParam*nbIm,0.0);
		aCoefsFixe[nbParam*aMasterNum+a]=1;
		double * coefsFixeAr=&aCoefsFixe[0];
		aSysR.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,0);
		aSysG.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,0);
		aSysB.AddEquation(pow(float(nbParcouru),3),coefsFixeAr,0);
	}
*/

cout<<" --- Getting the equations from homologous points (model is : G1*poly(X1) + G1*poly(Y1) - G2*poly(X2) - G2*poly(Y2) = 0 )"<<endl;
nbParcouru=0;
//For each linked couples :
for (int i=0;i<int(aPtsHomol[0].size());i++){

	int numImage1=(i/(nbIm));
	int numImage2=i-numImage1*(nbIm);

if (numImage1!=numImage2){
	if(aPtsHomol[0][i]!=0){//if there are homologous points between images
		vector<double> RIm1(aPtsHomol[2].begin()+nbParcouru,aPtsHomol[2].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> GIm1(aPtsHomol[3].begin()+nbParcouru,aPtsHomol[3].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> BIm1(aPtsHomol[4].begin()+nbParcouru,aPtsHomol[4].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> RIm2(aPtsHomol[6].begin()+nbParcouru,aPtsHomol[6].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> GIm2(aPtsHomol[7].begin()+nbParcouru,aPtsHomol[7].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> BIm2(aPtsHomol[8].begin()+nbParcouru,aPtsHomol[8].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> X1(aPtsHomol[10].begin()+nbParcouru,aPtsHomol[10].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> Y1(aPtsHomol[11].begin()+nbParcouru,aPtsHomol[11].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> X2(aPtsHomol[12].begin()+nbParcouru,aPtsHomol[12].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> Y2(aPtsHomol[13].begin()+nbParcouru,aPtsHomol[13].begin()+nbParcouru+aPtsHomol[0][i]-1);
		nbParcouru=nbParcouru+aPtsHomol[0][i];
		
		//adding equations for each point 
		for (int j=0;j<int(RIm1.size());j++){
			vector<double> aCoefsR(nbParam*nbIm, 0.0);
			vector<double> aCoefsG(nbParam*nbIm, 0.0);
			vector<double> aCoefsB(nbParam*nbIm, 0.0);
			aCoefsR[nbParam*numImage1]=RIm1[j];
			aCoefsG[nbParam*numImage1]=GIm1[j];
			aCoefsB[nbParam*numImage1]=BIm1[j];
			aCoefsR[nbParam*numImage2]=-RIm2[j];
			aCoefsG[nbParam*numImage2]=-GIm2[j];
			aCoefsB[nbParam*numImage2]=-BIm2[j];
			for(int k=1;k<=((nbParam-1)/2);k++){
				aCoefsR[nbParam*numImage1+2*k-1]=RIm1[j]*pow(X1[j],k);
				aCoefsG[nbParam*numImage1+2*k-1]=GIm1[j]*pow(X1[j],k);
				aCoefsB[nbParam*numImage1+2*k-1]=BIm1[j]*pow(X1[j],k);
				aCoefsR[nbParam*numImage1+2*k]=RIm1[j]*pow(Y1[j],k);
				aCoefsG[nbParam*numImage1+2*k]=GIm1[j]*pow(Y1[j],k);
				aCoefsB[nbParam*numImage1+2*k]=BIm1[j]*pow(Y1[j],k);

				aCoefsR[nbParam*numImage2+2*k-1]=-RIm2[j]*pow(X2[j],k);
				aCoefsG[nbParam*numImage2+2*k-1]=-GIm2[j]*pow(X2[j],k);
				aCoefsB[nbParam*numImage2+2*k-1]=-BIm2[j]*pow(X2[j],k);
				aCoefsR[nbParam*numImage2+2*k]=-RIm2[j]*pow(Y2[j],k);
				aCoefsG[nbParam*numImage2+2*k]=-GIm2[j]*pow(Y2[j],k);
				aCoefsB[nbParam*numImage2+2*k]=-BIm2[j]*pow(Y2[j],k);
			}
					
			//cout<<aCoefsR[0]<<" - "<<aCoefsR[1]<<" - "<<aCoefsR[2]<<" - "<<aCoefsR[3]<<" - "<<aCoefsR[4]<<" - "<<aCoefsR[5]<<" - "<<aCoefsR[6]<<" - "<<aCoefsR[7]<<" - "<<aCoefsR[8]<<" - "<<aCoefsR[9]<<" - "<<aCoefsR[10]<<" - "<<aCoefsR[11]<<" - "<<aCoefsR[12]<<" - "<<aCoefsR[13]<<" - "<<aCoefsR[14]<<" - "<<aCoefsR[15]<<" - "<<aCoefsR[16]<<" - "<<aCoefsR[17]<<endl;
			double * coefsArR=&aCoefsR[0];
			double * coefsArG=&aCoefsG[0];
			double * coefsArB=&aCoefsB[0];
			double aPds=1;
			if(numImage1==aMasterNum || numImage2==aMasterNum){aPds=1;}
			aSysR.AddEquation(aPds,coefsArR,0);
			aSysG.AddEquation(aPds,coefsArG,0);
			aSysB.AddEquation(aPds,coefsArB,0);
		}
	}
}
}

cout<<"Solving the final system"<<endl;
//Clearing the output vector
aParam3Chan.clear();
aParamR.clear(); aParamG.clear(); aParamB.clear();

aSolR = aSysR.GSSR_Solve(&Ok1);
aSolG = aSysG.GSSR_Solve(&Ok2);
aSolB = aSysB.GSSR_Solve(&Ok3);

if (Ok1 && Ok2 && Ok3)
{
    double* aDataR = aSolR.data();
    double* aDataG = aSolG.data();
    double* aDataB = aSolB.data();
	for(unsigned i=0;i<int(nbParam*nbIm);i++){
		aParamR.push_back(aDataR[i]);
		aParamG.push_back(aDataG[i]);
		aParamB.push_back(aDataB[i]);
	}
	//Normalize the result :
	double maxFactorR=1/(*max_element(aParamR.begin(),aParamR.end()));
	double maxFactorG=1/(*max_element(aParamG.begin(),aParamG.end()));
	double maxFactorB=1/(*max_element(aParamB.begin(),aParamB.end()));
	for(unsigned i=0;i<int(nbParam*nbIm);i++){
		if(maxFactorR>1){aParamR[i]=aParamR[i]*maxFactorR;}
		if(maxFactorG>1){aParamG[i]=aParamG[i]*maxFactorG;}
		if(maxFactorB>1){aParamB[i]=aParamB[i]*maxFactorB;}
	}
	//Creating output
	aParam3Chan.push_back(aParamR);
	aParam3Chan.push_back(aParamG);
	aParam3Chan.push_back(aParamB);
}
return aParam3Chan;
}

void Egal_correct(string aDir,std::vector<std::string> * aSetIm,vector<vector<double> >  aParam3chan,string aDirOut)
{
	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
	//Reading input files
    int nbIm=(aSetIm)->size();
	int nbParam=aParam3chan[0].size()/nbIm;
	cout<<nbParam<<endl;
    for(int i=0;i<nbIm;i++)
	{
	    string aNameIm=(*aSetIm)[i];
		cout<<"Correcting "<<aNameIm<<endl;
		string aNameOut=aDir + aDirOut + aNameIm +"_egal.tif";

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
		
		//cout<<"Correction factors for image n"<<i<<" -> R : "<<aParam3chan[0][0+nbParam*i]<<endl;//" + "<<aParam3chan[0][1+nbParam*i]<<" * X + "<<aParam3chan[0][2+nbParam*i]<<" * Y + "<<aParam3chan[0][3+nbParam*i]<<" * X^2 + "<<aParam3chan[0][4+nbParam*i]<<" * Y^2 + "<<aParam3chan[0][5+nbParam*i]<<" * X^3 + "<<aParam3chan[0][6+nbParam*i]<<" * Y^3"<<endl;
		//cout<<"Correction factors for image n"<<i<<" -> G : "<<aParam3chan[1][0+nbParam*i]<<endl;//" + "<<aParam3chan[1][1+nbParam*i]<<" * X + "<<aParam3chan[1][2+nbParam*i]<<" * Y + "<<aParam3chan[1][3+nbParam*i]<<" * X^2 + "<<aParam3chan[1][4+nbParam*i]<<" * Y^2 + "<<aParam3chan[1][5+nbParam*i]<<" * X^3 + "<<aParam3chan[1][6+nbParam*i]<<" * Y^3"<<endl;
		//cout<<"Correction factors for image n"<<i<<" -> B : "<<aParam3chan[2][0+nbParam*i]<<endl;//" + "<<aParam3chan[2][1+nbParam*i]<<" * X + "<<aParam3chan[2][2+nbParam*i]<<" * Y + "<<aParam3chan[2][3+nbParam*i]<<" * X^2 + "<<aParam3chan[2][4+nbParam*i]<<" * Y^2 + "<<aParam3chan[2][5+nbParam*i]<<" * X^3 + "<<aParam3chan[2][6+nbParam*i]<<" * Y^3"<<endl;
		for (int aY=0 ; aY<aSz.y  ; aY++)
			{
				for (int aX=0 ; aX<aSz.x  ; aX++)
				{

					double corR=aParam3chan[0][nbParam*i],corG=aParam3chan[1][nbParam*i],corB=aParam3chan[2][nbParam*i];
					for(int j=1;j<=(nbParam-1)/2;j++){
						corR = corR + pow(float(aX),j) * aParam3chan[0][2*j-1+nbParam*i] + pow(float(aY),j) * aParam3chan[0][2*j+nbParam*i] ;
						corG = corG + pow(float(aX),j) * aParam3chan[1][2*j-1+nbParam*i] + pow(float(aY),j) * aParam3chan[1][2*j+nbParam*i] ;
						corB = corB + pow(float(aX),j) * aParam3chan[2][2*j-1+nbParam*i] + pow(float(aY),j) * aParam3chan[2][2*j+nbParam*i] ;
					}
					double R = aDataR[aY][aX] * corR;
					double G = aDataG[aY][aX] * corG;
					double B = aDataB[aY][aX] * corB;
					if(R>255){aDataR[aY][aX]=255;}else if(R<0){aDataR[aY][aX]=0;}else{aDataR[aY][aX]=R;}
					if(G>255){aDataG[aY][aX]=255;}else if(G<0){aDataG[aY][aX]=0;}else{aDataG[aY][aX]=G;}
					if(B>255){aDataB[aY][aX]=255;}else if(B<0){aDataB[aY][aX]=0;}else{aDataB[aY][aX]=B;}
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

int  Arsenic_main(int argc,char ** argv)
{

	std::string aFullPattern,aDirOut="Egal/",aMaster="";
	bool InTxt=false,DoCor=false;
	int aDegPoly=3;
	  //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern"),
            LArgMain()  << EAM(aDirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
						//<< EAM(InVig,"InVig",true,"Input vignette parameters")
						<< EAM(InTxt,"InTxt",true,"True if homologous points have been exported in txt (Defaut=false)")
						<< EAM(DoCor,"DoCor",true,"Use the computed parameters to correct the images (Defaut=false)")
						<< EAM(aMaster,"Master",true,"Manually define a Master Image (to be used a reference)")
						<< EAM(aDegPoly,"DegPoly",true,"Set the dergree of the corretion polynom (Def=3)")
        );
		std::string aDir,aPatIm;
		SplitDirAndFile(aDir,aPatIm,aFullPattern);

		std::string Extension = "dat";
		if (InTxt){Extension="txt";}

		cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
		const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

		std::vector<std::string> aVectIm=*aSetIm;
		int nbIm=aVectIm.size();
		//Looking for master image NUM:
		int aMasterNum=-1;
		for (int i=0;i<int(nbIm);i++){
			if(aVectIm[i]==aMaster){aMasterNum=i;cout<<"Found Master image "<<aMaster<<" as image NUM "<<i<<endl;}
		}
		//Reading homologous points
		vector<vector<double> > aPtsHomol=ReadPtsHom(aDir, & aVectIm, Extension);

		cout<<"Computing equalization factors"<<endl;
		vector<vector<double> > aParam3chan=Egalisation_factors(aPtsHomol,nbIm,aMasterNum,aDegPoly);
		if(aParam3chan.size()!=3){
			cout<<"Couldn't compute parameters "<<aParam3chan.size()<<endl;
		}else{

		if(DoCor){
			Egal_correct(aDir, & aVectIm, aParam3chan, aDirOut);
		}
		}
		Arsenic_Banniere();

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

