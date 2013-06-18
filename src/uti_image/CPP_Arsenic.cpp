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

	vector<double> NbPtsCouple,Gr1,Gr2,R1,G1,B1,R2,G2,B2;//Elements of output (distance from SIFT pts to center for Im1 and Im2, and respective grey lvl 
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

vector<vector<double> > Egalisation_factors(vector<vector<double> > aPtsHomol, int nbIm, int aMasterNum)
{
vector<vector<double> > K;
vector<double> Kgr,KR,KG,KB,Gr;

//int aNbIm=(1+sqrt(1+4*aPtsHomol[0].size()))/2;
int nbParcouru=0;
for (int i=0;i<int(aPtsHomol[0].size());i++){

	//Mutliplication factor between an image an itself is 1 (obviously)
	if ((i/(nbIm))*(nbIm)==i){
	//Add 1 for K(i,i)
			//cout<<"i = "<<i<<" and (i/(nbIm))*(nbIm) = "<<(i/(nbIm))*(nbIm)<<endl;
			Kgr.push_back(1);
			KR.push_back(1);
			KG.push_back(1);
			KB.push_back(1);
			Gr.push_back(0);
		}
	if(aPtsHomol[0][i]!=0){//if there are homologous points between images
		//Ransac
		vector<double> GrIm1(aPtsHomol[1].begin()+nbParcouru,aPtsHomol[1].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> RIm1(aPtsHomol[2].begin()+nbParcouru,aPtsHomol[2].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> GIm1(aPtsHomol[3].begin()+nbParcouru,aPtsHomol[3].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> BIm1(aPtsHomol[4].begin()+nbParcouru,aPtsHomol[4].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> GrIm2(aPtsHomol[5].begin()+nbParcouru,aPtsHomol[5].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> RIm2(aPtsHomol[6].begin()+nbParcouru,aPtsHomol[6].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> GIm2(aPtsHomol[7].begin()+nbParcouru,aPtsHomol[7].begin()+nbParcouru+aPtsHomol[0][i]-1);
		vector<double> BIm2(aPtsHomol[8].begin()+nbParcouru,aPtsHomol[8].begin()+nbParcouru+aPtsHomol[0][i]-1);
		nbParcouru=nbParcouru+aPtsHomol[0][i];

		//cout<<"finding optimal correction factor with RANSAC for couple "<<i<<endl;
		Kgr.push_back(oneParamRANSAC(GrIm1, GrIm2));
		KR.push_back(oneParamRANSAC(RIm1, RIm2));
		KG.push_back(oneParamRANSAC(GIm1, GIm2));
		KB.push_back(oneParamRANSAC(BIm1, BIm2));

		//Finding the brightest image step 1 : recording grey values
		double sumGr1 = std::accumulate(aPtsHomol[1].begin()+nbParcouru,aPtsHomol[1].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
		Gr.push_back(sumGr1/aPtsHomol[0][i]);
		//Moyenne des Gr1/Gr2
		/*
			double sumGr1 = std::accumulate(aPtsHomol[1].begin()+nbParcouru,aPtsHomol[1].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
			double sumR1  = std::accumulate(aPtsHomol[2].begin()+nbParcouru,aPtsHomol[2].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
			double sumG1  = std::accumulate(aPtsHomol[3].begin()+nbParcouru,aPtsHomol[3].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
			double sumB1  = std::accumulate(aPtsHomol[4].begin()+nbParcouru,aPtsHomol[4].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
			double sumGr2 = std::accumulate(aPtsHomol[5].begin()+nbParcouru,aPtsHomol[5].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
			double sumR2  = std::accumulate(aPtsHomol[6].begin()+nbParcouru,aPtsHomol[6].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
			double sumG2  = std::accumulate(aPtsHomol[7].begin()+nbParcouru,aPtsHomol[7].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
			double sumB2  = std::accumulate(aPtsHomol[8].begin()+nbParcouru,aPtsHomol[8].begin()+nbParcouru+aPtsHomol[0][i]-1,0.0);
			nbParcouru=nbParcouru+aPtsHomol[0][i];
			Kgr.push_back(sumGr1/sumGr2);
			KR.push_back(sumR1/sumR2);
			KG.push_back(sumG1/sumG2);
			KB.push_back(sumB1/sumB2);
			*/
		}else{
			Kgr.push_back(0);
			KR.push_back(0);
			KG.push_back(0);
			KB.push_back(0);
			Gr.push_back(0);

			}

}
			//Add 1 for K(last,last)
			Kgr.push_back(1);
			KR.push_back(1);
			KG.push_back(1);
			KB.push_back(1);
K.push_back(Kgr);K.push_back(KR);K.push_back(KG);K.push_back(KB);

/*
//Find transitions if not everything intervisible
while (*std::min_element(K[0].begin(),K[0].end())==0){
for (int i=0;i<int(aPtsHomol[0].size());i++){
	if(aPtsHomol[0][i]==0){
		long aScoreMax=0;
		int numImage1=(i/(nbIm-1));
		int numImage2=i-numImage1*(nbIm-1);
		int numImageLiaison;
		int ifSupMe1=0,ifSupMe2=0;
		for(int j=0;j<nbIm;j++){
			if(j==numImage1){ifSupMe1=1;continue;}
			if(j==numImage2){ifSupMe2=1;continue;}
			int nbPt1j=aPtsHomol[0][numImage1*(nbIm-1)+j-ifSupMe1];
			int nbPt2j=aPtsHomol[0][numImage2*(nbIm-1)+j-ifSupMe2];
			if(nbPt1j!=0 && nbPt2j!=0){
				long aScore=(nbPt1j+nbPt2j)/(fabs(double(nbPt1j-nbPt2j))+1);//sum of nb of homologous points with third party image, weighted by propotions
				if(aScoreMax<aScore){aScoreMax=aScore;numImageLiaison=j;}
			}
		}
		if(aScoreMax=!0){
			cout<<"Link created between "<<numImage1<<" and "<<numImage2<<endl;
			int ifSupMe=0,ifSupMe2=0;
			if(numImageLiaison>numImage1){ifSupMe1=1;}
			if(numImageLiaison>numImage2){ifSupMe2=1;}
			K[0][i]=K[0][numImage1*(nbIm-1)+numImageLiaison-ifSupMe1]*K[0][numImage2*(nbIm-1)+numImageLiaison-ifSupMe2];
			K[1][i]=K[1][numImage1*(nbIm-1)+numImageLiaison-ifSupMe1]*K[1][numImage2*(nbIm-1)+numImageLiaison-ifSupMe2];
			K[2][i]=K[2][numImage1*(nbIm-1)+numImageLiaison-ifSupMe1]*K[2][numImage2*(nbIm-1)+numImageLiaison-ifSupMe2];
			K[3][i]=K[3][numImage1*(nbIm-1)+numImageLiaison-ifSupMe1]*K[3][numImage2*(nbIm-1)+numImageLiaison-ifSupMe2];
		}

	}

}
}
vector<vector<double> >K_used;
vector<double> KR_used,KG_used,KB_used;
for(int i=0;i<int(nbIm);i++)
{
	if(i<imMax){KR_used.push_back(K[1][imMax*(nbIm-1)+i]);
				KG_used.push_back(K[2][imMax*(nbIm-1)+i]);
				KB_used.push_back(K[3][imMax*(nbIm-1)+i]);}
	else if(i>imMax){KR_used.push_back(K[1][imMax*(nbIm-1)+i-1]);
						KG_used.push_back(K[2][imMax*(nbIm-1)+i-1]);
						KB_used.push_back(K[3][imMax*(nbIm-1)+i-1]);}
	else{KR_used.push_back(1);KG_used.push_back(1);KB_used.push_back(1);}
}

K_used.push_back(KR_used);K_used.push_back(KG_used);K_used.push_back(KB_used);
*/

//Counting the amount of available equations:

cout<<"Choosing correction to apply"<<endl;

//Selecting the brightest image for reference
if(aMasterNum==-1){
	int imGMax;
	double Gmax = 0;
	for(int i=0;i<int(Gr.size());i++)
	{
		if(Gr[i]>Gmax){Gmax=Gr[i];imGMax=i;}
		//cout<<K[0][i]<<endl;
	}
	aMasterNum=(imGMax/(nbIm));
	cout<<"The brightest image is NUM "<<aMasterNum<<endl;
}


cout<<"Solving the system"<<endl;
vector<vector<double> > aParam3Chan;

for (unsigned aChan=1;aChan<=3;aChan++){
// Create L2SysSurResol to solve least square equation with nbIm unknown (equa is Kij*Ki-Kj=0)
	L2SysSurResol aSys(nbIm);

	//The brightest image is fixed:
	vector<double> coefsFixe(nbIm,0.0);
		coefsFixe[aMasterNum]=1;
		double * coefsFixeAr=&coefsFixe[0];
		aSys.AddEquation(10000,coefsFixeAr,1);

  	//For Each K, add equation
	for(unsigned i=0;i<int(nbIm*nbIm);i++){
			int numImage1=(i/(nbIm));
			int numImage2=i-numImage1*(nbIm);
			vector<double> coefs(nbIm,0.0);
			if(numImage1==numImage2 || K[aChan][i]==0){
			}else{
			coefs[numImage1]=K[aChan][i];
			coefs[numImage2]=-1;
			}
			//cout<<coefs[0]<< " " <<coefs[1]<< " " <<coefs[2]<< " " <<coefs[3]<< " " <<coefs[4]<< " " <<coefs[5]<< " " <<coefs[6]<< " " <<coefs[7]<< " " <<coefs[8]<< " " <<coefs[9]<< " " <<coefs[10]<< " " <<coefs[11]<< " " <<coefs[12]<< " " <<coefs[13]<< " " <<coefs[14]<< " " <<coefs[15]<< " " <<coefs[16]<<endl;
			double * coefsAr=&coefs[0];
			aSys.AddEquation(1,coefsAr,0);
	}
	bool Ok;
    Im1D_REAL8 aSol = aSys.GSSR_Solve(&Ok);
	vector<double> aParam;
    if (Ok)
    {
        double* aData = aSol.data();
		for(unsigned i=0;i<int(nbIm);i++){
			aParam.push_back(aData[i]);
		}
		aParam3Chan.push_back(aParam);
    }
}
return aParam3Chan;
}

void Egal_correct(string aDir,std::vector<std::string> * aSetIm,vector<vector<double> > K_used,string aDirOut)
{
	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
	//Reading input files
    int nbIm=(aSetIm)->size();
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
		
		double aCorR=K_used[0][i];
		double aCorG=K_used[1][i];
		double aCorB=K_used[2][i];
		cout<<"Correction factors for image n"<<i<<" =(R * "<<aCorR<<" - G * "<<aCorG<<" - B * "<<aCorB<<" )"<<endl;
		for (int aY=0 ; aY<aSz.y  ; aY++)
			{
				for (int aX=0 ; aX<aSz.x  ; aX++)
				{
					double R = aDataR[aY][aX] * aCorR;
					double G = aDataG[aY][aX] * aCorG;
					double B = aDataB[aY][aX] * aCorB;
					if(R>255){aDataR[aY][aX]=255;}else{aDataR[aY][aX]=R;}
					if(G>255){aDataG[aY][aX]=255;}else{aDataG[aY][aX]=G;}
					if(B>255){aDataB[aY][aX]=255;}else{aDataB[aY][aX]=B;}
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
		vector<vector<double> > K=Egalisation_factors(aPtsHomol,nbIm,aMasterNum);


		if(DoCor){
			Egal_correct(aDir, & aVectIm, K, aDirOut);
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

