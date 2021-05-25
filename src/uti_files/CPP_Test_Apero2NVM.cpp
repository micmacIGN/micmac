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

/**********************************/
/*   Author: Mathieu Monneyron	  */
/**********************************/


#include "StdAfx.h"
#include <algorithm>
#include "hassan/reechantillonnage.h"

vector<string> FindMatchFileAndIndex(string aNameDir, string aPattern, int ExpTxt, vector<vector<string> > &aVectImSift, vector<vector<int> > &aMatrixIndex) //finished
{
	//make a comment
	std::string aCom = "Finding File" + std::string(" +Ext=") + (ExpTxt?"txt":"dat");
	std::cout << "Com = " << aCom << "\n";

	//preparation for list and extension of file
	std::string aSiftDir= aNameDir + "Homol/Pastis";
	std::string aKee=std::string(".*.") + (ExpTxt?"txt":"dat");
	cout<<"sift_file_&_extension : "<<aKee<<endl;

	//Reading the list of input files (images)
    	list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    	int nbIm = (int)ListIm.size();
    	cout<<"Number of images to process: "<<nbIm<<endl;
	vector<string> VectIm;
	vector<string> aFullDir;

	for (int i=0;i<nbIm;i++)
    {
        //Reading the images list
        string aFullName=ListIm.front();
        cout<<aFullName<<endl;
	//transform the list in vector for manipulation
	VectIm.push_back(aFullName);
        ListIm.pop_front(); //cancel images once read
    }//end of "for each image"

	cout<<"findind file's list and index"<<endl;	

	for (int i=0;i<nbIm;i++)
	{//path for sift folders and list of file for each folder
	string aFullName=VectIm[i];
	string aFullSiftDir = aSiftDir + aFullName +"/";
	cout<<aFullSiftDir<<endl;
	aFullDir.push_back(aFullSiftDir);
	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aFullSiftDir);
	list<string> aFileList = anICNM->StdGetListOfFile(aKee,1); //here akee is quite alike aPattern
	int nbFiles = (int)aFileList.size();
	cout<<"nombre de fichier = "<< nbFiles <<endl;
	//delete anICNM;
	std::vector<string> VectSift;
	std::vector<int> VectInd;
		for (int j=0;j<nbFiles;j++)
    		{
      		  //Reading the sift-images list
        	string aFullName2=aFileList.front();
        	//cout<<aFullName2<<endl;
		//transform the list in vector for manipulation
		VectSift.push_back(aFullName2);
        	aFileList.pop_front(); //cancel name once read
		int k = 0;
		string aFullNameComp;
    			//look for the same name in the list of images and add corresponding index
			do
    			{
			aFullNameComp=VectIm[k] + "." + (ExpTxt?"txt":"dat");
        		k = k + 1;
			}
    			while (aFullNameComp.compare(aFullName2) != 0);
   			{VectInd.push_back(k-1);
			//cout<<"l'index est : "<<VectInd[j]<<endl;
			}
    		}//end of "for each file of sift of the concerned image"
	aVectImSift.push_back(VectSift);//write every name of the list of files of the list of images
	aMatrixIndex.push_back(VectInd);// write index associate to every files
	}//end for each images

	cout<<"nb of Images processed : "<<aVectImSift.size()<<endl;
	//cout<<"nb of file first image : "<<aVectImSift[0].size()<<endl;
	//cout<<"nb of index first image : "<<aMatrixIndex[0].size()<<endl;

	return aFullDir;
}

vector<vector<double> > CopyAndMergeMatchFile(string aNameDir, string aPattern, string DirOut, int ExpTxt) //finished
{
	//Bulding the output file system
   	ELISE_fp::MkDir(aNameDir + DirOut);
	std::cout<<"dossier " + DirOut + " cree"<<std::endl;
	std::cout<<aNameDir<<" & "<<aPattern<<" & "<<DirOut<<std::endl;

	//recuperate the matrix of file name and of associated index
	vector<vector<string> > V_ImSift;
	vector<vector<int> > Matrix_Index;
	vector<string> aFullSiftDir= FindMatchFileAndIndex(aNameDir, aPattern, ExpTxt, V_ImSift, Matrix_Index );

	//make a comment
	std::string aCom = "Reading and copying in a single File" + std::string(" +Ext=") + (ExpTxt?"txt":"dat");
	std::cout << "Com = " << aCom << "\n";

	//Name a new file which will contains all the previous data
	//string aOutputFileName= aNameDir+ DirOut + "SumOfSiftData.txt";
	int nbIm = (int)V_ImSift.size();
	double n1,n2,n3,n4;
	//bool ExpTxt;
	// Declare vector which will contains the value
	vector<double> N1,N2,N3,N4,N5,N6;
	vector<vector<double> > aFullTabSift;

	for (int i=0;i<nbIm;i++)
	{
	int nbFl = (int)V_ImSift[i].size();
		for (int j=0;j<nbFl;j++)
		{ //make a string for every file which include the directory
		string aFullFileName=aFullSiftDir[i] + V_ImSift[i][j]; 
		cout<<aFullFileName<<endl;
		int k= Matrix_Index[i][j];
		 //-------- !!! ----- Ne fonctionne qu'avec les fichiers .txt pour l'instant  ------ !!! ------ //
		if (ExpTxt != 1)
		{
                   ElPackHomologue aPack = ElPackHomologue::FromFile(aFullFileName);
                   for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
                   {
			N1.push_back(i); 
                        N2.push_back(itP->P1().x); // , N3.push_back(n2), N4.push_back(k), N5.push_back(n3), N6.push_back(n4);		
                        N3.push_back(itP->P1().y); 
                        N4.push_back(k);
                        N5.push_back(itP->P2().x);
                        N6.push_back(itP->P2().y);
                   }

/*
FILE *If = fopen(aFullFileName.c_str(), "rb"  );
		//FILE *Of = fopen(aOutputFileName.c_str(), "a" );		
			do
			{ //read all the file one by one
			int aNbVal = fscanf(If, "%lf %lf %lf %lf\n", &n1, &n2, &n3, &n4);
                        ELISE_ASSERT(aNbVal==4,"Bad nb val while scanning file in CopyAndMergeMatchFile");
			//fill the vectors with all the elements	
			N1.push_back(i), N2.push_back(n1), N3.push_back(n2), N4.push_back(k), N5.push_back(n3), N6.push_back(n4);		
			//fprintf(Of,"%d %0.1f% 0.1f %d %0.1f %0.1f\n", i, n1, n2, k, n3, n4);//write the file just for control
			}while(!feof(If));
		fclose(If);
*/
		//fclose(Of);
		}
		else
        	{FILE *If = fopen(aFullFileName.c_str(), "r"  );
		//FILE *Of = fopen(aOutputFileName.c_str(), "a" );		
			do
			{ //read all the file one by one
			int aNbVal = fscanf(If, "%lf %lf %lf %lf\n", &n1, &n2, &n3, &n4);
                        ELISE_ASSERT(aNbVal==4,"Bad nb val while scanning file in CopyAndMergeMatchFile");
			//fill the vectors with all the elements	
			N1.push_back(i), N2.push_back(n1), N3.push_back(n2), N4.push_back(k), N5.push_back(n3), N6.push_back(n4);		
			//fprintf(Of,"%d %0.1f% 0.1f %d %0.1f %0.1f\n", i, n1, n2, k, n3, n4);//write the file just for control
			}while(!feof(If));
		fclose(If);
		//fclose(Of);
		}
		}
	}
	//fill the vector of vectors with all the data vectors	
	aFullTabSift.push_back(N1),aFullTabSift.push_back(N2),aFullTabSift.push_back(N3),aFullTabSift.push_back(N4),aFullTabSift.push_back(N5),aFullTabSift.push_back(N6);
	/*
	// obtain file size in octets (not an obligation)
	FILE *Of = fopen(aOutputFileName.c_str(), "r" );
  	fseek (Of , 0 , SEEK_END);
  	long aFileSize = ftell (Of);
  	rewind (Of);
	cout<<"taille du fichier final : "<<aFileSize<<" octets"<<endl;
	*/
	cout<<"Nombre de lignes/de mesures : "<<aFullTabSift[0].size()<<endl;
	
	return aFullTabSift;
}

vector<vector<double> > CorrectTiePoint(string aNameDir, string aPattern, string DirOut, int ExpTxt) //not finished yet
{
	//load the data from the previous function	
	vector<vector<double> > aFullTabSift = CopyAndMergeMatchFile(aNameDir, aPattern, DirOut, ExpTxt);
	int NBSift = (int)aFullTabSift[0].size();
	//cout<<"Nombre de lignes/de mesures : "<<aFullTabSift[0].size()<<endl;
	//Reading the list of input files
	list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    	int nbIm = (int)ListIm.size();

	//loop for know where are the change of image in the vector aFullTabSift[0]
	int Value =0;
	vector<int> aIndexImage;
	aIndexImage.push_back(0);
	for (int j=0;j<NBSift;j++)
	{int ValIndex=aFullTabSift[0][j];
		if(ValIndex==Value+1)
		{aIndexImage.push_back(j);
		//cout<<"nb : "<<j<<endl;
		Value=Value+1;
		}
	}
	aIndexImage.push_back(NBSift);//add the last value

	//-------------------Double point Detection and cancellation--------------//
	//DECTECTION PART
	//Declare  vector which will contains the points who have the same coordinate
	vector<double> aListPtDouble;
	double diffx1=10;
	double diffy1=10;
	double diffx2=10;
	double diffy2=10;
	for (int i=0;i<nbIm-1;i++)
    	{
	int Alimitinf=aIndexImage[i];
	int Alimitsup=aIndexImage[i+1];
	//iter for every image, look at the points in the other image
	for (int h=Alimitinf;h<Alimitsup;h++)
	{
	double Valrefx1=aFullTabSift[1][h];
	double Valrefy1=aFullTabSift[2][h];
	double Valrefx2=aFullTabSift[4][h];
	double Valrefy2=aFullTabSift[5][h];
		//run the rest of the point of the all tab from the next image
		for (int m=Alimitsup;m<NBSift; m++)
		{
		double Valcompx1=aFullTabSift[4][m];
		double Valcompy1=aFullTabSift[5][m];
		double Valcompx2=aFullTabSift[1][m];
		double Valcompy2=aFullTabSift[2][m];
		diffx1=abs(Valcompx1-Valrefx1);
		diffy1=abs(Valcompy1-Valrefy1);
		diffx2=abs(Valcompx2-Valrefx2);
		diffy2=abs(Valcompy2-Valrefy2);
			if(diffx1<1.0 && diffy1<1.0 && diffx2<1.0 && diffy2<1.0 )//condition to accept that is the same point
			{aListPtDouble.push_back(m);//fill the vector of the n° of ligne of the double point
			//aListPtDoubleIni.push_back(h);
			//cout<<"Valeur de m : "<<m+1<<" pour le point : "<<h+1<<endl;
			}
		}
	}
	}
	//CANCELLATION PART
	sort( aListPtDouble.begin(),aListPtDouble.end() );
	int K = (int)aListPtDouble.size();
	cout<<"NB Double: "<<aListPtDouble.size()<<endl;
	for (int i=K-1;i>=0; i--)
	{
	long Val=aListPtDouble[i];
	//cout<<"Val :"<<Val<<endl;
		for (int j=0;j<6; j++)
		{aFullTabSift[j].erase(aFullTabSift[j].begin()+Val);}
	}
	int NBSiftM = (int)aFullTabSift[0].size();
	cout<<"NB restant: "<<NBSiftM<<endl;
	aListPtDouble.clear();

	//-------------------Fabrication of a new list -------------------------//
	//////////just for verif
	/*
	string aOutputFileName2= aNameDir+ DirOut + "SumOfSiftData2.txt"; 
	FILE *Of = fopen(aOutputFileName2.c_str(), "a" );		
	for( int l=0;l<NBSiftM; l++)
	{
	//fill the vectors with all the elements
	int a=aFullTabSift[0][l]; 
	double b=aFullTabSift[1][l];
	double c=aFullTabSift[2][l];
	int d=aFullTabSift[3][l];
	double e=aFullTabSift[4][l];
	double f=aFullTabSift[5][l];			
	fprintf(Of,"%d %0.1f% 0.1f %d %0.1f %0.1f\n",a,b,c,d,e,f );//write the file just for control
	}
	fclose(Of);
	*/

	//loop for know where are the change of image in the modified vector aFullTabSift[0]
	vector<int> aIndexImageM;
	aIndexImageM.push_back(0);
	int ValueM=0;
	for (int k=0;k<NBSiftM;k++)
	{int ValInd=aFullTabSift[0][k];
	//cout<<k<<endl;
	if (ValueM!=ValInd)
	{aIndexImageM.push_back(k);
	cout<<"nbr : "<<k<<endl;
	ValueM=ValueM+1;
	}
	}
	aIndexImageM.push_back(NBSiftM);//add the last value
	cout<<"size IndexIm : "<<aIndexImageM.size()<<endl;
	
	//----------------- Multiple  Point detection and cancellation-----------------//
	//DETECTION PART
	//Declare  vector which will contains the points who have the same coordinate
	vector<double> aListPtMulti,aListPtMultiIni;
	double diffx=10;
	double diffy=10;
	for (int i=0;i<nbIm;i++)
    	{
	int Alimitinf=aIndexImageM[i];
	int Alimitsup=aIndexImageM[i+1];
	//fill the vector of the n° of ligne of the multiple point
	for (int h=Alimitinf;h<Alimitsup-1;h++)
	{
		int l=h+1;
		double Valrefx=aFullTabSift[1][h];
		double Valrefy=aFullTabSift[2][h];
		//run the rest of the point of the same image from the next-current point
		while (l<Alimitsup)
		{
		double Valcompx=aFullTabSift[1][l];
		double Valcompy=aFullTabSift[2][l];
		diffx=abs(Valcompx-Valrefx);
		diffy=abs(Valcompy-Valrefy);
			if(diffx<1.0 && diffy<1.0 )
			{aListPtMulti.push_back(l);
			aListPtMultiIni.push_back(h);
			//cout<<"Valeur de l : "<<l<<" & Valeur de h : "<<h<<endl;
			}
		l=l+1;
		if(diffx<1.0 && diffy<1.0) break;//condition to accept that is the same point
		}
		//cout<<"diffx : "<<diffx<<endl;
		//cout<<"diffy : "<<diffy<<endl; 	
	}
	}
	cout<<"NB Multiple: "<<aListPtMulti.size()<<endl;
	cout<<"NB MultipleIni: "<<aListPtMultiIni.size()<<endl;

	//multiple point list tab inversion
	//creation of a new list
	vector<vector<double> > aFinalTabSift;
	for (int i=0;i<NBSiftM;i++)
	{
	vector<double> aPointTie;
	for (int j=0;j<6;j++)
	{
	double Val=aFullTabSift[j][i];
	aPointTie.push_back(Val);
	}
	aFinalTabSift.push_back(aPointTie);
	}
	aFullTabSift.clear();//suppression of the old vector
	cout<<"taille de la liste : "<<aFinalTabSift.size()<<endl;
	cout<<"taille pour un couple : "<<aFinalTabSift[0].size()<<endl;

	//CANCELLATION PART
	int N = (int)aListPtMulti.size();
	int Q=0;
	for (int i=0;i<N; i++)
	{int L=aListPtMulti[i];
	//cout<<"valeur de l "<<L<<endl;
		for (int j=0;j<N; j++)
		{int H=aListPtMultiIni[j];
		//cout<<"valeur de h "<<H<<endl;
		if (H==L)
		{//cout<<"l'indice "<<i<<" vaut "<<j<<endl;
		Q=Q+1;}
		}
	}
	//cout<<"NB Quadruple ou plus : "<<Q<<endl;

	int Q2=0;
	for (int i=0;i<N; i++)
	{
	int IndReceiv=aListPtMulti[i];
	int IndTaken=aListPtMultiIni[i];
	IndReceiv=IndReceiv - i;
	IndTaken=IndTaken - i;
	int S = (int)aFinalTabSift[IndTaken].size();
	if(S>6)
	{//cout<<"redondance :"<<S-6<<endl;
	Q2=Q2+1;}
	aFinalTabSift[IndTaken].erase(aFinalTabSift[IndTaken].begin(),aFinalTabSift[IndTaken].begin()+3);
	std::vector<double>::iterator it;
	it = aFinalTabSift[IndReceiv].begin();
	aFinalTabSift[IndReceiv].insert(it+6,aFinalTabSift[IndTaken].begin(),aFinalTabSift[IndTaken].end());

		//push_back the value from aListPtMultiIni to aListPtMulti
		//push_back ne fonctionne pas, erreur de memoire
		/*for (int j=3;j>S; j++)
		{
		double Val2Put= aFinalTabSift[IndTaken][j];
		aFinalTabSift[IndReceiv].resize(S+3,Val2Put);
		}*/

	//cout<<"Size of the merged point : "<<aFinalTabSift[IndReceiv].size()<<endl;
	//cancel the value corresponding to aListPtMultiIni in aFinalTabSift
	aFinalTabSift.erase(aFinalTabSift.begin()+IndTaken);
	}
	cout<<"NB Quadruple ou plus : "<<Q2<<endl;

	//cout the number of point
	int NBSiftFinal = (int)aFinalTabSift.size();
	cout<<"NB final restant: "<<NBSiftFinal<<endl;
	aListPtMulti.clear();
	aListPtMultiIni.clear();

	//-------------------Fabrication of a new list -------------------------//
	//////////just for verif
	/*
	string aOutputFileName3= aNameDir+ DirOut + "SumOfSiftData3.txt"; 
	FILE *Ot = fopen(aOutputFileName3.c_str(), "a" );		
	for( int l=0;l<NBSiftFinal; l++)
	{
	int NB=aFinalTabSift[l].size();
		for ( int m=0;m<NB; m++)
		{double maVal=aFinalTabSift[l][m];
	//fill the vectors with all the elements			
		fprintf(Ot,"% 1.1f",maVal );//write the file just for control
		}
	fprintf(Ot,"\n");
	}
	fclose(Ot);
	*/

	//loop for know where are the change of image in the vector aFinalTabSift[j][0]
	int ValueN=0;
	vector<int> aIndexImageN;
	aIndexImageN.push_back(0);
	for (int j=0;j<NBSiftFinal;j++)
	{int ValIndex=aFinalTabSift[j][0];
		if(ValIndex==ValueN+1)
		{aIndexImageN.push_back(j);
		cout<<"nb : "<<j<<endl;
		ValueN=ValueN+1;
		}
	}
	aIndexImageN.push_back(NBSiftFinal);//add the last value
	int p = (int)aIndexImageN.size();
	cout<<"size IndexIm : "<<p<<endl;
	cout<<"nombre d'image : "<<nbIm<<endl;

	//-----------------------Non-double point detection and cancelation-----------------//
	//DETECTION PART
	//Declare  vector which will contains the points who have the same coordinate
	vector<double> aListSimple,aListSimpleIni;
	for (int i=0;i<nbIm-1;i++)
    	{
	int Alimitinf=aIndexImageN[i];
	int Alimitsup=aIndexImageN[i+1];
	//fill the vector of the n° of ligne of the multiple point
	//cout<<Alimitinf<<" & "<<Alimitsup<<endl;
	double diff1=10;
	double diff2=10;
		for (int h=Alimitinf;h<Alimitsup;h++)
		{
		int ValInd1=aFinalTabSift[h][0];	
		double Valrefx=aFinalTabSift[h][1];
		double Valrefy=aFinalTabSift[h][2];
		//run the rest of the point on the other images in order to find the next same point
			for (int l=Alimitsup;l<NBSiftFinal;l++)
			{
			int ValInd2=aFinalTabSift[l][3];
			double Valcompx=aFinalTabSift[l][4];
			double Valcompy=aFinalTabSift[l][5];
			diff1=abs(Valcompx-Valrefx);
			diff2=abs(Valcompy-Valrefy);	
			if(ValInd1==ValInd2 && diff1<0.1 && diff2<0.1)
			{
			//cout<<Valrefx<<" & "<<Valrefy<<endl;
			//cout<<h<<endl;
			//cout<<Valcompx<<" & "<<Valcompy<<endl;
			//cout<<l<<endl;
			//cout<<"Size of the point : "<<aFinalTabSift[h].size()<<endl;
			//get the value
			double ValInD=aFinalTabSift[l][0];
			double ValX=aFinalTabSift[l][1];
			double ValY=aFinalTabSift[l][2];
			aFinalTabSift[h].push_back(ValInD),aFinalTabSift[h].push_back(ValX),aFinalTabSift[h].push_back(ValY);
			//cout<<"Size of the merged point : "<<aFinalTabSift[h].size()<<endl;
			aListSimple.push_back(l);
			aListSimpleIni.push_back(h);
			}
			//if(diffx<1.0 && diffy<1.0) break;//condition to accept that is the same point
			}	
		}
	}
	
	//CANCELLATION PART
	int Z = (int)aListSimple.size();
	cout<<"Nombre de Point Oubliés : "<<Z<<endl;
	int Q3=0;
	for (int i=0;i<Z; i++)
	{int L=aListSimple[i];
	//cout<<"valeur de l "<<L<<endl;
		for (int j=0;j<Z; j++)
		{int H=aListSimpleIni[j];
		//cout<<"valeur de h "<<H<<endl;
		if (H==L)
		{//cout<<"l'indice "<<i<<" vaut "<<j<<endl;
		Q3=Q3+1;}
		}
	}
	cout<<"NB pts Oubliés 2 fois ou plus : "<<Q3<<endl;

	sort( aListSimple.begin(),aListSimple.end() );
	for (int i=Z-1;i>=0; i--)
	{
	long Val=aListSimple[i];
	//cout<<"Val :"<<Val<<endl;
	aFinalTabSift.erase(aFinalTabSift.begin()+Val);
	}
	int NBTiePointFinal = (int)aFinalTabSift.size();
	cout<<"NB restant: "<<NBTiePointFinal<<endl;
	/*
	//-------------------Fabrication of a new list -------------------------//
	//////////just for verif
	string aOutputFileName4= aNameDir+ DirOut + "SumOfSiftData4.txt"; 
	FILE *Ou = fopen(aOutputFileName4.c_str(), "a" );		
	for( int l=0;l<NBTiePointFinal; l++)
	{
	int NB=aFinalTabSift[l].size();
		for ( int m=0;m<NB; m++)
		{double maVal=aFinalTabSift[l][m];
	//fill the vectors with all the elements			
		fprintf(Ou,"% 1.1f",maVal );//write the file just for control
		}
	fprintf(Ou,"\n");
	}
	fclose(Ou);
	*/
	return aFinalTabSift;
}

vector<vector<double> > GlobalCorrectionTiePoint(string aNameDir, string aPattern, string DirOut, string aOri, int ExpTxt, bool ExpTieP) //not finished yet
{
	//load the data from the previous function	
	vector<vector<double> >	aFinalTabSift=CorrectTiePoint(aNameDir, aPattern, DirOut, ExpTxt);
	int NBTiePoints = (int)aFinalTabSift.size();
	//Reading the list of input files
	list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    	int nbIm = (int)ListIm.size();
	//loop for know where are the change of image in the vector aFullTabSift[j][0]
	int Value =0;
	vector<int> aIndexImage;
	aIndexImage.push_back(0);
	for (int j=0;j<NBTiePoints;j++)
	{int ValIndex=aFinalTabSift[j][0];
		if(ValIndex==Value+1)
		{aIndexImage.push_back(j);
		//cout<<"nb : "<<j<<endl;
		Value=Value+1;
		}
	}
	aIndexImage.push_back(NBTiePoints);//add the last value
	cout<<"Global initialisation for correction ok !"<<endl;

	//-----------------------Global detection and cancelation-----------------//
	//DETECTION PART
	//Declare  vector which will contains the points who have the same coordinate
	vector<double> aList,aListIni;
	for (int i=0;i<nbIm-1;i++)
    	{
	int Alimitinf=aIndexImage[i];
	int Alimitsup=aIndexImage[i+1];
	//fill the vector of the n° of ligne of the multiple point
	//cout<<Alimitinf<<" & "<<Alimitsup<<endl;
	double diff1=10;
	double diff2=10;

	for (int h=Alimitinf;h<Alimitsup;h++)
	{//iterate the number of point of the same image
	int S1 = (int)(aFinalTabSift[h].size() / 3);
		//run the rest of the point on the other images in order to find a next point with a same measurement
		for (int l=Alimitsup;l<NBTiePoints;l++)
		{//iterate the number of point of the others images
		int S2 = (int)(aFinalTabSift[l].size() / 3);
		int exitNumber=0;
			for ( int i=1; i<S1; i++)
			{
			int ValInd1=aFinalTabSift[h][3*i];
			double Valrefx=aFinalTabSift[h][3*i+1];
			double Valrefy=aFinalTabSift[h][3*i+2];	
			int j=0;
				while (j<S2)
				{
				int ValInd2=aFinalTabSift[l][j*3];
				double Valcompx=aFinalTabSift[l][j*3+1];
				double Valcompy=aFinalTabSift[l][j*3+2];
				diff1=abs(Valcompx-Valrefx);
				diff2=abs(Valcompy-Valrefy);
				if(ValInd1==ValInd2 && diff1<0.1 && diff2<0.1)
				{aList.push_back(l),aListIni.push_back(h);
				exitNumber=l;
				//cout<<Valrefx<<" & "<<Valrefy<<endl;
				//cout<<h<<endl;
				//cout<<Valcompx<<" & "<<Valcompy<<endl;
				//cout<<l<<endl;
				//get the value
				}
				j=j+1;
				//we do only detection
				if(ValInd1==ValInd2 && diff1<0.1 && diff2<0.1) break;//condition to accept that is the same point
				}
			if (j!=S2+1) break;	
			}
		if (exitNumber!=0) break;
		}
	}
	}
	//CANCELLATION PART
	int Z = (int)(aList.size());
	cout<<"Nombre de Point réécris : "<<Z<<endl;
	int Q=0;
	for (int i=0;i<Z; i++)
	{int L=aList[i];
	//cout<<"valeur de l "<<L<<endl;
		for (int j=0;j<Z; j++)
		{int H=aListIni[j];
		//cout<<"valeur de h "<<H<<endl;
		if (H==L)
		{//cout<<"l'indice "<<i<<" vaut "<<j<<endl;
		Q=Q+1;}
		}
	}
	cout<<"NB pts réécris 2 fois ou plus : "<<Q<<endl;

	for (int i=0;i<Z; i++)
	{
	long IndVal1=aListIni[i]-i;
	long IndVal2=aList[i]-i;
	int S1 = (int)(aFinalTabSift[IndVal1].size() / 3);
	int S2 = (int)(aFinalTabSift[IndVal2].size() / 3);
	//cout<<"Val :"<<Val<<endl;
		for (int j=0;j<S1;j++)
		{//iterate for every measurement
		int Ind1=aFinalTabSift[IndVal1][j*3];
		int k=0;
			while (k<S2)
			{
			int Ind2=aFinalTabSift[IndVal2][k*3];
			if (Ind1==Ind2) break;
			//if the same value, measurement already exist
			k=k+1;
			}
		if(k==S2)//if not k==S2, add the value of Ind1 to Ind2 position
		{double VX=aFinalTabSift[IndVal1][j*3+1];
		double VY=aFinalTabSift[IndVal1][j*3+2];
		aFinalTabSift[IndVal2].push_back(Ind1),aFinalTabSift[IndVal2].push_back(VX),aFinalTabSift[IndVal2].push_back(VY);
		}
		}
	aFinalTabSift.erase(aFinalTabSift.begin()+IndVal1);
	}

	int NBTiePointFinal = (int)aFinalTabSift.size();
	cout<<"NB restant: "<<NBTiePointFinal<<endl;

	
	//------------------------- Tie Points corrected for distorsion---------------------//Actually-Not-Yet
	//define Boundaries and list of camera	
	int SizeImX=10000;
	int SizeImY=10000;
	vector<CamStenope *> aCam;
	for (int i=0;i<nbIm;i++)
	{
	string aFullName=ListIm.front();
        //cout<<aFullName<<" ("<<i+1<<" of "<<nbIm<<")"<<endl;
        ListIm.pop_front();
	//Formating the camera name
        string aNameCam="Ori-"+ aOri + "/Orientation-" + aFullName + ".xml";
        //Loading the camera
        cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
        CamStenope * aCS = CamOrientGenFromFile(aNameCam,anICNM);
	aCam.push_back(aCS);
	if (i==nbIm-1)
	{Tiff_Im aTF= Tiff_Im::StdConvGen(aNameDir + aFullName,3,false);
	//We suppose here that all the images have the same size, otherwise it will need a list of SizeIm...
	SizeImX	=aTF.sz().x;
	SizeImY	=aTF.sz().y;
	}
	}
	cout<<"Length of images : "<<SizeImX<<" & "<<" width of the images : "<<SizeImY<<endl;
	
	//make a comment
	std::string aCom = "Undistorting points" + std::string(" +Ext=") + (ExpTxt?"txt":"dat");
	std::cout << "Com = " << aCom << "\n";

	// Correction of the distorsion for sift point, no need to sampled after correction
	int P=0;
	int R=0;
	for (int j=0;j<NBTiePointFinal;j++)
	{
		int T = (int)(aFinalTabSift[j].size() / 3);
		for (int k=0;k<T;k++)
		{
		int indice=aFinalTabSift[j][3*k];
		CamStenope * aCS=aCam[indice];
		Pt2dr ptOut;
		Pt2dr NewPt( aFinalTabSift[j][3*k+1], aFinalTabSift[j][3*k+2]);
		ptOut=aCS->DistDirecte(NewPt);//here use only for filtered the UndistortIMed data
		double ValX=NewPt.x;//so no change for x
		double ValY=NewPt.y;//so no change for y
		//if the value UndistortIM is not in the space image, go to the next measurement
		if (ValX>SizeImX || ValY>SizeImY || ValX<0 || ValY<0)
		{k=k+1,R=R+1;}
		else
		{aFinalTabSift[j][k*3+1]=ValX;
		aFinalTabSift[j][k*3+2]=ValY;
		//cout<<"Point ancien: "<<NewPt<<endl;
		//cout<<"Point nouveau: "<<ptOut<<endl;
		P=P+1;}
		}		
	}
	cout<<"Nombre de Point 2D out: "<<R<<endl;
	cout<<"Nombre de Point 2D: "<<P<<endl;

	if ( ExpTieP !=0) 
	{
	//-------------------Fabrication of a new list -------------------------//
	//////////just for verif
	string aOutputFileName= aNameDir+ DirOut + "OriginalTiePtsList.txt"; 
	FILE *Ou = fopen(aOutputFileName.c_str(), "a" );
	int NBTiePointF = (int)aFinalTabSift.size();
	for( int l=0;l<NBTiePointF; l++)
	{
	int NB = (int)aFinalTabSift[l].size();
		for ( int m=0;m<NB; m++)
		{double maVal=aFinalTabSift[l][m];
	//fill the vectors with all the elements			
		if (mod(m,3)==0)
		{fprintf(Ou,"%2.0f",maVal );}
		else{fprintf(Ou,"% 1.2f",maVal );}
		}
	fprintf(Ou,"\n");
	}
	fclose(Ou);
	}
	
	return 	aFinalTabSift;
}

vector<double> LeastSquareSolv(vector<double> &centre1, vector<double> &Vdirecteur1, vector<double> &centre2, vector<double> &Vdirecteur2)
{
	////////Least square resolution////////
	//initialisation
	double alpha0=1; 
	double beta0=1;
	ElMatrix<double> A(2,3,0.0); ElMatrix<double> B(1,3,0.0); ElMatrix<double> dX(1,2,0.0); ElMatrix<double> N(2,2,0.0); ElMatrix<double> Atr(3,2,0.0); ElMatrix<double> Ninv(2,2,0.0);

	A(0,0)=Vdirecteur1[0]; A(1,0)=-Vdirecteur2[0];
	A(0,1)=Vdirecteur1[1]; A(1,1)=-Vdirecteur2[1];
	A(0,2)=Vdirecteur1[2]; A(1,2)=-Vdirecteur2[2];

	//here apha0 and beta0 useless because=1 but still written for formalism
	B(0,0)=-centre1[0]+centre2[0]-alpha0*Vdirecteur1[0]+beta0*Vdirecteur2[0];
	B(0,1)=-centre1[1]+centre2[1]-alpha0*Vdirecteur1[1]+beta0*Vdirecteur2[1];
	B(0,2)=-centre1[2]+centre2[2]-alpha0*Vdirecteur1[2]+beta0*Vdirecteur2[2];
	
	//transposed matrix
	Atr(0,0)=Vdirecteur1[0]; Atr(0,1)=-Vdirecteur2[0];
	Atr(1,0)=Vdirecteur1[1]; Atr(1,1)=-Vdirecteur2[1];
	Atr(2,0)=Vdirecteur1[2]; Atr(2,1)=-Vdirecteur2[2];

	//resolution
	N=Atr*A;
	//calcul of the inverse
	double K=1/(N(0,0)*N(1,1)-(N(1,0)*N(0,1)));
	//cout<<K<<endl;
	Ninv(0,0)=N(1,1), Ninv(0,1)=-N(0,1),Ninv(1,1)=N(0,0), Ninv(1,0)=-N(1,0);
	Ninv=Ninv*K;
	//calcul of the residus
	dX=Ninv*Atr*B;
	//Vc=B-A*dX;
	//Sig02=Vc'*Vc;
	double Alphafi=alpha0+dX(0,0);
	double Betafi=beta0+dX(0,1);
	//calculate the point
	double Xpt=(centre1[0]+Alphafi*Vdirecteur1[0]+centre2[0]+Betafi*Vdirecteur2[0])*0.5;
	double Ypt=(centre1[1]+Alphafi*Vdirecteur1[1]+centre2[1]+Betafi*Vdirecteur2[1])*0.5;
	double Zpt=(centre1[2]+Alphafi*Vdirecteur1[2]+centre2[2]+Betafi*Vdirecteur2[2])*0.5;	
	//cout<<" x: "<<Xpt<<" & "<<" y: "<<Ypt<<" & "<<" z: "<<Zpt<<endl;

	vector<double> aPt3D;
	aPt3D.push_back(Xpt), aPt3D.push_back(Ypt), aPt3D.push_back(Zpt);

	return aPt3D;
}

void Triangulation(string aNameDir, string aPattern, string aOri, string DirOut, int ExpTxt, bool ExpTieP, vector<vector<double> > &aFullTabTiePoint)
{ 
	vector<vector<double> > aFinalTabSift=GlobalCorrectionTiePoint(aNameDir, aPattern, DirOut, aOri, ExpTxt, ExpTieP);
	//Reading the list of input files
    	list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    	int nbIm = (int)ListIm.size();
    	//cout<<"Number of images to process: "<<nbIm<<endl;
	vector<CamStenope *> aCam;
	vector<Pt3dr> aListPtCentre;
	vector<double> aDeepth;
	//define Boundaries and list of camera	
	int SizeImX=10000;
	int SizeImY=10000;
	// And find the centre and read the camera of evry image
	for (int i=0;i<nbIm;i++)
   	{
        //Reading the images list
        string aFullName=ListIm.front();
        ListIm.pop_front(); //cancel images once read
	//Formating the camera name
        string aNameCam="Ori-"+ aOri + "/Orientation-" + aFullName + ".xml";
        //Loading the camera
	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
        CamStenope * aCS = CamOrientGenFromFile(aNameCam,anICNM);
	aCam.push_back(aCS);
	//get some values
	double aProf=aCS->GetProfondeur();
	aDeepth.push_back(aProf);
	//double x=aCS->VraiOpticalCenter().x ...
	//another way to find the centre
	Pt2dr aPtcentre(1000.0, 1000.0);
	Pt3dr centre=aCS->ImEtProf2Terrain(aPtcentre,0.0);
	aListPtCentre.push_back(centre);
	cout<<"centre Image "<<i<<" : "<<centre<<endl;
	//get the size ot the images
	if (i==nbIm-1)
	{Tiff_Im aTF= Tiff_Im::StdConvGen(aNameDir + aFullName,3,false);
	//We suppose here that all the images have the same size, otherwise it will need a list of SizeIm...
	SizeImX	=aTF.sz().x;
	SizeImY	=aTF.sz().y;
	}
	}
	cout<<"Length of the images : "<<SizeImX<<" & "<<" width of the images : "<<SizeImY<<endl;
	
	//make a comment
	std::string aCom = "Undistorting points" + std::string(" +Ext=") + (ExpTxt?"txt":"dat");
	std::cout << "Com = " << aCom << "\n";
	
	//////UndistortIMing tie points
	//////make list of centre and direction vector for each ligne
	int NPt = (int)aFinalTabSift.size();
	//int P=0;
	//int R=0;
	//vector<vector<double> > aFullTabTiePoint;//=aFinalTabSift with the cancellated point
	vector<vector<double> > aFullTabDroite;
	for (int j=0;j<NPt;j++)
	{
		vector<double> aTabDroite;
		vector<double> aTiePt;
		int T = (int)(aFinalTabSift[j].size() / 3);
		for (int k=0;k<T;k++)
		{
		int indice=aFinalTabSift[j][3*k];
		//recuperate the camera, the centrer and the profondeur
		CamStenope * aCS=aCam[indice];
		Pt3dr aPtCentr=aListPtCentre[indice];
		double aProfIm=aDeepth[indice];

		Pt2dr aPtIm(aFinalTabSift[j][3*k+1], aFinalTabSift[j][3*k+2]);
		//use of profondeur for put the point in 3d
		Pt3dr aPtTer=aCS->ImEtProf2Terrain(aPtIm, aProfIm);
		//Calculate the point UndistortIMed
		Pt2dr ptOut;
		ptOut=aCS->DistDirecte(aPtIm);
		double ValX=ptOut.x;
		double ValY=ptOut.y;	
			//if the value UndistortIM is not in the space image, go to the next measurement
			if (ValX>SizeImX || ValY>SizeImY || ValX<0 || ValY<0)
			{k=k+1;}//R=R+1;
			else
			{
			double xc,yc,zc,vx,vy,vz;
			xc=aPtCentr.x;
			yc=aPtCentr.y;
			zc=aPtCentr.z;
			vx=aPtTer.x;
			vy=aPtTer.y;
			vz=aPtTer.z;
			aTabDroite.push_back(xc),aTabDroite.push_back(yc),aTabDroite.push_back(zc);
			aTabDroite.push_back(vx-xc),aTabDroite.push_back(vy-yc),aTabDroite.push_back(vz-zc);
			double Ind=indice;	
			aTiePt.push_back(Ind),aTiePt.push_back(ValX),aTiePt.push_back(ValY);
			//cout<<"Point ancien: "<<NewPt<<endl;
			//cout<<"Point nouveau: "<<ptOut<<endl;
			}//P=P+1;
		}
		if (aTabDroite.size() >=12)//we keep if only tow or more new point are inside the box
		{aFullTabDroite.push_back(aTabDroite);
		aFullTabTiePoint.push_back(aTiePt);
		}	
	}
	//supression of the previous tab of tie points	 
	aFinalTabSift.clear();
	int NewNBLimit = (int)aFullTabTiePoint.size();
	cout<<"nombre de points final : "<<NewNBLimit<<endl;
	cout<<"fini list of droite !"<<endl;
	cout<<"Begin 3D intersection"<<endl;
	///////////////////////intersection in 3D space//////////////////////////////////
	vector<vector<double> > aTabPoint3D;
	for (int j=0;j<NewNBLimit;j++) //NewNBLimit
	{
	vector<vector<double> > aSolForaPt;
	int T = (int)(aFullTabDroite[j].size() / 6);
		//we work only with couple of point
		for (int k=0;k<T-1;k++)
		{//initialise data for first point
		vector<double> C1;
		vector<double> V1;
		C1.push_back(aFullTabDroite[j][6*k]),C1.push_back(aFullTabDroite[j][6*k+1]),C1.push_back(aFullTabDroite[j][6*k+2]);
		V1.push_back(aFullTabDroite[j][6*k+3]),V1.push_back(aFullTabDroite[j][6*k+4]),V1.push_back(aFullTabDroite[j][6*k+5]);
		//initialise data for second point
		int l=k+1;
		while (l<T)
		{
		vector<double> C2;
		vector<double> V2;
		C2.push_back(aFullTabDroite[j][6*l]),C2.push_back(aFullTabDroite[j][6*l+1]),C2.push_back(aFullTabDroite[j][6*l+2]);
		V2.push_back(aFullTabDroite[j][6*l+3]),V2.push_back(aFullTabDroite[j][6*l+4]),V2.push_back(aFullTabDroite[j][6*l+5]);
		//solve the intersection
		vector<double> aPt3D=LeastSquareSolv(C1,V1,C2,V2);
		aSolForaPt.push_back(aPt3D);
		l=l+1;
		}
		}
	//do a mean of every x,y,z and push_back the final x,y,z table
	int N = (int)aSolForaPt.size();
		if (N>1)
		{
		double X=0;double Y=0;double Z=0;
		vector<double> aSolPtmean;
			for (int i=0;i<N;i++)
			{
			X=X+aSolForaPt[i][0];
			Y=Y+aSolForaPt[i][1];
			Z=Z+aSolForaPt[i][2];
			}
		double Mx=X/N;double My=Y/N;double Mz=Z/N;
		aSolPtmean.push_back(Mx),aSolPtmean.push_back(My),aSolPtmean.push_back(Mz);
		aTabPoint3D.push_back(aSolPtmean);
		}
		//if size=1 dont require the mean operation
		else {aTabPoint3D.push_back(aSolForaPt[0]);}
	//aSolForaPt.clear();needent
	}
	
	//-------------------Fabrication of a new lists -------------------------//
	//////////just for verif/////List1
	string aOutputFileName= aNameDir+ DirOut + "DroitesList.txt"; 
	FILE *Ou = fopen(aOutputFileName.c_str(), "a" );		
	for( int l=0;l<NewNBLimit; l++)
	{
	int NB = (int)aFullTabDroite[l].size();
		for ( int m=0;m<NB; m++)
		{double maVal=aFullTabDroite[l][m];
	//fill the vectors with all the elements
		fprintf(Ou,"%-10.6f",maVal );
		}
	fprintf(Ou,"\n");
	}
	fclose(Ou);
	////////////////////////List2
	string aOutputFileName2= aNameDir+ DirOut + "3DPtsList.txt"; 
	FILE *Out = fopen(aOutputFileName2.c_str(), "a" );		
	for( int l=0;l<NewNBLimit; l++)
	{
		for ( int m=0;m<3; m++)
		{double maVal=aTabPoint3D[l][m];
	//fill the vectors with all the elements			
		fprintf(Out,"%-10.6f",maVal );
		}
	fprintf(Out,"\n");
	}
	fclose(Out);
	///////////////////List3
	string aOutputFileName3= aNameDir+ DirOut + "TiePtsUndistortedList.txt"; 
	FILE *Ouv = fopen(aOutputFileName3.c_str(), "a" );		
	for( int l=0;l<NewNBLimit; l++)
	{int NB = (int)aFullTabTiePoint[l].size();
		for ( int m=0;m<NB; m++)
		{double maVal=aFullTabTiePoint[l][m];
	//fill the vectors with all the elements			
		if (mod(m,3)==0)
		{fprintf(Ouv,"%2.0f",maVal );}
		else{fprintf(Ouv,"% 1.2f",maVal );}
		}
	fprintf(Ouv,"\n");
	}
	fclose(Ouv);
	cout<<"listes of points written"<<endl;

	//return aFullTabTiePoint;
}

void UndistortIM(string aNameDir, string aPattern, string aOri, string DirOut, bool KeepImC)
{
	if (KeepImC==1)
	{//if KeepImC==0 the undistortion will be done after in the last function
	//Reading the list of input files
    	list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    	int nbIm = (int)ListIm.size();
    	cout<<"Number of images to process: "<<nbIm<<endl;

	string cmdDRUNK,cmdConv,cmdDel;
    	list<string> ListDrunk,ListConvert,ListDel;

	for (int i=0;i<nbIm;i++)
    {
        //Reading the images list
        string aFullName=ListIm.front();
        //cout<<aFullName<<" ("<<i+1<<" of "<<nbIm<<")"<<endl;
        ListIm.pop_front(); //cancel images once read

	//Creating the lists of DRUNK and Convert commands
	//new images will be in the data folder
        cmdDRUNK=MMDir() + "bin/Drunk " + aNameDir + aFullName + " " + aOri + " " + "Out="+ DirOut;
        ListDrunk.push_back(cmdDRUNK);
        #if (ELISE_unix || ELISE_Cygwin || ELISE_MacOs)
            cmdConv="convert " + aNameDir + DirOut + aFullName + ".tif " + aNameDir + DirOut + aFullName + ".jpg";
			cmdDel = "rm " + aNameDir + DirOut + aFullName + ".tif ";
        #endif
        #if (ELISE_windows)
            cmdConv=MMDir() + "binaire-aux/windows/convert.exe " + aNameDir + DirOut + aFullName + ".tif " + aNameDir + DirOut + aFullName + ".jpg";
			cmdDel = "del " + aNameDir + DirOut + aFullName + ".tif ";
        #endif
        ListConvert.push_back(cmdConv);
		ListDel.push_back(cmdDel);

    }//end of "for each image"

    //Undistorting the images with Drunk
    cout<<"Undistorting the images with Drunk"<<endl;
    cEl_GPAO::DoComInParal(ListDrunk,aNameDir + "MkDrunk");

    //Converting into .jpg (CV solution may not use .tif) with Convert
    cout<<"Converting into .jpg"<<endl;
    cEl_GPAO::DoComInParal(ListConvert,aNameDir + "MkConvert");

	//Removing .tif
	cout << "Removing .tif" << endl;
	cEl_GPAO::DoComInParal(ListDel, aNameDir + "MkDel");

	}

}
vector <double> MakeQuaternion (CamStenope * aCS, ElMatrix<double> Rotc )
{
	//how to make the quaternion ?	
	vector <double> aQuaterion;
	ElMatrix<double> R(3,3,0.0);
	ElMatrix<double> Rot(3,3,0.0);
	R=aCS->Orient().Mat();
	double epsilon=0.000001;
	double z=0.5*pow(abs((1+R(0,0)+R(1,1)+R(2,2))),0.5);

	Rot=Rotc*R;
	double b=0;double c=0;double d=0;
	double a=0.5*pow(abs((1+Rot(0,0)+Rot(1,1)+Rot(2,2))),0.5);
	//double a2=-a;	
	//division by zero forbidden
	if (a<epsilon)
	{
	if (Rot(0,0)>=1.0-epsilon && Rot(0,0)<=1.0+epsilon)
	{b=1;}
	if (Rot(1,1)>=1-epsilon && Rot(1,1)<=1+epsilon)
	{c=1;}
	if (Rot(2,2)>=1-epsilon && Rot(2,2)<=1+epsilon)
	{d=1;}
	}
	else
	{
	b=0.25*(Rot(1,2)-Rot(2,1))/a;
	c=0.25*(Rot(2,0)-Rot(0,2))/a;
	d=0.25*(Rot(0,1)-Rot(1,0))/a;
	}
	//manage arrond problem
	if (z<epsilon)
	{
	if (R(0,0)>=1.0-epsilon && R(0,0)<=1.0+epsilon)
	{b=pow(1-a*a-c*c-d*d,0.5);
	}
	if (R(1,1)>=1-epsilon && R(1,1)<=1+epsilon)
	{c=pow(1-a*a-b*b-d*d,0.5);
	}
	if (R(2,2)>=1-epsilon && R(2,2)<=1+epsilon)
	{d=pow(1-a*a-b*b-c*c,0.5);
	}
	}
	else 
	{
	double K=pow(a*a+b*b+c*c+d*d,0.5);
	a=a/K; b=b/K; c=c/K; d=d/K;
	}
	aQuaterion.push_back(a),aQuaterion.push_back(b),aQuaterion.push_back(c),aQuaterion.push_back(d);
	cout<<a<<" & "<<b<<" & "<<c<<" & "<<d<<endl;

	return aQuaterion;
}

ElMatrix<double> CorrectRotation (double f, Pt2dr PP, bool KeepImC)
{	
	//We make a little rotation in order to correct the difference between PP and image centre
	double fx=pow(f*f+PP.x*PP.x,0.5);
	double fy=pow(f*f+PP.y*PP.y,0.5);
	//shift in x is rotation in y, and shift in y is rotation in x
	double cosa=f/fy;	
	double cosb=f/fx;
	double sina=PP.y/fy;
	double sinb=-PP.x/fx;
	//cout<<"cosinus : "<<cosa<<" & "<<cosb<<" sinus : "<<sina<<" & "<<sinb<<endl;
	ElMatrix<double> Rotc(3,3,0.0);//rotation matrix initialized at 0

	if (KeepImC==1)
	{Rotc(0,0)=1;Rotc(1,1)=1;Rotc(2,2)=1;}
	else
	{
	Rotc(0,0)=cosb; Rotc(2,0)=-sinb;
	Rotc(0,1)=sina*sinb; Rotc(1,1)=cosa; Rotc(2,1)=sina*cosb;
	Rotc(0,2)=cosa*sinb; Rotc(1,2)=-sina; Rotc(2,2)=cosa*cosb;
	cout<<"handle the difference between PP and Image centre"<<endl;
	}

	return Rotc;
}
Pt2dr CorrectDecentre(double f, Pt2dr PP,  Pt2di DSizeIm, Pt2dr PtIn, ElMatrix<double> Rotc )
{
	//Plan to Plan Projection, for each point just a scale factor k to find	
	double d=(f*f)+(PP.x*PP.x)+(PP.y*PP.y);
	double DX=PtIn.x-DSizeIm.x;
	double DY=PtIn.y-DSizeIm.y;
	//DZ=f;
	double k=d/(PP.x*DX+PP.y*DY+f*f);

	//Matrix inverse rotation 	
	ElMatrix<double> Rotinv(3,3,0.0);//Inverse rotation matrix initialized at 0
	Rotinv(0,0)=Rotc(0,0); Rotinv(0,1)=Rotc(1,0); Rotinv(0,2)=Rotc(2,0);
	Rotinv(1,0)=Rotc(0,1); Rotinv(1,1)=Rotc(1,1); Rotinv(1,2)=Rotc(2,1);
	Rotinv(2,0)=Rotc(0,2); Rotinv(2,1)=Rotc(1,2); Rotinv(2,2)=Rotc(2,2);
	
	ElMatrix<double> PtSpace(1,3,0.0);//point initialized for inverse rotation
	ElMatrix<double> PtSpaceF(1,3,0.0);//point resulted with the inverse rotation
	PtSpace(0,0)=k*DX;PtSpace(0,1)=k*DY;PtSpace(0,2)=k*f;
	//obtained the 3D point in the space
	PtSpaceF=Rotinv*PtSpace;
	//Z forgotten for the image
	Pt2dr PtOut(PtSpaceF(0,0)+PP.x+DSizeIm.x,PtSpaceF(0,1)+PP.y+DSizeIm.y);

	return PtOut;
}

void TransfORI_andWFile(string aNameDir, string aPattern, string aOri, string DirOut, string aNAME, vector<vector<double> > aFullTabTiePoint, bool ExpCloud, bool KeepImC)
{
	//Reading the list of input files
    	list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    	int nbIm = (int)ListIm.size();
    	cout<<"Convert the Orientations"<<endl;
	//initialiezd every list of internal data and its name
	// And define boundaries and list of camera
	vector<string> aListUndImName;
	vector<double> aListFocalOri;
	vector<Pt2di> aListDImSize;
	vector<Pt2dr> aListDeltaPP;
	vector<double> aListFocalF;
	vector<Pt3dr> aListPtCentre;
	ElMatrix<double> Rotc(3,3,0.0);
	vector<vector<double> > alistQuaternion;
	string cmdConv,cmdDel;
    list<string> ListConvert,ListDel;
	
	// And find the centre and read the camera of evry image
	for (int i=0;i<nbIm;i++)
   	{
        //Reading the images list
        string aFullName=ListIm.front();
        ListIm.pop_front(); //cancel images once read
	//Formating the camera name
        string aNameCam="Ori-"+ aOri + "/Orientation-" + aFullName + ".xml";
	//get the name of Undistorted images
	string UndImName=aFullName + ".jpg";
	aListUndImName.push_back(UndImName);
	string aNameOut=aNameDir + aFullName + ".tif";
        //Loading the camera
	cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
        CamStenope * aCS = CamOrientGenFromFile(aNameCam,anICNM);
	//get the  Undistorted images
	Tiff_Im aTF= Tiff_Im::StdConvGen(aNameDir+ aFullName,3,false);
	//get the focal lenght
	double FocL=aCS->Focale();
	aListFocalOri.push_back (FocL);
	//get the half size/centre of the images
	Pt2di aSz=aTF.sz();
	Pt2di aDSize=aSz/2;
	aListDImSize.push_back (aDSize);
	//cout<<"Valeur Demi Image : "<<aDSize.x<<" & "<<aDSize.y<<endl;
	//get the Principal Point
	Pt2dr PP=aCS->PP();
	Pt2dr DeltaPP(PP.x+0.5-aDSize.x, PP.y+0.5-aDSize.y);
	aListDeltaPP.push_back (DeltaPP);
	//calcul adjustement between PP and image centre
	double F=pow(FocL*FocL+DeltaPP.x*DeltaPP.x+DeltaPP.y*DeltaPP.y,0.5);
	aListFocalF.push_back(F);
	Rotc=CorrectRotation (FocL, DeltaPP, KeepImC);
	//correction of decentring
	//------------------------Fabrication of the Undistorted and Decentred image----------------------------//
	if (KeepImC==0)
	{
		Im2D_U_INT1  aImR(aSz.x,aSz.y);
    		Im2D_U_INT1  aImG(aSz.x,aSz.y);
    		Im2D_U_INT1  aImB(aSz.x,aSz.y);
    		Im2D_U_INT1  aImROut(aSz.x,aSz.y);
    		Im2D_U_INT1  aImGOut(aSz.x,aSz.y);
    		Im2D_U_INT1  aImBOut(aSz.x,aSz.y);

    		ELISE_COPY
    		(
    		  aTF.all_pts(),
     		  aTF.in(),
     		  Virgule(aImR.out(),aImG.out(),aImB.out())
   		 );

   		 U_INT1 ** aDataR = aImR.data();
   		 U_INT1 ** aDataG = aImG.data();
   		 U_INT1 ** aDataB = aImB.data();
   		 U_INT1 ** aDataROut = aImROut.data();
   		 U_INT1 ** aDataGOut = aImGOut.data();
   		 U_INT1 ** aDataBOut = aImBOut.data();

    		//Parcours des points de l'image de sortie et remplissage des valeurs
		cout<<"Undistorting and shifting the image"<<endl;
   		Pt2dr ptOut;Pt2dr ptIn;
		double x; double y;
   		 for (int aY=0 ; aY<aSz.y  ; aY++)
    		{
      		  for (int aX=0 ; aX<aSz.x  ; aX++)
       		 	{
			x=aX; y=aY;
			ptIn.x=x; ptIn.y=y;
			ptIn=aCS->DistDirecte(ptIn);
         		ptOut=CorrectDecentre(FocL, DeltaPP, aDSize, ptIn, Rotc );

            		aDataROut[aY][aX] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
            		aDataGOut[aY][aX] = Reechantillonnage::biline(aDataG, aSz.x, aSz.y, ptOut);
            		aDataBOut[aY][aX] = Reechantillonnage::biline(aDataB, aSz.x, aSz.y, ptOut);

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
      		   Virgule(aImROut.in(),aImGOut.in(),aImBOut.in()),
      		   aTOut.out()
    		  );

		#if (ELISE_unix || ELISE_Cygwin || ELISE_MacOs)
            	cmdConv="convert " + aNameDir + aFullName + ".tif " + aNameDir + DirOut + aFullName + ".jpg";
				cmdDel = "rm " + aNameDir + aFullName + ".tif";
       	#endif
       	#if (ELISE_windows)
           	cmdConv=MMDir() + "binaire-aux/windows/convert.exe ephemeral:" + aNameDir + aFullName + ".tif " + aNameDir + DirOut + aFullName + ".jpg";
			cmdDel = "del " + aNameDir + aFullName + ".tif";
        #endif
       	ListConvert.push_back(cmdConv);	
		ListDel.push_back(cmdDel);
	}
	///////////////////////////////////////////
	//get some values
	//double x=aCS->VraiOpticalCenter().x ...
	//another way to find the centre
	Pt2dr aPtcentre(1000.0, 1000.0);
	Pt3dr centre=aCS->ImEtProf2Terrain(aPtcentre,0.0);
	aListPtCentre.push_back(centre);
	//cout<<"centre Image "<<i<<" : "<<centre<<endl;
	//get the quaternion
	vector<double> aQuaternion=MakeQuaternion (aCS, Rotc);
	alistQuaternion.push_back(aQuaternion);
	}
	if (KeepImC==0)
	{
	cEl_GPAO::DoComInParal(ListConvert,aNameDir + "MkConvert");
	}
	int N = (int)aFullTabTiePoint.size();

	//////-------------------Fabrication of the final file -------------------------//////
	//if Point cloud Chosen initiate the ply file
	string aApeCloud= aNameDir+ DirOut + "3D.ply"; 
	if (ExpCloud ==1)
	{
	FILE *Plyf = fopen(aApeCloud.c_str(), "a" );
	fprintf(Plyf, "ply\n");
	fprintf(Plyf, "format ascii 1.0\n");
	fprintf(Plyf, "element vertex %d\n", N);
	fprintf(Plyf, "property float x\n");
	fprintf(Plyf, "property float y\n");
	fprintf(Plyf, "property float z\n");
	fprintf(Plyf, "property uchar red\n");
	fprintf(Plyf, "property uchar green\n");
	fprintf(Plyf, "property uchar blue\n");
	fprintf(Plyf, "element face 0\n");
	fprintf(Plyf, "property list uchar int vertex_indices\n");
	fprintf(Plyf, "end_header\n");
	fclose(Plyf);
	cout<<"Writting the ply file"<<endl;
	}

	/////////////      Begin of NVM_File           //////////////////
	string aOutputFileName=aNameDir + DirOut + aNAME + ".nvm"; 
	FILE *FOri = fopen(aOutputFileName.c_str(), "a" );
	//Begin to write the file and the orientation
	fprintf(FOri,"NVM_V3\n" ); //FixedK %d %d %d %d\n",F,PPx,F,PPy
	fprintf(FOri,"\n");	
	fprintf(FOri,"%d\n",nbIm );
	for( int i=0;i<nbIm; i++)
	{////write orientations//////
	string aName=aListUndImName[i];
	double F=aListFocalF[i];
	double VaQ1=alistQuaternion[i][0];double VaQ2=alistQuaternion[i][1];double VaQ3=alistQuaternion[i][2];double VaQ4=alistQuaternion[i][3];
	double aX=aListPtCentre[i].x;double aY=aListPtCentre[i].y;double aZ=aListPtCentre[i].z;
	fprintf(FOri,"%-10.100s %-5.6f %-2.6f %-2.6f %-2.6f %-2.6f %-2.6f %-2.6f %-2.6f 0 0\n", aName.c_str(),F,VaQ1,VaQ2,VaQ3,VaQ4,aX,aY,aZ);	
	}
	fprintf(FOri,"\n");
	fprintf(FOri,"%d\n", N );
	fclose(FOri);
	
	//call pts 3D list to include and continue to build the file
	double n1,n2,n3;
	string aReadFileNameP3= aNameDir+ DirOut + "3DPtsList.txt"; 
	FILE *If = fopen(aReadFileNameP3.c_str(), "r"  );
	FILE *Of = fopen(aOutputFileName.c_str(), "a" );			
	for ( int i=0;i<N; i++)
	{ 
	  int aNbVal =  fscanf(If, "%lf %lf %lf\n", &n1, &n2, &n3);
          ELISE_ASSERT(aNbVal==3,"Bad nb val while scanning file in TransfORI_andWFile");

	  int Sz=aFullTabTiePoint[i].size()/3.0;
	  fprintf(Of,"%-3.6f %-3.6f %-3.6f 128 128 128 %d",n1,n2,n3,Sz);
	//complete 3d ply if wanted
	  if (ExpCloud ==1)
	  {
	      FILE *Plyf = fopen(aApeCloud.c_str(), "a" );
	    fprintf(Plyf, "%-3.6f %-3.6f %-3.6f 128 128 128\n",n1,n2,n3);
	    fclose(Plyf);
	  }
		//Add the measurement (0.5 pixel difference between the two Image system origin )
		for (int j=0;j<Sz; j++)
		{
		int Indice=aFullTabTiePoint[i][3*j];
		double X=0; double Y=0;
		Pt2dr PtOut;
			if (KeepImC==0)
			{
			Pt2dr PtIn(aFullTabTiePoint[i][3*j+1],aFullTabTiePoint[i][3*j+2]);
			PtOut=CorrectDecentre(aListFocalOri[Indice], aListDeltaPP[Indice], aListDImSize[Indice], PtIn, Rotc );
			X=PtOut.x-aListDImSize[Indice].x+0.5;
			Y=PtOut.y-aListDImSize[Indice].y+0.5;
			}
			else
			{
			X=aFullTabTiePoint[i][3*j+1]-aListDImSize[Indice].x+0.5;
			Y=aFullTabTiePoint[i][3*j+2]-aListDImSize[Indice].y+0.5;
			}
		fprintf(Of,"% d %d %-4.6f %-4.6f",Indice,i,X,Y);
		}
	fprintf(Of,"\n");
	}
	fclose(If);
	//build footer
	fprintf(Of,"\n");
	fprintf(Of,"0\n");
	fprintf(Of,"#the last part of NVM file points to the PLY files\n");
	fprintf(Of,"#File generated by MicMac, a French and Open-Source solution\n");
	if (KeepImC==0)
	{fprintf(Of,"#feature are corrected from distortion and shift among PP and Image centre\n");}
	else
	{fprintf(Of,"#feature are corrected from distortion\n");}
	fprintf(Of,"1 0\n");
	fclose(Of);
	cout<<"file "<<aNAME<<".nvm written"<<endl;	
}

int  Apero2NVM_main(int argc,char ** argv)
{
   
   	std::string aFullPattern,aOri; // declaration of two arguments 
	std::string aNAME="ORI"; // default value ( for optional args )
	std::string DirOut="data/";
	int ExpTxt=0;
	//bool CMPMVS; finally not done
	bool ExpTieP=0;
	bool ExpCloud=0;
	bool KeepImC=0;
	vector<vector<double> > aFullTabTiePoint;
	//std::vector<vector<string> > V_ImSift;
	//std::vector<vector<int> > Matrix_Index;

	ElInitArgMain
	( 
		//MMD_InitArgcArgv(argc,argv);
		argc , argv , // list of args	
		LArgMain ()<< EAMC( aFullPattern , "Images Pattern", eSAM_IsPatFile ) //EAMC means mandatory argument
				<< EAMC( aOri , "Orientation name", eSAM_IsExistDirOri ),
		LArgMain ()<< EAM( aNAME, "Nom", false, "NVM file name" )//EAM means optional argument	
				//<< EAM(CMPMVS, "CMPMVS", false , "file mvs.ini & matrix contour" )
				<< EAM(DirOut,"Out",true,"Output folder (end with /)")
				<< EAM(ExpTxt,"ExpTxt",true,"Point in txt format ? (Def=false)", eSAM_IsBool)
				<< EAM(ExpCloud,"ExpApeCloud",false,"Exporte Ply? (Def=false)", eSAM_IsBool)
				<< EAM(ExpTieP,"ExpTiePt",false,"Export list of Tie Points uncorrected of the distortion ?(Def=false)", eSAM_IsBool)
				<< EAM(KeepImC,"KpImCen",false,"Dont add a little rotation for pass from Image Centre to PP ?(To be right fix LibPP=0 in tapas before)(Def=false)", eSAM_IsBool)
	);
	
	std::string aPattern,aNameDir;
   	SplitDirAndFile(aNameDir,aPattern,aFullPattern);

        StdCorrecNameOrient(aOri,aNameDir);

	
	//FindMatchFileAndIndex(aNameDir, aPattern, ExpTxt, &aVectImSift, &aMatrixIndex );
	//CopyAndMergeMatchFile(aNameDir, aPattern, DirOut, ExpTxt);
	//CorrectTiePoint(aNameDir, aPattern, DirOut, ExpTxt); 
	//GlobalCorrectionTiePoint(aNameDir, aPattern, DirOut, aOri, ExpTxt, ExpTieP); 
	Triangulation(aNameDir, aPattern, aOri, DirOut, ExpTxt, ExpTieP, aFullTabTiePoint);
	UndistortIM(aNameDir, aPattern, aOri, DirOut, KeepImC);
	TransfORI_andWFile(aNameDir, aPattern, aOri, DirOut, aNAME, aFullTabTiePoint, ExpCloud, KeepImC);


	return EXIT_SUCCESS ;

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
seule une responsabilitÃ© restreinte pÃšse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques DeltaPProfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃšmes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
