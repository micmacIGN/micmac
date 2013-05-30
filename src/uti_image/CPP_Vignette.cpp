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

vector<vector<double> > ReadPtsHom(string aDir,std::vector<std::string> * aSetIm,std::vector<std::vector<double> > vectOfExpTimeISO,string Extension)
{

	vector<double> D1,D2,G1,G2;//Elements of output (distance from SIFT pts to center for Im1 and Im2, and respective grey lvl 
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
			Im2D_U_INT1  aIm1(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF1.all_pts(),
				   aTF1.in(),
				   aIm1.out()
				);

			U_INT1 ** aData1 = aIm1.data();


        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
			Tiff_Im aTF2= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK2],1,false);
			Im2D_U_INT1  aIm2(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF2.all_pts(),
				   aTF2.in(),
				   aIm2.out()
				);

			U_INT1 ** aData2 = aIm2.data();

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
							   double Grey1 =(vectOfExpTimeISO[aK1][0]*vectOfExpTimeISO[aK1][1])/(maxExpTime*maxISO)*(Reechantillonnage::biline(aData1, aSz.x, aSz.y, itP->P1()));
							   double Grey2 =(vectOfExpTimeISO[aK2][0]*vectOfExpTimeISO[aK2][1])/(maxExpTime*maxISO)*(Reechantillonnage::biline(aData2, aSz.x, aSz.y, itP->P2()));

							   //Check that the distances are different-> might be used in filter?
							   //double rap=Dist1/Dist2;
							   if(1){//(Dist1>aSz.x/3 || Dist2>aSz.x/3)){// && (rap<0.75 || rap>1.33)){Filtre à mettre en place?
								   D1.push_back(Dist1);
								   D2.push_back(Dist2);
								   //X1.push_back(itP->P1().x);Unused, was for possible linear shift in vignette
								   //Y1.push_back(itP->P1().y);
								   //X2.push_back(itP->P2().x);
								   //Y2.push_back(itP->P2().y);
								   G1.push_back(Grey1);
								   G2.push_back(Grey2);
							   }
                   }
				   }
                   else
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
            }
        }
    }
	int nbpts=G1.size();
	std::cout<<"Total number of points used in least square : "<<nbpts<<" out of "<<cpt<<endl;
	vector<vector<double> > aPtsHomol;
	vector<double> SZ;
	SZ.push_back(aSz.x);SZ.push_back(aSz.y);
	aPtsHomol.push_back(D1);
	aPtsHomol.push_back(D2);
	aPtsHomol.push_back(G1);
	aPtsHomol.push_back(G2);
	aPtsHomol.push_back(SZ);
	//aPtsHomol.push_back(X1);
	//aPtsHomol.push_back(Y1);
	//aPtsHomol.push_back(X2);
	//aPtsHomol.push_back(Y2);
   return aPtsHomol;
}

void Vignette_correct(string aDir,std::vector<std::string> * aSetIm,vector<double> aParam,string aDirOut){

	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
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
					double aCor=255*(aParam[0]*D+aParam[1]*pow(D,2)+aParam[2]*pow(D,3));
					double R = aDataR[aY][aX] + aCor;
					double G = aDataG[aY][aX] + aCor;
					double B = aDataB[aY][aX] + aCor;
					if(R>255){aDataR[aY][aX]=255;}else if(aCor<0){continue;}else{aDataR[aY][aX]=R;}
					if(G>255){aDataG[aY][aX]=255;}else if(aCor<0){continue;}else{aDataG[aY][aX]=G;}
					if(B>255){aDataB[aY][aX]=255;}else if(aCor<0){continue;}else{aDataB[aY][aX]=B;}
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

vector<double> Vignette_Solve(vector<vector<double> > aPtsHomol)
{

	   // Create L2SysSurResol to solve least square equation with 3 unknown
	L2SysSurResol aSys(3);

  	//For Each SIFT point
	for(int i=0;i<int(aPtsHomol[0].size());i++){
				 double aPds[3]={(aPtsHomol[3][i]*pow(aPtsHomol[1][i],2)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],2)),
								 (aPtsHomol[3][i]*pow(aPtsHomol[1][i],4)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],4)),
								 (aPtsHomol[3][i]*pow(aPtsHomol[1][i],6)-aPtsHomol[2][i]*pow(aPtsHomol[0][i],6)),
								 //(aPtsHomol[7][i]*aPtsHomol[4][i]-aPtsHomol[6][i]*aPtsHomol[1][i]),
								 //(aPtsHomol[7][i]*aPtsHomol[5][i]-aPtsHomol[6][i]*aPtsHomol[2][i])
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
        std::cout << "Vignette parameters : " << aData[0] << " " << aData[1] << " " << aData[2] << "\n";
		aParam.push_back(aData[0]);
		aParam.push_back(aData[1]);
		aParam.push_back(aData[2]);
    }
		return aParam;
}

int  Vignette_main(int argc,char ** argv)
{

	std::string aFullPattern,aDirOut="Vignette/",InVig,InCal="",OutCal="Vignette.xml";
	bool InTxt=false,DoCor=false;
	  //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern"),
            LArgMain()  << EAM(aDirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
						//<< EAM(InVig,"InVig",true,"Input vignette parameters")
						<< EAM(InTxt,"InTxt",true,"True if homologous points have been exported in txt (Defaut=false)")
						<< EAM(InCal,"InCal",true,"Name of vignette calibration xml file (if previously computed)")
						<< EAM(DoCor,"DoCor",true,"Use the computed parameters to correct the images (Defaut=false)")
						<< EAM(OutCal,"OutCal",true,"Name of outgoing vignette calibration xml file (Default=Vignette.xml or InCal if exists)")
        );
		std::string aDir,aPatIm;
		SplitDirAndFile(aDir,aPatIm,aFullPattern);

		std::string Extension = "dat";
		if (InTxt){Extension="txt";}

		cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
		const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

		vector<vector<double> > vectOfDiaphFoc;
		vector<vector<string> > listOfListIm;
		vector<vector<vector<double> > > vectOfvectOfExpTimeISO;
		/*Test for insertion of read data
		if(1){
			vector<double> diaphFoc;diaphFoc.push_back(2.4);diaphFoc.push_back(4.1);vectOfDiaphFoc.push_back(diaphFoc);
			vector<string> newSetOfIm;listOfListIm.push_back(newSetOfIm);//init of the image groupe with diaph and Foc equal to thing read in xml
			vector<vector<double> > vectOfExpTimeISO; vectOfvectOfExpTimeISO.push_back(vectOfExpTimeISO);//idem with this
		}
		*/

		int nbInCal=0;
		//Read InCal
		if (InCal!=""){
			if(OutCal=="Vignette.xml"){OutCal=InCal;}
			//NEED TO READ XML
		}

		//Creating a new list of images for each combination of Diaph & Foc, and recording their ExpTime and ISO for future normalisation
		for (int j=0;j<(int)aSetIm->size();j++){
			std::string aFullName=(*aSetIm)[j];
			const cMetaDataPhoto & infoIm = cMetaDataPhoto::CreateExiv2(aFullName);
			vector<double> diaphFoc;diaphFoc.push_back(infoIm.Diaph());diaphFoc.push_back(infoIm.FocMm());
			vector<double> expTimeISO;expTimeISO.push_back(infoIm.ExpTime());expTimeISO.push_back(infoIm.IsoSpeed());
			cout<<"Getting Diaph and Focal from "<<aFullName<<endl;
			if (vectOfDiaphFoc.size()==0){
				vectOfDiaphFoc.push_back(diaphFoc);
				vector<string> newSetOfIm;
				newSetOfIm.push_back(aFullName);
				listOfListIm.push_back(newSetOfIm);
				vector<vector<double> > vectOfExpTimeISO;
				vectOfExpTimeISO.push_back(expTimeISO);
				vectOfvectOfExpTimeISO.push_back(vectOfExpTimeISO);
			}else{
				for (int i=0;i<(int)vectOfDiaphFoc.size();i++){
					if (diaphFoc==vectOfDiaphFoc[i]){
						listOfListIm[i].push_back(aFullName);
						vectOfvectOfExpTimeISO[i].push_back(expTimeISO);
						break;
						}else{if(i==(int)vectOfDiaphFoc.size()-1){
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
		for(int i=nbInCal;i<(int)listOfListIm.size();i++){
			std::cout << "--- Computing the parameters of the vignette effect for the set of "<<listOfListIm[i].size()<<" images with Diaph="<<vectOfDiaphFoc[i][0]<<" and Foc="<<vectOfDiaphFoc[i][1]<<endl<<endl;

		//Avec Points homol
			vector<vector<double> > aPtsHomol=ReadPtsHom(aDir, & listOfListIm[i], vectOfvectOfExpTimeISO[i],Extension);
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

			   //Il faut maintenant ecrire un fichier xml contenant foc+diaph+les params de vignette
			   cout<<"--- Writing XML"<<endl;
			   ofstream file_out(OutCal, ios::out | ios::app);
					if(file_out)  // if file successfully opened
					{
						file_out << endl <<"<SetParam> " <<endl;
							file_out << "    <Aperture> " << vectOfDiaphFoc[i][0] << " </Aperture>"<<endl;
							file_out << "    <Focal> " << vectOfDiaphFoc[i][1] << " </Focal>"<<endl;
								file_out << "    <p1> " << aParam[0] << " </p1>"<<endl;
								file_out << "    <p2> " << aParam[1] << " </p2>"<<endl;
								file_out << "    <p3> " << aParam[2] << " </p3>"<<endl;
						file_out << "</SetParam> " <<endl;
						file_out.close();
					}
					else{ cerr << "Couldn't wrie file" << endl;}
			   if (DoCor){
			   //Correction des images avec les params calculés
			   cout<<"Correcting the images"<<endl;
			   Vignette_correct(aDir, & listOfListIm[i],aParam,aDirOut);
	   
						}
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

