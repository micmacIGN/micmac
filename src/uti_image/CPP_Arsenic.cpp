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
#include "../src/uti_image/Arsenic.h"
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
    std::cout <<  " *     E-qualization and          *\n";
    std::cout <<  " *     N-ormalization for         *\n";
    std::cout <<  " *     I-nter-image               *\n";
    std::cout <<  " *     C-orrection                *\n";
    std::cout <<  " **********************************\n\n";
}

vector<ArsenicImage> LoadGrpImages(string aDir, std::string aPatIm, int ResolModel, string InVig)
{
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);
	std::vector<std::string> aVectIm=*aSetIm;

	//Scaling the images the comply with MM-Initial-Model
	list<string> ListConvert, ListVig;
	vector<std::string> VectImSc,VectMasq;
	int nbIm=aVectIm.size();
	char ResolModelch[3];sprintf(ResolModelch, "%02d", ResolModel);string ResolModelStr=(string)ResolModelch;
	if(ResolModel<10){ResolModelStr=ResolModelStr.substr(1,1);}
	//If a vignette correction folder is entered
	string postfix="";
	if(InVig!=""){
		string cmdVig=MMDir() + "bin/mm3d Vodka \"" + aPatIm + "\" DoCor=1 Out=" + InVig + " InCal=" + InVig;
		postfix="_Vodka.tif";
		ListVig.push_back(cmdVig);
		cEl_GPAO::DoComInParal(ListVig,aDir + "MkVig");
	}


	for (int aK1=0 ; aK1<nbIm ; aK1++)
    {
		string cmdConv=MMDir() + "bin/ScaleIm " + InVig + (aVectIm)[aK1] + postfix + " " + ResolModelStr + " F8B=1 Out=" + (aVectIm)[aK1] + "_Scaled.tif";
		ListConvert.push_back(cmdConv);

		//VectMasq.push_back("Masq-TieP-" + (aVectIm)[aK1] + "/RN" + (aVectIm)[aK1] + "_Masq.tif");
		VectMasq.push_back("MM-Malt-Img-" + StdPrefix((aVectIm)[aK1]) + "/Masq_STD-MALT_DeZoom" + ResolModelStr + ".tif");
		//cout<<VectMasq[aK1]<<endl;
		VectImSc.push_back((aVectIm)[aK1]+std::string("_Scaled.tif"));
	}
	cEl_GPAO::DoComInParal(ListConvert,aDir + "MkScale");

	vector<ArsenicImage> aGrIm;

	for (int aK1=0 ; aK1<int(nbIm) ; aK1++)
	{
		ArsenicImage aIm;
		//reading 3D info
		//cElNuage3DMaille * info3D1 = cElNuage3DMaille::FromFileIm("MM-Malt-Img-" + StdPrefix(aVectIm[aK1]) + "/NuageImProf_STD-MALT_Etape_1.xml");

		string arr[] = {"NaN", "7" ,  "6" , "NaN" , "5" , "NaN" , "NaN" , "NaN" , "4", "NaN" , "NaN" , "NaN" , "NaN", "NaN" , "NaN" , "NaN" , "3", "NaN" , "NaN" , "NaN" , "NaN", "NaN" , "NaN" , "NaN" , "NaN" , "NaN" , "NaN" , "NaN", "NaN" , "NaN" , "NaN" , "NaN", "2"};
		vector<string> numZoom(arr, arr+33);


		//cElNuage3DMaille * info3D1 = cElNuage3DMaille::FromFileIm("Masq-TieP-" + aVectIm[aK1] + "/NuageImProf_LeChantier_Etape_4.xml");
		cElNuage3DMaille * info3D1 = cElNuage3DMaille::FromFileIm("MM-Malt-Img-" + StdPrefix(aVectIm[aK1]) + "/NuageImProf_STD-MALT_Etape_"+ numZoom[ResolModel] + ".xml");

		aIm.info3D=info3D1;

		Tiff_Im aTF1= Tiff_Im::StdConvGen(aDir + VectImSc[aK1],3,false);
		Tiff_Im aTFM= Tiff_Im::StdConvGen(aDir + VectMasq[aK1],1,false);
		Pt2di aSz = aTF1.sz();
		Im2D_REAL4  aIm1R(aSz.x,aSz.y);
		Im2D_REAL4  aIm1G(aSz.x,aSz.y);
		Im2D_REAL4  aIm1B(aSz.x,aSz.y);
		Im2D_INT1  aMasq(aSz.x,aSz.y);
		ELISE_COPY
			(
				aTF1.all_pts(),
				aTF1.in(),
				Virgule(aIm1R.out(),aIm1G.out(),aIm1B.out())
			);
		
		ELISE_COPY
			(
				aTFM.all_pts(),
				aTFM.in(),
				aMasq.out()
			);

		aIm.Mask=aMasq;
		aIm.RChan=aIm1R;
		aIm.GChan=aIm1G;
		aIm.BChan=aIm1B;
		aIm.SZ=aSz;
		aGrIm.push_back(aIm);
	}

	return aGrIm;
}

double Dist3d(Pt3d<double> aP1, Pt3d<double> aP2 ){
	return (double)std::sqrt(pow(double(aP1.x-aP2.x),2)+pow(double(aP1.y-aP2.y),2)+pow(double(aP1.z-aP2.z),2));
}

double Dist2d(Pt2dr aP1, Pt2dr aP2 ){
	return (double)std::sqrt(pow(double(aP1.x-aP2.x),2)+pow(double(aP1.y-aP2.y),2));
}

void drawTP(PtsHom aPtsHomol, string aDir, string aNameOut, int ResolModel)
{
		//Bulding the output file system
		ELISE_fp::MkDirRec(aDir + "TP/");
		Pt2di aSz=aPtsHomol.SZ;
		cout<<aSz.x<<" "<<aSz.y<<endl;
		//Reading the image and creating the objects to be manipulated
		aNameOut=aDir + "TP/"+ aNameOut +".tif";
		Tiff_Im aTF=Tiff_Im(aNameOut.c_str(), aSz, GenIm::u_int1, Tiff_Im::No_Compr, Tiff_Im::RGB);

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
					aDataR[aY][aX]=0;
					aDataG[aY][aX]=0;
					aDataB[aY][aX]=0;
				}
		}
		for (int i=0;i<aPtsHomol.size();i++)
		{
			//cout<<int(aPtsHomol.Y1[i])<<" "<<int(aPtsHomol.X1[i])<<endl;
			aDataR[int(aPtsHomol.Pt1[i].y/ResolModel)][int(aPtsHomol.Pt1[i].x/ResolModel)]=255;
			aDataG[int(aPtsHomol.Pt1[i].y/ResolModel)][int(aPtsHomol.Pt1[i].x/ResolModel)]=255;
			aDataB[int(aPtsHomol.Pt1[i].y/ResolModel)][int(aPtsHomol.Pt1[i].x/ResolModel)]=255;
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

vector<PtsHom> ReadPtsHom3D(string aDir,string aPatIm,string Extension, string InVig, int ResolModel, double TPA)
{
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);
	std::vector<std::string> aVectIm=*aSetIm;
	int nbIm=aVectIm.size();
	vector<PtsHom> aVectPtsHomol(nbIm*nbIm);

	//Loading all images and usefull metadata (masks...)
	vector<ArsenicImage> aGrIm=LoadGrpImages(aDir, aPatIm, ResolModel, InVig);
	std::cout<<"===== "<<aGrIm.size()<< " images loaded"<<endl;


	//going throug each pair of different images
    for (int aK1=0 ; aK1<nbIm ; aK1++)
    {
		cout<<"Extracting point from image "<<aK1+1<<" out of "<<nbIm<<endl;
		//going throug each point of image
		for (int aY=0 ; aY<aGrIm[aK1].SZ.y  ; aY++)
		{
			for (int aX=0 ; aX<aGrIm[aK1].SZ.x  ; aX++)
			{				
				Pt2dr pos2DPtIm1;pos2DPtIm1.x=aX;pos2DPtIm1.y=aY;
				if(aGrIm[aK1].Mask.data()[aY][aX]==0){continue;}else{//If pts in masq, go look for 3D position
					Pt3d<double> pos3DPtIm1=aGrIm[aK1].info3D->PreciseCapteur2Terrain(pos2DPtIm1);	
						//Testing the position of the point in other images	
						vector<double> distances(nbIm,10000000); //distances between original 3D point and reprojection from other images
						vector<Pt2dr> pos2DOtherIm(nbIm);
						for (int aK2=0 ; aK2<int(nbIm) ; aK2++)
						{
							if (aK1!=aK2)
							{
							Pt2dr pos2DPtIm2=aGrIm[aK2].info3D->Ter2Capteur(pos3DPtIm1);
							//if pt in image and in masq, go look for 2D position, then 3D position
							if(	pos2DPtIm2.x>0 && pos2DPtIm2.x<aGrIm[aK2].SZ.x && pos2DPtIm2.y>0 && pos2DPtIm2.y<aGrIm[aK2].SZ.y){
										if(aGrIm[aK2].Mask.data()[int(pos2DPtIm2.y)][int(pos2DPtIm2.x)]){
											pos2DOtherIm[aK2]=pos2DPtIm2;
											Pt3d<double> pos3DPtIm2=aGrIm[aK2].info3D->PreciseCapteur2Terrain(pos2DPtIm2);
											//Compute Distance between the 2 3D points to check if they are the same ones (occlusion, beware!)
											distances[aK2]=Dist3d(pos3DPtIm1,pos3DPtIm2);
										}
									}
							}else{aVectPtsHomol[aK1*nbIm].SZ=aGrIm[aK1].SZ;}
						}
						for (int aK2=aK1+1 ; aK2<int(nbIm) ; aK2++){
							if(distances[aK2]<(aGrIm[aK1].info3D->ResolSolOfPt(pos3DPtIm1))/TPA){//id pos3DPtIm1~=pos3DPtIm2 -->pt is considered homologous,it is added to PtsHom (Gr1, R1, G1, B1, X1, Y1, idem 2, NbPtsCouple++)
								//Go looking for grey value of the point for each chan
								double Red1   =Reechantillonnage::biline(aGrIm[aK1].RChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
								double Green1 =Reechantillonnage::biline(aGrIm[aK1].GChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
								double Blue1  =Reechantillonnage::biline(aGrIm[aK1].BChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
								double Red2   =Reechantillonnage::biline(aGrIm[aK2].RChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
								double Green2 =Reechantillonnage::biline(aGrIm[aK2].GChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
								double Blue2  =Reechantillonnage::biline(aGrIm[aK2].BChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
								aVectPtsHomol[(aK1*nbIm)+aK2].Pt1.push_back(pos2DPtIm1.mul(ResolModel)) ;
								aVectPtsHomol[(aK1*nbIm)+aK2].Pt2.push_back(pos2DOtherIm[aK2].mul(ResolModel)) ;
								aVectPtsHomol[(aK1*nbIm)+aK2].Gr1.push_back((Red1+Green1+Blue1)/3);
								aVectPtsHomol[(aK1*nbIm)+aK2].Gr2.push_back((Red2+Green2+Blue2)/3);
								aVectPtsHomol[(aK1*nbIm)+aK2].R1.push_back(Red1);
								aVectPtsHomol[(aK1*nbIm)+aK2].G1.push_back(Green1);
								aVectPtsHomol[(aK1*nbIm)+aK2].B1.push_back(Blue1);
								aVectPtsHomol[(aK1*nbIm)+aK2].R2.push_back(Red2);
								aVectPtsHomol[(aK1*nbIm)+aK2].G2.push_back(Green2);
								aVectPtsHomol[(aK1*nbIm)+aK2].B2.push_back(Blue2);
								aVectPtsHomol[(aK1*nbIm)+aK2].NbPtsCouple++;
								aVectPtsHomol[(aK1*nbIm)+aK2].SZ=aGrIm[aK1].SZ;

								//Copy in inverse image (1->2==2->1)
								aVectPtsHomol[(aK2*nbIm)+aK1].Pt2.push_back(pos2DPtIm1.mul(ResolModel)) ;
								aVectPtsHomol[(aK2*nbIm)+aK1].Pt1.push_back(pos2DOtherIm[aK2].mul(ResolModel)) ;
								aVectPtsHomol[(aK2*nbIm)+aK1].Gr2.push_back((Red1+Green1+Blue1)/3);
								aVectPtsHomol[(aK2*nbIm)+aK1].Gr1.push_back((Red2+Green2+Blue2)/3);
								aVectPtsHomol[(aK2*nbIm)+aK1].R2.push_back(Red1);
								aVectPtsHomol[(aK2*nbIm)+aK1].G2.push_back(Green1);
								aVectPtsHomol[(aK2*nbIm)+aK1].B2.push_back(Blue1);
								aVectPtsHomol[(aK2*nbIm)+aK1].R1.push_back(Red2);
								aVectPtsHomol[(aK2*nbIm)+aK1].G1.push_back(Green2);
								aVectPtsHomol[(aK2*nbIm)+aK1].B1.push_back(Blue2);
								aVectPtsHomol[(aK2*nbIm)+aK1].NbPtsCouple++;
								aVectPtsHomol[(aK2*nbIm)+aK1].SZ=aGrIm[aK2].SZ;
							}
						}
				}
			}
		}
	}

		//drawTP(aVectPtsHomol[1], aDir, "1-2",ResolModel);
		//drawTP(aVectPtsHomol[2], aDir, "1-3",ResolModel);
		//drawTP(aVectPtsHomol[3], aDir, "1-4",ResolModel);
		//drawTP(aVectPtsHomol[4], aDir, "1-5",ResolModel);
		//drawTP(aVectPtsHomol[5], aDir, "2-1",ResolModel);
		//drawTP(aVectPtsHomol[7], aDir, "2-3",ResolModel);
		//drawTP(aVectPtsHomol[8], aDir, "2-4",ResolModel);
		//drawTP(aVectPtsHomol[9], aDir, "2-5",ResolModel);
		//drawTP(aVectPtsHomol[10], aDir, "3-1",ResolModel);
		//drawTP(aVectPtsHomol[11], aDir, "3-2",ResolModel);
		//drawTP(aVectPtsHomol[13], aDir, "3-4",ResolModel);
		//drawTP(aVectPtsHomol[14], aDir, "3-5",ResolModel);
		//drawTP(aVectPtsHomol[15], aDir, "4-1",ResolModel);
		//drawTP(aVectPtsHomol[16], aDir, "4-2",ResolModel);
		//drawTP(aVectPtsHomol[17], aDir, "4-3",ResolModel);
		//drawTP(aVectPtsHomol[19], aDir, "4-5",ResolModel);
		//drawTP(aVectPtsHomol[20], aDir, "5-1",ResolModel);
		//drawTP(aVectPtsHomol[21], aDir, "5-2",ResolModel);
		//drawTP(aVectPtsHomol[22], aDir, "5-3",ResolModel);
		//drawTP(aVectPtsHomol[23], aDir, "5-4",ResolModel);
		int nbPtsHomols=0;
		for(int i=0 ; i<int(aVectPtsHomol.size()) ; i++){nbPtsHomols=nbPtsHomols + aVectPtsHomol[i].NbPtsCouple;}
		ELISE_ASSERT(nbPtsHomols!=0,"No homologous points (resolution of ResolModel might be too small");
		return aVectPtsHomol;
		
}

void Egal_field_correct(string aDir,std::vector<std::string> * aSetIm,vector<PtsHom> aVectPtsHomol, string aDirOut, string InVig, int ResolModel, int nbIm)
{
int aNbCouples=aVectPtsHomol.size();
vector<PtsRadioTie> vectPtsRadioTie(nbIm);

//truc à iterer--------------------------------------------------------------------------------------------------------------------------------------
for(int iter=0;iter<4;iter++){
	cout<<"Pass "<<iter<<endl;
	int nbPts=0;
	vector<PtsRadioTie> vectPtsRadioTie2(nbIm);
	//filling up the factors from homologous points
	for (int i=0;i<int(aNbCouples);i++){

		int numImage1=(i/(nbIm));
		int numImage2=i-numImage1*(nbIm);
		//string PtsTxt="PtsTxt.txt";
		//ofstream file_out(PtsTxt, ios::out | ios::app);
		//cout<<numImage1<<" "<<numImage2<<" "<<aVectPtsHomol[i].NbPtsCouple<<endl;
		if (numImage1!=numImage2){
			for(int j=0; j<aVectPtsHomol[i].NbPtsCouple; j++){//if there are homologous points between images
				//For each chan, compute GreyLevelImage2/GreyLevelImage1 and add it to image1 vector of tie points
				double kR=aVectPtsHomol[i].R2[j]/aVectPtsHomol[i].R1[j];
				double kG=aVectPtsHomol[i].G2[j]/aVectPtsHomol[i].G1[j];
				double kB=aVectPtsHomol[i].B2[j]/aVectPtsHomol[i].B1[j];
				vectPtsRadioTie2[numImage1].kR.push_back((kR+1)/2.0);
				vectPtsRadioTie2[numImage1].kG.push_back((kG+1)/2.0);
				vectPtsRadioTie2[numImage1].kB.push_back((kB+1)/2.0);
				vectPtsRadioTie2[numImage1].Pos.push_back(aVectPtsHomol[i].Pt1[j]);
				vectPtsRadioTie2[numImage1].OtherIm.push_back(numImage2);
				//vectPtsRadioTie[numImage1].multiplicity.push_back(1);
				//if(vectPtsRadioTie[numImage1].Pos[vectPtsRadioTie[numImage1].Pos.size()-2]==vectPtsRadioTie[numImage1].Pos.back())
				//{
				//	int previousMultiplicity=vectPtsRadioTie[numImage1].multiplicity[vectPtsRadioTie[numImage1].multiplicity.size()-2];
				//	for(int mul=1;mul<=previousMultiplicity+1;mul++)
				//	{
				//		vectPtsRadioTie[numImage1].multiplicity[vectPtsRadioTie[numImage1].multiplicity.size()-mul]++;
				//		//cout<<vectPtsRadioTie[numImage1].multiplicity[vectPtsRadioTie[numImage1].multiplicity.size()-mul]<<endl;
				//	}
				//}
				nbPts++;
				//file_out <<kR<<endl;
			}
		}
		//file_out.close();
	}
	//cout<<nbPts<<" tie points loaded"<<endl;


//Correcting the tie points

//#pragma omp parallel for

    for(int i=0;i<nbIm;i++)
	{
		vector<int> cpt(nbIm,0);
		
		//For each tie point point, compute correction value (distance-ponderated mean value of all the tie points)
		for(int k = 0; k<int(vectPtsRadioTie2[i].size()) ; k++){//go through each tie point
			double aCorR=0.0,aCorG=0.0,aCorB=0.0;
			double aSumDist=0;
			Pt2dr aPt; aPt.x=vectPtsRadioTie2[i].Pos[k].x/ResolModel; ; aPt.y=vectPtsRadioTie2[i].Pos[k].x/ResolModel;;
			for(int j = 0; j<int(vectPtsRadioTie2[i].size()) ; j++){//go through each tie point
				if(vectPtsRadioTie2[i].kR[j]>5||vectPtsRadioTie2[i].kG[j]>5||vectPtsRadioTie2[i].kB[j]>5 || vectPtsRadioTie2[i].kR[j]<0.2||vectPtsRadioTie2[i].kG[j]<0.2||vectPtsRadioTie2[i].kB[j]<0.2){continue;}
				Pt2dr aPtIn; aPtIn.x=vectPtsRadioTie2[i].Pos[j].x/ResolModel; aPtIn.y=vectPtsRadioTie2[i].Pos[j].y/ResolModel;
				double aDist=Dist2d(aPtIn, aPt);
				if(aDist<1){aDist=1;}
				aSumDist=aSumDist+1/(aDist);
				aCorR = aCorR + vectPtsRadioTie2[i].kR[j]/(aDist);
				aCorG = aCorG + vectPtsRadioTie2[i].kG[j]/(aDist);
				aCorB = aCorB + vectPtsRadioTie2[i].kB[j]/(aDist);	
			}
			//Normalize
			aCorR = aCorR/aSumDist;
			aCorG = aCorG/aSumDist; 
			aCorB = aCorB/aSumDist; 
			//correcting Tie points color with computed surface
			int image1=i,image2=vectPtsRadioTie2[i].OtherIm[k];
			int pos=cpt[image2];cpt[image2]++;
			if(aVectPtsHomol[image1*nbIm+image2].R1[pos]*aCorR>255)
			{
				aCorR=255/aVectPtsHomol[image1*nbIm+image2].R1[pos];
			}
			if(aVectPtsHomol[image1*nbIm+image2].G1[pos]*aCorB>255)
			{
				aCorG=255/aVectPtsHomol[image1*nbIm+image2].G1[pos];
			}
			if(aVectPtsHomol[image1*nbIm+image2].B1[pos]*aCorG>255)
			{
				aCorB=255/aVectPtsHomol[image1*nbIm+image2].B1[pos];
			}
			aVectPtsHomol[image1*nbIm+image2].R1[pos]=aVectPtsHomol[image1*nbIm+image2].R1[pos]*aCorR;
			aVectPtsHomol[image1*nbIm+image2].G1[pos]=aVectPtsHomol[image1*nbIm+image2].G1[pos]*aCorG;
			aVectPtsHomol[image1*nbIm+image2].B1[pos]=aVectPtsHomol[image1*nbIm+image2].B1[pos]*aCorB;

		    aVectPtsHomol[image2*nbIm+image1].R2[pos]=aVectPtsHomol[image1*nbIm+image2].R1[pos];
		    aVectPtsHomol[image2*nbIm+image1].G2[pos]=aVectPtsHomol[image1*nbIm+image2].G1[pos];
		    aVectPtsHomol[image2*nbIm+image1].B2[pos]=aVectPtsHomol[image1*nbIm+image2].B1[pos];
		}
		//cout<<cpt<<endl;
	}
	if (iter==0){vectPtsRadioTie=vectPtsRadioTie2;}else{
		for(int i=0;i<nbIm;i++)
		{
			for(int j=0;j<vectPtsRadioTie[i].size();j++)
			{										   
			vectPtsRadioTie[i].kR[j]=vectPtsRadioTie[i].kR[j]*vectPtsRadioTie2[i].kR[j];
			vectPtsRadioTie[i].kG[j]=vectPtsRadioTie[i].kG[j]*vectPtsRadioTie2[i].kG[j];
			vectPtsRadioTie[i].kB[j]=vectPtsRadioTie[i].kB[j]*vectPtsRadioTie2[i].kB[j];
			}
		}
	}
}
cout<<"Factors were computed"<<endl;
//end truc à iterer--------------------------------------------------------------------------------------------------------------------------------------



//Applying the correction to the images
	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
	//Reading input files
	string suffix="";if(InVig!=""){suffix="_Vodka.tif";}


#pragma omp parallel for

    for(int i=0;i<nbIm;i++)
	{
	    string aNameIm=InVig + (*aSetIm)[i] + suffix;//if vignette is used, change the name of input file to read
		cout<<"Correcting "<<aNameIm<<" (with "<<vectPtsRadioTie[i].size()<<" data points)"<<endl;
		string aNameOut=aDir + aDirOut + (*aSetIm)[i] +"_egal.tif";

		Pt2di aSzMod=aVectPtsHomol[i*nbIm].SZ;//Size of the correction surface, taken from the size of the scaled image
		cout<<"aSzMod"<<aSzMod<<endl;
		Im2D_REAL4  aImCorR(aSzMod.x,aSzMod.y,0.0);
		Im2D_REAL4  aImCorG(aSzMod.x,aSzMod.y,0.0);
		Im2D_REAL4  aImCorB(aSzMod.x,aSzMod.y,0.0);
		REAL4 ** aCorR = aImCorR.data();
		REAL4 ** aCorG = aImCorG.data();
		REAL4 ** aCorB = aImCorB.data();
		cout<<vectPtsRadioTie[i].size()<<endl;
		//For each point of the surface, compute correction value (distance-ponderated mean value of all the tie points)
		for (int aY=0 ; aY<aSzMod.y  ; aY++)
			{
				for (int aX=0 ; aX<aSzMod.x  ; aX++)
				{
					double aSumDist=0;
					Pt2dr aPt; aPt.x=aX ; aPt.y=aY;
					for(int j = 0; j<int(vectPtsRadioTie[i].size()) ; j++){//go through each tie point
						if(vectPtsRadioTie[i].kR[j]>5||vectPtsRadioTie[i].kG[j]>5||vectPtsRadioTie[i].kB[j]>5 || vectPtsRadioTie[i].kR[j]<0.2||vectPtsRadioTie[i].kG[j]<0.2||vectPtsRadioTie[i].kB[j]<0.2){continue;}
						Pt2dr aPtIn; aPtIn.x=vectPtsRadioTie[i].Pos[j].x/ResolModel; aPtIn.y=vectPtsRadioTie[i].Pos[j].y/ResolModel;
						double aDist=Dist2d(aPtIn, aPt);
						if(aDist<1){aDist=1;}
						aSumDist=aSumDist+1/(aDist);//*vectPtsRadioTie[i].multiplicity[j]);
						aCorR[aY][aX] = aCorR[aY][aX] + vectPtsRadioTie[i].kR[j]/(aDist);//*vectPtsRadioTie[i].multiplicity[j]);
						aCorG[aY][aX] = aCorG[aY][aX] + vectPtsRadioTie[i].kG[j]/(aDist);//*vectPtsRadioTie[i].multiplicity[j]);
						aCorB[aY][aX] = aCorB[aY][aX] + vectPtsRadioTie[i].kB[j]/(aDist);//*vectPtsRadioTie[i].multiplicity[j]);						
					}
					//Normalize
					aCorR[aY][aX] = aCorR[aY][aX]/aSumDist;
					aCorG[aY][aX] = aCorG[aY][aX]/aSumDist; 
					aCorB[aY][aX] = aCorB[aY][aX]/aSumDist; 
				}
			}
		
		cout<<"Correction field computed, applying..."<<endl;

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
					Pt2dr aPt; aPt.x=double(aX/ResolModel); aPt.y=double(aY/ResolModel);
					//To be able to correct the edges
						if(aPt.x>aSzMod.x-2){aPt.x=aSzMod.x-2;}
						if(aPt.y>aSzMod.y-2){aPt.y=aSzMod.y-2;}
					//Bilinear interpolation from the scaled surface to the full scale image
					double R = aDataR[aY][aX]*Reechantillonnage::biline(aCorR, aSzMod.x, aSzMod.y, aPt);
					double G = aDataG[aY][aX]*Reechantillonnage::biline(aCorG, aSzMod.x, aSzMod.y, aPt);
					double B = aDataB[aY][aX]*Reechantillonnage::biline(aCorB, aSzMod.x, aSzMod.y, aPt);
					//Overrun management:
					if(R>255){aDataR[aY][aX]=255;}else if(R<0){aDataR[aY][aX]=0;}else{aDataR[aY][aX]=R;}
					if(G>255){aDataG[aY][aX]=255;}else if(G<0){aDataG[aY][aX]=0;}else{aDataG[aY][aX]=G;}
					if(B>255){aDataB[aY][aX]=255;}else if(B<0){aDataB[aY][aX]=0;}else{aDataB[aY][aX]=B;}
				}
		}
		//Writing ouput image
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

	std::string aFullPattern,aDirOut="Egal/",aMaster="",InVig="";
    bool InTxt=false;
	int ResolModel=16;
	double TPA=16;
	  //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern"),
            LArgMain()  << EAM(aDirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
						<< EAM(InVig,"InVig",true,"Input vignette folder (for example : Vignette/ )")
						<< EAM(ResolModel,"ResolModel",true,"Resol of input model (Def=16)")
						<< EAM(TPA,"TPA",true,"Tie Point Accuracy (Higher is better, lower gives more points Def=16)")
        );
		std::string aDir,aPatIm;
		SplitDirAndFile(aDir,aPatIm,aFullPattern);

		std::string Extension = "dat";
		if (InTxt){Extension="txt";}

		cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
		const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

		std::vector<std::string> aVectIm=*aSetIm;
		int nbIm=aVectIm.size();

		//Reading homologous points
		vector<PtsHom> aVectPtsHomol=ReadPtsHom3D(aDir, aPatIm, Extension, InVig, ResolModel, TPA);
		
		//Computing and applying the equalization surface
		Egal_field_correct(aDir, & aVectIm, aVectPtsHomol, aDirOut, InVig, ResolModel, nbIm);
	
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

