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
    std::cout <<  " *     E-qualization              *\n";
    std::cout <<  " *     N-ormalization             *\n";
    std::cout <<  " *     I-nter-images              *\n";
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
	//If a vignette correction is entered
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

		string arr[] = {"NaN", "7" ,  "6" , "NaN" , "5" , "NaN" , "NaN" , "NaN" , "4", "NaN" , "NaN" , "NaN" , "NaN", "NaN" , "NaN" , "NaN" , "3"};
		vector<string> numZoom(arr, arr+17);
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

double DistBetween(Pt3d<double> aP1, Pt3d<double> aP2 ){
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
			aDataR[int(aPtsHomol.Y1[i]/ResolModel)][int(aPtsHomol.X1[i]/ResolModel)]=255;
			aDataG[int(aPtsHomol.Y1[i]/ResolModel)][int(aPtsHomol.X1[i]/ResolModel)]=255;
			aDataB[int(aPtsHomol.Y1[i]/ResolModel)][int(aPtsHomol.X1[i]/ResolModel)]=255;
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

vector<PtsHom> ReadPtsHom3D(string aDir,string aPatIm,string Extension, string InVig, int ResolModel)
{
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);
	std::vector<std::string> aVectIm=*aSetIm;
	int nbIm=aVectIm.size();
	vector<PtsHom> aVectPtsHomol(nbIm*nbIm);
	//vector<int> NbPtsCoupleInit(nbIm*nbIm,0);
	//aPtsHomol.NbPtsCouple=NbPtsCoupleInit;

	//Loading all images
	vector<ArsenicImage> aGrIm=LoadGrpImages(aDir, aPatIm, ResolModel, InVig);
	std::cout<<"===== "<<aGrIm.size()<< " images loaded"<<endl;


	//On parcours toutes les paires d'images différentes (->testé dans le if)
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
											//Compute Distance between the 2 3D points to check if they are the same ones (occlusion beware!)
											distances[aK2]=DistBetween(pos3DPtIm1,pos3DPtIm2);
										}
									}
							}else{aVectPtsHomol[aK1].SZ=aGrIm[aK1].SZ;}
						}
						for (int aK2=0 ; aK2<int(nbIm) ; aK2++){
							if(distances[aK2]<(aGrIm[aK1].info3D->ResolSolOfPt(pos3DPtIm1))/16){//id pos3DPtIm1~=pos3DPtIm2 -->pt is considered homologous,it is added to PtsHom (Gr1, R1, G1, B1, X1, Y1, idem 2, NbPtsCouple++)
								aVectPtsHomol[(aK1*nbIm)+aK2].X1.push_back(ResolModel*pos2DPtIm1.x) ;
								aVectPtsHomol[(aK1*nbIm)+aK2].Y1.push_back(ResolModel*pos2DPtIm1.y) ;
								aVectPtsHomol[(aK1*nbIm)+aK2].X2.push_back(ResolModel*pos2DOtherIm[aK2].x) ;
								aVectPtsHomol[(aK1*nbIm)+aK2].Y2.push_back(ResolModel*pos2DOtherIm[aK2].y) ;
								//Go looking for grey value of the point for each chan
								double Red1   =Reechantillonnage::biline(aGrIm[aK1].RChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
								double Green1 =Reechantillonnage::biline(aGrIm[aK1].GChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
								double Blue1  =Reechantillonnage::biline(aGrIm[aK1].BChan.data(), aGrIm[aK1].SZ.x, aGrIm[aK1].SZ.y, pos2DPtIm1);
								double Red2   =Reechantillonnage::biline(aGrIm[aK2].RChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
								double Green2 =Reechantillonnage::biline(aGrIm[aK2].GChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
								double Blue2  =Reechantillonnage::biline(aGrIm[aK2].BChan.data(), aGrIm[aK2].SZ.x, aGrIm[aK2].SZ.y, pos2DOtherIm[aK2]);
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

vector<PtsHom> ReadPtsHom(string aDir,std::vector<std::string> * aSetIm,string Extension, bool useMasq, string InVig)
{
	vector<PtsHom> aVectPtsHomol;
	Pt2di aSz;
	//REAL4 ** aDataV1;
	//REAL4 ** aDataV2;

    // Permet de manipuler les ensemble de nom de fichier
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

//On parcours toutes les paires d'images différentes (->testé dans le if)
    for (int aK1=0 ; aK1<int(aSetIm->size()) ; aK1++)
    {
		/*
		//Read InVig
		if (InVig!=""){
			const cMetaDataPhoto & infoIm = cMetaDataPhoto::CreateExiv2(aDir + (*aSetIm)[aK1]);
			char foc[5],dia[4];
			sprintf(foc, "%04d", int(infoIm.FocMm()));
			sprintf(dia, "%03d", int(infoIm.Diaph()));
			string aNameVignette=InVig + "Foc" + (string)foc + "Diaph" + (string)dia + ".tif";
			std::ofstream file_out(aNameVignette.c_str());
			if(!file_out){
				cout<<"Couldn't find vignette tif file for "<<(*aSetIm)[aK1]<<" (Foc = "<<foc<<" Diaph = "<<dia<<endl;
			}else{
			//Reading the vignette
			Tiff_Im aTFV1= Tiff_Im::StdConvGen(aNameVignette,1,false);
			aSz = aTFV1.sz();
			Im2D_REAL4  aImV1(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTFV1.all_pts(),
				   aTFV1.in(),
				   aImV1.out()
				);

			aDataV1 = aImV1.data();
			}
		}
*/
		std::cout<<"Getting homologous points from: "<<(*aSetIm)[aK1]<<endl;
		//Reading the image and creating the objects to be manipulated
			Tiff_Im aTF1= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK1],3,false);
			aSz = aTF1.sz();
			Im2D_REAL4  aIm1R(aSz.x,aSz.y);
			Im2D_REAL4  aIm1G(aSz.x,aSz.y);
			Im2D_REAL4  aIm1B(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF1.all_pts(),
				   aTF1.in(),
				   Virgule(aIm1R.out(),aIm1G.out(),aIm1B.out())
				);

			REAL4 ** aDataR1 = aIm1R.data();
			REAL4 ** aDataG1 = aIm1G.data();
			REAL4 ** aDataB1 = aIm1B.data();

		//read masq if activeted
		Im2D_U_INT1  aMasq(aSz.x,aSz.y);
        unsigned char ** aMasqData = NULL;
		if(useMasq){
			Tiff_Im aTFMasq= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK1] + "_Masq.tif",1,false);
			ELISE_COPY
				(
				   aTFMasq.all_pts(),
				   aTFMasq.in(),
				   aMasq.out()
				);

			aMasqData = aMasq.data();
			}


        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
			if (aK1!=aK2)
            {
			PtsHom aPtsHomol;
			Tiff_Im aTF2= Tiff_Im::StdConvGen(aDir + (*aSetIm)[aK2],3,false);
			Im2D_REAL4  aIm2R(aSz.x,aSz.y);
			Im2D_REAL4  aIm2G(aSz.x,aSz.y);
			Im2D_REAL4  aIm2B(aSz.x,aSz.y);
			ELISE_COPY
				(
				   aTF2.all_pts(),
				   aTF2.in(),
				   Virgule(aIm2R.out(),aIm2G.out(),aIm2B.out())
				);

			REAL4 ** aDataR2 = aIm2R.data();
			REAL4 ** aDataG2 = aIm2G.data();
			REAL4 ** aDataB2 = aIm2B.data();

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
								if(useMasq){
									if(Reechantillonnage::biline(aMasqData, aSz.x, aSz.y, itP->P1())<0.2){continue;}
								}
								cpt++;
								//Go looking for grey value of the point for each chan
								double Red1   =Reechantillonnage::biline(aDataR1, aSz.x, aSz.y, itP->P1());
								double Green1 =Reechantillonnage::biline(aDataG1, aSz.x, aSz.y, itP->P1());
								double Blue1  =Reechantillonnage::biline(aDataB1, aSz.x, aSz.y, itP->P1());
								double Red2   =Reechantillonnage::biline(aDataR2, aSz.x, aSz.y, itP->P2());
								double Green2 =Reechantillonnage::biline(aDataG2, aSz.x, aSz.y, itP->P2());
								double Blue2  =Reechantillonnage::biline(aDataB2, aSz.x, aSz.y, itP->P2());
							aPtsHomol.Gr1.push_back((Red1+Green1+Blue1)/3);
							aPtsHomol.Gr2.push_back((Red2+Green2+Blue2)/3);
							aPtsHomol.R1.push_back(Red1);
							aPtsHomol.G1.push_back(Green1);
							aPtsHomol.B1.push_back(Blue1);
							aPtsHomol.R2.push_back(Red2);
							aPtsHomol.G2.push_back(Green2);
							aPtsHomol.B2.push_back(Blue2);
							aPtsHomol.X1.push_back(itP->P1().x);
							aPtsHomol.Y1.push_back(itP->P1().y);
							aPtsHomol.X2.push_back(itP->P2().x);
							aPtsHomol.Y2.push_back(itP->P2().y);
							}
							aPtsHomol.NbPtsCouple=cpt;
				   }
                   else{
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
					  aPtsHomol.NbPtsCouple=0;
				   }
				   aVectPtsHomol.push_back(aPtsHomol);
            }
        }
    }
   return aVectPtsHomol;
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

Param3Chan SolveAndArrange(L2SysSurResol aSysR,L2SysSurResol aSysG,L2SysSurResol aSysB, int nbIm, int nbParam){
	
Param3Chan aParam3Chan;

bool Ok1,Ok2,Ok3;
Im1D_REAL8 aSolR = aSysR.GSSR_Solve(&Ok1);
Im1D_REAL8 aSolG = aSysG.GSSR_Solve(&Ok2);
Im1D_REAL8 aSolB = aSysB.GSSR_Solve(&Ok3);

if (Ok1 && Ok2 && Ok3)
{
    double* aDataR = aSolR.data();
    double* aDataG = aSolG.data();
    double* aDataB = aSolB.data();
	//cout<<aSysRInit.ResiduOfSol(aDataR)<<endl;
	//cout<<aSysGInit.ResiduOfSol(aDataG)<<endl;
	//cout<<aSysBInit.ResiduOfSol(aDataB)<<endl;
    for(int i=0;i<int(nbIm*nbParam);i++){
		//cout<<"For im NUM "<<i<<" CorR = "<<aDataR[i]<<" CorG = "<<aDataG[i]<<" CorB = "<<aDataB[i]<<endl;
		aParam3Chan.parRed.push_back(aDataR[i]);
		aParam3Chan.parGreen.push_back(aDataG[i]);
		aParam3Chan.parBlue.push_back(aDataB[i]);
	}
	if (nbParam==1){
		//Normalize the result :
		double maxFactorR=1/(*max_element(aParam3Chan.parRed.begin(),aParam3Chan.parRed.end()));
		double maxFactorG=1/(*max_element(aParam3Chan.parGreen.begin(),aParam3Chan.parGreen.end()));
		double maxFactorB=1/(*max_element(aParam3Chan.parBlue.begin(),aParam3Chan.parBlue.end()));
        for(int i=0;i<int(nbIm);i++){
			if(maxFactorR>1){aParam3Chan.parRed[i]  =aParam3Chan.parRed[i]  *maxFactorR;}
			if(maxFactorG>1){aParam3Chan.parGreen[i]=aParam3Chan.parGreen[i]*maxFactorG;}
			if(maxFactorB>1){aParam3Chan.parBlue[i] =aParam3Chan.parBlue[i] *maxFactorB;}
		}
	}
}

	return aParam3Chan;
}

double ScoreRANSAC(Param3Chan aParam3Chan, vector<PtsHom> aVectPtsHomol, int nbIm)
{
	double error=0;
	vector<double> vErrorCouple;
	int nbParam=aParam3Chan.size()/nbIm;
	int degPoly=(nbParam-1)/(2);
	double nbPts=0, nbBlack=0;
	for(int indCouple=0 ; indCouple<int(aVectPtsHomol.size()) ; indCouple++)
	{
		double errorCouple=0;
		int numImage1=(indCouple/(nbIm));
		int numImage2=indCouple-numImage1*(nbIm);

		for(int indPt=0 ; indPt<int(aVectPtsHomol[indCouple].NbPtsCouple) ; indPt++)
		{
			nbPts++;
			//cout<<"indPt : "<<indPt<< " X1 = "<<aVectPtsHomol[indCouple].X1[indPt]<< " Y1 = "<<aVectPtsHomol[indCouple].Y1[indPt]<< " X2 = "<<aVectPtsHomol[indCouple].X2[indPt]<< " Y2 = "<<aVectPtsHomol[indCouple].Y2[indPt]<<endl;
			double corR1=aParam3Chan.parRed[nbParam*numImage1],corG1=aParam3Chan.parGreen[nbParam*numImage1],corB1=aParam3Chan.parBlue[nbParam*numImage1];
			double corR2=aParam3Chan.parRed[nbParam*numImage2],corG2=aParam3Chan.parGreen[nbParam*numImage2],corB2=aParam3Chan.parBlue[nbParam*numImage2];
			for	(int j=1 ; j<int(degPoly) ; j++)															 
			{
				corR1 = corR1 + pow(float(aVectPtsHomol[indCouple].X1[indPt]),j) * aParam3Chan.parRed[2*j-1+nbParam*numImage1]   + pow(float(aVectPtsHomol[indCouple].Y1[indPt]),j) * aParam3Chan.parRed[2*j+nbParam*numImage1] ;
				corG1 = corG1 + pow(float(aVectPtsHomol[indCouple].X1[indPt]),j) * aParam3Chan.parGreen[2*j-1+nbParam*numImage1] + pow(float(aVectPtsHomol[indCouple].Y1[indPt]),j) * aParam3Chan.parGreen[2*j+nbParam*numImage1] ;
				corB1 = corB1 + pow(float(aVectPtsHomol[indCouple].X1[indPt]),j) * aParam3Chan.parBlue[2*j-1+nbParam*numImage1]  + pow(float(aVectPtsHomol[indCouple].Y1[indPt]),j) * aParam3Chan.parBlue[2*j+nbParam*numImage1] ;
				corR2 = corR2 + pow(float(aVectPtsHomol[indCouple].X2[indPt]),j) * aParam3Chan.parRed[2*j-1+nbParam*numImage2]   + pow(float(aVectPtsHomol[indCouple].Y2[indPt]),j) * aParam3Chan.parRed[2*j+nbParam*numImage2] ;
				corG2 = corG2 + pow(float(aVectPtsHomol[indCouple].X2[indPt]),j) * aParam3Chan.parGreen[2*j-1+nbParam*numImage2] + pow(float(aVectPtsHomol[indCouple].Y2[indPt]),j) * aParam3Chan.parGreen[2*j+nbParam*numImage2] ;
				corB2 = corB2 + pow(float(aVectPtsHomol[indCouple].X2[indPt]),j) * aParam3Chan.parBlue[2*j-1+nbParam*numImage2]  + pow(float(aVectPtsHomol[indCouple].Y2[indPt]),j) * aParam3Chan.parBlue[2*j+nbParam*numImage2] ;
			}		

		double G1Poly1R=corR1*aVectPtsHomol[indCouple].R1[indPt];double G1Poly1G=corG1*aVectPtsHomol[indCouple].G1[indPt]; double G1Poly1B=corB1*aVectPtsHomol[indCouple].B1[indPt];
		double G2Poly2R=corR2*aVectPtsHomol[indCouple].R2[indPt];double G2Poly2G=corG2*aVectPtsHomol[indCouple].G2[indPt]; double G2Poly2B=corB2*aVectPtsHomol[indCouple].B2[indPt];
		
		//cout<<"G1Poly1 = "<<G1Poly1<<" G2Poly2 = "<<G2Poly2<<endl;

		if(G1Poly1R>255){G1Poly1R=255;}if(G1Poly1G>255){G1Poly1G=255;}if(G1Poly1B>255){G1Poly1B=255;}
		if(G2Poly2R>255){G2Poly2R=255;}if(G2Poly2G>255){G2Poly2G=255;}if(G2Poly2B>255){G2Poly2B=255;}
		if(G1Poly1R<0){G1Poly1R=0;}if(G1Poly1G<0){G1Poly1G=0;}if(G1Poly1B<0){G1Poly1B=0;}
		if(G2Poly2R<0){G2Poly2R=0;}if(G2Poly2G<0){G2Poly2G=0;}if(G2Poly2B<0){G2Poly2B=0;}
        if((G1Poly1R==0 && G2Poly2R==0) || (G1Poly1G==0 && G2Poly2G==0) || (G1Poly1B==0 && G2Poly2B==0) ||
                (G1Poly1R==255 && G2Poly2R==255) || (G1Poly1G==255 && G2Poly2G==255) || (G1Poly1B==255 && G2Poly2B==255) ){nbBlack++;error=error+1000;}
		error=error+fabs(G1Poly1R-G2Poly2R)+fabs(G1Poly1G-G2Poly2G)+fabs(G1Poly1B-G2Poly2B);
		errorCouple=errorCouple+fabs(G1Poly1R-G2Poly2R)+fabs(G1Poly1G-G2Poly2G)+fabs(G1Poly1B-G2Poly2B);
	
		}
		vErrorCouple.push_back(errorCouple);
	}
	//cout<<"Error = "<<error<<endl;
	double ratioBlack=double(nbBlack/nbPts);
	if(ratioBlack==0){ratioBlack=0.01;}
	//if(double(nbBlack/nbPts)>0.5){return -1;cout<<double(nbBlack/nbPts)<<endl;}else{
	ELISE_ASSERT(error!=0,"Error=0, something is wrong");
	//cout<<"Error per couple : " <<vErrorCouple<<endl;
	return (3*nbPts)/(error);//*ratioBlack);//}
		
}

Param3Chan Egalisation_factors(vector<PtsHom> aVectPtsHomol, int nbIm, int aMasterNum, int aDegPoly, bool useRANSAC)
{
	vector<vector<double> > Gr(nbIm);
	int aNbCouples=aVectPtsHomol.size();
// Create L2SysSurResol to solve least square equation with nbIm unknown
	L2SysSurResol aSysRInit(nbIm);
	L2SysSurResol aSysGInit(nbIm);
	L2SysSurResol aSysBInit(nbIm);

	
//Finding and Selecting the brightest image for reference
int nbParcouru=0;
for (int i=0;i<int(aNbCouples);i++){

	int numImage1=(i/(nbIm));
	int numImage2=i-numImage1*(nbIm);

if (numImage1!=numImage2){
	if(aVectPtsHomol[i].NbPtsCouple!=0){//if there are homologous points between images

		nbParcouru=nbParcouru+aVectPtsHomol[i].NbPtsCouple;
							  
		std::copy (aVectPtsHomol[i].Gr1.begin(),aVectPtsHomol[i].Gr1.end(),back_inserter(Gr[numImage1]));
		std::copy (aVectPtsHomol[i].Gr2.begin(),aVectPtsHomol[i].Gr2.end(),back_inserter(Gr[numImage2]));
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
	aSysRInit.AddEquation(pow(float(nbParcouru),2),coefsFixeAr,255/grMax);
	aSysGInit.AddEquation(pow(float(nbParcouru),2),coefsFixeAr,255/grMax);
	aSysBInit.AddEquation(pow(float(nbParcouru),2),coefsFixeAr,255/grMax);
}

//The brightest image is fixed, using superbig weight in least square:
double grMax=*max_element(Gr[aMasterNum].begin(),Gr[aMasterNum].end());
cout<<grMax<<endl;
vector<double> aCoefsFixe(nbIm,0.0);
aCoefsFixe[aMasterNum]=1;
double * coefsFixeAr=&aCoefsFixe[0];
aSysRInit.AddEquation(pow(float(nbParcouru),5),coefsFixeAr,255/grMax);
aSysGInit.AddEquation(pow(float(nbParcouru),5),coefsFixeAr,255/grMax);
aSysBInit.AddEquation(pow(float(nbParcouru),5),coefsFixeAr,255/grMax);

cout<<"Solution of zeros preventing equations written"<<endl;


//For each linked couples :
for (int i=0;i<int(aNbCouples);i++){

	int numImage1=(i/(nbIm));

	int numImage2=i-numImage1*(nbIm);
if (numImage1!=numImage2){
	if(aVectPtsHomol[i].NbPtsCouple!=0){//if there are homologous points between images
		
		//adding equations for each point 
		for (int j=0;j<int(aVectPtsHomol[i].NbPtsCouple);j++){
			vector<double> aCoefsR(nbIm, 0.0);
			vector<double> aCoefsG(nbIm, 0.0);
			vector<double> aCoefsB(nbIm, 0.0);
			aCoefsR[numImage1]=aVectPtsHomol[i].R1[j];
			aCoefsG[numImage1]=aVectPtsHomol[i].G1[j];
			aCoefsB[numImage1]=aVectPtsHomol[i].B1[j];
			aCoefsR[numImage2]=-aVectPtsHomol[i].R2[j];
			aCoefsG[numImage2]=-aVectPtsHomol[i].G2[j];
			aCoefsB[numImage2]=-aVectPtsHomol[i].B2[j];
			double * coefsArR=&aCoefsR[0];
			double * coefsArG=&aCoefsG[0];
			double * coefsArB=&aCoefsB[0];
			double aPds=1;
			if(numImage1==aMasterNum || numImage2==aMasterNum){aPds=nbParcouru/nbIm;}
			aSysRInit.AddEquation(aPds,coefsArR,0);
			aSysGInit.AddEquation(aPds,coefsArG,0);
			aSysBInit.AddEquation(aPds,coefsArB,0);
		}
	}
}
}
cout<<nbParcouru<<" points were read"<<endl;
cout<<"Solving the initial system"<<endl;

Param3Chan aParam3ChanInit=SolveAndArrange(aSysRInit, aSysGInit, aSysBInit, nbIm, 1);

cout<<aParam3ChanInit.parRed<<endl;
cout<<aParam3ChanInit.parGreen<<endl;
cout<<aParam3ChanInit.parBlue<<endl;


/*****************************************************************************************************/
cout<<"Introducing more parameters (model is : G1*poly(X1) + G1*poly(Y1) - G2*poly(X2) - G2*poly(Y2) = 0 )"<<endl;
/*****************************************************************************************************/
int nbParam=aDegPoly*2+1;//nb param in the model
Param3Chan aParam3Chan;
double aScoreMax=0;
int nbRANSACMax=10000;
int subsetSize=2*nbParam;
if(!useRANSAC){nbRANSACMax=1;}
srand(time(NULL));//Initiate the rand value
for(int nbRANSAC=0 ; nbRANSAC<int(nbRANSACMax) ; nbRANSAC++){
	if(nbRANSAC % 500==0 && useRANSAC){cout<<"RANSAC progress : "<<nbRANSAC/100<<" %"<<endl;}
	// Create L2SysSurResol to solve least square equation with nbParam*nbIm unknown
		L2SysSurResol aSysR(nbParam*nbIm);
		L2SysSurResol aSysG(nbParam*nbIm);
		L2SysSurResol aSysB(nbParam*nbIm);


	for(int i=0;i<int(nbIm);i++){
        //double grMax=*max_element(Gr[i].begin(),Gr[i].end());
		vector<double> aCoefsFixe(nbParam*nbIm,0.0);
		aCoefsFixe[nbParam*i]=1;
		double * coefsFixeAr=&aCoefsFixe[0];
		aSysR.AddEquation(pow(float(nbParam*nbIm),6),coefsFixeAr,aParam3ChanInit.parRed[i]);
		aSysG.AddEquation(pow(float(nbParam*nbIm),6),coefsFixeAr,aParam3ChanInit.parGreen[i]);
		aSysB.AddEquation(pow(float(nbParam*nbIm),6),coefsFixeAr,aParam3ChanInit.parBlue[i]);
		for (int a=1;a<nbParam;a++){		 
			vector<double> aCoefsFixe(nbParam*nbIm,0.0);
			aCoefsFixe[nbParam*i+a]=1;
			double * coefsFixeAr=&aCoefsFixe[0];
			aSysR.AddEquation(0.001,coefsFixeAr,0);
			aSysG.AddEquation(0.001,coefsFixeAr,0);
			aSysB.AddEquation(0.001,coefsFixeAr,0);
		}
	}

																	/*
		//The brightest image is fixed, using superbig weight in least square:
	vector<double> aCoefsFixe2(nbParam*nbIm,0.0);
	aCoefsFixe2[nbParam*aMasterNum]=1;
	double * coefsFixe2Ar=&aCoefsFixe2[0];
	aSysR.AddEquation(pow(float(nbParcouru),3),coefsFixe2Ar,255/grMax);
	aSysG.AddEquation(pow(float(nbParcouru),3),coefsFixe2Ar,255/grMax);
	aSysB.AddEquation(pow(float(nbParcouru),3),coefsFixe2Ar,255/grMax);
	for (int a=1;a<nbParam;a++){
		vector<double> aCoefsFixe(nbParam*nbIm,0.0);
		aCoefsFixe[nbParam*aMasterNum+a]=1;
		double * coefsFixeAr=&aCoefsFixe[0];
		aSysR.AddEquation(pow(float(nbParcouru),4),coefsFixeAr,0);
		aSysG.AddEquation(pow(float(nbParcouru),4),coefsFixeAr,0);
		aSysB.AddEquation(pow(float(nbParcouru),4),coefsFixeAr,0);
	}
*/

	//For each linked couples :
	for (int i=0;i<int(aNbCouples);i++){	

		int numImage1=(i/(nbIm));
		int numImage2=i-numImage1*(nbIm);
		
		if (numImage1!=numImage2){
			if(aVectPtsHomol[i].NbPtsCouple!=0){//if there are homologous points between images																	 
				//adding equations for each point 
				if(!useRANSAC){subsetSize=aVectPtsHomol[i].size();}
				for (int cpt=0;cpt<int(subsetSize);cpt++){
					int j;
					if(!useRANSAC){j=cpt;}else{j=rand() % aVectPtsHomol[i].NbPtsCouple;}

					vector<double> aCoefsR(nbParam*nbIm, 0.0);
					vector<double> aCoefsG(nbParam*nbIm, 0.0);
					vector<double> aCoefsB(nbParam*nbIm, 0.0);
					aCoefsR[numImage1]=aVectPtsHomol[i].R1[j]; 
					aCoefsG[numImage1]=aVectPtsHomol[i].G1[j]; 
					aCoefsB[numImage1]=aVectPtsHomol[i].B1[j]; 
					aCoefsR[numImage2]=-aVectPtsHomol[i].R2[j];
					aCoefsG[numImage2]=-aVectPtsHomol[i].G2[j];
					aCoefsB[numImage2]=-aVectPtsHomol[i].B2[j];
					for(int k=1;k<=((nbParam-1)/2);k++){
						aCoefsR[nbParam*numImage1+2*k-1]=aVectPtsHomol[i].R1[j]*pow(aVectPtsHomol[i].X1[j],k);
						aCoefsG[nbParam*numImage1+2*k-1]=aVectPtsHomol[i].G1[j]*pow(aVectPtsHomol[i].X1[j],k);
						aCoefsB[nbParam*numImage1+2*k-1]=aVectPtsHomol[i].B1[j]*pow(aVectPtsHomol[i].X1[j],k);
						aCoefsR[nbParam*numImage1+2*k]=aVectPtsHomol[i].R1[j]*pow(aVectPtsHomol[i].Y1[j],k);
						aCoefsG[nbParam*numImage1+2*k]=aVectPtsHomol[i].G1[j]*pow(aVectPtsHomol[i].Y1[j],k);
						aCoefsB[nbParam*numImage1+2*k]=aVectPtsHomol[i].B1[j]*pow(aVectPtsHomol[i].Y1[j],k);

						aCoefsR[nbParam*numImage1+2*k-1]=aVectPtsHomol[i].R2[j]*pow(aVectPtsHomol[i].X2[j],k);
						aCoefsG[nbParam*numImage1+2*k-1]=aVectPtsHomol[i].G2[j]*pow(aVectPtsHomol[i].X2[j],k);
						aCoefsB[nbParam*numImage1+2*k-1]=aVectPtsHomol[i].B2[j]*pow(aVectPtsHomol[i].X2[j],k);
						aCoefsR[nbParam*numImage1+2*k]=aVectPtsHomol[i].R2[j]*pow(aVectPtsHomol[i].Y2[j],k);
						aCoefsG[nbParam*numImage1+2*k]=aVectPtsHomol[i].G2[j]*pow(aVectPtsHomol[i].Y2[j],k);
						aCoefsB[nbParam*numImage1+2*k]=aVectPtsHomol[i].B2[j]*pow(aVectPtsHomol[i].Y2[j],k);
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

	Param3Chan aParam3ChanRANSAC=SolveAndArrange(aSysR, aSysG, aSysB, nbIm, nbParam);
	//cout<<aParam3ChanRANSAC.parRed<<endl;
	//cout<<aParam3ChanRANSAC.parGreen<<endl;
	//cout<<aParam3ChanRANSAC.parBlue<<endl;
	//aParam3Chan=aParam3ChanRANSAC;

	double aScore=ScoreRANSAC(aParam3ChanRANSAC, aVectPtsHomol, nbIm);//cout<<aScore<<endl;
	if(aScore==-1){nbRANSACMax++;if(nbRANSACMax % 100==0 ){cout<<"Total nb of iteration iteration increased to = "<<nbRANSACMax<<" at iteration "<<nbRANSAC<<endl;}}
	else if(aScoreMax<aScore){aScoreMax=aScore;aParam3Chan=aParam3ChanRANSAC;cout<<"New highest score : "<<aScoreMax<<" at iteration "<<nbRANSAC<<endl;}
}
	cout<<"Final RANSAC Score : "<<aScoreMax<<endl;
	cout<<aParam3Chan.parRed<<endl;
	cout<<aParam3Chan.parGreen<<endl;
	cout<<aParam3Chan.parBlue<<endl;

return aParam3Chan;
}

void Egal_field_correct(string aDir,std::vector<std::string> * aSetIm,vector<PtsHom> aVectPtsHomol, string aDirOut, string InVig, int ResolModel, int nbIm)
{

	int aNbCouples=aVectPtsHomol.size();
	vector<PtsRadioTie> vectPtsRadioTie(nbIm);
	int nbPts=0;
	//filling up the factors from homologous points
	for (int i=0;i<int(aNbCouples);i++){

		int numImage1=(i/(nbIm));
		int numImage2=i-numImage1*(nbIm);

		if (numImage1!=numImage2){
			for(int j=0; j<aVectPtsHomol[i].NbPtsCouple; j++){//if there are homologous points between images
				double kR=aVectPtsHomol[i].R2[j]/aVectPtsHomol[i].R1[j];
				double kG=aVectPtsHomol[i].G2[j]/aVectPtsHomol[i].G1[j];
				double kB=aVectPtsHomol[i].B2[j]/aVectPtsHomol[i].B1[j];
				vectPtsRadioTie[numImage1].kR.push_back(kR);
				vectPtsRadioTie[numImage1].kG.push_back(kG);
				vectPtsRadioTie[numImage1].kB.push_back(kB);
				Pt2dr aPoint; aPoint.x=aVectPtsHomol[i].X1[j]; aPoint.y=aVectPtsHomol[i].Y1[j];
				vectPtsRadioTie[numImage1].Pos.push_back(aPoint);
				nbPts++;
			}
		}
	}
	cout<<nbPts<<" loaded"<<endl;

	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
	//Reading input files
	string suffix="";if(InVig!=""){suffix="_Vodka.tif";}
    //long int cptBcl=0;
    for(int i=0;i<nbIm;i++)
	{
		
	    string aNameIm=InVig + (*aSetIm)[i] + suffix;
		cout<<"Correcting "<<aNameIm<<" (with "<<vectPtsRadioTie[i].size()<<" data points)"<<endl;
		string aNameOut=aDir + aDirOut + (*aSetIm)[i] +"_egal.tif";

		Pt2di aSzMod=aVectPtsHomol[i*nbIm].SZ;//cout<<aSzMod<<endl;
		Im2D_REAL4  aImCorR(aSzMod.x,aSzMod.y,0.0);
		Im2D_REAL4  aImCorG(aSzMod.x,aSzMod.y,0.0);
		Im2D_REAL4  aImCorB(aSzMod.x,aSzMod.y,0.0);
		REAL4 ** aCorR = aImCorR.data();
		REAL4 ** aCorG = aImCorG.data();
		REAL4 ** aCorB = aImCorB.data();
		
		for (int aY=0 ; aY<aSzMod.y  ; aY++)
			{
				for (int aX=0 ; aX<aSzMod.x  ; aX++)
				{
					double aSumDist=0;
					Pt2dr aPt; aPt.x=aX ; aPt.y=aY;
					for(int j = 0; j<int(vectPtsRadioTie[i].size()) ; j++){
						Pt2dr aPtIn; aPtIn.x=vectPtsRadioTie[i].Pos[j].x/ResolModel; aPtIn.y=vectPtsRadioTie[i].Pos[j].y/ResolModel;
						double aDist=Dist2d(aPtIn, aPt);
						if(aDist<1){aDist=1;}
						aSumDist=aSumDist+1/aDist;
						aCorR[aY][aX] = aCorR[aY][aX] + vectPtsRadioTie[i].kR[j]/aDist,2;
						aCorG[aY][aX] = aCorG[aY][aX] + vectPtsRadioTie[i].kG[j]/aDist,2;
						aCorB[aY][aX] = aCorB[aY][aX] + vectPtsRadioTie[i].kB[j]/aDist,2;						
					}
					
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
				//if((aY % int(aSz.y/4)) == 0){cout<<"Progress for this image : "<<double(aY)/double(aSz.y)*100<<"%"<<endl;}
				for (int aX=0 ; aX<aSz.x  ; aX++)
				{
					Pt2dr aPt; aPt.x=double(aX/ResolModel); aPt.y=double(aY/ResolModel);
					if(aPt.x>aSzMod.x-2){aPt.x=aSzMod.x-2;}
					if(aPt.y>aSzMod.y-2){aPt.y=aSzMod.y-2;}
					double R = aDataR[aY][aX]*Reechantillonnage::biline(aCorR, aSzMod.x, aSzMod.y, aPt);
					double G = aDataG[aY][aX]*Reechantillonnage::biline(aCorG, aSzMod.x, aSzMod.y, aPt);
					double B = aDataB[aY][aX]*Reechantillonnage::biline(aCorB, aSzMod.x, aSzMod.y, aPt);
					//Overrun management:
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

void Egal_correct(string aDir,std::vector<std::string> * aSetIm,Param3Chan  aParam3chan,string aDirOut, string InVig)
{
	//Bulding the output file system
    ELISE_fp::MkDirRec(aDir + aDirOut);
	//Reading input files
    int nbIm=(aSetIm)->size();
	int nbParam=aParam3chan.size()/nbIm;//nbParam par image
	string suffix="";if(InVig!=""){suffix="_Vodka.tif";}
    for(int i=0;i<nbIm;i++)
	{
	    string aNameIm=InVig + (*aSetIm)[i] + suffix;
		cout<<"Correcting "<<aNameIm<<endl;
		string aNameOut=aDir + aDirOut + (*aSetIm)[i] +"_egal.tif";

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

					double corR=aParam3chan.parRed[nbParam*i],corG=aParam3chan.parGreen[nbParam*i],corB=aParam3chan.parBlue[nbParam*i];
					for(int j=1;j<=(nbParam-1)/2;j++){
						corR = corR + pow(float(aX),j) *   aParam3chan.parRed[2*j-1+nbParam*i] + pow(float(aY),j) *   aParam3chan.parRed[2*j+nbParam*i] ;
						corG = corG + pow(float(aX),j) * aParam3chan.parGreen[2*j-1+nbParam*i] + pow(float(aY),j) * aParam3chan.parGreen[2*j+nbParam*i] ;
						corB = corB + pow(float(aX),j) *  aParam3chan.parBlue[2*j-1+nbParam*i] + pow(float(aY),j) *  aParam3chan.parBlue[2*j+nbParam*i] ;
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

	std::string aFullPattern,aDirOut="Egal/",aMaster="",InVig="";
    bool InTxt=false/*,useRANSAC=false*/;
    //int aDegPoly=3;
	int ResolModel=16;
	  //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern"),
            LArgMain()  << EAM(aDirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
						<< EAM(InVig,"InVig",true,"Input vignette folder")
						//<< EAM(InTxt,"InTxt",true,"True if homologous points have been exported in txt (Defaut=false)")
						//<< EAM(DoCor,"DoCor",true,"Use the computed parameters to correct the images (Defaut=false)")
						//<< EAM(aMaster,"Master",true,"Manually define a Master Image (to be used a reference)")
						//<< EAM(aDegPoly,"DegPoly",true,"Set the dergree of the corretion polynom (Def=3)")
						//<< EAM(useRANSAC,"useRANSAC",true,"Activate the use of RANSAC (Instead of Least Square)")
						<< EAM(ResolModel,"ResolModel",true,"Resol of input model (Def=16)")
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
        //int aMasterNum=-1;
		for (int i=0;i<int(nbIm);i++){
            if(aVectIm[i]==aMaster){/*aMasterNum=i*/;cout<<"Found Master image "<<aMaster<<" as image NUM "<<i<<endl;}
		}

		//Reading homologous points
		vector<PtsHom> aVectPtsHomol=ReadPtsHom3D(aDir, aPatIm, Extension, InVig, ResolModel);
		
		cout<<"Computing equalization factors"<<endl;
		//Param3Chan aParam3chan=Egalisation_factors(aVectPtsHomol,nbIm,aMasterNum,aDegPoly,useRANSAC);
		//CorrectionFields aCorFlds=Egalisation_fields(aVectPtsHomol, nbIm, ResolModel);
		/*if(aParam3chan.size()==0){
			cout<<"Couldn't compute parameters "<<endl;
		}else*/
			//Egal_correct(aDir, & aVectIm, aParam3chan, aDirOut, InVig);
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

