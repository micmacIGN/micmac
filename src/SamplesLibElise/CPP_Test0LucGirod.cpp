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
#include "StdAfx.h"
#include <algorithm>
#include "hassan/reechantillonnage.h"

// Reading Homologous point associated to a set of images


int  Luc_test_ptshom_main(int argc,char ** argv)
{
  //=====================  PARAMETRES EN DUR ==============

   std::string aDir = "C:/Users/Luc Girod/Desktop/TFE/Vignettage/vignette_sift3/";
   std::string aPatIm = ".*NEF";
   std::string Prefix = "";
   // std::string Prefix =  "_SRes" ; 
   std::string Extension = "dat";

  //===================== 

    // Permet de manipuler les ensemble de nom de fichier
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

//On parcours toutes les paires d'images différentes (->testé dans le if)
    for (int aK1=0 ; aK1<int(aSetIm->size()) ; aK1++)
    {
		cout<<(*aSetIm)[aK1]<<endl;
        for (int aK2=0 ; aK2<int(aSetIm->size()) ; aK2++)
        {
            if (aK1!=aK2)
            {
               std::string aNamePack =  aDir +  aICNM->Assoc1To2
                                        (
                                           "NKS-Assoc-CplIm2Hom@"+ Prefix + "@"+Extension,
                                           (*aSetIm)[aK1],
                                           (*aSetIm)[aK2],
                                           true
                                    );
             
               if (aK1==0) 
               {
                   bool Exist = ELISE_fp::exist_file(aNamePack);
                   if (Exist)
                   {
                      ElPackHomologue aPack = ElPackHomologue::FromFile(aNamePack);
                      if (aK2==1)
                      {
                           int aNb=0;
                           for 
                           (
                               ElPackHomologue::const_iterator itP=aPack.begin();
                               itP!=aPack.end();
                               itP++
                           )
                           {
                              if (aNb<10)
								  std::cout  << itP->P1() << itP->P2() <<"\n";
                              aNb++;
                           }
                      }
                      std::cout << aNamePack  << " " << aPack.size() << "\n";
                   }
                   else
                      std::cout  << "     # NO PACK FOR  : " << aNamePack  << "\n";
               }
            }
        }
    }


	return 0;
}

void RotateImage(double alpha, Pt2di aSzOut, vector<Pt2dr> Pts , string aNameDir, string aNameIm)
{
	cout<<"Rotating "<<aNameIm<<endl;
	string aNameOut=aNameDir + "Croped_images/" + aNameIm + ".tif";

	//Reading the image and creating the objects to be manipulated
    Tiff_Im aTF= Tiff_Im::StdConvGen(aNameDir + aNameIm,1,false);

	Pt2di aSz = aTF.sz();
	Pt2dr P1Cor=Rot2D(alpha, Pts[0], Pts[1]); //P1Cor.x=cos(alpha)*(Pts[0].x-Pts[1].x)+sin(alpha)*(Pts[0].y-Pts[1].y)+Pts[1].x; P1Cor.y=-sin(alpha)*(Pts[0].x-Pts[1].x)+cos(alpha)*(Pts[0].y-Pts[1].y)+Pts[1].y;
	Pt2dr P3Cor=Rot2D(alpha, Pts[2], Pts[1]); //P3Cor.x=cos(alpha)*(Pts[2].x-Pts[1].x)+sin(alpha)*(Pts[2].y-Pts[1].y)+Pts[1].x; P3Cor.y=-sin(alpha)*(Pts[2].x-Pts[1].x)+cos(alpha)*(Pts[2].y-Pts[1].y)+Pts[1].y;

    Im2D_U_INT1  aImR(aSz.x,aSz.y);
    //Im2D_U_INT1  aImG(aSz.x,aSz.y);
    //Im2D_U_INT1  aImB(aSz.x,aSz.y);
    Im2D_U_INT1  aImROut(aSzOut.x,aSzOut.y);
    Im2D_U_INT1  aImGOut(aSzOut.x,aSzOut.y);
    Im2D_U_INT1  aImBOut(aSzOut.x,aSzOut.y);

    ELISE_COPY
    (
       aTF.all_pts(),
       aTF.in(),
       aImR.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
    );

    U_INT1 ** aDataR = aImR.data();
    //U_INT1 ** aDataG = aImG.data();
    //U_INT1 ** aDataB = aImB.data();
    U_INT1 ** aDataROut = aImROut.data();
    U_INT1 ** aDataGOut = aImGOut.data();
    U_INT1 ** aDataBOut = aImBOut.data();

    //Parcours des points de l'image de sortie et remplissage des valeurs
    Pt2dr ptOut;
	Pt2di imageTopCorner, imageBottomCorner;
	imageTopCorner.x    = (int)(P1Cor.x + 0.5) + (max(abs(P1Cor.x-Pts[1].x),abs(P1Cor.x-P3Cor.x))-aSzOut.x)/2;
	imageTopCorner.y    = (int)(P1Cor.y + 0.5) + (max(abs(P3Cor.y-Pts[1].y),abs(P1Cor.y-P3Cor.y))-aSzOut.y)/2;
	imageBottomCorner.x = imageTopCorner.x + aSzOut.x;
	imageBottomCorner.y = imageTopCorner.y + aSzOut.y;

    for (int aY=imageTopCorner.y ; aY<imageBottomCorner.y  ; aY++)
    {
        for (int aX=imageTopCorner.x ; aX<imageBottomCorner.x  ; aX++)
		{
			ptOut.x=cos(-alpha)*(aX-Pts[1].x)+sin(-alpha)*(aY-Pts[1].y)+Pts[1].x;
			ptOut.y=-sin(-alpha)*(aX-Pts[1].x)+cos(-alpha)*(aY-Pts[1].y)+Pts[1].y;
			aDataROut[aY-imageTopCorner.y][aX-imageTopCorner.x] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
			aDataGOut[aY-imageTopCorner.y][aX-imageTopCorner.x] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
			aDataBOut[aY-imageTopCorner.y][aX-imageTopCorner.x] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
		}
    }

    Tiff_Im  aTOut
    (
         aNameOut.c_str(),
         aSzOut,
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
    string aCom="convert ephemeral:" + aNameDir + "Croped_images/" + aNameIm + ".tif " + aNameDir + "Croped_images/" + aNameIm;
	system_call(aCom.c_str());
    #endif
    #if (ELISE_windows)
		string aCom=MMDir() + "binaire-aux/convert ephemeral:" + aNameDir + "Croped_images/" + aNameIm + ".tif " + aNameDir + "Croped_images/" + aNameIm;
		system_call(aCom.c_str());
    #endif

}

int  Luc_main_corner_crop(int argc,char ** argv){

	std::string aFullPattern, cornersTxt;
	//Reading the arguments
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullPattern,"Images Pattern")
					<< EAMC(cornersTxt,"Corner txt File"),
        LArgMain()  
    );

	std::string aDir,aPatIm;
	SplitDirAndFile(aDir,aPatIm,aFullPattern);
	
	ELISE_fp::MkDirRec(aDir + "Croped_images/");

	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);

	std::vector<std::string> aVectIm=*aSetIm;
	int nbIm=aVectIm.size();

	vector<vector<Pt2dr> > Pts;
	vector<int> SzX, SzY;
    std::ifstream file(cornersTxt.c_str(), ios::in);
	for(int i=0 ; i<nbIm ; i++)
	{
		vector<Pt2dr> PtsIm(3);
		string name;
		file >> name >> PtsIm[0].x >> PtsIm[0].y >> name >> PtsIm[1].x >> PtsIm[1].y >> name >> PtsIm[2].x >> PtsIm[2].y;  
		Pts.push_back(PtsIm);
		SzX.push_back(euclid(PtsIm[0],PtsIm[1])); SzY.push_back(euclid(PtsIm[2],PtsIm[1]));
	}

	file.close();
	cout<<Pts<<endl;
	Pt2di aCrop; int border=10;
    aCrop.x=min(*min_element(SzX.begin(), SzX.end())-2*border,*min_element(SzY.begin(), SzY.end())-2*border);
	aCrop.y=aCrop.x;
	//aCrop.x=*min_element(std::begin(SzX), std::end(SzX))-2*border;
	//aCrop.y=*min_element(std::begin(SzY), std::end(SzY))-2*border;
	cout<<"Cropping to : "<<aCrop.x<<" "<<aCrop.y<<endl;

	for(int i=0 ; i<nbIm ; i++)
	{
		double alpha=(atan((Pts[i][0].y-Pts[i][1].y)/(Pts[i][0].x-Pts[i][1].x))+atan(-(Pts[i][2].x-Pts[i][1].x)/(Pts[i][2].y-Pts[i][1].y)))/2;
		cout<<"Alpha = "<<alpha<<endl;
		RotateImage(alpha, aCrop, Pts[i], aDir, aVectIm[i]);
	}

	//Pt2dr P1,P2,P3;
	//P1.x= 795 ; P1.y= 1064;
	//P2.x= 7401; P2.y= 926 ;
	//P3.x= 7518; P3.y= 7598;
	//cout<<(P1.y-P2.y)/(P1.x-P2.x)<<endl;
	//cout<<atan((P1.y-P2.y)/(P1.x-P2.x))<<endl;
	//cout<<(P3.x-P2.x)/(P3.y-P2.y)<<endl;
	//cout<<atan((P3.x-P2.x)/(P3.y-P2.y))<<endl;
	//double aT1=atan((P1.y-P2.y)/(P1.x-P2.x));
	//double aT2=atan(-(P3.x-P2.x)/(P3.y-P2.y));
	//cout<<aT1<<" + "<<aT2<< " = " <<(aT1+aT2)<<endl;


	return 0;
}

int Luc_main(int argc,char ** argv)
{
	//MMD_InitArgcArgv(argc,argv,3);

	std::string aFilePtsIn;
	//Reading the arguments
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFilePtsIn,"Input file"),
        LArgMain()  
    );

	std::string aFilePtsOut="GCP_xAligned.xml";

	std::ifstream file(aFilePtsIn.c_str(), ios::in);
	int nbIm;
	file >> nbIm;
	std::vector<Pt3dr> aVPts(nbIm);
    std::vector<Pt3dr> aVInc(nbIm);
    std::vector<std::string> aVName(nbIm,"");
	for(int i=0 ; i<nbIm ; i++)
	{
		string name;
		file >> aVName[i] >> aVPts[i].x >> aVPts[i].y >> aVPts[i].z >> aVInc[i].x >> aVInc[i].y >> aVInc[i].z;
	}

	file.close();
	//Least Square

	// Create L2SysSurResol to solve least square equation with 3 unknown
	L2SysSurResol aSys(2);

  	//For Each SIFT point
	double sumX=0, sumY=0;
	for(int i=0;i<int(aVPts.size());i++){
		double aPds[2]={aVPts[i].x,1};
		double poids=1;
		aSys.AddEquation(poids,aPds,aVPts[i].y);
		sumX=sumX+aVPts[i].x;
		sumY=sumY+aVPts[i].y;
	}

	Pt2dr aRotCenter; aRotCenter.x=sumX/aVPts.size();aRotCenter.y=sumY/aVPts.size();

    bool Ok;
    Im1D_REAL8 aSol = aSys.GSSR_Solve(&Ok);

	double aAngle;
    if (Ok)
    {
        double* aData = aSol.data();
		aAngle=atan(aData[0]);
		cout<<"Angle = "<<aAngle<<endl<<"Rot Center = "<<aRotCenter<<endl;
    
	for(int i=0;i<int(aVPts.size());i++){
		Pt2dr aPt; aPt.x=aVPts[i].x; aPt.y=aVPts[i].y;
		aPt=Rot2D(aAngle, aPt, aRotCenter);aVPts[i].x=aPt.x;aVPts[i].y=aPt.y;
	}
	}
	
//End Least Square

	cDicoAppuisFlottant  aDico;
    for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
    {
        cOneAppuisDAF aOAD;
        aOAD.Pt() = aVPts[aKP];
        aOAD.NamePt() = aVName[aKP];
        aOAD.Incertitude() = aVInc[aKP];

        aDico.OneAppuisDAF().push_back(aOAD);
    }

	
    MakeFileXML(aDico,aFilePtsOut);

	return 0;
}
#if (0)
#endif




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
