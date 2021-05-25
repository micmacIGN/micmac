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
#include "StdAfx.h"


/**
 * Zlimit: crop a DEM in depth
 * */

//
//----------------------------------------------------------------------------

int Zlimit_main(int argc,char ** argv)
{
    std::string aNameOriDEM;
    std::string aMasqSup="?";
    std::string aCorrelIm="?";
    std::string aOriginIm="?";
    int aOriginImMinVal=300;
	double aMaxZ, aMinZ, aCorThr=0.02;
	bool aBoolDEM=true;

    cout<<endl<<"Zlimit: create mask for depth image and resulting masked depth image."<<endl<<endl;
	ElInitArgMain
		(
		argc, argv,
		//mandatory arguments
		LArgMain() << EAMC(aNameOriDEM, "DEM xml file name (such as Z_NumX_DeZoomX_STD-MALT.xml)", eSAM_IsExistFile)
		<< EAMC(aMinZ, "Min Z (m)")
		<< EAMC(aMaxZ, "Max Z (m)"),
		//optional arguments
		LArgMain() << EAM(aMasqSup, "MasqSup", true, "Supplementary mask")
		<< EAM(aCorrelIm, "CorrelIm", true, "Use correl image as a mask")
        << EAM(aCorThr, "CorrelThr", true, "Correl Threshold for acceptance (def=0.02)")
        << EAM(aOriginIm, "OriginIm", true, "Original image (used only with OriginImMinVal)")
        << EAM(aOriginImMinVal, "OriginImMinVal", true, "Minimal value for origin image (used only with OriginIm)")
        << EAM(aBoolDEM, "DEM", true, "Output masked DEM (def=true)")
		);

    if (MMVisualMode) return EXIT_SUCCESS;
    
    ELISE_ASSERT(aMinZ<aMaxZ,"Please try with MinZ<MaxZ...");

    ELISE_ASSERT( !((aOriginIm=="?")^(aOriginImMinVal==300)),"Please set both OriginIm and OriginImMinVal.");
    
	string aMasqName = StdPrefix(aNameOriDEM) + "_MasqZminmax";
	string aFileDEMCorName = StdPrefix(aNameOriDEM) + "_Masked";

	// Read DEM
	cFileOriMnt aOriDEM = StdGetFromPCP(aNameOriDEM, FileOriMnt);

	string aNameFileDEM = aOriDEM.NameFileMnt();
    string aNameFileMasque="";
    
	if (aOriDEM.NameFileMasque().IsInit())
		aNameFileMasque = aOriDEM.NameFileMasque().Val();

	double aOrigineAlti = aOriDEM.OrigineAlti();
	double aResolutionAlti = aOriDEM.ResolutionAlti();
	//cout << "z: *" << aResolutionAlti << " +" << aOrigineAlti << endl;
    cout<<"MinZ: "<<aMinZ<<"    MaxZ:"<<aMaxZ<<endl;
    
	// Create image objects
	Tiff_Im aFileDEM(aNameFileDEM.c_str());
	//image for max
	TIm2D<U_INT1, INT4> tmpImage1T(aOriDEM.NombrePixels());
    Im2D<U_INT1,INT4>  tmpImage1(tmpImage1T._the_im);
	//image for min
	TIm2D<U_INT1, INT4> tmpImage2T(aOriDEM.NombrePixels());
    Im2D<U_INT1,INT4>  tmpImage2(tmpImage2T._the_im);
	//image for masq
	TIm2D<U_INT1, INT4> masqImageT(aOriDEM.NombrePixels());
    Im2D<U_INT1,INT4>  masqImage(masqImageT._the_im);

    // Compute min and max masks
    if ((aOriDEM.Geometrie()==eGeomMNTFaisceauPrChSpherik)||
        (aOriDEM.Geometrie()==eGeomMNTFaisceauIm1PrCh_Px1D)||
        (aOriDEM.Geometrie()==eGeomMNTFaisceauIm1PrCh_Px2D))
    {
        ELISE_COPY(tmpImage1.all_pts(), (1.0/(aFileDEM.in()*aResolutionAlti + aOrigineAlti)<aMaxZ), tmpImage1.out());
        ELISE_COPY(tmpImage2.all_pts(), (1.0/(aFileDEM.in()*aResolutionAlti + aOrigineAlti)>aMinZ), tmpImage2.out());
    }else{
        ELISE_COPY(tmpImage1.all_pts(), ((aFileDEM.in()*aResolutionAlti + aOrigineAlti)<aMaxZ), tmpImage1.out());
        ELISE_COPY(tmpImage2.all_pts(), ((aFileDEM.in()*aResolutionAlti + aOrigineAlti)>aMinZ), tmpImage2.out());
    }
 
	// Multiply min and max masks to creat a minmax mask
    ELISE_COPY(masqImage.all_pts(),(tmpImage1.in()*tmpImage2.in()),masqImage.out());

	// Apply external mask
    if (aMasqSup!="?")
    {
      cout<<"Using MasqSup: "<<aMasqSup<<endl;
      Tiff_Im aFileMasqSup(aMasqSup.c_str());
	  ELISE_ASSERT(aFileMasqSup.sz() == aOriDEM.NombrePixels(), "Masq Sup and DEM must have the same size!");
      TIm2D<U_INT1,INT4> aMasqSupImT(aFileMasqSup.sz());//image for masq sup
      Im2D<U_INT1,INT4>  aMasqSupIm(aMasqSupImT._the_im);
      ELISE_COPY(aMasqSupIm.all_pts(),(aFileMasqSup.in()>0),aMasqSupIm.out());
      ELISE_COPY(masqImage.all_pts(),(masqImage.in()*aMasqSupIm.in()),masqImage.out());
    }

	// Apply mask from correl (out if cor<0.02 (5/255) or <aCorThr)
    if (aCorrelIm!="?")
    {
      cout<<"Using CorrelIm: "<<aCorrelIm<<endl;
      Tiff_Im aFileCorrelIm(aCorrelIm.c_str());
	  ELISE_ASSERT(aFileCorrelIm.sz() == aOriDEM.NombrePixels(), "Correl Image and DEM must have the same size!");
      TIm2D<U_INT1,INT4> aCorrelImImT(aFileCorrelIm.sz());//image for Correl Im
      Im2D<U_INT1,INT4>  aCorrelImIm(aCorrelImImT._the_im);
	  ELISE_COPY(aCorrelImIm.all_pts(), (aFileCorrelIm.in() > (aCorThr * 255)), aCorrelImIm.out());
	  ELISE_COPY(masqImage.all_pts(), (masqImage.in()*aCorrelImIm.in()), masqImage.out());
    }


    // Apply mask from value in original image (out if val<aOriginImMinVal)
    if (aOriginIm!="?")
    {
      cout<<"Using OriginIm: "<<aOriginIm<<endl;
      Tiff_Im aFileOriginIm(aOriginIm.c_str());
      int dezoom_factor=aFileOriginIm.sz().x/aFileDEM.sz().x;
      std::cout<<"Dezoom factor: "<<dezoom_factor<<"\n";

      TIm2D<U_INT1,INT4> aOriginImImT(aFileOriginIm.sz());
      Im2D<U_INT1,INT4>  aOriginImIm(aOriginImImT._the_im);
      ELISE_COPY(aOriginImIm.all_pts(), (aFileOriginIm.in() ), aOriginImIm.out());

      TIm2D<U_INT1,INT4> aOriginImMiniT(aFileOriginIm.sz()/dezoom_factor);
      Im2D<U_INT1,INT4>  aOriginImMini(aOriginImMiniT._the_im);
      ELISE_COPY(aOriginImMini.all_pts(), (aOriginImIm.in()[Virgule(FX*dezoom_factor,FY*dezoom_factor)])>aOriginImMinVal, aOriginImMini.out());

      ELISE_COPY(masqImage.all_pts(), (masqImage.in()*aOriginImMini.in()), masqImage.out());
    }

	// Output masked DEM
	if (aBoolDEM) 
	{
		Tiff_Im  aDEMout
			(
			(aFileDEMCorName + ".tif").c_str(),
			aFileDEM.sz(),
			GenIm::real4,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero
			);


		ELISE_COPY
			(
			aDEMout.all_pts(),
			aFileDEM.in()*masqImage.in(),
			//(aFileDEM.in()*aResolutionAlti + aOrigineAlti)*(masqImage.in()),
			aDEMout.out()
			);


		cFileOriMnt anOriMaskedDEM = aOriDEM;
		anOriMaskedDEM.NameFileMnt() = aFileDEMCorName + ".tif";
		MakeFileXML(anOriMaskedDEM, aFileDEMCorName + ".xml");
		GenTFW(anOriMaskedDEM, (aFileDEMCorName + ".tfw").c_str());

		cout << "Masked DEM created: " << aFileDEMCorName + ".tif/.tfw/.xml" << endl;

	}

	// Output mask
	ELISE_COPY(masqImage.all_pts(), (masqImage.in() * 255), masqImage.out());
    Tiff_Im::CreateFromIm(masqImage, aMasqName + ".tif");

    //for debug
    /*ELISE_COPY(tmpImage1.all_pts(),((1/(aFileMnt.in()*aResolutionAlti+aOrigineAlti))<aMaxZ)*255,tmpImage1.out());
    ELISE_COPY(tmpImage2.all_pts(),((1/(aFileMnt.in()*aResolutionAlti+aOrigineAlti))>aMinZ)*255,tmpImage2.out());
    Tiff_Im::CreateFromIm(tmpImage1,"tmp1.tif");
    Tiff_Im::CreateFromIm(tmpImage2,"tmp2.tif");*/

    
	cFileOriMnt anOriMasq = aOriDEM;
	anOriMasq.NameFileMnt() = aMasqName + ".tif";
	MakeFileXML(anOriMasq, aMasqName + ".xml");
	GenTFW(anOriMasq, (aMasqName + ".tfw").c_str());
    
	cout << "New masq created: " << aMasqName + ".tif/.tfw/.xml" << endl;
	cout << "Zlimit finished." << endl << endl;



    //for debug
    /*TIm2D<REAL4,REAL8> mDepthImageT(aFileMnt.sz());
    Im2D<REAL4,REAL8>  mDepthImage(mDepthImageT._the_im);
    ELISE_COPY(mDepthImage.all_pts(),aFileMnt.in(),mDepthImage.out());
    double min=5000;
    double max=-5000;
    double val;
    for (int anY=0 ; anY<mDepthImage.sz().y ; anY++)
    {
        for (int anX=0 ; anX<mDepthImage.sz().x ; anX++)
        {
            Pt2di aP(anX,anY);
            val=1/(mDepthImageT.get(aP)*aResolutionAlti+aOrigineAlti);
            //cout<<val<<" ";
            if (val<min) min=val;
            if (val>max) max=val;
        }
        //cout<<endl;
    }
    cout<<"min: "<<min<<"   max: "<<max<<endl;*/
    
    
    /*ELISE_COPY(tmpImage2.all_pts(),(1/(aFileMnt.in()*aResolutionAlti+aOrigineAlti)),tmpImage2.out());
    Tiff_Im::CreateFromIm(tmpImage2,"out.tif");*/


    return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a l'utilisation,  a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/
