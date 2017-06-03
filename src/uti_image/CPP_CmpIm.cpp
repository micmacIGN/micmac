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
#include "RStats/cRStats.h"



int CmpIm_main(int argc,char ** argv)
{
     std::string aName1;
     std::string aName2;
     std::string  aFileDiff="";
     bool  OkSzDif= false;
     double aDyn=1.0;
     Pt2di  aBrd(0,0);
     double aMulIm2 = 1.0;
     bool aUseXmlFOM = false;
     double   aColDif = 0;
     std::string mXmlG ="";
     bool aHisto = false;
     bool a16Bit = false;
     // bool aFloat = false;
     
	 ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(aName1,"First image name", eSAM_IsExistFile)
                      << EAMC(aName2,"Second image name", eSAM_IsExistFile) ,
           LArgMain()  << EAM(aFileDiff,"FileDiff",true,"Difference image output file")
                       << EAM(aDyn,"Dyn",true,"Dynamic of difference")
                       << EAM(aBrd,"Brd",true,"Border to eliminate")
                       << EAM(OkSzDif,"OkSzDif",true,"Process files with different sizes")
                       << EAM(aMulIm2,"Mul2",true,"Multiplier of file2 (Def 1.0)")
                       << EAM(aUseXmlFOM,"UseFOM",true,"Consider file as DTSM and use XML FileOriMnt")
                       << EAM(aColDif,"ColDif",true,"Color file of diff using Red/Blue for sign")
                       << EAM(a16Bit,"16Bit",true,"Output file in float format !!")
                       // << EAM(aFloat,"16Bit",true,"Output diff in float")
                       << EAM(mXmlG,"XmlG",true,"Generate Xml")
                       << EAM(aHisto,"Hist",true,"Generate histogram stats")
	 );

	if(aHisto && aFileDiff=="")
		aFileDiff="Diff.tif";
    if(aHisto && aColDif)
		ELISE_ASSERT(false,"You can't produce an RGB file and histogram at the same time (Hist,ColDif);");
	


	if (!MMVisualMode)
	{

        Tiff_Im aFile1 = Tiff_Im::BasicConvStd(aName1);
        Tiff_Im aFile2 = Tiff_Im::BasicConvStd(aName2);
        
        cXmlTNR_TestImgReport aImg;
        aImg.ImgName() = aName1;

        if (aUseXmlFOM && (! EAMIsInit(&OkSzDif))) OkSzDif = true;
        Pt2di aSz = aFile1.sz();
        if (aFile1.sz() != aFile2.sz())
        {
           
           std::cout << "Tailles Differentes " << aFile1.sz() << aFile2.sz() << "\n";
           aImg.TestImgDiff() = false;
           aImg.NbPxDiff() = 99999;
		   aImg.SumDiff() = 99999;
		   aImg.MoyDiff() = 99999;
		   Pt3dr Diff(99999,99999,99999);
		   aImg.DiffMaxi()= Diff;
           if (OkSzDif)
               aSz = Inf( aFile1.sz(),aFile2.sz());
           else
              return -1;
        }
		
		std::cout << "Origin=" << aBrd << ", end=" << aSz-aBrd << "\n"; 

        Fonc_Num aFonc2 = aMulIm2*aFile2.in_proj();
        if (aUseXmlFOM)
        {
              
              cFileOriMnt anFOM1 = StdGetFromPCP(StdPrefix(aName1)+".xml",FileOriMnt);
              cFileOriMnt anFOM2 = StdGetFromPCP(StdPrefix(aName2)+".xml",FileOriMnt);

              aFonc2 = AdaptFonc2FileOriMnt
                       (
                         "CmpIm",
                          anFOM1,
                          anFOM2,
                          aFonc2,
                          true,
                          0.0, 
                          Pt2dr(0,0)
                       );
        }


        Symb_FNum aFDifAbs(Rconv(Abs(aFile1.in()-aFonc2)));

        double aNbDif,aSomDif,aMaxDif,aMinDif,aSom1;
        int  aPtDifMax[2];

        ELISE_COPY
        (
            //aFile1.all_pts(),
            rectangle(aBrd,aSz-aBrd),
            Virgule
            (
                  Rconv(aFDifAbs),
                  aFDifAbs!=0,
                  1.0
            ),
            Virgule
            (
               sigma(aSomDif) | VMax(aMaxDif) | VMin(aMinDif) | WhichMax(aPtDifMax,2),
               sigma(aNbDif),
               sigma(aSom1)
            )
        );
       

        if (aNbDif)
        {

           if (aFileDiff!="")
           {
                Symb_FNum aFDifSigne(aFile1.in()-aFonc2);

                Fonc_Num  aRes = aDyn*aFDifSigne;
                if (aColDif)
                {
                    aRes = Virgule
                           (
                               aRes + (aFDifSigne>0) * aColDif,
                               aRes,
                               aRes + (aFDifSigne<0) * aColDif
                           );
                }
				if(a16Bit)
					Tiff_Im::CreateFromFonc
		  			(
						aFileDiff,
						aSz,
						aRes,
						GenIm::real4
					);
				else
                	Tiff_Im::Create8BFromFonc
                	(
                   		aFileDiff,
                   		aSz,
                   		Max(0,Min(255,128+round_ni(aRes)))
                	);
			
				if(aHisto)
				{
					int aNbV=256;
					cRobustStats aRStat(aRes,aNbV,aBrd,aSz);
					

            /*ELISE_COPY
            (
                rectangle(Pt2di(0,0),aSz),
                Square(aResNOIm.in() - aSomDifNO/aSom1NO),
                sigma(aVarSomNO)
            );*/


           		}

           		std::cout << aName1 << " et " << aName2 << " sont differentes\n";
           		std::cout << "Nombre de pixels differents  = " << aNbDif << "\n";
           		std::cout << "Somme des differences        = " << aSomDif << "\n";
           		std::cout << "Moyenne des differences        = " << (aSomDif/aSom1 )<< "\n";
           		std::cout << "Difference maximale          = " << aMaxDif << " (position " << aPtDifMax[0] << " " << aPtDifMax[1] << ")\n";
		
			}
 
			if(mXmlG!="")
    		{
				
				aImg.TestImgDiff() = false;
				aImg.NbPxDiff() = aNbDif;
				aImg.SumDiff() = aSomDif;
				aImg.MoyDiff() = (aSomDif/aSom1);
				Pt3dr Diff(aPtDifMax[0],aPtDifMax[1],aMaxDif);
				aImg.DiffMaxi()= Diff;
	   		}
		}
        else
        {
           std::cout << "FICHIERS IDENTIQUES SUR LEURS DOMAINES\n";
           aImg.TestImgDiff() = true;
           aImg.NbPxDiff() = aNbDif;
		   aImg.SumDiff() = aSomDif;
		   aImg.MoyDiff() = (aSomDif/aSom1);
		   Pt3dr Diff(aPtDifMax[0],aPtDifMax[1],aMaxDif);
		   aImg.DiffMaxi()= Diff;
        }

        if (mXmlG!="")  // MPD
            MakeFileXML(aImg, mXmlG);
    }
    else{return EXIT_SUCCESS;}
    return 0;
}

int TestCmpIm_Ewelina(int argc,char ** argv)
{
	std::string aName;
	double aRandVal  = 10.0;
	double aRandProc = 0.15;
	double aSystVal  = 0.0;
	double aOutProc  = 0.0;  	

	Pt2di aSz(256,256);

    ElInitArgMain
    (
           argc,argv,
           LArgMain()  << EAMC(aName,"Image name", eSAM_IsExistFile),
           LArgMain()  << EAM(aRandVal,"Rand",true,"Add noise Def=10")
                       << EAM(aRandProc,"RandPr",true,"\% of noise Def=0.15")
                       << EAM(aSystVal,"Syst",true,"Add systematism Def=0")
                       << EAM(aOutProc,"Outlier",true,"\% of outliers Def=0")
    );
	
	Im2D_U_INT1 Im(aSz.x,aSz.y);

	
	//ajoute le bruiit
	TIm2D<INT2,INT>    aImb(aSz);

	srand (time(NULL));
	double aBX = 0, aBY=0;
	std::cout << aRandProc << " " << floor(double(aRandProc)*(aSz.x*aSz.y)) << "\n";
	
	if(aSystVal)
	{
		ELISE_COPY
		(
			rectangle(Pt2di(0,0),Pt2di(aSz.x,aSz.y)),//.chc(2*(FX,FY)),
			FY/aSystVal,
			aImb._the_im.out()
		);
	}

	if(aOutProc)
	{
		for(int aK=0; aK<floor(aOutProc*aSz.x*aSz.y); aK++)
		{
 			aBX = rand() % aSz.x;   
 			aBY = rand() % aSz.y; 
	
			aImb.oset(Pt2di(aBX,aBY),200);
			
		}
	}
	
	for(int aK=0; aK<floor(aRandProc*aSz.x*aSz.y); aK++)
	{ 
 		aBX = rand() % aSz.x;   
 		aBY = rand() % aSz.y; 
	
		aImb.oset(Pt2di(aBX,aBY),aRandVal);
	}
	
	//normalize
	double aVMax, aVMin;
	ELISE_COPY
	(
		rectangle(Pt2di(0,0),Pt2di(aSz.x,aSz.y)),
		aImb._the_im.in(),
		VMax(aVMax) | VMin(aVMin)
	);
	
	ELISE_COPY
	(
		aImb._the_im.all_pts(),
		255/(aVMax-aVMin)*aImb._the_im.in(),
		Im.out()
	);

	Tiff_Im::CreateFromFonc
	(
		aName,
		aSz,
		Im.in(),
		GenIm::real4
	);

	return(1);
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
