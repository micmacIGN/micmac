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

double GetMin(Fonc_Num f, Pt2di aBrd, Pt2di aSz)
{
	double aRes;
	ELISE_COPY
	(	
		rectangle(aBrd,aSz-aBrd),
		f,
		VMin(aRes)
	);
	return(aRes);
}

double GetMax(Fonc_Num f,  Pt2di aBrd, Pt2di aSz)
{
	double aRes;
	ELISE_COPY
	(	
		rectangle(aBrd,aSz-aBrd),
		f,
		VMax(aRes)
	);
	return(aRes);
}

int GetQuantile(Im1D_INT4 *aData, int aNBin, int aNum)
{
	int aCount=0;
	for(int aK=0; aK<aNBin; aK++)
	{
		aCount+=aData->data()[aK];
		
		if(aCount>=aNum)
			return(aK);
	}

	ELISE_ASSERT(false,"CmpIm_main::GetQuantile  aNum overflows the available samples");
	return(0.0);
}


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
                       << EAM(a16Bit,"16Bit",true,"Output file in 16bits")
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
					
					//calculation of the histogram					
					INT NbV = 256;
					double aNormFac = (NbV-1)/(aMaxDif-aMinDif); 
					
					Im1D_INT4  H(NbV,0);
					Flux_Pts aFlux = rectangle(aBrd,aSz-aBrd);
		
					ELISE_COPY
					(
						aFlux.chc(round_ni(Abs(aRes-aMinDif)*aNormFac)),
						1,
						H.histo()
					);
				

						
					FILE * aFp = FopenNN("Stats.txt","w","CmpIm");
					fprintf(aFp,"================= PERC  : RESIDU ==================\n");
					for (int aK=0 ; aK<NbV ; aK++)
					{
								//std::cout << "aK " << aK << "=" << H.data()[aK] << "\n";
								fprintf(aFp,"Res[%f]=%d\n",aK/aNormFac,H.data()[aK]);
					}
					fclose(aFp);
							
						
					//calculation of NMAD
					Im1D_INT4  HAD(NbV,0);
					Flux_Pts aFluxAD = rectangle(aBrd,aSz-aBrd);
							
					Fonc_Num aResMAD = (aRes - GetQuantile(&H, NbV, round_ni(0.5*aSom1))/aNormFac+aMinDif);
							
					double aMinDifAD=GetMin(aResMAD,aBrd,aSz), aMaxDifAD=GetMax(aResMAD,aBrd,aSz);
					double aNormFacAD = double(NbV-1)/(aMaxDifAD-aMinDifAD);

				

					ELISE_COPY
					(
						aFluxAD.chc(round_ni( Abs((aResMAD - aMinDifAD)*aNormFacAD) )),
						1,
						HAD.histo()//already sorted
					);
			
									
                    //calculation of the standard accuracy measures
                    double aSomSquare, aStDevAvant, aVarSom;
                    ELISE_COPY
                    (
                        rectangle(aBrd,aSz-aBrd),
                        Square(aRes),
                        sigma(aSomSquare)
                    );
                    ELISE_COPY
                    (
                        rectangle(aBrd,aSz-aBrd),
                        Square(Abs(aRes) - (aSomDif/aSom1)),
                        sigma(aVarSom)
                    );
                    aStDevAvant = sqrt(aVarSom/(aSom1-1));
			


                    //calculation of the standard accuracy measures on data without outliers
				 	double aSeuil = 2*(sqrt(aSomSquare/aSom1));
					if(2*(sqrt(aSomSquare/aSom1))>50)
						aSeuil = 50;
	
					//a trick to move from Fonc to Im2D_REAL4	
					std::string aNameTmp="NONAME.tif";
					Tiff_Im::CreateFromFonc(aNameTmp,aSz,(aRes),GenIm::real4);
					Im2D_REAL4 aResIm = Im2D_REAL4::FromFileBasic(aNameTmp);	

					Im2D_REAL4 aResNOIm(aSz.x,aSz.y);
					ELISE_COPY
					(
						select(aResIm.all_pts(),Abs(aResIm.in())<aSeuil),
						aResIm.in(),
						aResNOIm.out()
					);
						
					
					if(0)
					{

						Tiff_Im::CreateFromFonc
						(
							"foundOutliers.tif",
							aSz-aBrd,
							aResNOIm.in(),
							GenIm::real4
						);
					}

					double aSomDifNO, aSom1NO, aSomSqNO, aStDevApres, aVarSomNO;
		
					Symb_FNum aResOutlier(aResIm.in()-aResNOIm.in());

					ELISE_COPY
        			(
            			rectangle(aBrd,aSz-aBrd),
						aResOutlier!=0,
						sigma(aSom1NO)
        			);
					ELISE_COPY
        			(
            			rectangle(aBrd,aSz-aBrd),
						Rconv(aResOutlier),
						sigma(aSomDifNO)
					);

       
                    ELISE_COPY
                    (
                        rectangle(aBrd,aSz-aBrd),
                        Square(aResNOIm.in()),
                        sigma(aSomSqNO)
                    );
                    ELISE_COPY
                    (
                        rectangle(aBrd,aSz-aBrd),
                        Square(Abs(aResNOIm.in()) - (aSomDif-aSomDifNO)/(aSom1-aSom1NO)),
                        sigma(aVarSomNO)
                    );
                    aStDevApres = sqrt(aVarSomNO/(aSom1-aSom1NO-1));
			
					ELISE_fp::RmFile(aNameTmp);

					std::cout << "*********************************************************\n";
					std::cout << "**     Accuracy measures by [Hoehle & Hoehle, 2009]    **\n";
					std::cout << "*********************************************************\n";
					std::cout << "              Robust   accuracy measures                 \n";
					std::cout << "                                                         \n";
					std::cout << "Q(0.5)         =" << GetQuantile(&H, NbV, round_ni(0.5*aSom1))/aNormFac+aMinDif 
                                                    << "   : median (50% quantile)\n";
					std::cout << "NMAD           =" <<   1.4826 * GetQuantile(&HAD, NbV, round_ni(0.5*aSom1))/aNormFacAD+aMinDifAD << ": 1.4826 ....\n";
					std::cout << "Q(0.683)       =" << GetQuantile(&H, NbV, round_ni(0.683*aSom1))/aNormFac+aMinDif 
                                                    << "   : 68% quantile\n";
					std::cout << "Q(0.95)        =" << GetQuantile(&H, NbV, round_ni(0.95*aSom1))/aNormFac+aMinDif 
                                                    << "   : 95% quantile\n";
					std::cout << "Q(1.0)         =" << GetQuantile(&H, NbV, round_ni(1*aSom1))/aNormFac+aMinDif 
                                                    << "   :100% quantile\n";
					std::cout << "dh corresponding to a histogram bin=" << 1/aNormFac  << "\n";
					std::cout << "                                                         \n";
					std::cout << "*********************************************************\n";
					std::cout << "              Standard accuracy measures                 \n";
					std::cout << "                                                         \n";
					std::cout << "RMSE                   =" << 
                                     sqrt(aSomSquare/aSom1) <<"\n";
					std::cout << "Mean                   =" << 
                                            (aSomDif/aSom1) <<"\n";
					std::cout << "Std dev                =" << 
											 aStDevAvant    <<"\n";
					std::cout << "                                                         \n";
					std::cout << "Mean                   =" << 
                        (aSomDif-aSomDifNO)/(aSom1-aSom1NO) <<" (no outliers) \n";
					std::cout << "Std dev                =" << 
                                                aStDevApres <<" (no outliers)\n";
					std::cout << "Rejection threshold    =" << aSeuil <<" -> 2*std_dev or 50m\n";
					std::cout << "Rejected outliers      =" << 
                    aSom1NO << "=" << aSom1NO*100/aSom1 << "%" << "\n";
					std::cout << "                                                         \n";
					std::cout << "*********************************************************\n";
						
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
