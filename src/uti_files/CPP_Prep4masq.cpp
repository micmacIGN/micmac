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

/******************************/
/*	   Author: Luc Girod	  */
/******************************/

#include "StdAfx.h"
#include <algorithm>

void Prep4masq_Banniere()
{
    std::cout << "\n*************************************************\n";
    std::cout << "  **                                             **\n";
    std::cout << "  **                   Prep                      **\n";
    std::cout << "  **                     4                       **\n";
    std::cout << "  **                   Masq                      **\n";
    std::cout << "  **                                             **\n";
    std::cout << "  *************************************************\n";
}



int Prep4masq_main(int argc,char ** argv)
{
    std::string aFullPattern;
      //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Full Directory (Dir+Pattern)", eSAM_IsPatFile),
            LArgMain()
        );

        if (MMVisualMode) return EXIT_SUCCESS;

        std::string aDir,aPatIm;
        SplitDirAndFile(aDir,aPatIm,aFullPattern);

        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        const std::vector<std::string> * aSetIm = aICNM->Get(aPatIm);
        std::vector<std::string> aVectIm=*aSetIm;

        list<string> ListConvert;
        for(int i=0;i<int(aVectIm.size());i++){
            //to make the file to manualy modify
            string aNameMasq=StdPrefix(aVectIm[i]) + "_Masq.tif";
			string cmdconv;
			#if (ELISE_unix || ELISE_Cygwin || ELISE_MacOs)
				cmdconv="convert -colorspace gray +compress " + aVectIm[i] + " " + aNameMasq;
			#endif
			#if (ELISE_windows)
				cmdconv=MMDir() + "binaire-aux/windows/convert -colorspace gray +compress " + aVectIm[i] + " " + aNameMasq;
			#endif
		            
            ListConvert.push_back(cmdconv);

            //to read Size
            Tiff_Im aTF1= Tiff_Im::StdConvGen(aDir + aVectIm[i],1,false);
            Pt2di aSz = aTF1.sz();

            cout<<"--- Writing XML"<<endl;
            string NameXML=StdPrefix(aVectIm[i]) + "_Masq.xml";
            std::ofstream file_out(NameXML.c_str(), ios::out);
                if(file_out)  // if file successfully opened
                {
                    file_out <<"<FileOriMnt>" <<endl;
                             file_out <<"<NameFileMnt>./"<<aNameMasq<<"</NameFileMnt>"<<endl;
                             file_out <<"<NombrePixels>"<<aSz.x<<" "<<aSz.y<<"</NombrePixels>"<<endl;
                             file_out <<"<OriginePlani>0 0</OriginePlani>"<<endl;
                             file_out <<"<ResolutionPlani>1 1</ResolutionPlani>"<<endl;
                             file_out <<"<OrigineAlti>0</OrigineAlti>"<<endl;
                             file_out <<"<ResolutionAlti>1</ResolutionAlti>"<<endl;
                             file_out <<"<Geometrie>eGeomMNTFaisceauIm1PrCh_Px1D</Geometrie>"<<endl;
                    file_out << "</FileOriMnt>" <<endl<<endl;
                    file_out.close();
                }
                else{ cerr << "Couldn't write file" << endl;}
        }
        cEl_GPAO::DoComInParal(ListConvert,aDir + "MkMasqFile");

        return 1;
}


int CPP_SetExif(int argc,char **argv)
{
    std::string aPat,aCam,aTps;
    double aFoc,aF35;
    bool aPurge=true;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aPat,"Pattern of images", eSAM_IsPatFile),
        LArgMain()  << EAM(aFoc,"F",true,"Focal lenght")
                    << EAM(aF35,"F35",true,"Focal lenght equiv 35mm")
                    << EAM(aCam,"Cam",true,"Camera model")
					<< EAM(aTps,"Tps",true,"Image timestamp")
                    << EAM(aPurge,"Purge",true,"Purge created exiv2 command file (Def=true)")
    );
    if (MMVisualMode) return EXIT_SUCCESS;

    cElemAppliSetFile anEASF(aPat);

    std::string aNameFile =  Dir2Write(anEASF.mDir) + "ExivBatchFile.txt";
    FILE * aFP = FopenNN(aNameFile,"w","CPP_SetExif");

    int aMul=1000;

    if (EAMIsInit(&aFoc))
         fprintf(aFP,"set Exif.Photo.FocalLength  Rational  %d/%d\n",round_ni(aFoc*aMul),aMul);

   
    if (EAMIsInit(&aFoc))
         // fprintf(aFP,"set Exif.Photo.FocalLengthIn35mmFilm Short  %d\n",aF35);
         fprintf(aFP,"set Exif.Photo.FocalLengthIn35mmFilm Rational  %d/%d\n",round_ni(aF35*aMul),aMul);

    if (EAMIsInit(&aCam))
         fprintf(aFP,"set Exif.Image.Model  Ascii  \"%s\"\n",aCam.c_str());
	
	if (EAMIsInit(&aTps))
         fprintf(aFP,"set Exif.Photo.DateTimeOriginal  Ascii  \"%s\"\n", aTps.c_str());

    fclose(aFP);


    std::list<std::string> aLCom;
    const cInterfChantierNameManipulateur::tSet * aVIm = anEASF.SetIm();

    for (int aKIm=0 ; aKIm<int(aVIm->size()) ; aKIm++)
    {
        std::string aCom ="exiv2 -m " + aNameFile + " " + (*aVIm)[aKIm];
        // std::cout << aCom<< "\n";
        aLCom.push_back(aCom);
        System(aCom);
    }
    cEl_GPAO::DoComInParal(aLCom);


    if (aPurge)
       ELISE_fp::RmFileIfExist(aNameFile);


    return EXIT_SUCCESS;
}


/***********************************************************************/
/*
--> GPS Version ID                  : 2.2.0.0
--> GPS Latitude Ref                : North
--> GPS Longitude Ref               : East
--> GPS Altitude Ref                : Above Sea Level
GPS Time Stamp                  : 12:21:40
GPS Status                      : Unknown ()
GPS Measure Mode                : 3-Dimensional Measurement
GPS Dilution Of Precision       : 0
GPS Speed Ref                   : Unknown ()
GPS Speed                       : 0
GPS Track Ref                   : Unknown ()
GPS Track                       : 0
GPS Img Direction Ref           : Magnetic North
GPS Img Direction               : 29.7
GPS Map Datum                   : WGS-84
GPS Dest Latitude Ref           : Unknown ()
GPS Dest Latitude               : 0 deg 0' 0.00"
GPS Dest Longitude Ref          : Unknown ()
GPS Dest Longitude              : 0 deg 0' 0.00"
GPS Dest Bearing Ref            : Unknown ()
GPS Dest Bearing                : 0
GPS Dest Distance Ref           : Unknown ()
GPS Dest Distance               : 0
GPS Date Stamp                  : 2013:08:23
--> GPS Altitude                    : 201.7 m Above Sea Level
GPS Date/Time                   : 2013:08:23 12:21:40Z
--> GPS Latitude                    : 43 deg 26' 44.80" N
--> GPS Longitude                   : 1 deg 22' 40.63" E
GPS Position                    : 43 deg 26' 44.80" N, 1 deg 22' 40.63" E
*/
/***********************************************************************/

class cAppli_AddGpsData2Xif
{
	public :
		cAppli_AddGpsData2Xif(int argc,char ** argv);
		std::vector<CamStenope *> readOri(std::string Ori, const std::vector<std::string> aSetOfIm, cInterfChantierNameManipulateur * ICNM);
		std::vector<Pt3dr> Cam2Pos(std::vector<CamStenope *> aVCam);
		std::vector<std::string> Cart2Geo(std::vector<Pt3dr> aVPts, std::string aFormat);
		std::vector<std::string> Geo2Geo(std::vector<Pt3dr> aVPts, std::string aFormat);
		std::string GenXifPosFromData(double aLat, double aLon);
		std::string GetLat(std::string aS);
		std::string GetLon(std::string aS);
		std::string GetAlt(double aVal);
		std::string SigneOfLL(double aVal, std::string aL);
		
	private :
		std::string mFormat;
		std::string mFullPat;
		std::string mOri;
};

std::string cAppli_AddGpsData2Xif::GetAlt(double aVal)
{
	std::string aSAlt;
	int aMult=1000;
	
	char aBuffer[100];
	
	sprintf(aBuffer,"%d/%d",static_cast<unsigned int>(aVal*aMult),aMult);
	
	aSAlt=aBuffer;
	
	return aSAlt;
}

std::string cAppli_AddGpsData2Xif::GetLat(std::string aS)
{
	std::string aSLat;
	
	std::size_t aPos = aS.find(",");
	
	aSLat = aS.substr(0,aPos);
	
	return aSLat;
}

std::string cAppli_AddGpsData2Xif::GetLon(std::string aS)
{
	std::string aSLat;
	
	std::size_t aPos = aS.find(",");
	
	aSLat = aS.substr(aPos+1,aS.size());
	
	return aSLat;
}

std::string cAppli_AddGpsData2Xif::SigneOfLL(double aVal, std::string aL)
{
	std::string aS = "";
	
	if(aVal < 0 && aL == "Lat")
	{
		aS = "S";
	}
	else if(aVal > 0 && aL == "Lat")
	{
		aS = "N";
	}
	else if(aVal < 0 && aL == "Lon")
	{
		aS = "W";
	}
	else if(aVal > 0 && aL == "Lon")
	{
		aS = "E";
	}
	else
	{
		ELISE_ASSERT(false,"Bad value in cAppli_AddGpsData2Xif::SigneOfLL");
	}
	
	return aS;
}

std::string cAppli_AddGpsData2Xif::GenXifPosFromData(double aLat, double aLon)
{
	std::string aS;
	
	int aMult=1000000;
	
	double aLatD = trunc(aLat);
	//~ std::cout << "aLatD = " << aLatD << std::endl;
	double aLatM = trunc((aLat - aLatD)*60);
	//~ std::cout << "aLatM = " << aLatM << std::endl;
	double aLatS = ((aLat-aLatD)*60 - aLatM)*60;
	//~ std::cout << "aLatS = " << aLatS << std::endl;
	
	double aLonD = trunc(aLon);
	//~ std::cout << "aLonD = " << aLonD << std::endl;
	double aLonM = trunc((aLon-aLonD)*60);
	//~ std::cout << "aLonM = " << aLonM << std::endl;
	double aLonS = ((aLon-aLonD)*60 - aLonM)*60;
	//~ std::cout << "aLonS = " << aLonS << std::endl;
	
	char aBuffer[100];
	sprintf(
			aBuffer, 
			"%d/1 %d/1 %d/%d, %d/1 %d/1 %d/%d",
	         static_cast<unsigned int>(abs(aLatD)),
	         static_cast<unsigned int>(abs(aLatM)),
	         static_cast<unsigned int>(abs(aLatS*aMult)),
	         aMult,
	         static_cast<unsigned int>(abs(aLonD)),
	         static_cast<unsigned int>(abs(aLonM)),
	         static_cast<unsigned int>(abs(aLonS*aMult)),
	         aMult
	         );
	
	aS = aBuffer;
	std::cout << "aS = " << aS << std::endl;
	return aS;
}

std::vector<std::string> cAppli_AddGpsData2Xif::Geo2Geo(std::vector<Pt3dr> aVPts, std::string aFormat)
{
	std::vector<std::string> aVS;
	
	if(aFormat=="WGS84_deg")
	{
		for(unsigned int aK=0; aK<aVPts.size(); aK++)
		{
			std::string aS = GenXifPosFromData(aVPts.at(aK).x,aVPts.at(aK).y);
			aVS.push_back(aS);
		}
	}
	
	else if(aFormat=="WGS84_rad")
	{
		for(unsigned int aK=0; aK<aVPts.size(); aK++)
		{
			std::string aS = GenXifPosFromData(aVPts.at(aK).x*180/PI,aVPts.at(aK).y*180/PI);
			aVS.push_back(aS);
		}
	}
	
	else
	{
		ELISE_ASSERT(false,"Bad format value in cAppli_AddGpsData2Xif::Geo2Geo");
	}
	
	return aVS;
}

std::vector<std::string> cAppli_AddGpsData2Xif::Cart2Geo(std::vector<Pt3dr> aVPts, std::string aFormat)
{
	std::vector<std::string> aVS;
	
	if(aFormat=="GeoC")
	{
		cChSysCo * aCSC = 0;
		aCSC = cChSysCo::Alloc("GeoC@WGS84","");
	    if(aCSC!=0)
		{
			aVPts = aCSC->Src2Cibl(aVPts);
		}
		else
		{
			ELISE_ASSERT(false,"Bad format value in cChSysCo::Alloc");
		}
		
		aVS = Geo2Geo(aVPts,"WGS84_deg");
	}
	
	else
	{
		ELISE_ASSERT(false,"Bad format value in cAppli_AddGpsData2Xif::Cart2Geo");
	}
	
	return aVS;
}

std::vector<Pt3dr> cAppli_AddGpsData2Xif::Cam2Pos(std::vector<CamStenope *> aVCam)
{
	std::vector<Pt3dr> aVPts;
	
	for (unsigned int aK=0; aK<aVCam.size(); aK++)
	{
		Pt3dr aPt;
		aPt.x = aVCam.at(aK)->PseudoOpticalCenter().x;
		aPt.y = aVCam.at(aK)->PseudoOpticalCenter().y;
		aPt.z = aVCam.at(aK)->PseudoOpticalCenter().z;
		
		aVPts.push_back(aPt);
	}
	
	return aVPts;
}

std::vector<CamStenope *> cAppli_AddGpsData2Xif::readOri(std::string Ori, const std::vector<std::string> aSetOfIm, cInterfChantierNameManipulateur * ICNM)
{
	std::vector<std::string> aVOriFiles(aSetOfIm.size());
    std::vector<CamStenope *> aVCam(aSetOfIm.size());
    
    for (unsigned int aK=0; aK<aSetOfIm.size(); aK++)
    {
		aVOriFiles.at(aK) = Ori+"Orientation-"+aSetOfIm.at(aK)+".xml";
		aVCam.at(aK) = CamOrientGenFromFile(aVOriFiles.at(aK),ICNM);
	}
	
	return aVCam;
}

cAppli_AddGpsData2Xif::cAppli_AddGpsData2Xif(int argc,char ** argv)
{
	std::string aDirImages, aPatIm, aGpsVId="2.2.0.0", aGpsTimeStamp="", aGpsLatRef="", aGpsLonRef=""; 
	bool aGpsAltRef=0; 
	bool aPurge=true;
	
	ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFormat, "Input positions format : WGS84_rad ? WGS84_deg ? GeoC")
					<< EAMC(mFullPat,"Pattern of images", eSAM_IsPatFile)
					<< EAMC(mOri,"Input Ori-folder",eSAM_IsExistDirOri),
        LArgMain()  << EAM(aGpsVId,"GpsVId",false,"GPS Version ID ; Def=2.2.0.0")
					<< EAM(aGpsLatRef,"LatRef",false,"GPS Latitude Ref")
					<< EAM(aGpsLonRef,"LonRef",false,"GPS Longitude Ref")
					<< EAM(aGpsAltRef,"AltRef",false,"GPS Altitude Ref ; Def=0 Above Sea Level (1=Below)",eSAM_IsBool)
					<< EAM(aGpsTimeStamp,"TimeStamp",false,"GPS Time Stamp ; (In Dev)")
					<< EAM(aPurge,"Purge",true,"Purge .xml & .dmp created files in Tmp-MM-Dir")
    );
    
    if(mFormat != "WGS84_rad" && mFormat != "WGS84_deg" && mFormat !="GeoC")
	{
		ELISE_ASSERT(false,"The value of Sys is incorrect ; try 'WGS84_rad' or 'WGS84_deg' or 'GeoC'");
	}
    
    SplitDirAndFile(aDirImages,aPatIm,mFullPat);
    
    std::cout<<"Working dir: "<<aDirImages<<std::endl;
    std::cout<<"Images pattern: "<<aPatIm<<std::endl;
    
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));
    
    //read ori directory
    std::vector<CamStenope *> aVCam = readOri(mOri, aSetIm, aICNM);
    
    //get positions
    std::vector<Pt3dr> aVPts = Cam2Pos(aVCam);
    
    //check their format
    std::vector<std::string> aVS;
    
    if(mFormat=="GeoC")
    {
		aVS = Cart2Geo(aVPts,mFormat);
	}
	else if(mFormat=="WGS84_deg")
	{
		aVS = Geo2Geo(aVPts,mFormat);
	}
	else if(mFormat=="WGS84_rad")
	{
		aVS = Geo2Geo(aVPts,mFormat);
	}
	else
	{
		ELISE_ASSERT(false,"The value of Sys is incorrect ; try 'WGS84_rad' or 'WGS84_deg' or 'GeoC'");
	}
		
	//if Ref Given same for all images
	if(aGpsLatRef == "")
	{
		aGpsLatRef = SigneOfLL(aVPts.at(0).x,"Lat");
	}
	
	if(aGpsLonRef == "")
	{
		aGpsLonRef = SigneOfLL(aVPts.at(0).y,"Lon");
	}
    
    //build good exiv2 format commands
    for(unsigned int aK=0; aK<aSetIm.size(); aK++)
    {
		
		std::string aCom = "exiv2 -M" +
							std::string("\"set Exif.GPSInfo.GPSLatitude ") +
							GetLat(aVS.at(aK)) + std::string("\" -M") +
							std::string("\"set Exif.GPSInfo.GPSLongitude ") +
							GetLon(aVS.at(aK)) + "\" -M" +
							std::string("\"set Exif.GPSInfo.GPSLatitudeRef ") +
							aGpsLatRef + "\" -M" +
							std::string("\"set Exif.GPSInfo.GPSLongitudeRef ") +
							aGpsLonRef + std::string("\" -M") +
							std::string("\"set Exif.GPSInfo.GPSAltitude ") +
							GetAlt(aVPts.at(aK).z) + std::string("\" ") +
							//std::string("\"set Exif.GPSInfo.GPSAltitudeRef ") +
							
							aSetIm.at(aK);
		
		System(aCom);				
		//~ std::cout << "aCom = " << aCom << std::endl;
	}
}

int CPP_SetGpsExif(int argc,char **argv)
{
	cAppli_AddGpsData2Xif anAppli(argc,argv);
	return EXIT_SUCCESS;
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
