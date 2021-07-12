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


class  cReadAppui3d2d : public cReadObject
{
    public :
        cReadAppui3d2d(char aComCar,const std::string & aFormat) :
               cReadObject(aComCar,aFormat,"S"),
               mPt3d(-1,-1,-1),
               mPt2d(-1,-1),
               mInc3(-1,-1,-1),
               mInc (-1)
        {
              AddString("N",&mName,true);
              AddPt3dr("XYZ",&mPt3d,true);
              AddPt2dr("xy",&mPt2d,true);
              AddDouble("Ix",&mInc3.x,false);
              AddDouble("Iy",&mInc3.y,false);
              AddDouble("Iz",&mInc3.z,false);
              AddDouble("I",&mInc,false);
        }

        std::string mName;
        Pt3dr       mPt3d;
        Pt2dr       mPt2d;
        Pt3dr       mInc3;
        double      mInc;
};


//save xml per image
//save txt 3d per image
//if more than 1 image :
//   * launch merge on xml
//   * concatenate all txt into one txt
//   - remove individual files xml and txt

int CPP_GCP2D3D2Xml_main(int argc,char ** argv)
{
	std::string aImNamePat;
	std::string aGCPNamePat;
	std::string aGCPOut = "GCP";
	cElemAppliSetFile aEASF;
	std::string aFormat;
	char aCom = '#';
	std::string aChSys = "";

	ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aFormat,"Format specification, eg. N X Y Z x y",eSAM_None)
		              << EAMC(aImNamePat,"Image file pattern", eSAM_IsPatFile)
                      << EAMC(aGCPNamePat,"GCP file pattern", eSAM_IsPatFile),
           LArgMain() << EAM(aGCPOut,"Out","Output filename,Def=GCP-S2D.xml and GCP-S3D.xml")
		              << EAM(aChSys,"ChSys","File defininf conversion between coordinate frames")
    );


	aEASF.Init(aImNamePat);
	if (aEASF.SetIm()->size()==0)
    {
        std::cout << "Cannot find " << aImNamePat << "\n";
        ELISE_ASSERT(false,"No image in pattern");
    }

	/* Map to store gcp in 3d */
	std::map<std::string,pair<Pt3dr,Pt3dr>> aGCPMap;


	cElRegex  anAutom(aImNamePat.c_str(),15);

	cListOfName aLON;

	for (size_t aKIm=0  ; aKIm< aEASF.SetIm()->size() ; aKIm++)
    {
		std::string aNameIm = (*aEASF.SetIm())[aKIm];
		std::string aNameGCP  =  MatchAndReplace(anAutom,aNameIm,aGCPNamePat);

		/* Read .gcp */

		// 2d xml
		cSetOfMesureAppuisFlottants 		 aPt2dXml;
		std::list< cMesureAppuiFlottant1Im > aLMAF;
		cMesureAppuiFlottant1Im              aMAF;
		aMAF.NameIm() = aNameIm;

		std::list< cOneMesureAF1I > aLOneMes;


		std::cout << "Comment=[" << aCom<<"]\n";
        std::cout << "Format=[" << aFormat<<"]\n";

		cReadAppui3d2d aReadApp(aCom,aFormat);

		ELISE_fp aFIn(aNameGCP.c_str(),ELISE_fp::READ);
        char * aLine;
		while ((aLine = aFIn.std_fgets()))
        {
			 //std::cout << "== " << aLine << "\n";
             if (aReadApp.Decode(aLine))
             {
				//3d
                double  aInc = aReadApp.GetDef(aReadApp.mInc,1);
				aGCPMap[aReadApp.mName] = std::pair<Pt3dr,Pt3dr>(aReadApp.mPt3d,aReadApp.GetDef(aReadApp.mInc3,aInc));

				//2d
				cOneMesureAF1I aOneMes;
				aOneMes.NamePt() = aReadApp.mName;
				aOneMes.PtIm() = aReadApp.mPt2d;

				aLOneMes.push_back(aOneMes);

            }

        }

        aFIn.close();
		
		aMAF.OneMesureAF1I() = aLOneMes;
		aLMAF.push_back(aMAF);
		aPt2dXml.MesureAppuiFlottant1Im() = aLMAF;

		//save image observations
		MakeFileXML(aPt2dXml,StdPrefix(aNameGCP)+".xml");
		aLON.Name().push_back(StdPrefix(aNameGCP)+".xml");

	}
	MakeFileXML(aLON,"GCP-S2D_2merge.xml");

	// merge individual image observation files into one global file 
	std::string aSysCom = "mm3d TestLib MergeMAF NKS-Set-OfFile@GCP-S2D_2merge.xml Out=" + aGCPOut + "_S2D.xml";
	System(aSysCom);
	ELISE_fp::RmFileIfExist("GCP-S2D_2merge.xml");

	//save 3d observations aGCPMap to txt file
	std::string aBlank = " ";
	std::string aFileGCP3d = aGCPOut + "_S3D.txt";
	ELISE_fp aFp(aFileGCP3d.c_str(),ELISE_fp::WRITE,false,ELISE_fp::eTxtTjs);

	aFp.PutCommentaire("F=N_X_Y_Z_Ix_Iy_Iz");
	aFp.SetFormatFloat("%.6f");

	for (auto aPt : aGCPMap)
	{
		//std::cout << aPt.first << " " << aPt.second.first << " " << aPt.second.second << "\n";

	    aFp.str_write(aPt.first.c_str());
	    aFp.str_write(aBlank.c_str());
		aFp.write_REAL8(aPt.second.first.x);
		aFp.write_REAL8(aPt.second.first.y);
		aFp.write_REAL8(aPt.second.first.z);
		aFp.write_U_INT4(aPt.second.second.x);
		aFp.write_U_INT4(aPt.second.second.y);
		aFp.write_U_INT4(aPt.second.second.z);
	    aFp.PutLine();


	}
	aFp.close();

	//save 3d observations aGCPMap to xml file
	aSysCom = "mm3d GCPConvert AppInFile " + aFileGCP3d + (aChSys=="" ? "" : " ChSys="+aChSys);
	System(aSysCom);

	return EXIT_SUCCESS;
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
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

