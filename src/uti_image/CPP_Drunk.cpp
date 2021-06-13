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
#include "hassan/reechantillonnage.h"

void Drunk_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************\n";
    std::cout <<  " *     D-istortion               *\n";
    std::cout <<  " *     R-emoving                 *\n";
    std::cout <<  " *     UN-iversal                *\n";
    std::cout <<  " *     K-it                      *\n";
    std::cout <<  " *********************************\n\n";
}

void Drunk(string aFullPattern,string aOri,string DirOut, bool Talk, bool RGB, Box2di aCrop, double maxSz,bool aCalibFileExtern=false)
{
    string aPattern,aNameDir;
    SplitDirAndFile(aNameDir,aPattern,aFullPattern);

    //Reading input files
    list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    int nbIm = (int)ListIm.size();
    if (Talk){cout<<"Images to process: "<<nbIm<<endl;}

    //Paralelizing (an instance of Drunk is called for each image)
    string cmdDRUNK;
    list<string> ListDrunk;
    if(nbIm!=1)
    {
        for(int i=1;i<=nbIm;i++)
        {
            string aFullName=ListIm.front();
            ListIm.pop_front();
            cmdDRUNK=MMDir() + "bin/mm3d Drunk " + aNameDir + aFullName + " " + aOri + " Out=" + DirOut + " Talk=0" + " RGB=";
            if (RGB)
                cmdDRUNK+="1 ";
            else
                cmdDRUNK+="0 ";
            cmdDRUNK+=" Crop="+ToString<Box2di>(aCrop)+" ";

            ListDrunk.push_back(cmdDRUNK);
        }
        cEl_GPAO::DoComInParal(ListDrunk,aNameDir + "MkDrunk");

        //Calling the banner at the end
        if (Talk){Drunk_Banniere();}
    }else{

    //Bulding the output file system
    ELISE_fp::MkDirRec(aNameDir + DirOfFile(DirOut));

    //Processing the image
    string aNameIm=ListIm.front();
    string aNameOut=aNameDir + DirOut + aNameIm + ".tif";

    //Loading the camera
    string aNameCam="Ori-"+aOri+"/Orientation-"+aNameIm+".xml";
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
    // CamStenope * aCam = CamOrientGenFromFile(aNameCam,anICNM);

    CamStenope * aCam = anICNM->StdCamStenOfNamesSVP (aNameIm,aOri) ; 
    bool WithExtern  = (aCam != nullptr);
    if (!WithExtern)
    {
       std::string  aNameCal = anICNM->StdNameCalib(aOri,aNameIm);
       aCam =  Std_Cal_From_File(aNameCal);
       // aCalibFileExtern= true;
    }

    //Reading the image and creating the objects to be manipulated
    Tiff_Im aTF= Tiff_Im::StdConvGen(aNameDir + aNameIm,3,false);

    Pt2di aSz = aTF.sz();

    //out size
    //if -1 => full picture
    if (aCrop.P0().x<0) {aCrop._p0.x=0; aCrop._p1.x=aSz.x-1;}
    if (aCrop.P0().y<0) {aCrop._p0.y=0; aCrop._p1.y=aSz.y-1;}

    Pt2dr aCenterOut = aCam->DistInverse( Pt2dr(aCrop.P0()+aCrop.P1())/2 );

    //transform crop into output image geometry
    Pt2dr crop0_out=aCam->DistInverse(Pt2dr(aCrop._p0));
    Pt2dr crop1_out=aCam->DistInverse(Pt2dr(aCrop._p1));
    set_min_max(crop0_out.x,crop1_out.x);
    set_min_max(crop0_out.y,crop1_out.y);

    //apply max image size
    if ((crop1_out-crop0_out).x>maxSz*aSz.x)
    {
        std::cout<<"Output picture too big in x: resize."<<std::endl;
        crop0_out.x=aCenterOut.x-maxSz*aSz.x/2;
        crop1_out.x=aCenterOut.x+maxSz*aSz.x/2;
    }
    if ((crop1_out-crop0_out).y>maxSz*aSz.y)
    {
        std::cout<<"Output picture too big in y: resize."<<std::endl;
        crop0_out.y=aCenterOut.y-maxSz*aSz.y/2;
        crop1_out.y=aCenterOut.y+maxSz*aSz.y/2;
    }

    //trunk crop0_out to avoid PP error
    crop0_out.x=(int)crop0_out.x;
    crop0_out.y=(int)crop0_out.y;

    std::cout<<"Crop: "<<aCrop._p0<<" => "<<crop0_out<<"\n";
    std::cout<<"Crop: "<<aCrop._p1<<" => "<<crop1_out<<"\n";
    Pt2dr aOrigOut = crop0_out;
    Pt2di aSzOut = Pt2di(crop1_out-crop0_out);
    std::cout<<aNameCam<<": size out="<<aSzOut<<std::endl;

    Im2D_U_INT1  aImR(aSz.x,aSz.y);
    Im2D_U_INT1  aImG(aSz.x,aSz.y);
    Im2D_U_INT1  aImB(aSz.x,aSz.y);
    Im2D_U_INT1  aImROut(aSzOut.x,aSzOut.y);
    Im2D_U_INT1  aImGOut(aSzOut.x,aSzOut.y);
    Im2D_U_INT1  aImBOut(aSzOut.x,aSzOut.y);

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
    Pt2dr ptIn;
    for (int aY=0 ; aY<aSzOut.y  ; aY++)
    {
        for (int aX=0 ; aX<aSzOut.x  ; aX++)
        {
            ptIn=aCam->DistDirecte(Pt2dr(aX,aY)+aOrigOut);
            aDataROut[aY][aX] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptIn);
            aDataGOut[aY][aX] = Reechantillonnage::biline(aDataG, aSz.x, aSz.y, ptIn);
            aDataBOut[aY][aX] = Reechantillonnage::biline(aDataB, aSz.x, aSz.y, ptIn);
        }
    }
	
	if(RGB)
	{
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
	}
	else
	{
			Tiff_Im  aTOut
		    (
		        aNameOut.c_str(),
                aSzOut,
		        GenIm::u_int1,
		        Tiff_Im::No_Compr,
		        Tiff_Im::BlackIsZero
		    );
			
			ELISE_COPY
			(
				aTOut.all_pts(),
				aImROut.in(),
				aTOut.out()
			);
    }

    if ( !g_externalToolHandler.get("exiftool").isCallable() )
        cerr << "WARNING: exiftool not found" << endl;
    else
    {
        std::string aCom ="exiftool -TagsFromFile " + aNameDir + aNameIm + " " + aNameOut;
        std::cout << aCom<< "\n";
        System(aCom);
    }

    Pt2dr aPPOut=aCam->DistInverse(aCam->PP())-aOrigOut;

    //export ori without disto
    string aDrunkOri=aNameDir + DirOfFile(DirOut) + "Ori-" + aOri + "-" + DirOfFile(DirOut);
    ELISE_fp::MkDirSvp(aDrunkOri);

    //create ideal camera
    if (!aCalibFileExtern)
    {
        std::vector<double> paramFocal;
        cCamStenopeDistPolyn anIdealCam(!aCam->DistIsDirecte(),aCam->Focale(),aPPOut,ElDistortionPolynomiale::DistId(3,1.0),paramFocal);
        if (aCam->ProfIsDef())
            anIdealCam.SetProfondeur(aCam->GetProfondeur());
        anIdealCam.SetSz(aSzOut);
        anIdealCam.SetIdentCam(aCam->IdentCam()+"_ideal");
        if (aCam->HasRayonUtile())
            anIdealCam.SetRayonUtile(aCam->RayonUtile(),30);
        anIdealCam.SetOrientation(aCam->Orient());
        if (aCam->AltisSolIsDef())
            anIdealCam.SetAltiSol(aCam->GetAltiSol());
        anIdealCam.SetIncCentre(aCam->IncCentre());
 
        MakeFileXML(anIdealCam.StdExportCalibGlob(),aDrunkOri+"/Orientation-" + NameWithoutDir(DirOut) + aNameIm + ".tif.xml","MicMacForAPERO");
    }
    else //Export calibration in external file
    {
         //had to pass by the idealcam so as not to have the Interne initialized ble	
		 std::vector<double> paramFocal;
             cCamStenopeDistPolyn anIdealCam(!aCam->DistIsDirecte(),aCam->Focale(),aPPOut,ElDistortionPolynomiale::DistId(3,1.0),paramFocal);
		 anIdealCam.SetSz(aSzOut);
             anIdealCam.SetIdentCam(aCam->IdentCam()+"_ideal");
		 anIdealCam.SetOrientation(aCam->Orient());
		 anIdealCam.SetIncCentre(aCam->IncCentre());
		 if (aCam->AltisSolIsDef())
                anIdealCam.SetAltiSol(aCam->GetAltiSol());
		 if (aCam->ProfIsDef())
                anIdealCam.SetProfondeur(aCam->GetProfondeur());
    
		 cOrientationConique aOCtmp = aCam->StdExportCalibGlob();
		 cOrientationConique aOC; 	 
		
		 std::string aFileInterne = aDrunkOri + NameWithoutDir(anICNM->StdNameCalib("Test",NameWithoutDir(aNameOut)));
		 std::string aFileExterne = aDrunkOri+"Orientation-" + NameWithoutDir(DirOut) + StdPrefix(aNameIm) + ".tif.xml";
    
		 if (! EAMIsInit(&aOCtmp.OrIntImaM2C()))
             	aOC.OrIntImaM2C() = aOCtmp.OrIntImaM2C();
		 if (! EAMIsInit(&aOCtmp.TypeProj()))
		 	aOC.TypeProj() = aOCtmp.TypeProj();
		 if (! EAMIsInit(&aOCtmp.ZoneUtileInPixel()))
		 	aOC.ZoneUtileInPixel() = aOCtmp. ZoneUtileInPixel();
		 if (! EAMIsInit(&aOCtmp.ConvOri()))
		 	aOC.ConvOri() = aOCtmp.ConvOri();
    
		 aOC.Externe() = anIdealCam.StdExportCalibGlob().Externe();
		 aOC.FileInterne().SetVal(aFileInterne);  // 
         MakeFileXML(aOC,aFileExterne);

         cCalibrationInternConique aCIO = StdGetObjFromFile<cCalibrationInternConique>
            	(
                  	Basic_XML_MM_File("Template-Calib-Basic.xml"),
                  	StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                  	"CalibrationInternConique",
                  	"CalibrationInternConique"
            	);

         aCIO.PP() = aPPOut;
         aCIO.F() = aCam->Focale();
         aCIO.SzIm() = aSzOut;
         aCIO.CalibDistortion()[0].ModRad().Val().CDist() = aPPOut;
	 if (aCam->HasRayonUtile())
	 	aCIO.RayonUtile() = aCam->RayonUtile();
	 
    MakeFileXML(aCIO,aFileInterne);

    }
    }
}

int Drunk_main(int argc,char ** argv)
{

    //Testing the existence of argument (if not, print help file)
    if(!MMVisualMode && argc==1)
    {
        argv[1]=(char*)"";//Compulsory to call MMD_InitArgcArgv
        MMD_InitArgcArgv(argc,argv);
        string cmdhelp;
        cmdhelp=MMDir()+"bin/mm3d Drunk -help";
        system_call(cmdhelp.c_str());
    }
    else
    {
        MMD_InitArgcArgv(argc,argv);

        string aFullPattern,aOri;
        string DirOut="DRUNK/";
        Box2di aCrop(Pt2di(-1,-1),Pt2di(-1,-1));
        bool Talk=true, RGB=true;
        double maxSz=1;
	bool aExportInterne=false;
        //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aFullPattern,"Images Pattern", eSAM_IsPatFile)
                        << EAMC(aOri,"Orientation name", eSAM_IsExistDirOri),
            LArgMain()  << EAM(DirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
                        << EAM(Talk,"Talk",true,"Turn on-off commentaries")
                        << EAM(RGB,"RGB",true,"Output file with RGB channels,Def=true,set to 0 for grayscale")
                        << EAM(aCrop,"Crop", true, "Rectangular crop in input image geometry; Def=[-1,-1,-1,-1]=full")
                        << EAM(maxSz,"MaxSz",true,"Maximal output image size (factor for input image); Def=1")
                        << EAM(aExportInterne,"FileInt",true,"Export interior orientation in a separate file; Def=0")
        );

        //Processing the files
		string aPattern, aDir;
		SplitDirAndFile(aDir, aPattern, aFullPattern);
		StdCorrecNameOrient(aOri, aDir);
        Drunk(aPattern,aOri,DirOut,Talk,RGB,aCrop,maxSz,aExportInterne);
    }

    return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

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
titulaire des droits patrimoniaux et les concédants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement, a l'utilisation, a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a  cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
