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

void Apero2PMVS_Banniere()
{
    std::cout << "\n  *************************************************\n";
    std::cout << "  **                                             **\n";
    std::cout << "  **                   Apero                     **\n";
    std::cout << "  **                     2                       **\n";
    std::cout << "  **                   PMVS                      **\n";
    std::cout << "  **                                             **\n";
    std::cout << "  *************************************************\n";
}

ElMatrix<double> OriMatrixConvertion(CamStenope * aCS)
{
    ElMatrix<double> RotMM2CV(4,3,0.0);	//External orientation matrix
    ElMatrix<double> P(4,3,0.0);		//Orientation (int & ext) matrix in the CV system
    ElMatrix<double> F(3,3,0.0);		//Internal orientation matrix
    ElMatrix<double> Rot(3,3,0.0);		//Rotation matrix between photogrammetry and CV systems
    ElMatrix<double> Rotx(3,3,0.0);		//180° rotation matrix along the x axis

    Rotx(0,0)=1;
    Rotx(1,1)=-1;
    Rotx(2,2)=-1;
    Rot=Rotx*aCS->Orient().Mat();

    for(int i=0;i<3;i++)
    {
        RotMM2CV(i,i)=1;
    }
    RotMM2CV(3,0)=-aCS->VraiOpticalCenter().x;
    RotMM2CV(3,1)=-aCS->VraiOpticalCenter().y;
    RotMM2CV(3,2)=-aCS->VraiOpticalCenter().z;

    RotMM2CV=Rot*RotMM2CV;

    F(0,0)=-aCS->Focale();
    F(1,1)=aCS->Focale();
    Pt2dr PPOut=aCS->DistDirecte(aCS->PP());
    F(2,0)=PPOut.x;
    F(2,1)=PPOut.y;
    F(2,2)=1;

    //Computation of the orientation matrix (P=-F*RotMM2CV)
    P.mul(F*RotMM2CV,-1);

    return P;
}

void Apero2PMVS(string aFullPattern, string aOri)
{
    string aPattern,aNameDir;
    SplitDirAndFile(aNameDir,aPattern,aFullPattern);

    //Bulding the output file system
    ELISE_fp::MkDirRec(aNameDir + "pmvs-"+ aOri +"/models/");
    ELISE_fp::MkDir(aNameDir + "pmvs-"+ aOri +"/visualize/");
    ELISE_fp::MkDir(aNameDir + "pmvs-"+ aOri +"/txt/");

    //Reading the list of input files
    list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    int nbIm = (int)ListIm.size();
    cout<<"Images to process: "<<nbIm<<endl;

    string cmdDRUNK,cmdConv,cmdDel;
    list<string> ListDrunk,ListConvert,ListDel;

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
    //Computing PMVS orientations and writing lists of DRUNK and Convert commands
    for(int i=0;i<nbIm;i++)
    {
        //Reading the images list
        string aFullName=ListIm.front();
        cout<<aFullName<<" ("<<i+1<<" of "<<nbIm<<")"<<endl;
        ListIm.pop_front();

        //Creating the numerical format for the output files names
        char nb[12];
        sprintf(nb, "%08d", i);

        //Creating the lists of DRUNK and Convert commands
        cmdDRUNK=MMDir() + "bin/Drunk " + aNameDir + aFullName + " " + aOri + " Out=" + "pmvs-" + aOri + "/visualize/ Talk=0";
        ListDrunk.push_back(cmdDRUNK);
        #if (ELISE_unix || ELISE_Cygwin || ELISE_MacOs)
            cmdConv="convert " + aNameDir + "pmvs-" + aOri + "/visualize/" + aFullName + ".tif " + aNameDir + "pmvs-"+ aOri +"/visualize/"+(string)nb + ".jpg";
			cmdDel = "rm " + aNameDir + "pmvs-" + aOri + "/visualize/" + aFullName + ".tif ";
        #endif
        #if (ELISE_windows)
            cmdConv=MMDir() + "binaire-aux/windows/convert.exe " + aNameDir + "pmvs-" + aOri + "/visualize/" + aFullName + ".tif " + aNameDir + "pmvs-"+ aOri +"/visualize/"+(string)nb + ".jpg";
			cmdDel = "del " + aNameDir + "pmvs-" + aOri + "/visualize/" + aFullName + ".tif ";
        #endif
		ListConvert.push_back(cmdConv);
		ListDel.push_back(cmdDel);

        //Formating the camera name
        string aNameCam="Ori-"+aOri+"/Orientation-"+aFullName+".xml";

        //Loading the camera
        CamStenope * aCS = CamOrientGenFromFile(aNameCam,anICNM);

        //Compute the Computer Vision calibration matrix
        ElMatrix<double> P(4,3,0.0);//Orientation (int & ext) matrix in the CV system
        P=OriMatrixConvertion(aCS);

        //Write the matrix in the PMVS fashion
        string oriFileName=aNameDir + "pmvs-"+ aOri +"/txt/"+(string)nb+".txt";
        FILE *f = fopen(oriFileName.c_str(), "w");

        fprintf(f,"CONTOUR\n");
        fprintf(f,"%0.6f %0.6f %0.6f %0.6f\n", P(0,0),P(1,0),P(2,0),P(3,0));
        fprintf(f,"%0.6f %0.6f %0.6f %0.6f\n", P(0,1),P(1,1),P(2,1),P(3,1));
        fprintf(f,"%0.6f %0.6f %0.6f %0.6f\n", P(0,2),P(1,2),P(2,2),P(3,2));
        fclose(f);

        delete aCS;
        // delete anICNM; => OBJET A NE PAS DETRUIRE ...
    }//end of "for each image"

    //Undistorting the images with Drunk
    cout<<"Undistorting the images with Drunk"<<endl;
    cEl_GPAO::DoComInParal(ListDrunk,aNameDir + "MkDrunk");

	//Converting into .jpg (pmvs can't use .tif) with Convert
	cout << "Converting into .jpg" << endl;
	cEl_GPAO::DoComInParal(ListConvert, aNameDir + "MkConvert");

	//Removing .tif
	cout << "Removing .tif" << endl;
	cEl_GPAO::DoComInParal(ListDel, aNameDir + "MkDel");

    // Write the options file with basic parameters
    cout<<"Writing the option file"<<endl;
    string optFileName=aNameDir + "pmvs-"+ aOri +"/pmvs_options.txt";
    FILE *f_opt = fopen(optFileName.c_str(), "w");

    fprintf(f_opt, "level 1\n");
    fprintf(f_opt, "csize 2\n");
    fprintf(f_opt, "threshold 0.7\n");
    fprintf(f_opt, "wsize 7\n");
    fprintf(f_opt, "minImageNum 3\n");
    fprintf(f_opt, "CPU 4\n");
    fprintf(f_opt, "setEdge 0\n");
    fprintf(f_opt, "useBound 0\n");
    fprintf(f_opt, "useVisData 0\n");
    fprintf(f_opt, "sequence -1\n");
    fprintf(f_opt, "timages -1 0 %d\n", nbIm);
    fprintf(f_opt, "oimages -3\n");

    fclose(f_opt);

    Apero2PMVS_Banniere();
}


int Apero2PMVS_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    //Reading the arguments
    string aFullPattern,aOri;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullPattern,"Images' name pattern", eSAM_IsPatFile)
                    << EAMC(aOri,"Orientation name", eSAM_IsExistDirOri),
        LArgMain()
                );

    if (MMVisualMode) return EXIT_SUCCESS;

    string aPattern, aDir;
    SplitDirAndFile(aDir, aPattern, aFullPattern);
    StdCorrecNameOrient(aOri, aDir);
    Apero2PMVS(aFullPattern,aOri);

    return EXIT_SUCCESS;
}


/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
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
   associes au chargement, a l'utilisation, a la modification et/ou au
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
