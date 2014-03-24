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
    int nbIm=ListIm.size();
    cout<<"Images to process: "<<nbIm<<endl;

    string cmdDRUNK,cmdConv;
    list<string> ListDrunk,ListConvert;

    //Computing PMVS orientations and writing lists of DRUNK and Convert commands
    for(int i=0;i<nbIm;i++)
    {
        //Reading the images list
        string aFullName=ListIm.front();
        cout<<aFullName<<" ("<<i+1<<" of "<<nbIm<<")"<<endl;
        ListIm.pop_front();

        //Creating the numerical format for the output files names
        char nb[9];
        sprintf(nb, "%08d", i);

        //Creating the lists of DRUNK and Convert commands
        cmdDRUNK=MMDir() + "bin/Drunk " + aNameDir + aFullName + " " + aOri + " Out=" + "pmvs-" + aOri + "/visualize/ Talk=0";
        ListDrunk.push_back(cmdDRUNK);
        #if (ELISE_unix || ELISE_Cygwin || ELISE_MacOs)
            cmdConv="convert ephemeral:" + aNameDir + "pmvs-" + aOri + "/visualize/" + aFullName + ".tif " + aNameDir + "pmvs-"+ aOri +"/visualize/"+(string)nb + ".jpg";
        #endif
        #if (ELISE_windows)
            cmdConv=MMDir() + "binaire-aux/convert ephemeral:" + aNameDir + "pmvs-" + aOri + "/visualize/" + aFullName + ".tif " + aNameDir + "pmvs-"+ aOri +"/visualize/"+(string)nb + ".jpg";
        #endif
        ListConvert.push_back(cmdConv);

        //Formating the camera name
        string aNameCam="Ori-"+aOri+"/Orientation-"+aFullName+".xml";

        //Loading the camera
        cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
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
        delete anICNM;
    }//end of "for each image"

    //Undistorting the images with Drunk
    cout<<"Undistorting the images with Drunk"<<endl;
    cEl_GPAO::DoComInParal(ListDrunk,aNameDir + "MkDrunk");

    //Converting into .jpg (pmvs can't use .tif) with Convert
    cout<<"Converting into .jpg"<<endl;
    cEl_GPAO::DoComInParal(ListConvert,aNameDir + "MkConvert");

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

    Apero2PMVS(aFullPattern,aOri);

    return EXIT_SUCCESS;
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
