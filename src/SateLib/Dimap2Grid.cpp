#include "StdAfx.h"
#include "RPC.h"

int Dimap2Grid_main(int argc, char **argv)
{
    std::string aNameFileDimap; // fichier Dimap
    std::string aNameImage;     // nom de l'image traitee
    std::string inputSyst = "+proj=latlong +datum=WGS84 "; //+ellps=WGS84"; //input syst proj
    std::string targetSyst="+init=IGNF:LAMB93";//systeme de projection cible - format proj4
    std::string refineCoef="processing/refineCoef.txt";

    //Creation d'un dossier pour les fichiers intermediaires
    ELISE_fp::MkDirSvp("processing");

    //Creation du fichier de coef par defaut (grille non affinee)
    std::ofstream ficWrite(refineCoef.c_str());
    ficWrite << std::setprecision(15);
    ficWrite << 0 <<" "<< 1 <<" "<< 0 <<" "<< 0 <<" "<< 0 <<" "<< 1 <<" "<<std::endl;

    double altiMin, altiMax;
    int nbLayers;

    double stepPixel = 100.f;
    double stepCarto = 50.f;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aNameFileDimap,"RPC Dimap file")
                  // << EAMC(aNameFileGrid,"Grid file")
                   << EAMC(aNameImage,"Image name")
                   << EAMC(altiMin,"min altitude (ellipsoidal)")
                   << EAMC(altiMax,"max altitude (ellipsoidal)")
                   << EAMC(nbLayers,"number of layers (min 4)"),
        LArgMain()
                 //caracteristique du systeme geodesique saisies sans espace (+proj=utm +zone=10 +north +datum=WGS84...)
                 << EAM(targetSyst,"targetSyst", true,"target system in Proj4 format")
                 << EAM(stepPixel,"stepPixel",true,"Step in pixel")
                 << EAM(stepCarto,"stepCarto",true,"Step in m (carto)")
                 << EAM(refineCoef,"refineCoef",true,"File of Coef to refine Grid")
     );

    // fichier GRID en sortie
    std::string aNameFileGrid = StdPrefixGen(aNameImage)+".GRI";

	RPC aRPC;
	aRPC.ReadDimap(aNameFileDimap);
	aRPC.info();

    std::vector<double> vAltitude;
    for(int i=0;i<nbLayers;++i)
        vAltitude.push_back(altiMin+i*(altiMax-altiMin)/(nbLayers-1));

    /* ISN'T THIS USELESS??
    //Parser du targetSyst
    std::size_t found = targetSyst.find_first_of("+");
    std::string str = "+";
    std::vector<int> position;
    while (found!=std::string::npos)
    {
        targetSyst[found]=' ';
        position.push_back(found);
        found=targetSyst.find_first_of("+",found+1);
    }
    for (int i=position.size()-1; i>-1;i--)
        targetSyst.insert(position[i]+1,str);
    */

    //recuperation des coefficients pour affiner le modele
    std::vector<double> vRefineCoef;
    std::ifstream ficRead(refineCoef.c_str());
    while(!ficRead.eof()&&ficRead.good())
    {
        double a0,a1,a2,b0,b1,b2;
        ficRead >> a0 >> a1 >> a2 >> b0 >> b1 >> b2;

        if (ficRead.good())
        {
            vRefineCoef.push_back(a0);
            vRefineCoef.push_back(a1);
            vRefineCoef.push_back(a2);
            vRefineCoef.push_back(b0);
            vRefineCoef.push_back(b1);
            vRefineCoef.push_back(b2);
        }
    }
    std::cout <<"coef "<<vRefineCoef[0]<<" "<<vRefineCoef[1]<<" "<<vRefineCoef[2]
        <<" "<<vRefineCoef[3]<<" "<<vRefineCoef[4]<<" "<<vRefineCoef[5]<<" "<<std::endl;




    //Test si le modele est affine pour l'appellation du fichier de sortie
    bool refine=false;
    double noRefine[]={0,1,0,0,0,1};

    for(int i=0; i<6;i++)
    {
        if(vRefineCoef[i] != noRefine[i])
            refine=true;
    }

    if (refine)
    {
        //Effacement du fichier de coefficients (affinite=identite) par defaut
        if (ifstream(refineCoef.c_str())) ELISE_fp::RmFile(refineCoef.c_str());

        //New folder
        std::string dir = "refine_" + aNameImage;
        ELISE_fp::MkDirSvp(dir);

        std::cout<<"le modele est affine"<<std::endl;
        aNameFileGrid = dir + ELISE_CAR_DIR + aNameFileGrid;
    }

	aRPC.clearing(aNameFileGrid, refine);
	aRPC.createGrid(aNameFileGrid, aNameImage,
                     stepPixel,stepCarto,
                     vAltitude,targetSyst,inputSyst,vRefineCoef);

    return EXIT_SUCCESS;
}

