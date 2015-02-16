#include "StdAfx.h"

//! Classe pour la conversion de fichier DIMAP en fichier Grid
class Dimap
{
public:
    ///
    /// \brief lecture du fichier DIMAP
    /// \param filename nom du fichier DIMAP
    ///
    Dimap(std::string const &filename);

    ///
    /// \brief transformation des coordonnées d'un point image en point terrain
    /// \param Pimg point image
    /// \param altitude altitude terrain
    /// \return point terrain
    ///
    Pt2dr direct(Pt2dr Pimg, double altitude)const;

    ///
    /// \brief transformation des coordonnées d'un point terrain en point image
    /// \param Pgeo point terrain
    /// \param altitude altitude terrain
    /// \param vRefineCoef vecteur contenant les six coefficients de l'affinité servant à affiner la grille
    /// \param rowCrop ligne du coin haut gauche pour croper
    /// \param sampCrop colonne du coin haut gauche pour croper
    /// \return
    ///
    Pt2dr indirect(Pt2dr Pgeo, double altitude,std::vector<double> vRefineCoef, double rowCrop, double sampCrop)const;

    ///
    /// \brief Conversion de coordonnées géographiques d'un point en coordonnées cartographiques
    /// \param Pgeo coordonnées géoégraphiques
    /// \param targetSyst système de projection cible, par defaut targetSyst="+init=IGNF:LAMB93"
    /// \return
    ///
	vector<Pt2dr> empriseCarto(vector<Pt2dr> Pgeo, std::string targetSyst, std::string inputSyst)const;

    ///
    /// \brief Application de l'affinité à la grille au point Pimg
    /// \param Pimg point image
    /// \param vRefineCoef vecteur contenant les six coefficients de l'affinité servant à affiner la grille
    /// \param rowCrop ligne du coin haut gauche pour croper
    /// \param sampCrop colonne du coin haut gauche pour croper
    /// \return
    ///
    Pt2dr ptRefined(Pt2dr Pimg,std::vector<double> vRefineCoef,double rowCrop, double sampCrop)const;

    ///
    /// \brief détermination des sommets de la grille en coordonnées image en fonction du pas (en pixels) puis conversion en coordonnées géographiques
    /// \param ulcSamp colonne du coin supérieur gauche de la grille en coordonnées image
    /// \param ulcLine ligne du coin supérieur gauche de la grille en coordonnées image
    /// \param stepPixel pas en pixels pour la grille en coordonnées image
    /// \param nbSamp nombre de colonnes de la grille en coordonnées image
    /// \param nbLine nombre de lignes de la grille en coordonnées image
    /// \param vAltitude vecteur contenant les altitudes de chaque « layer »
    /// \param vPtCarto vecteur de structures de points Pt2dr qui contient les sommets de la grille directe (pour l'ensemble des « layers »)
    /// \param targetSyst système de projection cible, suivant la nomenclature proj4
    /// \param vRefineCoef vecteur contenant les six coefficients de l'affinité servant à affiner la grille
    /// \param rowCrop ligne du coin supérieur gauche de l'image - pour crop
    /// \param sampCrop colonne du coin supérieur gauche de l'image - pour crop
    ///
    void createDirectGrid(double ulcSamp, double ulcLine,
                          double stepPixel,
                          int nbSamp, int  nbLine,
                          std::vector<double> const &vAltitude,
						  std::vector<Pt2dr> &vPtCarto, std::string targetSyst, std::string inputSyst,
                          std::vector<double> vRefineCoef,double rowCrop, double sampCrop)const;

    ///
    /// \brief calcul des sommets de la grille en coordonnées terrain (cartographiques) en fonction du pas puis conversion en coordonnées géographiques et enfin image
    /// \param ulcX longitude du coin supérieur gauche de la grille en coordonnées cartographiques
    /// \param ulcY latitude du coin supérieur gauche de la grille en coordonnées cartographiques
    /// \param nbrSamp nombre de colonnes de la grille en coordonnées cartographiques
    /// \param nbrLine nombre de lignes de la grille en coordonnées cartographiques
    /// \param stepCarto pas en mètres pour la grille en coordonnées cartographiques
    /// \param vAltitude vecteur contenant les altitudes de chaque « layer »
    /// \param vPtImg vecteur de sommets de la grille inverse (pour l'ensemble des « layers »)
    /// \param targetSyst système de projection cible, suivant la nomenclature proj4
    /// \param vRefineCoef vecteur contenant les six coefficients de l'affinité servant à affiner la grille
    /// \param rowCrop ligne du coin supérieur gauche de l'image - pour crop
    /// \param sampCrop colonne du coin supérieur gauche de l'image - pour crop
    ///
    void createIndirectGrid(double ulcX, double ulcY,
                            int nbrSamp, int nbrLine,
                            double stepCarto,
                            std::vector<double> const &vAltitude,
							std::vector<Pt2dr> &vPtImg, std::string targetSyst, std::string inputSyst,
                            std::vector<double> vRefineCoef, double rowCrop, double sampCrop)const;

    ///
    /// \brief creation du fichier XML et calculs intermediaires
    /// \param nomGrid nom du fichier Grid en sortie
    /// \param nomImage nom de l'image concernée
    /// \param stepPixel pas en pixels pour la grille en coordonnées image
    /// \param stepCarto pas en mètres pour la grille en coordonnées cartographiques
    /// \param rowCrop ligne du coin supérieur gauche de l'image - pour crop
    /// \param sampCrop colonne du coin supérieur gauche de l'image - pour crop
    /// \param vAltitude vecteur contenant les altitudes de chaque « layer »
    /// \param targetSyst système de projection cible, suivant la nomenclature proj4
    /// \param vRefineCoef vecteur contenant les six coefficients de l'affinité servant à affiner la grille
    ///
    void createGrid(std::string const &nomGrid, std::string const &nomImage,
                    double stepPixel, double stepCarto,
                    double rowCrop, double sampCrop,
					std::vector<double> vAltitude, std::string targetSyst, std::string inputSyst,
                    std::vector<double> vRefineCoef)const;

    ///
    /// \brief infos fichier DIMAP
    ///
    void info()
    {
        std::cout << "Dimap info:"<<std::endl;
        std::cout << "==========================================================="<<std::endl;
        std::cout << "long_scale   : "<<long_scale<<  " | long_off   : "<<long_off<<std::endl;
        std::cout << "lat_scale    : "<<lat_scale<<   " | lat_off    : "<<lat_off <<std::endl;
        std::cout << "height_scale : "<<height_scale<<" | height_off : "<<height_off<<std::endl;
        std::cout << "samp_scale   : "<<samp_scale<<  " | samp_off   : "<<samp_off<<std::endl;
        std::cout << "line_scale   : "<<line_scale<<  " | line_off   : "<<line_off<<std::endl;
        std::cout << "first_row    : "<<first_row<<   " | last_row   : "<<last_row<<std::endl;
        std::cout << "first_col    : "<<first_col<<   " | last_col   : "<<last_col<<std::endl;
        std::cout << "first_lon    : "<<first_lon<<   " | last_lon   : "<<last_lon<<std::endl;
        std::cout << "first_lat    : "<<first_lat<<   " | last_lat   : "<<last_lat<<std::endl;
        std::cout << "direct_samp_num_coef : "<<direct_samp_num_coef.size()<<std::endl;
        std::cout << "direct_samp_den_coef : "<<direct_samp_den_coef.size()<<std::endl;
        std::cout << "direct_line_num_coef : "<<direct_line_num_coef.size()<<std::endl;
        std::cout << "direct_line_den_coef : "<<direct_line_den_coef.size()<<std::endl;
        std::cout << "indirect_samp_num_coef : "<<indirect_samp_num_coef.size()<<std::endl;
        std::cout << "indirect_samp_den_coef : "<<indirect_samp_den_coef.size()<<std::endl;
        std::cout << "indirect_line_num_coef : "<<indirect_line_num_coef.size()<<std::endl;
        std::cout << "indirect_line_den_coef : "<<indirect_line_den_coef.size()<<std::endl;
        std::cout << "==========================================================="<<std::endl;
    }

    ///
    /// \brief effacement des fichiers relatifs à la creation des grilles ssi le modèle n'est pas affiné
    /// \param nomGrid nom du fichier Grid en sortie
    /// \param refine la grille est-elle affinée
    ///
    void clearing(std::string const &nomGrid, bool refine)
    {
        if (refine == false)
        {
            if (ifstream("processing/conv_ptGeo.txt"))       ELISE_fp::RmFile("processing/conv_ptGeo.txt");
            if (ifstream("processing/conv_ptCarto.txt"))     ELISE_fp::RmFile("processing/conv_ptCarto.txt");
            if (ifstream("processing/direct_ptGeo.txt"))     ELISE_fp::RmFile("processing/direct_ptGeo.txt");
            if (ifstream("processing/direct_ptCarto.txt"))   ELISE_fp::RmFile("processing/direct_ptCarto.txt");
            if (ifstream("processing/indirect_ptGeo.txt"))   ELISE_fp::RmFile("processing/indirect_ptGeo.txt");
            if (ifstream("processing/indirect_ptCarto.txt")) ELISE_fp::RmFile("processing/indirect_ptCarto.txt");
            if (ELISE_fp::IsDirectory("processing"))         ELISE_fp::RmDir("processing");
        }
        //effacement de la grille affinee + grilles GRC et binaire
        std::string gridGRC = nomGrid;
        std::string refGridGRC2 = nomGrid;
        refGridGRC2.append("Bin");

        if (ifstream(nomGrid.c_str()))     ELISE_fp::RmFile(nomGrid.c_str());
        if (ifstream(gridGRC.c_str()))     ELISE_fp::RmFile(gridGRC.c_str());
        if (ifstream(refGridGRC2.c_str())) ELISE_fp::RmFile(refGridGRC2.c_str());
    }

    ///
    /// \brief vecteur des 20 coefficients du numérateur de la fonction de calcul de la longitude par transformation  RFM directe
    ///
    std::vector<double> direct_samp_num_coef;
    ///
    /// \brief vecteur des 20 coefficients du dénominateur de la fonction de calcul de la longitude par transformation  RFM directe
    ///
    std::vector<double> direct_samp_den_coef;
    ///
    /// \brief vecteur des 20 coefficients du numérateur de la fonction de calcul de la latitude par transformation  RFM directe
    ///
    std::vector<double> direct_line_num_coef;
    ///
    /// \brief vecteur des 20 coefficients du dénominateur de la fonction de calcul de la latitude par transformation  RFM directe
    ///
    std::vector<double> direct_line_den_coef;
    ///
    /// \brief vecteur des 20 coefficients du numérateur de la fonction de calcul de la colonne par transformation  RFM indirecte
    ///
    std::vector<double> indirect_samp_num_coef;
    ///
    /// \brief vecteur des 20 coefficients du dénominateur de la fonction de calcul de la colonne par transformation  RFM indirecte
    ///
    std::vector<double> indirect_samp_den_coef;
    ///
    /// \brief vecteur des 20 coefficients du numérateur de la fonction de calcul de la ligne par transformation  RFM indirecte
    ///
    std::vector<double> indirect_line_num_coef;
    ///
    /// \brief vecteur des 20 coefficients du dénominateur de la fonction de calcul de la ligne par transformation  RFM indirecte
    ///
    std::vector<double> indirect_line_den_coef;

    double indirErrBiasRow;
    double indirErrBiasCol;
    double dirErrBiasX;
    double dirErrBiasY;

    ///
    /// \brief ligne minimale du domaine de validité de la transformation directe (RFM) issue du fichier DIMAP
    ///
    double first_row;
    ///
    /// \brief colonne minimale du domaine de validité de la transformation directe (RFM) issue du fichier DIMAP
    ///
    double first_col;
    ///
    /// \brief ligne maximale du domaine de validité de la transformation directe (RFM) issue du fichier DIMAP
    ///
    double last_row;
    ///
    /// \brief colonne maximale du domaine de validité de la transformation directe (RFM) issue du fichier DIMAP
    ///
    double last_col;
    ///
    /// \brief longitude minimale du domaine de validité de la transformation indirecte (RFM) issue du fichier DIMAP
    ///
    double first_lon;
    ///
    /// \brief latitude minimale du domaine de validité de la transformation indirecte (RFM) issue du fichier DIMAP
    ///
    double first_lat;
    ///
    /// \brief longitude maximale du domaine de validité de la transformation indirecte (RFM) issue du fichier DIMAP
    ///
    double last_lon;
    ///
    /// \brief latitude maximale du domaine de validité de la transformation indirecte (RFM) issue du fichier DIMAP
    ///
    double last_lat;
    ///
    /// \brief facteur d'échelle de la longitude (géographique) issu du fichier Dimap
    ///
    double long_scale;
    ///
    /// \brief offset de la longitude (géographique) issu du fichier DIMAP
    ///
    double long_off;
    ///
    /// \brief acteur d'échelle de la latitude (géographique) issu du fichier Dimap
    ///
    double lat_scale;
    ///
    /// \brief offset de la latitude (géographique) issu du fichier DIMAP
    ///
    double lat_off;
    ///
    /// \brief facteur d'échelle de la colonne (coordonnées image) issu du fichier DIMAP
    ///
    double samp_scale;
    ///
    /// \brief offset de la colonne (coordonnées image) issu du fichier DIMAP
    ///
    double samp_off;
    ///
    /// \brief facteur d'échelle de la ligne (coordonnées image)  issu du fichier DIMAP
    ///
    double line_scale;
    ///
    /// \brief offset de la ligne (coordonnées image)  issu du fichier DIMAP
    ///
    double line_off;

    ///
    /// \brief facteur d'échelle de l'altitude issu du fichier DIMAP
    ///
    double height_scale;
    ///
    /// \brief offset de l'altitude issu du fichier DIMAP
    ///
    double height_off;

};



Pt2dr Dimap::direct(Pt2dr Pimg, double altitude)const
{
    //Computing coordinates for normalized image
    double Y=(Pimg.y-line_off)/line_scale;
    double X=(Pimg.x-samp_off)/samp_scale;
    double Z=(altitude-height_off)/height_scale;
    double vecteurD[]={1,X,Y,Z,Y*X,X*Z,Y*Z,X*X,Y*Y,Z*Z,X*Y*Z,X*X*X,Y*Y*X,X*Z*Z,X*X*Y,Y*Y*Y,Y*Z*Z,X*X*Z,Y*Y*Z,Z*Z*Z};

    double long_den = 0.;
    double long_num = 0.;
    double lat_den = 0.;
    double lat_num = 0.;

    for (int i=0; i<20; i++)
    {
        lat_num  += vecteurD[i]*direct_line_num_coef[i];
        lat_den  += vecteurD[i]*direct_line_den_coef[i];
        long_num += vecteurD[i]*direct_samp_num_coef[i];
        long_den += vecteurD[i]*direct_samp_den_coef[i];
    }

    //Final computation
    Pt2dr Pgeo;
    if ((lat_den != 0) && (long_den !=0))
    {
        Pgeo.x = (lat_num / lat_den) * lat_scale + lat_off;
        Pgeo.y = (long_num / long_den) * long_scale + long_off;
    }
    else
    {
		std::cout << "Computing error - denominator = 0" << std::endl;
    }
    return Pgeo;
}

Pt2dr Dimap::indirect(Pt2dr Pgeo, double altitude, std::vector<double> vRefineCoef,double rowCrop, double sampCrop)const
{
    //Computing coordinates for normalized image
    double Y=(Pgeo.y-long_off)/long_scale;
    double X=(Pgeo.x-lat_off)/lat_scale;
    double Z=(altitude-height_off)/height_scale;
    double vecteurD[]={1,Y,X,Z,Y*X,Y*Z,X*Z,Y*Y,X*X,Z*Z,X*Y*Z,Y*Y*Y,Y*X*X,Y*Z*Z,X*Y*Y,X*X*X,X*Z*Z,Y*Y*Z,X*X*Z,Z*Z*Z};

    double samp_den = 0.;
    double samp_num = 0.;
    double line_den = 0.;
    double line_num = 0.;

    for (int i=0; i<20; i++)
    {
        line_num  += vecteurD[i]*indirect_line_num_coef[i];
        line_den  += vecteurD[i]*indirect_line_den_coef[i];
        samp_num  += vecteurD[i]*indirect_samp_num_coef[i];
        samp_den  += vecteurD[i]*indirect_samp_den_coef[i];
    }
    //Final computation
    Pt2dr Pimg;
    if ((samp_den != 0) && (line_den !=0))
    {
        Pimg.x = (samp_num / samp_den) * samp_scale + samp_off;
        Pimg.y = (line_num / line_den) * line_scale + line_off;
    }
    else
    {
        std::cout << "Computing error - denominator = 0"<<std::endl;
    }
    Pt2dr PimgRefined = ptRefined(Pimg,vRefineCoef,rowCrop,sampCrop);
    return PimgRefined;
}

//Returns the vector [Pt2dr(xmin,ymin),Pt2dr(xmax,ymax)]

vector<Pt2dr> Dimap::empriseCarto(vector<Pt2dr> Pgeo, std::string targetSyst, std::string inputSyst)const
{
    std::ofstream fic("processing/conv_ptGeo.txt");
    fic << std::setprecision(15);
	for (int i = 0; i<Pgeo.size(); i++)
	{
		fic << Pgeo[i].y << " " << Pgeo[i].x << endl;
	}
    // transformation in the ground coordinate system
    std::string command;
	command = g_externalToolHandler.get("cs2cs").callName() + " " + inputSyst + " +to " + targetSyst + " processing/conv_ptGeo.txt > processing/conv_ptCarto.txt";
    int res = system(command.c_str());
	ELISE_ASSERT(res == 0, " error calling cs2cs in ptGeo2Carto ");
    // loading the coordinate of the converted point
	vector<double> PtsCartox, PtsCartoy;
    std::ifstream fic2("processing/conv_ptCarto.txt");
    while(!fic2.eof()&&fic2.good())
    {
        double X,Y,Z;
        fic2 >> Y >> X >> Z;
        if (fic2.good())
        {
			PtsCartox.push_back(X);
			PtsCartoy.push_back(Y);
        }
    }

	vector<Pt2dr> anEmpriseCarto;
	anEmpriseCarto.push_back(Pt2dr(*std::min_element(PtsCartox.begin(), PtsCartox.end()), *std::min_element(PtsCartoy.begin(), PtsCartoy.end())));
	anEmpriseCarto.push_back(Pt2dr(*std::max_element(PtsCartox.begin(), PtsCartox.end()), *std::max_element(PtsCartoy.begin(), PtsCartoy.end())));
	
	return anEmpriseCarto;
}


/* MPD :

   Sur UBUNTU J'Otiens les erreurs :

home/marc/MMM/culture3d/src/TpMMPD/Dimap2Grid.cpp:379:51: erreur: ¿begin¿ is not a member of ¿std¿
/home/marc/MMM/culture3d/src/TpMMPD/Dimap2Grid.cpp:379:74: erreur: ¿end¿ is not a member of ¿std¿
/home/marc/MMM/culture3d/src/TpMMPD/Dimap2Grid.cpp:379:114: erreur: ¿begin¿ is not a member of ¿std¿
/home/marc/MMM/culture3d/src/TpMMPD/Dimap2Grid.cpp:379:137: erreur: ¿end¿ is not a member of ¿std¿
/home/marc/MMM/culture3d/src/TpMMPD/Dimap2Grid.cpp:380:51: erreur: ¿begin¿ is not a member of ¿std¿
/home/marc/MMM/culture3d/src/TpMMPD/Dimap2Grid.cpp:380:74: erreur: ¿end¿ is not a member of ¿std¿
/home/marc/MMM/culture3d/src/TpMMPD/Dimap2Grid.cpp:380:114: erreur: ¿begin¿ is not a member of ¿std¿
/home/marc/MMM/culture3d/src/TpMMPD/Dimap2Grid.cpp:380:137: erreur: ¿end¿ is not a member of ¿std¿

J'imagine qu'il faut remplacer  :


   std::begin(PtsCartox)   par PtsCartox.begin()

Mais je prefere que ce soit l'auteur du code qui valide ce changement.

*/


Pt2dr Dimap::ptRefined(Pt2dr Pimg, std::vector<double> vRefineCoef,double rowCrop, double sampCrop)const
{
    //Pour calculer les coordonnees affinees d'un point
    Pt2dr pImgRefined;
    Pt2dr pCropRefined;
    Pt2dr pCrop;
    pCrop.x=Pimg.x-sampCrop;
    pCrop.y=Pimg.y-rowCrop;
    pCropRefined.x= vRefineCoef[0] + pCrop.x * vRefineCoef[1] + pCrop.y * vRefineCoef[2];
    pCropRefined.y= vRefineCoef[3] + pCrop.x * vRefineCoef[4] + pCrop.y * vRefineCoef[5];

    pImgRefined.x= pCropRefined.x+sampCrop;
    pImgRefined.y= pCropRefined.y+rowCrop;

    return pImgRefined;
}


void Dimap::createDirectGrid(double ulcSamp, double ulcLine,
                             double stepPixel,
                             int nbSamp, int  nbLine,
                             std::vector<double> const &vAltitude,
							 std::vector<Pt2dr> &vPtCarto, std::string targetSyst, std::string inputSyst,
                             std::vector<double> vRefineCoef,double rowCrop, double sampCrop)const
{
    vPtCarto.clear();
    // On cree un fichier de points geographiques pour les transformer avec proj4
    {
        std::ofstream fic("processing/direct_ptGeo.txt");
        fic << std::setprecision(15);
        for(size_t i=0;i<vAltitude.size();++i)
        {
            double altitude = vAltitude[i];
            for(int l=0;l<nbLine;++l)
            {
                for(int c = 0;c<nbSamp;++c)
                {
                    Pt2dr Pimg(ulcSamp + c * stepPixel, ulcLine + l * stepPixel);

                    //pour affiner les coordonnees
                    Pt2dr PimgRefined = ptRefined(Pimg, vRefineCoef, rowCrop, sampCrop);

                    Pt2dr Pgeo = direct(PimgRefined,altitude);
					fic << Pgeo.y << " " << Pgeo.x << std::endl;
                }
            }
        }
    }
	// transformation in the ground coordinate system
    std::string command;
	command = g_externalToolHandler.get("cs2cs").callName() + " " + inputSyst + " +to " + targetSyst + " -s processing/direct_ptGeo.txt > processing/direct_ptCarto.txt";
	int res = system(command.c_str());
    if (res != 0) std::cout<<"error calling cs2cs in createDirectGrid"<<std::endl;
    // loading points
    std::ifstream fic("processing/direct_ptCarto.txt");
    while(!fic.eof()&&fic.good())
    {
        double X,Y,Z;
        fic >> Y >> X >> Z;
        if (fic.good())
            vPtCarto.push_back(Pt2dr(X,Y));
    }
    std::cout << "Number of points in direct grid : "<<vPtCarto.size()<<std::endl;
}

void Dimap::createIndirectGrid(double ulcX, double ulcY, int nbrSamp, int nbrLine,
                               double stepCarto, std::vector<double> const &vAltitude,
							   std::vector<Pt2dr> &vPtImg, std::string targetSyst, std::string inputSyst,
                               std::vector<double> vRefineCoef, double rowCrop, double sampCrop)const
{
    vPtImg.clear();

    // On cree un fichier de points cartographiques pour les transformer avec proj4
    {
        std::ofstream fic("processing/indirect_ptCarto.txt");
        fic << std::setprecision(15);
        for(int l=0;l<nbrLine;++l)
        {
            double Y = ulcY - l*stepCarto;
            for(int c = 0;c<nbrSamp;++c)
            {
                double X =ulcX + c*stepCarto;
				fic << X << " " << Y << std::endl;
            }
        }
    }
    // transfo en Geo
    std::string command;

    command = g_externalToolHandler.get( "cs2cs" ).callName() + " " + targetSyst + " +to " + inputSyst + " -f %.12f -s processing/indirect_ptCarto.txt >processing/indirect_ptGeo.txt";
    int res = system(command.c_str());
    if (res != 0) std::cout<<"error calling cs2cs in createIndirectGrid"<<std::endl;
    for(size_t i=0;i<vAltitude.size();++i)
    {
        double altitude = vAltitude[i];
        // chargement des points
        std::ifstream fic("processing/indirect_ptGeo.txt");
        while(!fic.eof()&&fic.good())
        {
            double lon ,lat ,Z;
            fic >> lat  >> lon >> Z;
            if (fic.good())
            {
                vPtImg.push_back(indirect(Pt2dr(lat,lon),altitude,vRefineCoef,rowCrop,sampCrop));
            }
        }
    }
    std::cout << "Number of points in inverse grid : "<<vPtImg.size()<<std::endl;
}



void Dimap::createGrid(std::string const &nomGrid, std::string const &nomImage,
                double stepPixel, double stepCarto,
                double rowCrop, double sampCrop,
				std::vector<double> vAltitude, std::string targetSyst, std::string inputSyst,
                       std::vector<double> vRefineCoef)const
{
    double firstSamp = first_col;
    double firstLine = first_row;
    double lastSamp  = last_col;
    double lastLine  = last_row;

    //Direct nbr Lignes et colonnes + step last ligne et colonne
    int nbLine, nbSamp;
    nbLine=(lastLine-firstLine)/stepPixel +1;
    nbSamp=(lastSamp-firstSamp)/stepPixel +1 ;

    std::vector<Pt2dr> vPtCarto;
	createDirectGrid(firstSamp, firstLine, stepPixel, nbSamp, nbLine, vAltitude, vPtCarto, targetSyst, inputSyst, vRefineCoef, rowCrop, sampCrop);

    // Estimation of the validity domaine in cartographic coordinates
	vector<Pt2dr> cornersGeo;
	cornersGeo.push_back(Pt2dr(first_lat, first_lon));
	cornersGeo.push_back(Pt2dr(first_lat, last_lon));
	cornersGeo.push_back(Pt2dr(last_lat, last_lon));
	cornersGeo.push_back(Pt2dr(last_lat, first_lon));
	vector<Pt2dr> anEmpriseCarto = empriseCarto(cornersGeo, targetSyst, inputSyst);

    //Corners of the validity domaine of the inverse RPC
	Pt2dr urc(anEmpriseCarto[1].x, anEmpriseCarto[1].y);
	Pt2dr llc(anEmpriseCarto[0].x, anEmpriseCarto[0].y);
	std::cout << "Corners of the area : " << llc << " " << urc << std::endl;

    //Indirect nbr Lignes et colonnes + step last ligne et colonne
    int nbrLine, nbrSamp;
    nbrSamp=(urc.x-llc.x)/stepCarto +1;
    nbrLine=(urc.y-llc.y)/stepCarto +1;

    std::vector<Pt2dr> vPtImg;
    //Calcul des coefficients de l'affinite pour la transformation inverse
    std::vector<double> vRefineCoefInv;

        double A0 = vRefineCoef[0];
        double A1 = vRefineCoef[1];
        double A2 = vRefineCoef[2];
        double B0 = vRefineCoef[3];
        double B1 = vRefineCoef[4];
        double B2 = vRefineCoef[5];

        double det = A1*B2 - A2*B1;

        double IA0 = -A0;
        double IA1 = B2/det;
        double IA2 = -A2/det;
        double IB0 = -B0;
        double IB1 = -B1/det;
        double IB2 = A1/det;

        vRefineCoefInv.push_back(IA0);
        vRefineCoefInv.push_back(IA1);
        vRefineCoefInv.push_back(IA2);
        vRefineCoefInv.push_back(IB0);
        vRefineCoefInv.push_back(IB1);
        vRefineCoefInv.push_back(IB2);

	createIndirectGrid(llc.x, urc.y, nbrSamp, nbrLine, stepCarto, vAltitude, vPtImg,
		targetSyst, inputSyst, vRefineCoefInv, rowCrop, sampCrop);

    //Creating grid and writing flux

    std::ofstream writeGrid(nomGrid.c_str());
    writeGrid <<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"<<std::endl;
    writeGrid <<"<trans_coord_grid version=\"5\" name=\"\">"<<std::endl;
        //creation of the date
        time_t t= time(0);
        struct tm * timeInfo =localtime(&t);
        std::string date;
        std::stringstream ssdate;
        ssdate<<timeInfo-> tm_year+1900;
        double adate []= {(double)timeInfo-> tm_mon, (double)timeInfo-> tm_mday,
                (double)timeInfo-> tm_hour,(double)timeInfo-> tm_min, (double)timeInfo-> tm_sec};
        std::vector<double> vdate (adate,adate+5);
        // Formating the date
        for (int ida=0; ida<5;ida++)
            {
                std::stringstream ssdateTempo;
                std::string dateTempo;
                ssdateTempo<<vdate[ida];
                dateTempo=ssdateTempo.str();
                if (dateTempo.length()==2)
                    ssdate<<dateTempo;
                else ssdate<<0<<dateTempo;
            }
        date=ssdate.str();
        writeGrid <<"\t<date>"<<date<<"</date>"<<std::endl;

        writeGrid <<"\t<trans_coord name=\"\">"<<std::endl;
            writeGrid <<"\t\t<trans_sys_coord name=\"\">"<<std::endl;
                writeGrid <<"\t\t\t<sys_coord name=\"sys1\">"<<std::endl;
                    writeGrid <<"\t\t\t\t<sys_coord_plani name=\"sys1\">"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<code>"<<nomImage<<"</code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<unit>"<<"p"<<"</unit>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<direct>"<<"0"<<"</direct>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<sub_code>"<<"*"<<"</sub_code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<vertical>"<<nomImage<<"</vertical>"<<std::endl;
                    writeGrid <<"\t\t\t\t</sys_coord_plani>"<<std::endl;
                    writeGrid <<"\t\t\t\t<sys_coord_alti name=\"sys1\">"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<code>"<<"Unused in MicMac"<<"</code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<unit>"<<"m"<<"</unit>"<<std::endl;
                    writeGrid <<"\t\t\t\t</sys_coord_alti>"<<std::endl;
                writeGrid <<"\t\t\t</sys_coord>"<<std::endl;

                writeGrid <<"\t\t\t<sys_coord name=\"sys2\">"<<std::endl;
                    writeGrid <<"\t\t\t\t<sys_coord_plani name=\"sys2\">"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<code>"<<"Unused in MicMac"<<"</code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<unit>"<<"m"<<"</unit>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<direct>"<<"1"<<"</direct>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<sub_code>"<<"*"<<"</sub_code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<vertical>"<<"Unused in MicMac"<<"</vertical>"<<std::endl;
                    writeGrid <<"\t\t\t\t</sys_coord_plani>"<<std::endl;
                    writeGrid <<"\t\t\t\t<sys_coord_alti name=\"sys2\">"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<code>"<<"Unused in MicMac"<<"</code>"<<std::endl;
                        writeGrid <<"\t\t\t\t\t<unit>"<<"m"<<"</unit>"<<std::endl;
                    writeGrid <<"\t\t\t\t</sys_coord_alti>"<<std::endl;
                writeGrid <<"\t\t\t</sys_coord>"<<std::endl;
            writeGrid <<"\t\t</trans_sys_coord>"<<std::endl;
            writeGrid <<"\t\t<category>"<<"1"<<"</category>"<<std::endl;
            writeGrid <<"\t\t<type_modele>"<<"2"<<"</type_modele>"<<std::endl;
            writeGrid <<"\t\t<direct_available>"<<"1"<<"</direct_available>"<<std::endl;
            writeGrid <<"\t\t<inverse_available>"<<"1"<<"</inverse_available>"<<std::endl;
        writeGrid <<"\t</trans_coord>"<<std::endl;

// For the direct grid
        writeGrid <<"\t<multi_grid version=\"1\" name=\"1-2\" >"<<std::endl;
		/*
            std::stringstream ssUL;
            std::stringstream ssStepPix;
            std::stringstream ssNumbCol;
            std::stringstream ssNumbLine;
            std::string sUpperLeft;
            std::string sStepPix;
            std::string sNumbCol;
            std::string sNumbLine;
            ssUL<<std::setprecision(15)<<firstSamp-sampCrop<<"  "<<std::setprecision(15)<<firstLine-rowCrop;
            ssStepPix<<stepPixel;
            ssNumbCol<<nbSamp;
            ssNumbLine<<nbLine;
            sUpperLeft=ssUL.str();
            sStepPix=ssStepPix.str();
            sNumbCol=ssNumbCol.str();
            sNumbLine=ssNumbLine.str();
			*/
			writeGrid << "\t\t<upper_left>" << std::setprecision(15) << firstSamp - sampCrop << "  " << std::setprecision(15) << firstLine - rowCrop << "</upper_left>" << std::endl;
			writeGrid << "\t\t<columns_interval>" << stepPixel << "</columns_interval>" << std::endl;
			writeGrid << "\t\t<rows_interval>" << "-" << stepPixel << "</rows_interval>" << std::endl;
			writeGrid << "\t\t<columns_number>" << nbSamp << "</columns_number>" << std::endl;
			writeGrid << "\t\t<rows_number>" << nbLine << "</rows_number>" << std::endl;
            writeGrid <<"\t\t<components_number>"<<"2"<<"</components_number>"<<std::endl;
            std::vector<Pt2dr>::const_iterator it = vPtCarto.begin();

            for(size_t i=0;i<vAltitude.size();++i)
                {
                        std::stringstream ssAlti;
                        std::string sAlti;
                        ssAlti<<std::setprecision(15)<<vAltitude[i];
                        sAlti=ssAlti.str();
                        writeGrid <<"\t\t\t<layer value=\""<<sAlti<<"\">"<<std::endl;

                        for(int l=0;l<nbLine;++l)
                            {
                                    for(int c = 0;c<nbSamp;++c)
                                        {
                                            Pt2dr const &PtCarto = (*it);
                                            ++it;
                                            std::stringstream ssCoord;
                                            std::string  sCoord;
                                            ssCoord<<std::setprecision(15)<<PtCarto.x<<"   "<<std::setprecision(15)<<PtCarto.y;
                                            sCoord=ssCoord.str();
                                            writeGrid <<"\t\t\t"<<sCoord<<std::endl;
                                        }
                            }
                        writeGrid <<"\t\t\t</layer>"<<std::endl;
                }
        writeGrid <<"\t</multi_grid>"<<std::endl;

// For the inverse grid
        writeGrid <<"\t<multi_grid version=\"1\" name=\"2-1\" >"<<std::endl;
		/*
            std::stringstream ssULInv;
            std::stringstream ssStepCarto;
            std::stringstream ssNumbColInv;
            std::stringstream ssNumbLineInv;
            std::string sUpperLeftInv;
            std::string sStepCarto;
            std::string sNumbColInv;
            std::string sNumbLineInv;
            ssULInv<<std::setprecision(15)<<vPtCarto[0].x<<"  "<<std::setprecision(15)<<vPtCarto[0].y;
            ssStepCarto<<std::setprecision(15)<<stepCarto;
            ssNumbColInv<<nbrSamp;
            ssNumbLineInv<<nbrLine;
            sUpperLeftInv=ssULInv.str();
            sStepCarto=ssStepCarto.str();
            sNumbColInv=ssNumbColInv.str();
            sNumbLineInv=ssNumbLineInv.str();
			*/
			writeGrid << "\t\t<upper_left>" << std::setprecision(15) << vPtCarto[0].x << "  " << std::setprecision(15) << vPtCarto[0].y << "</upper_left>" << std::endl;
			writeGrid << "\t\t<columns_interval>" << std::setprecision(15) << stepCarto << "</columns_interval>" << std::endl;
			writeGrid << "\t\t<rows_interval>" << std::setprecision(15) << stepCarto << "</rows_interval>" << std::endl;
			writeGrid << "\t\t<columns_number>" << nbrSamp << "</columns_number>" << std::endl;
			writeGrid << "\t\t<rows_number>" << nbrLine << "</rows_number>" << std::endl;
            writeGrid <<"\t\t<components_number>"<<"2"<<"</components_number>"<<std::endl;
            std::vector<Pt2dr>::const_iterator it2 = vPtImg.begin();

            for(size_t i=0;i<vAltitude.size();++i)
                {
                    std::stringstream ssAlti;
                    std::string sAlti;
                    ssAlti<<std::setprecision(15)<<vAltitude[i];
                    sAlti=ssAlti.str();
                    writeGrid <<"\t\t\t<layer value=\""<<sAlti<<"\">"<<std::endl;

                    for(int l=0;l<nbrLine;++l)
                        {
                            for(int c = 0;c<nbrSamp;++c)
                                {
                                    Pt2dr const &PtImg = (*it2);
                                    ++it2;
                                    std::stringstream ssCoordInv;
                                    std::string  sCoordInv;
                                    ssCoordInv<<std::setprecision(15)<<PtImg.x - sampCrop<<"   "
                                        <<std::setprecision(15)<<PtImg.y - rowCrop;
                                    sCoordInv=ssCoordInv.str();
                                    writeGrid <<"\t\t\t"<<sCoordInv<<std::endl;
                                }
                        }
                    writeGrid <<"\t\t\t</layer>"<<std::endl;
                }
        writeGrid <<"\t</multi_grid>"<<std::endl;

    writeGrid <<"</trans_coord_grid>"<<std::endl;
 }

//Lecture du fichier DIMAP
Dimap::Dimap(std::string const &filename)
{
    direct_samp_num_coef.clear();
    direct_samp_den_coef.clear();
    direct_line_num_coef.clear();
    direct_line_den_coef.clear();

    indirect_samp_num_coef.clear();
    indirect_samp_den_coef.clear();
    indirect_line_num_coef.clear();
    indirect_line_den_coef.clear();

    cElXMLTree tree(filename.c_str());

    {
        std::list<cElXMLTree*> noeuds=tree.GetAll(std::string("Direct_Model"));
        std::list<cElXMLTree*>::iterator it_grid,fin_grid=noeuds.end();


        std::string coefSampN="SAMP_NUM_COEFF";
        std::string coefSampD="SAMP_DEN_COEFF";
        std::string coefLineN="LINE_NUM_COEFF";
        std::string coefLineD="LINE_DEN_COEFF";

        for (int c=1; c<21;c++)
        {
            std::stringstream ss;
            ss<<"_"<<c;
            coefSampN.append(ss.str());
            coefSampD.append(ss.str());
            coefLineN.append(ss.str());
            coefLineD.append(ss.str());
            for(it_grid=noeuds.begin();it_grid!=fin_grid;++it_grid)
            {
                double value;
                std::istringstream buffer((*it_grid)->GetUnique(coefSampN.c_str())->GetUniqueVal());
                buffer >> value;
                direct_samp_num_coef.push_back(value);
                std::istringstream buffer2((*it_grid)->GetUnique(coefSampD.c_str())->GetUniqueVal());
                buffer2 >> value;
                direct_samp_den_coef.push_back(value);
                std::istringstream buffer3((*it_grid)->GetUnique(coefLineN.c_str())->GetUniqueVal());
                buffer3 >> value;
                direct_line_num_coef.push_back(value);
                std::istringstream buffer4((*it_grid)->GetUnique(coefLineD.c_str())->GetUniqueVal());
                buffer4 >> value;
                direct_line_den_coef.push_back(value);
            }
            coefSampN=coefSampN.substr(0,14);
            coefSampD=coefSampD.substr(0,14);
            coefLineN=coefLineN.substr(0,14);
            coefLineD=coefLineD.substr(0,14);
        }
        for(it_grid=noeuds.begin();it_grid!=fin_grid;++it_grid)
        {
            std::istringstream buffer((*it_grid)->GetUnique("ERR_BIAS_X")->GetUniqueVal());
            buffer >> dirErrBiasX;
            std::istringstream bufferb((*it_grid)->GetUnique("ERR_BIAS_Y")->GetUniqueVal());
            bufferb >> dirErrBiasY;
        }
    }

    {
        std::list<cElXMLTree*> noeudsInv=tree.GetAll(std::string("Inverse_Model"));
        std::list<cElXMLTree*>::iterator it_gridInd,fin_gridInd=noeudsInv.end();

        std::string coefSampN="SAMP_NUM_COEFF";
        std::string coefSampD="SAMP_DEN_COEFF";
        std::string coefLineN="LINE_NUM_COEFF";
        std::string coefLineD="LINE_DEN_COEFF";

        for (int c=1; c<21;c++)
        {
            double value;
            std::stringstream ss;
            ss<<"_"<<c;
            coefSampN.append(ss.str());
            coefSampD.append(ss.str());
            coefLineN.append(ss.str());
            coefLineD.append(ss.str());
            for(it_gridInd=noeudsInv.begin();it_gridInd!=fin_gridInd;++it_gridInd)
            {
                std::istringstream bufferInd((*it_gridInd)->GetUnique(coefSampN.c_str())->GetUniqueVal());
                bufferInd >> value;
                indirect_samp_num_coef.push_back(value);
                std::istringstream bufferInd2((*it_gridInd)->GetUnique(coefSampD.c_str())->GetUniqueVal());
                bufferInd2 >> value;
                indirect_samp_den_coef.push_back(value);
                std::istringstream bufferInd3((*it_gridInd)->GetUnique(coefLineN.c_str())->GetUniqueVal());
                bufferInd3 >> value;
                indirect_line_num_coef.push_back(value);
                std::istringstream bufferInd4((*it_gridInd)->GetUnique(coefLineD.c_str())->GetUniqueVal());
                bufferInd4 >> value;
                indirect_line_den_coef.push_back(value);
            }
            coefSampN=coefSampN.substr(0,14);
            coefSampD=coefSampD.substr(0,14);
            coefLineN=coefLineN.substr(0,14);
            coefLineD=coefLineD.substr(0,14);
        }
        for(it_gridInd=noeudsInv.begin();it_gridInd!=fin_gridInd;++it_gridInd)
        {
            std::istringstream buffer((*it_gridInd)->GetUnique("ERR_BIAS_ROW")->GetUniqueVal());
            buffer >> indirErrBiasRow;
            std::istringstream bufferb((*it_gridInd)->GetUnique("ERR_BIAS_COL")->GetUniqueVal());
            bufferb >> indirErrBiasCol;
        }
    }

    {
        std::list<cElXMLTree*> noeudsRFM=tree.GetAll(std::string("RFM_Validity"));
        std::list<cElXMLTree*>::iterator it_gridRFM,fin_gridRFM=noeudsRFM.end();

        {
            std::list<cElXMLTree*> noeudsInv=tree.GetAll(std::string("Direct_Model_Validity_Domain"));
            std::list<cElXMLTree*>::iterator it_gridInd,fin_gridInd=noeudsInv.end();


            for(it_gridInd=noeudsInv.begin();it_gridInd!=fin_gridInd;++it_gridInd)
            {
                std::istringstream bufferInd((*it_gridInd)->GetUnique("FIRST_ROW")->GetUniqueVal());
                bufferInd >> first_row;
                std::istringstream bufferInd2((*it_gridInd)->GetUnique("FIRST_COL")->GetUniqueVal());
                bufferInd2 >> first_col;
                std::istringstream bufferInd3((*it_gridInd)->GetUnique("LAST_ROW")->GetUniqueVal());
                bufferInd3 >> last_row;
                std::istringstream bufferInd4((*it_gridInd)->GetUnique("LAST_COL")->GetUniqueVal());
                bufferInd4 >> last_col;
            }
        }


        {
            std::list<cElXMLTree*> noeudsInv=tree.GetAll(std::string("Inverse_Model_Validity_Domain"));
            std::list<cElXMLTree*>::iterator it_gridInd,fin_gridInd=noeudsInv.end();

            for(it_gridInd=noeudsInv.begin();it_gridInd!=fin_gridInd;++it_gridInd)
            {
                std::istringstream bufferInd((*it_gridInd)->GetUnique("FIRST_LON")->GetUniqueVal());
                bufferInd >> first_lon;
                std::istringstream bufferInd2((*it_gridInd)->GetUnique("FIRST_LAT")->GetUniqueVal());
                bufferInd2 >> first_lat;
                std::istringstream bufferInd3((*it_gridInd)->GetUnique("LAST_LON")->GetUniqueVal());
                bufferInd3 >> last_lon;
                std::istringstream bufferInd4((*it_gridInd)->GetUnique("LAST_LAT")->GetUniqueVal());
                bufferInd4 >> last_lat;
            }
        }
        for(it_gridRFM=noeudsRFM.begin();it_gridRFM!=fin_gridRFM;++it_gridRFM)
        {
            std::istringstream buffer((*it_gridRFM)->GetUnique("LONG_SCALE")->GetUniqueVal());
            buffer>> long_scale;
            std::istringstream buffer2((*it_gridRFM)->GetUnique("LONG_OFF")->GetUniqueVal());
            buffer2 >> long_off;
            std::istringstream buffer3((*it_gridRFM)->GetUnique("LAT_SCALE")->GetUniqueVal());
            buffer3 >> lat_scale;
            std::istringstream buffer4((*it_gridRFM)->GetUnique("LAT_OFF")->GetUniqueVal());
            buffer4 >> lat_off;
            std::istringstream buffer5((*it_gridRFM)->GetUnique("HEIGHT_SCALE")->GetUniqueVal());
            buffer5 >> height_scale;
            std::istringstream buffer6((*it_gridRFM)->GetUnique("HEIGHT_OFF")->GetUniqueVal());
            buffer6 >> height_off;
            std::istringstream buffer7((*it_gridRFM)->GetUnique("SAMP_SCALE")->GetUniqueVal());
            buffer7 >> samp_scale;
            std::istringstream buffer8((*it_gridRFM)->GetUnique("SAMP_OFF")->GetUniqueVal());
            buffer8 >> samp_off;
            std::istringstream buffer9((*it_gridRFM)->GetUnique("LINE_SCALE")->GetUniqueVal());
            buffer9 >> line_scale;
            std::istringstream buffer10((*it_gridRFM)->GetUnique("LINE_OFF")->GetUniqueVal());
            buffer10 >> line_off;
        }
    }
}



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

    int rowCrop  = 0;
    int sampCrop = 0;

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
                 << EAM(sampCrop,"sampCrop",true,"upper left samp - crop")
                 << EAM(rowCrop,"rowCrop",true,"upper left row - crop")
                 << EAM(refineCoef,"refineCoef",true,"File of Coef to refine Grid")
     );

    // fichier GRID en sortie
    std::string aNameFileGrid = StdPrefixGen(aNameImage)+".GRI";

    Dimap dimap(aNameFileDimap);
    dimap.info();

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

    dimap.clearing(aNameFileGrid, refine);
    dimap.createGrid(aNameFileGrid,aNameImage,
                     stepPixel,stepCarto,
                     rowCrop, sampCrop,
                     vAltitude,targetSyst,inputSyst,vRefineCoef);

    return EXIT_SUCCESS;
}

