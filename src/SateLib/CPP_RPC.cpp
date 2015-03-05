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

MicMa cis an open source software specialized in image matching
for research in geographic information. MicMac is built on the
eLiSe image library. MicMac is governed by the  "Cecill-B licence".
See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"
#include <algorithm>
#include "hassan/reechantillonnage.h"
#include "RPC.h"



//RPC2Grid transforms a loaded RPC to a .GRI (and GRIBin) file
int RPC::RPC2Grid(int nbLayers, int altiMin, int altiMax, std::string refineCoef, std::string aNameIm, double stepPixel, double stepCarto, std::string targetSyst, std::string inputSyst, bool binaire)
{
	//Creation d'un dossier pour les fichiers intermediaires
	ELISE_fp::MkDirSvp("processing");

	//Creation du fichier de coef par defaut (grille non affinee)
	std::ofstream ficWrite(refineCoef.c_str());
	ficWrite << std::setprecision(15);
	ficWrite << 0 << " " << 1 << " " << 0 << " " << 0 << " " << 0 << " " << 1 << " " << std::endl;

	// fichier GRID en sortie
	std::string aNameFileGrid = StdPrefix(aNameIm) + ".GRI";

	std::vector<double> vAltitude;
	for (int i = 0; i<nbLayers; ++i)
		vAltitude.push_back(altiMin + i*(altiMax - altiMin) / (nbLayers - 1));

	//recuperation des coefficients pour affiner le modele
	std::vector<double> vRefineCoef;
	std::ifstream ficRead(refineCoef.c_str());
	while (!ficRead.eof() && ficRead.good())
	{
		double a0, a1, a2, b0, b1, b2;
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
	std::cout << "coef " << vRefineCoef[0] << " " << vRefineCoef[1] << " " << vRefineCoef[2]
	<< " " << vRefineCoef[3] << " " << vRefineCoef[4] << " " << vRefineCoef[5] << " " << std::endl;


	//Test si le modele est affine pour l'appellation du fichier de sortie
	bool refine = false;
	double noRefine[] = { 0, 1, 0, 0, 0, 1 };

	for (int i = 0; i<6; i++)
	{
		if (vRefineCoef[i] != noRefine[i])
			refine = true;
	}

	if (refine)
	{
		//Effacement du fichier de coefficients (affinite=identite) par defaut
		if (ifstream(refineCoef.c_str())) ELISE_fp::RmFile(refineCoef.c_str());

		//New folder
		std::string dir = "refine_" + StdPrefix(aNameIm);
		ELISE_fp::MkDirSvp(dir);

		std::cout << "Model is affine" << std::endl;
		aNameFileGrid = dir + ELISE_CAR_DIR + aNameFileGrid;
	}

	clearing(aNameFileGrid, refine);
	createGrid(aNameFileGrid, aNameIm,
		stepPixel, stepCarto,
		vAltitude, targetSyst, inputSyst, vRefineCoef);

	if (binaire)
	{
		string cmd = MMDir() + "bin/mm3d Gri2Bin " + aNameFileGrid + " " + aNameFileGrid + "Bin";
		system_call(cmd.c_str());
	}
	return EXIT_SUCCESS;
}

//From Image coordinates to geographic
Pt3dr RPC::DirectRPC(Pt3dr Pimg)const
{
	Pt3dr PimgNorm;
	//Converting into normalized coordinates
	PimgNorm.x = (Pimg.x - samp_off) / samp_scale;
	PimgNorm.y = (Pimg.y - line_off) / line_scale;
	PimgNorm.z = (Pimg.z - height_off) / height_scale;

	//Applying direct RPC
	Pt3dr PgeoNorm = DirectRPCNorm(PimgNorm);

	//Converting into real coordinates
	Pt3dr Pgeo;
	Pgeo.x = PgeoNorm.x * lat_scale + lat_off;
	Pgeo.y = PgeoNorm.y * long_scale + long_off;
	Pgeo.z = PgeoNorm.z * height_scale + height_off;

	return Pgeo;
}

Pt3dr RPC::DirectRPCNorm(Pt3dr PimgNorm)const
	{
	double X = PimgNorm.x, Y = PimgNorm.y, Z = PimgNorm.z;
	double vecteurD[] = { 1, X, Y, Z, Y*X, X*Z, Y*Z, X*X, Y*Y, Z*Z, X*Y*Z, X*X*X, Y*Y*X, X*Z*Z, X*X*Y, Y*Y*Y, Y*Z*Z, X*X*Z, Y*Y*Z, Z*Z*Z };

	double long_den = 0.;
	double long_num = 0.;
	double lat_den = 0.;
	double lat_num = 0.;

	for (int i = 0; i<20; i++)
	{
		lat_num += vecteurD[i] * direct_line_num_coef[i];
		lat_den += vecteurD[i] * direct_line_den_coef[i];
		long_num += vecteurD[i] * direct_samp_num_coef[i];
		long_den += vecteurD[i] * direct_samp_den_coef[i];
	}

	//Final computation
	Pt3dr PgeoNorm;
	if ((lat_den != 0) && (long_den != 0))
	{
		PgeoNorm.x = (lat_num / lat_den);
		PgeoNorm.y = (long_num / long_den);
		PgeoNorm.z = Z;
	}
	else
	{
		std::cout << "Computing error - denominator = 0" << std::endl;
	}
	return PgeoNorm;
}

//From geographic (LAT, LONG, Z) coordinates to image
Pt3dr RPC::InverseRPC(Pt3dr Pgeo, std::vector<double> vRefineCoef)const
{
	Pt3dr PgeoNorm;
	//Converting into normalized coordinates
	PgeoNorm.x = (Pgeo.x - lat_off) / lat_scale;
	PgeoNorm.y = (Pgeo.y - long_off) / long_scale;
	PgeoNorm.z = (Pgeo.z - height_off) / height_scale;

	//Applying inverse RPC
	Pt3dr PimNorm = InverseRPCNorm(PgeoNorm);

	///Converting into Real Coordinates
	Pt3dr Pimg;
	Pimg.x = PimNorm.x * samp_scale + samp_off;
	Pimg.y = PimNorm.y * line_scale + line_off;
	Pimg.z = Pgeo.z;

	Pt3dr PimgRefined = ptRefined(Pimg, vRefineCoef);

	return PimgRefined;
}

Pt3dr RPC::InverseRPCNorm(Pt3dr PgeoNorm)const
{
	double X = PgeoNorm.x, Y = PgeoNorm.y, Z = PgeoNorm.z;
	double vecteurD[] = { 1, Y, X, Z, Y*X, Y*Z, X*Z, Y*Y, X*X, Z*Z, X*Y*Z, Y*Y*Y, Y*X*X, Y*Z*Z, X*Y*Y, X*X*X, X*Z*Z, Y*Y*Z, X*X*Z, Z*Z*Z };
	double samp_den = 0.;
	double samp_num = 0.;
	double line_den = 0.;
	double line_num = 0.;

	for (int i = 0; i<20; i++)
	{
		line_num += vecteurD[i] * inverse_line_num_coef[i];
		line_den += vecteurD[i] * inverse_line_den_coef[i];
		samp_num += vecteurD[i] * inverse_samp_num_coef[i];
		samp_den += vecteurD[i] * inverse_samp_den_coef[i];
	}
	//Final computation
	Pt3dr PimgNorm;
	if ((samp_den != 0) && (line_den != 0))
	{
		PimgNorm.x = (samp_num / samp_den);
		PimgNorm.y = (line_num / line_den);
		PimgNorm.z = PgeoNorm.z;
	}
	else
	{
		std::cout << "Computing error - denominator = 0" << std::endl;
	}
	return PimgNorm;
}

void RPC::ReadDimap(std::string const &filename)
{
	direct_samp_num_coef.clear();
	direct_samp_den_coef.clear();
	direct_line_num_coef.clear();
	direct_line_den_coef.clear();

	inverse_samp_num_coef.clear();
	inverse_samp_den_coef.clear();
	inverse_line_num_coef.clear();
	inverse_line_den_coef.clear();

	cElXMLTree tree(filename.c_str());

	{
		std::list<cElXMLTree*> noeuds = tree.GetAll(std::string("Direct_Model"));
		std::list<cElXMLTree*>::iterator it_grid, fin_grid = noeuds.end();


		std::string coefSampN = "SAMP_NUM_COEFF";
		std::string coefSampD = "SAMP_DEN_COEFF";
		std::string coefLineN = "LINE_NUM_COEFF";
		std::string coefLineD = "LINE_DEN_COEFF";

		for (int c = 1; c<21; c++)
		{
			std::stringstream ss;
			ss << "_" << c;
			coefSampN.append(ss.str());
			coefSampD.append(ss.str());
			coefLineN.append(ss.str());
			coefLineD.append(ss.str());
			for (it_grid = noeuds.begin(); it_grid != fin_grid; ++it_grid)
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
			coefSampN = coefSampN.substr(0, 14);
			coefSampD = coefSampD.substr(0, 14);
			coefLineN = coefLineN.substr(0, 14);
			coefLineD = coefLineD.substr(0, 14);
		}
		for (it_grid = noeuds.begin(); it_grid != fin_grid; ++it_grid)
		{
			std::istringstream buffer((*it_grid)->GetUnique("ERR_BIAS_X")->GetUniqueVal());
			buffer >> dirErrBiasX;
			std::istringstream bufferb((*it_grid)->GetUnique("ERR_BIAS_Y")->GetUniqueVal());
			bufferb >> dirErrBiasY;
		}
	}

	{
		std::list<cElXMLTree*> noeudsInv = tree.GetAll(std::string("Inverse_Model"));
		std::list<cElXMLTree*>::iterator it_gridInd, fin_gridInd = noeudsInv.end();

		std::string coefSampN = "SAMP_NUM_COEFF";
		std::string coefSampD = "SAMP_DEN_COEFF";
		std::string coefLineN = "LINE_NUM_COEFF";
		std::string coefLineD = "LINE_DEN_COEFF";

		for (int c = 1; c<21; c++)
		{
			double value;
			std::stringstream ss;
			ss << "_" << c;
			coefSampN.append(ss.str());
			coefSampD.append(ss.str());
			coefLineN.append(ss.str());
			coefLineD.append(ss.str());
			for (it_gridInd = noeudsInv.begin(); it_gridInd != fin_gridInd; ++it_gridInd)
			{
				std::istringstream bufferInd((*it_gridInd)->GetUnique(coefSampN.c_str())->GetUniqueVal());
				bufferInd >> value;
				inverse_samp_num_coef.push_back(value);
				std::istringstream bufferInd2((*it_gridInd)->GetUnique(coefSampD.c_str())->GetUniqueVal());
				bufferInd2 >> value;
				inverse_samp_den_coef.push_back(value);
				std::istringstream bufferInd3((*it_gridInd)->GetUnique(coefLineN.c_str())->GetUniqueVal());
				bufferInd3 >> value;
				inverse_line_num_coef.push_back(value);
				std::istringstream bufferInd4((*it_gridInd)->GetUnique(coefLineD.c_str())->GetUniqueVal());
				bufferInd4 >> value;
				inverse_line_den_coef.push_back(value);
			}
			coefSampN = coefSampN.substr(0, 14);
			coefSampD = coefSampD.substr(0, 14);
			coefLineN = coefLineN.substr(0, 14);
			coefLineD = coefLineD.substr(0, 14);
		}
		for (it_gridInd = noeudsInv.begin(); it_gridInd != fin_gridInd; ++it_gridInd)
		{
			std::istringstream buffer((*it_gridInd)->GetUnique("ERR_BIAS_ROW")->GetUniqueVal());
			buffer >> indirErrBiasRow;
			std::istringstream bufferb((*it_gridInd)->GetUnique("ERR_BIAS_COL")->GetUniqueVal());
			bufferb >> indirErrBiasCol;
		}
	}

	{
		std::list<cElXMLTree*> noeudsRFM = tree.GetAll(std::string("RFM_Validity"));
		std::list<cElXMLTree*>::iterator it_gridRFM, fin_gridRFM = noeudsRFM.end();

		{
			std::list<cElXMLTree*> noeudsInv = tree.GetAll(std::string("Direct_Model_Validity_Domain"));
			std::list<cElXMLTree*>::iterator it_gridInd, fin_gridInd = noeudsInv.end();


			for (it_gridInd = noeudsInv.begin(); it_gridInd != fin_gridInd; ++it_gridInd)
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
			std::list<cElXMLTree*> noeudsInv = tree.GetAll(std::string("Inverse_Model_Validity_Domain"));
			std::list<cElXMLTree*>::iterator it_gridInd, fin_gridInd = noeudsInv.end();

			for (it_gridInd = noeudsInv.begin(); it_gridInd != fin_gridInd; ++it_gridInd)
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
		for (it_gridRFM = noeudsRFM.begin(); it_gridRFM != fin_gridRFM; ++it_gridRFM)
		{
			std::istringstream buffer((*it_gridRFM)->GetUnique("LONG_SCALE")->GetUniqueVal());
			buffer >> long_scale;
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

vector<Pt2dr> RPC::empriseCarto(vector<Pt2dr> Pgeo, std::string targetSyst, std::string inputSyst)const
{
	std::ofstream fic("processing/conv_ptGeo.txt");
	fic << std::setprecision(15);
	for (unsigned int i = 0; i<Pgeo.size(); i++)
	{
		fic << Pgeo[i].y << " " << Pgeo[i].x << endl;
	}
	// transformation in the ground coordinate system
	std::string command;
	command = g_externalToolHandler.get("cs2cs").callName() + " " + inputSyst + " +to " + targetSyst + " -s processing/conv_ptGeo.txt > processing/conv_ptCarto.txt";
	int res = system(command.c_str());
	ELISE_ASSERT(res == 0, " error calling cs2cs in ptGeo2Carto ");
	// loading the coordinate of the converted point
	vector<double> PtsCartox, PtsCartoy;
	std::ifstream fic2("processing/conv_ptCarto.txt");
	while (!fic2.eof() && fic2.good())
	{
		double X, Y, Z;
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

Pt3dr RPC::ptRefined(Pt3dr Pimg, std::vector<double> vRefineCoef)const
{
	//Pour calculer les coordonnees affinees d'un point
	Pt3dr pImgRefined;

	pImgRefined.x = vRefineCoef[0] + Pimg.x * vRefineCoef[1] + Pimg.y * vRefineCoef[2];
	pImgRefined.y = vRefineCoef[3] + Pimg.x * vRefineCoef[4] + Pimg.y * vRefineCoef[5];
	pImgRefined.z = Pimg.z;

	return pImgRefined;
}

void RPC::createDirectGrid(double ulcSamp, double ulcLine,
	double stepPixel,
	int nbSamp, int  nbLine,
	std::vector<double> const &vAltitude,
	std::vector<Pt2dr> &vPtCarto, std::string targetSyst, std::string inputSyst,
	std::vector<double> vRefineCoef)const
{
	vPtCarto.clear();
	// On cree un fichier de points geographiques pour les transformer avec proj4
	{
		std::ofstream fic("processing/direct_ptGeo.txt");
		fic << std::setprecision(15);
		for (size_t i = 0; i<vAltitude.size(); ++i)
		{
			double altitude = vAltitude[i];
			for (int l = 0; l<nbLine; ++l)
			{
				for (int c = 0; c<nbSamp; ++c)
				{
					Pt3dr Pimg(ulcSamp + c * stepPixel, ulcLine + l * stepPixel, altitude);

					//pour affiner les coordonnees
					Pt3dr PimgRefined = ptRefined(Pimg, vRefineCoef);

					Pt3dr Pgeo = DirectRPC(PimgRefined);
					fic << Pgeo.y << " " << Pgeo.x << std::endl;
				}
			}
		}
	}
	// transformation in the ground coordinate system
	std::string command;
	command = g_externalToolHandler.get("cs2cs").callName() + " " + inputSyst + " +to " + targetSyst + " -s processing/direct_ptGeo.txt > processing/direct_ptCarto.txt";
	int res = system(command.c_str());
	if (res != 0) std::cout << "error calling cs2cs in createDirectGrid" << std::endl;
	// loading points
	std::ifstream fic("processing/direct_ptCarto.txt");
	while (!fic.eof() && fic.good())
	{
		double X, Y, Z;
		fic >> Y >> X >> Z;
		if (fic.good())
			vPtCarto.push_back(Pt2dr(X, Y));
	}
	std::cout << "Number of points in direct grid : " << vPtCarto.size() << std::endl;
}

void RPC::createInverseGrid(double ulcX, double ulcY, int nbrSamp, int nbrLine,
	double stepCarto, std::vector<double> const &vAltitude,
	std::vector<Pt3dr> &vPtImg, std::string targetSyst, std::string inputSyst,
	std::vector<double> vRefineCoef)const
{
	vPtImg.clear();

	//Creation of a file with points in cartographic coordinates to be transformed with proj4
	{
		std::ofstream fic("processing/inverse_ptCarto.txt");
		fic << std::setprecision(15);
		for (int l = 0; l<nbrLine; ++l)
		{
			double Y = ulcY - l*stepCarto;
			for (int c = 0; c<nbrSamp; ++c)
			{
				double X = ulcX + c*stepCarto;
				fic << X << " " << Y << std::endl;
			}
		}
	}
	// convert to geographic coordinates
	std::string command;
	command = g_externalToolHandler.get("cs2cs").callName() + " " + targetSyst + " +to " + inputSyst + " -f %.12f -s processing/inverse_ptCarto.txt >processing/inverse_ptGeo.txt";
	int res = system(command.c_str());
	ELISE_ASSERT(res == 0, "error calling cs2cs in createinverseGrid");
	for (size_t i = 0; i<vAltitude.size(); ++i)
	{
		double altitude = vAltitude[i];
		// loading points
		std::ifstream fic("processing/inverse_ptGeo.txt");
		while (!fic.eof() && fic.good())
		{
			double lon, lat, Z;
			fic >> lat >> lon >> Z;
			if (fic.good())
			{
				vPtImg.push_back(InverseRPC(Pt3dr(lat, lon, altitude), vRefineCoef));
			}
		}
	}
	std::cout << "Number of points in inverse grid : " << vPtImg.size() << std::endl;
}

void RPC::createGrid(std::string const &nomGrid, std::string const &nomImage,
	double stepPixel, double stepCarto,
	std::vector<double> vAltitude, std::string targetSyst, std::string inputSyst,
	std::vector<double> vRefineCoef)
{
	double firstSamp = first_col;
	double firstLine = first_row;
	double lastSamp = last_col;
	double lastLine = last_row;

	//Direct nbr Lignes et colonnes + step last ligne et colonne
	int nbLine, nbSamp;
	nbLine = (lastLine - firstLine) / stepPixel + 1;
	nbSamp = (lastSamp - firstSamp) / stepPixel + 1;

	std::vector<Pt2dr> vPtCarto;
	createDirectGrid(firstSamp, firstLine, stepPixel, nbSamp, nbLine, vAltitude, vPtCarto, targetSyst, inputSyst, vRefineCoef);

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

	//inverse nbr Lignes et colonnes + step last ligne et colonne
	int nbrLine, nbrSamp;
	nbrSamp = (urc.x - llc.x) / stepCarto + 1;
	nbrLine = (urc.y - llc.y) / stepCarto + 1;

	std::vector<Pt3dr> vPtImg;
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
	double IA1 = B2 / det;
	double IA2 = -A2 / det;
	double IB0 = -B0;
	double IB1 = -B1 / det;
	double IB2 = A1 / det;

	vRefineCoefInv.push_back(IA0);
	vRefineCoefInv.push_back(IA1);
	vRefineCoefInv.push_back(IA2);
	vRefineCoefInv.push_back(IB0);
	vRefineCoefInv.push_back(IB1);
	vRefineCoefInv.push_back(IB2);

	createInverseGrid(llc.x, urc.y, nbrSamp, nbrLine, stepCarto, vAltitude, vPtImg,
		targetSyst, inputSyst, vRefineCoefInv);

	//Creating grid and writing flux

	std::ofstream writeGrid(nomGrid.c_str());
	writeGrid << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
	writeGrid << "<trans_coord_grid version=\"5\" name=\"\">" << std::endl;
	//creation of the date
	time_t t = time(0);
	struct tm * timeInfo = localtime(&t);
	std::string date;
	std::stringstream ssdate;
	ssdate << timeInfo->tm_year + 1900;
	double adate[] = { (double)timeInfo->tm_mon, (double)timeInfo->tm_mday,
		(double)timeInfo->tm_hour, (double)timeInfo->tm_min, (double)timeInfo->tm_sec };
	std::vector<double> vdate(adate, adate + 5);
	// Formating the date
	for (int ida = 0; ida<5; ida++)
	{
		std::stringstream ssdateTempo;
		std::string dateTempo;
		ssdateTempo << vdate[ida];
		dateTempo = ssdateTempo.str();
		if (dateTempo.length() == 2)
			ssdate << dateTempo;
		else ssdate << 0 << dateTempo;
	}
	date = ssdate.str();
	writeGrid << "\t<date>" << date << "</date>" << std::endl;

	writeGrid << "\t<trans_coord name=\"\">" << std::endl;
	writeGrid << "\t\t<trans_sys_coord name=\"\">" << std::endl;
	writeGrid << "\t\t\t<sys_coord name=\"sys1\">" << std::endl;
	writeGrid << "\t\t\t\t<sys_coord_plani name=\"sys1\">" << std::endl;
	writeGrid << "\t\t\t\t\t<code>" << nomImage << "</code>" << std::endl;
	writeGrid << "\t\t\t\t\t<unit>" << "p" << "</unit>" << std::endl;
	writeGrid << "\t\t\t\t\t<direct>" << "0" << "</direct>" << std::endl;
	writeGrid << "\t\t\t\t\t<sub_code>" << "*" << "</sub_code>" << std::endl;
	writeGrid << "\t\t\t\t\t<vertical>" << nomImage << "</vertical>" << std::endl;
	writeGrid << "\t\t\t\t</sys_coord_plani>" << std::endl;
	writeGrid << "\t\t\t\t<sys_coord_alti name=\"sys1\">" << std::endl;
	writeGrid << "\t\t\t\t\t<code>" << "Unused in MicMac" << "</code>" << std::endl;
	writeGrid << "\t\t\t\t\t<unit>" << "m" << "</unit>" << std::endl;
	writeGrid << "\t\t\t\t</sys_coord_alti>" << std::endl;
	writeGrid << "\t\t\t</sys_coord>" << std::endl;

	writeGrid << "\t\t\t<sys_coord name=\"sys2\">" << std::endl;
	writeGrid << "\t\t\t\t<sys_coord_plani name=\"sys2\">" << std::endl;
	writeGrid << "\t\t\t\t\t<code>" << "Unused in MicMac" << "</code>" << std::endl;
	writeGrid << "\t\t\t\t\t<unit>" << "m" << "</unit>" << std::endl;
	writeGrid << "\t\t\t\t\t<direct>" << "1" << "</direct>" << std::endl;
	writeGrid << "\t\t\t\t\t<sub_code>" << "*" << "</sub_code>" << std::endl;
	writeGrid << "\t\t\t\t\t<vertical>" << "Unused in MicMac" << "</vertical>" << std::endl;
	writeGrid << "\t\t\t\t</sys_coord_plani>" << std::endl;
	writeGrid << "\t\t\t\t<sys_coord_alti name=\"sys2\">" << std::endl;
	writeGrid << "\t\t\t\t\t<code>" << "Unused in MicMac" << "</code>" << std::endl;
	writeGrid << "\t\t\t\t\t<unit>" << "m" << "</unit>" << std::endl;
	writeGrid << "\t\t\t\t</sys_coord_alti>" << std::endl;
	writeGrid << "\t\t\t</sys_coord>" << std::endl;
	writeGrid << "\t\t</trans_sys_coord>" << std::endl;
	writeGrid << "\t\t<category>" << "1" << "</category>" << std::endl;
	writeGrid << "\t\t<type_modele>" << "2" << "</type_modele>" << std::endl;
	writeGrid << "\t\t<direct_available>" << "1" << "</direct_available>" << std::endl;
	writeGrid << "\t\t<inverse_available>" << "1" << "</inverse_available>" << std::endl;
	writeGrid << "\t</trans_coord>" << std::endl;

	// For the direct grid
	writeGrid << "\t<multi_grid version=\"1\" name=\"1-2\" >" << std::endl;
	writeGrid << "\t\t<upper_left>" << std::setprecision(15) << firstSamp << "  " << std::setprecision(15) << firstLine << "</upper_left>" << std::endl;
	writeGrid << "\t\t<columns_interval>" << stepPixel << "</columns_interval>" << std::endl;
	writeGrid << "\t\t<rows_interval>" << "-" << stepPixel << "</rows_interval>" << std::endl;
	writeGrid << "\t\t<columns_number>" << nbSamp << "</columns_number>" << std::endl;
	writeGrid << "\t\t<rows_number>" << nbLine << "</rows_number>" << std::endl;
	writeGrid << "\t\t<components_number>" << "2" << "</components_number>" << std::endl;
	std::vector<Pt2dr>::const_iterator it = vPtCarto.begin();

	for (size_t i = 0; i<vAltitude.size(); ++i)
	{
		std::stringstream ssAlti;
		std::string sAlti;
		ssAlti << std::setprecision(15) << vAltitude[i];
		sAlti = ssAlti.str();
		writeGrid << "\t\t\t<layer value=\"" << sAlti << "\">" << std::endl;

		for (int l = 0; l<nbLine; ++l)
		{
			for (int c = 0; c<nbSamp; ++c)
			{
				Pt2dr const &PtCarto = (*it);
				++it;
				std::stringstream ssCoord;
				std::string  sCoord;
				ssCoord << std::setprecision(15) << PtCarto.x << "   " << std::setprecision(15) << PtCarto.y;
				sCoord = ssCoord.str();
				writeGrid << "\t\t\t" << sCoord << std::endl;
			}
		}
		writeGrid << "\t\t\t</layer>" << std::endl;
	}
	writeGrid << "\t</multi_grid>" << std::endl;

	// For the inverse grid
	writeGrid << "\t<multi_grid version=\"1\" name=\"2-1\" >" << std::endl;
	writeGrid << "\t\t<upper_left>" << std::setprecision(15) << vPtCarto[0].x << "  " << std::setprecision(15) << vPtCarto[0].y << "</upper_left>" << std::endl;
	writeGrid << "\t\t<columns_interval>" << std::setprecision(15) << stepCarto << "</columns_interval>" << std::endl;
	writeGrid << "\t\t<rows_interval>" << std::setprecision(15) << stepCarto << "</rows_interval>" << std::endl;
	writeGrid << "\t\t<columns_number>" << nbrSamp << "</columns_number>" << std::endl;
	writeGrid << "\t\t<rows_number>" << nbrLine << "</rows_number>" << std::endl;
	writeGrid << "\t\t<components_number>" << "2" << "</components_number>" << std::endl;
	std::vector<Pt3dr>::const_iterator it2 = vPtImg.begin();

	for (size_t i = 0; i<vAltitude.size(); ++i)
	{
		std::stringstream ssAlti;
		std::string sAlti;
		ssAlti << std::setprecision(15) << vAltitude[i];
		sAlti = ssAlti.str();
		writeGrid << "\t\t\t<layer value=\"" << sAlti << "\">" << std::endl;

		for (int l = 0; l<nbrLine; ++l)
		{
			for (int c = 0; c<nbrSamp; ++c)
			{
				Pt3dr const &PtImg = (*it2);
				++it2;
				std::stringstream ssCoordInv;
				std::string  sCoordInv;
				ssCoordInv << std::setprecision(15) << PtImg.x << "   "
					<< std::setprecision(15) << PtImg.y ;
				sCoordInv = ssCoordInv.str();
				writeGrid << "\t\t\t" << sCoordInv << std::endl;
			}
		}
		writeGrid << "\t\t\t</layer>" << std::endl;
	}
	writeGrid << "\t</multi_grid>" << std::endl;

	writeGrid << "</trans_coord_grid>" << std::endl;
}

void RPC::WriteAirbusRPC(std::string aFileOut)
{
	std::ofstream fic(aFileOut.c_str());
	fic << std::setprecision(16);

	fic << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>" << endl;
	fic << "<Dimap_Document>" << endl;
	fic << "\t<Metadata_Identification>" << endl;
	fic << "\t\t<METADATA_FORMAT version=\"2.0\">DIMAP</METADATA_FORMAT>" << endl;
	fic << "\t\t<METADATA_PROFILE>PHR_SENSOR</METADATA_PROFILE>" << endl;
	fic << "\t\t<METADATA_SUBPROFILE>RPC</METADATA_SUBPROFILE>" << endl;
	fic << "\t\t<METADATA_LANGUAGE>en</METADATA_LANGUAGE>" << endl;
	fic << "\t</Metadata_Identification>" << endl;
	fic << "\t<Rational_Function_Model>" << endl;
	fic << "\t\t<Resource_Reference>" << endl;
	fic << "\t\t\t<RESOURCE_TITLE version=\"2.1\">NITF</RESOURCE_TITLE>" << endl;
	fic << "\t\t\t<RESOURCE_ID>RPC00B</RESOURCE_ID>" << endl;
	fic << "\t\t</Resource_Reference>" << endl;
	fic << "\t\t<Global_RFM>" << endl;
	fic << "\t\t\t<Direct_Model>" << endl;

		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<SAMP_NUM_COEFF_" << i + 1 << ">" << direct_samp_num_coef[i] << "</SAMP_NUM_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<SAMP_DEN_COEFF_" << i + 1 << ">" << direct_samp_den_coef[i] << "</SAMP_DEN_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<LINE_NUM_COEFF_" << i + 1 << ">" << direct_line_num_coef[i] << "</LINE_NUM_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<LINE_DEN_COEFF_" << i + 1 << ">" << direct_line_den_coef[i] << "</LINE_DEN_COEFF_" << i + 1 << ">" << endl;
		}
		fic << "\t\t\t\t<ERR_BIAS_X>" << dirErrBiasX << "</ERR_BIAS_X>" << endl;
		fic << "\t\t\t\t<ERR_BIAS_Y>" << dirErrBiasY << "</ERR_BIAS_Y>" << endl;
	fic << "\t\t\t</Direct_Model>" << endl;
	fic << "\t\t\t<Inverse_Model>" << endl;
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<SAMP_NUM_COEFF_" << i + 1 << ">" << inverse_samp_num_coef[i] << "</SAMP_NUM_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<SAMP_DEN_COEFF_" << i + 1 << ">" << inverse_samp_den_coef[i] << "</SAMP_DEN_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<LINE_NUM_COEFF_" << i + 1 << ">" << inverse_line_num_coef[i] << "</LINE_NUM_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<LINE_DEN_COEFF_" << i + 1 << ">" << inverse_line_den_coef[i] << "</LINE_DEN_COEFF_" << i + 1 << ">" << endl;
		}
		fic << "\t\t\t\t<ERR_BIAS_ROW>" << indirErrBiasRow << "</ERR_BIAS_ROW>" << endl;
		fic << "\t\t\t\t<ERR_BIAS_COL>" << indirErrBiasCol << "</ERR_BIAS_COL>" << endl;
	fic << "\t\t\t</Inverse_Model>" << endl;

	fic << "\t\t\t<RFM_Validity>" << endl;
		fic << "\t\t\t\t<Direct_Model_Validity_Domain>" << endl;
		fic << "\t\t\t\t\t<FIRST_ROW>" << first_row << "</FIRST_ROW>" << endl;
		fic << "\t\t\t\t\t<FIRST_COL>" << first_col << "</FIRST_COL>" << endl;
		fic << "\t\t\t\t\t<LAST_ROW>" << last_row << "</LAST_ROW>" << endl;
		fic << "\t\t\t\t\t<LAST_COL>" << last_col << "</LAST_COL>" << endl;
		fic << "\t\t\t\t</Direct_Model_Validity_Domain>" << endl;
		fic << "\t\t\t\t<Inverse_Model_Validity_Domain>" << endl;
		fic << "\t\t\t\t\t<FIRST_LON>" << first_lon << "</FIRST_LON>" << endl;
		fic << "\t\t\t\t\t<FIRST_LAT>" << first_lat << "</FIRST_LAT>" << endl;
		fic << "\t\t\t\t\t<LAST_LON>" << last_lon << "</LAST_LON>" << endl;
		fic << "\t\t\t\t\t<LAST_LAT>" << last_lat << "</LAST_LAT>" << endl;
		fic << "\t\t\t\t</Inverse_Model_Validity_Domain>" << endl;			
		fic << "\t\t\t\t<LONG_SCALE>" << long_scale << "</LONG_SCALE>" << endl;
		fic << "\t\t\t\t<LONG_OFF>" << long_off << "</LONG_OFF>" << endl;
		fic << "\t\t\t\t<LAT_SCALE>" << lat_scale << "</LAT_SCALE>" << endl;
		fic << "\t\t\t\t<LAT_OFF>" << lat_off << "</LAT_OFF>" << endl;
		fic << "\t\t\t\t<HEIGHT_SCALE>" << height_scale << "</HEIGHT_SCALE>" << endl;
		fic << "\t\t\t\t<HEIGHT_OFF>" << height_off << "</HEIGHT_OFF>" << endl;
		fic << "\t\t\t\t<SAMP_SCALE>" << samp_scale << "</SAMP_SCALE>" << endl;
		fic << "\t\t\t\t<SAMP_OFF>" << samp_off << "</SAMP_OFF>" << endl;
		fic << "\t\t\t\t<LINE_SCALE>" << line_scale << "</LINE_SCALE>" << endl;
		fic << "\t\t\t\t<LINE_OFF>" << line_off << "</LINE_OFF>" << endl;
	fic << "\t\t\t</RFM_Validity>" << endl;

	fic << "\t\t</Global_RFM>" << endl;
	fic << "\t</Rational_Function_Model>" << endl;
	fic << "</Dimap_Document>";
}

void RPC::ReconstructValidity()
{
	first_row = -1 * line_scale + line_off;
	first_col = -1 * samp_scale + samp_off;
	last_row = 1 * line_scale + line_off;
	last_col = 1 * samp_scale + samp_off;

	first_lon = -1 * long_scale + long_off;
	first_lat = -1 * lat_scale + lat_off;
	first_height = -1 * height_scale + height_off;
	last_lon = 1 * long_scale + long_off;
	last_lat = 1 * lat_scale + lat_off;
	last_height = 1 * height_scale + height_off;

}

void RPC::Inverse2Direct(int gridSize)
{
	//Cleaning potential data in RPC object
	direct_samp_num_coef.clear();
	direct_samp_den_coef.clear();
	direct_line_num_coef.clear();
	direct_line_den_coef.clear();

	//Generating a 20*20 grid on the normalized space with random normalized heights
	vector<Pt3dr> aGridGeoNorm, aGridImNorm;
	srand(time(NULL));//Initiate the rand value
	for (u_int i = 0; i <= gridSize; i++)
	{
		for (u_int j = 0; j <= gridSize; j++)
		{
			Pt3dr aPt;
			aPt.x = (double(i) - (gridSize / 2)) / (gridSize / 2);
			aPt.y = (double(j) - (gridSize / 2)) / (gridSize / 2);
			aPt.z = double(rand() % 2000 - 1000) / 1000;
			aGridGeoNorm.push_back(aPt);
		}
	}
	//cout << aGridGeoNorm << endl;
	//Converting the points to image space
	for (u_int i = 0; i < aGridGeoNorm.size(); i++)
	{
		aGridImNorm.push_back(InverseRPCNorm(aGridGeoNorm[i]));
	}
	//cout << aGridImNorm << endl;
	//cout << "Im grid Computed" << endl;


	//Parameters too get parameters of P1 and P2 in ---  lon=P1(column,row,Z)/P2(column,row,Z)  --- where (column,row,Z) are image coordinates (idem for lat)
	//To simplify notations : Column->X and Row->Y
	//Function is 0=Poly1(X,Y,Z)-long*Poly2(X,Y,Z) with poly 3rd degree (up to X^3,Y^3,Z^3,XXY,XXZ,XYY,XZZ,YYZ,YZZ)
	//First param (cst) of Poly2=1 to avoid sol=0

	L2SysSurResol aSysLon(39), aSysLat(39);

	//For all lattice points
	for (u_int i = 0; i<aGridGeoNorm.size(); i++){

		//Simplifying notations
		double X = aGridImNorm[i].x;
		double Y = aGridImNorm[i].y;
		double Z = aGridImNorm[i].z;
		double lat = aGridGeoNorm[i].x;
		double lon = aGridGeoNorm[i].y;

		double aEqLon[39] = {
							1, X, Y, Z, X*Y, X*Z, Y*Z, X*X, Y*Y, Z*Z, Y*X*Z, X*X*X, X*Y*Y, X*Z*Z, Y*X*X, Y*Y*Y, Y*Z*Z, X*X*Z, Y*Y*Z, Z*Z*Z,
							-lon*X, -lon*Y, -lon*Z, -lon*X*Y, -lon*X*Z, -lon*Y*Z, -lon*X*X, -lon*Y*Y, -lon*Z*Z, -lon*Y*X*Z, -lon*X*X*X, -lon*X*Y*Y, -lon*X*Z*Z, -lon*Y*X*X, -lon*Y*Y*Y, -lon*Y*Z*Z, -lon*X*X*Z, -lon*Y*Y*Z, -lon*Z*Z*Z
							};
		aSysLon.AddEquation(1, aEqLon, lon);


		double aEqLat[39] = {
							1, X, Y, Z, X*Y, X*Z, Y*Z, X*X, Y*Y, Z*Z, Y*X*Z, X*X*X, X*Y*Y, X*Z*Z, Y*X*X, Y*Y*Y, Y*Z*Z, X*X*Z, Y*Y*Z, Z*Z*Z,
							-lat*X, -lat*Y, -lat*Z, -lat*X*Y, -lat*X*Z, -lat*Y*Z, -lat*X*X, -lat*Y*Y, -lat*Z*Z, -lat*Y*X*Z, -lat*X*X*X, -lat*X*Y*Y, -lat*X*Z*Z, -lat*Y*X*X, -lat*Y*Y*Y, -lat*Y*Z*Z, -lat*X*X*Z, -lat*Y*Y*Z, -lat*Z*Z*Z
							};
		aSysLat.AddEquation(1, aEqLat, lat);
	}

	//Computing the result
	bool Ok;
	Im1D_REAL8 aSolLon = aSysLon.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolLat = aSysLat.GSSR_Solve(&Ok);
	double* aDataLat = aSolLat.data();
	double* aDataLon = aSolLon.data();

	//Copying Data in RPC object
	//Numerators
	for (int i = 0; i<20; i++)
	{
		direct_samp_num_coef.push_back(aDataLon[i]);
		direct_line_num_coef.push_back(aDataLat[i]);
	}
	//Denominators (first one = 1)
	direct_line_den_coef.push_back(1);
	direct_samp_den_coef.push_back(1);
	for (int i = 20; i<39; i++)
	{
		direct_samp_den_coef.push_back(aDataLon[i]);
		direct_line_den_coef.push_back(aDataLat[i]);
	}
}

void RPC::ReadRPB(std::string const &filename)
{
	std::ifstream RPBfile(filename.c_str());
	ELISE_ASSERT(RPBfile.good(), " RPB file not found ");
	std::string line;
	std::string a, b;
	//Pass 6 lines
	for (u_int i = 0; i < 6; i++)
		std::getline(RPBfile, line);
	//Line Offset
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> line_off;
	}
	//Samp Offset
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> samp_off;
	}
	//Lat Offset
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> lat_off;
	}
	//Lon Offset
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> long_off;
	}
	//Height Offset
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> height_off;
	}
	//Line Scale
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> line_scale;
	}
	//Samp Scale
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> samp_scale;
	}
	//Lat Scale
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> lat_scale;
	}
	//Lon Scale
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> long_scale;
	}
	//Height Scale
	{
		std::getline(RPBfile, line);
		std::istringstream iss(line);
		iss >> a >> b >> height_scale;
	}
	double aCoef;
	//inverse_line_num_coef
	{
		std::getline(RPBfile, line);
		for (u_int i = 0; i < 20; i++)
		{
			std::getline(RPBfile, line);
			std::istringstream iss(line);
			iss >> aCoef;
			inverse_line_num_coef.push_back(aCoef);
		}
	}
	//inverse_line_den_coef
	{
		std::getline(RPBfile, line);
		for (u_int i = 0; i < 20; i++)
		{
			std::getline(RPBfile, line);
			std::istringstream iss(line);
			iss >> aCoef; 
			inverse_line_den_coef.push_back(aCoef);
		}
	}
	//inverse_samp_num_coef
	{
		std::getline(RPBfile, line);
		for (u_int i = 0; i < 20; i++)
		{
			std::getline(RPBfile, line);
			std::istringstream iss(line);
			iss >> aCoef;
			inverse_samp_num_coef.push_back(aCoef);
		}
	}
	//inverse_samp_den_coef
	{
		std::getline(RPBfile, line);
		for (u_int i = 0; i < 20; i++)
		{
			std::getline(RPBfile, line);
			std::istringstream iss(line);
			iss >> aCoef;
			inverse_samp_den_coef.push_back(aCoef);
		}
	}
}

int RPC_main(int argc, char ** argv)
{
	//GET PSEUDO-RPC2D FOR ASTER FROM LATTICE POINTS
	std::string aTxtImage, aTxtCarto;
	std::string aFileOut = "RPC2D-params.xml";
	//Reading the arguments
	ElInitArgMain
		(
		argc, argv,
		LArgMain()
		<< EAMC(aTxtImage, "txt file contaning the lattice point in the image coordinates", eSAM_IsPatFile)
		<< EAMC(aTxtCarto, "txt file contaning the lattice point in the carto coordinates", eSAM_IsPatFile),
		LArgMain()
		<< EAM(aFileOut, "Out", true, "Output xml file with RPC2D coordinates")
		);

	//Reading the files
	vector<Pt2dr> aPtsIm, aPtsCarto;
	{
		std::ifstream fic(aTxtImage.c_str());
		while (!fic.eof() && fic.good())
		{
			double X, Y;
			fic >> X >> Y;
			Pt2dr aPt(X, Y);
			if (fic.good())
			{
				aPtsIm.push_back(aPt);
			}
		}
		cout << "Read " << aPtsIm.size() << " points in image coordinates" << endl;
		//cout << aPtsIm << endl;
		std::ifstream fic2(aTxtCarto.c_str());
		while (!fic2.eof() && fic2.good())
		{
			double X, Y, Z;
			fic2 >> X >> Y >> Z;
			Pt2dr aPt(X, Y);
			if (fic2.good())
			{
				aPtsCarto.push_back(aPt);
			}
		}
		cout << "Read " << aPtsCarto.size() << " points in cartographic coordinates" << endl;
		//cout << aPtsCarto << endl;
	}

	//Finding normalization parameters
	//divide Pts into X and Y
	vector<double> aPtsCartoX, aPtsCartoY, aPtsImX, aPtsImY;
	for (u_int i = 0; i < aPtsCarto.size(); i++)
	{
		aPtsCartoX.push_back(aPtsCarto[i].x);
		aPtsCartoY.push_back(aPtsCarto[i].y);
		aPtsImX.push_back(aPtsIm[i].x);
		aPtsImY.push_back(aPtsIm[i].y);
	}

	Pt2dr aPtCartoMin(*std::min_element(aPtsCartoX.begin(), aPtsCartoX.end()), *std::min_element(aPtsCartoY.begin(), aPtsCartoY.end()));
	Pt2dr aPtCartoMax(*std::max_element(aPtsCartoX.begin(), aPtsCartoX.end()), *std::max_element(aPtsCartoY.begin(), aPtsCartoY.end()));
	Pt2dr aPtImMin(*std::min_element(aPtsImX.begin(), aPtsImX.end()), *std::min_element(aPtsImY.begin(), aPtsImY.end()));
	Pt2dr aPtImMax(*std::max_element(aPtsImX.begin(), aPtsImX.end()), *std::max_element(aPtsImY.begin(), aPtsImY.end()));
	Pt2dr aCartoScale((aPtCartoMax.x - aPtCartoMin.x) / 2, (aPtCartoMax.y - aPtCartoMin.y) / 2);
	Pt2dr aImScale((aPtImMax.x - aPtImMin.x) / 2, (aPtImMax.y - aPtImMin.y) / 2);
	Pt2dr aCartoOffset(aPtCartoMin.x + (aPtCartoMax.x - aPtCartoMin.x) / 2, aPtCartoMin.y + (aPtCartoMax.y - aPtCartoMin.y) / 2);
	Pt2dr aImOffset(aPtImMin.x + (aPtImMax.x - aPtImMin.x) / 2, aPtImMin.y + (aPtImMax.y - aPtImMin.y) / 2);

	//Parameters too get parameters of P1 and P2 in ---  Column=P1(X,Y)/P2(X,Y)  --- where (X,Y) are Carto coordinates (idem for Row)
	//Function is 0=Poly1(X,Y)-Column*Poly2(X,Y) ==> Column*k=a+bX+cY+dXY+eX^2+fY^2+gX^2Y+hXY^2+iX^3+jY^3-Column(lX+mY+nXY+oX^2+pY^2+qX^2Y+rXY^2+sX^3+tY^3)
	//k=1 to avoid sol=0
	L2SysSurResol aSysCol(19), aSysRow(19);

	//For all lattice points
	for (u_int i = 0; i<aPtsCarto.size(); i++){

		//NORMALIZATION
		double X = (aPtsCarto[i].x - aCartoOffset.x) / aCartoScale.x;
		double Y = (aPtsCarto[i].y - aCartoOffset.y) / aCartoScale.y;
		double COL = (aPtsIm[i].x - aImOffset.x) / aImScale.x;
		double ROW = (aPtsIm[i].y - aImOffset.y) / aImScale.y;

		double aEqCol[19] = {
			(1),
			(X),
			(Y),
			(X*Y),
			(pow(X, 2)),
			(pow(Y, 2)),
			(pow(X, 2)*Y),
			(X*pow(Y, 2)),
			(pow(X, 3)),
			(pow(Y, 3)),
			//(COL),
			(-COL*X),
			(-COL*Y),
			(-COL*X*Y),
			(-COL*pow(X, 2)),
			(-COL*pow(Y, 2)),
			(-COL*pow(X, 2)*Y),
			(-COL*X*pow(Y, 2)),
			(-COL*pow(X, 3)),
			(-COL*pow(Y, 3)),
		};
		aSysCol.AddEquation(1, aEqCol, COL);


		double aEqRow[19] = {
			(1),
			(X),
			(Y),
			(X*Y),
			(pow(X, 2)),
			(pow(Y, 2)),
			(pow(X, 2)*Y),
			(X*pow(Y, 2)),
			(pow(X, 3)),
			(pow(Y, 3)),
			//(ROW),
			(-ROW*X),
			(-ROW*Y),
			(-ROW*X*Y),
			(-ROW*pow(X, 2)),
			(-ROW*pow(Y, 2)),
			(-ROW*pow(X, 2)*Y),
			(-ROW*X*pow(Y, 2)),
			(-ROW*pow(X, 3)),
			(-ROW*pow(Y, 3)),
		};
		aSysRow.AddEquation(1, aEqRow, ROW);
	}

	//Computing the result
	bool Ok;
	Im1D_REAL8 aSolCol = aSysCol.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolRow = aSysRow.GSSR_Solve(&Ok);
	double* aDataCol = aSolCol.data();
	double* aDataRow = aSolRow.data();

	//Outputting results
	{
		std::ofstream fic(aFileOut.c_str());
		fic << std::setprecision(15);
		fic << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>" << endl;
		fic << "<RPC2D>" << endl;
		fic << "\t<RFM_Validity>" << endl;
		fic << "\t\t<Direct_Model_Validity_Domain>" << endl;
		fic << "\t\t\t<FIRST_ROW>" << aPtImMin.x << "</FIRST_ROW>" << endl;
		fic << "\t\t\t<FIRST_COL>" << aPtImMin.y << "</FIRST_COL>" << endl;
		fic << "\t\t\t<LAST_ROW>" << aPtImMax.x << "</LAST_ROW>" << endl;
		fic << "\t\t\t<LAST_COL>" << aPtImMax.y << "</LAST_COL>" << endl;
		fic << "\t\t</Direct_Model_Validity_Domain>" << endl;
		fic << "\t\t<Inverse_Model_Validity_Domain>" << endl;
		fic << "\t\t\t<FIRST_X>" << aPtCartoMin.x << "</FIRST_X>" << endl;
		fic << "\t\t\t<FIRST_Y>" << aPtCartoMin.y << "</FIRST_Y>" << endl;
		fic << "\t\t\t<LAST_X>" << aPtCartoMax.x << "</LAST_X>" << endl;
		fic << "\t\t\t<LAST_Y>" << aPtCartoMax.y << "</LAST_Y>" << endl;
		fic << "\t\t</Inverse_Model_Validity_Domain>" << endl;

		fic << "\t\t<X_SCALE>" << aCartoScale.x << "</X_SCALE>" << endl;
		fic << "\t\t<X_OFF>" << aCartoOffset.x << "</X_OFF>" << endl;
		fic << "\t\t<Y_SCALE>" << aCartoScale.y << "</Y_SCALE>" << endl;
		fic << "\t\t<Y_OFF>" << aCartoOffset.y << "</Y_OFF>" << endl;

		fic << "\t\t<SAMP_SCALE>" << aImScale.x << "</SAMP_SCALE>" << endl;
		fic << "\t\t<SAMP_OFF>" << aImOffset.x << "</SAMP_OFF>" << endl;
		fic << "\t\t<LINE_SCALE>" << aImScale.y << "</LINE_SCALE>" << endl;
		fic << "\t\t<LINE_OFF>" << aImOffset.y << "</LINE_OFF>" << endl;

		fic << "\t</RFM_Validity>" << endl;

		for (int i = 0; i<10; i++)
		{
			fic << "<COL_NUMERATOR_" << i + 1 << ">" << aDataCol[i] << "</COL_NUMERATOR_" << i + 1 << ">" << endl;
		}
		fic << "<COL_DENUMERATOR_1>1</COL_DENUMERATOR_1>" << endl;
		for (int i = 10; i<19; i++)
		{
			fic << "<COL_DENUMERATOR_" << i - 8 << ">" << aDataCol[i] << "</COL_DENUMERATOR_" << i - 8 << ">" << endl;
		}
		for (int i = 0; i<10; i++)
		{
			fic << "<ROW_NUMERATOR_" << i + 1 << ">" << aDataRow[i] << "</ROW_NUMERATOR_" << i + 1 << ">" << endl;
		}
		fic << "<ROW_DENUMERATOR_1>1</ROW_DENUMERATOR_1>" << endl;
		for (int i = 10; i<19; i++)
		{
			fic << "<ROW_DENUMERATOR_" << i - 8 << ">" << aDataRow[i] << "</ROW_DENUMERATOR_" << i - 8 << ">" << endl;
		}
		fic << "</RPC2D>" << endl;
	}
	cout << "Written functions in file " << aFileOut << endl;

	return 0;
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
