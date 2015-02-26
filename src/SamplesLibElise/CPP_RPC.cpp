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

class RPC
{
public:
	//Elements of RPC
	std::vector<double> direct_line_num_coef;
	std::vector<double> direct_line_den_coef;
	std::vector<double> direct_samp_num_coef;
	std::vector<double> direct_samp_den_coef;
	std::vector<double> indirect_line_num_coef;
	std::vector<double> indirect_line_den_coef;
	std::vector<double> indirect_samp_num_coef;
	std::vector<double> indirect_samp_den_coef;
	//Offsets and scale for inverse RPC
	double lat_off, lat_scale, long_off, long_scale, height_off, height_scale;
	//Offsets and scale for direct RPC
	double line_off, line_scale, samp_off, samp_scale;

	//Boundaries of RPC validity for image space
	double first_row, first_col, last_row, last_col;
	//Boundaries of RPC validity for geo space
	double first_lon, first_lat, first_height, last_lon, last_lat, last_height;

	//Errors indicated in DIMAP files
/*
      MPD -> in ANSI C++, variable initialisation here, not allowed
	double indirErrBiasRow=0;
	double indirErrBiasCol=0;
	double dirErrBiasX=0;
	double dirErrBiasY=0;
*/
        RPC() :
             indirErrBiasRow (0),
             indirErrBiasCol (0),
             dirErrBiasX (0),
             dirErrBiasY (0)
        {
        }
	double indirErrBiasRow;
	double indirErrBiasCol;
	double dirErrBiasX;
	double dirErrBiasY;

	Pt3dr DirectRPCNorm(Pt2dr, double);
	Pt3dr InverseRPCNorm(Pt3dr);
	Pt3dr DirectRPC(Pt2dr, double);
	Pt3dr InverseRPC(Pt3dr);

	//Showing Info
	void info()
	{
		std::cout << "RPC info:" << std::endl;
		std::cout << "===========================================================" << std::endl;
		std::cout << "long_scale   : " << long_scale << " | long_off   : " << long_off << std::endl;
		std::cout << "lat_scale    : " << lat_scale << " | lat_off    : " << lat_off << std::endl;
		std::cout << "height_scale : " << height_scale << " | height_off : " << height_off << std::endl;
		std::cout << "samp_scale   : " << samp_scale << " | samp_off   : " << samp_off << std::endl;
		std::cout << "line_scale   : " << line_scale << " | line_off   : " << line_off << std::endl;
		std::cout << "first_row    : " << first_row << " | last_row   : " << last_row << std::endl;
		std::cout << "first_col    : " << first_col << " | last_col   : " << last_col << std::endl;
		std::cout << "first_lon    : " << first_lon << " | last_lon   : " << last_lon << std::endl;
		std::cout << "first_lat    : " << first_lat << " | last_lat   : " << last_lat << std::endl;
		std::cout << "direct_samp_num_coef : " << direct_samp_num_coef.size() << std::endl;
		std::cout << "direct_samp_den_coef : " << direct_samp_den_coef.size() << std::endl;
		std::cout << "direct_line_num_coef : " << direct_line_num_coef.size() << std::endl;
		std::cout << "direct_line_den_coef : " << direct_line_den_coef.size() << std::endl;
		std::cout << "indirect_samp_num_coef : " << indirect_samp_num_coef.size() << std::endl;
		std::cout << "indirect_samp_den_coef : " << indirect_samp_den_coef.size() << std::endl;
		std::cout << "indirect_line_num_coef : " << indirect_line_num_coef.size() << std::endl;
		std::cout << "indirect_line_den_coef : " << indirect_line_den_coef.size() << std::endl;
		std::cout << "===========================================================" << std::endl;
	}

	//For Dimap
	void ReadDimap(std::string const &filename);
	void WriteAirbusRPC(std::string aFileOut);

	//For DigitalGlobe data
	// void ReadRPB(std::string const &filename); MPD -> in ANSI C++ , scope specification inside class innot allowed
	void ReadRPB(std::string const &filename);
	void ReconstructValidity();
	void Inverse2Direct(double gridSize);

};

//From Image coordinates to geographic
Pt3dr RPC::DirectRPC(Pt2dr Pimg, double altitude)
{
	Pt2dr PimgNorm;
	//Converting into normalized coordinates
	PimgNorm.x = (Pimg.x - samp_off) / samp_scale;
	PimgNorm.y = (Pimg.y - line_off) / line_scale;
	double altitudeNorm = (altitude - height_off) / height_scale;

	//Applying direct RPC
	Pt3dr PgeoNorm = DirectRPCNorm(PimgNorm, altitudeNorm);

	//Converting into real coordinates
	Pt3dr Pgeo;
	Pgeo.x = PgeoNorm.x * lat_scale + lat_off;
	Pgeo.y = PgeoNorm.y * long_scale + long_off;
	Pgeo.z = PgeoNorm.z * height_scale + height_off;

	return Pgeo;
}

Pt3dr RPC::DirectRPCNorm(Pt2dr PimgNorm, double altitudeNorm)
	{
	double X = PimgNorm.x, Y = PimgNorm.y, Z = altitudeNorm;
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
		PgeoNorm.z = altitudeNorm;
	}
	else
	{
		std::cout << "Computing error - denominator = 0" << std::endl;
	}
	return PgeoNorm;
}

//From geographic coordinates to image
Pt3dr RPC::InverseRPC(Pt3dr Pgeo)
{
	Pt3dr PgeoNorm;
	//Converting into normalized coordinates
	PgeoNorm.x = (Pgeo.x - long_off) / long_scale;
	PgeoNorm.y = (Pgeo.y - lat_off) / lat_scale;
	PgeoNorm.z = (Pgeo.z - height_off) / height_scale;

	//Applying inverse RPC
	Pt3dr PimNorm = InverseRPCNorm(PgeoNorm);

	///Converting into Real Coordinates
	Pt3dr Pimg;
	Pimg.x = PimNorm.x * samp_scale + samp_off;
	Pimg.y = PimNorm.y * line_scale + line_off;
	Pimg.z = Pgeo.z;

	return Pimg;
}

Pt3dr RPC::InverseRPCNorm(Pt3dr PgeoNorm)
{
	double X = PgeoNorm.x, Y = PgeoNorm.y, Z = PgeoNorm.z;
	double vecteurD[] = { 1, Y, X, Z, Y*X, Y*Z, X*Z, Y*Y, X*X, Z*Z, X*Y*Z, Y*Y*Y, Y*X*X, Y*Z*Z, X*Y*Y, X*X*X, X*Z*Z, Y*Y*Z, X*X*Z, Z*Z*Z };
	double samp_den = 0.;
	double samp_num = 0.;
	double line_den = 0.;
	double line_num = 0.;

	for (int i = 0; i<20; i++)
	{
		line_num += vecteurD[i] * indirect_line_num_coef[i];
		line_den += vecteurD[i] * indirect_line_den_coef[i];
		samp_num += vecteurD[i] * indirect_samp_num_coef[i];
		samp_den += vecteurD[i] * indirect_samp_den_coef[i];
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

	indirect_samp_num_coef.clear();
	indirect_samp_den_coef.clear();
	indirect_line_num_coef.clear();
	indirect_line_den_coef.clear();

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
			fic << "\t\t\t\t<SAMP_NUM_COEFF_" << i + 1 << ">" << indirect_samp_num_coef[i] << "</SAMP_NUM_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<SAMP_DEN_COEFF_" << i + 1 << ">" << indirect_samp_den_coef[i] << "</SAMP_DEN_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<LINE_NUM_COEFF_" << i + 1 << ">" << indirect_line_num_coef[i] << "</LINE_NUM_COEFF_" << i + 1 << ">" << endl;
		}
		for (int i = 0; i<20; i++)
		{
			fic << "\t\t\t\t<LINE_DEN_COEFF_" << i + 1 << ">" << indirect_line_den_coef[i] << "</LINE_DEN_COEFF_" << i + 1 << ">" << endl;
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
	fic << "</Dimap_Document>" << endl;
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

void RPC::Inverse2Direct(double gridSize)
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
		double lon = aGridGeoNorm[i].x;
		double lat = aGridGeoNorm[i].y;

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
	double* aDataLat = aSolLon.data();
	double* aDataLon = aSolLat.data();

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
	// std::ifstream RPBfile(filename);  MPD : no implicit conversion std::string -> char * in ANSI C++
	std::ifstream RPBfile(filename.c_str());
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
	//indirect_line_num_coef
	{
		std::getline(RPBfile, line);
		for (u_int i = 0; i < 20; i++)
		{
			std::getline(RPBfile, line);
			std::istringstream iss(line);
			iss >> aCoef;
			indirect_line_num_coef.push_back(aCoef);
		}
	}
	//indirect_line_den_coef
	{
		std::getline(RPBfile, line);
		for (u_int i = 0; i < 20; i++)
		{
			std::getline(RPBfile, line);
			std::istringstream iss(line);
			iss >> aCoef; 
			indirect_line_den_coef.push_back(aCoef);
		}
	}
	//indirect_samp_num_coef
	{
		std::getline(RPBfile, line);
		for (u_int i = 0; i < 20; i++)
		{
			std::getline(RPBfile, line);
			std::istringstream iss(line);
			iss >> aCoef;
			indirect_samp_num_coef.push_back(aCoef);
		}
	}
	//indirect_samp_den_coef
	{
		std::getline(RPBfile, line);
		for (u_int i = 0; i < 20; i++)
		{
			std::getline(RPBfile, line);
			std::istringstream iss(line);
			iss >> aCoef;
			indirect_samp_den_coef.push_back(aCoef);
		}
	}
}

int RPC_main(int argc, char ** argv)
{
	string aFileOut = "RPC.xml", aNameFile;
	double gridSize = 50;
	ElInitArgMain
		(
		argc, argv,
		LArgMain() << EAMC(aNameFile, "RPB DigitalGlob file"),
		LArgMain()
		<< EAM(aFileOut, "Out", true, "Name of the output file")
		<< EAM(gridSize, "GridSize", true, "Size of the grid of generated GCPs")
		);

	RPC Banane;
	Banane.ReadRPB(aNameFile);
	cout << "RPB File read" << endl;
	Banane.Inverse2Direct(gridSize);
	cout << "Direct RPC estimated" << endl;
	Banane.ReconstructValidity();
	//Banane.ReadDimap(aNameFileDimap);
	Banane.info();
	//Banane.Inverse2Direct(gridSize);
	Banane.WriteAirbusRPC(aFileOut);
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
