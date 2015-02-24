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
	//For DigitalGlobe data
	void ReconstructValidity();
	void Inverse2Direct();

};

//From Image coordinates to geographic
Pt3dr RPC::DirectRPC(Pt2dr Pimg, double altitude)
{
	//Computing coordinates for normalized image
	double Y = (Pimg.y - line_off) / line_scale;
	double X = (Pimg.x - samp_off) / samp_scale;
	double Z = (altitude - height_off) / height_scale;
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
	Pt3dr Pgeo;
	if ((lat_den != 0) && (long_den != 0))
	{
		Pgeo.x = (lat_num / lat_den) * lat_scale + lat_off;
		Pgeo.y = (long_num / long_den) * long_scale + long_off;
		Pgeo.z = altitude;
	}
	else
	{
		std::cout << "Computing error - denominator = 0" << std::endl;
	}
	return Pgeo;
}

//From geographic coordinates to image
Pt3dr RPC::InverseRPC(Pt3dr Pgeo)
{

	//Computing coordinates for normalized image
	double X = (Pgeo.x - lat_off) / lat_scale;
	double Y = (Pgeo.y - long_off) / long_scale;
	double Z = (Pgeo.z - height_off) / height_scale;
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
	Pt3dr Pimg;
	if ((samp_den != 0) && (line_den != 0))
	{
		Pimg.x = (samp_num / samp_den) * samp_scale + samp_off;
		Pimg.y = (line_num / line_den) * line_scale + line_off;
		Pimg.z = Z;
	}
	else
	{
		std::cout << "Computing error - denominator = 0" << std::endl;
	}
	return Pimg;
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

void RPC::Inverse2Direct()
{
	//Generating a 20*20 grid on the validity space with random heights
	vector<Pt3dr> aGridGeoNorm, aGridImNorm;
	for (u_int i = 0; i < 20; i++)
	{
		for (u_int j = 0; j < 20; j++)
		{
			Pt3dr aPt((double(i) - 10) / 10, (double(j) - 10) / 10, double(rand() % 200 - 100) / 100);
			aGridGeoNorm.push_back(aPt);
		}
	}


	//Converting the points in image space
	for (u_int i = 0; i < aGridGeoNorm.size(); i++)
	{
		aGridImNorm.push_back(InverseRPC(aGridGeoNorm[i]));
	}
	std::cout << aGridImNorm << endl;

	
}

int RPC_main(int argc, char ** argv)
{
	RPC Banane;
	Banane.Inverse2Direct();
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