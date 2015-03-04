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
	RPC() :
		indirErrBiasRow(0),
		indirErrBiasCol(0),
		dirErrBiasX(0),
		dirErrBiasY(0)
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
	void ReadRPB(std::string const &filename);
	void ReconstructValidity();
	void Inverse2Direct(double gridSize);

};

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