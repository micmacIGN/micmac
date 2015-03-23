#include "StdAfx.h"
#include "RPC.h"

//reads the txt file with lattice point in im coordinates (unit : pixels)
vector<vector<Pt2dr> > ReadLatticeIm(string aTxtImage)
{
	vector<vector<Pt2dr> > aMatIm;
	vector<double> Xs, Ys;
	//Reading the files
	std::ifstream fic(aTxtImage.c_str());
	u_int X_count = 0;
	while (!fic.eof() && fic.good() && X_count<11)
	{
		double X;
		fic >> X;
		//std::cout << "X=" << X << endl;
		Xs.push_back(X);
		X_count++;
	}
	while (!fic.eof() && fic.good())
	{
		double Y;
		fic >> Y;
		//std::cout << "Y=" << Y << endl;
		Ys.push_back(Y);
	}
	for (u_int i = 0; i < Ys.size(); i++)
	{
		vector<Pt2dr> aVectPts;
		for (u_int j = 0; j < Xs.size(); j++)
		{
			Pt2dr aPt(Xs[j], Ys[i]);
			aVectPts.push_back(aPt);
		}
		aMatIm.push_back(aVectPts);
		//std::cout << aMatIm[i] << endl;
	}

	std::cout << "Loaded " << aMatIm.size()*aMatIm[0].size() << " lattice points in image coordinates" << endl;

	return aMatIm;
}

//reads the txt file with lattice point in geocentric coordinates and transforms them into geodetic (unit : degrees)
vector<vector<Pt3dr> > ReadLatticeGeo(string aTxtLong, string aTxtLat)
{
	//Reading the file
	vector<vector<Pt3dr> > aMatGeo;
	//Reading longitudes of lattice pts
	std::ifstream fic1(aTxtLong.c_str());
	while (!fic1.eof() && fic1.good())
	{
		double L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11;
		fic1 >> L1 >> L2 >> L3 >> L4 >> L5 >> L6 >> L7 >> L8 >> L9 >> L10 >> L11;
		vector<Pt3dr> aVectPts;
		//for (u_int i = 0; i < 11; i++)
		//{
		//		
		//}
		Pt3dr aPt1(L1, 0.0, 0.0); aVectPts.push_back(aPt1);
		Pt3dr aPt2(L2, 0.0, 0.0); aVectPts.push_back(aPt2);
		Pt3dr aPt3(L3, 0.0, 0.0); aVectPts.push_back(aPt3);
		Pt3dr aPt4(L4, 0.0, 0.0); aVectPts.push_back(aPt4);
		Pt3dr aPt5(L5, 0.0, 0.0); aVectPts.push_back(aPt5);
		Pt3dr aPt6(L6, 0.0, 0.0); aVectPts.push_back(aPt6);
		Pt3dr aPt7(L7, 0.0, 0.0); aVectPts.push_back(aPt7);
		Pt3dr aPt8(L8, 0.0, 0.0); aVectPts.push_back(aPt8);
		Pt3dr aPt9(L9, 0.0, 0.0); aVectPts.push_back(aPt9);
		Pt3dr aPt10(L10, 0.0, 0.0); aVectPts.push_back(aPt10);
		Pt3dr aPt11(L11, 0.0, 0.0); aVectPts.push_back(aPt11);

		aMatGeo.push_back(aVectPts);
	}
	//Reading latitudes of lattice pts
	std::ifstream fic2(aTxtLat.c_str());
	u_int i = 0;
	while (!fic2.eof() && fic2.good())
	{
		double L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11;
		fic2 >> L1 >> L2 >> L3 >> L4 >> L5 >> L6 >> L7 >> L8 >> L9 >> L10 >> L11;
		double WGSCorFact = 0.99330562;
		//cout << setprecision(15) << "pi = " << M_PI << endl;
		//geocentric->geodetic
		aMatGeo[i][0].y = atan(tan(L1 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][1].y = atan(tan(L2 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][2].y = atan(tan(L3 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][3].y = atan(tan(L4 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][4].y = atan(tan(L5 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][5].y = atan(tan(L6 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][6].y = atan(tan(L7 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][7].y = atan(tan(L8 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][8].y = atan(tan(L9 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][9].y = atan(tan(L10 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		aMatGeo[i][10].y = atan(tan(L11 *M_PI / 180) / WGSCorFact) * 180 / M_PI;
		//std::cout << "Ligne " << i << " : " << aMatGeo[i] << endl;
		i++;
	}

	std::cout << "Loaded " << aMatGeo.size()*aMatGeo[0].size() << " lattice points in geodetic coordinates" << endl;

	return aMatGeo;
}

//reads the txt file with the satellite positions
vector<vector<Pt3dr> > ReadSatPos(string aTxtSatPos)
{
	//Reading the file
	vector<vector<Pt3dr> > aSatPos;
	std::ifstream fic2(aTxtSatPos.c_str());
	u_int i = 0;
	while (!fic2.eof() && fic2.good())
	{
		double X, Y, Z;
		fic2 >> X >> Y >> Z;

		Pt3dr aPt(X, Y, Z);
		//std::cout << i << " " << aPt << endl;
		vector<Pt3dr> aVectPts;

		for (u_int j = 0; j < 11; j++)
			aVectPts.push_back(aPt);

		aSatPos.push_back(aVectPts);
		//std::cout << i << " " << aSatPos[i] << endl;
		i++;
	}
	cout << "Loaded " << aSatPos.size() << " satellite position points in ECEF coordinates" << endl;

	return aSatPos;
}

int Aster2Grid_main(int argc, char ** argv)
{
	//GET PSEUDO-RPC2D FOR ASTER FROM LATTICE POINTS
	std::string aTxtImage, aTxtLong, aTxtLat, aTxtSatPos;
	std::string aFileOut = "RPC2D-params.xml";
	double aHMin = 0, aHMax = 3000;
	//Reading the arguments
	ElInitArgMain
		(
		argc, argv,
		LArgMain()
		<< EAMC(aTxtImage, "txt file contaning the lattice pointd in image coordinates", eSAM_IsPatFile)
		<< EAMC(aTxtLong, "txt file contaning the longitude of the lattice points", eSAM_IsPatFile)
		<< EAMC(aTxtLat, "txt file contaning the geocentric latitude of the lattice points", eSAM_IsPatFile)
		<< EAMC(aTxtSatPos, "txt file contaning the satellite position in ECEF", eSAM_IsPatFile),
		LArgMain()
		<< EAM(aFileOut, "Out", true, "Output xml file with RPC2D coordinates")
		<< EAM(aHMin, "HMin", true, "Min elipsoid height of scene (default=0)")
		<< EAM(aHMax, "HMax", true, "Max elipsoid height of scene (default=3000)")
		);

	//TODO : READ THIS FROM HDF
	//Reading files
	vector<vector<Pt2dr> > aMatPtsIm = ReadLatticeIm(aTxtImage);// cout << aMatPtsIm << endl;
	vector<vector<Pt3dr> > aMatPtsGeo = ReadLatticeGeo(aTxtLong, aTxtLat);// cout << aMatPtsGeo << endl;
	vector<vector<Pt3dr> > aMatSatPos = ReadSatPos(aTxtSatPos);

	RPC2D aRPC2D;
	RPC aRPC3D;
	cout << "Computing RPC2D" << endl;
	aRPC2D.ComputeRPC2D(aMatPtsIm, aMatPtsGeo, aHMax, aHMin);
	aRPC2D.info();

	////Test

	//Pt3dr aPtTest3(-150.9131, 63.3609, 683);
	//Pt2dr aPtDir3 = aRPC2D.InverseRPC2D(aPtTest3, 0, 0);
	//cout << "Canyon = " << aPtDir3 << endl;
	//Pt3dr aPtTest2(-150.5422, 63.1457, 3311);
	//Pt2dr aPtDir2 = aRPC2D.InverseRPC2D(aPtTest2, 0, 0);
	//cout << "East mountain = " << aPtDir2 << endl;
	//Pt3dr aPtTest(-151.0321, 63.2009, 2097);
	//Pt2dr aPtDir = aRPC2D.InverseRPC2D(aPtTest, 0, 0);
	//cout << "West mountain = " << aPtDir << endl;
	//Pt3dr aPtTest4(aMatPtsGeo[3][3].x, aMatPtsGeo[3][3].y, 0);
	//Pt2dr aPtDir4 = aRPC2D.InverseRPC2D(aPtTest4, 0, 0);
	//cout << "Lattice (3,3) Geo : " << setprecision(15) << aMatPtsGeo[3][3] << endl;
	//cout << "Lattice (3,3) Ima : " << aMatPtsIm[3][3] << endl;
	//cout << "LatticePT33 = " << aPtDir4 << endl;
	//aPtTest.x = (aPtTest.x - aRPC2D.long_off) / aRPC2D.long_scale;
	//aPtTest.y = (aPtTest.y - aRPC2D.lat_off) / aRPC2D.lat_scale;
	//aPtTest.z = (aPtTest.z - aRPC2D.height_off) / aRPC2D.height_scale;
	//Pt3dr aPt = aRPC2D.InversePreRPCNorm(aPtTest, aMatPtsGeo, aMatSatPos);
	//cout << aPt << endl;

	//Creating a folder for intemediate files
	ELISE_fp::MkDirSvp("processing");

	//Generate ramdom points in geodetic
	cout << "Generating ramdom points in normalized geodetic coordinates" << endl;
	vector<Pt3dr> aVectGeoNorm = aRPC3D.GenerateRandNormGrid(49);
	//Filtering points out of image
	cout << "Generated points : " << aVectGeoNorm.size() << endl;
	aVectGeoNorm = aRPC2D.filterOutOfBound(aVectGeoNorm, aMatPtsGeo);
	cout << "Filtered points : " << aVectGeoNorm.size() << endl;

	//Compute their image coord with pre-RPC method
	cout << "Converting points points in normalized image coordinates" << endl;

	vector<Pt3dr> aVectImNorm;
	for (u_int i = 0; i < aVectGeoNorm.size(); i++)
	{
		Pt3dr aPt = aRPC2D.InversePreRPCNorm(aVectGeoNorm[i], aMatPtsGeo, aMatSatPos);
		//cout << "Original point : " << aGridGeoNorm[i] << endl;
		//cout << "Inverted point : " << aPt << endl;
		aVectImNorm.push_back(aPt);
	}

	//Compute Direct and Inverse RPC
	aRPC3D.Validity2Dto3D(aRPC2D);
	aRPC3D.GCP2Direct(aVectGeoNorm, aVectImNorm);
	cout << "Direct RPC estimated" << endl;
	aRPC3D.GCP2Inverse(aVectGeoNorm, aVectImNorm);
	cout << "Inverse RPC estimated" << endl;
	aRPC3D.ReconstructValidity();
	aRPC3D.info();

	//Testing the reconstructed RPC
	Pt3dr aPtTestGeo(aMatPtsGeo[3][3].x, aMatPtsGeo[3][3].y, 0);
	Pt3dr aPtTestIma(aMatPtsIm[3][3].x, aMatPtsIm[3][3].y, 0);
	//cout << "Point Test Geo    : " << aPtTest4 << endl;
	double noRefine[] = { 0, 1, 0, 0, 0, 1 };
	vector<double> vRefineCoef;
	for (int i = 0; i<6; i++)
	{
		vRefineCoef.push_back(noRefine[i]);
	}
	Pt3dr aPtInverse3D = aRPC3D.InverseRPC(aPtTestGeo, vRefineCoef);
	Pt3dr aPtDirect3D = aRPC3D.DirectRPC(aPtTestIma);
	Pt2dr aPtInverse2D = aRPC2D.InverseRPC2D(aPtTestGeo, 0, 0);

	cout << "Lattice (3,3) Geo         = " << setprecision(15) << aMatPtsGeo[3][3] << endl;
	cout << "LatticePT33 RPC2D         = " << aPtInverse2D << endl;
	cout << "LatticePT33 Inverse RPC3D = " << aPtInverse3D << endl;
	cout << "Lattice (3,3) Ima         = " << aMatPtsIm[3][3] << endl;
	cout << "LatticePT33 Direct RPC3D  = " << aPtDirect3D << endl;

	Pt3dr aPtTest1(-150.9131, 63.3609, 683);//Canyon
	Pt2dr aPtInverse2D1 = aRPC2D.InverseRPC2D(aPtTest1, 0, 0);
	//Pt3dr aPtInverse2D3D1 = aRPC2D.InversePreRPCNorm(aPtTest1, aMatPtsGeo, aMatSatPos);
	Pt3dr aPtInverse3D1 = aRPC3D.InverseRPC(aPtTest1, vRefineCoef);
	Pt3dr aPtTest2(-150.5422, 63.1457, 3311);//East Mountain
	Pt2dr aPtInverse2D2 = aRPC2D.InverseRPC2D(aPtTest2, 0, 0);
	//Pt3dr aPtInverse2D3D2 = aRPC2D.InversePreRPCNorm(aPtTest2, aMatPtsGeo, aMatSatPos);
	Pt3dr aPtInverse3D2 = aRPC3D.InverseRPC(aPtTest2, vRefineCoef);
	Pt3dr aPtTest3(-151.0321, 63.2009, 2097);//West Mountain
	Pt2dr aPtInverse2D3 = aRPC2D.InverseRPC2D(aPtTest3, 0, 0);
	//Pt3dr aPtInverse2D3D3 = aRPC2D.InversePreRPCNorm(aPtTest3, aMatPtsGeo, aMatSatPos);
	Pt3dr aPtInverse3D3 = aRPC3D.InverseRPC(aPtTest3, vRefineCoef);
	cout << "Canyon Geo           = " << aPtTest1 << endl;
	cout << "Canyon Ima           = " << "[2111.3,1654.99]" << endl;
	cout << "Canyon RPC2D         = " << aPtInverse2D1 << endl;
	//cout << "Canyon RPC2D3D       = " << aPtInverse2D3D1 << endl;
	cout << "Canyon RPC3D         = " << aPtInverse3D1 << endl;

	cout << "East mountain Geo    = " << aPtTest2 << endl;
	cout << "East mountain Ima    = " << "[3719.5,2605.6]" << endl;
	cout << "East mountain RPC2D  = " << aPtInverse2D2 << endl;
	//cout << "East mountain RPC2D3D= " << aPtInverse2D3D2 << endl;
	cout << "East mountain RPC3D  = " << aPtInverse3D2 << endl;

	cout << "West mountain Geo    = " << aPtTest3 << endl;
	cout << "West mountain Ima    = " << "[2090.5,2856.1]" << endl;
	cout << "West mountain RPC2D  = " << aPtInverse2D3 << endl;
	//cout << "West mountain RPC2D3D= " << aPtInverse2D3D3 << endl;
	cout << "West mountain RPC3D  = " << aPtInverse3D3 << endl;



	return 0;
}