#include "StdAfx.h"

typedef struct{
	string name;
	int (*func)( int argc, char **argv );
} command_t;

int command_correctPlanarPolygons( int argc, char **argv );
int command_maskContent( int argc, char **argv );
int command_renameImageSet( int argc, char **argv );
int command_toto( int argc, char **argv );

command_t commands[] = {
	{ "correctplanarpolygons", &command_correctPlanarPolygons },
	{ "maskcontent", &command_maskContent },
	{ "renameimageset", &command_renameImageSet },
	{ "toto", &command_toto },
	{ "", NULL }
};

void correctPlanarPolygon( vector<Pt3dr> &aPolygon )
{
	if (aPolygon.size() < 3) ELISE_ERROR_EXIT("aPolygon.size() = " << aPolygon.size() << " < 3");
	Pt3dr u = vunit(aPolygon[1] - aPolygon[0]), minV = vunit(aPolygon[2] - aPolygon[0]);
	size_t minI = 2;
	REAL minScal = ElAbs(scal(u, minV));
	for (size_t i = 3; i < aPolygon.size(); i++)
	{
		Pt3dr v = vunit(aPolygon[i] - aPolygon[0]);
		REAL d = ElAbs(scal(u, v));
		if (d < minScal)
		{
			minScal = d;
			minI = i;
			minV = v;
		}
	}
	cout << "minI = " << minI << endl;
	cout << "minScal = " << minScal << endl;
	cout << "minV = " << minV << endl;
	Pt3dr n = u ^ minV;
	cout << "minV = " << minV << endl;
	ElMatrix<REAL> planToGlobalMatrix = MatFromCol(u, minV, n);
	if (planToGlobalMatrix.Det() < 1e-10) ELISE_ERROR_EXIT("matrix is not inversible");

	ElRotation3D planToGlobalRot(aPolygon[0], planToGlobalMatrix, true);
	ElRotation3D globalToPlanRot = planToGlobalRot.inv();

	//~ const size_t nbVertices = aPolygon.size();
	//~ ostringstream ss;
	//~ static int ii = 0;
	//~ const REAL extrudSize = 1e4;
	//~ ss << "polygon_" << (ii++) << ".ply";
	//~ ofstream f(ss.str().c_str());
	//~ f << "ply" << endl;
	//~ f << "format ascii 1.0" << endl;
	//~ f << "element vertex " << 4 * nbVertices << endl;
	//~ f << "property float x" << endl;
	//~ f << "property float y" << endl;
	//~ f << "property float z" << endl;
	//~ f << "property uchar red" << endl;
	//~ f << "property uchar green" << endl;
	//~ f << "property uchar blue" << endl;
	//~ f << "element face " << nbVertices << endl;
	//~ f << "property list uchar int vertex_indices" << endl;
	//~ f << "end_header" << endl;

	REAL zDiffSum = 0.;
	for (size_t i = 0; i < aPolygon.size(); i++)
	{
		Pt3dr p = globalToPlanRot.ImAff(aPolygon[i]);
		zDiffSum += ElAbs(p.z);
		aPolygon[i] = planToGlobalRot.ImAff(Pt3dr(p.x, p.y, 0.));

		//~ Pt3dr p0 = (i == 0) ? aPolygon[aPolygon.size() - 1] : aPolygon[i - 1], p1 = aPolygon[i], p2 = p1 + n * extrudSize, p3 = p0 + n * extrudSize;
		//~ f << p0.x << ' ' << p0.y << ' ' << p0.z << " 128 128 128" << endl;
		//~ f << p1.x << ' ' << p1.y << ' ' << p1.z << " 128 128 128" << endl;
		//~ f << p2.x << ' ' << p2.y << ' ' << p2.z << " 128 128 128" << endl;
		//~ f << p3.x << ' ' << p3.y << ' ' << p3.z << " 128 128 128" << endl;
	}

	//~ for (size_t i = 0; i < aPolygon.size(); i++)
		//~ f << 4 << ' ' << i * 4 << ' ' << i * 4 + 1 << ' ' << i * 4 + 2 << ' ' << i* 4 + 3 << endl;
	//~ f.close();
	cout << "zDiffSum = " << zDiffSum << endl;
}

void correctPlanarPolygons( vector<cItem> &aPolygons )
{
	for (size_t i = 0; i < aPolygons.size(); i++)
		correctPlanarPolygon(aPolygons[i].Pt());
}

void writePolygons( const cPolyg3D &aPolygons, const string &aFilename )
{
	cElXMLFileIn xmlFile(aFilename);
	cElXMLTree *xmlTree = ToXMLTree(aPolygons);
	xmlFile.PutTree(xmlTree);
	delete xmlTree;
}

//~ #if ELISE_QT_VERSION >=4
//~ #include "../saisieQT/include_QT/Cloud.h"
//~ #endif

int command_correctPlanarPolygons( int argc, char **argv )
{
	if (argc < 1) ELISE_ERROR_EXIT("an XML filename is needed");

	const string xmlFilename = argv[0];
	cPolyg3D polygons = StdGetFromSI(xmlFilename, Polyg3D);
	correctPlanarPolygons(polygons.Item());
	writePolygons(polygons, "corrected_polygons.xml");

	return EXIT_SUCCESS;
}


//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------

#if ELISE_QT_VERSION >=4
//~ #if 0
	#include "../saisieQT/include_QT/Cloud.h"
#endif

void getPlyBoundingBox( const string &aFilename, Pt3dr &oP0, Pt3dr &oP1 )
{
	#if ELISE_QT_VERSION >=4
	//~ #if 0
		GlCloud *ply = GlCloud::loadPly(aFilename);
		if ( !ply) ELISE_ERROR_EXIT("cannot load ply file [" << aFilename << ']');

		if (ply->size() == 0) return;

		float min[3], max[3];
		{
			QVector3D p = ply->getVertex(0).getPosition();
			min[0] = max[0] = p.x();
			min[1] = max[1] = p.y();
			min[2] = max[2] = p.z();
		}
		const int plySize = ply->size();
		for (int i = 1; i < plySize; i++)
		{
			QVector3D p = ply->getVertex(i).getPosition();

			if (p.x() < min[0]) min[0] = p.x();
			if (p.y() < min[1]) min[1] = p.y();
			if (p.z() < min[2]) min[2] = p.z();

			if (p.x() > max[0]) max[0] = p.x();
			if (p.y() > max[1]) max[1] = p.y();
			if (p.z() > max[2]) max[2] = p.z();
		}

		oP0 = Pt3dr((REAL)min[0], (REAL)min[1], (REAL)min[2]);
		oP1 = Pt3dr((REAL)max[0], (REAL)max[1], (REAL)max[2]);
	#else
		ELISE_ERROR_EXIT("getPlyBoundingBox: no Qt");
	#endif
}

void makeGrid( const Pt3dr &aP0, const Pt3dr &aSize, unsigned int aPointsPerAxis, const cMasqBin3D &aMask, list<Pt3dr> &oPoints )
{
	ELISE_DEBUG_ERROR(aSize.x <= 0. || aSize.y <= 0. || aSize.z <= 0., "makeGrid", "invalid box size = " << aSize);

	oPoints.clear();
	if (aPointsPerAxis == 0) return;

	Pt3dr pace = aSize / (REAL)(aPointsPerAxis - 1), p;
	const Pt3dr p1 = aP0 + aSize;
	REAL z = aP0.z;
	while (z <= p1.z)
	{
		REAL y = aP0.y;
		while (y <= p1.y)
		{
			REAL x = aP0.x;
			while (x <= p1.x)
			{
				p.x = x;
				p.y = y;
				p.z = z;
				if (aMask.IsInMasq(p)) oPoints.push_back(p);

				x += pace.x;
			}
			y += pace.y;
		}
		z += pace.z;
	}
}

bool writePly( const list<Pt3dr> &aPoints, const string &aFilename )
{
	ofstream f(aFilename.c_str());

	if ( !f) return false;

	f << "ply" << endl;
	f << "format ascii 1.0" << endl;
	f << "element vertex " << aPoints.size() << endl;
	f << "property float x" << endl;
	f << "property float y" << endl;
	f << "property float z" << endl;
	f << "property uchar red" << endl;
	f << "property uchar green" << endl;
	f << "property uchar blue" << endl;
	f << "element face 0" << endl;
	f << "property list uchar int vertex_indices" << endl;
	f << "end_header" << endl;

	list<Pt3dr>::const_iterator itPoint = aPoints.begin();
	while (itPoint != aPoints.end())
	{
		const Pt3dr &p = *itPoint++;
		f << p.x << ' ' << p.y << ' ' << p.z << " 128 128 128" << endl;
	}

	return true;
}

int command_maskContent( int argc, char ** argv )
{
	if (argc < 1) ELISE_ERROR_EXIT("a *_polyg3d.ply file is needed");

	const string maskFilename = argv[1];
	cMasqBin3D *masqBin3D = cMasqBin3D::FromSaisieMasq3d(maskFilename);

	if (masqBin3D == NULL) ELISE_ERROR_EXIT("cannot load mask 3d file [" << maskFilename << "]");

	Pt3dr bbP0, bbP1;
	if (argc >= 2)
	{
		const string plyFilename = argv[1];
		getPlyBoundingBox(plyFilename, bbP0, bbP1);
		cout << "--- bounding box [" << bbP0 << ", " << bbP1 << ']' << endl;

		list<Pt3dr> inMaskPoints;
		makeGrid(bbP0, bbP1 - bbP0, 1000, *masqBin3D, inMaskPoints);
		cout << "--- " << inMaskPoints.size() << " points in mask" << endl;

		const string outputFilename = "maskContent.ply";
		if ( !writePly(inMaskPoints, outputFilename)) ELISE_ERROR_EXIT("failed to write point cloud in [" << outputFilename << ']');
	}

	delete masqBin3D;

	return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------

unsigned int nbDigits( unsigned int i_n, unsigned int i_base )
{
	//~ double l = log((double)i_n) / log(i_base);
	//~ unsigned int i = (unsigned int)l;
	//~ if (l != double(i)) return i + 1;
	//~ return i;

	stringstream ss;
	ss << i_n;
	return (unsigned int)ss.str().length();
}

int command_renameImageSet( int argc, char **argv )
{
	if (argc < 2) ELISE_ERROR_EXIT("arg0 is a path + regular expression, arg1 is output prefix");

	cElPathRegex fullPattern(argv[0]);
	cElFilename outputBase(argv[1]);
	cout << "input pattern = [" << fullPattern.str() << "]" << endl;
	cout << "output base = [" << outputBase.str() << "]" << endl;

	list<cElFilename> filenames;
	fullPattern.getFilenames(filenames);

	const unsigned int nbdigits = nbDigits((unsigned int)filenames.size(), 10);
	unsigned int iFilename = 0;
	list<cElFilename>::const_iterator itFilename = filenames.begin();
	while (itFilename != filenames.end())
	{
		const cElFilename src = (*itFilename++);
		const string srcStr = src.str();

		stringstream ss;
		ss << outputBase.str() << setw(nbdigits) << setfill('0') << (iFilename++) << getShortestExtension(srcStr);
		cElFilename dst(ss.str());

		cout << '[' << src.str() << "] -> " << dst.str() << endl;

		if (dst.exists())
			cerr << "[" << dst.str() << "] already exists" << endl;
		else if ( !src.copy(dst))
			cerr << "failed to copy [" << src.str() << "] to [" << dst.str() << "]" << endl;
	}

	return EXIT_SUCCESS;
}


//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------

int command_toto( int argc, char **argv )
{
	if (argc != 1) ELISE_ERROR_EXIT("missing 3d points filename");

	const string filename = argv[0];
	if ( !ELISE_fp::exist_file(filename)) ELISE_ERROR_EXIT("file [" << filename << "] does not exist");

	ifstream f(filename.c_str(), ios::binary);
	ELISE_DEBUG_ERROR( !f, "command_toto", "failed to open file [" << filename << "] for reading");

	vector<Pt3dr> points;

	INT4 nbPoints;
	f.read((char *)&nbPoints, 4);
	ELISE_DEBUG_ERROR(nbPoints < 0, "command_toto", "invalid nbPoints = " << nbPoints);
	cout << "nbPoints = " << nbPoints << ' ' << f.tellg() << endl;
	points.resize((size_t)nbPoints);

	REAL readPoint[3];
	Pt3dr *itDst = points.data();
	//~ while (nbPoints--)
	for (int i = 0; i < nbPoints; i++)
	{
		f.read((char *)readPoint, sizeof(readPoint));
		itDst->x = readPoint[0];
		itDst->y = readPoint[1];
		(*itDst++).z = readPoint[2];

		cout << i << ": " << itDst[-1] << ' ' << f.tellg() << endl;
	}

	return EXIT_SUCCESS;
}

int TestJB_main( int argc, char **argv )
{
	string command;

	if (argc > 1)
	{
		command = argv[1];
		for (size_t i = 0; i < command.size(); i++)
			command[i] = tolower(command[i]);
	}

	int res = EXIT_FAILURE;
	command_t *itCommand = commands;
	while (itCommand->func != NULL)
	{
		if (itCommand->name == command)
		{
			res = (*itCommand->func)( argc-2, argv+2 );
			break;
		}
		itCommand++;
	}

	if (argc == 2)
	{
		ctPath path = getWorkingDirectory();
		cout << "working directory: [" << path.str_unix() << ']' << endl;

		ctPath newPath(argv[1]);
		if ( !setWorkingDirectory(newPath)) ELISE_ERROR_EXIT("failed to change directory to [" << newPath.str() << "]");

		path = getWorkingDirectory();
		cout << "working directory: [" << path.str_unix() << ']' << endl;

		list<cElFilename> filenames;
		ctPath current(".");
		current.getContent(filenames);
		for (list<cElFilename>::const_iterator it = filenames.begin(); it != filenames.end(); it++)
			cout << '[' << it->str() << ']' << endl;
	}

	if (itCommand->func == NULL)
	{
		cout << "command [" << command << "] is not valid" << endl;
		cout << "commands are :" << endl;
		itCommand = commands;
		while (itCommand->func != NULL)
		{
			cout << '\t' << itCommand->name << endl;
			itCommand++;
		}
	}

	return res;
}
