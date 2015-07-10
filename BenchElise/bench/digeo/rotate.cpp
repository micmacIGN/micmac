#include "StdAfx.h"

using namespace std;

void print_usage( const char *aExecutable, int aReturnCode=EXIT_FAILURE )
{
	cout << "usage: " << basename(aExecutable) << " image_filename radian_rotation_angle scale" << endl;
	exit(aReturnCode);
}

double stringToDouble( const char *aStr )
{
	try
	{
		return stod( string(aStr) );
	}
	catch ( exception &e )
	{
		cerr << "ERROR: cannot convert string [" << aStr << "] to double" << endl;
		exit(EXIT_FAILURE);
	}
	return 0.;
}

void multMatrix( const double aA[4], const double aB[4], double oRes[4] )
{
	oRes[0] = aB[0]*aA[0]+aB[2]*aA[1]; oRes[1] = aB[1]*aA[0]+aB[3]*aA[1]; 
	oRes[2] = aB[0]*aA[2]+aB[2]*aA[3]; oRes[3] = aB[1]*aA[2]+aB[3]*aA[3]; 
}

void multVector( const double aA[4], const double aB[2], double oRes[2] )
{
	oRes[0] = aB[0]*aA[0]+aB[1]*aA[1];
	oRes[1] = aB[0]*aA[2]+aB[1]*aA[3]; 
}

void makeMatrices( double aAngle, double aScale, double aDirectMatrix[4], double aInverseMatrix[4] )
{
	const double c = cos(aAngle), s = sin(aAngle);

	aDirectMatrix[0] = aDirectMatrix[3] = c*aScale;
	aDirectMatrix[1] = -( aDirectMatrix[2]=s*aScale );

	aInverseMatrix[0] = aInverseMatrix[3] = c/aScale;
	aInverseMatrix[2] = -( aInverseMatrix[1]=s/aScale );
}

void printMatrix( const string &aName, const double aMatrix[4] )
{
	cout << aName << " = " << endl;
	cout << "[ " << aMatrix[0] << ' ' << aMatrix[1] << endl;
	cout << "  " << aMatrix[2] << ' ' << aMatrix[3] << " ]" << endl;
}

void printVector( const string &aName, const double aVector[2] )
{
	cout << aName << " = (" << aVector[0] << ", " << aVector[1] << ")" << endl;
}

void setMax( double oV1[2], const double aV0[] )
{
	ElSetMax( oV1[0], aV0[0] );
	ElSetMax( oV1[1], aV0[1] );
}

void getBoundingBoxMax( double aMaxX, double aMaxY, const double aMatrix[4], double oMax[2] )
{
	double src[2], dst[2];

	src[0] = -aMaxX;
	src[1] = -aMaxY;
	multVector( aMatrix, src, oMax );

	src[0] = aMaxX;
	src[1] = -aMaxY;
	multVector( aMatrix, src, dst );
	setMax(oMax,dst);

	src[0] = -aMaxX;
	src[1] = aMaxY;
	multVector( aMatrix, src, dst );
	setMax(oMax,dst);

	src[0] = aMaxX;
	src[1] = aMaxY;
	multVector( aMatrix, src, dst );
	setMax(oMax,dst);

	printVector("bounding box max (double)",oMax);
}

void getBoundingBoxMax( int aWidth, int aHeight, const double aMatrix[4], int &oMaxX, int &oMaxY )
{
	double maxD[2];
	getBoundingBoxMax( double(aWidth)/2., double(aHeight)/2., aMatrix, maxD );
	oMaxX = ceil( maxD[0] );
	oMaxY = ceil( maxD[1] );

	cout << "bounding box max (int) = (" << oMaxX << ", " << oMaxY << ')' << endl;
}

template <class tData>
tData nearestPixelValue( const tData **aData, int aWidth, int aHeight, const double aCoordinates[2] )
{
	int x = int( aCoordinates[0]+0.5 ), y = int( aCoordinates[1]+0.5 );
	if ( x<0 || x>=aWidth || y<0 || y>=aHeight ) return 0;
	return aData[y][x];
}

template <class tData>
tData linearPixelValue( const tData **aData, int aWidth, int aHeight, const double aCoordinates[2] )
{
	int xmin = aCoordinates[0], ymin = aCoordinates[1];

	if ( xmin<0 || xmin>=aWidth || ymin<0 || ymin>=aHeight )
	{
		cerr << "point (" << aCoordinates[0] << ", " << aCoordinates[1] << ") out of image of size " << aWidth << 'x' << aHeight << endl;
		exit(EXIT_FAILURE);
	}

	//~ if ( x<0 || x>=aWidth || y<0 || y>=aHeight ) return 0;
	return aData[ymin][xmin];
}

template <class tData>
void transform( Im2DGen aSrc, const double aDirectMatrix[4], const double aInverseMatrix[4] )
{
	const int width = aSrc.tx(), height = aSrc.ty();
	const double halfWidth = double(width)/2., halfHeight = double(height)/2.;

	int maxX, maxY;
	getBoundingBoxMax( width, height, aDirectMatrix, maxX, maxY );

	Im2D<tData,TBASE> dst( maxX*2, maxY*2 );
	const tData **srcData = ( const tData ** )( ( Im2D<tData,TBASE>* )&aSrc )->data();
	tData **dstData = dst.data();
	double srcV[2], dstV[2];
	for ( int y=0; y<dst.ty(); y++ )
		for ( int x=0; x<dst.tx(); x++ )
		{
			srcV[0] = double(x-maxX);
			srcV[1] = double(y-maxY);
			multVector( aInverseMatrix, srcV, dstV );
			dstV[0] += halfWidth;
			dstV[1] += halfHeight;
			dstData[y][x] = nearestPixelValue<tData>( srcData, width, height, dstV );
		}

	string dstFilename = "toto.tif";
	ELISE_COPY
	(
		dst.all_pts(),
		dst.in(),
		Tiff_Im(
			dstFilename.c_str(),
			dst.sz(),
			GenIm::u_int1,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero,
			Tiff_Im::Empty_ARG ).out()
	);
}

int main( int argc, char **argv )
{
	if (argc<4) print_usage(argv[0]);

	const string inputFilename = argv[1];
	double angle = stringToDouble( argv[2] );
	double scale = stringToDouble( argv[3] );

	Tiff_Im tiff( inputFilename.c_str() );
	cout << "--- [" << inputFilename << "]: " << tiff.sz().x << 'x' << tiff.sz().y << 'x' << tiff.nb_chan() << endl;
	cout << "--- angle: " << angle << endl;
	cout << "--- scale: " << scale << endl;

	double directMatrix[4], inverseMatrix[4];
	makeMatrices( angle, scale, directMatrix, inverseMatrix );

	//~ double m[4];
	//~ printMatrix("direct",directMatrix);
	//~ printMatrix("inverse",inverseMatrix);
	//~ multMatrix(directMatrix,inverseMatrix,m);
	//~ printMatrix("direct*inverse",m);

	//~ double v0[2] = {10,25}, v1[2], v2[2];
	//~ multVector(directMatrix,v0,v1);
	//~ multVector(inverseMatrix,v1,v2);
	//~ printVector("v0",v0);
	//~ printVector("v1",v1);
	//~ printVector("v2",v2);

	if ( tiff.type_el()==GenIm::u_int1 )
		transform<U_INT1>( tiff.ReadIm(), directMatrix, inverseMatrix );

	return EXIT_SUCCESS;
}
