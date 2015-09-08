#include "StdAfx.h"

#define __DRAW_POINTS
#define __CHECK_MULTIPLICATION
//~ #define __SAVE_THEORICAL_POINTS
//~ #define __PRINT_COMMANDS

using namespace std;

typedef double tReal;

template <class T>
class Matrix33
{
public:
	T mCoefficients[9];

	void set(
		const T &aM11, const T &aM12, const T &aM13,
		const T &aM21, const T &aM22, const T &aM23,
		const T &aM31, const T &aM32, const T &aM33 )
	{
		mCoefficients[0] = aM11; mCoefficients[1] = aM12; mCoefficients[2] = aM13;
		mCoefficients[3] = aM21; mCoefficients[4] = aM22; mCoefficients[5] = aM23;
		mCoefficients[6] = aM31; mCoefficients[7] = aM32; mCoefficients[8] = aM33;
	}

	void set( const T *aCoefficients )
	{
		mCoefficients[0] = aCoefficients[0]; mCoefficients[1] = aCoefficients[1]; mCoefficients[2] = aCoefficients[2];
		mCoefficients[3] = aCoefficients[3]; mCoefficients[4] = aCoefficients[4]; mCoefficients[5] = aCoefficients[5];
		mCoefficients[6] = aCoefficients[6]; mCoefficients[7] = aCoefficients[7]; mCoefficients[8] = aCoefficients[8];
	}

	Matrix33(){}

	void set( T (*aFunc)() )
	{
		mCoefficients[0] = aFunc();
		mCoefficients[1] = aFunc();
		mCoefficients[2] = aFunc();
		mCoefficients[3] = aFunc();
		mCoefficients[4] = aFunc();
		mCoefficients[5] = aFunc();
		mCoefficients[6] = aFunc();
		mCoefficients[7] = aFunc();
		mCoefficients[8] = aFunc();
	}

	Matrix33( T (*aFunc)() ){ set(aFunc); }

	Matrix33(
		const T &aM11, const T &aM12, const T &aM13,
		const T &aM21, const T &aM22, const T &aM23,
		const T &aM31, const T &aM32, const T &aM33 )
	{
		set(
			aM11, aM12, aM13,
			aM21, aM22, aM23,
			aM31, aM32, aM33 );
	}

	Matrix33( const T *aCoefficients )
	{
		set(aCoefficients);
	}

	static Matrix33 identity()
	{
		return Matrix33(
			T(1), T(0), T(0),
			T(0), T(1), T(0),
			T(0), T(0), T(1) );
	}

	Matrix33 operator *( const Matrix33 &aB ) const
	{
		Matrix33 result;
		const T *a = mCoefficients;
		const T *b = aB.mCoefficients;
		T * c = result.mCoefficients;
		c[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6];     c[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7];     c[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];
		c[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6];     c[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7];     c[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8];
		c[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6];     c[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7];     c[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8];
		return result;
	}

	Matrix33 operator *( const T &aB ) const
	{
		Matrix33 result;
		T * c = result.mCoefficients;
		c[0] = mCoefficients[0] * aB; c[1] = mCoefficients[1] * aB; c[2] = mCoefficients[2] * aB;
		c[3] = mCoefficients[3] * aB; c[4] = mCoefficients[4] * aB; c[5] = mCoefficients[5] * aB;
		c[6] = mCoefficients[6] * aB; c[7] = mCoefficients[7] * aB; c[8] = mCoefficients[8] * aB;
		return result;
	}

	void mult( const T *aVector, T oVector ) const
	{
		oVector[0] = mCoefficients[0] * aVector[0] + mCoefficients[1] * aVector[1] + mCoefficients[2] * aVector[2];
		oVector[1] = mCoefficients[3] * aVector[0] + mCoefficients[4] * aVector[1] + mCoefficients[5] * aVector[2];
		oVector[2] = mCoefficients[6] * aVector[0] + mCoefficients[7] * aVector[1] + mCoefficients[8] * aVector[2];
	}

	void mult( const T &aX, const T &aY, T &oX, T &oY ) const
	{
		ELISE_DEBUG_ERROR(&aX==&oX, "", "aX and oX are the same variable");
		ELISE_DEBUG_ERROR(&aY==&oY, "", "aX and oX are the same variable");

		const T d = mCoefficients[6] * aX + mCoefficients[7] * aY + mCoefficients[8];

		ELISE_DEBUG_ERROR( d==T(0), "Matrix33::mult(const T &, const T &, T &, T & )", "d = 0" );

		oX = (mCoefficients[0] * aX + mCoefficients[1] * aY + mCoefficients[2]) / d;
		oY = (mCoefficients[3] * aX + mCoefficients[4] * aY + mCoefficients[5]) / d;
	}

	Matrix33 operator /( const T &aB ) const
	{
		Matrix33 result;
		T * c = result.mCoefficients;
		c[0] = mCoefficients[0] / aB; c[1] = mCoefficients[1] / aB; c[2] = mCoefficients[2] / aB;
		c[3] = mCoefficients[3] / aB; c[4] = mCoefficients[4] / aB; c[5] = mCoefficients[5] / aB;
		c[6] = mCoefficients[6] / aB; c[7] = mCoefficients[7] / aB; c[8] = mCoefficients[8] / aB;
		return result;
	}

	Matrix33 operator -( const Matrix33 &aB )
	{
		Matrix33 result;
		const T * b = aB.mCoefficients;
		T * c = result.mCoefficients;
		c[0] = mCoefficients[0] - b[0]; c[1] = mCoefficients[1] - b[1]; c[2] = mCoefficients[2] - b[2];
		c[3] = mCoefficients[3] - b[3]; c[4] = mCoefficients[4] - b[4]; c[5] = mCoefficients[5] - b[5];
		c[6] = mCoefficients[6] - b[6]; c[7] = mCoefficients[7] - b[7]; c[8] = mCoefficients[8] - b[8];
		return result;
	}

	Matrix33 operator *=( const Matrix33 &b ) { return ( *this = (*this)*b ); }
	Matrix33 operator *=( const T &b ) { return ( *this = (*this)*b ); }
	Matrix33 operator /=( const T &b ) { return ( *this = (*this)/b ); }

	void setTranslation( const T &aX, const T &aY )
	{
		ELISE_DEBUG_ERROR( aX == T(0) || aY == T(0), "Matrixx::setTranslation", "aX == 0 || aY == 0" );
		mCoefficients[0] = T(1); mCoefficients[1] = T(0); mCoefficients[2] = aX;
		mCoefficients[3] = T(0); mCoefficients[4] = T(1); mCoefficients[5] = aY;
		mCoefficients[6] = T(0); mCoefficients[7] = T(0); mCoefficients[8] = T(1);
	}

	void setRotation( double aAngle ){ setRotation( cos(aAngle), sin(aAngle) ); }

	void setRotation( const T &aCosinus, const T &aSinus )
	{
		//~ ELISE_DEBUG_ERROR( aCosinus*aCosinus+aSinus*aSinus!=T(1), "setRotation(cosinus,sinus)", "cosinus²+sinus² != 1" );

		mCoefficients[0] = mCoefficients[4] = aCosinus;
		mCoefficients[1] = -( mCoefficients[3]=aSinus );
		mCoefficients[2] = mCoefficients[5] = mCoefficients[6] = mCoefficients[7] = T(0);
		mCoefficients[8] = T(1);
	}

	void setScale( const T &aScaleX, const T &aScaleY  )
	{
		mCoefficients[0] = aScaleX; mCoefficients[1] = T(0); mCoefficients[2] = T(0);
		mCoefficients[3] = T(0); mCoefficients[4] = aScaleY; mCoefficients[5] = T(0);
		mCoefficients[6] = T(0); mCoefficients[7] = T(0); mCoefficients[8] = T(1);
	}

	T determinant() const
	{
		return (
			mCoefficients[0]*( mCoefficients[4]*mCoefficients[8]-mCoefficients[7]*mCoefficients[5] )
			-mCoefficients[1]*( mCoefficients[3]*mCoefficients[8]-mCoefficients[6]*mCoefficients[5] )
			+mCoefficients[2]*( mCoefficients[3]*mCoefficients[7]-mCoefficients[6]*mCoefficients[4] )
		);
	}

	void transpose()
	{
		swap( mCoefficients[1], mCoefficients[3] );
		swap( mCoefficients[2], mCoefficients[6] );
		swap( mCoefficients[5], mCoefficients[7] );
	}

	void getComatrix( Matrix33 &oComatrix ) const
	{
		T *c = oComatrix.mCoefficients;

		c[0] = mCoefficients[4]*mCoefficients[8]-mCoefficients[7]*mCoefficients[5];
		c[1] = mCoefficients[6]*mCoefficients[5]-mCoefficients[3]*mCoefficients[8];
		c[2] = mCoefficients[3]*mCoefficients[7]-mCoefficients[6]*mCoefficients[4];

		c[3] = mCoefficients[7]*mCoefficients[2]-mCoefficients[1]*mCoefficients[8];
		c[4] = mCoefficients[0]*mCoefficients[8]-mCoefficients[6]*mCoefficients[2];
		c[5] = mCoefficients[6]*mCoefficients[1]-mCoefficients[0]*mCoefficients[7];

		c[6] = mCoefficients[1]*mCoefficients[5]-mCoefficients[4]*mCoefficients[2];
		c[7] = mCoefficients[3]*mCoefficients[2]-mCoefficients[0]*mCoefficients[5];
		c[8] = mCoefficients[0]*mCoefficients[4]-mCoefficients[3]*mCoefficients[1];
	}

	bool getInverse( Matrix33 &oInverse ) const
	{
		T d = determinant();
		if ( d==T(0) )
		{
			ELISE_DEBUG_ERROR(true, "Matrix33:inverse", "determinant()==0");
			return false;
		}
		getComatrix(oInverse);
		oInverse.transpose();
		oInverse /= d;
		return true;
	}

	void print( const string &aPrefix=string(), ostream &aStream=cout ) const
	{
		if ( aPrefix.length()!=0 ) aStream << aPrefix << ':' << endl;

		// get string values and the max length of string values for each column
		size_t columnMaxLength[3] = { 0, 0, 0 };
		string strValues[9];
		size_t k = 0;
		for ( size_t j=0; j<3; j++ )
			for ( size_t i=0; i<3; i++, k++ )
			{
				stringstream ss;
				ss << mCoefficients[k];
				strValues[k] = ss.str();
				if ( strValues[k].length()>columnMaxLength[i] ) columnMaxLength[i] = strValues[k].length();
			}

		// append whitespace to complete string values with less than the max for its column
		k = 0;
		for ( size_t j=0; j<3; j++ )
			for ( size_t i=0; i<3; i++, k++ )
			{
				if ( strValues[k].length()<columnMaxLength[i] )
				{
					strValues[k] = string( columnMaxLength[i]-strValues[k].length(), ' ' )+strValues[k];
				}
			}

		aStream << "[ " << strValues[0] << " " << strValues[1] << " " << strValues[2] << endl;
		aStream << "  " << strValues[3] << " " << strValues[4] << " " << strValues[5] << endl;
		aStream << "  " << strValues[6] << " " << strValues[7] << " " << strValues[8] << " ]" << endl;
	}

	void difference( const Matrix33 &aB, T &oMin, T &oMax ) const
	{
		const T *b = aB.mCoefficients;
		oMin = mCoefficients[0]-b[0];
		if ( oMin<T(0) ) oMin = -oMin;
		oMax = oMin;
		for ( int i=1; i<9; i++ )
		{
			T d = mCoefficients[i]-b[i];
			if ( d<T(0) ) d = -d;
			if ( d<oMin )
				oMin = d;
			else if ( d>oMax )
				oMax = d;
		}
	}
};


typedef Matrix33<tReal> Matrix33R;

void print_usage( const char *aExecutable, int aReturnCode=EXIT_FAILURE )
{
	cout << "usage: " << basename(aExecutable) << " input_filename degree_rotation_angle scale output_directory " << endl;
	exit(aReturnCode);
}

void stringToReal( const char *aStr, tReal &oReal )
{
	try
	{
		oReal = stod( string(aStr) );
	}
	catch ( exception &e )
	{
		ELISE_ERROR_EXIT("cannot convert string [" << aStr << "] to double");
	}
}

void makeMatrices( double aAngle, double aScale, double oDirectMatrix[4], double oInverseMatrix[4] )
{
	const double c = cos(aAngle), s = sin(aAngle);

	oDirectMatrix[0] = oDirectMatrix[3] = c*aScale;
	oDirectMatrix[1] = -( oDirectMatrix[2]=s*aScale );

	oInverseMatrix[0] = oInverseMatrix[3] = c/aScale;
	oInverseMatrix[2] = -( oInverseMatrix[1]=s/aScale );
}

void toIntegers( double aValue, double &aMantisse, int &aExponent )
{
	aMantisse = frexp(aValue,&aExponent);
}

void printExactReal( const double &aValue, ostream &aStream=cout )
{
	double m;
	int e;
	toIntegers(aValue,m,e);
	aStream << m << "x2^" << e;
}

void printVector( const string &aName, const double aVector[2] )
{
	cout << aName << " = (" << aVector[0] << ", " << aVector[1] << ")" << endl;
}

void getBoundingBoxMax( double aMaxX, double aMaxY, const Matrix33R &aMatrix, double &oMaxX, double &oMaxY )
{
	aMatrix.mult(-aMaxX, -aMaxY, oMaxX, oMaxY);

	double x, y;
	aMatrix.mult(aMaxX, -aMaxY, x, y);
	ElSetMax(oMaxX,x);
	ElSetMax(oMaxY,y);

	aMatrix.mult(-aMaxX, aMaxY, x, y);
	ElSetMax(oMaxX,x);
	ElSetMax(oMaxY,y);

	aMatrix.mult(aMaxX, aMaxY, x, y);
	ElSetMax(oMaxX,x);
	ElSetMax(oMaxY,y);
}

void getDstImageSize( double aMaxX, double aMaxY, const Matrix33R &aMatrix, Pt2di &oDstSize )
{
	double maxX, maxY;
	getBoundingBoxMax( aMaxX, aMaxY, aMatrix, maxX, maxY );
	oDstSize.x = (int)ceil( 2*maxX+0.5 );
	oDstSize.y = (int)ceil( 2*maxY+0.5 );
}

#ifdef __DEBUG
	void __check_inverse( const string &aWhere, const Matrix33R &aMatrix )
	{
		const double epsilon = 1e-9;
		Matrix33R inv;
		ELISE_DEBUG_ERROR( !aMatrix.getInverse(inv), aWhere << " -> __check_inverse", "singular matrix");
		double minDiff, maxDiff;
		Matrix33R p = aMatrix*inv;
		p.difference(Matrix33<double>::identity(), minDiff, maxDiff);
		ELISE_DEBUG_ERROR(maxDiff>epsilon, "__check_inverse", "maxDiff > epsilon");
	}
#endif

bool makeTransformMatrices( double aAngleRadian, double aScale, const Pt2di &aSrcImageSize, Pt2di &oDstImageSize, Matrix33R &oDirectMatrix, Matrix33R &oInverseMatrix )
{
	Matrix33R rotation, scale, transformMatrix;
	rotation.setRotation(aAngleRadian);
	scale.setScale(aScale,aScale);
	transformMatrix = scale*rotation;

	Matrix33R translation0, translation1;
	double srcHalfWidth = double(aSrcImageSize.x-1)/2., srcHalfHeight = double(aSrcImageSize.y-1)/2.;
	translation0.setTranslation(-srcHalfWidth, -srcHalfHeight);

	getDstImageSize( srcHalfWidth, srcHalfHeight, transformMatrix, oDstImageSize );
	double dstHalfWidth = double(oDstImageSize.x-1)/2., dstHalfHeight = double(oDstImageSize.y-1)/2.;
	translation1.setTranslation(dstHalfWidth, dstHalfHeight);

	oDirectMatrix = translation1*transformMatrix*translation0;

	#ifdef __DEBUG
		__check_inverse("makeTransformMatrices: rotation", rotation);
		__check_inverse("makeTransformMatrices: scale", scale);
		__check_inverse("makeTransformMatrices: translation0", translation0);
		__check_inverse("makeTransformMatrices: translation1", translation1);
		__check_inverse("makeTransformMatrices: transformMatrix", transformMatrix);
	#endif

	return oDirectMatrix.getInverse(oInverseMatrix);

	//~ if ( !transformMatrix.inverse(oInverseMatrix) ) return false;
	//~ translation0.setTranslation(-dstHalfWidth, -dstHalfHeight);
	//~ translation1.setTranslation(srcHalfWidth, srcHalfHeight);
	//~ oInverseMatrix = translation0*oInverseMatrix*translation1;
	//~ return true;
}

template <class tData>
tData nearestPixelValue( const tData **aData, int aWidth, int aHeight, double aX, double aY )
{
	int x = (int)round(aX), y = (int)round(aY);
	//~ int x = (int)( aCoordinates[0] ), y = (int)round( aCoordinates[1] );
	if ( x<0 || x>=aWidth || y<0 || y>=aHeight ) return 0;
	return aData[y][x];
}

inline double distance( double aX0, double aY0, const double &aX1, const double &aY1 )
{
	aX0 -= aX1;
	aY0 -= aY1;
	return sqrt(aX0 * aX0 + aY0 * aY0);
}

template <class tData>
tData linearPixelValue( const tData **aData, const int &aWidth, const int &aHeight, const double &aX, const double &aY )
{
	pair<tData,double> neighbours[4];
	int nbNeighbours = 0;

	int x = (int)aX, y = (int)aY;
	if ( x>=0 && x<aWidth && y>=0 && y<aHeight )
		neighbours[nbNeighbours++] = pair<tData,double>(aData[y][x], distance((double)x, (double)y, aX, aY));

	x++;
	if ( x>=0 && x<aWidth && y>=0 && y<aHeight )
		neighbours[nbNeighbours++] = pair<tData,double>(aData[y][x], distance((double)x, (double)y, aX, aY));

	y++;
	if ( x>=0 && x<aWidth && y>=0 && y<aHeight )
		neighbours[nbNeighbours++] = pair<tData,double>(aData[y][x], distance((double)x, (double)y, aX, aY));

	x--;
	if ( x>=0 && x<aWidth && y>=0 && y<aHeight )
		neighbours[nbNeighbours++] = pair<tData,double>(aData[y][x], distance((double)x, (double)y, aX, aY));

	double coefficientsSum = 0.;
	for ( int i=0; i<nbNeighbours; i++)
		coefficientsSum += neighbours[i].second;

	double result = 0.;
	for (int i=0; i<nbNeighbours; i++)
		result += (coefficientsSum-neighbours[i].second)*(double)neighbours[i].first;

	//~ if (result>=4*coefficientsSum) return numeric_limits<tData>::max();

	return (tData)(result / (nbNeighbours * coefficientsSum));
}

template <class tData>
void transformInverse( const Im2D<tData,TBASE> &aSrc, const Matrix33R &aInverseMatrix, Im2D<tData,TBASE> &oDst )
{
	const int width = aSrc.tx(), height = aSrc.ty();
	const tData **srcData = ( const tData ** )( ( Im2D<tData,TBASE>* )&aSrc )->data();
	tData **dstData = oDst.data();
	double x, y;
	for ( int j=0; j<oDst.ty(); j++ )
		for ( int i=0; i<oDst.tx(); i++ )
		{
			aInverseMatrix.mult( (double)i, (double)j, x, y );
			dstData[j][i] = nearestPixelValue<tData>(srcData, width, height, x, y);
			//~ dstData[j][i] = linearPixelValue<tData>(srcData, width, height, x, y);
		}
}

void saveTiffGray( Im2DGen aSrc, const string &aFilename )
{
	ELISE_COPY
	(
		aSrc.all_pts(),
		aSrc.in(),
		Tiff_Im(
			aFilename.c_str(),
			aSrc.sz(),
			aSrc.TypeEl(),
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero,
			Tiff_Im::Empty_ARG ).out()
	);
}

void saveTiffRgb( Im2D_U_INT1 &aRed, Im2D_U_INT1 &aGreen, Im2D_U_INT1 &aBlue, const string &aFilename )
{
	if ( aRed.sz()!=aGreen.sz() || aRed.sz()!=aBlue.sz() ) ELISE_ERROR_EXIT( "saveTiffRgb: inconsistent sizes " << aRed.sz() << ' ' << aGreen.sz() << ' ' << aBlue.sz() );

	ELISE_COPY
	(
		aRed.all_pts(),
		Virgule( aRed.in(), aGreen.in(), aBlue.in() ),
		Tiff_Im(
			aFilename.c_str(),
			aRed.sz(),
			GenIm::u_int1,
			Tiff_Im::No_Compr,
			Tiff_Im::RGB,
			Tiff_Im::Empty_ARG ).out()
	);
}

template <class tData>
void transformAndSave( const Im2D<tData,TBASE> &aSrc, const Matrix33R &aMatrix, const Pt2di &aDstSize, const string &aFilename )
{
	Im2D<tData,TBASE> dst(aDstSize.x, aDstSize.y);
	transformInverse(aSrc, aMatrix, dst);
	//~ transformDirect(aSrc, aMatrix, dst);
	saveTiffGray(dst,aFilename);
}

inline int realToInteger( const double &aReal )
{
	return (int)( aReal*65535+0.5 );
}

string makeDstBasename( const string aSrcBasename, int aDegreeAngle, double aScale )
{
	stringstream ss;
	ss << aSrcBasename << "_r" << setw(3) << setfill('0') << aDegreeAngle << "_s" << setw(5) << setfill('0') << realToInteger(aScale) << ".tif";
	return ss.str();
}

inline string makePointsFilename( const string &aBasename, const string &aDirectory )
{
	return aDirectory+aBasename+".dat";
}

inline string makeMatchesFilename( const string &aImageBasename0, const string &aImageBasename1, const string &aDstDirectory )
{
	return aDstDirectory+aImageBasename0+"_"+aImageBasename1+".result";
}

void systemChecked( const string &aCommand, const string &aFilename )
{
	#ifdef __PRINT_COMMANDS
		cout << "command[" << aCommand << ']' << endl;
	#endif

	int result = system( aCommand.c_str() );
	if ( result!=0 ) ELISE_ERROR_EXIT( "command [" << aCommand << "] failed with value " << result );
	if ( !ELISE_fp::exist_file(aFilename) ) ELISE_ERROR_EXIT( "command [" << aCommand << "] failed to create file " << aFilename );
}

void computePointsOfInterest( const string &aSrcFilename, const string &aDstFilename, const string &aPrefix=string() )
{
	cout << aPrefix << "--- computing points of interest for file [" << aSrcFilename << "] -> [" << aDstFilename << ']' << endl;
	string command = string("../../../bin/mm3d Sift ") + aSrcFilename + " -o " + aDstFilename + " >/dev/null";
	//~ string command = string("../../../bin/mm3d Digeo ") + aSrcFilename + " -o " + aDstFilename + " >/dev/null";
	systemChecked(command,aDstFilename);
}

void computeMatches( const string &aSrcFilename0, const string &aSrcFilename1, const string &aDstFilename )
{
	cout << '\t' << "--- computing matches for file [" << aSrcFilename0 << "], [" << aSrcFilename1 << "] -> [" << aDstFilename << ']' << endl;
	//~ string command = string("../../../bin/mm3d Ann -newFormat ") + aSrcFilename0 + " " + aSrcFilename1 + " " + aDstFilename + " >/dev/null";
	string command = string("./dumb_matcher ") + aSrcFilename0 + " " + aSrcFilename1 + " " + aDstFilename + " >/dev/null";
	systemChecked(command,aDstFilename);
}

bool readMatches( const string &aFilename, list<pair<Pt2dr,Pt2dr> > &oMatches )
{
	oMatches.clear();

	ifstream f( aFilename.c_str() );
	if ( !f ) return false;

	int i = 0;
	REAL x0, y0, x1, y1;
	while ( !f.eof() && i<1000 )
	{
		f >> x0 >> y0 >> x1 >> y1;
		oMatches.push_back( pair<Pt2dr,Pt2dr>( Pt2dr(x0,y0), Pt2dr(x1,y1) ) );
		i++;
	}
	oMatches.pop_back();

	return true;
}

void print( const list<pair<Pt2dr,Pt2dr> > &aMatches )
{
	list<pair<Pt2dr,Pt2dr> >::const_iterator it = aMatches.begin();
	while ( it!=aMatches.end() )
	{
		cout << it->first << ' ' << it->second << endl;
		it++;
	}
}

double dist(double x0, double y0, double x1, double y1 )
{
	x0 -= x1;
	y0 -= y1;
	return sqrt(x0*x0+y0*y0);
}

void toIntegers( double aValue, int &aMantisse, int &aExponent )
{
	double significand = frexp(aValue,&aExponent);
	cout << significand << "x2^" << aExponent << endl;
}

//~ #define DIFFERENCE_PRECISION 1e-9

double difference( const Pt2dr &aSrcPoint, const Matrix33R &aMatrix, const Pt2dr &aDstPoint )
{
	double x, y;
	aMatrix.mult(aSrcPoint.x, aSrcPoint.y, x, y);
	double dx = x-aDstPoint.x, dy = y-aDstPoint.y;
	double result = sqrt(dx*dx+dy*dy);
	#ifdef DIFFERENCE_PRECISION
		return round( result/DIFFERENCE_PRECISION )*DIFFERENCE_PRECISION;
	#else
		return result;
	#endif
}

void trim( string &io_str )
{
	const size_t length = io_str.length();

	if ( length==0 ) return;

	size_t pos0 = 0;
	while ( pos0<length && ( io_str[pos0]==' ' || io_str[pos0]=='\t' ) ) pos0++;

	size_t pos1 = length-1;
	while ( pos1<length && ( io_str[pos1]==' ' || io_str[pos1]=='\t' ) ) pos1--;

	io_str = io_str.substr( pos0, pos1-pos0+1 );
}

void computePointsImages( const string &aFilename, const Matrix33R &aMatrix, vector<double> &oDifferences )
{
	vector<PointMatch> matches;
	readMatchesFile(aFilename,matches);
	cout << '\t' << "--- [" << aFilename << "]: " << matches.size() << " matches" << endl;

	oDifferences.resize(matches.size());
	double *itDifference = oDifferences.data();

	// __DEL
	size_t iDiff = 0;

	const PointMatch *itMatch = matches.data();
	size_t iMatch = matches.size();
	while ( iMatch-- )
	{
		*itDifference++ = difference(itMatch->first, aMatrix, itMatch->second);

		// __DEL
		if ( oDifferences[iDiff]>100 )
		{
			cout << "difference[" << iDiff << "] = " << oDifferences[iDiff] << " > 100" << endl;
			cout << '\t' << "p0 = " << itMatch->first << endl;
			cout << '\t' << "p1 = " << itMatch->second << endl;
			exit(EXIT_FAILURE);
		}
		iDiff++;

		itMatch++;
	}
}

void drawMatches( const string &aMatchesFilename, const string &aSrcImageFilename0, const string &aSrcImageFilename1, const string &aDstImageFilename0, const string &aDstImageFilename1)
{
	list<pair<Pt2dr,Pt2dr> > matches;
	if ( !ELISE_fp::exist_file(aMatchesFilename) ) ELISE_ERROR_EXIT("file [" << aMatchesFilename << "] does not exist");
	if ( !readMatches(aMatchesFilename, matches) ) ELISE_ERROR_EXIT("cannot read matches from [" << aMatchesFilename << "]");

	if ( !ELISE_fp::exist_file(aSrcImageFilename0) ) ELISE_ERROR_EXIT("file [" << aSrcImageFilename0 << "] does not exist");
	if ( !ELISE_fp::exist_file(aSrcImageFilename1) ) ELISE_ERROR_EXIT("file [" << aSrcImageFilename1 << "] does not exist");

	Im2D<U_INT1,INT> src = Im2D<U_INT1,INT>::FromFileBasic(aSrcImageFilename0),
                    dst = Im2D<U_INT1,INT>::FromFileBasic(aSrcImageFilename1),
                    copySrc( src.tx(), src.ty() ),
                    copyDst( dst.tx(), dst.ty() );
	copySrc.dup(src);
	copyDst.dup(dst);
	U_INT1 **srcData = copySrc.data();
	U_INT1 **dstData = copyDst.data();
	int x, y;
	list<pair<Pt2dr,Pt2dr> >::const_iterator itMatch = matches.begin();
	while ( itMatch!=matches.end() )
	{
		const Pt2dr &p0 = itMatch->first, &p1 = (*itMatch++).second;

		x = (int)round(p0.x);
		y = (int)round(p0.y);
		if ( x>=0 && x<src.tx() && y>=0 && y<src.ty() ) srcData[y][x] = 255;

		x = (int)round(p1.x);
		y = (int)round(p1.y);
		if ( x>=0 && x<dst.tx() && y>=0 && y<dst.ty() ) dstData[y][x] = 255;
	}

	saveTiffRgb(copySrc, src, src, aDstImageFilename0);
	saveTiffRgb(dst, dst, copyDst, aDstImageFilename1);
	cout << '\t' << "--- ploted matches written to [" << aDstImageFilename0 << "] and [" << aDstImageFilename1 << ']' << endl;
}

inline string plotFilename( const string &aFilename, const string &aOutputDirectory )
{
	string directory, basename;
	SplitDirAndFile(directory, basename, aFilename);
	return aOutputDirectory + basename + ".plot.tif";
}

void transformPoints( vector<DigeoPoint> &aPoints, const Matrix33R &aMatrix )
{
	DigeoPoint *itPoint = aPoints.data();
	double x, y;
	size_t iPoint = aPoints.size();
	while ( iPoint-- )
	{
		aMatrix.mult(itPoint->x, itPoint->y, x, y);
		itPoint->x = x;
		(*itPoint++).y = y;
	}
}

string checkDstDirectory( string aDirectory )
{
	// make sure directory exists
	if ( /*aDirectory.length()<1 ||*/ !ELISE_fp::MkDirSvp(aDirectory) )
		ELISE_ERROR_EXIT( "failed to create output directory [" << aDirectory << ']' );

	// make sure there is a trailing '/'
	if ( aDirectory.back()=='\\' )
		aDirectory.back()='/';
	else if ( aDirectory.back()!='/' )
		aDirectory.push_back('/');
	return aDirectory;
}

inline bool is_number( const char &c ){ return c>='0' && c<='9'; }

bool stringToSize_t( const string &aString, size_t &oUint )
{
	size_t mult = 1, oldUint = 0;
	size_t i = aString.length();
	const char *it = aString.data()+(i-1);
	oUint = 0;
	while (i--)
	{
		const char &c = *it--;

		if ( !is_number(c) )
		{
			ELISE_DEBUG_ERROR( true, "stringToUint", "invalid character '" << c << "' (value " << (int)c << ')' );
			return false;
		}

		if ( c!='0' )
		{
			oUint += ( c-'0' )*mult;

			if ( oUint<oldUint )
			{
				ELISE_DEBUG_ERROR( true, "stringToUint", "value out of size_t range" );
				return false;
			}

			oldUint = oUint;
		}

		mult *= 10;
	}
	return true;
}

bool stringToLongLongInt(string aString, long long int &oInt)
{
	if ( aString.length()==0 )
	{
		ELISE_DEBUG_ERROR(true, "stringToInt", "empty string");
		return false;
	}

	bool isNegative = false;
	if ( aString.front()=='+' )
		aString.front() = '0';
	else if ( aString.front()=='-' )
	{
		isNegative = true;
		aString.front() = '0';
	}

	size_t vUnsigned;
	if ( !stringToSize_t(aString, vUnsigned) ) return false;

	ELISE_DEBUG_ERROR(vUnsigned>size_t(numeric_limits<long long int>::max()), "stringToInt", "vUnsigned > size_t(numeric_limits<long long>::max())");
	oInt = (isNegative ? -(long long int)vUnsigned : (long long int)vUnsigned);

	return true;
}

long long int stringToLongLongInt_error( const string &aStr )
{
	long long int result;
	if ( !stringToLongLongInt(aStr, result) ) ELISE_ERROR_EXIT("cannot convert [" << aStr << "] to an integer value");
	return result;
}

int stringToInt_error( const string &aStr )
{
	long long int result = stringToLongLongInt_error(aStr);
	ELISE_DEBUG_ERROR(result<numeric_limits<int>::min() || result>numeric_limits<int>::max(), "stringToInt_error", "value " << result << "out of int range");
	return result;
}

void stringToIntegerSet( const string &aStr, int &oV0, int &oV1 )
{
	size_t pos = aStr.find_first_of("-");
	string vStr = aStr.substr(0, pos);

	oV0 = stringToInt_error(vStr);

	if ( pos==string::npos )
	{
		oV1 = oV0;
		return;
	}

	oV1 = stringToInt_error(aStr.substr(pos + 1));

	if ( oV0>oV1 ) ELISE_ERROR_EXIT("invalid set [" << oV0 << ';' << oV1 << "], " << oV0 << '>' << oV1 );
}

void statDifferences( const vector<vector<double> > &aDifferences, const string &aFilename )
{
	double overAllMinDiff = numeric_limits<double>::max(), overAllMaxDiff = 0.;
	double overAllMeanDiff = 0.;
	size_t overAllNbDiff = 0;

	ofstream f(aFilename.c_str());
	if ( !f ) ELISE_ERROR_EXIT("canot open file [" << aFilename << "] for writing");

	f << "### format: index min max mean" << endl;
	for ( size_t i=0; i<aDifferences.size(); i++ )
	{
		double minDiff = numeric_limits<double>::max(), maxDiff = 0.;
		double meanDiff = 0.;

		const vector<double> &differences = aDifferences[i];
		const double *itDiff = differences.data();
		size_t iDiff = differences.size();
		while ( iDiff-- )
		{
			if ( *itDiff<minDiff ) minDiff = *itDiff;
			if ( *itDiff>maxDiff ) maxDiff = *itDiff;
			meanDiff += *itDiff++;
		}
		overAllMeanDiff += meanDiff;
		overAllNbDiff += differences.size();
		meanDiff /= (double)differences.size();

		f << i << ' ' << minDiff << ' ' << maxDiff << ' ' << meanDiff << endl;

		if ( minDiff<overAllMinDiff ) overAllMinDiff = minDiff;
		if ( maxDiff>overAllMaxDiff ) overAllMaxDiff = maxDiff;
	}

	overAllMeanDiff /= (double)overAllNbDiff;
	f << endl;
	f << "### overall min/max/mean " << overAllMinDiff << ' ' << overAllMaxDiff << ' ' << overAllMeanDiff << endl;
}

int main( int argc, char **argv )
{
	if (argc<5) print_usage(argv[0]);

	const string srcImageFilename = argv[1];
	string srcDirectory, srcImageBasename;
	SplitDirAndFile(srcDirectory, srcImageBasename, srcImageFilename);

	int angleDegree0, angleDegree1;
	stringToIntegerSet(argv[2], angleDegree0, angleDegree1);
	cout << "--- angles: " << angleDegree0 << " -> " << angleDegree1 << endl;

	tReal scale;
	stringToReal(argv[3], scale);
	cout << "--- scale: " << scale << endl;

	const string dstDirectory = checkDstDirectory( argv[4] );

	string srcPointsFilename = makePointsFilename(srcImageBasename, dstDirectory);
	computePointsOfInterest(srcImageFilename, srcPointsFilename);

	#ifdef __SAVE_THEORICAL_POINTS
		vector<DigeoPoint> srcPoints, theoricalPoints;
		if ( !DigeoPoint::readDigeoFile(srcPointsFilename, true, srcPoints) ) ELISE_ERROR_RETURN( "cannot load points from file [" << srcPointsFilename << ']' );
		cout << "--- [" << srcPointsFilename << "]: " << srcPoints.size() << " points" << endl;
	#endif

	vector<vector<double> > differences(angleDegree1-angleDegree0+1);
	vector<double> *itDifferences = differences.data();
	for ( int angleDegree=angleDegree0; angleDegree<=angleDegree1; angleDegree++ )
	{
		tReal angleRadian = double(angleDegree)*(M_PI/180.);
		const string dstImageBasename = makeDstBasename(srcImageBasename, angleDegree, scale);
		angleRadian = -angleRadian;
		const string dstImageFilename = dstDirectory+dstImageBasename;

		Tiff_Im tiff( srcImageFilename.c_str() );
		cout << '\t' << "--- angle (degree): " << angleDegree << endl;
		cout << '\t' << "--- [" << srcImageFilename << "]: " << tiff.sz().x << 'x' << tiff.sz().y << 'x' << tiff.nb_chan() << endl;

		Matrix33R directMatrix, inverseMatrix;
		Pt2di dstImageSize;
		makeTransformMatrices( angleRadian, scale, tiff.sz(), dstImageSize, directMatrix, inverseMatrix );

		Im2DGen srcImage = tiff.ReadIm();
		if ( tiff.type_el()!=GenIm::u_int1 ) ELISE_ERROR_EXIT("image type " << eToString(tiff.type_el()) << " is not handled" );
		transformAndSave<U_INT1>(*(Im2D<U_INT1,INT> *)&srcImage, inverseMatrix, dstImageSize, dstImageFilename);
		//~ transformAndSave<U_INT1>(*(Im2D<U_INT1,INT> *)&srcImage, directMatrix, dstImageSize, dstImageFilename);

		string dstPointsFilename = makePointsFilename(dstImageBasename, dstDirectory);
		#ifdef __SAVE_THEORICAL_POINTS
			theoricalPoints = srcPoints;
			transformPoints(theoricalPoints, directMatrix);
			if ( !DigeoPoint::writeDigeoFile(dstPointsFilename, theoricalPoints) ) ELISE_ERROR_RETURN( "cannot save points to file [" << dstPointsFilename << ']' );
			cout << '\t' << "--- theorical transformed points saved to [" << dstPointsFilename << "]" << endl;
		#else
			computePointsOfInterest(dstImageFilename, dstPointsFilename, "\t");
		#endif

		string matchesFilename = makeMatchesFilename(srcImageBasename, dstImageBasename, dstDirectory);
		computeMatches(srcPointsFilename, dstPointsFilename, matchesFilename);

		//~ computePointsImages(srcPointsFilename,directMatrix);

		#ifdef __DRAW_POINTS
			const string plotDirectory = dstDirectory+"plot/";
			if ( !ELISE_fp::MkDirSvp(plotDirectory) ) ELISE_ERROR_EXIT( "failed to create plot directory [" << plotDirectory << ']' );
			drawMatches(matchesFilename, srcImageFilename, dstImageFilename, plotFilename(srcImageFilename, plotDirectory), plotFilename(dstImageFilename, plotDirectory));
		#endif

		computePointsImages(matchesFilename, directMatrix, *itDifferences++);

		cout << endl;
	}

	const string statsFilename = dstDirectory + "stats.txt";
	statDifferences(differences, statsFilename);

	return EXIT_SUCCESS;
}
