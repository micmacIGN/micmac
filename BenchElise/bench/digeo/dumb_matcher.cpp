#include "StdAfx.h"

using namespace std;

void print_v128( const string &aPrefix, const REAL8 *v )
{
	int i = 127;
	cout << aPrefix << (*v++);
	while ( i-- ) cout << ' ' << (*v++);
	cout << endl;
}

inline char is_different( const REAL8 &a, const REAL8 &b )
{
	return a==b?'0':'1';
}

void print_is_different_v128( const string aPrefix, const REAL8 *a, const REAL8 *b )
{
	int i = 127;
	cout << aPrefix << is_different(*a++,*b++);
	while ( i-- ) cout << ' ' << is_different(*a++,*b++);
	cout << endl;
}

class GraphPoint
{
public:
	const DigeoPoint *mPoint;
	list<GraphPoint*> mNearestNeighbours;
	double mNeighbourDistance;

	// statistic data
	list<GraphPoint*> mBackwardNeighbours; // a list of points considering this point as a neighbours
	bool mHasBeenWritten;

	GraphPoint( const DigeoPoint *aPoint=NULL ):
		mPoint(aPoint),
		mNeighbourDistance( numeric_limits<REAL>::max() ),
		mHasBeenWritten(false){}

	bool tryNeighbour( GraphPoint *aNeighbour, double aDistance )
	{
		if ( aDistance==mNeighbourDistance )
		{
			//~ ELISE_DEBUG_ERROR(true, "tryNeighbour", "two neighbours with the same descriptor distance");
			mNearestNeighbours.push_back(aNeighbour);
		}
		else if ( aDistance<mNeighbourDistance )
		{
			mNearestNeighbours.clear();
			mNearestNeighbours.push_back(aNeighbour);
			mNeighbourDistance = aDistance;
			return true;
		}
		return false;
	}

	bool tryNeighbour( GraphPoint &aNeighbour )
	{
		return tryNeighbour( &aNeighbour, mPoint->minDescriptorDistance2( *aNeighbour.mPoint ) );
	}

	const REAL8 *descriptor() const
	{
		ELISE_DEBUG_ERROR( mPoint==NULL, "GraphPoint::descriptor", "mPoint==NULL" );
		ELISE_DEBUG_ERROR( mPoint->entries.size()!=1, "GraphPoint::descriptor", "mPoint->entries.size()!=1" );
		return mPoint->entries[0].descriptor;
	}

	void addBackwardNeighbours()
	{
		list<GraphPoint*>::const_iterator itNeighbour = mNearestNeighbours.begin();
		while ( itNeighbour!=mNearestNeighbours.end() ) (**itNeighbour++).mBackwardNeighbours.push_back(this);
	}

	bool isNeighbour( const GraphPoint *aPoint ) const
	{
		list<GraphPoint*>::const_iterator itNeighbour = mNearestNeighbours.begin();
		while ( itNeighbour!=mNearestNeighbours.end() )
			if ( *itNeighbour++==aPoint ) return true;
		return false;
	}

	bool isBackwardNeighbour( const GraphPoint *aPoint ) const
	{
		list<GraphPoint*>::const_iterator itNeighbour = mBackwardNeighbours.begin();
		while ( itNeighbour!=mBackwardNeighbours.end() )
			if ( *itNeighbour++==this ) return true;
		return false;
	}

	size_t removeOneWayLinks()
	{
		size_t nbRemoved = 0;
		list<GraphPoint*>::iterator itNeighbour = mNearestNeighbours.begin();
		while ( itNeighbour!=mNearestNeighbours.end() )
		{
			if ( !(**itNeighbour).isNeighbour(this) )
			{
				itNeighbour = mNearestNeighbours.erase(itNeighbour);
				nbRemoved++;
			}
			else
				itNeighbour++;
		}
		return nbRemoved;
	}

	void toPointMatchVector( PointMatch *&oMatches )
	{
		if (mHasBeenWritten) return;

		const Pt2dr p(mPoint->x, mPoint->y);
		list<GraphPoint*>::iterator itSrc = mNearestNeighbours.begin();
		while ( itSrc!=mNearestNeighbours.end() )
		{
			GraphPoint &neighbour = **itSrc++;
			if ( neighbour.mHasBeenWritten ) continue;

			const DigeoPoint &p1 = *neighbour.mPoint;
			*oMatches++ = PointMatch(p, Pt2dr(p1.x, p1.y));

			mHasBeenWritten = true;
			neighbour.mHasBeenWritten = true;
		}
	}
};

void print_nbNeighbours_histogram( const vector<GraphPoint> &aVector )
{
	cout << "histogram:" << endl;

	if ( aVector.size()==0 ) return;

	size_t maxNbNeighbours = aVector[0].mBackwardNeighbours.size();
	for ( size_t i=1; i<aVector.size(); i++ )
		if ( aVector[i].mBackwardNeighbours.size()>maxNbNeighbours ) maxNbNeighbours = aVector[i].mBackwardNeighbours.size();

	vector<size_t> histogram( size_t(maxNbNeighbours+1), 0 );
	for ( size_t i=1; i<aVector.size(); i++ )
		histogram[ aVector[i].mBackwardNeighbours.size() ]++;

	for ( size_t i=0; i<histogram.size(); i++ )
		cout << i << ": " << histogram[i] << endl;
}

void print_irregulars( vector<GraphPoint> &aPoints )
{
	for ( size_t iPoint=0; iPoint<aPoints.size(); iPoint++ )
	{
		GraphPoint &point = aPoints[iPoint];
		const size_t nbBackwardNeigbours = point.mBackwardNeighbours.size();
		if ( nbBackwardNeigbours!=1 || nbBackwardNeigbours!=1 )
		{
			cout << iPoint << "/" << aPoints.size()-1 << ": "  << point.mNearestNeighbours.size() << " neighbours, " << nbBackwardNeigbours << " backward neighbours, "
			     << point.mPoint->entries.size() << " descriptors" << endl;

			//~ ELISE_DEBUG_ERROR( point.mPoint->entries.size()==0, "print_irregulars", "point.mPoint->entries.size()==0" );
//~ 
			//~ const REAL8 *descriptor = point.mPoint->entries[0].descriptor;
			//~ print_v128( "", descriptor );
			//~ list<GraphPoint*>::iterator itNeighbour = point.mBackwardNeighbours.begin();
			//~ size_t iNeighbour = 0;
			//~ while ( itNeighbour!=point.mBackwardNeighbours.end() )
			//~ {
				//~ cout << "\tbackward neighbour " << iNeighbour << endl;
				//~ ELISE_DEBUG_ERROR( (**itNeighbour).mPoint->entries.size()==0, "print_irregulars", "(**itNeighbour).mPoint->entries.size()==0" );
				//~ cout << "\t\t" << (**itNeighbour).mNeighbourDistance << endl;
				//~ print_v128( "\t\t", (**itNeighbour).mPoint->entries[0].descriptor );
				//~ print_is_different_v128( "\t\t", descriptor, (**itNeighbour).mPoint->entries[0].descriptor );
				//~ itNeighbour++;
				//~ iNeighbour++;
			//~ }
		}
	}
}

void count_types( const vector<DigeoPoint> &aPoints )
{
	size_t iPoint = aPoints.size();
	cout << "count_types: " << iPoint << " points" << endl;
	vector<size_t> count( DigeoPoint::nbDetectTypes, 0 );
	const DigeoPoint *itPoint = aPoints.data();
	while ( iPoint-- )
	{
		#ifdef __DEBUG
			int t = itPoint->type;
			if ( t<0 || ((unsigned int)t)>=DigeoPoint::nbDetectTypes ) ELISE_ERROR_EXIT( "count_types: invalid type: " << t );
		#endif
		count[(size_t)(*itPoint++).type]++;
	}

	for ( unsigned int i=0; i<DigeoPoint::nbDetectTypes; i++ )
		cout << '\t' << DetectType_to_string( (DigeoPoint::DetectType)i ) << "(" << i << "): " << count[i] << endl;
}

size_t nbMatches( const vector<GraphPoint> &aPoints )
{
	size_t result = 0;
	size_t iPoint = aPoints.size();
	const GraphPoint *it = aPoints.data();
	while ( iPoint-- ) result += (*it++).mNearestNeighbours.size();
	return result;
}

void toPointMatchVector( vector<GraphPoint> &oMatches, PointMatch *&oDst )
{
	size_t i = oMatches.size();
	GraphPoint *it = oMatches.data();
	while (i--) (*it++).toPointMatchVector(oDst);
}

class GraphPointPack
{
public:
	vector<DigeoPoint> mPoints0, mPoints1;
	vector<GraphPoint> mGraphPoints0, mGraphPoints1;
	bool mHasBeenWritten;

	GraphPointPack():mHasBeenWritten(false){}

	static void initGraphPoints( const vector<DigeoPoint> &aPoints, vector<GraphPoint> &oGraphPoints )
	{
		size_t iPoint = aPoints.size();
		oGraphPoints.resize(iPoint);
		const DigeoPoint *itPoint = aPoints.data();
		GraphPoint *itGraphPoint = oGraphPoints.data();
		while ( iPoint-- )
			(*itGraphPoint++).mPoint = itPoint++;
	}

	void initGraphPoints()
	{
		initGraphPoints(mPoints0, mGraphPoints0);
		initGraphPoints(mPoints1, mGraphPoints1);
	}

	static void attemptMatching( vector<GraphPoint> &aPoints0, vector<GraphPoint> &aPoints1 )
	{
		size_t iPoint0 = aPoints0.size();
		GraphPoint *itPoint0 = aPoints0.data();
		while ( iPoint0-- )
		{
			GraphPoint *itPoint1 = aPoints1.data();
			size_t iPoint1 = aPoints1.size();
			while ( iPoint1-- ) itPoint0->tryNeighbour( *itPoint1++ );
			itPoint0++;
		}
	}

	void attemptMatchingBothWays()
	{
		attemptMatching(mGraphPoints0,mGraphPoints1);
		attemptMatching(mGraphPoints1,mGraphPoints0);
	}

	size_t nbMatches() const
	{
		return ::nbMatches(mGraphPoints0); //+::nbMatches(mGraphPoints1);
	}

	void toPointMatchVector( vector<PointMatch> &oMatches )
	{
		oMatches.resize(nbMatches());
		PointMatch *itDst = oMatches.data();

		if (mHasBeenWritten) resetHasBeenWritten();

		::toPointMatchVector(mGraphPoints0, itDst);
		//~ ::toPointMatchVector(mGraphPoints1, itDst);
	}

	static void resetHasBeenWritten( vector<GraphPoint> &aPoints )
	{
		GraphPoint *itPoint = aPoints.data();
		size_t iPoint = aPoints.size();
		while ( iPoint-- ) (*itPoint++).mHasBeenWritten = false;
	}

	void resetHasBeenWritten()
	{
		resetHasBeenWritten(mGraphPoints0);
		resetHasBeenWritten(mGraphPoints1);
	}

	void writeMatches( const string &aFilename )
	{
		vector<PointMatch> pointMatches;
		toPointMatchVector(pointMatches);
		mHasBeenWritten = writeMatchesFile(aFilename, pointMatches);
	}

	static void addBackwardNeighbours( vector<GraphPoint> &aPoints )
	{
		GraphPoint *itPoint = aPoints.data();
		size_t iPoint = aPoints.size();
		while ( iPoint-- ) (*itPoint++).addBackwardNeighbours();
	}

	void addBackwardNeighbours()
	{
		addBackwardNeighbours(mGraphPoints0);
		addBackwardNeighbours(mGraphPoints1);
	}

	static void clearBackwardNeigbours( vector<GraphPoint> &aPoints )
	{
		GraphPoint *itPoint = aPoints.data();
		size_t iPoint = aPoints.size();
		while ( iPoint-- ) (*itPoint++).mBackwardNeighbours.clear();
	}

	void clearBackwardNeigbours()
	{
		clearBackwardNeigbours(mGraphPoints0);
		clearBackwardNeigbours(mGraphPoints1);
	}

	void getBackwardNeighbours()
	{
		clearBackwardNeigbours();
		addBackwardNeighbours();
	}

	static size_t removeOneWayLinks( vector<GraphPoint> &aPoints )
	{
		size_t nbRemoved = 0;
		GraphPoint *itPoint = aPoints.data();
		size_t iPoint = aPoints.size();
		while ( iPoint-- ) nbRemoved += (*itPoint++).removeOneWayLinks();
		return nbRemoved;
	}

	size_t removeOneWayLinks()
	{
		size_t nbRemoved = removeOneWayLinks(mGraphPoints0);
		nbRemoved += removeOneWayLinks(mGraphPoints1);
		return nbRemoved;
	}
};

void remove_identical_entries( vector<DigeoPoint> &aPoints )
{
	size_t iPoint0 = aPoints.size();
	DigeoPoint *itPoint0 = &aPoints.back();

	if ( iPoint0<2 ) return;
	iPoint0--;

	const size_t descriptorSize = 128*sizeof(REAL8);
	size_t nbSuppressedEntries = 0, nbToSuppress = 0;
	while ( iPoint0-- )
	{
		for ( size_t iEntry0=0; iEntry0<itPoint0->entries.size(); iEntry0++ )
		{
			const REAL8 *descriptor0 = itPoint0->entries[iEntry0].descriptor;
			DigeoPoint *itPoint1 = itPoint0-1;
			size_t iPoint1 = iPoint0;
			while ( iPoint1-- )
			{
				vector<DigeoPoint::Entry>::iterator itEntry1 = itPoint1->entries.begin();
				while ( itEntry1!=itPoint1->entries.end() )
				{
					if ( memcmp( descriptor0, itEntry1->descriptor, descriptorSize )==0 )
					{
						itEntry1 = itPoint1->entries.erase(itEntry1);
						nbSuppressedEntries++;
					}
					else
						itEntry1++;
				}
			}
			if ( itPoint1->entries.size()==0 ) nbToSuppress++;
			itPoint1--;
		}
		itPoint0--;
	}

	cout << nbSuppressedEntries << " suppressed entries" << endl;
	cout << nbToSuppress << " points to suppress" << endl;

	vector<DigeoPoint> points( aPoints.size()-nbToSuppress );
	itPoint0 = aPoints.data();
	DigeoPoint *itPoint1 = points.data();
	iPoint0 = aPoints.size();
	while ( iPoint0-- )
	{
		if ( itPoint0->entries.size()!=0 ) *itPoint1++ = *itPoint0;
		itPoint0++;
	}

	ELISE_DEBUG_ERROR( itPoint1!=points.data()+points.size(), "remove_identical_entries", "itPoint1!=points.data()+points.size()" );
	points.swap(aPoints);
}

void loadDigeoPoints( const string &aFilename, vector<DigeoPoint> &oPoints )
{
	if ( !ELISE_fp::exist_file(aFilename) ) ELISE_ERROR_EXIT( "loadDigeoPoints: digeo file [" << aFilename << "] does not exist" );
	if ( !DigeoPoint::readDigeoFile(aFilename,true,oPoints) ) ELISE_ERROR_EXIT( "loadDigeoPoints: cannot read point list from digeo file [" << aFilename << ']' );

	//~ remove_identical_entries(oPoints);
	count_types(oPoints);

	cout << "[" << aFilename << "]: " << oPoints.size() << " points" << endl;
}

void writeMatches( GraphPointPack &oPack, const string &aFilename )
{
	oPack.writeMatches(aFilename);
	if ( !oPack.mHasBeenWritten ) ELISE_ERROR_EXIT( "cannot open file [" << aFilename << "] to write matches" );
	ELISE_DEBUG_ERROR( !ELISE_fp::exist_file(aFilename), "writeMatches", "!ELISE_fp::exist_file(" << aFilename << ")" );
}

int main( int argc, char **argv )
{
	if ( argc<4 ) ELISE_ERROR_EXIT( "usage: points_filename0 points_filename1 output_filename" );

	string srcFilename0 = argv[1];
	string srcFilename1 = argv[2];
	string dstFilename = argv[3];

	if ( !ELISE_fp::exist_file(srcFilename1) ) ELISE_ERROR_RETURN( "input file [" << srcFilename1 << "] does not exist" );

	GraphPointPack matchPack;
	loadDigeoPoints( srcFilename0, matchPack.mPoints0 );
	loadDigeoPoints( srcFilename1, matchPack.mPoints1 );
	matchPack.initGraphPoints();
	matchPack.attemptMatchingBothWays();
	cout << matchPack.removeOneWayLinks() << " one-way links removed" << endl;
	writeMatches(matchPack,dstFilename);

	matchPack.getBackwardNeighbours();
	print_nbNeighbours_histogram(matchPack.mGraphPoints0);
	print_nbNeighbours_histogram(matchPack.mGraphPoints1);

	print_irregulars(matchPack.mGraphPoints0);
	print_irregulars(matchPack.mGraphPoints1);

	return EXIT_SUCCESS;
}
