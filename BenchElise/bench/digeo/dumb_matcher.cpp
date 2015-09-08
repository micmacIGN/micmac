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

class MatchPoint
{
public:
	const DigeoPoint *mPoint;
	list<MatchPoint*> mNearestNeighbours;
	double mNeighbourDistance;

	// statistic data
	list<MatchPoint*> mBackwardNeighbours; // a list of points considering this point as a neighbours
	bool mHasBeenWritten;

	MatchPoint( const DigeoPoint *aPoint=NULL ):
		mPoint(aPoint),
		mNeighbourDistance( numeric_limits<REAL>::max() ),
		mHasBeenWritten(false){}

	bool tryNeighbour( MatchPoint *aNeighbour, double aDistance )
	{
		if ( aDistance==mNeighbourDistance )
			mNearestNeighbours.push_back(aNeighbour);
		else if ( aDistance<mNeighbourDistance )
		{
			mNearestNeighbours.clear();
			mNearestNeighbours.push_back(aNeighbour);
			mNeighbourDistance = aDistance;
			return true;
		}
		return false;
	}

	bool tryNeighbour( MatchPoint &aNeighbour )
	{
		return tryNeighbour( &aNeighbour, mPoint->minDescriptorDistance2( *aNeighbour.mPoint ) );
	}

	const REAL8 *descriptor() const
	{
		ELISE_DEBUG_ERROR( mPoint==NULL, "MatchPoint::descriptor", "mPoint==NULL" );
		ELISE_DEBUG_ERROR( mPoint->entries.size()!=1, "MatchPoint::descriptor", "mPoint->entries.size()!=1" );
		return mPoint->entries[0].descriptor;
	}

	void writeMatches( ostream &aStream )
	{
		if (mHasBeenWritten) return;

		list<MatchPoint*>::const_iterator itNeighbour = mNearestNeighbours.begin();
		while ( itNeighbour!=mNearestNeighbours.end() )
		{
			MatchPoint &neighbour = **itNeighbour++;
			const DigeoPoint &p1 = *neighbour.mPoint;
			aStream << mPoint->x << ' ' << mPoint->y << ' ' << p1.x << ' ' << p1.y << endl;
			mHasBeenWritten = true;
			neighbour.mHasBeenWritten = true;
		}
	}

	void addBackwardNeighbours()
	{
		list<MatchPoint*>::const_iterator itNeighbour = mNearestNeighbours.begin();
		while ( itNeighbour!=mNearestNeighbours.end() ) (**itNeighbour++).mBackwardNeighbours.push_back(this);
	}

	bool isNeighbour( const MatchPoint *aPoint ) const
	{
		list<MatchPoint*>::const_iterator itNeighbour = mNearestNeighbours.begin();
		while ( itNeighbour!=mNearestNeighbours.end() )
			if ( *itNeighbour++==aPoint ) return true;
		return false;
	}

	bool isBackwardNeighbour( const MatchPoint *aPoint ) const
	{
		list<MatchPoint*>::const_iterator itNeighbour = mBackwardNeighbours.begin();
		while ( itNeighbour!=mBackwardNeighbours.end() )
			if ( *itNeighbour++==this ) return true;
		return false;
	}

	size_t removeOneWayLinks()
	{
		size_t nbRemoved = 0;
		list<MatchPoint*>::iterator itNeighbour = mNearestNeighbours.begin();
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
};

void print_nbNeighbours_histogram( const vector<MatchPoint> &aVector )
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

void print_irregulars( vector<MatchPoint> &aPoints )
{
	for ( size_t iPoint=0; iPoint<aPoints.size(); iPoint++ )
	{
		MatchPoint &point = aPoints[iPoint];
		const size_t nbBackwardNeigbours = point.mBackwardNeighbours.size();
		if ( nbBackwardNeigbours!=1 || nbBackwardNeigbours!=1 )
		{
			cout << iPoint << "/" << aPoints.size()-1 << ": "  << point.mNearestNeighbours.size() << " neighbours, " << nbBackwardNeigbours << " backward neighbours, "
			     << point.mPoint->entries.size() << " descriptors" << endl;

			//~ ELISE_DEBUG_ERROR( point.mPoint->entries.size()==0, "print_irregulars", "point.mPoint->entries.size()==0" );
//~ 
			//~ const REAL8 *descriptor = point.mPoint->entries[0].descriptor;
			//~ print_v128( "", descriptor );
			//~ list<MatchPoint*>::iterator itNeighbour = point.mBackwardNeighbours.begin();
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

class MatchPointPack
{
public:
	vector<DigeoPoint> mPoints0, mPoints1;
	vector<MatchPoint> mMatchPoints0, mMatchPoints1;
	bool mHasBeenWritten;

	MatchPointPack():mHasBeenWritten(false){}

	static void initMatchPoints( const vector<DigeoPoint> &aPoints, vector<MatchPoint> &oMatchPoints )
	{
		size_t iPoint = aPoints.size();
		oMatchPoints.resize(iPoint);
		const DigeoPoint *itPoint = aPoints.data();
		MatchPoint *itMatchPoint = oMatchPoints.data();
		while ( iPoint-- )
			(*itMatchPoint++).mPoint = itPoint++;
	}

	void initMatchPoints()
	{
		initMatchPoints(mPoints0,mMatchPoints0);
		initMatchPoints(mPoints1,mMatchPoints1);
	}

	static void attemptMatching( vector<MatchPoint> &aPoints0, vector<MatchPoint> &aPoints1 )
	{
		size_t iPoint0 = aPoints0.size();
		MatchPoint *itPoint0 = aPoints0.data();
		while ( iPoint0-- )
		{
			MatchPoint *itPoint1 = aPoints1.data();
			size_t iPoint1 = aPoints1.size();
			while ( iPoint1-- ) itPoint0->tryNeighbour( *itPoint1++ );
			itPoint0++;
		}
	}

	void attemptMatchingBothWays()
	{
		attemptMatching(mMatchPoints0,mMatchPoints1);
		attemptMatching(mMatchPoints1,mMatchPoints0);
	}

	static void writeMatches( ostream &aStream, MatchPoint *aPoints, size_t aNbPoints )
	{
		while ( aNbPoints-- ) (*aPoints++).writeMatches(aStream);
	}

	static bool writeMatches( vector<MatchPoint> &aPoints0, vector<MatchPoint> &aPoints1, const string &aFilename )
	{
		ofstream f( aFilename.c_str() );
		if ( !f ) return false;
		writeMatches( f, aPoints0.data(), aPoints0.size() );
		writeMatches( f, aPoints1.data(), aPoints1.size() );

		//~ cout << "writeMatches: " << nbWritten << " matches written in [" << aFilename << ']' << endl;

		return true;
	}

	static void resetHasBeenWritten( vector<MatchPoint> &aPoints )
	{
		MatchPoint *itPoint = aPoints.data();
		size_t iPoint = aPoints.size();
		while ( iPoint-- ) (*itPoint++).mHasBeenWritten = false;
	}

	void resetHasBeenWritten()
	{
		resetHasBeenWritten(mMatchPoints0);
		resetHasBeenWritten(mMatchPoints1);
	}

	void writeMatches( const string &aFilename )
	{
		if (mHasBeenWritten) resetHasBeenWritten();
		mHasBeenWritten = writeMatches(mMatchPoints0,mMatchPoints1,aFilename);
	}

	static void addBackwardNeighbours( vector<MatchPoint> &aPoints )
	{
		MatchPoint *itPoint = aPoints.data();
		size_t iPoint = aPoints.size();
		while ( iPoint-- ) (*itPoint++).addBackwardNeighbours();
	}

	void addBackwardNeighbours()
	{
		addBackwardNeighbours(mMatchPoints0);
		addBackwardNeighbours(mMatchPoints1);
	}

	static void clearBackwardNeigbours( vector<MatchPoint> &aPoints )
	{
		MatchPoint *itPoint = aPoints.data();
		size_t iPoint = aPoints.size();
		while ( iPoint-- ) (*itPoint++).mBackwardNeighbours.clear();
	}

	void clearBackwardNeigbours()
	{
		clearBackwardNeigbours(mMatchPoints0);
		clearBackwardNeigbours(mMatchPoints1);
	}

	void getBackwardNeighbours()
	{
		clearBackwardNeigbours();
		addBackwardNeighbours();
	}

	static size_t removeOneWayLinks( vector<MatchPoint> &aPoints )
	{
		size_t nbRemoved = 0;
		MatchPoint *itPoint = aPoints.data();
		size_t iPoint = aPoints.size();
		while ( iPoint-- ) nbRemoved += (*itPoint++).removeOneWayLinks();
		return nbRemoved;
	}

	size_t removeOneWayLinks()
	{
		size_t nbRemoved = removeOneWayLinks(mMatchPoints0);
		nbRemoved += removeOneWayLinks(mMatchPoints1);
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

void writeMatches( MatchPointPack &oPack, const string &aFilename )
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

	MatchPointPack matchPack;
	loadDigeoPoints( srcFilename0, matchPack.mPoints0 );
	loadDigeoPoints( srcFilename1, matchPack.mPoints1 );
	matchPack.initMatchPoints();
	matchPack.attemptMatchingBothWays();
	cout << matchPack.removeOneWayLinks() << " one-way links removed" << endl;
	writeMatches(matchPack,dstFilename);

	matchPack.getBackwardNeighbours();
	print_nbNeighbours_histogram(matchPack.mMatchPoints0);
	print_nbNeighbours_histogram(matchPack.mMatchPoints1);

	print_irregulars(matchPack.mMatchPoints0);
	print_irregulars(matchPack.mMatchPoints1);

	return EXIT_SUCCESS;
}
