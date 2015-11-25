#include <math.h>

#include "AnnSearcher.h"

using namespace std;

#ifdef __ANN_SEARCHER_TIME
	#include <sys/time.h>

	double g_construct_tree_time = 0;
	double g_query_time = 0;

	typedef struct timeval timeval_t;
	timeval_t g_chrono_start_time; 
	void start_chrono(){ gettimeofday( &g_chrono_start_time, NULL ); }

	double get_chrono_time()
	{
		timeval_t t;
		gettimeofday( &t, NULL );
		return (t.tv_sec-g_chrono_start_time.tv_sec)*1000.+(t.tv_usec-g_chrono_start_time.tv_usec)/1000.;
	}
#endif

//
// AnnArray class
//

// fills m_annArray with pointers to i_array's data
void AnnArray::set( vector<DigeoPoint> &i_array, SIFT_ANN_SEARCH_MODE i_mode )
{
	m_annArray.resize( i_array.size() );
	m_searchMode = i_mode;
	if ( i_array.size()==0 ) return;
    // array sizes are supposed to be equal
    ANNpoint  *itANN  = m_annArray.data();
    DigeoPoint *itSift = &i_array[0];
    size_t     iPoint = m_annArray.size();
    switch( i_mode )
    {
		case SIFT_ANN_DESC_SEARCH:
			// fills annArray for search based on descriptors
			while ( iPoint-- )
				( *itANN++ ) = ( itSift++ )->descriptor(0);
			return;
		case SIFT_ANN_2D_SEARCH:
			// fills annArray for search based on 2d coordinates
			while ( iPoint-- )
				( *itANN++ ) = &( ( itSift++ )->x );
			return;
		default:;
			#ifdef __DEBUG_ANN_SEARCHER
				cerr << "ERROR: SiftAnnArray::fillAnnArray: search mode is not handled" << endl;
			#endif
	}
}

//
// class AnnSearch
//

void AnnSearcher::setNbNeighbours( int i_nbNeighbours )
{
    if ( i_nbNeighbours<0 ){
        #if (_VERBOSE>1)
            cerr << "ERROR: AnnSearcher::setNbNeighbours [" << i_nbNeighbours << "<0]" << endl;
        #endif
        m_nbNeighbours = 0;
    }
    else
        m_nbNeighbours = i_nbNeighbours;

    if ( (int)m_neighboursIndices.size()<m_nbNeighbours )
    {
        m_neighboursIndices.resize( m_nbNeighbours );
        m_neighboursDistances.resize( m_nbNeighbours );
    }
}

void AnnSearcher::createTree( AnnArray &i_dataArray )
{
	#ifdef __ANN_SEARCHER_TIME
		start_chrono();
	#endif

	clearTree();

	int spaceSize = -1; // dimension of space

	switch ( i_dataArray.getSearchMode() )
	{
		case SIFT_ANN_DESC_SEARCH: spaceSize=SIFT_DESCRIPTOR_SIZE; break;
		case SIFT_ANN_2D_SEARCH: spaceSize=2; break;
		default:;
			#ifdef __DEBUG_ANN_SEARCHER
				cerr << "ERROR: AnnSearcher::createTree called with an unhandled search mode" << endl;
			#endif
	}
	#ifdef __DEBUG_ANN_SEARCHER
		if ( i_dataArray.size()==0 ) cerr << "ERROR: AnnSearcher::createTree called with an empty array, cannot construct a tree" << endl;
	#endif

	m_kdTree = new ANNkd_tree( i_dataArray.getANNpointArray(), i_dataArray.size(), spaceSize );

	#ifdef __ANN_SEARCHER_TIME
		g_construct_tree_time += get_chrono_time();
	#endif
}

void AnnSearcher::search( ANNpoint i_point )
{
	#ifdef __ANN_SEARCHER_TIME
		start_chrono();
	#endif

	if ( m_kdTree!=NULL )
		//kdTree->annkSearch(
		m_kdTree->annkPriSearch( i_point, m_nbNeighbours, m_neighboursIndices.data(), m_neighboursDistances.data(), m_errorBound );
	#ifdef __DEBUG_ANN_SEARCHER
	else
		cerr << "ERROR: AnnSearcher::search trying to search in a null tree" << endl;
	#endif

	#ifdef __ANN_SEARCHER_TIME
		g_query_time += get_chrono_time();
	#endif
}

//
// function related to SiftAnnArray class
//

// perform a matching between the points of the two lists
// this function may change the search mode of i_arrayData
void match_lebris( vector<DigeoPoint> &i_array0, vector<DigeoPoint> &i_array1, std::list<V2I> &o_matchingCouples,
                   double i_closenessRatio, int i_nbMaxPriPoints )
{
    o_matchingCouples.clear();
    
    if ( i_array0.size()==0 || i_array1.size()==0 ) return;

    AnnArray dataArray( i_array0, SIFT_ANN_DESC_SEARCH );

    AnnSearcher anns;
    anns.setNbNeighbours( 2 );
    anns.setErrorBound( 0. );
    anns.setMaxVisitedPoints( i_nbMaxPriPoints );
    anns.createTree( dataArray );

    ANNdist R = i_closenessRatio*i_closenessRatio;

    const ANNidx *neighIndices   = anns.getNeighboursIndices();
    const ANNdist *neighDistances = anns.getNeighboursDistances();

	DigeoPoint *itQuery = &i_array1[0];
	int         nbQueries = (int)i_array1.size(),
	            iQuery;

	for ( iQuery=0; iQuery<nbQueries; iQuery++ )
	{
		anns.search( itQuery->descriptor(0) );

		#ifdef __DEBUG_ANN_SEARCHER
			if ( neighIndices[0]==-1 || neighIndices[1]==-1 )
				 cerr << "Ann: match_lebris: invalid neighbour found " << neighIndices[0] << '(' << neighDistances[0] << ") " << neighIndices[1] << '(' << neighDistances[0] << ')' << endl;
			if ( itQuery->type==i_array0[neighIndices[0]].type )
				cerr << "WARNING: a point of type " << DetectType_to_string(itQuery->type) << " matched with a point of type " << DetectType_to_string(i_array0[neighIndices[0]].type) << endl;
		#endif

		if ( itQuery->type==i_array0[neighIndices[0]].type && // matchings of points of different types are discarded
		     neighDistances[0]<( R*neighDistances[1] ) )
			o_matchingCouples.push_back( V2I( neighIndices[0], iQuery ) );

		itQuery++;
	}
}

// print a list of matching points 2d coordinates
bool write_matches_ascii( const std::string &i_filename, const vector<DigeoPoint> &i_array0, const vector<DigeoPoint> &i_array1, const list<V2I> &i_matchingCouples, bool i_appendToFile )
{
	ofstream f( i_filename.c_str(), i_appendToFile?ios::out|ios::app:ios::out );
	if ( !f ) return false;
	f.precision(6);
	list<V2I>::const_iterator itCouple = i_matchingCouples.begin();
	const DigeoPoint *p0 = i_array0.data(),
	                 *q0 = i_array1.data(),
	                 *p, *q;
	while ( itCouple!=i_matchingCouples.end() )
	{
		p = p0+itCouple->x;
		q = q0+( itCouple++ )->y;
		f << p->x << '\t' << p->y << '\t' << q->x << '\t' << q->y << endl;
	}
	return true;
}

// unfold couples described in the i_matchedCoupleIndices list and split data in the two arrays io_array0, io_array1
void unfoldMatchingCouples( vector<DigeoPoint> &io_array0, vector<DigeoPoint> &io_array1, const list<V2I> &i_matchedCoupleIndices )
{
	static vector<DigeoPoint> array0, array1;

	size_t iCouple = i_matchedCoupleIndices.size();
	array0.resize( iCouple );
	array1.resize( iCouple );
	if ( iCouple==0 ) return;
	list<V2I>::const_iterator itCouple = i_matchedCoupleIndices.begin();
	const DigeoPoint *p0 = io_array0.data(),
	                 *q0 = io_array1.data();
	DigeoPoint *itArray0 = &array0[0],
	           *itArray1 = &array1[0];
    while ( iCouple-- )
    {
        *itArray0++ = p0[itCouple->x];
        *itArray1++ = q0[( itCouple++ )->y];
    }
    io_array0.swap( array0 );
    io_array1.swap( array1 );
}

// count number of elements of A which are also in B
// this is also the number of elements common to A and B if neither of the vectors has duplicated elements
static inline int _count_A_in_B( const vector<int> &A, const vector<int> &B)
{
    int count = 0;
    int nbA = (int)A.size(),
        nbB = (int)B.size(),
        iB;
    vector<int>::const_iterator itA = A.begin(),
                                itB;
    while ( nbA-- )
    {
        iB  = nbB;
        itB = B.begin();
        while ( iB-- ){
            if ( (*itB++)==(*itA) ){ count++; break; }
        }
        itA++;
    }
    return count;
}

// returns a vector of the i_nbNeighbours nearest neighbours (euclidean distance)
// this method may change the search mode
void getNeighbours( vector<DigeoPoint> &i_array, vector<vector<ANNidx> > &o_neighbourhood, int i_nbNeighbours )
{
    vector<ANNidx> neighbours( i_nbNeighbours );

    int k = i_nbNeighbours+1; // number of nearest neighbors for ANN search (+1 beacuse the point itself while be found as a neighbour)
    AnnSearcher anns;
    anns.setNbNeighbours( k );
    anns.setErrorBound( 0. );
    anns.setMaxVisitedPoints( SIFT_ANN_DEFAULT_MAX_PRI_POINTS );
    AnnArray dataArray( i_array, SIFT_ANN_2D_SEARCH );
    anns.createTree( dataArray );

    const ANNidx *neighIndices = anns.getNeighboursIndices();

    vector<int>::iterator           itNeighbour;
    int                             iNeighbour, iQuery;
    vector<vector<int> >::iterator  itQueryNeighbourhood = o_neighbourhood.begin();
    DigeoPoint                     *itQuery              = &i_array[0];
    int                             nbQueries            = (int)i_array.size();
    for ( iQuery=0; iQuery<nbQueries; iQuery++ )
    {
        anns.search( &itQuery->x );

        // fill a vector with query point nearest neighbours' indices
        #ifdef __DEBUG_ANN_SEARCHER
            vector<int>::iterator neigh_end = ( *itQueryNeighbourhood ).end();
        #endif
        itNeighbour = ( *itQueryNeighbourhood++ ).begin();
        for ( iNeighbour=0; iNeighbour<k; iNeighbour++ )
        {
            if ( neighIndices[iNeighbour]!=iQuery ) // do not add query point's index as a neighbour index
                *itNeighbour++ = neighIndices[iNeighbour];
        }
        /*
        #ifdef __DEBUG_ANN_SEARCHER
            // check we ignored exactly one neighbour
            if ( itNeighbour!=neigh_end ) cerr << "Ann: WARN: getNeighbours : a point has not been found as its nearest neighbour or has been found more than once" << endl;
        #endif
        */

        itQuery++;
    }
}

// check if more than i_ratio of a point's neighbours are homologue to its homologue's neighbours
// if not, both the point and its homologue are erased
// this function may change i_array0 and/or i_array1 search mode
void neighbourFilter( vector<DigeoPoint> &i_array0, vector<DigeoPoint> &i_array1, list<V2I> &o_keptCouples, double i_ratio )
{
    o_keptCouples.clear();

    if ( i_array0.size()==0 ) return;

    #ifdef __DEBUG_ANN_SEARCHER
        if ( i_array0.size()!=i_array1.size() )
            cerr << "PROG_WARN : neighbourFilter: i_array0 and i_array1 are of different size" << endl;
    #endif

    int nbCouples = (int)i_array0.size();
    //vector<int> neighbours( SIFT_ANN_DEFAULT_NB_NEIGHBOURS );
	vector<int> neighbours( SIFT_ANN_DEFAULT_NB_NEIGHBOURS+1 ); // TODO : voir pourquoi un point n'est pas toujours dans ses 8 plus proches voisins
    vector<vector<int> > neighbours0( nbCouples, neighbours ),
                         neighbours1( nbCouples, neighbours );
    getNeighbours( i_array0, neighbours0, SIFT_ANN_DEFAULT_NB_NEIGHBOURS );
    getNeighbours( i_array1, neighbours1, SIFT_ANN_DEFAULT_NB_NEIGHBOURS );
    double nbMinHomologueNeighbours = i_ratio*neighbours0[0].size(),
           nbHomologueNeighbours;

    int iCouple;
    vector<vector<int> >::const_iterator itNeigh0 = neighbours0.begin(),
                                         itNeigh1 = neighbours1.begin();
    for ( iCouple=0; iCouple<nbCouples; iCouple++ )
    {
        // count the number of indices in both lists.
        // points with the same index are considered homologue.
        nbHomologueNeighbours = (double)_count_A_in_B( *itNeigh0++, *itNeigh1++ );

        if ( nbHomologueNeighbours>nbMinHomologueNeighbours )
            o_keptCouples.push_back( V2I( iCouple, iCouple ) );
    }
}


//-------------------------------------------------
// DigeoTypedVectors methods
//-------------------------------------------------

DigeoTypedVectors::DigeoTypedVectors( const vector<DigeoPoint> &i_points ):
	m_points( (int)DigeoPoint::DETECT_UNKNOWN+1 )
{
	// count the number of points of each type
	vector<unsigned int> countTypes(DigeoPoint::nbDetectTypes, 0);
	const DigeoPoint *itSrc = i_points.data();
	size_t iPoint = i_points.size();
	while ( iPoint-- ) countTypes[(size_t)( *itSrc++ ).type]++;

	// resize vectors
	m_points.resize(DigeoPoint::nbDetectTypes);
	for ( unsigned int iType=0; iType<DigeoPoint::nbDetectTypes; iType++ )
	{
		m_points[iType].resize(countTypes[iType]);
		countTypes[iType] = 0;
	}

	// copy points
	itSrc = i_points.data();
	iPoint = i_points.size();
	while ( iPoint-- )
	{
		const size_t iType = (size_t)itSrc->type;
		m_points[iType][countTypes[iType]++] = *itSrc;
		itSrc++;
	}

	#ifdef __DEBUG_ANN_SEARCHER
		size_t total = m_points[0].size();
		for ( size_t iType=1; iType<DigeoPoint::nbDetectTypes; iType++ )
			total += m_points[iType].size();
		if ( total!=i_points.size() )
		{
			cerr << "DEBUG_ERROR: DigeoTypedVectors::DigeoTypedVectors: split is incorrect" << endl;
			exit(1);
		}
	#endif
}
