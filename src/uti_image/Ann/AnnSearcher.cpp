#include <math.h>

#include "AnnSearcher.h"

using namespace std;

typedef Siftator::SiftPoint SiftPoint;

//
// AnnArray class
//

// fills m_annArray with pointers to i_array's data
void AnnArray::set( vector<SiftPoint> &i_array, SIFT_ANN_SEARCH_MODE i_mode )
{
	m_annArray.resize( i_array.size() );
	m_searchMode = i_mode;
	if ( i_array.size()==0 ) return;
    // array sizes are supposed to be equal
    ANNpoint  *itANN  = m_annArray.data();
    SiftPoint *itSift = &i_array[0];
    size_t     iPoint = m_annArray.size();
    switch( i_mode )
    {
		case SIFT_ANN_DESC_SEARCH:
			// fills annArray for search based on descriptors
			while ( iPoint-- )
				( *itANN++ ) = ( itSift++ )->descriptor;
			return;
		case SIFT_ANN_2D_SEARCH:
			// fills annArray for search based on 2d coordinates
			while ( iPoint-- )
				( *itANN++ ) = &( ( itSift++ )->x );
			return;
		default:;
			#ifdef _DEBUG
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
    clearTree();

    int spaceSize = -1; // dimension of space

	switch ( i_dataArray.getSearchMode() )
	{
		case SIFT_ANN_DESC_SEARCH: spaceSize=m_descriptorSize; break;
		case SIFT_ANN_2D_SEARCH: spaceSize=2; break;
		default:;
			#ifdef _DEBUG
				cerr << "ERROR: AnnSearcher::createTree called with an unhandled search mode" << endl;
			#endif
	}
	#ifdef _DEBUG
		if ( i_dataArray.size()==0 ) cerr << "ERROR: AnnSearcher::createTree called with an empty array, cannot construct a tree" << endl;
	#endif
            
    m_kdTree = new ANNkd_tree( i_dataArray.getANNpointArray(), i_dataArray.size(), spaceSize );
}

void AnnSearcher::search( ANNpoint i_point )
{
	if ( m_kdTree!=NULL )
		//kdTree->annkSearch(
		m_kdTree->annkPriSearch( i_point, m_nbNeighbours, m_neighboursIndices.data(), m_neighboursDistances.data(), m_errorBound );
	#ifdef _DEBUG
	else
		cerr << "ERROR: AnnSearcher::search trying to search in a null tree" << endl;
	#endif
}

//
// function related to SiftAnnArray class
//

// perform a matching between the points of the two lists
// this function may change the search mode of i_arrayData
void match_lebris( vector<SiftPoint> &i_array0, vector<SiftPoint> &i_array1, std::list<V2I> &o_matchingCouples,
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

	SiftPoint *itQuery = &i_array1[0];
	int        nbQueries = i_array1.size(),
			   iQuery;

	for ( iQuery=0; iQuery<nbQueries; iQuery++ )
	{
		anns.search( itQuery->descriptor );

		#ifdef _DEBUG
			if ( neighIndices[0]==-1 || neighIndices[1]==-1 )
				 cerr << "Ann: match_lebris: invalid neighbour found " << neighIndices[0] << '(' << neighDistances[0] << ") " << neighIndices[1] << '(' << neighDistances[0] << ')' << endl;
		#endif
		
		if ( neighDistances[0]<( R*neighDistances[1] ) )
			o_matchingCouples.push_back( V2I( neighIndices[0], iQuery ) );

		itQuery++;
	}
}

// print a list of matching points 2d coordinates
bool write_matches_ascii( const std::string &i_filename, const vector<SiftPoint> &i_array0, const vector<SiftPoint> &i_array1, const list<V2I> &i_matchingCouples )
{
	ofstream f( i_filename.c_str() );
	if ( !f ) return false;
	f.precision(6);
    list<V2I>::const_iterator itCouple = i_matchingCouples.begin();
    const SiftPoint *p0 = i_array0.data(),
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
void unfoldMatchingCouples( vector<SiftPoint> &io_array0, vector<SiftPoint> &io_array1, const list<V2I> &i_matchedCoupleIndices )
{
    static vector<SiftPoint> array0, array1;

    size_t iCouple = i_matchedCoupleIndices.size();
	array0.resize( iCouple );
	array1.resize( iCouple );
	if ( iCouple==0 ) return;
    list<V2I>::const_iterator itCouple = i_matchedCoupleIndices.begin();
    const SiftPoint *p0 = io_array0.data(),
					*q0 = io_array1.data();
    SiftPoint *itArray0 = &array0[0],
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
    int nbA = A.size(),
        nbB = B.size(),
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
void getNeighbours( vector<SiftPoint> &i_array, vector<vector<ANNidx> > &o_neighbourhood, int i_nbNeighbours )
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
    SiftPoint 						*itQuery             = &i_array[0];
    int                             nbQueries            = i_array.size();
    for ( iQuery=0; iQuery<nbQueries; iQuery++ )
    {
        anns.search( &itQuery->x );

        // fill a vector with query point nearest neighbours' indices
        #ifdef _DEBUG
			vector<int>::iterator neigh_end = ( *itQueryNeighbourhood ).end();
        #endif
        itNeighbour = ( *itQueryNeighbourhood++ ).begin();
        for ( iNeighbour=0; iNeighbour<k; iNeighbour++ )
        {
            if ( neighIndices[iNeighbour]!=iQuery ) // do not add query point's index as a neighbour index
                *itNeighbour++ = neighIndices[iNeighbour];
        }
        #ifdef _DEBUG
            // check we ignored exactly one neighbour
            if ( itNeighbour!=neigh_end ) cerr << "Ann: WARN: getNeighbours : a point has not been found as its nearest neighbour or has been found more than once" << endl;
        #endif

        itQuery++;
    }
}

// check if more than i_ratio of a point's neighbours are homologue to its homologue's neighbours
// if not, both the point and its homologue are erased
// this function may change i_array0 and/or i_array1 search mode
void neighbourFilter( vector<SiftPoint> &i_array0, vector<SiftPoint> &i_array1, list<V2I> &o_keptCouples, double i_ratio )
{
    o_keptCouples.clear();

    if ( i_array0.size()==0 ) return;

    #ifdef _DEBUG
        if ( i_array0.size()!=i_array1.size() )
            cerr << "PROG_WARN : neighbourFilter: i_array0 and i_array1 are of different size" << endl;
    #endif

    int nbCouples = i_array0.size();
    vector<int> neighbours( SIFT_ANN_DEFAULT_NB_NEIGHBOURS );
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
