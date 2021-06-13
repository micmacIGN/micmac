#ifndef __ANN_SEARCHER__
#define __ANN_SEARCHER__

//#define __DEBUG_ANN_SEARCHER
//#define __ANN_SEARCHER_TIME

#include <vector>
#include <list>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

#include "../../../CodeExterne/ANN/include/ANN/ANN.h"

#include "../Digeo/DigeoPoint.h"
#include "ann_utils.h"

#define SIFT_POINT_UNDEFINED_INDEX -1

#define SIFT_BASE_TYPE ANNcoord

#ifndef SIFT_ANN_DEFAULT_NB_NEIGHBOURS
    #define SIFT_ANN_DEFAULT_NB_NEIGHBOURS 8
#endif

#ifndef SIFT_ANN_DEFAULT_CLOSENESS_RATIO
    #define SIFT_ANN_DEFAULT_CLOSENESS_RATIO 0.6
#endif

#ifndef SIFT_ANN_DEFAULT_MAX_PRI_POINTS
    #define SIFT_ANN_DEFAULT_MAX_PRI_POINTS 200
#endif

// a simple couple class to store matched point indices
class V2I{
public:
	size_t x, y;
	V2I(){}
	V2I( size_t xx, size_t yy ):x(xx),y(yy){}
};
	
typedef enum{
    SIFT_ANN_DESC_SEARCH, // ANN search based on sift descriptors
    SIFT_ANN_2D_SEARCH    // ANN search base on 2D coordinates
} SIFT_ANN_SEARCH_MODE;

// an array of DigeoPoint, viewable as an array of ANNpoint on 2d coordinates or on sift descriptors
// used as a seed for AnnSearcher::createTree
// setSearchMode is to switch between modes
class AnnArray
{
private:
    std::vector<ANNpoint>	m_annArray;
    SIFT_ANN_SEARCH_MODE	m_searchMode;

    // fills m_annArray with pointers to m_siftArray's data (2d coordinates or descriptors depending of te mode)
    void fillAnnArray();

public:
	inline AnnArray();
    inline AnnArray( std::vector<DigeoPoint> &i_siftArray, SIFT_ANN_SEARCH_MODE m_searchMode );

	void set( std::vector<DigeoPoint> &i_siftArray, SIFT_ANN_SEARCH_MODE m_searchMode );

	// getters
    inline       ANNpointArray getANNpointArray();
    inline const ANNpointArray getANNpointArray() const;
    inline SIFT_ANN_SEARCH_MODE getSearchMode() const;

    inline unsigned int size() const;

    // returns a vector of the i_nbNeighbours nearest neighbours (euclidean distance)
    // this method may change the search mode
    void getNeighbours( std::vector<std::vector<int> > &o_neighbourhood, int i_nbNeighbours=SIFT_ANN_DEFAULT_NB_NEIGHBOURS );
};

class AnnSearcher
{
private:
    int                  m_nbNeighbours;
    double               m_errorBound;
    std::vector<ANNidx>  m_neighboursIndices;
    std::vector<ANNdist> m_neighboursDistances;
    ANNkd_tree          *m_kdTree;

    void clearTree();
public:
    AnnSearcher();
    ~AnnSearcher();

    void setNbNeighbours( int i_nbNeighbours );
    void setErrorBound( double i_errorBound );
    void setMaxVisitedPoints( int i_nbMaxVisitedPoints );

    const ANNdist *getNeighboursDistances() const;
    const ANNidx  *getNeighboursIndices() const;

    void createTree( AnnArray &i_dataArray );

    void search( ANNpoint i_point );
};

// split a vector of DigeoPoint into n vectors of DigeoPoints of the same type
class DigeoTypedVectors
{
public:
	std::vector<std::vector<DigeoPoint> > m_points;

	DigeoTypedVectors( const std::vector<DigeoPoint> &i_points );
};

// try to match points of i_arrayQuery with points of i_arrayData
// this function may change the search mode of i_arrayData
void match_lebris( std::vector<DigeoPoint> &i_arrayData, std::vector<DigeoPoint> &i_arrayQuery, // arrays of points to be matched
                   std::list<V2I> &o_matchCouples,                              	// the returned list of indices to matching points in their respective array
                   double i_closenessRatio = SIFT_ANN_DEFAULT_CLOSENESS_RATIO,  	// a distance ratio to validate a matching
                   int i_nbMaxPriPoints    = SIFT_ANN_DEFAULT_MAX_PRI_POINTS ); 	// max number of point per search (with annkPriSearch)

bool write_matches_ascii( const std::string &i_filename, const std::vector<DigeoPoint> &i_array0, const std::vector<DigeoPoint> &i_array1, const std::list<V2I> &i_matchingCouples, bool i_appendToFile=false );

// unfold couples described in the i_matchedCoupleIndices list and split data in the two arrays io_array0, io_array1
// after a call, arrays have the same size, and points with the same index are homologues
// search mode of io_array0 is set to i_newMode
void unfoldMatchingCouples( std::vector<DigeoPoint> &io_array0, std::vector<DigeoPoint> &io_array1, const std::list<V2I> &i_matchedCoupleIndices );

// check if more than i_ratio of a point's neighbours are homologue to its homologue's neighbours
// if not, both the point and its homologue are erased
// this function may change i_array0 and/or i_array1 search mode
void neighbourFilter( std::vector<DigeoPoint> &i_array0, std::vector<DigeoPoint> &i_array1, std::list<V2I> &o_keptCouples, double i_ratio=0.5 );

//
// inline methods
//

// class AnnArray

inline AnnArray::AnnArray(){}

inline AnnArray::AnnArray( std::vector<DigeoPoint> &i_b, SIFT_ANN_SEARCH_MODE i_mode ){ set( i_b, i_mode ); }

inline       ANNpointArray AnnArray::getANNpointArray()       { return (ANNpointArray)m_annArray.data(); }
inline const ANNpointArray AnnArray::getANNpointArray() const { return (const ANNpointArray)m_annArray.data(); }

inline SIFT_ANN_SEARCH_MODE AnnArray::getSearchMode() const { return m_searchMode; }

inline unsigned int AnnArray::size() const { return (unsigned int)m_annArray.size(); }

// class AnnSearcher

inline AnnSearcher::AnnSearcher():m_kdTree(NULL){}

inline AnnSearcher::~AnnSearcher(){ clearTree(); }

inline void AnnSearcher::setErrorBound( double i_errorBound ){ m_errorBound=i_errorBound; }

inline void AnnSearcher::setMaxVisitedPoints( int i_nbMaxVisitedPoints ){ annMaxPtsVisit( i_nbMaxVisitedPoints ); }

inline const ANNdist *AnnSearcher::getNeighboursDistances() const{ return m_neighboursDistances.data(); }

inline const ANNidx *AnnSearcher::getNeighboursIndices() const{ return m_neighboursIndices.data(); }

inline void AnnSearcher::clearTree(){ if ( m_kdTree!=NULL ){ delete m_kdTree; m_kdTree=NULL; } }

#endif // ifndef __SIFT__
