#ifndef __SIFT__
#define __SIFT__

typedef double Real_;

#include "RealImage1.h"

#include <list>
#include <iostream>

typedef struct{
    int x, y, o, s;
    bool isMax;
    Real_ rx, ry, rs;
} Extremum;

typedef Extremum RefinedPoint;

// default values for soft-coded parameters
extern const Real_ default_strength_threshold;
extern const Real_ default_onedge_threshold;

// thoses constants should be static members of Siftator but they could not be initialize (waiting for c++11)

// default values for soft-coded parameters
#define default_strength_threshold (0.02/3) // 0.02/S ?
#define default_onedge_threshold   10.

#define m_maxNbAngles        4
#define SIFT_DESCRIPTOR_SIZE     128

#define m_smin                -1
#define m_sigman              .5
// for orientation computation
#define m_nbBins              36  // number of bins for the full circle
#define m_windowFactor        1.5 // factor by which is multiplied point's sigma to obtain gaussian's sigma
// for descriptor computation
#define m_NBO                 8
#define m_NBP                 4
#define m_magnify             3.
#define m_descriptorTreshold 0.2 // truncateDescriptor() clips higher values to this one
    
// store all the data and enable processes leading to sift computation
class Siftator
{
public:
    typedef struct{
        Real_ x, y;
        Real_ scale;
        Real_ angle;
        Real_ descriptor[SIFT_DESCRIPTOR_SIZE];
    } SiftPoint;
private:
    // soft-coded parameters
    Real_ m_strengthThreshold;  // used by getExtrema, refinePoint and refinePoints
    Real_ m_onEdgeThreshold;    // used by refinePoint and refinePoints
    Real_ m_scoreMax;           // a score used for point rejection, used by refinePoint (computed from edge threshold)

    // values fixed when the scale-space format is known (number of octaves and levels, index of the first octave)
    int  m_nbOctaves,
         m_nbLevels,       // number of levels to cover a full octave
         m_nbStoredLevels, // stored gaussian levels, meaning two more than the number of levels per octave
         m_smax,
         m_nbDoG;
    int  m_max_neighbour_distance; // this is the distance to the farest neighbour needed for computation (distance along one dimension)
    Real_ m_sigma0,
         m_sigmak,
         m_dsigma0;
    int c0, c1, c2, c6, c8;

    // storage data
    Real_ m_histo[m_nbBins];

    std::vector<std::vector<RealImage1> > m_octaves; // octaves data
    int m_firstOctave; // index of the first octave (0 being the index of the original image)

    // pre-computed values when an octave index is set
    int m_iTrueOctave, m_iOctave;
    std::vector<RealImage1> m_DoG;      // Differences of Gaussians
    std::vector<RealImage1> m_gradients; // gradient splited in two values modulus and angle
    int m_width, m_height;
    Real_ m_samplingPace;
public:
    // getter/setters
    Real_ getOnEdgeThreshold() const;
    Real_ getStrengthThreshold() const;
    void setOnEdgeThreshold( Real_ i_threshold );
    void setStrengthThreshold( Real_ threshold );

    Siftator( int i_nbOctaves, int i_nbLevels, int i_firstOctave );

    void scale_space_format( int i_nbOctaves, int i_nbLevels, int i_firstOctave );

    int max_neighbour_distance() const;

    // fills the pyramid, using i_image for octave 0
    void compute_gaussians( const RealImage1 &i_image );

    // specify octave to process
    void setCurrentOctave( int i_iOctave );

    // methods depending of the current octave
        // compute the difference of gaussians for all scales of current octave
        void compute_differences_of_gaussians();
        // compute the gradients for all scales of current octave
        void compute_gradients();

        // return a list of extrema in differences of gaussians for octave i_octave
        //void getExtrema( Real_ i_threshold, Real_ i_edgeThreshold, std::list<Extremum> &o_extrema, std::list<RefinedPoint> &o_refinedPoints );
        void getExtrema( std::list<Extremum> &o_extrema );
        void refinePoints( const std::list<Extremum> &o_extrema, std::list<RefinedPoint> &o_refinedPoints );

        // return true if i_p passes the edge test
        // o_p is set to refined coordinates whether or not it passes edge test
        bool refinePoint( const Extremum &i_p, RefinedPoint &o_p );

        int orientations( RefinedPoint &i_p, Real_ o_angles[m_maxNbAngles] );

        // o_descritpor must be of size SIFT_DESCRIPTOR_SIZE
        void descriptor( RefinedPoint &i_p, Real_ i_angle, Real_ *o_descriptor );

    // static methods
    // o_descritpor must be of size SIFT_DESCRIPTOR_SIZE]
    static void normalizeDescriptor( Real_ *o_descriptor );

    // o_descritpor must be of size SIFT_DESCRIPTOR_SIZE]
    static void truncateDescriptor( Real_ *o_descriptor );

    void print_parameters( std::ostream &o ) const;
    // i_basename is image name withour extension
    // save in PGM fileformat
    void save_gaussians( const std::string &i_basename, bool i_verbose=false ) const;

	// raw binary read/write of a SiftPoint (not endian-wise)
    static void write_SiftPoint_binary( std::ostream &output, const SiftPoint &p );
    static void read_SiftPoint_binary( std::istream &output, Siftator::SiftPoint &p );
    
	// same format as siftpp_tgi
	// Real_ values are cast to float
	// descriptor is cast to unsigned char values : d[i]->(unsigned char)(512*d[i])
	static void write_SiftPoint_binary_legacy( std::ostream &output, const Siftator::SiftPoint &p );
    static void read_SiftPoint_binary_legacy( std::istream &output, Siftator::SiftPoint &p );
};

typedef Siftator::SiftPoint SiftPoint;

// Sift-related functions

// read/write in sitfpp_tgi format
bool write_siftPoint_list( const std::string &i_filename, const std::list<Siftator::SiftPoint> &i_list );
bool read_siftPoint_list( const std::string &i_filename, std::vector<Siftator::SiftPoint> &o_list );

// inline methods

// class Siftator

// getters/setters
inline void Siftator::setStrengthThreshold( Real_ i_threshold ){ m_strengthThreshold=i_threshold; }
inline void Siftator::setOnEdgeThreshold( Real_ i_threshold ){
    m_onEdgeThreshold = i_threshold;
    m_scoreMax        = ( ( ( i_threshold+1 )*( i_threshold+1 ) )/i_threshold );
}
inline Real_ Siftator::getOnEdgeThreshold() const { return m_onEdgeThreshold; }
inline Real_ Siftator::getStrengthThreshold() const { return m_strengthThreshold; }

inline void Siftator::setCurrentOctave( int i_iOctave )
{
    m_iOctave      = i_iOctave;
    m_iTrueOctave  = i_iOctave+m_firstOctave;
    m_width        = m_octaves[m_iOctave][0].width();
    m_height       = m_octaves[m_iOctave][0].height();
    m_samplingPace = Real_( ( m_iTrueOctave<0 ) ? 1.0f/( 1<<( -m_iTrueOctave ) ) : 1<<m_iTrueOctave );

    // neighbours' offsets
    //c3 = -1;
    //c5 = 1;
    //c7 = m_width;
    c6 = m_width-1;
    c8 = m_width+1;
    c1 = -m_width;
    c0 = c1-1;
    c2 = c1+1;
}

inline int Siftator::max_neighbour_distance() const { return m_max_neighbour_distance; }

// raw binary output of a SiftPoint (not endian-wise)
inline void Siftator::write_SiftPoint_binary( std::ostream &output, const Siftator::SiftPoint &p ){
    output.write( (char*)&p.x, sizeof( Siftator::SiftPoint ) );
}

inline void Siftator::read_SiftPoint_binary( std::istream &output, Siftator::SiftPoint &p ){
    output.read( (char*)&p.x, sizeof( Siftator::SiftPoint ) );
}

// same format as siftpp_tgi (not endian-wise)
// Real_ values are cast to float
// descriptor is cast to unsigned char values d[i]->(unsigned char)(512*d[i])
inline void Siftator::write_SiftPoint_binary_legacy( std::ostream &output, const Siftator::SiftPoint &p )
{
	float float_value = (float)p.x; output.write( (char*)&float_value, sizeof( float ) );
	float_value = (float)p.y; output.write( (char*)&float_value, sizeof( float ) );
	float_value = (float)p.scale; output.write( (char*)&float_value, sizeof( float ) );
	float_value = (float)p.angle; output.write( (char*)&float_value, sizeof( float ) );
	static unsigned char uchar_desc[SIFT_DESCRIPTOR_SIZE];
	int i=SIFT_DESCRIPTOR_SIZE; const Real_ *itReal=p.descriptor; unsigned char *it_uchar=uchar_desc;
	while (i--)
		(*it_uchar++)=(unsigned char)( 512*(*itReal++) );
    output.write( (char*)uchar_desc, SIFT_DESCRIPTOR_SIZE );
}

// same format as siftpp_tgi (not endian-wise)
// Real_ values are cast to float
// descriptor is cast to unsigned char values d[i]->(unsigned char)(512*d[i])
inline void Siftator::read_SiftPoint_binary_legacy( std::istream &output, Siftator::SiftPoint &p )
{
	float float_values[4];
	output.read( (char*)float_values, 4*sizeof( float ) );
	p.x 	= (Real_)float_values[0];
	p.y 	= (Real_)float_values[1];
	p.scale = (Real_)float_values[2];
	p.angle = (Real_)float_values[3];
	static unsigned char uchar_desc[SIFT_DESCRIPTOR_SIZE];
    output.read( (char*)uchar_desc, SIFT_DESCRIPTOR_SIZE );
	int i=SIFT_DESCRIPTOR_SIZE; Real_ *itReal=p.descriptor; unsigned char *it_uchar=uchar_desc;
	while (i--) (*itReal++)=( (Real_)(*it_uchar++)/512 );
}

#endif // __SIFT__
