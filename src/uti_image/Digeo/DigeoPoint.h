#ifndef __DIGEO_POINT__
#define __DIGEO_POINT__

#include <string>

#define DIGEO_FILEFORMAT_MAGIC_NUMBER_LSBF 989008845ul
#define DIGEO_FILEFORMAT_MAGIC_NUMBER_MSBF 3440636730ul
#define DIGEO_FILEFORMAT_CURRENT_VERSION 1
#define DIGEO_DESCRIPTOR_SIZE 128

extern const U_INT4 digeo_fileformat_magic_number;
extern const U_INT4 digeo_fileformat_current_version;

class DigeoPoint
{
public:
   typedef enum
   {
      DETECT_UNKNOWN,
      DETECT_LOCAL_MIN,
      DETECT_LOCAL_MAX
   } DetectType;
   
   REAL8 x, y,
         scale,
	 angle,
	 descriptor[DIGEO_DESCRIPTOR_SIZE];
   DetectType type;
   
   static unsigned char sm_uchar_descriptor[DIGEO_DESCRIPTOR_SIZE];
   
   DigeoPoint();
   
   void write_v0( std::ostream &output ) const;
   void read_v0( std::istream &output );
   
   void write_v1( std::ostream &output, bool reverseByteOrder ) const;
   void read_v1( std::istream &output, bool reverseByteOrder );
};

typedef struct
{
   unsigned char byteOrder;
   U_INT4 	 version;
   U_INT4	 nbPoints;
   U_INT4	 descriptorSize;
   
   bool reverseByteOrder;
} DigeoFileHeader;

// read/write a list of Digeo points
bool writeDigeoFile( const string &i_filename, const vector<DigeoPoint> &i_list, U_INT4 i_version=DIGEO_FILEFORMAT_CURRENT_VERSION, bool i_writeBigEndian=MSBF_PROCESSOR() );
// this reading function detects the fileformat and can be used with old siftpp_tgi files
// if o_header is not null, addressed variable is filled
bool readDigeoFile( const string &i_filename, vector<DigeoPoint> &o_list, DigeoFileHeader *o_header=NULL );




//----
// DigeoPoint inline methods
//----


inline DigeoPoint::DigeoPoint():type( DETECT_UNKNOWN ){}

#endif // __DIGEO_POINT__
