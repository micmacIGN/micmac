#ifndef __ANN_UTILS__
#define __ANN_UTILS__

#include <string>

std::string clipShortestExtension( const std::string &i_filename );
std::string getBasename( const std::string &i_filename );

// create an output filename from two input name
std::string ann_create_output_filename( const std::string &i_inFilename0, const std::string &i_inFilename1 );

#endif
