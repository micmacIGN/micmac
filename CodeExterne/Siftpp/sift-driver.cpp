// file:        sift-driver.cpp
// author:      Andrea Vedaldi
// description: SIFT command line utility implementation

// AUTORIGTHS

#include<sift.hpp>

#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<algorithm>

extern "C" {
#include<getopt.h>
#if defined (VL_MAC)
#include<libgen.h>
#else
#include<string.h>
#endif
#include<assert.h>
}
#include<memory>

using namespace std ;

size_t const not_found = numeric_limits<size_t>::max() - 1 ;

/** @brief Case insensitive character comparison
 **
 ** This predicate returns @c true if @a a and @a b are equal up to
 ** case.
 **
 ** @return predicate value.
 **/
inline
bool ciIsEqual(char a, char b)
{
  return 
    tolower((char unsigned)a) == 
    tolower((char unsigned)b) ;
}

/** @brief Case insensitive extension removal
 **
 ** The function returns @a name with the suffix $a ext removed.  The
 ** suffix is matched case-insensitve.
 **
 ** @return @a name without @a ext.
 **/
string
removeExtension(string name, string ext)
{
  string::iterator pos = 
    find_end(name.begin(),name.end(),ext.begin(),ext.end(),ciIsEqual) ;

  // make sure the occurence is at the end
  if(pos+ext.size() == name.end()) {
    return name.substr(0, pos-name.begin()) ;
  } else {
    return name ;
  }
}


/** @brief Insert descriptor into stream
 **
 ** The function writes a descriptor in ASCII/binary format
 ** and in integer/floating point format into the stream.
 **
 ** @param os output stream.
 ** @param descr_pt descriptor (floating point)
 ** @param binary write binary descriptor?
 ** @param fp write floating point data?
 **/
std::ostream&
insertDescriptor(std::ostream& os,
                 VL::float_t const * descr_pt,
                 bool binary,
                 bool fp )
{
#define RAW_CONST_PT(x) reinterpret_cast<char const*>(x)
#define RAW_PT(x)       reinterpret_cast<char*>(x)

  if( fp ) {

    /* convert to 32 bits floats (single precision) */
    VL::float32_t fdescr_pt [128] ;
    for(int i = 0 ; i < 128 ; ++i)
      fdescr_pt[i] = VL::float32_t( descr_pt[i]) ;

    if( binary ) {
      /* 
         Test for endianess. Recall: big_endian = the most significant
         byte at lower memory address.
      */
      short int const word = 0x0001 ;
      bool little_endian = RAW_CONST_PT(&word)[0] ;
      
      /* 
         We save in big-endian (network) order. So if this machine is
         little endiand do the appropriate conversion.
      */
      if( little_endian ) {
        for(int i = 0 ; i < 128 ; ++i) {
          VL::float32_t tmp = fdescr_pt[ i ] ;        
          char* pt  = RAW_PT(fdescr_pt + i) ;
          char* spt = RAW_PT(&tmp) ;
          pt[0] = spt[3] ;
          pt[1] = spt[2] ;
          pt[2] = spt[1] ;
          pt[3] = spt[0] ;
        }
      }            
      os.write( RAW_PT(fdescr_pt), 128 * sizeof(VL::float32_t) ) ;

    } else {

      for(int i = 0 ; i < 128 ; ++i) 
        os << ' ' 
           << fdescr_pt[i] ;
    }

  } else {

    VL::uint8_t idescr_pt [128] ;

    for(int i = 0 ; i < 128 ; ++i)
      idescr_pt[i] = uint8_t(float_t(512) * descr_pt[i]) ;
    
    if( binary ) {

      os.write( RAW_PT(idescr_pt), 128) ;	

    } else { 
      
      for(int i = 0 ; i < 128 ; ++i) 
        os << ' ' 
           << uint32_t( idescr_pt[i] ) ;
    }
  }
  return os ;
}

/* keypoint list */
typedef vector<pair<VL::Sift::Keypoint,VL::float_t> > Keypoints ;

/* predicate used to order keypoints by increasing scale */
bool cmpKeypoints (Keypoints::value_type const&a,
		   Keypoints::value_type const&b) {
  return a.first.sigma < b.first.sigma ;
}

// -------------------------------------------------------------------
//                                                                main
// -------------------------------------------------------------------
int
main(int argc, char** argv)
{
  int    first          = -1 ;
  int    octaves        = -1 ;
  int    levels         = 3 ;
  float  threshold      = 0.04f / levels / 2.0f ;
  float  edgeThreshold  = 10.0f;
  float  magnif         = 3.0 ;
  int    nodescr        = 0 ;
  int    noorient       = 0 ;
  int    stableorder    = 0 ;
  int    savegss        = 0 ;
  int    verbose        = 0 ;
  int    binary         = 0 ;
  int    haveKeypoints  = 0 ;
  int    unnormalized   = 0 ;
  int    fp             = 0 ;
  string outputFilenamePrefix ;
  string outputFilename ;
  string descriptorsFilename ;
  string keypointsFilename ;

  static struct option longopts[] = {
    { "verbose",         no_argument,            NULL,              'v' },
    { "help",            no_argument,            NULL,              'h' },
    { "output",          required_argument,      NULL,              'o' },
    { "prefix",          required_argument,      NULL,              'p' },
    { "first-octave",    required_argument,      NULL,              'f' },
    { "keypoints",       required_argument,      NULL,              'k' },
    { "octaves",         required_argument,      NULL,              'O' },
    { "levels",          required_argument,      NULL,              'S' },
    { "threshold",       required_argument,      NULL,              't' },
    { "edge-threshold",  required_argument,      NULL,              'e' },
    { "magnif",          required_argument,      NULL,              'm' },
    { "binary",          no_argument,            NULL,              'b' }, 
    { "no-descriptors",  no_argument,            &nodescr,          1   },
    { "no-orientations", no_argument,            &noorient,         1   },
    { "stable-order",    no_argument,            &stableorder,      1   },
    { "save-gss",        no_argument,            &savegss,          1   },
    { "unnormalized",    no_argument,            &unnormalized,     1   },
    { "floating-point",  no_argument,            &fp,               1   },
    { NULL,              0,                      NULL,              0   }
  };
  
  int ch ;

  try {
    
    while ( (ch = getopt_long(argc, argv, "vho:p:f:k:O:S:t:e:b", longopts, NULL)) != -1) {
      switch (ch) {

      case '?' :
        VL_THROW("Invalid option '"<<argv[optind-1]<<"'.") ;
        break;
        
      case ':' :
        VL_THROW("Missing argument of option '"<<argv[optind-1]<<"'.") ;
        break;
        
      case 'h' :
        std::cout
          << argv[0] << " [--verbose|=v] [--help|-h]" << endl
	  << "     [--output|-o NAME] [--prefix|-p PREFIX] [--binary|-b] [--save-gss] " << endl
          << "     [--no-descriptors] [--no-orientations] " << endl
          << "     [--levels|-S NUMBER] [--octaves|-O NUMBER] [--first-octave|-f NUMBER] " << endl
          << "     [--threshold|-t NUMBER] [--edge-threshold|-e NUMBER] " << endl
          << "     [--floating-point] [--unnormalized] " << endl
          << "     IMAGE [IMAGE2 ...]" << endl
          << endl
	  << "* Options *" << endl
          << " --verbose             Be verbose"<< endl
          << " --help                Print this message"<<endl
          << " --output=NAME         Write to this file"<<endl
	  << " --prefix=PREFIX       Derive output filename prefixing this string to the input file"<<endl
          << " --binary              Write descriptors to a separate file in binary format"<<endl
	  << " --keypoints=FILE      Reads keypoint frames from here; do not run SIFT detector" << endl
          << " --save-gss            Save Gaussian scale space on disk" << endl
          << " --octaves=O           Number of octaves" << endl
          << " --levels=S            Number of levels per octave" << endl
          << " --first-octave=MINO   Index of the first octave" << endl
          << " --threshold=THR       Keypoint strength threhsold" << endl
          << " --magnif=MAG          Keypoint magnification" << endl
          << " --edge-threshold=THR  On-edge threshold" << endl 
          << " --no-descriptors      Do not compute descriptors" << endl
          << " --no-orientations     Do not compute orientations" << endl
	  << " --stable-order        Do not reorder keypoints" << endl
          << " --unnormalzied        Do not normalize descriptors" << endl
          << " --floating-point      Save floating point descriptors" << endl
	  << endl
	  << " * Examples *" << endl
	  << argv[0] << " [OPTS...] image.pgm" << endl
	  << argv[0] << " [OPTS...] image.pgm --output=file.key" << endl
	  << argv[0] << " [OPTS...] image.pgm --keypoints=frames.key" << endl
	  << argv[0] << " [OPTS...] *.pgm --prefix=/tmp/" << endl
	  << argv[0] << " [OPTS...] *.pgm --prefix=/tmp/ --binary" << endl
	  << endl
	  << " * This build: " ;
#if defined VL_USEFASTMATH
	std::cout << "has fast approximate math" ;
#else
	std::cout << "has slow accurate math" ;
#endif
	std::cout << " (fp datatype is '"
		  << VL_EXPAND_AND_STRINGIFY(VL_FASTFLOAT)
		  << "') *"<<endl ;
        return 0 ;
	
      case 'v' : // verbose
        verbose = 1 ;
        break ;
        
      case 'f': // first octave
        {
          std::istringstream iss(optarg) ;
          iss >> first ;
          if( iss.fail() )
            VL_THROW("Invalid argument '" << optarg << "'.") ;
        }
        break ;
        
      case 'O' : // octaves
        {
          std::istringstream iss(optarg) ;
          iss >> octaves ;
          if( iss.fail() )
            VL_THROW("Invalid argument '" << optarg << "'.") ;
          if( octaves < 1 ) {
            VL_THROW("The number of octaves cannot be smaller than one."); 
          }
        }
        break ;
        
      case 'S' : // levels
        {
          std::istringstream iss(optarg) ;
          iss >> levels ;
          if( iss.fail() )
            VL_THROW("Invalid argument '" << optarg << "'.") ;
          if( levels < 1 ) {
            VL_THROW("The number of levels cannot be smaller than one.") ;
          }
        }      
        break ;

      case 't' : // threshold
        {
          std::istringstream iss(optarg) ;
          iss >> threshold ;
          if( iss.fail() )
            VL_THROW("Invalid argument '" << optarg << "'.") ;
        }
        break ;

      case 'e' : // edge-threshold
        {
          std::istringstream iss(optarg) ;
          iss >> edgeThreshold ;
          if( iss.fail() )
            VL_THROW("Invalid argument '" << optarg << "'.") ;
        }
        break ;

      case 'm' : // magnification
        {
          std::istringstream iss(optarg) ;
          iss >> magnif ;
          if( iss.fail() )
            VL_THROW("Invalid argument '" << optarg << "'.") ;
        }
        break ;


      case 'o' : // output filename
        {
          outputFilename = std::string(optarg) ;
          break ;
        }

      case 'p' : // output prefix
        {
          outputFilenamePrefix = std::string(optarg) ;
          break ;
        }

      case 'k' : // keypoint file
	{
	  keypointsFilename = std::string(optarg) ;
	  haveKeypoints = 1 ;
	  break ;
	}

      case 'b' : // write descriptors to a binary file
        {
          binary = 1 ;
          break ;
        }

      case 0 : // all other options
        break ;
        
      default:
        assert(false) ;
      }
    }
    
    argc -= optind;
    argv += optind;

    // check for argument consistency
    if(argc == 0) VL_THROW("No input image specfied.") ;
    if(outputFilename.size() != 0 && (argc > 1 | binary)) {
      VL_THROW("--output cannot be used with multiple images or --binary.") ;
    }

    if(outputFilename.size() !=0 && 
       outputFilenamePrefix.size() !=0) {
      VL_THROW("--output cannot be used in combination with --prefix.") ;
    }

    /* end option try-catch block */
  }  
  catch( VL::Exception const & e ) {
    cerr << "siftpp: error: "
         << e.msg 
         << endl ;
    exit(1) ;
  } 

  // -----------------------------------------------------------------
  //                                            Loop over input images
  // -----------------------------------------------------------------      
  while( argc > 0 ) {

    string name(argv[0]) ;

    try {
      VL::PgmBuffer buffer ;
      
      // compute the output filenames:
      //
      // 1) if --output is specified, then we just use the one provided
      //    by the user
      //
      // 2) if --output is not specified, we derive the output filename
      //    from the input filename by
      //    - removing the extension part from the output filename
      //    - and if outputFilenamePrefix is non void, removing 
      //      the directory part and prefixing outputFilenamePrefix.
      //
      // 3) in any case we derive the binary descriptor filename by
      //    removing from the output filename the .key extension (if any)
      //    and adding a .desc extension.
      
      if(outputFilename.size() == 0) {
	// case 2) above
	outputFilename = name ;
	
	// if we specify an output directory, then extract
	// the basename
	if(outputFilenamePrefix.size() != 0) {
          char * tmp = new char [outputFilename.length()+1] ;
          strcpy(tmp, outputFilename.c_str()) ;
	  outputFilename = outputFilenamePrefix + 
            std::string(basename(tmp)) ;
          delete [] tmp ;
	}
	
      // remove .pgm extension, add .key
	outputFilename = removeExtension(outputFilename, ".pgm") ;
	outputFilename += ".key" ;
      }
      
      // remove .key extension, add .desc
      descriptorsFilename = removeExtension(outputFilename, ".key") ;
      descriptorsFilename += ".desc" ;
      
      // ---------------------------------------------------------------
      //                                                  Load PGM image
      // ---------------------------------------------------------------    
      verbose && cout
        << "siftpp: lodaing PGM image '" << name << "' ..."
        << flush;
      
      try {          
	ifstream in(name.c_str(), ios::binary) ; 
	if(! in.good()) VL_THROW("Could not open '"<<name<<"'.") ;      
	extractPgm(in, buffer) ;
      }    
      catch(VL::Exception const& e) {
	throw VL::Exception("PGM read error: "+e.msg) ;
      }
      
      verbose && cout 
        << " read "
        << buffer.width  <<" x "
        << buffer.height <<" pixels" 
        << endl ;
      
      // ---------------------------------------------------------------
      //                                            Gaussian scale space
      // ---------------------------------------------------------------    
      verbose && cout 
        << "siftpp: computing Gaussian scale space" 
        << endl ;
      
      int         O      = octaves ;    
      int const   S      = levels ;
      int const   omin   = first ;
      float const sigman = .5 ;
      float const sigma0 = 1.6 * powf(2.0f, 1.0f / S) ;
      
      // optionally autoselect the number number of octaves
      // we downsample up to 8x8 patches
      if(O < 1) {
	O = std::max
	  (int
	   (std::floor
	    (log2
	     (std::min(buffer.width,buffer.height))) - omin -3), 1) ;
      }

      verbose && cout
        << "siftpp:   number of octaves     : " << O << endl 
        << "siftpp:   first octave          : " << omin << endl 
        << "siftpp:   levels per octave     : " << S 
        << endl ;
      
      // initialize scalespace
      VL::Sift sift(buffer.data, buffer.width, buffer.height, 
		    sigman, sigma0,
		    O, S,
		    omin, -1, S+1) ;
      
      verbose && cout 
        << "siftpp: Gaussian scale space completed"
        << endl ;
      
      // ---------------------------------------------------------------
      //                                       Save Gaussian scale space
      // ---------------------------------------------------------------    
      
      if(savegss) {
	verbose && cout<<"siftpp: saving Gaussian scale space"<<endl ;
	
	string imageBasename = removeExtension(outputFilename, ".key") ;
	
	for(int o = omin ; o < omin + O ; ++o) {
	  for(int s = 0 ; s < S ; ++s) {
	    
	    ostringstream suffix ;
	    suffix<<'.'<<o<<'.'<<s<<".pgm" ;
	    string imageFilename = imageBasename + suffix.str() ;
	    
	    verbose && cout 
              << "siftpp:   octave " << setw(3) << o
              << " level " << setw(3) << s
              << " to '" << imageFilename
              << "' ..." << flush ;
	    
	    ofstream fout(imageFilename.c_str(), ios::binary) ;
	    if(!fout.good()) 
	      VL_THROW("Could not open '"<<imageFilename<<'\'') ;
	    
	    VL::insertPgm(fout,
			  sift.getLevel(o,s),
			  sift.getOctaveWidth(o),
			  sift.getOctaveHeight(o)) ;
	    fout.close() ;
	    
	    verbose && cout
              << " done." << endl ;
	  }
	}
      }
      
      // -------------------------------------------------------------
      //                                             Run SIFT detector
      // -------------------------------------------------------------    
      if( ! haveKeypoints ) {

	verbose && cout 
          << "siftpp: running detector  "<< endl
          << "siftpp:   threshold             : " << threshold << endl
          << "siftpp:   edge-threshold        : " << edgeThreshold
          << endl ;
	
	sift.detectKeypoints(threshold, edgeThreshold) ;
	
	verbose && cout 
          << "siftpp: detector completed with " 
          << sift.keypointsEnd() - sift.keypointsBegin() 
          << " keypoints" 
          << endl ;
      }
      
      // -------------------------------------------------------------
      //                  Run SIFT orientation detector and descriptor
      // -------------------------------------------------------------    

      /* set descriptor options */
      sift.setNormalizeDescriptor( ! unnormalized ) ;
      sift.setMagnification( magnif ) ;

      if( verbose ) {
        cout << "siftpp: " ;
	if( ! noorient &   nodescr) cout << "computing keypoint orientations" ;
	if(   noorient & ! nodescr) cout << "computing keypoint descriptors" ;
	if( ! noorient & ! nodescr) cout << "computing orientations and descriptors" ;
	if(   noorient &   nodescr) cout << "finalizing" ; 
	cout << endl ;
      }
      
      {            
        // open output file
        ofstream out(outputFilename.c_str(), ios::binary) ;
        
        if( ! out.good() ) 
          VL_THROW("Could not open output file '"
                   << outputFilename
                   << "'.") ;
        
        verbose && cout
          << "siftpp:   write keypoints to    : '" << outputFilename << "'"         << endl
          << "siftpp:   floating point descr. : "  << (fp           ? "yes" : "no") << endl
          << "siftpp:   binary descr.         : "  << (binary       ? "yes" : "no") << endl
          << "siftpp:   unnormalized descr.   : "  << (unnormalized ? "yes" : "no") << endl
          << "siftpp:   descr. magnif.        : "  << setprecision(3) << magnif
          << endl ;
        
        out.flags(ios::fixed) ;
      
        /* If a keypoint file is provided, then open it now */
        auto_ptr<ifstream> keypointsIn_pt ;
        
        if( haveKeypoints ) {
          keypointsIn_pt = auto_ptr<ifstream>
            (new ifstream(keypointsFilename.c_str(), ios::binary)) ;
          
          if( ! keypointsIn_pt->good() ) 
            VL_THROW("Could not open keypoints file '"
                     << keypointsFilename
                     << "'.") ;
          
          verbose && cout
            << "siftpp:   read keypoints from   : '" 
            << keypointsFilename << "'"
            << endl ;
        }
        
        /* If the descriptors are redirected to a binary file, then open it now */
        auto_ptr<ofstream> descriptorsOut_pt ;
        
        if( binary ) {        
          descriptorsOut_pt = auto_ptr<ofstream>
            (new ofstream(descriptorsFilename.c_str(), ios::binary)) ;
          
          if( ! descriptorsOut_pt->good() )
            VL_THROW("Could not open descriptors file '"
                     << descriptorsFilename 
                     << "'.") ;
          
          verbose && cout 
            << "siftpp:   write descriptors to  : '" 
            << descriptorsFilename << "'"
            << endl ;         
        }
        
        if( haveKeypoints ) {
          // -------------------------------------------------------------
          //                 Reads keypoint from file, compute descriptors
          // -------------------------------------------------------------
          Keypoints keypoints ;
          
          while( !keypointsIn_pt->eof() ) {
            VL::float_t x,y,sigma,th ;
            
            /* read x, y, sigma and th from the beginning of the line */
            (*keypointsIn_pt) 
              >> x
              >> y
              >> sigma
              >> th ;

            /* skip the rest of the line */
            (*keypointsIn_pt).ignore(numeric_limits<streamsize>::max(),'\n') ;

            /* break the loop if end of file reached */
            if( keypointsIn_pt->eof() ) break ;

            /* trhow an error if something wrong */
            if( ! keypointsIn_pt->good() ) 
              VL_THROW("Error reading keypoints file.") ;
            
            /* compute integer components */
            VL::Sift::Keypoint key 
              = sift.getKeypoint(x,y,sigma) ;
            
            Keypoints::value_type entry ;
            entry.first  = key ;
            entry.second = th ;
            keypoints.push_back(entry) ;
          }
          
          /* sort keypoints by scale if not required otherwise */
          if(! stableorder)
            sort(keypoints.begin(), keypoints.end(), cmpKeypoints) ;
          
          // process in batch
          for(Keypoints::const_iterator iter = keypoints.begin() ;
              iter != keypoints.end() ;
              ++iter) {
	    VL::Sift::Keypoint const& key = iter->first ;
	    VL::float_t th = iter->second ;
            
	    /* write keypoint */
	    out << setprecision(2) << key.x     << " "
		<< setprecision(2) << key.y     << " "
		<< setprecision(2) << key.sigma << " "
		<< setprecision(3) << th ;
	    
	    /* compute descriptor */
	    VL::float_t descr [128] ;
	    sift.computeKeypointDescriptor(descr, key, th) ;
            
            /* save to appropriate file */
            if( descriptorsOut_pt.get() ) {
              ostream& os = *descriptorsOut_pt.get() ;
              insertDescriptor(os, descr, true, fp) ;
            } else {
              insertDescriptor(out, descr, false, fp) ;
            }
            
            /* next keypoint */
	    out << endl ;    
	  } // next keypoint
          
	} else {
          
	  // -------------------------------------------------------------
	  //            Run detector, compute orientations and descriptors
	  // -------------------------------------------------------------
	  for( VL::Sift::KeypointsConstIter iter = sift.keypointsBegin() ;
	       iter != sift.keypointsEnd() ; ++iter ) {
	    
	    // detect orientations
	    VL::float_t angles [4] ;
	    int nangles ;
	    if( ! noorient ) {
	      nangles = sift.computeKeypointOrientations(angles, *iter) ;
	    } else {
	    nangles = 1;
	    angles[0] = VL::float_t(0) ;
	  }
	    
	    // compute descriptors
	    for(int a = 0 ; a < nangles ; ++a) {

	      out << setprecision(2) << iter->x << ' '
		  << setprecision(2) << iter->y << ' '
		  << setprecision(2) << iter->sigma << ' ' 
		  << setprecision(3) << angles[a] ;

              /* compute descriptor */
              VL::float_t descr_pt [128] ;
              sift.computeKeypointDescriptor(descr_pt, *iter, angles[a]) ;
	
              /* save descriptor to to appropriate file */	      
	      if( ! nodescr ) {
                if( descriptorsOut_pt.get() ) {
                  ostream& os = *descriptorsOut_pt.get() ;
                  insertDescriptor(os, descr_pt, true, fp) ;
                } else {
                  insertDescriptor(out, descr_pt, false, fp) ;
                }
              }
              /* next line */
	      out << endl ;
	    } // next angle
	  } // next keypoint
	}
	
	out.close() ;
	if(descriptorsOut_pt.get()) descriptorsOut_pt->close(); 
	if(keypointsIn_pt.get())    keypointsIn_pt->close(); 
	verbose && cout 
          << "siftpp: job completed"<<endl ;
      }
      
      argc-- ;
      argv++ ;
      outputFilename = string("") ;
    }
    catch(VL::Exception &e) {
      cerr<<endl<<"Error processing '"<<name<<"': "<<e.msg<<endl ;
      return 1 ;
    }    
  } // next image
  
  return 0 ;
}
