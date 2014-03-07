#include "StdAfx.h"

#include "TracePack.h"

using namespace std;

typedef struct
{
   string name;
   int (*func)( int, char ** );
} command_t;

int snapshot_func( int, char ** );
int list_func( int, char ** );
int getfile_func( int, char ** );
int setstate_func( int, char ** );
int pack_from_script_func( int, char ** );
int replay_func( int, char ** );
int test_func( int, char ** );
int add_ignored_func( int, char ** );
int help_func( int, char ** );

command_t g_commands[] = {
   { "snapshot", snapshot_func },
   { "list", list_func },
   { "getfile", getfile_func },
   { "setstate", setstate_func },
   { "pack_from_script", pack_from_script_func },
   { "replay", replay_func },
   { "test", test_func },
   { "add_ignored", add_ignored_func },
   { "help", help_func },
   { "", NULL } // signify the end of the list
};

int help_func( int argc, char **argv )
{
   cout << "Commands" << endl;
   command_t *itCommand = g_commands;
   while ( itCommand->func!=NULL )
   {
      cout << "\t" << itCommand->name << endl;
      itCommand++;
   }
   return EXIT_SUCCESS;
}

int snapshot_func( int argc, char **argv )
{
	if ( argc!=2 ){
		cout << "not enough args" << endl;
		return EXIT_FAILURE;
	}
	const ctPath anchor( (string)(argv[1]) );
	const cElFilename packname( (string)(argv[0]) );
	TracePack pack( packname, anchor );

	if ( packname.exists() ){
		if ( !pack.load() ){
			cerr << "[" << packname.str_unix() << "] exists but cannot be read as a TracePack file" << endl;
			return EXIT_FAILURE;
		}
		else
			cout << "[" << packname.str_unix() << "] loaded, the pack will be updated" << endl;
	}
	else
		cout << "[" << packname.str_unix() << "] does not exist, a new pack has been created at date " << pack.date() << endl;
   
	cout << "snaping [" << anchor.str() << "] to [" << packname.str_unix() << "]" << endl;
	pack.addState();

	const unsigned int nbStates = pack.nbStates();
	cerr << nbStates << " state" << (nbStates>1?'s':'\0') << endl;

	if ( !pack.save() ){
		cerr << RED_ERROR << "unable to save to [" << packname.str_unix() << "]" << endl;
		return EXIT_FAILURE;
	}

	#ifdef __DEBUG_TRACE_PACK
		TracePack pack2( packname, anchor );  
		pack2.load();
		if ( !pack.trace_compare( pack2 ) ){
			cerr << RED_DEBUG_ERROR << "pack [" << packname.str_unix() << "] is not equal to its write+read copy" << endl;
			exit(EXIT_FAILURE);
		}
	#endif

	return EXIT_SUCCESS;
}

bool get_list_registry_number( const string &i_argument, unsigned int &o_iRegistry, bool &o_listState )
{
   if ( i_argument.length()<2 ) return false;
   if ( i_argument[0]=='s' ) o_listState=true;
   else if ( i_argument[0]=='r' ) o_listState=false;
   else return false;
   int i = atoi( i_argument.substr(1).c_str() );
   if ( i<0 ) return false;
   o_iRegistry = (unsigned int)i;
   return true;
}

int list_func( int argc, char **argv )
{
	if ( argc==0 ){ cerr << "not enough args" << endl; return EXIT_FAILURE; }
	const cElFilename filename( (string)(argv[0]) );

	TracePack pack( filename, ctPath() );
	if ( !pack.load() ){
		cerr << RED_ERROR << "unable to load [" << filename.str_unix() << "]" << endl;
		return EXIT_FAILURE;
	}
   
	if ( argc>0 ){
		const unsigned int nbStates = pack.nbStates();
		cout << "pack [" << filename.str_unix() << ']' << endl;
		cout << "\t- creation date " << pack.date() << " (UTC)"<< endl;
		cout << "\t- " << nbStates << (nbStates>1?" registries":" registry") << endl;
		cout << "commands" << endl;
		vector<cElCommand> commands;
		pack.getAllCommands( commands );
		for ( unsigned int iCmd=0; iCmd<nbStates; iCmd++ )
			cout << '\t' << iCmd << " [" << commands[iCmd].str() << "]" << endl;
	}
	if ( argc==2 ){
		bool listState;
		unsigned int iRegistry;
		if ( !get_list_registry_number( argv[1], iRegistry, listState ) ){
			cerr << RED_ERROR << "third argument should be sXXX if you want to list a state or rXXX if you want to list a raw registry" << endl;
			return EXIT_FAILURE;
		}

		if ( iRegistry>=pack.nbStates() ){
			cerr << RED_ERROR << "index " << iRegistry << " out of range (the pack has " << pack.nbStates() << " states)" << endl;
			return EXIT_FAILURE;
		}
		if ( listState ){
			TracePack::Registry reg;
			pack.getState( (unsigned int)iRegistry, reg );
			cout << "state " << iRegistry << endl;
			reg.dump( cout, "\t" );
		}
		else{
			cout << "raw registry " << iRegistry << endl;
			pack.getRegistry(iRegistry).dump( cout, "\t" );
		}
	}

   return EXIT_SUCCESS;
}

int getfile_func( int argc, char **argv )
{
   if ( argc!=4 ){
      cerr << "usage: packname iState file_to_extract output_directory" << endl;
      return EXIT_FAILURE;
   }
      
   const cElFilename itemName( (string)(argv[2]) );
   const ctPath outputDirectory( argv[3] );
   if ( !outputDirectory.exists() )
   {
      cerr << RED_ERROR << "output directory [" << outputDirectory.str() << "] does not exist" << endl;
      return EXIT_FAILURE;
   }
   const cElFilename outputFile( outputDirectory, itemName );
   if ( outputFile.exists() )
   {
      cerr << RED_ERROR << "output file [" << outputFile.str_unix() << "] already not exist" << endl;
      return EXIT_FAILURE;
   }
   
   const cElFilename packname( (string)(argv[0]) );   
   TracePack pack( packname, outputDirectory );
   if ( !pack.load() )
   {
      cerr << RED_ERROR << "unable to load [" << packname.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   
   const unsigned int nbStates = pack.nbStates();
   const int signed_iState = atoi( argv[1] );
   const unsigned int iState = (unsigned int)signed_iState;
   if ( signed_iState<0 || iState>=nbStates )
   {
      cerr << RED_ERROR << "invalid state index : " << signed_iState << " (pack has " << nbStates << " state" << (nbStates>1?"s":"") << endl;
      return EXIT_FAILURE;
   }
   
   if ( !pack.copyItemOnDisk( iState, itemName ) )
   {
      cerr << RED_ERROR << "unable to copy item [" << itemName.str_unix() << "] from [" << packname.str_unix() << "]:s" << iState << " to directory [" << outputDirectory.str() << ']' << endl;
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}

int setstate_func( int argc, char **argv )
{
   //setstate packname istate anchor
   
   if ( argc!=3 )
   { 
      cerr << "usage: packname iState output_directory" << endl;
      return EXIT_FAILURE;
   }
   
   const cElFilename packname( (string)(argv[0]) );
   if ( !packname.exists() )
   {
      cerr << RED_ERROR << "pack [" << packname.str_unix() << "] does not exist" << endl;
      return EXIT_FAILURE;
   }
   
   const int iState_s = atoi( argv[1] );
   if ( iState_s<0 )
   {
      cerr << RED_ERROR << "state index " << iState_s << " is invalid" << endl;
      return EXIT_FAILURE;
   }
   const unsigned int iState = (unsigned int)iState_s;
   
   const ctPath anchor( argv[2] );
   if ( !anchor.exists() && !anchor.create() )
   {
      cerr << RED_ERROR << "cannot create directory [" << anchor.str() << "]" << endl;
      return EXIT_FAILURE;
   }
   
   TracePack pack( packname, anchor );
   if ( !pack.load() )
   {
      cerr << RED_ERROR << "loading a pack from file [" << packname.str_unix() << "] failed" << endl;
      return EXIT_FAILURE;
   }
   const unsigned int nbStates = pack.nbStates();
   if ( iState>=nbStates )
   {
      cerr << "ERROR: state index " << iState << " out of range (" << nbStates << " state" << (nbStates>1?'s':'\0') << " in pack [" << packname.str_unix() << "])" << endl;
      return EXIT_FAILURE;
   }
   pack.setState( iState );
   
   return EXIT_SUCCESS;
}

void generate_dictionary( int &io_argc, char **i_argv, map<string,string> &o_dictionary )
{
   const string prefix = "${",
                suffix = "}";
   for ( int i=io_argc-1; i>=0; i-- )
   {
      string arg=i_argv[i],
             var, val;
      size_t pos = arg.find( '=' );
      if ( pos==string::npos || pos==0 || pos==arg.length()-1 ) return;
      
      var = arg.substr( 0, pos );
      val = arg.substr( pos+1 );
      o_dictionary[prefix+var+suffix] = val;
      io_argc--;
   }
}

int pack_from_script_func( int argc, char **argv )
{   
   map<string,string> dictionary;
   generate_dictionary( argc, argv, dictionary );
   
   if ( argc<2 )
   {
      cerr << "usage: packfile scriptfile source_directory [VAR1=value1 VAR2=value2 ...]" << endl;
      return EXIT_FAILURE;
   }

   const cElFilename packname( (string)(argv[0]) );
   if ( packname.exists() )
   {
      cerr << RED_ERROR << "pack [" << packname.str_unix() << "] already exist" << endl;
      return EXIT_FAILURE;
   }

   const cElFilename scriptname( (string)(argv[1]) );
   if ( !scriptname.exists() )
   {
      cerr << RED_ERROR << "script [" << scriptname.str_unix() << "] does not exist" << endl;
      return EXIT_FAILURE;
   }

   ctPath anchor;
   anchor = ctPath( (string)(argv[2]) );

   if ( anchor.isAncestorOf( packname ) )
   {
      cerr << RED_ERROR << "pack file [" << packname.str_unix() << "] is inside source directory [" << anchor.str() << "]" << endl;
      return EXIT_FAILURE;
   }

   if ( !anchor.exists() )
   {
      cerr << RED_ERROR << "reference directory [" << anchor.str() << "] does not exist" << endl;
      return EXIT_FAILURE;
   }
   dictionary["${SITE_PATH}"] = anchor.str();
   cout << "--- using directory [" << anchor.str() << "]" << endl;

   cout << "--- dictionary" << endl;
   map<string,string>::iterator itDico = dictionary.begin();
   while ( itDico!=dictionary.end() )
   {
      cout << itDico->first << " -> " << itDico->second << endl;
      itDico++;
   }
   cout << endl;

   list<cElCommand> commands;
   if ( !load_script( scriptname, commands ) )
   {
      cerr << RED_ERROR << "script file [" << scriptname.str_unix() << "] cannot be loaded" << endl;
      return EXIT_FAILURE;
   }

   // create initial state
   TracePack pack( packname, anchor );
   pack.addState();
	pack.save();

   list<cElCommand>::iterator itCmd = commands.begin();
   unsigned int iCmd = 0;
   while ( itCmd!=commands.end() ){
		cout << "command " << iCmd << " : [" << itCmd->str() << ']' << endl;
		cElCommand originalCommand = *itCmd;
		if ( itCmd->replace( dictionary ) ) cout << "command " << iCmd << " : [" << itCmd->str() << ']' << endl;
		if ( !itCmd->system() ){
			cerr << RED_ERROR << "command " << iCmd << " = " << endl;
			cerr << "[" << itCmd->str() << "]" << endl;
			cerr << "failed." << endl;
			return EXIT_FAILURE;
		}
		pack.addState( originalCommand );
		pack.save();
      
		#ifdef __DEBUG_TRACE_PACK      
			TracePack pack2( packname, anchor );
			pack2.load();
			if ( !pack.trace_compare(pack2) ){
				cerr << RED_DEBUG_ERROR << "pack!=load(write(pack))" << endl;
				exit(EXIT_FAILURE);
			}
		#endif

		itCmd++; iCmd++;
   }
      
   return EXIT_SUCCESS;
}

// this function make sure an empty directory of name i_directory exists
// return false if the directory already exist but is not empty
// or if the directory does not exist and couldn't be created
bool is_empty_or_create( const ctPath &i_directory, bool &o_removeEmpty )
{
   if ( i_directory.exists() ){
      if ( !i_directory.isEmpty() ){
			cerr << RED_ERROR << "directory [" << i_directory.str() << "] exists and is not empty" << endl;
			return false;
      }
      o_removeEmpty = false;
   }
   else if ( !i_directory.create() ){
      cerr << RED_ERROR << "cannot create directory [" << i_directory.str() << "]" << endl;
      return false;
   }
   else
      o_removeEmpty = true;
   return true;
}

void print_items_differences( unsigned int i_iStep, const list<TracePack::Registry::Item> &i_inRefItems, const list<TracePack::Registry::Item> &i_inRunItems )
{
	cout << RED_ERROR << "differences occured at step " << i_iStep << endl;
	if ( i_inRefItems.size()!=0 ){
		cout << "reference only items :" << endl;
		list<TracePack::Registry::Item>::const_iterator itItem = i_inRefItems.begin();
		while ( itItem!=i_inRefItems.end() ) (*itItem++).dump(cout,"\t");
		if ( i_inRunItems.size()!=0 ) cout << endl;
	}
	if ( i_inRunItems.size()!=0 ){
		cout << "run only items :" << endl;
		list<TracePack::Registry::Item>::const_iterator itItem = i_inRunItems.begin();
		while ( itItem!=i_inRunItems.end() ) (*itItem++).dump(cout,"\t");
	}
}

int replay_func( int argc, char **argv )
{   
	map<string,string> dictionary;
	generate_dictionary( argc, argv, dictionary );

	if ( argc<3 ){
		cerr << "usage: packname destination_directory temporary_directory [VAR1=value1 VAR2=value2 ...]" << endl;
		return EXIT_FAILURE;
	}

	const cElFilename packname( (string)(argv[0]) );
	if ( !packname.exists() ){
		cerr << RED_ERROR << "pack [" << packname.str_unix() << "] does not exist" << endl;
		return EXIT_FAILURE;
	}
	cout << "--- pack name = [" << packname.str_unix() << ']' << endl;

	ctPath run_directory( (string)(argv[1]) );
	bool runRemoveEmpty;
	if ( !is_empty_or_create( run_directory, runRemoveEmpty ) ) return EXIT_FAILURE;
	//setWorkingDirectory( run_directory );
	cout << "--- monitored directory = [" << run_directory.str() << ']' << endl;

	// create a temporary directory for file content comparison
	ctPath ref_directory( (string)(argv[2]) );
	bool refRemoveEmpty;
	if ( !is_empty_or_create( ref_directory, refRemoveEmpty ) ) return EXIT_FAILURE;
	cout << "--- temporary directory = [" << ref_directory.str() << ']' << endl;

	dictionary["${SITE_PATH}"] = run_directory.str();

	TracePack ref_pack( packname, run_directory ), // anchor is set to run directory for initial state extraction, it is later set to ref directory
	          run_pack( cElFilename("run_pack.pack"), run_directory );
   
	if ( !ref_pack.load() ) cerr << RED_ERROR << "cannot load pack [" << packname.str_unix() << "]" << endl;
	unsigned int nbStates = ref_pack.nbStates();
	if ( nbStates==0 ){
		cerr << RED_WARNING << "pack [" << packname.str_unix() << "] is empty" << endl;
		return EXIT_SUCCESS;
	}
	// run pack ignore the same files as reference pack
	list<cElFilename> files = ref_pack.getIgnoredFiles();
	run_pack.addIgnored( files );
	list<ctPath> paths = ref_pack.getIgnoredDirectories();
	run_pack.addIgnored( paths );

	vector<cElCommand> commands;
	ref_pack.getAllCommands( commands );

	// set anchor to pack's initial state
	cout << "--- setting [" << run_directory.str() << "] to initial state" << endl;
	ref_pack.setState(0);
	ref_pack.setAnchor( ref_directory ); // ref directory is where file are extracted for comparison
	run_pack.addState();
	TracePack::Registry refReg = ref_pack.getRegistry(0),
	                    runReg = run_pack.getRegistry(0);

	list<TracePack::Registry::Item> onlyInRef, onlyInRun, inBoth, differentFromDisk;
	if ( !TracePack::Registry::compare_states( refReg, runReg, onlyInRef, onlyInRun, inBoth ) ){
		print_items_differences( 0, onlyInRef, onlyInRun );
		return EXIT_FAILURE;
	}

	for ( unsigned int iState=1; iState<nbStates; iState++ ){
		cout << "--- processing registry " << iState << endl;
		cout << "\toriginal command = [" << commands[iState].str() << "]" << endl;
		commands[iState].replace( dictionary );
		cout << "\tcommand = [" << commands[iState].str() << "]" << endl;
		if ( commands[iState].nbTokens()==0 ) cerr << RED_WARNING << "the command of this registry is empty" << endl;
		if ( !commands[iState].system() ){
			cout << RED_ERROR << "command failed" << endl;
			return EXIT_FAILURE;
		}
		run_pack.addState();

		refReg = ref_pack.getRegistry( iState );
		runReg = run_pack.getRegistry( iState );
		list<TracePack::Registry::Item> onlyInRef, onlyInRun, inBoth, differentFromDisk;
		if ( !TracePack::Registry::compare_states( refReg, runReg, onlyInRef, onlyInRun, inBoth ) ){
			print_items_differences( iState, onlyInRef, onlyInRun );
			return EXIT_FAILURE;
		}

		ref_pack.compareWithItemsOnDisk( iState, run_directory, inBoth, differentFromDisk );
		if ( differentFromDisk.size()!=0 ){
			cout << "\033[1;31m---> differences occured at step " << iState << "\033[0m" << endl;
			cout << "files with data different from pack" << endl;
			list<TracePack::Registry::Item>::const_iterator itItem = differentFromDisk.begin();
			while ( itItem!=differentFromDisk.end() ){
				cout << "\t" << TD_Type_to_string( itItem->m_type ) << " on ["  << itItem->m_filename.str_unix() << "]" << endl;
				itItem++;
			}
			return EXIT_FAILURE;
		}

		cout << endl;
	}

	if ( runRemoveEmpty )
		run_directory.remove();
	else
		run_directory.removeContent();
	if ( refRemoveEmpty )
		ref_directory.remove();
	else
		ref_directory.removeContent();
   
   return EXIT_SUCCESS;
}

string normalizeCommand( char *i_command )
{
   vector<char> res( strlen( i_command )+1 );
   // remove starting '-'
   char *itSrc = i_command;
   while ( *itSrc!='\0' && *itSrc=='-' )
      itSrc++;
   // lower all letters
   char *itDst = res.data();
   while ( *itSrc!='\0' )
      *itDst++ = tolower( *itSrc++ );
   *itDst = '\0';
   return string( res.data() );
}

void test_type_raw_conversion()
{
   // test type<->raw conversions
   {
      // string<->raw
      const string str_ref = "I'm a string<->raw conversion test and I am correct";
      
      // writing
      vector<char> buffer;
      buffer.resize( string_raw_size(str_ref) );
      char *it = buffer.data();
      string_to_raw_data( str_ref, false, it );
      
      // reading
      string str2;
      const char *it_const = (const char *)buffer.data();
      string_from_raw_data( it_const, false, str2 );
      
      if ( str_ref!=str2 ){
			cerr << RED_DEBUG_ERROR << "string<->raw conversion failed, str2=[" << str2 << "]" << endl;
			exit(EXIT_FAILURE);
      }
   }
   {
      // cElDate<->raw
      cElDate date_ref = cElDate::NoDate;
      cElDate::getCurrentDate_UTC( date_ref );	 
      
      // writing
      vector<char> buffer;
      buffer.resize( date_ref.raw_size() );
      char *it = buffer.data();
      date_ref.to_raw_data( false, it );
      
      // reading
      cElDate date2 = cElDate::NoDate;
      const char *it_const = (const char *)buffer.data();
      date2.from_raw_data( it_const, false );
      
      if ( date_ref!=date2 ){
			cerr << RED_DEBUG_ERROR << "cElDate<->raw conversion failed, cElDate2=[" << date_ref << "]" << endl;
			exit(EXIT_FAILURE);
      }
   }
   
   {
      // cElCommandToken<->raw
      //ctRawString cmd_ref("ls");
      ctPath cmd_ref("ls");
   
      // writing
      vector<char> buffer;
      buffer.resize( cmd_ref.raw_size() );
      char *it = buffer.data();
      cmd_ref.to_raw_data( false, it );
	      
      // reading
      const char *it_const = (const char *)buffer.data();
      cElCommandToken *cmd2 = cElCommandToken::from_raw_data( it_const, false );
      
      if ( *cmd2!=cmd_ref )
      {
	 cerr << RED_DEBUG_ERROR << "cElCommand<->raw conversion failed, cmd2=";
	 cmd2->trace(cerr);
	 exit(EXIT_FAILURE);
      }
      delete cmd2;
   }
   
   {
      // cElCommande<->raw
      cElCommand cmd_ref;
      cmd_ref.add( ctRawString("ls") );
      cmd_ref.add( ctRawString("-l") );
      cmd_ref.add( ctPath("../") );

      // writing
      vector<char> buffer;
      buffer.resize( cmd_ref.raw_size() );
      char *it = buffer.data();
      cmd_ref.to_raw_data( false, it );

      // reading
      cElCommand cmd2;
      const char *it_const = (const char *)buffer.data();
      cmd2.from_raw_data( it_const, false );
      
      if ( cmd_ref!=cmd2 )
      {
	 string s = cmd2.str();
	 cerr << RED_DEBUG_ERROR << "cElCommand<->raw conversion failed, cmd2=[" << s << "] ---" << endl;
	 exit(EXIT_FAILURE);
      }
   }
}

bool create_random_file( const cElFilename &i_filename, U_INT8 i_size )
{
   srand( time(NULL) );
   
   ofstream fdst( i_filename.str_unix().c_str(), ios::binary );
   if ( !fdst )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "create_random_file: unable to open file [" << i_filename.str_unix() << "] for writing" << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
      
   const U_INT8 buffer_size = 10e6;
   U_INT8 remaining = i_size;
   vector<char> buffer( buffer_size );
   char *data = buffer.data();
   while ( remaining )
   {
      U_INT8 blockSize = std::min( remaining, buffer_size );      
      // generate random block
      char *itData = data;
      U_INT8 i = blockSize;
      while ( i-- ) (*itData++)=(char)( rand()%256 );
      // write block
      fdst.write( data, blockSize );
      
      remaining -= blockSize;
   }
   fdst.close();
   
   #ifdef __DEBUG_TRACE_PACK
      if ( i_size!=i_filename.getSize() ){
	 cerr << RED_ERROR << "create_random_file: size of generated file [" << i_filename.str_unix() << "] = " << i_filename.getSize() << " != " << i_size << endl;
	 exit(EXIT_FAILURE);
      }
   #endif
   
   return true;
}

cElFilename getfilename( unsigned int i_i )
{
   //return ( stringstream() << "a_list_of_data_chunk." << i_i ).str();
   stringstream ss;
   ss << "a_list_of_data_chunks." << i_i;
   return cElFilename( ss.str() );
}

class RandomEntry
{
public:
   virtual ~RandomEntry(){}
   virtual bool isFileEntry(){ return false; }
   template <class T> T & spe(){ return *((T*)this); }
   virtual bool createRandom() = 0;
   virtual void trace( ostream &io_out ) const = 0;
};

class FileEntry : public RandomEntry
{
public:
   cElFilename m_filename;
   U_INT8 m_fileSize;
   
   FileEntry( const cElFilename &i_filename, U_INT8 i_size ):m_filename(i_filename),m_fileSize(i_size){}
   
   void trace( ostream &io_out ) const { io_out << "\t\tentries.push_back( new FileEntry( cElFilename( \"" << m_filename.str_unix() << "\" ), " << m_fileSize << " ) );" << endl; }

   bool isFileEntry(){ return true; }
   
   bool createRandom()
   {
      bool res = ( m_filename.m_path.create() && create_random_file( m_filename, m_fileSize ) );
      if ( !res ){
	 cout << RED_ERROR << "unable to create a random file of size " << m_fileSize << " named [" << m_filename.str_unix() << "]" << endl;
	 return false;
      }
      return true;
   }
   
   void remove_file() const
   {
      if ( !m_filename.remove() ) cerr << RED_WARNING << "test_chunk_io: cannot remove file [" << m_filename.str_unix() << ']' << endl;
   }
   
   void remove_path() const
   {
		if ( !m_filename.m_path.isWorkingDirectory() && !m_filename.m_path.removeEmpty() )
			cerr << RED_WARNING << "test_chunk_io: cannot remove directory [" << m_filename.m_path.str() << ']' << endl;
   }
};

class BufferEntry : public RandomEntry
{
public:
   unsigned char m_type;
   vector<char>  m_buffer;
   
   BufferEntry( unsigned char i_type, unsigned int i_size ):m_type(i_type),m_buffer(i_size){}
   
   void trace( ostream &io_out ) const { io_out << "\t\tentries.push_back( new BufferEntry( " << (int)m_type << ", " << m_buffer.size() << " ) );" << endl; }
   
   bool createRandom()
   {
      for ( unsigned int i=0; i<m_buffer.size(); i++ )
	 m_buffer[i] = (char)( rand()%256 );
      return true;
   }
   
   bool isFileEntry(){ return false; }
};

bool check_items( const list<ChunkStream::Item*> &i_items, const list<RandomEntry*> &i_entries, const cElFilename i_temporaryFile )
{
   if ( i_items.size()<i_entries.size() ){
      cout << RED_ERROR << "test_chunk_io: nb ChunkStream items = " << i_items.size() << " < nb entries = " << i_entries.size() << endl;
      return false;
   }
   else
   {
      // comparing read files and files on disk
      list<ChunkStream::Item*>::const_reverse_iterator itItem = i_items.rbegin();
      list<RandomEntry*>::const_reverse_iterator itEntry = i_entries.rbegin();
      unsigned int iItem = 0;
      while ( itEntry!=i_entries.rend() )
      {
	 if ( (**itItem).isFileItem()!=(**itEntry).isFileEntry() ){
	    cout << RED_ERROR << "test_chunk_io: item " << iItem << " is of type " << ((**itItem).isFileItem()?"File":"Buffer") << " but entry if of type "
		 << ((**itEntry).isFileEntry()?"File":"Buffer") << endl;
	    return false;
	 }
	 else
	 {
	    if ( (**itItem).isFileItem() )
	    {
	       ChunkStream::FileItem &item = (**itItem).specialize<ChunkStream::FileItem>();
	       FileEntry &entry = (**itEntry).spe<FileEntry>();
	       if ( item.copyToFile( i_temporaryFile ) )
	       {
		  if ( is_equivalent( entry.m_filename, i_temporaryFile ) )
		     cout << "\t\t--- [" << entry.m_filename.str_unix() << "] checked." << endl;
		  else{
		     cout << RED_ERROR << "test_chunk_io: file [" << item.m_storedFilename.str_unix() << "] is not equivalent to its original self" << endl;
		     return false;
		  }
	       }
	       else{
		  cout << RED_ERROR << "test_chunk_io: cannot extract [" << item.m_storedFilename.str_unix() << "]" << endl;
		  return false;
	       }
	    }
	    else
	    {
	       ChunkStream::BufferItem &item = (**itItem).specialize<ChunkStream::BufferItem>();
	       BufferEntry &entry = (**itEntry).spe<BufferEntry>();
	       if ( item.m_type!=entry.m_type ){
		  cout << RED_ERROR << "test_chunk_io: BufferItem of type " << (int)item.m_type << " but type " << entry.m_type << " is expected" << endl;
		  return false;
	       }
	       else if ( item.m_buffer.size()!=entry.m_buffer.size() ){
		  cout << RED_ERROR << "test_chunk_io: BufferItem of length " << (int)item.m_buffer.size() << " but a length of " << entry.m_buffer.size() << " is expected" << endl;
		  return false;
	       }
	       else if ( memcmp( item.m_buffer.data(), entry.m_buffer.data(), entry.m_buffer.size() )!=0 ){
		  cout << RED_ERROR << "test_chunk_io: BufferItem content is different from expected" << endl;
		  return false;
	       }
	       else
		  cout << "\t\t--- a BufferItem of type " << (int)item.m_type << " and length " << item.m_buffer.size() << " checked." << endl;
	    }
	 }
	 itItem++; itEntry++; iItem++;
      }
   }
   return true;
}

void generate_random_entries( const unsigned int i_nbSets, const unsigned int i_nbItemsPerSet, const U_INT8 i_maxEntrySize, list<list<RandomEntry*> > &o_entries )
{
   cout << "---> generate random entries list" << endl;
   o_entries.clear();   
   for ( unsigned int iSet=0; iSet<i_nbSets; iSet++ ){
      list<RandomEntry*> entries;
      for ( unsigned int iEntry=0; iEntry<i_nbItemsPerSet; iEntry++ ){
	 unsigned int size = ( rand()%i_maxEntrySize )+1;
	 if ( ( rand()%2 )==0 )
	    entries.push_back( new FileEntry( cElFilename( ( (rand()%2)==0?generate_random_string(5,10):string(".") )+"/"+generate_random_string(5,10) ), size ) );
	 else
	    entries.push_back( new BufferEntry( rand()%64, size ) );
      }
      o_entries.push_back(entries);
   }
   cout << "<--- done\n" << endl;

   /*
   cout << "---> load entries list" << endl;
   {
		list<RandomEntry*> entries;		
		entries.push_back( new FileEntry( cElFilename( "tzJbjtjACE/vgalkqZ" ), 605 ) );
		entries.push_back( new BufferEntry( 33, 240 ) );
		entries.push_back( new FileEntry( cElFilename( "jfJcxt/tybHW" ), 462 ) );
		entries.push_back( new FileEntry( cElFilename( "ZrXHT" ), 860 ) );
		entries.push_back( new FileEntry( cElFilename( "ELBjUqplOl" ), 721 ) );
		entries.push_back( new BufferEntry( 42, 957 ) );
		entries.push_back( new BufferEntry( 48, 874 ) );
		entries.push_back( new BufferEntry( 4, 115 ) );
		entries.push_back( new BufferEntry( 16, 387 ) );
		entries.push_back( new BufferEntry( 47, 992 ) );
		o_entries.push_back(entries);
	}
	{
		list<RandomEntry*> entries;
		entries.push_back( new FileEntry( cElFilename( "TlngVIpdEX" ), 336 ) );
		entries.push_back( new FileEntry( cElFilename( "waeSFv" ), 192 ) );
		entries.push_back( new FileEntry( cElFilename( "an_empty_file0" ), 0 ) );
		entries.push_back( new FileEntry( cElFilename( "XQxfSeBmg" ), 66 ) );
		entries.push_back( new FileEntry( cElFilename( "GJCJL/mzWMaVXK" ), 207 ) );
		entries.push_back( new BufferEntry( 0, 742 ) );
		entries.push_back( new FileEntry( cElFilename( "tuZRLrtb" ), 802 ) );
		entries.push_back( new BufferEntry( 21, 185 ) );
		entries.push_back( new BufferEntry( 44, 719 ) );
		entries.push_back( new BufferEntry( 17, 214 ) );
		entries.push_back( new BufferEntry( 50, 989 ) );
		entries.push_back( new FileEntry( cElFilename( "an_empty_file" ), 0 ) );
		// a long name
		//entries.push_back( new FileEntry( cElFilename( generate_random_string(i_maxEntrySize) ), 1024 ) );
		o_entries.push_back(entries);
	}
   cout << "<--- done\n" << endl;
   */
}

bool add_items( const list<list<RandomEntry*> > &i_entries, ChunkStream &io_stream, const cElFilename &i_temporaryFile )
{
   list<list<RandomEntry*> >::const_iterator itList = i_entries.begin();
   size_t iSet = 0;
   while ( itList!=i_entries.end() ){
      cout << "---> set " << iSet++ << endl;
      const list<RandomEntry*> &entries = *itList++;
      
      if ( !io_stream.writeOpen() ){
	 cout << RED_ERROR << "add_items: unable to open ChunkStream [" << io_stream.getFilename(0).str_unix() << "] for writing" << endl;
	 return false;
      }
      
      cout << "\t---> writing chunks" << endl;
      list<RandomEntry*>::const_iterator itEntry = entries.begin();
      while ( itEntry!=entries.end() )
      {
	 if ( (*itEntry)->isFileEntry() )
	 {
	    // generate a random file
	    FileEntry &entry = (*itEntry)->spe<FileEntry>();
	    if ( entry.createRandom() ) io_stream.write_file( entry.m_filename, entry.m_filename );
	 }
	 else // a buffer chunk
	 {
	    // generate a random buffer
	    BufferEntry &entry = (*itEntry)->spe<BufferEntry>();
	    if ( entry.createRandom() ) io_stream.write_buffer( entry.m_type, entry.m_buffer );
	 }
	 itEntry++;
      }
      cout << "\t<--- done" << endl;
            
      cout << "\t---> reading Items" << endl;
      list<ChunkStream::Item*> items;
      if ( !io_stream.read( 0, 0, items ) )
	 cout << RED_ERROR << "add_items: error while reading chunkStream [" << io_stream.getFilename(0).str_unix() << "]" << endl;
      cout << "\t<--- done" << endl;
      
      // compare write+read Items and initial Entries
      cout << "\t---> checking read items" << endl;
      if ( !check_items( items, entries, i_temporaryFile ) ){
	 cout << RED_ERROR << "add_items: error checking write+read items in stream [" << io_stream.getFilename(0).str_unix() << "]" << endl;
	 return false;
      }
      cout << "\t<--- done" << endl;

      clear_item_list( items );
      cout << "<--- done" << endl;
   }
   
   return true;
}

void trace_all_entries( const cElFilename &i_filename, const list<list<RandomEntry*> > &i_all_entries )
{
   ofstream dst( i_filename.str_unix().c_str() );
   if ( !dst ){
      cout << RED_ERROR << "trace_all_entries: cannot open trace file [" << i_filename.str_unix() << endl; 
      exit(EXIT_FAILURE);
   }

   list<list<RandomEntry*> >::const_iterator itList = i_all_entries.begin();
   while ( itList!=i_all_entries.end() ){
      dst << "\t{" << endl;
      dst << "\t\tlist<RandomEntry*> entries;" << endl;
      
      list<RandomEntry*>::const_iterator itEntry = itList->begin();
      while ( itEntry!=itList->end() ) (*itEntry++)->trace(dst);
         
      dst << "\t\to_entries.push_back(entries);" << endl;
      dst << "\t}" << endl;
      
      itList++;
   }
}
   
void remove_all_entries( const list<list<RandomEntry*> > &i_all_entries )
{
   // remove all files
   list<list<RandomEntry*> >::const_iterator itList = i_all_entries.begin();
   while ( itList!=i_all_entries.end() ){
      list<RandomEntry*>::const_iterator itEntry = itList->begin();
      while ( itEntry!=itList->end() ){
	 if ( (*itEntry)->isFileEntry() ) (*itEntry)->spe<FileEntry>().remove_file();
	 itEntry++;
      }
      itList++;
   }
   
   // remove all empty directories 
   itList = i_all_entries.begin();
   while ( itList!=i_all_entries.end() ){
      list<RandomEntry*>::const_iterator itEntry = itList->begin();
      while ( itEntry!=itList->end() ){
	 if ( (*itEntry)->isFileEntry() ) (*itEntry)->spe<FileEntry>().remove_path();
	 itEntry++;
      }
      itList++;
   }
   
   // delete RandomEntries items 
   itList = i_all_entries.begin();
   while ( itList!=i_all_entries.end() ){
      list<RandomEntry*>::const_iterator itEntry = itList->begin();
      while ( itEntry!=itList->end() )
	 delete (*itEntry++);
      itList++;
   }
}

void test_chunk_stream()
{
   // create random items registry, ignored lists or file
   srand( time(NULL) );

   cElFilename streamName("a_stream");
   const U_INT8 maxChunkSize = 4e9;
   ChunkStream chunkStream( streamName, maxChunkSize, false ); // false = reverseByteOrder

   const unsigned int nbSets = 2;
   const unsigned int nbEntriesPerSet = 10;
   list<list<RandomEntry*> > all_entries;
   generate_random_entries( nbSets, nbEntriesPerSet, 1<<20, all_entries );
   cElFilename traceFile("trace");
   trace_all_entries( traceFile, all_entries );
   
   cElFilename temporaryFile( "comparisonTemporaryFile" ); // temporary file for file data comparison
   if ( !add_items( all_entries, chunkStream, temporaryFile ) )
      cerr << RED_ERROR << "test_chunk_stream: items couldn't be written then read correctly" << endl;
   
   cout << endl;
   cout << "---> cleaning" << endl;
   // remove temporary file for file comparison
   if ( temporaryFile.exists() ) temporaryFile.remove();
   // remove entries trace file
   if ( traceFile.exists() ) traceFile.remove();
   // remove entry files and delete allocated objects
   remove_all_entries( all_entries );
   // clear stream
   if ( !chunkStream.remove() ) cout << RED_ERROR << "test_chunk_stream: unable to remove chunkStream [" << chunkStream.getFilename(0).str_unix() << ']' << endl;
   cout << "--- done" << endl;
}

char * __copy_string( const string &i_str )
{
   char *cstr = new char[i_str.length()+1];
   strcpy( cstr, i_str.c_str() );
   return cstr;
}

int test_func( int argc, char **argv )
{
   test_type_raw_conversion();
   test_chunk_stream();
   
   int return_code = EXIT_FAILURE;
   char **args = NULL;
   ofstream f;
   ctPath tempDirectory("test_trace_pack");
   cElFilename packName("test.pack");
   
   if ( tempDirectory.exists() ){ cerr << RED_ERROR << "test_trace_pack: temporary directory [" << tempDirectory.str() << "] already exists" << endl; return EXIT_FAILURE; }
   if ( !tempDirectory.create() ){ cerr << RED_ERROR << "test_trace_pack: cannot create temporary directory [" << tempDirectory.str() << "]" << endl; return EXIT_FAILURE; }
   
   // create a test script
   cElFilename scriptName( "test.script" );
   if ( scriptName.exists() ){ cerr << RED_ERROR << "test_trace_pack: test script [" << scriptName.str_unix() << "] already exists" << endl; goto test_trace_pack_clean; }
   f.open( scriptName.str_unix().c_str() );
   if ( !f ){ cerr << RED_ERROR << "test_trace_pack: test script file [" << scriptName.str_unix() << "] cannot be opened for writing" << endl; goto test_trace_pack_clean; }
   f << "echo \"i am toto\" > toto" << endl;
   f << "echo ls > tata && chmod +x tata" << endl;
	#if ELISE_windows
		f << "date /T > titi && time /T >> titi" << endl;
		f << "del toto" << endl;
	#else
		f << "date > titi" << endl;
		f << "rm titi" << endl;
	#endif
   f.close();
   
   // create arguments to call pack_from_script_func
   
   setWorkingDirectory( tempDirectory );
   args = new char*[3];
   args[0] = __copy_string( string("../")+packName.str_unix() );
   args[1] = __copy_string( string("../")+scriptName.str_unix() );
   args[2] = __copy_string( "./" );
   
   return_code = pack_from_script_func( 3, args );
   setWorkingDirectory( ctPath("../") );
   
test_trace_pack_clean:
   if ( scriptName.exists() ) scriptName.remove();
   if ( tempDirectory.exists() ) tempDirectory.remove();
   if ( args!=NULL ){      
      delete [] args[0];
      delete [] args[1];
      delete [] args[2];
      delete [] args;
   }
   ChunkStream stream( packName, 0, true );
   stream.remove();
   
   return return_code;
}

void parse_ignored_files_and_paths( int i_argc, char **i_argv, list<cElFilename> &o_filenames, list<ctPath> &o_paths )
{
	o_filenames.clear();
	o_paths.clear();
	int iarg;
	for ( iarg=0; iarg<i_argc; iarg++ ){
		if ( strcmp(i_argv[iarg], "PATH")==0 ) break;
		o_filenames.push_back( cElFilename( string(i_argv[iarg]) ) );
	}
	for ( iarg++; iarg<i_argc; iarg++ )	o_paths.push_back( ctPath( string(i_argv[iarg]) ) );
}

int add_ignored_func( int argc, char **argv )
{
	if ( argc<1 ){
      cerr << "usage: packname [file0 file1 ...] [PATH path0 path1 ...]" << endl;
      return EXIT_FAILURE;
	}
	
   const cElFilename packname( (string)(argv[0]) );
   if ( !packname.exists() ){
		cerr << RED_ERROR << "pack [" << packname.str_unix() << "] does not exist" << endl;
		return EXIT_FAILURE;
	}
	TracePack pack( packname, ctPath() );
	if ( !pack.load() ){
		cerr << RED_ERROR << "cannot load pack [" << packname.str_unix() << "]" << endl;
		return EXIT_FAILURE;
	}

	if ( argc>1 ){
		list<cElFilename> filenames;
		list<ctPath> paths;
		parse_ignored_files_and_paths( argc-1, argv+1, filenames, paths );
		if ( pack.addIgnored( filenames ) || pack.addIgnored( paths ) ){
			cout << "ignored lists changed" << endl;
			if ( !pack.save() ){
				cout << RED_ERROR << "cannot save pack [" << packname.str_unix() << endl;
				return EXIT_FAILURE;
			}
		}
	}
	
	const list<ctPath> &paths = pack.getIgnoredDirectories();
	cout << paths.size() << " ignored path" << (paths.size()==0?'\0':'s') << endl;
	list<ctPath>::const_iterator itPath = paths.begin();
	while ( itPath!=paths.end() )	cout << "\t[" << (*itPath++).str() << "]" << endl;
	
	const list<cElFilename> &files = pack.getIgnoredFiles();
	cout << files.size() << " ignored file" << (files.size()==0?'\0':'s') << endl;
	list<cElFilename>::const_iterator itFile = files.begin();
	while ( itFile!=files.end() )	cout << "\t[" << (*itFile++).str_unix() << "]" << endl;
	
	return EXIT_SUCCESS;
}

bool test_prerequisites()
{
   #ifdef __DEBUG_TRACE_PACK
      if ( sizeof(U_INT4)!=4 ) { cerr << RED_ERROR << "sizeof(INT4)!=4" << endl; return false; }
      if ( sizeof(U_INT8)!=8 ) { cerr << RED_ERROR << "sizeof(INT8)!=8" << endl; return false; }
      if ( sizeof(REAL4)!=4 ) { cerr << RED_ERROR << "sizeof(REAL4)!=4" << endl; return false; }
      if ( sizeof(REAL8)!=8 ) { cerr << RED_ERROR << "sizeof(REAL8)!=8" << endl; return false; }
      if ( FLT_MANT_DIG!=24 || DBL_MANT_DIG!=53 )
			cerr << "WARNING: single and/or double precision floating-point numbers do not conform to IEEE 754, you may experiment problems while sharing pack files with other systems" << endl;
      cout << "test_prerequisites: ok." << endl;
   #endif
   return true;
}

int main( int argc, char **argv )
{
   if ( !test_prerequisites() ) return EXIT_FAILURE;
       
   if ( argc<2 ) return help_func(0,NULL);

   string command = normalizeCommand( argv[1] );
   command_t *itCommand = g_commands;
   while ( itCommand->func!=NULL )
   {
      if (command==itCommand->name)
	 return (*itCommand->func)(argc-2, argv+2);
      itCommand++;
   }

   #ifdef __DEBUG_C_EL_COMMAND
      cout << "__nb_distroyed : " << cElCommandToken::__nb_distroyed << endl;
      cout << "__nb_created   : " << cElCommandToken::__nb_created << endl;
      if ( cElCommandToken::__nb_distroyed!=cElCommandToken::__nb_created )
	 cerr << RED_DEBUG_ERROR << "some cElCommandToken have not been freed : " << cElCommandToken::__nb_created << " created, " << cElCommandToken::__nb_distroyed << " distroyed" << endl;
   #endif
   
   return help_func(0,NULL);
}
