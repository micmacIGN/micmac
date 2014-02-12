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
int help_func( int, char ** );

command_t g_commands[] = {
   { "snapshot", snapshot_func },
   { "list", list_func },
   { "getfile", getfile_func },
   { "setstate", setstate_func },
   { "pack_from_script", pack_from_script_func },
   { "replay", replay_func },
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
   if ( argc!=2 )
   {
      cout << "not enough args" << endl;
      return EXIT_FAILURE;
   }
   const ctPath anchor( (string)(argv[1]) );
   const cElFilename packname( (string)(argv[0]) );
   TracePack pack( packname, anchor );
   
   if ( packname.exists() )
   {
      if ( !pack.load() )
      {
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

   
   if ( !pack.save() )
   {
      cerr << RED_ERROR << "unable to save to [" << packname.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      TracePack pack2( packname, anchor );  
      pack2.load();
      if ( !pack.trace_compare( pack2 ) )
	 cerr << RED_DEBUG_ERROR << "pack [" << packname.str_unix() << "] is not equal to its write+read copy" << endl;
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
   if ( !pack.load() )
   {
      cerr << RED_ERROR << "unable to load [" << filename.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   
   if ( argc>0 )
   {
      const unsigned int nbStates = pack.nbStates();
      cout << "pack [" << filename.str_unix() << ']' << endl;
      cout << "\t- creation date " << pack.date() << " (UTC)"<< endl;
      cout << "\t- " << nbStates << (nbStates>1?" registries":" registry") << endl;
      cout << "commands" << endl;
      vector<cElCommand> commands;
      pack.getAllCommands( commands );
      for ( unsigned int iCmd=0; iCmd<nbStates; iCmd++ )
	 cout << "\t[" << commands[iCmd].str() << "]" << endl;
   }
   if ( argc==2 )
   {
      bool listState;
      unsigned int iRegistry;
      if ( !get_list_registry_number( argv[1], iRegistry, listState ) )
      {
	 cerr << RED_ERROR << "third argument should be sXXX if you want to list a state or rXXX if you want to list a raw registry" << endl;
	 return EXIT_FAILURE;
      }

      if ( iRegistry>=pack.nbStates() )
      {
	 cerr << RED_ERROR << "index " << iRegistry << " out of range (the pack has " << pack.nbStates() << " states)" << endl;
	 return EXIT_FAILURE;
      }
      if ( listState )
      {
	 TracePack::Registry reg;
	 pack.getState( (unsigned int)iRegistry, reg );
	 cout << "state " << iRegistry << endl;
	 reg.dump( cout, "\t" );
      }
      else
      {
	 cout << "raw registry " << iRegistry << endl;
	 pack.getRegistry(iRegistry).dump( cout, "\t" );
      }
   }

   return EXIT_SUCCESS;
}

int getfile_func( int argc, char **argv )
{
   if ( argc!=4 )
   {
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
   
   if ( anchor.contains( packname ) )
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
   
   list<cElCommand>::iterator itCmd = commands.begin();
   unsigned int iCmd = 0;
   while ( itCmd!=commands.end() )
   {
      cout << "command " << iCmd << " : [" << itCmd->str() << ']' << endl;
      cElCommand originalCommand = *itCmd;
      itCmd->replace( dictionary );
      if ( !itCmd->system() )
      {
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
	 if ( !pack.trace_compare(pack2) )
	 {
	    cerr << RED_DEBUG_ERROR << "load(write(pack))!=pack" << endl;
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
bool is_empty_or_create( const ctPath &i_directory )
{
   if ( i_directory.exists() )
   {
      if ( !i_directory.isEmpty() )
      {
	 cerr << RED_ERROR << "directory [" << i_directory.str() << "] exists and is not empty" << endl;
	 return false;
      }
   }
   else if ( !i_directory.create() )
   {
      cerr << RED_ERROR << "unable to create directory [" << i_directory.str() << "]" << endl;
      return false;
   }
   return true;
}

int replay_func( int argc, char **argv )
{   
   map<string,string> dictionary;
   generate_dictionary( argc, argv, dictionary );
   
   if ( argc<3 )
   {
      cerr << "usage: packname destination_directory temporary_directory" << endl;
      return EXIT_FAILURE;
   }
    
   const cElFilename packname( (string)(argv[0]) );
   if ( !packname.exists() )
   {
      cerr << RED_ERROR << "pack [" << packname.str_unix() << "] does not exist" << endl;
      return EXIT_FAILURE;
   }
   
   ctPath run_directory( (string)(argv[1]) );
   if ( !is_empty_or_create( run_directory ) ) return EXIT_FAILURE;
   //setWorkingDirectory( run_directory );
   
   // create a temporary directory for file content comparison
   ctPath ref_directory( (string)(argv[2]) );
   if ( !is_empty_or_create( ref_directory ) ) return EXIT_FAILURE;
   
   dictionary["${SITE_PATH}"] = run_directory.str();
   
   TracePack ref_pack( packname, ref_directory ),
             run_pack( cElFilename("run_pack.pack"), run_directory );
   if ( !ref_pack.load() )
   {
      cerr << RED_ERROR << "unable to load pack [" << packname.str_unix() << "]" << endl;
   }
   unsigned int nbStates = ref_pack.nbStates();
   if ( nbStates==0 )
   {
      cerr << RED_WARNING << "pack [" << packname.str_unix() << "] is empty" << endl;
      return EXIT_SUCCESS;
   }
   
   vector<cElCommand> commands;
   ref_pack.getAllCommands( commands );
   
   // set anchor to pack's initial state
   ref_pack.setState( 0 );
   run_pack.addState();
   TracePack::Registry refReg = ref_pack.getRegistry(0),
		       runReg = run_pack.getRegistry(0);
   
   list<TracePack::Registry::Item> onlyInRef, onlyInRun, inBoth, differentFromDisk;
   TracePack::Registry::compare_states( refReg, runReg, onlyInRef, onlyInRun, inBoth );
   if ( onlyInRef.size()!=0 || onlyInRun.size()!=0 )
   {
      cout << "\t---> a difference occured" << endl;
      return EXIT_FAILURE;
   }
   
   for ( unsigned int iState=1; iState<nbStates; iState++ )
   {
      cout << "--- processing registry " << iState << endl;
      cout << "\toriginal command = [" << commands[iState].str() << "]" << endl;
      commands[iState].replace( dictionary );
      cout << "\tcommand = [" << commands[iState].str() << "]" << endl;
      if ( commands[iState].nbTokens()==0 )
	 cerr << RED_WARNING << "the command of this registry is empty" << endl;
      commands[iState].system();
      run_pack.addState();
      
      refReg = ref_pack.getRegistry( iState );
      runReg = run_pack.getRegistry( iState );      
      TracePack::Registry::compare_states( refReg, runReg, onlyInRef, onlyInRun, inBoth );
      
      if ( onlyInRef.size()!=0 || onlyInRun.size()!=0 )
      {
	 cout << "\033[1;31m---> differences occured at step " << iState << "\033[0m" << endl;
	 if ( onlyInRef.size()!=0 )
	 {
	    cout << "actions that should have occured but didn't" << endl;
	    list<TracePack::Registry::Item>::const_iterator itItem = onlyInRef.begin();
	    while ( itItem!=onlyInRef.end() )
	    {
	       cout << "\t" << TD_Type_to_string( itItem->m_type ) << " on ["  << itItem->m_filename.str_unix() << "]" << endl;
	       itItem++;
	    }
	 }
	 if ( onlyInRun.size()!=0 )
	 {
	    if ( onlyInRef.size()!=0 ) cout << endl;
	    cout << "actions that occured but shouldn't have" << endl;
	    list<TracePack::Registry::Item>::const_iterator itItem = onlyInRun.begin();
	    while ( itItem!=onlyInRun.end() )
	    {
	       cout << "\t" << TD_Type_to_string( itItem->m_type ) << " on ["  << itItem->m_filename.str_unix() << "]" << endl;
	       itItem++;
	    }
	 }
	 return EXIT_FAILURE;
      }
      
      ref_pack.compareWithItemsOnDisk( iState, run_directory, inBoth, differentFromDisk );
      if ( differentFromDisk.size()!=0 )
      {
	 cout << "\033[1;31m---> differences occured at step " << iState << "\033[0m" << endl;
	 cout << "files with data different from pack" << endl;
	 list<TracePack::Registry::Item>::const_iterator itItem = differentFromDisk.begin();
	 while ( itItem!=differentFromDisk.end() )
	 {
	    cout << "\t" << TD_Type_to_string( itItem->m_type ) << " on ["  << itItem->m_filename.str_unix() << "]" << endl;
	    itItem++;
	 }
	 return EXIT_FAILURE;
      }
      
      cout << endl;
   }
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
      it = buffer.data();
      string_from_raw_data( it, false, str2 );
      
      if ( str_ref!=str2 )
      {
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
      it = buffer.data();
      date2.from_raw_data( it, false );
      
      if ( date_ref!=date2 )
      {
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
      it = buffer.data();
      cElCommandToken *cmd2 = cElCommandToken::from_raw_data( it, false );
      
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
      it = buffer.data();
      cmd2.from_raw_data( it, false );
      
      if ( cmd_ref!=cmd2 )
      {
	 string s = cmd2.str();
	 cerr << RED_DEBUG_ERROR << "cElCommand<->raw conversion failed, cmd2=[" << s << "] ---" << endl;
	 exit(EXIT_FAILURE);
      }
   }
}

bool create_random_file( const cElFilename &i_filename, unsigned int i_size )
{
   srand( time(NULL) );
   
   ofstream f( i_filename.str_unix().c_str(), ios::binary );
   if ( !f )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "create_random_file: unable to open file [" << i_filename.str_unix() << "] for writing" << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
      
   unsigned int i = i_size;
   while ( i-- ) f.put( (char)( rand()%256 ) );
   f.close();
   
   if ( i_filename.getSize()!=i_size )
   {
      #ifdef __DEBUG_TRACE_PACK
	 cerr << RED_DEBUG_ERROR << "create_random_file: size of [" << i_filename.str_unix() << "] = " << i_filename.getSize() << " expecting " << i_size << endl;
	 exit(EXIT_FAILURE);
      #endif
      return false;
   }
   
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
   virtual bool isFileEntry(){ return false; }
   template <class T> T & spe(){ return *((T*)this); }
   virtual void remove_file() const {}
   virtual void remove_path() const {}
};

class FileEntry : public RandomEntry
{
public:
   cElFilename m_filename;
   unsigned int m_fileSize;
   
   FileEntry( const cElFilename &i_filename, unsigned int i_size ):m_filename(i_filename),m_fileSize(i_size){}
   bool isFileEntry(){ return true; }
   void remove_file() const
   {
      if ( !m_filename.remove() ) cerr << RED_WARNING << "test_chunk_io: cannot remove file [" << m_filename.str_unix() << ']' << endl;
   }
   void remove_path() const
   {
      if ( !m_filename.m_path.isWorkingDirectory() && !m_filename.m_path.remove_empty() )
	 cerr << RED_WARNING << "test_chunk_io: cannot remove directory [" << m_filename.m_path.str() << ']' << endl;
   }
};

class RawDataEntry : public RandomEntry
{
public:
   FileChunk::FileChunkType m_type;
   vector<char> m_buffer;
   
   RawDataEntry( FileChunk::FileChunkType i_type, unsigned int i_size ):m_type(i_type),m_buffer(i_size){}
   bool isFileEntry(){ return false; }
};

void check_file_and_data_chunks( const cElFilename &i_filename, const list<FileChunk> &o_chunks )
{
   unsigned int dataSum = 0;
   list<FileChunk>::const_iterator it = o_chunks.begin();
   while ( it!=o_chunks.end() )
      dataSum += ( it++ )->m_dataSize;
   unsigned int actualSize = i_filename.getSize()+string_raw_size( i_filename.str_unix() );
   if ( dataSum!=actualSize )
      cerr << RED_DEBUG_ERROR << "check_file_and_data_chunks: chunks sum = " << dataSum << " != " << " actualSize = " << actualSize << endl;
}

void test_chunk_io()
{
   //vector<ChunkStreamEntry> chunkStream;
   //list<FileChunk> chunks;
      
   // create random items registry, ignored lists or file
   srand( time(NULL) );
   
   const unsigned int nbEntries = 10;
   const U_INT4 maxChunkSize = 512;
   unsigned int remain = maxChunkSize;
   list<RandomEntry*> entries;
   
   // random items
   for ( unsigned int iEntry=0; iEntry<nbEntries; iEntry++ )
   {
      const unsigned int itemType = ( rand()%2 );
      if ( itemType==0 ) // a file
      {
	 cElFilename filename( ( (rand()%2)==0?generate_random_string(5,10):string(".") )+"/"+generate_random_string(5,10) );
	 unsigned int fileSize = rand()%600+1;
	 cout << "entries.push_back( new FileEntry( cElFilename( \"" << filename.str_unix() << "\" ), " << fileSize << " ) );" << endl;
	 entries.push_back( new FileEntry( filename, fileSize ) );
      }
      else // a buffer chunk
      {
	 unsigned int bufferSize = rand()%(maxChunkSize-FileChunk::headerSize)+1;
	 FileChunk::FileChunkType type = ( (rand()%2)==0 ? FileChunk::FCT_Registry : FileChunk::FCT_Ignore );
	 cout << "entries.push_back( new RawDataEntry( FileChunk::" << FileChunkType_to_string(type) << ", " << bufferSize << " ) );" << endl;
	 entries.push_back( new RawDataEntry( type, bufferSize ) );
      }
   }
   cout << endl;
   cout << "-----------------------> random list done" << endl;
   
   /*
   entries.push_back( new FileEntry( cElFilename( "dvlMU/sZAXnRdGH" ), 499 ) );
   entries.push_back( new RawDataEntry( FileChunk::FCT_Registry, 0 ) );
   entries.push_back( new RawDataEntry( FileChunk::FCT_Ignore, 177 ) );
   entries.push_back( new FileEntry( cElFilename( "eWswf" ), 536 ) );
   entries.push_back( new FileEntry( cElFilename( "FtOzc/ZqHPvB" ), 142 ) );
   entries.push_back( new FileEntry( cElFilename( "JpbINQmnS/BNZLrWUpW" ), 408 ) );
   entries.push_back( new FileEntry( cElFilename( "COvgyi" ), 486 ) );
   entries.push_back( new RawDataEntry( FileChunk::FCT_Registry, 14 ) );
   entries.push_back( new RawDataEntry( FileChunk::FCT_Ignore, 479 ) );
   entries.push_back( new RawDataEntry( FileChunk::FCT_Ignore, 489 ) );
   cout << "-----------------------> saved list loaded" << endl;
   */
   
   list<FileChunk> chunks;
   list<RandomEntry*>::iterator itEntry = entries.begin();
   while ( itEntry!=entries.end() )
   {
      if ( (*itEntry)->isFileEntry() )
      {
	 FileEntry &entry = (*itEntry)->spe<FileEntry>();
	 if ( !entry.m_filename.m_path.create() || !create_random_file( entry.m_filename, entry.m_fileSize ) )
	 {
	    cout << RED_ERROR << "test_chunk_io: unable to create file [" << entry.m_filename.str_unix() << ']' << endl;
	    exit(EXIT_FAILURE);
	 }
	 file_to_chunk_list( entry.m_filename.str_unix(), entry.m_fileSize, remain, maxChunkSize, chunks );
      }
      else // a buffer chunk
      {
	 RawDataEntry &entry = (*itEntry)->spe<RawDataEntry>();
	 for ( unsigned int i=0; i<entry.m_buffer.size(); i++ )
	    entry.m_buffer[i] = (char)( rand()%256 );
	 chunks.push_back( FileChunk( entry.m_type, entry.m_buffer ) );
	 unsigned int fullSize = chunks.back().fullSize();
	 if ( remain>=fullSize )
	    remain -= fullSize;
	 else
	    remain = maxChunkSize-fullSize;
      }
      itEntry++;
   }
   cout << "-----------------------> creation of chunks done" << endl;
   
   ChunkStream chunkStream( cElFilename( "a_stream" ), maxChunkSize, false ); // false = reverseByteOrder
   cout << "--- writing chunks" << endl;
   cout << "nb chunks = " << chunks.size() << endl;
   chunkStream.writeChunks( chunks );
   cout << "--- writing done\n" << endl;
   
   // read the chunkStream
   cout << "--- reading chunks" << endl;
   chunkStream.readChunks(chunks);
   cout << "nb chunks = " << chunks.size() << endl;
   cout << "--- reading done." << endl;
   
   // comparing read files and files on disk
   cElFilename temporaryFile( "comparisonTemporaryFile" );
   list<FileChunk>::const_iterator itc = chunks.begin();
   itEntry = entries.begin();
   while ( itc!=chunks.end() )
   {
      if ( itc->m_type==FileChunk::FCT_Data )
      {
	 cElFilename storedFilename = cElFilename( itc->m_filename );
	 list<FileChunk> fileChunks;
	 while ( itc->m_hasMore ) fileChunks.push_back( *itc++ );
	 fileChunks.push_back( *itc );
	 
	 if ( chunk_list_to_file( temporaryFile, fileChunks ) )
	 {
	    if ( is_equivalent( storedFilename, temporaryFile ) )
	       cout << "--- [" << storedFilename.str_unix() << "] checked." << endl;
	    else
	       cout << RED_ERROR << "test_chunk_io: file [" << storedFilename.str_unix() << "] extracted from ChunkStream [" << chunkStream.filename(0).str_unix()
		    << "] is not equivalent to it original self" << endl;
	 }
	 else
	    cout << RED_ERROR << "test_chunk_io: cannot extract file [" << storedFilename.str_unix() << "] from ChunkStream [" << chunkStream.filename(0).str_unix() << "]" << endl;
      }
      else
      {
	 if ( !(*itEntry)->isFileEntry() )
	 {
	    RawDataEntry &entry = (*itEntry)->spe<RawDataEntry>();
	    const string chunkTypeStr = FileChunkType_to_string(itc->m_type);
	    if ( itc->m_contentSize==entry.m_buffer.size() &&
		 itc->m_type==entry.m_type &&
		 memcmp( itc->m_rawData.data(), entry.m_buffer.data(), itc->m_contentSize )==0 )
	       cout << "--- a " << chunkTypeStr << " of size " << itc->m_contentSize << " checked." << endl;
	    else
	       cout << RED_ERROR << "test_chunk_io: " << chunkTypeStr << " of size " << itc->m_contentSize << " != entry = "
		    << FileChunkType_to_string(entry.m_type) << " of size " << entry.m_buffer.size() << endl;
	 }
	 else
	    cout << RED_ERROR << "test_chunk_io: a RawDataEntry is expected" << endl;
      }
      itc++; itEntry++;
   }
   
   // remove files
   itEntry = entries.begin();
   while ( itEntry!=entries.end() ) ( *itEntry++ )->remove_file();
   itEntry = entries.begin();
   while ( itEntry!=entries.end() ) ( *itEntry++ )->remove_path();
   
   // remove temporary file for file comparison
   if ( temporaryFile.exists() ) temporaryFile.remove();
   
   // delete entries
   itEntry = entries.begin();
   while ( itEntry!=entries.end() ) delete *itEntry++;
   
   if ( !chunkStream.remove() ) cout << RED_ERROR << "test_chunk_io: unable to remove chunkStream [" << chunkStream.filename(0).str_unix() << ']' << endl;
}

bool test_prerequisites()
{   
   #ifdef __DEBUG_TRACE_PACK
      if ( sizeof(U_INT4)!=4 ) { cerr << "sizeof(INT4)!=4" << endl; return false; }
      if ( sizeof(REAL4)!=4 ) { cerr << "sizeof(REAL4)!=4" << endl; return false; }
      if ( sizeof(REAL8)!=8 ) { cerr << "sizeof(REAL8)!=8" << endl; return false; }
      if ( FLT_MANT_DIG!=24 || DBL_MANT_DIG!=53 )
	 cerr << "WARNING: single and/or double precision floating-point numbers do not conform to IEEE 754, you may experiment problems while sharing pack files with other systems" << endl;
      
      test_type_raw_conversion();
      test_chunk_io();
      
      cout << "test_prerequisites: ok." << endl;
   #endif
   return true;
}

int main( int argc, char **argv )
{
   if ( !test_prequisites() ) return EXIT_FAILURE;
   
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
