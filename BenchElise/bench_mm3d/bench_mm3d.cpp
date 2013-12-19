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
int help_func( int, char ** );

command_t g_commands[] = {
   { "snapshot", snapshot_func },
   { "list", list_func },
   { "getfile", getfile_func },
   { "setstate", setstate_func },
   { "pack_from_script", pack_from_script_func },
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
   const cElPath anchor( (string)(argv[1]) );
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
   
   cout << "snaping [" << anchor.str_unix() << "] to [" << packname.str_unix() << "]" << endl;
   pack.addState();

   const unsigned int nbStates = pack.nbStates();
   cerr << nbStates << " state" << (nbStates>1?'s':'\0') << endl;

   
   if ( !pack.save() )
   {
      cerr << "ERROR: unable to save to [" << packname.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   
   #ifdef __DEBUG_TRACE_PACK
      TracePack pack2( packname, anchor );  
      pack2.load();
      if ( !pack.compare( pack2 ) )
	 cerr << "DEBUG_ERROR: pack [" << packname.str_unix() << "] is not equal to its write+read copy" << endl;
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
   
   TracePack pack( filename, cElPath("./") );
   if ( !pack.load() )
   {
      cerr << "ERROR: unable to load [" << filename.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   
   if ( argc>0 )
   {
      const unsigned int nbStates = pack.nbStates();
      cout << "pack [" << filename.str_unix() << ']' << endl;
      cout << "\t- creation date " << pack.date() << " (UTC)"<< endl;
      cout << "\t- " << nbStates << (nbStates>1?" registries":" registry") << endl;
      cout << "commands" << endl;
      //list<string> commands;
      //pack.getAllCommands( commands );
   }
   if ( argc==2 )
   {
      bool listState;
      unsigned int iRegistry;
      if ( !get_list_registry_number( argv[1], iRegistry, listState ) )
      {
	 cerr << "ERROR: third argument should be sXXX if you want to list a state or rXXX if you want to list a raw registry" << endl;
	 return EXIT_FAILURE;
      }

      if ( iRegistry>=pack.nbStates() )
      {
	 cerr << "ERROR: index " << iRegistry << " out of range (the pack has " << pack.nbStates() << " states)" << endl;
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
   if ( argc!=4 ){ cerr << "not enough args" << endl; return EXIT_FAILURE; }
   
   const cElFilename packname( (string)(argv[0]) );
   const cElPath anchor( argv[3] );
   
   TracePack pack( packname, anchor );
   if ( !pack.load() )
   {
      cerr << "ERROR: unable to load [" << packname.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   
   const int iRegistry = atoi( argv[1] );
   if ( iRegistry<0 )
   {
      cerr << "ERROR: invalid state index : " << iRegistry << endl;
      return EXIT_FAILURE;
   }
   
   const cElFilename itemName( (string)(argv[2]) );
   
   if ( !pack.copyItemOnDisk( iRegistry, itemName ) )
   {
      cerr << "ERROR: unable to copy item [" << itemName.str_unix() << "] from [" << packname.str_unix() << "]:s" << iRegistry << " to directory [" << anchor.str_unix() << ']' << endl;
      return EXIT_FAILURE;
   }
   return EXIT_SUCCESS;
}

int setstate_func( int argc, char **argv )
{
   //setstate packname istate anchor
   
   if ( argc!=3 ){ cerr << "not enough args " << argc << " != " << 3 << endl; return EXIT_FAILURE; }
   
   const cElFilename packname( (string)(argv[0]) );
   if ( !packname.exists() )
   {
      cerr << "ERROR: pack [" << packname.str_unix() << "] does not exist" << endl;
      return EXIT_FAILURE;
   }
   
   const int iState_s = atoi( argv[1] );
   if ( iState_s<0 )
   {
      cerr << "ERROR: state index " << iState_s << " is invalid" << endl;
      return EXIT_FAILURE;
   }
   const unsigned int iState = (unsigned int)iState_s;
   
   const cElPath anchor( argv[2] );
   if ( !anchor.exists() && !anchor.create() )
   {
      cerr << "ERROR: cannot create directory [" << anchor.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   
   TracePack pack( packname, anchor );
   if ( !pack.load() )
   {
      cerr << "ERROR: loading a pack from file [" << packname.str_unix() << "] failed" << endl;
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

string generare_random_string( unsigned int i_strLength )
{
   // generate a random filename
   srand( time(NULL) );
   string randomName;
   char c;
   for ( unsigned int i=0; i<i_strLength; i++ )
   {
      c = (char)(rand()%26);
      if ( (rand()%2)==0 )
	 c += 65;
      else
	 c += 97;
      randomName.append( string(1,c) );
   }
   return randomName;
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
   
   if ( argc<2 ){ cerr << "not enough args " << argc << " != " << 2 << endl; return EXIT_FAILURE; }
    
   const cElFilename packname( (string)(argv[0]) );
   if ( packname.exists() )
   {
      cerr << "ERROR: pack [" << packname.str_unix() << "] already exist" << endl;
      return EXIT_FAILURE;
   }
     
   const cElFilename scriptname( (string)(argv[1]) );
   if ( !scriptname.exists() )
   {
      cerr << "ERROR: script [" << scriptname.str_unix() << "] does not exist" << endl;
      return EXIT_FAILURE;
   }
   
   cElPath anchor;
   anchor = cElPath( (string)(argv[2]) );
   
   if ( !anchor.exists() )
   {
      cerr << "ERROR: reference directory [" << anchor.str_unix() << "] does not exist" << endl;
      return EXIT_FAILURE;
   }
   dictionary["${SITE_PATH}"] = anchor.str_unix();
   cout << "--- using directory [" << anchor.str_unix() << "]" << endl;
   
   cout << "--- dictionary" << endl;
   map<string,string>::iterator itDico= dictionary.begin();
   while ( itDico!=dictionary.end() )
   {
      cout << itDico->first << " -> " << itDico->second << endl;
      itDico++;
   }
   cout << endl;
   
   list<cElCommand> commands;
   if ( !load_script( scriptname, commands ) )
   {
      cerr << "ERROR: script file [" << scriptname.str_unix() << "] cannot be loaded" << endl;
      return EXIT_FAILURE;
   }
   list<cElCommand>::iterator itCmd = commands.begin();
   while ( itCmd!=commands.end() )
   {
      string cmd = itCmd->str();
      if ( itCmd->replace( dictionary ) )
      {
	 cout << "command [" << cmd << "]" << endl;
	 cout << "becomes [" << itCmd->str() << "]" << endl;
      }
      itCmd++;
   }
   
   // create initial state
   TracePack pack( packname, anchor );
   pack.addState();
   
   itCmd = commands.begin();
   unsigned int iCmd = 0;
   while ( itCmd!=commands.end() )
   {
      cout << "command " << iCmd << " : " << itCmd->str() << endl;
      if ( !itCmd->system() )
      {
	 cerr << "ERROR: command " << iCmd << " = " << endl;
	 cerr << "[" << itCmd->str() << "]" << endl;
	 cerr << "failed." << endl;
	 return EXIT_FAILURE;
      }
      pack.addState();
      itCmd++; iCmd++;
   }
   pack.save();
   
   //if ( !anchor.re
   
   return EXIT_SUCCESS;
}

string normalizedCommand( char *i_command )
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

int main( int argc, char **argv )
{
   if ( argc<2 ) return help_func(0,NULL);

   string command = normalizedCommand( argv[1] );
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
	 cerr << "DEBUG_ERROR: some cElCommandToken have not been freed : " << cElCommandToken::__nb_created << " created, " << cElCommandToken::__nb_distroyed << " distroyed" << endl;
   #endif

   return help_func(0,NULL);
}
