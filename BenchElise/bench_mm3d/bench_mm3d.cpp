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
int help_func( int, char ** );

command_t g_commands[] = {
   { "snapshot", snapshot_func },
   { "list", list_func },
   { "getfile", getfile_func },
   { "setstate", setstate_func },
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
      {
	 cerr << "ERROR: pack [" << packname.str_unix() << "] is not equal to its write+read copy" << endl;
	 return EXIT_FAILURE;
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

   return help_func(0,NULL);
}
