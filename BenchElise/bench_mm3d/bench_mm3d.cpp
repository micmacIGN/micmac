#include "StdAfx.h"

#include "TracePack.h"

using namespace std;

typedef struct
{
   string name;
   int (*func)( int, char ** );
} command_t;

int help_func( int, char ** );
int snapshot_func( int, char ** );
int difference_func( int, char ** );
int list_func( int, char ** );

command_t g_commands[] = {
   { "snapshot", snapshot_func },
   { "list", list_func },
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
   const cElPath directory( (string)(argv[1]) );
   const cElFilename filename( (string)(argv[0]) );
   TracePack pack;
   
   if ( pack.load( filename ) )
   {
      cout << "--- updating [" << filename.str_unix() << ']' << endl;
   }
   else
   {
      cout << "--- creating [" << filename.str_unix() << "] the " << pack.date() << " (UTC)" << endl;
   }
   
   cout << "snaping [" << directory.str_unix() << "] to [" << filename.str_unix() << "]" << endl;
   pack.addStateFromDirectory( directory );
      
   const unsigned int nbStates = pack.nbStates();
   cerr << nbStates << " state" << (nbStates>1?'s':'\0') << endl;
      
   if ( !pack.save( filename ) )
   {
      cerr << "ERROR: unable to save to [" << filename.str_unix() << "]" << endl;
      return EXIT_FAILURE;
   }
   
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
   
   TracePack pack;
   if ( !pack.load( filename ) )
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
	 cerr << "ERROR: third argument should be sXXX if you want to list a state or rXXX if you want to list a raw registry" << endl;

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
