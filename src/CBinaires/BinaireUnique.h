#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32
   #include <windows.h>
#elif __APPLE__
   #include <mach-o/dyld.h>
#else
   #include <unistd.h>
   #define ELISE_CAR_DIR  '/' 
#endif

// !!  LE PATH PEUT ETRE TRES GRAND -CAS DE PATH GENERE AUTO, IL FAUDRAIT CHANGE CELA !!
// A COURT TERME 1024 => 100000

#define PATH_BUFFER_SIZE 100000

// increase buffers' sizes of BUFFER_CHUNCK_SIZE when no specified size is specified
#define BUFFER_CHUNCK_SIZE 10

typedef struct{
   char         * data;
   unsigned int   size;
} buffer_t;

buffer_t g_buffer0 = { NULL, 0 },
         g_buffer1 = { NULL, 0 },
	 g_command = { NULL, 0 };

void FataleError(const char * aMes)
{
    printf("Error: %s\n",aMes);
    printf("Fatal error : exit\n");
    exit(-1);
}

void freeBuffer( buffer_t *i_buffer )
{
   if ( i_buffer->data!=NULL )
   {
      free(i_buffer->data);
      i_buffer->data = NULL;
      i_buffer->size = 0;
   }
}

// ensures i_buffer is at least i_size long
// if i_keepData is true, the beginning of buffer's content is unchanged
// returns i_buffer->data
char * getBuffer( buffer_t *i_buffer, unsigned int i_size, int i_keepData )
{
   if ( i_size<=i_buffer->size ) return i_buffer->data;
   
   if ( i_keepData )
   {
      i_buffer->size = i_size;
      return ( i_buffer->data=realloc(i_buffer->data,i_size) );
   }
   
   freeBuffer( i_buffer );
   i_buffer->size = i_size;
   return ( i_buffer->data=(char*)malloc(i_size) );
}

// insert backslahes before spaces
// returns a pointer to g_buffer1->data
const char * protect_spaces( const char *src, buffer_t *i_buffer )
{   
   // count space characters
   unsigned int length   = 0,
	        nbSpaces = 0;
   const char *itSrc = src;
   while ( *itSrc!='\0' )
   {
      if ( *itSrc==' ' ) nbSpaces++;
      length++;
      itSrc++;
   }
   
   char * const buffer = getBuffer( i_buffer, length+nbSpaces+1, 0 );
   itSrc = src;
   char *itDst = buffer;
   while ( *itSrc!='\0' )
   {
      if (*itSrc==' ')
      {
	 itDst[0] = '\\';
	 itDst[1] = ' ';
	 itDst += 2;
      }
      else
	 *itDst++ = *itSrc;
      itSrc++;
   }
   *itDst = '\0';
   
   return buffer;
}

char * getExecutableName( buffer_t *i_buffer )
{
   printf( "--> a getExecutableName %d\n", i_buffer->size );
   
   unsigned int retrievedSize = 0;
   #ifdef WIN32
      {
	 // get full path of current executable
	 retrievedSize = (unsigned int)GetModuleFileName(NULL, i_buffer->data, i_buffer->size );
      }
   #elif __APPLE__
      {
	 char *itChar;
	 uint32_t size = (uint32_t)i_buffer->size;
	 if ( _NSGetExecutablePath(i_buffer->data, &size)!=0 )
	    retrievedSize = i_buffer->size;
      }
   #else // Linux
	 retrievedSize = (unsigned int)readlink( "/proc/self/exe", i_buffer->data, i_buffer->size );
   #endif
   
   // (retrived size) = (max size) may mean the path has been truncated
   // try again with a bigger buffer
   if ( retrievedSize==i_buffer->size )
   {
      getBuffer( i_buffer, i_buffer->size+BUFFER_CHUNCK_SIZE, 0 );
      return getExecutableName( i_buffer );
   }
   else
   {
      char * buffer = getBuffer( i_buffer, retrievedSize+1, 1 );
      buffer[retrievedSize] = '\0';
      return buffer;
   }
}

void str_append( buffer_t *i_buffer, const char *i_str )
{
   unsigned int cmd_len = strlen(i_buffer->data),
	        str_len = strlen(i_str),
		needed_size = cmd_len+str_len+1;
   char * buffer = getBuffer( i_buffer, needed_size, 1 );
   memcpy( buffer+cmd_len, i_str, str_len );
   buffer[cmd_len+str_len] = '\0';
}

char * get_MM3D_name( buffer_t *i_buffer )
{
   char *it, *it2;
   
   getBuffer( i_buffer, BUFFER_CHUNCK_SIZE, 0 );
   getExecutableName( i_buffer );
   
   printf( "executable name [%s]\n", i_buffer->data );
   
   // find last '/' or '\'
   it = it2 = i_buffer->data;
   while ( *it!='\0' )
   {
      if ( (*it)=='\\' || (*it)=='/' )
	 it2 = it;
      it++;
   }
   *it2 = '\0';
   str_append( i_buffer, "/mm3d" );
   
   printf( "mm3d name [%s]\n", i_buffer->data );
   return i_buffer->data;
}

void str_copy( buffer_t *i_buffer, char *i_str )
{
   unsigned int len = strlen(i_str)+1;
   memcpy( getBuffer( i_buffer, len, 0 ), i_str, len );
}

int  BinaireUnique
      (
           const char * aCom,
           int argc,
           char ** argv
      )
{
   int aK;
   
   // initialize g_command with an empty string
   getBuffer( &g_command, BUFFER_CHUNCK_SIZE, 0);
   g_command.data[0] = '\0';
   
   str_append( &g_command, protect_spaces( get_MM3D_name(&g_buffer0), &g_buffer1 ) );

  for (aK=1 ; aK<argc ; aK++)
  {
      str_append( &g_command, " \"" );
      str_append( &g_command, argv[aK] );
      #if (WIN32)
	 // a '\' protects a '"' under windows
	 // add a space to break the "\\" special sequence if an argument is a '\' ended directory
	 if ( argv[aK][strlen( argv[aK] )-1]=='\\' )
	    str_append( &g_command, " \"" );
	 else
      #endif
      str_append( &g_command, "\"" );
  }
  
  printf("ComF =[%s]\n",g_command.data); fflush(stdout);
  return system(g_command.data);
}

