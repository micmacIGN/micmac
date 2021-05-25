#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32
   #include <windows.h>
#elif __APPLE__
   #include <mach-o/dyld.h>
#else
   #include <unistd.h>
#endif

// !!  LE PATH PEUT ETRE TRES GRAND -CAS DE PATH GENERE AUTO, IL FAUDRAIT CHANGE CELA !!
// A COURT TERME 1024 => 100000
// #define PATH_BUFFER_SIZE 100000


// increase buffers' sizes of BUFFER_CHUNCK_SIZE when no specified size is specified
#define BUFFER_CHUNCK_SIZE 256
 
// a data type for a reallocating buffer (for strings here)
typedef struct{
	char *data;
	size_t size;
} buffer_t;

buffer_t g_buffer0 = { NULL, 0 },
         g_buffer1 = { NULL, 0 },
         g_command = { NULL, 0 };

void FataleError(const char * aMes)
{
    printf("Error: %s\n",aMes);
    printf("Fatal error : e-xit\n");
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
// 	ex: a buffer of size 256 becomes a buffer of size 1024 with the 256 first bytes unchanged, the other 768 bytes are undefined
// returns i_buffer->data
char * getBuffer( buffer_t *i_buffer, size_t i_size, int i_keepData )
{
   if ( i_size<=i_buffer->size ) return i_buffer->data;
   
   if ( i_keepData )
   {
      i_buffer->size = i_size;
      return ( i_buffer->data=realloc(i_buffer->data, i_size) );
   }
   
   freeBuffer( i_buffer );
   i_buffer->size = i_size;
   return ( i_buffer->data=(char*)malloc(i_size) );
}

char * getExecutableName( buffer_t *i_buffer )
{
	size_t retrievedSize = 0;

	#ifdef WIN32
	{
		// get full path of current executable
		retrievedSize = (size_t)GetModuleFileName(NULL, i_buffer->data, (DWORD)i_buffer->size );
	}
	#elif __APPLE__
	{
		uint32_t size = (uint32_t)i_buffer->size;
		if ( _NSGetExecutablePath(i_buffer->data, &size)==-1 ) _NSGetExecutablePath( getBuffer(i_buffer,size,0), &size);
		return i_buffer->data;
	}
	#else // Linux
		retrievedSize = (size_t)readlink( "/proc/self/exe", i_buffer->data, i_buffer->size );
	#endif
   
   // (retrived size) = (max size) may mean the path has been truncated
   // try again with a bigger buffer
   if (retrievedSize == i_buffer->size)
   {
      getBuffer(i_buffer, i_buffer->size + BUFFER_CHUNCK_SIZE, 0);
      return getExecutableName( i_buffer );
   }
   else
   {
      char * buffer = getBuffer(i_buffer, retrievedSize + 1, 1);
      buffer[retrievedSize] = '\0';
      return buffer;
   }
}

void str_append( buffer_t *i_buffer, const char *i_str )
{
	size_t cmd_len = strlen(i_buffer->data),
	       str_len = strlen(i_str),
	       needed_size = cmd_len + str_len + 1;
	char * buffer = getBuffer(i_buffer, needed_size, 1);
	memcpy(buffer + cmd_len, i_str, str_len);
	buffer[cmd_len+str_len] = '\0';
}

#ifdef WIN32
	// a '\' protects a '"' under windows
	// add a space to break the \" special sequence if an argument is a '\' ended directory

	// insert double quotes before, and space+double quote after the string
	// c:\titi\toto\ -> "c:\titi\toto\ "
	// returns a pointer to g_buffer1->data
	const char * add_double_quotes_and_space( const char *i_src, buffer_t *i_buffer )
	{
		size_t len = strlen(i_src);
		char *buffer = getBuffer(i_buffer, len + 4, 0);
		memcpy(buffer + 1, i_src, len);
		buffer[0] = buffer[len + 2] = '\"';
		buffer[len+1] = ' ';
		buffer[len+3] = '\0';
		return buffer;
	}
#endif

// insert double quotes before and after the string
// returns a pointer to g_buffer1->data
const char * add_double_quotes( const char *i_src, buffer_t *i_buffer )
{
	size_t len = strlen(i_src);
	char *buffer = getBuffer(i_buffer, len + 3, 0);
	memcpy(buffer + 1, i_src, len);
	buffer[0] = buffer[len + 1] = '\"';
	buffer[len + 2] = '\0';
	return buffer;
}

char * get_MM3D_name( buffer_t *i_buffer )
{
   char *it, *it2;
   
   getBuffer( i_buffer, BUFFER_CHUNCK_SIZE, 0 );
   getExecutableName( i_buffer );
   
   //printf( "executable name [%s]\n", i_buffer->data );
   
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
   
    //printf( "mm3d name [%s]\n", i_buffer->data );
    //printf( "sub command %d [%s]\n", subComand_len, o_subName );
   return i_buffer->data;
}

int has_space( const char *i_str )
{
	while (*i_str!='\0')
	{
		if ( *i_str==' ' ) return 1;
		i_str++;
	}
	return 0;
}

int ends_with_backslash( const char *i_str )
{
	size_t len = strlen(i_str);
	if (len < 1) return 0;
	return (i_str[len - 1] == '\\');
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
    
	str_append( &g_command, add_double_quotes( get_MM3D_name(&g_buffer0), &g_buffer1 ) );
	str_append( &g_command, " ");
    str_append( &g_command, aCom);
    
	for (aK=1 ; aK<argc ; aK++)
	{
		str_append( &g_command, " " );
		
		// every argument is quoted because it can be :
		// - a regular expression (and unices must not interpret it)
		// - a filename (and it must not be split by the command interpretter if there is a space in it)
		#ifdef _WIN32
			if ( ends_with_backslash(argv[aK]) )
				str_append( &g_command, add_double_quotes_and_space( argv[aK], &g_buffer0 ) );
			else
		#endif
		str_append( &g_command, add_double_quotes( argv[aK], &g_buffer0 ) );
	}
  

        #ifdef __TRACE_SYSTEM__
		// a utiliser avec precautions, genere des problemes a l'utilisation de MpDcraw
		//printf("%s calls [%s]\n", argv[0], g_command.data); fflush(stdout);
	#endif
        fflush(stdout);
  
	#ifdef _WIN32
		// an extra double quote on the whole line seem to be necessary for lines starting with a double quote
		return system( add_double_quotes( g_command.data, &g_buffer0 ) );
	#else
		return system( g_command.data );
	#endif
}

