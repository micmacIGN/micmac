#include "VersionedFileHeader.h"

#include <cstdlib>
#include <fstream>
#include <cstring>

using namespace std;

class Check
{
public:
   bool m_isGood;
   Check():m_isGood(true){}
   void operator ()( bool i_isGood, int i_level=0 )
   {
      cout << string(i_level, '\t') << (i_isGood?"succes":"echec") << endl;
      if ( !i_isGood ) m_isGood=false;
   }
};

// verifie la cohérence de g_versioned_headers_list's
// finit par {0,0,0,""} (à la position nbVersionedTypes)
// les deux premier champs ne sont pas égaux
// les deux premier champs sont l'inverse l'un de l'autre
// aucun nombre magique n'apparait deux fois
// aucun nombre magique n'est l'inverse d'un autre nombre magique
bool check_versioned_headers_list()
{   
   unsigned int i = 0;
   versioned_file_header_t *itHeader = g_versioned_headers_list+nbVersionedTypes();
   if ( itHeader->magic_number_LSBF!=0 ||
        itHeader->magic_number_MSBF!=0 ||
	itHeader->last_handled_version!=0 ||
	itHeader->name!="unknown" )
   {
      cerr << "incoherence entre VFH_Unknown et la fin de g_versioned_headers_list" << endl;
      return false;
   }
   
   bool res = true;
   i = 0;
   itHeader = g_versioned_headers_list;
   while ( itHeader->magic_number_MSBF!=0 )
   {
      if ( itHeader->magic_number_LSBF==itHeader->magic_number_MSBF )
      {
	 cerr << "entree " << i << " : LSBF = MSBF" << endl;
	 res = false;
      }
      
      uint32_t magic = itHeader->magic_number_LSBF,
               magic_inverse = magic;
      byte_inv_4( &magic_inverse );
      if ( magic_inverse!=itHeader->magic_number_MSBF )
      {
	 cerr << "entree " << i << " : LSBF != byte_inverse( MSBF )" << endl;
	 res = false;
      }
      
      versioned_file_header_t *itHeader2 = itHeader+1;
      unsigned int i2 = i+1;
      while ( itHeader2->magic_number_MSBF!=0 )
      {
	 if ( itHeader->magic_number_LSBF==itHeader2->magic_number_LSBF ||
	      itHeader->magic_number_LSBF==itHeader2->magic_number_MSBF )
	 {
	    cerr << "les entrees " << i << " et " << i2 << " ont le meme nombre magique, ou des nombres inverses" << endl;
	 }
	 itHeader2++; i2++;
      }
      
      itHeader++; i++;
   }
   return res;
}

void print_versioned_file_header_list()
{
   versioned_file_header_t *itHeader = g_versioned_headers_list;
   int i = 0;
   while ( itHeader->magic_number_MSBF!=0 )
   {
      cout << i << " : " << endl;
      cout << "\tnom                    : " << itHeader->name << endl;
      cout << "\tnombre magique L/M-SBF : " << itHeader->magic_number_LSBF << '/' << itHeader->magic_number_MSBF << endl;
      cout << "\tversion courante       : " << itHeader->last_handled_version << endl;
      itHeader++; i++;
   }
}

#define BUFFER_SIZE 1024

bool versioned_copy( const string &i_inFilename, const string &i_outFilename, VFH_Type i_type )
{   
   cout << "\tcree [" << i_inFilename << "], une copie de [" << i_outFilename << "] avec une en-tete de versionnement TracePack" << endl;
   
   ifstream fin( i_inFilename.c_str(), ios::binary );
   if ( !fin )
   {
      cerr << "\t\t__versioned_copy: impossible de lire [" << i_inFilename << "]" << endl;
      return false;
   }
   ofstream fout( i_outFilename.c_str(), ios::binary );
   if ( !fout )
   {
      cerr << "\t\tversioned_copy: impossible d'ecrire dans [" << i_outFilename << "]" << endl;
      return false;
   }
   
   VersionedFileHeader header(i_type);
   header.write( fout );
   char buffer[BUFFER_SIZE];
   while ( !fin.eof() )
   {
      fin.read( buffer, BUFFER_SIZE );
      fout.write( buffer, fin.gcount() );
   }
   fin.close();
   fout.close();
   
   cout << "\t\t[" << i_inFilename << "] est de type \"" << g_versioned_headers_list[versionedFileType( i_inFilename )].name << "\"" << endl;
   cout << "\t\t[" << i_outFilename << "] est de type \"" << g_versioned_headers_list[versionedFileType( i_outFilename )].name << "\"" << endl;
   
   return true;
}

bool compare_data( istream &io_istream1, istream &io_istream2 )
{
   char buffer1[BUFFER_SIZE],
        buffer2[BUFFER_SIZE];
   while ( !io_istream1.eof() && !io_istream2.eof() )
   {
      io_istream1.read( buffer1, BUFFER_SIZE );
      io_istream2.read( buffer2, BUFFER_SIZE );      
      if ( io_istream1.gcount()!=io_istream2.gcount() ) return false;
      if ( memcmp( buffer1, buffer2, (size_t)io_istream2.gcount() )!=0 ) return false;
   }
   return ( io_istream1.eof() && io_istream2.eof() );
}

void read_header( const string &i_filename, istream &io_istream )
{
   VersionedFileHeader header;
   VFH_Type type;
   if ( header.read_unknown( io_istream, type ) )
      cout << "\t\t[" << i_filename << "] est versionne (" << g_versioned_headers_list[type].name << ", v" << header.version() << ")" << endl;
   else
      cout << "\t\t[" << i_filename << "] n'est pas versionne" << endl;
}

bool compare_data( const string &i_filename1, const string &i_filename2 )
{
   cout << "\tcompare les donnees de  [" << i_filename1 << "] et [" << i_filename2 << "] sans type impose" << endl;   
   ifstream f1( i_filename1.c_str(), ios::binary );
   if ( !f1 )
   {
      cerr << "\t\tcompare_data: impossible de lire [" << i_filename1 << "]" << endl;
      return false;
   }
   ifstream f2( i_filename2.c_str(), ios::binary );
   if ( !f2 )
   {
      cerr << "\t\tcompare_data: impossible de lire [" << i_filename2 << "]" << endl;
      return false;
   }
   
   read_header( i_filename1, f1 );
   read_header( i_filename2, f2 );
   
   return compare_data( f1, f2 );
}

bool check_versioned_file_io( const string &i_inFilename )
{
   Check check;
   
   string outFilenameDigeo     = i_inFilename+".digeo.versioned";
   string outFilenameTracePack = i_inFilename+".tracepack.versioned";
 
   check( versioned_copy( i_inFilename, outFilenameDigeo, VFH_Digeo ), 1 );
   check( versioned_copy( i_inFilename, outFilenameTracePack, VFH_TracePack ), 1 );
   
   check( compare_data( i_inFilename, outFilenameDigeo ), 1 );
      
   return check.m_isGood;
}

int main( int argc, char **argv )
{
   Check check;
   cout << "processeur : " << (MSBF_PROCESSOR()?"big-endian":"little-endian") << endl;
   cout << endl;
   
   cout << "verifie la coherence de la liste des fichiers versionnes connus" << endl;
   check( check_versioned_headers_list() );
   cout << endl;
   
   print_versioned_file_header_list();
   
   // génère un nouveau nombre magique à utiliser   
   uint32_t magic_direct, magic_reverse;
   generate_new_magic_number( magic_direct, magic_reverse );
   cout << "\nnouveau nombre magique = " << magic_direct << '/' << magic_reverse << endl << endl;
   
   if ( argc==2 )
   {
      cout << "teste la lecture/ecriture d'un fichier versionne" << endl;
      check( check_versioned_file_io( string(argv[1]) ) );
      cout << endl;
   }
   
   return check.m_isGood;
}
