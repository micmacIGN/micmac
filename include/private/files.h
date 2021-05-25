/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#ifndef _ELISE_PRIVATE_FILES_H
#define _ELISE_PRIVATE_FILES_H

#ifndef S_ISREG
	inline int S_ISREG(int v) { return v&_S_IFREG;}
#endif

#ifndef S_ISDIR
	inline int S_ISDIR(int v){ return v&_S_IFDIR; }
#endif

  class cXmlExivEntry;
  class cXmlDataBase;

class cGlobXmlGen
{
    public :
       cGlobXmlGen();
       INT1   mPrec;  // Gere la precision dans l'ecriture des fichiers
};

class cElXMLTree;
class Fich_Im2d;
class cArgCreatXLMTree;

std::list<std::string>  ListFileMatch
                        (
                               const std::string & aDir,
                               const std::string & aPattern,
                               INT NivMax,
                               bool NameComplet = true // Si Oui inclu la dir dans les noms de fichier
                        );

std::list<std::string>  RegexListFileMatch
                        (
                               const std::string & aDir,
                               const std::string & aPattern,
                               INT NivMax,
                               bool NameComplet = true // Si Oui inclu la dir dans les noms de fichier a matcher sur le pattern
                        );


bool MatchPattern
     (
         const std::string & aName,
         const std::string & aPattern
     );



class ElResParseDir
{
     public :
         enum
         {
             level_max = 20
         };
         ElResParseDir
         (
             const char * Name,
             bool   Isdir,
             int    Level
         );
         void show() const;

         const char * name () const;
         bool  is_dir() const;
         INT   level () const;
         const char * sub (INT lev) const;

     private :

          const char * _name;
          bool   _is_dir;
          INT    _level;
          const char  * _sub[level_max+1];
};

class ElActionParseDir
{
     public :
        virtual void act(const ElResParseDir &) = 0;
     virtual ~ElActionParseDir() {}
};


void ElParseDir
     (
          const char * ndir,
          ElActionParseDir &,
          INT    NivMax = ElResParseDir::level_max 
     );



class  ELISE_fp
{

       public :

         FILE * FP() ; // DEBUG, sinon a eviter absolument , court-circuite tout ...
         typedef enum
         {
               READ        = 0,
               WRITE       = 1,
               READ_WRITE  = 2
         } mode_open;

         typedef enum
         {
               sbegin        = 0,
               scurrent      = 1,
               send          = 2
         } mode_seek;

         typedef enum
         {
              eBinTjs,
              eTxtTjs,
              eTxtOnPremierLigne
         } eModeBinTxt;

         void SetFormatFloat(const std::string & aFormat);
         void SetFormatDouble(const std::string & aFormat);
         void PutLine();
         void PutCommentaire(const std::string & aComment);
         void PasserCommentaire();
         // void UnGetc(int car);
         // bool GetGivenCar(int car);


         static void if_not_exist_create_0(const char *,struct stat *); 
         static bool exist_file(const char *); 
         static bool exist_file(const std::string &);
	 static bool MkDirSvp(const std::string &);
	 static void MkDir(const std::string &);
         static void MkDirRec(const std::string &  aName );
         static bool IsDirectory(const std::string &  aName );
         static void AssertIsDirectory(const std::string &  aName );
		 static bool copy_file( const std::string i_src, const std::string i_dst, bool i_overwrite );
         static int  file_length( const std::string &i_filename );
         
         static bool lastModificationDate(const std::string &i_filename, cElDate &o_date ); // returns if the date could be retrieved

	 static void RmFileIfExist(const std::string &);  // evite les erreurs qd fichier inexistant
	 static void RmFile(const std::string &);
	 static void MvFile(const std::string & aFile,const std::string & aDest);
	 static void CpFile(const std::string & aFile,const std::string & aDest);
	 static void PurgeDir(const std::string &,bool WithRmDir=false);
	 static void RmDir(const std::string &);
	 static void PurgeDirGen(const std::string &,bool Recurs);
	 static void PurgeDirRecursif(const std::string &);
         static int  CmpFiles(const std::string & aF1,const std::string & aF2); // return -1, 0 ou 1 
      
         ~ELISE_fp();
         ELISE_fp(eModeBinTxt ModeBin=eTxtOnPremierLigne);
         ELISE_fp(const char *,mode_open=READ,bool svp = false,eModeBinTxt  =eTxtOnPremierLigne);

         U_INT1 read_U_INT1();
         U_INT2 read_U_INT2();
         U_INT4 read_U_INT4();
         U_INT8 read_U_INT8();

         INT2   read_INT2();
         INT4   read_INT4();
         _INT8  read_INT8();
         tFileOffset read_FileOffset4();
         tFileOffset read_FileOffset8();
         REAL4   read_REAL4();
         REAL8   read_REAL8();

         const std::string & NameFile() const {return  mNameFile;}


         void CKK_write_FileOffset4(tFileOffset);
         void CKK_write_FileOffset8(tFileOffset);
         void write_U_INT1(INT);
         void write_U_INT2(INT);
         void write_U_INT4(U_INT4);
         void write_U_INT8(U_INT8);

         void write_INT4(INT);
         void write_REAL4(REAL4);
         void write_REAL8(REAL8);

         void write_line(const std::string &); // Pendant de fgets
         void write(const std::string &);
         void write(const bool &);
         void write(const Pt2di &);
         void write(const Pt2dr &);
         void write(const Pt2df &);
         void write(const Seg2d &);
         void write(const std::list<Seg2d> &);
         void write(const std::list<std::string> &);
         void write(const REAL8 &);
         void write(const REAL4 &);
         void write(const INT4 &);
	 void write(const std::vector<REAL8> &);
	 void write(const std::vector<int> &);
	 void write(const std::vector<Pt2di> &);
	 void write(const Polynome2dReal &);
	 void write(const PolynomialEpipolaireCoordinate &);
	 void write(const CpleEpipolaireCoord &);
         void write(const RImGrid &);
         void write(const ElCplePtsHomologues &);
         void write(const cNupletPtsHomologues &);

         void write (const cXmlExivEntry &);
         void write (const cXmlDataBase &);

         void write(const Pt3df &);
         void write(const Pt3dr &);
	 void write(const ElMatrix<REAL> &);
	 void write(const ElRotation3D &);


	 Pt3dr read(Pt3dr *);
	 Pt3df read(Pt3df *);
	 ElMatrix<REAL> read(ElMatrix<REAL> *);
	 ElRotation3D   read(ElRotation3D *);

          // En gal, les ptr ne sont pas utilise (sert pour les tpl)
	 Polynome2dReal   read(Polynome2dReal *);
	 REAL8            read(REAL8 *);
	 INT4            read(INT4 *);
         Pt2dr            read(Pt2dr *);
         Pt2df            read(Pt2df *);
         Pt2di            read(Pt2di *);
         Seg2d            read(Seg2d *);
         std::string      read(std::string *);
         bool             read(bool *);
	 PolynomialEpipolaireCoordinate  read
		  (PolynomialEpipolaireCoordinate *);
	 CpleEpipolaireCoord * read(CpleEpipolaireCoord *);
         std::list<Seg2d> read(std::list<Seg2d> *);
         std::vector<REAL8> read(std::vector<REAL8> *);
         std::vector<int> read(std::vector<int> *);
         std::vector<Pt2di> read(std::vector<Pt2di> *);
	 ElCplePtsHomologues read(ElCplePtsHomologues *);
	 cNupletPtsHomologues read(cNupletPtsHomologues *);




         U_INT1 lsb_read_U_INT1();
         U_INT2 lsb_read_U_INT2();
         INT4   lsb_read_INT4();
         U_INT1 msb_read_U_INT1();
         U_INT2 msb_read_U_INT2();
         INT4   msb_read_INT4();



         void lsb_write_U_INT2(INT);

         void set_byte_ordered(bool);
         bool byte_ordered() const { return _byte_ordered;}


         bool open(const char *,mode_open,bool svp = false);
         bool ropen(const char * name,bool svp = false)
         {
              return open(name,READ,svp);
         }
         bool close(bool svp = false);
         bool closed() const;
         bool seek(tRelFileOffset,mode_seek,bool svp = false);

         bool seek_cur(tRelFileOffset offset,bool svp = false)
         {
              return seek(offset,scurrent,svp);
         }
         bool seek_begin(tRelFileOffset offset,bool svp = false)
         {
              return seek(offset,sbegin,svp);
         }
         bool seek_end(tRelFileOffset offset,bool svp = false)
         {
              return seek(offset,send,svp);
         }

         void set_last_act_read(bool read_mode)
         {
              // Required, because on some Unix system, read-write
              // file, must receive a no-op fseek (used for its
              // synchronization sides effect) when swaping from
              // write to read

              if (_last_act_read != read_mode)
                 seek_cur(0);
              _last_act_read = read_mode;
         }


         void read(void *,tFileOffset size,tFileOffset nb,const char * format=0);
         void write(const void *,tFileOffset size,tFileOffset nb);
         void str_write(const char *);
         void write_dummy(tFileOffset  nb);  // write nb byte (when you just need
                                       // to extend size of file
         tFileOffset tell();
         INT  fgetc(); // name it fgetc because getc is a macro 

		  
		  // get the next line in the file
          bool fgets( std::string &s, bool & endof );
          //bool fgets(char *,INT sz_buf,bool & endof,bool svp = false); TEST_OVERFLOW

          // Renvoie false en fin de fichier, renvoie un ligne ascii "standard" (Tab-> space ..)
          // Buf a 2000
          char * std_fgets();

#if BUG_CPP_Fclose
         typedef enum
         {
             eof = -1
         } symbolics;
#else
         typedef enum
         {
             eof = EOF
         } symbolics;
#endif

       private :

          static const int code_mope_seek[3];

          void init(eModeBinTxt ModeBin);
          bool _byte_ordered;
          bool _last_act_read;
          bool         mIsModeBin;
          eModeBinTxt  mModeBinTxt;
          std::string mNameFile;
#if BUG_CPP_Fclose
          int   _fd;
          static const int code_mope_open[3];
#else
          FILE * _fp;
          std::string  mFormatFloat;
          std::string  mFormatDouble;
          char         mCarCom;
          static const char * name_mope_open[3];
#endif
         static void InterneMkDirRec(const  std::string  & aName );
};

template <class Type> void  WritePtr(ELISE_fp & aFile,tFileOffset aNb,const Type * aPtr);
template <class Type> void  ReadPtr(ELISE_fp & aFile,tFileOffset aNb,Type * aPtr);

bool FileStrictPlusRecent(const std::string & aF1,const std::string & aF2);

#if (ELISE_unix || ELISE_MacOs)
	#include <dirent.h>
	class cElDirectory
	{
	public :
		cElDirectory(const std::string& aName) :
		   mDir (opendir(aName.c_str()))
		   {
		   }
		   ~cElDirectory()
		   {
			   if (mDir !=0)
				   closedir(mDir);
		   }

		   bool IsDirectory()
		   {
			   return mDir != 0;
		   }
		   const char * GetNextName()
		   {
			   struct dirent * cont_dir = readdir(mDir);
			   return (cont_dir != 0) ? cont_dir->d_name : 0;
		   }
	private :
		DIR  * mDir;
	};
#elif (ELISE_windows)
	#undef INT
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include "shlobj.h"
	class cElDirectory
	{
	public :
		cElDirectory(const std::string& aName):
				mFirst(true)
		   {
			   if ( ELISE_fp::exist_file(aName) )
					mPattern = aName;
			   else
			   {
					char lastChar = *aName.rbegin();
					if ( lastChar!='\\' && lastChar!='/' )
						mPattern = aName+"/*";
					else
						mPattern = aName+'*';
			   }
			   mHandleFind = ::FindFirstFile (mPattern.c_str(), &mWFD);
		   }
		   ~cElDirectory()  { ::FindClose(mHandleFind); }

		   bool IsDirectory()
		   {
			   return    (mHandleFind != INVALID_HANDLE_VALUE)
				   && (mWFD.dwFileAttributes &FILE_ATTRIBUTE_DIRECTORY);
		   }

		   const char * GetNextName()
		   {
			   if (mFirst)
			   {
				   mFirst=false;
				   return mWFD.cFileName;
			   }
			   if (FindNextFile (mHandleFind, &mWFD))
			   {
				   return mWFD.cFileName;
			   }
			   return 0;

		   }


	private :
		std::string        mPattern;
		bool               mFirst;
		HANDLE             mHandleFind;
		WIN32_FIND_DATA	   mWFD;

	};
#endif

INT sizeofile (const char * nom);
typedef long int  FILE_offset;

FILE  * FopenNN
        (
	    const std::string & aName,
	    const std::string & aMode,
	    const std::string & aMes
	);


/*
     Can simply be a file with getchar(), can be something more complicated;
    as a gif flow where the integer sum ar to be skipped
*/
class Flux_Of_Byte  : public Mcheck
{
      public :
          virtual U_INT1 Getc()       = 0;
          virtual void Putc(U_INT1) = 0;
          virtual ~Flux_Of_Byte();
          virtual tFileOffset  tell() = 0;

      private :
};

// a small intergface class to make packed flux, looks like 
// Flux_Of_Byte;

class UnPacked_FOB  : public Flux_Of_Byte
{
      public :
          UnPacked_FOB(class Packed_Flux_Of_Byte *,bool to_flush);
          virtual ~UnPacked_FOB();
          virtual U_INT1 Getc();
          virtual void Putc(U_INT1);
          virtual tFileOffset  tell();

      private :
          class Packed_Flux_Of_Byte * _packed;
          bool                        _to_flush;
};



class Packed_Flux_Of_Byte : public Mcheck
{
      public :
          // return the number of bits really readen

          virtual ~Packed_Flux_Of_Byte();
          virtual bool      compressed() const = 0;

          virtual tFileOffset Read(U_INT1 * res,tFileOffset nb) = 0;

          // def value : fatal error 
          virtual tFileOffset Write(const U_INT1 * res,tFileOffset nb);
          virtual tRelFileOffset Rseek(tRelFileOffset nb) ;  // only forward, nb relative
                                       // to curent position

          virtual void  AseekFp(tFileOffset nb) ; 
             //   !!!!!!!!
             // absolute seek of File * in byte, not in number of el
             // def value => Erreur fatale
             // rather sad (break the abstraction) but needed for
             // some things like run over  padding 


          inline INT sz_el() const {return _sz_el;}

          virtual tFileOffset         tell()=0;   // debug pupose

           
      protected :
          Packed_Flux_Of_Byte(INT sz_el);
          INT     _sz_el;
      private :
};

class Std_Packed_Flux_Of_Byte : public Packed_Flux_Of_Byte
{
      public :

          // return the number of bits really readen

          virtual void  AseekFp(tFileOffset nb) ; // absolute seek of File *
          virtual void  Aseek(tFileOffset nb) ; // absolute seek / offset 0, in szel
          Std_Packed_Flux_Of_Byte
          (
                 const char * name,
                 INT sz_el,
                 tFileOffset off_0,
                 ELISE_fp::mode_open
          );

          ELISE_fp & fp();
          virtual tFileOffset         tell();   // debug pupose
          virtual ~Std_Packed_Flux_Of_Byte();


      private :
          virtual bool      compressed()  const;

          virtual tFileOffset Read(U_INT1 * res,tFileOffset  nb) ;
          virtual tFileOffset Write(const U_INT1 * res,tFileOffset  nb);
          virtual tRelFileOffset Rseek(tRelFileOffset  nb) ;  // forward or backward, nb relative
                                       // to curent position, in szel

          ELISE_fp   _fp;
          tFileOffset        _offset_0;
};


class Mem_Packed_Flux_Of_Byte : public Packed_Flux_Of_Byte
{
      public :

          // return the number of bits really readen

          Mem_Packed_Flux_Of_Byte (INT sz_init,INT sz_el);

          virtual ~Mem_Packed_Flux_Of_Byte();

          tFileOffset nbbyte () const {return _nb* tFileOffset(sz_el());}
          INT operator [] (tFileOffset k)
          {
              El_Internal.ElAssert
              (
                  (k.BasicLLO()>=0) && (k<_nb*tFileOffset(sz_el())),
                  EEM0 << "Out of range in Mem_Packed_Flux_Of_Byte"
              );
              return _data[k.BasicLLO()];
          }
          void reset();

      private :
          virtual tFileOffset         tell();   // debug pupose
          virtual tFileOffset Write(const U_INT1 * res,tFileOffset nb);
          virtual bool      compressed()  const;

          // For now, I do not need it, so : not implanted 
          // But can, of course, be added

          virtual tFileOffset Read(U_INT1 * res,tFileOffset nb) ;

          tFileOffset        _nb;
          tFileOffset        _sz;
          U_INT1 *   _data;
};



       // Only for 1,2,4 bits.

class  BitsPacked_PFOB : public Packed_Flux_Of_Byte
{
    public :


       virtual ~BitsPacked_PFOB();
       BitsPacked_PFOB
       (
             Packed_Flux_Of_Byte *,
             INT nbb,
             bool msbf,
             bool read_mode,
             INT  nb_el
       );
       virtual tFileOffset         tell();   // debug pupose

    protected :

       virtual bool      compressed()  const;

       INT set_kieme_val (INT byte,INT val,INT k)
       {
           return _tbb.set_kieme_val(byte,val,k);
       }

       inline INT kieme_val (INT byte,INT k)
       {
              return _tbb.kieme_val(byte,k);
       }

       


       Packed_Flux_Of_Byte * _pfob;
       INT _nb_pb; // nb per byte
       INT _v_max; 
       bool _read_mode;
       bool _pf_compr;
       const Tabul_Bits_Gen & _tbb;

       U_INT1 _v_buf;
       INT    _i_buf;  // index of val buffered in v_buf
       tFileOffset    _nb_el;

       void flush_write();

       virtual void  AseekFp(tFileOffset nb) ; 
       virtual tFileOffset Read(U_INT1 * res,tFileOffset nb) ;
       virtual tFileOffset Write(const U_INT1 * res,tFileOffset nb) ;
       virtual tRelFileOffset Rseek(tRelFileOffset nb) ;  
       
    private :
       
       tFileOffset _Read(U_INT1 * res,tFileOffset nb) ;
       tFileOffset _Write(const U_INT1 * res,tFileOffset nb) ;
       tRelFileOffset _Rseek(tRelFileOffset nb) ;  

};




      // Flux of integers with variable number of bits

class Flux_Of_VarLI :  public Mcheck
{
      public :
          virtual INT   nexti(INT nb_bits) = 0;
          virtual void  reset() = 0; 
          virtual ~Flux_Of_VarLI();

          static Flux_Of_VarLI * new_flx(Flux_Of_Byte *,bool msbf,bool flx_flush);
          tFileOffset  tell();

      protected :
          Flux_Of_VarLI(Flux_Of_Byte *,bool flx_flush);
          Flux_Of_Byte * _flx_byte;
          bool           _flx_flush;
    
      private :
};


      // Most Significant bit first variation
      // of Flux_Of_VarLI
      // Assume the folloziw packing
      // (with 5 bits integers) a b c ..
      //
      //       a0 | a1 | a2 | a3 | a4 | b0 | b1 | b2
      //       b3 | b4 | c0 | c1 | c2 | c3 | c4 | d0



class MSBitFirst_Flux_Of_VarLI  :  public Flux_Of_VarLI
{
      public :
          virtual INT nexti(INT nb);
          MSBitFirst_Flux_Of_VarLI(Flux_Of_Byte *,bool flx_flush);

          INT     _last_bit_read;
          U_INT1  _last_char_read;
          virtual void  reset();

};


      // Less Significant bit first variation
      // of Flux_Of_VarLI
      // Assume the folloziw packing
      // (with 5 bits integers) a b c ..
      //
      //       b2 | b3 | b4 | a0 | a1 | a2 | a3 | a4
      //       d4 | c0 | c1 | c2 | c3 | c4 | b0 | b1


class LSBitFirst_Flux_Of_VarLI  :  public Flux_Of_VarLI
{
      public :
          virtual INT nexti(INT nb);
          LSBitFirst_Flux_Of_VarLI(Flux_Of_Byte *,bool flx_flush);

          INT     _last_bit_read;
          U_INT1  _last_char_read;
          virtual void  reset();
};

class Flux_OutVarLI  :  public Mcheck
{
      public :
          virtual INT puti(INT nb,INT nbb) = 0;
          static Flux_OutVarLI * new_flx(Flux_Of_Byte *,bool msbf,bool flx_flush);
          virtual  ~Flux_OutVarLI();
          virtual void  reset() = 0;

          tFileOffset  tell();
          INT kth();

      protected :
          Flux_OutVarLI(Flux_Of_Byte *,bool flx_flush);

          Flux_Of_Byte * _flx;
          INT     _bit_to_write;
          U_INT1  _char_to_write;
          bool    _flx_flush;

     private :
};

class MSBF_Flux_OutVarLI  :  public Flux_OutVarLI
{
      public :
          virtual INT puti(INT val,INT nb);
          MSBF_Flux_OutVarLI(Flux_Of_Byte *,bool flx_flush);

          virtual void  reset();
};

class LSBF_Flux_OutVarLI  :  public Flux_OutVarLI
{
      public :
          virtual INT puti(INT val,INT nb);
          LSBF_Flux_OutVarLI(Flux_Of_Byte *,bool flx_flush);

          virtual void  reset();
};




      // The flow of decompressed value of LZW compressed flow

void test_lzw(char * ch);
class LZW_Protocols
{
      public :
            typedef enum {gif,tif} mode;
};




class Packed_LZW_Decompr_Flow :  public Packed_Flux_Of_Byte
{
   public :


        virtual ~Packed_LZW_Decompr_Flow();
        Packed_LZW_Decompr_Flow
        (
              Flux_Of_Byte *,          // compressed flow
              bool read,
              bool msbf,               // is it a most sign bit first flow
              LZW_Protocols::mode,     // fixes some option of LZW protocol
              INT nb_bit_init          // number of bits of initial LZW tables
                                       // pass 8 with tiff is required
        );

         virtual tFileOffset Read(U_INT1 * res,tFileOffset nb);
         virtual tFileOffset Write(const U_INT1 * res,tFileOffset nb);
         void Write(const INT * res,tFileOffset nb); // convert to U_INT1 *
         virtual tRelFileOffset Rseek(tRelFileOffset nb);
         void    assert_end_code(); 
         void    reset();

         virtual tFileOffset  tell();

   private :
        void init();
       virtual bool      compressed() const ;
        U_INT1  *            _buf;
        tFileOffset                        _nb_buffered;
        tFileOffset                        _deb_buffered;
        Flux_Of_VarLI   *          _flxi;
        Flux_OutVarLI   *          _flxo;
        class LZW_decoder *        _decoder;
        bool                       _read;
};


class Pack_Bits_Flow :  public Packed_Flux_Of_Byte
{
   public :


        virtual ~Pack_Bits_Flow();
        Pack_Bits_Flow
        (
              Packed_Flux_Of_Byte *,          // compressed flow
              bool read,
              INT  tx
        );

         virtual tFileOffset Read(U_INT1 * res,tFileOffset nb);
         virtual tFileOffset Write(const U_INT1 * res,tFileOffset nb);
         void    reset();

         virtual tFileOffset  tell();

   private :
       virtual bool      compressed() const ;

       Packed_Flux_Of_Byte *    _flx;
       bool                     _read;
       U_INT1      *            _buf;
       INT                      _tx;
       INT                      _n;
};

template <class Type> class cTplHuffmanTree;



class Huffman_FOB_Codec : public Mcheck
{
      public :
         ~Huffman_FOB_Codec();
      protected :

         Huffman_FOB_Codec
         (
               Packed_Flux_Of_Byte*,
               bool    read,
               bool    msbf,
               bool    flush_flx
         );

         inline INT getbit();
         INT geti(INT nbb);

         const cTplHuffmanTree<INT>  *    get(const cTplHuffmanTree<INT>  *);
         inline void put(const cTplHuffmanTree<INT>  * ht);
         void reset();
         bool                  _read;

         inline void put(INT val,INT nbb);

         void show(); // debug


      private :


         Packed_Flux_Of_Byte * _flx;
         Flux_OutVarLI       * _flx_varli;
         INT                   _kth;
         U_INT1                _v_buf;
         bool                  _msbf;
         bool                  _flush_flx;




};

class Huff_Ccitt_1D_Codec : public Huffman_FOB_Codec 
{
      public :

         Huff_Ccitt_1D_Codec
         (
               Packed_Flux_Of_Byte*,
               bool    read,
               bool    msbf,
               bool    flush_flx
         );

         void read(U_INT1 * res,INT nb_tot);
         void write(const U_INT1 * res,INT nb_tot);

      protected :


         int   get_length(INT coul);
         void  put_length(INT l,INT coul);


         typedef enum
         {
               ucomp_mod = -3000,
               eofb     = -2000,
               eol      = -1000,
               pass_mod =  100,
               hor_mod  =  200,
               huf_black = 888
         }     codes;

      private :

         void  put_length_partial(INT l,const class HuffmanCodec * h);

          const class HuffmanCodec *  _hw;  // white run lenght Huff tree
          const class HuffmanCodec *  _hb;  // black ...

};

class  Huff_Ccitt_2D_T6 : public Huff_Ccitt_1D_Codec
{
       public :
           Huff_Ccitt_2D_T6
           (
               Packed_Flux_Of_Byte*,
               bool    read,
               bool    msbf,
               bool    flush_flx,
               INT     sz_buf
           );

           void new_block(INT tx);
           void end_block(bool phys_end_block);

           void read(U_INT1 * res);
           void write(const U_INT1 * val);


          virtual ~Huff_Ccitt_2D_T6();
       protected :

          void uncomp_line();

          INT            _tx;
          INT            _coul;
          INT           _a0;
          INT           _a1;
          INT           _a2;
          INT           _b1;
          INT           _b2;
          bool          _eofb;

          U_INT1 *      _prec;
          U_INT1 *      _cur;

          void  calc_b1();
          void  calc_b2();
          void  calc_a1();
          void  calc_a2();

          const class HuffmanCodec *  _hvert;
          const class HuffmanCodec *  _huc;
          const cTplHuffmanTree<INT>   *  _ht_pass;
          const cTplHuffmanTree<INT>   *  _ht_horz;
          const cTplHuffmanTree<INT>   *  _ht_eofb;
};


class MPD_CCIT_T6 : public Huff_Ccitt_2D_T6
{

      public :
          void write(const U_INT1 * vals);
          void read(U_INT1 * vals);


          MPD_CCIT_T6
          (
               Packed_Flux_Of_Byte*    flx,
               bool                    read,
               bool                    msbf,
               bool                    flush_flx,
               INT                     sz_buf,
               INT                     nbb
          );
          virtual ~MPD_CCIT_T6();
      private :

          U_INT1 * _vals;
          inline INT end_pl_gray(INT a);
          inline INT end_pl_pure_black(INT a);
          inline INT end_pl_white(INT a);

          void put_length_gray(INT l);
          void put_plage_gray(INT a1,INT a2);
          INT get_length_gray();
          INT get_plage_gray(INT a,bool last);

          INT      _nbb;
          INT      _vmax;  // (1<<nbb) -1
          U_INT1 * _bin;

          const class HuffmanCodec * _hmpd;
          const cTplHuffmanTree<INT>   * _ht_huf_bl;

        // (! VISUAL)  static const INT max_l_gr = 4;
		  enum
		  {
			  max_l_gr = 4 
		  };

          INT _line; // debug
};



class Tile_F2d : public Mcheck
{
   friend class Fich_Im2d;

   public :
      Tile_F2d(Packed_Flux_Of_Byte *); // eventually 0
      virtual ~Tile_F2d();


   protected :

           // x0,x1,y0,y1 : relative to tile's coord
      
          // --------  def value : N fread with a sufficient local buffer 

                    virtual void seek_in_line(Fich_Im2d *,INT x0,INT x1);


          // --------  def value : n call to seek_in_line

                    virtual void seek_pack_line
                                 (Fich_Im2d *,INT y0,INT y1,bool read_mode);


          // --------  def value :  fread of sz_el
     
                    virtual void read_seg
                           (class Fich_Im2d *,void * buf,INT x0,INT x1);
                    virtual void write_seg
                           (class Fich_Im2d *,void * buf,INT x0,INT x1);

          // --------  def value : do nothing;  
          //           is sent each time a new line appears in the same tile
          //           example of use : Tiff differential predictor

                    virtual  void r_new_line(Fich_Im2d *,INT y);
                    virtual  void w_new_line(Fich_Im2d *,INT y);

                    virtual  void r_end_line(Fich_Im2d *,INT y);
                    virtual  void w_end_line(Fich_Im2d *,INT y);


          // -------- this message is send the fisrt time the tiles is used. 
          //          This allow to economize some  memory allocation because the
          //          tiles can "decide" to alloc the bufferization space only when needed
     
                     // def value init file on with name and seek to offset

                    virtual  void r_use_this_tile(class Fich_Im2d *);
                    virtual  void w_use_this_tile(class Fich_Im2d *);



          // -------- This message is sent when the tiles change; def : do nothing

                      virtual  void r_new_tile(class Fich_Im2d *) ;
                      virtual  void w_new_tile(class Fich_Im2d *) ;

          // -------- This message is sent when the tiles ends; def : do nothing

                      virtual  void r_end_tile(class Fich_Im2d *) ;
                      virtual  void w_end_tile(class Fich_Im2d *) ;


         // -------- flush to free the memory or close file

                     // close _fp if opened, call inst_flush, free

                     //  def : do nothing


         Packed_Flux_Of_Byte  * _pfob;
                              //  open _fp (for example in use_this_tile)

          INT  _n_tile;      //  number of tile : 0,1 ....
          INT  _sz_tile_log;     //  size of tile  : fich->_sz_til.x except for last tile
          INT  _sz_tile_phys;     //  size of tile  : fich->_sz_til.x except for last tile
          INT  _last_til_Y;       //  numero en Y de la derniere dalle lue
          INT  _last_x;
          INT  _last_y;

          static const INT NO_LAST_TIL_Y;
        
};


template <class Type> class Fonc_Fich_Im2d;


class Fich_Im2d : public Mcheck
{
      friend class Tile_F2d;
      friend class Fonc_Fich_Im2d<INT>;
      friend class Fonc_Fich_Im2d<REAL>;
      friend class Out_Fich_Im2d;

      public :

           inline bool integral_type() const {return _integral_type;}
           inline INT * tab_or() {return  _tab_or;}
           inline INT * tab_sz() {return  _tab_sz;}
           void    init_tile(Tile_F2d *,INT kth,INT padding,bool clip_last);

          inline INT dim_out() const{ return _dim_out;}

      protected :

          void SetByteInversed();

          virtual ~Fich_Im2d();
          Fich_Im2d
          (
              Flux_Pts_Computed *     flx,
              char *                  usr_buf,
              Pt2di                   sz_file,
              Pt2di                   sz_tiles,
              INT                     sz_el,
              INT                     dim_out,
              bool                    integral,
              bool                    compressed,
              const char *            name
          );

          inline void assert_not_wc(bool read_mode) const;

          void read_write_buf(const RLE_Pack_Of_Pts * pack,bool read_mode);
          virtual void input_transfere(Std_Pack_Of_Pts_Gen *) = 0;
          virtual void output_transfere(const Std_Pack_Of_Pts_Gen *) = 0;

          virtual void post_traite(Std_Pack_Of_Pts_Gen *);

          virtual const Pack_Of_Pts *
                  pre_traite
                  (
                         const Pack_Of_Pts * values,
                         Pack_Of_Pts *       empty_buf,
                         Pack_Of_Pts *       buf
                  );


          INT      _sz_el;    // in general : number of channel * sizeof channel elt 
                              //  for exemple, with a 16 bits RGB => = 2;


          INT      _dim_out;  // number of channel
          Pt2di    _sz_file;
          Pt2di    _sz_til;
          INT   _sztx;  // sz_til.x

          INT _tab_sz[2];
          INT _tab_or[2];

          INT            _nb_tiles;
          Tile_F2d  **   _tiles;  // This is inherited class job to intialize each _tiles[i]
                                  // by init_tile(Tile_F2d *,INT kth)
          Tprov_char *   _tprov_name;
          char *         _name;
          char *          _buf;  // rather a void*, but need some arithmetics on it
          bool            _usr_buf;
          bool            _integral_type;
          bool            _compressed;
          bool            _byte_inversed;




      private   :
};



class Std_Bitm_Fich_Im_2d : public Fich_Im2d
{
      public :

          typedef void (* r_special_transf)
                       (   Std_Pack_Of_Pts_Gen *,
                           const void *buf
                       );

          typedef void (* w_special_transf)
                       (   const Std_Pack_Of_Pts_Gen *,
                           void *buf
                       );

          virtual ~Std_Bitm_Fich_Im_2d();

          Std_Bitm_Fich_Im_2d
          (
              Flux_Pts_Computed * flx,
              Pt2di           sz_file,
              Pt2di           sz_tiles,
              INT             dim_out,
              const char *    name,
              GenIm           gi,
              bool            compressed,
              INT             sz_el_spec = -1,
              r_special_transf  = 0,
              w_special_transf  = 0
          );


           GenIm           _gi;
           DataGenIm *     _bim;

           // _spec_transf est utilisee pour les format ou un simple
           // stripping est insuffisant pour passer du pacquet de valeurs
           // a la representation binaire des donnees;
           //
           //  par exemple : TGA 16 bits, TGA 32 bits,

           r_special_transf  _r_spec_transf;
           w_special_transf  _w_spec_transf;

          virtual void input_transfere(Std_Pack_Of_Pts_Gen *);
          virtual void output_transfere(const Std_Pack_Of_Pts_Gen *);
};



Fonc_Num_Computed * fonc_num_std_f2d
                    (
                         const Arg_Fonc_Num_Comp &,
                         Fich_Im2d *,
                         bool      with_def_val,
                         REAL      def_val
                    );

Output_Computed * out_std_f2d
                  (
                         const Arg_Output_Comp & arg,
                         Fich_Im2d *            f2d
                  );


void PackBitsUCompr
     (
          Packed_Flux_Of_Byte * pfob,
          U_INT1 * res,
          INT nb_tot
     );
INT PackBitsCompr
     (
          Packed_Flux_Of_Byte * pfob,
          const U_INT1 * line,
          INT nb_tot
     );


void PackBitsUCompr_B2
     (
          Packed_Flux_Of_Byte * pfob,
          U_INT1 * res,
          INT nb_tot
     );
INT PackBitsCompr_B2
     (
          Packed_Flux_Of_Byte * pfob,
          const U_INT1 * line,
          INT nb_tot
     );



class ElDataGenFileIm :  public  RC_Object
{
      public :

         friend class ElGenFileIm;

      protected :
          ElDataGenFileIm();
          virtual   ~ElDataGenFileIm();

          void init
               (
                  int          dim,
                  const int *  sz,
                  INT          nb_channel,
                  bool         signedtype,
                  bool         integral,
                  int          nbbits,
                  const int *  sz_tile,
                  bool         compressed
               );

      private  :

          // caracteristique logique :
          int        _dim;
          int *      _sz;
          INT        _nb_channel;

         
          // caracteristique de la taille de representation
          // des elements :
          bool       _signedtype;
          bool       _integral;
          int        _nbbits;

          // carateristique d'organisation du fichier
          int *      _sz_tile;
          bool       _compressed;

          virtual   Fonc_Num in()      = 0;
          virtual   Fonc_Num in(REAL) = 0;
          virtual   Output out()     = 0;
};

/*   RANGE-CODE, a C++ interface to Michael Schindler's code.
     For detail see :

  +   (c) Michael Schindler
      1997, 1998
      http://www.compressconsult.com/ or http://eiunix.tuwien.ac.at/~michael
      michael@compressconsult.com        michael@eiunix.tuwien.ac.at

  +   GPL

  +    "Range encoding is based on an article by G.N.N. Martin, submitted
        March 1979 and presented on the Video & Data Recording Conference,
        Southampton, July 24-27, 1979. "
*/                     


class  Martin_Schindler_RCODE
{
      public :

        typedef U_INT4 code_value;
        typedef U_INT4 freq;

        enum
        {
             CODE_BITS  = 32,
             SHIFT_BITS =  CODE_BITS -9,
             EXTRA_BITS =  ((CODE_BITS-2) % 8 + 1)
        };
        static const U_INT4 Top_value;
        static const U_INT4 Bottom_value;

    protected :

        U_INT4 low;       // low end of interval
        U_INT4 range;     // length of interval
        U_INT4 help;      // bytes_to_follow resp. intermediate value
        unsigned char buffer;   // buffer for input/output
        //  the following is used only when encoding
         U_INT4 bytecount;     // counter for outputed bytes
};



class MS_RANGE_ENCODER :
          public Martin_Schindler_RCODE
{
      public :

        MS_RANGE_ENCODER(Flux_OutVarLI * flxo) :
          _flxo (flxo)
        {}

      /***********************************/
      /*    ENCODER                      */
      /***********************************/

          /* Start the encoder                                         */
          /* c is written as first byte in the datastream (header,...) */

        void start_encoding(U_INT1 c);

           /* Encode a symbol using frequencies                         */
           /* sy_f is the interval length (frequency of the symbol)     */
           /* lt_f is the lower end (frequency sum of < symbols)        */
           /* tot_f is the total interval length (total frequency sum)  */
           /* or (a lot faster): tot_f = 1<<shift                       */

        void encode_freq(freq sy_f, freq lt_f, freq tot_f );
        void encode_shift(freq sy_f, freq lt_f, freq shift );

           /* Encode a byte/short without modelling                     */
           /* b,s is the data to be encoded                             */

         void encode_byte(freq b);
         void encode_short(freq b);

             /* Finish encoding                                           */
          void done_encoding();

      private :

        Flux_OutVarLI   * _flxo;
        void enc_normalize();
        void outbyte(INT x);
};                                                  
                                                                        

class MS_RANGE_DECODER :
          public Martin_Schindler_RCODE
{
      public :
          MS_RANGE_DECODER(Flux_Of_VarLI * flxi) :
              _flxi (flxi)
          {}

         U_INT1 start_decoding();
         void dec_normalize();
         freq decode_culfreq(freq tot_f);
         freq decode_culshift(freq tot_f);
         void decode_update(freq sy_f,freq lt_f,freq tot_f);

         unsigned char  decode_byte();
         unsigned short decode_short();
         void done_decoding();

      private :
          int inbyte();
          Flux_Of_VarLI * _flxi;
};


class  cMS_SimpleArithmEncoder 
{
     public :
         cMS_SimpleArithmEncoder
         (
              const std::vector<REAL> & aVProbas ,
              INT               aNbBits,
              Flux_OutVarLI *,
	      char  aV0
	      
          );

	 void PushCode(INT aCode);
         const std::vector<INT> &  Cumuls() const;
         const std::vector<INT> &  Freqs() const;
	 INT   Tot () const;
	 void  Done();
        
     private :

         MS_RANGE_ENCODER  mEnc;
         INT               mNbBits;
         INT               mTot;
         INT               mNbVals;
         std::vector<INT>  mFreqs;
         std::vector<INT>  mCumuls;
};

class  cMS_SimpleArithmDecoder 
{
       public :
           cMS_SimpleArithmDecoder
           (
                 const std::vector<INT> &  Cumuls,
                 Flux_Of_VarLI *
           );

	   INT   Dec();
	   void  Done();
	   U_INT1  V0();
       private :
           MS_RANGE_DECODER     mDec;
	   INT                  mNbVals;
	   INT                  mP2;
	   INT                  mNbBits;
	   std::vector<U_INT1>  mVDecod;
           std::vector<INT>     mFreqs;
           std::vector<INT>     mCumuls;
	   U_INT1               mV0;
};



class cArgCreatXLMTree
{
    public :
         cArgCreatXLMTree(const std::string & aNF,bool aModifTree,bool aModifDico);
         void Add2EntryDic(cElXMLTree *,const std::string &);
         cElXMLTree * ReferencedVal(const std::string &);

         void AddRefs(const std::string & aTag,const std::string & aFile);
         ~cArgCreatXLMTree();

          std::string mNF;
          bool  ModifTree() const;
          bool  ModifDico() const;
          void SetDico(const std::string & aKey,std::string  aVal,bool IsMMCALL);
          void DoSubst(std::string & aStr);
          void DoSubst(std::string & aStr,bool ForcSubst);
    private :

          cArgCreatXLMTree (const cArgCreatXLMTree &) ; // N.I.
          std::map<std::string,cElXMLTree *> mDico;
          std::list<std::string> mAddedFiles;   // Pour  ne pas les mettre plusieurs fois
          std::list<cElXMLTree *> mAddedTree;   // Pour  ne pas les mettre plusieurs fois

          bool                               mModifTree;
          bool                               mModifDico;
          std::map<std::string,std::string>  mDicSubst;
          std::set<std::string>              mSymbMMCall;
};



class cElXMLFileIn
{
	public :
		cElXMLFileIn(const std::string &);
		~cElXMLFileIn();
 
                // Interface entre nouvelle (cElXMLTree) et ancienne bibliotheque
                void PutTree (cElXMLTree *);

                void PutMonome
                     (
                        const Monome2dReal & aMonome,
                        const double & aCoeff
                     );
                void PutPoly 
                     (
                         const Polynome2dReal &,
                         const std::string &
                     );
		void PutElComposHomographie
                    (
                         const cElComposHomographie &,
                         const std::string &
                    );
                void PutElHomographie
                     (
                           const cElHomographie &,
                           const std::string &
                     );
		void PutCamGen(const CamStenope &);
		void PutDist(const ElDistRadiale_PolynImpair & aDist);
		void PutCam(const cCamStenopeDistRadPol & aCam);
		void PutString(const std::string &,const std::string & Tag);
		void PutInt(const int &,const std::string & Tag);
		void PutDouble(const double &,const std::string & Tag);
		void PutGrid(const PtImGrid &,const std::string &);
		void SensorPutDbleGrid
                     (
		          Pt2di aSzIm,
		          bool XMLAutonome, // Si true les donnees binaires sont incluse dans le XML
                          cDbleGrid &,
                          const char * ThomFile = 0,
                          const char * aNameXMMLCapteur = 0,
                          ElDistRadiale_PolynImpair * = 0,
                          Pt2dr * aPP =0,
                          double * aFoc=0
                     );
		void PutDbleGrid(bool XMLAutonome,const cDbleGrid &,const std::string & = "doublegrid");

		void PutPt2di(const Pt2di &,const std::string & = "pt2di");
		void PutPt2dr(const Pt2dr &,const std::string & = "pt2d");
		void PutPt3dr(const Pt3dr &,const std::string & = "pt3d");
		void PutCpleHom(const ElCplePtsHomologues &,const std::string & = "CpleHom");
		void PutPackHom(const ElPackHomologue &,const std::string & = "ListeCpleHom");

		void PutTabInt(const std::vector<INT> &,const std::string & );

		class cTag{
			public :
			   cTag(cElXMLFileIn & aFile,const std::string &,bool SimpleTag=false);
			   ~cTag();
			   void NoOp();
			private :
			   cElXMLFileIn &      mFile;
			   std::string  mName;
			   bool  mSimpleTag;
		};
	private :
		friend class cTag;
		cElXMLFileIn(const cElXMLFileIn &);

		void PutIncr();
		void PutTagBegin(const std::string &,bool SimpleTag=false);
		void PutTagEnd(const std::string &,bool SimpleTag=false);

		FILE *      mFp;
		INT         mCurIncr;
		std::string mStrIncr;
};

typedef enum
{
	eXMLOpen,
	eXMLClose,
	eXMLStd,
	eXMLEOF,
	eXMLOpenClose

} eElXMLKindToken;

class cElXMLAttr
{
     public :
        std::string mSymb;
        std::string mVal;
};

class cVirtStream
{
   public :
       
      virtual int my_getc() =0 ;
      virtual int my_eof() = 0;
      virtual void my_ungetc(int)=0;

       virtual const char * Ending(); // Permet pour les string-file de reprendre 
                                      // la lecture la ou elle s'est arretee

      virtual void fread(void *dest,int aNbOct);

      bool IsFilePredef() const ;
      bool IsFileSpec() const ;

      static cVirtStream * StdOpen(const std::string &);
      virtual ~cVirtStream();
       cVirtStream(const std::string & aName,bool isPreDef,bool IsSpec);
      const std::string & Name();
      static cVirtStream * VStreamFromCharPtr(const char* aCharPtr);
      static cVirtStream * VStreamFromIsStream(std::istringstream &);
   private :
      std::string mName;
      bool         mIsPredef;
      bool         mIsSpec;
};



class cElXMLToken
{
	public :
		cElXMLToken(cArgCreatXLMTree &,cVirtStream *,bool & aUseSubst);
		const std::string &  Val() const;
		eElXMLKindToken  Kind() const;
                const std::list<cElXMLAttr> & Attrs() const;
	        void Show(FILE *,int);
                    
                void GetSequenceBinaire(cVirtStream * aFp);
	private :
		std::string            mVal;
		eElXMLKindToken        mKind;
                std::list<cElXMLAttr>  mAttrs;
};

typedef enum
{
    eXMLTop,
    eXMLBranche,
    eXMLFeuille ,
    eXMLClone     // D'une autre Feuille
} eElXMLKindTree;

eElXMLKindTree MergeForComp(eElXMLKindTree);

class cElXMLTreeFilter
{
     public :
         virtual bool Select(const cElXMLTree &) const;
     virtual ~cElXMLTreeFilter() {}
};


class cElXMLTree
{
	public :
		class Functor
		{
		public:
			virtual void operator ()( cElXMLTree &i_node ) = 0;
		};
		
         cGlobXmlGen mGXml;
         
         std::list<cElXMLTree *>  Interprete();

	 cElXMLTree * ReTagThis(const std::string & aNameTag);
         bool IsFeuille() const;
         bool IsBranche() const;

	 void Debug(const std::string &);

          void AddFils(cElXMLTree *);
          const std::string & ValAttr(const std::string &) const;
          const std::string & ValAttr(const std::string &,
                                      const std::string & Def) const;

          // Return true si valeur nouvelle
          bool SetAttr(const std::string & aSymb,const std::string & aVal);
          
          // 
          const std::string & StdValAttr(const std::string &,bool &) const;
          bool HasAttr(const std::string &) const;

           static cElXMLTree * ValueNode(const std::string & aNameTag,const std::string & aVal);
           static cElXMLTree * MakePereOf(const std::string & aNameTag,cElXMLTree * aFils);


           cElXMLTree * Clone();
	   cElXMLTree( cElXMLTree * aPere,
                       const std::string & aVal, 
                       eElXMLKindTree  aKind
                      );
	   cElXMLTree(const std::string & ,cArgCreatXLMTree * Arg = 0,bool DoFileInclu = true);
           ~cElXMLTree();
           // si isTermOnLine met les terminaux sur la meme ligne
           void ShowAscendance(FILE * aFile);
           void ShowOpeningTag(FILE * aFile);

           void Show(const std::string & mIncr,FILE *,INT aCpt,INT aLevelMin,bool isTermOnLine,const cElXMLTreeFilter &);
           void Show(const std::string & mIncr,FILE *,INT aCpt,INT aLevelMin,bool isTermOnLine);

            void StdShow(const std::string & aNameFile);

            cElXMLTree * Get(const std::string & ,int aDepthMax=1000000);
            cElXMLTree * GetUnique(const std::string &,bool ByAttr=false );
            cElXMLTree * GetOneOrZero(const std::string & );
	    const  std::list<cElXMLTree *>   & Fils() const;
	    std::list<cElXMLTree *>   & Fils();

            cElXMLTree * GetUniqueFils();

            std::list<cElXMLTree *> GetAll(const std::string & ,bool ByAttr=false,int aDepthMax=1000000);
            std::string & GetUniqueVal() ;
            int     GetUniqueValInt();
            double  GetUniqueValDouble();
            int     GetUniqueValInt(const std::string & aName);
            double  GetUniqueValDouble(const std::string & aName);

            RImGrid *GetRImGrid( const std::string & aTagNameFile,
                                 const std::string& aDir
                     );

            PtImGrid GetPtImGrid(const std::string& aDir);

            Monome2dReal  GetElMonome2D(double & aCoeff,double anAmpl);
            Polynome2dReal GetPolynome2D();
            cElHomographie GetElHomographie();
            cElComposHomographie GetElComposHomographie();
            Pt2dr     GetPt2dr();
            Pt2di     GetPt2di();
            Box2di    GetDicaRect();
            ElCplePtsHomologues  GetCpleHomologues();
            ElPackHomologue      GetPackHomologues(const std::string & = "ListeCpleHom");
            const std::list<cElXMLAttr> & Attrs() const;
            bool  TopVerifMatch (cElXMLTree* aTSpecif,const std::string& aName,bool SVP=false);
            bool  TopVerifMatch 
	          (
		       const std::string& aNameObj,
		       cElXMLTree* aTSpecif,
		       const std::string& aNameType,
		       bool ByAttr=false,
		       bool SVP=false
		  );

            void ModifLC(int argc,char ** argv,cElXMLTree * aSpecif);

            void GenCppGlob(const std::string & aNameFile,
                        const std::string & aNameSpace);
            void StdGenCppGlob
                 (
                      const std::string & aNameCpp,
                      const std::string & aNameH,
                      const std::string & aNameSpace
                 );


          bool  HasFilsPorteeGlob(const std::string &);
          const std::string & ValTag() const;
          // Renvoit le contenu du Fils qui doit etre unique et branche
          const std::string & Contenu() const;
          std::string & NCContenu() ;
	  bool  IsVide() const;
           INT Profondeur () const;

			// walk through the tree with a breadth-first strategy and execute i_functor on all nodes
			void breadthFirstFunction( Functor &i_functor );

	private :
          bool  VerifMatch (cElXMLTree* aTSpecifi,bool SVP=false);
          void GenOneCppNameSpace (FILE * aFileCpp,FILE* aFileH,std::string aDefaultNameSpace);
          void Verbatim(FILE * aFileCpp,FILE * aFileH);

          void GenEnum(FILE * aFileCpp,FILE* aFileH);
          void ModifMangling(cMajickChek &);
          void GenCppClass
	      (
	         FILE * aFileCpp,
		 FILE* aFileH,
                 const std::list<std::string> & aLTypeLoc,
		 int aProf
	      );
          void GenAccessor
               (
	            bool        Recurs,
                    cElXMLTree * anAnc,
                    int aProf,
                    FILE* aFileH,
                    std::list<cElXMLTree *> & ,
                    bool  isFileH
               );
          void GenStdXmlCall
              (
                  const std::string & aPrefix,
                  const std::string & aNamePere,
                  const std::string & aNameObj,
                  FILE * aFileCpp
              );
          std::string NameOfClass();
          std::string NameImplemOfClass();
          bool IsDeltaPrec();


          void ModifLC(char * anArg,cElXMLTree * aSpecif);
          void VerifCreation();
          static bool ValidNumberOfPattern(const std::string &,int aN);
          const std::string & ValAttr(const std::string &,
                                      const std::string * Def) const;
           void GetAll(const std::string & ,std::list<cElXMLTree *> &,bool byAttr,int aDepthMax=1000000);

	   cElXMLTree
           (  bool DoFileFinclu,
              bool aUseSubst,
              cVirtStream *,
              const cElXMLToken &,
              cElXMLTree * aPere,cArgCreatXLMTree &
           );


           cElXMLTree *              Missmatch
                                     (
                                          cElXMLTree* aT2,
                                          bool isSpecOn2,
                                          std::string & aMes
                                     );

          void ExpendRef
	       (
	             cArgCreatXLMTree &anArg,
		     const std::string & TagExpend,
		     const std::string & TagExpendFile,
		     bool MustBeEmpty
               );

           
	   std::string               mValTag;
	   std::list<cElXMLTree *>   mFils;
           cElXMLTree *              mPere;
           std::list<cElXMLAttr>     mAttrs;
           eElXMLKindTree            mKind;
};


class XmlXml
{
    public :
         XmlXml();
         cElXMLTree * mTree;
};


void xml_init(XmlXml    &,cElXMLTree * aTree);

void xml_init(bool           &,cElXMLTree * aTree);
void xml_init(double         &,cElXMLTree * aTree);
void xml_init(int            &,cElXMLTree * aTree);
void xml_init(Box2dr         &,cElXMLTree * aTree);
void xml_init(Box2di         &,cElXMLTree * aTree);
void xml_init(Pt2dr         &,cElXMLTree * aTree);
void xml_init(Pt3dr         &,cElXMLTree * aTree);
void xml_init(Pt3di         &,cElXMLTree * aTree);
void xml_init(Pt2di         &,cElXMLTree * aTree);
void xml_init(std::string    &,cElXMLTree * aTree);
void xml_init(std::vector<double>   &,cElXMLTree * aTree);
void xml_init(std::vector<int>   &,cElXMLTree * aTree);
void xml_init(std::vector<std::string>   &,cElXMLTree * aTree);
void xml_init(cElRegex_Ptr &,cElXMLTree * aTree);

void xml_init(cCpleString &,cElXMLTree * aTree);
void xml_init(cMonomXY &,cElXMLTree * aTree);

void xml_init(BoolSubst &,cElXMLTree * aTree);
void xml_init(IntSubst &,cElXMLTree * aTree);
void xml_init(DoubleSubst &,cElXMLTree * aTree);
void xml_init(Pt2diSubst &,cElXMLTree * aTree);
void xml_init(Pt2drSubst &,cElXMLTree * aTree);





#define TypeForDump(aType)\
void BinaryDumpInFile(ELISE_fp &,const aType &);\
void BinaryUnDumpFromFile(aType &,ELISE_fp &);\
std::string Mangling(aType *);

TypeForDump(bool)
TypeForDump(double)
TypeForDump(int)
TypeForDump(Box2dr)
TypeForDump(Box2di)
TypeForDump(Pt2dr)
TypeForDump(Pt2di)
TypeForDump(std::string)
TypeForDump(std::vector<double>)
TypeForDump(std::vector<int>)
TypeForDump(std::vector<std::string>)
TypeForDump(Pt3dr)
TypeForDump(Pt3di)
TypeForDump(cElRegex_Ptr)
TypeForDump(cCpleString)
TypeForDump(cMonomXY)
TypeForDump(IntSubst)
TypeForDump(BoolSubst)
TypeForDump(DoubleSubst)
TypeForDump(Pt2diSubst)
TypeForDump(Pt2drSubst)
TypeForDump(XmlXml)

/*
*/

template <class T1,class T2> void BinaryDumpInFile(ELISE_fp &,const Im2D<T1,T2> &      anObj);
template <class T1,class T2> void BinaryUnDumpFromFile(Im2D<T1,T2> &,ELISE_fp &);
template <class T1,class T2> std::string Mangling(Im2D<T1,T2> *);



template <class T1,class T2> void xml_init( Im2D<T1,T2>  & anIm,cElXMLTree * aTree);


template <class Type> 
void xml_init(cTplValGesInit<Type> & aVGI,cElXMLTree * aTree)
{
    if (aTree==0) return;
    Type aVal;
    xml_init(aVal,aTree);
    aVGI.SetVal(aVal);
}

template <class Type> 
void xml_init(cTplValGesInit<Type> & aVGI,cElXMLTree * aTree,const Type & aVDef)
{
    if (aTree==0) 
    {
       aVGI.SetVal(aVDef);
    }
    else
    {
       Type aVal;
       xml_init(aVal,aTree);
       aVGI.SetVal(aVal);
    }
}

template <class TypeCont> 
void xml_init_cont(TypeCont & aContObj,const std::list<cElXMLTree *> & aLTree)
{
    for 
    (
        std::list<cElXMLTree *>::const_iterator itTree = aLTree.begin();
        itTree != aLTree.end();
        itTree++
    )
    {
        typename TypeCont::value_type anObj;
        xml_init(anObj,*itTree);
        aContObj.push_back(anObj);
    }
}

template <class Type> 
void xml_init(std::list<Type> & aLObj,const std::list<cElXMLTree *> & aLTree)
{
    xml_init_cont(aLObj,aLTree);
}

template <class Type> 
void xml_init(std::vector<Type> & aVObj,const std::list<cElXMLTree *> & aLTree)
{
    xml_init_cont(aVObj,aLTree);
}

template <class TKey,class TVal> 
void xml_init
    (
        std::map<TKey,TVal> & aMObj,
        const std::list<cElXMLTree *> & aLTree,
        const std::string & aKGV
   )
{
    for 
    (
        std::list<cElXMLTree *>::const_iterator itTree = aLTree.begin();
        itTree != aLTree.end();
        itTree++
    )
    {
        TVal anObj;
        xml_init(anObj,*itTree);
        

        cElXMLTree * aTreeKey = (*itTree)->GetUnique(aKGV);
        TKey aKey;
        xml_init(aKey,aTreeKey);

        if (aMObj.find(aKey) != aMObj.end())
        {
            ELISE_ASSERT(false,"Multiple Key in XML Map");
        }
        aMObj[aKey] = anObj;
    }
}



template <class Type> void BinDumpObj(const Type & anObj,const std::string & aFile)
{
    ELISE_fp aFPOut(aFile.c_str(),ELISE_fp::WRITE);
    // NumHgRev doesn't work with the new Git version
    //BinaryDumpInFile(aFPOut,NumHgRev());
    BinaryDumpInFile(aFPOut,0);
    BinaryDumpInFile(aFPOut,Mangling((Type*)0));
    BinaryDumpInFile(aFPOut,anObj);
    aFPOut.close();
}




bool IsFileDmp(const std::string &);

extern std::vector<std::string> VCurXmlFile;
template <class Type> Type StdGetObjFromFile_WithLC
                      (
		          int argc,
		          char ** argv,
                          const std::string & aNameFileObj,
                          const std::string & aNameFileSpecif,
                          const std::string & aNameTagObj,
                          const std::string & aNameTagType,
			  bool ByAttr = false,
                          cArgCreatXLMTree * anArg = 0,
                          const char *  aNameSauv = 0
                      )
{
   if (IsFileDmp(aNameFileObj))
   {
        Type aRes;
        BinUndumpObj(aRes,aNameFileObj,aNameTagObj,aNameTagType);
        return aRes;
   }

   cElXMLTree aFullTreeParam(aNameFileObj,anArg);
   cElXMLTree * aTreeParam = aFullTreeParam.GetUnique(aNameTagObj,ByAttr);
   cElXMLTree aTreeSpec(aNameFileSpecif);

   aTreeParam->TopVerifMatch(aNameTagObj,&aTreeSpec,aNameTagType,ByAttr); //"ParamChantierPhotogram");

   if (argv)
      aTreeParam->ModifLC(argc,argv,&aTreeSpec);

   if (aNameSauv)
       aTreeParam->StdShow(aNameSauv);

   Type aRes;
   xml_init(aRes,aTreeParam);

   return aRes;
}








template <class Type> Type StdGetObjFromFile
                      (
                          const std::string & aNameFileObj,
                          const std::string & aNameFileSpecif,
                          const std::string & aNameTagObj,
                          const std::string & aNameTagType,
			  bool  ByAttr = false,
                          cArgCreatXLMTree * anArg = 0
                      )
{
    return StdGetObjFromFile_WithLC<Type>
           (
	       0,0,
	       aNameFileObj,
	       aNameFileSpecif,
	       aNameTagObj,
	       aNameTagType,
	       ByAttr,
               anArg
	   );

}

template <class Type> Type * GetImRemanenteFromFile(const std::string & aName)
{
   static std::map<std::string,Type *> aDic;
   Type * aRes = aDic[aName];

   if (aRes != 0) return aRes;

   Tiff_Im aTF(aName.c_str());
   Pt2di aSz = aTF.sz();

   aRes = new Type(aSz.x,aSz.y);
   ELISE_COPY(aTF.all_pts(),aTF.in(),aRes->out());
   aDic[aName] = aRes;
   return aRes;
}


template <class Type> Type * RemanentStdGetObjFromFile
                      (
                          const std::string & aNameFileObj,
                          const std::string & aNameFileSpecif,
                          const std::string & aNameTagObj,
                          const std::string & aNameTagType
                      )
{
    static std::map<std::string,Type *> aDic;
    Type * aRes = aDic[aNameFileObj];
    if (aRes!=0) return aRes;

    aRes = new Type(StdGetObjFromFile<Type>(aNameFileObj,aNameFileSpecif,aNameTagObj,aNameTagType));
    aDic[aNameFileObj] = aRes;
    return aRes;
}


template <class Type> Type * OptionalGetObjFromFile_WithLC
                      (
		          int argc,
		          char ** argv,
                          const std::string & aNameFileObj,
                          const std::string & aNameFileSpecif,
                          const std::string & aNameTagObj,
                          const std::string & aNameTagType
                      )
{
   if (! ELISE_fp::exist_file(aNameFileObj)) 
   {
       return 0;
   }
   if (IsFileDmp(aNameFileObj))
   {
        Type aRes;
        BinUndumpObj(aRes,aNameFileObj,aNameTagObj,aNameTagType);
        return new Type(aRes);
   }
   cElXMLTree aFullTreeParam(aNameFileObj);
   cElXMLTree * aTreeParam = aFullTreeParam.GetOneOrZero(aNameTagObj);
   if (! aTreeParam) return 0;
   cElXMLTree aTreeSpec(aNameFileSpecif);

   aTreeParam->TopVerifMatch(aNameTagObj,&aTreeSpec,aNameTagType,false); //"ParamChantierPhotogram");

   if (argv)
      aTreeParam->ModifLC(argc,argv,&aTreeSpec);


   Type *  aRes = new Type;
   xml_init(*aRes,aTreeParam);

   return aRes;
}



std::string  GetValLC(int,char **,const std::string & aKey, const std::string & aDef);

cElXMLTree * ToXMLTree(const Pt3dr &      anObj);  // Pour CPP_GrapheHom.cpp
cElXMLTree * ToXMLTree(const std::string & aNameTag,const bool   &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const double &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const int    &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Box2dr &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Box2di &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt2di &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt2dr &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const std::string & anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const std::vector<double> & anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const std::vector<int> & anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const std::vector<std::string> & anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt3dr &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt3di &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const cElRegex_Ptr &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const XmlXml &      anObj);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const cCpleString   &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const cMonomXY   &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const IntSubst   &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const BoolSubst   &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const DoubleSubst   &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt2diSubst   &      anObj);
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt2drSubst   &      anObj);
template <class T1,class T2>
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<T1,T2> &      anObj);


        


template <class Type>
void MakeFileXML(const Type & anObj,const std::string & aName,const std::string & aTagEnglob="")
{
   if (IsFileDmp(aName))
   {
       BinDumpObj(anObj,aName);
       return;
   }

   cElXMLTree * aTree = ToXMLTree(anObj);
   // FILE * aFp = Fopen(aName.c_str(),"w");
   // Fclose(aFp);
   if (aTagEnglob!="")
   {
      aTree = cElXMLTree::MakePereOf(aTagEnglob,aTree);
   }
   aTree->StdShow(aName);
   delete aTree;
}

template <class Type>
void AddFileXML(const Type & anObj,const std::string & aName)
{
   cElXMLTree * aTree = new cElXMLTree(aName);
    cElXMLTree *  aFils = aTree->GetUniqueFils();  // Pour virer le noeud de fichier
   aFils->AddFils(ToXMLTree(anObj));
   aFils->StdShow(aName);
   delete aTree;
}


template <class Type> const typename Type::value_type *  
                     GetOnlyUseIt(const Type & aCont)
{
   const typename Type::value_type * aRes = 0;
   for ( typename Type::const_iterator it=aCont.begin(); it!=aCont.end(); it++)
   {
      if (it->UseIt())
      {
          ELISE_ASSERT(aRes==0,"Multiple UseIt in GetOnlyUseIt");
	  aRes = & (*it);
      }
   }

   return aRes;
}

void Stringify
     (
         const std::string &aNameInput,
         const std::string &aNameOutput,
         const std::string &aNameString
     );
void XML_StdStringify (const std::string &aNameInput);
void StdXMl2CppAndString(const std::string &aNameInput);


void AddEntryStringifie(const std::string &,const char ** aTab,bool formal);


double PolonaiseInverse(const std::string & aStr);

void  XMLPushContext(const cGlobXmlGen & aGXml);
void  XMLPopContext(const cGlobXmlGen & aGXml);
 

// ===============  ParamChantierPhotogram.xml

#define StdGetFromPCP(aStr,aObj)\
StdGetObjFromFile<c##aObj>\
    (\
    aStr,\
        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),\
        #aObj ,\
        #aObj \
     )

#define OptStdGetFromPCP(aStr,aObj)\
OptionalGetObjFromFile_WithLC<c##aObj>\
    (\
        0,0,\
        aStr,\
        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),\
        #aObj ,\
        #aObj \
     )

// ===============  SuperposImage.xml

#define StdGetFromSI(aStr,aObj)\
StdGetObjFromFile<c##aObj>\
    (\
    aStr,\
        StdGetFileXMLSpec("SuperposImage.xml"),\
        #aObj ,\
        #aObj \
     )

#define OptStdGetFromSI(aStr,aObj)\
OptionalGetObjFromFile_WithLC<c##aObj>\
    (\
        0,0,\
        aStr,\
        StdGetFileXMLSpec("SuperposImage.xml"),\
        #aObj ,\
        #aObj \
     )

// ===============  ParamMICMAC.xml

#define StdGetFromMM(aStr,aObj)\
StdGetObjFromFile<c##aObj>\
    (\
    aStr,\
        StdGetFileXMLSpec("ParamMICMAC.xml"),\
        #aObj ,\
        #aObj \
     )

#define OptStdGetFromMM(aStr,aObj)\
OptionalGetObjFromFile_WithLC<c##aObj>\
    (\
        0,0,\
        aStr,\
        StdGetFileXMLSpec("ParamMICMAC.xml"),\
        #aObj ,\
        #aObj \
     )


// ===============  ParamApero.xml


#define StdGetFromAp(aStr,aObj)\
StdGetObjFromFile<c##aObj>\
    (\
    aStr,\
        StdGetFileXMLSpec("ParamApero.xml"),\
        #aObj ,\
        #aObj \
     )


#define OptStdGetFromAp(aStr,aObj)\
OptionalGetObjFromFile_WithLC<c##aObj>\
    (\
        0,0,\
        aStr,\
        StdGetFileXMLSpec("ParamApero.xml"),\
        #aObj ,\
        #aObj \
     )




// Par ex :    cFileOriMnt aFileZ = StdGetFromPCP(aStrZ,FileOriMnt);



//   const char * GetEntryStringifie(const std::string &);



//template <class Type> Type eFromString(const std::string & aName);

inline cElXMLTree * GetRemanentFromFileAndTag(const std::string & aNameFile,const std::string & aNameTag)
{
   static std::map<std::string,cElXMLTree *> DicoHead;
   if (DicoHead[aNameFile] ==0)
       DicoHead[aNameFile] = new cElXMLTree(aNameFile);

   return DicoHead[aNameFile]->GetOneOrZero(aNameTag);
}

template <class Type> bool InitObjFromXml
                           (
                                 Type & anObj,
                                 const std::string & aNameFile,
                                 const std::string& aFileSpec,
                                 const std::string & aNameTagObj,
                                 const std::string & aNameTagType
                           )
{
   if (GetRemanentFromFileAndTag(StdGetFileXMLSpec(aFileSpec),aNameTagType))
   {
       anObj = StdGetObjFromFile<Type>(aNameFile,StdGetFileXMLSpec(aFileSpec),aNameTagObj,aNameTagType);
       return true;
   }
   return false;
}
template <class Type> bool StdInitObjFromXml
                           (
                                 Type & anObj,
                                 const std::string & aNameFile,
                                 const std::string & aNameTagObj,
                                 const std::string & aNameTagType
                           )
{
     return
              InitObjFromXml(anObj,aNameFile,"ParamApero.xml",aNameTagObj,aNameTagType)
         ||   InitObjFromXml(anObj,aNameFile,"ParamMICMAC.xml",aNameTagObj,aNameTagType)
         ||   InitObjFromXml(anObj,aNameFile,"SuperposImage.xml",aNameTagObj,aNameTagType)
         ||   InitObjFromXml(anObj,aNameFile,"ParamChantierPhotogram.xml",aNameTagObj,aNameTagType)
     ;
}


template <class Type> void BinUndumpObj(Type & anObj,const std::string & aFile, const std::string & aNameTagObj="", const std::string & aNameTagType="")
{
     ELISE_fp aFPIn(aFile.c_str(),ELISE_fp::READ);
     int aNum;

     BinaryUnDumpFromFile(aNum,aFPIn);
     // NumHgRev doesn't work with the new Git version
     //if (aNum!=NumHgRev())
     //{
     //}

     std::string aVerifMangling;
     BinaryUnDumpFromFile(aVerifMangling,aFPIn);
     if (aVerifMangling!=Mangling((Type*)0))
     {
        std::string aXmlName = StdPrefix(aFile)+".xml";
        if (ELISE_fp::exist_file(aXmlName))
        {
           std::cout << "Dump version problem for "<<  aFile << " , try to recover from xml\n";
           if (StdInitObjFromXml(anObj,aXmlName,aNameTagObj,aNameTagType))
           {
               MakeFileXML(anObj,aFile);
               std::cout << "    OK recovered " << aFile << "\n";
               return;
           }
        }

        std::cout << "For file " << aFile << " TagO="  << aNameTagObj << " TagT=" << aNameTagType << "\n";
        ELISE_ASSERT(false,"Type has changed between Dump/Undump")
     }


     BinaryUnDumpFromFile(anObj,aFPIn);
     aFPIn.close();
}





#endif //  _ELISE_PRIVATE_FILES_H


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
