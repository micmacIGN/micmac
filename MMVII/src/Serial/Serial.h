#ifndef  _SERIAL_H_
#define  _SERIAL_H_
#include "MMVII_Stringifier.h"
#include "cMMVII_Appli.h"



/** \file Serial.h
    \brief 

*/

namespace MMVII
{

/**  Schema of serialization by tree
 *        
 *      - for read/write object in "serialization by tree"
 *
 *          *  read   
 *                           "Complex process :  Unfold"
 *                               |
 *               File ---> tree  --> list of token  -> fill the obect
 *
 *                                            "Complex process: PrettyPrint"
 *          *  write                                     |
 *                       
 *                 "token ar acumulated in RawAddData of cOMakeTreeAr"
 *                       |
 *               Object -> List of token |  --->   Tree --->  File
 *                                       |    
 *                                       "at destruction of cOMakeTreeAr"
 */

class cAr2007 ;                // base class of all archives (serializer)
class cEOF_Exception;          // use to catch End of File w/o by exception
enum class eLexP;              //  possible value of lexical analysis
class cSerialGenerator;           //  base class for stuff generaing token (file, list of token ..)
class cSerialFileParser ;      // base class for token generator resulting from file parsing
class cXmlSerialTokenParser ;  // instantiation of cSerialFileParser to xml files
class cJsonSerialTokenParser ; // instantiation of cSerialFileParser to json files
class cSerialTree;             //  class for representing in a tree the "grammatical" parsing of a token generator


// From boost:: ...
template <class T>
static inline void hash_combine(std::size_t& seed, T const& v)
{
   std::hash<T> hasher;
   seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// template <class T> void HashCombine(std::size_t& seed, T const& v);



///  Use to handle End Of File using exception
class cEOF_Exception
{
};

/** Store the type of token ,  in fact there is only 4 values critical Up/Down/SzCont */
enum class eLexP
     {
          eStdToken_UK, ///< standard string unknown
          eStdToken_Int, ///< standard string int
          eStdToken_Size_t, ///< standard string int
          eStdToken_Double, ///< standard string int
          eStdToken_String,   ///< string in ""
          eStdToken_RD4S,   ///< Data 
          eSizeCont,    ///< mean that the value is the size of a container that will require special treatment
          eBegin,    ///< begin, before any read, 
          eEnd,      ///< end of read, like encounter EOF
          eUp,       ///< increase the depth of tree, like  {[(  <hhh>
          eDown,     ///<  decrease the depth of tree, like  )]}  </hhh>
          eSep       ///<  single separator like ,&:
     };


class cResLex
{
     public :
         cResLex(std::string,eLexP,eTAAr);

         std::string  mVal;  ///< the value = string of the token
         eLexP        mLexP; ///< nature of this value => comes from the files
	 eTAAr        mTAAr;  ///< Type of element, comes from C++ (is it a vector, a map ...)
         std::string  mComment;  ///< possible comment associated
};

/**  Abstraction of files as a stuff that generates token, can be used also on list
 */
class cSerialGenerator
{
	public :
          /// Fundamental method, generate token
          virtual cResLex GetNextLex() = 0;
	  ///  method called to check the tree when it is finished (closing tag occurs)
	  virtual void CheckOnClose(const cSerialTree &,const std::string & aTagClose) const;

          cResLex GetNextLexSizeCont() ; /// Get a token that "must" be a size of container
          cResLex GetNextLexNotSizeCont() ; /// Get a token, "skeeping" all size of container
};

class cTokenGeneByList : public cSerialGenerator
{
      public :
        typedef std::list<cResLex>   tContToken;
        cResLex GetNextLex() override;
        cTokenGeneByList(tContToken &);

      private :
        tContToken *           mContToken;
        tContToken::iterator   mItToken;
};


/** Specialization of cSerialGenerator for file, contain basic element of grammar */

typedef  std::pair<bool,std::string> tTestFileSerial;

class cSerialFileParser : public cSerialGenerator,
	                   public cMemCheck
{
     public    :
          cSerialFileParser(const std::string & aName,eTypeSerial aTypeS); ///< constructor
	  virtual ~cSerialFileParser(); ///< destructor
	  static cSerialFileParser *  Alloc(const std::string & aName,eTypeSerial aTypeS); ///< allocator of inheriting class

          cResLex GetNextLex() override; ///< generate tokens from files

	  /// Return if the file is ok and, if ok,  indicate the master tag
	  static tTestFileSerial TestFirstTag(const std::string & aNameFile);
     protected :

          virtual bool BeginPonctuation(char aC) const ; ///< is the caracter the begining of a punctuation
          virtual cResLex AnalysePonctuation(char aC)  ; ///<  analyse of token after a punction is detected

          cResLex GetNextLex_NOEOF(); ///< get token, w/o accepting end-of-file

          inline std::istream  & Ifs() {return mMMIs.Ifs();} ///< accessor
          /// Get a char, and check its not EOF, only access to mMMIs.get() in this class
          int GetNotEOF();
          /// error specific handler
          void Error(const std::string & aMes);
           /// Skeep all series of space and comment
           int  SkeepWhite();

           /// Skeep one <!-- --> or <? ?>
           bool SkeepOneKindOfCom(const char * aBeg,const char * anEnd);
           /// Skeep a comment
           bool SkeepCom();

          /// If found Skeep one extpected string, and indicate if it was found,
          bool SkeepOneString(const char * aString);
          std::string  GetQuotedString();  /// extract "ddgg \\  kk "

          cMMVII_Ifs                        mMMIs; ///< secured istream
          eTypeSerial                       mTypeS; ///< Type of serialization (xml,json ...)
};

extern const char * TheXMLBeginCom  ; ///<  string begining xml-comment
extern const char * TheXMLEndCom    ; ///<  string ending xml-comment
extern  const char * TheXMLHeader;    ///<  header of xml-file


/** Specialization to parse xml-files */
class cXmlSerialTokenParser : public cSerialFileParser
{
     public :
          cXmlSerialTokenParser(const std::string & aName);
     protected :
          bool BeginPonctuation(char aC) const override;
          cResLex AnalysePonctuation(char aC)  override;
	  void CheckOnClose(const cSerialTree &,const std::string &) const override;
};

/** Specialization to parse json-files */
class cJsonSerialTokenParser : public cSerialFileParser
{
     public :
          cJsonSerialTokenParser(const std::string & aName);

	  static  const  std::string  ComentString;
     protected :

	  static  const  std::string  OpenCars;  ///< cars generating an opening "{["
	  static  const  std::string  CloseCars; ///< cars generating a closing "]}"
	  static  const  std::string  SepCars;

          bool BeginPonctuation(char aC) const override;
          cResLex AnalysePonctuation(char aC)  override;
	  void CheckOnClose(const cSerialTree &,const std::string &) const override;
};

class cResDifST
{
     public :
        cResDifST(const cSerialTree*,const cSerialTree*);

        const cSerialTree * mST1;
        const cSerialTree * mST2;
};

/**  Class to represent as a tree  the object, in write, or the file, in read, for the serialization by tree
 */
class cSerialTree : public cMemCheck
{
      public :
	  cSerialTree(const std::string & aValue,int aDepth,eLexP aLexP,eTAAr); ///< For leaf
	  /// for "standard" nodes
          cSerialTree(cSerialGenerator &,const std::string & aValue,int aDepth,eLexP aLexP,eTAAr);
          /// top call
          cSerialTree(cSerialGenerator &);

	  static cSerialTree* AllocSimplify(const std::string &);


	  /// Compute firt occurence of tree difference return as res diff 
	  cResDifST AnalyseDiffTree(const cSerialTree &,const std::string &aSkeep) const;
	        // "pretty printing" functions
	  void  Xml_PrettyPrint(cMMVII_Ofs& anOfs) const;  /// xml-pretty print
	  void  Json_PrettyPrint(cMMVII_Ofs& anOfs) const; /// json-pretty print
	  void  Raw_PrettyPrint(cMMVII_Ofs& anOfs) const;  /// Tagt-pretty print
	  void  CSV_PrettyPrint(std::vector<std::string>& aRes,bool IsSpecif) const;  /// print 

      /// Extract a descendant from its name
      std::vector<const cSerialTree *> GetAllDescFromName(const std::string &) const;
      /// Test if there is a one and only one descendant
      const cSerialTree * GetUniqueDescFromName(const std::string &) const;


	  /// Assert that there is only 1 son and return it
	  const cSerialTree & UniqueSon() const; 
	  /// Assert Father !=0
	  const cSerialTree & Father() const;

	  /// put linearly the contain of the node in a list of token , will be used as a token generator
	  void Unfold(std::list<cResLex> &,eTypeSerial) const;

	  const std::vector<cSerialTree>&  Sons() const; /// acessor
          const std::string & Value() const ;            /// accessor
     private :
      void RecGetAllDescFromName(std::vector<const cSerialTree *>&,const std::string &) const;

      void RecursSetFather(cSerialTree *);
	  // cSerialTree(const cSerialTree &) ;
	  /// Implement using exception
	  void Rec_AnalyseDiffTree(const cSerialTree &,const std::string & aSkeep) const;

	  bool IsTerminalNode() const;     ///< is it a node w/o son and not tagged
	  bool IsTabulable() const;        ///< can it be printed as a tab == is it a non tagged node with all son terminal
	  bool IsSingleTaggedVal() const;  ///< is it  a 

	  void  Rec_Xml_PrettyPrint(cMMVII_Ofs& anOfs) const;  /// xml-pretty print
	  /// recursive json pretty print , IsLast to handle "," , aCptComment to generat different comment tag at each occurence
	  void Rec_Json_PrettyPrint(cMMVII_Ofs& anOfs,bool IsLast,int &aCptComment) const; 
	  ///  print a tab on a single line for fixed size tab (like "cPt2di" )
	  void Json_PrintSingleTab(cMMVII_Ofs&,bool Last,int &aCpt) const;
	  /// print a "terminal" node  on a signle line
	  void Json_PrintTerminalNode(cMMVII_Ofs&,bool Last,int &aCpt) const;
	  /// print the eventually the comment (will increment the counter)
	  void Json_Comment(cMMVII_Ofs&,bool Last,int & aCpt) const;
	  /// is it a key to ommit , like the "el" , "Pair" that would have multi occurence
	  bool Json_OmitKey() const;

	  /// 
	  void  Rec_CSV_PrettyPrint(std::vector<std::string> & aRes,bool IsSpecif) const;  /// print 

	  /// Was the generated for comment
	  static bool IsJsonComment(const std::string&) ;
	  ///  Generate a new comment tag, increment counnetr
	  static std::string TagJsonComment(int&);

	  /// Udpate the maximal depth of sons + the total length
	  void  UpdateMaxDSon();
	  ///  Create a idention prop to  "mDepth+aDeltaInd"
	  void  Indent(cMMVII_Ofs& anOfs,int aDeltaInd) const;

	  eLexP       mLexP;   ///  lexical element, essentially "Up" Or not  is used, extracted from file parsing
	  eTAAr       mTAAr;   /// semantical element, exracted from C++ file by AddData
          std::string mValue;  /// value of the node, like tag "<F>"  of final value (int, string ...)
          std::string mComment;  /// possible comment associated to the node by C++
	  cSerialTree * mFather;
	  std::vector<cSerialTree>  mSons;  ///  sons of the tree, because it is a tree ;-)
	  int         mDepth;       /// depth computed in recursion by "D+1"
	  int         mMaxDSon;     /// maximal depth of all its son
	  size_t      mLength;      /// lenght of the pontially unfolded tree
};

///  external allocator for outing by tree-serialization
cAr2007 * Alloc_cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS,bool IsSpecif=false);
///  external allocator for inputing by tree-serialization
cAr2007 * Alloc_cIMakeTreeAr(const std::string & aName,eTypeSerial aTypeS);



};

#endif // _SERIAL_H_
