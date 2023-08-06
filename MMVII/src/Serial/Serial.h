#ifndef  _SERIAL_H_
#define  _SERIAL_H_
#include "MMVII_Stringifier.h"
#include "cMMVII_Appli.h"



/** \file Serial.h
    \brief 

*/

namespace MMVII
{

class cAr2007 ;
class cEOF_Exception;
enum class eLexP;
class cSerialTokenGenerator;
class cSerialTokenParser ;
class cXmlSerialTokenParser ;
class cSerialTree;


/// Base class of all archive class

/**
     Base class of all archive class;
 
     Adding a new kind of archive, essentially consist to indicate how to read/write atomic values.
    It is a bit more complicated with tagged format
*/

class cAr2007 : public cMemCheck
{
    public  :
         friend class cAuxAr2007;

         template <class Type,class TypeCast> inline void TplAddDataTermByCast (const cAuxAr2007& anOT,Type&  aValInit,TypeCast* UnUsed)
         {
// StdOut() << "TplAddDataTermByCast " << (int) aValInit << "BINAY" << mBinary << "\n";
              if (mBinary)
              {
                  cRawData4Serial aRDS = cRawData4Serial::Tpl(&aValInit,1);
                  RawAddDataTerm(aRDS);
              }
              else
              {
                   TypeCast aCast = aValInit;
                   RawAddDataTerm(aCast);
                   if (mInput) 
                      aValInit = aCast;
              }
         }

         /// default do nothing)
         virtual void AddComment(const std::string &);
         ///  Tagged File = xml Like, important for handling optionnal parameter
         bool  Tagged() const;
         ///  May optimize the action
         bool  Input() const;
         /// Allow to  know by advance if next optionnal value is present, usefull with Xml
         /// Default return error
         virtual int NbNextOptionnal(const std::string &);
         virtual ~cAr2007();
         virtual void Separator(); /**< Used in final but non atomic type, 
                                        for ex with Pt : in text separate x,y, in bin do nothing */
         virtual size_t HashKey() const;

      // Final atomic type for serialization
         virtual void RawAddDataTerm(int &    anI) =  0; ///< Heriting class descrine how they serialze int
         virtual void RawAddDataTerm(size_t &    anI) =  0; ///< Heriting class descrine how they serialze int
         virtual void RawAddDataTerm(double &    anI) =  0; ///< Heriting class descrine how they serialze double
         virtual void RawAddDataTerm(std::string &    anI) =  0; ///< Heriting class descrine how they serialze string
         virtual void RawAddDataTerm(cRawData4Serial  &    aRDS) =  0; ///< Heriting class descrine how they serialze string
                                                                       //

         virtual void OnBeginTab() {}
         virtual void OnEndTab() {}
         /**  Called when we add the size of vect/list, for compatibility, just add int whit tag Nb, can be overloaded
          * when "well parenthesis struct" is used to compute the size */
         virtual  void AddDataSizeCont(int & aNb,const cAuxAr2007 & anAux);

    protected  :
         cAr2007(bool InPut,bool Tagged,bool Binary);
         int   mLevel;
         bool  mInput;
         bool  mTagged;
         bool  mBinary;   //  != from tagged iw we implemant a pure txt format
     private  :

         /// By default error, to redefine in hashing class
         /// This message is send before each data is serialized, tagged file put/read their opening tag here
         virtual void RawBeginName(const cAuxAr2007& anOT);
         /// This message is send each each data is serialized, tagged file put/read their closing tag here
         virtual void RawEndName(const cAuxAr2007& anOT);


      // Final non atomic type for serialization
};



class cEOF_Exception
{
};


/**  Facility for xml parsing whatever you do after
 */

enum class eLexP
     {
          eStdToken_UK, ///< standard string unknown
          eStdToken_Int, ///< standard string int
          eStdToken_Size_t, ///< standard string int
          eStdToken_Double, ///< standard string int
          eStdToken_String,   ///< string in ""
          eStdToken_RD4S,   ///< string in ""
          eBegin,    ///< begin, before any read, 
          eEnd,      ///< end of read, like encounter EOF
          eUp,       ///< increase the depth of tree, like  {[(  <hhh>
          eDown,     ///<  decrease the depth of tree, like  )]}  </hhh>
          eSep       ///<  single separator like ,&:
     };

typedef std::pair<std::string,eLexP>  tResLex;

class cSerialTokenGenerator
{
	public :
          virtual tResLex GetNextLex() = 0;
};

class cSerialTokenParser : public cSerialTokenGenerator
{
     public    :
          cSerialTokenParser(const std::string & aName,eTypeSerial aTypeS);
          tResLex GetNextLex() override;

     protected :

          virtual bool BeginPonctuation(char aC) const = 0;
          virtual tResLex AnalysePonctuation(char aC)  = 0;

          tResLex GetNextLex_NOEOF();

          inline std::istream  & Ifs() {return mMMIs.Ifs();}
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
          eTypeSerial                       mTypeS;
};

extern const char * TheXMLBeginCom  ;
extern const char * TheXMLEndCom    ;

class cXmlSerialTokenParser : public cSerialTokenParser
{
     public :
          cXmlSerialTokenParser(const std::string & aName,eTypeSerial aTypeS);
     protected :
          bool BeginPonctuation(char aC) const override;
          tResLex AnalysePonctuation(char aC)  override;
};

class cSerialTree
{
      public :
          cSerialTree(cSerialTokenGenerator &,int aDepth);
	  cSerialTree(const std::string & aValue,int aDepth);
	  void  Xml_PrettyPrint(cMMVII_Ofs& anOfs) const;


          std::string mValue;
	  std::vector<cSerialTree>  mSons;
	  int         mDepth;
     private :
	  void  Indent(cMMVII_Ofs& anOfs) const;
};

cAr2007 * Alloc_cOMakeTreeAr(const std::string & aName,eTypeSerial aTypeS);


};

#endif // _SERIAL_H_
