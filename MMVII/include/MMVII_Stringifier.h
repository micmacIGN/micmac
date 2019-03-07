#ifndef  _MMVII_Stringifier_H_
#define  _MMVII_Stringifier_H_

namespace MMVII
{


/** \file MMVII_Stringifier.h
    \brief Interface for services related to read/write values

    It contains : (1) basic "Value <-> string" read write for atomic type
    (2) service for arg reading in MMVII applications 
    (3) services for serialization
*/

class  cSpecOneArg2007 ;
class cCollecSpecArg2007;
class cAr2007;             ///< Mother class of archive, do not need to export
class cAuxAr2007;          ///< Auxilary class, only neccessry

///  string<-->Value conversion
/**
    This class handle conversion (two way) between
    atomic type and string.  Contain only static members.
*/ 
template <class Type> class  cStrIO
{
    public :
       /// Atomic -> string
       static std::string ToStr(const Type &);     
      /// String -> Atomic object
       static Type  FromStr(const std::string &);  
      /// Readable name for type
       static const std::string  msNameType;
};

/// Facilities when the type is well defined
template <class Type> std::string ToStr(const Type & aV) {return cStrIO<Type>::ToStr(aV);}
std::string ToStr(int aVal,int aSzMin);
std::string FixDigToStr(double aSignedVal,int aNbDig);


// std::string  ToS_NbDigit(int aNb,int aNbDig,bool AcceptOverFlow=false); ///< To generate with a given number of digits



/** Do the test using only specialization ...
   Could make something generic with isstream, but remember it was tricky ...
*/

template <>  std::string cStrIO<bool>::ToStr(const bool & anI);
template <>  bool cStrIO<bool>::FromStr(const std::string & aStr);
template <>  std::string cStrIO<int>::ToStr(const int & anI);
template <>  int cStrIO<int>::FromStr(const std::string & aStr);
template <>  std::string cStrIO<double>::ToStr(const double & anI);
template <>  double cStrIO<double>::FromStr(const std::string & aStr);
template <>  std::string cStrIO<std::string>::ToStr(const std::string & anI);
template <>  std::string cStrIO<std::string>::FromStr(const std::string & aStr);

/*
template <>  std::string cStrIO<cPt2dr>::ToStr(const cPt2dr & anI);
template <>  cPt2dr cStrIO<cPt2dr>::FromStr(const std::string & aStr);
template <>  std::string cStrIO<cPt2di>::ToStr(const cPt2di & anI);
template <>  cPt2di cStrIO<cPt2di>::FromStr(const std::string & aStr);
*/


/** These functions offer an"easy" interface to cStrIO, however I think
*    cStrIO is still usefull when type inference becomes too compliicated
*/
template  <class Type> std::string ToS(const Type & aV) {return cStrIO<Type>::ToStr(aV);}
template  <class Type> void FromS(const std::string & aStr,Type & aV) { aV= cStrIO<Type>::FromStr(aStr);}

/*  ================================================== */
/*                                                     */
/*          MMVII ARGS                                 */
/*                                                     */
/*  ================================================== */

/// Semantic of cSpecOneArg2007
/**
   This "semantic" are usefull to have a finer process
   od the parameter in the global constructor. For example
   indicate that a parameter is  internal (dont show) or
   common to all command (show only when required).

   Many parameter are at a low level just string , those indicating
   pattern of file or those indicatif 

     
*/
class cSemA2007
{
   public :
      cSemA2007(eTA2007 aType,const std::string & anAux);
      cSemA2007(eTA2007 aType);

      eTA2007 Type()            const;  ///< Accessor
      const std::string & Aux() const;  ///< Accessor
      std::string  Name4Help() const;   ///< Use  E2Str(const eTA2007 &) but filter to usefull, add Aux

    private :

      eTA2007      mType;
      std::string  mAux;
};



/// Base class  to describe one paramater specification
/**  The job will be done by template inheriting classes
    who knows how to use a string for computing a value
*/
class  cSpecOneArg2007 : public cMemCheck
{
     public :
        typedef std::vector<cSemA2007> tVSem;

        /// Default empty semantique
        static const tVSem   TheEmptySem;
        ///  Name + comment  + semantic
        cSpecOneArg2007(const std::string & aName,const std::string & aCom,const tVSem & = TheEmptySem);

        ///  This action defined in heriting-template class initialize "real" the value from its string value 
        virtual void InitParam(const std::string & aStr) = 0;
        virtual void * AdrParam() = 0;    ///< cast to void * of typed adress, used by Application know if init 
        virtual const std::string & NameType() const = 0;  ///< as int, bool, ....
        virtual std::string  NameValue() const = 0;  ///< Used to print def value
        /// Does any of  mVSem contains aType
        bool HasType(const eTA2007 & aType,std::string * aValue=0)            const;

        const tVSem & VSem() const;         ///< Accessor
        const std::string  & Name() const;  ///< Accessor
        const std::string  & Com() const;   ///< Accessor
        int NbMatch () const;         ///< Accessor
        void IncrNbMatch() ;

        std::string  Name4Help() const;   ///< concat and format the different Name4Help of tVSem

     private :

         std::string     mName; ///< Name for optionnal
         std::string     mCom;  ///< Comment for all
         tVSem           mVSem;    ///< Vector of semantic
         int             mNbMatch;  ///< Number of match, to generate error on multiple names
};

typedef std::shared_ptr<cSpecOneArg2007>  tPtrArg2007;
typedef std::vector<tPtrArg2007>        tVecArg2007;


/// Collection of arg spec
/**
    Class for representing the collections  of parameter specificification (mandatory/optional/global)
    This class is nothing more than a encapsultion of a vectot of cSpecOneArg2007*,  but it
    was easier for defining the "operator <<" and  controling the access
*/

class cCollecSpecArg2007
{
   public :
      friend class cMMVII_Appli; ///< only authorizd to construct
      friend void   Bench_0000_Param(); ///< authorized for bench
      size_t size() const;
      tPtrArg2007 operator [] (int) const;
      void clear() ;
      cCollecSpecArg2007 & operator << (tPtrArg2007 aVal);
   private :
      friend class cMMVII_Appli;
      tVecArg2007 & Vec();
      cCollecSpecArg2007(const cCollecSpecArg2007&) = delete;
      tVecArg2007  mV;
      cCollecSpecArg2007();
};


///  Two auxilary fonction to create easily cSpecOneArg2007 , one for mandatory
template <class Type> tPtrArg2007 Arg2007(Type &, const std::string & aCom, const std::vector<cSemA2007> & = cSpecOneArg2007::TheEmptySem);
///  One for optional args
template <class Type> tPtrArg2007 AOpt2007(Type &,const std::string & aName, const std::string & aCom,const std::vector<cSemA2007> & = cSpecOneArg2007::TheEmptySem);



/*  ================================================== */
/*                                                     */
/*          SERIALIZATION                              */
/*                                                     */
/*  ================================================== */



/// Auxiliary class for Archive manipulation

/**
   The serialization "file" inherit from the mother class cAr2007. This
   class is accessible via the  the  cAuxAr2007 , the automitazation of 
   calling level (usefull for example in XML pretty printing) is done by
   constructor and destructor of cAuxAr2007.
*/

class cAuxAr2007
{
     friend class cAr2007;
     public :
         /// No usefull copy constructor inhibit it
         cAuxAr2007 (const cAuxAr2007 &) = delete;
         /// Increase counter, send the  virtual message  of opening  tag
         cAuxAr2007 (const std::string & aName,cAr2007 &);
         /// Decrease counter, send the  virtual message  of closing  tag
         ~cAuxAr2007 ();
         ///  Just a more connvenient way to call 
         cAuxAr2007 (const std::string & aName, const cAuxAr2007 &);

         const std::string  Name () const {return mName;}
         cAr2007 & Ar()             const {return mAr;}
         /// Call mAr, indique if for read or write
         bool Input() const;
         /// Call mAr, indicate if xml-like (more sophisticated optional handling)
         bool Tagged() const;
         /// Call mAr, return 0 or 1, indicating if next optionnal value is present
         int NbNextOptionnal(const std::string &) const;
     private : 
         const std::string  mName;
         cAr2007 & mAr;
};



/// Create an archive structure, its type (xml, binary, text) is determined by extension
 cAr2007* AllocArFromFile(const std::string & aName,bool Input);
//  std::unique_ptr<cAr2007 > AllocArFromFile(const std::string & aName,bool Input);

/** Here are the atomic serialization function */

void AddData(const  cAuxAr2007 & anAux, int  &  aVal); ///< for int
void AddData(const  cAuxAr2007 & anAux, double  &  aVal) ; ///< for double
void AddData(const  cAuxAr2007 & anAux, std::string  &  aVal) ; ///< for string
void AddData(const  cAuxAr2007 & anAux, cPt2dr  &  aVal) ;  ///<for cPt2dr
void AddData(const  cAuxAr2007 & anAux, tNamePair  &  aVal) ;  ///< for Pair of string
void AddData(const  cAuxAr2007 & anAux, tNameOCple  &  aVal) ;  ///< for Ordered Cple of string

/// Serialization for optional
// template <class Type> void AddOptData(const cAuxAr2007 & anAux,const std::string & aTag0,boost::optional<Type> & aL);


void DeleteAr(cAr2007 *); /// call delete, don't want to export a type only to delete it!

/// By default no MMV1 save, def value is required for general template SaveInFile
template<class Type> void  MMv1_SaveInFile(const Type & aVal,const std::string & aName)
{
     // MMVII_INTERNAL_ASSERT_always(false,"No MMV1 save for " + std::string(typeid(Type).name()));
     MMVII_INTERNAL_ASSERT_always(false,"No MMV1 save for type" );
}

/// Specialisation for existing value : tNameRel
template<> void  MMv1_SaveInFile(const tNameRel & aVal,const std::string & aName);
/// Specialisation for existing value : tNameSet
template<> void  MMv1_SaveInFile(const tNameSet & aVal,const std::string & aName);

/// call static function of cMMVII_Appli, cannot make forward declaration of static function
bool GlobOutV2Format();

/// Indicate if a file is really XML, created by MMVII and containing the expected Tag
bool IsFileXmlOfGivenTag(bool Is2007,const std::string & aName,const std::string & aTag); 


template <class Type> const std::string  & XMLTagSet();
template <> const std::string  &           XMLTagSet<std::string> ();
template <> const std::string  &           XMLTagSet<tNamePair>   ();

template <class Type> const std::string  & MMv1_XMLTagSet();
template <> const std::string  &           MMv1_XMLTagSet<std::string> ();
template <> const std::string  &           MMv1_XMLTagSet<tNamePair>   ();

/*****************************************************************/


/*
The serialization could be genarlized to comparison, however I don think its usefull
to do it too generaly with only one serialization mechanism, the price would be :

   - some "lourdeur" in standard serialization, if we oblige to have two parameter identic
   - some time consuption in comparison if we oblige 

So maybe later will add a cCmpSerializer package


class cCmpSerializer
{
     public :
         virtual bool Cmp(const double&, const double &) =0;
         virtual bool Cmp(const std::string &, const std::string  &) =0;

         bool Cmp(const cPt2dr &aP1, const cPtd2r  &aP2) 
         {
              return Cmp(aP1.x(),aP1.y()) && Cmp(aP2.x()&&aP2.y());
         }
         virtual bool Cmp(const int& aV1, const int &aV2) {return Cmp(double(aV1),double(aV2));}
};
*/



};
#endif  //  _MMVII_Stringifier_H_
