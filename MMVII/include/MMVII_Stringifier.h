#ifndef  _MMVII_Stringifier_H_
#define  _MMVII_Stringifier_H_


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
};

/** Do the test using only specialization ...
   Could make something generic with isstream, but remember it was tricky ...
*/

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


/** These functions offer an"easy" interface to cStrIO, however I thinj
*    cStrIO is still usefull when type inference becomes too compliicated
*/
template  <class Type> std::string ToS(const Type & aV) {return cStrIO<Type>::ToStr(aV);}
template  <class Type> void FromS(const std::string & aStr,Type & aV) { aV= cStrIO<Type>::FromStr(aStr);}


/*  ================================================== */
/*                                                     */
/*          SERIALIZATION                              */
/*                                                     */
/*  ================================================== */


/// Mother class of archive, do not need to export
class cAr2007; 
/// Auxilary class, only neccessry
class cAuxAr2007;

std::unique_ptr<cAr2007> AllocArFromFile(const std::string & aName,bool Input);

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
         bool Input() const;
         bool Tagged() const;
         int NbNextOptionnal(const std::string &) const;
     private : 
         const std::string  mName;
         cAr2007 & mAr;
};

/** Here are the atomic serialization function */

void AddData(const  cAuxAr2007 & anAux, int  &  aVal);
void AddData(const  cAuxAr2007 & anAux, double  &  aVal) ;
void AddData(const  cAuxAr2007 & anAux, std::string  &  aVal) ;
void AddData(const  cAuxAr2007 & anAux, cPt2dr  &  aVal) ;

/** Template for list (will be easily extended to other containter */

template <class Type> void AddData(const cAuxAr2007 & anAux,std::list<Type> & aL)
{
    int aNb=aL.size();
    AddData(cAuxAr2007("Nb",anAux),aNb);
    if (aNb!=int(aL.size()))
    {
       Type aV0;
       aL = std::list<Type>(aNb,aV0);
    }
    for (auto el : aL)
    {
         AddData(cAuxAr2007("el",anAux),el);
    }
}

/** Template for optional parameter, complicated becaus in xml forms, it handles the compatibility with new
added parameters */

template <class Type> void OptAddData(const cAuxAr2007 & anAux,const std::string & aTag0,boost::optional<Type> & aL)
{
    /// Not mandatory, but optionality being an important feature I thought usefull to see it in XML file
    std::string aTagOpt;
    const std::string * anAdrTag = & aTag0;
    if (anAux.Tagged())
    {
        aTagOpt = "Opt:" + aTag0;
        anAdrTag = & aTagOpt;
    }

    /// In input mode, we must decide if the value is present
    if (anAux.Input())
    {
        /// The archive knows if the object is present
        if (anAux.NbNextOptionnal(*anAdrTag))
        {
           /// If yes read it and initialize optional value
           Type  aV;
           AddData(cAuxAr2007(*anAdrTag,anAux),aV);
           aL = aV;
        }
        /// If no just put it initilized
        else
           aL = boost::none;
        return;
    }

    /// Now in writing mode
    int aNb =  aL.is_initialized() ? 1 : 0;
    /// Tagged format (xml) is a special case
    if (anAux.Tagged())
    {
       /// It the value exist put it normally else do nothing (the absence of tag will be analysed at reading)
       if (aNb)
          AddData(cAuxAr2007(*anAdrTag,anAux),*aL);
    }
    else
    {
       /// Indicate if the value is present and if yes put it
       AddData(anAux,aNb);
       if (aNb)
          AddData(anAux,*aL);
    }
}

extern const std::string TagMMVIISerial;

template<class Type> void  SaveInFile(const Type & aVal,const std::string & aName)
{
    std::unique_ptr<cAr2007 > anAr = AllocArFromFile(aName,false);

    cAuxAr2007  aGLOB(TagMMVIISerial,*anAr);
    // cAuxAr2007  anOpen(aTag,*anAr);
    AddData(aGLOB,const_cast<Type&>(aVal));
}

template<class Type> void  ReadFromFile(Type & aVal,const std::string & aName)
{
    std::unique_ptr<cAr2007 > anAr = AllocArFromFile(aName,true);

    cAuxAr2007  aGLOB(TagMMVIISerial,*anAr);
    AddData(aGLOB,aVal);
}









#endif  //  _MMVII_Stringifier_H_
