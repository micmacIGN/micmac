#ifndef  _MMVII_Util_TPL_H_
#define  _MMVII_Util_TPL_H_

namespace MMVII
{

/** \file MMVII_util_tpl.h
    \brief Utilitaries for some interface to template type

*/

template <class Type> class cInterfSet ;
template <class Type> class cSelector ;
template <class Type> class cDataSelector ;

/* ============================================= */
/*                                               */
/*            cInterfSet                         */
/*                                               */
/* ============================================= */

///  Interface to sets services (set in extension)

/** Derived class will implement the services on :
 
    * std::unordered_set (done)
    * std::set  probably
    * std::vector
    * maybe hybrid (i.e. type that may change when size grows)
*/

template <class Type> class cInterfSet : public cMemCheck
{
    public :
         virtual bool Add(const Type &) = 0;
         virtual bool In(const Type &) const = 0;
         virtual bool Suppress(const Type &)  = 0;
         virtual void    clear() = 0;
         virtual int    size() const = 0;

         virtual void  PutInSet(std::vector<const Type *> &) const = 0; ///< Some type requires iteration 


         cInterfSet<Type> &  operator =(const cInterfSet<Type> &);
         virtual ~cInterfSet() ;
};

template <class Type> cInterfSet<Type> * AllocUS(); ///<  Allocator, unordered_set

template <class Type>  cInterfSet<Type> &  operator *= (cInterfSet<Type> &,const cInterfSet<Type> &);
template <class Type>  cInterfSet<Type> &  operator += (cInterfSet<Type> &,const cInterfSet<Type> &);
template <class Type>  cInterfSet<Type> &  operator -= (cInterfSet<Type> &,const cInterfSet<Type> &);

/* ============================================= */
/*                                               */
/*            cSelector                          */
/*                                               */
/* ============================================= */

/// Base class for Selector
/**  a selector is just a boolean predicate,
     the heriting class will implement override Match
     Class is accessible via cSelector
*/

template <class Type> class cDataSelector : public cMemCheck
{
    public :
        virtual bool Match(const Type &) const = 0;  ///< fundamuntal function
        virtual ~cDataSelector() ;
};

/// Access class to Selector
/** Selector are used via smart pointer
   which allow to build automatically the 

   Also the function "BenchSelector" illustrates all the
 functionnalities of selectors.
*/

template <class Type> class  cSelector : public  std::shared_ptr<cDataSelector<Type> >
{
     public :

          bool Match(const Type &) const;
          cSelector(cDataSelector<Type> *);
     private :
};

template<class Type> cSelector<Type> operator && (const cSelector<Type> &,const cSelector<Type>&); ///<  And/Inter of  selector
template<class Type> cSelector<Type> operator || (const cSelector<Type> &,const cSelector<Type>&); ///< Or/Union of selector
template<class Type> cSelector<Type> operator !  (const cSelector<Type> &); ///< Negation of selector

/**  Selector corresponding to interval  V1..V2,  boost::none mean no bound
    aInclLow,InclUp  indicate if bounds are included.

     (V1,V2,true,true)  => [V1,V2]

     (none,V2,true,false)  =>  ]-infinity,V2[ 
*/
template<class Type> cSelector<Type> GenIntervalSelector( const boost::optional<Type> & aV1,
                                                       const boost::optional<Type> & aV2,
                                                       bool aInclLow,bool InclUp);

/// Facilty to GenIntervalSelector : left interval  ]-infinity,...
template<class Type> cSelector<Type> LeftHalfIntervalSelector(const Type & aV, bool aIncl);

/// Facilty to GenIntervalSelector : right interval  ..+infinity[
template<class Type> cSelector<Type> RightHalfIntervalSelector(const Type & aV, bool aIncl);
/// Facilty to GenIntervalSelector : interval with no infinity
template<class Type> cSelector<Type> IntervalSelector(const Type&aV1,const Type&aV2,bool aInclLow,bool InclUp);
/// Constant selector, always true or false
template<class Type> cSelector<Type> CsteSelector  (bool aVal);
/**  convert a string into an union of interval

     "],5] [8,9[ ]12,15] ]17,]" => (x<=5) || (x>=8 && x<9) || ...
*/
template <class Type> cSelector<Type> Str2Interv(const std::string & aStr);


/* ============================================= */
/*                                               */
/*      String  Selector specialization          */
/*                                               */
/* ============================================= */

typedef cSelector<std::string> tNameSelector;
tNameSelector  BoostAllocRegex(const std::string& aRegEx);

/// Exract name of files located in the directory, by return value
std::vector<std::string>  GetFilesFromDir(const std::string & aDir,tNameSelector );
/// Exract name of files, by ref
void GetFilesFromDir(std::vector<std::string>&,const std::string & aDir,tNameSelector);
/// Recursively exract name of files located in the directory, by return value
void RecGetFilesFromDir( std::vector<std::string> & aRes, const std::string & aDir, tNameSelector  aNS,int aLevMin, int aLevMax);
/// Recursively exract name of files, by return value
std::vector<std::string> RecGetFilesFromDir(const std::string & aDir,tNameSelector  aNS,int aLevMin, int aLevMax);


#if (0)
//===============================
class cNameSelector : public cMemCheck
{
    public :
        virtual bool Match(const std::string &) const = 0;
        virtual ~cNameSelector();
};


/// Interface class to Regex

/** This class is a purely interface class to regex.
    I separate from the implementation :
       * because I am not 100% sure I will rely on boost
       * I don't want to slow dow all compliation witrh boost (or other) header 
       * I think it's good policy that class show the minimum required in header
*/

class cInterfRegex : public cNameSelector
{
    public :
        cInterfRegex(const std::string &);
        virtual ~cInterfRegex();
        static cInterfRegex * BoostAlloc();
        const std::string & Name() const;
    private :
        std::string mName;
};

/// Exract name of files located in the directory, by return value
std::vector<std::string>  GetFilesFromDir(const std::string & aDir,cNameSelector &);
/// Exract name of files, by ref
void GetFilesFromDir(std::vector<std::string>&,const std::string & aDir,cNameSelector &);
/// Recursively exract name of files located in the directory, by return value
void RecGetFilesFromDir( std::vector<std::string> & aRes, const std::string & aDir, cNameSelector & aNS,int aLevMin, int aLevMax);
/// Recursively exract name of files, by return value
std::vector<std::string> RecGetFilesFromDir(const std::string & aDir,cNameSelector & aNS,int aLevMin, int aLevMax);
#endif



};

#endif  //  _MMVII_Util_TPL_H_
