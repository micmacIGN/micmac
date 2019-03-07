#ifndef  _MMVII_Util_TPL_H_
#define  _MMVII_Util_TPL_H_

namespace MMVII
{

/** \file MMVII_util_tpl.h
    \brief Utilitaries for some interface to template type

*/

template <class Type> class cExtSet ;
template <class Type> class cDataExtSet ;
template <class Type> class cSelector ;
template <class Type> class cDataSelector ;


/* ============================================= */
/*                                               */
/*            cSelector                          */
/*                                               */
/* ============================================= */

/// Access class to Selector
/** A selector is just a boolean predicate,
      Selector are used via smart pointer.
   Also the function "BenchSelector" illustrates all the
 functionnalities of selectors.
*/

template <class Type> class  cSelector 
{
     public :

          virtual bool Match(const Type &) const;
          cSelector(cDataSelector<Type> *);
     protected :
          std::shared_ptr<cDataSelector<Type> > mDS;
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

/**  convert a string into an union of interval. For example :

     "],5] [8,9[ ]12,15] ]17,]" => (x<=5) || (x>=8 && x<9) || ...
*/
template <class Type> cSelector<Type> Str2Interv(const std::string & aStr);


/* ============================================= */
/*                                               */
/*            cExtSet                            */
/*                                               */
/* ============================================= */


///  Bench some sets functionnalities
void BenchSet(const std::string & aDir);

///  Interface to sets services (set in extension)

/** Derived class will implement the services on :
 
    * std::unordered_set (done)
    * std::set  probably
    * std::vector
    * maybe hybrid (i.e. type that may change when size grows)
 
    See also  "void BenchSet(const std::string & aDir)" for yy

*/

template <class Type> class cExtSet  : public  cSelector<Type>
{
    public :
         ~cExtSet() ;
         cExtSet<Type>   Dupl() const ; // return a duplicata
         cExtSet<Type>   EmptySet() const ; // return a set from same type


         bool IncludedIn(const  cExtSet &) const; ///< Is this set in include in other ?
         bool Equal(const  cExtSet &) const;      ///< By double inclusion
         // static cExtSet<Type>  AllocUS(); ///<  Allocator, return empty set implemanted by unordered_set

         bool Add(const Type &) ;
         bool In(const Type &) const ;
         bool Suppress(const Type &)  ;
         void    clear() ;
         int    size() const ;
         void Filter(const cSelector<Type> &);

         virtual void  PutInVect(std::vector<const Type *> &,bool Sorted) const ; ///< Some type requires iteration 


         // cExtSet<Type>   operator=(const cExtSet<Type> &); => Basic semantic, = shared the same pointed value
         cExtSet(eTySC = eTySC::US);
         bool IsInit() const;
         bool Match(const Type &) const override ;
         void OpAff(const eOpAff &,const cExtSet<Type> aS);
    private :
         cExtSet(cDataExtSet<Type> *);
         std::shared_ptr<cDataExtSet<Type> > mDES;
         // void AssertInit();
};


template <class Type> void operator *= (cExtSet<Type> &,const cExtSet<Type> &);
template <class Type> void operator += (cExtSet<Type> &,const cExtSet<Type> &);
template <class Type> void operator -= (cExtSet<Type> &,const cExtSet<Type> &);

template <class Type> cExtSet<Type> operator * (const cExtSet<Type> & aS1,const cExtSet<Type> & aS2);
template <class Type> cExtSet<Type> operator + (const cExtSet<Type> & aS1,const cExtSet<Type> & aS2);
template <class Type> cExtSet<Type> operator - (const cExtSet<Type> & aS1,const cExtSet<Type> & aS2);

/// Sort on pointed value and not adress  
template <class Type> void SortPtrValue(std::vector<Type*> &aV);


// Xml or pat
tNameSet SetNameFromPat    (const std::string&); ///< create a set of file from a pattern
tNameSet SetNameFromFile   (const std::string&, int aNumV); ///< create from a file xml, V1 or V2
tNameSet SetNameFromString (const std::string&, bool AllowPat); ///< general case, try to recognize automatically V1, V2 or pattern

/** read from file, select version, accept empty, error if file exist bud in bad format */
tNameRel  RelNameFromFile (const std::string&);


/** indicate when file to good format exist, if not no error return emty */
tNameRel  RelNameFromXmlFileIfExist 
          (
              const std::string& aNameFile, ///< Name of file
              bool &Exist  ///< indicate if file exist, used by RelNameFromFile
          ); 


/* ================================================ */
/*                                                  */
/*                 cOrderedPair                     */
/*                                                  */
/* ================================================ */

/// Pair where we want (a,b) == (b,a)

/** cOrderedPair are pretty match like pair<T,T> ,
    the main difference being that they modelise symetric graph.
    To assure that they are always in a single way, we
    force V1 <= V2
*/
template <class Type> class cOrderedPair
{
      public :
           typedef cOrderedPair<Type> value;
           cOrderedPair(const Type & aV1,const Type & aV2); ///< Pair will be reordered
           cOrderedPair(); ///< Default constructor, notably for serializer
           bool operator < (const cOrderedPair<Type> & aP2) const;
           bool operator == (const cOrderedPair<Type> & aP2) const;
           const Type & V1() const;
           const Type & V2() const;
           Type & V1() ;
           Type & V2() ;


      private :
           static const Type &  Min4OP(const Type & aV1,const Type & aV2);
           Type mV1;
           Type mV2;
};




};

#endif  //  _MMVII_Util_TPL_H_
