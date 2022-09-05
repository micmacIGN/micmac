#ifndef  _MMVII_Util_TPL_H_
#define  _MMVII_Util_TPL_H_

#include <algorithm>

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
template<class Type> cSelector<Type> GenIntervalSelector( const std::optional<Type> & aV1,
                                                       const std::optional<Type> & aV2,
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
         virtual ~cExtSet() ;
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
std::vector<std::string>  ToVect(const tNameSet &);  ///< Less economic but more convenient than PutInVect

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


#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny)
#define  M_VectorAt(V,aK) (V.at(aK))
#else
#define  M_VectorAt(V,aK) (V[aK])
#endif 


/* *************************************************** */
/*                                                     */
/*   Complement to STL, thing in C++20 and not 14 .... */
/*                                                     */
/* *************************************************** */

template<class TCont,class TVal> bool  BoolFind(const TCont & aCont,const TVal & aVal)
{
    return std::find(aCont.begin(),aCont.end(),aVal) != aCont.end();
}

template <class TV,class TF> void erase_if(TV & aVec,const TF& aFonc)
{
   aVec.erase(std::remove_if(aVec.begin(),aVec.end(),aFonc),aVec.end());
}

/// return -1 0 or 1 , regarding < , == or >
template <class Type> int LexicoCmp(const std::vector<Type> & aV1,const std::vector<Type> & aV2);
/// return if aV1 < aV2 as LexicoCmp==-1
template <class Type> bool operator < (const std::vector<Type> & aV1,const std::vector<Type> & aV2);
template <class Type> bool operator == (const std::vector<Type> & aV1,const std::vector<Type> & aV2);
template <class Type> bool operator != (const std::vector<Type> & aV1,const std::vector<Type> & aV2);



template <class Type> void AppendIn(std::vector<Type> & aRes, const std::vector<Type> & aVec)
{
   for (const auto & aVal : aVec)
       aRes.push_back(aVal);
}

template <class Type> void Append(std::vector<Type> & aRes, const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
   aRes = aV1;
   AppendIn(aRes,aV2);
   // for (const auto & aVal : aV2) aRes.push_back(aVal);
}
template <class Type> std::vector<Type> Append(const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
    std::vector<Type> aRes;
    Append(aRes,aV1,aV2);
    return aRes;
}

///  set value or push at end, 
template <class Type> void SetOrPush(std::vector<Type> & aVec,size_t aIndex,const Type & aVal)
{
     MMVII_INTERNAL_ASSERT_tiny(aIndex<=aVec.size(),"Bad size for SetOrPush");
     if (aIndex==aVec.size())
        aVec.push_back(aVal);
     else
        aVec.at(aIndex) = aVal;
}


//  resize only in increasing mode
template <class Type> void ResizeUp(std::vector<Type> & aV1,size_t aSz,const Type &aVal)
{
   if (aSz>aV1.size())
      aV1.resize(aSz,aVal);
}


///  compute the cartesian product of all seq {12,A,xyz} ->  {1Ax,2Ax,1Ay....} 
template <class Type>  std::vector<std::vector<Type>>  ProdCart(const std::vector<std::vector<Type>>  aVV)
{
     std::vector<std::vector<Type>> aVVRes;
     aVVRes.push_back(std::vector<Type>()); /// initialize with neutral elemnt for cart prod of setq {{}} 
     for (const auto & aVNewV : aVV)  // for each input sequence
     {
          std::vector<std::vector<Type>>  aNewVVRes;
          for (const auto & aVR : aVVRes)  // for each sequence already compute
          {
              for (const auto & aNewVal : aVNewV)  // add each val to existing sequence
              {
                  std::vector<Type> aNewVect = aVR;
                  aNewVect.push_back(aNewVal);
                  aNewVVRes.push_back(aNewVect);
              }
          }
          aVVRes = aNewVVRes;
     }
     return aVVRes;
}

template <class TVal,class TFunc> void SortOnCriteria(std::vector<TVal> & aVec,const TFunc & aFunc)
{   
    std::sort
    (
         aVec.begin(),aVec.end(),
         [&aFunc](const auto & aV1,const auto & aV2) {return aFunc(aV1) < aFunc(aV2);}
    );
}



};

#endif  //  _MMVII_Util_TPL_H_
