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
       static std::string ToS(const Type &);     
      /// String -> Atomic object
       static Type  FromS(const std::string &);  
};

/// Do the test using only specialization ...

template <>  std::string cStrIO<int>::ToS(const int & anI);
template <>  int cStrIO<int>::FromS(const std::string & aStr);
template <>  std::string cStrIO<double>::ToS(const double & anI);
template <>  double cStrIO<double>::FromS(const std::string & aStr);
template <>  std::string cStrIO<std::string>::ToS(const std::string & anI);
template <>  std::string cStrIO<std::string>::FromS(const std::string & aStr);



#endif  //  _MMVII_Stringifier_H_
