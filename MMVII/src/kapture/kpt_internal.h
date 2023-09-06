#ifndef KPT_INTERNAL_H
#define KPT_INTERNAL_H

#include <string>
#include <vector>
#include <chrono>
#include <ostream>
#include <functional>

#include "kpt_common.h"

#define dbg             if (Kapture::debugOn) fprintf(stderr,"%d\n",__LINE__)
#define debug(fmt,...)  if (debugOn) fprintf(stderr,"Line %u: " fmt "\n",__LINE__, __VA_ARGS__)

namespace Kapture {


template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    const char *sep = "";
    for (const auto& e : v) {
        os << sep << e ;
        sep = ",";
    }
    os << "]";
    return os;
}

static inline std::ostream& operator<<(std::ostream& os, const DType& dt)
{
    os << dtypeToStr(dt);
    return os;
}


std::string formatErrorMsg(const char *fmt, ...) __attribute__ ((format(printf,1,2))) ;

#define error(except,msg)       throw except(msg, __FILE__,__LINE__, __func__ )
#define errorf(except,fmt,...)  throw except(formatErrorMsg(fmt,__VA_ARGS__),__FILE__,__LINE__, __func__ )

bool strICaseEqual(const std::string& a, const std::string& b);


class PosixLocale {
public:
    PosixLocale() : oldLocale((locale_t)0) { begin(); }
    ~PosixLocale() { end(); }
    void begin();
    void end();
private:
    locale_t oldLocale;
    static locale_t posixLocale;
};



class Chrono {
public:
    Chrono() {start();}
    void start() {mStart = std::chrono::high_resolution_clock::now();  }
    auto end() {
        auto end = std::chrono::high_resolution_clock::now();
        return (std::chrono::duration_cast<std::chrono::nanoseconds>(end - mStart)).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
};

template<typename  T>
class EnumAndStr
{
public:
    // First element must be default value
    EnumAndStr(std::initializer_list<std::pair<T,const char*>> l) : list(l) {}
    T fromStr(const std::string& s) const;
    const char* toStr(T n) const;

private:
    std::vector<std::pair<T,const char*>> list;
};

template<typename T>
T EnumAndStr<T>::fromStr(const std::string &s) const
{
    for (const auto& e : list) {
        if (strICaseEqual(e.second,s))
            return e.first;
    }
    if (! list.size())
        error(Error,"Internal Error: enum list has no element");
    return list[0].first;
}

template<typename T>
const char *EnumAndStr<T>::toStr(T n) const
{
    for (const auto& e : list) {
        if (n == e.first)
            return e.second;
    }
    if (! list.size() )
        error(Error,"Internal Error: enum list has no element");
    return list[0].second;
}


#define MAKE_ENUM_STR(x)    {x,#x}
#define MAKE_TENUM_STR(t,x) {t::x,#x}


void csvParse(const Path &path, std::vector<int> nbValues,
              std::function<bool(const StringList& values, const std::string& fName, unsigned line)> f);



} // namespace Kapture

#endif // KPT_INTERNAL_H
