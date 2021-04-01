#include "kpt_internal.h"
#include <cstdarg>
#include <cstdio>
#include <fstream>

namespace Kapture {

std::string formatErrorMsg(const char *fmt, ...)
{
    static char errorBufferMsg[1024];
    va_list args;
    va_start(args,fmt);

    std::vsnprintf(errorBufferMsg,sizeof(errorBufferMsg)-1,fmt,args);
    va_end(args);

    return std::string(errorBufferMsg);
}

bool strICaseEqual(const std::string &a, const std::string &b)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](unsigned char a, unsigned char b) {
        return toupper(a) == toupper(b);
    });
}


void PosixLocale::begin()
{
    if (! posixLocale)
        posixLocale = newlocale(LC_ALL_MASK,"C",(locale_t(0)));
    if (! oldLocale)
        oldLocale = uselocale(posixLocale);
}

void PosixLocale::end()
{
    if (oldLocale) {
        uselocale(oldLocale);
        oldLocale = (locale_t)0;
    }
}




} // namespace Kapture  va_list vl,vl_count;
