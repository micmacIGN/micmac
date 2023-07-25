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

static StringList parseLine(const std::string line)
{
    std::vector<std::string> result;
    size_t pos = 0;
    size_t wordStart;
    size_t wordEnd;

    // Nota: std::string is garanteed to be null-terminated (C++ 11)
    while (::isspace(line[pos])) pos++;
    if (line[pos] == 0 || line[pos] == '#')     // empty or comment line
        return result;

    while (true) {
        wordStart = pos;
        while (line[pos] != ',' && line[pos] != 0) pos++;
        wordEnd = pos;
        if (wordEnd > wordStart) {
            while (::isspace(line[--wordEnd]));
            result.push_back(line.substr(wordStart, wordEnd - wordStart+1));
        } else {
            result.push_back("");
        }
        if (line[pos] == 0)
            return result;
        pos++;
        while (::isspace(line[pos])) pos++;
    }
}


void csvParse(const Path &path, std::vector<int> nbValues,
              std::function<bool(const StringList& values, const std::string& fName, unsigned line)> f)
{
    std::string line;
    std::ifstream is(path);
    unsigned nLine = 0;
    bool ok;

    if (! is) {
        errorf(Error,"Can't read file '%s'.",path.string().c_str());
        return;
    }
    while (getline(is, line)) {
        nLine++;
        auto values = parseLine(line);
        if (values.size() == 0)
            continue;
        ok=false;
        for (auto nbVal : nbValues) {
            if (nbVal > 0 && (int)values.size() == nbVal) {
                ok = true;
                break;
            }
            if (nbVal < 0 && (int)values.size() >= -nbVal) {
                ok = true;
                break;
            }
        }
        if (! ok)
            errorf(Error,"In '%s', line %u has an incorrect number of fields.",path.string().c_str(),nLine);
        if (! f(values,path.string(),nLine))
            return;
    }
}




} // namespace Kapture  va_list vl,vl_count;
