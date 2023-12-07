#ifndef GLOBAL_H
#define GLOBAL_H

#include <vector>
#include <set>
#include <algorithm>
#include <QString>
#include <QStringList>
#include <QTextStream>


constexpr const char *APP_ORGANIZATION = "IGN-LASTIG";
constexpr const char *APP_NAME = "vMMVII";
constexpr const char *APP_VERSION = "1.0";

constexpr const char *MMVII_EXE_FILE = "MMVII";
constexpr const char *MMVII_LOG_FILE = "MMVII-LogFile.txt";

constexpr const char *MMVII_VMMVII_SPEC_ARG = "ExecFrom=vMMVII";


constexpr const char *OUTPUT_CONSOLE_INFO_COLOR = "blue";
constexpr const char *OUTPUT_CONSOLE_ERROR_COLOR = "red";

extern bool showDebug;


#if QT_VERSION < QT_VERSION_CHECK(5, 14, 0)
namespace Qt
{
    static auto endl = ::endl;
}
#endif



typedef std::vector<QString> StrList;
typedef std::set<QString> StrSet;

QString quotedArg(const QString& arg);

template<class Container>
Container parseList(const QString &lv);

template<class Container,typename T>
bool contains(const Container& c,const T &item);

template<class Container, typename T>
bool contains(const Container& c, const std::initializer_list<T>& items);


/****************************************************************************
* Implementation
****************************************************************************/

template<class Container>
Container parseList(const QString &lv)
{
    Container c;

    QStringList ls;
    if (lv.size()>=2 && lv.front() == '['  && lv.back() == ']')
        ls = lv.mid(1,lv.size()-2).split(',');
    else
        ls = lv.split(',');

    for (const auto &s : std::as_const(ls))
        c.insert(c.end(),s);

    return c;
}


template<class Container, typename T>
bool contains(const Container &c, const T &item)
{
    return std::find(std::begin(c),std::end(c),item) != std::end(c);
}

template<class Container, typename T>
bool contains(const Container &c, const std::initializer_list<T> &items)
{
    for (const auto& s: items)
        if (contains(c,s))
            return true;
    return false;
}

template<class Iterable, class UnaryPredicate>
bool anyMatch(const Iterable &c, UnaryPredicate p)
{
    for (const auto& it: c) {
        if (p(it))
            return true;
    }
    return false;
}



#endif // GLOBAL_H
