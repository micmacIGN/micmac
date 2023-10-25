#include "global.h"
#include <QRegularExpression>

bool showDebug = false;

QString quotedArg(const QString& arg)
{
    static const  QRegularExpression re("[][()<>{}*?|$^%!#~;`&' \t\"]");
    QString quote;
    QString quotedArg;

    if (arg.isEmpty())
        return "\"\"";

    if (arg.contains(re))
        quote = "\"";

    quotedArg = quote;
    for (const auto& c : arg) {
        if (c == '\\' || c == '$' || c == '"')
            quotedArg += "\\";
        quotedArg += c;
    }
    quotedArg += quote;
    return quotedArg;
}

