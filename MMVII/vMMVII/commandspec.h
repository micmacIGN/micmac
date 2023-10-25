#ifndef COMMANDSPEC_H
#define COMMANDSPEC_H

#include <functional>
#include <QString>
#include <QObject>
#include <QVector>
#include <QMap>
#include <stdexcept>

struct ArgSpec
{
    enum Type {T_UNKNOWN, T_CHAR, T_BOOL, T_INT, T_DOUBLE, T_STRING, T_VEC_DOUBLE, T_VEC_INT, T_VEC_STRING, T_PTXD2_INT, T_PTXD3_INT, T_PTXD2_DOUBLE, T_BOX2_INT, T_ENUM};


    ArgSpec(bool mandatory=true) : mandatory(mandatory),hasInitValue(false) {}

    bool mandatory;
    bool isEnabled;
    int number;
    QString name;
    QString level;
    QString cppTypeStr;
    Type cppType;
    QString def;
    QString comment;
    QStringList semantic;
    QStringList allowed;
    QString range;
    int vSizeMin,vSizeMax;

    QString value;
    bool hasInitValue;
    QString initValue;
    bool check;
    QString json;
};


struct CommandSpec
{
public:
    CommandSpec() {}
    bool empty() { return name.isEmpty();}
    bool fromJson(const QString& file, const QString& command, QString& errorMsg);
    bool initFrom(const CommandSpec& spec);

    QString name;
    QString comment;
    QString source;

    QVector<ArgSpec> mandatories;
    QVector<ArgSpec> optionals;

};


struct MMVIISpecs : public QObject {
    Q_OBJECT
public:
    void fromJson(const QByteArray &specsTxt);

    QString errorMsg;

    QMap<QString,CommandSpec> commands;

    QString mmviiBin;
    QString phpDir;
    QString orientDir;
    QString homolDir;
    QString meshDevDir;
    QString radiomDir;
    QString testDir;
    QMap <QString,QStringList> extensions;

private:
    void error(const QString& msg);
    void parseConfig(const QJsonObject &config);
    void parseCommands(const QJsonArray &applets);
    QString toString(const QJsonObject& obj, const QString& key, const QString& context, bool needed = true);
    QStringList toStringList(const QJsonObject &obj, const QString &key, const QString &context, bool needed = true);
    QVector<ArgSpec> parseArgsSpecs(const QJsonObject &argsSpecs, const QString &key, QString context, const QString& command);
};

class ParseJSonException : public std::runtime_error
{
public:
    ParseJSonException(const char * msg) : std::runtime_error(msg) {}

};

QStringList parseList(const QString &lv);


#endif // COMMANDSPEC_H
