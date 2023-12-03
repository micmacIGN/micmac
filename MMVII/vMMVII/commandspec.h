#ifndef COMMANDSPEC_H
#define COMMANDSPEC_H

#include <functional>
#include <QString>
#include <QObject>
#include <QVector>
#include <QMap>
#include <stdexcept>
#include <set>

#include "global.h"

class eTA2007 {
public:
    enum Enum {
        DirProject,    ///< Exact Dir of Proj
        FileDirProj,   ///< File that define the  Dir Proj
        MPatFile,      ///< Major PaternIm => "" or "0" in sem for set1, "1" or other for set2
        FFI,           ///< File Filter Interval
        Input,         ///< Is this parameter used as input/read
        Output,        ///< Is this parameter used as output/write
        OptionalExist, ///< if given, the file (image or other) can be unexisting (interface mut allow seizing "at hand")
        PatParamCalib, ///< It's a pattern for parameter of calibration
        AddCom,        ///< Not an attribute, used to embed additionnal comment in Help mode
        AllowedValues, ///< String of possible values for enums type, automagically added for args of enum type
        Shared,        ///< Parameter  Shared by many (several) command
        Global,        ///< Parameter  Common to all commands
        Internal,      ///< Reserved to internall use by MMVII
        Tuning,        ///< Used for testing/tuning command but not targeted for user
        HDV,           ///< Has Default Value, will be printed on help
        ISizeV,        ///< Interval size vect, print on help
        XmlOfTopTag,   ///< Parameter must be a XML-file containing certain tag
        Range,         ///< Range of allowed numerical values: "[min,max]" | "[min,]" | "[,max]"
        vMMVII_FilesType,
        vMMVII_PhpPrjDir,
    };

    static Enum val(const QString &str);
    static QString str(Enum e);

private:
    static const std::map<Enum,QString> enumMap;
    static std::map<QString,Enum> strMap;
};

struct ArgSpec
{
    enum Type {T_UNKNOWN, T_CHAR, T_BOOL, T_INT, T_DOUBLE, T_STRING, T_VEC_DOUBLE, T_VEC_INT, T_VEC_STRING, T_PTXD2_INT, T_PTXD3_INT, T_PTXD2_DOUBLE, T_PTXD3_DOUBLE, T_BOX2_INT, T_ENUM};


    ArgSpec(bool mandatory=true) : mandatory(mandatory),hasInitValue(false) {}

    bool mandatory;
    bool isEnabled;
    int number;
    QString name;
    QString level;
    QString cppTypeStr;
    QString phpPrjDir;
    QString fileType;
    Type cppType;
    QString def;
    QString comment;
    std::set<eTA2007::Enum> semantic;
    StrSet allowed;
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
    StrList features;
    StrList inputs;
    StrList outputs;

    std::vector<ArgSpec> mandatories;
    std::vector<ArgSpec> optionals;
    QString json;
};


struct MMVIISpecs : public QObject {
    Q_OBJECT
public:
    void fromJson(const QByteArray &specsTxt);

    QString errorMsg;

    QMap<QString,CommandSpec> commands;

    QString mmviiBin;
    QString phpDir;
    QString testDir;
    StrList dirTypes;
    StrList fileTypes;
    QMap <QString,StrList> extensions;
    StrList allowed;
    StrList denied;

private:
    void error(const QString& msg);
    void parseConfig(const QJsonObject &config);
    void parseCommands(const QJsonArray &applets);
    QString toString(const QJsonObject& obj, const QString& key, const QString& context, bool needed = true);
    template<class Container>
    Container toStringList(const QJsonObject &obj, const QString &key, const QString &context, bool needed = true);
    void parseETA2007Set(ArgSpec& as, const QJsonObject& obj, const QString& key, const QString& context);
    std::vector<ArgSpec> parseArgsSpecs(const QJsonObject &argsSpecs, const QString &key, QString context, const QString& command);
};

class ParseJSonException : public std::runtime_error
{
public:
    ParseJSonException(const char * msg) : std::runtime_error(msg) {}

};


#endif // COMMANDSPEC_H
