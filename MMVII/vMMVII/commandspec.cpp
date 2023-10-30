#include "commandspec.h"
#include "global.h"
#include <QFile>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QJsonArray>
#include <QJsonObject>
#include <iostream>


const std::map<eTA2007::Enum,QString> eTA2007::enumMap=
    {
        {eTA2007::DirProject,"DP"},
        {eTA2007::FileDirProj,"FDP"},
        {eTA2007::FileImage,"Im"},
        {eTA2007::FileCloud,"Cloud"},
        {eTA2007::File3DRegion,"3DReg"},
        {eTA2007::MPatFile,"MPF"},
        {eTA2007::Orient,"Ori"},
        {eTA2007::RadiomData,"RadData"},
        {eTA2007::RadiomModel,"RadModel"},
        {eTA2007::MeshDev,"MeshDev"},
        {eTA2007::Mask,"Mask"},
        {eTA2007::MetaData,"MetaData"},
        {eTA2007::PointsMeasure,"PointsMeasure"},
        {eTA2007::TieP,"TieP"},
        {eTA2007::MulTieP,"MulTieP"},
        {eTA2007::Input,"In"},
        {eTA2007::Output,"Out"},
        {eTA2007::OptionalExist,"OptEx"},
        {eTA2007::PatParamCalib,"ParamCalib"},
        {eTA2007::AddCom,"AddCom"},
        {eTA2007::AllowedValues,"Allowed"},
        {eTA2007::Internal,"##Intern"},
        {eTA2007::Tuning,"##Tune"},
        {eTA2007::Global,"##Glob"},
        {eTA2007::Shared,"##Shar"},
        {eTA2007::HDV,"##HDV"},
        {eTA2007::ISizeV,"##ISizeV"},
        {eTA2007::XmlOfTopTag,"##XmlOfTopTag"},
        {eTA2007::Range,"##Range"},
        {eTA2007::FFI,"FFI"}
};

std::map<QString, eTA2007::Enum> eTA2007::strMap;

QString eTA2007::str(eTA2007::Enum e)
{
    return enumMap.at(e);
}

eTA2007::Enum eTA2007::val(const QString& s)
{
    if (strMap.empty()) {
        for (const auto& m: enumMap)
            strMap[m.second] = m.first;
    }
    return strMap[s];
}


static void typeNameToEnum(ArgSpec& as, const QString& command)
{
    static const std::map<QString, ArgSpec::Type> typeNameEnumMap = {
        {"char",ArgSpec::T_CHAR},
        {"bool",ArgSpec::T_BOOL},
        {"int", ArgSpec::T_INT},
        {"size_t", ArgSpec::T_INT},
        {"double",ArgSpec::T_DOUBLE},
        {"string",ArgSpec::T_STRING},
        {"std::vector<int>",ArgSpec::T_VEC_INT},
        {"std::vector<double>",ArgSpec::T_VEC_DOUBLE},
        {"std::vector<std::string>",ArgSpec::T_VEC_STRING},
        {"cPtxd<int,2>",ArgSpec::T_PTXD2_INT},
        {"cPtxd<int,3>",ArgSpec::T_PTXD3_INT},
        {"cPtxd<double,2>",ArgSpec::T_PTXD2_DOUBLE},
        {"cTplBox<int,2>",ArgSpec::T_BOX2_INT}
    };
    static QStringList unknows;

    auto it = typeNameEnumMap.find(as.cppTypeStr);
    if (it != typeNameEnumMap.end()) {
        as.cppType = it->second;
    } else if (as.allowed.size() != 0) {
        as.cppType = ArgSpec::T_ENUM;
    } else {
        as.cppType = ArgSpec::T_UNKNOWN;
        if (! unknows.contains(as.cppTypeStr)) {
            unknows.append(as.cppTypeStr);
            QString name = as.name;
            if (as.mandatory)
                name = QString("ArgObl#%1").arg(as.number - 1);
            std::cerr << APP_NAME << ": unknown C++ type '" << qPrintable(as.cppTypeStr)
                      << "' for " << qPrintable(command) << "." << qPrintable(name) << std::endl;
        }
    }
}



void MMVIISpecs::error(const QString &msg)
{
    errorMsg = msg;
    throw ParseJSonException("Error parsing JSON file");
}

QString MMVIISpecs::toString(const QJsonObject &obj, const QString &key, const QString &context, bool needed)
{
    if (!obj.contains(key)) {
        if (!needed)
            return "";
        error(tr("Missing key \"%1\" %2").arg(key, context));
    }
    if (!obj[key].isString())
        error (tr("Key \"%1\" is not a string %2").arg(key, context));
    return obj[key].toString();
}


template<class Container>
Container MMVIISpecs::toStringList(const QJsonObject& obj, const QString& key, const QString& context, bool needed)
{
    Container val;

    if (!obj.contains(key)) {
        if (!needed)
            return val;
        error (tr("Missing key \"%1\" %2").arg(key, context));
    }
    if (!obj[key].isArray())
        error(tr("Key \"%1\" is not an array of string %2").arg(key, context));

    const QJsonArray a = obj[key].toArray();
    for (const auto& v : a) {
        if (! v.isString())
            error (tr("Key \"%1\" is not an array of string  %2").arg(key, context));
        val.insert(val.end(),v.toString());
    }
    return val;
}

void MMVIISpecs::parseConfig(const QJsonObject& config)
{
    QString context = tr("in \"config\" at top level");
    mmviiBin  = toString(config,"Bin2007",context);
    phpDir    = toString(config,"MMVIIDirPhp",context);
    testDir   = toString(config,"MMVIITestDir",context);
    fileTypes = toStringList<StrSet>(config,"eTa2007FileTypes",context);
    dirTypes  = toStringList<StrSet>(config,"eTa2007DirTypes",context);

    if (!config.contains("extensions"))
        error(tr("Missing key \"extensions\" in \"config\" at top level"));
    if (! config["extensions"].isObject())
        error(tr("\"extensions\" is not an object in \"config\" at top level"));
    const auto extensionsSpecs = config["extensions"].toObject();
    for (auto& key: extensionsSpecs.keys())
        extensions[key] = toStringList<StrList>(extensionsSpecs,key,"\"extensions\" in \"config\"");
}

std::vector<ArgSpec> MMVIISpecs::parseArgsSpecs(const QJsonObject& argsSpecs, const QString& key, QString context, const QString& command)
{
    std::vector<ArgSpec> vSpecs;

    if (! argsSpecs.contains(key))
        error(tr("Missing key \"%1\" %2").arg(key, context));

    if (! argsSpecs[key].isArray())
        error(tr("Key '%1' must be an array %2").arg(key, context));

    const QJsonArray specsArray = argsSpecs[key].toArray();
    int n=0;
    for (const auto& s : specsArray) {
        n++;
        if (!s.isObject())
            error (tr("%1 element #%2 is not a JSON object %3").arg(key).arg(n).arg(context));

        QJsonObject spec = s.toObject();
        ArgSpec as(key == "mandatory");
        as.json = QJsonDocument(spec).toJson();
        as.isEnabled = as.mandatory;
        as.number = n;
        context = tr("%1 argument #%2 ").arg(key).arg(n) + context;
        as.name       = toString(spec,"name",context,key != "mandatory");
        as.level      = toString(spec,"level",context,false);
        as.cppTypeStr = toString(spec,"type",context);
        as.def        = toString(spec,"default",context,false);
        as.comment    = toString(spec,"comment",context,false);
        as.semantics  = toStringList<StrSet>(spec,"semantic",context,false);
        for (const auto &s: std::as_const(as.semantics))
            as.semantic.insert(eTA2007::val(s));

        as.allowed    = toStringList<StrSet>(spec,"allowed",context,false);
        as.range      = toString(spec,"range",context,false);
        as.vSizeMin   = as.vSizeMax = 1;
        typeNameToEnum(as, command);
        QString vsize;
        vsize    = toString(spec,"vsize",context,false);
        if (vsize.length()) {
            auto vsizel = parseList<StrList>(vsize);
            if (vsizel.size() == 2) {
                bool ok1,ok2;
                as.vSizeMin = vsizel[0].toInt(&ok1);
                as.vSizeMax = vsizel[1].toInt(&ok2);
                if (ok1 && !ok2)
                    as.vSizeMax = as.vSizeMin;
                else if (!ok1 && ok2)
                    as.vSizeMin = as.vSizeMax;
                else if (!ok1 && !ok2)
                    as.vSizeMin = as.vSizeMax = 1;
            }
        }

        vSpecs.push_back(as);
    }
    return vSpecs;
}


void MMVIISpecs::parseCommands(const QJsonArray& applets)
{
    for (const auto& applet: applets) {
        CommandSpec cmdSpec;
        if (! applet.isObject())
                error(tr("Invalid JSON specification"));
        auto theSpec = applet.toObject();
        cmdSpec.name = toString(theSpec,"name","\"applets\" array");

        QString context = tr(" in \"applets\" array");
        auto features = toStringList<StrSet>(theSpec,"features",context);
        if (contains(features,"nogui"))
            continue;
        cmdSpec.name = toString(theSpec,"name",context);
        context = " in command \"" + cmdSpec.name + "\"";
        cmdSpec.comment = toString(theSpec,"comment",context);
        cmdSpec.source = toString(theSpec,"source",context,false);
        cmdSpec.mandatories = parseArgsSpecs(theSpec, "mandatory", context, cmdSpec.name);
        cmdSpec.optionals = parseArgsSpecs(theSpec, "optional", context, cmdSpec.name);
        commands[cmdSpec.name] = cmdSpec;
    }
}


void MMVIISpecs::fromJson(const QByteArray& specsTxt)
{
    if (specsTxt.isNull())
        error(tr("Empty JSON specifications"));

    QJsonParseError jError;
    QJsonDocument jDoc = QJsonDocument::fromJson(specsTxt,&jError);

    if (jError.error != QJsonParseError::NoError)
        error(tr("Error parsing JSON specifications : \n%1 at %2")
              .arg(jError.errorString())
              .arg(jError.offset));

    if (jDoc.isNull())
        error(tr("Empty JSON specifications"));

    if (! jDoc.isObject())
        error(tr("Expecting a full JSON MMVII arguments specifications "));

    if (! jDoc.object().contains("config"))
        error(tr("Missing key \"config\" in top level object"));
    auto config = jDoc.object().value("config");
    if (! config.isObject())
        error(tr("\"config\" is not a JSON object in the top level object"));
    parseConfig(config.toObject());

    if (! jDoc.object().contains("applets"))
        error(tr("Missing key \"applets\" in top level object"));
    auto applets = jDoc.object().value("applets");
    if (! applets.isArray())
        error(tr("\"applets\" is not a JSON array in the top level object"));
    parseCommands(applets.toArray());
}

bool CommandSpec::initFrom(const CommandSpec &spec)
{
    if (spec.name != name ||
        spec.mandatories.size() != mandatories.size() ||
        spec.optionals.size() != optionals.size())
        return false;

    for (unsigned i=0; i<mandatories.size(); i++) {
        mandatories[i].hasInitValue = spec.mandatories[i].hasInitValue;
        if (spec.mandatories[i].hasInitValue)
            mandatories[i].initValue = spec.mandatories[i].initValue;
    }

    for (unsigned i=0; i<optionals.size(); i++) {
        optionals[i].hasInitValue = spec.optionals[i].hasInitValue;
        if (spec.optionals[i].hasInitValue)
            optionals[i].initValue = spec.optionals[i].initValue;
    }
    return true;
}

