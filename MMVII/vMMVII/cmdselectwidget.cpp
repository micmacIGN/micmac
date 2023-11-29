#include "cmdselectwidget.h"
#include "global.h"
#include "settings.h"

#include <QTextStream>
#include <QFile>
#include <QMenu>


CmdSelectWidget::CmdSelectWidget(const MMVIISpecs &allSpecs, QWidget *parent)
    : QWidget(parent),allSpecs(allSpecs)
{
    cmdSelectUi = new Ui::CmdSelectUI();
    cmdSelectUi->setupUi(this);
    
    for (const auto& spec : allSpecs.commands) {
        if (contains(spec.features,"NoGui"))
            continue;
        if (!allSpecs.allowed.empty() && !anyMatch(allSpecs.allowed,[&spec](const auto &e){return QRegExp(e,Qt::CaseInsensitive).exactMatch(spec.name); }))
            continue;
        if (anyMatch(allSpecs.denied,[&spec](const auto &e){return QRegExp(e,Qt::CaseInsensitive).exactMatch(spec.name); }))
            continue;
        QListWidgetItem *lwi = new QListWidgetItem(spec.name, cmdSelectUi->commandList);
        if (showDebug)
            lwi->setToolTip(spec.comment + "<pre>" + spec.json.toHtmlEscaped() + "</pre>");
        else
            lwi->setToolTip(spec.comment);
        
    }

    connect(cmdSelectUi->commandList,&QListWidget::itemSelectionChanged,this,&CmdSelectWidget::commandListSelChanged);
    connect(cmdSelectUi->commandList,&QListWidget::itemDoubleClicked,this,&CmdSelectWidget::doConfigure);
    connect(cmdSelectUi->historyList,&QListWidget::itemSelectionChanged,this,&CmdSelectWidget::historyListSelChanged);
    connect(cmdSelectUi->historyList,&QListWidget::itemDoubleClicked,this,&CmdSelectWidget::doConfigure);

    cmdSelectUi->commandList->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(cmdSelectUi->commandList, &QListWidget::customContextMenuRequested, this, &CmdSelectWidget::commandContextMenu);
    cmdSelectUi->historyList->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(cmdSelectUi->historyList, &QListWidget::customContextMenuRequested, this, &CmdSelectWidget::historyContextMenu);

    checkConfigureOk();
    return;
}

CmdSelectWidget::~CmdSelectWidget()
{
    delete cmdSelectUi;
}


static StrList parseCmdLine(const QString& line)
{
    enum State {WAIT_ARG, UNQUOTED, UNQUOTED_BACKSLASH, SINGLE_QUOTED, DOUBLE_QUOTED, DOUBLE_QUOTED_BACKSLASH};
    State state = WAIT_ARG;

    QString arg;
    StrList args;

    for (auto c : line) {
        switch (state) {
        case WAIT_ARG:
            if (c.isSpace())
                break;
            if (c == '\'') {
                state = SINGLE_QUOTED;
            } else if (c == '"') {
                state = DOUBLE_QUOTED;
            } else if (c == '\\') {
                state = UNQUOTED_BACKSLASH;
            } else {
                arg += c;
                state = UNQUOTED;
            }
            break;
        case SINGLE_QUOTED:
            if (c == '\'') {
                state = UNQUOTED;
            } else {
                arg += c;
            }
            break;
        case DOUBLE_QUOTED:
            if (c == '"') {
                state = UNQUOTED;
            } else if (c == '\\') {
                state = DOUBLE_QUOTED_BACKSLASH;
            } else {
                arg += c;
            }
            break;
        case DOUBLE_QUOTED_BACKSLASH:
            if (c != '$' && c != '`' && c != '"' && c != '\\' && c != '\n')
                arg += '\\';
            arg += c;
            state = DOUBLE_QUOTED;
            break;
        case UNQUOTED:
            if (c.isSpace()) {
                state = WAIT_ARG;
                args.push_back(arg);
                arg.clear();
                break;
            }
            if (c == '\'') {
                state = SINGLE_QUOTED;
            } else if (c == '"') {
                state = DOUBLE_QUOTED;
            } else if (c == '\\') {
                state = UNQUOTED_BACKSLASH;
            } else {
                arg += c;
            }
            break;
        case UNQUOTED_BACKSLASH:
            arg += c;
            state = UNQUOTED;
            break;
        }
    }
    if (arg.size())
        args.push_back(arg);
    return args;
}



CommandSpec CmdSelectWidget::parseArgs(const StrList& args, QString& newLine)
{
    CommandSpec spec;

    if (args.size() < 2)
        return CommandSpec();

    if (! args[0].endsWith(MMVII_EXE_FILE) && !args[0].endsWith(QString(MMVII_EXE_FILE) + ".exe"))
        return CommandSpec();

    QString command = args[1];

    if (!allSpecs.commands.contains(command))
        return CommandSpec();
    spec = allSpecs.commands[command];

    if (args.size() < 2 + spec.mandatories.size())
        return CommandSpec();

    newLine = "MMVII " + command;
    for (unsigned i=0; i<spec.mandatories.size(); i++) {
        spec.mandatories[i].hasInitValue = true;
        spec.mandatories[i].initValue = args[2+i];
        newLine += " " + quotedArg(args[2+i]);
    }

    for (unsigned i=0; i<spec.optionals.size(); i++) {
        spec.optionals[i].hasInitValue = false;
    }

    for (unsigned i=2+spec.mandatories.size(); i < args.size(); i++) {
        QStringList keyVal=args[i].split('=');
        if (keyVal.size()< 2)
            continue;
        for (unsigned j=0; j<spec.optionals.size(); j++) {
            if (keyVal[0] == spec.optionals[j].name) {
                spec.optionals[j].hasInitValue = true;
                spec.optionals[j].initValue = keyVal[1];
                for (int k=2; k<keyVal.size(); k++)
                    spec.optionals[j].initValue += "=" + keyVal[k];
                newLine += " " + quotedArg(args[i]);
                break;
            }
        }
    }
    return spec;
}

Q_DECLARE_METATYPE(CommandSpec);

void CmdSelectWidget::workingDirChanged()
{
    cmdSelectUi->historyList->clear();
    cmdSelectUi->historyList->clearSelection();
    QFile logFile( MMVII_LOG_FILE);
    if (! logFile.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QTextStream in(&logFile);
    while(true) {
        auto line = in.readLine();
        if (line.isNull())
            break;
        line = line.trimmed();
        if (line.isEmpty())
            continue;
        if (line.startsWith("===") ||
            line.startsWith("Id :") ||
            line.startsWith("begining at") ||
            line.startsWith("beginning at") ||
            line.startsWith("ending ") ||
            line.startsWith("ABORT ") ||
            line.startsWith("> "))
            continue;

        auto args = parseCmdLine(line);
        CommandSpec spec = parseArgs(args,line);
        if (spec.empty())
            continue;

        bool found=false;
        for (int i=0; i<cmdSelectUi->historyList->count(); i++) {
            if (cmdSelectUi->historyList->item(i)->text() == line) {
                auto* lwi = cmdSelectUi->historyList->takeItem(i);
                cmdSelectUi->historyList->insertItem(0,lwi);
                found = true;
                break;
            }
        }
        if (found)
            continue;

        QListWidgetItem *lwi = new QListWidgetItem(line);
        lwi->setData(Qt::UserRole, QVariant::fromValue(spec));
        lwi->setToolTip(line);
        cmdSelectUi->historyList->insertItem(0,lwi);
        while (cmdSelectUi->historyList->count() > Settings::maxCommandHistory())
            delete cmdSelectUi->historyList->takeItem(Settings::maxCommandHistory());
    }
}

void CmdSelectWidget::commandListSelChanged()
{
    auto sel = cmdSelectUi->commandList->selectedItems();
    if (sel.size() == 0)
        return;
    cmdSelectUi->historyList->clearSelection();
    auto command = sel[0]->text();
    cmdSelectUi->desc->setText("<b>" + command + "</b>: " + allSpecs.commands[command].comment);
    checkConfigureOk();
}

void CmdSelectWidget::historyListSelChanged()
{
    auto sel = cmdSelectUi->historyList->selectedItems();
    if (sel.size() == 0)
        return;
    cmdSelectUi->commandList->clearSelection();
    auto cmdLine = sel[0]->text();
    cmdSelectUi->desc->setText(cmdLine);
    checkConfigureOk();
}

void CmdSelectWidget::historyContextMenu(const QPoint& pos)
{
    QPoint globalPos = cmdSelectUi->historyList->mapToGlobal(pos);
    QMenu menu;
    menu.addAction("&Run", this, SLOT(doRun()));
    menu.addAction("&Edit",  this, SLOT(doConfigure()));
    menu.exec(globalPos);
}

void CmdSelectWidget::commandContextMenu(const QPoint& pos)
{
    QPoint globalPos = cmdSelectUi->commandList->mapToGlobal(pos);
    QMenu menu;
    menu.addAction("&Edit",  this, SLOT(doConfigure()));
    menu.exec(globalPos);
}

void CmdSelectWidget::checkConfigureOk()
{
    emit canRunSignal(cmdSelectUi->historyList->selectedItems().size() != 0);
    emit canEditSignal(
        cmdSelectUi->commandList->selectedItems().size() != 0 ||
        cmdSelectUi->historyList->selectedItems().size() != 0
        );
}

void CmdSelectWidget::doConfigure()
{
    auto sel = cmdSelectUi->commandList->selectedItems();
    if (sel.size() != 0) {
        emit selectedSignal(sel[0]->text(),CommandSpec());
    } else {
        sel = cmdSelectUi->historyList->selectedItems();
        if (sel.size() == 0)
            return;
        CommandSpec spec = qvariant_cast<CommandSpec>(sel[0]->data(Qt::UserRole));
        emit selectedSignal(spec.name,spec);
    }
}

void CmdSelectWidget::doRun()
{
    auto sel = cmdSelectUi->historyList->selectedItems();
    if (sel.size() == 0)
        return;
    CommandSpec specs = qvariant_cast<CommandSpec>(sel[0]->data(Qt::UserRole));
    QStringList args(specs.name);
    for (const auto& as : specs.mandatories)
        args.append(as.initValue);

    for (const auto& as : specs.optionals) {
        if (as.hasInitValue)
            args.append(as.name + "=" + as.initValue);
    }
    emit runSignal(args);
}


