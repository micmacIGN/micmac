#include "mainwindow.h"
#include "global.h"
#include "settings.h"
#include "actionbox.h"
#include "processwidget.h"

#include <iostream>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QMessageBox>
#include <QTextStream>
#include <QProcess>
#include <QStatusTipEvent>
#include <QCloseEvent>


MainWindow::MainWindow(const QString &mmviiPath, const QString &specPath, const QStringList &command, QWidget *parent)
    : QWidget(parent)
    , initCompleted(false)
    , specs(nullptr)
    , cmdSelectWidget(nullptr)
    , cmdConfigureWidget(nullptr)
    , procWidget(nullptr)
{
#if 0
    QFile file("style.qss");
    if(file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
    //        Q_INIT_RESOURCE(icones);

        qApp->setStyleSheet(file.readAll());
        file.close();
    }
    else
        QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("Can't find qss file"));
#endif

    setMinimumSize(700,600);

    if (mmviiPath.size() && specPath.size())
        doError(tr("Specify only one of -s or -m option."));

    if (command.size() > 1)
        doError(tr("Specify only one command on command line."));

    readSpecs(mmviiPath, specPath);
    readCmdFilters("allow",allSpecs.allowed);
    readCmdFilters("deny",allSpecs.denied);
    buildUI(command.size() > 0);

    if (command.size())
        cmdConfigure(command[0], CommandSpec());
    else
        cmdSelect();
    initCompleted = true;
}

MainWindow::~MainWindow()
{
    delete procWidget;
}

bool MainWindow::initOk()
{
    return initCompleted;
}

void MainWindow::doError(const QString &msg1, const QString& msg2)
{
    QMessageBox::critical(nullptr, QGuiApplication::applicationDisplayName(),
                         "<html><head/><body><h4>" + msg1 + "</h4>"
                             + (msg2.size() ? msg2  : "")
                             + "</body></html>"
                         );
    QTextStream(stderr) << msg1 << "\n" << msg2 << Qt::endl;
    exit (1);
}

bool MainWindow::getSpecsFromMMVII(const QString mmviiPath, QByteArray& specsText, bool haltOnError)
{
    QProcess mmviiProc;

    mmviiProc.start(mmviiPath,QStringList() << "GenArgsSpec", QIODevice::ReadOnly);
    if (! mmviiProc.waitForFinished(5000)) {
        switch(mmviiProc.error()) {
        case QProcess::FailedToStart:
            if (! haltOnError)
                return false;
            doError(tr("Can't execute '%1'").arg(mmviiPath));
        case QProcess::Crashed:
            doError(tr("'%1' crashed !").arg(mmviiPath));
        case QProcess::Timedout:
            doError(tr("'%1' didn't respond for 5s").arg(mmviiPath));
        case QProcess::WriteError:
        case QProcess::ReadError:
            doError(tr("Error in communication with '%1").arg(mmviiPath));
        case QProcess::UnknownError:
        default:
            doError(tr("Unknown error while executing '%1'").arg(mmviiPath));
        }
    }
    if (mmviiProc.exitStatus() == QProcess::CrashExit)
        doError(tr("'%1' crashed !").arg(mmviiPath));
    if (mmviiProc.exitCode() != 0)
        doError(tr("'%1' terminated with error code %2").arg(mmviiPath).arg(mmviiProc.exitCode()));

    auto errorMsg = mmviiProc.readAllStandardError();
    if (errorMsg.contains("WARNING:"))
    {
        QTextStream(stderr) << errorMsg << Qt::endl;
    }

    specsText  = mmviiProc.readAllStandardOutput();
    return true;
}


void MainWindow::readSpecs(const QString& mmviiPath, const QString& specPath)
{
    QByteArray specsText;
    if (specPath.size()) {
        QFile in(specPath);
        if (! in.open(QIODevice::ReadOnly))
            doError(tr("Can't read '%1' specifications file").arg(specPath));
        specsText = in.readAll();
        in.close();
    } else {
        if (mmviiPath.size()) {
            getSpecsFromMMVII(mmviiPath, specsText, true);
        } else {
            QString vmmviiDir;
            vmmviiDir = QCoreApplication::applicationDirPath();  // Default location for MMVII and Specs file
            if (!getSpecsFromMMVII(vmmviiDir+"/" + MMVII_EXE_FILE, specsText, false))
                if (!getSpecsFromMMVII(vmmviiDir+"/../bin/" + MMVII_EXE_FILE, specsText, false))
                    getSpecsFromMMVII(vmmviiDir+"/../../bin/" + MMVII_EXE_FILE, specsText, true);
        }
    }

    try {
        allSpecs.fromJson(specsText);
    } catch (ParseJSonException& ) {
        doError("Internal error from MMVII arguments specification :",allSpecs.errorMsg);
    }

    QFileInfo exeFI(allSpecs.mmviiBin);
    if (! exeFI.isFile())
        doError(tr("MMVII executable not found"),allSpecs.mmviiBin);
    if (! exeFI.isExecutable())
        doError(tr("MMVII is not executable"),allSpecs.mmviiBin);
}


void MainWindow::readCmdFilters(const QString& type, StrList& filter)
{
    auto vmmviiFilePath = QCoreApplication::applicationFilePath().remove(QRegExp("\\.exe$",Qt::CaseInsensitive));
    auto allowFile = vmmviiFilePath + "." + type;
    filter.clear();
    QFile file(allowFile);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;
    QTextStream ts(&file);
    while (!ts.atEnd()) {
        QString words;
        ts >> words;
        const QStringList wordList = words.split(',');
        for (const auto& cmd : wordList) {
            if (!cmd.isEmpty()) {
                filter.push_back(cmd);
            }
        }
    }
}



void MainWindow::buildUI(bool hasCommand)
{
    procWidget = new ProcessWidget(nullptr);
    connect(procWidget,&ProcessWidget::runningSignal,this,&MainWindow::commandRunning);

    mainVBoxLayout = new  QVBoxLayout();
    this->setLayout(mainVBoxLayout);
    workingDirWidget = new WorkingDirWidget(hasCommand, this);
    mainVBoxLayout->addWidget(workingDirWidget);

    actionBox = new ActionBox(this);
    mainVBoxLayout->addWidget(actionBox);

    cmdSelectWidget = new CmdSelectWidget(allSpecs,this);
    connect(cmdSelectWidget,&CmdSelectWidget::selectedSignal,this,&MainWindow::cmdConfigure);
    connect(cmdSelectWidget,&CmdSelectWidget::runSignal,this,&MainWindow::runCommand);
    cmdSelectWidget->workingDirChanged();

    cmdConfigureWidget = new CmdConfigureWidget(allSpecs,this);
    connect(cmdConfigureWidget,&CmdConfigureWidget::runSignal,this,&MainWindow::runCommand);

    connect (workingDirWidget,&WorkingDirWidget::workingDirChanged,cmdSelectWidget,&CmdSelectWidget::workingDirChanged);
    connect (workingDirWidget,&WorkingDirWidget::logSignal,procWidget,&ProcessWidget::open);

    connect(actionBox,&ActionBox::quitSignal,this,&MainWindow::close);
    connect(actionBox,&ActionBox::settingsSignal,this,&MainWindow::settings);
    connect(actionBox,&ActionBox::backSignal,this,&MainWindow::cmdSelect);

    connect(actionBox,&ActionBox::runSelectedSignal,cmdSelectWidget,&CmdSelectWidget::doRun);
    connect(actionBox,&ActionBox::editSignal,cmdSelectWidget,&CmdSelectWidget::doConfigure);

    connect(actionBox,&ActionBox::runEditedSignal,cmdConfigureWidget,&CmdConfigureWidget::doRun);
    connect(actionBox,&ActionBox::clearSignal,cmdConfigureWidget,&CmdConfigureWidget::resetValues);

    connect(cmdSelectWidget,&CmdSelectWidget::canRunSignal,actionBox,&ActionBox::runEnabled);
    connect(cmdSelectWidget,&CmdSelectWidget::canEditSignal,actionBox,&ActionBox::editEnabled);
    connect(cmdConfigureWidget,&CmdConfigureWidget::canRunSignal,actionBox,&ActionBox::runEnabled);
}


void MainWindow::cmdSelect()
{
    mainVBoxLayout->removeWidget(cmdConfigureWidget);
    cmdConfigureWidget->hide();

    workingDirWidget->setLocked(false);
    actionBox->setCommandSelection(true);
    mainVBoxLayout->insertWidget(1,cmdSelectWidget);
    cmdSelectWidget->workingDirChanged();
    cmdSelectWidget->show();
    cmdSelectWidget->checkConfigureOk();
    setWindowTitle(QString(APP_NAME) + " - " + tr("Command selection"));
}


void MainWindow::cmdConfigure(const QString& command, CommandSpec spec)
{
    specs = nullptr;
    for (auto& cmd : allSpecs.commands) {
        if (command.toLower() == cmd.name.toLower()) {
            specs = &cmd;
            break;
        }
    }
    if (! specs) {
        QString msg = tr("\"%1\" is not a valid vMMVII command").arg(command);
        QMessageBox::critical(NULL, QObject::tr("Error"), msg);
        std::cerr << "Error: " << qPrintable(msg) << "\n";
        cmdSelect();
        return;
    }

    mainVBoxLayout->removeWidget(cmdConfigureWidget);
    cmdSelectWidget->hide();

    workingDirWidget->setLocked(true);
    actionBox->setCommandSelection(false);
    cmdConfigureWidget->setSpecs(specs, spec);
    mainVBoxLayout->insertWidget(1,cmdConfigureWidget);
    cmdConfigureWidget->show();
    cmdConfigureWidget->checkAllParams();

    setWindowTitle(QString(APP_NAME) + " - " + specs->name + " - " + specs->comment);
}

void MainWindow::settings()
{
    Settings *settings = new Settings(this);
    connect (settings,&Settings::workingDirsClearedSignal,workingDirWidget,&WorkingDirWidget::workingDirCleared);
    connect (settings,&Settings::maxCommandHistoryChanged,cmdSelectWidget,&CmdSelectWidget::workingDirChanged);
    connect (settings,&Settings::outputFontSizeChanged,procWidget,&ProcessWidget::setSettings);
    connect (settings,&Settings::maxOutputLinesChanged,procWidget,&ProcessWidget::setSettings);
    settings->exec();
    delete settings;
}


bool MainWindow::event(QEvent *ev)
{
    if (ev->type() == QEvent::StatusTip) {
        QStatusTipEvent *ste = static_cast<QStatusTipEvent*>(ev);
        actionBox->setStatusMessage(ste->tip());
        return true;
    }
    return QWidget::event(ev);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (procWidget->isRunning()) {
        event->ignore();
        return;
    }
    QWidget::closeEvent(event);
    procWidget->close();
}

void MainWindow::commandRunning(bool running)
{
    cmdSelectWidget->setEnabled(! running);
    cmdConfigureWidget->setEnabled(! running);
    workingDirWidget->setEnabled(! running);
    actionBox->setEnabled(! running);
    if (!running)
        cmdSelectWidget->workingDirChanged();
}


void MainWindow::runCommand(const QStringList& args)
{
    auto dirs = Settings::workingDirs();
    dirs.removeAll(QDir::currentPath());
    dirs.append(QDir::currentPath());
    Settings::setWorkingDirs(dirs);

    procWidget->runCommand(allSpecs.mmviiBin, args);
}
