#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "commandspec.h"
#include "workingdirwidget.h"
#include "cmdselectwidget.h"
#include "cmdconfigurewidget.h"
#include "actionbox.h"
#include "processwidget.h"

#include <QWidget>
#include <QPushButton>
#include <QTextEdit>
#include <QListWidgetItem>


class MainWindow : public QWidget
{
    Q_OBJECT

public:
    MainWindow(const QString& mmviiPath, const QString& specPath, const QStringList& command, QWidget *parent = nullptr);
    ~MainWindow();

    bool initOk();

private:
    [[noreturn]] void doError(const QString& msg1, const QString &msg2="");
    
    void readCmdFilters(const QString &type, StrList &filter);
    void readSpecs(const QString &mmviiPath, const QString &specPath);
    void buildUI(bool hasCommand);
    bool getSpecsFromMMVII(const QString mmviiPath, QByteArray& specsText, bool haltOnError);

    void cmdConfigure(const QString& command, CommandSpec spec);
    void cmdSelect();

    void runCommand(const QStringList &args);
    void commandRunning(bool running);

    void settings();

    bool event(QEvent *ev) override;
    void closeEvent(QCloseEvent *event) override;

    bool initCompleted;

    MMVIISpecs allSpecs;
    CommandSpec *specs;

    WorkingDirWidget *workingDirWidget;
    ActionBox *actionBox;
    CmdSelectWidget *cmdSelectWidget;
    CmdConfigureWidget *cmdConfigureWidget;
    ProcessWidget *procWidget;
    
    QVBoxLayout *mainVBoxLayout;
};
#endif // MAINWINDOW_H
