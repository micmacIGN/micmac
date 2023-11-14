#ifndef CMDSELECT_WIDGET_H
#define CMDSELECT_WIDGET_H

#include <QWidget>
#include "commandspec.h"
#include "ui_cmdSelect.h"


class CmdSelectWidget : public QWidget
{
    Q_OBJECT

public:
    CmdSelectWidget(const MMVIISpecs& allSpecs, QWidget *parent = nullptr);
    ~CmdSelectWidget();
    void checkConfigureOk();

public slots:
    void workingDirChanged();
    void doConfigure();
    void doRun();
    void historyContextMenu(const QPoint& pos);
    void commandContextMenu(const QPoint& pos);

signals:
    void selectedSignal(const QString& cmd, CommandSpec spec);
    void runSignal(const QStringList& args);
    void canRunSignal(bool on);
    void canEditSignal(bool on);

private:
    void commandListSelChanged();
    void historyListSelChanged();

    CommandSpec parseArgs(const StrList &args, QString& newLine);

    Ui::CmdSelectUI *cmdSelectUi;
    const MMVIISpecs& allSpecs;
};

#endif // CMDSELECT_WIDGET_H
