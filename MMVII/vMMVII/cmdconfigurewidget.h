#ifndef CMDCONFIGUREWIDGET_H
#define CMDCONFIGUREWIDGET_H

#include "commandspec.h"
#include "inputwidget.h"

#include <QWidget>
#include <QPushButton>
#include <QTextEdit>

class CmdConfigureWidget : public QWidget
{
    Q_OBJECT

public:
    explicit CmdConfigureWidget(MMVIISpecs& allSpecs, QWidget *parent = nullptr);
    void setSpecs(CommandSpec *specs, CommandSpec initSpec);
    void checkAllParams();

public slots:
    void doRun();
    void resetValues();

signals:
    void runSignal(const QStringList& args);
    void canRunSignal(bool);

private:
    enum Level {Mandatory, Normal, Global, Tuning};

    MMVIISpecs& allSpecs;
    CommandSpec *specs;
    QString cmdLine;

    QTextEdit *teCommand;
    QTabWidget *tabWidget;

    QWidget *wMandatory;
    QWidget *wOptional;
    QWidget *wTuning;
    QWidget *wGlobal;

    QWidget *createPage(QVector<ArgSpec> &argSpecs, const QString& level);
    InputWidget *createInput(QWidget *widget, QGridLayout *layout, ArgSpec& as);

    QString quoteString(const QString &s);

    void updateCommand();
    void valueUpdated(const ArgSpec& as);

};

#endif // CMDCONFIGUREWIDGET_H
