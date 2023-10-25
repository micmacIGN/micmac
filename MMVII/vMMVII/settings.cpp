#include "settings.h"
#include "ui_settings.h"
#include <QSettings>

Settings::Settings(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Settings)
{
    ui->setupUi(this);
    ui->clearWorkingDirsPB->setEnabled(workingDirs().size()>0);
    ui->maxCommandHistorySB->setValue(maxCommandHistory());
    ui->mmviiWindowCB->setChecked(mmviiWindows());
    ui->maxOutputLineSB->setValue(maxOutputLines());
    ui->outputFontSizeSB->setValue(outputFontSize());
    connect (ui->clearWorkingDirsPB,&QPushButton::clicked,this,&Settings::clearWorkingDirs);
    connect (this,&Settings::accepted,this,&Settings::saveSettings);
}

Settings::~Settings()
{
    delete ui;
}

void Settings::accept()
{
    QWidget *w=QApplication::focusWidget();
    if (dynamic_cast<QSpinBox*>(w)) {
        w->clearFocus();
        return;
    }
    QDialog::accept();
}

void Settings::saveSettings()
{
    if (ui->maxCommandHistorySB->value() != maxCommandHistory()) {
        setMaxCommandHistory(ui->maxCommandHistorySB->value());
        emit maxCommandHistoryChanged();
    }
    setMmviiWindows(ui->mmviiWindowCB->isChecked());
    int lines = ui->maxOutputLineSB->value();
    if (lines == ui->maxOutputLineSB->minimum())
        lines = -1;
    if (ui->maxOutputLineSB->value() != maxOutputLines()) {
        setMaxOutputLines(lines);
        emit maxOutputLinesChanged();
    }
    if (ui->outputFontSizeSB->value() != outputFontSize()) {
        setOutputFontSize(ui->outputFontSizeSB->value());
        emit outputFontSizeChanged();
    }
}

void Settings::clearWorkingDirs()
{
    setWorkingDirs(QStringList());
    ui->clearWorkingDirsPB->setEnabled(false);
    emit workingDirsClearedSignal();
}


void Settings::setWorkingDirs(const QStringList& dirs)
{
    QSettings().setValue("WorkingDir",dirs);
}

QStringList Settings::workingDirs()
{
    return QSettings().value("WorkingDir",QStringList()).toStringList();
}


void Settings::setMaxCommandHistory(int max)
{
    QSettings().setValue("maxCommandHistory",max);
}

int Settings::maxCommandHistory()
{
    return QSettings().value("maxCommandHistory",30).toInt();
}

void Settings::setMmviiWindows(bool on)
{
    QSettings().setValue("mviiWindow",on);
}

bool Settings::mmviiWindows()
{
    return QSettings().value("mviiWindow",true).toBool();
}

void Settings::setMaxOutputLines(int lines)
{
    QSettings().setValue("outputLines",lines);
}

int Settings::maxOutputLines()
{
    return QSettings().value("outputLines",10000).toInt();
}

int Settings::outputFontSize()
{
    QFont font("monospace");
    return QSettings().value("outputFontSize",font.pointSize()).toInt();
}


void Settings::setOutputFontSize(int size)
{
    QSettings().setValue("outputFontSize",size);
}


QSize Settings::outputSize()
{
    return QSettings().value("outputWindowSize").toSize();
}


void Settings::setOutputSize(const QSize &size)
{
    QSettings().setValue("outputWindowSize",size);
}


