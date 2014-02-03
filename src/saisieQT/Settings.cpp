#include "Settings.h"


cSettingsDlg::cSettingsDlg(QWidget *parent) : QDialog(parent), Ui::settingsDialog()
{
    setupUi(this);

    setWindowFlags(Qt::Tool/*Qt::Dialog | Qt::WindowStaysOnTopHint*/);

    //_oldSettings = _settings;

    refresh();

    setUpdatesEnabled(true);
}

void  cSettingsDlg::on_actionAccept_triggered()
{
    accept();
}

void cSettingsDlg::on_actionCancel_triggered()
{
    emit hasChanged();

    reject();
}

void cSettingsDlg::on_actionApply_triggered()
{
    emit hasChanged();
}

void cSettingsDlg::on_actionReset_triggered()
{

}

void cSettingsDlg::refresh()
{

}
