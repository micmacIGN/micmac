#ifndef SETTINGS_H
#define SETTINGS_H

#include "ui_Settings.h"

#include <QDialog>
#include <QSettings>

//! Dialog to setup display settings
class cSettingsDlg : public QDialog, public Ui::settingsDialog
{
    Q_OBJECT

public:

    //! Default constructor
    cSettingsDlg(QWidget* parent);

signals:
    void hasChanged();

protected slots:

    void on_actionAccept_triggered();
    void on_actionCancel_triggered();

    void on_actionApply_triggered();
    void on_actionReset_triggered();

protected:

    //! Refreshes dialog to reflect new settings values
    void refresh();

    //! Old settings (for restore)
    QSettings oldSettings;
};


#endif
