#include "Settings.h"

cSettingsDlg::cSettingsDlg(QWidget *parent, cParameters &params) : QDialog(parent), Ui::settingsDialog()
{
    setupUi(this);

    setWindowFlags(Qt::Tool/*Qt::Dialog | Qt::WindowStaysOnTopHint*/);

    _oldParameters = _parameters = &params;

    refresh();

    setUpdatesEnabled(true);
}

void cSettingsDlg::setParameters(cParameters &params)
{
    if (!_parameters)
        _parameters = new cParameters();

     _parameters = &params;
}

void cSettingsDlg::on_FullscreenCheckBox_clicked()
{
    _parameters->setFullScreen(FullscreenCheckBox->isChecked());
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
    setParameters(*_parameters);

    emit hasChanged();
}

void cSettingsDlg::on_actionReset_triggered()
{
    _parameters->read();

    refresh();
}

void cSettingsDlg::refresh()
{
    FullscreenCheckBox->setChecked(_parameters->getFullScreen());

    NBF_x_spinBox->setValue(_parameters->getNbFen().x());
    NBF_y_spinBox->setValue(_parameters->getNbFen().y());

    WindowWidth_spinBox->setValue(_parameters->getSzFen().width());
    WindowHeight_spinBox->setValue(_parameters->getSzFen().height());
}

cParameters& cParameters::operator =(const cParameters &params)
{
    _mode           = params._mode;
    _nbFen          = params._nbFen;
    _openFullScreen = params._openFullScreen;
    _ptName         = params._ptName;

    return *this;
}

void cParameters::read()
{
     QSettings settings(QApplication::organizationName(), QApplication::applicationName());

     settings.beginGroup("MainWindow");
     setSzFen(settings.value("size", QSize(800, 600)).toSize());
     setNbFen(settings.value("NbFen", QPoint(1, 1)).toPoint());
     setFullScreen(settings.value("openInFullScreen", false).toBool());
     setPosition(settings.value("pos", QPoint(200, 200)).toPoint());
     settings.endGroup();

     settings.beginGroup("Misc");
     setDefPtName(settings.value("defPtName", "").toString());
     setZoomWindowValue(settings.value("zoom", 3.f).toFloat());
     settings.endGroup();
}

void cParameters::write()
{
     QSettings settings(QApplication::organizationName(), QApplication::applicationName());

     settings.beginGroup("MainWindow");
     settings.setValue("size", getSzFen());
     settings.setValue("pos", getPosition());
     settings.setValue("NbFen", getNbFen());
     settings.setValue("openInFullScreen", getFullScreen());
     settings.endGroup();

     settings.beginGroup("Misc");
     settings.setValue("zoom", getZoomWindowValue());
     settings.setValue("defPtName", getDefPtName());
     settings.endGroup();
}
