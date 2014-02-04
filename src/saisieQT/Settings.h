#ifndef SETTINGS_H
#define SETTINGS_H

#include "ui_Settings.h"

#include <QDialog>
#include <QSettings>

class cParameters
{
public:
    cParameters(){}

    void setNbFen(QPoint aNbFen)        { _nbFen = aNbFen;  }
    void setSzFen(QSize aSzFen)         { _szFen = aSzFen;  }
    void setFullScreen(bool fullscreen) { _openFullScreen = fullscreen; }
    void setZoomWindowValue(float aZoom){ _zoomWindow = aZoom; }
    void setDefPtName(QString name)     { _ptName = name;   }
    void setPosition(QPoint pos)        { _position = pos;  }

    QPoint  getNbFen()      { return _nbFen;    }
    QSize   getSzFen()      { return _szFen;    }
    bool    getFullScreen() { return _openFullScreen;  }
    float   getZoomWindowValue() { return _zoomWindow; }
    QString getDefPtName()  { return _ptName;   }
    QPoint  getPosition()   { return _position; }

    //! Copy operator
    cParameters& operator =(const cParameters& params);

    void    read();
    void    write();

private:

    //appli mode (MASK2D, MASK3D, SAISIEPT_INIT, SAISIEPT_PREDIC)
    int         _mode;

    //main window parameters
    QPoint      _position;
    QPoint      _nbFen;
    QSize       _szFen;
    bool        _openFullScreen;

    //other parameters
    float       _zoomWindow;
    QString     _ptName;

};

//! Dialog to setup display settings
class cSettingsDlg : public QDialog, public Ui::settingsDialog
{
    Q_OBJECT

public:

    //! Default constructor
    cSettingsDlg(QWidget* parent, cParameters &params);

    void setParameters(cParameters &params);

signals:
    void hasChanged();

protected slots:

    void on_FullscreenCheckBox_clicked();

    void on_actionAccept_triggered();
    void on_actionCancel_triggered();

    void on_actionApply_triggered();
    void on_actionReset_triggered();

protected:

    //! Refreshes dialog to reflect new settings values
    void refresh();

    //! settings
    cParameters *_parameters;

    //! settings
    cParameters *_oldParameters;
};


#endif
